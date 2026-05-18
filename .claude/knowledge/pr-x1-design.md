# PR-X1 — SIMD-Staged Inner-Loop Primitives: MultiLaneColumn, Fingerprint::as_u8x64, array_window, simd::* re-export sweep

> READ BY: all ndarray agents that touch the cognitive shader stack
> (savant-architect, l3-strategist, cascade-architect,
> cognitive-architect, arm-neon-specialist, sentinel-qa, product-engineer,
> truth-architect, vector-synthesis, splat3d-architect).
>
> **Design doc v1** — carved out from the W3-W6 P2 savant review
> (A4 finding: `aos_to_soa` hardwired to `f32`; the SIMD-staged inner-loop
> primitives blocked on missing `MultiLaneColumn` / `array_window` /
> `Fingerprint::as_u8x64` in `crate::simd::*`).
>
> Parallel docs:
> - `.claude/knowledge/pr-x2-design.md` — `#[soa(pad_to_lanes=N)]` + `aos_to_soa<T, U, N>` generalization
> - `.claude/knowledge/pr-x3-cognitive-grid-design.md` — BlockedGrid (PR-X3 builds on X1)
> - `.claude/knowledge/cognitive-shader-foundation.md` — ndarray's role in the 7-layer stack
> - `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a layering rule
> - `.claude/knowledge/cognitive-distance-typing.md` — no umbrella distance rule

## Context for a fresh session

If you arrive here without conversational context (token reset, new session, handover), here is the minimum you need to know:

1. **W3-W6 shipped** (PR #156, merged 2026-05-18). Added `SoaVec<T, N>`, `soa_struct!`, `aos_to_soa<T, N, F>` (f32-only output), `soa_to_aos`, `bulk_apply`, `bulk_scan` to `src/hpc/{soa,bulk}.rs`. Scalar only. No SIMD.
2. **PR #157 shipped** (P2 savant follow-up). Added f32-only-scope docs, `hpc::soa`-vs-`simd_ops` layering rationale, and ungated the `bulk_apply` x `aos_to_soa` integration test.
3. **PR-X1 (this doc)**: Fills the SIMD-staged inner-loop gap flagged by `cognitive-shader-foundation.md` §"Current Gaps" items 1–3 and the W3-W6 P2 savant A4 finding. Adds four primitives to `crate::simd::*`: `MultiLaneColumn`, `Fingerprint::as_u8x64`, `array_window`, and completes the `simd::*` re-export sweep.
4. **PR-X2 (sibling doc at `.claude/knowledge/pr-x2-design.md`)**: Generalizes `aos_to_soa`/`soa_to_aos` to `<T, U, N>` and adds `#[soa(pad_to_lanes=N)]` to `soa_struct!`. Logically follows X1 — the generalized helpers use `U64x8`, `U8x64`, etc., which must be fully re-exported first.
5. **PR-X3 (open, in sprint)**: `BlockedGrid<T, BR, BC>` hierarchical block grid. Uses `crate::simd::U64x8` etc. in consumer closure bodies; requires the re-export sweep to be complete.
6. **`Fingerprint<N>` already exists** in `src/hpc/fingerprint.rs`. It has `as_bytes()` (zero-copy `&[u8]`), `chunks_u8x64()` (iterator over 64-byte chunks), and `chunks_u64x8()` (iterator over 8-u64 chunks). PR-X1 adds `as_u8x64()` — a typed `&[u8; 64]` view for `Fingerprint<8>` (8 × 8 bytes = 64 bytes, the AVX-512 register-width unit). See Q1 for the N=8 vs N=1 naming issue.
7. **`MultiLaneColumn` and `array_window` do not exist yet**. They are listed in `cognitive-shader-foundation.md` §"Current Gaps" items 1 and part of item 3, and in `simd.rs` §"Types that MUST be in ndarray::simd::*".

## Why this exists

The cognitive shader stack (Layer 1) needs SIMD-staged inner loops that walk N lanes at a time. W3-W6 established the SoA layout shape but left three critical SIMD-staged primitives unimplemented:

1. **`MultiLaneColumn`**: Layer 1 BindSpace column consumers need to project the same byte buffer as different SIMD lane widths per operation — U8x64 for palette, F32x16 for f32 cognition, F64x8 for double-precision ops — without copying or re-allocating. Without this type, each consumer writes its own unsafe reinterpret cast, violating the W1a consumer contract (158 raw-intrinsic violations were catalogued in `E-SIMD-SWEEP-1`).

2. **`Fingerprint::as_u8x64`**: `Fingerprint<8>` is exactly 64 bytes (8 u64 words). AVX-512 U8x64 is also 64 bytes. An aligned `&[u8; 64]` view over a `Fingerprint<8>` enables zero-copy AVX-512 register loads for byte-level ops (palette scan, popcount, nibble unpack). Without this, consumers write raw `std::mem::transmute` — the exact W1a violation pattern.

3. **`array_window`**: consumers need const-size windows over `&[T]` without heap allocation. Currently every consumer either calls `as_chunks::<N>` directly (fine, but undiscoverable) or rolls a raw slice index without the compile-time bounds check. The `array_window` helper centralizes the safety assertion and makes the N-wide-window pattern discoverable from `crate::simd::*`.

4. **`simd::*` re-export sweep**: `cognitive-shader-foundation.md` §"Current Gaps" item 3 lists `MultiLaneColumn` and `array_window` as missing from `crate::simd::*`. The other types (`Fingerprint`, `VectorWidth`, `VectorConfig`, `CollapseGate`) are already present (confirmed at `simd.rs:1715–1719`). This PR adds the two missing entries and closes the gap.

## The API

All four items surface via `crate::simd::*` per the W1a consumer contract. Implementation homes:

- `MultiLaneColumn` → `src/hpc/column.rs` (new file), re-exported from `src/simd.rs`
- `Fingerprint::as_u8x64` → `src/hpc/fingerprint.rs` (extend existing `impl Fingerprint<8>` block)
- `array_window` / `array_window_checked` → `src/hpc/array_window.rs` (new file), re-exported from `src/simd.rs`
- `simd::*` sweep → two `pub use` additions to `src/simd.rs`

### 1. `MultiLaneColumn`

```rust
// src/hpc/column.rs

//! Multi-lane typed column view over a shared byte backing store.
//!
//! [`MultiLaneColumn`] wraps one `Arc<[u8]>` backing buffer and provides
//! zero-copy typed lane views at different SIMD widths. Consumers pick
//! the lane width per operation; the backing store is never copied.
//!
//! This module is **layout-only**. No `#[target_feature]`, no per-arch
//! imports, no raw intrinsics. The SIMD register load happens inside the
//! consumer's loop using `crate::simd::F32x16::from_array` etc.
//!
//! # Layering
//! Lives in `hpc::column`, re-exported from `crate::simd::*` per the
//! W1a consumer contract at `.claude/knowledge/vertical-simd-consumer-contract.md`.
//!
//! # Distance typing
//! This type is layout-only. No distance-aware API. See
//! `.claude/knowledge/cognitive-distance-typing.md`.

extern crate alloc;
use alloc::sync::Arc;

/// Multi-lane (N-wide) typed column view over a shared `Arc<[u8]>` buffer.
///
/// Useful for SIMD-staged inner loops that view the same backing bytes as
/// different SIMD lane widths without copying. The caller allocates the
/// backing buffer once; `MultiLaneColumn` holds an `Arc` reference so the
/// column can be cloned cheaply for multi-consumer access.
///
/// The backing store must be a multiple of 64 bytes (the AVX-512 register
/// width and cache-line size). `new` returns `Err(())` otherwise.
///
/// # Example
///
/// ```
/// use ndarray::simd::MultiLaneColumn;
/// let data: alloc::sync::Arc<[u8]> = vec![0u8; 128].into();
/// let col = MultiLaneColumn::new(data).unwrap();
/// assert_eq!(col.len_bytes(), 128);
/// assert_eq!(col.len_u8x64(), 2);
/// ```
pub struct MultiLaneColumn {
    data: Arc<[u8]>,
}

impl MultiLaneColumn {
    /// Construct a `MultiLaneColumn` from a shared byte buffer.
    ///
    /// Returns `Err(())` if `data.len()` is not a multiple of 64.
    ///
    /// An empty buffer (`data.len() == 0`) is accepted — `is_empty()`
    /// will return `true` and all iterators yield zero windows.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::simd::MultiLaneColumn;
    /// let data: alloc::sync::Arc<[u8]> = vec![1u8; 64].into();
    /// let col = MultiLaneColumn::new(data).expect("64 is a multiple of 64");
    /// assert_eq!(col.len_u8x64(), 1);
    ///
    /// // Rejected: 100 is not a multiple of 64.
    /// let bad: alloc::sync::Arc<[u8]> = vec![0u8; 100].into();
    /// assert!(MultiLaneColumn::new(bad).is_err());
    /// ```
    pub fn new(data: Arc<[u8]>) -> Result<Self, ()> {
        if data.len() % 64 != 0 {
            return Err(());
        }
        Ok(Self { data })
    }

    /// Total byte length of the backing store.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::simd::MultiLaneColumn;
    /// let col = MultiLaneColumn::new(vec![0u8; 192].into()).unwrap();
    /// assert_eq!(col.len_bytes(), 192);
    /// ```
    pub fn len_bytes(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the column has zero bytes.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of 64-byte (U8x64) chunks in this column.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::simd::MultiLaneColumn;
    /// let col = MultiLaneColumn::new(vec![0u8; 256].into()).unwrap();
    /// assert_eq!(col.len_u8x64(), 4);
    /// ```
    pub fn len_u8x64(&self) -> usize {
        self.data.len() / 64
    }

    /// Number of F32x16-shaped (16 × f32 = 64-byte) chunks.
    pub fn len_f32x16(&self) -> usize {
        self.data.len() / 64
    }

    /// Number of F64x8-shaped (8 × f64 = 64-byte) chunks.
    pub fn len_f64x8(&self) -> usize {
        self.data.len() / 64
    }

    /// View the backing store as a raw byte slice.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::simd::MultiLaneColumn;
    /// let col = MultiLaneColumn::new(vec![42u8; 64].into()).unwrap();
    /// assert!(col.as_bytes().iter().all(|&b| b == 42));
    /// ```
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Iterate the column as contiguous `&[u8; 64]` windows (U8x64 shape).
    ///
    /// Each window is exactly 64 bytes — one AVX-512 U8x64 register load.
    /// Zero-copy: each window is a reference into the backing store.
    ///
    /// Feed each window into `U8x64::from_array(*win)` or
    /// `crate::simd::U8x64::from_slice(win)` inside the consumer's loop.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::simd::MultiLaneColumn;
    /// let data: alloc::sync::Arc<[u8]> = (0u8..128).collect::<Vec<_>>().into();
    /// let col = MultiLaneColumn::new(data).unwrap();
    /// let windows: Vec<&[u8; 64]> = col.iter_u8x64().collect();
    /// assert_eq!(windows.len(), 2);
    /// assert_eq!(windows[0][0], 0u8);
    /// assert_eq!(windows[1][0], 64u8);
    /// ```
    pub fn iter_u8x64(&self) -> impl Iterator<Item = &[u8; 64]> {
        // `as_chunks` is stable on Rust 1.77+; this repo requires 1.94.
        self.data.as_chunks::<64>().0.iter()
    }

    /// Iterate the column as contiguous `&[f32; 16]` windows (F32x16 shape).
    ///
    /// Reinterprets the backing bytes as f32 (byte-for-byte cast, no
    /// conversion). The consumer is responsible for ensuring the bytes encode
    /// valid f32 bit patterns for their use case. Palette / bit-packed
    /// consumers should use `iter_u8x64` instead.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::simd::MultiLaneColumn;
    /// let data: alloc::sync::Arc<[u8]> = vec![0u8; 64].into();
    /// let col = MultiLaneColumn::new(data).unwrap();
    /// let wins: Vec<&[f32; 16]> = col.iter_f32x16().collect();
    /// assert_eq!(wins.len(), 1);
    /// assert_eq!(wins[0][0], 0.0f32); // all-zero bytes = 0.0f32
    /// ```
    pub fn iter_f32x16(&self) -> impl Iterator<Item = &[f32; 16]> {
        self.data.as_chunks::<64>().0.iter().map(|c| {
            // SAFETY: `c` is `&[u8; 64]`. `[f32; 16]` has the same size
            // (16 × 4 = 64 bytes). `f32` has no invalid bit patterns for
            // load purposes (NaN is valid f32). Alignment of `Arc<[u8]>`
            // is at least 8 bytes (u64 backing), which satisfies `f32`'s
            // 4-byte alignment requirement. The returned reference lifetime
            // is tied to `&self`, so the backing Arc outlives the reference.
            unsafe { &*(c.as_ptr() as *const [f32; 16]) }
        })
    }

    /// Iterate the column as contiguous `&[f64; 8]` windows (F64x8 shape).
    ///
    /// Same byte-reinterpret semantics as `iter_f32x16`. Consumer ensures
    /// byte layout encodes valid f64 values.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::simd::MultiLaneColumn;
    /// let data: alloc::sync::Arc<[u8]> = vec![0u8; 128].into();
    /// let col = MultiLaneColumn::new(data).unwrap();
    /// let wins: Vec<&[f64; 8]> = col.iter_f64x8().collect();
    /// assert_eq!(wins.len(), 2);
    /// ```
    pub fn iter_f64x8(&self) -> impl Iterator<Item = &[f64; 8]> {
        self.data.as_chunks::<64>().0.iter().map(|c| {
            // SAFETY: `[f64; 8]` = 8 × 8 = 64 bytes. Same justification
            // as `iter_f32x16`. `f64` alignment = 8 bytes; `Arc<[u8]>`
            // allocation is at least 8-byte aligned.
            unsafe { &*(c.as_ptr() as *const [f64; 8]) }
        })
    }
}
```

### 2. `Fingerprint<8>::as_u8x64`

Add to `src/hpc/fingerprint.rs` after the existing `impl<const N: usize> Fingerprint<N>` block:

```rust
/// Specialized impl for `Fingerprint<8>` — the 64-byte (512-bit) cognitive
/// identity hash unit. Exactly one AVX-512 U8x64 register in width.
impl Fingerprint<8> {
    /// Zero-copy view of this fingerprint as a `&[u8; 64]` (U8x64 shape).
    ///
    /// `Fingerprint<8>` = 8 × u64 words = 64 bytes. This is exactly the
    /// width of one AVX-512 U8x64 register. Use this method to pass a
    /// cognitive identity hash into a U8x64 SIMD load without allocation.
    ///
    /// For larger fingerprints (e.g. `Fingerprint<256>`), use the existing
    /// `chunks_u8x64()` iterator which yields 64-byte windows.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::simd::Fingerprint;
    /// let fp: Fingerprint<8> = Fingerprint::zero();
    /// let view: &[u8; 64] = fp.as_u8x64();
    /// assert_eq!(view.len(), 64);
    /// assert!(view.iter().all(|&b| b == 0));
    /// ```
    ///
    /// # Example — round-trip via known word values
    ///
    /// ```
    /// use ndarray::simd::Fingerprint;
    /// let fp: Fingerprint<8> = Fingerprint::from_words([
    ///     0x0102030405060708u64, 0, 0, 0, 0, 0, 0, 0,
    /// ]);
    /// let view = fp.as_u8x64();
    /// // Words are stored little-endian; word[0] low byte = 0x08.
    /// assert_eq!(view[0], 0x08);
    /// assert_eq!(view[7], 0x01);
    /// assert_eq!(view[8], 0x00); // word[1] = 0
    /// ```
    pub fn as_u8x64(&self) -> &[u8; 64] {
        // SAFETY: `Fingerprint<8>` is `{ words: [u64; 8] }` = 64 bytes.
        // `[u8; 64]` has the same size and alignment 1 ≤ alignment of u64.
        // We cast a pointer to the first word to `*const [u8; 64]`.
        // The returned reference has lifetime tied to `&self`.
        unsafe { &*(self.words.as_ptr() as *const [u8; 64]) }
    }
}
```

### 3. `array_window` / `array_window_checked`

```rust
// src/hpc/array_window.rs

//! Fixed-size window helper for SIMD-staged slice iteration.
//!
//! [`array_window`] returns a const-length `&[T; N]` reference into a
//! `&[T]` at a given offset. The window size is a compile-time constant,
//! enabling zero-cost SIMD type construction (e.g. `F32x16::from_array`).
//!
//! This module is **scalar-shaped**. No `#[target_feature]`, no per-arch
//! imports, no raw intrinsics. The `&[T; N]` feeds directly into SIMD
//! wrapper constructors from `crate::simd::*`.
//!
//! # Distance typing
//! Geometry-free. No distance-aware API. See
//! `.claude/knowledge/cognitive-distance-typing.md`.

/// Return a fixed-size `&[T; N]` window into `slice` at position `offset`.
///
/// Zero heap allocation; the window is a reference into the existing slice.
/// The window size `N` is a compile-time constant, so this feeds directly
/// into SIMD wrapper constructors that take `[T; N]` arrays.
///
/// # Panics
///
/// Panics if `offset + N > slice.len()`. The panic message includes
/// the offset, N, and slice length for easy diagnosis.
///
/// # Compile-time assertion
///
/// `const { assert!(N > 0) }` — a zero-width window would cause type
/// mismatches in SIMD constructors and is caught at compile time.
///
/// # Example
///
/// ```
/// use ndarray::simd::array_window;
/// let data: &[u32] = &[10, 20, 30, 40, 50, 60, 70, 80];
/// let w: &[u32; 4] = array_window::<u32, 4>(data, 2);
/// assert_eq!(w, &[30, 40, 50, 60]);
/// ```
///
/// # SIMD compose pattern
///
/// ```
/// use ndarray::simd::array_window;
/// // Walk 16-element windows of f32:
/// let floats: Vec<f32> = (0..32).map(|i| i as f32).collect();
/// let mut sum = 0.0f32;
/// for offset in (0..floats.len()).step_by(16) {
///     if offset + 16 > floats.len() { break; }
///     let win: &[f32; 16] = array_window::<f32, 16>(&floats, offset);
///     sum += win.iter().sum::<f32>();
/// }
/// assert_eq!(sum, (0..32).map(|i| i as f32).sum::<f32>());
/// ```
pub fn array_window<T, const N: usize>(slice: &[T], offset: usize) -> &[T; N] {
    const { assert!(N > 0, "array_window: N must be > 0") };
    assert!(
        offset + N <= slice.len(),
        "array_window: offset {} + N {} exceeds slice.len() {}",
        offset,
        N,
        slice.len()
    );
    // SAFETY: we asserted `offset + N <= slice.len()`, so the pointer
    // arithmetic is in-bounds. The returned reference has lifetime 'a
    // tied to `slice: &'a [T]`, so it cannot outlive the input.
    unsafe { &*(slice.as_ptr().add(offset) as *const [T; N]) }
}

/// Non-panicking variant. Returns `None` if `offset + N > slice.len()`.
///
/// # Example
///
/// ```
/// use ndarray::simd::array_window_checked;
/// let data = &[1u8, 2, 3, 4, 5];
/// assert!(array_window_checked::<u8, 3>(data, 2).is_some());
/// assert!(array_window_checked::<u8, 3>(data, 3).is_none()); // 3+3=6 > 5
/// assert!(array_window_checked::<u8, 3>(data, 0).is_some());
/// ```
pub fn array_window_checked<T, const N: usize>(slice: &[T], offset: usize) -> Option<&[T; N]> {
    const { assert!(N > 0, "array_window_checked: N must be > 0") };
    if offset.checked_add(N)? > slice.len() {
        return None;
    }
    // SAFETY: bounds checked above.
    Some(unsafe { &*(slice.as_ptr().add(offset) as *const [T; N]) })
}
```

### 4. `simd::*` re-export sweep

Additions to the cognitive-shader re-export block in `src/simd.rs` (after the existing `CollapseGate` / `Fingerprint` / `VectorWidth` block at lines 1714–1719):

```rust
// PR-X1: MultiLaneColumn — multi-lane typed column view (src/hpc/column.rs)
pub use crate::hpc::column::MultiLaneColumn;

// PR-X1: array_window — fixed-size slice window helper (src/hpc/array_window.rs)
pub use crate::hpc::array_window::{array_window, array_window_checked};
```

**Already present — do NOT re-add:**
- `CollapseGate` at `simd.rs:1715`
- `Fingerprint`, `VectorWidth`, `VectorConfig`, `vector_config`, `Fingerprint1K`, `Fingerprint2K`, `Fingerprint64K` at `simd.rs:1717–1719`
- All SIMD vector types (`F32x16`, `F64x8`, `U8x64`, `U64x8`, `I8x32`, `BF16x16`, etc.) at lines 223–290 / 1541–1583

**`src/hpc/mod.rs` additions:**

```rust
// Alphabetical position:
pub mod array_window;
pub mod column;
```

## Layering rule recap

PR-X1 lives at the **user-code layer** (same as `hpc/soa.rs`). The W1a contract (`vertical-simd-consumer-contract.md`) requires:

1. No `#[target_feature(enable = "...")]` in `hpc/column.rs` or `hpc/array_window.rs`
2. No `cfg(target_feature = "...")` gates
3. No `use crate::simd_avx512::*` / `simd_avx2::*` / `simd_neon::*` from those files
4. No raw `_mm*_*` / `vld*_*` intrinsics
5. No `is_x86_feature_detected!()` calls

`MultiLaneColumn::iter_f32x16` and `iter_f64x8` use `unsafe` pointer casts for zero-copy reinterpretation. These are type-layout-only operations (no SIMD registers). Both carry `// SAFETY:` comments.

The actual SIMD register loads happen inside consumer closure bodies via `crate::simd::F32x16::from_array` etc.

## Distance-typing guardrail

PR-X1 is **layout-only**. None of the four primitives bakes in a distance metric. Workers MUST NOT add:
- `fn distance(...)` or `fn similarity(...)` on `MultiLaneColumn`
- Hamming / palette / Base17 / BF16 distance logic in `hpc/column.rs` or `hpc/array_window.rs`
- `DistanceMetric` enum or `Box<dyn Distance>` trait

See `.claude/knowledge/cognitive-distance-typing.md`.

## Tests required

### `src/hpc/column.rs`

- `new` rejects non-multiple-of-64 lengths (e.g. 100)
- `new` accepts 0, 64, 128, 256 bytes
- `len_bytes` / `len_u8x64` / `len_f32x16` / `len_f64x8` return correct values
- `is_empty` true for zero-byte buffer, false for 64-byte buffer
- `as_bytes` returns the full backing slice
- `iter_u8x64` on 256 bytes yields 4 windows; verify windows[0][0] == input[0], windows[3][0] == input[192]
- `iter_f32x16` on 64 bytes of known bit-pattern f32s: write `1.0f32.to_bits()` bytes into buffer, read back via `iter_f32x16`, verify `wins[0][0] == 1.0f32`
- `iter_f64x8` same round-trip coverage
- `MultiLaneColumn: Send + Sync` static assertion
- `clone()` via `Arc::clone` — two columns sharing the same buffer; mutations not visible because `Arc<[u8]>` is immutable

### `src/hpc/fingerprint.rs` additions

- `Fingerprint<8>::as_u8x64()` returns a 64-element slice
- Round-trip: `Fingerprint<8>::from_words([0x0102030405060708u64, 0, 0, 0, 0, 0, 0, 0])`, call `as_u8x64()`, check `view[0] == 0x08` (little-endian)
- `as_u8x64` does not allocate (pointer equality: `view.as_ptr() == fp.words.as_ptr() as *const u8`)
- `as_u8x64` on `zero()` returns all-zero bytes
- `as_u8x64` on `ones()` returns all-`0xFF` bytes

### `src/hpc/array_window.rs`

- `array_window::<u32, 4>(data, 0)` at start
- `array_window::<u32, 4>(data, len-4)` at last valid position
- `array_window` panics when `offset + N > slice.len()`; panic message contains offset, N, and slice length
- `array_window` with N=1 (single-element window)
- Round-trips for T=u8, T=f32, T=u64
- `array_window_checked` returns `Some` at valid offset, `None` at invalid offset
- `array_window_checked` on empty slice returns `None` for any N
- Zero-allocation: returned pointer equals `&slice[offset]` cast

### Doc-tests

Every public fn / method has a working `# Example` doctest (included in the API section above). Module-level doctest for `column.rs` demonstrates the canonical compose pattern with `iter_u8x64`.

## Out of scope

1. **Aligned allocation** (`new_aligned` constructor for VMOVAPS 64-byte alignment) — follow-up, bench-gated
2. **Mutable lane views** (`iter_u8x64_mut`) — requires `Arc::make_mut` or `MultiLaneColumnMut`; future PR
3. **`Fingerprint<N>::as_u8x64_slice()` for N > 8** — already covered by existing `chunks_u8x64()` iterator
4. **SIMD-accelerated `MultiLaneColumn` deinterleave** — W7, bench-gated
5. **`SoaVec<MultiLaneColumn, N>` pattern** — PR-X2 or later
6. **W1.5 primitives** (signature PDE sweep, randomized projection, Lyndon pack) — wait for `sigker` certification
7. **`is_64byte_aligned() -> bool` probe** — deferred (see Q3)

## Worker decomposition (SEQUENTIAL)

Three Sonnet sprint workers + 1 Opus coordinator.

| # | Phase | Agent role | Scope | Coordinator action |
|---|---|---|---|---|
| 1 | **plan** | (this doc, v1) | design-doc drafter | commit to branch |
| 2 | **review** | plan-review savant | rules on Q1–Q7; READY or NEEDS-FIX | apply P0/P1; commit v2 |
| 3 | **sprint worker A** | `src/hpc/column.rs` (new) + `src/hpc/array_window.rs` (new) + `src/hpc/mod.rs` (two `pub mod` lines) | MultiLaneColumn + array_window + array_window_checked. All inline tests. | verify green; cherry-pick |
| 4 | **sprint worker B** | `src/hpc/fingerprint.rs` — add `impl Fingerprint<8> { pub fn as_u8x64(...) }` per savant Q1 ruling | as_u8x64 method + tests | verify green; cherry-pick |
| 5 | **sprint worker C** | `src/simd.rs` — add two `pub use` lines for MultiLaneColumn + array_window/array_window_checked. Update module-level doc to list all re-exported types. | Single commit. Depends on A + B. | verify green; cherry-pick |
| 6 | **codex P0 audit** | audits combined diff | zero `#[target_feature]`, zero per-arch imports, zero raw intrinsics (except the two approved `unsafe` blocks in column.rs and fingerprint.rs), `// SAFETY:` on every `unsafe` block, all public fns have doctests | apply P0 fixes |
| 7 | **PR open + P2 savant** | P2 ergonomics review | naming, alignment, distance-typing visibility | same-day follow-up if recommended |

## Verification commands

```bash
cargo check -p ndarray --no-default-features --features std
cargo test -p ndarray --lib --no-default-features --features std hpc::column hpc::array_window hpc::fingerprint
cargo test --doc -p ndarray --no-default-features --features std hpc::column hpc::array_window
cargo fmt --all -- --check
cargo clippy -p ndarray --no-default-features --features std -- -D warnings
```

All five must pass green.

## Cross-references

- `.claude/knowledge/cognitive-shader-foundation.md` — §"Current Gaps" items 1–3 that this PR closes
- `.claude/knowledge/pr-x2-design.md` — sibling doc: `#[soa(pad_to_lanes=N)]` + `aos_to_soa<T, U, N>`
- `.claude/knowledge/pr-x3-cognitive-grid-design.md` — BlockedGrid (uses `crate::simd::U64x8` in closure bodies)
- `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a consumer contract
- `.claude/knowledge/cognitive-distance-typing.md` — no-umbrella distance rule
- `.claude/knowledge/w3-w6-p2-savant-review.md` — A4 finding ("aos_to_soa hardwired to f32") that drove the PR-X1/X2 carve-out
- `src/hpc/fingerprint.rs` — existing `Fingerprint<N>` with `as_bytes()`, `chunks_u8x64()`, `chunks_u64x8()`
- `src/hpc/soa.rs` — W3-W6 SoA foundation; PR-X1 completes the `crate::simd::*` surface it relies on
- `src/simd.rs` — the re-export hub; PR-X1 adds `MultiLaneColumn` + `array_window` entries

## Open questions (for the plan-review savant to rule on)

1. **Q1 — `Fingerprint<N>` sizing for `as_u8x64`**: The source spec references "64-byte cognitive identity hash". `Fingerprint<1>` = 1 × 8 = 8 bytes; `Fingerprint<8>` = 8 × 8 = 64 bytes. Design uses `Fingerprint<8>`. Savant: (a) confirm `Fingerprint<8>` is the canonical 64-byte unit, (b) add `pub type Fingerprint64 = Fingerprint<8>` for discoverability, or (c) add a const-generic restriction impl `where [(); N == 8]:`. Option (b) best for consumer ergonomics.

2. **Q2 — `MultiLaneColumn` with zero-length Arc**: Zero is a valid multiple of 64. Design accepts it (`new` returns `Ok`; `is_empty()` = true; iterators yield nothing). Savant: confirm empty is allowed, or should it be `Err(())`?

3. **Q3 — `is_64byte_aligned() -> bool` probe**: AVX-512 aligned load (`vmovdqa64`) requires 64-byte alignment. `Arc<[u8]>` may not be 64-byte aligned. The design uses unaligned semantics. Savant: add `is_64byte_aligned() -> bool` now (2-line impl) or defer to the `new_aligned` follow-up?

4. **Q4 — `unsafe` centralization in `iter_f32x16` / `iter_f64x8`**: Current design centralizes the `unsafe` pointer cast in `hpc/column.rs`. Alternative: return `&[u8; 64]` from all iterators and let the consumer call `unsafe { transmute }`. Design rationale: centralizing in `column.rs` (with `// SAFETY:`) is more auditable than dispersed consumer-side `unsafe`. Savant: confirm centralization is correct W1a pattern.

5. **Q5 — `array_window` vs `slice::array_windows`**: Rust std has `slice::array_windows` (sliding window yielding all windows). PR-X1 adds `array_window` (singular, one window at offset). The names differ by one character. Savant: add a cross-reference note in the docstring pointing to `std::slice::array_windows`, or rename to `array_window_at` to make the distinction sharper?

6. **Q6 — `hpc::column` module naming**: `column` implies column-store domain. `MultiLaneColumn` is a general SIMD-view primitive with no column-store business logic. Alternative module name: `hpc::lane_view`. Savant: `hpc::column` (matches `cognitive-shader-foundation.md` phrasing) or `hpc::lane_view` (more neutral)?

7. **Q7 — `array_window_checked` overflow safety**: Current impl uses `offset.checked_add(N)?` which returns `None` on overflow. This is correct. Savant: confirm this is sufficient or add an explicit overflow test case to the required tests list.

## Done criteria

PR-X1 is done when:
- `MultiLaneColumn` + `array_window` + `array_window_checked` + `Fingerprint<8>::as_u8x64` all compile and test green
- All four primitives re-exported via `crate::simd::*`
- Codex P0 audit: 0 P0 (zero `#[target_feature]`, zero per-arch imports, zero raw intrinsics except the two approved `unsafe` blocks with `// SAFETY:` comments, all public fns have working doctests)
- Layering rule verified per W1a contract
- Distance-typing guardrail verified (zero distance-aware API surface)
- P2 savant review delivers SHIP verdict
