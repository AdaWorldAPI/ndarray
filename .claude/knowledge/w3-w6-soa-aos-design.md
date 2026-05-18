# KNOWLEDGE: W3–W6 — SoA/AoS Handoff Helpers (Shape Settlement, No SIMD Yet)

## READ BY
- Worker agents executing W3 (`src/hpc/soa.rs`), W5+W6 (additions to `src/simd_ops.rs`), W4 (`src/hpc/bulk.rs`) — each agent reads this BEFORE writing code
- The plan-review savant agent (before workers spawn)
- The code-review codex agent (after workers commit)
- Downstream consumer sessions that need SoA containers or AoS↔SoA conversion

## P0 TRIGGERS
- About to add a per-arch SIMD intrinsic (`#[target_feature]`, raw `_mm*_*` call, `vld3q_*` / `vst3q_*`, etc.) to any W3–W6 file → STOP, that's not in scope, see §"Why no SIMD yet"
- About to import directly from `simd_avx512.rs`, `simd_avx2.rs`, `simd_neon.rs` from user code → STOP, violates the layering rule, see §"Layering rule"
- About to use `&[T]` / `&mut [T]` as the public type for a kernel-layer fn → STOP, kernels use `ArrayView` per W2 (PR #154); only the SIMD primitive layer takes slices

## What this wave is and is not

**Is:** Establish the SoA container shape, AoS↔SoA conversion ergonomics, and a chunked `bulk_apply` wrapper across the codebase. Pure-scalar implementations. Pure user-facing API. No SIMD.

**Is not:** SIMD-accelerated deinterleave (VPGATHERDD on AVX-512, LD3/LD4 on NEON), gather/scatter primitives, perf-claim work. Those wait for bench data on actual hot paths.

**Why no SIMD yet:** Per the §"Recommendation" of the W3-W7 outlook plan, designing SIMD primitives without measured hot-path data produces speculative APIs. The scalar helpers in this wave are useful in their own right (settles SoA shape so cognitive bulk ops, splat3d, and quantized batches share one ergonomic). A future wave (post-bench-harness) can swap the scalar bodies for per-arch SIMD without changing the public API.

## Layering rule (re-stated — violation is P0)

```
                ┌─────────────────────────────────────┐
                │ user code (hpc/*, splat3d/, downstream crates)
                └──────────────┬──────────────────────┘
                               │ only these imports allowed
                               ▼
                ┌─────────────────────────────────────┐
                │ crate::simd, crate::simd_ops        │  ← dispatch layer
                │ (src/simd.rs, src/simd_ops.rs)      │     LazyLock-frozen function-pointer tables
                └──────────────┬──────────────────────┘     in simd_dispatch.rs
                               │ internal
                               ▼
                ┌─────────────────────────────────────┐
                │ simd_avx512.rs, simd_avx2.rs,        │  ← per-tier impls
                │ simd_neon.rs, simd_wasm.rs           │     these CARRY #[target_feature]
                └─────────────────────────────────────┘
```

User code (this wave's W3, W4, W7 files) MUST NOT:
1. Add `#[target_feature(enable = "…")]` attributes
2. Add `#[cfg(target_feature = "…")]` gates
3. `use crate::simd_avx512::*` / `use crate::simd_avx2::*` / etc.
4. Call `is_x86_feature_detected!()` on hot paths
5. Use raw `_mm*_*` / `vld*_*` / `_pdep_u64` intrinsics

W5/W6 additions to `src/simd_ops.rs` are at the **dispatch layer**. In this wave they are scalar-only; future SIMD impls would live in `simd_avx512.rs` / `simd_avx2.rs` and be dispatched via the LazyLock table. Even now, do NOT add `#[target_feature]` to the simd_ops.rs entries — they stay scalar.

## Exact API contracts (workers do not deviate)

### W3 — `src/hpc/soa.rs` (NEW FILE)

Two complementary primitives: a **macro** for named-field SoA structs, and a **generic struct** for ad-hoc cases.

```rust
//! SoA (Struct of Arrays) containers.
//!
//! Two complementary primitives:
//! - [`soa_struct!`] macro — generates a named-field SoA struct from a struct-like
//!   declaration. Use when the field names matter to callers (e.g. `means_x`,
//!   `means_y`, `means_z` for a Gaussian batch).
//! - [`SoaVec`] generic — `[Vec<T>; N]` wrapper. Use when fields are positional /
//!   anonymous and you want a single type to talk about an N-field SoA batch.
//!
//! Both are SIMD-friendly storage shapes: each field is a contiguous `Vec<T>`,
//! so per-field SIMD loops just iterate one `Vec`. AoS↔SoA conversion helpers
//! live in [`crate::simd_ops`].
//!
//! This module is intentionally scalar — no `#[target_feature]`, no per-arch
//! dispatch. A future wave will add SIMD-accelerated chunk iteration once
//! bench data justifies the per-arch implementations. The public API on this
//! module is forward-compatible with that future swap.

use core::array;

/// SoA container generic over field type `T` and field count `N`.
///
/// Internally: `[Vec<T>; N]`. All N fields are guaranteed to have the same
/// length (enforced by `push` and asserted in `len()`).
///
/// # Example
/// ```
/// use ndarray::hpc::soa::SoaVec;
/// let mut soa: SoaVec<f32, 3> = SoaVec::new();
/// soa.push([1.0, 2.0, 3.0]);
/// soa.push([4.0, 5.0, 6.0]);
/// assert_eq!(soa.len(), 2);
/// assert_eq!(soa.field(0), &[1.0, 4.0]);
/// assert_eq!(soa.field(1), &[2.0, 5.0]);
/// assert_eq!(soa.field(2), &[3.0, 6.0]);
/// ```
pub struct SoaVec<T, const N: usize> {
    fields: [Vec<T>; N],
}

impl<T, const N: usize> SoaVec<T, N> {
    /// Construct an empty SoaVec.
    pub fn new() -> Self {
        Self { fields: array::from_fn(|_| Vec::new()) }
    }

    /// Construct an empty SoaVec with each field pre-allocated to `cap`.
    pub fn with_capacity(cap: usize) -> Self {
        Self { fields: array::from_fn(|_| Vec::with_capacity(cap)) }
    }

    /// Append a row (one value per field) to the SoaVec.
    pub fn push(&mut self, row: [T; N]) {
        for (slot, val) in self.fields.iter_mut().zip(row) {
            slot.push(val);
        }
    }

    /// Number of rows. All fields are guaranteed to have this length.
    ///
    /// # Panics
    /// In debug builds, panics if fields disagree on length (a bug in custom
    /// `unsafe` mutation paths). In release, returns the length of field 0.
    pub fn len(&self) -> usize {
        let n = self.fields[0].len();
        debug_assert!(
            self.fields.iter().all(|f| f.len() == n),
            "SoaVec field-length invariant violated"
        );
        n
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    /// Borrow field `i` as a slice. Panics if `i >= N`.
    ///
    /// For known-at-compile-time indices, prefer [`field_n`](Self::field_n)
    /// to elide the bounds check.
    pub fn field(&self, i: usize) -> &[T] { &self.fields[i] }

    /// Compile-time-checked field accessor. Use when the index is a literal.
    /// `field_n::<2>()` is free of runtime bounds checking.
    pub fn field_n<const I: usize>(&self) -> &[T] {
        const { assert!(I < N, "SoaVec::field_n: I out of bounds for N"); }
        &self.fields[I]
    }

    /// Mutably borrow field `i` as a slice. Panics if `i >= N`.
    pub fn field_mut(&mut self, i: usize) -> &mut [T] { &mut self.fields[i] }

    /// Compile-time-checked mutable field accessor.
    pub fn field_n_mut<const I: usize>(&mut self) -> &mut [T] {
        const { assert!(I < N, "SoaVec::field_n_mut: I out of bounds for N"); }
        &mut self.fields[I]
    }

    /// Borrow all fields at once as an array of slices.
    pub fn all_fields(&self) -> [&[T]; N] {
        array::from_fn(|i| self.fields[i].as_slice())
    }

    /// Iterate over chunks of `chunk_len` rows, yielding `[&[T]; N]` per chunk.
    /// The last chunk may be shorter than `chunk_len`. Fast: zero-copy slice borrow.
    pub fn chunks(&self, chunk_len: usize) -> SoaChunks<'_, T, N> {
        SoaChunks { soa: self, chunk_len, cursor: 0 }
    }
}

impl<T, const N: usize> Default for SoaVec<T, N> {
    fn default() -> Self { Self::new() }
}

/// Iterator yielded by [`SoaVec::chunks`].
pub struct SoaChunks<'a, T, const N: usize> {
    soa: &'a SoaVec<T, N>,
    chunk_len: usize,
    cursor: usize,
}

impl<'a, T, const N: usize> Iterator for SoaChunks<'a, T, N> {
    type Item = [&'a [T]; N];

    fn next(&mut self) -> Option<Self::Item> {
        let len = self.soa.len();
        if self.cursor >= len { return None; }
        let end = (self.cursor + self.chunk_len).min(len);
        let chunk: [&'a [T]; N] = array::from_fn(|i| &self.soa.fields[i][self.cursor..end]);
        self.cursor = end;
        Some(chunk)
    }
}

/// Generate a named-field SoA struct from a struct-like declaration.
///
/// # Example
/// ```
/// use ndarray::soa_struct;
///
/// soa_struct! {
///     pub struct GaussianBatch {
///         pub means_x: f32,
///         pub means_y: f32,
///         pub means_z: f32,
///     }
/// }
///
/// // Generates:
/// // pub struct GaussianBatch {
/// //     pub means_x: Vec<f32>,
/// //     pub means_y: Vec<f32>,
/// //     pub means_z: Vec<f32>,
/// // }
/// // impl GaussianBatch { pub fn new() -> Self {...}, ... }
///
/// let mut b = GaussianBatch::new();
/// b.push(1.0, 2.0, 3.0);
/// b.push(4.0, 5.0, 6.0);
/// assert_eq!(b.len(), 2);
/// assert_eq!(b.means_x.as_slice(), &[1.0, 4.0]);
/// ```
#[macro_export]
macro_rules! soa_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $($field_vis:vis $field:ident : $ty:ty),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis struct $name {
            $($field_vis $field: ::std::vec::Vec<$ty>),*
        }

        impl $name {
            /// Construct an empty instance.
            pub fn new() -> Self {
                Self { $($field: ::std::vec::Vec::new()),* }
            }

            /// Construct with each field pre-allocated to `cap`.
            pub fn with_capacity(cap: usize) -> Self {
                Self { $($field: ::std::vec::Vec::with_capacity(cap)),* }
            }

            /// Append one row across all fields.
            pub fn push(&mut self, $($field: $ty),*) {
                $(self.$field.push($field);)*
            }

            /// Length (all fields share this length; debug-asserted).
            pub fn len(&self) -> usize {
                let lens = [$(self.$field.len()),*];
                debug_assert!(
                    lens.iter().all(|&l| l == lens[0]),
                    concat!(stringify!($name), ": field-length invariant violated")
                );
                lens[0]
            }

            pub fn is_empty(&self) -> bool { self.len() == 0 }

            /// Clear all fields. Capacity is retained.
            pub fn clear(&mut self) {
                $(self.$field.clear();)*
            }
        }

        impl ::std::default::Default for $name {
            fn default() -> Self { Self::new() }
        }
    };
}
```

**Tests** (inline `#[cfg(test)] mod tests`):
- `SoaVec::new` / `with_capacity` / `push` / `len` / `is_empty`
- `SoaVec::field` / `field_mut` / `all_fields`
- `SoaVec::chunks` for chunk_len that divides len, chunk_len that doesn't, chunk_len > len
- `soa_struct!` macro: generate a 3-field struct, push, len, clear, default
- `soa_struct!` macro with `pub` / private fields
- `soa_struct!` debug-assert fires on field-length-mismatch (use a custom block to corrupt, then call len, assert panic)

### W5 + W6 — additions to `src/hpc/soa.rs`

**Plan-review correction (P0-1):** these helpers were originally specced into `src/simd_ops.rs`, but that module's charter (`simd_ops.rs:1`) is "Slice-level elementwise ops built on the polyfill SIMD types" — every existing fn dispatches through `F32x16`/`F64x8`. Pure-scalar helpers in that module would violate the W1a consumer contract ("ndarray's SIMD surface is shaped to fit exactly what the Ada stack vertically needs — not a generic library", `vertical-simd-consumer-contract.md:31`). The free-function `aos_to_soa(&[T], extract)` shape is exactly the kind the W1a litmus rejects.

**Decision: helpers live in `src/hpc/soa.rs`, co-located with `SoaVec`.** When a future bench-justified SIMD wave lands, the dispatcher inside the entry grows internal per-tier arms (calling per-arch impls under `simd_*.rs` for the gather/scatter intrinsics). The public API at `ndarray::hpc::soa::aos_to_soa` stays stable forever.

```rust
//! Add at the bottom of src/hpc/soa.rs (same file as SoaVec):

/// Deinterleave an AoS slice into a `SoaVec` by extracting `N` field values
/// per item via the user-supplied `extract` closure.
///
/// Scalar implementation. A future wave may add per-arch SIMD gather
/// (VPGATHERDD on AVX-512, LD3/LD4 on NEON) for stride-known dense layouts;
/// the public API is forward-compatible.
///
/// `T` need not be `Copy`; only the extracted `[f32; N]` row is materialized.
///
/// # Inference
/// If the const-generic `N` fails to infer from the closure return type, annotate:
/// `aos_to_soa::<_, 3, _>(&aos, |it| [it.a, it.b, it.c])`
/// or  `aos_to_soa(&aos, |it| -> [f32; 3] { [it.a, it.b, it.c] })`.
///
/// # Example
/// ```
/// use ndarray::hpc::soa::aos_to_soa;
/// struct Item { a: f32, b: f32, c: f32 }
/// let aos = vec![Item { a: 1.0, b: 2.0, c: 3.0 }, Item { a: 4.0, b: 5.0, c: 6.0 }];
/// let soa = aos_to_soa::<_, 3, _>(&aos, |it| [it.a, it.b, it.c]);
/// assert_eq!(soa.field(0), &[1.0, 4.0]);
/// assert_eq!(soa.field(1), &[2.0, 5.0]);
/// assert_eq!(soa.field(2), &[3.0, 6.0]);
/// ```
pub fn aos_to_soa<T, const N: usize, F>(aos: &[T], extract: F) -> SoaVec<f32, N>
where
    F: Fn(&T) -> [f32; N],
{
    let mut soa = SoaVec::<f32, N>::with_capacity(aos.len());
    for item in aos {
        soa.push(extract(item));
    }
    soa
}

/// Interleave a `SoaVec` into an AoS `Vec<T>` by building each item from
/// the per-field values via the user-supplied `build` closure.
///
/// Scalar implementation. See [`aos_to_soa`] for the forward-compatibility
/// note.
///
/// # Example
/// ```
/// use ndarray::hpc::soa::{aos_to_soa, soa_to_aos};
/// struct Item { a: f32, b: f32, c: f32 }
/// let aos = vec![Item { a: 1.0, b: 2.0, c: 3.0 }, Item { a: 4.0, b: 5.0, c: 6.0 }];
/// let soa = aos_to_soa::<_, 3, _>(&aos, |it| [it.a, it.b, it.c]);
/// let back: Vec<Item> = soa_to_aos(&soa, |[a, b, c]| Item { a, b, c });
/// assert_eq!(back[0].a, 1.0);
/// assert_eq!(back[1].c, 6.0);
/// ```
pub fn soa_to_aos<T, const N: usize, F>(soa: &SoaVec<f32, N>, build: F) -> Vec<T>
where
    F: Fn([f32; N]) -> T,
{
    let n = soa.len();
    let fields = soa.all_fields();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row: [f32; N] = core::array::from_fn(|k| fields[k][i]);
        out.push(build(row));
    }
    out
}
```

**Tests** (inline in `simd_ops.rs` `#[cfg(test)] mod tests`):
- `aos_to_soa` round-trip: build AoS, deinterleave, check per-field values
- `aos_to_soa` with N=2, N=3, N=4 (most common cases)
- `aos_to_soa` empty input → empty SoaVec
- `soa_to_aos` round-trip: aos_to_soa then soa_to_aos, deep-equal original
- `soa_to_aos` with N=2, N=3, N=4

### W4 — `src/hpc/bulk.rs` (NEW FILE)

Thin chunked wrapper. Caller-managed scalar parallelism.

```rust
//! Bulk traversal helpers for AoS slices.
//!
//! [`bulk_apply`] chunks a `&mut [T]` and invokes a closure with each chunk
//! plus its starting index. Useful when you want predictable cache behavior
//! (chunk_size matched to L1 working-set) or when staging chunks to SoA for
//! SIMD processing inside the closure.
//!
//! Scalar — no `#[target_feature]`, no per-arch dispatch. Composes with
//! [`crate::simd_ops::aos_to_soa`] / [`crate::simd_ops::soa_to_aos`] inside
//! the closure body if the caller wants SIMD per chunk.

/// Apply `f` to consecutive chunks of `items`. Each invocation receives the
/// chunk slice and the absolute index of the chunk's first element.
///
/// The last chunk may be shorter than `chunk_size`.
///
/// # Panics
/// Panics if `chunk_size == 0` (would loop forever).
///
/// # Example
/// ```
/// use ndarray::hpc::bulk::bulk_apply;
/// let mut v: Vec<i32> = (0..10).collect();
/// bulk_apply(&mut v, 3, |chunk, start| {
///     for (i, x) in chunk.iter_mut().enumerate() {
///         *x = (start + i) as i32 * 10;
///     }
/// });
/// assert_eq!(v, vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
/// ```
pub fn bulk_apply<T, F>(items: &mut [T], chunk_size: usize, mut f: F)
where
    F: FnMut(&mut [T], usize),
{
    assert!(chunk_size > 0, "bulk_apply: chunk_size must be > 0");
    let mut start = 0;
    for chunk in items.chunks_mut(chunk_size) {
        f(chunk, start);
        start += chunk.len();
    }
}

/// Read-only sibling of [`bulk_apply`]. Same chunking semantics, immutable
/// chunks.
///
/// # Example
/// ```
/// use ndarray::hpc::bulk::bulk_scan;
/// let v: Vec<i32> = (0..10).collect();
/// let mut sum = 0i32;
/// bulk_scan(&v, 4, |chunk, _start| {
///     sum += chunk.iter().sum::<i32>();
/// });
/// assert_eq!(sum, 45);
/// ```
pub fn bulk_scan<T, F>(items: &[T], chunk_size: usize, mut f: F)
where
    F: FnMut(&[T], usize),
{
    assert!(chunk_size > 0, "bulk_scan: chunk_size must be > 0");
    let mut start = 0;
    for chunk in items.chunks(chunk_size) {
        f(chunk, start);
        start += chunk.len();
    }
}
```

**Tests** (inline):
- `bulk_apply` with chunk_size that divides len, doesn't divide, > len
- `bulk_apply` start index is correct across multiple chunks
- `bulk_apply` panics on chunk_size == 0
- `bulk_scan` same coverage
- `bulk_apply` composed with `aos_to_soa` inside the closure (integration smoke test)

## Reserved field names (do not collide)

The `soa_struct!` macro generates inherent methods on the struct. Fields named identically to these methods will cause cryptic compile errors (the macro deliberately does NOT alias around them — choose different field names):

| Reserved name | Reason |
|---|---|
| `new` | macro generates `pub fn new()` |
| `with_capacity` | macro generates `pub fn with_capacity(cap)` |
| `len` | macro generates `pub fn len(&self)` |
| `is_empty` | macro generates `pub fn is_empty(&self)` |
| `clear` | macro generates `pub fn clear(&mut self)` |
| `push` | macro generates `pub fn push(...)` |
| `default` | macro implements `Default` trait |

If you need a `len` field semantically (e.g., a per-row count), name it `count`, `n`, or `row_len`.

## Invariant ownership (macro fields are `pub` by design)

The macro respects user-specified visibility per field (`pub means_x: f32` stays `pub`; private stays private). Fields stay `pub` (or whatever the user wrote) intentionally:

1. The SoA layout's entire ergonomic win is direct `&[T]` access for SIMD-style loops; hiding fields behind getters defeats the purpose.
2. The existing hand-rolled SoA pattern in `splat3d/gaussian.rs` uses `pub` fields — staying consistent.
3. The macro's generated `len()` carries a `debug_assert!` that catches field-length-mismatch during development.

**Caller-owned invariant rule:** if you mutate fields directly (e.g., `batch.means_x.truncate(5)` to shrink one field), you OWN the field-length invariant until you restore it. `len()` will `debug_assert!` in dev builds; release builds will return the length of field 0 and downstream code may misbehave. Push and clear are the safe mutation paths.

## Module registration

Both new files need to be registered in `src/hpc/mod.rs`. Worker writing each new file is responsible for the corresponding `pub mod` line:

```rust
// In src/hpc/mod.rs (alphabetical position):
pub mod bulk;
pub mod soa;
```

The macro is already `#[macro_export]` which puts it at the crate root, so `ndarray::soa_struct!` works. **Do NOT** add a manual `pub use crate::hpc::soa::soa_struct;` re-export in `lib.rs` — `#[macro_export]` already does this, and a manual re-export will fail to compile (macro and item namespaces collide).

## Worker sprint plan (post plan-review)

Two workers in parallel, each isolated worktree:

| Worker | Files | Scope |
|---|---|---|
| **Worker A — SoA combined** | `src/hpc/soa.rs` (new), `src/hpc/mod.rs` (add `pub mod soa;`) | W3 + W5 + W6: `SoaVec<T,N>` + `soa_struct!` macro + `aos_to_soa` + `soa_to_aos`. Single file, single commit. |
| **Worker B — bulk** | `src/hpc/bulk.rs` (new), `src/hpc/mod.rs` (add `pub mod bulk;`) | W4: `bulk_apply` + `bulk_scan`. Single file, single commit. |

Worker A and Worker B can both edit `src/hpc/mod.rs` (one line each, different lines — merge clean). Otherwise non-overlapping.

After both commit, codex audit reviews the combined diff for:
- Zero `#[target_feature]` attributes added
- Zero `use crate::simd_avx512::*` / `simd_avx2::*` / `simd_neon::*` imports
- Zero `cfg(target_feature = …)` gates
- Zero raw `_mm*_*` / `vld*_*` / `_pdep_*` intrinsics
- All public fns have working `///` doc-examples
- Tests cover all spec'd cases

## What workers commit per file

1. Implement the spec above exactly. No deviation in API.
2. Add inline tests covering the cases listed under §"Tests" for the file.
3. Add the `pub mod` registration in `src/hpc/mod.rs`.
4. Run from worktree root:
   - `cargo check -p ndarray --no-default-features --features std`
   - `cargo test -p ndarray --lib --no-default-features --features std hpc::soa hpc::bulk` (worker scopes its `--test` filter)
   - `cargo test --doc -p ndarray --no-default-features --features std hpc::soa hpc::bulk`
   - `cargo fmt --all --check`
   - `cargo clippy --no-default-features --features std -- -D warnings`
   - All green before commit.
5. Commit message format:
   - Worker A: `feat(hpc/soa): add SoaVec + soa_struct! macro + aos_to_soa + soa_to_aos (W3+W5+W6)`
   - Worker B: `feat(hpc/bulk): add bulk_apply + bulk_scan (W4)`

## Out of scope — distance metrics

This sprint is layout helpers only. **Workers MUST NOT** extend `SoaVec`, the macro, `aos_to_soa`/`soa_to_aos`, or `bulk_apply` toward distance computation. Specifically:

- No `fn bulk_distance<T>(...)` umbrella API
- No `enum DistanceMetric { Palette256, Hamming, Base17, … }`
- No `Box<dyn Distance>` trait object
- No generic `fn distance<T>(a: &T, b: &T) -> f32`
- No collapsing palette-256 distance and HDR popcount early-exit into one helper

Distance metrics in this codebase are **typed** — each metric is its own named fn with its own output type, and conversions between metrics are explicit. See `/home/user/ndarray/.claude/knowledge/cognitive-distance-typing.md` for the binding rule and the canonical worst-case roundtrip anti-pattern (palette-256 → Fisher-z → "cosine" → hamming → popcount → palette-256). The W3-W6 helpers stay generic over `T` and never bake in any metric.

## W7 explicit deferral

W7 (cognitive bulk ops on `&[Plane]` / `&[VSA]` / `&[Fingerprint256]` / `&[PaletteIdx]`) is **NOT** in this wave. Two reasons:

1. **No bench data.** Actual hot paths in the cognitive layer aren't measured. The SIMD wins require per-arch gather/scatter primitives whose design should follow bench evidence, not imagination.
2. **Typed distance scope.** W7's bulk primitives must respect the per-metric typing rule (see `cognitive-distance-typing.md`). Each metric — palette-256 distance, HDR popcount early-exit, Base17 L1, BF16 mantissa direct transform — gets its own named bulk fn with its own output type. Designing those entries without a measured hot-path-vs-target-metric pairing risks shipping a generic API that erases the typing.

When W7 revisits, the typed-bulk primitives will be (representative shape, not exhaustive):
```rust
pub fn bulk_hdr_popcount_early_exit(
    query: &Fingerprint256, db: &[Fingerprint256], threshold: u16,
) -> Vec<Option<HammingDistance>>;

pub fn bulk_palette256_distance(
    query: PaletteIdx, db: &[PaletteIdx],
    buckets: &Buckets, offset: EulerGammaOffset,
) -> Vec<PaletteDistance>;

pub fn bulk_palette256_bf16_mantissa_transform(
    palettes: &[PaletteIdx], offset: EulerGammaOffset, mantissa: BF16MantissaCtx,
) -> Vec<PaletteIdx>;
```

These MAY internally use `SoaVec` + `bulk_apply` from W3/W4 for layout staging. They MUST NOT collapse into a `bulk_distance<T>` umbrella.
