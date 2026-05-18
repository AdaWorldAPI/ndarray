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
    pub fn field(&self, i: usize) -> &[T] { &self.fields[i] }

    /// Mutably borrow field `i` as a slice. Panics if `i >= N`.
    pub fn field_mut(&mut self, i: usize) -> &mut [T] { &mut self.fields[i] }

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

### W5 + W6 — additions to `src/simd_ops.rs`

These two functions are pure-scalar, generic-over-N-via-const-generics, added to the existing `simd_ops.rs`. They reference `crate::hpc::soa::SoaVec` for the W3 type.

```rust
//! Add at the end of src/simd_ops.rs:

use crate::hpc::soa::SoaVec;

/// Deinterleave an AoS slice into a `SoaVec` by extracting `N` field values
/// per item via the user-supplied `extract` closure.
///
/// Scalar implementation. A future wave may add per-arch SIMD gather
/// (VPGATHERDD on AVX-512, LD3/LD4 on NEON) for stride-known dense layouts;
/// the public API is forward-compatible.
///
/// # Example
/// ```
/// use ndarray::simd_ops::aos_to_soa;
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
/// use ndarray::simd_ops::{aos_to_soa, soa_to_aos};
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

## Module registration

Both new files need to be registered in `src/hpc/mod.rs`. Worker writing each new file is responsible for the corresponding `pub mod` line:

```rust
// In src/hpc/mod.rs (alphabetical position):
pub mod bulk;
pub mod soa;
```

Plus the macro re-export at crate root in `src/lib.rs` — the macro is already `#[macro_export]` which puts it at the crate root, so `ndarray::soa_struct!` works. No manual re-export needed.

## What workers commit per file

1. Implement the spec above exactly. No deviation in API.
2. Add inline tests covering the cases listed under §"Tests" for the file.
3. Add the `pub mod` registration in `src/hpc/mod.rs` (W3 worker for `soa`, W4 worker for `bulk`; W5+W6 worker doesn't add a new module).
4. Run from worktree root:
   - `cargo check -p ndarray --no-default-features --features std`
   - `cargo test -p ndarray --lib --no-default-features --features std hpc::soa hpc::bulk simd_ops`
   - `cargo test --doc -p ndarray --no-default-features --features std hpc::soa hpc::bulk simd_ops::aos_to_soa simd_ops::soa_to_aos`
   - All green before commit.
5. Commit message: `feat(hpc/{soa|bulk}): add SoA container + macro (W3)` / `feat(simd_ops): add aos_to_soa + soa_to_aos helpers (W5+W6)` / `feat(hpc/bulk): add bulk_apply + bulk_scan (W4)`.

The post-sprint codex audit will verify:
- Zero `#[target_feature]` attributes added in W3/W4/W5/W6 files
- Zero `use crate::simd_avx512::*` / `simd_avx2::*` / `simd_neon::*` imports
- Zero `cfg(target_feature = …)` gates
- Zero raw `_mm*_*` / `vld*_*` / `_pdep_*` intrinsics
- All public fns have working `///` doc-examples
- Tests cover both `Some` and `None`-equivalent boundary cases per fn

## W7 explicit deferral

W7 (cognitive bulk ops on `&[Plane]` / `&[VSA]`) is **NOT** in this wave. Reason: actual hot paths in the cognitive layer aren't known without bench harness data, and the SIMD wins require per-arch gather/scatter primitives that are separate design work. W7 revisits after a bench harness ships and a measured hot path identifies the first cognitive bulk op worth accelerating.
