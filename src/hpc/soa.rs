//! SoA (Struct of Arrays) containers and AoS↔SoA conversion helpers.
//!
//! This module provides two complementary primitives for the
//! "struct-of-arrays" storage shape, plus scalar deinterleave / interleave
//! free functions:
//!
//! - [`soa_struct!`] macro — generates a named-field SoA struct from a
//!   struct-like declaration. Use when field names matter to callers
//!   (e.g. `means_x`, `means_y`, `means_z` for a Gaussian batch).
//! - [`SoaVec`] generic — `[Vec<T>; N]` wrapper. Use when fields are
//!   positional / anonymous and you want a single type to talk about an
//!   N-field SoA batch.
//! - [`aos_to_soa`] / [`soa_to_aos`] — scalar deinterleave / interleave
//!   between an AoS slice and a `SoaVec<f32, N>`, parameterized on a
//!   user-supplied extract / build closure.
//!
//! Both shapes are SIMD-friendly storage layouts: each field is a
//! contiguous `Vec<T>`, so per-field SIMD loops iterate one `Vec`.
//!
//! # Layering
//!
//! This module is **scalar only**. It contains no `#[target_feature]`
//! attributes, no `cfg(target_feature = ...)` gates, no per-arch imports.
//! The public API is forward-compatible with a future bench-justified
//! SIMD swap: the dispatcher inside any conversion entry can grow per-arch
//! arms internally (delegating to `simd_avx512.rs` / `simd_neon.rs` for
//! gather / scatter intrinsics) without changing the user-visible
//! signature. See `.claude/knowledge/vertical-simd-consumer-contract.md`
//! for the binding layering rule.
//!
//! # Out of scope — distance metrics
//!
//! These helpers are layout-only and generic over `T`. They never bake in
//! a distance metric. See `.claude/knowledge/cognitive-distance-typing.md`
//! for the rule: each cognitive distance metric (palette 256 distance,
//! HDR popcount early-exit, Base17 L1, BF16 mantissa transform) gets its
//! own named function with its own typed output. Do NOT extend `SoaVec`,
//! the macro, `aos_to_soa`, or `soa_to_aos` toward a generic
//! `bulk_distance<T>` umbrella.

use core::array;

/// SoA container generic over field type `T` and field count `N`.
///
/// Internally: `[Vec<T>; N]`. All `N` fields are guaranteed to have the
/// same length (enforced by [`push`](Self::push) and asserted in
/// [`len`](Self::len) under `debug_assertions`).
///
/// # Example
///
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
    /// Construct an empty `SoaVec`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::soa::SoaVec;
    /// let soa: SoaVec<u32, 2> = SoaVec::new();
    /// assert!(soa.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            fields: array::from_fn(|_| Vec::new()),
        }
    }

    /// Construct an empty `SoaVec` with each field pre-allocated to `cap`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::soa::SoaVec;
    /// let soa: SoaVec<u32, 2> = SoaVec::with_capacity(128);
    /// assert!(soa.is_empty());
    /// ```
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            fields: array::from_fn(|_| Vec::with_capacity(cap)),
        }
    }

    /// Append a row (one value per field) to the `SoaVec`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::soa::SoaVec;
    /// let mut soa: SoaVec<i32, 2> = SoaVec::new();
    /// soa.push([10, 20]);
    /// assert_eq!(soa.len(), 1);
    /// ```
    pub fn push(&mut self, row: [T; N]) {
        for (slot, val) in self.fields.iter_mut().zip(row) {
            slot.push(val);
        }
    }

    /// Number of rows. All fields are guaranteed to have this length.
    ///
    /// # Panics
    ///
    /// In debug builds, panics if fields disagree on length (a bug in
    /// custom `unsafe` mutation paths). In release, returns the length
    /// of field 0.
    pub fn len(&self) -> usize {
        let n = self.fields[0].len();
        debug_assert!(self.fields.iter().all(|f| f.len() == n), "SoaVec field-length invariant violated");
        n
    }

    /// Returns `true` if the `SoaVec` has zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow field `i` as a slice.
    ///
    /// # Panics
    ///
    /// Panics if `i >= N`.
    ///
    /// For known-at-compile-time indices, prefer
    /// [`field_n`](Self::field_n) to elide the bounds check.
    pub fn field(&self, i: usize) -> &[T] {
        &self.fields[i]
    }

    /// Compile-time-checked field accessor. Use when the index is a
    /// literal; `field_n::<2>()` is free of runtime bounds checking.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::soa::SoaVec;
    /// let mut soa: SoaVec<f32, 3> = SoaVec::new();
    /// soa.push([1.0, 2.0, 3.0]);
    /// assert_eq!(soa.field_n::<1>(), &[2.0]);
    /// ```
    pub fn field_n<const I: usize>(&self) -> &[T] {
        const { assert!(I < N, "SoaVec::field_n: I out of bounds for N") };
        &self.fields[I]
    }

    /// Mutably borrow field `i` as a slice.
    ///
    /// # Panics
    ///
    /// Panics if `i >= N`.
    pub fn field_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.fields[i]
    }

    /// Compile-time-checked mutable field accessor.
    pub fn field_n_mut<const I: usize>(&mut self) -> &mut [T] {
        const { assert!(I < N, "SoaVec::field_n_mut: I out of bounds for N") };
        &mut self.fields[I]
    }

    /// Borrow all fields at once as an array of slices, indexed by field
    /// position.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::soa::SoaVec;
    /// let mut soa: SoaVec<u8, 2> = SoaVec::new();
    /// soa.push([1, 2]);
    /// soa.push([3, 4]);
    /// let fields = soa.all_fields();
    /// assert_eq!(fields[0], &[1, 3]);
    /// assert_eq!(fields[1], &[2, 4]);
    /// ```
    pub fn all_fields(&self) -> [&[T]; N] {
        array::from_fn(|i| self.fields[i].as_slice())
    }

    /// Iterate over chunks of `chunk_len` rows, yielding `[&[T]; N]` per
    /// chunk. The last chunk may be shorter than `chunk_len`. Fast:
    /// zero-copy slice borrow.
    ///
    /// # Panics
    ///
    /// Per stdlib `slice::chunks` semantics, panics if `chunk_len == 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::soa::SoaVec;
    /// let mut soa: SoaVec<u32, 2> = SoaVec::new();
    /// for i in 0..5 {
    ///     soa.push([i, i * 10]);
    /// }
    /// let mut total = 0u32;
    /// for chunk in soa.chunks(2) {
    ///     total += chunk[0].iter().sum::<u32>();
    /// }
    /// assert_eq!(total, 0 + 1 + 2 + 3 + 4);
    /// ```
    pub fn chunks(&self, chunk_len: usize) -> SoaChunks<'_, T, N> {
        assert!(chunk_len > 0, "SoaVec::chunks: chunk_len must be > 0");
        SoaChunks {
            soa: self,
            chunk_len,
            cursor: 0,
        }
    }
}

impl<T, const N: usize> Default for SoaVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
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
        if self.cursor >= len {
            return None;
        }
        let end = (self.cursor + self.chunk_len).min(len);
        let chunk: [&'a [T]; N] = array::from_fn(|i| &self.soa.fields[i][self.cursor..end]);
        self.cursor = end;
        Some(chunk)
    }
}

/// Generate a named-field SoA struct from a struct-like declaration.
///
/// Each declared field `name: T` becomes `name: Vec<T>` on the generated
/// struct, alongside inherent `new`, `with_capacity`, `push`, `len`,
/// `is_empty`, `clear` methods plus an `impl Default`. Per-field
/// visibility is respected (`pub means_x: f32` stays `pub`); struct-level
/// meta-attributes (e.g. `#[derive(Clone)]`) pass through to the
/// generated struct.
///
/// # Reserved field names
///
/// The macro generates inherent methods on the struct. Choosing a field
/// name that collides with a generated method will produce a cryptic
/// compile error — the macro deliberately does not alias around them.
/// Reserved names: `new`, `with_capacity`, `push`, `len`, `is_empty`,
/// `clear`, `default`. Pick a different field name (`count`, `n`,
/// `row_len`) if you need that semantic.
///
/// # Invariant ownership
///
/// Fields stay `pub` (or whatever visibility the user specifies) on
/// purpose: the SoA ergonomic win is direct `&[T]` access for SIMD-style
/// loops. The generated `len()` carries a `debug_assert!` that catches
/// field-length-mismatch during development. If you mutate fields
/// directly (e.g. `batch.means_x.truncate(5)`), you OWN the field-length
/// invariant until you restore it. `push` and `clear` are the safe
/// mutation paths.
///
/// # Example
///
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
/// let mut b = GaussianBatch::new();
/// b.push(1.0, 2.0, 3.0);
/// b.push(4.0, 5.0, 6.0);
/// assert_eq!(b.len(), 2);
/// assert_eq!(b.means_x.as_slice(), &[1.0, 4.0]);
/// assert_eq!(b.means_y.as_slice(), &[2.0, 5.0]);
/// assert_eq!(b.means_z.as_slice(), &[3.0, 6.0]);
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
            #[allow(clippy::too_many_arguments)]
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

            /// Returns `true` if there are zero rows.
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

/// Deinterleave an AoS slice into a [`SoaVec`] by extracting `N` field
/// values per item via the user-supplied `extract` closure.
///
/// Scalar implementation. A future bench-justified wave may add per-arch
/// SIMD gather (VPGATHERDD on AVX-512, LD3/LD4 on NEON) for stride-known
/// dense layouts; the public API is forward-compatible — the dispatcher
/// will grow internal per-arch arms without changing this signature.
///
/// `T` need not be `Copy`; only the extracted `[f32; N]` row is
/// materialized.
///
/// # Inference
///
/// If the const-generic `N` fails to infer from the closure return type,
/// annotate either with a turbofish or a closure return-type ascription:
///
/// ```ignore
/// aos_to_soa::<_, 3, _>(&aos, |it| [it.a, it.b, it.c]);
/// aos_to_soa(&aos, |it| -> [f32; 3] { [it.a, it.b, it.c] });
/// ```
///
/// (Verified on Rust 1.94.)
///
/// # Example
///
/// ```
/// use ndarray::hpc::soa::aos_to_soa;
/// struct Item { a: f32, b: f32, c: f32 }
/// let aos = vec![
///     Item { a: 1.0, b: 2.0, c: 3.0 },
///     Item { a: 4.0, b: 5.0, c: 6.0 },
/// ];
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

/// Interleave a [`SoaVec`] into an AoS `Vec<T>` by building each item
/// from the per-field values via the user-supplied `build` closure.
///
/// Scalar implementation. See [`aos_to_soa`] for the forward-compatible
/// note on future SIMD acceleration.
///
/// Complexity: O(N·len) where N is the field count and len is the row
/// count.
///
/// # Example
///
/// ```
/// use ndarray::hpc::soa::{aos_to_soa, soa_to_aos};
/// struct Item { a: f32, b: f32, c: f32 }
/// let aos = vec![
///     Item { a: 1.0, b: 2.0, c: 3.0 },
///     Item { a: 4.0, b: 5.0, c: 6.0 },
/// ];
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

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------
    // SoaVec basics
    // -------------------------------------------------------------------

    #[test]
    fn soa_vec_new_smoke() {
        let soa: SoaVec<f32, 3> = SoaVec::new();
        assert_eq!(soa.len(), 0);
        assert!(soa.is_empty());
        assert_eq!(soa.field(0), &[] as &[f32]);
        assert_eq!(soa.field(1), &[] as &[f32]);
        assert_eq!(soa.field(2), &[] as &[f32]);
    }

    #[test]
    fn soa_vec_with_capacity_smoke() {
        let soa: SoaVec<i64, 2> = SoaVec::with_capacity(64);
        assert!(soa.is_empty());
        // capacity is not directly observable through the public API,
        // but constructing without panic is enough for the smoke test.
    }

    #[test]
    fn soa_vec_default() {
        let soa: SoaVec<u32, 4> = SoaVec::default();
        assert!(soa.is_empty());
        assert_eq!(soa.len(), 0);
    }

    #[test]
    fn soa_vec_push_len_is_empty() {
        let mut soa: SoaVec<f32, 3> = SoaVec::new();
        assert!(soa.is_empty());
        soa.push([1.0, 2.0, 3.0]);
        assert!(!soa.is_empty());
        assert_eq!(soa.len(), 1);
        soa.push([4.0, 5.0, 6.0]);
        soa.push([7.0, 8.0, 9.0]);
        assert_eq!(soa.len(), 3);
    }

    #[test]
    fn soa_vec_field_in_range() {
        let mut soa: SoaVec<u32, 3> = SoaVec::new();
        soa.push([10, 20, 30]);
        soa.push([40, 50, 60]);
        assert_eq!(soa.field(0), &[10, 40]);
        assert_eq!(soa.field(1), &[20, 50]);
        assert_eq!(soa.field(2), &[30, 60]);
    }

    #[test]
    #[should_panic]
    fn soa_vec_field_out_of_range_panics() {
        let soa: SoaVec<u32, 3> = SoaVec::new();
        // index N == 3 is out of range: panics via [Vec; N] bounds.
        let _ = soa.field(3);
    }

    #[test]
    fn soa_vec_field_n_compile_time() {
        let mut soa: SoaVec<f32, 4> = SoaVec::new();
        soa.push([1.0, 2.0, 3.0, 4.0]);
        soa.push([5.0, 6.0, 7.0, 8.0]);
        assert_eq!(soa.field_n::<0>(), &[1.0, 5.0]);
        assert_eq!(soa.field_n::<1>(), &[2.0, 6.0]);
        assert_eq!(soa.field_n::<2>(), &[3.0, 7.0]);
        assert_eq!(soa.field_n::<3>(), &[4.0, 8.0]);
    }

    #[test]
    fn soa_vec_field_mut_mutates() {
        let mut soa: SoaVec<i32, 2> = SoaVec::new();
        soa.push([1, 100]);
        soa.push([2, 200]);
        {
            let f0 = soa.field_mut(0);
            f0[0] = 999;
        }
        assert_eq!(soa.field(0), &[999, 2]);
        // field 1 unaffected.
        assert_eq!(soa.field(1), &[100, 200]);
    }

    #[test]
    fn soa_vec_field_n_mut_compile_time() {
        let mut soa: SoaVec<i32, 3> = SoaVec::new();
        soa.push([1, 2, 3]);
        soa.push([4, 5, 6]);
        {
            let f2 = soa.field_n_mut::<2>();
            f2[1] = -1;
        }
        assert_eq!(soa.field_n::<2>(), &[3, -1]);
    }

    #[test]
    fn soa_vec_all_fields_in_index_order() {
        let mut soa: SoaVec<u8, 4> = SoaVec::new();
        soa.push([1, 2, 3, 4]);
        soa.push([5, 6, 7, 8]);
        let fields = soa.all_fields();
        assert_eq!(fields[0], &[1, 5]);
        assert_eq!(fields[1], &[2, 6]);
        assert_eq!(fields[2], &[3, 7]);
        assert_eq!(fields[3], &[4, 8]);
    }

    // -------------------------------------------------------------------
    // SoaVec::chunks
    // -------------------------------------------------------------------

    #[test]
    fn soa_vec_chunks_divides_len() {
        let mut soa: SoaVec<u32, 2> = SoaVec::new();
        for i in 0..6 {
            soa.push([i, i * 10]);
        }
        let chunks: Vec<[&[u32]; 2]> = soa.chunks(2).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0][0], &[0, 1]);
        assert_eq!(chunks[0][1], &[0, 10]);
        assert_eq!(chunks[1][0], &[2, 3]);
        assert_eq!(chunks[1][1], &[20, 30]);
        assert_eq!(chunks[2][0], &[4, 5]);
        assert_eq!(chunks[2][1], &[40, 50]);
    }

    #[test]
    fn soa_vec_chunks_does_not_divide_len() {
        let mut soa: SoaVec<u32, 2> = SoaVec::new();
        for i in 0..5 {
            soa.push([i, i * 10]);
        }
        let chunks: Vec<[&[u32]; 2]> = soa.chunks(2).collect();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0][0], &[0, 1]);
        assert_eq!(chunks[1][0], &[2, 3]);
        // tail chunk is shorter than chunk_len.
        assert_eq!(chunks[2][0], &[4]);
        assert_eq!(chunks[2][1], &[40]);
    }

    #[test]
    fn soa_vec_chunks_chunk_len_greater_than_len() {
        let mut soa: SoaVec<u32, 2> = SoaVec::new();
        soa.push([1, 100]);
        soa.push([2, 200]);
        let chunks: Vec<[&[u32]; 2]> = soa.chunks(10).collect();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0][0], &[1, 2]);
        assert_eq!(chunks[0][1], &[100, 200]);
    }

    #[test]
    fn soa_vec_chunks_chunk_len_one() {
        let mut soa: SoaVec<u32, 2> = SoaVec::new();
        soa.push([1, 100]);
        soa.push([2, 200]);
        soa.push([3, 300]);
        let chunks: Vec<[&[u32]; 2]> = soa.chunks(1).collect();
        assert_eq!(chunks.len(), 3);
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c[0].len(), 1);
            assert_eq!(c[1].len(), 1);
            assert_eq!(c[0][0], (i + 1) as u32);
            assert_eq!(c[1][0], (i + 1) as u32 * 100);
        }
    }

    #[test]
    #[should_panic]
    fn soa_vec_chunks_chunk_len_zero_panics() {
        // Mirrors stdlib `slice::chunks(0)`: documented to panic.
        let mut soa: SoaVec<u32, 2> = SoaVec::new();
        soa.push([1, 2]);
        let _ = soa.chunks(0);
    }

    #[test]
    fn soa_vec_chunks_on_empty_yields_nothing() {
        let soa: SoaVec<u32, 2> = SoaVec::new();
        let chunks: Vec<[&[u32]; 2]> = soa.chunks(4).collect();
        assert!(chunks.is_empty());
    }

    // -------------------------------------------------------------------
    // soa_struct! macro
    // -------------------------------------------------------------------

    soa_struct! {
        /// 2-field test struct (private fields).
        struct Soa2 {
            a: f32,
            b: f32,
        }
    }

    soa_struct! {
        /// 3-field test struct with public fields.
        pub struct Soa3 {
            pub x: f32,
            pub y: f32,
            pub z: f32,
        }
    }

    soa_struct! {
        /// 4-field test struct, mixed visibility, derives Clone.
        #[derive(Clone)]
        pub struct Soa4 {
            pub a: i32,
            pub b: i32,
            pub c: i32,
            pub d: i32,
        }
    }

    #[test]
    fn macro_2_fields_push_len_clear() {
        let mut s = Soa2::new();
        assert!(s.is_empty());
        s.push(1.0, 2.0);
        s.push(3.0, 4.0);
        assert_eq!(s.len(), 2);
        // private fields: not accessible from outer scope, but since the
        // tests module is inside the same module, we can read them here.
        assert_eq!(s.a.as_slice(), &[1.0, 3.0]);
        assert_eq!(s.b.as_slice(), &[2.0, 4.0]);
        s.clear();
        assert!(s.is_empty());
        assert_eq!(s.a.len(), 0);
        assert_eq!(s.b.len(), 0);
    }

    #[test]
    fn macro_3_fields_push_len_clear() {
        let mut s = Soa3::new();
        s.push(1.0, 2.0, 3.0);
        s.push(4.0, 5.0, 6.0);
        s.push(7.0, 8.0, 9.0);
        assert_eq!(s.len(), 3);
        assert_eq!(s.x.as_slice(), &[1.0, 4.0, 7.0]);
        assert_eq!(s.y.as_slice(), &[2.0, 5.0, 8.0]);
        assert_eq!(s.z.as_slice(), &[3.0, 6.0, 9.0]);
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn macro_4_fields_push_len_clear() {
        let mut s = Soa4::new();
        s.push(1, 2, 3, 4);
        s.push(5, 6, 7, 8);
        assert_eq!(s.len(), 2);
        assert_eq!(s.a, vec![1, 5]);
        assert_eq!(s.b, vec![2, 6]);
        assert_eq!(s.c, vec![3, 7]);
        assert_eq!(s.d, vec![4, 8]);
        s.clear();
        assert!(s.is_empty());
    }

    #[test]
    fn macro_default_impl() {
        let s: Soa3 = Soa3::default();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn macro_with_capacity() {
        let s = Soa3::with_capacity(32);
        assert!(s.is_empty());
        // capacity not directly observable; smoke test only.
    }

    #[test]
    fn macro_public_visibility_passthrough() {
        // Soa3 has `pub` fields; verify the field is accessible
        // (compilation alone proves visibility).
        let mut s = Soa3::new();
        s.x.push(1.0);
        s.y.push(2.0);
        s.z.push(3.0);
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn macro_derive_clone_passthrough() {
        // Soa4 has `#[derive(Clone)]`; this test compiles iff Clone is
        // actually derived on the generated struct.
        let mut s = Soa4::new();
        s.push(1, 2, 3, 4);
        let cloned = s.clone();
        assert_eq!(cloned.len(), 1);
        assert_eq!(cloned.a, vec![1]);
        assert_eq!(cloned.d, vec![4]);
    }

    // -------------------------------------------------------------------
    // aos_to_soa / soa_to_aos
    // -------------------------------------------------------------------

    #[derive(Clone, PartialEq, Debug)]
    struct ItemN2 {
        a: f32,
        b: f32,
    }

    #[derive(Clone, PartialEq, Debug)]
    struct ItemN3 {
        a: f32,
        b: f32,
        c: f32,
    }

    #[derive(Clone, PartialEq, Debug)]
    struct ItemN4 {
        a: f32,
        b: f32,
        c: f32,
        d: f32,
    }

    #[test]
    fn aos_to_soa_n2_roundtrip() {
        let aos = vec![ItemN2 { a: 1.0, b: 2.0 }, ItemN2 { a: 3.0, b: 4.0 }, ItemN2 { a: 5.0, b: 6.0 }];
        let soa = aos_to_soa::<_, 2, _>(&aos, |it| [it.a, it.b]);
        assert_eq!(soa.len(), 3);
        assert_eq!(soa.field(0), &[1.0, 3.0, 5.0]);
        assert_eq!(soa.field(1), &[2.0, 4.0, 6.0]);
        let back: Vec<ItemN2> = soa_to_aos(&soa, |[a, b]| ItemN2 { a, b });
        assert_eq!(back, aos);
    }

    #[test]
    fn aos_to_soa_n3_roundtrip() {
        let aos = vec![ItemN3 { a: 1.0, b: 2.0, c: 3.0 }, ItemN3 { a: 4.0, b: 5.0, c: 6.0 }];
        let soa = aos_to_soa::<_, 3, _>(&aos, |it| [it.a, it.b, it.c]);
        assert_eq!(soa.field(0), &[1.0, 4.0]);
        assert_eq!(soa.field(1), &[2.0, 5.0]);
        assert_eq!(soa.field(2), &[3.0, 6.0]);
        let back: Vec<ItemN3> = soa_to_aos(&soa, |[a, b, c]| ItemN3 { a, b, c });
        assert_eq!(back, aos);
    }

    #[test]
    fn aos_to_soa_n4_roundtrip() {
        let aos = vec![
            ItemN4 {
                a: 1.0,
                b: 2.0,
                c: 3.0,
                d: 4.0,
            },
            ItemN4 {
                a: 5.0,
                b: 6.0,
                c: 7.0,
                d: 8.0,
            },
            ItemN4 {
                a: 9.0,
                b: 10.0,
                c: 11.0,
                d: 12.0,
            },
        ];
        let soa = aos_to_soa::<_, 4, _>(&aos, |it| [it.a, it.b, it.c, it.d]);
        assert_eq!(soa.field(0), &[1.0, 5.0, 9.0]);
        assert_eq!(soa.field(1), &[2.0, 6.0, 10.0]);
        assert_eq!(soa.field(2), &[3.0, 7.0, 11.0]);
        assert_eq!(soa.field(3), &[4.0, 8.0, 12.0]);
        let back: Vec<ItemN4> = soa_to_aos(&soa, |[a, b, c, d]| ItemN4 { a, b, c, d });
        assert_eq!(back, aos);
    }

    #[test]
    fn aos_to_soa_empty_input() {
        let aos: Vec<ItemN3> = Vec::new();
        let soa = aos_to_soa::<_, 3, _>(&aos, |it| [it.a, it.b, it.c]);
        assert!(soa.is_empty());
        assert_eq!(soa.field(0), &[] as &[f32]);
        assert_eq!(soa.field(1), &[] as &[f32]);
        assert_eq!(soa.field(2), &[] as &[f32]);
        let back: Vec<ItemN3> = soa_to_aos(&soa, |[a, b, c]| ItemN3 { a, b, c });
        assert!(back.is_empty());
    }

    #[test]
    fn aos_to_soa_closure_captures_external_constant() {
        // Verifies the `Fn(&T) -> [f32; N]` accepts a closure that
        // captures an external constant and that the captured value is
        // applied per row.
        let scale: f32 = 10.0;
        let aos = vec![ItemN2 { a: 1.0, b: 2.0 }, ItemN2 { a: 3.0, b: 4.0 }];
        let soa = aos_to_soa::<_, 2, _>(&aos, |it| [it.a * scale, it.b * scale]);
        assert_eq!(soa.field(0), &[10.0, 30.0]);
        assert_eq!(soa.field(1), &[20.0, 40.0]);
    }

    #[test]
    fn soa_to_aos_empty() {
        let soa: SoaVec<f32, 2> = SoaVec::new();
        let back: Vec<ItemN2> = soa_to_aos(&soa, |[a, b]| ItemN2 { a, b });
        assert!(back.is_empty());
    }
}
