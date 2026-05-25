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
//! # Element-type scope (PR-X2)
//!
//! `SoaVec`, the `soa_struct!` macro, and the closure-based conversion
//! helpers [`aos_to_soa`] / [`soa_to_aos`] are **fully generic over the
//! element type `U`** (was f32-hardwired through W3-W6; PR-X2 lifted the
//! constraint). Common element types now flow through directly:
//!
//! - `f32` — Gaussian batch means, covariances (original W3-W6 case)
//! - `u64` — `CausalEdge64` mantissa cells, NARS evidence packs
//! - `u16` — BF16 carrier values, packed depth fields
//! - `u8`  — palette indices, quantized embeddings
//! - `i8`  — quantized weights with signed range
//!
//! Callers passing turbofish should now use four type params:
//! `aos_to_soa::<_, U, N, _>(...)` instead of the pre-PR-X2 form
//! `aos_to_soa::<_, N, _>(...)`. Callers using return-type inference are
//! unaffected by the generalisation.
//!
//! # Layering — why `hpc::soa` and not `simd_ops`
//!
//! `crate::simd_ops` is the SIMD-dispatch glue layer (every fn there
//! dispatches through `F32x16` / `F64x8`). Per the W1a consumer contract
//! at `.claude/knowledge/vertical-simd-consumer-contract.md`, free-function
//! shapes like `fn aos_to_soa(&[T], extract) -> SoaVec<U, N>` belong
//! at the `crate::hpc` level, co-located with the data types they
//! convert between. Putting pure-scalar helpers in `simd_ops` would
//! contradict that module's charter and the W1a litmus that rejects
//! "generic-library free functions" from the SIMD layer.
//!
//! This module is **scalar only**. It contains no `#[target_feature]`
//! attributes, no `cfg(target_feature = ...)` gates, no per-arch imports.
//! The public API is forward-compatible with a future bench-justified
//! SIMD swap: the dispatcher inside any conversion entry can grow per-arch
//! arms internally (delegating to `simd_avx512.rs` / `simd_neon.rs` for
//! gather / scatter intrinsics) without changing the user-visible
//! signature.
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
///
/// # Example — `#[soa(pad_to_lanes = N)]` field attribute (PR-X2 Worker B)
///
/// Tag a field with `#[soa(pad_to_lanes = N)]` to make `push` pad the
/// underlying `Vec` up to the next multiple of `N` (filling with
/// `Default::default()`). SIMD-staged kernels then walk the field with
/// one uniform N-lane loop — no tail-case branch.
///
/// `len()` returns the **logical** row count (unchanged by padding);
/// `self.<field>.len()` returns the **physical** Vec length. The difference
/// is the lane-alignment tail.
///
/// ```
/// use ndarray::soa_struct;
///
/// soa_struct! {
///     pub struct Cells {
///         #[soa(pad_to_lanes = 8)]
///         pub palette: u8,
///         pub label: u32,           // unpadded
///     }
/// }
///
/// let mut c = Cells::new();
/// c.push(7, 100);
/// assert_eq!(c.len(), 1);           // logical: 1 row
/// assert_eq!(c.palette.len(), 8);   // physical: rounded up to lane 8
/// assert_eq!(c.label.len(), 1);     // unpadded: physical == logical
/// assert_eq!(c.palette[0], 7);
/// assert_eq!(c.palette[1..8], [0u8; 7]); // padded tail is Default::default()
/// ```
#[macro_export]
macro_rules! soa_struct {
    // ───────────────────────────────────────────────────────────────────
    // Arm 1 — unpadded (no `#[soa(...)]` attribute on any field).
    // This is byte-for-byte the pre-PR-X2 emit: no `_logical_len` field,
    // `len()` reads from field lengths under `debug_assert`. Existing
    // callers (struct-literal construction, exhaustive patterns) are
    // unaffected. macro_rules! tries this arm first; if any field has
    // a `#[soa(pad_to_lanes = N)]` attribute the pattern fails to match
    // and arm 2 is tried.
    // ───────────────────────────────────────────────────────────────────
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

    // ───────────────────────────────────────────────────────────────────
    // Arm 2 — padded (at least one field has `#[soa(pad_to_lanes = N)]`).
    // Adds a `#[doc(hidden)] _logical_len: usize` field so `len()` can
    // return the semantic row count independent of lane-tail padding.
    // Reached only when arm 1's no-attribute pattern fails to match —
    // existing callers without padding never see this struct shape.
    // ───────────────────────────────────────────────────────────────────
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $(
                $(#[soa(pad_to_lanes = $pad:literal)])?
                $field_vis:vis $field:ident : $ty:ty
            ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        $vis struct $name {
            $($field_vis $field: ::std::vec::Vec<$ty>,)*
            /// Shared logical row count across all fields. Padded fields may
            /// have `self.<field>.len() > _logical_len` after `push`.
            /// Updated by `push` / `clear`; treat as private.
            ///
            /// Only present on padded structs (at least one field has
            /// `#[soa(pad_to_lanes = N)]`); unpadded structs keep the
            /// pre-PR-X2 all-public shape.
            #[doc(hidden)]
            _logical_len: usize,
        }

        impl $name {
            /// Construct an empty instance.
            pub fn new() -> Self {
                Self {
                    $($field: ::std::vec::Vec::new(),)*
                    _logical_len: 0,
                }
            }

            /// Construct with each field pre-allocated to `cap`.
            ///
            /// Padded fields per `#[soa(pad_to_lanes = N)]` get
            /// `cap` worth of physical capacity, not `cap.div_ceil(N) * N` —
            /// the lane padding happens lazily inside `push` so the up-front
            /// reservation is a hint, not a hard size guarantee.
            pub fn with_capacity(cap: usize) -> Self {
                Self {
                    $($field: ::std::vec::Vec::with_capacity(cap),)*
                    _logical_len: 0,
                }
            }

            /// Append one row across all fields.
            ///
            /// For fields tagged `#[soa(pad_to_lanes = N)]`, the underlying
            /// `Vec` is padded with `<$ty as Default>::default()` up to the
            /// next multiple of `N` before the new value is written. Padded
            /// elements occupy slots `[_logical_len + 1 .. padded_len)` and
            /// are guaranteed to compare equal to `Default::default()`.
            #[allow(clippy::too_many_arguments)]
            pub fn push(&mut self, $($field: $ty),*) {
                let logical = self._logical_len;
                $(
                    $crate::soa_struct!(@push_field
                        self, $field, $field, $ty, logical
                        $(, pad = $pad)?
                    );
                )*
                self._logical_len = logical + 1;
            }

            /// Logical row count (shared across all fields).
            ///
            /// For padded fields this may be **less than** `self.<field>.len()`;
            /// the difference is the lane-alignment tail. Use `len()` for the
            /// semantic count, `self.<field>.len()` for the physical Vec length.
            pub fn len(&self) -> usize {
                self._logical_len
            }

            /// Returns `true` if there are zero logical rows.
            pub fn is_empty(&self) -> bool { self._logical_len == 0 }

            /// Clear all fields. Capacity is retained; logical length resets to 0.
            ///
            /// Padded fields' physical `Vec`s are cleared along with the
            /// unpadded ones — re-pushing into a cleared struct rebuilds the
            /// padding from scratch.
            pub fn clear(&mut self) {
                $(self.$field.clear();)*
                self._logical_len = 0;
            }
        }

        impl ::std::default::Default for $name {
            fn default() -> Self { Self::new() }
        }
    };

    // Internal — padded field push: grow Vec to the next multiple of $pad
    // with Default::default() before writing the new value at `logical`.
    (@push_field $self:ident, $vec:ident, $val:ident, $ty:ty, $logical:ident, pad = $pad:literal) => {{
        const _: () = {
            // Compile-time guard: pad_to_lanes = 0 is nonsensical.
            assert!($pad > 0, "soa_struct! #[soa(pad_to_lanes = N)] requires N > 0");
        };
        let needed = ($logical + 1).div_ceil($pad) * $pad;
        while $self.$vec.len() < needed {
            $self.$vec.push(<$ty as ::std::default::Default>::default());
        }
        $self.$vec[$logical] = $val;
    }};

    // Internal — plain (unpadded) field push inside a padded struct
    // (mixed cadence: some fields padded, others not).
    (@push_field $self:ident, $vec:ident, $val:ident, $ty:ty, $logical:ident) => {{
        $self.$vec.push($val);
    }};
}

/// Deinterleave an AoS slice into a [`SoaVec<U, N>`] by extracting `N`
/// field values per item via the user-supplied `extract` closure.
///
/// `U` is the element type of the resulting `SoaVec` — generic over all
/// `Copy` types. Common values:
/// - `f32` — Gaussian batch means, covariances (original W3-W6 use case)
/// - `u64` — `CausalEdge64` mantissa cells, NARS evidence packs
/// - `u16` — BF16 carrier values, packed depth fields
/// - `u8` — palette indices, quantized embeddings
///
/// Scalar implementation. A future bench-justified wave may add per-arch
/// SIMD gather (VPGATHERDD on AVX-512, LD3/LD4 on NEON). The public
/// signature is forward-compatible — the dispatcher will grow internal
/// per-arch arms without changing this signature.
///
/// `T` need not be `Copy`; only the extracted `[U; N]` row is materialised.
///
/// # Inference
///
/// If `N` fails to infer from the closure return type, annotate via
/// turbofish (note: 4 type params now, was 3 in the f32-only era):
///
/// ```ignore
/// aos_to_soa::<_, u64, 3, _>(&aos, |it| [it.a, it.b, it.c]);
/// aos_to_soa(&aos, |it| -> [u64; 3] { [it.a, it.b, it.c] });
/// ```
///
/// # Example — f32 (backwards-compatible)
///
/// ```
/// use ndarray::hpc::soa::aos_to_soa;
/// struct Item { a: f32, b: f32, c: f32 }
/// let aos = vec![
///     Item { a: 1.0, b: 2.0, c: 3.0 },
///     Item { a: 4.0, b: 5.0, c: 6.0 },
/// ];
/// let soa = aos_to_soa::<_, f32, 3, _>(&aos, |it| [it.a, it.b, it.c]);
/// assert_eq!(soa.field(0), &[1.0, 4.0]);
/// assert_eq!(soa.field(1), &[2.0, 5.0]);
/// assert_eq!(soa.field(2), &[3.0, 6.0]);
/// ```
///
/// # Example — u64 (CausalEdge64-style)
///
/// ```
/// use ndarray::hpc::soa::aos_to_soa;
/// struct Edge { src: u64, dst: u64, weight: u64 }
/// let aos = vec![
///     Edge { src: 1, dst: 2, weight: 10 },
///     Edge { src: 3, dst: 4, weight: 20 },
/// ];
/// let soa = aos_to_soa::<_, u64, 3, _>(&aos, |e| [e.src, e.dst, e.weight]);
/// assert_eq!(soa.field(0), &[1u64, 3]);
/// assert_eq!(soa.field(2), &[10u64, 20]);
/// ```
///
/// # Example — u8 (palette indices)
///
/// ```
/// use ndarray::hpc::soa::aos_to_soa;
/// struct Cell { palette: u8, alpha: u8 }
/// let aos = vec![Cell { palette: 7, alpha: 255 }, Cell { palette: 3, alpha: 128 }];
/// let soa = aos_to_soa::<_, u8, 2, _>(&aos, |c| [c.palette, c.alpha]);
/// assert_eq!(soa.field(0), &[7u8, 3]);
/// assert_eq!(soa.field(1), &[255u8, 128]);
/// ```
pub fn aos_to_soa<T, U, const N: usize, F>(aos: &[T], extract: F) -> SoaVec<U, N>
where
    F: Fn(&T) -> [U; N],
{
    let mut soa = SoaVec::<U, N>::with_capacity(aos.len());
    for item in aos {
        soa.push(extract(item));
    }
    soa
}

/// Interleave a [`SoaVec<U, N>`] into an AoS `Vec<T>` by building each item
/// from the per-field values via the user-supplied `build` closure.
///
/// `U` is the element type of the input `SoaVec` (must be `Copy` so a
/// per-row `[U; N]` can be materialised by indexing). Scalar implementation;
/// the public signature is forward-compatible per [`aos_to_soa`].
///
/// Complexity: O(N·len) where N is the field count and len is the row
/// count.
///
/// # Example — f32 (backwards-compatible)
///
/// ```
/// use ndarray::hpc::soa::{aos_to_soa, soa_to_aos};
/// struct Item { a: f32, b: f32, c: f32 }
/// let aos = vec![
///     Item { a: 1.0, b: 2.0, c: 3.0 },
///     Item { a: 4.0, b: 5.0, c: 6.0 },
/// ];
/// let soa = aos_to_soa::<_, f32, 3, _>(&aos, |it| [it.a, it.b, it.c]);
/// let back: Vec<Item> = soa_to_aos(&soa, |[a, b, c]| Item { a, b, c });
/// assert_eq!(back[0].a, 1.0);
/// assert_eq!(back[1].c, 6.0);
/// ```
///
/// # Example — u16 (BF16 carrier)
///
/// ```
/// use ndarray::hpc::soa::{aos_to_soa, soa_to_aos};
/// #[derive(Debug, PartialEq)]
/// struct Pair { lo: u16, hi: u16 }
/// let aos = vec![Pair { lo: 0x1234, hi: 0xABCD }, Pair { lo: 0x5678, hi: 0xEF01 }];
/// let soa = aos_to_soa::<_, u16, 2, _>(&aos, |p| [p.lo, p.hi]);
/// let back: Vec<Pair> = soa_to_aos(&soa, |[lo, hi]| Pair { lo, hi });
/// assert_eq!(back[0], Pair { lo: 0x1234, hi: 0xABCD });
/// assert_eq!(back[1], Pair { lo: 0x5678, hi: 0xEF01 });
/// ```
pub fn soa_to_aos<T, U, const N: usize, F>(soa: &SoaVec<U, N>, build: F) -> Vec<T>
where
    F: Fn([U; N]) -> T,
    U: Copy,
{
    let n = soa.len();
    let fields = soa.all_fields();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row: [U; N] = core::array::from_fn(|k| fields[k][i]);
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
        // (compilation alone proves visibility). Soa3 is unpadded → uses
        // arm 1 of the macro → fields drive `len()` directly, so pushing
        // into individual fields still gives the right count.
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
        let soa = aos_to_soa::<_, _, 2, _>(&aos, |it| [it.a, it.b]);
        assert_eq!(soa.len(), 3);
        assert_eq!(soa.field(0), &[1.0, 3.0, 5.0]);
        assert_eq!(soa.field(1), &[2.0, 4.0, 6.0]);
        let back: Vec<ItemN2> = soa_to_aos(&soa, |[a, b]| ItemN2 { a, b });
        assert_eq!(back, aos);
    }

    #[test]
    fn aos_to_soa_n3_roundtrip() {
        let aos = vec![ItemN3 { a: 1.0, b: 2.0, c: 3.0 }, ItemN3 { a: 4.0, b: 5.0, c: 6.0 }];
        let soa = aos_to_soa::<_, _, 3, _>(&aos, |it| [it.a, it.b, it.c]);
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
        let soa = aos_to_soa::<_, _, 4, _>(&aos, |it| [it.a, it.b, it.c, it.d]);
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
        let soa = aos_to_soa::<_, _, 3, _>(&aos, |it| [it.a, it.b, it.c]);
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
        let soa = aos_to_soa::<_, _, 2, _>(&aos, |it| [it.a * scale, it.b * scale]);
        assert_eq!(soa.field(0), &[10.0, 30.0]);
        assert_eq!(soa.field(1), &[20.0, 40.0]);
    }

    #[test]
    fn soa_to_aos_empty() {
        let soa: SoaVec<f32, 2> = SoaVec::new();
        let back: Vec<ItemN2> = soa_to_aos(&soa, |[a, b]| ItemN2 { a, b });
        assert!(back.is_empty());
    }

    // ------------------------------------------------------------------
    // PR-X2 — generic-U coverage (was f32-hardwired through W3-W6)
    // ------------------------------------------------------------------

    /// `aos_to_soa` over `u64` (CausalEdge64-style fields).
    #[test]
    fn aos_to_soa_u64_round_trip() {
        struct Edge {
            src: u64,
            dst: u64,
            weight: u64,
        }
        let aos = [
            Edge {
                src: 1,
                dst: 2,
                weight: 10,
            },
            Edge {
                src: 3,
                dst: 4,
                weight: 20,
            },
            Edge {
                src: 0xDEAD_BEEF_CAFE_BABE,
                dst: 0,
                weight: u64::MAX,
            },
        ];
        let soa = aos_to_soa::<_, u64, 3, _>(&aos, |e| [e.src, e.dst, e.weight]);
        assert_eq!(soa.len(), 3);
        assert_eq!(soa.field(0), &[1u64, 3, 0xDEAD_BEEF_CAFE_BABE]);
        assert_eq!(soa.field(1), &[2u64, 4, 0]);
        assert_eq!(soa.field(2), &[10u64, 20, u64::MAX]);
    }

    /// `aos_to_soa` over `u8` (palette indices) plus `soa_to_aos` round-trip.
    #[test]
    fn aos_to_soa_u8_round_trip() {
        #[derive(Debug, PartialEq, Eq)]
        struct Cell {
            palette: u8,
            alpha: u8,
        }
        let aos = vec![Cell { palette: 7, alpha: 255 }, Cell { palette: 3, alpha: 128 }, Cell { palette: 0, alpha: 0 }];
        let soa = aos_to_soa::<_, u8, 2, _>(&aos, |c| [c.palette, c.alpha]);
        assert_eq!(soa.field(0), &[7u8, 3, 0]);
        assert_eq!(soa.field(1), &[255u8, 128, 0]);

        let back: Vec<Cell> = soa_to_aos(&soa, |[palette, alpha]| Cell { palette, alpha });
        assert_eq!(back, aos);
    }

    /// `aos_to_soa` over `u16` (BF16 carrier bytes).
    #[test]
    fn aos_to_soa_u16_round_trip() {
        #[derive(Debug, PartialEq, Eq)]
        struct Bf16Pair {
            lo: u16,
            hi: u16,
        }
        let aos = vec![
            Bf16Pair { lo: 0x1234, hi: 0xABCD },
            Bf16Pair { lo: 0x5678, hi: 0xEF01 },
            Bf16Pair { lo: 0xFFFF, hi: 0x0000 },
        ];
        let soa = aos_to_soa::<_, u16, 2, _>(&aos, |p| [p.lo, p.hi]);
        assert_eq!(soa.field(0), &[0x1234u16, 0x5678, 0xFFFF]);
        assert_eq!(soa.field(1), &[0xABCDu16, 0xEF01, 0x0000]);

        let back: Vec<Bf16Pair> = soa_to_aos(&soa, |[lo, hi]| Bf16Pair { lo, hi });
        assert_eq!(back, aos);
    }

    // ------------------------------------------------------------------
    // PR-X2 Worker B — `#[soa(pad_to_lanes = N)]` field attribute
    // ------------------------------------------------------------------

    soa_struct! {
        /// 3-field SoA with two padded fields at different lane widths and
        /// one unpadded field. Exercises the mixed-cadence macro arm.
        pub struct PadMixed {
            #[soa(pad_to_lanes = 8)]
            pub palette: u8,
            #[soa(pad_to_lanes = 16)]
            pub depth: u16,
            pub label: u32,
        }
    }

    /// Single push into a `pad_to_lanes = 8` field rounds the physical Vec
    /// up to 8 elements; logical len is 1.
    #[test]
    fn pad_to_lanes_single_push_grows_to_lane() {
        let mut s = PadMixed::new();
        s.push(7u8, 0x1234u16, 99u32);
        assert_eq!(s.len(), 1, "logical len = 1");
        assert_eq!(s.palette.len(), 8, "palette padded to lane 8");
        assert_eq!(s.depth.len(), 16, "depth padded to lane 16");
        assert_eq!(s.label.len(), 1, "label unpadded — physical = logical");
        assert_eq!(s.palette[0], 7);
        assert_eq!(s.depth[0], 0x1234);
        assert_eq!(s.label[0], 99);
        // Padded tail is Default::default().
        for &b in &s.palette[1..8] {
            assert_eq!(b, 0u8);
        }
        for &d in &s.depth[1..16] {
            assert_eq!(d, 0u16);
        }
    }

    /// Crossing a lane boundary on a padded field grows the Vec by another N.
    #[test]
    fn pad_to_lanes_crosses_lane_boundary() {
        let mut s = PadMixed::new();
        for i in 0..9u8 {
            s.push(i, i as u16, i as u32);
        }
        assert_eq!(s.len(), 9);
        // palette: 9 pushes → next multiple of 8 is 16
        assert_eq!(s.palette.len(), 16);
        // depth: 9 pushes → still inside lane 16
        assert_eq!(s.depth.len(), 16);
        // label: unpadded
        assert_eq!(s.label.len(), 9);
        // first 9 slots carry user values
        for i in 0..9 {
            assert_eq!(s.palette[i], i as u8);
            assert_eq!(s.depth[i], i as u16);
            assert_eq!(s.label[i], i as u32);
        }
        // tail is default-zeroed
        for &b in &s.palette[9..16] {
            assert_eq!(b, 0u8);
        }
    }

    /// `clear()` resets logical_len and clears physical Vecs.
    #[test]
    fn pad_to_lanes_clear_resets_both() {
        let mut s = PadMixed::new();
        s.push(1, 2, 3);
        s.push(4, 5, 6);
        assert_eq!(s.len(), 2);
        s.clear();
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert_eq!(s.palette.len(), 0);
        assert_eq!(s.depth.len(), 0);
        assert_eq!(s.label.len(), 0);
        // Reuse after clear works — padding rebuilds from scratch.
        s.push(99, 0xFFFF, 7);
        assert_eq!(s.len(), 1);
        assert_eq!(s.palette.len(), 8);
        assert_eq!(s.depth.len(), 16);
    }

    soa_struct! {
        /// All-padded variant — every field gets the same lane width.
        pub struct PadUniform {
            #[soa(pad_to_lanes = 4)]
            pub a: i32,
            #[soa(pad_to_lanes = 4)]
            pub b: i32,
        }
    }

    /// All-padded struct: every field grows in sync with the lane cadence.
    #[test]
    fn pad_to_lanes_uniform_cadence() {
        let mut s = PadUniform::new();
        s.push(10, 20);
        s.push(30, 40);
        s.push(50, 60);
        assert_eq!(s.len(), 3);
        // 3 pushes → next multiple of 4 is 4
        assert_eq!(s.a.len(), 4);
        assert_eq!(s.b.len(), 4);
        assert_eq!(s.a[0..3], [10, 30, 50]);
        assert_eq!(s.b[0..3], [20, 40, 60]);
        assert_eq!(s.a[3], 0);
        assert_eq!(s.b[3], 0);
    }

    /// `with_capacity` initialises an empty padded struct correctly.
    #[test]
    fn pad_to_lanes_with_capacity_empty() {
        let s = PadMixed::with_capacity(64);
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
        assert_eq!(s.palette.len(), 0);
        assert_eq!(s.depth.len(), 0);
        assert_eq!(s.label.len(), 0);
    }

    /// Inference-only entry: caller relies on closure return-type ascription,
    /// no turbofish at all.
    #[test]
    fn aos_to_soa_inference_only() {
        struct Triple {
            a: i8,
            b: i8,
            c: i8,
        }
        let aos = [Triple { a: 1, b: 2, c: 3 }, Triple { a: -1, b: -2, c: -3 }];
        let soa = aos_to_soa(&aos, |t| -> [i8; 3] { [t.a, t.b, t.c] });
        assert_eq!(soa.field(0), &[1i8, -1]);
        assert_eq!(soa.field(2), &[3i8, -3]);
    }
}
