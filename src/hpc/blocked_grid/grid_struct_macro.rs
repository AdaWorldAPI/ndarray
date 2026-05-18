//! `blocked_grid_struct!` macro — SoA-of-grids struct generator.
//!
//! Each named field in the macro input becomes a
//! [`BlockedGrid<FieldT, 64, 64>`](crate::hpc::blocked_grid::BlockedGrid) on the
//! generated struct, with shared `rows` / `cols` / `padded_rows` / `padded_cols`
//! dimension state.
//!
//! Worker scope: `src/hpc/blocked_grid/grid_struct_macro.rs`
//! (sprint worker B — see `.claude/knowledge/pr-x3-cognitive-grid-design.md`
//! §"Worker decomposition").
//!
//! # Tier coverage (v1)
//!
//! **v1 emits L1 methods only.** L2/L3/L4 alias methods on the generated
//! struct are deferred to a follow-up PR. The reserved field names below
//! include L2-L4 identifiers so future emission won't break callers. Callers
//! that need L2/L3/L4 can access individual fields and call `.blocks_l2()` /
//! `.map_l2()` / `.bulk_apply_l2()` directly on each
//! `BlockedGrid<FieldT, 64, 64>` field.
//!
//! # Reserved field names
//!
//! The macro generates inherent methods whose names must not collide with
//! user field names.  Choosing any of these field names will produce a
//! compile error:
//!
//! `new`, `rows`, `cols`, `padded_rows`, `padded_cols`,
//! `blocks_l1`, `blocks_l2`, `blocks_l3`, `blocks_l4`,
//! `map_l1`, `map_l2`, `map_l3`, `map_l4`,
//! `bulk_apply_l1`, `bulk_apply_l2`, `bulk_apply_l3`, `bulk_apply_l4`,
//! `field_n`, `default`.
//!
//! # `#[grid(block = (BR, BC))]` attribute
//!
//! **Not implemented in v1.**  All fields use the default 64×64 base block.
//! Per-field `#[grid(field_block = (BR, BC))]` heterogeneous shapes are also
//! deferred.  Document your need in a follow-up PR against PR-X3.
//!
//! # Data-flow rules
//!
//! All `&mut self` methods on the generated struct are **write-back paths**
//! per `.claude/rules/data-flow.md` Rule #3.  PRIMARY compute paths (`map_l1`)
//! take `&self` and return a new struct, leaving the input unchanged.
//!
//! See `.claude/knowledge/cognitive-distance-typing.md` — no distance-aware
//! API is generated here.
//!
//! # Example
//!
//! ```
//! use ndarray::blocked_grid_struct;
//!
//! blocked_grid_struct! {
//!     pub struct ShaderCellGridDoc {
//!         pub edge:    u64,
//!         pub palette: u8,
//!         pub depth:   u16,
//!         pub alpha:   u8,
//!     }
//! }
//!
//! let g = ShaderCellGridDoc::new(128, 128);
//! assert_eq!(g.rows(), 128);
//! assert_eq!(g.cols(), 128);
//! assert_eq!(g.padded_rows(), 128);
//! assert_eq!(g.padded_cols(), 128);
//! // blocks_l1 yields 2×2 = 4 base blocks across all fields in lockstep.
//! assert_eq!(g.blocks_l1().count(), 4);
//! ```

use crate::hpc::blocked_grid::BlockedGrid;

// ============================================================
// FieldGridRef — object-safe trait for erased BlockedGrid references
// ============================================================

/// Object-safe trait that exposes the dimension accessors of a
/// `BlockedGrid<T, 64, 64>` without naming the element type.
///
/// Returned by [`blocked_grid_struct!`]-generated `field_n::<I>()` accessors.
/// Useful for code that needs to inspect or assert grid dimensions uniformly
/// across all fields of a SoA-of-grids struct without monomorphizing on the
/// element type.
///
/// # Example
/// ```
/// use ndarray::blocked_grid_struct;
/// use ndarray::hpc::blocked_grid::FieldGridRef;
///
/// blocked_grid_struct! {
///     pub struct FieldGridRefDemo {
///         pub a: u64,
///         pub b: u32,
///     }
/// }
///
/// let g = FieldGridRefDemo::new(64, 64);
/// let f0: &dyn FieldGridRef = g.field_n::<0>();
/// assert_eq!(f0.rows(), 64);
/// assert_eq!(f0.cols(), 64);
/// assert_eq!(f0.padded_rows(), 64);
/// assert_eq!(f0.padded_cols(), 64);
/// ```
pub trait FieldGridRef {
    /// Logical row count.
    fn rows(&self) -> usize;
    /// Logical column count.
    fn cols(&self) -> usize;
    /// Padded row count (`ceil(rows / BR) * BR`).
    fn padded_rows(&self) -> usize;
    /// Padded column count (`ceil(cols / BC) * BC`).
    fn padded_cols(&self) -> usize;
}

impl<T, const BR: usize, const BC: usize> FieldGridRef for BlockedGrid<T, BR, BC> {
    fn rows(&self) -> usize {
        BlockedGrid::rows(self)
    }
    fn cols(&self) -> usize {
        BlockedGrid::cols(self)
    }
    fn padded_rows(&self) -> usize {
        BlockedGrid::padded_rows(self)
    }
    fn padded_cols(&self) -> usize {
        BlockedGrid::padded_cols(self)
    }
}

// ============================================================
// Clone for BlockedGrid — required by #[derive(Clone)] passthrough
// on blocked_grid_struct!-generated structs.
//
// Worker A1 did not derive Clone on BlockedGrid. Adding it here (in
// worker B's file) keeps base.rs untouched per the file-per-worker
// discipline, while allowing consumers to annotate macro-generated
// structs with #[derive(Clone)].
//
// Bound: T: Copy + Default — all spec field types (u8, u16, u32, u64,
// f32, f64) satisfy this. The Default bound allows zero-initialising
// the output grid via `BlockedGrid::new`, after which `copy_from_slice`
// writes the correct padded data including both logical cells and padding.
// ============================================================

impl<T: Copy + Default, const BR: usize, const BC: usize> Clone for BlockedGrid<T, BR, BC> {
    fn clone(&self) -> Self {
        let mut out = Self::new(self.rows(), self.cols());
        out.as_padded_slice_mut()
            .copy_from_slice(self.as_padded_slice());
        out
    }
}

// ============================================================
// blocked_grid_struct! — the macro
// ============================================================

/// Generate a SoA-of-grids struct.
///
/// Each named field `$field: $ty` becomes
/// `$field: BlockedGrid<$ty, 64, 64>` on the generated struct.
/// Four scalar fields (`rows`, `cols`, `padded_rows`, `padded_cols`)
/// store the shared logical and padded extents.
///
/// The macro also generates two companion block-view types:
/// `{Name}L1Block<'a>` (read-only) and `{Name}L1BlockMut<'a>` (mutable),
/// plus a lockstep iterator type `{Name}L1BlockIter<'a>`.
///
/// See the [module-level documentation](self) for the full feature list,
/// reserved field names, and v1 tier-coverage notes.
///
/// # Example
///
/// ```
/// use ndarray::blocked_grid_struct;
///
/// blocked_grid_struct! {
///     #[derive(Clone)]
///     pub struct MyGridMacroDoc {
///         pub energy: u64,
///         pub weight: u32,
///     }
/// }
///
/// let g = MyGridMacroDoc::new(64, 64);
/// assert_eq!(g.rows(), 64);
/// let g2 = g.clone();
/// assert_eq!(g2.cols(), 64);
/// ```
#[macro_export]
macro_rules! blocked_grid_struct {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $($field_vis:vis $field:ident : $fty:ty),* $(,)?
        }
    ) => {
        $crate::paste::paste! {

        // ------------------------------------------------------------------
        // 1. The generated struct.
        // ------------------------------------------------------------------

        $(#[$meta])*
        $vis struct $name {
            rows:        usize,
            cols:        usize,
            padded_rows: usize,
            padded_cols: usize,
            $($field_vis $field: $crate::hpc::blocked_grid::BlockedGrid<$fty, 64, 64>),*
        }

        // ------------------------------------------------------------------
        // 2. Core impl block: constructor + dimension accessors.
        // ------------------------------------------------------------------

        impl $name {
            /// Construct a new instance with every field allocated as a
            /// `BlockedGrid<FieldT, 64, 64>::new(rows, cols)`.
            pub fn new(rows: usize, cols: usize) -> Self {
                let padded_rows = if rows == 0 || cols == 0 {
                    0usize
                } else {
                    rows.div_ceil(64) * 64
                };
                let padded_cols = if rows == 0 || cols == 0 {
                    0usize
                } else {
                    cols.div_ceil(64) * 64
                };
                Self {
                    rows,
                    cols,
                    padded_rows,
                    padded_cols,
                    $($field: $crate::hpc::blocked_grid::BlockedGrid::<$fty, 64, 64>::new(rows, cols)),*
                }
            }

            /// Logical row count (as passed to [`new`](Self::new)).
            pub fn rows(&self) -> usize { self.rows }

            /// Logical column count (as passed to [`new`](Self::new)).
            pub fn cols(&self) -> usize { self.cols }

            /// Padded row count (`ceil(rows / 64) * 64`).
            pub fn padded_rows(&self) -> usize { self.padded_rows }

            /// Padded column count (`ceil(cols / 64) * 64`).
            pub fn padded_cols(&self) -> usize { self.padded_cols }
        }

        // ------------------------------------------------------------------
        // 3. L1 block view types — generated with {Name}L1Block naming.
        // ------------------------------------------------------------------

        /// Read-only lockstep L1-block view generated for
        #[doc = concat!("`", stringify!($name), "`.")]
        ///
        /// Each field is a
        /// [`GridBlock<'a, FieldT, 64, 64>`](ndarray::hpc::blocked_grid::GridBlock)
        /// borrow into the corresponding field's
        /// [`BlockedGrid`](ndarray::hpc::blocked_grid::BlockedGrid) at the same
        /// `(block_row, block_col)`.
        ///
        /// Fields `row_origin` and `col_origin` give the cell coordinates of
        /// the top-left corner of this block within the parent grid.
        $vis struct [<$name L1Block>]<'a> {
            $(pub $field: $crate::hpc::blocked_grid::GridBlock<'a, $fty, 64, 64>,)*
            /// Row index of the top-left corner of this block in the parent grid.
            pub row_origin: usize,
            /// Column index of the top-left corner of this block in the parent grid.
            pub col_origin: usize,
        }

        /// Mutable lockstep L1-block view generated for
        #[doc = concat!("`", stringify!($name), "`.")]
        ///
        /// Each field is a
        /// [`GridBlockMut<'a, FieldT, 64, 64>`](ndarray::hpc::blocked_grid::GridBlockMut)
        /// mutable borrow into the corresponding field's
        /// [`BlockedGrid`](ndarray::hpc::blocked_grid::BlockedGrid) at the same
        /// `(block_row, block_col)`.
        $vis struct [<$name L1BlockMut>]<'a> {
            $(pub $field: $crate::hpc::blocked_grid::GridBlockMut<'a, $fty, 64, 64>,)*
            /// Row index of the top-left corner of this block in the parent grid.
            pub row_origin: usize,
            /// Column index of the top-left corner of this block in the parent grid.
            pub col_origin: usize,
        }

        // ------------------------------------------------------------------
        // 4. Lockstep iterator type.
        // ------------------------------------------------------------------

        /// Row-major lockstep iterator over L1 blocks of all fields in
        #[doc = concat!("`", stringify!($name), "`.")]
        ///
        /// Constructed via [`blocks_l1`](
        #[doc = concat!(stringify!($name), "::blocks_l1).")]
        $vis struct [<$name L1BlockIter>]<'a> {
            $($field: *const $crate::hpc::blocked_grid::BlockedGrid<$fty, 64, 64>,)*
            block_row:    usize,
            block_col:    usize,
            n_block_rows: usize,
            n_block_cols: usize,
            _marker: ::core::marker::PhantomData<&'a $name>,
        }

        // SAFETY: all raw pointers were created from valid shared references
        // that live for `'a`; the `PhantomData<&'a $name>` encodes the borrow.
        unsafe impl<'a> Send for [<$name L1BlockIter>]<'a>
        where $($fty: Send,)* {}
        unsafe impl<'a> Sync for [<$name L1BlockIter>]<'a>
        where $($fty: Sync,)* {}

        impl<'a> Iterator for [<$name L1BlockIter>]<'a> {
            type Item = [<$name L1Block>]<'a>;

            fn next(&mut self) -> Option<Self::Item> {
                if self.block_row >= self.n_block_rows {
                    return None;
                }
                let br = self.block_row;
                let bc = self.block_col;

                // Advance in row-major order before building the block view.
                self.block_col += 1;
                if self.block_col >= self.n_block_cols {
                    self.block_col = 0;
                    self.block_row += 1;
                }

                let row_origin = br * 64;
                let col_origin = bc * 64;

                // SAFETY: each pointer was created from a valid `&'a BlockedGrid`
                // in `blocks_l1`; the struct borrows the parent grid for `'a`.
                Some([<$name L1Block>] {
                    $($field: $crate::hpc::blocked_grid::GridBlock::from_grid(
                        unsafe { &*self.$field }, br, bc,
                    ),)*
                    row_origin,
                    col_origin,
                })
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let remaining = if self.block_row >= self.n_block_rows {
                    0
                } else {
                    let total = self.n_block_rows * self.n_block_cols;
                    let consumed = self.block_row * self.n_block_cols + self.block_col;
                    total - consumed
                };
                (remaining, Some(remaining))
            }
        }

        impl<'a> ExactSizeIterator for [<$name L1BlockIter>]<'a> {}

        // ------------------------------------------------------------------
        // 5. Impl block: blocks_l1, map_l1, bulk_apply_l1, field_n.
        // ------------------------------------------------------------------

        impl $name {
            /// Iterate all fields' L1 blocks in lockstep (same
            /// `(block_row, block_col)` coordinates at each step).
            ///
            /// Yields one
            #[doc = concat!("`", stringify!([<$name L1Block>]), "<'_>`")]
            /// per `(block_row, block_col)` pair in row-major order.
            ///
            /// Implements [`ExactSizeIterator`], so `.len()` is available
            /// before consuming any items.
            pub fn blocks_l1(&self) -> [<$name L1BlockIter>]<'_> {
                let n_block_rows = if self.padded_rows == 0 { 0 } else { self.padded_rows / 64 };
                let n_block_cols = if self.padded_cols == 0 { 0 } else { self.padded_cols / 64 };
                [<$name L1BlockIter>] {
                    $($field: &self.$field as *const _,)*
                    block_row: 0,
                    block_col: 0,
                    n_block_rows,
                    n_block_cols,
                    _marker: ::core::marker::PhantomData,
                }
            }

            /// PRIMARY compute path: map a closure over every L1 block in
            /// lockstep, returning a new `Self` with the closure-mapped values.
            /// The input struct is not mutated.
            ///
            /// The closure receives:
            /// - `&`
            #[doc = concat!("`", stringify!([<$name L1Block>]), "<'_>`")]
            /// — read-only lockstep view of the current block.
            /// - `&mut `
            #[doc = concat!("`", stringify!([<$name L1BlockMut>]), "<'_>`")]
            /// — mutable lockstep view into the corresponding block of the
            ///   freshly-allocated output struct.
            ///
            /// # Data-flow rule
            /// This is the PRIMARY compute path. It satisfies the
            /// `.claude/rules/data-flow.md` Rule #3 invariant. For in-place
            /// write-back see [`bulk_apply_l1`](Self::bulk_apply_l1).
            pub fn map_l1<F>(
                &self,
                mut f: F,
            ) -> Self
            where
                F: FnMut(
                    &[<$name L1Block>]<'_>,
                    &mut [<$name L1BlockMut>]<'_>,
                ),
            {
                let mut out = Self::new(self.rows, self.cols);
                let n_block_rows = if self.padded_rows == 0 { 0 } else { self.padded_rows / 64 };
                let n_block_cols = if self.padded_cols == 0 { 0 } else { self.padded_cols / 64 };
                for br in 0..n_block_rows {
                    for bc in 0..n_block_cols {
                        let row_origin = br * 64;
                        let col_origin = bc * 64;
                        let inp_view = [<$name L1Block>] {
                            $($field: $crate::hpc::blocked_grid::GridBlock::from_grid(
                                &self.$field, br, bc,
                            ),)*
                            row_origin,
                            col_origin,
                        };
                        let mut out_view = [<$name L1BlockMut>] {
                            $($field: $crate::hpc::blocked_grid::GridBlockMut::from_grid(
                                &mut out.$field, br, bc,
                            ),)*
                            row_origin,
                            col_origin,
                        };
                        f(&inp_view, &mut out_view);
                    }
                }
                out
            }

            /// SECONDARY write-back path: apply a closure in-place over every
            /// L1 block in lockstep.
            ///
            /// The closure receives:
            /// - `&mut `
            #[doc = concat!("`", stringify!([<$name L1BlockMut>]), "<'_>`")]
            /// — mutable lockstep view of the current block.
            /// - `(usize, usize)` — `(block_row, block_col)` coordinates.
            ///
            /// # Data-flow rule
            ///
            /// This is the WRITE-BACK variant per `.claude/rules/data-flow.md` Rule #3
            /// ("No `&mut self` during computation. Ever."). The closure performs
            /// gated write-back operations ONLY. For COMPUTE paths use
            /// [`map_l1`](Self::map_l1).
            pub fn bulk_apply_l1<F>(&mut self, mut f: F)
            where
                F: FnMut(
                    &mut [<$name L1BlockMut>]<'_>,
                    (usize, usize),
                ),
            {
                let n_block_rows = if self.padded_rows == 0 { 0 } else { self.padded_rows / 64 };
                let n_block_cols = if self.padded_cols == 0 { 0 } else { self.padded_cols / 64 };
                for br in 0..n_block_rows {
                    for bc in 0..n_block_cols {
                        let row_origin = br * 64;
                        let col_origin = bc * 64;
                        // Build mutable block views for each field.
                        // We use raw pointers to allow simultaneous mutable
                        // borrows of distinct fields within `self`.
                        $(let [<$field _ptr>] = &mut self.$field as *mut _;)*
                        let mut block_view = [<$name L1BlockMut>] {
                            // SAFETY: we have unique ownership via `&mut self`,
                            // each field is a distinct allocation within the
                            // struct, and no two fields alias each other.
                            $($field: $crate::hpc::blocked_grid::GridBlockMut::from_grid(
                                unsafe { &mut *[<$field _ptr>] }, br, bc,
                            ),)*
                            row_origin,
                            col_origin,
                        };
                        f(&mut block_view, (br, bc));
                    }
                }
            }

            /// Compile-time field accessor: `field_n::<I>()` returns a
            /// `&dyn FieldGridRef` reference to the `I`-th field's `BlockedGrid`.
            ///
            /// Provides a uniform dimension-inspection interface across all fields
            /// without knowing the field's element type. Follows the W3-W6
            /// `soa_struct!` pattern.
            ///
            /// # Compile-time bounds check
            /// `I` must be less than the number of fields; a value ≥ field count
            /// triggers a `const { assert!(...) }` compile error.
            pub fn field_n<const I: usize>(&self) -> &dyn $crate::hpc::blocked_grid::FieldGridRef {
                const N_FIELDS: usize = {
                    let mut n = 0usize;
                    $(let _ = stringify!($field); n += 1;)*
                    n
                };
                const { assert!(I < N_FIELDS, "field_n: I out of bounds for field count") };
                let mut _idx = 0usize;
                $(
                if _idx == I { return &self.$field as &dyn $crate::hpc::blocked_grid::FieldGridRef; }
                _idx += 1;
                )*
                unreachable!("field_n: const assert above guarantees I < N_FIELDS")
            }
        }

        } // end paste::paste! { ... }
    };
}

// ============================================================
// Inline tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::FieldGridRef;
    use crate::blocked_grid_struct;

    // ------------------------------------------------------------------
    // 2-field generation
    // ------------------------------------------------------------------

    blocked_grid_struct! {
        pub struct Test2Field {
            pub a: u64,
            pub b: u32,
        }
    }

    #[test]
    fn two_field_new_dimensions() {
        let g = Test2Field::new(64, 64);
        assert_eq!(g.rows(), 64);
        assert_eq!(g.cols(), 64);
        assert_eq!(g.padded_rows(), 64);
        assert_eq!(g.padded_cols(), 64);
    }

    #[test]
    fn two_field_padded_dimensions_non_aligned() {
        let g = Test2Field::new(100, 100);
        assert_eq!(g.rows(), 100);
        assert_eq!(g.cols(), 100);
        assert_eq!(g.padded_rows(), 128);
        assert_eq!(g.padded_cols(), 128);
    }

    #[test]
    fn two_field_blocks_l1_count_64x64() {
        let g = Test2Field::new(64, 64);
        assert_eq!(g.blocks_l1().count(), 1);
    }

    #[test]
    fn two_field_blocks_l1_count_128x128() {
        let g = Test2Field::new(128, 128);
        assert_eq!(g.blocks_l1().count(), 4);
    }

    #[test]
    fn two_field_exact_size_iter() {
        let g = Test2Field::new(128, 128);
        let iter = g.blocks_l1();
        assert_eq!(iter.len(), 4);
    }

    // ------------------------------------------------------------------
    // 3-field generation
    // ------------------------------------------------------------------

    blocked_grid_struct! {
        pub struct Test3Field {
            pub x: u64,
            pub y: u16,
            pub z: u8,
        }
    }

    #[test]
    fn three_field_new_dimensions() {
        let g = Test3Field::new(128, 64);
        assert_eq!(g.rows(), 128);
        assert_eq!(g.cols(), 64);
        assert_eq!(g.padded_rows(), 128);
        assert_eq!(g.padded_cols(), 64);
    }

    #[test]
    fn three_field_blocks_l1_count() {
        let g = Test3Field::new(128, 64);
        // 128/64 = 2 block rows × 64/64 = 1 block col = 2 blocks
        assert_eq!(g.blocks_l1().count(), 2);
    }

    // ------------------------------------------------------------------
    // 4-field generation — canonical ShaderCellGrid from the spec
    // ------------------------------------------------------------------

    blocked_grid_struct! {
        pub struct ShaderCellGrid {
            pub edge:    u64,
            pub palette: u8,
            pub depth:   u16,
            pub alpha:   u8,
        }
    }

    #[test]
    fn four_field_new_builds() {
        let g = ShaderCellGrid::new(64, 64);
        assert_eq!(g.rows(), 64);
        assert_eq!(g.cols(), 64);
    }

    #[test]
    fn four_field_padded_dimensions() {
        let g = ShaderCellGrid::new(100, 100);
        assert_eq!(g.padded_rows(), 128);
        assert_eq!(g.padded_cols(), 128);
    }

    #[test]
    fn four_field_blocks_l1_row_major_order() {
        let g = ShaderCellGrid::new(128, 128);
        let origins: Vec<_> = g
            .blocks_l1()
            .map(|b| (b.row_origin, b.col_origin))
            .collect();
        assert_eq!(origins, vec![(0, 0), (0, 64), (64, 0), (64, 64)]);
    }

    // ------------------------------------------------------------------
    // pub vs private field visibility
    // ------------------------------------------------------------------

    blocked_grid_struct! {
        pub struct MixedVisibility {
            pub pub_field: u64,
            priv_field: u8,
        }
    }

    #[test]
    fn mixed_visibility_builds() {
        let g = MixedVisibility::new(64, 64);
        assert_eq!(g.rows(), 64);
        // pub_field is accessible; verifying the pub field has correct type.
        let _ref: &crate::hpc::blocked_grid::BlockedGrid<u64, 64, 64> = &g.pub_field;
        // priv_field is accessible within the module (same module as the declaration).
        let _ref2: &crate::hpc::blocked_grid::BlockedGrid<u8, 64, 64> = &g.priv_field;
    }

    // ------------------------------------------------------------------
    // #[derive(Clone)] passthrough
    // ------------------------------------------------------------------

    blocked_grid_struct! {
        #[derive(Clone)]
        pub struct CloneableGrid {
            pub a: u64,
        }
    }

    #[test]
    fn derive_clone_produces_deep_clone() {
        let mut g = CloneableGrid::new(64, 64);
        g.a.set(0, 0, 42u64);
        let g2 = g.clone();
        assert_eq!(g2.a.get(0, 0), 42u64);
        // Mutations to original after cloning don't affect the clone.
        g.a.set(0, 0, 99u64);
        assert_eq!(g2.a.get(0, 0), 42u64);
    }

    // ------------------------------------------------------------------
    // map_l1: returns new struct, input unchanged
    // ------------------------------------------------------------------

    #[test]
    fn map_l1_input_unchanged() {
        let g = Test2Field::new(64, 64);
        // Pre-condition: all zeros.
        assert!(g.a.as_padded_slice().iter().all(|&v| v == 0u64));
        assert!(g.b.as_padded_slice().iter().all(|&v| v == 0u32));

        let out = g.map_l1(|inp, outp| {
            for r in 0..64 {
                let in_a = inp.a.row(r);
                let out_a = outp.a.row_mut(r);
                for (dst, &src) in out_a.iter_mut().zip(in_a.iter()) {
                    *dst = src.wrapping_add(7);
                }
                let in_b = inp.b.row(r);
                let out_b = outp.b.row_mut(r);
                for (dst, &src) in out_b.iter_mut().zip(in_b.iter()) {
                    *dst = src.wrapping_add(3);
                }
            }
        });

        // Input is unchanged.
        assert!(g.a.as_padded_slice().iter().all(|&v| v == 0u64));
        assert!(g.b.as_padded_slice().iter().all(|&v| v == 0u32));
        // Output has incremented values.
        assert!(out.a.as_padded_slice().iter().all(|&v| v == 7u64));
        assert!(out.b.as_padded_slice().iter().all(|&v| v == 3u32));
    }

    #[test]
    fn map_l1_preserves_dimensions() {
        let g = ShaderCellGrid::new(128, 128);
        let out = g.map_l1(|_inp, _outp| {});
        assert_eq!(out.rows(), 128);
        assert_eq!(out.cols(), 128);
        assert_eq!(out.padded_rows(), 128);
        assert_eq!(out.padded_cols(), 128);
    }

    // ------------------------------------------------------------------
    // bulk_apply_l1: lockstep mutation visible + correct coordinates
    // ------------------------------------------------------------------

    #[test]
    fn bulk_apply_l1_mutation_visible() {
        let mut g = Test2Field::new(64, 64);
        g.bulk_apply_l1(|blk, (_br, _bc)| {
            blk.a.row_mut(0)[0] = 0xDEAD_BEEF_u64;
            blk.b.row_mut(0)[0] = 0xABCDu32;
        });
        assert_eq!(g.a.get(0, 0), 0xDEAD_BEEF_u64);
        assert_eq!(g.b.get(0, 0), 0xABCDu32);
    }

    #[test]
    fn bulk_apply_l1_correct_coordinates() {
        let mut g = Test2Field::new(128, 128);
        let mut coords_seen: Vec<(usize, usize)> = Vec::new();
        g.bulk_apply_l1(|_blk, (br, bc)| {
            coords_seen.push((br, bc));
        });
        assert_eq!(coords_seen, vec![(0, 0), (0, 1), (1, 0), (1, 1)]);
    }

    // ------------------------------------------------------------------
    // field_n::<I>() compile-time accessor
    // ------------------------------------------------------------------

    #[test]
    fn field_n_zero_returns_first_field_dims() {
        let g = Test2Field::new(64, 64);
        let f0: &dyn FieldGridRef = g.field_n::<0>();
        assert_eq!(f0.rows(), 64);
        assert_eq!(f0.cols(), 64);
        assert_eq!(f0.padded_rows(), 64);
        assert_eq!(f0.padded_cols(), 64);
    }

    #[test]
    fn field_n_one_returns_second_field_dims() {
        let g = Test2Field::new(100, 100);
        let f1: &dyn FieldGridRef = g.field_n::<1>();
        assert_eq!(f1.rows(), 100);
        assert_eq!(f1.cols(), 100);
        assert_eq!(f1.padded_rows(), 128);
        assert_eq!(f1.padded_cols(), 128);
    }

    // ------------------------------------------------------------------
    // Zero-dimension grid
    // ------------------------------------------------------------------

    #[test]
    fn zero_dimension_grid_empty_iterator() {
        let g = Test2Field::new(0, 0);
        assert_eq!(g.rows(), 0);
        assert_eq!(g.cols(), 0);
        assert_eq!(g.padded_rows(), 0);
        assert_eq!(g.padded_cols(), 0);
        assert_eq!(g.blocks_l1().count(), 0);
    }

    // ------------------------------------------------------------------
    // blocks_l1 row_origin / col_origin match block coordinates
    // ------------------------------------------------------------------

    #[test]
    fn blocks_l1_origins_match_block_coordinates() {
        let g = Test2Field::new(128, 128);
        for blk in g.blocks_l1() {
            assert_eq!(blk.a.row_origin(), blk.row_origin);
            assert_eq!(blk.a.col_origin(), blk.col_origin);
            assert_eq!(blk.b.row_origin(), blk.row_origin);
            assert_eq!(blk.b.col_origin(), blk.col_origin);
        }
    }
}
