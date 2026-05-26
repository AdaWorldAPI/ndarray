//! Integration tests for `src/hpc/blocked_grid` — PR-X3 worker A6.
//!
//! These tests span multiple submodules (base, iter, super_block, compute,
//! aliases) and exercise cross-cutting concerns that inline `#[cfg(test)]`
//! blocks inside individual submodules cannot elegantly cover.
//!
//! Test groups
//! -----------
//! 1. W4 bulk_apply composition  — map_l1 composes with `hpc::bulk::bulk_apply`
//!    and `bulk_for_each` over per-row slices inside the closure.
//! 2. L1→L2 cascade              — 256×256 ShaderMantissaGrid map_l1 populates
//!    cell-by-cell, then map_l2 aggregates per super-block.
//! 3. Half-square AMX INT8       — AmxInt8Grid::new(32, 128), blocks_base coords.
//! 4. All seven type aliases      — instantiate and verify padding math.
//! 5. as_padded_slice footgun     — right-vs-wrong flat-index computation on a
//!    non-aligned 100×100 grid.
//! 6. Const-generic compile-fail  — see doctest in mod.rs.

use crate::hpc::blocked_grid::{
    AmxBf16Grid, AmxInt8Grid, BlockedGrid, HalfSquareU64, ShaderMantissaGrid, SquareF64Stack8, StripF32Stack2,
    StripF32Stack4,
};

// ============================================================
// 1. W4 bulk_apply / bulk_for_each composition
//
// Demonstrates that PR-X3's map_l1 composes with the W4 primitives:
//   - outer loop = map_l1 (one closure per 64×64 base block)
//   - inner loop = bulk_apply / bulk_for_each over each row slice in the block
//
// This proves the two design layers nest without either re-implementing the
// other's chunking logic.
// ============================================================

/// map_l1 closure using bulk_for_each to read each row and compute a per-row
/// sum, storing it into the first cell of the corresponding output row.
///
/// Demonstrates: bulk_for_each(row_slice, chunk_size, closure) correctly
/// accumulates the sum; no re-implemented chunking inside map_l1.
#[test]
fn w4_bulk_for_each_inside_map_l1_row_sum() {
    use crate::hpc::bulk::bulk_for_each;

    // Build a 64×64 grid filled with known values.
    let mut g = BlockedGrid::<u64>::new(64, 64);
    // Fill each cell with its column index (so row sums = 0+1+…+63 = 2016).
    for r in 0..64 {
        for c in 0..64usize {
            g.set(r, c, c as u64);
        }
    }

    // map_l1: for each block row, use bulk_for_each to compute the row sum and
    // store it in the first cell of the output row.
    let out = g.map_l1::<u64, _>(|inp, outp| {
        for r in 0..64 {
            let row = inp.row(r);
            let mut row_sum = 0u64;
            bulk_for_each(row, 16, |chunk, _start| {
                row_sum += chunk.iter().sum::<u64>();
            });
            outp.row_mut(r)[0] = row_sum;
        }
    });

    // Input unchanged.
    for r in 0..64 {
        for c in 0..64usize {
            assert_eq!(g.get(r, c), c as u64, "input mutated at ({}, {})", r, c);
        }
    }

    // Each output row's first cell should be 0+1+…+63 = 2016.
    let expected_sum: u64 = (0u64..64).sum();
    for r in 0..64 {
        assert_eq!(out.get(r, 0), expected_sum, "row {} sum mismatch", r);
    }
}

/// map_l1 closure using bulk_apply to write-scale each row element by its
/// column index (element[c] *= c).  Demonstrates mutable composition.
#[test]
fn w4_bulk_apply_inside_map_l1_scale_by_col() {
    use crate::hpc::bulk::bulk_apply;

    let g = BlockedGrid::<u64>::new(64, 64);
    // Input is all-zero; map to a grid where each cell equals its column index.
    let out = g.map_l1::<u64, _>(|_inp, outp| {
        for r in 0..64 {
            let row = outp.row_mut(r);
            // Use bulk_apply to initialise each chunk: set element[i] = start + i.
            bulk_apply(row, 8, |chunk, start| {
                for (i, dst) in chunk.iter_mut().enumerate() {
                    *dst = (start + i) as u64;
                }
            });
        }
    });

    // Input unchanged (all zeros).
    assert!(g.as_padded_slice().iter().all(|&v| v == 0), "input was mutated");

    // Output: cell (r, c) == c for all logical cells.
    for r in 0..64 {
        for c in 0..64usize {
            assert_eq!(out.get(r, c), c as u64, "cell ({}, {}) wrong", r, c);
        }
    }
}

// ============================================================
// 2. L1→L2 cascade
//
// Build a 256×256 ShaderMantissaGrid, map_l1 to populate each cell with
// (row_origin + col_origin) derived from the block origins, then map_l2 to
// aggregate: for each 256×256 super-block sum the four corner cells of the
// four L1 sub-blocks.  Verify the cascade produces the expected value.
// ============================================================

/// Populate cell (r, c) = (block_row_origin + block_col_origin) via map_l1,
/// then verify by reading back.
#[test]
fn l1_cascade_populate_from_block_origins() {
    let g = ShaderMantissaGrid::new(256, 256);

    // map_l1: fill every cell in block with (row_origin + col_origin).
    let populated = g.map_l1::<u64, _>(|inp, outp| {
        // block origins from the input block view are encoded in row/col origin.
        // We use the block coordinates: row_origin = block_row * 64.
        // We access the raw block_row / block_col from the GridBlock.
        let row_origin = inp.block_row() * 64;
        let col_origin = inp.block_col() * 64;
        let fill = (row_origin + col_origin) as u64;
        for r in 0..64 {
            let out_row = outp.row_mut(r);
            for dst in out_row.iter_mut() {
                *dst = fill;
            }
        }
    });

    // Input unchanged (all zeros).
    assert!(g.as_padded_slice().iter().all(|&v| v == 0), "input mutated by map_l1");

    // Verify the 4 blocks:
    // block (0,0): row_origin=0, col_origin=0, fill=0
    assert_eq!(populated.get(0, 0), 0);
    assert_eq!(populated.get(63, 63), 0);
    // block (0,1): row_origin=0, col_origin=64, fill=64
    assert_eq!(populated.get(0, 64), 64);
    assert_eq!(populated.get(63, 127), 64);
    // block (1,0): row_origin=64, col_origin=0, fill=64
    assert_eq!(populated.get(64, 0), 64);
    assert_eq!(populated.get(127, 63), 64);
    // block (1,1): row_origin=64, col_origin=64, fill=128
    assert_eq!(populated.get(64, 64), 128);
    assert_eq!(populated.get(127, 127), 128);
}

/// L1→L2 cascade: after map_l1 populates the grid, map_l2 aggregates per
/// super-block by summing the first cell of each of the 4×4 = 16 L1 sub-blocks
/// via blocks_l1 on the super-block.
#[test]
fn l1_l2_cascade_aggregate_super_block() {
    let g = ShaderMantissaGrid::new(256, 256);

    // Stage 1: map_l1 → each cell gets value 1.
    let stage1 = g.map_l1::<u64, _>(|_inp, outp| {
        for r in 0..64 {
            for dst in outp.row_mut(r).iter_mut() {
                *dst = 1u64;
            }
        }
    });

    // Stage 2: map_l2 → for the single 256×256 super-block, count the
    // number of cells that equal 1 by iterating the super-block's base blocks.
    let stage2 = stage1.map_l2::<u64, _>(|inp_sb, outp_sb| {
        // Count cells == 1 across all base blocks in this super-block.
        let mut cell_count = 0u64;
        for blk in inp_sb.base_blocks() {
            for r in 0..64 {
                for &v in blk.row(r).iter() {
                    if v == 1 {
                        cell_count += 1;
                    }
                }
            }
        }
        // Write the total into the first cell of the output super-block's first
        // base block.
        outp_sb.base_blocks_mut().next().unwrap().row_mut(0)[0] = cell_count;
    });

    // stage1 is unchanged.
    assert!(stage1.as_padded_slice().iter().all(|&v| v == 1), "stage1 mutated by map_l2");

    // stage2: first cell = 256*256 = 65536 (all padded cells were set to 1).
    assert_eq!(stage2.get(0, 0), 256 * 256);
}

// ============================================================
// 3. Half-square AMX INT8 pattern
//
// AmxInt8Grid::new(32, 128) → 2 block-rows × 2 block-cols = 4 base blocks.
// Each block is 16×64.  Verify block coords and origins.
// ============================================================

/// AmxInt8Grid (16×64 blocks) on a 32×128 grid yields 4 base blocks with
/// the expected (block_row, block_col) coordinates and (row_origin, col_origin).
#[test]
fn amx_int8_half_square_block_coords_and_origins() {
    // 32 rows / 16 = 2 block rows; 128 cols / 64 = 2 block cols → 4 blocks.
    let g = AmxInt8Grid::new(32, 128);
    assert_eq!(g.padded_rows(), 32);
    assert_eq!(g.padded_cols(), 128);

    let blocks: Vec<_> = g.blocks_base().collect();
    assert_eq!(blocks.len(), 4, "expected 4 base blocks");

    // Row-major order: (0,0), (0,1), (1,0), (1,1)
    let expected = [
        (0usize, 0usize, 0usize, 0usize), // (block_row, block_col, row_origin, col_origin)
        (0, 1, 0, 64),
        (1, 0, 16, 0),
        (1, 1, 16, 64),
    ];

    for (i, blk) in blocks.iter().enumerate() {
        let (exp_br, exp_bc, exp_ro, exp_co) = expected[i];
        assert_eq!(blk.block_row(), exp_br, "block {} block_row", i);
        assert_eq!(blk.block_col(), exp_bc, "block {} block_col", i);
        assert_eq!(blk.row_origin(), exp_ro, "block {} row_origin", i);
        assert_eq!(blk.col_origin(), exp_co, "block {} col_origin", i);
    }
}

/// Block_dims for AmxInt8Grid is (16, 64).
#[test]
fn amx_int8_block_dims() {
    assert_eq!(AmxInt8Grid::block_dims(), (16, 64));
}

// ============================================================
// 4. All seven type aliases instantiate with correct padding
// ============================================================

/// ShaderMantissaGrid (64×64 blocks): 100×100 → padded 128×128.
#[test]
fn alias_shader_mantissa_grid_padding() {
    let g = ShaderMantissaGrid::new(100, 100);
    assert_eq!(g.rows(), 100);
    assert_eq!(g.cols(), 100);
    assert_eq!(g.padded_rows(), 128); // ceil(100/64)*64 = 128
    assert_eq!(g.padded_cols(), 128);
    assert_eq!(g.as_padded_slice().len(), 128 * 128);
}

/// AmxBf16Grid (16×16 blocks): 30×30 → padded 32×32.
#[test]
fn alias_amx_bf16_grid_padding() {
    let g = AmxBf16Grid::new(30, 30);
    assert_eq!(g.padded_rows(), 32); // ceil(30/16)*16 = 32
    assert_eq!(g.padded_cols(), 32);
    assert_eq!(g.as_padded_slice().len(), 32 * 32);
}

/// AmxInt8Grid (16×64 blocks): 17×65 → padded 32×128.
#[test]
fn alias_amx_int8_grid_padding() {
    let g = AmxInt8Grid::new(17, 65);
    assert_eq!(g.padded_rows(), 32); // ceil(17/16)*16 = 32
    assert_eq!(g.padded_cols(), 128); // ceil(65/64)*64 = 128
    assert_eq!(g.as_padded_slice().len(), 32 * 128);
}

/// StripF32Stack2 (2×16 blocks): 3×17 → padded 4×32.
#[test]
fn alias_strip_f32_stack2_padding() {
    let g = StripF32Stack2::new(3, 17);
    assert_eq!(g.padded_rows(), 4); // ceil(3/2)*2 = 4
    assert_eq!(g.padded_cols(), 32); // ceil(17/16)*16 = 32
    assert_eq!(g.as_padded_slice().len(), 4 * 32);
}

/// StripF32Stack4 (4×16 blocks): 5×33 → padded 8×48.
#[test]
fn alias_strip_f32_stack4_padding() {
    let g = StripF32Stack4::new(5, 33);
    assert_eq!(g.padded_rows(), 8); // ceil(5/4)*4 = 8
    assert_eq!(g.padded_cols(), 48); // ceil(33/16)*16 = 48
    assert_eq!(g.as_padded_slice().len(), 8 * 48);
}

/// SquareF64Stack8 (8×8 blocks): 9×9 → padded 16×16.
#[test]
fn alias_square_f64_stack8_padding() {
    let g = SquareF64Stack8::new(9, 9);
    assert_eq!(g.padded_rows(), 16); // ceil(9/8)*8 = 16
    assert_eq!(g.padded_cols(), 16);
    assert_eq!(g.as_padded_slice().len(), 16 * 16);
}

/// HalfSquareU64 (32×64 blocks): 33×65 → padded 64×128.
#[test]
fn alias_half_square_u64_padding() {
    let g = HalfSquareU64::new(33, 65);
    assert_eq!(g.padded_rows(), 64); // ceil(33/32)*32 = 64
    assert_eq!(g.padded_cols(), 128); // ceil(65/64)*64 = 128
    assert_eq!(g.as_padded_slice().len(), 64 * 128);
}

/// All seven aliases are default-initialised (all-zero / all-default).
#[test]
fn alias_all_seven_zero_initialised() {
    let g1 = ShaderMantissaGrid::new(64, 64);
    assert!(g1.as_padded_slice().iter().all(|&v| v == 0u64));

    let g2 = AmxBf16Grid::new(16, 16);
    assert!(g2.as_padded_slice().iter().all(|&v| v == 0u16));

    let g3 = AmxInt8Grid::new(16, 64);
    assert!(g3.as_padded_slice().iter().all(|&v| v == 0u8));

    let g4 = StripF32Stack2::new(2, 16);
    assert!(g4.as_padded_slice().iter().all(|&v| v == 0.0f32));

    let g5 = StripF32Stack4::new(4, 16);
    assert!(g5.as_padded_slice().iter().all(|&v| v == 0.0f32));

    let g6 = SquareF64Stack8::new(8, 8);
    assert!(g6.as_padded_slice().iter().all(|&v| v == 0.0f64));

    let g7 = HalfSquareU64::new(32, 64);
    assert!(g7.as_padded_slice().iter().all(|&v| v == 0u64));
}

// ============================================================
// 5. as_padded_slice footgun verification
//
// On a 100×100 grid with 64×64 blocks, padded_cols == 128 (not 100).
// The WRONG flat index for cell (r, c) is `r * cols() + c` = `r * 100 + c`.
// The RIGHT flat index is `idx(r, c)` = `r * padded_cols() + c` = `r * 128 + c`.
//
// This test exercises both paths and verifies they DIVERGE for any cell
// beyond the first logical row (where the stride gap first appears).
// ============================================================

/// The wrong formula r*cols()+c diverges from the right formula idx(r,c)
/// for any row r >= 1 on a non-aligned grid (100×100 padded to 128×128).
#[test]
fn padded_slice_footgun_right_vs_wrong_diverge() {
    let mut g = BlockedGrid::<u64>::new(100, 100);
    // Fill all logical cells with a sentinel to distinguish from pad.
    for r in 0..100 {
        for c in 0..100 {
            g.set(r, c, 0xABCD_0000_0000_0001_u64);
        }
    }

    let slice = g.as_padded_slice();

    // For cell (1, 0):
    // RIGHT index = 1 * 128 + 0 = 128
    let right_idx = g.idx(1, 0);
    assert_eq!(right_idx, 128, "right idx for (1,0) should be 128");
    assert_eq!(slice[right_idx], 0xABCD_0000_0000_0001_u64, "right idx reads cell (1,0)");

    // WRONG index = 1 * 100 + 0 = 100
    let wrong_idx = 1 * g.cols() + 0;
    assert_eq!(wrong_idx, 100, "wrong idx for (1,0) should be 100");
    // Index 100 falls in the padding gap of row 0 (cells 100..128 are pad = 0).
    assert_eq!(slice[wrong_idx], 0u64, "wrong idx reads pad cell (should be 0)");

    // The two indices differ.
    assert_ne!(right_idx, wrong_idx, "right and wrong indices must diverge");
    assert_ne!(slice[right_idx], slice[wrong_idx], "values at right vs wrong index must differ");
}

/// For every cell (r, c) in the logical extent, `idx(r, c)` and
/// `r * cols() + c` agree only for row 0 (no stride gap yet).
/// For rows r >= 1 they disagree.
#[test]
fn padded_slice_footgun_row0_agrees_row1_diverges() {
    let g = BlockedGrid::<u64>::new(100, 100);

    // Row 0: both formulas agree (no stride gap before row 0).
    for c in 0..100 {
        let right = g.idx(0, c);
        let wrong = 0 * g.cols() + c;
        assert_eq!(right, wrong, "row 0, col {}: should agree", c);
    }

    // Row 1: stride gap of 28 padding cells separates them.
    // right = 1 * 128 + c; wrong = 1 * 100 + c → differ by 28.
    for c in 0..100 {
        let right = g.idx(1, c);
        let wrong = 1 * g.cols() + c;
        assert_ne!(right, wrong, "row 1, col {}: should diverge", c);
        assert_eq!(right - wrong, 28, "stride gap should be 28 cells");
    }
}

/// The padded-slice length is padded_rows * padded_cols, NOT rows * cols.
#[test]
fn padded_slice_len_is_padded_not_logical() {
    let g = BlockedGrid::<u64>::new(100, 100);
    let logical_len = g.rows() * g.cols(); // 100 * 100 = 10_000
    let padded_len = g.as_padded_slice().len(); // 128 * 128 = 16_384
    assert_ne!(padded_len, logical_len, "padded len must differ from logical len");
    assert_eq!(padded_len, 128 * 128);
    assert_eq!(logical_len, 100 * 100);
}
