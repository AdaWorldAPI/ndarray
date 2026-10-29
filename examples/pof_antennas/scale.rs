//! A6, one bounded scaling antenna: a 64×64 logical field.
//!
//! The field is the 4096 cells of a [`PermTable12`]; cell `code` is the map
//! `table.for_code(code)`. One question over the whole field:
//!
//! `Σ_cells #{ i : A[i] && B[cell[i]] }`
//!
//! where `A` is a predicate in output coordinates and `B` a predicate that
//! moves with each cell. Four executions, one answer:
//!
//! * per cell, materialized: compose the cell, move `B`'s bytes, count;
//! * per cell, conjugated: compose the cell, carry `B`'s mask through it
//!   (`conjugate_mask`), AND, popcount; no data written;
//! * whole field through the existing exposure ([`PermTable12::exposure`]):
//!   one 64×64 kernel, one weighted count, no per-cell work;
//! * the two per-cell forms again in Morton order of the 64×64 grid.
//!
//! No dense 64³ (or larger) structure is allocated: the largest object is the
//! 64×64 `u16` exposure (8 KiB).

#![allow(dead_code)]

use ndarray::hpc::perm::{Perm64, PermTable12};

/// Bytes written by each execution, so traffic is reported, not guessed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Traffic {
    pub compositions: u64,
    pub bytes_written: u64,
}

/// A table with twelve distinct, non-trivial base steps.
pub fn table(seed: u64) -> PermTable12 {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    PermTable12::new(core::array::from_fn(|_| {
        let mut idx: [u8; 64] = core::array::from_fn(|i| i as u8);
        for i in (1..64).rev() {
            idx.swap(i, (next() % (i as u64 + 1)) as usize);
        }
        Perm64::from_indices(idx).unwrap()
    }))
}

/// The cell order of the 64×64 grid. Row-major is `code = y * 64 + x`;
/// Morton interleaves the bits of `x` and `y`.
pub fn row_major() -> Vec<u16> {
    (0..4096u16).collect()
}

pub fn morton() -> Vec<u16> {
    (0..4096u32)
        .map(|m| {
            let (mut x, mut y) = (0u32, 0u32);
            for b in 0..6 {
                x |= (m >> (2 * b) & 1) << b;
                y |= (m >> (2 * b + 1) & 1) << b;
            }
            (y * 64 + x) as u16
        })
        .collect()
}

pub fn per_cell_materialized(t: &PermTable12, a: &[u8; 64], b: &[u8; 64], order: &[u16]) -> (u64, Traffic) {
    let mut tr = Traffic::default();
    let mut total = 0u64;
    for &code in order {
        let p = t.for_code(code);
        tr.compositions += 1;
        let moved = p.materialize(b);
        tr.bytes_written += 64;
        total += a
            .iter()
            .zip(&moved)
            .filter(|(x, y)| **x != 0 && **y != 0)
            .count() as u64;
    }
    (total, tr)
}

pub fn per_cell_conjugated(t: &PermTable12, a: u64, b: u64, order: &[u16]) -> (u64, Traffic) {
    let mut tr = Traffic::default();
    let mut total = 0u64;
    for &code in order {
        let p = t.for_code(code);
        tr.compositions += 1;
        total += (a & p.conjugate_mask(b)).count_ones() as u64;
    }
    (total, tr)
}

pub fn whole_field_exposure(t: &PermTable12, a: u64, b: u64) -> (u64, Traffic) {
    let e = t.exposure();
    let total = e.count_where(|i, j| a >> i & 1 == 1 && b >> j & 1 == 1);
    // One 64x64 u16 kernel is written once.
    (
        total,
        Traffic {
            compositions: 0,
            bytes_written: 64 * 64 * 2,
        },
    )
}

/// Scalar oracle from the index vectors alone.
pub fn oracle(t: &PermTable12, a: u64, b: u64) -> u64 {
    let mut total = 0u64;
    for code in 0..4096u16 {
        let idx = t.for_code(code).indices();
        for (i, &j) in idx.iter().enumerate() {
            if a >> i & 1 == 1 && b >> j & 1 == 1 {
                total += 1;
            }
        }
    }
    total
}

pub fn bytes_of(mask: u64) -> [u8; 64] {
    core::array::from_fn(|i| (mask >> i & 1) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_execution_agrees_with_the_oracle() {
        let t = table(0x51_7CC1_B727_220A);
        for (a, b) in [
            (0xF0F0_1234_8000_0001u64, 0x0FF0_F00F_5555_AAAAu64),
            (u64::MAX, 0x8000_0000_0000_0001),
            (0x0123_4567_89AB_CDEF, 0xFEDC_BA98_7654_3210),
        ] {
            let want = oracle(&t, a, b);
            // Anti-vacuity: the answer depends on the field, not just popcounts.
            assert!(want > 0 && want < 4096 * 64);
            assert_eq!(per_cell_materialized(&t, &bytes_of(a), &bytes_of(b), &row_major()).0, want);
            assert_eq!(per_cell_conjugated(&t, a, b, &row_major()).0, want);
            assert_eq!(per_cell_conjugated(&t, a, b, &morton()).0, want);
            assert_eq!(whole_field_exposure(&t, a, b).0, want);
        }
    }

    #[test]
    fn morton_visits_every_cell_once() {
        let mut m = morton();
        assert_ne!(m, row_major(), "Morton order is a different traversal");
        m.sort_unstable();
        assert_eq!(m, row_major());
    }

    /// The field answer is not a function of the popcounts alone: two `b`
    /// with equal popcount give different totals, so the fixture exercises
    /// the per-cell routing.
    #[test]
    fn the_answer_depends_on_positions() {
        let t = table(9);
        let a = 0x0000_0000_FFFF_FFFFu64;
        assert_ne!(oracle(&t, a, 0x0000_0000_0000_FFFF), oracle(&t, a, 0xFFFF_0000_0000_0000));
    }
}
