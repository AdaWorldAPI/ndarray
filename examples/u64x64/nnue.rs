//! D-U64X64-NNUE-1 lab route G: an NNUE-style weighted accumulator over a
//! bound 64x64 Boolean tile.
//!
//! The tile is presence only: word `k` bit `r` set means cell `(k, r)` is in
//! the relation. A bit carries no sign and no count. Each cell owns a weight
//! row `w[k][r]` of `H = 16` signed `i16`s; the accumulator is `H` `i32`s:
//! `h = bias + Σ_{(k, r) set} w[k][r]`.
//!
//! Exact by bound, not by wrapping: [`CellWeights::new`] refuses `|bias| > 2^15`
//! and any weight equal to `i16::MIN`, so `|h| <= 2^15 + 4096 · (2^15 - 1)`,
//! which is below `2^28`.
//!
//! - [`refresh`] walks set bits with `trailing_zeros`; its work scales with
//!   the set bits, not with 4,096.
//! - [`update`] takes `old ^ new`, adds the rows that turned on and subtracts
//!   the rows that turned off; its work scales with the changed bits. It is
//!   correct only when `acc` is the accumulator of `old`.
//! - `refresh_simd` / `update_simd` do the same with
//!   `I32x16::from_i16_slice` and `Add` / `Sub`.
//! - [`refresh_bound`] / [`update_bound`] read `Source::present` through a
//!   `fixture::Bound`. An update refuses two frames of different sources,
//!   shapes or row orders; the versions may differ, because an update is
//!   between versions.
//!
//! A Boolean tile is a set. It cannot hold two equal `(k, r)` edges, so a bag
//! of edges whose weights must count with multiplicity needs a count plane,
//! not this tile.
//!
//! Lab fixture only: no public type, no production caller.

use crate::fixture::{BitTile, Bound, Refusal, Witness, F_PRESENT};
use ndarray::simd::I32x16;

/// Accumulator width (one `I32x16`).
pub const H: usize = 16;
/// The accumulator: `H` exact `i32` sums.
pub type Acc = [i32; H];
/// Largest admitted `|bias|`.
pub const BIAS_MAX: u32 = 1 << 15;

/// One weight row per cell, plus a bias, admitted against the overflow bound.
pub struct CellWeights {
    bias: Acc,
    w: Box<[[[i16; H]; 64]]>,
}

impl CellWeights {
    /// `w[k][r]` is the row of cell `(k, r)`. Refuses a table that is not
    /// 64 x 64 (`Shape`), a bias past `2^15` or a weight equal to `i16::MIN`
    /// (`TooLarge`).
    pub fn new(bias: Acc, w: Box<[[[i16; H]; 64]]>) -> Result<Self, Refusal> {
        if w.len() != 64 {
            return Err(Refusal::Shape);
        }
        if bias.iter().any(|b| b.unsigned_abs() > BIAS_MAX) {
            return Err(Refusal::TooLarge);
        }
        if w.iter().flatten().flatten().any(|&x| x == i16::MIN) {
            return Err(Refusal::TooLarge);
        }
        Ok(CellWeights { bias, w })
    }

    pub fn bias(&self) -> Acc {
        self.bias
    }

    pub fn row(&self, k: usize, r: usize) -> &[i16; H] {
        &self.w[k][r]
    }
}

#[inline(always)]
fn add_row(acc: &mut Acc, row: &[i16; H]) {
    for j in 0..H {
        acc[j] += i32::from(row[j]);
    }
}

#[inline(always)]
fn sub_row(acc: &mut Acc, row: &[i16; H]) {
    for j in 0..H {
        acc[j] -= i32::from(row[j]);
    }
}

/// `bias + Σ` of the rows of every set cell, scalar adds.
#[no_mangle]
#[inline(never)]
pub fn nnue_refresh(w: &CellWeights, tile: &BitTile) -> Acc {
    let mut acc = w.bias;
    for (k, &word) in tile.iter().enumerate() {
        let mut m = word;
        while m != 0 {
            let r = m.trailing_zeros() as usize;
            add_row(&mut acc, &w.w[k][r]);
            m &= m - 1;
        }
    }
    acc
}

/// Move `acc` from the accumulator of `old` to that of `new`, scalar adds.
#[no_mangle]
#[inline(never)]
pub fn nnue_update(w: &CellWeights, acc: &mut Acc, old: &BitTile, new: &BitTile) {
    for k in 0..64 {
        let d = old[k] ^ new[k];
        if d == 0 {
            continue;
        }
        let mut on = d & new[k];
        while on != 0 {
            add_row(acc, &w.w[k][on.trailing_zeros() as usize]);
            on &= on - 1;
        }
        let mut off = d & old[k];
        while off != 0 {
            sub_row(acc, &w.w[k][off.trailing_zeros() as usize]);
            off &= off - 1;
        }
    }
}

/// [`nnue_refresh`] with one `I32x16` accumulator: each row is widened from
/// 16 `i16` with `from_i16_slice` and added.
#[no_mangle]
#[inline(never)]
pub fn nnue_refresh_simd(w: &CellWeights, tile: &BitTile) -> Acc {
    let mut acc = I32x16::from_array(w.bias);
    for (k, &word) in tile.iter().enumerate() {
        let mut m = word;
        while m != 0 {
            acc += I32x16::from_i16_slice(&w.w[k][m.trailing_zeros() as usize]);
            m &= m - 1;
        }
    }
    acc.to_array()
}

/// [`nnue_refresh_simd`] with two independent `I32x16` accumulators:
/// consecutive set bits of a word go to alternate accumulators, so two adds
/// are in flight instead of one dependent chain. The two are summed at the
/// end; the result is identical because integer addition is associative and
/// cannot overflow under the admitted bound.
#[no_mangle]
#[inline(never)]
pub fn nnue_refresh_simd2(w: &CellWeights, tile: &BitTile) -> Acc {
    let mut a0 = I32x16::from_array(w.bias);
    let mut a1 = I32x16::splat(0);
    for (k, &word) in tile.iter().enumerate() {
        let mut m = word;
        while m != 0 {
            a0 += I32x16::from_i16_slice(&w.w[k][m.trailing_zeros() as usize]);
            m &= m - 1;
            if m == 0 {
                break;
            }
            a1 += I32x16::from_i16_slice(&w.w[k][m.trailing_zeros() as usize]);
            m &= m - 1;
        }
    }
    (a0 + a1).to_array()
}

/// [`nnue_refresh_simd2`] with four accumulators, bits taken round-robin.
#[no_mangle]
#[inline(never)]
pub fn nnue_refresh_simd4(w: &CellWeights, tile: &BitTile) -> Acc {
    let mut acc = [I32x16::from_array(w.bias), I32x16::splat(0), I32x16::splat(0), I32x16::splat(0)];
    for (k, &word) in tile.iter().enumerate() {
        let mut m = word;
        let mut lane = 0usize;
        while m != 0 {
            acc[lane & 3] += I32x16::from_i16_slice(&w.w[k][m.trailing_zeros() as usize]);
            m &= m - 1;
            lane += 1;
        }
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])).to_array()
}

/// [`nnue_update`] with one `I32x16` accumulator.
#[no_mangle]
#[inline(never)]
pub fn nnue_update_simd(w: &CellWeights, acc: &mut Acc, old: &BitTile, new: &BitTile) {
    let mut a = I32x16::from_array(*acc);
    for k in 0..64 {
        let d = old[k] ^ new[k];
        if d == 0 {
            continue;
        }
        let mut on = d & new[k];
        while on != 0 {
            a += I32x16::from_i16_slice(&w.w[k][on.trailing_zeros() as usize]);
            on &= on - 1;
        }
        let mut off = d & old[k];
        while off != 0 {
            a -= I32x16::from_i16_slice(&w.w[k][off.trailing_zeros() as usize]);
            off &= off - 1;
        }
    }
    *acc = a.to_array();
}

/// The accumulator of a bound source's `present` tile. Refuses a frame
/// without the presence field.
pub fn refresh_bound(w: &CellWeights, b: &Bound<'_>) -> Result<Acc, Refusal> {
    if b.witness.fields & F_PRESENT == 0 {
        return Err(Refusal::Field);
    }
    Ok(nnue_refresh_simd(w, &b.src.present))
}

/// Two frames describe the same population: one source, one shape, one row
/// order. Versions are not compared.
fn same_population(a: &Witness, b: &Witness) -> Result<(), Refusal> {
    if a.source_id != b.source_id {
        return Err(Refusal::ForeignSource);
    }
    if a.rows != b.rows || a.cols != b.cols {
        return Err(Refusal::Shape);
    }
    if a.row_order_hash != b.row_order_hash || a.row_order != b.row_order {
        return Err(Refusal::RowOrder);
    }
    Ok(())
}

/// Move `acc` from `old`'s accumulator to `new`'s. Refuses frames of
/// different populations or without the presence field; on refusal `acc` is
/// untouched.
pub fn update_bound(w: &CellWeights, acc: &mut Acc, old: &Bound<'_>, new: &Bound<'_>) -> Result<(), Refusal> {
    same_population(&old.witness, &new.witness)?;
    if old.witness.fields & new.witness.fields & F_PRESENT == 0 {
        return Err(Refusal::Field);
    }
    nnue_update_simd(w, acc, &old.src.present, &new.src.present);
    Ok(())
}
