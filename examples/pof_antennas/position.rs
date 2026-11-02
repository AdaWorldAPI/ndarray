//! A1, the position antenna: one 64-position basis, three operations.
//!
//! The 64 logical positions carry four planes: a magnitude byte, a polarity
//! bit (1 = positive), a predicate bit and an operation-selector bit. Three
//! operations read them:
//!
//! 1. predicate evaluation: `count(pred AND positive)`;
//! 2. masked numeric accumulation: `sum(mag | pred, positive) - sum(mag | pred,
//!    negative)`, an exact signed `i64`;
//! 3. programmable XOR/AND selection, per position:
//!    `sel ? (pred XOR positive) : (pred AND positive)`.
//!
//! Three ways to run them:
//!
//! * P0 prepares every plane separately for every operation;
//! * P1 prepares the planes once ([`Prepared`]) and runs the operations on
//!   the shared basis;
//! * P2 re-bases a prepared basis through a [`Perm64`] ([`Prepared::rebase`]).
//!   The masks follow their positions through [`Perm64::conjugate_mask`]; the
//!   magnitudes are not moved, because operations 1 and 2 are invariant under
//!   a consistent permutation and are evaluated in the source basis. Only
//!   operation 3's result is owed in the new basis, and it gets there as a
//!   conjugated mask, not as a data move.
//!
//! A prepared basis records the frame it was prepared from; running it
//! against another frame is refused ([`StaleBasis`]).

#![allow(dead_code)]

use ndarray::hpc::perm::Perm64;
use ndarray::simd::{U64x8, U8x64};

/// One block of 64 positions as raw byte planes, the form P0 starts from.
#[derive(Clone, Copy)]
pub struct RawPlanes {
    pub frame: u64,
    pub mag: [u8; 64],
    /// 1 = positive, 0 = negative.
    pub polarity: [u8; 64],
    /// 1 = predicate holds.
    pub pred: [u8; 64],
    /// 1 = XOR, 0 = AND for operation 3.
    pub sel: [u8; 64],
}

/// The three results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Results {
    pub count: u32,
    pub signed_sum: i64,
    pub select: u64,
}

/// Bit plane of the lanes whose byte is 1.
#[inline(always)]
fn plane(bytes: &[u8; 64]) -> u64 {
    U8x64::from_array(*bytes).cmpeq_mask(U8x64::splat(1))
}

#[inline(always)]
fn op_count(pred: u64, positive: u64) -> u32 {
    (pred & positive).count_ones()
}

#[inline(always)]
fn op_signed_sum(mag: U8x64, pred: u64, positive: u64) -> i64 {
    let zero = U64x8::splat(0);
    let pos = zero.sum8_u8_add_masked(mag, pred & positive).reduce_sum();
    let neg = zero.sum8_u8_add_masked(mag, pred & !positive).reduce_sum();
    pos as i64 - neg as i64
}

#[inline(always)]
fn op_select(pred: u64, positive: u64, sel: u64) -> u64 {
    (sel & (pred ^ positive)) | (!sel & (pred & positive))
}

/// P0: every operation prepares the planes it needs on its own.
#[inline(always)]
pub fn p0_separate(r: &RawPlanes) -> Results {
    let count = op_count(plane(&r.pred), plane(&r.polarity));
    let signed_sum = op_signed_sum(U8x64::from_array(r.mag), plane(&r.pred), plane(&r.polarity));
    let select = op_select(plane(&r.pred), plane(&r.polarity), plane(&r.sel));
    Results {
        count,
        signed_sum,
        select,
    }
}

/// A basis prepared once: the bit planes, the magnitudes, the frame they came
/// from and the coordinate map they are seen through.
#[derive(Clone, Copy)]
pub struct Prepared {
    pub frame: u64,
    /// The map from source positions to this view's positions.
    pub basis: Perm64,
    mag: U8x64,
    positive: u64,
    pred: u64,
    sel: u64,
}

/// A prepared basis was run against a frame it was not prepared from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaleBasis {
    pub prepared: u64,
    pub current: u64,
}

impl Prepared {
    /// P1 preparation: each plane once.
    #[inline(always)]
    pub fn new(r: &RawPlanes) -> Self {
        Self {
            frame: r.frame,
            basis: Perm64::IDENTITY,
            mag: U8x64::from_array(r.mag),
            positive: plane(&r.polarity),
            pred: plane(&r.pred),
            sel: plane(&r.sel),
        }
    }

    /// Run all three operations on this basis.
    #[inline(always)]
    pub fn run(&self, current_frame: u64) -> Result<Results, StaleBasis> {
        if current_frame != self.frame {
            return Err(StaleBasis {
                prepared: self.frame,
                current: current_frame,
            });
        }
        Ok(Results {
            count: op_count(self.pred, self.positive),
            signed_sum: op_signed_sum(self.mag, self.pred, self.positive),
            select: op_select(self.pred, self.positive, self.sel),
        })
    }

    /// P2: the same basis seen through `p`. See [`Rebased`].
    #[inline(always)]
    pub fn rebase(&self, p: Perm64) -> Rebased {
        Rebased {
            source: *self,
            basis: p,
            pred: p.conjugate_mask(self.pred),
            positive: p.conjugate_mask(self.positive),
            sel: p.conjugate_mask(self.sel),
        }
    }

    pub fn mask_pred(&self) -> u64 {
        self.pred
    }
    pub fn mask_positive(&self) -> u64 {
        self.positive
    }
    pub fn mask_sel(&self) -> u64 {
        self.sel
    }

    /// P2-lazy: the same answer as `rebase(p).run(frame)` with ONE mask
    /// conjugation instead of three. Licensed by equivariance alone: select
    /// is lane-local, so `p∘select(pred, pos, sel) == select(p∘pred, p∘pos,
    /// p∘sel)`, and count and the signed sum are invariant, so they need no
    /// conjugation at all. The rebase is deferred to the one compact result
    /// that carries positions.
    #[inline(always)]
    pub fn run_rebased_lazy(&self, p: Perm64, current_frame: u64) -> Result<Results, StaleBasis> {
        let src = self.run(current_frame)?;
        Ok(Results {
            select: p.conjugate_mask(src.select),
            ..src
        })
    }
}

/// The same map as [`Perm64::conjugate_mask`], with the mask expanded to
/// bytes by one masked select instead of a 64-lane scalar loop. Uses only the
/// public map API (`materialize`) for the byte move.
#[inline(always)]
pub fn conjugate_mask_blended(p: &Perm64, mask: u64) -> u64 {
    let flags = U8x64::mask_blend(mask, U8x64::splat(0), U8x64::splat(0x80));
    U8x64::from_array(p.materialize(&flags.to_array())).movemask()
}

/// A prepared basis re-based through a permutation. The masks are carried to
/// the new positions; the magnitudes stay in the source basis.
#[derive(Clone, Copy)]
pub struct Rebased {
    source: Prepared,
    pub basis: Perm64,
    pred: u64,
    positive: u64,
    sel: u64,
}

impl Rebased {
    /// Run the three operations. The invariant ones (count, signed sum) are
    /// evaluated in whichever basis holds consistent planes; the equivariant
    /// one (select) on the conjugated masks, so its result is in the new
    /// basis. No data moves.
    #[inline(always)]
    pub fn run(&self, current_frame: u64) -> Result<Results, StaleBasis> {
        let src = self.source.run(current_frame)?;
        Ok(Results {
            count: op_count(self.pred, self.positive),
            signed_sum: src.signed_sum,
            select: op_select(self.pred, self.positive, self.sel),
        })
    }

    /// The deliberate mistake the falsifier checks for: summing the source
    /// magnitudes under the RE-BASED masks, i.e. mixing two bases.
    pub fn mixed_basis_sum(&self) -> i64 {
        op_signed_sum(self.source.mag, self.pred, self.positive)
    }

    /// The other deliberate mistake: control bits that did not follow their
    /// positions (the selector left in the source basis).
    pub fn stale_selector_select(&self) -> u64 {
        op_select(self.pred, self.positive, self.source.sel)
    }
}

/// Per-position scalar oracle, independent of the planes above.
pub fn oracle(r: &RawPlanes) -> Results {
    let mut count = 0u32;
    let mut signed_sum = 0i64;
    let mut select = 0u64;
    for i in 0..64 {
        let pred = r.pred[i] == 1;
        let pos = r.polarity[i] == 1;
        if pred && pos {
            count += 1;
        }
        if pred {
            let m = r.mag[i] as i64;
            signed_sum += if pos { m } else { -m };
        }
        let bit = if r.sel[i] == 1 { pred ^ pos } else { pred && pos };
        select |= (bit as u64) << i;
    }
    Results {
        count,
        signed_sum,
        select,
    }
}

/// Deterministic fixture: 64 distinct positions (each magnitude is unique),
/// mixed planes.
pub fn fixture(frame: u64, seed: u64) -> RawPlanes {
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut mag = [0u8; 64];
    for (i, m) in mag.iter_mut().enumerate() {
        // Distinct per position, so a position mix-up changes the sum.
        *m = (i as u8).wrapping_mul(3).wrapping_add(17) | 1;
    }
    let bits = |x: u64| -> [u8; 64] { core::array::from_fn(|i| (x >> i & 1) as u8) };
    RawPlanes {
        frame,
        mag,
        polarity: bits(next()),
        pred: bits(next()),
        sel: bits(next()),
    }
}

/// The same block with every plane permuted by `p` (the reference for P2).
pub fn permute_raw(r: &RawPlanes, p: Perm64) -> RawPlanes {
    RawPlanes {
        frame: r.frame,
        mag: p.materialize(&r.mag),
        polarity: p.materialize(&r.polarity),
        pred: p.materialize(&r.pred),
        sel: p.materialize(&r.sel),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perm(seed: u64) -> Perm64 {
        let mut idx: [u8; 64] = core::array::from_fn(|i| i as u8);
        let mut s = seed | 1;
        for i in (1..64).rev() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            idx.swap(i, (s % (i as u64 + 1)) as usize);
        }
        Perm64::from_indices(idx).unwrap()
    }

    #[test]
    fn p0_p1_and_the_oracle_agree() {
        for seed in 1..50u64 {
            let r = fixture(3, seed * 0x9E37_79B9);
            let want = oracle(&r);
            // Anti-vacuity: every operation sees a non-trivial selection.
            assert!(want.count > 3 && want.count < 40, "seed {seed}: {want:?}");
            assert_eq!(p0_separate(&r), want);
            assert_eq!(Prepared::new(&r).run(3), Ok(want));
        }
    }

    /// Re-basing keeps the meaning: invariant results unchanged, the
    /// equivariant result carried to the new positions, equal to running the
    /// oracle on the physically permuted block.
    #[test]
    fn rebase_preserves_meaning() {
        let r = fixture(3, 0xDEAD_BEEF);
        let base = Prepared::new(&r).run(3).unwrap();
        let mut moved_any = false;
        for k in 1..12 {
            let p = perm(k);
            let got = Prepared::new(&r).rebase(p).run(3).unwrap();
            let truth = oracle(&permute_raw(&r, p));
            assert_eq!(got, truth, "perm {k}");
            assert_eq!((got.count, got.signed_sum), (base.count, base.signed_sum));
            assert_eq!(got.select, p.conjugate_mask(base.select));
            moved_any |= got.select != base.select;
        }
        assert!(moved_any, "no permutation moved the select result: vacuous");
    }

    /// Falsifier: mixing bases (re-based masks over source magnitudes) is
    /// detected; position i of one basis is not position i of another.
    #[test]
    fn mixing_bases_changes_the_result() {
        let r = fixture(3, 0x1234_5678);
        let truth = oracle(&r).signed_sum;
        let wrong = (1..12)
            .map(|k| Prepared::new(&r).rebase(perm(k)).mixed_basis_sum())
            .filter(|&s| s != truth)
            .count();
        assert!(wrong >= 10, "only {wrong} of 11 permutations exposed the mix-up");
    }

    /// Falsifier: control bits that do not follow their positions give a
    /// different select result.
    #[test]
    fn selector_must_follow_its_positions() {
        let r = fixture(3, 0xABCD_0001);
        let wrong = (1..12)
            .filter(|&k| {
                let rb = Prepared::new(&r).rebase(perm(k));
                rb.stale_selector_select() != rb.run(3).unwrap().select
            })
            .count();
        assert!(wrong >= 10, "only {wrong} of 11 permutations exposed a stale selector");
    }

    /// Falsifier: a basis prepared from one frame is refused on another.
    #[test]
    fn a_stale_basis_is_refused() {
        let r = fixture(3, 7);
        let prep = Prepared::new(&r);
        assert_eq!(
            prep.run(4),
            Err(StaleBasis {
                prepared: 3,
                current: 4
            })
        );
        assert_eq!(
            prep.rebase(perm(5)).run(4).map(|_| ()),
            Err(StaleBasis {
                prepared: 3,
                current: 4
            })
        );
        assert!(prep.run(3).is_ok());
    }

    /// Two domains in different bases: combining them needs the relative
    /// map, not the raw slot numbers.
    #[test]
    fn two_domains_meet_through_the_relative_map() {
        let r = fixture(3, 99);
        let (p, q) = (perm(21), perm(22));
        // Domain X: predicate seen through p. Domain Y: polarity seen through q.
        let x = p.conjugate_mask(plane(&r.pred));
        let y = q.conjugate_mask(plane(&r.polarity));
        // Correct: carry y into p's basis with q.relative_to(p)... expressed
        // on masks: back to source with q's inverse, then into p.
        let y_in_p = p.conjugate_mask(q.inverse().conjugate_mask(y));
        let truth = p.conjugate_mask(plane(&r.pred) & plane(&r.polarity));
        assert_eq!(x & y_in_p, truth);
        // Naive: slot i of y treated as slot i of x.
        assert_ne!(x & y, truth);
    }

    /// The lazy rebase (one conjugation of the result) equals the eager one
    /// (three conjugations of the inputs) for every map tried; and the
    /// blended expansion equals `conjugate_mask` bit for bit.
    #[test]
    fn lazy_rebase_and_blended_conjugation_agree() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for k in 1..40 {
            let r = fixture(5, k);
            let prep = Prepared::new(&r);
            let p = perm(k + 100);
            assert_eq!(prep.run_rebased_lazy(p, 5).unwrap(), prep.rebase(p).run(5).unwrap());
            assert_eq!(
                prep.run_rebased_lazy(p, 6),
                Err(StaleBasis {
                    prepared: 5,
                    current: 6
                })
            );
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            for m in [s, !s, 0, u64::MAX, 1 << (k % 64)] {
                assert_eq!(conjugate_mask_blended(&p, m), p.conjugate_mask(m));
            }
        }
    }

    /// Negative: lazy rebase is only licensed for the EQUIVARIANT result.
    /// Applying it to an order-dependent result (a prefix count of the
    /// selected positions) gives a wrong answer.
    #[test]
    fn lazy_rebase_is_refused_for_a_positional_result() {
        let r = fixture(5, 7);
        let prep = Prepared::new(&r);
        let src_sel = prep.run(5).unwrap().select;
        // Positional result: mask of positions whose prefix popcount is odd.
        let prefix_odd = |m: u64| {
            (0..64).fold(0u64, |acc, i| acc | (u64::from((m & (u64::MAX >> (63 - i))).count_ones() % 2 == 1) << i))
        };
        let mut differs = false;
        for k in 1..12 {
            let p = perm(k);
            let truth = prefix_odd(prep.rebase(p).run(5).unwrap().select);
            let lazy = p.conjugate_mask(prefix_odd(src_sel));
            differs |= truth != lazy;
        }
        assert!(differs, "a positional result must not commute with the rebase");
    }
}
