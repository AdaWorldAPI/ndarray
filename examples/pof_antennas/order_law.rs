//! Order antenna (A3) and law antenna (A4) for the permutation-of-fields probe.
//!
//! Three orders get mixed up whenever a lane-wise operation is reordered:
//!
//! - **traversal order**: the sequence in which lanes are visited. A fold over
//!   the multiset of lane values (count, sum) does not care.
//! - **coordinate order**: the basis ([`Perm64`]) through which a block is
//!   viewed. Invariant folds and equivariant lane-wise ops do not care;
//!   positional ops (prefix scan, index-weighted sum) do.
//! - **effect order**: the order in which writes land on one cell. This is a
//!   partial order: only non-commuting pairs must keep their relative order.
//!
//! The A3 half models each of the three and refuses illegal reorders
//! ([`check_reorder`]). The A4 half records the algebraic laws of each
//! operation as plain data ([`LAW_TABLE`]) and shows which laws admit which
//! optimization (skip a permutation, fuse two ops, defer a reduction, carry a
//! result in its basis). Nothing here restates [`PermInvariant`] or
//! [`PermEquivariant`]; the law table is a record, the traits stay the gate.
//!
//! Plain safe Rust. All lane movement goes through [`Perm64`]'s own methods.

#![allow(dead_code)]

use ndarray::hpc::perm::{
    combine_in_basis, CountNonzero, InBasis, LaneFold, LaneOp, Perm64, PermEquivariant, PermInvariant, SumLanes, LANES,
};

// ---------------------------------------------------------------------------
// A3 — order antenna
// ---------------------------------------------------------------------------

/// One write on a single `i64` cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    /// Replace the cell with the value.
    Set(i64),
    /// Add the value to the cell.
    Add(i64),
    /// Multiply the cell by the value.
    Mul(i64),
}

impl Effect {
    /// Apply to one cell with checked arithmetic; `None` on overflow.
    pub fn apply(&self, cell: i64) -> Option<i64> {
        match *self {
            Effect::Set(v) => Some(v),
            Effect::Add(v) => cell.checked_add(v),
            Effect::Mul(v) => cell.checked_mul(v),
        }
    }
}

/// Run `effs` in order on a cell starting at `init`.
///
/// Returns `None` if any step overflows `i64`. Arithmetic is checked, never
/// wrapping, so an overflow is reported instead of being folded into a result.
pub fn run_effects(init: i64, effs: &[Effect]) -> Option<i64> {
    effs.iter().try_fold(init, |cell, e| e.apply(cell))
}

/// Whether two effects can be swapped without changing the result of
/// [`run_effects`] for ANY starting value, overflow included: a swap must not
/// turn a reported overflow (`None`) into a value or back.
///
/// Under checked arithmetic, exact-integer commutativity is not enough:
///
/// * `Add(a)`/`Add(b)` commute iff `a` and `b` do not have opposite signs.
///   With both `>= 0` (or both `<= 0`) the intermediate lies between the
///   start and the final value, so it overflows only if the final value does.
///   With opposite signs it does not: from `i64::MAX`, `Add(1), Add(-1)`
///   overflows while `Add(-1), Add(1)` returns `i64::MAX`.
/// * `Mul(a)`/`Mul(b)` commute iff `a == b`, either factor is `1`, or both
///   have magnitude `>= 2`. In the last case the intermediate has at most half
///   the final magnitude, so it fits whenever the final value does. A `0` or
///   `-1` factor breaks it: from `i64::MIN`, `Mul(-1), Mul(0)` overflows while
///   `Mul(0), Mul(-1)` returns `0`.
/// * `Set(x)`/`Set(y)` commute iff `x == y`.
/// * Every mixed pair (`Set`/`Add`, `Set`/`Mul`, `Add`/`Mul`) is
///   order-sensitive.
pub fn commutes(a: &Effect, b: &Effect) -> bool {
    match (*a, *b) {
        (Effect::Add(x), Effect::Add(y)) => !((x < 0 && y > 0) || (x > 0 && y < 0)),
        (Effect::Mul(x), Effect::Mul(y)) => {
            x == y || x == 1 || y == 1 || (x.unsigned_abs() >= 2 && y.unsigned_abs() >= 2)
        }
        (Effect::Set(x), Effect::Set(y)) => x == y,
        _ => false,
    }
}

/// A reorder was refused: effects `first` and `second` (original indices,
/// `first < second`) do not commute but `order` runs `second` before `first`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderRefusal {
    /// Original index of the effect that must stay first.
    pub first: usize,
    /// Original index of the effect that must stay second.
    pub second: usize,
}

/// Check a proposed execution order against the effect partial order.
///
/// `order` is a permutation of `0..effs.len()` giving the new execution order
/// (`order[k]` is the original index run `k`-th). It is legal iff every pair
/// `i < j` with `!commutes(effs[i], effs[j])` keeps `i` before `j`. The first
/// violating pair (lowest `i`, then lowest `j`) is reported.
///
/// # Panics
///
/// If `order` is not a permutation of `0..effs.len()`.
pub fn check_reorder(effs: &[Effect], order: &[usize]) -> Result<(), OrderRefusal> {
    let n = effs.len();
    assert_eq!(order.len(), n, "order must have one entry per effect");
    let mut pos = vec![usize::MAX; n];
    for (k, &orig) in order.iter().enumerate() {
        assert!(orig < n, "order entry {orig} is out of range 0..{n}");
        assert_eq!(pos[orig], usize::MAX, "order is not a permutation: {orig} appears twice");
        pos[orig] = k;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if !commutes(&effs[i], &effs[j]) && pos[i] > pos[j] {
                return Err(OrderRefusal { first: i, second: j });
            }
        }
    }
    Ok(())
}

/// An immutable 64-lane view with a version. Writers return a new snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Snapshot {
    /// Monotone version; a write produces `version + 1`.
    pub version: u64,
    /// The lane values.
    pub lanes: [u8; 64],
}

/// Reader 1: number of nonzero lanes (via [`CountNonzero`]).
pub fn read_count(s: &Snapshot) -> u32 {
    CountNonzero.fold(&s.lanes)
}

/// Reader 2: sum of lane values (via [`SumLanes`]).
pub fn read_sum(s: &Snapshot) -> u32 {
    SumLanes.fold(&s.lanes)
}

/// Write one lane into a NEW snapshot with `version + 1`; `s` is unchanged.
///
/// # Panics
///
/// If `lane >= 64`.
pub fn write_lane(s: &Snapshot, lane: usize, v: u8) -> Snapshot {
    assert!(lane < LANES, "lane {lane} out of range");
    let mut lanes = s.lanes;
    lanes[lane] = v;
    Snapshot {
        version: s.version + 1,
        lanes,
    }
}

/// Inclusive prefix sum over lanes: `out[i] = x[0] + ... + x[i]`.
///
/// The sum wraps mod 256 deliberately: the result is itself a lane value.
/// Output lane `i` reads lanes `0..=i`, so this is NOT [`PermEquivariant`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefixSum;

impl LaneOp<1> for PrefixSum {
    fn apply(&self, [x]: [&[u8; LANES]; 1]) -> [u8; LANES] {
        let mut out = [0u8; LANES];
        let mut acc = 0u8;
        for (o, &v) in out.iter_mut().zip(x.iter()) {
            acc = acc.wrapping_add(v);
            *o = acc;
        }
        out
    }
}

/// Positional fold: `sum of i * lane[i]`. Reads the lane index, so it is NOT
/// [`PermInvariant`].
#[derive(Clone, Copy, Debug, Default)]
pub struct IndexWeightedSum;

impl LaneFold for IndexWeightedSum {
    type Out = u64;
    fn fold(&self, b: &[u8; LANES]) -> u64 {
        b.iter()
            .enumerate()
            .map(|(i, &v)| i as u64 * v as u64)
            .sum()
    }
}

/// A small program separating the three orders.
#[derive(Clone, Debug)]
pub struct Program {
    /// Coordinate basis through which both operand blocks are viewed.
    pub coord_basis: Perm64,
    /// Effects run on the cell before the computed count is added.
    pub effects: Vec<Effect>,
}

/// Count lanes where `(a & b) != 0`, visiting positions in `visit` order,
/// after viewing both blocks through `p.coord_basis`; then feed the count as
/// `Add(count)` appended after `p.effects` into [`run_effects`] from 0.
///
/// Traversal order (`visit`, any permutation of `0..64`) and coordinate basis
/// do not change the count: both operands share the basis, and a count over
/// the multiset of lanes ignores visit order. Effect order is a separate
/// partial order, checked by [`check_reorder`].
///
/// # Panics
///
/// If `visit` is not a permutation of `0..64`.
pub fn run_lanewise_then_effects(p: &Program, a: &[u8; 64], b: &[u8; 64], visit: &[usize; 64]) -> (u32, Option<i64>) {
    let mut seen = 0u64;
    for &i in visit {
        assert!(i < LANES, "visit entry {i} out of range");
        assert_eq!(seen >> i & 1, 0, "visit is not a permutation: {i} twice");
        seen |= 1u64 << i;
    }
    let am = p.coord_basis.materialize(a);
    let bm = p.coord_basis.materialize(b);
    let mut count = 0u32;
    for &i in visit {
        if am[i] & bm[i] != 0 {
            count += 1;
        }
    }
    let mut effs = p.effects.clone();
    effs.push(Effect::Add(count as i64));
    (count, run_effects(0, &effs))
}

/// Combine two blocks, each viewed through its own basis, with a 2-ary
/// equivariant op. Thin wrapper over [`combine_in_basis`].
pub fn combine_two<Op: PermEquivariant<2>>(op: &Op, pa: Perm64, a: &[u8; 64], pb: Perm64, b: &[u8; 64]) -> InBasis {
    combine_in_basis(op, [(pa, a), (pb, b)])
}

// ---------------------------------------------------------------------------
// A4 — law antenna
// ---------------------------------------------------------------------------

/// How an operation behaves at the edge of its integer range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overflow {
    /// Cannot overflow for any admitted input.
    None,
    /// Wraps modulo the lane width, deliberately.
    Wraps,
    /// Exact, but only within the stated bound.
    ExactWithinBound(&'static str),
}

/// Algebraic metadata of one operation: a DATA record, not a trait.
#[derive(Clone, Copy, Debug)]
pub struct Laws {
    /// Row name (unique).
    pub name: &'static str,
    /// Input/output shape in words.
    pub shape: &'static str,
    /// Output lane `i` depends only on input lane `i`.
    pub lane_local: bool,
    /// `(x op y) op z == x op (y op z)` over the combining step.
    pub associative: bool,
    /// `x op y == y op x` over the combining step.
    pub commutative: bool,
    /// Result unchanged when the input is permuted (a reduction).
    pub perm_invariant: bool,
    /// `op(P x) == P op(x)` for every permutation.
    pub perm_equivariant: bool,
    /// Writes whose order matters (a partial order on effects).
    pub ordered_effects: bool,
    /// Result is exact (no rounding, no silent loss).
    pub exact: bool,
    /// Range behaviour.
    pub overflow: Overflow,
    /// Must read from one immutable snapshot.
    pub needs_snapshot: bool,
    /// Needs a coordinate frame (mask or basis) carried alongside the data.
    pub needs_frame: bool,
}

/// One row per operation in the probe. Every field is stated; the tests below
/// check the ones that admit an optimization (skip a permutation, fuse ops,
/// defer a reduction, carry a result in its basis, refuse a reorder).
pub const LAW_TABLE: &[Laws] = &[
    Laws {
        name: "count(A AND B)",
        shape: "64 u8 x 64 u8 -> u32",
        lane_local: false,
        associative: true,
        commutative: true,
        perm_invariant: true,
        perm_equivariant: false,
        ordered_effects: false,
        exact: true,
        overflow: Overflow::None,
        needs_snapshot: false,
        needs_frame: false,
    },
    Laws {
        name: "sum of lanes",
        shape: "64 u8 -> u32",
        lane_local: false,
        associative: true,
        commutative: true,
        perm_invariant: true,
        perm_equivariant: false,
        ordered_effects: false,
        exact: true,
        overflow: Overflow::ExactWithinBound("u32: 64 lanes x 255 per block"),
        needs_snapshot: false,
        needs_frame: false,
    },
    Laws {
        name: "masked signed U9 dot (B kernel, i32 lanes flushed every 16384 blocks)",
        shape: "64 u8 x 64 i8 -> 16 i32 -> i64",
        lane_local: false,
        associative: true,
        commutative: true,
        perm_invariant: true,
        perm_equivariant: false,
        ordered_effects: false,
        exact: true,
        overflow: Overflow::ExactWithinBound("i32 lanes: 16384 blocks x 130560 < i32::MAX"),
        needs_snapshot: false,
        needs_frame: true,
    },
    Laws {
        name: "E5 anchor masked byte sum (u64 lanes)",
        shape: "64 u8 + u64 mask -> u64",
        lane_local: false,
        associative: true,
        commutative: true,
        perm_invariant: true,
        perm_equivariant: false,
        ordered_effects: false,
        exact: true,
        overflow: Overflow::ExactWithinBound("u64 lanes: 255 per byte, no wrap below 2^56 bytes"),
        needs_snapshot: false,
        needs_frame: true,
    },
    Laws {
        name: "programmable XOR/AND select (ternlog)",
        shape: "3 x 64 u8 -> 64 u8",
        lane_local: true,
        associative: false,
        commutative: false,
        perm_invariant: false,
        perm_equivariant: true,
        ordered_effects: false,
        exact: true,
        overflow: Overflow::None,
        needs_snapshot: false,
        needs_frame: false,
    },
    Laws {
        name: "lane-wise XOR",
        shape: "2 x 64 u8 -> 64 u8",
        lane_local: true,
        associative: true,
        commutative: true,
        perm_invariant: false,
        perm_equivariant: true,
        ordered_effects: false,
        exact: true,
        overflow: Overflow::None,
        needs_snapshot: false,
        needs_frame: false,
    },
    Laws {
        name: "index-weighted sum",
        shape: "64 u8 -> u64",
        lane_local: false,
        associative: true,
        commutative: true,
        perm_invariant: false,
        perm_equivariant: false,
        ordered_effects: false,
        exact: true,
        overflow: Overflow::ExactWithinBound("u64: 63 x 255 x 64 per block"),
        needs_snapshot: false,
        needs_frame: true,
    },
    Laws {
        name: "prefix sum",
        shape: "64 u8 -> 64 u8",
        lane_local: false,
        associative: false,
        commutative: false,
        perm_invariant: false,
        perm_equivariant: false,
        ordered_effects: false,
        exact: false,
        overflow: Overflow::Wraps,
        needs_snapshot: false,
        needs_frame: true,
    },
    Laws {
        name: "Set/Add/Mul effects on one cell",
        shape: "i64 cell x effect list -> Option<i64>",
        lane_local: false,
        associative: false,
        commutative: false,
        perm_invariant: false,
        perm_equivariant: false,
        ordered_effects: true,
        exact: true,
        overflow: Overflow::ExactWithinBound("i64: checked ops, None on overflow"),
        needs_snapshot: true,
        needs_frame: false,
    },
    Laws {
        name: "predicate mask conjugated by Perm64",
        shape: "u64 lane mask x Perm64 -> u64 lane mask",
        lane_local: false,
        associative: false,
        commutative: false,
        perm_invariant: false,
        perm_equivariant: false,
        ordered_effects: false,
        exact: true,
        overflow: Overflow::None,
        needs_snapshot: false,
        needs_frame: true,
    },
];

/// Reorder a fold: skip the permutation entirely.
///
/// Relies on the [`PermInvariant`] law, `fold(P·x) == fold(x)`. The generic
/// bound is the admission: a positional fold such as [`IndexWeightedSum`]
/// does not implement the marker and cannot be passed here.
pub fn fold_skipping_perm<F: PermInvariant>(f: &F, _p: Perm64, block: &[u8; 64]) -> F::Out {
    f.fold(block)
}

/// Fuse `(a ^ b) & c` into one ternlog immediate.
///
/// Relies on the lane-local, per-bit nature of XOR and AND: the fused result
/// is a function of the three input bits alone, so it is a truth table. The
/// immediate convention matches [`LaneTernlog`]: output bit = bit
/// `(a << 2) | (b << 1) | c` of the immediate.
pub fn fuse_xor_then_and_imm() -> u8 {
    let mut imm = 0u8;
    for a in 0..2u8 {
        for b in 0..2u8 {
            for c in 0..2u8 {
                let sel = (a << 2) | (b << 1) | c;
                let out = (a ^ b) & c;
                imm |= out << sel;
            }
        }
    }
    imm
}

/// Defer a reduction: accumulate in one `i32` (wrapping) and flush to `i64`
/// every `flush_every` items.
///
/// Relies on associativity and commutativity of addition plus a bound: exact
/// only while `flush_every * max|product| <= i32::MAX`. Past the bound the
/// `i32` wraps silently.
///
/// # Panics
///
/// If `flush_every == 0`.
pub fn deferred_i32_total(products: &[i32], flush_every: usize) -> i64 {
    assert!(flush_every > 0, "flush_every must be positive");
    let mut total = 0i64;
    let mut acc = 0i32;
    let mut n = 0usize;
    for &p in products {
        acc = acc.wrapping_add(p);
        n += 1;
        if n == flush_every {
            total += acc as i64;
            acc = 0;
            n = 0;
        }
    }
    total + acc as i64
}

/// Reference total: every product widened to `i64` first. Always exact.
pub fn exact_i64_total(products: &[i32]) -> i64 {
    products.iter().map(|&p| p as i64).sum()
}

// ---------------------------------------------------------------------------
// Deterministic test data
// ---------------------------------------------------------------------------

/// SplitMix64: tiny deterministic generator for fixtures.
#[derive(Clone, Debug)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A block where about three lanes in four are random bytes, the rest 0.
    pub fn block(&mut self) -> [u8; 64] {
        core::array::from_fn(|_| {
            if !self.next_u64().is_multiple_of(4) {
                (self.next_u64() & 0xff) as u8
            } else {
                0
            }
        })
    }

    /// A random permutation of `0..64` as `usize` (Fisher-Yates).
    pub fn visit_order(&mut self) -> [usize; 64] {
        let mut v: [usize; 64] = core::array::from_fn(|i| i);
        for i in (1..64).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
        v
    }

    /// A random [`Perm64`].
    pub fn perm(&mut self) -> Perm64 {
        let v = self.visit_order();
        let idx: [u8; 64] = core::array::from_fn(|i| v[i] as u8);
        Perm64::from_indices(idx).expect("shuffled indices form a permutation")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::hpc::perm::{LaneAnd, LaneTernlog, LaneXor};

    fn bitand(a: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
        LaneAnd.apply([a, b])
    }

    // ----- A3 -----

    #[test]
    fn count_and_is_permutation_invariant() {
        let mut rng = SplitMix64(1);
        let a = rng.block();
        let b = rng.block();
        let base = bitand(&a, &b);
        let nz = CountNonzero.fold(&base);
        assert!((10..=54).contains(&nz), "fixture must be non-trivial, got {nz} nonzero lanes");
        for _ in 0..8 {
            let p = rng.perm();
            let moved = bitand(&p.materialize(&a), &p.materialize(&b));
            assert_eq!(CountNonzero.fold(&moved), nz);
            assert_eq!(CountNonzero.fold(&p.materialize(&base)), nz);
        }
    }

    #[test]
    fn xor_is_permutation_equivariant() {
        let mut rng = SplitMix64(2);
        let a = rng.block();
        let b = rng.block();
        for _ in 0..6 {
            let p = rng.perm();
            let got = combine_two(&LaneXor, p, &a, p, &b).materialize();
            assert_eq!(got, p.materialize(&LaneXor.apply([&a, &b])));
            let q = rng.perm();
            let mixed = combine_two(&LaneXor, p, &a, q, &b).materialize();
            let (pa, qb) = (p.materialize(&a), q.materialize(&b));
            let expected: [u8; 64] = core::array::from_fn(|i| pa[i] ^ qb[i]);
            assert_eq!(mixed, expected);
        }
    }

    #[test]
    fn index_weighted_sum_is_not_invariant() {
        let mut rng = SplitMix64(3);
        let x = rng.block();
        let base = IndexWeightedSum.fold(&x);
        assert_eq!(IndexWeightedSum.fold(&Perm64::IDENTITY.materialize(&x)), base);
        let perms: Vec<Perm64> = (0..8).map(|_| rng.perm()).collect();
        assert!(
            perms
                .iter()
                .any(|p| IndexWeightedSum.fold(&p.materialize(&x)) != base),
            "no random permutation changed the positional fold"
        );
    }

    #[test]
    fn prefix_sum_is_not_equivariant() {
        let mut rng = SplitMix64(4);
        let x = rng.block();
        let id = Perm64::IDENTITY;
        assert_eq!(id.materialize(&PrefixSum.apply([&x])), PrefixSum.apply([&id.materialize(&x)]));
        let perms: Vec<Perm64> = (0..8).map(|_| rng.perm()).collect();
        assert!(
            perms
                .iter()
                .any(|p| p.materialize(&PrefixSum.apply([&x])) != PrefixSum.apply([&p.materialize(&x)])),
            "no random permutation separated P*prefix(x) from prefix(P*x)"
        );
    }

    #[test]
    fn set_then_add_differs_from_add_then_set() {
        let fwd = [Effect::Set(1), Effect::Add(1)];
        let rev = [Effect::Add(1), Effect::Set(1)];
        assert_eq!(run_effects(0, &fwd), Some(2));
        assert_eq!(run_effects(0, &rev), Some(1));
        assert_eq!(check_reorder(&fwd, &[1, 0]), Err(OrderRefusal { first: 0, second: 1 }));
        assert_eq!(check_reorder(&fwd, &[0, 1]), Ok(()));
    }

    #[test]
    fn adds_commute_and_reorder_is_allowed() {
        let effs = [Effect::Add(2), Effect::Add(3), Effect::Add(5)];
        let orders: [[usize; 3]; 6] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
        for o in orders {
            assert_eq!(check_reorder(&effs, &o), Ok(()), "order {o:?}");
            let reordered: Vec<Effect> = o.iter().map(|&i| effs[i]).collect();
            assert_eq!(run_effects(0, &reordered), Some(10));
        }
    }

    /// Soundness of `commutes` under checked arithmetic: every admitted pair
    /// gives the same `run_effects` result in both orders for every boundary
    /// start value, and the overflow counterexamples are refused.
    #[test]
    fn commutes_is_sound_under_checked_overflow() {
        let vals = [i64::MIN, i64::MIN + 1, -(1 << 62), -3, -2, -1, 0, 1, 2, 3, 1 << 62, i64::MAX - 1, i64::MAX];
        let mut effs = Vec::new();
        for &v in &vals {
            effs.extend([Effect::Add(v), Effect::Mul(v), Effect::Set(v)]);
        }
        let (mut admitted, mut refused) = (0, 0);
        for a in &effs {
            for b in &effs {
                if commutes(a, b) {
                    admitted += 1;
                    for &x in &vals {
                        assert_eq!(run_effects(x, &[*a, *b]), run_effects(x, &[*b, *a]), "{a:?} {b:?} from {x}");
                    }
                } else {
                    refused += 1;
                }
            }
        }
        // Anti-vacuity: both sides are populated.
        assert!(admitted > 100 && refused > 100, "admitted {admitted}, refused {refused}");
        // The two counterexamples.
        assert_eq!(
            check_reorder(&[Effect::Add(1), Effect::Add(-1)], &[1, 0]),
            Err(OrderRefusal { first: 0, second: 1 })
        );
        assert_ne!(
            run_effects(i64::MAX, &[Effect::Add(1), Effect::Add(-1)]),
            run_effects(i64::MAX, &[Effect::Add(-1), Effect::Add(1)])
        );
        assert!(!commutes(&Effect::Mul(-1), &Effect::Mul(0)));
        assert_ne!(
            run_effects(i64::MIN, &[Effect::Mul(-1), Effect::Mul(0)]),
            run_effects(i64::MIN, &[Effect::Mul(0), Effect::Mul(-1)])
        );
    }

    #[test]
    fn mixed_sequence_partial_order() {
        let effs = [Effect::Add(1), Effect::Set(10), Effect::Add(2), Effect::Add(3)];
        assert_eq!(run_effects(0, &effs), Some(15));
        // Swapping the last two commuting adds is allowed and keeps the result.
        assert_eq!(check_reorder(&effs, &[0, 1, 3, 2]), Ok(()));
        assert_eq!(run_effects(0, &[effs[0], effs[1], effs[3], effs[2]]), Some(15));
        // Moving Add(1) after Set(10) is refused, and would change the value.
        assert_eq!(check_reorder(&effs, &[1, 0, 2, 3]), Err(OrderRefusal { first: 0, second: 1 }));
        assert_eq!(run_effects(0, &[effs[1], effs[0], effs[2], effs[3]]), Some(16));
        // Moving Add(2) before Set(10) is refused too.
        assert_eq!(check_reorder(&effs, &[0, 2, 1, 3]), Err(OrderRefusal { first: 1, second: 2 }));
    }

    #[test]
    fn two_readers_share_a_snapshot() {
        let s1 = Snapshot {
            version: 7,
            lanes: core::array::from_fn(|i| if i % 3 == 0 { i as u8 } else { 0 }),
        };
        let (c_first, s_first) = (read_count(&s1), read_sum(&s1));
        let (s_second, c_second) = (read_sum(&s1), read_count(&s1));
        assert_eq!((c_first, s_first), (c_second, s_second));
        assert!(c_first > 0 && s_first > 0);
        assert_eq!(s1.lanes[4], 0);
        let s2 = write_lane(&s1, 4, 200);
        assert_eq!(s2.version, 8);
        assert_eq!(s1.version, 7);
        assert_eq!((read_count(&s1), read_sum(&s1)), (c_first, s_first));
        assert_eq!(read_count(&s2), c_first + 1);
        assert_eq!(read_sum(&s2), s_first + 200);
    }

    #[test]
    fn sequential_dependency_must_see_first_result() {
        let fwd = [Effect::Add(3), Effect::Mul(2)];
        let rev = [Effect::Mul(2), Effect::Add(3)];
        assert_eq!(run_effects(1, &fwd), Some(8));
        assert_eq!(run_effects(1, &rev), Some(5));
        assert_eq!(check_reorder(&fwd, &[1, 0]), Err(OrderRefusal { first: 0, second: 1 }));
    }

    #[test]
    fn traversal_and_basis_do_not_change_the_count() {
        let mut rng = SplitMix64(5);
        let a = rng.block();
        let b = rng.block();
        let identity_visit: [usize; 64] = core::array::from_fn(|i| i);
        let reversed_visit: [usize; 64] = core::array::from_fn(|i| 63 - i);
        let random_visit = rng.visit_order();
        let bases = [Perm64::IDENTITY, rng.perm(), rng.perm()];
        let commuting = vec![Effect::Add(2), Effect::Add(3)];

        let reference = {
            let p = Program {
                coord_basis: Perm64::IDENTITY,
                effects: commuting.clone(),
            };
            run_lanewise_then_effects(&p, &a, &b, &identity_visit)
        };
        assert!((10..=54).contains(&reference.0), "anti-vacuity: count {}", reference.0);
        assert_eq!(reference.1, Some(reference.0 as i64 + 5));

        for basis in bases {
            for visit in [&identity_visit, &reversed_visit, &random_visit] {
                let p = Program {
                    coord_basis: basis,
                    effects: commuting.clone(),
                };
                assert_eq!(run_lanewise_then_effects(&p, &a, &b, visit), reference);
            }
        }

        // Non-commuting effects: same count, but the effect order is fixed.
        let ordered = vec![Effect::Set(1), Effect::Mul(3)];
        let p = Program {
            coord_basis: bases[1],
            effects: ordered.clone(),
        };
        let (count, result) = run_lanewise_then_effects(&p, &a, &b, &random_visit);
        assert_eq!(count, reference.0);
        assert_eq!(result, Some(3 + count as i64));
        let full = [ordered[0], ordered[1], Effect::Add(count as i64)];
        assert_eq!(check_reorder(&full, &[1, 0, 2]), Err(OrderRefusal { first: 0, second: 1 }));
        assert_ne!(
            run_effects(0, &[full[1], full[0], full[2]]),
            result,
            "the refused order would have changed the value"
        );
    }

    // ----- A4 -----

    #[test]
    fn reorder_a_fold_positive_and_negative() {
        let mut rng = SplitMix64(6);
        let x = rng.block();
        let perms: Vec<Perm64> = (0..8).map(|_| rng.perm()).collect();
        for &p in &perms {
            assert_eq!(fold_skipping_perm(&CountNonzero, p, &x), CountNonzero.fold(&p.materialize(&x)));
            assert_eq!(fold_skipping_perm(&SumLanes, p, &x), SumLanes.fold(&p.materialize(&x)));
        }
        // Negative: IndexWeightedSum is not PermInvariant, so it cannot be
        // passed to fold_skipping_perm; skipping the permutation would be wrong.
        assert!(perms
            .iter()
            .any(|p| IndexWeightedSum.fold(&x) != IndexWeightedSum.fold(&p.materialize(&x))));
    }

    #[test]
    fn fuse_two_ops_positive_and_negative() {
        let mut rng = SplitMix64(7);
        let (a, b, c) = (rng.block(), rng.block(), rng.block());
        let imm = fuse_xor_then_and_imm();
        let fused = LaneTernlog(imm).apply([&a, &b, &c]);
        let staged = LaneAnd.apply([&LaneXor.apply([&a, &b]), &c]);
        assert_eq!(fused, staged);
        assert_ne!(staged, [0u8; 64], "anti-vacuity: composition must be nonzero");

        // Negative: treating prefix as lane-local identity is not the same op.
        let true_comp = LaneAnd.apply([&PrefixSum.apply([&a]), &b]);
        let fake_fused = LaneAnd.apply([&a, &b]);
        assert_ne!(true_comp, fake_fused);
    }

    #[test]
    fn defer_reduction_positive_and_negative() {
        let product = 4 * 255 * 128i32;
        assert_eq!(product, 130_560);
        let bound = (i32::MAX / product) as usize;
        assert!((16384..16500).contains(&bound), "bound {bound} should sit just above 16384");
        let items = vec![product; 3 * 16384 + 5];
        assert!(items.len() > bound + 1);
        let exact = exact_i64_total(&items);
        assert_eq!(exact, product as i64 * items.len() as i64);
        // Positive: flushing every 16384 items stays inside the i32 range.
        assert_eq!(deferred_i32_total(&items, 16384), exact);
        // Boundary: exactly the bound still fits.
        assert_eq!(deferred_i32_total(&items, bound), exact);
        // Negative: one past the bound wraps.
        assert_ne!(deferred_i32_total(&items, bound + 1), exact);
    }

    #[test]
    fn propagate_compact_result_invariant_vs_equivariant() {
        let mut rng = SplitMix64(8);
        let a = rng.block();
        let b = rng.block();
        let p = rng.perm();
        assert_ne!(p, Perm64::IDENTITY);

        // Invariant result: no basis needed, equal across bases.
        let base = CountNonzero.fold(&bitand(&a, &b));
        for q in [Perm64::IDENTITY, p, rng.perm()] {
            let moved = bitand(&q.materialize(&a), &q.materialize(&b));
            assert_eq!(CountNonzero.fold(&moved), base);
        }

        // Equivariant result: data alone is in the wrong coordinates.
        let r = combine_two(&LaneXor, p, &a, p, &b);
        assert_ne!(r.data, r.materialize(), "data without its basis is not the answer");
        let (pa, pb) = (p.materialize(&a), p.materialize(&b));
        let expected: [u8; 64] = core::array::from_fn(|i| pa[i] ^ pb[i]);
        assert_eq!(r.materialize(), expected);
    }

    #[test]
    fn refuse_invalid_composition() {
        let bad = [Effect::Set(4), Effect::Add(1)];
        assert!(check_reorder(&bad, &[1, 0]).is_err());
        let ok = [Effect::Add(4), Effect::Add(1)];
        assert_eq!(check_reorder(&ok, &[1, 0]), Ok(()));
        assert!(commutes(&Effect::Set(2), &Effect::Set(2)));
        assert!(!commutes(&Effect::Set(2), &Effect::Set(3)));
        assert!(!commutes(&Effect::Add(2), &Effect::Mul(3)));
    }

    #[test]
    fn effect_overflow_is_reported_not_wrapped() {
        assert_eq!(run_effects(i64::MAX, &[Effect::Add(1)]), None);
        assert_eq!(run_effects(i64::MAX, &[Effect::Add(1), Effect::Set(0)]), None);
        assert_eq!(run_effects(5, &[Effect::Mul(2)]), Some(10));
    }

    #[test]
    #[should_panic(expected = "not a permutation")]
    fn check_reorder_rejects_a_non_permutation() {
        let effs = [Effect::Add(1), Effect::Add(2)];
        let _ = check_reorder(&effs, &[0, 0]);
    }

    #[test]
    fn law_table_sanity() {
        let mut names: Vec<&str> = LAW_TABLE.iter().map(|r| r.name).collect();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(names.len(), before, "row names must be unique");
        assert_eq!(before, 10);

        for r in LAW_TABLE {
            if r.perm_invariant {
                assert!(!r.lane_local, "{}: an invariant row is a fold, not lane-local", r.name);
                assert!(!r.perm_equivariant, "{}: invariant and equivariant are exclusive", r.name);
            }
            if r.ordered_effects {
                assert!(!r.commutative, "{}: ordered effects do not commute", r.name);
            }
            if r.perm_equivariant {
                assert!(r.lane_local, "{}: equivariance needs lane locality", r.name);
            }
        }
        for name in ["prefix sum", "index-weighted sum"] {
            let r = LAW_TABLE
                .iter()
                .find(|r| r.name == name)
                .expect("row present");
            assert!(!r.perm_invariant && !r.perm_equivariant, "{name}");
        }
        // Both ordering-relevant rows are present and flagged.
        assert!(LAW_TABLE.iter().any(|r| r.ordered_effects));
        assert!(LAW_TABLE.iter().any(|r| r.perm_invariant));
        assert!(LAW_TABLE.iter().any(|r| r.perm_equivariant));
    }
}
