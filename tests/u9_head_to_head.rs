#![cfg(feature = "std")]
//! Correctness of the two U9 x U9 candidates in
//! `examples/u9_head_to_head/kernels.rs` against one independent `i64` oracle.

#[path = "../examples/u9_head_to_head/kernels.rs"]
mod kernels;

use kernels::{check, oracle, AccA, AccB, Block, NeverTouched, QueryA, QueryB, A_FLUSH_BLOCKS, B_FLUSH_BLOCKS};
use ndarray::simd::U8x64;

fn block(mag: [u8; 64], positive: u64) -> Block {
    Block {
        mag: U8x64::from_array(mag),
        positive,
    }
}

/// (magnitude, positive bit) of a numeric value; 0 is touched +0.
fn encode(v: i32) -> (u8, bool) {
    (v.unsigned_abs() as u8, v >= 0)
}

fn run_a(a: &[([u8; 64], u64)], b: &[([u8; 64], u64)], mullo: bool) -> i64 {
    let mut acc = AccA::new();
    for ((am, ap), (bm, bp)) in a.iter().zip(b) {
        let (x, y) = (block(*am, *ap), block(*bm, *bp));
        check(x, y).unwrap();
        if mullo {
            acc.step_mullo(x, y)
        } else {
            acc.step(x, y)
        }
    }
    acc.finish()
}

fn run_b(a: &[([u8; 64], u64)], b: &[([u8; 64], u64)]) -> i64 {
    let mut acc = AccB::new();
    for ((am, ap), (bm, bp)) in a.iter().zip(b) {
        let (x, y) = (block(*am, *ap), block(*bm, *bp));
        check(x, y).unwrap();
        acc.step(x, y);
    }
    acc.finish()
}

/// Scan forms: `b` is the query, prepared once per query block; `a` streams.
fn run_scan(a: &[([u8; 64], u64)], b: &[([u8; 64], u64)]) -> (i64, i64) {
    let (mut sa, mut sb) = (AccA::new(), AccB::new());
    for ((am, ap), (bm, bp)) in a.iter().zip(b) {
        let (x, q) = (block(*am, *ap), block(*bm, *bp));
        check(x, q).unwrap();
        sa.scan_step(x, &QueryA::new(q));
        sb.scan_step(x, &QueryB::new(q));
    }
    (sa.finish(), sb.finish())
}

/// Every one of the 511 x 511 numeric pairs, each alone in its block (the
/// other 63 lanes are touched +0), at a lane that rotates with the pair so
/// every lane position is exercised.
#[test]
fn all_511_by_511_pairs_are_exact() {
    let mut n = 0usize;
    for va in -255..=255i32 {
        for vb in -255..=255i32 {
            let lane = n % 64;
            let (ma, pa) = encode(va);
            let (mb, pb) = encode(vb);
            let mut am = [0u8; 64];
            let mut bm = [0u8; 64];
            am[lane] = ma;
            bm[lane] = mb;
            let ap = if pa { u64::MAX } else { !(1u64 << lane) };
            let bp = if pb { u64::MAX } else { !(1u64 << lane) };
            let (a, b) = ([(am, ap)], [(bm, bp)]);
            let want = i64::from(va) * i64::from(vb);
            assert_eq!(oracle(&a, &b), Ok(want), "oracle {va} x {vb}");
            assert_eq!(run_a(&a, &b, false), want, "A {va} x {vb}");
            assert_eq!(run_a(&a, &b, true), want, "A mullo {va} x {vb}");
            assert_eq!(run_b(&a, &b), want, "B {va} x {vb}");
            assert_eq!(run_scan(&a, &b), (want, want));
            n += 1;
        }
    }
    assert_eq!(n, 511 * 511);
}

/// Full blocks: 64 different pairs per block, so a lane mix-up between the
/// operands (an order-changing widening or narrowing) cannot cancel out.
#[test]
fn full_blocks_match_the_oracle() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as u32
    };
    let mut val = || (next() % 511) as i32 - 255;
    for blocks in [1usize, 2, 7, 64] {
        let mk = |val: &mut dyn FnMut() -> i32| {
            let mut m = [0u8; 64];
            let mut p = 0u64;
            for (i, slot) in m.iter_mut().enumerate() {
                let (mi, pi) = encode(val());
                *slot = mi;
                p |= u64::from(pi) << i;
            }
            (m, p)
        };
        let a: Vec<_> = (0..blocks).map(|_| mk(&mut val)).collect();
        let b: Vec<_> = (0..blocks).map(|_| mk(&mut val)).collect();
        let want = oracle(&a, &b).unwrap();
        assert_eq!(run_a(&a, &b, false), want, "A, {blocks} blocks");
        assert_eq!(run_a(&a, &b, true), want, "A mullo, {blocks} blocks");
        assert_eq!(run_b(&a, &b), want, "B, {blocks} blocks");
        assert_eq!(run_scan(&a, &b), (want, want));
    }
}

/// Worst-case magnitudes past both flush intervals, all one sign and mixed:
/// nothing wraps, rounds or saturates.
#[test]
fn worst_case_runs_cross_the_flush_bounds_exactly() {
    let n = 3 * B_FLUSH_BLOCKS + 5;
    assert!(n > 4 * A_FLUSH_BLOCKS);
    for (pa, pb) in [(u64::MAX, u64::MAX), (u64::MAX, 0), (0, 0), (0xF0F0_F0F0_0F0F_0F0F, 0x3333_CCCC_5555_AAAA)] {
        let a = vec![([255u8; 64], pa); n];
        let b = vec![([255u8; 64], pb); n];
        let want = oracle(&a, &b).unwrap();
        assert!(want.unsigned_abs() > i32::MAX as u64 || pa != pb, "the run must leave the i32 range");
        assert_eq!(run_a(&a, &b, false), want);
        assert_eq!(run_a(&a, &b, true), want);
        assert_eq!(run_b(&a, &b), want);
        assert_eq!(run_scan(&a, &b), (want, want));
    }
}

/// The never-touched lane (plane 0, magnitude 0) is refused in either
/// operand; touched +0 (plane 1, magnitude 0) is accepted.
#[test]
fn never_touched_is_refused_and_touched_zero_is_not() {
    let ok = block([0u8; 64], u64::MAX);
    assert_eq!(check(ok, ok), Ok(()));
    for lane in [0usize, 17, 63] {
        let mut m = [7u8; 64];
        m[lane] = 0;
        let sentinel = block(m, !(1u64 << lane));
        assert_eq!(check(sentinel, ok), Err(NeverTouched(1 << lane)));
        assert_eq!(check(ok, sentinel), Err(NeverTouched(1 << lane)));
        // Negative nonzero magnitudes are numbers, not the sentinel.
        let negative = block([7u8; 64], 0);
        assert_eq!(check(negative, ok), Ok(()));
        assert_eq!(oracle(&[(m, !(1u64 << lane))], &[([0u8; 64], u64::MAX)]), Err(NeverTouched(1 << lane)));
    }
}

/// The schoolbook split behind B, for every magnitude 0..=255, through the
/// kernel's own `prepare_b`: the two partial operands recombine to the
/// original byte by XOR (no bit lost, nothing negated), the signed value is
/// `|b| = low7 - top` with `top` in {0, -128}, and flipping either polarity
/// bit changes only the routing plane, never the byte operands.
#[test]
fn split_recombines_and_polarity_only_routes() {
    let mag: [u8; 64] = core::array::from_fn(|i| i as u8);
    for base in [0u8, 64, 128, 192] {
        let m: [u8; 64] = core::array::from_fn(|i| mag[i].wrapping_add(base));
        let a = block([1u8; 64], u64::MAX);
        let (plane, low7, top) = kernels::prepare_b(a, block(m, u64::MAX));
        assert_eq!(plane, 0, "equal polarities route every lane positive");
        let (low7, top) = (low7.to_array(), top.to_array());
        for i in 0..64 {
            let (l, t) = (low7[i], top[i]);
            assert!(t == 0 || t == -128, "top of {} is {t}", m[i]);
            assert_eq!(l.to_ne_bytes()[0] ^ t.to_ne_bytes()[0], m[i], "XOR recombination of {}", m[i]);
            assert_eq!(i32::from(l) - i32::from(t), i32::from(m[i]), "low7 - top of {}", m[i]);
        }
        for (pa, pb) in [(0u64, u64::MAX), (u64::MAX, 0), (0x5555_5555_5555_5555, 0x0F0F_0F0F_0F0F_0F0F)] {
            let (flipped, l2, t2) = kernels::prepare_b(block([1u8; 64], pa), block(m, pb));
            assert_eq!(flipped, pa ^ pb, "the plane is the XOR of the two polarities");
            assert_eq!(l2.to_array(), low7, "polarity must not touch low7");
            assert_eq!(t2.to_array(), top, "polarity must not touch top");
        }
    }
}
