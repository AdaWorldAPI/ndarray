#![cfg(feature = "std")]
//! Exactness of the D-U9-ABACUS probe (`examples/u9_abacus/abacus.rs`)
//! against the `i64` oracle of the U9 head-to-head and against its B kernel.

#[path = "../examples/u9_head_to_head/kernels.rs"]
mod kernels;
#[path = "../examples/u9_abacus/abacus.rs"]
mod abacus;

use abacus::{carry_save3, planes, Abacus, AccAbacus, AccCsa, AccPopcnt, QueryC};
use kernels::{check, oracle, AccB, Block, NeverTouched, B_FLUSH_BLOCKS};
use ndarray::simd::U8x64;

type Raw = ([u8; 64], u64);

fn block(r: &Raw) -> Block {
    Block {
        mag: U8x64::from_array(r.0),
        positive: r.1,
    }
}

fn encode(v: i32) -> (u8, bool) {
    (v.unsigned_abs() as u8, v >= 0)
}

/// The three C folds, the C scan (b as the prepared query) and B, on the same data.
fn run_all(a: &[Raw], b: &[Raw]) -> [i64; 5] {
    let (mut abacus, mut pop, mut bk) = (AccAbacus::new(), AccPopcnt::new(), AccB::new());
    let mut csa = AccCsa::new();
    let mut scan = AccAbacus::new();
    for (ra, rb) in a.iter().zip(b) {
        let (x, y) = (block(ra), block(rb));
        check(x, y).unwrap();
        let (pa, pb) = (planes(x.mag), planes(y.mag));
        let negative = x.positive ^ y.positive;
        abacus.mac(&pa, negative, &pb);
        pop.mac(&pa, negative, &pb);
        csa.mac(&pa, negative, &pb);
        let q = QueryC::new(y.mag, y.positive);
        scan.mac(&planes(x.mag), x.positive ^ q.positive, &q.planes);
        bk.step(x, y);
    }
    [abacus.finish(), pop.finish(), csa.finish(), scan.finish(), bk.finish()]
}

/// The full-adder identity bit by bit, for all eight input combinations at
/// every bit position, and on random words (summed over all bits).
#[test]
fn carry_save3_is_a_full_adder_per_bit() {
    for combo in 0u8..8 {
        let bit = |n: u8| if combo >> n & 1 == 1 { u64::MAX } else { 0 };
        let (s, c) = carry_save3(bit(0), bit(1), bit(2));
        let want = u32::from(combo.count_ones() as u8);
        for k in 0..64 {
            let got = (s >> k & 1) + 2 * (c >> k & 1);
            assert_eq!(got as u32, want, "combo {combo:03b} bit {k}");
        }
    }
    let mut x = 0x243F_6A88_85A3_08D3u64;
    for _ in 0..10_000 {
        let mut r = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let (a, b, c) = (r(), r(), r());
        let (s, cy) = carry_save3(a, b, c);
        assert_eq!(
            a.count_ones() + b.count_ones() + c.count_ones(),
            s.count_ones() + 2 * cy.count_ones()
        );
    }
}

/// Planes recover every magnitude 0..=255 in every lane.
#[test]
fn planes_recover_every_magnitude() {
    for base in [0u8, 64, 128, 192] {
        let m: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_add(base));
        let p = planes(U8x64::from_array(m));
        for (k, &mk) in m.iter().enumerate() {
            let back: u8 = (0..8).map(|i| ((p[i] >> k & 1) as u8) << i).sum();
            assert_eq!(back, mk);
        }
    }
}

/// Carries forced through many consecutive columns: 2^20 all-ones planes at
/// column 0 must count exactly 64 * 2^20, and one all-ones plane at each of
/// columns 0..=62 must not lose the chain of carries it sets off.
#[test]
fn abacus_carries_through_many_columns() {
    let mut ab = Abacus::new();
    for _ in 0..(1u32 << 20) {
        ab.add(0, u64::MAX);
    }
    assert_eq!(ab.value(), 64i128 << 20);
    let mut ab = Abacus::new();
    let mut want = 0i128;
    for round in 0..3 {
        for w in 0..40 {
            ab.add(w, u64::MAX);
            want += 64i128 << w;
        }
        assert_eq!(ab.value(), want, "round {round}");
    }
    // A single-lane pattern: each lane gets a different count.
    let mut ab = Abacus::new();
    let mut want = 0i128;
    for t in 0..5000u64 {
        let p = t.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ab.add((t % 7) as usize, p);
        want += i128::from(p.count_ones()) << (t % 7);
    }
    assert_eq!(ab.value(), want);
}

#[test]
fn all_511_by_511_pairs_are_exact() {
    let mut n = 0usize;
    for va in -255..=255i32 {
        for vb in -255..=255i32 {
            let lane = n % 64;
            let (ma, pa) = encode(va);
            let (mb, pb) = encode(vb);
            let (mut am, mut bm) = ([0u8; 64], [0u8; 64]);
            am[lane] = ma;
            bm[lane] = mb;
            let ap = if pa { u64::MAX } else { !(1u64 << lane) };
            let bp = if pb { u64::MAX } else { !(1u64 << lane) };
            let want = i64::from(va) * i64::from(vb);
            assert_eq!(run_all(&[(am, ap)], &[(bm, bp)]), [want; 5], "{va} x {vb}");
            n += 1;
        }
    }
    assert_eq!(n, 511 * 511);
}

#[test]
fn full_blocks_match_the_oracle_and_b() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut val = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 33) % 511) as i32 - 255
    };
    for blocks in [1usize, 2, 7, 64, 300] {
        let mut mk = || {
            let mut m = [0u8; 64];
            let mut p = 0u64;
            for (i, slot) in m.iter_mut().enumerate() {
                let (mi, pi) = encode(val());
                *slot = mi;
                p |= u64::from(pi) << i;
            }
            (m, p)
        };
        let a: Vec<_> = (0..blocks).map(|_| mk()).collect();
        let b: Vec<_> = (0..blocks).map(|_| mk()).collect();
        let want = oracle(&a, &b).unwrap();
        assert_eq!(run_all(&a, &b), [want; 5], "{blocks} blocks");
    }
}

/// +-255 x +-255 in every lane, past B's flush bound, totals beyond `i32`.
#[test]
fn worst_case_runs_exceed_i32_exactly() {
    let n = 3 * B_FLUSH_BLOCKS + 5;
    for (pa, pb) in [(u64::MAX, u64::MAX), (u64::MAX, 0), (0, 0), (0xF0F0_F0F0_0F0F_0F0F, 0x3333_CCCC_5555_AAAA)] {
        let a = vec![([255u8; 64], pa); n];
        let b = vec![([255u8; 64], pb); n];
        let want = oracle(&a, &b).unwrap();
        assert!(want.unsigned_abs() > i32::MAX as u64 || pa != pb);
        assert_eq!(run_all(&a, &b), [want; 5]);
    }
}

/// Never-touched is refused before any arithmetic; touched zero is a number.
#[test]
fn never_touched_is_refused_and_zero_is_not() {
    let zero = ([0u8; 64], u64::MAX);
    assert_eq!(run_all(&[zero], &[zero]), [0; 5]);
    let mut m = [9u8; 64];
    m[5] = 0;
    let sentinel = block(&(m, !(1u64 << 5)));
    assert_eq!(check(sentinel, block(&zero)), Err(NeverTouched(1 << 5)));
    assert_eq!(check(block(&zero), sentinel), Err(NeverTouched(1 << 5)));
}
