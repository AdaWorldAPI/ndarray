#![cfg(feature = "std")]
//! Exactness of the D-U9-ABACUS probe (`examples/u9_abacus/abacus.rs`)
//! against the `i64` oracle of the U9 head-to-head and against its B kernel.

#[path = "../examples/u9_head_to_head/kernels.rs"]
mod kernels;
#[path = "../examples/u9_abacus/abacus.rs"]
mod abacus;
#[path = "../examples/u9_abacus/carry16.rs"]
mod carry16;
#[path = "../examples/u9_abacus/anchor.rs"]
mod anchor;

use abacus::{carry_save3, planes, Abacus, AccAbacus, AccCsa, AccPopcnt, QueryC};
use anchor::{bit7, sign_bytes, split_anchor, AccE3, AccE4, AccE5, AccESum, QueryE};
use carry16::{carry_after, carry_before, AccD, QueryD, D_FLUSH_BLOCKS};
use kernels::{check, oracle, AccB, Block, NeverTouched, B_FLUSH_BLOCKS};
use ndarray::simd::{U16x32, U8x64};

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
        assert_eq!(a.count_ones() + b.count_ones() + c.count_ones(), s.count_ones() + 2 * cy.count_ones());
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

/// D (react, anticipate) and their scans (b as the prepared query).
fn run_d(a: &[Raw], b: &[Raw]) -> [i64; 4] {
    let (mut r, mut an) = (AccD::<false>::new(), AccD::<true>::new());
    let (mut sr, mut sa) = (AccD::<false>::new(), AccD::<true>::new());
    for (ra, rb) in a.iter().zip(b) {
        let (x, y) = (block(ra), block(rb));
        check(x, y).unwrap();
        r.step(x, y);
        an.step(x, y);
        let q = QueryD::new(y);
        sr.scan_step(x, &q);
        sa.scan_step(x, &q);
    }
    [r.finish(), an.finish(), sr.finish(), sa.finish()]
}

/// Both carry rules against `u32` arithmetic, for every `low` in 0..=65535,
/// with `p` on both sides of the headroom (`!low`, `!low + 1`) and spread
/// over the product range.
#[test]
fn carry_rules_match_u32_for_every_low() {
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    for base in (0u32..65_536).step_by(32) {
        let lows: [u16; 32] = core::array::from_fn(|k| (base + k as u32) as u16);
        let low = U16x32::from_array(lows);
        let mut cases: Vec<[u16; 32]> = vec![
            core::array::from_fn(|k| !lows[k]),
            core::array::from_fn(|k| (!lows[k]).wrapping_add(1)),
            core::array::from_fn(|k| (!lows[k]).wrapping_sub(1)),
            [0; 32],
            [65_025; 32],
        ];
        for _ in 0..4 {
            cases.push(core::array::from_fn(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % 65_026) as u16
            }));
        }
        for ps in cases {
            let p = U16x32::from_array(ps);
            let sum = low + p;
            let (after, before) = (carry_after(low, p, sum).to_array(), carry_before(low, p).to_array());
            for k in 0..32 {
                let want = u16::from(u32::from(lows[k]) + u32::from(ps[k]) > 0xFFFF);
                assert_eq!(after[k], want, "react low {} p {}", lows[k], ps[k]);
                assert_eq!(before[k], want, "anticipate low {} p {}", lows[k], ps[k]);
            }
        }
    }
}

#[test]
fn d_all_511_by_511_pairs_are_exact() {
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
            assert_eq!(run_d(&[(am, ap)], &[(bm, bp)]), [want; 4], "{va} x {vb}");
            n += 1;
        }
    }
    assert_eq!(n, 511 * 511);
}

#[test]
fn d_full_blocks_match_the_oracle() {
    let mut s = 0x0123_4567_89AB_CDEFu64;
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
        assert_eq!(run_d(&a, &b), [oracle(&a, &b).unwrap(); 4], "{blocks} blocks");
    }
}

/// All-255 runs past D's flush bound: a carry on almost every add, `high`
/// close to its limit, totals far beyond `i32`.
#[test]
fn d_worst_case_runs_cross_the_flush_bound() {
    let n = 2 * D_FLUSH_BLOCKS + 5;
    for (pa, pb) in [(u64::MAX, u64::MAX), (u64::MAX, 0), (0xF0F0_F0F0_0F0F_0F0F, 0x3333_CCCC_5555_AAAA)] {
        let a = vec![([255u8; 64], pa); n];
        let b = vec![([255u8; 64], pb); n];
        assert_eq!(run_d(&a, &b), [oracle(&a, &b).unwrap(); 4]);
    }
}

/// E4, E3, Esum and their scans (b as the prepared query).
fn run_e(a: &[Raw], b: &[Raw]) -> [i64; 6] {
    let (mut e4, mut e3, mut es) = (AccE4::new(), AccE3::new(), AccESum::new());
    let (mut s4, mut s3, mut ss) = (AccE4::new(), AccE3::new(), AccESum::new());
    for (ra, rb) in a.iter().zip(b) {
        let (x, y) = (block(ra), block(rb));
        check(x, y).unwrap();
        e4.step(x, y);
        e3.step(x, y);
        es.step(x, y);
        let q = QueryE::new(y);
        s4.scan_step(x, &q);
        s3.scan_step(x, &q);
        ss.scan_step(x, &q);
    }
    [e4.finish(), e3.finish(), es.finish(), s4.finish(), s3.finish(), ss.finish()]
}

/// The anchor split for every magnitude: `|b| = low7 + 128 * bit7`, and the
/// sign bytes are `+1` / `-1` exactly by the plane.
#[test]
fn anchor_split_and_sign_bytes() {
    for base in [0u8, 64, 128, 192] {
        let m: [u8; 64] = core::array::from_fn(|i| (i as u8).wrapping_add(base));
        let (low7, top) = split_anchor(U8x64::from_array(m));
        assert_eq!(top, bit7(U8x64::from_array(m)));
        let low7 = low7.to_array();
        for k in 0..64 {
            let t = i32::from((top >> k & 1) as u8);
            assert!((0..=127).contains(&low7[k]));
            assert_eq!(i32::from(low7[k]) + 128 * t, i32::from(m[k]), "magnitude {}", m[k]);
        }
    }
    let plane = 0xF0F0_1234_8000_0001u64;
    let s = sign_bytes(plane).to_array();
    for (k, &v) in s.iter().enumerate() {
        assert_eq!(v, if plane >> k & 1 == 1 { -1 } else { 1 });
    }
}

#[test]
fn e_all_511_by_511_pairs_are_exact() {
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
            assert_eq!(run_e(&[(am, ap)], &[(bm, bp)]), [want; 6], "{va} x {vb}");
            n += 1;
        }
    }
    assert_eq!(n, 511 * 511);
}

#[test]
fn e_full_blocks_match_the_oracle() {
    let mut s = 0x7F4A_7C15_9E37_79B9u64;
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
        assert_eq!(run_e(&a, &b), [oracle(&a, &b).unwrap(); 6], "{blocks} blocks");
    }
}

#[test]
fn e_worst_case_runs_cross_the_flush_bound() {
    let n = 3 * B_FLUSH_BLOCKS + 5;
    for (pa, pb) in [(u64::MAX, u64::MAX), (u64::MAX, 0), (0, 0), (0xF0F0_F0F0_0F0F_0F0F, 0x3333_CCCC_5555_AAAA)] {
        let a = vec![([255u8; 64], pa); n];
        let b = vec![([255u8; 64], pb); n];
        assert_eq!(run_e(&a, &b), [oracle(&a, &b).unwrap(); 6]);
    }
}

/// E5 (vertical masked byte sums, #390) end to end and as a scan (b as the
/// prepared query), next to B on the same data.
fn run_e5(a: &[Raw], b: &[Raw]) -> [i64; 3] {
    let (mut e5, mut s5, mut bk) = (AccE5::new(), AccE5::new(), AccB::new());
    for (ra, rb) in a.iter().zip(b) {
        let (x, y) = (block(ra), block(rb));
        check(x, y).unwrap();
        e5.step(x, y);
        s5.scan_step(x, &QueryE::new(y));
        bk.step(x, y);
    }
    [e5.finish(), s5.finish(), bk.finish()]
}

#[test]
fn e5_all_511_by_511_pairs_are_exact() {
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
            assert_eq!(run_e5(&[(am, ap)], &[(bm, bp)]), [want; 3], "{va} x {vb}");
            n += 1;
        }
    }
    assert_eq!(n, 511 * 511);
}

#[test]
fn e5_full_blocks_with_mixed_signs_match_the_oracle() {
    let mut s = 0x2545_F491_4F6C_DD1Du64;
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
        // Mixed signs and both bit-7 states present, or the case proves little.
        assert!(a
            .iter()
            .chain(&b)
            .any(|(m, p)| *p != 0 && *p != u64::MAX && m.iter().any(|&x| x >= 128)));
        assert_eq!(run_e5(&a, &b), [oracle(&a, &b).unwrap(); 3], "{blocks} blocks");
    }
}

/// All-255 runs past three flush bounds, every sign mix; totals beyond `i32`.
#[test]
fn e5_worst_case_runs_cross_the_flush_bound() {
    let n = 3 * B_FLUSH_BLOCKS + 5;
    for (pa, pb) in [(u64::MAX, u64::MAX), (u64::MAX, 0), (0, 0), (0xF0F0_F0F0_0F0F_0F0F, 0x3333_CCCC_5555_AAAA)] {
        let a = vec![([255u8; 64], pa); n];
        let b = vec![([255u8; 64], pb); n];
        assert_eq!(run_e5(&a, &b), [oracle(&a, &b).unwrap(); 3]);
    }
}

/// The anchor accumulators are reduced once per flush, never per block:
/// `n` blocks give `n / B_FLUSH_BLOCKS` flushes, plus one at `finish`.
#[test]
fn e5_reduces_the_anchor_only_at_flush() {
    let n = 2 * B_FLUSH_BLOCKS + 7;
    let (x, y) = (block(&([200u8; 64], 0x0F0F)), block(&([131u8; 64], u64::MAX)));
    let mut e5 = AccE5::new();
    for _ in 0..n {
        e5.step(x, y);
    }
    assert_eq!(e5.reductions, n / B_FLUSH_BLOCKS);
}

/// Touched zero scores 0 in E5; never-touched is refused before E5 sees it.
#[test]
fn e5_touched_zero_scores_zero_and_never_touched_is_refused() {
    let zero = ([0u8; 64], u64::MAX);
    let big = ([255u8; 64], 0xAAAA_5555_AAAA_5555);
    assert_eq!(run_e5(&[zero, big], &[big, zero]), [0; 3]);
    let mut m = [200u8; 64];
    m[9] = 0;
    let sentinel = block(&(m, !(1u64 << 9)));
    assert_eq!(check(sentinel, block(&big)), Err(NeverTouched(1 << 9)));
    assert_eq!(check(block(&big), sentinel), Err(NeverTouched(1 << 9)));
}
