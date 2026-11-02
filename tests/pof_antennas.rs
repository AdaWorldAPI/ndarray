#![cfg(feature = "std")]
//! D-POF-ANT-0 antennas: exactness and falsifiers. The A1, A3/A4, A5 and A6
//! modules carry their own tests; A2 (the arithmetic exposure) is checked here
//! against #388's `i64` oracle and the unchanged B and E5 kernels.

#[path = "../examples/u9_head_to_head/kernels.rs"]
mod kernels;
#[path = "../examples/u9_abacus/anchor.rs"]
mod anchor;
#[path = "../examples/pof_antennas/exposure.rs"]
mod exposure;
#[path = "../examples/pof_antennas/meta.rs"]
mod meta;
#[path = "../examples/pof_antennas/order_law.rs"]
mod order_law;
#[path = "../examples/pof_antennas/position.rs"]
mod position;
#[path = "../examples/pof_antennas/scale.rs"]
mod scale;

use anchor::{split_anchor, AccE5, QueryE};
use exposure::{pos_mag_oracle, AccDual, AccE2, AccPosMag};
use kernels::{check, oracle, prepare_b, AccB, Block, B_FLUSH_BLOCKS};
use ndarray::simd::{U64x8, U8x64};

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

/// E2 end to end and as a scan, E3's two consumers fused and separate, and
/// B and E5 on the same data.
fn run_a2(a: &[Raw], b: &[Raw]) -> ([i64; 4], (i64, u64), (i64, u64)) {
    let (mut e2, mut s2, mut bk, mut e5) = (AccE2::new(), AccE2::new(), AccB::new(), AccE5::new());
    let mut dual = AccDual::new();
    let (mut sep_b, mut sep_m) = (AccB::new(), AccPosMag::new());
    for (ra, rb) in a.iter().zip(b) {
        let (x, y) = (block(ra), block(rb));
        check(x, y).unwrap();
        let (low7, top) = split_anchor(y.mag);
        e2.mac(x.mag, x.positive ^ y.positive, low7, top);
        s2.scan_step(x, &QueryE::new(y));
        bk.step(x, y);
        e5.step(x, y);
        let (neg, l, t) = prepare_b(x, y);
        dual.mac(x.mag, neg, l, t);
        sep_b.mac(x.mag, neg, l, t);
        sep_m.mac(x.mag, neg);
    }
    (
        [e2.finish(), s2.finish(), bk.finish(), e5.finish()],
        dual.finish(),
        (sep_b.finish(), sep_m.finish()),
    )
}

fn check_a2(a: &[Raw], b: &[Raw]) {
    let want = oracle(a, b).unwrap();
    let want_m = pos_mag_oracle(a, b);
    let (four, dual, sep) = run_a2(a, b);
    assert_eq!(four, [want; 4]);
    assert_eq!(dual, (want, want_m));
    assert_eq!(sep, (want, want_m));
}

#[test]
fn a2_all_511_by_511_pairs_are_exact() {
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
            assert_eq!(run_a2(&[(am, ap)], &[(bm, bp)]).0, [i64::from(va) * i64::from(vb); 4], "{va} x {vb}");
            n += 1;
        }
    }
    assert_eq!(n, 511 * 511);
}

#[test]
fn a2_mixed_blocks_and_worst_case_runs_are_exact() {
    let mut s = 0x00DD_BA11_u64;
    let mut val = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((s >> 33) % 511) as i32 - 255
    };
    for blocks in [1usize, 3, 64, 300] {
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
        check_a2(&a, &b);
    }
    let n = 2 * B_FLUSH_BLOCKS + 3;
    for (pa, pb) in [(u64::MAX, u64::MAX), (u64::MAX, 0), (0xF0F0_F0F0_0F0F_0F0F, 0x3333_CCCC_5555_AAAA)] {
        check_a2(&vec![([255u8; 64], pa); n], &vec![([255u8; 64], pb); n]);
    }
}

/// The reuse law E2 relies on, byte by byte: narrowing a zero-masked vector
/// by a second mask equals masking once by the AND of both masks, and the
/// narrower mask is not the same mask (sign alone keeps more bytes).
#[test]
fn a2_narrowing_law_holds_and_is_not_identity() {
    let a = U8x64::from_array(core::array::from_fn(|i| (i as u8).wrapping_mul(37) | 0x81));
    let mut seen_difference = false;
    for (s, t) in [
        (0xF0F0_1234_8000_0001u64, 0x0FF0_F00F_5555_AAAAu64),
        (u64::MAX, 0x8000_0000_0000_0001),
        (0, u64::MAX),
    ] {
        let zero = U8x64::splat(0);
        let a_s = U8x64::mask_blend(s, zero, a);
        let narrowed = U64x8::splat(0).sum8_u8_add_masked(a_s, t);
        let direct = U64x8::splat(0).sum8_u8_add_masked(a, s & t);
        assert_eq!(narrowed.to_array(), direct.to_array());
        let sign_only = U64x8::splat(0).sum8_u8_add_masked(a, s);
        seen_difference |= sign_only.to_array() != direct.to_array();
    }
    assert!(seen_difference, "sign and sign AND bit7 never differed: vacuous");
}
