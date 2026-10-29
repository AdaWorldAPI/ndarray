//! D-U9-ABACUS probe: schoolbook U9 x U9 on bit planes (`abacus.rs`) against
//! the U9 head-to-head's B kernel (`../u9_head_to_head/kernels.rs`) on the
//! same data. Every kernel is checked against the `i64` oracle before it is
//! timed. `cargo run --release --example u9_abacus --features std`; the tier
//! is whatever this build compiled (printed in the header).

#[path = "../u9_head_to_head/kernels.rs"]
mod kernels;
mod abacus;
mod carry16;

use abacus::*;
use carry16::*;
use kernels::*;
use ndarray::simd::{I32x16, I8x64, U16x32, U8x64};
use std::hint::black_box;
use std::time::Instant;

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512vnni") {
        "x86_64 AVX-512 + VNNI"
    } else if cfg!(target_feature = "avx512bw") {
        "x86_64 AVX-512 (no VNNI)"
    } else if cfg!(target_feature = "avx2") {
        "x86_64 AVX2"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64 NEON"
    } else {
        "other"
    }
}

type Raw = ([u8; 64], u64);

/// Uniform over the 511 numeric values; never-touched never appears.
fn data(blocks: usize, seed: u64) -> (Vec<Raw>, Vec<Raw>) {
    let mut s = seed;
    let mut mk = || {
        let mut m = [0u8; 64];
        let mut p = 0u64;
        for (i, mi) in m.iter_mut().enumerate() {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let v = ((s >> 33) % 511) as i32 - 255;
            *mi = v.unsigned_abs() as u8;
            p |= u64::from(v >= 0) << i;
        }
        (m, p)
    };
    let a = (0..blocks).map(|_| mk()).collect();
    let b = (0..blocks).map(|_| mk()).collect();
    (a, b)
}

fn time(name: &str, blocks: usize, runs: usize, base: Option<f64>, f: &dyn Fn() -> i64) -> f64 {
    black_box(f());
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let t0 = Instant::now();
            black_box(f());
            t0.elapsed().as_nanos() as f64 / blocks as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    let med = t[runs / 2];
    let rel = base
        .map(|b| format!("  {:6.2}x B", med / b))
        .unwrap_or_default();
    println!("{name:<60} {med:8.3} ns/64{rel}");
    med
}

fn bench(label: &str, n: usize, runs: usize, seed: u64) {
    let (ra, rb) = data(n, seed);
    let blk = |v: &[([u8; 64], u64)]| -> Vec<Block> {
        v.iter()
            .map(|(m, p)| Block {
                mag: U8x64::from_array(*m),
                positive: *p,
            })
            .collect()
    };
    let (ab, bb) = (blk(&ra), blk(&rb));
    let want = oracle(&ra, &rb).unwrap();
    let qraw = vec![rb[0]; n];
    let want_scan = oracle(&ra, &qraw).unwrap();
    let q = bb[0];

    // Execution-form operands for the arithmetic-only rows, built once.
    let split: Vec<(u64, I8x64, I8x64)> = ab.iter().zip(&bb).map(|(a, b)| prepare_b(*a, *b)).collect();
    let pa: Vec<[u64; 8]> = ab.iter().map(|x| planes(x.mag)).collect();
    let pb: Vec<[u64; 8]> = bb.iter().map(|x| planes(x.mag)).collect();
    let neg: Vec<u64> = ab
        .iter()
        .zip(&bb)
        .map(|(a, b)| a.positive ^ b.positive)
        .collect();

    let b_e2e = || {
        let mut acc = AccB::new();
        for (a, b) in ab.iter().zip(&bb) {
            check(*a, *b).unwrap();
            acc.step(*a, *b);
        }
        acc.finish()
    };
    let b_ari = || {
        let mut acc = AccB::new();
        for (a, (ng, l, t)) in ab.iter().zip(&split) {
            acc.mac(a.mag, *ng, *l, *t);
        }
        acc.finish()
    };
    let b_scan = || {
        let qb = QueryB::new(q);
        let mut acc = AccB::new();
        for x in &ab {
            check(*x, q).unwrap();
            acc.scan_step(*x, &qb);
        }
        acc.finish()
    };
    let base = || {
        let w0 = I8x64::from_array(rb[0].0.map(|x| i8::from_ne_bytes([x])));
        let mut acc = I32x16::splat(0);
        let mut total = 0i64;
        for (k, a) in ab.iter().enumerate() {
            acc = baseline_step(acc, a.mag, w0);
            if k % B_FLUSH_BLOCKS == B_FLUSH_BLOCKS - 1 {
                total += lanes_i64(acc);
                acc = I32x16::splat(0);
            }
        }
        total + lanes_i64(acc)
    };

    let c_e2e_abacus = || {
        let mut acc = AccAbacus::new();
        for (a, b) in ab.iter().zip(&bb) {
            check(*a, *b).unwrap();
            acc.mac(&planes(a.mag), a.positive ^ b.positive, &planes(b.mag));
        }
        acc.finish()
    };
    let c_e2e_pop = || {
        let mut acc = AccPopcnt::new();
        for (a, b) in ab.iter().zip(&bb) {
            check(*a, *b).unwrap();
            acc.mac(&planes(a.mag), a.positive ^ b.positive, &planes(b.mag));
        }
        acc.finish()
    };
    let c_ari_abacus = || {
        let mut acc = AccAbacus::new();
        for ((x, y), ng) in pa.iter().zip(&pb).zip(&neg) {
            acc.mac(x, *ng, y);
        }
        acc.finish()
    };
    let c_ari_abacus_norm_each = || {
        // Deferred carries normalized after every block instead of once.
        let mut total = 0i64;
        for ((x, y), ng) in pa.iter().zip(&pb).zip(&neg) {
            let mut acc = AccAbacus::new();
            acc.mac(x, *ng, y);
            total += acc.finish();
        }
        total
    };
    let c_ari_csa = || {
        let mut acc = AccCsa::new();
        for ((x, y), ng) in pa.iter().zip(&pb).zip(&neg) {
            acc.mac(x, *ng, y);
        }
        acc.finish()
    };
    let c_e2e_csa = || {
        let mut acc = AccCsa::new();
        for (a, b) in ab.iter().zip(&bb) {
            check(*a, *b).unwrap();
            acc.mac(&planes(a.mag), a.positive ^ b.positive, &planes(b.mag));
        }
        acc.finish()
    };
    let c_scan_csa = || {
        let qc = QueryC::new(q.mag, q.positive);
        let mut acc = AccCsa::new();
        for x in &ab {
            check(*x, q).unwrap();
            acc.mac(&planes(x.mag), x.positive ^ qc.positive, &qc.planes);
        }
        acc.finish()
    };
    let c_ari_pop = || {
        let mut acc = AccPopcnt::new();
        for ((x, y), ng) in pa.iter().zip(&pb).zip(&neg) {
            acc.mac(x, *ng, y);
        }
        acc.finish()
    };
    let c_prep = || {
        let mut s = 0u64;
        for (a, b) in ab.iter().zip(&bb) {
            let (x, y) = (planes(a.mag), planes(b.mag));
            s = s.wrapping_add(x[0] ^ x[7] ^ y[3] ^ y[5]);
        }
        s as i64
    };
    let c_scan_abacus = || {
        let qc = QueryC::new(q.mag, q.positive);
        let mut acc = AccAbacus::new();
        for x in &ab {
            check(*x, q).unwrap();
            acc.mac(&planes(x.mag), x.positive ^ qc.positive, &qc.planes);
        }
        acc.finish()
    };
    let c_scan_pop = || {
        let qc = QueryC::new(q.mag, q.positive);
        let mut acc = AccPopcnt::new();
        for x in &ab {
            check(*x, q).unwrap();
            acc.mac(&planes(x.mag), x.positive ^ qc.positive, &qc.planes);
        }
        acc.finish()
    };

    // D: A' products with the f32 detour replaced by a 16-bit carry column.
    let routed: Vec<(U8x64, U8x64)> = ab.iter().zip(&bb).map(|(a, b)| route_a(*a, *b)).collect();
    let bw: Vec<(U16x32, U16x32)> = bb
        .iter()
        .map(|b| (U16x32::from_u8x64_lo(b.mag), U16x32::from_u8x64_hi(b.mag)))
        .collect();
    let a2_e2e = || {
        let mut acc = AccA::new();
        for (a, b) in ab.iter().zip(&bb) {
            check(*a, *b).unwrap();
            acc.step_mullo(*a, *b);
        }
        acc.finish()
    };
    let a2_ari = || {
        let mut acc = AccA::new();
        for ((p, q), b) in routed.iter().zip(&bb) {
            acc.mac_mullo(*p, *q, b.mag);
        }
        acc.finish()
    };
    fn d_e2e<const X: bool>(ab: &[Block], bb: &[Block]) -> i64 {
        let mut acc = AccD::<X>::new();
        for (a, b) in ab.iter().zip(bb) {
            check(*a, *b).unwrap();
            acc.step(*a, *b);
        }
        acc.finish()
    }
    fn d_ari<const X: bool>(routed: &[(U8x64, U8x64)], bw: &[(U16x32, U16x32)]) -> i64 {
        let mut acc = AccD::<X>::new();
        for ((p, q), (lo, hi)) in routed.iter().zip(bw) {
            acc.mac(*p, *q, *lo, *hi);
        }
        acc.finish()
    }
    fn d_scan<const X: bool>(ab: &[Block], q: Block) -> i64 {
        let qd = QueryD::new(q);
        let mut acc = AccD::<X>::new();
        for x in ab {
            check(*x, q).unwrap();
            acc.scan_step(*x, &qd);
        }
        acc.finish()
    }
    let dr_e2e = || d_e2e::<false>(&ab, &bb);
    let da_e2e = || d_e2e::<true>(&ab, &bb);
    let dr_ari = || d_ari::<false>(&routed, &bw);
    let da_ari = || d_ari::<true>(&routed, &bw);
    let dr_scan = || d_scan::<false>(&ab, q);
    let da_scan = || d_scan::<true>(&ab, q);

    for (name, f, w) in [
        ("A' mullo", &a2_e2e as &dyn Fn() -> i64, want),
        ("A' mullo arith", &a2_ari, want),
        ("D react", &dr_e2e, want),
        ("D anticipate", &da_e2e, want),
        ("D react arith", &dr_ari, want),
        ("D anticipate arith", &da_ari, want),
        ("D react scan", &dr_scan, want_scan),
        ("D anticipate scan", &da_scan, want_scan),
    ] {
        assert_eq!(f(), w, "{name} disagrees with the oracle");
    }
    for (name, f, w) in [
        ("B", &b_e2e as &dyn Fn() -> i64, want),
        ("B arith", &b_ari, want),
        ("C abacus", &c_e2e_abacus, want),
        ("C popcnt", &c_e2e_pop, want),
        ("C abacus arith", &c_ari_abacus, want),
        ("C abacus norm each", &c_ari_abacus_norm_each, want),
        ("C popcnt arith", &c_ari_pop, want),
        ("C csa arith", &c_ari_csa, want),
        ("C csa", &c_e2e_csa, want),
        ("C csa scan", &c_scan_csa, want_scan),
        ("B scan", &b_scan, want_scan),
        ("C abacus scan", &c_scan_abacus, want_scan),
        ("C popcnt scan", &c_scan_pop, want_scan),
    ] {
        assert_eq!(f(), w, "{name} disagrees with the oracle");
    }

    println!("-- {label}: {n} blocks of 64 values per operand");
    time("plain u8 x i8 dot4 against one fixed query (baseline)", n, runs, None, &base);
    let be = time("B end-to-end: check + AND/XOR split + 4 masked dot4", n, runs, None, &b_e2e);
    time("C abacus end-to-end: check + planes(a,b) + CSA columns", n, runs, Some(be), &c_e2e_abacus);
    time("C popcnt end-to-end: check + planes(a,b) + 128 popcounts", n, runs, Some(be), &c_e2e_pop);
    time("C csa end-to-end: check + planes(a,b) + static CSA schedule", n, runs, Some(be), &c_e2e_csa);
    let ba = time("B arithmetic only (split operands given)", n, runs, None, &b_ari);
    let ca = time("C abacus arithmetic only (planes given), deferred", n, runs, Some(ba), &c_ari_abacus);
    time("C abacus arithmetic only, normalized every block", n, runs, Some(ba), &c_ari_abacus_norm_each);
    let cp = time("C popcnt arithmetic only (planes given)", n, runs, Some(ba), &c_ari_pop);
    let cc = time("C csa arithmetic only (planes given), static schedule", n, runs, Some(ba), &c_ari_csa);
    let prep = time("C plane extraction alone, both operands (cold)", n, runs, None, &c_prep);
    let bs = time("B scan: query split once", n, runs, None, &b_scan);
    time("C abacus scan: query planes once, index planes per block", n, runs, Some(bs), &c_scan_abacus);
    time("C popcnt scan: query planes once, index planes per block", n, runs, Some(bs), &c_scan_pop);
    time("C csa scan: query planes once, index planes per block", n, runs, Some(bs), &c_scan_csa);
    println!("   D: A' products, 16-bit place-value column with carry (no f32)");
    time("A' end-to-end: route + mullo + f32 add (#388)", n, runs, Some(be), &a2_e2e);
    time("D react end-to-end: route + mullo + add, carry after", n, runs, Some(be), &dr_e2e);
    time("D anticipate end-to-end: route + mullo + add, carry before", n, runs, Some(be), &da_e2e);
    time("A' arithmetic only (routed operands given)", n, runs, Some(ba), &a2_ari);
    time("D react arithmetic only (routed, b widened given)", n, runs, Some(ba), &dr_ari);
    time("D anticipate arithmetic only (routed, b widened given)", n, runs, Some(ba), &da_ari);
    time("D react scan: query widened once", n, runs, Some(bs), &dr_scan);
    time("D anticipate scan: query widened once", n, runs, Some(bs), &da_scan);
    let best_c = ca.min(cp).min(cc);
    if best_c >= be {
        println!(
            "break-even: none. C with both operands' planes prepared for free ({best_c:.3}) is slower than B paying all its preparation ({be:.3}); extraction alone costs {prep:.3}."
        );
    } else {
        println!("break-even: C arithmetic {best_c:.3} < B end-to-end {be:.3}; extraction {prep:.3} per use");
    }
}

// Out-of-line copies for the disassembly (`objdump -d --no-show-raw-insn`).
#[inline(never)]
pub fn asm_carry_save3(a: u64, b: u64, c: u64) -> (u64, u64) {
    carry_save3(a, b, c)
}
#[inline(never)]
pub fn asm_planes(mag: U8x64) -> [u64; 8] {
    planes(mag)
}
#[inline(never)]
pub fn asm_abacus_mac(acc: &mut AccAbacus, a: &[u64; 8], negative: u64, b: &[u64; 8]) {
    acc.mac(a, negative, b)
}
#[inline(never)]
pub fn asm_csa_mac(acc: &mut AccCsa, a: &[u64; 8], negative: u64, b: &[u64; 8]) {
    acc.mac(a, negative, b)
}
#[inline(never)]
pub fn asm_d_react(acc: &mut AccD<false>, a_pos: U8x64, a_neg: U8x64, lo: U16x32, hi: U16x32) {
    acc.mac(a_pos, a_neg, lo, hi)
}
#[inline(never)]
pub fn asm_d_anticipate(acc: &mut AccD<true>, a_pos: U8x64, a_neg: U8x64, lo: U16x32, hi: U16x32) {
    acc.mac(a_pos, a_neg, lo, hi)
}
#[inline(never)]
pub fn asm_popcnt_mac(acc: &mut AccPopcnt, a: &[u64; 8], negative: u64, b: &[u64; 8]) {
    acc.mac(a, negative, b)
}

fn main() {
    println!("tier: {} (target-cpu as compiled)", tier());
    {
        // Keep the out-of-line copies in the binary.
        let p = black_box(asm_planes(black_box(U8x64::splat(0xA5))));
        let z = black_box(U16x32::splat(7));
        let mut dr = AccD::<false>::new();
        asm_d_react(&mut dr, black_box(U8x64::splat(3)), black_box(U8x64::splat(4)), z, z);
        let mut da = AccD::<true>::new();
        asm_d_anticipate(&mut da, black_box(U8x64::splat(3)), black_box(U8x64::splat(4)), z, z);
        black_box((dr.finish(), da.finish()));
        let mut s = AccCsa::new();
        asm_csa_mac(&mut s, &p, black_box(0x55), &p);
        black_box(s.finish());
        let mut a = AccAbacus::new();
        asm_abacus_mac(&mut a, &p, black_box(0x55), &p);
        let mut c = AccPopcnt::new();
        asm_popcnt_mac(&mut c, &p, black_box(0x55), &p);
        black_box((asm_carry_save3(black_box(1), black_box(2), black_box(3)), a.finish(), c.finish()));
    }
    bench("L1-resident, repeated 4096-value dot", 64, 2001, 1);
    bench("streamed", 1 << 20, 5, 2);
}
