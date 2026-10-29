//! U9 x U9 head-to-head through the `ndarray::simd` facade: candidate A
//! (U8 x U8, sign routed, `mul_add`) against candidate B (U8 x I8 byte-dot on
//! a signed 7+1 split), plus the plain u8 x i8 byte-dot baseline.
//!
//! Every kernel is checked against the `i64` oracle on the same data before
//! it is timed. `cargo run --release --example u9_head_to_head --features std`;
//! the tier is whatever this build compiled (printed in the header).

mod kernels;

use kernels::*;
use ndarray::simd::{I32x16, I8x64, U8x64};
use std::hint::black_box;
use std::time::Instant;

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512vnni") {
        "x86_64 AVX-512 + VNNI"
    } else if cfg!(target_feature = "avx512bw") {
        "x86_64 AVX-512 (no VNNI)"
    } else if cfg!(target_feature = "avx2") {
        "x86_64 AVX2"
    } else if cfg!(target_feature = "avx") {
        "x86_64 AVX"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64 NEON"
    } else {
        "scalar"
    }
}

struct Data {
    a: Vec<([u8; 64], u64)>,
    b: Vec<([u8; 64], u64)>,
}

/// Uniform over the 511 numeric values; never-touched never appears.
fn data(blocks: usize, seed: u64) -> Data {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as u32
    };
    let mut mk = || {
        let mut m = [0u8; 64];
        let mut p = 0u64;
        for (i, mi) in m.iter_mut().enumerate() {
            let v = (next() % 511) as i32 - 255;
            *mi = v.unsigned_abs() as u8;
            p |= u64::from(v >= 0) << i;
        }
        (m, p)
    };
    let a = (0..blocks).map(|_| mk()).collect();
    let b = (0..blocks).map(|_| mk()).collect();
    Data { a, b }
}

/// Median of `runs` timings of `f` over `blocks` blocks, in ns per 64 values.
fn time(name: &str, blocks: usize, runs: usize, baseline: Option<f64>, f: &dyn Fn() -> i64) -> f64 {
    black_box(f());
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let t0 = Instant::now();
            black_box(f());
            t0.elapsed().as_nanos() as f64 / blocks as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    let (min, med) = (t[0], t[runs / 2]);
    let rel = baseline
        .map(|b| format!("  {:5.2}x baseline", med / b))
        .unwrap_or_default();
    println!("{name:<52} {med:7.3} ns/64 (min {min:7.3}){rel}");
    med
}

fn bench(label: &str, d: &Data, runs: usize) {
    let n = d.a.len();
    let blocks = |v: &[([u8; 64], u64)]| -> Vec<Block> {
        v.iter()
            .map(|(m, p)| Block {
                mag: U8x64::from_array(*m),
                positive: *p,
            })
            .collect()
    };
    let (ab, bb) = (blocks(&d.a), blocks(&d.b));
    let want = oracle(&d.a, &d.b).unwrap();

    // Baseline operand: the b magnitudes reinterpreted as signed bytes are not
    // the same numbers, but the byte-dot costs the same for any bytes. Built
    // once, outside the timed region, because the baseline has no preparation.
    let w: Vec<I8x64> =
        d.b.iter()
            .map(|(m, _)| I8x64::from_array(m.map(|x| i8::from_ne_bytes([x]))))
            .collect();
    // Execution-form operands for the arithmetic-only rows, also built once.
    let routed: Vec<(U8x64, U8x64)> = ab.iter().zip(&bb).map(|(a, b)| route_a(*a, *b)).collect();
    let split: Vec<(I8x64, I8x64)> = ab.iter().zip(&bb).map(|(a, b)| prepare_b(*a, *b)).collect();

    let e2e_a = || {
        let mut acc = AccA::new();
        for (a, b) in ab.iter().zip(&bb) {
            check(*a, *b).unwrap();
            acc.step(*a, *b);
        }
        acc.finish()
    };
    let e2e_a2 = || {
        let mut acc = AccA::new();
        for (a, b) in ab.iter().zip(&bb) {
            check(*a, *b).unwrap();
            acc.step_mullo(*a, *b);
        }
        acc.finish()
    };
    let e2e_b = || {
        let mut acc = AccB::new();
        for (a, b) in ab.iter().zip(&bb) {
            check(*a, *b).unwrap();
            acc.step(*a, *b);
        }
        acc.finish()
    };
    let ari_a = || {
        let mut acc = AccA::new();
        for ((p, q), b) in routed.iter().zip(&bb) {
            acc.mac(*p, *q, b.mag);
        }
        acc.finish()
    };
    let ari_a2 = || {
        let mut acc = AccA::new();
        for ((p, q), b) in routed.iter().zip(&bb) {
            acc.mac_mullo(*p, *q, b.mag);
        }
        acc.finish()
    };
    let ari_b = || {
        let mut acc = AccB::new();
        for (a, (lo, hi)) in ab.iter().zip(&split) {
            acc.mac(a.mag, *lo, *hi);
        }
        acc.finish()
    };
    let base = || {
        let mut acc = I32x16::splat(0);
        let mut total = 0i64;
        for (k, (a, w)) in ab.iter().zip(&w).enumerate() {
            acc = baseline_step(acc, a.mag, *w);
            if k % B_FLUSH_BLOCKS == B_FLUSH_BLOCKS - 1 {
                total += i64::from(acc.reduce_sum());
                acc = I32x16::splat(0);
            }
        }
        total + i64::from(acc.reduce_sum())
    };
    let prep_b = || {
        let mut x = 0i64;
        for (a, b) in ab.iter().zip(&bb) {
            let (lo, hi) = prepare_b(*a, *b);
            x = x.wrapping_add(i64::from(lo.to_array()[0]) ^ i64::from(hi.to_array()[63]));
        }
        x
    };
    // Scan forms: operand b plays the query. One query block is reused for
    // every index block, as in a search; its preparation runs once per call
    // and is inside the timed region.
    let q = bb[0];
    let scan_a = || {
        let qa = QueryA::new(q);
        let mut acc = AccA::new();
        for x in &ab {
            check(*x, q).unwrap();
            acc.scan_step(*x, &qa);
        }
        acc.finish()
    };
    let scan_b = || {
        let qb = QueryB::new(q);
        let mut acc = AccBScan::new();
        for x in &ab {
            check(*x, q).unwrap();
            acc.scan_step(*x, &qb);
        }
        acc.finish()
    };
    let qs: Vec<_> = vec![d.b[0]; n];
    let want_scan = oracle(&d.a, &qs).unwrap();
    assert_eq!(scan_a(), want_scan, "A scan disagrees with the oracle");
    assert_eq!(scan_b(), want_scan, "B scan disagrees with the oracle");
    for (name, f) in [
        ("A", &e2e_a as &dyn Fn() -> i64),
        ("A mullo", &e2e_a2),
        ("B", &e2e_b),
        ("A arith", &ari_a),
        ("A mullo arith", &ari_a2),
        ("B arith", &ari_b),
    ] {
        assert_eq!(f(), want, "{name} disagrees with the oracle on the benchmark data");
    }

    println!("-- {label}: {n} blocks of 64 values per operand ({} KiB per operand incl. plane)", n * 72 / 1024);
    let b0 = time("baseline: plain u8 x i8 dot4 (one byte-dot)", n, runs, None, &base);
    time("A end-to-end: check + route + widen + mul_add", n, runs, Some(b0), &e2e_a);
    time("A' end-to-end: check + route + mullo + add", n, runs, Some(b0), &e2e_a2);
    time("B end-to-end: check + split + sign + i8 + 2 dot4", n, runs, Some(b0), &e2e_b);
    time("A arithmetic only (routed operands given)", n, runs, Some(b0), &ari_a);
    time("A' arithmetic only (routed operands given)", n, runs, Some(b0), &ari_a2);
    time("B arithmetic only (signed i8 operands given)", n, runs, Some(b0), &ari_b);
    time("B operand preparation alone (split + sign + i8)", n, runs, Some(b0), &prep_b);
    let base_scan = || {
        let w0 = w[0];
        let mut acc = I32x16::splat(0);
        let mut total = 0i64;
        for (k, a) in ab.iter().enumerate() {
            acc = baseline_step(acc, a.mag, w0);
            if k % B_FLUSH_BLOCKS == B_FLUSH_BLOCKS - 1 {
                total += i64::from(acc.reduce_sum());
                acc = I32x16::splat(0);
            }
        }
        total + i64::from(acc.reduce_sum())
    };
    let bs = time("baseline scan: plain dot4, one fixed query", n, runs, None, &base_scan);
    time("A scan: query widened once, index routed + mul_add", n, runs, Some(bs), &scan_a);
    time("B scan: query signed once, 4 masked dot4 (#387)", n, runs, Some(bs), &scan_b);
}

fn main() {
    println!("tier: {} (target-cpu as compiled)", tier());
    // 64 blocks = 4.5 KiB per operand: L1-resident. One dot product of 4096
    // values, repeated.
    bench("L1-resident, repeated 4096-value dot", &data(64, 1), 4001);
    // 2^20 blocks = 72 MiB per operand: streamed from DRAM.
    bench("streamed", &data(1 << 20, 2), 9);
}
