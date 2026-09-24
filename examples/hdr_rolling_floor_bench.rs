//! Cost split of the HDR rolling floor: the per-observation hot path versus
//! the periodic shape path.
//!
//! ```sh
//! cargo run --release --example hdr_rolling_floor_bench
//! ```
//!
//! Hot path, per observation:
//!   1. popcount / Hamming distance of two 2048-byte vectors
//!   2. exact moments update (`MomentsU32::observe`, and `moments_u32` batch)
//!   3. reservoir update
//!   4. full rolling-floor update (moments + reservoir + checkpoint test)
//!
//! Periodic path, once per 1000 observations:
//!   5. shape evaluation (sort 1000 samples, median, kurtosis)
//!   6. empirical shape: locate 8 σ-lattice levels
//!
//! Query path, on demand (nothing is stored):
//!   7. Gaussian thresholds of 8 levels
//!   8. shade of a response over 8 levels

use ndarray::hpc::bitwise::hamming_distance_raw;
use ndarray::hpc::rolling_floor::{quantile_of_sorted, EmpiricalShape, ReservoirU32, RollingFloor, SigmaLevel};
use ndarray::hpc::statistics::{moments_u32, MomentsU32};
use std::hint::black_box;
use std::time::Instant;

fn xorshift(n: usize, mut s: u64) -> Vec<u64> {
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        })
        .collect()
}

fn ns_per(label: &str, n: usize, f: impl FnOnce()) {
    let t = Instant::now();
    f();
    let dt = t.elapsed();
    println!("{label:<44} {:>9.2} ns/op  ({n} ops)", dt.as_nanos() as f64 / n as f64);
}

fn main() {
    println!("avx512f={} avx2={}", cfg!(target_feature = "avx512f"), cfg!(target_feature = "avx2"));

    const VBYTES: usize = 2048; // 16384-bit vectors
    const N: usize = 2_000_000;
    let words = xorshift(VBYTES / 8 * 65, 1);
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let query = &bytes[..VBYTES];
    let db = &bytes[VBYTES..];
    let dists: Vec<u32> = (0..N)
        .map(|i| {
            let j = i % 64;
            hamming_distance_raw(query, &db[j * VBYTES..(j + 1) * VBYTES]) as u32
        })
        .collect();

    println!("-- hot path, per observation --");
    ns_per("1. popcount/Hamming (2048 B)", N, || {
        let mut acc = 0u64;
        for i in 0..N {
            let j = i % 64;
            acc += hamming_distance_raw(black_box(query), &db[j * VBYTES..(j + 1) * VBYTES]);
        }
        black_box(acc);
    });
    ns_per("2a. MomentsU32::observe (scalar)", N, || {
        let mut m = MomentsU32::default();
        for &d in &dists {
            m.observe(black_box(d));
        }
        black_box(m);
    });
    ns_per("2b. moments_u32 (batch)", N, || {
        black_box(moments_u32(black_box(&dists)));
    });
    ns_per("3. ReservoirU32::observe (cap 1000)", N, || {
        let mut r = ReservoirU32::new(1000);
        for &d in &dists {
            r.observe(black_box(d));
        }
        black_box(r.len());
    });
    ns_per("4a. RollingFloor::observe (incl. checkpoints)", N, || {
        let mut f = RollingFloor::for_width(16384);
        for &d in &dists {
            if let Some(s) = f.observe(black_box(d)) {
                f.recalibrate(&s);
            }
        }
        black_box(f.mu());
    });
    ns_per("4b. RollingFloor::observe_batch (incl. checkpoints)", N, || {
        let mut f = RollingFloor::for_width(16384);
        let mut rest: &[u32] = &dists;
        while !rest.is_empty() {
            let (used, s) = f.observe_batch(rest);
            rest = &rest[used..];
            if let Some(s) = s {
                f.recalibrate(&s);
            }
        }
        black_box(f.mu());
    });

    println!("-- periodic path, per checkpoint (every 1000 observations) --");
    let mut r = ReservoirU32::new(1000);
    dists[..5000].iter().for_each(|&d| r.observe(d));
    const K: usize = 20_000;
    ns_per("5. shape: sort + median + kurtosis", K, || {
        for _ in 0..K {
            let sorted = black_box(&r).sorted();
            black_box(quantile_of_sorted(&sorted, 5000));
            black_box(r.kurtosis(8192, 64));
        }
    });
    let lattice = [4u8, 6, 7, 8, 9, 10, 11, 12].map(SigmaLevel);
    let shape = EmpiricalShape::from_sample(r.samples()).unwrap();
    ns_per("6. empirical: locate 8 lattice levels", K, || {
        for _ in 0..K {
            black_box(lattice.map(|l| black_box(&shape).locate(l, 8200, 70)));
        }
    });

    println!("-- query path, on demand --");
    let mut g = RollingFloor::for_width(16384);
    dists[..3000].iter().for_each(|&d| {
        if let Some(s) = g.observe(d) {
            g.recalibrate(&s);
        }
    });
    ns_per("7. Gaussian: thresholds of 8 levels", N, || {
        for _ in 0..N {
            black_box(black_box(&g).thresholds(&lattice));
        }
    });
    ns_per("8. Gaussian: shade over 8 levels", N, || {
        let mut acc = 0usize;
        for &d in &dists {
            acc += black_box(&g).shade(d, &lattice);
        }
        black_box(acc);
    });
}
