//! Bench for `hpc::randomized_signature::randomized_signature_sweep` — the
//! SIMD successor to lance-graph `sigker::randomized`'s scalar encode loop
//! (`TD-NDARRAY-SIMD-RANDOMIZED-PROJECTION`, W1.5 item #7).
//!
//! Reports the SIMD (GEMV + axpy on `F64x8`) speedup over the naive
//! row-major scalar recurrence across the state widths sigker's own tests and
//! performance envelope name (k = 32 … 512), plus the envelope's headline
//! shape: k = 4096, d = 8, T = 64 ("~8.6 GFLOPS per path", `randomized.rs`
//! module doc) — run SIMD-only, because the scalar baseline at that size
//! takes minutes.
//!
//! cargo run --release --example randomized_signature_bench

use ndarray::hpc::randomized_signature::randomized_signature_sweep;
use std::time::Instant;

/// SplitMix64 + Box-Muller — sigker's own generator, so the projections
/// benchmarked here are the projections the consumer would encode with.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn projections(d: usize, k: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let scale = (k as f64).recip().sqrt();
    let mut rng = SplitMix64(seed);
    let matrices = (0..d * k * k).map(|_| rng.normal() * scale).collect();
    let biases = (0..d * k).map(|_| rng.normal() * scale).collect();
    (matrices, biases)
}

fn path(t: usize, d: usize, seed: f64) -> Vec<Vec<f64>> {
    (0..=t)
        .map(|i| {
            let s = i as f64 / t as f64;
            (0..d)
                .map(|a| {
                    let phase = seed + a as f64 * 1.7;
                    s * (a as f64 + 1.0) + 0.05 * (37.0 * s + phase).cos()
                })
                .collect()
        })
        .collect()
}

/// Naive row-major scalar recurrence — the baseline the SIMD sweep is
/// measured against; a transcription of `sigker::randomized`'s `encode`.
fn scalar_encode(path: &[Vec<f64>], matrices: &[f64], biases: &[f64], k: usize) -> Vec<f64> {
    let d = path[0].len();
    let mut z = vec![0.0f64; k];
    for window in path.windows(2) {
        let delta_x: Vec<f64> = window[1]
            .iter()
            .zip(window[0].iter())
            .map(|(a, b)| a - b)
            .collect();
        let mut z_next = z.clone();
        let mut activated = vec![0.0f64; k];
        for (i, &dx_i) in delta_x.iter().enumerate().take(d) {
            if dx_i.abs() < 1e-15 {
                continue;
            }
            let a_offset = i * k * k;
            let b_offset = i * k;
            for row in 0..k {
                let mut sum = biases[b_offset + row];
                let row_off = a_offset + row * k;
                for col in 0..k {
                    sum += matrices[row_off + col] * z[col];
                }
                activated[row] = sum.tanh();
            }
            for row in 0..k {
                z_next[row] += activated[row] * dx_i;
            }
        }
        z = z_next;
    }
    z
}

fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs() / x.abs().max(1.0))
        .fold(0.0f64, f64::max)
}

fn bench_one(t: usize, d: usize, k: usize) {
    let (matrices, biases) = projections(d, k, 0xBEEF);
    let p = path(t, d, 0.3);

    let start = Instant::now();
    let scalar = scalar_encode(&p, &matrices, &biases, k);
    let s_scalar = start.elapsed().as_secs_f64();

    let start = Instant::now();
    let simd = randomized_signature_sweep(&p, &matrices, &biases, k);
    let s_simd = start.elapsed().as_secs_f64();

    println!(
        "T={t:<4} d={d:<2} k={k:<5} scalar={s_scalar:>9.4}s  simd={s_simd:>9.4}s  \
         speedup={:>6.2}x  max_rel_err={:>10.3e}",
        s_scalar / s_simd.max(1e-12),
        rel_err(&scalar, &simd)
    );
}

fn main() {
    println!("== randomized_signature_sweep — SIMD GEMV+axpy vs. row-major scalar ==\n");

    for &k in &[32usize, 64, 128, 256, 512] {
        bench_one(64, 8, k);
    }

    println!();
    // sigker's own stated performance envelope: k=4096, d=8, T=64.
    // SIMD-only — the scalar baseline at this size is minutes, not seconds.
    let (matrices, biases) = projections(8, 4096, 0xBEEF);
    let p = path(64, 8, 0.3);
    let start = Instant::now();
    let z = randomized_signature_sweep(&p, &matrices, &biases, 4096);
    let elapsed = start.elapsed().as_secs_f64();
    let gflops = (64.0 * 8.0 * 4096.0 * 4096.0 * 2.0) / elapsed / 1e9;
    println!(
        "sigker envelope shape: T=64 d=8 k=4096 -> simd = {elapsed:.4}s ({gflops:.1} GFLOP/s), \
         |z|_inf = {:.4}",
        z.iter().fold(0.0f64, |m, v| m.max(v.abs()))
    );
}
