//! Bench for `hpc::signature_pde::signature_pde_sweep` — the general-
//! dimension successor to `jc`'s `goursat_substrate_probe` dim=2 falsifier.
//!
//! Reports the SIMD-wavefront speedup over the row-major scalar recurrence
//! at the probe's own shapes, plus the exact leg shape jc's Pillar-11
//! certification (Hambly-Lyons uniqueness) runs internally: 8 pairs of
//! length-4609 paths — the shape that motivated this primitive in the first
//! place (`TD-PILLAR11-SCIENTIFIC-LOOPS-BYPASS-NDARRAY-SIMD-1`).
//!
//! cargo run --release --example signature_pde_bench

use ndarray::hpc::signature_pde::signature_pde_sweep;
use std::time::Instant;

fn path(n: usize, dim: usize, seed: f64) -> Vec<Vec<f64>> {
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            (0..dim)
                .map(|a| {
                    let phase = seed + a as f64 * 1.7;
                    t * (a as f64 + 1.0) + 0.05 * (260.0 * t + phase).cos()
                })
                .collect()
        })
        .collect()
}

/// Row-major scalar reference — same recurrence, no SIMD, no anti-diagonal
/// reorder. The baseline the wavefront is measured against.
fn goursat_scalar(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    let (n, m) = (x.len(), y.len());
    let dim = x[0].len();
    let mut k = vec![1.0f64; n * m];
    for i in 0..n - 1 {
        for j in 0..m - 1 {
            let c: f64 = (0..dim)
                .map(|a| (x[i + 1][a] - x[i][a]) * (y[j + 1][a] - y[j][a]))
                .sum();
            let (left, up, diag) = (k[(i + 1) * m + j], k[i * m + j + 1], k[i * m + j]);
            k[(i + 1) * m + j + 1] = left + up - diag + c * diag;
        }
    }
    k[n * m - 1]
}

fn bench_one(label: &str, n: usize, dim: usize) {
    let (x, y) = (path(n, dim, 0.3), path(n, dim, 1.1));
    let t = Instant::now();
    let scalar = goursat_scalar(&x, &y);
    let s_scalar = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let simd = signature_pde_sweep(&x, &y);
    let s_simd = t.elapsed().as_secs_f64();
    let rel = ((scalar - simd) / scalar).abs();
    println!(
        "{label:<22} len={n:<6} dim={dim:<2} scalar={s_scalar:>9.4}s  simd={s_simd:>9.4}s  \
         speedup={:>6.2}x  rel_err={rel:>10.3e}",
        s_scalar / s_simd.max(1e-12)
    );
}

fn main() {
    println!("== signature_pde_sweep — SIMD wavefront vs. row-major scalar ==\n");

    for &n in &[256usize, 1024, 2048, 4096] {
        bench_one("probe shape (dim=2)", n, 2);
    }

    // The exact jc Pillar-11 leg shape: 8 pairs of length-4609 paths, dim=2
    // (the shape whose 25-26s cost motivated this primitive; see the W1.5
    // gate check in TD-PILLAR11-SCIENTIFIC-LOOPS-BYPASS-NDARRAY-SIMD-1).
    println!();
    let t_total = Instant::now();
    for pair in 0..8 {
        let (x, y) = (path(4608, 2, pair as f64), path(4608, 2, pair as f64 + 0.7));
        let _ = signature_pde_sweep(&x, &y);
    }
    let simd_total = t_total.elapsed().as_secs_f64();
    println!("jc Pillar-11 leg shape: 8 pairs x len=4609, dim=2 -> simd total = {simd_total:.4}s");

    // Higher-dimension sanity: dim=5, moderate length, proving the
    // generalization isn't free-riding on dim=2 special-casing.
    println!();
    bench_one("dim=5 sanity", 1024, 5);
}
