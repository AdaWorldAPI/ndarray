//! Pre-registration: does the truncated Pillar-11 Gram survive an f32
//! Cholesky, or does it need f64 accumulation?
//!
//! `sigker_hl` returns f32. Cholesky on an f32 Gram can report `info > 0` on
//! a matrix that IS positive semi-definite but ill-conditioned — a false
//! alarm on the real battery. This measures the margin on the ACTUAL pool
//! before any gate is pinned.
use ndarray::hpc::lapack::LapackOps;
use ndarray::hpc::pillar::signature::PILLAR_11_SEED;
use ndarray::hpc::pillar::signature::{brownian_path_d2, sigker_hl, signature_d2_deg3};
use ndarray::hpc::pillar::SplitMix64;
use ndarray::Array2;

const N_STEPS: usize = 50;

fn sigs(n: usize) -> Vec<[f32; 15]> {
    let mut rng = SplitMix64::new(PILLAR_11_SEED);
    (0..n)
        .map(|_| {
            let p = brownian_path_d2(&mut rng, N_STEPS);
            signature_d2_deg3(&p, N_STEPS + 1)
        })
        .collect()
}

fn main() {
    println!(
        "{:>7} {:>12} {:>12} {:>14} {:>14} {:>12}",
        "subset", "f32 info", "f64 info", "min diag", "max |K_ij|", "cond-ish"
    );
    for &n in &[8usize, 12, 14, 15, 16, 17, 32, 50] {
        let s = sigs(n);
        let mut g32 = Array2::<f32>::zeros((n, n));
        let mut g64 = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let v = sigker_hl(&s[i], &s[j]);
                g32[[i, j]] = v;
                g64[[i, j]] = v as f64;
            }
        }
        let i32_ = g32.cholesky().info;
        let i64_ = g64.cholesky().info;
        let min_diag = (0..n).map(|i| g64[[i, i]]).fold(f64::INFINITY, f64::min);
        let max_off = (0..n)
            .flat_map(|i| (0..n).filter(move |j| *j != i).map(move |j| (i, j)))
            .map(|(i, j)| g64[[i, j]].abs())
            .fold(0.0f64, f64::max);
        let max_diag = (0..n).map(|i| g64[[i, i]]).fold(0.0f64, f64::max);
        println!("{n:>7} {i32_:>12} {i64_:>12} {min_diag:>14.4e} {max_off:>14.4e} {:>12.2e}", max_diag / min_diag);
    }
    println!("\ninfo == 0 means positive definite; info > 0 is the failing leading minor.");
}
