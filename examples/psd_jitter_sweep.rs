//! Pick the PSD jitter by measurement, and prove it still discriminates.
//!
//! The truncated signature is a 15-dimensional feature map (1+2+4+8), so the
//! Gram of N > 15 paths is rank-deficient BY CONSTRUCTION and plain Cholesky
//! — which tests positive DEFINITE — must fail at leading minor 16. The
//! property the battery actually wants is positive SEMI-definite, i.e.
//! Cholesky with a relative jitter on the diagonal.
//!
//! A jitter large enough to admit the real (singular, PSD) Gram must still
//! REJECT a genuinely indefinite one. This sweeps both together; a jitter
//! that admits both is worthless.
use ndarray::hpc::lapack::LapackOps;
use ndarray::hpc::pillar::signature::PILLAR_11_SEED;
use ndarray::hpc::pillar::signature::{brownian_path_d2, sigker_hl, signature_d2_deg3};
use ndarray::hpc::pillar::SplitMix64;
use ndarray::Array2;

const N_STEPS: usize = 50;
const SUBSET: usize = 50;

fn real_gram(n: usize) -> Array2<f64> {
    let mut rng = SplitMix64::new(PILLAR_11_SEED);
    let s: Vec<[f32; 15]> = (0..n)
        .map(|_| {
            let p = brownian_path_d2(&mut rng, N_STEPS);
            signature_d2_deg3(&p, N_STEPS + 1)
        })
        .collect();
    let mut g = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            g[[i, j]] = sigker_hl(&s[i], &s[j]) as f64;
        }
    }
    g
}

/// A genuinely indefinite matrix that PASSES both weak criteria: every
/// diagonal positive, Cauchy-Schwarz satisfied for every pair.
fn indefinite_gram(n: usize) -> Array2<f64> {
    let mut g = real_gram(n);
    let s01 = (g[[0, 0]] * g[[1, 1]]).sqrt();
    let s02 = (g[[0, 0]] * g[[2, 2]]).sqrt();
    let s12 = (g[[1, 1]] * g[[2, 2]]).sqrt();
    g[[0, 1]] = -0.999 * s01;
    g[[1, 0]] = g[[0, 1]];
    g[[0, 2]] = 0.999 * s02;
    g[[2, 0]] = g[[0, 2]];
    g[[1, 2]] = 0.999 * s12;
    g[[2, 1]] = g[[1, 2]];
    g
}

fn chol_info(g: &Array2<f64>, jitter: f64) -> i32 {
    let n = g.nrows();
    let mean_diag = (0..n).map(|i| g[[i, i]]).sum::<f64>() / n as f64;
    let eps = jitter * mean_diag.abs().max(1.0);
    let mut j = g.clone();
    for i in 0..n {
        j[[i, i]] += eps;
    }
    j.cholesky().info
}

fn weak_criteria_hold(g: &Array2<f64>) -> (bool, bool) {
    let n = g.nrows();
    let diag_ok = (0..n).all(|i| g[[i, i]] > 0.0);
    let mut cs_ok = true;
    for i in 0..n {
        for j in i + 1..n {
            if g[[i, j]] * g[[i, j]] > g[[i, i]] * g[[j, j]] * 1.001 {
                cs_ok = false;
            }
        }
    }
    (diag_ok, cs_ok)
}

fn main() {
    let real = real_gram(SUBSET);
    let bad = indefinite_gram(SUBSET);

    let (rd, rc) = weak_criteria_hold(&real);
    let (bd, bc) = weak_criteria_hold(&bad);
    println!("weak criteria (diag>0, Cauchy-Schwarz):");
    println!("  real Gram       diag={rd} cs={rc}");
    println!("  indefinite Gram diag={bd} cs={bc}   <- both PASS, which is the point");

    println!("\n{:>12} {:>14} {:>18} {:>10}", "jitter", "real info", "indefinite info", "verdict");
    for &j in &[0.0f64, 1e-12, 1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] {
        let r = chol_info(&real, j);
        let b = chol_info(&bad, j);
        let verdict = match (r == 0, b == 0) {
            (true, false) => "USABLE",
            (false, false) => "too tight",
            (true, true) => "TOO LOOSE",
            (false, true) => "impossible?",
        };
        println!("{j:>12.0e} {r:>14} {b:>18} {verdict:>10}");
    }
}
