//! W4 — PSD at depth-infinity (census M-3).
//!
//! ndarray's `prove_pillar_11` certifies the TRUNCATED (d=2, deg-3) kernel's
//! stability. The property kernel machines actually rely on — PSD-ness of the
//! Gram — is uncertified for the depth-infinity kernel everywhere. This leg
//! re-runs that machinery over `sigker::signature_kernel_pde` values on the
//! SAME Brownian pool, per ruling Q2 (cross-repo call, not a second f32
//! Goursat port; a port waits for W5's trigger).
//!
//! # Strengthened relative to the truncated battery
//!
//! `prove_pillar_11`'s "PSD" criteria are diagonal positivity plus
//! Cauchy-Schwarz. Both are NECESSARY conditions, neither is sufficient: a
//! matrix can satisfy them and still have a negative eigenvalue. This leg
//! adds the sufficient test — a Cholesky factorization, which exists iff the
//! matrix is positive definite — and pairs it with the falsifier that the
//! weaker criteria cannot supply (see `a_non_psd_gram_is_rejected`).
//!
//! # Tolerance
//!
//! The Gram is built in f64 from a first-order solver, so the diagonal is not
//! exact; Cholesky is run with a relative jitter of `JITTER` times the mean
//! diagonal, the standard numerical allowance. The falsifier below confirms
//! that this jitter does not swallow a genuinely indefinite matrix.

use ndarray::hpc::pillar::signature::brownian_path_d2;
use ndarray::hpc::pillar::SplitMix64;
use sigker::signature_kernel_pde;

/// Same seed as the truncated battery, so both certify the same pool.
const PILLAR_11_SEED: u64 = 0x5EED_1111_5164_A7AB;
/// Gram pool — O(N^2) kernel solves, so kept small.
const N_PATHS: usize = 64;
/// Concentration pool — O(N) solves, matched to the truncated battery's 1000
/// so the two numbers are directly comparable (see that leg's doc).
const N_CONC: usize = 1000;
const N_STEPS: usize = 50;
const JITTER: f64 = 1e-9;

/// The f32 hardware path buffer, in sigker's point-list shape.
fn as_points(flat: &[f32], n_points: usize) -> Vec<Vec<f64>> {
    (0..n_points)
        .map(|k| vec![flat[2 * k] as f64, flat[2 * k + 1] as f64])
        .collect()
}

fn brownian_pool(seed: u64, n_paths: usize) -> Vec<Vec<Vec<f64>>> {
    let mut rng = SplitMix64::new(seed);
    (0..n_paths)
        .map(|_| {
            let p = brownian_path_d2(&mut rng, N_STEPS);
            as_points(&p, N_STEPS + 1)
        })
        .collect()
}

fn gram(pool: &[Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
    let n = pool.len();
    let mut k = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let v = signature_kernel_pde(&pool[i], &pool[j]);
            k[i][j] = v;
            k[j][i] = v; // the kernel is symmetric by construction
        }
    }
    k
}

/// Cholesky with relative jitter. `Some(_)` iff the matrix is positive
/// definite to that tolerance — the SUFFICIENT PSD test the truncated
/// battery lacks.
fn cholesky(k: &[Vec<f64>], jitter: f64) -> Option<usize> {
    let n = k.len();
    let mean_diag = (0..n).map(|i| k[i][i]).sum::<f64>() / n as f64;
    let eps = jitter * mean_diag.abs().max(1.0);
    let mut l = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = k[i][j];
            if i == j {
                s += eps;
            }
            // Dot of the two already-computed row prefixes. Both borrows are
            // immutable and end before the write below.
            let dot: f64 = l[i][..j].iter().zip(&l[j][..j]).map(|(a, b)| a * b).sum();
            s -= dot;
            if i == j {
                if s <= 0.0 {
                    return Some(i); // failed at this leading minor
                }
                l[i][i] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }
    None
}

#[test]
fn the_depth_infinity_gram_is_positive_definite() {
    let pool = brownian_pool(PILLAR_11_SEED, N_PATHS);
    let k = gram(&pool);

    // Necessary conditions first — the same two the truncated battery uses,
    // so a failure here is directly comparable to that report.
    let mut cs_violations = 0usize;
    for i in 0..N_PATHS {
        assert!(k[i][i] > 0.0, "depth-inf self-kernel K[{i},{i}] = {} is not positive", k[i][i]);
        for j in i + 1..N_PATHS {
            if k[i][j] * k[i][j] > k[i][i] * k[j][j] * 1.001 {
                cs_violations += 1;
            }
        }
    }
    assert_eq!(cs_violations, 0, "Cauchy-Schwarz violated at depth-infinity");

    // The sufficient one.
    assert!(
        cholesky(&k, JITTER).is_none(),
        "depth-infinity Gram is NOT positive definite (Cholesky failed at \
         leading minor {:?}) — M-3 would be falsified",
        cholesky(&k, JITTER)
    );
    println!("W4: depth-inf Gram over {N_PATHS} Brownian paths — diag > 0, Cauchy-Schwarz clean, Cholesky OK");
}

/// Concentration is measured at `N_CONC`, NOT at `N_PATHS`.
///
/// The 0.20 bound comes from the truncated battery, which runs 1000 paths.
/// Applying it to the 64-path Gram pool would silently be a DIFFERENT gate —
/// measured, at N = 64 the truncated kernel itself concentrates to 0.3461 and
/// would fail its own bound. Half-mean agreement is a sample-size statistic
/// before it is a kernel property, so the comparison is only meaningful at
/// matched N (`examples/w4_concentration_sweep.rs`):
///
/// ```text
///   depth-INFINITY          depth-3 TRUNCATED
///        N  concentr             N  concentr
///       64    0.2892            64    0.3461
///      128    0.2156          1000    0.0481
///      256    0.1884
///      512    0.0053
///     1000    0.0038
/// ```
///
/// At matched N = 1000 the depth-infinity kernel concentrates BETTER than the
/// truncated one it extends (0.0038 vs 0.0481). The Gram/Cholesky leg above
/// stays at 64 paths because it costs O(N^2) solves; this leg is O(N).
#[test]
fn the_depth_infinity_self_kernel_concentrates() {
    let pool = brownian_pool(PILLAR_11_SEED, N_CONC);
    let self_k: Vec<f64> = pool.iter().map(|p| signature_kernel_pde(p, p)).collect();
    let concentration = half_mean_gap(&self_k);
    println!(
        "W4: depth-inf self-kernel concentration {concentration:.4} over {N_CONC} paths \
         (bound 0.20, the truncated battery's; truncated scores 0.0481 at the same N)"
    );
    assert!(
        concentration < 0.20,
        "depth-inf self-kernel half-means disagree by {concentration:.3} at N = {N_CONC}"
    );
}

/// Can-stay-silent's partner: a pool that genuinely does NOT concentrate must
/// be caught at the SAME N, so the green above is a property of the kernel
/// and not of the bound being loose. The fixture rescales the second half of
/// the pool, which is exactly the drift a half-mean statistic exists to see.
#[test]
fn a_non_concentrating_pool_is_caught() {
    let pool = brownian_pool(PILLAR_11_SEED, N_CONC);
    let self_k: Vec<f64> = pool
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i < N_CONC / 2 {
                signature_kernel_pde(p, p)
            } else {
                let scaled: Vec<Vec<f64>> = p
                    .iter()
                    .map(|pt| pt.iter().map(|v| v * 1.35).collect())
                    .collect();
                signature_kernel_pde(&scaled, &scaled)
            }
        })
        .collect();
    let concentration = half_mean_gap(&self_k);
    assert!(
        concentration >= 0.20,
        "ANTI-VACUITY FAILED: a pool whose second half is rescaled 1.35x still \
         concentrated to {concentration:.4} — the bound cannot see drift"
    );
    println!("W4 anti-vacuity: rescaled-half pool reaches concentration {concentration:.4} (bound 0.20)");
}

/// Relative gap between the two half-means — the truncated battery's statistic.
fn half_mean_gap(v: &[f64]) -> f64 {
    let h = v.len() / 2;
    let m1 = v[..h].iter().sum::<f64>() / h as f64;
    let m2 = v[h..].iter().sum::<f64>() / (v.len() - h) as f64;
    let combined = v.iter().sum::<f64>() / v.len() as f64;
    if combined > 0.0 {
        (m1 - m2).abs() / combined
    } else {
        f64::INFINITY
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Falsifier pair for the PSD gate.
// ════════════════════════════════════════════════════════════════════════════

/// Can-fire: an indefinite matrix of the same shape and scale must be
/// REJECTED. Without this, `cholesky` returning `None` would be evidence of
/// nothing — and it is precisely the test the truncated battery cannot run,
/// because diagonal-positivity + Cauchy-Schwarz both HOLD on this matrix.
#[test]
fn a_non_psd_gram_is_rejected() {
    let pool = brownian_pool(PILLAR_11_SEED, 8);
    let mut k = gram(&pool);
    // Flip the sign of one off-diagonal pair. Diagonals are untouched (still
    // positive) and the magnitudes are unchanged (so Cauchy-Schwarz still
    // holds) — only definiteness is destroyed.
    let scale = (k[0][0] * k[1][1]).sqrt();
    k[0][1] = -0.999 * scale;
    k[1][0] = -0.999 * scale;
    k[0][2] = 0.999 * (k[0][0] * k[2][2]).sqrt();
    k[2][0] = k[0][2];
    k[1][2] = 0.999 * (k[1][1] * k[2][2]).sqrt();
    k[2][1] = k[1][2];

    // The weak criteria still pass — that is the point.
    for i in 0..3 {
        assert!(k[i][i] > 0.0, "diagonal must stay positive in the fixture");
        for j in i + 1..3 {
            assert!(k[i][j] * k[i][j] <= k[i][i] * k[j][j] * 1.001, "Cauchy-Schwarz must still hold in the fixture");
        }
    }
    // The strong one must catch it.
    assert!(cholesky(&k, JITTER).is_some(), "ANTI-VACUITY FAILED: an indefinite Gram passed the Cholesky gate");
    println!(
        "W4 anti-vacuity: indefinite Gram passes diag>0 AND Cauchy-Schwarz, rejected by Cholesky at minor {:?}",
        cholesky(&k, JITTER)
    );
}
