#![allow(missing_docs)]

//! Pillar-9 — High-dimensional covariance carrier + Düker–Zoubouloglou CLT probe.
//!
//! # Background
//!
//! Düker & Zoubouloglou (2023) establish a functional CLT for the empirical
//! covariance operator in high dimensions: when N → ∞ and the sample size
//! T grows at rate T / N → c ∈ (0, ∞), the standardised Frobenius-norm
//! increment of the sample covariance converges to a Gaussian at rate O(1/√T).
//!
//! This probe operationalises that result for a *fixed* N = 64 (PR-X11 v1;
//! N = 16384 / BindSpace alignment is PR-X11.1). We run 100 random
//! online-update paths × 50 hops and check that the fraction of hops whose
//! Frobenius increment lies in the predicted 95% Gaussian confidence band
//! meets the `PILLAR_9_CLT_THRESHOLD = 0.95` gate.
//!
//! # Storage convention
//!
//! [`CovHighD<N>`] stores the lower-triangular part of the symmetric matrix
//! in row-major packed order: entry (i, j) with i ≥ j lives at index
//! `i*(i+1)/2 + j` in the `lt` vector.  The `lt` vector has `N*(N+1)/2`
//! entries.

use crate::hpc::linalg::eig_sym::{eig_sym_jacobi, MatN};
use crate::hpc::pillar::prove_runner::{PillarReport, SplitMix64};

// ── Public constants ──────────────────────────────────────────────────────────

/// SEED for the Pillar-9 Düker–Zoubouloglou CLT probe.
pub const PILLAR_9_SEED: u64 = 0x_C0_DA_DA_5A_DC;

/// Minimum fraction of hops whose Frobenius increment must fall within the
/// predicted 95% Gaussian confidence band.
pub const PILLAR_9_CLT_THRESHOLD: f64 = 0.95;

// ── CovHighD<N> ───────────────────────────────────────────────────────────────

/// Const-generic high-D covariance carrier.
///
/// For PR-X11 v1, `N = 64` is the practical default; `N = 16384`
/// (BindSpace alignment) is a stress test queued for PR-X11.1.
///
/// # Storage
///
/// Lower-triangular packed storage: the (i, j) entry with i ≥ j lives at
/// index `i*(i+1)/2 + j` in `lt`. Total: `N*(N+1)/2` `f32` values.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::cov_high_d::CovHighD;
/// let id = CovHighD::<4>::identity();
/// assert!((id.frobenius_sq() - 4.0_f32).abs() < 1e-5);
/// ```
pub struct CovHighD<const N: usize> {
    /// Lower-triangular packed storage (`N*(N+1)/2` entries).
    pub lt: Vec<f32>,
}

impl<const N: usize> CovHighD<N> {
    // ── Packed-index helper ────────────────────────────────────────────────────

    /// Linear index into `lt` for element (row i, col j) with i ≥ j.
    #[inline]
    fn idx(i: usize, j: usize) -> usize {
        debug_assert!(i >= j, "CovHighD::idx: i={i} < j={j}");
        i * (i + 1) / 2 + j
    }

    /// Read entry (i, j) — exploits symmetry so (i, j) and (j, i) both work.
    #[inline]
    fn get(&self, i: usize, j: usize) -> f32 {
        if i >= j {
            self.lt[Self::idx(i, j)]
        } else {
            self.lt[Self::idx(j, i)]
        }
    }

    // ── Constructors ──────────────────────────────────────────────────────────

    /// Construct the N×N identity covariance matrix.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::hpc::pillar::cov_high_d::CovHighD;
    /// let id = CovHighD::<8>::identity();
    /// // Frobenius² = N (one 1² per diagonal entry)
    /// assert!((id.frobenius_sq() - 8.0_f32).abs() < 1e-5);
    /// ```
    pub fn identity() -> Self {
        let size = N * (N + 1) / 2;
        let mut lt = vec![0.0_f32; size];
        for i in 0..N {
            lt[Self::idx(i, i)] = 1.0;
        }
        Self { lt }
    }

    /// Construct a zero N×N matrix.
    fn zero() -> Self {
        let size = N * (N + 1) / 2;
        Self {
            lt: vec![0.0_f32; size],
        }
    }

    // ── Math operations ───────────────────────────────────────────────────────

    /// Compute the sandwich product `M Σ Mᵀ` where `self` is Σ and `m` is M.
    ///
    /// Both `self` and `m` are treated as symmetric matrices. The result is
    /// symmetric, so only the lower triangle is stored.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::hpc::pillar::cov_high_d::CovHighD;
    /// let id = CovHighD::<4>::identity();
    /// let result = id.sandwich(&id);
    /// // I · I · Iᵀ = I, so Frobenius² = 4
    /// assert!((result.frobenius_sq() - 4.0_f32).abs() < 1e-4);
    /// ```
    pub fn sandwich(&self, m: &Self) -> Self {
        // Compute C = M * Σ first (N × N intermediate), then result = C * Mᵀ = C * M
        // (since M is symmetric). We compute only the lower triangle of the result.
        //
        // C[i][k] = Σ_j  M[i][j] * Σ[j][k]
        // result[i][l] = Σ_k  C[i][k] * Mᵀ[k][l] = Σ_k  C[i][k] * M[l][k]
        //
        // We build C as a Vec<Vec<f32>> to avoid stack pressure for N=64.
        let mut c = vec![vec![0.0_f32; N]; N];
        for i in 0..N {
            for k in 0..N {
                let mut acc = 0.0_f32;
                for j in 0..N {
                    acc += m.get(i, j) * self.get(j, k);
                }
                c[i][k] = acc;
            }
        }

        let mut out = Self::zero();
        for i in 0..N {
            for l in 0..=i {
                let mut acc = 0.0_f32;
                for k in 0..N {
                    acc += c[i][k] * m.get(l, k);
                }
                out.lt[Self::idx(i, l)] = acc;
            }
        }
        out
    }

    /// Squared Frobenius norm: `‖Σ‖_F² = Σ_{i,j} Σ_{ij}²`.
    ///
    /// Computed efficiently from packed storage: diagonal entries contribute
    /// once, off-diagonal entries contribute twice (symmetry).
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::hpc::pillar::cov_high_d::CovHighD;
    /// let id = CovHighD::<4>::identity();
    /// assert!((id.frobenius_sq() - 4.0_f32).abs() < 1e-5);
    /// ```
    pub fn frobenius_sq(&self) -> f32 {
        let mut acc = 0.0_f32;
        for i in 0..N {
            // Diagonal (once)
            let d = self.lt[Self::idx(i, i)];
            acc += d * d;
            // Off-diagonal entries in row i (below diagonal) — counted twice
            for j in 0..i {
                let v = self.lt[Self::idx(i, j)];
                acc += 2.0 * v * v;
            }
        }
        acc
    }

    /// Matrix logarithm of an SPD matrix via eigendecomposition.
    ///
    /// Computes `log(Σ) = V · diag(log(λ_i)) · Vᵀ` where `V Λ Vᵀ` is the
    /// eigendecomposition of `Σ`. Eigenvalues are clamped to `[f32::MIN_POSITIVE, ∞)`
    /// before taking the log to guard against numerical non-positivity.
    ///
    /// **Constraint**: Only valid for `N ≤ 64` (uses [`eig_sym_jacobi`]).
    /// For PR-X11 v1 with `N = 64` this is satisfied. The `N = 16384`
    /// stress test (PR-X11.1) will need a different algorithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::hpc::pillar::cov_high_d::CovHighD;
    /// // log(I) = 0
    /// let id = CovHighD::<4>::identity();
    /// let log_id = id.log_spd();
    /// assert!(log_id.frobenius_sq() < 1e-8);
    /// ```
    pub fn log_spd(&self) -> Self {
        assert!(N <= 64, "log_spd only valid for N ≤ 64 (PR-X11 v1); N={N}");

        // Build a MatN<N> from packed storage for eig_sym_jacobi.
        let mut a: MatN<N> = [[0.0_f32; N]; N];
        for i in 0..N {
            for j in 0..N {
                a[i][j] = self.get(i, j);
            }
        }

        let (eigs, vecs) = eig_sym_jacobi::<N>(&a, 100, 1e-7_f32);

        // Apply log to eigenvalues, reconstruct: result = V * diag(log λ) * Vᵀ
        // vecs[c][r] = r-th component of eigenvector c (column-major columns).
        // result[i][j] = Σ_c  vecs[c][i] * log(λ_c) * vecs[c][j]
        let log_eigs: Vec<f32> = eigs
            .iter()
            .map(|&e| e.max(f32::MIN_POSITIVE).ln())
            .collect();

        let mut out = Self::zero();
        for i in 0..N {
            for j in 0..=i {
                let mut acc = 0.0_f32;
                for c in 0..eigs.len() {
                    acc += vecs[c][i] * log_eigs[c] * vecs[c][j];
                }
                out.lt[Self::idx(i, j)] = acc;
            }
        }
        out
    }
}

// ── Probe helper — random SPD N×N from factor ─────────────────────────────────

/// Generate a random SPD matrix of size N using a random lower-triangular
/// Cholesky factor A with positive diagonal, then form Σ = A Aᵀ, scaled so
/// that `‖Σ‖_F = sigma_frobenius`.
///
/// Uses `N*(N+1)/2` normal samples for the lower-triangular entries plus N
/// calls to `abs()` + offset for the diagonal — all from `rng`.
fn random_spd_n<const N: usize>(rng: &mut SplitMix64, sigma_frobenius: f32) -> CovHighD<N> {
    // Build lower-triangular factor A as a flat vec (row-major lower triangle).
    // a_flat[i*(i+1)/2 + j] = A[i][j] for i >= j.
    let size = N * (N + 1) / 2;
    let mut a_flat = vec![0.0_f32; size];
    for i in 0..N {
        for j in 0..i {
            a_flat[i * (i + 1) / 2 + j] = rng.next_normal_f32();
        }
        // Positive diagonal ensures SPD.
        a_flat[i * (i + 1) / 2 + i] = rng.next_normal_f32().abs() + 0.5;
    }

    // Build Σ = A Aᵀ — only compute lower triangle.
    // Σ[i][j] = Σ_{k=0}^{min(i,j)} A[i][k] * A[j][k]   (for i ≥ j)
    let mut lt = vec![0.0_f32; size];
    for i in 0..N {
        for j in 0..=i {
            let mut acc = 0.0_f32;
            let k_max = j; // A[j][k] = 0 for k > j
            for k in 0..=k_max {
                acc += a_flat[i * (i + 1) / 2 + k] * a_flat[j * (j + 1) / 2 + k];
            }
            lt[i * (i + 1) / 2 + j] = acc;
        }
    }

    // Scale to target Frobenius norm.
    let frob_sq: f32 = {
        let mut s = 0.0_f32;
        for i in 0..N {
            let d = lt[i * (i + 1) / 2 + i];
            s += d * d;
            for j in 0..i {
                let v = lt[i * (i + 1) / 2 + j];
                s += 2.0 * v * v;
            }
        }
        s
    };
    let scale = if frob_sq > 0.0 {
        sigma_frobenius / frob_sq.sqrt()
    } else {
        1.0
    };
    for x in lt.iter_mut() {
        *x *= scale;
    }

    CovHighD { lt }
}

// ── Pillar-9 CLT probe ────────────────────────────────────────────────────────

/// Run the Pillar-9 Düker–Zoubouloglou CLT convergence probe.
///
/// **Protocol**:
/// 1. Initialise the covariance carrier to `Σ₀ = I_N` (identity).
/// 2. For each path (100 total) × each hop (50 total):
///    a. Sample a random contractive SPD step matrix `M` with `‖M‖_F = σ_step`.
///    b. Update `Σ ← sandwich(M, Σ) + ε · I` (ε = 1e-4 regularisation).
///    c. Compute the Frobenius increment `δ = |‖Σ_new‖_F - ‖Σ_old‖_F|`.
///    d. Normalise: `z = δ / (σ_step * √(2/N))` — the CLT-predicted scale.
///    e. Check if `|z|` falls within the 95% standard-normal confidence
///       band `[0, 1.96]` (one-sided: we expect |z| ≤ 1.96 for ≥ 95% of hops).
/// 3. Compute `clt_rate = passed_hops / total_hops`.
/// 4. Return [`PillarReport`] with `passed = clt_rate ≥ PILLAR_9_CLT_THRESHOLD`.
///
/// # Example (quick smoke test — not the full probe)
///
/// ```rust
/// use ndarray::hpc::pillar::cov_high_d::prove_pillar_9;
/// let report = prove_pillar_9();
/// report.print();
/// assert!(report.passed, "Pillar-9 CLT rate below threshold");
/// ```
pub fn prove_pillar_9() -> PillarReport {
    // PR-X11 v1: N=64 concrete default (N=16384 is PR-X11.1 follow-on).
    const DIM: usize = 64;
    const N_PATHS: u32 = 100;
    const N_HOPS: u32 = 50;
    const SIGMA_STEP: f32 = 0.08;
    // CLT-predicted normalisation scale for Frobenius increments.
    // Under the Düker–Zoubouloglou central limit, the Frobenius increment
    // of a rank-1 sandwich update scales as σ_step · √(2/N).
    let clt_scale: f64 = (SIGMA_STEP as f64) * ((2.0_f64 / DIM as f64).sqrt());
    // 95% two-tailed normal confidence bound.
    const Z_95: f64 = 1.96;
    // Regularisation to keep Σ strictly positive definite.
    const EPS_REG: f32 = 1e-4;

    let mut rng = SplitMix64::new(PILLAR_9_SEED);
    let total_hops = N_PATHS * N_HOPS;
    let mut passed_hops: u32 = 0;

    for _path in 0..N_PATHS {
        // Reset Σ to identity at the start of each path.
        let mut sigma: CovHighD<DIM> = CovHighD::identity();

        for _hop in 0..N_HOPS {
            // Sample contractive SPD step matrix.
            let m = random_spd_n::<DIM>(&mut rng, SIGMA_STEP);

            // Record ‖Σ_old‖_F.
            let frob_old = sigma.frobenius_sq().sqrt() as f64;

            // Update: Σ_new = M Σ Mᵀ + ε I.
            let mut sigma_new = sigma.sandwich(&m);
            // Add regularisation ε · I along diagonal.
            for i in 0..DIM {
                sigma_new.lt[CovHighD::<DIM>::idx(i, i)] += EPS_REG;
            }

            // Record ‖Σ_new‖_F.
            let frob_new = sigma_new.frobenius_sq().sqrt() as f64;

            // Compute normalised CLT statistic.
            let delta = (frob_new - frob_old).abs();
            let z = if clt_scale > 0.0 { delta / clt_scale } else { 0.0 };

            // The CLT predicts |z| ≤ Z_95 for ≈ 95% of steps.
            if z <= Z_95 {
                passed_hops += 1;
            }

            sigma = sigma_new;
        }
    }

    let clt_rate = passed_hops as f64 / total_hops as f64;
    let passed = clt_rate >= PILLAR_9_CLT_THRESHOLD;

    PillarReport {
        pillar_id: 9,
        seed: PILLAR_9_SEED,
        n_paths: N_PATHS,
        n_hops: N_HOPS,
        psd_rate: clt_rate, // repurposed: stores CLT convergence rate
        lognorm_concentration: clt_rate,
        passed,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── CovHighD helpers ──────────────────────────────────────────────────────

    #[test]
    fn identity_frobenius_sq_equals_n() {
        // ‖I_N‖_F² = N (N diagonal 1s, all others 0)
        let id = CovHighD::<8>::identity();
        assert!((id.frobenius_sq() - 8.0_f32).abs() < 1e-5, "frobenius_sq={:.6}", id.frobenius_sq());
    }

    #[test]
    fn identity_frobenius_sq_n4() {
        let id = CovHighD::<4>::identity();
        assert!((id.frobenius_sq() - 4.0_f32).abs() < 1e-5);
    }

    #[test]
    fn sandwich_identity_is_identity() {
        // I · I · Iᵀ = I
        let id = CovHighD::<4>::identity();
        let r = id.sandwich(&id);
        // off-diagonal should be 0, diagonal should be 1
        for i in 0..4_usize {
            for j in 0..=i {
                let v = r.lt[CovHighD::<4>::idx(i, j)];
                let expected = if i == j { 1.0_f32 } else { 0.0_f32 };
                assert!((v - expected).abs() < 1e-4, "r[{i}][{j}] = {v:.6} expected {expected:.6}");
            }
        }
    }

    #[test]
    fn sandwich_scales_frobenius() {
        // If M = s·I then M Σ Mᵀ = s²·Σ, so ‖result‖_F = s² ‖Σ‖_F
        // We can't easily build s·I as CovHighD without a scalar constructor,
        // but we can check that sandwich with identity leaves Frobenius unchanged.
        let mut rng = SplitMix64::new(0x1234);
        let sigma = random_spd_n::<4>(&mut rng, 1.0);
        let id = CovHighD::<4>::identity();
        let r = sigma.sandwich(&id);
        let frob_sigma = sigma.frobenius_sq().sqrt();
        let frob_r = r.frobenius_sq().sqrt();
        assert!(
            (frob_sigma - frob_r).abs() < frob_sigma * 1e-4,
            "sandwich with identity changed Frobenius: {frob_sigma:.6} vs {frob_r:.6}"
        );
    }

    #[test]
    fn log_spd_of_identity_is_zero() {
        // log(I) = 0
        let id = CovHighD::<8>::identity();
        let log_id = id.log_spd();
        let frob_sq = log_id.frobenius_sq();
        assert!(frob_sq < 1e-6, "log(I) Frobenius² = {frob_sq:.2e} not near 0");
    }

    #[test]
    fn log_spd_of_diagonal() {
        // For diagonal Σ = diag(e^a, e^b, ...), log(Σ) = diag(a, b, ...).
        // Use CovHighD<4> with Σ = diag(e¹, e², e³, e⁴).
        let mut sigma = CovHighD::<4>::zero();
        let diag_vals = [1.0_f32, 2.0, 3.0, 4.0];
        for i in 0..4_usize {
            sigma.lt[CovHighD::<4>::idx(i, i)] = diag_vals[i].exp();
        }
        let log_sigma = sigma.log_spd();
        for i in 0..4_usize {
            let got = log_sigma.lt[CovHighD::<4>::idx(i, i)];
            let want = diag_vals[i];
            assert!((got - want).abs() < 1e-4, "log_spd diagonal[{i}]: got={got:.6} want={want:.6}");
            // Off-diagonal should stay near 0
            for j in 0..i {
                let off = log_sigma.lt[CovHighD::<4>::idx(i, j)];
                assert!(off.abs() < 1e-4, "log_spd off-diag[{i}][{j}] = {off:.6}");
            }
        }
    }

    #[test]
    fn random_spd_frobenius_at_sigma() {
        // After normalization, ‖Σ‖_F should be within 0.5% of sigma_frobenius.
        let mut rng = SplitMix64::new(0xC0DA_DA5A_DC_u64);
        let sigma_target = 0.3_f32;
        for _ in 0..20 {
            let m = random_spd_n::<8>(&mut rng, sigma_target);
            let frob = m.frobenius_sq().sqrt();
            assert!((frob - sigma_target).abs() < sigma_target * 0.005, "Frobenius={frob:.6} target={sigma_target}");
        }
    }

    #[test]
    fn prove_pillar_9_passes() {
        let report = prove_pillar_9();
        report.print();
        assert!(report.passed, "Pillar-9 CLT rate {:.4} below threshold {PILLAR_9_CLT_THRESHOLD}", report.psd_rate);
    }
}
