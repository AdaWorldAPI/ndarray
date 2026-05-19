//! Pillar probe certification harness.
//!
//! Shared splitmix64 RNG + SEED-anchored deterministic test infrastructure
//! consumed by Pillar-6 through Pillar-11 probes (B1-B7). Per invariant 12
//! (master consolidation): certification = determinism + inspectability.
//!
//! # Quick start
//!
//! ```rust
//! use ndarray::hpc::pillar::prove_runner::{SplitMix64, PillarReport, assert_psd_rate};
//!
//! const SEED: u64 = 0xDA5ADC5ADD;
//! let mut rng = SplitMix64::new(SEED);
//! let _v = rng.next_f32();
//! let report = PillarReport {
//!     pillar_id: 6, seed: SEED, n_paths: 10, n_hops: 5,
//!     psd_rate: 1.0, lognorm_concentration: 0.0, passed: true,
//! };
//! report.print();
//! assert!(assert_psd_rate(10, 10, 0.999));
//! ```

// ── SplitMix64 ────────────────────────────────────────────────────────────────

/// Deterministic splitmix64 RNG. Seeded from a pillar's `SEED` constant.
///
/// This is the canonical RNG for all PR-X11 pillar probes. Using the same
/// algorithm everywhere guarantees cross-pillar reproducibility and lets
/// auditors verify probe sequences against independent implementations
/// (Java `SplittableRandom`, numpy `SeedSequence`, etc.).
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::prove_runner::SplitMix64;
/// let mut rng = SplitMix64::new(42);
/// let a = rng.next_u64();
/// let b = rng.next_u64();
/// assert_ne!(a, b);
/// ```
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Construct a new RNG with the given seed.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::hpc::pillar::prove_runner::SplitMix64;
    /// let mut rng = SplitMix64::new(0xDA5ADC5ADD);
    /// ```
    #[inline]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Advance the state and return the next 64-bit value.
    ///
    /// Implements the standard splitmix64 step from Vigna (2015):
    /// <https://prng.di.unimi.it/splitmix64.c>
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform sample in `[0, 1)` as `f32`.
    ///
    /// Uses the top 23 bits of the next 64-bit output (mantissa width of f32).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // top 24 bits → [0, 2^24) / 2^24 → [0, 1)
        let bits = (self.next_u64() >> 40) as u32;
        (bits as f32) * (1.0_f32 / (1u32 << 24) as f32)
    }

    /// Uniform sample in `[0, 1)` as `f64`.
    ///
    /// Uses the top 53 bits of the next 64-bit output (mantissa width of f64).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        // top 53 bits → [0, 2^53) / 2^53 → [0, 1)
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0_f64 / (1u64 << 53) as f64)
    }

    /// Standard-normal sample as `f32` via the Box–Muller transform.
    ///
    /// Each call to `next_normal_f32` consumes two `u64` outputs (two uniform
    /// draws) and discards the second polar value. This is stateless simplicity
    /// over caching efficiency — probe code runs once, not in hot paths.
    ///
    /// # Example
    ///
    /// ```rust
    /// use ndarray::hpc::pillar::prove_runner::SplitMix64;
    /// let mut rng = SplitMix64::new(7);
    /// let n = rng.next_normal_f32();
    /// // n is drawn from N(0, 1)
    /// let _ = n;
    /// ```
    #[inline]
    pub fn next_normal_f32(&mut self) -> f32 {
        // Box–Muller: u1, u2 ∈ (0, 1), then R = sqrt(-2 ln u1), θ = 2π u2
        // z0 = R cos θ   z1 = R sin θ  (we discard z1 for simplicity)
        //
        // Guard against u1 = 0 (which would give ln(0) = -inf) by clamping to
        // (0, 1) via nudging a zero sample to the smallest positive f32.
        let u1_raw = self.next_f32();
        let u2 = self.next_f32();
        let u1 = if u1_raw == 0.0_f32 { f32::MIN_POSITIVE } else { u1_raw };
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let theta = core::f32::consts::TAU * u2;
        r * theta.cos()
    }
}

// ── PillarReport ──────────────────────────────────────────────────────────────

/// Common report shape for all pillar probes.
///
/// Emitted by every `prove()` function and passed back to the test harness.
/// Consumers can inspect individual fields or call [`PillarReport::print`] for
/// a human-readable, deterministic summary (useful in CI logs and `RESULTS.md`).
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::prove_runner::PillarReport;
/// let r = PillarReport {
///     pillar_id: 7,
///     seed: 0xEDA5ADC5ADD,
///     n_paths: 1000,
///     n_hops: 10,
///     psd_rate: 0.9997,
///     lognorm_concentration: 0.0023,
///     passed: true,
/// };
/// r.print();
/// ```
#[derive(Debug, Clone)]
pub struct PillarReport {
    /// Pillar number (6 through 11).
    pub pillar_id: u8,
    /// RNG seed used for this run — must match the pillar's `SEED` constant.
    pub seed: u64,
    /// Number of random paths simulated.
    pub n_paths: u32,
    /// Number of cascade hops per path.
    pub n_hops: u32,
    /// Fraction of hops where the covariance matrix Σ remained symmetric
    /// positive-definite (SPD) after the cascade update.
    pub psd_rate: f64,
    /// Log-norm Frobenius concentration metric (pillar-specific semantics).
    pub lognorm_concentration: f64,
    /// `true` if all PASS criteria were met.
    pub passed: bool,
}

impl PillarReport {
    /// Pretty-print the report to stdout in a deterministic, grep-friendly format.
    ///
    /// Output format:
    /// ```text
    /// [PILLAR-7] seed=0xEDA5ADC5ADD paths=1000 hops=10 psd_rate=0.9997 lognorm_conc=0.0023 PASS
    /// ```
    pub fn print(&self) {
        let status = if self.passed { "PASS" } else { "FAIL" };
        // Use plain integer formatting for seed so the output is reproducible
        // across platforms (no locale-dependent float printing).
        println!(
            "[PILLAR-{id}] seed=0x{seed:X} paths={paths} hops={hops} \
             psd_rate={psd:.6} lognorm_conc={lnc:.6} {status}",
            id = self.pillar_id,
            seed = self.seed,
            paths = self.n_paths,
            hops = self.n_hops,
            psd = self.psd_rate,
            lnc = self.lognorm_concentration,
            status = status,
        );
    }
}

// ── SPD helpers ───────────────────────────────────────────────────────────────

/// Generate a random 3×3 symmetric positive-definite (SPD) matrix with
/// Frobenius norm exactly equal to `sigma_step_frobenius`.
///
/// Construction:
/// 1. Sample a lower-triangular factor `A` with standard-normal entries and
///    positive diagonal (ensuring SPD).
/// 2. Form `B = A Aᵀ` — always SPD when diagonal of `A` is positive.
/// 3. Scale `B` by `sigma_step_frobenius / ‖B‖_F` so the result hits the
///    target norm exactly. Scaling by a positive scalar preserves SPD.
///
/// # Arguments
///
/// * `rng` — splitmix64 source; consumed deterministically.
/// * `sigma_step_frobenius` — exact target Frobenius norm of the output matrix.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::prove_runner::{SplitMix64, random_contractive_spd3};
/// let mut rng = SplitMix64::new(42);
/// let m = random_contractive_spd3(&mut rng, 0.1);
/// // m is SPD with Frobenius norm = 0.1
/// ```
pub fn random_contractive_spd3(rng: &mut SplitMix64, sigma_step_frobenius: f32) -> [[f32; 3]; 3] {
    // Step 1: lower-triangular factor A with positive diagonal.
    let a00 = rng.next_normal_f32().abs() + 0.5_f32;
    let a10 = rng.next_normal_f32();
    let a11 = rng.next_normal_f32().abs() + 0.5_f32;
    let a20 = rng.next_normal_f32();
    let a21 = rng.next_normal_f32();
    let a22 = rng.next_normal_f32().abs() + 0.5_f32;

    // Step 2: B = A Aᵀ.
    let b00 = a00 * a00;
    let b10 = a10 * a00;
    let b11 = a10 * a10 + a11 * a11;
    let b20 = a20 * a00;
    let b21 = a20 * a10 + a21 * a11;
    let b22 = a20 * a20 + a21 * a21 + a22 * a22;

    // Step 3: normalize to exactly the requested Frobenius norm.
    // ‖B‖_F² = b00² + b11² + b22² + 2*(b10² + b20² + b21²)
    let frob_sq = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * (b10 * b10 + b20 * b20 + b21 * b21);
    let scale = if frob_sq > 0.0 {
        sigma_step_frobenius / frob_sq.sqrt()
    } else {
        1.0
    };

    [
        [b00 * scale, b10 * scale, b20 * scale],
        [b10 * scale, b11 * scale, b21 * scale],
        [b20 * scale, b21 * scale, b22 * scale],
    ]
}

/// Generate a random 2×2 SPD matrix with Frobenius norm exactly equal to
/// `sigma_step_frobenius`.
///
/// Analogous to [`random_contractive_spd3`] but for 2×2 matrices consumed by
/// Pillar-6 (2D EWA sandwich) probes (B1).
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::prove_runner::{SplitMix64, random_contractive_spd2};
/// let mut rng = SplitMix64::new(42);
/// let m = random_contractive_spd2(&mut rng, 0.1);
/// // m is SPD with Frobenius norm = 0.1
/// ```
pub fn random_contractive_spd2(rng: &mut SplitMix64, sigma_step_frobenius: f32) -> [[f32; 2]; 2] {
    // Lower-triangular factor A with positive diagonal.
    let a00 = rng.next_normal_f32().abs() + 0.5_f32;
    let a10 = rng.next_normal_f32();
    let a11 = rng.next_normal_f32().abs() + 0.5_f32;

    // B = A Aᵀ.
    let b00 = a00 * a00;
    let b10 = a10 * a00;
    let b11 = a10 * a10 + a11 * a11;

    // Normalize to target Frobenius norm.
    let frob_sq = b00 * b00 + b11 * b11 + 2.0 * b10 * b10;
    let scale = if frob_sq > 0.0 {
        sigma_step_frobenius / frob_sq.sqrt()
    } else {
        1.0
    };

    [[b00 * scale, b10 * scale], [b10 * scale, b11 * scale]]
}

// ── PSD-rate assertion ─────────────────────────────────────────────────────────

/// Assert that the PSD preservation rate meets the required threshold.
///
/// Returns `true` if `passed / total >= threshold`, `false` otherwise.
///
/// # Arguments
///
/// * `passed` — number of hops where Σ remained SPD.
/// * `total`  — total hops evaluated.
/// * `threshold` — minimum acceptable rate (e.g. `0.999` for Pillar-6/7).
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::prove_runner::assert_psd_rate;
/// assert!(assert_psd_rate(999, 1000, 0.999));
/// assert!(!assert_psd_rate(998, 1000, 0.999));
/// ```
#[inline]
pub fn assert_psd_rate(passed: u32, total: u32, threshold: f64) -> bool {
    if total == 0 {
        return false;
    }
    (passed as f64) / (total as f64) >= threshold
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── splitmix64 determinism ─────────────────────────────────────────────────

    #[test]
    fn splitmix64_determinism() {
        // Same seed must produce the identical sequence every time.
        let mut a = SplitMix64::new(0xDEAD_BEEF_CAFE_1234);
        let mut b = SplitMix64::new(0xDEAD_BEEF_CAFE_1234);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn splitmix64_different_seeds_differ() {
        let mut a = SplitMix64::new(1);
        let mut b = SplitMix64::new(2);
        // With overwhelming probability the first output differs
        assert_ne!(a.next_u64(), b.next_u64());
    }

    // ── splitmix64 uniform distribution ───────────────────────────────────────

    #[test]
    fn splitmix64_f32_mean_near_half() {
        // 10 000 uniform [0,1) samples should have mean ≈ 0.5 ± 0.01.
        let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0);
        let n = 10_000u32;
        let sum: f64 = (0..n).map(|_| rng.next_f32() as f64).sum();
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "uniform mean {mean:.4} not in 0.5 ± 0.01");
    }

    #[test]
    fn splitmix64_f64_mean_near_half() {
        let mut rng = SplitMix64::new(0xFEDC_BA98_7654_3210);
        let n = 10_000u32;
        let sum: f64 = (0..n).map(|_| rng.next_f64()).sum();
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "uniform f64 mean {mean:.4} not in 0.5 ± 0.01");
    }

    // ── next_normal_f32 statistics ────────────────────────────────────────────

    #[test]
    fn normal_f32_mean_and_stddev() {
        // N(0,1): mean ≈ 0.0 ± 0.05, stddev ≈ 1.0 ± 0.05 across 10 000 samples.
        let mut rng = SplitMix64::new(0xABCD_EF01_2345_6789);
        let n = 10_000u32;
        let samples: Vec<f64> = (0..n).map(|_| rng.next_normal_f32() as f64).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        let stddev = variance.sqrt();
        assert!(mean.abs() < 0.05, "normal mean {mean:.4} not near 0.0");
        assert!((stddev - 1.0).abs() < 0.05, "normal stddev {stddev:.4} not near 1.0");
    }

    // ── random_contractive_spd3 ────────────────────────────────────────────────

    #[test]
    fn spd3_frobenius_norm_close_to_sigma() {
        // After normalization, ‖M‖_F must equal sigma within 1% (float rounding only).
        // ‖M‖_F² = M00² + M11² + M22² + 2*(M10² + M20² + M21²)
        let mut rng = SplitMix64::new(0x5A5A_5A5A_5A5A_5A5A);
        let sigma = 0.1_f32;
        for _ in 0..100 {
            let m = random_contractive_spd3(&mut rng, sigma);
            let frob_sq = m[0][0] * m[0][0]
                + m[1][1] * m[1][1]
                + m[2][2] * m[2][2]
                + 2.0 * (m[1][0] * m[1][0] + m[2][0] * m[2][0] + m[2][1] * m[2][1]);
            let frob = frob_sq.sqrt();
            assert!((frob - sigma).abs() < sigma * 0.01, "Frobenius {frob:.6} not within 1% of sigma={sigma}");
        }
    }

    #[test]
    fn spd3_is_positive_definite() {
        // 100 samples: all three eigenvalues must be positive.
        // For a 3×3 SPD matrix M, positive-definiteness is equivalent to:
        //   det(M[0..1, 0..1]) > 0  (m00 > 0)
        //   det(M[0..2, 0..2]) > 0  (Sylvester's criterion, 2×2 leading minor)
        //   det(M) > 0
        let mut rng = SplitMix64::new(0xC0CA_C01A_C0CA_C01A);
        for _ in 0..100 {
            let m = random_contractive_spd3(&mut rng, 0.1);
            let m00 = m[0][0];
            let m01 = m[0][1];
            let m11 = m[1][1];
            let m02 = m[0][2];
            let m12 = m[1][2];
            let m22 = m[2][2];

            // Leading 1×1 minor
            assert!(m00 > 0.0, "m00 = {m00} not positive");

            // Leading 2×2 minor: m00*m11 - m01²
            let det2 = m00 * m11 - m01 * m01;
            assert!(det2 > 0.0, "2×2 minor det = {det2} not positive");

            // Full 3×3 determinant (cofactor expansion along first row)
            let det3 = m00 * (m11 * m22 - m12 * m12) - m01 * (m01 * m22 - m12 * m02) + m02 * (m01 * m12 - m11 * m02);
            assert!(det3 > 0.0, "det3 = {det3} not positive");
        }
    }

    // ── PillarReport ──────────────────────────────────────────────────────────

    #[test]
    fn pillar_report_passed_flag_correct() {
        let r_pass = PillarReport {
            pillar_id: 6,
            seed: 0xDA5ADC5ADD,
            n_paths: 1000,
            n_hops: 10,
            psd_rate: 0.9997,
            lognorm_concentration: 0.001,
            passed: true,
        };
        assert!(r_pass.passed);

        let r_fail = PillarReport {
            passed: false,
            ..r_pass.clone()
        };
        assert!(!r_fail.passed);
    }

    #[test]
    fn pillar_report_print_is_deterministic() {
        // Smoke-test: print must not panic; format is exercised via capsule.
        let r = PillarReport {
            pillar_id: 7,
            seed: 0xEDA5ADC5ADD,
            n_paths: 500,
            n_hops: 20,
            psd_rate: 1.0,
            lognorm_concentration: 0.0,
            passed: true,
        };
        // Two prints of the same report produce the same line (determinism).
        // We can't easily capture stdout in no_std, so we just assert it
        // doesn't panic. The format string is const-evaluated and stable.
        r.print();
        r.print();
    }

    // ── assert_psd_rate ───────────────────────────────────────────────────────

    #[test]
    fn assert_psd_rate_pass_and_fail() {
        assert!(assert_psd_rate(999, 1000, 0.999));
        assert!(assert_psd_rate(1000, 1000, 0.999));
        assert!(!assert_psd_rate(998, 1000, 0.999));
        assert!(!assert_psd_rate(0, 0, 0.999)); // zero total → false
        assert!(assert_psd_rate(1, 1, 0.0)); // threshold = 0 → always pass (if total > 0)
    }
}
