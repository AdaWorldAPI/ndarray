//! # Phyllotactic Manifold
//!
//! Four encoding strategies for 34-byte scent seeds, benchmarked head-to-head:
//!
//! | Strategy       | Lanes | Rotation      | Radius    | Contradiction |
//! |----------------|-------|---------------|-----------|---------------|
//! | `Flat8`        | 8     | none          | none      | none          |
//! | `Spiral8`      | 8     | golden angle  | √n        | implicit      |
//! | `Spiral8Gamma` | 8     | golden angle  | √(n+γ)   | implicit      |
//! | `SevenPlusOne` | 7+1   | golden angle  | √(n+γ)   | explicit      |
//!
//! The hypothesis: `SevenPlusOne` achieves higher ρ at lower cost because
//! 7 is prime (zero algebraic aliasing) and γ heals the Lane 0 singularity.

#![allow(clippy::excessive_precision)]

// ============================================================================
// Constants
// ============================================================================

pub mod consts {
    use std::f64::consts::{GOLDEN_RATIO, TAU};

    /// Golden angle θ = 2π / φ² ≈ 2.3999632 rad ≈ 137.508°
    pub const GOLDEN_ANGLE: f64 = TAU / (GOLDEN_RATIO * GOLDEN_RATIO);

    /// Base dimensionality for ZeckBF17 projection.
    pub const BASE_DIM: usize = 17;

    /// Golden step: round(17/φ) = 11. gcd(11,17)=1 → visits all 17 residues.
    pub const GOLDEN_STEP: usize = 11;

    // ── Precomputed cos(n × golden_angle) for n = 0..8 ─────────────────
    //
    // Verified against runtime f64::cos() in test `verify_trig_constants`.
    // These are FIXED for 8 lanes — compile-time, zero runtime cost.

    pub const COS_GOLDEN: [f64; 8] = [
         1.000_000_000_000_000_0,  // n=0
        -0.737_368_878_078_319_7,  // n=1
         0.087_425_724_716_959_9,  // n=2
         0.608_438_860_978_862_6,  // n=3
        -0.984_713_485_315_428_7,  // n=4
         0.843_755_294_812_397_2,  // n=5
        -0.259_604_304_901_488_6,  // n=6
        -0.460_907_024_713_369_2,  // n=7
    ];

    pub const SIN_GOLDEN: [f64; 8] = [
         0.000_000_000_000_000_0,  // n=0
         0.675_490_294_261_523_8,  // n=1
        -0.996_171_040_864_827_8,  // n=2
         0.793_600_751_291_695_9,  // n=3
        -0.174_181_950_379_311_6,  // n=4
        -0.536_728_052_626_322_7,  // n=5
         0.965_715_074_375_778_3,  // n=6
        -0.887_448_429_245_254_6,  // n=7
    ];

    // ── Radii ──────────────────────────────────────────────────────────

    /// √n — standard Vogel spiral. Lane 0 = 0.0 (SINGULARITY).
    pub const RADII_SQRT: [f64; 8] = [
        0.000_000_000_000_000_0,  // √0 — dead
        1.000_000_000_000_000_0,  // √1
        1.414_213_562_373_095_1,  // √2
        1.732_050_807_568_877_2,  // √3
        2.000_000_000_000_000_0,  // √4
        2.236_067_977_499_789_8,  // √5
        2.449_489_742_783_177_9,  // √6
        2.645_751_311_064_590_7,  // √7
    ];

    /// √(n + γ) — Euler-corrected. Lane 0 = 0.7598 (singularity healed).
    pub const RADII_GAMMA: [f64; 8] = [
        0.759_747_105_885_591_9,  // √(0 + γ)
        1.255_872_471_591_575_7,  // √(1 + γ)
        1.605_370_880_794_071_4,  // √(2 + γ)
        1.891_352_866_310_655_6,  // √(3 + γ)
        2.139_442_839_830_392_2,  // √(4 + γ)
        2.361_612_937_147_307_4,  // √(5 + γ)
        2.564_608_286_834_762_0,  // √(6 + γ)
        2.752_674_275_118_931_0,  // √(7 + γ)
    ];

    // ── Composite spiral coordinates: cos(nθ) × radius ────────────────

    /// Flat 8-lane: X = cos(nθ) × √n
    pub const SPIRAL8_X: [f64; 8] = precompute_spiral_x(&COS_GOLDEN, &RADII_SQRT);
    pub const SPIRAL8_Y: [f64; 8] = precompute_spiral_y(&SIN_GOLDEN, &RADII_SQRT);

    /// Gamma-corrected 8-lane: X = cos(nθ) × √(n+γ)
    pub const SPIRAL8G_X: [f64; 8] = precompute_spiral_x(&COS_GOLDEN, &RADII_GAMMA);
    pub const SPIRAL8G_Y: [f64; 8] = precompute_spiral_y(&SIN_GOLDEN, &RADII_GAMMA);

    /// 7+1: same as gamma-corrected but Lane 7 zeroed (contradiction channel)
    pub const SPIRAL7_X: [f64; 8] = {
        let mut t = precompute_spiral_x(&COS_GOLDEN, &RADII_GAMMA);
        t[7] = 0.0;
        t
    };
    pub const SPIRAL7_Y: [f64; 8] = {
        let mut t = precompute_spiral_y(&SIN_GOLDEN, &RADII_GAMMA);
        t[7] = 0.0;
        t
    };

    // ── Const helper functions ────────────────────────────────────────

    const fn precompute_spiral_x(cos: &[f64; 8], radii: &[f64; 8]) -> [f64; 8] {
        let mut t = [0.0; 8];
        let mut i = 0;
        while i < 8 {
            t[i] = cos[i] * radii[i];
            i += 1;
        }
        t
    }

    const fn precompute_spiral_y(sin: &[f64; 8], radii: &[f64; 8]) -> [f64; 8] {
        let mut t = [0.0; 8];
        let mut i = 0;
        while i < 8 {
            t[i] = sin[i] * radii[i];
            i += 1;
        }
        t
    }

    /// Inter-lane gap Δ for uniformity analysis.
    pub const fn lane_gaps(radii: &[f64; 8]) -> [f64; 7] {
        let mut g = [0.0; 7];
        let mut i = 0;
        while i < 7 {
            g[i] = radii[i + 1] - radii[i];
            i += 1;
        }
        g
    }

    pub const GAPS_SQRT: [f64; 7] = lane_gaps(&RADII_SQRT);
    pub const GAPS_GAMMA: [f64; 7] = lane_gaps(&RADII_GAMMA);
}

// ============================================================================
// Seed → Slices extraction (shared by all strategies)
// ============================================================================

/// Extract 8 × i32 payload slices + heel + gamma from a 34-byte seed.
#[inline]
pub fn extract_slices(data: &[i8; 34]) -> ([f64; 8], f64, f64) {
    let heel = data[0] as f64;
    let gamma = data[33] as f64;

    let mut slices = [0.0f64; 8];
    for i in 0..8 {
        let start = 1 + (i * 4);
        let bytes = [
            data[start] as u8,
            data[start + 1] as u8,
            data[start + 2] as u8,
            data[start + 3] as u8,
        ];
        slices[i] = i32::from_le_bytes(bytes) as f64;
    }

    (slices, heel, gamma)
}

/// Extract 7 payload slices + contradiction scalar + heel + gamma.
#[inline]
pub fn extract_7plus1(data: &[i8; 34]) -> ([f64; 8], f64, f64, f64) {
    let heel = data[0] as f64;
    let gamma_byte = data[33] as f64;

    let mut slices = [0.0f64; 8];
    for i in 0..7 {
        let start = 1 + (i * 4);
        let bytes = [
            data[start] as u8,
            data[start + 1] as u8,
            data[start + 2] as u8,
            data[start + 3] as u8,
        ];
        slices[i] = i32::from_le_bytes(bytes) as f64;
    }

    // Lane 7 bytes → contradiction scalar (does NOT enter the spiral)
    let contra_bytes = [
        data[29] as u8,
        data[30] as u8,
        data[31] as u8,
        data[32] as u8,
    ];
    let contradiction = i32::from_le_bytes(contra_bytes) as f64;
    slices[7] = 0.0; // Lane 7 stays zero on the manifold

    (slices, contradiction, heel, gamma_byte)
}

// ============================================================================
// Strategy 1: Flat8 — baseline, no rotation, no radius
// ============================================================================

pub mod flat8 {
    use super::*;

    /// 8 lanes, raw values scaled by heel + gamma. No geometric structure.
    #[inline]
    pub fn encode(data: &[i8; 34]) -> [f64; 8] {
        let (slices, heel, gamma) = extract_slices(data);
        let bias = heel + gamma;
        let mut out = [0.0f64; 8];
        for i in 0..8 {
            out[i] = slices[i] + bias;
        }
        out
    }

    /// Resonance: which lanes exceed threshold (absolute value).
    #[inline]
    pub fn resonance(encoded: &[f64; 8], threshold: f64) -> u8 {
        let mut mask = 0u8;
        for i in 0..8 {
            if encoded[i].abs() >= threshold {
                mask |= 1 << i;
            }
        }
        mask
    }
}

// ============================================================================
// Strategy 2: Spiral8 — golden angle rotation, √n radii
// ============================================================================

pub mod spiral8 {
    use super::*;
    use super::consts::{SPIRAL8_X, SPIRAL8_Y};

    /// 8 lanes projected onto phyllotactic spiral. Lane 0 = dead (√0 = 0).
    #[inline]
    pub fn encode(data: &[i8; 34]) -> ([f64; 8], [f64; 8]) {
        let (slices, heel, gamma) = extract_slices(data);
        let bias = heel + gamma;
        let mut x = [0.0f64; 8];
        let mut y = [0.0f64; 8];
        for i in 0..8 {
            let val = slices[i] + bias;
            x[i] = val * SPIRAL8_X[i];
            y[i] = val * SPIRAL8_Y[i];
        }
        (x, y)
    }

    /// Resonance on 2D magnitude: √(x² + y²) ≥ threshold.
    #[inline]
    pub fn resonance(x: &[f64; 8], y: &[f64; 8], threshold: f64) -> u8 {
        let t2 = threshold * threshold;
        let mut mask = 0u8;
        for i in 0..8 {
            if x[i] * x[i] + y[i] * y[i] >= t2 {
                mask |= 1 << i;
            }
        }
        mask
    }
}

// ============================================================================
// Strategy 3: Spiral8Gamma — golden angle rotation, √(n+γ) radii
// ============================================================================

pub mod spiral8_gamma {
    use super::*;
    use super::consts::{SPIRAL8G_X, SPIRAL8G_Y};

    /// 8 lanes on Euler-corrected spiral. Lane 0 lives (√γ ≈ 0.76).
    #[inline]
    pub fn encode(data: &[i8; 34]) -> ([f64; 8], [f64; 8]) {
        let (slices, heel, gamma) = extract_slices(data);
        let bias = heel + gamma;
        let mut x = [0.0f64; 8];
        let mut y = [0.0f64; 8];
        for i in 0..8 {
            let val = slices[i] + bias;
            x[i] = val * SPIRAL8G_X[i];
            y[i] = val * SPIRAL8G_Y[i];
        }
        (x, y)
    }

    #[inline]
    pub fn resonance(x: &[f64; 8], y: &[f64; 8], threshold: f64) -> u8 {
        let t2 = threshold * threshold;
        let mut mask = 0u8;
        for i in 0..8 {
            if x[i] * x[i] + y[i] * y[i] >= t2 {
                mask |= 1 << i;
            }
        }
        mask
    }
}

// ============================================================================
// Strategy 4: SevenPlusOne — 7 spiral lanes + 1 contradiction channel
// ============================================================================

pub mod seven_plus_one {
    use super::*;
    use super::consts::{SPIRAL7_X, SPIRAL7_Y};

    /// Result of 7+1 encoding.
    #[derive(Debug, Clone, Copy)]
    pub struct Manifold {
        /// X coordinates on the 7-lane spiral (Lane 7 = 0.0)
        pub x: [f64; 8],
        /// Y coordinates on the 7-lane spiral (Lane 7 = 0.0)
        pub y: [f64; 8],
        /// Contradiction magnitude (scalar, from Lane 7 payload)
        pub contradiction: f64,
        /// HEEL value (global pivot)
        pub heel: f64,
        /// GAMMA value (tension governor)
        pub gamma: f64,
    }

    /// 7 lanes spiral + 1 contradiction scalar.
    #[inline]
    pub fn encode(data: &[i8; 34]) -> Manifold {
        let (slices, contradiction, heel, gamma) = extract_7plus1(data);
        let bias = heel + gamma;
        let mut x = [0.0f64; 8];
        let mut y = [0.0f64; 8];
        for i in 0..7 {
            let val = slices[i] + bias;
            x[i] = val * SPIRAL7_X[i];
            y[i] = val * SPIRAL7_Y[i];
        }
        // Lane 7: x=0, y=0 (does not participate in spiral)
        x[7] = 0.0;
        y[7] = 0.0;

        Manifold { x, y, contradiction, heel, gamma }
    }

    /// Resonance check on the 7 spiral lanes. Returns 7-bit mask.
    #[inline]
    pub fn resonance(m: &Manifold, threshold: f64) -> u8 {
        let t2 = threshold * threshold;
        let mut mask = 0u8;
        for i in 0..7 {
            if m.x[i] * m.x[i] + m.y[i] * m.y[i] >= t2 {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// NARS truth value from 7+1 encoding.
    ///
    /// - frequency = how many of the 7 lanes fire (0.0 .. 1.0)
    /// - confidence = inverse of contradiction magnitude (0.0 .. 1.0)
    #[inline]
    pub fn nars_truth(resonance_7bit: u8, contradiction: f64, max_contra: f64) -> (f64, f64) {
        let f = (resonance_7bit.count_ones() as f64) / 7.0;
        let c = if max_contra > 0.0 {
            1.0 - (contradiction.abs() / max_contra).min(1.0)
        } else {
            1.0
        };
        (f, c)
    }

    /// CLAM-48 extraction from manifold.
    ///
    /// ```text
    /// B1 (HEEL)    = heel byte
    /// B2 (BRANCH)  = 7-bit resonance | 1-bit contradiction-fires
    /// B3 (TWIG A)  = X-peak lane index + sign
    /// B4 (TWIG B)  = Y-peak lane index + sign
    /// B5 (LEAF)    = magnitude of peak lane (quantized)
    /// B6 (GAMMA)   = contradiction magnitude (quantized)
    /// ```
    #[inline]
    pub fn to_clam48(m: &Manifold, threshold: f64, max_contra: f64) -> [u8; 6] {
        let res = resonance(m, threshold);
        let contra_fires = if m.contradiction.abs() > max_contra * 0.5 { 1u8 } else { 0u8 };

        // Find peak-magnitude lane (among the 7)
        let mut peak_lane = 0usize;
        let mut peak_mag2 = 0.0f64;
        for i in 0..7 {
            let mag2 = m.x[i] * m.x[i] + m.y[i] * m.y[i];
            if mag2 > peak_mag2 {
                peak_mag2 = mag2;
                peak_lane = i;
            }
        }
        let peak_mag = peak_mag2.sqrt();

        // B3: peak lane (3 bits) + X sign (1 bit) + 4 bits X fractional
        let x_sign = if m.x[peak_lane] >= 0.0 { 0u8 } else { 1u8 };
        let b3 = ((peak_lane as u8) << 5) | (x_sign << 4)
            | ((m.x[peak_lane].abs().min(15.0) as u8) & 0x0F);

        // B4: Y sign (1 bit) + 7 bits Y fractional
        let y_sign = if m.y[peak_lane] >= 0.0 { 0u8 } else { 1u8 };
        let b4 = (y_sign << 7) | ((m.y[peak_lane].abs().min(127.0) as u8) & 0x7F);

        // B5: peak magnitude quantized to 0..255
        let b5 = (peak_mag.min(255.0)) as u8;

        // B6: contradiction magnitude quantized to 0..255
        let b6 = if max_contra > 0.0 {
            ((m.contradiction.abs() / max_contra * 255.0).min(255.0)) as u8
        } else {
            0u8
        };

        [
            m.heel as i8 as u8,       // B1: HEEL
            (res & 0x7F) | (contra_fires << 7), // B2: 7-bit resonance + 1-bit contradiction
            b3,                        // B3: TWIG A
            b4,                        // B4: TWIG B
            b5,                        // B5: LEAF
            b6,                        // B6: GAMMA
        ]
    }
}

// ============================================================================
// Dead Zone Benchmark
// ============================================================================

pub mod dead_zone {
    use super::seven_plus_one;

    /// Flip a single bit in the 34-byte seed at position `bit_pos` (0..272).
    pub fn flip_bit(data: &[i8; 34], bit_pos: usize) -> [i8; 34] {
        assert!(bit_pos < 34 * 8, "bit_pos out of range");
        let mut corrupted = *data;
        let byte_idx = bit_pos / 8;
        let bit_idx = bit_pos % 8;
        corrupted[byte_idx] ^= (1i8 << bit_idx) as i8;
        corrupted
    }

    /// Correlation ρ between two manifolds (Pearson on the 7 X-Y coordinates).
    pub fn correlation(a: &seven_plus_one::Manifold, b: &seven_plus_one::Manifold) -> f64 {
        // 14 values: 7 X + 7 Y
        let mut vals_a = [0.0f64; 14];
        let mut vals_b = [0.0f64; 14];
        for i in 0..7 {
            vals_a[i] = a.x[i];
            vals_a[7 + i] = a.y[i];
            vals_b[i] = b.x[i];
            vals_b[7 + i] = b.y[i];
        }

        let n = 14.0f64;
        let mean_a = vals_a.iter().sum::<f64>() / n;
        let mean_b = vals_b.iter().sum::<f64>() / n;

        let mut cov = 0.0f64;
        let mut var_a = 0.0f64;
        let mut var_b = 0.0f64;
        for i in 0..14 {
            let da = vals_a[i] - mean_a;
            let db = vals_b[i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        let denom = (var_a * var_b).sqrt();
        if denom < 1e-15 {
            0.0
        } else {
            cov / denom
        }
    }

    /// Run the full dead zone benchmark: flip each bit 0..271, measure ρ.
    ///
    /// Returns `(bit_pos, rho, region)` for each bit.
    /// Regions: "heel" (byte 0), "payload_0..6" (bytes 1-28, spiral),
    ///          "contradiction" (bytes 29-32), "gamma" (byte 33).
    pub fn run_benchmark(
        seed: &[i8; 34],
        _threshold: f64,
    ) -> Vec<(usize, f64, &'static str)> {
        let original = seven_plus_one::encode(seed);
        let mut results = Vec::with_capacity(34 * 8);

        for bit in 0..(34 * 8) {
            let corrupted_seed = flip_bit(seed, bit);
            let corrupted = seven_plus_one::encode(&corrupted_seed);
            let rho = correlation(&original, &corrupted);

            let byte_idx = bit / 8;
            let region = match byte_idx {
                0 => "heel",
                1..=28 => "payload",
                29..=32 => "contradiction",
                33 => "gamma",
                _ => unreachable!(),
            };

            results.push((bit, rho, region));
        }

        results
    }

    /// Summarize dead zone results by region.
    pub fn summarize<'a>(results: &'a [(usize, f64, &'a str)]) -> Vec<(&'a str, f64, f64, f64)> {
        let regions = ["heel", "payload", "contradiction", "gamma"];
        let mut out = Vec::new();

        for &region in &regions {
            let rhos: Vec<f64> = results
                .iter()
                .filter(|(_, _, r)| *r == region)
                .map(|(_, rho, _)| *rho)
                .collect();

            if rhos.is_empty() {
                continue;
            }

            let min = rhos.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = rhos.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean = rhos.iter().sum::<f64>() / rhos.len() as f64;

            out.push((region, min, mean, max));
        }

        out
    }
}

// ============================================================================
// Uniformity metrics — compare gap distributions
// ============================================================================

pub mod uniformity {
    /// Coefficient of variation of inter-lane gaps.
    /// Lower = more uniform spacing. 0.0 = perfectly even.
    pub fn gap_cv(gaps: &[f64]) -> f64 {
        let n = gaps.len() as f64;
        let mean = gaps.iter().sum::<f64>() / n;
        if mean.abs() < 1e-15 {
            return f64::INFINITY;
        }
        let variance = gaps.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / n;
        variance.sqrt() / mean
    }

    /// Max/min gap ratio. 1.0 = perfectly uniform. Higher = worse.
    pub fn gap_ratio(gaps: &[f64]) -> f64 {
        let min = gaps.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = gaps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if min.abs() < 1e-15 {
            f64::INFINITY
        } else {
            max / min
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{EULER_GAMMA, GOLDEN_RATIO, TAU};

    // ── Constant verification ──────────────────────────────────────────

    #[test]
    fn verify_golden_angle() {
        let expected = TAU / (GOLDEN_RATIO * GOLDEN_RATIO);
        assert!((consts::GOLDEN_ANGLE - expected).abs() < 1e-14);
    }

    #[test]
    fn verify_trig_constants() {
        for n in 0..8 {
            let angle = n as f64 * consts::GOLDEN_ANGLE;
            let cos_expected = angle.cos();
            let sin_expected = angle.sin();

            let cos_err = (consts::COS_GOLDEN[n] - cos_expected).abs();
            let sin_err = (consts::SIN_GOLDEN[n] - sin_expected).abs();

            assert!(
                cos_err < 1e-12,
                "cos[{n}]: expected {cos_expected}, got {}, err={cos_err}",
                consts::COS_GOLDEN[n]
            );
            assert!(
                sin_err < 1e-12,
                "sin[{n}]: expected {sin_expected}, got {}, err={sin_err}",
                consts::SIN_GOLDEN[n]
            );
        }
    }

    #[test]
    fn verify_radii_gamma() {
        for n in 0..8 {
            let expected = (n as f64 + EULER_GAMMA).sqrt();
            let err = (consts::RADII_GAMMA[n] - expected).abs();
            assert!(
                err < 1e-12,
                "radius_gamma[{n}]: expected {expected}, got {}, err={err}",
                consts::RADII_GAMMA[n]
            );
        }
    }

    #[test]
    fn verify_lane0_singularity() {
        // √n: Lane 0 is dead
        assert_eq!(consts::RADII_SQRT[0], 0.0);
        assert_eq!(consts::SPIRAL8_X[0], 0.0);
        assert_eq!(consts::SPIRAL8_Y[0], 0.0);

        // √(n+γ): Lane 0 lives
        assert!(consts::RADII_GAMMA[0] > 0.75);
        assert!(consts::SPIRAL8G_X[0].abs() > 0.5);
    }

    #[test]
    fn verify_7plus1_lane7_zero() {
        assert_eq!(consts::SPIRAL7_X[7], 0.0);
        assert_eq!(consts::SPIRAL7_Y[7], 0.0);
        // Lanes 0..6 are non-zero
        for i in 0..7 {
            assert!(
                consts::SPIRAL7_X[i].abs() > 1e-10 || consts::SPIRAL7_Y[i].abs() > 1e-10,
                "Lane {i} is zero — should be active"
            );
        }
    }

    // ── Uniformity comparison ──────────────────────────────────────────

    #[test]
    fn gamma_more_uniform_than_sqrt() {
        let cv_sqrt = uniformity::gap_cv(&consts::GAPS_SQRT);
        let cv_gamma = uniformity::gap_cv(&consts::GAPS_GAMMA);

        eprintln!("Gap CV (√n):   {cv_sqrt:.4}");
        eprintln!("Gap CV (√n+γ): {cv_gamma:.4}");
        eprintln!("Improvement:   {:.1}%", (1.0 - cv_gamma / cv_sqrt) * 100.0);

        assert!(
            cv_gamma < cv_sqrt,
            "Euler-gamma should produce more uniform gaps"
        );

        let ratio_sqrt = uniformity::gap_ratio(&consts::GAPS_SQRT);
        let ratio_gamma = uniformity::gap_ratio(&consts::GAPS_GAMMA);

        eprintln!("Gap ratio (√n):   {ratio_sqrt:.4}");
        eprintln!("Gap ratio (√n+γ): {ratio_gamma:.4}");

        assert!(
            ratio_gamma < ratio_sqrt,
            "Euler-gamma should reduce max/min gap ratio"
        );
    }

    // ── Encoding round-trip ────────────────────────────────────────────

    fn make_test_seed() -> [i8; 34] {
        let mut seed = [0i8; 34];
        // HEEL = 42, GAMMA = 7
        seed[0] = 42;
        seed[33] = 7;
        // Fill payload with recognizable pattern
        for i in 1..33 {
            seed[i] = (i as i8).wrapping_mul(13).wrapping_add(37);
        }
        seed
    }

    #[test]
    fn flat8_basic() {
        let seed = make_test_seed();
        let encoded = flat8::encode(&seed);
        // All 8 lanes should have values (heel+gamma bias = 49)
        for i in 0..8 {
            assert!(encoded[i].abs() > 1.0, "Lane {i} too small");
        }
        let mask = flat8::resonance(&encoded, 10.0);
        assert!(mask.count_ones() > 0);
    }

    #[test]
    fn spiral8_lane0_dead() {
        let seed = make_test_seed();
        let (x, y) = spiral8::encode(&seed);
        // Lane 0 should be ~zero due to √0 radius
        assert!(
            x[0].abs() < 1e-10 && y[0].abs() < 1e-10,
            "Lane 0 should be dead in spiral8"
        );
    }

    #[test]
    fn spiral8_gamma_lane0_alive() {
        let seed = make_test_seed();
        let (x, y) = spiral8_gamma::encode(&seed);
        let mag = (x[0] * x[0] + y[0] * y[0]).sqrt();
        assert!(mag > 1.0, "Lane 0 should be alive with gamma correction, mag={mag}");
    }

    #[test]
    fn seven_plus_one_separation() {
        let seed = make_test_seed();
        let m = seven_plus_one::encode(&seed);

        // Lane 7 should be zero on the manifold
        assert_eq!(m.x[7], 0.0);
        assert_eq!(m.y[7], 0.0);

        // Contradiction should be non-zero (from payload bytes 29-32)
        assert!(m.contradiction.abs() > 1.0, "Contradiction should carry signal");

        // Lanes 0-6 should be non-zero
        for i in 0..7 {
            let mag = (m.x[i] * m.x[i] + m.y[i] * m.y[i]).sqrt();
            assert!(mag > 1.0, "Lane {i} magnitude too low: {mag}");
        }
    }

    #[test]
    fn nars_truth_extraction() {
        let seed = make_test_seed();
        let m = seven_plus_one::encode(&seed);
        let res = seven_plus_one::resonance(&m, 100.0);
        let (f, c) = seven_plus_one::nars_truth(res, m.contradiction, 1e8);

        assert!(f >= 0.0 && f <= 1.0, "frequency out of range: {f}");
        assert!(c >= 0.0 && c <= 1.0, "confidence out of range: {c}");

        eprintln!("NARS truth: <f={f:.3}, c={c:.3}>");
        eprintln!("Resonance mask: {res:07b} ({} of 7 fire)", res.count_ones());
    }

    #[test]
    fn clam48_roundtrip() {
        let seed = make_test_seed();
        let m = seven_plus_one::encode(&seed);
        let clam = seven_plus_one::to_clam48(&m, 100.0, 1e8);

        // B1 should be HEEL
        assert_eq!(clam[0], 42u8);

        // B2 low 7 bits = resonance, bit 7 = contradiction flag
        let resonance_bits = clam[1] & 0x7F;
        let contra_flag = (clam[1] >> 7) & 1;
        eprintln!("CLAM-48: [{:02X} {:02X} {:02X} {:02X} {:02X} {:02X}]",
            clam[0], clam[1], clam[2], clam[3], clam[4], clam[5]);
        eprintln!("  B1 HEEL:          {}", clam[0] as i8);
        eprintln!("  B2 resonance:     {:07b} | contra={contra_flag}", resonance_bits);
        eprintln!("  B3 TWIG A:        {:08b}", clam[2]);
        eprintln!("  B4 TWIG B:        {:08b}", clam[3]);
        eprintln!("  B5 LEAF (peak):   {}", clam[4]);
        eprintln!("  B6 GAMMA (contra):{}", clam[5]);
    }

    // ── Dead Zone ──────────────────────────────────────────────────────

    #[test]
    fn dead_zone_structure() {
        let seed = make_test_seed();
        let original = seven_plus_one::encode(&seed);

        // Measure L2 displacement per region instead of Pearson ρ.
        // Pearson is scale-invariant — HEEL shifts all lanes equally,
        // leaving ρ = 1.0. L2 captures the actual geometric movement.
        let mut heel_disp = Vec::new();
        let mut payload_disp = Vec::new();
        let mut contra_disp = Vec::new();
        let mut gamma_disp = Vec::new();

        for bit in 0..(34 * 8) {
            let corrupted_seed = dead_zone::flip_bit(&seed, bit);
            let corrupted = seven_plus_one::encode(&corrupted_seed);

            let mut d2 = 0.0f64;
            for i in 0..7 {
                let dx = original.x[i] - corrupted.x[i];
                let dy = original.y[i] - corrupted.y[i];
                d2 += dx * dx + dy * dy;
            }
            let disp = d2.sqrt();

            let byte_idx = bit / 8;
            match byte_idx {
                0 => heel_disp.push(disp),
                1..=28 => payload_disp.push(disp),
                29..=32 => contra_disp.push(disp),
                33 => gamma_disp.push(disp),
                _ => unreachable!(),
            };
        }

        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let heel_mean = mean(&heel_disp);
        let payload_mean = mean(&payload_disp);
        let contra_mean = mean(&contra_disp);
        let gamma_mean = mean(&gamma_disp);

        eprintln!("\n=== Dead Zone: L2 Displacement per bit-flip ===");
        eprintln!("{:<15} {:>12}", "Region", "mean Δ");
        eprintln!("{:-<15} {:->12}", "", "");
        eprintln!("{:<15} {:>12.2}", "heel", heel_mean);
        eprintln!("{:<15} {:>12.2}", "payload", payload_mean);
        eprintln!("{:<15} {:>12.2}", "contradiction", contra_mean);
        eprintln!("{:<15} {:>12.2}", "gamma", gamma_mean);

        // HEEL errors affect ALL 7 lanes (global bias shift) → large displacement
        // GAMMA errors only scale the bias → large displacement (heel+gamma)
        // Both heel and gamma are global, so both cause large shifts.
        // PAYLOAD errors affect 1 of 7 lanes → localized displacement.
        // CONTRADICTION errors affect ZERO spiral lanes → zero manifold displacement.
        assert!(
            contra_mean < payload_mean,
            "Contradiction errors should not move the spiral (contra={contra_mean}, payload={payload_mean})"
        );
        assert_eq!(
            contra_mean, 0.0,
            "Contradiction errors must produce zero manifold displacement"
        );
    }

    // ── Primality advantage ────────────────────────────────────────────

    #[test]
    fn seven_is_prime_no_aliasing() {
        // For 7 lanes: gcd(7, step) = 1 for ALL steps 1..6
        // For 8 lanes: gcd(8, step) = 1 only for odd steps
        let mut gcd7_all_coprime = true;
        let mut gcd8_all_coprime = true;

        for step in 1..7 {
            if gcd(7, step) != 1 {
                gcd7_all_coprime = false;
            }
        }
        for step in 1..8 {
            if gcd(8, step) != 1 {
                gcd8_all_coprime = false;
            }
        }

        assert!(gcd7_all_coprime, "7 should be coprime with all steps 1..6");
        assert!(!gcd8_all_coprime, "8 should NOT be coprime with all steps 1..7");

        eprintln!("7 lanes: coprime with ALL steps 1..6 = {gcd7_all_coprime}");
        eprintln!("8 lanes: coprime with ALL steps 1..7 = {gcd8_all_coprime}");
        eprintln!("Aliasing-free subgroups: 7 has none (prime), 8 has {{2,4}}");
    }

    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
}
