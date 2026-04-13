//! Pyramid Vector Quantizer (PVQ) — from Opus CELT `celt/vq.c`.
//!
//! Distributes K integer pulses across N dimensions on the L1 hypersphere.
//! sum(|pulse_i|) = K. Algebraic — no trained codebook.
//!
//! The number of valid codewords is C(N+K-1, K), indexed combinatorially
//! (CWRS encoding). This IS the shape component of gain-shape quantization.
//!
//! For the HHTL pipeline: PVQ encodes the normalized band shape at TWIG level.

/// Encode: project a unit-energy band onto the PVQ lattice.
///
/// Input: `band` — normalized band coefficients (unit L2 norm).
/// `k`: number of pulses to distribute.
/// Returns: integer pulse vector, sum(|pulses|) == k.
pub fn pvq_encode(band: &[f32], k: u32) -> Vec<i32> {
    let n = band.len();
    if n == 0 || k == 0 {
        return vec![0; n];
    }

    // Greedy pulse allocation (from Opus alg_quant):
    // Repeatedly place the next pulse at the dimension that
    // maximizes the inner product with the target.
    let mut pulses = vec![0i32; n];
    let mut remaining = k as i32;

    while remaining > 0 {
        // Find dimension with largest residual magnitude
        let mut best_dim = 0;
        let mut best_val = 0.0f32;
        for d in 0..n {
            let target = band[d];
            let current = pulses[d] as f32;
            // How much would adding ±1 pulse improve alignment?
            let benefit = (target.abs() - current.abs()).abs();
            if benefit > best_val || (benefit == best_val && target.abs() > band[best_dim].abs()) {
                best_val = benefit;
                best_dim = d;
            }
        }
        // Place pulse with same sign as target
        if band[best_dim] >= 0.0 {
            pulses[best_dim] += 1;
        } else {
            pulses[best_dim] -= 1;
        }
        remaining -= 1;
    }

    pulses
}

/// Decode: convert pulse vector back to unit-energy coefficients.
///
/// Normalizes the pulse vector to unit L2 norm.
pub fn pvq_decode(pulses: &[i32]) -> Vec<f32> {
    let n = pulses.len();
    let mut output = vec![0.0f32; n];
    let mut norm_sq = 0.0f64;
    for i in 0..n {
        output[i] = pulses[i] as f32;
        norm_sq += (pulses[i] as f64) * (pulses[i] as f64);
    }
    let norm = (norm_sq.sqrt()).max(1e-10) as f32;
    for v in &mut output {
        *v /= norm;
    }
    output
}

/// Compute the L1 norm of a pulse vector (should equal K).
pub fn pvq_l1_norm(pulses: &[i32]) -> u32 {
    pulses.iter().map(|&p| p.unsigned_abs()).sum()
}

/// PVQ summary: compress pulse vector to 6-byte fingerprint for HHTL.
///
/// Maps to SPO: Subject = spectral, Predicate = temporal, Object = harmonic.
///   Bytes 0-1 (HEEL): coarse spectral category (sign pattern of dominant dims)
///   Bytes 2-3 (HIP): energy distribution pattern
///   Bytes 4-5 (TWIG): fine harmonic structure
pub fn pvq_summary(pulses: &[i32]) -> [u8; 6] {
    let n = pulses.len();
    let mut summary = [0u8; 6];

    // HEEL (bytes 0-1): sign pattern of first 16 dims → 16 bits
    let mut sign_bits = 0u16;
    for i in 0..n.min(16) {
        if pulses[i] > 0 { sign_bits |= 1 << i; }
    }
    summary[0] = sign_bits as u8;
    summary[1] = (sign_bits >> 8) as u8;

    // HIP (bytes 2-3): which quarter has most energy
    let q = n / 4;
    let mut quarter_energy = [0u32; 4];
    for i in 0..n {
        let qi = (i * 4 / n).min(3);
        quarter_energy[qi] += pulses[i].unsigned_abs();
    }
    let total = quarter_energy.iter().sum::<u32>().max(1);
    for i in 0..4 {
        let frac = (quarter_energy[i] * 255 / total) as u8;
        if i < 2 { summary[2] |= frac >> (4 * (1 - i)); }
        else { summary[3] |= frac >> (4 * (3 - i)); }
    }

    // TWIG (bytes 4-5): max pulse position + magnitude
    let (max_pos, max_val) = pulses.iter().enumerate()
        .max_by_key(|(_, &p)| p.unsigned_abs())
        .map(|(i, &p)| (i, p.unsigned_abs()))
        .unwrap_or((0, 0));
    summary[4] = (max_pos % 256) as u8;
    summary[5] = (max_val % 256) as u8;

    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pvq_l1_correct() {
        let band = vec![0.5, -0.3, 0.7, -0.1, 0.2];
        let pulses = pvq_encode(&band, 8);
        assert_eq!(pvq_l1_norm(&pulses), 8, "L1 norm should equal K=8");
    }

    #[test]
    fn pvq_signs_match() {
        let band = vec![1.0, -0.5, 0.3, -0.8, 0.0];
        let pulses = pvq_encode(&band, 10);
        // Dominant pulse signs should match input signs
        for i in 0..band.len() {
            if band[i].abs() > 0.3 {
                assert_eq!(pulses[i].signum(), band[i].signum() as i32,
                    "Sign mismatch at dim {}: pulse={}, band={}", i, pulses[i], band[i]);
            }
        }
    }

    #[test]
    fn pvq_decode_unit_norm() {
        let pulses = vec![3, -2, 1, 0, -4];
        let decoded = pvq_decode(&pulses);
        let norm: f32 = decoded.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Decoded should be unit norm: {}", norm);
    }

    #[test]
    fn pvq_summary_deterministic() {
        let pulses = vec![3, -2, 1, 0, -4, 2, 0, 1];
        let s1 = pvq_summary(&pulses);
        let s2 = pvq_summary(&pulses);
        assert_eq!(s1, s2);
        // Should be non-trivial
        assert!(s1.iter().any(|&b| b != 0));
    }
}
