//! Per-block quantized dot products (Q4_0 × Q8_0 → f32, etc.).
//!
//! Distinct from `hpc::quantized::int8_gemm_f32` (full GEMM, M×N×K).
//! These functions compute a single inner product between two
//! pre-quantized vectors, mirroring candle's `vec_dot_q4_0_q8_0`
//! kernels at `candle-core/src/quantized/{k_quants,avx,neon,simd128}.rs`.

/// Block size (number of elements per block) for Q4_0 quantization.
///
/// Matches the GGUF / llama.cpp constant `QK4_0 = 32`.
pub const QK4_0: usize = 32;

/// Block size (number of elements per block) for Q8_0 quantization.
///
/// Matches the GGUF / llama.cpp constant `QK8_0 = 32`.
pub const QK8_0: usize = 32;

/// Per-block dot product of a Q4_0-quantized vector and a Q8_0-quantized vector.
///
/// Returns the approximate inner product `Σ a[i] * b[i]` where `a` is encoded
/// in Q4_0 format and `b` is encoded in Q8_0 format.
///
/// # Block layout
///
/// **Q4_0** (`a_q4`, `a_scales`):
/// - Each block covers [`QK4_0`] = 32 elements packed as 16 bytes.
/// - Byte `j` (0 ≤ j < 16) stores element `j` in the low nibble and element
///   `j + 16` in the high nibble.
/// - The signed value is recovered as `(nibble as i32) - 8`.
/// - `a_scales[b]` is the per-block scale `d` such that `x ≈ signed_nibble * d`.
///
/// **Q8_0** (`b_q8`, `b_scales`):
/// - Each block covers [`QK8_0`] = 32 elements stored as 32 `i8` bytes.
/// - The signed value is the raw `i8`.
/// - `b_scales[b]` is the per-block scale `d` such that `y ≈ q8 * d`.
///
/// # Arguments
///
/// - `a_q4`    — packed Q4_0 nibbles; length must be `n_blocks * 16`.
/// - `a_scales` — Q4_0 per-block scales; length must be `n_blocks`.
/// - `b_q8`    — Q8_0 signed bytes; length must be `n_blocks * 32`.
/// - `b_scales` — Q8_0 per-block scales; length must be `n_blocks`.
///
/// # Panics
///
/// Panics if the slice lengths are inconsistent (i.e. `a_q4.len() != a_scales.len() * 16`
/// or `b_q8.len() != b_scales.len() * 32` or `a_scales.len() != b_scales.len()`).
///
/// # Example
///
/// ```rust
/// use ndarray_hpc::hpc::quantized_dot::vec_dot_q4_0_q8_0;
///
/// // One all-zero block — dot product should be 0.
/// let a_q4    = vec![0x88u8; 16]; // nibble 8 - 8 = 0 for all elements
/// let a_scale = vec![1.0f32];
/// let b_q8    = vec![0i8; 32];
/// let b_scale = vec![1.0f32];
/// let result  = vec_dot_q4_0_q8_0(&a_q4, &a_scale, &b_q8, &b_scale);
/// assert!((result - 0.0).abs() < 1e-6);
/// ```
pub fn vec_dot_q4_0_q8_0(
    a_q4: &[u8],
    a_scales: &[f32],
    b_q8: &[i8],
    b_scales: &[f32],
) -> f32 {
    // --- input validation ---
    let n_blocks = a_scales.len();
    assert_eq!(
        a_q4.len(),
        n_blocks * 16,
        "a_q4.len() must equal a_scales.len() * 16 (got {} vs {} * 16 = {})",
        a_q4.len(),
        n_blocks,
        n_blocks * 16,
    );
    assert_eq!(
        b_q8.len(),
        n_blocks * 32,
        "b_q8.len() must equal b_scales.len() * 32 (got {} vs {} * 32 = {})",
        b_q8.len(),
        n_blocks,
        n_blocks * 32,
    );
    assert_eq!(
        b_scales.len(),
        n_blocks,
        "a_scales.len() must equal b_scales.len() (got {} vs {})",
        n_blocks,
        b_scales.len(),
    );

    // TODO(ws2): AVX2 SIMD via _mm256_maddubs_epi16 lane pattern (see candle/quantized/avx.rs)
    // TODO(ws2): NEON SIMD via vmull_s8 / vaddlvq_s16 lane pattern (see candle/quantized/neon.rs)

    // --- scalar reference implementation ---
    //
    // For each block b:
    //   sum_i = Σ_{j=0}^{15}  (lo_nibble(a[j]) - 8) * b_q8[j]
    //         + Σ_{j=0}^{15}  (hi_nibble(a[j]) - 8) * b_q8[j + 16]
    //   result += sum_i as f32 * a_scales[b] * b_scales[b]
    //
    // This mirrors `BlockQ4_0::vec_dot_unopt` in candle-core/src/quantized/k_quants.rs.

    let mut sumf = 0.0f32;

    for b in 0..n_blocks {
        let q4_block = &a_q4[b * 16..(b + 1) * 16]; // 16 packed bytes → 32 nibbles
        let q8_block = &b_q8[b * 32..(b + 1) * 32]; // 32 i8 values

        let mut sum_i: i32 = 0;
        for j in 0..16 {
            let byte = q4_block[j];
            // low nibble → element j; high nibble → element j+16
            let v0 = (byte & 0x0F) as i32 - 8;
            let v1 = ((byte >> 4) & 0x0F) as i32 - 8;
            sum_i += v0 * (q8_block[j] as i32);
            sum_i += v1 * (q8_block[j + 16] as i32);
        }

        sumf += sum_i as f32 * a_scales[b] * b_scales[b];
    }

    sumf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::quantized::quantize_f32_to_q4_0;

    // -----------------------------------------------------------------------
    // Helper: encode a &[f32] slice into Q8_0 format (scale + i8 values).
    // Q8_0: d = max(|x|) / 127.0 ; q = clamp(round(x / d), -127, 127).
    // Returns (q8_bytes: Vec<i8>, scales: Vec<f32>).
    // -----------------------------------------------------------------------
    fn quantize_f32_to_q8_0(data: &[f32]) -> (Vec<i8>, Vec<f32>) {
        assert!(
            data.len() % QK8_0 == 0,
            "Q8_0 requires data.len() to be a multiple of {}",
            QK8_0
        );
        let n_blocks = data.len() / QK8_0;
        let mut q8 = Vec::with_capacity(data.len());
        let mut scales = Vec::with_capacity(n_blocks);

        for b in 0..n_blocks {
            let block = &data[b * QK8_0..(b + 1) * QK8_0];
            let amax = block.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            scales.push(d);
            for &x in block {
                let q = (x * id).round().clamp(-127.0, 127.0) as i8;
                q8.push(q);
            }
        }
        (q8, scales)
    }

    // -----------------------------------------------------------------------
    // Test 1 — one block, all zeros
    // -----------------------------------------------------------------------
    #[test]
    fn q4_0_q8_0_one_block_zeros() {
        // Nibble 8 → 8 - 8 = 0 for all 32 elements.
        let a_q4 = vec![0x88u8; 16];
        let a_scales = vec![1.0f32];
        let b_q8 = vec![0i8; 32];
        let b_scales = vec![1.0f32];

        let result = vec_dot_q4_0_q8_0(&a_q4, &a_scales, &b_q8, &b_scales);
        assert!(
            (result - 0.0).abs() < 1e-9,
            "expected 0.0, got {result}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 2 — one block, all ones (verify scale arithmetic)
    // Q4_0: nibble = 9 → signed value = 9 - 8 = 1; scale = 1.0 → dequant = 1.
    // Q8_0: byte = 1; scale = 1.0 → dequant = 1.
    // Dot = 32 * 1 * 1 * (1.0 * 1.0) = 32.
    // -----------------------------------------------------------------------
    #[test]
    fn q4_0_q8_0_one_block_ones() {
        // Nibble 9 in both lo and hi → both signed values = +1.
        let a_q4 = vec![0x99u8; 16]; // lo nibble = 9, hi nibble = 9
        let a_scales = vec![1.0f32];
        let b_q8 = vec![1i8; 32];
        let b_scales = vec![1.0f32];

        let result = vec_dot_q4_0_q8_0(&a_q4, &a_scales, &b_q8, &b_scales);
        // 16 bytes × 2 nibbles each × (9-8)=1 × q8=1 × 1.0 × 1.0 = 32
        assert!(
            (result - 32.0).abs() < 1e-6,
            "expected 32.0, got {result}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 3 — two blocks with different scales
    // Block 0: nibble=9 (+1), q8=2, scale_a=3.0, scale_b=0.5
    //          sum_i = 32 * 1 * 2 = 64  → 64 * 3.0 * 0.5 = 96
    // Block 1: nibble=10 (+2), q8=1, scale_a=1.0, scale_b=1.0
    //          sum_i = 32 * 2 * 1 = 64  → 64 * 1.0 * 1.0 = 64
    // Total = 160.
    // -----------------------------------------------------------------------
    #[test]
    fn q4_0_q8_0_two_blocks() {
        let mut a_q4 = vec![0x99u8; 16]; // block 0: nibble 9
        a_q4.extend(vec![0xAAu8; 16]);   // block 1: nibble 10
        let a_scales = vec![3.0f32, 1.0f32];

        let mut b_q8 = vec![2i8; 32]; // block 0: q8 = 2
        b_q8.extend(vec![1i8; 32]);   // block 1: q8 = 1
        let b_scales = vec![0.5f32, 1.0f32];

        let result = vec_dot_q4_0_q8_0(&a_q4, &a_scales, &b_q8, &b_scales);
        let expected = 96.0f32 + 64.0f32;
        assert!(
            (result - expected).abs() < 1e-4,
            "expected {expected}, got {result}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 4 — round-trip using quantize_f32_to_q4_0 from hpc::quantized
    // Encodes a pseudo-random f32 vector with Q4_0, encodes the same vector
    // with Q8_0, computes the quantized dot, and checks that the relative
    // error against the exact f32 dot is < 0.05.
    // -----------------------------------------------------------------------
    #[test]
    fn q4_0_q8_0_round_trip() {
        // Generate a deterministic pseudo-random f32 vector (4 blocks = 128 elements).
        let n = 128usize;
        let mut data_a = Vec::with_capacity(n);
        let mut data_b = Vec::with_capacity(n);
        let mut seed_a = 0x1234_5678u64;
        let mut seed_b = 0xDEAD_BEEFu64;
        for _ in 0..n {
            // LCG step
            seed_a = seed_a.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed_b = seed_b.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to [-2.0, 2.0]
            let fa = ((seed_a >> 33) as f32) / (u32::MAX as f32) * 4.0 - 2.0;
            let fb = ((seed_b >> 33) as f32) / (u32::MAX as f32) * 4.0 - 2.0;
            data_a.push(fa);
            data_b.push(fb);
        }

        // Exact f32 dot product
        let exact_dot: f32 = data_a.iter().zip(data_b.iter()).map(|(&a, &b)| a * b).sum();

        // Quantize a → Q4_0, b → Q8_0
        let (a_q4, a_scales) = quantize_f32_to_q4_0(&data_a);
        let (b_q8, b_scales) = quantize_f32_to_q8_0(&data_b);

        let quantized_dot = vec_dot_q4_0_q8_0(&a_q4, &a_scales, &b_q8, &b_scales);

        // Relative error < 5 %
        let rel_err = if exact_dot.abs() > 1e-6 {
            (quantized_dot - exact_dot).abs() / exact_dot.abs()
        } else {
            (quantized_dot - exact_dot).abs()
        };
        assert!(
            rel_err < 0.05,
            "relative error {rel_err:.4} ≥ 0.05 (exact={exact_dot:.4}, quantized={quantized_dot:.4})"
        );
    }

    // -----------------------------------------------------------------------
    // Test 5 — signed nibbles / negative values exercise the -8 bias
    // Nibble 0 → 0 - 8 = -8; q8 = 1; scale_a = 1.0, scale_b = 1.0
    // 32 elements × (-8) × 1 × 1 × 1 = -256.
    // -----------------------------------------------------------------------
    #[test]
    fn q4_0_q8_0_signed_nibbles() {
        // All nibbles = 0 → signed value = -8 for all 32 elements.
        let a_q4 = vec![0x00u8; 16];
        let a_scales = vec![1.0f32];
        let b_q8 = vec![1i8; 32];
        let b_scales = vec![1.0f32];

        let result = vec_dot_q4_0_q8_0(&a_q4, &a_scales, &b_q8, &b_scales);
        // 16 bytes × 2 nibbles each × (-8) × 1 × 1.0 × 1.0 = -256
        assert!(
            (result - (-256.0)).abs() < 1e-6,
            "expected -256.0, got {result}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 6 — mixed positive and negative nibbles cancel out
    // Byte = 0x0F: lo nibble = 0 → -8, hi nibble = F (15) → +7
    // sum per byte = -8 * q8_lo + 7 * q8_hi
    // With q8 = [1; 32] and all bytes = 0x0F:
    //   sum_i = 16 * (-8 + 7) = 16 * (-1) = -16
    //   result = -16 * 1.0 * 1.0 = -16
    // -----------------------------------------------------------------------
    #[test]
    fn q4_0_q8_0_mixed_nibbles() {
        let a_q4 = vec![0x0Fu8; 16]; // lo=0(-8), hi=F(+7)
        let a_scales = vec![1.0f32];
        let b_q8 = vec![1i8; 32];
        let b_scales = vec![1.0f32];

        let result = vec_dot_q4_0_q8_0(&a_q4, &a_scales, &b_q8, &b_scales);
        // 16 × ((-8)*1 + 7*1) = 16 × (-1) = -16
        assert!(
            (result - (-16.0)).abs() < 1e-6,
            "expected -16.0, got {result}"
        );
    }

    // -----------------------------------------------------------------------
    // Test 7 — scale multiplication: non-trivial scales
    // Nibble 9 → +1, q8 = 3, scale_a = 2.5, scale_b = 0.4
    // sum_i = 32 * 1 * 3 = 96
    // result = 96 * 2.5 * 0.4 = 96.0
    // -----------------------------------------------------------------------
    #[test]
    fn q4_0_q8_0_scale_multiplication() {
        let a_q4 = vec![0x99u8; 16];
        let a_scales = vec![2.5f32];
        let b_q8 = vec![3i8; 32];
        let b_scales = vec![0.4f32];

        let result = vec_dot_q4_0_q8_0(&a_q4, &a_scales, &b_q8, &b_scales);
        let expected = 96.0f32 * 2.5 * 0.4;
        assert!(
            (result - expected).abs() < 1e-4,
            "expected {expected}, got {result}"
        );
    }
}
