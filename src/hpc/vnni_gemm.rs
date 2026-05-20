//! VNNI-accelerated INT8 GEMM: C += A x B where A is u8, B is i8, C is i32.
//!
//! Uses `VPDPBUSD` (AVX-512 VNNI) to compute 4-element u8*i8 dot products
//! in a single instruction, accumulating into i32. Falls back to the scalar
//! [`int8_gemm_i32`](super::quantized::int8_gemm_i32) on hardware without
//! VNNI support.
//!
//! # VNNI dot semantics
//!
//! For each 32-bit lane, `VPDPBUSD` takes 4 consecutive u8 values from `a`
//! and 4 consecutive i8 values from `b`, computes:
//!
//! ```text
//! acc[lane] += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
//! ```
//!
//! With 16 lanes per zmm register, that is 64 multiply-accumulates per
//! instruction.
//!
//! # Performance
//!
//! On Sapphire Rapids / Zen 4 without AMX, this kernel provides ~4x
//! throughput vs scalar `int8_gemm_i32` for medium matrices (32x32 and up).

use super::quantized::int8_gemm_i32;
use super::simd_caps::simd_caps;

/// VNNI-accelerated INT8 GEMM: C = A * B where A is u8, B is i8, C is i32.
///
/// Uses VPDPBUSD (AVX-512 VNNI) to compute 4-element dot products
/// in a single instruction. Falls back to the scalar `int8_gemm_i32` on
/// hardware without VNNI support.
///
/// # Arguments
///
/// * `a` - M x K matrix, row-major, u8 values
/// * `b` - K x N matrix, row-major, i8 values
/// * `c` - M x N output matrix, row-major, i32 values (overwritten, not accumulated)
/// * `m` - number of rows in A / C
/// * `n` - number of columns in B / C
/// * `k` - inner dimension (columns of A, rows of B)
///
/// # Panics
///
/// Panics if the slice lengths are inconsistent with the given dimensions.
pub fn int8_gemm_vnni(a: &[u8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    assert!(a.len() >= m * k, "a.len()={} < m*k={}", a.len(), m * k);
    assert!(b.len() >= k * n, "b.len()={} < k*n={}", b.len(), k * n);
    assert!(c.len() >= m * n, "c.len()={} < m*n={}", c.len(), m * n);

    #[cfg(target_arch = "x86_64")]
    {
        let caps = simd_caps();
        if caps.has_avx512_vnni() {
            unsafe { int8_gemm_vnni_avx512(a, b, c, m, n, k) }
            return;
        }
    }
    // Scalar fallback
    int8_gemm_i32(a, b, c, m, n, k);
}

/// Returns true if VNNI (AVX-512 VNNI) is available on this CPU.
///
/// Useful for tests and benchmarks that want to report whether the
/// accelerated path was taken.
pub fn has_vnni() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        simd_caps().has_avx512_vnni()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

// ── AVX-512 VNNI inner kernel ─────────────────────────────────────────────

/// AVX-512 VNNI GEMM inner kernel.
///
/// Strategy:
/// - For each row i of A, for each group of 16 columns j..j+16 of C:
///   - Accumulate VPDPBUSD over K in groups of 4
///   - VPDPBUSD needs: a_broadcast = 4 bytes of A[i,p..p+4] broadcast to all lanes
///     and b_col = 16 groups of 4 bytes from B columns j..j+16 at rows p..p+4
///   - B is row-major, so B[p,j..j+16] are 16 contiguous i8 values, but we need
///     4 consecutive rows interleaved: for lane L, bytes are [B[p,j+L], B[p+1,j+L],
///     B[p+2,j+L], B[p+3,j+L]].
///   - We pre-pack B into VNNI layout: b_packed[p/4][j..j+16] where each i32
///     contains 4 bytes from consecutive rows.
/// AVX-512 VNNI INT8 GEMM kernel — `pub(crate)` so the agnostic
/// `simd_int_ops::gemm_u8_i8` surface can call it directly under a
/// compile-time `target_feature = "avx512vnni"` gate, bypassing the
/// per-call caps branch in [`int8_gemm_vnni`]. See § "compile-time
/// dispatch table" in `.claude/knowledge/td-simd-integration-plan.md`.
///
/// # Safety
///
/// Caller must guarantee the CPU supports AVX-512F + AVX-512VNNI +
/// AVX-512BW. Compile-time gating via `#[cfg(target_feature = …)]` at
/// the call site is the standard contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni,avx512bw")]
pub(crate) unsafe fn int8_gemm_vnni_avx512(a: &[u8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    use core::arch::x86_64::*;

    // Zero output
    for v in c.iter_mut() {
        *v = 0;
    }

    // Pre-pack B into VNNI layout: groups of 4 rows, each i32 lane holds
    // [b[p+0,j], b[p+1,j], b[p+2,j], b[p+3,j]] as 4 bytes.
    // Dimensions: k_groups x n i32 values
    let k_groups = (k + 3) / 4;
    let mut b_packed = vec![0i32; k_groups * n];

    for pg in 0..k_groups {
        let p_base = pg * 4;
        for j in 0..n {
            let mut bytes = [0u8; 4];
            for q in 0..4 {
                let p = p_base + q;
                if p < k {
                    // Cast i8 to u8 for byte packing; VPDPBUSD interprets
                    // the second operand as i8 regardless.
                    bytes[q] = b[p * n + j] as u8;
                }
            }
            b_packed[pg * n + j] = i32::from_le_bytes(bytes);
        }
    }

    // Main GEMM loop
    for i in 0..m {
        // Process columns in chunks of 16 (zmm width for i32)
        let mut j = 0;
        while j + 16 <= n {
            let mut acc = _mm512_setzero_si512();

            for pg in 0..k_groups {
                let p_base = pg * 4;

                // Load 4 bytes of A[i, p_base..p_base+4], broadcast as i32
                let mut a_bytes = [0u8; 4];
                for q in 0..4 {
                    let p = p_base + q;
                    if p < k {
                        a_bytes[q] = a[i * k + p];
                    }
                }
                let a_val = u32::from_le_bytes(a_bytes) as i32;
                let a_broadcast = _mm512_set1_epi32(a_val);

                // Load 16 packed i32 values from b_packed
                let b_ptr = b_packed.as_ptr().add(pg * n + j);
                let b_vec = _mm512_loadu_si512(b_ptr as *const __m512i);

                // VPDPBUSD: acc += dot4(a_broadcast, b_vec) per lane
                acc = _mm512_dpbusd_epi32(acc, a_broadcast, b_vec);
            }

            // Store 16 i32 results
            _mm512_storeu_si512(c.as_mut_ptr().add(i * n + j) as *mut __m512i, acc);

            j += 16;
        }

        // Handle remaining columns (j..n where n-j < 16)
        if j < n {
            let remaining = n - j;

            // Use masked operations for the tail
            let mask: u16 = (1u32 << remaining).wrapping_sub(1) as u16;
            let kmask = __mmask16::from(mask);
            let mut acc = _mm512_setzero_si512();

            for pg in 0..k_groups {
                let p_base = pg * 4;

                let mut a_bytes = [0u8; 4];
                for q in 0..4 {
                    let p = p_base + q;
                    if p < k {
                        a_bytes[q] = a[i * k + p];
                    }
                }
                let a_val = u32::from_le_bytes(a_bytes) as i32;
                let a_broadcast = _mm512_set1_epi32(a_val);

                // Masked load of remaining b_packed values
                let b_ptr = b_packed.as_ptr().add(pg * n + j);
                let b_vec = _mm512_maskz_loadu_epi32(kmask, b_ptr as *const i32);

                acc = _mm512_dpbusd_epi32(acc, a_broadcast, b_vec);
            }

            // Masked store
            _mm512_mask_storeu_epi32(c.as_mut_ptr().add(i * n + j) as *mut i32, kmask, acc);
        }
    }
}

// ── AVX-VNNI (ymm) inner kernel ──────────────────────────────────────────

/// AVX-VNNI (256-bit ymm) INT8 GEMM kernel.
///
/// VEX-encoded `VPDPBUSD` over 8-wide i32 accumulators, for Alder Lake /
/// Arrow Lake / Zen 4 / Sapphire Rapids (whenever the dispatcher resolves
/// to AVX2 + AVX-VNNI without selecting the AVX-512 zmm path). Half the
/// lane count of [`int8_gemm_vnni_avx512`], and the VEX encoding has no
/// masked load/store, so the column tail (`n % 8 != 0`) runs scalar.
///
/// `pub(crate)` so [`crate::simd_int_ops::gemm_u8_i8`] can target it
/// directly under a compile-time `target_feature = "avxvnni"` gate.
///
/// # Safety
///
/// Caller must guarantee the CPU supports AVX + AVX2 + AVX-VNNI.
/// Compile-time gating via `#[cfg(target_feature = "avxvnni")]` at the
/// call site is the standard contract.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,avxvnni")]
pub(crate) unsafe fn int8_gemm_avxvnni_ymm(a: &[u8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    use core::arch::x86_64::*;

    // Zero output
    for v in c.iter_mut() {
        *v = 0;
    }

    // Pre-pack B into VNNI layout: groups of 4 rows, each i32 lane holds
    // [b[p+0,j], b[p+1,j], b[p+2,j], b[p+3,j]] as 4 bytes.
    let k_groups = (k + 3) / 4;
    let mut b_packed = vec![0i32; k_groups * n];

    for pg in 0..k_groups {
        let p_base = pg * 4;
        for j in 0..n {
            let mut bytes = [0u8; 4];
            for q in 0..4 {
                let p = p_base + q;
                if p < k {
                    bytes[q] = b[p * n + j] as u8;
                }
            }
            b_packed[pg * n + j] = i32::from_le_bytes(bytes);
        }
    }

    // Main GEMM loop — 8 i32 columns per ymm register.
    for i in 0..m {
        let mut j = 0;
        while j + 8 <= n {
            let mut acc = _mm256_setzero_si256();

            for pg in 0..k_groups {
                let p_base = pg * 4;

                let mut a_bytes = [0u8; 4];
                for q in 0..4 {
                    let p = p_base + q;
                    if p < k {
                        a_bytes[q] = a[i * k + p];
                    }
                }
                let a_val = u32::from_le_bytes(a_bytes) as i32;
                let a_broadcast = _mm256_set1_epi32(a_val);

                let b_ptr = b_packed.as_ptr().add(pg * n + j);
                let b_vec = _mm256_loadu_si256(b_ptr as *const __m256i);

                // VEX-encoded VPDPBUSD: acc += dot4(a_broadcast, b_vec) per lane.
                acc = _mm256_dpbusd_avx_epi32(acc, a_broadcast, b_vec);
            }

            _mm256_storeu_si256(c.as_mut_ptr().add(i * n + j) as *mut __m256i, acc);
            j += 8;
        }

        // Scalar tail for `n - j < 8` columns — no masked ymm VPDPBUSD on VEX.
        while j < n {
            let mut sum = 0i32;
            for p in 0..k {
                sum += (a[i * k + p] as i32) * (b[p * n + j] as i32);
            }
            c[i * n + j] = sum;
            j += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference scalar GEMM for verification.
    fn scalar_gemm(a: &[u8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for p in 0..k {
                let a_val = a[i * k + p] as i32;
                for j in 0..n {
                    c[i * n + j] += a_val * b[p * n + j] as i32;
                }
            }
        }
        c
    }

    #[test]
    fn test_vnni_gemm_4x4() {
        let m = 4;
        let n = 4;
        let k = 4;
        // Simple identity-like test
        let a: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let b: Vec<i8> = vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let expected = scalar_gemm(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        int8_gemm_vnni(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected, "4x4 GEMM mismatch");
    }

    #[test]
    fn test_vnni_gemm_4x4_mixed_values() {
        let m = 4;
        let n = 4;
        let k = 4;
        let a: Vec<u8> = vec![128, 64, 32, 16, 255, 0, 128, 64, 1, 2, 3, 4, 200, 100, 50, 25];
        let b: Vec<i8> = vec![1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8];
        let expected = scalar_gemm(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        int8_gemm_vnni(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected, "4x4 mixed values GEMM mismatch");
    }

    #[test]
    fn test_vnni_gemm_16x16() {
        let m = 16;
        let n = 16;
        let k = 16;
        let a: Vec<u8> = (0..m * k).map(|i| (i % 251) as u8).collect();
        let b: Vec<i8> = (0..k * n)
            .map(|i| ((i % 127) as i8).wrapping_sub(63))
            .collect();
        let expected = scalar_gemm(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        int8_gemm_vnni(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected, "16x16 GEMM mismatch");
    }

    #[test]
    fn test_vnni_gemm_17x17_tail() {
        let m = 17;
        let n = 17;
        let k = 17;
        let a: Vec<u8> = (0..m * k).map(|i| ((i * 7 + 3) % 256) as u8).collect();
        let b: Vec<i8> = (0..k * n)
            .map(|i| ((i * 11 + 5) % 256) as u8 as i8)
            .collect();
        let expected = scalar_gemm(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        int8_gemm_vnni(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected, "17x17 (tail handling) GEMM mismatch");
    }

    #[test]
    fn test_vnni_gemm_1x1() {
        let a: Vec<u8> = vec![200];
        let b: Vec<i8> = vec![-50];
        let expected = scalar_gemm(&a, &b, 1, 1, 1);
        let mut c = vec![0i32; 1];
        int8_gemm_vnni(&a, &b, &mut c, 1, 1, 1);
        assert_eq!(c, expected, "1x1 GEMM mismatch");
    }

    #[test]
    fn test_vnni_gemm_rectangular() {
        // M=3, N=5, K=8 — non-square, non-power-of-2
        let m = 3;
        let n = 5;
        let k = 8;
        let a: Vec<u8> = (0..m * k).map(|i| (i % 200) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 100) as i8 - 50).collect();
        let expected = scalar_gemm(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        int8_gemm_vnni(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected, "3x5x8 rectangular GEMM mismatch");
    }

    #[test]
    fn test_vnni_gemm_64x64() {
        let m = 64;
        let n = 64;
        let k = 64;
        let a: Vec<u8> = (0..m * k).map(|i| (i % 256) as u8).collect();
        let b: Vec<i8> = (0..k * n)
            .map(|i| ((i * 3 + 7) % 256) as u8 as i8)
            .collect();
        let expected = scalar_gemm(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        int8_gemm_vnni(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected, "64x64 GEMM mismatch");
    }

    #[test]
    fn test_vnni_gemm_zero_matrices() {
        let m = 8;
        let n = 8;
        let k = 8;
        let a = vec![0u8; m * k];
        let b = vec![0i8; k * n];
        let mut c = vec![99i32; m * n]; // pre-fill with non-zero
        int8_gemm_vnni(&a, &b, &mut c, m, n, k);
        assert!(c.iter().all(|&v| v == 0), "zero input should produce zero output");
    }

    #[test]
    fn test_vnni_reports_capability() {
        // Just verify has_vnni() doesn't panic and returns a bool
        let _vnni = has_vnni();
    }

    #[test]
    fn test_vnni_gemm_k_not_multiple_of_4() {
        // K=6: tests the zero-padding for the last incomplete 4-group
        let m = 4;
        let n = 4;
        let k = 6;
        let a: Vec<u8> = (0..m * k).map(|i| ((i + 1) % 256) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i + 1) % 127) as i8).collect();
        let expected = scalar_gemm(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        int8_gemm_vnni(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected, "K=6 (not multiple of 4) GEMM mismatch");
    }

    #[test]
    fn test_vnni_gemm_large_values() {
        // Stress test with max u8 and extreme i8 values
        let m = 4;
        let n = 4;
        let k = 8;
        let a = vec![255u8; m * k];
        let b: Vec<i8> = (0..k * n)
            .map(|i| if i % 2 == 0 { 127i8 } else { -128i8 })
            .collect();
        let expected = scalar_gemm(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        int8_gemm_vnni(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected, "large values GEMM mismatch");
    }
}
