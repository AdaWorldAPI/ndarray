//! BF16 tile GEMM polyfill — AMX (TDPBF16PS) with AVX-512 F32x16 fallback.
//!
//! Same API, runtime tier dispatch via `amx_available()`. The AMX path uses
//! the raw primitives in `hpc::amx_matmul`. The fallback decodes BF16→f32
//! and uses `crate::simd::F32x16` + `mul_add` (VFMADD231PS on AVX-512,
//! emulated as 2× F32x8 FMA on AVX2).
//!
//! Pattern: one dispatch check per call; caller supplies preallocated
//! output and (for AMX) VNNI-packed B.
//!
//! Tile shape: M=16, N=16, K = multiple of 32.
//!
//! Usage:
//! ```ignore
//! use ndarray::hpc::bf16_tile_gemm::bf16_tile_gemm_16x16;
//! let mut c = vec![0.0f32; 16*16];
//! bf16_tile_gemm_16x16(&a_bf16, &b_bf16_row_major, &mut c, k);
//! ```

use crate::hpc::amx_matmul::{
    amx_available, TileConfig, tile_loadconfig, tile_zero,
    tile_load, tile_store, tile_release, tile_dpbf16ps, vnni_pack_bf16,
};
use crate::simd::{F32x16, bf16_to_f32_batch};

// ═════════════════════════════════════════════════════════════════════
// Public API — safe dispatching wrapper
// ═════════════════════════════════════════════════════════════════════

/// Compute C[16, 16] += A[16, K] × B[K, 16] where A, B are BF16 row-major
/// and C is f32 row-major. K must be a multiple of 32.
///
/// Tier dispatch (runtime):
///   AMX available  → TDPBF16PS tile GEMM  (16×16 × K/32 tile iterations)
///   AMX unavailable → AVX-512 F32x16 FMA fallback (decode BF16→f32, gemm)
///
/// Both paths produce identical results up to BF16 precision (~1/128 per
/// multiply, O(sqrt(K)) accumulated).
pub fn bf16_tile_gemm_16x16(a_bf16: &[u16], b_bf16: &[u16], c: &mut [f32], k: usize) {
    assert_eq!(k % 32, 0, "K must be multiple of 32");
    assert_eq!(a_bf16.len(), 16 * k);
    assert_eq!(b_bf16.len(), k * 16);
    assert_eq!(c.len(), 16 * 16);

    if amx_available() {
        // AMX path: pack B into VNNI, call tile GEMM
        let mut b_vnni = vec![0u16; k * 16];
        vnni_pack_bf16(b_bf16, &mut b_vnni, k, 16);
        // SAFETY: amx_available() just confirmed CPUID + XCR0 + prctl.
        unsafe { amx_path(a_bf16, &b_vnni, c, k); }
    } else {
        fallback_path(a_bf16, b_bf16, c, k);
    }
}

// ═════════════════════════════════════════════════════════════════════
// AMX path (TDPBF16PS)
// ═════════════════════════════════════════════════════════════════════

/// AMX tile GEMM. B must be pre-VNNI-packed (see `vnni_pack_bf16`).
/// # Safety
/// Caller must have verified `amx_available() == true`.
#[inline]
unsafe fn amx_path(a_bf16: &[u16], b_vnni: &[u16], c: &mut [f32], k: usize) {
    // Tile config: shapes at K_bytes=64 match BF16 K=32 case
    let cfg = TileConfig::for_dpbusd(64);
    tile_loadconfig(&cfg);
    tile_zero(0);

    // Accumulate over K/32 tile blocks
    let k_blocks = k / 32;
    let a_stride = (k * 2) as usize;    // full A row stride in bytes (bf16 = 2B)
    let b_stride = 64usize;             // VNNI row stride in bytes

    for kb in 0..k_blocks {
        let a_ptr = a_bf16.as_ptr().add(kb * 32) as *const u8;
        let b_ptr = b_vnni.as_ptr().add(kb * 16 * 32) as *const u8;
        tile_load(1, a_ptr, a_stride);
        tile_load(2, b_ptr, b_stride);
        tile_dpbf16ps();
    }

    tile_store(0, c.as_mut_ptr() as *mut u8, 64);
    tile_release();
}

// ═════════════════════════════════════════════════════════════════════
// AVX-512 fallback (F32x16 + mul_add FMA)
// ═════════════════════════════════════════════════════════════════════

/// Fallback: decode BF16→f32 and run a tight F32x16 GEMM with mul_add FMA.
/// When AVX-512 is the compile-time baseline, this uses native __m512 FMA;
/// on AVX2 it uses the emulated F32x16 = (F32x8, F32x8) pair — same logic.
fn fallback_path(a_bf16: &[u16], b_bf16: &[u16], c: &mut [f32], k: usize) {
    // Decode BF16 → f32 (batch via SIMD when avx512bf16 / avx2 available)
    let mut a_f32 = vec![0.0f32; a_bf16.len()];
    let mut b_f32 = vec![0.0f32; b_bf16.len()];
    bf16_to_f32_batch(a_bf16, &mut a_f32);
    bf16_to_f32_batch(b_bf16, &mut b_f32);

    // Tight GEMM: for each output (i,j), dot row-of-A with col-of-B via F32x16+FMA.
    // B is row-major [K, 16]; j-th column is b_f32[kk*16 + j] over kk=0..K.
    // We gather the column into a stack-sized buffer once per (i,j) pair to hit
    // the chunks_exact(16) + mul_add fast path on contiguous memory.
    for i in 0..16 {
        let a_row = &a_f32[i * k .. i * k + k];
        for j in 0..16 {
            // Stream the column into a contiguous buffer
            let mut col = vec![0.0f32; k];
            for kk in 0..k { col[kk] = b_f32[kk * 16 + j]; }

            // Accumulate via F32x16::mul_add (FMA)
            let mut acc = F32x16::splat(0.0);
            for (ra, rb) in a_row.chunks_exact(16).zip(col.chunks_exact(16)) {
                let av = F32x16::from_slice(ra);
                let bv = F32x16::from_slice(rb);
                acc = av.mul_add(bv, acc);
            }
            c[i * 16 + j] += acc.reduce_sum();
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{f32_to_bf16_batch, bf16_to_f32_batch};

    /// Scalar BF16 reference (f32-accumulated) — ground truth.
    fn ref_gemm(a: &[f32], b: &[f32], c: &mut [f32], k: usize) {
        for i in 0..16 {
            for j in 0..16 {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * 16 + j];
                }
                c[i * 16 + j] = s;
            }
        }
    }

    #[test]
    fn fallback_matches_scalar_reference_k64() {
        let k = 64;
        // Deterministic pseudo-random inputs
        let mut a_f32 = vec![0.0f32; 16 * k];
        let mut b_f32 = vec![0.0f32; k * 16];
        for i in 0..a_f32.len() {
            a_f32[i] = (((i as i32).wrapping_mul(1103515245).wrapping_add(12345) >> 8) as f32
                        / 2147483648.0).clamp(-1.0, 1.0);
        }
        for i in 0..b_f32.len() {
            b_f32[i] = (((i as i32).wrapping_mul(69069).wrapping_add(1) >> 8) as f32
                        / 2147483648.0).clamp(-1.0, 1.0);
        }
        let mut a_bf16 = vec![0u16; a_f32.len()];
        let mut b_bf16 = vec![0u16; b_f32.len()];
        f32_to_bf16_batch(&a_f32, &mut a_bf16);
        f32_to_bf16_batch(&b_f32, &mut b_bf16);

        // Reference uses bf16-truncated inputs (matches what the GEMM sees)
        let mut a_back = vec![0.0f32; a_f32.len()];
        let mut b_back = vec![0.0f32; b_f32.len()];
        bf16_to_f32_batch(&a_bf16, &mut a_back);
        bf16_to_f32_batch(&b_bf16, &mut b_back);
        let mut c_ref = vec![0.0f32; 16 * 16];
        ref_gemm(&a_back, &b_back, &mut c_ref, k);

        // Fallback GEMM
        let mut c_fb = vec![0.0f32; 16 * 16];
        fallback_path(&a_bf16, &b_bf16, &mut c_fb, k);

        // Compare — should match exactly (same arithmetic, f32 precision)
        let mut max_err = 0.0f32;
        for i in 0..(16 * 16) {
            let e = (c_fb[i] - c_ref[i]).abs();
            if e > max_err { max_err = e; }
        }
        assert!(max_err < 1e-3, "fallback vs scalar ref max_err = {}", max_err);
    }

    #[test]
    fn public_api_runs_on_any_hardware() {
        // Just sanity: calling the public API doesn't panic regardless of AMX.
        // On AMX hardware it takes the tile path; on this test host likely fallback.
        let k = 32;
        let a = vec![0u16; 16 * k];
        let b = vec![0u16; k * 16];
        let mut c = vec![0.0f32; 16 * 16];
        bf16_tile_gemm_16x16(&a, &b, &mut c, k);
        // All zeros × all zeros = 0
        for v in c.iter() { assert_eq!(*v, 0.0); }
    }
}
