//! INT8 tile GEMM polyfill — AMX (TDPBUSD) tile kernel.
//!
//! Mirror of `hpc::bf16_tile_gemm` for the `u8 × i8 → i32` shape, the
//! native TDPBUSD operand type. One TDPBUSD: 16×16 output tile × 64
//! K-elements per A row × 4 K-elements per inner product = **16 384
//! multiply-accumulates per instruction**. That's 256× the VPDPBUSD
//! zmm throughput per instruction (which does 16 × 4 = 64 MACs).
//!
//! Public surface:
//!   * [`int8_tile_gemm_16x16`] — the 16×16 tile kernel; M=16, N=16,
//!     K a multiple of 64. AMX path requires runtime feature
//!     detection (`amx_available()`); falls back to a scalar reference
//!     when AMX isn't OS-enabled.
//!
//! Caller responsibility:
//!   * B comes in row-major K × 16 i8; the kernel pre-packs it into
//!     VNNI quad layout via [`super::amx_matmul::vnni_pack_i8`].
//!   * A is row-major 16 × K u8 (TDPBUSD's unsigned operand).
//!   * C accumulates into the caller's i32 buffer (16 × 16 = 256 i32).
//!
//! Same shape as `bf16_tile_gemm::bf16_tile_gemm_16x16`. The two kernels
//! together cover the SPR/GNR AMX dispatch tier for both `BF16 × BF16
//! → f32` and `u8 × i8 → i32` — the two operand families that AMX
//! supports natively.

use crate::hpc::amx_matmul::{
    amx_available, tile_dpbusd, tile_load, tile_loadconfig, tile_release, tile_store, tile_zero, vnni_pack_i8,
    TileConfig,
};

// ═════════════════════════════════════════════════════════════════════
// Public API — safe dispatching wrapper
// ═════════════════════════════════════════════════════════════════════

/// Compute C[16, 16] += A[16, K] × B[K, 16] where A is u8 row-major,
/// B is i8 row-major, C is i32 row-major. K must be a multiple of 64.
///
/// Tier dispatch (runtime):
///   AMX available   → TDPBUSD tile GEMM  (16×16 × K/64 tile iterations,
///                                          16 384 MACs per instruction)
///   AMX unavailable → scalar u8 × i8 → i32 reference
///
/// Output behavior: this function **accumulates** into `c` (does NOT
/// zero it first). Callers wanting fresh `C = A·B` semantics should
/// zero `c` before calling, the same convention `bf16_tile_gemm_16x16`
/// uses.
pub fn int8_tile_gemm_16x16(a_u8: &[u8], b_i8: &[i8], c: &mut [i32], k: usize) {
    assert_eq!(k % 64, 0, "K must be multiple of 64 for TDPBUSD tiles");
    assert_eq!(a_u8.len(), 16 * k);
    assert_eq!(b_i8.len(), k * 16);
    assert_eq!(c.len(), 16 * 16);

    if amx_available() {
        // AMX path: pack B into VNNI quad layout, call tile GEMM.
        let mut b_vnni = vec![0i8; k * 16];
        vnni_pack_i8(b_i8, &mut b_vnni, k, 16);
        // SAFETY: amx_available() just confirmed CPUID + XCR0 + prctl.
        unsafe {
            amx_path(a_u8, &b_vnni, c, k);
        }
    } else {
        fallback_path(a_u8, b_i8, c, k);
    }
}

// ═════════════════════════════════════════════════════════════════════
// AMX path (TDPBUSD)
// ═════════════════════════════════════════════════════════════════════

/// AMX tile GEMM. B must be pre-VNNI-packed (see `vnni_pack_i8`).
/// # Safety
/// Caller must have verified `amx_available() == true`.
#[inline]
unsafe fn amx_path(a_u8: &[u8], b_vnni: &[i8], c: &mut [i32], k: usize) {
    // Tile config: 16×64-byte tiles, identical shape to the BF16 tile
    // (BF16 is 32 elements × 2 bytes per row, INT8 is 64 elements × 1
    // byte — same 64-byte row width either way).
    let cfg = TileConfig::for_dpbusd(64);
    tile_loadconfig(&cfg);
    tile_zero(0);

    // Accumulate over K/64 tile blocks. Each TDPBUSD consumes 64
    // K-elements per A row × 4 K-elements per inner-product = 256 MACs
    // per output cell × 16 × 16 = 16 384 MACs per instruction.
    let k_blocks = k / 64;
    let a_stride = k; // bytes per A row (u8 = 1 byte each)
    let b_stride = 64usize; // VNNI: 16 columns × 4 bytes per row

    for kb in 0..k_blocks {
        let a_ptr = a_u8.as_ptr().add(kb * 64);
        // B sits in VNNI layout: K/4 outer rows × 64 bytes. Each
        // 64-K-element block spans 16 outer rows × 64 bytes = 1024
        // bytes.
        let b_ptr = b_vnni.as_ptr().add(kb * 16 * 64) as *const u8;
        tile_load(1, a_ptr, a_stride);
        tile_load(2, b_ptr, b_stride);
        tile_dpbusd();
    }

    tile_store(0, c.as_mut_ptr() as *mut u8, 64);
    tile_release();
}

// ═════════════════════════════════════════════════════════════════════
// Scalar fallback (i32 reference)
// ═════════════════════════════════════════════════════════════════════

/// Direct scalar u8 × i8 → i32 reference. Accumulates into `c`.
fn fallback_path(a_u8: &[u8], b_i8: &[i8], c: &mut [i32], k: usize) {
    for i in 0..16 {
        for kk in 0..k {
            let a_val = a_u8[i * k + kk] as i32;
            for j in 0..16 {
                c[i * 16 + j] += a_val * b_i8[kk * 16 + j] as i32;
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: scalar u8 × i8 → i32 (matches `fallback_path`).
    fn ref_gemm(a: &[u8], b: &[i8], c: &mut [i32], k: usize) {
        for i in 0..16 {
            for j in 0..16 {
                let mut s = 0i32;
                for kk in 0..k {
                    s += a[i * k + kk] as i32 * b[kk * 16 + j] as i32;
                }
                c[i * 16 + j] = s;
            }
        }
    }

    #[test]
    fn fallback_matches_scalar_reference_k64() {
        let k = 64;
        // Deterministic pseudo-random inputs covering the u8 / i8 ranges.
        let a: Vec<u8> = (0..16 * k).map(|i| ((i * 7 + 3) % 256) as u8).collect();
        let b: Vec<i8> = (0..k * 16)
            .map(|i| ((i * 11 + 5) % 256) as u8 as i8)
            .collect();

        let mut c_ref = vec![0i32; 256];
        ref_gemm(&a, &b, &mut c_ref, k);

        let mut c_fb = vec![0i32; 256];
        fallback_path(&a, &b, &mut c_fb, k);

        for i in 0..256 {
            assert_eq!(c_fb[i], c_ref[i], "fallback mismatch at {}", i);
        }
    }

    #[test]
    fn public_api_runs_on_any_hardware_k64() {
        let k = 64;
        let a = vec![0u8; 16 * k];
        let b = vec![0i8; k * 16];
        let mut c = vec![0i32; 256];
        int8_tile_gemm_16x16(&a, &b, &mut c, k);
        for v in c.iter() {
            assert_eq!(*v, 0, "zero × zero must be 0");
        }
    }

    #[test]
    fn public_api_diagonal_k128() {
        // A = identity-like (only A[i, i] = 1, but we need 16 × 128), so
        // pick A[i, i*8..i*8+8] = 1 (8 ones per i-row). B = constant 2.
        // Expected: C[i, j] = sum_{kk in i*8..i*8+8}(1 × 2) = 16.
        let k = 128;
        let mut a = vec![0u8; 16 * k];
        for i in 0..16 {
            for off in 0..8 {
                a[i * k + i * 8 + off] = 1;
            }
        }
        let b = vec![2i8; k * 16];
        let mut c = vec![0i32; 256];
        int8_tile_gemm_16x16(&a, &b, &mut c, k);
        for i in 0..16 {
            for j in 0..16 {
                assert_eq!(c[i * 16 + j], 16, "diagonal accumulator at ({}, {})", i, j);
            }
        }
    }

    #[test]
    fn vnni_pack_i8_roundtrip() {
        // Pack then verify the VNNI layout matches the spec:
        // dst[kb*N*4 + j*4 + p] = src[(4*kb + p) * N + j]
        let k = 8usize;
        let n = 4usize;
        let src: Vec<i8> = (0..(k * n) as i8).collect();
        let mut dst = vec![0i8; k * n];
        vnni_pack_i8(&src, &mut dst, k, n);
        for kb in 0..(k / 4) {
            for j in 0..n {
                for p in 0..4 {
                    let dst_idx = kb * n * 4 + j * 4 + p;
                    let expected = src[(4 * kb + p) * n + j];
                    assert_eq!(dst[dst_idx], expected, "vnni quad mismatch at kb={} j={} p={}", kb, j, p);
                }
            }
        }
    }
}
