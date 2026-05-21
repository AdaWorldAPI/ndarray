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
/// **Accumulates** into the caller's `c` buffer — matches the
/// documented `C += A·B` semantics. The C tile (tmm0) is preloaded
/// from `c` before the TDPBUSD loop so any pre-existing values are
/// preserved.
///
/// # Safety
/// Caller must have verified `amx_available() == true`.
#[inline]
unsafe fn amx_path(a_u8: &[u8], b_vnni: &[i8], c: &mut [i32], k: usize) {
    // Tile config: 16×64-byte tiles, identical shape to the BF16 tile
    // (BF16 is 32 elements × 2 bytes per row, INT8 is 64 elements × 1
    // byte — same 64-byte row width either way).
    let cfg = TileConfig::for_dpbusd(64);
    tile_loadconfig(&cfg);
    // Preload C accumulator from caller's buffer so TDPBUSD truly
    // accumulates into the existing values (fixes codex P1 from PR
    // #184 — the prior `tile_zero(0)` discarded pre-existing C values
    // even though the docs promise `C += A·B`).
    tile_load(0, c.as_ptr() as *const u8, 64);

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
// VPDPBUSD-zmm middle tier (avx512vnni without AMX)
// ═════════════════════════════════════════════════════════════════════

/// AVX-512 VNNI `u8 × i8 → i32` GEMM kernel for arbitrary M × N × K.
///
/// One `_mm512_dpbusd_epi32` instruction: 16 i32 accumulator lanes,
/// each receiving the sum of 4 `u8 × i8` products = **64 MACs per
/// instruction**. Pre-packs B in VNNI quad layout once per j-block
/// (16-wide column band) and reuses across all M i-iterations,
/// amortizing the gather cost.
///
/// K-tail (when K is not a multiple of 4) handled with scalar
/// u8 × i8 multiplies per output cell; N-tail (when the j-block has
/// fewer than 16 valid columns) handled by trimming the store after
/// the VPDPBUSD chain.
///
/// This is the middle dispatch tier between AMX TDPBUSD (Sapphire
/// Rapids+) and the scalar reference — covers Cooper Lake, Cascade
/// Lake, Ice Lake-SP, Zen 4+ silicon that has avx512vnni but not
/// AMX. Mirrors the VDPBF16PS arm structure shipped for BF16 in
/// PR #182.
///
/// Output behavior: overwrites `c` (does NOT accumulate). Caller's
/// responsibility to zero `c` first if a fresh-write GEMM is wanted.
///
/// # Safety
/// Caller must have feature-detected `avx512vnni + avx512f` at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512f")]
pub unsafe fn int8_gemm_vpdpbusd_zmm(a_u8: &[u8], b_i8: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    use core::arch::x86_64::{
        __m512i, _mm512_dpbusd_epi32, _mm512_loadu_si512, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_storeu_si512,
    };

    let k_quads = k / 4;
    let k_tail = k % 4;

    // Pre-pack scratch for B columns of the current j-block:
    // 16 i32 lanes per k_quad, each holding 4 consecutive K-bytes
    // packed (b[2q+0..2q+4] for output column j+lane).
    let mut b_col_quads = vec![0i32; k_quads.max(1) * 16];
    // Scratch for the 16-wide store + N-tail trim.
    let mut out_buf = [0i32; 16];

    for j_base in (0..n).step_by(16) {
        let j_count = 16.min(n - j_base);

        // Pack B[0..k, j_base..j_base+j_count] in quad-interleaved layout.
        // For lanes j >= j_count (the N-tail of this j_block), pad with 0
        // so the VPDPBUSD doesn't read uninitialized memory; they're not
        // stored back.
        for k_quad in 0..k_quads {
            let row0 = 4 * k_quad * n;
            let row1 = (4 * k_quad + 1) * n;
            let row2 = (4 * k_quad + 2) * n;
            let row3 = (4 * k_quad + 3) * n;
            for jj in 0..j_count {
                let b0 = b_i8[row0 + j_base + jj] as u8 as u32;
                let b1 = b_i8[row1 + j_base + jj] as u8 as u32;
                let b2 = b_i8[row2 + j_base + jj] as u8 as u32;
                let b3 = b_i8[row3 + j_base + jj] as u8 as u32;
                // Pack as i32: bottom byte is k_quad*4+0, top is k_quad*4+3.
                b_col_quads[k_quad * 16 + jj] = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) as i32;
            }
            for jj in j_count..16 {
                b_col_quads[k_quad * 16 + jj] = 0;
            }
        }

        for i in 0..m {
            let mut acc = _mm512_setzero_si512();
            let a_row_off = i * k;
            for k_quad in 0..k_quads {
                // Broadcast A[i, 4*k_quad..4*k_quad+4] (4 u8) across all
                // 16 i32 lanes via _mm512_set1_epi32.
                let a0 = a_u8[a_row_off + 4 * k_quad] as u32;
                let a1 = a_u8[a_row_off + 4 * k_quad + 1] as u32;
                let a2 = a_u8[a_row_off + 4 * k_quad + 2] as u32;
                let a3 = a_u8[a_row_off + 4 * k_quad + 3] as u32;
                let packed_a = a0 | (a1 << 8) | (a2 << 16) | (a3 << 24);
                let a_v = _mm512_set1_epi32(packed_a as i32);
                let b_v = _mm512_loadu_si512(b_col_quads.as_ptr().add(k_quad * 16) as *const __m512i);
                acc = _mm512_dpbusd_epi32(acc, a_v, b_v);
            }
            _mm512_storeu_si512(out_buf.as_mut_ptr() as *mut __m512i, acc);

            // K-tail: scalar multiplies for k = k_quads*4 .. k.
            if k_tail > 0 {
                for kk in (k_quads * 4)..k {
                    let a_val = a_u8[a_row_off + kk] as i32;
                    let tail_row = kk * n;
                    for jj in 0..j_count {
                        out_buf[jj] += a_val * b_i8[tail_row + j_base + jj] as i32;
                    }
                }
            }

            // Store j_count valid lanes (drops N-tail padding lanes).
            let dst_off = i * n + j_base;
            c[dst_off..dst_off + j_count].copy_from_slice(&out_buf[..j_count]);
        }
    }
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

    /// Codex P1 regression on PR #184: `int8_tile_gemm_16x16` is
    /// documented as `C += A·B`, but the AMX path used to `tile_zero(0)`
    /// then `tile_store(0, c)`, **overwriting** `c` on AMX hosts (the
    /// scalar fallback correctly accumulated). This test pre-loads C
    /// with a known marker, runs A·B=0 (B is all zeros so the product
    /// is zero), and asserts the marker is preserved — would fail on
    /// the pre-fix AMX path because the tile_store would zero everything.
    #[test]
    fn amx_path_preserves_c_accumulator() {
        let k = 64;
        let a = vec![1u8; 16 * k];
        let b = vec![0i8; k * 16]; // product is exactly 0
                                   // Pre-load C with a non-zero marker pattern.
        let mut c: Vec<i32> = (0..256).map(|i| i as i32 * 7 - 100).collect();
        let snapshot = c.clone();
        int8_tile_gemm_16x16(&a, &b, &mut c, k);
        // After: c[i] += 0 → c[i] unchanged from snapshot.
        for i in 0..256 {
            assert_eq!(c[i], snapshot[i], "accumulator marker clobbered at {}: {} → {}", i, snapshot[i], c[i]);
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

    /// Direct test for the VPDPBUSD-zmm arm, exercising the path the
    /// `matmul_i8_to_i32` dispatcher would skip when AMX is available.
    /// Verifies bit-exact parity against the scalar reference for
    /// arbitrary (M, N, K) — including non-multiple-of-4 K (so the
    /// scalar K-tail branch fires) and non-multiple-of-16 N (so the
    /// j-count trim branch fires).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vpdpbusd_zmm_matches_scalar() {
        if !std::is_x86_feature_detected!("avx512vnni") {
            eprintln!("avx512vnni not detected; skipping");
            return;
        }

        fn ref_gemm(a: &[u8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
            let mut c = vec![0i32; m * n];
            for i in 0..m {
                for kk in 0..k {
                    let av = a[i * k + kk] as i32;
                    for j in 0..n {
                        c[i * n + j] += av * b[kk * n + j] as i32;
                    }
                }
            }
            c
        }

        // Sweep shapes spanning aligned cases, K-tail (k % 4), and
        // N-tail (n % 16) to exercise every code path.
        for (m, n, k) in [(16, 16, 64), (3, 5, 7), (17, 33, 100), (1, 17, 12), (8, 16, 4)] {
            let a: Vec<u8> = (0..m * k).map(|i| ((i * 31 + 7) % 256) as u8).collect();
            let b: Vec<i8> = (0..k * n)
                .map(|i| ((i * 17 + 3) % 256) as u8 as i8)
                .collect();
            let expected = ref_gemm(&a, &b, m, n, k);
            let mut got = vec![0i32; m * n];
            // SAFETY: avx512vnni confirmed at the top of the test.
            unsafe { int8_gemm_vpdpbusd_zmm(&a, &b, &mut got, m, n, k) };
            assert_eq!(got, expected, "VPDPBUSD-zmm mismatch at (M={}, N={}, K={})", m, n, k);
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
