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

    // Operand placement (verified empirically on Emerald Rapids — see the
    // `tile_dpbusd` doc): the AMX operand convention is the mirror of the
    // naive SDM reading. The plain M×K operand goes in tmm2 (ModRM.rm) and is
    // treated UNSIGNED; the VNNI K×N operand goes in tmm1 (VEX.vvvv) and is
    // treated SIGNED. TDPBUSD then computes
    //   dst[m][n] = Σ_k a_u8(rm, unsigned)[m][k] · b_i8(vvvv, signed)[k][n]
    // — exactly the u8 × i8 this kernel promises.
    for kb in 0..k_blocks {
        let a_ptr = a_u8.as_ptr().add(kb * 64);
        // B sits in VNNI layout: K/4 outer rows × 64 bytes. Each
        // 64-K-element block spans 16 outer rows × 64 bytes = 1024
        // bytes.
        let b_ptr = b_vnni.as_ptr().add(kb * 16 * 64) as *const u8;
        tile_load(1, b_ptr, b_stride); // B (VNNI) → tmm1 (vvvv, signed)
        tile_load(2, a_ptr, a_stride); // A (plain) → tmm2 (rm, unsigned)
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
// VPDPBUSD-ymm AVX-VNNI tier (Arrow Lake / Meteor Lake U / Alder Lake)
// ═════════════════════════════════════════════════════════════════════

/// AVX-VNNI ymm `u8 × i8 → i32` GEMM kernel for arbitrary M × N × K.
///
/// One `_mm256_dpbusd_avx_epi32` instruction: 8 i32 accumulator lanes,
/// each receiving the sum of 4 `u8 × i8` products = **32 MACs per
/// instruction**. Half the throughput-per-instruction of the
/// `_mm512_dpbusd_epi32` zmm version (which does 64 MACs); fires on
/// Arrow Lake / Meteor Lake U / Alder Lake silicon that has AVX-VNNI
/// but NOT AVX-512.
///
/// Same B pre-packing scheme as the zmm version (quad-interleaved per
/// 8-wide j-block), same K-tail and N-tail handling, just narrower.
/// Mirrors the `vnni2_dot_u8_i8` shape in `simd_amx.rs` but as a
/// matrix-product instead of single-row dot.
///
/// Output behavior: overwrites `c` (does NOT accumulate). Caller's
/// responsibility to zero `c` first if needed.
///
/// # Safety
/// Caller must have feature-detected `avxvnni + avx2` at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnni,avx2")]
pub unsafe fn int8_gemm_vpdpbusd_ymm(a_u8: &[u8], b_i8: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    use core::arch::x86_64::{
        __m256i, _mm256_dpbusd_avx_epi32, _mm256_loadu_si256, _mm256_set1_epi32, _mm256_setzero_si256,
        _mm256_storeu_si256,
    };

    let k_quads = k / 4;
    let k_tail = k % 4;

    // Pre-pack scratch: 8 i32 lanes per k_quad (vs 16 in the zmm
    // version). Same per-lane layout: each i32 holds 4 consecutive
    // B K-bytes for output column j+lane.
    let mut b_col_quads = vec![0i32; k_quads.max(1) * 8];
    let mut out_buf = [0i32; 8];

    for j_base in (0..n).step_by(8) {
        let j_count = 8.min(n - j_base);

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
                b_col_quads[k_quad * 8 + jj] = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)) as i32;
            }
            for jj in j_count..8 {
                b_col_quads[k_quad * 8 + jj] = 0;
            }
        }

        for i in 0..m {
            let mut acc = _mm256_setzero_si256();
            let a_row_off = i * k;
            for k_quad in 0..k_quads {
                let a0 = a_u8[a_row_off + 4 * k_quad] as u32;
                let a1 = a_u8[a_row_off + 4 * k_quad + 1] as u32;
                let a2 = a_u8[a_row_off + 4 * k_quad + 2] as u32;
                let a3 = a_u8[a_row_off + 4 * k_quad + 3] as u32;
                let packed_a = a0 | (a1 << 8) | (a2 << 16) | (a3 << 24);
                let a_v = _mm256_set1_epi32(packed_a as i32);
                let b_v = _mm256_loadu_si256(b_col_quads.as_ptr().add(k_quad * 8) as *const __m256i);
                acc = _mm256_dpbusd_avx_epi32(acc, a_v, b_v);
            }
            _mm256_storeu_si256(out_buf.as_mut_ptr() as *mut __m256i, acc);

            if k_tail > 0 {
                for kk in (k_quads * 4)..k {
                    let a_val = a_u8[a_row_off + kk] as i32;
                    let tail_row = kk * n;
                    for jj in 0..j_count {
                        out_buf[jj] += a_val * b_i8[tail_row + j_base + jj] as i32;
                    }
                }
            }

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
// AMX tiled helper — arbitrary 16/16/64-aligned M × N × K via 16×16 tile loop
// ═════════════════════════════════════════════════════════════════════

/// `u8 × i8 → i32` GEMM using AMX `TDPBUSD` for arbitrary M × N × K
/// shapes that satisfy `m % 16 == 0 && n % 16 == 0 && k % 64 == 0`.
///
/// Tile-decomposes the M × N output into 16×16 blocks and calls
/// [`int8_tile_gemm_16x16`] per (i_tile, j_tile). B sub-block extracted
/// into K × 16 scratch once per j-tile, reused across all M i-tiles —
/// amortizes the column gather cost.
///
/// **Overwrite semantics**: `c` is written, not accumulated. Caller
/// does NOT need to zero `c` beforehand. (The underlying
/// `int8_tile_gemm_16x16` accumulates into its tile buffer, but we
/// zero the tile buffer before each call so the per-tile write to `c`
/// is pure overwrite.)
///
/// # Panics
/// Panics if `a_u8`, `b_i8`, or `c` are too small for the requested
/// `(m, n, k)`, mirroring the boundary contract from `gemm_u8_i8`. Also
/// panics in debug builds when AMX isn't OS-enabled or when the shape
/// alignment constraints aren't met (production builds skip those for
/// performance — callers must runtime-check
/// `crate::hpc::amx_matmul::amx_available()` and the 16/16/64
/// alignment themselves).
pub fn int8_gemm_amx_tiled(a_u8: &[u8], b_i8: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    // Length assertions (codex P1 from PR #185 — the function reads
    // `b_i8` via a 16-wide window per (kk, j_tile) iteration and a_u8
    // via a 16-row slice per i_tile, so mismatched shapes would
    // trigger out-of-bounds reads without these gates).
    assert!(a_u8.len() >= m * k, "int8_gemm_amx_tiled: a_u8.len()={} < m*k={}", a_u8.len(), m * k);
    assert!(b_i8.len() >= k * n, "int8_gemm_amx_tiled: b_i8.len()={} < k*n={}", b_i8.len(), k * n);
    assert!(c.len() >= m * n, "int8_gemm_amx_tiled: c.len()={} < m*n={}", c.len(), m * n);

    debug_assert!(crate::hpc::amx_matmul::amx_available());
    debug_assert_eq!(m % 16, 0, "int8_gemm_amx_tiled: M must be multiple of 16");
    debug_assert_eq!(n % 16, 0, "int8_gemm_amx_tiled: N must be multiple of 16");
    debug_assert_eq!(k % 64, 0, "int8_gemm_amx_tiled: K must be multiple of 64");

    // With `rayon`, fan the M/16 row-tiles across the pool — but only for LARGE
    // GEMMs. This AMX kernel is memory-bandwidth-bound, so on a few-core host the
    // cores contend for bandwidth and row-tile parallelism scales sublinearly
    // (~1.4× at 2048³ on 4 cores); below ~2 GMAC the thread-dispatch + shared
    // B-prepack overhead actually REGRESSES it (measured: 512³ 125→73 GMAC/s).
    // The threshold keeps the fast serial path for small/medium shapes and only
    // parallelizes where it nets a win (and where many-core servers gain most).
    // The per-tile kernel is byte-for-byte the validated serial one, so
    // correctness is unchanged. (AMX permission is process-wide; the tile CONFIG
    // is per-thread CPU state, so each worker runs its own LDTILECFG — see
    // `int8_gemm_amx_tiled_par`.)
    #[cfg(feature = "rayon")]
    {
        let work = (m as u64).saturating_mul(n as u64).saturating_mul(k as u64);
        if m >= 32 && work >= 2_000_000_000 {
            int8_gemm_amx_tiled_par(a_u8, b_i8, c, m, n, k);
            return;
        }
    }
    int8_gemm_amx_tiled_serial(a_u8, b_i8, c, m, n, k);
}

/// Single-thread AMX int8 tiled GEMM — the validated core kernel. LDTILECFG and
/// the per-band VNNI pack are hoisted out of the M/16 × N/16 tile loops (1
/// LDTILECFG total, not one per 16×16 tile); the 16×16 result tile is
/// TILESTOREd straight into its strided slot in `c` (row pitch n·4 bytes).
fn int8_gemm_amx_tiled_serial(a_u8: &[u8], b_i8: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    let mut b_tile = vec![0i8; k * 16]; // one column band of B (row-major K×16)
    let mut b_vnni = vec![0i8; k * 16]; // its VNNI-quad packing, reused across i-tiles
    let k_blocks = k / 64;

    // SAFETY: caller asserted `amx_available()` + 16/16/64 alignment + slice
    // bounds. The tile config is loaded once and released once; every
    // tile_load/tile_store stays inside the a_u8 / b_vnni / c bounds (16×64-byte
    // tiles, K in 64-wide blocks, strided store row pitch n·4 bytes).
    unsafe {
        let cfg = TileConfig::for_dpbusd(64);
        tile_loadconfig(&cfg);

        for j_tile in (0..n).step_by(16) {
            // Pack B[:, j_tile..+16] (row-major K×16) then VNNI-quad it ONCE
            // per column band; reused across all M/16 row tiles below.
            for kk in 0..k {
                let row = kk * n + j_tile;
                b_tile[kk * 16..(kk + 1) * 16].copy_from_slice(&b_i8[row..row + 16]);
            }
            vnni_pack_i8(&b_tile, &mut b_vnni, k, 16);

            for i_tile in (0..m).step_by(16) {
                let a_tile = &a_u8[i_tile * k..(i_tile + 16) * k];
                tile_zero(0); // C tile = 0 (this driver overwrites, not accumulates)
                for kb in 0..k_blocks {
                    // B(VNNI K×N) → tmm1 (vvvv, signed); A(plain M×K) → tmm2 (rm, unsigned).
                    tile_load(1, b_vnni.as_ptr().add(kb * 16 * 64) as *const u8, 64);
                    tile_load(2, a_tile.as_ptr().add(kb * 64), k);
                    tile_dpbusd();
                }
                // Store tmm0 (16×16 i32) straight into the strided C location —
                // row pitch n·4 bytes — with no scratch buffer or copy loop.
                let c_ptr = c.as_mut_ptr().add(i_tile * n + j_tile) as *mut u8;
                tile_store(0, c_ptr, n * 4);
            }
        }

        tile_release();
    }
}

/// Rayon-parallel AMX int8 tiled GEMM. B is VNNI-packed ONCE into a shared,
/// read-only buffer (all N/16 column bands), then the M/16 row-tiles are fanned
/// across the rayon pool — one task per 16-row block of `c`. Each worker runs
/// the same validated tile sequence as the serial path.
#[cfg(feature = "rayon")]
fn int8_gemm_amx_tiled_par(a_u8: &[u8], b_i8: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    use rayon::prelude::*;

    let n_jtiles = n / 16;
    let k_blocks = k / 64;
    let band = k * 16; // bytes per VNNI-packed column band

    // Pre-pack every B column band into one shared VNNI buffer (read-only in the
    // parallel region). O(K·N) — cheap vs the O(M·N·K) GEMM.
    let mut b_vnni_all = vec![0i8; n_jtiles * band];
    {
        let mut b_tile = vec![0i8; band];
        for jt in 0..n_jtiles {
            let j_tile = jt * 16;
            for kk in 0..k {
                let row = kk * n + j_tile;
                b_tile[kk * 16..(kk + 1) * 16].copy_from_slice(&b_i8[row..row + 16]);
            }
            vnni_pack_i8(&b_tile, &mut b_vnni_all[jt * band..(jt + 1) * band], k, 16);
        }
    }

    // One task per 16-row block of C. `c[..m*n]` guarantees exactly m/16 chunks.
    c[..m * n]
        .par_chunks_mut(16 * n)
        .enumerate()
        .for_each(|(it, c_rows)| {
            let i_tile = it * 16;
            let a_tile = &a_u8[i_tile * k..(i_tile + 16) * k];
            // SAFETY: AMX permission is process-wide (arch_prctl granted once via the
            // `amx_available()` LazyLock the caller already triggered) and inherited
            // by every thread; the tile CONFIG is per-thread CPU state, so this
            // worker loads its own config and releases it. `b_vnni_all` is read-only
            // and shared; `c_rows` is this task's exclusive 16-row slice. All
            // loads/stores stay within bounds (a_tile is 16×k; the strided store row
            // pitch is n·4 bytes within this 16-row chunk).
            unsafe {
                let cfg = TileConfig::for_dpbusd(64);
                tile_loadconfig(&cfg);
                for jt in 0..n_jtiles {
                    let j_tile = jt * 16;
                    let b_vnni = &b_vnni_all[jt * band..(jt + 1) * band];
                    tile_zero(0);
                    for kb in 0..k_blocks {
                        tile_load(1, b_vnni.as_ptr().add(kb * 16 * 64) as *const u8, 64);
                        tile_load(2, a_tile.as_ptr().add(kb * 64), k);
                        tile_dpbusd();
                    }
                    let c_ptr = c_rows.as_mut_ptr().add(j_tile) as *mut u8;
                    tile_store(0, c_ptr, n * 4);
                }
                tile_release();
            }
        });
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

    /// Codex P1 regression on PR #185: `int8_gemm_amx_tiled` is a
    /// safe public function — mismatched (m, n, k) vs slice lengths
    /// must panic at the function boundary, not trigger UB inside
    /// the unsafe slice/pointer arithmetic in the inner loop. This
    /// test passes deliberately-undersized buffers and expects a
    /// panic (which `#[should_panic]` catches).
    #[test]
    #[should_panic(expected = "b_i8.len()")]
    fn amx_tiled_panics_on_undersized_b() {
        let m = 16;
        let n = 32;
        let k = 64;
        let a = vec![0u8; m * k];
        let b = vec![0i8; k * (n - 16)]; // half a j_tile short of what's claimed
        let mut c = vec![0i32; m * n];
        // Even on non-AMX hosts the assertion fires before reaching
        // the (debug-asserted) amx_available() check.
        int8_gemm_amx_tiled(&a, &b, &mut c, m, n, k);
    }

    /// Direct test for the VPDPBUSD-ymm arm (AVX-VNNI tier of
    /// `matmul_i8_to_i32`). Same shape / bit-exactness contract as
    /// the zmm version's test, just on the narrower 8-wide kernel.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vpdpbusd_ymm_matches_scalar() {
        if !std::is_x86_feature_detected!("avxvnni") {
            eprintln!("avxvnni not detected; skipping");
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

        // Sweep shapes spanning 8-aligned, K-tail (k % 4), N-tail
        // (n % 8), and small shapes to exercise every code path.
        for (m, n, k) in [(16, 8, 64), (3, 5, 7), (17, 33, 100), (1, 17, 12), (8, 8, 4)] {
            let a: Vec<u8> = (0..m * k).map(|i| ((i * 31 + 7) % 256) as u8).collect();
            let b: Vec<i8> = (0..k * n)
                .map(|i| ((i * 17 + 3) % 256) as u8 as i8)
                .collect();
            let expected = ref_gemm(&a, &b, m, n, k);
            let mut got = vec![0i32; m * n];
            // SAFETY: avxvnni confirmed at the top of the test.
            unsafe { int8_gemm_vpdpbusd_ymm(&a, &b, &mut got, m, n, k) };
            assert_eq!(got, expected, "VPDPBUSD-ymm mismatch at (M={}, N={}, K={})", m, n, k);
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
