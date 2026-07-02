//! BF16 tile GEMM — three-tier runtime dispatch, no decode on the tile tiers.
//!
//! Tier ladder (selected per host, report via [`bf16_tile_gemm_tier`]):
//! 1. **AMX `TDPBF16PS`** — bf16×bf16 tile pairs, f32 tile accumulator
//!    (raw primitives in `hpc::amx_matmul`).
//! 2. **AVX-512 `VDPBF16PS`** — bf16×bf16 register pairs, f32 zmm
//!    accumulator. Same VNNI operand layout as tier 1, no bf16→f32 decode.
//! 3. **Decode + `F32x16` FMA polyfill** (`crate::simd::bf16_tile_gemm_16x16`)
//!    — bf16→f32 decode then FMA; the polyfill owns AVX-512/AVX2/NEON/scalar.
//!
//! All tiers accumulate (`C += A·B`) and are **bit-exact for bf16-exact
//! integer operands** with accumulation below 2^24 (bf16 products are exact
//! in f32) — asserted by the parity tests below with `assert_eq!`, not
//! tolerance. For general float operands the tiers agree up to accumulation
//! order (BF16-precision class).
//!
//! Tile shape: M=16, N=16, K = multiple of 32. Hot loops should pre-pack B
//! ([`PackedBf16B`]) or stage it directly in VNNI layout
//! ([`PackedBf16B::vnni_index`]) and call [`bf16_tile_gemm_16x16_packed`] —
//! the row-major entry [`bf16_tile_gemm_16x16`] VNNI-packs (and allocates)
//! per call on the tile tiers.
//!
//! ⚠ Gotcha 14 (`.claude/AMX_GOTCHAS.md`): on oversubscribed VMs, AMX tile
//! state silently corrupts under host CPU contention. Certify tier-1
//! numerics on dedicated silicon; see `tile_parity_under_cpu_contention`
//! (`--ignored`).

use crate::hpc::amx_matmul::{
    amx_available, tile_dpbf16ps, tile_load, tile_loadconfig, tile_release, tile_store, tile_zero, vnni_pack_bf16,
    TileConfig,
};

// ═════════════════════════════════════════════════════════════════════
// Public API — safe dispatching wrapper
// ═════════════════════════════════════════════════════════════════════

/// Compute C[16, 16] += A[16, K] × B[K, 16] where A, B are BF16 row-major
/// and C is f32 row-major. K must be a multiple of 32.
///
/// Tier dispatch (runtime): AMX `TDPBF16PS` → AVX-512 `VDPBF16PS` → decode +
/// `F32x16` FMA polyfill (see module doc). Bit-exact across tiers for
/// bf16-exact integer operands; identical up to BF16-precision accumulation
/// order otherwise. On the tile tiers this entry VNNI-packs B (and
/// allocates) per call — hot loops should use [`bf16_tile_gemm_16x16_packed`].
pub fn bf16_tile_gemm_16x16(a_bf16: &[u16], b_bf16: &[u16], c: &mut [f32], k: usize) {
    assert_eq!(k % 32, 0, "K must be multiple of 32");
    assert_eq!(a_bf16.len(), 16 * k);
    assert_eq!(b_bf16.len(), k * 16);
    assert_eq!(c.len(), 16 * 16);

    if amx_available() || is_x86_feature_detected!("avx512bf16") {
        // Tile tiers want VNNI-packed B; pack per call (see `PackedBf16B` +
        // `bf16_tile_gemm_16x16_packed` to hoist this out of hot loops).
        let mut b_vnni = vec![0u16; k * 16];
        vnni_pack_bf16(b_bf16, &mut b_vnni, k, 16);
        dispatch_vnni(a_bf16, &b_vnni, c, k);
    } else {
        // Pure-FMA tier consumes row-major B directly — no pack needed.
        fallback_path(a_bf16, b_bf16, c, k);
    }
}

/// Which kernel `bf16_tile_gemm_16x16` / `_packed` will take on this host,
/// as a human-readable tier name. For run reports per AMX Gotcha 9 ("a
/// skipped test is not a passing test" — always PRINT which tier ran).
pub fn bf16_tile_gemm_tier() -> &'static str {
    if amx_available() {
        "AMX TDPBF16PS"
    } else if is_x86_feature_detected!("avx512bf16") {
        "AVX-512 VDPBF16PS"
    } else {
        "F32x16 FMA polyfill (bf16→f32 decode)"
    }
}

// ═════════════════════════════════════════════════════════════════════
// Pre-packed B — hoists VNNI packing (and its allocation) out of hot loops
// ═════════════════════════════════════════════════════════════════════

/// B\[K, 16\] operand held in the VNNI pair layout shared by AMX `TDPBF16PS`
/// and AVX-512 `VDPBF16PS`: pair-row `i` holds `(B[2i, j], B[2i+1, j])`
/// interleaved over `j` — 32 bf16 = 64 bytes = one tile row / one zmm.
///
/// Two ways to fill it:
/// - [`PackedBf16B::pack`] / [`PackedBf16B::pack_from`] — from row-major B
///   (`pack_from` reuses the buffer: no per-call allocation).
/// - Stage **directly** in VNNI layout via [`PackedBf16B::vnni_index`] +
///   [`PackedBf16B::data_mut`] — zero packing cost. This is the right shape
///   for sparse staging (e.g. one-hot group-by indicators): write the few
///   live entries, run the GEMM, clear the same entries.
///
/// # Examples
/// ```
/// use ndarray::simd::{bf16_tile_gemm_16x16_packed, PackedBf16B};
/// let k = 32;
/// let mut b = PackedBf16B::zeroed(k);
/// // B[5][3] = 1.0 staged directly in VNNI layout (0x3F80 = bf16 1.0):
/// let idx = PackedBf16B::vnni_index(5, 3);
/// b.data_mut()[idx] = 0x3F80;
/// let a = vec![0x3F80u16; 16 * k]; // A = all 1.0
/// let mut c = vec![0.0f32; 256];
/// bf16_tile_gemm_16x16_packed(&a, &b, &mut c);
/// assert_eq!(c[3], 1.0); // C[0][3] = Σ_k A[0][k]·B[k][3] = A[0][5]·1.0
/// ```
pub struct PackedBf16B {
    data: Vec<u16>,
    k: usize,
}

impl PackedBf16B {
    /// All-zero packed B for direct VNNI staging. `k` must be a multiple of 32.
    pub fn zeroed(k: usize) -> Self {
        assert_eq!(k % 32, 0, "K must be multiple of 32");
        PackedBf16B {
            data: vec![0u16; k * 16],
            k,
        }
    }

    /// Pack row-major B\[K, 16\] (allocates once; prefer [`Self::pack_from`]
    /// for repeated packs).
    pub fn pack(b_bf16_row_major: &[u16], k: usize) -> Self {
        let mut p = Self::zeroed(k);
        p.pack_from(b_bf16_row_major);
        p
    }

    /// Re-pack row-major B\[K, 16\] into the existing buffer — no allocation.
    pub fn pack_from(&mut self, b_bf16_row_major: &[u16]) {
        assert_eq!(b_bf16_row_major.len(), self.k * 16);
        vnni_pack_bf16(b_bf16_row_major, &mut self.data, self.k, 16);
    }

    /// Flat index of logical `B[row, col]` inside the VNNI buffer.
    /// `vnni[(row/2)·32 + 2·col + (row & 1)] == B[row, col]`.
    #[inline(always)]
    pub const fn vnni_index(row: usize, col: usize) -> usize {
        (row / 2) * 32 + 2 * col + (row & 1)
    }

    /// The K dimension this buffer was sized for.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Read access to the raw VNNI buffer.
    pub fn data(&self) -> &[u16] {
        &self.data
    }

    /// Mutable access to the raw VNNI buffer, for direct staging via
    /// [`Self::vnni_index`].
    pub fn data_mut(&mut self) -> &mut [u16] {
        &mut self.data
    }
}

/// `C[16, 16] += A[16, K] × B[K, 16]` with B pre-packed — the hot-loop
/// sibling of [`bf16_tile_gemm_16x16`] with the per-call VNNI pack (and its
/// allocation) hoisted into [`PackedBf16B`].
///
/// Tier dispatch (runtime): AMX `TDPBF16PS` → AVX-512 `VDPBF16PS` → decode +
/// `F32x16` FMA polyfill. All tiers accumulate into `c` and are exact for
/// bf16-exact-integer operands (products of bf16 values are exact in f32;
/// integer accumulation is exact below 2^24). Use
/// [`bf16_tile_gemm_tier`] to report which tier ran.
pub fn bf16_tile_gemm_16x16_packed(a_bf16: &[u16], b: &PackedBf16B, c: &mut [f32]) {
    let k = b.k;
    assert_eq!(a_bf16.len(), 16 * k);
    assert_eq!(c.len(), 16 * 16);
    dispatch_vnni(a_bf16, &b.data, c, k);
}

/// Shared tier dispatch over a VNNI-packed B.
fn dispatch_vnni(a_bf16: &[u16], b_vnni: &[u16], c: &mut [f32], k: usize) {
    if amx_available() {
        // SAFETY: amx_available() confirmed CPUID + XCR0 + arch_prctl grant.
        unsafe {
            amx_path(a_bf16, b_vnni, c, k);
        }
    } else if is_x86_feature_detected!("avx512bf16") {
        // SAFETY: runtime detection confirmed avx512f+avx512bf16.
        unsafe {
            avx512bf16_path(a_bf16, b_vnni, c, k);
        }
    } else {
        // Slow tier from a packed buffer: unpack back to row-major and take
        // the decode+FMA polyfill. Allocates — acceptable off the tile tiers
        // (a host without AMX and without avx512bf16 calling the packed entry
        // is not a hot-path configuration).
        let mut b_rm = vec![0u16; k * 16];
        for row in 0..k {
            for col in 0..16 {
                b_rm[row * 16 + col] = b_vnni[PackedBf16B::vnni_index(row, col)];
            }
        }
        fallback_path(a_bf16, &b_rm, c, k);
    }
}

// ═════════════════════════════════════════════════════════════════════
// AVX-512 BF16 path (VDPBF16PS) — the no-decode middle tier
// ═════════════════════════════════════════════════════════════════════

/// AVX-512 BF16 GEMM over VNNI-packed B — no bf16→f32 decode roundtrip:
/// `VDPBF16PS` multiplies bf16 pairs natively, accumulating into f32 lanes
/// (same accumulator semantics as the AMX tile, register-sized instead of
/// tile-sized). Per output row: one zmm accumulator preloaded from `c`
/// (accumulate semantics), K/2 pair steps, one store.
///
/// Numerics: bf16×bf16 products are exact in f32; accumulation order is
/// row-major over pair index (matches the AMX per-row order). VDPBF16PS
/// flushes input denormals to zero — irrelevant for the integer workloads
/// this crate certifies, documented for float callers.
///
/// # Safety
/// Caller must have verified `is_x86_feature_detected!("avx512bf16")`.
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn avx512bf16_path(a_bf16: &[u16], b_vnni: &[u16], c: &mut [f32], k: usize) {
    use core::arch::x86_64::{
        __m512bh, __m512i, _mm512_dpbf16_ps, _mm512_loadu_ps, _mm512_loadu_si512, _mm512_set1_epi32, _mm512_storeu_ps,
    };
    debug_assert_eq!(k % 2, 0);
    let pairs = k / 2;
    for i in 0..16 {
        // SAFETY (whole block): a_bf16 is 16×k and c is 16×16 (asserted by
        // the public callers), b_vnni is k×16 pairs; every pointer below
        // stays inside those bounds. read_unaligned handles the u16→u32
        // pair load without alignment requirements.
        unsafe {
            let mut acc = _mm512_loadu_ps(c.as_ptr().add(i * 16));
            let a_row = a_bf16.as_ptr().add(i * k);
            for p in 0..pairs {
                // Broadcast the A pair (a[2p], a[2p+1]) to all 16 lanes; lane j
                // then accumulates A[i][2p]·B[2p][j] + A[i][2p+1]·B[2p+1][j].
                let pair = (a_row.add(2 * p) as *const u32).read_unaligned();
                let av: __m512bh = core::mem::transmute(_mm512_set1_epi32(pair as i32));
                let bv: __m512bh =
                    core::mem::transmute(_mm512_loadu_si512(b_vnni.as_ptr().add(p * 32) as *const __m512i));
                acc = _mm512_dpbf16_ps(acc, av, bv);
            }
            _mm512_storeu_ps(c.as_mut_ptr().add(i * 16), acc);
        }
    }
}

// ═════════════════════════════════════════════════════════════════════
// AMX path (TDPBF16PS)
// ═════════════════════════════════════════════════════════════════════

/// AMX tile GEMM. B must be pre-VNNI-packed (see `vnni_pack_bf16`).
/// **Accumulates** into the caller's `c` buffer — matches the
/// documented `C += A·B` semantics. The C tile (tmm0) is preloaded
/// from `c` before the TDPBF16PS loop so any pre-existing values are
/// preserved. (Same accumulator-preservation fix the int8 sibling
/// got after codex P1 on PR #184: prior `tile_zero(0)` discarded
/// pre-existing C values even though docs promised accumulation.)
///
/// # Safety
/// Caller must have verified `amx_available() == true`.
#[inline]
unsafe fn amx_path(a_bf16: &[u16], b_vnni: &[u16], c: &mut [f32], k: usize) {
    // Tile config: shapes at K_bytes=64 match BF16 K=32 case
    let cfg = TileConfig::for_dpbusd(64);
    tile_loadconfig(&cfg);
    // Preload C accumulator from caller's buffer (was tile_zero(0)
    // pre-fix — see method-level note above).
    tile_load(0, c.as_ptr() as *const u8, 64);

    // Accumulate over K/32 tile blocks
    let k_blocks = k / 32;
    let a_stride = (k * 2) as usize; // full A row stride in bytes (bf16 = 2B)
    let b_stride = 64usize; // VNNI row stride in bytes

    // Operand placement (verified empirically — see `tile_dpbusd` doc): the
    // AMX convention is dst[m][n] = Σ_k rm[m][k] · vvvv[k][n], with rm =
    // ModRM (tmm2) = plain M×K and vvvv (tmm1) = VNNI K×N. So B (VNNI) → tmm1
    // and A (plain) → tmm2. (bf16 has no signed/unsigned split, so the
    // single TDPBF16PS variant suffices once the operands are placed right.)
    for kb in 0..k_blocks {
        let a_ptr = a_bf16.as_ptr().add(kb * 32) as *const u8;
        let b_ptr = b_vnni.as_ptr().add(kb * 16 * 32) as *const u8;
        tile_load(1, b_ptr, b_stride); // B (VNNI) → tmm1 (vvvv)
        tile_load(2, a_ptr, a_stride); // A (plain) → tmm2 (rm)
        tile_dpbf16ps();
    }

    tile_store(0, c.as_mut_ptr() as *mut u8, 64);
    tile_release();
}

// ═════════════════════════════════════════════════════════════════════
// AVX-512 fallback (F32x16 + mul_add FMA)
// ═════════════════════════════════════════════════════════════════════

/// Fallback: delegate to the single source-of-truth SIMD-polyfill kernel
/// [`crate::simd::bf16_tile_gemm_16x16`] (BF16→f32 decode + `F32x16` FMA). The
/// `F32x16` wrapper owns the AVX-512 / AVX2 / NEON / scalar dispatch, so this
/// AMX wrapper only adds the TDPBF16PS tile path on top of the same kernel.
fn fallback_path(a_bf16: &[u16], b_bf16: &[u16], c: &mut [f32], k: usize) {
    crate::simd::bf16_tile_gemm_16x16(a_bf16, b_bf16, c, k);
}

// ═════════════════════════════════════════════════════════════════════
// Tests
// ═════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::{bf16_to_f32_batch, f32_to_bf16_batch};

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
            a_f32[i] =
                (((i as i32).wrapping_mul(1103515245).wrapping_add(12345) >> 8) as f32 / 2147483648.0).clamp(-1.0, 1.0);
        }
        for i in 0..b_f32.len() {
            b_f32[i] = (((i as i32).wrapping_mul(69069).wrapping_add(1) >> 8) as f32 / 2147483648.0).clamp(-1.0, 1.0);
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
            if e > max_err {
                max_err = e;
            }
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
        for v in c.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    /// bf16 encode by truncation — exact for the small integers used below.
    fn bf16_int(v: i32) -> u16 {
        let f = v as f32;
        debug_assert_eq!(f.to_bits() & 0xFFFF, 0);
        (f.to_bits() >> 16) as u16
    }

    /// Deterministic small-integer operands (all bf16-exact) + i64 reference.
    /// Every tier must reproduce the reference EXACTLY on this input class.
    fn integer_case(k: usize) -> (Vec<u16>, Vec<u16>, Vec<f32>) {
        let mut a = vec![0u16; 16 * k];
        let mut b = vec![0u16; k * 16];
        let mut s = 0x1BC0FFEEu64;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) % 15) as i32 - 7 // integers in [-7, 7]
        };
        for v in a.iter_mut() {
            *v = bf16_int(next());
        }
        for v in b.iter_mut() {
            *v = bf16_int(next());
        }
        // i64 reference on the decoded integers
        let dec = |u: u16| f32::from_bits((u as u32) << 16) as i64;
        let mut c_ref = vec![0.0f32; 256];
        for i in 0..16 {
            for j in 0..16 {
                let mut acc = 0i64;
                for kk in 0..k {
                    acc += dec(a[i * k + kk]) * dec(b[kk * 16 + j]);
                }
                c_ref[i * 16 + j] = acc as f32; // |acc| ≤ k·49 ≪ 2^24 → exact
            }
        }
        (a, b, c_ref)
    }

    #[test]
    fn vnni_index_matches_vnni_pack_bf16() {
        let k = 64;
        let (_, b, _) = integer_case(k);
        let packed = PackedBf16B::pack(&b, k);
        for row in 0..k {
            for col in 0..16 {
                assert_eq!(
                    packed.data()[PackedBf16B::vnni_index(row, col)],
                    b[row * 16 + col],
                    "vnni_index disagrees with vnni_pack_bf16 at [{row},{col}]"
                );
            }
        }
    }

    #[test]
    fn packed_and_unpacked_exact_on_integers() {
        // Whatever tier this host dispatches to, integer operands must come
        // out EXACT and identical between the packed and unpacked entries.
        let k = 64;
        let (a, b, c_ref) = integer_case(k);
        let mut c_unpacked = vec![0.0f32; 256];
        bf16_tile_gemm_16x16(&a, &b, &mut c_unpacked, k);
        let packed = PackedBf16B::pack(&b, k);
        let mut c_packed = vec![0.0f32; 256];
        bf16_tile_gemm_16x16_packed(&a, &packed, &mut c_packed);
        assert_eq!(c_unpacked, c_ref, "unpacked tier [{}] not exact", bf16_tile_gemm_tier());
        assert_eq!(c_packed, c_ref, "packed tier [{}] not exact", bf16_tile_gemm_tier());
    }

    #[test]
    fn packed_entry_accumulates() {
        // C += A·B semantics: pre-existing C values must survive.
        let k = 32;
        let (a, b, c_ref) = integer_case(k);
        let packed = PackedBf16B::pack(&b, k);
        let mut c = vec![5.0f32; 256];
        bf16_tile_gemm_16x16_packed(&a, &packed, &mut c);
        for i in 0..256 {
            assert_eq!(c[i], c_ref[i] + 5.0, "accumulate semantics broken at {i}");
        }
    }

    #[test]
    fn avx512bf16_path_matches_fallback() {
        if !is_x86_feature_detected!("avx512bf16") {
            eprintln!("SKIP (honestly): no avx512bf16 on this host — Gotcha 9: a skipped test is not a passing test");
            return;
        }
        let k = 64;
        // Exact-integer parity: bit-for-bit.
        let (a, b, c_ref) = integer_case(k);
        let packed = PackedBf16B::pack(&b, k);
        let mut c = vec![0.0f32; 256];
        // SAFETY: detection checked above.
        unsafe { avx512bf16_path(&a, packed.data(), &mut c, k) };
        assert_eq!(c, c_ref, "VDPBF16PS not exact on integer operands");

        // Float parity vs the decode+FMA fallback: same products (exact),
        // different accumulation order → tolerance-compare.
        let mut a_f = vec![0.0f32; 16 * k];
        let mut b_f = vec![0.0f32; k * 16];
        for (i, v) in a_f.iter_mut().enumerate() {
            *v = ((i as f32) * 0.37).sin();
        }
        for (i, v) in b_f.iter_mut().enumerate() {
            *v = ((i as f32) * 0.73).cos();
        }
        let mut a_bf = vec![0u16; a_f.len()];
        let mut b_bf = vec![0u16; b_f.len()];
        f32_to_bf16_batch(&a_f, &mut a_bf);
        f32_to_bf16_batch(&b_f, &mut b_bf);
        let packed_f = PackedBf16B::pack(&b_bf, k);
        let mut c_v = vec![0.0f32; 256];
        // SAFETY: detection checked above.
        unsafe { avx512bf16_path(&a_bf, packed_f.data(), &mut c_v, k) };
        let mut c_fb = vec![0.0f32; 256];
        fallback_path(&a_bf, &b_bf, &mut c_fb, k);
        for i in 0..256 {
            assert!((c_v[i] - c_fb[i]).abs() < 1e-3, "VDPBF16PS vs fallback at {i}: {} vs {}", c_v[i], c_fb[i]);
        }
    }

    /// Gotcha 14 parity-under-load gate. IGNORED by default: on an
    /// oversubscribed VM this FAILS BY DESIGN (AMX tile state silently
    /// corrupts under host CPU contention — see `.claude/AMX_GOTCHAS.md`
    /// § Gotcha 14). Run explicitly on dedicated silicon to certify:
    ///   cargo test -p ndarray --release bf16_tile_gemm -- --ignored
    #[test]
    #[ignore = "Gotcha 14: fails on oversubscribed VMs by design — run on dedicated silicon"]
    fn tile_parity_under_cpu_contention() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;
        if !amx_available() && !is_x86_feature_detected!("avx512bf16") {
            eprintln!("SKIP (honestly): no tile/vector-bf16 tier on this host");
            return;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let busy: Vec<_> = (0..3)
            .map(|_| {
                let s = Arc::clone(&stop);
                std::thread::spawn(move || {
                    let mut x = 0u64;
                    while !s.load(Ordering::Relaxed) {
                        x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
                        std::hint::black_box(x);
                    }
                })
            })
            .collect();
        // Large K on purpose: long tile residency (K=4096 holds tmm0 for
        // 128 TDPBF16PS iterations) maximizes the Gotcha-14 exposure window.
        let k = 4096;
        let (a, b, c_ref) = integer_case(k);
        let packed = PackedBf16B::pack(&b, k);
        let mut bad = 0usize;
        for _ in 0..200 {
            let mut c = vec![0.0f32; 256];
            bf16_tile_gemm_16x16_packed(&a, &packed, &mut c);
            if c != c_ref {
                bad += 1;
            }
        }
        stop.store(true, Ordering::Relaxed);
        for h in busy {
            let _ = h.join();
        }
        assert_eq!(bad, 0, "{bad}/200 GEMMs corrupted under load on tier [{}]", bf16_tile_gemm_tier());
    }
}
