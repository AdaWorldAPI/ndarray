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
        dispatch_vnni(a_bf16, k, &b_vnni, c, k);
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

    // ── Little-endian byte contract ─────────────────────────────────────
    //
    // The persistence/mailbox face of the buffer (the lance-graph
    // SoaEnvelope discipline: envelope bytes are LE from creation to
    // tombstone). This module is `cfg(target_arch = "x86_64")`, an LE-only
    // ISA, so the native `u16` lanes ARE the LE bytes — the view below is
    // a zero-cost reinterpret, and the contract costs nothing to state
    // explicitly. The read side decodes with `u16::from_le_bytes`, which
    // is endian-correct anywhere and compiles to a plain copy here.

    /// The buffer as **little-endian** bytes — `2·k·16` bytes, each bf16
    /// lane low-byte-first. Zero-copy on this (LE-only) architecture. This
    /// is the face a batch writer hands to columnar storage; pair with
    /// [`Self::from_le_bytes`] to round-trip.
    pub fn as_le_bytes(&self) -> &[u8] {
        // SAFETY: `&[u16]` → `&[u8]` reinterpret is always valid (alignment
        // requirement shrinks from 2 to 1; length doubles within the same
        // allocation). Byte order is LE by construction: x86_64 is
        // little-endian and this module is x86_64-only.
        unsafe { core::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 2) }
    }

    /// Rebuild from **little-endian** bytes (the inverse of
    /// [`Self::as_le_bytes`]). `bytes.len()` must be `2·k·16` and `k` a
    /// multiple of 32. Endian-correct on any architecture; compiles to a
    /// straight copy on LE targets.
    pub fn from_le_bytes(bytes: &[u8], k: usize) -> Self {
        assert_eq!(k % 32, 0, "K must be multiple of 32");
        assert_eq!(bytes.len(), 2 * k * 16, "expected 2·K·16 LE bytes");
        let data = bytes
            .chunks_exact(2)
            .map(|p| u16::from_le_bytes([p[0], p[1]]))
            .collect();
        PackedBf16B { data, k }
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
    dispatch_vnni(a_bf16, k, &b.data, c, k);
}

/// `C[16, 16] += A[16, K] × B[K, 16]` with B pre-packed and A read in place
/// from a strided buffer: row `i` of A is `a_bf16[i·a_row_stride ..][..K]`.
///
/// This is the entry for an A operand that already lives in memory with a
/// row pitch other than K, for example 16 equally spaced bf16 lanes of one
/// slab (row = lane, column = position within the lane). The tile tiers load
/// it where it is: AMX `TILELOADD` takes the row stride as an operand, and
/// the AVX-512 tier reads each row at its offset. Nothing is copied on those
/// tiers. The decode+FMA tier gathers the 16 rows into a dense 16×K buffer
/// first (it allocates), because the polyfill kernel consumes dense rows.
///
/// `a_row_stride` must be at least K, and `a_bf16` must hold
/// `15·a_row_stride + K` elements; both are checked. With `a_row_stride == K`
/// this is exactly [`bf16_tile_gemm_16x16_packed`].
///
/// # Examples
/// ```
/// use ndarray::simd::{bf16_tile_gemm_16x16_packed_strided, PackedBf16B};
/// let k = 32;
/// let pitch = 100; // 16 lanes, each 100 bf16 long, in one slab
/// let mut slab = vec![0u16; 16 * pitch];
/// slab[3 * pitch + 7] = 0x3F80; // lane 3, position 7 = 1.0
/// let mut b = PackedBf16B::zeroed(k);
/// b.data_mut()[PackedBf16B::vnni_index(7, 2)] = 0x3F80; // B[7][2] = 1.0
/// let mut c = vec![0.0f32; 256];
/// bf16_tile_gemm_16x16_packed_strided(&slab, pitch, &b, &mut c);
/// assert_eq!(c[3 * 16 + 2], 1.0); // C[3][2] = A[3][7]·B[7][2]
/// ```
pub fn bf16_tile_gemm_16x16_packed_strided(a_bf16: &[u16], a_row_stride: usize, b: &PackedBf16B, c: &mut [f32]) {
    let k = b.k;
    assert!(a_row_stride >= k, "A row stride {a_row_stride} is shorter than K = {k}");
    let needed = a_row_stride
        .checked_mul(15)
        .and_then(|n| n.checked_add(k))
        .expect("15·a_row_stride + K overflows usize");
    assert!(a_bf16.len() >= needed, "A holds {} bf16, needs {needed}", a_bf16.len());
    assert_eq!(c.len(), 16 * 16);
    dispatch_vnni(a_bf16, a_row_stride, &b.data, c, k);
}

/// The 16 rows of a strided A, copied into a dense 16×K buffer.
fn dense_rows(a_bf16: &[u16], a_row_stride: usize, k: usize) -> Vec<u16> {
    let mut dense = vec![0u16; 16 * k];
    for (i, row) in dense.chunks_exact_mut(k).enumerate() {
        row.copy_from_slice(&a_bf16[i * a_row_stride..i * a_row_stride + k]);
    }
    dense
}

/// Shared tier dispatch over a VNNI-packed B. Row `i` of A starts at
/// `a_bf16[i * a_row_stride]`; callers have checked the bounds.
fn dispatch_vnni(a_bf16: &[u16], a_row_stride: usize, b_vnni: &[u16], c: &mut [f32], k: usize) {
    if amx_available() {
        // SAFETY: amx_available() confirmed CPUID + XCR0 + arch_prctl grant.
        unsafe {
            amx_path(a_bf16, a_row_stride, b_vnni, c, k);
        }
    } else if is_x86_feature_detected!("avx512bf16") {
        // SAFETY: runtime detection confirmed avx512f+avx512bf16.
        unsafe {
            avx512bf16_path(a_bf16, a_row_stride, b_vnni, c, k);
        }
    } else {
        fallback_vnni(a_bf16, a_row_stride, b_vnni, c, k);
    }
}

/// The decode+FMA tier over a VNNI-packed B and a strided A. Allocates —
/// acceptable off the tile tiers (a host without AMX and without avx512bf16
/// calling the packed entry is not a hot-path configuration).
fn fallback_vnni(a_bf16: &[u16], a_row_stride: usize, b_vnni: &[u16], c: &mut [f32], k: usize) {
    if k == 0 {
        // C += A·B over an empty K leaves C unchanged, as on the tile tiers.
        return;
    }
    if a_row_stride != k {
        // The polyfill consumes dense rows: gather the 16 strided rows.
        return fallback_vnni(&dense_rows(a_bf16, a_row_stride, k), k, b_vnni, c, k);
    }
    // Unpack B back to row-major and take the decode+FMA polyfill.
    let mut b_rm = vec![0u16; k * 16];
    for row in 0..k {
        for col in 0..16 {
            b_rm[row * 16 + col] = b_vnni[PackedBf16B::vnni_index(row, col)];
        }
    }
    // The polyfill requires exactly 16·K elements; the strided entry accepts
    // a longer buffer, so pass only the 16 rows it uses.
    fallback_path(&a_bf16[..16 * k], &b_rm, c, k);
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
unsafe fn avx512bf16_path(a_bf16: &[u16], a_row_stride: usize, b_vnni: &[u16], c: &mut [f32], k: usize) {
    use core::arch::x86_64::{
        __m512bh, __m512i, _mm512_dpbf16_ps, _mm512_loadu_ps, _mm512_loadu_si512, _mm512_set1_epi32, _mm512_storeu_ps,
    };
    debug_assert_eq!(k % 2, 0);
    let pairs = k / 2;
    for i in 0..16 {
        // SAFETY (whole block): row i of A is a_bf16[i·a_row_stride..][..k]
        // and c is 16×16 (both asserted by the public callers), b_vnni is
        // k×16 pairs; every pointer below
        // stays inside those bounds. read_unaligned handles the u16→u32
        // pair load without alignment requirements.
        unsafe {
            let mut acc = _mm512_loadu_ps(c.as_ptr().add(i * 16));
            let a_row = a_bf16.as_ptr().add(i * a_row_stride);
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
unsafe fn amx_path(a_bf16: &[u16], a_row_stride: usize, b_vnni: &[u16], c: &mut [f32], k: usize) {
    // Tile config: shapes at K_bytes=64 match BF16 K=32 case
    let cfg = TileConfig::for_dpbusd(64);
    tile_loadconfig(&cfg);
    // Preload C accumulator from caller's buffer (was tile_zero(0)
    // pre-fix — see method-level note above).
    tile_load(0, c.as_ptr() as *const u8, 64);

    // Accumulate over K/32 tile blocks
    let k_blocks = k / 32;
    let a_stride = a_row_stride * 2; // A row stride in bytes (bf16 = 2B); TILELOADD takes it as an operand
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
    fn le_byte_view_roundtrips_and_is_truly_le() {
        let k = 64;
        let (_, b, _) = integer_case(k);
        let packed = PackedBf16B::pack(&b, k);

        // Contract: byte 2i is the LOW byte of lane i (little-endian),
        // independent of how the platform happens to lay out u16.
        let bytes = packed.as_le_bytes();
        assert_eq!(bytes.len(), 2 * k * 16);
        for (i, &lane) in packed.data().iter().enumerate() {
            assert_eq!([bytes[2 * i], bytes[2 * i + 1]], lane.to_le_bytes(), "lane {i} not LE in byte view");
        }

        // Round-trip: bytes → PackedBf16B → identical lanes AND identical
        // GEMM result (the envelope face and the compute face agree).
        let rebuilt = PackedBf16B::from_le_bytes(bytes, k);
        assert_eq!(rebuilt.data(), packed.data());
        let (a, _, c_ref) = integer_case(k);
        let mut c = vec![0.0f32; 256];
        bf16_tile_gemm_16x16_packed(&a, &rebuilt, &mut c);
        assert_eq!(c, c_ref, "GEMM over LE-roundtripped buffer not exact");
    }

    /// Scatter a dense 16×k A into a slab where row i starts at i·pitch,
    /// with a distinct sentinel in every gap so a wrong offset shows up.
    fn strided_slab(a: &[u16], k: usize, pitch: usize) -> Vec<u16> {
        let mut slab = vec![bf16_int(99); 15 * pitch + k];
        for i in 0..16 {
            slab[i * pitch..i * pitch + k].copy_from_slice(&a[i * k..(i + 1) * k]);
        }
        slab
    }

    #[test]
    fn strided_entry_reads_rows_in_place_exactly() {
        let k = 64;
        let (a, b, c_ref) = integer_case(k);
        let packed = PackedBf16B::pack(&b, k);
        for pitch in [k, k + 1, k + 32, 4 * k + 7] {
            let slab = strided_slab(&a, k, pitch);
            let mut c = vec![0.0f32; 256];
            bf16_tile_gemm_16x16_packed_strided(&slab, pitch, &packed, &mut c);
            assert_eq!(c, c_ref, "pitch {pitch}, tier [{}]", bf16_tile_gemm_tier());
        }
    }

    #[test]
    fn strided_rows_are_exact_on_every_tier_this_host_has() {
        // The dispatch above takes one tier per host; drive the others here.
        let k = 64;
        let pitch = k + 32;
        let (a, b, c_ref) = integer_case(k);
        let packed = PackedBf16B::pack(&b, k);
        let slab = strided_slab(&a, k, pitch);

        let mut c = vec![0.0f32; 256];
        let mut b_rm = vec![0u16; k * 16];
        for row in 0..k {
            for col in 0..16 {
                b_rm[row * 16 + col] = packed.data()[PackedBf16B::vnni_index(row, col)];
            }
        }
        fallback_path(&dense_rows(&slab, pitch, k), &b_rm, &mut c, k);
        assert_eq!(c, c_ref, "gather tier not exact");

        if is_x86_feature_detected!("avx512bf16") {
            let mut c = vec![0.0f32; 256];
            // SAFETY: detection checked above; slab holds 15·pitch + k.
            unsafe { avx512bf16_path(&slab, pitch, packed.data(), &mut c, k) };
            assert_eq!(c, c_ref, "VDPBF16PS strided not exact");
        } else {
            eprintln!("SKIP (honestly): no avx512bf16 — strided VDPBF16PS not exercised");
        }
        if amx_available() {
            let mut c = vec![0.0f32; 256];
            // SAFETY: amx_available() confirmed; slab holds 15·pitch + k.
            unsafe { amx_path(&slab, pitch, packed.data(), &mut c, k) };
            assert_eq!(c, c_ref, "TDPBF16PS strided not exact");
        } else {
            eprintln!("SKIP (honestly): no AMX — strided TDPBF16PS not exercised");
        }
    }

    /// The decode+FMA tier, driven directly so a host with a tile tier still
    /// covers it: an A slice longer than the 16 rows it uses, at stride == K.
    #[test]
    fn fallback_tier_accepts_a_longer_dense_a() {
        let k = 32;
        let (a, b, c_ref) = integer_case(k);
        let packed = PackedBf16B::pack(&b, k);
        let mut long_a = a.clone();
        long_a.push(bf16_int(99)); // 16·K + 1 elements: valid for the strided entry
        let mut c = vec![0.0f32; 256];
        fallback_vnni(&long_a, k, packed.data(), &mut c, k);
        assert_eq!(c, c_ref);
    }

    /// K = 0 is a valid call on every tier and leaves C unchanged.
    #[test]
    fn fallback_tier_accepts_zero_k_with_a_stride() {
        let packed = PackedBf16B::zeroed(0);
        let a = vec![0u16; 15];
        let mut c = vec![3.0f32; 256];
        fallback_vnni(&a, 1, packed.data(), &mut c, 0);
        assert_eq!(c, vec![3.0f32; 256]);
        // The public entry accepts the same call on whatever tier this host has.
        let mut c = vec![3.0f32; 256];
        bf16_tile_gemm_16x16_packed_strided(&a, 1, &packed, &mut c);
        assert_eq!(c, vec![3.0f32; 256]);
    }

    #[test]
    fn strided_entry_does_not_read_the_gap() {
        // Reading the slab densely (stride k) instead of at its pitch picks
        // up the sentinel gap and must give a different result.
        let k = 32;
        let (a, b, c_ref) = integer_case(k);
        let packed = PackedBf16B::pack(&b, k);
        let slab = strided_slab(&a, k, k + 32);
        let mut dense_misread = vec![0.0f32; 256];
        bf16_tile_gemm_16x16_packed(&slab[..16 * k], &packed, &mut dense_misread);
        assert_ne!(dense_misread, c_ref, "fixture cannot tell the pitch from K");
    }

    #[test]
    #[should_panic(expected = "shorter than K")]
    fn strided_entry_rejects_a_stride_below_k() {
        let packed = PackedBf16B::zeroed(32);
        let a = vec![0u16; 16 * 32];
        let mut c = vec![0.0f32; 256];
        bf16_tile_gemm_16x16_packed_strided(&a, 31, &packed, &mut c);
    }

    #[test]
    #[should_panic(expected = "needs")]
    fn strided_entry_rejects_a_short_buffer() {
        let packed = PackedBf16B::zeroed(32);
        let a = vec![0u16; 15 * 40 + 31];
        let mut c = vec![0.0f32; 256];
        bf16_tile_gemm_16x16_packed_strided(&a, 40, &packed, &mut c);
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
        unsafe { avx512bf16_path(&a, k, packed.data(), &mut c, k) };
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
        unsafe { avx512bf16_path(&a_bf, k, packed_f.data(), &mut c_v, k) };
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
