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
//!     when AMX isn't OS-enabled. VNNI-packs B on every call.
//!   * [`PackedI8B`] + [`int8_tile_gemm_16x16_packed`] /
//!     [`int8_tile_gemm_16x16_packed_strided`] — the same kernel with B
//!     packed once and, for the strided entry, A read in place at the
//!     caller's row pitch. The INT8 twins of `PackedBf16B` and
//!     `bf16_tile_gemm_16x16_packed[_strided]`.
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
    amx_available, tile_dpbusd, tile_dpbusd_2x2, tile_load, tile_loadconfig, tile_release, tile_store, tile_zero,
    vnni_pack_i8, TileConfig,
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
    let band = packed_len(k);
    assert_eq!(a_u8.len(), band);
    assert_eq!(b_i8.len(), band);
    assert_eq!(c.len(), 16 * 16);

    if amx_available() {
        // AMX path: pack B into VNNI quad layout, call tile GEMM.
        let mut b_vnni = vec![0i8; band];
        vnni_pack_i8(b_i8, &mut b_vnni, k, 16);
        // SAFETY: amx_available() just confirmed CPUID + XCR0 + prctl.
        unsafe {
            amx_path(a_u8, k, &b_vnni, c, k);
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
/// preserved. Row `i` of A starts at `a_u8[i * a_row_stride]`.
///
/// # Safety
/// Caller must have verified `amx_available() == true`, and that A holds
/// `15·a_row_stride + K` bytes with `a_row_stride >= K`.
#[inline]
unsafe fn amx_path(a_u8: &[u8], a_row_stride: usize, b_vnni: &[i8], c: &mut [i32], k: usize) {
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
    let a_stride = a_row_stride; // bytes per A row (u8 = 1 byte each)
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
// Pre-packed B — hoists VNNI packing (and its allocation) out of hot loops
// ═════════════════════════════════════════════════════════════════════

/// Which kernel [`int8_tile_gemm_16x16_packed`] /
/// [`int8_tile_gemm_16x16_packed_strided`] take on this host, as a
/// human-readable tier name, for run reports (AMX Gotcha 9: always print
/// which tier ran).
pub fn int8_tile_gemm_tier() -> &'static str {
    if amx_available() {
        "AMX TDPBUSD"
    } else {
        "scalar u8 x i8 -> i32"
    }
}

/// B\[K, 16\] i8 operand held in the VNNI quad layout AMX `TDPBUSD` reads:
/// quad-row `q` holds `(B[4q, j], B[4q+1, j], B[4q+2, j], B[4q+3, j])`
/// interleaved over `j` — 64 i8 = one 64-byte tile row. K must be a
/// multiple of 64.
///
/// Two ways to fill it, as for `PackedBf16B`:
/// - [`PackedI8B::pack`] / [`PackedI8B::pack_from`] — from row-major B
///   (`pack_from` reuses the buffer: no per-call allocation).
/// - Stage **directly** in VNNI layout via [`PackedI8B::vnni_index`] +
///   [`PackedI8B::data_mut`] — no packing pass.
///
/// B is the **signed** operand of `TDPBUSD`; A is unsigned. Nothing here
/// reinterprets either.
///
/// # Examples
/// ```
/// use ndarray::simd::{int8_tile_gemm_16x16_packed, PackedI8B};
/// let k = 64;
/// let mut b = PackedI8B::zeroed(k);
/// b.data_mut()[PackedI8B::vnni_index(5, 3)] = -2; // B[5][3] = -2
/// let a = vec![3u8; 16 * k]; // A = all 3
/// let mut c = vec![0i32; 256];
/// int8_tile_gemm_16x16_packed(&a, &b, &mut c);
/// assert_eq!(c[3], -6); // C[0][3] = A[0][5]·B[5][3]
/// ```
pub struct PackedI8B {
    data: Vec<i8>,
    k: usize,
}

/// `16·K`, the byte length of a 16-wide band (and of a 16-row A). Panics
/// instead of wrapping: a wrapped length would let an undersized buffer
/// through the length checks in front of the AMX kernel.
fn packed_len(k: usize) -> usize {
    k.checked_mul(16).expect("16·K overflows usize")
}

impl PackedI8B {
    /// All-zero packed B for direct VNNI staging. `k` must be a multiple of 64.
    pub fn zeroed(k: usize) -> Self {
        assert_eq!(k % 64, 0, "K must be multiple of 64 for TDPBUSD tiles");
        PackedI8B {
            data: vec![0i8; packed_len(k)],
            k,
        }
    }

    /// Pack row-major B\[K, 16\] (allocates once; prefer [`Self::pack_from`]
    /// for repeated packs).
    pub fn pack(b_i8_row_major: &[i8], k: usize) -> Self {
        let mut p = Self::zeroed(k);
        p.pack_from(b_i8_row_major);
        p
    }

    /// Re-pack row-major B\[K, 16\] into the existing buffer — no allocation.
    pub fn pack_from(&mut self, b_i8_row_major: &[i8]) {
        assert_eq!(b_i8_row_major.len(), self.k * 16);
        vnni_pack_i8(b_i8_row_major, &mut self.data, self.k, 16);
    }

    /// Flat index of logical `B[row, col]` inside the VNNI buffer.
    /// `vnni[(row/4)·64 + 4·col + (row & 3)] == B[row, col]`.
    #[inline(always)]
    pub const fn vnni_index(row: usize, col: usize) -> usize {
        (row / 4) * 64 + 4 * col + (row & 3)
    }

    /// The K dimension this buffer was sized for.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Read access to the raw VNNI buffer.
    pub fn data(&self) -> &[i8] {
        &self.data
    }

    /// Mutable access to the raw VNNI buffer, for direct staging via
    /// [`Self::vnni_index`].
    pub fn data_mut(&mut self) -> &mut [i8] {
        &mut self.data
    }

    /// The buffer as bytes — `k·16` bytes, one per i8 lane. An i8 has no
    /// byte order, so this is the same in any endianness; it is the face a
    /// batch writer hands to columnar storage. Pair with
    /// [`Self::from_le_bytes`].
    pub fn as_le_bytes(&self) -> &[u8] {
        // SAFETY: `&[i8]` → `&[u8]` has the same size and alignment per
        // element; the slice stays within the same allocation.
        unsafe { core::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len()) }
    }

    /// Rebuild from bytes (the inverse of [`Self::as_le_bytes`]).
    /// `bytes.len()` must be `k·16` and `k` a multiple of 64.
    pub fn from_le_bytes(bytes: &[u8], k: usize) -> Self {
        assert_eq!(k % 64, 0, "K must be multiple of 64 for TDPBUSD tiles");
        assert_eq!(bytes.len(), packed_len(k), "expected K·16 bytes");
        let data = bytes.iter().map(|&b| i8::from_ne_bytes([b])).collect();
        PackedI8B { data, k }
    }
}

/// `C[16, 16] += A[16, K] × B[K, 16]` with B pre-packed — the hot-loop
/// sibling of [`int8_tile_gemm_16x16`] with the per-call VNNI pack (and its
/// allocation) hoisted into [`PackedI8B`]. A is u8 row-major (`16·K`), C is
/// i32 row-major (256). **Accumulates.**
///
/// Tier dispatch (runtime): AMX `TDPBUSD` → scalar reference. Both tiers are
/// exact (integer arithmetic). [`int8_tile_gemm_tier`] names the tier.
pub fn int8_tile_gemm_16x16_packed(a_u8: &[u8], b: &PackedI8B, c: &mut [i32]) {
    let k = b.k;
    assert_eq!(a_u8.len(), packed_len(k));
    assert_eq!(c.len(), 16 * 16);
    dispatch_vnni(a_u8, k, &b.data, c, k);
}

/// `C[16, 16] += A[16, K] × B[K, 16]` with B pre-packed and A read in place
/// from a strided buffer: row `i` of A is `a_u8[i·a_row_stride ..][..K]`.
///
/// For an A operand that already lives in memory with a row pitch other than
/// K, for example 16 equally spaced u8 lanes of one slab (row = lane, column
/// = position within the lane). AMX `TILELOADD` takes the stride as an
/// operand, so nothing is copied on that tier. The scalar tier reads each row
/// at its offset.
///
/// `a_row_stride` must be at least K, and `a_u8` must hold
/// `15·a_row_stride + K` bytes; both are checked. With `a_row_stride == K`
/// this is exactly [`int8_tile_gemm_16x16_packed`].
///
/// # Examples
/// ```
/// use ndarray::simd::{int8_tile_gemm_16x16_packed_strided, PackedI8B};
/// let k = 64;
/// let pitch = 100; // 16 lanes, each 100 u8 long, in one slab
/// let mut slab = vec![0u8; 16 * pitch];
/// slab[3 * pitch + 7] = 200; // lane 3, position 7
/// let mut b = PackedI8B::zeroed(k);
/// b.data_mut()[PackedI8B::vnni_index(7, 2)] = -1; // B[7][2] = -1
/// let mut c = vec![0i32; 256];
/// int8_tile_gemm_16x16_packed_strided(&slab, pitch, &b, &mut c);
/// assert_eq!(c[3 * 16 + 2], -200); // A is unsigned: 200, not -56
/// ```
pub fn int8_tile_gemm_16x16_packed_strided(a_u8: &[u8], a_row_stride: usize, b: &PackedI8B, c: &mut [i32]) {
    let k = b.k;
    assert!(a_row_stride >= k, "A row stride {a_row_stride} is shorter than K = {k}");
    let needed = a_row_stride
        .checked_mul(15)
        .and_then(|n| n.checked_add(k))
        .expect("15·a_row_stride + K overflows usize");
    assert!(a_u8.len() >= needed, "A holds {} bytes, needs {needed}", a_u8.len());
    assert_eq!(c.len(), 16 * 16);
    dispatch_vnni(a_u8, a_row_stride, &b.data, c, k);
}

/// `C[M, N] = A[M, K] × B[K, N]` over **prepacked** B bands and a **strided**
/// A: the multi-tile sibling of [`int8_tile_gemm_16x16_packed_strided`].
///
/// - `b` holds B as `N/16` [`PackedI8B`] bands (band `t` = output columns
///   `16t..16t+16`), all of the same K. Packed once by the caller and reused
///   across calls, so no VNNI pack runs here.
/// - Row `i` of A is `a_u8[i·a_row_stride ..][..K]`, read in place.
/// - `c` is `M x N` row-major and is **overwritten**, as by
///   [`int8_gemm_amx_tiled`].
///
/// On AMX this is `int8_gemm_amx_tiled`'s 2×2 register-blocked kernel (one
/// tile configuration for the whole call, 32×32 blocks, 16-wide remainders)
/// minus its per-call band gather and pack. Without AMX it runs the scalar
/// reference per 16×16 tile. `M` must be a multiple of 16. Both tiers are
/// exact.
///
/// # Examples
/// ```
/// use ndarray::simd::{int8_gemm_packed_strided, PackedI8B};
/// let (m, k, pitch) = (32, 64, 80);
/// let a = vec![2u8; (m - 1) * pitch + k];
/// let bands = [PackedI8B::pack(&vec![-3i8; k * 16], k), PackedI8B::pack(&vec![1i8; k * 16], k)];
/// let mut c = vec![0i32; m * 32];
/// int8_gemm_packed_strided(&a, pitch, &bands, &mut c, m);
/// assert_eq!((c[0], c[16]), (2 * -3 * 64, 2 * 64));
/// ```
pub fn int8_gemm_packed_strided(a_u8: &[u8], a_row_stride: usize, b: &[PackedI8B], c: &mut [i32], m: usize) {
    assert!(!b.is_empty(), "int8_gemm_packed_strided: no B bands");
    let k = b[0].k;
    assert!(b.iter().all(|band| band.k == k), "int8_gemm_packed_strided: B bands differ in K");
    assert_eq!(m % 16, 0, "int8_gemm_packed_strided: M must be multiple of 16");
    assert!(a_row_stride >= k, "A row stride {a_row_stride} is shorter than K = {k}");
    let n = b
        .len()
        .checked_mul(16)
        .expect("N = 16·bands overflows usize");
    if m == 0 {
        return;
    }
    let needed = a_row_stride
        .checked_mul(m - 1)
        .and_then(|x| x.checked_add(k))
        .expect("(M-1)·a_row_stride + K overflows usize");
    assert!(a_u8.len() >= needed, "A holds {} bytes, needs {needed}", a_u8.len());
    let out = m
        .checked_mul(n)
        .expect("int8_gemm_packed_strided: M·N overflows usize");
    assert!(c.len() >= out, "int8_gemm_packed_strided: c.len()={} < m*n={out}", c.len());
    if amx_available() {
        // SAFETY: amx_available() confirmed CPUID + XCR0 + arch_prctl; every
        // bound the kernel relies on is asserted above.
        unsafe { packed_strided_amx(a_u8, a_row_stride, b, c, m, n, k) }
    } else {
        packed_strided_scalar(a_u8, a_row_stride, b, c, m, n, k);
    }
}

/// The scalar tier of [`int8_gemm_packed_strided`], per 16×16 tile.
fn packed_strided_scalar(
    a_u8: &[u8], a_row_stride: usize, b: &[PackedI8B], c: &mut [i32], m: usize, n: usize, k: usize,
) {
    let mut tile = [0i32; 256];
    for i in (0..m).step_by(16) {
        for (t, band) in b.iter().enumerate() {
            tile.fill(0);
            fallback_vnni(&a_u8[i * a_row_stride..], a_row_stride, &band.data, &mut tile, k);
            for r in 0..16 {
                let at = (i + r) * n + t * 16;
                c[at..at + 16].copy_from_slice(&tile[r * 16..(r + 1) * 16]);
            }
        }
    }
}

/// `int8_gemm_amx_tiled_rb`'s kernel over prepacked bands and a strided A.
///
/// # Safety
/// `amx_available()` must hold, and the bounds asserted by
/// [`int8_gemm_packed_strided`] must hold.
unsafe fn packed_strided_amx(
    a_u8: &[u8], a_row_stride: usize, b: &[PackedI8B], c: &mut [i32], m: usize, n: usize, k: usize,
) {
    let k_blocks = k / 64;
    let m32 = (m / 32) * 32;
    let pitch = n * 4;
    let s = a_row_stride;
    let cp = c.as_mut_ptr();
    tile_loadconfig(&TileConfig::for_dpbusd_8());
    let mut t = 0;
    while t + 1 < b.len() {
        let (b0p, b1p) = (b[t].data.as_ptr() as *const u8, b[t + 1].data.as_ptr() as *const u8);
        let j2 = t * 16;
        let mut i2 = 0;
        while i2 < m32 {
            let a0 = a_u8.as_ptr().add(i2 * s);
            let a1 = a_u8.as_ptr().add((i2 + 16) * s);
            tile_zero(0);
            tile_zero(1);
            tile_zero(2);
            tile_zero(3);
            for kb in 0..k_blocks {
                tile_load(4, a0.add(kb * 64), s);
                tile_load(5, a1.add(kb * 64), s);
                tile_load(6, b0p.add(kb * 16 * 64), 64);
                tile_load(7, b1p.add(kb * 16 * 64), 64);
                tile_dpbusd_2x2();
            }
            tile_store(0, cp.add(i2 * n + j2) as *mut u8, pitch);
            tile_store(1, cp.add(i2 * n + j2 + 16) as *mut u8, pitch);
            tile_store(2, cp.add((i2 + 16) * n + j2) as *mut u8, pitch);
            tile_store(3, cp.add((i2 + 16) * n + j2 + 16) as *mut u8, pitch);
            i2 += 32;
        }
        if m32 < m {
            let ap = a_u8.as_ptr().add(m32 * s);
            for (off, bv) in [(0usize, b0p), (16usize, b1p)] {
                tile_zero(0);
                for kb in 0..k_blocks {
                    tile_load(1, bv.add(kb * 16 * 64), 64);
                    tile_load(2, ap.add(kb * 64), s);
                    tile_dpbusd();
                }
                tile_store(0, cp.add(m32 * n + j2 + off) as *mut u8, pitch);
            }
        }
        t += 2;
    }
    if t < b.len() {
        let bvp = b[t].data.as_ptr() as *const u8;
        let j = t * 16;
        let mut i = 0;
        while i < m {
            let ap = a_u8.as_ptr().add(i * s);
            tile_zero(0);
            for kb in 0..k_blocks {
                tile_load(1, bvp.add(kb * 16 * 64), 64);
                tile_load(2, ap.add(kb * 64), s);
                tile_dpbusd();
            }
            tile_store(0, cp.add(i * n + j) as *mut u8, pitch);
            i += 16;
        }
    }
    tile_release();
}

/// Shared tier dispatch over a VNNI-packed B. Row `i` of A starts at
/// `a_u8[i * a_row_stride]`; callers have checked the bounds.
fn dispatch_vnni(a_u8: &[u8], a_row_stride: usize, b_vnni: &[i8], c: &mut [i32], k: usize) {
    if amx_available() {
        // SAFETY: amx_available() confirmed CPUID + XCR0 + arch_prctl grant;
        // the public callers checked A's stride and length and C's length.
        unsafe {
            amx_path(a_u8, a_row_stride, b_vnni, c, k);
        }
    } else {
        fallback_vnni(a_u8, a_row_stride, b_vnni, c, k);
    }
}

/// The scalar tier over a VNNI-packed B and a strided A: reads B through
/// [`PackedI8B::vnni_index`] and each A row at its offset. Accumulates.
fn fallback_vnni(a_u8: &[u8], a_row_stride: usize, b_vnni: &[i8], c: &mut [i32], k: usize) {
    for i in 0..16 {
        let row = &a_u8[i * a_row_stride..i * a_row_stride + k];
        for (kk, &a) in row.iter().enumerate() {
            let a_val = i32::from(a);
            for j in 0..16 {
                c[i * 16 + j] += a_val * i32::from(b_vnni[PackedI8B::vnni_index(kk, j)]);
            }
        }
    }
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
/// `(m, n, k)`, mirroring the boundary contract from `gemm_u8_i8`; if `m`
/// or `n` is not a multiple of 16 or `k` not a multiple of 64; or if AMX
/// is not available (`crate::hpc::amx_matmul::amx_available()`). These
/// were debug-only until 2026-10-11, which let a release build index
/// past the slices.
pub fn int8_gemm_amx_tiled(a_u8: &[u8], b_i8: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    // Length assertions (codex P1 from PR #185 — the function reads
    // `b_i8` via a 16-wide window per (kk, j_tile) iteration and a_u8
    // via a 16-row slice per i_tile, so mismatched shapes would
    // trigger out-of-bounds reads without these gates).
    assert!(a_u8.len() >= m * k, "int8_gemm_amx_tiled: a_u8.len()={} < m*k={}", a_u8.len(), m * k);
    assert!(b_i8.len() >= k * n, "int8_gemm_amx_tiled: b_i8.len()={} < k*n={}", b_i8.len(), k * n);
    assert!(c.len() >= m * n, "int8_gemm_amx_tiled: c.len()={} < m*n={}", c.len(), m * n);

    // Real asserts, not `debug_assert!`: this is a safe `pub fn`, and the
    // tile loops index whole 16×64 tiles, so a misaligned shape reads and
    // writes past the slices in a release build (unsafe inventory,
    // 2026-10-10). Shape first, so a misaligned call on a machine without
    // AMX names the shape. `amx_available()` is a cached static.
    assert_eq!(m % 16, 0, "int8_gemm_amx_tiled: M must be multiple of 16");
    assert_eq!(n % 16, 0, "int8_gemm_amx_tiled: N must be multiple of 16");
    assert_eq!(k % 64, 0, "int8_gemm_amx_tiled: K must be multiple of 64");
    assert!(crate::hpc::amx_matmul::amx_available(), "int8_gemm_amx_tiled: AMX is not available on this CPU");

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
    // 2×2 register blocking halves the load bytes/MAC — the right lever for this
    // bandwidth-bound kernel — whenever there is at least one full 32×32 block.
    if m >= 32 && n >= 32 {
        int8_gemm_amx_tiled_rb(a_u8, b_i8, c, m, n, k);
        return;
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

/// 2×2 register-blocked AMX int8 GEMM (single thread). Tiles the M×N output
/// into 32×32 blocks computed with [`tile_dpbusd_2x2`] — four C accumulators
/// (tmm0-3) fed by two A row-tiles (tmm4-5) and two B col-tiles (tmm6-7), so
/// each A/B tile load serves TWO products (half the tile loads per MAC).
///
/// Loop order is BLIS-style for cache behaviour: OUTER over 32-col panels —
/// pack just that panel's two B bands (≈2·K·16 bytes, kept L2-resident) and
/// reuse them across every row-block — INNER over 32-row blocks. This halves
/// the tile loads (register reuse) AND halves A's DRAM re-reads vs the 16×16
/// kernel (32-col panels ⇒ half as many full-A sweeps). The earlier version
/// pre-packed ALL of B (≈4 MB at 2048²) and thrashed cache, regressing large
/// shapes — this packs only the live panel. 16-wide M/N remainders (m or n ≡
/// 16 mod 32) finish on the validated 16×16 path. Overwrites `c`.
fn int8_gemm_amx_tiled_rb(a_u8: &[u8], b_i8: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    let band = k * 16; // bytes per VNNI-packed 16-col band
    let k_blocks = k / 64;
    let m32 = (m / 32) * 32; // rows covered by full 32-row blocks
    let n32 = (n / 32) * 32; // cols covered by full 32-col blocks
    let pitch = n * 4; // C row pitch in bytes

    // The live panel's two B bands + a gather scratch. Packed once per panel,
    // reused across all row-blocks (hot in L2).
    let mut b0 = vec![0i8; band];
    let mut b1 = vec![0i8; band];
    let mut t = vec![0i8; band];

    // SAFETY: caller asserted amx_available + 16/16/64 alignment + slice bounds.
    // Eight tiles are configured (for_dpbusd_8); every tile_load/tile_store is a
    // 16×64 tile within the a_u8 / b0 / b1 / c allocations; all stores use the
    // full-matrix row pitch n·4. b0/b1 raw pointers stay valid across the inner
    // loops (the bands are repacked only at the top of the next panel).
    unsafe {
        tile_loadconfig(&TileConfig::for_dpbusd_8());

        // ── Full 32-col panels ──
        let mut j2 = 0;
        while j2 < n32 {
            // Pack this panel's two 16-col B bands once; reused across all i2.
            for kk in 0..k {
                t[kk * 16..(kk + 1) * 16].copy_from_slice(&b_i8[kk * n + j2..kk * n + j2 + 16]);
            }
            vnni_pack_i8(&t, &mut b0, k, 16);
            for kk in 0..k {
                t[kk * 16..(kk + 1) * 16].copy_from_slice(&b_i8[kk * n + j2 + 16..kk * n + j2 + 32]);
            }
            vnni_pack_i8(&t, &mut b1, k, 16);
            let b0p = b0.as_ptr() as *const u8;
            let b1p = b1.as_ptr() as *const u8;

            // 32×32 register-blocked blocks down the rows.
            let mut i2 = 0;
            while i2 < m32 {
                let a0 = a_u8.as_ptr().add(i2 * k);
                let a1 = a_u8.as_ptr().add((i2 + 16) * k);
                tile_zero(0);
                tile_zero(1);
                tile_zero(2);
                tile_zero(3);
                for kb in 0..k_blocks {
                    tile_load(4, a0.add(kb * 64), k);
                    tile_load(5, a1.add(kb * 64), k);
                    tile_load(6, b0p.add(kb * 16 * 64), 64);
                    tile_load(7, b1p.add(kb * 16 * 64), 64);
                    tile_dpbusd_2x2();
                }
                let cp = c.as_mut_ptr();
                tile_store(0, cp.add(i2 * n + j2) as *mut u8, pitch);
                tile_store(1, cp.add(i2 * n + j2 + 16) as *mut u8, pitch);
                tile_store(2, cp.add((i2 + 16) * n + j2) as *mut u8, pitch);
                tile_store(3, cp.add((i2 + 16) * n + j2 + 16) as *mut u8, pitch);
                i2 += 32;
            }

            // Bottom strip for THIS panel: rows [m32..m) × cols [j2..j2+32),
            // two 16×16 tiles reusing the already-packed b0 / b1.
            if m32 < m {
                let ap = a_u8.as_ptr().add(m32 * k);
                for (off, bv) in [(0usize, b0p), (16usize, b1p)] {
                    tile_zero(0);
                    for kb in 0..k_blocks {
                        tile_load(1, bv.add(kb * 16 * 64), 64);
                        tile_load(2, ap.add(kb * 64), k);
                        tile_dpbusd();
                    }
                    tile_store(0, c.as_mut_ptr().add(m32 * n + j2 + off) as *mut u8, pitch);
                }
            }
            j2 += 32;
        }

        // ── Right strip: cols [n32..n) (one 16-band), ALL rows (covers the
        // bottom-right corner too). ──
        if n32 < n {
            for kk in 0..k {
                t[kk * 16..(kk + 1) * 16].copy_from_slice(&b_i8[kk * n + n32..kk * n + n32 + 16]);
            }
            vnni_pack_i8(&t, &mut b0, k, 16);
            let bvp = b0.as_ptr() as *const u8;
            let mut i = 0;
            while i < m {
                let ap = a_u8.as_ptr().add(i * k);
                tile_zero(0);
                for kb in 0..k_blocks {
                    tile_load(1, bvp.add(kb * 16 * 64), 64);
                    tile_load(2, ap.add(kb * 64), k);
                    tile_dpbusd();
                }
                tile_store(0, c.as_mut_ptr().add(i * n + n32) as *mut u8, pitch);
                i += 16;
            }
        }

        tile_release();
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

    /// Deterministic u8 A (16 rows at `pitch`) and i8 B (K x 16), covering
    /// the full u8 and i8 ranges.
    fn operands(k: usize, pitch: usize, seed: usize) -> (Vec<u8>, Vec<i8>) {
        let a = (0..15 * pitch + k)
            .map(|i| ((i * 37 + seed) % 256) as u8)
            .collect();
        let b = (0..k * 16)
            .map(|i| ((i * 59 + seed * 3) % 256) as u8 as i8)
            .collect();
        (a, b)
    }

    /// `C = A·B` with A read at `pitch`, by the scalar definition.
    fn ref_strided(a: &[u8], pitch: usize, b: &[i8], k: usize) -> Vec<i32> {
        let mut c = vec![0i32; 256];
        for i in 0..16 {
            for j in 0..16 {
                c[i * 16 + j] = (0..k)
                    .map(|kk| i32::from(a[i * pitch + kk]) * i32::from(b[kk * 16 + j]))
                    .sum();
            }
        }
        c
    }

    #[test]
    fn vnni_index_matches_vnni_pack_i8() {
        let k = 128;
        let (_, b) = operands(k, k, 1);
        let p = PackedI8B::pack(&b, k);
        for row in 0..k {
            for col in 0..16 {
                assert_eq!(p.data()[PackedI8B::vnni_index(row, col)], b[row * 16 + col], "B[{row}][{col}]");
            }
        }
    }

    #[test]
    fn packed_equals_the_row_major_entry() {
        for k in [64, 128, 192] {
            let (a, b) = operands(k, k, k);
            let a = &a[..16 * k];
            let mut want = vec![0i32; 256];
            int8_tile_gemm_16x16(a, &b, &mut want, k);
            let mut got = vec![0i32; 256];
            int8_tile_gemm_16x16_packed(a, &PackedI8B::pack(&b, k), &mut got);
            assert_eq!(got, want, "k = {k}");
            assert_eq!(got, ref_strided(a, k, &b, k), "k = {k}");
        }
    }

    /// A read in place at a pitch longer than K gives the dense result; on
    /// an AMX host this runs TILELOADD at that stride.
    #[test]
    fn strided_reads_a_in_place_at_its_pitch() {
        let k = 128;
        for pitch in [k, k + 1, k + 64, 3 * k + 7] {
            let (a, b) = operands(k, pitch, pitch);
            let p = PackedI8B::pack(&b, k);
            let mut got = vec![0i32; 256];
            int8_tile_gemm_16x16_packed_strided(&a, pitch, &p, &mut got);
            assert_eq!(got, ref_strided(&a, pitch, &b, k), "pitch = {pitch} on {}", int8_tile_gemm_tier());
        }
    }

    /// The scalar tier, tested directly so an AMX host covers it too.
    #[test]
    fn scalar_tier_matches_reference_at_a_pitch() {
        let (k, pitch) = (128, 200);
        let (a, b) = operands(k, pitch, 9);
        let mut got = vec![0i32; 256];
        fallback_vnni(&a, pitch, PackedI8B::pack(&b, k).data(), &mut got, k);
        assert_eq!(got, ref_strided(&a, pitch, &b, k));
    }

    /// A is unsigned and B signed: 200 x -1 is -200 per position, never
    /// 200 x 255 or -56 x -1.
    #[test]
    fn packed_keeps_the_unsigned_a_signed_b_contract() {
        let k = 64;
        let a = vec![200u8; 16 * k];
        let p = PackedI8B::pack(&vec![-1i8; k * 16], k);
        let mut c = vec![0i32; 256];
        int8_tile_gemm_16x16_packed(&a, &p, &mut c);
        assert!(c.iter().all(|&v| v == -200 * k as i32), "{:?}", &c[..4]);
    }

    #[test]
    fn packed_accumulates_into_c() {
        let k = 64;
        let (a, b) = operands(k, k, 4);
        let a = &a[..16 * k];
        let marker: Vec<i32> = (0..256).map(|i| i * 7 - 100).collect();
        let mut c = marker.clone();
        int8_tile_gemm_16x16_packed(a, &PackedI8B::pack(&b, k), &mut c);
        let prod = ref_strided(a, k, &b, k);
        for i in 0..256 {
            assert_eq!(c[i], marker[i] + prod[i], "at {i}");
        }
    }

    #[test]
    fn pack_from_reuses_the_buffer_and_bytes_round_trip() {
        let k = 64;
        let (_, b1) = operands(k, k, 1);
        let (_, b2) = operands(k, k, 2);
        let mut p = PackedI8B::pack(&b1, k);
        let ptr = p.data().as_ptr();
        p.pack_from(&b2);
        assert_eq!(p.data().as_ptr(), ptr, "pack_from must not reallocate");
        assert_eq!(p.data(), PackedI8B::pack(&b2, k).data());
        let back = PackedI8B::from_le_bytes(p.as_le_bytes(), k);
        assert_eq!(back.data(), p.data());
        assert_eq!(back.k(), k);
    }

    /// `C = A·B` for M rows at `pitch` and N = 16·bands columns.
    fn ref_mn(a: &[u8], pitch: usize, b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] = (0..k)
                    .map(|kk| i32::from(a[i * pitch + kk]) * i32::from(b[kk * n + j]))
                    .sum();
            }
        }
        c
    }

    /// Every shape the 2×2 kernel distinguishes: full 32×32 blocks, a
    /// 16-row bottom strip, an odd band count (right strip), and M = 16.
    #[test]
    fn multi_tile_packed_strided_matches_reference_on_every_strip() {
        for (m, bands, k, pitch) in
            [(32, 2, 64, 64), (48, 3, 128, 200), (16, 1, 64, 64), (64, 4, 192, 256), (80, 5, 64, 97)]
        {
            let n = bands * 16;
            let a: Vec<u8> = (0..(m - 1) * pitch + k)
                .map(|i| ((i * 37 + m) % 256) as u8)
                .collect();
            let b: Vec<i8> = (0..k * n)
                .map(|i| ((i * 59 + n) % 256) as u8 as i8)
                .collect();
            let packed: Vec<PackedI8B> = (0..bands)
                .map(|t| {
                    let band: Vec<i8> = (0..k)
                        .flat_map(|kk| b[kk * n + t * 16..kk * n + t * 16 + 16].to_vec())
                        .collect();
                    PackedI8B::pack(&band, k)
                })
                .collect();
            let mut c = vec![9i32; m * n];
            int8_gemm_packed_strided(&a, pitch, &packed, &mut c, m);
            assert_eq!(
                c,
                ref_mn(&a, pitch, &b, m, n, k),
                "m {m} bands {bands} k {k} pitch {pitch} on {}",
                int8_tile_gemm_tier()
            );
            let mut cs = vec![9i32; m * n];
            packed_strided_scalar(&a, pitch, &packed, &mut cs, m, n, k);
            assert_eq!(cs, c, "scalar tier, m {m} bands {bands}");
        }
    }

    #[test]
    #[should_panic(expected = "differ in K")]
    fn multi_tile_refuses_bands_of_different_k() {
        let bands = [PackedI8B::zeroed(64), PackedI8B::zeroed(128)];
        int8_gemm_packed_strided(&vec![0u8; 16 * 128], 128, &bands, &mut vec![0i32; 16 * 32], 16);
    }

    #[test]
    #[should_panic(expected = "needs")]
    fn multi_tile_refuses_an_a_too_short_for_m_rows() {
        let bands = [PackedI8B::zeroed(64)];
        int8_gemm_packed_strided(&vec![0u8; 31 * 100 + 63], 100, &bands, &mut vec![0i32; 32 * 16], 32);
    }

    #[test]
    #[should_panic(expected = "shorter than K")]
    fn strided_refuses_a_pitch_shorter_than_k() {
        let k = 64;
        let p = PackedI8B::zeroed(k);
        int8_tile_gemm_16x16_packed_strided(&vec![0u8; 16 * k], k - 1, &p, &mut vec![0i32; 256]);
    }

    #[test]
    #[should_panic(expected = "needs")]
    fn strided_refuses_an_a_too_short_for_its_pitch() {
        let k = 64;
        let p = PackedI8B::zeroed(k);
        int8_tile_gemm_16x16_packed_strided(&vec![0u8; 15 * 100 + k - 1], 100, &p, &mut vec![0i32; 256]);
    }

    /// K = 0 is a legal degenerate size: the multi-tile entry still
    /// overwrites C with zeros, over full 32-row blocks, a 16-row strip and an
    /// odd band count, on the dispatched tier (AMX on an AMX host) and on the
    /// scalar tier called directly. A and its stride are empty.
    #[test]
    fn k0_multi_tile_overwrites_with_zeros_on_both_tiers() {
        let bands = [PackedI8B::zeroed(0), PackedI8B::zeroed(0), PackedI8B::zeroed(0)];
        let (m, n) = (48, 48);
        let mut c = vec![7i32; m * n];
        int8_gemm_packed_strided(&[], 0, &bands, &mut c, m);
        assert!(c.iter().all(|&v| v == 0), "dispatched tier {}", int8_tile_gemm_tier());
        let mut c = vec![7i32; m * n];
        packed_strided_scalar(&[], 0, &bands, &mut c, m, n, 0);
        assert!(c.iter().all(|&v| v == 0), "scalar tier");
    }

    /// K = 0 through the accumulating single-tile entry leaves C unchanged on
    /// the dispatched tier and the scalar tier.
    #[test]
    fn k0_single_tile_accumulates_nothing_on_both_tiers() {
        let b = PackedI8B::zeroed(0);
        let mut c = vec![5i32; 256];
        int8_tile_gemm_16x16_packed(&[], &b, &mut c);
        int8_tile_gemm_16x16_packed_strided(&[], 0, &b, &mut c);
        assert!(c.iter().all(|&v| v == 5), "dispatched tier {}", int8_tile_gemm_tier());
        fallback_vnni(&[], 0, b.data(), &mut c, 0);
        assert!(c.iter().all(|&v| v == 5), "scalar tier");
    }

    /// M = 0 returns before any bound that depends on M and never touches C.
    #[test]
    fn m0_multi_tile_is_a_no_op() {
        let bands = [PackedI8B::zeroed(64)];
        int8_gemm_packed_strided(&[], 64, &bands, &mut [], 0);
    }

    /// `(M-1)·stride` overflowing must panic, not wrap into a small "needed".
    #[test]
    #[should_panic(expected = "overflows usize")]
    fn multi_tile_refuses_a_stride_whose_extent_overflows() {
        let bands = [PackedI8B::zeroed(64)];
        int8_gemm_packed_strided(&vec![0u8; 64], usize::MAX / 4, &bands, &mut vec![0i32; 32 * 16], 32);
    }

    /// `15·stride` overflowing must panic, not wrap into a small "needed".
    #[test]
    #[should_panic(expected = "overflows usize")]
    fn strided_refuses_a_stride_whose_extent_overflows() {
        let p = PackedI8B::zeroed(64);
        int8_tile_gemm_16x16_packed_strided(&vec![0u8; 64], usize::MAX / 8, &p, &mut vec![0i32; 256]);
    }

    /// `1 << 60` is a multiple of 64 and `16·K` wraps to 0 on 64-bit, which
    /// would build an empty buffer claiming a huge K (codex P1, #418).
    #[test]
    #[cfg(target_pointer_width = "64")]
    #[should_panic(expected = "16·K overflows")]
    fn packed_refuses_a_k_whose_size_overflows() {
        let _ = PackedI8B::zeroed(1usize << 60);
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    #[should_panic(expected = "16·K overflows")]
    fn from_le_bytes_refuses_a_k_whose_size_overflows() {
        let _ = PackedI8B::from_le_bytes(&[], 1usize << 60);
    }

    /// With K = 0 and stride 0 every A bound holds, and `M·N` wraps to 0 on
    /// 64-bit, so an empty C would pass and the kernel would store into it
    /// (codex P1, #418).
    #[test]
    #[cfg(target_pointer_width = "64")]
    #[should_panic(expected = "M·N overflows")]
    fn multi_tile_refuses_an_output_size_that_overflows() {
        let bands = [PackedI8B::zeroed(0)];
        int8_gemm_packed_strided(&[], 0, &bands, &mut [], 1usize << 60);
    }

    #[test]
    #[should_panic(expected = "multiple of 64")]
    fn packed_refuses_a_k_off_the_tile_step() {
        let _ = PackedI8B::zeroed(96);
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

    /// A misaligned shape must panic in every profile and name the shape,
    /// even on a CPU without AMX (shape is checked first). Until 2026-10-11
    /// these were `debug_assert!`s and a release build indexed past the
    /// slices; run `cargo test --release` to see the difference.
    #[test]
    #[should_panic(expected = "M must be multiple of 16")]
    fn int8_gemm_amx_tiled_rejects_a_misaligned_m() {
        let (m, n, k) = (17, 16, 64);
        let a = vec![0u8; m * k];
        let b = vec![0i8; k * n];
        let mut c = vec![0i32; m * n];
        int8_gemm_amx_tiled(&a, &b, &mut c, m, n, k);
    }

    /// On a CPU without AMX an aligned call must panic with the AMX message
    /// instead of executing tile instructions.
    #[test]
    fn int8_gemm_amx_tiled_without_amx_panics_by_name() {
        if crate::hpc::amx_matmul::amx_available() {
            return;
        }
        let (m, n, k) = (16, 16, 64);
        let a = vec![0u8; m * k];
        let b = vec![0i8; k * n];
        let mut c = vec![0i32; m * n];
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| int8_gemm_amx_tiled(&a, &b, &mut c, m, n, k)));
        let msg = match r {
            Ok(()) => String::new(),
            Err(e) => e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default(),
        };
        assert!(msg.contains("AMX is not available"), "got: {msg:?}");
    }
}
