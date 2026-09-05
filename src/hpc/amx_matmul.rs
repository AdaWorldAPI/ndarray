//! AMX tile-based matrix multiplication via inline asm (stable Rust 1.94).
//!
//! TDPBUSD: 16×16 tile of u8×i8 → i32 = 256 MACs per instruction.
//! For the ThinkingEngine: builds the 4096² distance table from codebook centroids.
//!
//! Hardware confirmed: AMX-TILE + AMX-INT8 + AMX-BF16 (Sapphire Rapids+).
//! OS enabled: kernel 6.18.5, XCR0 bits 17+18 set.
//! Rust intrinsics: NIGHTLY ONLY (issue #126622).
//! This module: STABLE via inline asm!().
//!
//! Tile registers: 8 tiles, each 16 rows × 64 bytes = 1 KB.
//! For u8: 16×64 = 1024 values per tile.
//! For i32: 16×16 = 256 values per tile (result).
//!
//! One TDPBUSD: C[16×16 i32] += A[16×64 u8] × B[64×16 i8] = 16384 MACs.
//! Compared to VPDPBUSD (64 MACs): 256× more per instruction.

use std::arch::asm;

/// Check if AMX is available AND OS-enabled.
pub fn amx_available() -> bool {
    crate::simd_amx::amx_available()
}

/// AMX tile configuration (64 bytes, must be 64-byte aligned).
#[repr(C, align(64))]
pub struct TileConfig {
    pub data: [u8; 64],
}

impl TileConfig {
    /// Configure all three tiles for the int8/bf16 16×16 tile GEMM. Every tile
    /// is 16 rows × 64 colbytes; the shapes are identical so the same config
    /// serves C, the plain M×K operand, and the VNNI K×N operand. *Which*
    /// operand lands in *which* tile is decided by the kernel — see
    /// [`tile_dpbusd`] for the empirically-verified placement (VNNI K×N → tmm1,
    /// plain M×K → tmm2, result → tmm0).
    ///
    /// Tiles (shapes):
    ///   tmm0 = C (result):     16 rows × 64 bytes (16×16 i32)
    ///   tmm1 = VNNI K×N (vvvv): kb/4 rows × 64 bytes
    ///   tmm2 = plain M×K (rm):  16 rows × kb bytes
    pub fn for_dpbusd(k_bytes: u16) -> Self {
        let mut cfg = TileConfig { data: [0u8; 64] };
        cfg.data[0] = 1; // palette 1
                         // byte 1 = start_row = 0; bytes 2-15 reserved = 0 (already zeroed).

        // XTILECFG layout (Intel SDM, LDTILECFG memory operand):
        //   colsb[t] : u16 at offset 16 + 2*t   (bytes per tile row, ≤ 64)
        //   rows[t]  : u8  at offset 48 + t      (rows per tile,       ≤ 16)
        //
        // The previous version had these two regions SWAPPED — it wrote the
        // row counts into the colsb region (offsets 16/17/18) and the column
        // widths into the rows region (offsets 48/50/52). That produced
        // colsb[0] = 0x1010 = 4112 and rows[0] = 64, both out of range, so
        // LDTILECFG #GP-faulted (delivered as SIGSEGV) the instant the AMX
        // path actually executed. (It never had — every AMX test early-returns
        // when `amx_available()` is false, which it always was until the
        // arch_prctl syscall-number fix in `simd_amx.rs`.)
        let kb = k_bytes.min(64); // colsb ≤ 64 ⇒ each u16 high byte stays 0

        // Tile 0 (C): 16 rows × 64 colbytes (16 × i32 per row = 64 bytes).
        cfg.data[16] = 64; // colsb[0] low (u16 @ 16); high byte @17 stays 0
        cfg.data[48] = 16; // rows[0]      (u8  @ 48)

        // Tile 1 (B, VNNI K×N → VEX.vvvv): kb/4 rows × 64 colbytes. The kernel
        // loads the VNNI operand into tmm1, so tile 1 must carry the VNNI shape.
        // (Was the plain 16×kb shape — equal to this only at kb=64; backwards
        // for kb<64, which would mis-shape a tail kernel / external caller.)
        cfg.data[18] = 64; // colsb[1] low (u16 @ 18); high byte @19 stays 0
        cfg.data[49] = (kb / 4) as u8; // rows[1] (u8 @ 49)

        // Tile 2 (A, plain M×K → ModRM.rm): 16 rows × kb colbytes.
        cfg.data[20] = kb as u8; // colsb[2] low (u16 @ 20); high byte @21 stays 0
        cfg.data[50] = 16; // rows[2] (u8 @ 50)

        cfg
    }

    /// Configure all EIGHT tiles (tmm0-7) as 16 rows × 64 colbytes, for the 2×2
    /// register-blocked int8 kernel: tmm0-3 = the four C accumulators, tmm4-5 =
    /// two plain A row-blocks (rm/unsigned), tmm6-7 = two VNNI B col-blocks
    /// (vvvv/signed). Every tile is 16×64 so one config serves all roles. Same
    /// XTILECFG layout as [`Self::for_dpbusd`]: colsb[t] u16 @ 16+2t, rows[t]
    /// u8 @ 48+t.
    ///
    /// # Examples
    /// ```ignore
    /// use ndarray::hpc::amx_matmul::{tile_loadconfig, tile_release, TileConfig};
    /// // SAFETY: requires AMX (gate on `amx_available()`); all 8 tiles are 16×64.
    /// unsafe {
    ///     tile_loadconfig(&TileConfig::for_dpbusd_8());
    ///     // load A→tmm4/tmm5, B-VNNI→tmm6/tmm7, zero tmm0-3, then tile_dpbusd_2x2()
    ///     tile_release();
    /// }
    /// ```
    pub fn for_dpbusd_8() -> Self {
        let mut cfg = TileConfig { data: [0u8; 64] };
        cfg.data[0] = 1; // palette 1
        for t in 0..8 {
            cfg.data[16 + 2 * t] = 64; // colsb[t] = 64 (low byte; high stays 0)
            cfg.data[48 + t] = 16; // rows[t] = 16
        }
        cfg
    }
}

/// Load tile configuration via inline asm.
///
/// # Safety
/// Config must be valid and 64-byte aligned.
#[inline]
pub unsafe fn tile_loadconfig(config: &TileConfig) {
    asm!(
        "ldtilecfg [{cfg}]",
        cfg = in(reg) config.data.as_ptr(),
        options(nostack),
    );
}

/// Zero a tile register.
///
/// # Safety
/// Tiles must be configured first via tile_loadconfig.
#[inline]
pub unsafe fn tile_zero(tile: u8) {
    match tile {
        0 => asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xc0", options(nostack, nomem)),
        1 => asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xc8", options(nostack, nomem)),
        2 => asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xd0", options(nostack, nomem)),
        3 => asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xd8", options(nostack, nomem)),
        _ => {} // tiles 4-7: add when needed
    }
}

/// Release all tile registers.
///
/// # Safety
/// Must be called when done with tile operations.
#[inline]
pub unsafe fn tile_release() {
    asm!(".byte 0xc4, 0xe2, 0x78, 0x49, 0xc0", options(nostack, nomem));
}

/// Load tile from memory.
///
/// Encoding: `TILELOADD tmmN, [rcx + rax]` is VEX `C4 E2 7B 4B /r` with a
/// SIB byte selecting base = `rcx` (the data pointer) and index = `rax`
/// (the row stride in bytes, scale 1). The ModR/M `/r` field encodes the
/// destination tile via `reg = N` (3-bit tile index). Per-tile bytes:
///
///   tmm0:  C4 E2 7B 4B **04** 01
///   tmm1:  C4 E2 7B 4B **0C** 01
///   tmm2:  C4 E2 7B 4B **14** 01
///
/// `04 | (N << 3)` gives the ModR/M byte; SIB `01` = (scale=1, index=rax,
/// base=rcx) is the same across tiles. The previous SIB `08` had base and
/// index swapped (base=rax, index=rcx), so the tile engine dereferenced the
/// *stride value* (~64) as the start address and SIGSEGV'd the moment the
/// AMX path went live. tmm0 was added when codex flagged the accumulator-
/// preservation bug on PR #184 (`tile_zero(0)` + `tile_store(0, c)`
/// discarded any pre-existing C values — the fix is `tile_load(0, c)`
/// instead of `tile_zero(0)` so TDPBUSD/TDPBF16PS truly accumulate as
/// the documented `C += A·B` semantics promise).
///
/// # Safety
/// Pointer must be valid, stride must match tile config.
#[inline]
pub unsafe fn tile_load(tile: u8, ptr: *const u8, stride: usize) {
    match tile {
        0 => asm!(
            ".byte 0xc4, 0xe2, 0x7b, 0x4b, 0x04, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        1 => asm!(
            ".byte 0xc4, 0xe2, 0x7b, 0x4b, 0x0c, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        2 => asm!(
            ".byte 0xc4, 0xe2, 0x7b, 0x4b, 0x14, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        // ModRM = 0x04 | (tile << 3); SIB 0x01 unchanged. tmm3-7 added for the
        // 2×2 register-blocked kernel (4 C accumulators + 2 A + 2 B tiles).
        3 => asm!(
            ".byte 0xc4, 0xe2, 0x7b, 0x4b, 0x1c, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        4 => asm!(
            ".byte 0xc4, 0xe2, 0x7b, 0x4b, 0x24, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        5 => asm!(
            ".byte 0xc4, 0xe2, 0x7b, 0x4b, 0x2c, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        6 => asm!(
            ".byte 0xc4, 0xe2, 0x7b, 0x4b, 0x34, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        7 => asm!(
            ".byte 0xc4, 0xe2, 0x7b, 0x4b, 0x3c, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        _ => {}
    }
}

/// Store tile to memory. ModRM = 0x04 | (tile << 3); SIB 0x01 = base=rcx (ptr),
/// index=rax (stride), scale=1. tmm0-3 are the four C accumulators of the 2×2
/// register-blocked kernel.
///
/// # Safety
/// Pointer must be valid and writable, stride must match.
#[inline]
pub unsafe fn tile_store(tile: u8, ptr: *mut u8, stride: usize) {
    match tile {
        0 => asm!(
            ".byte 0xc4, 0xe2, 0x7a, 0x4b, 0x04, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        1 => asm!(
            ".byte 0xc4, 0xe2, 0x7a, 0x4b, 0x0c, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        2 => asm!(
            ".byte 0xc4, 0xe2, 0x7a, 0x4b, 0x14, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        3 => asm!(
            ".byte 0xc4, 0xe2, 0x7a, 0x4b, 0x1c, 0x01",
            in("rcx") ptr,
            in("rax") stride,
            options(nostack),
        ),
        _ => {}
    }
}

/// TDPBUSD: C(i32, tmm0) += unsigned ⊗ signed → i32. 16×16 output, 64 products
/// per element = 16384 MACs in ONE instruction.
///
/// **Empirical operand convention** (measured on Emerald Rapids — the naive
/// SDM reading is mirrored on BOTH axes, so do not infer roles from the
/// mnemonic):
///   * INDEX: `dst[m][n] = Σ_k tmm2(ModRM.rm)[m][k] · tmm1(VEX.vvvv)[k][n]`.
///     The plain **M×K** operand goes in **tmm2**; the VNNI-packed **K×N**
///     operand goes in **tmm1** (the opposite of what the SDM operand order
///     suggests).
///   * SIGN: **tmm2 (rm) is the UNSIGNED operand, tmm1 (vvvv) is SIGNED.**
///     Verified by sweeping all four `TDPB**D` opcodes against sign-sensitive
///     constant inputs — only `0x71` gives rm=unsigned, vvvv=signed.
///
/// So `int8_tile_gemm::amx_path` loads `A(u8) → tmm2` and `B_vnni(i8) → tmm1`.
///
/// Encoding: VEX.128.66.0F38.W0 5E /r. byte2 `0x71` = W0.vvvv=1110(tmm1).L0.
/// pp=01(66); ModRM `0xC2` = mod=11, reg=000 (tmm0 dst), rm=010 (tmm2). The
/// three tile operands MUST be distinct (tmm0/tmm1/tmm2 are). Two earlier
/// bugs in this file: byte2 `0x73` (F2 = signed×signed, wrong sign variant)
/// and ModRM `0xC1` (rm=tmm1, aliasing the two sources → same-tile #UD, the
/// first SIGILL the live AMX path hit).
///
/// # Safety
/// Tiles must be loaded with valid data and AMX OS-enabled (`amx_available()`).
#[inline]
pub unsafe fn tile_dpbusd() {
    asm!(".byte 0xc4, 0xe2, 0x71, 0x5e, 0xc2", options(nostack, nomem));
}

/// 2×2 register-blocked TDPBUSD — four accumulations in one call:
/// ```text
///   C00(tmm0) += A0(tmm4) ⊗ B0(tmm6)    C01(tmm1) += A0(tmm4) ⊗ B1(tmm7)
///   C10(tmm2) += A1(tmm5) ⊗ B0(tmm6)    C11(tmm3) += A1(tmm5) ⊗ B1(tmm7)
/// ```
/// Same operand convention as [`tile_dpbusd`]: the plain M×K operand is rm
/// (A, unsigned), the VNNI K×N operand is vvvv (B, signed). Reusing the two A
/// and two B tile loads across four products halves the load bytes per MAC —
/// the lever for this memory-bandwidth-bound kernel.
///
/// Encodings (VEX `C4 E2 <byte2> 5E <modrm>`, byte2 = ((~vvvv & 0xF)<<3)|0x01,
/// modrm = 0xC0 | dst<<3 | rm):
///   C00 dst0 rm4 vvvv6 → C4 E2 49 5E C4   C01 dst1 rm4 vvvv7 → C4 E2 41 5E CC
///   C10 dst2 rm5 vvvv6 → C4 E2 49 5E D5   C11 dst3 rm5 vvvv7 → C4 E2 41 5E DD
/// All eight operand tiles (0/1/2/3 dst, 4/5 A, 6/7 B) are distinct → no #UD.
///
/// # Examples
/// ```ignore
/// use ndarray::hpc::amx_matmul::*;
/// // SAFETY: requires AMX; full 32×32 register-blocked tile contract.
/// unsafe {
///     tile_loadconfig(&TileConfig::for_dpbusd_8());
///     tile_zero(0); tile_zero(1); tile_zero(2); tile_zero(3); // C accumulators
///     tile_load(4, a0_ptr, k); tile_load(5, a1_ptr, k);       // A rows (rm)
///     tile_load(6, b0_vnni, 64); tile_load(7, b1_vnni, 64);   // B cols (vvvv)
///     tile_dpbusd_2x2();                                       // 4 TDPBUSDs
///     tile_store(0, c00, n * 4); /* … tmm1/2/3 → other quadrants … */
///     tile_release();
/// }
/// ```
///
/// # Safety
/// Tiles 0-7 configured (`TileConfig::for_dpbusd_8`) and 4/5/6/7 loaded.
#[inline]
pub unsafe fn tile_dpbusd_2x2() {
    asm!(
        ".byte 0xc4, 0xe2, 0x49, 0x5e, 0xc4", // C00 = A0·B0
        ".byte 0xc4, 0xe2, 0x41, 0x5e, 0xcc", // C01 = A0·B1
        ".byte 0xc4, 0xe2, 0x49, 0x5e, 0xd5", // C10 = A1·B0
        ".byte 0xc4, 0xe2, 0x41, 0x5e, 0xdd", // C11 = A1·B1
        options(nostack, nomem)
    );
}

/// TDPBF16PS: C += A(bf16) × B(bf16_vnni) → f32.
/// tmm0 += tmm1 × tmm2.
///
/// 16×16 output accumulator (f32), 32 bf16 values per A row × 32 bf16 values
/// per B row in VNNI layout = 512 mul-adds in one instruction.
///
/// Encoding (analogous to TDPBUSD: opcode 5E→5C, pp 66→F3):
///   TDPBUSD   tmm0, tmm1, tmm2 → C4 E2 71 5E C2
///   TDPBF16PS tmm0, tmm1, tmm2 → C4 E2 72 5C C2
/// ModRM 0xC2 = mod=11, reg=000 (tmm0 dst), rm=010 (tmm2 src2); VEX.vvvv =
/// tmm1 (src1). The three tile operands MUST be distinct — the prior ModRM
/// 0xC1 set rm=tmm1, making src1==src2 → same-tile #UD (SIGILL) once the AMX
/// path actually executed.
///
/// Tile shapes at K=32, M=N=16 (identical to TDPBUSD max at K_bytes=64):
///   tmm0 (C): 16×16 f32   (16 rows × 64 bytes)
///   tmm1 (A): 16×32 bf16  (16 rows × 64 bytes, plain row-major)
///   tmm2 (B): 16×16 bf16 pairs (K/2=16 rows × 64 bytes, VNNI pairs)
///
/// # Safety
/// Tiles 0/1/2 must be configured via `tile_loadconfig(&TileConfig::for_dpbusd(64))`
/// and loaded with valid data; AMX must be OS-enabled (check `amx_available()`).
#[inline]
pub unsafe fn tile_dpbf16ps() {
    asm!(".byte 0xc4, 0xe2, 0x72, 0x5c, 0xc2", options(nostack, nomem));
}

/// Pack B[K, N] bf16 row-major into K/2 × (N*2) VNNI pairs (in-place target).
/// Output layout required by TDPBF16PS tile 2:
///   dst[i, 2j]   = src[2i,   j]
///   dst[i, 2j+1] = src[2i+1, j]
///
/// For N=16 (AMX tile width), each output "row" holds 16 bf16 pairs = 64 bytes.
/// K must be even.
#[inline]
pub fn vnni_pack_bf16(src: &[u16], dst: &mut [u16], k: usize, n: usize) {
    debug_assert_eq!(src.len(), k * n);
    debug_assert_eq!(dst.len(), k * n);
    debug_assert_eq!(k % 2, 0, "K must be even for VNNI BF16 pairs");
    for i in 0..(k / 2) {
        let dst_row = i * n * 2;
        for j in 0..n {
            dst[dst_row + 2 * j] = src[(2 * i) * n + j];
            dst[dst_row + 2 * j + 1] = src[(2 * i + 1) * n + j];
        }
    }
}

/// Pack B[K, N] i8 row-major into K/4 × (N*4) VNNI quads for `TDPBUSD`.
///
/// Output layout required by `TDPBUSD` tile 2 (16 rows × 64 bytes):
///   dst[kb*N*4 + j*4 + p] = src[(4*kb + p) * N + j]
///
/// For N=16 (AMX tile width), each output "row" holds 16 i8 quads = 64
/// bytes (matches the 64-byte tile row width). K must be a multiple of
/// 4. The same layout is used for `u8` operands (just bit-cast through
/// — VNNI doesn't care about sign at the packing layer; sign
/// interpretation happens inside TDPBUSD which treats A as u8 and B
/// as i8 for the multiply).
#[inline]
pub fn vnni_pack_i8(src: &[i8], dst: &mut [i8], k: usize, n: usize) {
    debug_assert_eq!(src.len(), k * n);
    debug_assert_eq!(dst.len(), k * n);
    debug_assert_eq!(k % 4, 0, "K must be multiple of 4 for VNNI INT8 quads");
    for kb in 0..(k / 4) {
        let dst_row = kb * n * 4;
        for j in 0..n {
            for p in 0..4 {
                dst[dst_row + j * 4 + p] = src[(4 * kb + p) * n + j];
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public ndarray-typed matmul API (sprint A4 / Burn parity item 6)
// ═══════════════════════════════════════════════════════════════════════════
//
// Three entry points operating on `ArrayView2` / `ArrayViewMut2`:
//   matmul_f32         — f32 × f32 → f32, EXACT: `backend::native::gemm_f32`
//                        (matrixmultiply). Never AMX — measured slower and
//                        less accurate than CPU f32 at every size (see
//                        `matmul_f32_amx_split`'s table).
//   matmul_f32_amx_split — three-pass BF16 split on AMX, f32-grade but ~6x
//                        slower than `matmul_f32`. Kept for re-measurement.
//   matmul_f32_bf16_fast — ONE BF16 pass on AMX (~1e-3 rel). Still slower
//                        than `matmul_f32`. Opt in by name only.
//   matmul_bf16_to_f32 — BF16 × BF16 → f32 (AMX TDPBF16PS or `bf16_gemm_f32`)
//   matmul_i8_to_i32   — i8 × i8 → i32   (AMX TDPBUSD or scalar `int8_gemm_i32`)
//
// Output constraint: row-stride-1, contiguous along columns. Inputs may be
// strided (e.g. `view.slice(s![.., ..;2])`). Strided inputs are repacked
// into contiguous staging buffers before the kernel runs.

use crate::hpc::quantized::{bf16_gemm_f32, BF16};
use crate::{ArrayView2, ArrayViewMut2};

/// Errors returned by the public AMX matmul API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatmulError {
    /// Shapes don't satisfy `lhs:(M,K), rhs:(K,N), out:(M,N)`.
    ShapeMismatch {
        /// Shape of the LHS view, `(rows, cols)`.
        lhs: (usize, usize),
        /// Shape of the RHS view, `(rows, cols)`.
        rhs: (usize, usize),
        /// Shape of the output view, `(rows, cols)`.
        out: (usize, usize),
    },
    /// AMX hardware/OS-state not available **and** caller asked for the
    /// strict AMX path. The default entry points fall back to the scalar
    /// kernels and never return this error.
    AmxUnavailable,
    /// Output tensor is not row-contiguous (column stride ≠ 1).
    NonContiguousOutput,
}

impl std::fmt::Display for MatmulError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulError::ShapeMismatch { lhs, rhs, out } => write!(
                f,
                "shape mismatch: lhs={:?} rhs={:?} out={:?}; expected lhs:(M,K), rhs:(K,N), out:(M,N)",
                lhs, rhs, out
            ),
            MatmulError::AmxUnavailable => f.write_str("AMX not available on this host"),
            MatmulError::NonContiguousOutput => f.write_str("output must be row-contiguous (col stride = 1)"),
        }
    }
}

impl std::error::Error for MatmulError {}

// ── Internal helpers ───────────────────────────────────────────────────────

/// Validate `lhs:(M,K) × rhs:(K,N) → out:(M,N)` and that `out` is row-contiguous.
fn check_shapes<A, B, C>(
    lhs: &ArrayView2<'_, A>, rhs: &ArrayView2<'_, B>, out: &ArrayViewMut2<'_, C>,
) -> Result<(usize, usize, usize), MatmulError> {
    let (m, k) = lhs.dim();
    let (kr, n) = rhs.dim();
    let (mo, no) = out.dim();
    if k != kr || m != mo || n != no {
        return Err(MatmulError::ShapeMismatch {
            lhs: (m, k),
            rhs: (kr, n),
            out: (mo, no),
        });
    }
    // Output must be row-stride-1 (writes are linear per row).
    let strides = out.strides();
    if strides[1] != 1 {
        return Err(MatmulError::NonContiguousOutput);
    }
    Ok((m, n, k))
}

/// Copy a possibly-strided 2-D view into a contiguous row-major Vec.
fn pack_contig<A: Copy>(view: &ArrayView2<'_, A>) -> Vec<A> {
    let (rows, cols) = view.dim();
    let mut buf = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            buf.push(view[[r, c]]);
        }
    }
    buf
}

/// Write a contiguous row-major buffer back into a 2-D mutable view.
fn write_contig<A: Copy>(view: &mut ArrayViewMut2<'_, A>, src: &[A]) {
    let (rows, cols) = view.dim();
    debug_assert_eq!(src.len(), rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            view[[r, c]] = src[r * cols + c];
        }
    }
}

// ── BF16 → f32 ─────────────────────────────────────────────────────────────

/// Matrix multiply BF16 × BF16 → f32: `out = lhs · rhs`.
///
/// On AMX hardware (Sapphire Rapids+, Granite Rapids), 16×16-aligned tiles
/// dispatch to [`crate::hpc::bf16_tile_gemm::bf16_tile_gemm_16x16`] which
/// emits `TDPBF16PS` via the asm-byte path in `simd_amx.rs` — 256
/// BF16×BF16 multiply-accumulates per instruction (16×16×32 = 8 192 FLOPs)
/// into f32 accumulator tiles. M/N/K tail blocks (when any dim isn't
/// 16/16/32-aligned) fall through to the validated scalar
/// [`crate::hpc::quantized::bf16_gemm_f32`] reference.
///
/// On non-AMX hosts the entire matmul goes through `bf16_gemm_f32`.
///
/// `out` must be row-contiguous (column stride = 1); inputs may be strided.
pub fn matmul_bf16_to_f32(
    lhs: ArrayView2<'_, BF16>, rhs: ArrayView2<'_, BF16>, mut out: ArrayViewMut2<'_, f32>,
) -> Result<(), MatmulError> {
    let (m, n, k) = check_shapes(&lhs, &rhs, &out)?;

    let a = pack_contig(&lhs);
    let b = pack_contig(&rhs);
    let mut c = vec![0.0f32; m * n];

    bf16_gemm_dispatch(&a, &b, &mut c, m, n, k);

    write_contig(&mut out, &c);
    Ok(())
}

/// BF16 × BF16 → f32 GEMM with three-tier dispatch (AMX → VDPBF16PS → scalar).
///
/// Inputs are packed row-major (`a` is M × K, `b` is K × N). Output `c`
/// is M × N row-major and is overwritten (not accumulated).
///
/// Tier selection:
///
/// 1. **AMX `TDPBF16PS`** (Sapphire Rapids+, Granite Rapids) when
///    `amx_available()` is true AND shapes are 16/16/32-aligned.
///    Dispatches through
///    [`crate::hpc::bf16_tile_gemm::bf16_tile_gemm_16x16`] →
///    `simd_amx::tile_dpbf16ps` via asm-byte (`TDPBF16PS` intrinsic is
///    nightly-only on Rust 1.95). 8 192 BF16×BF16 multiplies + 256 f32
///    accumulates per instruction.
/// 2. **`VDPBF16PS`** (Cooper Lake, Cascade Lake AVX-512BF16, Zen 4+)
///    when `is_x86_feature_detected!("avx512bf16")` is true. The
///    intrinsic `_mm512_dpbf16_ps` is stable on Rust 1.95 (no asm-byte
///    needed). Per instruction: 32 BF16×BF16 multiplies + 16 f32
///    accumulates, single-rounded. Handles arbitrary shapes — M / N
///    tails fall through the per-iteration j-block trimming; K-tail
///    (odd K) is handled with a final scalar pair.
/// 3. **Scalar reference** [`bf16_gemm_f32`] for hosts without either
///    extension or for shapes the AMX arm rejects.
///
/// The per-tier dispatch table comes from PR #180's BF16 GEMM column.
fn bf16_gemm_dispatch(a: &[BF16], b: &[BF16], c: &mut [f32], m: usize, n: usize, k: usize) {
    if amx_available() && m % 16 == 0 && n % 16 == 0 && k % 32 == 0 {
        // SAFETY: BF16 is `#[repr(transparent)] struct BF16(pub u16)`
        // (per `hpc::quantized::BF16`). Reinterpreting `&[BF16]` as
        // `&[u16]` is bit-pattern preserving.
        let a_u16: &[u16] = unsafe { core::slice::from_raw_parts(a.as_ptr() as *const u16, a.len()) };

        // B is packed row-major K × N; the 16×16 tile kernel wants a
        // K × 16 contiguous sub-block. Extract per (j_tile) into a
        // scratch buffer once and reuse across i_tile.
        let mut b_tile = vec![0u16; k * 16];
        let mut tile_c = vec![0.0f32; 256];

        for j_tile in (0..n).step_by(16) {
            // Pack b[0..k, j_tile..j_tile+16] into row-major 16-wide K-rows.
            for kk in 0..k {
                let row = kk * n + j_tile;
                for jj in 0..16 {
                    b_tile[kk * 16 + jj] = b[row + jj].0;
                }
            }
            for i_tile in (0..m).step_by(16) {
                // A_tile = a[i_tile..i_tile+16, 0..k] — already contiguous
                // since `a` is packed row-major M × K.
                let a_tile = &a_u16[i_tile * k..(i_tile + 16) * k];
                tile_c.fill(0.0);
                crate::hpc::bf16_tile_gemm::bf16_tile_gemm_16x16(a_tile, &b_tile, &mut tile_c, k);
                // Write tile_c (16 × 16, row-major) into c (M × N, row-major).
                for ii in 0..16 {
                    let dst_off = (i_tile + ii) * n + j_tile;
                    c[dst_off..dst_off + 16].copy_from_slice(&tile_c[ii * 16..(ii + 1) * 16]);
                }
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512bf16") {
            // SAFETY: feature-detected at runtime; the kernel is
            // `#[target_feature(enable = "avx512bf16,avx512f")]`.
            unsafe {
                bf16_gemm_vdpbf16ps(a, b, c, m, n, k);
            }
            return;
        }
    }

    bf16_gemm_f32(a, b, c, m, n, k, 1.0, 0.0);
}

/// AVX-512BF16 BF16 GEMM using `_mm512_dpbf16_ps` (`VDPBF16PS`).
///
/// One VDPBF16PS instruction: 16 f32 accumulator lanes each receive
/// `acc[j] += a.bf16[2j] * b.bf16[2j] + a.bf16[2j+1] * b.bf16[2j+1]`,
/// single-rounded. The kernel maps the 16 output lanes to a row of 16
/// j-columns of C[i, ·], with one i row processed at a time and a K-pair
/// inner loop accumulating into the same 16 f32 lanes across iterations.
///
/// B-column packing: VDPBF16PS wants the 32 B BF16s per call laid out
/// as 16 lane-pairs (lane j contains `B[2k_pair, j_base+j]` followed by
/// `B[2k_pair+1, j_base+j]`, packed into one u32). We pre-pack B for
/// the current j-block into `b_col_pairs[k_pair * 16 + j] = u32` once
/// per j_block and reuse across all i — amortizes the gather cost.
///
/// K-tail (when K is odd) is handled with a final scalar BF16 multiply
/// per output cell; N-tail (when the j-block has < 16 valid columns)
/// is handled by trimming the store after the VDPBF16PS chain.
///
/// # Safety
/// Caller must have feature-detected `avx512bf16` at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bf16,avx512f")]
unsafe fn bf16_gemm_vdpbf16ps(a: &[BF16], b: &[BF16], c: &mut [f32], m: usize, n: usize, k: usize) {
    use core::arch::x86_64::{
        __m512bh, __m512i, _mm512_dpbf16_ps, _mm512_loadu_si512, _mm512_set1_epi32, _mm512_setzero_ps, _mm512_storeu_ps,
    };

    let k_pairs = k / 2;
    let k_tail = k % 2;

    // SAFETY: BF16 is repr(transparent) over u16.
    let a_u16: &[u16] = core::slice::from_raw_parts(a.as_ptr() as *const u16, a.len());
    let b_u16: &[u16] = core::slice::from_raw_parts(b.as_ptr() as *const u16, b.len());

    // Pre-pack scratch: 16 u32 lanes per k_pair, holding (b_lo | b_hi << 16).
    let mut b_col_pairs = vec![0u32; k_pairs.max(1) * 16];
    // Scratch for the 16-wide store + N-tail trim.
    let mut out_buf = [0.0f32; 16];

    for j_base in (0..n).step_by(16) {
        let j_count = 16.min(n - j_base);

        // Pack B columns [j_base..j_base+j_count] in pair-interleaved layout.
        // For lanes j >= j_count (the N-tail of this j_block), pad with 0 —
        // they're not stored back, but the VDPBF16PS still touches them.
        for k_pair in 0..k_pairs {
            let row_lo = 2 * k_pair * n;
            let row_hi = (2 * k_pair + 1) * n;
            for jj in 0..j_count {
                let b_lo = b_u16[row_lo + j_base + jj] as u32;
                let b_hi = b_u16[row_hi + j_base + jj] as u32;
                b_col_pairs[k_pair * 16 + jj] = (b_hi << 16) | b_lo;
            }
            for jj in j_count..16 {
                b_col_pairs[k_pair * 16 + jj] = 0;
            }
        }

        for i in 0..m {
            let mut acc = _mm512_setzero_ps();
            let a_row_off = i * k;
            for k_pair in 0..k_pairs {
                // Broadcast A[i, 2k_pair..2k_pair+2] as the (BF16 lo, BF16 hi)
                // pair across all 16 lanes.
                let a_lo = a_u16[a_row_off + 2 * k_pair] as u32;
                let a_hi = a_u16[a_row_off + 2 * k_pair + 1] as u32;
                let pair = (a_hi << 16) | a_lo;
                let a_bh: __m512bh = core::mem::transmute(_mm512_set1_epi32(pair as i32));
                let b_bh: __m512bh =
                    core::mem::transmute(_mm512_loadu_si512(b_col_pairs.as_ptr().add(k_pair * 16) as *const __m512i));
                acc = _mm512_dpbf16_ps(acc, a_bh, b_bh);
            }
            _mm512_storeu_ps(out_buf.as_mut_ptr(), acc);

            // K-tail: one extra scalar BF16 multiply for k = k_pairs*2.
            if k_tail == 1 {
                let a_last_f32 = BF16(a_u16[a_row_off + k - 1]).to_f32();
                let tail_row = (k - 1) * n;
                for jj in 0..j_count {
                    let b_last_f32 = BF16(b_u16[tail_row + j_base + jj]).to_f32();
                    out_buf[jj] += a_last_f32 * b_last_f32;
                }
            }

            // Store the j_count valid lanes (drops N-tail padding lanes).
            let dst_off = i * n + j_base;
            c[dst_off..dst_off + j_count].copy_from_slice(&out_buf[..j_count]);
        }
    }
}

// ── f32 → f32 (BF16 compute on AMX) ────────────────────────────────────────

/// Matrix multiply f32 × f32 → f32: `out = lhs · rhs`.
///
/// On AMX hosts the inputs are converted to BF16 and computed via
/// `TDPBF16PS` (≤ ~1% relative error on well-scaled inputs). Without AMX,
/// computation runs in pure f32 and is bit-stable.
///
/// `out` must be row-contiguous; inputs may be strided.
/// Splits `v` into a BF16 head and a BF16 tail such that
/// `head.to_f32() + tail.to_f32()` reproduces `v` to ~17 significand bits.
///
/// Both halves are exactly BF16-representable, so a later `from_f32_rounded`
/// on either is the identity — that is what keeps the three-pass product at
/// one rounding per operand rather than two.
fn split_bf16(v: &[f32]) -> (Vec<BF16>, Vec<BF16>) {
    let mut hi = Vec::with_capacity(v.len());
    let mut lo = Vec::with_capacity(v.len());
    for &x in v {
        let h = BF16::from_f32_rounded(x);
        hi.push(h);
        lo.push(BF16::from_f32_rounded(x - h.to_f32()));
    }
    (hi, lo)
}

/// Single-BF16-pass `f32` matmul: one third of [`matmul_f32`]'s tile work,
/// at BF16 precision (~1e-3 relative error, measured).
///
/// Use only where BF16-grade accuracy has been measured as acceptable for the
/// workload. [`matmul_f32`] is the default because its name promises f32 and
/// a silent 1e-3 is outside what f32 callers — linear algebra especially —
/// tolerate: routing burn's `burn-ndarray` through this path fails 50 of its
/// linalg tests (qr / lu / svd / det) that pass on the three-pass version.
pub fn matmul_f32_bf16_fast(
    lhs: ArrayView2<'_, f32>, rhs: ArrayView2<'_, f32>, mut out: ArrayViewMut2<'_, f32>,
) -> Result<(), MatmulError> {
    let (m, n, k) = check_shapes(&lhs, &rhs, &out)?;
    let a_f32 = pack_contig(&lhs);
    let b_f32 = pack_contig(&rhs);
    let mut c = vec![0.0f32; m * n];

    if amx_available() {
        let a_bf16: Vec<BF16> = a_f32.iter().map(|&v| BF16::from_f32_rounded(v)).collect();
        let b_bf16: Vec<BF16> = b_f32.iter().map(|&v| BF16::from_f32_rounded(v)).collect();
        bf16_gemm_dispatch(&a_bf16, &b_bf16, &mut c, m, n, k);
    } else {
        for i in 0..m {
            for p in 0..k {
                let av = a_f32[i * k + p];
                for j in 0..n {
                    c[i * n + j] += av * b_f32[p * n + j];
                }
            }
        }
    }

    write_contig(&mut out, &c);
    Ok(())
}

/// Exact f32 matmul. Never routes through AMX — measured slower AND less
/// accurate than the CPU f32 GEMM at every size tried (see the table in
/// [`matmul_f32_amx_split`]).
///
/// Delegates to `backend::native::gemm_f32` (matrixmultiply), which is
/// f32 end to end: no BF16 downcast, no tile emulation.
pub fn matmul_f32(
    lhs: ArrayView2<'_, f32>, rhs: ArrayView2<'_, f32>, mut out: ArrayViewMut2<'_, f32>,
) -> Result<(), MatmulError> {
    let (m, n, k) = check_shapes(&lhs, &rhs, &out)?;
    let a_f32 = pack_contig(&lhs);
    let b_f32 = pack_contig(&rhs);
    let mut c = vec![0.0f32; m * n];
    crate::backend::native::gemm_f32(m, n, k, 1.0, &a_f32, k, &b_f32, n, 0.0, &mut c, n);
    write_contig(&mut out, &c);
    Ok(())
}

/// AMX three-pass BF16-split matmul: f32-grade accuracy on the tile unit.
///
/// KEPT FOR THE RECORD, NOT RECOMMENDED. Benchmarked against the exact CPU
/// paths on a Xeon with AMX + AVX-512 (`gemm_paths_bench`), square f32:
///
/// | size  | native gemm_f32 | F32x16 sgemm_blocked | this (3-pass) | 1-pass BF16 |
/// |-------|-----------------|----------------------|---------------|-------------|
/// | 256   | 0.42ms 2.9e-7   | 0.37ms 2.9e-7        | 2.78ms 1.4e-6 | 0.77ms 3.7e-4 |
/// | 512   | 3.51ms 1.2e-6   | 3.22ms 1.2e-6        | 20.9ms 1.4e-6 | 6.37ms 2.2e-4 |
/// | 1024  | 28.8ms 1.4e-6   | 27.3ms 1.4e-6        | 159ms  1.5e-6 | 45.9ms 1.6e-4 |
///
/// AMX loses on BOTH axes at every size — the packing, the BF16 conversion
/// and (here) three passes cost more than the tile unit saves, and even the
/// single pass is slower than plain f32 while being ~1000x less accurate.
/// Exposed so the measurement is reproducible and so a future host (or a
/// real f32 tile op) can be re-measured against it, not because a caller
/// should reach for it.
pub fn matmul_f32_amx_split(
    lhs: ArrayView2<'_, f32>, rhs: ArrayView2<'_, f32>, mut out: ArrayViewMut2<'_, f32>,
) -> Result<(), MatmulError> {
    let (m, n, k) = check_shapes(&lhs, &rhs, &out)?;

    let a_f32 = pack_contig(&lhs);
    let b_f32 = pack_contig(&rhs);
    let mut c = vec![0.0f32; m * n];

    if amx_available() {
        // AMX has no f32 tile op, so f32 operands must reach the tile
        // kernel as BF16. A single RNE downcast discards 16 of f32's 24
        // significand bits — measured ~1e-3 relative error, which is
        // BF16 grade, not f32 grade, and this function's name promises
        // f32. So split each operand into a BF16 head plus a BF16 tail:
        //
        //     a = a_hi + a_lo,  b = b_hi + b_lo   (all four exactly BF16)
        //     a·b = a_hi·b_hi + a_hi·b_lo + a_lo·b_hi + a_lo·b_lo
        //
        // and drop the last term: it is O(2^-18) relative, below the f32
        // significand, so three passes suffice. Each pass still rounds
        // exactly once (its inputs are already BF16-representable, so the
        // downcast inside the kernel is the identity) and accumulates in
        // f32 — the one-rounding discipline is preserved, and the bits it
        // used to discard are now carried by the tail.
        //
        // Measured on Xeon w/ AMX, vs an f32 reference:
        //     16x32x16   1-pass 1.09e-3  ->  3-pass 2.6e-6
        //     32x64x32   1-pass 8.40e-4  ->  3-pass 2.1e-6
        // i.e. ~400x, at 3x the tile work. Callers that want the single
        // BF16 pass ask for it by name: `matmul_f32_bf16_fast`.
        let (a_hi, a_lo) = split_bf16(&a_f32);
        let (b_hi, b_lo) = split_bf16(&b_f32);

        let mut scratch = vec![0.0f32; m * n];
        bf16_gemm_dispatch(&a_hi, &b_hi, &mut c, m, n, k);
        bf16_gemm_dispatch(&a_hi, &b_lo, &mut scratch, m, n, k);
        for (dst, add) in c.iter_mut().zip(scratch.iter()) {
            *dst += *add;
        }
        bf16_gemm_dispatch(&a_lo, &b_hi, &mut scratch, m, n, k);
        for (dst, add) in c.iter_mut().zip(scratch.iter()) {
            *dst += *add;
        }
    } else {
        // Pure f32 reference path.
        for i in 0..m {
            for p in 0..k {
                let av = a_f32[i * k + p];
                for j in 0..n {
                    c[i * n + j] += av * b_f32[p * n + j];
                }
            }
        }
    }

    write_contig(&mut out, &c);
    Ok(())
}

// ── i8 → i32 ───────────────────────────────────────────────────────────────

/// Matrix multiply i8 × i8 → i32: `out = lhs · rhs`.
///
/// On AMX hosts with 16/16/64-aligned shapes uses `TDPBUSD` via the
/// 16×16 tile kernel in [`crate::hpc::int8_tile_gemm::int8_tile_gemm_16x16`]
/// — 16 384 MACs per instruction. Mis-aligned shapes (or non-AMX hosts)
/// fall back to the scalar i8×i8 → i32 reference.
///
/// Note: `TDPBUSD` natively expects unsigned-by-signed (u8 × i8). For
/// the signed-by-signed surface required here, the LHS is shifted into
/// the unsigned domain (i8 + 128 → u8) and the bias `128 · sum(B[:, j]
/// over k)` is subtracted from the accumulator. The public result is
/// bit-identical to the scalar reference because all arithmetic stays
/// in i32 (no float rounding).
///
/// `out` must be row-contiguous; inputs may be strided.
pub fn matmul_i8_to_i32(
    lhs: ArrayView2<'_, i8>, rhs: ArrayView2<'_, i8>, mut out: ArrayViewMut2<'_, i32>,
) -> Result<(), MatmulError> {
    let (m, n, k) = check_shapes(&lhs, &rhs, &out)?;

    let a_i8 = pack_contig(&lhs);
    let b_i8 = pack_contig(&rhs);
    let mut c = vec![0i32; m * n];

    if amx_available() && m % 16 == 0 && n % 16 == 0 && k % 64 == 0 {
        // Tier 1 — AMX TDPBUSD tile path: shift LHS i8 → u8 (+128),
        // delegate to the shared int8_gemm_amx_tiled helper, subtract
        // the sign-shift bias.
        let a_u8: Vec<u8> = a_i8.iter().map(|&v| (v as i32 + 128) as u8).collect();
        crate::hpc::int8_tile_gemm::int8_gemm_amx_tiled(&a_u8, &b_i8, &mut c, m, n, k);
        subtract_i8_to_u8_bias(&mut c, &b_i8, m, n, k);
    } else if cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avx512vnni") {
        // Tier 2 — AVX-512 VPDPBUSD zmm: 64 MACs per instruction, no
        // shape-alignment requirement (M/N/K all handled via per-block
        // trim and scalar K-tail). Same sign-shift bias trick as AMX.
        let a_u8: Vec<u8> = a_i8.iter().map(|&v| (v as i32 + 128) as u8).collect();
        // SAFETY: runtime feature-detected avx512vnni above.
        unsafe {
            crate::hpc::int8_tile_gemm::int8_gemm_vpdpbusd_zmm(&a_u8, &b_i8, &mut c, m, n, k);
        }
        subtract_i8_to_u8_bias(&mut c, &b_i8, m, n, k);
    } else if cfg!(target_arch = "x86_64") && std::is_x86_feature_detected!("avxvnni") {
        // Tier 3 — AVX-VNNI ymm VPDPBUSD: 32 MACs per instruction.
        // Arrow Lake, Meteor Lake U, Alder Lake silicon that has
        // AVX-VNNI but dropped AVX-512. Same sign-shift bias trick.
        let a_u8: Vec<u8> = a_i8.iter().map(|&v| (v as i32 + 128) as u8).collect();
        // SAFETY: runtime feature-detected avxvnni above.
        unsafe {
            crate::hpc::int8_tile_gemm::int8_gemm_vpdpbusd_ymm(&a_u8, &b_i8, &mut c, m, n, k);
        }
        subtract_i8_to_u8_bias(&mut c, &b_i8, m, n, k);
    } else {
        // Tier 4 — Scalar i8×i8 → i32 reference for non-x86 hosts,
        // pre-AVX-VNNI silicon, or shapes that don't satisfy any of
        // the SIMD tiers' alignment requirements.
        for i in 0..m {
            for p in 0..k {
                let av = a_i8[i * k + p] as i32;
                for j in 0..n {
                    c[i * n + j] += av * b_i8[p * n + j] as i32;
                }
            }
        }
    }

    // Write i32 result back into the (possibly strided) output.
    let (rows, cols) = out.dim();
    debug_assert_eq!(c.len(), rows * cols);
    for r in 0..rows {
        for col in 0..cols {
            out[[r, col]] = c[r * cols + col];
        }
    }
    Ok(())
}

/// Subtract `128 · colsum(B[:, j])` from each `c[i, j]` lane.
///
/// Used by both the AMX and AVX-512-VNNI arms of `matmul_i8_to_i32`
/// to undo the LHS sign-shift bias (A_i8 → A_u8 via +128 means
/// `A_u8 · B = (A_i8 + 128) · B = A_i8 · B + 128 · sum_k B[k, j]`).
/// Pure integer arithmetic, no rounding — the public result is
/// bit-identical to the scalar i8 × i8 → i32 reference.
fn subtract_i8_to_u8_bias(c: &mut [i32], b_i8: &[i8], m: usize, n: usize, k: usize) {
    let mut colsum = vec![0i32; n];
    for p in 0..k {
        for j in 0..n {
            colsum[j] += b_i8[p * n + j] as i32;
        }
    }
    for i in 0..m {
        for j in 0..n {
            c[i * n + j] -= 128 * colsum[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_config_creation() {
        let cfg = TileConfig::for_dpbusd(64);
        assert_eq!(cfg.data[0], 1, "palette 1");
        // Intel SDM XTILECFG: colsb[t] is u16 @ 16+2t, rows[t] is u8 @ 48+t.
        // All three tiles are 16 rows × 64 colbytes at k_bytes = 64.
        assert_eq!(u16::from_le_bytes([cfg.data[16], cfg.data[17]]), 64, "tile0 colsb");
        assert_eq!(cfg.data[48], 16, "tile0 rows");
        assert_eq!(u16::from_le_bytes([cfg.data[18], cfg.data[19]]), 64, "tile1 colsb");
        assert_eq!(cfg.data[49], 16, "tile1 rows");
        assert_eq!(u16::from_le_bytes([cfg.data[20], cfg.data[21]]), 64, "tile2 colsb");
        assert_eq!(cfg.data[50], 16, "tile2 rows (k_bytes/4)");
    }

    #[test]
    fn test_tile_zero_and_release() {
        if !amx_available() {
            eprintln!("AMX not available, skipping");
            return;
        }
        unsafe {
            // Minimal valid tile 0: 1 row × 4 colbytes, using the CORRECTED
            // XTILECFG offsets (colsb[t] u16 @ 16+2t, rows[t] u8 @ 48+t). The
            // old code wrote data[16]=1/data[48]=4 which under the fixed layout
            // means colsb=1/rows=4 — still valid, but mislabeled; now explicit.
            let mut cfg = TileConfig { data: [0u8; 64] };
            cfg.data[0] = 1; // palette 1
            cfg.data[16] = 4; // colsb[0] = 4 bytes  (u16 @ 16)
            cfg.data[48] = 1; // rows[0]  = 1 row    (u8  @ 48)

            tile_loadconfig(&cfg);
            // TILEZERO tmm0
            asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xc0", options(nostack, nomem));
            // TILERELEASE
            asm!(".byte 0xc4, 0xe2, 0x78, 0x49, 0xc0", options(nostack, nomem));
        }
        eprintln!("AMX tile_zero + tile_release: OK on stable Rust");
    }

    // ── Public matmul API tests (sprint A4) ────────────────────────────────

    use crate::hpc::quantized::BF16;
    use crate::{s, Array2};

    /// Reference f32 matmul, fully scalar.
    fn ref_matmul_f32(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let mut c = Array2::<f32>::zeros((m, n));
        for i in 0..m {
            for p in 0..k {
                let av = a[[i, p]];
                for j in 0..n {
                    c[[i, j]] += av * b[[p, j]];
                }
            }
        }
        c
    }

    /// Reference i8×i8 → i32 matmul.
    fn ref_matmul_i8(a: &Array2<i8>, b: &Array2<i8>) -> Array2<i32> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let mut c = Array2::<i32>::zeros((m, n));
        for i in 0..m {
            for p in 0..k {
                let av = a[[i, p]] as i32;
                for j in 0..n {
                    c[[i, j]] += av * b[[p, j]] as i32;
                }
            }
        }
        c
    }

    fn rel_max(actual: &Array2<f32>, expect: &Array2<f32>) -> f32 {
        let mut worst = 0.0f32;
        for (a, b) in actual.iter().zip(expect.iter()) {
            let denom = b.abs().max(1.0);
            let r = (a - b).abs() / denom;
            if r > worst {
                worst = r;
            }
        }
        worst
    }

    #[test]
    fn matmul_bf16_to_f32_16x16() {
        let m = 16;
        let n = 16;
        let k = 16;
        let a_f32 = Array2::<f32>::from_shape_fn((m, k), |(i, j)| ((i + j) as f32) * 0.01);
        let b_f32 = Array2::<f32>::from_shape_fn((k, n), |(i, j)| ((i * 2 + j) as f32) * 0.013);
        let a_bf = a_f32.mapv(BF16::from_f32_rounded);
        let b_bf = b_f32.mapv(BF16::from_f32_rounded);

        let mut out = Array2::<f32>::zeros((m, n));
        matmul_bf16_to_f32(a_bf.view(), b_bf.view(), out.view_mut()).expect("bf16 matmul");

        let expect = ref_matmul_f32(&a_f32, &b_f32);
        let r = rel_max(&out, &expect);
        assert!(r < 0.01, "bf16 matmul exceeded 1% relative error: {}", r);
    }

    /// Inputs whose significands do NOT fit in BF16's 8 bits.
    ///
    /// The previous version of this test used `(i + j) * 0.5` and
    /// `(i * 3 + j) * 0.25` — every one of those values is exactly
    /// BF16-representable, so the downcast was lossless and the test
    /// reported success no matter how much precision the kernel threw
    /// away. Paired with a 1% AMX tolerance, it could neither detect
    /// nor fail on the ~1e-3 error the single-pass path actually had.
    fn irrational_pair(m: usize, k: usize, n: usize) -> (Array2<f32>, Array2<f32>) {
        let a = Array2::<f32>::from_shape_fn((m, k), |(i, j)| (((i * 7 + j * 3) % 97) as f32).sqrt() * 0.3137 - 1.0);
        let b = Array2::<f32>::from_shape_fn((k, n), |(i, j)| (((i * 5 + j * 2) % 89) as f32).sqrt() * 0.2713 - 0.5);
        (a, b)
    }

    #[test]
    fn matmul_f32_16x16() {
        let (m, k, n) = (16, 16, 16);
        let (a, b) = irrational_pair(m, k, n);
        let mut out = Array2::<f32>::zeros((m, n));
        matmul_f32(a.view(), b.view(), out.view_mut()).expect("f32 matmul");
        let expect = ref_matmul_f32(&a, &b);
        // f32 grade on both paths: exact without AMX, three-pass BF16 split
        // with it. 1e-5 is ~100x tighter than one BF16 pass can reach, so
        // this fails if `matmul_f32` ever regresses to a single downcast.
        let r = rel_max(&out, &expect);
        assert!(r <= 1e-5, "f32 matmul exceeded 1e-5 tol: {}", r);
    }

    /// Two-sided: the split must beat one pass by a wide margin AND the
    /// single-pass path must still be measurably lossy.
    ///
    /// If the second half ever fails, AMX gained a real f32 op (or the
    /// host has none and both paths are the exact reference) — either way
    /// the three-pass cost should be re-justified rather than assumed.
    #[test]
    fn three_pass_split_beats_one_bf16_pass() {
        if !amx_available() {
            return; // both paths are the exact f32 reference; nothing to compare
        }
        for (m, k, n) in [(16usize, 32usize, 16usize), (32, 64, 32)] {
            let (a, b) = irrational_pair(m, k, n);
            let expect = ref_matmul_f32(&a, &b);

            let mut exact = Array2::<f32>::zeros((m, n));
            matmul_f32(a.view(), b.view(), exact.view_mut()).expect("exact");
            let r_exact = rel_max(&exact, &expect);
            assert!(r_exact <= 1e-5, "{m}x{k}x{n}: default matmul_f32 {r_exact} is not f32 grade");

            let mut split = Array2::<f32>::zeros((m, n));
            matmul_f32_amx_split(a.view(), b.view(), split.view_mut()).expect("split");
            let r_split = rel_max(&split, &expect);

            let mut fast = Array2::<f32>::zeros((m, n));
            matmul_f32_bf16_fast(a.view(), b.view(), fast.view_mut()).expect("fast");
            let r_fast = rel_max(&fast, &expect);

            assert!(r_split <= 1e-5, "{m}x{k}x{n}: split {r_split} is not f32 grade");
            assert!(
                r_fast > 1e-4,
                "{m}x{k}x{n}: single pass {r_fast} is unexpectedly accurate \
                 — AMX may have a real f32 path now; re-justify the 3x cost"
            );
            assert!(
                r_split * 50.0 < r_fast,
                "{m}x{k}x{n}: split {r_split} vs fast {r_fast} \
                 — less than the ~400x measured; the tail term may not be reaching the kernel"
            );
        }
    }

    #[test]
    fn matmul_i8_to_i32_16x16_exact() {
        let m = 16;
        let n = 16;
        let k = 16;
        let a = Array2::<i8>::from_shape_fn((m, k), |(i, j)| (((i + j) as i32 % 11) - 5) as i8);
        let b = Array2::<i8>::from_shape_fn((k, n), |(i, j)| (((i * 2 + j) as i32 % 13) - 6) as i8);
        let mut out = Array2::<i32>::zeros((m, n));
        matmul_i8_to_i32(a.view(), b.view(), out.view_mut()).expect("i8 matmul");
        let expect = ref_matmul_i8(&a, &b);
        assert_eq!(out, expect);
    }

    #[test]
    fn matmul_bf16_tail_row_17x16() {
        // 17×16 @ 16×16: M has a 1-row tail past the 16-row tile boundary.
        let m = 17;
        let n = 16;
        let k = 16;
        let a_f32 = Array2::<f32>::from_shape_fn((m, k), |(i, j)| ((i + 2 * j) as f32) * 0.02);
        let b_f32 = Array2::<f32>::from_shape_fn((k, n), |(i, j)| ((3 * i + j) as f32) * 0.005);
        let a_bf = a_f32.mapv(BF16::from_f32_rounded);
        let b_bf = b_f32.mapv(BF16::from_f32_rounded);

        let mut out = Array2::<f32>::zeros((m, n));
        matmul_bf16_to_f32(a_bf.view(), b_bf.view(), out.view_mut()).expect("bf16 matmul");

        let expect = ref_matmul_f32(&a_f32, &b_f32);
        let r = rel_max(&out, &expect);
        assert!(r < 0.01, "tail-row bf16 matmul exceeded 1%: {}", r);
    }

    #[test]
    fn matmul_bf16_k_tail_16x65_65x16() {
        // K = 65: one element past a 64-K tile boundary (BF16 tile = 32 elems
        // per dpbf16ps, so 65 lands one past the next-clean boundary).
        let m = 16;
        let n = 16;
        let k = 65;
        let a_f32 = Array2::<f32>::from_shape_fn((m, k), |(i, j)| ((i * 7 + j) as f32) * 0.001);
        let b_f32 = Array2::<f32>::from_shape_fn((k, n), |(i, j)| ((i + j * 5) as f32) * 0.002);
        let a_bf = a_f32.mapv(BF16::from_f32_rounded);
        let b_bf = b_f32.mapv(BF16::from_f32_rounded);

        let mut out = Array2::<f32>::zeros((m, n));
        matmul_bf16_to_f32(a_bf.view(), b_bf.view(), out.view_mut()).expect("bf16 K-tail matmul");

        let expect = ref_matmul_f32(&a_f32, &b_f32);
        let r = rel_max(&out, &expect);
        assert!(r < 0.01, "K-tail bf16 matmul exceeded 1%: {}", r);
    }

    #[test]
    fn matmul_strided_lhs_bf16() {
        // Build a wider source then take every other column with `slice(s![..,
        // ..;2])` so the resulting view is non-contiguous along the inner axis.
        let m = 16;
        let k_full = 32;
        let n = 16;
        let a_f32 = Array2::<f32>::from_shape_fn((m, k_full), |(i, j)| ((i + j) as f32) * 0.01);
        // Take 16 columns out of 32 with stride 2.
        let a_strided = a_f32.slice(s![.., ..;2]); // shape (16, 16)
        assert_eq!(a_strided.dim(), (m, 16));
        assert_ne!(a_strided.strides()[1], 1, "test setup: lhs must be non-contiguous");

        let b_f32 = Array2::<f32>::from_shape_fn((16, n), |(i, j)| ((i + 2 * j) as f32) * 0.01);
        let a_bf = a_strided.mapv(BF16::from_f32_rounded);
        let b_bf = b_f32.mapv(BF16::from_f32_rounded);

        let mut out = Array2::<f32>::zeros((m, n));
        matmul_bf16_to_f32(a_bf.view(), b_bf.view(), out.view_mut()).expect("strided bf16 matmul");

        // Compute reference using the same strided LHS.
        let a_dense = a_strided.to_owned();
        let expect = ref_matmul_f32(&a_dense, &b_f32);
        let r = rel_max(&out, &expect);
        assert!(r < 0.01, "strided bf16 matmul exceeded 1%: {}", r);
    }

    #[test]
    fn matmul_shape_mismatch() {
        let a = Array2::<f32>::zeros((3, 4));
        let b = Array2::<f32>::zeros((5, 6)); // K mismatch
        let mut out = Array2::<f32>::zeros((3, 6));
        let err = matmul_f32(a.view(), b.view(), out.view_mut()).unwrap_err();
        match err {
            MatmulError::ShapeMismatch { lhs, rhs, out: o } => {
                assert_eq!(lhs, (3, 4));
                assert_eq!(rhs, (5, 6));
                assert_eq!(o, (3, 6));
            }
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn matmul_non_contiguous_output_rejected() {
        // Build a (4, 8) source and take every-other column → col stride 2.
        let mut buf = Array2::<f32>::zeros((4, 8));
        let a = Array2::<f32>::zeros((4, 4));
        let b = Array2::<f32>::zeros((4, 4));
        let out = buf.slice_mut(s![.., ..;2]);
        let err = matmul_f32(a.view(), b.view(), out).unwrap_err();
        assert_eq!(err, MatmulError::NonContiguousOutput);
    }

    #[test]
    fn matmul_amx_unavailable_falls_through() {
        // The public surface never returns AmxUnavailable: it falls back.
        let a = Array2::<f32>::ones((4, 4));
        let b = Array2::<f32>::ones((4, 4));
        let mut out = Array2::<f32>::zeros((4, 4));
        matmul_f32(a.view(), b.view(), out.view_mut()).expect("fallback should succeed");
        // 4-wide row of 1s × 4-tall col of 1s = 4
        for v in out.iter() {
            assert!((*v - 4.0).abs() < 1e-4);
        }
    }
    /// Accuracy + throughput of every f32 GEMM path the fork offers, on
    /// one host. `cargo test --release --lib gemm_paths_bench -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gemm_paths_bench() {
        use std::time::Instant;
        let time = |f: &mut dyn FnMut()| {
            f(); // warm
            let reps = 5;
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_secs_f64() * 1e3 / reps as f64
        };
        let have_avx512 = std::is_x86_feature_detected!("avx512f");
        eprintln!("host amx={} avx512f={}", amx_available(), have_avx512);
        if !have_avx512 {
            eprintln!("no avx512f — skipping the F32x16 arm");
            return;
        }
        for &sz in &[256usize, 512, 1024] {
            let (m, k, n) = (sz, sz, sz);
            let (a, b) = irrational_pair(m, k, n);
            let expect = ref_matmul_f32(&a, &b);
            let av: Vec<f32> = a.iter().copied().collect();
            let bv: Vec<f32> = b.iter().copied().collect();
            let mut c = vec![0.0f32; m * n];

            let mut mm_ms =
                time(&mut || crate::backend::native::gemm_f32(m, n, k, 1.0, &av, k, &bv, n, 0.0, &mut c, n));
            let r_mm = rel_max(&Array2::from_shape_vec((m, n), c.clone()).unwrap(), &expect);
            let mut blk_ms = time(&mut || {
                c.iter_mut().for_each(|x| *x = 0.0);
                // SAFETY: `#[target_feature(enable = "avx512f")]`; guarded by the
                // runtime check below, and this bench only runs when it passes.
                unsafe { crate::backend::kernels_avx512::sgemm_blocked(m, n, k, 1.0, &av, k, &bv, n, &mut c, n) }
            });
            let r_blk = rel_max(&Array2::from_shape_vec((m, n), c.clone()).unwrap(), &expect);

            let mut out = Array2::<f32>::zeros((m, n));
            let mut fast_ms = time(&mut || matmul_f32_bf16_fast(a.view(), b.view(), out.view_mut()).unwrap());
            let r_fast = rel_max(&out, &expect);
            let mut split_ms = time(&mut || matmul_f32_amx_split(a.view(), b.view(), out.view_mut()).unwrap());
            let r_split = rel_max(&out, &expect);

            for v in [&mut mm_ms, &mut blk_ms, &mut fast_ms, &mut split_ms] {
                *v = (*v * 100.0).round() / 100.0;
            }
            eprintln!("{sz}^3  matrixmultiply {mm_ms:>8.2}ms rel={r_mm:.1e} | F32x16 sgemm_blocked {blk_ms:>8.2}ms rel={r_blk:.1e} | AMX 1-pass {fast_ms:>8.2}ms rel={r_fast:.1e} | AMX 3-pass {split_ms:>8.2}ms rel={r_split:.1e}");
        }
    }
}
