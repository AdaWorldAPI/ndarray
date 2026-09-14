//! The full AMX instruction surface as mnemonics — every tile op LLVM's
//! `X86InstrAMX.td` defines, on stable Rust, with the tile numbers as
//! `const` operands.
//!
//! [`amx_matmul`](super::amx_matmul) carries the GEMM subset (`TILEZERO`
//! tmm0..3, `TILELOADD`, `TILESTORED`, `TDPBUSD`, `TDPBF16PS`) as hand-written
//! `.byte` sequences, because in 1.94 that was the only way. Measured on
//! 1.98.1 (LLVM 22.1.8): the integrated assembler accepts every AMX mnemonic
//! inside `asm!` WITHOUT any target feature, and `asm_const` lets the tile
//! index be a generic parameter, so `tilezero tmm{t}` assembles for all eight
//! tiles from one body. This module is that surface. The `.byte` tables stay
//! where they are; the encoding test below reads this module's emitted bytes
//! back and pins them to those tables, so the two can never disagree silently.
//!
//! # Operand order — the "mirror" resolved
//!
//! Intel syntax and LLVM's `MRMSrcReg4VOp3` agree: `tdpbusd tmmD, tmmS1, tmmS2`
//! encodes `D` in ModRM.reg, `S1` in ModRM.rm and `S2` in VEX.vvvv, and the
//! semantics are `D[m][n] += S1[m][k] · S2[k][n]` with S1 the plain M×K
//! operand and S2 the VNNI-packed K×N operand; the letters in `TDPB<U><S>D`
//! name S1's and S2's signedness in that order. The repo's validated table
//! entry `C4 E2 71 5E C2` (rm = tmm2, vvvv = tmm1) is therefore the mnemonic
//! `tdpbusd tmm0, tmm2, tmm1`, i.e. [`tdpbusd::<0, 2, 1>`] — exactly the
//! kernel's placement (plain u8 A in tmm2, VNNI i8 B in tmm1, per
//! `AMX_GOTCHAS.md` Gotcha 12). Nothing was mirrored; the byte table had
//! been read as `(dst, vvvv, rm)`. With mnemonics the question does not
//! arise: name the tiles in Intel order and the assembler does the rest.
//!
//! # Feature tiers (CPUID bits per LLVM `Host.cpp`)
//!
//! | tier | ops | CPUID | first silicon |
//! |---|---|---|---|
//! | AMX-TILE | config, zero, load/store, release | 7.0:EDX[24] | Sapphire Rapids |
//! | AMX-INT8 | `tdpb{ss,su,us,uu}d` | 7.0:EDX[25] | Sapphire Rapids |
//! | AMX-BF16 | `tdpbf16ps` | 7.0:EDX[22] | Sapphire Rapids |
//! | AMX-FP16 | `tdpfp16ps` | 7.1:EAX[21] | Granite Rapids |
//! | AMX-COMPLEX | `tcmm{im,rl}fp16ps` | 7.1:EDX[8] | Granite Rapids-D |
//! | AMX-FP8 | `tdp{b,bh,hb,h}f8ps` | 1E.1:EAX[4] | Diamond Rapids |
//! | AMX-TF32 | `tmmultf32ps` | 1E.1:EAX[6] | Diamond Rapids (see note) |
//! | AMX-AVX512 | `tcvtrow*`, `tilemovrow` | 1E.1:EAX[7] | Diamond Rapids |
//! | AMX-MOVRS | `tileloaddrs{,t1}` | 1E.1:EAX[8] | Diamond Rapids |
//!
//! Note on TF32: LLVM `main` has removed `amx-tf32` (and `amx-transpose`)
//! from both the assembler and `Host.cpp`. The 22.1.8 assembler in the
//! stable toolchain still accepts `tmmultf32ps`, but nightly's LLVM 23
//! already rejects the mnemonic (measured 2026-09-14: `invalid instruction
//! mnemonic 'tmmultf32ps'` on `1.100.0-nightly` / LLVM 23.1.1 — the lib
//! builds because the wrapper is generic, but the first instantiation
//! fails). So that ONE wrapper is emitted as its fixed ISA byte encoding
//! (`VEX.128.66.0F38.W0 48 /r`) instead of the mnemonic — the encoding is
//! defined by the ISA, not by which LLVM still knows the name — and it is
//! gated on the CPUID bit the older `Host.cpp` used. Treat it as CLAIMED,
//! never executed.
//!
//! # What has executed
//!
//! Only the AMX-TILE / INT8 / BF16 tier has ever run in this workspace
//! (Emerald Rapids, `AMX_GOTCHAS.md`). Every other tier here is
//! assembler-verified — the bytes are what LLVM emits for the mnemonic — and
//! NOT execution-verified: no Granite/Diamond Rapids host has run them. The
//! detection API says which tier a host has; the caller must still gate on it.
//!
//! # Safety model
//!
//! Every op is `unsafe` with three preconditions, and the FIRST is split by
//! tier — `amx_available()` is the INT8 gate (AMX-TILE + AMX-INT8 + OS +
//! permission) and must NOT be the precondition for every op, because a host
//! or hypervisor can expose TILE with BF16 / FP16 / FP8 while masking INT8:
//!
//! 1. **Tile state + permission**: [`crate::simd_amx::amx_tile_available`]
//!    returned `true` (AMX-TILE, XSAVE, tile XSTATE, XTILEDATA permission —
//!    no compute-tier bit). Sufficient on its own for the tile-STATE ops
//!    (`ldtilecfg`, `sttilecfg`, `tilezero`, `tileloadd*`, `tilestored`,
//!    `tilerelease`).
//! 2. **Tier**: the compute op's tier is advertised — INT8 ops via
//!    [`super::amx_matmul::amx_available`] (which is gate 1 plus the INT8
//!    bit), every other op via gate 1 AND its [`AmxFeatures`] bit
//!    (`bf16`, `fp16`, `complex`, `fp8`, `tf32`, `avx512`, `movrs`).
//! 3. `LDTILECFG` has been executed with a config covering every tile named,
//!    and pointers/strides are valid for the configured rows × colsb.
//!
//! Tile-operand aliasing (Gotcha 11, `#UD` → SIGILL) is a COMPILE error here:
//! every three-tile op asserts `D != S1 != S2` in a `const` block.

use core::arch::asm;

// ── AMX-TILE: configuration and data movement ───────────────────────────────

/// `LDTILECFG [cfg]` — load the 64-byte tile configuration.
///
/// # Safety
/// `cfg` must point to 64 readable bytes, 64-byte aligned (`TileConfig`), with
/// a valid palette and in-range rows/colsb (Gotchas 2, 6, 7).
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::hpc::amx_matmul::TileConfig;
/// use ndarray::hpc::amx_ops::{ldtilecfg, tilerelease};
/// use ndarray::simd_amx::amx_tile_available;
///
/// if amx_tile_available() {
///     let cfg = TileConfig::for_dpbusd(64);
///     // SAFETY: tile permission held (checked above); `cfg` is 64 aligned bytes
///     // with palette 1 and in-range shapes, covering tiles 0..3.
///     unsafe {
///         ldtilecfg(cfg.data.as_ptr());
///         tilerelease();
///     }
/// }
/// ```
#[inline(always)]
pub unsafe fn ldtilecfg(cfg: *const u8) {
    asm!("ldtilecfg [{c}]", c = in(reg) cfg, options(nostack, readonly));
}

/// `STTILECFG [cfg]` — store the current tile configuration (64 bytes).
///
/// # Safety
/// `cfg` must point to 64 writable, 64-byte-aligned bytes.
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::hpc::amx_matmul::TileConfig;
/// use ndarray::hpc::amx_ops::{ldtilecfg, tilerelease, sttilecfg};
/// use ndarray::simd_amx::amx_tile_available;
///
/// if amx_tile_available() {
///     let cfg = TileConfig::for_dpbusd(64);
///     // SAFETY: tile permission held (checked above); `cfg` is 64 aligned bytes
///     // with palette 1 and in-range shapes, covering tiles 0..3.
///     unsafe {
///         ldtilecfg(cfg.data.as_ptr());
///         let mut back = TileConfig { data: [0u8; 64] };
///         sttilecfg(back.data.as_mut_ptr());
///         assert_eq!(back.data[0], 1, "palette 1 reads back");
///         tilerelease();
///     }
/// }
/// ```
#[inline(always)]
pub unsafe fn sttilecfg(cfg: *mut u8) {
    asm!("sttilecfg [{c}]", c = in(reg) cfg, options(nostack));
}

/// `TILERELEASE` — return all tiles to the init state.
///
/// # Safety
/// AMX must be available; no tile may be needed afterwards.
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::hpc::amx_matmul::TileConfig;
/// use ndarray::hpc::amx_ops::{ldtilecfg, tilerelease};
/// use ndarray::simd_amx::amx_tile_available;
///
/// if amx_tile_available() {
///     let cfg = TileConfig::for_dpbusd(64);
///     // SAFETY: tile permission held (checked above); `cfg` is 64 aligned bytes
///     // with palette 1 and in-range shapes, covering tiles 0..3.
///     unsafe {
///         ldtilecfg(cfg.data.as_ptr());
///         tilerelease();
///     }
/// }
/// ```
#[inline(always)]
pub unsafe fn tilerelease() {
    asm!("tilerelease", options(nostack, nomem));
}

/// `TILEZERO tmm{T}` for any of the eight tiles.
///
/// # Safety
/// Tiles configured; `T < 8`.
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::hpc::amx_matmul::TileConfig;
/// use ndarray::hpc::amx_ops::{ldtilecfg, tilerelease, tilezero};
/// use ndarray::simd_amx::amx_tile_available;
///
/// if amx_tile_available() {
///     let cfg = TileConfig::for_dpbusd(64);
///     // SAFETY: tile permission held (checked above); `cfg` is 64 aligned bytes
///     // with palette 1 and in-range shapes, covering tiles 0..3.
///     unsafe {
///         ldtilecfg(cfg.data.as_ptr());
///         tilezero::<0>();
///         tilerelease();
///     }
/// }
/// ```
#[inline(always)]
pub unsafe fn tilezero<const T: u8>() {
    const { assert!(T < 8) }
    asm!("tilezero tmm{t}", t = const T, options(nostack, nomem));
}

/// `TILELOADD tmm{T}, [base + stride]` — load a tile, one row per `stride`
/// bytes.
///
/// # Safety
/// `base` must be readable for `rows × colsb` of tile `T` at the given row
/// stride; tile configured.
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::hpc::amx_matmul::TileConfig;
/// use ndarray::hpc::amx_ops::{ldtilecfg, tilerelease, tileloadd};
/// use ndarray::simd_amx::amx_tile_available;
///
/// if amx_tile_available() {
///     let cfg = TileConfig::for_dpbusd(64);
///     // SAFETY: tile permission held (checked above); `cfg` is 64 aligned bytes
///     // with palette 1 and in-range shapes, covering tiles 0..3.
///     unsafe {
///         ldtilecfg(cfg.data.as_ptr());
///         // tmm2 is the 16-row × 64-byte M×K operand: one 64-byte row per stride.
///         let a = [0u8; 16 * 64];
///         tileloadd::<2>(a.as_ptr(), 64);
///         tilerelease();
///     }
/// }
/// ```
#[inline(always)]
pub unsafe fn tileloadd<const T: u8>(base: *const u8, stride: usize) {
    const { assert!(T < 8) }
    asm!("tileloadd tmm{t}, [{b} + {s}*1]", t = const T, b = in(reg) base, s = in(reg) stride, options(nostack, readonly));
}

/// `TILELOADDT1` — same as [`tileloadd`] with the non-temporal (T1) hint.
///
/// # Safety
/// As [`tileloadd`].
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::hpc::amx_matmul::TileConfig;
/// use ndarray::hpc::amx_ops::{ldtilecfg, tilerelease, tileloaddt1};
/// use ndarray::simd_amx::amx_tile_available;
///
/// if amx_tile_available() {
///     let cfg = TileConfig::for_dpbusd(64);
///     // SAFETY: tile permission held (checked above); `cfg` is 64 aligned bytes
///     // with palette 1 and in-range shapes, covering tiles 0..3.
///     unsafe {
///         ldtilecfg(cfg.data.as_ptr());
///         let a = [0u8; 16 * 64];
///         tileloaddt1::<2>(a.as_ptr(), 64);
///         tilerelease();
///     }
/// }
/// ```
#[inline(always)]
pub unsafe fn tileloaddt1<const T: u8>(base: *const u8, stride: usize) {
    const { assert!(T < 8) }
    asm!("tileloaddt1 tmm{t}, [{b} + {s}*1]", t = const T, b = in(reg) base, s = in(reg) stride, options(nostack, readonly));
}

/// `TILESTORED [base + stride], tmm{T}` — store a tile.
///
/// # Safety
/// `base` must be writable for `rows × colsb` of tile `T` at the given row
/// stride; tile configured.
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::hpc::amx_matmul::TileConfig;
/// use ndarray::hpc::amx_ops::{ldtilecfg, tilerelease, tilestored, tilezero};
/// use ndarray::simd_amx::amx_tile_available;
///
/// if amx_tile_available() {
///     let cfg = TileConfig::for_dpbusd(64);
///     // SAFETY: tile permission held (checked above); `cfg` is 64 aligned bytes
///     // with palette 1 and in-range shapes, covering tiles 0..3.
///     unsafe {
///         ldtilecfg(cfg.data.as_ptr());
///         // tmm0 is the 16×16 i32 accumulator: 16 rows of 64 bytes.
///         let mut c = [0i32; 16 * 16];
///         tilezero::<0>();
///         tilestored::<0>(c.as_mut_ptr().cast::<u8>(), 64);
///         assert!(c.iter().all(|&x| x == 0));
///         tilerelease();
///     }
/// }
/// ```
#[inline(always)]
pub unsafe fn tilestored<const T: u8>(base: *mut u8, stride: usize) {
    const { assert!(T < 8) }
    asm!("tilestored [{b} + {s}*1], tmm{t}", t = const T, b = in(reg) base, s = in(reg) stride, options(nostack));
}

// ── AMX-MOVRS (Diamond Rapids): read-shared loads ───────────────────────────

/// `TILELOADDRS tmm{T}, [base + stride]` — load with the read-shared hint
/// (AMX-MOVRS). Assembler-verified, not execution-verified.
///
/// # Safety
/// As [`tileloadd`], and the host must report [`AmxFeatures::movrs`].
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::hpc::amx_matmul::TileConfig;
/// use ndarray::hpc::amx_ops::{amx_features, ldtilecfg, tilerelease, tileloaddrs};
/// use ndarray::simd_amx::amx_tile_available;
///
/// // Tile state AND the MOVRS tier — the INT8 gate says nothing about MOVRS.
/// if amx_tile_available() && amx_features().movrs {
///     let cfg = TileConfig::for_dpbusd(64);
///     let a = [0u8; 16 * 64];
///     // SAFETY: tile permission held, config covers tile 2, `a` is 16 rows × 64 B.
///     unsafe {
///         ldtilecfg(cfg.data.as_ptr());
///         tileloaddrs::<2>(a.as_ptr(), 64);
///         tilerelease();
///     }
/// }
/// ```
#[inline(always)]
pub unsafe fn tileloaddrs<const T: u8>(base: *const u8, stride: usize) {
    const { assert!(T < 8) }
    asm!("tileloaddrs tmm{t}, [{b} + {s}*1]", t = const T, b = in(reg) base, s = in(reg) stride, options(nostack, readonly));
}

/// `TILELOADDRST1` — read-shared load with the T1 hint (AMX-MOVRS).
///
/// # Safety
/// As [`tileloaddrs`].
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::hpc::amx_matmul::TileConfig;
/// use ndarray::hpc::amx_ops::{amx_features, ldtilecfg, tilerelease, tileloaddrst1};
/// use ndarray::simd_amx::amx_tile_available;
///
/// // Tile state AND the MOVRS tier — the INT8 gate says nothing about MOVRS.
/// if amx_tile_available() && amx_features().movrs {
///     let cfg = TileConfig::for_dpbusd(64);
///     let a = [0u8; 16 * 64];
///     // SAFETY: tile permission held, config covers tile 2, `a` is 16 rows × 64 B.
///     unsafe {
///         ldtilecfg(cfg.data.as_ptr());
///         tileloaddrst1::<2>(a.as_ptr(), 64);
///         tilerelease();
///     }
/// }
/// ```
#[inline(always)]
pub unsafe fn tileloaddrst1<const T: u8>(base: *const u8, stride: usize) {
    const { assert!(T < 8) }
    asm!("tileloaddrst1 tmm{t}, [{b} + {s}*1]", t = const T, b = in(reg) base, s = in(reg) stride, options(nostack, readonly));
}

// ── Three-tile dot products: D += S1 · S2 ───────────────────────────────────

macro_rules! tdp3 {
    ($(#[$m:meta])* $name:ident, $mn:literal, $tier:ident) => {
        $(#[$m])*
        ///
        /// `D += S1 · S2`; S1 is the plain M×K operand (ModRM.rm), S2 the
        /// VNNI-packed K×N operand (VEX.vvvv). The three tiles must be
        /// distinct — enforced at compile time.
        ///
        /// # Safety
        /// Tiles configured with compatible shapes,
        /// [`crate::simd_amx::amx_tile_available`] true, and the host must
        #[doc = concat!("report this op's tier: [`AmxFeatures::", stringify!($tier), "`].")]
        ///
        /// # Examples
        ///
        /// Gate on the tile state AND this op's own tier — never on the INT8
        /// gate for a non-INT8 op — then run it on the three distinct tiles
        /// the GEMM config lays out (`C → tmm0`, VNNI K×N → tmm1, M×K → tmm2).
        ///
        /// ```rust,no_run
        /// use ndarray::hpc::amx_matmul::TileConfig;
        #[doc = concat!("use ndarray::hpc::amx_ops::{amx_features, ldtilecfg, tilerelease, tilezero, ", stringify!($name), "};")]
        /// use ndarray::simd_amx::amx_tile_available;
        ///
        #[doc = concat!("if amx_tile_available() && amx_features().", stringify!($tier), " {")]
        ///     let cfg = TileConfig::for_dpbusd(64);
        ///     // SAFETY: tile permission held (checked above), the config covers
        ///     // tiles 0..3 with compatible shapes, and the operands are distinct.
        ///     unsafe {
        ///         ldtilecfg(cfg.data.as_ptr());
        ///         tilezero::<0>();
        #[doc = concat!("        ", stringify!($name), "::<0, 2, 1>();")]
        ///         tilerelease();
        ///     }
        /// }
        /// ```
        #[inline(always)]
        pub unsafe fn $name<const D: u8, const S1: u8, const S2: u8>() {
            const {
                assert!(D < 8 && S1 < 8 && S2 < 8);
                assert!(D != S1 && D != S2 && S1 != S2, "tile operands must be distinct (#UD otherwise)");
            }
            asm!(concat!($mn, " tmm{d}, tmm{a}, tmm{b}"), d = const D, a = const S1, b = const S2, options(nostack, nomem));
        }
    };
}

tdp3!(
    /// `TDPBSSD` — signed i8 × signed i8 → i32 (AMX-INT8).
    tdpbssd, "tdpbssd", int8
);
tdp3!(
    /// `TDPBSUD` — signed i8 (S1) × unsigned u8 (S2) → i32 (AMX-INT8).
    tdpbsud, "tdpbsud", int8
);
tdp3!(
    /// `TDPBUSD` — unsigned u8 (S1) × signed i8 (S2) → i32 (AMX-INT8). The
    /// kernel's op: [`tdpbusd::<0, 2, 1>`] is the validated `C4 E2 71 5E C2`.
    tdpbusd, "tdpbusd", int8
);
tdp3!(
    /// `TDPBUUD` — unsigned u8 × unsigned u8 → i32 (AMX-INT8).
    tdpbuud, "tdpbuud", int8
);
tdp3!(
    /// `TDPBF16PS` — bf16 × bf16 → f32 (AMX-BF16). [`tdpbf16ps::<0, 2, 1>`] is
    /// the validated `C4 E2 72 5C C2`.
    tdpbf16ps, "tdpbf16ps", bf16
);
tdp3!(
    /// `TDPFP16PS` — fp16 × fp16 → f32 (AMX-FP16, Granite Rapids).
    /// Assembler-verified only.
    tdpfp16ps, "tdpfp16ps", fp16
);
tdp3!(
    /// `TCMMIMFP16PS` — imaginary part of a complex fp16 matrix product → f32
    /// (AMX-COMPLEX). Assembler-verified only.
    tcmmimfp16ps, "tcmmimfp16ps", complex
);
tdp3!(
    /// `TCMMRLFP16PS` — real part of a complex fp16 matrix product → f32
    /// (AMX-COMPLEX). Assembler-verified only.
    tcmmrlfp16ps, "tcmmrlfp16ps", complex
);
tdp3!(
    /// `TDPBF8PS` — E5M2 × E5M2 → f32 (AMX-FP8, Diamond Rapids).
    /// Assembler-verified only.
    tdpbf8ps, "tdpbf8ps", fp8
);
tdp3!(
    /// `TDPBHF8PS` — E5M2 (S1) × E4M3 (S2) → f32 (AMX-FP8). Assembler-verified only.
    tdpbhf8ps, "tdpbhf8ps", fp8
);
tdp3!(
    /// `TDPHBF8PS` — E4M3 (S1) × E5M2 (S2) → f32 (AMX-FP8). Assembler-verified only.
    tdphbf8ps, "tdphbf8ps", fp8
);
tdp3!(
    /// `TDPHF8PS` — E4M3 × E4M3 → f32 (AMX-FP8). Assembler-verified only.
    tdphf8ps, "tdphf8ps", fp8
);
/// `TMMULTF32PS` — tf32 × tf32 → f32 (AMX-TF32). CLAIMED — no host has
/// executed it.
///
/// Emitted as raw bytes, not a mnemonic: LLVM `main` dropped `amx-tf32`, and
/// nightly's LLVM 23 rejects `tmmultf32ps` while stable's 22.1.8 still
/// assembles it. The encoding is fixed by the ISA — `C4 E2 <vex> 48 <modrm>`
/// with `vex = (!S2 & 0xF) << 3 | 0b01` (W0, vvvv = S2 inverted, L0, pp=66)
/// and `modrm = 0xC0 | D << 3 | S1` — and reproduces the same byte table the
/// mnemonic form did (`C4 E2 69 48 C1` for tiles 0, 1, 2), which the
/// `extended_tiers_assemble_to_their_llvm_encodings` test pins.
///
/// `D += S1 · S2`; S1 is the plain M×K operand (ModRM.rm), S2 the
/// VNNI-packed K×N operand (VEX.vvvv). The three tiles must be distinct —
/// enforced at compile time.
///
/// # Safety
/// Tiles configured with compatible shapes, AMX available, and the host
/// must report AMX-TF32.
///
/// # Examples
///
/// ```rust,no_run
/// use ndarray::hpc::amx_matmul::TileConfig;
/// use ndarray::hpc::amx_ops::{amx_features, ldtilecfg, tilerelease, tilezero, tmmultf32ps};
/// use ndarray::simd_amx::amx_tile_available;
///
/// if amx_tile_available() && amx_features().tf32 {
///     let cfg = TileConfig::for_dpbusd(64);
///     // SAFETY: tile permission held, TF32 advertised, config covers tiles
///     // 0..3, operands distinct.
///     unsafe {
///         ldtilecfg(cfg.data.as_ptr());
///         tilezero::<0>();
///         tmmultf32ps::<0, 2, 1>();
///         tilerelease();
///     }
/// }
/// ```
#[inline(always)]
pub unsafe fn tmmultf32ps<const D: u8, const S1: u8, const S2: u8>() {
    const {
        assert!(D < 8 && S1 < 8 && S2 < 8);
        assert!(D != S1 && D != S2 && S1 != S2, "tile operands must be distinct (#UD otherwise)");
    }
    asm!(
        ".byte 0xC4, 0xE2, {vex}, 0x48, {modrm}",
        vex = const ((!S2 & 0x0F) << 3) | 0x01,
        modrm = const 0xC0 | (D << 3) | S1,
        options(nostack, nomem)
    );
}

// ── AMX-AVX512 (Diamond Rapids): tile row → zmm ─────────────────────────────
//
// These need a zmm operand, which `asm!` only accepts when `avx512f` is a
// compile-time target feature, so they exist under the v4/native configs
// only. Same compile-time selection as everything else in this crate — no
// `#[target_feature]`, no runtime dispatch.

#[cfg(target_feature = "avx512f")]
macro_rules! tile_row_to_zmm {
    ($(#[$m:meta])* $name:ident, $name_imm:ident, $mn:literal, $ty:ty) => {
        $(#[$m])*
        ///
        /// Register-row form: `row` selects the tile row at run time.
        ///
        /// # Safety
        /// Tile `T` configured and holding data,
        /// [`crate::simd_amx::amx_tile_available`] true, and the host must
        /// report [`AmxFeatures::avx512`].
        ///
        /// # Examples
        ///
        /// `ignore`d rather than `no_run` because this function exists only
        /// when `avx512f` is a compile-time target feature (v4 / native
        /// builds); a v3 doctest build would not find it.
        ///
        /// ```rust,ignore
        /// use ndarray::hpc::amx_matmul::TileConfig;
        #[doc = concat!("use ndarray::hpc::amx_ops::{amx_features, ldtilecfg, tilerelease, tilezero, ", stringify!($name), "};")]
        /// use ndarray::simd_amx::amx_tile_available;
        ///
        /// if amx_tile_available() && amx_features().avx512 {
        ///     let cfg = TileConfig::for_dpbusd(64);
        ///     // SAFETY: tile permission held, AMX-AVX512 advertised, tile 0
        ///     // configured and zeroed before its row is read.
        ///     unsafe {
        ///         ldtilecfg(cfg.data.as_ptr());
        ///         tilezero::<0>();
        #[doc = concat!("        let _row0 = ", stringify!($name), "::<0>(0);")]
        ///         tilerelease();
        ///     }
        /// }
        /// ```
        #[inline(always)]
        pub unsafe fn $name<const T: u8>(row: u32) -> $ty {
            const { assert!(T < 8) }
            let out: $ty;
            asm!(concat!($mn, " {o}, tmm{t}, {r:e}"), o = out(zmm_reg) out, t = const T, r = in(reg) row, options(nostack, nomem));
            out
        }
        $(#[$m])*
        ///
        /// Immediate-row form: `ROW` is a compile-time constant.
        ///
        /// # Safety
        /// As the register-row form.
        ///
        /// # Examples
        ///
        /// `ignore`d for the same reason as the register-row form (the
        /// function exists only under a compile-time `avx512f`).
        ///
        /// ```rust,ignore
        /// use ndarray::hpc::amx_matmul::TileConfig;
        #[doc = concat!("use ndarray::hpc::amx_ops::{amx_features, ldtilecfg, tilerelease, tilezero, ", stringify!($name_imm), "};")]
        /// use ndarray::simd_amx::amx_tile_available;
        ///
        /// if amx_tile_available() && amx_features().avx512 {
        ///     let cfg = TileConfig::for_dpbusd(64);
        ///     // SAFETY: tile permission held, AMX-AVX512 advertised, tile 0
        ///     // configured and zeroed; ROW 3 < 16 configured rows.
        ///     unsafe {
        ///         ldtilecfg(cfg.data.as_ptr());
        ///         tilezero::<0>();
        #[doc = concat!("        let _row3 = ", stringify!($name_imm), "::<0, 3>();")]
        ///         tilerelease();
        ///     }
        /// }
        /// ```
        #[inline(always)]
        pub unsafe fn $name_imm<const T: u8, const ROW: u8>() -> $ty {
            const { assert!(T < 8 && ROW < 16) }
            let out: $ty;
            asm!(concat!($mn, " {o}, tmm{t}, {r}"), o = out(zmm_reg) out, t = const T, r = const ROW, options(nostack, nomem));
            out
        }
    };
}

#[cfg(target_feature = "avx512f")]
tile_row_to_zmm!(
    /// `TCVTROWD2PS` — one tile row of 16 × i32 converted to 16 × f32.
    /// Assembler-verified only.
    tcvtrowd2ps, tcvtrowd2ps_imm, "tcvtrowd2ps", core::arch::x86_64::__m512
);
#[cfg(target_feature = "avx512f")]
tile_row_to_zmm!(
    /// `TCVTROWPS2PHH` — tile row of f32 → fp16, high halves. Assembler-verified only.
    tcvtrowps2phh, tcvtrowps2phh_imm, "tcvtrowps2phh", core::arch::x86_64::__m512i
);
#[cfg(target_feature = "avx512f")]
tile_row_to_zmm!(
    /// `TCVTROWPS2PHL` — tile row of f32 → fp16, low halves. Assembler-verified only.
    tcvtrowps2phl, tcvtrowps2phl_imm, "tcvtrowps2phl", core::arch::x86_64::__m512i
);
#[cfg(target_feature = "avx512f")]
tile_row_to_zmm!(
    /// `TCVTROWPS2BF16H` — tile row of f32 → bf16, high halves. Assembler-verified only.
    tcvtrowps2bf16h, tcvtrowps2bf16h_imm, "tcvtrowps2bf16h", core::arch::x86_64::__m512i
);
#[cfg(target_feature = "avx512f")]
tile_row_to_zmm!(
    /// `TCVTROWPS2BF16L` — tile row of f32 → bf16, low halves. Assembler-verified only.
    tcvtrowps2bf16l, tcvtrowps2bf16l_imm, "tcvtrowps2bf16l", core::arch::x86_64::__m512i
);
#[cfg(target_feature = "avx512f")]
tile_row_to_zmm!(
    /// `TILEMOVROW` — one 64-byte tile row moved into a zmm unchanged.
    /// Assembler-verified only.
    tilemovrow, tilemovrow_imm, "tilemovrow", core::arch::x86_64::__m512i
);

// ── Per-tier detection ──────────────────────────────────────────────────────

/// Which AMX tiers this CPU advertises, per LLVM `Host.cpp`'s bit positions.
///
/// Silicon bits only — [`super::amx_matmul::amx_available`] is still the gate
/// for "may I execute a tile op" (OS XSAVE state + `arch_prctl` permission);
/// this struct answers "which ops exist once I may".
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct AmxFeatures {
    /// AMX-TILE (7.0:EDX[24]).
    pub tile: bool,
    /// AMX-INT8 (7.0:EDX[25]).
    pub int8: bool,
    /// AMX-BF16 (7.0:EDX[22]).
    pub bf16: bool,
    /// AMX-FP16 (7.1:EAX[21]).
    pub fp16: bool,
    /// AMX-COMPLEX (7.1:EDX[8]).
    pub complex: bool,
    /// AMX-FP8 (1E.1:EAX[4]).
    pub fp8: bool,
    /// AMX-TF32 (1E.1:EAX[6]; the bit LLVM used before dropping the feature).
    pub tf32: bool,
    /// AMX-AVX512 (1E.1:EAX[7]).
    pub avx512: bool,
    /// AMX-MOVRS (1E.1:EAX[8]).
    pub movrs: bool,
}

fn detect_amx_features() -> AmxFeatures {
    use core::arch::x86_64::{__cpuid, __cpuid_count, CpuidResult};
    let max_leaf = __cpuid(0).eax;
    // An out-of-range basic leaf may return the HIGHEST basic leaf's data, so
    // every leaf is guarded by `max_leaf` and an unavailable one reads as all
    // zero — never as leaf 0 (vendor string + max leaf), whose bits are not
    // feature bits either.
    let zero = || CpuidResult {
        eax: 0,
        ebx: 0,
        ecx: 0,
        edx: 0,
    };
    let l7_0 = if max_leaf >= 7 { __cpuid_count(7, 0) } else { zero() };
    let l7_1 = if max_leaf >= 7 && l7_0.eax >= 1 {
        __cpuid_count(7, 1)
    } else {
        zero()
    };
    let l1e_1 = if max_leaf >= 0x1e {
        __cpuid_count(0x1e, 1)
    } else {
        zero()
    };
    let bit = |v: u32, b: u32| (v >> b) & 1 == 1;
    AmxFeatures {
        tile: bit(l7_0.edx, 24),
        int8: bit(l7_0.edx, 25),
        bf16: bit(l7_0.edx, 22),
        fp16: bit(l7_1.eax, 21),
        complex: bit(l7_1.edx, 8),
        fp8: max_leaf >= 0x1e && bit(l1e_1.eax, 4),
        tf32: max_leaf >= 0x1e && bit(l1e_1.eax, 6),
        avx512: max_leaf >= 0x1e && bit(l1e_1.eax, 7),
        movrs: max_leaf >= 0x1e && bit(l1e_1.eax, 8),
    }
}

static AMX_FEATURES: std::sync::LazyLock<AmxFeatures> = std::sync::LazyLock::new(detect_amx_features);

/// The advertised AMX tiers, cached (CPUID is a serializing instruction; once
/// is enough).
///
/// # Examples
///
/// Runs on any x86_64 host — it only reads CPUID:
///
/// ```
/// use ndarray::hpc::amx_ops::amx_features;
/// let f = amx_features();
/// // Every compute tier rides on AMX-TILE; a tier without TILE is not a CPU.
/// if f.int8 || f.bf16 || f.fp16 {
///     assert!(f.tile);
/// }
/// ```
pub fn amx_features() -> AmxFeatures {
    *AMX_FEATURES
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read the machine code of a monomorphized op out of the OBJECT FILE —
    /// the test binary itself, `/proc/self/exe` — via its ELF `.symtab`: the
    /// symbol's `st_value`/`st_size` are the linker's own statement of where
    /// the wrapper starts and how long it is, so the extent is validated by
    /// the producer of the bytes, never inferred from a function pointer, a
    /// fixed window, or a `ret`-byte heuristic (0xC3 can sit inside another
    /// instruction's immediate). No executable memory is dereferenced at all.
    /// Each wrapper carries an `export_name` so it can be found by name, and
    /// `#[inline(never)]` so the op's bytes are its own symbol's bytes. Runs
    /// on ANY x86_64 Linux host — it inspects encodings, never executes a
    /// tile op — so the `.byte` tables in `amx_matmul` and the mnemonics here
    /// are pinned to each other by CI, not by an EMR box. Requires an
    /// unstripped test binary (cargo's default for every test profile).
    fn symbol_bytes(name: &str) -> Vec<u8> {
        let exe = std::fs::read("/proc/self/exe").expect("read /proc/self/exe");
        let u16_at = |o: usize| u16::from_le_bytes([exe[o], exe[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(exe[o..o + 4].try_into().expect("4 bytes"));
        let u64_at = |o: usize| u64::from_le_bytes(exe[o..o + 8].try_into().expect("8 bytes"));
        assert_eq!(&exe[..4], b"\x7fELF", "test binary is ELF");
        assert_eq!(exe[4], 2, "ELF64");
        let shoff = u64_at(0x28) as usize;
        let shentsize = u16_at(0x3a) as usize;
        let shnum = u16_at(0x3c) as usize;
        // (sh_type, sh_addr, sh_offset, sh_size, sh_link, sh_entsize)
        let section = |i: usize| {
            let b = shoff + i * shentsize;
            (
                u32_at(b + 4),
                u64_at(b + 0x10),
                u64_at(b + 0x18),
                u64_at(b + 0x20),
                u32_at(b + 0x28),
                u64_at(b + 0x38),
            )
        };
        const SHT_SYMTAB: u32 = 2;
        let symtab = (0..shnum)
            .map(section)
            .find(|s| s.0 == SHT_SYMTAB)
            .expect("test binary carries .symtab — do not strip test binaries");
        let strtab = section(symtab.4 as usize);
        let entsize = symtab.5 as usize;
        assert_eq!(entsize, 24, "Elf64_Sym");
        for i in 0..(symtab.3 as usize / entsize) {
            let b = symtab.2 as usize + i * entsize;
            let name_off = strtab.2 as usize + u32_at(b) as usize;
            let name_len = exe[name_off..]
                .iter()
                .position(|&c| c == 0)
                .expect("NUL-terminated symbol name");
            if &exe[name_off..name_off + name_len] != name.as_bytes() {
                continue;
            }
            let st_shndx = u16_at(b + 6) as usize;
            let st_value = u64_at(b + 8);
            let st_size = u64_at(b + 16) as usize;
            assert!(st_size > 0, "{name}: symbol has no size");
            let sec = section(st_shndx);
            let file_off = (st_value - sec.1 + sec.2) as usize;
            return exe[file_off..file_off + st_size].to_vec();
        }
        panic!("symbol {name} not found in .symtab");
    }

    /// A wrapper as (fn pointer, exported symbol name). The pointer is only
    /// ever passed through `black_box` — it is never dereferenced — so that
    /// the otherwise-unreferenced wrapper is actually codegen'd into the test
    /// binary (an `export_name` alone does not keep a dead fn alive here;
    /// measured: 0 probe symbols in `.symtab` without the reference).
    macro_rules! probe {
        ($w:ident) => {
            ($w as unsafe fn(), concat!("ndarray_amx_probe_", stringify!($w)))
        };
    }

    /// Does the wrapper's own symbol contain the exact encoding? Bounded by
    /// the symbol's linker-recorded size, so a neighbouring wrapper's bytes
    /// can neither fail a negative assertion nor pass a positive one.
    fn contains((f, name): (unsafe fn(), &str), needle: &[u8]) -> bool {
        std::hint::black_box(f as usize);
        symbol_bytes(name)
            .windows(needle.len())
            .any(|w| w == needle)
    }

    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tilezero0"]
    unsafe fn w_tilezero0() {
        tilezero::<0>()
    }
    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tilezero7"]
    unsafe fn w_tilezero7() {
        tilezero::<7>()
    }
    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tilerelease"]
    unsafe fn w_tilerelease() {
        tilerelease()
    }
    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tdpbusd_021"]
    unsafe fn w_tdpbusd_021() {
        tdpbusd::<0, 2, 1>()
    }
    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tdpbf16ps_021"]
    unsafe fn w_tdpbf16ps_021() {
        tdpbf16ps::<0, 2, 1>()
    }
    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tdpbusd_012"]
    unsafe fn w_tdpbusd_012() {
        tdpbusd::<0, 1, 2>()
    }
    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tdpbssd_012"]
    unsafe fn w_tdpbssd_012() {
        tdpbssd::<0, 1, 2>()
    }
    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tdpfp16ps_012"]
    unsafe fn w_tdpfp16ps_012() {
        tdpfp16ps::<0, 1, 2>()
    }
    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tcmmimfp16ps_012"]
    unsafe fn w_tcmmimfp16ps_012() {
        tcmmimfp16ps::<0, 1, 2>()
    }
    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tdpbf8ps_012"]
    unsafe fn w_tdpbf8ps_012() {
        tdpbf8ps::<0, 1, 2>()
    }
    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tdphf8ps_012"]
    unsafe fn w_tdphf8ps_012() {
        tdphf8ps::<0, 1, 2>()
    }
    #[inline(never)]
    #[export_name = "ndarray_amx_probe_w_tmmultf32ps_012"]
    unsafe fn w_tmmultf32ps_012() {
        tmmultf32ps::<0, 1, 2>()
    }

    /// The mnemonic path must reproduce `amx_matmul`'s validated `.byte`
    /// table byte for byte — these are the sequences measured on Emerald
    /// Rapids (`amx-enablement-and-kernel.md` §5).
    #[test]
    fn mnemonics_reproduce_the_validated_byte_table() {
        assert!(contains(probe!(w_tilezero0), &[0xc4, 0xe2, 0x7b, 0x49, 0xc0]), "TILEZERO tmm0");
        assert!(contains(probe!(w_tilerelease), &[0xc4, 0xe2, 0x78, 0x49, 0xc0]), "TILERELEASE");
        assert!(
            contains(probe!(w_tdpbusd_021), &[0xc4, 0xe2, 0x71, 0x5e, 0xc2]),
            "TDPBUSD tmm0, tmm2, tmm1 == table C4 E2 71 5E C2"
        );
        assert!(
            contains(probe!(w_tdpbf16ps_021), &[0xc4, 0xe2, 0x72, 0x5c, 0xc2]),
            "TDPBF16PS tmm0, tmm2, tmm1 == table C4 E2 72 5C C2"
        );
    }

    /// The operand convention, stated as bytes: swapping S1/S2 swaps
    /// ModRM.rm and VEX.vvvv, nothing else. A body that silently reordered
    /// the operands (the "mirror" the gotchas warn about) would fail one half.
    #[test]
    fn operand_order_is_intel_order_rm_then_vvvv() {
        assert!(
            contains(probe!(w_tdpbusd_012), &[0xc4, 0xe2, 0x69, 0x5e, 0xc1]),
            "tdpbusd tmm0,tmm1,tmm2 → rm=1 vvvv=2"
        );
        assert!(
            contains(probe!(w_tdpbusd_021), &[0xc4, 0xe2, 0x71, 0x5e, 0xc2]),
            "tdpbusd tmm0,tmm2,tmm1 → rm=2 vvvv=1"
        );
        assert!(!contains(probe!(w_tdpbusd_012), &[0xc4, 0xe2, 0x71, 0x5e, 0xc2]));
    }

    /// Beyond the GEMM tier: the bytes LLVM 22.1.8 emits for the mnemonics
    /// that have never executed here (assembler-verified, per the module doc).
    #[test]
    fn extended_tiers_assemble_to_their_llvm_encodings() {
        assert!(contains(probe!(w_tilezero7), &[0xc4, 0xe2, 0x7b, 0x49, 0xf8]), "TILEZERO tmm7");
        assert!(contains(probe!(w_tdpbssd_012), &[0xc4, 0xe2, 0x6b, 0x5e, 0xc1]), "TDPBSSD (F2 prefix)");
        assert!(contains(probe!(w_tdpfp16ps_012), &[0xc4, 0xe2, 0x6b, 0x5c, 0xc1]), "TDPFP16PS = 5C with F2");
        assert!(contains(probe!(w_tcmmimfp16ps_012), &[0xc4, 0xe2, 0x69, 0x6c, 0xc1]), "TCMMIMFP16PS = 6C with 66");
        assert!(contains(probe!(w_tdpbf8ps_012), &[0xc4, 0xe5, 0x68, 0xfd, 0xc1]), "TDPBF8PS = map5 FD, no prefix");
        assert!(contains(probe!(w_tdphf8ps_012), &[0xc4, 0xe5, 0x69, 0xfd, 0xc1]), "TDPHF8PS = map5 FD, 66");
        assert!(contains(probe!(w_tmmultf32ps_012), &[0xc4, 0xe2, 0x69, 0x48, 0xc1]), "TMMULTF32PS = 48 with 66");
    }

    #[test]
    fn feature_bits_are_consistent_with_the_legacy_detector() {
        let f = amx_features();
        // The three SPR-era bits are exactly what `amx_report` reads; the
        // extended tiers imply TILE.
        let l7 = core::arch::x86_64::__cpuid_count(7, 0);
        assert_eq!(f.tile, (l7.edx >> 24) & 1 == 1);
        assert_eq!(f.int8, (l7.edx >> 25) & 1 == 1);
        assert_eq!(f.bf16, (l7.edx >> 22) & 1 == 1);
        for ext in [f.fp16, f.complex, f.fp8, f.tf32, f.avx512, f.movrs] {
            if ext {
                assert!(f.tile, "an extended AMX tier without AMX-TILE is not a real CPU");
            }
        }
    }
}
