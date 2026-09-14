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
//! Every op is `unsafe` with the same three preconditions as `amx_matmul`:
//! [`super::amx_matmul::amx_available`] returned `true`, `LDTILECFG` has been
//! executed with a config covering every tile named, and pointers/strides
//! are valid for the configured rows × colsb. Tile-operand aliasing
//! (Gotcha 11, `#UD` → SIGILL) is a COMPILE error here: every three-tile op
//! asserts `D != S1 != S2` in a `const` block.

use core::arch::asm;

// ── AMX-TILE: configuration and data movement ───────────────────────────────

/// `LDTILECFG [cfg]` — load the 64-byte tile configuration.
///
/// # Safety
/// `cfg` must point to 64 readable bytes, 64-byte aligned (`TileConfig`), with
/// a valid palette and in-range rows/colsb (Gotchas 2, 6, 7).
#[inline(always)]
pub unsafe fn ldtilecfg(cfg: *const u8) {
    asm!("ldtilecfg [{c}]", c = in(reg) cfg, options(nostack, readonly));
}

/// `STTILECFG [cfg]` — store the current tile configuration (64 bytes).
///
/// # Safety
/// `cfg` must point to 64 writable, 64-byte-aligned bytes.
#[inline(always)]
pub unsafe fn sttilecfg(cfg: *mut u8) {
    asm!("sttilecfg [{c}]", c = in(reg) cfg, options(nostack));
}

/// `TILERELEASE` — return all tiles to the init state.
///
/// # Safety
/// AMX must be available; no tile may be needed afterwards.
#[inline(always)]
pub unsafe fn tilerelease() {
    asm!("tilerelease", options(nostack, nomem));
}

/// `TILEZERO tmm{T}` for any of the eight tiles.
///
/// # Safety
/// Tiles configured; `T < 8`.
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
#[inline(always)]
pub unsafe fn tileloadd<const T: u8>(base: *const u8, stride: usize) {
    const { assert!(T < 8) }
    asm!("tileloadd tmm{t}, [{b} + {s}*1]", t = const T, b = in(reg) base, s = in(reg) stride, options(nostack, readonly));
}

/// `TILELOADDT1` — same as [`tileloadd`] with the non-temporal (T1) hint.
///
/// # Safety
/// As [`tileloadd`].
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
#[inline(always)]
pub unsafe fn tileloaddrs<const T: u8>(base: *const u8, stride: usize) {
    const { assert!(T < 8) }
    asm!("tileloaddrs tmm{t}, [{b} + {s}*1]", t = const T, b = in(reg) base, s = in(reg) stride, options(nostack, readonly));
}

/// `TILELOADDRST1` — read-shared load with the T1 hint (AMX-MOVRS).
///
/// # Safety
/// As [`tileloaddrs`].
#[inline(always)]
pub unsafe fn tileloaddrst1<const T: u8>(base: *const u8, stride: usize) {
    const { assert!(T < 8) }
    asm!("tileloaddrst1 tmm{t}, [{b} + {s}*1]", t = const T, b = in(reg) base, s = in(reg) stride, options(nostack, readonly));
}

// ── Three-tile dot products: D += S1 · S2 ───────────────────────────────────

macro_rules! tdp3 {
    ($(#[$m:meta])* $name:ident, $mn:literal) => {
        $(#[$m])*
        ///
        /// `D += S1 · S2`; S1 is the plain M×K operand (ModRM.rm), S2 the
        /// VNNI-packed K×N operand (VEX.vvvv). The three tiles must be
        /// distinct — enforced at compile time.
        ///
        /// # Safety
        /// Tiles configured with compatible shapes, AMX available, and the
        /// host must report the feature tier this op belongs to.
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
    tdpbssd, "tdpbssd"
);
tdp3!(
    /// `TDPBSUD` — signed i8 (S1) × unsigned u8 (S2) → i32 (AMX-INT8).
    tdpbsud, "tdpbsud"
);
tdp3!(
    /// `TDPBUSD` — unsigned u8 (S1) × signed i8 (S2) → i32 (AMX-INT8). The
    /// kernel's op: [`tdpbusd::<0, 2, 1>`] is the validated `C4 E2 71 5E C2`.
    tdpbusd, "tdpbusd"
);
tdp3!(
    /// `TDPBUUD` — unsigned u8 × unsigned u8 → i32 (AMX-INT8).
    tdpbuud, "tdpbuud"
);
tdp3!(
    /// `TDPBF16PS` — bf16 × bf16 → f32 (AMX-BF16). [`tdpbf16ps::<0, 2, 1>`] is
    /// the validated `C4 E2 72 5C C2`.
    tdpbf16ps, "tdpbf16ps"
);
tdp3!(
    /// `TDPFP16PS` — fp16 × fp16 → f32 (AMX-FP16, Granite Rapids).
    /// Assembler-verified only.
    tdpfp16ps, "tdpfp16ps"
);
tdp3!(
    /// `TCMMIMFP16PS` — imaginary part of a complex fp16 matrix product → f32
    /// (AMX-COMPLEX). Assembler-verified only.
    tcmmimfp16ps, "tcmmimfp16ps"
);
tdp3!(
    /// `TCMMRLFP16PS` — real part of a complex fp16 matrix product → f32
    /// (AMX-COMPLEX). Assembler-verified only.
    tcmmrlfp16ps, "tcmmrlfp16ps"
);
tdp3!(
    /// `TDPBF8PS` — E5M2 × E5M2 → f32 (AMX-FP8, Diamond Rapids).
    /// Assembler-verified only.
    tdpbf8ps, "tdpbf8ps"
);
tdp3!(
    /// `TDPBHF8PS` — E5M2 (S1) × E4M3 (S2) → f32 (AMX-FP8). Assembler-verified only.
    tdpbhf8ps, "tdpbhf8ps"
);
tdp3!(
    /// `TDPHBF8PS` — E4M3 (S1) × E5M2 (S2) → f32 (AMX-FP8). Assembler-verified only.
    tdphbf8ps, "tdphbf8ps"
);
tdp3!(
    /// `TDPHF8PS` — E4M3 × E4M3 → f32 (AMX-FP8). Assembler-verified only.
    tdphf8ps, "tdphf8ps"
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
        /// Tile `T` configured and holding data; the host must report
        /// [`AmxFeatures::avx512`].
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
    use core::arch::x86_64::{__cpuid, __cpuid_count};
    let max_leaf = __cpuid(0).eax;
    let l7_0 = __cpuid_count(7, 0);
    let l7_1 = if max_leaf >= 7 && __cpuid_count(7, 0).eax >= 1 {
        __cpuid_count(7, 1)
    } else {
        __cpuid_count(0, 0)
    };
    let l1e_1 = if max_leaf >= 0x1e {
        __cpuid_count(0x1e, 1)
    } else {
        __cpuid_count(0, 0)
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
pub fn amx_features() -> AmxFeatures {
    *AMX_FEATURES
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read the machine code of a monomorphized op back out of the text
    /// segment. The wrapper is `#[inline(never)]` so the op's bytes sit inside
    /// one small function; we scan a bounded window for the expected sequence.
    /// This runs on ANY x86_64 host — it inspects encodings, never executes a
    /// tile op — so the `.byte` tables in `amx_matmul` and the mnemonics here
    /// are pinned to each other by CI, not by an EMR box.
    fn contains(f: unsafe fn(), needle: &[u8]) -> bool {
        // SAFETY: `f` is a real function in this binary's text segment; the
        // first 96 bytes from its entry are mapped and readable (the linker
        // packs these wrappers back to back, so the window runs into the
        // NEXT wrapper long before it leaves the section).
        let code = unsafe { core::slice::from_raw_parts(f as *const u8, 96) };
        // ...which is exactly why the window must stop at this wrapper's own
        // `ret` (0xC3): without that bound a negative assertion reads the
        // neighbouring wrapper's encoding and fails, and a positive one can
        // pass on a neighbour's bytes. None of the needles below contains
        // 0xC3, and the wrappers carry no other 0xC3 before their return.
        let end = code
            .iter()
            .position(|&b| b == 0xc3)
            .map_or(code.len(), |p| p + 1);
        code[..end].windows(needle.len()).any(|w| w == needle)
    }

    #[inline(never)]
    unsafe fn w_tilezero0() {
        tilezero::<0>()
    }
    #[inline(never)]
    unsafe fn w_tilezero7() {
        tilezero::<7>()
    }
    #[inline(never)]
    unsafe fn w_tilerelease() {
        tilerelease()
    }
    #[inline(never)]
    unsafe fn w_tdpbusd_021() {
        tdpbusd::<0, 2, 1>()
    }
    #[inline(never)]
    unsafe fn w_tdpbf16ps_021() {
        tdpbf16ps::<0, 2, 1>()
    }
    #[inline(never)]
    unsafe fn w_tdpbusd_012() {
        tdpbusd::<0, 1, 2>()
    }
    #[inline(never)]
    unsafe fn w_tdpbssd_012() {
        tdpbssd::<0, 1, 2>()
    }
    #[inline(never)]
    unsafe fn w_tdpfp16ps_012() {
        tdpfp16ps::<0, 1, 2>()
    }
    #[inline(never)]
    unsafe fn w_tcmmimfp16ps_012() {
        tcmmimfp16ps::<0, 1, 2>()
    }
    #[inline(never)]
    unsafe fn w_tdpbf8ps_012() {
        tdpbf8ps::<0, 1, 2>()
    }
    #[inline(never)]
    unsafe fn w_tdphf8ps_012() {
        tdphf8ps::<0, 1, 2>()
    }
    #[inline(never)]
    unsafe fn w_tmmultf32ps_012() {
        tmmultf32ps::<0, 1, 2>()
    }

    /// The mnemonic path must reproduce `amx_matmul`'s validated `.byte`
    /// table byte for byte — these are the sequences measured on Emerald
    /// Rapids (`amx-enablement-and-kernel.md` §5).
    #[test]
    fn mnemonics_reproduce_the_validated_byte_table() {
        assert!(contains(w_tilezero0, &[0xc4, 0xe2, 0x7b, 0x49, 0xc0]), "TILEZERO tmm0");
        assert!(contains(w_tilerelease, &[0xc4, 0xe2, 0x78, 0x49, 0xc0]), "TILERELEASE");
        assert!(
            contains(w_tdpbusd_021, &[0xc4, 0xe2, 0x71, 0x5e, 0xc2]),
            "TDPBUSD tmm0, tmm2, tmm1 == table C4 E2 71 5E C2"
        );
        assert!(
            contains(w_tdpbf16ps_021, &[0xc4, 0xe2, 0x72, 0x5c, 0xc2]),
            "TDPBF16PS tmm0, tmm2, tmm1 == table C4 E2 72 5C C2"
        );
    }

    /// The operand convention, stated as bytes: swapping S1/S2 swaps
    /// ModRM.rm and VEX.vvvv, nothing else. A body that silently reordered
    /// the operands (the "mirror" the gotchas warn about) would fail one half.
    #[test]
    fn operand_order_is_intel_order_rm_then_vvvv() {
        assert!(contains(w_tdpbusd_012, &[0xc4, 0xe2, 0x69, 0x5e, 0xc1]), "tdpbusd tmm0,tmm1,tmm2 → rm=1 vvvv=2");
        assert!(contains(w_tdpbusd_021, &[0xc4, 0xe2, 0x71, 0x5e, 0xc2]), "tdpbusd tmm0,tmm2,tmm1 → rm=2 vvvv=1");
        assert!(!contains(w_tdpbusd_012, &[0xc4, 0xe2, 0x71, 0x5e, 0xc2]));
    }

    /// Beyond the GEMM tier: the bytes LLVM 22.1.8 emits for the mnemonics
    /// that have never executed here (assembler-verified, per the module doc).
    #[test]
    fn extended_tiers_assemble_to_their_llvm_encodings() {
        assert!(contains(w_tilezero7, &[0xc4, 0xe2, 0x7b, 0x49, 0xf8]), "TILEZERO tmm7");
        assert!(contains(w_tdpbssd_012, &[0xc4, 0xe2, 0x6b, 0x5e, 0xc1]), "TDPBSSD (F2 prefix)");
        assert!(contains(w_tdpfp16ps_012, &[0xc4, 0xe2, 0x6b, 0x5c, 0xc1]), "TDPFP16PS = 5C with F2");
        assert!(contains(w_tcmmimfp16ps_012, &[0xc4, 0xe2, 0x69, 0x6c, 0xc1]), "TCMMIMFP16PS = 6C with 66");
        assert!(contains(w_tdpbf8ps_012, &[0xc4, 0xe5, 0x68, 0xfd, 0xc1]), "TDPBF8PS = map5 FD, no prefix");
        assert!(contains(w_tdphf8ps_012, &[0xc4, 0xe5, 0x69, 0xfd, 0xc1]), "TDPHF8PS = map5 FD, 66");
        assert!(contains(w_tmmultf32ps_012, &[0xc4, 0xe2, 0x69, 0x48, 0xc1]), "TMMULTF32PS = 48 with 66");
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
