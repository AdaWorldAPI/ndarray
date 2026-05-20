# SIMD Tier Integration Plan

> **Companion to:** `.claude/knowledge/td-simd-tier-audit.md` — read it first. This document is the *fix plan* for the 22 verified findings recorded there, plus the architecture decision for **intra-bucket dispatch** (Sapphire Rapids vs Ice Lake-SP vs Cooper Lake within `Tier::Avx512`, A76 vs A72 within `Tier::Neon`, etc.).

## Goals

1. `crate::simd::*` reaches the silicon ceiling on every CPU. No scalar where SIMD exists; no AVX2 where AVX-512 exists; no AVX-512F-only where VNNI / BF16 / FP16 / AMX exists.
2. **One binary, all ISAs** (default): runtime feature detection at startup, frozen dispatch tables, zero per-call branch cost.
3. **Optional pinning** for users distributing per-tier binaries: cargo features that statically resolve dispatch to one profile, removing the runtime branch entirely. Same source, two ergonomics.
4. Wire what already exists before adding new abstractions. Phase 1 closes ~70% of audit debt by routing existing kernels through their existing dispatch sites.

## Non-goals

- Inventing new GEMM kernels. `matrixmultiply` covers AVX2 GEMM; AMX tile kernels already exist; VNNI GEMM already exists. The work is routing, not implementation, for Phase 1.
- Replacing the `simd_caps()` singleton or the three existing `Tier` enums in one shot. Phase 3 consolidates them; Phases 1–2 use them as-is.
- Nightly intrinsics (`x86_amx_intrinsics`, etc.). The codebase has already chosen inline asm for AMX on stable; we extend that pattern, not bypass it.

---

## Architecture decision: `SimdProfile` + static dispatch tables

### The core problem

The audit's TD-T12 / TD-T13 / TD-T14 — three independent `Tier` enums, each collapsing Skylake-X through Granite Rapids into one `Avx512` bucket. The audit also showed `simd_caps()` (the 20-bit per-feature singleton) already exists and is correct; the gap is that consumers branching on `tier()` get coarse answers, and consumers wanting to specialize on `(BF16, FP16, AMX-INT8, AMX-BF16)` simultaneously have to write 4-deep conditional cascades.

### The architecture

Introduce `SimdProfile` — a flat enum that enumerates *silicon profiles*, each defined as the combination of sub-features that distinguishes it from its neighbors. One profile = one set of best primitives.

```rust
// src/hpc/simd_profile.rs (new file)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdProfile {
    // ── x86_64 profiles ──
    // Ordered by capability superset for readability; detection is explicit.

    /// Sapphire Rapids / Emerald Rapids:
    /// AVX-512F+BW+VL+CD+DQ+VNNI+VBMI+BF16+FP16 + AMX-TILE+INT8+BF16.
    SapphireRapids,
    /// Granite Rapids: SPR + AMX-FP16 + AMX-COMPLEX.
    GraniteRapids,
    /// Ice Lake-SP / Tiger Lake server: AVX-512F+VNNI+VBMI, no BF16/FP16/AMX.
    IceLakeSp,
    /// Cooper Lake: AVX-512F+VNNI+BF16, no VBMI/FP16/AMX.
    /// (Rare in the wild; included for completeness.)
    CooperLake,
    /// Cascade Lake: AVX-512F+VNNI, no VBMI/BF16/FP16/AMX.
    CascadeLake,
    /// Skylake-X / SP / W: AVX-512F only, no VNNI/VBMI/BF16/FP16/AMX.
    SkylakeX,
    /// Zen 4 / Zen 5 (Genoa, Ryzen 7000+, EPYC 9004+):
    /// AVX-512F+VNNI+VBMI+BF16+FP16, no AMX.
    /// Same capability set on Zen 5; we don't distinguish microarch within AMD.
    Zen4Avx512,
    /// Arrow Lake / Lunar Lake / Meteor Lake-H consumer:
    /// AVX2 + AVX-VNNI-INT8 + AVX-IFMA + AVX-NE-CONVERT, no AVX-512.
    ArrowLake,
    /// Haswell through Coffee Lake / Zen 1-3 desktop:
    /// AVX2 + FMA only, no AVX-512.
    HaswellAvx2,

    // ── aarch64 profiles ──

    /// ARMv8.2-A: A76 (Pi 5), Apple M-series, Snapdragon 8 Gen 2+.
    /// NEON + dotprod + fp16 + bf16+ (BFMMLA/BFDOT).
    A76DotProd,
    /// ARMv8.0 fallback: A72 (Pi 4), A53 (Pi 3 / Pi Zero 2 W), any other
    /// ARMv8.0 part. The two cannot be distinguished by HWCAP/CPUID alone
    /// — they expose identical ISA flags (NEON + AES + SHA2 + CRC32, no
    /// dotprod). The difference is microarchitectural (dual vs single
    /// NEON pipeline width). Distinguishing requires reading
    /// /proc/cpuinfo `CPU part` (0xd03 = A53, 0xd08 = A72). Future
    /// improvement: split into A72Fast/A53Baseline once that lookup is
    /// wired. For now, dispatch table is identical for both.
    Armv8Neon,

    // ── Fallback ──
    /// Anything else: wasm32, riscv, x86 baseline, unknown aarch64.
    Scalar,
}
```

### Detection

```rust
impl SimdProfile {
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let caps = simd_caps();
            // Order matters: most-specific superset first.
            if caps.amx_tile && caps.amx_bf16 && caps.avx512fp16
                && /* GNR-specific bit: amx_fp16 if/when detected */ false
            {
                return SimdProfile::GraniteRapids;
            }
            if caps.amx_tile && caps.amx_bf16 {
                return SimdProfile::SapphireRapids;
            }
            if caps.avx512f && caps.avx512vnni && caps.avx512vbmi && caps.avx512bf16 {
                return SimdProfile::Zen4Avx512; // ICX has VBMI but no BF16
            }
            if caps.avx512f && caps.avx512vnni && caps.avx512vbmi {
                return SimdProfile::IceLakeSp;
            }
            if caps.avx512f && caps.avx512vnni && caps.avx512bf16 {
                return SimdProfile::CooperLake;
            }
            if caps.avx512f && caps.avx512vnni {
                return SimdProfile::CascadeLake;
            }
            if caps.avx512f {
                return SimdProfile::SkylakeX;
            }
            if caps.avxvnniint8 {
                return SimdProfile::ArrowLake;
            }
            if caps.avx2 && caps.fma {
                return SimdProfile::HaswellAvx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            let caps = simd_caps();
            if caps.asimd_dotprod && caps.fp16 {
                return SimdProfile::A76DotProd;
            }
            // A72 and A53 expose the same ISA flags (NEON + AES + SHA2 + CRC32,
            // no dotprod/fp16/bf16). They cannot be distinguished by
            // `is_aarch64_feature_detected!` alone — the difference is dual vs
            // single NEON pipeline width, a microarchitectural property not
            // reported by CPUID/HWCAP. Distinguishing requires reading
            // /proc/cpuinfo `CPU part` (0xd03 = A53, 0xd08 = A72) or doing a
            // microbenchmark probe. Until then, collapse both into a single
            // ARMv8.0 profile — the dispatch tables would be identical at the
            // ISA level (same NEON instructions); the only difference would
            // be tile/block sizes tuned to dual vs single pipeline.
            return SimdProfile::Armv8Neon;
        }
        SimdProfile::Scalar
    }
}

static PROFILE: LazyLock<SimdProfile> = LazyLock::new(SimdProfile::detect);

#[inline(always)]
pub fn simd_profile() -> SimdProfile {
    *PROFILE
}
```

### The "switch hashtable" — static dispatch tables per profile

For each primitive family, define a `*Dispatch` struct of function pointers and one `static` table per profile. Profiles share entries where the silicon is identical (e.g. CascadeLake and SkylakeX share the AVX-512F path for ops that don't use VNNI/BF16).

```rust
// src/hpc/gemm_dispatch.rs (new)

pub struct GemmDispatch {
    pub bf16_gemm: fn(&[BF16], &[BF16], &mut [f32], usize, usize, usize, f32, f32),
    pub int8_gemm: fn(&[u8], &[i8], &mut [i32], usize, usize, usize),
    pub f32_gemv: fn(usize, usize, f32, &[f32], usize, &[f32], f32, &mut [f32]),
    // ... etc
}

// One table per silicon profile. Compile-time const, lives in .rodata.

static SPR_GEMM: GemmDispatch = GemmDispatch {
    // TDPBF16PS: 16×16 output tile, K=32 per pass → 16·16·32 = 8192 mul-adds/instr
    // (per `src/hpc/amx_matmul.rs:15` and `bf16_tile_gemm.rs:155-157`).
    bf16_gemm: amx_bf16_tile_gemm,
    // TDPBUSD: 16×16 output tile, K=64 per pass → 16·16·64 = 16384 mul-adds/instr
    // (per `src/hpc/amx_matmul.rs:15`).
    int8_gemm: amx_int8_tile_gemm,
    f32_gemv: avx512_f32x16_gemv,     // shared with all AVX-512 profiles
};
static ICX_GEMM: GemmDispatch = GemmDispatch {
    bf16_gemm: avx512_f32x16_bf16_gemm,  // no native BF16 dot, use F32x16 mul_add
    int8_gemm: avx512vnni_gemm,          // VPDPBUSD, 64 mul-adds/instr
    f32_gemv: avx512_f32x16_gemv,
};
static CPL_GEMM: GemmDispatch = GemmDispatch {
    bf16_gemm: avx512bf16_vdpbf16ps_gemm,  // native VDPBF16PS
    int8_gemm: avx512vnni_gemm,
    f32_gemv: avx512_f32x16_gemv,
};
static CLX_GEMM: GemmDispatch = GemmDispatch {
    bf16_gemm: avx512_f32x16_bf16_gemm,
    int8_gemm: avx512vnni_gemm,
    f32_gemv: avx512_f32x16_gemv,
};
static SKX_GEMM: GemmDispatch = GemmDispatch {
    bf16_gemm: avx512_f32x16_bf16_gemm,
    int8_gemm: scalar_int8_gemm,   // no VNNI on SKX
    f32_gemv: avx512_f32x16_gemv,
};
static ZEN4_GEMM: GemmDispatch = GemmDispatch {
    bf16_gemm: avx512bf16_vdpbf16ps_gemm,
    int8_gemm: avx512vnni_gemm,
    f32_gemv: avx512_f32x16_gemv,
};
static ARROW_GEMM: GemmDispatch = GemmDispatch {
    bf16_gemm: avx2_f32x8_bf16_gemm,
    int8_gemm: avxvnniint8_gemm,
    f32_gemv: avx2_f32x8_gemv,
};
static HSW_GEMM: GemmDispatch = GemmDispatch {
    bf16_gemm: avx2_f32x8_bf16_gemm,
    int8_gemm: scalar_int8_gemm,
    f32_gemv: avx2_f32x8_gemv,
};
static A76_GEMM: GemmDispatch = GemmDispatch {
    bf16_gemm: neon_bfmmla_bf16_gemm,    // ARMv8.2-A BFMMLA
    int8_gemm: neon_dotprod_int8_gemm,   // ARMv8.2-A SDOT
    f32_gemv: neon_f32x4_gemv,
};
static A72_GEMM: GemmDispatch = GemmDispatch {
    bf16_gemm: neon_f32x4_bf16_gemm,
    int8_gemm: neon_f32x4_int8_gemm,    // no SDOT pre-A76
    f32_gemv: neon_f32x4_gemv,
};
static A53_GEMM: GemmDispatch = GemmDispatch { /* same as A72, lower MR */ ..A72_GEMM };
static SCALAR_GEMM: GemmDispatch = GemmDispatch {
    bf16_gemm: scalar_bf16_gemm,
    int8_gemm: scalar_int8_gemm,
    f32_gemv: scalar_gemv,
};

#[inline(always)]
pub fn gemm_dispatch() -> &'static GemmDispatch {
    static TABLE: LazyLock<&'static GemmDispatch> = LazyLock::new(|| {
        match simd_profile() {
            SimdProfile::SapphireRapids | SimdProfile::GraniteRapids => &SPR_GEMM,
            SimdProfile::IceLakeSp => &ICX_GEMM,
            SimdProfile::CooperLake => &CPL_GEMM,
            SimdProfile::CascadeLake => &CLX_GEMM,
            SimdProfile::SkylakeX => &SKX_GEMM,
            SimdProfile::Zen4Avx512 => &ZEN4_GEMM,
            SimdProfile::ArrowLake => &ARROW_GEMM,
            SimdProfile::HaswellAvx2 => &HSW_GEMM,
            SimdProfile::A76DotProd => &A76_GEMM,
            SimdProfile::Armv8Neon => &ARMV8_NEON_GEMM,
            SimdProfile::Scalar => &SCALAR_GEMM,
        }
    });
    *TABLE
}
```

Call site:

```rust
pub fn bf16_gemm_f32(a: &[BF16], b: &[BF16], c: &mut [f32], m: usize, n: usize, k: usize, alpha: f32, beta: f32) {
    (gemm_dispatch().bf16_gemm)(a, b, c, m, n, k, alpha, beta);
}
```

One pointer deref, one indirect call. Branch predictor warms on the first call; every subsequent call is monomorphic to whatever silicon detected at startup. **This is the "switch hashtable" the user asked about.** Conceptually a `HashMap<CpuId, FnPointers>` resolved once.

### Compile-time pinning (the user's "what if native can handle compile time dispatch")

Add cargo features that pin the runtime selection at compile time:

```toml
# Cargo.toml
[features]
# Default: runtime LazyLock dispatch (one binary, all ISAs).
default = ["runtime-dispatch"]
runtime-dispatch = []

# Compile-time pins: pick ONE. Set RUSTFLAGS and skip LazyLock detection.
cpu-spr = []          # implies -Ctarget-cpu=sapphirerapids
cpu-gnr = []          # implies -Ctarget-cpu=graniterapids-d
cpu-icx = []          # implies -Ctarget-cpu=icelake-server
cpu-cpl = []          # implies -Ctarget-cpu=cooperlake
cpu-clx = []          # implies -Ctarget-cpu=cascadelake
cpu-skx = []          # implies -Ctarget-cpu=skylake-avx512
cpu-zen4 = []         # implies -Ctarget-cpu=znver4
cpu-arrowlake = []    # implies -Ctarget-cpu=arrowlake
cpu-haswell = []      # implies -Ctarget-cpu=haswell
cpu-a76 = []          # implies -Ctarget-cpu=cortex-a76 (or -Ctarget-feature=+dotprod,+fp16,+bf16)
```

And in code:

```rust
#[inline(always)]
pub fn gemm_dispatch() -> &'static GemmDispatch {
    #[cfg(feature = "cpu-spr")]
    return &SPR_GEMM;
    #[cfg(feature = "cpu-icx")]
    return &ICX_GEMM;
    #[cfg(feature = "cpu-zen4")]
    return &ZEN4_GEMM;
    // ... etc
    #[cfg(all(feature = "runtime-dispatch", not(any(feature = "cpu-spr", feature = "cpu-icx", feature = "cpu-zen4", /* ... */))))]
    {
        static TABLE: LazyLock<&'static GemmDispatch> = LazyLock::new(|| /* runtime switch */);
        *TABLE
    }
}
```

The `cpu-*` arm is a const reference returned from a `#[inline(always)]` function. After monomorphization the compiler folds the entire dispatch out and inlines the chosen kernel directly at the callsite. **Zero runtime cost. Same source as runtime dispatch.**

The same primitive functions (`amx_bf16_tile_gemm`, `avx512vnni_gemm`, etc.) are referenced by both the static profile tables AND the inline `cpu-*` returns. No duplication.

### Per-function `target_feature` annotations remain mandatory

The dispatch table picks the function pointer; the function itself still needs `#[target_feature(enable = "...")]` for the compiler to emit the right instructions. Pattern continues unchanged:

```rust
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16,avx512vl")]
unsafe fn avx512bf16_vdpbf16ps_gemm(...) {
    // _mm512_dpbf16_ps lives here
}

fn avx512bf16_vdpbf16ps_gemm_wrapper(...) {
    // SAFETY: dispatch verified profile == CooperLake | Zen4Avx512 | SPR | GNR
    unsafe { avx512bf16_vdpbf16ps_gemm(...) }
}
```

The dispatch table holds the safe wrapper; the wrapper relies on profile detection as its safety invariant.

---

## Phase 1 — Wire existing infrastructure (P0)

No new abstractions. Each task routes existing kernels through existing dispatch primitives.

| Task | File | Change | Effort |
|---|---|---|---|
| TD-T1 | `src/hpc/amx_matmul.rs:304-331` | `matmul_bf16_to_f32` AMX arm calls `bf16_tile_gemm_16x16` for 16×16 sub-tiles, `bf16_gemm_f32` for tails. | 1h |
| TD-T2 | `src/hpc/amx_matmul.rs:342-370` | `matmul_f32` AMX arm: convert to BF16, call `bf16_tile_gemm_16x16`. Drop the duplicate `bf16_gemm_f32` line. | 30m |
| TD-T3 | `src/hpc/amx_matmul.rs:386-434` | `matmul_i8_to_i32`: AMX arm wires `tile_dpbusd` (primitives at `amx_matmul.rs:146-150` already work). Non-AMX arm calls `int8_gemm_vnni` (at `vnni_gemm.rs:46`) instead of `int8_gemm_i32`. | 1.5h |
| TD-T5 | `src/hpc/quantized.rs:618-630` | Rename current `int8_gemm_i32` → `int8_gemm_i32_scalar`. Add a public `int8_gemm_i32` that calls `int8_gemm_vnni` (which has its own VNNI/scalar dispatch). | 30m |
| TD-T7 | `src/backend/native.rs:271-278` | Replace `gemv_f32`/`gemv_f64` bodies with `dispatch!` macro invocations. Add `kernels_avx512::gemv_f32/f64` (F32x16 / F64x8 over rows). Add `avx2::gemv_f32/f64` (F32x8 / F64x4). | 2h |
| TD-T6 | `src/backend/native.rs:544-561` | Implement `avx2::{scal,nrm2,asum}_f32/f64` with real `_mm256_*` intrinsics. Pattern matches the existing `dot_f32_avx2` at lines 567-602. | 2h |
| TD-T4 | `src/hpc/quantized.rs:444-481` | Rewrite `bf16_gemm_f32` using F32x16 mul_add over decoded BF16 rows (the fallback pattern already exists in `bf16_tile_gemm.rs::fallback_path` at lines 96-126). Reuse that, don't duplicate. | 3h |

**Phase 1 total: ~10–12h.** Closes 7 of 22 audit findings, all CRITICAL.

After Phase 1: `crate::simd::*` consumers on Sapphire Rapids actually run AMX tile kernels; consumers on Cascade Lake / Ice Lake-SP / Zen 4 actually run VNNI; consumers on AVX2 silicon actually run `_mm256_*` intrinsics for BLAS-1.

---

## Phase 2 — aarch64 fill (P1)

| Task | File | Change | Effort |
|---|---|---|---|
| TD-T10 | `src/simd_neon_bf16.rs:149-204` | Replace `BF16x8Stub` / `BF16x16Stub` with real wrappers backed by `bfloat16x8_t` pairs. Implement `splat`, `from_slice`, `to_array`, `dot_f32` (via BFDOT asm-byte `0x4e40_ec00 | ...`), `cvt_to_f32_lo/hi`, `fma` (via BFMLALB/T or BFMMLA). Mirror `simd_amx.rs` for the asm-byte pattern. Parity tests vs scalar. | 4h |
| TD-T11 | `src/simd_neon_dotprod.rs:115-148` | Replace `F16x16Stub` with `F16x16(pub [float16x8_t; 2])`. Implement the intrinsic map documented at lines 121-131 via asm-byte (`fmla v.8h` = `0x0e40_cc20 \| ...`). Parity tests. | 4h |
| TD-T21 | `src/simd.rs:351-354` | Replace the `scalar::*` integer re-exports on aarch64 with `simd_neon::aarch64_simd_int::*` (new module). Implement `I32x8` (= `int32x4x2_t`), `U8x64` (= `uint8x16x4_t`), `U16x32` (= `uint16x8x4_t`), etc., backed by 128-bit NEON quartets. | 8h |
| TD-T8 | `src/hpc/simd_dispatch.rs:150-163` | Replace `Self::scalar()` aarch64 dispatch with real NEON wrappers. Add `byte_find_all_neon`, `byte_count_neon`, `squared_distances_neon`, `nibble_unpack_neon`, etc. Match the x86_64 wrapper pattern at lines 207-288. | 6h |

**Phase 2 total: ~22h.** Closes 4 findings, all HIGH. Pi 5 / M2 silicon ceiling.

Verification: needs aarch64 hardware (Pi 5 or Apple M-series). The asm-byte encodings can be unit-tested via objdump on a build machine, but throughput / correctness verification needs real hardware. Land behind a CI gate that runs on aarch64 runners (already exists per the `nostd` job).

---

## Phase 3 — `SimdProfile` architecture (P1, addresses the user's intra-bucket question)

| Task | File | Change | Effort |
|---|---|---|---|
| T3.1 | `src/hpc/simd_profile.rs` (new) | Define `SimdProfile` enum + `detect()` per the architecture section above. ~150 LoC. | 3h |
| T3.2 | `src/hpc/simd_profile.rs` | Cargo features `cpu-spr` / `cpu-icx` / `cpu-cpl` / `cpu-clx` / `cpu-skx` / `cpu-zen4` / `cpu-arrowlake` / `cpu-haswell` / `cpu-a76` / `cpu-a72`. `.cargo/config-{profile}.toml` for each. Each pins `target-cpu` and shortcuts `simd_profile()` to a const. | 4h |
| T3.3 | `src/hpc/gemm_dispatch.rs` (new) | First `*Dispatch` struct: `GemmDispatch` with `bf16_gemm`, `int8_gemm`, `f32_gemv`. 12 static tables (one per profile). | 4h |
| T3.4 | `src/hpc/blas1_dispatch.rs` (new) | `Blas1Dispatch` for `dot_f32/f64`, `axpy_f32/f64`, `scal_f32/f64`, `nrm2_f32/f64`, `asum_f32/f64`. | 3h |
| T3.5 | `src/backend/native.rs` | Migrate `dispatch!` macro to consume `simd_profile()` instead of the local `Tier` enum. Delete the local enum. | 2h |
| T3.6 | `src/simd.rs` | Migrate `tier()` to alias `simd_profile().coarse()` (`fn coarse(self) -> Tier`). Preserve existing public callers; let the migration happen incrementally. | 2h |
| T3.7 | `src/hpc/simd_dispatch.rs` | Migrate `detect()` to consume `simd_profile()`. Add the missing dispatch tables (`Avx512f-only`, `AvxVnniInt8`, `IceLakeSp` for VBMI permute paths). | 3h |

**Phase 3 total: ~21h.** Closes TD-T12, T13, T14 (the three Tier enum collapses) and provides the framework for Phase 4.

Migration path is **additive**: `simd_profile()` ships alongside `tier()`, callers opt in, the three local `Tier` enums get deleted last. No big-bang refactor.

---

## Phase 4 — Intra-bucket SIMD fills (P2, parallelizable)

Each entry below is one PR-sized task. They can land independently in any order.

| Task | Profile that unlocks it | What gets faster | File(s) |
|---|---|---|---|
| `crate::simd::BF16x16` native dispatch | CooperLake, Zen4, SPR, GNR | BF16 lane-wise ops, BF16 dot via `vdpbf16ps` | TD-T15 — `src/simd.rs:291-292, 531-532`; convert compile-time gate to runtime via `simd_profile()` |
| F16 native dispatch | Zen4, SPR, GNR (avx512fp16); A76 (fp16) | `F16x16::{add,sub,mul,fma}` via `_mm512_*_ph` or `vaddq_f16` | New `src/simd_fp16.rs` for AVX-512-FP16; extend `simd_neon_dotprod.rs` for aarch64 |
| AVX-VNNI-INT8 ymm GEMM | ArrowLake | int8 GEMM on no-AVX-512 silicon | `src/hpc/vnni_gemm.rs` add `int8_gemm_avxvnniint8_ymm` (256-bit `_mm256_dpbusd_*`) |
| VPOPCNTDQ paths | IceLakeSp, SPR, GNR, Zen4 | Hamming distance, bit counting | Audit `src/hpc/bitwise.rs` for sites currently using avx512bw popcount where `_mm512_popcnt_epi64` exists |
| VBMI byte permute | IceLakeSp, SPR, GNR, Zen4 | Sprite atlas reorder, palette remap > 16 entries | Already wired at `simd_avx512.rs:695` — sweep for other byte-permute sites |
| GFNI bitmatrix multiply | IceLakeSp+, Zen4 | 8×8 bit transpose, palette swaps | New finding — needs audit pass |
| VAES 4×-wide AES | IceLakeSp+, Zen3+ | AES-CTR / AES-GCM batch | Audit `src/hpc` for AES use; may not exist yet |
| Nibble AVX-512BW path | All AVX-512BW | `nibble_unpack`, `nibble_above_threshold` 2× width | TD-T16 — `src/hpc/nibble.rs` |
| Real `_mm256_*` nibble intrinsics | HaswellAvx2 | `nibble_unpack_avx2` proper pshufb | TD-T17 — `src/hpc/nibble.rs:59-94` |
| `simd_ln_f32` Remez polynomial | All AVX-512F | 16-wide log (currently scalar loop) | TD-T18 — `src/simd.rs:479-486` |
| `distance::squared_distances_f32` AVX-512 path | All AVX-512F | 16-wide 3D L2 distance | TD-T19 — `src/hpc/distance.rs:101` |
| `spatial_hash::batch_sq_dist` AVX-512 path | All AVX-512F | same | TD-T20 — `src/hpc/spatial_hash.rs:273` |
| AMX FP16 (when GNR ships) | GraniteRapids | `_tile_dpfp16ps` for FP16 GEMM | New, gated on `caps.amx_fp16` (needs CPUID leaf update in `simd_amx.rs:50`) |

**Phase 4 total: 30–50h, parallelizable.** Each task is small, isolated, and has clear silicon-tier acceptance criteria.

---

## What dispatches where — quick reference

For each named primitive, the silicon-by-silicon route after all 4 phases land:

### `bf16_gemm_f32` (BF16 × BF16 → f32 matmul)

| Profile | Implementation |
|---|---|
| GraniteRapids, SapphireRapids | `tile_dpbf16ps` 16×16×k tile kernel (`hpc/bf16_tile_gemm.rs::amx_path`) |
| Zen4Avx512, CooperLake | `_mm512_dpbf16_ps` over 16-wide BF16 dot-products + f32 accumulate (new) |
| IceLakeSp, CascadeLake, SkylakeX | F32x16 mul_add over decoded BF16 rows (`hpc/bf16_tile_gemm.rs::fallback_path`) |
| ArrowLake, HaswellAvx2 | F32x8 mul_add over decoded BF16 rows (new) |
| A76DotProd | NEON BFMMLA via asm-byte (new in Phase 2 TD-T10) |
| Armv8Neon | NEON F32x4 mul_add over decoded BF16 (new) |
| Scalar | Scalar triple loop (current `quantized.rs:444`) — kept as the reference |

### `int8_gemm_i32` (u8 × i8 → i32 matmul)

| Profile | Implementation |
|---|---|
| GraniteRapids, SapphireRapids | `tile_dpbusd` 16×16×k AMX tile kernel |
| Zen4Avx512, IceLakeSp, CooperLake, CascadeLake | `int8_gemm_vnni_avx512` (existing at `vnni_gemm.rs:94`) |
| SkylakeX | F32x16-accumulate-i32 scalar packing fallback |
| ArrowLake | `_mm256_dpbusd_epi32` (existing `vnni2_dot_u8_i8` at `simd_amx.rs:203`) |
| HaswellAvx2 | Scalar i32 accumulate (no VNNI pre-Cascade Lake) |
| A76DotProd | NEON SDOT (`vdotq_s32`, existing in `simd_neon.rs`) |
| Armv8Neon | NEON int16x8 widen + multiply-accumulate |

### `gemv_f32` (BLAS-2 matrix-vector)

| Profile | Implementation |
|---|---|
| All AVX-512F+ | `kernels_avx512::gemv_f32` over F32x16 row-dot-products (new in Phase 1 TD-T7) |
| ArrowLake, HaswellAvx2 | `avx2::gemv_f32` over F32x8 (new in Phase 1 TD-T7) |
| aarch64 | NEON F32x4 gemv (new in Phase 2) |
| Scalar | Existing `scalar::gemv_f32` |

### `BF16x16::add/sub/mul/fma/dot`

| Profile | Implementation |
|---|---|
| Zen4Avx512, CooperLake, SPR, GNR | Native `__m256bh` via `vdpbf16ps` / scalar conversion |
| All other AVX-512 (SKX, CLX, ICX) | F32x16 upcast → op → BF16 downcast |
| AVX2 | F32x8 upcast → op → BF16 downcast (currently scalar polyfill) |
| A76DotProd | NEON `bfloat16x8_t` BFDOT for dot product, upcast-via-f32 for other ops |
| Scalar | `simd_half::BF16x16` (existing) |

### `F16x16::add/sub/mul/fma`

| Profile | Implementation |
|---|---|
| Zen4Avx512, SPR, GNR (avx512fp16) | Native `__m256h` (Phase 4) |
| Other AVX-512 with F16C | F16C `_mm256_cvtph_ps` upcast → F32x16 op → `_mm256_cvtps_ph` downcast |
| ArrowLake, HaswellAvx2 (F16C) | Same F16C upcast/downcast pattern |
| A76DotProd | Native `float16x8_t` via asm-byte (Phase 2 TD-T11) |
| Scalar | `simd_half::F16x16` (existing) |

---

## Risks and open questions

1. **Cargo feature combinations.** If a user sets `cpu-spr` AND has runtime detection on Zen 4, the binary SIGILLs on AMX instructions. Solution: make `cpu-*` features mutually exclusive via a `compile_error!` in the crate, AND document that `cpu-*` binaries are NOT portable across silicon. The `runtime-dispatch` default stays the safe option.

2. **The `Zen4Avx512` profile question.** AMD's BF16 / FP16 implementation on Zen 4 has different latency than Intel's CPL / SPR. The dispatch picks the same `vdpbf16ps` path on both, but the surrounding tile sizes may need to differ. Out of scope for the initial integration; revisit when benchmarks show divergence.

3. **Detection robustness across hypervisors.** Cloud hypervisors sometimes mask CPUID. The `simd_amx.rs::amx_available()` function already handles this (XCR0 + `arch_prctl` checks). Extend the same defense-in-depth to `simd_profile::detect()` — if a feature is CPUID-reported but XSAVE-state-disabled, demote to the next-best profile.

4. **No GNR-specific detection yet.** `caps.amx_fp16` doesn't exist as a `SimdCaps` field. Add it before TD-T3.1, gated on CPUID.07H.1H:EAX bit 21 (AMX-FP16) — this is a separate leaf from the existing AMX-INT8 / AMX-BF16 bits at leaf 7,0.

5. **Three Tier enums → one SimdProfile.** Phase 3 deletes the local `Tier` enums in `simd.rs`, `backend/native.rs`, and `simd_dispatch.rs`. Migration is additive (new alongside old) until all callers move. Risk of stale `Tier` callers re-emerging during the migration window — mitigate via a deprecation lint.

6. **AVX2 polyfill audit (TD-T22).** The audit deferred verification of whether the 256-bit polyfills in `simd_avx2.rs` are real `__m256i` intrinsics or scalar-storage arrays under `#[target_feature]`. If they're scalar storage, that's a Phase 4 entry: real `__m256i` implementations for the 5 new 256-bit int types (U16x16, U32x8, U64x4, I32x8, I64x4) plus the existing ones.

7. **BLAS-1/2/3 / LAPACK / statistics scalar implementations.** Audit grep showed no `crate::simd::*` use in `blas_level{1,2,3}.rs`, `statistics.rs`, `lapack.rs`, `quantized.rs`. These are flagship public surfaces. Phase 4 needs to either (a) route them through `gemm_dispatch()` / `blas1_dispatch()` / etc., or (b) replace them with `crate::simd::*`-using implementations directly. Decision deferred until the dispatch tables exist.

---

## Sequencing and dependencies

```
Phase 1 (wiring)  ──────────►  ships independently, no deps
Phase 2 (aarch64) ──────────►  ships independently, no deps
Phase 3 (SimdProfile) ──────►  builds on Phase 1 (uses Phase 1 kernels as table entries)
Phase 4 (intra-bucket) ─────►  builds on Phase 3 (each entry is a new profile-specific kernel)
```

Phase 1 and Phase 2 can run in parallel — they touch disjoint files. Phase 3 needs Phase 1's wired kernels as the function pointers in its dispatch tables (otherwise the tables would point at the same scalar stub for every profile). Phase 4 is fully parallelizable; each entry is one PR.

