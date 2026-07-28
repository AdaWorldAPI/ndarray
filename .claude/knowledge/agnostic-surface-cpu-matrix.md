# Agnostic SIMD Surface — Per-CPU Resolution Matrix + Integration Plan

> **Companion to:** `td-simd-cpu-dispatch-matrix.md` (CPU feature presence),
> `td-simd-tier-audit.md` (debt inventory), `td-simd-integration-plan.md`
> (`SimdProfile` architecture). This doc is the **cross-tab**: every public
> primitive in `crate::simd::*` × every CPU profile we target, showing the
> kernel that actually runs on that silicon. Gaps drive the integration plan.

## CPU profile columns (abbreviations)

Same set as `td-simd-cpu-dispatch-matrix.md` § "Master matrix — x86_64" and
§ "aarch64 profiles", with two-letter codes for table width:

| Code | Profile (Cargo cpu / SimdProfile)       | Generation         | Critical features              |
|------|-----------------------------------------|--------------------|--------------------------------|
| SKX  | `skylake-avx512` / `SkylakeX`           | Intel 2017         | AVX-512F+BW+DQ+CD+VL           |
| CLX  | `cascadelake` / `CascadeLake`           | Intel 2019         | + AVX-512 VNNI                 |
| CPL  | `cooperlake` / `CooperLake`             | Intel 2020         | + AVX-512 BF16 (no VBMI)       |
| ICX  | `icelake-server` / `IceLakeSp`          | Intel 2021         | + VBMI, no BF16                |
| SPR  | `sapphirerapids` / `SapphireRapids`     | Intel 2023         | + BF16+FP16+VBMI+AMX-INT8+BF16 |
| GNR  | `graniterapids-d` / `GraniteRapids`     | Intel 2024         | + AMX-FP16                     |
| Z4   | `znver4` / `Zen4Avx512`                 | AMD 2022           | AVX-512 + VNNI+BF16+VBMI       |
| Z5   | `znver5` / `Zen4Avx512` (same dispatch) | AMD 2024           | same as Z4 + minor uarch       |
| ARL  | `arrowlake` / `ArrowLake`               | Intel 2024         | AVX2+FMA + AVX-VNNI+VNNI-INT8  |
| HSW  | `x86-64-v3` / `HaswellAvx2`             | Intel 2013→2021    | AVX2+FMA (no VNNI/AVX-512)     |
| A76  | `cortex-a76` / `A76DotProd`             | ARMv8.2 (Pi 5)     | NEON+dotprod+fp16 (no bf16 / i8mm — those are V8.6+, see § M) |
| A72  | `cortex-a72` / `A72Fast`                | ARMv8.0 (Pi 4)     | NEON only (no dotprod)         |
| A53  | `cortex-a53` / `A53Baseline`            | ARMv8.0 (Pi 3/Z2W) | NEON, lower IPC                |
| SCA  | scalar fallback                         | wasm32/riscv/i686  | no SIMD                        |

Cell legend:

- ✅ `kernel-name`  — wired today, exercises the indicated kernel/intrinsic
- ⏳ `kernel-name`  — kernel exists but **not** dispatched here yet (debt)
- 🟦 `kernel-name`  — planned, no kernel exists yet (new code needed)
- 🟡 polyfill-pass — the call delegates to the polyfilled SIMD *type*; that
   type's per-CPU lowering does the work (transparent dispatch — entry on
   table A)
- ✗ scalar        — falls back to a triple-loop scalar reference
- —               — N/A on this profile

---

## A. Polyfilled SIMD types — backing storage per CPU

The polyfilled types in `crate::simd::*` ARE the CPU DTO surface (per the
session's "polyfill is everything" rule). Consumers write `F32x16`, the
type chooses native storage at compile time. Storage selection is driven
by `target_feature` cfg gates in `src/simd.rs` (lines 221-366).

### Float vectors

| Type     | SKX        | CLX        | CPL        | ICX        | SPR        | GNR        | Z4         | Z5         | ARL        | HSW        | A76        | A72        | A53        | SCA        |
|----------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| `F32x16` | `__m512`   | `__m512`   | `__m512`   | `__m512`   | `__m512`   | `__m512`   | `__m512`   | `__m512`   | 2×`__m256` | 2×`__m256` | 4×`float32x4_t` (paired-load) | 4×`float32x4_t` | 4×`float32x4_t` | `[f32;16]` |
| `F32x8`  | `__m256`   | `__m256`   | `__m256`   | `__m256`   | `__m256`   | `__m256`   | `__m256`   | `__m256`   | `__m256`   | `__m256`   | 2×`float32x4_t` | 2×`float32x4_t` | 2×`float32x4_t` | `[f32;8]`  |
| `F64x8`  | `__m512d`  | `__m512d`  | `__m512d`  | `__m512d`  | `__m512d`  | `__m512d`  | `__m512d`  | `__m512d`  | 2×`__m256d`| 2×`__m256d`| 4×`float64x2_t` | 4×`float64x2_t` | 4×`float64x2_t` | `[f64;8]`  |
| `F64x4`  | `__m256d`  | `__m256d`  | `__m256d`  | `__m256d`  | `__m256d`  | `__m256d`  | `__m256d`  | `__m256d`  | `__m256d`  | `__m256d`  | 2×`float64x2_t` | 2×`float64x2_t` | 2×`float64x2_t` | `[f64;4]`  |

### Half-precision vectors

| Type      | SKX | CLX | CPL                      | ICX | SPR                      | GNR                      | Z4                       | Z5                       | ARL | HSW | A76         | A72 | A53 | SCA |
|-----------|-----|-----|--------------------------|-----|--------------------------|--------------------------|--------------------------|--------------------------|-----|-----|-------------|-----|-----|-----|
| `BF16x16` (avx512bf16) | — | — | `__m256bh` (`simd_avx512`) | — | `__m256bh` | `__m256bh` | `__m256bh` | `__m256bh` | — | — | — | — | — | — |
| `BF16x16` (portable)   | `[u16;16]` | `[u16;16]` | (uses native) | `[u16;16]` | (uses native) | (uses native) | (uses native) | (uses native) | `[u16;16]` | `[u16;16]` | `[u16;16]` 🚨 | `[u16;16]` | `[u16;16]` | `[u16;16]` |
| `BF16x8` (avx512bf16) | — | — | `__m128bh` | — | `__m128bh` | `__m128bh` | `__m128bh` | `__m128bh` | — | — | — | — | — | — |
| `F16x16`              | `[u16;16]` 🚨 | `[u16;16]` 🚨 | `[u16;16]` 🚨 | `[u16;16]` 🚨 | `[u16;16]` 🚨 | `[u16;16]` 🚨 | `[u16;16]` 🚨 | `[u16;16]` 🚨 | `[u16;16]` 🚨 | `[u16;16]` 🚨 | `[u16;16]` 🚨 (has fp16 HW!) | `[u16;16]` | `[u16;16]` | `[u16;16]` |

🚨 = scalar polyfill where hardware exists — see TD-SIMD-8 in
`simd-dispatch-architecture.md` and § F gaps below.

### Integer vectors (lane widths matching the audit's "missing lanes" sweep PR #179)

Storage shape per CPU. "AVX-512" means native `__m512i`; "2×AVX2" means
two `__m256i` halves; "4×NEON" means four 128-bit NEON registers (e.g.
`int8x16x4_t`); "scalar" means `[T; N]` array, no SIMD register.

| Type     | SKX        | CLX | CPL | ICX | SPR | GNR | Z4  | Z5  | ARL        | HSW        | A76        | A72        | A53        | SCA        |
|----------|------------|-----|-----|-----|-----|-----|-----|-----|------------|------------|------------|------------|------------|------------|
| `I8x64`  | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`| 2×`__m256i`| 4×`int8x16_t`  | ←  | ←  | `[i8;64]`  |
| `I8x32`  | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`  | `__m256i`  | 2×`int8x16_t`  | ←  | ←  | `[i8;32]`  |
| `U8x64`  | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`| 2×`__m256i`| 4×`uint8x16_t` | ←  | ←  | `[u8;64]`  |
| `U8x32`  | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`  | `__m256i`  | 2×`uint8x16_t` | ←  | ←  | `[u8;32]`  |
| `I16x32` | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`| 2×`__m256i`| 4×`int16x8_t`  | ←  | ←  | `[i16;32]` |
| `I16x16` | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`  | `__m256i`  | 2×`int16x8_t`  | ←  | ←  | `[i16;16]` |
| `U16x32` | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`⏳| 2×`__m256i`⏳| 4×`uint16x8_t`  | ←  | ←  | `[u16;32]` |
| `U16x16` | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`  | `__m256i`  | 2×`uint16x8_t` | ←  | ←  | `[u16;16]` |
| `I32x16` | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`| 2×`__m256i`| 4×`int32x4_t`  | ←  | ←  | `[i32;16]` |
| `I32x8`  | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`  | `__m256i`  | 2×`int32x4_t`  | ←  | ←  | `[i32;8]`  |
| `U32x16` | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`⏳| 2×`__m256i`⏳| 4×`uint32x4_t` | ←  | ←  | `[u32;16]` |
| `U32x8`  | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`⏳ | `__m256i`⏳ | 2×`uint32x4_t` | ←  | ←  | `[u32;8]`  |
| `I64x8`  | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`| 2×`__m256i`| 4×`int64x2_t`  | ←  | ←  | `[i64;8]`  |
| `I64x4`  | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`  | `__m256i`  | 2×`int64x2_t`  | ←  | ←  | `[i64;4]`  |
| `U64x8`  | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`| 2×`__m256i`| 4×`uint64x2_t` | ←  | ←  | `[u64;8]`  |
| `U64x4`  | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`  | `__m256i`  | 2×`uint64x2_t` | ←  | ←  | `[u64;4]`  |

⏳ = TD-T22 polyfill audit — the 256-bit `U16x16/U16x32/U32x8/U32x16`
inner ops may currently use scalar storage under `#[target_feature]` rather
than real `__m256i` intrinsics. Needs verification (see § J integration plan).

> **⏳ RESOLVED — TD-T22 CLOSED, no gap (2026-07-28).** The audit is done and
> the answer is: the storage IS scalar in the SOURCE, and that costs nothing.
> `.cargo/config.toml` pins `-Ctarget-cpu=x86-64-v3` for every x86_64 build,
> so LLVM auto-vectorizes the `avx2_int_type!` loop bodies into packed AVX2.
> Measured on the ChaCha20 ARX triple over `U32x16`: **zero scalar ALU
> instructions**, `rotate_left(16)` strength-reduced to `vpshufb`, and the
> 10-round double-round loop emits exactly **8 `vpaddd` for 64 u32 lanes** —
> the AVX2 instruction-count floor, with no headroom a hand-written
> `__m256i` version could recover. `reduce_sum` emits a logarithmic
> `vpaddd`/`vpshufd`/`vextracti128` reduction tree, not the scalar fold its
> source spells out. The float side matches: `F32x16::mul_add` is the same
> `to_array` → loop → `from_array` shape and emits real `vfmadd213ps`.
>
> **So these ⏳ cells are ACCURATE AS WRITTEN and must not be read as a
> performance defect.** A lowering can only be justified by `repr(align(64))`
> cacheline guarantees (which the polyfill already has and a
> `repr(transparent)` wrapper would LOSE), non-inlined ABI shape, or
> `opt-level`/LLVM-version independence — never by speed.
>
> Full artifact with probe source, exact commands, and per-symbol instruction
> histograms: `.claude/knowledge/td-t22-asm-investigation.md`.

### Mask vectors

| Type      | SKX/CLX/CPL/ICX/SPR/GNR/Z4/Z5 | HSW/ARL | A76/A72/A53 | SCA |
|-----------|-------------------------------|---------|-------------|-----|
| `F32Mask16` | `__mmask16` (1 bit per lane) | `__m256i` (two-half mask) | 4×`uint32x4_t` (lane-mask) | `[bool;16]` |
| `F32Mask8`  | `__mmask8`  | `__m256i` (one-half mask) | 2×`uint32x4_t` | `[bool;8]`  |
| `F64Mask8`  | `__mmask8`  | `__m256i` (two-half mask) | 4×`uint64x2_t` | `[bool;8]`  |
| `F64Mask4`  | `__mmask8`  | `__m256i` (one-half mask) | 2×`uint64x2_t` | `[bool;4]`  |

### Critical type-method per-CPU lowerings (where it matters)

Most methods (add, sub, mul, div, simd_lt, etc.) just delegate to the
storage's native op. The non-obvious lowerings:

| Method                  | SKX        | CLX        | CPL        | ICX        | SPR        | GNR        | Z4 | Z5 | ARL        | HSW        | A76        | A72        | A53        | SCA              |
|-------------------------|------------|------------|------------|------------|------------|------------|----|----|------------|------------|------------|------------|------------|------------------|
| `F32x16::mul_add`       | `vfmadd231ps zmm` | ← | ← | ← | ← | ← | ← | ← | 2×`vfmadd231ps ymm` (FMA3) | 2×`vfmadd231ps ymm` | 4×`vfmaq_f32` | 4×`vfmaq_f32` | 4×`vfmaq_f32` | `f32::mul_add`   |
| `F64x8::mul_add`        | `vfmadd231pd zmm` | ← | ← | ← | ← | ← | ← | ← | 2×`vfmadd231pd ymm` | 2×`vfmadd231pd ymm` | 4×`vfmaq_f64` | 4×`vfmaq_f64` | 4×`vfmaq_f64` | `f64::mul_add`   |
| `F32x16::simd_min/max`  | `vminps/vmaxps zmm` | ← | ← | ← | ← | ← | ← | ← | 2×`vminps/vmaxps ymm` | 2×`vminps/vmaxps ymm` | 4×`vminq/vmaxq_f32` | ← | ← | scalar loop      |
| `F32x16::reduce_sum`    | `vaddps` + `_mm512_reduce_add_ps` ladder | ← | ← | ← | ← | ← | ← | ← | ymm reduce ladder | ymm reduce ladder | NEON paired-add ladder | ← | ← | iter sum         |
| `simd_exp_f32`          | Remez poly (F32x16) | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | (lib expects F32x16 from polyfill — currently no scalar override; scalar reduces lane-by-lane) |
| `simd_ln_f32`           | scalar `f32::ln` per lane 🚨 | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← (TD-T18 in audit — no SIMD path on any backend) |

---

## B. `simd_ops` — float slice ops

All `simd_ops` slice functions are written **once** against the
polyfilled types (`F32x16`, `F64x8`) and inherit their per-CPU lowering.
The 🟡 cells indicate "transparent polyfill dispatch — see table A".

| Function             | SKX–GNR/Z4/Z5/ARL/HSW   | A76/A72/A53           | SCA               | Notes |
|----------------------|-------------------------|-----------------------|-------------------|-------|
| `add_f32`            | 🟡 F32x16 + scalar tail | 🟡                    | 🟡 + scalar tail  | binary_f32 helper |
| `sub_f32`            | 🟡                      | 🟡                    | 🟡                |       |
| `mul_f32`            | 🟡                      | 🟡                    | 🟡                |       |
| `div_f32`            | 🟡                      | 🟡                    | 🟡                |       |
| `add_f32_inplace`    | 🟡                      | 🟡                    | 🟡                | inplace_f32 helper |
| `sub_f32_inplace`    | 🟡                      | 🟡                    | 🟡                |       |
| `mul_f32_inplace`    | 🟡                      | 🟡                    | 🟡                |       |
| `div_f32_inplace`    | 🟡                      | 🟡                    | 🟡                |       |
| `scale_f32`          | 🟡                      | 🟡                    | 🟡                | F32x16::mul broadcast |
| `add_scalar_f32`     | 🟡                      | 🟡                    | 🟡                | F32x16::add broadcast |
| `scale_f32_inplace`  | 🟡                      | 🟡                    | 🟡                |       |
| **`add_mul_f32`** ✅ | 🟡 F32x16::mul_add + scalar tail (f32::mul_add) | 🟡 | 🟡 | NEW (this session) — FMA into accumulator |
| `add_f64`            | 🟡 F64x8 + scalar tail  | 🟡                    | 🟡                | binary_f64 helper |
| `mul_f64`            | 🟡                      | 🟡                    | 🟡                |       |
| `add_f64_inplace`    | 🟡                      | 🟡                    | 🟡                |       |
| **`add_mul_f64`** ✅ | 🟡 F64x8::mul_add + scalar tail (f64::mul_add)  | 🟡 | 🟡 | NEW (this session) |
| `array_chunks`       | uniform — `slice::as_chunks` (stable) | uniform | uniform | const-size **non-overlapping** |
| `array_chunks_checked` | uniform                | uniform               | uniform           |       |
| **`array_windows`** ✅  | uniform — index-based iter | uniform              | uniform           | NEW (this session) — const-size **overlapping** |
| **`array_windows_checked`** ✅ | uniform           | uniform               | uniform           | NEW (this session) |

**Gap:** none — every `simd_ops` surface ride on the polyfill primitives.
Floats are the well-served path. Any speedup at this layer requires the
polyfilled types themselves to expose a faster primitive (e.g. a `dpbusd`
op on `I32x16`, see § J integration plan Phase 4).

---

## C. `simd_int_ops` — integer slice ops

| Function           | SKX        | CLX        | CPL        | ICX        | SPR        | GNR        | Z4         | Z5         | ARL        | HSW        | A76        | A72        | A53        | SCA |
|--------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|-----|
| `add_i8` ✅ MX-T1a | ✅ `_mm512_add_epi8` via `I8x64` | ← | ← | ← | ← | ← | ← | ← | ✅ `_mm256_add_epi8` ×2 via `I8x64` polyfill | ← | ✅ `vaddq_s8` via `I8x16` | ← | ← | ✅ scalar wrapping_add |
| `sub_i8` ✅ MX-T1a | ✅ `_mm512_sub_epi8`         | ← | ← | ← | ← | ← | ← | ← | ✅ `_mm256_sub_epi8` ×2          | ← | ✅ `vsubq_s8`            | ← | ← | ✅ scalar wrapping_sub |
| `add_i16` ✅ MX-T1a| ✅ `_mm512_add_epi16` via `I16x32` | ← | ← | ← | ← | ← | ← | ← | ✅ `_mm256_add_epi16` via `I16x32` polyfill | ← | ✅ `vaddq_s16` via `I16x8` | ← | ← | ✅ scalar wrapping_add |
| `dot_i8`           | ✗ scalar 🚨 | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ✗ |
| `dot_i16`          | ✗ scalar 🚨 | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ←         | ✗ |
| `min_i8`           | ✅ `vpminsb zmm` via I8x64 | ← | ← | ← | ← | ← | ← | ← | ✅ `vpminsb ymm` via I8x32 polyfill of I8x64 | ← | ✅ `vminq_s8` via I8x16 | ← | ← | ✗ scalar loop |
| `max_i8`           | ✅ `vpmaxsb zmm` via I8x64 | ← | ← | ← | ← | ← | ← | ← | ✅ `vpmaxsb ymm` | ← | ✅ `vmaxq_s8`        | ← | ← | ✗ |
| **`gemm_u8_i8`** ✅ | ✗ scalar (no VNNI) | ✅ `vpdpbusd zmm` (CLX+) | ← | ← | ← | ← | ← | ← | ✅ `vpdpbusd ymm` (avxvnni) | ✗ scalar | 🟦 `sdot+128-bias` (planned) | ✗ scalar | ✗ scalar | ✗ scalar |
| `gemm_u8_i8` AMX preempt | — | — | — | — | 🟦 `tdpbusd` 16×16 tile (planned) | 🟦 `tdpbusd` | — | — | — | — | — | — | — | — |

🚨 = scalar where SIMD exists. Each of these has 16-wide `I8x64::add` etc.
already in the polyfill but the slice ops don't reach for them. Trivial fix
once we decide to land an int-slice-ops sweep — see § J Phase 1b.

---

## D. `simd_half` — BF16 / F16 ops

The half-precision surface is **uniformly scalar** today: every op upcasts
to f32 lane-by-lane, computes, downcasts back via round-to-nearest-even.
This is TD-SIMD-8 in the audit — hardware paths exist on every CPU class
but only one (`BF16x16` on avx512bf16) is wired.

| Function                  | SKX        | CLX        | CPL        | ICX        | SPR        | GNR        | Z4         | Z5         | ARL        | HSW        | A76        | A72        | A53        | SCA |
|---------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|-----|
| `BF16x16::from_slice`     | uniform — `[u16;16]` load | ← | ← | ← | ← (native `__m256bh` swap-in) | ← | ← (native) | ← (native) | ← | ← | ← | ← | ← | ← |
| `BF16x16::add/sub/mul`    | 🚨 scalar f32 upcast | ← | ⏳ `vdpbf16ps`-able via F32x16 mul | ← | ⏳ ditto + AMX-BF16 tile | ← | ⏳ | ⏳ | 🚨 scalar | 🚨 scalar | 🚨 scalar (BFMLALB-able) | 🚨 scalar | 🚨 scalar | 🚨 scalar |
| `BF16x16::fma`            | 🚨 scalar f32 mul_add | ← | ⏳ `vdpbf16ps zmm` | ← | ⏳ AMX-BF16 / VDPBF16PS | ← | ⏳ VDPBF16PS | ⏳ | 🚨 scalar | 🚨 scalar | 🚨 scalar (BFMMLA-able) | 🚨 | 🚨 | 🚨 |
| `BF16x16::to_f32x16`      | 🚨 scalar bit-shift | ← | ⏳ `vcvtne2ps2bf16` reverse | ← | ⏳ | ⏳ | ⏳ | ⏳ | 🚨 scalar | 🚨 | 🚨 (BFCVTN-able) | 🚨 | 🚨 | 🚨 |
| `F16x16::add/sub/mul`     | 🚨 scalar | ← | ← | ← | ⏳ `vmulph zmm` (avx512fp16) | ← | ⏳ avx512fp16 | ⏳ | 🚨 | 🚨 | 🚨 (FMLA `v.8h`) | 🚨 | 🚨 | 🚨 |
| `F16x16::fma`             | 🚨 scalar mul_add | ← | ← | ← | ⏳ `vfmadd231ph zmm` | ← | ⏳ | ⏳ | 🚨 | 🚨 | 🚨 (FMLA `v.8h`) | 🚨 | 🚨 | 🚨 |
| `F16x16::to_f32x16`       | 🚨 scalar | ← | ← | ← | ← (could use F16C `vcvtph_ps` for ymm halves on every x86 from Ivy Bridge — TD-SIMD-8 misses this on ALL profiles) | ← | ← | ← | 🚨 | 🚨 (F16C wired-able) | 🚨 (`vcvt_f32_f16`) | 🚨 | 🚨 | 🚨 |
| `add_bf16_inplace`        | 🟡 BF16x16 + scalar tail (inherits whatever BF16x16::add does) | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← |
| `mul_bf16_inplace`        | 🟡 BF16x16 | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← |
| `add_f16_inplace`         | 🟡 F16x16  | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← |
| `mul_f16_inplace`         | 🟡 F16x16  | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← |
| `cast_bf16_to_f32_batch`  | 🟡 BF16x16::to_f32x16 + tail | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← |
| `cast_f16_to_f32_batch`   | 🟡 F16x16::to_f32x16  | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← |
| `cast_f32_to_bf16_batch`  | ✗ scalar per-element 🚨 | ← | ⏳ should call `f32_to_bf16_batch_rne` (already exists for AVX-512) | ← | ⏳ AMX-BF16 / `vcvtne2ps2bf16` | ← | ⏳ | ⏳ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| `cast_f32_to_f16_batch`   | ✗ scalar per-element 🚨 | ← | ← | ← | ⏳ `vcvtps2phx zmm` (avx512fp16) | ← | ⏳ | ⏳ | ✗ (F16C wired-able) | ✗ (F16C) | ✗ (`vcvt_f16_f32`) | ✗ | ✗ | ✗ |

**Gap, severe.** F16/BF16 is the AI/ML hot path and the entire surface is
scalar-equivalent on every CPU. Even where F16C has been stable since 2012
(Ivy Bridge) the dispatch doesn't reach for it. Phases F1–F3 in the
integration plan below.

---

## E. Batch converters + transcendentals (`crate::simd::*` direct)

These don't go through the polyfilled types — they're standalone
functions in `src/simd.rs` and `src/simd_avx512.rs`.

| Function                          | SKX        | CLX        | CPL        | ICX        | SPR        | GNR        | Z4         | Z5         | ARL        | HSW        | A76        | A72        | A53        | SCA |
|-----------------------------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|-----|
| `bf16_to_f32_batch`               | ✅ scalar batch via `<< 16` cast | ← | ✅ same | ← | ✅ same | ← | ✅ | ✅ | ✅ | ✅ | ✅ (NEON-batchable, currently scalar) | ✅ | ✅ | ✅ |
| `bf16_to_f32_scalar`              | uniform — scalar reference | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← |
| `f32_to_bf16_batch`               | ✅ scalar truncate (no rounding) | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← |
| `f32_to_bf16_scalar`              | uniform — scalar reference | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← |
| **`f32_to_bf16_batch_rne`**       | ✅ AVX-512-F bit-fiddle (no avx512bf16 dep!) 500–20000× faster than scalar; byte-exact vs `_mm512_cvtneps_pbh` | ← | ← | ← | ← | ← | ← | ← | ✗ scalar 🚨 (uses AVX-512-F-only ops on byte loads — could be lifted to AVX2 in principle) | ✗ scalar 🚨 | ✗ scalar 🚨 | ✗ scalar | ✗ scalar | ✗ scalar |
| `f32_to_bf16_scalar_rne`          | uniform — reference impl, must NOT be in hot loops | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← |
| `simd_exp_f32`                    | ✅ Remez poly via F32x16 | ← | ← | ← | ← | ← | ← | ← | ✅ (lower lane count via F32x16 polyfill of two ymm) | ✅ same | ✅ | ✅ | ✅ | ✗ scalar |
| `simd_ln_f32`                     | ✗ scalar `f32::ln` per lane on ALL profiles 🚨 | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← | ← |

---

## F. `simd_soa` — SoA carriers (`MultiLaneColumn`)

Layout-only. Every method is uniform across CPUs — the per-CPU dispatch
lives inside the polyfilled types returned by `iter_u8x64` / `iter_f32x16`
/ `iter_f64x8` / `iter_u64x8`. See table A.

| Method                | Behavior across all CPUs                                            |
|-----------------------|---------------------------------------------------------------------|
| `MultiLaneColumn::new`| `Arc<[u8]>` carrier validation (multiple-of-64 byte buffer)         |
| `len_*` / `is_empty`  | u64 arithmetic on `Arc.len()`                                       |
| `iter_u8x64`          | `as_chunks::<64>` + `U8x64::from_array` (delegates to polyfill)     |
| `iter_f32x16`         | `as_chunks::<64>` + per-chunk `f32::from_le_bytes` × 16 + `from_array` |
| `iter_f64x8`          | `as_chunks::<64>` + per-chunk `f64::from_le_bytes` × 8 + `from_array`  |
| `iter_u64x8`          | `as_chunks::<64>` + per-chunk `u64::from_le_bytes` × 8 + `from_array`  |
| `as_bytes`            | Arc-aliased `&[u8]` view                                            |

**Gap:** none at this layer — gaps in the polyfilled types propagate
transparently, gain from filling them is automatic.

---

## G. Cognitive / HPC re-exports surfaced through `crate::simd::*`

These are re-exports of functions that themselves use `crate::simd::*` —
their per-CPU resolution is the polyfill's, but they're listed here for
inventory completeness since they appear in the public `crate::simd::*` API.

| Symbol                                                          | Behavior across CPUs |
|-----------------------------------------------------------------|---------------------|
| `Fingerprint{,1K,2K,64K}`, `VectorConfig`, `VectorWidth`        | 🟡 polyfill-pass (uses F32x16 / U64x8 internally) |
| `hamming_distance_raw`, `popcount_raw`                          | TD-T-? — needs audit. AVX-512 VPOPCNTDQ wiring partially landed. |
| `wht_f32`, `wht_f32_new`                                        | 🟡 polyfill-pass (uses F32x16) |
| `CollapseGate`                                                  | 🟡 polyfill-pass |
| `kmeans`, `squared_l2`                                          | 🟡 polyfill-pass (uses F32x16) |
| `cosine_f32_to_f64_simd` (heel_f64x8)                           | 🟡 polyfill-pass (uses F64x8 + F32x16) |
| `quantize_f32_to_{i2,i4,i8}`, `dequantize_{i2,i4,i8}_to_f32`    | TD-? — needs audit. Likely scalar today. |
| `QuantParams`                                                   | data carrier, no per-CPU divergence |
| `MultiLaneColumn`                                               | covered in § F |
| `array_chunks` / `array_windows`                                | covered in § B |
| `add_f32` / … / `add_mul_f32` / `add_mul_f64`                   | covered in § B |
| `add_bf16_inplace`, `cast_*_batch`, `BF16x16`, `F16x16`         | covered in § D |

---

## H. Currently-MISSING agnostic surfaces (mentioned in integration plans but not yet present)

Things we know we want but haven't built yet — sourced from the audit
+ integration plan + dispatch matrix companions.

| Symbol                                  | Purpose                                              | Currently |
|-----------------------------------------|------------------------------------------------------|-----------|
| `simd_int_ops::gemm_i8` (s8 × s8 → i32) | True symmetric VNNI2 surface (Arrow Lake / GNR `vpdpbssd`) | ✗ missing |
| `simd_int_ops::gemm_u8`  (u8 × u8 → u32) | Symmetric unsigned VNNI2 (`vpdpbuud`)                | ✗ missing |
| `simd_int_ops::dot4_u8_i8` (vector op)  | The polyfilled dot-4 primitive on `I32x{8,16}`       | ✗ missing |
| `simd_ops::axpy_f32` (scalar α)         | BLAS-1 `y += α * x` (different from `add_mul_f32`'s vector β) | ✗ missing |
| `simd_ops::dot_f32`                     | BLAS-1 f32 dot product                               | ✗ missing |
| `simd_ops::nrm2_f32`, `asum_f32`        | BLAS-1 vector norms                                  | ✗ missing |
| `simd_ops::gemv_f32`                    | BLAS-2 matrix-vector (currently TD-T7 scalar)        | ✗ missing |
| `simd_ops::gemm_f32`                    | BLAS-3 (currently uses `matrixmultiply` workspace)   | ✗ deferred — `matrixmultiply` is the production path |
| `simd_int_ops::dot_i32` / `dot_i32_i64` | INT32 dot, INT16×INT16→INT32 via VPDPWSSD            | ✗ missing |
| `SimdProfile` enum + `simd_profile()`   | Phase 3 dispatch foundation per integration plan      | ✗ missing |
| `cpu-spr` / `cpu-zen4` / etc. features  | Compile-time pin cargo features (integration plan)    | ✗ missing |

---

## I. Cross-cutting infrastructure status

| Item                                            | Status        |
|-------------------------------------------------|---------------|
| **`.cargo/config.toml`** default `x86-64-v3`   | ✅ (CI baseline) |
| **`.cargo/config-avx512.toml`** = `sapphirerapids` | ✅ (this session) |
| **`.cargo/config-native.toml`** = `native`     | ✅ already in tree |
| **`.cargo/config-apple-m2.toml`**              | ✅ in tree    |
| **`.cargo/config-pi5.toml`** (A76+)            | ✅ in tree    |
| **`.cargo/config-graviton.toml`** (A72/A76 AWS)| ✅ in tree    |
| Cargo features `cpu-spr` / `cpu-icx` / `cpu-zen4` / etc. | ✗ missing (Phase 3) |
| Cargo feature `runtime-dispatch` (LazyLock-once table) | ✗ missing (Phase 3) |
| `SimdProfile` enum                              | ✗ missing (Phase 3) |
| GitHub CI matrix (default v3, nightly-simd, avx512, aarch64) | ✅ partial — verified per CI doc |
| Bench harness for `gemm_u8_i8`                  | ✅ this session (ignored test) |
| Bench harness for BF16 / F16 ops                | ✗ missing    |
| Bench harness for `simd_ops` slice ops          | ✗ missing    |

---

## J. INTEGRATION PLAN

Filling the matrix in deliberate phases. Each item is one PR-sized unit.

### Phase 0 — Already landed (this session)

- ✅ `simd_int_ops::gemm_u8_i8` agnostic surface with `avx512vnni` / `avxvnni` / scalar arms (compile-time cfg chain).
- ✅ `int8_gemm_avxvnni_ymm` kernel (VEX `vpdpbusd` ymm).
- ✅ `int8_gemm_vnni_avx512` promoted to `pub(crate)` for direct dispatcher call.
- ✅ `.cargo/config-avx512.toml` → `sapphirerapids` (was bare v4 without VNNI).
- ✅ `simd_ops::array_windows` + `array_windows_checked` (overlapping const-size).
- ✅ `simd_ops::add_mul_f32` + `add_mul_f64` (slice-level FMA, polyfill-routed).
- ✅ "Foundation primitives — do not remove" doc-callout in `simd_ops.rs`.
- ✅ Bench harness (`bench_gemm_u8_i8_vs_scalar`, `#[ignore]`'d).
- ✅ MX-T1a — `add_i8` / `sub_i8` / `add_i16` lifted from scalar to polyfilled
  `I8x64` / `I8x16` / `I16x32` / `I16x8` (matrix § C cells flipped).

### Design rule for AMX / F16 / FP16 paths: inline asm-byte encoding

> **Hard constraint for Phases 1b (AMX-INT8), 3b (AVX-512-FP16),
> 3c (NEON BF16+FP16), 4d (AMX-FP16):** every instruction that lacks
> stable Rust intrinsics on the project's pinned 1.95 stable toolchain
> MUST be emitted via raw-`.byte`-string inline asm, matching the
> pattern already proven in `src/simd_amx.rs` (lines 16-19 of its
> module docs). Rationale:
>
> 1. **AMX intrinsics are nightly-only** (Rust issue #126622). The
>    project pins Rust 1.95 stable per `CLAUDE.md` line 9. The
>    existing `simd_amx.rs` lifts AMX onto stable today via
>    `asm!(".byte 0xc4, 0xe2, 0x7b, 0x49, 0xc0", options(nostack, nomem))`
>    for TILEZERO and equivalent encodings for TDPBUSD / TDPBF16PS.
> 2. **AVX-512-FP16 intrinsics** (`_mm512_add_ph`, `_mm512_fmadd_ph`,
>    `vcvtph2ps`/`vcvtps2ph` zmm forms) — historically have had
>    stabilization churn. Asm-byte encoding skips the version dance.
> 3. **NEON FP16** (FMLA `v.8h`, BFDOT, BFMMLA, USDOT) — likewise
>    nightly-gated for several Rust releases. The existing
>    `simd_neon_bf16.rs` and `simd_neon_dotprod.rs` stub files (TD-T10
>    / TD-T11) are placeholders meant to be filled with asm-byte
>    encodings per the same pattern.
>
> Concrete recipe:
>
> ```rust
> #[cfg(target_arch = "x86_64")]
> #[target_feature(enable = "amx-tile,amx-int8")]
> unsafe fn tdpbusd_t0_t1_t2() {
>     // TDPBUSD tmm0, tmm1, tmm2 — opcode VEX C4 E2 73 5E C1
>     // 5E = TDPBUSD, prefix bits = unsigned-by-signed selector
>     // C1 = ModR/M (tmm0 dest, tmm1 src1, tmm2 src2 via /r encoding)
>     // The byte sequence is the canonical VEX form documented in
>     // Intel SDM Vol. 2D § TDPBUSD; verify with `objdump -d` of a
>     // gas-assembled stub the first time it lands.
>     core::arch::asm!(
>         ".byte 0xc4, 0xe2, 0x73, 0x5e, 0xc1",
>         options(nostack, nomem)
>     );
> }
> ```
>
> Same pattern for NEON F16:
>
> ```rust
> #[cfg(target_arch = "aarch64")]
> #[target_feature(enable = "neon,fp16")]
> unsafe fn fmla_v8h(_acc: &mut float16x8_t, _a: float16x8_t, _b: float16x8_t) {
>     // FMLA v0.8h, v1.8h, v2.8h — encoding 0x0e40_cc20 | (Rd << 0) | (Rn << 5) | (Rm << 16)
>     // Same byte-encoded pattern as simd_amx.rs uses for AMX on x86.
>     core::arch::asm!(
>         ".inst 0x0e42cc20",   // FMLA v0.8h, v1.8h, v2.8h
>         options(nostack, nomem)
>     );
> }
> ```
>
> **Verification harness:** each newly-encoded instruction lands with an
> `objdump -d` check in the doc-comment showing the gas-disassembly
> matches the intended mnemonic. The first such verification in this
> project is recorded in `simd_amx.rs:16-19` ("verified working" line).
>
> **What this rule does NOT apply to:** instructions with already-stable
> intrinsics on Rust 1.95 — `_mm512_dpbusd_epi32` (avx512vnni),
> `_mm256_dpbusd_avx_epi32` (avxvnni), `_mm256_cvtph_ps` (F16C),
> `_mm512_cvtne2ps2bf16` (avx512bf16). Those continue to use the
> intrinsics directly per the existing `simd_avx512.rs` patterns.

### Phase 1 — Wire what already exists (highest ROI per audit)

P0 — closes 7 of 22 audit findings. From `td-simd-integration-plan.md` Phase 1, refined with this matrix's findings:

| Task    | Surface affected                | Change | Effort |
|---------|--------------------------------|--------|--------|
| TD-T1   | `hpc::amx_matmul::matmul_bf16_to_f32` | Route AMX arm through `bf16_tile_gemm_16x16` instead of scalar `bf16_gemm_f32` | 1h |
| TD-T2   | `hpc::amx_matmul::matmul_f32`  | AMX arm: convert to BF16, call tile kernel — drop duplicate scalar call | 30m |
| TD-T3   | `hpc::amx_matmul::matmul_i8_to_i32` | AMX arm wires `tile_dpbusd`; non-AMX arm uses `int8_gemm_vnni` instead of scalar | 1.5h |
| TD-T4   | `hpc::quantized::bf16_gemm_f32` | Rewrite using `F32x16::mul_add` over decoded BF16 rows | 3h |
| TD-T6   | `backend::native::avx2::{scal,nrm2,asum}_f32/f64` | Replace scalar delegations with real `_mm256_*` intrinsics | 2h |
| TD-T7   | `backend::native::gemv_f32/f64` | Wire through `dispatch!` macro to AVX-512/AVX2 row-dot kernels | 2h |

**Plus from this matrix (new):**

| Task    | Surface affected               | Change | Effort |
|---------|--------------------------------|--------|--------|
| MX-T1   | `simd_int_ops::{add_i8, sub_i8, add_i16, dot_i8, dot_i16}` | Lift from scalar to polyfilled `I8x{32,64}` / `I16x{16,32}` ops. They already exist as types on every backend; just route the slice ops through them. | 3h |
| MX-T2   | `simd::cast_f32_to_bf16_batch` | Currently scalar — route to existing `f32_to_bf16_batch_rne` (AVX-512-F-only; works on every AVX-512 CPU) when available, scalar otherwise. | 30m |
| MX-T3   | `simd::cast_f32_to_f16_batch`  | Add F16C (`vcvtps2ph`) fast path — stable since 2012 Ivy Bridge — currently scalar on every x86 profile. | 2h |

**Phase 1 total: ~15–18h.** Closes all 7 CRITICAL audit findings plus the
three new "low-hanging integer/cast" wins surfaced here.

### Phase 2 — aarch64 fills (Pi 5 / Apple M-series silicon ceiling)

From `td-simd-integration-plan.md` Phase 2, restated:

| Task    | Surface | Change | Effort |
|---------|---------|--------|--------|
| TD-T10  | `simd_neon_bf16::BF16x{8,16}Stub` → real `bfloat16x8_t` pairs, BFDOT via asm-byte, BFMMLA wiring | Live BF16 NEON arithmetic | 4h |
| TD-T11  | `simd_neon_dotprod::F16x16Stub` → real `float16x8_t` pair via asm-byte FMLA `v.8h` | Live FP16 NEON arithmetic | 4h |
| TD-T21  | `simd::*` aarch64 integer re-exports (currently scalar polyfill from `simd_scalar::*`) → real NEON quartets | Live integer NEON for I32x8, U8x64 etc. | 8h |
| TD-T8   | `hpc::simd_dispatch` aarch64 dispatch — currently `Self::scalar()` → real NEON wrappers | byte_find_all_neon, byte_count_neon, … | 6h |
| MX-T4   | `simd_int_ops::gemm_u8_i8` NEON arm | New `int8_gemm_sdot_neon` kernel using `vdotq_s32` + +128-bias for u8×i8 | 4h |

**Phase 2 total: ~26h.** Requires aarch64 CI runner / cross-compile verification (Pi 5 or Apple M-series).

### Phase 3 — `SimdProfile` dispatch foundation

From `td-simd-integration-plan.md` Phase 3 — unchanged:

| Task    | Surface | Change | Effort |
|---------|---------|--------|--------|
| T3.1   | `src/hpc/simd_profile.rs` (new) | `SimdProfile` enum + `detect()` per dispatch matrix | 3h |
| T3.2   | `Cargo.toml` features + `.cargo/config-{profile}.toml` per silicon profile | `cpu-spr`, `cpu-icx`, …, mutually exclusive | 4h |
| T3.3   | `src/hpc/gemm_dispatch.rs` (new) | First `*Dispatch` table — `bf16_gemm`, `int8_gemm`, `f32_gemv` | 4h |
| T3.4   | `src/hpc/blas1_dispatch.rs` (new) | `Blas1Dispatch` for dot/axpy/scal/nrm2/asum f32/f64 | 3h |
| T3.5   | `backend::native::dispatch!` | Migrate from local `Tier` to `simd_profile()` | 2h |
| T3.6   | `simd::tier()` | Alias to `simd_profile().coarse()` (preserve callers) | 2h |
| T3.7   | `hpc::simd_dispatch::detect()` | Migrate to `simd_profile()`; add Avx512f-only, AvxVnniInt8, IceLakeSp dispatches | 3h |
| MX-T5  | `simd_int_ops::gemm_u8_i8` | Migrate cfg chain to `GemmDispatch.int8_gemm` pointer (both compile-time pin and LazyLock-once modes) | 2h |

**Phase 3 total: ~23h.** Provides the framework for Phase 4 and removes
the three duplicate Tier enums (TD-T12/T13/T14).

### Phase 4 — Intra-bucket SIMD fills (parallelizable)

Each task is one PR. Restated from `td-simd-integration-plan.md` Phase 4
with priority rebalanced based on this matrix:

| Task    | Profile unlocking it     | Surface that gets faster | Effort |
|---------|--------------------------|--------------------------|--------|
| MX-F1 (HOT) | SPR/GNR/CPL/Z4/Z5 | `BF16x16::add/sub/mul/fma` via `vdpbf16ps`-style F32x16 mul_add (drop scalar f32 round-trip) | 4h |
| MX-F2 (HOT) | All x86 (F16C stable since 2012) | `F16x16::to_f32x16` + `add/sub/mul/fma` via `vcvtph_ps`/`vcvtps_ph` round-trip + F32x16 ops | 4h |
| MX-F3 (HOT) | A76 + (arm fp16) | `F16x16` arm with FMLA `v.8h` asm-byte | 3h |
| MX-F4   | SPR/GNR (avx512fp16)     | Native `F16x{8,16}` `__m{256,512}h` storage on Sapphire+/Granite (skips F32 round-trip)| 6h |
| MX-F5   | All AVX-512F             | `simd_ln_f32` Remez polynomial (currently scalar everywhere) | 3h |
| MX-F6   | All AVX-512BW            | `nibble_unpack`, `nibble_above_threshold` 2× width — TD-T16 | 2h |
| MX-F7   | HSW                      | `nibble_unpack_avx2` real `_mm256_*` (TD-T17) | 2h |
| MX-F8   | All AVX-512F             | `distance::squared_distances_f32` 16-wide L2 (TD-T19) | 2h |
| MX-F9   | All AVX-512F             | `spatial_hash::batch_sq_dist` 16-wide (TD-T20) | 2h |
| MX-F10  | IceLakeSp+/SPR/GNR/Z4/Z5 | VPOPCNTDQ paths — Hamming/popcount audit | 4h |
| MX-F11  | IceLakeSp+/SPR/GNR/Z4/Z5 | VBMI byte-permute audit beyond `simd_avx512.rs:695` | 4h |
| MX-F12  | IceLakeSp+/Z4/Z5         | GFNI bitmatrix multiply audit | 6h |
| MX-F13  | ARL/GNR                  | `simd_int_ops::gemm_i8` (s8×s8 → i32) via `vpdpbssd` ymm/zmm — NEW agnostic surface | 4h |
| MX-F14  | ARL/GNR/A76(+usdot)      | `simd_int_ops::gemm_u8` (u8×u8 → u32) via `vpdpbuud` / NEON `udot` | 4h |
| MX-F15  | SPR/GNR (amx-int8)       | AMX arm of `simd_int_ops::gemm_u8_i8` — `tile_dpbusd` 16×16 (the kernel exists in `bf16_tile_gemm.rs`-shape, needs INT8 sibling) | 6h |
| MX-F16  | GNR (amx-fp16)           | AMX-FP16 `tdpfp16ps` — gated on CPUID.07H.1H:EAX[21], needs SimdCaps extension | 4h |

**Phase 4 total: ~60h, parallelizable.** Every task is gated on Phase 3's
`SimdProfile` infrastructure but otherwise independent. Land in any order.

### Phase 5 — BLAS-graph GEMM kernel polish (the JIT-parity zone)

The kernels that the user's earlier session brought to within "a few %" of
a Cranelift-JIT inner loop, via `array_chunks` + `array_windows` + the
polyfilled `mul_add` + `add_mul_*`. Once Phases 1–4 land, this phase
verifies that no per-CPU regression has crept in vs the historical baseline:

| Task    | Surface | Action | Effort |
|---------|---------|--------|--------|
| MX-P1   | `gemm_u8_i8` bench | Land the `#[ignore]` bench from Phase 0 as a published `benches/int8_gemm.rs` criterion bench so CI can detect regressions per arm | 2h |
| MX-P2   | `gemm_u8_i8` AMX path | Verify AMX kernel reaches ≥ 2× of avx512vnni zmm on SPR (audit's expected 256:64 mul-add ratio) | 2h |
| MX-P3   | `add_mul_f32` bench | Add as `benches/blas1.rs` — compare to scalar reference and to `f32::mul_add` per-element loop. Floor: SIMD ≥ 4× scalar at length ≥ 256 on each arm | 2h |
| MX-P4   | `bgz17_bridge` GEMM | Re-bench against JIT path (now retired). Confirm the original within-a-few-% gap still holds with the post-Phase-4 polyfill | 4h |
| MX-P5   | NO_REMOVE doc audit  | Walk `simd_ops.rs`, `simd_int_ops.rs`, `simd_half.rs`, `simd_soa.rs`. Confirm every helper that bench-shows ≥ 1.5× over scalar has a "Foundation primitive — do not remove" call-out with the bench number cited inline | 1h |

**Phase 5 total: ~11h.**

### Phase 6 — Future / out-of-current-scope

| Item                           | Why deferred |
|--------------------------------|--------------|
| `gemm_f32` BLAS-3              | `matrixmultiply` workspace dep handles this — wrapping it is API design, not SIMD work |
| GPU offload                    | Out of scope per CLAUDE.md "HPC Rust transformation" charter |
| Cranelift-JIT GEMM revival     | Dropped after the BLAS-graph polyfill reached parity — only reconsider if Phase 5 shows > 5% gap |
| `wasm32` SIMD128 backend       | `core::simd` via `nightly-simd` covers it; no per-target intrinsic wiring planned |
| RISC-V Vector extension       | `core::simd` ditto                                                                |
| Multi-core threading           | `matrixmultiply-threading` feature exists; deeper threading is a separate phase |

---

## K. How to read this doc

1. **Picking the cfg config for a deployment:** find your CPU profile column.
   Cells with ✅ on that column are wired. Cells with ⏳ are the speedups
   that landed kernels but didn't wire (low-hanging gains).
2. **Adding a new agnostic surface:** copy the `simd_int_ops::gemm_u8_i8`
   pattern — compile-time `#[cfg(target_feature)]` chain on `simd_int_ops`
   (the entry point), kernels in `hpc::vnni_gemm` / `hpc::neon_dotprod_gemm`
   / etc., scalar fallback as the universal arm.
3. **Verifying a per-CPU lowering is correct:** run the matching
   `bench_*_vs_scalar` ignored test under `RUSTFLAGS='-Ctarget-cpu=$CPU'`
   — the runner must have the silicon to execute the emitted instructions
   (Sapphire Rapids covers everything down to and including A76's intrinsic
   semantics; aarch64 needs a separate runner).
4. **Spotting matrix drift:** when adding a new public symbol to
   `crate::simd::*`, this table must grow a row. Reviewers should reject
   PRs that add a public symbol without a corresponding matrix entry.

## M. AArch64 ground-truth core enumeration (GCC source)

> **Scope correction (appended 2026-07-27, operator-stated).** The heading and
> the "authoritative" wording below overstate GCC's role. **GCC is the fill-in
> for what we could not execute**; everything reachable was verified by running
> it. Two distinct mechanisms, not to be conflated:
>
> - **Validation (what the lanes compute):** `scripts/neon-parity.sh` cross-builds
>   `crates/neon-simd-parity` for `aarch64-unknown-linux-gnu` and runs it under
>   `qemu-aarch64-static`, asserting the exercised lanes are bit-identical to
>   their scalar reference; `scripts/wasm-parity.sh` is the wasm32+simd128 twin
>   under node. **Coverage as of 2026-07-28 (from `selfcheck` in
>   `crates/neon-simd-parity/src/main.rs`, not the full export surface):**
>   `U32x16` (Add / BitXor / rotate_left — the ChaCha20/BLAKE ARX triple),
>   `F32x16` (splat / roundtrip / add / reduce_sum), `I8x16` (roundtrip / add).
>   Exported lanes NOT yet exercised there (e.g. `I16x8`, `U8x16`, `U64x2`) are
>   **unverified by this harness** — a later SIMD audit must not treat them as
>   measured; extending `selfcheck` is the way to promote one. Within its
>   coverage these runs are the measurement of record for lane arithmetic — and
>   they need **no physical silicon**, which is why the aarch64 surface could be
>   measured at all.
> - **Runtime detection (what a given CPU admits to having):**
>   `sysctl hw.optional.arm.FEAT_*` on Darwin / `getauxval(AT_HWCAP)` on
>   Linux/Android (`src/simd_neon_dotprod.rs:29-30`), `__cpuid_count` on x86
>   (`src/simd_caps.rs:160-167`).
>
> GCC's role is the third thing neither of those gives you: **which shipping core
> carries which feature.** Emulation proves an instruction works; it cannot tell
> you that `cortex-a76` has DOTPROD and `cortex-a72` does not. Read the table
> below as *GCC's declared per-core feature membership* — authoritative for
> untestable parts, corroborating elsewhere.
>
> The URL cited at the end of this section points at mutable `master`; pin a
> commit when re-scraping (see `.claude/knowledge/gcc-intrinsic-spec-reference.md`,
> which also documents the intrinsic-semantics layers of the same source).

The matrix above uses three aarch64 columns (A53 / A72 / A76) that
each cover a *dispatch tier* — multiple physical cores share the same
SIMD primitive set. The authoritative per-core feature membership is
in GCC's `gcc/config/aarch64/aarch64-cores.def`, scraped 2026-05-21:

| Core | GCC arch | Explicit feature flags |
|---|---|---|
| **A53/A72/A76 tier** (baseline NEON, optional dotprod+fp16, NO bf16) | | |
| `cortex-a53` | V8-A | `(CRC)` |
| `cortex-a72` | V8-A | `(CRC)` |
| `cortex-a76` | V8.2-A | `F16, RCPC, DOTPROD` |
| `cortex-a78` | V8.2-A | `F16, RCPC, DOTPROD, SSBS, PROFILE` |
| `cortex-x1`  | V8.2-A | `F16, RCPC, DOTPROD, SSBS, PROFILE` |
| `neoverse-n1`| V8.2-A | `F16, RCPC, DOTPROD, PROFILE` |
| `apple-m1`   | V8.5-A | `()` — V8.5 baseline includes F16+dotprod, NO bf16/i8mm |
| **V8.6-A tier** (BF16 + I8MM via baseline) | | |
| `apple-m2`   | V8.6-A | `()` — V8.6 baseline → bf16, i8mm, sve, sve2 |
| `apple-m3`   | V8.6-A | same |
| `oryon-1`    | V8.6-A | `CRYPTO, SM4, SHA3, F16` (Snapdragon X Elite/Plus) |
| `ampere1`    | V8.6-A | `F16, RNG, AES, SHA3` |
| `ampere1a`   | V8.6-A | `F16, RNG, AES, SHA3, SM4, MEMTAG` |
| **V8.7-A tier** (baseline + LS64 + MOPS) | | |
| `apple-m4`   | V8.7-A | `()` |
| `ampere1b`   | V8.7-A | `F16, RNG, AES, SHA3, SM4, MEMTAG, CSSC` |
| **V9.0-A tier** (SVE2 baseline + explicit bf16/i8mm) | | |
| `cortex-a510`| V9-A | `SVE2_BITPERM, MEMTAG, I8MM, BF16` |
| `cortex-a710`| V9-A | `SVE2_BITPERM, MEMTAG, I8MM, BF16` |
| `cortex-a715`| V9-A | `SVE2_BITPERM, MEMTAG, I8MM, BF16` |
| `cortex-x2`  | V9-A | `SVE2_BITPERM, MEMTAG, I8MM, BF16` |
| `cortex-x3`  | V9-A | `SVE2_BITPERM, MEMTAG, I8MM, BF16` |
| `neoverse-n2`| V9-A | `I8MM, BF16, SVE2_BITPERM, RNG, MEMTAG, PROFILE` |
| `neoverse-v2`| V9-A | `I8MM, BF16, SVE2_BITPERM, RNG, MEMTAG, PROFILE` (Graviton 4) |
| `grace`      | V9-A | `I8MM, BF16, SVE2_BITPERM, SVE2_AES, SVE2_SHA3, SVE2_SM4, PROFILE` |
| **V8.4-A SVE tier** (Graviton 3's odd one) | | |
| `neoverse-v1`| V8.4-A | `SVE, I8MM, BF16, PROFILE, SSBS, RNG` |
| **V9.2-A tier** (V9 + V8.7 features) | | |
| `cortex-a520`| V9.2-A | `SVE2_BITPERM, MEMTAG` |
| `cortex-a720`| V9.2-A | `SVE2_BITPERM, MEMTAG, PROFILE` |
| `cortex-a725`| V9.2-A | `SVE2_BITPERM, MEMTAG, PROFILE` |
| `cortex-x4`  | V9.2-A | `SVE2_BITPERM, MEMTAG, PROFILE` |
| `cortex-x925`| V9.2-A | `SVE2_BITPERM, MEMTAG, PROFILE` |
| `neoverse-n3`| V9.2-A | `SVE2_BITPERM, RNG, MEMTAG, PROFILE` |
| `neoverse-v3`| V9.2-A | `SVE2_BITPERM, RNG, LS64, MEMTAG, PROFILE` |

**Dispatch tier mapping (which matrix column each core lands in):**

| Tier (matrix col.) | Cores |
|---|---|
| A53 | `cortex-a53`, older V8.0-A |
| A72 | `cortex-a72`, V8.0-A + CRC |
| A76 (V8.2 with dotprod+fp16, NO bf16/i8mm) | `cortex-a76`, `cortex-a78`, `cortex-x1`, `neoverse-n1`, `apple-m1` |
| **(new tier — V8.6+/V9 with bf16+i8mm)** | `apple-m2`+, `oryon-1` (Snapdragon X), `cortex-a510`+, `neoverse-n2`/`v2`/`grace`, `ampere1`+ |
| **(new tier — V8.4-A + SVE + bf16+i8mm)** | `neoverse-v1` (Graviton 3 — only V8.4-A core with explicit SVE+bf16+i8mm) |

The matrix's three aarch64 columns cover the bottom of the dispatch
ladder. The bf16/i8mm tier (which would carry NEON BFMMLA / BFDOT /
USDOT / FMLA.8h) needs its own column in a future revision — when the
NEON BF16 asm-byte arm lands (Phase 3b in § J), every V8.6+ core
listed above gets covered by the same dispatch arm.

**Source provenance:** scraped from
`https://raw.githubusercontent.com/gcc-mirror/gcc/master/gcc/config/aarch64/aarch64-cores.def`
(GCC trunk, 2026-05-21). The `AARCH64_CORE(...)` macro emits the
canonical name → arch → feature-string mapping; GCC's
`(define_insn ...)` patterns in `aarch64-simd.md` give the bit
encodings for the asm-byte rule (`.inst 0xXXXXXXXX`) that Phase 3b
will use for BFMMLA / BFDOT / FMLA.8h / USDOT.

## L. Provenance

- CPU feature presence: sourced from `td-simd-cpu-dispatch-matrix.md`.
- Audit findings (TD-T*): sourced from `td-simd-tier-audit.md`.
- Phase 1–4 effort estimates: cross-referenced with
  `td-simd-integration-plan.md`; new MX-T* / MX-F* items estimated in this
  doc.
- Polyfilled type backing: read directly from `src/simd.rs` lines 197–366
  (cfg-gated re-exports per `target_feature`), `src/simd_avx512.rs`
  re-exports at 2260, `src/simd_avx2.rs` (256-bit polyfills), `src/simd_neon.rs`
  paired-load wrappers, `src/simd_scalar.rs` arrays.
- Surface function inventory: read directly from
  `src/simd_ops.rs`, `src/simd_int_ops.rs`, `src/simd_half.rs`,
  `src/simd_soa.rs`, `src/simd.rs` re-exports.
- No grep / tail / head sampling — every entry traceable to a full-file
  Read per the workspace rule.
