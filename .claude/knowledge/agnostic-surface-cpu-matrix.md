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
| `I32x16` | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`†| 2×`__m256i`†| 4×`int32x4_t`† | ←  | ←  | `[i32;16]` |
| `I32x8`  | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`  | `__m256i`  | 2×`int32x4_t`  | ←  | ←  | `[i32;8]`  |
| `U32x16` | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`⏳| 2×`__m256i`⏳| 4×`uint32x4_t` | ←  | ←  | `[u32;16]` |
| `U32x8`  | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`⏳ | `__m256i`⏳ | 2×`uint32x4_t` | ←  | ←  | `[u32;8]`  |
| `I64x8`  | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`| 2×`__m256i`| 4×`int64x2_t`  | ←  | ←  | `[i64;8]`  |
| `I64x4`  | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`  | `__m256i`  | 2×`int64x2_t`  | ←  | ←  | `[i64;4]`  |
| `U64x8`  | `__m512i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | 2×`__m256i`†| 2×`__m256i`†| 4×`uint64x2_t`† | ←  | ←  | `[u64;8]`  |
| `U64x4`  | `__m256i`  | ←   | ←   | ←   | ←   | ←   | ←   | ←   | `__m256i`  | `__m256i`  | 2×`uint64x2_t` | ←  | ←  | `[u64;4]`  |

† = native since 2026-09-13 (PR #306 five-flavour audit; the AVX2 column's "2×`__m256i`" is the storage SHAPE the codegen lowers to — the type stays the `#[repr(align(64))]` `[T; N]` array, measured packed for the bit-logic half and given two-half intrinsic bodies for rotate / reduce / compare-bitmask on 2026-09-14). Until then these two rows were WRONG: on aarch64 and wasm32 `simd.rs` re-exported the SCALAR `U64x8`/`I32x16`, and on the v3 arm they were `avx2_int_type!` array polyfills — the mask family (`simd_masking_ops`) rides exactly these two types, so it ran scalar on three of five flavours.

⏳ = TD-T22 polyfill audit — the 256-bit `U16x16/U16x32/U32x8/U32x16`
inner ops may currently use scalar storage under `#[target_feature]` rather
than real `__m256i` intrinsics. Needs verification (see § J integration plan).

> **⏳ RESOLVED — TD-T22 CLOSED, no gap (2026-07-28).** The audit is done and
> the answer is: the storage IS scalar in the SOURCE, and that costs nothing.
> `.cargo/config.toml` pins `-Ctarget-cpu=x86-64-v3` for every x86_64 build,
> so LLVM auto-vectorizes the `avx2_int_type!` loop bodies into packed AVX2.
> Measured on the ChaCha20 ARX triple over `U32x16`: **no scalar arithmetic
> touches lane data** (the only non-vector ops across all three probes are
> `retq` and the loop's `movl`/`decl`/`jne` trip counter),
> `rotate_left(16)` strength-reduced to `vpshufb`, and the
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

---

> Correction (this PR): `bitwise.rs`'s `popcount_batch_u64` and the former
> `hamming_avx2` now route through `U64x8` (PR #323); the N2/G rows below
> reflect the **pre-#323** state where noted (`popcount_batch_u64` — no
> dispatch at all, on any backend; `hamming_avx2` — mislabelled, `u64`
> XOR + scalar `count_ones()` per 8-byte word, no AVX2 intrinsic/type).

## N. Masking operations — per-realization lowering

> Scope: `src/simd_masking_ops.rs` — compare-to-mask family, mask algebra,
> ternary (care-masked) match family, gated `_under` family, masked
> scalar-reduction family, `masked_strided_group_sum`, data-indexed
> gather/scatter, the keyed group-by family, `masked_key_run_count_u32`,
> `mask_shift_morton` — plus `popcount_batch_u64`/`hamming_distance_raw`/
> `popcount_raw` (`src/bitwise.rs`, re-exported via `src/simd.rs`).
> Consolidated from two worker passes (compare/mask-algebra; reductions/
> groups) — every `file:line` citation from both source passes is kept.
>
> Cell legend: ✅ wired native instruction · ⏳ kernel exists, not dispatched
> here (debt) · 🟦 planned, no kernel · 🟡 polyfill-pass (delegates to the
> polyfilled type; its own per-CPU lowering does the work) · ✗ scalar (no
> vector instruction reachable on this backend for this shape) · — N/A.
>
> **"AVX-512 (v4)" vs "AVX-512+VPOPCNTDQ (config-avx512)"** are the SAME
> column except on the small number of rows that reach `U64x8::popcnt`
> (`mask_ternlog_popcount`, and §N.12's `popcount_raw`/`hamming_distance_raw`/
> `popcount_batch_u64`): this repo's own default AVX-512 config
> (`.cargo/config-v4.toml`, plain x86-64-v4/SKX baseline) does **not** enable
> `avx512vpopcntdq`, so a popcnt-touching row is real `_mm512_popcnt_epi64`
> only under `.cargo/config-avx512.toml` (Sapphire-Rapids-class) and falls to
> a scalar `count_ones()` loop under the repo's own default v4 tier — the two
> columns are split apart specifically so this distinction is visible per
> row instead of being flattened into one "✅ AVX-512" cell.

### N.0 Shared building blocks (referenced by file:line in every row below)

| type method | AVX-512 lowering | AVX2 lowering | NEON lowering | wasm128 lowering | scalar lowering | nightly (`core::simd`) |
|---|---|---|---|---|---|---|
| `I32x16::gt_bitmask` (`simd_avx512.rs:988`, `simd_avx2.rs:2601`, `simd_neon.rs:3236`, `simd_wasm.rs:2239`, `simd_scalar.rs:996`) | `_mm512_cmpgt_epi32_mask` → `__mmask16` directly (native hw bitmask) | `avx2_halves()` → 2×`_mm256_cmpgt_epi32` + 2×`_mm256_movemask_ps` combined (WAS a 23-op mixed scalar/packed loop before an intrinsic rewrite, 2026-09-14) | 4×`vcgtq_s32` + `quad_mask4` narrowing pack | 4×`i32x4_gt` + `i32x4_bitmask` pack | scalar `for i in 0..16 { if self.0[i] > other.0[i] { mask \|= 1<<i } }` — the **correctness anchor** | `i_word_types.rs:406`, `SimdPartialOrd`/`to_bitmask`-shaped op over `i32x16` |
| `U32x16::eq_bitmask` (`simd_avx512.rs:1658`, `simd_avx2.rs:1913`, `simd_neon.rs:1867`, `simd_wasm.rs:1250`, `simd_scalar.rs:1361`) | `_mm512_cmpeq_epu32_mask` → `__mmask16` directly | **scalar index loop** over `[u32;16]` — doc comment claims LLVM auto-vec, but **no codegen-oracle measurement is cited in this body** (unlike `gt_bitmask`'s history) | **scalar index loop** — doc comment cites the codegen-oracle finding for the shape *generally*, not re-measured for this fn | **scalar index loop** — same caveat | scalar loop, correctness anchor | `u_word_types.rs:756`, `u32x16` compare + `to_bitmask` |
| `U8x64::cmpeq_mask`/`cmpgt_mask` (`simd_avx512.rs:609/654`, `simd_avx2.rs:1381/1429`, no native struct on NEON/wasm, `simd_scalar.rs:1444/1483`) | `_mm512_cmpeq_epi8_mask`/`_mm512_cmpgt_epu8_mask` — single AVX-512BW mask instrs | composed from 2×`U8x32::cmpeq_mask`/`cmpgt_mask` (`_mm256_cmpeq_epi8`+movemask; `cmpgt` biases both `^0x80` then `_mm256_cmpgt_epi8`) — real vector ops (fixed 2026-09-14, `3a5da8c`) | **U8x64 IS the scalar fallback type here** (`simd.rs:410-411`) — ✗ | **U8x64 IS the scalar fallback type here too** (`simd.rs:430-431`) — ✗ | 64-iter scalar loop, correctness anchor | `u8_types.rs:340`, portable `u8x64` compare |
| `U64x8::cmpeq_mask`/`cmpgt_mask` (`simd_avx512.rs:1829/1848`, `simd_avx2.rs:2365/2397`, `simd_neon.rs:2910/2921`←`U64x2` at `2083/2103`, `simd_wasm.rs:1940/1964`, `simd_scalar.rs:1966/1996`) | `_mm512_cmpeq_epu64_mask`/`_mm512_cmpgt_epu64_mask` — direct unsigned mask intrs | **scalar 2×4-unrolled loop** — comment: "this arm is the scalar polyfill" (measured 0.55× LOSS vs scalar baseline) | 4×`U64x2::cmpeq_mask`/`cmpgt_mask` = `vceqq_u64`/`vcgtq_u64` (`CMHI`, genuine unsigned hw compare) + `vgetq_lane_u64` reads, OR-shift composed | 4×`u64x2_eq`/(bias-XOR+`i64x2_gt`) + `i64x2_bitmask` pack — wasm SIMD128 has no native unsigned 64-bit compare, sign-biases both operands first, same trick AVX2 uses at byte width | scalar 2×4-unrolled loop, correctness anchor | `u_word_types.rs:125/131` |
| `U64x8::ternlog<IMM>` (`simd_avx512.rs:4992`, `simd_avx2.rs:3966`, `simd_neon.rs:1930`, `simd_wasm.rs:1019`, `simd_scalar.rs:2155`) | single `_mm512_ternarylogic_epi64::<IMM>` — one `VPTERNLOGQ` | Shannon-decomposed on `c` into ≤2 two-input tables, each ≤2 native `BitAnd`/`BitOr`/`BitXor`/`Not` on `U64x8` — whose own array-polyfill loops are unverified vectorized (N.GAPS #7) | same decomposition, `U64x8::BitAnd`/`BitOr` etc are REAL 4×`vandq_u64`/`vorrq_u64`/… | same, `U64x8`'s bitwise ops are 4×`v128_and`/`v128_or`/… | same over scalar `[u64;8]`, correctness anchor | `u_word_types.rs:1109`, per-minterm bitwise ops |
| `U32x16::ternlog<IMM>` (`simd_avx512.rs:5030`, `simd_avx2.rs:4018`, `simd_neon.rs:2863`, `simd_wasm.rs:1896`, `simd_scalar.rs:2207`) | single `_mm512_ternarylogic_epi32::<IMM>` — one `VPTERNLOGD` | same decomposition over `[u32;16]` array-polyfill bitwise ops (unverified vectorized) | same, `U32x16::BitAnd` etc are 4×`vandq_u32`-class real NEON ops | same, real wasm128 4×`v128_and`-class ops | scalar decomposition, correctness anchor | per-minterm bitwise, `u32x16` |
| `U64x8::popcnt` | **CONDITIONAL**: `_mm512_popcnt_epi64` only under `avx512vpopcntdq`; the repo's own `.cargo/config-v4.toml` does **not** enable it → falls to a **scalar `count_ones()` loop**; only `.cargo/config-avx512.toml` gets the real instruction (`simd_avx512.rs:3027`) | scalar `count_ones()` loop, doc-labelled "scalar polyfill" | `simd_neon.rs:2831` — not independently re-read; presumed scalar-loop (no native 64-bit popcount on this backend) | not independently confirmed | scalar `count_ones()` loop, correctness anchor | `u_word_types.rs:146` |
| `U64x8::reduce_sum` | `_mm512_reduce_add_epi64` when reached via `xor_popcount`'s comment path; standalone at `simd_avx512.rs:1817` not re-read | not independently re-read | `simd_neon.rs:2789` not independently re-read | `simd_wasm.rs` ~L1837 not independently re-read | scalar fold, correctness anchor | `u_word_types.rs:95` |

### N.1 Compare-to-mask family (ungated)

Every 🟡 row: "backend detail is in N.0's row for the cited method." Function-local scalar glue (`clear_mask_tail` at `simd_masking_ops.rs:1976`, and the `for w in out_words.iter_mut() { *w = !*w }` complement loop) is a **plain scalar word loop, no SIMD type**, on every backend — noted once, not repeated per row.

| fn (file:line) | AVX-512 (v4) | AVX-512+VPOPCNTDQ | AVX2 (v3) | NEON | wasm128 | scalar | nightly `core::simd` | parity-tested? |
|---|---|---|---|---|---|---|---|---|
| `eq_u32_to_mask` (184) — `U32x16::eq_bitmask` (N.0) | ✅ `VPCMPEQD`→mask | ✅ (unaffected) | 🟡 scalar idx loop | 🟡 scalar idx loop | 🟡 scalar idx loop | ✗ scalar (anchor) | 🟡 `core::simd` | `check_predicates_to_mask` (lib.rs~430) |
| `eq_u32_strided_to_mask` (235) — `U32x16::eq_bitmask` for `stride_bytes==4` fast path (window-chunked byte load, own algorithmic branch, no `target_arch` cfg); scalar byte-gather otherwise + tail | ✅ (via eq_bitmask) for stride=4; ✗ scalar gather otherwise | same | 🟡 | 🟡 | 🟡 | ✗ (both paths scalar) | 🟡 | `check_predicates_to_mask` (lib.rs:446) — **exercises the GENERAL non-4-stride gather path only**; the `stride_bytes==4` fast path is not separately exercised (GAP #6) |
| `gt_i32_to_mask` (344) — `I32x16::gt_bitmask` (N.0) | ✅ `VPCMPGTD` | ✅ | ✅ 2×`vpcmpgtd`+movemask-via-ps | ✅ 4×`vcgtq_s32` | ✅ 4×`i32x4_gt` | ✗ (anchor) | ✅ | `check_predicates_to_mask` 0x540 |
| `lt_i32_to_mask` (2011) — `I32x16::gt_bitmask` operands swapped (`t.gt_bitmask(v)`), exact at `i32::MIN` by construction | ✅ | ✅ | ✅ | ✅ | ✅ | ✗ (anchor) | ✅ | 0x520 |
| `ge_i32_to_mask` (2037) — calls `lt_i32_to_mask` then complements (scalar `!*w` + `clear_mask_tail`) | ✅+✗ | ✅+✗ | ✅+✗ | ✅+✗ | ✅+✗ | ✗+✗ | ✅+✗ | 0x550 |
| `le_i32_to_mask` (2064) — calls `gt_i32_to_mask` then complements (scalar) | ✅+✗ | ✅+✗ | ✅+✗ | ✅+✗ | ✅+✗ | ✗+✗ | ✅+✗ | 0x530 |
| `ne_i32_to_mask` (2092) — `I32x16::gt_bitmask` TWICE, OR'd as scalar `u16\|u16` | ✅×2+✗ | ✅×2+✗ | ✅×2+✗ | ✅×2+✗ | ✅×2+✗ | ✗×2+✗ | ✅×2+✗ | 0x510 |
| `eq_i32_to_mask` (2120) — calls `ne_i32_to_mask` then complements | ✅+✗ | ✅+✗ | ✅+✗ | ✅+✗ | ✅+✗ | ✗+✗ | ✅+✗ | 0x500 |
| `ne_u32_to_mask` (2147) — calls `eq_u32_to_mask` (N.0 `eq_bitmask`) then complements | ✅+✗ | ✅+✗ | 🟡+✗ | 🟡+✗ | 🟡+✗ | ✗+✗ | 🟡+✗ | `check_predicates_to_mask` (ne_u32 block) |
| `eq_u8_to_mask` (2215) — `U8x64::cmpeq_mask` (N.0) | ✅ `VPCMPEQB` | ✅ | ✅ 2×`vpcmpeqb`+movemask | ✗ (U8x64=scalar arm on aarch64) | ✗ (U8x64=scalar arm on wasm) | ✗ (anchor) | 🟡 `core::simd` u8x64 | `check_unsigned_compare_to_mask` 0xC00 |
| `gt_u8_to_mask` (2259) — `U8x64::cmpgt_mask` | ✅ `VPCMPGTUB` (epu8) | ✅ | ✅ 2×(bias-xor+`vpcmpgtb`)+movemask | ✗ scalar | ✗ scalar | ✗ (anchor) | 🟡 | 0xC40 |
| `lt_u8_to_mask` (2290) — `U8x64::cmpgt_mask` swapped | ✅ | ✅ | ✅ | ✗ | ✗ | ✗ (anchor) | 🟡 | 0xC20 |
| `ge_u8_to_mask` (2316) — `lt_u8_to_mask` + complement | ✅+✗ | ✅+✗ | ✅+✗ | ✗+✗ | ✗+✗ | ✗+✗ | 🟡+✗ | 0xC50 |
| `le_u8_to_mask` (2343) — `gt_u8_to_mask` + complement | ✅+✗ | ✅+✗ | ✅+✗ | ✗+✗ | ✗+✗ | ✗+✗ | 🟡+✗ | 0xC30 |
| `ne_u8_to_mask` (2370) — `eq_u8_to_mask` + complement | ✅+✗ | ✅+✗ | ✅+✗ | ✗+✗ | ✗+✗ | ✗+✗ | 🟡+✗ | 0xC10 |
| `eq_u64_to_mask` (2433) — `U64x8::cmpeq_mask` (N.0) | ✅ `VPCMPEQUQ` (epu64) | ✅ | ✗ scalar 2×4-unrolled | ✅ 4×`vceqq_u64` (`CMEQ`) | ✅ 4×`u64x2_eq`+bitmask | ✗ (anchor) | 🟡 | `check_unsigned_compare_to_mask` (eq_u64 block) |
| `gt_u64_to_mask` (2467) — `U64x8::cmpgt_mask` (N.0, unsigned `epu64`/`CMHI`/bias-XOR) | ✅ `VPCMPGTUQ` | ✅ | ✗ scalar (measured 0.55× LOSS vs scalar baseline) | ✅ 4×`vcgtq_u64` (`CMHI`, native unsigned hw compare) | ✅ 4×(bias-xor+`i64x2_gt`)+bitmask | ✗ (anchor) | 🟡 | (u64 gt block) |
| `lt_u64_to_mask` (2502) — `U64x8::cmpgt_mask` swapped | ✅ | ✅ | ✗ scalar | ✅ | ✅ | ✗ (anchor) | 🟡 | (u64 lt block) |
| `ge_u64_to_mask` (2528) — `lt_u64_to_mask` + complement | ✅+✗ | ✅+✗ | ✗+✗ | ✅+✗ | ✅+✗ | ✗+✗ | 🟡+✗ | (u64 ge) |
| `le_u64_to_mask` (2555) — `gt_u64_to_mask` + complement | ✅+✗ | ✅+✗ | ✗+✗ | ✅+✗ | ✅+✗ | ✗+✗ | 🟡+✗ | (u64 le) |
| `ne_u64_to_mask` (2582) — `eq_u64_to_mask` + complement | ✅+✗ | ✅+✗ | ✗+✗ | ✅+✗ | ✅+✗ | ✗+✗ | 🟡+✗ | (u64 ne) |
| `eq_u32_via_to_mask` (1805) — **PURE SCALAR**, per-element `index[base+lane]` gather into `table[]`, bounds-checked, bit-set — no `crate::simd::*` type used at all, by design | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ (anchor = actual shape) | ✗ | `check_predicates_to_mask` (via_to_mask block, lib.rs:1352-1370) |

### N.2 Mask algebra (`mask_*` over `&[u64]`)

`mask_and/or/xor/andnot[_assign]` and `mask_ternlog*` walk `as_chunks::<{U64x8::LANES}>()` and call the U64x8 operator/method per chunk (N.0 rows), with `pad_tail` for the remainder — real chunked SIMD dispatch, no `target_arch` cfg of its own. `mask_not`/`mask_not_assign` and `mask_any`/`mask_all` are the exception (GAPS #1, #2).

| fn (file:line) | AVX-512 (v4) | AVX-512+VPOPCNTDQ | AVX2 (v3) | NEON | wasm128 | scalar | nightly `core::simd` | parity-tested? |
|---|---|---|---|---|---|---|---|---|
| `mask_and` (366) — `U64x8::BitAnd` | ✅ `VPANDQ` (`impl_bin_op!`) | ✅ | 🟡 `[u64;8]` array-loop (unverified vectorized, GAPS #7) | ✅ 4×`vandq_u64` | ✅ 4×`v128_and` | ✗ scalar loop | 🟡 `core::simd &` | `check_mask_algebra` (lib.rs~576) |
| `mask_or` (399) — `U64x8::BitOr` | ✅ `VPORQ` | ✅ | 🟡 array-loop | ✅ 4×`vorrq_u64` | ✅ 4×`v128_or` | ✗ | 🟡 | `check_mask_algebra` |
| `mask_and_assign` (428) | ✅ | ✅ | 🟡 | ✅ | ✅ | ✗ | 🟡 | `check_mask_algebra` |
| `mask_or_assign` (454) | ✅ | ✅ | 🟡 | ✅ | ✅ | ✗ | 🟡 | `check_mask_algebra` |
| `mask_andnot` (497) — `U64x8::BitAnd`+`Not` (`va & !vb`) | ✅ `VPANDNQ`-shaped (may be LLVM-folded from AND+NOT; not independently disassembled) | ✅ | 🟡 array-loop AND+NOT | ✅ vector AND+NOT | ✅ vector AND+NOT | ✗ | 🟡 | `check_mask_algebra` |
| `mask_andnot_assign` (526) — same | ✅ (unfused-status unverified) | ✅ | 🟡 | ✅ | ✅ | ✗ | 🟡 | `check_mask_algebra` |
| `mask_ternlog<IMM>` (577) — `U64x8::ternlog` (N.0) | ✅ single `VPTERNLOGQ` | ✅ | 🟡 Shannon-decomposed AND/OR/XOR/NOT | ✅ decomposed, native NEON bitwise | ✅ decomposed, native wasm128 bitwise | ✗ decomposed scalar | 🟡 decomposed `core::simd` | `check_mask_algebra` (0x6xx block) — **all 256 `IMM` tables exercised** |
| `mask_ternlog_assign<IMM>` (612) — same | ✅ | ✅ | 🟡 | ✅ | ✅ | ✗ | 🟡 | `check_mask_algebra`, all 256 tables |
| `mask_ternlog_popcount<IMM>` (673) — `U64x8::ternlog`+`popcnt`+`reduce_sum` (N.0) | ⚠ ✅ ternlog, but `popcnt` is ✗ scalar `count_ones` under the repo's OWN default v4 config (no `avx512vpopcntdq`) | ✅ ternlog + ✅ popcnt (`_mm512_popcnt_epi64`, SPR/GNR/Z4/Z5-class silicon) | 🟡 ternlog + ✗ popcnt (scalar `count_ones` loop, doc-labelled "scalar polyfill") | 🟡 ternlog (native) + popcnt not re-verified this pass | 🟡 ternlog (native) + popcnt not re-verified | ✗+✗ (anchor) | 🟡+🟡 (`core::simd` popcount likely via `SimdUint::count_ones`, not re-verified) | `check_mask_algebra`, 256 tables |
| `mask_ternlog_any<IMM>` (734) — `U64x8::ternlog` (N.0) OR-accumulate; horizontal test (`.to_array().iter().any(...)`) is a **plain scalar loop over 8 u64s on every backend** | ✅ ternlog + ✗ scalar horizontal test | ✅+✗ | 🟡 ternlog + ✗ scalar test | ✅ ternlog + ✗ scalar test | ✅ ternlog + ✗ scalar test | ✗+✗ (anchor) | 🟡+✗ | `check_mask_algebra`, 256 tables |
| `mask_not` (2610) — **NONE, plain scalar word loop**, no `as_chunks`/`U64x8` at all (GAPS #1) | ✗ (LLVM MAY auto-vec `!u64` at `-Ctarget-cpu`≥v3 per TD-T22 precedent for arithmetic; UNVERIFIED for this loop shape) | ✗ | ✗ | ✗ | ✗ | ✗ (anchor) | ✗ | `check_mask_algebra` (tail-length-covering `row_lens()`) |
| `mask_not_assign` (2642) — same shape | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | `check_mask_algebra` |
| `mask_set_range` (2715) — `fill_words` (interior: `U64x8::splat`+chunked store, memset-shaped) + `word_range_mask` (edge: scalar bit arithmetic) | interior ✅ (trivial), edges ✗ scalar | same | interior ✅-ish/edge ✗ | interior ✅/edge ✗ | interior ✅/edge ✗ | interior ✗ (scalar)/edge ✗ (anchor) | interior 🟡/edge ✗ | `check_set_range` (exhaustive lo/hi sweep) |
| `mask_xor` (2800) — `U64x8::BitXor` | ✅ `VPXORQ` | ✅ | 🟡 array-loop | ✅ 4×`veorq_u64` (inferred, not independently re-read) | ✅ 4×`v128_xor` (inferred, not independently re-read) | ✗ | 🟡 | `check_mask_algebra` |
| `mask_xor_assign` (2837) — same | ✅ | ✅ | 🟡 | ✅ | ✅ | ✗ | 🟡 | `check_mask_algebra` |
| `mask_any` (2876) — **NONE, plain scalar `for &w in words { acc \|= w }`** (GAPS #2), no chunking, no SIMD type | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ (anchor=actual) | ✗ | `check_mask_algebra` (incl. `mask_ternlog_any` cross-check) |
| `mask_all` (2903) — **NONE, plain scalar `for &w in &words[..full] { acc &= w }`** + tail-word compare | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ (anchor=actual) | ✗ | `check_mask_algebra` |

### N.3 Ternary (care-masked) match family

| fn (file:line) | AVX-512 (v4) | AVX-512+VPOPCNTDQ | AVX2 (v3) | NEON | wasm128 | scalar | nightly `core::simd` | parity-tested? |
|---|---|---|---|---|---|---|---|---|
| `ternary_match_u32_to_mask` (2946) — `U32x16::ternlog::<XOR_AND>` (N.0) then `.eq_bitmask(zero)` (N.0) | ✅ ternlog (`VPTERNLOGD`)+✅ eq_bitmask (`VPCMPEQD`) | same | 🟡 ternlog (decomposed) + 🟡 eq_bitmask (scalar idx loop) | ✅ ternlog (native) + 🟡 eq_bitmask (scalar idx loop) | ✅ ternlog (native) + 🟡 eq_bitmask (scalar idx loop) | ✗+✗ (anchor) | 🟡+🟡 | `check_care_match` 0x700 |
| `ternary_match_u64_to_mask` (2978) — `U64x8::ternlog::<XOR_AND>` (N.0) then a **plain scalar per-lane zero-test loop** `bits \|= (x==0) as u64 << lane` — NOT `U64x8::eq_bitmask`-against-zero (GAPS #5) | ✅ ternlog + ✗ scalar zero-test loop | same | 🟡 ternlog + ✗ scalar zero-test | ✅ ternlog + ✗ scalar zero-test | ✅ ternlog + ✗ scalar zero-test | ✗+✗ (anchor) | 🟡+✗ | `check_care_match` 0x710 — **plausible GAP, see GAPS #5** |
| `ternary_match_strided_to_mask` (3031) — hi 4B: `U32x16::ternlog`+`eq_bitmask` (N.0, same pattern) over a gathered `[u32;16]`; lo 8B: `U64x8::ternlog` + scalar per-lane zero test (same shape as `ternary_match_u64_to_mask`); byte gather itself **scalar**, own doc says so explicitly | hi ✅+✅, lo ✅ternlog+✗scalar, gather ✗ | same | hi 🟡+🟡, lo 🟡+✗, gather ✗ | hi ✅+🟡, lo ✅+✗, gather ✗ | hi ✅+🟡, lo ✅+✗, gather ✗ | all ✗ (anchor) | hi 🟡+🟡, lo 🟡+✗, gather ✗ | `check_care_match` 0x720, incl. anti-vacuity guard (0x721) |

### N.4 Gated `*_to_mask_under` family (`Pred { under }`, survivor-word skip)

Every `_under` fn shares the exact same lowering as its ungated sibling for the COMPARE itself (N.1's method rows), composed through `pack_under` (`simd_masking_ops.rs:3139`) instead of `pack` — the only structural difference is a per-word `if gate==0 { skip }` scalar branch (control flow, not compute) on every backend. Complement forms apply the NOT inside the closure on the already-packed bits (scalar bitwise-NOT of a scalar word) — a different shape from the ungated family's whole-word complement loop, but the same "scalar NOT on packed result" classification.

| fn (file:line) | AVX-512 (v4) | AVX-512+VPOPCNTDQ | AVX2 (v3) | NEON | wasm128 | scalar | nightly `core::simd` | parity-tested? |
|---|---|---|---|---|---|---|---|---|
| `gt_i32_to_mask_under` (3285) — `I32x16::gt_bitmask` + scalar gate-skip | ✅+✗skip | ✅+✗skip | ✅+✗skip | ✅+✗skip | ✅+✗skip | ✗(anchor)+✗ | ✅+✗ | `check_predicates_under` 0xA00 |
| `lt_i32_to_mask_under` (3320) — swapped + gate-skip | ✅+✗ | ✅+✗ | ✅+✗ | ✅+✗ | ✅+✗ | ✗+✗ | ✅+✗ | 0xA10 |
| `ge_i32_to_mask_under` (3355) — + scalar-NOT-on-packed-bits + gate-skip | ✅+✗+✗ | ✅+✗+✗ | ✅+✗+✗ | ✅+✗+✗ | ✅+✗+✗ | ✗+✗+✗ | ✅+✗+✗ | 0xA20 |
| `le_i32_to_mask_under` (3389) — + scalar-NOT + gate-skip | ✅+✗+✗ | ✅+✗+✗ | ✅+✗+✗ | ✅+✗+✗ | ✅+✗+✗ | ✗+✗+✗ | ✅+✗+✗ | 0xA30 |
| `ne_i32_to_mask_under` (3424) — ×2 OR (scalar) + gate-skip | ✅×2+✗+✗ | ✅×2+✗+✗ | ✅×2+✗+✗ | ✅×2+✗+✗ | ✅×2+✗+✗ | ✗×2+✗+✗ | ✅×2+✗+✗ | 0xA40 |
| `eq_i32_to_mask_under` (3459) — ×2 OR + scalar-NOT + gate-skip | ✅×2+✗+✗ | ✅×2+✗+✗ | ✅×2+✗+✗ | ✅×2+✗+✗ | ✅×2+✗+✗ | ✗×2+✗+✗ | ✅×2+✗+✗ | 0xA50 |
| `eq_u32_to_mask_under` (3493) — `U32x16::eq_bitmask` + gate-skip | ✅+✗ | ✅+✗ | 🟡+✗ | 🟡+✗ | 🟡+✗ | ✗(anchor)+✗ | 🟡+✗ | 0xA60 |
| `ne_u32_to_mask_under` (3527) — + scalar-NOT + gate-skip | ✅+✗+✗ | ✅+✗+✗ | 🟡+✗+✗ | 🟡+✗+✗ | 🟡+✗+✗ | ✗+✗+✗ | 🟡+✗+✗ | 0xA70 |
| `ternary_match_u32_to_mask_under` (3562) — `U32x16::ternlog`+`eq_bitmask` (same as ungated) + gate-skip | ✅+✅+✗ | ✅+✅+✗ | 🟡+🟡+✗ | ✅+🟡+✗ | ✅+🟡+✗ | ✗+✗+✗ | 🟡+🟡+✗ | 0xA80 |
| `ternary_match_u64_to_mask_under` (3601) — `U64x8::ternlog` + scalar zero-test loop (same shape as ungated) + gate-skip | ✅+✗+✗ | ✅+✗+✗ | 🟡+✗+✗ | ✅+✗+✗ | ✅+✗+✗ | ✗+✗+✗ | 🟡+✗+✗ | 0xA90 |

**Absent siblings** (not gaps — deliberate non-builds, tracked in `masking-ops-state.md`): no `_under` siblings for u8/u64 families ("the i32/u32 families have them; the shape is mechanical; no caller needs one").

### N.5 `pub mod ternlog` immediates (`src/simd.rs:588`)

Plain compile-time `i32` constants (`AND3`, `AND2_ANDNOT`, `AND_ANDNOT2`, `OR2_AND`, `XOR3`, `MAJ3`, others not re-enumerated) — no backend, no dispatch, no cfg; lives on the facade specifically so it resolves identically on every arm. N/A for the per-CPU columns; excluded from N.13's summary counts.

### N.6 Masked scalar-reduction family (`masked_sum_i32`, `masked_sum_wrapping_add_i32`, `masked_min_i32`, `masked_max_i32`, `blend_i32`)

All four reductions walk `mask_words` via `trailing_zeros`/`bits &= bits - 1` (popcount-order set-bit walk), index `values[base + lane]`, fold into a scalar accumulator. **None of the four calls any polyfilled SIMD type at all.** Doc-justified per function: `masked_sum_i32`'s own doc says the obvious vector shape (`I32x16::reduce_sum`, accumulates in `i32`) is "wrong, and quietly so" — it would overflow the promised `i64` bound. `blend_i32` is doc-labelled "no lane wrapper in the mask vocabulary yet — a scalar loop is the honest shape until one is measured to be needed." No `cfg(target_arch)`/`cfg(target_feature)` branch appears in any of the five bodies, so every backend column below is identical BY CONSTRUCTION (one Rust body compiled once per target), not by five independent measurements.

| fn | core mechanism (file:line) | AVX-512 (v4) | AVX-512+VPOPCNTDQ | AVX2 (v3) | NEON | wasm128 | scalar | nightly | parity-tested? |
|---|---|---|---|---|---|---|---|---|---|
| `masked_sum_i32` | scalar `trailing_zeros` bit-walk + `wrapping_add`; `values.rs:814-841` | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ (no cfg; same body compiles everywhere) | yes — `simd-masking-parity/src/lib.rs:794` |
| `masked_sum_wrapping_add_i32` | same bit-walk, folds `a[i].wrapping_add(b[i]) as i64`; :878-905 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | yes — `lib.rs:800` |
| `masked_min_i32` | `masked_fold_i32(values, mask, i32::min)`, same bit-walk; :3634-3636, fold :3659-3687 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | yes — `lib.rs:803` (incl. `i32::MIN`/`MAX` boundary case :813-814) |
| `masked_max_i32` | `masked_fold_i32(values, mask, i32::max)`; :3655-3657 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | yes — `lib.rs:806` |
| `blend_i32` | plain `for i in 0..n { dst[i] = if bit {a[i]} else {b[i]} }`; :3714-3728 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | yes — `lib.rs:858` |

### N.7 `masked_strided_group_sum` — sub-word field, large-stride gather

| fn | core mechanism (file:line) | all backends |
|---|---|---|
| `masked_strided_group_sum` | scalar bit-walk selects records; per selected record, `groups` sub-word LE fields read byte-by-byte (`for k in 0..group_bytes { v \|= (bytes[o+k] as u32) << (8*k) }`, unaligned-safe) and widened into an `i128` accumulator, range-checked to `i64` at the end; :965-1015 | ✗ scalar, identical on every target |

Doc-justified: one record (≤16 bytes) per cache line at a large stride (512B motivating case) is memory-bound, records are non-adjacent so no vector gather of several records at once, and the per-record payload (≤12B) does not fill a lane — "vectorising the *decode* would optimise the part that is already free." Parity-tested: yes — `lib.rs:842` (varying `n`, `groups`, `group_bytes`).

### N.8 Data-indexed gather/scatter (`mask_gather_u32`, `mask_scatter_or_u32`)

| fn | core mechanism (file:line) | all backends | parity-tested? |
|---|---|---|---|
| `mask_gather_u32` | per output bit, scalar `idx = index[i]; if idx < src_rows && bit(src, idx) { set }`; :1099-1128 | ✗ scalar, identical on every target | yes — `lib.rs:1173` |
| `mask_scatter_or_u32` | per selected source bit, scalar `out_words[idx/64] \|= 1<<(idx%64)` guarded by `idx < out_rows`; :1194- | ✗ scalar, identical on every target | yes — `lib.rs:1202`, `:1225` (accumulate-across-calls) |

Doc, verbatim reason: "there is no vector gather over individual BITS on any of this crate's backends (a byte/word/dword gather exists; a bit gather does not)"; mirror claim for scatter. Real, named gap against the backends' own byte/dword gather instructions (AVX-512 `vpgatherdd`/`vpscatterdd`; AVX2 `vpgatherdd`) — see GAPS #C3; the doc is honest a coarser (row-level, not bit-level) primitive is what's missing, not that no vector gather exists at all on these ISAs.

### N.9 Keyed group-by family (`masked_group_{sum,sum_sym,count,min,max}_i32/_u32`, `_via` variants, `SYM_EMPTY_I64`)

All nine public functions (10 counting `_sum`/`_sum_sym` separately) are named instances of ONE shared scalar engine, `group_walk` (:1521-1544), parameterized by a closure (`fold`) and a `GroupKeyAddr` (`Resident(&[u32])` or `Via{index,table}` two-hop indirection, :1484-1505). `group_walk` itself: scalar bit-walk (same `trailing_zeros` pattern as N.6) resolving each selected row's group via `GroupKeyAddr::group_of` (data-dependent `usize` lookup, zero-fallback on out-of-range) and calling `fold(&mut out[group], i)`. No `cfg(target_arch)` anywhere — same one-body-compiled-once situation as N.6-N.8.

| fn | fold closure (file:line) | key address | all backends | parity-tested? |
|---|---|---|---|---|
| `masked_group_sum_i32` | `*slot = slot.wrapping_add(values[i] as i64)`; :1305-1307 | `Resident(keys)` | ✗ scalar (via `group_walk`) | yes — `lib.rs:1251`, `:1274` (accumulate-across-calls) |
| `masked_group_sum_i32_via` | same fold; :1360-1367 | `Via{index,table}` | ✗ scalar | yes — `lib.rs:1313`, `:1341` |
| `masked_group_sum_sym_i32` | `sym_sum_fold(values)` — replaces `SYM_EMPTY_I64` on first hit, else `wrapping_add`; :1426-1433, fold :1458-1468 | `Resident(keys)` | ✗ scalar | **no** — not in `simd-masking-parity`; unit-tested only (`simd_masking_ops.rs` :3206, :3224, :3248) |
| `masked_group_sum_sym_i32_via` | same fold; :1447-1454 | `Via` | ✗ scalar | **no** — unit-tested only (`:3208`) |
| `masked_group_count_u32` | `*slot = slot.wrapping_add(1)`; :1574-1576 | `Resident(keys)` | ✗ scalar | **no** — unit-tested only (`:3083`, `:3157-3158`, `:3173`) |
| `masked_group_count_u32_via` | same; :1601-1608 | `Via` | ✗ scalar | **no** — unit-tested only (`:3086`, `:3177`) |
| `masked_group_min_i32` | `*slot = (*slot).min(values[i] as i64)`; :1642-1644 | `Resident(keys)` | ✗ scalar | **no** — unit-tested only (`:3097`, `:3142`, `:3161`, `:3264`) |
| `masked_group_min_i32_via` | same; :1672-1679 | `Via` | ✗ scalar | **no** — unit-tested only (`:3100`) |
| `masked_group_max_i32` | `*slot = (*slot).max(values[i] as i64)`; :1706-1708 | `Resident(keys)` | ✗ scalar | **no** — unit-tested only (`:3104`, `:3145`) |
| `masked_group_max_i32_via` | same; :1712 area | `Via` | ✗ scalar | **no** — unit-tested only (`:3107`) |
| `SYM_EMPTY_I64` | `pub const SYM_EMPTY_I64: i64 = i64::MIN` — reserved sentinel, not a fn; :1386 | n/a | n/a (const) | n/a |

**Coverage note (documentation-accuracy finding, not a bug):** of the 10 named group-by functions, only `masked_group_sum_i32` and `masked_group_sum_i32_via` are exercised by the cross-backend `simd-masking-parity` harness (grep of `crates/simd-masking-parity/src/lib.rs` for every function name in this section — only the two `sum` names hit). Since every one of the 10 shares the identical scalar `group_walk` engine with no backend-conditional code, this is low real risk — but the harness's own module doc frames itself as covering "the permutation/scatter family (`mask_gather_u32`/`mask_scatter_or_u32`/`masked_group_sum_i32`/`masked_group_sum_i32_via`...)" (`lib.rs:25-28`), honestly scoped to exactly the two sum functions; no OTHER file claims cross-backend coverage of the other 8, so a reader could still assume the whole group-by family is parity-proven when 8 of 10 are single-build `#[cfg(test)]`-only.

### N.10 `masked_key_run_count_u32` + `KeyRunCarry`

| fn | core mechanism (file:line) | all backends | parity-tested? |
|---|---|---|---|
| `masked_key_run_count_u32` | scalar `for (i, &k) in keys.iter()` loop; compares `k` against the open run's key (refuses on decrease, `None`), tests the mask bit inline (`(mask_words[i/64] >> (i%64)) & 1`), commits only on full success (local-copy-then-commit, `None`-must-not-poison-carry); :1932-1959 | ✗ scalar | yes — `simd-masking-parity/src/lib.rs:1400` (tiled multi-call carry test), `:1424` (refusal leaves carry untouched) |
| `KeyRunCarry::finish` | `usize::from(self.key.is_some() && self.hit)`, resets `*self`; :1861-1866 | n/a (plain arithmetic on a 2-field struct) | covered transitively by the same test |

Doc, explicit: "Boundaries...vectorise as a shifted compare and hits are a mask word, but 'first hit per run' is a segmented scan with a serial carry; at one compare + one bit test per element it is memory-bound already. Left scalar until a measurement says otherwise" (:1901-1906) — a stated, falsifiable deferral, not an unexamined gap.

### N.11 `mask_shift_morton` (+ `MortonDir`) — the ONE function in scope that is genuinely vectorized

| fn | core mechanism (file:line) | AVX-512 (v4) | AVX-512+VPOPCNTDQ | AVX2 (v3) | NEON | wasm128 | scalar | nightly | parity-tested? |
|---|---|---|---|---|---|---|---|---|---|
| `mask_shift_morton` | **Pass 1 (interior permutation): real SIMD**, chunked over `U64x8::LANES` via `as_chunks`, per chunk `U64x8::from_array` + 3× `morton_case_shift` (masked `Shl`/`Shr` via `U64x8::splat`+`<<`/`>>`) OR-accumulated, tail zero-padded through the same path; :3956-3979. **Pass 2 (one-word carry): scalar**, one `dst[nb] \|= carry_bits << / >> shift` per source word with nonzero carry mask, via `inc_word`/`dec_word` dilated-integer address arithmetic; :3980-4005 | 🟡 (`U64x8::from_array`/`Shl`/`Shr`/`BitOr`/`&`/`splat` → `_mm512_sllv_epi64`/`_mm512_srlv_epi64`/native word ops, `simd_avx512.rs:1879-1895`) pass 1; pass 2 ✗ scalar | same as v4 (unaffected by popcnt) | 🟡 (`U64x8::{Shl,Shr}` on AVX2 exist, `simd_avx2.rs:1696,1712`; body not independently re-read, same delegation pattern) pass 1; pass 2 ✗ scalar | 🟡 (`U64x8::{BitAnd,BitOr,Shl,Shr}` on NEON exist, `simd_neon.rs:2961-3034`) pass 1; pass 2 ✗ scalar | 🟡 (`U64x8::{BitAnd,BitOr,Shl,Shr}` on wasm exist, `simd_wasm.rs:2002-2062`) pass 1; pass 2 ✗ scalar | 🟡 (`U64x8` scalar realization's own `Shl`/`Shr`, `simd_scalar.rs:1373-1385`); pass 2 ✗ scalar | not independently checked — polyfill-pass, follows whichever `nightly-simd` realization ships | yes — `simd-masking-parity/src/lib.rs:954`, `:965` |

Doc for pass 2's scalar-ness: "there is exactly one neighbour word per source word, so there is nothing here for a vector lane to parallelize over" (:3980-3982). Per this file's own header note, `mask_shift_morton` is the one documented exception to the module's trailing-zero-bit guarantee — an OR-accumulator, not a fresh overwrite (module header, :70-75).

**Historical/OUTLOOK note (`.claude/knowledge/masking-ops-state.md`, G4):** the op shipped and its own pre-registered falsifier fired — over the full field the word-level op measured **−14%** (14.5 vs 17.0 µs), not the modelled ~98%, and *lost* to an unrelated NNUE delta arm (9.3 µs); the real win (**−66%**) came from restricting the op to a trie node's own word span, a mechanism the original gap row never named. A documented, graded `[H]` finding about the OP'S MEASURED VALUE, not its backend coverage — included because it's the one place this section and that knowledge doc intersect.

### N.12 `popcount_batch_u64`, `hamming_distance_raw`, `popcount_raw` (`src/bitwise.rs`, re-exported via `src/simd.rs`)

> **Pre-#323 state** — see the correction line at the top of this document.

| fn | core mechanism (file:line) | AVX-512 (v4) | AVX-512+VPOPCNTDQ | AVX2 (v3) | NEON | wasm128 | scalar | parity-tested? |
|---|---|---|---|---|---|---|---|---|
| `popcount_batch_u64` | **`words.iter().map(\|w\| w.count_ones() as u64).sum()`** — plain scalar-per-word loop; `bitwise.rs:274-277` | ✗ scalar (no dispatch of any kind — see doc-drift below) | ✗ (same, no dispatch means the vpopcntdq-gated tier is unreachable here regardless) | ✗ scalar | ✗ scalar | ✗ scalar | ✗ scalar | not found in `simd-masking-parity`, `neon-simd-parity`, or `wasm-simd-parity` (grep across all three crates returned zero hits) |
| `hamming_distance_raw` | `dispatch_hamming(a, b)`, `bitwise.rs:180-182` — real per-arch/feature dispatch chain, **only under `#[cfg(target_arch = "x86_64")]`**; tries VPOPCNTDQ (`kernels_avx512::hamming_distance`) → AVX-512BW nibble-LUT (`hamming_avx512bw`, real `U8x64` vpshufb popcount-LUT SIMD, :88-132) → "AVX2" (`hamming_avx2`, :60-82 — **mislabelled**, see doc-drift: `u64` XOR + scalar `count_ones()` per 8-byte word, NOT vectorized) → `hamming_scalar` (:51-58) | ✅ `_mm512_popcnt_epi64` (VPOPCNTDQ) when available (`kernels_avx512.rs:892-915`); else ✅ `U8x64` nibble-popcount-LUT `vpshufb` at `hamming_avx512bw` (AVX-512BW-only) | ✅ (VPOPCNTDQ path reached) | 🟡-mislabelled: named `hamming_avx2`, gated `#[target_feature(enable="avx2")]`, but body is `u64::from_ne_bytes`+XOR+scalar `count_ones()` per 8 bytes — no AVX2 intrinsic, no `U8x32`/`U8x64` type; effectively a hardware-POPCNT scalar loop compiling under an avx2 feature gate | **✗ scalar — no NEON path exists at all** (whole dispatch chain is `#[cfg(target_arch="x86_64")]`; `grep -n "neon\|aarch64" bitwise.rs` → zero hits) | **✗ scalar — no wasm path exists at all**, same reason | ✗ scalar (`hamming_scalar`, universal fallback, ONLY path on non-x86_64) | not found in the three parity crates |
| `popcount_raw` | `dispatch_popcount(a)`, `:185-187` — same shape: `#[cfg(target_arch="x86_64")]` tries VPOPCNTDQ (`kernels_avx512::popcount`, real `_mm512_popcnt_epi64`) → AVX-512BW nibble-LUT (`popcount_avx512bw`, :137-175, real `U8x64` SIMD) → **falls straight to `popcount_scalar`** — no AVX2 arm at all (unlike hamming) | ✅ VPOPCNTDQ / ✅ AVX-512BW nibble-LUT (both real) | ✅ | **✗ scalar — literally no AVX2-gated function exists** for popcount (`grep -n "fn popcount" bitwise.rs` → only `popcount_scalar` and `popcount_avx512bw`) | ✗ scalar — no NEON path | ✗ scalar — no wasm path | ✗ scalar (universal fallback, only path on non-x86_64/non-AVX-512BW x86_64) | not found in the three parity crates |

**Post-#323 state** (this is what the code does now):

| fn | AVX-512 (v4) | AVX-512+VPOPCNTDQ | AVX2 (v3) | NEON | wasm128 | scalar |
|---|---|---|---|---|---|---|
| `popcount_batch_u64` | 🟡 `U64x8::popcnt` → per-lane `count_ones` | ✅ VPOPCNTQ | 🟡 per-lane `count_ones` | ✅ `vcntq_u8` | ✅ `i8x16_popcnt` | ✗ scalar |
| `hamming_distance_raw` | ✅ AVX-512BW nibble-LUT (runtime) | ✅ VPOPCNTDQ | 🟡 `hamming_u64x8` (per-lane `count_ones`) | ✅ `hamming_u64x8` → `vcntq_u8` | ✅ `hamming_u64x8` → `i8x16_popcnt` | ✗ scalar via `hamming_u64x8` |
| `hamming_distance_within` (new) | same kernel as `hamming_distance_raw`, per 256-byte block, exact early exit | ← | ← | ← | ← | ← |
| `popcount_raw` | unchanged (below AVX-512BW → `popcount_scalar`) | | | | | |

Tests: `bitwise::tests::test_{popcount_batch_u64,hamming_u64x8}_matches_reference`,
`test_hamming_distance_within_*`, `cascade::tests::*_respect_the_prefix_budget`.

### N.13 Summary — functions × backends with a real SIMD lowering vs scalar-only

**N.1/N.3/N.4 (39 compare/ternary-match rows, PRIMARY op only, complement/gate-skip scalar glue excluded — already itemized ✗ per row):**

| backend | rows with ≥1 real vector instruction (primary op) | rows scalar-loop-only (primary op) | notes |
|---|---|---|---|
| AVX-512 | 39 / 39 | 0 | every primary op has a dedicated `__mmask`-producing intrinsic or `VPTERNLOG*` |
| AVX2 | 21 / 39 | 18 / 39 | u64 family (12 rows) + `U32x16::eq_bitmask`-based rows (eq_u32, ne_u32, eq/ne_u32_under, ternary_match_u32 eq-half) are scalar; i32 (`gt_bitmask`) and u8 families are real 2-half vector ops |
| NEON | 27 / 39 | 12 / 39 | u8 family (6 rows: U8x64 has no native aarch64 type) is scalar; `eq_bitmask`-based rows are scalar-loop (claimed LLVM-vectorized in doc comment, unverified); i32, u64, ternlog-based rows real |
| wasm128 | 27 / 39 | 12 / 39 | same shape as NEON |
| scalar | 0 / 39 | 39 / 39 | by design (correctness anchor) |
| nightly | ~39/39 real `core::simd` (portable) but ~9 rows ride `U32x16::eq_bitmask`, whose nightly body was not independently disassembled — assumed 🟡 real per the `to_bitmask()` shape | — | "18 compare-to-mask pairs across every width" claim in masking-ops-state.md was spot-checked and holds |

Mask-algebra family (N.2, 16 rows): AVX-512 16/16 real (both popcnt sub-columns for the ternlog/algebra rows — only `mask_ternlog_popcount` diverges v4 vs +VPOPCNTDQ), AVX2 5/16 real (NEON/wasm bitwise-op families real, `mask_not`/`mask_not_assign`/`mask_any`/`mask_all` scalar-word-loop on **every** backend incl. AVX-512 — 4 rows are architecturally scalar regardless of ISA), NEON 12/16, wasm128 12/16, scalar 0/16.

**N.6-N.10 (23 functions, section-counted):**

| section | functions | real SIMD lowering (any backend) | scalar-on-every-backend | mixed |
|---|---|---|---|---|
| N.6 (masked_sum/min/max/blend) | 5 | 0 | 5 | 0 |
| N.7 (masked_strided_group_sum) | 1 | 0 | 1 | 0 |
| N.8 (gather/scatter) | 2 | 0 | 2 | 0 |
| N.9 (group-by family, 10 named fns) | 10 | 0 | 10 | 0 |
| N.10 (key_run_count) | 1 | 0 | 1 | 0 |
| N.11 (mask_shift_morton) | 1 | 1 (pass 1 only; pass 2 scalar on all) | 0 | 1 |
| N.12 (popcount_batch_u64, hamming_distance_raw, popcount_raw — pre-#323) | 3 | 2 (hamming_distance_raw, popcount_raw — AVX-512-only real SIMD) | 1 (popcount_batch_u64 — scalar on literally every backend incl. AVX-512) | 2 |
| **total (N.6-N.12)** | **23** | **3 have ANY real SIMD path, and only on AVX-512** | **20 scalar on every backend** | — |

Of the 23 functions in N.6-N.12, **19 are scalar by explicit, specific, per-function documented design** (N.6-N.10: data-dependent addressing, scatter-reduce conflict, memory-bound large-stride access, or overflow-safety reasons stated in each doc comment) — not oversight. The remaining 4 (N.11's pass 2, and all of N.12 pre-#323) are either a documented partial gap or genuinely under-realized relative to what the ISAs offer.

### N.14 GAPS — cells that are ✗/scalar where a SIMD instruction plausibly exists

**From the compare/mask-algebra pass:**

1. **`mask_not`/`mask_not_assign` never route through `U64x8` at all** (`simd_masking_ops.rs:2610-2653`) — every sibling word-algebra op walks `as_chunks::<U64x8::LANES>()`; these two are a bare `for (d,&s) in dst.zip(src) { *d = !s }`. A `Not` impl already exists on `U64x8` on every backend (used by `mask_andnot`'s `!vb`). Converting to the same chunked shape as siblings is a same-file, zero-new-backend-code fix.
2. **`mask_any`/`mask_all` are pure scalar word loops on every backend, including AVX-512** (`simd_masking_ops.rs:2876-2946`) — no `U64x8` OR/AND reduction used despite one existing (`U64x8::BitOr`/`BitAnd` + `reduce`-shaped horizontal test would let AVX-512 do this in `VPORQ`×N + `_mm512_test_epi64_mask`-class, or at minimum `as_chunks` + register OR-accumulate the way `mask_ternlog_any` already does internally). `mask_ternlog_any` proves the pattern is already known in this file.
3. **`U32x16::eq_bitmask` is a scalar index loop on AVX2/NEON/wasm128** (N.0), unlike its sibling `I32x16::gt_bitmask`, which received explicit intrinsic overrides on all three backends after a **measured** mixed-codegen finding (`simd_avx2.rs:2595-2599`: "the earlier index-loop spelling measured MIXED… 23 packed but lanes 0 and 13..=15 peeled off"). `eq_bitmask`'s doc comments (AVX2/NEON/wasm) all assert LLVM auto-vectorizes the identical shape, but **cite no matching codegen-oracle measurement of eq_bitmask itself** — only `gt_bitmask`'s history, generically. Affects roughly 9 of the 39 compare rows in N.1/N.3/N.4. Given `gt_bitmask`'s own history shows the "LLVM probably vectorizes this" assumption was WRONG once already on this exact codebase, `eq_bitmask` is the single highest-value re-measurement target in this whole matrix.
4. **`U64x8::popcnt` on AVX-512 is CONDITIONAL on `avx512vpopcntdq`**, which this repo's own default AVX-512 compile config (`.cargo/config-v4.toml`) does **not** enable — so `mask_ternlog_popcount` measured/labelled "✅ AVX-512" in casual reading is actually scalar `count_ones()` under the repo's own v4 CI/measurement tier, real only under `.cargo/config-avx512.toml`. This is why N.2's `mask_ternlog_popcount` row and N.12 split "AVX-512 (v4)" from "AVX-512+VPOPCNTDQ" explicitly.
5. **`ternary_match_u64_to_mask` (and its `_under` sibling) never calls `U64x8::cmpeq_mask` against a zero splat** — reads `.to_array()` after `ternlog` and tests each lane with a scalar `(x==0) as u64 << lane` loop (`simd_masking_ops.rs:2978-2990`, `3601-3612`), where `ternary_match_u32_to_mask` DOES chain into `U32x16::eq_bitmask` for the equivalent step. A `U64x8::cmpeq_mask(self, U64x8::splat(0))` call would let AVX-512/NEON/wasm do the zero-test in the same register op, removing the only remaining scalar step in that function's AVX-512/NEON/wasm lowering.
6. **`eq_u32_strided_to_mask`'s `stride_bytes==4` fast SIMD path is not independently exercised by the parity suite** — `check_predicates_to_mask` (lib.rs:446) calls it with `stride_bytes=16` (a 16-byte facet layout, deliberately NOT 4), which only reaches the general per-element scalar-gather branch. The vectorized `windows.as_chunks::<64>()` + `U32x16::eq_bitmask` branch (the "facet-major column" fast path per this file's own doc comment) has no dedicated parity call site found in this pass.
7. **Every `U64x8`/`U32x16` `BitAnd`/`BitOr`/`BitXor`/`Not` operator lowering on AVX2/scalar backing arrays is presumed-but-not-independently-verified vectorized** — the TD-T22 precedent (main matrix) measured this FOR ARITHMETIC (`add`/`mul`/`rotate_left` on ChaCha20), not for the bitwise `ternlog`-composition shape used throughout N.2/N.0's `ternlog` AVX2 rows. Same category as GAP #3, lower confidence of a real regression (bitwise ops vectorize more reliably than compare-and-pack shapes) but still unverified.

**From the reductions/groups pass:**

8. **`popcount_batch_u64` (N.12) — no dispatch at all, on ANY backend, including AVX-512, pre-#323.** The function's own inline comment claimed "Use POPCNT instruction if available, else scalar" but the body was a flat `.iter().map(count_ones).sum()` with no `cfg`, no `simd_caps()` check, no call into `crate::backend::kernels_avx512` or `U64x8::popcnt()`. `mask_ternlog_popcount` (same file) proves the SIMD shape already exists and is cheap (`U64x8::ternlog::<IMM>(..).popcnt()` register-accumulated, reduced once) — this could have been `U64x8::from_array(chunk).popcnt()` chunked exactly like every `mask_*` function above already does. **Closed by PR #323 per the correction note at the top of this document.**
9. **`popcount_raw`/`hamming_distance_raw` have real AVX-512 lowerings but degrade straight to scalar on AVX2, NEON, and wasm, pre-#323** — despite this crate having, elsewhere, real AVX2 (`U8x32`) and NEON `U8x*` polyfilled types a nibble-LUT popcount could be written against (the same `vpshufb`-family technique the AVX-512BW arm already uses at a narrower width), and despite NEON having a genuinely fast native popcount instruction (`vcnt`/`CNT`). `hamming_avx2`'s "AVX2" name is doubly misleading: gated on the `avx2` feature but contains no AVX2 intrinsic or polyfilled-type call, so even the one non-AVX-512 x86_64 arm that LOOKS covered is not actually vectorized — POPCNT-per-scalar-word wearing an AVX2 `#[target_feature]` tag. **Addressed by PR #323 per the correction note.**
10. **`mask_gather_u32`/`mask_scatter_or_u32` (N.8) are scalar with a stated reason** ("no bit-level gather/scatter on any backend") that is true as stated but narrower than available hardware: AVX-512 and AVX2 both have DWORD-granularity `vpgatherdd`/`vpscatterdd`. A byte- or dword-granularity redesign (dword-packed row groups rather than individual mask bits) could plausibly use them; today's bit-level API contract cannot without a design change the doc does not propose. Flagged as an ISA-vs-primitive-shape gap, not a bug.

No gap is claimed for N.6-N.10's popcount-order/scatter-reduce functions beyond what their own doc comments already state and justify (data-dependent destination, memory-bound access pattern, `i64`-overflow requirement `I32x16::reduce_sum` cannot satisfy).

### N.15 Doc drift — `masking-ops-state.md` claims vs code, this scope

1. **G1's "both realizations" claim implicitly excludes NEON/wasm, and the doc never says so.** `masking-ops-state.md`'s G1 entry and measured-numbers table only cite v4/AVX-512 and v3/AVX2. `U8x64` has **no native type on aarch64 or wasm32 at all** (N.0) — scalar fallback there, always; the doc's G1 write-up never states this. Vectorized on exactly 2 of 5 backend families.
2. **G2's doc entry ("only avx512 (`epu64`) and NEON (`cmhi`) have the instruction; wasm has no unsigned ordered 64-bit compare and uses the sign-bias trick; scalar/avx2 are flat polyfills") is CONFIRMED accurate** against code read this pass (N.0's `U64x8::cmpeq_mask`/`cmpgt_mask` row). No drift — positive confirmation.
3. **The main matrix's "Mask vectors" table (`F32Mask16`/`F32Mask8`/`F64Mask8`/`F64Mask4`) does not cover the INTEGER compare-to-mask masks this section's functions produce** (`u16`/`u8`/`u64` packed bitmasks from `gt_bitmask`/`cmpeq_mask`/etc). Different, narrower surface (float-comparison mask TYPES) — not a contradiction, but worth a cross-reference note.
4. **The main matrix's "Critical type-method per-CPU lowerings" table has no row at all for `gt_bitmask`, `eq_bitmask`, `U8x64::cmpeq_mask`/`cmpgt_mask`, or `U64x8::cmpeq_mask`/`cmpgt_mask`/`ternlog`/`popcnt`** — exactly the kind of "non-obvious lowering" that table's own purpose calls out. A real gap in that table.
5. **`masking-ops-state.md`'s "3-layer contract" framing is accurate as architecture**, but its corollary claim in `simd_masking_ops.rs`'s own module doc ("no backend semantics live here… it never branches on an ISA") is true for every function read this pass, EXCEPT the file does contain one algorithmic (non-ISA) data-shape branch: `eq_u32_strided_to_mask`'s `stride_bytes == 4` fast path vs general gather. Not an ISA branch, does not violate the stated rule, but is the one control-flow branch that depends on something other than element type/count.
6. **No direct contradiction found for N.6-N.10.** `masking-ops-state.md` does not discuss any of these functions by name (its G1-G8 list covers compare-to-mask families, argmin/argmax, compaction, lane-vs-lane compare, tree-depth — none overlap masked_sum/group-by/gather/scatter/key-run/morton). Its `popcount_batch_u64` mention (`:234`, "the wrong primitive for ranking" — a semantic point about tree-depth metrics) is consistent with, and does not depend on, whether the function is vectorized. The `popcount_batch_u64` GAP found here (inline comment misdescribing its own body) is a **self-contradiction inside `bitwise.rs`**, not a contradiction with the knowledge doc.
7. One adjacent-but-not-contradicted item for whoever maintains `masking-ops-state.md`: its DONE table and OUTLOOK section do not mention that 8 of the 10 `masked_group_*` functions have zero cross-backend parity coverage — not wrong, since the doc never claimed otherwise, but its stated purpose ("READ BY: any agent about to add, extend, or cite a `*_to_mask`/`mask_*`/`masked_*` primitive") would be a natural place to record it.

## O. Crypto lane

> New section — no `U32x16`/`U64x8` rows exist in the main matrix for the
> ARX/crypto angle (its existing `U32x16`/`U64x8` rows in §A are generic
> storage shape only, confirmed by absence). Legend reused from the main
> matrix header: ✅ wired/native · ⏳ exists, not dispatched here · 🟦
> planned, no kernel · 🟡 polyfill-pass (delegates to the type) · ✗ scalar
> fallback · — N/A. Every cell is read from source, cited `file:line`.
> Backends: AVX-512 (`src/simd_avx512.rs`), AVX2/v3-baseline
> (`src/simd_avx2.rs`, array polyfill autovectorized under the repo's
> target-cpu pin), NEON (`src/simd_neon.rs`, native `[U32x4;4]`/`[U64x2;4]`
> fan-out), wasm32+simd128 (`src/simd_wasm.rs`, native `v128` fan-out),
> scalar (`src/simd_scalar.rs`, `[T;N]` array), nightly `core::simd`
> (`src/simd_nightly/u_word_types.rs`).

### O.1 The ARX lane methods on `U32x16`/`U64x8`

**`U32x16` — Add / BitXor / rotate_left (ChaCha20 + BLAKE3 quarter-round):**

| backend | storage | Add | BitXor | rotate_left(n) |
|---|---|---|---|---|
| AVX-512 | native `__m512i` (`simd_avx512.rs:1446`) | `_mm512_add_epi32` via `impl_bin_op!` (`:1679`) | `_mm512_xor_si512` (`:1686`) | `_mm512_rolv_epi32` — VPROLVD, single instruction (`:1672-1678`) |
| AVX2 (v3 baseline) | `[u32;16]` array polyfill, `#[repr(align(64))]`, `avx2_int_type!` (`simd_avx2.rs:1101`, instantiated `:1578`) | plain `wrapping_add` loop (macro body `:1145-1153`) | plain `^` loop (`BitXor` arm ~`:1180`) | per-lane `u32::rotate_left` loop (`:1891-1899`) |
| NEON | native `[U32x4;4]` fan-out (`simd_neon.rs:1627`) | delegates to `U32x4::add` = `vaddq_u32` (`:1880-1886`, intrinsic `:1579-1581`) | delegates to `U32x4::bitxor` = `veorq_u32` (`:1964-1972`, intrinsic `:1596-1598`) | fan `U32x4::rotate_left` ×4 (`:1839-1848`), each `vshlq_u32(self,+n)` \| `vshlq_u32(self,n-32)` → `vorrq_u32` (`:1601-1612`) |
| wasm32 (simd128) | native `[U32x4;4]` (`U32x4`=`v128`) fan-out (`simd_wasm.rs:949`) | delegates to `U32x4::add`=`u32x4_add` (`:1262-1268`, intrinsic `:919-921`) | delegates to `U32x4::bitxor`=`v128_xor` (`:1270-1280`, intrinsic `:923-925`) | fan `U32x4::rotate_left` ×4 (`:1227-1236`), each `v128_or(u32x4_shl(x,n), u32x4_shr(x,32-n))` (`:927-940`) |
| scalar | `[u32;16]` array, `impl_int_type!` (`simd_scalar.rs:337`, instantiated `:525`) | plain `wrapping_add` loop (macro `:383-391`) | plain `^` loop (macro) | per-lane `u32::rotate_left` loop (`:1332-1344`) |
| nightly `core::simd` | `core::simd::u32x16` newtype (`simd_nightly/u_word_types.rs`) | `Add` via `core::simd` op (not read this pass) | `BitXor` via `core::simd` op | per-lane `u32::rotate_left` loop, doc-noted "`core::simd` has no rotate" (`~24-30`, same pattern as the `U64x8` impl below) |

Verdict: **AVX-512 = ✅ native VPROLVD; AVX2/scalar/nightly = 🟡 polyfill-pass** (array + index loop, LLVM-autovectorized on the AVX2 tier — O.5); **NEON/wasm = ✅ native intrinsics** (hand-written shift-or — no rotate instruction to call directly).

**`U64x8` — rotate_left/rotate_right (BLAKE2b → argon2 lane).** The ONE lane this codebase calls "the crate's one earned intrinsic override" on AVX-512, and genuinely non-uniform across backends (unlike `U32x16` above, where every backend runs the SAME source shape — index loop — and only AVX-512 differs by intrinsic override):

| backend | storage | rotate_left(n) | rotate_right(n) | mechanism class |
|---|---|---|---|---|
| AVX-512 | native `__m512i` (`simd_avx512.rs:~1735`) | `_mm512_rolv_epi64` — VPROLVQ (`:1748-1757`) | `_mm512_rorv_epi64` — VPRORVQ (`:1768-1778`) | ✅ native, single instruction |
| AVX2 | `[u64;8]` array (`avx2_int_type!`), rotate is a **hand-written override**, not the macro body | 2×`_mm256_sll_epi64`/`_mm256_srl_epi64`/`_mm256_or_si256` per 256-bit half via `rotl_half` (`simd_avx2.rs:1634-1645`, called `:1650-1659`) | `rotl_half(v, 64-n)` (`:1663-1672`) | ✅ native intrinsics, hand-composed shift-or (LLVM does NOT autovec at u64 width — O.5) |
| NEON | native `[U64x2;4]` fan-out (`simd_neon.rs:2725`) | `vshlq_u64(x,+n)` \| `vshlq_u64(x,n-64)` → `vorrq_u64`, per quad (`:2801-2811`) | `self.rotate_left(64-n)` (`:2818-2825`) | ✅ native intrinsics |
| wasm32 (simd128) | native `[U64x2;4]` fan-out (`simd_wasm.rs:1786`) | `v128_or(i64x2_shl(x,n), u64x2_shr(x,64-n))`, per quad (`:1841-1848`) | `self.rotate_left(64-n)` (`:1851-1858`) | ✅ native intrinsics |
| scalar | `[u64;8]` array | per-lane `u64::rotate_left` loop (`simd_scalar.rs:544-556`) | per-lane `u64::rotate_right` loop (`:558-570`) | ✗ scalar, **measured NOT to autovectorize** (O.5) |
| nightly `core::simd` | `core::simd::u64x8` newtype (`u_word_types.rs:18`) | per-lane `u64::rotate_left` loop, "unmeasured whether this backend's codegen does better" (`:23-45`) | per-lane `u64::rotate_right` loop (`:47-58`) | ✗ scalar, explicitly marked unmeasured |

Verdict: **AVX-512 = ✅ native (earned override); AVX2/NEON/wasm = ✅ native hand-composed shift-or intrinsics (NOT autovec — LLVM declines the u64 rotate operation even with explicit `vpsllq`/`vpsrlq`-shaped source, O.5); scalar = ✗ correct-but-unaccelerated, known/accepted cost; nightly = ✗ same, explicitly unmeasured.**

### O.2 BLAKE3 shuffle/transpose surface on `U32x16` (built, currently unconsumed — see O.4)

Six methods, one impl block per backend, same names/semantics everywhere (`simd_avx512.rs:1477-1606`, `simd_avx2.rs:1754-1878`, `simd_neon.rs:1663-1793`, `simd_wasm.rs:1052-1183`, `simd_scalar.rs:1206-1330`; per `simd.rs:394-397` comment, NEON's is the arm actually dispatched on aarch64, not a separate scalar path).

| method | AVX-512 lowering (real, `simd_avx512.rs`) | other backends |
|---|---|---|
| `interleave_lo_u32` | `_mm512...` per-256-bit-half unpack (verified against real x86 intrinsics by `u32x16_blake3_shuffles_match_x86_intrinsics_per_half`, `simd.rs:1003-1046`) | index loop, same lane-exact semantics on AVX2/NEON/wasm/scalar (`simd_avx2.rs:1755-1768`) |
| `interleave_hi_u32` | ditto | ditto |
| `interleave_lo_u64` | ditto | ditto |
| `interleave_hi_u64` | ditto | ditto |
| `concat_lo_halves` | `_mm256_permute2x128_si256(_,_,0x20)` semantics per half | index-copy loop (`simd_avx2.rs:1821-1831`) |
| `concat_hi_halves` | `_mm256_permute2x128_si256(_,_,0x31)` semantics per half | index-copy loop |
| `exchange::<G>` | generic butterfly, `G∈{1,2,4,8}` compose a 16×16 transpose (proven by `u32x16_exchange_stages_compose_a_transpose`, `simd.rs:1096-1123`) | same generic index-swap on every backend (`simd_avx2.rs:1861-1878`) |

All six are 🟡 polyfill-pass on every backend except AVX-512, which is parity-tested against **real x86 intrinsics** it reproduces (`_mm256_unpacklo_epi32`/`_mm256_unpackhi_epi32`/`_mm256_unpacklo_epi64`/`_mm256_unpackhi_epi64`/`_mm256_permute2x128_si256`), not merely against a second index loop. **No in-tree crypto code calls any of these six methods** — O.4.

### O.3 In-tree crypto consumers

| primitive | crate/file | rides `ndarray::simd`? | type(s) | raw intrinsics / vendored own-SIMD? |
|---|---|---|---|---|
| **ChaCha20 keystream** (incl. XChaCha20) | `vendor/chacha20/src/backends/ndarray_simd.rs` (fork, patched over `chacha20poly1305 → chacha20`, `crates/encryption/src/aead.rs:17-19`) | **YES** — `use ndarray::simd::U32x16;` (`:19`) | `U32x16`: `Add`/`BitXor`/`rotate_left` only, transpose (vertical, 16-lanes-parallel) layout so **no shuffle needed** (`:60-99`) | No — module doc: "no raw intrinsics and no `unsafe` here" (`:5-6`), literally true of this file |
| ChaCha20 keystream, non-AVX-512/non-wasm128 path | `vendor/chacha20/src/backends/{avx2,sse2,soft}.rs` | **NO** — selected instead of `ndarray_simd` whenever the build is x86_64 but NOT `avx512f` (i.e. the repo's own x86-64-v3 distribution baseline, `.cargo/config-v3.toml`), per `cfg_if!` dispatch (`backends.rs:6-19`) | — | **YES** — `avx2.rs`/`sse2.rs` import `core::arch::x86_64::*` directly (`:9,11` each), vendored RustCrypto raw-intrinsic backends |
| ChaCha20 keystream, aarch64 path | `vendor/chacha20/src/backends/neon.rs` | **NO** — only reached if `chacha20_force_neon` cfg is set (not default); default aarch64 falls to `soft` (`backends.rs:33,36`) | — | **YES** — `core::arch::aarch64::*` (`:11`), vendored raw intrinsics, unrelated to `src/simd_neon.rs` |
| Poly1305 MAC | `poly1305` crate (transitive via `chacha20poly1305`) | **NO** | — | AVX2 backend exists upstream (424 `_mm*` calls, `src/backend/avx2/helpers.rs` per `.cargo/config.toml:96-98`) but **compiled out** via `--cfg poly1305_force_soft` (`.cargo/config.toml:114-119`); falls to poly1305-donna soft impl |
| Ed25519 sign/verify | `ed25519-dalek` → `curve25519-dalek` | **NO** | — | AVX2 "vector" backend exists upstream (57 `_mm*` calls, `backend/vector/{avx2/field.rs,packed_simd.rs}` per `.cargo/config.toml:69-74`) but **compiled out** via `--cfg curve25519_dalek_backend="serial"` (`.cargo/config.toml:92-97`) |
| X448 key agreement | AdaWorldAPI fork of `elliptic-curves` (`x448`, pinned by git rev, `crates/encryption/Cargo.toml:20`) | **NO** | — | `crypto-bigint` underneath, per `.cargo/config.toml:22`: "no foreign AVX2 for the polyfill to fix" — this dependency has no raw-intrinsic SIMD surface to gate |
| Argon2id KDF | `argon2` crate (`crates/encryption/src/kdf.rs:6`) | **NO** | — | `?` not traced this pass (external, not vendored); presumably pulls its own `blake2` for BLAKE2b compression, the exact target the O.1 `U64x8::rotate_{left,right}` doc comment names by name (`simd_avx512.rs:1753-1757`) but **is not wired to** |
| SHA-384 (hashing, HMAC/HKDF) | `sha2` crate (`crates/encryption/src/hash.rs:7`, `hkdf_sha384.rs`) | **NO** | — | `?` not traced; `sha2` 0.10's own x86_64/aarch64 backends are not mentioned anywhere in `.cargo/config.toml` (only dalek + poly1305 cfgs exist) — if `sha2` auto-selects a raw-intrinsic backend for SHA-512/384, it is **ungated**, unconfirmed this pass |
| BLAKE3 (in-tree hash) | `src/hpc/blake3.rs` (864 lines) | **NO — deliberately scalar-only** | none | No — module doc: "portable-only (no SIMD, no `unsafe`)" (`:3`); `hash_many`/`compress_subtree_wide`/the whole SIMD-batching machinery is explicitly **NOT transcribed** (`:620-627`), so the O.2 shuffle surface (built per `.claude/knowledge/blake3-on-ndarray-simd.md`) is never called from here |

### O.4 Coverage table

| primitive / method | AVX-512 | AVX2 | NEON | wasm128 | scalar | nightly | tested by |
|---|---|---|---|---|---|---|---|
| `U32x16::{Add,BitXor,rotate_left}` (ChaCha20/BLAKE3 quarter-round) | ✅ `_mm512_{add,xor,rolv}_epi32` | 🟡 array+loop, autovec (O.5) | ✅ `vaddq/veorq/vshlq_u32` ×4 | ✅ `u32x4_add/v128_xor/shl+shr+or` ×4 | ✗ scalar loop | 🟡 `core::simd` op + scalar rotate loop | `u32x16_arx_ops_match_scalar` (`simd.rs:907-931`) |
| `U64x8::{rotate_left,rotate_right}` (BLAKE2b/argon2) | ✅ `_mm512_{rolv,rorv}_epi64` | ✅ hand `sll/srl/or` per half | ✅ hand `vshlq_u64×2/vorrq_u64` | ✅ hand `i64x2_shl/u64x2_shr/v128_or` | ✗ scalar loop, **measured non-vectorizing** | ✗ scalar loop, unmeasured | `u64x8_arx_rotate_matches_scalar` (`simd.rs:950-984`) |
| `U32x16` BLAKE3 shuffle sextet (`interleave_*`/`concat_*`/`exchange`) | ✅ parity-tested vs real x86 intrinsics | 🟡 index-loop reference impl (same source as scalar) | 🟡 index-loop | 🟡 index-loop | 🟡 index-loop (the reference) | ? not traced this pass | `u32x16_blake3_shuffles_match_x86_intrinsics_per_half`, `u32x16_blake3_shuffles_are_lane_exact`, `u32x16_exchange_stages_compose_a_transpose`, `u32x16_exchange_is_lane_exact_per_granularity` (`simd.rs:1003-1157`) |
| ChaCha20 keystream (real cipher, not raw lane) | 🟡 rides `U32x16` — only reached when build is `avx512f` (rarely the default v3/AVX2 distribution baseline, O.5) | ✗ bypasses polyfill: vendored raw AVX2 intrinsics (`avx2.rs`) | ✗ bypasses polyfill unless `chacha20_force_neon` set; default falls to `soft` | 🟡 rides `U32x16` (native wasm128 lane) | ✗ vendored `soft.rs` (RustCrypto reference, own scalar) | — | `vendor/chacha20/tests/mod.rs` RFC 8439 vectors (contents not read this pass) + `crates/encryption` 48 `#[test]` fns exercising `XChaCha20Poly1305::{encrypt,decrypt}` end to end |
| Poly1305 MAC | ✗ soft only (AVX2 backend force-disabled repo-wide) | ✗ soft only | — (poly1305 has no NEON backend in RustCrypto's own crate, unconfirmed) | ? | ✗ soft (poly1305-donna) | — | same 48 `#[test]` fns (AEAD-level, not MAC-isolated) |
| Ed25519 sign/verify | ✗ "serial" (vector backend force-disabled) | ✗ same | ? not traced | ? | ✗ serial | — | `crates/encryption/src/sign.rs` tests (count not isolated) |
| SHA-384/HMAC/HKDF | ? unconfirmed — no gating cfg exists for `sha2` | ? | ? | ? | ? | — | RFC 4231 HMAC-SHA384 vectors per `hkdf_sha384.rs:9` |
| BLAKE3 (in-tree) | ✗ deliberately scalar-only, no SIMD at any tier | ✗ | ✗ | ✗ | ✗ (this IS the implementation) | — | official BLAKE3 test-vector JSON, `checked == 35` assertion (`src/hpc/blake3.rs:819`) |
| Widening/carry-less multiply (GF(2^130-5) Poly1305 field arith, GHASH-shaped) | 🟦 no such primitive anywhere | 🟦 | 🟦 | 🟦 | 🟦 | 🟦 | none — `grep -rn "clmul\|carryless\|pclmul\|ghash"` across `src/simd*.rs` returns zero hits |
| AES round primitives | 🟦 none found | 🟦 | 🟦 | 🟦 | 🟦 | 🟦 | none — `grep -in "aesenc\|aeskeygenassist"` returns zero hits (moot today: no in-tree AES consumer either) |

### O.5 GAPS — crypto primitives running scalar/raw-intrinsic where the polyfill could carry them, or that bypass it

1. **ChaCha20 on the repo's own default distribution baseline bypasses the polyfill entirely.** `vendor/chacha20/src/backends.rs:6-19`'s `cfg_if!` only auto-selects `ndarray_simd` (the `U32x16` path) when the build is `avx512f` or `wasm32+simd128`. The documented portable/distribution baseline is **v3 (AVX2)**, per `.cargo/config.toml`'s own header (`.cargo/config-v3.toml` "REQUIRED for anything you ship") — and on that exact tier, ChaCha20 falls to the vendored raw-AVX2 RustCrypto backend (`avx2.rs`, `core::arch::x86_64::*`), not `ndarray::simd`. Already documented as an open item in `.claude/knowledge/chacha20-vendoring-blast-radius.md:89,160,169` ("No CI job compiles the chacha20 AVX-512 backend").
2. **Same bypass on aarch64 by default.** `neon.rs` (raw `core::arch::aarch64::*`) is only reached under `chacha20_force_neon` (not default); without it, aarch64 falls to `soft.rs`. Neither path touches `src/simd_neon.rs`'s own, now-native `U32x16`.
3. **The BLAKE3 shuffle/transpose sextet (O.2) has zero real consumers.** Six methods, tested against real x86 intrinsics, doc-commented as "the BLAKE3 shuffle surface" (`simd.rs:989-996`), built per the 15-intrinsic census in `.claude/knowledge/blake3-on-ndarray-simd.md` — but the only in-tree BLAKE3 (`src/hpc/blake3.rs`) is explicitly, permanently scalar-only by scope decision (`:620-627`), and no other file imports these six methods (only non-test call sites are inside `src/simd.rs`'s own `#[cfg(test)]` module). Built and proven, then structurally stranded by a separate architectural decision (drop the external `blake3` crate to break a cargo cycle, per `.claude/knowledge/blake3-in-tree-measured.md` — "swapped... 1.3× on typical inputs and ~5× at 64 KB" accepted cost).
4. **`U64x8` rotate exists on all 6 backends now, but has no BLAKE2b/argon2 consumer to ride it.** The AVX-512 doc comment names BLAKE2b and argon2 explicitly (`simd_avx512.rs:1753-1757`), clearly built for that purpose — but `crates/encryption`'s `argon2` dependency is vanilla RustCrypto (`Cargo.toml:16`), not a vendored/patched fork like `chacha20`. No `vendor/argon2` or `vendor/blake2` directory. Same "built, unconsumed" shape as finding 3.
5. **No widening-multiply/carry-less-multiply primitive anywhere in the polyfill** (`src/simd_int_ops.rs`, `src/simd_ops.rs` both checked — zero `widening_mul`/`mul_wide`/`clmul` hits). Poly1305's field arithmetic (radix-2^26 accumulate-and-reduce mod 2^130-5) and any future GHASH/GCM work both need this shape and have nothing to call. Poly1305 is soft-only by *policy* (raw-intrinsic surface too large to audit, `.cargo/config.toml:93-119`) rather than by *capability gap* today — but even if the policy changed, no polyfill primitive exists to carry a from-scratch Poly1305 without hand intrinsics.
6. **`sha2` (SHA-384) is entirely untraced/ungated.** `.cargo/config.toml` only names two crypto cfgs (`curve25519_dalek_backend`, `poly1305_force_soft`) — nothing for `sha2`. Whether RustCrypto's `sha2` 0.10 auto-selects a raw x86_64/aarch64 intrinsic backend for SHA-512/384 (SHA-NI covers SHA-1/SHA-256 only, so any such backend would not be the hardware SHA extension) is **unconfirmed** — `sha2`'s own source is not vendored here and was not read.

### O.6 Doc drift (prior status docs vs current code)

1. **`.claude/knowledge/crypto-lane-status.md` (dated "MEASURED, 2026-07-28") is stale on its central claim.** Its headline — "The u64 ARX lane: DOES NOT EXIST... zero `rotate_left` or `rotate_right` methods on `U64x8`... on any backend" and its Summary row "u64 ARX (BLAKE2b → argon2): **absent on all 6 backends**" — is **false against current code**. `U64x8::{rotate_left,rotate_right}` exist on AVX-512, AVX2, NEON, wasm, scalar, and nightly (O.1), with a dedicated parity test (`u64x8_arx_rotate_matches_scalar`, `simd.rs:950-984`). The doc's own later "What SHIPPED, per backend (#268)" section is internally inconsistent with its own headline — never reconciled with the top of the same file.
2. **The same doc's "What SHIPPED" section is *itself* stale on NEON/wasm.** States "NEON and wasm re-export the scalar `U64x8` and are covered by that arm." Current code contradicts this directly: `simd.rs:407` (`pub use crate::simd_neon::{..., U64x8};`) and `simd.rs:422` (`pub use crate::simd_wasm::wasm32_simd::{..., U64x8};`) both re-export **native** `U64x8` types (`[U64x2;4]` fan-out over real NEON/wasm128 intrinsics — O.1). `simd.rs:398-406`'s own comment dates this to "the five-flavour audit of #306" and states the prior state explicitly: "both used to resolve to the scalar backend here" — the crypto-lane-status.md doc predates #306 and was never updated after it landed.
3. **The main matrix's own `U64x8`/`U32x16` rows (§A) carry a `†` footnote dated 2026-09-13/09-14 describing the same #306 audit** — so the storage-shape matrix (§A) is current on this point while the crypto-specific doc (`crypto-lane-status.md`) is not. Same underlying PR, two documents diverged.
4. **`.claude/knowledge/blake3-on-ndarray-simd.md` and `blake3-in-tree-measured.md` are NOT contradicted** by current code — both are explicit that the O.2 shuffle surface was built to close a 15-intrinsic census gap for a *hypothetical* SIMD BLAKE3 backend, and that the operator subsequently chose the scalar in-tree transcode instead (dependency-cycle reasons, not a performance verdict). This section's O.4/O.5 findings about the shuffle surface being unconsumed are consistent with, not contradicting, those two docs — flagged here only because a reader of the main matrix alone (no crypto section previously) would not know this capability exists at all.
5. **`.claude/knowledge/chacha20-vendoring-blast-radius.md` is consistent with current code** on the AVX2/CI-coverage gap (O.5.1) and additionally notes upstream `chacha20` 0.10.1 now ships its own `backends/avx512.rs` — **confirmed absent from this repo's vendored copy** (`ls vendor/chacha20/src/backends/` → `avx2.rs ndarray_simd.rs neon.rs soft.rs sse2.rs`, no `avx512.rs`), consistent with that doc's claim the vendored tree is one minor release behind the fork.

**Footnotes:** `?` cells (sha2 backend selection, poly1305 NEON, nightly shuffle sextet, X448 internals) were not traced to source this pass — external-crate source, not vendored here, or out of the time budget; not asserted either way. Scalar/nightly `U32x16`/`U64x8` are the SAME array-index-loop source shape described for AVX2 in O.1; "🟡 polyfill-pass" there means "delegates to the generic array-loop body", not "delegates to a different backend's intrinsics".

## P. `core::simd` counterpart column — `src/simd_nightly/`

> Read-only survey of the nightly (`core::simd`/portable-simd) backend.
> Classes: **NATIVE** (1:1 delegation to a `core::simd` API) ·
> **COMPOSED** (chains 2+ native `core::simd` ops to realize one polyfill
> method — no single primitive covers it) · **SCALAR-FALLBACK** (a lane
> loop / array op / `unsafe transmute`; no `core::simd` API invoked in the
> body) · **MISSING** (the stable AVX-512/AVX2 backend has the method; the
> nightly backend does not implement it at all). `_original_draft.rs` (674
> lines) exists in the directory but is dead code — not a `pub mod` in
> `mod.rs`, contributes nothing to the public surface, excluded below.
> Counts are **row counts** (a row often bundles several near-identical
> methods on one line), matching the source survey's own convention — not
> raw method counts, but the proportions are representative.

### P.1 Per-type summary

| type | NATIVE | COMPOSED | SCALAR-FALLBACK | MISSING | file |
|---|---|---|---|---|---|
| F32x16 | 25 | 0 | 1 (`gather`) | 1 (`cast_i32`) | f32_types.rs |
| F32x8 | 7 | 0 | 0 | 0 | f32_types.rs |
| F64x8 | 6 | 0 | 0 | 0 | f64_types.rs |
| F64x4 | 1 | 0 | 0 | 0 | f64_types.rs |
| F32Mask16 | 5 | 0 | 0 | 0 | masks.rs |
| F32Mask8 | 1 | 0 | 0 | 0 | masks.rs |
| F64Mask8 | 1 | 0 | 0 | 0 | masks.rs |
| F64Mask4 | 1 | 0 | 0 | 0 | masks.rs |
| ops.rs arithmetic/bitwise macros (cross-type: F32x16/F32x8/F64x8/F64x4, U8x32/U8x64/U16x32/U32x16/U32x8/U64x8/U64x4/I8x32/I8x64/I16x16/I16x32/I32x16/I64x8) | 4 | 0 | 0 | 0 | ops.rs |
| I16x16 | 6 | 1 | 2 (PartialEq, Display) | 0 | i_word_types.rs |
| I16x32 | ~6 | ~1 | ~2 | 0 (bundled row, same pattern as I16x16) | i_word_types.rs |
| I32x16 | 6 | 2 | 2 (PartialEq, Display) | 0 | i_word_types.rs |
| I64x8 | 2 | 1 | 1 (PartialEq/Display, bundled) | 0 | i_word_types.rs |
| I32x8 | 2 | 1 | 1 | 1 (Add/Sub/BitAnd/BitOr/BitXor/Not/*Assign, whole surface) | i_word_types.rs |
| I64x4 | 2 | 1 | 1 | 1 (same whole-surface gap) | i_word_types.rs |
| U64x8 | 5 | 2 | 1 (`rotate_left`/`rotate_right`) | 0 | u_word_types.rs |
| U64x4 | 1 | 1 (bundled) | 0 | 0 | u_word_types.rs |
| U32x8 | 3 | 2 | 0 | 0 | u_word_types.rs |
| U32x16 | 2 | 2 | 2 (interleave sextet, `exchange`) | 0 | u_word_types.rs |
| U16x32 | 3 | 1 | 0 | 0 | u_word_types.rs |
| U16x16 | 2 | 1 | 0 | 5 (Add/Sub/BitAnd/BitOr/BitXor/Not/*Assign; `zero()`; `shr`/`shl`; `mullo`; `permute2x128`/`blend_epi32`/`to_f32x8_lo`/`to_f32x8_hi`) | u_word_types.rs |
| I8x64 | 6 | 1 | 1 (PartialEq) | 0 | i8_types.rs |
| I8x32 | ~6 | ~1 | ~1 (bundled row) | 0 | i8_types.rs |
| U8x64 | 5 | 3 | 1 (`shr_epi16`/`shl_epi16`) | 0 | u8_types.rs |
| U8x32 | ~5 | ~3 | ~1 (bundled row) | 0 | u8_types.rs |
| BF16x16 | 0 | 0 | 4 (all methods) | 0 | bf16_types.rs |
| BF16x8 | 0 | 0 | 1 (bundled row, all methods) | 0 | bf16_types.rs |
| F16x16 | 0 | 0 | 6 (all methods) | 0 | f16_types.rs |
| I8x16 | 4 | 2 | 1 (PartialEq/Debug, bundled) | 0 | w1a_types.rs |
| U16x8 | 4 | 0 | 2 (`gather_u16`; PartialEq/Debug) | 0 | w1a_types.rs |
| U8x8 | 2 | 0 | 1 (PartialEq/Debug) | 0 | w1a_types.rs |
| free fns (`palette_lookup_u8x8`, `prefetch_read_t0/1/2`, `batch_packed_i4_16`) | 0 | 0 | 3 rows | 0 | w1a_types.rs |
| U8x64 exotic extension (`permute_bytes`, `shuffle_bytes`, `mask_blend`, `unpack_lo/hi_epi8`) | 0 | 0 | 4 | 0 | exotic_methods.rs |
| U8x32 exotic extension (same 4 methods) | 0 | 0 | 1 (bundled row) | 0 | exotic_methods.rs |
| **total (approx, row-count basis)** | **~110** | **~25** | **~39** | **~8** | — |

**Per-type SCALAR-FALLBACK concentration (structural, not incidental):**
- **BF16x16, BF16x8, F16x16 — 100% SCALAR-FALLBACK.** Every method on these three types is scalar, because `core::simd` has no half-precision lane type at all. The single largest, cleanest "whole-type" gap in the file.
- **U8x64/U8x32's `exotic_methods.rs` extension — 100% SCALAR-FALLBACK**, all 10 methods (5×2 widths), because `core::simd::Swizzle` requires compile-time-constant indices and cannot express a runtime permute/shuffle vector, and there is no portable "select-by-raw-bitmask" primitive.
- **U64x8's `rotate_left`/`rotate_right`** and **U32x16's six BLAKE3 transpose/interleave/`exchange` methods** — scalar by explicit choice (each doc-cites a codegen-oracle measurement showing the scalar-shaped SOURCE still compiles to packed shuffles on this backend) — counted SCALAR-FALLBACK per the source's own definition (no `core::simd` API invoked in source), with the caveat that compiled OUTPUT may still be vectorized by LLVM.
- **U8x64/U8x32's `shr_epi16`/`shl_epi16`** — `unsafe { core::mem::transmute }` reinterpret to 16-bit lanes + scalar shift loop; no portable "relane" primitive exists to reinterpret `Simd<u8,N>` as `Simd<u16,N/2>` directly.
- **U8x64/U16x8's `gather_u16`/`palette_lookup_u8x8`, F32x16's `gather`** — scalar loops over a runtime `&[T]` LUT with bounds checking, because `core::simd`'s gather primitives (`Simd::gather_or`/`gather_select`) only gather from a fixed-size array or another `Simd`, never a `*const T` base pointer or a runtime-length slice.

### P.2 `core::simd` GAPS — grouped by capability (candidates for upstream contribution to rust-lang/portable-simd)

1. **Bit-rotate on integer lanes** (`U64x8::rotate_left`/`rotate_right`, `U32x8::rotate_left`, `U32x16::rotate_left`). `core::simd` has `Shl`/`Shr` but no `rotate_left`/`rotate_right` method on `Simd<T,N>` for any integer width. Two of the four call sites (U32x8, U32x16) work around this with a shift-or composition (COMPOSED, not a true gap in practice); U64x8 instead reaches for a scalar loop despite having the same `Shl<Self>`/`Shr<Self>` impls available in the SAME file — suggesting the missing primitive is real enough that even this codebase didn't consistently work around it. AVX-512's `VPROLVQ`/`VPRORVQ` is the hardware analog.
2. **Gather from a runtime-length slice or raw pointer.** `Simd::gather_or`/`gather_select` require a fixed-size array/another `Simd` as the source; there is no `Simd<T,N>::gather_from_slice(&[T], indices) -> Self` or `unsafe fn gather_from_ptr(*const T, indices) -> Self`. Affects `F32x16::gather` (raw pointer + signed offsets, AVX-512-gather-shaped), `U16x8::gather_u16` and `palette_lookup_u8x8` (LUT-from-slice-by-index).
3. **Runtime (non-compile-time-constant) permute/shuffle.** `core::simd::Swizzle`/`simd_swizzle!` require the index pattern to be a `const`; there is no `Simd<T,N>::permute(self, idx: Simd<T,N>) -> Self` (cross-lane, VPERMB-shaped) or within-lane-shuffle-by-runtime-vector (PSHUFB-shaped) primitive. Affects `U8x64::permute_bytes`/`shuffle_bytes` and `U8x32::permute_bytes`/`shuffle_bytes`.
4. **Select-by-raw-integer-bitmask (not `Mask<T,N>`).** `U8x64::mask_blend`/`U8x32::mask_blend` take a raw `u64`/`u32` and manually test bits per lane; `core::simd`'s `Mask::select` exists but needs a `Mask<T,N>` constructed first (via `from_bitmask`, itself native — see `masks.rs`), so this specific gap is arguably closeable with existing primitives and reads more like a missed-native-op than a true `core::simd` absence.
5. **Lane-width reinterpretation ("relane") of an integer vector.** `Simd::from_bits`/`to_bits` only convert between float and same-lane-count unsigned-int of the SAME total width (e.g. `f32x16 <-> u32x16`); there is no portable way to reinterpret a `Simd<u8,64>` as `Simd<u16,32>` (same 512 total bits, different lane count/width) without `unsafe transmute`. Affects `U8x64::shr_epi16`/`shl_epi16` and the `U8x32` twins.
6. **Rounding/averaging unsigned-int arithmetic (`avg_epu8`-shaped).** No `SimdUint::avg`/rounding-average primitive in `core::simd`; must be composed by hand via `+1, >>1` after widening to avoid overflow. Affects `U8x64::pairwise_avg`/`U8x32::pairwise_avg` — explicitly documented as a gap in the source ("`core::simd` has no native `avg_epu8`").
7. **Half-precision (`bf16`/`f16`) SIMD lane type.** No `Simd<bf16,N>` or `Simd<f16,N>` exists in `core::simd` at all — `bf16_types.rs` and `f16_types.rs` are therefore ENTIRELY scalar emulation over `[u16;N]`, every constructor/conversion/broadcast included. The single largest capability gap by method count in the whole directory (10 methods across BF16x16/BF16x8, 6 methods on F16x16, all SCALAR-FALLBACK).
8. **Software prefetch hint.** No portable prefetch intrinsic in `core::simd` at all (inherently a `core::arch`-only concept, e.g. `_mm_prefetch`/ARM `PRFM`); `prefetch_read_t0`/`t1`/`t2` are deliberate empty-body no-ops on this backend, documented as such in the source.
9. **Widening/narrowing float→int truncating cast on `F32x16` specifically — NOT a `core::simd` gap, a polyfill omission.** `Simd<f32,N>::cast::<i32>()` IS available in `core::simd` (used correctly elsewhere in this same file family, e.g. `I32x16::from_i16_slice`'s `.cast::<i32>()`); `F32x16` alone never calls it to realize the stable-backend `cast_i32` method it is missing. Listed as MISSING, not as a `core::simd` gap.

**`core::arch`/raw-intrinsics check:** none found. Every `use` block across all 13 in-scope files was read in full and none imports `core::arch::*`/`std::arch::*` — this backend is portable by construction, per `mod.rs`'s own doc comment ("Wraps `core::simd::*` so miri can execute the polyfill paths... Intrinsics backends are opaque to miri; `core::simd` is not"). The one non-`core::simd`, non-portable-in-spirit escape hatch is `unsafe { core::mem::transmute }`, used exactly twice per width (`U8x64`/`U8x32`'s `shr_epi16`/`shl_epi16`, both in `u8_types.rs`) to reinterpret a byte array as a `u16` array for a scalar shift loop — not an arch intrinsic (compiles on every target Rust supports), but unsafe, and working around the "relane" gap (item 5) rather than using any SIMD API.

<details>
<summary>Full per-method `core::simd` counterpart table (13 files, click to expand)</summary>

#### f32_types.rs — F32x16 (16 lanes), F32x8 (8 lanes)

| type | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| F32x16 | splat | `core_f32x16::splat` | NATIVE | f32_types.rs:35 |
| F32x16 | from_array | `core_f32x16::from_array` | NATIVE | f32_types.rs:41 |
| F32x16 | from_slice | `core_f32x16::from_slice` | NATIVE | f32_types.rs:51 |
| F32x16 | to_array | `.to_array()` | NATIVE | f32_types.rs:57 |
| F32x16 | gather | none (per-lane raw pointer deref loop) | SCALAR-FALLBACK — portable-simd has no gather over a raw base pointer + index array (only `Simd::gather_or`/`gather_select` gather from a **slice**, not a `*const T` base); missing: a `Simd::<T,N>::gather_ptr`-shaped primitive | f32_types.rs:92-99 |
| F32x16 | copy_to_slice | `.copy_to_slice()` | NATIVE | f32_types.rs:109 |
| F32x16 | reduce_sum | `SimdFloat::reduce_sum` | NATIVE | f32_types.rs:117 |
| F32x16 | reduce_min | `SimdFloat::reduce_min` | NATIVE | f32_types.rs:123 |
| F32x16 | reduce_max | `SimdFloat::reduce_max` | NATIVE | f32_types.rs:129 |
| F32x16 | simd_min | `SimdFloat::simd_min` | NATIVE | f32_types.rs:137 |
| F32x16 | simd_max | `SimdFloat::simd_max` | NATIVE | f32_types.rs:143 |
| F32x16 | simd_clamp | `SimdFloat::simd_clamp` | NATIVE | f32_types.rs:149 |
| F32x16 | mul_add | `StdFloat::mul_add` | NATIVE | f32_types.rs:160 |
| F32x16 | sqrt | `StdFloat::sqrt` | NATIVE | f32_types.rs:166 |
| F32x16 | round | `StdFloat::round` | NATIVE | f32_types.rs:172 |
| F32x16 | floor | `StdFloat::floor` | NATIVE | f32_types.rs:178 |
| F32x16 | abs | `SimdFloat::abs` | NATIVE | f32_types.rs:184 |
| F32x16 | to_bits | `SimdFloat::to_bits` | NATIVE | f32_types.rs:192 |
| F32x16 | from_bits | `core_f32x16::from_bits` | NATIVE | f32_types.rs:198 |
| F32x16 | simd_eq | `SimdPartialEq::simd_eq` | NATIVE | f32_types.rs:209 |
| F32x16 | simd_ne | `SimdPartialEq::simd_ne` | NATIVE | f32_types.rs:215 |
| F32x16 | simd_lt | `SimdPartialOrd::simd_lt` | NATIVE | f32_types.rs:221 |
| F32x16 | simd_le | `SimdPartialOrd::simd_le` | NATIVE | f32_types.rs:227 |
| F32x16 | simd_gt | `SimdPartialOrd::simd_gt` | NATIVE | f32_types.rs:233 |
| F32x16 | simd_ge | `SimdPartialOrd::simd_ge` | NATIVE | f32_types.rs:239 |
| F32x16 | Default::default | (delegates to splat) | NATIVE | f32_types.rs:246 |
| F32x16 | cast_i32 | none — absent from nightly entirely | **MISSING** — stable AVX-512 backend has `cast_i32` (truncating f32→i32 lane cast, `simd_avx512.rs:198`); `core::simd` DOES have this (`Simd<f32,N>::cast::<i32>()` via `SimdCast`/`SimdElement`, a genuine polyfill omission, not a core::simd gap) | avx512.rs:198 (absent in f32_types.rs) |
| F32x8 | splat/from_array/from_slice/to_array/copy_to_slice | same core::simd ops, f32x8 | NATIVE | f32_types.rs:270-303 |
| F32x8 | reduce_sum/min/max | `SimdFloat::reduce_*` | NATIVE | f32_types.rs:310,316,322 |
| F32x8 | simd_min/max/clamp | `SimdFloat::simd_*` | NATIVE | f32_types.rs:330,336,342 |
| F32x8 | mul_add/sqrt/round/floor/abs | `StdFloat`/`SimdFloat` | NATIVE | f32_types.rs:350,356,362,368,374 |
| F32x8 | to_bits/from_bits | `SimdFloat::to_bits`/`core_f32x8::from_bits` | NATIVE | f32_types.rs:382,388 |
| F32x8 | simd_eq/ne/lt/le/gt/ge | `SimdPartialEq`/`SimdPartialOrd` | NATIVE | f32_types.rs:399-429 |
| F32x8 | Default::default | (delegates to splat) | NATIVE | f32_types.rs:436 |

Arithmetic operators (`Add`/`Sub`/`Mul`/`Div`/`Neg` + `*Assign`) for F32x16/F32x8 live in `ops.rs`'s `impl_fp_ops!` macro (invoked `ops.rs:231-232`), delegating straight to `core::simd`'s own `Add`/`Sub`/`Mul`/`Div`/`Neg` impls on `Simd<f32,N>`. NATIVE.

#### f64_types.rs — F64x8 (8 lanes), F64x4 (4 lanes)

Every method delegates 1:1 to `core::simd::{SimdFloat,SimdPartialEq,SimdPartialOrd,StdFloat}` on `core_f64x8`/`core_f64x4` — all NATIVE, no exceptions, no scalar fallback, no arch intrinsics.

| type | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| F64x8 | splat/from_slice/from_array/to_array/copy_to_slice | `core_f64x8::{splat,from_slice,from_array,to_array,copy_to_slice}` | NATIVE | f64_types.rs:31-58 |
| F64x8 | reduce_sum/reduce_min/reduce_max | `SimdFloat::reduce_*` | NATIVE | f64_types.rs:65,71,77 |
| F64x8 | simd_min/simd_max/simd_clamp | `SimdFloat::simd_*` | NATIVE | f64_types.rs:85,91,97 |
| F64x8 | mul_add/sqrt/round/floor/abs | `StdFloat`/`SimdFloat` | NATIVE | f64_types.rs:105,111,117,123,129 |
| F64x8 | to_bits | `SimdFloat::to_bits` | NATIVE | f64_types.rs:137 |
| F64x8 | simd_eq/ne/lt/le/gt/ge | `SimdPartialEq`/`SimdPartialOrd` | NATIVE | f64_types.rs:145-175 |
| F64x4 | (same set, half width) | same core::simd ops on `core_f64x4` | NATIVE | f64_types.rs:200-344 |

`F64x8`/`F64x4` have `to_bits` but no `from_bits` constructor in EITHER the nightly file or a quick grep of avx512.rs — symmetric absence, not a nightly-specific gap. `Add`/`Sub`/`Mul`/`Div`/`Neg`/assign + `Default` via `ops.rs`'s `impl_fp_ops!`/`impl_default!` (`ops.rs:235-238`) — NATIVE.

#### masks.rs — F32Mask16/F32Mask8/F64Mask8/F64Mask4

All four mask types wrap `core::simd::Mask<iN, LANES>` and every method is a direct passthrough.

| type | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| F32Mask16 | to_bitmask | `Mask::to_bitmask` | NATIVE | masks.rs:25 |
| F32Mask16 | from_bitmask | `Mask::<i32,16>::from_bitmask` | NATIVE | masks.rs:31 |
| F32Mask16 | select | `core::simd::prelude::Select` (`Mask::select`) | NATIVE | masks.rs:38 |
| F32Mask16 | all | `Mask::all` | NATIVE | masks.rs:44 |
| F32Mask16 | any | `Mask::any` | NATIVE | masks.rs:50 |
| F32Mask8 | to_bitmask/from_bitmask/select/all/any | same, `Mask<i32,8>` | NATIVE | masks.rs:67-92 |
| F64Mask8 | to_bitmask/from_bitmask/select/all/any | same, `Mask<i64,8>` | NATIVE | masks.rs:112-137 |
| F64Mask4 | to_bitmask/from_bitmask/select/all/any | same, `Mask<i64,4>` | NATIVE | masks.rs:153-179 |

No `!`/`&`/`\|`/`^` operator impls on masks in this file (no `impl BitAnd for F32Mask16` etc.).

#### ops.rs — Add/Sub/Mul/Div/Neg/bitwise operator macros

Both macros (`impl_fp_ops!`, `impl_int_ops!`, `impl_int_neg!`, `impl_default!`) are 1:1 delegations to `core::simd`'s own `Add`/`Sub`/`Mul`/`Div`/`Neg`/`BitAnd`/`BitOr`/`BitXor`/`Not` trait impls on the wrapped `Simd<T,N>` — every invocation NATIVE, no exceptions.

| type(s) | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| F32x16, F32x8, F64x8, F64x4 | Add/Sub/Mul/Div/Neg + *Assign | `Simd<T,N>`'s own `core::ops` impls | NATIVE | ops.rs:25-91 (macro), invoked 231-238 |
| U8x32,U8x64,U16x32,U32x16,U32x8,U64x8,U64x4,I8x32,I8x64,I16x16,I16x32,I32x16,I64x8 | Add/Sub/BitAnd/BitOr/BitXor/Not + *Assign | `Simd<T,N>`'s own int `core::ops` impls | NATIVE | ops.rs:107-187 (macro), invoked 245-277 |
| I8x32,I8x64,I16x16,I16x32,I32x16,I64x8 | Neg | `Simd<T,N>::neg` (signed only) | NATIVE | ops.rs:196-206 |
| F64x8,F64x4,I8x32,I8x64,I16x16,I16x32,I32x16,I64x8 | Default | `Simd::<T,N>::default()` | NATIVE | ops.rs:215-224 |

**GAP** (confirmed by reading `ops.rs`'s invocation list, lines 230-277, against `u_word_types.rs`/`i_word_types.rs` struct definitions): `U16x16`, `I32x8`, `I64x4` (the "256-bit aliases for the missing-lanes sweep", `mod.rs:90-96`) have NO invocation of `impl_int_ops!`/`impl_default!` anywhere, and none is implemented inline in their own `_types.rs` files either (confirmed for `U16x16` by reading `u_word_types.rs:896-969` in full). `simd_avx2.rs`'s `U16x16` has the full arithmetic+bitwise+Not surface plus `zero()`, `shr`/`shl`, `mullo`, `reduce_sum`(as u32), `permute2x128`, `blend_epi32`, `to_f32x8_lo`/`to_f32x8_hi` — see below.

#### i_word_types.rs — I16x16, I16x32, I32x16, I64x8, I32x8, I64x4

| type | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| I16x16 | splat/zero/from_array/from_slice/to_array/copy_to_slice | `core::simd::i16x16::{splat,from_array,from_slice,to_array,copy_to_slice}` (zero = splat(0)) | NATIVE | i_word_types.rs:23-52 |
| I16x16 | reduce_sum/reduce_min/reduce_max | `SimdInt::reduce_sum`, `SimdOrd`-derived | NATIVE | i_word_types.rs:60,65,69 |
| I16x16 | simd_min/simd_max | `SimdOrd::simd_min`/`simd_max` | NATIVE | i_word_types.rs:77,81 |
| I16x16 | min/max (custom names) | thin aliases for simd_min/max | NATIVE | i_word_types.rs:87-95 |
| I16x16 | add/sub (custom names) | `core::ops::Add`/`Sub` on `Simd<i16,16>` (wrapping) | NATIVE | i_word_types.rs:99-107 |
| I16x16 | saturating_add/saturating_sub | `SimdInt::saturating_add`/`saturating_sub` | NATIVE | i_word_types.rs:113,118 |
| I16x16 | cmpeq_mask/cmpgt_mask/cmp_gt | `SimdPartialEq::simd_eq`/`SimdPartialOrd::simd_gt` then `.to_bitmask()` | COMPOSED | i_word_types.rs:127,133,138 |
| I16x16 | PartialEq::eq | `to_array() == to_array()` | SCALAR-FALLBACK — `Simd<T,N>` DOES implement `PartialEq` natively; missed-native-op, not a missing capability | i_word_types.rs:145 |
| I16x16 | Display::fmt | `{:?}` over `to_array()` | SCALAR-FALLBACK (formatting, expected) | i_word_types.rs:151-153 |
| I16x32 | (mirrors I16x16 at 32 lanes) | same core::simd ops on `i16x32` | NATIVE (arith/reduce/cmp) / COMPOSED (mask methods) / SCALAR-FALLBACK (PartialEq, Display) — same pattern as I16x16 | i_word_types.rs:172-301 |
| I32x16 | splat/from_array/from_slice/to_array/copy_to_slice | `core::simd::i32x16::*` | NATIVE | i_word_types.rs:320-343 |
| I32x16 | reduce_sum/min/max, simd_min/max | `SimdInt`/`SimdOrd` | NATIVE | i_word_types.rs:350,355,360,367,372 |
| I32x16 | cmpeq_mask/cmpgt_mask/gt_bitmask | `SimdPartialEq::simd_eq`/`SimdPartialOrd::simd_gt` + `to_bitmask()` | COMPOSED | i_word_types.rs:381,387,407 |
| I32x16 | from_i16_slice | `i16x16::from_slice(..).cast::<i32>()` | NATIVE — `Simd::cast` is a first-class widening/narrowing conversion op | i_word_types.rs:430 |
| I32x16 | to_i16_array | `self.0.cast::<i16>().to_array()` | NATIVE — same `Simd::cast` narrowing op | i_word_types.rs:445 |
| I32x16 | abs | `SimdInt::abs` (wraps at `i32::MIN`, matching VPABSD) | NATIVE | i_word_types.rs:461 |
| I32x16 | cmpge_zero_mask | `SimdPartialOrd::simd_ge` against `splat(0)` + `to_bitmask()` | COMPOSED | i_word_types.rs:479 |
| I32x16 | PartialEq::eq | array compare | SCALAR-FALLBACK (missed-native-op) | i_word_types.rs:485 |
| I32x16 | Mul/MulAssign | `core::ops::Mul` on `Simd<i32,16>` (VPMULLD-equivalent) | NATIVE | i_word_types.rs:496,503 |
| I32x16 | Display::fmt | formatting | SCALAR-FALLBACK (expected) | i_word_types.rs:509 |
| I64x8 | splat/from_array/from_slice/to_array/copy_to_slice | `core::simd::i64x8::*` | NATIVE | i_word_types.rs:529-552 |
| I64x8 | reduce_sum/min/max, simd_min/max | `SimdInt`/`SimdOrd` | NATIVE | i_word_types.rs:559,564,569,576,581 |
| I64x8 | cmpeq_mask/cmpgt_mask | `SimdPartialEq`/`SimdPartialOrd` + `to_bitmask()` | COMPOSED | i_word_types.rs:590,596 |
| I64x8 | PartialEq::eq / Display::fmt | array compare / formatting | SCALAR-FALLBACK | i_word_types.rs:603,609 |
| I32x8 | splat/from_slice/from_array/to_array/copy_to_slice/reduce_*/simd_min/max/cmpeq_mask/cmpgt_mask | `core::simd::i32x8::*`, traits | NATIVE (ctors/reduce/min-max) / COMPOSED (mask methods) | i_word_types.rs:629-683 |
| I32x8 | Default::default | `Self::splat(0)` | NATIVE | i_word_types.rs:689 |
| I32x8 | PartialEq::eq | array compare | SCALAR-FALLBACK | i_word_types.rs:696 |
| I32x8 | Add/Sub/BitAnd/BitOr/BitXor/Not/*Assign | `core::ops::{Add,...}` on `Simd<i32,8>` | **MISSING** from nightly (present on stable avx2/avx512 backend); not invoked via `ops.rs` (only I16x16/I16x32/I32x16/I64x8 are in the signed-int block, `ops.rs:266-277`), not implemented inline | n/a — absent |
| I64x4 | splat/from_slice/from_array/to_array/copy_to_slice/reduce_*/simd_min/max/cmpeq_mask/cmpgt_mask | `core::simd::i64x4::*` | NATIVE / COMPOSED (mask methods) | i_word_types.rs:716-770 |
| I64x4 | Default::default | `Self::splat(0)` | NATIVE | i_word_types.rs:776 |
| I64x4 | PartialEq::eq | array compare | SCALAR-FALLBACK | i_word_types.rs:783 |
| I64x4 | Add/Sub/BitAnd/BitOr/BitXor/Not/*Assign | `core::ops::{Add,...}` on `Simd<i64,4>` | **MISSING** from nightly (present on stable backend) | n/a — absent |

`I32x8`/`I64x4` lack the richer `I32x16`-only surface (`from_i16_slice`, `to_i16_array`, `abs`, `cmpge_zero_mask`, `gt_bitmask`, `cmp_gt`) and `I16x16`/`I16x32`'s `zero()`/`min`/`max`/`add`/`sub`(named)/`saturating_add`/`saturating_sub` — not flagged individually as MISSING (narrower-tier additions on the wider types, not cross-checked exhaustively against `simd_avx2.rs`'s `I32x8`/`I64x4` given scope).

#### u_word_types.rs — U64x8, U64x4, U32x8, U32x16, U16x32, U16x16

| type | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| U64x8 | rotate_left/rotate_right | none — per-lane `u64::rotate_left`/`rotate_right` loop | **SCALAR-FALLBACK** — explicit in the doc comment: "`core::simd` has no rotate... whether this backend's codegen does better is unmeasured". Missing: a portable `Simd<T,N>::rotate_lanes_left_by(count)`-shaped variable-rotate primitive (AVX-512 has `VPROLVQ`/`VPRORVQ`, why the avx512 backend overrides this with a real intrinsic) | u_word_types.rs:35-63 |
| U64x8 | splat/from_slice/from_array/to_array/copy_to_slice | `core::simd::u64x8::*` | NATIVE | u_word_types.rs:67-90 |
| U64x8 | reduce_sum/min/max, simd_min/max | `SimdUint`/`SimdOrd` | NATIVE | u_word_types.rs:96,101,106,113,118 |
| U64x8 | cmpeq_mask/cmpgt_mask | `SimdPartialEq`/`SimdPartialOrd` + `to_bitmask()` | COMPOSED | u_word_types.rs:126,132 |
| U64x8 | popcnt | `SimdUint::count_ones()` | NATIVE | u_word_types.rs:147 |
| U64x8 | xor_popcount | `BitXor`(`^`) → `SimdUint::count_ones()` → `SimdUint::reduce_sum()` | COMPOSED — three chained core::simd ops | u_word_types.rs:164 |
| U64x8 | Shl\<Self\>/Shr\<Self\> (variable per-lane shift) | `core::simd`'s own `Shl<Self>`/`Shr<Self>` on `Simd<u64,8>` | NATIVE (with a `debug_assert!` count-contract wrapper) | u_word_types.rs:186,199 |
| U64x8 | Default::default | `Self::splat(0)` | NATIVE | u_word_types.rs:206 |
| U64x4 | (same set minus rotate/xor_popcount/Shl/Shr) | same `core::simd::u64x4` ops | NATIVE (ctors/reduce/min-max/popcnt/Default) / COMPOSED (cmp masks) | u_word_types.rs:226-313 |
| U32x8 | splat/from_slice/from_array/to_array/copy_to_slice/reduce_*/simd_min/max | `core::simd::u32x8::*`, traits | NATIVE | u_word_types.rs:332-384 |
| U32x8 | cmpeq_mask/cmpgt_mask | `SimdPartialEq`/`SimdPartialOrd` + `to_bitmask()` | COMPOSED | u_word_types.rs:391,397 |
| U32x8 | rotate_left | `n%32==0` guard + `(self.0 << splat(n)) \| (self.0 >> splat(32-n))` | **COMPOSED** — shift-or rotate built from native `Shl`/`Shr`/`BitOr`; no scalar loop (contrast with U64x8's rotate above, which DOES use a scalar loop for the same conceptual op — an inconsistency between two files in this same module) | u_word_types.rs:413-419 |
| U32x8 | interleave_lo_u32/interleave_hi_u32/interleave_lo_u64/interleave_hi_u64/concat_lo_halves/concat_hi_halves | `core::simd::simd_swizzle!` (compile-time two-source shuffle) | NATIVE | u_word_types.rs:442,451,460,469,478,487 |
| U32x8 | Default::default | `Self::splat(0)` | NATIVE | u_word_types.rs:494 |
| U32x16 | interleave_lo_u32/interleave_hi_u32/interleave_lo_u64/interleave_hi_u64/concat_lo_halves/concat_hi_halves | none — explicit `to_array()`/index-loop/`from_array()` round-trip, NOT `simd_swizzle!` (contrast with U32x8's version) | **SCALAR-FALLBACK by the file's own admission**, footnoted: the enclosing `impl` block's doc comment (`u_word_types.rs:509-537`) states the codegen oracle measured that fixed two-source permutations written as index loops compile to real packed shuffles (`vpunpcklqdq`/`vpermq`/`vinserti128`) — scalar-SHAPED at the source level, native OUTPUT. Missing at the *source* level: a portable two-source-permute-within-128-bit-groups primitive | u_word_types.rs:542-633 |
| U32x16 | exchange\<const G: usize\> | none — index loop keyed by `c & G` | SCALAR-FALLBACK, same caveat (doc comment cites 79 packed / 0 scalar-lane-arith when composed 4× for a 16×16 transpose). Missing: a generic `Simd<T,N>::butterfly_exchange<const G: usize>` swizzle primitive | u_word_types.rs:656-665 |
| U32x16 | splat/from_slice/from_array/to_array/copy_to_slice/reduce_*/simd_min/max | `core::simd::u32x16::*`, traits | NATIVE | u_word_types.rs:673-725 |
| U32x16 | cmpeq_mask/cmpgt_mask/eq_bitmask | `SimdPartialEq`/`SimdPartialOrd` + `to_bitmask()` (`eq_bitmask` is a renamed duplicate of `cmpeq_mask`) | COMPOSED | u_word_types.rs:732,738,757 |
| U32x16 | rotate_left | shift-or composition, same shape as U32x8's | COMPOSED — native `Shl`/`Shr`/`BitOr` on `Simd<u32,16>` | u_word_types.rs:771 |
| U32x16 | Default::default | `Self::splat(0)` | NATIVE | u_word_types.rs:778 |
| U16x32 | splat/from_slice/from_array/to_array/copy_to_slice/reduce_*/simd_min/max | `core::simd::u16x32::*`, traits | NATIVE | u_word_types.rs:798-850 |
| U16x32 | saturating_add/saturating_sub | `SimdUint::saturating_add`/`saturating_sub` | NATIVE | u_word_types.rs:857,862 |
| U16x32 | cmpeq_mask/cmpgt_mask | `SimdPartialEq`/`SimdPartialOrd` + `to_bitmask()` | COMPOSED | u_word_types.rs:870,876 |
| U16x32 | Default::default | `Self::splat(0)` | NATIVE | u_word_types.rs:883 |
| U16x16 | splat/from_slice/from_array/to_array/copy_to_slice/reduce_*/simd_min/max | `core::simd::u16x16::*`, traits | NATIVE | u_word_types.rs:902-951 |
| U16x16 | cmpeq_mask/cmpgt_mask | `SimdPartialEq`/`SimdPartialOrd` + `to_bitmask()` | COMPOSED | u_word_types.rs:954,960 |
| U16x16 | Default::default | `Self::splat(0)` | NATIVE | u_word_types.rs:967 |
| U16x16 | Add/Sub/BitAnd/BitOr/BitXor/Not/*Assign | `core::ops::{Add,...}` on `Simd<u16,16>` | **MISSING** from nightly (present on stable avx2 backend) — confirmed by reading full `impl U16x16` block, `u_word_types.rs:898-969`, no `impl core::ops::*` anywhere, not in `ops.rs`'s invocation list | n/a — absent |
| U16x16 | zero() | (would be `Self::splat(0)`) | **MISSING** (present on avx2's U16x16 and on nightly's OWN I16x16/I16x32) | n/a — absent |
| U16x16 | shr(imm)/shl(imm) (fixed-count named shift) | `core::simd`'s `Shl<u32>`/`Shr<u32>`-shaped ops (would need `Simd::shl`/`shr` by a splat) | **MISSING** — `core::simd` DOES support this via the same `Shl<Self>`/`Shr<Self>` pattern U64x8 already demonstrates in this same file | n/a — absent |
| U16x16 | mullo | `core::ops::Mul` on `Simd<u16,16>` (wrapping low-16 multiply) | **MISSING** — `core::simd` supports `Mul` for integer types (as I32x16 demonstrates) | n/a — absent |
| U16x16 | permute2x128/blend_epi32/to_f32x8_lo/to_f32x8_hi | `simd_swizzle!` / bit-select / cast | **MISSING**, not cross-checked against a `core::simd` counterpart individually given scope | n/a — absent |

`simd_avx2.rs`'s `U16x16` (`impl Add for U16x16` etc., `avx2.rs` lines ~119-198) has the full arithmetic + bitwise surface plus `zero()`, `shr`, `shl`, `mullo`, `reduce_sum` (as **u32**, not u16 — a WIDENING reduce the nightly version does not replicate; nightly's `reduce_sum` truncates to `u16`), `permute2x128`, `blend_epi32`, `to_f32x8_lo`/`to_f32x8_hi`.

#### i8_types.rs — I8x64 (64 lanes), I8x32 (32 lanes)

Every method is a direct delegation; no scalar loops, no arch intrinsics.

| type | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| I8x64 | splat/zero/from_slice/from_array/to_array/copy_to_slice | `core::simd::i8x64::*` (zero = splat(0)) | NATIVE | i8_types.rs:37-70 |
| I8x64 | reduce_sum/min/max | `SimdInt::reduce_sum`, `SimdOrd`-family | NATIVE | i8_types.rs:77,83,89 |
| I8x64 | simd_min/simd_max, min/max (aliases) | `SimdOrd::simd_min`/`simd_max` | NATIVE | i8_types.rs:97,103,120,135 |
| I8x64 | add/sub (named wrappers) | `core::ops::Add`/`Sub` on `Simd<i8,64>` (wrapping) | NATIVE | i8_types.rs:141,147 |
| I8x64 | saturating_abs | `SimdInt::saturating_abs` (correctly saturates `\|i8::MIN\|` to `i8::MAX`, unlike VPABSB — noted correct-by-construction) | NATIVE | i8_types.rs:165 |
| I8x64 | saturating_add/saturating_sub | `SimdInt::saturating_add`/`saturating_sub` | NATIVE | i8_types.rs:173,179 |
| I8x64 | cmpeq_mask/cmpgt_mask/cmp_gt | `SimdPartialEq::simd_eq`/`SimdPartialOrd::simd_gt` + `to_bitmask()` (`cmp_gt` aliases `cmpgt_mask`) | COMPOSED | i8_types.rs:189,197,203 |
| I8x64 | PartialEq::eq | array compare | SCALAR-FALLBACK (missed-native-op) | i8_types.rs:209 |
| I8x32 | (identical method set, 32 lanes) | same core::simd ops on `i8x32` | NATIVE (ctors/reduce/min-max/add/sub/saturating_*) / COMPOSED (cmp masks) / SCALAR-FALLBACK (PartialEq) | i8_types.rs:240-413 |

No arithmetic `Mul`/bitwise ops defined inline for I8x64/I8x32 (from `ops.rs`'s macros, `ops.rs:259-264` — NATIVE, already tabulated). No MISSING found on a spot check against `simd_avx512.rs`'s `I8x64`/`I8x32` (not exhaustively diffed given scope, but the API is unusually complete — `zero`, `min`/`max`, `add`/`sub`, `saturating_abs`, `cmp_gt` all covered).

#### u8_types.rs — U8x64 (64 lanes), U8x32 (32 lanes)

| type | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| U8x64 | splat/from_slice/from_array/to_array/copy_to_slice | `core_u8x64::{splat,from_slice,from_array,to_array,copy_to_slice}` | NATIVE | u8_types.rs:48-112 |
| U8x64 | reduce_sum/reduce_min/reduce_max | `SimdUint`/`SimdOrd` | NATIVE | u8_types.rs:128,144,160 |
| U8x64 | sum_bytes_u64 | `Simd::cast::<u16>()` (widen) then `SimdUint::reduce_sum()` | COMPOSED — two chained core::simd ops, sidesteps u8 wraparound | u8_types.rs:178-179 |
| U8x64 | simd_min/simd_max | `SimdOrd::simd_min`/`simd_max` | NATIVE | u8_types.rs:197,213 |
| U8x64 | saturating_add/saturating_sub | `SimdUint::saturating_add`/`saturating_sub` | NATIVE | u8_types.rs:231,247 |
| U8x64 | pairwise_avg | cast to `Simd<u16,64>`, `+`, `+Simd::splat(1)`, `>> Simd::splat(1)`, cast back to u8 | **COMPOSED, doc explains why**: "`core::simd` has no native `avg_epu8`; computed via u16 promotion to avoid overflow. LLVM MAY lower to `vpavgb`." — a genuine `core::simd` capability gap | u8_types.rs:266-271 |
| U8x64 | shr_epi16/shl_epi16 | none — `unsafe { core::mem::transmute }` reinterpret to `[u16;32]`, plain per-element `for w in words.iter_mut() { *w >>= imm }` scalar loop, transmute back | **SCALAR-FALLBACK** — no `core::simd` call at all in the body (not even a `Simd<u16,32>` shift, which WOULD be available); the missing piece is a byte-vector reinterpreted-as-16-bit-lanes shift — `core::simd` has no `bitcast`/`transmute` between differently-laned `Simd<T,N>` types of the same total width | u8_types.rs:290-299, 315-322 |
| U8x64 | cmpeq_mask/cmpgt_mask/movemask | `SimdPartialEq`/`SimdPartialOrd` (movemask = `simd_gt(splat(0x7F))`) + `to_bitmask()` | COMPOSED | u8_types.rs:341,357,375 |
| U8x64 | nibble_popcount_lut | `Self::from_array([..])` — compile-time constant table | NATIVE (trivial, no arithmetic op) | u8_types.rs:399-402 |
| U8x64 | Default::default | `Self::splat(0)` | NATIVE | u8_types.rs:409 |
| U8x32 | (identical method set, 32 lanes / u16×16 reinterpret) | same core::simd ops, `core_u8x32`/`Simd<u16,32>`/`Simd<u16,16>` | same classification pattern as U8x64: NATIVE (ctors/reduce/min-max/saturating) / COMPOSED (sum_bytes_u64, pairwise_avg, cmp masks, movemask) / SCALAR-FALLBACK (shr_epi16/shl_epi16, `unsafe transmute` + scalar loop) | u8_types.rs:453-810 |

No `core::arch`/`std::arch` import or call anywhere in this file (only `core::simd::*` and `core::mem::transmute`) — confirmed by reading the full file including all `use` statements (`u8_types.rs:4-6`).

#### bf16_types.rs — BF16x16, BF16x8 (WHOLE-TYPE SCALAR EMULATION)

Module doc: "`core::simd` has no native half-precision type, so bit patterns are stored as `u16` and operations upcast through `f32` where needed." Storage is a plain `[u16; N]`, not `core::simd::Simd<u16,N>` — NOT ONE method in this file invokes any `core::simd` API at all.

| type | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| BF16x16 | splat | `f32_to_bf16_bits` (scalar `f32::to_bits() >> 16` truncation) broadcast into a plain array literal | **SCALAR-FALLBACK** — no `core::simd` half type exists to splat into | bf16_types.rs:44-46 |
| BF16x16 | from_slice/from_array/to_array/copy_to_slice | plain array copy (`copy_from_slice`, direct field access) | **SCALAR-FALLBACK** — plain `[u16;16]` array ops, no vector type involved | bf16_types.rs:50-74 |
| BF16x16 | to_f32_lossy | per-lane scalar loop calling `bf16_bits_to_f32` (`f32::from_bits((bits as u32) << 16)`) | **SCALAR-FALLBACK** — explicit `for i in 0..16` loop, zero vector ops | bf16_types.rs:78-84 |
| BF16x16 | from_f32_truncate | per-lane scalar loop calling `f32_to_bf16_bits` (`v.to_bits() >> 16`) | **SCALAR-FALLBACK** — explicit `for i in 0..16` loop | bf16_types.rs:88-94 |
| BF16x8 | (identical method set, 8 lanes) | same scalar helpers | **SCALAR-FALLBACK**, all methods | bf16_types.rs:107-157 |

**core::simd GAP, whole-family:** no half-precision (`bf16`) `Simd<T,N>` type exists in `core::simd` at all — every arithmetic/conversion op on BF16x16/BF16x8 in this file is therefore necessarily scalar. Contrast: `simd_avx512.rs`'s `BF16x16`/`BF16x8` (`avx512.rs:3255-3320`) wrap the **real hardware** `__m256bh`/`__m128bh` intrinsic types with `unsafe fn from_u16_slice`/`to_f32x16`/`to_f32x8` (`vcvtneebf162ps`, one-instruction hardware conversion, gated `#[target_feature(enable = "avx512bf16")]`) — a genuinely NATIVE realization this nightly/portable-simd backend has no way to reach (method names differ too much for a strict 1:1 MISSING claim). No arithmetic (`Add`/`Mul`/etc.) is implemented on BF16x16/BF16x8 in EITHER backend.

#### f16_types.rs — F16x16 (WHOLE-TYPE SCALAR EMULATION)

Same situation as `bf16_types.rs`: "`core::simd` has no native `f16` lane type, so this is a full scalar emulation." Storage is `[u16;16]`; every conversion is a hand-rolled IEEE-754 round-to-nearest-even bit-twiddling routine (explicitly noted as copied from `src/hpc/quantized.rs`'s scalar `F16` type — a re-use of an EXISTING scalar implementation, not a fresh core::simd-adjacent design).

| type | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| F16x16 | f32_to_f16_bits (private helper) | none | **SCALAR-FALLBACK** — full IEEE-754 round-to-nearest-even bit manipulation, pure scalar integer arithmetic on `u32`/`u16`, no vector op anywhere | f16_types.rs:29-93 |
| F16x16 | f16_bits_to_f32 (private helper) | none | **SCALAR-FALLBACK** — subnormal-normalizing `while` loop + bit reconstruction, pure scalar | f16_types.rs:99-131 |
| F16x16 | splat | `[Self::f32_to_f16_bits(v); 16]` (array-literal broadcast of a scalar helper) | **SCALAR-FALLBACK** | f16_types.rs:138 |
| F16x16 | from_slice/from_array/to_array/copy_to_slice | plain array copy | **SCALAR-FALLBACK** — plain array, no vector type | f16_types.rs:143-167 |
| F16x16 | to_f32_array | `for i in 0..16 { .. f16_bits_to_f32(..) }` | **SCALAR-FALLBACK** — explicit per-lane loop | f16_types.rs:173-179 |
| F16x16 | from_f32_array | `for i in 0..16 { .. f32_to_f16_bits(..) }` | **SCALAR-FALLBACK** — explicit per-lane loop | f16_types.rs:183-189 |

**core::simd GAP:** identical to BF16 above — no `Simd<f16,N>` (or any half-precision lane type) exists in `core::simd`; ALL 6 methods on this type are scalar by necessity, not by omission. The language-level nightly `f16` primitive type (`#![feature(f16)]`) exists separately in Rust but `core::simd` has not grown a lane type around it — this file stores raw `u16` bit patterns instead.

#### w1a_types.rs — I8x16, U16x8, U8x8, palette_lookup_u8x8, prefetch_read_t0/1/2, batch_packed_i4_16

| type | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| I8x16 | splat/from_slice/from_array/to_array/copy_to_slice | `core_i8x16::*` | NATIVE | w1a_types.rs:52-78 |
| I8x16 | from_i4_packed_u64 | `u64x16::splat`/`>>`/`&` (mask nibbles) + `Simd::cast::<i8>()` (narrow) + `<<`/`>>` on `core_i8x16` (arithmetic-shift sign extension) | **COMPOSED** — 5+ chained core::simd ops realize a nibble-unpack-and-sign-extend that has no single `core::simd` primitive | w1a_types.rs:101-105 |
| I8x16 | lane_i8\<const N\> | `Simd`'s `Index<usize>` (`self.0[N]`) | NATIVE — `core::simd::Simd` implements `core::ops::Index` | w1a_types.rs:119 |
| I8x16 | saturating_abs | `SimdInt::saturating_abs` | NATIVE | w1a_types.rs:134 |
| I8x16 | simd_min/simd_max | `SimdOrd::simd_min`/`simd_max` | NATIVE | w1a_types.rs:143,152 |
| I8x16 | cmpeq_mask/cmpgt_mask | `SimdPartialEq`/`SimdPartialOrd` + `to_bitmask()` | COMPOSED | w1a_types.rs:173,182 |
| I8x16 | PartialEq::eq / Debug::fmt | array compare / formatting | SCALAR-FALLBACK (convenience, expected) | w1a_types.rs:188,193 |
| U16x8 | splat/from_slice/from_array/to_array | `core_u16x8::*` | NATIVE | w1a_types.rs:222-241 |
| U16x8 | gather_u16 | none — scalar `for k in 0..8 { out[k] = table.get(idx[k]).copied().unwrap_or(0) }` loop over a runtime-length `&[u16]` slice | **SCALAR-FALLBACK** — same gap as `F32x16::gather` above: portable-simd has no gather-from-arbitrary-length-slice-by-index primitive | w1a_types.rs:261-272 |
| U16x8 | lane | `Simd`'s `Index<usize>` | NATIVE | w1a_types.rs:277 |
| U16x8 | simd_min/simd_max | `SimdOrd::simd_min`/`simd_max` | NATIVE | w1a_types.rs:283,289 |
| U16x8 | reduce_sum | `SimdUint::reduce_sum` | NATIVE | w1a_types.rs:306 |
| U16x8 | PartialEq::eq / Debug::fmt | array compare / formatting | SCALAR-FALLBACK (expected) | w1a_types.rs:311,317 |
| U8x8 | splat/from_array/to_array | `core_u8x8::*` | NATIVE | w1a_types.rs:335-347 |
| U8x8 | reduce_sum | `SimdUint::reduce_sum` | NATIVE | w1a_types.rs:362 |
| U8x8 | PartialEq::eq / Debug::fmt | array compare / formatting | SCALAR-FALLBACK (expected) | w1a_types.rs:367,373 |
| — | palette_lookup_u8x8 (free fn) | none — same scalar `.get(idx).unwrap_or(0)` loop pattern as `gather_u16` | **SCALAR-FALLBACK** — same gather-from-runtime-length-LUT gap | w1a_types.rs:393-403 |
| — | prefetch_read_t0/t1/t2 (free fns) | none — empty function bodies | **SCALAR-FALLBACK (deliberate no-op)** — "`core::simd` carries no prefetch". No software-prefetch hint intrinsic exists anywhere in `core::simd` (a `core::arch`-only concept, e.g. `_mm_prefetch`) | w1a_types.rs:423,428,433 |
| — | batch_packed_i4_16 (free fn, generic) | delegates per-element to `I8x16::from_i4_packed_u64` (COMPOSED, above) inside a scalar `for i in 0..n` driver loop over the OUTER batch dimension | SCALAR-FALLBACK at the OUTER (batch) level by design; inner per-element unpack is COMPOSED core::simd | w1a_types.rs:463-466 |

No `core::arch`/`std::arch` anywhere in this file (`use` block at `w1a_types.rs:20-23` is exclusively `core::fmt` + `core::simd::*`) — the prefetch no-ops are the file's own acknowledgment that reaching real prefetch would require `core::arch`, deliberately declined to keep the backend portable.

#### exotic_methods.rs — U8x64/U8x32 permute_bytes, shuffle_bytes, mask_blend, unpack_lo_epi8, unpack_hi_epi8

Module doc: "Scalar fallbacks for U8x32/U8x64 methods `core::simd` doesn't natively support (cross-lane permute, within-lane shuffle, bitmask blend, lane interleave...)." Every method is `to_array()` → scalar index loop → `from_array()`, with NO `core::simd` op invoked anywhere in the body.

| type | method | core::simd counterpart | class | file:line |
|---|---|---|---|---|
| U8x64 | permute_bytes | none | **SCALAR-FALLBACK** — "`core::simd::Swizzle::swizzle` requires a `const N: usize` index and cannot take a RUNTIME `idx` vector" — a cross-lane VPERMB-shaped gather-by-runtime-vector, which `core::simd` structurally cannot express (its `Swizzle` trait is compile-time-constant-indices only) | exotic_methods.rs:35-43 |
| U8x64 | shuffle_bytes | none | **SCALAR-FALLBACK** — PSHUFB/`_mm512_shuffle_epi8`-shaped within-128-bit-lane runtime shuffle with a high-bit-zero rule; same `Swizzle`-is-compile-time-only gap, confined to 16-byte sub-lanes | exotic_methods.rs:66-83 |
| U8x64 | mask_blend | none | **SCALAR-FALLBACK** — `_mm512_mask_blend_epi8`-shaped select-by-raw-`u64`-bitmask; `core::simd::Mask::select` (used natively elsewhere, e.g. `masks.rs`) takes a `core::simd::Mask<T,N>`, not a raw integer bitmask, so this bypasses that native primitive entirely — a genuine gap: no `core::simd` API converts a raw `uN` bitmask directly into a blend without first materializing a `Mask<T,N>` via `from_bitmask` | exotic_methods.rs:103-111 |
| U8x64 | unpack_lo_epi8/unpack_hi_epi8 | none | **SCALAR-FALLBACK** — `_mm512_unpacklo_epi8`/`unpackhi_epi8`-shaped within-128-bit-lane interleave; unlike `U32x8`'s equivalent interleave methods, which DO use `simd_swizzle!`, this file does NOT reach for `simd_swizzle!` even though the shuffle pattern here is ALSO compile-time-constant — a missed-native-op inconsistency within this same repo | exotic_methods.rs:133-146, 168-180 |
| U8x32 | permute_bytes/shuffle_bytes/mask_blend/unpack_lo_epi8/unpack_hi_epi8 | none | **SCALAR-FALLBACK** — identical shapes and gap reasoning as the U8x64 versions above, at 32-lane width | exotic_methods.rs:208-352 |

No `core::arch`/`std::arch` anywhere (`use` block at `exotic_methods.rs:7` is only `super::u8_types::{U8x32, U8x64}`) — confirmed portable, purely scalar.

</details>

## Q. Staleness corrections to sections A–L

> Read-only audit of the main matrix (679 lines, sections A–M) against
> current `src/` (`simd.rs` now 1750 lines, up from the ~370 lines implied
> by the doc's own line-range citations). Section A's per-cell tables (14
> CPU columns × ~30 rows) were spot-checked, not re-derived cell-by-cell.
> Sections B, C, D(part), E, F, H(part), I were read in full. Only **STALE**
> rows are listed (CURRENT and CANNOT-VERIFY findings are in the audit's
> own working notes, not repeated here), ordered most-important-first.

| section | doc says | code says (file:line) | action |
|---|---|---|---|
| §I.5 | Cargo feature `runtime-dispatch` (LazyLock-once table) — ✗ missing (Phase 3) | `Cargo.toml:406`: `runtime-dispatch = ["std"]` is a real feature, gating `src/simd_runtime/` (5-file, ~62 KB module: `mod.rs`, `add_mul.rs`, `casts.rs`, `cpu_ops.rs`, `matmul.rs`, `vnni_dot.rs`), re-exported `simd.rs:730,738` as `{gemm_u8_i8, matmul_bf16_to_f32, matmul_f32, vnni_dot_u8_i8, matmul_i8_to_i32}`. Closes the Phase 3 T3.3/T3.4 "gemm_dispatch"/"blas1_dispatch" framing in spirit, though not confirmed to be the exact `SimdProfile`-keyed table T3.1–T3.7 describe (`SimdProfile` itself is still absent) | edit — a reader planning Phase 3 work would duplicate an existing module |
| §I.8 | GitHub CI matrix (default v3, nightly-simd, avx512, aarch64) — ✅ partial — verified per CI doc | `.github/workflows/simd-matrix.yaml` (282 lines) is a **6-job matrix**: `native` (x86_64 AVX2, `.cargo/config-v3.toml`, incl. ternlog-body freshness + masking parity + codegen-witness + AMX realization-report/encoding tests, `:91-106`), `host-native` (informational, `continue-on-error: true`, deliberately unpinned — the job's own comments record a REAL measured heterogeneity across runs, `:120-133`), `native-v4` (AVX-512 via `.cargo/config-v4.toml`, gated on `/proc/cpuinfo`, `:163-208`), `neon` (aarch64 under qemu-user-static, 3 sub-checks, `:210-235`), `wasm` (wasm32 with AND without `+simd128`, `:237-260`), `nightly` (`core::simd` on nightly rustc + lib tests, `:262-281`). No row is named "default v3" anymore (see the v3/v4/native-config row below) | edit — "✅ partial" undersells this by an order of magnitude: AVX2/AVX-512/NEON/wasm128/wasm-scalar/nightly each get a dedicated job, several with parity+codegen-witness+asm-rung layers |
| §J (TD-T1, TD-T3, TD-T6, TD-T7) | all four listed "open" (effort 1h/1.5h/2h/2h), Phase 1 total "~15–18h, closes all 7 CRITICAL findings" | TD-T1: `amx_matmul.rs:571–605` (`bf16_gemm_dispatch`) routes the AMX tier through `bf16_tile_gemm_16x16` directly (line 597) — DONE. TD-T3: `amx_matmul.rs:951–1010` (`matmul_i8_to_i32`) is a full 4-tier dispatch (AMX → AVX-512 VNNI zmm → AVX-VNNI ymm → scalar) — DONE. TD-T6: `backend/native.rs:549–610` (`mod avx2`) — `scal_f32`/`nrm2_f32`/`asum_f32` (+f64 twins) each call a dedicated `..._avx2` unsafe fn under `#[cfg(target_arch="x86_64")]` — DONE. TD-T7: `backend/native.rs:288–305` — every non-scalar tier computes each GEMV row via the tier-dispatched `dot_f32` — DONE. Only TD-T2 (unverified this pass) and TD-T4 (confirmed still open, `hpc/quantized.rs:444–481` plain nested-loop scalar) remain from the 6 re-checked | edit — most of Phase 1's own "~15–18h" total is already shipped; re-scope the plan to TD-T2/TD-T4 only |
| §C | `gemm_u8_i8` AMX preempt — "SPR/GNR 🟦 tdpbusd (planned)" | `simd_int_ops.rs:267–287` has a live **Tier 0 runtime AMX check** ahead of the VNNI arms (`crate::hpc::amx_matmul::amx_available()` + 16/16/64-aligned shapes → `crate::hpc::int8_tile_gemm::int8_gemm_amx_tiled`, `src/hpc/int8_tile_gemm.rs:358`, a real `TDPBUSD` tile kernel) | edit — on SPR/GNR with aligned shapes the dispatch is now AMX, not the VPDPBUSD-zmm arm the doc's SPR/GNR cell implies is the only path. VNNI tiers 1–2 and the scalar fallback rows stay accurate (`simd_int_ops.rs:312–335`) |
| §D | `cast_f16_to_f32_batch`/`cast_f32_to_f16_batch` — "🟡 F16x16::to_f32x16" / "✗ scalar per-element 🚨 (F16C wired-able)" | `simd_half.rs:359–381`/`:398–419`: both runtime-check `is_x86_feature_detected!("f16c")`+`"avx"`; on success call `cast_f16_to_f32_batch_f16c`/`cast_f32_to_f16_batch_f16c` (`_mm256_cvtph_ps`/`_mm256_cvtps_ph::<8>`, 8 lanes at a time, MXCSR save/restore), falling to scalar bit-fiddle only when F16C is absent (x86_64-only, `#[cfg(target_arch="x86_64")]`) | edit — closes Phase-1 MX-T3 and materially advances Phase-4 MX-F2; the underlying `F16x16` TYPE method (`to_f32x16`) itself is still scalar-only and stays CURRENT as a narrower claim |
| §D | `BF16x16::add/sub/mul/fma` — per-CPU cells implying "⏳ `vdpbf16ps`-able (kernel exists, not dispatched)" | Two DISTINCT `BF16x16` types exist: the **portable** `simd_half::BF16x16` (`[u16;16]`, re-exported only `not(avx512bf16)`, `simd.rs:678-683`) HAS `add`/`sub`/`mul`/`fma` (`simd_half.rs:55-112`, f32-upcast scalar loops); the **native** `simd_avx512::BF16x16` (`__m256bh`, re-exported when `avx512bf16` IS a feature, `simd.rs:316-317`) has **only** `from_u16_slice`/`to_f32x16` (`simd_avx512.rs:3258-3283`) — **no `add`, no `sub`, no `mul`, no `fma` at all** | edit (framing, not just cells) — `BF16x16::add(...)` on a CPL/SPR/Z4/Z5 build is a **compile error**, not a slow scalar path; there is no un-dispatched kernel sitting next to a working method — the arithmetic path simply isn't there on that type |
| §I.1 | `.cargo/config.toml` default `x86-64-v3` — ✅ (CI baseline) | `.cargo/config.toml:2` + this repo's own `CLAUDE.md` ("HPC Rust Transformation" §Build config): the default flipped to `target-cpu=native` on 2026-09-16. `x86-64-v3` now lives in the dedicated `.cargo/config-v3.toml` (header: "Until 2026-09-16 it was `.cargo/config.toml`'s default … the default is now `target-cpu=native`") | edit — the doc describes a config that no longer exists in that form |
| Phase 6 | wasm32 SIMD128 backend — "`core::simd` via nightly-simd covers it; no per-target intrinsic wiring planned" | `src/simd_wasm.rs` (2379 lines) IS exactly per-target intrinsic wiring — a hand-written `v128`-backed backend (`v128_load`/`v128_store`/`f32x4_relaxed_madd` etc., NOT `core::simd`), selected `simd.rs:420-423` under `target_feature="simd128"`, entirely independent of `nightly-simd` (works on **stable** wasm32), with its own CI job (`simd-matrix.yaml:237-260`) and cargo config (`.cargo/config-wasm.toml`) | edit — it shipped as a first-class stable-Rust realization with more engineering investment (2379 LOC, native `v128` types for `F32x16`/`F64x8`/`I8x16`/`I32x16`/`U32x16`/`U64x8`, a balanced-tree `reduce_sum`) than several of the doc's own "landed" x86 rows |
| §H | `TD-T7` — `simd_ops::gemv_f32` "currently TD-T7 scalar" | `backend/native.rs:288–305`: `match tier() { Tier::Scalar => scalar::gemv_f32(...), _ => { per-row dot_f32(...) } }` — every non-scalar tier is tier-dispatched via `dot_f32` | edit — the literal "`simd_ops::gemv_f32` … missing" claim stays technically true (wrong module — see §H's facade-scoping nuance), but the underlying capability gap TD-T7 cites is closed |
| §A/§G | §A's per-type tables carry 14 CPU columns and no wasm128/nightly column; §G lists 13 re-export groups as the full `crate::simd::*` inventory | wasm128 and `nightly-simd` are both wired and shipped (`simd.rs:234-241` nightly re-export; `simd.rs:420-432` wasm32+simd128 re-export from `src/simd_wasm.rs`); §G is missing ~14 more re-export groups live in `crate::simd::*` today: AMX report/features/availability (`simd_amx::{amx_report,cpu_model,CpuModel}` `:757`; `hpc::amx_ops::{amx_features,AmxFeatures}` `:762`; `simd_amx::amx_tile_available` `:764`), `simd_runtime::{gemm_u8_i8,matmul_bf16_to_f32,matmul_f32,vnni_dot_u8_i8}` (`:730`) + its non-x86_64 alias (`:738`), `hpc::blas_level2::Uplo` (`:692`), `hpc::blas_level3::{BlasLevel3,Side}` (`:693`), `hpc::cascade` (whole module, `:706`), `hpc::bf16_tile_gemm::{bf16_tile_gemm_16x16_amx,bf16_tile_gemm_16x16_packed,bf16_tile_gemm_tier,PackedBf16B}` (`:750-752`), `ternlog` (9 consts, `:588-612`), the entire `simd_masking_ops` re-export block (~90 names, `:777-861` — see §N of this document), `bitwise::popcount_batch_u64` (`:866`), `simd_ops::{bf16_tile_gemm_16x16,gemm_f64_tiled,gemm_f64_tiled_fma}` (`:871`, `:645`) | edit — 16 realizations shipped, 14 documented (§A); ~14 re-export groups live, 13 documented (§G) |
| §I.4 | table names 6 cargo configs (`config.toml`, `config-avx512.toml`, `config-native.toml`, `config-apple-m2.toml`, `config-pi5.toml`, `config-graviton.toml`) | `ls .cargo/` lists **9** files — `.cargo/config-v3.toml` (new portable-baseline pin, supersedes the old implicit default), `.cargo/config-v4.toml` (plain AVX-512 **compile** contract, `x86-64-v4`, `-Dwarnings`, documented as never-run-on-non-AVX512-hardware), `.cargo/config-wasm.toml` (enables `+simd128` for `wasm32-unknown-unknown`/`wasm32-wasip1`, gating `simd_wasm`) are all undocumented | edit — 3 of 9 cargo configs entirely missing from the table |
| §A | provenance citation: cfg re-export blocks "live at `src/simd.rs` lines 221–366 / 197–366" | re-export blocks now span `simd.rs:210–447` (file grew from the ~370-line implied citation to 1750 total); content shape largely unchanged (`simd.rs:243–447`) | edit (citation only) — a future session following "lines 197–366" lands mid-table, not at the top |
| §B | gap statement (doc's list of what's NOT yet a `simd_ops::*` export) | file has grown two more polyfill-routed exports not in the doc's table: `gemm_f64_tiled` (`simd_ops.rs:952`) and `gemm_f64_tiled_fma` (`simd_ops.rs:1003`), both re-exported `simd.rs:645`; also `bf16_tile_gemm_16x16` (`simd_ops.rs:587`, re-exported `simd.rs:871`) lives in this file but is BF16/AMX-shaped, conceptually §D/§G territory — the doc's §B table doesn't mention it either | edit (incomplete, not wrong) |
| §F | "Gap: none at this layer" for `simd_soa::MultiLaneColumn` | type has grown three more typed iterators (+ `len_*` companions) since the doc's table: `iter_u32x16` (`simd_soa.rs:314`), `iter_i32x16` (`:325`), `iter_i64x8` (`:335`), plus `len_u32x16`/`len_i32x16`/`len_i64x8` (`:219,224,229`) | edit — not a functional gap (they inherit polyfill dispatch the same as the others, so the *behavior* claim still holds), but a coverage gap in the table itself per §K.4's own stated rule ("when adding a new public symbol … this table must grow a row") |

### NEW SURFACE NOT IN THE DOC

Every `pub use`/`pub mod` in `src/simd.rs` not named by any section of the main matrix (masking and crypto sections deliberately left as name-only lists — see this document's §N and §O).

| Symbol(s) | `simd.rs` line | One-line description |
|---|---|---|
| `pub mod ternlog { AND3, AND2_ANDNOT, AND_ANDNOT2, OR2_AND, XOR3, MAJ3, AND2, OR3, XOR_AND, AND2_OR }` | 588–612 | 9 named VPTERNLOG truth-table immediates (`i32` consts), backend-agnostic — lives on the facade so it resolves on every dispatch arm, not just the AVX-512 backend that first hosted it |
| `simd_ops::{gemm_f64_tiled, gemm_f64_tiled_fma}` | 645 | Crate-native tiled f64 GEMM with a bit-exactness contract (unfused / fused-FMA variants); backs `backend::native::gemm_f64`; not in doc §B or §G |
| `hpc::blas_level2::Uplo` | 692 | Upper/lower-triangular selector enum for BLAS-2 symmetric ops |
| `hpc::blas_level3::{BlasLevel3, Side}` | 693 | Real BLAS-3 GEMM/SYRK/SYMM/TRMM/TRSM trait + left/right operand-side selector, surfaced "so a consumer reaches the native-SIMD-dispatching matmul through the canonical `ndarray::simd::*` import" |
| `hpc::cascade` (whole module) | 706 | "The Belichtungsmesser" — banded multi-resolution cascade search (`Cascade::expose`, `PackedDatabase`, `adaptive_resolution`); re-exported as a module alias, not an item list |
| `hpc::amx_matmul::{amx_available, matmul_i8_to_i32}` | 720 (`std` + x86_64) | Dispatched i8×i8→i32 AMX/VNNI/scalar matmul + a runtime AMX-availability probe; this IS the function audited in §J TD-T3 above |
| `simd_runtime::{gemm_u8_i8, matmul_bf16_to_f32, matmul_f32, vnni_dot_u8_i8}` | 730 (feature `runtime-dispatch`) | Thin `#[inline(always)]` runtime-dispatch trampolines over the same tier ladders, for consumers who want the polyfill import path instead of reaching into `hpc::*` directly |
| `simd_runtime::matmul_i8_to_i32` | 738 (feature `runtime-dispatch`, non-x86_64) | Arch-uniform alias of the above, for platforms where the x86_64-only `hpc::amx_matmul` re-export doesn't apply |
| `hpc::bf16_tile_gemm::{bf16_tile_gemm_16x16_amx, bf16_tile_gemm_16x16_packed, bf16_tile_gemm_tier, PackedBf16B}` | 750–752 (`std` + x86_64) | Tile-dispatching BF16 GEMM sibling of the pure-polyfill `bf16_tile_gemm_16x16` (§B); `PackedBf16B` hoists the VNNI B-operand pack out of hot loops; `bf16_tile_gemm_tier()` reports which tier will run |
| `simd_amx::{amx_report, cpu_model, CpuModel}` | 757 (x86_64) | Cached CPU-generation detection (SPR/EMR/GNR/Sierra Forest) + a combined "silicon present vs OS-enabled" report |
| `hpc::amx_ops::{amx_features, AmxFeatures}` | 762 (x86_64) | The tier-agnostic AMX tile-gate + per-tier silicon-bit struct |
| `simd_amx::amx_tile_available` | 764 (x86_64) | Boolean gate: precondition check before issuing any AMX tile op |
| `simd_masking_ops::{…}` — ~90 names (comparison-to-mask family, `mask_and/or/xor/not/andnot` + `_assign` variants, `mask_ternlog*`, `masked_group_*`, `masked_*_i32`, `ternary_match_*`, `blend_i32`, `KeyRunCarry`, `MortonDir`, `SYM_EMPTY_I64`) | 777–861 (feature `std`) | Packed-bitmask predicate + mask-algebra + masked-reduction surface (the "columnar-selection lane" for DuckDB-shaped query pushdown) — now tabled in full in §N of this document |
| `bitwise::popcount_batch_u64` | 866 | Batch popcount closing the loop on the mask family above — "a mask producer and its reducer share one import path"; see §N.12 |
| `simd_ops::bf16_tile_gemm_16x16` | 871 (`std`) | The pure-polyfill (non-AMX-tile-dispatching) BF16 tile GEMM kernel itself (source at `simd_ops.rs:587`) — distinct from the `_amx`-suffixed dispatching wrapper above |

## R. AMX cells — verification discipline

Any cell in this matrix (or its companion sections above) that claims an AMX
tile-instruction realization (`TDPBUSD`/`TDPBF16PS`/`TILELOADD`/`TILESTORED`/
`LDTILECFG`-class) must state, explicitly, which of these three it is:

- **encoded-only** — the mnemonic/byte sequence assembles and the encoding is
  pinned (e.g. against an ELF-symtab disassembly or a known-good byte table),
  but no tile op has been run against real silicon in this pass.
- **executed-on-bare-metal** — run to completion on real AMX hardware, idle
  (no concurrent load on the host), with a `correct=`/parity assertion
  checked bit-for-bit against a non-AMX reference.
- **executed-under-contention** — run on real AMX hardware WHILE the host is
  under deliberate CPU contention (busy-loop competitors), because idle
  correctness does not establish correctness under load.

This is not a stylistic preference; each tier corresponds to a distinct,
previously-measured failure mode in `/home/user/ndarray/.claude/AMX_GOTCHAS.md`:

- **Gotcha 9 — "tests pass" can mean "tests skipped."** *"Every AMX test
  guards with `if !amx_available() { return; }`. While detection was broken
  (Gotcha 4), 100% of them early-returned green without running a single
  tile instruction. A skipped test is not a passing test. Validate AMX with
  `examples/amx_probe` (unconditional) on real AMX silicon, and require a
  `correct=`/parity assertion, not just 'didn't crash.'"* — a green AMX row
  in this matrix is worthless without confirming the guard actually let the
  test run.
- **Gotcha 14 — on oversubscribed VMs, tile state is silently corrupted
  under host CPU contention.** Measured on a 4-vCPU EMR-class VM: *"idle:
  413/413 stations bit-exact (10M and 100M rows); 4 busy-loop competitors:
  89/413, 152/413 exact — whole rows LOST, no fault; probe pinned to core 0,
  load pinned to cores 1-3: 124/413 exact — pinning does NOT mitigate; idle
  control right after: 413/413 exact again."* Signature: *"no crash, no
  SIGSEGV/SIGILL — results are silently wrong, and only under load. An
  AVX-512 path in the same process, same run, stays bit-exact, isolating the
  corruption to TMM tile state."* An "executed-on-bare-metal" claim alone
  therefore does NOT establish correctness on a shared/virtualized host —
  only an "executed-under-contention" claim does, and the two must not be
  conflated.
- **Gotcha 15 — the operand "mirror" was a misread of the byte table; use
  mnemonics.** *"`src/hpc/amx_ops.rs` (2026-09-14) assembles every AMX
  mnemonic on stable 1.98.1 with `const` tile operands... Aliased tile
  operands are now a compile error (`const` assert), so the SIGILL of
  Gotcha 11 cannot be written THROUGH `amx_ops` (the `.byte` path in
  `amx_matmul` is untouched and still lets a caller alias tiles). Encoding
  tests read each wrapper's bytes out of the test binary's ELF symtab and
  pin at least one op of every tier that has a register-only or masked
  encoding."* This is the shape of a legitimate "encoded-only" claim: an
  ELF-symtab byte-pin is real evidence of correct ENCODING, but it is not,
  by itself, evidence of correct EXECUTION — the two claims must be kept
  separate in any cell that cites it, and a cell relying only on this kind
  of evidence must be marked **encoded-only**, not ✅.

## S. Consumer audits — Belichtungsmesser and Fisher-z (2026-09-23)

Read-only audits of whether two consumer hot paths could use `ndarray::simd`.
Verified against code where stated; the full working notes were scratch files.

**Belichtungsmesser (`src/hpc/cascade.rs`).** No raw intrinsics — every
distance goes through `bitwise::hamming_distance_raw`, one candidate at a time.

| mechanism | verdict |
|---|---|
| popcount stacking (strokes) | per-candidate `hamming_distance_raw`; `hamming_batch_raw` is itself a per-row loop — no cross-candidate lane parallelism exists yet |
| early exit | **done (#323):** `hamming_distance_within` wired into every exact integer comparison (`test`, small-vector path, `query` Stroke 2, `PackedDatabase` Stroke 3); Stroke 1 f64 estimates untouched |
| confidence-interval thresholds | only `Cascade::calibrate`'s mean/variance over `&[u32]` is a loop; `F64x8` `mul_add`/`reduce_sum` fit after a u32→f64 widening |
| preheating (128-candidate warm-up) | same loop shape as the strokes |
| rolling floor | not a SIMD candidate (O(1) scalar / sequential recurrence) |
| bucket assignment (`Cascade::expose`) | a 4-threshold chain; `*_u64_to_mask` fit only if restructured into a batched second pass over finalists |

A second implementation lives in lance-graph `crates/holograph`: no ndarray
dependency, raw `std::arch` AVX-512/AVX2 Hamming (`src/hamming.rs:527-567`),
and a per-word early exit (`compute_with_threshold`, `:75`) that
`hamming_distance_within` now covers per 256-byte block.

**Fisher-z (lance-graph).** `helix::residue::ResidueEdge` (3 B, "helix 24")
vs `Signed360` (6 B = `ResidueEdge` + polar + azimuth, "signed360 48") store
quantized indices; `atanh` runs at encode time only, and the runtime distance
is a table read. `helix::simd::{batch_fisher_z, batch_l1_u8}` already use
`ndarray::simd` but have no callers outside their own tests.
`bgz-tensor::fisher_z::cosine_f32` (`fisher_z.rs:204`) is scalar while six
files of the same crate call `heel_f64x8::cosine_f32_to_f64_simd`; not a
drop-in (zero-denominator cut-off 1e-15 vs 1e-12, different summation order)
and it runs at table-build time. `lance-graph-contract` and `deepnsm-v2` are
zero-dependency by design, so their Fisher-z code cannot reach `ndarray::simd`.
Missing in ndarray: an f64 `ln`/`atanh` (only `F32x16` `simd_ln_f32` exists).
