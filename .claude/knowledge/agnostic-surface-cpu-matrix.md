# Agnostic SIMD Surface — Per-CPU Resolution Matrix

> **Companion to:** `td-simd-tier-audit.md` (debt inventory),
> `td-simd-integration-plan.md` (`SimdProfile` architecture),
> `td-simd-cpu-dispatch-matrix.md` (per-CPU feature table).
>
> **This document is the cross-tab.** For every public symbol on the
> agnostic surface (`crate::simd::*`, `crate::simd_int_ops::*`,
> `crate::simd_half::*`, `crate::simd_soa::*`), it lists which kernel
> the compile-time `#[cfg]` chain (or polyfilled type) resolves to on
> each CPU profile, **plus** the shape-ingress status (does the
> surface take `ArrayView<N>` or does it drop to a flat `&[T]` and
> require the caller to pass `(m, n, k)` separately).
>
> Authored 2026-05-20. Update whenever a surface is added, a kernel
> arm is wired, or a CPU profile is added.

## Goals

1. **One look-up table** for "what runs where". When debugging a perf
   surprise, find the row, find the CPU column, see what kernel
   actually ran.
2. **Gap map.** Every cell that is not `✅ live` is an integration
   target. Phase ordering in the integration plan section is derived
   directly from the gap counts per profile.
3. **Shape-debt inventory.** Every surface that takes `&[T] +
   (dims…)` is debt against the consumer-facing API: it forces
   `.as_slice().unwrap()` (panics on non-contiguous views) and loses
   the type-level shape guarantee that `ArrayView2`/`ArrayViewMut2`
   provides. Tagged per surface; addressed in the integration plan.

## Legend

Per cell in the kernel matrix:

| Marker | Meaning |
|---|---|
| ✅ `kernel-name` | Compile-time `#[cfg]` arm is live and tested in the codebase. |
| ⏳ `kernel-name` | Planned arm — kernel exists OR is straightforward to write; consumer surface or dispatch chain not yet wired. |
| 🟡 *polyfill-transparent* | Surface routes through a polyfilled type (`F32x16`, etc.); the CPU-specific intrinsic lives one layer down. The polyfilled type's row in § A is the authoritative answer. |
| ⚠️ `scalar` | Falls to scalar on this profile even though hardware support exists (debt). |
| — | Profile cannot run this op (e.g. AMX op on a non-AMX CPU). |
| n/a | Op type does not apply to this lane width / architecture. |

Per cell in the **shape-ingress** column:

| Marker | Meaning |
|---|---|
| 📐 `ArrayView` | Function accepts `ArrayView<N>` / `ArrayViewMut<N>` directly. Shape preserved through the call; strided / non-contiguous inputs handled at the boundary. |
| 🔪 `&[T] + dims` | Function takes flat slice + explicit `(m,n,k)` or implicit length. Loses shape. Caller must call `.as_slice().unwrap()` (panics on non-contiguous arrays) and re-wrap the output. **Debt.** |
| 🔪 `&[T]` | Function takes flat slice only (no dims — implicit single-axis). Acceptable for `add_f32`/`mul_f32`/etc. where the op is dimension-independent. **Tolerated**, not debt. |
| 🔪 `&mut [T] += &[T]` | In-place slice op. Single-axis, no shape information needed. Tolerated. |

## CPU profile columns

The matrix uses these abbreviated columns (see
`td-simd-cpu-dispatch-matrix.md` for the canonical feature table):

| Abbr | Codename | Key feature delta |
|---|---|---|
| **SKX** | Skylake-X / -SP / -W | AVX-512F (no VNNI, no BF16, no AMX) |
| **CLX** | Cascade Lake | + AVX-512 VNNI |
| **CPL** | Cooper Lake | + AVX-512 BF16 (no VBMI, no AMX) |
| **ICX** | Ice Lake-SP | + VBMI / VBMI2 / VPOPCNTDQ / BITALG / GFNI / VAES (no BF16) |
| **SPR** | Sapphire Rapids / Emerald Rapids | + AVX-512 BF16 + FP16 + AMX (TILE / INT8 / BF16) + AVX-VNNI |
| **GNR** | Granite Rapids | + AMX-FP16 (CPUID leaf 7.1 EAX bit 21) |
| **Zn4** | Zen 4 (Genoa, Ryzen 7000) | AVX-512 + VNNI + BF16 + FP16 (no AMX) |
| **Zn5** | Zen 5 | same ISA as Zen 4 |
| **ARL** | Arrow Lake / Lunar Lake / Meteor Lake-H | AVX-VNNI + AVX-VNNI-INT8 + AVX-IFMA + AVX-NE-CONVERT (no AVX-512) |
| **HSW** | Haswell ⇢ Coffee Lake | AVX2 + FMA only |
| **A76** | A76+ / Apple M / Snapdragon 8G1+ | NEON + dotprod (+ bf16/fp16 on ARMv8.2+) |
| **A72** | A72 / A53-with-crypto | NEON + AES (no dotprod) |
| **A53** | A53 without crypto / minimal aarch64 | NEON baseline |
| **SCA** | wasm32 / riscv / x86 baseline / unknown | Scalar |

---

## A. Polyfilled SIMD types — backing storage per CPU

Source: `src/simd.rs`, `simd_avx512.rs`, `simd_avx2.rs`, `simd_neon.rs`,
`simd_scalar.rs`, `simd_half.rs`. The polyfilled types are the **agnostic
substrate** — every consumer-facing op above ultimately lowers through them.

Cells show the **backing storage / register kind** the type compiles to on each
profile. Method-level intrinsic selection (e.g. `add` → `_mm512_add_ps` vs
`vaddq_f32`) follows the storage.

| Type | SKX | CLX | CPL | ICX | SPR | GNR | Zn4 | Zn5 | ARL | HSW | A76 | A72 | A53 | SCA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `F32x16` | `__m512` | `__m512` | `__m512` | `__m512` | `__m512` | `__m512` | `__m512` | `__m512` | 2× `__m256` | 2× `__m256` | 4× `float32x4_t` | 4× `float32x4_t` | 4× `float32x4_t` | `[f32; 16]` |
| `F32x8` | `__m256` | `__m256` | `__m256` | `__m256` | `__m256` | `__m256` | `__m256` | `__m256` | `__m256` | `__m256` | 2× `float32x4_t` | 2× `float32x4_t` | 2× `float32x4_t` | `[f32; 8]` |
| `F64x8` | `__m512d` | `__m512d` | `__m512d` | `__m512d` | `__m512d` | `__m512d` | `__m512d` | `__m512d` | 2× `__m256d` | 2× `__m256d` | 4× `float64x2_t` | 4× `float64x2_t` | 4× `float64x2_t` | `[f64; 8]` |
| `F64x4` | `__m256d` | `__m256d` | `__m256d` | `__m256d` | `__m256d` | `__m256d` | `__m256d` | `__m256d` | `__m256d` | `__m256d` | 2× `float64x2_t` | 2× `float64x2_t` | 2× `float64x2_t` | `[f64; 4]` |
| `I8x64` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | 2× `__m256i` | 2× `__m256i` | 4× `int8x16_t` (sca polyfill) | scalar | scalar | `[i8; 64]` |
| `I8x32` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | 2× `int8x16_t` (sca polyfill) | scalar | scalar | `[i8; 32]` |
| `U8x64` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | 2× `__m256i` | 2× `__m256i` | 4× `uint8x16_t` (sca polyfill) | scalar | scalar | `[u8; 64]` |
| `U8x32` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | 2× `uint8x16_t` (sca polyfill) | scalar | scalar | `[u8; 32]` |
| `I16x32` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | 2× `__m256i` | 2× `__m256i` | scalar | scalar | scalar | `[i16; 32]` |
| `I16x16` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | scalar | scalar | scalar | `[i16; 16]` |
| `I32x16` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | 2× `__m256i` | 2× `__m256i` | scalar | scalar | scalar | `[i32; 16]` |
| `I32x8` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | scalar | scalar | scalar | `[i32; 8]` |
| `I64x8` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | 2× `__m256i` | 2× `__m256i` | scalar | scalar | scalar | `[i64; 8]` |
| `I64x4` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | scalar | scalar | scalar | `[i64; 4]` |
| `U16x32` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | 2× `__m256i` | 2× `__m256i` | scalar | scalar | scalar | `[u16; 32]` |
| `U16x16` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | scalar | scalar | scalar | `[u16; 16]` |
| `U32x16` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | 2× `__m256i` | 2× `__m256i` | scalar | scalar | scalar | `[u32; 16]` |
| `U32x8` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | scalar | scalar | scalar | `[u32; 8]` |
| `U64x8` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | `__m512i` | 2× `__m256i` | 2× `__m256i` | scalar | scalar | scalar | `[u64; 8]` |
| `U64x4` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | `__m256i` | scalar | scalar | scalar | `[u64; 4]` |
| `BF16x16` | ⚠️ `[u16; 16]` scalar | ⚠️ `[u16; 16]` scalar | ✅ `__m256bh` (AVX-512BF16) | ⚠️ `[u16; 16]` scalar | ✅ `__m256bh` (AVX-512BF16) | ✅ `__m256bh` | ✅ `__m256bh` | ✅ `__m256bh` | ⚠️ `[u16; 16]` scalar | ⚠️ `[u16; 16]` scalar | ⚠️ `[u16; 16]` scalar (NEON `bfloat16x8_t` paired — see TD-T10) | ⚠️ scalar | ⚠️ scalar | `[u16; 16]` |
| `BF16x8` | n/a | n/a | ✅ `__m128bh` (avx512bf16) | n/a | ✅ `__m128bh` | ✅ `__m128bh` | ✅ `__m128bh` | ✅ `__m128bh` | n/a | n/a | n/a | n/a | n/a | n/a |
| `F16x16` | ⚠️ `[u16; 16]` scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar (would be `__m256h` via avx512fp16; see TD-T11) | ⚠️ scalar (same) | ⚠️ scalar (same) | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar (NEON `float16x8_t` paired — see TD-T11) | ⚠️ scalar | ⚠️ scalar | `[u16; 16]` |
| `F32Mask16` | `__mmask16` | `__mmask16` | `__mmask16` | `__mmask16` | `__mmask16` | `__mmask16` | `__mmask16` | `__mmask16` | 2× ymm bitmask | 2× ymm bitmask | NEON bitmask polyfill | scalar | scalar | `[bool; 16]` |
| `F32Mask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | ymm bitmask | ymm bitmask | NEON bitmask | scalar | scalar | `[bool; 8]` |
| `F64Mask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | 2× ymm bitmask | 2× ymm bitmask | NEON bitmask | scalar | scalar | `[bool; 8]` |
| `F64Mask4` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | `__mmask8` | ymm bitmask | ymm bitmask | NEON bitmask | scalar | scalar | `[bool; 4]` |

**Notes**

- `⚠️ [u16; 16] scalar` for `BF16x16` / `F16x16` is the TD-SIMD-8 honesty
  finding: storage is a plain `[u16; 16]`, every op upcasts to f32, computes
  lane-by-lane, downcasts back. The hardware instructions (`vcvtneps2pbh` /
  `_mm256_cvtph_ps` / NEON `bfcvt` / `vcvt_f16_f32`) exist on the listed
  silicon but are not wired into the polyfilled type. Wired path on `BF16x16`
  for CPL / SPR / Zen4 routes through `__m256bh` via `simd_avx512.rs`; F16x16
  is uniformly scalar (no profile yet has hardware-backed storage).
- Integer-lane scalar fallbacks on aarch64 (`I16x32`, `I32x*`, `I64x*`,
  `U16x*`, `U32x*`, `U64x*`) are TD-T21 — NEON has 128-bit `int{16,32,64}x*_t`
  quartets that would back these, but the dispatch currently selects the
  scalar polyfill from `simd_scalar.rs`. Float lanes (`F32x16`, `F64x8`) are
  wired to real NEON via `simd_neon::aarch64_simd` per the agent-A7 work.
- On HSW / ARL (no AVX-512), the 16-wide x86 types decompose into two ymm
  halves via the `simd_avx2.rs` macros. Hot operations (`add`, `mul`, FMA via
  `mul_add`) issue paired ymm instructions; this is **not** a regression vs
  AVX-512 in arithmetic throughput per cycle, only in lane count per
  instruction.

---

## B. Method-level kernel selection (F32x16 / F64x8 hot methods)

For the polyfilled-type methods where CPU divergence matters (FMA vs
mul-then-add, hardware min/max vs select, masked reductions, etc.). Less-hot
methods (`splat`, `from_array`, `to_array`, `copy_to_slice`, `simd_eq` etc.)
follow the storage row above without per-CPU specialization.

| Method | SKX..GNR / Zn4..Zn5 (AVX-512) | ARL / HSW (AVX2 + FMA) | A76..A53 (NEON) | SCA |
|---|---|---|---|---|
| `F32x16::mul_add(b, c)` | `vfmadd231ps zmm` | 2× `_mm256_fmadd_ps` | 4× `vfmaq_f32` | scalar `f32::mul_add` (FMA inst if host has FMA, else two-step) |
| `F32x16::reduce_sum()` | `vextractf64x4` + `vaddps` tree | ymm `vhaddps` cascade | `vaddvq_f32` paired | naive `iter().sum()` |
| `F32x16::reduce_min()` | `_mm512_reduce_min_ps` (helper) | ymm `vminps` tree | `vminvq_f32` paired | `iter().fold(INF, min)` |
| `F32x16::simd_min(b)` | `vminps zmm` | `vminps ymm` × 2 | `vminq_f32` × 4 | scalar `min` |
| `F32x16::simd_clamp(lo, hi)` | `vminps`+`vmaxps zmm` | ymm pair | NEON pair | scalar pair |
| `F32x16::simd_lt(b)` → mask | `vcmpltps` → `__mmask16` | `vcmpltps ymm` → bitmask | `vcltq_f32` → NEON bitmask | `[bool; 16]` |
| `mask.select(a, b)` | `vpblendmps zmm{k}` masked move | `vblendvps ymm` | `vbslq_f32` | scalar `if-else` |
| `F64x8::mul_add(b, c)` | `vfmadd231pd zmm` | 2× `_mm256_fmadd_pd` | 4× `vfmaq_f64` | scalar `f64::mul_add` |

**FMA semantics note.** `mul_add` on every backend, including scalar, lowers
to a single-rounding FMA when the CPU has FMA — on HSW+ that's `vfmadd231ps`;
on A53 that's `vfmaq_f32` (NEON FMA is mandatory in ARMv8.0). Only on a
host without FMA (pre-Haswell x86, pre-NEON aarch64 — neither in our matrix)
does `f32::mul_add` fall back to a two-step. **`add_mul_f32` is therefore
single-rounding everywhere in our supported profile set.**

---

## C. Float slice ops — `simd_ops`

All ops route through the polyfilled type (column 🟡 in CPU cells —
the per-CPU kernel selection is the row in § A / § B). The
**shape-ingress** column captures whether the function takes
`ArrayView` (good) or flat `&[T]` (debt).

| Function | Shape ingress | Polyfill path | Status |
|---|---|---|---|
| `add_f32(a, b)` → `Vec<f32>` | 🔪 `&[f32]` (tolerated, 1D op) | 🟡 `F32x16::add` | ✅ live |
| `sub_f32`, `mul_f32`, `div_f32` | 🔪 `&[f32]` | 🟡 `F32x16::{sub,mul,div}` | ✅ live |
| `add_f32_inplace`, `sub_f32_inplace`, `mul_f32_inplace`, `div_f32_inplace` | 🔪 `&mut [f32] += &[f32]` (tolerated) | 🟡 same | ✅ live |
| `scale_f32(a, scalar)` | 🔪 `&[f32]` + `f32` | 🟡 `F32x16::splat` + `mul` | ✅ live |
| `add_scalar_f32(a, scalar)` | 🔪 `&[f32]` + `f32` | 🟡 `F32x16::splat` + `add` | ✅ live |
| `scale_f32_inplace(a, scalar)` | 🔪 `&mut [f32]` + `f32` | 🟡 same | ✅ live |
| **`add_mul_f32(acc, a, b)` ✨ new** | 🔪 `&mut [f32], &[f32], &[f32]` | 🟡 `F32x16::mul_add` | ✅ live (PR `0a46e7f`) |
| `add_f64`, `mul_f64`, `add_f64_inplace` | 🔪 `&[f64]` | 🟡 `F64x8::{add,mul}` | ✅ live |
| **`add_mul_f64(acc, a, b)` ✨ new** | 🔪 `&mut [f64], &[f64], &[f64]` | 🟡 `F64x8::mul_add` | ✅ live (PR `0a46e7f`) |
| `array_chunks::<T, N>` | 🔪 `&[T]` → iter `&[T; N]` (helper) | n/a — slicing primitive | ✅ live |
| `array_chunks_checked::<T, N>` | 🔪 `&[T]` → `Result<iter, ()>` | n/a | ✅ live |
| **`array_windows::<T, N>` ✨ new** | 🔪 `&[T]` → iter `&[T; N]` overlapping | n/a — slicing primitive | ✅ live (PR `0a46e7f`) |
| **`array_windows_checked::<T, N>` ✨ new** | 🔪 `&[T]` → `Result<iter, ()>` | n/a | ✅ live (PR `0a46e7f`) |

**Surface debt.** None of these need `ArrayView` — they're slice-shaped
in name (e.g. `add_f32` on two parallel buffers), single-axis, and
shape would add no information. **Tolerated**, not debt.

---

## D. Integer slice ops — `simd_int_ops`

| Function | Shape ingress | SKX | CLX | CPL | ICX | SPR | GNR | Zn4 | Zn5 | ARL | HSW | A76 | A72 | A53 | SCA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `add_i8(dst, src)` | 🔪 `&mut [i8] += &[i8]` | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `sub_i8(dst, src)` | 🔪 `&mut [i8]` | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `add_i16(dst, src)` | 🔪 `&mut [i16]` | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `dot_i8(a, b) -> i32` | 🔪 `&[i8], &[i8]` | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `dot_i16(a, b) -> i64` | 🔪 `&[i16], &[i16]` | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `min_i8(s) -> i8` | 🔪 `&[i8]` | 🟡 `I8x64::min` | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 `I8x32::min` ×2 | 🟡 ×2 | 🟡 `I8x16::min` ×4 | ⚠️ scalar | ⚠️ scalar | scalar |
| `max_i8(s) -> i8` | 🔪 `&[i8]` | 🟡 `I8x64::max` | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 ×2 | 🟡 ×2 | 🟡 `I8x16::max` ×4 | ⚠️ scalar | ⚠️ scalar | scalar |
| **`gemm_u8_i8(a, b, c, m, n, k)` ✨ new** | 🔪 `&[u8], &[i8], &mut [i32] + (m,n,k)` ⚠️ **debt** | ⚠️ scalar | ✅ `int8_gemm_vnni_avx512` | ✅ same | ✅ same | ✅ same | ✅ same | ✅ same | ✅ same | ✅ `int8_gemm_avxvnni_ymm` | ⚠️ scalar | ⏳ `neon_sdot_int8_gemm` | ⚠️ scalar | ⚠️ scalar | scalar |

**Surface debt.** `gemm_u8_i8` flags 🔪 debt: caller must pass `(m,n,k)`
separately and the result `&mut [i32]` is interpreted as `[m, n]`
row-major by convention. Lifting to `ArrayView2<u8>, ArrayView2<i8>,
ArrayViewMut2<i32>` would: (a) carry shape through the signature, (b)
accept strided inputs without `.as_slice().unwrap()` panics, (c) match
the `hpc::amx_matmul::matmul_*` family's shape (which already does the
right thing). Targeted in Phase 1 of the integration plan below.

The integer-elementwise ops (`add_i8`, `sub_i8`, `dot_i8`, `dot_i16`,
`add_i16`) are CPU-uniform scalar today — they pre-date the integer
polyfill types being widened (I8x64, I16x32, I32x16 etc. now exist).
Lifting them to the polyfilled types would mirror what `min_i8` /
`max_i8` already do and pick up the integer SIMD throughput for free.

---

## E. Half-precision ops — `simd_half`

| Function | Shape ingress | SKX | CLX | CPL | ICX | SPR | GNR | Zn4 | Zn5 | ARL | HSW | A76 | A72 | A53 | SCA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `BF16x16::{add,sub,mul,fma}` | n/a (method) | ⚠️ scalar | ⚠️ scalar | 🟡 `__m256bh` ops via avx512bf16 | ⚠️ scalar | 🟡 `__m256bh` | 🟡 `__m256bh` | 🟡 `__m256bh` | 🟡 `__m256bh` | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar (NEON BFDOT/BFMMLA — TD-T10) | ⚠️ scalar | ⚠️ scalar | scalar |
| `F16x16::{add,sub,mul,fma}` | n/a (method) | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar (avx512fp16 — TD-SIMD-8) | ⚠️ scalar (same) | ⚠️ scalar (same) | ⚠️ scalar | ⚠️ scalar (F16C upcast — TD-SIMD-8) | ⚠️ scalar (F16C upcast) | ⚠️ scalar (NEON fp16 — TD-T11) | ⚠️ scalar | ⚠️ scalar | scalar |
| `BF16x16::to_f32x16()` | n/a | ⚠️ scalar | ⚠️ scalar | ✅ `vcvtne2ps2bf16` inverse via shift | ⚠️ scalar | ✅ same | ✅ same | ✅ same | ✅ same | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `F16x16::to_f32x16()` | n/a | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar (avx512fp16 `vcvtph2ps`) | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar (F16C `_mm256_cvtph_ps`) | ⚠️ scalar (F16C) | ⚠️ scalar (NEON `vcvt_f32_f16`) | ⚠️ scalar | ⚠️ scalar | scalar |
| `add_bf16_inplace` | 🔪 `&mut [BF16] += &[BF16]` | ⚠️ scalar | ⚠️ scalar | ✅ `BF16x16::add` (`__m256bh`) | ⚠️ scalar | ✅ same | ✅ same | ✅ same | ✅ same | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `mul_bf16_inplace` | 🔪 `&mut [BF16]` | ⚠️ scalar | ⚠️ scalar | ✅ `BF16x16::mul` | ⚠️ scalar | ✅ | ✅ | ✅ | ✅ | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `add_f16_inplace` | 🔪 `&mut [F16] += &[F16]` | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `mul_f16_inplace` | 🔪 `&mut [F16]` | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `cast_bf16_to_f32_batch` | 🔪 `&[BF16], &mut [f32]` | ⚠️ scalar | ⚠️ scalar | ✅ `BF16x16::to_f32x16` | ⚠️ scalar | ✅ | ✅ | ✅ | ✅ | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `cast_f16_to_f32_batch` | 🔪 `&[F16], &mut [f32]` | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `cast_f32_to_bf16_batch` | 🔪 `&[f32], &mut [BF16]` | ⚠️ scalar (truncate) | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |
| `cast_f32_to_f16_batch` | 🔪 `&[f32], &mut [F16]` | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | ⚠️ scalar | scalar |

**Coverage gap.** Of the 12 `simd_half` slice surfaces, only 4
(`add_bf16_inplace`, `mul_bf16_inplace`, `cast_bf16_to_f32_batch`, plus
indirect via `BF16x16` methods) light up on CPL / SPR / GNR / Zen4 /
Zen5 — the rest are uniformly scalar across every profile. F16 has
**zero** SIMD wiring. The hardware exists (avx512fp16 on SPR+/Zen4+,
F16C on every AVX2 chip, NEON `+fp16` on A76+) — none of it is
plumbed. Tracked as TD-SIMD-8 in the audit.

**Surface debt.** Same shape as § C — single-axis, tolerated.

---

## F. Batch converters + transcendentals — re-exported through `crate::simd::*`

| Function | Shape ingress | Source | SKX | CLX | CPL | ICX | SPR | GNR | Zn4 | Zn5 | ARL | HSW | A76 | A72 | A53 | SCA |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bf16_to_f32_batch` | 🔪 `&[u16], &mut [f32]` | `simd_avx512.rs` | ✅ AVX-512F shift-extract | ✅ same | ✅ avx512bf16 native | ✅ AVX-512F | ✅ avx512bf16 | ✅ same | ✅ avx512bf16 | ✅ same | ✅ AVX2 `_mm256_cvtepu16_epi32` + shift | ✅ same | ✅ NEON `vshlq_n` | ✅ same | ✅ same | scalar |
| `f32_to_bf16_batch` | 🔪 `&[f32], &mut [u16]` | `simd_avx512.rs` | ✅ AVX-512F truncate | ✅ same | ✅ avx512bf16 `vcvtne2ps2bf16` | ✅ AVX-512F truncate | ✅ avx512bf16 | ✅ same | ✅ avx512bf16 | ✅ same | ✅ AVX2 truncate (no rne) | ✅ same | ✅ NEON `vshrn` truncate | ✅ same | ✅ same | scalar truncate |
| `f32_to_bf16_batch_rne` | 🔪 `&[f32], &mut [u16]` | `simd_avx512.rs` | ✅ AVX-512F RNE polyfill | ✅ same | ✅ avx512bf16 (hardware RNE) | ✅ AVX-512F | ✅ avx512bf16 hw RNE | ✅ same | ✅ avx512bf16 hw RNE | ✅ same | ⚠️ AVX2 RNE polyfill | ⚠️ same | ⏳ NEON RNE polyfill | ⏳ same | ⏳ same | scalar RNE |
| `simd_exp_f32(F32x16)` | n/a (typed) | `simd.rs` | 🟡 polyfill Remez | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | 🟡 | scalar Remez |
| `simd_ln_f32(F32x16)` | n/a (typed) | `simd.rs` | ⚠️ scalar `f32::ln` (TD-T18) | ⚠️ same | ⚠️ same | ⚠️ same | ⚠️ same | ⚠️ same | ⚠️ same | ⚠️ same | ⚠️ same | ⚠️ same | ⚠️ same | ⚠️ same | ⚠️ same | scalar |

**Coverage gap.** `simd_ln_f32` is openly admitted scalar (TD-T18 in
the audit). The `_rne` round-to-nearest-even path is hardware-direct
only on AVX-512-BF16 silicon; everywhere else it's a polyfill that
matches semantics but not throughput.

---

## G. SoA carriers — `simd_soa`

`MultiLaneColumn` is layout-only — every method (`iter_u8x64`,
`iter_f32x16`, `iter_f64x8`, `iter_u64x8`) is uniform across CPUs by
construction: it yields polyfilled-type instances and the per-CPU
intrinsic selection lives one layer down (§ A).

| Function | Shape ingress | All CPU profiles | Status |
|---|---|---|---|
| `MultiLaneColumn::new(Arc<[u8]>)` | 🔪 `Arc<[u8]>` (single-axis byte view) | uniform — 64-byte alignment check + `Arc` clone | ✅ live |
| `iter_u8x64()` | n/a | 🟡 `U8x64::from_array` per chunk (§ A) | ✅ live |
| `iter_f32x16()` | n/a | 🟡 LE-decode → `F32x16::from_array` | ✅ live |
| `iter_f64x8()` | n/a | 🟡 LE-decode → `F64x8::from_array` | ✅ live |
| `iter_u64x8()` | n/a | 🟡 LE-decode → `U64x8::from_array` | ✅ live |

**Surface debt.** None. SoA is fundamentally a byte-shaped surface —
the polyfilled type yields are where shape would re-enter, and that's
the consumer's responsibility.

---

## H. HPC surfaces with shape-preserving entry — reference shape

These are **already correct** — they take `ArrayView` / `ArrayViewMut`
and the agnostic int8 / bf16 GEMM surfaces should match this shape.

| Function | Signature | Source |
|---|---|---|
| `hpc::amx_matmul::matmul_bf16_to_f32` | `(ArrayView2<BF16>, ArrayView2<BF16>, ArrayViewMut2<f32>) -> Result<(), MatmulError>` | 📐 ArrayView |
| `hpc::amx_matmul::matmul_f32` | `(ArrayView2<f32>, ArrayView2<f32>, ArrayViewMut2<f32>) -> Result<()>` | 📐 ArrayView |
| `hpc::amx_matmul::matmul_i8_to_i32` | `(ArrayView2<i8>, ArrayView2<i8>, ArrayViewMut2<i32>) -> Result<()>` | 📐 ArrayView |
| `ndarray::linalg::general_mat_mul` | `(α, &ArrayBase, &ArrayBase, β, &mut ArrayBase)` | 📐 ArrayView (BLAS-3 GEMM) |
| `ndarray::linalg::general_mat_vec_mul` | `(α, &ArrayBase, &ArrayBase, β, &mut ArrayBase)` | 📐 ArrayView (BLAS-2 GEMV) |

The shape-debt fix for `simd_int_ops::gemm_u8_i8` is to mirror
`hpc::amx_matmul::matmul_i8_to_i32`'s signature exactly: `(ArrayView2<u8>,
ArrayView2<i8>, ArrayViewMut2<i32>) -> Result<(), MatmulError>`. Internally
the function still drops to flat slices to feed the VNNI kernel, but the
boundary preserves shape and the caller no longer needs `.as_slice().unwrap()`.

---

## I. Cross-cutting gap counts

Aggregating the matrix:

| CPU profile | Cells live (✅) | Cells planned (⏳) | Cells scalar-debt (⚠️) | Coverage % (of cells that have a hardware path) |
|---|---|---|---|---|
| **SKX** | 6 ops × 0 SIMD-int → ~30 ⚠️ | 0 ⏳ | many | ~40 % |
| **CLX** | + VNNI int8 GEMM | — | many | ~50 % |
| **CPL** | + BF16 lanes (BF16x16 ops, batch converts) | — | F16 + int-i8/i16 ops still scalar | ~65 % |
| **ICX** | + VBMI / VPOPCNTDQ (currently unused by surfaces) | VBMI byte-permute consumers | F16 scalar; int-i8/i16 scalar | ~50 % |
| **SPR** | + AMX (currently unused) + BF16 hw RNE + FP16 (unused) | AMX-INT8 tile arm for `gemm_u8_i8`; AVX-512 FP16 for F16x16 | F16 scalar, int-i8/i16 scalar | ~60 % |
| **GNR** | + AMX-FP16 (caps detection missing) | AMX-FP16 (CPUID 7.1 EAX:21 detection) | same as SPR | ~60 % |
| **Zn4 / Zn5** | + BF16 lanes + AVX-VNNI | — | no AMX (architectural), F16 scalar | ~65 % |
| **ARL** | + AVX-VNNI ymm for `gemm_u8_i8` | F16 via F16C; AVX-VNNI-INT8 for s×s gemm (separate surface) | F16 scalar, int-i8/i16 scalar | ~30 % |
| **HSW** | F32x16/F64x8 paired-ymm baseline | AVX2 int8/i16 SIMD via `I8x32` / `I16x16` | i8/i16 scalar surfaces; no INT8 dot product | ~20 % |
| **A76** | F32x16/F64x8 NEON-paired (verified) | NEON SDOT for `gemm_u8_i8`; NEON BFDOT/BFMMLA for BF16x16 ops (TD-T10); NEON fp16 for F16x16 (TD-T11); aarch64 int polyfills (TD-T21) | most ops scalar | ~15 % |
| **A72** | F32x16/F64x8 NEON-paired | aarch64 int polyfills (TD-T21) for I8x32/I16x16/etc. | most ops scalar; no SDOT | ~10 % |
| **A53** | F32x16/F64x8 NEON-paired | same as A72 | same | ~10 % |
| **SCA** | uniform scalar (correct floor) | — | n/a | 100 % (by definition) |

**Surface debt count** (functions taking `&[T] + (dims…)` that
*should* take `ArrayView`):

| Surface | Current | Target | Debt |
|---|---|---|---|
| `simd_int_ops::gemm_u8_i8` | 🔪 `&[u8], &[i8], &mut [i32], m, n, k` | 📐 `ArrayView2, ArrayView2, ArrayViewMut2` | **yes** |
| Future `simd_int_ops::gemm_bf16_to_f32` | (planned) | 📐 `ArrayView2<BF16>, ArrayView2<BF16>, ArrayViewMut2<f32>` | design-in |
| Future `simd_int_ops::gemv_f32` | (planned) | 📐 `(α, ArrayView2<f32>, ArrayView1<f32>, β, ArrayViewMut1<f32>)` | design-in |
| Future `simd_int_ops::syrk_f32` | (planned) | 📐 same shape | design-in |

Slice-shaped 1D ops (`add_f32`, `dot_i8`, `add_bf16_inplace`, etc.) are
**not** debt — single-axis, slice signature is correct.

---

## J. Integration plan

Phasing derived directly from the gap counts and shape-debt list above.
Each phase is one PR-sized landing; landings are additive (no caller-side
changes between phases).

### Phase 0 — Shape-debt fix for `gemm_u8_i8` (1 PR, ~1h)

Promote the agnostic surface to `ArrayView2 / ArrayViewMut2` (the
shape-preserving signature). Internal dispatch chain unchanged (cfg
arms still pick AVX-512 VNNI / AVX-VNNI / scalar); the entry-point
boundary now mirrors `hpc::amx_matmul::matmul_i8_to_i32` exactly.

* Rename old `gemm_u8_i8(&[u8], &[i8], &mut [i32], m, n, k)` →
  `gemm_u8_i8_slices(...)` and keep as a `pub(crate)` internal that the
  ArrayView surface lowers into.
* New public `gemm_u8_i8(ArrayView2<u8>, ArrayView2<i8>, ArrayViewMut2<i32>)
  -> Result<(), MatmulError>` — checks shapes, packs to contiguous if
  strided, calls the slice form.
* Update the ignored timing harness to use the ArrayView signature.

### Phase 1 — Wire the existing-but-unwired hardware paths (3 PRs, ~6h total)

Each PR adds one `#[cfg]` arm to the `gemm_u8_i8` dispatch chain. The
kernel exists; only routing needs to land.

* **1a · NEON SDOT arm** — `int8_gemm_neon_sdot` kernel using
  `vdotq_s32` on A76 / A78 / Apple M. Calls `vdotq_s32` over 4-wide i32
  accumulators with the +128 bias trick on the u8 LHS (since SDOT is
  s×s). Cfg gate: `target_arch = "aarch64", target_feature = "dotprod"`.
* **1b · AMX-INT8 arm** — wires the existing `tile_dpbusd` primitive
  from `simd_amx.rs` and `bf16_tile_gemm.rs` into a `int8_gemm_amx_tile`
  kernel. Cfg gate: `target_arch = "x86_64", target_feature = "amx-int8"`.
  Lights up on SPR / GNR builds with `--config .cargo/config-avx512.toml`
  (which now sets `-Ctarget-cpu=sapphirerapids`).
* **1c · AVX-VNNI-INT8 symmetric arm** — new surface
  `gemm_i8_i8(ArrayView2<i8>, ArrayView2<i8>, ArrayViewMut2<i32>)` for
  the signed×signed case, using `VPDPBSSD` on ARL / GNR. Separate
  function (different element-type signature than `gemm_u8_i8`).

### Phase 2 — Lift integer-elementwise surfaces to the polyfilled types (2 PRs, ~4h)

The integer-elementwise ops (`add_i8`, `sub_i8`, `add_i16`, `dot_i8`,
`dot_i16`) are uniformly scalar today — predate `I8x64` / `I8x32` /
`I16x32` / `I16x16` becoming polyfilled. Lift each to use the typed
SIMD lanes; `min_i8` / `max_i8` already do this and are the template.

* **2a · `add_i8` / `sub_i8` / `add_i16` via `I8x64` / `I16x32` chunks.**
* **2b · `dot_i8` / `dot_i16` via VPDPBUSD / VPDPWSSD on x86 (VNNI gate)
  + NEON `vdotq_s32` on aarch64.** Falls back to widening + horizontal
  add on AVX-512F-only / pre-dotprod NEON.

### Phase 3 — TD-SIMD-8: BF16x16 / F16x16 hardware-backing (3 PRs, ~12h)

Currently the entire half-precision surface is scalar polyfill on every
CPU. The intrinsics exist on most profiles.

* **3a · BF16x16 native on CPL / SPR / GNR / Zn4 / Zn5.** Already
  partially wired through `simd_avx512` re-exports when
  `target_feature = "avx512bf16"`; verify and extend ops (`add`, `sub`,
  `mul`, `fma`, `to_f32x16`) to use the `__m256bh` ops.
* **3b · F16x16 native on SPR / GNR / Zn4 / Zn5 (avx512fp16) and on
  ARL / HSW / Zen 1+ (F16C upcast).** Separate paths because
  avx512fp16 is true native 16-wide; F16C is upcast-to-f32-do-op-downcast
  via `_mm256_cvtph_ps` / `_mm256_cvtps_ph`.
* **3c · NEON BF16 + FP16 on A76+ (TD-T10 + TD-T11).** Real
  `bfloat16x8_t` / `float16x8_t` backing storage; ops via BFDOT /
  BFMMLA / `vfmaq_f16` / `vcvt_f32_f16`.

### Phase 4 — Wire the remaining hardware (4 PRs, parallelizable)

* **4a · aarch64 integer polyfill — TD-T21.** Replace scalar fallbacks
  for `I8x32`, `I16x16`, `I32x8`, `U16x16`, `U32x8`, `U64x4` etc. with
  real 128-bit NEON `intNx_t` quartets on aarch64. Currently `simd.rs`
  re-exports these from `scalar::*` on aarch64.
* **4b · `simd_ln_f32` Remez polynomial (TD-T18).** Mirror
  `simd_exp_f32`'s structure — currently `simd_ln_f32` is a scalar loop
  inside an `F32x16`-shaped wrapper.
* **4c · `cast_f32_to_bf16_batch_rne` for non-AVX-512BF16 silicon.**
  Currently the RNE path uses an AVX-512F polyfill (verified
  byte-exact). Extend to AVX2 (via `__m256` rounding) and NEON (via
  `vshrn_n_u32`).
* **4d · AMX-FP16 detection.** Add CPUID leaf 7.1 EAX:21 to
  `simd_caps.rs::detect()` so GNR's AMX-FP16 lights up. Currently
  `caps.amx_fp16` doesn't exist as a field, blocking the FP16 AMX arm
  even though SPR-class CPUID detection is in place.

### Phase 5 — Sweep remaining shape-debt on new surfaces (rolling)

Every new agnostic surface lands with `ArrayView` ingress from day one
(BLAS-2 GEMV, GER, SYRK, TRSM; BLAS-3 SYMM, TRMM; LAPACK; etc.). Phase
0's `gemm_u8_i8` lift is the template.

---

## Verification checklist before marking a cell ✅

When promoting a cell from ⏳/⚠️ to ✅:

1. The kernel exists at `src/...` and has a `#[target_feature(enable = ...)]`
   annotation that matches the cfg-gate selecting it.
2. The agnostic surface's cfg chain routes to that kernel under the
   profile's `target_feature` set.
3. Correctness verified: parity-test against the scalar arm produces
   byte-equal results on at least three input shapes (small / mid / tail).
4. Performance verified: timing-harness shows the kernel beats the
   scalar reference at representative sizes. (The sanity check from
   PR `0134916` showed `gemm_u8_i8` AVX-VNNI ymm = 1.77×–5.88× over
   scalar, AVX-512 VNNI zmm = 3.11×–8.04×.)
5. Doc-comment on the surface function's build matrix updated (the
   table inside the rustdoc of `gemm_u8_i8` is the pattern).
6. This document's cell flipped from ⏳/⚠️ to ✅ in the same PR.
