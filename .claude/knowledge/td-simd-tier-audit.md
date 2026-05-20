# SIMD Tier Technical Debt Audit

> **Design principle:** `crate::simd::*` exposes the maximum hardware performance available on the current silicon via runtime-detected polyfill. Every CPU trick is applied at its tier. Consumers must never get scalar code when the hardware offers a SIMD path. The polyfill **is** the dispatch layer, not the fallback.

## Audit scope (2026-05-20)

Every finding below was verified by reading the file at the cited line range. Files read end-to-end or in dispatch-relevant sections:

| File | LoC read | Why |
|---|---|---|
| `src/simd.rs` | 1–720 (full) | Top-level dispatch |
| `src/simd_amx.rs` | 1–421 (full) | AMX detection + VNNI dispatch |
| `src/hpc/amx_matmul.rs` | 1–671 (full) | Public ndarray-typed matmul API |
| `src/hpc/bf16_tile_gemm.rs` | 1–205 (full) | AMX tile kernel |
| `src/hpc/simd_caps.rs` | 1–514 (full) | Capability singleton |
| `src/hpc/simd_dispatch.rs` | 1–361 (full) | Frozen dispatch table |
| `src/backend/native.rs` | 1–763 (full) | Backend BLAS-1 + GEMM dispatch |
| `src/backend/kernels_avx512.rs` | 1–100 + grep | AVX-512 BLAS-1 kernels |
| `src/simd_neon_bf16.rs:130–204` | stub section | BF16 NEON stubs |
| `src/simd_neon_dotprod.rs:96–157` | stub section | F16 NEON stub |
| `src/simd_avx512.rs:680–720, 2360–2420` | VBMI + BF16 conv | VBMI permute, BF16 batch |
| `src/hpc/bgz17_bridge.rs:35–135` | dispatch sites | bgz17 L1 kernels |
| `src/hpc/nibble.rs:1–270` | dispatch sites | Nibble ops |
| `src/hpc/quantized.rs:444–630` | GEMM kernels | bf16/int8 GEMM |
| `src/hpc/vnni_gemm.rs:1–130` | VNNI INT8 GEMM | VNNI dispatch |

Files NOT yet read for this audit (next sweep):

- `src/simd_avx512.rs` remainder (~3700 LoC unread)
- `src/simd_avx2.rs` (2805 LoC unread)
- `src/simd_neon.rs` (1917 LoC unread)
- `src/simd_scalar.rs` (1308 LoC unread)
- `src/simd_half.rs` (762 LoC unread)
- `src/simd_nightly/*`
- HPC modules: `vml.rs`, `activations.rs`, `reductions.rs`, `kernels.rs`, `fft.rs`, `statistics.rs`, `lapack.rs`, `blas_level{1,2,3}.rs`, `cam_pq.rs`, `palette_distance.rs`, `aabb.rs`, `distance.rs`, `bitwise.rs`, `p64_bridge.rs`, `spatial_hash.rs`, `jitson_cranelift/detect.rs`, all of `src/hpc/styles/*`

## Microscopic silicon tier matrix

| CPU | AVX-512F | VNNI | VBMI | BF16 | FP16 | AMX-INT8 | AMX-BF16 | AVX-VNNI-INT8 |
|---|---|---|---|---|---|---|---|---|
| Skylake-X / SP / W (2017) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Cascade Lake (2019) | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Cooper Lake (2020) | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ |
| Ice Lake-SP / Tiger Lake (2021) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Sapphire Rapids (2023) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Granite Rapids (2024) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | (+ AMX-FP16, AMX-COMPLEX) |
| Zen 4 (Genoa, Ryzen 7000, 2022) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Zen 5 (2024) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Arrow Lake / Lunar Lake (2024) | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Pi 5 / Orange Pi 5 (A76, ARMv8.2) | (NEON) | (dotprod) | – | (bf16+) | (fp16) | – | – | – |
| Pi 4 (A72, ARMv8.0) | (NEON) | – | – | – | – | – | – | – |

---

## Findings — CRITICAL

### TD-T1 · `src/hpc/amx_matmul.rs:319-327` · stub fast path

`matmul_bf16_to_f32`: `if amx_available() { bf16_gemm_f32(...) } else { bf16_gemm_f32(...) }` — both arms identical. Comment 320-323 admits "Future: AMX-tiled fast path. Today we route through the same f32 reference kernel; correctness is identical regardless of hardware. The `amx_available()` branch is preserved so callers can be sure the AMX detection runs."

Working AMX kernel exists at `src/hpc/bf16_tile_gemm.rs::bf16_tile_gemm_16x16` (lines 39-87) — full TDPBF16PS dispatch, tested at lines 151-204.

**Hit on Sapphire Rapids / Granite Rapids:** scalar instead of 256-mul-add/instr tile op.

### TD-T2 · `src/hpc/amx_matmul.rs:351-356` · stub fast path

`matmul_f32` AMX branch: converts f32 → BF16 and calls `bf16_gemm_f32` (scalar). Same shape as TD-T1.

### TD-T3 · `src/hpc/amx_matmul.rs:395-412` · stub fast path + wrong fallback

`matmul_i8_to_i32` AMX branch: shifts LHS i8 → u8 (+128) and calls `int8_gemm_i32` (scalar reference). 

Two debts here:
1. AMX path never reaches `tile_dpbusd` (the working primitive at `amx_matmul.rs:146-150`).
2. The fallback when AMX is absent should be `int8_gemm_vnni` (at `src/hpc/vnni_gemm.rs:46`), which dispatches AVX-512 VNNI `VPDPBUSD` (64 MACs/instr) — but it calls the scalar `int8_gemm_i32` directly.

**Hit on Sapphire Rapids:** ~256× slower than AMX TDPBUSD.  
**Hit on Cascade Lake / Ice Lake-SP / Zen 4 (AVX-512 + VNNI but no AMX):** ~64× slower than VNNI.

### TD-T4 · `src/hpc/quantized.rs:444-481` · scalar kernel labeled GEMM

`bf16_gemm_f32` is a triple-nested scalar loop with per-element `.to_f32()` upcast. No `crate::simd::*` types, no F32x16 mul_add, no FMA. This is the function `matmul_bf16_to_f32` falls back to — so the entire BF16 GEMM public surface bottoms out in scalar.

**Hit on every CPU:** even AVX-512F-only Skylake-X loses the F32x16 mul_add (16-wide FMA per instr) that would lift this 16×.

### TD-T5 · `src/hpc/quantized.rs:618-630` · scalar kernel labeled GEMM

`int8_gemm_i32` is a triple-nested scalar loop. The VNNI dispatch path `int8_gemm_vnni` (lines 46-61 of `vnni_gemm.rs`) exists and is correct (uses `simd_caps().has_avx512_vnni()` and calls `int8_gemm_vnni_avx512`), but it's a separate symbol — nothing routes the public `int8_gemm_i32` callers through it.

### TD-T6 · `src/backend/native.rs:544-561` · scalar-only "avx2" implementations

The `avx2` module's `scal_f32`, `scal_f64`, `nrm2_f32`, `nrm2_f64`, `asum_f32`, `asum_f64` all unconditionally delegate to `super::scalar::*`. The dispatch macro thinks it's dispatching to AVX2 but the body is scalar.

```rust
pub fn scal_f32(alpha: f32, x: &mut [f32]) {
    super::scalar::scal_f32(alpha, x);  // ← line 545
}
```

Effect: on Haswell–Coffee Lake / Zen 1-3 (AVX2 + FMA but no AVX-512), all of `scal_*`, `nrm2_*`, `asum_*` run scalar. The dispatch macro at lines 92-165 routes through `avx2::name()` which is itself scalar.

### TD-T7 · `src/backend/native.rs:271-278` · GEMV scalar everywhere

`gemv_f32` and `gemv_f64` skip the `dispatch!` macro entirely and call `scalar::gemv_*` unconditionally. No AVX-512, no AVX2, no NEON. Every consumer of the backend GEMV path runs the scalar nested loop on every CPU.

```rust
pub fn gemv_f32(...) {
    scalar::gemv_f32(...);  // ← line 272
}
```

---

## Findings — HIGH

### TD-T8 · `src/hpc/simd_dispatch.rs:150-163` · aarch64 dispatch = scalar

```rust
#[cfg(target_arch = "aarch64")]
fn detect() -> Self {
    let caps = simd_caps();
    let tier = if caps.asimd_dotprod { SimdTier::NeonDotProd } else { SimdTier::Neon };
    // NEON uses the same scalar wrapper signatures — NEON intrinsics
    // will be wired when simd_neon.rs types are activated. For now,
    // dispatch to scalar which auto-vectorizes well on aarch64 with
    // `-C target-feature=+neon` (mandatory on aarch64).
    Self { tier, ..Self::scalar() }
}
```

The frozen dispatch table reports `NeonDotProd` or `Neon` tier to consumers but every function pointer in the struct is the scalar wrapper. Pi 5 / Pi 4 / M2 get the scalar implementations for `byte_find_all`, `byte_count`, `squared_distances_f32`, `nibble_unpack`, `nibble_above_threshold`, `batch_sq_dist`.

### TD-T9 · `src/hpc/simd_dispatch.rs:128-134` · AVX-512 dispatch falls to AVX2 wrappers

Even when `caps.avx512bw` is true, the AVX-512 tier branch fills in 4 of 6 function pointers with AVX2 wrappers:

```rust
if caps.avx512bw {
    Self {
        tier: SimdTier::Avx512,
        byte_find_all: byte_find_all_avx512_wrapper,           // ← real
        byte_count: byte_count_avx512_wrapper,                  // ← real
        squared_distances_f32: squared_distances_avx2_wrapper, // ← AVX2!
        nibble_unpack: nibble_unpack_avx2_wrapper,             // ← AVX2!
        nibble_above_threshold: nibble_above_threshold_avx2_wrapper, // ← AVX2!
        batch_sq_dist: batch_sq_dist_avx2_wrapper,             // ← AVX2!
    }
}
```

Comment at line 130 admits `// no avx512 variant for 3D dist`. For `nibble_*`, the variant is missing per TD-T17.

### TD-T10 · `src/simd_neon_bf16.rs:149-177` · stub structs that panic

`BF16x8Stub` (line 149) and `BF16x16Stub` (line 156) are placeholder structs whose only method is `unimplemented()` panicking with the message documenting the BFMMLA / BFDOT asm-byte encoding still to wire up: `BFMMLA = 0x6e40_ec00 | (Vm << 16) | (Vn << 5) | Vd`, `BFDOT = 0x4e40_ec00 | (Vm << 16) | (Vn << 5) | Vd`. Module docs at lines 187-204 spell out the implementation plan; nothing is wired.

**Hit on Pi 5 A76, Apple M2+, Snapdragon 8 Gen 2+:** consumers reaching for BF16 NEON ops panic or fall through to scalar `simd_half::BF16x16`.

### TD-T11 · `src/simd_neon_dotprod.rs:115-148` · F16x16 stub

`F16x16Stub` (line 136) is a placeholder; `unimplemented()` panics (line 141-147). Module docs at lines 96-113 give the full intrinsic map (`vfmaq_f16`, `vaddvq_f16`, `vsqrtq_f16`, `vcgtq_f16`) and the stable-Rust asm-byte encoding `0x0e40_cc20` for `fmla v0.8h, v1.8h, v2.8h`.

**Hit on Pi 5 A76, Apple M-series:** consumers reaching `crate::simd::F16x16` get `simd_avx2::F16Scaler` scalar polyfill (line 134 comment) or `simd_nightly::F16x16`.

### TD-T12 · `src/simd.rs:18-26` + `:49-88` · top-level Tier enum collapses

```rust
enum Tier {
    Avx512 = 1,
    Avx2 = 2,
    NeonDotProd = 3,
    Neon = 4,
    Scalar = 5,
}

fn detect_tier() -> Tier {
    if is_x86_feature_detected!("avx512f") { return Tier::Avx512; }
    if is_x86_feature_detected!("avx2") { return Tier::Avx2; }
    ...
}
```

Skylake-X (no VNNI / VBMI / BF16 / FP16 / AMX) and Granite Rapids (all of them) both → `Tier::Avx512`. Arrow Lake (`avxvnniint8`, no AVX-512F) → `Tier::Avx2`. Every caller of `tier()` (line 97) gets a coarse answer.

Mitigation: `simd_caps()` at `src/hpc/simd_caps.rs:98` exists with 20 per-feature bits — but it's a separate dispatch channel, and consumers who use `tier()` don't see the sub-features.

### TD-T13 · `src/backend/native.rs:22-26` · second Tier enum, same collapse

Backend defines its own `Tier { Avx512, Avx2, Scalar }` enum (line 21-26), independent of the one in `simd.rs:18`. Same 3-bucket collapse. Same lack of VNNI / VBMI / BF16 / FP16 / AMX awareness.

### TD-T14 · `src/hpc/simd_dispatch.rs:30-49` · third Tier enum, same collapse

`SimdTier { Avx512, Avx2, Sse2, NeonDotProd, Neon, Scalar, WasmSimd128 }` — 7 variants, but `detect()` at lines 121-148 only branches on `caps.avx512bw` and `caps.avx2`. SSE2 never selected. No AVX-512-VNNI / VBMI / BF16 / FP16 / AMX paths.

Three independent Tier enums total (TD-T12, TD-T13, TD-T14).

### TD-T15 · `src/simd.rs:291-292 + 531-532` · BF16x16 polyfill-not-max under default config

```rust
// 291: hardware-native, ONLY if compile-time avx512bf16 is on
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bf16", not(feature = "nightly-simd")))]
pub use crate::simd_avx512::{BF16x16, BF16x8};

// 531: scalar polyfill, the default
#[cfg(all(feature = "std", not(all(target_arch = "x86_64", target_feature = "avx512bf16"))))]
pub use crate::simd_half::BF16x16;
```

The cargo default is `x86-64-v3` (per `.cargo/config.toml:25`), which is AVX2 only — no AVX-512F, definitely no avx512bf16. So even on Sapphire Rapids / Zen 4 silicon under default cargo, `crate::simd::BF16x16` resolves to scalar `simd_half::BF16x16`.

Compile-time gate where runtime dispatch would lift the entire AVX-512 + BF16 install base out of the scalar polyfill.

---

## Findings — MEDIUM

### TD-T16 · `src/hpc/nibble.rs:23-41, 227-237` · nibble ops cap at AVX2

`nibble_unpack` (line 23) and `nibble_above_threshold` (line 227) check `caps.avx2` only — no AVX-512 path. Sapphire Rapids / Ice Lake / Zen 4 process 32 nibbles per AVX2 iteration when 64 per AVX-512BW iteration would be possible.

### TD-T17 · `src/hpc/nibble.rs:59-94, 169-189, 257-278` · "AVX2" funcs are scalar loops

`nibble_unpack_avx2` (line 59), `nibble_sub_clamp_avx2` (line 170), `nibble_above_threshold_avx2` (line 258) all carry `#[target_feature(enable = "avx2")]` but their bodies are plain scalar loops:

```rust
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn nibble_unpack_avx2(packed: &[u8], count: usize, out: &mut Vec<u8>) {
    // ...
    for j in 0..16 {
        lo[j] = data[j] & 0x0F;     // ← scalar loop
        hi[j] = (data[j] >> 4) & 0x0F;
    }
    // ...
}
```

The autovectorizer may emit reasonable code, but this is not true `_mm256_*` intrinsics. `nibble_sub_clamp_avx512` at line 197 IS real (uses `U8x64::saturating_sub`). So nibble has one real SIMD path and two pretend-SIMD paths.

### TD-T18 · `src/simd.rs:479-486` · simd_ln_f32 is a scalar loop

```rust
pub fn simd_ln_f32(x: F32x16) -> F32x16 {
    let arr = x.to_array();
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = arr[i].ln();  // ← scalar per-lane
    }
    F32x16::from_array(out)
}
```

`simd_exp_f32` at lines 419-450 is a real Remez polynomial with FMA via `mul_add` chain. `simd_ln_f32` is its asymmetric scalar twin. A consumer thinking they're getting 16-wide log gets 16× scalar `ln`.

### TD-T19 · `src/hpc/distance.rs:101` · single tier, no AVX-512

The 3D `squared_distances` function checks `caps.avx2` only — line 101: `if super::simd_caps::simd_caps().avx2`. No AVX-512F variant. Sapphire Rapids etc. fall to AVX2 8-wide instead of AVX-512 16-wide.

### TD-T20 · `src/hpc/spatial_hash.rs:273` · same as TD-T19

`batch_sq_dist` checks `caps.avx2` only. No AVX-512F variant.

### TD-T21 · `src/simd.rs:351-354` · aarch64 integers come from scalar

```rust
#[cfg(all(target_arch = "aarch64", not(feature = "nightly-simd")))]
pub use scalar::{
    f32x8, f64x4, i32x16, i32x8, i64x4, i64x8, u16x16, u32x16, u32x8, u64x4, u64x8, u8x64,
    F32x8, F64x4, I32x16, I32x8, I64x4, I64x8, U16x16, U16x32, U32x16, U32x8, U64x4, U64x8, U8x64,
};
```

On aarch64, the only types from `simd_neon::aarch64_simd` (line 349) are `f32x16, f64x8, F32Mask16, F32x16, F64Mask8, F64x8`. Every integer width — `I32x16`, `I8x32`, `U8x64`, `U16x32`, etc. — comes from `scalar::*`. Pi 5 / M2 get scalar integer SIMD even though NEON has `int32x4_t`, `uint8x16_t`, etc.

### TD-T22 · `src/simd.rs:310, 318-321` · 256-bit int types in AVX2 build come from `simd_avx512`

```rust
// 310: AVX2-baseline arm uses simd_avx512 for the 256-bit shapes
pub use crate::simd_avx512::{f32x8, f64x4, i16x16, i8x32, F32x8, F64x4, I16x16, I8x32};
```

Inverted naming: `I32x8` / `U32x8` / `I64x4` / `U64x4` (the natural AVX2 widths) come from `simd_avx2.rs` (which polyfills them as scalar storage with `[u32; 8]` arrays per the comment in the AMX matmul work), not from native `__m256i`. The polyfill IS the AVX2 module on AVX2 builds — verify whether the AVX2 module's polyfills wrap real `_mm256_*` intrinsics or scalar arrays. (Audit pending — requires reading `src/simd_avx2.rs`.)

---

## Verified — code is correct (rejected agent claims)

### `src/simd_amx.rs:282-301` · AVX-VNNI-INT8 dispatch IS done

`matvec_dispatch` correctly routes `is_x86_feature_detected!("avxvnniint8")` to `vnni2_matvec` (256-bit VPDPBUSD path) when avx512vnni is absent. No debt.

### `src/simd_avx512.rs:695-710` · VBMI dispatch IS done

`permute_bytes` checks `if simd_caps().avx512vbmi { permute_bytes_vbmi(...) } else { scalar }`. Native `_mm512_permutexvar_epi8` reaches Ice Lake / SPR / Zen 4. The scalar branch is correct fallback for Skylake-X / Cascade Lake / Cooper Lake. No debt.

### `src/hpc/bgz17_bridge.rs:43-86` · multi-versioning is correct

5 dispatch sites at lines 78, 142, 197, 250, 349 each route `avx512f → avx2 → scalar` with proper `#[target_feature]` annotations on the inner functions. The `is_x86_feature_detected!("avx512f")` vs `avx2` granularity is appropriate for L1 absolute-difference kernels — VNNI / VBMI / BF16 don't help on `abs(a-b)` reductions. No tier-collapse debt here.

### `src/hpc/vnni_gemm.rs:46-61` · VNNI dispatch is correct

`int8_gemm_vnni` checks `simd_caps().has_avx512_vnni()` and calls `int8_gemm_vnni_avx512`. The only debt is that other paths (TD-T3, TD-T5) don't route through this function.

### `src/hpc/p64_bridge.rs:109` · VPOPCNTDQ dispatch is correct

`simd_caps().avx512vpopcntdq` runtime-detected. No debt at this site.

### `src/hpc/cam_pq.rs:202, 215` · AVX-512F dispatch is correct

`simd_caps().avx512f` runtime-detected. No debt at this site.

### `src/hpc/aabb.rs:284, 440` · AVX-512F + SSE2 dispatch present

Dispatches `avx512f` then falls to `sse2`. Missing intermediate AVX2 — `aabb` uses AVX-512F at one tier and SSE2 at the other. Probably acceptable for AABB (BV-shape ops), but a sub-finding to investigate on next sweep.

---

## Prioritized action list

| ID | Severity | Effort | Description |
|---|---|---|---|
| TD-T1 | CRIT | 1h | Wire `matmul_bf16_to_f32` to `bf16_tile_gemm_16x16` |
| TD-T2 | CRIT | 30m (after T1) | Same for `matmul_f32` |
| TD-T3 | CRIT | 1.5h | Wire `matmul_i8_to_i32` to AMX tile / VNNI fallback |
| TD-T5 | CRIT | 30m | Route `int8_gemm_i32` callers through `int8_gemm_vnni` |
| TD-T4 | CRIT | 3-4h | Rewrite `bf16_gemm_f32` with F32x16 mul_add + tiling |
| TD-T6 | CRIT | 2h | Implement `avx2::{scal,nrm2,asum}_*` with real AVX2 intrinsics |
| TD-T7 | CRIT | 2h | Implement `gemv_f32`/`gemv_f64` with tier dispatch |
| TD-T8 | HIGH | 4-6h | Wire `simd_dispatch.rs` aarch64 tier to real NEON impls |
| TD-T9 | HIGH | 2-3h | Add AVX-512 variants for `squared_distances`, `nibble_*`, `batch_sq_dist` |
| TD-T10 | HIGH | 3-4h | Implement `BF16x8/16` NEON via asm-byte BFMMLA/BFDOT |
| TD-T11 | HIGH | 3-4h | Implement `F16x16` NEON via asm-byte fmla v.8h |
| TD-T15 | HIGH | 4-6h | Convert `BF16x16` from compile-time `target_feature` gate to runtime dispatch |
| TD-T16 | MED | 1.5h | Add AVX-512BW variants for `nibble_unpack` / `nibble_above_threshold` |
| TD-T17 | MED | 2h | Replace scalar-loop "avx2" funcs in nibble with `_mm256_*` intrinsics |
| TD-T18 | MED | 2h | Rewrite `simd_ln_f32` as real Remez polynomial like `simd_exp_f32` |
| TD-T19 | MED | 1h | Add AVX-512F path to `distance::squared_distances_f32` |
| TD-T20 | MED | 1h | Same for `spatial_hash::batch_sq_dist` |
| TD-T21 | HIGH | 8-12h | Replace aarch64 scalar integer types in `simd.rs` with NEON impls |
| TD-T22 | – | – | Investigation only — needs `simd_avx2.rs` read first |
| TD-T12/T13/T14 | HIGH | (audit-wide) | Consolidate three Tier enums OR route all callers through `simd_caps()` for sub-feature dispatch |

## Next-sweep targets (unread)

These files are listed in the dispatch site grep but not yet read for this audit. Findings in them are unverified:

- Full `src/simd_avx512.rs`, `simd_avx2.rs`, `simd_neon.rs`, `simd_scalar.rs`, `simd_half.rs`
- HPC SIMD-consuming: `vml.rs`, `activations.rs`, `reductions.rs`, `kernels.rs`, `fft.rs`
- HPC suspected scalar: `statistics.rs`, `lapack.rs`, `blas_level{1,2,3}.rs`
- HPC dispatch sites with `is_x86_feature_detected!`: `cam_pq.rs`, `palette_distance.rs`, `aabb.rs`, `distance.rs`, `bitwise.rs`, `p64_bridge.rs`, `spatial_hash.rs`, `jitson_cranelift/detect.rs`
- All 34 `src/hpc/styles/*` primitives

The most likely-debt-rich unread targets:

1. `src/hpc/blas_level{1,2,3}.rs` — grep showed NO use of `crate::simd::*` types. The flagship BLAS public API may be entirely scalar (separate audit needed).
2. `src/hpc/statistics.rs`, `lapack.rs` — same, no `crate::simd::*` use.
3. `src/simd_avx2.rs` — the 256-bit polyfills for 512-bit types. TD-T22 needs this read to know whether the polyfills are real `__m256i` intrinsics or scalar arrays under `#[target_feature]`.

