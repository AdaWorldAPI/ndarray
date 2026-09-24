# Per-CPU SIMD inventory — generated from LLVM's TableGen source

> **GENERATED** by `tools/gen_llvm_inventory.py` from llvm-project `llvmorg-22.1.8`
> (the tag matching `rustc -vV` under this repo's pinned toolchain). Do not edit by hand:
> run `python3 tools/gen_llvm_inventory.py --write`. `--verify-rustc` checks every row
> against rustc's own `--print cfg -C target-cpu=...`.

`backend` mirrors the `simd.rs` cfg ladder. Primitive columns list the instruction(s)
that implement the primitive on that CPU; `—` means the CPU has none and a portable
fallback is required.

## x86 (92 CPUs)

| cpu | backend | avx2 | avx512f | avx512vl | avx512ifma | avxifma | pclmul | vpclmulqdq | aes | vaes | gfni | mul_lo32 | shift_var_u32 | permute_var_u32 | ifma52 | clmul | aes_round |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `alderlake` | simd_avx2 | ✓ |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `arrowlake` | simd_avx2 | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (VEX) | PCLMULQDQ | AESENC |
| `arrowlake-s` | simd_avx2 | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (VEX) | PCLMULQDQ | AESENC |
| `arrowlake_s` | simd_avx2 | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (VEX) | PCLMULQDQ | AESENC |
| `atom` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `atom_sse4_2` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | — |
| `atom_sse4_2_movbe` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | AESENC |
| `bdver1` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | AESENC |
| `bdver2` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | AESENC |
| `bdver3` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | AESENC |
| `bdver4` | simd_avx2 | ✓ |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `bonnell` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `broadwell` | simd_avx2 | ✓ |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | — |
| `btver2` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | AESENC |
| `cannonlake` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ | AESENC |
| `cascadelake` | simd_avx512 | ✓ | ✓ | ✓ |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `clearwaterforest` | simd_avx2 | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (VEX) | PCLMULQDQ | AESENC |
| `cooperlake` | simd_avx512 | ✓ | ✓ | ✓ |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `core-avx-i` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | — |
| `core-avx2` | simd_avx2 | ✓ |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | — |
| `core2` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `core_2_duo_sse4_1` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `core_2_duo_ssse3` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `core_2nd_gen_avx` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | — |
| `core_3rd_gen_avx` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | — |
| `core_4th_gen_avx` | simd_avx2 | ✓ |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | — |
| `core_4th_gen_avx_tsx` | simd_avx2 | ✓ |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | — |
| `core_5th_gen_avx` | simd_avx2 | ✓ |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | — |
| `core_5th_gen_avx_tsx` | simd_avx2 | ✓ |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | — |
| `core_aes_pclmulqdq` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | — |
| `core_i7_sse4_2` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `corei7` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `corei7-avx` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | — |
| `diamondrapids` | simd_avx512 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX)<br>VPMADD52LUQ/HUQ (VEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `emeraldrapids` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `generic` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | — | — | — | — | — | — |
| `goldmont` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | AESENC |
| `goldmont-plus` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | AESENC |
| `goldmont_plus` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | AESENC |
| `gracemont` | simd_avx2 | ✓ |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `grandridge` | simd_avx2 | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (VEX) | PCLMULQDQ | AESENC |
| `graniterapids` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `graniterapids-d` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `graniterapids_d` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `haswell` | simd_avx2 | ✓ |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | — |
| `icelake-client` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `icelake-server` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `icelake_client` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `icelake_server` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `ivybridge` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | — |
| `knl` | simd_avx512 | ✓ | ✓ |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `knm` | simd_avx512 | ✓ | ✓ |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `lunarlake` | simd_avx2 | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (VEX) | PCLMULQDQ | AESENC |
| `meteorlake` | simd_avx2 | ✓ |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `mic_avx512` | simd_avx512 | ✓ | ✓ |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `nehalem` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `nocona` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `novalake` | simd_avx512 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX)<br>VPMADD52LUQ/HUQ (VEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `pantherlake` | simd_avx2 | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (VEX) | PCLMULQDQ | AESENC |
| `penryn` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `pentium-m` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `pentium4` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `pentium4m` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `pentium_4` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `pentium_4_sse3` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `pentium_m` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `prescott` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `raptorlake` | simd_avx2 | ✓ |  |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `rocketlake` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `sandybridge` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | — |
| `sapphirerapids` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `sierraforest` | simd_avx2 | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (VEX) | PCLMULQDQ | AESENC |
| `silvermont` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | — |
| `skx` | simd_avx512 | ✓ | ✓ | ✓ |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `skylake` | simd_avx2 | ✓ |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `skylake-avx512` | simd_avx512 | ✓ | ✓ | ✓ |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `skylake_avx512` | simd_avx512 | ✓ | ✓ | ✓ |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `slm` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | — |
| `tigerlake` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `tremont` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  | ✓ |  | ✓ | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | AESENC |
| `westmere` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  | ✓ |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | PCLMULQDQ | — |
| `wildcatlake` | simd_avx2 | ✓ |  |  |  | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (VEX) | PCLMULQDQ | AESENC |
| `x86-64` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `x86-64-v2` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `x86-64-v3` | simd_avx2 | ✓ |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | — | — |
| `x86-64-v4` | simd_avx512 | ✓ | ✓ | ✓ |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | — | — | — |
| `yonah` | simd_avx2 (no avx2: array polyfill) |  |  |  |  |  |  |  |  |  |  | PMULUDQ (xmm, SSE2) | — | — | — | — | — |
| `znver1` | simd_avx2 | ✓ |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `znver2` | simd_avx2 | ✓ |  |  |  |  | ✓ |  | ✓ |  |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `znver3` | simd_avx2 | ✓ |  |  |  |  | ✓ | ✓ | ✓ | ✓ |  | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2) | VPSLLVD / VPSRLVD | VPERMD | — | PCLMULQDQ | AESENC |
| `znver4` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |
| `znver5` | simd_avx512 | ✓ | ✓ | ✓ | ✓ |  | ✓ | ✓ | ✓ | ✓ | ✓ | PMULUDQ (xmm, SSE2)<br>VPMULUDQ (ymm, AVX2)<br>VPMULUDQ (zmm, AVX-512F) | VPSLLVD / VPSRLVD | VPERMD | VPMADD52LUQ/HUQ (EVEX) | PCLMULQDQ<br>VPCLMULQDQ (zmm) | AESENC<br>VAESENC (zmm) |

## aarch64 (99 CPUs)

| cpu | backend | neon | dotprod | i8mm | aes | sha2 | sve2 | mul_lo32 | shift_var_u32 | permute_var_u32 | ifma52 | clmul | aes_round |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `a64fx` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `ampere1` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `ampere1a` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `ampere1b` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `ampere1c` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a10` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a11` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a12` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a13` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a14` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a15` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a16` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a17` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a18` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a19` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a7` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a8` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-a9` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-latest` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-m1` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-m2` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-m3` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-m4` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-m5` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-s10` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-s4` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-s5` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-s6` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-s7` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-s8` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `apple-s9` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `c1-nano` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `c1-premium` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `c1-pro` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `c1-ultra` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `carmel` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cobalt-100` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-a320` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-a34` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a35` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a510` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-a520` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-a520ae` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-a53` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a55` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a57` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a65` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a65ae` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a710` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-a715` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-a72` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a720` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-a720ae` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-a725` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-a73` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a75` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a76` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a76ae` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a77` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a78` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a78ae` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-a78c` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-r82` | simd_neon +dotprod | ✓ | ✓ |  |  |  |  | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-r82ae` | simd_neon +dotprod | ✓ | ✓ |  |  |  |  | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-x1` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-x1c` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `cortex-x2` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-x3` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-x4` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cortex-x925` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `cyclone` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `exynos-m3` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `exynos-m4` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `exynos-m5` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `falkor` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `fujitsu-monaka` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `gb10` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `generic` | simd_neon | ✓ |  |  |  |  |  | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `grace` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `kryo` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `neoverse-512tvb` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `neoverse-e1` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `neoverse-n1` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `neoverse-n2` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `neoverse-n3` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `neoverse-v1` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `neoverse-v2` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `neoverse-v3` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `neoverse-v3ae` | simd_neon +dotprod | ✓ | ✓ | ✓ |  |  | ✓ | UMULL / UMULL2 | USHL | TBL | — | — | — |
| `olympus` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `oryon-1` | simd_neon +dotprod | ✓ | ✓ | ✓ | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `saphira` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `thunderx` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `thunderx2t99` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `thunderx3t110` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `thunderxt81` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `thunderxt83` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `thunderxt88` | simd_neon | ✓ |  |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
| `tsv110` | simd_neon +dotprod | ✓ | ✓ |  | ✓ | ✓ |  | UMULL / UMULL2 | USHL | TBL | — | PMULL (64x64->128) | AESE + AESMC |
