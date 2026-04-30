# Direct-to-master audit — burn-parity post-sprint (2026-04-30)

5 commits pushed directly to master during live session. This file
documents the rationale for each — the audit trail that was skipped
when pushing directly.

## Commits

| SHA | Title | LOC |
|---|---|---|
| `ccf5b77b` | fix(deps): surgical hpc-extras gate | +24/-19 |
| `dfa25a62` | fix(backend): missing cfg gate + CBLAS aliases | +40/-1 |
| `2cd3d8b1` | feat(backend): unified INT8/BF16 GEMM dispatch | +75 |
| `00b6ee57` | feat(backend): re-export all slice-level ops | +44 |
| `c1c7ae42` | feat(simd): elementwise slice ops (simd_ops.rs) | +294 |

## ccf5b77b — surgical hpc-extras gate

PR #116 (sprint A1) gated ALL of `pub mod hpc;` behind `hpc-extras`.
This hid BF16, F16, quantization, fingerprints, VSA, plane, seal —
everything burn-ndarray and lance-graph need daily.

Fix: `pub mod hpc;` now `#[cfg(feature = "std")]` (always available).
Only 5 research modules gated: p64_bridge, crystal_encoder, deepnsm,
spo_bundle, compression_curves. blake3 made unconditional.

## dfa25a62 — CBLAS-compat aliases

`pub use mkl::{ gemm_f32, ... }` was missing its `#[cfg(feature = "intel-mkl")]`
gate — broken without the feature. Fixed + added `cblas_sgemm` / `cblas_dgemm`
as MKL drop-in replacements routing through native SIMD.

## 2cd3d8b1 — unified GEMM dispatch

INT8 GEMM existed in 3 places, BF16 in 2, with no unified entry point.
Added `backend::gemm_i8()` (VNNI → scalar) and `backend::gemm_bf16()`.
Plus CBLAS aliases `cblas_gemm_s8s8s32` / `cblas_gemm_bf16bf16f32`.

## 00b6ee57 — unified slice-op re-exports

Scattered across kernels_avx512 (pub(crate)), simd_int_ops, simd_half,
hpc/reductions. Now all reachable from `ndarray::backend::*`.

## c1c7ae42 — simd_ops.rs

Portable elementwise slice ops using operator traits on polyfill types.
`ndarray::simd::{add_f32, mul_f32, scale_f32, ...}`.
Works on all platforms. 11 tests. 1778 total pass.
