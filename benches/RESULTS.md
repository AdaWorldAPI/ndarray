# splat3d bench results

Per-kernel timing baseline for the `splat3d` feature. Regression > 5% on
any row blocks merge per the sprint discipline. Update this file in the
same commit as any change to a `splat3d` kernel.

## Run

```bash
cargo bench --features splat3d --bench splat3d_bench
```

Hardware notes: record the CPU model + topology + relevant target
features (`avx512f`, `avx512bw`, `neon`, `dotprod`) for each row so the
comparison is meaningful across reviewers' boxes.

## PR 1 — Spd3 + EWA-sandwich SIMD batch

| Bench | Tier | Notes |
|---|---|---|
| `spd3_sandwich_scalar_x16_loop` | reference | 16 distinct (M, N) pairs; per-lane scale + per-lane quaternion so the optimizer cannot constant-fold |
| `spd3_sandwich_simd_x16` | SIMD batch | same 16 inputs, single `F32x16` pass via `crate::simd` polyfill — target ≥10× faster than the scalar loop on AVX-512 (16 native lanes), ≥4× on AVX2 (2× __m256 emulation), ≥2× on NEON (4× float32x4_t) |
| `spd3_eig_smith_1961` | reference | one Smith-1961 closed-form eigendecomp, no batching yet (PR 2+ will SIMD-batch the diag-fast-path branch) |
| `spd3_from_scale_quat` | reference | the 3DGS canonical Σ = R · diag(s²) · Rᵀ — a microbench for PR 2's `GaussianBatch::covariance` hot path |

### Hardware: <fill on first measured run>

| Bench | Median (ns) | StdDev | Speedup vs scalar |
|---|---|---|---|
| `spd3_sandwich_scalar_x16_loop` | TBD | TBD | 1.0× |
| `spd3_sandwich_simd_x16` | TBD | TBD | TBD |
| `spd3_eig_smith_1961` | TBD | TBD | — |
| `spd3_from_scale_quat` | TBD | TBD | — |

> **Note** Initial commit lands the kernels + bench harness; absolute
> timings are baselined on the first CI run on the reference hardware
> (Zen4 8-core AVX-512 per the sprint prompt). Subsequent PRs append
> new rows; never overwrite prior PR rows.

## PR 2 — GaussianBatch SoA + SH eval

(populated when PR 2 lands)

## PR 3 — Projection kernel

(populated when PR 3 lands)
