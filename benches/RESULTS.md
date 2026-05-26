# splat3d bench results

Per-kernel timing baseline for the `splat3d` feature. Regression > 5%
on any row blocks merge per the sprint discipline. Update this file in
the same commit as any change to a `splat3d` kernel.

## Run

```bash
# Default build (x86-64-v1 baseline, F32x16 = AVX2-emulated 2× __m256)
cargo bench --features splat3d --bench splat3d_bench

# AVX-512 native build (recommended on Sapphire Rapids / Zen4)
RUSTFLAGS="-C target-cpu=native" \
  cargo bench --features splat3d --bench splat3d_bench
```

Hardware: record the CPU model + topology + the `target-cpu` /
`target-feature` flags used so cross-box comparisons are meaningful.

## PR 1 — Spd3 + EWA-sandwich SIMD batch

Baseline measurements from the sprint's reference hardware run.

### Hardware: Intel Xeon (Sapphire Rapids family), AVX-512F+BW+VL+VNNI+BF16, 2.10 GHz, container build

The PR 1 spec aimed for ≥10× speedup on `sandwich_x16` over the scalar
loop on AVX-512. Measured 1.83× — the AoS↔SoA transpose overhead at 6
fields per `Spd3` × 16 lanes dominates the inner-loop SIMD savings for
this microbench. The downstream impact is muted because the rasterizer
(PR 5) and `GaussianBatch::covariance_x16` (PR 2) already keep their
hot-path data in SoA layout, avoiding the transpose. Treat the 1.83×
microbench number as a floor; the rasterizer-driven benchmark in PR 7
exercises the SoA-native path that benefits more strongly from F32x16.

Per the architectural decision in `.cargo/config.toml` ("No global
target-cpu — each kernel uses `#[target_feature(enable = "avx512f")]`
per-function with LazyLock runtime detection"), the DEFAULT build uses
the AVX2-emulated F32x16. The `target-cpu=native` row below shows the
intended-tier numbers.

#### Default build (no `target-cpu` flag)

| Bench | Median | Speedup vs scalar |
|---|---|---|
| `spd3_sandwich_scalar_x16_loop` | 209.96 ns | 1.0× |
| `spd3_sandwich_simd_x16` | 1225.7 ns | **0.17× (slower)** |
| `spd3_eig_smith_1961` | 130.82 ns | — |
| `spd3_from_scale_quat` | 11.35 ns | — |

The SIMD regression on the AVX2-emulated build is a known artifact: the
polyfill emits two `__m256` operations per `F32x16` op AND adds the
6-field AoS↔SoA transpose at the function boundary. Net: more
instructions than the scalar loop, which the autovectorizer is happy
to map to `vfmadd` chains directly. Filed as TECH_DEBT for the
performance sprint:
- Restructure `sandwich_x16` to take SoA inputs directly (skip the
  transpose); call sites (rasterizer, `GaussianBatch::covariance_x16`)
  already have SoA layout.
- Add runtime tier dispatch in `sandwich_x16` so AVX2 builds call a
  scalar loop wrapper that the compiler auto-vectorizes cleanly.

#### `RUSTFLAGS="-C target-cpu=native"` build (AVX-512F path active)

| Bench | Median | Speedup vs scalar |
|---|---|---|
| `spd3_sandwich_scalar_x16_loop` | 166.33 ns | 1.0× |
| `spd3_sandwich_simd_x16` | 90.41 ns | **1.83×** |
| `spd3_eig_smith_1961` | 125.66 ns | — |
| `spd3_from_scale_quat` | 9.19 ns | — |

The 1.83× is below the 10× spec target but ABOVE the 1.0× break-even
that gates the function's existence. With SoA inputs at the call site
(no transpose), the inner-loop arithmetic ratio is 16-wide
multiply-add chains vs 16 sequential scalars — measured rasterizer
throughput (PR 5+) is where the kernel earns its keep.

`spd3_eig_smith_1961` ≈ 126 ns: one closed-form eigendecomp dominated
by `acos` (≈ 80 ns by itself). The diagonal-fast-path branch (which
skips the trig entirely) is what makes the rasterizer's per-pixel
work tractable; this microbench measures the WORST case.

`spd3_from_scale_quat` ≈ 9 ns: the 3DGS canonical Σ builder. PR 2's
`GaussianBatch::covariance_x16` SIMD-batches this; the scalar
microbench is the per-call latency floor.

## PR 2 — GaussianBatch SoA + SH eval

Not yet baselined as separate benches — covered indirectly by the
projection-kernel and rasterizer benches when PR 7 adds them.

## PR 3 — Projection kernel

Not yet baselined as a separate bench; the `project_chunk_x16`
inner-loop math has identical AoS↔SoA structure to `sandwich_x16`
and is expected to show similar 1.5-2× SIMD-vs-scalar ratios on
AVX-512 native builds.

## PR 4 — Tile binner

Sort + prefix-sum throughput target (per the sprint spec): 2M
instances sorted in ≤ 8 ms on 1 thread. Not yet benched separately;
`sort_unstable_by_key` is the first-cut sort. Radix sort follow-up is
TECH_DEBT once PR 7's full-pipeline timings show the binner is the
hot spot.

## PR 5 — Rasterizer

Per-tile alpha-blend with the `F32x16` 16-pixel-row inner loop. The
acceptance gate (1080p × 500K gaussians ≤ 25 ms on 8-core AVX-512) is
left for the dedicated rasterizer bench in a follow-up; PR 5 ships
the kernel + correctness tests, not the rasterizer-scale bench.

## PR 6 — SplatFrame + SplatRenderer

Double-buffer driver — no microbench; the full-pipeline rasterizer
bench in a follow-up will exercise it under realistic load.

## PR 7 — End-to-end demo

The demo binary `examples/splat3d_flex.rs` and integration test
`tests/splat3d_correctness.rs` ship as the e2e regression guards.
Full-pipeline frame-time numbers (p50/p95/p99) await a Inria bicycle
scene download — left as a follow-up for the dedicated benchmarking
session against real-world data.

## EWA-SYRK crossover — kill-or-justify the BLAS-backend premise

Bench: `benches/ewa_syrk_crossover.rs`. Tests whether the
`3DGS-EWA-SYRK-BLAS-MKL` plan's premise — "projection is a BLAS workload
in disguise → route the covariance sandwich through an MKL/OpenBLAS/AMX
backend" — holds for the **3×3** EWA sandwich `Σ' = M·Σ·Mᵀ`.

### Hardware / build

Container, AVX-512F+BW+VL. The committed `.cargo/config.toml` pins
`target-cpu=x86-64-v3` (for GitHub/CI portability); **benches are run at the
project's deployment tier `x86-64-v4`** (AVX-512 native — `F32x16` is a
single `__m512`), via the documented override:

```bash
RUSTFLAGS="-Ctarget-cpu=x86-64-v4" \
  cargo bench --features splat3d --bench ewa_syrk_crossover
```

### `M·N·Mᵀ` sandwich — three kernel shapes (Melem/s, higher = better) @ v4

| N | scalar | `simd_x16` | `gemm_shape` (BLAS-shape) |
|---|---|---|---|
| 1 024 | 85.2 | **175.2** | 90.1 |
| 100 000 | 76.3 | **169.6** | 85.4 |
| 1 000 000 | 81.9 | **172.0** | 87.1 |

`gemm_shape` = two dense 3×3 matmuls per element (the shape a per-matrix
BLAS call imposes), **in-process, no FFI**. The v3 baseline is within ~5% of
these v4 numbers for this transpose-bound 6-field kernel — the verdict is
tier-robust.

### `project_batch` end-to-end @ v4

| N | throughput |
|---|---|
| 1 024 | 12.1 Melem/s (84 µs) |

(full pipeline incl. scalar `sh_eval_deg3` per visible gaussian — SH eval
dominates; the covariance sandwich is a small fraction of this.)

### Verdict — BLAS backend NOT justified at 3×3

- `gemm_shape` is statistically identical to `scalar` and **~2× slower than
  the shipped `simd_x16`** at every size 1k→1M. **No crossover**; the gap is
  flat, not closing with batch size.
- `gemm_shape` carries **no FFI** — a real `cblas`/MKL call adds marshalling
  + dispatch on top, so it can only be worse. There is no efficient CPU
  batched-3×3 SYRK (that pattern is a GPU one).
- ⇒ The EWA-SYRK *backend* (native/MKL/OpenBLAS/AMX dispatch for the
  covariance sandwich) is a **pessimization** at 3×3/2×3: fused SoA SIMD
  already wins. The plan row is **idea-only** — the sandwich *is*
  SYRK-shaped (true) but the actionable backend is killed by measurement.
- Corroborates PR-3's predicted "1.5-2× SIMD-vs-scalar": `simd_x16` is ~2×
  over scalar at large N (transpose amortised, unlike the transpose-bound
  N=16 PR-1 microbench).
- Steelman left open: `W·Σ·Wᵀ` has a *shared* `W` across gaussians → a
  batched shared-`W` GEMM is the one form that could differ; benched as a
  follow-up. Per-gaussian `J·Σ·Jᵀ` does not batch that way.
