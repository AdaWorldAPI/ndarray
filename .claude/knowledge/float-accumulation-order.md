# Float accumulation order across CPU tiers (2026-10-30)

READ BY: anyone setting a float tolerance against an outside oracle
(ONNX Runtime, numpy, PyTorch), or changing `gemm_f32`, `sum`, `mean_axis`.

Asked by lance-graph D-TRACT-HARVEST-0 (wishlist item 4): which order do
the f32 reductions accumulate in, and does it differ between backends?

## Measured

`examples/float_accum_probe.rs` hashes the raw bits of a fixed `gemm_f32`
(64×1024 · 1024×48) and a fixed `mean_axis` / `sum` (64×1024). Run on
five tiers; matrixmultiply picks its kernel at runtime from CPUID, so qemu
`-cpu` selects it:

| tier (how run) | gemm kernel | `gemm_f32` | `mean_axis` | `sum` |
|---|---|---|---|---|
| host, AVX-512 | avx512 | `5bacd0823f5dcadd` | `0a7a5b9e602c690b` | `c11253de` |
| qemu Haswell | fma+avx2 | `5bacd0823f5dcadd` | `0a7a5b9e602c690b` | (not printed) |
| qemu SandyBridge | avx, no FMA | `632fcf6db284df8e` | `0a7a5b9e602c690b` | `c11253de` |
| qemu Nehalem | sse2 fallback | `632fcf6db284df8e` | `0a7a5b9e602c690b` | `c11253de` |
| qemu aarch64 | neon | `5bacd0823f5dcadd` | `0a7a5b9e602c690b` | `c11253de` |

## What it means

- **`sum`, `sum_axis`, `mean_axis` are bit-identical on every tier.** Over a
  contiguous lane they use `numeric_util::unrolled_fold`: eight interleaved
  partial sums, combined `(p0+p4)+(p1+p5)+(p2+p6)+(p3+p7)`, then the
  remainder in order. The order is written in Rust, there is no FMA, and
  LLVM may not reassociate IEEE adds. A non-contiguous axis sums sub-views
  in axis order (`res = res + &subview`), also fixed. `mean_axis` divides
  that sum by the length.
- **`gemm_f32` has two classes, not five.** The FMA kernels (AVX-512,
  FMA+AVX2, NEON) agree with each other bit for bit across x86 and ARM; the
  non-FMA kernels (AVX, SSE2) agree with each other. So the accumulation
  order over `k` is the same in every kernel; only fused versus separate
  multiply-add rounding differs. Within one tier the result is
  deterministic (`matrixmultiply-threading` splits over `m`/`n`, not `k`).
- **The difference from ONNX Runtime is order, not backend.** A bound such
  as lance-graph's "≤ 4 ULP on RMSNorm, all from the mean" is the gap
  between the eight-way order above and ORT's own order. It does not move
  between ndarray's tiers, so one tolerance holds on all of them. Matching
  ORT exactly would need ORT's order (synergy-map V13), measured first.
- `masked_sum_*` are integer only and exact.

## Pinned

- `numeric_util::accumulation_order_tests::sum_and_mean_axis_follow_the_documented_eight_way_order`:
  `sum` and `mean_axis` equal a hand-written eight-way reference, bit for
  bit, and the fixture is one where a left-to-right sum differs.
- `backend::native::tests::gemm_f32_bits_fall_in_the_fma_or_non_fma_class`:
  the hash must be the FMA-class value when the host takes an FMA kernel,
  else the non-FMA value. A matrixmultiply upgrade that moves either class
  fails here; re-measure with the probe and update both this table and the
  constants.
