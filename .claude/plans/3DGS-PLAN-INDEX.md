# 3DGS Implementation Plan Index — ndarray

This directory contains the ndarray-side implementation plans for the 3DGS geospatial rebuild.

## ndarray responsibility

`ndarray` owns the numerical and rendering substrate:

- CPU-SIMD 3D Gaussian Splatting forward renderer.
- SPD covariance construction and EWA projection math.
- Pillar-certified error probes under `src/hpc/pillar`.
- Columnar splat payload formats and quantization carriers.
- HHTL / HEEL-HIP-TWIG-LEAF selection primitives callable by `lance-graph`.
- Benchmarks, golden vectors, reproducibility, and failure gates.

## Plans

1. `3DGS-SIMD-forward-renderer-plan.md`
2. `3DGS-error-certification-pillars-plan.md`
3. `3DGS-columnar-splat-codec-plan.md`
4. `3DGS-HHTL-CPU-cascade-plan.md`
5. `3DGS-validation-benchmark-plan.md`

## Cross-repo boundary

`ndarray` should not own 3D Tiles, Cesium compatibility, ArcGIS service ingestion, graph query planning, or tile serving. Those live in `lance-graph`.

The intended interface is:

```text
lance-graph camera/tile decision request
        ->
ndarray HHTL/SIMD/certification kernels
        ->
certified tile/splat decision report
```

Central principle: renderer decisions should be fast, inspectable, and mathematically auditable.
