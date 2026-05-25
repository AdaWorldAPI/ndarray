# 3DGS Implementation Plan Index — ndarray

This directory contains the ndarray-side implementation plans for the 3DGS geospatial rebuild.

## ndarray responsibility

`ndarray` owns the numerical and rendering substrate:

- CPU-SIMD 3D Gaussian Splatting forward renderer.
- SPD covariance construction and EWA projection math.
- Pillar-certified error probes under `src/hpc/pillar`.
- Columnar splat payload formats and quantization carriers.
- HHTL / HEEL-HIP-TWIG-LEAF selection primitives callable by `lance-graph`.
- 4x4 cognitive-shader SoA carriers and lift/extract invariants.
- Benchmarks, golden vectors, reproducibility, and failure gates.

## Markdown convention

Program-related material should use fenced Markdown blocks so Claude Code, GitHub review, and future handovers can parse it cleanly.

Use fences for:

```text
crate/module layouts
commands
Cargo feature sets
Rust DTO sketches
schema sketches
endpoint lists
call-flow diagrams
file paths when shown as groups
```

Use inline code only for short identifiers such as `ndarray::hpc::splat3d` or `TileId`.

## Plans

```text
3DGS-SIMD-forward-renderer-plan.md
3DGS-error-certification-pillars-plan.md
3DGS-columnar-splat-codec-plan.md
3DGS-HHTL-CPU-cascade-plan.md
3DGS-validation-benchmark-plan.md
3DGS-4x4-cognitive-shader-SoA-plan.md
```

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

For 4x4 fanout, the intended interface is:

```text
lance-graph raw-field / cognitive-shader block request
        ->
ndarray Mat4 / Sym4 / Block4 SoA kernels
        ->
certified block decision report
```

Central principle: renderer decisions should be fast, inspectable, and mathematically auditable.
