# 3DGS Implementation Plan Index — ndarray

This directory contains the ndarray-side implementation plans for the 3DGS geospatial rebuild, the PR-X12 tensor-container expansion line, and PhiSpiral256 leaf-location codec work.

## ndarray responsibility

`ndarray` owns the numerical and rendering substrate:

- CPU-SIMD 3D Gaussian Splatting forward renderer.
- SPD covariance construction and EWA projection math.
- Render-depth, visibility, and occlusion-certification kernels.
- EWA/SYRK/BLAS/MKL backend dispatch for large-batch covariance projection.
- Domain-neutral certified field-kernel substrate pieces when repeated by multiple consumers.
- PR-X12 tensor-container expansion kernels and benchmark hooks where they touch decode-during-GEMM.
- PhiSpiral256 center generation, atom packing, distance/neighbor tables, and calibration kernels.
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

## 3DGS plans

```text
3DGS-SIMD-forward-renderer-plan.md
3DGS-error-certification-pillars-plan.md
3DGS-columnar-splat-codec-plan.md
3DGS-HHTL-CPU-cascade-plan.md
3DGS-validation-benchmark-plan.md
3DGS-4x4-cognitive-shader-SoA-plan.md
3DGS-EWA-SYRK-BLAS-MKL-crosspollination-plan.md
3DGS-certified-field-kernel-substrate-plan.md
3DGS-render-depth-certification-plan.md
```

## PhiSpiral256 leaf-location plans

```text
PhiSpiral256-LeafPlanetarium-integration-plan.md
```

This plan keeps the lanes distinct:

```text
CAM_PQ       -> meaning / semantic basin lane
PolarQuant   -> magnitude / similarity lane
PhiSpiral256 -> orthogonal local residual location lane
BGZ17        -> golden offset/stride recoverable sampling skeleton
Fisher-z     -> optional statistical angular scorer/gate after candidate ranking
```

## PR-X12 tensor-container capstone

```text
PR-X12-tensor-container-expansion-capstone.md
```

This capstone connects:

```text
x265 / HEVC through BLAS
        ->
x266 / 3DGS scene anchors
        ->
GGUF / safetensors tensor CTUs
        ->
Lance / Arrow tensor chunks
        ->
decode-during-GEMM and HHTL traversal
```

## Cross-repo boundary

`ndarray` should not own 3D Tiles, Cesium compatibility, ArcGIS service ingestion, Blender scene semantics, graph query planning, SplatShaderBlas orchestration, datalake semantics, domain adapters, or tile serving. Those live in `lance-graph`.

The intended interface is:

```text
lance-graph camera/tile decision request
        ->
ndarray HHTL/SIMD/certification kernels
        ->
certified tile/splat decision report
```

For render-depth certification, the intended interface is:

```text
lance-graph scene / mesh / splat candidate request
        ->
ndarray depth interval / visibility / occlusion kernels
        ->
render-depth certificate summary
```

For 4x4 fanout, the intended interface is:

```text
lance-graph raw-field / cognitive-shader block request
        ->
ndarray Mat4 / Sym4 / Block4 SoA kernels
        ->
certified block decision report
```

For BLAS-backed EWA projection, the intended interface is:

```text
lance-graph 3DGS block schedule
        ->
ndarray EWA/SYRK/BLAS backend selection
        ->
batched covariance projection report
```

For PhiSpiral256, the intended interface is:

```text
lance-graph leaf-location / SoA residual request
        ->
ndarray PhiSpiral256 encode / neighbor / distance / calibration kernels
        ->
packed orthogonal residual location atoms and calibration report
```

For certified field kernels, the intended interface is:

```text
lance-graph domain adapter / datalake block summary
        ->
ndarray field-kernel scoring/certification kernels
        ->
behavior-affecting certificate summary
```

For PR-X12 tensor containers, the intended interface is:

```text
GGUF / safetensors / Lance tensor block adapter
        ->
ndarray PR-X12 block decode / codebook / GEMM-adjacent kernels
        ->
cache-resident decode-during-GEMM report
```

Central principle: renderer, codec, depth, location, and field-kernel decisions should be fast, inspectable, and mathematically auditable.
