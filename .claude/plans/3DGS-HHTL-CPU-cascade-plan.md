# 3DGS HHTL CPU Cascade Plan — ndarray

## Goal

Adapt the existing HHTL / HEEL-HIP-TWIG-LEAF cascade ideas into reusable CPU-side kernels for 3DGS tile and splat preselection.

This plan is not the full geospatial traversal policy. It defines fast kernels that higher layers can call.

## Cascade shape

```text
HEEL  coarse tile/block rejection
HIP   screen-space and density scoring
TWIG  covariance/opacity/quantization refinement
LEAF  exact projection/render payload
```

## Kernel responsibilities

### HEEL

Fast broad-phase rejection:

- bounding sphere/box vs frustum
- tile availability bitmask scan
- distance band classification
- optional Morton range locality scan

### HIP

Mid-tier priority estimation:

- screen-space error estimate
- projected tile/block radius
- splat density estimate
- opacity budget estimate
- camera motion/fovea relaxation factor

### TWIG

Block-level refinement:

- covariance stability estimate
- quantization error estimate
- depth bucket estimate
- weak-dependence correction hook
- certificate-ready stats

### LEAF

Exact hot path:

- call `splat3d` projection kernel
- optionally call raster kernel
- emit exact counters and certificate values

## DTOs

```rust
pub struct Hhtl3dgsRequest<'a> {
    pub camera: Splat3dCamera,
    pub block_bounds: &'a [BlockBounds],
    pub block_stats: &'a [SplatBlockStats],
    pub budget: Hhtl3dgsBudget,
}

pub struct Hhtl3dgsBudget {
    pub max_error_px: f32,
    pub min_confidence: f32,
    pub max_projected_radius_px: f32,
    pub max_blocks: usize,
    pub allow_skip_lod: bool,
}

pub struct Hhtl3dgsDecision {
    pub block_index: usize,
    pub tier_reached: HhtlTier,
    pub priority: f32,
    pub estimated_error_px: f32,
    pub action: HhtlAction,
}

pub enum HhtlAction {
    Reject,
    KeepCoarse,
    Refine,
    ProjectExact,
    RenderExact,
}
```

## SIMD strategy

- Batch bounding-volume tests.
- Batch distance-to-camera.
- Batch projected radius approximations.
- Use bitset outputs for rejected/accepted blocks.
- Keep scalar reference path for every tier.

## Integration with existing ndarray capabilities

Use or extend existing modules where possible:

- `hpc::cascade`
- `hpc::clam`
- `hpc::cam_pq`
- `distance`, `byte_scan`, `spatial_hash`
- `simd_dispatch`
- `splat3d`
- `pillar`

## Runtime principles

- No heap allocation in inner loops.
- No hidden global state except existing SIMD dispatch tables.
- No policy decisions that belong to `lance-graph`.
- Return measurable reasons for rejection/refinement.

## Acceptance criteria

- Scalar and SIMD broad-phase decisions match.
- Benchmarks for 1k, 10k, 100k block candidates.
- HHTL decision output can be consumed without depending on renderer internals.
- Exact LEAF mode can call `splat3d` projection and merge reports.
- Test cases include camera motion and foveated relaxation inputs, even if policy remains in `lance-graph`.

## Cross-repo usage

`lance-graph` calls this cascade after it has selected candidate tiles from 3D Tiles / Lance metadata.

```text
lance-graph tile candidates
        ->
ndarray HHTL cascade
        ->
ranked block decisions
        ->
lance-graph traversal/render scheduler
```
