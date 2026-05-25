# 3DGS SIMD Forward Renderer Plan — ndarray

## Goal

Productionize `ndarray::hpc::splat3d` as the CPU-SIMD forward renderer for 3D Gaussian Splatting.

The renderer should be usable by higher layers as a deterministic kernel package:

```text
camera + splat block + render budget
        ->
projected splat footprints + optional framebuffer / visibility report
```

## Non-goals

- Do not implement 3D Tiles parsing here.
- Do not implement ArcGIS or Cesium service APIs here.
- Do not own tile graph traversal policy here.
- Do not require WGPU or GPU availability.

## Core modules

Target namespace:

```text
src/hpc/splat3d/
  mod.rs
  types.rs
  camera.rs
  covariance.rs
  projection.rs
  ewa.rs
  raster.rs
  visibility.rs
  simd.rs
  report.rs
```

## Required public DTOs

```rust
pub struct Splat3dCamera {
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub near: f32,
    pub far: f32,
}

pub struct Splat3dBlockView<'a> {
    pub pos_x: &'a [f32],
    pub pos_y: &'a [f32],
    pub pos_z: &'a [f32],
    pub scale_x: &'a [f32],
    pub scale_y: &'a [f32],
    pub scale_z: &'a [f32],
    pub quat_w: &'a [f32],
    pub quat_x: &'a [f32],
    pub quat_y: &'a [f32],
    pub quat_z: &'a [f32],
    pub opacity: &'a [f32],
    pub rgba: Option<&'a [[u8; 4]]>,
}

pub struct ProjectedSplat {
    pub screen_x: f32,
    pub screen_y: f32,
    pub depth: f32,
    pub radius_px: f32,
    pub covariance_2d: [f32; 3],
    pub opacity: f32,
    pub valid: bool,
}
```

## Hot path

1. Load SoA splat columns.
2. SIMD transform world coordinates to view space.
3. Reject behind-camera, outside-near/far, NaN, invalid scale, invalid quaternion.
4. Construct 3D SPD covariance from scale/quaternion.
5. Push covariance through EWA sandwich.
6. Compute projected screen footprint.
7. Emit projected splat data or render into framebuffer.
8. Return counters and failure reasons.

## SIMD tiers

Use existing ndarray SIMD dispatch style:

- scalar baseline
- AVX2/FMA
- AVX-512
- NEON
- runtime-dispatch optional path

Every SIMD kernel must have a scalar reference implementation and deterministic tests.

## Error handling

Do not panic on malformed splats in batch paths. Return counters:

```rust
pub struct Splat3dRenderReport {
    pub input_count: usize,
    pub projected_count: usize,
    pub rejected_behind_camera: usize,
    pub rejected_invalid_covariance: usize,
    pub rejected_too_small: usize,
    pub rejected_outside_view: usize,
    pub max_radius_px: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}
```

## Acceptance criteria

- `cargo test -p ndarray --features std,linalg,splat3d`
- Scalar and SIMD paths match within explicit tolerances.
- Invalid splats are counted, not fatal.
- EWA covariance remains PSD or is rejected with reason.
- Renderer can process a columnar block without heap allocation in the inner loop.
- Benchmarks include 1k, 10k, 100k, and 1M splat projection-only workloads.

## Follow-up hooks

- Connect to `src/hpc/pillar/ewa_sandwich_3d.rs` for certification.
- Expose projection-only mode for `lance-graph` tile preflight.
- Expose framebuffer mode for CPU preview / headless validation.
