# 3DGS Render-Depth Certification Plan — ndarray

## Goal

Capture the mathematical-bounds cross-pollination for rendering depth, visibility, occlusion, and progressive refinement.

This plan connects:

```text
3DGS EWA projection
Cesium screen-space error
Blender/CAD depth buffers
CPU-SIMD HHTL traversal
statistical error certificates
```

The renderer should not only decide what to draw. It should know how deep, how certain, and how much refinement is mathematically justified.

## Core problem

Depth in 3DGS and CAD/BIM scenes is not a single scalar problem.

A view may contain:

```text
mesh surfaces
transparent splats
overlapping Gaussian support
scan noise
quantized local frames
LOD substitutions
query-relevant hidden objects
```

A naive depth test is insufficient for certified traversal.

## Depth certificate components

```rust
pub struct RenderDepthCertificate {
    pub min_depth: f32,
    pub max_depth: f32,
    pub depth_variance: f32,
    pub projected_radius_px: f32,
    pub occlusion_confidence: f32,
    pub ordering_uncertainty: f32,
    pub quantization_depth_error: f32,
    pub covariance_depth_error: f32,
    pub total_depth_error: f32,
    pub passed: bool,
}
```

## Error terms

```text
E_depth_total =
    E_camera_transform
  + E_local_frame_quantization
  + E_covariance_projection
  + E_splat_support_overlap
  + E_sort_bucket_width
  + E_lod_substitution
  + E_sampling_discrepancy
```

The exact formula can evolve, but the certificate should keep terms separate for auditability.

## Projection path

For each splat or block:

```text
world/local center
  -> view-space position
  -> projected screen center
  -> EWA image covariance
  -> approximate depth support interval
  -> depth certificate
```

Depth support interval:

```text
z_center ± k * sigma_z
```

where `k` is selected by the render budget / confidence target.

## HHTL usage

```text
HEEL
  reject blocks outside frustum or behind near/far planes

HIP
  estimate depth interval and projected radius

TWIG
  refine overlap/ordering uncertainty

LEAF
  exact projection and optional raster/composite path
```

## Blender/CAD relevance

Blender/CAD scenes add exact mesh depth or proxy depth:

```text
mesh object depth
  -> exact or rasterized depth interval

3DGS texture/radiance skin
  -> probabilistic splat depth interval

combined object
  -> mesh depth anchors splat depth confidence
```

This enables reality-skinned CAD/BIM:

```text
CAD mesh says wall is here.
3DGS scan says visual surface is here.
Certificate reports whether the visual skin aligns within tolerance.
```

## Cesium relevance

Cesium-style traversal uses screen-space error.

3DGS depth certification adds:

```text
screen-space error
  + depth uncertainty
  + ordering uncertainty
  + occlusion confidence
```

This improves decisions for:

```text
skip LOD
refine tile
hydrate exact mesh
render splat preview
reject visually hidden block
```

## CPU-SIMD kernels

Candidate kernels:

```text
batch view-space transform
batch depth interval estimate
batch frustum/depth rejection
batch projected-radius estimate
batch occlusion bucket assignment
batch certificate reduction
```

The exact alpha compositing path may remain GPU/WGPU optional. CPU-SIMD owns planning, prefiltering, and certification.

## Sorting and buckets

Full exact splat sorting can be expensive.

Use tiered depth buckets:

```text
coarse bucket
  -> render/skip/refine decision

uncertain bucket overlap
  -> TWIG refinement or exact LEAF sort
```

Certificate reports whether bucket uncertainty is acceptable.

## Acceptance criteria

- Depth certificate DTO exists at plan/API level before implementation.
- Scalar reference computes depth interval and projected-radius estimates.
- SIMD path can batch depth rejection.
- Certificate terms remain separate, not only one opaque score.
- Blender/CAD mesh depth can be used as an anchor for 3DGS skin alignment.
- Cesium-style SSE decisions can include depth uncertainty.

## First demo

```text
one camera
one mesh plane
one small splat block near the plane
  -> compute mesh depth
  -> compute splat depth interval
  -> report alignment / uncertainty / pass-fail
```

## Wall sentence

```text
Depth is not just where something is; depth is how confidently the renderer may stop looking behind it.
```
