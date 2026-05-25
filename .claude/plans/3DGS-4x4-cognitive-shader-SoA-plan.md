# 3DGS 4x4 Cognitive-Shader SoA Plan — ndarray

## Goal

Promote the 3DGS math carrier from a narrow 3x3-only view into a 4x4 cognitive-shader-friendly SoA carrier while preserving the correct 3DGS covariance spine.

Important distinction:

```text
3x3 SPD covariance
  remains the mathematically correct spatial covariance for each 3D Gaussian

4x4 carrier
  becomes the homogeneous / temporal / semantic / shader-lane envelope around that covariance
```

Do not destroy the 3x3 SPD invariant. Lift it into a 4x4 block where needed.

## Why 4x4

The 4x4 shape maps better to:

```text
homogeneous transforms
SIMD lane groups
cognitive-shader-driver BindSpace columns
Mat4 camera/world transforms
quaternion / covariance / opacity packing
(4x4)^4 tensor-block fanout
```

It also gives a natural bridge between rendering, registration, ultrasound frame fusion, genetics-like transition matrices, and neuronal adjacency blocks.

## Core representation

Add or stabilize a Mat4 carrier under the existing linalg path:

```text
src/hpc/linalg/
  mat4.rs
  spd4.rs
  block4.rs
```

Candidate types:

```rust
pub struct Mat4x4<T> {
    pub m: [[T; 4]; 4],
}

pub struct Sym4<T> {
    pub a00: T, pub a01: T, pub a02: T, pub a03: T,
                pub a11: T, pub a12: T, pub a13: T,
                            pub a22: T, pub a23: T,
                                        pub a33: T,
}

pub struct Splat4Carrier<T> {
    pub spatial_sigma_3x3: [T; 6],
    pub lane4: [T; 4],
    pub transform4x4: Mat4x4<T>,
}
```

## 3x3 to 4x4 lift

Use explicit lifts instead of pretending 3x3 and 4x4 mean the same thing.

```text
3DGS spatial covariance Sigma3
        ->
4x4 homogeneous carrier Sigma4
        ->
projection / temporal / semantic / shader lane operations
        ->
extract spatial 3x3 or image-space 2x2 when needed
```

Example lift:

```rust
pub fn lift_spd3_to_sym4(sigma3: [f32; 6], w_lane: [f32; 4]) -> Sym4<f32>;
pub fn extract_spd3_from_sym4(sym4: Sym4<f32>) -> [f32; 6];
```

## Cognitive-shader SoA layout

Prefer 4-lane grouped SoA columns:

```text
BindSpace4
  lane0: position / nucleotide / neuron source / feature id shard
  lane1: covariance / transition / edge weight / local statistic
  lane2: opacity / expression / activation / confidence
  lane3: time / phase / semantic role / provenance
```

Concrete splat columns:

```text
pos_xyzw[]
scale_xyz_opacity[]
quat_xyzw[]
color_rgba[]
feature_id_time[]
certificate_confidence[]
```

## (4x4)^4 tensor-block fanout

Interpret `(4x4)^4` as a four-level block grammar, not as one giant dense matrix.

```text
level 0: Mat4 local carrier
level 1: 4x4 block of carriers
level 2: 4x4 block of blocks
level 3: 4x4 graph/tile/neural field super-block
```

This gives hierarchical locality:

```text
splat -> block -> tile -> region -> graph domain
```

## Domain-neutral kernels

Create kernels that do not care whether the 4x4 block is geospatial, ultrasound, genetic, or neuronal.

```rust
pub trait Block4Kernel<T> {
    fn score_block(&self, block: &Block4<T>) -> f32;
    fn contract(&self, parent: &Block4<T>, child: &Block4<T>) -> f32;
    fn certify(&self, block: &Block4<T>) -> Block4Certificate;
}
```

## 3DGS compatibility rules

- Keep `Spd3` as the authoritative spatial covariance.
- Use `Mat4x4` for transforms and carriers.
- Use `Sym4` only when the extra lane has defined meaning.
- Image-space EWA still extracts the proper spatial covariance path.
- Pillar tests must assert that 4x4 lifts do not break 3x3 invariants.

## Pillar additions

Add substrate probes:

```text
Pillar-18: 3x3-to-4x4 lift preserves spatial SPD invariants
Pillar-19: Block4 contraction under HHTL hierarchy
Pillar-20: 4-lane SoA equivalence against scalar AoS reference
```

## Acceptance criteria

- 4x4 carrier compiles behind `linalg` or a specific feature gate.
- `splat3d` can use 4x4 transforms without losing 3x3 covariance correctness.
- SoA 4-lane layout has scalar reference tests.
- HHTL cascade can score 4x4 blocks.
- Pillar tests prove lift/extract invariants.

## Cross-repo link

`lance-graph` should use this as the numeric substrate for the 4x4 cross-domain fanout plans.
