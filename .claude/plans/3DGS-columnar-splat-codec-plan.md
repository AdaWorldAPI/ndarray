# 3DGS Columnar Splat Codec Plan — ndarray

## Goal

Define the ndarray-side columnar representation and codec primitives for 3DGS splat blocks.

The representation must be friendly to:

- SIMD batch projection
- Arrow/Lance sidecar storage in `lance-graph`
- quantization and compression
- deterministic validation
- future glTF/SPZ/3D Tiles import-export bridges

## Non-goals

- Do not implement 3D Tiles package writing here.
- Do not own metadata schemas for full geospatial tilesets.
- Do not implement a server.

## Core representation

Prefer struct-of-arrays for hot paths:

```rust
pub struct SplatBlockColumns<T> {
    pub pos_x: Vec<T>,
    pub pos_y: Vec<T>,
    pub pos_z: Vec<T>,
    pub scale_x: Vec<T>,
    pub scale_y: Vec<T>,
    pub scale_z: Vec<T>,
    pub quat_w: Vec<T>,
    pub quat_x: Vec<T>,
    pub quat_y: Vec<T>,
    pub quat_z: Vec<T>,
    pub opacity: Vec<T>,
    pub color_rgba: Vec<[u8; 4]>,
    pub feature_id: Vec<u32>,
}
```

For zero-copy interop, expose borrowed views:

```rust
pub struct SplatBlockView<'a, T> {
    pub pos_x: &'a [T],
    pub pos_y: &'a [T],
    pub pos_z: &'a [T],
    pub scale_x: &'a [T],
    pub scale_y: &'a [T],
    pub scale_z: &'a [T],
    pub quat_w: &'a [T],
    pub quat_x: &'a [T],
    pub quat_y: &'a [T],
    pub quat_z: &'a [T],
    pub opacity: &'a [T],
    pub color_rgba: &'a [[u8; 4]],
    pub feature_id: &'a [u32],
}
```

## Quantization tiers

Implement explicit tiers, not hidden magic:

1. `F32Raw`
   - reference format
   - easiest validation

2. `F16Packed`
   - scale/quaternion/opacity where tolerable
   - use existing f16 carrier machinery

3. `Bf16Packed`
   - wider exponent for geospatial scale variation
   - useful for world-space values after local origin rebasing

4. `I16LocalTile`
   - local tile coordinate frame
   - positions stored as i16/u16 with scale/offset

5. `PaletteBlock`
   - optional future tier for repeated covariance/color/SH patterns

## Tile-local coordinate policy

World geospatial coordinates should not be projected directly in f32 at large scales.

Use:

```text
ECEF / projected world coordinate
        ->
tile local origin
        ->
local f32 or quantized i16 block coordinate
```

The codec should support a `TileLocalFrame`:

```rust
pub struct TileLocalFrame {
    pub origin_world: [f64; 3],
    pub axis_x: [f64; 3],
    pub axis_y: [f64; 3],
    pub axis_z: [f64; 3],
    pub scale: f64,
}
```

## Stats emitted by codec

Every encode/decode must produce stats:

```rust
pub struct SplatCodecStats {
    pub splat_count: usize,
    pub bytes_raw: usize,
    pub bytes_encoded: usize,
    pub max_position_error: f32,
    pub rms_position_error: f32,
    pub max_scale_error: f32,
    pub max_opacity_error: f32,
    pub invalid_quaternion_count: usize,
    pub invalid_scale_count: usize,
}
```

These stats feed `3DGS-error-certification-pillars-plan.md`.

## Module layout

```text
src/hpc/splat3d/codec/
  mod.rs
  columns.rs
  local_frame.rs
  quant_f16.rs
  quant_bf16.rs
  quant_i16_tile.rs
  palette.rs
  stats.rs
```

## Acceptance criteria

- Lossless f32 roundtrip test.
- Quantized roundtrip tests with explicit max/RMS error checks.
- Tile-local coordinate tests for large ECEF-like origins.
- SIMD renderer accepts borrowed views without copying.
- Codec stats can feed certificate generation.

## Cross-repo integration

`lance-graph` should own Arrow/Lance schema and durable storage.

`ndarray` should own the hot-path memory layout and conversion kernels.

Recommended bridge:

```text
lance-graph Arrow RecordBatch
        ->
borrowed ndarray SplatBlockView
        ->
projection / certification / report
```
