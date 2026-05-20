# A1: TileInstance v2 + BlockedGrid<SplatBinList, 1, 1> refactor

Worker A1 of PR-X4 (W4-W5). **Chain dep** for A2-A6 — must land
before any other worker spawns. Owns the structural port from the
bespoke 16×16 binner to the typed BlockedGrid substrate.

## Scope

Lift the existing splat3d binner (`src/hpc/splat3d/{tile,frame,
gaussian,project,raster,sh,spd3,ply,mod}.rs`) into `splat3d_v2/` as a
sibling tree on `BlockedGrid<SplatBinList, 1, 1>` from PR-X3. Same
algorithmic shape (project → bin → sort → rasterize) on the typed
substrate, with the tier field A2 will populate.

## File moves

| v1 path                       | v2 path                              |
|-------------------------------|--------------------------------------|
| `splat3d/tile.rs`             | `splat3d_v2/tile.rs`                 |
| `splat3d/frame.rs`            | `splat3d_v2/frame.rs`                |
| `splat3d/gaussian.rs`         | `splat3d_v2/gaussian.rs`             |
| `splat3d/project.rs`          | `splat3d_v2/project.rs`              |
| `splat3d/raster.rs`           | `splat3d_v2/raster.rs`               |
| `splat3d/sh.rs`               | `splat3d_v2/sh.rs` (A3 expands)      |
| `splat3d/spd3.rs`             | `splat3d_v2/spd3.rs`                 |
| `splat3d/mod.rs`              | `splat3d_v2/mod.rs`                  |

Side-by-side per pr-x4-design § Q1 — `splat3d/` stays unchanged until
W7 closure swap. Both compile.

## Verbatim struct (pre-sprint lines 95-103)

```rust
#[repr(C, align(16))]
pub struct TileInstance {
    pub tier: u8,          // 1 = L1, 2 = L2, 3 = L3, 4 = L4
    pub _pad: [u8; 3],
    pub block_row: u16,
    pub block_col: u16,
    pub gaussian_id: u32,
    pub confidence: f32,   // replaces depth — sort key, highest-first
}
```

A1 emits `tier == 1` only. L2-L4 emission is A2's deliverable, so
**A1 is NOT gated on PR-X10 A12b's L4 Hilbert-3D fix.** For the
graphics-compat layer, `confidence = 1.0 / (depth + EPS)` so
highest-first sort recovers front-to-back order under the new key.

## BlockedGrid<SplatBinList, 1, 1> migration

`SplatBinList` is the per-block payload — `SmallVec<[TileInstance; 8]>`
or equivalent — replacing v1's `Vec<TileInstance> + Vec<u32> prefix`
hand-rolled CSR. The `<1, 1>` block-params mean **1×1 cells per
substrate block**: each tile is its own atomic block. Cascade-tier
striding belongs to A2.

Constructor: `BlockedGrid::<SplatBinList, 1, 1>::with_dims(rows, cols)`,
populated by the two-pass count+emit pattern v1 uses. The packed-u64
radix sort survives unchanged; the prefix-sum CSR is replaced by
`BlockedGrid::iter_blocks()`.

The PP-13 PR4 P0 boundary-tile fix (`floor + 1` instead of `ceil`) at
`splat3d/tile.rs:241-243` MUST be preserved verbatim in the v2 port
— a regression here silently breaks SG3.

## SIMD bundles — B-Splat + B-Interleave-Transpose

A1 consumes exactly two bundles:

- **B-Splat** (`splat_f32x16`, `splat_i32x16`): broadcast a Gaussian
  center across 16 tile lanes during the bin step.
- **B-Interleave-Transpose** (`interleave_f32x16 ∘ transpose_inplace`):
  the row-major splat3d ↔ lane-major splat4d boundary primitive. A1
  IS the v1↔v2 boundary, so this is its primary tool.

B-Gather-FMA and B-Cascade-Permute belong to A3, A2. A1 must not
reach past either bundle into raw intrinsics — breaks SG2 p95.

## Parity tests

1. **v1-vs-v2 binner parity**: feed both binners the same 16-Gaussian
   fixture (seed `0xA1_B1_NA_RY`), assert emitted `TileInstance`
   streams agree on `(tile_id ↔ (block_row, block_col), gaussian_id,
   sort order under the depth/confidence transform)`.
2. **Boundary-tile coverage regression**: the PP-13 PR4 P0 case —
   3σ-ellipse straddling a tile boundary — must produce identical
   per-tile splat counts in v2 as v1.

The 2370-test no-regression line (done-criteria #6) requires the v1
suite to still pass with `splat3d/` untouched.

## Exit criteria — when A2-A6 may spawn

- [ ] `cargo test -p ndarray --lib splat3d_v2::` green
- [ ] v1↔v2 binner parity on the 16-Gaussian fixture
- [ ] Boundary-tile coverage regression passes
- [ ] `cargo clippy -- -D warnings` clean
- [ ] `splat3d_v2::TileInstance` + `BlockedGrid<SplatBinList,1,1>`
      exported from `splat3d_v2::mod`
- [ ] A6's `frame_pipeline` skeleton can call into `splat3d_v2`
      without depending on A2..A5

No AABB or Hilbert dep, no SH or INT4 dep. A1 is the chain dep gate.
