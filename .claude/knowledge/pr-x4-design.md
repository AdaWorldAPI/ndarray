# PR-X4 — Gaussian Splat L1–4 Cascade onto BlockedGrid (cognitive spacetime evolution kernel)

> READ BY: all ndarray agents that touch splat3d, the cognitive shader stack,
> or the spacetime cascade layer
> (savant-architect, l3-strategist, cascade-architect,
> splat3d-architect, cognitive-architect, arm-neon-specialist, sentinel-qa,
> product-engineer, truth-architect, vector-synthesis).
>
> Status: design v1 (drafted in conversation 2026-05-18). PENDS plan-review savant.
>
> Parallel docs:
> - `.claude/knowledge/pr-x3-cognitive-grid-design.md` — the BlockedGrid substrate this builds on (shipped at PR #158)
> - `.claude/knowledge/cognitive-shader-foundation.md` — the 7-layer cognitive shader vision
> - `.claude/knowledge/cognitive-distance-typing.md` — no-umbrella distance rule (still binding here)
> - `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a layering rule (still binding)
> - `.claude/rules/data-flow.md` — Rule #3 (no `&mut self` during compute, still binding)

## Context for a fresh session

If you arrive here without conversational context (token reset, new session, handover):

1. **PR-X3 shipped** (PR #158, merged 2026-05-18). It added `crate::hpc::blocked_grid::*`: `BlockedGrid<T, BR, BC>` + `blocked_grid_struct!` macro + tier-iterators (L1/L2/L3/L4) + map/bulk_apply split per data-flow Rule #3.
2. **splat3d module already exists** at `src/hpc/splat3d/{spd3,gaussian,sh,project,tile,raster,frame,ply}.rs`. It implements 3D Gaussian splatting with a **bespoke fixed 16×16 pixel tile** abstraction (`TILE_SIZE: u32 = 16` in `tile.rs`). The pipeline is project → bin → sort → rasterize.
3. **PR-X4 (this doc)** generalizes the tile abstraction onto `BlockedGrid<TileBin, BR, BC>` so:
   - Tile size becomes const-generic (BR×BC), defaulting to 64×64 to match PR-X3 L1
   - Tile binning gets multi-resolution L1/L2/L3/L4 cascade for free
   - The splat3d pipeline becomes the **cognitive spacetime evolution kernel** for the cognitive shader (per `cognitive-shader-foundation.md` and the conversation reasoning in PR #158 review)
4. **PR-X5 (queued)**: typed SIMD register-bank stacks. Runs INSIDE the per-tile composition closures.
5. **W7 (deferred, bench-gated)**: NARS truth-revision kernel that REPLACES alpha-compositing as the splat blend operator. PR-X4 ships the substrate; the closure body for W7 stays scalar in this PR.
6. **PR-X7 (queued)**: typed `cognitive_shader!` cell-DSL. Defines the typed cell signature (edge u64 + thinking [i4;32] + qualia [i4;16] + vocab u16 + covariance + opacity). PR-X4's `SplatCell<D>` type is the bridge.

**This PR is PR-X4 only.** PR-X5, PR-X7, W7 are explicit non-goals; the API is forward-compatible with each.

## Why this exists — the unification

**Gaussian splatting IS the cognitive spacetime cascade.** The two systems are mathematically identical at the substrate level:

| Aspect | 3D Gaussian Splatting (graphics) | Cognitive Shader (PR-X4 reframing) |
|---|---|---|
| **Primitive** | 3D Gaussian splat with position + anisotropic covariance + SH color + opacity | Cognitive cell with spacetime position + SPO covariance + typed state + NARS confidence |
| **L1 tile** | 16×16 screen-pixel bin | 64×64 cognitive cell block (PR-X3 default) |
| **L2 cascade** | View-frustum tile clustering | Regional resonance super-block (4×4 of L1) |
| **L3 cascade** | Scene-level LOD bucket | Scene aggregation super-block (16×16 of L2) |
| **L4 cascade** | Framebuffer / final composite | Experience memory super-block (4×4 of L3) |
| **Splat projection** | 3D → 2D screen ellipse (Jacobian of view transform) | Cognitive state → cell footprint (Jacobian of inquiry transform) |
| **Tile binning** | Each splat into all tiles its 3σ ellipse covers | Each cognitive activation into all L1 blocks its SPO footprint covers |
| **Sort order** | Front-to-back by depth | Most-confident-first by NARS truth-projection |
| **Composition** | Alpha-compositing: `C_out = α·C_splat + (1-α)·C_accum` | NARS truth-revision: `T_out = revise(T_splat, T_accum)` — W7 |
| **Spherical harmonics** | View-direction-dependent color (deg-3 = 16 coefs/channel) | Inquiry-direction-dependent cognitive state (vocab × thinking_style projection) |
| **Anisotropic covariance** | 3×3 SPD encoded via 6-param Cholesky | SPO superposition shape: which cognitive dimensions are uncertain |
| **Saturation early-exit (T_SATURATION_EPS)** | Stop compositing when alpha is near-zero | Stop revision when confidence floor reached |

**This means**: the existing splat3d pipeline (project → bin → sort → rasterize), refactored onto BlockedGrid with `crate::simd::*` inside its closures, IS the cognitive shader's spacetime evolution kernel. We don't need a separate "PR-X8 SpacetimeStream" — the Gaussian splat cascade IS the spacetime stream when L4 is interpreted as the time axis and the cascade runs every tick.

The strategic shift: PR-X4 stops being "refactor a bespoke binner onto a generic primitive" and becomes "promote the splat3d pipeline to a typed, multi-resolution cognitive evolution operator."

## The (4×4)×(4×4)×(4×4)×(4×4) tier scheme — splat math grounding

The PR-X3 L1→L2→L3→L4 hierarchy with dim progression `64 → 256 → 4096 → 16384` (per-side) and tier-stride branching `4×4 / 16×16 / 4×4` corresponds to a **4-level Gaussian pyramid** with anisotropic per-tier covariance support of (4×4) cells:

```
L4 (16384²)  framebuffer / experience memory   ←  4×4 super-grid of L3
L3 (4096²)   scene aggregation                  ←  16×16 super-grid of L2
L2 (256²)    regional resonance                  ←  4×4 super-grid of L1
L1 (64²)     per-cell context                    ←  4×4 covariance footprint per splat
```

Each tier's "(4×4)" is the local Gaussian covariance support — the 16-sample anisotropic kernel that defines the splat's footprint at that scale. A splat at L1 has a 4×4 cell footprint with anisotropic covariance; at L2, the SAME splat (downsampled) has a 4×4 super-block footprint = 16×16 cells of its L1 footprint; etc. Cascading the covariance through 4 tiers gives the L1→L4 cognitive context window.

**Area-wise**: each tier covers 16× more area than the previous (uniform area-branching), giving L4/L1 area ratio = 16⁴ = 65,536. This matches the cell-count ratio 16384²/64² = 65,536 exactly.

**Per-dim branching** is non-uniform (4 / 16 / 4) because the pyramid is not isotropic in tier-stride — the L2→L3 transition has a wider gather to span the scene-aggregation scale. Splat composition handles this naturally: at each tier, sort and composite the splats whose 3σ ellipse intersects the current tile.

## The splat3d refactor — from bespoke 16×16 to BlockedGrid

### Current state (master)

```rust
// src/hpc/splat3d/tile.rs (existing)
pub const TILE_SIZE: u32 = 16;
pub struct TileInstance { tile_id: u32, gaussian_id: u32, depth: f32 }
pub struct TileBinning {
    instances: Vec<TileInstance>,    // sorted: tile_id ASC, depth ASC
    tile_prefix: Vec<u32>,            // tile i's instances = instances[tile_prefix[i]..tile_prefix[i+1]]
    n_tiles_x: u32,
    n_tiles_y: u32,
}
impl TileBinning {
    pub fn from_projected(...) -> Self { ... }
    pub fn tile_instances(&self, tile_id: u32) -> &[TileInstance] { ... }
}
```

Issues:
- `TILE_SIZE` is a hardcoded `const`, not const-generic — can't pick 64×64 for cache fit
- No tier cascade (no L2/L3/L4 awareness — just one flat tile grid)
- The `Vec<TileInstance> + Vec<u32> prefix` is a hand-rolled CSR — `BlockedGrid<SplatBinList, 1, 1>` would express the same with the typed substrate
- Per-tile composition (`raster::rasterize_tile`) is bespoke per-tile loop; no `map_l1` / `bulk_apply_l1` integration

### PR-X4 target shape

```rust
// src/hpc/splat3d_v2/tile.rs (new — kept side-by-side with splat3d/ until splat3d/raster.rs migrates)
//
// (Or in-place migration of src/hpc/splat3d/tile.rs — see open question Q1.)

/// One (tile, gaussian) binding emitted during binning. Same shape as the
/// existing TileInstance, but `tile_id` is now (tier, block_row, block_col)
/// for multi-resolution cascade.
#[repr(C, align(16))]
pub struct TileInstance {
    pub tier: u8,          // 1 = L1, 2 = L2, 3 = L3, 4 = L4
    pub _pad: [u8; 3],     // to keep block_row 4-byte aligned
    pub block_row: u16,
    pub block_col: u16,
    pub gaussian_id: u32,
    pub confidence: f32,   // replaces `depth` — sort key (highest-first for NARS revision)
}

/// Tier-aware tile binning. Generic over tile shape (BR×BC) — defaults to
/// 64×64 to match PR-X3 L1 cache fit; pick 16×16 if rendering to AMX BF16 tile.
pub struct TileBinning<const BR: usize = 64, const BC: usize = 64> {
    /// One bin list per (tier, block_row, block_col). Indexed by
    /// `tier_offset[tier] + block_row * n_block_cols[tier] + block_col`.
    instances: Vec<TileInstance>,
    /// Per-tier per-block prefix sums into `instances`.
    /// `tier_prefix[tier]` is the (n_block_rows × n_block_cols + 1)-length
    /// prefix-sum vector for tier `tier`.
    tier_prefix: [Vec<u32>; 4],
    /// Dimensions per tier (for index recovery).
    tier_dims: [(u32, u32); 4],
}

impl<const BR: usize, const BC: usize> TileBinning<BR, BC> {
    /// Multi-tier binning. Each splat is inserted into all tiles AT EACH TIER
    /// whose 3σ ellipse (projected at that tier's covariance scale) intersects.
    pub fn from_projected_cascade<const D: usize>(
        splats: &ProjectedBatch<D>,
        framebuffer_dim: (u32, u32),
    ) -> Self { ... }

    /// Backward-compat single-tier constructor — only emits L1 bins.
    pub fn from_projected_l1<const D: usize>(
        splats: &ProjectedBatch<D>,
        framebuffer_dim: (u32, u32),
    ) -> Self { ... }

    /// O(1) per-tile slice access. `tier ∈ 1..=4`.
    pub fn tile_instances(&self, tier: u8, block_row: u16, block_col: u16) -> &[TileInstance] { ... }

    /// Iterate L1 tiles paired with their bin lists, suitable for
    /// `BlockedGrid::map_l1` consumption.
    pub fn l1_bins(&self) -> impl Iterator<Item = (u16, u16, &[TileInstance])> { ... }
}
```

### Per-cell splat representation

```rust
/// A single Gaussian splat in cognitive spacetime. Generic over the inquiry
/// dimension `D` (D=2 for 2D screen-space, D=3 for 3D scene-space,
/// D=4 for spacetime, D=N for high-dim cognitive inquiry space).
#[repr(C)]
pub struct Splat<const D: usize> {
    /// Spacetime position in the cognitive grid. For D=2: (row, col).
    /// For D=3: (row, col, t). For D=4: (row, col, t, thinking_axis).
    pub pos: [f32; D],

    /// Anisotropic covariance, encoded as the strictly-lower-triangular
    /// part of the Cholesky factor (D=2 → 3 floats; D=3 → 6 floats).
    /// At tier `n`, the effective covariance is `cov * 4^(n-1)` — the
    /// (4×4) per-tier branching of the cascade.
    pub cov: SplatCovariance<D>,

    /// Typed cognitive cell state (forward-compatible with PR-X7).
    pub cell: SplatCell,

    /// NARS truth projection. Sort key for tile-list ordering (higher = composited first).
    /// Replaces `depth` from the bespoke splat3d pipeline.
    pub confidence: f32,
}

/// Typed cognitive cell state (matches the conversation cell layout from the
/// PR #158 review). Forward-compatible with PR-X7's `cognitive_shader!`
/// macro: when X7 lands, this becomes a generated struct.
#[repr(C, align(8))]
pub struct SplatCell {
    /// CausalEdge64 mantissa — the particle / collapsed truth state.
    pub edge: u64,
    /// 32-dim thinking-style vector, INT4 packed (16 bytes).
    pub thinking: [u8; 16],
    /// 16-dim qualia vector, INT4 packed (8 bytes).
    pub qualia: [u8; 8],
    /// 12-bit codebook index into the 4096-atom CAM vocabulary (u16 carrier).
    pub vocab: u16,
}

pub enum SplatCovariance<const D: usize> {
    /// Identity covariance scaled by `sigma²` (isotropic).
    Isotropic { sigma2: f32 },
    /// Diagonal covariance (axis-aligned anisotropy).
    Diagonal { diag: [f32; D] },
    /// Full anisotropic, stored as lower Cholesky factor.
    /// Length = D * (D + 1) / 2.
    Cholesky { lt: [f32; { D * (D + 1) / 2 }] },  // const-generic length when stable
}
```

For v1 we ship `D ∈ {2, 3}` only (matches existing splat3d use cases). Higher-D cognitive inquiry spaces (D=4 spacetime, D=N high-dim) deferred to PR-X4.1.

### Composition surface — forward-compatible with W7

```rust
impl<const BR: usize, const BC: usize> TileBinning<BR, BC> {
    /// Compose all L1 splats into a target BlockedGrid. The closure is the
    /// per-cell splat blend operator. In v1 this is alpha-compositing
    /// (existing splat3d::raster behavior). W7 replaces it with NARS
    /// truth-revision; the closure boundary is the swap point.
    ///
    /// # Data-flow rule
    ///
    /// PRIMARY compute path — returns a new grid; input splat list and bin
    /// structure are not mutated. Per `.claude/rules/data-flow.md` Rule #3.
    pub fn compose_l1<F, T>(
        &self,
        splats: &[Splat<2>],
        blend: F,
    ) -> BlockedGrid<T, BR, BC>
    where
        T: Copy + Default,
        F: Fn(SplatCell, T) -> T,  // (incoming splat cell, accumulator) -> new accumulator
    { ... }

    /// Multi-tier cascade composition. Builds L1 then bubbles up through
    /// L2/L3/L4 with anisotropic Gaussian downsampling at each step.
    /// The cognitive shader's spacetime evolution operator.
    ///
    /// # Data-flow rule
    /// PRIMARY compute path — returns the full pyramid as four new grids.
    pub fn compose_cascade<F, T>(
        &self,
        splats: &[Splat<2>],
        blend: F,
    ) -> SplatPyramid<T, BR, BC>
    where
        T: Copy + Default,
        F: Fn(SplatCell, T) -> T,
    { ... }
}

/// 4-level Gaussian pyramid output from `compose_cascade`. Each level is a
/// `BlockedGrid<T, 64, 64>` at the dimension specified by PR-X3's L1-L4 alias
/// impls. L1 is the finest (per-cell); L4 is the coarsest (framebuffer-scale).
pub struct SplatPyramid<T, const BR: usize = 64, const BC: usize = 64> {
    pub l1: BlockedGrid<T, BR, BC>,  // 64×64 base blocks
    pub l2: BlockedGrid<T, BR, BC>,  // 4× downsample
    pub l3: BlockedGrid<T, BR, BC>,  // 16× from L2
    pub l4: BlockedGrid<T, BR, BC>,  // 4× from L3
}
```

## Layering rule (still binding)

PR-X4 is **pure layout + scheduling**. It contains:
- ZERO `#[target_feature]` attributes
- ZERO `use crate::simd_avx512` / `simd_avx2` / `simd_neon` / `simd_arm` per-arch imports
- ZERO `cfg(target_feature = ...)` gates
- ZERO raw `_mm*` / `vld*` / `_pdep_*` intrinsics
- ZERO distance-aware API (no `distance(splat_a, splat_b)`, no `enum DistanceMetric`)

SIMD dispatch happens **inside the consumer's closure body** passed to `compose_l1` / `compose_cascade`, via `crate::simd::*` (W1a contract). For PR-X4 the closure body uses scalar inner loops; PR-X5 will land typed register-bank primitives that closures call.

The W7 NARS revision closure is identical-shape to the v1 alpha-compositing closure — same `Fn(SplatCell, T) -> T` signature. W7 replaces the function body, not the boundary.

## Distance-typing guardrail

The cognitive distance between two splats (e.g., NARS truth-similarity, palette-256 hamming, Base17 L1) IS a typed metric that lives in `crate::hpc::cognitive::*` (W7), NOT a method on `TileBinning` or `Splat`. PR-X4 must not introduce:
- `Splat::distance_to(&self, other: &Splat) -> f32`
- `TileBinning::sort_by_distance<F>(...)` umbrella
- `enum BlendMode { AlphaCompose, NarsRevise, HammingMerge, ... }` (the closure boundary IS the dispatch — no enum needed)

Module headers reference `.claude/knowledge/cognitive-distance-typing.md` and warn against extension toward distance.

## Tier semantics — splat covariance per level

When `compose_cascade` builds the pyramid, each tier downsamples the splat covariance by a fixed factor:

| Source tier | Target tier | Per-side downsample | Covariance scale | Cognitive interpretation |
|---|---|---|---|---|
| L1 (64²) | L2 (256²) | 1× cells but 4× block-area | `cov * 16` (4² area scale) | Regional resonance — broaden the wave |
| L2 (256²) | L3 (4096²) | 16× cells | `cov * 256` (16² area scale) | Scene aggregation — full-context bubble-up |
| L3 (4096²) | L4 (16384²) | 4× cells | `cov * 16` (4² area scale) | Framebuffer / experience snapshot |

The cascade is a **multi-resolution Gaussian pyramid** — standard graphics technique, repurposed as the cognitive context-window scaling operator. Splats with small L1 covariance (high local certainty) contribute primarily at fine scales; splats with large L4 covariance (broad uncertainty) contribute primarily at coarse scales. The pyramid IS attention.

## Tests required

### Unit tests for `TileBinning<BR, BC>`

- `from_projected_l1` on 100 splats with 128×128 framebuffer (4 L1 tiles): every splat appears in at least one tile's bin
- `from_projected_l1` correctness: a splat with center (50, 50) and 3σ=10 lands in tile (0, 0) but NOT in tile (0, 1) (column boundary at 64)
- `from_projected_l1` boundary: a splat spanning two tiles appears in BOTH tiles' bins
- `from_projected_cascade` correctness: same splat appears in L1, L2, L3, L4 bins with appropriately scaled covariance
- `tile_instances(tier, br, bc)` returns sorted-by-confidence-DESC slice
- Empty splat list → empty bins, no panic
- 64×64 tile shape (default) parity vs 16×16 tile shape (legacy splat3d): same splats produce same L1 composite under alpha-compositing

### Unit tests for `Splat<D>` / `SplatCell`

- `SplatCell` exact size: 8 (edge) + 16 (thinking) + 8 (qualia) + 2 (vocab) + 6 (padding to 40) = 40 bytes (verify with `std::mem::size_of`)
- `SplatCovariance::Cholesky` round-trips through projection (decompose, project, recompose)
- `Splat<3>` projection to `Splat<2>` via the existing `splat3d::project::project_batch` Jacobian still produces valid 2D ellipses

### Composition tests

- `compose_l1` with no-op blend closure (`|_, acc| acc`) produces the default-filled grid
- `compose_l1` with sum blend (`|s, acc| acc + s.cell.vocab as T`) produces sum of all splat vocab values per cell
- `compose_cascade` parity: L1 output of cascade equals standalone `compose_l1` output
- `compose_cascade` Gaussian-pyramid property: each level's mean cell value ≈ next-coarser level's mean cell value (within float tolerance)
- Empty splat list cascade → all four levels are default-filled

### Integration tests

- splat3d-parity: the existing `splat3d::raster::rasterize_frame` produces equivalent pixel output to PR-X4 `compose_l1` with alpha-blend closure. Bit-exact under a deterministic splat batch; tolerance ε under SIMD reorderings.
- BlockedGrid integration: `compose_l1` output is a valid `BlockedGrid<u32, 64, 64>` whose `blocks_l1()` iterator works as expected.
- W4 `bulk_apply` composition: `compose_l1` output can be post-processed via `bulk_apply` (no-op cleanup pass) without crashing.

## Out of scope — explicitly NOT in PR-X4

1. **Typed SIMD register-bank stacks** (`StackedF32x16<N>`, `AmxBf16Tile`, `Int4x32`) → PR-X5
2. **The `cognitive_shader!` typed cell-DSL** that emits `SplatCell` from a user-friendly declaration → PR-X7
3. **NARS truth-revision blend kernel** — v1 ships alpha-compositing as the default blend closure; the W7 closure is a drop-in replacement → W7
4. **Higher-D inquiry space** (`Splat<4>`, `Splat<N>` for cognitive thinking-axis) → PR-X4.1
5. **GPU dispatch** — v1 is CPU only via `crate::simd::*`. GPU shipping is a separate PR (likely deferred until WebGPU bindings are stable in our stack)
6. **Streaming temporal axis** as an explicit type (`Stream<Item = SplatPyramid>`) — for v1 the time axis is implicit (caller re-runs the cascade per tick). Explicit streaming = PR-X4.2 if needed
7. **Sparse splat storage** (`HashMap<(u16,u16), Splat>`) — out of scope; if needed, separate PR
8. **PLY/USD I/O changes** — `splat3d::ply` stays as-is; PR-X4's `Splat<D>` is convertible to/from `Gaussian3D` via `From`/`Into` impls

## Worker decomposition (SEQUENTIAL — the binding protocol)

Same sequential 5-10 Sonnet + 1 Opus coordinator pattern as PR-X3. Per-worker file scoping enforced via `.claude/settings.json` per-area allowlist (already tightened in PR-X3).

### File layout

```
src/hpc/splat3d/v2/        (NEW directory — kept side-by-side with existing splat3d files
                            until raster.rs migrates; final rename in cleanup commit)
├── mod.rs                  — coordinator: submodule decls + re-exports
├── tile.rs                 — A1: TileInstance + TileBinning<BR, BC> struct + accessors
├── bin.rs                  — A2: from_projected_l1 + from_projected_cascade impls
├── splat.rs                — A3: Splat<D> + SplatCell + SplatCovariance
├── compose.rs              — A4: compose_l1 + compose_cascade + SplatPyramid
└── tests.rs                — A5: integration tests + parity vs existing splat3d
```

### Worker phases

| # | Phase | Owns | Depends on |
|---|---|---|---|
| 1 | Plan v1 (this doc) | coordinator | — |
| 2 | Plan-review savant | savant agent | this doc |
| 3 | Plan v2 corrector | coordinator | savant verdict |
| 4 | Worker A1 (tile.rs) | new TileInstance + struct + accessors | PR-X3 BlockedGrid |
| 5 | Worker A2 (bin.rs) | from_projected_l1 + from_projected_cascade | A1 |
| 6 | Worker A3 (splat.rs) | Splat<D> + SplatCell + SplatCovariance | (parallel with A2 — different file) |
| 7 | Worker A4 (compose.rs) | compose_l1 + compose_cascade + SplatPyramid | A1 + A2 + A3 |
| 8 | Worker A5 (tests.rs) | integration tests + splat3d parity | A4 |
| 9 | Codex P0 audit | codex agent | A1-A5 combined |
| 10 | Coordinator fix P0s | coordinator | audit verdict |
| 11 | P2 savant pre-merge | savant agent | post-P0 branch |
| 12 | Coordinator apply tightenings | coordinator | P2 verdict |
| 13 | Merge ladder | — | — |

**Parallelism**: A2 + A3 can spawn in parallel after A1 lands (different files, A2 uses `TileBinning` from A1 to bin, A3 defines `Splat`/`SplatCell` independently). A4 needs both A2 and A3.

### Worker isolation rule

Every Sonnet sprint worker runs with `isolation: "worktree"` and explicit per-file scope in the prompt. Coordinator (Opus) integrates by cherry-picking. Settings.json already has per-area scoping (`Edit(src/{**})`); workers cannot escape their assigned file without prompt-level override.

## Verification commands

Identical to PR-X3 protocol:

```bash
cargo check -p ndarray --no-default-features --features std,splat3d
cargo test -p ndarray --lib --no-default-features --features std,splat3d hpc::splat3d
cargo test --doc -p ndarray --no-default-features --features std,splat3d hpc::splat3d
cargo fmt --all -- --check
cargo clippy -p ndarray --no-default-features --features std,splat3d -- -D warnings
```

All five must pass green.

## Cross-references

- `.claude/knowledge/pr-x3-cognitive-grid-design.md` — the BlockedGrid substrate (PR #158, merged)
- `.claude/knowledge/pr-x3-plan-review.md` — Phase 2 savant protocol shape reference
- `.claude/knowledge/pr-x3-codex-audit.md` — Phase 11 audit protocol shape reference
- `.claude/knowledge/pr-x3-p2-savant-review.md` — Phase 13 pre-merge protocol shape reference
- `.claude/knowledge/cognitive-shader-foundation.md` — full 7-layer cognitive shader vision
- `.claude/knowledge/cognitive-distance-typing.md` — no-umbrella distance rule (still binding)
- `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a layering rule (still binding)
- `.claude/rules/data-flow.md` — Rule #3 (still binding)
- `src/hpc/splat3d/tile.rs` — existing bespoke 16×16 tile binner (the refactor target)
- `src/hpc/splat3d/raster.rs` — existing alpha-compositing kernel (the W7 swap target)
- `src/hpc/splat3d/project.rs` — existing 3D → 2D projection (reused, no change needed)
- `src/hpc/splat3d/gaussian.rs` — existing `Gaussian3D` (convertible to `Splat<3>` via `From`/`Into`)
- `src/hpc/blocked_grid/aliases.rs` — PR-X3 L1/L2/L3/L4 alias impls used by `SplatPyramid`

## Open questions (for the plan-review savant)

1. **Side-by-side `splat3d/v2/` vs in-place migration of `splat3d/`?** Side-by-side lets the old code keep running through the PR; in-place forces all callers to migrate atomically. Lean: side-by-side, with a `splat3d::v2::` re-export deprecating the bespoke API over a cycle.

2. **`SplatCovariance<D>` enum vs trait object?** Three variants (Isotropic / Diagonal / Cholesky) cover 99% of splat use cases. An `enum` keeps `Splat<D>` `Copy` (no heap, fast); a `trait object` would generalize. For PR-X4 lean: `enum` for the perf path, document the trait-extension path as future work.

3. **Sort key: `confidence: f32` vs `confidence: u16` (fixed-point)?** Existing `splat3d` uses `depth: f32`. Float sort is fine but breaks bit-exact determinism across SIMD reorderings. Fixed-point u16 (Q1.15) would be deterministic. Lean: keep `f32` for v1 parity with splat3d; revisit when W7 lands.

4. **`SplatPyramid<T, BR, BC>` always 4-level vs configurable depth?** Always-4-level matches the cognitive L1-L4 tier scheme exactly. Configurable depth would require runtime tier-count handling, complicating the impl. Lean: always 4. Cognitive use cases that want fewer tiers can ignore the higher levels (zero-cost since they're just additional `BlockedGrid` allocations).

5. **Per-tier compose closure or single closure for all tiers?** A single closure unifies the blend op across tiers (cleaner). Per-tier closures would allow tier-specific blend (e.g., NARS-revision at L1, mean-pool at L4). Lean: single closure for v1; per-tier override as a follow-up if cognitive practice demands it.

6. **`SplatCell` packed layout vs explicit fields?** 40-byte size (8 + 16 + 8 + 2 + 6 pad) fits 1.6 cache lines per cell — not aligned to 64-byte cache line. Could pad to 64 bytes (waste 24 bytes per cell) for cache-line alignment, or pack tighter via bit-packing (combine vocab + tier hint into one u16). Lean: 40-byte unpadded for v1, document the alignment trade-off; PR-X7 may repack via the typed cell-DSL.

7. **D=2 only in v1 vs D=2 and D=3?** D=2 matches `compose_l1` / `compose_cascade` over a 2D grid (the standard splat output). D=3 is needed for splat-space operations BEFORE projection (e.g., culling, sorting in scene space). The existing `splat3d::gaussian::Gaussian3D` is effectively `Splat<3>` already. Lean: ship D=2 AND D=3 in v1 (the existing pipeline needs both); D=4 and D=N deferred.

## Done criteria

PR-X4 is done when:
- All worker spec items implemented per the 5-worker decomposition (A1-A5)
- Codex P0 audit passes with 0 P0 — **including SAFETY-claim verification gate added per PR-X3.1 backlog** (simulate adversarial iterator usage on `TileBinning::tile_instances` to catch latent aliasing UB)
- `cargo check / test --lib / test --doc / fmt / clippy` all green with `--features std,splat3d`
- Layering rule verified (zero per-arch imports / target_feature / raw intrinsics in `src/hpc/splat3d/v2/`)
- Distance-typing guardrail verified (zero umbrella-distance API surface)
- splat3d-parity test passes: existing `splat3d::raster::rasterize_frame` output matches PR-X4 `compose_l1` with alpha-blend closure on a deterministic splat batch
- P2 savant pre-merge review delivers SHIP verdict
- PR description includes the cognitive-spacetime-cascade framing so downstream agents understand WHY this isn't "just a refactor"

## Token-reset safety notes (for fresh sessions)

If you're picking up after a token reset:

1. Read this entire doc first.
2. Then read `pr-x3-cognitive-grid-design.md` — the BlockedGrid substrate this builds on.
3. Then read `cognitive-shader-foundation.md` — the full 7-layer vision.
4. Check `git log --oneline -10` on the PR-X4 branch and on `master` to see what shipped.
5. The conversation context that led to this doc: PR #158 (PR-X3) merged on 2026-05-18. In the post-merge discussion, the user mapped the cognitive shader's L1-L4 tier scheme to 3D Gaussian splatting's multi-resolution pyramid, observing that the (4×4)×(4×4)×(4×4)×(4×4) cascade is exactly the splat3d tile-binning pipeline at increasing scales. This PR promotes splat3d/tile.rs from "bespoke binner" to "cognitive spacetime evolution kernel". The framing matters for understanding why we're not just refactoring — we're unifying the graphics splat pipeline with the cognitive shader pipeline at the substrate level. PR-X5 (typed SIMD) and PR-X7 (typed cell-DSL) layer on top.
6. The (4×4)×(4×4)×(4×4)×(4×4) tier scheme corresponds to a 4-level Gaussian pyramid with 16× area branching at each tier (per-dim branching is non-uniform 4 / 16 / 4 due to the cognitive context-window scaling — see §"The (4×4)×(4×4)×(4×4)×(4×4) tier scheme" above).
7. W7 will replace the alpha-compositing closure with NARS truth-revision. PR-X4 ships the substrate; the closure swap is W7.
