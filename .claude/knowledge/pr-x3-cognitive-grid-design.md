# PR-X3 — CognitiveGrid: hierarchical block layout for cognitive shader + spatial splat BLAS

> READ BY: all ndarray agents that touch the cognitive shader stack
> (savant-architect, l3-strategist, cascade-architect,
> cognitive-architect, arm-neon-specialist, sentinel-qa, product-engineer,
> truth-architect, vector-synthesis, splat3d-architect).
>
> P0 TRIGGERS for this doc:
> - A new sprint touching `src/hpc/cognitive_grid.rs` or its consumers
> - Any block-shape change (BLK_ROW / BLK_COL const generics)
> - Any hierarchical-tier change (L1/L2/L3/L4 boundary semantics)
> - Any macro-emission change in `cognitive_grid_struct!`
>
> Parallel docs:
> - `.claude/knowledge/w3-w6-soa-aos-design.md` — the SoA/AoS foundation this builds on
> - `.claude/knowledge/cognitive-shader-foundation.md` — ndarray's role in the 7-layer stack
> - `.claude/knowledge/cognitive-distance-typing.md` — the no-umbrella distance rule
> - `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a layering rule (user code → crate::simd → simd_{type}.rs)

## Context for a fresh session

If you arrive here without conversational context (token reset, new session, handover), here is the minimum you need to know:

1. **W3-W6 shipped** (PR #156, merged 2026-05-18). It added `SoaVec<T, N>`, `soa_struct!` macro, `aos_to_soa<T, N, F: Fn(&T) -> [f32; N]>`, `soa_to_aos<T, N, F>`, `bulk_apply`, `bulk_scan` to `src/hpc/{soa,bulk}.rs`. All scalar, no `#[target_feature]`, no per-arch imports, no distance baked in.
2. **PR #157 is open** (P2 savant follow-up): adds f32-only-scope docs + `hpc::soa`-vs-`simd_ops` rationale + ungated integration test.
3. **PR-X1 / PR-X2 are designed but not started** — see `cognitive-shader-foundation.md` §"Current Gaps":
   - PR-X1: `MultiLaneColumn`, `Fingerprint::as_u8x64`, `array_window`, `simd::*` re-exports
   - PR-X2: `#[soa(pad_to_lanes=N)]` macro attribute + generalized `aos_to_soa<T, U, N>`
4. **PR-X3 (this doc)**: `CognitiveGrid<T, BR, BC>` hierarchical block-padded grid + `cognitive_grid_struct!` SoA-of-grids macro. **Layout only**. Scalar. No SIMD primitives. Forward-compatible with the per-arch SIMD swap that lands in PR-X5 / W7.
5. **PR-X5 (planned)**: Typed SIMD register-bank stacks (`StackedU64x8<N>`, `StackedF32x16<N>`, `StackedF64x8<N>`, `AmxTile<T, R, C>`) in `crate::simd::*`. Per-arch LazyLock dispatch.
6. **W7 (deferred, bench-gated)**: Typed cognitive distance bulk fns (palette-256, hamming popcount early-exit, Base17 L1, BF16 mantissa direct-transform) + the actual CausalEdge64 mantissa cell kernel.

**This PR is PR-X3 only.** PR-X5 and W7 are explicit non-goals here.

## Why this exists

The cognitive shader stack (per `cognitive-shader-foundation.md`) operates on 2-D grids at multiple block hierarchies that correspond simultaneously to:

- **Cache hierarchy** — L1 (~32 KB), L2 (~512 KB), L3/LLC (~128 MB), RAM (~2 GB)
- **Resolution pyramid** — irreducible cell block → regional refinement → scene aggregation → full framebuffer
- **SIMD register banks** — single SIMD register, stacked-register tile, multi-tile sub-block, full block

Existing W3-W6 helpers (`SoaVec<T, N>`) handle 1-D batches. They do not address the 2-D hierarchical-block layout the cognitive shader needs. The hand-rolled `splat3d/tile.rs` 16×16-tile binning has the right shape but is bespoke to Gaussian splats and fixed at one tile size.

PR-X3 ships the generic primitive: `CognitiveGrid<T, BR, BC>`. Const-generic over cell type and base block shape. Hierarchical tier iterators. SoA-of-grids macro. Composes with W4 `bulk_apply`.

## The hierarchical block tiers

Reference table — the default 64×64 base block hierarchy:

| Tier | Size | u64 cells | Bytes | Cache fit | Role |
|---|---|---|---|---|---|
| L0 | 8 u64 | 8 | 64 B | L1 cache line | atomic SIMD load (U64x8 / F32x16 reinterpret) |
| **L1** | 64×64 u64 | 4 096 | **32 KB** | L1 half-cache | innermost cell-block — CausalEdge64 mantissa pass |
| **L2** | 256×256 u64 | 65 536 | **512 KB** | L2-fit | regional refinement; 4×4 super-grid of L1 blocks |
| **L3** | 4 096×4 096 u64 | ~16 M | **128 MB** | LLC + RAM | scene aggregation; 16×16 super-grid of L2 blocks |
| **L4** | 16 384×16 384 u64 | ~268 M | **2 GB** | RAM tier | full framebuffer; 4×4 super-grid of L3 blocks |

The hierarchy is exact: every higher tier is a 4×4 / 16×16 / 4×4 sub-grid of the next-finer tier. The 64×64 base divides L2 cleanly into 4×4 super-blocks, L3 into 64×64 super-blocks (4096/64), L4 into 256×256 super-blocks (16384/64).

**Padding rounds storage to BASE block boundary only** — not to L4. A 100×100 grid pads to 128×128 storage (next multiple of 64), not to 16,384². Higher tiers express as iteration patterns over the padded base storage, not as extra storage.

**Tier semantics map to cognitive shader passes**:
- L4: coarse framebuffer pass (palette / depth / alpha at full extent)
- L3: scene-level aggregation (occlusion pre-pass, multi-resolution downsample)
- L2: regional refinement (per-block normalization, neighborhood scan)
- L1: per-cell CausalEdge64 mantissa pass — the irreducible bit-packed cognition unit

## CausalEdge64-as-mantissa

CausalEdge64 is a u64-packed structure (see `CLAUDE.md` and `cognitive-shader-foundation.md`) representing one cognitive edge identity. In a cognitive shader grid:

- Each L1 cell carries ONE u64 CausalEdge64 — the "mantissa" of the cognitive shader cell
- Mantissa = the precision-controlling identity bits (BF16-analogous role: BF16 mantissa is the bits that control numerical precision; CausalEdge64 is the bits that control cognitive identity precision)
- Other cell fields (palette index, depth, alpha) are coarser-grain attributes living in parallel grids
- A cognitive shader pass iterates the L1 tier and operates per-cell on the CausalEdge64 mantissa, using the coarser fields as context

The grid type does **not** know what CausalEdge64 means — `T = u64` is just storage. The semantics live in the consumer (lance-graph-cognitive, p64-bridge, ...).

## Hardware-block × cell-type matrix

The 64×64 base block is the **lowest common multiple** of useful hardware register-bank shapes:

| Hardware op | Natural shape | Cells per op | u64-equivalent shape | Stacked unit |
|---|---|---|---|---|
| AMX BF16 (TDPBF16PS) | 16×16 BF16 tile | 256 BF16 | 16×4 u64 (4 BF16/u64) | one AMX tile = one instruction |
| AMX INT8 (TDPBUSD) | 16×64 INT8 tile | 1024 INT8 | 16×8 u64 | **half-square** by design |
| AVX-512 F32x16 | 1×16 strip | 16 f32 | 1×4 u64 (4 f32/u64) | 2× = 2×16, 4× = 4×16, 8× = 8×16 |
| AVX-512 F64x8 | 1×8 strip | 8 f64 | 1×8 u64 | 8×F64x8 = 8×8 square |
| AVX-512 U64x8 | 1×8 strip | 8 u64 | 1×8 u64 (native) | 8×U64x8 = 8×8 — **CausalEdge64 natural** |
| AVX-512 U8x64 | 1×64 strip | 64 u8 | 1×8 u64 | 1× = 64-byte cache line |
| NEON dotprod int8 (Pi 5) | 1×16 strip | 16 i8 | 1×2 u64 | 4× vertical = 64 cells |
| Scalar fallback | 1 cell | 1 | 1 | n/a |

What 64×64 u64 contains for each hardware:
- **AMX BF16** (each u64 → 4 BF16): block becomes 64×256 BF16 = 4×16 = **64 AMX tiles** per L1 block
- **AVX-512 F32x16** (each u64 → 4 f32): block becomes 64×256 f32 = 64×16 F32x16 = **1024 register-load operations**
- **AVX-512 F64x8** (each u64 → 1 f64): block becomes 64×64 f64 = 64×8 F64x8 = **512 F64x8 operations**
- **AVX-512 U64x8** (native): block is 64×64 u64 = 64×8 U64x8 = **512 native u64 operations**, stacks 8-deep into 8×8 cells = **64 stack-groups per block**
- **NEON dotprod** (each u64 → 8 i8): block becomes 64×512 i8 = 64×32 NEON-int8 registers = **2048 NEON ops**

## Square vs half-square block shapes — when to use each

The 64×64 default is "squarish" — equal row and col dimensions. But several use cases want **half-square**:

| Use case | Shape | Why |
|---|---|---|
| AMX BF16 mantissa pass | 16×16 | AMX BF16 tile is square (TDPBF16PS) |
| AMX INT8 dot pass | **16×64** | TDPBUSD tile is 16 rows × 64 byte-cols (each col is 4 int8 in dot) — half-square in row direction |
| Row-stride-expensive iteration | 32×64 | When iterating column-major over row-major storage, smaller row dimension reduces stride cost |
| Column-stride-expensive iteration | 64×32 | Mirror case |
| Single F32x16 strip | 1×16 | Coarsest possible blocking; no vertical stacking |
| 2×F32x16 vertical stack | 2×16 | Two register loads per cell-block |
| 8×F64x8 stack | 8×8 | Natural F64 8×8 square in 8 register loads |
| CausalEdge64 cell-block (default) | 64×64 | LCM of AMX / AVX-512 / NEON / cache-line — universal compromise |

Consumers pick the shape via const generics on `CognitiveGrid<T, BR, BC>`. The library provides convenience aliases for the common cases (see §"Convenience aliases" below).

## The CognitiveGrid type — full Rust API

```rust
//! src/hpc/cognitive_grid.rs (new module)

use core::marker::PhantomData;

/// 2-D grid of `T` with hierarchical block-aware padding and iteration.
///
/// Padded to a multiple of `BLK_ROW` x `BLK_COL` in storage. Higher
/// tiers express as iteration patterns over the same storage, not as
/// extra padding.
pub struct CognitiveGrid<T, const BLK_ROW: usize = 64, const BLK_COL: usize = 64> {
    rows: usize,         // logical row count
    cols: usize,         // logical col count
    padded_rows: usize,  // = ceil(rows / BLK_ROW) * BLK_ROW
    padded_cols: usize,  // = ceil(cols / BLK_COL) * BLK_COL
    data: Vec<T>,        // row-major, length = padded_rows * padded_cols
}

impl<T: Copy + Default, const BR: usize, const BC: usize> CognitiveGrid<T, BR, BC> {
    /// Create a grid sized to (rows, cols), with storage padded to (BR, BC)
    /// block boundary, all cells initialized to `T::default()`.
    pub fn new(rows: usize, cols: usize) -> Self {
        const { assert!(BR > 0 && BC > 0, "CognitiveGrid: block dims must be > 0") };
        let padded_rows = rows.div_ceil(BR) * BR;
        let padded_cols = cols.div_ceil(BC) * BC;
        let data = vec![T::default(); padded_rows * padded_cols];
        Self { rows, cols, padded_rows, padded_cols, data }
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn padded_rows(&self) -> usize { self.padded_rows }
    pub fn padded_cols(&self) -> usize { self.padded_cols }
    pub fn block_dims() -> (usize, usize) { (BR, BC) }

    /// Logical (row, col) → flat index into storage. Asserts in bounds of
    /// logical extent, NOT padded extent (padding cells are unreachable
    /// via this method — use `block_at_logical` for tier iteration).
    pub fn idx(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.rows && col < self.cols);
        row * self.padded_cols + col
    }

    pub fn get(&self, row: usize, col: usize) -> T { self.data[self.idx(row, col)] }
    pub fn set(&mut self, row: usize, col: usize, v: T) {
        let i = self.idx(row, col);
        self.data[i] = v;
    }

    /// Borrow the full padded storage as a flat slice. Useful for SIMD-stage
    /// closures that walk the storage as a 1-D vector at the BR×BC base tier.
    pub fn as_padded_slice(&self) -> &[T] { &self.data }
    pub fn as_padded_slice_mut(&mut self) -> &mut [T] { &mut self.data }

    /// Iterator over BR×BC base blocks. Yields one `Block` per (block_row, block_col)
    /// pair, row-major order.
    pub fn blocks_base(&self) -> BaseBlockIter<'_, T, BR, BC> {
        BaseBlockIter {
            grid: self,
            block_row: 0,
            block_col: 0,
            n_block_rows: self.padded_rows / BR,
            n_block_cols: self.padded_cols / BC,
        }
    }

    pub fn blocks_base_mut(&mut self) -> BaseBlockIterMut<'_, T, BR, BC> { ... }

    /// Iterator over super-blocks of `N` base-blocks per side (N×N grid of base blocks).
    /// Valid when N divides `padded_rows / BR` and `padded_cols / BC`.
    pub fn blocks_tier<const N: usize>(&self) -> TierBlockIter<'_, T, BR, BC, N> { ... }

    /// In-place `bulk_apply` at the base-block tier. Closure receives a mutable
    /// block window and the (block_row, block_col) coordinates.
    pub fn bulk_apply_base<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut BlockMut<'_, T, BR, BC>, (usize, usize)),
    {
        let n_block_rows = self.padded_rows / BR;
        let n_block_cols = self.padded_cols / BC;
        for br in 0..n_block_rows {
            for bc in 0..n_block_cols {
                let mut block = self.block_mut_at(br, bc);
                f(&mut block, (br, bc));
            }
        }
    }

    pub fn bulk_apply_tier<const N: usize, F>(&mut self, f: F)
    where
        F: FnMut(&mut SuperBlockMut<'_, T, BR, BC, N>, (usize, usize)),
    { ... }
}

/// Read-only base block window.
pub struct Block<'a, T, const BR: usize, const BC: usize> {
    block_row: usize,        // base-block row index (0..n_block_rows)
    block_col: usize,        // base-block col index
    row_origin: usize,       // = block_row * BR
    col_origin: usize,       // = block_col * BC
    padded_cols: usize,      // stride into parent grid's flat storage
    data: &'a [T],           // length = BR * BC, BUT laid out with stride padded_cols
}

impl<'a, T, const BR: usize, const BC: usize> Block<'a, T, BR, BC> {
    pub fn block_row(&self) -> usize { self.block_row }
    pub fn block_col(&self) -> usize { self.block_col }
    pub fn row_origin(&self) -> usize { self.row_origin }
    pub fn col_origin(&self) -> usize { self.col_origin }

    /// Borrow one row of the block as a contiguous &[T] of length BC.
    /// (Block storage uses parent-grid stride, so a row IS contiguous.)
    pub fn row(&self, r: usize) -> &[T] {
        debug_assert!(r < BR);
        let start = r * self.padded_cols;
        &self.data[start..start + BC]
    }

    /// Iterator over the BR rows of the block, each row is &[T] of length BC.
    pub fn rows(&self) -> impl Iterator<Item = &[T]> {
        (0..BR).map(move |r| self.row(r))
    }
}

/// Mutable base block window.
pub struct BlockMut<'a, T, const BR: usize, const BC: usize> { ... }

/// Super-block = N×N grid of base-blocks viewed as a single window.
pub struct SuperBlock<'a, T, const BR: usize, const BC: usize, const N: usize> {
    super_row: usize,        // = base_block_row / N
    super_col: usize,
    row_origin: usize,
    col_origin: usize,
    // ...
}

impl<'a, T, const BR: usize, const BC: usize, const N: usize> SuperBlock<'a, T, BR, BC, N> {
    /// Iterate the N×N base-blocks inside this super-block.
    pub fn base_blocks(&self) -> impl Iterator<Item = Block<'_, T, BR, BC>> { ... }
}

/// Iterators.
pub struct BaseBlockIter<'a, T, const BR: usize, const BC: usize> { ... }
pub struct BaseBlockIterMut<'a, T, const BR: usize, const BC: usize> { ... }
pub struct TierBlockIter<'a, T, const BR: usize, const BC: usize, const N: usize> { ... }
```

## Convenience aliases for the cognitive-shader hierarchy

```rust
/// Default cognitive shader cell-block: 64×64 u64 mantissa grid.
pub type ShaderMantissaGrid = CognitiveGrid<u64, 64, 64>;

/// AMX BF16 tile grid — each cell-block is one AMX BF16 tile (16×16 BF16).
/// Storage type u16 because BF16 lives in u16 carriers.
pub type AmxBf16Grid = CognitiveGrid<u16, 16, 16>;

/// AMX INT8 tile grid — half-square TDPBUSD shape (16×64).
pub type AmxInt8Grid = CognitiveGrid<u8, 16, 64>;

/// F32 vertical-stack-2 strip — 2 F32x16 registers per cell-block.
pub type StripF32Stack2 = CognitiveGrid<f32, 2, 16>;

/// F32 vertical-stack-4 strip — 4 F32x16 registers per cell-block.
pub type StripF32Stack4 = CognitiveGrid<f32, 4, 16>;

/// F64 8×8 square — 8 F64x8 registers per cell-block.
pub type SquareF64Stack8 = CognitiveGrid<f64, 8, 8>;

/// Half-square U64 grid — when row stride is expensive.
pub type HalfSquareU64 = CognitiveGrid<u64, 32, 64>;
```

For the default 64×64 grid, also expose the L2/L3/L4 super-tier aliases:

```rust
impl<T: Copy + Default> CognitiveGrid<T, 64, 64> {
    /// L1 tier: 64×64 base blocks (innermost, ~32 KB).
    pub fn blocks_l1(&self) -> BaseBlockIter<'_, T, 64, 64> { self.blocks_base() }

    /// L2 tier: 256×256 super-blocks (4×4 L1 blocks, ~512 KB).
    pub fn blocks_l2(&self) -> TierBlockIter<'_, T, 64, 64, 4> { self.blocks_tier::<4>() }

    /// L3 tier: 4096×4096 super-blocks (64×64 L1 blocks, ~128 MB).
    pub fn blocks_l3(&self) -> TierBlockIter<'_, T, 64, 64, 64> { self.blocks_tier::<64>() }

    /// L4 tier: 16384×16384 super-blocks (256×256 L1 blocks, ~2 GB framebuffer).
    pub fn blocks_l4(&self) -> TierBlockIter<'_, T, 64, 64, 256> { self.blocks_tier::<256>() }
}
```

## The cognitive_grid_struct! macro

Generates SoA-of-grids: each named field is its own `CognitiveGrid<FieldT, BR, BC>` with shared `rows` / `cols` / `padded_rows` / `padded_cols`.

Usage:

```rust
cognitive_grid_struct! {
    pub struct ShaderCellGrid {
        /// CausalEdge64 mantissa — the truth-bearing identity per cell.
        pub edge: u64,
        pub palette: u8,
        pub depth: u16,    // F16 carrier
        pub alpha: u8,
    }
}
```

Generates:

```rust
pub struct ShaderCellGrid {
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
    pub edge:    CognitiveGrid<u64, 64, 64>,
    pub palette: CognitiveGrid<u8,  64, 64>,
    pub depth:   CognitiveGrid<u16, 64, 64>,
    pub alpha:   CognitiveGrid<u8,  64, 64>,
}

impl ShaderCellGrid {
    pub fn new(rows: usize, cols: usize) -> Self { ... }
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn padded_rows(&self) -> usize { self.padded_rows }
    pub fn padded_cols(&self) -> usize { self.padded_cols }

    /// Iterate all fields' L1 blocks in lockstep (same coordinates).
    /// Yields a tuple of borrows, one per field, at each (block_row, block_col).
    pub fn blocks_l1(&self) -> impl Iterator<Item = ShaderCellL1Block<'_>> { ... }

    /// In-place per-L1-block work with all fields available.
    pub fn bulk_apply_l1<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut ShaderCellL1BlockMut<'_>, (usize, usize)),
    { ... }
}

pub struct ShaderCellL1Block<'a> {
    pub edge:    Block<'a, u64, 64, 64>,
    pub palette: Block<'a, u8,  64, 64>,
    pub depth:   Block<'a, u16, 64, 64>,
    pub alpha:   Block<'a, u8,  64, 64>,
    pub row_origin: usize,
    pub col_origin: usize,
}

pub struct ShaderCellL1BlockMut<'a> { /* mutable variant */ }
```

Macro syntax — accepts:
- `pub` or omitted visibility on struct
- `pub` or omitted visibility per field
- `#[derive(...)]` attributes on the struct (forwarded to the generated struct)
- Optional `#[grid(block = (BR, BC))]` attribute to override the default 64×64 base block — applies to ALL fields uniformly
- Optional `#[grid(field_block = (BR, BC))]` per-field attribute (advanced — sub-grids of different block shapes; **out of scope for v1**, document as future work)

Reserved field names — the macro deliberately does NOT alias around these collisions, so callers must avoid them:
- `new`, `rows`, `cols`, `padded_rows`, `padded_cols`, `blocks_l1`, `blocks_l2`, `blocks_l3`, `blocks_l4`, `bulk_apply_l1`, `bulk_apply_l2`, `bulk_apply_l3`, `bulk_apply_l4`, `default`

## Layering rule recap (where the AMX-vs-AVX-512-vs-NEON dispatch lives)

`CognitiveGrid` is **pure layout**. It contains no `#[target_feature]`, no per-arch imports, no raw intrinsics, no SIMD primitives. The hardware dispatch happens **inside the consumer's closure body**, via `crate::simd::*` calls that route through the existing `simd_caps()` LazyLock.

Example:

```rust
let mut grid: CognitiveGrid<u64, 64, 64> = CognitiveGrid::new(1024, 768);

grid.bulk_apply_base(|block: &mut BlockMut<'_, u64, 64, 64>, (row_origin, col_origin)| {
    // Inside the closure: typed SIMD register-stack primitives from crate::simd.
    // These dispatch AMX / AVX-512 / NEON via simd_caps() LazyLock (existing infra).
    // PR-X5 will add StackedU64x8<N> et al. to crate::simd; for PR-X3 we just
    // demonstrate that the closure boundary is the right place for them.
    for r in 0..64 {
        let row: &mut [u64] = block.row_mut(r);
        // row.len() == 64. Process in stacked-U64x8 chunks of 8 cells each
        // (8 chunks per row, 64 chunks per block).
        // Future: crate::simd::stacked_u64x8_apply::<8>(row, |stack| { ... });
        // Today: scalar loop (the API is forward-compatible).
        for chunk in row.chunks_exact_mut(8) {
            for cell in chunk {
                // CausalEdge64 mantissa pass body
            }
        }
    }
});
```

Two clean layers meet at the closure boundary:
1. **`crate::hpc::cognitive_grid::CognitiveGrid<T, BR, BC>`** — pure layout, generic over cell type and block shape
2. **`crate::simd::{StackedU64x8<N>, StackedF32x16<N>, …, AmxTile<T, R, C>}`** — typed register-stack primitives (PR-X5 ships these; not part of PR-X3)

PR-X3 does **NOT** ship layer 2. PR-X3 ships layer 1 only. The closure-supplied work in PR-X3 tests / doctests uses **scalar** inner loops to demonstrate forward-compatibility.

## Padding strategy — explicit

- Storage is padded to **base block boundary only** (`padded_rows = ceil(rows / BR) * BR`, same for cols).
- Higher tiers (L2/L3/L4) do NOT require additional padding. They iterate the existing padded storage; tier-N iteration is valid only when `padded_rows % (BR * N) == 0` and `padded_cols % (BC * N) == 0`.
- For the default 64×64 base on a 100×100 grid: `padded_rows = padded_cols = 128`. L1 iteration yields 2×2 = 4 base blocks. L2 (N=4) is invalid because `128 % (64*4) = 128 % 256 != 0` → L2 iteration must panic or return Empty (design decision: **panic with a clear message** so the caller picks the right tier for their grid size).
- Padding cells are initialized to `T::default()`. For `u64` CausalEdge64, that's `0` (causally-null edge), which is a safe identity for most cascade ops (XOR with 0 is no-op; popcount of 0 is 0).
- `CognitiveGrid::new(0, 0)` is valid → produces a zero-cell grid with `padded_rows == padded_cols == 0`. Block iterators yield empty.
- `CognitiveGrid::new(rows, 0)` or `(0, cols)` — same: empty grid.

## Tests required (per file, written by workers)

### Unit tests for `CognitiveGrid<T, BR, BC>`

- `new(0, 0)` produces zero-cell grid with empty iterators
- `new(100, 100)` with BR=BC=64 produces 128×128 padded storage, 2×2 L1 blocks
- `new(64, 64)` exactly matches a single L1 block, 1×1 L1 iteration
- `new(256, 256)` with BR=BC=64 produces 4×4 L1 blocks, 1×1 L2 super-block
- `new(100, 100)` with BR=BC=64 — L2 iteration must panic (100 < 256 padded)
- `idx(r, c)` correct for in-range logical (r, c)
- `get` / `set` round-trip
- Padding cells initialized to `T::default()` — verify via `as_padded_slice` at indices past logical extent
- `blocks_base` iterator yields blocks in row-major order with correct `block_row` / `block_col`
- `blocks_base_mut` mutation visible in subsequent `blocks_base` read
- `bulk_apply_base` invokes closure once per block with correct coordinates
- Half-square shape: `CognitiveGrid<u8, 16, 64>` (AMX INT8) — verify padding, iteration
- Single-strip shape: `CognitiveGrid<f32, 1, 16>` (one F32x16 strip) — verify
- 8×8 square: `CognitiveGrid<f64, 8, 8>` — verify
- `blocks_tier::<4>()` on a 256×256 grid yields one super-block; `blocks_tier::<4>()` on 128×128 panics (the assertion explained above)
- Const-generic compile-time assertion: `CognitiveGrid::<u64, 0, 64>::new(...)` fails to compile (BR > 0 const assert)

### Doc-tests

Every public fn / method gets a working `# Example` doctest. Module-level doctest demonstrates the canonical compose pattern: build a `ShaderMantissaGrid::new(1024, 768)`, iterate L1 blocks, mutate one block's CausalEdge64 mantissa pattern, verify.

### Unit tests for `cognitive_grid_struct!` macro

- 2-field, 3-field, 4-field struct generation
- `pub` and private field visibility per field
- `#[derive(Clone)]` passthrough on the macro input
- Override `#[grid(block = (16, 16))]` produces AMX-shaped sub-grids
- `bulk_apply_l1` closure receives all fields in lockstep with same `(block_row, block_col)`
- New struct's `rows()` / `cols()` / `padded_rows()` / `padded_cols()` are consistent across all fields

### Integration test with W4 `bulk_apply`

A single test composing W4 `bulk_apply` over the L1-block iterator's output. Demonstrates that PR-X3 composes cleanly with the W3-W6 primitives without re-implementing chunking.

## Out of scope — explicitly NOT in PR-X3

These are NOT part of PR-X3 (each becomes its own future PR):

1. **SIMD register-bank stack types** (`StackedU64x8<N>`, `StackedF64x8<N>`, `StackedF32x16<N>`, `AmxTile<T, R, C>`) → PR-X5
2. **Typed distance bulk fns** (palette-256, hamming popcount early-exit, Base17 L1, BF16 mantissa direct-transform) → W7, bench-gated
3. **CausalEdge64 mantissa cell kernel** (the actual L1 pass body) → W7
4. **splat3d adoption** (refactor `splat3d/tile.rs` onto `CognitiveGrid`) → PR-X4 (depends on this PR)
5. **Per-field `#[grid(field_block = ...)]` heterogeneous block shapes** → document as future work; not in v1
6. **Sparse storage variant** (`HashMap<(u16,u16), CognitiveGrid<T>>` for sparse Gaussian distributions) → out of scope; if needed, separate PR
7. **Cascade orchestrator** (`cascade_topk_per_tile` composing L1→L2→L3 typed metrics over the grid) → W8, depends on W7

## Distance-typing guardrail

**`CognitiveGrid` is layout-only and explicitly does NOT bake in any distance metric.** Per the binding rule in `.claude/knowledge/cognitive-distance-typing.md`:
- No `fn bulk_distance<T>` umbrella
- No `enum DistanceMetric { Palette256, Hamming, Base17, … }`
- No `Box<dyn Distance>` trait object
- No generic `fn distance<T>(a: &T, b: &T) -> f32`

The grid type holds `T`. It doesn't know what `T` means. The semantics live in:
- Consumer closures passed to `bulk_apply_l{1,2,3,4}` (W1a contract — closure absorbs domain semantics)
- Typed primitives in `crate::simd::*` that the closures call (PR-X5)
- Typed distance bulk fns in `crate::hpc::cognitive::*` (W7)

Workers MUST NOT add any distance-aware API to this PR. Module headers reference `cognitive-distance-typing.md` and warn against extension toward distance.

## Worker decomposition (SEQUENTIAL — the binding protocol)

**Protocol:** 5–10 Sonnet workers + 1 Opus coordinator (this session). Workers run **sequentially**, one at a time. Each worker's output is reviewed / verified before the next worker spawns. This matches the binding protocol established 2026-05-18:

> sequentially 5-10 sonnet agents + 1 Koordinator
> plan → review → correct → sprint → review code → fix P0 → commit → repeat

### The seven-phase agent sequence for PR-X3

Each agent runs **sequentially**, with coordinator review between phases. All workers use **Sonnet** (not Opus — coordinator is Opus). All workers operate in isolated worktrees via `isolation: "worktree"`.

| # | Phase | Agent role | Scope | Coordinator action between this and next |
|---|---|---|---|---|
| 1 | **plan** | (this doc) | written by coordinator | N/A — already done |
| 2 | **review** | plan-review savant | audits this design, returns READY-WITH-DOC-FIXES or NEEDS-FIX with P0/P1 list | apply patches to design doc; commit v2 |
| 3 | **correct** | (coordinator) | applies savant's P0/P1 to design doc | commit doc v2; ready for sprint |
| 4 | **sprint worker A** | CognitiveGrid core | `src/hpc/cognitive_grid/mod.rs` (new): `CognitiveGrid<T, BR, BC>`, `Block` / `BlockMut`, `SuperBlock` / `SuperBlockMut`, `BaseBlockIter` / `BaseBlockIterMut`, `TierBlockIter`, `bulk_apply_base`, `bulk_apply_tier`, convenience aliases (ShaderMantissaGrid, AmxBf16Grid, …), L1/L2/L3/L4 alias impls on 64×64 default base. Inline unit tests for the core type. Cargo check/test/fmt/clippy must pass. Single commit. | cherry-pick onto coordinator branch; verify green |
| 5 | **sprint worker B** | cognitive_grid_struct! macro | `src/hpc/cognitive_grid/grid_struct_macro.rs` (new submodule, `pub mod grid_struct_macro;` inside `cognitive_grid/mod.rs`): the `cognitive_grid_struct!` macro definition, generated-struct iterator types (`ShaderCellL1Block`-style helpers), all macro tests. Cargo check/test/fmt/clippy must pass. Single commit. **Depends on Worker A's `CognitiveGrid` API being on the branch.** | cherry-pick onto coordinator branch; verify green |
| 6 | **review code** | codex P0 auditor | audits combined diff (Worker A + Worker B) for: zero `#[target_feature]`, zero `use crate::simd_avx{512,2}` / `simd_neon` / `simd_wasm` imports, zero `cfg(target_feature = …)` gates, zero raw `_mm*_*` / `vld*_*` / `_pdep_*` intrinsics, zero distance-aware API surface, all public fns have working `///` doc-examples, tests cover all spec'd cases (including const-generic compile-fail cases via `compile_fail` doctests where feasible). Verdict: READY-FOR-PR or NEEDS-FIX with P0 list. | apply P0 fixes (if any); commit |
| 7 | **fix P0** | (coordinator) | applies codex P0 patches; commit | push; open PR |
| 8 | **review pr (P2)** | P2 codex savant | reviews the open PR for API ergonomics, naming drift, doc-prose quality, distance-typing visibility on the public PR, future-proofing for the SIMD swap, CI signal. Verdict: SHIP-AS-IS / SHIP-WITH-FOLLOWUPS / RECONSIDER. | apply highest-leverage pre-merge tightening; push; merge ladder |
| 9 | **repeat / next sprint** | (coordinator) | if P2 savant recommends follow-ups too heavy for this PR, queue PR-X3.1; otherwise advance to PR-X4 / PR-X5 / W7 |

### Workers cap at 5–10 — when to add more

If the §"Sprint worker" phase becomes too coarse (e.g., Worker A's scope at >800 LOC overruns the agent's effective single-pass attention), split:

- **Worker A1**: `CognitiveGrid<T, BR, BC>` struct + `Block` / `BlockMut` types + `new` / `idx` / `get` / `set` / `as_padded_slice*`. Single commit.
- **Worker A2**: `BaseBlockIter` / `BaseBlockIterMut` + `blocks_base` / `blocks_base_mut`. Single commit. Depends on A1.
- **Worker A3**: `SuperBlock` / `SuperBlockMut` + `TierBlockIter` + `blocks_tier::<N>`. Single commit. Depends on A2.
- **Worker A4**: `bulk_apply_base` + `bulk_apply_tier`. Single commit. Depends on A3.
- **Worker A5**: convenience aliases (ShaderMantissaGrid, AmxBf16Grid, AmxInt8Grid, StripF32Stack2, StripF32Stack4, SquareF64Stack8, HalfSquareU64) + L1/L2/L3/L4 alias impls on 64×64. Single commit.
- **Worker A6**: full unit test coverage + doctests. Single commit. Depends on A5.

Then Worker B (the macro) runs after A6. Then codex P0 audit. Total: 7 sprint workers + 1 audit + 1 P2 review = **9 sequential Sonnet agents + 1 Opus coordinator**.

For PR-X3 the recommended cut is **5 sequential workers** (one composite A handling 1+2+3+4 since these are tightly coupled type definitions, one A handling 5+6 for aliases+tests, one B for the macro, one codex audit, one P2 savant). The coordinator escalates to the 9-worker split only if a worker's first commit fails the green-check.

### Worker isolation rule

Every Sonnet sprint worker runs with `isolation: "worktree"` (NOT in the coordinator's main tree). Workers commit to their own branch; coordinator cherry-picks. This prevents the worker-B-bleeding-into-W2-branch incident from the W3-W6 sprint.

### Sequential vs parallel — why sequential

The earlier W3-W6 sprint ran Worker A and Worker B in **parallel**. That worked for W3-W6 because A (`hpc/soa.rs`) and B (`hpc/bulk.rs`) were independent files. For PR-X3, Worker B (the macro) emits code that depends on Worker A's `Block` / `BlockMut` / `bulk_apply_l1` API. Sequential ordering eliminates the integration risk of "Worker B writes against a mock API; Worker A ships a slightly different API; integration breaks."

The user's binding protocol clarifies: **sequential is the default; parallel is only when files are truly independent**.

## What workers commit per file

1. Implement the spec above exactly. No deviation in API.
2. Add inline tests covering the cases listed under §"Tests required" for the file.
3. Add the `pub mod cognitive_grid;` registration in `src/hpc/mod.rs` (Worker A).
4. Run from worktree root:
   - `cargo check -p ndarray --no-default-features --features std`
   - `cargo test -p ndarray --lib --no-default-features --features std hpc::cognitive_grid`
   - `cargo test --doc -p ndarray --no-default-features --features std hpc::cognitive_grid`
   - `cargo fmt --all -- --check`
   - `cargo clippy -p ndarray --no-default-features --features std -- -D warnings`
   - All green before commit.
5. Commit message format:
   - Worker A: `feat(hpc/cognitive_grid): add CognitiveGrid<T, BR, BC> hierarchical-block layout (PR-X3 core)`
   - Worker B: `feat(hpc/cognitive_grid): add cognitive_grid_struct! macro for SoA-of-grids (PR-X3 macro)`

## Verification commands (run from /home/user/ndarray)

Identical to W3-W6 protocol:

```bash
cargo check -p ndarray --no-default-features --features std
cargo test -p ndarray --lib --no-default-features --features std hpc::cognitive_grid
cargo test --doc -p ndarray --no-default-features --features std hpc::cognitive_grid
cargo fmt --all -- --check
cargo clippy -p ndarray --no-default-features --features std -- -D warnings
```

All five must pass green.

## Sprint protocol (the established multi-agent pattern)

1. ✅ **Design v1** committed (this doc)
2. ⬜ **Plan-review savant** spawned — audits this design, returns READY-WITH-DOC-FIXES or NEEDS-FIX with P0/P1 findings
3. ⬜ **Design v2** absorbs all P0/P1 patches + open-question rulings
4. ⬜ **Two workers in parallel** in isolated worktrees:
   - Worker A: `src/hpc/cognitive_grid.rs` (core)
   - Worker B: `src/hpc/cognitive_grid/macro.rs` (macro)
5. ⬜ Worker commits cherry-picked onto branch
6. ⬜ **Codex P0 audit** spawned on combined diff
7. ⬜ Fix any P0s
8. ⬜ Open PR
9. ⬜ **P2 codex savant** review on the open PR (ergonomics / drift / naming)
10. ⬜ Same-day follow-up PR for any pre-merge tightenings the P2 savant recommends

## Cross-references

- `.claude/knowledge/w3-w6-soa-aos-design.md` — the SoA/AoS foundation this builds on; same protocol shape, same layering rule
- `.claude/knowledge/cognitive-shader-foundation.md` — ndarray's role in the 7-layer cognitive shader stack; identifies the gaps PR-X3 fills
- `.claude/knowledge/cognitive-distance-typing.md` — the binding rule that PR-X3 must respect (no umbrella distance, no roundtrips, typed metrics only)
- `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a layering rule (user code → `crate::simd` → `simd_{type}.rs`); PR-X3 is user-level code
- `.claude/knowledge/w3-w6-codex-audit.md` — example codex P0 audit output for protocol reference
- `.claude/knowledge/w3-w6-p2-savant-review.md` — example P2 savant review output for protocol reference
- `src/hpc/soa.rs` — W3-W6 SoaVec + soa_struct! (the 1-D primitive PR-X3 extends to 2-D)
- `src/hpc/bulk.rs` — W4 bulk_apply / bulk_scan (the chunked-traversal primitive PR-X3 composes with at the tier level)
- `src/hpc/splat3d/tile.rs` — the bespoke 16×16-tile binning that PR-X4 (future) refactors onto `CognitiveGrid`

## Open questions (for the plan-review savant)

1. **Naming**: `CognitiveGrid` vs `BlockedGrid` vs `TiledGrid` vs `HierarchicalGrid`. The "cognitive" prefix in the type name leans into the consumer use case but may overstate the type's generality (it's actually a generic 2-D blocked grid usable anywhere). Alternative: `BlockedGrid<T, BR, BC>` with a separate type alias `ShaderMantissaGrid = BlockedGrid<u64, 64, 64>`.

2. **`bulk_apply_tier` const-generic ergonomics**: invoking `grid.bulk_apply_tier::<4, _>(...)` requires the caller to pick `N` explicitly. Convenience aliases (`bulk_apply_l2`) bury the `N` choice. Worth offering both? Or pick one?

3. **Block lifetime variance**: should `Block<'a, T, BR, BC>` carry a `PhantomData<&'a mut T>` for mutability tracking, or rely on `BlockMut` as a separate type? (Decision in spec: separate `Block` / `BlockMut` — but verify this is idiomatic Rust 2024.)

4. **`#[grid(field_block = ...)]` per-field heterogeneous block shapes** — out of scope for v1, but is the macro structurally compatible with adding it later, or does v1 lock us out?

5. **Padding init value**: the spec says `T::default()`. For CausalEdge64 (u64) that's 0 = causally-null. For floats, that's 0.0. Should we offer `CognitiveGrid::new_with_pad(rows, cols, pad_value: T)` to let callers pick a non-default init for padding cells?

6. **`as_padded_slice` / `as_padded_slice_mut` exposure**: exposing the flat padded storage lets consumers do "treat the grid as a 1-D flat batch" — useful for SoA-staging via `aos_to_soa` over the entire grid. But it also exposes the padding cells. Is this a footgun, or a feature? (Lean: feature, document clearly.)

7. **L4 alias on non-64×64 grids**: should we offer L1/L2/L3/L4 aliases on grids with non-default base block? The 4×4 / 16×16 / 4×4 hierarchy is specific to 64×64. For a 16×16 AMX grid, the natural higher tiers might be 4×4 (64×64) / 16×16 (256×256) / 64×64 (1024×1024) / 256×256 (4096×4096) — different tier semantics. Decision: leave the L1-L4 aliases ON the 64×64 base only. AMX grids get their own per-shape aliases if needed.

## Done criteria

PR-X3 is done when:
- All worker spec items implemented
- Codex P0 audit passes with 0 P0
- `cargo check / test --lib / test --doc / fmt / clippy` all green
- Layering rule verified (zero per-arch imports / target_feature / raw intrinsics in the new files)
- Distance-typing guardrail verified (zero umbrella-distance API surface)
- Module headers reference `cognitive-distance-typing.md` and warn against distance extension
- P2 savant review delivers SHIP verdict (with optional same-day follow-up PR for the highest-leverage P2)

## Token-reset safety notes (for fresh sessions)

This doc was written when the conversation was at 96% context. If you're picking up after a token reset:

1. Read this entire doc first.
2. Check `.claude/knowledge/` for any newer planning docs.
3. Check `git log --oneline -10` on this branch and on `master` to see what shipped.
4. The W2/W3-W6 multi-agent sprint protocol is the canonical pattern — see `.claude/knowledge/w3-w6-soa-aos-design.md` §"Sprint protocol" for the same shape.
5. Open PRs to track: #155 (sigmoid orphan rescue, may be merged), #157 (P2 savant follow-up, may be merged), this branch's PR (not yet open).
6. PR-X1 and PR-X2 are designed in conversation but not yet specced to disk. If you need them, see `cognitive-shader-foundation.md` §"Current Gaps" and the savant A1/A4 P2 findings in `w3-w6-p2-savant-review.md`.
7. The hardware-block × cell-type matrix in §"Hardware-block × cell-type matrix" is the canonical reference for which block shape fits which SIMD tier. Memorize it before proposing API changes.
