# PR-X3 — BlockedGrid: hierarchical block layout for cognitive shader + spatial splat BLAS

> READ BY: all ndarray agents that touch the cognitive shader stack
> (savant-architect, l3-strategist, cascade-architect,
> cognitive-architect, arm-neon-specialist, sentinel-qa, product-engineer,
> truth-architect, vector-synthesis, splat3d-architect).
>
> **Design doc revision v2** — incorporates plan-review savant verdict
> (`.claude/knowledge/pr-x3-plan-review.md`, READY-WITH-DOC-FIXES, 2 P0 + 7 P1 + 4 P2).
> P0 patches applied: A1 (map_* / bulk_apply_* split) + A2 (macro emits map_l*).
> Q1–Q7 ruled (see §"Resolved questions" — was §"Open questions" in v1).
>
> Parallel docs:
> - `.claude/knowledge/w3-w6-soa-aos-design.md` — the SoA/AoS foundation this builds on
> - `.claude/knowledge/cognitive-shader-foundation.md` — ndarray's role in the 7-layer stack
> - `.claude/knowledge/cognitive-distance-typing.md` — the no-umbrella distance rule
> - `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a layering rule (user code → crate::simd → simd_{type}.rs)
> - `.claude/rules/data-flow.md` — the binding rule that A1/A2 P0 fixes respect ("No `&mut self` during computation. Ever.")

## Context for a fresh session

If you arrive here without conversational context (token reset, new session, handover), here is the minimum you need to know:

1. **W3-W6 shipped** (PR #156, merged 2026-05-18). It added `SoaVec<T, N>`, `soa_struct!` macro, `aos_to_soa<T, N, F: Fn(&T) -> [f32; N]>`, `soa_to_aos<T, N, F>`, `bulk_apply`, `bulk_scan` to `src/hpc/{soa,bulk}.rs`. All scalar, no `#[target_feature]`, no per-arch imports, no distance baked in.
2. **PR #157 is open** (P2 savant follow-up): adds f32-only-scope docs + `hpc::soa`-vs-`simd_ops` rationale + ungated integration test.
3. **PR-X1 / PR-X2 are designed but not started** — see `cognitive-shader-foundation.md` §"Current Gaps":
   - PR-X1: `MultiLaneColumn`, `Fingerprint::as_u8x64`, `array_window`, `simd::*` re-exports
   - PR-X2: `#[soa(pad_to_lanes=N)]` macro attribute + generalized `aos_to_soa<T, U, N>`
4. **PR-X3 (this doc)**: `BlockedGrid<T, BR, BC>` hierarchical block-padded grid + `blocked_grid_struct!` SoA-of-grids macro. **Layout only**. Scalar. No SIMD primitives. Forward-compatible with the per-arch SIMD swap that lands in PR-X5 / W7.
5. **PR-X5 (planned)**: Typed SIMD register-bank stacks (`StackedU64x8<N>`, `StackedF32x16<N>`, `StackedF64x8<N>`, `AmxTile<T, R, C>`) in `crate::simd::*`. Per-arch LazyLock dispatch.
6. **W7 (deferred, bench-gated)**: Typed cognitive distance bulk fns (palette-256, hamming popcount early-exit, Base17 L1, BF16 mantissa direct-transform) + the actual CausalEdge64 mantissa cell kernel.

**This PR is PR-X3 only.** PR-X5 and W7 are explicit non-goals here.

## Why this exists

The cognitive shader stack (per `cognitive-shader-foundation.md`) operates on 2-D grids at multiple block hierarchies that correspond simultaneously to:

- **Cache hierarchy** — L1 (~32 KB), L2 (~512 KB), L3/LLC (~128 MB), RAM (~2 GB)
- **Resolution pyramid** — irreducible cell block → regional refinement → scene aggregation → full framebuffer
- **SIMD register banks** — single SIMD register, stacked-register tile, multi-tile sub-block, full block

Existing W3-W6 helpers (`SoaVec<T, N>`) handle 1-D batches. They do not address the 2-D hierarchical-block layout the cognitive shader needs. The hand-rolled `splat3d/tile.rs` 16×16-tile binning has the right shape but is bespoke to Gaussian splats and fixed at one tile size.

PR-X3 ships the generic primitive: `BlockedGrid<T, BR, BC>`. Const-generic over cell type and base block shape. Hierarchical tier iterators. SoA-of-grids macro. Composes with W4 `bulk_apply`.

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

**Cache-hierarchy convention.** L1 = innermost (32 KB, fastest); L4 = framebuffer-scale (2 GB, RAM tier). This matches CPU cache vocabulary — DO NOT invert.

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
| AMX INT8 dot-product | 16×64 | TDPBUSD takes 16×64 byte-tile — half-square in u64 terms |
| F32x16 single-strip kernel | 1×16 | One AVX-512 register |
| F32x16 vertical stack-2 | 2×16 | Cache-line aligned pair |
| F64x8 8×8 GEMM kernel | 8×8 | Square BLAS micro-kernel for matrix multiplication |
| U8x64 cache-line scan | 1×64 | Exactly one cache line (64 bytes), one AVX-512 register |
| U64x8 vertical stack-8 | 8×8 | 8 stacked U64x8 registers, square AMX-analogous shape |

`BlockedGrid<T, BR, BC>` is const-generic over both dimensions. The 64×64 default is just one shape. Convenience type aliases (below) pin the common shapes.

## The `BlockedGrid` type — full API

`crate::hpc::blocked_grid::BlockedGrid<T, BR, BC>` — generic 2-D block-padded grid.

```rust
/// Generic block-padded 2-D grid. Storage is row-major, padded to
/// (BR, BC) base-block boundaries on both axes. Higher tiers (L2 / L3 / L4
/// for the 64×64 default) are expressed as iteration patterns, not as
/// extra padding.
pub struct BlockedGrid<T, const BLK_ROW: usize = 64, const BLK_COL: usize = 64> {
    rows: usize,         // logical row count
    cols: usize,         // logical col count
    padded_rows: usize,  // = ceil(rows / BLK_ROW) * BLK_ROW
    padded_cols: usize,  // = ceil(cols / BLK_COL) * BLK_COL
    data: Vec<T>,        // row-major, length = padded_rows * padded_cols
}

// === Constructors ===

impl<T: Copy, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Create a grid sized to (rows, cols), with storage padded to (BR, BC)
    /// block boundary, all cells initialized to `pad_value`. The `T: Copy`
    /// bound is the only constraint on `T` — no `Default` required.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u64, 64, 64>::new_with_pad(100, 100, 0xDEAD_BEEF);
    /// assert_eq!(g.padded_rows(), 128);
    /// ```
    pub fn new_with_pad(rows: usize, cols: usize, pad_value: T) -> Self {
        const { assert!(BR > 0 && BC > 0, "BlockedGrid: block dims must be > 0") };
        let padded_rows = rows.div_ceil(BR) * BR;
        let padded_cols = cols.div_ceil(BC) * BC;
        let data = vec![pad_value; padded_rows * padded_cols];
        Self { rows, cols, padded_rows, padded_cols, data }
    }
}

impl<T: Copy + Default, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Create a grid initialized to `T::default()`. Convenience wrapper
    /// over [`new_with_pad`] for types where the default value is the
    /// natural padding fill (e.g. `u64` → 0 = causally-null edge).
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::new_with_pad(rows, cols, T::default())
    }
}

// === Accessors ===

impl<T, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
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
}

impl<T: Copy, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    pub fn get(&self, row: usize, col: usize) -> T { self.data[self.idx(row, col)] }
    pub fn set(&mut self, row: usize, col: usize, v: T) {
        let i = self.idx(row, col);
        self.data[i] = v;
    }

    /// Borrow the full padded storage as a flat slice. Useful for SIMD-stage
    /// closures that walk the storage as a 1-D vector at the BR×BC base tier.
    ///
    /// # Footgun
    /// The returned slice **includes padding cells** at the right and bottom
    /// of the logical extent. The slice length is `padded_rows() * padded_cols()`,
    /// not `rows() * cols()`. Cells at indices that map outside the logical
    /// (rows, cols) box are padding cells (default-initialized via [`new`] or
    /// set explicitly via [`new_with_pad`]).
    ///
    /// To compute a logical-cell flat index correctly, use [`idx`]:
    /// `as_padded_slice()[grid.idx(r, c)]`. NEVER index the slice as
    /// `r * cols() + c` — that ignores stride and reads the wrong cell.
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::blocked_grid::BlockedGrid;
    /// let g = BlockedGrid::<u8, 64, 64>::new(100, 100);
    /// assert_eq!(g.as_padded_slice().len(), 128 * 128); // padded extent
    /// ```
    pub fn as_padded_slice(&self) -> &[T] { &self.data }

    /// Mutable variant — see [`as_padded_slice`] footgun note.
    pub fn as_padded_slice_mut(&mut self) -> &mut [T] { &mut self.data }
}

// === Iterators (read-only) ===

impl<T, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Iterator over BR×BC base blocks. Yields one `GridBlock` per
    /// (block_row, block_col) pair, row-major order.
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
    /// Valid when N divides `padded_rows / BR` and `padded_cols / BC`; panics otherwise.
    pub fn blocks_tier<const N: usize>(&self) -> TierBlockIter<'_, T, BR, BC, N> { ... }
}

// === Compute paths (PRIMARY): map_* — immutable self, returns a new grid ===
//
// These are the data-flow-correct compute paths per `.claude/rules/data-flow.md`
// Rule #3: "No `&mut self` during computation. Ever." The closure receives a
// read-only view of the input block and a mutable view of the OUTPUT block
// (in the freshly-allocated result grid), so the input is never mutated.

impl<T: Copy, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// Map a closure over every base-block, producing a new grid with element
    /// type `U`. The closure reads from the input block and writes to the
    /// corresponding output block; the input grid is not mutated.
    ///
    /// # Data-flow rule
    /// This is the PRIMARY compute path. It satisfies the
    /// `.claude/rules/data-flow.md` Rule #3 invariant. For in-place
    /// write-back (e.g., scratch-buffer pipelines) see [`bulk_apply_base`].
    pub fn map_base<U: Copy + Default, F>(&self, mut f: F) -> BlockedGrid<U, BR, BC>
    where
        F: FnMut(&GridBlock<'_, T, BR, BC>, &mut GridBlockMut<'_, U, BR, BC>),
    {
        let mut out = BlockedGrid::<U, BR, BC>::new(self.rows, self.cols);
        let n_block_rows = self.padded_rows / BR;
        let n_block_cols = self.padded_cols / BC;
        for br in 0..n_block_rows {
            for bc in 0..n_block_cols {
                let inp = self.block_at(br, bc);
                let mut outp = out.block_mut_at(br, bc);
                f(&inp, &mut outp);
            }
        }
        out
    }

    /// Map over super-blocks of `N` base-blocks per side. Same data-flow
    /// invariant as [`map_base`].
    pub fn map_tier<U: Copy + Default, const N: usize, F>(&self, mut f: F) -> BlockedGrid<U, BR, BC>
    where
        F: FnMut(&GridSuperBlock<'_, T, BR, BC, N>, &mut GridSuperBlockMut<'_, U, BR, BC, N>),
    { ... }
}

// === Write-back paths (SECONDARY): bulk_apply_* — &mut self, gated mutation ===

impl<T, const BR: usize, const BC: usize> BlockedGrid<T, BR, BC> {
    /// In-place per-base-block work. Closure receives a mutable block window
    /// and the (block_row, block_col) coordinates.
    ///
    /// # Data-flow rule
    ///
    /// This is the WRITE-BACK variant per `.claude/rules/data-flow.md` Rule #3
    /// ("No `&mut self` during computation. Ever."). The closure performs
    /// gated write-back operations ONLY (single-target XOR, BUNDLE majority
    /// merge, or scratch-buffer fill). For COMPUTE paths — anything that
    /// reads and derives a new value — use [`map_base`] instead, which
    /// returns a fresh grid.
    ///
    /// Workers MUST NOT place CausalEdge64 mantissa-pass logic, cascade
    /// reasoning, or any other compute kernel inside this closure. Sentinel
    /// reviews will flag violations.
    pub fn bulk_apply_base<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut GridBlockMut<'_, T, BR, BC>, (usize, usize)),
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

    /// Write-back variant at tier `N`. See [`bulk_apply_base`] for the
    /// data-flow rule note.
    pub fn bulk_apply_tier<const N: usize, F>(&mut self, f: F)
    where
        F: FnMut(&mut GridSuperBlockMut<'_, T, BR, BC, N>, (usize, usize)),
    { ... }
}

// === Block view types ===

/// Read-only base block window. Carries an explicit `PhantomData<&'a T>` for
/// lifetime variance (idiomatic Rust 2024).
pub struct GridBlock<'a, T, const BR: usize, const BC: usize> {
    block_row: usize,        // base-block row index (0..n_block_rows)
    block_col: usize,        // base-block col index
    row_origin: usize,       // = block_row * BR
    col_origin: usize,       // = block_col * BC
    padded_cols: usize,      // stride into parent grid's flat storage
    data: &'a [T],           // length = BR * BC, BUT laid out with stride padded_cols
}

impl<'a, T, const BR: usize, const BC: usize> GridBlock<'a, T, BR, BC> {
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

/// Mutable base block window. Carries `PhantomData<&'a mut T>` for variance.
pub struct GridBlockMut<'a, T, const BR: usize, const BC: usize> { ... }

/// Super-block = N×N grid of base-blocks viewed as a single window.
/// Read-only variant; mutable is `GridSuperBlockMut`.
pub struct GridSuperBlock<'a, T, const BR: usize, const BC: usize, const N: usize> {
    super_row: usize,        // = base_block_row / N
    super_col: usize,
    row_origin: usize,
    col_origin: usize,
    // ...
}

impl<'a, T, const BR: usize, const BC: usize, const N: usize> GridSuperBlock<'a, T, BR, BC, N> {
    /// Iterate the N×N base-blocks inside this super-block.
    pub fn base_blocks(&self) -> impl Iterator<Item = GridBlock<'_, T, BR, BC>> { ... }
}

/// Iterators.
pub struct BaseBlockIter<'a, T, const BR: usize, const BC: usize> { ... }
pub struct BaseBlockIterMut<'a, T, const BR: usize, const BC: usize> { ... }
pub struct TierBlockIter<'a, T, const BR: usize, const BC: usize, const N: usize> { ... }
```

## Convenience aliases for the cognitive-shader hierarchy

```rust
/// Default cognitive shader cell-block: 64×64 u64 mantissa grid.
pub type ShaderMantissaGrid = BlockedGrid<u64, 64, 64>;

/// AMX BF16 tile grid — each cell-block is one AMX BF16 tile (16×16 BF16).
/// Storage type u16 because BF16 lives in u16 carriers.
pub type AmxBf16Grid = BlockedGrid<u16, 16, 16>;

/// AMX INT8 tile grid — half-square TDPBUSD shape (16×64).
pub type AmxInt8Grid = BlockedGrid<u8, 16, 64>;

/// F32 vertical-stack-2 strip — 2 F32x16 registers per cell-block.
pub type StripF32Stack2 = BlockedGrid<f32, 2, 16>;

/// F32 vertical-stack-4 strip — 4 F32x16 registers per cell-block.
pub type StripF32Stack4 = BlockedGrid<f32, 4, 16>;

/// F64 8×8 square — 8 F64x8 registers per cell-block.
pub type SquareF64Stack8 = BlockedGrid<f64, 8, 8>;

/// Half-square U64 grid — when row stride is expensive.
pub type HalfSquareU64 = BlockedGrid<u64, 32, 64>;
```

For the default 64×64 grid, also expose the L2/L3/L4 super-tier aliases. These aliases live ONLY on `BlockedGrid<T, 64, 64>` (Q7 ruling — non-64×64 grids use raw `blocks_tier::<N>` / `map_tier::<N>` / `bulk_apply_tier::<N>`):

```rust
/// Cache-hierarchy convention: L1 = innermost (32 KB), L4 = framebuffer-scale (2 GB).
impl<T: Copy + Default> BlockedGrid<T, 64, 64> {
    // --- Read-only tier iterators ---

    /// L1 tier: 64×64 base blocks (innermost, ~32 KB).
    pub fn blocks_l1(&self) -> BaseBlockIter<'_, T, 64, 64> { self.blocks_base() }

    /// L2 tier: 256×256 super-blocks (4×4 L1 blocks, ~512 KB).
    pub fn blocks_l2(&self) -> TierBlockIter<'_, T, 64, 64, 4> { self.blocks_tier::<4>() }

    /// L3 tier: 4096×4096 super-blocks (64×64 L1 blocks, ~128 MB).
    pub fn blocks_l3(&self) -> TierBlockIter<'_, T, 64, 64, 64> { self.blocks_tier::<64>() }

    /// L4 tier: 16384×16384 super-blocks (256×256 L1 blocks, ~2 GB framebuffer).
    pub fn blocks_l4(&self) -> TierBlockIter<'_, T, 64, 64, 256> { self.blocks_tier::<256>() }

    // --- Compute paths (PRIMARY) — see map_base data-flow note ---

    pub fn map_l1<U: Copy + Default, F>(&self, f: F) -> BlockedGrid<U, 64, 64>
    where F: FnMut(&GridBlock<'_, T, 64, 64>, &mut GridBlockMut<'_, U, 64, 64>) { self.map_base(f) }

    pub fn map_l2<U: Copy + Default, F>(&self, f: F) -> BlockedGrid<U, 64, 64>
    where F: FnMut(&GridSuperBlock<'_, T, 64, 64, 4>, &mut GridSuperBlockMut<'_, U, 64, 64, 4>) { self.map_tier::<U, 4, _>(f) }

    pub fn map_l3<U: Copy + Default, F>(&self, f: F) -> BlockedGrid<U, 64, 64>
    where F: FnMut(&GridSuperBlock<'_, T, 64, 64, 64>, &mut GridSuperBlockMut<'_, U, 64, 64, 64>) { self.map_tier::<U, 64, _>(f) }

    pub fn map_l4<U: Copy + Default, F>(&self, f: F) -> BlockedGrid<U, 64, 64>
    where F: FnMut(&GridSuperBlock<'_, T, 64, 64, 256>, &mut GridSuperBlockMut<'_, U, 64, 64, 256>) { self.map_tier::<U, 256, _>(f) }

    // --- Write-back paths (SECONDARY) — see bulk_apply_base data-flow note ---

    pub fn bulk_apply_l1<F>(&mut self, f: F)
    where F: FnMut(&mut GridBlockMut<'_, T, 64, 64>, (usize, usize)) { self.bulk_apply_base(f) }

    pub fn bulk_apply_l2<F>(&mut self, f: F)
    where F: FnMut(&mut GridSuperBlockMut<'_, T, 64, 64, 4>, (usize, usize)) { self.bulk_apply_tier::<4, _>(f) }

    pub fn bulk_apply_l3<F>(&mut self, f: F)
    where F: FnMut(&mut GridSuperBlockMut<'_, T, 64, 64, 64>, (usize, usize)) { self.bulk_apply_tier::<64, _>(f) }

    pub fn bulk_apply_l4<F>(&mut self, f: F)
    where F: FnMut(&mut GridSuperBlockMut<'_, T, 64, 64, 256>, (usize, usize)) { self.bulk_apply_tier::<256, _>(f) }
}
```

## The `blocked_grid_struct!` macro

Generates SoA-of-grids: each named field is its own `BlockedGrid<FieldT, BR, BC>` with shared `rows` / `cols` / `padded_rows` / `padded_cols`.

Usage:

```rust
blocked_grid_struct! {
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
    pub edge:    BlockedGrid<u64, 64, 64>,
    pub palette: BlockedGrid<u8,  64, 64>,
    pub depth:   BlockedGrid<u16, 64, 64>,
    pub alpha:   BlockedGrid<u8,  64, 64>,
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

    /// Compile-time field accessor: `field_n::<I>()` returns a reference to
    /// the I-th field's BlockedGrid. Matches the `soa_struct!` pattern from
    /// W3-W6 — avoids runtime field-index lookups in hot paths.
    pub fn field_n<const I: usize>(&self) -> &dyn FieldGridRef { ... }

    // === COMPUTE paths (PRIMARY) — see BlockedGrid::map_base data-flow note ===
    //
    // Each `map_l*` returns a new ShaderCellGrid (or a generated variant)
    // with the closure-mapped values. Input grid is not mutated.

    pub fn map_l1<F>(&self, f: F) -> Self
    where F: FnMut(&ShaderCellL1Block<'_>, &mut ShaderCellL1BlockMut<'_>) { ... }

    pub fn map_l2<F>(&self, f: F) -> Self where /* L2 super-block sig */ { ... }
    pub fn map_l3<F>(&self, f: F) -> Self where /* L3 super-block sig */ { ... }
    pub fn map_l4<F>(&self, f: F) -> Self where /* L4 super-block sig */ { ... }

    // === WRITE-BACK paths (SECONDARY) — see BlockedGrid::bulk_apply_base note ===

    pub fn bulk_apply_l1<F>(&mut self, f: F)
    where F: FnMut(&mut ShaderCellL1BlockMut<'_>, (usize, usize)) { ... }

    pub fn bulk_apply_l2<F>(&mut self, f: F) where /* L2 sig */ { ... }
    pub fn bulk_apply_l3<F>(&mut self, f: F) where /* L3 sig */ { ... }
    pub fn bulk_apply_l4<F>(&mut self, f: F) where /* L4 sig */ { ... }
}

pub struct ShaderCellL1Block<'a> {
    pub edge:    GridBlock<'a, u64, 64, 64>,
    pub palette: GridBlock<'a, u8,  64, 64>,
    pub depth:   GridBlock<'a, u16, 64, 64>,
    pub alpha:   GridBlock<'a, u8,  64, 64>,
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
- `new`, `rows`, `cols`, `padded_rows`, `padded_cols`, `blocks_l1`, `blocks_l2`, `blocks_l3`, `blocks_l4`, `map_l1`, `map_l2`, `map_l3`, `map_l4`, `bulk_apply_l1`, `bulk_apply_l2`, `bulk_apply_l3`, `bulk_apply_l4`, `field_n`, `default`

## Layering rule recap (where the AMX-vs-AVX-512-vs-NEON dispatch lives)

`BlockedGrid` is **pure layout**. It contains no `#[target_feature]`, no per-arch imports, no raw intrinsics, no SIMD primitives. The hardware dispatch happens **inside the consumer's closure body**, via `crate::simd::*` calls that route through the existing `simd_caps()` LazyLock.

Example (compute path):

```rust
let grid: BlockedGrid<u64, 64, 64> = BlockedGrid::new(1024, 768);

let out: BlockedGrid<u64, 64, 64> = grid.map_base(|inp, out| {
    // Inside the closure: typed SIMD register-stack primitives from crate::simd.
    // These dispatch AMX / AVX-512 / NEON via simd_caps() LazyLock (existing infra).
    // PR-X5 will add StackedU64x8<N> et al. to crate::simd; for PR-X3 we just
    // demonstrate that the closure boundary is the right place for them.
    for r in 0..64 {
        let in_row: &[u64] = inp.row(r);
        let out_row: &mut [u64] = out.row_mut(r);
        // row.len() == 64. Process in stacked-U64x8 chunks of 8 cells each.
        // Future: crate::simd::stacked_u64x8_apply::<8>(in_row, out_row, |stack| { ... });
        // Today: scalar loop (the API is forward-compatible).
        for (i, &cell) in in_row.iter().enumerate() {
            out_row[i] = /* CausalEdge64 mantissa derivation */ cell;
        }
    }
});
```

Two clean layers meet at the closure boundary:
1. **`crate::hpc::blocked_grid::BlockedGrid<T, BR, BC>`** — pure layout, generic over cell type and block shape
2. **`crate::simd::{StackedU64x8<N>, StackedF32x16<N>, …, AmxTile<T, R, C>}`** — typed register-stack primitives (PR-X5 ships these; not part of PR-X3)

PR-X3 does **NOT** ship layer 2. PR-X3 ships layer 1 only. The closure-supplied work in PR-X3 tests / doctests uses **scalar** inner loops to demonstrate forward-compatibility.

## Padding strategy — explicit

- Storage is padded to **base block boundary only** (`padded_rows = ceil(rows / BR) * BR`, same for cols).
- Higher tiers (L2/L3/L4) do NOT require additional padding. They iterate the existing padded storage; tier-N iteration is valid only when `padded_rows % (BR * N) == 0` and `padded_cols % (BC * N) == 0`.
- For the default 64×64 base on a 100×100 grid: `padded_rows = padded_cols = 128`. L1 iteration yields 2×2 = 4 base blocks. L2 (N=4) is invalid because `128 % (64*4) = 128 % 256 != 0` → L2 iteration must panic or return Empty (design decision: **panic with a clear message** so the caller picks the right tier for their grid size).
- Padding cells default to `T::default()` via `new`, or to a caller-chosen value via `new_with_pad(rows, cols, pad_value)`. For `u64` CausalEdge64, `T::default() = 0` (causally-null edge), which is a safe identity for most cascade ops (XOR with 0 is no-op; popcount of 0 is 0). Consumers who need a non-zero sentinel (e.g., bit-pattern `0xFFFF_FFFF_FFFF_FFFF` to mean "uninitialized") use `new_with_pad`.
- `BlockedGrid::new(0, 0)` is valid → produces a zero-cell grid with `padded_rows == padded_cols == 0`. Block iterators yield empty.
- `BlockedGrid::new(rows, 0)` or `(0, cols)` — same: empty grid.

## Tests required (per file, written by workers)

### Unit tests for `BlockedGrid<T, BR, BC>`

- `new(0, 0)` produces zero-cell grid with empty iterators
- `new(100, 100)` with BR=BC=64 produces 128×128 padded storage, 2×2 L1 blocks
- `new(64, 64)` exactly matches a single L1 block, 1×1 L1 iteration
- `new(256, 256)` with BR=BC=64 produces 4×4 L1 blocks, 1×1 L2 super-block
- `new(100, 100)` with BR=BC=64 — L2 iteration must panic (100 < 256 padded)
- `new_with_pad(100, 100, 0xDEAD)` — padding cells equal 0xDEAD, logical cells equal 0xDEAD until set
- `new_with_pad` works for `T: Copy` even when `T: !Default` (use a wrapper type without Default)
- `idx(r, c)` correct for in-range logical (r, c)
- `get` / `set` round-trip
- Padding cells initialized correctly — verify via `as_padded_slice` at indices past logical extent
- `blocks_base` iterator yields blocks in row-major order with correct `block_row` / `block_col`
- `blocks_base_mut` mutation visible in subsequent `blocks_base` read
- `map_base` returns a new grid; input is unchanged after call (verify via pre/post snapshot)
- `map_base` closure receives input and output blocks at matching coordinates
- `bulk_apply_base` invokes closure once per block with correct coordinates
- Half-square shape: `BlockedGrid<u8, 16, 64>` (AMX INT8) — verify padding, iteration
- Single-strip shape: `BlockedGrid<f32, 1, 16>` (one F32x16 strip) — verify
- 8×8 square: `BlockedGrid<f64, 8, 8>` — verify
- `blocks_tier::<4>()` on a 256×256 grid yields one super-block; `blocks_tier::<4>()` on 128×128 panics (the assertion explained above)
- Const-generic compile-time assertion: `BlockedGrid::<u64, 0, 64>::new(...)` fails to compile (BR > 0 const assert)

### Doc-tests

Every public fn / method gets a working `# Example` doctest. Module-level doctest demonstrates the canonical compose pattern: build a `ShaderMantissaGrid::new(1024, 768)`, use `map_l1` to derive a transformed grid, verify the input is unchanged.

### Unit tests for `blocked_grid_struct!` macro

- 2-field, 3-field, 4-field struct generation
- `pub` and private field visibility per field
- `#[derive(Clone)]` passthrough on the macro input
- Override `#[grid(block = (16, 16))]` produces AMX-shaped sub-grids
- `map_l1` returns a new struct with mapped fields; input unchanged
- `bulk_apply_l1` closure receives all fields in lockstep with same `(block_row, block_col)`
- `field_n::<0>()` returns the first field's BlockedGrid
- New struct's `rows()` / `cols()` / `padded_rows()` / `padded_cols()` are consistent across all fields

### Integration test with W4 `bulk_apply`

A single test composing W4 `bulk_apply` over the L1-block iterator's output. Demonstrates that PR-X3 composes cleanly with the W3-W6 primitives without re-implementing chunking.

## Out of scope — explicitly NOT in PR-X3

These are NOT part of PR-X3 (each becomes its own future PR). The module-level docstring (`//!`) must repeat this list in three lines max to deflect "why isn't aos_to_soa SIMD-accelerated"-shaped issues.

1. **SIMD register-bank stack types** (`StackedU64x8<N>`, `StackedF64x8<N>`, `StackedF32x16<N>`, `AmxTile<T, R, C>`) → PR-X5
2. **Typed distance bulk fns** (palette-256, hamming popcount early-exit, Base17 L1, BF16 mantissa direct-transform) → W7, bench-gated
3. **CausalEdge64 mantissa cell kernel** (the actual L1 pass body) → W7
4. **splat3d adoption** (refactor `splat3d/tile.rs` onto `BlockedGrid`) → PR-X4 (depends on this PR)
5. **Per-field `#[grid(field_block = ...)]` heterogeneous block shapes** → document as future work; not in v1
6. **Sparse storage variant** (`HashMap<(u16,u16), BlockedGrid<T>>` for sparse Gaussian distributions) → out of scope; if needed, separate PR
7. **Cascade orchestrator** (`cascade_topk_per_tile` composing L1→L2→L3 typed metrics over the grid) → W8, depends on W7

## Distance-typing guardrail

**`BlockedGrid` is layout-only and explicitly does NOT bake in any distance metric.** Per the binding rule in `.claude/knowledge/cognitive-distance-typing.md`:
- No `fn bulk_distance<T>` umbrella
- No `enum DistanceMetric { Palette256, Hamming, Base17, … }`
- No `Box<dyn Distance>` trait object
- No generic `fn distance<T>(a: &T, b: &T) -> f32`

The grid type holds `T`. It doesn't know what `T` means. The semantics live in:
- Consumer closures passed to `map_l{1,2,3,4}` (PRIMARY compute path — W1a contract — closure absorbs domain semantics)
- Consumer closures passed to `bulk_apply_l{1,2,3,4}` (SECONDARY write-back path — same contract)
- Typed primitives in `crate::simd::*` that the closures call (PR-X5)
- Typed distance bulk fns in `crate::hpc::cognitive::*` (W7)

Workers MUST NOT add any distance-aware API to this PR. Module headers reference `cognitive-distance-typing.md` and warn against extension toward distance.

## Worker decomposition (SEQUENTIAL — the binding protocol)

**Protocol:** 5–10 Sonnet workers + 1 Opus coordinator (this session). Workers run **sequentially**, one at a time. Each worker's output is reviewed / verified before the next worker spawns. This matches the binding protocol established 2026-05-18:

> sequentially 5-10 sonnet agents + 1 Koordinator
> plan → review → correct → sprint → review code → fix P0 → commit → repeat

### Per-worker file scoping (binding)

PR-X3 splits the implementation across one file per sprint worker. Each worker writes ONLY to their assigned file plus inline `#[cfg(test)] mod tests`. The coordinator owns `mod.rs` and refactors it across the sprint.

| Worker | Owns file | Public items |
|---|---|---|
| A1 | `src/hpc/blocked_grid/base.rs` | `BlockedGrid<T, BR, BC>` struct + `GridBlock` + `GridBlockMut` + all accessors (`new`, `new_with_pad`, `idx`, `get`, `set`, `as_padded_slice*`, `block_dims`, `rows`/`cols`/`padded_rows`/`padded_cols`) |
| A2 | `src/hpc/blocked_grid/iter.rs` | `BaseBlockIter`, `BaseBlockIterMut`, `blocks_base`, `blocks_base_mut` (added as `impl` on `BlockedGrid` from `super::base`) |
| A3 | `src/hpc/blocked_grid/super_block.rs` | `GridSuperBlock`, `GridSuperBlockMut`, `TierBlockIter`, `blocks_tier::<N>` |
| A4 | `src/hpc/blocked_grid/compute.rs` | `map_base`, `map_tier`, `bulk_apply_base`, `bulk_apply_tier` (with data-flow Rule #3 docstring on each `&mut self` method) |
| A5 | `src/hpc/blocked_grid/aliases.rs` | `ShaderMantissaGrid`, `AmxBf16Grid`, `AmxInt8Grid`, `StripF32Stack2`, `StripF32Stack4`, `SquareF64Stack8`, `HalfSquareU64` type aliases + L1/L2/L3/L4 alias impls on `BlockedGrid<T, 64, 64>` |
| A6 | adds inline doctests + integration tests across existing files (coordinator approves the touch list before spawn) | none new — test density only |
| B | `src/hpc/blocked_grid/grid_struct_macro.rs` | `blocked_grid_struct!` macro + macro-generated struct iterator types |
| (coord) | `src/hpc/blocked_grid/mod.rs` | submodule declarations + `pub use` re-exports — workers do NOT touch this file |

Workers MUST NOT modify a file outside their assigned scope. The coordinator updates `mod.rs` re-exports after each worker lands. This file-per-worker discipline enables **safe parallel spawns** — workers writing to different files cannot collide on the merge.

### The agent sequence for PR-X3

Each agent runs **sequentially** for type-dependency reasons (A2 needs A1's `BlockedGrid`; A3 needs A2's iterators; etc.) UNLESS the coordinator pre-lands the file-split scaffolding — in that case A2-A5 can spawn in parallel because each writes against the committed design spec rather than against the previous worker's live output.

All workers use **Sonnet** (not Opus — coordinator is Opus). All workers operate in isolated worktrees via `isolation: "worktree"`.

| # | Phase | Agent role | Scope | Coordinator action between this and next |
|---|---|---|---|---|
| 1 | **plan** | (this doc, v1) | written by coordinator | committed at b348d43c |
| 2 | **review** | plan-review savant | audits design, returns READY-WITH-DOC-FIXES + P0/P1 list | apply patches; commit v2 |
| 3 | **correct** | (coordinator, v2) | applies savant's P0/P1 + Q1–Q7 rulings to design doc | **WE ARE HERE** → commit v2; ready for sprint |
| 4 | **sprint worker A1** | `BlockedGrid<T, BR, BC>` struct + `GridBlock` / `GridBlockMut` types + `new` / `new_with_pad` / `idx` / `get` / `set` / `as_padded_slice*`. Inline unit tests for these. Single commit. | cherry-pick onto coordinator branch; verify green |
| 5 | **sprint worker A2** | `BaseBlockIter` / `BaseBlockIterMut` + `blocks_base` / `blocks_base_mut`. Inline tests. Single commit. Depends on A1. | cherry-pick; verify green |
| 6 | **sprint worker A3** | `GridSuperBlock` / `GridSuperBlockMut` + `TierBlockIter` + `blocks_tier::<N>`. Inline tests. Single commit. Depends on A2. | cherry-pick; verify green |
| 7 | **sprint worker A4** | `map_base` / `map_tier` (PRIMARY compute paths) + `bulk_apply_base` / `bulk_apply_tier` (SECONDARY write-back) with data-flow rule docstrings. Inline tests covering both invariants (input unchanged after `map_base`; closure sees correct coordinates in `bulk_apply_base`). Single commit. Depends on A3. | cherry-pick; verify green |
| 8 | **sprint worker A5** | Convenience aliases (ShaderMantissaGrid, AmxBf16Grid, AmxInt8Grid, StripF32Stack2, StripF32Stack4, SquareF64Stack8, HalfSquareU64) + L1/L2/L3/L4 alias impls on 64×64 base (both `blocks_l*` / `map_l*` / `bulk_apply_l*`). Single commit. Depends on A4. | cherry-pick; verify green |
| 9 | **sprint worker A6** | Full unit-test coverage + doctests for every public fn. Module-level doctest demonstrates the canonical `map_l1` compose pattern. Single commit. Depends on A5. | cherry-pick; verify green |
| 10 | **sprint worker B** | `blocked_grid_struct!` macro in `src/hpc/blocked_grid/grid_struct_macro.rs`. Emits BOTH `map_l*` (compute) AND `bulk_apply_l*` (write-back) on the generated struct + `field_n::<I>` const-generic field accessors. All macro tests. Cargo check/test/fmt/clippy must pass. Single commit. **Depends on Worker A6's `BlockedGrid` API being on the branch.** | cherry-pick; verify green |
| 11 | **review code** | codex P0 auditor | audits combined diff (A1-A6 + B) for: zero `#[target_feature]`, zero `use crate::simd_avx{512,2}` / `simd_neon` / `simd_wasm` imports, zero `cfg(target_feature = …)` gates, zero raw `_mm*_*` / `vld*_*` / `_pdep_*` intrinsics, zero distance-aware API surface, **data-flow Rule #3 compliance on all `&mut self` methods** (the new gate), all public fns have working `///` doc-examples, tests cover all spec'd cases. Verdict: READY-FOR-PR or NEEDS-FIX with P0 list. | apply P0 fixes (if any); commit |
| 12 | **fix P0** | (coordinator) | applies codex P0 patches; commit | push; open PR |
| 13 | **review pr (P2)** | P2 codex savant | reviews the open PR for API ergonomics, naming drift, doc-prose quality, distance-typing visibility on the public PR, future-proofing for the SIMD swap, CI signal. Verdict: SHIP-AS-IS / SHIP-WITH-FOLLOWUPS / RECONSIDER. | apply highest-leverage pre-merge tightening; push; merge ladder |
| 14 | **repeat / next sprint** | (coordinator) | if P2 savant recommends follow-ups too heavy for this PR, queue PR-X3.1; otherwise advance to PR-X4 / PR-X5 / W7 |

### The 7-worker split is the DEFAULT (H1 P1 ruling)

Per the plan-review savant: **A1–A6 + B is the default decomposition**, not a fallback. The composite "Worker A handles all six together" path overruns Sonnet's reliable single-pass attention now that A4 includes both the `map_*` (compute) and `bulk_apply_*` (write-back) families per the A1 P0 patch. The coordinator does NOT downshift to a composite-A worker; the seven cuts above are the binding decomposition.

If a sprint worker's first commit fails the green-check (cargo check/test/fmt/clippy), the coordinator narrows that worker's scope further (e.g., split A4 into A4a `map_base/map_tier` and A4b `bulk_apply_base/bulk_apply_tier`) rather than retrying the same scope.

### Worker isolation rule

Every Sonnet sprint worker runs with `isolation: "worktree"` (NOT in the coordinator's main tree). Workers commit to their own branch; coordinator cherry-picks. This prevents the worker-B-bleeding-into-W2-branch incident from the W3-W6 sprint.

### Sequential vs parallel — why sequential

The earlier W3-W6 sprint ran Worker A and Worker B in **parallel**. That worked for W3-W6 because A (`hpc/soa.rs`) and B (`hpc/bulk.rs`) were independent files. For PR-X3, every worker after A1 depends on the previous worker's types (A2 needs A1's `BlockedGrid`; A3 needs A2's iterators; A4 needs A3's super-block types; etc.). Worker B (the macro) emits code that depends on Worker A6's complete `BlockedGrid` API. Sequential ordering eliminates the integration risk of "Worker N writes against a mock API; Worker N-1 ships a slightly different API; integration breaks."

The user's binding protocol clarifies: **sequential is the default; parallel is only when files are truly independent**.

## What workers commit per file

1. Implement the spec above exactly. No deviation in API.
2. Add inline tests covering the cases listed under §"Tests required" for the file.
3. Add the `pub mod blocked_grid;` registration in `src/hpc/mod.rs` (Worker A1).
4. Run from worktree root:
   - `cargo check -p ndarray --no-default-features --features std`
   - `cargo test -p ndarray --lib --no-default-features --features std hpc::blocked_grid`
   - `cargo test --doc -p ndarray --no-default-features --features std hpc::blocked_grid`
   - `cargo fmt --all -- --check`
   - `cargo clippy -p ndarray --no-default-features --features std -- -D warnings`
   - All green before commit.
5. Commit message format:
   - Worker A1: `feat(hpc/blocked_grid): add BlockedGrid<T, BR, BC> struct + accessors (PR-X3 A1)`
   - Worker A2: `feat(hpc/blocked_grid): add base-block iterators (PR-X3 A2)`
   - Worker A3: `feat(hpc/blocked_grid): add super-block + tier iterators (PR-X3 A3)`
   - Worker A4: `feat(hpc/blocked_grid): add map_* compute + bulk_apply_* write-back (PR-X3 A4)`
   - Worker A5: `feat(hpc/blocked_grid): add convenience aliases + L1-L4 impls (PR-X3 A5)`
   - Worker A6: `test(hpc/blocked_grid): full unit + doctest coverage (PR-X3 A6)`
   - Worker B: `feat(hpc/blocked_grid): add blocked_grid_struct! macro (PR-X3 macro)`

## Verification commands (run from /home/user/ndarray)

Identical to W3-W6 protocol:

```bash
cargo check -p ndarray --no-default-features --features std
cargo test -p ndarray --lib --no-default-features --features std hpc::blocked_grid
cargo test --doc -p ndarray --no-default-features --features std hpc::blocked_grid
cargo fmt --all -- --check
cargo clippy -p ndarray --no-default-features --features std -- -D warnings
```

All five must pass green.

## Sprint protocol (the established multi-agent pattern)

1. ✅ **Design v1** committed (this doc @ b348d43c)
2. ✅ **Plan-review savant** spawned — returned READY-WITH-DOC-FIXES with 2 P0 + 7 P1 + 4 P2; verdict at `.claude/knowledge/pr-x3-plan-review.md`
3. ✅ **Design v2** absorbs all P0/P1 patches + Q1–Q7 rulings (THIS REVISION)
4. ⬜ **Spawn sprint workers SEQUENTIALLY** per §"Worker decomposition" (A1 → A2 → A3 → A4 → A5 → A6 → B). NOT in parallel — each depends on the previous. Each runs with `isolation: "worktree"`.
5. ⬜ Worker commits cherry-picked onto branch after each worker passes green-check
6. ⬜ **Codex P0 audit** spawned on combined diff (A1–A6 + B)
7. ⬜ Fix any P0s
8. ⬜ Open PR
9. ⬜ **P2 codex savant** review on the open PR (ergonomics / drift / naming)
10. ⬜ Same-day follow-up PR for any pre-merge tightenings the P2 savant recommends

## Cross-references

- `.claude/knowledge/pr-x3-plan-review.md` — savant verdict that produced this v2
- `.claude/rules/data-flow.md` — Rule #3 ("No `&mut self` during computation") — the binding rule that drove the A1/A2 P0 patches
- `.claude/knowledge/w3-w6-soa-aos-design.md` — the SoA/AoS foundation this builds on; same protocol shape, same layering rule
- `.claude/knowledge/cognitive-shader-foundation.md` — ndarray's role in the 7-layer cognitive shader stack; identifies the gaps PR-X3 fills
- `.claude/knowledge/cognitive-distance-typing.md` — the binding rule that PR-X3 must respect (no umbrella distance, no roundtrips, typed metrics only)
- `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a layering rule (user code → `crate::simd` → `simd_{type}.rs`); PR-X3 is user-level code
- `.claude/knowledge/w3-w6-codex-audit.md` — example codex P0 audit output for protocol reference
- `.claude/knowledge/w3-w6-p2-savant-review.md` — example P2 savant review output for protocol reference
- `src/hpc/soa.rs` — W3-W6 SoaVec + soa_struct! (the 1-D primitive PR-X3 extends to 2-D)
- `src/hpc/bulk.rs` — W4 bulk_apply / bulk_scan (the chunked-traversal primitive PR-X3 composes with at the tier level)
- `src/hpc/splat3d/tile.rs` — the bespoke 16×16-tile binning that PR-X4 (future) refactors onto `BlockedGrid`

## Resolved questions (savant rulings on v1 §"Open questions")

The plan-review savant ruled definitively on all seven open questions. Workers MUST follow these rulings without further consultation.

1. **Q1 — Naming**: **BlockedGrid** (NOT CognitiveGrid). Rationale: the type is a generic 2-D blocked grid usable anywhere a hierarchical layout matters (BLAS GEMM blocking, image processing, scientific computing); the "cognitive" prefix overstates the type's scope. The cognitive-shader framing is carried by the alias `pub type ShaderMantissaGrid = BlockedGrid<u64, 64, 64>;`. Module path: `crate::hpc::blocked_grid::*`. Macro: `blocked_grid_struct!`.

2. **Q2 — Tier API surface**: **Both**. Provide the const-generic `map_tier::<N>` / `bulk_apply_tier::<N>` / `blocks_tier::<N>` AS WELL AS the L1/L2/L3/L4 alias methods (only on the 64×64 base). Aliases are convenience; const-generic is the escape hatch for non-default bases.

3. **Q3 — Block lifetime variance**: **Separate types**. `GridBlock<'a, T, BR, BC>` and `GridBlockMut<'a, T, BR, BC>` are distinct types with explicit `PhantomData<&'a T>` / `PhantomData<&'a mut T>` markers for lifetime variance (idiomatic Rust 2024 — do not rely on by-virtue-of-having-a-`&'a [T]`-field).

4. **Q4 — Per-field heterogeneous block shapes**: **Compatible / future work**. v1 locks to uniform block shape (all fields share `BR, BC`). Per-field `#[grid(field_block = ...)]` extension is additive and would NOT break v1's API. Documented as future work in macro docstring; NOT implemented in v1.

5. **Q5 — Padding init value**: **Add `new_with_pad`**. Provide `BlockedGrid::new_with_pad(rows, cols, pad_value: T)` alongside `new(rows, cols)`. The `new_with_pad` ctor has bound `T: Copy` only (no `Default`); `new` keeps `T: Copy + Default` and delegates to `new_with_pad(rows, cols, T::default())`. The split lets consumers (a) use sentinel padding values (`0xFFFF_FFFF_FFFF_FFFF` for "uninitialized"), and (b) use types that don't implement `Default`.

6. **Q6 — `as_padded_slice` exposure**: **Feature, with explicit `# Footgun` doc section**. Keep `as_padded_slice` / `as_padded_slice_mut` public. Each method's docstring carries a `# Footgun` section explaining: slice includes padding cells; use `idx()` to compute logical-cell flat indices; do NOT use `r * cols() + c` (that ignores stride and reads the wrong cell).

7. **Q7 — L1-L4 aliases on non-64×64 grids**: **64×64-only**. The L1/L2/L3/L4 alias methods (`blocks_l1` / `map_l1` / `bulk_apply_l1` etc.) live ONLY on `BlockedGrid<T, 64, 64>` (and on macro-generated SoA-of-grids structs built from 64×64 fields). AMX (16×16), strip (1×16), half-square (32×64), and other non-default-base grids use the raw `blocks_tier::<N>` / `map_tier::<N>` / `bulk_apply_tier::<N>` const-generic methods. Documented in the alias docstring.

## Done criteria

PR-X3 is done when:
- All worker spec items implemented per the 7-worker split (A1–A6 + B)
- Codex P0 audit passes with 0 P0 — **including the data-flow Rule #3 gate** on every `&mut self` method
- `cargo check / test --lib / test --doc / fmt / clippy` all green
- Layering rule verified (zero per-arch imports / target_feature / raw intrinsics in the new files)
- Distance-typing guardrail verified (zero umbrella-distance API surface)
- Module headers reference `cognitive-distance-typing.md` AND `.claude/rules/data-flow.md` and warn against distance extension + `&mut self` compute paths
- P2 savant review delivers SHIP verdict (with optional same-day follow-up PR for the highest-leverage P2)

## Token-reset safety notes (for fresh sessions)

This doc was written when the conversation was at 96% context. v2 added at 97%. If you're picking up after a token reset:

1. Read this entire doc first.
2. Then read `.claude/knowledge/pr-x3-plan-review.md` — the savant verdict that drove the v2 patches.
3. Check `.claude/knowledge/` for any newer planning docs.
4. Check `git log --oneline -10` on this branch (`claude/pr-x3-cognitive-grid-design`, may be renamed to `claude/pr-x3-blocked-grid-design`) and on `master` to see what shipped.
5. The W2/W3-W6 multi-agent sprint protocol is the canonical pattern — see `.claude/knowledge/w3-w6-soa-aos-design.md` §"Sprint protocol" for the same shape.
6. Open PRs to track: #155 (sigmoid orphan rescue, may be merged), #157 (P2 savant follow-up, may be merged), this branch's PR (not yet open).
7. PR-X1 and PR-X2 are designed in conversation but not yet specced to disk. If you need them, see `cognitive-shader-foundation.md` §"Current Gaps" and the savant A1/A4 P2 findings in `w3-w6-p2-savant-review.md`.
8. The hardware-block × cell-type matrix in §"Hardware-block × cell-type matrix" is the canonical reference for which block shape fits which SIMD tier. Memorize it before proposing API changes.
9. **The A1/A2 P0 ruling is non-negotiable**: every `&mut self` method on `BlockedGrid` must have an explicit data-flow rule docstring section pointing readers to the `map_*` PRIMARY compute path. Workers who emit `bulk_apply_*` methods without this docstring will fail the codex P0 audit (gate added in Phase 11).
