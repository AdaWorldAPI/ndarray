# PR-X3 Plan Review — Savant Verdict

Auditor: Sonnet plan-review savant (Phase 2 of sequential PR-X3 sprint)
Design doc reviewed: `.claude/knowledge/pr-x3-cognitive-grid-design.md` @ b348d43c
Verdict: **READY-WITH-DOC-FIXES**

P0 count: 2 | P1 count: 7 | P2 count: 4

## P0 findings (must fix before sprint can spawn)

### A1 — `bulk_apply_base(&mut self, F)` violates data-flow Rule #3

The methods `bulk_apply_base(&mut self, F)` and `bulk_apply_tier(&mut self, F)` violate `.claude/rules/data-flow.md` Rule #3 ("No `&mut self` during computation. Ever."). The methods are explicitly framed as compute paths in the design doc (§"Tier semantics map to cognitive shader passes" places "the CausalEdge64 mantissa pass" on `bulk_apply_l1`), not builder/constructor paths.

**Patch language**: split the API into two named families:

```rust
// PRIMARY compute path - immutable self, returns new grid (builder pattern)
pub fn map_base<U: Copy + Default, F>(&self, f: F) -> BlockedGrid<U, BR, BC>
where
    F: FnMut(&Block<'_, T, BR, BC>, &mut BlockMut<'_, U, BR, BC>);

pub fn map_tier<U: Copy + Default, const N: usize, F>(&self, f: F) -> BlockedGrid<U, BR, BC>
where
    F: FnMut(&SuperBlock<'_, T, BR, BC, N>, &mut SuperBlockMut<'_, U, BR, BC, N>);

// SECONDARY write-back variant - in-place mutation, explicit gated write-back
//
// # Data-flow rule
//
// This is the gated write-back variant of [`map_base`]. The closure performs
// write-back operations ONLY (per `.claude/rules/data-flow.md` Rule #3).
// For compute paths use `map_base` which returns a new grid.
pub fn bulk_apply_base<F>(&mut self, f: F)
where
    F: FnMut(&mut BlockMut<'_, T, BR, BC>);

pub fn bulk_apply_tier<const N: usize, F>(&mut self, f: F)
where
    F: FnMut(&mut SuperBlockMut<'_, T, BR, BC, N>);
```

Same split applies to the L1/L2/L3/L4 convenience aliases on `BlockedGrid<T, 64, 64>`:
- `map_l1` / `map_l2` / `map_l3` / `map_l4` — primary compute paths
- `bulk_apply_l1` / `bulk_apply_l2` / `bulk_apply_l3` / `bulk_apply_l4` — write-back variants

### A2 — Macro-generated bulk_apply methods inherit A1 violation

The `cognitive_grid_struct!`-generated `bulk_apply_l1` / `bulk_apply_l2` / `bulk_apply_l3` / `bulk_apply_l4` carry the same `&mut self` + compute framing problem. Fix propagates from A1: the macro emits BOTH `map_l1` (compute, returns new struct with mapped fields) AND `bulk_apply_l1` (write-back) alongside each other.

## P1 findings

### H1 — Sprint protocol step 4 contradicts the binding sequential rule

§"Sprint protocol" step 4 in the design doc currently says "Two workers in parallel" (carryover from W3-W6 protocol shape). This contradicts the explicit "5–10 sequential Sonnet workers + 1 Opus coordinator" protocol in §"Worker decomposition". Fix: align step 4 to read "Spawn sprint workers SEQUENTIALLY (per §"Worker decomposition")" and remove "in parallel".

Additionally: with P0 patches adding the `map_*` family alongside `bulk_apply_*`, the composite Worker A scope grows past reliable single-pass Sonnet attention. **Adopt the 7-worker split (A1–A6 + B) as the DEFAULT, not the fallback.**

### F1 — Type name `CognitiveGrid` overstates scope

The type is a generic 2-D blocked grid usable anywhere a hierarchical layout matters (BLAS GEMM blocking, image processing tiles, scientific computing). The "cognitive" prefix in the type name leans into one use case but couples the generic primitive to it semantically. **Rename `CognitiveGrid` → `BlockedGrid`. Add `pub type ShaderMantissaGrid = BlockedGrid<u64, 64, 64>;` to carry the cognitive framing as an alias.** Module path stays at `crate::hpc::blocked_grid::*` (with `cognitive_grid` deprecated alias if needed for back-compat — but PR-X3 is greenfield, so just `blocked_grid`).

The macro renames consistently: `cognitive_grid_struct!` → `blocked_grid_struct!`.

### F2 — Block / BlockMut naming

Cross-checked against existing ndarray types: `Block` is used in `crate::backend::native` BLAS kernels for a different concept (cache-blocked GEMM block sizes). To avoid collision, prefer `GridBlock` / `GridBlockMut` / `GridSuperBlock` / `GridSuperBlockMut` in `crate::hpc::blocked_grid::*`.

### F3 — L1/L2/L3/L4 tier names

Cache hierarchy convention is innermost=L1=fastest, outermost=L4=RAM. The design doc uses this convention. Verify the doc states this explicitly (currently implicit). Add a one-sentence note in the L1-L4 alias docstring: "Following cache-hierarchy convention: L1 = innermost (32 KB), L4 = framebuffer-scale (2 GB)."

### G2 — `bulk_apply_tier::<N>` + L2/L3/L4 aliases — keep both

Ruling on Q2: provide BOTH the const-generic `map_tier::<N>` / `bulk_apply_tier::<N>` AND the L2/L3/L4 alias methods. Aliases are convenience for the default 64×64 base; const-generic is the escape hatch for non-default bases. Same applies to the `map_*` family.

### G6 — `T::default()` padding bound is too restrictive

Q5 ruling: ADD `BlockedGrid::new_with_pad(rows, cols, pad_value: T)` constructor that takes the padding fill explicitly. Bound: `T: Copy` only, no `Default`. The `new` constructor stays as `T: Copy + Default` calling `new_with_pad(rows, cols, T::default())`.

### G3 — `as_padded_slice` exposure

Q6 ruling: KEEP `as_padded_slice` / `as_padded_slice_mut` as a feature (not footgun). Add a `# Footgun` section to each method's docstring explaining: "Returned slice includes padding cells at the right and bottom of the logical extent. Use [`rows`]/[`cols`] to compute logical bounds; do NOT use slice indices past `rows() * padded_cols() + cols()` for logical-only data." Plus an example showing how to compute the logical-cell flat index.

### G4 — `field_n::<I>` compile-time accessors on macro output

Q4 ruling: The `blocked_grid_struct!` macro should emit `field_n::<I>()` const-generic field accessors on the generated struct's L1 block type (matching the `soa_struct!` pattern from W3-W6). Failure to do so would force consumers into runtime field-index lookups in hot paths.

## P2 findings

### J1 — Open question Q3 ruling

`Block<'a, T, BR, BC>` and `BlockMut<'a, T, BR, BC>` are separate types (current spec). Verify they carry `PhantomData<&'a T>` / `PhantomData<&'a mut T>` markers explicitly for lifetime variance, not just by-virtue-of-having-`&'a [T]`-field. Idiomatic Rust 2024.

### J2 — Open question Q4 ruling

Per-field `#[grid(field_block = ...)]` heterogeneous block shapes — v1 locks to uniform block shape (all fields share `BR, BC`). Per-field extension is additive, not breaking. Document as "future work" in the macro docstring; do NOT support in v1.

### J3 — Open question Q7 ruling

L1-L4 aliases ONLY on `BlockedGrid<T, 64, 64>`. AMX (16×16) / strip (1×16) / half-square (32×64) grids use raw `blocks_tier::<N>` / `map_tier::<N>` / `bulk_apply_tier::<N>`. Document this constraint in the alias docstring.

### J4 — Add explicit "out of scope: SIMD primitives" warning in module header

The design doc has §"Out of scope" but the module-level docstring (`//!`) should also carry a concise version. Three lines max. Saves consumers from filing "why isn't aos_to_soa SIMD-accelerated" issues.

## Rulings on open questions (Q1–Q7 from design doc)

- **Q1: BlockedGrid** — rename `CognitiveGrid → BlockedGrid`. Add `ShaderMantissaGrid` alias for the cognitive-shader use case.
- **Q2: Both** — `map_tier::<N>` / `bulk_apply_tier::<N>` const-generic entries AND L1-L4 alias methods.
- **Q3: Separate** — keep `Block` / `BlockMut` as distinct types with `PhantomData` lifetime markers (rename to `GridBlock`/`GridBlockMut` per F2).
- **Q4: Compatible** — v1 uniform block shape; per-field extension additive, future work.
- **Q5: Add** — `new_with_pad(rows, cols, pad_value: T)` alongside `new`; `T: Copy` only, no `Default` bound on the new constructor.
- **Q6: Feature** — keep `as_padded_slice*`; add `# Footgun` doc section.
- **Q7: 64×64-only** — L1-L4 aliases only on `BlockedGrid<T, 64, 64>`. AMX / strip / half-square grids use raw `blocks_tier::<N>`.

## Net call

**Recommended next phase: Phase 3 (corrector)** — apply A1+A2 P0 patches + all P1 fixes to the design doc, commit as v2, then spawn Phase 4 sprint workers using the 7-worker split (A1–A6 + B) as the default decomposition. No structural rethink required. The P0 fixes (map/bulk_apply split, BlockedGrid rename) are mechanical edits that propagate cleanly through the doc.
