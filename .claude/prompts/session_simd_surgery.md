# SESSION: std::simd Polyfill — Move from rustynum to ndarray

## Phase 1: Additive Only (no modifications, no deletions)

Copy three files from rustynum to ndarray `src/` root. Same filenames.
Same position in the crate — peers of `lib.rs`, not inside `backend/`.

```bash
# Source (rustynum, working, tested)
<rustynum>/rustynum-core/src/simd.rs
<rustynum>/rustynum-core/src/simd_avx512.rs
<rustynum>/rustynum-core/src/simd_avx2.rs

# Target (ndarray src/ root)
ndarray/src/simd.rs
ndarray/src/simd_avx512.rs
ndarray/src/simd_avx2.rs
```

Add to `ndarray/src/lib.rs`:
```rust
pub mod simd_avx512;
pub mod simd_avx2;
pub mod simd;
```

Same declarations as rustynum's `lib.rs` (lines 16, 50, 53).

After this: `crate::simd::` works in ndarray. When `std::simd` stabilizes:
`crate::simd` → `std::simd`, delete the three files. One word change.

**Do NOT modify any existing files beyond lib.rs.**
**Do NOT delete simd_compat.rs or change its imports.**
**Do NOT touch kernels_avx512.rs, activations.rs, or vml.rs.**

Just add the three files and the three `pub mod` lines. Verify it compiles.

## Phase 2: Surgery (separate step, after Phase 1 is verified)

ndarray currently has `src/backend/simd_compat.rs` (1072 lines) created
by a session that didn't read rustynum's architecture. It works (820 tests
pass) but has problems. Three other files import from it.

Before touching anything in Phase 2, read ALL of these:

```bash
# What exists now (save snapshots first)
cat src/backend/simd_compat.rs
cat src/backend/kernels_avx512.rs
cat src/hpc/activations.rs
cat src/hpc/vml.rs

# What existed BEFORE simd_compat was introduced
git show 22bfb7a:src/backend/kernels_avx512.rs
git show 22bfb7a:src/hpc/activations.rs
git show 22bfb7a:src/hpc/vml.rs

# The real polyfill you just added in Phase 1
cat src/simd.rs
cat src/simd_avx512.rs
cat src/simd_avx2.rs
```

Then answer these questions IN WRITING to `.claude/SIMD_SURGERY_REPORT.md`:

1. What types does simd_compat.rs define vs simd_avx512.rs? What's missing?
2. Does simd_compat.rs have a router? Does it have an AVX2 tier?
3. What did kernels_avx512.rs look like BEFORE vs AFTER the simd_compat session?
4. What did activations.rs gain? Is the new code correct?
5. What did vml.rs gain? Is the new code correct?
6. Do kernels_avx512.rs, activations.rs, vml.rs even need to exist as
   separate files, or should their contents live inside the type definitions
   like rustynum does it?
7. If separate kernel files DO need to exist — should it be `kernels.rs`
   (architecture-neutral, using `crate::simd::`) or `kernels_avx512.rs`
   (architecture-specific name)?
8. What should happen to simd_compat.rs? Keep? Delete? Merge into the
   real polyfill? Something else?

Based on your answers, rewire the imports from `crate::backend::simd_compat::`
to `crate::simd::`, resolve any API differences, and verify all tests pass.

## Contamination Warning

A previous session wrote 5 prompt files that reference `simd_compat.rs` by
name as the intended target (24 total references). This is why other sessions
keep creating or wiring to that filename. After Phase 2 is complete, update
these prompt files to reference whatever the correct result actually is:

```
session_ndarray_migration_inventory.md    6 references to simd_compat
session_bgz17_similarity.md              3 references
research_quantized_graph_algebra.md       1 reference
session_unified_vector_search.md          1 reference
this file (session_simd_surgery.md)      references are to document the problem
```
