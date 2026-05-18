# Integration Plan: ndarray's role in the four-repo convergence

**This repo**: `AdaWorldAPI/ndarray` — SIMD distance kernels + tensor primitives, shared across the stack.

**Status**: planning document. Companion plans at the same path in the other repos:
- `AdaWorldAPI/lance-graph:.claude/plans/integration-plan.md`
- `AdaWorldAPI/surrealdb:.claude/plans/integration-plan.md`
- `AdaWorldAPI/sea-orm:.claude/plans/integration-plan.md`

---

## 1. The convergence target

Across all four repos:

> *Foundry-style ontology + BEAM-style supervision + ClickHouse-style analytic + Postgres-style ACID + cognitive primitives — all on one Arrow substrate, surfaced to consumers as a typed sea-orm API.*

Four glue crates close the gap:

| # | Glue crate | Owner repo | Bridges |
|---|---|---|---|
| 1 | `surrealdb-ractor` | surrealdb | `cf` / live queries → ractor mailboxes |
| 2 | `lance-graph-tikv-provider` | lance-graph | TiKV ranges → Arrow `TableProvider` |
| 3 | `sea-orm-ractor` | sea-orm | `Entity::PK` → ractor process registry |
| 4 | `cognitive-shader-actor` | lance-graph | cognitive shaders → `ractor::Actor` adapter |

**This repo owns no glue crate.** It owns the **shared low-level numeric substrate** that the other three depend on — SIMD distance kernels (cosine, L1, L2, Linf), `F64x8` polyfills, `heel_f64x8` helpers, `hpc-extras` feature.

### Integration principle: additive contract shape (this repo IS the canonical case)

**This repo is the load-bearing example of the contract-shape discipline.** Every symbol this repo exposes is consumed by surrealdb-core (`idx/trees/vector.rs`) and lance-graph cognitive crates (`bgz-tensor`, `holograph`, `deepnsm`, `causal-edge`). One signature change breaks the entire stack. The discipline:

1. **Existing stable APIs never change signature.** Period. If a hypothetical improvement requires a different signature, the new signature ships as a new function next to the old one. The old function stays forever or for a 5+-version deprecation runway, whichever is longer.
2. **New kernels are added as new functions in new or existing modules.** Adding `F32x16` doesn't touch `F64x8`. Adding `hamming_u8_simd` doesn't touch `cosine_f64_simd`.
3. **Internal SIMD backends (AVX2/AVX-512/NEON paths) are not public surface.** They can change without notice. Only the public entry points are load-bearing.
4. **The `[patch.crates-io]` block in surrealdb's root Cargo.toml is the diamond-dep guard.** This repo's existence + that patch line is what makes downstream `ort` (ONNX runtime) link the same `ndarray` as surrealdb-core. Breaking the patch contract breaks ONNX interop.

**Per-repo enforcement**: every Sprint item below is read as "add this; don't change what's there."

### Contracts (existing + new)

| Contract | Owner repo | Status today | This plan adds |
|---|---|---|---|
| `ndarray::hpc::F64x8` + `heel_f64x8::*` | **this repo** | 0.17 fork, stable per §5 below | **unchanged — only new kernels (e.g. `F32x16`, int8, Hamming) added in new symbols** |
| `[patch.crates-io] ndarray = ...` in surrealdb root Cargo.toml | surrealdb | active (diamond-dep guard) | not touched |
| `lance-graph-contract` (for cognitive shader / IR vocabulary) | lance-graph | 0.1.x → 0.2.0 additive | not touched by us |
| surrealdb `MvccSource` / `CfStream` | surrealdb | new additive traits | not touched by us |
| sea-orm `EntityActor` / `SelectArrowExt` | sea-orm | new additive trait/derive | not touched by us |

---

## 2. Architecture diagram

```
                ┌──────────────────────────────────────────┐
                │              consumer crate              │
                └──────────────────┬───────────────────────┘
                                   │ typed entities
                                   ▼
                ┌──────────────────────────────────────────┐
                │            sea-orm-arrow 2.0             │
                └────┬─────────────────┬───────────────┬───┘
                     │                 │               │
                     ▼                 ▼               ▼
              ┌───────────┐     ┌───────────┐    ┌───────────┐
              │  ractor   │◄────│ surrealdb │    │lance-graph│
              │ (actors,  │ #1  │  (cf +    │    │ (Cypher,  │
              │ mailboxes,│     │   live    │    │ ontology, │
              │ supervis.)│     │  queries) │    │cognitive) │
              └─────┬─────┘     └─────┬─────┘    └─────┬─────┘
                    │ #3              │                │ #2,#4
                    ▼                 ▼                ▼
              ┌─────────────────────────────────────────────┐
              │       TiKV substrate (Raft + Percolator)    │
              └─────────────────────────────────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │      THIS REPO (ndarray)   │
                    │  - hpc-extras feature      │
                    │  - F64x8 polyfill          │
                    │  - heel_f64x8 distances    │
                    │  - diamond-dep guard       │
                    └────────────────────────────┘
```

---

## 3. Role of ndarray in the integration

This is the **shared low-level numeric substrate**. The AdaWorldAPI fork of ndarray 0.17 with `hpc-extras` lives at the bottom of the stack. Two direct consumers:

1. **surrealdb-core**
   - `core/Cargo.toml:71-77` — `vector-hpc` feature flips on cfg-gated dispatch in `idx/trees/vector.rs`
   - `core/src/idx/trees/vector.rs` — distance helpers (l1/l2/linf) inlined here, using this repo's SIMD kernels
   - Comment from surrealdb's root `Cargo.toml:88-93`:
     > *Always the AdaWorldAPI fork — never crates.io. Direct git dep at the workspace level. Distance helpers (l1/l2/linf) are inlined in surrealdb/core/src/idx/trees/vector.rs.*

2. **lance-graph cognitive crates**
   - `crates/bgz-tensor/` — element-wise ops use ndarray's `Zip` + `F64x8` chunks
   - `crates/holograph/` — holographic distance metrics
   - `crates/deepnsm/` — neural state machine distance kernels
   - `crates/causal-edge/` — causality scoring uses cosine over embedding vectors

Indirectly via sea-orm and the planner, every vector / distance / similarity operation in the stack lands here.

---

## 4. Current state — what makes this fork special

### `F64x8` polyfill

`hpc-extras` feature exposes an 8-wide `f64` SIMD vector type that works on:
- **x86_64 AVX-512** — native 8-wide
- **x86_64 AVX2** — two 4-wide ops, software-packed
- **aarch64 NEON** — two 4-wide via NEON 128-bit, software-packed
- **other archs** — scalar fallback

This is the kernel both surrealdb's `idx/trees/vector.rs` and lance-graph's cognitive shaders rely on.

### `heel_f64x8` distance kernels

Functions composing `F64x8` chunks into a distance:

```
heel_f64x8::cosine_f64_simd(a: &[f64], b: &[f64]) -> f64
heel_f64x8::l1_f64_simd    (a: &[f64], b: &[f64]) -> f64
heel_f64x8::l2_f64_simd    (a: &[f64], b: &[f64]) -> f64
heel_f64x8::linf_f64_simd  (a: &[f64], b: &[f64]) -> f64
```

### Diamond-dep guard

The `[patch.crates-io]` block at the bottom of surrealdb's root `Cargo.toml`:

```toml
[patch.crates-io]
ndarray = { git = "https://github.com/AdaWorldAPI/ndarray.git" }
```

ensures any transitive consumer of `ndarray = "0.17.x"` from crates.io lands on this fork. Without the patch, `ort` (ONNX runtime, optional `ml` feature in surrealdb) would link a separate `ndarray` and surrealdb-core would link this one — two distinct `TypeId`s, no interop.

**This repo's existence is what makes the patch work.** Without it, the diamond-dep workaround has no target to redirect to.

### The `lance-index` 0.16 gap (known)

From surrealdb root `Cargo.toml:100-101`:

> *Scope: 0.17 line only. `lance-index 4.0` depends on `ndarray = "0.16"`, a separate major version that this patch does not affect; eliminating that crates.io 0.16 entry requires upstream `lance-index` to bump.*

**Plan**: watch upstream `lance-index` for the 0.17 bump (see §6 Sprint 2). When it lands, the diamond-dep guard becomes single-version-clean.

---

## 5. API stability commitment (this repo's contract)

This repo doesn't own a glue *crate* — it owns the **API contract that the SIMD layer of three downstream repos depends on**. The commitment is absolute:

### Stable public surface (no break without major bump, none planned)

| Symbol | Kind |
|---|---|
| `ndarray::hpc::F64x8` | type — layout, lane count (8) frozen |
| `ndarray::hpc::heel_f64x8::cosine_f64_simd(a, b) -> f64` | signature frozen |
| `ndarray::hpc::heel_f64x8::l1_f64_simd(a, b) -> f64` | signature frozen |
| `ndarray::hpc::heel_f64x8::l2_f64_simd(a, b) -> f64` | signature frozen |
| `ndarray::hpc::heel_f64x8::linf_f64_simd(a, b) -> f64` | signature frozen |
| feature `hpc-extras` | name + what it enables frozen |

**"Frozen" means**: no signature change, no rename, no semantic drift. If we want to refine — e.g., a fused multiply-add variant of cosine — we add `cosine_f64_simd_fma(a, b) -> f64` as a NEW function. Both coexist forever (or 5+ versions, whichever is longer).

### Internal / unstable

- Polyfill backends (AVX2/AVX-512/NEON paths) — implementation detail
- Auto-dispatch heuristics — can change without notice
- Numeric tolerance in non-cancellation-prone paths — within `f64::EPSILON * len` of scalar reference

### Doc commitment

- Each stable function gets a doc-test
- Cross-arch behaviour documented in `docs/hpc-stability.md` (Sprint 0)
- A CI matrix runs the doc-tests on x86_64-AVX2, x86_64-AVX-512, aarch64-NEON, and scalar-fallback

---

## 6. Sprint sequence (this repo)

All work is **additive** — new symbols in new or existing modules; no existing symbol changes signature.

### Sprint 0 — API freeze + doc (1 week)
- Mark stable APIs with `#[stable]`-style doc tag (custom attribute or doc-comment convention)
- Write `docs/hpc-stability.md` listing the commitment from §5
- Add CI cross-arch doc-test matrix
- Cross-link from this plan

### Sprint 1 — `bgz-tensor` direct coupling (1 week)
- `bgz-tensor` (lance-graph crate) takes a direct dep on this fork (additive: new dep line, no existing dep changes)
- Ensures `bgz-tensor` users always get the SIMD kernels regardless of feature-flag composition
- Coordinate with lance-graph plan §4

### Sprint 2 — `lance-index` 0.17 readiness (timing depends on upstream)
- Watch upstream `lance-index` for the 0.17 bump
- Have a forked `lance-index` 0.17 ready to slot in if upstream delays
- Once available, extend the surrealdb `[patch.crates-io]` block to cover both 0.16 (if still needed) and 0.17
- This is purely additive on this repo's side (we add no symbols; we are the target of the patch)

### Sprint 3 — additional kernels as needed (ad-hoc; all additive)
- Add `F32x16` polyfill if cognitive shaders migrate to f32 (NEW type, F64x8 unchanged)
- Add quantised int8 distance kernels for embedding compression (NEW module `heel_i8x32::*`)
- Add Hamming distance kernel for binary embeddings (NEW function `heel_u8x32::hamming_u8_simd`)

---

## 7. Examples

### Example 1 — surrealdb using the fork's SIMD

```rust
// surrealdb/core/src/idx/trees/vector.rs — sketch of what's already wired
use ndarray::hpc::heel_f64x8;

pub fn cosine_distance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(feature = "vector-hpc")]
    { 1.0 - heel_f64x8::cosine_f64_simd(a, b) }
    #[cfg(not(feature = "vector-hpc"))]
    { scalar_cosine(a, b) }
}
```

### Example 2 — lance-graph cognitive shader using the fork

```rust
// lance-graph/crates/holograph/src/distance.rs
use ndarray::hpc::heel_f64x8;
use crate::HolographEmbedding;

impl HolographEmbedding {
    pub fn similarity(&self, other: &Self) -> f64 {
        heel_f64x8::cosine_f64_simd(self.as_slice(), other.as_slice())
    }
}
```

### Example 3 — `bgz-tensor` element-wise ops via the fork

```rust
// lance-graph/crates/bgz-tensor/src/ops.rs
use ndarray::hpc::F64x8;
use ndarray::Zip;

impl BgzTensor<f64> {
    pub fn elementwise_mul(&self, other: &Self) -> Self {
        let mut out = self.clone();
        Zip::from(&mut out.data)
            .and(&other.data)
            .for_each(|a, &b| *a *= b);
        // F64x8-chunked path handled by ndarray's Zip internals for large tensors.
        out
    }
}
```

### Example 4 — The diamond-dep guard (replicated for cross-reference)

```toml
# surrealdb root Cargo.toml (already in place; documented here so the
# fork knows what surfaces are load-bearing).
[patch.crates-io]
ndarray = { git = "https://github.com/AdaWorldAPI/ndarray.git" }
```

Without this patch:
- `ort` pulls `ndarray = "0.17.2"` from crates.io
- `surrealdb-core` pulls this fork
- They have distinct `TypeId`s → no interop between ONNX outputs and surrealdb's index code

With this patch, both link the same crate. **This fork's stability is the diamond-dep fix.**

### Example 5 — New kernel landing as a new symbol (additive)

Hypothetical: a fused multiply-add cosine variant lands. Old + new coexist:

```rust
// crates/ndarray/src/hpc/heel_f64x8.rs — new function, existing unchanged
pub fn cosine_f64_simd(a: &[f64], b: &[f64]) -> f64 { /* existing */ }

/// FMA variant. Lower latency on AVX-512 + AVX2-FMA hosts.
/// Numerically identical within f64::EPSILON * len.
pub fn cosine_f64_simd_fma(a: &[f64], b: &[f64]) -> f64 { /* new */ }
```

Consumers pick. Nothing breaks.

---

## 8. What this plan asks of the other repos

Nothing structural — only that consumers stay on the stable surface (§5) and report breakage promptly. Specifically:

- **surrealdb**: `idx/trees/vector.rs` should only use `ndarray::hpc::*` items listed in §5. Anything else is a non-stable detail and may break without notice.
- **lance-graph**: cognitive crates should use `heel_f64x8` distance kernels; if a kernel is missing (e.g. Hamming), file an issue here rather than implementing locally.
- **sea-orm**: no direct dep on this fork; touches it only transitively if a consumer uses sea-orm-arrow with `f64` Arrow columns.

---

## 9. Open questions

1. **`F32x16` priority** — is a cognitive shader consumer planning to move to f32? If yes, Sprint 3 fast-track. If no, defer.
2. **Quantised int8 distance kernels** — trigger Sprint 3 item when a concrete consumer surfaces.
3. **WASM target** — surrealdb has a WASM build path. Does it need `vector-hpc`? Today the scalar fallback covers it. Confirm with surrealdb plan.
4. **Numeric tolerance documentation** — currently "within `f64::EPSILON * len`"; doc-test it in Sprint 0.
5. **`#[stable]` attribute convention** — use Rust nightly `#[stable]` (not available on stable) or a doc-comment convention? Probably the latter for portability; revisit when nightly `#[stable]` stabilises.

---

## 10. Cross-references

- **Glue #1** (surrealdb-ractor): `AdaWorldAPI/surrealdb:.claude/plans/integration-plan.md` §5
- **Glue #2** (TiKV TableProvider): `AdaWorldAPI/lance-graph:.claude/plans/integration-plan.md` §5
- **Glue #3** (sea-orm-ractor): `AdaWorldAPI/sea-orm:.claude/plans/integration-plan.md` §5
- **Glue #4** (cognitive-shader-actor): `AdaWorldAPI/lance-graph:.claude/plans/integration-plan.md` §6
- **Cognitive crate consumers** (the load-bearing reason this fork exists): `AdaWorldAPI/lance-graph:.claude/plans/integration-plan.md` §3 + §4
- **surrealdb's `vector-hpc` feature**: `AdaWorldAPI/surrealdb:.claude/plans/integration-plan.md` §4 (`core/Cargo.toml:71-77`)
- **`lance-projection` sibling** (analytic view of cognitive crate outputs): `AdaWorldAPI/surrealdb:.claude/plans/integration-plan.md` §6
