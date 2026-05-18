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

Its contribution to the integration is **API stability**: every kernel signature this repo exposes is a load-bearing contract for two downstream consumers (surrealdb's `idx/trees/vector.rs` + lance-graph's cognitive crates).

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

## 5. API stability commitment

This repo doesn't own a glue *crate* — but it owns the **API contract that the SIMD layer of three downstream repos depends on**.

### Stable public surface (no break without major bump)

| Symbol | Surface |
|---|---|
| `ndarray::hpc::F64x8` | type, layout, lane count (8) |
| `ndarray::hpc::heel_f64x8::cosine_f64_simd(a, b) -> f64` | signature |
| `ndarray::hpc::heel_f64x8::l1_f64_simd(a, b) -> f64` | signature |
| `ndarray::hpc::heel_f64x8::l2_f64_simd(a, b) -> f64` | signature |
| `ndarray::hpc::heel_f64x8::linf_f64_simd(a, b) -> f64` | signature |
| feature `hpc-extras` | name + what it enables |

### Internal / unstable

- Polyfill backends (AVX2/AVX-512/NEON paths) — implementation detail
- Auto-dispatch heuristics — can change without notice
- Numeric tolerance in non-cancellation-prone paths — within `f64::EPSILON * len` of scalar reference

### Doc commitment

- Each stable function gets a doc-test
- Cross-arch behaviour documented in `docs/hpc-stability.md` (to be created — Sprint 0)
- A CI matrix runs the doc-tests on x86_64-AVX2, x86_64-AVX-512, aarch64-NEON, and a scalar-fallback target

---

## 6. Sprint sequence (this repo)

### Sprint 0 — API freeze + doc (1 week)
- Mark stable APIs with `#[stable]`-style doc tag (custom attribute or doc comment convention)
- Write `docs/hpc-stability.md` listing the commitment
- Add CI cross-arch doc-test matrix
- Cross-link from this plan

### Sprint 1 — `bgz-tensor` direct coupling (1 week)
- `bgz-tensor` (lance-graph crate) takes a direct dep on this fork, not via lance-graph workspace transitively
- Ensures bgz-tensor users always get the SIMD kernels regardless of feature flag composition
- Coordinate with lance-graph plan §4

### Sprint 2 — `lance-index` 0.17 readiness (timing depends on upstream)
- Watch upstream `lance-index` for the 0.17 bump
- Have a forked `lance-index` 0.17 ready to slot in if upstream delays
- Once available, extend the `[patch.crates-io]` block in surrealdb to cover both 0.16 (if still needed) and 0.17

### Sprint 3 — additional kernels as needed (ad-hoc)
- Add `F32x16` polyfill if cognitive shaders migrate to f32 for memory pressure
- Add quantised int8 distance kernels for embedding compression (if requested by `bgz-tensor` or `holograph`)
- Add Hamming distance kernel for binary embeddings (if requested by `bge-m3`-style consumers)

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
        // F64x8-chunked path for large tensors handled by ndarray's Zip internals.
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

---

## 8. What this plan asks of the other repos

Nothing structural — only that consumers stay on the stable surface (§5) and report breakage promptly. Specifically:

- **surrealdb**: `idx/trees/vector.rs` should only use `ndarray::hpc::*` items listed in §5. Anything else is a non-stable detail and may break.
- **lance-graph**: cognitive crates should use `heel_f64x8` distance kernels; if a kernel is missing (e.g. Hamming), file an issue here rather than implementing locally.
- **sea-orm**: no direct dep on this fork; touches it only transitively if a consumer uses sea-orm-arrow with `f64` Arrow columns.

---

## 9. Open questions

1. **`F32x16` priority** — is a cognitive shader consumer planning to move to f32? If yes, add to Sprint 3. If no, defer.
2. **Quantised int8 distance kernels** — when do cognitive shaders move to int8 embeddings? Trigger Sprint 3 item when concrete consumer surfaces.
3. **WASM target** — surrealdb has a WASM build path. Does it need `vector-hpc`? Today no; the scalar fallback path covers it. Confirm with surrealdb plan.
4. **Numeric tolerance documentation** — what's the precision guarantee vs scalar reference? Currently "within `f64::EPSILON * len`"; doc-test it in Sprint 0.

---

## 10. Cross-references

- **Glue #1** (surrealdb-ractor): `AdaWorldAPI/surrealdb:.claude/plans/integration-plan.md` §5
- **Glue #2** (TiKV TableProvider): `AdaWorldAPI/lance-graph:.claude/plans/integration-plan.md` §5
- **Glue #3** (sea-orm-ractor): `AdaWorldAPI/sea-orm:.claude/plans/integration-plan.md` §5
- **Glue #4** (cognitive-shader-actor): `AdaWorldAPI/lance-graph:.claude/plans/integration-plan.md` §6
- **Cognitive crate consumers** (the load-bearing reason this fork exists): `AdaWorldAPI/lance-graph:.claude/plans/integration-plan.md` §3 + §4
- **surrealdb's `vector-hpc` feature**: `AdaWorldAPI/surrealdb:.claude/plans/integration-plan.md` §4 (`core/Cargo.toml:71-77`)
