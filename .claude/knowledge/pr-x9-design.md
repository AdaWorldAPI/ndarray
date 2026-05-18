# PR-X9 — Lazy Splat Cascade with Basin-Codebook + Perturbation Storage

> READ BY: all ndarray agents that touch BlockedGrid, the cognitive shader
> stack, the splat3d cascade, or persistence
> (savant-architect, l3-strategist, cascade-architect,
> splat3d-architect, cognitive-architect, ogit-architect, arm-neon-specialist,
> sentinel-qa, product-engineer, truth-architect, vector-synthesis).
>
> Status: design v1 (drafted in conversation 2026-05-18). PENDS plan-review savant.
> **Drafted as sibling to PR-X4** — savant rules on the (a) fold-into-X4 vs
> (b) ship-as-PR-X9 trade-off as part of the joint plan review.
>
> Parallel docs:
> - `.claude/knowledge/pr-x3-cognitive-grid-design.md` — BlockedGrid dense substrate (shipped at PR #158)
> - `.claude/knowledge/pr-x4-design.md` — Gaussian splat cascade onto BlockedGrid (sibling design)
> - `.claude/knowledge/cognitive-shader-foundation.md` — 7-layer vision
> - `.claude/knowledge/cognitive-distance-typing.md` — no-umbrella distance rule
> - `.claude/rules/data-flow.md` — Rule #3 (no `&mut self` during compute)

## Context for a fresh session

If you arrive after a token reset / new session / handover:

1. **PR-X3 shipped** (PR #158, merged 2026-05-18). `crate::hpc::blocked_grid::*` provides `BlockedGrid<T, BR, BC>` — a **dense** row-major padded grid. L4 (16384²) at u64 cells = 2 GB materialized.
2. **PR-X4 (sibling design)** ships the Gaussian splat cascade as the cognitive spacetime evolution kernel. It uses dense `BlockedGrid<T>` as substrate storage and emits a 4-level `SplatPyramid<T, BR, BC>` per cascade.
3. **PR-X9 (this doc)** keeps the PR-X4 API surface IDENTICAL but swaps the storage substrate from dense `BlockedGrid<T>` to **lazy basin-relative `LazyBlockedGrid<T>`**:
   - Codebook of 4096 `BasinAtom`s, materialized once (~256 KB total)
   - Per-cell δ stored as 8-bit perturbation from nearest basin
   - Skip-mode (δ=0) cells stored as 1 bit in a bitmap
   - Merge-mode cells inherit δ from a neighbor (2-bit direction code)
   - L4 view materializes 10-50 MB instead of 2 GB (200× memory reduction)
4. **OGIT dependency** (CORRECTED 2026-05-18 per https://github.com/AdaWorldAPI/OGIT):
   OGIT is the **Turtle (TTL) ontology specification**, NOT a Rust crate. It defines
   namespaces (NTO/Healthcare, NTO/WorkOrder, NTO/Network, etc.) where each entity
   is a TTL file with `rdfs:Class subClassOf ogit:Entity`, `ogit:scope`, `ogit:parent`,
   mandatory/optional/indexed property lists, and `ogit:allowed [ ogit:relates /
   ogit:belongs ]` relation declarations.
   The **Rust consumer is `lance-graph/crates/lance-graph-ontology/`** (already exists,
   provides `OntologyRegistry` + per-namespace bridges like `MedcareBridge` —
   see OGIT `.claude/AGENT_LOG.md` 2026-05-07 entry for the working pattern that
   bootstrapped Healthcare's 14-entity / 690-triple namespace).
   So PR-X9's actual dependency chain is:
     `ndarray` → `lance-graph-ontology` (via `OntologyRegistry` + new `CognitiveBridge`)
     → OGIT TTL files (loaded at startup or build-time).
   Q1 below covers the 3-repo coordination required. The runtime data structure
   (`CamCodebook` + `OgitSchema`) is materialized once at startup from the bridge's
   hydrate path — NOT a per-query lookup over RDF — so hot-path basin lookup stays
   O(1) over an in-memory index, not O(triple-store-query).
5. **The unification**: lazy basin-relative storage IS x265's coding-tree-unit recursion (CTU/CU/PU/TU + skip/merge/intra/inter modes) applied to a semantic substrate instead of pixel substrate. x265 averages ~4 bits per pixel on HD video despite 8-12 bits raw; this PR targets ~2-8 bits per CausalEdge64 despite 64 bits raw — same ratio, same mechanism, different content.
6. **PR-X5 (queued)**: typed SIMD register-banks. PR-X9 must keep PR-X5's `Fn(StackedU64x8<N>) -> ...` closure boundary identical — materialization happens AT the SIMD-load site (`view.gather_u64x8(row, col)`), not before.

**This PR is PR-X9 only.** PR-X5, PR-X7, W7 are explicit non-goals; the lazy storage is forward-compatible with all three.

## Why this exists — the storage swap

PR-X4 ships the cognitive spacetime cascade with **dense storage**. That works correctness-wise but doesn't scale: an L4 ShaderMantissaGrid is 2 GB at u64 cells, 10 GB once thinking/qualia/vocab fields are added (PR-X7). Realistic cognitive use cases need:
- Thousands of cascades per second (per-tick re-cascade for a streaming inquiry)
- Hundreds of pyramids alive simultaneously (parallel candidate exploration)
- Persistence across ticks (PR-X6 Lance bridge — emit per-L1-block fragments)

Dense storage hits a wall at maybe 10 simultaneous pyramids on a typical 64 GB workstation. Lazy storage breaks the wall: hundreds of simultaneous pyramids fit in the same memory budget because every pyramid SHARES the same immutable codebook + most cells are skip-mode.

The other reason it exists: **ergonomics grow beyond information-theoretic compression bounds when references are factored out across reuse**. JL projection / PolarQuant preserve pairwise distance with distortion bounds but lose location AND can't share information across queries. Basin-codebook lookup gives O(1) "what family is this?" answers that are AS-COMPRESSED-AS-POSSIBLE for the marginal query because the family identity is already free (it's in the shared codebook). Every cascade pays the codebook cost once and rides cheap thereafter.

The trick is precisely what GPU shaders and video codecs already do: don't materialize possible outcomes; encode them as operations over a shared reference set.

## The three-layer decomposition

```
┌─────────────────────────────────────────────────────────────────────┐
│ Layer 1: Immutable substrate (materialized ONCE, shared system-wide) │
│  - CamCodebook:           4096 BasinAtom × 64 B = 256 KB             │
│                           Materialized at startup from the OGIT       │
│                           Cognitive namespace via lance-graph-        │
│                           ontology's CognitiveBridge hydrate path     │
│                           (mirrors MedcareBridge per                  │
│                           OGIT/.claude/AGENT_LOG.md 2026-05-07).      │
│  - OgitSchema:            heel/hip/twig/leaf inheritance DAG          │
│                           Built from OGIT entities'                   │
│                           rdfs:subClassOf chains within the           │
│                           Cognitive namespace. O(1) basin → family    │
│                           / family → basins via flat-indexed maps     │
│                           (NOT a runtime SPARQL query).               │
│  - PerTierCovariance:     4 SPD matrices = ~96 bytes                 │
│  - BasinFamilyBitmaps:    one 4096-bit bitmap per family (~hundreds  │
│                           of families × 512 B = small)                │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 2: Sparse perturbations (the ONLY scaled storage)              │
│  - SparseGrid<Tier> {                                                │
│      basin_idx:    Vec<u16>,         // per-cell basin pointer       │
│      mode:         BitVec,           // 2 bits/cell: skip|merge|δ|reserved│
│      delta:        Vec<u8>,          // 8-bit perturbation (only for │
│                                       δ-mode cells; absent otherwise) │
│      merge_dir:    Vec<u8>,          // 2 bits/cell (only for merge) │
│    }                                                                  │
│  - Per-L1 block: ~10-30 explicit δ + ~70% skip + ~25% merge          │
│  - Per-L4 pyramid: ~10-50 MB total (vs 2 GB dense)                   │
├─────────────────────────────────────────────────────────────────────┤
│ Layer 3: Virtual grid views (NEVER materialized as dense)            │
│  - LazyBlockedGrid<T, BR, BC>:                                       │
│      - holds &CamCodebook + SparseGrid + parent_link                 │
│      - implements GridStorage<T> trait                                │
│      - `gather_u64x8(r, c)` materializes 8 cells on demand           │
│        → exactly enough to load one AVX-512 / SVE register           │
│      - `materialize_dense() -> BlockedGrid<T>` for tests / debug     │
└─────────────────────────────────────────────────────────────────────┘
```

The dense `BlockedGrid<T>` from PR-X3 stays in the codebase — it's the **reference implementation** used in parity tests and for codebook-building (encoder feeds dense input, produces lazy output). The lazy storage is the **production path** for cascade outputs.

## x265-inspired encoding modes

Each cell in a `LazyBlockedGrid` carries a 2-bit mode tag identifying how its value is encoded:

| Mode (2-bit) | Meaning | Storage cost per cell | Decode operation |
|---|---|---|---|
| `00` **Skip** | Cell exactly equals its basin atom; no delta needed | 0 bytes (mode tag only) | `cam[basin_idx]` |
| `01` **Merge** | Cell inherits delta from a neighbor (N/E/W/S, 2-bit direction in `merge_dir`) | 0.25 bytes (2 bits) | `cam[basin_idx] + decode_delta(neighbor.delta)` |
| `10` **Delta** | Cell stores its own 8-bit perturbation in `delta` | 1 byte | `cam[basin_idx] + decode_delta(delta)` |
| `11` **Escape** | Cell stores a full 64-bit value in an escape vector (rare; for outliers) | 8 bytes + 32-bit index | direct from escape vector |

Plus a per-cell `basin_idx: u16` (2 bytes, supports up to 65K codebook atoms; v1 caps at 4096 so the high 4 bits are reserved for tier hint).

**Per-cell average storage estimate** for a coherent cognitive cascade:
- 70% skip → 2 bits + 2 bytes basin_idx = 2.25 bytes
- 25% merge → 2 bits + 2 bits + 2 bytes basin_idx = 2.5 bytes
- 4.5% delta → 2 bits + 1 byte + 2 bytes basin_idx = 3.25 bytes
- 0.5% escape → 2 bits + 8 bytes + 2 bytes basin_idx + 4 bytes escape_idx ≈ 14.25 bytes

Weighted average: `0.70*2.25 + 0.25*2.5 + 0.045*3.25 + 0.005*14.25 ≈ 2.43 bytes per cell` — vs 8 bytes dense = **3.3× compression on a typical cell**. At the pyramid level (including codebook overhead amortized once), realistic compression ratio is **~10-50× per simultaneous pyramid** because the codebook + schema are shared across all pyramids in the system.

Worst case (incoherent / random state): 95% delta + 5% escape → ~4 bytes per cell, still 2× over dense. **No regression vs dense** even for adversarial inputs.

### Cognitive RDO (rate-distortion optimization)

The encoder picks the mode that minimizes `storage_bits × λ × ε_truth_loss`. For cognitive content:
- ε_truth_loss for skip-mode = 0 (exact match) — always pick skip when possible
- ε_truth_loss for merge-mode = 0 if neighbor's delta produces same value (rare exact match)
- ε_truth_loss for delta-mode = `|true_value - basin - decode(quantized_delta)|`
- ε_truth_loss for escape-mode = 0 (lossless)

λ = NARS confidence weight. High-confidence cells (`f ≥ 0.95`) tolerate less ε; low-confidence cells (`f ≤ 0.7`) tolerate more. The RDO loop sweeps λ to fit within a storage budget.

This mirrors x265's λ-RDO loop exactly. We can borrow x265's λ tables as starting heuristics and tune via training feedback.

## The `GridStorage` trait — polymorphic substrate

PR-X4's `compose_l1` / `compose_cascade` currently return `BlockedGrid<T>` directly. PR-X9 introduces a `GridStorage` trait that BOTH `BlockedGrid<T>` (dense) AND `LazyBlockedGrid<T>` (basin-relative) implement. Callers pick storage; the cascade API is identical.

```rust
/// A 2-D block-padded grid storage backend. Implemented by both the dense
/// `BlockedGrid<T>` (PR-X3) and the lazy `LazyBlockedGrid<T>` (PR-X9).
///
/// Callers that don't care about storage shape parameterize over `S: GridStorage<T>`.
pub trait GridStorage<T: Copy> {
    /// Block dimensions (compile-time const) — must match between caller
    /// expectation and the underlying grid.
    const BR: usize;
    const BC: usize;

    /// Logical extent (runtime).
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn padded_rows(&self) -> usize;
    fn padded_cols(&self) -> usize;

    /// Read a single cell. Always materializes on demand for lazy storage.
    fn get(&self, row: usize, col: usize) -> T;

    /// Gather 8 consecutive cells (one AVX-512 / SVE register). Fast path for
    /// SIMD kernels. Lazy storage materializes the 8 cells inline via codebook
    /// + perturbation lookup — never touches dense memory.
    ///
    /// Requires `col + 8 <= padded_cols()`. Returns the 8 cells as an array.
    fn gather_u64x8(&self, row: usize, col: usize) -> [u64; 8]
    where
        T: Into<u64> + Copy;

    /// Iterate base blocks (read-only). Returns an iterator yielding lightweight
    /// `BlockView<T>` handles. Materialization is per-block on demand.
    type BaseBlockIter<'a>: Iterator<Item = BlockView<'a, T, { Self::BR }, { Self::BC }>>
    where
        Self: 'a;
    fn blocks_base(&self) -> Self::BaseBlockIter<'_>;

    /// Materialize the entire grid as a dense `BlockedGrid<T>` — escape hatch
    /// for tests, debugging, and dense-vs-lazy parity gates. Linear in cell
    /// count; never call on hot paths.
    fn materialize_dense(&self) -> BlockedGrid<T, { Self::BR }, { Self::BC }>;
}

impl<T, const BR, const BC> GridStorage<T> for BlockedGrid<T, BR, BC> { /* trivial: existing API */ }
impl<T, const BR, const BC> GridStorage<T> for LazyBlockedGrid<T, BR, BC> { /* basin lookup */ }
```

PR-X4's `SplatPyramid<T, BR, BC>` becomes `SplatPyramid<T, S: GridStorage<T>, BR, BC>` where the storage shape is plug-replaceable. Production code picks `LazyBlockedGrid<T>`; tests pick `BlockedGrid<T>` for parity verification.

## `LazyBlockedGrid<T>` — the lazy storage type

```rust
pub struct LazyBlockedGrid<'a, T, const BR: usize = 64, const BC: usize = 64> {
    // Immutable refs to the shared substrate
    codebook: &'a CamCodebook,
    schema: &'a OgitSchema,
    tier_cov: &'a [TierCovariance; 4],

    // Per-grid sparse storage
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
    sparse: SparseGrid<BR, BC>,

    // Optional parent-tier link for cascade inheritance (None for L4 root)
    parent: Option<&'a LazyBlockedGrid<'a, T, BR, BC>>,

    _marker: PhantomData<T>,
}

pub struct SparseGrid<const BR: usize, const BC: usize> {
    basin_idx: Vec<u16>,       // length = n_block_rows * n_block_cols * BR * BC
                                // packed row-major (NOT block-major) for stride-friendly scan
    mode: BitVec,              // 2 bits/cell
    delta: Vec<u8>,            // dense-packed for δ-mode cells only
    merge_dir: BitVec,         // 2 bits/cell for merge-mode cells only
    escape: Vec<u64>,          // overflow values for escape-mode cells
    escape_idx: Vec<u32>,      // index into `escape` for each escape cell
}

pub struct CamCodebook {
    atoms: Vec<BasinAtom>,     // 4096 entries
    family_membership: Vec<u16>, // atom → family index (OGIT schema link)
}

pub struct BasinAtom {
    edge: u64,                  // canonical CausalEdge64 for this basin
    thinking: [u8; 16],         // canonical thinking-style vector (PR-X7)
    qualia: [u8; 8],            // canonical qualia (PR-X7)
    vocab: u16,                 // canonical vocab index
    confidence_floor: f32,      // NARS truth floor for this basin
    _pad: [u8; 4],              // align to 40 bytes
}

/// Construction: encode a dense BlockedGrid into a lazy one.
impl<'a, T: Copy, const BR: usize, const BC: usize> LazyBlockedGrid<'a, T, BR, BC> {
    /// Encode a dense grid using the supplied codebook. Each cell is matched
    /// to its nearest basin via OGIT schema (O(log basin_count) per cell), and
    /// the encoder picks skip/merge/delta/escape mode per the RDO loop.
    pub fn encode_from_dense(
        dense: &BlockedGrid<T, BR, BC>,
        codebook: &'a CamCodebook,
        schema: &'a OgitSchema,
        tier_cov: &'a [TierCovariance; 4],
        parent: Option<&'a LazyBlockedGrid<'a, T, BR, BC>>,
        rdo: RdoConfig,
    ) -> Self { ... }

    /// Decode the entire grid back to dense. Used by `GridStorage::materialize_dense`
    /// and by the dense-vs-lazy parity gate. Linear in cell count.
    pub fn decode_to_dense(&self) -> BlockedGrid<T, BR, BC>
    where
        T: From<u64>,
    { ... }
}

pub struct RdoConfig {
    /// Lagrange multiplier for the bit-vs-error trade-off. Higher = prefer
    /// fewer bits at cost of more truth loss. Default 1.0 (= x265 medium preset).
    pub lambda: f32,
    /// Maximum allowed quantization error per cell (NARS truth distance).
    /// Cells exceeding this go to escape mode regardless of bit cost.
    pub epsilon_floor: f32,
    /// Whether to enable merge mode (slightly slower encode but lower bit cost).
    pub allow_merge: bool,
}
```

## Migration path — PR-X4 dense → PR-X9 polymorphic

After PR-X4 lands with dense storage, PR-X9 migrates each call site to be storage-polymorphic via `GridStorage<T>`. Existing tests stay; they parameterize over `S = BlockedGrid` for the dense path. New tests add `S = LazyBlockedGrid` for the lazy path. Parity gates compare the two paths cell-by-cell.

The migration is non-breaking for existing callers because `BlockedGrid<T>` still implements `GridStorage<T>` trivially — old code keeps using dense storage with no change.

## Layering rule (still binding)

PR-X9 is **pure storage + encoding**. It contains:
- ZERO `#[target_feature]` attributes
- ZERO `use crate::simd_avx512` / per-arch imports
- ZERO `cfg(target_feature = ...)` gates
- ZERO raw `_mm*` / `vld*` / `_pdep_*` intrinsics
- ZERO distance-aware API surface (basin matching uses OGIT-rs schema lookup, NOT a distance metric — see Q3 below)

The `gather_u64x8` fast path materializes 8 cells via scalar codebook+perturbation lookups; PR-X5's typed SIMD register-banks will pick up these gathered cells from `crate::simd::*` primitives.

## Distance-typing guardrail

Basin matching during `encode_from_dense` is **OGIT-schema-driven**, NOT distance-metric-driven. The OGIT semantic schema provides O(1) family lookup; the encoder walks the family bitmap to find the basin with minimum delta (a single u64 XOR + popcount per candidate, NOT a generic distance call). This:
- Stays within the no-umbrella-distance rule (`.claude/knowledge/cognitive-distance-typing.md`)
- Avoids `Box<dyn Distance>` indirection
- Is O(log basin_count) per cell instead of O(basin_count) — the OGIT schema is the index

If a use case demands a custom basin-matching predicate, it's a closure parameter to `encode_from_dense`, NOT a trait method on `LazyBlockedGrid`. The closure boundary IS the dispatch.

## Tests required

### Unit tests for `LazyBlockedGrid<T>`

- **Encode-decode round-trip**: random `BlockedGrid<u64, 64, 64>` → encode → decode → must equal original within `epsilon_floor` per cell (lossy by RDO design; bit-exact only when ε=0)
- **Skip-mode dominance**: a grid where every cell matches a codebook atom exactly → encodes with 100% skip cells, 0 bytes of delta storage
- **Merge-mode opportunism**: a grid with adjacent cells sharing the same δ → encodes with merge mode for the trailing cell, saves bits
- **Escape-mode safety**: a grid with outliers exceeding `epsilon_floor` → encodes those cells as escape, no truth loss
- **`gather_u64x8` correctness**: materializes the same 8 cells as 8 individual `get()` calls
- **`materialize_dense` parity**: produces a `BlockedGrid<T>` cell-equal to the lazy-encoded source (within ε)
- **Compression ratio**: for a coherent test grid (70% skip / 25% merge / 4.5% delta / 0.5% escape), produced byte-size ≤ 0.5× the dense size (2× compression target verified)

### Property tests (proptest-style)

- For any `BlockedGrid<u64, 64, 64>` and any `RdoConfig`, `encode → decode` produces a grid where the per-cell L1 distance to the source ≤ `epsilon_floor`
- For any two `BlockedGrid<u64>` differing in K cells, the encoded `LazyBlockedGrid` differ in O(K) bits (linearity property)
- `gather_u64x8(r, c)` ≡ `(0..8).map(|i| get(r, c+i))` for all valid `(r, c)`

### Integration tests

- PR-X4 splat cascade: `compose_l1` with `S = LazyBlockedGrid` produces same pyramid as `S = BlockedGrid` modulo ε per cell (the parity gate)
- W4 `bulk_apply` over a `LazyBlockedGrid`: the iterator yields valid cells, mutation visible after re-read
- Memory budget gate: encoding a 16384×16384 u64 grid produces a `LazyBlockedGrid` with `total_size() < 50 MB` under default RDO config (verified via `std::mem::size_of_val` over the sparse vectors)

## Out of scope — explicitly NOT in PR-X9

1. **Typed SIMD register-banks** (`StackedU64x8<N>`, `AmxBf16Tile`) → PR-X5
2. **The `cognitive_shader!` typed cell-DSL** → PR-X7
3. **NARS truth-revision blend kernel** → W7
4. **Direct GPU dispatch** — v1 is CPU only via `crate::simd::*`
5. **Codebook learning** — v1 assumes the codebook is built ahead-of-time and frozen. Online codebook learning (additions / refinements / merges) → PR-X9.1 if needed
6. **Multi-codebook switching** — v1 has one global `CamCodebook` per `LazyBlockedGrid`. Per-tier or per-domain codebooks → PR-X9.2
7. **Lance persistence** — v1 keeps `LazyBlockedGrid` in-memory only. Lance fragment emit (per-L1-block) → PR-X6 (already drafted as separate roadmap item)
8. **`splat3d::ply` loader** — out of scope; PR-X4 handles splat3d compat
9. **Adversarial / privacy basin lookup** — out of scope

## Worker decomposition (SEQUENTIAL — binding protocol)

Same sequential 4-6 Sonnet + 1 Opus coordinator pattern as PR-X3 / PR-X4.

### File layout

```
src/hpc/blocked_grid/
├── storage.rs        — A1: `GridStorage<T>` trait + impl for BlockedGrid (re-export)
└── ... (existing PR-X3 files unchanged)

src/hpc/lazy_grid/    — NEW directory
├── mod.rs            — coordinator: submodule decls + re-exports
├── codebook.rs       — A2: `CamCodebook` + `BasinAtom`
├── sparse.rs         — A3: `SparseGrid<BR, BC>` + mode encoding
├── lazy.rs           — A4: `LazyBlockedGrid<T, BR, BC>` + `GridStorage` impl
├── encode.rs         — A5: `encode_from_dense` + RDO loop
└── tests.rs          — A6: parity tests + property tests + memory-budget gate
```

### Worker phases

| # | Phase | Owns | Depends on |
|---|---|---|---|
| 1 | Plan v1 (this doc) | coordinator | PR-X3 + PR-X4 |
| 2 | Plan-review savant | savant agent | this doc + sibling PR-X4 doc |
| 3 | Plan v2 corrector | coordinator | savant verdict (joint with PR-X4) |
| 4 | Worker A1 (storage.rs) | `GridStorage<T>` trait + impl for BlockedGrid | PR-X3 |
| 5 | Worker A2 (codebook.rs) | `CamCodebook` + `BasinAtom` + OGIT schema bridge | OGIT-rs API (see Q1) |
| 6 | Worker A3 (sparse.rs) | `SparseGrid<BR, BC>` + 2-bit mode + escape vector | A1 |
| 7 | Worker A4 (lazy.rs) | `LazyBlockedGrid` + `GridStorage` impl + `gather_u64x8` | A1, A2, A3 |
| 8 | Worker A5 (encode.rs) | `encode_from_dense` + RDO loop + mode picker | A4 |
| 9 | Worker A6 (tests.rs) | Parity gate vs dense, property tests, memory-budget | A5 |
| 10 | Codex P0 audit (with SAFETY-claim gate per PR-X3.1 backlog) | codex agent | A1-A6 combined |
| 11 | Coordinator fix P0s | coordinator | audit verdict |
| 12 | P2 savant pre-merge | savant agent | post-P0 branch |
| 13 | Coordinator apply tightenings | coordinator | P2 verdict |
| 14 | Merge ladder | — | — |

**Parallelism**: A2 (codebook) and A3 (sparse) can spawn in parallel after A1 lands — different files, no type dependencies between them (both use only A1's `GridStorage` trait). A4 needs A1+A2+A3.

## Verification commands

```bash
cargo check -p ndarray --no-default-features --features std,lazy-grid
cargo test -p ndarray --lib --no-default-features --features std,lazy-grid hpc::lazy_grid
cargo test --doc -p ndarray --no-default-features --features std,lazy-grid hpc::lazy_grid
cargo fmt --all -- --check
cargo clippy -p ndarray --no-default-features --features std,lazy-grid -- -D warnings
```

All five must pass green.

## Cross-references

- `.claude/knowledge/pr-x3-cognitive-grid-design.md` — BlockedGrid dense substrate
- `.claude/knowledge/pr-x4-design.md` — Gaussian splat cascade (sibling design; uses `GridStorage<T>` polymorphically post-PR-X9)
- `.claude/knowledge/cognitive-shader-foundation.md` — 7-layer cognitive shader vision
- `.claude/knowledge/cognitive-distance-typing.md` — no-umbrella distance rule (binding here too — basin matching via OGIT schema, NOT a distance metric)
- `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a layering (binding)
- `.claude/rules/data-flow.md` — Rule #3 (binding)
- `src/hpc/blocked_grid/*` — PR-X3 substrate (used as both dense reference and `GridStorage` impl)
- **AdaWorldAPI/OGIT** (https://github.com/AdaWorldAPI/OGIT) — Turtle ontology
  spec; PR-X9 hydrates the Cognitive namespace into `CamCodebook` + `OgitSchema`
  at startup. See OGIT `.claude/AGENT_LOG.md` 2026-05-07 entry for the
  bootstrap pattern (Healthcare namespace, 14 entities, 690 triples) PR-X9's
  Cognitive namespace mirrors.
- **AdaWorldAPI/lance-graph** `crates/lance-graph-ontology/` — the Rust consumer
  of OGIT. Provides `OntologyRegistry::namespace_id` + per-namespace `*Bridge`
  pattern (e.g. `MedcareBridge`). PR-X9's `CognitiveBridge` is a sibling to
  `MedcareBridge`. See Q1 for the 3-repo coordination plan.
- x265 source for reference: x265's `Mode::set*` family, `analyseLayout()` quad-tree split, RDO loop in `analyse.cpp`

## Open questions (for the plan-review savant)

1. **3-repo coordination: OGIT + lance-graph-ontology + ndarray** —
   OGIT is the Turtle ontology spec (https://github.com/AdaWorldAPI/OGIT). Rust
   consumption already exists via `lance-graph/crates/lance-graph-ontology/` with
   the `OntologyRegistry` + per-namespace `*Bridge` pattern. PR-X9 needs:

   **Prerequisite A** (OGIT repo): a `Cognitive` namespace under `NTO/Cognitive/`
   defining the basin atoms (heel/hip/twig/leaf entities, CausalEdge64 carriers,
   tier-covariance literals). ~14-30 TTL entity files following the
   `NTO/WorkOrder/entities/Position.ttl v4 baseline` style. Mirrors the
   2026-05-07 Healthcare bootstrap (846 lines, 14 entities, 690 triples).
   Probably its own small PR against OGIT; ~1 sprint session.

   **Prerequisite B** (lance-graph repo): a `CognitiveBridge` sibling to
   `MedcareBridge` in `lance-graph-ontology/src/bridges/`, plus the
   namespace registration in `OntologyRegistry::namespace_id`. ~1 sprint
   session.

   **PR-X9 (this repo)** then consumes `lance-graph-ontology` and hydrates the
   `CamCodebook` + `OgitSchema` once at startup.

   The 3-repo dependency means PR-X9 cannot start until A and B land. Options:
   - **Sequential**: ship OGIT/Cognitive → ship lance-graph CognitiveBridge →
     ship PR-X9. Clean but slow (3 sprints).
   - **Parallel with stubs**: PR-X9 ships a `CognitiveBridge` stub trait in
     ndarray itself; OGIT/Cognitive + lance-graph CognitiveBridge ship in
     parallel; the wire-up happens in a final integration PR. Faster but more
     coordination overhead.
   - **Embedded TTL bundle**: ndarray ships the OGIT Cognitive TTL files
     directly as a build-time embedded resource + a tiny TTL parser. Bypasses
     the lance-graph hop entirely. Simplest for v1 but duplicates the
     hydrate path — savant should rule on whether this violates the
     "single source of truth" intent of OGIT + lance-graph-ontology.

   Lean: **embedded TTL bundle for v1** (no inter-repo blocker), document
   the migration path to lance-graph-ontology integration as PR-X9.1.

   Coordinator: verify OGIT repo's NTO/Cognitive/ status before sprint kickoff.
   If Cognitive namespace doesn't exist, bootstrap it FIRST (mirroring the
   Healthcare pattern in the AGENT_LOG).

2. **PR-X4 fold-in vs sibling PR-X9** — savant rules on the trade-off:
   - **(a) Fold into PR-X4**: single sprint, basin-relative from day one, but PR-X4 worker count balloons from 5 to ~10 and risks scope creep
   - **(b) Sibling PR-X9** (this doc): ship PR-X4 dense first, swap to lazy via `GridStorage` trait in PR-X9. Easier correctness verification (parity gate), but two storage paths during the interim
   - Recommendation in conversation: **(b)**. Savant overrides if disagreement.

3. **Basin matching: closure or trait method?** — the encoder needs to find the nearest basin for each cell. Options:
   - Closure parameter: `encode_from_dense(... , basin_matcher: impl Fn(T) -> u16)` — most flexible, no umbrella concern
   - Trait method on `BasinAtom`: `impl BasinAtom { fn matches(&self, value: T) -> bool }` — clean but feels close to a distance-metric umbrella
   - OGIT-schema-direct: `OgitSchema::nearest_basin(value)` — best, leverages the index, no umbrella
   - Lean: **OGIT-schema-direct** with closure-parameter escape hatch for custom predicates.

4. **2-bit mode tag vs 4-bit?** — 2-bit gives 4 modes (skip / merge / delta / escape). 4-bit gives 16 modes, leaving room for tier-specific encodings or sub-mode variants (e.g., merge-N vs merge-NE vs merge-NESW-quad). Lean: **2-bit for v1**; 4-bit deferred to PR-X9.3 if cognitive practice demands.

5. **`epsilon_floor` per-cell vs global?** — global `epsilon_floor` is simpler but lossy on outliers (forces escape mode often). Per-cell ε would allow cognitive RDO to spend more bits on high-confidence cells. Lean: **global for v1** with the per-cell variant as an opt-in mode flag in `RdoConfig` for future cycles.

6. **`materialize_dense` allocates a fresh `Vec<T>`** — escape hatch for tests. Should it be `#[cfg(test)]`-only, or `pub` for debug tooling? Lean: `pub` with a `# Footgun` doc note (it's O(cells) and defeats lazy storage) — analogous to PR-X3's `as_padded_slice` footgun.

7. **Sparse `Vec<u16>` basin_idx vs `Vec<u8>` with codebook ≤ 256?** — if the codebook fits in 256 atoms (very tight, probably too small for cognitive use cases), basin_idx becomes 1 byte. Default is u16 (4096 atoms). Make codebook size a const generic? Lean: **default u16, expose const-generic over codebook_size_log2 as future cycle (PR-X9.4)**.

## Done criteria

PR-X9 is done when:
- All worker spec items implemented per the 6-worker decomposition (A1-A6)
- Codex P0 audit passes with 0 P0 — including the SAFETY-claim verification gate added per PR-X3.1 backlog
- `cargo check / test --lib / test --doc / fmt / clippy` all green with `--features std,lazy-grid`
- Layering rule verified (zero per-arch imports / target_feature / raw intrinsics in `src/hpc/lazy_grid/`)
- Distance-typing guardrail verified — basin matching via OGIT schema, NO `Box<dyn Distance>`, NO `enum DistanceMetric`
- Dense-vs-lazy parity gate passes: `encode → decode → equals original within ε` for all property-test inputs
- Memory-budget gate passes: 16384×16384 u64 encoding fits within 50 MB at default RDO config
- P2 savant pre-merge review delivers SHIP verdict
- PR description includes the x265-cascade analogy + skip/merge/delta/escape mode table so downstream agents understand the encoding semantics

## Token-reset safety notes (for fresh sessions)

If you're picking up after a token reset:

1. Read this entire doc first.
2. Read `pr-x4-design.md` next — the splat cascade that uses this storage substrate.
3. Read `pr-x3-cognitive-grid-design.md` — the dense substrate `LazyBlockedGrid` parallels.
4. The conversation context that led to this doc: after PR #158 (PR-X3) merged on 2026-05-18, the user observed that the cognitive cascade L1-L4 propagation (64→256→4096→16384) should be zero-copy via basin-relative storage rather than materializing dense grids. The mechanism is precisely x265's coding-tree-unit recursion + skip/merge/intra/inter modes, applied to a semantic codebook substrate (OGIT-rs CAM) instead of a pixel substrate. The "ergonomics grow beyond information-theoretic compression bounds" claim is real: it's amortization across reuse, not violation of Shannon. The codebook is paid once and rides cheap for every subsequent query — same trick as GPU shaders not buffering all spacetime outcomes, same trick as x265 not storing every frame.
5. Check `git log --oneline -10` on the PR-X9 branch and on `master`.
6. **The OGIT dependency is a 3-repo coordination, not a missing crate.** OGIT is the
   Turtle ontology at https://github.com/AdaWorldAPI/OGIT. The Rust consumer pattern
   already exists at `lance-graph/crates/lance-graph-ontology/` with `OntologyRegistry`
   + per-namespace `*Bridge`. The blockers (in order):
   (a) OGIT repo needs an `NTO/Cognitive/` namespace bootstrap (mirroring Healthcare
       2026-05-07, ~14 TTL files defining basin atoms);
   (b) lance-graph repo needs a `CognitiveBridge` sibling to `MedcareBridge`;
   (c) THEN PR-X9 wires the hydrate path in ndarray.
   For v1, leaning toward an **embedded TTL bundle** in ndarray that bypasses
   lance-graph (simpler), with migration to the proper bridge path as PR-X9.1.
   Savant rules in Q1.
7. The dense storage `BlockedGrid<T>` (PR-X3) stays in the codebase as both the parity-test reference and the `GridStorage<T>` trivial impl. PR-X9 doesn't deprecate PR-X3; it joins it via the trait.
