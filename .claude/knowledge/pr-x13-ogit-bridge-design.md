# PR-X13 — `ndarray::hpc::ogit_bridge::*` — embedded TTL → in-memory schema (subsumes PR-Z1/Z2)

> READ BY: savant-architect, ogit-architect, cognitive-architect,
> l3-strategist, sentinel-qa, product-engineer.
>
> Status: design v1 — drafted 2026-05-18 in the master-consolidation arc.
>
> **Subsumes**: PR-Z1 (OGIT Cognitive namespace bootstrap) +
> PR-Z2 (lance-graph CognitiveBridge sibling to MedcareBridge).
>
> **Used by**: PR-X9 (LazyBlockedGrid basin-codebook needs O(1) schema lookup),
> PR-X4 (Gaussian splat cascade may consume family bitmaps).

## Why

PR-X9 (lazy basin-codebook) needs the OGIT Cognitive namespace at startup
for the heel/hip/twig/leaf hierarchy lookup. Three options surfaced:

- **(a)** Sequential: bootstrap OGIT Cognitive → ship lance-graph CognitiveBridge → ship PR-X9 (3 sprints, inter-repo coordination)
- **(b)** Parallel with stubs (Cognitive bridge trait stubbed in ndarray)
- **(c)** **Embedded TTL bundle in ndarray** — bypass lance-graph hop entirely

The master consolidation picks **(c)** because it:
- Removes inter-repo blockers (1 sprint instead of 3)
- Removes the lance-graph-ontology dependency entirely
- Makes the cognitive-shader stack self-contained in ndarray
- Costs ~150 KB of TTL files baked into the binary via `include_bytes!`
- Bardioc REST client integration (for live schema queries) becomes a separate optional follow-on, not a blocker

PR-X13 ships the **embedded-TTL bridge** + ships **the OGIT Cognitive namespace TTL itself** (the PR-Z1 bootstrap content) as build-time-embedded data.

## Module layout — `crate::hpc::ogit_bridge::*`

```
src/hpc/ogit_bridge/
├── mod.rs                  — pub surface + feature gate
├── turtle_parser.rs        — A1: minimal RDF Turtle parser (~250 LoC, no rdflib dep)
├── schema.rs               — A2: OntologySchema in-memory representation
├── cognitive_bridge.rs     — A3: heel/hip/twig/leaf hierarchy + O(1) family lookup
├── assets/                 — A4: embedded TTL files
│   ├── cognitive/
│   │   ├── entities/
│   │   │   ├── Heel.ttl
│   │   │   ├── Hip.ttl
│   │   │   ├── Twig.ttl
│   │   │   ├── Leaf.ttl
│   │   │   ├── CognitiveCell.ttl
│   │   │   ├── SplatCovariance.ttl
│   │   │   └── CognitiveTier.ttl
│   │   └── instances/
│   │       ├── heels/        # 4 seed heels
│   │       ├── hips/         # 8 seed hips
│   │       ├── twigs/        # 3 seed twigs
│   │       └── leaves/       # 4 seed leaves
└── tests/
```

4 workers, each owns one slice.

## Minimal Turtle parser (no rdflib dep)

We don't need a full SPARQL-capable RDF parser. We need to read OGIT TTL files at startup and build an in-memory schema graph. The RDF subset we consume:
- Prefix declarations: `@prefix ogit: <...> .`
- Triple statements: `subject predicate object .`
- Literal types: `xsd:string`, `xsd:long`, `xsd:int`, `xsd:byte`, `xsd:double`, `xsd:base64Binary`
- `rdfs:Class`, `rdfs:subClassOf`, `ogit:scope`, `ogit:parent`, `ogit:allowed`, `ogit:relates`, `ogit:belongs`

That's ~12 token types. A hand-rolled parser is ~250 LoC. Existing `sophia_turtle` works but adds a dep + features we don't use.

```rust
pub struct TurtleLexer<'a> { /* ... */ }

#[derive(Debug)]
pub enum TurtleToken<'a> {
    Iri(&'a str),
    Literal { value: &'a str, datatype: Option<&'a str> },
    Prefix { name: &'a str, iri: &'a str },
    Dot,
    Semicolon,
    Comma,
    OpenBracket,
    CloseBracket,
}

pub struct TurtleParser<'a> { /* ... */ }

impl<'a> TurtleParser<'a> {
    pub fn parse(input: &'a str) -> Result<Vec<Triple<'a>>, TurtleError> { ... }
}

pub struct Triple<'a> {
    pub subject: TripleNode<'a>,
    pub predicate: TripleNode<'a>,
    pub object: TripleNode<'a>,
}
```

Performance target: parse 26 TTL files (~700-900 lines, ~700 triples) in **< 50 ms** at startup. Triples then build the in-memory schema.

## In-memory schema

```rust
pub struct OntologySchema {
    pub namespace: Box<str>,                    // "Cognitive"
    pub entities: HashMap<Box<str>, EntityClass>,  // IRI → class
    pub families: Vec<FamilyBitmap>,            // family ID → bitmap of leaf IRIs
    pub leaf_to_family: HashMap<Box<str>, u32>, // leaf IRI → family ID (O(1) lookup)
    pub heel_count: u8,
    pub leaf_count: u32,
}

pub struct EntityClass {
    pub iri: Box<str>,
    pub label: Box<str>,
    pub parent: Option<Box<str>>,        // rdfs:subClassOf
    pub mandatory: Vec<Property>,
    pub optional: Vec<Property>,
    pub indexed: Vec<Property>,
    pub allowed_relates: Vec<Box<str>>,
    pub allowed_belongs: Vec<Box<str>>,
}

pub struct FamilyBitmap {
    pub family_id: u32,
    pub heel_iri: Box<str>,
    pub hip_iri: Box<str>,
    pub bitmap: BitVec,                   // length = leaf_count; bit i set iff leaf i is in this family
}
```

**O(1) family lookup**: `schema.leaf_to_family[leaf_iri]` returns the family ID; `schema.families[family_id].bitmap.iter_ones()` yields the leaf IRIs in that family. PR-X9's basin-XOR-popcount inner loop iterates only the family bitmap (~16-64 candidates), NOT the full 4096-leaf codebook.

## CognitiveBridge

```rust
pub struct CognitiveBridge {
    pub schema: Arc<OntologySchema>,
    pub codebook: Arc<CamCodebook>,       // built from leaf instances at startup
}

impl CognitiveBridge {
    /// Load the Cognitive namespace from the embedded TTL bundle.
    pub fn load_embedded() -> Result<Self, OgitError> {
        let ttls = embedded::cognitive_ttls();   // include_bytes! at compile time
        let triples = TurtleParser::parse_all(&ttls)?;
        let schema = OntologySchema::from_triples(&triples)?;
        let codebook = CamCodebook::from_leaf_instances(&schema)?;
        Ok(Self { schema: Arc::new(schema), codebook: Arc::new(codebook) })
    }

    /// O(1) basin → family → candidate leaves lookup.
    pub fn family_of(&self, basin_idx: u16) -> &FamilyBitmap {
        let leaf_iri = self.codebook.iri_of(basin_idx);
        let family_id = self.schema.leaf_to_family[leaf_iri];
        &self.schema.families[family_id as usize]
    }

    /// For PR-X9's encoder: given a cell value, find the best basin in O(family_size).
    pub fn nearest_basin(&self, cell_value: u64, hint_basin_idx: u16) -> u16 {
        let family = self.family_of(hint_basin_idx);
        family.bitmap.iter_ones()
            .map(|leaf_idx| {
                let dist = (cell_value ^ self.codebook.atoms[leaf_idx].edge).count_ones();
                (leaf_idx, dist)
            })
            .min_by_key(|&(_, d)| d)
            .map(|(idx, _)| idx as u16)
            .unwrap()
    }
}
```

## Worker decomposition — 4 workers

| Worker | File | Scope | LoC |
|---|---|---|---|
| A1 | `turtle_parser.rs` | Turtle lexer + parser (subset of RDF 1.1 Turtle); ~250 LoC | ~300 |
| A2 | `schema.rs` | `OntologySchema` + `EntityClass` + `FamilyBitmap`; build from triples | ~350 |
| A3 | `cognitive_bridge.rs` | `CognitiveBridge` + `CamCodebook` integration + O(1) family lookup | ~250 |
| A4 | `assets/cognitive/*.ttl` + `embedded.rs` | The 26 TTL files (mirror PR-Z1's spec) + `include_bytes!` wiring | ~50 LoC + 900 lines TTL |

**Sprint composition**: all 4 spawn in parallel. A2 + A3 depend on A1's parser output type, but they can develop against the same parser interface stubbed in advance.

The A4 TTL content mirrors `pr-z1-ogit-cognitive-bootstrap.md` exactly — 26 files: 4 abstract classes (Heel/Hip/Twig/Leaf) + 3 cell carriers (CognitiveCell/SplatCovariance/CognitiveTier) + 19 seed instances (4 heels + 8 hips + 3 twigs + 4 leaves). Validation gate: rdflib 7.6.0 turtle-parses cleanly, 26 ok / 0 bad, ~700-900 triples total.

**Sprint duration**: 1 week with 4-way parallelism.

## API surface for PR-X9 (the consumer)

```rust
// In PR-X9's LazyBlockedGrid::encode_from_dense:
let bridge = CognitiveBridge::load_embedded()?;     // at startup, ~50 ms
for cell in dense_grid.cells() {
    let basin = bridge.nearest_basin(cell.value, cell.hint_basin_idx);
    let delta = cell.value ^ bridge.codebook.atoms[basin].edge;
    // ... rdo loop picks mode (skip/merge/delta/escape) per PR-X12 ...
}
```

PR-X9's encoder uses ONLY `CognitiveBridge::nearest_basin` and `CognitiveBridge::codebook`. The Turtle parser, OntologySchema, and FamilyBitmap internals are not surfaced to PR-X9 consumers.

## Why "embedded TTL" instead of "Bardioc REST client"

| Embedded TTL (PR-X13) | Bardioc REST client |
|---|---|
| Zero-startup loads (50 ms) | Network round-trip per query (~10-50 ms) |
| No runtime dependency | Requires Bardioc server up |
| Schema frozen at compile time | Live schema updates |
| Binary +150 KB | No binary growth |
| Offline-capable | Requires connectivity |

Cognitive shader practice = thousands of basin lookups per cascade tick. Network latency is fatal. **Embedded is the right call for the hot path.**

Bardioc REST client integration could ship later as `ndarray::hpc::ogit_bridge::bardioc::*` (optional feature) for the cold-path schema-management workflows (admin tools, schema versioning). NOT a v1 requirement.

## Cross-references

- `.claude/knowledge/pr-z1-ogit-cognitive-bootstrap.md` — superseded by PR-X13; the TTL content spec stays canonical, just embeds in ndarray instead of bootstrapping in OGIT
- `.claude/knowledge/pr-x9-design.md` — the consumer of `CognitiveBridge`
- `.claude/knowledge/pr-master-consolidation.md` — the strategic frame
- `AdaWorldAPI/OGIT` — upstream ontology spec (we embed a snapshot of NTO/Cognitive/ subset)
- `AdaWorldAPI/lance-graph/crates/lance-graph-ontology/src/bridges/medcare_bridge.rs` — the bridge pattern we mirror (offline / embedded version)

## Open questions (joint savant ruling)

1. **TTL files embedded via `include_bytes!` or `include_str!`?** Lean: **`include_str!`** (TTL is UTF-8 text; `include_str!` lets us format-check at compile time).

2. **Schema rebuild on startup OR serialize an in-memory binary blob?** Lean: **rebuild on startup** for v1 — 50 ms parse + build is fine; binary blob optimization (msgpack or rkyv) is PR-X13.1.

3. **Support OGIT namespaces beyond Cognitive (Healthcare, WorkOrder, Network)?** Lean: **Cognitive only in v1** — the namespace-agnostic API is in place, but only Cognitive ships embedded. Other namespaces add via PR-X13.2 / X13.3 / etc., or via runtime `CognitiveBridge::load_from_disk()`.

4. **OGIT schema snapshot version pinning?** Lean: **embed a git-commit-sha** in the TTL bundle metadata; downstream consumers can verify the embedded schema matches the OGIT upstream commit they expect.

5. **rdflib parity gate?** Lean: **yes** — for the 26 embedded TTL files, our minimal parser MUST produce the same triple count and triple set as rdflib 7.6.0. Bit-exact gate, runs in CI.

6. **`FamilyBitmap` storage: bitvec or `Vec<u16>`?** Lean: **bitvec** for the 4096-leaf case (512 bytes vs 8 KB). Bitvec popcount-iter is fast on AVX-512 via `vpopcntq`.

7. **Should `CognitiveBridge` cache lookups (memoize `nearest_basin`)?** Lean: **no** for v1 — each call is ~16-64 ops (family iteration), no memoization needed. Memoization adds invalidation complexity; revisit if profiling shows a bottleneck.

## Done criteria

- All 4 workers complete
- 26 embedded TTL files parse cleanly via the minimal parser
- rdflib parity gate green (same triple count, same triples)
- `CognitiveBridge::load_embedded()` completes in < 50 ms on Zen4
- `CognitiveBridge::nearest_basin()` finds the correct basin on ≥ 99.5% of synthetic test cases (the 0.5% are ambiguous-family cases that go to escape mode in PR-X9)
- Codex P0 audit (especially SAFETY-claim on `include_bytes!` byte handling + UTF-8 boundary correctness)
- P2 savant SHIP verdict

## Deprecation path

- PR-X13 lands → PR-Z1 (OGIT bootstrap) becomes "future work; the embedded snapshot in ndarray is canonical for v1"
- PR-Z2 (lance-graph CognitiveBridge) becomes "deprecated; superseded by ndarray::hpc::ogit_bridge::CognitiveBridge"
- lance-graph-ontology stays for non-Cognitive namespaces (Healthcare, etc.); cognitive-shader stack no longer depends on it
- If/when live schema updates become a requirement, `bardioc-rs` integration ships as PR-X13.10 or similar

## Forward compatibility

When OGIT NTO/Cognitive/ namespace evolves upstream (new heels, new leaves), an `update-embedded-cognitive-ttls` build script re-downloads and re-embeds:

```bash
cargo run --release -p ndarray-tools --bin update-embedded-cognitive-ttls
# downloads OGIT@latest, validates with rdflib, copies to src/hpc/ogit_bridge/assets/cognitive/
# reports diff: 12 new triples, 0 removed; bumps schema version
```

The embed is regenerated; downstream consumers rebuild against the new schema version. **Zero runtime cost; one rebuild per upstream schema bump.**
