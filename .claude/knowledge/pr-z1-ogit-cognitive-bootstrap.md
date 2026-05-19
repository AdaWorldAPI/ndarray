# PR-Z1 — OGIT NTO/Cognitive/ Namespace Bootstrap (upstream prerequisite for PR-X9)

> READ BY: all agents touching OGIT, lance-graph-ontology, or the cognitive
> shader stack
> (savant-architect, ogit-architect, cognitive-architect, l3-strategist,
> truth-architect, sentinel-qa, product-engineer).
>
> Status: planning doc — drafted in conversation 2026-05-18.
> **This is an OGIT-repo prerequisite for ndarray PR-X9 + lance-graph
> CognitiveBridge work.** Doc lives in ndarray's `.claude/knowledge/`
> because that's where the PR-X9 sprint context lives; the actual TTL
> bootstrap commits go to https://github.com/AdaWorldAPI/OGIT.
>
> Parallel docs:
> - `.claude/knowledge/pr-x9-design.md` — the ndarray consumer that needs this namespace
> - `.claude/knowledge/pr-x4-design.md` — the splat cascade that uses PR-X9's lazy storage
> - `.claude/knowledge/pr-x3-cognitive-grid-design.md` — the BlockedGrid substrate
> - https://github.com/AdaWorldAPI/OGIT/blob/main/.claude/AGENT_LOG.md — the
>   working bootstrap pattern (2026-05-07 Healthcare namespace, 846 TTL lines,
>   14 entities, 690 triples, rdflib-validated) — PR-Z1 mirrors this exactly

## Context for a fresh session

If you arrive after a token reset / handover:

1. **OGIT** at https://github.com/AdaWorldAPI/OGIT is a Turtle (TTL) ontology spec.
   Namespaces live under `NTO/<Namespace>/` (e.g., `NTO/Healthcare/`,
   `NTO/WorkOrder/`, `NTO/Network/`). Each namespace contains
   `entities/<Name>.ttl` files and `enumerations/<enum>.ttl` files.
2. **The Rust consumer is `lance-graph/crates/lance-graph-ontology/`** with
   `OntologyRegistry` + per-namespace `*Bridge` types (e.g., `MedcareBridge`
   for `NTO/Healthcare/`, `NetworkBridge` for `NTO/Network/`). Bridges hydrate
   the TTL files into in-memory graph state at startup.
3. **PR-Z1 (this doc)**: bootstrap `NTO/Cognitive/` — a new namespace defining
   the heel/hip/twig/leaf cognitive abstraction hierarchy + cell-carrier
   entities (CognitiveCell, SplatCovariance, CognitiveTier). PR-Z1 unblocks
   the lance-graph `CognitiveBridge` work (a sibling to `MedcareBridge`),
   which in turn unblocks ndarray PR-X9.
4. **Style template**: `NTO/WorkOrder/entities/Position.ttl v4 baseline` per
   OGIT/.claude/AGENT_LOG.md 2026-05-07. Prefix block → rdfs:Class
   subClassOf ogit:Entity → ogit:scope "NTO" → ogit:parent ogit:Node →
   mandatory/optional/indexed lists → ogit:allowed [ ogit:relates /
   ogit:belongs ] → per-property triples with ogit:type "xsd:...".
   Field predicates camelCase. dcterms:source provenance on every entity.
5. **Validation gate**: `rdflib 7.6.0 turtle-parsed all files cleanly` (the
   2026-05-07 Healthcare gate). PR-Z1 hits the same bar.
6. **Scope**: bootstrap the CLASS HIERARCHY + a SEED set of example leaf
   instances (~10-15), NOT the full 4096-atom CAM codebook. Codebook
   atom enumeration is a follow-up PR (PR-Z1.1) once cognitive shader
   practice surfaces which leaves are actually needed.

## Why this exists

PR-X9 needs `CamCodebook` (4096 BasinAtom entries) materialized at startup.
The basin atoms are NOT arbitrary cluster centroids — they're cognitive
primitives organized in a 4-level abstraction hierarchy:

- **Heel** = root cognitive family anchor (broadest category, ~1-4 instances total)
- **Hip** = sub-family branch (~16 per heel — `4×4` branching factor)
- **Twig** = specific cognitive operation within a hip (~16 per hip)
- **Leaf** = concrete basin atom = actual codebook entry (~16 per twig)

→ Total addressable space: `1 × 16 × 16 × 16 = 4096` leaves (matches the
CAM codebook size exactly, by construction).

This hierarchy lives in OGIT as `rdfs:subClassOf` chains:
`Leaf rdfs:subClassOf Twig rdfs:subClassOf Hip rdfs:subClassOf Heel rdfs:subClassOf ogit:Entity`.

PR-X9's `OgitSchema` walks the chain in O(1) via flat-indexed parent
pointers built at hydrate time. The chain walk gives "what family is this
basin in?" answers without any runtime graph query.

The cognitive cell carrier (`CognitiveCell`) is a separate entity holding
the typed cell state per the design:
- `edge: u64` (CausalEdge64 mantissa)
- `thinking: 32-dim INT4` (16 bytes, base64Binary in TTL)
- `qualia: 16-dim INT4` (8 bytes, base64Binary in TTL)
- `vocab: u16` (CAM codebook index)
- `confidence: f32` (NARS truth projection)

`SplatCovariance` carries the anisotropic per-tier covariance encoding.
`CognitiveTier` carries the L1-L4 tier metadata.

## Files to add (TTL bootstrap, mirrors 2026-05-07 Healthcare pattern)

Branch: `claude/ogit-cognitive-bootstrap-<TOKEN>` (per OGIT branch-policy).

### Class hierarchy (4 abstract classes, ~70 lines each)

```
NTO/Cognitive/entities/
├── Heel.ttl              — abstract root: rdfs:Class subClassOf ogit:Entity
│                          mandatory: name, scope, heelIdx; optional: description
├── Hip.ttl               — rdfs:subClassOf Heel
│                          mandatory: heelParent (FK), hipIdx; optional: description
├── Twig.ttl              — rdfs:subClassOf Hip
│                          mandatory: hipParent (FK), twigIdx; optional: description
└── Leaf.ttl              — rdfs:subClassOf Twig
                           mandatory: twigParent (FK), leafIdx, basinSignature (u64)
                           optional: description
```

The `*Idx` fields are `xsd:byte` (0-255) for compact indexing within a
parent. The `basinSignature` on Leaf is `xsd:long` carrying the
representative CausalEdge64 for this basin (the codebook atom's canonical
truth state).

### Cell carrier entities (3 entities, ~100 lines each)

```
NTO/Cognitive/entities/
├── CognitiveCell.ttl     — rdfs:Class subClassOf ogit:Entity
│                          mandatory: edge (xsd:long), thinking (xsd:base64Binary),
│                                     qualia (xsd:base64Binary), vocab (xsd:int)
│                          optional:  confidence (xsd:double)
│                          allowed:   ogit:relates Leaf (the basin this cell maps to)
├── SplatCovariance.ttl   — rdfs:Class subClassOf ogit:Entity
│                          mandatory: variant (xsd:string, one of
│                                     "isotropic" / "diagonal" / "cholesky"),
│                                     params (xsd:base64Binary, variant-dependent)
│                          optional:  dim (xsd:byte, 2-4)
└── CognitiveTier.ttl     — rdfs:Class subClassOf ogit:Entity
                           mandatory: tierIdx (xsd:byte, 1-4),
                                      blockDim (xsd:int, 64/256/4096/16384),
                                      areaBranch (xsd:byte, 16 for all tiers)
                           optional:  description
                           allowed:   ogit:relates SplatCovariance (per-tier covariance)
```

### Seed Heel instances (4 cognitive families, ~30 lines each)

These are CLASS INSTANCES showing the pattern, NOT exhaustive. The full
4096-leaf catalog is PR-Z1.1.

```
NTO/Cognitive/instances/heels/
├── reasoning.ttl    — Heel: cognitive reasoning operations (deduction, abduction, ...)
├── perception.ttl   — Heel: sensory/input cognitive primitives
├── memory.ttl       — Heel: storage/recall cognitive primitives
└── resonance.ttl    — Heel: field-resonance / cascade cognitive primitives
                      (the NARS-style truth-revision family)
```

### Seed Hip instances (8-16 sub-families, ~30 lines each)

A few per Heel to seed the pattern. Examples under `instances/hips/`:

```
NTO/Cognitive/instances/hips/
├── deduction.ttl       — Hip under reasoning: classical deductive operations
├── abduction.ttl       — Hip under reasoning: best-explanation inference
├── induction.ttl       — Hip under reasoning: generalization operations
├── intuition.ttl       — Hip under reasoning: holistic / fan-out
├── episodic.ttl        — Hip under memory: time-indexed recall
├── semantic.ttl        — Hip under memory: typed entity recall
├── nars_revision.ttl   — Hip under resonance: NARS truth-revision
└── nars_choice.ttl     — Hip under resonance: NARS choice/preference rule
```

### Seed Twig + Leaf instances (~16 total, ~20 lines each)

A few concrete leaves under each seeded Hip to anchor the pattern. The
full enumeration (~4096 leaves) is PR-Z1.1 — too large for a bootstrap.

```
NTO/Cognitive/instances/twigs/
├── modus_ponens.ttl              — Twig under deduction
├── modus_tollens.ttl             — Twig under deduction
└── single_evidence_abduce.ttl    — Twig under abduction

NTO/Cognitive/instances/leaves/
├── classical_mp.ttl              — Leaf under modus_ponens (basinSig = canonical CE64)
├── classical_mt.ttl              — Leaf under modus_tollens
├── single_evidence_warm.ttl      — Leaf under single_evidence_abduce (high-conf variant)
└── single_evidence_cool.ttl      — Leaf under single_evidence_abduce (low-conf variant)
```

### Total file count for the bootstrap

- 4 abstract class TTLs (Heel/Hip/Twig/Leaf)
- 3 cell-carrier TTLs (CognitiveCell, SplatCovariance, CognitiveTier)
- 4 seed Heel instances
- 8 seed Hip instances
- 3 seed Twig instances
- 4 seed Leaf instances

= **26 TTL files**, ~700-900 lines total. Comparable to the 2026-05-07
Healthcare bootstrap (14 entities + 7 enums = 21 files, 846 lines).

## Style notes (mirrors Healthcare bootstrap exactly)

- Prefix block per file:
  ```turtle
  @prefix ogit:           <http://www.purl.org/ogit/> .
  @prefix ogit.Cognitive: <http://www.purl.org/ogit/Cognitive/> .
  @prefix rdf:            <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
  @prefix rdfs:           <http://www.w3.org/2000/01/rdf-schema#> .
  @prefix xsd:            <http://www.w3.org/2001/XMLSchema#> .
  @prefix dcterms:        <http://purl.org/dc/terms/> .
  ```
- `ogit:scope "NTO"` on every entity (consistent with Healthcare / WorkOrder)
- `ogit:parent ogit:Node` for top-level cognitive abstractions
- Field predicates **camelCase** (heelIdx, hipParent, basinSignature, etc.)
- `dcterms:source` provenance on every entity. For PR-Z1 bootstrap files,
  source = `"AdaWorldAPI/ndarray/.claude/knowledge/pr-x9-design.md:layer-1-substrate"`
  (cites the design doc that drove this).
- `ogit:allowed [ ogit:relates Foo ; ogit:belongs Bar ]` for relations.
  Bootstrap relations:
  - `Hip belongs Heel` (heelParent FK)
  - `Twig belongs Hip` (hipParent FK)
  - `Leaf belongs Twig` (twigParent FK)
  - `CognitiveCell relates Leaf` (the basin this cell maps to)
  - `CognitiveTier relates SplatCovariance` (per-tier anisotropic kernel)

## Validation gate (mirrors Healthcare gate)

```bash
# From OGIT repo root, after bootstrap commit
python3 -c "
import rdflib, glob, sys
ok, bad = 0, 0
for f in sorted(glob.glob('NTO/Cognitive/**/*.ttl', recursive=True)):
    g = rdflib.Graph()
    try:
        g.parse(f, format='turtle')
        print(f'OK  {f} ({len(g)} triples)')
        ok += 1
    except Exception as e:
        print(f'BAD {f}: {e}', file=sys.stderr)
        bad += 1
print(f'TOTAL {ok} ok / {bad} bad', file=sys.stderr)
sys.exit(1 if bad else 0)
"
```

Pass criteria: **26 ok / 0 bad**, ~600-900 triples total (the Healthcare
bootstrap hit 690 triples; PR-Z1 should land in 600-900 depending on how
verbose the seed instances are).

## Commit shape (mirrors 2026-05-07 Healthcare commit)

Single commit on `claude/ogit-cognitive-bootstrap-<TOKEN>`:

```
feat(ogit): bootstrap Cognitive namespace — heel/hip/twig/leaf hierarchy + cell carriers (PR-Z1)

D-ids touched: D-OGIT-COGNITIVE-BOOTSTRAP (new); unblocks
lance-graph CognitiveBridge (parallel to MedcareBridge) and
ndarray PR-X9 (LazyBlockedGrid with basin-codebook storage).

Files added (26 TTL, ~700-900 lines, branch
claude/ogit-cognitive-bootstrap-<TOKEN>):

Entities under NTO/Cognitive/entities/:
- Heel.ttl                  — abstract: cognitive family root anchor
- Hip.ttl                   — sub-family branch (16 per Heel target)
- Twig.ttl                  — specific cognitive operation
- Leaf.ttl                  — concrete basin atom (codebook entry)
- CognitiveCell.ttl         — typed cell carrier (edge u64 + thinking
                              [i4;32] + qualia [i4;16] + vocab u16 +
                              confidence f32)
- SplatCovariance.ttl       — anisotropic per-tier covariance encoding
                              (isotropic / diagonal / cholesky variants)
- CognitiveTier.ttl         — L1-L4 tier metadata + area-branch=16

Seed instances under NTO/Cognitive/instances/:
- heels/      (4 files): reasoning, perception, memory, resonance
- hips/       (8 files): deduction, abduction, induction, intuition,
                          episodic, semantic, nars_revision, nars_choice
- twigs/      (3 files): modus_ponens, modus_tollens, single_evidence_abduce
- leaves/     (4 files): classical_mp, classical_mt, single_evidence_warm,
                          single_evidence_cool

Relations:
- Hip belongs Heel (heelParent FK)
- Twig belongs Hip (hipParent FK)
- Leaf belongs Twig (twigParent FK)
- CognitiveCell relates Leaf
- CognitiveTier relates SplatCovariance

Style: matches NTO/WorkOrder/entities/Position.ttl v4 baseline.
Namespace ogit.Cognitive: <http://www.purl.org/ogit/Cognitive/>.
Field predicates camelCase. Provenance on every entity:
dcterms:source "AdaWorldAPI/ndarray/.claude/knowledge/pr-x9-design.md:layer-1-substrate".

Validation: rdflib 7.6.0 turtle-parsed all 26 files cleanly,
26 ok / 0 bad, ~XXX triples total.

Out of scope (deferred):
- Full 4096-leaf basin atom catalog (PR-Z1.1) — bootstrap seeds the
  pattern with 4 leaves; full enumeration awaits cognitive shader
  practice surfacing actual needed basins.
- CognitiveBridge in lance-graph (PR-Z2, sibling to MedcareBridge).
- ndarray LazyBlockedGrid consumer (PR-X9, ndarray repo).
- Per-Hip property extensions (e.g., nars-specific weighting params on
  Hip nars_revision) — bootstrap keeps Hip schema minimal; extension
  via subclassing in follow-up PRs.

Commit: <SHA> feat(ogit): bootstrap Cognitive namespace — heel/hip/twig/leaf
hierarchy + cell carriers (PR-Z1). Not pushed (per branch policy).

Outcome: Cognitive namespace bootstrap complete. lance-graph
CognitiveBridge is unblocked from the OGIT side; ndarray PR-X9 is
unblocked from the OGIT-data side. Awaiting main-thread push and
downstream registry wiring (OntologyRegistry::namespace_id +
CognitiveBridge::new).
```

## Worker decomposition (sequential)

OGIT bootstrap work is small enough for a single Sonnet worker. No
splitting needed (the 14-entity Healthcare bootstrap landed as a single
worker commit in the 2026-05-07 pattern).

| # | Phase | Owns |
|---|---|---|
| 1 | Plan v1 (this doc) | coordinator (ndarray-side) |
| 2 | Plan-review savant (light) | savant agent — verify entity coverage + style match |
| 3 | OGIT bootstrap worker | Sonnet, isolation: worktree (in OGIT repo) — writes 26 TTL files, runs validation gate, commits |
| 4 | Validation savant | savant agent — verify rdflib gate passes + triple count matches |
| 5 | Push + open PR on AdaWorldAPI/OGIT | coordinator |

The OGIT bootstrap worker's prompt should explicitly cite:
- OGIT/.claude/AGENT_LOG.md 2026-05-07 Healthcare entry as the working template
- OGIT/NTO/WorkOrder/entities/Position.ttl as the v4 baseline style reference
- This doc as the entity schema source

## Out of scope (PR-Z1)

1. **Full 4096-leaf catalog** — bootstrap seeds 4 example leaves; full enumeration is PR-Z1.1, driven by what the cognitive shader actually needs in practice
2. **lance-graph CognitiveBridge** — PR-Z2, sibling sprint in the lance-graph repo
3. **ndarray LazyBlockedGrid (PR-X9)** — needs both PR-Z1 (this) and PR-Z2 to land first OR uses the embedded-TTL-bundle escape hatch from PR-X9 Q1 option 3
4. **Per-Hip property extensions** — keeps Hip schema minimal; subclassing for nars-specific / abduction-specific extensions deferred
5. **SHACL shape validation** — Healthcare bootstrap also deferred this; PR-Z1 follows the same precedent
6. **SPARQL query examples** — the ndarray consumer doesn't query OGIT at runtime (data is loaded once at startup), so SPARQL examples are not required for the bootstrap. Add to docs only if cognitive practice demands runtime queries
7. **i18n / multi-language labels** — Healthcare bootstrap is English-only; PR-Z1 follows

## Cross-references

- `.claude/knowledge/pr-x9-design.md` — the ndarray consumer this unblocks (esp. §"Open question Q1")
- `.claude/knowledge/pr-x4-design.md` — the splat cascade that uses PR-X9
- `.claude/knowledge/pr-x3-cognitive-grid-design.md` — the BlockedGrid substrate
- **AdaWorldAPI/OGIT** (https://github.com/AdaWorldAPI/OGIT) — the target repo for PR-Z1
- **AdaWorldAPI/OGIT** `.claude/AGENT_LOG.md` 2026-05-07 entry — the bootstrap template
- **AdaWorldAPI/OGIT** `NTO/Healthcare/entities/Patient.ttl` — concrete style reference for a "complex" entity (166 lines, 142 triples)
- **AdaWorldAPI/OGIT** `NTO/WorkOrder/entities/Position.ttl` — the v4 baseline reference cited in the AGENT_LOG
- **AdaWorldAPI/lance-graph** `crates/lance-graph-ontology/src/bridges/medcare_bridge.rs` — the bridge pattern the future CognitiveBridge will mirror (PR-Z2)

## Open questions (for the plan-review savant)

1. **Heel count** — bootstrap proposes 4 (reasoning, perception, memory, resonance). Should there be more (e.g., affect, intention, attention, embodiment)? Lean: **4 for v1**, extension via sibling Heels in follow-up PRs. The hip/twig/leaf hierarchy fans out enough that 4 heels × 16 hips × 16 twigs × 16 leaves = 16384 ≫ 4096 codebook size, so we have ample room even with just 4 heels.

2. **`basinSignature` storage type** — `xsd:long` (signed 64-bit) is the closest TTL type to u64. The high bit gets interpreted as sign in some RDF libraries. Alternatives: `xsd:unsignedLong` (cleaner semantically but less universally supported), or `xsd:hexBinary` (stores 8 bytes, no signedness issue, but harder to query). Lean: **xsd:long** for v1 (matches Healthcare's use of xsd:long for IDs), document the sign-interpretation footgun.

3. **`thinking` / `qualia` as `xsd:base64Binary` vs split into individual `xsd:byte`?** — base64Binary is compact and matches the storage shape. Individual bytes would make per-dimension queries possible (SPARQL-friendly) but bloat the TTL by ~30×. Lean: **base64Binary** for v1 (ndarray consumer doesn't query individual dimensions at the RDF layer).

4. **`SplatCovariance.params` as `xsd:base64Binary` vs typed variants?** — same trade-off. Lean: **base64Binary** for v1; variant-specific entity subclasses (IsotropicCov, DiagonalCov, CholeskyCov) deferred to PR-Z1.1 if needed.

5. **Should `CognitiveCell` carry `confidence` directly or via a relation to a `NarsTruth` entity?** — direct field is simpler (one less hop). Separate entity would allow richer NARS metadata (frequency + confidence pair, source-lane, time-bucket). Lean: **direct `confidence: xsd:double` for v1** (single projection scalar); full NARS truth carrier as PR-Z1.2 when NARS-rs integration matures.

6. **Bootstrap scope: 4 leaves vs more?** — 4 is the minimum to seed each Hip-class pattern (4 hips touched). Could go higher (~12-16 leaves) to seed more Twigs. Risk: more leaves means more domain decisions baked into bootstrap before cognitive shader practice surfaces real needs. Lean: **4 leaves** (minimum viable), PR-Z1.1 expands as needs surface.

7. **Validation: rdflib 7.6.0 vs newer?** — 2026-05-07 Healthcare used rdflib 7.6.0. Should PR-Z1 use the same for consistency, or upgrade? Lean: **same version** for bit-exact reproducibility of the validation gate. Upgrade in a separate housekeeping PR.

## Done criteria

PR-Z1 is done when:
- All 26 TTL files committed to `claude/ogit-cognitive-bootstrap-<TOKEN>` on AdaWorldAPI/OGIT
- rdflib 7.6.0 validation gate: **26 ok / 0 bad** with ~600-900 triples total
- All entity files match the v4 baseline style (Position.ttl reference)
- All entities carry `dcterms:source` provenance citing this design doc
- AGENT_LOG.md updated with the PR-Z1 entry (mirroring the 2026-05-07 Healthcare entry shape)
- PR opened on AdaWorldAPI/OGIT with the commit message above + a link back to this design doc

## Token-reset safety notes (for fresh sessions)

If you're picking up after a token reset:

1. Read this entire doc first.
2. Read OGIT/.claude/AGENT_LOG.md 2026-05-07 Healthcare entry — that's the
   working template PR-Z1 mirrors.
3. Read OGIT/NTO/WorkOrder/entities/Position.ttl — the v4 baseline style
   reference cited in the AGENT_LOG.
4. The conversation context that led to this doc: after PR #158 (PR-X3) merged
   on 2026-05-18, the cognitive shader stack roadmap surfaced PR-X9 (lazy
   basin-codebook storage). PR-X9 depends on OGIT data for the heel/hip/
   twig/leaf hierarchy. PR-Z1 is the upstream OGIT bootstrap that unblocks
   PR-X9 (or the embedded-TTL-bundle alternative — see PR-X9 Q1).
5. The bootstrap is intentionally MINIMAL — 26 files, ~700-900 lines, ~600-900
   triples — to land quickly and unblock downstream work. Full 4096-leaf
   enumeration is deferred to PR-Z1.1.
6. The 4 heel families (reasoning, perception, memory, resonance) are not
   exhaustive of cognition — they're the minimum viable set to seed the
   hierarchy. Don't argue about completeness in v1; the hierarchy is
   extensible by adding more heels in follow-up PRs.
7. The work is in the OGIT repo, NOT ndarray. The Sonnet worker spawned for
   this should be told explicitly to write to AdaWorldAPI/OGIT, not
   AdaWorldAPI/ndarray, and to follow OGIT's branch policy (not pushed by
   default; awaits main-thread push).
