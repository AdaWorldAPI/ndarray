# GUID Prefix → Shape Routing — the GridLake continuation (crystallization)

> **READ BY:** savant-architect, vector-synthesis, product-engineer, any agent
> touching `simd_soa.rs`, `blocked_grid/`, `splat3d/`, `cam_pq`, or proposing
> a routing/dispatch surface.
>
> **Date:** 2026-06-10. **Canon source:** `OGAR/CLAUDE.md` (the operator-pinned
> canonical GUID; auto-loaded there, cited here — do not fork the definition).
> **Evidence discipline (per blackboard):** L0 receipts are cited by path;
> everything not yet coded is marked **CONJECTURE** with a named probe, per the
> insight-update cycle. No unmarked conjectures.

## 1. The canon this doc serves (cited, not redefined)

From `OGAR/CLAUDE.md` P0 — counted in HEX; the UUID dash-groups ARE the
semantic delimiters:

```
xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
classid    HEEL   HIP    TWIG   family-basin-leaf(6)+identity(6)
8 hex      4      4      4      12 hex
```

- **Key-of-key-value:** node = 4096 bits = `key(128/GUID) + value(3968)`.
  The key prerenders/routes/compares with **zero value decode**; the value
  compresses freely (Lance) — compression never costs addressability.
- **3×4 uniform:** 3 tiers × 4 nibbles; `tier_of(nibble) = nibble >> 2` —
  a shift, never a branch/divide. RFC 9562 is a *wrapper* concern; wrappers
  adapt at their membrane, never the canon.
- **Centroid-tile reading [H]:** each tier's 4 nibbles = a 256×256 centroid
  tile (two byte-axes); path (HEEL+HIP+TWIG) = **6 bytes = the CAM-PQ 6×256
  code** — path distance = 3 tier-LUT lookups, O(1). Rigor condition:
  codebooks built as **4-level 4-ary hierarchies (256 = 4⁴)** so nibble
  prefixes = centroid ancestry. Codebooks are **scoped by class routing
  prefix** (longest-prefix wins).

## 2. The split — ndarray is MECHANISM, never policy

Same division as `simd.rs` dispatch and the W1a contract: ndarray ships the
*routing mechanism*; the consumer (lance-graph) registers the *table*.
ndarray never learns what a class, DOLCE category, or codebook MEANS.

Receipts for why this layering is already the house style:
- `src/simd_soa.rs:14-27` — `MultiLaneColumn` is **layout-only**: "No
  `#[target_feature]` … No distance-aware API" (the no-umbrella rule,
  `cognitive-distance-typing.md`).
- Blackboard "Settled architecture": distance lives in `hpc::cam_pq`
  (integer ADC) gated by `distance::similarity_z` — **the router must not
  duplicate or wrap it**.
- `CLAUDE.md` architecture rule: "ndarray = hardware (SIMD …), lance-graph
  = thinking."

## 3. Proposed surface — `PrefixShapeTable` (CONJECTURE — not yet coded)

Sibling of `MultiLaneColumn` in the `simd_soa` layer. Layout-only carrier +
two methods; lanes go through `crate::simd::*` per W1a.

```rust
/// Opaque to ndarray — the consumer's shape registry key.
pub struct ShapeId(pub u16);

pub struct PrefixShapeEntry { pub prefix: u64, pub nibbles: u8, pub shape: ShapeId }

/// classid-byte direct index → sorted prefix runs (longest-prefix wins).
pub struct PrefixShapeTable { /* [256] heads + sorted &[PrefixShapeEntry] */ }

impl PrefixShapeTable {
    pub fn route(&self, key: &[u8; 16]) -> ShapeId;
    pub fn route_batch(&self, keys: &[[u8; 16]], out: &mut [ShapeId]);
}
```

**Cheapness budget (the whole point):**
- `route` = one u32 load (GUID group 1) → 256-way direct index (2 KB head
  table, L1-resident) → ≤1 binary-search step in a short sorted run.
  Worst-case full `entity_type` table: 65,536 × u16 = **128 KB = L2-resident**.
- `route_batch` = SoA gather: classid bytes via `U8x64`-width loads, one
  table gather per lane group. No alloc, no `&mut self` (data-flow rule 1+2).
- Tier selection downstream of routing: `nibble >> 2` — already canon.
- The router returns `ShapeId` and STOPS. Distance/semantics stay in
  `hpc::cam_pq` / consumer land (no-umbrella).

**PROBE-ROUTE-1 (pass/fail):** `route_batch` ≡ scalar `route` on 10⁶ random
keys (parity), and ≥4× scalar throughput at N=1024 on the v4 host.

## 4. GridLake, continued — the key selects the grid; the value stays one byte-store

The column-substrate identity is load-bearing and already written
(`hhtl-gridlake-pre-sprint-prompt.md`): **Lance column ≡ Arrow buffer ≡
ndarray SoA — same bytes, different lane width, no copy.** `MultiLaneColumn`
is the value plane. This crystallization adds the key plane:

```
NodeGuid (128b) ──route()──► ShapeId ──consumer registry──► which MultiLaneColumn
       │                                                     family + lane width
       └─ tier nibbles (>>2) ──► pyramid LEVEL within the blocked grid
```

- Grid stack points that already exist (L0): `hpc/blocked_grid/{compute,
  super_block,aliases}.rs`, `hpc/splat3d/depth_cascade.rs`,
  `hpc/pillar/hhtl_contraction.rs`.
- **Stacked-pyramid perturbation (CONJECTURE):** pyramid levels are
  *generated, not stored* — `level(L) = perturb(level(L-1)) + residual(L)`,
  residuals living in the value plane; the KEY's tier nibbles select L.
  This is the immaterialized-cascade doctrine (OGAR D-IMMAT) landing on
  `blocked_grid`. **PROBE-PYR-1:** reconstruct a level-L tile from parent +
  residual **byte-exact**; corrupt-residual must fail loudly.

## 5. The φ-quorum — so Morton cheapness never becomes eigenvalue theater

**Eigenvalue theater, defined by this repo's own casebook**
(`pp13-brutally-honest-tester-verdict.md`): cheap arithmetic wearing
spectral/metric language it does not earn —
- P0-1: a PSD gate **structurally unsatisfiable** at its constants
  (contractive cascade → denormals; absolute ε vs relative needed);
- P0-2: thresholds "chosen on optimism, not measured";
- P0-3: placeholder thresholds **enforced** as PASS gates;
- P0-4: "verified" claims whose level-4 test was never run;
- P1-1: `nearest_basin` ordering by **raw XOR-u64** instead of popcount —
  "a function whose name promises one thing and silently delivers another."

**The guard — four rules (CONJECTURE as a typed surface; the mechanisms all
exist):**

1. **Quorum certificate or escalate.** A cheap-path answer (Morton prefix /
   palette LUT / scent) ships only with a certificate: **k-of-n probes agree
   within τ**, where τ comes from *measured* anchors (Pflug-10 certifies the
   palette; Jirak 2016 sets the noise floor — never optimism), and the
   metric is a **named typed fn** (popcount Hamming, palette L1 ADC) — never
   raw-XOR-u64 ordering (P1-1). Quorum fail → **escalate one HHTL tier**
   (`RouteAction::Escalate` already exists in `bgz-tensor::hhtl_cache`);
   never silently accept.
2. **φ-stride probe placement.** The n probe indices are golden-stride
   placed (`idx_k = (k · ⌊N/φ⌋) mod N`) so they equidistribute (Weyl; the
   proof-side twin is jc pillar P3). A quorum over clustered probes is
   theater with extra steps.
3. **Spectral claims only via the pillar path.** Anything claiming
   PSD/eigen/Σ-propagation routes through `hpc/pillar/*` + `linalg/eig_sym`
   with **relative** tolerances (P0-1 lesson) and **measured** thresholds
   (P0-2); a placeholder threshold must not gate (P0-3).
4. **Hierarchy claims need their level test.** No L-deep cascade-addressing
   claim ships while its round-trip at that level is red — **PROBE-HILBERT-L4
   is currently the named blocker** (P0-4: `hilbert3d_encode([15,15,15],4)`
   returns 2925, expected 4095; do not export to consumers until green).

**PROBE-QUORUM-1 (pass/fail):** on a sampled workload, quorum-accepted
answers re-checked against full-plane recompute satisfy ρ ≥ the measured
anchors (0.9973 HIP / 0.965 TWIG); quorum-rejected answers escalate and
the escalated tier's answer satisfies the same bound. **PROBE-PHI-1:**
φ-stride probe sets beat uniform-random sets on discrepancy at equal n.

## 6. Codebook build contract (the 4⁴ condition)

Per-class centroid codebooks (registered by the consumer, scoped by class
prefix) must be built as **4-level 4-ary hierarchies** — flat k-means-256
breaks nibble-prefix ancestry and with it `is_ancestor_of` in centroid
space. **PROBE-CODEBOOK-44:** hierarchical-4⁴ vs flat-256 fidelity ρ on the
same corpus; acceptance = within the Pflug-10 certification band of flat.

## 7. Cross-references

- `OGAR/CLAUDE.md` — the canon (GUID, key-of-key-value, 3×4, centroid tile,
  prefix-scoped codebooks, 3×4-vs-4×3 standing watch).
- `OGAR/docs/INTEGRATION-MAP.md` — seams S1/S7/S9, gates F10–F14 (the
  jc×hpc floor this doc's probes extend).
- lance-graph `.claude/knowledge/guid-canon-and-prefix-routing.md` — the
  policy-side counterpart (registry mint, codebook shelf, quorum type).
- `hhtl-gridlake-pre-sprint-prompt.md` — the column-substrate identity.
- `pp13-brutally-honest-tester-verdict.md` — the theater casebook.
