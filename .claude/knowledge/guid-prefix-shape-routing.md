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
- **Stacked-pyramid perturbation = DETERMINISTIC PHASE (operator-pinned
  2026-06-10; CONJECTURE as code):** pyramid levels are *generated, not
  stored*, and the perturbation decomposes into four terms of which
  **three are already in the key**:

  ```
  perturb(addr, L) = M[addr @ coarse] · P( phase(addr, L) )  at  loc(addr)

  exponent  = level L            → the KEY's tier nibbles (>>2)  — 0 bits stored
  location  = sub-tile placement → implied mantissa (√u/golden)  — 0 bits stored
  phase     = deterministic recurrence from the address          — 0 bits stored
  magnitude = the envelope M     → THE ONLY STORED BITS
              (palette-quantized, coarser granularity than the phase varies)
  ```

  Phase is *convention, not data* — the decoder regenerates it bit-exact
  from the address, so synthesis is **lossless by construction** and
  stored cost scales with **magnitude smoothness, not perturbation
  bandwidth** (a 256×256 tile with a smooth envelope stores a 16×16
  magnitude plane — 256× cheaper than a full residual). DLSS/Halton-
  jitter logic made exact — and **helix already ships the split**
  ("HHTL = deterministic PLACE; helix = orthogonal RESIDUE"):
  `HemispherePoint::lift` (√u) = the location mantissa; **`CurveRuler`
  stride-4-over-17 = the deterministic phase walk** (coprime → full
  permutation; integer → cross-platform bit-exact); `RollingFloor` =
  the magnitude quantizer.

  **Fences (no theater):** (a) NOT lossless for arbitrary residuals —
  the unaligned remainder overflows to the next level or full-residual
  escalation, **decided per tile by the quorum certificate (§5)**,
  never assumed; (b) **D-QUANTGATE:** in quantized layers the phase
  generator MUST be the coprime-integer walk — float φ-recurrence
  drifts across platforms and loses aperiodicity under quantization;
  golden recurrence is build-time muscle-memory only. The deterministic
  phase doubles as the **anti-moiré dither** (aperiodic by coprimality).

  **PROBE-PYR-1:** the full-residual escalation tier reconstructs
  byte-exact; corrupt-residual fails loudly. **PROBE-PHASE-1:** phase
  regeneration bit-exact across AVX-512/NEON/scalar and across
  platforms (integer walk only). **PROBE-PERT-RHO:** magnitude-only
  encoding meets the ρ anchors on a representative tile corpus; the
  measured escalation rate is reported, not hidden.

### 4b. Bipolar phase = signed bits = Walsh-Hadamard on VSA (CONJECTURE)

When §4's deterministic phase is **signed (±1)** — one bit per
(addr, level) — the pyramid is the Walsh-Hadamard transform of the
address tree carried on the workspace's bipolar VSA algebra
(`Vsa16kF32` is already bipolar ±1 in role-key slices). Cascade:

```
cell(addr) = ⊕_L  sign(addr, L) ·_VSA  M(addr, L)
                  │                    │
                  XOR (vsa_bind)       SUM+THRESHOLD (vsa_bundle)
                  one shift+xor SIMD   Markov-respecting
```

**Receipts shipped:** bipolar carrier `Vsa16kF32` (lance-graph-contract
`crystal/fingerprint.rs`); `vsa_bind` (multiply ±1 = XOR of sign bits)
+ `vsa_bundle` (sum + threshold) = the iron-rule algebra; bit-exact
integer phase walk = helix `CurveRuler` stride-4-over-17
(D-QUANTGATE-compliant); Markov guarantee = `I-SUBSTRATE-MARKOV`
(bundle ≡ Chapman-Kolmogorov).

**Quantum-shaped properties, all DETERMINISTIC:**
- **Superposition** — cells hold many bundled contributions; unbinding
  with a role key (XOR with key's sign pattern) extracts one.
- **Heisenberg-shaped bound** — `I-VSA-IDENTITIES` Test 1:
  N ≤ √d/4 ≈ 32 distinct readouts per cell before SNR collapses. This
  IS the substrate's uncertainty principle; classical bound, real wall.
- **Resonance field** — a region's "value" = inner product of its
  address-signature with the magnitude pyramid (Walsh-resonance, not
  Fourier).
- **Roundtrip bit-exact** — phase generated not stored; Walsh-Hadamard
  self-inverse up to scale.

**TWO-ALGEBRA RULE (load-bearing, do not violate):**
- **Sign side = XOR** (one SIMD op; allowed for single-target deltas per
  `data-flow.md` rule "single target: gated XOR").
- **Magnitude side = `vsa_bundle`** (sum + threshold) — **NEVER** raw XOR
  on magnitudes; `MergeMode::Xor` breaks Markov per `I-SUBSTRATE-MARKOV`.
- Two operators, two algebras, one pyramid. PP-13 P1-1
  ("raw-XOR-u64 ordering as 'nearest'") is the named anti-pattern that
  confuses them.

**Honest fences (no theater):**
1. "Quantum-like" is the BUNDLING ALGEBRA, not measurement randomness.
   No headline drift to "quantum substrate"; we shipped Walsh-Hadamard +
   VSA-bipolar — the win IS bit-exactness, not probabilism.
2. Bipolar = 1-bit phase. Multi-bit phases stack above only when
   measured to be needed.
3. Parseval-preservation requires the bundle. Raw-XOR-only =
   permutation algebra = no L2 conservation = no "top gaussian preserved".

**Probes (new):**
- **PROBE-WHP-1 (Parseval):** for random ±1 sign-fields,
  `Σ|cell|² = Σ|M_L|²` within a Jirak-derived noise floor (never
  optimism).
- **PROBE-WHP-2 (roundtrip):** encode→decode→encode is byte-identical
  across AVX-512/NEON/scalar.
- **PROBE-WHP-3 (unbind):** binding then unbinding with a role-key
  recovers the bound element with measured margin; fails cleanly past
  N > √d/4 (the Heisenberg bound made explicit).
- **PROBE-WHP-4 (two-algebra guard):** explicit failing test asserting
  raw-XOR on magnitudes breaks Chapman-Kolmogorov consistency; guards
  against future `MergeMode::Xor` drift.

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
4. **Hierarchy claims need their level test — and HILBERT-L4 is
   VERIFIED GREEN (2026-06-10, run first-hand):** 13/13 tests pass
   including `level4_all_indices_unique` (**bijective onto [0,4096)**
   — exactly what cascade addressing needs) and
   `level4_curve_is_connected` (adjacent indices Manhattan-dist 1).
   **PP-13 P0-4's expectation (`encode([15,15,15],4) == 4095`) was an
   ORIENTATION assumption, not the contract** — under the shipped
   orientation the curve ends at a different corner; 2925 is a valid
   endpoint. The blocker framing is retired (Codex catch on #215); the
   exhaustive L4 suite stays as the **standing gate** — any future
   table change must keep it green before L-deep addressing claims.

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
