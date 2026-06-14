# KNOWLEDGE: Codec / SoA Facet Map — speed and fidelity are separable knobs

## READ BY: truth-architect, family-codec-smith, palette-engineer, savant-architect,
##          cascade-architect, integration-lead, resonance-cartographer

## STATUS: probe-backed map (ndarray PR #218, 10 reproducible probes). The holy
##         grail = ONE SoA where every facet composes for accuracy AND speed.
##         Mechanism established; white patches listed at the bottom.

---

## The one-line thesis (measured this session)

**No single vector subsumes the others (Correction-6 / I-VSA-IDENTITIES category
boundary). The unified representation is a STRUCT of orthogonal facets — one SoA
column per category — and accuracy vs speed are TWO SEPARABLE KNOBS that compose:
cascade-prune the coarse code (speed, lossless), then residue-refine only the
survivors (fidelity, +bytes).**

---

## The facets — one SoA column per native category

| Facet (SoA column) | Codec | Category | Measured this session | Knob |
|---|---|---|---|---|
| place / semantic basin | HHTL (HEEL·HIP·TWIG) | hierarchical key | cascade prune (CLAM dfs-sieve 2.3×; CAM-PQ coarse→fine 16–128× lossless) | speed |
| episodic basin | rolling floor (Belichtungsmesser / EWMA) | self-calibrating μ+3σ | ρ=1.0 tracking under SD drift; shipped global-Welford **inert** (bug) | speed/adaptivity |
| position (high-D) | CAM-PQ | NN-recall position | recall ~0.66 vs truth; cascade-prunable losslessly (recall 1.0 vs flat) | — |
| orientation (phase+mag) | helix-48 | 3-DOF direction | 24-bit lossless vs ≤f16; needs +1 sign bit; ⊥ HHTL (ρ≈0); +13.6× recon | — |
| spatial perturbation | helix → Morton pyramid | parametric field | 32,768× amortized, on-demand exact at every level, fine-scale coherent | speed/memory |
| relation + truth | CausalEdge64 (3×8 SPO + 2³ + f/c) | relational triple | SPO = 3× CAM-PQ palette + Pearl mask; entropy ρ=−0.78 reliability proxy | — |
| reliability / entropy | entropy_class → CausalEdge64 spare [63:61] | Staunen↔Wisdom scalar | nars_entropy validated as reliability proxy | — |
| value refinement | edge_codec CoarseResidue / turbovec | per-item residue | ICC 0.97–0.99, 14× error cut (vs coarse-only) | fidelity |
| time / recurrence | EpisodicWitness64 | temporal | **NOT PROBED — white patch** | — |

Bit budgets are the same order (≈6 bytes each) but the **domains differ** — the
6-byte coincidence is why "one vector" is tempting and wrong.

## The two knobs (the holy-grail mechanism, measured)

- **SPEED = the cascade.** Coarse→fine prune (partial-ADC / HHTL lower bound is
  admissible) + 2×2/4×4 register-blocked LUT (FastScan/AMX `pshufb`) + Morton-order
  contiguity + rolling-floor adaptive cut. **16–128× fewer full evals at recall
  1.000 vs flat** (`campq_cascade_probe`). Lossless — adds no error.
- **FIDELITY = the residue plane.** Coarse centroid + signed-4-bit / SVD residue.
  **ICC 0.97–0.99, 14×** error cut (`edge_codec_compare`). Adds bytes, not error.
- **They compose, orthogonally:** prune the coarse code to a small survivor set,
  then residue-refine only those. Fast AND accurate, each from its own mechanism.
  This composition is the holy grail's load-bearing claim (each half measured;
  the end-to-end compose is a white patch — see below).

## The category boundary (the iron rule that kills the WRONG holy grail)

Per Correction-6 (`bf16-hhtl-terrain.md`) + I-VSA-IDENTITIES:
- Do NOT float-reconstruct a byte register (bgz-hhtl-d on Qwen: cos~0.1, dead).
- Do NOT squeeze a relation OR a high-D point into a 3-DOF helix (`codec_overlap_probe`:
  helix recall 0.245 vs CAM-PQ 0.657 on high-D; SPO is a different category entirely).
- Do NOT measure a router by reconstruction fidelity (it routes; only calibration matters).
- ⇒ The SoA stays a struct of facets; new capability = a new column, not a fold.

## The reproducible probe family (ndarray PR #218)

`reliability` (Pearson/Spearman/Cronbach/ICC) · `edge_codec` (coarse/residue/PQ) ·
`entropy_ladder` (Staunen↔Wisdom). Probes: `edge_codec_compare`,
`instrument_mtmm_probe`, `cakes_grail_probe`, `entropy_ladder_probe`,
`helix_orthogonality_probe`, `helix_bitdepth_probe`, `morton_perturbation_probe`,
`rolling_floor_probe`, `codec_overlap_probe`, `campq_cascade_probe`. Each settles a
claim with a number; two found shipped bugs (Cascade Welford-inert; the bgz17 OOB
gather, fixed).

## White patches on the map (unbuilt / unmeasured — be honest)

1. **EpisodicWitness64 / temporal facet** — referenced, never probed. Biggest gap.
2. **End-to-end compose** — cascade-prune × residue-refine measured *separately*,
   never together as one `coarse→prune→refine` pipeline.
3. **`cam_pq_cascade_search`** — probe-proven lossless, NOT wired into real `cam_pq.rs`.
4. **AMX-accelerated CAM-PQ assignment** — proven pattern (`edge_residue_probe` 100%
   assign), not wired into `cam_pq.rs`.
5. **`TD-CASCADE-WELFORD-INERT`** — shipped `Cascade::observe` never fires `ShiftAlert`
   per-sample (cumulative Δμ ≪ 2σ); needs windowed/EWMA. Found, not fixed.
6. **Real COCA codebook** — every probe is synthetic-COCA-like (labeled); none run on
   the actual baked CAM index codebook.
7. **Full SoA assembly** — facets validated individually; the unified SoA (all columns,
   one cascade sweep) is not assembled or measured end-to-end.
8. **entropy_class → CausalEdge64 spare bits** (R2) — computed, not stored.
9. **bf16-hhtl probe queue M1/M3/M4** — the routing-not-reconstruction versions, NOT RUN.

## Cross-refs

lance-graph: `.claude/knowledge/encoding-ecosystem.md` (encoding map),
`.claude/knowledge/bf16-hhtl-terrain.md` (Correction chain incl. #6),
`.claude/plans/entropy-ladder-spo-rung-v1.md` (R1–R6), `lance-graph-contract`
(`CausalEdge64`, `EpisodicWitness64`, `EdgeCodecFlavor`, the BindSpace SoA columns).
