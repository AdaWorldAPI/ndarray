# Plastic activation seam — research map v1

**Status:** DISCOVERY MAP. No learner proposed for implementation. Every number
below is either a MEASUREMENT taken this session, a FACT read from source, or is
marked UNRECOVERED. Probes are committed; nothing production changed.

Grading used throughout: **[F]** fact read from source · **[M]** measured this
session · **[I]** inference from F/M · **[S]** speculation.

---

## C. The old 95% finding — what it proves and what it does not

### C.1 What was actually recovered

The pipeline is real and its shape matches the reconstruction **[F]**: ELF →
r2sleigh/libsla lift → `ruff_r2il::behavior::FunctionBehavior` (a lossless ore
carrier that "invents no vocabulary, and it decides nothing") →
`furnace::smelt` against a seven-opcode convention (`convention.rs:104-110`:
`Copy, IntAdd, Load, Store, CBranch, Call, Return`) → ore TSV → forward def-use
chains → greedy BPE merges.

`ogar-loco` exists as a crate and **contains no learner** — one doc-comment
mentions R2IL and nothing else **[F]**. The only BPE code in the four roots is
a throwaway example, `lance-graph-planner/examples/probe_bpe_r2il_loco_microcode.rs`,
which hardcodes loco's constants and asks whether learned merges would fit its
geometry. "The ogar-loco experiment" is that fit question, not a learner in
the crate.

### C.2 The numbers, classified

| number | status |
|---|---|
| 1,872 chains · 33 macros | live-derivable from the committed pass-1 corpus; **never asserted in any test or output file** |
| 8,659 / 8,652 / 2.409 / 2.529 / 0.994 | **prose only.** The instrumented harness was reverted; the entry itself says "no source change landed" |
| `ore_all.tsv` (94,536 rows), the 72,567-row serde_json ore | **not on disk anywhere** |
| "99.6% hexagon op-decode, 3 training programs, muscle-memory arc" | **UNRECOVERED.** One operator sentence, propagated into three derived docs, and graded *artifact not located* by a prior session on the same day it was written |

### C.3 The boring explanations, and which one bites

The board already deflated its own headline **[F]**: "608/608 looked
tautological for a 7-symbol alphabet"; and the PAL 0.99–1.00 transfer was
corrected the same day as "a **needle** test, not a distribution test … exact
addresses transfer when the address space is shared."

Reported controls were two shuffle nulls. **No opcode-frequency baseline, no
bigram baseline, and no occurrence-matched control was ever run** **[F]**.

**The decisive structural fact: the carrier cannot see a loop.** The def-use
chain extractor is forward-only by construction —
`probe_r2il_defuse_macros.rs:234` calls it "the **back-edge exclusion fence**"
**[F]**. So no experiment on this carrier could ever have discovered
WHILE-like structure, however it scored. The "child learns the grammar without
being handed the grammar book" hypothesis is not merely unproven on this
substrate; it is **inexpressible** on it.

### C.4 This session's independent replication attempt

Rather than trust the numbers, the same *shape* of experiment was rebuilt on
ore that does exist (`examples/behavioral_soak_probe.rs`): soak one never-reset
codebook across corpora and measure reuse against new mint, with a granularity
ladder so the trivial rung is visible.

**[M] Cross-ISA exact-match transfer.** A codebook minted from 1986 6502 game
code covers 50.4% and 56.8% of two x86-64 C binaries' basic blocks; the
marginal-preserving null reaches 17.1% and 37.5%.

**[M] Where that transfer lives.** Stratified by block length:

| block length | transfer, binary 1 | transfer, binary 2 |
|---|---|---|
| 1–4 ops | 83–100% | 56–96% |
| 8+ ops | **0.0%** (135 blocks) | **1.9%** (103 blocks) |

Exact sequence identity is the wrong atom for a long block, and the largest
bucket in both binaries transfers essentially not at all.

**[M] The composition question, and the null that kills it.** Covering long
blocks by concatenating short known tokens — the actual BPE question — gives
98.3% and 97.1% of x86 ops covered by tokens of length ≥ 2. **A shuffled soak
that preserves every block length and the opcode marginal exactly reaches 97.6%
and 95.6%.** Coverage is therefore not evidence.

**[M] What survives the null: the state budget.** The shuffled codebook needs
**4,526 tokens** where the real one needs **681** — a 6.6× smaller codebook at
matched coverage. The alphabet control explains why coverage saturates: seven
symbols, and the real codebook holds 44.9% of possible bigrams, 15.2% of
trigrams, 0.3% of 6-grams.

**[M] Direction control — transfer is strongly asymmetric.** Soaking the other
way round, an x86-64 codebook of 275 atoms covers **81.1% and 81.0%** of the two
6502 images, against 50.4% and 56.8% in the forward direction.

| direction | soak atoms | exact-match reuse on the other ISA |
|---|---|---|
| 6502 → x86-64 | 370 | 50.4% · 56.8% |
| x86-64 → 6502 | 275 | **81.1% · 81.0%** |

If two architectures had genuinely converged on a shared behavioural basis,
transfer would be roughly symmetric. A 30-point asymmetry says instead that one
repertoire largely *contains* the other: the 6502 corpus is repetitive (299
atoms over 1,430 blocks) while the x86 corpus is varied (275 atoms over 529
blocks). Containment of a simpler repertoire by a richer one is the ordinary
expectation, not evidence of convergence.

**Ruling on F2.** The cross-language regularity is real but it is **not** a
coverage phenomenon and **not** evidence of a behavioural basis in the strong
sense. Restated honestly: *lowering to a seven-opcode IR makes coverage
saturate for anyone, the direction of transfer is asymmetric in the way
containment predicts, and the learned vocabulary's only measured advantage is
that it does the same job with 6.6× less state.* Compression, not
comprehension.

---

## E. Ternlog amortization — measured, and the claim is half right

`examples/ternlog_amortization_probe.rs`. All arms compute `A ∧ M₁ ∧ … ∧ M_K`
with a correctness gate on the surviving set, and every arm runs to a 60 ms
floor.

**[M] Depth.** Folding two constraints into one `VPTERNLOGQ` saves
instructions, not bandwidth. T3/T1 sits at 0.44–0.70 while the working set is
L1-resident and **returns to ~1.0 by K=32**, where the masks stop fitting and
cost becomes memory-bound. Per-constraint cost is **flat** in K, not falling.

**[M] Residency, derived from achieved mask traffic rather than read off
timings.** L1 peaks near 139 GB/s, L2 holds 90–93 GB/s, L3 settles at 29–30
GB/s. Host is 48 KiB L1d, 2 MiB L2 per core, 260 MiB shared L3.

**[M] Materialization.** 1,000 chained ternlog steps allocate **0 bytes**,
counted with a global allocator.

**[M] The density crossover — the result that actually matters.** A bitset pays
for every bit, so the mask arm is **flat at ~3.2 µs per constraint at every
density** over a 2²⁰-bit substrate, while a survivor-walking sparse arm scales
with what survives. They cross **between 0.1% and 0.8% active** (roughly 2,000–7,000
survivors out of 1,048,576).

**[I] Consequence for an attention field.** A focus mask's cost is independent
of how few things are lit. For attention — whose whole point is that few things
are active — that is the wrong cost curve **unless the substrate is partitioned
so a mask only ever covers a resident region**. The honest architecture is
therefore hierarchical: sparse index across the substrate, mask within a tier.

---

## F. Membrane — which half is mechanism

**[F] The mechanical half is already shipped and is not a metaphor.**
`ndarray::simd::ternlog` exposes eight truth tables, and one of them is exactly
the membrane update: `AND2_ANDNOT = 0x40` computes `a & b & !c` over three
same-shaped masks in one instruction. Read as *activation ∧ permeability ∧
¬inhibition*, that is a local gate deciding what propagates, what is blocked,
and it composes with `OR2_AND` (`(a|b) & c`, either prerequisite gated by a
third) and `MAJ3` (two-of-three majority). Three states — structural,
instantaneous, learned-permeability — can share one geometry with no conversion
and, per E, no allocation.

**[S] The metaphor half.** Nothing here models calcium channels, and no
measurement supports calling a mask a synapse. The useful content is the
computational role only.

**[I] What learning would have to be.** Under the membrane reading, learning is
`P[i] ↑ / P[i] ↓` on a permeability mask — a bit flip, or a small counter that
thresholds into a bit. That is the cheapest possible plastic state and it is
ternlog-native. Whether it is *sufficient* is exactly what probe H1 below asks.

---

## G. Hex ruling

**Killed again, and this time the reason generalizes.** Beyond E-Q6 (failed G1,
G2, G3) and Q8's degree-1 ablation ("identical to four decimals, at 5.5× less
memory … B is a bigram successor table; the topology is decoration"), Q8b
recorded the structural reason **[F]**: *"The atom alphabet is 7. A degree-6
neighbourhood is therefore the complete graph minus self."*

**[I]** A six-neighbour topology cannot carry information over an alphabet of
seven. The hex experiments were not unlucky; they were degenerate by
construction. Any future spatial-locality claim must first show its alphabet is
large enough for degree-6 to be a restriction rather than a tautology — that is
now the entry gate, and it is cheap to check.

An audit also found **no hexagonal or axial adjacency exists anywhere** in
lance-graph, ndarray, or OGAR **[F]**. There was never a hex substrate to have
measured 99.6% on.

---

## J. Scale law — measured, and it says no

**[M]** From the soak curve, codebook size against cumulative corpus:

| corpus | cumulative occurrences | codebook | codebook/occ | marginal atoms per occ |
|---|---|---|---|---|
| 6502 image 1 | 1,430 | 299 | 0.209 | 0.209 |
| 6502 image 2 | 1,787 | 370 | 0.207 | 0.199 |
| x86-64 binary 1 | 2,015 | 483 | 0.240 | **0.496** |
| x86-64 binary 2 | 2,316 | 613 | 0.265 | 0.432 |

**No sublinear growth.** The ratio *rises*, and the marginal cost per new
occurrence more than doubles at the ISA boundary. On this carrier the plastic
state grows at least linearly and accelerates when the domain changes.

**[I]** The scale-inversion hope is not supported by any measurement in this
workspace. It may still hold for a different atom — this measures block-opcode
sequences, not conductance over a fixed channel set, and a fixed channel set is
bounded by construction in a way a vocabulary is not. That distinction is the
one thing that keeps the hypothesis alive, and it is testable.

---

## K. Kill conditions

| claim | already dead | would die if |
|---|---|---|
| hex locality | **YES** — degree-6 over a 7-symbol alphabet is the complete graph; three failed gates | — |
| tiny-plastic-state / scale inversion | wounded — [M] shows linear-and-accelerating growth | a fixed-channel conductance field also grows with corpus |
| ternlog amortization | **half dead** — [M] flat, not falling; saves instructions not bandwidth | already answered; do not re-ask |
| membrane masks | alive | a permeability bit cannot beat a per-channel counter at equal bytes |
| learned activation | alive, untested | it cannot beat a bigram successor table at equal state — the Q8 outcome, one level up |
| behavioural basis (F2 strong form) | **dead** — [M] the null reaches 97.6% | — |
| behavioural basis (F2 weak form: 6.6× less state at matched coverage) | alive | an occurrence-matched control closes the gap |

---

## B. Dormancy map — the number is zero

Measured over MedCare-rs, the richest consumer of the substrate **[F]**:

| count | what |
|---|---|
| 54 | distinct callable capabilities enumerated |
| 26 | reached only transitively from an HTTP handler, request-local |
| 6 | run once at boot |
| 22 | **no production caller at all** — tests, examples and bake binaries only |
| 1 | a caller that is neither route, test, nor boot (a cohort *generator*), feeding nothing back |
| **0** | **whose output is consumed by another adapter across requests, by a scheduler, or by any persisted outcome** |

The five-level distinction resolves sharply. Capabilities *exist* (54) and are
mostly *reachable* (32). Selection is by URL: `patient.icd` → one of fifteen
curated enum values → an `is_a` ancestor walk to a depth the query string
supplies. Nothing is *selected* by a policy, and **nothing's activation changes
a later selection**. Every reasoning call recomputes from a boot-time synthetic
cohort and renders. The only per-request mutable state is memoization of
identical bytes, so latency changes and content cannot.

Dormant-by-design capabilities include the abductive frontier (zero consumers),
a reinforcement lane whose own doc calls persistence "deferred", the cognitive
cycle driver (built only under a feature production never enables), lab trends,
vital stats, the Cypher engine, and audit-event emission — an audit sink is
constructed at boot and **no route ever emits an event**, so its merkle chain
stays at zero.

**[F] The seam already exists and has zero callers.** `bake_hydrate::append_witness`
plus a `patient_nodes` Lance table are fully implemented, with the module doc
stating that appends "are the normal, expected operation". The table is seeded
once and never appended. Every overlay the reasoners produce is explicitly
ephemeral — one module says its overlay "is transient and is discarded after the
merge"; another pins its cycle counter to a constant so two requests are
byte-identical.

**[I] So the missing thing is not knowledge and not an adapter.** It is a
*write*. The substrate has an outcome-recording seam, it is finished, and it is
not wired. Everything in §D is a proposal for what to write through it.

---

## D. Plasticity candidates, ranked smallest first

Ranking is by *state bytes and mechanism size*, not by appeal. Each is judged
against the same standard the workspace has already applied and failed things
by: it must beat a bigram successor table at equal state.

**D1 — Visit-count conductance on a permeability mask.** *State:* one bit per
channel, flipped when a saturating counter crosses a threshold. *Update:* on a
useful outcome, increment channels on the successful path; decrement on
failure. *Readout:* `ternlog(active, permeable, ¬inhibited)`, one instruction.
*Bytes:* channels/8. At 4,096 channels, **512 bytes**. *Ternlog-native:* yes,
by construction. *Locality:* none required. *Biggest falsifier:* a per-channel
frequency count at the same bytes reaches the same recall — i.e. the Q8 outcome
one level up.

**D2 — Successor table over channel pairs.** *State:* sparse `(from, to) →
count`. *Bytes:* grows with observed pairs; at 4,096 channels the dense bound
is 2 MB and the sparse reality far less. *Ternlog-native:* no — it is a lookup,
not a mask. *Falsifier:* it is the baseline everything else must beat; if it
wins, the answer is "a transition matrix" and the romance is over. **Rank it
first to run, not first to hope for.**

**D3 — Eligibility trace over the mask.** *State:* D1 plus a decaying trace of
recently active channels, so credit reaches a path rather than an endpoint.
*Bytes:* one small counter per channel; 4,096 channels × u8 = **4 KB**.
*Ternlog-native:* the readout is, the decay is not. *Falsifier:* a trace of
length 1 (i.e. D1) matches it.

**D4 — Learned inhibition only.** *State:* one *suppression* mask; nothing is
ever excited, only ruled out. *Bytes:* channels/8. *Rationale:* the substrate's
measured strength is Boolean elimination, and `AND2_ANDNOT` makes inhibition
one instruction. *Falsifier:* inhibition alone cannot bring a useful target
*earlier*, only cheaper — which section H1 measures directly.

**D5 — Per-class conductance keyed by classid.** *State:* one counter per
`classid`, not per node — so state is bounded by the codebook (98 concepts
today), not by the graph. *Bytes:* ~100 counters, **under 1 KB, and constant in
graph size**. *This is the only candidate whose state provably cannot grow with
the substrate*, which is exactly what §J found the vocabulary carrier could not
promise. *Falsifier:* class granularity is too coarse to discriminate useful
from useless targets within a class.

**D6 — Bloom-shaped "seen-useful" filter.** *State:* one bitset, hashed
membership of channel sets that preceded a good outcome. *Bytes:* tunable.
*Falsifier:* false-positive rate makes precision worse than D1 at equal bytes.

**D7 — Logistic regression over channel features.** *State:* one weight per
channel, f32. *Bytes:* 16 KB at 4,096 channels. *Ternlog-native:* no — this is
the numeric rail. *Included deliberately* as the conventional baseline §14 of
the brief demands; if it wins decisively the Boolean framing is wrong.

**D8 — Do nothing plastic; cache the frontier.** *State:* memoized BFS results.
*Rationale:* the honest null for the whole programme. If cached traversal at
equal bytes matches every learner, there is no plasticity finding.

---

## H. Two minimal probes

### H1 — focus propagation (arm A, smallest honest form)

*Question:* at an identical expansion budget, does any plastic state reach a
useful target earlier than the shipped controller?

*Substrate:* the largest edge set that can be fetched — SNOMED at 2,053,329
edges over 484,036 concepts, bridged to MONDO by 9,161 xref pairs. Note this is
2M, not 10M; the 10M figure is reachable only as transitive closure (4,273,203
ancestor pairs under `is_a ∪ part_of`), and closure is not the same object.

*Seeds:* per patient, `labs(id)` plus the grounded anamnese entries; the oracle
is `SoaPatient.icd` resolved through `crosswalk::resolve_icd10`.

*Arms:* A0 shipped controller (curated enum + `is_a` walk) · A1 candidate from
§D · A2 shuffled A1 · A3 bigram successor · A4 random state at identical bytes ·
A5 cached BFS.

*Gates, all mandatory:* the candidate space must be the open-set 5,750-key axis,
never the fifteen-value enum — the repo's own note calls the latter "15×
hindsight contamination". Activation sparsity is reported beside recall, so a
learner that lights everything is visibly disqualified. State bytes are equal
across arms by construction.

### H2 — persistence (arm B, the actual plasticity question)

*Question:* does episode *n+1* get behaviourally easier without the ontology
changing?

*Mechanism:* wire `append_witness` — the seam that exists and has no caller —
so an episode's outcome lands in `patient_nodes`. Run E1…En updating `P`, freeze,
measure on held-out.

*Curves:* expansions-to-target vs episode · precision vs episode · **state bytes
vs episode** (the §J question, re-asked on a fixed channel set where it may
answer differently) · recovery after unrelated episodes.

*Kill:* the curve is flat, or state bytes track episodes linearly.

**Neither probe is authorized here.** H2 in particular is a *write* into a
private clinical repo and is the operator's call, not an agent's.

---

## I. Surprise criterion, quantitatively

An event counts as unprogrammed-but-useful when **all four** hold:

1. A capability becomes active that the shipped controller would **not** have
   selected for this seed — verified by running A0 on the same seed and
   confirming absence from its expansion set.
2. The activation is **useful**: it lies on the oracle path, or it strictly
   reduces expansions-to-target versus A0.
3. It is **attributable to learned state**: the shuffled arm A2, at identical
   bytes, does not produce it. This is the clause that makes the criterion
   falsifiable rather than decorative.
4. The full causal trace is logged — seed, channel sequence, the plastic entries
   consulted, and the outcome.

Reported as a rate, not an anecdote: *surprises per 100 held-out episodes*, with
its A2 rate beside it. A single spectacular trace is not evidence.

---

## L. Epiphanies

**[F] The activation seam is already built and unwired.** `append_witness` plus
`patient_nodes` are complete, documented as the normal path, and have zero
callers. The gap between "has knowledge" and "has behaviour" is currently one
missing write, not a missing subsystem.

**[M] Zero of fifty-four capabilities are outcome-driven.** Nothing in the
richest consumer becomes easier or harder to reach because of anything that
happened before.

**[M] Coverage saturates; only the state budget carries signal.** Real and
shuffled soaks reach 98.3% and 97.6%. The real codebook does it with 6.6× less
state. Report budgets, never coverage.

**[M] Ternlog saves instructions, not bandwidth.** Its advantage is real while
L1-resident and gone by K=32. Chaining is not a way to make more constraints
free; it is a way to make each one cheaper until the masks stop fitting.

**[M] A bitset's cost is independent of sparsity, and attention is sparse.** The
mask/sparse crossover sits between 0.1% and 0.8% active. This is the strongest
architectural constraint found today: masks belong *within* a resident tier,
sparse indices *across* the substrate.

**[F] The def-use carrier excludes back-edges by construction.** Loop discovery
was never expressible on the substrate every macro experiment used.

**[F] Degree-6 over a seven-symbol alphabet is the complete graph minus self.**
The hex experiments were degenerate, not unlucky.

**[I] The one candidate with a bounded state promise is per-class conductance.**
Everything keyed per node or per vocabulary item grew with the corpus when
measured; state keyed by classid cannot, because the codebook is 98 entries and
minted by hand.

**[S] Focus as an emergent observable rather than a subsystem.** Untested. The
substrate has no salience, LOD, or label-budget layer — its own UX contract says
"layer 2 does not exist" — so there is currently nothing for focus to emerge
*into*, and that gap is a prerequisite, not a consequence.
