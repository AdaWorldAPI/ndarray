# Architecture Entropy Ledger — RESOLVED

> Append-only. Each section is a dated closure of one or more rows from the OPEN ledger
> (`.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md`). Rows move here when their entropy
> reaches 1 (Clean — fully resolved). Recognitions (no code change, just naming what was
> already true) are also valid closures and are flagged as such.

---

## 2026-05-07 — RECOGNITION-1: the architecture is largely already shipped (sprint-2 meta-finding)

| Field | Value |
|---|---|
| **ID** | RECOGNITION-1 |
| **Region** | workspace meta-finding (no R-id) |
| **Component** | Architectural pattern recognition framework |
| **State** | Documented (no code change) |
| **DupCount** | 0 (this is a closure, not a duplicate) |
| **Maturity** | Stage 4 (canonical recognition surface) |
| **Smart/Dumb** | n/a (meta) |
| **Entropy** | 1 (Clean — fully resolved as a recognition) |
| **Plan-status** | n/a (informational closure) |

### Framing

A 16-turn architectural conversation explored what looked like a green-field design
problem — build an OGIT-G overlay system with 15 patterns A through O. The realization
that surfaced across the conversation: **~80% of the architecture is ALREADY SHIPPED**
in workspace. Only ~20% (Patterns A, B, C, D, E, F, J, K wiring) is genuinely new work.

Without naming this fact, sessions kept re-discovering or re-designing pieces that
already existed. This entry captures the meta-finding so future sessions skip the
rediscovery tax.

### Pattern-to-file recognition map

Full version in `.claude/knowledge/tier-0-pattern-recognition.md` (W2's deliverable).
Key rows:

| Pattern | Already shipped in |
|---|---|
| **H** Switchable Cognitive Vessel | `crates/p64-bridge/src/lib.rs::cognitive_shader::CognitiveShader` (8 predicate planes + bgz17 semiring + HHTL cascade) |
| **N** Fingerprint-as-Codebook-Address | `crates/thinking-engine/src/prime_fingerprint.rs`, `qualia::FAMILY_CENTROIDS`, `p64-bridge::STYLES`, cam_pq codebook, bgz17 palette |
| **O** Phenomenological Memory Layers | `crates/thinking-engine/src/qualia.rs` (17D, music-calibrated, Bach 7+1 = CausalEdge64 7+1), `awareness_dto.rs` |
| **M** Wave-Particle Bimodal (primitives) | bgz17/resonance/qualia (wave) + AriGraph/SPO/NARS (particle) in workspace |
| **F** ractor actor message shape | `crates/cognitive-shader-driver/src/grpc.rs` (tonic service trait shape) |
| **I** Implicit Cognition (CycleAccumulator) | PR #337 shipped CycleAccumulator |

### Anti-Pattern surfaced: "Designing What's Already Built"

Over 16 turns, the conversation repeatedly described future Pattern-X work that was
discovered, mid-conversation, to already exist in workspace. The user pointed me at
`qualia.rs` after I had sketched a 17D qualia pattern as future work — and the
existing file was richer than my sketch. Same for `p64-bridge`, `thinking-engine`,
and `cognitive-shader-driver/grpc.rs`.

This generalizes the **Discovery-Loop** anti-pattern already documented in
`.claude/patterns.md` from cycle-level ("find the existing crate before writing a
new one") to **architecture-level** ("recognize the existing pattern before
proposing a new design"). The cure is the same shape, scaled up:
**READ existing code BEFORE proposing new design.**

### Pre-work checklist amendment (cross-ref W3's patterns.md update)

The pre-work checklist in `.claude/patterns.md` should grow one step:

> **Read `.claude/knowledge/tier-0-pattern-recognition.md` to see if the proposed
> architectural piece is already shipped.**

This is the recognition gate. If the proposed piece maps to a row in the Tier-0
table, the work item collapses from "design + build" to "wire + name."

### Cross-row reframes (detail in W6's OPEN ledger update)

The RECOGNITION-1 lens re-scored five sibling rows in the OPEN ledger. Each reframe
is pure recognition — zero code change, just correctly classifying what is already
shipped:

| Row | Old entropy | New entropy | Reason |
|---|---|---|---|
| **THINK-1** | Spaghetti-5 | **3** | Not 4-copy drift; intentional 12-base codebook (`p64-bridge::STYLES`) + 36-entry composed surface (`contract::thinking::ThinkingStyle`). |
| **HEEL-1** | 4 | **2** | HHTL cascade canonical impl in `p64-bridge::cognitive_shader::cascade`; doc says "No POPCNT. No Hamming. Distance is PRECOMPUTED. O(1)." |
| **ADJ-THINK-1** | 4 | **2** | `[u64; 64]; 8` planes IN p64-bridge IS the adjacency store; just needs `tau_write()` public API. |
| **CRYSTAL-1** | 4 | **2** | Two legitimate codebooks at different Pattern-N layers; not a collision. |
| **CAM-DIST-1** | 3 | **2** | One-line fix; substrate shipped. |

See W6's OPEN-ledger entry for the per-row evidence and follow-up actions.

### Aggregate entropy delta

| Channel | Delta | Code touched? |
|---|---|---|
| Direct row reframes (5 rows above) | **−11 entropy** | No |
| Cluster reorganization | ~37 cluster-entropy units reorganized into architectural slots | No |
| Pattern catalog | 15 patterns A–O catalogued; ~8 already-shipped, ~7 genuinely new wiring work | No |

The sprint's largest single entropy reduction came from **naming**, not building.

### Evidence

- Conversation between user and Claude, 2026-05-07, 16 turns.
- Cross-references to sister deliverables:
  - **W1** — `.claude/plans/unified-ogit-architecture-v1.md` (the 15-pattern plan-doc, Tier 1–4 wiring scope for the genuinely-new ~20%)
  - **W2** — `.claude/knowledge/tier-0-pattern-recognition.md` (the canonical pattern-to-file map, source-of-truth for the recognition gate)
  - **W3** — `.claude/patterns.md` (Discovery-Loop scaled up to architecture-level; pre-work checklist amendment)
  - **W4** — `.claude/board/EPIPHANIES.md` (entry `E-RECOGNITION-OVER-DESIGN-17`)
  - **W5** — `.claude/board/TECH_DEBT.md` (recognition-debt entries flagged with `RECOGNITION-1` upstream)
  - **W6** — `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` (OPEN ledger; five-row reframes with per-row evidence)
  - **W7** — this section
  - **W8** — naming and glossary unification (Pattern A–O canonical names)
  - **W9** — test-coverage map for the already-shipped 80%
  - **W10** — wiring-work backlog for the genuinely-new 20%
  - **W11** — PR retro: which past PRs already implemented which patterns (closes the loop on "quietly converging")
  - **W12** — sprint-2 summary and Definition-of-Done verification
- File listings inspected during the conversation:
  - `crates/thinking-engine/` — 47 files, ~570 KB
  - `crates/p64-bridge/src/lib.rs` — cognitive_shader module with 8 predicate planes, bgz17 semiring, HHTL cascade
  - `crates/thinking-engine/src/qualia.rs` — 39 KB, 17D, music-calibrated, Bach 7+1 alignment with CausalEdge64
  - `crates/cognitive-shader-driver/src/grpc.rs` — tonic service trait shape matching Pattern F

### Future-session implications

Sessions that propose "let's build the cognitive vessel" should hit RECOGNITION-1
first and be redirected to `p64-bridge::cognitive_shader::CognitiveShader`. The
recognition gate (W3's pre-work checklist amendment) makes this redirect mechanical
rather than memory-dependent.

The work that genuinely remains is the ~20% wiring, captured in W1's plan-doc as
Tier 1–4. That backlog is finite, scoped, and not green-field.

---
