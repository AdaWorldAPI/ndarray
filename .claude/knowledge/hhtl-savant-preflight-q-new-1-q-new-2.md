# HHTL Substrate — Plan-Review Savant Preflight (Q-NEW-1 + Q-NEW-2)

> Date: 2026-05-19 (post PR #163 + PR #164 merge; master tip `13dfcf9d`)
>
> Audience: **plan-review savant** (joint, P2-protocol-equivalent), invoked
> by the W1 PR-X10 coordinator before kicking off the substrate-pair arc.
>
> Status: Two decisions are pending the savant's verdict. Until both are
> recorded, the implementation sprints for GridLake (PR-X1 + PR-X2) and the
> column-contract (PR-X14′) cannot spawn. The schedule in
> `hhtl-substrate-execution-prompt.md` § "W1-W8 master schedule" stays at
> its current PR-X10 + X11/X13 + X12/X4 + X9 + canary shape until the
> savant ships these decisions.
>
> **Scope**: this brief is the input to the savant; the savant's verdict
> is the output. The verdict gets appended to
> `pr-arithmetic-inventory.md` § "Shopping-list addendum" as a 2026-05-XX
> dated entry, then the schedule prompt is amended in a follow-up PR.

## The two questions

### Q-NEW-1 — GridLake slot placement

**Posed in**: `hhtl-gridlake-pre-sprint-prompt.md` § "Schedule slot —
Q-NEW-1 for plan-review savant" (master commit `ade8edb2`, lines
122-180).

**Choice**:

| Path | Description | Workers | Schedule cost | Coordinator load |
|---|---|---|---|---|
| **(a)** | W2.5 dedicated 0.5-week slot, after PR-X10 merges and before W3 spawns X11/X13 | 4 | +0.5 week (arc becomes 8.5 weeks) | +1 coordinator cycle, +1 savant cycle |
| **(b)** | Absorb as PR-X10 A13/A14 (MultiLaneColumn + `#[derive(SoA)]` become workers inside PR-X10's fan-out) | +2 inside PR-X10 (12 → 14) | 0 (fits inside W1-W2) | 0 (single PR-X10 coordinator + savant) |

### Q-NEW-2 — PR-X14′ placement

**Posed in**: `hhtl-pr-x14-substrate-contract-prompt.md` § "Schedule slot
— Q-NEW-2 for plan-review savant" (master commit `56b26716`, lines
246-292).

**Choice**:

| Path | Description | Workers | Schedule cost | Critical-path heat |
|---|---|---|---|---|
| **(α)** | Concurrent with GridLake at W2.5 (A1 MultiLaneColumn is the chain dep; A2-A10 fan out parallel) | 10 in W2.5 (heavy fan-out) | 0 vs path (a); -0.5 week if combined with path (a) | A1 critical-path slip cascades to 9 downstream workers |
| **(β)** | Sequential at W3, after GridLake lands at W2.5 | 4 in W3 | +0.5 week (arc extends by another half week) | A1 has fully stabilised before X14′ workers see it; low heat |

### The 2×2 grid

The two questions are not fully independent. The savant must pick one cell:

|   | Q-NEW-2 = α (concurrent) | Q-NEW-2 = β (sequential) |
|---|---|---|
| **Q-NEW-1 = (a)** W2.5 dedicated GridLake | **Cell A-α**: W2.5 holds 10 workers (4 GridLake + 6 contract/bridge sub-DAG after A1). Single coordinator + single savant cycle covers all 10. Arc = 8.5 weeks. | **Cell A-β**: W2.5 holds 4 GridLake workers; W3 first half holds 4 X14′ workers. Two coordinator cycles, two savant cycles. Arc = 9 weeks. |
| **Q-NEW-1 = (b)** X10-A13/A14 absorption | **Cell B-α**: GridLake = A13/A14 inside PR-X10 (W1-W2); X14′ spawns at A13/A14 merge boundary inside W2 (concurrent with A2-A12 tail). PR-X10 sprint balloons to 16 workers. Arc = 8 weeks. | **Cell B-β**: GridLake = A13/A14 inside PR-X10 (W1-W2); X14′ takes W3 first half. PR-X10 stays at 14 workers. Arc = 8.5 weeks. |

## Inputs the savant should load before deciding

Read in this order:

1. **`.claude/knowledge/hhtl-gridlake-pre-sprint-prompt.md`** — the PR-X1
   + PR-X2 sprint scope, forbidden constraints (incl. Forbidden #1's
   `linalg::distance` deferral to PR-X10 A6), done criteria, the Q-NEW-1
   trade-off paragraph
2. **`.claude/knowledge/hhtl-pr-x14-substrate-contract-prompt.md`** — the
   X14′ sprint scope, the four current duplicate column-access patterns
   it collapses (BindSpace `Box<[u64]>` + `Box<[f32]>`; lance-graph
   `HashMap<String, RecordBatch>`; planner `Morsel` placeholder enum;
   `lance::Dataset.scan()` per feature), the worker DAG, forbidden
   constraints, the Q-NEW-2 trade-off paragraph
3. **`.claude/knowledge/hhtl-substrate-execution-prompt.md`** § "W1-W8
   master schedule" lines 76-91 — the current schedule shape that needs
   amending
4. **`.claude/knowledge/pr-x10-linalg-core-design.md`** — PR-X10's worker
   DAG (A1-A12) for the absorption-vs-dedicated trade-off; check whether
   the existing PR-X10 A1 dep-chain has slack for A13/A14 absorption
5. **`.claude/knowledge/pr-arithmetic-inventory.md`** § "Shopping-list
   addendum (2026-05-19 substrate-pair update)" — the three architectural
   decisions that pinned the scope (skip SQL, don't evict datafusion, use
   storage format as-is) and the superseded earlier-session proposals
6. **`.claude/knowledge/stack-consolidation-bardioc-to-hhtl.md`** §
   "Column-substrate identity — Lance ≡ Arrow ≡ ndarray SoA" lines
   126-220 — the load-bearing claim that GridLake makes literal

## Decision criteria the savant should weigh

### For Q-NEW-1 (GridLake slot)

| Criterion | Favours path (a) | Favours path (b) |
|---|---|---|
| **Scope cleanliness** | GridLake's surface is `MultiLaneColumn` + `#[derive(SoA)]`; it does not touch linalg primitives. A dedicated PR keeps the cognitive cell + storage-carrier concerns visible separately | PR-X10 already has 12 workers handling layered linalg; adding 2 more (A13/A14) absorbs the substrate carrier into the same coherent sprint |
| **Coordinator overhead** | Adds 1 coordinator cycle + 1 savant cycle (~0.5 day overhead per cycle) | Single PR-X10 coordinator + savant covers A1-A14 (no extra overhead) |
| **Schedule cost** | +0.5 week (arc becomes 8.5 weeks) | 0 (fits in W1-W2) |
| **PR-X10 A1 critical-path risk** | Isolated — GridLake A1 (MultiLaneColumn) is independent of PR-X10 A1 (MatN const-generic carrier) | PR-X10 A1 still has slack vs. A2-A12 chain length; A13/A14 spawn after A1 lands and run parallel to A2-A12 tail. Worth verifying PR-X10 A1 is the only chain dep |
| **W3+ coordinator visibility of the Q-NEW-1 trade-off** | High — the dedicated PR makes the substrate decision visible in the PR title + commit history | Lower — A13/A14 visible only in PR-X10's worker DAG, easy to miss for W3+ coordinators |
| **Forbidden #1 (`linalg::distance` deferral to PR-X10 A6)** | Compatible — GridLake at W2.5 lands after A6 merges, so Forbidden #1's "stub `linalg::distance` until A6" caveat does not apply | Compatible by stricter constraint — A13/A14 spawn alongside A2-A12 tail (after A1 lands), but A6 may still be in-flight; A13/A14 SIMD-staged inner loops must stub `linalg::distance` until A6 merges within the same PR |

### For Q-NEW-2 (X14′ placement)

| Criterion | Favours path (α) concurrent | Favours path (β) sequential |
|---|---|---|
| **Fan-out heat** | 10 workers in W2.5 is heavy; A1 slip cascades to 9 downstream workers (A2-A4 + A5-A10) | 4 workers in W3 first half; matches GridLake's W2.5 shape; low cascade risk |
| **A1 stability** | A1 (`MultiLaneColumn` signature) is the chain dep; A5-A10 (contract + bridge) consume A1's type signature. If A1's surface needs late tweaks, 6 downstream workers re-base | A1 has fully stabilised (GridLake merged) before X14′ workers see it; X14′ workers see a stable target |
| **Coordinator + savant cycles** | Single combined cycle covers GridLake A1-A4 + contract A5-A7 + bridge A8-A10 | Two cycles (one for GridLake W2.5, one for X14′ W3); +0.5 day total overhead |
| **Schedule cost** | 0 vs Q-NEW-1's choice; -0.5 week vs (β) | +0.5 week stacked on top of Q-NEW-1's cost |
| **Discoverability** | Substrate-pair is one PR (or one cell of PR-X10 fan-out); single review surface | Substrate-pair is two PRs; each reviewable in isolation; lower per-PR review load |
| **Failure recovery** | If contract/bridge sub-DAG (A5-A10) finds a MLC signature gap mid-sprint, A1 is still in motion and tweaks land in-cycle | If contract/bridge finds an MLC signature gap, GridLake has already merged; an additional `MultiLaneColumn`-amendment PR is needed before X14′ proceeds |

### Cross-Q interactions to weigh

- **Cell A-α (W2.5 dedicated, concurrent)** maximises substrate-pair coherence (single coordinator, single savant) at the cost of W2.5 fan-out heat (10 workers, one chain dep). Recommended IF the coordinator is confident in PR-X10 A6's `linalg::distance` landing on time AND the W2.5 worker pool has the capacity for 10 concurrent.

- **Cell B-α (X10 absorption, concurrent)** is the densest option — PR-X10 sprint balloons to 16 workers; A1 + A6 chain deps both sit inside PR-X10's preflight. Highest scope-pollution risk. Probably reject unless the team explicitly wants a single mega-sprint.

- **Cell A-β (W2.5 dedicated, sequential)** is the most conservative — clean per-PR scope, lowest fan-out heat, but costs +1 full week (arc → 9 weeks). Recommended IF Q-NEW-2 risk-aversion dominates.

- **Cell B-β (X10 absorption, sequential)** is the middle-ground — GridLake folds into PR-X10's natural sprint, X14′ takes W3 first half as a clean 4-worker job. Arc = 8.5 weeks. Likely the sweet spot.

The session-author (this brief) has no preference between A-β and B-β; the
savant should weigh PR-X10 sprint-pollution against per-sprint-clarity per
the verdict criteria above.

## Output format the savant should produce

Append the verdict to `pr-arithmetic-inventory.md` § "Shopping-list
addendum" as a new dated entry, in this shape:

```markdown
### Plan-review savant verdict (2026-05-XX)

- **Q-NEW-1**: path (a) | (b) — [chosen cell]
- **Q-NEW-2**: path (α) | (β) — [chosen cell]
- **Combined cell**: A-α | A-β | B-α | B-β
- **Resulting schedule**: [updated W1-W8 table]
- **Rationale**:
    - Q-NEW-1: [why]
    - Q-NEW-2: [why]
    - Cross-Q interaction: [why this cell over the other three]
- **Carry-overs**: [any conditions, e.g., "verify PR-X10 A1 slack before
  spawning A13/A14", "amend GridLake Forbidden #1 if cell is B-α", etc.]
- **Schedule-prompt amendment**: [diff to apply to
  `hhtl-substrate-execution-prompt.md` § "W1-W8 master schedule"]
- **PR-X10-design amendment** (only if cell B-α or B-β): [diff to apply
  to `pr-x10-linalg-core-design.md` adding A13/A14 to the worker DAG]
```

The savant's verdict is the input to a follow-up "schedule amendment"
PR that applies the diffs and bumps the inventory. That PR is opened
**after** the verdict lands; the savant does not edit the schedule
prompt directly.

## Constraints the savant MUST NOT violate

Three forbidden directions from the 2026-05-19 design session
(inventory § "Decisions that pinned the scope"):

1. **Additive only**. The verdict and the follow-up amendment PR must not
   delete existing prompts, inventory entries, linalg modules, pillar
   modules, ogit_bridge, or anything else. All changes are append-only.

2. **Per-repo boundary preserved**. The verdict applies only to ndarray's
   PR-X10 + GridLake + X14′ slots. lance-graph's `jc` crate copies of
   pillar primitives (e.g., `ewa_sandwich_3d`) stay where they are. No
   cross-repo deletions, no "jc-side cleanup" sub-tasks.

3. **No scope creep into deferred decisions**:
   - SQL frontend work (sea-orm contract, sqlparser-rs, PostgREST, fusion
     parser) is deferred to Phase 3 or later — the savant must not
     re-introduce these as W2.5 or W3 line items
   - lance-graph upstream `datafusion = "52"` deps stay untouched
   - Lance storage format stays as-is — no Arrow extension types, no
     custom Lance encoders

## Cross-references

- `.claude/knowledge/hhtl-gridlake-pre-sprint-prompt.md` — Q-NEW-1 source
  (master commit `ade8edb2`)
- `.claude/knowledge/hhtl-pr-x14-substrate-contract-prompt.md` — Q-NEW-2
  source (master commit `56b26716`)
- `.claude/knowledge/hhtl-substrate-execution-prompt.md` § "W1-W8 master
  schedule" — the schedule to amend
- `.claude/knowledge/pr-x10-linalg-core-design.md` — PR-X10 worker DAG
  (needed for cell B-α / B-β path evaluation)
- `.claude/knowledge/pr-arithmetic-inventory.md` § "Shopping-list
  addendum (2026-05-19 substrate-pair update)" — the cost-of-ownership
  framework and pinned-scope decisions
- `.claude/knowledge/stack-consolidation-bardioc-to-hhtl.md` § "Column-
  substrate identity" — the load-bearing claim
- `.claude/knowledge/pr-master-consolidation-savant-verdict.md` —
  reference for the savant's verdict format (Phase 1 precedent)
- Merged PRs: #162 (column-substrate identity), #163 (GridLake + X14′
  prompts), #164 (F-order sigmoid test) — current master tip
  `13dfcf9d`
