# Agent Coordination MCP Specification

> **Status:** DRAFT  ·  **Version:** 0.1.0  ·  **Last updated:** 2026-05-17
> **Format:** NLSpec (per [strongdm/attractor](https://github.com/strongdm/attractor))
> **Substrate:** Designed as a sibling to `unified-llm-spec.md` —
> a *coordination* SDK the way attractor's spec is an *LLM* SDK.

---

## §1 Overview

### §1.1 Purpose

Claude Code spawns each `Agent()` as an isolated subprocess. Each
subprocess gets a fresh context window, cannot call other agents'
tools, cannot read other agents' in-flight state, and returns a
single blob to its parent. This breaks three patterns that worked
in earlier Claude / Gemini setups:

1. **Role teleportation** — switching persona in-context with zero loss.
2. **Mid-flight coordination** — agent A tells agent B what it found.
3. **Cross-session handoff** — session A's work feeds session B in real-time.

This spec defines the **Agent Coordination MCP**: the three-layer
coordination model needed to restore these patterns, and the
file / git / MCP primitives that implement it.

### §1.2 Two operating modes

| Mode | Substrate | When to use |
|---|---|---|
| **Native** | A future MCP server exposing `post_entry` / `read_entries` / `subscribe` endpoints (§5). | When available — preferred for low-latency. |
| **Workaround** | `tee -a` markdown files in git + GitHub PR webhooks via `mcp__github__subscribe_pr_activity`. | Today, on every Claude Code session. Defined here as the canonical fallback. |

Both modes implement the same three-layer model (§3) with the same
schemas (§6). The native server is a thin serde layer over the
workaround's wire format.

### §1.3 The universal wire format: `tee -a [bug/proposal]`

The same markdown blob serves **three purposes simultaneously**:

| Purpose | Reader | Mutability |
|---|---|---|
| **MCP orchestration message** | the orchestrator (and any subscribed session via `subscribe_pr_activity`) | append-only |
| **A2A wire format** | sibling agents (worker → worker, worker → meta, savant → savant) | append-only |
| **Log format (audit + replay)** | any future session reading git history | immutable once committed |

One write, three uses. This is by design: it means the same code
that posts an entry to the file-blackboard (Layer 1) ALSO emits the
log line ALSO becomes a cross-session-readable orchestration message
the moment it lands on a coordination PR's branch.

The blob's shape is **always a typed envelope + a body**:

```
## YYYY-MM-DDTHH:MM — KIND[severity]: one-line headline (author, branch)

**Author:** <agent-id | session-id>
**Kind:** BUG | PROPOSAL | HANDOVER | FINDING | DECISION | RFC | STATUS
**Severity:** P0 | P1 | INFO       (optional; defaults to INFO)
**Refs:** <commit-sha> | <PR-#> | <handover-id>
**Proof-of-read:**
- file=<path> sha256=<...> lines=<N> depth=<D>
- file=<path> sha256=<...> lines=<N> depth=<D>

---

<body — free-form markdown, but `## ` sub-headings preferred over prose>
```

The `tee -a` envelope is the wire; the body is the payload. Routers
(orchestrator, meta-agent, MCP server) dispatch by `Kind` and
`Severity`; readers grep by `Author` + `Refs`; auditors replay by
the immutable git-log order.

### §1.4 Tables over prose (NLSpec discipline)

Implementations SHOULD prefer tabular content over prose for every
catalog (anti-patterns, validation rules, verdict vocabularies,
configuration defaults, parity matrices). Rationale:

| Property | Table | Prose |
|---|---|---|
| Machine-parseable | yes | no |
| Forces consistency (all rows have the same columns) | yes | no |
| Compressible (each row is a fact) | yes | no |
| Diffs cleanly in PRs | yes | partial |
| Resists hedging language | yes (cell width caps qualifiers) | no |
| Suitable for English narrative | no | yes |

Use prose for: section overviews, rationale paragraphs, conflict
explanations. Use tables for: everything else.

### §1.5 What this spec is NOT

- Not a workflow engine. Wave / sprint structure is in
  [`autoattended-orchestrator-spec.md`](./autoattended-orchestrator-spec.md).
- Not a per-agent loop. Worker behavior is in
  [`anti-skim-agent-spec.md`](./anti-skim-agent-spec.md).
- Not provider-coupled. Both modes are LLM-provider-agnostic.

---

## §2 Terminology

| Term | Definition |
|---|---|
| **Layer-0 Teleportation** | The main thread loads an agent card and "wears its hat" in-context. Zero transport. |
| **Layer-1 File Blackboard** | A per-session append-only markdown file (`AGENT_LOG.md`) that workers read on entry and append to on exit. |
| **Layer-2 Branch Pub/Sub** | A draft PR's push activity feed used as a real-time cross-session message bus. |
| **Handover** | A structured markdown file (`.claude/handovers/YYYY-MM-DD-HHMM-*.md`) written at session end for the next session to read at startup. |
| **Blackboard entry** | One append to `AGENT_LOG.md`; schema in §6.1. |
| **Cross-session broadcast** | A committed-and-protected file separate from `AGENT_LOG.md`; the durable mirror of important entries. |
| **Coordination PR** | A draft PR (never merged) whose push activity is the Layer-2 transport. |

---

## §3 The Three Coordination Layers

### §3.1 Layer-0: Teleportation

**When:** the task needs FULL conversation context (not a summary)
and the role switch is temporary.

**How:** on the main thread (NOT via `Agent()`):

```
1. Read .claude/EN/agents/<role-card>.md
2. Load its Tier-1 knowledge docs (per the card's §Inputs)
3. Do the work with full session context intact
4. When done, optionally switch: read another role card
5. Review the work from the other role's perspective
6. Back to the original role — nothing lost
```

**Cost:** zero transport, single-threaded, context budget fills as
role cards are read. No isolation: a mistake made "as role-X" is
visible to a subsequent "as role-Y" review (a feature, not a bug).

**Not for:** mechanical grind-work (use Layer-1 + `Agent()`) or
truly independent parallel work (use `Agent()` directly).

### §3.2 Layer-1: File Blackboard

**Replaces:** Mid-flight coordination (partially).
**Path:** `.claude/board/AGENT_LOG.md` (or `META/AGENT_LOG.md` per
repo convention).
**Permission:** pre-allowed in `.claude/settings.json` as
`Bash(tee -a *)` and `Bash(tee -a **)`.

**Setup (native mode):**

A future MCP server exposes the file as a structured stream
(`post_entry`, `read_entries`, `subscribe`). Until that exists, the
fallback is the markdown file directly.

**Agent prompt template (workaround mode):**

```
Before starting work, read .claude/board/AGENT_LOG.md to see what
other agents already shipped or found.

After committing, append your entry:

tee -a .claude/board/AGENT_LOG.md > /dev/null <<'EOF'

## YYYY-MM-DDTHH:MM — description (model, branch)

**Agent-id:** <id>
**Bundle:** <name>
**Sentinel:** <token>
**Commit:** `abc1234`
**Tests:** N pass (M new)
**Outcome:** One-line summary.
**Proof-of-read:**
- file=<path> sha256=<...> lines=<N> depth=<D>
EOF
```

**Limitations:**

- Not real-time: agent B only sees what agent A *committed*, not
  what A is currently working on.
- Git staging: if agent A and B append without committing, only the
  last `git add` wins. Mitigation: commit immediately after append.
- Ordering: entries appear at the bottom (per `tee -a`); convention
  is newest-first. The main thread reorders during board hygiene.

### §3.3 Layer-2: Branch Pub/Sub

**Replaces:** Cross-session handoff.
**How:** open a coordination PR. Both sessions subscribe. Push
events arrive as `<github-webhook-activity>` tags via
`mcp__github__subscribe_pr_activity`.

**Setup:**

```bash
# Session A (creates the bus):
git checkout -b claude/blackboard
echo "# Coordination Blackboard" > .claude/board/AGENT_LOG.md
git add .claude/board/AGENT_LOG.md
git commit -m "init coordination blackboard"
git push -u origin claude/blackboard

mcp__github__create_pull_request(
  owner="<O>", repo="<R>",
  title="A2A coordination blackboard",
  head="claude/blackboard", base="main",
  body="Cross-session pub/sub bus. Do not merge.",
  draft=true
)

mcp__github__subscribe_pr_activity(owner="<O>", repo="<R>", pullNumber=NNN)

# Session B (joins):
mcp__github__subscribe_pr_activity(owner="<O>", repo="<R>", pullNumber=NNN)
git fetch origin claude/blackboard
git checkout claude/blackboard
# Read AGENT_LOG.md
```

**Coordination loop:**

```
Session A:                              Session B:
  [does work]
  tee -a AGENT_LOG.md > /dev/null <<EOF ...
  git add && commit && push
                                        ← <github-webhook-activity>
                                        git pull origin claude/blackboard
                                        cat AGENT_LOG.md
                                        [builds on A's findings]
                                        tee -a AGENT_LOG.md ...
                                        git add && commit && push
  ← <github-webhook-activity>
  git pull
  # reads B's entry, continues
```

**Limitations:**

- GitHub webhook latency: seconds to low minutes.
- Rate limits: GitHub API 5000/hr authenticated.
- Requires network: doesn't work offline.
- PR MUST stay open: closing it kills the subscription.

### §3.4 Layer-3 (handover): Structured handover files

**Replaces:** Session-to-session context transfer.
**Path:** `.claude/handovers/YYYY-MM-DD-HHMM-<from>-to-<to>.md`.

The `SessionStart` hook (`.claude/hooks/session-start.sh`) cats the
latest handover file into the next session's context.

Schema is in §6.3.

---

## §4 Decision Matrix

| Need | Workaround | Native MCP equivalent | Cost |
|---|---|---|---|
| Agent A's findings feed agent B (same session) | Layer-1 File Blackboard | `post_entry` + `read_entries` | Low: `tee -a` + `git add` |
| Session A's work feeds session B (real-time) | Layer-2 Branch Pub/Sub | `subscribe` on entries | Medium: PR + subscribe |
| Full-context role switch (no loss) | Layer-0 Teleportation | n/a (in-process) | Zero: just read the card |
| Session-to-session knowledge transfer | Handover files | `post_handover` + `read_latest_handover` | Low: write once, read at startup |
| Parallel independent grind-work | Standard `Agent()` spawns | n/a | Low: fire and forget |
| Multi-source synthesis needing judgment | Teleportation on Opus main thread | n/a | Zero |

---

## §5 Native MCP Server Contract

A future MCP server SHOULD expose the following endpoints. They are
specified here so the workaround mode's wire format is forward-compatible
with the native mode.

### §5.1 `post_entry`

```yaml
endpoint: post_entry
params:
  board: string                          # e.g. ".claude/board/AGENT_LOG.md"
  entry:
    timestamp: string (ISO 8601)
    agent_id: string
    bundle: string | null
    sentinel: string | null
    commit: string | null                # short SHA
    outcome: enum (SUCCESS | PARTIAL_SUCCESS | RETRY | FAIL | SKIPPED)
    summary: string
    proof_of_read: list of ProofOfRead   # see §6.2
returns:
  entry_id: string
  position: integer
errors:
  - INVALID_OUTCOME
  - SENTINEL_MISMATCH
  - PROOF_OF_READ_MISSING_REQUIRED
```

### §5.2 `read_entries`

```yaml
endpoint: read_entries
params:
  board: string
  since: string (entry_id) | null
  limit: integer (default 100, max 1000)
  filter:
    agent_id: string | null
    outcome: enum | null
    bundle: string | null
returns:
  entries: list of BlackboardEntry
  next_cursor: string | null
```

### §5.3 `subscribe`

```yaml
endpoint: subscribe
params:
  board: string
  cursor: string | null                  # start position
streams:
  type: server_sent_events
  event_kinds:
    - entry_appended
    - entry_redacted
    - board_archived
returns_initial:
  cursor: string
errors:
  - BOARD_NOT_FOUND
  - SUBSCRIPTION_LIMIT_EXCEEDED
```

### §5.4 `post_handover` / `read_latest_handover`

```yaml
endpoint: post_handover
params:
  from_session: string
  to_session: string | null              # null = "next session, whoever"
  handover: Handover                     # see §6.3
returns:
  handover_id: string

endpoint: read_latest_handover
params:
  to_session: string                     # current session id
returns:
  handover: Handover | null
```

### §5.5 Authentication

The native server SHOULD use the same auth substrate as the GitHub
MCP server (per-session OAuth or PAT). Workaround mode uses git
push permissions on the coordination PR's branch.

---

## §6 Schemas

### §6.1 BlackboardEntry

```json
{
  "entry_id": "01HXYZ-WAVE12-A4-SUCCESS",
  "timestamp": "2026-05-17T17:42:00Z",
  "agent_id": "A4",
  "bundle": "customer-master-data",
  "sentinel": "WAVE-12-A4-7f3c",
  "commit": "a1b2c3d4",
  "outcome": "SUCCESS",
  "summary": "customer-master ported; 7 parity tests green; no Iron Rule violations.",
  "proof_of_read": [
    { "file": "META/INVARIANTS.md", "sha256": "...", "lines": 412, "depth": "thorough" }
  ]
}
```

### §6.2 ProofOfRead

```json
{
  "file": "path/relative/to/repo-root",
  "sha256": "64-hex",
  "lines": 412,
  "depth": "thorough"
}
```

`depth` values: `grep` | `sed-partial` | `skim` | `read` | `thorough`
| `troubleshooting` | `fan-out` | `truncated:head-N` | `truncated:tail-N`.

### §6.3 Handover

```yaml
handover:
  id: 2026-05-17-2200-wave12-to-wave13
  from_session: 2026-05-17-evening
  to_session: null                       # any next session
  topic: customer-master-port

  what_i_did:
    - bullet: ported 12 bundles (A1..A12) for customer-master
      commits: [a1b2c3, d4e5f6, ...]

  finding:
    - bullet: BBB-Barrier compile-error happens at A5 when sea-orm 0.12 is in path
      severity: P0
      source: META/INVARIANTS.md §BBB

  conjecture:
    - bullet: switching to sea-orm 0.13 might break the migration path; needs probe
      severity: P1

  blockers:
    - description: A5 cannot be merged until the sea-orm RFC is accepted
      filed_in: META/REQUESTS-FROM-AGENTS.md#A5

  open_questions:
    - "Should A11 be a separate PR or rolled into the consolidation?"

  proof_of_read:
    - file: META/INVARIANTS.md
      sha256: "..."
      lines: 412
      depth: thorough
```

### §6.4 RequestEntry (`META/REQUESTS-FROM-AGENTS.md`)

```yaml
request:
  agent_id: A5
  file: src/customer/master/sea_orm.rs
  timestamp: 2026-05-17T18:14:32Z

  question: |
    The reference Python uses an idempotent UPSERT but the sea-orm
    builder I have does INSERT-OR-FAIL. Spec is ambiguous.

  tried:
    - sea-orm `on_conflict().do_nothing()` — fails with `Error: ...`
    - manual transaction wrap — works but doubles latency

  blocker: AMBIGUITY

  proof_of_read:
    - file: vendor/ogit/NTO/WorkOrder/entities/customer.ttl
      sha256: "..."
      lines: 137
      depth: full
    - file: ../WoA/woa/blueprints/customer.py
      sha256: "..."
      lines: 891
      depth: full
```

### §6.5 AnswerEntry (`META/ANSWERS-TO-AGENTS.md`)

```yaml
answer:
  agent_id: A5
  in_response_to: 2026-05-17T18:14:32Z

  decision: |
    Use sea-orm transaction wrap. The latency cost is acceptable.

  propagation:
    invariants_section: META/INVARIANTS.md §UPSERT-PATTERN added
    rfcs: null

  signed_off_by: meta-agent (auto)
  timestamp: 2026-05-17T18:42:00Z
```

---

## §7 Append-only Governance

### §7.1 Mutability invariant

Across all coordination files:

| File | Mutability |
|---|---|
| `Stand.md` / `STATUS_BOARD.md` | Mutable (the only one). |
| `AGENT_LOG.md` | Append-only. |
| `REQUESTS-FROM-AGENTS.md` | Append-only. |
| `ANSWERS-TO-AGENTS.md` | Append-only. |
| `INVARIANTS.md` | Append-only by section; sections may be marked SUPERSEDED but never deleted. |
| Handover files | Immutable after write. |
| Cross-session broadcast files (`PR_ARC_INVENTORY.md`, `LATEST_STATE.md`, `EPIPHANIES.md`, `IDEAS.md`, `INTEGRATION_PLANS.md`, `ISSUES.md`, `TECH_DEBT.md`, `STATUS_BOARD.md`) | Append-only at the entry level; specific fields are mutable state. |

### §7.2 Enforcement

Implementations SHOULD enforce append-only governance at the
permissions layer. The lance-graph reference (see
[`AdaWorldAPI/lance-graph` `.claude/settings.json`](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/settings.json))
denies `Edit` and `Write` on the 8 bookkeeping files and allows
only `Bash(tee -a *)`. This is the recommended pattern.

---

## §8 Relation to Runtime A2A (Layer 1, in-process)

When a project also runs an in-process Blackboard in its own runtime
(e.g. `lance_graph_contract::a2a_blackboard::Blackboard`), the
session-level workarounds are structurally isomorphic to it:

| Runtime (in-process) | Session (this spec) |
|---|---|
| `Blackboard.entries` | `AGENT_LOG.md` entries |
| `BlackboardEntry.expert_id` | Agent description + model |
| `BlackboardEntry.capability` | Bundle / D-ids |
| `BlackboardEntry.result` | Commit hash + outcome |
| `BlackboardEntry.confidence` | Tier-1 gate pass count |
| `Blackboard.round` | Git commit sequence |
| Experts read prior rounds | Agents read prior log entries |

The isomorphism is intentional: the same coordination pattern works
on both layers because the problem is the same — independent experts
composing results on a shared substrate.

---

## §9 Definition of Done

An implementation is conformant if it satisfies ALL of:

- [ ] Three layers (Teleport / Blackboard / Branch Pub/Sub) are
      implementable with the existing primitives (file + git + MCP
      `subscribe_pr_activity`).
- [ ] Native MCP endpoints §5.1–§5.4 exist OR the workaround mode
      is documented as the explicit fallback.
- [ ] `BlackboardEntry`, `ProofOfRead`, `Handover`, `RequestEntry`,
      `AnswerEntry` schemas are implemented and validated.
- [ ] Append-only governance (§7) is enforced at the
      `.claude/settings.json` layer for the 8 bookkeeping files.
- [ ] The single-mutable-file invariant is enforced: `Stand.md` /
      `STATUS_BOARD.md` is the ONLY file workers may overwrite.
- [ ] Decision matrix (§4) is followed: workers use the right
      transport for the right kind of message.
- [ ] Coordination PRs are draft, named `claude/<topic>`, and
      explicitly marked `Do not merge` in the body.
- [ ] Handover files use the §6.3 schema with required sections:
      what-I-did / FINDING / CONJECTURE / blockers / open-questions
      / proof-of-read.

---

## §10 Cross-Provider Parity Matrix

| Capability | Claude Code | OpenAI Codex / codex-rs | Gemini CLI | Notes |
|---|---|---|---|---|
| Layer-0 Teleportation | Read agent card on main thread | Same | Same | Provider-agnostic; depends only on file read. |
| Layer-1 File Blackboard | `Bash(tee -a)` + `git add` | Same | Same | POSIX-only; provider-agnostic. |
| Layer-2 Branch Pub/Sub | `mcp__github__subscribe_pr_activity` | `gh pr` polling | `gh pr` polling | Claude Code has native MCP; others poll. |
| Native MCP server (§5) | TBD (this spec) | TBD | TBD | All three providers can speak MCP. |
| Append-only governance | `.claude/settings.json` deny list | provider-specific permission file | provider-specific | Pattern transfers; allowed-list syntax differs. |
| Handover files | `.claude/hooks/session-start.sh` cats latest | Equivalent shell hook | Equivalent shell hook | Hook script is bash; provider-agnostic. |

---

## §11 Appendix A — Sample `AGENT_LOG.md` entry (verbatim)

```markdown
## 2026-05-17T17:42 — customer-master ported; 7 parity tests green (sonnet, claude/wave-12-customer-master)

**Agent-id:** A4
**Bundle:** customer-master-data
**Sentinel:** WAVE-12-A4-7f3c
**Commit:** `a1b2c3d4`
**Tests:** 7 pass (3 new)
**Outcome:** SUCCESS — all Tier-1 gates green; no Iron Rule violations.
**Proof-of-read:**
- file=META/INVARIANTS.md sha256=fa39a3... lines=412 depth=thorough
- file=../WoA/woa/blueprints/customer.py sha256=8e1a45... lines=891 depth=full
- file=vendor/ogit/NTO/WorkOrder/entities/customer.ttl sha256=4c0b... lines=137 depth=full
```

---

## §12 Appendix B — When NOT to use coordination

If the task is...

- truly independent and parallel → use standard `Agent()` spawns; no coordination layer needed.
- a single one-shot question → use `Agent()` with sub-agent_type=Explore; no `AGENT_LOG.md` write.
- a security-sensitive review → use Layer-0 Teleportation on the main thread; do NOT spawn a separate Agent() that might be compromised by an untrusted source it reads.

---

## §13 Cooperative-Savant Scratchpad Bus

> Added 2026-05-18. The transport layer for the Cooperative Savant
> Council defined in `autoattended-orchestrator-spec.md` §15.

### §13.1 What the scratchpad bus is

A purpose-shaped extension of the Layer-1 File Blackboard (§3.2)
used specifically for the savant council's iterative cooperation.
Same wire format (`tee -a` markdown), same append-only governance
(§7), one new directory convention:

```
.claude/board/savant-council-<topic>/
├── ROUND-0-ARTIFACT.md                  snapshot of artifact under review
├── ROUND-<N>-<savant>.md                one file per (round, savant)
└── COUNCIL-VERDICT.md                   chairman's synthesis after CONVERGE
```

Topic naming: `savant-council-<sprint-id>` (e.g.
`savant-council-SPRINT-17-attractor-fold-in/`).

### §13.2 Per-savant file schema

```markdown
# ROUND-{N} {savant_id} — {topic}

**Round:** N
**Savant:** PP-13 | PP-14 | PP-15 | PP-16
**Read at round start:** [list of peer files read this round]
**Status:** IN-PROGRESS | ROUND-COMPLETE

## New findings (this round)

### F{round}-{savant}-{N}: {one-line headline}
**Axis:** AP1..AP9 | BAP1..BAP10 | PD1..PD10 | EP1..EP8
**Severity:** P0 | P1
**File:** <path>:<line>
**Detail:** <one paragraph>
**Remediation:** <one paragraph>

## Cross-references to peers' findings

- F{R}-PP-X-N: NOTED — my angle: <one-line>
- F{R}-PP-Y-N: SUPERSEDED-BY my F{R+1}-{self}-M

## Withdrawals (my prior findings I am retracting)

- F{R-1}-{self}-N: WITHDRAWN — covered by PP-X round R from a stronger angle
```

### §13.3 Reading-and-writing protocol

Each savant in each round:

1. **Read** ALL `ROUND-(R-1)-*.md` files (peer findings from previous
   round) plus its own `ROUND-(R-1)-<self>.md` if any.
2. **Decide** cross-refs, withdrawals, new findings.
3. **Write** `ROUND-R-<self>.md` in one `tee -a` call (no
   partial writes — savants share a directory and partial
   writes confuse peers' next round).
4. **Emit** `Status: ROUND-COMPLETE` if no new findings AND no
   un-addressed cross-refers from peers.

### §13.4 Chairman synthesis

After CONVERGE (all four savants `ROUND-COMPLETE` in the same
round) the chairman savant (declared in sprint config; default
PP-16 for Protocol A, PP-15 for Protocol B) reads ALL ROUND-*.md
files in lexical order and writes a single `COUNCIL-VERDICT.md`
matching the schema in `autoattended-orchestrator-spec.md` §15.5.

The chairman:
- MUST cite every kept finding with its raised-by + cross-referrals.
- MUST consolidate duplicates that survived the cooperation rounds
  (rare — most should withdraw themselves, but races can leave a
  pair that needs one merged into the other).
- MUST NOT add findings of its own that did not appear in any
  `ROUND-*.md` (the chairman is a synthesizer, not a fifth voice).
- MUST set `super_verdict` per §15.5 rules.

### §13.5 Concurrency contract

Savants may run on separate sessions; the scratchpad bus is the
synchronization point.

- **Round boundary** is implicit: a savant starts round R only after
  it has read all four `ROUND-(R-1)-*.md` files (including its own).
- **Last-writer-wins** is acceptable for the per-savant per-round
  file since each (savant, round) pair has exactly one writer by
  construction. If a savant writes twice in the same round (e.g. a
  retry after a tool-call loop), the second write replaces the first.
- **Round skipping** is forbidden: a savant MUST NOT write
  `ROUND-(R+2)` until `ROUND-(R+1)` exists for it.

### §13.6 Decision matrix update

Extends §4 with one new row:

| Need | Workaround | Native MCP equivalent | Cost |
|---|---|---|---|
| Cooperative multi-agent review with iterative cross-refer + withdraw | Layer-1 File Blackboard scoped to `savant-council-<topic>/` directory (§13) | `post_council_finding` + `read_council_round` + `subscribe_council_round` | Medium: `tee -a` per round + git commit per round |

### §13.7 Native MCP endpoints (sketch)

```yaml
endpoint: post_council_finding
params:
  topic: string                          # e.g. "SPRINT-17-attractor-fold-in"
  round: integer
  savant: enum (PP-13 | PP-14 | PP-15 | PP-16)
  payload: FindingsFile                  # matches §13.2 schema
returns:
  file_id: string

endpoint: read_council_round
params:
  topic: string
  round: integer
returns:
  findings_by_savant: map[savant -> FindingsFile]
  complete_savants: list[savant]         # those that emitted ROUND-COMPLETE

endpoint: subscribe_council_round
params:
  topic: string
streams:
  event_kinds: [ findings_appended, savant_round_complete, council_converged, council_stalled, council_blocked ]
```

### §13.8 Validation rules

| Rule | Description | Severity |
|---|---|---|
| `COORD-001 directory-naming` | Council scratchpad MUST live at `.claude/board/savant-council-<topic>/`. | ERROR |
| `COORD-002 file-naming` | Per-savant files MUST be named `ROUND-<N>-PP-<NN>.md`. | ERROR |
| `COORD-003 append-only` | The scratchpad directory inherits the append-only governance of §7. | ERROR |
| `COORD-004 chairman-no-new-findings` | The chairman's `COUNCIL-VERDICT.md` MUST NOT introduce findings absent from any `ROUND-*.md`. | ERROR |
| `COORD-005 round-monotonic` | Each savant MUST write `ROUND-(R+1)` only after `ROUND-R-<self>.md` exists. | ERROR |

---

*End of `agent-coordination-mcp-spec.md`.*
