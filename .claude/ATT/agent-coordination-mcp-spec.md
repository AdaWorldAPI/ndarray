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

### §1.3 What this spec is NOT

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

*End of `agent-coordination-mcp-spec.md`.*
