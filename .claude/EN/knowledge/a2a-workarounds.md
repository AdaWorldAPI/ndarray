# A2A Workarounds — Cross-Agent Coordination Without Native Support

> **READ BY:** all agents, all sessions.
> **Status:** FINDING (2026-04-24). Tested in-session with 6+ concurrent agents.
> **Context:** Claude Code agents are isolated processes. No shared memory,
> no MCP channel between them, no role-switching within a session.
> These workarounds restore coordination using existing primitives.
>
> **Language-agnostic.** Examples reference `lance-graph` (Rust), but
> the patterns transfer to Python / TypeScript / Go without code change —
> the wire format is Markdown via `tee -a`, a POSIX tool with no
> language binding.

---

## The Problem

Claude Code's `Agent()` tool spawns isolated subprocesses. Each agent:
- Gets a fresh context window (no memory of the conversation)
- Cannot call other agents' tools
- Cannot read other agents' in-flight state
- Returns a single result blob to the main thread

This breaks three patterns that worked in earlier Claude/Gemini setups:
1. **Role teleportation** — switching persona in-context with zero loss
2. **Mid-flight coordination** — agent A tells agent B what it found
3. **Cross-session handoff** — session A's work feeds session B in real-time

---

## Workaround 1: File Blackboard (`AGENT_LOG.md`)

**Replaces:** Mid-flight coordination (partially).
**How:** Append-only log file that all agents read before starting
and write to after committing.

### Setup

Live at `.claude/board/AGENT_LOG.md` (or `META/AGENT_LOG.md` per
project convention). Permission pre-allowed in `.claude/settings.json`:

```json
"Bash(tee -a .claude/board/AGENT_LOG.md:*)"
```

### Agent prompt template (include in every spawn)

```
Before starting work, read `.claude/board/AGENT_LOG.md` to see what
other agents already shipped or found.

After committing, append your entry:

tee -a .claude/board/AGENT_LOG.md > /dev/null <<'EOF'

## YYYY-MM-DDTHH:MM — description (model, branch)

**D-ids:** ...
**Commit:** `abc1234`
**Tests:** N pass (M new)
**Outcome:** One-line summary.
EOF
```

### Limitations

- Not real-time: agent B only sees what agent A committed, not
  what A is currently working on.
- Git staging: if agent A and B both append without committing,
  only the last `git add` wins. Mitigation: commit immediately
  after append.
- Ordering: entries are appended at bottom (`tee -a`), but convention
  is newest-first. Main thread can reorder during board-hygiene.

---

## Workaround 2: Branch Pub/Sub (`subscribe_pr_activity`)

**Replaces:** Cross-session handoff.
**How:** Open a coordination PR. Both sessions subscribe. Push events
arrive as `<github-webhook-activity>` tags.

### Setup

```bash
# Session A (creates the bus):
git checkout -b claude/blackboard
echo "# Coordination Blackboard" > .claude/board/AGENT_LOG.md
git add .claude/board/AGENT_LOG.md
git commit -m "init coordination blackboard"
git push -u origin claude/blackboard
# Open PR:
mcp__github__create_pull_request(
  owner="...", repo="...",
  title="A2A coordination blackboard",
  head="claude/blackboard", base="main",
  body="Cross-session pub/sub bus. Do not merge.",
  draft=true
)
# Subscribe:
mcp__github__subscribe_pr_activity(owner="...", repo="...", pullNumber=NNN)

# Session B (joins):
mcp__github__subscribe_pr_activity(owner="...", repo="...", pullNumber=NNN)
git fetch origin claude/blackboard
git checkout claude/blackboard
# Read AGENT_LOG.md → see what session A did
```

### Coordination loop

```
Session A:                              Session B:
  [does work]
  tee -a AGENT_LOG.md > /dev/null <<'EOF'
  ...entry...
  EOF
  git add && git commit && git push
                                        ← <github-webhook-activity> push event
                                        git pull origin claude/blackboard
                                        cat AGENT_LOG.md  # read A's entry
                                        [builds on A's findings]
                                        tee -a AGENT_LOG.md > /dev/null <<'EOF'
                                        ...entry...
                                        EOF
                                        git add && git commit && git push
  ← <github-webhook-activity> push event
  git pull
  # reads B's entry, continues
```

### Why it works

- `subscribe_pr_activity` is already in the MCP toolkit — zero infra.
- GitHub webhooks fire on any push, regardless of content.
- Append-only files merge cleanly (no conflict on concurrent appends
  if entries are at different positions).
- The draft PR never merges — it's the bus, not a deliverable.

### Limitations

- GitHub webhook latency: seconds to low minutes.
- Rate limits: GitHub API limits apply (5000/hour authenticated).
- Requires network: doesn't work offline.
- PR must stay open: closing it kills the subscription.

---

## Workaround 3: Role Teleportation via Agent Cards

**Replaces:** In-context role switching.
**How:** Load an agent card's knowledge docs, adopt its perspective,
do the work — all on the main thread. No subprocess spawned.

### When to use

- The task requires seeing the FULL conversation context (not a summary).
- The task is accumulation (multi-source synthesis), not grindwork.
- The role switch is temporary (do 10 minutes of codec work, then
  switch back to architecture).

### How

```
# On the main thread, not via Agent():
1. Read `.claude/agents/<role-card>.md`
2. Load its Tier-1 knowledge docs
3. Do the work with full session context intact
4. When done, switch: read `.claude/agents/<other-role>.md`
5. Review the work from the other role's perspective
6. Back to main thread — nothing lost
```

### When NOT to use

- The task is mechanical grindwork → spawn a Sonnet agent instead.
- The task is truly independent → parallel Agent() spawns are faster.
- The task is long-running and would block the main thread →
  background Agent() is better.

### Limitations

- Main thread is single-threaded: no parallelism.
- Context window fills: role-switching adds knowledge doc content
  to the conversation, consuming context budget.
- No isolation: mistakes made "as role-X" are visible to the
  role-Y review (actually a feature, not a bug).

---

## Workaround 4: Structured Handover Files

**Replaces:** Session-to-session context transfer.
**How:** Write a structured handover file that the next session
reads at startup via the SessionStart hook.

### Format

```markdown
# Handover — YYYY-MM-DD-HHMM — <from-session> to <next-session>

## What I did
- [bullet list of completed work with commit hashes]

## FINDING
- [verified facts that the next session can rely on]

## CONJECTURE
- [unverified ideas that need probing]

## Blockers
- [things I couldn't resolve]

## Open questions
- [decisions the next session should make]
```

### Where

`.claude/handovers/YYYY-MM-DD-HHMM-<topic>.md`

The SessionStart hook (`.claude/hooks/session-start.sh`) can be
extended to cat the latest handover file into the session context.

---

## Decision Matrix

| Need | Workaround | Cost |
|---|---|---|
| Agent A's findings feed agent B (same session) | File Blackboard (#1) | Low: tee -a + git add |
| Session A's work feeds session B (real-time) | Branch Pub/Sub (#2) | Medium: PR + subscribe |
| Full-context role switch (no loss) | Teleportation (#3) | Zero: just read the card |
| Session-to-session knowledge transfer | Handover Files (#4) | Low: write once, read at startup |
| Parallel independent grindwork | Standard Agent() spawns | Low: fire and forget |
| Multi-source synthesis needing judgment | Teleportation (#3) on Opus main thread | Zero |

---

## Relation to Runtime A2A (Layer 1)

These workarounds mirror runtime-Blackboard structures (e.g.
`lance_graph_contract::a2a_blackboard`):

| Runtime (Layer 1) | Session (Layer 2 workaround) |
|---|---|
| `Blackboard.entries` | `AGENT_LOG.md` entries |
| `BlackboardEntry.expert_id` | Agent description + model |
| `BlackboardEntry.capability` | D-ids |
| `BlackboardEntry.result` | Commit hash + outcome |
| `BlackboardEntry.confidence` | Test pass count |
| `Blackboard.round` | Git commit sequence |
| Experts read prior rounds | Agents read prior log entries |

The structural isomorphism is intentional: the same coordination
pattern works on both layers because the problem is the same —
independent experts composing results on a shared substrate.

---

## Future: Native A2A MCP Server

If Claude Code or a third party ships an A2A MCP server with
`post_entry` / `read_entries` / `subscribe` endpoints, these
workarounds can be replaced. Contract types already exist
(`BlackboardEntry`, `ExpertCapability`, `Blackboard`). The MCP
server is a thin serde layer over them.

Until then: `tee -a AGENT_LOG.md > /dev/null <<'EOF'`.

---

## Convergence note: Multi-file board ↔ A2A workarounds

Both systems implement the same kanban with the same wire format
(`tee -a` Markdown append). The difference is field granularity
and extensions:

| Multi-file board has MORE | A2A has MORE |
|---|---|
| `META/REQUESTS-FROM-AGENTS.md` inbox with BEHAVIOUR_QUESTION types | Branch pub/sub via `subscribe_pr_activity` (real-time) |
| P0/P1 severity classification (P2 does not exist) | Role-teleportation decision matrix |
| Anti-skim primitives (sentinel / proof-of-read / reading-depth) | Permission-whitelist snippet for `tee -a` |
| Meta-agent drift detection with 10 drift signals | Decision matrix: when Agent() vs. Teleport vs. Pub/Sub |

→ **Full convergence** is the combination of both toolsets. Multi-file
board + Workaround #2 (Pub/Sub) + #3 (Teleportation) + Decision Matrix
yields the combined A2A toolbox for multi-sprint setups.
