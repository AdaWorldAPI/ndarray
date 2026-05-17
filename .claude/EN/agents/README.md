# `.claude/EN/agents/` — Agent Ensemble Index

The project-agnostic 5-card agent ensemble distilled from
`AdaWorldAPI/WoA` + `AdaWorldAPI/woa-rs` + `AdaWorldAPI/lance-graph`
(2026-05-17). Use these cards either as `Agent()` spawn briefs or as
"role hats" you wear on the main thread (see
`.claude/EN/knowledge/a2a-workarounds.md` Workaround 3).

## The 5 cards

| Card | Phase | Verdict scale | When |
|---|---|---|---|
| [`worker-template.md`](./worker-template.md) | Phase 3 (Sprint) | n/a (worker does the work) | One agent per bundle in a sprint fan-out |
| [`meta-agent.md`](./meta-agent.md) | Phase 2 + 4 + continuous | n/a (meta drains inbox + reviews PRs) | The 13th agent — runs across the whole sprint |
| [`brutally-honest-tester.md`](./brutally-honest-tester.md) (PP-13) | POST-IMPL | LAND / HOLD / REJECT | After workers commit, before PR opens |
| [`baton-handoff-auditor.md`](./baton-handoff-auditor.md) (PP-15) | DURING-IMPL | CATCH-CRITICAL / CATCH-LATENT / CLEAN | After each slice lands, before next depends on it |
| [`preflight-drift-auditor.md`](./preflight-drift-auditor.md) (PP-16) | PRE-SPAWN | SPAWN-CLEAR / SPAWN-CAUTION / SPAWN-BLOCKED | After plan approved, before worker fan-out |

## What's NOT here

- **PP-14 convergence-architect** (PRE-PLAN ideation) — exists in
  lance-graph's ensemble but is OPTIONAL for most repos. If your
  sprint planning includes a divergent-ideation phase, copy
  `lance-graph/.claude/agents/convergence-architect.md`.
- **Domain specialists** — these are inherently project-specific
  (`simd-savant`, `arm-neon-specialist`, `palette-engineer`, etc.).
  Keep them in your repo's `.claude/agents/` (NOT under `EN/agents/`).

## Verdict-vocabulary non-overlap (design invariant)

Each card has a **non-overlapping verdict vocabulary** so a finding
cannot cross phases without re-classification:

| Verdict | Owner |
|---|---|
| LAND / HOLD / REJECT | PP-13 only |
| CATCH-CRITICAL / CATCH-LATENT / CLEAN | PP-15 only |
| SPAWN-CLEAR / SPAWN-CAUTION / SPAWN-BLOCKED | PP-16 only |
| GO / GO-WITH-CONDITIONS / BLOCK | meta-agent (plan review) only |
| P0 / P1 | meta-agent (PR review) only |

If you see a finding tagged with the wrong vocabulary, the wrong
agent wrote it. Route it back.

## How to spawn one of these

### As an `Agent()` subprocess (parallel, isolated context)

```
Agent({
  subagent_type: "general-purpose",  // or specialized type
  description: "<short>",
  prompt: "<contents of the .md file with {{ ... }} slots filled>"
})
```

### As a role-hat on the main thread (full session context)

```
1. Read `.claude/EN/agents/<role>.md` thoroughly
2. Read its required knowledge docs:
   - `.claude/EN/knowledge/autoattended-multi-agent-pattern.md`
   - `.claude/EN/knowledge/a2a-workarounds.md` (if coordination required)
3. Do the work in-context. No subprocess spawned.
```

See `.claude/EN/CLAUDE-AGENT-PATTERN.md §2` for the activation
trigger table.
