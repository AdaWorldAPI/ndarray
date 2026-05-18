# `.claude/ATT/` — Attractor-style NLSpecs of our kit

> **Status:** DRAFT  ·  **Version:** 0.1.0  ·  **Last updated:** 2026-05-17
>
> **What this is.** Our `.claude/EN/` kit's ideas, restated in the
> [strongdm/attractor](https://github.com/strongdm/attractor) NLSpec
> format so a coding agent can implement them directly.
> [NLSpec](https://github.com/strongdm/attractor#terminology) =
> "human-readable spec intended to be directly usable by coding agents
> to implement/validate behavior."
>
> **What this is NOT.** A replacement for `.claude/EN/`. The two are
> complementary: `.claude/EN/` is the operator's cheat-sheet
> (prose, in-session use); `.claude/ATT/` is the engineering spec
> (NLSpec, build-time use).

## The three specs

| File | Mirrors attractor's | Adds our innovation |
|---|---|---|
| [`autoattended-orchestrator-spec.md`](./autoattended-orchestrator-spec.md) | [`attractor-spec.md`](https://github.com/strongdm/attractor/blob/main/attractor-spec.md) (DOT-graph pipeline runner) | Wave-based 12-worker fan-out; 4-savant verdict gates (PP-13/14/15/16); 6 worker iron rules; sprint-token budget (~300k/wave); multi-file board pattern with single-mutable-file invariant |
| [`anti-skim-agent-spec.md`](./anti-skim-agent-spec.md) | [`coding-agent-loop-spec.md`](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md) (the per-LLM-call agent library) | Reading-Depth-Ladder (grep→read→thorough→fan-out); Lie-Detector LD-1..5 (sentinel token / proof-of-read SHA / 3-section challenge / negative-knowledge test / line-range quote); typed stuck-protocol blockers (AMBIGUITY / MISSING_INVARIANT / SPEC_SOURCE_MISMATCH / BEHAVIOUR_QUESTION / EXTERNAL_DEPENDENCY) |
| [`agent-coordination-mcp-spec.md`](./agent-coordination-mcp-spec.md) | [`unified-llm-spec.md`](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md) (provider-agnostic LLM SDK) | Three coordination layers (role-teleport / file-blackboard / branch-pub-sub) the way a native A2A MCP server should expose them; structured handover schema; decision matrix for when each layer fits |

## What we adopted from attractor (the five wins)

These are concrete things attractor's specs nail down that our prose
docs in `.claude/EN/` didn't have, now incorporated:

| # | Attractor concept | Lands in our NLSpec | Adoption |
|---|---|---|---|
| 1 | Typed `status.json` schema + 5-value `StageStatus` enum (attractor Appendix C: `{outcome, preferred_label, suggested_next_ids, context_updates, notes}`) | [`autoattended-orchestrator-spec.md` §9](./autoattended-orchestrator-spec.md#9-status-file-schema) | Adopted **with `auto_status=false` mandatory** (see "Where it conflicts" below). |
| 2 | DOT graph DSL for the workflow + lint rules (attractor §2 grammar + §7 validation: `reachability`, `start_no_incoming`, `goal_gate_has_retry`, `condition_syntax`) | [`autoattended-orchestrator-spec.md` §6](./autoattended-orchestrator-spec.md#6-sprint-plan-format) (DOT + YAML mirror) + [§7](./autoattended-orchestrator-spec.md#7-validation-rules) (WAVE-001..WAVE-017 with ERROR/WARNING severity) | Adopted with three wave-specific additions: `unique-write`, `declared-shared`, `auto-status-false`. |
| 3 | Context Fidelity ladder (attractor §5.4: `full` / `truncate` / `compact` / `summary:low/medium/high` with token budgets + edge > node > graph > default precedence) | [`autoattended-orchestrator-spec.md` §11A](./autoattended-orchestrator-spec.md#11a-context-fidelity) | Adopted with one tightening: `fidelity=truncate` does NOT exempt a worker from the §3.3 Reading-Depth-Ladder of `anti-skim-agent-spec.md`. |
| 4 | In-loop tool-call loop detection (attractor coding-agent §2.10: last 10 calls scanned for length-1/2/3 repeating patterns → inject steering warning) | [`anti-skim-agent-spec.md` §6](./anti-skim-agent-spec.md#6-tool-call-loop-detection) + AP9 in [§9](./anti-skim-agent-spec.md#9-anti-pattern-catalog-ap1ap9) | Adopted verbatim; elevated to a system-level invariant. PP-13's post-hoc AP9 catches what the in-loop detector misses. |
| 5 | Definition-of-Done checklists + Cross-Provider Parity Matrix per spec (attractor §10-style conformance tables) | Each NLSpec ends with `§ Definition of Done` + `§ Cross-{Language,Provider} Parity Matrix` | Adopted as the structural template. The 26-repo rollout is now machine-checkable. |

## Why this format

Three properties we get from attractor's NLSpec format that our prose
docs in `.claude/EN/` don't have:

1. **Definition of Done checklists** at the end of each spec — gives
   us a conformance test for "is this implementation complete?"
2. **Cross-Provider Parity Matrix** tables — gives us a per-language /
   per-runtime mapping so the same NLSpec can land in Rust, Python,
   TypeScript, Go without three-way drift.
3. **Validation rules with ERROR/WARNING/INFO severity** — turns
   linting into a deterministic process, not a judgment call.

## Where it conflicts with attractor's posture (and why we keep our position)

| Attractor's default | Our position | Why |
|---|---|---|
| `auto_status=true` (§4.5 + Appendix C: "if the handler writes no status, auto-generate SUCCESS") | `auto_status=false` is mandatory | This is exactly the silent-skim failure mode our Lie-Detector LD-1..5 exists to prevent. Missing status = FAIL, not SUCCESS. |
| Single-threaded graph traversal (§3.8: "Only one node executes at a time") | Wave fan-out is the baseline, not a special-case `parallel` node | Our token budget is per-wave (~300k for 12 workers), not per-node. Modeling waves as one giant `parallel` node is syntactically awkward. |
| Engine-level retries with exponential backoff (§3.5-3.6) | Stuck-agents file typed blockers in REQUESTS-FROM-AGENTS.md; meta-agent decides | Retries should be contextual and inspected, not silent and uniform. |
| `wait.human` default; LLM gates are test fixtures (§6.4: `AutoApproveInterviewer` "Used for automated testing") | LLM meta-agent is the production gate | Our meta-agent's plan review + P0/P1 PR review + inbox drain is a production role, not a test fixture. |
| Subagent depth = 1 default (coding-agent §7.3) | Wave fan-out routinely runs 12 workers from one orchestrator | We override depth to ≥2 — workers should be able to spawn sub-investigations. |

## Conformance

A repo that imports these NLSpecs is conformant if it satisfies the
"Definition of Done" checklist at the end of each spec. The lance-graph
implementation (see [`AdaWorldAPI/lance-graph` `.claude/agents/`](https://github.com/AdaWorldAPI/lance-graph/tree/main/.claude/agents))
is the most-mature reference; the WoA + woa-rs implementations are
the wave-based reference.

## Provenance

- Source kit: `.claude/EN/` in this repo (see [`.claude/EN/README.md`](../EN/README.md))
- Format inspiration: [strongdm/attractor](https://github.com/strongdm/attractor) (MIT-style NLSpecs)
- Distillation handover: [`META/HANDOVER-AGENTKIT-CONSOLIDATION-2026-05-17.md`](https://github.com/AdaWorldAPI/WoA/blob/main/META%2FHANDOVER-AGENTKIT-CONSOLIDATION-2026-05-17.md)  (in `AdaWorldAPI/WoA`)
- Sister handover (Rust hardening pass): [`META/HANDOVER-WOA-RS-AGENT-HARDENING-2026-05-17.md`](https://github.com/AdaWorldAPI/WoA/blob/main/META%2FHANDOVER-WOA-RS-AGENT-HARDENING-2026-05-17.md)

## Build

To turn these NLSpecs into a runnable implementation, supply this
prompt to a modern coding agent (Claude Code, Codex, OpenCode, Amp,
Cursor, ...):

```
codeagent> Implement the Autoattended Orchestrator + Anti-Skim Agent
           + Agent Coordination MCP as described by
           https://github.com/AdaWorldAPI/<repo>/tree/main/.claude/ATT
           together with strongdm/attractor as the substrate.
```
