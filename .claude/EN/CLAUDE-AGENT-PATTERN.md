# Agent Pattern Cheat-Sheet (project-agnostic, English canonical)

Lookup-style add-on to your repo's `.claude/CLAUDE.md`. Read this before
prompting any multi-agent work. Companion knowledge files under
`.claude/EN/knowledge/` and `.claude/EN/agents/`.

> Source: distilled from `AdaWorldAPI/WoA` `.claude/CLAUDE.md` §3.5 + §3.6
> as of 2026-05-17. Reference implementations: `WoA` (Python), `woa-rs`
> (Rust), `lance-graph` (Rust, most mature), `ndarray` (Rust, compact).

---

## 1. Compressed baseline

```
12 workers + 1 coordinator · autoattended · full authorization · auto-resolve · log all issues
plan/preflight → review → correct → sprint → review code → fix P0 → commit → repeat
```

## 2. Agent ensemble — Function · Activation · Card

| Function | Activation | Card |
|---|---|---|
| Worker-Impl | Sprint Phase 3 spawn out of `SPRINT-N-PLAN.md`-bundle row | `.claude/EN/agents/worker-template.md` |
| Meta-Agent (13th) | Phase 2 plan review + Phase 4 code review + continuous inbox-drain | `.claude/EN/agents/meta-agent.md` |
| PP-13 brutally-honest-tester | "about to commit" / "PR diff touches code" | `.claude/EN/agents/brutally-honest-tester.md` |
| PP-14 convergence-architect | PRE-PLAN divergent ideation (optional, see knowledge/autoattended-multi-agent-pattern.md §3) | — |
| PP-15 baton-handoff-auditor | "cross-crate types" / "DTO match" / "lib.rs / mod.rs touch" / "sprint handover" / "ID collision" | `.claude/EN/agents/baton-handoff-auditor.md` |
| PP-16 preflight-drift-auditor | "before sprint spawn" / "preflight check" / "verify spec against main" | `.claude/EN/agents/preflight-drift-auditor.md` |

Full specs in `.claude/EN/agents/<role>.md`. Pattern theory in
`.claude/EN/knowledge/autoattended-multi-agent-pattern.md`. Coordination
primitives in `.claude/EN/knowledge/a2a-workarounds.md`.

---

## 3. Reading-Depth-Ladder

Anti-skim discipline as a lookup. **Default bias: read too deep rather
than too shallow.** If in doubt, upgrade the depth; never downgrade.

| Depth | When appropriate | Guardrail / Proof-of-Depth |
|---|---|---|
| `grep` (anti) | **NEVER** as primary read. Only as a symbol locator AFTER or PARALLEL to a real read. | If you grepped: declare `depth=grep`, **never** `depth=full`. Grep is not a substitute for reading. |
| `sed -n` / `awk '/re/'` / `head` / `tail`-only (anti) | **NEVER** as primary read. Partial-range tools deliver lines without section context — hallucination trap. | If you used `sed -n '10,20p'`: declare `depth=sed-partial`, NOT `depth=full`. Use `Read(offset, limit)` with a clear anchor or read the whole file. Never claim a 5-line snippet means "file X read". |
| `skim` | Huge file, you need ONE section located | Find anchor → read that section in full. Never claim you know the whole file. Output: only claims about the read section. |
| `read` | File < ~500 lines, standard read | Top to bottom, no skips. For larger files: offset/limit chunks but actually read every chunk. |
| `thorough read` (twice-full) | Iron Rules, INVARIANTS, RFCs, schema migrations, patches before live-apply | Read twice: once for comprehension, once for verification. Self-check: can I name 3 sections? |
| `troubleshooting` | Known bug + error message | Error → grep symbol → READ function in full → READ caller(s). No spot-fix without call-site context. |
| `fan-out research` | Cross-file pattern, refactor planning, audit across >5 files | Spawn an Explore subagent OR write a suspect list, then read each file fully. Inventory file as output. Never trust in-context inference alone. |

### When → minimum depth

| If you are about to… | …then at minimum |
|---|---|
| Read memory files (CONTEXT / JOURNAL / TODO) at session start | `thorough read` |
| Read `.claude/CLAUDE.md` / `BOOT.md` / RFCs / INVARIANTS | `thorough read` |
| Touch a schema / migration file | `thorough read` + check downstream drift detectors |
| Open an unknown file for the first time | at least `read`, preferably `thorough` |
| Pure symbol lookup ("where is `foo` defined?") | `grep` OK but then read the definition in full |
| Triage a bug report | `troubleshooting` (error → grep → read function + caller) |
| Plan a refactor / wave / audit | `fan-out research` (inventory file mandatory) |
| You are unsure which depth | upgrade one rung |

---

## 4. Lie-Detector — shallowness + drift detection

> **P0 rule:** No claims without thorough comprehension of the subject
> AND all adjacent context.
>
> **Cognitive frame (Kahneman/Tversky + Dunning-Kruger):** the problem is
> NOT vagueness ("about 60 lessons") — that is **honest uncertainty**
> (System 2 reporting in). The problem is **overconfidence**: a specific
> claim without the read that would support it. System-1 "easy-path"
> heuristic yields a plausible-sounding answer BEFORE System 2 verifies.
>
> **Trigger heuristic:** vague answer ("not sure") = honest → no trigger.
> Confident false answer = lie-detector fires. Confident correct answer
> without proof-of-read = also fires (luck vs. hallucination is
> distinguishable only via LD-1..5).

Five concrete tests, cheap to expensive:

| # | Test | How | Honest agent | Shallow agent |
|---|---|---|---|---|
| LD-1 | **Sentinel-Token** | Brief ends with: "If you have read this fully, begin your first reply with `<TOKEN>`" | replays token verbatim | token missing / wrong / paraphrased |
| LD-2 | **Proof-of-Read with SHA** | Output must contain: `Read: file=X sha256=Y lines=Z depth=full` | SHA + line count match | SHA missing, wrong, or `<computed>` placeholder |
| LD-3 | **3-sections name challenge** | "Name 3 sections from file X (heading + approx. line span)" | 3 concrete headings, plausible spans | vague theme labels, no real headings |
| LD-4 | **Negative-knowledge test** | "Does file X say anything about topic Y?" — where Y is NOT in the file | "no, not contained" | hallucinates plausible-sounding content |
| LD-5 | **Line-range quote** | "Quote lines N-M from file X verbatim" | exact quote OR "range does not exist (file only K lines)" | paraphrases, deviates, or refuses without reason |

### Drift signals (passive detection, meta-agent duty)

| Signal | What meta sees | Action |
|---|---|---|
| `depth=grep` on a semantic task | HANDOVER field mismatch | Re-dispatch with mandatory `depth=full` |
| `depth=sed-partial` or `depth=head-only` on a semantic task | Partial-range tool abused as primary read | Re-dispatch; require SHA over the whole file |
| `depth=full` without `sha256` | Proof field gap | Stop; agent must back-fill proof |
| FINDING cites a non-existent section | LD-3 equivalent failed post-facto | P0 re-dispatch; tighten sentinel requirement |
| FINDING is word-identical to a sibling agent's without cross-read note | Lockstep drift (both LLMs defaulted without real read) | Spot-check LD-5 on one of the two |
| Opening output enumerates lessons with confidence ("62 lessons") WITHOUT proof-of-read on JOURNAL | Overconfidence trigger | Hard stop: agent must produce SHA + 3 L-numbers by name. Approximate count would have been OK (= honest uncertainty). |
| Conjecture section contains what INVARIANTS clearly decided | INVARIANTS drift (not read) | Mandatory `thorough` re-read of INVARIANTS |

**Burden of proof on suspicion lies with the agent:** never "explain
away" suspicion — either produce the proof or honestly downgrade to
`depth=skim/grep`.

---

## 5. Cross-references

- 6-step orchestrator loop, sprint sizing, 4-savant scope partition,
  worker iron rules → `.claude/EN/knowledge/autoattended-multi-agent-pattern.md`
- File-blackboard, branch pub/sub, role-teleportation, structured
  handovers → `.claude/EN/knowledge/a2a-workarounds.md`
- Worker brief slots → `.claude/EN/agents/worker-template.md`
- Meta-agent role + inbox protocol → `.claude/EN/agents/meta-agent.md`
- Post-impl quality gate → `.claude/EN/agents/brutally-honest-tester.md`
- Cross-boundary audit → `.claude/EN/agents/baton-handoff-auditor.md`
- Pre-spawn drift check → `.claude/EN/agents/preflight-drift-auditor.md`
