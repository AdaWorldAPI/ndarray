> **Sprache:** Deutsch · siehe `../README.md` für die englische Quellfassung.

# `.claude/ATT/` — NLSpecs unseres Kits im Attractor-Stil

> **Status:** DRAFT  ·  **Version:** 0.1.0  ·  **Stand:** 2026-05-17
>
> **Was das ist.** Die Ideen unseres `.claude/EN/`-Kits, neu formuliert
> im [strongdm/attractor](https://github.com/strongdm/attractor) NLSpec-Format,
> damit ein Coding-Agent sie direkt umsetzen kann.
> [NLSpec](https://github.com/strongdm/attractor#terminology) =
> "human-readable spec intended to be directly usable by coding agents
> to implement/validate behavior."
>
> **Was das NICHT ist.** Kein Ersatz für `.claude/EN/`. Die beiden sind
> komplementär: `.claude/EN/` ist der Cheat-Sheet für Operatoren
> (Prosa, in-session-Use); `.claude/ATT/` ist die Engineering-Spec
> (NLSpec, build-time-Use).

## Die drei Specs

| Datei | Spiegelt Attractors | Ergänzt unsere Innovation |
|---|---|---|
| [`autoattended-orchestrator-spec.md`](./autoattended-orchestrator-spec.md) | [`attractor-spec.md`](https://github.com/strongdm/attractor/blob/main/attractor-spec.md) (DOT-Graph-Pipeline-Runner) | Wave-basierter 12-Worker-Fan-out; 4-savant Verdict-Gates (PP-13/14/15/16); 6 Worker-Iron-Rules; Sprint-Token-Budget (~300k/Wave); Multi-File-Board-Pattern mit single-mutable-file-Invariante |
| [`anti-skim-agent-spec.md`](./anti-skim-agent-spec.md) | [`coding-agent-loop-spec.md`](https://github.com/strongdm/attractor/blob/main/coding-agent-loop-spec.md) (die Per-LLM-Call-Agent-Library) | Reading-Depth-Ladder (grep→read→thorough→fan-out); Lie-Detector LD-1..5 (Sentinel-Token / Proof-of-Read SHA / 3-Section-Challenge / Negative-Knowledge-Test / Line-Range-Quote); typisierte Stuck-Protocol-Blocker (AMBIGUITY / MISSING_INVARIANT / SPEC_SOURCE_MISMATCH / BEHAVIOUR_QUESTION / EXTERNAL_DEPENDENCY) |
| [`agent-coordination-mcp-spec.md`](./agent-coordination-mcp-spec.md) | [`unified-llm-spec.md`](https://github.com/strongdm/attractor/blob/main/unified-llm-spec.md) (Provider-agnostisches LLM-SDK) | Drei Koordinations-Layer (Role-Teleport / File-Blackboard / Branch-Pub-Sub) so wie sie ein nativer A2A-MCP-Server exponieren sollte; strukturiertes Handover-Schema; Decision-Matrix dafür, wann welcher Layer passt |

## Was wir von Attractor übernommen haben (die fünf Wins)

Das sind konkrete Dinge, die Attractors Specs festnageln und die unseren
Prosa-Docs in `.claude/EN/` gefehlt haben — jetzt eingearbeitet:

| # | Attractor-Konzept | Landet in unserer NLSpec | Übernahme |
|---|---|---|---|
| 1 | Typisiertes `status.json`-Schema + 5-Wert-`StageStatus`-Enum (Attractor Appendix C: `{outcome, preferred_label, suggested_next_ids, context_updates, notes}`) | [`autoattended-orchestrator-spec.md` §9](./autoattended-orchestrator-spec.md#9-status-file-schema) | Übernommen **mit verpflichtendem `auto_status=false`** (siehe "Wo es kollidiert" unten). |
| 2 | DOT-Graph-DSL für den Workflow + Lint-Regeln (Attractor §2 Grammatik + §7 Validierung: `reachability`, `start_no_incoming`, `goal_gate_has_retry`, `condition_syntax`) | [`autoattended-orchestrator-spec.md` §6](./autoattended-orchestrator-spec.md#6-sprint-plan-format) (DOT + YAML-Mirror) + [§7](./autoattended-orchestrator-spec.md#7-validation-rules) (WAVE-001..WAVE-017 mit ERROR/WARNING-Severity) | Übernommen mit drei Wave-spezifischen Zusätzen: `unique-write`, `declared-shared`, `auto-status-false`. |
| 3 | Context-Fidelity-Ladder (Attractor §5.4: `full` / `truncate` / `compact` / `summary:low/medium/high` mit Token-Budgets + Vorrang edge > node > graph > default) | [`autoattended-orchestrator-spec.md` §11](./autoattended-orchestrator-spec.md#11-context-fidelity) | Übernommen mit einer Verschärfung: `fidelity=truncate` entbindet einen Worker NICHT von der §3.3-Reading-Depth-Ladder aus `anti-skim-agent-spec.md`. |
| 4 | In-Loop-Tool-Call-Loop-Detection (Attractor coding-agent §2.10: letzte 10 Calls scannen auf wiederholende Patterns der Länge 1/2/3 → Steering-Warning einspielen) | [`anti-skim-agent-spec.md` §6](./anti-skim-agent-spec.md#6-tool-call-loop-detection) + AP9 in [§9](./anti-skim-agent-spec.md#9-anti-pattern-catalog-ap1ap9) | Wortgleich übernommen; auf System-Level-Invariante erhoben. PP-13s Post-hoc-AP9 fängt, was der In-Loop-Detector verpasst. |
| 5 | Definition-of-Done-Checklisten + Cross-Provider-Parity-Matrix pro Spec (Konformanz-Tabellen im Stil von Attractor §10) | Jede NLSpec endet mit `§ Definition von Fertig` + `§ Cross-{Language,Provider}-Parity-Matrix` | Als strukturelles Template übernommen. Der 26-Repo-Rollout ist jetzt maschinell prüfbar. |

## Warum dieses Format

Drei Eigenschaften, die wir aus Attractors NLSpec-Format bekommen und
die unseren Prosa-Docs in `.claude/EN/` fehlen:

1. **Definition-von-Fertig-Checklisten** am Ende jeder Spec — gibt uns
   einen Konformanz-Test für "ist diese Implementierung fertig?"
2. **Cross-Provider-Parity-Matrix**-Tabellen — geben uns ein Per-Language /
   Per-Runtime-Mapping, sodass dieselbe NLSpec in Rust, Python, TypeScript,
   Go landen kann ohne Drei-Wege-Drift.
3. **Validierungs-Regeln mit ERROR/WARNING/INFO-Severity** — macht aus
   Linting einen deterministischen Prozess, kein Bauchgefühl.

## Wo es mit Attractors Haltung kollidiert (und warum wir bei unserer Position bleiben)

| Attractors Default | Unsere Position | Warum |
|---|---|---|
| `auto_status=true` (§4.5 + Appendix C: "wenn der Handler keinen Status schreibt, auto-generiere SUCCESS") | `auto_status=false` ist Pflicht | Genau das ist der Silent-Skim-Failure-Mode, gegen den unser Lie-Detector LD-1..5 existiert. Fehlender Status = FAIL, nicht SUCCESS. |
| Single-Threaded Graph-Traversal (§3.8: "Nur ein Node läuft zur Zeit") | Wave-Fan-out ist die Baseline, kein Spezial-Fall `parallel`-Node | Unser Token-Budget ist pro Wave (~300k für 12 Workers), nicht pro Node. Waves als einen riesigen `parallel`-Node zu modellieren ist syntaktisch hässlich. |
| Engine-Level-Retries mit Exponential-Backoff (§3.5-3.6) | Stuck-Agents filen typisierte Blocker in REQUESTS-FROM-AGENTS.md; Meta-Agent entscheidet | Retries sollen kontextuell und inspiziert sein, nicht silent und uniform. |
| `wait.human` als Default; LLM-Gates sind Test-Fixtures (§6.4: `AutoApproveInterviewer` "Used for automated testing") | LLM-Meta-Agent ist das Production-Gate | Unser Meta-Agent macht Plan-Review + P0/P1-PR-Review + Inbox-Drain als Production-Rolle, nicht als Test-Fixture. |
| Subagent-Tiefe = 1 Default (coding-agent §7.3) | Wave-Fan-out fährt routinemäßig 12 Workers aus einem Orchestrator | Wir überschreiben die Tiefe auf ≥2 — Workers sollen Sub-Investigations spawnen dürfen. |

## Konformanz

Ein Repo, das diese NLSpecs einbindet, ist konform, wenn es die
"Definition von Fertig"-Checkliste am Ende jeder Spec erfüllt. Die
lance-graph-Implementierung (siehe [`AdaWorldAPI/lance-graph` `.claude/agents/`](https://github.com/AdaWorldAPI/lance-graph/tree/main/.claude/agents))
ist die reifste Referenz; die WoA + woa-rs-Implementierungen sind die
Wave-basierte Referenz.

## Provenienz

- Quell-Kit: `.claude/EN/` in diesem Repo (siehe [`.claude/EN/README.md`](../../EN/README.md))
- Format-Inspiration: [strongdm/attractor](https://github.com/strongdm/attractor) (NLSpecs im MIT-Stil)
- Distillation-Handover: [`META/HANDOVER-AGENTKIT-CONSOLIDATION-2026-05-17.md`](https://github.com/AdaWorldAPI/WoA/blob/main/META%2FHANDOVER-AGENTKIT-CONSOLIDATION-2026-05-17.md)  (in `AdaWorldAPI/WoA`)
- Schwester-Handover (Rust-Hardening-Pass): [`META/HANDOVER-WOA-RS-AGENT-HARDENING-2026-05-17.md`](https://github.com/AdaWorldAPI/WoA/blob/main/META%2FHANDOVER-WOA-RS-AGENT-HARDENING-2026-05-17.md)

## Build

Um diese NLSpecs in eine lauffähige Implementierung zu verwandeln,
gib einem modernen Coding-Agent (Claude Code, Codex, OpenCode, Amp,
Cursor, ...) diesen Prompt:

```
codeagent> Implement the Autoattended Orchestrator + Anti-Skim Agent
           + Agent Coordination MCP as described by
           https://github.com/AdaWorldAPI/<repo>/tree/main/.claude/ATT
           together with strongdm/attractor as the substrate.
```

*Ende der Datei README.md.*
