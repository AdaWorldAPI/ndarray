# `.claude/EN/` — Project-Agnostic Multi-Agent Kit (English Canonical)

This is the project-agnostic distillation of the multi-agent pattern hardened
on `AdaWorldAPI/WoA` + `AdaWorldAPI/woa-rs` on 2026-05-17. It is **additive**:
it does not replace any pre-existing `.claude/` content in this repo.

## What lives here

```
.claude/EN/
├── README.md                                    ← this file
├── CLAUDE-AGENT-PATTERN.md                      ← agnostic agent cheat-sheet
│                                                  + Reading-Depth-Ladder
│                                                  + Lie-Detector (LD-1..5)
├── knowledge/
│   ├── autoattended-multi-agent-pattern.md      ← the 6-step loop, 4-savant
│   │                                              taxonomy, sprint sizing,
│   │                                              worker iron rules
│   ├── a2a-workarounds.md                       ← file-blackboard, branch
│   │                                              pub/sub, role-teleportation,
│   │                                              structured handover
│   ├── reading-depth-ladder.md                  ← anti-skim primitive
│   └── lie-detector.md                          ← shallowness detection
└── agents/
    ├── README.md                                ← agent ensemble index
    ├── worker-template.md                       ← slot-based worker brief
    ├── meta-agent.md                            ← 13th agent (review + inbox)
    ├── brutally-honest-tester.md                ← PP-13 (POST-IMPL)
    ├── baton-handoff-auditor.md                 ← PP-15 (DURING-IMPL)
    └── preflight-drift-auditor.md               ← PP-16 (PRE-SPAWN)
```

## How to adopt this kit in your repo

1. **Read** `CLAUDE-AGENT-PATTERN.md` first. It is the cheat-sheet.
2. **Read** `knowledge/autoattended-multi-agent-pattern.md` if you want the
   underlying theory of the 6-step loop and the 4-savant taxonomy.
3. **Read** `knowledge/a2a-workarounds.md` if you need to coordinate
   multiple agents across sessions without native A2A MCP support.
4. When you spawn a worker: fill the `{{ ... }}` slots in
   `agents/worker-template.md` and pass it to the worker agent.
5. When you run a sprint: the meta-agent (`agents/meta-agent.md`) is the
   13th persona; it reviews plans + PRs and drains the agent inbox.
6. **Toolchain adaptation:** every file references `<toolchain>` /
   `<lint>` / `<test>` placeholders. Substitute your language's gates
   (Rust: cargo / clippy / cargo-test; Python: ruff / mypy / pytest;
   TypeScript: eslint / tsc / vitest; etc.).

## What this kit does NOT include

- **Language-specific tooling configuration.** Use your repo's existing
  `Cargo.toml` / `pyproject.toml` / `package.json` / `go.mod`.
- **Behavior rules.** Project-specific behavior rules belong in your
  repo's `.claude/CLAUDE.md` Iron Rules section, not here.
- **Board files.** Sprint state (`Stand.md` / `STATUS_BOARD.md`),
  open debts (`Altlasten.md` / `TECH_DEBT.md`), lessons learned
  (`Goldstaub.md` / `EPIPHANIES.md`) are per-repo — see
  `knowledge/autoattended-multi-agent-pattern.md §6` for the multi-file
  board pattern.

## Provenance + cross-references

- Canonical reference implementation: `AdaWorldAPI/WoA` `.claude/v0.1/CLAUDE-CONTEXT.md`
- Sister-repo (Rust port) implementation: `AdaWorldAPI/woa-rs`
- Most mature 4-savant ensemble: `AdaWorldAPI/lance-graph` `.claude/agents/`
- Compact worker-role ensemble: `AdaWorldAPI/ndarray` `.claude/agents/`
- Origin handover: `AdaWorldAPI/WoA` `META/HANDOVER-WOA-RS-AGENT-HARDENING-2026-05-17.md`

## Companion DE/ kit

A German-language sibling lives at `.claude/DE/` when a repo's primary
communication language is German. The DE kit mirrors this EN kit; either
can stand alone.
