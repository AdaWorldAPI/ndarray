# Agent Model Tiers — Haiku for supervised harvest and mapping

DECISION (operator, 2026-10-14): "Use Haiku agents for grindwork and take
Haiku off the guard list, but watch the agents closely and start using them
for very focused and scoped tasks that you use for cheap harvest discovery";
"Goal is to outsource deeper mapping due to enormous token equivalent of the
substrate." The canonical statement lives in lance-graph
(`.claude/knowledge/tiered-agent-execution-protocol.md` § Haiku harvest
tier); this file is the ndarray copy of the rules that apply here.

## Tiers

| tier | does | never |
|---|---|---|
| Opus (orchestrator) | plans, decides, reviews, compiles, lints, tests | mechanical loops |
| Sonnet | edits from a spec, drafting from a complete brief | synthesis across sources, gate decisions |
| Haiku | read-only harvest, deep mapping, exact spec edits | authoring content, conclusions, review, cargo |

## The five Haiku cards (`.claude/agents/`)

- `harvest-scout`: candidates as `path:line` + the matching line.
- `census-counter`: raw counts, each with the exact command.
- `quote-harvester`: verbatim text with line numbers; FOUND / NOT FOUND.
- `substrate-mapper`: a fact map of one module or directory, written to one
  file in the session scratchpad.
- `spec-editor`: exact edits to one file, each applied only if its anchor
  occurs exactly once.

No `guarded-executor` here: in ndarray only the orchestrator compiles
(`agent-cargo-hygiene.md`), because a second compiler in the shared
`target/` replaces the realization the orchestrator last built and makes a
measured tier unattributable.

## Watching closely

1. Harvest is navigation: search finds, reading decides. The orchestrator
   Reads every item it relies on; a Haiku list is never cited as evidence.
2. Re-run at least one reported count per `census-counter` reply.
3. A `hits: 0` is never taken as absence; Read the place the thing would live.
4. For every map, Read at least two mapped items at their stated lines;
   `mentioned by` is candidates, never a call graph.
5. Read the diff of every `spec-editor` edit before committing.
6. Record per run, in `.claude/blackboard.md`: card, runs, items checked,
   items wrong, items missed.
7. A card that misses, miscounts or concludes twice on one kind of task is
   not used for that kind again; that kind goes to Sonnet.

## What makes a brief Haiku-sized

One question; a closed search space (exact paths); patterns written out with
their aliases and fully-qualified spellings; a hit cap; nothing that needs
two sources held in mind together.
