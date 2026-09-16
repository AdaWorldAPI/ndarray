# Agent Output Durability — `tee -a`, and why a brief's SIZE is a reliability parameter

## The incident, 2026-09-16

Two Sonnet workers were dispatched in the same message, on disjoint files.

| worker | brief | outcome |
|---|---|---|
| A | ONE file, ONE goal (vectorize two methods) | returned with a full report and a landed edit |
| B | THREE parts, FOUR files, twelve functions + a parity group + a correction | **vanished after 3-5 hours having written NOTHING** |

`ListAgents` showed no running agent. There was no partial file, no scratch
output, no trace — the work was not slow, it was lost, and nothing about it was
recoverable.

## Rule 1 — every worker tees its progress to its own tag-file

A worker that writes only at the end has a single point of failure: a timeout, a
tool error, or an encoding problem takes the whole run with it. Build output
incrementally and let each chunk land on disk as it is produced:

```sh
: > "$TAG"                       # init once
cat <<'CHUNK' | tee -a "$TAG" > /dev/null
...
CHUNK
```

Each chunk is visible in the session log AND on disk, so partial progress
survives a timeout and the orchestrator can salvage it. The sibling `MedCare-rs`
repo already carries this as its own discipline for any file over ~200 lines,
for exactly the same reason ("partial progress survives a timeout; the final
file is assembled atomically before commit").

**One writer per file.** A worker tees to ITS OWN tag-file, never to a shared
log — a shared append-log is a lost-write race, which is the same defect the
substrate removed at runtime and must not be re-created one layer up
(lance-graph `E-AGENT-LOG-SHARED-SINK-ANTIPATTERN-1`).

## Rule 2 — brief size is a reliability parameter, not just a cost

The A/B contrast above is one observation, not a law, but it points one way and
it is cheap to obey: **one chunk = one file, one deliverable.** Split a wave into
sequential chunks and dispatch them one at a time. A brief that names three parts
and four files is not "efficient", it is a bet that nothing goes wrong for hours.

Corollary for the orchestrator: prefer a chunk small enough that you could do it
yourself if the dispatch is lost. Then a lost dispatch costs one chunk, not an
afternoon.
