# Agent Cargo Hygiene — one target dir, no 12× build residue

## The problem

When the orchestrator fans out a fleet of subagents (the Sonnet build/fix
agents), each agent that runs a full `cargo build`/`check`/`test` in its **own**
isolated working copy materialises its own `target/`. This workspace's
`target/` is ~7 GB. Twelve agents in twelve worktrees = ~84 GB of duplicated
build residue and twelve cold compiles competing for the same cores.

## The rule

- **Opus (orchestrator + Opus agents): run cargo freely.** No restriction.
- **Sonnet fleet agents: do NOT each run a full compile.** They edit code and
  reason; they must not spawn isolated worktrees or trigger their own cold
  `cargo build`/`check`/`test` that each grow a separate 7 GB `target/`.
  - ⊘ **SUPERSEDED — this file used to carve out "tests yes, compile no": a
    targeted `cargo test`/`clippy` against the shared `target/` is fine;
    clippy already compiles.** Both halves are now wrong, and the file
    contradicted its own BACKEND POLLUTION section below (which says the
    prohibition is absolute) for as long as the carve-out stood. Corrected
    2026-09-16 after coderabbit flagged the inconsistency on PR #309:

    1. **Operator ruling: workers do not run cargo. At all.** Not `build`,
       not `check`, not `test`, not `clippy`. The pollution argument below is
       what makes this absolute rather than a budget — a worker's plain
       `cargo test` takes `.cargo/config.toml` (v3) and REPLACES whatever
       realization the orchestrator last built, so the next probe reports a
       tier nobody can reconstruct.
    2. **"clippy already compiles" is false** (operator correction to my own
       framing): clippy type-checks and lints, it does not produce a runnable
       artifact. Practical consequence for the ORCHESTRATOR's own gates, which
       is where cargo is still allowed: a green `cargo clippy` proves types and
       lints, never that the thing builds and runs. Evidence for a landing is a
       `cargo run`/`cargo test` result, not a lint.
- **Verification is centralised.** The orchestrator (Opus) runs
  `cargo fmt` + `cargo clippy` + `cargo test` **once**, in the single shared
  `target/`, after the fleet's edits land. One build, not twelve.

## How the orchestrator fans out work

- Spawn the fleet **without** `isolation: "worktree"` so all agents share the
  one repo checkout and one `target/`.
- Tell each agent explicitly: *edit only; do not run `cargo build`/`check`; do
  not create a worktree; the orchestrator compiles and lints centrally.*
- After edits, the orchestrator runs the gates (`cargo fmt -p <crate>`,
  `cargo clippy -p <crate>`, `cargo test -p <crate>`) — keeping the tree
  `cargo clippy -- -D warnings`-clean (see `CLAUDE.md` Hard Rules) and
  `cargo fmt`-clean on the pinned toolchain, with no residue blowup.


## The second reason, and it is the stronger one: BACKEND POLLUTION

(Operator, 2026-09-16: *"Würdest du mit jedem worker kompilieren hättest du
target residue backend pollution."* Residue is the cost. Pollution is the
correctness failure, and it was not written down here before.)

The one shared `target/` holds **one realization at a time**. A target-cpu
change invalidates the cache, so a worker running a plain `cargo test` — which
takes `.cargo/config.toml`, i.e. **v3/AVX2** — after the orchestrator built
`--config .cargo/config-v4.toml` does not merely ADD residue. It **replaces**
the v4 artifacts with v3 ones. The next probe then reports whichever tier
compiled last, and nothing in its output says which.

That is exactly the defect recorded as plan §17 (`gemm-ternlog-mask-consolidation-v1.md`),
one scale down: a timing without its target-cpu is an anecdote. The difference
is severity. On the orchestrator's own runs the tier is merely UNLABELLED, and a
re-run under a named config fixes it. With N workers compiling on their own
schedule it becomes UNATTRIBUTABLE — no one can reconstruct which backend a
number came from, because the interleaving is gone.

**So: one shared `target/`, one realization, one compiler — the orchestrator.**
Workers edit. This is why the prohibition is absolute rather than a budget.
