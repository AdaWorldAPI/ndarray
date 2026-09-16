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
  - "tests yes, compile no": a targeted `cargo test`/`clippy` against the
    **shared** workspace `target/` is fine; a bare compile-only
    (`cargo check`/`build`) is wasted residue — clippy already compiles.
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
