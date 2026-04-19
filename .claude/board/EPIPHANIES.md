# ndarray — Epiphanies (append-only)

## 2026-04-19 — Prompt↔PR ledger is 10⁷× cheaper than code grep
**Status:** FINDING
**Scope:** @workspace-primer domain:bookkeeping

To answer "what did we ship for topic X":

- **Grep across code:** ~100 MB of Rust across N crates, ~25M tokens of context, minutes of agent turns.
- **Grep the ledger:** one `grep X .claude/board/PROMPTS_VS_PRS.md` returns `<prompt file> | #N <title>`. ~25 tokens, sub-second.

Seven orders of magnitude cheaper. The pairing **prompt-file ↔ PR** is the
minimum addressable record of "this artifact was built to answer this
brief" — the hyperlink that replaces re-discovery by full-text scan.

The line is mechanical bookkeeping (Haiku-level, no synthesis). The
value accumulates on every subsequent "what about X" query thereafter:
ledger-first, code-never-unless-necessary.

Cross-ref: PR #213 (lance-graph, 41 prompts × merged PRs), PR #110
(ndarray, 25 prompts × merged PRs). Both shipped in ~90s on a dumb
enumerate+match+append loop. No code reads, no MCP, no synthesis.
