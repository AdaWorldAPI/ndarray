# Anti-Skim Agent Specification

> **Status:** DRAFT  ·  **Version:** 0.1.0  ·  **Last updated:** 2026-05-17
> **Format:** NLSpec (per [strongdm/attractor](https://github.com/strongdm/attractor))
> **Substrate:** Designed to compose on top of `coding-agent-loop-spec.md`;
> adds the verification + escalation layer attractor's spec omits.

---

## §1 Overview

### §1.1 Purpose

The Anti-Skim Agent is a per-worker LLM loop that is verifiably
read-before-acting. It treats agent **overconfidence** as the
primary failure mode (cf. Kahneman / Tversky System-1 easy-path,
Dunning-Kruger) and shifts the burden of proof onto the agent:

- Every claim about a file's content MUST be backed by a SHA-256 +
  line-count + reading-depth declaration.
- Every brief contains a sentinel token the agent MUST replay verbatim.
- A spot-check passes one of five lie-detector tests per supervisor pass.

### §1.2 Relation to `coding-agent-loop-spec.md`

The Anti-Skim Agent is implementable as a coding-agent-loop session
with the additions in §3–§6. Specifically:

- The `Session.run()` loop (coding-agent §2.5) is wrapped with a
  proof-of-read precondition (§7).
- The `steering_queue` (coding-agent §2.6) is extended with the
  typed stuck-protocol blockers in §5.
- The tool-call loop detector in coding-agent §2.10 is adopted
  verbatim and elevated to a system-level invariant (§6).
- Tool-output truncation (coding-agent §5) is adopted with one
  addition: truncated reads MUST appear in the proof-of-read entry
  with `depth=truncated`, not `depth=full`.

### §1.3 What this spec is NOT

- Not an orchestrator. Wave-level coordination is in
  [`autoattended-orchestrator-spec.md`](./autoattended-orchestrator-spec.md).
- Not a transport. Cross-agent messages flow through the file
  blackboard + handover files in
  [`agent-coordination-mcp-spec.md`](./agent-coordination-mcp-spec.md).

---

## §2 Terminology

| Term | Definition |
|---|---|
| **Sentinel token** | A unique opaque string in the worker's brief that the worker MUST replay verbatim in its first reply. |
| **Proof-of-read** | A typed entry `{ file, sha256, lines, depth }` declared by the worker for every file it consumed. |
| **Reading depth** | One of seven escalating levels: `grep` (anti) / `sed-partial` (anti) / `skim` / `read` / `thorough` / `troubleshooting` / `fan-out`. |
| **Lie-Detector test** | One of five tests (LD-1..5) the meta-agent / orchestrator spot-checks against a worker's output. |
| **Drift signal** | A passive indicator of skim or overconfidence (§4.3). |
| **Stuck-protocol blocker** | A typed entry in `META/REQUESTS-FROM-AGENTS.md` (one of five types) signaling "I cannot proceed". |
| **Tool-call loop** | A repeating pattern of length 1, 2, or 3 in the worker's last 10 tool calls. |

---

## §3 The Reading-Depth Ladder

### §3.1 Seven depths, two anti-depths

| Depth | Appropriate use | Proof |
|---|---|---|
| `grep` (anti) | Symbol locator AFTER or PARALLEL to a real read. | Worker declares `depth=grep`; NEVER `depth=full`. |
| `sed-partial` / `head-only` (anti) | Snapshotting a few lines for orientation. | Worker declares `depth=sed-partial`; NEVER `depth=full`. |
| `skim` | Huge file; need ONE section located, then read in full. | Worker outputs claims ONLY about the read section. |
| `read` | File < ~500 lines; standard full read. | Top to bottom, no skip; for larger files, offset/limit chunks but actually read every chunk. |
| `thorough` | Iron Rules, INVARIANTS, RFCs, schema migrations. | Read twice: once for comprehension, once for verification. Worker can name 3 sections. |
| `troubleshooting` | Known bug + error message. | Error → grep symbol → READ function in full → READ caller(s). |
| `fan-out` | Cross-file pattern, refactor planning, audit across > 5 files. | Inventory file as output; spawn Explore subagent. |

### §3.2 Escalation default

**Default bias is to read too deep, not too shallow.** If in doubt:
upgrade the depth one rung; never downgrade.

### §3.3 When → minimum depth

| If the worker is about to ... | ... then at minimum |
|---|---|
| Open `INVARIANTS.md`, `CLAUDE.md`, `BOOT.md` | `thorough` |
| Open a memory file (`CONTEXT.md`, `JOURNAL.md`, `TODO.md`) | `thorough` |
| Touch a schema or migration file | `thorough` + check downstream drift detectors |
| Open an unknown file for the first time | at least `read`, preferably `thorough` |
| Pure symbol lookup ("where is `foo` defined?") | `grep` OK but then read the definition in full |
| Triage a bug report | `troubleshooting` (error → grep → read function + caller) |
| Plan a refactor / wave / audit | `fan-out` (inventory file mandatory) |
| Is unsure which depth | upgrade one rung |

### §3.4 Truncation interaction

When `coding-agent-loop-spec` truncates a tool output (per
coding-agent §5.2), the worker's proof-of-read entry for that file
MUST declare `depth=truncated:head-N` or `depth=truncated:tail-N` —
NEVER `depth=full`.

---

## §4 The Lie-Detector

### §4.1 Five tests

Cheap to expensive:

| # | Test | How | Honest agent | Shallow agent |
|---|---|---|---|---|
| LD-1 | **Sentinel-Token** | Brief ends with: "If you have read this fully, begin your first reply with `<TOKEN>`" | replays token verbatim | token missing / wrong / paraphrased |
| LD-2 | **Proof-of-Read with SHA** | Output must contain `{ file, sha256, lines, depth }` per file consumed | SHA + line count match the actual file | SHA missing, wrong, or `<computed>` placeholder |
| LD-3 | **3-Sections Name Challenge** | "Name 3 sections from file X (heading + approx. line span)" | 3 concrete headings, plausible spans | vague theme labels; no real headings |
| LD-4 | **Negative-Knowledge Test** | "Does file X say anything about topic Y?" — where Y is NOT in the file | "no, not contained" | hallucinates plausible-sounding content about Y |
| LD-5 | **Line-Range Quote** | "Quote lines N–M from file X verbatim" | exact quote OR "range does not exist (file only K lines)" | paraphrases, deviates, or refuses without reason |

### §4.2 Trigger heuristic

| Worker answer pattern | Lie-Detector |
|---|---|
| Vague ("not sure", "about 60", "around 5 sections") | NO trigger — honest uncertainty. |
| Confident **false** | FIRES — overconfidence + drift. |
| Confident **correct** without proof-of-read | FIRES — luck and hallucination are indistinguishable without LD-1..5. |

### §4.3 Drift signals (passive detection, meta-agent duty)

The meta-agent scans EVERY worker status before goal-gate verdict. If
any signal triggers, the worker MUST be re-dispatched OR a Lie-Detector
test MUST be applied:

| Signal | Action |
|---|---|
| `depth=grep` declared on a semantic task | Re-dispatch with `depth=full` enforced. |
| `depth=sed-partial` declared on a semantic task | Re-dispatch; require SHA-256 over the whole file. |
| `depth=full` declared without `sha256` in proof-of-read | Stop; worker MUST back-fill SHA. |
| Status `notes` cites a non-existent section of a file | LD-3 equivalent failed post-facto → P0 re-dispatch. |
| Status `notes` is word-identical to a sibling agent's without cross-read note | Lockstep drift; spot-check LD-5 on one of the two. |
| Opening output enumerates a specific count ("62 lessons") WITHOUT proof-of-read on the source | Overconfidence trigger; agent MUST back-fill SHA + 3 entries by name. |
| Status touches an INVARIANT but its `proof_of_read` does not include `INVARIANTS.md` | Mandatory `thorough` re-read. |

### §4.4 Burden-of-proof rule

On suspicion: the worker MUST upgrade. Never "explain away" the
suspicion — either produce the proof, or honestly downgrade
(`depth=skim`/`grep` with a note).

---

## §5 Stuck Protocol

### §5.1 Five typed blockers

| Type | Meaning |
|---|---|
| `AMBIGUITY` | Spec is mehrdeutig. More than one sensible interpretation. |
| `MISSING_INVARIANT` | An Iron Rule should cover this case but does not; the convention is missing. |
| `SPEC_SOURCE_MISMATCH` | The authoritative spec says X; the reference source does Y. |
| `BEHAVIOUR_QUESTION` | Possible bug in the reference source; behavioral divergence requires RFC. |
| `EXTERNAL_DEPENDENCY` | Third-party system zickt; workaround unclear. |

### §5.2 The protocol

When stuck, the worker MUST:

1. STOP writing code.
2. Append ONE entry to `META/REQUESTS-FROM-AGENTS.md` with the
   §5.3 schema.
3. Set its `status.json` to `outcome=RETRY` and `notes=stuck:<type>`.
4. Idle until the meta-agent writes to `META/ANSWERS-TO-AGENTS.md`
   OR updates `META/INVARIANTS.md`.

### §5.3 Request schema

```markdown
## Agent {{agent_id}}, file <path>, timestamp <ISO 8601>

### Question
<one paragraph>

### Tried
<what was tried, what failed>

### Blocker
<one of: AMBIGUITY | MISSING_INVARIANT | SPEC_SOURCE_MISMATCH | BEHAVIOUR_QUESTION | EXTERNAL_DEPENDENCY>

### Proof-of-read attached
- file=<path-A> sha256=<...> lines=<...> depth=<...>
- file=<path-B> sha256=<...> lines=<...> depth=<...>
```

The proof-of-read MUST cover EVERY file the worker references in the
question. Stuck requests without proof-of-read are rejected at
intake (§7.5 in `autoattended-orchestrator-spec.md`).

### §5.4 Forbidden alternatives

The worker MUST NOT:

- Guess.
- Write a `TODO:` comment in code and proceed.
- Refactor outside the bundle "while waiting".
- Refile the same request twice without new information.

---

## §6 Tool-Call Loop Detection

### §6.1 Adopt attractor coding-agent §2.10

A worker MUST scan its own last `N=10` tool calls (signature = name +
arguments hash) for repeating patterns of length 1, 2, or 3.

```
Pattern length 1:  [A][A][A]                — same tool, same args, 3+ times
Pattern length 2:  [A][B][A][B]              — alternating, 2+ cycles
Pattern length 3:  [A][B][C][A][B][C]        — 3-cycle, 2+ cycles
```

### §6.2 Action on detection

When detected:

1. The worker MUST NOT issue the next predicted call in the pattern.
2. The worker MUST emit a typed RETRY-WITH-STEER status:
   `outcome=RETRY`, `notes=tool-call-loop:length=<L>:pattern=<sig>`.
3. The worker MAY file a `META/REQUESTS-FROM-AGENTS.md` entry under
   the `AMBIGUITY` blocker type (§5.1) if the loop has no obvious
   external resolution.

### §6.3 Anti-pattern AP9

This rule is also enforced by PP-13 brutally-honest-tester as
anti-pattern AP9 (tool-call-loop). PP-13's post-hoc detection
applies if the in-loop detector failed.

---

## §7 Proof-of-Read Schema

### §7.1 Per-file entry

```json
{
  "file": "path/relative/to/repo-root.md",
  "sha256": "abc...64-hex...def",
  "lines": 412,
  "depth": "thorough",
  "comment": "two-pass read; INVARIANTS §3.1-3.4 cited in worker output"
}
```

### §7.2 Allowed depth values

```
grep         sed-partial      skim
read         thorough         troubleshooting
fan-out      truncated:head-N truncated:tail-N
```

`grep`, `sed-partial`, `truncated:*` MUST NOT be combined with claims
that imply full knowledge of the file (LD-3 / LD-4 / LD-5).

### §7.3 Multiple-file entries

Workers MUST emit one entry per file consumed. The orchestrator's
goal-gate (PP-13) MAY spot-check up to three entries per worker per
wave via LD-2 (recompute the SHA, compare line count).

### §7.4 Storage

Proof-of-read entries live inside the worker's `status.json`
(per `autoattended-orchestrator-spec.md` §9.1). They are also
mirrored to `META/AGENT_LOG.md` as a one-line summary after each
chunk landed:

```
## YYYY-MM-DDTHH:MM — A<id> <file> chunk N/M
- chunk N landed (sha <short>)
- typecheck: green | <error count>
- proof-of-read: <file> sha=<short> lines=<N> depth=<D>
```

### §7.5 Sentinel-token storage

The sentinel token is mirrored from the worker's first-reply line
into the `status.json` `sentinel_token` field. Mismatch = FAIL.

---

## §8 Toolchain Tiers

### §8.1 Tier 1 (every PR — mandatory)

Owned by PP-13 brutally-honest-tester. Per-language adapter:

| Purpose | Rust | Python | TypeScript | Go |
|---|---|---|---|---|
| Lint, no warnings | `cargo clippy --all-targets --all-features -- -D warnings` | `ruff check` | `eslint --max-warnings 0` | `golangci-lint run` |
| Formatter check | `cargo fmt --check` | `ruff format --check` | `prettier --check` | `gofmt -l` |
| Advisory scan | `cargo audit` | `pip-audit` | `npm audit --omit=dev` | `govulncheck ./...` |
| Dep policy / vet | `cargo deny check` | `deptry .` | (project-defined) | `go vet -all` |
| Typecheck | (clippy implies) | `mypy --strict` | `tsc --noEmit --strict` | (compiler implies) |
| Tests | `cargo test --all-features` | `pytest` | `vitest run` | `go test ./...` |

### §8.2 Tier 2 (quality / maintenance — opt-in)

| Purpose | Rust | Python | TypeScript |
|---|---|---|---|
| Unused-dep detector | `cargo machete` | `deptry .` | `depcheck` |
| Unsafe scan | `cargo geiger` | `bandit` | (n/a) |
| Public-API SemVer compat | `cargo semver-checks check-release` | `griffe check` | `api-extractor` |
| Spellcheck | `cargo spellcheck` | CSpell | CSpell |

### §8.3 Tier 3 (heavier — opt-in, all stable)

| Purpose | Rust | Python |
|---|---|---|
| Bounded model checker | `kani` | (none standard) |
| Concurrency model checker | `loom` (lib) | (none standard) |
| Mutation testing | `cargo mutants` | `mutmut` |
| Coverage | `cargo tarpaulin` | `pytest --cov` |

### §8.4 Tier-1 invariant

For every PR, every language: the worker proves it satisfies a
no-warning gate BEFORE the orchestrator opens the PR. Tier-1 failure
maps directly to `outcome=FAIL` in the status file (§9.2 of
`autoattended-orchestrator-spec.md`).

---

## §9 Anti-Pattern Catalog (AP1..AP9)

PP-13 brutally-honest-tester actively hunts these:

| # | Anti-pattern | Detection | Finding |
|---|---|---|---|
| AP1 | Silent fallback that swallows errors (`unwrap_or_default()`, bare `try { ... } catch {}`, `result, _ := f()`) | grep PR diff for these idioms | P0 if in non-test code path |
| AP2 | Hardcoded secret / token / URL in source | grep for plausible secret regex | P0 always |
| AP3 | Compile-time guard dropped (e.g. `#[cfg(test)]` block removed in non-test context) | diff inspection | P0 if guard was load-bearing |
| AP4 | Test passes by tautology (asserts `true`, `assert x == x`) | read every new test body | P1 unless it's the ONLY test for the function (then P0) |
| AP5 | Behavior divergence from spec without merged RFC | spec vs implementation diff | P0 always |
| AP6 | Missing parity-test for a ported handler | check `tests/parity_<bundle>/` | P0 always |
| AP7 | New workspace dependency not declared in RFC | metadata diff vs `git log --oneline rfcs/` | P0 always |
| AP8 | `#[allow(clippy::*)]` / `# noqa` / `// eslint-disable` without rationale | grep for these directives, check commit body | P1 unless rationale is missing (then P0) |
| **AP9** | **Tool-call loop (length 1/2/3 over last 10 calls; see §6)** | **in-loop detector OR post-hoc replay of tool-call history** | **P1; re-dispatch with tighter scope** |

---

## §10 Definition of Done

An implementation is conformant if it satisfies ALL of:

- [ ] Every worker brief contains a unique `sentinel_token`.
- [ ] Every worker first-reply MUST begin with the sentinel verbatim.
- [ ] Every worker emits a `status.json` per
      `autoattended-orchestrator-spec.md` §9.1 with one
      proof-of-read entry per file consumed.
- [ ] Proof-of-read entries declare `sha256` + `lines` + a depth
      from §7.2's allowed list.
- [ ] Workers run the §6.1 tool-call loop detector after EVERY
      tool call, with `N=10`.
- [ ] On loop detection, the worker emits `outcome=RETRY` with
      `notes=tool-call-loop:...`.
- [ ] On stuck, the worker uses one of the five §5.1 blocker types
      and attaches proof-of-read for every referenced file.
- [ ] The meta-agent / supervisor spot-checks ONE of LD-1..LD-5 per
      worker per wave (rotating, so workers cannot game which test
      to fake).
- [ ] Drift signals (§4.3) are scanned for every worker before the
      goal-gate verdict.
- [ ] PP-13 runs the §8.1 Tier-1 toolchain for the worker's
      language; Tier-1 failure → `outcome=FAIL`.
- [ ] PP-13's anti-pattern scan covers AP1..AP9.
- [ ] Tool-output truncation (per `coding-agent-loop-spec.md` §5)
      surfaces in proof-of-read as `depth=truncated:head-N` or
      `depth=truncated:tail-N`; NEVER `depth=full`.
- [ ] `auto_status=false` is mandatory; missing `status.json` = FAIL.

---

## §11 Cross-Provider Parity Matrix

| Capability | Claude Code | OpenAI Codex / codex-rs | Gemini CLI | Notes |
|---|---|---|---|---|
| Sentinel-token replay (LD-1) | Available via system prompt addendum | Same | Same | Provider-agnostic; uses standard chat completion. |
| Proof-of-read schema (LD-2) | Worker writes status.json | Same | Same | File-based; provider-agnostic. |
| 3-sections challenge (LD-3) | Supervisor prompt | Same | Same | Provider-agnostic. |
| Negative-knowledge test (LD-4) | Supervisor prompt | Same | Same | Provider-agnostic. |
| Line-range quote (LD-5) | Supervisor prompt + file re-read by supervisor | Same | Same | Provider-agnostic. |
| Tool-call loop detection (§6) | Implementable in `Session.run()` loop wrapper | Same | Same | Loop-detector code is provider-independent (operates on local tool-call history). |
| Toolchain tiers (§8) | Bash tools | Bash tools | Bash tools | Per-language commands identical. |
| Truncation → `depth=truncated:*` | Wrapper around tool-output | Wrapper | Wrapper | Provider-specific in WHAT it truncates (different default tool sets), uniform in HOW it's recorded. |

---

## §12 Appendix A — Worker brief template

```markdown
You are agent {{agent_id}} in wave {{wave_id}} of sprint {{sprint_id}}.

You own bundle: {{bundle_name}}
Owned files (read-write):
{{owned_files_table}}

Read-only files:
{{read_only_files_table}}

Spec files (authoritative):
{{spec_files_table}}

INVARIANTS (read these first, depth=thorough):
- META/INVARIANTS.md  sha256={{invariants_sha}}  lines={{invariants_lines}}

Reading depth required (per anti-skim-agent-spec.md §3.3):
- INVARIANTS.md → thorough
- spec files → full
- reference source files → full
- skeleton files → read
- everything else → at least read

Status file: write to {{status_file_path}} matching the schema in
autoattended-orchestrator-spec.md §9.1.

When stuck: file ONE entry in META/REQUESTS-FROM-AGENTS.md per
anti-skim-agent-spec.md §5; set status to outcome=RETRY; idle.

Done criteria: per autoattended-orchestrator-spec.md §10 + this
spec's §10.

Tool-call loop detector: run after every tool call per §6.

SENTINEL TOKEN: {{sentinel_token}}

Begin your first reply with the sentinel token verbatim. Then state,
in one paragraph, what you understand the bundle to be. Then begin
work.
```

---

*End of `anti-skim-agent-spec.md`.*
