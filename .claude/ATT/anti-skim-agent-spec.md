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

## §13 Skeleton-Fill Contract (Protocol A)

> Added 2026-05-18. Applies to workers operating in
> `autoattended-orchestrator-spec.md` Protocol A (implementation),
> where PP-16's preflight has produced a commented-out Rust
> skeleton (per §14.4 of the orchestrator spec).

### §13.1 What the worker receives

Three inputs:

1. The skeleton file(s) at `skeleton_output_path` containing
   `todo!("SOURCE: <path>:<lines>")` macros at every body site.
2. The original spec / reference-source files referenced in each
   `SOURCE:` annotation.
3. A SHA-256 of both the skeleton AND each referenced source file,
   pinned by PP-16 at preflight time.

### §13.2 Worker's obligations

For every `todo!("SOURCE: <path>:<lines>")` the worker fills:

1. **Read the source range** at the declared depth (§3.3); default
   for ports is `full`.
2. **Record proof-of-read** for the source file at the declared
   line range (§7.1).
3. **Replace `todo!(...)` with the body**, preserving the surrounding
   signature exactly. The signature came from PP-16; the worker
   does not change it without an Iron-Rule-amending RFC.
4. **Quote the source line range in the commit message body** (per
   `autoattended-orchestrator-spec.md` §5.6).
5. **Confirm the skeleton's SHA-256 against the pinned value** at
   start of work. If the skeleton has drifted (e.g. a sibling worker
   touched it), the worker MUST STOP and file an `EXTERNAL_DEPENDENCY`
   blocker in `META/REQUESTS-FROM-AGENTS.md`.

### §13.3 Forbidden in skeleton-fill mode

- Changing a signature provided by the skeleton (RFC required).
- Adding a `todo!()` that did NOT come from the skeleton (would
  bypass PP-16's source-annotation discipline).
- Removing a `// SAFETY:` comment from an `unsafe` block (PP-13
  will reject; PP-16 wrote it for a reason).
- Filling bodies in a file outside `skeleton_output_path` (§5.1
  unique-file write discipline).

### §13.4 Reading depth required

| Source kind | Minimum depth |
|---|---|
| `SOURCE: <reference-source>:<lines>` (the function being ported) | `full` for the named line range |
| `UNSAFE-SOURCE: <reference-source>:<lines>` | `thorough` for the named range AND the function's callers |
| The skeleton file itself | `read` (to confirm the surrounding signature you're filling into) |
| `META/INVARIANTS.md` | `thorough` (per §3.3) |

### §13.5 Validation rules

| Rule | Description | Severity |
|---|---|---|
| `FILL-001 source-range-read` | Every `todo!("SOURCE: P:L-M")` the worker fills MUST appear in the worker's proof-of-read with `file=P, lines covering L-M, depth=full`. | ERROR |
| `FILL-002 skeleton-sha-pin` | Worker MUST verify the skeleton's SHA-256 against the value PP-16 pinned, at start of work. Drift → STOP + `EXTERNAL_DEPENDENCY`. | ERROR |
| `FILL-003 signature-preserved` | Worker MUST NOT change a signature provided by the skeleton without a merged RFC. Diff inspection by PP-15 enforces. | ERROR |
| `FILL-004 unsafe-safety-preserved` | `// SAFETY:` comments on `unsafe` blocks from the skeleton MUST survive into the filled body. PP-13 enforces. | ERROR |
| `FILL-005 no-new-todo` | Worker MUST NOT introduce new `todo!()` calls; every `todo!()` in the filled file MUST trace back to a skeleton entry. | ERROR |

### §13.6 Definition of Done (skeleton-fill)

- [ ] Every `todo!()` in the assigned skeleton file(s) is replaced.
- [ ] No new `todo!()` introduced (FILL-005).
- [ ] Skeleton SHA matched at start; final SHA different (FILL-002).
- [ ] Proof-of-read covers every `SOURCE:` annotation (FILL-001).
- [ ] All signatures preserved (FILL-003).
- [ ] All `// SAFETY:` comments preserved (FILL-004).
- [ ] Tier-1 toolchain green on filled file (§8.1).
- [ ] Status file written per `autoattended-orchestrator-spec.md` §9.1.

---

## §14 Reading Phases (orthogonal to depth)

> The Reading-Depth Ladder in §3 says **how much** of a file you cover.
> The Reading Phases in this section say **what you do** with what you
> covered. Both are required for a read to count as complete.

### §14.1 The four phases

| # | Phase | Question it answers | Output the worker MUST be able to produce |
|---|---|---|---|
| 1 | **Survey** | "What is in this file? What is its shape?" | Section list with line numbers; file shape (N sections, K loc, language); top-level headline. |
| 2 | **Evaluation** | "What of this matters for my task?" | A relevance map: which sections / line ranges are relevant to the current bundle, prioritized P0 / P1 / unused. |
| 3 | **Critical findings** | "What is wrong, missing, contradictory, drifted?" | A typed finding list (severity P0 / P1) — Iron-Rule violations, spec-vs-source drift, missing sections, anchors that no longer resolve, dead references. |
| 4 | **Internalize** | "Can I act on this without re-reading?" | Pass LD-3 (3-section name challenge), LD-4 (negative-knowledge), LD-5 (line-range quote). Can paraphrase faithfully; can answer "what is NOT in this file?". |

Phase ordering is typically **Survey → Evaluation → Critical findings →
Internalize**, but findings often surface DURING internalize (the act of
internalizing exposes contradictions). A complete reading reaches all
four; a partial reading stops earlier and MUST declare so.

### §14.2 Phase × depth matrix

| Depth | Survey | Evaluation | Critical findings | Internalize |
|---|:-:|:-:|:-:|:-:|
| `grep` (anti) | partial | no | no | no |
| `sed-partial` / `head-only` (anti) | partial | partial | no | no |
| `skim` | yes | partial | partial (per-section only) | no |
| `read` | yes | yes | partial | partial |
| `thorough` | yes | yes | **yes** | **yes** |
| `troubleshooting` | yes | yes | yes (focused on the bug) | yes (focused) |
| `fan-out` | yes (per-file shallow) | yes | yes | partial (per file) |
| `truncated:head-N` / `truncated:tail-N` | partial | partial | no | no |

Only `thorough` and `troubleshooting` (within its focused scope) achieve
the full ladder. `fan-out` achieves internalize *per-file* but the
inventory output is the synthesis — the worker MUST treat each file in
the inventory as needing its own `thorough` pass before acting on it.

### §14.3 Phase outputs in the worker's status.json

The `proof_of_read` schema (§7.1) is extended with a `phase_reached`
field naming the highest phase completed for each file:

```json
{
  "file": "META/INVARIANTS.md",
  "sha256": "fa39a3...",
  "lines": 412,
  "depth": "thorough",
  "phase_reached": "internalize",
  "phases_evidence": {
    "survey": "9 sections; INVARIANTS canonical structure recognized",
    "evaluation": "§BBB + §UPSERT-PATTERN are load-bearing for this bundle",
    "critical_findings": "§UPSERT-PATTERN line 187 contradicts the customer.ttl ogit:CustomerWriter mandatory-attributes (filed REQUESTS-FROM-AGENTS.md#A4-2026-05-18T14:22)",
    "internalize": "can answer LD-3/4/5 verbatim"
  }
}
```

The `phases_evidence` map is OPTIONAL when `phase_reached=survey`
(no evidence beyond the section list is required) but BECOMES required
at `phase_reached >= evaluation` because that phase claims judgment.

### §14.4 Mapping Lie-Detector tests to phases

Each LD-1..LD-5 test (§4.1) probes a specific phase. The meta-agent
SHOULD rotate the spot-check across the four phases so workers cannot
game which phase to fake.

| Test | Probes phase | What a passing answer proves |
|---|---|---|
| LD-1 Sentinel-Token | **Survey** of the brief | The worker actually loaded the brief into context. |
| LD-2 Proof-of-Read with SHA | **Survey** of the file | The worker accessed the file at the declared content. |
| LD-3 3-Sections Name Challenge | **Survey + Evaluation** | The worker can locate structure AND chose which sections to attend to. |
| LD-4 Negative-Knowledge Test | **Internalize** | The worker built a faithful mental model — can answer "what is NOT in this file" without hallucinating. |
| LD-5 Line-Range Quote | **Internalize + Critical findings** | The worker can recall verbatim AND detect drift between recall and source. |

A worker declaring `phase_reached=internalize` MUST be able to pass
LD-3, LD-4, **and** LD-5. Spot-check failure at a claimed phase ⇒
phase claim is rejected and the proof-of-read is auto-downgraded to
the highest demonstrably-passed phase.

### §14.5 Per-file required phase by file kind

Different file kinds REQUIRE different minimum phases. The worker
brief MUST declare per-file phase requirements alongside depth
requirements:

| File kind | Minimum depth | Minimum phase |
|---|---|---|
| `META/INVARIANTS.md` | `thorough` | **internalize** |
| `CLAUDE.md` / `BOOT.md` / RFCs | `thorough` | **internalize** |
| Memory files (`CONTEXT.md` / `JOURNAL.md` / `TODO.md`) | `thorough` | **internalize** |
| Spec files for the bundle (e.g. TTL, OpenAPI, JSON-schema) | `full` (read) | **internalize** |
| Reference-source for ports | `full` | **internalize** |
| Skeleton files the worker fills | `read` | **evaluation** |
| Sibling-bundle files (read-only context) | `skim` | **evaluation** |
| Files referenced for general orientation | `skim` | **survey** |
| Files referenced for symbol lookup only | `grep` | **survey** |

When a worker stops at a phase lower than required, the read DOES
NOT count as complete — even if the depth was sufficient. Both axes
must clear the bar.

### §14.6 Critical findings escalation

Findings produced in phase 3 (Critical findings) are routed by
severity:

| Finding severity | Filed where | Worker action |
|---|---|---|
| P0 — Iron-Rule violation in input, spec-vs-source contradiction, broken anchor that affects the bundle | `META/REQUESTS-FROM-AGENTS.md` with blocker type from §5.1; worker idles | STOP work; do not proceed to commit |
| P1 — minor inconsistency, dead reference outside bundle scope, typo, stale comment | `Altlasten.md` / `TECH_DEBT.md` row with the worker's bundle id | Continue work; the orchestrator triages later |
| INFO — observation that does not affect the bundle | Notes field in `status.json`; not filed elsewhere | Continue work |

A worker that internalizes but does NOT escalate a P0 finding is in
violation: missing-escalation is itself a P0 finding that PP-13
will catch (anti-pattern AP1 — "silent fallback that swallows
errors" — generalizes here as "silent skim that swallows findings").

### §14.7 Validation rules

| Rule | Description | Severity |
|---|---|---|
| `PHASE-001 phase-declared` | Every `proof_of_read` entry MUST include a `phase_reached` field. Absent ⇒ treated as `phase_reached=survey` (the weakest claim). | ERROR |
| `PHASE-002 phase-monotonic-with-depth` | A `phase_reached` claim MUST be consistent with the §14.2 matrix. E.g. `depth=grep` + `phase_reached=internalize` is invalid. | ERROR |
| `PHASE-003 phase-evidence-required` | When `phase_reached >= evaluation`, the `phases_evidence` map MUST be present with non-empty entries for each phase claimed. | ERROR |
| `PHASE-004 file-kind-phase-bar` | When the worker brief declares a minimum phase per §14.5 for a file, the worker's `phase_reached` for that file MUST be ≥ the declared minimum. | ERROR |
| `PHASE-005 critical-findings-routed` | P0 findings produced during phase 3 MUST be filed to `META/REQUESTS-FROM-AGENTS.md` BEFORE the worker commits any code that depends on the affected input. | ERROR |
| `PHASE-006 internalize-passes-LD-3-4-5` | A claim of `phase_reached=internalize` MUST survive spot-checks of LD-3, LD-4, and LD-5 on rotation. Failure ⇒ auto-downgrade to highest-passed phase. | ERROR |

### §14.8 Definition of Done (per-file read)

A read of a single file is complete when ALL of:

- [ ] Depth from §3 is at the declared level.
- [ ] Phase from §14.1 reaches the required minimum per §14.5.
- [ ] `proof_of_read` entry includes `sha256`, `lines`, `depth`,
      `phase_reached`, and `phases_evidence` for `evaluation+`.
- [ ] Any P0 critical finding is filed before the worker commits.
- [ ] LD-3 / LD-4 / LD-5 spot-checks pass if `phase_reached=internalize`
      is claimed.

---

## §15 Cognitive Anti-Patterns (CA1..CA4)

> Where AP1..AP9 (§9) catch **output** problems — the code itself
> looks wrong — CA1..CA4 catch **cognition** problems — the way the
> worker arrived at the output is wrong. CA findings often surface
> at the same time as AP findings; the distinction is that CA fixes
> require process change (re-read, re-think, re-spawn) while AP
> fixes can sometimes be edited in place.
>
> The cognitive anti-patterns are jointly owned: the meta-agent
> spots them during PR review via the Lie-Detector (§4); PP-13
> spots them during code review by correlating commit timestamps
> with proof-of-read timestamps.

### §15.1 The four cognitive anti-patterns

| # | Name | What it looks like | Detection signature | Counter-pattern | Severity |
|---|---|---|---|---|---|
| **CA1** | **Cognitive dissonance** | Worker sees a contradiction between two authoritative sources (spec vs reference source, INVARIANT vs comment, TTL vs code) and resolves by hand-waving or picking one without investigation. The dissonance gets papered over instead of escalated. | Output contains phrases like "I went with", "I chose", "I preferred Y because it feels right / is already there / compiles", with no corresponding `SPEC_SOURCE_MISMATCH` entry in `META/REQUESTS-FROM-AGENTS.md`. | File `SPEC_SOURCE_MISMATCH` (§5.1) and idle until the meta-agent writes an RFC. Never resolve dissonance unilaterally. | **P0** |
| **CA2** | **Dunning-Kruger overconfidence** | Worker confidently claims knowledge in an area where its actual depth is shallow. The agent doesn't know what it doesn't know, so its sense of certainty is un-calibrated. The output reads as definite where the read was thin. | A `phase_reached=internalize` claim (§14) that fails LD-3 / LD-4 / LD-5 spot-checks. A specific numerical claim ("62 lessons", "the signature has 3 arguments") with no corresponding `proof_of_read` SHA. Confident paraphrase where the source actually says something else. | Auto-downgrade the phase claim to the highest demonstrably-passed phase (per `PHASE-006`). Re-read at the proper depth + phase. If the worker keeps overshooting, route them to a smaller bundle. | **P0** |
| **CA3** | **Kahneman/Tversky easy-path (System-1 short-circuit)** | Worker pattern-matches surface features of the input — "file looks like a CRUD handler, so it must be a CRUD handler" — without running the System-2 check (read the actual function body, compare against the spec). Easy-path is fastest when surface matches reality, but devastating when it doesn't. | The first reply is plausibly-correct-sounding but the proof-of-read is `depth=grep` or `depth=sed-partial`. Output describes structure in generic terms ("standard route handler", "typical migration") rather than specific terms ("the function on lines 47-91 dispatches on req.path"). | Force System-2: require LD-2 (SHA + line count) and LD-5 (line-range quote) before accepting any structural claim. Reading depth MUST be at least `read`; phase MUST be at least `evaluation` before any output that makes a structural claim. | **P0** |
| **CA4** | **Eager amok** | Worker starts writing code (or commits, or pushes) before completing the required reading + planning phases. Enthusiasm runs ahead of discipline. The work *looks* productive — there's a diff — but it's built on incomplete understanding. | First code-write timestamp predates the `proof_of_read` SHA-pin timestamps for one or more required files (§14.5). `status.json` shows commits landing while `phase_reached` is still `survey` or `evaluation` on files that should be at `internalize`. Worker's narrative jumps from brief-read to first-commit with no visible thinking step. | STOP and require complete proof-of-read for ALL files at their required §14.5 minimum phase BEFORE any commit. The Iron Rule applies regardless of how "obvious" the implementation seems. | **P0** |

### §15.2 Why all four are P0

Each of CA1..CA4 produces output that *looks* right. AP1..AP9 produce
output that looks wrong (a `unwrap_or_default()` is visible; a missing
parity test is visible). Cognitive anti-patterns produce output whose
correctness depends entirely on the unverifiable claim that the
worker read + understood + thought before writing. They are P0
because they break the trust contract the meta-agent depends on when
reviewing diffs without re-doing the worker's read.

### §15.3 Joint ownership: who catches what

| Anti-pattern | Catch site | Detection method |
|---|---|---|
| CA1 cognitive dissonance | Meta-agent (PR review) + PP-15 (cross-source diff) | Grep PR commit messages for "I went with" / "I chose" / "preferred" against spec-vs-source mismatch; check `REQUESTS-FROM-AGENTS.md` for absence of corresponding blocker. |
| CA2 Dunning-Kruger overconfidence | Meta-agent (Lie-Detector spot-check) | Rotate LD-3 / LD-4 / LD-5 on one worker per wave; cross-check `phase_reached` claims against actual passing. |
| CA3 Kahneman/Tversky easy-path | Meta-agent (proof-of-read inspection) + PP-13 (output-vs-source diff) | Worker output reads as paraphrase rather than quote; SHA missing; reading depth declared inconsistent with output claims. |
| CA4 eager amok | PP-13 (commit-timestamp audit) + Meta-agent (status.json ordering check) | Commit timestamps predate the `proof_of_read` SHA-pin timestamps; `phase_reached` was below the required minimum at first commit. |

### §15.4 Counter-patterns: how a healthy worker behaves

| Anti-pattern | Healthy alternative |
|---|---|
| CA1 | "I noticed the spec says X but the existing code does Y. Filing `SPEC_SOURCE_MISMATCH`. Idling." |
| CA2 | "I am confident about §3 of INVARIANTS.md (depth=thorough, phase=internalize). I am uncertain about §6 (depth=skim, phase=survey). I will deepen §6 before claiming anything about it." |
| CA3 | "Before describing this file's structure, here is `proof_of_read: { file=X, sha256=..., depth=read, phase_reached=evaluation }`. The structure is: [specific section names with line numbers]." |
| CA4 | "Reading phase complete on all 4 required files. Filing critical findings (zero P0). Status: `phase_reached=internalize` on all files. NOW beginning the first commit." |

### §15.5 Validation rules

| Rule | Description | Severity |
|---|---|---|
| `CA-001 dissonance-escalation` | When a worker's output mentions spec-vs-source divergence, `META/REQUESTS-FROM-AGENTS.md` MUST contain a corresponding `SPEC_SOURCE_MISMATCH` blocker entry. Absence ⇒ CA1 P0 finding. | ERROR |
| `CA-002 confidence-calibration` | A `phase_reached=internalize` claim that fails LD-3 / LD-4 / LD-5 spot-check ⇒ CA2 P0; phase claim auto-downgraded. | ERROR |
| `CA-003 system-2-required-before-structural-claim` | Any structural claim about a file's content (function count, section names, signature shapes) MUST be preceded by a `proof_of_read` entry with `depth >= read` AND `phase_reached >= evaluation`. Otherwise CA3 P0. | ERROR |
| `CA-004 read-before-write-ordering` | `status.json` MUST show all required-minimum `proof_of_read` entries timestamped BEFORE the first code-commit timestamp for files at the §14.5-required minimum phase. Otherwise CA4 P0. | ERROR |

### §15.6 Mindset-level relation to the four savants

The four savants in `autoattended-orchestrator-spec.md` §4.0 each
have a *cognitive* posture that resists a specific CA:

| Savant | Mindset | Primarily resists |
|---|---|---|
| PP-13 brutally-honest-tester | "what would break in production at 3 a.m. that the author talked themselves out of seeing?" | CA1 (talking-yourself-out-of-seeing IS cognitive dissonance) + CA4 (production doesn't break because of enthusiasm) |
| PP-14 convergence-architect | "what could this become that we aren't seeing?" | CA3 (easy-path closes possibilities prematurely) |
| PP-15 baton-handoff-auditor | "do these contracts actually line up at the seams?" | CA1 (dissonance often hides at handoffs) |
| PP-16 preflight-drift-auditor | "does the plan still match the system as it actually is?" | CA2 (overconfidence in a plan that has drifted from reality) |

The full 4-savant pass is jointly an antibody against all four
cognitive anti-patterns. A wave that runs only one savant catches
only the CAs that savant's mindset resists; a wave that runs the
cooperative council (orchestrator spec §15) catches all four.

### §15.7 Definition of Done (cognitive-hygiene)

A worker passes the cognitive-hygiene gate when ALL of:

- [ ] `CA-001` clean: every spec-vs-source mention is matched by a filed `SPEC_SOURCE_MISMATCH` blocker.
- [ ] `CA-002` clean: every `phase_reached=internalize` claim survives LD-3/4/5 spot-checks.
- [ ] `CA-003` clean: every structural claim is backed by a sufficient `proof_of_read`.
- [ ] `CA-004` clean: read timestamps precede commit timestamps for all required-minimum files.
- [ ] Worker did not output `I went with` / `I chose` / `I preferred` without an associated blocker filing.

---

*End of `anti-skim-agent-spec.md`.*
