# PR-X12 Docs Audit — session MAOO0 perspective

> Date: 2026-05-25
> Author: session MAOO0 (the session that authored much of the recent
> PR-X12 doc damage in PR #198, then was corrected by the maintainer
> across a long review).
> Status: **recommendations-only**. No edits to any audited doc. This
> file is one of several independent audits the maintainer is collecting
> for a cross-audit drift check.

## Method & independence note

- Every claim below tagged **[grounded]** is backed by an **end-to-end
  read of the cited file** (not grep/tail/head/sed/offset fragments).
  Fragment-reading is the documented root cause of the damage (see §2),
  so this audit refuses to use it for judgments.
- Claims tagged **[unverified]** are things I did NOT ground. They are
  listed honestly as TODO rather than asserted.
- I deliberately did **not** read the parallel audit in PR #200
  (`.claude/PR-X12-docs-audit.md`, session wN5kw). The maintainer's plan
  is a drift check across independent audits; reading the other audit
  first would defeat that.

## Load-bearing summary

The PR-X12 perspective-doc cluster is a self-referential system that
over-extrapolated from one real, modest, well-scoped artifact
(`src/simd_soa.rs`, tagged **PR-X1** — layout-only SoA carriers,
fully tested) into a sprawling present-tense "canon" describing a
substrate that does not exist as documented. The single primary root
cause is fragment-reading: sessions (including mine) grepped excerpts
instead of reading whole files, formed confident pseudo-models, and
laundered each other's enthusiasm into successive docs. The most
load-bearing factual error I introduced and merged is a flat denial of
the runtime CPU-detection that the real code (`src/simd_caps.rs`,
`src/simd.rs`) actually implements.

---

## 1. STATUS

### 1.1 What the docs claim to be

A cluster of `.claude/knowledge/pr-x12-*.md` "perspective docs" plus
two "canon" docs (`pr-x12-substrate-canon-resolutions.md`,
`pr-x12-canon-resolutions-delta.md`) describing a video-codec substrate
("PR-X12") built on `ndarray::hpc` / `ndarray::simd`, with a numbered
resolutions series (R-1 … R-15), named architectural decisions (M:E-*),
and phased plans (Plan G/H).

### 1.2 What is actually true (grounded)

- **`src/simd_soa.rs` is PR-X1, not PR-X12.** [grounded — read whole]
  It is a 373-line, layout-only `MultiLaneColumn` (an `Arc<[u8]>`
  carrier with typed 64-byte chunk iterators), explicitly documented as
  "No `#[target_feature]`, no per-arch imports, no raw intrinsics," with
  10 passing tests. This is the real "SoA is implementation-ready"
  milestone the perspective docs extrapolated from. It is solid, small,
  and unrelated to most of what PR-X12 docs claim.
- **The SIMD polyfill is real and works as the W1a contract says.**
  [grounded — read `src/simd.rs`, `src/simd_soa.rs` whole] Consumers
  write `crate::simd::F32x16` and never name a backend file. `simd.rs`
  re-exports one backend per build via `cfg(target_feature = ...)`
  (`simd_avx512` / `simd_avx2` / `simd_neon` / `scalar`). The module
  header states the intent verbatim: "detect once, dispatch forever …
  When `std::simd` stabilizes: swap this file. Zero consumer changes."
- **Runtime CPU detection exists and is the supported mechanism.**
  [grounded — read `src/simd_caps.rs`, `src/simd.rs` whole]
  `simd_caps.rs` defines `#[non_exhaustive] SimdCaps` (24 capability
  bits including `avx512vnni`, `amx_tile`, `amx_int8`, `amx_bf16`,
  `asimd_dotprod`), a `static CAPS: LazyLock<SimdCaps>` detect-once
  singleton (`simd_caps.rs:111`), detection via `__cpuid_count(7,0)` /
  `__cpuid_count(7,1)` / `is_x86_feature_detected!` /
  `is_aarch64_feature_detected!`, an `ArmProfile` enum (A53/A72/A76 Pi
  tiers), convenience methods (`has_avx512_vnni`, `has_amx`, …), and a
  full test module. `simd.rs` additionally has a `LazyLock<Tier>`
  detect-once (`detect_tier()`, `simd.rs:49-92`). This is exactly the
  "LazyLock CPU-detect-once + detailed lookup table" mechanism; it is
  not a hypothetical.
- **AMX is real, via stable inline asm.** [grounded — read
  `src/simd_amx.rs` whole] `LDTILECFG` + `.byte`-encoded `TILEZERO` /
  `TILERELEASE`, CPUID + `_xgetbv(0)` XCR0 checks, and a Linux
  `prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` syscall. Doc comment:
  "Rust intrinsics: NIGHTLY ONLY … Inline asm: STABLE." The one
  `target_os = "linux"` gate in the SIMD tree lives here.

### 1.3 Current damage state in merged master

- **PR #198 (merged, my work)** — commits `1bb4561f`, `8b1d3764`,
  `8d043389`. Doc edits to `pr-x12-woa-multiarch-orchestration.md` and
  `pr-x12-substrate-canon-resolutions.md`. These merged my
  hallucinations (see §3).
- **PR #200 (open)** — adds a parallel audit `.claude/PR-X12-docs-audit.md`
  (session wN5kw). Not read by me, by design.
- **PR #199 (merged)** + recent master commits — the doc cluster is
  still *actively growing*: `f0412198`..`48a14dc2` add PhiSpiral256 Leaf
  Planetarium, 3DGS render-depth certification, tensor-container
  capstone, EWA SYRK BLAS MKL cross-pollination, 4×4 3DGS cognitive
  shader SoA plans. [unverified content] The accretion rate is itself a
  status concern: new perspective docs are landing faster than the
  existing ones are being verified.
- **Dangling commit `c3c49042`** — my `blasgraph → bgz17` retarget of
  woa §8.3; force-pushed off its branch, unreachable from any ref. The
  original woa §8.3 (with the `M:E-H` citation) survives intact in
  merged master. [grounded — read woa doc whole, line 315-317]

---

## 2. ROOT CAUSE

**Primary: fragment-reading.** Sessions used grep / tail / head / sed /
Read-with-offset to form judgments about files they never read whole.
A grep hit plus three lines of surrounding context produced confident
claims that were wrong about the larger context. Everything below is
downstream of this.

The downstream cascade has four compounding layers:

1. **Tense collapse.** The R-* series is *future-conditional* planning
   for a Struct-of-Arrays transition. Sessions read R-N rows as
   *present-tense canon*. "In a perfect world, after the SoA transition,
   this would be canonical" became "this is canonical." [architectural
   truth supplied by maintainer; consistent with `simd_soa.rs` being
   tagged PR-X1 and being layout-only]
2. **Scope inflation (enthusiasm laundering).** Genuine excitement that
   `src/simd_soa.rs` reached an implementation-ready milestone *in the
   SoA context* got written up as unqualified "canonical," stripping the
   scope. Later sessions read the unqualified word and extrapolated
   outward.
3. **Correctness-class collapse.** The docs flatten three distinct
   classes into one undifferentiated "kernel":
   - bit-exact (blasgraph — the canonical kernel),
   - top-k-ranking-preserving-under-lab-conditions (EWA splat at
     10000×10000),
   - lossy (bgz17, and its siblings cam_pq / highheelbgz / bgz-hhtl-d).
   Treating lossy `bgz17` as a drop-in for bit-exact blasgraph is a
   soundness violation, not a naming preference. [architectural truth
   supplied by maintainer]
4. **Performative correction.** When wrongness was flagged, edits
   word-substituted ("dispatch" → "polyfill") or renamed symbols instead
   of deleting redundant/incorrect content — cosmetic acknowledgment
   that left the error in place or made it worse (my §3 "no runtime CPU
   detection" rewrite is the worst instance: it replaced a roughly-right
   capability-struct description with a flatly false absolute).

---

## 3. HALLUCINATION INVENTORY (what I introduced/merged)

Grouped by severity. All locations in
`.claude/knowledge/pr-x12-woa-multiarch-orchestration.md` unless noted.

### 3.1 Factually false against real code (highest severity)

- **"No runtime CPU detection / no `if has_avx512 else …`" (§3, line
  110, merged).** Directly contradicted by `simd_caps.rs:111`
  (`LazyLock<SimdCaps>`), `simd_caps.rs:158-207` (`SimdCaps::detect`
  with `is_x86_feature_detected!`), and `simd.rs:49-92` (`detect_tier`,
  `LazyLock<Tier>`). The real architecture detects once at runtime and
  caches. [grounded]
- **Deleting the §3.2 capability struct (commit `8b1d3764`).** I removed
  a `HwCaps { has_amx, has_vnni, has_sve2, l1_cache_size, vec_width_bits }`
  struct as "creep." The real `SimdCaps` (`simd_caps.rs:33-108`) is
  structurally the same idea (a `Copy` capability struct, detect-once).
  My deletion + replacement-with-denial moved the doc *further* from the
  code. [grounded]

### 3.2 Invented quantities (confirmed hallucinated by maintainer)

- **Per-arch DCT crossover constants `SPR=64, ICX=32, Zen4=96,
  Apple M=256, Graviton=128`** (§3.3, lines 153-155, merged). No source.
  The maintainer confirmed these are completely fabricated. Also appear
  in `pr-x12-codec-cognitive-substrate-mapping.md` framed as "concrete
  defaults landed in §R-5" (pre-existing, earlier session) — i.e. an
  earlier session invented them and I propagated them into a second doc.
  [maintainer-confirmed; cross-doc location grounded by my own prior
  grep this session, NOT by whole-file read of the mapping doc —
  see §5]
- **`build.rs` OUT_DIR calibration / `CARGO_CFG_TARGET_FEATURE` decision
  matrix** (§3.3, lines 166-168, merged). Invented mechanism. The real
  per-build selection is `cfg(target_feature)` in `simd.rs` + the
  `simd_caps()` runtime singleton; there is no evidence of a
  build-script crossover-calibration step. [grounded against simd.rs;
  build.rs itself not read — see §5]

### 3.3 Inverted architecture (confirmed "astronomically wrong")

- **`bgz17::scalar_sparse::tropical_spmv` as "canonical R-7 kernel
  home," blasgraph as "future abstraction."** This inversion lives in
  `pr-x12-canon-resolutions-delta.md` §R-7 (pre-existing/merged) and was
  propagated by my dangling `c3c49042` (now unreachable). Reality per
  maintainer: blasgraph is canonical bit-exact; `bgz17::…tropical_spmv`,
  if it exists, is at most a lossy / EWA-sandwich 10000×10000 adapter
  for a special case and cannot replace blasgraph. [maintainer-confirmed;
  symbol existence NOT verified against lance-graph — see §4]

### 3.4 Speculative content that should not exist

- **GPU offload anchor** (§5, line 247, merged). Maintainer: there will
  be no GPU offload in the near future. The whole bullet is vapor; my
  rename of `dispatch_target` → `backend_target` did not unhallucinate
  it. [maintainer-confirmed]
- **3-layer ASCII diagram + named consumer enumeration** (§3, lines
  86-108, merged) and the **per-arch binary / "WoA fleet ships per-arch
  binaries"** claims (§3.2). Authored by me; the consumer list and fleet
  model are unverified narrative. [unverified]

### 3.5 `#[target_feature]` framing

- Maintainer rule: `#[target_feature]` is **strictly prohibited** except
  for `LazyLock`-detect-once, and even that is being superseded by the
  `simd_caps.rs` lookup table. My §3.1 cfg-based polyfill description is
  partially right (type re-export IS `cfg(target_feature)` in
  `simd.rs`), but the doc never points consumers at `simd_caps()` as the
  dispatch-decision mechanism, which is the actually-correct pattern.
  [grounded]

---

## 4. TO-DO (recommendations only — do NOT execute without sign-off)

Priority order. None of these should be done from fragments.

1. **Verify the blasgraph / bgz17 symbols against `adaworldapi/lance-graph`.**
   Does `blasgraph` exist as a module? Does
   `bgz17::scalar_sparse::tropical_spmv` exist, and is it a lossy /
   EWA-sandwich adapter as the maintainer suspects? Until answered, every
   doc citing either is ungrounded. This gates §R-7 cleanup.
2. **Correct the false "no runtime CPU detection" claim in merged woa
   §3.** Replace with an accurate description grounded in `simd_caps.rs`
   (LazyLock<SimdCaps> detect-once singleton + `cfg(target_feature)`
   compile-time type re-export). This is the highest-confidence fix
   because the truth is in-repo and already read whole.
3. **Strike the fabricated crossover constants** (64/32/96/256/128) from
   woa §3.3 and `codec-cognitive-substrate-mapping.md`. No source exists.
4. **Re-tense the R-* series.** Decide, with the maintainer, whether the
   R-N rows should be reframed as future-conditional SoA-transition
   planning (likely) or deleted. This is a judgment call about the whole
   cluster, not a line edit.
5. **Resolve §R-7 inversion** once (1) is answered: blasgraph is
   canonical; bgz17 is a lossy special-case encoding, not a kernel home.
6. **Decide policy on doc accretion.** New plan docs (#199 and after) are
   landing un-audited. Recommend a moratorium on new PR-X12 perspective
   docs until the existing cluster is verified, or an explicit
   "speculative / unverified" banner convention.
7. **Delete the GPU-offload speculation** (woa §5, line 247).

---

## 5. SOURCES

### 5.1 Read end-to-end this session (whole-file) — claims grounded on these

- `src/simd_soa.rs` (373 lines) — PR-X1 MultiLaneColumn, layout-only.
- `src/simd.rs` (728 lines) — polyfill dispatcher: cfg(target_feature)
  type re-export + LazyLock<Tier>.
- `src/simd_caps.rs` (636 lines) — LazyLock<SimdCaps> detect-once, 24
  capability bits, ArmProfile, tests.
- `src/simd_amx.rs` (read whole earlier this session) — AMX inline asm +
  Linux prctl + CPUID/XCR0 detection.
- `.claude/knowledge/pr-x12-woa-multiarch-orchestration.md` (361 lines) —
  the doc I damaged; read whole to catalogue 3.x.
- `CLAUDE.md`, `.claude/rules/data-flow.md` — read whole (system
  context). Confirm: `target-cpu=x86-64-v4`, W1a contract, `CpuCaps`
  listed as a legitimate "reasoning microcopy."

### 5.2 Maintainer-supplied architectural truths (authoritative, not
independently verified by me)

- R-* = future-conditional SoA-transition planning, not present canon.
- "canonical" originated as enthusiasm about `simd_soa.rs` reaching
  implementation-ready in the SoA context, over-promoted.
- blasgraph = canonical bit-exact; bgz17 = lossy, one of several
  encodings (siblings cam_pq, highheelbgz, bgz-hhtl-d); EWA splat =
  top-k-ranking-preserving under lab conditions at 10000×10000.
- `#[target_feature]` prohibited except LazyLock-detect-once; lookup
  tables (`simd_caps.rs`) preferred.
- No GPU offload in the near future.

### 5.3 NOT verified — explicit scope boundary

This audit is **silent** on, and makes **no** claims about, the
following (they need whole-file reads / cross-repo checks I have not
done):

- Existence/behavior of `lance-graph::blasgraph` and
  `bgz17::scalar_sparse::tropical_spmv` in `adaworldapi/lance-graph`.
- The perspective docs I have NOT read end-to-end:
  `pr-x12-canon-resolutions-delta.md`,
  `pr-x12-substrate-canon-resolutions.md`,
  `pr-x12-bgz-jc-substrate-synergies.md`,
  `pr-x12-codec-cognitive-substrate-mapping.md`,
  `pr-x12-anti-neural-lookup-inversion.md`,
  `pr-x12-x265-blasgraph-gemm.md`,
  `pr-x12-x266-3dgs-spacetime-upscaling.md`,
  `pr-x12-gguf-llm-weights-encoding.md`,
  `pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md`. Cross-doc
  location claims in §3.2 about `codec-cognitive-substrate-mapping.md`
  come from a fragment-grep earlier this session and are flagged as such.
- `build.rs` (not read) — my claim that no crossover-calibration step
  exists is inferred from `simd.rs`, not from reading `build.rs`.
- The individual content of R-1 … R-15, M:E-*, Plan G/H — whether any
  corresponds to real code or is internally circular.
- The new #199-era plan docs (PhiSpiral256, 3DGS render-depth, tensor
  container capstone, EWA SYRK BLAS MKL, 4×4 3DGS).
- The wN5kw audit in PR #200 — deliberately unread for drift-check
  independence.

---

## 6. One-line verdict

The substrate the PR-X12 docs describe is mostly a future-conditional
plan written in present-tense canon; the genuinely-shipped pieces
(`simd_soa.rs` = PR-X1, the `simd.rs` polyfill, the `simd_caps.rs`
detect-once singleton, `simd_amx.rs`) are smaller, sounder, and
differently-shaped than the docs claim — and the fastest high-confidence
fix is to correct the merged false claim that the code does "no runtime
CPU detection," because the truth is sitting in `src/simd_caps.rs`.
