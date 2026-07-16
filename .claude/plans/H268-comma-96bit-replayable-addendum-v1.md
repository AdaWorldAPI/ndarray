# H.268 Addendum Plan v1 — comma closure · 96-bit facet carving · kernel-shape rule · replayable-tile synergies

> Date: 2026-07-16. Status: ACTIVE (this session).
> Base doc: `.claude/knowledge/pr-x12-h268-morton-wgpu-synergies.md` (merged in #242).
> Execution model per operator directive: **plan first, then documentation;
> Sonnet agents for grindwork (draft-from-spec), Opus agent for filigree
> (overclaim/receipt review).** Main thread = spec + gates + landing.

## Content blocks (all facts verified in-session with receipts)

### B1 — Comma closure & constants correction (new §7 of the matrix doc)
- Pythagorean comma / Fujifilm X-Trans framing: anti-moiré = generator must
  not resonate with the sampling lattice; comma = the residue of a fold that
  never closes.
- The workspace surrogate: **coprime-integer comma closure** — helix
  `CurveRuler` stride-4-over-17 (`constants.rs`: MODULUS=17, STRIDE=4,
  gcd=1 → full permutation, tested; banned pattern = Fibonacci mod 17,
  misses {6,7,10,11}); Base17 = same trick vertically.
- **D-QUANTGATE rationale RESTATED (correction of over-attribution):** the
  integer walk is canon for the quantized layer because of (1) libm
  non-portability (receipt: the 2026-07-06 blackboard libm-fma cliff),
  (2) WGSL/GPU floats not IEEE-pinned (C9 verdict), (3) **bijective
  closure** (quantized float-Weyl does not guarantee a permutation) —
  **NOT because float constants "round differently":**
  `std::f64::consts::{GOLDEN_RATIO, EULER_GAMMA}` exist (Rust ≥ 1.94/1.95;
  verified bit patterns φ=0x3FF9E3779B97F4A8, γ=0x3FE2788CFC6FB619; the
  `std::simd::const::*` path does NOT exist — helix `constants.rs:17-23`
  documents that exact correction), and IEEE basic ops in pinned unfused
  order are bit-exact across all five backends (receipt: `gemm_f64_tiled`
  contract + wasm parity CI).
- Division of labor (canon already encodes it): **φ PLACES / walk QUANTIZES
  / γ CORRECTS** (`helix/src/constants.rs` roles) — irrational-constant f64
  math is replayable on the CPU/wasm surface and owns placement; the
  integer walk owns the quantized/GPU/checksum layer.
- x264/AV1 contrast: encoder-side dither unspecified vs AV1 parameterized
  grain vs ours = address-spec'd, replayable, checksummable. [H] until the
  OGAR probes run — J2 fence unchanged.

### B2 — 96-bit facet carving (new §8)
- Verified lane widths: CAM-PQ 48-bit (6×8, `cam_pq.rs:3-12`); helix
  `ResidueEdge` 24-bit unsigned hemisphere / `Signed360` 48-bit signed
  full-sphere (`residue.rs:23-107`); turbovec Lloyd-Max 2/3/4-bit — 6×4-bit
  = 24-bit refinement lane (`lance-graph-turbovec/src/lib.rs`).
- **48 + 24 + 24 = 96 bit = the V3 12-byte content-blind payload.** A legal
  carving: classid(4B) + [CAM-PQ 6B basin | helix 3B residue location |
  turbovec 3B nibbles]. ClassView = the carving/LUT selector (which lane
  indexes which table).
- Budget constraint: `Signed360` (6B) does NOT fit alongside both other
  lanes; it is the out-of-row/alternate-carving variant.
- 6×256² clarification: the six 256×256 distance tables belong to CAM-PQ
  (6×64KB = 384KB ≈ the 388KB SpoDistanceMatrices benchmark); bgz17's
  palette layer is one 256² u16 + k×k u8 compose per palette.

### B3 — Kernel-shape rule (new §9)
- **Match engine to operation shape:** VNNI/AMX for matmul-shaped stages
  (ME/SSD, batched DCT, GEMM scoring); LUT engines (SIMD nibble-gather,
  GPU texture fetch) for lookup-shaped stages. Receipt: turbovec measured
  NativeLut **11.4×** faster than the VPDPBUSD GEMM polyfill (n=20k,
  dim=512, 4-bit) — AMX accelerates exactly the op TurboQuant removed.
- ITU-implementability claim scoped: the W1a pattern (one source, five
  bit-identical backends) covers any ITU codec's **compute kernels**; NOT
  CABAC serial contexts, conformance corners, or ECM-scale tool counts.
- Encode-on-AVX512/VNNI + decode-on-wgpu/WebGL asymmetry endorsed with the
  C5/C9 caveats (wgpu tier is roadmap; integer path only for GPU
  bit-exactness).

### B4 — Replayable-tile synergies: H.268 × cognitive shaders (new §10)
The object: 4×4 Morton tile (2bit x ⊕ 2bit y), phase address-derived via the
bijective walk, magnitudes the only stored content. Consequences:
- H.268: (a) anti-CABAC random access (no serial phase state; strengthens
  the C4 path once A8 lands); (b) seekable grain (vs AV1 seed bookkeeping;
  integer walk survives WGSL per C9); (c) conformance = the period-
  permutation self-test; error localizes to magnitudes; (d) parallelism:
  16 cells = one SIMD lane group / wgpu workgroup tile — NATIVE to the
  H.268 scene codec; HEVC-compat lane keeps 8×8/64×64 (C6 correction
  preserved).
- Cognitive shaders (the bigger half): (e) RNG-free exploration — phase =
  pure function of position; deletes the last shared-mutable-state
  candidate from the thinking loop (composes with E-NOBODY-WAITS-1);
  (f) replayable thinking = auditable cognition — with temporal-stream
  replayability (E-MARKOV-TEMPORAL-STREAM-1), full trajectory incl.
  exploration noise re-runs bit-exactly; counterfactual replay stores zero
  exploration state; (g) anti-confabulation = anti-moiré in concept space
  (coprime probe schedule decorrelated from the palette lattice; known
  period-17 dependence structure is friendlier to I-NOISE-FLOOR-JIRAK than
  unknown PRNG correlations); (h) exact phase-side unbinding (sign
  recomputable per address; cleanup codebook needed for magnitudes only;
  two-algebra rule intact); (i) the 4×4 tile = cache-native working set
  (16×2B ×6 lanes = 192B = 3 cache lines; L4 substrate is flat Morton SoA
  by ruling — the C1 arena-tree corrective applies to the codec CTU, not
  L4).
- The four-role loop: **φ PLACES → walk QUANTIZES → γ CORRECTS → F
  DECIDES**; λ-RDO and free-energy dispatch are the same tile-local
  decision procedure over the same replayable substrate.
- Honesty ledger: ALL conditional on the standing probes — D-MTS-1..3,
  PHASE-1/PERT-RHO/PYR-1 (J2 dither fence), WHP-1..4, L4 tenant doc-locked.
  No kill condition weakened.

## Landing map

| Repo | Branch (restarted from default; both prior PRs merged) | Files |
|---|---|---|
| ndarray | `claude/x265-x266-plans-review-h9osnl` @ origin/master | matrix doc §7-§10 + probe-queue additions; this plan file; blackboard append |
| lance-graph | `claude/x265-x266-plans-review-h9osnl` @ origin/main | EPIPHANIES prepend (E-H268-REPLAYABLE-TILE-1); capstone pointer line; PR_ARC #697 post-merge entry + LATEST_STATE entry (board-hygiene rule) |

Gates: ndarray knowledge-doc suite (117 tests) green; no affirmative stale
symbols introduced (grep gate); graded labels on every claim; Opus filigree
review verdicts applied before commit.

## Agent split (operator directive)

- **Sonnet drafter A (grindwork):** ndarray matrix-doc sections from B1-B4
  spec verbatim-precision; edit-only, no cargo.
- **Sonnet drafter B (grindwork):** lance-graph board/capstone entries from
  spec; edit-only, no cargo.
- **Opus filigree reviewer:** overclaim/receipt/grade audit of both diffs
  against this plan + the PR-X12 audit discipline; findings applied by main
  thread.
- Main thread: gates, commits, PRs.
