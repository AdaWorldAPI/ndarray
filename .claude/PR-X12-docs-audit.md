# PR-X12 perspective-docs audit (2026-05-22)

> Status: audit findings, no edits made to the audited docs. Recommendations
> only.
>
> Scope: the `pr-x12-*.md` perspective-doc cluster under
> `.claude/knowledge/`, as it stands on master after PR #198 merged
> (2026-05-22). The audit re-grounds the cluster's central factual
> claims against the actual `ndarray` and `lance-graph` source trees.
>
> Methodology: whole-file reads (no grep / head / tail / sed /
> Read-with-offset for forming judgments — only for narrow factual
> lookups such as "does file X exist"). Every entry in the findings
> table below is backed by an end-to-end read of either the cited
> doc section or the cited source file, or both.

---

## 0. Background — why this audit exists

The PR-X12 doc cluster grew over several sessions of architectural
synthesis. Earlier sessions correctly identified an
implementation-ready milestone (`src/simd_soa.rs`, the SoA carrier) and
generalised the framing through `.claude/knowledge/pr-x12-*.md`
perspective docs. The generalisation lost contact with the codebase in
several places: invented symbols, inverted canonical/adapter
relationships, fabricated per-arch numbers, misattributed citation
IDs, and a couple of confident "known bugs" in code whose tests
actually prove the opposite. PR #198 ("purge runtime-dispatch creep
from woa-multiarch §3") then deleted the doc text that matched the
codebase, leaving a confident-but-wrong description of `src/simd.rs`
in merged master.

This audit catalogues what's wrong, what's grounded, and which docs
need rewrites versus targeted corrections.

---

## 1. Verified-real artifacts (the anchors)

These exist as claimed and ground the rest of the architecture
discussion. Whole-file reads, not grep.

| Artifact | Path | Whole-file read | What it actually is |
|---|---|---|---|
| `src/simd_soa.rs` | ndarray | 373 lines | Layout-only `MultiLaneColumn`: `Arc<[u8]>` carrier with 4 typed lane-width chunk iters (`iter_u8x64`, `iter_f32x16`, `iter_f64x8`, `iter_u64x8`), goes through `crate::simd::*`, 8 tests. Self-described as PR-X1, not PR-X12. No kernels, no codec, no basin codebook. |
| `src/simd_amx.rs` | ndarray | not whole-read in this pass (per task: "verified real, doc comments inside are sound") | Real AMX inline-asm + Linux prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA). |
| `.cargo/config.toml` | ndarray | 27 lines | Pins `target-cpu=x86-64-v3` (AVX2 baseline). AVX-512 (v4) is opt-in via separate `config-avx512.toml`. Comment lines 21-24 explicitly name "Runtime LazyLock dispatch" as a fifth supported mode. |
| `CLAUDE.md` "Hard Rules" + W1a consumer contract at `.claude/knowledge/vertical-simd-consumer-contract.md` | ndarray | per task statement: "real" | Authoritative. |
| `crates/lance-graph/src/graph/blasgraph/` | lance-graph | mod.rs (104) + semiring.rs (492) whole-read; 18 other files not whole-read | GraphBLAS-style sparse matrix algebra over 16384-bit binary `BitVec` carriers. Exports `Semiring`, `HdrSemiring` (7 variants: XorBundle, BindFirst, HammingMin, SimilarityMax, Resonance, Boolean, XorField), `GrBMatrix`, `GrBVector`, sparse-storage types. Bit-exact binary-Hamming substrate. Matches user's architectural truth #3. |
| `crates/bgz17/src/scalar_sparse.rs` | lance-graph | 158 lines | Defines `ScalarCsr` struct (lightweight scalar f32 CSR for palette distance matrices) with two methods: `spmv` (standard) and `spmv_min_plus` (lines 95-113, "shortest-path / nearest-neighbor in palette space"). No free function named `tropical_spmv`. |
| `crates/sigker/src/{lib.rs, signature.rs, kernel.rs}` | lance-graph | 63 + 295 + 357 lines | Real `signature_truncated` (signature.rs:96), real `signature_kernel` (kernel.rs:37), real `signature_kernel_pde` (kernel.rs:106). Module docstring + 4 passing tests prove PDE form converges to `I_0(2·√⟨u,v⟩)` — the closed-form analytic limit. |
| `crates/jc/src/lib.rs` | lance-graph | 123 lines | 12-pillar proof-in-code: substrate, weyl, jirak, pearl, cartan, precond, koestenberger, dueker_zoubouloglou, ewa_sandwich, ewa_sandwich_3d, pflug (Pillar 10), hambly_lyons (Pillar 11). Activation date 2026-05-07 / PR #348 cited in module docstring. |
| `crates/bgz-tensor/src/lib.rs` | lance-graph | 122 lines | Real exports include `Codebook4096`/`CodebookIndex` (line 108), `AttentionSemiring`/`AttentionTable`/`ComposeTable` (105), `HhtlDEntry`/`HhtlDMeta`/`HhtlDTensor` (112), `Base17`/`Base17Fz` (115). |
| `src/hpc/cam_pq.rs` | ndarray | 847 lines | `CamCodebook` struct (line 67) — 6-subspace × 256-centroid CAM-PQ scheme; `encode`/`decode`/`precompute_distances`/`distance_batch` methods; `train_geometric` (373), `train_semantic` (408), `train_hybrid` (454); exposed `kmeans` (541) + `squared_l2` (473) for downstream consumers. PackedDatabase stroke cascade (HEEL/HEEL+BRANCH/full) for 99% rejection before full ADC. |
| `src/hpc/codec/` | ndarray | listed only | A1/A2/A3-intra codec foundation: `ctu.rs`, `mode.rs`, `predict.rs`, `mod.rs`. No `ndarray-codec` crate yet. |
| `src/hpc/{dn_tree.rs, merkle_tree.rs, blas_level2.rs, bf16_tile_gemm.rs}` | ndarray | blas_level2.rs whole-read (487 lines) | blas_level2 surfaces 8 BLAS-L2 methods (gemv/ger/symv/trmv/trsv/syr/syr2/gbmv/sbmv). **No `batched_ssd_search` symbol.** Other files exist; not whole-read. |

---

## 2. Architectural truths (user-established ground)

Treat these as load-bearing canon for this audit and any followup:

1. The PR-X12 "R-*" series is **future-conditional planning** for an
   SoA transition. Any framing presenting R-N rows as present-tense
   canon is wrong.
2. The "canonical" framing in the perspective docs originated from
   genuine enthusiasm about `src/simd_soa.rs` (gridlake) reaching an
   implementation-ready milestone in the SoA context. That enthusiasm
   got over-promoted to universal canonicity in subsequent docs.
3. **Blasgraph (in lance-graph) is the canonical kernel — bit-exact.**
4. EWA splat is bit-exact in a weaker sense: top-k results preserve
   ranking. At 10000×10000 scale this property holds only under lab
   conditions.
5. **bgz17 is a LOSSY encoding format.** It is one of several encoding
   alternatives. Siblings include highheelbgz, bgz-hhtl-d, and cam_pq.
   Substituting lossy bgz17 for bit-exact Blasgraph is a soundness
   violation, not a re-targeting.
6. cam_pq is a separate encoding alternative with different
   (non-bgz17) semantics. It must not be force-coupled to bgz17.
7. Shared patterns across some (not all) alternatives:
   256-palette-distance-ranking, attention headers.

---

## 3. Findings table

Severity legend: **Wrong** = contradicted by source · **Inverted** =
canonical/adapter relationship reversed · **Fabricated** = symbol or
value invented · **Misattributed** = real concept under wrong citation
ID · **Soft** = simplification corrected elsewhere in same doc ·
**Grounded** = verified real.

| # | Doc & section | Claim | Reality | Severity |
|---|---|---|---|---|
| 1 | `pr-x12-canon-resolutions-delta.md` §3.2 lines 220-222; `pr-x12-substrate-canon-resolutions.md` R-7 lines 550-557 | "Actual kernel home (current): `lance-graph::bgz17::scalar_sparse::tropical_spmv`; `blasgraph` namespace is the eventual abstraction" | Inverts truth #3 (Blasgraph IS canon; bgz17 is lossy adapter). | Inverted |
| 2 | Same R-7 sections | Symbol `bgz17::scalar_sparse::tropical_spmv` exists | `bgz17/src/scalar_sparse.rs` (whole-read): actual function is method `ScalarCsr::spmv_min_plus` at line 98, signature `fn(&self, x: &[f32]) -> Vec<f32>`. Two-argument `tropical_spmv(edge_weights, dag)` form quoted in R-7 line 556 does not exist. | Fabricated |
| 3 | `pr-x12-codec-cognitive-substrate-mapping.md` §13.3 lines 521-525 | Symbol `lance-graph::blasgraph::tropical_gemm` exists | `blasgraph/mod.rs` (whole-read) exports `Semiring`, `HdrSemiring`, `GrBMatrix`, `GrBVector`, sparse-storage types. No `tropical_gemm`. The 7 HDR semirings operate on 16384-bit BitVec/Bool/Float via XOR/AND/Bundle/MinPopcount — none is numerical min-plus over weighted f32 edges. | Fabricated |
| 4 | `pr-x12-bgz-jc-substrate-synergies.md` §2.5 / §7-§8 | bgz17's `tropical_spmv` is a "canonical-path correction" for R-7 | (a) Symbol doesn't exist as named. (b) Substituting lossy bgz17 for bit-exact blasgraph is a soundness violation. | Inverted |
| 5 | `pr-x12-canon-resolutions-delta.md` §3.1; `pr-x12-substrate-canon-resolutions.md` R-6 lines 491-499; `pr-x12-codec-cognitive-substrate-mapping.md` §13.1 line 513 | "API lives at `ndarray::hpc::blas_level2::batched_ssd_search`" | `src/hpc/blas_level2.rs` (487 lines, whole-read): 8 BLAS-L2 methods only (gemv/ger/symv/trmv/trsv/syr/syr2/gbmv/sbmv). No `batched_ssd_search`, no VNNI dot helper. | Fabricated |
| 6 | `pr-x12-canon-resolutions-delta.md` §6 lines 326-334; `pr-x12-substrate-canon-resolutions.md` R-5 lines 432-441 | Per-arch DCT crossover constants SPR=64, SKX/ICX=32, Zen4=96, Apple Silicon=256 | User-flagged as wholly invented. No measurement, no calibration source. | Fabricated |
| 7 | `pr-x12-codec-cognitive-substrate-mapping.md` §5.3 line 186 | "Concrete defaults landed in canon-resolutions-delta §R-5 — SPR=64, ICX=32, Zen4=96, Apple M=256, Graviton=128" | Graviton=128 does not appear in canon-resolutions-delta §R-5 (whole-read) nor anywhere else in the cluster. Self-fabricating cross-reference. | Fabricated |
| 8 | `pr-x12-canon-resolutions-delta.md` §2.3 lines 161-164 | "64×64 CTU: 132,710 CTUs/frame; Per-CTU budget: 125 ns/CTU" at 4K | 8.3M / 4096 = ~2,025 CTUs. The 132,710 figure is **leaves** at 8×8 (per substrate-canon R-11 line 789, which labels it correctly). Delta doc mis-labels leaves as CTUs. | Wrong |
| 9 | `pr-x12-substrate-canon-resolutions.md` R-14 lines 1008-1012; R-15 lines 1050-1055; `pr-x12-canon-resolutions-delta.md` §11 line 432 | `sigker::signature_kernel_pde` "ships a known math bug (PR #350: Goursat-PDE form diverges from the true kernel `I₀(2·√⟨u,v⟩)` at moderate inner products)" | `sigker/src/kernel.rs` (357 lines, whole-read): module docstring lines 22-32 says PDE and truncated forms compute the same kernel. Test `pde_kernel_converges_to_closed_form_for_linear_paths` (240-267) asserts convergence to `I_0` at `rel<1e-3` for N=128. Test `pde_kernel_refinement_reduces_error` (269-291) proves O(1/N) convergence. Test `pde_and_truncated_agree_on_linear_paths_in_the_limit` (316-345) asserts the two forms agree to `rel<5e-3`. The "bug" claim is contradicted by passing tests. | Wrong |
| 10 | `pr-x12-woa-multiarch-orchestration.md` §3 lines 110-128 | "No runtime CPU detection, no `HwCaps`/`CpuCaps` branching, no `if has_avx512 else…` dispatch" | `src/simd.rs` (727 lines, whole-read): `pub use` re-exports at 229-329 *are* compile-time `cfg(target_feature)` arms (consistent with claim), BUT lines 1-4 docstring says "dispatches via LazyLock<Tier>", and lines 49-99 build a real LazyLock<Tier> dispatcher using `is_x86_feature_detected!`. Type dispatch is cfg-selected; the file documents itself as runtime LazyLock and builds the machinery. | Partial / Wrong |
| 11 | `pr-x12-woa-multiarch-orchestration.md` §3 line 110; §3.2 line 140 | "`.cargo/config.toml` (`target-cpu=x86-64-v4`) makes AVX-512 mandatory on x86_64" | `.cargo/config.toml` (27 lines, whole-read) sets `target-cpu=x86-64-v3` (AVX2). v4 is opt-in via separate `config-avx512.toml`. Config file's own comment at 21-24 lists "Runtime LazyLock dispatch" as a supported mode. | Wrong |
| 12 | `pr-x12-woa-multiarch-orchestration.md` §3 lines 101-107 | Backend file enumeration: "simd_avx512.rs / simd_neon.rs / simd_scalar.rs … AMX bytecode, AVX-512 asm, NEON loads, **SVE2 predicates** LIVE" | `ls`: simd_avx512.rs, simd_amx.rs (NOT in doc list), simd_neon.rs, simd_scalar.rs. No SVE2 file. Doc lists nonexistent SVE2 and omits real simd_amx.rs. | Wrong |
| 13 | `pr-x12-woa-multiarch-orchestration.md` §3.3 lines 146-170 | `cfg(target_feature)`-selected `DCT_BATCH_CROSSOVER` const: avx512=64, neon=256, scalar=MAX | No such constant in source. Plus internally inconsistent with R-5: 3-backend cfg cannot distinguish SPR from Zen4 (both `target_feature="avx512f"`). | Fabricated + Internally inconsistent |
| 14 | `pr-x12-woa-multiarch-orchestration.md` §4 lines 184-190 | "Block-level ME ≤ 10 µs per CTU on SPR (R-11 spec); Tropical-GEMM RDO ≤ 50 µs per CTU on SPR; Basis::apply (DCT) ≤ 2 µs per 32×32 block" | Numbers absent from substrate-canon R-11 (lines 778-822, whole-read), which gives only ~960 ns scalar / ~210 ns SIMD-batched per-leaf. 10 µs/CTU = ~47× over the 210 ns budget. | Fabricated |
| 15 | `pr-x12-woa-multiarch-orchestration.md` §1 line 28 | "R-4 says 'Plan G clears on each of: SPR / Zen 4 / Graviton 3 / Apple M-class' (per-arch CI matrix)" | substrate-canon R-4 (lines 347-395, whole-read) commits *compression-ratio thresholds* for video / 3DGS / KV cache / gradient. No per-arch CI matrix; no Graviton 3. Different commitment under same citation ID. | Misattributed |
| 16 | `pr-x12-woa-multiarch-orchestration.md` §6 lines 258-280 | "R-1, M:E-A: `trait Reducer<T> { fn reduce_pair(&self, lhs: T, rhs: T) -> T }` + `OrderedKahanReducer`" | substrate-canon R-1 (1132-1153, whole-read) commits a different shape: `Basis<T>` + `LinearReduce` with `reduce(&self, src, basis)`. No `reduce_pair`, no `Reducer<T>`, no `OrderedKahanReducer`. | Misattributed |
| 17 | `pr-x12-bgz-jc-substrate-synergies.md` §0 thesis (lines 6, 12) + §2.1 table | "bgz17's 4-layer cascade IS the Skip/Merge/Delta/Escape grammar"; "HHTL 16×16×16 = 4096-leaf lattice IS the basin codebook" | Substitutes lossy bgz17 (ρ=0.937/0.965/0.992) for bit-exact codec mode taxonomy. "16×16×16=4096-leaf" doesn't match any real codec: CAM-PQ is 6×256, `Codebook4096` is flat 4096, `HhtlDEntry` is 4×16×256=16,384. Doc's own §2.2 (138-157) corrects the §0 simplification — but a reader who stops at §0 walks away with the wrong unification. | Soft (Inverted in §0, corrected in §2.2) |
| 18 | `pr-x12-bgz-jc-substrate-synergies.md` §2.7 lines 213-223 | "jc's Pillar 11 IS the formal proof that any bgz-encoded source maps uniquely to its bitstream" | `jc/src/lib.rs:21-22` (whole-read): Pillar 11 proves "signature uniqueness on tree-quotient" certifying "sigker's Index-regime classification" — about path signatures, not bgz-encoded bitstreams. | Misattributed |
| 19 | `pr-x12-substrate-canon-resolutions.md` §9 falsifiability matrix lines 1222-1250 | 24+ rows mapping every M:H / R-N to "test / metric / pass condition" | Most rows cite Plan G's bench-harness binary (which does not exist) as the test. The matrix structurally cites its own forward-conditional plans as the evidence for what those plans were supposed to falsify. | Circular |
| 20 | R-6 cross-doc range collapse | "Speedup: 30-50×" in canon-resolutions-delta §3.1 / substrate-canon R-6 → "~50× over hand-tuned NEON/AVX2 SAD" in `pr-x12-bgz-jc-substrate-synergies.md` §4.2 line 282 + `pr-x12-codec-cognitive-substrate-mapping.md` E-7 line 262 | Estimate range upper bound presented as settled fact in downstream docs. | Range collapsed to upper bound |
| 21 | R-13 primitives | `cam_pq::CamCodebook`, `bgz-tensor::Codebook4096`, `bgz-hhtl-d` shared palette, `dn_tree`, `merkle_tree` exist as cited | All verified real (see §1 anchors). | Grounded |
| 22 | R-14 module references | `lance-graph::jc::pflug` (Pillar 10) and `lance-graph::jc::hambly_lyons` (Pillar 11) exist | `jc/src/lib.rs` (whole-read): both modules registered; activation date 2026-05-07 / PR #348 matches. Module level grounded; specific probe numerics inside `hambly_lyons.rs` not whole-file verified. | Grounded (module level) |
| 23 | R-15 sigker reference | `sigker::signature_truncated` exists | `sigker/src/{lib.rs, signature.rs}` (whole-read): real, signature `fn signature_truncated(path: &[Vec<f64>], depth: usize) -> Signature`. | Grounded |
| 24 | Task-confirmed gridlake anchor | `src/simd_soa.rs` is implementation-ready milestone | `simd_soa.rs` (373 lines, whole-read): layout-only `MultiLaneColumn` (Arc<[u8]> carrier with 4 typed lane-width chunk iters), 8 tests. Self-described as PR-X1. Codec at `src/hpc/codec/` does not currently use it; `predict.rs:57` forward-references it in a doc comment for follow-up batched-encode path. | Grounded (as 373-line layout primitive — narrower than perspective-doc framings) |
| 25 | Task-confirmed canonical kernel | `lance-graph::blasgraph` exists as bit-exact substrate | `crates/lance-graph/src/graph/blasgraph/` (20 files, ~9.5 KLoC). mod.rs + semiring.rs whole-read: GraphBLAS-style binary-Hamming substrate over 16384-bit BitVec with 7 HDR semirings. | Grounded |
| 26 | `pr-x12-codec-cognitive-substrate-mapping.md` §0 line 27; §3.2 lines 99-108 | "every 'efficient transformer', 'gradient compression' and 'neural codec' paper from 2023-2025 has been rediscovering corners of the HEVC/x265 design space"; "Mistral / Llama4 sliding-window attention is **exactly depth-3 leaf processing**" | Sweeping retrospective unification claim presented without engagement with cited literature. Enthusiasm-promoted-to-canon. | Within perspective-doc scope but over-promoted |

---

## 4. Concentration by document

Ranked by hallucination concentration:

1. **`pr-x12-woa-multiarch-orchestration.md`** — 6 distinct
   wrong/fabricated items (#10-#16). The most-edited doc in PR #198.
   Confidently contradicts the very `simd.rs` file it documents.
2. **`pr-x12-bgz-jc-substrate-synergies.md`** — 3 items (#4 plus
   #17-#18). §0 thesis force-couples lossy bgz17 to bit-exact
   blasgraph; §2.7 appropriates Pillar 11's theorem.
3. **`pr-x12-canon-resolutions-delta.md` + `pr-x12-substrate-canon-resolutions.md`** —
   6 items jointly (#1 #2 #5 #6 #8 #9). The R-7 inversion and the
   PDE-bug claim are the load-bearing ones.
4. **`pr-x12-codec-cognitive-substrate-mapping.md`** — 2 items (#3
   #7), plus enthusiasm framing #26.

---

## 5. Recommendations

### Tier 1 — needs full rewrite or quarantine

These docs make load-bearing factual claims the codebase contradicts.
Leaving them in master will continue to mis-train the next session.

- **`pr-x12-woa-multiarch-orchestration.md`** — §3 contradicts
  `src/simd.rs` (partial, see #10) and `.cargo/config.toml` (#11);
  §3 backend enumeration includes nonexistent SVE2 file and omits
  real `simd_amx.rs` (#12); §3.3 cfg-crossover constants don't exist
  (#13); §4 µs latency numbers invented (#14); §6 `Reducer<T>` +
  `OrderedKahanReducer` contradicts R-1 trait shape (#16); §1 R-4
  misattribution (#15); §8 consumer-crate consumption assertions
  unverified. Suggest: replace the body with a shorter doc whose
  claims are limited to (a) what `src/simd.rs` actually does —
  compile-time `cfg(target_feature)` `pub use` arms over a polyfill
  that *also* exposes a runtime `LazyLock<Tier>` for callers that
  need it, with build-time pinning as opt-in per `config-avx512.toml`
  — and (b) the genuine W1a consumer contract from
  `.claude/knowledge/vertical-simd-consumer-contract.md`. Strip
  the cross-domain "consumer ecosystem" framing entirely until
  concrete consumer migrations land.
- **`pr-x12-bgz-jc-substrate-synergies.md`** — the §0 thesis line
  ("bgz17's 4-layer cascade IS the Skip/Merge/Delta/Escape grammar")
  is the user-flagged soundness violation. §2.5 / §2.7 / §4 / §7-§8
  propagate the substitution. Suggest: full rewrite that preserves
  bgz17 / highheelbgz / bgz-hhtl-d / cam_pq as **siblings** (not
  substitutes) and removes all "blasgraph is the eventual
  abstraction" framing; OR archive under
  `.claude/knowledge/archived/` with a header note "superseded —
  see audit 2026-05-22, source: lossy/bit-exact substitution". §2.2
  inside the same doc already provides the correct nuance; lift it
  to §0 and drop the inverted framing.

### Tier 2 — needs targeted corrections

The doc is mostly framing/synthesis within perspective-doc scope
but contains specific factual claims that mislead.

- **`pr-x12-canon-resolutions-delta.md`** — four targeted fixes:
  1. R-7 paragraph lines 220-222 — strip the
     `bgz17::scalar_sparse::tropical_spmv` "actual kernel home"
     framing entirely. Rewrite to acknowledge that blasgraph is the
     canonical home, with bgz17's `ScalarCsr::spmv_min_plus`
     (correctly named) listed as a separate concrete lossy adapter.
  2. R-5 per-arch table lines 326-334 — mark all four crossover
     numbers as `[uncalibrated estimate, no source]` until a real
     codec-bench runs and produces measurements. Don't delete the
     numbers (they may turn out roughly right) but stop presenting
     them as commitments.
  3. R-11 budget table lines 161-164 — fix the "132,710 CTUs/frame"
     → "132,710 leaves/frame at 8×8" mislabel and recompute the
     budget headers per-leaf.
  4. §11 R-14 wording lines 421-433 — remove the "PR #350 corrects
     `signature_kernel_pde`'s known Goursat-PDE math bug" claim.
     The function tests its own convergence to `I_0(2·√⟨u,v⟩)` and
     passes; there is no known bug to correct.
- **`pr-x12-substrate-canon-resolutions.md`** — same R-7 rewrite,
  same R-5 calibration-status note, plus:
  - R-14 lines 1008-1012 + R-15 lines 1050-1055 — remove the PDE
    math-bug claim (same reason as above). The R-15 framing "use
    `signature_truncated` not `signature_kernel_pde` because the
    latter is buggy" is wrong against passing tests.
  - R-9 production-path audit assertion (lines 666-671) about
    `predict.rs` should either be verified end-to-end against the
    current file or marked unverified.
  - §9 falsifiability matrix lines 1222-1250 should be tagged as
    forward-conditional. Every row that names "Plan G binary" as the
    test should carry a footnote so a future reader doesn't mistake
    the matrix for a passed-tests dashboard.
- **`pr-x12-codec-cognitive-substrate-mapping.md`** — two targeted
  fixes:
  1. §5.3 line 186 — strip the Graviton=128 number (it
     cross-references a sister doc that doesn't carry it).
  2. §13.3 lines 521-525 — strip the
     `lance-graph::blasgraph::tropical_gemm` symbol (no such
     symbol exists). Rewrite R-7 framing to match the corrected
     §3.2 of canon-resolutions-delta.

### Tier 3 — sound enough to keep, with header annotations

- The §1 four-axis mapping and §8 epiphanies of
  `pr-x12-codec-cognitive-substrate-mapping.md` are within-scope
  perspective synthesis. Flag with a header note: "perspective doc —
  claims are session conjecture, not codebase reality. Read alongside
  the merged-canon companion docs for current state."
- §3 of `pr-x12-substrate-canon-resolutions.md` (the five "merged
  well" items) is internally consistent synthesis.

### Tier 4 — meta-recommendation

The whole `pr-x12-*` doc cluster's circular citation pattern (delta
cites canon cites synergies cites cognitive-mapping cites
x265-blasgraph-gemm cites delta) is the structural enabler of every
hallucination above. Two structural moves to prevent regrowth:

- **Doc-tree pruning.** Keep at most two canonical sources (the
  merged canon + its resolutions delta). Demote the rest to
  perspective docs that *explicitly cannot be cited as evidence* —
  same way `.claude/knowledge/` already distinguishes verified
  artifacts from session conjecture.
- **New rule analogous to W1a.** Add a
  `.claude/knowledge/` policy file: "Perspective docs may not be
  cited as evidence for claims about current code. Only the source
  files and contracts at the W1a level are evidence."

---

## 6. What was NOT verified in this audit pass

Honest scope boundary so a future session knows where to pick up:

- The remaining 18 files under
  `crates/lance-graph/src/graph/blasgraph/` (~7800 lines) were not
  whole-read. The "no tropical-GEMM in blasgraph" verdict rests on
  whole-reads of `mod.rs` (public surface) and `semiring.rs` (the
  7-variant HDR semiring set). The verdict at the public-API and
  canonical-semiring level is grounded; an absolute-absence claim
  covering every implementation file would need those further reads.
- The specific R-14 probe numerics inside
  `crates/jc/src/hambly_lyons.rs` (the `forward<1e-9, converse>0.05,
  ratio≥1e6` thresholds). The pillar module exists; the numerics
  remain unverified.
- The R-9 production-path audit assertion about `predict.rs`. Not
  verified end-to-end against the current file in this pass.
- PR numbers cited in docs (PR #348 for sigker landing — matches
  `jc/src/lib.rs:25-27`; PR #350 for the claimed PDE bug — but the
  bug itself is contradicted by passing tests, so PR #350's
  existence is moot for the substantive claim).
- Empirical compression ratios in
  `pr-x12-bgz-jc-substrate-synergies.md` §1.4 (the 343:1 on
  Qwen3-TTS-1.7B, 26 palette groups, etc.). These would require
  running the bgz-hhtl-d harness against real model weights, which
  is outside this audit's scope.
- Lance-graph PR #348 / #350 themselves (GitHub state not consulted).

---

## 7. The single load-bearing paragraph

If you read nothing else of this audit:

> The PR-X12 perspective-doc cluster mostly inverts what
> `src/simd_soa.rs`, `lance-graph::blasgraph`, and the lance-graph
> sister crates actually are. The grounded gridlake milestone is a
> 373-line layout-only `Arc<[u8]>` carrier under PR-X1, not the
> implementation hub of a cross-domain codec. Blasgraph is the
> canonical bit-exact kernel; bgz17 is a sibling lossy adapter and
> the docs that present bgz17 as "the actual kernel home" with
> blasgraph as "eventual abstraction" have the relationship
> backwards. R-7's cited symbol `bgz17::scalar_sparse::tropical_spmv`
> does not exist (the real kernel is `ScalarCsr::spmv_min_plus`,
> a method, not a free function). R-6's
> `ndarray::hpc::blas_level2::batched_ssd_search` does not exist
> at all. R-14 / R-15's claim that `sigker::signature_kernel_pde`
> ships a known math bug is contradicted by the function's own
> passing convergence tests against `I_0(2·√⟨u,v⟩)`. The per-arch
> crossover constants (SPR=64, ICX=32, Zen4=96, Apple=256,
> Graviton=128) are uncalibrated estimates promoted to canon —
> Graviton=128 in particular is a self-fabricating cross-reference
> (cited as "landed in canon-resolutions-delta §R-5" by
> codec-cognitive-substrate-mapping §5.3, but absent from that
> section). The cluster's R-* citation IDs sometimes commit
> different shapes in different docs (R-1 has two contradicting
> trait surfaces; R-4 has codec thresholds in the canon docs and
> a per-arch CI matrix in woa-multiarch §1). What is solidly
> grounded: `Codebook4096`, `CamCodebook`, `jc::pflug`,
> `jc::hambly_lyons`, `sigker::signature_truncated`, the
> blasgraph module, the simd_soa carrier itself — all real, all
> verified by whole-file read.

_Last edit: 2026-05-22._

---

## 8. Addendum — corrections applied (2026-07-16)

This audit sat unapplied from 2026-05-22 to 2026-07-16 ("no edits made to
the audited docs"). The following recommendations were executed on branch
`claude/x265-x266-plans-review-h9osnl`:

**Tier 2 targeted fixes (applied):**

- Findings #1/#2/#3/#4 — R-7 inversion + fabricated symbols: corrected in
  `pr-x12-canon-resolutions-delta.md` (§0 item 3, §3.2),
  `pr-x12-substrate-canon-resolutions.md` (R-7, §12 item R-7),
  `pr-x12-codec-cognitive-substrate-mapping.md` (§13.3),
  `pr-x12-x265-blasgraph-gemm.md` (§1 table row 4, §2.4). blasgraph is
  restored as canon; `ScalarCsr::spmv_min_plus` correctly named as the only
  shipped min-plus (lossy sibling, prototype only); `tropical_spmv` /
  `tropical_gemm` free-function citations removed.
- Finding #5 — `batched_ssd_search`: marked **[PLANNED symbol]** at every
  citation site (delta §3.1, substrate-canon R-6, mapping §13.1, GEMM lens
  §1/§5).
- Findings #6/#7 — per-arch DCT crossover constants: marked
  **[UNCALIBRATED ESTIMATES]** in delta §6, substrate-canon R-5, GEMM lens
  §2.2; the self-fabricating "Graviton=128" withdrawn from mapping §5.3.
- Finding #8 — "132,710 CTUs/frame" mislabel: fixed to leaves-at-8×8 in
  delta §2.3 (budget recomputed per-leaf). Review follow-up (same day):
  the count itself was also wrong — exact 8×8 accounting at 3840×2160 is
  **129,600** leaves (padded 64×64: 2,040 CTUs → 130,560); the audit's own
  finding #8 had accepted 132,710 as "correct-as-leaves," which it wasn't.
  Budget recomputed to ~129 ns/leaf; the "~132K CTUs" figures in the three
  RDO paragraphs corrected to ~2,040 CTUs; R-11 matrix row re-unitized to
  ≤210 ns/leaf.
- Finding #9 — false `signature_kernel_pde` Goursat-PDE bug claim:
  withdrawn at all four sites (delta §11/§12, substrate-canon R-14/R-15).
- Finding #19 — §9 falsifiability matrix: tagged FORWARD-CONDITIONAL
  (Plan G binary does not exist; rows are planned tests, not passed ones).

**Tier 1 (quarantined, rewrite still owed):**

- `pr-x12-woa-multiarch-orchestration.md` and
  `pr-x12-bgz-jc-substrate-synergies.md` carry ⛔ QUARANTINED headers
  naming the findings and forbidding citation as evidence. Full rewrites
  per §5 Tier 1 remain open work.

**Status change since audit:** `src/hpc/codec/` now also contains `ans.rs`
(A7 rANS) and `rdo.rs` (A6 λ-RDO) — the audit-era debts D-CODEC-2 (P0) and
D-CODEC-3 (P1) have shipped code. Still no `ndarray-codec` crate (Plan H
unextracted).

**New in the same pass:** the x265/x266 lens line is grounded against the
public JVET trajectory in `pr-x12-h266-h267-standards-landscape.md`
(H.266/VVC facts, ECM, NNVC, H.267 CfP-Jul-2026 → finalize-2028 timeline,
≥40%-over-VVC requirement), with a dated §12 reality-check addendum in
`pr-x12-x266-3dgs-spacetime-upscaling.md`.
