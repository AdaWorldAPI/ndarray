# Consolidation Arc P2 Savant Review

Reviewer: Opus P2 codex savant
PR: AdaWorldAPI/ndarray#... (consolidation arc — PR-X10 + PR-X11 + PR-X13 integrated; PR opens after PP-13 / codex-P0 clear)
Branch: `claude/pr-x4-splat-cascade-design` @ `5e266d19`
Date: 2026-05-18
Verdict (advisory, not blocking): **SHIP-WITH-FOLLOWUPS**

Three sprint-worths of code (linalg-core foundation, six pillar probes, the OGIT bridge with embedded TTL bundle + cognitive bridge) integrate cleanly enough to ship once codex-P0 and PP-13 pass. The math contract holds, distance-typing guardrail holds (no umbrella anywhere in the new surface), and the architectural invariants (Spd3 32-B/half-zmm shape, Quat 16-B SSE word, BasinAtom 40-B repr-aligned) are honoured. But there are **two pre-merge nudges** the consolidation cannot ship without — (F1) cargo-fmt is failing on 141 hunks across 19 files, which will fail the fmt CI check unconditionally, and (F2) `#![allow(missing_docs)]` at module-scope on 15 of the new files silently bypasses the `-D warnings` docs gate for ~60% of the new public surface — and **two highest-leverage doc edits** (G1 / G2 — the cross-sprint surface seam where PR-X9's planned `OgitSchema::nearest_basin(value)` call does not match PR-X13's actual `CognitiveBridge::nearest_basin(cell_value, hint_basin_idx)` signature, plus the `OgitSchema` → `OntologySchema` rename drift). Everything else (the `linalg::Spd3` vs `splat3d::Spd3` import inconsistency at A-tier worker output, the `eig_sym::Spd2/Spd4/MatN` shadowing of `linalg::Spd2/Mat2/MatN`, the missing Quat operator overloads, the missing scalar-helper extraction for the SIMD swap) is correctly post-merge follow-up territory.

## A. API ergonomics findings

- **A1 — `linalg::Spd3` / `linalg::Spd2` are discoverable but the routing guide is HALF-correctly cited.** The module docstring at `src/hpc/linalg/mod.rs:32-37` lists the N∈{2,3,4} closed-form vs N≥5 dispatch rule clearly — and `src/hpc/linalg/eig_sym.rs:5-18` repeats the same table in higher detail with the parity-equivalence note. Good. **But** the routing guide promises `eig_sym_3` takes `Spd3` (per PR-X10 design doc `pr-x10-linalg-core-design.md:113`: `pub fn eig_sym_3(a: &Spd3) -> ...`), and the actual signature is `pub fn eig_sym_3(m: &[[f32; 3]; 3]) -> (f32, f32, f32, [[f32; 3]; 3])` (`eig_sym.rs:295`). Same for `eig_sym_n<const N>` taking `&MatN<N>` where `MatN` is `eig_sym`'s own `pub type MatN<const N: usize> = [[f32; N]; N]` (`eig_sym.rs:127`) — NOT `linalg::MatN<N>`, the `#[repr(C, align(64))]` struct from `matrix.rs:29`. The 3 type-aliases that share names but are different types (`eig_sym::Spd2` vs `linalg::Spd2`, `eig_sym::MatN` vs `linalg::MatN`) are an ergonomic gotcha. The `linalg::Spd3LinalgExt` trait at `matrix.rs:444` exists precisely to bridge this but is referenced exactly once (in one test). Recommended pre-merge: one paragraph in `linalg/mod.rs` warning that `eig_sym` carries its own carrier types for the closed-form fast paths and explaining the bridge. Recommended post-merge: collapse to a single shared `Spd2/Spd3/Spd4/MatN` carrier across the `linalg::*` submodule, with `eig_sym_*` taking the shared types directly. This is the highest-leverage post-merge cleanup.

- **A2 — `MatN<const N>` carrier coexists gracefully with `Mat3 = MatN<3>` aliases.** `matrix.rs:155-161` ships `Mat2/Mat3/Mat4` as `pub type X = MatN<N>;`. Doctests use `Mat3::identity()` and the generic `MatN::<5>::identity()` correctly (`matrix.rs:530-535`). No issues with the alias coexistence. Same pattern as the `SoaVec<T, N>` / `SoaBatch<T, N>` discussion in the W3-W6 P2 review (B1) — the alias works without confusion.

- **A3 — `Quat` has NO operator overloads.** No `impl Add/Mul/Neg` anywhere in `src/hpc/linalg/quat.rs`. Users compose rotations via `q1.mul(&q2)` (`quat.rs:395`), invert via `q.inverse()` (`quat.rs:335`), and chain via `q.normalize().mul(&q2)`. The design doc (`pr-x10-linalg-core-design.md:80-85`) DID specify these as methods, not operators — so this is "as designed". But consumers coming from `glam` / `nalgebra` / `cgmath` will type `q1 * q2` first. Recommend a post-merge pass adding `impl Mul<&Quat> for &Quat`, `impl Mul<[f32; 3]> for &Quat` (rotation of a 3-vec), and `impl Neg for Quat` (conjugate). Single-file change, ~30 LOC, no API break (methods stay). Not a blocker; design intent was method-first.

- **A4 — `attention_f32` is missing 2 of the 3 knobs a real consumer needs.** `AttentionConfig` (`attention.rs:43-51`) exposes `num_heads`, `head_dim`, `causal_mask`. **Missing**: RoPE composition (consumers would have to call `RopeCache::apply_qk_f32(q, k, ...)` BEFORE `attention_f32`, separate-step, no closure boundary), and KV-cache hook (an `Option<KvCache<'a>>` or a `prefill_position: Option<usize>` field that the next-token path needs). The openchat / gpt2 / qwen3 inference modules in `hpc::*` (per CLAUDE.md) need both. **Comment**: this is the canonical "ships in cycle N+1" knob. The RoPE separation IS what most reference implementations do (Llama, vLLM, HF transformers) so this is defensible — but a 3-line note in `attention.rs:43` saying "callers compose RoPE via the `RopeCache::apply_qk_f32` pre-pass; KV-cache support queued for PR-X10.1" would close the question for readers. Recommend.

- **A5 — `Spd3LinalgExt` trait at `matrix.rs:444` is undiscoverable in practice.** Only one caller (a test on line 595). The trait exists to give `linalg::Spd3` (which is `splat3d::Spd3` re-export when `splat3d` is enabled) the methods `is_symmetric_pd`, `to_mat_n`, `from_mat_n_symmetric` — but those methods exist as inherent methods on the stand-alone `Spd3` defined when `splat3d` is disabled (`matrix.rs:399-428`). Consumers won't import the trait, will write `spd.to_mat_n()`, and get a method-not-found error under `--features splat3d`. Recommend either (a) move all three methods into a `Spd3Ops` trait blanket-implemented for both paths so callers never need to know which build they're in, or (b) merge the two definitions into one. The current dual-definition split makes the linalg ↔ splat3d feature interaction fragile.

## B. Naming / discoverability

- **B1 — `linalg::Spd3` vs `splat3d::Spd3` import inconsistency in the SAME sprint.** Three call sites:
  - `src/hpc/pillar/koestenberger.rs:34`: `use crate::hpc::linalg::Spd3;` (good)
  - `src/hpc/pillar/ewa_sandwich_2d.rs:44`: `use crate::hpc::linalg::Spd2;` (good)
  - `src/hpc/pillar/ewa_sandwich_3d.rs:41`: `use crate::hpc::splat3d::spd3::{sandwich, Spd3};` (inconsistent — imports from splat3d, not linalg)

  The commit `fb925de5 fix(pillar/koestenberger): import Spd3 via linalg (consistent with pillar deps)` shows the team was actively normalizing on the `linalg::` path. The Pillar-7 file (B2) missed it. Two-line fix:
  ```rust
  use crate::hpc::linalg::Spd3;
  use crate::hpc::splat3d::spd3::sandwich;
  ```
  Recommend pre-merge or same-day follow-up.

- **B2 — `ewa_sandwich_2d` / `ewa_sandwich_3d` naming convention is good but undocumented.** Both modules ship `prove()`, `ewa_sandwich_2d()`, `ewa_sandwich_3d()` named-suffix free functions (consistent with the `_2`/`_3` numeric suffix convention used by `eig_sym_2`/`eig_sym_3`/`eig_sym_4`). The convention works. **But** `pillar::cov_high_d::CovHighD<const N: usize>` (`pillar/cov_high_d.rs:56`) uses `_high_d` as a SUFFIX rather than the `_n` numeric convention. The design doc (`pr-x11-jc-consolidation-design.md`) calls Pillar-9 "Cov16384" implying the intended default was N=16384, but the implementation defaults to N=64 (the file's `cov_high_d.rs:13-16` literally says "N = 64 (PR-X11 v1; N = 16384 / BindSpace alignment is PR-X11.1)"). With N=64 the "high_d" name reads strange — a 64-D covariance is small by most conventions. Recommend renaming to `CovGeneric<N>` (matches `MatN`) or `CovN<N>` in PR-X11.1. Not a blocker; "high_d" carries the literature reference (Düker–Zoubouloglou) so future readers will not be confused if they read the module docstring.

- **B3 — `CognitiveBridge` is consistent with the sibling `MedcareBridge` / `BardiocBridge` future naming convention.** The PR-X13 design doc references `MedcareBridge` as the sibling pattern (line 9). The implementation ships `CognitiveBridge` (cognitive_bridge.rs:149) following the `<Namespace>Bridge` pattern. Convention holds; no concern.

- **B4 — Macro discoverability is fine.** `blocked_grid_struct!` (`blocked_grid/grid_struct_macro.rs:198`) and `soa_struct!` (`soa.rs:318`) are both `#[macro_export]`. The 3 docstring `use ndarray::blocked_grid_struct;` invocations confirm crate-root re-export. No new macros in this consolidation arc. No discoverability gap.

- **B5 — `linalg::Spd3LinalgExt` name carries the `Ext` suffix common in Rust idiom** (e.g., `IteratorExt`, `OnceExt`) signalling "extension trait, import to use". Discoverable via rustdoc and consistent. The single-call-site usage flagged in A5 is an adoption gap, not a naming gap.

## C. Doc-prose quality

- **C1 — Module headers lead with the math reference in 4 of 6 modules, but skip it in the 2 inference modules.** `linalg/mod.rs:24-30` cites Smith-1961 in the SPD math reference subsection — good. `eig_sym.rs` cites Smith-1961, Ferrari, Jacobi, Wilkinson. `svd.rs:19-24` cites Golub-Reinsch 1970 + Demmel-Veselić 1992. `matfn.rs:14-20` cites Higham 2005 + Al-Mohy-Higham 2009. `rope.rs:25-28` cites Su et al. 2021. `attention.rs:22-25` cites Dao 2022. `polar.rs` — **no math reference in the header**. `conv.rs` — **no math reference in the header** (im2col is a Caffe/cuDNN technique, attributable). Pillar files do better — `pflug.rs:10` cites Pflug-Pichler 2012, `cov_high_d.rs:7` cites Düker-Zoubouloglou 2023, `signature.rs` cites Hambly-Lyons. Recommend adding ~2 lines per gap: (a) `polar.rs` should cite Higham 1986 (the canonical Newton iteration), (b) `conv.rs` should cite Jia et al. 2014 "Caffe" or Chetlur et al. 2014 "cuDNN" for im2col.

- **C2 — Doctests are realistic, NOT smoke-test toys.** Massive improvement over the W3-W6 P2 finding C1. `attention.rs:83-99` shows a 4-token / 1-head / 4-dim uniform-vector example where the output equals V (showing the softmax-of-constants identity). `flash_attention_f32` doctest (line 208-220) shows the same pattern at 8 tokens / block=4 (the parity test between naive and tiled). `RopeCache` doctest (rope.rs:115-124) shows a real batch-1 / seq-4 / heads-2 / head_dim=4 forward pass with explicit positions. `mat_exp` and `polar` doctests show actual 3×3 examples with realistic numerics. **No regression** vs the W3-W6 toy-doctest concern.

- **C3 — Disambiguation prose "this is scalar; SIMD swap comes in PR-X5" is COMPLETELY ABSENT from every `linalg::*` and `pillar::*` free-function docstring.** I grep'd for `scalar today|scalar; SIMD swap|SIMD swap|future SIMD` across `linalg/*.rs` and `pillar/*.rs` — **zero matches**. The W3-W6 P2 review's C2 finding (the SIMD-disambiguation prose deficit) is fully reproduced here. The `linalg/mod.rs:41-42` "Out of scope (hard boundary)" section says "No SIMD primitives — use `crate::simd::{F32x16, …}` directly" — that's about WHERE NOT to put SIMD code, not about WHETHER the current path IS SIMD. A consumer reading `attention_f32`'s docstring (line 57-99) gets no signal that this is scalar today. **Recommend one canonical sentence in each compute-heavy free-fn docstring**: `"This implementation is scalar today; the public signature is forward-compatible with a future SIMD body (PR-X5)."` This is the highest-leverage doc edit in C-tier. ~12 free fns affected: `attention_f32`, `flash_attention_f32`, `sinkhorn_knopp_f32`, `hungarian_f32`, `wasserstein_1_f32`, `polar`, `mat_exp`, `mat_log`, `svd`, `eig_sym_3`, `eig_sym_jacobi`, `RopeCache::apply_qk_f32`.

- **C4 — `cognitive-distance-typing.md` is NOT cited in pillar/linalg module headers.** I grep'd — only `soa.rs`, `bulk.rs`, `blocked_grid/grid_struct_macro.rs`, `blocked_grid/aliases.rs`, `blocked_grid/super_block.rs`, `blocked_grid/compute.rs`, `blocked_grid/iter.rs` cite the doc. The new modules (`linalg/*.rs`, `pillar/*.rs`, `ogit_bridge/*.rs`) do not. The `linalg/mod.rs:43` notes "No distance metrics — those live in `crate::hpc::distance`" which is the right intent, but it doesn't cite the typing doc that explains WHY. The Wasserstein module ships `wasserstein_1_f32` (mathematically standard distance metric, so no umbrella concern — but a fresh reader can't tell). The `pillar/pflug.rs` says "nested-distance probe" 9 times. Recommend adding one paragraph to `linalg/mod.rs`, `pillar/mod.rs`, and `ogit_bridge/mod.rs` along the lines of: "Distance terminology in this module refers to specific named metrics (Wasserstein-1, nested Pflug-Pichler, XOR-popcount basin similarity). Per `.claude/knowledge/cognitive-distance-typing.md`, these are typed-specific functions, not instances of a `Box<dyn Distance>` / `enum DistanceMetric` umbrella — that pattern is forbidden crate-wide."

- **C5 — All 26 TTL files have `dcterms:source` provenance.** I verified — `find src/hpc/ogit_bridge/assets/ -name "*.ttl" -exec grep -L dcterms:source {} \;` returns empty. Every TTL cites `dcterms:source "AdaWorldAPI/ndarray/.claude/knowledge/pr-x9-design.md:layer-1-substrate"` (the seed reference) or `pr-x13-ogit-bridge-design.md` (the bridge reference). Provenance gate satisfied.

## D. Distance-typing guardrail

- **D1 — Zero `Box<dyn Distance>` / `enum DistanceMetric` umbrella across all new files.** Confirmed via grep — no matches in `linalg/`, `pillar/`, `ogit_bridge/`. The math IS named-metric specific (`wasserstein_1_f32`, `hungarian_f32`, `nested_distance_single_level`, `nearest_basin` returns minimum XOR distance) and stays properly bounded.

- **D2 — `CognitiveBridge::nearest_basin(cell_value, hint_basin_idx)` (`cognitive_bridge.rs:335-354`) is properly bounded as basin-XOR-popcount, NOT a distance-metric umbrella.** The docstring explicitly says "minimum XOR distance" and the implementation is `cell_value ^ atoms[i].edge` followed by `<` comparison on the u64. No `Box<dyn Metric>` boundary, no `enum DistanceKind` parameter. The `hint_basin_idx` parameter signals locality-based pruning (currently used only as the initial candidate, with the rest of the codebook scanned linearly — see also G1 below for the cross-sprint seam). Type stays unboxed and the distance semantics are explicit XOR-popcount, not "abstract distance metric". Good.

- **D3 — Pillar probes use "distance" in module/file headers and free-fn names but always with a specific metric attached.** `pflug.rs` says "nested Wasserstein distance" — that's a specific named OT metric. `wasserstein.rs` says "Wasserstein-1 distance" — same. `rope.rs:182` says "L∞ distance" inside a HELPER (the parity-test float-comparison utility) — typed. **No incidental "distance" terminology that would mislead a future contributor toward umbrella adoption.** Coverage holds.

- **D4 — `pillar::cov_high_d::CovHighD` does NOT expose a `distance(&self, &other)` method.** It exposes `frobenius_sq`, `eig`, packed lower-triangular storage, online update. Correct — the Frobenius norm of the difference is a metric but is computed point-wise by the prove() harness, not via a method on the struct itself. No umbrella creep.

## E. Future-proofing for SIMD swap (PR-X5)

- **E1 — `linalg::eig_sym_3` body is NOT extracted into a private `eig_sym_3_scalar` helper.** `eig_sym.rs:295-450` (closed-form Smith-1961) is one ~155-line public function. When PR-X5 lands, the desired refactor is:
  ```rust
  pub fn eig_sym_3(m: &[[f32; 3]; 3]) -> (f32, f32, f32, [[f32; 3]; 3]) {
      // future: SIMD_DISPATCH.eig_sym_3_avx512(m)
      eig_sym_3_scalar(m)
  }
  fn eig_sym_3_scalar(m: &[[f32; 3]; 3]) -> ... { /* current body */ }
  ```
  This is a 5-line edit today, 20-line edit if deferred until PR-X5 (because the public signature must stay stable while the body moves). Recommend doing it now or as a same-day follow-up. Same argument applies to `eig_sym_2`, `eig_sym_4`, `eig_sym_jacobi`, `eig_sym_qr`, `eig_sym_n`. Six 5-line edits.

- **E2 — `linalg::attention_f32` and `flash_attention_f32` do NOT factor out their scalar inner loops.** `attention.rs:100-175` (naive O(N²)) and `attention.rs:221-...` (flash tiled). The inner Q·Kᵀ dot product (`attention.rs:132-136`, `attention.rs:279-283`) is the exact body PR-X5 will want to SIMD-swap. When PR-X5 lands the refactor must move this body AND maintain the public signature. Same recommendation as E1 — extract the inner dot-product loop into a private `fn attention_inner_dot_scalar` helper now. ~15-line edit today.

- **E3 — `pillar::temporal_sandwich::sandwich_update_3x3` (`temporal_sandwich.rs:165-183`) is a `[[f32; 3]; 3]` array kernel that does NOT route through `splat3d::spd3::sandwich`.** The Pillar-8 probe duplicates the 18-FLOP Mat3×Mat3 followed by Mat3×Mat3ᵀ. The reason given in the file is "the input is non-SPD (it's M_t = sqrt(σ_step)) so the SPD-typed `sandwich` doesn't apply" — but that's a half-truth: `splat3d::spd3::sandwich(M, Σ)` expects `M` to be SPD only because the SIGNATURE says `&Spd3`, the math itself works for any symmetric M. **Pre-PR-X5 fix**: lift the `sandwich_update_3x3` body into a free `pub fn sandwich_mat3_scalar(m: &[[f32; 3]; 3], n: &[[f32; 3]; 3]) -> [[f32; 3]; 3]` shared between `splat3d` and `pillar`, so PR-X5 routes one SIMD kernel. **Today**: it's an ~18 op kernel duplicated. Not a perf concern at Pillar-8's 30k checks/run — but it's the kind of duplication PR-X5 would otherwise re-introduce. Recommend follow-up.

- **E4 — `RopeCache::apply_qk_f32` has its inner rotation loop in a private `fn rotate_pairs` (`rope.rs:162-172`).** Good — that's already the E1 pattern applied. PR-X5 can SIMD-swap `rotate_pairs` without touching `apply_qk_f32`'s public signature. One file out of 12 got this right; the rest should follow.

## F. CI signal

- **F1 — `cargo fmt --check` is FAILING on 141 hunks across 19 files** at HEAD. Files affected: `linalg/{activations_ext,attention,batched,conv,hilbert,loss,norm,rope,sh,wasserstein}.rs`, `ogit_bridge/{cognitive_bridge,mod}.rs`, `pillar/{cov_high_d,ewa_sandwich_2d,ewa_sandwich_3d,koestenberger,pflug,signature,temporal_sandwich}.rs`. The CI fmt-check job will fail this PR. **This is a blocker for the fmt CI gate** (though out of P2-scope — P0 territory). Recommended pre-merge: run `cargo fmt` on the worktree, commit as `style(linalg,pillar,ogit_bridge): apply cargo fmt`. ~2 minutes of work. Note: I am flagging this in F1 even though fmt-check normally falls under codex-P0's gate because (a) it's deterministic / cheap to fix, (b) the 141-diff count suggests the workers were committing without running fmt locally, which is a workflow issue worth surfacing.

- **F2 — `#![allow(missing_docs)]` is used as a module-level attribute in 15 of the new files** to bypass the `-D warnings` docs gate. Files: `linalg/{sh,polar,wasserstein,rope,attention,hilbert,matfn,conv,eig_sym,svd}.rs`, `pillar/{pflug,cov_high_d}.rs`, `ogit_bridge/{mod,schema,cognitive_bridge}.rs`. CLAUDE.md's hard rule is "All public APIs need `///` doc comments with examples" and "`cargo clippy -- -D warnings` must pass". The `#![allow]` shortcut LETS clippy pass while leaving public items without docstrings. The fix is to either (a) write the missing docstrings, or (b) downscope the public items to `pub(crate)` where they're not actually consumer-facing. Spot check: in `attention.rs:1` the `#![allow(missing_docs)]` covers the file even though `attention_f32` and `flash_attention_f32` ARE documented — so the suppress is broader than necessary. Recommend a targeted post-merge sweep to either remove the `#![allow]` (and document or downscope the offenders) or qualify the suppress to `#[allow(missing_docs)]` on the specific items that need it. **This is the silent doc-gate bypass that the codex audit's missing-docs lint would otherwise catch.**

- **F3 — `cargo clippy --no-default-features --features std,linalg,pillar,ogit_bridge,splat3d -- -D warnings` passes locally.** Confirmed by running it during this review (`Finished dev profile [unoptimized + debuginfo] target(s) in 12.13s` with zero warnings). So the clippy CI step will pass on the relevant feature matrix — modulo F2's suppression making the docs lint a no-op for those files.

- **F4 — Cargo features cohere with the consolidation plan.** `linalg = []`, `pillar = ["linalg", "splat3d"]`, `ogit_bridge = []` per `Cargo.toml:228-238`. The `pillar` feature correctly depends on `splat3d` (B3 koestenberger needs `splat3d::spd3::sandwich`). No feature-gate drift relative to the master consolidation doc.

- **F5 — Default-feature build (`cargo build --no-default-features --features std,linalg`)** compiles cleanly. Confirmed in 16.4s.

- **F6 — 14-check CI matrix expected outcomes**: `tests/stable`, `tests/beta`, `tests/1.95.0`, `native-backend/stable`, `cross-test`, `blas-msrv`, `clippy/stable`, `docs/nightly`, `cargo-careful`, `miri` (limited targets), `hpc-stream-parallel/rayon`, `nostd-thumbv6m`. The `fmt-check` job will FAIL per F1. The `docs/nightly` job may pass because `#![allow(missing_docs)]` (F2) silences the warnings the docs job would otherwise flag. Of the 14, expect: 13 green, 1 red (fmt). Re-run after `cargo fmt` lands.

## G. Cross-sprint surface seams

- **G1 — `OgitSchema` (PR-X9 doc) vs `OntologySchema` (PR-X13 actual).** PR-X9's design doc references `OgitSchema` 7 times (`pr-x9-design.md:46, 78, 204, 252, 403, 433, 463`). PR-X13's implementation ships `OntologySchema` (`src/hpc/ogit_bridge/schema.rs:88` and re-export `ogit_bridge/mod.rs:30`). When PR-X9 lands and the worker writes `use ndarray::hpc::ogit_bridge::OgitSchema;` they get a not-found error. **Recommend renaming `OntologySchema` → `OgitSchema` BEFORE merging the consolidation arc** so PR-X9's design imports compile against the actual type name. Alternatively add `pub type OgitSchema = OntologySchema;` to `ogit_bridge/mod.rs:30` (one-line follow-up) to keep both names valid during the transition cycle. **Two-line follow-up** is the pre-merge nudge here.

- **G2 — `CognitiveBridge::nearest_basin(cell_value, hint_basin_idx)` does NOT match PR-X9's planned call site.** PR-X9 doc (line 463) plans `OgitSchema::nearest_basin(value)` — single-arg, on the schema. PR-X13 actual (line 335) is `CognitiveBridge::nearest_basin(cell_value, hint_basin_idx)` — two-arg, on the bridge. Two seams:
  1. Method is on `CognitiveBridge`, not `OntologySchema`. PR-X9 must call `bridge.nearest_basin(...)`, not `schema.nearest_basin(...)`. Plausibly fine — but the design doc disagrees.
  2. Extra `hint_basin_idx` parameter — PR-X9 may want to thread a `last_basin_idx` cursor (locality-based pruning, the very thing the parameter is reserved for). API is forward-compatible if PR-X9 plumbs the cursor; if PR-X9 expects a single-arg API it must change its call site.

  Recommend the PR description (when the consolidation PR opens) explicitly call out this signature divergence so the PR-X9 sprint owner files an adjustment ticket. **Not pre-merge blocking** since PR-X9 is not yet implemented — but the worker will hit it on day 1.

- **G3 — `linalg::Spd3` for tile-bin metadata (PR-X12).** PR-X12 design (per `pr-x12-codec-x265-design.md`) does not yet enumerate concrete Spd3 usage — I grep'd and found no `Spd3`/`splat3d` references in the design doc. So the PR-X12 surface is type-agnostic for now. The Spd3 type itself is stable (32-byte repr-aligned, re-exportable from `linalg::Spd3`). When PR-X12 lands its tile-bin work the import will be `use ndarray::hpc::linalg::Spd3;` and everything composes. **No action**.

- **G4 — PR-X4 (splat cascade) consumes `linalg::eig_sym_3` AND `splat3d::Spd3`.** With the type-alias trick (`linalg/matrix.rs:280: pub use crate::hpc::splat3d::spd3::Spd3;`) both paths resolve to the same type. The catch: `eig_sym_3` takes `&[[f32; 3]; 3]` (A1's gotcha), so PR-X4 must convert `Spd3 → [[f32; 3]; 3]` via `to_mat_n().data` or similar. The `Spd3LinalgExt::to_mat_n` trait method exists but returns `Mat3`, not `[[f32; 3]; 3]`. PR-X4 will hit the same alias mismatch as A1. **Recommendation merged with A1**: add a `Spd3::to_array_3x3()` method, or accept either type in `eig_sym_3` via a sealed `Eig3Input` trait. Post-merge follow-up.

- **G5 — Pillar-8's `sandwich_update_3x3(sigma: &[[f32; 3]; 3], m: &[[f32; 3]; 3])` (`temporal_sandwich.rs:165`) collides namespace-wise with `splat3d::spd3::sandwich(m: &Spd3, n: &Spd3)`.** Same name, different signature, different module path. A future PR-X12 codec consumer wanting "the sandwich operation" may grep `sandwich` and find both. Recommend renaming the pillar variant to `sandwich_mat3_array` or routing it through `splat3d::spd3::sandwich` (per E3). Follow-up. Not pre-merge blocking.

## Net call

**SHIP-WITH-FOLLOWUPS**, gated on three pre-merge nudges and queueing the rest:

**Pre-merge (10 minutes total)**:
1. **F1** — `cargo fmt` on the 19 affected files (one commit).
2. **B1** — fix the `ewa_sandwich_3d.rs:41` import to use `crate::hpc::linalg::Spd3` (one line).
3. **G1** — add `pub type OgitSchema = OntologySchema;` to `ogit_bridge/mod.rs:30` to keep PR-X9's planned name valid (one line).

**Same-day follow-ups (~1 hour, file a single PR-X10.1 / PR-X11.1 / PR-X13.1 sweep)**:
- **C3** — SIMD-disambiguation prose on the 12 compute-heavy free fns
- **C4** — `cognitive-distance-typing.md` paragraph in `linalg/mod.rs`, `pillar/mod.rs`, `ogit_bridge/mod.rs`
- **F2** — replace the 15 `#![allow(missing_docs)]` blanket suppresses with targeted `#[allow(missing_docs)]` per-item OR write the missing docstrings
- **E1/E2/E3** — extract scalar-helper bodies from `eig_sym_3`, `attention_f32`, `flash_attention_f32`, `sandwich_update_3x3`

**Post-merge cycle (queue PR-X10.2 / cycle N+1)**:
- **A1/A5/G4** — collapse `eig_sym::Spd2/Spd4/MatN` shadow types into shared `linalg::Spd*/MatN`, OR write the bridge methods/traits explicitly
- **A3** — `impl Mul/Neg` for `Quat`
- **A4** — KV-cache hook + RoPE composition in `AttentionConfig`
- **B2** — rename `CovHighD` to `CovN` (matching `MatN` convention) when the N=16384 stress test lands in PR-X11.1
- **C1** — math reference citations on `polar.rs`, `conv.rs`
- **G2** — coordinate `nearest_basin` signature with PR-X9 worker on day 1
- **G5** — rename `sandwich_update_3x3` to `sandwich_mat3_array` or route through `splat3d::sandwich`

The math contract holds, distance-typing holds, architectural invariants hold. Three sprints of code merge cleanly into one consolidated PR if codex-P0 / PP-13 don't surface a P0 blocker. None of my P2-level findings warrant pausing the merge — but F1 is a deterministic CI fail and F2 is silently bypassing the docs gate, so both want resolving before the PR opens for human review.

## Sentinel: p2-savant-completed
