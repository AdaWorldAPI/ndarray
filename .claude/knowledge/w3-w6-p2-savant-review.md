# W3-W6 P2 Savant Review of PR #156

Reviewer: P2 codex savant
PR: https://github.com/AdaWorldAPI/ndarray/pull/156
Branch: `claude/w3-w6-soa-aos-helpers` @ `7ba653c`
Date: 2026-05-18
Verdict (advisory, not blocking): **SHIP-WITH-FOLLOWUPS**

The strict P0 codex audit already cleared the PR (0 P0, 1 P1 patched, 3 P2 deferred). This pass looks at API ergonomics, naming, doc-prose, distance-typing visibility on the public PR, future-proofing for the SIMD swap, and CI signal. Everything below is non-blocking; one item (C3) is the highest-leverage pre-merge nudge.

## A. API ergonomics findings

- **A1 — `soa_struct!` cannot replace `GaussianBatch` in `splat3d/gaussian.rs:87-274` today.** The hand-rolled batch carries (a) a separate `len: usize` AND `capacity: usize` (active-vs-padded distinction, `gaussian.rs:88-91`), (b) `with_capacity` that *eagerly fills* each `Vec<f32>` to `capacity` rounded to `PREFERRED_F32_LANES` and zeros the tail (`gaussian.rs:116-134`), (c) heterogeneous arities — eleven `Vec<f32>` channels alongside one wide `Vec<f32>` SH buffer of `SH_COEFFS_PER_GAUSSIAN * capacity` floats (`gaussian.rs:110`). The macro generates `pub fn push(self, x, y, z, ...)` with one row per call and treats `len()` as the field-length invariant. There is **no padded-capacity story**, no eager-zero, no per-field stride-multiplier (the SH lane). Macro consumers will get the simple AoS-row-per-push pattern; SIMD-tail-safe batches with stride-N fields must stay hand-rolled until a future revision adds a `with_capacity_padded(n, lanes)` knob and an arity-attribute (`#[soa(stride = SH_COEFFS_PER_GAUSSIAN)]` on the field). Not blocking — but the PR description should set expectation that this macro is the "simple SoA struct" path and `GaussianBatch` stays bespoke. Today the docstring at `soa.rs:251-299` invites readers to use the macro for `GaussianBatch`-like cases via the literal `pub struct GaussianBatch` example, which oversells the fit.

- **A2 — `aos_to_soa` lacks `#[inline]` (`soa.rs:392-401`).** For a 1M-Gaussian batch with a 12-channel extract closure, the loop body is non-trivial and rustc's default cross-crate inlining heuristic for a generic function with three generic parameters will land inline most of the time, but not guaranteed. A bare `#[inline]` on a non-trivial body is usually wrong, but a `#[inline]` here is justified because (a) the body IS the loop the closure feeds, (b) the closure can only be inlined if the outer function is, and (c) callers will overwhelmingly compile this in a hot path. Same argument for `soa_to_aos` (`soa.rs:426-438`) and both `bulk_apply`/`bulk_scan` (`bulk.rs:68-79`, `100-111`). Recommend adding `#[inline]` on all four free functions in a follow-up.

- **A3 — `bulk_apply` × `aos_to_soa` integration test is still `#[cfg(any())]`-gated at `bulk.rs:297-326` despite both modules landing in the same PR.** The codex audit's D4 already flagged this. Ungating it now adds compile-time validation that the documented composition pattern (which is itself a `///` doctest in the `bulk.rs:25-37` ignored block) actually compiles. Two-line change: drop `#[cfg(any())]`, the `use crate::hpc::soa::aos_to_soa` import is already in place. The `ignore` on the doctest at `bulk.rs:25-37` should also be flipped to a runnable example once the gate is dropped.

- **A4 — `aos_to_soa<T, N, F: Fn(&T) -> [f32; N]>` is hardwired to `f32` output (`soa.rs:392-401`).** Downstream consumers with `i8`/`u8`/`u16`/`bf16` SoA fields (palette indices, quantized embeddings, BF16 mantissa bytes) cannot use the public helper without writing their own. This is consistent with the file header's "layout-only" framing, BUT nothing in the module header or the `aos_to_soa` docstring says "the output type is fixed to `f32` for now; non-f32 fields require a hand-rolled extract loop." A 2-line note added to the module header would prevent a downstream session from filing a "why doesn't this take `i8`?" issue. Recommend adding before merge or in the immediate follow-up.

## B. Naming / discoverability

- **B1 — `SoaVec` vs sibling `GaussianBatch`.** The in-tree precedent uses the `Batch` suffix (`GaussianBatch`); the new generic uses `Vec` as the suffix to mirror `Vec<T>`. Both names are defensible, but a consumer searching for "the SoA primitive" will not naturally type `SoaVec` — they'll search for `SoaBatch`. Adding a one-line `pub type SoaBatch<T, const N: usize> = SoaVec<T, N>;` alias at the bottom of `soa.rs` would make both names discoverable without ambiguity. Optional follow-up; not a real blocker since rustdoc search will surface `SoaVec` under either query.

- **B2 — `soa_struct!` macro name.** Discoverable via the `ndarray::soa_struct!` crate-root export (`#[macro_export]` at `soa.rs:300`). The names that would compete (`struct_of_arrays!`, `soa!`) are not in use anywhere else in the crate per `grep`. No ambiguity issue.

- **B3 — `bulk_scan` naming.** The codex P0 audit (`w3-w6-codex-audit.md:35`) deferred this with the note "rename to `bulk_for_each`/`bulk_inspect` if downstream consumers find the name misleading." Looking at the PR diff, no downstream call site has materialized to validate the name. The Rust idiom (`Iterator::scan`) means "fold with state" — the current `bulk_scan` is exactly NOT that. Recommend the follow-up rename to `bulk_for_each` or `bulk_view` before any external consumer adopts the name; if it lands as `bulk_scan` and ships in a release, the rename becomes a breaking change.

- **B4 — `field` / `field_n` / `field_mut` / `field_n_mut`.** Four method names for "borrow field i". The runtime/compile-time split is justified (`field_n::<I>` elides the bounds check via the `const { assert!(I < N) }` block at `soa.rs:154`). Collapsing the four into a `Field` trait with const-vs-runtime auto-selection would be cleaner if Rust had specialization, but it doesn't on stable. The current four-method API is the right pragmatic choice. Worth one sentence in the docstring explicitly recommending `field_n::<2>()` over `field(2)` for hot paths — currently mentioned only at `soa.rs:136-137`, easy to miss.

## C. Doc-prose quality

- **C1 — Doctests are minimal smoke tests, not SIMD-staging compose patterns.** Every `soa.rs` doctest pushes 2 rows and asserts on a 2-element field. The `bulk.rs` doctests are 10-element vectors. There is no doctest showing the real intended usage — a per-tile SoA-stage-then-SIMD-loop over `SoaVec::chunks(16)` with `crate::simd::F32x16`. Even the `bulk.rs:25-37` composition example is `ignore`'d. A consumer reading rustdoc gets the impression these are toy primitives. Recommend adding ONE realistic doctest per module (a 32-row push, a chunk(8) walk, a closure that touches each field — does not need to call F32x16, just demonstrate the layout intent). Follow-up, not blocker.

- **C2 — Doc-prose mentions "SIMD" 11 times across the two files without crisply qualifying that the module is scalar-only.** The module headers DO say "scalar only" (`soa.rs:22`, `bulk.rs:10`), but interior fn docstrings then say "SIMD gather (VPGATHERDD…)" (`soa.rs:359-360`), "SIMD-style loops" (`soa.rs:272`), "SIMD-friendly storage" (`soa.rs:17`). A consumer skimming `aos_to_soa`'s docstring in isolation may think the call is SIMD-accelerated today. Recommend one canonical disambiguating sentence in each free-function docstring: "This call is scalar today; the public signature is forward-compatible with a future SIMD body." Currently only `aos_to_soa` says this in a roundabout way (`soa.rs:358-361`); `soa_to_aos`, `bulk_apply`, `bulk_scan`, and `SoaVec::chunks` do not.

- **C3 — The module header does NOT explain the layering rule's WHY** (the `crate::simd_ops` charter and why these helpers do NOT live there). The `soa.rs:20-29` header says "scalar only, no `#[target_feature]`, forward-compatible with a future SIMD swap" — but a reader who lands here cold has no idea why a scalar SoA helper isn't in `crate::simd_ops`. The savant's P0-1 finding in `w3-w6-plan-review.md:16-27` is the canonical answer ("`simd_ops.rs` charter is SIMD-only; free-function `aos_to_soa` violates the W1a consumer contract litmus"). One paragraph in the module header pointing at `vertical-simd-consumer-contract.md` and explaining "these helpers live here NOT in `simd_ops` because…" would save a future session 20 minutes of re-derivation. **This is the highest-leverage single doc edit.**

## D. Distance-typing guardrail

- **D1 — The PR description on GitHub DOES cite the typing doc** (PR body §"Distance typing guardrail" enumerates the four typed metrics and warns against the umbrella). The module headers also cite `cognitive-distance-typing.md` (`soa.rs:33-39`, `bulk.rs:39-44`). Strong coverage. A future W7 PR description should echo the same guardrail; no action needed in this PR.

- **D2 — Closure parameter names `extract` and `build` are unambiguous.** `extract: F where F: Fn(&T) -> [f32; N]` does not read as a metric-computing closure — the return type `[f32; N]` is a layout-row, not a scalar distance. Similarly `build: F where F: Fn([f32; N]) -> T` reads as a constructor. No rename needed. The closure name `f` in `bulk_apply` / `bulk_scan` is generic-callback-conventional and fine.

## E. Future-proofing for SIMD swap

- **E1 — The scalar loop body in `aos_to_soa` (`soa.rs:396-400`) is inline in the public fn, not factored into a private `aos_to_soa_scalar` helper.** When the SIMD swap lands, the refactor is:
  ```rust
  pub fn aos_to_soa<...>(...) -> SoaVec<f32, N> {
      // future: SIMD_DISPATCH.aos_to_soa_f32_n3(...)
      aos_to_soa_scalar(aos, extract)
  }
  fn aos_to_soa_scalar<...>(...) -> SoaVec<f32, N> { /* the current body */ }
  ```
  Doing the extraction now (in this PR or as a same-day follow-up) is a 5-line edit. Deferring it forces the SIMD-wave PR to also move the body, which mixes two concerns. Same applies to `soa_to_aos`. Recommend.

- **E2 — Future `aos_to_soa_strided(packed: &[T], field_offsets: [usize; N])` alt-entry compatibility.** The plan-review savant's F2 (`w3-w6-plan-review.md:74`) commits to two stable entries side-by-side: today's closure-based `aos_to_soa` and a future stride-based `aos_to_soa_strided`. The current name `aos_to_soa` is generic enough to accept that sibling — no naming clash. Naming for the strided variant should be decided when it lands; `_strided` is fine.

## F. CI signal

- **F1 — At time of review (~5 minutes after PR open), 4 of 14 checks are still `in_progress`** (`tests/beta`, `tests/stable`, `tests/1.95.0`, `native-backend/stable`, `hpc-stream-parallel/rayon`). All 10 completed checks are green (`success` or correctly `skipped`). The `unstable` mergeable state in the PR JSON reflects this incomplete-checks situation, not a real failure. Re-poll before merge.

- **F2 — CI matrix matches W2 PR #154's matrix exactly.** Same 14 check names, same skip/run profile. No new modules-not-being-exercised gap. Note: `tests/stable` and `tests/beta` are the targets that actually run `cargo test --lib`, so the `hpc::soa` (29) + `hpc::bulk` (16) unit-test counts will be picked up there. The doctests on the new modules (10 + 2 passing, 1 ignored each) are exercised by `tests/stable`. No CI gap.

## Net call

**Ship with two same-day follow-ups queued:** (1) the C3 module-header expansion (single highest-leverage edit), (2) the A3 integration-test ungate. Both are 10-minute edits that materially improve the PR's signal-to-consumer ratio. Everything else (A1's `GaussianBatch` expectation-setting, A2/E1's `#[inline]` + scalar-body extraction, B3's `bulk_scan` rename, C1's realistic doctests) is correctly post-merge follow-up territory. The PR is technically correct and should not be paused; consumers will materialize and reveal which P2s actually need addressing.
