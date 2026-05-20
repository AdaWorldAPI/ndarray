# PR-X4 Risk Register + Fallback Decision Tree

> Date: 2026-05-19 (pre-sprint, W4-W5 splat cascade)
> Companion to: `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md`
> Scope: live risks against W4-W5 sprint timeline; mitigations BEFORE
> fire, fallbacks AFTER fire. Decision tree resolves multi-fire
> compositions.

## Risk taxonomy

Each risk carries: **Probability** (LOW/MED/HIGH), **Impact**
(severity to the W4-W5 timeline), **Trigger** (the observation that
declares the risk has fired), **Mitigation** (what we do before), and
**Fallback** (what we do after).

---

### R1 — PR-X10 A12b L4 Hilbert-3D fix slips past W3

- **Probability**: MED. The P0-4 bug is well-characterized
  (`hilbert3d_encode([15,15,15], 4) → 2925, expected 4095`), but the
  fix requires symbolic re-derivation of `NEXT_STATE` from Hamilton
  2006 Table 2, not a one-line patch.
- **Impact**: MED on sprint timeline; HIGH on L4 deliverables. A2
  worker cannot spawn against `linalg::hilbert::hilbert3d_encode` at
  level=4. Done-criterion #2 (CascadeAddr round-trip exhaustive at
  level=4) cannot close.
- **Trigger**: end of W3 with A12b PR not merged to master, OR A12b
  merged but `round_trip_level4_exhaustive` not present / failing.
- **Mitigation**: Track A12b daily during W2-W3; flag the
  Hamilton-2006-Table-2 re-derivation task as critical-path on the
  PR-X10 board. Pre-stage A2's stubbed `Err(NotReadyL4)` return path so
  the worker can ship without blocking.
- **Fallback**: A2 stubs L4 path (`from_position` returns
  `Err(NotReadyL4)` for level=4); A6 ships alpha-only with the L1-L3
  cascade path. Done criterion #2 marked `partial` with the level-4
  exhaustive test deferred to the A12b-completion follow-up PR.

### R2 — GridLake (Q-NEW-1) lands path-β (W3) not path-α (W2.5)

- **Probability**: MED. The savant brief is still open (PR #165). β is
  the conservative cell; α requires the Cell A-α scope to hold.
- **Impact**: MED. PR-X4 starts late by exactly the same margin
  (~half a week). No schedule extension owed by PR-X4 itself, but the
  W4-W5 window compresses.
- **Trigger**: end of W2.5 with PR-X1/PR-X2 not merged OR the savant
  verdict landing on path-β.
- **Mitigation**: Pre-stage A1's refactor against the
  `MultiLaneColumn` contract surface (not its impl) so A1 can begin
  the moment the contract crate exists, independent of GridLake's
  carrier landing.
- **Fallback**: Shift PR-X4 spawn to W3.5 (half-week slip). Run A3/A4
  (SH + INT4×32, both pure-SIMD, no GridLake dep) in parallel during
  the half-week gap. Done criteria all hold; only the calendar moves.

### R3 — G4 `fast_exp_x16` audit fails

- **Probability**: MED. 3% relative error is benign for alpha but
  multiplicative-noise-prone for `(c1+c2)/(c1+c2+k)` near confidence
  floor.
- **Impact**: LOW on W4-W5 timeline (audit is the deliverable, not a
  kernel ship); MED on the W7 closure swap.
- **Trigger**: A5's audit run reports `worst_conf_diff ≥ 1e-3` on the
  10K random-revision sweep.
- **Mitigation**: Run the audit early in W4 (block 1 of A5) before
  committing to the alpha-default-only narrative. Reserve a follow-up
  PR slot against PR-X10 A6.
- **Fallback**: Open `precise_exp_x16` PR against PR-X10 A6
  follow-up, citing the audit's worst-case numbers as motivation.
  Done criterion #3.G4 closes via the OR-branch (audit-or-followup-PR).
  W7 closure swap consumes the new kernel.

### R4 — A3 SH deg-3 fails bit-exact parity

- **Probability**: LOW. Coefficient set is fixed; reference impl
  `splat3d::sh::sh_eval_deg3` is already shipping.
- **Impact**: LOW. Bounded to A3 worker; does not block A1/A2/A4/A5/A6.
- **Trigger**: parity test against `splat3d::sh::sh_eval_deg3`
  diverges by more than 0 ULP on any of 10K random inquiry-directions.
- **Mitigation**: Pull coefficient tables from
  `splat3d::sh::sh_eval_deg3` verbatim at A3 spawn — do not re-derive.
- **Fallback**: Re-derive coefficients symbolically from
  `splat3d::sh::sh_eval_deg3`'s reference impl; treat the reference
  as canonical. If parity still fails, flag a SIMD-lane-ordering bug
  and audit B-Splat / B-Gather-FMA bundle alignment.

### R5 — A4 INT4×32 fails cross-backend parity

- **Probability**: LOW-MED. Three backends; AVX-512 VNNI vs NEON
  `sdot` have different overflow semantics, but the dot output is
  `i32` accumulator (no saturation).
- **Impact**: MED. Blocks G2 done criterion; B-Pack-Dot bundle ships
  partial.
- **Trigger**: parity test on 10K random pairs reports any
  cross-backend disagreement.
- **Mitigation**: Spec scalar as canonical UP FRONT in A4's worker
  brief; align test seeds across backends; use the
  vertical-simd-consumer-contract's parity-test template.
- **Fallback**: Scalar is canonical; debug VNNI and NEON paths
  against scalar. If a backend cannot be reconciled, ship behind a
  cfg-gated feature flag with a routing note in B-Pack-Dot.

### R6 — SG4 fails (NARS B-Compose worse latency than alpha)

- **Probability**: MED. NARS revision has a divide; alpha is a fused
  MAD. Latency parity is asserted by design, not measured.
- **Impact**: MED on W4-W5 (smoke-gate SG4 fails); HIGH on W7 closure
  swap (the swap cannot ship at parity).
- **Trigger**: A6 deployment with `splat4d-nars-compose` feature on
  reports p95 > 20 ms or stutter events > 0 over 10 minutes, while
  the alpha path passes the same envelope.
- **Mitigation**: Latency-budget the `revise_truth_f32x16` lane
  primitive at A5 design time; profile in microbenchmark before
  shipping to A6.
- **Fallback**: Re-stage the closure-swap before W7. File a B-Compose
  follow-up against `vertical-simd-consumer-contract.md` requesting a
  reciprocal-throughput-bounded `revise_truth_f32x16` variant.
  Closure swap deferred until parity is hit.

### R7 — SG1 < 60 fps even on the alpha path

- **Probability**: LOW. Alpha-compositing is a known quantity from
  splat3d v1.
- **Impact**: HIGH — STRUCTURAL. The cascade is dropping behind on its
  steady-state throughput budget. Halt sprint.
- **Trigger**: A6 deployment reports median FPS < 60 on 1080p Big
  Buck Bunny with alpha-only path.
- **Mitigation**: Burn-in A6 on a Railway preview environment during
  W4 (not W5), so a structural cliff surfaces with a week of slack.
- **Fallback**: **Halt sprint.** Convene a structural design review:
  is the (4×4)⁴ tier scheme miscompiling against the bundle latency
  budgets? Are bundles internally saturating an L1d / port? Engage
  savant-architect + sentinel-qa. Do not proceed to W6 until SG1 is
  green.

### R8 — Bundle violation (worker reaches past a bundle into raw intrinsics)

- **Probability**: LOW-MED. Forbidden constraint #2 is explicit, but
  workers under time pressure may inline an `_mm512_*` call.
- **Impact**: LOW-MED. Re-introduces the bespoke-binner pathology v1
  is leaving behind; falsifies the SG2/SG3 contract narrative.
- **Trigger**: code review on any A1-A6 PR observes a direct
  `std::arch::*` intrinsic outside `src/simd_*.rs`.
- **Mitigation**: Bundle list (B-Splat, B-Gather-FMA, B-Pack-Dot,
  B-Cascade-Permute, B-Compose, B-Interleave-Transpose) restated at
  worker spawn; CI grep rule for `std::arch::` outside `src/simd_*.rs`.
- **Fallback**: Reject the PR in review. File the missing primitive
  against `vertical-simd-consumer-contract.md`; primitive lands in
  `src/simd_*.rs` (3 backends + parity test) before the consumer PR
  re-spawns.

### R9 — PR-X11 jc Spd3 not ready by W3

- **Probability**: LOW-MED. PR-X11 is W3 with 6 workers; tight but
  scheduled.
- **Impact**: MED. L6 Pillar-8 temporal sandwich loses its
  `Spd3::sandwich` consumer. Done criterion #4 (L6) partial.
- **Trigger**: end of W3 with PR-X11 not merged OR
  `Spd3::{sandwich, sqrt, from_rows}` missing.
- **Mitigation**: Track PR-X11 daily during W3; mirror the Spd3 API
  shape in a PR-X4 trait alias so the consumer code compiles against
  a stub.
- **Fallback**: L6 temporal sandwich stubbed (returns identity);
  L5 moment-match ships unaffected. Done criterion #4.L6 marked
  `deferred-to-PR-X11-completion`. Smoke gates SG1-SG3 unaffected.

### R10 — PR-X14′ contract bridge fails

- **Probability**: LOW. Contract crate is the smallest deliverable;
  bridge surface is well-typed.
- **Impact**: HIGH. `MultiLaneColumn` consumer path is broken, A1
  refactor cannot wire splat-bytes through the bridge, the whole
  GridLake-carrier story collapses.
- **Trigger**: A1 cannot consume `lance-graph-contract::column::MultiLaneColumn`
  through the gridlake-bridge — compile error, ABI mismatch, or
  runtime panic on basin load.
- **Mitigation**: Pre-flight the bridge surface in a 50-LoC smoke
  binary during W3 (post-merge of PR-X14′), independent of A1's
  spawn.
- **Fallback**: **Emergency design review.** Engage
  savant-architect on the bridge ABI; potentially re-spec
  `MultiLaneColumn` surface. PR-X4 spawn deferred until bridge is
  green. This is the only risk that can trigger a sprint reset.

---

## Fallback decision tree

Multi-fire compositions resolved here. Single-risk fires use the
per-row Fallback above.

- **If R1 fires AND R6 fires** → A2 stubs L4 path AND closure-swap
  re-staged. Ship alpha-only L1-L3 cascade as the W4-W5 deliverable;
  W7 closure swap deferred to W7+ (post-`precise_exp_x16` follow-up,
  post-Spd3 readiness). Sprint closes with done criteria #1, #3.G1-G2,
  partial #2, partial #4, partial #5, full #6, partial #7 (SG1-SG3
  green, SG4 deferred).

- **If R1 fires AND R9 fires** → A2 stubs L4 path AND L6 sandwich
  stubbed. L5 moment-match ships; only L1-L3 cascade exercised by A6.
  Done criterion #4 marked `partial` on both L4 and L6 axes; the
  Pillar-8 temporal sandwich and the L4 cascade address both land
  in their respective follow-up PRs.

- **If R2 fires AND R10 fires** → Sprint reset. R10 alone forces an
  emergency design review; R2 compounds the calendar. Defer PR-X4
  spawn to W4.5 at earliest; reassess scope at the design review.

- **If R3 fires AND R6 fires** → `precise_exp_x16` follow-up PR
  opened AND closure swap re-staged. The audit-failure motivates the
  same kernel that the latency-failure needs (a precise, latency-bounded
  revision compose). Couple the two: spec `precise_exp_x16` with the
  latency budget that closes SG4.

- **If R7 fires (alone OR with any other)** → **Halt sprint.**
  Structural failure on the alpha path subsumes every other failure
  mode; the cascade is dropping behind regardless of which path is
  active. No partial close; design review precedes any further work.

- **If R8 fires (alone OR with any other)** → Reject the offending PR;
  re-spawn the worker after the missing primitive lands. Other
  workers continue in parallel; sprint timeline absorbs the worker's
  re-spawn cost (typically 1-2 days).

- **If three or more risks fire simultaneously (excluding R4/R8)** →
  Convene savant-architect + sentinel-qa + l3-strategist for a
  scope-reset review. The sprint plan is no longer the operating
  document; replan against the live constraint set before continuing.
