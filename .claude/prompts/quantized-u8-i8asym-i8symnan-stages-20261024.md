# Claude Code staged implementation prompt: ndarray three 8-bit modes

Repository: `AdaWorldAPI/ndarray`. Continue **draft PR #370** on branch `research/i8-rne-dual-range-20261024`. Do not open a duplicate PR. This PR owns only the additive contracts in `src/hpc/quantized.rs`, tests, and this stage prompt.

## Architectural rule

**ndarray already is the contract. Add the missing leg, do not create an abstraction framework.** Reuse existing RNE primitives. Keep existing public APIs and production wire formats unchanged. No dependencies on Helix, FamilyGamma, BGZ, Fisher-Z, DR, Palette256, or ONNX runtime. Those are consumers/references, not prerequisites.

Three modes to examine distinctly:

| Mode | Source of truth | Finite code range | Invalid semantics |
| --- | --- | --- | --- |
| U8 legacy | `quantize_f32_to_u8` already in ndarray | 0..255; fitted affine parameters | Legacy behavior, no new NaN claim |
| I8 asymmetric/full | `quantize_f32_to_i8_rne_full(data, scale, zero_point)` in #370 | -128..127; externally supplied scale, arbitrary signed zero point | Reject nonfinite; -128 is **numeric** |
| I8 symmetric + NaN | `quantize_f32_to_i8_rne_symmetric_nan(data, scale)` in #370 | -127..127; zero point 0 | -128 is **NaN sentinel**; reject +/-infinity |

The NaN sentinel is a custom **integer codec marker**. IEEE-754 NaN applies only after decoding at a named observation boundary. Plain i8 arithmetic does NOT automatically propagate it. All 256 `u8` Palette indices remain valid; they are *identities*, not signed numbers.

## Stage 0: source audit and baseline, read-only

1. Read `src/hpc/quantized.rs` and existing `src/hpc/vml.rs` / SIMD rounding. Do not recreate ties-to-even or replace existing `quantize_f32_to_i8`.
2. Check the legacy U8 affine mapping with [0, 0.5, 1]. Code 128 is the rounded midpoint under fitted `scale=1/255`, **but decoded 128/255 is not exactly 0.5**.
3. Separately evaluate the hypothetical exact-midpoint mapping `scale=0.5/128=1/256`: 128 maps exactly to 0.5, but 255 maps to 255/256, not 1.0. Do not silently change U8 semantics to resolve this tradeoff. Record the SPOFC analogy as a hypothesis, not a verified equivalence.
4. Establish current branch/CI state and check #370 for newer edits before modifying any file.

**Gate:** short baseline inventory with source paths and exact formulas. No code in this stage.

## Stage 1: complete only the additive signed leg inside #370

1. Keep the full signed-i8 RNE path with -128 valid. Keep any existing callers untouched.
2. The symmetric alternative must map input NaN to `I8_SYMMETRIC_NAN_CODE == i8::MIN`, clamp finite values to -127..127, reject infinities, and validate positive finite scale. It must not emit -128 for finite values, even on overflow.
3. Provide an explicit decoder for test/export use; never assume decoding belongs in the native hot path.
4. Run the repo's fmt, targeted unit tests and clippy gates. Check against current `master` and resolve real failures, reporting commands and results honestly.

**Gate:** all 256 raw codes decoded (one NaN + 255 finite); all 255 finite codes re-encode identically at scale 1; ties-even including negative half-ties; both saturations; NaN and infinity; invalid scale; negative and positive zero; signed symmetry; nonzero asym zero point; explicit -128 divergence. Avoid a `Vec<f32>` claim of IEEE-exact intermediate execution.

## Stage 2: three-mode comparison, measurement only

1. Run all three on identical **documented data distributions**, including symmetric, negatively skewed, positively skewed, near-zero, saturated, and NaN-bearing inputs.
2. Compare signed paths with identical supplied scales and an independent scalar oracle. Also report fitted-calibration outcomes separately: absmax/127 often makes full and symmetric produce exactly the same finite codes. Preserve this negative control.
3. Compare U8 as its actual fitted affine baseline. If also testing exact-midpoint `1/256`, name it a *separate policy*, not a secret rewrite of U8.
4. Measure MAE, RMSE, signed bias, saturation, sentinel propagation, code occupancy and, where relevant, decision/rank agreement. Separate conversion cost from integer runtime cost; do not claim a speedup without benchmarks.
5. Verify that preserving -128 numerically can sometimes have **lower** reconstruction error than symmetric+NaN. Never presume symmetric is universally more accurate.

**Gate:** reproducible fixture, concise result table with units, seed, platform and commands. No changes to legacy APIs based solely on this experiment.

## Stage 3: optional native LUT closure, separate follow-up PR only

Treat `Mish` as an annotated **semantic operator**, and `LUT` as an annotated **execution strategy**. Never conflate them. Demonstrate a table-driven encoded-to-encoded path where a unary operation is at most a 256-entry lookup and an actually binary operation may use a 256x256 table. Reserve -128 as invalid in signed numerical LUTs and propagate it explicitly. Palette identity indices remain u8 0..255.

The table may be generated/calibrated offline with floats. **The hot path must stay entirely in integer identities/codes.** Instrument or audit this condition rather than merely asserting it. Fusion is allowed only if a separately declared observable error bound or exact domain proof survives composition.

**Gate:** deterministic LUT construction plus independent oracle, exhaustive finite-domain behavior, NaN propagation, layout and memory accounting, and real measured cost. Do not implement a new ONNX runtime or a graph compiler in ndarray.

## Stage 4: independent ONNX facade cross-check, separate PR

Only after Stage 1 lands, use `AdaWorldAPI/lance-graph/crates/lance-graph-onnx` as a **comparison facade**, not a runtime:
- Cross-check full signed-i8 RNE against an independent scalar ONNX QuantizeLinear reference (and ONNX Runtime where accessible). The current path is only the finite scalar/i8 sub-contract, not full ONNX shape/axis coverage.
- Cross-check symmetric+NaN against its own independent oracle. It is NOT an ONNX QuantizeLinear alias.
- Keep `Mish` semantics separate from any LUT execution annotation.
- State explicitly whether observation-boundary equivalence is exact, bounded, or refused; do not infer it from an annotation.

Do not change existing lance-graph draft PR #1758 or production wire formats unless separately authorized.

## Reporting contract for CC

After each stage: `commit/branch | changed files | tests actually run | measured evidence | contradictions | next gate`. Mark unrun work **NOT RUN**. No FamilyGamma/Helix/HighHeelBGZ migrations, no speculative performance claims, and no duplicate PRs.