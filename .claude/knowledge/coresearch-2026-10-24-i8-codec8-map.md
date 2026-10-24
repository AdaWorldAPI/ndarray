# /coresearch exploration map: ndarray 8-bit code spaces + tanh tables (2026-10-24)

> Banked verbatim from the session scratchpad so the council record survives the container. **Read through `.claude/plans/quantized-codec8-v2.md`:** the plan supersedes X1 (runtime enum → compile-time `Codec8` markers), X11 (U8 is in scope now) and the X12 key (calibration is compile-time). The verdicts and probes otherwise stand.


Live consumers: ndarray, lance-graph-onnx (#1758 draft). Not consumers: FamilyGamma, bgz-tensor i8 encoders (operator). palette256 = index consumer of all 256 codes. thinking-engine: unconfirmed.

## Verdicts
| # | idea | verdict | reason |
|---|---|---|---|
| X1 | `I8CodeSpace {Asym, SymNaN}` (#[non_exhaustive], U8 later) + `Rounding {TiesAway, TiesEven}` beside, not inside, QuantParams | ADOPT-NOW (design) | all three co-architects agree; QuantParams has pub fields |
| X2 | Asym = today's quantize/dequantize byte-identical (ties-away) | ADOPT-NOW | A3 "as is"; pin with byte-identity test |
| X3/X4 | SymNaN: clamp(-127,127), NaN -> -128, decode(-128) = NaN | PROBE (P-04, P-12) then adopt | precedent posit NaR / ONNX float8 FNUZ; OCP MX rejected variant recorded |
| X5 | today's NaN->0, inf->scale inf defect | PROBE (P-03 pin) | behaviour change to Asym needs its own decision; not in this work |
| X6 | NaN-preserving i8 neg/abs | PARK | cheaper: decode to f32, no i8 arithmetic in SymNaN space; saturating_abs untouched |
| X7 | rounding as explicit field | PROBE (P-02) | precedents split (ONNX/OCP RNE, TFLite/ndarray away) |
| X8 | H-SYM head-to-head | PROBE (P-01, P-13) | expected: no gain on data-derived scales (-128 never emitted); gain only where -128 is emitted |
| X9 | 255 vs 256 doc table | ADOPT-NOW (doc) | A7; SymNaN never as index |
| X10 | i16 pair-accumulation | PARK | only s8xs8 paths; no consumer |
| X11 | U8 | PARK (deferred, A3) | offset = Asym ^ 0x80; midrise has no exact zero |
| X12 | `CodeLut256` (keyed spec+scale+zp+f) + mode-agnostic `lut256_apply` via ndarray::simd (all backends) | PROBE (P-07, P-09 gate) | adopt only if <= 0.5x vstanh(decode) ns/elem |
| X13 | 128-entry odd-mirror table | PROBE (P-08) | needs vstanh bitwise odd |
| X14 | "free" measured in ns | ADOPT-NOW (gate) | release, tier named, == not <=ULP |
| X15 | Arrow bitmap vs in-band NaN | PARK (Lance boundary) | decide when a Lance column carries SymNaN |
| X16 | lance-graph-onnx per-mode exactness tests (test-only) | ADOPT-NOW (after X1/X12 land) | A8; guard: non-test src never names the codec |
| X17 | "LUT next to mish" | SKIP (placement) | mechanism kept, placed with the codec |
| X18 | vml.rs:637 "2 ULP" -> "<= 1 ULP" | ADOPT-NOW (separate one-liner) | doc drift |

## Corrections found during synthesis
- H-SYM count: 127 neg + 0 + 127 pos (0 is the unique median code); I8asym mean -0.5 LSB only where -128 is emitted.
- Falsifier P-01 refutation (a) contradicted its own silent outcome; fixed: refute only where Asym emits -128, strict >; elsewhere assert equality.
- lance-graph-onnx already depends on ndarray normally; "dev-dep only" is moot. A2 enforcement = non-test src does not reference the codec module.

## Not searched
XNNPACK/oneDNN LUT internals; ONNX opset diffs 19->28; Lance's own null docs; real-data histograms (D3, NOT RUN); thinking-engine live-ness.

## Addendum — helix / Signed360 (operator detour, 2026-10-24)
| # | idea | verdict | reason |
|---|---|---|---|
| X19 | Cyclic code space (Z_256): wrap, no NaN, neg=(256-c)&0xFF, ring metric, circular mean | PARK (record as a named kind; `#[non_exhaustive]` leaves room) | helix ships the metric (`DistanceLut::circular`), not a codec; no live consumer; its tables are sin/cos (PhaseLut), not tanh |
| X20 | **The tanh table is the decode, not a post-step**: with codes uniform in z, value = tanh(z0 + (c[+0.5])·dz) | ADOPT-NOW (design correction) | helix measured uniform-in-z axis 1.0x distortion vs 11.6x (y) / 204x (r); bounded pole error vs divergent |
| X21 | Table key must carry the full decode affine (z0, dz, midtread vs midrise), not just "scale" | ADOPT-NOW | otherwise a tanh of a linearly decoded value is a different code |
| X22 | SymNaN is i8-only; never reinterpret u8 helix bytes (0x80 = +0 / bucket 128 / azimuth π) | ADOPT-NOW (doc + type) | namespace collision on 0x80 |
| X23 | U8 sign-partition (helix polar: two zeros, neg = c^0xFF) | PARK with U8 | measured pole penalty 1/r |
| X24 | Helix two azimuth rounding conventions | report only | out of scope; helix not live |

## Addendum — Moore tenant / Palette256 law (operator, 2026-10-24)
Read on origin/main: `cognitive-shader-driver/src/palette_perturbation.rs` (module doc + PaletteLut/PairAddress), `tests/moore_palette_pairs_lut.rs`, board entry `2026-10-08-hhtl-nars-moore-value-tenants.md`.
- `ValueTenant::MoorePalettePairs` (ord 19): 8 Moore slots NW..SE, each a `(u8:u8)` Palette256 pair; law address `(first<<8)|second`. Layout TEST-PINNED; no production writer/reader yet.
- The law is closed: `Palette256 × Palette256 -> Palette256` (u8,u8 -> u8), 64 KiB, "Runtime code does not reconstruct cosine, Fisher-Z, covariance, or any other continuous representation." Calibration pays the tax before the hot path; evaluation = one indexed byte load. Composition `[a,b]->r0, [c,d]->r1, [r0,r1]->delta` keeps intermediates in registers; no 256^4 table, no intermediate population.

| # | idea | verdict | reason |
|---|---|---|---|
| X25 | **Stay in code space**: the native hot path for tanh-domain values is a closed code->code law (u8 or i8, 256 / 256×256 entries returning a CODE), never decode->f32->tanh per element | ADOPT-NOW (design correction) | Moore/Palette256 precedent; matches helix "table bake once, never reconstruct" |
| X26 | The f32 `CodeLut256` (tanh(decode(c))) is the BOUNDARY / ORACLE table: used where a float must leave (ONNX facade conformance, export), and as the oracle the code->code law is baked from | ADOPT-NOW (reposition X12) | keeps A8 exactness tests; removes the per-element f32 path from the native claim |
| X27 | Unary code law `[f32? no] code -> code` (256 B) for element-wise ops in a code space (e.g. tanh re-quantized into an output code space, TFLite S3 shape) beside the binary 64 KiB law | PROBE | TFLite int8 tanh LUT is exactly this (int8 out); bake = decode -> vstanh -> encode once |
| X28 | Which code space the law runs in: SymNaN value lanes (NaN slot must map NaN->NaN in every law) vs Asym/Palette256 indices (all 256) | PROBE | law totality: a SymNaN law must send code -128 to -128 in both operands; test exhaustively |
