# Plan: 8-bit code spaces as one compile-time codec, plus certified tanh tables (v3, RATIFIED)

**Status:** RATIFIED by a 5+3 council on 2026-10-24 (v1 → v2 → v3; council record in §9).
Phase (a) is executable. The filename keeps `-v2` so links stay stable; the content is v3.
Replaces the staged prompt `.claude/prompts/quantized-u8-i8asym-i8symnan-stages-20261024.md`
(superseded). PR #370 is being refactored to this plan.

**Source:** /coresearch 2026-10-24 (exploration map in
`.claude/knowledge/coresearch-2026-10-24-i8-codec8-map.md`), the TFLM harvest (AdaWorldAPI/tflite-micro
`.claude/harvest/2026-10-24-tflm-quant-lut-harvest.md`, merged), lab probes from the same session, and
the operator's rulings quoted in §2.

## 1. Goal

Three 8-bit modes, kept and run head to head (no single winner):

| mode | code space | range | NaN | storage |
|---|---|---|---|---|
| I8 asymmetric | `Asym` | -128..127, -128 a value | none | i8, or u8 = code ^ 0x80 |
| I8 symmetric + NaN | `Sym` + `NaNAtMin` | -127..127 | -128 | i8 (u8 + NaN parked, §8 item 3) |
| U8 for spoFC truth and codebooks | `UnitFull` | 0..255, value c/255 | none | u8, raw code |

U8 is a separate code space, not an XOR image of I8: `UnitFull` has no I8 counterpart. The XOR
relation applies only between I8 and u8 storage of `Asym` / `Sym`.

Plus 256-entry tables so tanh of a decoded code costs one byte load in native lance-graph.
Hypothesis under test (H-SYM, operator): -127..0 and 0..127 are symmetric with 0 as the median, so
the symmetric range has no skew. Not assumed; P-01 measures it.

## 2. Rulings this plan is bound by

- **No runtime dispatch.** Operator: "Lookup runtime dispatch is a category error we want no
  dispatch overhead cost that's why the compile time parameters. For Lab calibration when
  performance is negligible and recompiling hurt's you can use it behind a Cargo parameter, the
  performance needs to have it compile time."
- **NaN policy is a compile-time parameter.** Storage (i8 | u8) is a compile-time parameter with
  storage-specific SIMD code.
- **LLVM output is measured**, so the parameters cannot degrade code silently.
- **Re-export, not re-implementation.** `lance-graph-tflite` and `lance-graph-onnx` are `pub use`
  facades; the implementations live here.
- **Unchanged:** `quantize_f32_to_i8` (`src/hpc/quantized.rs:545`), `dequantize_i8_to_f32` (:586),
  `quantize_f32_to_u8` (:517), `QuantParams` (:496), FamilyGamma and its -128 interpretation, the
  persisted Fisher-Z format, every wire format.
- **Not consumers:** FamilyGamma and bgz-tensor. No dependency on them.

## 3. Facts the design rests on

Verified by reading the code at `62c43c9` unless marked otherwise.

- `quantize_f32_to_i8` (:545): scale `abs_max / 127.0`, falling back to 1.0 when abs_max is 0
  (:549); `(v / scale).round().clamp(-128.0, 127.0) as i8` (:553). `f32::round` rounds ties away from
  zero. A NaN input becomes 0 through the saturating cast. The min/max folds use `f32::min`/`max`,
  which ignore NaN.
- `pub mod hpc` (`src/lib.rs:524-526`) is `#[cfg(feature = "std")]`, so codec8 never compiles on the
  no_std row. It also carries `#[allow(clippy::all, unused_imports, unused_variables, unused_mut,
  dead_code)]`, so clippy does not check files under `hpc/` today. `#![warn(missing_docs)]` is
  crate-wide (:31).
- `ci.yaml:48` sets `RUSTFLAGS` to `-D warnings` plus two cfgs. Jobs that run `env -u RUSTFLAGS`
  (e.g. `ci.yaml:334-336`) do not get `-D warnings`.
- Per the council's S1/S2 searches of `src/simd_avx512.rs` (not re-read for every backend):
  - `I8x64` (:2003-2081) has splat, zero, from/to slice/array, add, sub, min, max, cmp_gt.
  - `U8x64` (:522-851) has cmpeq_mask (:609), cmpgt_mask, mask_blend, saturating ops,
    permute/shuffle_bytes and BitXor.
  - `F32x16` has round, floor, simd_clamp, cast_i32.
  - No f32→i8 round-and-narrow primitive turned up in that file.
- Generator precedent: `tools/gen_ternlog_bodies.py` has a `--check` mode that runs in
  `.github/workflows/simd-matrix.yaml`. A repo-wide glob found no `*_generated.rs` and no tablegen
  crate. `Cargo.toml:516-518` makes every `crates/*` directory a workspace member.
- The simd-matrix path filter (`simd-matrix.yaml:3-19`) does not include `src/hpc/codec8*`.
- Reported by council savant S3, not re-run by the reviewers: over tie-adjacent inputs, `v / s` and
  `v * (1/s)` produced 25,928 different rounded codes. For example, at s = 0.0137 and v = -1.73305,
  `v / s` is -126.5 but `v * (1/s)` is -126.49999.
- Measured earlier this session (lab probes, scratchpad, not in the repo):
  - Generic vs hand-written kernels produce label-normalised-identical assembly on x86-64-v3 and
    -v4 (1.311 vs 1.308 ns/elem). A per-element mode match does not vectorise (1.85-1.96 ns/elem).
  - `f64::tanh` calls glibc, which is up to 1.95 ulp off. TiesEven and TiesAway give identical tanh
    tables, because 8-bit tanh outputs never land on a tie.
  - A rounded spoFC confidence reaches 255 (k=1, m=510), so a rounded encoder needs a cap at 254.
  - A u8×s8 VPMADDUBSW pair sum saturates int16 even with valid weights.

## 4. Design (committed)

**D1 Module.** `src/hpc/codec8.rs`, plus `pub mod codec8;` in `src/hpc/mod.rs`.
- The file opens with `#![warn(clippy::all)]` to undo the parent allow. A negative control (§6)
  proves this override is real.
- No `unsafe`. Tests are endian-neutral, because the i686 and s390x rows compile hpc.

**D2 Markers.** All markers are sealed zero-sized types.
- `Storage`: `I8`, `U8` (associated `Elem`).
- `CodeSpace`: `Asym` (-128..127), `Sym` (-127..127), `UnitFull` (c/255, 0..255).
- `#[cfg(test)]`-only P-16 arms: `UnitOdd` (c/254) and `UnitMidrise` ((c+0.5)/256). They become
  public only if P-16 adopts one.
- `Rounding`: `TiesAway`, `TiesEven`, `Floor`.
- `NanPolicy<S>`: `NoNaN` for every space; `NaNAtMin` implemented for `Sym` only.
- `struct Codec8<T, S, R, N> where N: NanPolicy<S>`. So `Codec8<I8, Asym, TiesAway, NaNAtMin>` is
  expected to fail with E0277; G3 checks it.

**D3 Calibration and encode/decode.** `trait Calibration { const SCALE: f32; const ZERO_POINT: i32; }`.
- Encode, per element: `R::round(v / C::SCALE)` (a true division; never a precomputed reciprocal
  in a rounding mode), then `+ ZERO_POINT`, then clamp `(S::MIN, S::MAX)` in f32, then a saturating
  cast. For NaN input, `N::NAN_CODE` replaces the result.
- Decode is defined per code space:
  - Asym/Sym: `(c - ZP) as f32 * C::SCALE`, mirroring `dequantize_i8_to_f32`.
  - UnitFull: `c as f32 / 255.0`, a division so that 255 decodes to exactly 1.0.
- UnitFull also has an integer encoder `encode_ratio(n, d) = floor(n·255/d).min(255)`, the spoFC
  form. G13 checks it against the spoFC encoder.
- Signatures are `encode(&[f32], &mut [T::Elem])` and `decode(&[T::Elem], &mut [f32])`: no `Vec`,
  no `Result` in the hot body.

**D4 Storage.** I8 stores the code itself. U8 stores `code ^ 0x80` for `Asym` / `Sym` (TFLM requantize
`requantize.h:47`) and the raw code for `UnitFull`. For U8 + `Sym` the unused slot is 0x00.

**D5 Boundary.**
- `encode_checked` / `decode_checked` are `pub` and validate once per slice, then run the unchecked
  body.
- The unchecked bodies are `pub(crate)`. Out-of-range or NaN input to them gives a defined
  saturating result, never undefined behaviour. So a missed validation produces a wrong but defined
  value, and a test pins that value.

**D6 Aliases.** Each alias is a named preset that fixes a rounding for a known external format; the
rounding stays a separate parameter.

| alias | parameters | exported |
|---|---|---|
| `TfliteAct` | `<I8, Asym, TiesAway, NoNaN>` | phase (a) |
| `TfliteWeight` | `<I8, Sym, TiesAway, NoNaN>` | phase (a) |
| `OnnxQ` | `<I8, Asym, TiesEven, NoNaN>` | phase (a) |
| `SpoFcTruth` | `<U8, UnitFull, Floor, NoNaN>` | phase (a) |
| `NativeLane` | `<I8, Sym, R, NaNAtMin>` | not until §8 item 1 is decided; meanwhile tested with both roundings |

**D7 Tables (phase c).**

| mode | tables | NaN slot |
|---|---|---|
| I8 asym | unary code→code tanh law `[i8; 256]`; f32 boundary `[f32; 256]` | none |
| I8 sym + NaN | same two tables | entry for code -128 is the NaN code / f32 NaN |
| U8 UnitFull | f32 boundary `[f32; 256]`; unary law into a named output space | none |

- **Generator.** `tools/gen_codec8_tables.py` (mpmath), following `tools/gen_ternlog_bodies.py`. It
  writes `src/hpc/codec8/tables.rs` (with a "generated, do not edit" header) and binary 64 KiB laws
  as `src/hpc/codec8/laws/*.bin` read with `include_bytes!`. Laws are generated only on demand.
- **Generator design** (not built yet):
  - interval narrowing for each cell;
  - an explicit error naming the cell when the precision limit is reached;
  - signed zero handled explicitly;
  - the NaN slot handled explicitly;
  - decode overflow defined: tanh(±inf) = ±1, or a generation error if the calibration makes it
    reachable unintentionally.
- **Reference.** The tables use correctly rounded tanh at the exact decoded f32 value. `vstanh` is
  only a differential accuracy test, never the definition.

**D8 Calibration source and lab.**
- **Calibration markers.** For a given model they come from a calibration manifest, read by the
  same generator.
- **Arbitrary-model conformance.** It runs in tests under the Cargo feature `lab-calibration`.
- **What `lab-calibration` adds:**
  - `LabCodec8<T, S, R, N>`, a concrete type with an owned `Box<[T::Elem; 256]>`;
  - `prepare() -> Result<_, PrepareError>`, which may fail;
  - lookup that takes `&self`.
- **What it never has:** no enum, no `dyn`, no `OnceLock`/`LazyLock`, and it is off by default.

**D9 Landing order.** Each step is its own PR.
- (a) codec8 scalar + Phase 0 tests + asm/timing probe.
- (b) SIMD primitives in `src/simd_*.rs` under the W1a contract
  (`.claude/knowledge/vertical-simd-consumer-contract.md`: all backends, parity test, `///` docs
  with examples), then codec8 SIMD. Candidate lookup primitive: `U8x64::permute_bytes`, measured
  against gather.
- (c) generator + tables.
- (d) laws.
- (e) head-to-head probes.
- (f) facade PRs in lance-graph.

## 5. Gates (pre-registered)

- **G1 Local checks.** Before any push, the orchestrator alone runs, with env
  `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0`:
  - `cargo fmt --check`;
  - `RUSTFLAGS="-D warnings" cargo clippy -p ndarray --all-targets`;
  - `cargo test --release -p ndarray --lib hpc::codec8`;
  - `cargo test --doc -p ndarray codec8`.
- **G2 Byte identity.** `TfliteAct` with `SCALE = f32::from_bits(B)` equals `quantize_f32_to_i8`
  bit for bit on a fixed corpus, including NaN and ±inf rows.
  - B is the fitted scale of that corpus. The test asserts that the runtime fit produces exactly B
    and that the corpus does not trigger the 1.0 fallback.
  - This gate tests **data rounding** only.
- **G3 Compile-fail.** A `compile_fail` doctest `let _: Codec8<I8, Asym, TiesAway, NaNAtMin>;`, plus
  a compiling twin with `Sym`.
- **G4 XOR storage.** U8 storage equals I8 storage ^ 0x80 on all 256 codes for every Asym/Sym alias.
- **G5 Assembly.** For each alias vs a hand-written kernel, label-normalised assembly on x86-64-v3
  and -v4 is identical, else FAIL with the diff reported. A negative control must differ, and the
  tier is printed.
- **G6 Timing.** Each alias is within 2% of the hand-written kernel, over ≥5 runs, order-swapped,
  in ns/elem, with the tier printed. The 2% is a pre-registered threshold, not derived from
  measured variance.
- **G7 Table cells.** Every table cell equals the reference, computed by mpmath **independently of
  the generator code path**. This gate tests **table rounding** only (decoded f32 → tanh → target
  cell).
  - Disable 1: index offset 128 (indexing).
  - Disable 2: wrong calibration (decode affine).
  - Either disable must turn the gate red.
- **G8 Generator check.** `tools/gen_codec8_tables.py --check` runs as a step in
  `.github/workflows/simd-matrix.yaml`, path-triggered on `src/hpc/codec8/**` and `tools/gen_codec8*`.
- **G9 Lab feature.** A `ci.yaml` step asserts that `lab-calibration` is absent from
  `cargo tree -e features -p ndarray`.
- **G10 Lookup speed.** Lookup ≤ 0.5× `vstanh(decode)` ns/elem at 64 Ki and 16 Mi elements
  (release). Preparation cost is reported separately (lab).
- **G11 Table size.** Generated table bytes are reported per landing; more than 1 MiB in total
  needs an explicit decision.
- **G12 Backend parity.** From (b), the simd-matrix path filter includes `src/hpc/codec8*` in the
  same PR, and rounding parity is checked on every arm (wasm, neon-qemu, avx-qemu, v2-qemu,
  nightly).
- **G13 spoFC bytes.** `encode_ratio` equals the spoFC encoder
  (lance-graph `crates/lance-graph-arm-discovery/src/translator.rs:61-120`, transcribed into the
  test with a source citation) over the evidence grid m in 0..=4096, k in {1, 2, 5}.
- **G14 Head to head.** In phase (e), the three modes run on the same corpus: per-mode error (RMSE,
  bias in LSB), -128 emissions and ns/elem, in one table. H-SYM is refuted only where Asym actually
  emits -128.

## 6. Checklist

### Phase (a)
- [ ] `src/hpc/codec8.rs`: D1-D6 scalar, docs with an example on every `pub` item.
- [ ] Tests:
  - [ ] G2, G3, G4, G13.
  - [ ] P-02 ties, including negative half-ties.
  - [ ] P-04 Sym+NaNAtMin: NaN, ±1e9, ±inf (as proposal), ±0.
  - [ ] P-12 encode∘decode on all 256 codes per alias.
  - [ ] Sym+NoNaN vs Sym+NaNAtMin agree everywhere except NaN and -128.
  - [ ] D5: the pinned defined result on out-of-range input.
  - [ ] Carried from #370: signed zero, both saturations, nonzero zero point.
- [ ] Negative control for D1, run before the commit and recorded: a deliberate clippy violation in
      codec8.rs fails G1, then is reverted.
- [ ] `examples/codec8_asm_probe.rs` (G5/G6), ported from the lab `codec-asm-probe`. It has its own
      doc comment and is not in the default test run.
- [ ] Commit discipline:
  - edit script `&&` `git add` `&&` `git commit` (never sequenced);
  - the blackboard entry and plan ticks are in the same commit;
  - no model identifier anywhere except the attribution trailer.
- [ ] Workers, if any, edit only. The orchestrator is the single compiler on the shared `target/`.
      Every timing names its tier.

### Phases (b)-(f)
Tracked per PR as listed in D9 and §5.

## 7. Non-goals

- Existing quantizers, FamilyGamma, Fisher-Z and wire formats.
- An ONNX runtime, a graph compiler, or operator fusion.
- A port of TFLM's gemmlowp code (facade work, later).
- The cyclic Z_256 space and the U8 sign-partition (parked).
- Real-model data runs (manifest D stays NOT RUN).

## 8. Open for the operator

1. **NativeLane rounding:** TiesEven (proposed) or TiesAway.
   - Restore condition: the decision exports the `NativeLane` alias.
2. **spoFC rounding:** Floor is kept as today's behaviour.
   - Rounding stays a P-16 head-to-head variable (Floor vs TiesEven with a 254 cap), not a settled
     fact.
3. **U8 + NaNAtMin** (a zero-filled buffer reads as all-NaN): a requested comparison arm, parked.
   - Un-park evidence: P-15 / P-16 show a consumer that stores NaN-capable values in u8, or you ask
     for the arm.
4. **Certification tool:** mpmath in `tools/` (proposed, matching `gen_ternlog_bodies.py`).
5. **±inf under Sym+NaNAtMin:** saturate to ±127 (proposed, and tested as a proposal) or NaN.

## 9. Council record (2026-10-24)

- **Five savants (Sonnet):**
  - prior-art: 8 findings;
  - iron-rule: 10;
  - code-truth (runtime-archaeologist charter): 9;
  - cascade-impact: 7;
  - creative-explorer: 6.
- **Draft v2:** 18 changes. The main ones:
  - a true division in encode;
  - exact scale bits for G2;
  - a concrete `compile_fail` with a twin;
  - the clippy override in codec8;
  - the generator in `tools/` instead of a crate;
  - SIMD primitives first;
  - the path filter;
  - XOR scoped to Asym/Sym;
  - UnitOdd/UnitMidrise test-only;
  - NativeLane held back;
  - the calibration source;
  - the size budget;
  - the blackboard rule.
- **Three reviewers (Sonnet), on draft v2:**
  - overclaim-auditor: 0 BLOCK, 3 FIX(P1), 3 FIX(P2);
  - dilution-collapse-sentinel: 0 BLOCK, 3 FIX(P1), 2 FIX(P2);
  - firewall-warden: 0 BLOCK, 1 FIX(P1), 3 FIX(P2).
- **v3 adds:**
  - claims labelled by evidence (§3);
  - the per-mode table inventory (D7);
  - the D5 defined result for unchecked calls;
  - data vs table rounding kept separate (G2, G7);
  - G7 reference independent of the generator;
  - G13 spoFC bytes and G14 head-to-head;
  - restore conditions for parked arms (§8);
  - the clippy negative control;
  - DEBUG=0 and single-compiler discipline;
  - named CI rows for G8/G9.
- **Superseded:**
  - the runtime mode enum and per-slice selector;
  - a separate `SymNaN` space;
  - runtime calibration in production and the runtime certifier;
  - "U8 deferred";
  - `crates/codec8-tablegen`;
  - PR #370's per-mode runtime-scale functions.

## 10. Not searched / not run

- TFLM per-target kernels (`cmsis_nn/`, `xtensa/`, `ceva/`) and the gemmlowp fixed-point tanh body.
- No real-model data.
- No ONNX Runtime comparison.
- SIMD backends other than AVX-512 not inventoried for the missing primitives (phase b does that).
