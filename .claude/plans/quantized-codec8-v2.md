# Plan: 8-bit code spaces as one compile-time codec, plus certified tanh tables (v2)

**Status:** PROPOSAL. Replaces v1 (scratchpad draft, never landed) and the staged prompt
`.claude/prompts/quantized-u8-i8asym-i8symnan-stages-20261024.md` (superseded, see its
header). Goes through /5plus3 before any code lands. PR #370 is being refactored to this plan.

**Source:** /coresearch 2026-10-24 (scouts, crosswalk, three co-architects, exploration map),
the TFLM harvest (AdaWorldAPI/tflite-micro `.claude/harvest/2026-10-24-tflm-quant-lut-harvest.md`,
merged), two lab probes in this session (asm identity, f64 tanh), and the operator's rulings
quoted below.

## 1. Goal

Three 8-bit modes in ndarray, run head to head and kept, not one winner:

| mode | range | -128 | today |
|---|---|---|---|
| I8 asymmetric | -128..127 | a value | `quantize_f32_to_i8` (ties away) |
| I8 symmetric + NaN | -127..127 | NaN | new |
| U8 | 0..255 | (per code space) | spoFC truth, codebooks |

Plus 256-entry tables so that tanh of a decoded code costs one byte load in native lance-graph.
Hypothesis under test (H-SYM, operator): -127..0 and 0..127 are symmetric with 0 as the median,
so the symmetric range has no skew. Not assumed; measured by P-01.

## 2. Rulings this plan is bound by

- **No runtime dispatch.** Operator: "Lookup runtime dispatch is a category error we want no
  dispatch overhead cost that's why the compile time parameters. For Lab calibration when
  performance is negligible and recompiling hurt's you can use it behind a Cargo parameter, the
  performance needs to have it compile time."
- **NaN is a compile-time parameter** ("Just make NaN another compile time parameter").
- **Signed/unsigned storage is a parameter**, with storage-specific SIMD code.
- **Measure LLVM output** so the parameters cannot introduce silent degradation.
- **Re-export, not re-implementation:** `lance-graph-tflite` and `lance-graph-onnx` are `pub use`
  facades; implementations live here.
- **Unchanged:** `quantize_f32_to_i8`, `dequantize_i8_to_f32`, `quantize_f32_to_u8`, FamilyGamma and
  its -128 interpretation, the persisted Fisher-Z format, every existing wire format.
- **Not consumers:** FamilyGamma and bgz-tensor (operator: nothing consumes FamilyGamma; bgz-tensor
  is listed only for palette256 and never touches it). No dependency on them.

## 3. Design

### 3.1 One codec, four marker parameters

```rust
pub struct Codec8<T: Storage, S: CodeSpace, R: Rounding, N: NanPolicy<S>>(PhantomData<(T, S, R, N)>);
```

All four are sealed zero-sized marker traits, selected by monomorphisation.

| parameter | markers | what it fixes |
|---|---|---|
| `Storage` | `I8`, `U8` | element type, the `ndarray::simd` wrapper (`I8x64` / `U8x64`), storage-specific kernels (compares, min/max, MAC shapes) |
| `CodeSpace` | `Asym` (-128..127), `Sym` (narrow, -128 never emitted), `UnitFull` (c/255, endpoints exact), `UnitOdd` (c/254, exact 0 / 0.5 / 1, code 255 free), `UnitMidrise` (test arm only) | `MIN`, `MAX`, decode affine |
| `Rounding` | `TiesAway`, `TiesEven`, `Floor` | the rounding function |
| `NanPolicy<S>` | `NoNaN`, `NaNAtMin` | `NAN_CODE: Option<i32>` |

`Index` (palette256 / codebook ordinals) is not a `CodeSpace`: it has no `Codec8` impl, only
lookup and ordinal operations. A cyclic Z_256 space (helix azimuth) is parked.

**Constraint at compile time:** `NaNAtMin` implements `NanPolicy<S>` only for narrow spaces, so
`Codec8<_, Asym, _, NaNAtMin>` does not compile (a `compile_fail` doc test pins it).

**One body.** Encode, decode and clamp read `S::MIN`, `S::MAX` and `N::NAN_CODE`. The NaN branch is
a match on a constant and disappears for `NoNaN`. Storage `U8` stores the I8 code XOR 0x80 (TFLM's
own requantize fast path, `requantize.h:47`).

**Aliases (the only names facades re-export):**

| alias | definition |
|---|---|
| TFLite activation | `Codec8<I8, Asym, TiesAway, NoNaN>` |
| TFLite weight | `Codec8<I8, Sym, TiesAway, NoNaN>` |
| ONNX `QuantizeLinear` | `Codec8<I8, Asym, TiesEven, NoNaN>` |
| native value lane | `Codec8<I8, Sym, TiesEven, NaNAtMin>` (rounding: proposal, operator open item 1) |
| spoFC truth | `Codec8<U8, UnitFull, Floor, NoNaN>` (today's behaviour; kept until P-16) |

### 3.2 Calibration is compile-time

```rust
pub trait Calibration { const SCALE: f32; const ZERO_POINT: i32; }
```

A calibration is a marker type. Production code never takes scale or zero point as a runtime
argument. Runtime calibration exists only behind the Cargo feature `lab-calibration` (off by
default, enabled by no production crate) for sweeps where recompiling hurts and speed does not
matter; a lab table that is adopted is promoted by generating its compile-time source.

### 3.3 Preparation and execution are separate

- **Preparation** happens offline (production) or in the lab prepare step. It may compute,
  certify, raise precision, and fail.
- **Execution** is a direct lookup into a `const` array (production) or an owned array held by its
  concrete type (lab). Execution performs no codec, rounding or NaN selection, uses no trait objects,
  function pointers or dispatchers, does no certification or precision raising, never calls tanh,
  and never initialises lazily (no `OnceLock` / `LazyLock` on the lookup path).

### 3.4 Certified tables

- Tables are `const TABLE: [T::Elem; 256]` (code-to-code laws) or `[f32; 256]` (boundary tables),
  one per (Storage, CodeSpace, Rounding, NanPolicy, Calibration, function), generated offline into
  checked-in Rust source. A regeneration test proves the committed source equals the generator's
  output (stronger than a stored hash).
- **Reference:** correctly rounded mathematical tanh at the exact value of the decoded f32, i.e.
  `c as f32 * scale` exactly as the runtime decode computes it. `vstanh` is a separate
  implementation with its own measured bound (differential accuracy test), never the definition.
- **Certificate per cell:** an enclosing interval from directed rounding; raise precision until
  both ends round to the same target cell (an 8-bit code under the encoder's affine and rounding,
  or f32 bits). Double-double is one precision level, not a proof. At the stated precision limit
  the generator returns an explicit error naming the cell. It never writes an uncertified table.
- **Tools:** existing ones only (MPFR via mpmath or `rug`; a correctly rounded binary64 tanh such as
  CORE-MATH as a candidate generator). No new transcendental library. Rust's `f64::tanh` calls the
  platform libm with unspecified precision, so no table may depend on it.
- **Edge cases stated in the contract:**
  - signed zero: tanh(-0.0) = -0.0 in f32 tables; code tables have one zero;
  - NaN slot per storage: i8 0x80, U8 (XOR) 0x00; every law maps NaN to NaN;
  - decode overflow: `c·scale` reaching ±inf in f32 maps to tanh(±inf) = ±1, or is a preparation
    error if the calibration makes it reachable unintentionally.

### 3.5 Native path stays in code space

Element-wise and pairwise ops on tanh-domain values are closed code-to-code laws baked before the
hot path: unary 256 B, binary 64 KiB indexed `(first << 8) | second` (the `PairAddress` orientation
of lance-graph's `MoorePalettePairs` / Palette256 law). The f32 table is the boundary and the
oracle, used where a float must leave (ONNX conformance, export). Helix lesson: with codes uniform
in z (Fisher-Z), the table IS the decode, `tanh(z0 + c·dz)`; uniform-in-z versus uniform-in-value is
a design axis for P-14, not only the range.

### 3.6 Boundaries for NoNaN

Validation happens once per slice at the boundary, never per element (per-element checks measured
+40% and stop vectorisation). `encode_checked` returns an error if any input is NaN or infinite where
the mode rejects it; `decode_checked` returns an error if any code is the unused -128 under
`Sym + NoNaN`. The unchecked hot paths state that their input was validated.

### 3.7 SIMD

Only through `ndarray::simd` typed wrappers, all backends (AVX-512 / AVX2 / NEON / wasm / scalar),
with a parity test, per the W1a consumer contract
(`.claude/knowledge/vertical-simd-consumer-contract.md`). No raw intrinsics.

## 4. What is already measured (lab probes, this session)

- **Compile-time parameters cost nothing.** `codec-asm-probe` (rustc 1.99.0, release,
  codegen-units 1): generic `Codec8` versus hand-written kernels produce identical
  label-normalised assembly for SymNaN+TiesEven, Asym+TiesAway and UnitFull+Floor on x86-64-v3 and
  -v4; a negative control differs; all vectorise. 1.311 vs 1.308 ns/elem, order-swapped 1.314 vs
  1.305. A per-element mode match does not vectorise (1.85-1.96 ns/elem).
- **f64 tanh.** `f64::tanh` compiles to `call tanh@GLIBC_2.2.5`; glibc 2.39 is up to 1.95 ulp off.
  8-bit tanh outputs never land on a tie (tanh of a nonzero rational is transcendental), so
  TiesEven and TiesAway give identical tanh tables; only Floor differs. Rounding matters for data,
  not for baked tanh tables. glibc `tanhf` is not correctly rounded on 6.5% of samples.
- **Confidence rounding hazard.** Rounded spoFC confidence reaches 255 (k=1: m=510 gives 254.501),
  manufacturing certainty; Floor never does. Any rounded confidence encoder needs a cap at 254.
- **MAC.** A -128 weight term gives 32640, breaking TFLM's stated per-product bound (±32512,
  `conv.h:104-110`) without overflowing int16; u8×s8 VPMADDUBSW pair sums (255·127·2 = 64770)
  saturate int16 even with valid weights. The masked MAC arm needs an i32 accumulator or a proven
  bound; NaN detection per dot product needs only an any-bit test.

## 5. Checklist

### Phase 0: pins and contracts (tests first)
- [ ] P-03 pin today's `quantize_f32_to_i8` on NaN, ±inf, inf-in-data (values from a run)
- [ ] Byte identity: `Codec8<I8, Asym, TiesAway, NoNaN>` with today's fitted calibration equals
      `quantize_f32_to_i8` on a fixed corpus
- [ ] P-04 Sym+NaNAtMin contract (NaN, ±1e9, ±inf, -0.0); disables: floor -128, drop the NaN branch
- [ ] P-12 encode∘decode identity on all 256 codes per flavour
- [ ] P-02 ties: TiesEven vs TiesAway, exact counts incl. negative half-ties; non-tie twin
- [ ] Sym+NoNaN and Sym+NaNAtMin agree on all finite inputs and codes -127..127; differ only on NaN
      and -128
- [ ] `compile_fail`: `Codec8<_, Asym, _, NaNAtMin>`
- [ ] U8 storage equals I8 storage XOR 0x80 on all 256 codes, encode and decode, every flavour
- [ ] Carried from #370: signed zero, both saturations, negative half-ties, nonzero asym zero point,
      the explicit -128 divergence (as test inputs)

### Phase 1: codec
- [ ] `src/hpc/codec8.rs`: marker traits, `Codec8`, `Calibration`, aliases (scalar first)
- [ ] SIMD via `ndarray::simd`, parity test per backend
- [ ] asm gate: alias vs hand-written kernel identical per flavour (port `codec-asm-probe`);
      negative control must differ
- [ ] timing gate: alias within noise of direct (repeated + order-swapped, tier printed, ns)
- [ ] `lab-calibration` feature + test that no production facade enables it

### Phase 2: certified tables
- [ ] offline generator (outside the hot crate): interval certificate per cell, explicit error at
      the precision limit; signed zero, NaN slot, decode overflow
- [ ] committed `const` tables for the alias set; regeneration test
- [ ] P-07 every cell equals the independent certified reference; disables: index offset 128,
      wrong calibration
- [ ] P-08 odd mirror: decides whether a 128-entry form exists at all (exhaustive, not assumed)
- [ ] `vstanh` differential: f32 table vs `vstanh(decode(c))`, ULP histogram (reported, not a gate)
- [ ] P-09 lookup ≤ 0.5× `vstanh(decode)` ns/elem at 64 Ki and 16 Mi, release, tier printed;
      preparation cost reported separately from lookup cost

### Phase 2b: code laws
- [ ] unary law exhaustive: `law[c] == encode(tanh(decode(c)))` per flavour, all 256
- [ ] NaNAtMin totality: `law[NaN] == NaN`; binary law NaN in either operand gives NaN (256×256)
- [ ] binary 64 KiB law, `(first << 8) | second`, swapped-operand anti-vacuity

### Phase 3: head to head (measurement only)
- [ ] P-01 modes × populations (centred, skewed, uniform codes, 90% sparse): RMSE, bias (LSB),
      exact zeros, oddness violations, -128 emissions; calibration rule frozen before the run;
      H-SYM refuted only where Asym actually emits -128
- [ ] P-13 entropy gap
- [ ] P-14 uniform-in-z vs uniform-in-value on bounded-cosine data, error by band
- [ ] P-15 MAC: Sym+NaNAtMin weights × Asym activations; arm 1 unmasked (anti-vacuity), arm 2
      any-bit NaN test + blend + i32 accumulator; equal to the no-NaN reference bit for bit on
      NaN-free rows; NaN-free VPMADDUBSW saturation counterexample
- [ ] P-16 spoFC: UnitFull×{Floor, TiesEven} vs UnitOdd vs UnitMidrise over the evidence grid
      (m in 0..=4096, k in {1, 2, 5}); semantic invariants first (confidence < 255, exact f = 0/1,
      gate selection F_MIN 153 / C_MIN 230), bias second; midrise rejected if it loses exact
      endpoints with no error gain

### Phase 4: facades (separate PRs)
- [ ] `lance-graph-onnx`: per flavour, all 256 codes, `bounded::tanh_f32(decode(c))` vs the
      certified table; ONNX reference dequant vs `decode`; Sym+NaNAtMin -128 as a stated divergence
- [ ] `lance-graph-tflite`: `pub use` only; conformance against a C++ oracle built from TFLM
      reference headers (`8f6596d`), exhaustive 256 (int8) / 65536 (int16) per (scale, zp)

## 6. Open for the operator

1. Rounding for the native value lane: TiesEven (proposed) or TiesAway.
2. spoFC stays Floor until P-16 (proposed yes).
3. A zero-filled U8 NaNAtMin buffer reads as all-NaN (fail-closed): accept or not.
4. Certification tool: mpmath offline or `rug`/MPFR.
5. ±inf under Sym+NaNAtMin: saturate to ±127 (proposed) or NaN.

## 7. Superseded (named once, not carried)

- runtime mode enum (`I8CodeSpace`, `Rounding` as values) and the per-slice selector;
- separate `SymNaN` code space (now `Sym` + `NaNAtMin`);
- runtime calibration and runtime table build in production; the runtime certifier;
- "U8 deferred" (U8 is in scope for spoFC and codebooks);
- PR #370's per-mode functions with runtime scale returning `Result<Vec<_>>`.

## 8. Not searched / not run

- TFLM per-target kernels (`cmsis_nn/`, `xtensa/`, `ceva/`) and the gemmlowp fixed-point tanh body.
- No real-model data (manifest item D in the 2026-10-24 report is NOT RUN).
- No ONNX Runtime comparison.
