# Plan: integrating the wide / pulp "fruits" — AVX-512 first, AVX2 second (v1)

Source: `.claude/knowledge/simd-crate-comparison-wide-pulp-2026-11-17.md` (inventory,
tech debt TD1–TD10, bf16 § 2b). This plan orders the ideas taken from wide and pulp by
**signal**: how directly each one removes a measured or source-read cost on AVX-512
(primary) and AVX2 (secondary), divided by its cost. Nothing here has been run yet; every item
names the witness that will grade it.

## Ground rules (apply to every item)

- **W1a consumer contract** (`.claude/knowledge/vertical-simd-consumer-contract.md`):
  methods on the typed wrappers. AVX-512, AVX2 (+ the AVX arm where it gates), NEON and
  scalar all ship together. Each item has a parity test and documents its
  saturating/NaN semantics with a `///` example.
- **FACET STRIDE**: no shuffle-as-compute (R1). No scalar lane loop outside
  `simd_scalar.rs` (R2); on AVX2 a 512-bit op is two 256-bit halves. No `as i8` on
  unsigned data (R3). `facet-stride-guard.sh --diff origin/master` must stay clean.
- **Tier is read, not assumed**: every number carries `avx512f=`/`avx2=` from the
  arm's own header. AVX-512 is pinned with `--config .cargo/config-v4.toml`, AVX2 with
  `--config .cargo/config-v3.toml`, both under `env -u RUSTFLAGS`.
- **One chunk = one file, one deliverable** (`agent-output-durability.md`). Workers
  edit; the orchestrator compiles, runs `masking-parity.sh` (native, v3-pinned,
  avx-qemu, neon-qemu, wasm) and `codegen-witness.sh avx512` / `avx2`.

## Priority list

### P0 — the 256-bit halves and the mask (unlocks everything else)

**F1. Finish `F32x8` and `F64x4` on every stable arm.** *(TD3; from wide)*
Today `F32x8` has `splat/from_slice/from_array/to_array/copy_to_slice/reduce_sum/abs`
(`simd_avx512.rs:2940`) and `F64x4` adds only `mul_add` (`:3078`). Add the `F64x8` set:
`simd_eq/ne/lt/le/gt/ge`, `simd_min/max/clamp`, `mul_add`, `sqrt`, `round`, `floor`,
`reduce_min/max`, `to_bits/from_bits`.
- AVX-512 (VL): `_mm256_cmp_p[sd]_mask` gives a `__mmask8` directly.
- AVX2: `_mm256_cmp_p[sd]` returns a vector mask (see F2).
- Why first: AVX2 `F64x8`/`F32x16` are built from these halves (`simd_avx2.rs:9`). Once
  the halves have vector compares, TD1's scalar loops (`simd_avx2.rs:644, 982, 1153`)
  become two compare instructions with nothing scalar in between.
- Closes the gap with the nightly backend: `F64Mask4` stops being nightly-only
  (`simd_nightly/masks.rs:148`).
- Witness: `F64x4`/`F32x8` parity tests and masking-parity on all arms; `codegen-witness avx2`
  shows `vcmppd`/`vcmpps` and no `vucomisd` in `F64x8::simd_le`.

> ⊘ **REVISED 2026-11-17 (operator: "the breaking changes are absolutely unacceptable").**
> The landed F2 first changed the AVX2 `F32Mask16`/`F64Mask8` from `pub u16`/`pub u8` to
> private lane-mask pairs, and switched AVX2 `simd_min`/`simd_max`/`reduce_*` from `f32::min`
> to `vminps` NaN semantics. Both were API/behaviour breaks and are reverted: the AVX2 masks
> keep their public bitmask field exactly as on master; compares are computed per 256-bit half
> (`vcmpp*` + `vmovmskp*`, no lane loop) and `select` re-expands with `F32Mask8::from_bitmask`
> + `vblendvp*`; min/max keep `f32::min` NaN semantics via `vminp*` + unordered-compare blend.
> **Rule from this: no public field, signature or observable result of an existing API may change
> in this plan — additions only.** The open question below is closed: answer "no".

**F2. Mask types that match each tier.** *(TD1–TD2; from pulp `m64s`, wide)*
- AVX-512: masks stay `__mmask8`/`__mmask16` (unchanged).
- AVX2: `F64Mask4(__m256d)`, `F32Mask8(__m256)`, `F64Mask8([__m256d; 2])` and
  `F32Mask16([__m256; 2])`, all full-width lane masks.
- Public API stays the same: `select(t, f)` becomes `vblendvpd` on AVX2 and
  `vblendmpd k` on AVX-512. `to_bitmask()` stays as the shared way out (`vmovmskpd` on
  AVX2, a plain move from the mask register on AVX-512). New: `from_bitmask(u8)`, which
  existing callers that build a mask from bits need.
- Decision for the operator: whether changing the AVX2 mask's representation is
  acceptable. The field is `pub` today (`F64Mask8(pub u8)`, `simd_avx2.rs:1145`).
  Recommendation: make it private and route every reader through `to_bitmask()`.
- Witness: the Mandelbrot `ndarray` arm (AdaWorldAPI/mandelbrot-simd) on the pinned v3
  build. Prediction P3 says ndarray is ≥2× slower than `wide`/`pulp` **before** F1+F2;
  after them it should be within 15% of `pulp` V3. A miss says the mask conversions
  still cost something.

**F3. `any()` / `all()` / `none()` plus `&`, `|`, `!` on every mask.** *(TD5; from wide)*
- AVX-512: `kortestw`/`kortestb`, and `kand`/`kor`/`knot`.
- AVX2: `vmovmskp[sd]` + `test`, or `vptest`, and `vandp[sd]`/`vorp[sd]`/`vxorp[sd]`.
- Trivial, but every early-exit loop (Mandelbrot, cascade, masking ops) currently writes
  `to_bitmask() == 0`.
- Witness: parity test; codegen shows `kortest` (v4) and `vmovmskpd`/`vtestpd` (v3).

### P1 — tails, classification, integer width

**F4. Masked tail loads and stores on every type.** *(TD8; from pulp `partial_load/store`, `MemMask`)*
`load_partial(&[T]) -> Self` (zero-fills the missing lanes) and
`store_partial(self, &mut [T])`.
- AVX-512: `_mm512_maskz_loadu_*` / `_mm512_mask_storeu_*`, with the mask built by
  `bzhi(!0, len)`. Byte and word types use AVX-512BW.
- AVX2, 32/64-bit lanes: `vmaskmovp[sd]` / `vpmaskmov[dq]`.
- AVX2, 8/16-bit lanes: there is no masked byte move. Copy the tail into a zeroed
  aligned buffer with `ptr::copy_nonoverlapping` (memcpy), then do one full load. That
  is a single copy, not a lane loop, so R2 is not violated.
- Fault safety: AVX-512 masked loads do not fault on masked-off lanes, so a tail can sit
  at the end of a page. Test this with a `mmap` guard page.
- Witness: the guard-page test plus a sweep of tail lengths 0..LANES on every type.

**F5. Float classification and reciprocals.** *(TD6; from wide)*
`is_nan` (`_CMP_UNORD_Q`), `is_finite`, `is_infinite`, `copysign`, `flip_signs`,
`recip`, `recip_sqrt`.
- AVX-512: `copysign` is one `vpternlogq` bit-select with a sign-bit mask operand. The immediate depends on operand order; derive it from the truth table and pin it in a test.
- AVX2: `copysign` is an `and`/`andnot`/`or` sequence.
- ⊘ **REMOVED 2026-11-17 (operator: "Every method MUST be ieee bit exact").** `recip_approx`
  (vrcp14 / vrcpps / exact by arm) returned different bits on different CPUs (1/3.0:
  `3eaaaa80` v4, `3eaaa000` v3, `3eaaaaab` scalar) — an estimate has no place in the facade.
  Only `recip` (exact IEEE `1/x`, identical on every arm) remains. The wide/pulp "fast vs exact"
  split (F12) is rejected for the same reason.
- Witness: parity tests over NaN, ±Inf, ±0 and denormals; scalar is the reference.

**F6. Widening multiply and high-half multiply.** *(TD9; from wide `widening_mul`, `mul_keep_high`)*
- `U32x16::mul_widen_even` → `U64x8`: `vpmuludq` on AVX-512, two `vpmuludq` on AVX2.
- `U16x32::mul_high`: `vpmulhuw` on both tiers.
- `I16x32::mul_high`: `vpmulhw` on both tiers.
- `U64x8::mul52_lo`/`mul52_hi`: `vpmadd52luq`/`vpmadd52huq`. AVX-512IFMA only, which
  the host has. Gated by `cfg(target_feature = "avx512ifma")`; AVX2 and scalar use an
  exact u128 path.
  ⊘ **Deferred 2026-11-17 (landing of F6):** `mul52_*` is not built. A non-IFMA
  realization is a per-lane u128 loop, which R2 forbids outside `simd_scalar.rs`,
  and an IFMA-only method would break the W1a "same API on every arm" rule. Revisit
  with a consumer that needs 52-bit limbs.
- Witness: parity test against u128 scalar products; codegen-witness.

**F7. AVX2 integer burn-down, done in order of measured cost.** *(TD4; already on the burn-down list)*
- The 13 types in `avx2_int_type!` (`simd_avx2.rs:1332-1705, 2439-2441`) are `[T; N]`
  arrays. They become pairs of `__m256i`, starting with the methods a measurement
  already blames.
- First targets: `U8x64::mask_blend`, `saturating_add` and `saturating_sub`. They are
  why D-ADD-PM costs 285 ns on AVX2 against 134 ns for the decode path (blackboard
  2026-11-15).
- Witness: re-run D-ADD-PM on v3. Target: below the 134 ns decode path.

### P2 — the AVX-512-only features this host has and no competitor uses well

> ⊘ **WITHDRAWN 2026-11-17 (operator: "I didn't ask for changing BF16, I told you where to learn
> from").** F8 is not to be executed. `BF16x16`, `F16`/`F32` + `add_mul` and the `as_chunks` call
> sites are the REFERENCE for how this facade is built, not debt (see blackboard, same date).

**F8. Fix the bf16 debt.** *(§ 2b)*
- Remove the name collision: `simd_half::BF16x16([u16;16])` becomes either an alias or a
  re-export of the hardware type on `avx512bf16`, and a scalar type elsewhere. One name
  should mean one cost.
- `add_bf16_inplace`/`mul_bf16_inplace`/`cast_*` (`simd_half.rs:245-384`) should use
  `vcvtneebf162ps` + `_mm512_cvtneps_pbh` on AVX-512.
- On AVX2: widen with `vpmovzxwd`, shift with `vpslld 16`, and narrow back with the
  certified software round-to-nearest-even. No shuffles (R1).
- Witness: the bf16 bench arms (a)–(d) and predictions P5–P7 in the knowledge doc, plus
  `examples/bf16_rne_exhaustive.rs` byte-equality.

**F9. AVX-512 FP16 (`F16x32` arithmetic).** *(from pulp coverage; host has `avx512_fp16`)*
Measure before building. Stable Rust lacks the `f16` type, so the `_mm512_*_ph`
intrinsics may not be callable. If they aren't, the same raw-byte-encoding approach that
`simd_amx.rs` uses applies. Gate: a one-file probe that adds two `F16x32` vectors and
checks bit-equality against `F16x16` through f32.
- Kill: if neither the intrinsics nor the byte-encoded route is clean, defer. f32-upcast
  through the existing `F16x16` stays the path.

**F10. Run-time dispatch per kernel, not per op.** *(from pulp `WithSimd` + `vectorize`)*
- Add `simd_runtime::with_tier(|t| ...)`: it monomorphizes the closure per tier and
  enters it through a `#[target_feature(enable = "avx512f,...")]` trampoline. The whole
  loop then inlines for the tier and pays the indirect call once, not once per op.
- Additive and opt-in (feature `runtime-dispatch`). The compile-time facade is untouched.
- Witness: Mandelbrot `ndarray` arm through `with_tier` on a v3-pinned binary running
  on this v4 host. It should match the v4-pinned build within 10%. That is what pulp's
  prediction P4 tests.

### P3 — breadth (valuable, lower signal for AVX-512)

- **F11.** f64 `exp`/`ln` (and f32 `exp2`/`log2`) as methods on `F32x16`/`F64x8`,
  moving the polynomials out of `hpc/vml.rs`. Witness: ULP sweep against libm.
- **F12.** `fast_min/fast_max` vs IEEE-correct `min/max`. First document what
  `simd_min` actually does (`vminpd`: returns the second operand on NaN), then add the
  correct variant (`vrangepd` on AVX-512DQ; cmp+blend on AVX2).
- **F13.** Complex `C32x8`/`C64x4` (`mul_add`, `conj_mul`, `abs2`) for `hpc/fft.rs`.
  Witness: FFT round-trip error and timing.
- **F14.** wasm relaxed-simd behind a feature, only for deterministic ops. Not AVX; listed
  for completeness.

## Sequencing

| wave | chunks (one file each) | gate before the next wave |
|---|---|---|
| 1 | F1 `simd_avx512.rs` (F32x8/F64x4 AVX-512 methods); F1 `simd_avx2.rs` (AVX2 bodies); F1 `simd_scalar.rs`; F1 `simd_neon.rs` | masking-parity green on all arms; guard clean |
| 2 | F2 mask types per arm (same four files) and F3 predicates | Mandelbrot v3 and v4 measured; P1/P3 graded |
| 3 | F4, F5, F6 (one per chunk) | parity, guard-page test, codegen-witness |
| 4 | F7 (U8x64 first), F8 | D-ADD-PM and bf16 bench re-run |
| 5 | F9 probe, F10 | probe verdict; dispatch witness |

## Open questions for the operator

1. F2: is the AVX2 mask representation allowed to change, i.e. make the public
   `F64Mask8(pub u8)` field private?
2. F10: may `simd_runtime` grow a kernel-level entry, or must it stay per-op?
3. Benchmarks: cargo was denied in the session that wrote this, in both checkouts. The
   waves need the orchestrator to be able to build and run.
