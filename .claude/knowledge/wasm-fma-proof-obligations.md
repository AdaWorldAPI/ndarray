# wasm `mul_add`: proof obligations

READ BY: anyone changing `F32x16::mul_add` / `F64x8::mul_add` in
`src/simd_wasm.rs`, the scalar emulation in `src/simd_fma.rs`, or the `0x11xx`
parity group in `crates/simd-masking-parity`.

Contract: every lane returns `RN(a * b + c)`, binary32 or binary64, round to
nearest, ties to even, one rounding of the exact value, subnormals included.
Any NaN result matches any NaN: payload and NaN sign are not part of the
contract.

The proofs below rest on one platform fact and three cited lemmas. Everything
else is derived here. Each item is marked **PROVEN** (derived here from the
lemmas), **CITED** (taken from the literature, not re-proven), **TESTED**
(executable witness only) or **OPEN**.

## Platform fact

WebAssembly scalar and SIMD float operations (`f64x2.add/sub/mul`,
`f32x4.demote_f64x2_zero`, `f64x2.promote_low_f32x4`) are IEEE 754 operations
with round to nearest, ties to even and gradual underflow. There is no flush to
zero, no trap and no status flag. Only NaN bit patterns are nondeterministic.
Rust emits no `contract` or reassociation flags, so `x * y + z` stays two
operations.

`+relaxed-simd` does not change this code. No source path is selected on it,
and the code section of the parity binary is byte-identical with and without
it. A byte scan for relaxed-SIMD opcodes (`0xFD`, sub-opcode `0x100..0x113`)
finds 0 hits there, and 1 hit in a control that calls `f64x2_relaxed_madd`
(blackboard 2026-10-27, audit entry). **TESTED** for the current compiler;
`scripts/masking-parity.sh wasm-relaxed` re-checks it in CI.

## Lemmas

- **L1 (Veltkamp split)** — **CITED**. With `C = 2^27 + 1`,
  `t = C*x; hi = t - (t - x); lo = x - hi` gives `hi + lo == x` exactly, `hi` and
  `lo` each fit in 26 bits, provided `C*x` does not overflow. Underflow does not
  break it. (Dekker 1971; Boldo, "Pitfalls of a full floating-point proof",
  IJCAR 2006.)
- **L2 (Dekker product)** — **CITED**. With L1 splits of `a` and `b`,
  `ph = RN(a*b)` and `pl = (((ah*bh - ph) + ah*bl) + al*bh) + al*bl` satisfy
  `ph + pl == a*b` exactly, provided nothing overflows and
  `e_a + e_b >= e_min + p - 1 = -970` (`e_x` the binary exponent of `x`, `x` in
  `[2^e, 2^(e+1))`). (Same references.)
- **L3 (TwoSum)** — **CITED**. `s = x + y; yy = s - x;
  e = (x - (s - yy)) + (y - yy)` gives `s + e == x + y` exactly for finite
  `x, y` when nothing overflows, underflow included. (Knuth, TAOCP vol. 2.)
- **L4 (round to odd)** — **PROVEN** below. Let `w` be real and `v` the binary64
  value obtained by `add_round_to_odd`, at grid spacing `g` (the ulp at `v`, or
  `2^-1074`). If `w` is representable, `v == w`. Otherwise `v` is the odd
  multiple of `g` with `|w - v| < g`, so `w` and `v` lie strictly between the
  same two consecutive even multiples of `g`.

  Proof. `add_round_to_odd(x, y)` takes `s = RN(x + y)` and the exact error `e`
  (L3). If `e == 0`, `x + y == s` is representable. Otherwise, if `s` is odd it
  is already the odd neighbour. If `s` is even, the code steps its bit pattern
  one place toward `x + y` (down in magnitude when `e` and `s` differ in sign).
  That gives the adjacent value on the side of the exact sum, which is odd;
  across a power of two it is the largest value of the lower binade, which is
  odd on the finer grid there. In every case the exact sum lies strictly
  between `v - g` and `v + g`.
- **L5 (rounding through odd)** — **PROVEN**. Let `x = T + w`, where `T` lies
  on a grid whose spacing is at least `8g`. Then
  `RN_target(T + v) == RN_target(x)` for any target format whose rounding
  boundaries (midpoints and the overflow threshold) near `x` are multiples of
  `2g`.

  Proof. If `w` is representable, `T + v == x`. Otherwise every rounding
  boundary `M` satisfies: `M - T` is an even multiple of `g`. By L4, `w` and
  `v` lie strictly between the same two even multiples of `g`. So `x` and
  `T + v` lie strictly on the same side of every boundary, and neither of them
  is one. Hence both round to the same value.

## F32x16 (no guard)

- **Product exact** — **PROVEN**. A binary32 significand has 24 bits, so the
  product has at most 48 bits (53 available). Its exponent is in `[-298, 256]`,
  inside the binary64 normal range.
- **Sum** — **PROVEN**. `|a*b| <= 2^256` and `|c| < 2^128`, so the binary64
  sum cannot overflow and L3 applies.
- **Final rounding to binary32** — **PROVEN** by L5. Take `T = 0` and `w` the
  exact sum. Every binary32 rounding boundary, including the subnormal
  midpoints and the overflow threshold `2^128 - 2^103`, is a multiple of
  `2^-150`. The binary64 grid at the value is at most `2^-52` times its
  magnitude, which is at least 29 bits finer than binary32. That holds down to
  the binary32 subnormal grid, since binary64 is still normal there.
- **Special values** — **PROVEN**. An infinite or NaN product, or an infinite
  `c`, makes `s` non-finite. The step is then masked off, and `demote` returns
  the IEEE infinity or NaN. Signed zeros: when `a*b + c` is exactly zero,
  `e == 0`, so `s` is the IEEE-signed `RN` sum and is passed through unchanged.

## F64x8 vector path

Guard: `2^-460 <= |a|, |b| <= 2^460` and `|c| <= 2^960`, required on BOTH lanes
of a v128 pair. The guard excludes zero, infinite and NaN `a`, `b`, and
infinite or NaN `c`.

- **No overflow anywhere** — **PROVEN**. `|C*x| < 2^488`; `|a*b| <= 2^920`;
  `|c + ph| <= 2^960 + 2^920`. Every partial product in L2 is at most `|a*b|`.
- **L2 applies** — **PROVEN**. `e_a + e_b >= -920 >= -970`. This is 50 binades
  of product exponent above what L2 needs (for example one operand at
  `2^-460` and the other at `2^-510`).
- **TwoSum `(th, tl)` of `c + ph`** — **PROVEN** by L3 (finite, no overflow).
- **`tl != 0` implies `|th| >= 2^-922`** — **PROVEN**. If `c + ph` is inexact,
  then `c` and `ph` are not opposite-signed within a factor of 2 (Sterbenz).
  So `|c + ph| >= |ph| / 2 >= 2^-921`, and also `|ph| <= 2 |c + ph|`.
- **Final rounding** — **PROVEN**. Let `x = c + a*b = th + (tl + pl)`, and let
  `v` be the round-to-odd of `w = tl + pl` (L4).
  - Case `tl == 0`: `w = pl` is representable (L2), so `v == pl` and
    `th + v == x` exactly. The final add then rounds once. This covers the
    near-zero cancellations: an exact zero, and a subnormal or tiny result.
  - Case `tl != 0`: we have `|tl| <= ulp(th)/2` and `|pl| <= ulp(ph)/2 <= 2 ulp(th)`.
    So `|w| <= 2.5 ulp(th)`, and `g <= max(2^-50 ulp(th), 2^-1074) <= ulp(th)/8`,
    because `ulp(th) >= 2^-974`. Binary64 midpoints near `x` are multiples of
    `ulp(th)/4` (`ulp(th)/2`, or `ulp(th)/4` just below a power of two), so
    they are multiples of `2g`. L5 with `T = th` gives `RN(th + v) == RN(x)`.
    The result is at least `2^-922` in magnitude, so it is never subnormal.
- **Special-value repair** — **PROVEN**. If `a` or `b` is `±0`, `±inf` or NaN,
  the binary64 product is exactly the IEEE product: `±0`, `±inf`, or NaN for
  `0 * inf`. `a*b + c` is then one rounding of the exact sum, which is IEEE
  `fusedMultiplyAdd`: invalid gives NaN, the zero-sum sign follows the
  round-to-nearest rule, and a nonzero `c` is returned exactly. It is the same
  identity `simd_fma::fma_f64` uses first.
- **Bit-select** — **PROVEN**. The masks come from `f64x2` comparisons, so each
  64-bit lane is all ones or all zeros. They stay so through `v128_and/or/not`.
  `v128_bitselect` therefore returns, per lane, exactly one of the two computed
  lane patterns. Both candidate computations are lane-wise: the kernels contain
  no shuffle, swizzle or lane extract (Haiku census, verified). WebAssembly has
  no float exceptions, so the NaN or garbage computed on a discarded lane has no
  effect on the kept lane.
- **Mixed pairs** — **PROVEN**. The dispatcher picks one path per pair. The
  vector path is entered only when every lane is in the guard domain or has an
  exact product; all other pairs go lane by lane to `simd_fma::fma_f64`.

## Margins: proven versus measured

- **Proven sufficient conditions:** `e_a + e_b >= -970` for L2 (the guard gives
  `-920`); `|C*x|` finite (`|x| < ~2^996`); `c + ph` not overflowing. Given the
  guard on `a` and `b`, a finite `c` cannot make `c + ph` overflow, so the
  `|c| <= 2^960` bound is not needed for exactness. It is kept because it
  excludes infinite and NaN `c`, which TwoSum turns into NaN.
- **Measured** (scalar copy of the unguarded emulation vs hardware FMA,
  200,000 samples per point): failures start near products of `2^-1000`
  (operands near `2^-497`), and every case fails at `|a| ~ 2^1000`. This is
  evidence about where the algorithm breaks, not part of the proof. The guard
  is not to be widened on its strength.

## Still open

- **OPEN:** no mechanized proof. L1–L3 are cited, not re-derived; L4, L5 and
  the case analysis above are hand proofs.
- **OPEN:** `simd_fma::fma_f64` (the integer lane fallback) and
  `simd_fma::fma_f32` are certified only by tests against the 4608-bit integer
  oracle and against hardware FMA. `fma_f32` uses the same argument as the
  F32x16 path above.
- **TESTED, not proven:** that each realization's compiler output matches the
  source semantics. This rests on the parity arms (wasm, wasm-scalar,
  wasm-relaxed, native, NEON, AVX, v2, nightly). Sampling covers no input space
  exhaustively.
