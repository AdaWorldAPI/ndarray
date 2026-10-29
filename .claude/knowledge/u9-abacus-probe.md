# D-U9-ABACUS: schoolbook U9 x U9 with carry escalation

Two different experiments, with two different verdicts. Do not read one
for the other:

- **C, bit-plane multiplication** (`abacus.rs`): the product itself rebuilt
  from binary partial products, compressed with carry-save logic.
  **Refused** as a replacement for B.
- **D, the 64K carry boundary** (`carry16.rs`): the ordinary U8 x U8 -> U16
  product kept, added to a wrapping 16-bit word, the carry escalated into a
  high tally. **Adopted (lab) as the exact replacement for A's f32 path**;
  it beats A' on AVX-512 and loses to B only because of its multiply.
- **E, the anchor at 128** (`anchor.rs`): B's own split `|b| = low7 + 128 *
  bit7`, with the anchor term a masked sum of `a` shifted by 7 instead of a
  byte-dot against -128. **Ties B on AVX-512; no gain worth adopting.**

READ BY: savant-architect, vector-synthesis, simd-savant, the TurboVec-U9
owner. Follows `turbovec9-u9-head-to-head.md` (#388), whose B kernel and
`i64` oracle this probe reuses unchanged.

Probe: `examples/u9_abacus/` (`abacus.rs` holds the arithmetic, `main.rs`
times it against #388's B). Correctness: `tests/u9_abacus.rs`. Plain Rust bit
operations and facade words only; no backend, facade or storage change.

```sh
cargo run --release --example u9_abacus --features std
cargo test --release --test u9_abacus
```

## Candidate E: anchored at the power of two inside the operand

No U9 operand reaches 64K, but every one has a known power-of-two anchor:
bit 7. B already splits there; E keeps the multiply only for the residual:

`a * |b| = a * low7 + (sum of a over lanes with bit 7 of b set) << 7`

The anchor term is a masked sum, weighted by 128 once, at flush. Three ways
to form it from existing facade words:

| | per block | anchor term |
|---|---|---|
| B (#388) | 4 masked byte-dots | `a` x `top` (-128), a full byte-dot |
| E4 | 4 masked byte-dots | `a` x all-ones bytes, masked by sign AND bit 7, one per sign |
| E3 | 3 masked byte-dots | `a` x a `+1/-1` byte vector (from the sign plane via `mask_blend`), masked by bit 7 |
| Esum | 2 masked byte-dots + 2 `mask_blend` + 2 `sum_bytes_u64` | masked byte sums, reduced horizontally every block |

Measured (ns per 64 values, 3 runs; L1 / streamed):

| | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| B end-to-end | 2.89 - 2.91 / 17.1 - 17.5 | 5.63 - 7.94 / 18.8 - 21.0 | 10.03 - 10.05 / 22.2 - 23.4 |
| **E3 end-to-end** | **2.92** / 17.1 - 17.7 | **5.52 - 7.81** / 19.0 - 20.9 | 40.2 - 43.9 / 219 - 220 |
| E4 end-to-end | 3.56 - 3.58 / 18.1 - 19.7 | 6.47 - 9.05 / 20.1 - 22.0 | 14.2 - 14.5 / 24.3 - 25.8 |
| Esum end-to-end | 4.42 - 4.44 / 16.9 - 20.6 | 5.95 - 8.31 / 19.4 - 21.3 | 59.8 - 65.9 / 276 - 288 |
| B arithmetic only | 2.39 / 18.8 - 22.2 | 5.02 - 7.09 / 26.1 - 27.2 | 9.00 - 9.17 / 31.4 - 36.9 |
| **E3 arithmetic only** | **2.28** / 14.8 - 18.4 | **4.95 - 6.88** / 22.4 - 24.5 | 40.9 - 45.8 / 214 - 217 |
| E4 arithmetic only | 2.72 - 2.73 / 15.8 - 19.9 | 5.81 - 7.89 / 22.9 - 24.1 | 12.9 / 32.4 - 33.5 |
| B scan | 2.64 / 11.4 - 12.8 | 4.84 - 6.94 / 14.6 - 16.7 | 9.11 - 9.19 / 19.6 - 27.2 |
| E3 scan | 2.66 - 2.67 / 10.6 - 12.8 | 5.22 - 7.08 / 13.0 - 15.2 | 36.9 - 43.7 / 216 - 219 |

(The third AVX-512-without-VNNI run was slower for every row alike; the
first two are 5.63 / 6.06 for B and 5.52 / 5.52 for E3.)

Reading it:

- **E3 saves one byte-dot and pays it back building the sign vector.** On
  VNNI it ties B end to end (2.92 vs 2.89 - 2.91) and in the scan, and is
  about 5 % faster in arithmetic only (2.28 vs 2.39); without VNNI it is
  about 2 % faster end to end and slower in the scan. Disassembly: 3
  `vpdpbusd` against B's 4, plus the `kmovq` and blend that make `+1/-1`.
- **E4 is slower everywhere.** A byte-dot against constant ones costs the
  same instruction as one against -128; the extra mask ANDs are pure cost.
- **Esum is slower everywhere.** `sum_bytes_u64` reduces horizontally on
  every call.
- **On AVX2, E3 and Esum lose to `mask_blend`'s per-lane realization**
  (40 - 66 ns); E4, which needs no blend, is 1.4x B.

Verdict for E: **no adoption.** The anchor idea is arithmetically right and
exact, but B already turns the anchor into one hardware instruction; the
saving is one byte-dot out of four, and every facade route to the masked sum
costs about that much. Two facade gaps would change it, both recorded and not
added:

1. **A vertical masked byte sum** (`vpsadbw` into a `U64x8` accumulator,
   reduced once at flush): the anchor term per block would be one SAD and
   one add instead of a byte-dot.
2. **A sign plane to `+1/-1` bytes** that does not go through `mask_blend`
   (which is a per-lane loop on AVX2).

Correctness: the split `|b| = low7 + 128 * bit7` for every magnitude and the
`+1/-1` sign bytes for a mixed plane; all 511 x 511 pairs through E4, E3,
Esum and all three scans; full blocks; all-255 runs past the flush bound.
Passed on all seven tiers. Disable runs, each red: anchor weighted `<< 6`;
E4's anchor ignoring bit 7; sign bytes all +1; Esum's blend inverted; `low7`
keeping bit 7.

Also found while checking E: on AVX2 the facade's masked byte-dot itself
contains 4 `vpshufb` (LLVM spreads the keep-mask bytes with them), so #388's
"no shuffle in B" holds on AVX-512 only. Corrected in #388's report.

## Verdict for C (bit-plane multiplication only)

- **`carry_save3`: ADOPT-LAB.** An exact, five-operation full adder over 64
  independent bit columns; LLVM lowers it to `xor`/`and` in general registers.
- **Bit-plane multiplication as a replacement for B: REFUSE.** No C variant
  comes close.
  The best one (one popcount per partial-product plane, no carry-save) is
  5.2x B's arithmetic on AVX-512 VNNI with both operands' planes prepared for
  free, and the carry-save variants are about 120x. There is no break-even
  for prepared-operand reuse: with planes given at no cost, C is still slower
  than B paying all of its own preparation.
- **Deferring C's carry-save normalization: not a gain.** Normalizing once
  at the end is no faster than normalizing every block, and slower when
  streamed. This says nothing about D, whose carry is resolved on every add.

The arithmetic is exact in every variant; the loss is purely in instruction
count and in what LLVM makes of the carry-save state.

## Candidate D: the 16-bit place-value column (the narrower experiment)

The question after the bit-plane result: keep #388's U8 multiplication as it
is (A': `U16x32::mullo`, sign routed by the polarity XOR) and replace only its
`f32` detour with exact positional accumulation. A product is below 2^16, so
it fits a `u16` lane. Each tally lane keeps `low` (wrapping `u16`) and `high`
(units of 2^16); the value is `low + 2^16 * high`. Crossing 2^16 is one carry
into `high`. 67,025 is `1,489 + 1 * 65,536`: the low word wraps, one unit
moves up. Nothing is widened before it must be. (`carry16.rs`.)

Two ways to find the carry, both plain bit logic on `U16x32` lanes:

- **react** (`carry_after`): add first, then read the carry out of the sum,
  `msb((low & p) | ((low | p) & !sum))`;
- **anticipate** (`carry_before`): before adding, the carry is set exactly
  when `p` exceeds the headroom `!low = 65535 - low`, the borrow out of
  `!low - p`: `msb((low & p) | (!(!low ^ p) & (!low - p)))`.

`high` gains at most one unit per block per lane, so it is flushed into the
`i64` total every 65,535 blocks. A flush moves `low + 2^16 * high` per lane.
Carries are counted, never XORed together: two carries are two units.

### What LLVM made of it

| | AVX-512 (VNNI and v4 alike) | AVX2 |
|---|---|---|
| react, per block (4 tallies x 32 lanes) | 53 insns: 8 `vpternlogq`, 8 `vpaddw`, 4 `vpsrlw`, 4 `vpmullw`, 4 `vpmovzxbw`, 2 `vextracti64x4` | 179 insns, incl. 16 `vmovq`, 6 `vpblendd` |
| anticipate, per block | 61 insns: 12 `vpternlogq`, 4 `vpsubw`, otherwise as react | 107 insns: `vpand`/`vpandn`/`vpor`/`vpxor` |

On AVX-512 LLVM folds each carry formula into ternary logic: two
`vpternlogq` per carry (react) or three (anticipate). That is the XOR/AND
carry written in Rust and lowered by the compiler, with no facade change.
There is no shuffle, permute, pack or float instruction. The `vpmovzxbw` and
`vextracti64x4` are the u8 -> u16 widening that `mullo` needs, the same as
in A'.

### Measured (ns per 64 values, 3 runs, L1 / streamed)

| | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| A' arithmetic only (f32 accumulate) | 6.33 - 6.56 / 26.0 - 26.6 | 6.33 - 6.39 / 26.2 - 26.9 | 10.7 - 12.0 / 29.5 - 32.1 |
| **D react arithmetic only** | **5.36 - 5.98** / 15.1 - 21.7 | **5.77 - 5.97** / 17.3 - 19.0 | 21.2 - 29.5 / 27.7 - 28.6 |
| D anticipate arithmetic only | 6.03 - 6.86 / 16.2 - 22.3 | 6.23 - 6.66 / 19.2 - 20.3 | 10.0 - 16.9 / 18.5 - 21.1 |
| B arithmetic only (#388) | 2.39 - 2.81 / 19.8 - 21.5 | 5.02 - 7.28 / 25.5 - 29.4 | 9.17 / 31.8 - 33.2 |
| A' end-to-end | 7.63 - 7.80 / 21.6 - 23.0 | 7.80 - 7.81 / 22.2 - 24.0 | 73.8 - 80.4 / 419 - 436 |
| **D react end-to-end** | **7.05 - 7.09** / 21.5 - 22.5 | **7.05 - 7.45** / 22.1 - 24.8 | 84.0 - 91.3 / 437 - 444 |
| D anticipate end-to-end | 7.81 / 21.6 - 23.8 | 7.78 - 8.20 / 22.5 - 23.0 | 76.9 - 104.8 / 428 - 435 |
| B end-to-end (#388) | 2.89 - 2.91 / 14.8 - 17.7 | 5.63 - 7.59 / 19.6 - 20.6 | 10.03 - 10.06 / 22.6 - 24.4 |
| D react scan, query widened once | 6.17 - 6.81 / 16.6 - 17.1 | 6.39 - 6.97 / 16.7 - 19.0 | 102 - 128 / 441 |
| D anticipate scan | 7.17 - 7.91 / 13.7 - 17.7 | 7.16 - 7.86 / 18.7 - 21.2 | 68.5 - 112.9 / 422 - 439 |
| B scan (#388) | 2.64 / 11.3 - 12.5 | 4.70 - 4.83 / 14.3 - 15.1 | 9.17 - 9.19 / 19.9 - 28.5 |

### Reading it

- **The carry column beats the f32 detour.** On AVX-512, react is about 10 %
  faster than A' in arithmetic (5.4 - 6.0 vs 6.3 - 6.6 ns in L1) and in the
  full path (7.05 vs 7.6 - 7.8). Streamed, it is clearly faster
  (15 - 22 vs 26 ns). Anticipate costs one more ternary op per carry and
  lands level with A'.
- **React vs anticipate depends on the tier.** On AVX-512, react wins (the
  carry needs the sum anyway, and two ternary ops suffice). On AVX2,
  anticipate wins (10 - 17 vs 21 - 30 ns); the react form gets 16 `vmovq`
  and 6 `vpblendd` from LLVM there.
- **B still wins, and the reason is not the accumulation.** D removes the
  float conversion, but it still widens u8 -> u16 (`vpmovzxbw`,
  `vextracti64x4`) and runs four `vpmullw` per block before any add. B
  multiplies straight from bytes and accumulates in `i32` in the same
  instruction. On VNNI, B's arithmetic is 2.4 ns against D's 5.4.
- **Sign-routing overhead** (end-to-end minus arithmetic-only medians, so it
  also contains `check` and widening `|b|`): D react on VNNI about 1.1 -
  1.7 ns, on AVX-512 without VNNI 1.3 - 1.5 ns, on AVX2 about 55 - 70 ns.
  A' carries the same routing (1.1 - 1.4 ns on VNNI, 63 - 70 ns on AVX2).
  B routes with the byte-dot's own mask instead (0.5 ns on VNNI).
- **On AVX2 the end-to-end and scan rows are dominated by routing.** D's
  routing is A's `U8x64::mask_blend`, whose AVX2 realization is a per-lane
  loop (74 - 128 ns). That cost has nothing to do with the carry.

Verdict for D: **ADOPT-LAB as the exact replacement for A's f32 detour**, and
the right way to accumulate any U16 product stream (the compiler gives
ternary logic for free on AVX-512). **DEFER as a B replacement**: it would
need the multiply to stay in bytes, which is what B already does.

Correctness: the carry rules against `u32` arithmetic for every `low` in
0..=65535, with `p` at, just above and just below the headroom, plus 0,
65,025 and random products; all 511 x 511 pairs; full blocks; all-255 runs of
131,075 blocks past the flush bound, with `high` near its limit. Passed on all
seven tiers. Disable runs, each red: react without the `(low | p) & !sum`
term; anticipate without the borrow term; carry not added to `high`; flush
at 70,000 blocks (`high` wraps); `high` weighted 2^15.

## The arithmetic

For magnitudes `a = sum a_i 2^i` and `b = sum b_j 2^j`,
`a * b = sum over i, j of (a_i AND b_j) * 2^(i + j)`. A block of 64 lanes is
read as eight bit planes (`planes`): bit `k` of `plane[i]` is bit `i` of
lane `k`. One AND of two planes gives one partial-product bit for all 64
lanes at significance `i + j`. The sign is `negative = plane_a XOR plane_b`;
it only selects the positive or the negative tally. No magnitude is negated.

`carry_save3(a, b, c)` returns `(sum, carry)` with, for every bit position
`k`, `a_k + b_k + c_k = sum_k + 2 * carry_k`. `sum` stays in its column,
`carry` belongs to the next; nothing is shifted across bit positions.

Three folds of the 128 partial-product planes per block (64 per sign):

| | fold | normalization |
|---|---|---|
| C popcnt | `popcount(plane) << (i + j)` into an `i64`, at once | every block (it is the fold) |
| C abacus | `Abacus`: at most two planes per column; a third compresses via `carry_save3`, the carry rippling up | once, at `finish` |
| C csa | one running plane per column and sign, folded on a fixed, data-independent schedule; carries leaving column 15 counted into `spill` | once, at `finish` |

## Correctness

`tests/u9_abacus.rs`, against #388's `i64` oracle and its B kernel:

- `carry_save3` bit by bit for all eight input combinations at every bit
  position, plus 10,000 random word triples (popcount identity);
- `planes` recovers every magnitude 0..=255 in every lane;
- carries through many columns: 2^20 all-ones planes in column 0
  (64 * 2^20 exactly), one all-ones plane per column 0..=39 three times, and
  5,000 mixed-pattern planes;
- all 511 x 511 pairs through C popcnt, C abacus, C csa, the C scan and B;
- full mixed blocks (1 to 300 blocks);
- all-255 runs of 49,157 blocks with all sign mixes, totals beyond `i32`;
- never-touched refused by `check` in either operand; touched zero scores 0.

All five folds equal the oracle in every case. Passed on native AVX-512 VNNI,
config-v4, config-v3, config-avx (qemu), config-v2 (qemu), aarch64 (qemu) and
nightly-simd. Disable runs, each red: `carry_save3` without `t & c`; the
abacus dropping its carry; carry kept in the same column; a plane extracted
from the wrong bit; the abacus sign routing swapped; the popcount weight
`i * j` instead of `i + j`; `spill` weighted `2^15`; the static schedule
dropping an odd plane; carries not forwarded to the next column.

Exactness domain: the 511 numeric values, any number of blocks while the
`i64` total stays in range (about 2.2 * 10^12 worst-case blocks). The
abacus's column bound is by value: column `w` can only hold a set bit for a
lane whose total is at least `2^w`, so 64 columns hold any per-lane total
below 2^64; `Abacus::add` panics rather than drop a carry past them.

## What LLVM made of it (`objdump` of `#[inline(never)]` copies)

| function | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| `carry_save3` | 3 `xor`, 2 `and`, 2 `mov`: scalar, no ternary logic | same | same |
| `planes` | 6 `vptestmb` + 2 `vpmovb2m` (+ `vpsllw` for bit 0): one instruction per plane | same | out of line (a `jmp`); extraction 8.6 ns vs 2.5 |
| C popcnt `mac` (140 / 196 / 216 insns) | vectorized: 32 `vpopcntq`, `vpand`, `vpsllvq` | **32 `vpshufb`** nibble-table popcount + `vpsadbw` | **32 `vpshufb`** + `vpsadbw` |
| C abacus `mac` (110 insns) | branchy loop over the column counters, columns in memory; scalar | same | same |
| C csa `mac` (437-488 insns) | not unrolled: loops over stack arrays, 122 `mov`, 37 `cmp`; scalar logic, no `vpternlog` | same | same |

Three findings:

1. LLVM lowers the plane extraction as well as possible and vectorizes the
   per-plane popcounts. It does not turn either carry-save fold into
   straight-line or vector code, even though the static schedule depends
   only on loop indices. Writing it out by hand would be manual scheduling,
   which this probe was told not to do.
2. **Without VPOPCNTDQ the popcount fold is a byte shuffle.** On v4 and AVX2,
   `u64::count_ones` lowers to the `vpshufb` nibble table, the shape the
   repository's no-shuffle rule forbids in hand-written code. Only the
   AVX-512 VNNI host (which has VPOPCNTDQ) avoids it. B uses no shuffle on
   any tier (#388).
3. No variant needed a facade word the facade lacks: `planes` is `BitAnd` +
   `cmpeq_mask`, and LLVM fused them into `vptestmb`.

## Measurements (ns per 64 values, medians of 3 runs, this host)

One AVX-512 VNNI Xeon; each tier a separate native build (`config-v4`,
`config-v3`). L1 = 64 blocks per operand; streamed = 2^20 blocks (72 MiB).
Every kernel equals the oracle on the benchmark data before it is timed.

| L1-resident | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| B end-to-end (#388) | 2.89 - 2.91 | 5.63 | 10.03 - 10.05 |
| C popcnt end-to-end | 28.1 - 28.8 | 46.9 - 47.8 | 42.4 - 43.4 |
| C abacus end-to-end | 294 - 297 | 288 - 293 | 309 - 319 |
| C csa end-to-end | 301 - 302 | 301 - 326 | 311 - 438 |
| B arithmetic only | 2.41 - 2.42 | 5.02 - 6.13 | 9.17 - 12.2 |
| C popcnt arithmetic only, planes given | 12.42 | 30.0 - 30.8 | 30.7 - 31.3 |
| C abacus arithmetic only, deferred | 291 - 296 | 288 - 319 | 292 - 314 |
| C abacus arithmetic only, normalized every block | 268 - 273 | 282 - 287 | 272 - 356 |
| C csa arithmetic only | 292.5 - 292.7 | 290.5 - 290.7 | 290.1 - 290.9 |
| C plane extraction, both operands (cold) | 2.52 | 2.59 | 8.6 - 8.9 |
| B scan (#388) | 2.64 - 2.66 | 4.70 - 4.83 | 8.98 - 9.19 |
| C popcnt scan, query planes once | 23.3 - 23.7 | 47.0 - 48.0 | 41.7 - 42.7 |
| C abacus scan | 282 - 289 | 293 - 297 | 300 - 302 |
| C csa scan | 302 - 303 | 301 - 319 | 300 - 311 |

| Streamed | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| B end-to-end | 18.2 - 19.3 | 17.9 - 21.1 | 20.6 - 24.0 |
| C popcnt end-to-end | 33.5 - 36.0 | 50.3 - 51.2 | 44.7 - 48.4 |
| C popcnt arithmetic only | 50.0 - 54.5 | 46.7 - 51.1 | 44.9 - 48.6 |
| C abacus, deferred / normalized every block | 434 - 445 / 280 - 284 | 385 - 388 / 289 - 295 | 404 - 416 / 279 - 282 |
| C plane extraction, both operands | 14.8 - 18.5 | 14.7 - 18.8 | 35.7 - 36.9 |
| B scan | 10.4 - 11.9 | 14.3 - 15.6 | 23.5 - 29.3 |
| C popcnt scan | 34.0 - 43.1 | 48.8 - 51.8 | 43.1 - 43.5 |

Streamed "arithmetic only" rows read precomputed planes (another 64 bytes
per block per operand), so they measure more memory, not less arithmetic.

### Break-even for prepared operands

None, on any tier. Prepared reuse can only remove plane extraction
(2.5 ns per block for both operands on VNNI), and C popcnt's arithmetic
alone, with both operands' planes free, is 12.4 ns against B's 2.9 ns with
everything included. In the scan, where the query's planes are prepared once,
C popcnt costs 23.5 ns against B's 2.65 ns.

## Why the gap is structural, not tuning

B's byte-dot multiplies 64 byte pairs in four instructions per block. The
bit-serial form needs 64 partial-product planes per sign, one AND each,
before any reduction: 128 ANDs per block. Even packing eight blocks into
one 512-bit register, with the reduction perfectly vectorized (ternary logic
for `carry_save3`), it would take about 16 ANDs plus at least as many
reduction operations per block. That is an estimate, not a measurement, but
it is already several times B's instruction count. Carry-save trades
popcounts for logic operations; it cannot remove the 64 partial products
that the hardware multiplier forms in one step.

## Falsification criteria (from the brief)

| criterion | result |
|---|---|
| bit-plane preparation dominates the saved arithmetic | no saving to dominate: C's arithmetic alone exceeds B's total |
| carry compression generates more instructions than the byte-dot | yes: 437-488 instructions per block for C csa, against 4 masked byte-dots for B |
| register pressure or spills negate the gain | yes: the carry-save folds keep their columns in memory (`mov` dominated) |
| deferred normalization costs more than immediate | yes when streamed (434 vs 280 ns), equal in L1 |
| exactness or provenance violated | no: every fold is exact on every tier |
| competitiveness needs backend-specific code | yes, it would: a hand-unrolled ternary-logic schedule; not built |

## Not done

- No hand-unrolled or vectorized-across-blocks carry-save schedule (that is
  manual instruction scheduling).
- No NEON or wasm timings (qemu timings mean nothing).
