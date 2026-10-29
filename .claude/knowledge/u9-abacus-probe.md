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
| B end-to-end | 2.81 - 2.89 / 5.46 - 6.96 | 5.62 / 8.85 - 13.0 | 10.0 - 10.1 / 24.0 - 25.2 |
| **E3 end-to-end** | **2.84 - 2.92** / 5.35 - 6.91 | **5.52** / 7.28 - 12.9 | **37.6 - 43.6** / 217 - 222 |
| E4 end-to-end | 3.47 - 3.56 / 5.67 - 7.25 | 6.47 - 6.48 / 8.63 - 15.5 | 14.5 / 38.6 - 43.1 |
| Esum end-to-end | 4.33 - 4.44 / 6.51 - 8.95 | 5.95 / 8.08 - 13.5 | 61.2 - 66.8 / 278 - 279 |
| B arithmetic only | 2.34 - 2.41 / 16.4 - 18.1 | 5.02 / 22.4 - 23.8 | 9.17 - 9.23 / 27.3 - 29.3 |
| **E3 arithmetic only** | **2.28** / 8.29 - 10.3 | **4.94 - 4.95** / 14.6 - 17.7 | **37.5 - 45.3** / 217 - 230 |
| E4 arithmetic only | 2.67 - 2.73 / 11.0 - 13.1 | 5.81 / 16.0 - 18.9 | 12.9 / 25.0 - 25.4 |
| B scan | 2.66 / 2.85 - 3.10 | 4.73 - 4.77 / 5.31 - 8.01 | 9.16 - 9.17 / 26.1 - 29.1 |
| E3 scan | 2.64 / 2.83 - 3.74 | 4.95 - 4.97 / 4.89 - 5.66 | 35.5 - 43.3 / 213 - 214 |

Reading it:

- **E3 saves one byte-dot and pays it back building the sign vector.** On
  VNNI it ties B end to end (2.84 - 2.92 vs 2.81 - 2.89) and in the scan, and is
  about 4 % faster in arithmetic only (2.28 vs 2.34 - 2.41); without VNNI it is
  about 2 % faster end to end and slower in the scan. Disassembly: 3
  `vpdpbusd` against B's 4, plus the `kmovq` and blend that make `+1/-1`.
- **E4 is slower everywhere.** A byte-dot against constant ones costs the
  same instruction as one against -128; the extra mask ANDs are pure cost.
- **Esum is slower everywhere.** `sum_bytes_u64` reduces horizontally on
  every call.
- **On AVX2, E3 and Esum lose to `mask_blend`'s per-lane realization**
  (38 - 67 ns); E4, which needs no blend, is 1.4x B.

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
| A' arithmetic only (f32 accumulate) | 6.34 - 6.50 / 18.4 - 18.8 | 6.28 - 6.69 / 19.0 - 19.6 | 10.7 - 11.0 / 21.7 - 22.0 |
| **D react arithmetic only** | **5.20 - 5.70** / 18.2 - 20.1 | **5.36 - 5.64** / 18.4 - 19.8 | **21.2 - 24.2** / 31.9 - 32.0 |
| D anticipate arithmetic only | 5.94 - 6.45 / 18.7 - 20.9 | 5.98 - 6.31 / 19.4 - 20.5 | 9.89 - 10.1 / 22.0 - 26.9 |
| B arithmetic only (#388) | 2.39 - 2.41 / 17.3 - 17.9 | 5.02 - 5.03 / 22.3 - 22.8 | 8.95 - 9.17 / 27.2 - 29.4 |
| A' end-to-end | 7.62 - 7.83 / 18.0 - 18.1 | 7.83 / 19.5 - 19.7 | 81.1 - 88.7 / 416 - 421 |
| **D react end-to-end** | **6.62 - 7.25** / 11.0 - 15.1 | **6.77 - 6.80** / 15.3 - 16.0 | **79.6 - 90.1** / 434 - 440 |
| D anticipate end-to-end | 7.78 - 7.97 / 15.3 - 19.5 | 7.78 - 7.80 / 19.1 - 19.7 | 77.6 - 79.7 / 422 - 426 |
| B end-to-end (#388) | 2.89 - 2.91 / 6.06 - 7.71 | 5.62 / 11.0 - 12.9 | 10.0 - 10.1 / 26.5 - 28.2 |
| D react scan, query widened once | 6.22 - 6.56 / 11.9 - 16.2 | 6.23 - 7.20 / 11.7 - 19.1 | 88.6 - 91.1 / 433 - 436 |
| D anticipate scan | 7.00 - 7.47 / 11.0 - 15.9 | 7.27 - 8.41 / 10.4 - 22.7 | 76.9 - 77.2 / 417 - 420 |
| B scan (#388) | 2.66 / 2.88 - 3.54 | 4.67 - 4.78 / 6.13 - 9.15 | 9.17 - 9.83 / 26.4 - 27.1 |

### Reading it

- **The carry column beats the f32 detour in L1.** On AVX-512, react is
  about 10 % faster than A' in arithmetic (5.2 - 5.7 vs 6.3 - 6.5 ns) and in
  the full path (6.6 - 7.3 vs 7.6 - 7.8). Streamed, on compact operands,
  their arithmetic-only rows are level (18 - 20 vs 18 - 19 ns: both read
  precomputed operands and are memory-bound), and react end to end is
  faster (11 - 15 vs 18 ns). Anticipate costs one more ternary op per carry
  and lands level with A'.
- **React vs anticipate depends on the tier.** On AVX-512, react wins (the
  carry needs the sum anyway, and two ternary ops suffice). On AVX2,
  anticipate wins (9.9 - 10.1 vs 21 - 24 ns); the react form gets 16 `vmovq`
  and 6 `vpblendd` from LLVM there.
- **B still wins, and the reason is not the accumulation.** D removes the
  float conversion, but it still widens u8 -> u16 (`vpmovzxbw`,
  `vextracti64x4`) and runs four `vpmullw` per block before any add. B
  multiplies straight from bytes and accumulates in `i32` in the same
  instruction. On VNNI, B's arithmetic is 2.4 ns against D's 5.2 - 5.7.
- **Sign-routing overhead** (end-to-end minus arithmetic-only medians, so it
  also contains `check` and widening `|b|`): D react on VNNI 1.45 ns, on AVX-512
  without VNNI 1.33 ns, on AVX2 about 64 ns. A' carries the same routing
  (1.47 ns on VNNI, 77 ns on AVX2). B routes with the byte-dot's own mask
  instead (0.48 ns on VNNI, 0.88 on AVX2).
- **On AVX2 the end-to-end and scan rows are dominated by routing.** D's
  routing is A's `U8x64::mask_blend`, whose AVX2 realization is a per-lane
  loop (77 - 91 ns). That cost has nothing to do with the carry.

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
| `planes` | 6 `vptestmb` + 2 `vpmovb2m` (+ `vpsllw` for bit 0): one instruction per plane | same | out of line (a `jmp`); extraction 11.1 ns vs 3.5 |
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

Timed operands use #388's compact layout (64 magnitude bytes and one plane
per block in two arrays, 72 bytes); each `Block` is built in registers. The
first version streamed a `Vec<Block>`, which pads every block to 128 bytes,
and its extraction row consumed four of the sixteen planes, so the optimizer
could drop the rest (it read 2.52 ns on VNNI L1). Both were fixed after
review (Codex, PR #389) and every table here was re-measured; the plane
extraction row now passes both complete plane arrays through `black_box`.

| L1-resident | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| B end-to-end (#388) | 2.89 - 2.91 | 5.62 | 10.0 - 10.1 |
| C popcnt end-to-end | 28.1 - 28.2 | 47.1 - 48.2 | 42.4 |
| C abacus end-to-end | 286 - 288 | 290 - 292 | 317 - 319 |
| C csa end-to-end | 298 | 295 - 296 | 304 - 305 |
| B arithmetic only | 2.39 - 2.41 | 5.02 - 5.03 | 8.95 - 9.17 |
| C popcnt arithmetic only, planes given | 12.4 - 12.8 | 30.1 | 30.7 |
| C abacus arithmetic only, deferred | 266 - 276 | 289 - 292 | 292 - 298 |
| C abacus arithmetic only, normalized every block | 271 - 278 | 276 - 282 | 281 - 284 |
| C csa arithmetic only | 293 | 291 | 290 - 291 |
| C plane extraction, both operands (cold) | 3.45 - 3.53 | 3.45 - 3.58 | 11.1 |
| B scan (#388) | 2.66 | 4.67 - 4.78 | 9.17 - 9.83 |
| C popcnt scan, query planes once | 23.4 - 23.9 | 46.7 - 47.8 | 41.8 - 42.9 |
| C abacus scan | 286 - 291 | 296 - 298 | 313 - 329 |
| C csa scan | 301 - 307 | 300 - 301 | 302 - 303 |

| Streamed | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| B end-to-end | 6.06 - 7.71 | 11.0 - 12.9 | 26.5 - 28.2 |
| C popcnt end-to-end | 51.1 - 55.9 | 50.0 - 50.7 | 54.5 - 57.2 |
| C popcnt arithmetic only | 47.0 - 50.0 | 60.1 - 60.7 | 57.4 - 60.7 |
| C abacus, deferred / normalized every block | 438 - 446 / 283 - 285 | 404 - 405 / 293 - 309 | 398 - 408 / 292 - 300 |
| C plane extraction, both operands | 7.81 - 11.4 | 11.2 - 11.6 | 56.1 - 57.6 |
| B scan | 2.88 - 3.54 | 6.13 - 9.15 | 26.4 - 27.1 |
| C popcnt scan | 54.3 - 57.2 | 61.8 - 64.3 | 60.1 - 62.6 |

Streamed "arithmetic only" rows read precomputed planes (another 64 bytes
per block per operand), so they measure more memory, not less arithmetic.

### Break-even for prepared operands

None, on any tier. Prepared reuse can only remove plane extraction
(3.5 ns per block for both operands on VNNI), and C popcnt's arithmetic
alone, with both operands' planes free, is 12.4 ns against B's 2.9 ns with
everything included. In the scan, where the query's planes are prepared once,
C popcnt costs 23.4 - 23.9 ns against B's 2.66 ns.

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
| deferred normalization costs more than immediate | yes when streamed (438 - 446 vs 283 - 285 ns), equal in L1 |
| exactness or provenance violated | no: every fold is exact on every tier |
| competitiveness needs backend-specific code | yes, it would: a hand-unrolled ternary-logic schedule; not built |

## Not done

- No hand-unrolled or vectorized-across-blocks carry-save schedule (that is
  manual instruction scheduling).
- No NEON or wasm timings (qemu timings mean nothing).
