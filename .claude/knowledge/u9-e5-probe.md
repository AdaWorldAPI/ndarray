# D-U9-E5: the U9 power-of-two anchor through vertical masked byte sums

READ BY: savant-architect, simd-savant, vector-synthesis, the TurboVec-U9 owner.
Follows `u9-abacus-probe.md` (#389, candidate E) and the facade word
`U64x8::sum8_u8_add_masked` (#390).

Scope: one new consumer candidate, `AccE5` in `examples/u9_abacus/anchor.rs`.
E3, E4, Esum and B are unchanged, so every earlier row stays reproducible. No
facade, backend, codec or storage change.

## Answer

**Does the vertical `U64x8` accumulator make the U9 anchor cheaper than B's
fourth byte-dot? Only where the byte-dot is not a single instruction.**

| tier | verdict | complete-consumer result against B |
|---|---|---|
| AVX-512 without VNNI (v4) | **ADOPT-LAB** | L1 end-to-end 5.14 - 5.30 vs 5.67 - 7.50 ns per 64 (about 9 % faster, 8 runs, ranges disjoint); arithmetic 4.47 vs 5.08 typically, lower in all 8 paired runs; scan 4.67 vs 4.83; streamed end-to-end lower in every paired run |
| AVX-512 VNNI | **REFUSE** as a B replacement | L1 end-to-end 3.55 - 3.95 vs 2.92 - 3.25 (1.2x slower); scan 3.31 vs 2.62 - 3.05; streamed end-to-end tie (5.55 - 5.76 vs 5.50 - 6.11) |
| AVX2 (v3) | **REFUSE** as a B replacement | L1 end-to-end 12.2 vs 10.0; streamed end-to-end 40.0 - 40.5 vs 27.1 - 31.7 |

Overall: **DEFER** as a general B replacement, **ADOPT-LAB** on AVX-512 without
VNNI. E5 is, on every tier, the best of the four E variants, and the first E
variant that beats B anywhere.

## What E5 does

E5 keeps E's split `|b| = low7 + 128 * bit7` and the two masked `low7`
byte-dots. The anchor term goes into two persistent vertical accumulators:

```rust
self.anchor[0] = self.anchor[0].sum8_u8_add_masked(a, !negative & top);
self.anchor[1] = self.anchor[1].sum8_u8_add_masked(a, negative & top);
```

`top` is the bit-7 plane of `|b|` and `negative = plane_a ^ plane_b`. At flush:

`total += (low7_pos - low7_neg) + ((reduce_sum(anchor_pos) - reduce_sum(anchor_neg)) << 7)`

The anchors are reduced only at flush, every `B_FLUSH_BLOCKS` = 16,384 blocks,
never per block. `AccE5::reductions` counts the reductions, so a premature
reduction is observable.

Bounds per flush:

- a `low7` lane gains at most `4 * 255 * 127 = 129,540` per block, at most
  2,122,383,360 per flush, below `i32::MAX`;
- an anchor lane gains at most `8 * 255 = 2,040` per block, at most 33,423,360
  per flush. The 8 lanes sum to at most 267,386,880 and `<< 7` gives at most
  3.4e10, inside `i64`.

## Measurements

ns per 64 values, compact operand layout (72 bytes per block, as in #388 and
#389), mask construction inside every timed row. VNNI and AVX2 are 3 runs each;
v4 is 8 runs (3, then 5 more to confirm the win). Ranges are min - max of
medians. L1 = 64 blocks, streamed = 2^20 blocks.

| L1 | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| B end-to-end | 2.92 - 3.25 | 5.67 - 7.50 | 10.0 |
| E3 end-to-end | 2.94 - 3.36 | 5.42 - 6.59 | 38.1 - 53.8 |
| E4 end-to-end | 3.59 - 4.11 | 6.36 - 7.78 | 14.5 |
| Esum end-to-end | 4.45 - 4.75 | 5.98 - 6.56 | 60.0 - 64.6 |
| **E5 end-to-end** | 3.55 - 3.95 | **5.14 - 5.30** | 12.2 |
| B arithmetic only | 2.44 - 2.69 | 5.02 - 5.64 | 9.19 - 10.8 |
| E3 arithmetic only | 2.31 - 2.47 | 4.98 - 5.88 | 37.9 - 40.0 |
| **E5 arithmetic only** | 2.72 - 2.95 | **4.47 - 5.17** | 10.8 - 11.0 |
| B scan | 2.62 - 3.05 | 4.83 - 4.97 | 9.11 - 9.17 |
| E3 scan | 2.67 - 3.05 | 4.98 - 7.89 | 38.4 - 48.2 |
| **E5 scan** | 3.31 | **4.67 - 8.39** | 11.3 - 11.6 |

| Streamed | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| B end-to-end | 5.50 - 6.11 | 8.38 - 14.2 | 27.1 - 31.7 |
| E3 end-to-end | 5.31 - 5.38 | 7.01 - 12.6 | 216 - 223 |
| **E5 end-to-end** | 5.55 - 5.76 | **6.80 - 12.1** | 40.0 - 40.5 |
| B scan | 2.82 - 3.31 | 4.61 - 8.91 | 26.6 - 27.5 |
| E3 scan | 2.74 - 2.80 | 4.88 - 5.99 | 212 - 225 |
| **E5 scan** | 3.19 - 3.33 | **4.64 - 5.13** | 25.1 - 28.7 |

E4 and Esum streamed are in the probe output and lose to E5 on every tier. The
streamed arithmetic-only rows read precomputed operands, so they measure
memory, not arithmetic, and are left out, as in #389.

Notes:

- v4 L1 scan: 7 of 8 runs were 4.67 against B's 4.83; one run read 8.39,
  noisy on that run only.
- v4 streamed end-to-end: the ranges overlap, because the host varies between
  runs. Paired by run (5 confirmation runs), E5 is lower than B every time:
  12.1/14.2, 10.2/12.3, 10.6/12.7, 7.8/11.9, 7.3/10.4.

### Flush overhead (ns per flush)

| | VNNI | v4 | AVX2 |
|---|---|---|---|
| B: 4 x `I32x16` -> `i64` | 48.3 - 48.9 | 48.2 - 72.7 | 27.1 - 28.2 |
| E3: 3 x `I32x16` | 36.9 - 37.1 | 37.0 - 37.4 | 19.8 - 19.9 |
| E5: 2 x `I32x16` + 2 x `U64x8::reduce_sum` | 27.5 - 27.7 | 27.3 - 30.2 | 15.3 - 15.8 |

The flush is cheaper for E5, but amortized over 16,384 blocks it is below
0.003 ns per 64 for every candidate, so it decides nothing.

## Assembly (steady-state loop body, `#[inline(never)]` loop copies)

| tier | B body | E5 body | spills in the body | E5 per block |
|---|---|---|---|---|
| VNNI | 18 instructions: 4 `vpdpbusd`, 2 `vmovdqu8 {z}` | 23: 2 `vpdpbusd`, 4 `vmovdqu8 {z}`, 2 `vpsadbw`, 2 `vpaddq`, `kandq`, `kandnq` | 0 / 0 | no shuffle, no horizontal reduction |
| v4 (no VNNI) | 40: 8 `vpmaddwd`, 8 `vpaddd`, shifts | 36: 4 `vpmaddwd`, 4 `vpaddd`, 2 `vpsadbw`, 2 `vpaddq`, 4 `vmovdqu8 {z}` | 0 / 0 | no shuffle, no horizontal reduction |
| AVX2 | 154, 46 stack operands | 185, 42 stack operands: 8 `vpshufb`, 16 `vpmuludq`, 8 `vpcmpeqb` | yes / yes (array-backed AVX2 types) | the keep mask is built in registers per call |

At flush, LLVM folds `reduce_sum(pos) - reduce_sum(neg)` into one `vpsubq`
followed by one horizontal reduction (`vextracti64x4`, `vextracti128`,
`vpshufd`). That runs once per 16,384 blocks.

## Why E5 loses where it loses

- **AVX-512 VNNI: mask expansion.** The cause is not register pressure (0
  spills), not the two extra accumulator chains (they are independent and the
  loop has no carried dependency beyond the adds), and not memory traffic
  (streamed, E5 ties B). B's fourth byte-dot is one `vpdpbusd` that reuses the
  zero-masked copy of `a` the sign already needs: both of B's dots per sign
  share one mask. E5's anchor mask is `sign AND bit7`, which differs from the
  `low7` mask, so every anchor term pays its own zero-masked move, a `vpsadbw`
  and a `vpaddq`. That is three instructions against one, per sign.
- **AVX2: mask expansion again.** `sum8_u8_add_masked` builds its keep mask in
  registers (variable shift, multiply, compare) on every call. That costs more
  than the byte-dot it replaces. Both kernels spill because the AVX2 `U8x64`
  and `I32x16` are array-backed; E5 spills slightly less.

## Why E5 wins on AVX-512 without VNNI

Without VNNI, a masked byte-dot is emulated with about 10 instructions
(`vpmaddwd`, `vpaddd` and shifts). E5 replaces two of B's four byte-dots with
two SAD sums. A SAD sum costs 3 instructions including its own mask, so the
mask overhead that sinks E5 on VNNI is smaller than the work it saves.

Conditions for the win:

- AVX-512BW without VNNI;
- the compact 72-byte block layout;
- the `b` side either prepared per block (end-to-end) or once as a query (scan).

The win holds in L1, is smaller in the scan (3 %), and streamed it is lower in
every paired run but within host noise.

## Correctness

`tests/u9_abacus.rs`, 5 new tests (20 in the file), against #388's `i64`
oracle and against B:

- all 511 x 511 numeric pairs, one lane each, end-to-end and as a scan;
- mixed full blocks of 1, 2, 7, 64 and 300 blocks with random signs. The test
  asserts that mixed signs and bit-7 magnitudes are present;
- all-255 runs past three flush bounds with all four sign mixes; totals beyond
  `i32`;
- one anchor reduction per flush (`AccE5::reductions`);
- touched zero scores 0, and never-touched is refused before E5 sees the block.

The file passes on native (AVX-512 VNNI), v4, v3, AVX (qemu SandyBridge), v2
(qemu Nehalem), aarch64 (qemu) and nightly-simd. `scripts/wasm-parity.sh` and
`scripts/neon-parity.sh` pass; the backends are unchanged. Clippy `-D warnings`
(native, v3) and rustfmt are clean. The facet-stride guard reports nothing on
the added lines.

Disable runs, each red:

| id | disable | result |
|---|---|---|
| X1 | bit-7 gating lost on the positive anchor (`!negative` instead of `!negative & top`) | RED (3 tests) |
| X2 | polarity inverted (positive and negative anchor masks swapped) | RED (3 tests) |
| X3 | reconstruction `<< 6` instead of `<< 7` | RED (3 tests) |
| X4 | anchor reduction missing at flush | RED (3 tests) |
| X5 | anchors not reset at flush (double count) | RED (1 test, the run across three flush bounds) |
| X6 | premature reduction, flushing every block | RED (1 test, the reduction count). Every exactness test stays green: an early reduction is a cost defect, not a wrong result, so it needs a counter to be visible |

## Not done

- NEON and wasm timings. The parity harnesses pass; qemu and node timings are
  not meaningful.
- An E5 variant whose anchor mask is shared with the `low7` mask. It would
  need a SAD word that takes an already-masked vector plus a second mask, or a
  different split. Not built; it would be a new facade word.
- Integration anywhere outside the probe.
