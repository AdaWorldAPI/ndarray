# TurboVec9 on ndarray: 9-bit sign-magnitude scoring, measured

READ BY: savant-architect, vector-synthesis, simd-savant, anyone wiring a
9-bit index (TurboVec9) or a DR512 dot product onto `ndarray::simd`.

Status: probe + one facade word (`I32x16::dot4_u8_i8_add_masked`). Numbers
below come from `examples/u9_cost_probe.rs` on one AVX-512 VNNI + VBMI Xeon
(target-cpu=native), three runs, reported as ranges. Every exact SM9 kernel
must match an independent i64 reference before its time is printed, and
every facade kernel folds all 16 accumulator lanes into its result. They are not TurboVec
recall numbers; the codebooks in the probe are probe-local Lloyd-Max fits.

## Two contracts, never mixed

- **DR512 / Codec9** (`hpc::codec9`, `simd_stride::lut9`): u8 magnitude plus
  one P bit per value, -255..=255, and `P=0, M=0` is the never-touched
  sentinel. `lut9` belongs here and nowhere else.
- **TurboVec9**: an ID 0..=511 selects one of 512 centroids. Every ID is
  valid. An ID is not a Codec9 value and is never read as sign-magnitude.

The facade word below is pure arithmetic over a u8 magnitude and a u64
plane. It does not know which contract produced them. For DR512 data both
`P=0, M=0` and `P=1, M=0` contribute 0 to a dot product; a caller that must
tell them apart reads the plane itself.

## The facade word

```rust
// Lane j gains the sum of a[i] * b[i] over bytes i in 4j..4j+4 with bit i of keep set.
I32x16::dot4_u8_i8_add_masked(self, a: U8x64, keep: u64, b: I8x64) -> I32x16
```

9-bit sign-magnitude times i8, exact for every magnitude and every i8
(-128 included), i32 accumulators, operands never widened:

```rust
let (mut pos, mut neg) = (I32x16::splat(0), I32x16::splat(0));
for blk in blocks {                       // 64 dims per block
    pos = pos.dot4_u8_i8_add_masked(blk.magnitude, blk.positive, w);
    neg = neg.dot4_u8_i8_add_masked(blk.magnitude, !blk.positive, w);
}
let score_lanes = pos - neg;              // 16 lanes, each a 4-dim partial sum
```

Overflow: each accumulator moves by at most `4 * 255 * 128 = 130_560` per
call, so 16_448 calls (65_792 dims per lane) cannot wrap; the test
`dot4_masked_worst_case_holds_16448_calls` pins both sides of that bound.
No saturating instruction is involved (no `vpmaddubsw`, no `vpdpbusds`).

Realizations: AVX-512 one `vmovdqu8{z}` before `vpdpbusd` (or the widen +
`vpmaddwd` path without VNNI); AVX2 builds the byte mask in registers
(variable shift, multiply by `0x0101_0101`, shift by 32) and applies it
inside the dot loop; AVX, NEON and wasm expand a stored spread with one AND
against `BIT_IN_BYTE` and one byte compare per register (no shuffle, no
per-lane loop); nightly selects through `core::simd::Mask`;
scalar is the reference loop.

Witnessed: `simd::dot4_nibble_facade_parity` 8/8 on native (AVX-512 VNNI),
config-v4 (AVX-512 without VNNI), config-v3 (AVX2), config-avx (qemu
SandyBridge), config-v2 (scalar, qemu Nehalem), aarch64 (qemu) and
nightly-simd; `neon-parity.sh` and `wasm-parity.sh` pass (codes 0x504,
0x505). Disable runs, all red: unmasked AVX-512 move (3 tests), reversed
`BIT_IN_BYTE` on AVX2 (2 tests), mask dropped in the fused AVX2 loop (3),
wrong byte shifts for the upper AVX2 half (2).

## Arithmetic: three exact forms compared (AVX-512 VNNI, L1, ns per 64 lanes)

| kernel | ns / 64 lanes |
|---|---|
| u8 x i8 `vpdpbusd` (#386 `dot4_u8_i8_add`) | 1.12 - 1.14 |
| i8 x i8 via `xor 0x80` bias | 1.20 - 1.25 |
| A: sign folded into w (exact only for w in -127..=127) | 1.23 - 1.26 |
| B: two's-complement limbs, `dot(lo) - 256 * dot(sign)`, sign limb skips magnitude 0 | 1.89 - 1.92 |
| C: `dot(m) - 2 * dot(m & neg)`, one accumulator | 1.79 - 1.88 |
| **C': `dot(m & pos) - dot(m & neg)`, two accumulators** | **1.28 - 1.30** |

The loop is one latency-bound chain. B and C put two dependent ops on the
accumulator; C' keeps two independent chains and lands within ~10% of the
plain u8 dot. A is not exact for w = -128 and was not added.

Through the facade, same host: `dot4_u8_i8_add` 1.08 - 1.15, two masked
accumulators 1.34 - 1.38. On config-v3 (AVX2, no VNNI): `dot4_u8_i8_add`
1.59 - 1.79, two plain accumulators 2.75 - 2.91, two masked accumulators
6.35 - 6.44. Without VNNI each masked call adds the mask expansion (about
1.4 ns, measured as one masked accumulator against one plain one) on top of
its own 16-bit widening, so the pair costs about 4x one dot4. The loop is
instruction-bound, not memory-bound: building the mask in registers instead
of from a stored spread did not change the time. A fused SM9 op that widens
once and negates in 16 bits would avoid both the second widening and the
mask, and is not built.

## 512-entry lookup (WP2): keep it out of the scan

ID = low byte + high-bit plane, decoded to magnitude + positive plane:

| lookup | ns / 64 lanes |
|---|---|
| per-position reads (`lut_u8` shape, two 512-entry tables) | 38 - 39 |
| 4x `vpermi2b` + blends for magnitude, same for the sign | 6.03 - 6.09 |
| 2x `vpermi2b` on an antisymmetric codebook (`255 - lo` for negatives) | 6.72 - 6.77 |

`lut9` and `lut_u8` keep their contracts; no 512-entry facade word was
added. The lookup alone costs ~5x the arithmetic it feeds. TurboVec's
levels do not depend on the query, so the decode can happen once at encode
time; then the scan stores values, not IDs (next section).

## Precision vs cost: the layouts (N(0,1) coordinates)

| layout | SNR | scan cost (AVX-512 VNNI) |
|---|---|---|
| i4, 16 Lloyd-Max levels (today) | 20.22 dB | 2.65 - 2.68 ns / 128 dims (L1) |
| i8 uniform grid, best clip | 40.54 dB | 1.12 - 1.14 ns / 64 dims |
| 512 Lloyd-Max IDs, executed by `vpermi2b` lookup | 49.86 dB | 5.37 - 5.70 ns / 64 dims |
| 512 Lloyd-Max levels snapped to the SM9 grid | 44.74 dB (329 distinct) | 1.29 - 1.32 ns / 64 dims |
| **SM9 uniform grid, best clip (4.21 sigma), stored** | **46.02 dB** | **1.29 - 1.32 ns / 64 dims** |
| 4+5 additive C16 + R32, two i8 grids | 43.25 dB | 2.61 - 2.65 ns / 64 dims |

Streamed from DRAM (64 MiB of codes, weights in L1), per dimension:
i4 0.035 - 0.037 ns, stored SM9 0.048 - 0.052 ns (72 bytes per 64 dims).

Reading the table:

- Stored values make the 512 Lloyd-Max IDs pointless: snapping them to
  the 9-bit grid keeps 329 distinct values and loses to a plain uniform
  9-bit grid with a tuned clip. The Lloyd-Max levels only pay off when
  executed as IDs, and the lookup makes that path ~4x slower than the
  stored one.
- The 4+5 additive competitor is worse than stored SM9 on both axes in this
  probe. Its residual codebook is one shared Lloyd fit after a 16-level
  split; a joint fit could do better, but it cannot beat the 512-point
  optimum and it pays two lookups.

## Answer to the acceptance question

On AVX-512 VNNI: yes, for the stored-value layout. 64 dims cost one
masked-pair of `vpdpbusd`, ~1.3 ns in L1 (about the same per dimension as
today's i4 permute-dot) and 1.4x i4 per dimension when streamed from DRAM,
for +25.8 dB over i4. With IDs plus an in-scan 512 lookup: no, the lookup
dominates. On AVX2 the pair costs ~4x one plain dot4; a fused SM9 op is the
next step there if AVX2 matters.

## Handoff to the TurboVec owner (owns ABI, encode, pack, search, mutation, serialization, Python)

Proposed scan layout, per 64 dimensions of one vector, vector-major as
today: 64 magnitude bytes plus one u64 positive plane (72 bytes). Scoring:
the two-accumulator loop above, then the existing `scale` / `zero`
correction (`zero` is 0 here: magnitudes are unbiased u8, weights stay i8).
Open decisions on the TurboVec side: grid clip per dimension or global, how
the encoder writes 0 (either plane bit scores 0), and recall measured on
real data with TurboVec's own codebook solver. None of this was changed
from this session.

## Not done

- No NEON or wasm timings (qemu timings mean nothing); NEON `SDOT` /
  `USDOT` and AVX-VNNI (256-bit) are not used by the masked word.
- No fused single-widen SM9 op for non-VNNI tiers.
- No signed i8 x i8 facade word.
- Recall on a real index.

## Second facade word: `U64x8::sum8_u8_add_masked` (2026-10-29)

`U64x8::sum8_u8_add_masked(self, a: U8x64, keep: u64) -> U64x8` adds to lane `j` the sum of the bytes `a[i]`, `i` in `8j..8j+8`, whose bit `i` in `keep` is set. It is exact. Each call adds at most 8 * 255 = 2,040 per lane, so the u64 lanes wrap only after more than 9 * 10^15 calls. There is no horizontal reduction; reduce once at the end.

It was added for the power-of-two anchor term of candidate E in PR #389. With `|b| = low7 + 128*bit7`, `a*|b| = a*low7 + 128*(sum of a where bit 7 of b is set)`. The facade only had `U8x64::sum_bytes_u64`, which reduces horizontally on every call.

| Tier | Lowering |
|---|---|
| AVX-512 | zero-masked `vmovdqu8`, `vpsadbw` against zero, `vpaddq` |
| AVX2 | mask built in registers as in `dot4_u8_i8_add_masked`, AND, `vpsadbw`, `vpaddq`, two halves (LLVM emits 2 `vpshufb`; none in the source) |
| AVX without AVX2 | `mask_byte_spread`, `cmpeq`, AND, `psadbw` per 16-byte quarter |
| NEON | `vtstq_u8` mask, AND, `uaddlp` x2, `uadalp`; no scalar loop |
| wasm simd128 | `i8x16_eq` mask, AND, two `extadd_pairwise`, fold of 32-bit halves, `i64x2_add`; no shuffle |
| scalar | per-lane loop, the correctness anchor |
| nightly-simd | `Mask::from_bitmask(keep).select`, SWAR fold |

Measured on an AVX-512 VNNI Xeon (4,096 random vectors and masks, medians of 1,001 runs): 0.687 ns per 64 bytes at target-cpu=native against 192.8 - 196.0 ns for the plain reference loop (281 - 285x); 1.987 - 1.991 ns at x86-64-v3 against 194.1 - 195.4 ns (98x). NEON and wasm were not timed.

Tests: `sum8_masked_matches_reference`, `sum8_masked_gates_each_byte_by_its_own_bit`, `sum8_masked_all_kept_agrees_with_sum_bytes` in `dot4_nibble_facade_parity`, plus the doc example (run as a doctest on the native and v3 builds) and parity codes 0x506 and 0x507 in `wasm-simd-parity` and `neon-simd-parity`.
