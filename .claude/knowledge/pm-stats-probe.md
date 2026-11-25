# D-PM-STATS: what P×M makes cheaper (probe, 2026-11-25)

> **READ BY:** anyone proposing a Codec9 / Abakus operation, a P×M reduction
> or compare, or citing P×M performance.
>
> **Status:** MEASURED-LAB on one shared host, two runs per tier. Research
> only. Nothing here is a production API, a kernel, or a layout decision.

Code: `examples/pm_stats/stats.rs` (the probes), `examples/pm_stats/main.rs`
(bench), `tests/pm_stats.rs` (oracles). Composed from existing
`ndarray::simd` words only. Base: ndarray master `a6b6c05`.

## What carries the result

An untouched lane has `M = 0`, so it contributes nothing to `Σx`, `Σ|x|`,
`Σx²` or `Σxy`. **Only the count `N` needs `touched`.** No sum, power sum or
product needs a validity filter; `x²` and `|x|` need no polarity at all.

## Contracts and oracles

| Probe | Contract | Oracle |
|---|---|---|
| Stats | `N` selected touched, `S = Σx`, `Q = Σx²`, `A = Σ|x|`; variance as the exact rational `(N·Q − S², N²)`, `None` for an empty population | exact scalar fold; boundary blocks: all untouched, only `+0`, full cancellation, nearly constant (63/4096) |
| Compare | `lt`, `le`, `eq` three-valued (`is`, `unknown`); `lt_const` for `x < c` | every 512 × 512 codeword pair; every constant −255..=255 |
| Select | `min`, `max`, untouched where either operand is | every 512 × 512 pair |
| Products | `Σxy`, as `Σ_same − Σ_opp` and as `Σ_all − 2Σ_opp` | every 512 × 512 pair, whole and per block |

11 tests, green on AVX-512 and on the v3 (AVX2) tier. NEON and wasm were not
run.

Disable runs, each red: `lt` ignoring polarity (2 tests), `lt` without the
known mask (1), `N` from P alone (3), arm B ignoring polarity (1), `prod_sum`
dropping its high byte (5), the dot product ignoring polarity (1), `x < 0`
without the touched check (1).

## Measured (ns per 64 lanes, best of 7; two runs, shared host)

The run-to-run spread is large on this host (some cells move 2–3×). Read the
ranges, not the digits. The i16 sum and stats baselines are scalar loops the
compiler did not vectorize well; the compare baseline is a real `I16x32` SIMD
loop, and the sum baseline is the library's `masked_sum_i32`.

**Signed sum S, 64K lanes**

| | split | `ΣM − 2ΣM⁻` | `masked_sum_i32` |
|---|---|---|---|
| AVX-512 | 2.9–3.9, flat over density and mix | 4.0 | 1.7 at ≤1% selected, 5.7 at 10%, 12 at ≥50% |
| AVX2 | 9.8–21 | 8.4–13.7 | 1.7–4 at ≤1%, 9–20 dense |

- P×M cost is independent of selection density and polarity mix. `masked_sum_i32` walks set bits, so it wins below roughly 1–10% density and loses 3–4× when dense.
- **`ΣM − 2ΣM⁻` is slower on AVX-512 and faster on AVX2.** The better form is ISA-dependent; neither is universally cheaper.

**All four statistics, 50% selection:** AVX-512 12–18 at every working set
(64, 64K, 16M lanes), against the scalar i16 loop at 41–51. On AVX2 there is
no win: 44–47 against 40–48. `prod_sum` (for `Q`) is the cost there; the
AVX2 `U16x32` is two halves.

**`a < b`:** 3.5–4.5 against `I16x32` at 3.1–4.2 when L2-resident (AVX2:
4.95 against 3.08, P×M slower). At 16M lanes 5.3–17 against 12–20; the gap
there is bytes. **`x < 0`** is one `cmpeq`: 1.7–1.9.

**Dot product:** AVX-512 16–17 against 11–15 L2-resident (no win), 20–25
against 26–32 at 16M lanes (win, bandwidth). **AVX2: 59–61 against 11–30, a
large loss.**

## What this establishes

- The strongest fruit is the **statistical fold** on AVX-512: N, S, Q, A in one pass, density-independent, no decode, no validity prefilter.
- **Signed comparison is not cheaper in ALU work.** It costs about the same as an `I16x32` compare and wins only when memory-bound. **`x < 0` is genuinely cheaper.**
- **The dot product is not cheaper in ALU work**, and is much worse on AVX2. Its only win is bandwidth at large working sets on AVX-512.
- Every advantage measured here comes from **bytes (72 vs 128–256 per 64 lanes) or from density-independence**, not from fewer instructions. The exception is `x < 0`.

## Not measured, or not established

- Instruction sequences: the arms inlined, and no disassembly was captured.
- Hot vs cold beyond working-set size; latency vs throughput (only throughput loops).
- NEON, wasm, AVX-only and v2 tiers.
- Allocations: the arms allocate nothing per block, but this was read, not instrumented.
- Wider widths (PM17/25/33) and widened accumulators: not probed.
- Composition across sealed DR tiles: N, S, Q, A add across tiles, but no tile fold was built.

## Reusable primitive inventory

`U8x64::{mask_blend, cmpgt_mask, cmpeq_mask, sum_bytes_u64}`,
`U16x32::{from_u8x64_lo, from_u8x64_hi, mullo, shr, shl, Sub,
pack_saturate_u8}` (lane-order preserving), `abakus::pm::touched`,
`I16x32::cmp_gt`. `masked_group_bounded_power_sums_u8` is a scalar keyed
fold (`group_walk`, one row at a time), not a SIMD kernel.

## Vertical byte accumulation (follow-up, 2026-11-25)

`sum_abs_vertical` widens each masked magnitude block into `U16x32` partials
and reduces horizontally once per 128 blocks (`2 × 255 × 128 = 65_280`; a
129-block tile wraps, and the disable run with 129 is red). It is composed
from existing words; no new primitive.

`S` and `A` together, 50% selection, ns per 64 lanes, two runs:

| | horizontal per block | vertical per tile |
|---|---|---|
| AVX-512, L2-resident | 2.97–3.56 | 2.15–2.73 |
| AVX-512, 16M lanes | 5.59–6.41 | 4.83–5.49 |
| AVX2, L2-resident | 7.6–9.6 | 8.0–10.9 |
| AVX2, 16M lanes | 7.7–11.0 | 9.7–16.6 |

- On AVX-512 vertical accumulation saves about 0.8 ns per 64 lanes (20–27%).
- On AVX2 it is no better and usually worse: the `U16x32` widen and add are two halves each, which costs more than the `vpsadbw` it replaces.
- So a vertical accumulator is an AVX-512 realization choice, not a portable rule. A polyfill word for it would need a per-backend body (on AVX2 the horizontal form stays).
- ⊘ CORRECTED 2026-11-25: this line said a `vpsadbw` that returns its `U64x8` partials did not exist. It does: `U64x8::sum8_u8_add_masked(a, keep)` (ndarray #407), on every backend, and it takes the selection mask directly, so no blend is needed either.

### The existing masked SAD word (`sum_abs_sad`)

Same population, same two runs per tier:

| | horizontal | vertical (`U16x32`) | `sum8_u8_add_masked` |
|---|---|---|---|
| AVX-512, L2-resident | 2.94–2.97 | 2.15 | **1.25–1.33** |
| AVX-512, 16M lanes | 3.8–5.9 | 2.8–5.0 | **2.7–4.4** |
| AVX2, L2-resident | 6.28–6.31 | 7.8–7.9 | **4.16–4.17** |
| AVX2, 16M lanes | 11.3–12.0 | 16.3–17.9 | **6.0–7.5** |

It is the fastest of the three on both tiers: 2.3× the horizontal form on AVX-512, 1.5× on AVX2, where the widening accumulator lost. So the portable rule is "accumulate with `sum8_u8_add_masked`", and the widening accumulator above is superseded. The disable run (negatives masked by `P` instead of `!P`) is red.
