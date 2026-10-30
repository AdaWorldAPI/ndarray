# D-BINARY-ABACUS-POLYFILL-0: binary positional arithmetic over the facade

Status: PR A shipped as #402 (draft). PR B and PR C deferred, with reasons
below. Host: AVX-512 VNNI Xeon. Date: 2026-10-30.

## What shipped (PR A, #402)

- `ndarray::simd::half_add_mask(a, carry) = (a ^ carry, a & carry)` and
  `full_add_mask(a, b, c) = (a ^ b ^ c, majority)`: the #391 `flip` and
  #389 `carry_save3` laws, moved unchanged into `simd_masking_ops` and
  re-exported. A `u64` is one significance column of 64 independent lanes;
  the carry belongs one column up in the same lane.
- The in-flight exponent change: `pow2n_from_int` no longer goes through
  `to_array`. It clamps `n` to [-126, 127], adds `2^23 + 127` so the biased
  exponent sits in the mantissa field, and shifts the bits left by 23. That
  needed `U32x16 << U32x16` on every backend; it existed only on AVX-512
  and scalar. Lowering, timings and disable runs are in #402.

## Candidate C: `Abacus<BITS>` (lab only, `examples/u9_domino/domino.rs`)

- F's eager ripple, built from `half_add_mask`.
- Each of the 64 lanes is an independent unsigned counter of `BITS` bits.
  `1 <= BITS <= 122`, so a lane value (< 2^122) and the 64-lane total
  (< 2^128) are both exact in `u128`.
- `try_add_at(significance, lanes)` is transactional. On overflow the
  touched columns are restored by recomputing the carry chain forward from
  the input mask: `old_k = new_k ^ m_k`, `m_{k+1} = old_k & m_k`. The
  success path keeps no undo log. A significance at or past `BITS` is
  refused without touching anything.
- F (`Eager`) is NOT transactional: on overflow its lower columns stay
  updated (its doc says so). C is the first counter in the fixtures that
  refuses without side effects.

Tests (`tests/u9_domino.rs`, 6 new):
- lanes 0/31/32/63 alone;
- carry chains of 1, 2, 8, 16, 32 and 64 columns in those lanes;
- repeated carries into one column add up;
- refusal on overflow and on an out-of-range significance leaves the
  counter unchanged;
- 24,000 random steps against 64 `u128` counters, with zero, all-ones,
  alternating, single-lane and random masks, refusals included
  (accepted > 5000 and refused > 1000 asserted);
- reversed traversal gives identical columns.

Disable runs, each red: C1 no rollback; C2 rollback stops after one
column; C3 no significance range check; C4 carry moved to the neighbouring
lane.

## Cost gate (ns per add, 4096 adds, median of 2001 runs; range over 3 processes)

| tier | workload | A native per-lane | B = F eager | C abacus |
|---|---|---|---|---|
| AVX-512 VNNI | dense all-ones | 8.79 – 9.08 | 2.46 – 2.66 | 2.27 – 2.35 |
| | random masks | 8.78 – 8.80 | 7.79 – 8.05 | 8.34 – 8.80 |
| | sparse one lane | 8.02 – 9.08 | 2.14 – 2.44 | 2.21 – 3.90 |
| | weighted (2^0/16/17/32) | 8.79 – 9.39 | 6.99 – 7.69 | 7.27 – 9.96 |
| AVX-512, no VNNI | dense | 8.01 – 8.78 | 2.39 – 2.45 | 2.21 – 2.22 |
| | random | 8.02 – 9.39 | 7.56 – 8.50 | 6.88 – 7.84 |
| | sparse | 8.78 – 8.79 | 2.30 – 2.35 | 3.72 – 5.26 |
| | weighted | 8.78 – 9.09 | 7.09 – 8.24 | 6.89 – 7.19 |
| AVX2 | dense | 9.04 – 9.92 | 2.25 – 2.52 | 2.33 – 2.43 |
| | random | 9.69 – 10.23 | 7.22 – 8.32 | 8.53 – 10.57 |
| | sparse | 9.91 – 10.33 | 2.09 – 2.36 | 2.29 – 3.42 |
| | weighted | 9.91 – 11.59 | 7.39 – 7.59 | 8.08 – 8.21 |

Reading it:
- Both bit-plane counters beat 64 per-lane adds by about 4x on dense and
  single-lane masks and are within 0.75 – 1.1x on random masks, the same
  shape #391 found.
- C costs the same as F on dense masks. On sparse masks it is slower and
  noisier (most visibly 3.7 – 5.3 vs 2.3 ns on AVX-512 without VNNI),
  although the rollback never runs in these rows. Not diagnosed further.
- Not timed: AVX, SSE2 and NEON (qemu only here) and wasm (node). The
  exactness tests run on every tier; timings from emulators mean nothing.
- D (U9 byte-dot) and E (vertical masked byte sum) are the existing
  kernels for their consumers (#388, #390, #393). The counter is not a
  replacement for either and was not imposed on them.

## Verdicts

- **PR B (public `BinaryAbacus64`): DEFER.** Semantics are settled and
  tested, and the cost matches the eager domino. What is missing is a real
  consumer: nothing in ndarray, and no lance-graph caller found, counts
  64-lane masks at chosen significances. A public type waits for one. C
  stays in the fixture as the reference.
- **PR C (facet bit-plane helper): DEFER, not built.** The eight planes of
  a `U8x64` are already one AND and one `cmpeq_mask` each on existing
  facade words (`examples/u9_abacus/abacus.rs::planes`). #389 measured
  plane preparation at 3.5 ns per block for both operands on VNNI against
  the full B kernel at 2.9 ns, and reuse never broke even. There are not
  two actual consumers of prepared planes to amortize across. A helper
  would duplicate an existing two-word expression.
- Presence stays with the caller. The counters carry no presence; a lane
  that was never touched and a lane that summed to zero both read 0, so a
  consumer that needs the distinction keeps its own presence plane (as
  `lut9` and Codec9 do).
