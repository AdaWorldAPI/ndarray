# TurboVec-U9: U8 x U8 x sign x mul_add against U8 x I8 byte-dot

READ BY: savant-architect, vector-synthesis, simd-savant, the TurboVec-U9
owner. Companion to `turbovec9-u9-substrate.md` (#387).

Probe: `examples/u9_head_to_head/` (`kernels.rs` holds both candidates,
`main.rs` times them). Correctness: `tests/u9_head_to_head.rs`. Both use only
`ndarray::simd` facade words; no backend file changed.

```sh
cargo run --release --example u9_head_to_head --features std
cargo test --release --test u9_head_to_head
```

## The question and the answer

For full-range signed 9-bit arithmetic (-255..=255) on the shared 8+1 block
(64 `u8` magnitudes + one `u64` positive plane per operand), which composes
the cheaper bit-exact multiply-accumulate through the existing facade?

**B, U8 x I8 byte-dot.** Its arithmetic costs the same as a plain byte-dot
(1.0x on AVX-512 VNNI); A's costs 3.3 - 3.9x, because A has to widen to f32 to
add a signed product without a u16 -> i32 widening in the facade. In the form
a search actually runs (query prepared once, index blocks streamed), B wins on
every tier measured: 2.2 ns against 6.5 ns per 64 values on AVX-512 VNNI.

A wins one row: both operands' signs resolved per block (the symmetric
"end-to-end" row), and only because B's operand preparation has to go through
an f32 detour (see the gap below). That row does not occur in a TurboVec scan.

## Candidates as built

| | product | sign | accumulate | flush |
|---|---|---|---|---|
| A | `\|a\| * \|b\|` in f32 after `u8 -> u16 -> f32` (or `U16x32::mullo` then `-> f32`, A') | `\|a\|` routed into a positive or a negative sum by `plane_a ^ plane_b` (`U8x64::mask_blend`) | `F32x16::mul_add` per quarter, two sums | every 256 blocks (lane sums stay below 2^24, exact in f32), then `cast_i32` + `i64` |
| B | `dot4_u8_i8_add(\|a\|, ±(\|b\| & 127))` and `dot4_u8_i8_add(\|a\|, ±(\|b\| >> 7))` | folded into the signed bytes of b | two `I32x16` byte-dot accumulators, result `lo + 128 * hi` | every 1024 blocks (`reduce_sum` of 16 lanes stays in `i32`), then `i64` |
| B scan | same split, query side only | query sign in its bytes, index sign routed by `dot4_u8_i8_add_masked` (#387) | four masked byte-dots, `(pos_lo - neg_lo) + 128 * (pos_hi - neg_hi)` | every 1024 blocks |
| A scan | query widened to f32 once | index `\|x\|` routed by `plane_x ^ plane_q` | as A | as A |

Every row also runs `check`, which refuses a block holding the DR512
never-touched codeword (plane 0, magnitude 0) in either operand, inside the
timed region, for both candidates alike.

## Correctness

One independent `i64` oracle (checked arithmetic) for both candidates:

- all 511 x 511 numeric pairs, each alone in a block at a rotating lane,
  through A, A', B and both scan forms;
- full blocks of 64 mixed pairs (catches any lane reordering between the
  operands);
- worst-case runs (all magnitudes 255, same and mixed signs) of 3077 blocks,
  past both flush intervals, with totals beyond the `i32` range;
- never-touched refused in either operand, touched +0 and negative nonzero
  magnitudes accepted, the oracle refusing the same lanes.

Passed on native AVX-512 VNNI, config-v4, config-v3, config-avx (qemu),
config-v2 (qemu), aarch64 (qemu) and nightly-simd. Disable runs, each red:
B's high limb weighted 64 instead of 128; A's routing swapped; A's flush at
4096 blocks (f32 loses exactness past 2^24); B's flush at 2^20 blocks (i32
wraps); `check` ignoring operand b; the oracle accepting never-touched; B
scan's index masks swapped; B scan's query sign inverted.

## Measurements (ns per 64 values, ranges over 3 runs, this host)

One Xeon with AVX-512 VNNI. Each tier is a separate build run natively on it
(`--config .cargo/config-v4.toml`, `config-v3.toml`); nothing is projected.
L1 = 64 blocks per operand (4.5 KiB) dotted repeatedly; streamed = 2^20 blocks
per operand (72 MiB). "x" is relative to the baseline in the same section
(plain `dot4_u8_i8_add`; the scan rows to the plain scan baseline).

| L1-resident | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| baseline: plain u8 x i8 dot4 | 1.78 - 1.83 | 1.92 - 1.97 | 2.09 - 2.13 |
| A end-to-end (check, route, widen, mul_add) | 8.70 - 8.72 | 8.48 - 8.70 | 73.0 - 74.8 |
| A' end-to-end (check, route, mullo, add) | 7.70 - 7.72 | 7.69 - 7.73 | 77.6 - 82.6 |
| B end-to-end (check, split, sign, to i8, 2 dot4) | 12.8 - 13.1 | 14.4 - 17.1 | 148 - 149 |
| A arithmetic only | 6.94 - 7.13 | 7.22 - 7.77 | 13.8 - 14.1 |
| A' arithmetic only | 6.06 - 6.09 | 6.06 - 6.91 | 10.7 - 11.0 |
| **B arithmetic only** | **1.89** | **2.70 - 3.14** | **3.42 - 3.52** |
| B operand preparation alone | 5.23 | 5.23 - 5.41 | 35.8 - 37.4 |
| baseline scan: plain dot4, fixed query | 1.84 | 1.44 - 1.58 | 1.55 - 1.59 |
| A scan | 6.50 - 6.61 | 6.44 - 7.38 | 75.8 - 76.6 |
| **B scan** | **2.19 - 2.23** | **4.25 - 4.86** | **10.6 - 10.9** |

| Streamed | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| baseline | 6.8 - 10.1 | 8.4 - 12.3 | 12.2 - 16.6 |
| A' end-to-end | 18.0 - 22.0 | 17.0 - 25.4 | 422 - 425 |
| B end-to-end | 21.2 - 24.7 | 23.3 - 28.7 | 328 - 331 |
| baseline scan | 4.1 - 5.5 | 5.4 - 5.9 | 9.9 - 14.6 |
| A scan | 8.2 - 14.7 | 10.4 - 12.2 | 420 - 422 |
| **B scan** | **5.4 - 6.4** | **8.1 - 10.0** | **26.1 - 28.3** |

The streamed "arithmetic only" rows are left out: they read precomputed
operands, which doubles the bytes streamed, so they do not isolate arithmetic.

Notes on the numbers:

- "Arithmetic only" receives each candidate's execution-form operands
  prepared before the timer (A: routed `|a|` planes; B: signed `i8` limbs).
  The end-to-end and scan rows pay all preparation inside the timer.
- On AVX2 every A row and the symmetric B row are dominated by facade words
  whose AVX2 realization is a per-lane loop (`U8x64::mask_blend` for A's
  routing, the array-backed f32/narrow chain for B's preparation). B scan
  avoids both: `dot4_u8_i8_add_masked` builds its mask in registers.
- 4096 values per L1 dot: a full dot product costs 64 x the per-64 figure.

## The facade gap B pays for

B needs signed `I8x64` operands derived from the shared U8 storage. The facade
has no `U8x64 -> I8x64` conversion, so `kernels::signed_bytes` goes
u8 -> u16 -> f32 (times a sign factor) -> i32 -> i16 -> i8 through
`narrow_saturating_*`, which never saturates here. That costs 5.2 ns per block
on VNNI and 36 ns on AVX2, and is the whole reason B loses the symmetric row.

Exact consumer need, if the symmetric form ever matters: a value-preserving
conversion of a `U8x64` whose bytes are all `<= 127` into an `I8x64` (or a
`from_sign_magnitude(magnitude <= 127, negative: u64) -> I8x64`). It is not
needed for TurboVec search, where the I8 operand is the query and is prepared
once. Not added here; recorded as a gap, not worked around.

## What this means for TurboVec-U9

Two contracts share the 8+1 bits and are not the same thing:

- **Arithmetic511**: signed magnitude, 511 numeric values; `P=0, M=0` is DR512
  never-touched and is refused. This probe proves both candidates exact over
  exactly that domain.
- **Codebook512**: 512 centroid IDs, all valid, all distinguishable. 512 IDs
  cannot map injectively onto 511 numeric values, so a numeric U9 index has
  at most 511 distinct scores per dimension (or 512 patterns if the encoder
  is allowed to write `P=0, M=0` as a second zero, which is then a TurboVec
  contract, not DR512's, and must not go through `check`).

Numeric U9 scoring (B scan) computes `sum q_d * x_d` over integer grid values
exactly. It reproduces Lloyd-Max512 scoring only if every centroid is an exact
integer multiple of one scale; arbitrary Lloyd-Max centroids are not. The #387
probe measured snapping 512 Lloyd-Max centroids onto the 9-bit grid: 329
distinct values remain, 44.7 dB against 49.9 dB for the exact codebook, and a
uniform 9-bit grid with a tuned clip reaches 46.0 dB. Arithmetic parity here
says nothing about retrieval parity.

Exact Lloyd-Max512 scoring needs the centroid value of each ID: a 512-entry
decode. The facade offers that only as per-position table reads (`lut_u8`
shape, 38 - 40 ns per 64 IDs in #387); the `vpermi2b` form (about 6 ns) is not
a facade word. Neither cost is hidden in the B scan figure; it is extra.

## Handoff to the TurboVec-U9 owner

1. **Winning composition:** B scan. Per query block, once:
   `lo = ±(|q| & 127)`, `hi = ±(|q| >> 7)` as `I8x64`, signed by the query
   plane (`kernels::QueryB::new`). Per index block of 64 dims: four
   `I32x16::dot4_u8_i8_add_masked(index_mag, index_plane or !index_plane,
   lo or hi)`; flush with `reduce_sum` into `i64` at most every 1024 blocks
   (65,536 dims): `(pos_lo - neg_lo) + 128 * (pos_hi - neg_hi)`.
2. **Inputs:** index stored as 64 `u8` magnitudes plus one `u64` positive
   plane per 64 dims, passed as `U8x64` and `u64` with no conversion. A query
   that is only i8 (|q| <= 127, today's weights) needs only the `lo` pair:
   two masked byte-dots, the #387 cost (1.28 - 1.30 ns per 64 in its probe).
3. **Native numeric U9 index:** yes, the kernel consumes it directly, as long
   as the index holds numeric grid values (Arithmetic511), not centroid IDs.
4. **Lloyd-Max512:** needs a decode from ID to centroid, which is either
   lossy (snap to the grid; IDs collide) or a lookup per dimension per query
   (not amortized by this kernel, and not a fast facade word today).
5. **Unverified:** recall of any 9-bit index on real data; the clip and scale
   of the grid; end-to-end search cost with TurboVec's layout, blocking and
   rescoring; NEON, wasm and nightly timings (only their correctness ran,
   under qemu or without timing).
6. **Smallest falsifying experiment:** in the TurboVec fork, encode one real
   benchmark set twice: (a) numeric U9 on a uniform clipped grid, scored with
   B scan; (b) Lloyd-Max512 IDs scored exactly in f32 through a decode. Report
   recall@10 and ns per query for both against the current i4 path. The
   recommendation fails if (a) loses more recall to (b) than the operator
   accepts, or if (a)'s scan costs more per dimension than the i4 path by more
   than the agreed margin.
