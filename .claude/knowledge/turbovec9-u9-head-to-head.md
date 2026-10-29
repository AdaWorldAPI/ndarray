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

**B, U8 x I8 byte-dot, in every row measured.** On AVX-512 VNNI: 3.03 ns per
64 values end to end against 7.78 for the best A, and 2.69 - 2.77 against
6.95 - 7.14 in the scan form. A pays for widening u8 -> u16 -> f32, because
the facade has no u16 -> i32 widening to add a signed product in integers.

## Encoding rule: AND, XOR and the presence bit, never a shuffle

No magnitude is ever negated. The sign lives in the presence plane and is
flipped with XOR; kernels apply it as a mask.

- `negative = plane_a ^ plane_b` (XOR flips the presence bit).
- B splits `|b|` with AND and XOR: `low7 = |b| & 0x7F`, `top = |b| ^ low7`
  (= `|b| & 0x80`). Read as signed bytes, `low7` is `|b| & 127` and `top` is
  -128 where bit 7 is set, so `|a| * |b| = dot(|a|, low7) - dot(|a|, top)`.
- The masked byte-dot (`dot4_u8_i8_add_masked`, #387) ANDs `|a|` with
  `!negative` or `negative`: four byte-dots per block,
  `(pos_low7 - pos_top) - (neg_low7 - neg_top)`.
- A routes `|a|` with `U8x64::mask_blend` by the same XOR plane.
- Lane totals leave through a stored array read lane by lane (`black_box`
  keeps the compiler from rebuilding a permute reduction).

Disassembly of the timed closures (AVX-512 VNNI build): no shuffle, permute,
pack, unpack, align, extract or insert instruction in any B row, in either
baseline, or in any flush. A's rows keep `vpmovzxbw`/`vpmovzxwd` and the
upper-half `vextracti64x4`, which the facade's `U16x32::from_u8x64_hi` and
`to_f32x16_*` need to widen; A cannot be written without them.

## Candidates as built

| | product | sign | accumulate | flush |
|---|---|---|---|---|
| A | `\|a\| * \|b\|` in f32 after `u8 -> u16 -> f32` (or `U16x32::mullo` then `-> f32`, A') | `\|a\|` routed into a positive or a negative sum by `plane_a ^ plane_b` | `F32x16::mul_add` per quarter, two sums | every 256 blocks (lane sums below 2^24, exact in f32), then `i64` |
| B | four `dot4_u8_i8_add_masked`: `\|a\|` against `low7` and `top` of `\|b\|` | mask `!negative` / `negative` from `plane_a ^ plane_b` | four `I32x16` accumulators | every 16_384 blocks (a lane moves at most 130_560 per block), then `i64` |
| B scan | as B, with `low7` / `top` of the query built once | mask from `plane_x ^ plane_q` | as B | as B |
| A scan | query widened to f32 once | `\|x\|` routed by `plane_x ^ plane_q` | as A | as A |

Every row also runs `check`, which refuses a block holding the DR512
never-touched codeword (plane 0, magnitude 0) in either operand, inside the
timed region, for both candidates alike.

## Correctness

One independent `i64` oracle (checked arithmetic) for both candidates:

- all 511 x 511 numeric pairs, each alone in a block at a rotating lane,
  through A, A', B and both scan forms;
- full blocks of 64 mixed pairs (catches any lane reordering between the
  operands);
- worst-case runs (all magnitudes 255, same and mixed signs) of 49_157 blocks,
  past both flush intervals, with totals beyond the `i32` range;
- never-touched refused in either operand, touched +0 and negative nonzero
  magnitudes accepted, the oracle refusing the same lanes.

Passed on native AVX-512 VNNI, config-v4, config-v3, config-avx (qemu),
config-v2 (qemu), aarch64 (qemu) and nightly-simd. Disable runs, each red:
A's routing swapped; A's flush at 4096 blocks (f32 loses exactness past
2^24); B's top limb added instead of subtracted; B's AND mask keeping bit 7;
B's presence plane not flipped by b; B scan ignoring the query plane; B's
flush at 2^20 blocks (i32 wraps); `check` ignoring operand b; the oracle
accepting never-touched.

## Measurements (ns per 64 values, ranges over 3 runs, this host)

One Xeon with AVX-512 VNNI. Each tier is a separate build run natively on it
(`--config .cargo/config-v4.toml`, `config-v3.toml`); nothing is projected.
L1 = 64 blocks per operand (4.5 KiB) dotted repeatedly; streamed = 2^20 blocks
per operand (72 MiB). Baseline = plain `dot4_u8_i8_add`; the scan rows compare
with the plain byte-dot against one fixed query.

| L1-resident | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| baseline: plain u8 x i8 dot4 | 1.89 - 1.94 | 1.97 - 2.02 | 2.13 - 2.16 |
| A end-to-end (check, route, widen, mul_add) | 8.59 - 8.80 | 8.77 - 8.80 | 74.0 - 99.7 |
| A' end-to-end (check, route, mullo, add) | 7.78 - 7.80 | 7.77 - 7.81 | 75.0 - 82.4 |
| **B end-to-end (check, AND/XOR split, 4 masked dot4)** | **3.03 - 3.05** | **5.67 - 6.22** | **9.88 - 10.4** |
| A arithmetic only | 7.31 | 7.28 - 7.36 | 13.7 - 14.9 |
| A' arithmetic only | 6.17 - 6.20 | 6.00 - 6.19 | 10.7 - 11.0 |
| **B arithmetic only** | **2.55 - 2.61** | **5.05 - 5.58** | **9.28 - 9.30** |
| B operand preparation alone (AND/XOR split) | 0.86 - 0.88 | 0.94 | 1.14 - 1.16 |
| baseline scan: plain dot4, fixed query | 1.86 - 1.91 | 2.08 - 2.16 | 1.59 - 1.63 |
| A scan | 6.95 - 7.14 | 6.97 - 7.14 | 70.1 - 76.8 |
| **B scan** | **2.69 - 2.77** | **4.81 - 4.95** | **9.00 - 9.23** |

| Streamed | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| baseline | 7.1 - 7.4 | 8.8 - 9.5 | 10.3 - 13.6 |
| A' end-to-end | 18.8 - 21.4 | 18.3 - 21.3 | 420 - 435 |
| **B end-to-end** | **12.9 - 16.3** | **17.0 - 18.8** | **20.8 - 23.6** |
| baseline scan | 4.7 - 4.9 | 5.4 - 5.8 | 6.2 - 7.2 |
| A scan | 9.3 - 12.1 | 9.2 - 11.6 | 419 - 420 |
| **B scan** | **5.9 - 6.9** | **7.2 - 10.7** | **11.1 - 11.7** |

The streamed "arithmetic only" rows are left out: they read precomputed
operands, which doubles the bytes streamed, so they do not isolate arithmetic.

Notes on the numbers:

- "Arithmetic only" receives each candidate's execution-form operands
  prepared before the timer (A: routed `|a|`; B: the XOR plane and the
  `low7` / `top` bytes). The end-to-end and scan rows pay all preparation
  inside the timer.
- Keeping the sign out of the bytes costs B four byte-dots per block instead
  of two: 2.55 - 2.61 ns against 1.89 for a single byte-dot on VNNI.
- On AVX2 every A row is dominated by `U8x64::mask_blend`, whose AVX2
  realization is a per-lane loop; B does not use it (the masked byte-dot
  builds its mask in registers).
- 4096 values per L1 dot: a full dot product costs 64 x the per-64 figure.

## The one facade gap

B needs `low7` and `top`, which are `U8x64`, as the `I8x64` operand of the
byte-dot. The facade has no `U8x64 -> I8x64` view, so `kernels::bits_as_i8`
stores the 64 bytes and loads them back as `i8` (every bit kept; no shuffle,
no arithmetic). Its cost is inside "B operand preparation" (0.86 - 1.16 ns per
block with the AND/XOR split). Exact consumer need: a bit view
`U8x64 -> I8x64` (or a byte-dot taking the signed operand as raw bytes).
Recorded, not added: it is a u8/i8 crossing, which the repository rules keep
out of the facade unless decided deliberately.

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
   `low7 = |q| & 0x7F`, `top = |q| ^ low7`, viewed as `I8x64`
   (`kernels::QueryB::new`); the query sign stays in its plane. Per index
   block of 64 dims: `negative = index_plane ^ query_plane`, then four
   `I32x16::dot4_u8_i8_add_masked(index_mag, !negative or negative, low7 or
   top)`; flush into `i64` at most every 16_384 blocks (1,048,576 dims):
   `(pos_low7 - pos_top) - (neg_low7 - neg_top)`.
2. **Inputs:** index stored as 64 `u8` magnitudes plus one `u64` positive
   plane per 64 dims, passed as `U8x64` and `u64` with no conversion and no
   negation. A query that is only 7-bit (|q| <= 127) has `top = 0`; its two
   `top` byte-dots can be dropped, leaving the #387 masked pair.
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
