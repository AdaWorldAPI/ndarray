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

**B, U8 x I8 byte-dot, in every row measured.** On AVX-512 VNNI: 3.05 ns per
64 values end to end against 7.64 - 7.83 for the best A, and 2.75 - 2.81
against 7.02 - 7.19 in the scan form. A, as composed here, widens u8 -> u16 -> f32,
because the facade offers no u16 -> i32 widening to add the products in
integers. That is a cost of this composition, not of U8 x U8 multiplication.

## The arithmetic contract both candidates implement

> Sign is a routing decision. Multiplication is an exact decomposition.
> Accumulation is a fold.

A value is an unsigned magnitude `m` in 0..=255 plus a polarity bit (Codec9:
`P=1, M=0` is numeric zero, `P=0, M=0` is never-touched and is refused). For
one lane pair:

```text
p        = m_a * m_b                  exact, always fits u16 (255 * 255 = 65_025)
negative = polarity_a XOR polarity_b  the sign of the product
acc      = acc - p   if negative
           acc + p   otherwise
```

The magnitudes are never negated or rewritten. The polarity XOR only picks
which side of the fold the product joins; one subtraction per flush applies
it.

A and B are two facade compositions of this one contract; neither has a
different numerical meaning, and neither is "more exact" than the other.
They differ only in how the product is broken into pieces the facade can
execute:

- **A** forms `m_a * m_b` whole (in `f32`, or in `u16` with `mullo`) and adds
  it to the positive or the negative sum.
- **B** does it like long multiplication (schriftliche Multiplikation): it
  splits `m_b` into two weighted digits the byte-dot can consume, and adds
  their partial products. With `low7 = m_b & 0x7F` and `top = m_b ^ low7`
  read as a signed byte (0 or -128), `m_b = low7 - top`, so
  `m_a * m_b = m_a * low7 - m_a * top`. For `255 x 255`:
  `255 * 127 - 255 * (-128) = 32_385 + 32_640 = 65_025`.

In a scan the query side of the decomposition (`low7`, `top`, its plane) is
prepared once and reused for every index block; only the routing XOR and the
partial products run per block.

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
`to_f32x16_*` use to widen. That is how this composition of A lowers through
the current facade; it says nothing about U8 x U8 multiplication itself, nor
about other possible lowerings of A, which were not explored.

**This holds on AVX-512 only (checked later, in #389).** On an AVX2 build
(`config-v3`) the masked byte-dot that B calls contains 4 `vpshufb`: LLVM
uses them to spread the bytes of the 64-bit keep mask across the register
inside `dot4_u8_i8_add_masked`'s AVX2 realization (the source uses a
variable shift, a multiply and a shift, not a shuffle). That is the facade's
realization, not this probe's code, and it was not changed here.

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
  magnitudes accepted, the oracle refusing the same lanes;
- the split itself, through the kernel's own `prepare_b`, for every magnitude
  0..=255: `low7 XOR top` gives back the original byte, `low7 - top` gives
  back its value with `top` in {0, -128}, and changing either polarity bit
  changes only the routing plane (`plane_a ^ plane_b`), never `low7` or `top`.

Passed on native AVX-512 VNNI, config-v4, config-v3, config-avx (qemu),
config-v2 (qemu), aarch64 (qemu) and nightly-simd. Disable runs, each red:
A's routing swapped; A's flush at 4096 blocks (f32 loses exactness past
2^24); B's top limb added instead of subtracted; B's AND mask keeping bit 7;
B's presence plane not flipped by b; B scan ignoring the query plane; B's
flush at 2^20 blocks (i32 wraps); `check` ignoring operand b; the oracle
accepting never-touched; `top` keeping all bits; the plane formed with AND
instead of XOR; polarity leaking into the split bytes.

CI runs this test file only in the test jobs on its own x86 runners; the
other tiers above were run locally, not in CI.

### Exactness domain

Both candidates are bit-exact, with no rounding, saturation or wrap, for:

- operands from the 511 numeric values -255..=255 (never-touched refused by
  `check`, which callers run before the kernel; the kernels do not check);
- any number of blocks whose running total fits `i64`, which at the worst case
  of 64 * 65_025 per block is about 2.2 * 10^12 blocks. The final `i64` adds are
  unchecked.

Inside that domain each stage is bounded:

- **A:** a lane gains one product, at most 65_025, per block. After 256 blocks
  that is 16_646_400, below 2^24 = 16_777_216, where every `f32` integer
  is exact; then A flushes.
- **B:** an `i32` lane gains at most 4 * 255 * 128 = 130_560 per block. After
  16_384 blocks that is 2_139_095_040, below `i32::MAX` = 2_147_483_647; then
  B flushes. The byte-dot does not saturate: the worst-case run (every byte
  255, so `top = -128` in all four bytes of a lane) passes on every tier.
- The worst-case test runs past both limits.

### Not proven here

- Exactness beyond the `i64` total above.
- Any lowering of A other than the two measured (f32 `mul_add`, `u16 mullo`).
- The CI matrix running this test on every tier.
- Anything about Codebook512 / Lloyd-Max512 decoding or retrieval parity (see
  the handoff).

## Measurements (ns per 64 values, ranges over 3 runs, this host)

One Xeon with AVX-512 VNNI. Each tier is a separate build run natively on it
(`--config .cargo/config-v4.toml`, `config-v3.toml`); nothing is projected.
L1 = 64 blocks per operand (4.5 KiB) dotted repeatedly; streamed = 2^20 blocks
per operand (72 MiB). Baseline = plain `dot4_u8_i8_add`; the scan rows compare
with the plain byte-dot against one fixed query.

Timed input layout: each operand is two compact arrays, 64 magnitude bytes and
one 8-byte plane per block (72 bytes), and each `Block` is built in registers
inside the loop. A `Vec<Block>` would pad every block to 128 bytes, because
`U8x64` is 64-byte aligned; the first version of this probe streamed that
padded layout and its streamed rows were inflated by it (B end-to-end 12.9 -
16.3, B scan 5.9 - 6.9 on VNNI). The baseline streams 64 + 64 bytes per block
(no planes), B and A 72 + 72, and a scan only the index side.

| L1-resident | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| baseline: plain u8 x i8 dot4 | 1.94 | 2.02 - 2.03 | 2.11 - 2.56 |
| A end-to-end (check, route, widen, mul_add) | 8.78 - 8.88 | 8.70 - 8.89 | 74.3 - 77.2 |
| A' end-to-end (check, route, mullo, add) | 7.64 - 7.83 | 7.64 - 8.98 | 75.1 - 81.4 |
| **B end-to-end (check, AND/XOR split, 4 masked dot4)** | **3.05 - 3.06** | **5.61 - 7.23** | **10.1 - 10.2** |
| A arithmetic only | 7.30 - 7.31 | 7.33 - 8.14 | 13.7 - 13.8 |
| A' arithmetic only | 6.16 - 6.22 | 6.16 - 7.69 | 10.7 |
| **B arithmetic only** | **2.59** | **5.05 - 6.17** | **9.31** |
| B operand preparation alone (AND/XOR split) | 1.16 | 1.16 - 1.80 | 1.25 - 1.27 |
| baseline scan: plain dot4, fixed query | 1.88 - 1.92 | 2.09 - 2.27 | 1.59 - 1.61 |
| A scan | 7.02 - 7.19 | 7.05 - 7.39 | 76.5 - 83.5 |
| **B scan** | **2.75 - 2.81** | **4.91 - 4.95** | **9.27 - 9.28** |

| Streamed | AVX-512 VNNI | AVX-512, no VNNI | AVX2 |
|---|---|---|---|
| baseline | 4.4 - 4.8 | 5.5 - 6.7 | 6.1 - 8.9 |
| A end-to-end | 22.4 - 25.7 | 22.2 - 25.0 | 421 - 426 |
| A' end-to-end | 16.6 - 19.6 | 18.2 - 19.8 | 420 - 427 |
| **B end-to-end** | **6.0 - 6.5** | **9.0 - 12.5** | **26.9 - 30.7** |
| B operand preparation alone | 2.77 - 2.80 | 2.77 - 2.83 | 2.94 - 3.11 |
| baseline scan | 2.23 - 2.30 | 2.46 - 2.52 | 2.60 - 2.95 |
| A scan | 10.6 - 12.0 | 10.7 - 11.2 | 422 - 424 |
| **B scan** | **2.75 - 2.80** | **4.68 - 4.75** | **27.6 - 32.6** |

The streamed "arithmetic only" rows are left out: they read precomputed
operands (B: one plane plus two 64-byte vectors per block), which more than
doubles the bytes streamed, so they do not isolate arithmetic.

Notes on the numbers:

- "Arithmetic only" receives each candidate's execution-form operands
  prepared before the timer (A: routed `|a|`; B: the XOR plane and the
  `low7` / `top` bytes). The end-to-end and scan rows pay all preparation
  inside the timer.
- "B operand preparation alone" passes the whole prepared tuple (plane,
  `low7`, `top`) through `black_box` on every block, so no lane can be
  dropped as unused; the row includes storing that tuple. The first version
  consumed one lane of each vector and read 0.86 - 1.16.
- Keeping the sign out of the bytes costs B four byte-dots per block instead
  of two: 2.59 ns against 1.94 for a single byte-dot on VNNI.
- Streamed on VNNI the B scan (2.75 - 2.80) equals its L1 figure: one
  index-side block per 64 values does not saturate memory bandwidth, so the
  scan stays compute-bound.
- On AVX2 every A row is dominated by `U8x64::mask_blend`, whose AVX2
  realization is a per-lane loop; B does not use it (the masked byte-dot
  builds its mask in registers). Streamed AVX2 B rows (27 - 33) are slower
  than L1 (9 - 10) by more than the extra bytes explain; not investigated.
- 4096 values per L1 dot: a full dot product costs 64 x the per-64 figure.

## The one facade gap

B needs `low7` and `top`, which are `U8x64`, as the `I8x64` operand of the
byte-dot. The facade has no `U8x64 -> I8x64` view, so `kernels::bits_as_i8`
stores the 64 bytes and loads them back as `i8` (every bit kept; no shuffle,
no arithmetic). Its cost is inside "B operand preparation" (1.16 - 1.27 ns per
block with the AND/XOR split, L1). Exact consumer need: a bit view
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
