# D-SIMD-MICRO-MACRO-1, step 1: the packed microtile oracle

Lab fixture only: `examples/microtile/` (realizations + cost gate) and
`tests/microtile.rs` (exactness). No public type, no backend change, no
production caller. Steps 2+ (binding to a KPEA method, Thought-0 delta reuse,
interlacing) are not started.

## What the microtile is

One 128-bit tile carries several isolated arithmetic fields:

| layout | fields | payload | other bits |
|---|---|---|---|
| `U24x5` | 5 × 24 bit (`U24`/`I24`) | 120 bits | 8 metadata bits, 120..128 |
| `U9x14` | 14 × 9 bit (`U9`/`I9`) | 126 bits | 2 spare bits, 126..128 |

Operations: field-isolated wrapping add and sub, per-field unsigned carry and
borrow out, per-field signed overflow, an intra-tile stencil
`f[j-1] + 2 f[j] + f[j+1]`, and an exact fold (unsigned and signed per-field
sums over many tiles). Every result keeps the left operand's metadata.

Isolation is the SWAR law: clear each field's top bit in the addends so a
carry stops at its own top bit, then fix that bit with
`(a ^ b) & HIGH`; for sub, set each top bit of `a` so a borrow stops there.
Signed fold: `Σ sext(f) = Σ f - 2^W · #(top bit set)`, so one pass yields
both sums.

## Four realizations, one oracle

The oracle unpacks every field and does the arithmetic in `u64`/`i64`.

1. **u128**: the tile as one 128-bit integer.
2. **SoA `U64x8`**: 8 tiles per op, the tile split into a low and a high
   64-bit plane. The fields that straddle bit 64 (`U24` field 2, `U9` field
   7) take an explicit carry/borrow out of bit 63, computed with lane-wise
   logic only (no shuffle).
3. **Widened `U32x16`**: field-major `u32` lanes. The fold flushes to `u64`
   every `floor((2^32-1)/(2^W-1))` steps (256 for `U24`).
4. **Bit planes**: one bit per tile, 64 tiles per word, add by ripple of
   `simd::full_add_mask`, fold by `Σ_k 2^k popcount(plane_k)`.

Exactness: 9 tests over random tiles plus every pairing of edge field values
(0, 1, sign bit - 1, sign bit, max - 1, max), the two straddling fields'
cross-word carry and borrow, 8192 maximal `U24` values in one `u32` lane, and
metadata against an all-ones right operand. Green on AVX-512, AVX2,
AVX-without-AVX2 (qemu SandyBridge), x86-64-v2 (qemu Nehalem), NEON (qemu
aarch64) and nightly `core::simd`.

Disable runs, each red: top bit not cleared in add (D1) or set in sub (D8),
no carry (D2) or borrow (D7) across bit 64, metadata dropped (D3), signed
fold without the top-bit correction (D4) and its plane form (D9), widened
fold never flushing (D5), carry-out as `a & b` only (D10), straddling field
read from the low word only (D11), fold tail skipped (D12), widened add
unmasked (D13). D6 (an extra mask on the stencil's left shift) stayed green:
`add` already masks its right operand, so the mask was removed rather than
kept as a guard that does not bind.

## Cost gate

`cargo run --release --example microtile --features std`, median ns per tile
over 31 runs, three runs (two AVX-512 v4, one AVX2 v3) on a 4-core host.
4096 tiles = 64 KB per operand (L2-resident); 65536 tiles = 1 MB.

| op | layout | u128 | SoA U64x8 | widened | planes |
|---|---|---|---|---|---|
| add, 4K | U24x5 | 2.00–2.10 | 0.95–1.73 | **0.69–0.71** | 1.87–1.93 |
| add, 4K | U9x14 | 2.03–2.10 | **1.08–1.66** | 1.92–1.97 | 1.83–2.03 |
| add, 64K | U24x5 | 2.29–2.31* | 1.94–3.16 | 2.18–2.35 | **1.88–2.07** |
| add, 64K | U9x14 | 2.04–2.41 | 1.54–2.81 | 6.19–6.39 | **1.87–2.06** |
| fold, 4K | U24x5 | — | 0.60–1.01 | **0.29–0.30** | 0.85–0.90 |
| fold, 4K | U9x14 | — | 2.71–3.60 | 0.76–1.20 | **0.73–1.42** |
| fold, 64K | U24x5 | — | 0.53–0.93 | **0.28–0.31** | 0.75–0.82 |
| fold, 64K | U9x14 | — | 2.81–3.22 | 1.85–2.16 | **0.60–0.76** |

\* one outlier at 4.19 excluded.

Storage: packed and planes 16 B/tile; widened 20 B/tile (`U24x5`) and
56 B/tile (`U9x14`).

Read:

- **No realization wins everything, so packed is not assumed.** Wide fields
  (`U24x5`) favour widened lanes for both add and fold: only 25% more bytes,
  and one native add per field. Narrow fields (`U9x14`) favour packed SoA for
  cache-resident add and bit planes for the fold; widened costs 3.5× the
  bytes and is 3× slower once the data leaves L2.
- **The single u128 is never the fast form.** It runs at about 2 ns/tile
  whatever the layout: rustc does not vectorize the two-word add. The packed
  win comes from the SoA split, which puts 8 tiles in each register.
- **At 1 MB every add is about 2 ns/tile** except widened `U9x14`: the
  operation is bandwidth-bound and the bytes per tile decide.
- **Bit planes earn their place on folds, not adds.** A narrow-field fold is
  W popcounts per 64 tiles; their add needs W full adders and is no faster
  than packed.
- Unexplained: SoA add is faster on AVX2 (0.95–1.08 at 4K) than on AVX-512
  (1.61–1.73). Not diagnosed.

## Findings outside the fixture

- **NEON and wasm `U32x16` lack most of the facade:** no `&`, `|`, `-`,
  `!`, `+=`, `from_slice` or `copy_to_slice`; they implement `+`, `^`, the
  two shifts and `from_array`/`to_array`. AVX-512, AVX2, scalar and nightly
  have the full set. The fixture works around it (mask by `<< k >> k`, load
  through `from_array`), so this is recorded, not fixed here. It is a real
  parity gap for any consumer that builds on `U32x16`.

## What this does not show yet

- No consumer: the gates for binding to a KPEA mask/count method (step 2),
  Thought-0 invariant reuse across thought deltas (step 3) and interlacing
  (step 4) are open. The fold here is the candidate primitive for step 2.
- Pack/unpack between layouts is untimed; a method that converts would pay it.
- Only x86 was timed; NEON and the other tiers ran under emulation for
  exactness only.
