# A4: G2 INT4×32 packed dot — 3 backends + parity test

Worker A4 of PR-X4 (W4-W5). Parallel after A1. **Hardware-tight:
this is the lowest-precision lane in the splat4d cascade.**

## Scope

Ship the INT4×32 packed dot product on three backends with a
cross-backend parity test. Consumed by `SplatCell<D>` for the
32-dim thinking_style and 16-dim qualia cell signatures.

## Packing layout

32 nibbles → 16 bytes. Convention (pinned to AVX-512 VNNI):

- **LHS**: signed `i4`, packed 2 per byte (low-nibble first). Treat
  as `i8` after sign-extension.
- **RHS**: unsigned `u4`, packed 2 per byte (low-nibble first).
  Treat as `u8` after zero-extension.
- **Accumulator**: `i32`, no saturation.

The signed × unsigned product fits in a single `vpdpbusd` lane,
which is the whole reason for this asymmetric convention.

Dequantisation: `output_f32 = (acc as f32) * scale_lhs * scale_rhs`
where the per-vector scales come from the cell's quantisation step
upstream of A4. The bundle's `dequant_f32` stage owns this.

## AVX-512 VNNI backend

```rust
// 32 nibbles in one i256, two vpdpbusd instructions
unsafe fn dot_i4x32_avx512vnni(lhs: __m128i, rhs: __m128i) -> i32 {
    let lhs8 = unpack_i4x32_to_i8x32(lhs);    // 1 vpunpcklbw + sign-extend
    let rhs8 = unpack_u4x32_to_u8x32(rhs);    // 1 vpunpcklbw + zero-extend
    let acc = _mm256_setzero_si256();
    let acc = _mm256_dpbusd_epi32(acc, rhs8, lhs8);  // vpdpbusd
    _mm256_reduce_add_epi32(acc)              // horizontal reduce
}
```

~2 `vpdpbusd` for 32-dim if we pack into 256-bit; 1 if `vpdpbusd`
takes 512-bit and we go full AVX-512. Spec line 152: "**2
instructions for 32-dim INT4**" — that's the unpack + dpbusd pair
in 512-bit-VNNI form.

## ARM NEON backend

`sdot` does 4-way × 8-lane = 32 dot products in one instruction, but
needs INT8 not INT4. Unpack INT4 → INT8 via `vshrq_n_s8` + masks (~4
instructions), then 8 × `sdot` for the full 32-dim dot — spec line
153: **8 instructions for 32-dim**.

## AMX caveat

AMX BF16 tile op handles INT8 not INT4 directly. No AMX path for A4
— it would require software unpacking that costs the same as VNNI.
Spec line 154: **skip**.

## Scalar fallback

```rust
fn dot_i4x32_scalar(lhs: &[u8; 16], rhs: &[u8; 16]) -> i32 {
    let mut acc: i32 = 0;
    for i in 0..16 {
        let lhs_lo = ((lhs[i] << 4) as i8) >> 4;  // sign-extend 4→8
        let lhs_hi = (lhs[i] as i8) >> 4;
        let rhs_lo = (rhs[i] & 0x0F) as i32;       // zero-extend
        let rhs_hi = ((rhs[i] >> 4) & 0x0F) as i32;
        acc += (lhs_lo as i32) * rhs_lo;
        acc += (lhs_hi as i32) * rhs_hi;
    }
    acc
}
```

32-way unrolled in the actual implementation; this shape is the
parity reference.

## B-Pack-Dot bundle compliance

From bundle table line 433: **B-Pack-Dot = `pack_int4x32 ∘
dot_i4x32_to_i32 ∘ dequant_f32`**. A4 ships the middle stage; the
outer two come from upstream and downstream. Consumers must call
the full bundle — reaching past into `dot_i4x32_to_i32` alone is
forbidden constraint #2.

If `pack_int4x32` or `dequant_f32` doesn't exist in `ndarray::simd`,
file the missing primitive against the vertical-simd-consumer-contract
BEFORE A4 spawns.

## Parity test

- **Fixture**: 10K random INT4×32 LHS/RHS pairs, seed
  `0xA4_D0_T_1NT4`. LHS sampled from `[-8, 7]` per nibble, RHS from
  `[0, 15]`.
- **Assertion**: all 3 backends produce the **same `i32`** (integer
  dot is exact — no ULP tolerance). Mismatch fails the gate.

## Consumers

- `SplatCell<32>` (thinking_style): one dot per cell per cascade tier
- `SplatCell<16>` (qualia): one dot per cell, half the lane width
- A3's SH coefficient packed-dot fallback path (if SH coefs are
  quantised to INT4 in some cells)

## Exit criteria

- [ ] 3 backends green on `cargo test` for `dot_i4x32_*`
- [ ] Cross-backend parity (10K random pairs, all equal)
- [ ] B-Pack-Dot bundle composable end-to-end via the consumer trait
- [ ] `cargo clippy -- -D warnings` clean
- [ ] No raw `std::arch::*` intrinsics outside `ndarray::simd::*`
