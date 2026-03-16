# Hardware Pipeline Map

## SIMD Dispatch Tiers (ndarray)

All cognitive types delegate bulk operations to `src/hpc/bitwise.rs` which
dispatches at runtime:

```
Tier 1: AVX-512 VPOPCNTDQ  — 64 bytes/iter, native popcount
Tier 2: AVX-512 BW vpshufb  — 64 bytes/iter, LUT popcount
Tier 3: AVX2 vpshufb         — 32 bytes/iter, LUT popcount
Tier 4: Scalar count_ones()  — 1 byte/iter
```

Detection: `is_x86_feature_detected!()` at first call, cached via LazyLock.

## Key Operations → Instructions

| Operation | Instruction | Tier | Throughput |
|-----------|------------|------|------------|
| XOR | VPXORD | AVX-512 | 64B/cycle |
| Popcount | VPOPCNTDQ | AVX-512 | 64B/cycle |
| Popcount (LUT) | VPSHUFB | AVX-512BW | 64B/cycle |
| Popcount (LUT) | VPSHUFB | AVX2 | 32B/cycle |
| SAD (reduce) | VPSADBW | AVX-512 | 64B/cycle |
| FMA | VFMADD231PS | AVX-512 | 16 f32/cycle |
| Dot i8 | VPDPBUSD | AVX-512 VNNI | 64 u8/cycle |
| BF16 dot | VDPBF16PS | AVX-512 BF16 | 32 bf16/cycle |

## Cache Boundaries

```
L1: 32-48 KB → Plane acc (16KB) fits
L2: 256-512 KB → 4 Planes fit
L3: shared → database scan
```

## Throughput Estimates

- Hamming distance (2KB fingerprint): ~100ns @ AVX-512
- Plane distance (alpha-masked): ~200ns (2 popcounts + 1 AND)
- Cascade Stroke 1 (128B): ~50ns
- BF16 weighted Hamming (1024D): ~500ns
- Full RL step per pair: ~2.8μs

## Alignment Requirements

- Acc16K: `#[repr(C, align(64))]` for AVX-512
- Blackboard buffers: 64-byte aligned via `alloc::Layout`
- Fingerprint words: naturally 8-byte aligned (u64)
