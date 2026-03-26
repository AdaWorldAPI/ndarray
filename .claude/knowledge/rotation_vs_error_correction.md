# Rotation vs Error Correction

## Core Finding (2026-03-26)

Google TurboQuant (ICLR 2026) validates bgz17's design choices independently:

- **Rotation before quantization**: TurboQuant uses Hadamard → Polar. bgz17 uses Euler-Gamma bundle rotation (Fujifilm X-Sensor). Same effect (overhead-free distribution equalization), different mechanism (geometry vs number theory).

- **No error correction needed**: TurboQuant requires QJL (1-bit sign correction) because uniform quantization creates boundary bias. bgz17 does NOT need this because Fibonacci codebook entries sit at 1/4σ discrete positions with 3σ inter-qualia separation (99.73% assignment accuracy).

- **No POPCOUNT needed**: Fibonacci encoding makes bits non-uniform (bit 4 = 8× bit 0). Hamming distance (POPCOUNT) would be wrong. Table lookup (`vpshufb`/`vtbl`) is both correct AND faster, and runs on any CPU since 2013.

## Key Implication for ndarray

All bgz17 distance kernels use `vpshufb` (AVX2) or `vtbl` (NEON) table lookups.
No VPOPCNTDQ (AVX-512 Ice Lake+) dependency. No FP arithmetic for distance.
NPU-compatible: INT8 table lookup is the NPU's native operation.

## Full Spec

See `docs/ROTATION_VS_ERROR_CORRECTION.md`
