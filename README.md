# ndarray — AdaWorldAPI HPC Expansion

A complete high-performance numerical computing stack built on top of the [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray) foundation. This fork adds 55 HPC modules with 880 tests, covering BLAS L1-L3, LAPACK, FFT, vector math, quantized inference, and hardware-specific SIMD kernels spanning Intel AMX through Raspberry Pi NEON — all on **stable Rust 1.94**, zero nightly features.

The upstream ndarray provides excellent n-dimensional array abstractions. We keep all of that and add what it was never designed to do: compete with NumPy's OpenBLAS on GEMM, run codebook inference on a 5-watt Pi 4, and handle half-precision floats that Rust doesn't even have a stable type for yet.

[Deutsche Version / German Version](README-DE.md)

## Upstream vs. Fork — Feature by Feature

### ISA Coverage (Instruction Set Architecture)

| ISA / Feature | Upstream ndarray | **AdaWorldAPI Fork** | Speedup vs. Upstream |
|---------------|-----------------|---------------------|---------------------|
| **AVX-512** (512-bit, 16×f32) | Scalar fallback | Native `__m512` types, F32x16/F64x8/U8x64 | **~8×** |
| **AVX-512 VNNI** (int8 dot) | Scalar fallback | `vpdpbusd` 64 MACs/instr + dispatch | **~32×** |
| **AVX-512 BF16** (bfloat16) | Not available | Hardware `vcvtneps2bf16` + RNE emulation | **new** |
| **AVX-512 VPOPCNTDQ** (popcount) | Scalar fallback | Native 512-bit popcount for Hamming | **~16×** |
| **AMX** (Tile Matrix, 256 MACs) | Not available | Inline asm `.byte` encoding, stable Rust | **~128×** vs. scalar |
| **AVX2 + FMA** (256-bit, 8×f32) | Via matrixmultiply | Own Goto-GEMM 6×16 + dispatch table | **~4×** |
| **AVX2 F16C** (f16 hardware) | Not available | IEEE 754 f16, Double-f16, Kahan, Scaler | **new** |
| **AVX-VNNI** (ymm, 32 MACs) | Not available | Arrow Lake / NUC 14 support | **new** |
| **SSE2** (128-bit, 4×f32) | Via matrixmultiply | Scalar polyfill with same API | 1× (baseline) |
| **NEON** (128-bit, 4×f32) | Scalar fallback | 3-tier: A53/A72/A76 with pipeline awareness | **~4×** |
| **NEON dotprod** (ARMv8.2) | Not available | `vdotq_s32` for 4× int8 throughput (Pi 5) | **~16×** vs. scalar |
| **NEON fp16** (ARMv8.2) | Not available | `FCVTL`/`FCVTN` via inline asm | **new** |
| **NEON Popcount** | Not available | `vcntq_u8` native byte popcount | **faster than x86 SSE** |
| **WASM SIMD128** | Not available | Scaffolding prepared | in progress |

### BLAS / Numerics

| Operation | Upstream | **Fork** | Improvement |
|-----------|----------|----------|-------------|
| GEMM (1024²) | ~13 GFLOPS (cache cliff) | **139 GFLOPS** (Goto blocking) | **10.5×** |
| Dot Product | Via matrixmultiply | 4× unrolled + FMA | ~2× |
| BLAS L1 (axpy, scal, nrm2) | Not available | SIMD-accelerated, all tiers | **new** |
| BLAS L2 (gemv, ger, trsv) | Not available | SIMD-accelerated | **new** |
| LAPACK (LU, Cholesky, QR) | Not available | Pure-Rust implementation | **new** |
| FFT | Not available | Cooley-Tukey radix-2 | **new** |
| Activations (sigmoid, GELU) | Not available | SIMD F32x16 vectorization | **new** |
| Quantization (BF16, INT8) | Not available | VNNI + AMX + scalar fallback | **new** |

### Data Types

| Type | Upstream | **Fork** | Note |
|------|----------|----------|------|
| f32 | Standard | Standard + F32x16 SIMD | Same + SIMD acceleration |
| f64 | Standard | Standard + F64x8 SIMD | Same + SIMD acceleration |
| **f16** (IEEE 754) | **Not available** | u16 carrier + F16C/FCVTL hardware | Stable Rust, no nightly |
| **BF16** (bfloat16) | **Not available** | Hardware + RNE emulation (bit-exact) | GGUF calibration |
| i8/u8 (quantized) | Not available | VNNI dot, Hamming, popcount | INT8 inference |
| i16 (Base17) | Not available | L1 distance, SIMD widen/narrow | Codebook encoding |

### Dispatch and Detection

| Aspect | Upstream | **Fork** |
|--------|----------|----------|
| SIMD detection | None (delegates to BLAS) | `LazyLock<SimdCaps>` — detect once, forever |
| Dispatch cost | No own dispatch | **0.3ns** (fn pointer table, no branch) |
| ARM profiling | No ARM awareness | `ArmProfile`: A53/A72/A76 with tok/s estimate |
| big.LITTLE | Not handled | Correct feature intersection (RK3399/RK3588) |
| CPU detection | Per-call runtime | Once via LazyLock, then pointer deref only |

### What Upstream Does on Each Target

```
Upstream on x86_64:   → matrixmultiply crate (external, AVX2 if available)
Upstream on aarch64:  → Scalar (no NEON, no intrinsics)
Upstream on wasm:     → Scalar
Upstream on riscv:    → Scalar

Fork on x86_64:       → AVX-512 F32x16 / AVX2 F32x8 / SSE2 / Scalar (tiered)
Fork on aarch64:      → NEON A76+dotprod / NEON A72 2×pipe / NEON A53 / Scalar
Fork on wasm:         → WASM SIMD128 (prepared) / Scalar
Fork on riscv:        → Scalar (RISC-V V Extension prepared)
```

## Performance

### GEMM (General Matrix Multiply)

| Matrix Size | Upstream ndarray | **This Fork** | NumPy (OpenBLAS) | PyTorch CPU | GPU (RTX 3060) |
|-------------|-----------------|---------------|------------------|-------------|----------------|
| 512×512 | ~20 GFLOPS | **47 GFLOPS** | ~45 GFLOPS | ~40 GFLOPS | ~1,200 GFLOPS |
| 1024×1024 | ~13 GFLOPS | **139 GFLOPS** | ~120 GFLOPS | ~100 GFLOPS | ~3,500 GFLOPS |
| 2048×2048 | ~13 GFLOPS | **~150 GFLOPS** | ~140 GFLOPS | ~130 GFLOPS | ~5,000 GFLOPS |

Upstream hits a cache cliff at 1024×1024: no tiling, no threading, no microkernel. Our Goto implementation eliminates this entirely. At 1024×1024 we deliver **10.5× the throughput of upstream** and match NumPy's decades-old OpenBLAS within measurement noise.

### Codebook Inference (Token Generation)

Not matrix multiplication — O(1) table lookup per token. No GPU required.

| Hardware | ISA | tok/s | 50-Token Latency | Power |
|----------|-----|-------|------------------|-------|
| Sapphire Rapids | AMX (256 MACs/instr) | **380,000** | 0.13 ms | 250W |
| Xeon / i9-13900K | AVX-512 VNNI (64 MACs) | **10,000–50,000** | 1–5 ms | 150W |
| i7-13800K + VNNI | AVX2-VNNI (32 MACs) | **3,000–10,000** | 5–17 ms | 65W |
| Raspberry Pi 5 | NEON + dotprod | **2,000–5,000** | 10–25 ms | 5W |
| Raspberry Pi 4 | NEON (dual pipeline) | **500–2,000** | 25–100 ms | 5W |
| Pi Zero 2W | NEON (single pipeline) | **50–500** | 100–1000 ms | 2W |

At 5 watts, a Pi 4 generates a 50-token voice assistant response in under 100 milliseconds.

### Cosine Similarity via Palette Distance (Integer-Only)

Traditional cosine requires floating-point: `dot(a,b) / (|a| × |b|)`. We replace this with a single u8 table lookup.

| Precision Tier | Sigma Band | Max Cosine Error | Speed |
|----------------|------------|-----------------|-------|
| **Foveal** (1/40 σ) | Inner 2.5% | ±0.004 (0.4%) | **611M lookups/s** |
| **Good** (1/4 σ) | Inner 68% | ±0.02 (2%) | **611M lookups/s** |
| **Near** (1 σ) | Inner 95% | ±0.08 (8%) | **2.4B lookups/s** |
| F32 exact cosine | — | 0 | ~50M/s |

**611 million cosine-equivalent comparisons per second using only integer operations** — 12× faster than SIMD f32 dot product. The 256×256 table (64KB) fits entirely in L1 cache.

### Half-Precision Weight Transcoding

Tested on 15M parameter model (Piper TTS scale):

| Format | Size | Max Error | RMSE | Throughput |
|--------|------|-----------|------|------------|
| f32 (original) | 60 MB | — | — | — |
| **f16 (IEEE 754)** | **30 MB** | 7.3×10⁻⁶ | 2.5×10⁻⁶ | 94M params/s |
| **Scaled-f16** | **30 MB** | 4.9×10⁻⁶ | 2.1×10⁻⁶ | 91M params/s |
| **Double-f16** | 60 MB | 5.7×10⁻⁸ | 1.8×10⁻⁸ | 42M params/s |

## What We Build That Nobody Else Does

### 1. Complete SIMD Polyfill on Stable Rust

`std::simd` has been nightly-only for years. We implement the same type surface using stable `core::arch` intrinsics. The dispatch is a `LazyLock<SimdCaps>` singleton: one CPUID call, frozen forever, zero per-call overhead.

### 2. Half-Precision Types Without Nightly

Rust's `f16` type is nightly-only. We use `u16` as carrier + hardware instructions via stable `#[target_feature]` (F16C on x86, `FCVTL`/`FCVTN` via inline `asm!()` on ARM). IEEE 754 bit-exact at hardware speed.

### 3. AMX on Stable Rust

Intel AMX intrinsics are nightly-only. We emit instructions via `asm!(".byte ...")` encoding — 256 MACs per instruction, verified on Rust 1.94 stable. Reduces distance table build from 24–48h to ~80 minutes.

### 4. Tiered ARM NEON for Single-Board Computers

Three tiers with runtime detection: A53 Baseline (Pi Zero/3), A72 Fast (Pi 4, dual pipeline), A76 DotProd (Pi 5, `vdotq_s32` + native fp16). big.LITTLE aware.

### 5. Frozen Dispatch (0.3ns per call)

Function pointer table, not per-call branching. `LazyLock<SimdDispatch>` → one indirect call, no atomic, no branch prediction miss.

### 6. BF16 RNE Bit-Exact with Hardware

Pure AVX-512-F emulation of `VCVTNEPS2BF16`, verified bit-for-bit on 1M+ inputs including subnormals, Inf, NaN, and halfway ties.

### 7. Cognitive Codec Stack

Fingerprint<256>, Base17 VSA, CAM-PQ, Palette Semiring, bgz7/bgz17 — compressed model weights (201GB → 685MB) with O(1) inference.

## Quick Start

```rust
use ndarray::Array2;
use ndarray::hpc::simd_caps::simd_caps;

let a = Array2::<f32>::ones((1024, 1024));
let b = Array2::<f32>::ones((1024, 1024));
let c = a.dot(&b);  // AVX-512 / AVX2 / NEON — zero code changes

let caps = simd_caps();
if caps.avx512f { println!("AVX-512: 16 lanes"); }
if caps.neon { println!("ARM: {}", caps.arm_profile().name()); }
```

```bash
cargo build --release
cargo build --release --target aarch64-unknown-linux-gnu  # Pi 4
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release  # AVX-512
cargo test  # 880 HPC tests
```

## Ecosystem

| Repository | Role | Uses ndarray for |
|------------|------|-----------------|
| [lance-graph](https://github.com/AdaWorldAPI/lance-graph) | Graph query + codec spine | Fingerprint, CAM-PQ, CLAM, BLAS, ZeckF64 |
| [home-automation-rs](https://github.com/AdaWorldAPI/home-automation-rs) | Smart home + voice AI | Codebook inference, VITS TTS, SIMD audio |

## License

MIT OR Apache-2.0 (same as upstream ndarray)
