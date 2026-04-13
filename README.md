# ndarray — AdaWorldAPI HPC Fork

> High-performance n-dimensional arrays with **pluggable SIMD backends**, f16/BF16 codecs, and ARM NEON support for Raspberry Pi.

Fork of [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray) v0.17.2 — all upstream functionality preserved.

## What's Different

| Feature | Upstream | **AdaWorldAPI Fork** |
|---------|----------|---------------------|
| SIMD dispatch | None (delegates to BLAS) | **LazyLock frozen dispatch**: AVX-512 → AVX2 → NEON → Scalar |
| ARM support | Scalar only | **Tiered NEON**: A53 (Pi Zero 2W) / A72 (Pi 4) / A76+dotprod (Pi 5) |
| f16 (half-precision) | None | **IEEE 754 f16** via F16C hardware + inline asm on ARM |
| BF16 (bfloat16) | None | **AVX-512 BF16** + RNE scalar + batch conversion |
| HPC modules | None | **55 modules**, 880 tests: BLAS L1-L3, FFT, LAPACK, activations, quantized GEMM |
| AMX (matrix tiles) | None | **Intel AMX** via inline asm (256 MACs/instruction, stable Rust) |
| VNNI (int8 dot) | None | **AVX-512 VNNI** + AVX-VNNI (ymm) + scalar fallback |
| Precision toolkit | None | **Double-f16** (~20-bit), Kahan summation, F16Scaler |
| Codebook inference | None | **Fingerprint<256>**, Base17, CAM-PQ, palette semiring |
| Burn backend | None | **burn-ndarray** with SIMD-augmented tensor ops |
| Binary size | ~500KB lib | Same + opt-level z → **~3MB total binary** on Pi |

## Benchmarks

### GEMM (General Matrix Multiply)

*Source: [AdaWorldAPI/rustynum](https://github.com/AdaWorldAPI/rustynum) — ported into this fork*

| Matrix Size | ndarray upstream | **This fork (Goto+MT)** | NumPy (OpenBLAS) | PyTorch CPU |
|-------------|-----------------|------------------------|------------------|-------------|
| 512×512 | ~20 GFLOPS | **47 GFLOPS** | ~45 GFLOPS | ~40 GFLOPS |
| 1024×1024 | ~13 GFLOPS¹ | **139 GFLOPS** | ~120 GFLOPS | ~100 GFLOPS |
| 2048×2048 | ~13 GFLOPS¹ | **~150 GFLOPS** | ~140 GFLOPS | ~130 GFLOPS |
| 4096×4096 (est.) | ~13 GFLOPS¹ | **~160 GFLOPS** | ~150 GFLOPS | ~140 GFLOPS |
| *GPU (est.)* | — | — | — | *~5,000 GFLOPS²* |

¹ Upstream hits a cache cliff at 1024×1024 (no Goto blocking, no threading).
² RTX 3060 FP32 tensor cores. GPU wins at large matrices but loses at small/latency-sensitive workloads.

**10.5× improvement** over upstream at 1024×1024 — closes the gap with NumPy's multi-threaded OpenBLAS.

### Codebook Inference (Token Generation)

*Not matrix multiplication — O(1) table lookup per token. No GPU needed.*

| Hardware | ISA | tok/s | Latency (50 tok) | Power |
|----------|-----|-------|-------------------|-------|
| Sapphire Rapids | AMX (256 MACs/instr) | **380,000** | 0.13 ms | 250W |
| Xeon / i9 | AVX-512 + VNNI | **10,000–50,000** | 1–5 ms | 150W |
| Consumer i5/i7 | AVX2 | **3,000–10,000** | 5–17 ms | 65W |
| **Pi 5** (A76) | **NEON + dotprod** | **2,000–5,000** | 10–25 ms | **5W** |
| **Pi 4** (A72) | **NEON (dual pipe)** | **500–2,000** | 25–100 ms | **5W** |
| Pi Zero 2W (A53) | NEON (single) | 50–500 | 100–1000 ms | **2W** |
| *RTX 3060* | *CUDA* | *~100,000* | *0.5 ms* | *170W* |

**Key insight**: At 5W, Pi 4 generates 50-token Alisa responses in <100ms — fast enough for real-time voice.

### SPO Palette Distance (Semantic Search)

| Metric | Value |
|--------|-------|
| Throughput | **611M lookups/sec** |
| Latency | **1.8 ns/lookup** |
| Memory | **388 KB** |
| Tokens/sec (triple model, 4096 heads) | **17,000** |

### f16 Weight Transcoding

*Tested with 15M parameter model (Piper TTS size):*

| Format | Size | Compression | Max Error | RMSE | Speed |
|--------|------|-------------|-----------|------|-------|
| f32 (original) | 60.0 MB | 1.0× | 0 | 0 | — |
| **f16 (IEEE 754)** | **30.0 MB** | **2.0×** | 7.3×10⁻⁶ | 2.5×10⁻⁶ | 94 M/s |
| **Scaled-f16** | **30.0 MB** | **2.0×** | 4.9×10⁻⁶ | 2.1×10⁻⁶ | 91 M/s |
| **Double-f16** | 60.0 MB | 1.0× | 5.7×10⁻⁸ | 1.8×10⁻⁸ | 42 M/s |

With AVX2 F16C hardware: **~500 M params/sec** (8-wide conversion per instruction).

## SIMD Architecture

### Detection (LazyLock — detect once, dispatch forever)

```rust
use ndarray::hpc::simd_caps::simd_caps;

let caps = simd_caps();
println!("AVX-512: {}", caps.avx512f);
println!("AVX2: {}", caps.avx2);
println!("NEON: {}", caps.neon);
println!("ARM profile: {}", caps.arm_profile().name());
// → "A72-fast (Pi 4 / Orange Pi 4)"
```

### Tier Table

```
┌──────────┬───────────────────┬────────┬──────────────────────────────┐
│ Priority │ Tier              │ Width  │ Guard                        │
├──────────┼───────────────────┼────────┼──────────────────────────────┤
│ 1        │ AVX-512 + AMX     │ 512-bit│ caps.avx512f                 │
│ 2        │ AVX-512 + VNNI    │ 512-bit│ caps.avx512f + avx512vnni    │
│ 3        │ AVX2 + FMA        │ 256-bit│ caps.avx2                    │
│ 4        │ NEON + dotprod    │ 128-bit│ caps.neon + caps.dotprod (Pi 5)│
│ 5        │ NEON baseline     │ 128-bit│ caps.neon (Pi 3/4)           │
│ 6        │ SSE2              │ 128-bit│ always on x86_64             │
│ 7        │ Scalar            │ 1 lane │ fallback                     │
└──────────┴───────────────────┴────────┴──────────────────────────────┘
```

### f16 Trick (stable Rust, no nightly)

`f16` type is nightly-only. We use `u16` as carrier + hardware instructions:
- **x86**: F16C `VCVTPH2PS`/`VCVTPS2PH` (stable `target_feature` since Rust 1.68)
- **ARM**: `FCVTL`/`FCVTN` via inline `asm!()` (same trick as AMX `.byte` encoding)
- **Scalar**: IEEE 754 bit manipulation fallback (all platforms)

### Precision Toolkit (simd_avx2.rs)

| Trick | Storage | Effective Bits | Use Case |
|-------|---------|---------------|----------|
| **f16** (plain) | 2 bytes | 10 mantissa | Sensors, audio, compact storage |
| **Scaled-f16** | 2 bytes + 8B header | 10 (optimized) | Narrow-range weights (1.5× better) |
| **Double-f16** | 4 bytes (hi+lo) | ~20 mantissa | When f16 too imprecise, f32 too big |
| **Kahan sum** | f32 accumulator | O(ε) error | Sum/dot of many f16 values |

## ARM Single-Board Computer Support

| Board | CPU | SIMD | Codebook tok/s | Status |
|-------|-----|------|---------------|--------|
| Pi Zero 2W | Cortex-A53 | NEON baseline | 50–500 | ✅ Tested |
| Pi 3B+ | Cortex-A53 | NEON baseline | 50–500 | ✅ Tested |
| **Pi 4** | **Cortex-A72** | **NEON 2× pipe** | **500–2,000** | ✅ Primary target |
| **Pi 5** | **Cortex-A76** | **NEON + dotprod + fp16** | **2,000–5,000** | ✅ Tested |
| Orange Pi 4 LTS | RK3399 (A72+A53) | NEON + crypto | 500–2,000 | ✅ big.LITTLE aware |
| Orange Pi 5 | RK3588 (A76+A55) | NEON + dotprod | 2,000–5,000 | ✅ big.LITTLE aware |

## HPC Module Inventory (55 modules, 880 tests)

```
src/hpc/
├── blas_level1.rs    BLAS L1: dot, axpy, scal, nrm2, asum
├── blas_level2.rs    BLAS L2: gemv, ger, symv, trmv, trsv
├── blas_level3.rs    BLAS L3: gemm, syrk, trsm, symm (Goto algorithm)
├── quantized.rs      BF16 GEMM, Int8 GEMM
├── lapack.rs         LU, Cholesky, QR decomposition
├── fft.rs            FFT/IFFT (Cooley-Tukey radix-2)
├── vml.rs            Vector math: exp, ln, sqrt, erf, cbrt
├── statistics.rs     Median, variance, percentile, top_k
├── activations.rs    Sigmoid, softmax, log_softmax, GELU, SiLU
├── fingerprint.rs    Fingerprint<256> (VSA, Hamming distance)
├── simd_caps.rs      LazyLock CPU detection + ArmProfile
├── simd_dispatch.rs  Frozen function pointer table (0.3ns dispatch)
└── ... (43 more modules: cognitive, codec, cascade, bridge)
```

## Quick Start

```rust
use ndarray::Array2;
use ndarray::hpc::simd_caps::simd_caps;

// Automatic SIMD — just use ndarray normally
let a = Array2::<f32>::zeros((1024, 1024));
let b = Array2::<f32>::zeros((1024, 1024));
let c = a.dot(&b); // Uses AVX-512/AVX2/NEON automatically

// Check what SIMD tier is active
let caps = simd_caps();
println!("Running on: {:?}", caps.arm_profile().name());
```

```bash
# Build (auto-detects SIMD)
cargo build --release

# Cross-compile for Pi 4
cargo build --release --target aarch64-unknown-linux-gnu

# With AVX-512 (x86 server)
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release
```

## Ecosystem

| Repo | Role |
|------|------|
| **AdaWorldAPI/ndarray** (this) | Foundation: SIMD, GEMM, HPC, Fingerprint, codecs |
| [AdaWorldAPI/lance-graph](https://github.com/AdaWorldAPI/lance-graph) | Spine: query engine, Cypher parser, codec stack |
| [AdaWorldAPI/home-automation-rs](https://github.com/AdaWorldAPI/home-automation-rs) | Smart home: MQTT, voice AI, MCP server |
| [AdaWorldAPI/ada-rs](https://github.com/AdaWorldAPI/ada-rs) | Cognitive substrate: persona, presence, feel |

## License

MIT OR Apache-2.0 (same as upstream ndarray)
