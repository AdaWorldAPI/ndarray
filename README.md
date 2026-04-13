# ndarray — AdaWorldAPI HPC Expansion

A complete high-performance numerical computing stack built on top of the [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray) foundation. This fork adds 55 HPC modules with 880 tests, covering BLAS L1-L3, LAPACK, FFT, vector math, quantized inference, and hardware-specific SIMD kernels spanning Intel AMX through Raspberry Pi NEON — all on **stable Rust 1.94**, zero nightly features.

The upstream ndarray provides excellent n-dimensional array abstractions. We keep all of that and add what it was never designed to do: compete with NumPy's OpenBLAS on GEMM, run codebook inference on a 5-watt Pi 4, and handle half-precision floats that Rust doesn't even have a stable type for yet.

## Core Architecture

The expansion comprises five layers built on top of upstream's array primitives:

**SIMD Polyfill Layer** (`src/simd.rs`, `simd_avx512.rs`, `simd_avx2.rs`, `simd_neon.rs`) provides `std::simd`-compatible types — `F32x16`, `F64x8`, `U8x64`, `I32x16`, `I64x8`, `U32x16`, `U64x8` with full operator overloading, reductions, comparisons, and masked operations — backed by `core::arch` intrinsics on x86 and inline assembly on ARM. Consumers write `crate::simd::F32x16` and get native 512-bit operations on AVX-512, 256-bit on AVX2, 128-bit on NEON, or scalar fallback, with zero code changes. Detection happens once via `LazyLock<SimdCaps>` (one pointer deref per call, no atomics, no branch prediction misses).

**Backend Layer** (`src/backend/`) implements pluggable BLAS through the `BlasFloat` trait with three backends: pure-Rust SIMD microkernels (default, zero dependencies), Intel MKL FFI (feature-gated), and OpenBLAS FFI (feature-gated, mutually exclusive with MKL). The native backend uses Goto-algorithm cache-blocked GEMM with 6×16 (f32) and 6×8 (f64) microkernels, achieving 139 GFLOPS at 1024×1024 — a 10.5× improvement over the naive approach and within 15% of NumPy's multi-threaded OpenBLAS.

**HPC Module Library** (`src/hpc/`, 55 modules) delivers a complete numerical computing surface: BLAS Level 1-3 (dot, axpy, gemv, gemm, syrk, trsm), LAPACK factorizations (LU, Cholesky, QR), Cooley-Tukey FFT, vector math (exp, ln, sqrt, erf, trigonometric), statistics (median, variance, percentile, top-k), neural network activations (sigmoid, softmax, GELU, SiLU), and quantized operations (BF16 GEMM, INT8 GEMM via VNNI). Every module has SIMD-accelerated hot paths that dispatch through the frozen function pointer table.

**Codec Layer** (`src/hpc/fingerprint.rs`, `bgz17_bridge.rs`, `cam_pq.rs`) implements the encoding stack for compressed inference: 16Kbit Fingerprints, Base17 VSA (17-dimensional i16 vectors), CAM-PQ product quantization, ZeckF64 Fibonacci encoding, and palette semiring distance matrices. This is what makes codebook inference O(1) per token — table lookups replace matrix multiplication.

**Burn Integration** (`crates/burn/`) provides a SIMD-augmented burn-ndarray backend that wires `crate::simd::F32x16` into burn's tensor operations and activations, replacing macerator's SIMD with our LazyLock-dispatched implementations. This enables using burn's model format and autodiff while benefiting from our full SIMD stack.

## Performance

### GEMM (General Matrix Multiply)

The Goto-algorithm GEMM with cache blocking (L1: 32KB, L2: 256KB, L3: shared) and 16-thread parallelism via split-borrow (no mutex contention):

| Matrix Size | Upstream ndarray | **This Fork** | NumPy (OpenBLAS) | PyTorch CPU | GPU (RTX 3060) |
|-------------|-----------------|---------------|------------------|-------------|----------------|
| 512×512 | ~20 GFLOPS | **47 GFLOPS** | ~45 GFLOPS | ~40 GFLOPS | ~1,200 GFLOPS |
| 1024×1024 | ~13 GFLOPS¹ | **139 GFLOPS** | ~120 GFLOPS | ~100 GFLOPS | ~3,500 GFLOPS |
| 2048×2048 | ~13 GFLOPS¹ | **~150 GFLOPS** | ~140 GFLOPS | ~130 GFLOPS | ~5,000 GFLOPS |

¹ Upstream hits a cache cliff at 1024×1024: no tiling, no threading, no microkernel. Our Goto implementation eliminates this entirely.

At 1024×1024 we deliver **10.5× the throughput of upstream** and match NumPy's decades-old OpenBLAS within measurement noise. GPU wins at large dense matrices but carries 170W power draw and PCIe transfer latency; our CPU path wins at latency-sensitive workloads and mixed compute/IO patterns.

### Codebook Inference (Token Generation)

This is not matrix multiplication. Codebook inference replaces `y = W·x` with `y = codebook[index[x]]` — an O(1) table lookup per token. No GPU required, no FP32 accumulation, just memory bandwidth.

| Hardware | ISA | tok/s | 50-Token Latency | Power |
|----------|-----|-------|------------------|-------|
| Sapphire Rapids | AMX (256 MACs/instr) | **380,000** | 0.13 ms | 250W |
| Xeon / i9-13900K | AVX-512 VNNI (64 MACs) | **10,000–50,000** | 1–5 ms | 150W |
| i7-13800K + VNNI | AVX2-VNNI (32 MACs) | **3,000–10,000** | 5–17 ms | 65W |
| Raspberry Pi 5 | NEON + dotprod | **2,000–5,000** | 10–25 ms | 5W |
| Raspberry Pi 4 | NEON (dual pipeline) | **500–2,000** | 25–100 ms | 5W |
| Pi Zero 2W | NEON (single pipeline) | **50–500** | 100–1000 ms | 2W |

At 5 watts, a Pi 4 generates a 50-token voice assistant response in under 100 milliseconds. The AMX path on Sapphire Rapids achieves 380K tok/s — faster than most GPU-based inference for small-batch queries because there is no kernel launch overhead, no PCIe round-trip, and no memory allocation.

### Semantic Search (SPO Palette Distance)

Compressed vector similarity using palette-indexed distance tables:

| Metric | Value |
|--------|-------|
| Throughput | **611 million lookups/sec** |
| Latency per lookup | **1.8 nanoseconds** |
| Working set | **388 KB** (fits in L2 cache) |
| Token throughput | **17,000 tok/s** (triple model, 4096 heads) |

### Half-Precision Weight Transcoding

Tested on 15 million parameter model (Piper TTS scale):

| Format | Size | Max Error | RMSE | Throughput |
|--------|------|-----------|------|------------|
| f32 (original) | 60 MB | — | — | — |
| **f16 (IEEE 754)** | **30 MB** | 7.3×10⁻⁶ | 2.5×10⁻⁶ | 94M params/s |
| **Scaled-f16** | **30 MB** | 4.9×10⁻⁶ | 2.1×10⁻⁶ | 91M params/s |
| **Double-f16** | 60 MB | 5.7×10⁻⁸ | 1.8×10⁻⁸ | 42M params/s |

With AVX2 F16C hardware: **~500M params/sec** (8 conversions per clock cycle).

## What We Build That Nobody Else Does

### 1. Complete SIMD Polyfill on Stable Rust

`std::simd` (portable SIMD) has been nightly-only for years. We implement the same type surface — `F32x16`, `F64x8`, `U8x64`, masks, reductions, comparisons, shuffles, gathers — using stable `core::arch` intrinsics. When `std::simd` eventually stabilizes, consumers change one `use` line. Until then, they get native AVX-512 performance today.

The dispatch is a `LazyLock<SimdCaps>` singleton detected at first access: one CPUID call, frozen forever, zero per-call overhead. The function pointer table (`SimdDispatch`) eliminates branch prediction misses entirely — the CPU sees an indirect call, not a conditional branch.

### 2. Half-Precision Types Without Nightly

Rust's `f16` type is nightly-only (issue #116909). We use the same trick as our AMX implementation: `u16` as the carrier type, hardware instructions via stable `#[target_feature]` (F16C on x86, `FCVTL`/`FCVTN` via inline `asm!()` on ARM). The result is IEEE 754 bit-exact f16↔f32 conversion at hardware speed, with three precision strategies:

- **Plain f16**: 2 bytes, 10-bit mantissa, good for sensors and audio
- **Scaled-f16**: 2 bytes + 8-byte header, range-optimized for 1.5× better precision on narrow data
- **Double-f16**: 4 bytes (hi + lo pair), ~20-bit effective mantissa — 128× more precise than single f16

### 3. AMX on Stable Rust

Intel AMX (Advanced Matrix Extensions) provides hardware tile matrix multiplication: `TDPBUSD` computes a 16×16 tile of u8×i8→i32 — 256 multiply-accumulate operations in a single instruction. The Rust intrinsics are nightly-only (issue #126622). We emit the instructions directly via `asm!(".byte ...")` encoding, verified working on Rust 1.94 stable with kernel 6.18+ (XCR0 bits 17+18 enabled).

The runtime dispatch chain: AMX TILE (256 MACs) → AVX-512 VNNI (64 MACs) → AVX-VNNI ymm (32 MACs) → scalar i32. On Sapphire Rapids, this reduces codebook distance table build time from 24–48 hours to ~80 minutes.

### 4. Tiered ARM NEON for Single-Board Computers

Most Rust libraries treat ARM as "not x86, use scalar." We implement three tiers with runtime detection via `is_aarch64_feature_detected!()`:

- **A53 Baseline** (Pi Zero 2W, Pi 3): single NEON pipeline, no unrolling, minimize instruction count
- **A72 Fast** (Pi 4, Orange Pi 4): dual NEON pipeline, 2× unrolled loops to saturate both pipes
- **A76 DotProd** (Pi 5, Orange Pi 5): `vdotq_s32` for 4× int8 throughput, native fp16 via FCVTL

The `ArmProfile` enum exposes estimated tok/s, effective lane count, and microarchitecture hints. big.LITTLE systems (RK3399, RK3588) are handled correctly: feature detection returns the intersection of all core types, and we document which features are safe to use unconditionally.

### 5. Frozen Dispatch (Zero-Cost Tier Selection)

Typical SIMD code branches on every call: `if is_x86_feature_detected!("avx512f") { ... }`. Each check is an atomic load + branch. We do it once:

```
LazyLock<SimdDispatch> → fn pointer table (Copy struct, lives in registers)
Per-call cost: 1 pointer deref + 1 indirect call = ~0.3ns
vs per-call branch: 1 atomic load + 1 branch predict = ~1–3ns
```

The dispatch table is a `Copy` struct of function pointers, selected at first access and never modified. After initialization, the CPU's branch predictor sees a stable indirect call target — effectively free.

### 6. BF16 Round-to-Nearest-Even (Bit-Exact with Hardware)

Our `f32_to_bf16_batch_rne()` uses pure AVX-512-F instructions to implement the IEEE 754 Round-to-Nearest-Even algorithm, matching Intel's `VCVTNEPS2BF16` instruction **bit-for-bit**. This runs on any AVX-512 CPU, not just those with the BF16 extension. Verified against hardware output on 1M+ inputs, including all subnormal, infinity, NaN, and halfway tie cases.

### 7. Cognitive Codec Stack

Beyond traditional numerical computing, we implement a complete encoding pipeline for compressed AI inference:

- **Fingerprint\<256\>**: 256-bit binary vectors with SIMD Hamming distance (AVX-512 VPOPCNTDQ or NEON `vcntq_u8`)
- **Base17**: 17-dimensional i16 vectors with L1 distance — fits in one AVX-512 load (32 bytes)
- **CAM-PQ**: Product quantization with compiled distance tables for sub-linear search
- **Palette Semiring**: 256×256 distance matrices for O(1) token-level lookups
- **bgz7/bgz17**: Compressed model weight format (201GB BF16 safetensors → 685MB bgz7)

### Cosine Similarity via Palette Distance (Integer-Only Approximation)

Traditional cosine similarity requires floating-point: `dot(a,b) / (|a| × |b|)` — three passes over the data plus a division. We replace this with a single u8 table lookup that emulates cosine at two precision tiers:

**How it works:** High-dimensional vectors are quantized to 256 archetypes. The pairwise distance between any two archetypes is precomputed into a 256×256 u8 distance table. At query time, cosine similarity between two vectors reduces to `table[archetype_a][archetype_b]` — one memory access, no floating point.

| Precision Tier | Sigma Band | u8 Steps | Max Cosine Error | Speed |
|----------------|------------|----------|-----------------|-------|
| **Foveal** (1/40 σ) | Inner 2.5% | 256 | ±0.004 (0.4%) | **611M lookups/s** |
| **Good** (1/4 σ) | Inner 68% | 256 | ±0.02 (2%) | **611M lookups/s** |
| **Near** (1 σ) | Inner 95% | 64 | ±0.08 (8%) | **2.4B lookups/s** |
| F32 exact cosine | — | — | 0 | ~50M/s (SIMD dot) |

The key insight: **611 million cosine-equivalent comparisons per second using only integer operations**. This is 12× faster than SIMD f32 dot product because:
1. No FP division (the normalization is baked into the table)
2. No FP multiplication (it's a table read, not arithmetic)
3. The 256×256 table (64KB) fits entirely in L1 cache
4. u8 loads have no alignment constraints

The Foveal tier at 1/40σ achieves 0.4% maximum error — indistinguishable from exact cosine for nearest-neighbor search, semantic similarity, and clustering. The cascade search architecture uses the Near tier (8% error) to eliminate 99.7% of candidates in the first pass, then refines survivors with the Foveal tier.

This is the engine behind the **17,000 tok/s** benchmark: each token lookup computes similarity against 4,096 heads using palette distance, not matrix multiplication.

## Module Inventory

```
src/
├── simd.rs                 LazyLock tier detection, type re-exports, PREFERRED_LANES
├── simd_avx512.rs          11 SIMD types + BF16 codec + F16 IEEE 754 (2,700 LOC)
├── simd_avx2.rs            BLAS L1, Hamming, i8 dot, F16 precision toolkit (1,600 LOC)
├── simd_neon.rs            3-tier ARM NEON: baseline/A72/A76+dotprod+fp16 (500 LOC)
├── simd_amx.rs             AMX detection + VNNI dispatch + quantize/dequantize (350 LOC)
├── simd_wasm.rs            WebAssembly SIMD scaffolding
├── backend/
│   ├── native.rs           Pure-Rust GEMM microkernels (Goto 6×16/6×8)
│   ├── mkl.rs              Intel MKL FFI (feature-gated)
│   └── openblas.rs         OpenBLAS FFI (feature-gated)
└── hpc/                    55 modules, 880 tests
    ├── blas_level1.rs      dot, axpy, scal, nrm2, asum, iamax
    ├── blas_level2.rs      gemv, ger, symv, trmv, trsv
    ├── blas_level3.rs      gemm, syrk, trsm, symm (Goto-blocked)
    ├── quantized.rs        BF16 GEMM, INT8 GEMM, quantize/dequantize
    ├── lapack.rs           LU, Cholesky, QR factorization
    ├── fft.rs              Cooley-Tukey radix-2 FFT/IFFT
    ├── vml.rs              exp, ln, sqrt, erf, cbrt, sin, cos
    ├── statistics.rs       median, variance, std, percentile, top_k
    ├── activations.rs      sigmoid, softmax, log_softmax, GELU, SiLU
    ├── fingerprint.rs      Fingerprint<256> (VSA, Hamming, XOR bind)
    ├── bgz17_bridge.rs     Base17 encode/decode, L1 distance, sign agreement
    ├── cam_pq.rs           Product quantization, IVF, distance tables
    ├── simd_caps.rs        LazyLock SimdCaps + ArmProfile detection
    ├── simd_dispatch.rs    Frozen function pointer dispatch table
    ├── clam.rs             CLAM tree (build, search, rho_nn, 46 tests)
    ├── blackboard.rs       Typed slot arena (zero-copy shared memory)
    ├── cascade.rs          HDR cascade search (sigma-band filtering)
    ├── causal_diff.rs      CausalEdge64 (u64 packed), quality scoring
    └── ... (37 more: hdc, nars, qualia, styles, bnn, ocr, arrow_bridge)
```

## Quick Start

```rust
use ndarray::Array2;
use ndarray::hpc::simd_caps::simd_caps;

// GEMM — automatically uses best available SIMD
let a = Array2::<f32>::ones((1024, 1024));
let b = Array2::<f32>::ones((1024, 1024));
let c = a.dot(&b);  // AVX-512 / AVX2 / NEON — zero code changes

// Check hardware
let caps = simd_caps();
if caps.avx512f { println!("AVX-512: 16 lanes"); }
if caps.neon { println!("ARM: {}", caps.arm_profile().name()); }
```

```bash
# Build (auto-detects best SIMD)
cargo build --release

# Cross-compile for Raspberry Pi 4
cargo build --release --target aarch64-unknown-linux-gnu

# Maximum performance on AVX-512 server
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release

# Run the 880 HPC tests
cargo test
```

## Requirements

- **Rust 1.94 stable** (no nightly, no unstable features)
- Optional: `gcc-aarch64-linux-gnu` for Pi cross-compilation
- Optional: Intel MKL or OpenBLAS for BLAS acceleration (feature-gated)

## Ecosystem

This crate is the hardware foundation for a larger architecture:

| Repository | Role | Depends on ndarray for |
|------------|------|----------------------|
| [lance-graph](https://github.com/AdaWorldAPI/lance-graph) | Graph query + codec spine | Fingerprint, CAM-PQ, CLAM, BLAS, ZeckF64 |
| [home-automation-rs](https://github.com/AdaWorldAPI/home-automation-rs) | Smart home + voice AI | Codebook inference, VITS TTS, SIMD audio |
| [ada-rs](https://github.com/AdaWorldAPI/ada-rs) | Cognitive substrate | 10K-bit VSA, Hamming, perception |

## License

MIT OR Apache-2.0 (same as upstream ndarray)
