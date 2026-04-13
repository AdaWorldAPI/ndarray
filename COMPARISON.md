# Complete Feature Comparison: rust-ndarray vs. AdaWorldAPI Fork

> 80,131 lines of new code across 146 HPC modules, 6 SIMD files, 5 backend files, 20 burn ops, and 2 subcrates.

## At a Glance

| Metric | Upstream [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray) | **[AdaWorldAPI/ndarray](https://github.com/AdaWorldAPI/ndarray)** |
|--------|-----------|------------|
| Base functionality | n-dimensional arrays, slicing, views | **Same** (full upstream preserved) |
| New LOC added | — | **80,131** |
| New files | — | **179** (146 HPC + 6 SIMD + 5 backend + 20 burn + 2 subcrates) |
| Test count | ~300 | **~1,180** (300 upstream + 880 new) |
| SIMD ISAs | SSE2 via matrixmultiply (external) | **7 ISAs**: AVX-512, AVX2, SSE2, AMX, VNNI, NEON (3 tiers), WASM |
| Numeric types | f32, f64 | **+f16, BF16, i8, u8, i16** (all with SIMD paths) |
| BLAS coverage | dot (via matrixmultiply) | **Full L1 + L2 + L3** (pure-Rust + MKL + OpenBLAS) |
| Target platforms | x86_64 (via external BLAS), scalar everywhere else | **x86_64 (tiered), aarch64 (3-tier NEON), wasm (prepared)** |
| Minimum Rust | 1.64 | **1.94 stable** (no nightly) |

---

## SIMD Layer (6 files, ~5,700 LOC)

| Component | Upstream | **Fork** |
|-----------|----------|----------|
| `simd.rs` — dispatch + re-exports | Not present | **LazyLock tier detection, PREFERRED_LANES, type re-exports** |
| `simd_avx512.rs` — 512-bit types | Not present | **11 types: F32x16, F64x8, U8x64, I32x16, I64x8, U32x16, U64x8, F32x8, F64x4, BF16x16, BF16x8 + F16 IEEE 754** (2,700 LOC) |
| `simd_avx2.rs` — 256-bit ops | Not present | **BLAS L1, Hamming, i8 dot, popcount, F16 precision toolkit** (1,600 LOC) |
| `simd_neon.rs` — ARM 128-bit | Not present | **3-tier NEON: A53 baseline, A72 dual-pipe, A76 dotprod+fp16; codebook gather, Hamming, Base17 L1** (500 LOC) |
| `simd_amx.rs` — Intel tile matrix | Not present | **AMX detection (CPUID+XCR0), VNNI 512/256, MatVec dispatch, quantize/dequantize** (350 LOC) |
| `simd_wasm.rs` — WebAssembly | Not present | **Scaffolding for WASM SIMD128** |

## Backend Layer (5 files, ~2,000 LOC)

| Component | Upstream | **Fork** |
|-----------|----------|----------|
| `backend/mod.rs` — BlasFloat trait | Not present | **Trait-based dispatch: Native / MKL / OpenBLAS** |
| `backend/native.rs` — pure-Rust GEMM | Not present | **Goto-algorithm 6x16/6x8 microkernels, cache-blocked (L1/L2/L3), AVX-512+AVX2 dispatch** |
| `backend/kernels_avx512.rs` | Not present | **AVX-512 SIMD GEMM kernels** |
| `backend/mkl.rs` | Not present | **Intel MKL FFI (feature = "intel-mkl")** |
| `backend/openblas.rs` | Not present | **OpenBLAS FFI (feature = "openblas")** |
| GEMM throughput (1024x1024) | ~13 GFLOPS (via matrixmultiply) | **139 GFLOPS** (10.5x improvement) |

## HPC Module Library (146 files, ~70,000 LOC, 880 tests)

### Linear Algebra (BLAS + LAPACK)

| Module | Upstream | **Fork** | Operations |
|--------|----------|----------|------------|
| `blas_level1.rs` | dot only (external) | **Full** | dot, axpy, scal, nrm2, asum, iamax, Givens rotation |
| `blas_level2.rs` | Not present | **Full** | gemv, ger, symv, trmv, trsv |
| `blas_level3.rs` | dot→gemm (external) | **Goto GEMM** | gemm, syrk, trsm, symm (cache-blocked, multithreaded) |
| `quantized.rs` | Not present | **New** | BF16 GEMM, INT8 GEMM, quantize/dequantize |
| `lapack.rs` | Not present | **New** | LU, Cholesky, QR factorization |

### Signal Processing

| Module | Upstream | **Fork** | Detail |
|--------|----------|----------|--------|
| `fft.rs` | Not present | **Cooley-Tukey** | Radix-2 FFT/IFFT, in-place |
| `vml.rs` | Not present | **Vector Math** | exp, ln, sqrt, erf, cbrt, sin, cos (SIMD F32x16 paths) |
| `statistics.rs` | Not present | **Statistics** | median, variance, std, percentile, top_k |
| `activations.rs` | Not present | **Neural Net** | sigmoid, softmax, log_softmax, GELU, SiLU (fused SIMD) |

### Hardware Detection + Dispatch

| Module | Upstream | **Fork** | Detail |
|--------|----------|----------|--------|
| `simd_caps.rs` | Not present | **SimdCaps** | LazyLock detection: AVX-512/AVX2/SSE2/FMA/NEON/dotprod/fp16/aes/sha2/crc32 + **ArmProfile** (A53/A72/A76) |
| `simd_dispatch.rs` | Not present | **SimdDispatch** | Frozen fn-pointer table: 0.3ns per call, no branch, no atomic |
| `amx_matmul.rs` | Not present | **AMX MatMul** | Tile configuration, TDPBUSD via inline asm |

### Encoding + Codec (Cognitive Computing)

| Module | Upstream | **Fork** | Detail |
|--------|----------|----------|--------|
| `fingerprint.rs` | Not present | **Fingerprint\<256\>** | 256-bit VSA, XOR bind, Hamming distance (VPOPCNTDQ / vcntq_u8) |
| `bgz17_bridge.rs` | Not present | **Base17** | 17-dim i16 vectors, L1 distance, sign agreement, xor_bind |
| `cam_pq.rs` | Not present | **CAM-PQ** | Product quantization, compiled distance tables, IVF index |
| `cam_index.rs` | Not present | **CAM Index** | Inverted file index for PQ search |
| `palette_codec.rs` | Not present | **Palette Codec** | 4-bit palette encoding, Minecraft-style chunk compression |
| `palette_distance.rs` | Not present | **Palette Distance** | 256x256 u8 distance tables, cosine emulation (611M/s) |
| `zeck.rs` | Not present | **ZeckF64** | Fibonacci/Zeckendorf encoding for sparse representations |
| `packed.rs` | Not present | **Packed DB** | 64-byte aligned packed storage for SIMD access |
| `prefilter.rs` | Not present | **INT8 Prefilter** | Approximate statistics for cascade search pruning |

### Byte-Level + Spatial Operations

| Module | Upstream | **Fork** | Detail |
|--------|----------|----------|--------|
| `byte_scan.rs` | Not present | **Byte Scan** | AVX-512 byte_find_all/byte_count (VPCMPEQB + KMOV) |
| `nibble.rs` | Not present | **Nibble Ops** | 4-bit unpack/threshold (AVX2 vpshufb) |
| `distance.rs` | Not present | **3D Distance** | Squared distance (AVX2 batch) |
| `spatial_hash.rs` | Not present | **Spatial Hash** | Batch radius query (AVX2 accelerated) |
| `aabb.rs` | Not present | **AABB** | Axis-aligned bounding box intersection |
| `bitwise.rs` | Not present | **Bitwise** | XOR, AND, OR, popcount on 8KB+ vectors |

### Search + Trees

| Module | Upstream | **Fork** | Detail |
|--------|----------|----------|--------|
| `clam.rs` | Not present | **CLAM Tree** | Build + search + rho_nn (46 tests) |
| `clam_search.rs` | Not present | **CLAM Search** | k-NN and range search on CLAM index |
| `clam_compress.rs` | Not present | **CLAM Compress** | Index compression for storage |
| `cascade.rs` | Not present | **HDR Cascade** | Sigma-band filtering, ranked hits, drift detection |
| `parallel_search.rs` | Not present | **Parallel Search** | Multi-threaded CLAM search |
| `dn_tree.rs` | Not present | **DN Tree** | Hierarchical path resolution |
| `merkle_tree.rs` | Not present | **Merkle Tree** | Hash-based integrity verification |

### Model Inference + AI

| Module | Upstream | **Fork** | Detail |
|--------|----------|----------|--------|
| `gguf.rs` | Not present | **GGUF Reader** | GGUF format parser (LLaMA, Qwen, Gemma) |
| `gguf_indexer.rs` | Not present | **GGUF Indexer** | Build bgz7 codebook index from GGUF weights |
| `safetensors.rs` | Not present | **Safetensors** | HuggingFace safetensors reader |
| `gpt2/` (4 files) | Not present | **GPT-2** | Inference engine (weights, layers, API) |
| `openchat/` (4 files) | Not present | **OpenChat** | Inference engine for OpenChat models |
| `stable_diffusion/` (7 files) | Not present | **Stable Diffusion** | CLIP, UNet, VAE, scheduler (image generation) |
| `models/` (5 files) | Not present | **Model Router** | Multi-model router, safetensors loader, layer abstractions |
| `jina/` (5 files) | Not present | **Jina v5** | Embedding cache, causal attention, codec, runtime |

### Cognitive Primitives

| Module | Upstream | **Fork** | Detail |
|--------|----------|----------|--------|
| `nars.rs` | Not present | **NARS** | Non-Axiomatic Reasoning System inference |
| `qualia.rs` | Not present | **Qualia** | Felt-sense quality encoding |
| `qualia_gate.rs` | Not present | **Qualia Gate** | Gated operations on quality values |
| `hdc.rs` | Not present | **HDC** | Hyperdimensional Computing primitives |
| `vsa.rs` | Not present | **VSA** | Vector Symbolic Architecture operations |
| `spo_bundle.rs` | Not present | **SPO Bundle** | Subject-Predicate-Object triple encoding |
| `causality.rs` | Not present | **Causality** | Causal graph operations |
| `causal_diff.rs` | Not present | **CausalEdge64** | u64-packed causal edges, quality scoring |
| `bf16_truth.rs` | Not present | **BF16 Truth** | Truth values in BF16 precision |
| `styles/` (34 files) | Not present | **Thinking Styles** | 34 cognitive primitives: rte, htd, smad, tcp, irs, mcp, tca, cdt, mct, lsi, pso, cdi, cws, are, tcf, ssr, etd, amp, zcf, hpm, cur, mpc, ssam, idr, spp, icr, sdd, dtmf, hkf |
| `blackboard.rs` | Not present | **Blackboard** | Typed slot arena (zero-copy shared memory) |
| `node.rs` | Not present | **Node** | Cognitive node representation |
| `plane.rs` | Not present | **Plane** | 16Kbit representation plane |
| `seal.rs` | Not present | **Seal** | Immutable snapshot encoding |
| `substrate.rs` | Not present | **Substrate** | Cognitive substrate operations |
| `binding_matrix.rs` | Not present | **Binding Matrix** | 3D permutation binding |
| `cyclic_bundle.rs` | Not present | **Cyclic Bundle** | Cyclic vector bundling |

### JIT Compilation

| Module | Upstream | **Fork** | Detail |
|--------|----------|----------|--------|
| `jitson/` (8 files) | Not present | **JITSON** | JSON parser + validator + template + scan pipeline |
| `jitson_cranelift/` (6 files) | Not present | **Cranelift JIT** | AVX-512 kernel compilation via Cranelift (feature-gated) |

### Audio / OCR / Media

| Module | Upstream | **Fork** | Detail |
|--------|----------|----------|--------|
| `holo.rs` | Not present | **Holographic** | Holographic reduced representations, cosine carriers |
| `ocr_felt.rs` | Not present | **OCR** | Character recognition via felt-sense matching |
| `ocr_simd.rs` | Not present | **OCR SIMD** | SIMD-accelerated binarization, Otsu threshold, density |
| `surround_metadata.rs` | Not present | **Surround** | Spatial audio metadata |
| `crystal_encoder.rs` | Not present | **Crystal** | Crystal symmetry encoding |

### Miscellaneous

| Module | Upstream | **Fork** | Detail |
|--------|----------|----------|--------|
| `arrow_bridge.rs` | Not present | **Arrow** | Apache Arrow zero-copy bridge |
| `bnn.rs` | Not present | **BNN** | Binary Neural Network operations |
| `bnn_causal_trajectory.rs` | Not present | **BNN Causal** | Causal trajectory tracking |
| `bnn_cross_plane.rs` | Not present | **BNN Cross-Plane** | Cross-plane BNN operations |
| `cogrecord.rs` | Not present | **CogRecord** | 4×16KB cognitive record unit |
| `compression_curves.rs` | Not present | **Compression** | Rate-distortion curve analysis |
| `graph.rs` | Not present | **Graph** | Basic graph operations |
| `heel_f64x8.rs` | Not present | **F64x8 Kernels** | SIMD dot product, cosine similarity |
| `http_reader.rs` | Not present | **HTTP Reader** | Stream weights from HTTP |
| `kernels.rs` | Not present | **SIMD Kernels** | Generic SIMD apply/map/reduce |
| `layered_distance.rs` | Not present | **Layered Distance** | Multi-layer distance computation |
| `organic.rs` | Not present | **Organic** | Organic growth patterns |
| `p64_bridge.rs` | Not present | **P64 Bridge** | Palette64 convergence point (ndarray <-> lance-graph) |
| `projection.rs` | Not present | **Projection** | Dimensionality reduction |
| `property_mask.rs` | Not present | **Property Mask** | Bitwise property filtering |
| `tekamolo.rs` | Not present | **Tekamolo** | Syntactic position encoding |
| `udf_kernels.rs` | Not present | **UDF Kernels** | User-defined function dispatch |
| `deepnsm.rs` | Not present | **DeepNSM** | Distributional semantic bridge |

## Subcrates (2 crates)

| Crate | Upstream | **Fork** | Detail |
|-------|----------|----------|--------|
| `crates/p64` | Not present | **P64** | Palette64 data structure — convergence highway between ndarray and lance-graph |
| `crates/phyllotactic-manifold` | Not present | **Phyllotactic Manifold** | Golden-angle spiral geometry for uniform point distribution |

## Burn Backend (20 ops files)

| Component | Upstream | **Fork** | Detail |
|-----------|----------|----------|--------|
| `crates/burn/` | Not present | **burn-ndarray** | SIMD-augmented burn backend (from tracel-ai/burn v0.21.0) |
| `ops/tensor.rs` | — | **try_vml_unary** | Routes f32 unary ops through ndarray hpc::vml (F32x16 SIMD) |
| `ops/activation.rs` | — | **Fused sigmoid** | SIMD-accelerated activation functions |
| `ops/matmul.rs` | — | **GEMM dispatch** | Routes to our Goto-algorithm GEMM |
| Remaining 17 ops files | — | **Standard burn ops** | conv, pooling, interpolate, quantization, etc. |

## Summary

| Category | Upstream Count | **Fork Count** | New |
|----------|---------------|----------------|-----|
| SIMD type files | 0 | 6 | +6 |
| Backend files | 0 | 5 | +5 |
| HPC modules | 0 | 146 | +146 |
| Burn ops | 0 | 20 | +20 |
| Subcrates | 0 | 2 | +2 |
| **Total new files** | — | — | **179** |
| **Total new LOC** | — | — | **80,131** |
| **Total new tests** | — | — | **~880** |
