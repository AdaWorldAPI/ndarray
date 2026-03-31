# Project NDARRAY Expansion

> HPC Rust Transformation — porting `adaworldapi/rustynum` features into this ndarray fork.

## Quick Context
- **What**: High-performance linear algebra with pluggable BLAS backends (Native SIMD, MKL, OpenBLAS)
- **Source**: `adaworldapi/rustynum` — reference GEMM, SIMD, and FFI implementations
- **Target**: This repo — ndarray fork enhanced with HPC backends
- **Rust**: 1.94 Stable only. No nightly features.

## Agent Protocol
This project uses specialized agents in `.claude/agents/`. Follow these rules:

1. **Always read `.claude/blackboard.md` before starting any task**
2. After completing work, update the blackboard with decisions and loose ends
3. Delegate appropriately:
   - GEMM kernels, SIMD, memory layout, Backend trait design → `savant-architect`
   - `unsafe` code, FFI audit, benchmarking → `sentinel-qa`
   - Embedding ops, distance metrics, vector store bridges → `vector-synthesis`
   - API surface, docs, feature gates, Cargo.toml → `product-engineer`
   - Feature prioritization, gap analysis, phase planning → `l3-strategist`
4. When encountering `unsafe` code, **always** delegate to sentinel-qa for audit
5. Write decisions to the blackboard, not just to chat

## Hard Rules
- OpenBLAS and MKL are **mutually exclusive** feature gates. Never both.
- Zero-cost abstractions: generics monomorphize, no `Box<dyn>` in hot paths.
- Every `unsafe` block needs a `// SAFETY:` comment.
- All public APIs need `///` doc comments with examples.
- `cargo clippy -- -D warnings` must pass.

## Compaction Preservation
When summarizing this conversation, preserve:
- All entries in `.claude/blackboard.md`
- Current epoch number and loose ends
- Which agents have been consulted and their verdicts
- Any BLOCK findings from sentinel-qa

## Repository Structure (Actual as of 2026-03-22)
```
src/
├── lib.rs              # Re-exports, feature gates
├── backend/
│   ├── mod.rs          # BlasFloat trait (was planned as LinalgBackend)
│   ├── native.rs       # Pure Rust + SIMD microkernels
│   ├── mkl.rs          # Intel MKL FFI (feature = "intel-mkl")
│   ├── openblas.rs     # OpenBLAS FFI (feature = "openblas")
├── simd.rs             # Consumer-facing SIMD module, re-exports all types
├── simd_avx512.rs      # AVX-512 type definitions (11 types from rustynum)
├── simd_avx2.rs        # AVX2 functions
│   └── kernels_avx512.rs  # AVX-512 kernel implementations
├── hpc/                # 55 modules — ALL DONE (880 lib tests)
│   ├── blas_level1.rs  # BLAS L1 (dot, axpy, scal, nrm2, asum, etc.)
│   ├── blas_level2.rs  # BLAS L2 (gemv, ger, symv, trmv, trsv)
│   ├── blas_level3.rs  # BLAS L3 (gemm, syrk, trsm, symm)
│   ├── quantized.rs    # BF16 GEMM, Int8 GEMM
│   ├── lapack.rs       # LU, Cholesky, QR
│   ├── fft.rs          # FFT/IFFT (Cooley-Tukey radix-2)
│   ├── vml.rs          # Vector math (exp, ln, sqrt, etc.)
│   ├── statistics.rs   # Median, var, std, percentile, top_k
│   ├── activations.rs  # Sigmoid, softmax, log_softmax
│   ├── fingerprint.rs, plane.rs, seal.rs, node.rs  # Cognitive core
│   ├── cascade.rs, bf16_truth.rs, causality.rs     # Truth/cascade
│   ├── blackboard.rs   # Typed slot arena
│   ├── bnn.rs, clam.rs, arrow_bridge.rs            # Additional crates
│   ├── hdc.rs, nars.rs, qualia.rs, spo_bundle.rs   # Cognitive extensions
│   └── ... (27 more modules)
```

## Status (2026-03-22 Audit)
- **All "must be ported" items: DONE** — see `.claude/blackboard.md` for full inventory
- **880 lib tests passing**, 2 doctest failures out of 302
- **Build currently fails (exit 101)** — needs investigation
- See blackboard for detailed per-module test counts

## Session: Qwen3.5 × Opus 4.5/4.6 Reverse Engineering (2026-03-31)

### New Modules
- `src/hpc/styles/` — 34 cognitive primitives (rte, htd, smad, tcp, irs, mcp, tca, cdt, mct, lsi, pso, cdi, cws, are, tcf, ssr, etd, amp, zcf, hpm, cur, mpc, ssam, idr, spp, icr, sdd, dtmf, hkf). Each is `fn(Base17, NarsTruth) → result`. 49 tests.
- `src/hpc/causal_diff.rs` — CausalEdge64 (u64 packed), scaffold_to_palette3d_layers(), quality scoring (GOOD/BAD/UNCERTAIN), NARS self-reinforcement LoRA, PAL8 serialization (4101 bytes).
- `.cargo/config.toml` — `target-cpu=x86-64-v4` (AVX-512 mandatory).
- `src/simd.rs` — compile-time AVX-512 dispatch via `cfg(target_feature = "avx512f")`.

### Key Data
- 5 Qwen3.5 models indexed: 685 MB bgz7 from 201 GB BF16 safetensors
- GitHub Release `v0.1.0-bgz-data` on AdaWorldAPI/lance-graph: 41 bgz7 files
- 4 diffs: FfnGate dominant (0.6%), v2 reverts v1, K stable at 27B, K shifted at 9B

### Benchmark
- SPO Palette Distance: 611M lookups/sec, 1.8 ns/lookup, 388 KB RAM
- 17K tokens/sec (triple model, 4096 heads, Pearl 2³)

### Architecture Rule
- ndarray = hardware (SIMD, Palette, Base17, SpoDistanceMatrices, read_bgz7_file)
- lance-graph = thinking (NarsTruth, NarsEngine, TripleModel, AutocompleteCache)
- causal-edge = protocol (CausalEdge64, NarsTables, forward/learn)
- p64 = convergence highway (both repos meet here)
