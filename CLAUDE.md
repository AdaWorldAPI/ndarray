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

## Repository Structure (Target)
```
src/
├── lib.rs              # Re-exports, feature gates
├── backend/
│   ├── mod.rs          # LinalgBackend trait
│   ├── native.rs       # Pure Rust + SIMD
│   ├── mkl.rs          # Intel MKL FFI (feature = "intel-mkl")
│   └── openblas.rs     # OpenBLAS FFI (feature = "openblas")
├── linalg/
│   ├── gemm.rs         # General matrix multiply
│   ├── syrk.rs         # Symmetric rank-k update
│   └── trsm.rs         # Triangular solve
├── simd/
│   ├── mod.rs          # Runtime detection, dispatch
│   ├── avx512.rs       # AVX-512 kernels
│   ├── avx2.rs         # AVX2 fallback
│   └── sse42.rs        # SSE4.2 minimum
└── vector/
    ├── distance.rs     # Cosine, L2, dot product
    ├── batch.rs        # Pairwise, top-k
    └── index.rs        # VectorIndex trait
benches/
├── gemm_bench.rs       # criterion benchmarks
└── vector_bench.rs
```
