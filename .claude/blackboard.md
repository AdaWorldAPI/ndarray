# Project NDARRAY Expansion — Blackboard

> Shared state surface for all agents. Read before starting, update after completing work.

## Epoch: 4 — Cognitive Layer Migration
## Global Goal: Port rustynum HPC features into ndarray fork

### Environment
- rust_version: 1.94-stable
- perf_target_blas: MKL (primary), OpenBLAS (alternative)
- simd_level: AVX-512 (primary), AVX2 (fallback), SSE4.2 (minimum)

---

## Cognitive Layer Migration Status

### Core Types (Step 3a — rustynum-core)

| Module | Source | Status | Tests | Notes |
|--------|--------|--------|-------|-------|
| `hpc/fingerprint.rs` | `rustynum-core/fingerprint.rs` | ✅ Done | 12 pass | Const-generic `Fingerprint<N>`, XOR group, SIMD hamming via `bitwise.rs` |
| `hpc/plane.rs` | `rustynum-core/plane.rs` | ✅ Done | 8 pass | 16384-bit i8 accumulator, L1 resident, 64-byte aligned |
| `hpc/seal.rs` | `rustynum-core/seal.rs` | ✅ Done | 5 pass | Blake3 merkle verification (blake3 dep added) |
| `hpc/node.rs` | `rustynum-core/node.rs` | ✅ Done | 6 pass | SPO cognitive atom, inline SplitMix64 RNG |
| `hpc/cascade.rs` | `rustynum-core/hdr.rs` | ✅ Done | 5 pass | 3-stroke search + PackedDatabase + Welford drift |
| `hpc/bf16_truth.rs` | `rustynum-core/bf16_hamming.rs` | ✅ Done | 8 pass | BF16 weights, awareness classify, PackedQualia |
| `hpc/causality.rs` | `rustynum-core/causality.rs` | ✅ Done | 6 pass | CausalityDirection, NarsTruthValue, decomposition |
| `hpc/blackboard.rs` | `rustynum-core/blackboard.rs` | ✅ Done | 10 pass | Zero-copy arena, 64-byte aligned, split-borrow API |

### Additional Crates (Step 3b)

| Module | Source | Status | Tests | Notes |
|--------|--------|--------|-------|-------|
| `hpc/bnn.rs` | `rustynum-bnn/bnn.rs` | ✅ Done | 6 pass | XNOR+popcount BNN inference, cascade search |
| `hpc/clam.rs` | `rustynum-clam/` | ✅ Done | 7 pass | CLAM tree, rho_nn, knn_brute, XOR compression |
| `hpc/arrow_bridge.rs` | `rustynum-arrow/` | ✅ Done | 5 pass | ThreePlaneFingerprintBuffer, SoakingBuffer, GateState |

### Infrastructure

| Item | Status | Notes |
|------|--------|-------|
| Agent definitions (4) | ✅ Done | cognitive-architect, cascade-architect, truth-architect, migration-tracker |
| Knowledge docs (5) | ✅ Done | plane_node_seal, cascade_search, bf16_truth, hardware_map, constants |
| Prompts transcoded (5) | ✅ Done | 01_clam_qualiacam, 02_crystal_encoder, 03_lance_schema, 04_lance_graph, 05_cross_repo |
| `Cargo.toml` blake3 dep | ✅ Done | `blake3 = "1"` |
| `hpc/mod.rs` declarations | ✅ Done | 11 new modules with `#[allow(missing_docs)]` |

### Test Summary
- **286 lib tests passing** (209 original + 77 new cognitive layer tests)
- **Clippy clean** (`cargo clippy -- -D warnings`)
- **All doctests passing**

---

## Stage 0: Gap Analysis

### Already exists in ndarray:
- [x] Array constructors: zeros, ones, range, linspace, logspace, geomspace
- [x] Element-wise float math: exp, ln, sqrt, sin, cos, tan, abs, floor, ceil, round, etc.
- [x] Dot product (general_mat_mul, general_mat_vec_mul, Dot trait)
- [x] Sum, product, mean (impl_numeric.rs)
- [x] Views: ArrayView, ArrayViewMut, slicing, strides
- [x] Transpose, reshape (via into_shape), swap_axes
- [x] Concatenate, stack (stacking.rs)
- [x] Broadcasting (built-in)
- [x] Clamp
- [x] **Bitwise**: hamming_distance, popcount, hamming_distance_batch (VPOPCNTDQ dispatch wired)
- [x] **SIMD binary**: hamming_batch, hamming_top_k (VPOPCNTDQ + raw-slice API)
- [x] **Cognitive layer**: Fingerprint, Plane, Node, Seal, Cascade, BF16Truth, Causality, Blackboard, BNN, CLAM, ArrowBridge

### Must be ported from rustynum:
- [ ] **Backend trait** (LinalgBackend) — pluggable BLAS dispatch
- [ ] **BLAS L1**: sdot/ddot, saxpy/daxpy, sscal/dscal, snrm2/dnrm2, sasum/dasum, isamax/idamax, scopy/dcopy, sswap/dswap
- [ ] **BLAS L1 SIMD**: add/sub/mul/div scalar+vec (f32/f64) — 16 functions
- [ ] **BLAS L2**: sgemv/dgemv, sger/dger, ssymv/dsymv, strmv/dtrmv, strsv/dtrsv
- [ ] **BLAS L3**: sgemm/dgemm, ssyrk/dsyrk, strsm, ssymm/dsymm
- [ ] **BF16 GEMM**: BF16 type, conversions, bf16_gemm_f32, mixed_precision_gemm
- [ ] **Int8 GEMM**: quantize_f32_to_u8/i8/i4, int8_gemm_i32/f32, per_channel variants
- [ ] **LAPACK**: LU (getrf/getrs), Cholesky (potrf/potrs), QR (geqrf)
- [ ] **FFT**: fft/ifft f32/f64, rfft_f32
- [ ] **VML**: vsexp/vdexp, vsln/vdln, vssqrt/vdsqrt, vsabs/vdabs, vsadd, vsmul, vsdiv, vssin, vscos, vspow
- [ ] **Statistics**: median, var, std, percentile (with axis variants)
- [ ] **Array ops**: argmin, argmax, top_k, cumsum, sigmoid, softmax, log_softmax, cosine_similarity, norm(p,axis,keepdims)
- [ ] **HDC**: bind, permute, bundle, bundle_byte_slices, dot_i8
- [ ] **Projection**: simhash_project, simhash_batch_project, simhash_int8_project
- [ ] **CogRecord**: 4-channel struct, new/zeros/container, hamming_4ch, sweep, to/from_bytes
- [ ] **Graph**: VerbCodebook, encode_edge, decode_target, causality_asymmetry, causality_check, find_non_causal_edges, infer_verb
- [ ] **Binding matrix**: binding_popcount_3d, find_holographic_sweet_spot, find_discriminative_spots

---

## Strategic Analysis
<!-- l3-strategist writes here -->
- Phase 1 (Stages 1-4): Core BLAS — highest impact, enables all downstream
- Phase 2 (Stages 5-6): LAPACK/FFT/VML + Array ops — ML-ready
- Phase 3 (Stages 7-8): HDC/CogRecord — domain-specific
- Phase 4 (Stages 9-10): QA + docs — ship-ready

---

## Architecture Decisions
<!-- savant-architect writes here -->
- LinalgBackend trait: generic monomorphized (no Box<dyn> in hot paths)
- SIMD dispatch: runtime detection via is_x86_feature_detected!
- Feature gates: native (default), intel-mkl, openblas — mutually exclusive
- Extension traits on ArrayBase for new operations
- Cognitive layer: all SIMD through `hpc/bitwise.rs` dispatch (no separate SIMD paths)
- Fingerprint<N>: zero-copy `as_bytes()` via unsafe ptr cast (SAFETY reviewed)
- Blackboard arena: 64-byte aligned allocations, PhantomData !Send/!Sync
- Node RNG: inline SplitMix64 (avoids rustynum dependency)

---

## QA Audit Log
<!-- sentinel-qa writes here -->
- [2026-03-15] VPOPCNTDQ hamming: kernel existed in kernels_avx512.rs but dispatch_hamming() in bitwise.rs only checked AVX2. FIXED: tiered dispatch AVX-512 → AVX2 → scalar. Benchmark: 64Kbit 1.84x → 1.14x.
- [2026-03-15] GEMM: native.rs used naive axpy-based tiling (16-20 GFLOPS). FIXED: ported Goto BLAS with 6×16 f32 / 6×8 f64 microkernels from rustyblas. Benchmark: 31-50 GFLOPS, matches reference.
- [2026-03-15] Correctness: all kernels bit-exact or 1-ULP (FMA rounding). sgemm 64×64 max_abs_err = 0.
- [2026-03-16] Cognitive layer migration: 11 modules ported, 286 lib tests passing, clippy clean, doctests passing.

---

## Loose Ends
- [x] Define feature-gate hierarchy (native/mkl/openblas) → DONE: mutually exclusive
- [x] Backend trait: generic (monomorphized) vs enum dispatch → DONE: monomorphized
- [x] Benchmark harness: custom bench binary with GFLOP/s reporting (see .claude/BENCHMARK_RESULTS.md)
- [ ] CI matrix: which feature combinations to test
- [x] Benchmark all areas at parity with rustynum (see .claude/BENCHMARK_RESULTS.md)
- [x] Cognitive layer migration from rustynum → DONE: 11 modules, 77 tests
- [ ] End-to-end pipeline verification (Step 6 from migration plan)

---

## Agent Handoff Log
<!-- Format: [agent] → [agent]: reason -->
- [2026-03-16] cognitive-architect: Migrated Fingerprint, Plane, Seal, Node
- [2026-03-16] cascade-architect: Migrated Cascade, PackedDatabase
- [2026-03-16] truth-architect: Migrated BF16Truth, Causality
