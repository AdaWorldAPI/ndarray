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

### Test Summary (STALE — see audit below)
- ~~**286 lib tests passing** (209 original + 77 new cognitive layer tests)~~ → **880 lib tests passing** (2026-03-22 audit)
- **Clippy clean** (`cargo clippy -- -D warnings`)
- ~~**All doctests passing**~~ → **2 doctest failures** out of 302

---

## Epoch 4 Completion Status (2026-03-22 Audit)

> The original blackboard test counts and "must be ported" checklist were massively stale.
> Every module has significantly more tests than originally documented. All porting work is complete.

### HPC Module Inventory (55 files in src/hpc/)

**Core types (Step 3a)** — all DONE, test counts grew:
| Module | Original claim | Actual tests |
|--------|---------------|-------------|
| fingerprint.rs | 12 | 12 |
| plane.rs | 8 | 16 |
| seal.rs | 5 | 4 |
| node.rs | 6 | 9 |
| cascade.rs | 5 | 12 |
| bf16_truth.rs | 8 | 23 |
| causality.rs | 6 | 17 |
| blackboard.rs | 10 | 36 |

**Additional crates (Step 3b)** — all DONE:
| Module | Original claim | Actual tests |
|--------|---------------|-------------|
| bnn.rs | 6 | 26 |
| clam.rs | 7 | 46 |
| arrow_bridge.rs | 5 | 26 |

**BLAS / Numerical** — ALL DONE:
- blas_level1.rs (11 tests), blas_level2.rs (10), blas_level3.rs (5)
- fft.rs (3), lapack.rs (4), vml.rs (5), statistics.rs (11), quantized.rs (7), activations.rs (9)

**Cognitive / Search / Advanced** — ALL DONE (27 additional modules, ~469 tests):
- nars, qualia, qualia_gate, hdc, spo_bundle, cogrecord, graph, merkle_tree
- cam_index, prefilter, clam_search, clam_compress, parallel_search
- crystal_encoder, deepnsm, dn_tree, organic, substrate, tekamolo, vsa
- bnn_cross_plane, bnn_causal_trajectory, binding_matrix
- bgz17_bridge, palette_distance, layered_distance, surround_metadata
- compression_curves, cyclic_bundle, packed, bitwise, kernels, udf_kernels, projection

### Backend Module (6 files in src/backend/)
- BlasFloat trait dispatch: DONE (mod.rs, native.rs)
- MKL FFI: DONE (mkl.rs)
- OpenBLAS FFI: DONE (openblas.rs)
- SIMD compat layer: DONE (simd.rs, simd_avx512.rs, simd_avx2.rs — LazyLock<Tier> AVX-512/AVX2/Scalar)
- AVX-512 kernels: DONE (kernels_avx512.rs)

### Build Status
- Build currently fails (exit 101) — needs investigation
- 880 lib tests pass when build succeeds
- 2 doctest failures out of 302

### Architecture Notes
- `LinalgBackend` trait from CLAUDE.md spec → actual impl is `BlasFloat` trait (different name, same purpose)
- `src/simd/` directory from spec → actual is `src/simd.rs`, `src/simd_avx512.rs`, `src/simd_avx2.rs` (three top-level files)
- `src/vector/` directory from spec → not created (functionality in hpc/)
- Blackboard uses `HashMap<String, Box<dyn Any>>`, not a true 64-byte aligned arena

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

### Must be ported from rustynum (ALL DONE as of 2026-03-22):
- [x] **Backend trait** (BlasFloat — renamed from LinalgBackend) — src/backend/mod.rs + native.rs
- [x] **BLAS L1** — hpc/blas_level1.rs (11 tests)
- [x] **BLAS L1 SIMD** — hpc/blas_level1.rs (ScalarArith + VecArith traits)
- [x] **BLAS L2** — hpc/blas_level2.rs (10 tests)
- [x] **BLAS L3** — hpc/blas_level3.rs (5 tests)
- [x] **BF16 GEMM** — hpc/quantized.rs (7 tests)
- [x] **Int8 GEMM** — hpc/quantized.rs
- [x] **LAPACK** — hpc/lapack.rs (4 tests)
- [x] **FFT** — hpc/fft.rs (3 tests)
- [x] **VML** — hpc/vml.rs (5 tests)
- [x] **Statistics** — hpc/statistics.rs (11 tests)
- [x] **Array ops** — hpc/statistics.rs + hpc/activations.rs (9 tests)
- [x] **HDC** — hpc/hdc.rs (5 tests)
- [x] **Projection** — hpc/projection.rs (4 tests)
- [x] **CogRecord** — hpc/cogrecord.rs (4 tests)
- [x] **Graph** — hpc/graph.rs (4 tests)
- [x] **Binding matrix** — hpc/binding_matrix.rs (9 tests)

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
- Feature gates: native (default), intel-mkl, openblas — mutu