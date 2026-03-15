# Project NDARRAY Expansion — Blackboard

> Shared state surface for all agents. Read before starting, update after completing work.

## Epoch: 3
## Global Goal: Port rustynum HPC features into ndarray fork

### Environment
- rust_version: 1.94-stable
- perf_target_blas: MKL (primary), OpenBLAS (alternative)
- simd_level: AVX-512 (primary), AVX2 (fallback), SSE4.2 (minimum)

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
- [x] **Bitwise**: hamming_distance, popcount, hamming_distance_batch (VPOPCNTDQ dispatch wired)
- [ ] **Projection**: simhash_project, simhash_batch_project, simhash_int8_project
- [x] **SIMD binary**: hamming_batch, hamming_top_k (VPOPCNTDQ + raw-slice API)
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

---

## API Surface
<!-- product-engineer writes here -->

---

## Vector Operations
<!-- vector-synthesis writes here -->

---

## QA Audit Log
<!-- sentinel-qa writes here -->
- [2026-03-15] VPOPCNTDQ hamming: kernel existed in kernels_avx512.rs but dispatch_hamming() in bitwise.rs only checked AVX2. FIXED: tiered dispatch AVX-512 → AVX2 → scalar. Benchmark: 64Kbit 1.84x → 1.14x.
- [2026-03-15] GEMM: native.rs used naive axpy-based tiling (16-20 GFLOPS). FIXED: ported Goto BLAS with 6×16 f32 / 6×8 f64 microkernels from rustyblas. Benchmark: 31-50 GFLOPS, matches reference.
- [2026-03-15] Correctness: all kernels bit-exact or 1-ULP (FMA rounding). sgemm 64×64 max_abs_err = 0.

---

## Loose Ends
- [x] Define feature-gate hierarchy (native/mkl/openblas) → DONE: mutually exclusive
- [x] Backend trait: generic (monomorphized) vs enum dispatch → DONE: monomorphized
- [x] Benchmark harness: custom bench binary with GFLOP/s reporting (see .claude/BENCHMARK_RESULTS.md)
- [ ] CI matrix: which feature combinations to test
- [x] Benchmark all areas at parity with rustynum (see .claude/BENCHMARK_RESULTS.md)

---

## Agent Handoff Log
<!-- Format: [agent] → [agent]: reason -->
