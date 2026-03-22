# Migration Inventory: rustynum → ndarray/hpc

> Generated 2026-03-22. Covers the full `src/hpc/` surface plus upstream
> `rustynum-core`, `rustyblas`, `rustynum-bnn`, `rustynum-clam`,
> `rustynum-arrow`, and `rustynum-holo` crates.

---

## Section 1 — Module Map & Line Counts

### ndarray `src/hpc/` (target, 36 868 LOC total)

| Module | LOC | Pub items | Origin crate |
|--------|----:|----------:|--------------|
| `activations.rs` | 86 | 1 | rustynum-core |
| `arrow_bridge.rs` | 931 | 26 | rustynum-arrow |
| `bf16_truth.rs` | 680 | 15 | rustynum-core |
| `binding_matrix.rs` | 416 | 6 | rustynum-core |
| `bitwise.rs` | 639 | 4 | rustynum-core |
| `blackboard.rs` | 781 | 48 | rustynum-core |
| `blas_level1.rs` | 278 | 3 | rustyblas |
| `blas_level2.rs` | 321 | 3 | rustyblas |
| `blas_level3.rs` | 345 | 2 | rustyblas |
| `bnn.rs` | 942 | 32 | rustynum-bnn |
| `bnn_causal_trajectory.rs` | 2116 | 62 | rustynum-bnn |
| `bnn_cross_plane.rs` | 1631 | 53 | rustynum-bnn |
| `cam_index.rs` | 478 | 18 | rustynum-core |
| `cascade.rs` | 758 | 17 | rustynum-core |
| `causality.rs` | 468 | 11 | rustynum-core |
| `clam.rs` | 2593 | 59 | rustynum-clam |
| `clam_compress.rs` | 707 | 16 | rustynum-clam |
| `clam_search.rs` | 612 | 6 | rustynum-clam |
| `cogrecord.rs` | 238 | 12 | rustynum-holo |
| `compression_curves.rs` | 1733 | — | ndarray-native |
| `crystal_encoder.rs` | 883 | 16 | ndarray-native |
| `cyclic_bundle.rs` | 741 | 13 | ndarray-native |
| `deepnsm.rs` | 845 | 13 | ndarray-native |
| `dn_tree.rs` | 739 | 21 | rustynum-core |
| `fft.rs` | 209 | 5 | ndarray-native |
| `fingerprint.rs` | 394 | 12 | rustynum-core |
| `graph.rs` | 282 | 15 | ndarray-native |
| `hdc.rs` | 178 | 1 | ndarray-native |
| `kernels.rs` | 1589 | 45 | rustynum-core |
| `lapack.rs` | 310 | 4 | ndarray-native |
| `merkle_tree.rs` | 521 | 8 | ndarray-native |
| `mod.rs` | 301 | — | — |
| `nars.rs` | 747 | 28 | ndarray-native |
| `node.rs` | 312 | 8 | rustynum-core |
| `organic.rs` | 783 | 18 | rustynum-core |
| `packed.rs` | 355 | 13 | rustynum-core |
| `plane.rs` | 758 | 25 | rustynum-core |
| `prefilter.rs` | 448 | 6 | rustynum-core |
| `projection.rs` | 143 | 3 | ndarray-native |
| `qualia.rs` | 613 | 8 | ndarray-native |
| `qualia_gate.rs` | 328 | 14 | rustynum-core |
| `quantized.rs` | 416 | 21 | rustyblas |
| `seal.rs` | 99 | 6 | rustynum-core |
| `spo_bundle.rs` | 1514 | 18 | ndarray-native |
| `statistics.rs` | 325 | 1 | ndarray-native (trait) |
| `substrate.rs` | 933 | 26 | rustynum-core |
| `surround_metadata.rs` | 1283 | 18 | ndarray-native |
| `tekamolo.rs` | 502 | 9 | ndarray-native |
| `udf_kernels.rs` | 789 | 10 | ndarray-native |
| `vml.rs` | 154 | 14 | ndarray-native |
| `vsa.rs` | 727 | 21 | ndarray-native |

### rustynum upstream crates (source, ≈ 57 256 LOC total)

| Crate | LOC | Key modules |
|-------|----:|-------------|
| `rustynum-core` | 26 055 | simd, kernels, bf16_hamming, substrate, blackboard, organic, packed, plane, prefilter, qualia_gate, cam_index, causality, cascade, dn_tree, fingerprint, node, seal, simd_avx2/512, backends/ |
| `rustyblas` | 5 584 | level1, level2, level3, bf16_gemm, int8_gemm |
| `rustynum-bnn` | 5 917 | bnn, causal_trajectory, cross_plane, rif_net_integration, belichtungsmesser |
| `rustynum-clam` | 5 869 | tree, search, compress, qualia_cam, semantic_protocol |
| `rustynum-arrow` | 5 010 | arrow_bridge, fragment_index, three_plane, horizontal_sweep, indexed_cascade, datafusion_bridge, lance_io, channel_index |
| `rustynum-holo` | 8 821 | holograph, focus, carrier, phase, cogrecord_v3, delta_layer, lod_pyramid, holo_search |

---

## Section 2 — BLAS Parity (rustyblas → ndarray)

### Level 1 (`blas_level1.rs` — 278 LOC, 3 pub fns)

| rustyblas `level1.rs` | ndarray `blas_level1.rs` | Status |
|-----------------------|--------------------------|--------|
| `pub fn dot()` | `pub fn dot()` | PRESENT |
| `pub fn axpy()` | `pub fn axpy()` | PRESENT |
| `pub fn nrm2()` | `pub fn nrm2()` | PRESENT |
| `pub fn scal()` | — | MISSING |
| `pub fn asum()` | — | MISSING |
| `pub fn iamax()` | — | MISSING |
| `pub fn swap()` | — | MISSING |
| `pub fn copy()` | — | MISSING |
| `pub fn rotg()` | — | MISSING |

**Gap**: 6 of 9 Level 1 routines missing. Only `dot`, `axpy`, `nrm2` ported.

### Level 2 (`blas_level2.rs` — 321 LOC, 3 pub fns)

| rustyblas `level2.rs` | ndarray `blas_level2.rs` | Status |
|-----------------------|--------------------------|--------|
| `pub fn gemv()` | `pub fn gemv()` | PRESENT |
| `pub fn ger()` | `pub fn ger()` | PRESENT |
| `pub fn trmv()` | `pub fn trmv()` | PRESENT |
| `pub fn trsv()` | — | MISSING |
| `pub fn symv()` | — | MISSING |
| `pub fn syr()` | — | MISSING |
| `pub fn syr2()` | — | MISSING |
| `pub fn gbmv()` | — | MISSING |
| `pub fn sbmv()` | — | MISSING |

**Gap**: 6 of 9 Level 2 routines missing. Only `gemv`, `ger`, `trmv` ported.

### Level 3 (`blas_level3.rs` — 345 LOC, 2 pub fns)

| rustyblas `level3.rs` | ndarray `blas_level3.rs` | Status |
|-----------------------|--------------------------|--------|
| `pub fn gemm()` | `pub fn gemm()` | PRESENT |
| `pub fn syrk()` | `pub fn syrk()` | PRESENT |
| `pub fn trmm()` | — | MISSING |
| `pub fn trsm()` | — | MISSING |
| `pub fn symm()` | — | MISSING |

**Gap**: 3 of 5 Level 3 routines missing. Only `gemm`, `syrk` ported.

### Summary: 15 of 23 BLAS routines not yet migrated.

---

## Section 3 — Statistics Trait (ndarray-native)

The `statistics.rs` module defines a trait `StatisticsExt` with **13 methods** (not derived from rustynum; ndarray-native):

| Method | Signature |
|--------|-----------|
| `sorted` | `fn sorted(&self) -> Array<A, Ix1>` |
| `median` | `fn median(&self) -> A` |
| `variance` | `fn variance(&self) -> A` |
| `std_dev` | `fn std_dev(&self) -> A` |
| `var_axis` | `fn var_axis(&self, axis: Axis) -> Array<A, IxDyn>` |
| `std_axis` | `fn std_axis(&self, axis: Axis) -> Array<A, IxDyn>` |
| `percentile` | `fn percentile(&self, p: A) -> A` |
| `cosine_similarity` | `fn cosine_similarity(&self, other: &Self) -> A` |
| `norm` | `fn norm(&self, p: u32) -> A` |
| `argmax` | `fn argmax(&self) -> usize` |
| `argmin` | `fn argmin(&self) -> usize` |
| `top_k` | `fn top_k(&self, k: usize) -> (Vec<usize>, Vec<A>)` |
| `cumsum` | `fn cumsum(&self) -> Array<A, Ix1>` |

---

## Section 4 — Quantized GEMM Verification

All required quantized GEMM functions from `rustyblas` are present in `ndarray/hpc/quantized.rs`.

### BF16 Type & Conversions

| Symbol | rustyblas `bf16_gemm.rs` | ndarray `quantized.rs` | Match? |
|--------|--------------------------|------------------------|--------|
| `struct BF16(pub u16)` | line 33 | line 26 | YES |
| `BF16::from_f32()` | line 45 (rounded) | line 30 (truncate) | PARTIAL — naming inverted |
| `BF16::to_f32()` | line 56 | line 44 | YES |
| `BF16::ZERO` / `BF16::ONE` | lines 60-61 | — | MISSING |
| `f32_to_bf16_slice()` | line 84 | line 50 | YES |
| `f32_to_bf16_rounded()` | line 125 | line 58 | YES |
| `bf16_to_f32_slice()` | line 191 | line 66 | YES |
| `f32_vec_to_bf16()` | line 230 | line 74 | YES |
| `bf16_vec_to_f32()` | line 237 | line 79 | YES |

### GEMM Functions

| Symbol | rustyblas | ndarray | Match? |
|--------|-----------|---------|--------|
| `bf16_gemm_f32()` | bf16_gemm.rs:257 | quantized.rs:86 | YES |
| `mixed_precision_gemm()` | bf16_gemm.rs:429 | quantized.rs:136 | YES |

### INT8 Quantization & GEMM

| Symbol | rustyblas `int8_gemm.rs` | ndarray `quantized.rs` | Match? |
|--------|--------------------------|------------------------|--------|
| `QuantParams` | line 70 | line 155 | YES |
| `PerChannelQuantParams` | line 79 | line 168 | YES |
| `quantize_f32_to_u8()` | line 92 | line 176 | YES |
| `quantize_f32_to_i8()` | line 199 | line 196 | YES |
| `quantize_per_channel_i8()` | line 272 | line 211 | YES |
| `int8_gemm_i32()` | line 357 | line 238 | YES |
| `int8_gemm_f32()` | line 533 | line 253 | YES |
| `int8_gemm_per_channel_f32()` | line 582 | line 283 | YES |
| `quantize_f32_to_i4()` | line 659 | line 306 | YES |
| `dequantize_i4_to_f32()` | line 742 | line 335 | YES |

### Quantized Gaps

- `BF16::ZERO` and `BF16::ONE` constants exist in rustyblas but not in ndarray.
- `from_f32` / `from_f32_truncate` naming convention is inverted between codebases.

---

## Section 5 — NaN Guard Audit

### Unguarded Division Risks (action required)

| File | Lines | Issue | Severity |
|------|-------|-------|----------|
| `statistics.rs` | 103, 123, 132 | `var_axis()` divides by `ax_len` without guarding for zero-length axis → NaN | **HIGH** |
| `cascade.rs` | 222-223 | Warmup mean/variance divides by `warmup_n` which is `128.min(num_vectors)` — zero if `num_vectors=0` | **MEDIUM** |
| `bf16_truth.rs` | 342-345 | `awareness_classify()` divides by `n_dims` (as f32) without guarding → `0.0/0.0 = NaN` | **MEDIUM** |

### Properly Guarded Divisions (no action needed)

| File | Location | Guard |
|------|----------|-------|
| `statistics.rs` | `median()` line 80 | n=0 early return (line 76-78) |
| `statistics.rs` | `variance()` lines 92, 96 | n=0 early return (line 88-90) |
| `statistics.rs` | `percentile()` line 158 | n=0,1 early returns (lines 151, 154) |
| `statistics.rs` | `cosine_similarity()` line 225 | norm=0 returns 0 (line 222) |
| `clam.rs` | LFD (line 92) | `count_half_r == 0` → return 0.0 (line 89) |
| `clam.rs` | leaf radius mean (line 228) | `num_leaves > 0` guard (line 227) |
| `clam.rs` | percentiles (lines 448-454) | n=0 early return (line 441) |
| `clam.rs` | cluster dist stats (lines 544, 552) | empty returns default (line 537-538) |
| `clam.rs` | inverse LFD (line 791) | clamped to `max(0.1)` (line 790) |
| `clam.rs` | NARS truth (line 1398) | overlap=0 returns ignorance (line 1395) |
| `clam.rs` | CHAODA anomaly (line 1579) | range clamped to `max(1e-10)` (line 1579) |
| `clam.rs` | compression ratio (line 1090) | `compressed_bytes > 0` (line 1089) |
| `cascade.rs` | calibrate (lines 109-110) | empty early return (line 105-106) |
| `cascade.rs` | observe (line 147) | observations incremented before division |
| `cascade.rs` | cosine similarity (lines 396, 428, 449) | norm guards at lines 383/410, 395/427, 445/448 |
| `cascade.rs` | BF16Hamming norm (line 471) | `max_total > 0` (line 470) |
| `bf16_truth.rs` | finest_distance (line 399) | `finest_max > 0` (line 399) |
| `nars.rs` | `from_evidence` (line 65) | `total <= 0.0` returns ignorance (line 62-63) |
| `nars.rs` | `to_evidence` (line 94) | `denom <= 1e-9` guard (line 87-92) |
| `nars.rs` | comparison (line 408) | `denom > 1e-9` guard |

### CLAM Distance Note

CLAM distances are `u64` Hamming — NaN is structurally impossible for the distance values themselves. Only derived floating-point statistics (LFD, anomaly scores, means) carry NaN risk, all audited above.

---

## Section 6 — Not-Yet-Migrated rustynum Modules

These modules exist in the upstream rustynum workspace but have **no counterpart** in ndarray `src/hpc/`:

### rustynum-core

| Module | LOC | Description |
|--------|----:|-------------|
| `simd.rs` | 1 092 | Portable SIMD abstractions |
| `simd_avx2.rs` | 600 | AVX2-specific kernels |
| `simd_avx512.rs` | 2 643 | AVX-512 kernels |
| `simd_isa.rs` | 215 | ISA detection |
| `simd.rs` / `simd_avx512.rs` / `simd_avx2.rs` | — | SIMD compat layer (re-exports, AVX-512 types, AVX2 functions) |
| `hybrid.rs` | 2 355 | Hybrid compute pipeline |
| `jitson.rs` | 1 688 | JIT JSON/binary codec |
| `jit_scan.rs` | 385 | JIT scan operations |
| `hdr.rs` | 631 | Header/metadata format |
| `tail_backend.rs` | 884 | Tail-read backend |
| `soaking.rs` | 407 | Soaking/warmup logic |
| `spatial_resonance.rs` | 758 | Spatial resonance |
| `layer_stack.rs` | 341 | Layer stack abstraction |
| `layout.rs` | 57 | Layout helpers |
| `mkl_ffi.rs` | 430 | MKL FFI bindings |
| `parallel.rs` | 101 | Parallelism utilities |
| `rng.rs` | 117 | RNG utilities |
| `compute.rs` | 316 | Compute dispatch |
| `delta.rs` | 209 | Delta encoding |
| `scalar_fns.rs` | 302 | Scalar math functions |
| `graph_hv.rs` | 869 | Graph hypervector ops |
| `backends/gemm.rs` | 453 | GEMM backend dispatch |
| `backends/popcnt.rs` | 153 | Popcount backend |
| `backends/xsmm.rs` | 659 | XSMM integration |

### rustynum-arrow (partially migrated)

| Module | LOC | In ndarray? |
|--------|----:|-------------|
| `arrow_bridge.rs` | 488 | YES (expanded to 931 LOC) |
| `fragment_index.rs` | 237 | NO |
| `three_plane.rs` | 857 | NO |
| `horizontal_sweep.rs` | 1 135 | NO |
| `indexed_cascade.rs` | 1 005 | NO |
| `datafusion_bridge.rs` | 779 | NO |
| `lance_io.rs` | 174 | NO |
| `channel_index.rs` | 239 | NO |

### rustynum-holo (partially migrated)

| Module | LOC | In ndarray? |
|--------|----:|-------------|
| `cogrecord_v3.rs` | 390 | YES (as `cogrecord.rs`, 238 LOC) |
| `holograph.rs` | 3 788 | NO |
| `focus.rs` | 1 378 | NO |
| `carrier.rs` | 1 090 | NO |
| `phase.rs` | 701 | NO |
| `delta_layer.rs` | 457 | NO |
| `lod_pyramid.rs` | 403 | NO |
| `holo_search.rs` | 477 | NO |

### rustynum-bnn (partially migrated)

| Module | LOC | In ndarray? |
|--------|----:|-------------|
| `bnn.rs` | 1 308 | YES (942 LOC) |
| `causal_trajectory.rs` | 2 072 | YES (2 116 LOC) |
| `cross_plane.rs` | 1 595 | YES (1 631 LOC) |
| `rif_net_integration.rs` | 776 | NO |
| `belichtungsmesser.rs` | 111 | NO |

### rustynum-clam (partially migrated)

| Module | LOC | In ndarray? |
|--------|----:|-------------|
| `tree.rs` | 1 112 | YES (as `clam.rs`, 2 593 LOC — expanded) |
| `search.rs` | 626 | YES (612 LOC) |
| `compress.rs` | 711 | YES (707 LOC) |
| `qualia_cam.rs` | 1 434 | NO |
| `semantic_protocol.rs` | 1 928 | NO |

---

## Section 7 — Priority Recommendations

### P0 — Fix Now (NaN bugs)
1. **`statistics.rs:var_axis()`** — add `if ax_len == 0 { return zeros }` guard
2. **`cascade.rs` warmup** — add `if warmup_n == 0 { return }` guard
3. **`bf16_truth.rs:awareness_classify()`** — add `if n_dims == 0 { return }` guard

### P1 — Complete BLAS Surface
Port remaining 15 BLAS routines from rustyblas (`scal`, `asum`, `iamax`, `swap`, `copy`, `rotg`, `trsv`, `symv`, `syr`, `syr2`, `gbmv`, `sbmv`, `trmm`, `trsm`, `symm`).

### P2 — Quantized Constants
Add `BF16::ZERO` and `BF16::ONE` constants. Reconcile `from_f32` naming convention with rustyblas.

### P3 — Migrate High-Value Modules
- `rustynum-core/simd*.rs` (4 554 LOC) — portable SIMD + AVX2/512 kernels
- `rustynum-core/hybrid.rs` (2 355 LOC) — hybrid compute pipeline
- `rustynum-arrow/*` (4 426 LOC unmigrated) — horizontal_sweep, indexed_cascade, datafusion_bridge
- `rustynum-clam/semantic_protocol.rs` + `qualia_cam.rs` (3 362 LOC)
- `rustynum-holo/*` (8 294 LOC unmigrated) — holograph, focus, carrier, phase

### P4 — Migrate Remaining Modules
- `rustynum-core/jitson.rs` + `jit_scan.rs` (2 073 LOC)
- `rustynum-core/backends/` (1 276 LOC)
- `rustynum-bnn/rif_net_integration.rs` + `belichtungsmesser.rs` (887 LOC)
- `rustynum-core` remaining small modules (~3 642 LOC)
