# HPC API Inventory — AdaWorldAPI/ndarray fork

**Generated**: 2026-05-18
**Branch**: `claude/lance-surrealdb-analysis-LXmug`
**Purpose**: Catalogue of the existing public HPC surface relevant to the
lance-graph ↔ surrealdb ↔ ndarray integration plan §1 stable-surface commitment.

---

## 1. Discovered HPC Modules

The `src/hpc/` directory contains **~100 Rust source files** across flat and
nested layouts. Below are the modules relevant to distance computation and the
stable surface claimed by the integration plan.

| Module | File | Notes |
|---|---|---|
| `hpc::heel_f64x8` | `src/hpc/heel_f64x8.rs` | **Primary distance surface** — SIMD cosine + HEEL plane Hamming |
| `hpc::distance` | `src/hpc/distance.rs` | Spatial k-NN, squared L2, radius filter (f32 AVX2 + f64 scalar) |
| `hpc::bitwise` | `src/hpc/bitwise.rs` | `hamming_distance_raw`, `popcount_raw`, batch Hamming + top-k |
| `hpc::palette_distance` | `src/hpc/palette_distance.rs` | Palette/SPO distance matrices (`SpoDistanceMatrices`) |
| `hpc::layered_distance` | `src/hpc/layered_distance.rs` | Lance-graph container layout (`[u64; 256]`), `palette_distance()` |
| `hpc::parallel_search` | `src/hpc/parallel_search.rs` | `parallel_search`, `lfd_from_palette`, `PaletteScope` |
| `hpc::cam_pq` | `src/hpc/cam_pq.rs` | `squared_l2`, `kmeans`, `CamCodebook`, `DistanceTables` |
| `hpc::blas_level1` | `src/hpc/blas_level1.rs` | `dot_f32/f64`, `nrm2_f32/f64`, `axpy_f32/f64`, `blas_rotg` |
| `hpc::vml` | `src/hpc/vml.rs` | `vsexp`, `vdexp`, `vsln`, `vdln`, `vssqrt`, `vdsqrt`, `vsabs`, `vdabs`, etc. |
| `hpc::reductions` | `src/hpc/reductions.rs` | `sum_f32/f64`, `mean_f32/f64`, `max/min_f32`, `argmax/argmin_f32`, `nrm2_f32` |
| `hpc::simd_caps` | `src/hpc/simd_caps.rs` | Runtime SIMD capability singleton |
| `hpc::simd_dispatch` | `src/hpc/simd_dispatch.rs` | `LazyLock`-frozen SIMD dispatch function pointers |
| `hpc::fingerprint` | `src/hpc/fingerprint.rs` | `Fingerprint<N>`, `Fingerprint1K/2K/64K`, `VectorConfig` |
| `hpc::clam` | `src/hpc/clam.rs` | `knn_brute`, `ClamTree::build` |
| `hpc::prefilter` | `src/hpc/prefilter.rs` | `approx_hamming_candidates` |
| `hpc::cyclic_bundle` | `src/hpc/cyclic_bundle.rs` | `hamming_128`, `cyclic_shift`, `bundle_spo` |
| `hpc::zeck` | `src/hpc/zeck.rs` | ZeckF64 progressive edge encoding, `hamming_distance_raw` consumer |
| `hpc::holo` | `src/hpc/holo.rs` | Phase-space holographic ops: `focus_hamming`, `focus_l1`, `wasserstein_sorted_i8` |

Additionally gated behind `feature = "hpc-extras"`:

| Module | File |
|---|---|
| `hpc::spo_bundle` | `src/hpc/spo_bundle.rs` |
| `hpc::deepnsm` | `src/hpc/deepnsm.rs` |
| `hpc::compression_curves` | `src/hpc/compression_curves.rs` |
| `hpc::crystal_encoder` | `src/hpc/crystal_encoder.rs` |
| `hpc::p64_bridge` | `src/hpc/p64_bridge.rs` |

---

## 2. F64x8 Type — Actual Definition

### AVX-512 path (canonical production backend)

**File**: `src/simd_avx512.rs`
**Line**: 304 (struct definition) / 314 (LANES constant)

```rust
// src/simd_avx512.rs:302–304
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F64x8(pub __m512d);

// src/simd_avx512.rs:314
pub const LANES: usize = 8;
```

Repr: `__m512d` — a native 512-bit AVX-512 register holding 8 × `f64`.
Lane count: **8**.
Backing: `_mm512_loadu_pd` (unaligned load), `_mm512_storeu_pd` (unaligned store).

Key methods available on `F64x8` (`src/simd_avx512.rs`):
`splat(v: f64)`, `from_slice(&[f64])`, `from_array([f64; 8])`, `to_array()`,
`copy_to_slice(&mut [f64])`, `reduce_sum()`, `reduce_min()`, `reduce_max()`,
`abs()`, `sqrt()`, `round()`, `floor()`, `mul_add(b, c)`, `simd_min/max/clamp`,
`simd_lt/le/gt/ge/eq/ne`, `to_bits()`, `from_bits()`.

### AVX2 fallback (non-AVX-512 x86_64)

**File**: `src/simd_avx2.rs`
The AVX2 path supplies `F64x8` as a polyfill backed by two `__m256d` (2 × 4
lanes). Same public API surface as the AVX-512 variant; `impl_float_type!` macro
used at line ~820 of `simd_avx2.rs`.

### Scalar fallback (non-x86 targets)

**File**: `src/simd.rs`, scalar module (not-x86 cfg block, line ~789)
```rust
impl_float_type!(F64x8, f64, 8, F64Mask8, u8);
```
Backed by `[f64; 8]`. Same API.

### Re-export path

`src/simd.rs:244` (AVX-512 path) / `src/simd.rs:280` (AVX2 fallback) / `src/simd.rs:1573` (NEON)
→ `pub use crate::simd::F64x8;` is the canonical consumer entry point.

---

## 3. `heel_f64x8` Functions — Signatures and File:Line

**File**: `src/hpc/heel_f64x8.rs`

| Function | Signature | Line | Description |
|---|---|---|---|
| `heel_weighted_distance` | `(distances: &[f64; 8], weights: &[f64; 8]) -> f64` | 23 | Weighted dot via F64x8 FMA; single vmulpd+vreducepd on AVX-512 |
| `heel_plane_distances` | `(a: &[u64; 8], b: &[u64; 8]) -> [f64; 8]` | 34 | Hamming (popcount of XOR) per plane → 8 f64 distances |
| `heel_weighted_hamming` | `(a_planes: &[u64; 8], b_planes: &[u64; 8], weights: &[f64; 8]) -> f64` | 44 | Full pipeline: planes → per-plane Hamming → weighted dot |
| `dot_f64_simd` | `(a: &[f64], b: &[f64]) -> f64` | 64 | SIMD dot product; 8 f64 per iteration with FMA accumulation |
| `sum_sq_f64_simd` | `(a: &[f64]) -> f64` | 86 | Sum of squares via F64x8 FMA |
| `cosine_f64_simd` | `(a: &[f64], b: &[f64]) -> f64` | 109 | SIMD cosine similarity, single-pass dot+norms |
| `cosine_f32_to_f64_simd` | `(a: &[f32], b: &[f32]) -> f64` | 149 | f32 inputs, f64 precision cosine via scalar widening + F64x8 FMA |

**Constants also defined**:
- `UNIFORM_WEIGHTS: [f64; 8] = [1.0; 8]` — line 50
- `HEEL_7PLUS1_WEIGHTS: [f64; 8] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]` — line 54

### Integration plan claim vs reality

The integration plan (`lance-graph/.claude/plans/integration-plan.md:32`) states:
> `hpc-extras` feature, `heel_f64x8::cosine_f64_simd` etc.

And the contracts table (line 58) states:
> `ndarray::hpc::F64x8` + `heel_f64x8::*` — ndarray 0.17 fork, stable

**Verdict: PRESENT and matches.** `cosine_f64_simd` exists at
`src/hpc/heel_f64x8.rs:109` with signature `(a: &[f64], b: &[f64]) -> f64`.

**Additional functions the plan's "etc." implies but does not name explicitly
are also present**: `heel_weighted_hamming`, `heel_plane_distances`,
`heel_weighted_distance`, `dot_f64_simd`, `sum_sq_f64_simd`, `cosine_f32_to_f64_simd`.

---

## 4. Other Distance Kernels Found

### 4a. Hamming (binary Hamming distance)

| Function | File | Line | Signature |
|---|---|---|---|
| `hamming_distance_raw` | `src/hpc/bitwise.rs` | 180 | `(a: &[u8], b: &[u8]) -> u64` |
| `popcount_raw` | `src/hpc/bitwise.rs` | 185 | `(a: &[u8]) -> u64` |
| `hamming_batch_raw` | `src/hpc/bitwise.rs` | 193 | `(query, database, num_rows, row_bytes) -> Vec<u64>` |
| `hamming_top_k_raw` | `src/hpc/bitwise.rs` | 201 | `(query, database, num_rows, row_bytes, k) -> Vec<(usize,u64)>` |
| `hamming_distance` | `src/simd_avx2.rs` | 276 | `(a: &[u8], b: &[u8]) -> u64` (AVX2-specific) |
| `hamming_batch` | `src/simd_avx2.rs` | 316 | `(query, database, num_rows, row_bytes) -> Vec<u64>` |
| `hamming_top_k` | `src/simd_avx2.rs` | 338 | `(query, database, num_rows, row_bytes, k) -> Vec<(usize,u64)>` |
| `hamming_128` | `src/hpc/cyclic_bundle.rs` | 153 | `(a: &[u64; N], b: &[u64; N]) -> u32` (128×64-bit) |
| `hamming_u8x16` | `src/simd_neon.rs` | 74 | `unsafe (a: &[u8; 16], b: &[u8; 16]) -> u32` (NEON) |
| `focus_hamming` | `src/hpc/holo.rs` | ~1951 | `(a: &[u8], b: &[u8], mask_x, mask_y, mask_z) -> (u64, u32)` |
| `approx_hamming_candidates` | `src/hpc/prefilter.rs` | 252 | `(query, db, bytes_per_vec, n_vectors, k) -> Vec<(usize,u32)>` |

Re-exported to `ndarray::simd` namespace:
- `src/simd.rs:1714`: `pub use crate::hpc::bitwise::{hamming_distance_raw, popcount_raw};`

### 4b. Cosine Similarity

| Function | File | Line | Signature |
|---|---|---|---|
| `cosine_f64_simd` | `src/hpc/heel_f64x8.rs` | 109 | `(a: &[f64], b: &[f64]) -> f64` |
| `cosine_f32_to_f64_simd` | `src/hpc/heel_f64x8.rs` | 149 | `(a: &[f32], b: &[f32]) -> f64` |

Re-exported to `ndarray::simd` namespace:
- `src/simd.rs:1751`: `pub use crate::hpc::heel_f64x8::cosine_f32_to_f64_simd;`
- `cosine_f64_simd` is **NOT** re-exported at the `ndarray::simd` level (only `cosine_f32_to_f64_simd` is). Consumers must import directly from `ndarray::hpc::heel_f64x8::cosine_f64_simd`.

### 4c. L2 / Squared L2

| Function | File | Line | Signature |
|---|---|---|---|
| `squared_l2` | `src/hpc/cam_pq.rs` | 473 | `(a: &[f32], b: &[f32]) -> f32` |
| `squared_distances_f32` | `src/hpc/distance.rs` | 98 | `(query: [f32;3], points: &[[f32;3]]) -> Vec<f32>` |
| `squared_distances_f64` | `src/hpc/distance.rs` | 142 | `(query: [f64;3], points: &[[f64;3]]) -> Vec<f64>` |
| `knn_f32` | `src/hpc/distance.rs` | 124 | `(query: [f32;3], points: &[[f32;3]], k) -> (Vec<usize>, Vec<f32>)` |
| `knn_f64` | `src/hpc/distance.rs` | 158 | `(query: [f64;3], points: &[[f64;3]], k) -> (Vec<usize>, Vec<f64>)` |
| `filter_by_radius_sq` | `src/hpc/distance.rs` | 113 | `(query: [f32;3], points, radius_sq) -> Vec<usize>` |
| `filter_by_radius_sq_f64` | `src/hpc/distance.rs` | 147 | `(query: [f64;3], points, radius_sq) -> Vec<usize>` |

Re-exported to `ndarray::simd` namespace:
- `src/simd.rs:1747`: `pub use crate::hpc::cam_pq::{kmeans, squared_l2};`

### 4d. L1 (Holographic / phase-space variants)

**No generic `l1_f64_simd` or `l1_f64` free function exists at the top-level HPC surface.**

L1-style distance found only in specialized contexts:

| Function | File | Signature | Context |
|---|---|---|---|
| `focus_l1` | `src/hpc/holo.rs` | `(a: &[u8], b: &[u8], mask_x, mask_y, mask_z) -> (u64, u32)` | Holographic phase-masked L1 |
| `wasserstein_sorted_i8` | `src/hpc/holo.rs` | `(a: &[u8], b: &[u8]) -> u64` | Wasserstein-style L1 distance |
| `carrier_distance_l1` | `src/hpc/holo.rs` | `(a: &[i8], b: &[i8]) -> u64` | Carrier-wave L1 distance |
| `histogram_l1_distance` | `src/hpc/holo.rs` | `(a: &[u16;16], b: &[u16;16]) -> u32` | Histogram L1 |
| `asum_f32` / `asum_f64` | `src/simd_avx2.rs` | `(x: &[f32]) -> f32` | L1 norm (sum of absolutes), not pairwise distance |

### 4e. Linf (Chebyshev) Distance

**No `linf_f64_simd` or generic Linf pairwise distance function exists in the HPC surface.** Not found anywhere in `src/`. `reduce_max()` on `F64x8` provides max-element reduction as a building block, but no composed Linf kernel is exported.

### 4f. Dot Products (BLAS L1 level)

| Function | File | Line | Signature |
|---|---|---|---|
| `dot_f32` | `src/simd_avx2.rs` | 56 | `(a: &[f32], b: &[f32]) -> f32` |
| `dot_f64` | `src/simd_avx2.rs` | 88 | `(a: &[f64], b: &[f64]) -> f64` |
| `dot_f64_simd` | `src/hpc/heel_f64x8.rs` | 64 | `(a: &[f64], b: &[f64]) -> f64` (F64x8 FMA path) |
| `dot_i8` | `src/simd_avx2.rs` | 406 | `(a: &[u8], b: &[u8]) -> i64` |

### 4g. Palette / SPO Distance

| Function | File | Signature |
|---|---|---|
| `palette_distance` | `src/hpc/layered_distance.rs:62` | `(dm: &SpoDistanceMatrices, a: &[u64;256], b: &[u64;256]) -> u32` |
| `SpoDistanceMatrices::spo_distance` | `src/hpc/palette_distance.rs:345` | `(&self, a_s, a_p, a_o, b_s, b_p, b_o) -> u32` |
| `DistanceTables::distance` | `src/hpc/cam_pq.rs:189` | `(&self, cam: &CamFingerprint) -> f32` |
| `parallel_search` | `src/hpc/parallel_search.rs:229` | `(scope: &PaletteScope, query: &PaletteEdge, k, gate: &TruthGate) -> Vec<SearchResult>` |

---

## 5. Feature Flags

### `hpc-extras` (defined `Cargo.toml:207`)

```toml
hpc-extras = ["std", "dep:p64", "dep:fractal", "fractal/std"]
```

Pulls in: `p64` (Palette64/3D attention NARS bridge) and `fractal` (manifold math).

Modules **gated** behind `hpc-extras` (from `src/hpc/mod.rs`):
- `hpc::spo_bundle` (line 121)
- `hpc::deepnsm` (line 124)
- `hpc::compression_curves` (line 131)
- `hpc::crystal_encoder` (line 134)
- `hpc::p64_bridge` (line 141)
- `jitson_cranelift` sub-module (gated separately on `jit-native`)
- `splat3d` (gated separately on `splat3d`)
- The `e2e_tests` integration test block (line 252)

**Default**: `hpc-extras` IS included in the crate default features (`Cargo.toml:174`):
```toml
default = ["std", "hpc-extras"]
```

Modules **not** gated behind `hpc-extras` (unconditionally compiled when `std` is on):
All of `heel_f64x8`, `distance`, `bitwise`, `blas_level1/2/3`, `cam_pq`, `fingerprint`,
`clam`, `prefilter`, `palette_distance`, `layered_distance`, `parallel_search`,
`holo`, `cyclic_bundle`, `vml`, `reductions`, etc.

### Other relevant feature flags

| Flag | Defined at | Purpose |
|---|---|---|
| `std` | `Cargo.toml:182` | Enables `hpc` module + blake3 for cognitive substrate |
| `native` | `Cargo.toml:219` | HPC backend: pure Rust + SIMD |
| `intel-mkl` | `Cargo.toml:220` | HPC backend: Intel MKL FFI (mutually exclusive with openblas) |
| `openblas` | `Cargo.toml:221` | HPC backend: OpenBLAS FFI (mutually exclusive with intel-mkl) |
| `jit-native` | `Cargo.toml:215` | Cranelift JIT backend |
| `splat3d` | `Cargo.toml:231` | CPU-SIMD 3D Gaussian Splatting |
| `nightly-simd` | `Cargo.toml:197` | Portable-SIMD miri-compatible backend (nightly only) |

---

## 6. Cross-References to Consumers

### lance-graph (`/home/user/lance-graph`)

The integration plan (`integration-plan.md:58`) explicitly contracts:
> `ndarray::hpc::F64x8` + `heel_f64x8::*` — stable, unchanged

Consuming crates in lance-graph that reference the ndarray HPC surface:

| Consumer file | ndarray function used | Reference |
|---|---|---|
| `crates/lance-graph/src/graph/blasgraph/ndarray_bridge.rs` | `hamming_distance_raw`, `U8x64::nibble_popcount_lut` | knowledge doc W1b row |
| `crates/lance-graph/src/graph/neighborhood/zeckf64.rs` | ZeckF64 (ndarray canonical copy) | `hpc::zeck` |
| `crates/lance-graph-contract/src/mul.rs` | `I8x16::from_i4_packed_u64`, `batch_packed_i4_16` | W1a W1b plan |
| `crates/holograph/hamming.rs` | `hamming_distance_raw`, `U64x8::popcnt` (W1a planned) | knowledge doc W1b row |
| `crates/bgz17/src/simd.rs` | `U16x8::gather_u16` (W1a planned), `hamming_distance_raw` | knowledge doc W1b row |
| `crates/thinking-engine/src/engine.rs` | `BF16x16`, `simd_amx::*`, `Fingerprint<256>` | W1b VNNI route |

Note: the **lance-graph knowledge doc** (`lance-graph/.claude/knowledge/ndarray-vertical-simd-alien-magic.md`)
specifies that `cosine_f64_simd` is part of the stable surface ("etc." in the plan) and that
no raw intrinsics should be used in consumer crates — all SIMD must flow through `ndarray::simd::*`.

### surrealdb (`AdaWorldAPI/surrealdb`)

Referenced via the integration plan; consuming `lance-graph-contract` which depends on ndarray.
The path is indirect: surrealdb → `lance-graph-contract` → ndarray.

The surrealdb vector distance machinery lives at (plan reference, not audited locally):
`surrealdb/core/src/idx/trees/vector.rs`

Plan claims this will consume `ndarray::hpc::heel_f64x8::cosine_f64_simd` via the
lance-graph-contract trait when wired. Not yet wired (integration §5 is a new crate,
`lance-graph-tikv-provider`, not vector indexing).

---

## 7. Gap Analysis — Plan §1 Stable Surface vs Current Reality

The integration plan's stable-surface table (`integration-plan.md:53–58`) claims:
> `ndarray::hpc::F64x8` + `heel_f64x8::*` — ndarray 0.17 fork, stable — unchanged: only new kernels added

### Present and confirmed

| Claimed API | Actual location | Status |
|---|---|---|
| `ndarray::hpc::F64x8` | `src/simd_avx512.rs:304` (AVX-512), polyfill at `simd_avx2.rs`, scalar in `simd.rs` | PRESENT |
| `F64x8::LANES = 8` | `src/simd_avx512.rs:314` | PRESENT |
| `F64x8::splat`, `from_slice`, `from_array`, `to_array`, `reduce_sum`, `mul_add`, `sqrt`, `abs` | `src/simd_avx512.rs:316–434` | PRESENT |
| `heel_f64x8::cosine_f64_simd` | `src/hpc/heel_f64x8.rs:109` | PRESENT — signature `(a: &[f64], b: &[f64]) -> f64` |
| `heel_f64x8::heel_weighted_hamming` | `src/hpc/heel_f64x8.rs:44` | PRESENT |
| `heel_f64x8::heel_plane_distances` | `src/hpc/heel_f64x8.rs:34` | PRESENT |
| `heel_f64x8::heel_weighted_distance` | `src/hpc/heel_f64x8.rs:23` | PRESENT |
| `heel_f64x8::dot_f64_simd` | `src/hpc/heel_f64x8.rs:64` | PRESENT |
| `heel_f64x8::cosine_f32_to_f64_simd` | `src/hpc/heel_f64x8.rs:149` | PRESENT; also re-exported at `ndarray::simd` level |
| `hamming_distance_raw` | `src/hpc/bitwise.rs:180`; re-exported `simd.rs:1714` | PRESENT |
| `squared_l2` | `src/hpc/cam_pq.rs:473`; re-exported `simd.rs:1747` | PRESENT |

### Missing from the plan's implied surface

| Claimed / implied API | Status | Notes |
|---|---|---|
| `l1_f64_simd` or generic pairwise L1 | **ABSENT** | Only L1-norm variants (`asum_f32/f64`) and specialized `focus_l1` / `carrier_distance_l1` / `wasserstein_sorted_i8` exist. No generic `l1_f64_simd(a: &[f64], b: &[f64]) -> f64`. |
| `l2_f64_simd` as free function | **ABSENT** | `squared_l2` exists for f32; no `l2_f64_simd(a: &[f64], b: &[f64]) -> f64` free function. The L2 distance on 3D points exists in `distance.rs` but is not a general-purpose slice kernel. |
| `linf_f64_simd` | **ABSENT** | No Linf / Chebyshev distance function at any level. |
| `cosine_f64_simd` re-export in `ndarray::simd` | **ABSENT** | `cosine_f32_to_f64_simd` IS re-exported at `simd.rs:1751`. `cosine_f64_simd` is **not** — consumers must import from `ndarray::hpc::heel_f64x8`. |
| `hamming_distance_raw` gating on `hpc-extras` | **NOT REQUIRED** — present unconditionally | `hamming_distance_raw` lives in `hpc::bitwise` which is not behind `hpc-extras`; always available with `std`. |

### W1a primitives claimed by the knowledge doc — current status

From `lance-graph/.claude/knowledge/ndarray-vertical-simd-alien-magic.md` §W1a table:

| W1a primitive | Status in ndarray today |
|---|---|
| `I8x16::from_i4_packed_u64` | **ABSENT** — not in `simd_avx512.rs` or `simd_avx2.rs`; W1a PR pending |
| `I8x16::lane_i8::<N>` | **ABSENT** — generic lane extractor not present |
| `I8x16::saturating_abs` | **ABSENT** — neither a free function nor method |
| `batch_packed_i4_16<E, F>` | **ABSENT** — closure-batch entry point not present |
| `U64x8::xor_popcount` / `U64x8::popcnt` | **ABSENT** — `U64x8` type exists (`simd_avx512.rs:1964`, LANES=8) but `popcnt`/`xor_popcount` methods are not present |
| `U16x8::gather_u16` | **ABSENT** — `U16x32` exists; `U16x8` does not |
| `prefetch_read_t0/t1/t2` | **ABSENT** — no prefetch hint wrappers |
| `U8x32::nibble_popcount_lut` | **ABSENT** — `U8x64::nibble_popcount_lut` exists (`simd_avx512.rs` AVX-512 BITALG path); 32-byte parity is not implemented |

All W1a items are **planned additions** (not yet committed), which is consistent with the
plan's statement that `heel_f64x8::*` is stable and "only new kernels added."

---

## Summary

The integration plan's §1 stable-surface commitment for ndarray resolves to:

- **PRESENT and stable**: `F64x8` type (8-lane f64 SIMD), all `heel_f64x8::*` functions,
  `hamming_distance_raw`, `squared_l2`, `Fingerprint<N>`, `CamCodebook` / `DistanceTables`.
- **ABSENT (not yet added, plan-deferred)**: generic `l1_f64_simd`, `l2_f64_simd`,
  `linf_f64_simd` free-function kernels; all W1a primitives
  (`I8x16::from_i4_packed_u64`, `U64x8::popcnt`, `U16x8::gather_u16`,
  `prefetch_read_t0`, `I8x16::saturating_abs`, `batch_packed_i4_16`).
- **PARTIAL re-export**: `cosine_f32_to_f64_simd` is re-exported at `ndarray::simd`;
  `cosine_f64_simd` is **not** and requires a direct `ndarray::hpc::heel_f64x8` import.
- **`hpc-extras` scope**: The core distance surface (`heel_f64x8::*`, `bitwise::*`,
  `cam_pq::*`, `distance::*`) does **not** require `hpc-extras`; only the p64/fractal
  convergence modules do.
