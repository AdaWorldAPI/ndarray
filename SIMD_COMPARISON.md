# SIMD Wishlist Audit: AdaWorldAPI/ndarray (March 2026)

## Codebase Snapshot

- **57 HPC modules** in `src/hpc/` (52K+ lines)
- **2,846 lines** of portable SIMD polyfill: `src/simd.rs` → `src/simd_avx512.rs` → scalar fallback
- SIMD types: `F32x16`, `F64x8`, `U8x64`, `I32x16`, `U32x16`, `U64x8` (AVX-512) + `f32x8`, `f64x4` (AVX2)
- Full operator overloading, `mul_add` (FMA), `sqrt`, `reduce_sum/min/max`, `simd_clamp`
- AVX2 dot product with 4× unrolled accumulators (`src/simd_avx2.rs:52`)
- Runtime dispatch via `is_x86_feature_detected!` (65+ sites in `bitwise.rs`)

---

## Wishlist Scorecard

| # | Item | Status | Key Evidence |
|---|---|---|---|
| 1 | `simd_map()` | **PARTIAL** | SIMD types exist (`F32x16` etc.), VML scalar. Missing: generic lane iteration |
| 2 | `SpatialArray3<T>` | **PARTIAL** | `cam_index.rs` CAM + `dn_tree.rs` spatial tree. Missing: f32 3D coordinates |
| 3 | `xor_diff()` | **DONE** | `bitwise.rs` AVX-512BW/AVX2/scalar XOR + popcount |
| 4 | `gather_scatter()` | **MISSING** | Only vpshufb nibble gathers in bitwise.rs |
| 5 | `columnar_view()` | **PARTIAL** | `arrow_bridge.rs` schema + `SoakingBuffer`. Missing: `ArrayView` bridge |
| 6 | `Zip::simd_apply()` | **PARTIAL** | `kernels.rs` K0→K1→K2 fusion. Missing: generic over closures |
| 7 | `runtime_dispatch()` | **DONE** | 65+ `is_x86_feature_detected!` sites + scalar fallbacks |
| 8 | `stencil()` | **MISSING** | BNN has neighbor patterns but no 3D stencil API |
| 9 | `compact_palette()` | **PARTIAL** | `palette_distance.rs` 256-entry codebook + `quantized.rs` + vpshufb |
| 10 | `prefetch/stream` | **PARTIAL** | `packed.rs` layout-for-prefetch. No explicit `_mm_prefetch` |

---

## Per-Item Detail

### 1. `simd_map()` — Lane-Native SIMD Iteration

**Exists:** `F32x16::from_slice()`, `copy_to_slice()`, `mul_add()`, `sqrt()`, all operators. AVX2 `dot_f32()` with 4-accumulator unrolling in `simd_avx2.rs:52-90`.

**Exists:** `src/hpc/vml.rs` has `vsexp`, `vssqrt`, `vsln`, `vsabs`, `vsadd`, `vsmul`, `vsdiv` — but ALL are scalar loops.

**Gap:** Need `vml.rs` to use `F32x16`/`f32x8` types. The types exist, the functions exist, they just aren't connected. Example:
```rust
// Current vml.rs:
pub fn vssqrt(x: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) { *o = v.sqrt(); }
}
// Should be:
pub fn vssqrt(x: &[f32], out: &mut [f32]) {
    let chunks = x.len() / 16;
    for i in 0..chunks {
        let v = F32x16::from_slice(&x[i*16..]);
        v.sqrt().copy_to_slice(&mut out[i*16..]);
    }
    // scalar remainder
}
```

### 2. `SpatialArray3<T>` — Content-Addressable Memory

**Exists:** `cam_index.rs` — multi-probe LSH CAM for 49,152-bit `GraphHV` binary vectors. `dn_tree.rs` — hierarchical spatial partitioning (739 lines). `parallel_search.rs` — dual-path HHTL + CLAM tree search.

**Gap:** All CAM infrastructure operates on binary hypervectors, not f32 spatial coordinates. Need a `SpatialCam3D` adapter that uses spatial hashing (floor(x/cell_size)) for the Pumpkin entity bind/unbind pattern.

### 3. `xor_diff()` — SIMD XOR Change Detection

**DONE.** `bitwise.rs`:
- `hamming_avx2()` (line 62): 32 bytes/iter via vpshufb
- `hamming_avx512bw()` (line 117): 64 bytes/iter via vpshufb-512
- `hamming_avx512_vpopcnt()`: native VPOPCNTDQ when available
- Runtime dispatch (line 234): `avx512vpopcntdq` → `avx512bw` → `avx2` → scalar
- `hamming_query_batch()`: batch mode for tick N vs N+1 comparison

**Only gap:** No sparse `nonzero_iter()` returning positions of changed elements.

### 4. `gather_scatter()` — Vectorized Gather

**MISSING.** `tekamolo.rs` and `cam_index.rs` use hash-based lookups (conceptually gather) but no `VPGATHERDD`/`VGATHERDPS` intrinsics anywhere.

### 5. `columnar_view()` — Zero-Copy Arrow Interop

**Exists:** `arrow_bridge.rs` has schema constants (`s_binary`, `p_binary`, `o_binary`, `node_id`), `GateState` lifecycle (Form→Flow→Freeze), `SoakingBuffer { data: Vec<i8>, n_entries, n_dims }`.

**Gap:** Missing the one-liner: `unsafe { ArrayView1::from_shape_ptr(len, arrow_buf.as_ptr()) }`.

### 6. `Zip::simd_apply()` — Multi-Array Fused SIMD Kernel

**Exists:** `kernels.rs` K0→K1→K2 fused cascade (1589 lines). `packed.rs` stroke-aligned cascade query. Both fuse multiple passes into one traversal.

**Gap:** Fusion is hardcoded for binary Hamming. Need generic version accepting `Fn(F32x16, F32x16) -> F32x16`.

### 7. `runtime_dispatch()` — CPU Feature Detection

**DONE.** Two complementary systems:
1. `bitwise.rs`: 65+ `is_x86_feature_detected!` with 4-tier fallback
2. `simd.rs` polyfill: compile-time dispatch via `#[cfg(target_arch)]` with scalar fallback types

### 8. `stencil()` — 3D Neighbor-Aware SIMD

**MISSING.** `bnn_causal_trajectory.rs`, `deepnsm.rs`, `clam_search.rs` have neighbor traversal patterns but nothing 3D-stencil-specific.

### 9. `compact_palette()` — Bit-Packed SIMD

**PARTIAL.** Three relevant modules:
- `palette_distance.rs`: 256-entry `Palette` codebook with precomputed pairwise L1 distance matrix
- `quantized.rs`: f32→u8 quantization with scale/zero-point
- `bitwise.rs`: vpshufb nibble lookup (4-bit table, proven in SIMD)

**Gap:** No variable-width (4-15 bit) pack/unpack for Minecraft block state encoding.

### 10. `prefetch_region()` + `stream_store()`

**PARTIAL.** `packed.rs` uses stroke-aligned layout for hardware prefetcher ("the prefetcher handles sequential access"). No explicit `_mm_prefetch` or `_mm_stream_ps`.

---

## What Changed Since Last Audit

| New Module | Lines | Wishlist Impact |
|---|---|---|
| `src/simd.rs` | 829 | #1 #6 #7 — portable SIMD types with scalar fallback |
| `src/simd_avx512.rs` | 1399 | #1 — F32x16/F64x8/U8x64 with FMA, sqrt, reduce |
| `src/simd_avx2.rs` | 618 | #1 — f32x8/f64x4 dot product, GEMM tile sizes |
| `hpc/holo.rs` | new | Phase + focus + carrier (94 tests) |
| `hpc/zeck.rs` | new | Zeckendorf encoding + batch/top_k |
| `hpc/palette_distance.rs` | new | #9 — 256-entry palette with O(1) distance |
| `hpc/parallel_search.rs` | new | #2 — dual-path HHTL + CLAM search |
| `hpc/layered_distance.rs` | new | O(1) distance via palette index + precomputed matrix |
| `hpc/bgz17_bridge.rs` | new | Base17 bridge for palette interop |

---

*Audit generated 2026-03-23. AdaWorldAPI/ndarray master @ 11633d06.*
