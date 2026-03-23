# Claude Code Session Prompt: ndarray SIMD Wishlist Implementation

Copy everything below the line into a new Claude Code session.

---

You are working on `AdaWorldAPI/ndarray` — a fork with 57 HPC modules (`src/hpc/`, 52K+ lines) and a 2,846-line portable SIMD polyfill (`src/simd.rs` + `src/simd_avx512.rs` + `src/simd_avx2.rs`). Develop on branch `claude/compare-simd-implementations-btTgj`. Push with `git push -u origin claude/compare-simd-implementations-btTgj`.

## What Already Exists (DO NOT REBUILD)

- **SIMD types:** `F32x16` (AVX-512), `f32x8`/`f64x4` (AVX2), scalar fallback on non-x86. Full ops: `+`, `-`, `*`, `/`, `mul_add` (FMA), `sqrt`, `reduce_sum/min/max`, `simd_clamp`, bitwise ops. File: `src/simd_avx512.rs`.
- **Portable polyfill:** `src/simd.rs` — `#[cfg(target_arch = "x86_64")]` re-exports real SIMD, `#[cfg(not(...))]` provides scalar `[f32; 16]` fallback with identical API. Drop-in `std::simd` replacement.
- **AVX2 dot:** `src/simd_avx2.rs:52` — 4-accumulator unrolled `dot_f32` using `f32x8`.
- **Runtime dispatch:** `src/hpc/bitwise.rs` — 65+ `is_x86_feature_detected!` calls. 4-tier: `avx512vpopcntdq` → `avx512bw` → `avx2` → scalar. Full Hamming/popcount/XOR kernels at each tier.
- **XOR diff:** `src/hpc/bitwise.rs` — `BitwiseOps` trait: `hamming_distance()`, `popcount()`, `hamming_distance_batch()`, `hamming_query_batch()`, `hamming_top_k()`. All SIMD-dispatched.
- **Fused cascade:** `src/hpc/kernels.rs` (1589 lines) — LIBXSMM-inspired K0 Probe → K1 Stats → K2 Exact pipeline with `SliceGate` integer thresholds.
- **Packed layout:** `src/hpc/packed.rs` — `PackedDatabase` with stroke-aligned memory for hardware prefetcher.
- **CAM index:** `src/hpc/cam_index.rs` — Multi-probe LSH for 49,152-bit `GraphHV` vectors.
- **Arrow bridge:** `src/hpc/arrow_bridge.rs` — Schema constants, `GateState`, `SoakingBuffer`.
- **Palette distance:** `src/hpc/palette_distance.rs` — 256-entry codebook with precomputed pairwise L1 matrix.
- **Quantization:** `src/hpc/quantized.rs` — f32↔u8/bf16 with scale/zero-point.
- **VML (scalar):** `src/hpc/vml.rs` — `vsexp`, `vssqrt`, `vsln`, `vsabs`, `vsadd`, `vsmul`, `vsdiv` — ALL scalar loops.

## 10 Implementation Tasks

### TIER 1: Wire existing SIMD types into existing scalar code

**Task 1: SIMD VML — connect `vml.rs` to `simd.rs` types**

`src/hpc/vml.rs` has 7 scalar functions. Rewrite each to use `F32x16`/`f32x8`:

```rust
use crate::simd::{f32x16, f32x8};

pub fn vssqrt(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let mut i = 0;

    // AVX-512: 16 elements per iteration
    while i + 16 <= n {
        let v = f32x16::from_slice(&x[i..]);
        v.sqrt().copy_to_slice(&mut out[i..]);
        i += 16;
    }
    // Scalar remainder
    while i < n {
        out[i] = x[i].sqrt();
        i += 1;
    }
}
```

Do the same for `vsexp`/`vdexp` (polynomial approx in SIMD lanes — Cephes or minimax), `vsln`/`vdln`, `vsabs` (bitwise AND with sign mask via `F32x16` bitcast), `vsadd`/`vsmul`/`vsdiv` (trivial: load, op, store).

For `exp` and `ln`: implement the polynomial approximation using `mul_add` chains:
```rust
// Fast exp(x) via range reduction + degree-5 polynomial
// x = n*ln(2) + r, exp(x) = 2^n * exp(r), exp(r) ≈ polynomial
fn vsexp_simd(x: &[f32], out: &mut [f32]) {
    let ln2_inv = f32x16::splat(1.4426950408889634);
    let ln2 = f32x16::splat(0.6931471805599453);
    // ... range reduce, polynomial via mul_add chain, reconstruct
}
```

**Task 2: `nonzero_iter()` for sparse XOR diff**

In `src/hpc/bitwise.rs`, add to `BitwiseOps`:
```rust
fn xor_diff_positions(&self, other: &Self) -> Vec<(usize, u8, u8)>;
```
SIMD: XOR 64 bytes (AVX-512), test zero with `_mm512_test_epi64_mask`, skip zero chunks, collect nonzero byte positions. Scalar fallback: XOR byte-by-byte, skip zeros.

**Task 3: Arrow zero-copy view bridge**

In `src/hpc/arrow_bridge.rs`, add:
```rust
use crate::prelude::*;

/// Zero-copy 1D view from raw pointer (Arrow Buffer → ndarray).
/// SAFETY: ptr must be valid, aligned, and live for lifetime 'a.
pub unsafe fn array_view_f32<'a>(ptr: *const f32, len: usize) -> ArrayView1<'a, f32> {
    ArrayView1::from_shape_ptr(len, ptr)
}

/// Zero-copy 2D columnar view: n_rows × n_cols.
pub unsafe fn columnar_view_f32<'a>(
    ptr: *const f32, n_rows: usize, n_cols: usize,
) -> ArrayView2<'a, f32> {
    ArrayView2::from_shape_ptr((n_rows, n_cols).f(), ptr)  // F-order for column-major
}
```
Add safe wrappers that take `&[f32]` for testing.

**Task 4: `simd_apply` generic batch processor**

Create `src/hpc/simd_apply.rs`:
```rust
use crate::simd::{f32x16, f32x8};

/// Apply f(lane) to every 16-element chunk of a slice, scalar remainder.
#[inline]
pub fn map_f32x16<F: Fn(f32x16) -> f32x16>(x: &[f32], out: &mut [f32], f: F) {
    let n = x.len().min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        let v = f32x16::from_slice(&x[i..]);
        f(v).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        // Scalar: load 1, broadcast to lane, apply, extract
        let v = f32x16::splat(x[i]);
        out[i] = f(v).to_array()[0];
        i += 1;
    }
}

/// Two-input version (Zip pattern).
#[inline]
pub fn zip_f32x16<F: Fn(f32x16, f32x16) -> f32x16>(
    a: &[f32], b: &[f32], out: &mut [f32], f: F
) {
    let n = a.len().min(b.len()).min(out.len());
    let mut i = 0;
    while i + 16 <= n {
        let va = f32x16::from_slice(&a[i..]);
        let vb = f32x16::from_slice(&b[i..]);
        f(va, vb).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        let va = f32x16::splat(a[i]);
        let vb = f32x16::splat(b[i]);
        out[i] = f(va, vb).to_array()[0];
        i += 1;
    }
}

/// In-place version.
#[inline]
pub fn map_inplace_f32x16<F: Fn(f32x16) -> f32x16>(data: &mut [f32], f: F) {
    let n = data.len();
    let mut i = 0;
    while i + 16 <= n {
        let v = f32x16::from_slice(&data[i..]);
        f(v).copy_to_slice(&mut data[i..]);
        i += 16;
    }
    while i < n {
        let v = f32x16::splat(data[i]);
        data[i] = f(v).to_array()[0];
        i += 1;
    }
}
```
Then rewrite `vml.rs` to use these (one-liner per function).

### TIER 2: New structures using existing infrastructure

**Task 5: `SpatialCam3D` — 3D spatial CAM**

Create `src/hpc/spatial_cam.rs`:
```rust
use std::collections::HashMap;

pub struct SpatialCam3D<T: Clone> {
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
    data: Vec<Option<T>>,
    positions: Vec<[f32; 3]>,  // SoA-friendly: could split into 3 Vec<f32>
    free_list: Vec<usize>,
    cell_size: f32,
    inv_cell_size: f32,
}

impl<T: Clone> SpatialCam3D<T> {
    pub fn new(cell_size: f32) -> Self;
    pub fn bind(&mut self, pos: [f32; 3], data: T) -> usize;
    pub fn unbind(&mut self, handle: usize);
    pub fn get(&self, handle: usize) -> Option<&T>;
    pub fn update_position(&mut self, handle: usize, new_pos: [f32; 3]);

    /// Query all entities in axis-aligned box.
    pub fn region(&self, min: [f32; 3], max: [f32; 3]) -> Vec<usize>;

    /// Extract contiguous position slices for SIMD batch processing.
    /// Returns (xs, ys, zs) for entities in the region.
    pub fn region_positions_soa(&self, min: [f32; 3], max: [f32; 3])
        -> (Vec<f32>, Vec<f32>, Vec<f32>);

    /// Batch distance check: which entities are within radius of point?
    /// Uses F32x16 for 16-entity-at-a-time distance computation.
    pub fn within_radius(&self, center: [f32; 3], radius: f32) -> Vec<usize>;
}
```
The `within_radius` method should use `F32x16` for distance computation:
```rust
// dx*dx + dy*dy + dz*dz < r*r, 16 entities at once
let dx = f32x16::from_slice(&xs[i..]) - cx;
let dy = f32x16::from_slice(&ys[i..]) - cy;
let dz = f32x16::from_slice(&zs[i..]) - cz;
let dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
// Compare with radius_sq, collect matching indices
```

**Task 6: `PaletteCodec` — variable-width bit packing**

Create `src/hpc/palette_codec.rs` (distinct from existing `palette_distance.rs`):
```rust
pub struct PaletteCodec {
    bits_per_entry: u8,   // 1-15
    entries_per_word: u8,  // 64 / bits_per_entry
    mask: u64,
}

impl PaletteCodec {
    pub fn new(bits_per_entry: u8) -> Self;

    /// Unpack N entries from packed u64 words into u16 output.
    pub fn unpack(&self, packed: &[u64], count: usize, out: &mut [u16]);

    /// Pack u16 values into u64 words.
    pub fn pack(&self, values: &[u16], out: &mut [u64]);

    /// Single-element get (scalar, for random access).
    pub fn get(&self, packed: &[u64], index: usize) -> u16;

    /// Single-element set.
    pub fn set(&mut self, packed: &mut [u64], index: usize, value: u16);
}
```
For 4-bit palettes (most common in Minecraft), SIMD unpack path: load 64 bytes (U8x64), mask low/high nibbles, expand to u16. Use existing vpshufb pattern from `bitwise.rs`.

**Task 7: AVX-512 gather wrapper**

Create `src/hpc/gather.rs`:
```rust
/// Gather f32 values from table at given u32 indices.
/// AVX-512: VPGATHERDD, AVX2: VGATHERDPS, scalar fallback.
pub fn gather_f32(table: &[f32], indices: &[u32], out: &mut [f32]) {
    let n = indices.len().min(out.len());
    let mut i = 0;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            while i + 16 <= n {
                unsafe { gather_f32_avx512(table, &indices[i..], &mut out[i..]) };
                i += 16;
            }
        } else if is_x86_feature_detected!("avx2") {
            while i + 8 <= n {
                unsafe { gather_f32_avx2(table, &indices[i..], &mut out[i..]) };
                i += 8;
            }
        }
    }
    // Scalar remainder
    while i < n {
        out[i] = table[indices[i] as usize];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn gather_f32_avx512(table: &[f32], indices: &[u32], out: &mut [f32]) {
    use core::arch::x86_64::*;
    let idx = _mm512_loadu_si512(indices.as_ptr() as *const _);
    let gathered = _mm512_i32gather_ps::<4>(idx, table.as_ptr() as *const u8);
    _mm512_storeu_ps(out.as_mut_ptr(), gathered);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn gather_f32_avx2(table: &[f32], indices: &[u32], out: &mut [f32]) {
    use core::arch::x86_64::*;
    let idx = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
    let gathered = _mm256_i32gather_ps::<4>(table.as_ptr(), idx);
    _mm256_storeu_ps(out.as_mut_ptr(), gathered);
}
```
This unblocks Perlin noise permutation table vectorization.

### TIER 3: Architecture

**Task 8: 3D Von Neumann stencil**

Create `src/hpc/stencil.rs`:
```rust
use crate::Array3;

/// Apply 3D Von Neumann stencil (6 face neighbors) to every interior cell.
/// Boundary cells use `boundary_val` for out-of-bounds neighbors.
pub fn stencil_vonneumann_3d<T, F>(
    input: &Array3<T>,
    output: &mut Array3<T>,
    boundary_val: T,
    f: F,
) where
    T: Copy,
    F: Fn(T, [T; 6]) -> T,  // center, [+x, -x, +y, -y, +z, -z]
{
    let (nx, ny, nz) = input.dim();
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let center = input[[x, y, z]];
                let neighbors = [
                    if x + 1 < nx { input[[x+1, y, z]] } else { boundary_val },
                    if x > 0 { input[[x-1, y, z]] } else { boundary_val },
                    if y + 1 < ny { input[[x, y+1, z]] } else { boundary_val },
                    if y > 0 { input[[x, y-1, z]] } else { boundary_val },
                    if z + 1 < nz { input[[x, y, z+1]] } else { boundary_val },
                    if z > 0 { input[[x, y, z-1]] } else { boundary_val },
                ];
                output[[x, y, z]] = f(center, neighbors);
            }
        }
    }
}

/// SIMD-optimized version for u8 (redstone signals): 64 cells per AVX-512 op.
/// Processes interior XY planes along Z axis using U8x64.
pub fn stencil_vonneumann_3d_u8_max_decay(
    input: &Array3<u8>,
    output: &mut Array3<u8>,
) {
    // For each cell: output = max(neighbors).saturating_sub(1)
    // This is redstone signal propagation.
    // SIMD: load Z-strips of 64 u8s, compute 6-neighbor max, subtract 1
    // ... implementation using U8x64 from crate::simd ...
}
```

**Task 9: SIMD FFT**

Rewrite `src/hpc/fft.rs` butterfly to use `F32x16`:
```rust
// Current: scalar butterfly per element
// Target: 16 butterflies per AVX-512 instruction
// Radix-2 Cooley-Tukey with SIMD twiddle factor multiplication
```
The butterfly is: `(u + t, u - t)` where `t = w * x`. With `F32x16`, process 16 frequency bins simultaneously.

**Task 10: Explicit prefetch for `SpatialCam3D`**

In `spatial_cam.rs`, add:
```rust
/// Hint the CPU to prefetch the next region's data into L2.
/// Call while processing current region for pipeline overlap.
pub fn prefetch_region(&self, min: [f32; 3], max: [f32; 3]) {
    let handles = self.region(min, max);
    if handles.is_empty() { return; }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::*;
        let base = self.positions.as_ptr().add(handles[0]);
        _mm_prefetch(base as *const i8, _MM_HINT_T1);  // L2
        if let Some(&last) = handles.last() {
            let end = self.positions.as_ptr().add(last);
            _mm_prefetch(end as *const i8, _MM_HINT_T1);
        }
    }
}
```

## Rules

1. **Use `crate::simd::` types** — `f32x16`, `f32x8`, `u8x64` etc. Never raw intrinsics in new code (except gather which has no polyfill wrapper yet).
2. **Follow `bitwise.rs` dispatch pattern** for any function needing runtime CPU detection.
3. **Scalar fallback always works** — the polyfill in `simd.rs` guarantees this on non-x86.
4. **Register every new module in `src/hpc/mod.rs`** with `#[allow(missing_docs)] pub mod name;`.
5. **Tests must pass on all platforms.** Use `#[cfg(target_arch = "x86_64")]` + feature detection to skip hardware-specific assertions.
6. **No new crate dependencies.** Use `core::arch::x86_64` for anything not in the polyfill.
7. **Run `cargo test --lib` and `cargo clippy` after each task.**
8. **Commit after each task.** Push to `claude/compare-simd-implementations-btTgj`.
9. **Read existing code before modifying.** VML, bitwise, kernels, packed — understand the patterns.
10. **Do not touch `src/simd.rs`, `src/simd_avx512.rs`, `src/simd_avx2.rs`** unless adding new type wrappers needed by your task.
