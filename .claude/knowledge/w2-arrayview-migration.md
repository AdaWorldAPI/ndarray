# KNOWLEDGE: W2 — HPC Kernel Slice→ArrayView In-Place Migration

## READ BY
- Worker agents executing W2-1 (reductions), W2-2a (vml), W2-2b (activations) in this repo
- Verifier agents for W2-3 (blas_level{1,2,3}) and W2-4 (statistics) — both already trait-impl on `ArrayBase`, so the audit is "confirm no slice fns remain"
- Downstream consumer sessions in `burn-ndarray`, `candle-ndarray`, `tract-onnx`, `ort`, `lance-graph`, and any AdaWorldAPI repo whose code calls `ndarray::hpc::{reductions,vml,activations}::*`
- The W3 codex-style audit agent reviewing the final diff for P0s

## P0 TRIGGERS
- About to write a NEW public kernel fn in `src/hpc/{reductions,vml,activations,blas_*,statistics,dot_*}.rs` → it MUST take `ArrayView<T,D>` / `ArrayViewMut<T,D>`, not `&[T]` / `&mut [T]`
- About to call a renamed kernel from downstream code and getting a "expected `ArrayView1<f32>`, found `&[f32]`" compile error → read §"Downstream consumer recipe"
- Reviewing a W2 worker PR → check every converted fn against the §"Bridge pattern" canonical example; the hot-path `as_slice_memory_order` arm AND the cold-path `Zip` arm both must be present

---

## Why this exists

The ergonomics of ndarray and the `hpc/` namespace **should not be slicing**. ArrayView carries strides, contiguity, and axis as type-level facts — and those facts ARE the SIMD vectorization plan: contiguous + lane-aligned → vectorize; non-contiguous → scalar fallback; reduce along axis → pick the contiguous axis, vectorize the inner. Flattening to `&[T]` deletes all three facts and forces the consumer to flatten before calling and reshape after, paying for a layout dance that the ArrayView surface would have absorbed for free.

The fork's HPC kernel layer was trimmed to slice signatures during a retrofit; this wave restores the inherited ArrayView shape. The SIMD primitive layer below (typed lanes `F32x16`/`I8x16`, packed-data closure-batch primitives in `src/simd_*.rs`, `hpc/quantized.rs`, `hpc/bf16_tile_gemm.rs`, `hpc/vnni_gemm.rs`, `hpc/palette_codec.rs`, `hpc/byte_scan.rs`, `hpc/bitwise.rs`, `hpc/heel_f64x8.rs`) STAYS slice-based — packed flat data genuinely has no shape, slices are the correct primitive there. This is the lance-graph polyfill territory contracted in PR #149.

**Two layers, two ergonomics:**

| Layer | Path | Ergonomic |
|---|---|---|
| HPC kernels (this wave's scope) | `src/hpc/{reductions,vml,activations,...}.rs` | `ArrayView<T,D>` / `ArrayViewMut<T,D>` / `Array<T,D>` |
| SIMD primitives (unchanged) | `src/simd_*.rs`, `hpc/{quantized,bf16_tile_gemm,vnni_gemm,palette_codec,byte_scan,bitwise,heel_f64x8}.rs` | typed lanes + slices over packed flat data |

The bridge from kernel to primitive lives INSIDE each kernel body: accept ArrayView, call `.as_slice_memory_order()`, hand the flat slice to the SIMD primitive. The slice never leaks above the kernel's body.

Cognitive modules (`plane`, `vsa`, `seal`, `merkle_tree`, `spo_bundle`, `nars`, `qualia`, `blackboard`, `holo`, `cyclic_bundle`, `causal_diff`, `organic`, `distance`, etc.) are OUT OF SCOPE — they work on their own cognitive types, not generic tensor data. Don't touch them.

---

## Bridge pattern (canonical — every converted fn follows this shape)

### Two-input elementwise (the `add_f32` archetype)

```rust
use ndarray::{ArrayView, ArrayViewMut, Dimension, Zip};

/// Element-wise addition: `out[i] = a[i] + b[i]`.
///
/// # Panics
/// Panics if `a`, `b`, `out` do not all have the same shape.
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::vml::vsadd};
/// let a = arr1(&[1.0_f32, 2.0, 3.0]);
/// let b = arr1(&[10.0_f32, 20.0, 30.0]);
/// let mut out = arr1(&[0.0_f32; 3]);
/// vsadd(a.view(), b.view(), out.view_mut());
/// assert_eq!(out.as_slice().unwrap(), &[11.0, 22.0, 33.0]);
/// ```
pub fn vsadd<D: Dimension>(
    a: ArrayView<f32, D>,
    b: ArrayView<f32, D>,
    mut out: ArrayViewMut<f32, D>,
) {
    assert_eq!(a.shape(), b.shape(), "vsadd: a/b shape mismatch");
    assert_eq!(a.shape(), out.shape(), "vsadd: a/out shape mismatch");

    // HOT PATH: all three contiguous + same memory order → flatten and dispatch
    // to the SIMD primitive layer. The primitive itself stays slice-based.
    if let (Some(a_s), Some(b_s), Some(out_s)) = (
        a.as_slice_memory_order(),
        b.as_slice_memory_order(),
        out.as_slice_memory_order_mut(),
    ) {
        crate::simd_ops::add_f32_inplace(a_s, b_s, out_s);
        return;
    }

    // COLD PATH: stride-aware Zip traversal.
    Zip::from(&mut out).and(a).and(b).for_each(|o, &x, &y| *o = x + y);
}
```

### Reduction (the `sum_f32` archetype — single input, scalar output)

```rust
/// Sum of an `f32` array via SIMD-tiered dispatch on the contiguous fast path,
/// stride-aware fold on the cold path.
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::reductions::sum_f32};
/// let x = arr1(&[1.0_f32, 2.0, 3.0, 4.0]);
/// assert_eq!(sum_f32(x.view()), 10.0);
/// ```
pub fn sum_f32<D: Dimension>(x: ArrayView<f32, D>) -> f32 {
    if let Some(s) = x.as_slice_memory_order() {
        return crate::simd::reduce_sum_f32(s); // SIMD primitive, slice-based
    }
    x.iter().copied().sum()
}
```

For optional-returning reductions (`mean`, `max`, `min`, `argmax`, `argmin` — `None` on empty):

```rust
pub fn max_f32<D: Dimension>(x: ArrayView<f32, D>) -> Option<f32> {
    if x.is_empty() { return None; }
    if let Some(s) = x.as_slice_memory_order() {
        return Some(crate::simd::reduce_max_f32(s));
    }
    x.iter().copied().reduce(f32::max)
}
```

`argmax`/`argmin` are special — they need ORIGINAL index, so they CANNOT use `as_slice_memory_order` directly (the flat index after reordering ≠ the logical index). Use stride-aware fold over `.indexed_iter()` or restrict to 1-D via `ArrayView1<f32>` and use the slice fast path with explicit index tracking. Document the choice in the function's `# Panics` / `# Returns` doc.

### Single-input in-place (the `vsexp` / `sigmoid_f32` archetype)

```rust
pub fn vsexp<D: Dimension>(
    x: ArrayView<f32, D>,
    mut out: ArrayViewMut<f32, D>,
) {
    assert_eq!(x.shape(), out.shape(), "vsexp: x/out shape mismatch");
    if let (Some(xs), Some(os)) = (
        x.as_slice_memory_order(),
        out.as_slice_memory_order_mut(),
    ) {
        crate::simd::vsexp_slice(xs, os);
        return;
    }
    Zip::from(&mut out).and(x).for_each(|o, &v| *o = v.exp());
}
```

### Axis-aware reduction (the `softmax` archetype — IMPORTANT)

Use `lanes(axis)` / `lanes_mut(axis)`, **NOT** `axis_iter` / `axis_iter_mut`. The Codex P2 on PR #150 caught this exact bug — `axis_iter_mut(Axis(1))` does NOT iterate the right dimension for softmax along axis 1. Iteration semantics:

- `array.lanes(Axis(k))` yields all 1-D lanes ALONG axis k (the axis the lane is parallel to). This is what softmax/argmax/sum-along-axis want.
- `array.axis_iter(Axis(k))` yields the (n-1)-D sub-arrays formed by SLICING perpendicular to axis k. Different operation.

```rust
pub fn softmax_axis_f32<D: Dimension>(
    x: ArrayView<f32, D>,
    mut out: ArrayViewMut<f32, D>,
    axis: Axis,
) -> Result<(), HpcError> {
    if axis.index() >= x.ndim() {
        return Err(HpcError::AxisOutOfBounds { axis: axis.index(), ndim: x.ndim() });
    }
    assert_eq!(x.shape(), out.shape());
    for (lane_in, mut lane_out) in x.lanes(axis).into_iter().zip(out.lanes_mut(axis)) {
        // each `lane_in` / `lane_out` is an ArrayView1 / ArrayViewMut1 along `axis`
        if let (Some(li), Some(lo)) = (lane_in.as_slice(), lane_out.as_slice_mut()) {
            crate::simd::softmax_inplace(li, lo); // SIMD primitive
        } else {
            softmax_scalar_lane(lane_in, lane_out); // local scalar helper
        }
    }
    Ok(())
}
```

For 1-D softmax (`softmax_f32` without axis arg), accept `ArrayView1<f32>` directly and skip the lane iteration — it's just the single-lane fast path.

---

## Generic-D vs fixed-D — when to pick which

Default: take `ArrayView<T, D>` generic over `D: Dimension`. This works for 1-D, 2-D, N-D, and is what most kernels want.

Use fixed dimension only when the operation is inherently 1-D or 2-D and the type system should enforce it:

- `dot_f32(x: ArrayView1<f32>, y: ArrayView1<f32>) -> f32` — dot product is 1-D by definition
- `argmax_f32(x: ArrayView1<f32>) -> Option<usize>` — argmax returns a flat index, makes sense only for 1-D (the N-D variant should be `argmax_axis_f32`)
- `softmax_f32(x: ArrayView1<f32>, out: ArrayViewMut1<f32>)` — 1-D softmax; N-D is `softmax_axis_f32`

Matrix-only ops stay `ArrayView2<T>` (matmul, gemv).

---

## Test conversion idioms

Old slice-based test:
```rust
#[test]
fn sum_of_threes() {
    let s = [1.0_f32, 2.0, 3.0];
    assert_eq!(sum_f32(&s), 6.0);
}
```

New ArrayView test (hot path, contiguous):
```rust
use ndarray::arr1;

#[test]
fn sum_of_threes_contig() {
    let s = arr1(&[1.0_f32, 2.0, 3.0]);
    assert_eq!(sum_f32(s.view()), 6.0);
}
```

ADD a non-contiguous case to exercise the cold path:
```rust
use ndarray::{arr1, s};

#[test]
fn sum_of_threes_strided() {
    let full = arr1(&[1.0_f32, 99.0, 2.0, 99.0, 3.0, 99.0]);
    let strided = full.slice(s![..;2]);   // [1.0, 2.0, 3.0] — non-contiguous
    assert_eq!(sum_f32(strided), 6.0);
}
```

For two-input ops, also add a SHAPE MISMATCH test that asserts panic:
```rust
#[test]
#[should_panic(expected = "shape mismatch")]
fn add_panics_on_shape_mismatch() {
    let a = arr1(&[1.0_f32, 2.0]);
    let b = arr1(&[1.0_f32, 2.0, 3.0]);
    let mut out = arr1(&[0.0_f32; 2]);
    vsadd(a.view(), b.view(), out.view_mut());
}
```

For N-D conversion verification, add a 2-D case to confirm `D: Dimension` actually works:
```rust
#[test]
fn sum_2d_array() {
    let m = ndarray::arr2(&[[1.0_f32, 2.0], [3.0_f32, 4.0]]);
    assert_eq!(sum_f32(m.view()), 10.0);
}
```

---

## Per-file conversion map

### W2-1 — `src/hpc/reductions.rs` (9 fns)

| Old | New |
|---|---|
| `pub fn sum_f32(s: &[f32]) -> f32` | `pub fn sum_f32<D: Dimension>(x: ArrayView<f32, D>) -> f32` |
| `pub fn sum_f64(s: &[f64]) -> f64` | `pub fn sum_f64<D: Dimension>(x: ArrayView<f64, D>) -> f64` |
| `pub fn mean_f32(s: &[f32]) -> Option<f32>` | `pub fn mean_f32<D: Dimension>(x: ArrayView<f32, D>) -> Option<f32>` |
| `pub fn mean_f64(s: &[f64]) -> Option<f64>` | `pub fn mean_f64<D: Dimension>(x: ArrayView<f64, D>) -> Option<f64>` |
| `pub fn max_f32(s: &[f32]) -> Option<f32>` | `pub fn max_f32<D: Dimension>(x: ArrayView<f32, D>) -> Option<f32>` |
| `pub fn min_f32(s: &[f32]) -> Option<f32>` | `pub fn min_f32<D: Dimension>(x: ArrayView<f32, D>) -> Option<f32>` |
| `pub fn argmax_f32(s: &[f32]) -> Option<usize>` | `pub fn argmax_f32(x: ArrayView1<f32>) -> Option<usize>` — **1-D only** (flat index semantics) |
| `pub fn argmin_f32(s: &[f32]) -> Option<usize>` | `pub fn argmin_f32(x: ArrayView1<f32>) -> Option<usize>` — **1-D only** |
| `pub fn nrm2_f32(s: &[f32]) -> f32` | `pub fn nrm2_f32<D: Dimension>(x: ArrayView<f32, D>) -> f32` |

### W2-2a — `src/hpc/vml.rs` (20 fns)

Single-input in-place pattern for all 14 unary fns (`vsexp`, `vdexp`, `vsln`, `vdln`, `vssqrt`, `vdsqrt`, `vsabs`, `vdabs`, `vssin`, `vscos`, `vstanh`, `vsfloor`, `vsceil`, `vsround`, `vsneg`, `vstrunc`) — convert to:
```rust
pub fn <name><D: Dimension>(x: ArrayView<T, D>, out: ArrayViewMut<T, D>)
```

Two-input in-place pattern for the 4 binary fns (`vsadd`, `vsmul`, `vsdiv`, `vspow`):
```rust
pub fn <name><D: Dimension>(a: ArrayView<f32, D>, b: ArrayView<f32, D>, out: ArrayViewMut<f32, D>)
```

### W2-2b — `src/hpc/activations.rs` (3 fns)

| Old | New |
|---|---|
| `pub fn sigmoid_f32(x: &[f32], out: &mut [f32])` | `pub fn sigmoid_f32<D: Dimension>(x: ArrayView<f32, D>, out: ArrayViewMut<f32, D>)` |
| `pub fn softmax_f32(x: &[f32], out: &mut [f32])` | `pub fn softmax_f32(x: ArrayView1<f32>, out: ArrayViewMut1<f32>)` — **1-D** |
| `pub fn log_softmax_f32(x: &[f32], out: &mut [f32])` | `pub fn log_softmax_f32(x: ArrayView1<f32>, out: ArrayViewMut1<f32>)` — **1-D** |

If `softmax_f32` had axis support inline (check the impl), split out a new `softmax_axis_f32` per the axis-aware pattern above; otherwise leave for a future PR.

### W2-3 — `src/hpc/blas_level{1,2,3}.rs` (VERIFY ONLY)

Already implemented as trait impls on `ArrayBase`:
- `blas_level1.rs:47` — `impl<A, S> BlasLevel1<A> for ArrayBase<S, Ix1>`
- `blas_level2.rs:97` — `impl<A, S> BlasLevel2<A> for ArrayBase<S, Ix2>`
- `blas_level3.rs:59` — `impl<A, S> BlasLevel3<A> for ArrayBase<S, Ix2>`

Verifier task: audit each file for any remaining `pub fn ... &[T]` free functions. The only known holdout is `blas_rotg` (Givens rotation, takes scalars — leave alone). Report any others found.

### W2-4 — `src/hpc/statistics.rs` (VERIFY ONLY)

Already implemented as trait impl on `ArrayBase`:
- `statistics.rs:65` — `impl<A, S, D> Statistics<A> for ArrayBase<S, D>`

Verifier task: confirm no slice-taking free fns exist. Report any found.

---

## Downstream consumer recipe (for burn-ndarray, candle, tract, ort, lance-graph)

After this wave merges, downstream code calling `ndarray::hpc::reductions::*`, `::vml::*`, `::activations::*` will fail to compile with "expected ArrayView, found `&[T]`" errors. Fix at the boundary.

### Pattern 1: you already have an `Array<T, D>` or `ArrayView<T, D>` — easiest

```rust
// OLD
let s = my_array.as_slice().unwrap();   // forced flatten + unwrap
let total = ndarray::hpc::reductions::sum_f32(s);

// NEW
let total = ndarray::hpc::reductions::sum_f32(my_array.view());
```

Net win: no `.as_slice().unwrap()` panic when the input is non-contiguous; the kernel's cold path handles strides natively.

### Pattern 2: you have a `&[f32]` slice (typical burn / candle tensor backend)

```rust
use ndarray::ArrayView1;

// OLD
let total = ndarray::hpc::reductions::sum_f32(my_slice);

// NEW — wrap the slice as a 1-D ArrayView (zero-copy borrow)
let total = ndarray::hpc::reductions::sum_f32(ArrayView1::from(my_slice));
```

`ArrayView1::from(&[T])` is zero-cost — it's just a fat pointer construction.

### Pattern 3: you have a `*mut f32` / `*const f32` + length (FFI boundary, candle write-back)

```rust
use ndarray::{ArrayView1, ArrayViewMut1};

// OLD — pass slice into ndarray, copy out, copy back
let in_slice = unsafe { std::slice::from_raw_parts(in_ptr, len) };
let out_vec = ndarray::hpc::vml::vsexp_returning_vec(in_slice);  // hypothetical old shape
unsafe { std::ptr::copy_nonoverlapping(out_vec.as_ptr(), out_ptr, len); }

// NEW — wrap pointers as views, write in place, zero copies
unsafe {
    let x = ArrayView1::from_shape_ptr(len, in_ptr);
    let out = ArrayViewMut1::from_shape_ptr(len, out_ptr);
    ndarray::hpc::vml::vsexp(x, out);
}
```

`from_shape_ptr` is `unsafe` — caller must ensure the pointer is valid for `len` elements and (for `ArrayViewMut1`) that no other reference aliases. Document the SAFETY contract at the FFI boundary.

### Pattern 4: you had a Vec-returning convenience (no longer exists)

The new API is write-back only. If the consumer wants an owned `Array`, allocate explicitly:

```rust
// OLD — ndarray allocated for you
let out = ndarray::hpc::vml::vsexp_old(x);   // returned Vec<f32>

// NEW — allocate the output, call write-back kernel
let mut out = ndarray::Array1::<f32>::zeros(x.len());
ndarray::hpc::vml::vsexp(ArrayView1::from(x), out.view_mut());
```

The explicit allocation makes the hot path visible and lets the consumer reuse a scratch buffer across calls.

### Pattern 5: axis-aware reductions

If you were doing manual axis iteration:

```rust
// OLD — loop in caller
let mut out = vec![0.0_f32; rows];
for (i, row) in matrix.axis_iter(Axis(0)).enumerate() {  // WRONG: see lanes vs axis_iter
    out[i] = ndarray::hpc::reductions::sum_f32(row.as_slice().unwrap());
}

// NEW — single call (when the kernel ships axis-aware variants)
let mut out = ndarray::Array1::<f32>::zeros(rows);
ndarray::hpc::reductions::sum_axis_f32(matrix.view(), Axis(1), out.view_mut())?;
```

Note: `axis_iter(Axis(0))` iterates ROWS of a 2-D matrix; `lanes(Axis(1))` iterates the same rows but yields them as `ArrayView1`s parallel to axis 1. For per-row sum, `lanes(Axis(1))` is the conceptually correct choice. The W2 wave doesn't ship `sum_axis_f32` (out of scope); for now consumers use `matrix.sum_axis(Axis(1))` (upstream ndarray method) until that primitive lands.

---

## What a worker commits per file

1. Convert all in-scope `pub fn`s in the file using the patterns above.
2. Update internal test cases: add the non-contiguous (cold path) variant per fn, plus a shape-mismatch panic test for two-input fns, plus a 2-D verification for generic-D fns.
3. If the fn is re-exported in `src/backend/mod.rs:260` (reductions only), the re-export keeps working without change — same name, different signature.
4. Run `cargo check --no-default-features --features std` AND `cargo test -p ndarray --lib --no-default-features --features std hpc::<module>::tests` for the affected file before committing.
5. Commit message format: `refactor(hpc/<file>): convert N pub fns to ArrayView-first (W2-X)`.

The W3 codex audit (post-sprint) will verify the bridge pattern is present (both arms) and that no kernel uses `axis_iter` where `lanes` is correct.
