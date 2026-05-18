//! SIMD-accelerated reduction dispatcher.
//!
//! Provides ArrayView-first reduction kernels that close the parity gap
//! with burn's hot reduction paths (softmax norms, argmax/argmin scans,
//! L2 distance). Each function picks the best 16-lane (`F32x16` /
//! `F32Mask16`) implementation at compile time on the contiguous fast
//! path; non-contiguous inputs fall back to a stride-aware iterator
//! traversal:
//!
//! - On `target_feature = "avx512f"` builds, F32x16 maps to native
//!   `__m512` and reductions use `_mm512_reduce_*` plus
//!   `_mm512_mask_blend_ps`.
//! - On AVX2-only builds, the polyfill in `simd_avx2.rs` backs the same
//!   API with `[F32x8; 2]` and the dispatcher becomes a 256-bit pair.
//! - On non-x86 builds, scalar fallback types in `simd.rs` preserve the
//!   API and produce identical numerical results modulo associativity.
//!
//! # ArrayView signatures
//!
//! Generic-D reductions (`sum_*`, `mean_*`, `max_*`, `min_*`, `nrm2_*`)
//! accept any `ArrayView<T, D>`. They use `as_slice_memory_order()` to
//! reach the SIMD primitive on contiguous inputs and fall back to
//! stride-aware `.iter()` on the cold path. `argmax_f32` / `argmin_f32`
//! are 1-D only (`ArrayView1<f32>`) because they return a flat index
//! whose semantics are only meaningful for 1-D inputs; use
//! `Axis::index_along` patterns for N-D argmax.
//!
//! # Empty convention
//!
//! Unbounded reductions (`max`, `min`, `argmax`, `argmin`, `mean`) return
//! [`Option`] — they have no defined value on an empty input. Sums and
//! norms have a well-defined zero element and return `0.0`.
//!
//! # Numerical notes
//!
//! - `sum_f32` / `nrm2_f32` accumulate in 16 lanes then horizontally
//!   reduce. Lane-summation is associative-different from a strict
//!   left fold; expect ~1.2e-4 relative error on a million-element
//!   reduction (vs ~1e-3 for naive `+=`).
//! - `nrm2_f32` uses a `mul_add` (FMA on AVX-512 / AVX2) — the squared
//!   accumulate has a single rounding instead of two.
//! - `argmax_f32` / `argmin_f32` follow the IEEE-754 convention: a
//!   strict greater-than comparison means a NaN never wins. The first
//!   index of the maximum (or minimum) value is returned; ties keep
//!   the lowest index, matching numpy's `np.argmax`.

use crate::simd::{F32x16, F64x8, U32x16};
use crate::{ArrayView, ArrayView1, Dimension};

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

// ===========================================================================
// Sum reductions
// ===========================================================================

/// Sum of an `f32` array. Returns `0.0` for an empty input.
///
/// Hot path (contiguous memory): SIMD-tiered dispatch via `F32x16` lanes
/// with a 4-way unroll. Cold path (strided): stride-aware iterator fold.
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::reductions::sum_f32};
/// let x = arr1(&[1.0_f32, 2.0, 3.0, 4.0]);
/// assert_eq!(sum_f32(x.view()), 10.0);
/// ```
#[inline]
pub fn sum_f32<D: Dimension>(x: ArrayView<f32, D>) -> f32 {
    if let Some(s) = x.as_slice_memory_order() {
        return sum_f32_slice(s);
    }
    x.iter().copied().sum()
}

/// SIMD-tiered slice sum for `f32`. Internal hot-path primitive.
#[inline]
fn sum_f32_slice(s: &[f32]) -> f32 {
    let chunks = s.len() / F32_LANES;
    let mut acc0 = F32x16::splat(0.0);
    let mut acc1 = F32x16::splat(0.0);
    let mut acc2 = F32x16::splat(0.0);
    let mut acc3 = F32x16::splat(0.0);

    let unrolled = chunks / 4;
    for i in 0..unrolled {
        let base = i * 4 * F32_LANES;
        acc0 += F32x16::from_slice(&s[base..]);
        acc1 += F32x16::from_slice(&s[base + F32_LANES..]);
        acc2 += F32x16::from_slice(&s[base + 2 * F32_LANES..]);
        acc3 += F32x16::from_slice(&s[base + 3 * F32_LANES..]);
    }
    for i in (unrolled * 4)..chunks {
        let base = i * F32_LANES;
        acc0 += F32x16::from_slice(&s[base..]);
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();
    for &v in &s[chunks * F32_LANES..] {
        sum += v;
    }
    sum
}

/// Sum of an `f64` array. Returns `0.0` for an empty input.
///
/// Hot path (contiguous memory): SIMD-tiered dispatch via `F64x8` lanes
/// with a 4-way unroll. Cold path (strided): stride-aware iterator fold.
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::reductions::sum_f64};
/// let x = arr1(&[1.0_f64, 2.0, 3.0, 4.0]);
/// assert_eq!(sum_f64(x.view()), 10.0);
/// ```
#[inline]
pub fn sum_f64<D: Dimension>(x: ArrayView<f64, D>) -> f64 {
    if let Some(s) = x.as_slice_memory_order() {
        return sum_f64_slice(s);
    }
    x.iter().copied().sum()
}

/// SIMD-tiered slice sum for `f64`. Internal hot-path primitive.
#[inline]
fn sum_f64_slice(s: &[f64]) -> f64 {
    let chunks = s.len() / F64_LANES;
    let mut acc0 = F64x8::splat(0.0);
    let mut acc1 = F64x8::splat(0.0);
    let mut acc2 = F64x8::splat(0.0);
    let mut acc3 = F64x8::splat(0.0);

    let unrolled = chunks / 4;
    for i in 0..unrolled {
        let base = i * 4 * F64_LANES;
        acc0 += F64x8::from_slice(&s[base..]);
        acc1 += F64x8::from_slice(&s[base + F64_LANES..]);
        acc2 += F64x8::from_slice(&s[base + 2 * F64_LANES..]);
        acc3 += F64x8::from_slice(&s[base + 3 * F64_LANES..]);
    }
    for i in (unrolled * 4)..chunks {
        let base = i * F64_LANES;
        acc0 += F64x8::from_slice(&s[base..]);
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();
    for &v in &s[chunks * F64_LANES..] {
        sum += v;
    }
    sum
}

/// Arithmetic mean of all elements. Returns `None` for an empty input.
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::reductions::mean_f32};
/// let x = arr1(&[1.0_f32, 2.0, 3.0, 4.0]);
/// assert_eq!(mean_f32(x.view()), Some(2.5));
/// ```
#[inline]
pub fn mean_f32<D: Dimension>(x: ArrayView<f32, D>) -> Option<f32> {
    if x.is_empty() {
        return None;
    }
    let n = x.len();
    Some(sum_f32(x) / n as f32)
}

/// Arithmetic mean of all elements as `f64`. Returns `None` for an empty input.
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::reductions::mean_f64};
/// let x = arr1(&[1.0_f64, 2.0, 3.0, 4.0]);
/// assert_eq!(mean_f64(x.view()), Some(2.5));
/// ```
#[inline]
pub fn mean_f64<D: Dimension>(x: ArrayView<f64, D>) -> Option<f64> {
    if x.is_empty() {
        return None;
    }
    let n = x.len();
    Some(sum_f64(x) / n as f64)
}

// ===========================================================================
// Min / Max
// ===========================================================================

/// Maximum element. Returns `None` if `x` is empty.
///
/// NaN inputs follow IEEE-754: any NaN propagates through `simd_max` so a
/// NaN in the SIMD lanes can become the lane-best. The horizontal
/// `reduce_max` then yields NaN. Caller should pre-filter NaNs if that is
/// undesirable.
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::reductions::max_f32};
/// let x = arr1(&[5.0_f32, 1.0, 9.0, -3.0]);
/// assert_eq!(max_f32(x.view()), Some(9.0));
/// ```
#[inline]
pub fn max_f32<D: Dimension>(x: ArrayView<f32, D>) -> Option<f32> {
    if x.is_empty() {
        return None;
    }
    if let Some(s) = x.as_slice_memory_order() {
        return Some(max_f32_slice(s));
    }
    x.iter().copied().reduce(f32::max)
}

/// SIMD-tiered slice max for `f32`. Internal hot-path primitive.
#[inline]
fn max_f32_slice(s: &[f32]) -> f32 {
    let chunks = s.len() / F32_LANES;
    let mut best = if chunks == 0 {
        // Pure scalar path (len < 16).
        F32x16::splat(s[0])
    } else {
        F32x16::from_slice(s)
    };

    for i in 1..chunks {
        let base = i * F32_LANES;
        let v = F32x16::from_slice(&s[base..]);
        best = best.simd_max(v);
    }

    let mut m = best.reduce_max();
    for &v in &s[chunks * F32_LANES..] {
        if v > m {
            m = v;
        }
    }
    m
}

/// Minimum element. Returns `None` if `x` is empty.
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::reductions::min_f32};
/// let x = arr1(&[5.0_f32, 1.0, 9.0, -3.0]);
/// assert_eq!(min_f32(x.view()), Some(-3.0));
/// ```
#[inline]
pub fn min_f32<D: Dimension>(x: ArrayView<f32, D>) -> Option<f32> {
    if x.is_empty() {
        return None;
    }
    if let Some(s) = x.as_slice_memory_order() {
        return Some(min_f32_slice(s));
    }
    x.iter().copied().reduce(f32::min)
}

/// SIMD-tiered slice min for `f32`. Internal hot-path primitive.
#[inline]
fn min_f32_slice(s: &[f32]) -> f32 {
    let chunks = s.len() / F32_LANES;
    let mut best = if chunks == 0 {
        F32x16::splat(s[0])
    } else {
        F32x16::from_slice(s)
    };

    for i in 1..chunks {
        let base = i * F32_LANES;
        let v = F32x16::from_slice(&s[base..]);
        best = best.simd_min(v);
    }

    let mut m = best.reduce_min();
    for &v in &s[chunks * F32_LANES..] {
        if v < m {
            m = v;
        }
    }
    m
}

// ===========================================================================
// Argmax / Argmin
// ===========================================================================

/// Generate the constant lane-index vector `[0, 1, 2, ..., 15]`.
#[inline(always)]
fn lane_index_seed() -> U32x16 {
    let arr: [u32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    U32x16::from_array(arr)
}

/// Index of the first occurrence of the maximum element, `None` if empty.
///
/// 1-D only — argmax over an N-D input returns a flat index whose
/// semantics depend on memory order; callers wanting an N-D argmax should
/// build it from per-axis reductions explicitly.
///
/// Hot path (contiguous): SIMD lane-tracked scan. Cold path (strided):
/// scalar `enumerate().reduce()` over `.iter()` so the returned index is
/// the logical index in the view, not the memory-order index.
///
/// Tie-break: lowest index wins (matches numpy `np.argmax`).
/// NaN handling: comparisons use strict greater-than, so NaN values never
/// displace a numeric maximum. If the input contains only NaNs the returned
/// index is the first index (0).
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::reductions::argmax_f32};
/// let x = arr1(&[5.0_f32, 1.0, 9.0, -3.0]);
/// assert_eq!(argmax_f32(x.view()), Some(2));
/// ```
#[inline]
pub fn argmax_f32(x: ArrayView1<f32>) -> Option<usize> {
    if x.is_empty() {
        return None;
    }
    if let Some(s) = x.as_slice() {
        return Some(argmax_f32_slice(s));
    }
    // Cold path: strided 1-D view. The logical index equals the iteration
    // index because `.iter()` walks in axis order for a 1-D view.
    let mut best_v = f32::NEG_INFINITY;
    let mut best_i = 0_usize;
    for (i, &v) in x.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    // If all values are NaN (or the only value is NaN), the strict-gt
    // check above never fires; return index 0 to match the SIMD path.
    Some(best_i)
}

/// SIMD-tiered slice argmax for `f32`. Internal hot-path primitive.
#[inline]
fn argmax_f32_slice(s: &[f32]) -> usize {
    let chunks = s.len() / F32_LANES;

    // Initialise lane-best to NEG_INFINITY (any finite element wins).
    let mut best_vals = F32x16::splat(f32::NEG_INFINITY);
    // Lane indices encoded as u32 — blended via `from_bits/to_bits` since
    // F32Mask16::select operates on f32 but is bit-exact for any 32-bit
    // pattern.
    let mut best_idx_bits = lane_index_seed();
    let lane_step = U32x16::splat(F32_LANES as u32);
    let mut current_indices = lane_index_seed();

    for i in 0..chunks {
        let base = i * F32_LANES;
        let v = F32x16::from_slice(&s[base..]);

        // Strict gt — NaN never updates, ties keep older (lower) index.
        let mask = v.simd_gt(best_vals);
        // Update values: where mask is set, take v; else keep best_vals.
        best_vals = mask.select(v, best_vals);
        // Update indices via f32 bit-blend (U32x16 has no native blend
        // helper but f32-mask blend is bit-exact for any 32-bit pattern).
        let new_idx_f = mask.select(F32x16::from_bits(current_indices), F32x16::from_bits(best_idx_bits));
        best_idx_bits = new_idx_f.to_bits();

        // Advance lane indices by 16 for the next chunk.
        current_indices = current_indices + lane_step;
    }

    // Horizontal reduce: highest value wins, ties → lowest index.
    let vals = best_vals.to_array();
    let idxs = best_idx_bits.to_array();
    let mut best_v;
    let mut best_i;
    if chunks == 0 {
        // No SIMD round — initialise from the first scalar.
        best_v = s[0];
        best_i = 0;
    } else {
        best_v = vals[0];
        best_i = idxs[0] as usize;
        for lane in 1..F32_LANES {
            let v = vals[lane];
            let i = idxs[lane] as usize;
            if v > best_v || (v == best_v && i < best_i) {
                best_v = v;
                best_i = i;
            }
        }
    }

    // Scalar tail.
    let tail_start = chunks * F32_LANES;
    for (offset, &v) in s[tail_start..].iter().enumerate() {
        let i = tail_start + offset;
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }

    best_i
}

/// Index of the first occurrence of the minimum element, `None` if empty.
///
/// 1-D only — see [`argmax_f32`] for the rationale.
///
/// Tie-break: lowest index wins.
/// NaN handling: comparisons use strict less-than, so NaN values never
/// displace a numeric minimum.
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::reductions::argmin_f32};
/// let x = arr1(&[5.0_f32, 1.0, 9.0, -3.0]);
/// assert_eq!(argmin_f32(x.view()), Some(3));
/// ```
#[inline]
pub fn argmin_f32(x: ArrayView1<f32>) -> Option<usize> {
    if x.is_empty() {
        return None;
    }
    if let Some(s) = x.as_slice() {
        return Some(argmin_f32_slice(s));
    }
    let mut best_v = f32::INFINITY;
    let mut best_i = 0_usize;
    for (i, &v) in x.iter().enumerate() {
        if v < best_v {
            best_v = v;
            best_i = i;
        }
    }
    Some(best_i)
}

/// SIMD-tiered slice argmin for `f32`. Internal hot-path primitive.
#[inline]
fn argmin_f32_slice(s: &[f32]) -> usize {
    let chunks = s.len() / F32_LANES;

    let mut best_vals = F32x16::splat(f32::INFINITY);
    let mut best_idx_bits = lane_index_seed();
    let lane_step = U32x16::splat(F32_LANES as u32);
    let mut current_indices = lane_index_seed();

    for i in 0..chunks {
        let base = i * F32_LANES;
        let v = F32x16::from_slice(&s[base..]);

        let mask = v.simd_lt(best_vals);
        best_vals = mask.select(v, best_vals);
        let new_idx_f = mask.select(F32x16::from_bits(current_indices), F32x16::from_bits(best_idx_bits));
        best_idx_bits = new_idx_f.to_bits();
        current_indices = current_indices + lane_step;
    }

    let vals = best_vals.to_array();
    let idxs = best_idx_bits.to_array();
    let mut best_v;
    let mut best_i;
    if chunks == 0 {
        best_v = s[0];
        best_i = 0;
    } else {
        best_v = vals[0];
        best_i = idxs[0] as usize;
        for lane in 1..F32_LANES {
            let v = vals[lane];
            let i = idxs[lane] as usize;
            if v < best_v || (v == best_v && i < best_i) {
                best_v = v;
                best_i = i;
            }
        }
    }

    let tail_start = chunks * F32_LANES;
    for (offset, &v) in s[tail_start..].iter().enumerate() {
        let i = tail_start + offset;
        if v < best_v {
            best_v = v;
            best_i = i;
        }
    }

    best_i
}

// ===========================================================================
// L2 norm (BLAS L1 nrm2)
// ===========================================================================

/// Euclidean (L2) norm: `sqrt(sum(x^2))`. Returns `0.0` for an empty input.
///
/// Uses fused multiply-add for the squared accumulate where supported on
/// the contiguous fast path; the cold-path stride fallback uses scalar
/// `x*x` accumulation.
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::reductions::nrm2_f32};
/// let x = arr1(&[3.0_f32, 4.0]);
/// assert!((nrm2_f32(x.view()) - 5.0).abs() < 1e-6);
/// ```
#[inline]
pub fn nrm2_f32<D: Dimension>(x: ArrayView<f32, D>) -> f32 {
    if let Some(s) = x.as_slice_memory_order() {
        return nrm2_f32_slice(s);
    }
    let sumsq: f32 = x.iter().map(|&v| v * v).sum();
    sumsq.sqrt()
}

/// SIMD-tiered slice nrm2 for `f32`. Internal hot-path primitive.
#[inline]
fn nrm2_f32_slice(s: &[f32]) -> f32 {
    let chunks = s.len() / F32_LANES;
    let mut acc0 = F32x16::splat(0.0);
    let mut acc1 = F32x16::splat(0.0);
    let mut acc2 = F32x16::splat(0.0);
    let mut acc3 = F32x16::splat(0.0);

    let unrolled = chunks / 4;
    for i in 0..unrolled {
        let base = i * 4 * F32_LANES;
        let v0 = F32x16::from_slice(&s[base..]);
        let v1 = F32x16::from_slice(&s[base + F32_LANES..]);
        let v2 = F32x16::from_slice(&s[base + 2 * F32_LANES..]);
        let v3 = F32x16::from_slice(&s[base + 3 * F32_LANES..]);
        // FMA path: acc + v*v in one rounding step.
        acc0 = v0.mul_add(v0, acc0);
        acc1 = v1.mul_add(v1, acc1);
        acc2 = v2.mul_add(v2, acc2);
        acc3 = v3.mul_add(v3, acc3);
    }
    for i in (unrolled * 4)..chunks {
        let base = i * F32_LANES;
        let v = F32x16::from_slice(&s[base..]);
        acc0 = v.mul_add(v, acc0);
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();
    for &v in &s[chunks * F32_LANES..] {
        sum += v * v;
    }
    sum.sqrt()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arr1, arr2, s, Array1};

    // ---- sum_f32 ----------------------------------------------------------

    #[test]
    fn sum_f32_empty_is_zero() {
        let v: Array1<f32> = Array1::zeros(0);
        assert_eq!(sum_f32(v.view()), 0.0);
    }

    #[test]
    fn sum_f32_small_scalar_only_contig() {
        let v = arr1(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]);
        assert!((sum_f32(v.view()) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn sum_f32_thousand_ones_contig() {
        let v: Array1<f32> = Array1::from_elem(1000, 1.0);
        assert!((sum_f32(v.view()) - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn sum_f32_misaligned_tails_contig() {
        // Lengths 17, 33, 65 — exercise SIMD body + scalar tail boundary.
        for &n in &[17_usize, 33, 65, 127, 1000] {
            let v: Array1<f32> = (0..n).map(|i| i as f32).collect();
            let expected: f32 = (0..n).map(|i| i as f32).sum();
            let got = sum_f32(v.view());
            assert!(
                (got - expected).abs() < (expected.abs() * 1e-5 + 1e-3),
                "n={}: got {}, expected {}",
                n,
                got,
                expected
            );
        }
    }

    #[test]
    fn sum_f32_strided_cold_path() {
        // Slice with step 2 → non-contiguous, cold-path fold.
        let full = arr1(&[1.0_f32, 99.0, 2.0, 99.0, 3.0, 99.0]);
        let strided = full.slice(s![..;2]); // [1.0, 2.0, 3.0]
        assert_eq!(sum_f32(strided), 6.0);
    }

    #[test]
    fn sum_f32_2d_generic_d() {
        let m = arr2(&[[1.0_f32, 2.0], [3.0_f32, 4.0]]);
        assert_eq!(sum_f32(m.view()), 10.0);
    }

    #[test]
    fn sum_f64_basic_contig() {
        let v: Array1<f64> = Array1::from_elem(100, 0.5);
        assert!((sum_f64(v.view()) - 50.0).abs() < 1e-12);
    }

    #[test]
    fn sum_f64_strided_cold_path() {
        let full = arr1(&[1.0_f64, 99.0, 2.0, 99.0, 3.0]);
        let strided = full.slice(s![..;2]);
        assert_eq!(sum_f64(strided), 6.0);
    }

    #[test]
    fn sum_f64_2d_generic_d() {
        let m = arr2(&[[1.0_f64, 2.0], [3.0_f64, 4.0]]);
        assert_eq!(sum_f64(m.view()), 10.0);
    }

    // ---- mean_f32 / mean_f64 ---------------------------------------------

    #[test]
    fn mean_f32_basic_contig() {
        let v = arr1(&[1.0_f32, 2.0, 3.0, 4.0]);
        let m = mean_f32(v.view()).expect("non-empty");
        assert!((m - 2.5).abs() < 1e-6);
    }

    #[test]
    fn mean_f32_empty_is_none() {
        let v: Array1<f32> = Array1::zeros(0);
        assert_eq!(mean_f32(v.view()), None);
    }

    #[test]
    fn mean_f32_strided_cold_path() {
        let full = arr1(&[1.0_f32, 99.0, 2.0, 99.0, 3.0]);
        let strided = full.slice(s![..;2]); // [1.0, 2.0, 3.0]
        assert_eq!(mean_f32(strided), Some(2.0));
    }

    #[test]
    fn mean_f32_2d_generic_d() {
        let m = arr2(&[[1.0_f32, 2.0], [3.0_f32, 4.0]]);
        assert_eq!(mean_f32(m.view()), Some(2.5));
    }

    #[test]
    fn mean_f64_empty_is_none() {
        let v: Array1<f64> = Array1::zeros(0);
        assert_eq!(mean_f64(v.view()), None);
    }

    #[test]
    fn mean_f64_strided_cold_path() {
        let full = arr1(&[1.0_f64, 99.0, 2.0, 99.0, 3.0]);
        let strided = full.slice(s![..;2]);
        assert_eq!(mean_f64(strided), Some(2.0));
    }

    #[test]
    fn mean_f64_2d_generic_d() {
        let m = arr2(&[[2.0_f64, 4.0], [6.0_f64, 8.0]]);
        assert_eq!(mean_f64(m.view()), Some(5.0));
    }

    // ---- max_f32 / min_f32 ------------------------------------------------

    #[test]
    fn max_f32_empty() {
        let v: Array1<f32> = Array1::zeros(0);
        assert_eq!(max_f32(v.view()), None);
    }

    #[test]
    fn max_f32_basic_contig() {
        let v = arr1(&[5.0_f32, 1.0, 9.0, -3.0]);
        assert_eq!(max_f32(v.view()), Some(9.0));
    }

    #[test]
    fn max_f32_long_contig() {
        let mut v: Array1<f32> = (0..1000).map(|i| (i as f32) * 0.5).collect();
        v[765] = 5000.0;
        assert_eq!(max_f32(v.view()), Some(5000.0));
    }

    #[test]
    fn max_f32_strided_cold_path() {
        let full = arr1(&[5.0_f32, -100.0, 1.0, -100.0, 9.0, -100.0, -3.0]);
        let strided = full.slice(s![..;2]); // [5.0, 1.0, 9.0, -3.0]
        assert_eq!(max_f32(strided), Some(9.0));
    }

    #[test]
    fn max_f32_2d_generic_d() {
        let m = arr2(&[[5.0_f32, 1.0], [9.0_f32, -3.0]]);
        assert_eq!(max_f32(m.view()), Some(9.0));
    }

    #[test]
    fn min_f32_basic_contig() {
        let v = arr1(&[5.0_f32, 1.0, 9.0, -3.0]);
        assert_eq!(min_f32(v.view()), Some(-3.0));
    }

    #[test]
    fn min_f32_empty() {
        let v: Array1<f32> = Array1::zeros(0);
        assert_eq!(min_f32(v.view()), None);
    }

    #[test]
    fn min_f32_strided_cold_path() {
        let full = arr1(&[5.0_f32, 100.0, 1.0, 100.0, 9.0, 100.0, -3.0]);
        let strided = full.slice(s![..;2]); // [5.0, 1.0, 9.0, -3.0]
        assert_eq!(min_f32(strided), Some(-3.0));
    }

    #[test]
    fn min_f32_2d_generic_d() {
        let m = arr2(&[[5.0_f32, 1.0], [9.0_f32, -3.0]]);
        assert_eq!(min_f32(m.view()), Some(-3.0));
    }

    #[test]
    fn max_min_misaligned_contig() {
        for &n in &[1_usize, 7, 16, 17, 31, 33, 64, 65, 127, 256, 1023] {
            let v: Array1<f32> = (0..n)
                .map(|i| ((i as i32) - (n as i32) / 2) as f32)
                .collect();
            let expected_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let expected_min = v.iter().copied().fold(f32::INFINITY, f32::min);
            assert_eq!(max_f32(v.view()), Some(expected_max), "max_f32 n={}", n);
            assert_eq!(min_f32(v.view()), Some(expected_min), "min_f32 n={}", n);
        }
    }

    // ---- argmax_f32 / argmin_f32 -----------------------------------------

    #[test]
    fn argmax_f32_empty() {
        let v: Array1<f32> = Array1::zeros(0);
        assert_eq!(argmax_f32(v.view()), None);
    }

    #[test]
    fn argmax_f32_basic_contig() {
        let v = arr1(&[5.0_f32, 1.0, 9.0, -3.0]);
        assert_eq!(argmax_f32(v.view()), Some(2));
    }

    #[test]
    fn argmax_f32_strided_cold_path() {
        // After step-2 slicing the logical view is [5.0, 1.0, 9.0, -3.0]
        // and the argmax should be the LOGICAL index 2 (the 9.0), NOT the
        // underlying memory-order index 4.
        let full = arr1(&[5.0_f32, 99.0, 1.0, 99.0, 9.0, 99.0, -3.0]);
        let strided = full.slice(s![..;2]);
        assert_eq!(argmax_f32(strided), Some(2));
    }

    #[test]
    fn argmin_f32_basic_contig() {
        let v = arr1(&[5.0_f32, 1.0, 9.0, -3.0]);
        assert_eq!(argmin_f32(v.view()), Some(3));
    }

    #[test]
    fn argmin_f32_strided_cold_path() {
        let full = arr1(&[5.0_f32, -99.0, 1.0, -99.0, 9.0, -99.0, -3.0]);
        let strided = full.slice(s![..;2]); // [5.0, 1.0, 9.0, -3.0]
        assert_eq!(argmin_f32(strided), Some(3));
    }

    #[test]
    fn argmin_f32_empty() {
        let v: Array1<f32> = Array1::zeros(0);
        assert_eq!(argmin_f32(v.view()), None);
    }

    #[test]
    fn argmax_f32_misaligned_tail_contig() {
        // Place the maximum at a position straddling the SIMD/tail boundary.
        for &(n, peak) in
            &[(17_usize, 16), (17, 0), (17, 8), (33, 32), (33, 17), (65, 64), (65, 32), (127, 100), (1000, 999)]
        {
            let mut v: Array1<f32> = Array1::zeros(n);
            v[peak] = 1.0;
            assert_eq!(argmax_f32(v.view()), Some(peak), "n={}, peak={}", n, peak);
        }
    }

    #[test]
    fn argmax_f32_tie_takes_first_contig() {
        let v = arr1(&[1.0_f32, 5.0, 2.0, 5.0, 3.0]);
        assert_eq!(argmax_f32(v.view()), Some(1));
    }

    #[test]
    fn argmax_f32_tie_across_chunks_contig() {
        let mut v: Array1<f32> = Array1::zeros(50);
        v[3] = 7.0;
        v[20] = 7.0;
        assert_eq!(argmax_f32(v.view()), Some(3));
    }

    #[test]
    fn argmax_f32_with_nan_skips_nan_contig() {
        let v = arr1(&[1.0_f32, f32::NAN, 5.0, f32::NAN, 2.0]);
        assert_eq!(argmax_f32(v.view()), Some(2));
    }

    #[test]
    fn argmin_f32_with_nan_skips_nan_contig() {
        let v = arr1(&[10.0_f32, f32::NAN, 1.0, f32::NAN, 5.0]);
        assert_eq!(argmin_f32(v.view()), Some(2));
    }

    #[test]
    fn argmax_f32_negative_values_contig() {
        let v = arr1(&[-5.0_f32, -1.0, -9.0, -3.0]);
        assert_eq!(argmax_f32(v.view()), Some(1));
    }

    #[test]
    fn argmax_f32_long_random_contig() {
        let n = 2049;
        let v: Array1<f32> = (0..n)
            .map(|i| {
                let x = (i as i64).wrapping_mul(0x9E37_79B9_7F4A_7C15_u64 as i64);
                f32::from_bits((x as u32) & 0x7FFF_FFFF) // non-NaN positive
            })
            .collect();
        let mut best_i = 0;
        let mut best_v = v[0];
        for (i, &x) in v.iter().enumerate() {
            if x > best_v {
                best_v = x;
                best_i = i;
            }
        }
        assert_eq!(argmax_f32(v.view()), Some(best_i));
    }

    // ---- nrm2_f32 ---------------------------------------------------------

    #[test]
    fn nrm2_f32_empty_is_zero() {
        let v: Array1<f32> = Array1::zeros(0);
        assert_eq!(nrm2_f32(v.view()), 0.0);
    }

    #[test]
    fn nrm2_f32_3_4_is_5_contig() {
        let v = arr1(&[3.0_f32, 4.0]);
        assert!((nrm2_f32(v.view()) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn nrm2_f32_unit_vector_contig() {
        let v: Array1<f32> = Array1::from_elem(1000, 1.0_f32 / (1000.0_f32).sqrt());
        assert!((nrm2_f32(v.view()) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn nrm2_f32_misaligned_tails_contig() {
        for &n in &[1_usize, 16, 17, 33, 65, 127, 1000] {
            let v: Array1<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
            let expected: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let got = nrm2_f32(v.view());
            assert!(
                (got - expected).abs() < (expected.abs() * 1e-4 + 1e-4),
                "n={}: got {}, expected {}",
                n,
                got,
                expected
            );
        }
    }

    #[test]
    fn nrm2_f32_strided_cold_path() {
        let full = arr1(&[3.0_f32, 99.0, 4.0]);
        let strided = full.slice(s![..;2]); // [3.0, 4.0]
        assert!((nrm2_f32(strided) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn nrm2_f32_2d_generic_d() {
        let m = arr2(&[[3.0_f32, 0.0], [0.0_f32, 4.0]]);
        assert!((nrm2_f32(m.view()) - 5.0).abs() < 1e-6);
    }
}
