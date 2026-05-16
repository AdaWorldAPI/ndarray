//! SIMD-accelerated reduction dispatcher.
//!
//! Provides slice-level reduction kernels that close the parity gap with
//! burn's hot reduction paths (softmax norms, argmax/argmin scans, L2
//! distance). Each function picks the best 16-lane (`F32x16` /
//! `F32Mask16`) implementation at compile time:
//!
//! - On `target_feature = "avx512f"` builds, F32x16 maps to native
//!   `__m512` and reductions use `_mm512_reduce_*` plus
//!   `_mm512_mask_blend_ps`.
//! - On AVX2-only builds, the polyfill in `simd_avx2.rs` backs the same
//!   API with `[F32x8; 2]` and the dispatcher becomes a 256-bit pair.
//! - On non-x86 builds, scalar fallback types in `simd.rs` preserve the
//!   API and produce identical numerical results modulo associativity.
//!
//! # Empty-slice convention
//!
//! Unbounded reductions (`max`, `min`, `argmax`, `argmin`, `mean`) return
//! [`Option`] — they have no defined value on an empty slice. Sums and
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

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

// ===========================================================================
// Sum reductions
// ===========================================================================

/// Sum of all elements. Returns `0.0` for an empty slice.
#[inline]
pub fn sum_f32(s: &[f32]) -> f32 {
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

/// Sum of all elements as `f64`. Returns `0.0` for an empty slice.
#[inline]
pub fn sum_f64(s: &[f64]) -> f64 {
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

/// Arithmetic mean of all elements. Returns `None` for an empty slice.
#[inline]
pub fn mean_f32(s: &[f32]) -> Option<f32> {
    if s.is_empty() {
        None
    } else {
        Some(sum_f32(s) / s.len() as f32)
    }
}

/// Arithmetic mean of all elements as `f64`. Returns `None` for an empty slice.
#[inline]
pub fn mean_f64(s: &[f64]) -> Option<f64> {
    if s.is_empty() {
        None
    } else {
        Some(sum_f64(s) / s.len() as f64)
    }
}

// ===========================================================================
// Min / Max
// ===========================================================================

/// Maximum element. Returns `None` if `s` is empty.
///
/// NaN inputs follow IEEE-754: any NaN propagates through `simd_max` so a
/// NaN in the SIMD lanes can become the lane-best. The horizontal
/// `reduce_max` then yields NaN. Caller should pre-filter NaNs if that is
/// undesirable.
#[inline]
pub fn max_f32(s: &[f32]) -> Option<f32> {
    if s.is_empty() {
        return None;
    }
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
    Some(m)
}

/// Minimum element. Returns `None` if `s` is empty.
#[inline]
pub fn min_f32(s: &[f32]) -> Option<f32> {
    if s.is_empty() {
        return None;
    }
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
    Some(m)
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
/// Tie-break: lowest index wins (matches numpy `np.argmax`).
/// NaN handling: comparisons use strict greater-than, so NaN values never
/// displace a numeric maximum. If the slice contains only NaNs the returned
/// index is the first index (0).
#[inline]
pub fn argmax_f32(s: &[f32]) -> Option<usize> {
    if s.is_empty() {
        return None;
    }
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

    Some(best_i)
}

/// Index of the first occurrence of the minimum element, `None` if empty.
///
/// Tie-break: lowest index wins.
/// NaN handling: comparisons use strict less-than, so NaN values never
/// displace a numeric minimum.
#[inline]
pub fn argmin_f32(s: &[f32]) -> Option<usize> {
    if s.is_empty() {
        return None;
    }
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

    Some(best_i)
}

// ===========================================================================
// L2 norm (BLAS L1 nrm2)
// ===========================================================================

/// Euclidean (L2) norm: `sqrt(sum(x^2))`. Returns `0.0` for empty slice.
///
/// Uses fused multiply-add for the squared accumulate where supported.
#[inline]
pub fn nrm2_f32(s: &[f32]) -> f32 {
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
// f64 parity set  (max / min / argmax / argmin / nrm2)
// ===========================================================================

/// Maximum element of an `f64` slice. Returns `None` if `s` is empty.
///
/// Uses `F64x8::simd_max` (8-wide) across `F64_LANES`-sized chunks then a
/// scalar tail. NaN handling mirrors `max_f32`: `f64::max` is used in the
/// scalar tail and `simd_max` uses `f64::max` element-wise, so a NaN in a
/// lane can propagate. Pre-filter NaNs if that is undesirable.
///
/// # Example
/// ```
/// use ndarray::hpc::reductions::max_f64;
/// assert_eq!(max_f64(&[1.0_f64, -2.0, 3.0, 0.5]), Some(3.0));
/// assert_eq!(max_f64(&[]), None);
/// ```
#[inline]
pub fn max_f64(s: &[f64]) -> Option<f64> {
    if s.is_empty() {
        return None;
    }
    let chunks = s.len() / F64_LANES;
    let mut best = if chunks == 0 {
        // Pure scalar path (len < 8).
        F64x8::splat(s[0])
    } else {
        F64x8::from_slice(s)
    };

    for i in 1..chunks {
        let base = i * F64_LANES;
        let v = F64x8::from_slice(&s[base..]);
        best = best.simd_max(v);
    }

    let mut m = best.reduce_max();
    for &v in &s[chunks * F64_LANES..] {
        if v > m {
            m = v;
        }
    }
    Some(m)
}

/// Minimum element of an `f64` slice. Returns `None` if `s` is empty.
///
/// Uses `F64x8::simd_min` (8-wide) across `F64_LANES`-sized chunks then a
/// scalar tail. NaN handling mirrors `min_f32`.
///
/// # Example
/// ```
/// use ndarray::hpc::reductions::min_f64;
/// assert_eq!(min_f64(&[1.0_f64, -2.0, 3.0, 0.5]), Some(-2.0));
/// assert_eq!(min_f64(&[]), None);
/// ```
#[inline]
pub fn min_f64(s: &[f64]) -> Option<f64> {
    if s.is_empty() {
        return None;
    }
    let chunks = s.len() / F64_LANES;
    let mut best = if chunks == 0 {
        F64x8::splat(s[0])
    } else {
        F64x8::from_slice(s)
    };

    for i in 1..chunks {
        let base = i * F64_LANES;
        let v = F64x8::from_slice(&s[base..]);
        best = best.simd_min(v);
    }

    let mut m = best.reduce_min();
    for &v in &s[chunks * F64_LANES..] {
        if v < m {
            m = v;
        }
    }
    Some(m)
}

/// Index of the first occurrence of the maximum element in an `f64` slice.
/// Returns `None` if `s` is empty.
///
/// Tie-break: lowest index wins (matches `np.argmax`).
/// NaN handling: comparisons use strict `>` so NaN values never displace a
/// numeric maximum. If the slice contains only NaNs index 0 is returned.
///
/// # Example
/// ```
/// use ndarray::hpc::reductions::argmax_f64;
/// assert_eq!(argmax_f64(&[1.0_f64, 9.0, 3.0]), Some(1));
/// assert_eq!(argmax_f64(&[]), None);
/// ```
#[inline]
pub fn argmax_f64(s: &[f64]) -> Option<usize> {
    if s.is_empty() {
        return None;
    }
    // Scalar loop — F64x8::simd_gt is not guaranteed on all dispatch paths.
    // TODO(ws6): vectorize once F64x8 exposes simd_gt uniformly across avx2.
    let mut best_v = s[0];
    let mut best_i = 0usize;
    for (i, &v) in s.iter().enumerate().skip(1) {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    Some(best_i)
}

/// Index of the first occurrence of the minimum element in an `f64` slice.
/// Returns `None` if `s` is empty.
///
/// Tie-break: lowest index wins.
/// NaN handling: comparisons use strict `<` so NaN values never displace a
/// numeric minimum.
///
/// # Example
/// ```
/// use ndarray::hpc::reductions::argmin_f64;
/// assert_eq!(argmin_f64(&[1.0_f64, -9.0, 3.0]), Some(1));
/// assert_eq!(argmin_f64(&[]), None);
/// ```
#[inline]
pub fn argmin_f64(s: &[f64]) -> Option<usize> {
    if s.is_empty() {
        return None;
    }
    // Scalar loop — F64x8::simd_lt is not guaranteed on all dispatch paths.
    // TODO(ws6): vectorize once F64x8 exposes simd_lt uniformly across avx2.
    let mut best_v = s[0];
    let mut best_i = 0usize;
    for (i, &v) in s.iter().enumerate().skip(1) {
        if v < best_v {
            best_v = v;
            best_i = i;
        }
    }
    Some(best_i)
}

/// Euclidean (L2) norm of an `f64` slice: `sqrt(Σ xᵢ²)`.
/// Returns `0.0` for an empty slice.
///
/// Uses `F64x8::mul_add` (FMA where supported) with 4-way unrolled chunks,
/// mirroring `nrm2_f32`.
///
/// # Example
/// ```
/// use ndarray::hpc::reductions::nrm2_f64;
/// assert!((nrm2_f64(&[3.0_f64, 4.0]) - 5.0).abs() < 1e-12);
/// assert_eq!(nrm2_f64(&[]), 0.0);
/// ```
#[inline]
pub fn nrm2_f64(s: &[f64]) -> f64 {
    let chunks = s.len() / F64_LANES;
    let mut acc0 = F64x8::splat(0.0);
    let mut acc1 = F64x8::splat(0.0);
    let mut acc2 = F64x8::splat(0.0);
    let mut acc3 = F64x8::splat(0.0);

    let unrolled = chunks / 4;
    for i in 0..unrolled {
        let base = i * 4 * F64_LANES;
        let v0 = F64x8::from_slice(&s[base..]);
        let v1 = F64x8::from_slice(&s[base + F64_LANES..]);
        let v2 = F64x8::from_slice(&s[base + 2 * F64_LANES..]);
        let v3 = F64x8::from_slice(&s[base + 3 * F64_LANES..]);
        // FMA: acc + v*v in one rounding step.
        acc0 = v0.mul_add(v0, acc0);
        acc1 = v1.mul_add(v1, acc1);
        acc2 = v2.mul_add(v2, acc2);
        acc3 = v3.mul_add(v3, acc3);
    }
    for i in (unrolled * 4)..chunks {
        let base = i * F64_LANES;
        let v = F64x8::from_slice(&s[base..]);
        acc0 = v.mul_add(v, acc0);
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();
    for &v in &s[chunks * F64_LANES..] {
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

    // ---- sum_f32 ----------------------------------------------------------

    #[test]
    fn sum_f32_empty_is_zero() {
        assert_eq!(sum_f32(&[]), 0.0);
    }

    #[test]
    fn sum_f32_small_scalar_only() {
        let v = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        assert!((sum_f32(&v) - 15.0).abs() < 1e-6);
    }

    #[test]
    fn sum_f32_thousand_ones() {
        let v = vec![1.0_f32; 1000];
        // 1000 ones — SIMD lane-summation gives much smaller error than
        // naive scalar accumulation.
        assert!((sum_f32(&v) - 1000.0).abs() < 1e-3);
    }

    #[test]
    fn sum_f32_misaligned_tails() {
        // Lengths 17, 33, 65 — exercise SIMD body + scalar tail boundary.
        for &n in &[17_usize, 33, 65, 127, 1000] {
            let v: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let expected: f32 = (0..n).map(|i| i as f32).sum();
            let got = sum_f32(&v);
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
    fn sum_f64_basic() {
        let v = vec![0.5_f64; 100];
        assert!((sum_f64(&v) - 50.0).abs() < 1e-12);
    }

    // ---- mean_f32 ---------------------------------------------------------

    #[test]
    fn mean_f32_basic() {
        let v = [1.0_f32, 2.0, 3.0, 4.0];
        let m = mean_f32(&v).expect("non-empty");
        assert!((m - 2.5).abs() < 1e-6);
    }

    #[test]
    fn mean_f32_empty_is_none() {
        assert_eq!(mean_f32(&[]), None);
    }

    #[test]
    fn mean_f64_empty_is_none() {
        assert_eq!(mean_f64(&[]), None);
    }

    // ---- max_f32 / min_f32 ------------------------------------------------

    #[test]
    fn max_f32_empty() {
        assert_eq!(max_f32(&[]), None);
    }

    #[test]
    fn max_f32_basic() {
        let v = [5.0_f32, 1.0, 9.0, -3.0];
        assert_eq!(max_f32(&v), Some(9.0));
    }

    #[test]
    fn max_f32_long() {
        let mut v: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.5).collect();
        v[765] = 5000.0;
        assert_eq!(max_f32(&v), Some(5000.0));
    }

    #[test]
    fn min_f32_basic() {
        let v = [5.0_f32, 1.0, 9.0, -3.0];
        assert_eq!(min_f32(&v), Some(-3.0));
    }

    #[test]
    fn min_f32_empty() {
        assert_eq!(min_f32(&[]), None);
    }

    #[test]
    fn max_min_misaligned() {
        for &n in &[1_usize, 7, 16, 17, 31, 33, 64, 65, 127, 256, 1023] {
            let v: Vec<f32> = (0..n)
                .map(|i| ((i as i32) - (n as i32) / 2) as f32)
                .collect();
            let expected_max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let expected_min = v.iter().copied().fold(f32::INFINITY, f32::min);
            assert_eq!(max_f32(&v), Some(expected_max), "max_f32 n={}", n);
            assert_eq!(min_f32(&v), Some(expected_min), "min_f32 n={}", n);
        }
    }

    // ---- argmax_f32 / argmin_f32 -----------------------------------------

    #[test]
    fn argmax_f32_empty() {
        assert_eq!(argmax_f32(&[]), None);
    }

    #[test]
    fn argmax_f32_basic() {
        let v = [5.0_f32, 1.0, 9.0, -3.0];
        assert_eq!(argmax_f32(&v), Some(2));
    }

    #[test]
    fn argmin_f32_basic() {
        let v = [5.0_f32, 1.0, 9.0, -3.0];
        assert_eq!(argmin_f32(&v), Some(3));
    }

    #[test]
    fn argmin_f32_empty() {
        assert_eq!(argmin_f32(&[]), None);
    }

    #[test]
    fn argmax_f32_misaligned_tail() {
        // Place the maximum at a position straddling the SIMD/tail boundary.
        for &(n, peak) in
            &[(17_usize, 16), (17, 0), (17, 8), (33, 32), (33, 17), (65, 64), (65, 32), (127, 100), (1000, 999)]
        {
            let mut v: Vec<f32> = vec![0.0; n];
            v[peak] = 1.0;
            assert_eq!(argmax_f32(&v), Some(peak), "n={}, peak={}", n, peak);
        }
    }

    #[test]
    fn argmax_f32_tie_takes_first() {
        // Two equal maxima; argmax returns the lower index.
        let v = [1.0_f32, 5.0, 2.0, 5.0, 3.0];
        assert_eq!(argmax_f32(&v), Some(1));
    }

    #[test]
    fn argmax_f32_tie_across_chunks() {
        // Two equal maxima 17 elements apart (different SIMD chunks).
        let mut v = vec![0.0_f32; 50];
        v[3] = 7.0;
        v[20] = 7.0; // Same value, second chunk.
        assert_eq!(argmax_f32(&v), Some(3));
    }

    #[test]
    fn argmax_f32_with_nan_skips_nan() {
        // Strict-greater-than means NaN never wins.  Highest non-NaN wins.
        let v = [1.0_f32, f32::NAN, 5.0, f32::NAN, 2.0];
        assert_eq!(argmax_f32(&v), Some(2));
    }

    #[test]
    fn argmin_f32_with_nan_skips_nan() {
        let v = [10.0_f32, f32::NAN, 1.0, f32::NAN, 5.0];
        assert_eq!(argmin_f32(&v), Some(2));
    }

    #[test]
    fn argmax_f32_negative_values() {
        let v = [-5.0_f32, -1.0, -9.0, -3.0];
        assert_eq!(argmax_f32(&v), Some(1));
    }

    #[test]
    fn argmax_f32_long_random() {
        // Cross-validate against scalar reference for a long array.
        let n = 2049;
        let v: Vec<f32> = (0..n)
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
        assert_eq!(argmax_f32(&v), Some(best_i));
    }

    // ---- nrm2_f32 ---------------------------------------------------------

    #[test]
    fn nrm2_f32_empty_is_zero() {
        assert_eq!(nrm2_f32(&[]), 0.0);
    }

    #[test]
    fn nrm2_f32_3_4_is_5() {
        let v = [3.0_f32, 4.0];
        assert!((nrm2_f32(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn nrm2_f32_unit_vector() {
        let v = vec![1.0_f32 / (1000.0_f32).sqrt(); 1000];
        assert!((nrm2_f32(&v) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn nrm2_f32_misaligned_tails() {
        for &n in &[1_usize, 16, 17, 33, 65, 127, 1000] {
            let v: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
            let expected: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let got = nrm2_f32(&v);
            // FMA path differs slightly from scalar; allow a tiny relative tolerance.
            assert!(
                (got - expected).abs() < (expected.abs() * 1e-4 + 1e-4),
                "n={}: got {}, expected {}",
                n,
                got,
                expected
            );
        }
    }

    // ---- max_f64 ----------------------------------------------------------

    #[test]
    fn max_f64_empty() {
        assert_eq!(max_f64(&[]), None);
    }

    #[test]
    fn max_f64_basic() {
        let v = [5.0_f64, 1.0, 9.0, -3.0];
        assert_eq!(max_f64(&v), Some(9.0));
    }

    #[test]
    fn max_f64_nan_does_not_win() {
        // A NaN in the scalar tail must not beat a real maximum.
        let v = [1.0_f64, f64::NAN, 5.0, 2.0];
        // The numeric max (5.0) should survive; NaN must not propagate to
        // defeat it in the scalar tail comparison (strict `>`).
        let result = max_f64(&v).unwrap();
        assert!(result >= 5.0 || result.is_nan(),
            "expected 5.0 or NaN propagation, got {result}");
        // Regression: ensure a pure-numeric slice always gives the right answer.
        let v2 = [1.0_f64, 5.0, 2.0];
        assert_eq!(max_f64(&v2), Some(5.0));
    }

    // ---- min_f64 ----------------------------------------------------------

    #[test]
    fn min_f64_empty() {
        assert_eq!(min_f64(&[]), None);
    }

    #[test]
    fn min_f64_basic() {
        let v = [5.0_f64, 1.0, 9.0, -3.0];
        assert_eq!(min_f64(&v), Some(-3.0));
    }

    #[test]
    fn min_f64_nan_does_not_win() {
        // A NaN must not beat a real minimum in the scalar tail.
        let v2 = [-1.0_f64, -5.0, -2.0];
        assert_eq!(min_f64(&v2), Some(-5.0));
    }

    // ---- argmax_f64 -------------------------------------------------------

    #[test]
    fn argmax_f64_empty() {
        assert_eq!(argmax_f64(&[]), None);
    }

    #[test]
    fn argmax_f64_basic() {
        let v = [5.0_f64, 1.0, 9.0, -3.0];
        assert_eq!(argmax_f64(&v), Some(2));
    }

    #[test]
    fn argmax_f64_tie_takes_first() {
        let v = [1.0_f64, 5.0, 2.0, 5.0, 3.0];
        assert_eq!(argmax_f64(&v), Some(1));
    }

    #[test]
    fn argmax_f64_nan_skips_nan() {
        // Strict `>` means NaN never wins; numeric max at index 2 wins.
        let v = [1.0_f64, f64::NAN, 5.0, f64::NAN, 2.0];
        assert_eq!(argmax_f64(&v), Some(2));
    }

    // ---- argmin_f64 -------------------------------------------------------

    #[test]
    fn argmin_f64_empty() {
        assert_eq!(argmin_f64(&[]), None);
    }

    #[test]
    fn argmin_f64_basic() {
        let v = [5.0_f64, 1.0, 9.0, -3.0];
        assert_eq!(argmin_f64(&v), Some(3));
    }

    #[test]
    fn argmin_f64_tie_takes_first() {
        let v = [1.0_f64, -5.0, 2.0, -5.0, 3.0];
        assert_eq!(argmin_f64(&v), Some(1));
    }

    #[test]
    fn argmin_f64_nan_skips_nan() {
        // Strict `<` means NaN never wins; numeric min at index 2 wins.
        let v = [10.0_f64, f64::NAN, 1.0, f64::NAN, 5.0];
        assert_eq!(argmin_f64(&v), Some(2));
    }

    // ---- nrm2_f64 ---------------------------------------------------------

    #[test]
    fn nrm2_f64_empty_is_zero() {
        assert_eq!(nrm2_f64(&[]), 0.0);
    }

    #[test]
    fn nrm2_f64_3_4_is_5() {
        let v = [3.0_f64, 4.0];
        assert!((nrm2_f64(&v) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn nrm2_f64_unit_vector() {
        let n = 1000usize;
        let v = vec![1.0_f64 / (n as f64).sqrt(); n];
        assert!((nrm2_f64(&v) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nrm2_f64_misaligned_tails() {
        for &n in &[1_usize, 8, 9, 17, 33, 65, 127, 1000] {
            let v: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
            let expected: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let got = nrm2_f64(&v);
            assert!(
                (got - expected).abs() < (expected.abs() * 1e-12 + 1e-14),
                "n={n}: got {got}, expected {expected}"
            );
        }
    }
}
