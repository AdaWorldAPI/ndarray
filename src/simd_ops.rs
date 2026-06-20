//! Slice-level elementwise ops built on the polyfill SIMD types.
//!
//! Every function uses `crate::simd::F32x16` (or the appropriate type),
//! which is already dispatched: AVX-512 → AVX2 → NEON → scalar.
//! These ops inherit that dispatch — no platform-specific code here.
//!
//! Re-exported flat through `ndarray::simd::add_f32`, etc.
//!
//! # ⚠️  Foundation primitives — do not remove ⚠️
//!
//! `array_chunks`, `array_chunks_checked`, `array_windows`,
//! `array_windows_checked`, `add_mul_f32`, and `add_mul_f64` are the
//! stable-Rust building blocks the BLAS-graph GEMM hand-rolled kernels
//! lean on. Combined with the polyfilled `F32x16::mul_add` / `F64x8::mul_add`
//! (defined on the SIMD types in `src/simd_*.rs`), they reach within a few
//! percent of a Cranelift-JIT'd inner loop on the bgz17 path — the reason
//! the JIT-native option was deemed unnecessary and dropped. **Prior
//! sessions have removed these helpers under the impression they were
//! unused cruft; the truth is they are the polyfill foundation. Do not
//! delete them without verifying every BLAS-graph kernel still compiles
//! AND that the JIT alternative has been re-evaluated.**

use crate::simd::{bf16_to_f32_batch, F32x16, F64x8};

// ═══════════════════════════════════════════════════════════════════
// f32 binary ops (out-of-place)
// ═══════════════════════════════════════════════════════════════════

/// Elementwise add: `out[i] = a[i] + b[i]`.
pub fn add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary_f32(a, b, |x, y| x + y, |x, y| x + y)
}

/// Elementwise subtract: `out[i] = a[i] - b[i]`.
pub fn sub_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary_f32(a, b, |x, y| x - y, |x, y| x - y)
}

/// Elementwise multiply: `out[i] = a[i] * b[i]`.
pub fn mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary_f32(a, b, |x, y| x * y, |x, y| x * y)
}

/// Elementwise divide: `out[i] = a[i] / b[i]`.
pub fn div_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
    binary_f32(a, b, |x, y| x / y, |x, y| x / y)
}

// ═══════════════════════════════════════════════════════════════════
// f32 inplace ops
// ═══════════════════════════════════════════════════════════════════

/// Inplace add: `dst[i] += src[i]`.
pub fn add_f32_inplace(dst: &mut [f32], src: &[f32]) {
    inplace_f32(dst, src, |d, s| d + s, |d, s| *d += s)
}

/// Inplace subtract: `dst[i] -= src[i]`.
pub fn sub_f32_inplace(dst: &mut [f32], src: &[f32]) {
    inplace_f32(dst, src, |d, s| d - s, |d, s| *d -= s)
}

/// Inplace multiply: `dst[i] *= src[i]`.
pub fn mul_f32_inplace(dst: &mut [f32], src: &[f32]) {
    inplace_f32(dst, src, |d, s| d * s, |d, s| *d *= s)
}

/// Inplace divide: `dst[i] /= src[i]`.
pub fn div_f32_inplace(dst: &mut [f32], src: &[f32]) {
    inplace_f32(dst, src, |d, s| d / s, |d, s| *d /= s)
}

// ═══════════════════════════════════════════════════════════════════
// f32 scalar ops
// ═══════════════════════════════════════════════════════════════════

/// Scalar multiply: `out[i] = a[i] * scalar`.
pub fn scale_f32(a: &[f32], scalar: f32) -> Vec<f32> {
    let s = F32x16::splat(scalar);
    let n = a.len();
    let mut out = vec![0.0f32; n];
    let mut i = 0;
    while i + 16 <= n {
        (F32x16::from_slice(&a[i..]) * s).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = a[i] * scalar;
        i += 1;
    }
    out
}

/// Scalar add: `out[i] = a[i] + scalar`.
pub fn add_scalar_f32(a: &[f32], scalar: f32) -> Vec<f32> {
    let s = F32x16::splat(scalar);
    let n = a.len();
    let mut out = vec![0.0f32; n];
    let mut i = 0;
    while i + 16 <= n {
        (F32x16::from_slice(&a[i..]) + s).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = a[i] + scalar;
        i += 1;
    }
    out
}

/// Inplace scalar multiply: `a[i] *= scalar`.
pub fn scale_f32_inplace(a: &mut [f32], scalar: f32) {
    let s = F32x16::splat(scalar);
    let n = a.len();
    let mut i = 0;
    while i + 16 <= n {
        (F32x16::from_slice(&a[i..]) * s).copy_to_slice(&mut a[i..]);
        i += 16;
    }
    while i < n {
        a[i] *= scalar;
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Fused multiply-add into accumulator slice
// ═══════════════════════════════════════════════════════════════════
//
// `add_mul_*` is the slice-level FMA-into-accumulator: `acc[i] += a[i] * b[i]`.
// Built directly on `F32x16::mul_add` / `F64x8::mul_add`, the polyfilled
// FMA primitives on the SIMD types (AVX-512: vfmadd231ps; AVX2 + FMA:
// _mm256_fmadd_ps; NEON: vfmaq_f32; scalar: f32::mul_add → also fused
// on FMA-capable hosts). Single rounding step, no intermediate
// `a[i] * b[i]` temporary in scalar emulation — same semantic as
// BLAS-1 `axpy` with a vector multiplier, and the dominant inner-loop
// shape in the BLAS-graph GEMM kernels.

/// Fused multiply-add into accumulator: `acc[i] += a[i] * b[i]`.
///
/// Operates on the prefix `min(acc.len(), a.len(), b.len())` lanes.
/// Each lane is computed with a single rounding step via the polyfilled
/// `F32x16::mul_add` (hardware FMA where available). On scalar fallback
/// targets, `f32::mul_add` is also fused if the host has FMA — never
/// less precise than the manual `acc[i] += a[i] * b[i]` two-step.
///
/// # Examples
///
/// ```
/// use ndarray::simd::add_mul_f32;
/// let mut acc = vec![1.0f32; 17];
/// let a = vec![2.0f32; 17];
/// let b = vec![3.0f32; 17];
/// add_mul_f32(&mut acc, &a, &b);
/// assert!(acc.iter().all(|&v| (v - 7.0).abs() < 1e-6));
/// ```
#[inline]
pub fn add_mul_f32(acc: &mut [f32], a: &[f32], b: &[f32]) {
    let n = acc.len().min(a.len()).min(b.len());
    let mut i = 0;
    while i + 16 <= n {
        let va = F32x16::from_slice(&a[i..]);
        let vb = F32x16::from_slice(&b[i..]);
        let vacc = F32x16::from_slice(&acc[i..]);
        va.mul_add(vb, vacc).copy_to_slice(&mut acc[i..]);
        i += 16;
    }
    while i < n {
        acc[i] = a[i].mul_add(b[i], acc[i]);
        i += 1;
    }
}

/// Fused multiply-add into accumulator (f64): `acc[i] += a[i] * b[i]`.
///
/// f64 sibling of [`add_mul_f32`]. Uses `F64x8::mul_add` (8-wide on
/// AVX-512 / 4-wide on AVX2-FMA / 2-wide on NEON / scalar `f64::mul_add`).
#[inline]
pub fn add_mul_f64(acc: &mut [f64], a: &[f64], b: &[f64]) {
    let n = acc.len().min(a.len()).min(b.len());
    let mut i = 0;
    while i + 8 <= n {
        let va = F64x8::from_slice(&a[i..]);
        let vb = F64x8::from_slice(&b[i..]);
        let vacc = F64x8::from_slice(&acc[i..]);
        va.mul_add(vb, vacc).copy_to_slice(&mut acc[i..]);
        i += 8;
    }
    while i < n {
        acc[i] = a[i].mul_add(b[i], acc[i]);
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════
// f64 binary ops
// ═══════════════════════════════════════════════════════════════════

/// Elementwise add f64: `out[i] = a[i] + b[i]`.
pub fn add_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    binary_f64(a, b, |x, y| x + y, |x, y| x + y)
}

/// Elementwise multiply f64: `out[i] = a[i] * b[i]`.
pub fn mul_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    binary_f64(a, b, |x, y| x * y, |x, y| x * y)
}

/// Inplace add f64: `dst[i] += src[i]`.
pub fn add_f64_inplace(dst: &mut [f64], src: &[f64]) {
    inplace_f64(dst, src, |d, s| d + s, |d, s| *d += s)
}

// ═══════════════════════════════════════════════════════════════════
// Internal dispatch helpers
// ═══════════════════════════════════════════════════════════════════

#[inline]
fn binary_f32(
    a: &[f32], b: &[f32], simd_op: impl Fn(F32x16, F32x16) -> F32x16, scalar_op: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let n = a.len().min(b.len());
    let mut out = vec![0.0f32; n];
    let mut i = 0;
    while i + 16 <= n {
        simd_op(F32x16::from_slice(&a[i..]), F32x16::from_slice(&b[i..])).copy_to_slice(&mut out[i..]);
        i += 16;
    }
    while i < n {
        out[i] = scalar_op(a[i], b[i]);
        i += 1;
    }
    out
}

#[inline]
fn inplace_f32(
    dst: &mut [f32], src: &[f32], simd_op: impl Fn(F32x16, F32x16) -> F32x16, scalar_op: impl Fn(&mut f32, f32),
) {
    let n = dst.len().min(src.len());
    let mut i = 0;
    while i + 16 <= n {
        simd_op(F32x16::from_slice(&dst[i..]), F32x16::from_slice(&src[i..])).copy_to_slice(&mut dst[i..]);
        i += 16;
    }
    while i < n {
        scalar_op(&mut dst[i], src[i]);
        i += 1;
    }
}

#[inline]
fn binary_f64(
    a: &[f64], b: &[f64], simd_op: impl Fn(F64x8, F64x8) -> F64x8, scalar_op: impl Fn(f64, f64) -> f64,
) -> Vec<f64> {
    let n = a.len().min(b.len());
    let mut out = vec![0.0f64; n];
    let mut i = 0;
    while i + 8 <= n {
        simd_op(F64x8::from_slice(&a[i..]), F64x8::from_slice(&b[i..])).copy_to_slice(&mut out[i..]);
        i += 8;
    }
    while i < n {
        out[i] = scalar_op(a[i], b[i]);
        i += 1;
    }
    out
}

#[inline]
fn inplace_f64(
    dst: &mut [f64], src: &[f64], simd_op: impl Fn(F64x8, F64x8) -> F64x8, scalar_op: impl Fn(&mut f64, f64),
) {
    let n = dst.len().min(src.len());
    let mut i = 0;
    while i + 8 <= n {
        simd_op(F64x8::from_slice(&dst[i..]), F64x8::from_slice(&src[i..])).copy_to_slice(&mut dst[i..]);
        i += 8;
    }
    while i < n {
        scalar_op(&mut dst[i], src[i]);
        i += 1;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_f32_aligned() {
        let a = vec![1.0f32; 32];
        let b = vec![2.0f32; 32];
        let c = add_f32(&a, &b);
        assert!(c.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn add_f32_misaligned_tail() {
        let a = vec![1.0f32; 33];
        let b = vec![2.0f32; 33];
        let c = add_f32(&a, &b);
        assert_eq!(c.len(), 33);
        assert!(c.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn mul_f32_inplace_works() {
        let mut dst = vec![2.0f32; 17];
        let src = vec![3.0f32; 17];
        mul_f32_inplace(&mut dst, &src);
        assert!(dst.iter().all(|&v| (v - 6.0).abs() < 1e-6));
    }

    #[test]
    fn scale_f32_works() {
        let a = vec![4.0f32; 35];
        let b = scale_f32(&a, 0.5);
        assert!(b.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn scale_f32_inplace_works() {
        let mut a = vec![10.0f32; 19];
        scale_f32_inplace(&mut a, 0.1);
        assert!(a.iter().all(|&v| (v - 1.0).abs() < 1e-5));
    }

    #[test]
    fn add_scalar_f32_works() {
        let a = vec![1.0f32; 20];
        let b = add_scalar_f32(&a, 99.0);
        assert!(b.iter().all(|&v| (v - 100.0).abs() < 1e-6));
    }

    #[test]
    fn sub_f32_works() {
        let c = sub_f32(&[5.0; 3], &[2.0; 3]);
        assert!(c.iter().all(|&v| (v - 3.0).abs() < 1e-6));
    }

    #[test]
    fn div_f32_works() {
        let c = div_f32(&[6.0; 4], &[3.0; 4]);
        assert!(c.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    #[test]
    fn add_f64_works() {
        let c = add_f64(&[1.0f64; 17], &[2.0f64; 17]);
        assert_eq!(c.len(), 17);
        assert!(c.iter().all(|&v| (v - 3.0).abs() < 1e-12));
    }

    #[test]
    fn empty_slices() {
        assert!(add_f32(&[], &[]).is_empty());
        assert!(mul_f32(&[], &[]).is_empty());
        assert!(scale_f32(&[], 2.0).is_empty());
    }

    #[test]
    fn mismatched_lengths_takes_min() {
        let c = add_f32(&[1.0; 10], &[2.0; 5]);
        assert_eq!(c.len(), 5);
    }
}

// ════════════════════════════════════════════════════════════════════
// PR-X1 — Const-size slice window helpers (non-overlapping + overlapping)
// ════════════════════════════════════════════════════════════════════
//
// Two stable-Rust helpers for SIMD-staged inner loops:
//
//   - `array_chunks::<N>` — **non-overlapping** `&[T; N]` iterator
//     (matches `std::slice::ArrayChunks` / stable `slice::as_chunks`).
//   - `array_windows::<N>` — **overlapping** `&[T; N]` iterator
//     (matches nightly `std::slice::array_windows::<N>()`; this file
//     ships the stable equivalent until the std API lands).
//
// Both surface compile-time-fixed window sizes so call sites can
// directly feed `F32x16::from_array` / `F32x8::from_array` etc. and
// let the compiler infer the lane count. The pair lets BLAS-graph
// GEMM-style kernels iterate over a row of `B` (overlapping windows
// of the inner K dimension) and a column of `A` (non-overlapping
// chunks of the M dimension) in one source — the polyfill type
// resolves both to native SIMD width per target.

/// Walk `data` as a sequence of non-overlapping const-size windows.
///
/// Returns an iterator over `&[T; N]` references into `data`. The tail
/// (`data.len() % N` items) is discarded; use [`array_chunks_checked`] to
/// fail-fast when the length is not a multiple of `N`.
///
/// Zero-cost: thin wrapper around [`slice::as_chunks`] that pins the
/// chunk size at the call site for type inference.
///
/// # Examples
///
/// ```
/// use ndarray::simd::array_chunks;
/// let data: Vec<u8> = (0..16).collect();
/// let windows: Vec<&[u8; 4]> = array_chunks::<u8, 4>(&data).collect();
/// assert_eq!(windows.len(), 4);
/// assert_eq!(windows[0], &[0, 1, 2, 3]);
/// assert_eq!(windows[3], &[12, 13, 14, 15]);
/// ```
///
/// # Examples — tail discarded
///
/// ```
/// use ndarray::simd::array_chunks;
/// let data: Vec<u8> = (0..7).collect();
/// let windows: Vec<&[u8; 4]> = array_chunks::<u8, 4>(&data).collect();
/// assert_eq!(windows.len(), 1);
/// ```
#[inline]
pub fn array_chunks<T, const N: usize>(data: &[T]) -> impl Iterator<Item = &[T; N]> + '_ {
    data.as_chunks::<N>().0.iter()
}

/// Walk `data` as `&[T; N]` windows, returning `Err(())` if `data.len()`
/// is not a multiple of `N`.
///
/// Strict variant of [`array_chunks`]: the consumer asserts up front that
/// the buffer is lane-aligned and wants the error surfaced rather than
/// silently truncating.
///
/// # Edge case — `N == 0`
///
/// Returns `Err(())` when `N == 0`. The underlying `slice::as_chunks::<0>`
/// would panic on a zero divisor; the strict-fallible contract here folds
/// that into the existing error path so callers never see an unexpected
/// panic on the checked surface.
///
/// # Examples
///
/// ```
/// use ndarray::simd::array_chunks_checked;
/// let data: Vec<u8> = (0..16).collect();
/// let it = array_chunks_checked::<u8, 4>(&data).expect("16 is a multiple of 4");
/// assert_eq!(it.count(), 4);
///
/// let bad: Vec<u8> = (0..7).collect();
/// assert!(array_chunks_checked::<u8, 4>(&bad).is_err());
///
/// // N == 0 surfaces as Err rather than panicking.
/// assert!(array_chunks_checked::<u8, 0>(&[0u8; 8]).is_err());
/// ```
#[inline]
#[allow(clippy::result_unit_err)] // matches PR-X1 design § 3 `Result<_, ()>` contract; no error variants needed
pub fn array_chunks_checked<T, const N: usize>(data: &[T]) -> Result<impl Iterator<Item = &[T; N]> + '_, ()> {
    if N == 0 {
        // `data.len() % 0` would panic; the strict-fallible contract
        // folds N==0 into Err. See P2 review on PR #167.
        return Err(());
    }
    if !data.len().is_multiple_of(N) {
        return Err(());
    }
    Ok(array_chunks::<T, N>(data))
}

/// Walk `data` as a sequence of **overlapping** const-size windows.
///
/// Stable-Rust equivalent of nightly `std::slice::array_windows::<N>()`.
/// For input length `L`, yields `L.saturating_sub(N - 1)` windows; each
/// window slides forward by exactly one element.
///
/// Zero-cost: wraps `slice::windows(N)` with a `TryInto` array-ref
/// conversion that the compiler proves infallible (the inner slice is
/// guaranteed length `N`).
///
/// # Edge case — `N == 0`
///
/// `slice::windows(0)` panics. To preserve a panic-free surface, this
/// helper returns an empty iterator when `N == 0`. Strict-fallible
/// callers should use [`array_windows_checked`].
///
/// # Examples
///
/// ```
/// use ndarray::simd::array_windows;
/// let data: Vec<u8> = (0..6).collect();
/// let ws: Vec<&[u8; 4]> = array_windows::<u8, 4>(&data).collect();
/// assert_eq!(ws.len(), 3);                   // 6 - 4 + 1 = 3 overlapping windows
/// assert_eq!(ws[0], &[0, 1, 2, 3]);
/// assert_eq!(ws[1], &[1, 2, 3, 4]);
/// assert_eq!(ws[2], &[2, 3, 4, 5]);
/// ```
///
/// # Examples — fewer elements than `N`
///
/// ```
/// use ndarray::simd::array_windows;
/// let data: Vec<u8> = (0..3).collect();
/// let ws: Vec<&[u8; 4]> = array_windows::<u8, 4>(&data).collect();
/// assert_eq!(ws.len(), 0);
/// ```
#[inline]
pub fn array_windows<T, const N: usize>(data: &[T]) -> impl Iterator<Item = &[T; N]> + '_ {
    // Index-based iteration sidesteps `slice::windows(0)`'s panic — when
    // N == 0 the count below evaluates to 0 and the iterator is empty.
    let count = if N == 0 || data.len() < N {
        0
    } else {
        data.len() - N + 1
    };
    (0..count).map(move |i| {
        // `&data[i..i + N]` is exactly N elements by construction (bounds
        // checked once when `count` was computed). `try_into` always
        // succeeds on the inner conversion; the optimizer folds the
        // length check out and lowers this to a pointer-cast.
        <&[T; N]>::try_from(&data[i..i + N]).expect("computed window length == N")
    })
}

/// Walk `data` as `&[T; N]` overlapping windows, returning `Err(())` if
/// the buffer is shorter than `N` or `N == 0`.
///
/// Strict variant of [`array_windows`]: the consumer asserts up front
/// that the slice is long enough for at least one window and wants the
/// error surfaced rather than receiving an empty iterator. Useful for
/// kernels that pre-pad to `K + N - 1` and want to fail loudly if the
/// pad got dropped.
///
/// # Edge case — `N == 0`
///
/// Returns `Err(())`. The unchecked variant would silently produce an
/// empty iterator; the checked surface makes the misuse explicit.
///
/// # Examples
///
/// ```
/// use ndarray::simd::array_windows_checked;
/// let data: Vec<u8> = (0..6).collect();
/// let it = array_windows_checked::<u8, 4>(&data).expect("6 >= 4");
/// assert_eq!(it.count(), 3);
///
/// let too_short: Vec<u8> = (0..3).collect();
/// assert!(array_windows_checked::<u8, 4>(&too_short).is_err());
///
/// // N == 0 surfaces as Err rather than yielding an empty iterator.
/// assert!(array_windows_checked::<u8, 0>(&[0u8; 8]).is_err());
/// ```
#[inline]
#[allow(clippy::result_unit_err)] // matches PR-X1 design § 3 `Result<_, ()>` contract; no error variants needed
pub fn array_windows_checked<T, const N: usize>(data: &[T]) -> Result<impl Iterator<Item = &[T; N]> + '_, ()> {
    if N == 0 || data.len() < N {
        return Err(());
    }
    Ok(array_windows::<T, N>(data))
}

/// BF16 16×16 tile GEMM: `C[16,16] += A[16,K] · B[K,16]` where `A`, `B` are
/// BF16 row-major (`u16` bit patterns) and `C` is f32 row-major. `K` must be a
/// multiple of 32.
///
/// Pure SIMD-polyfill kernel: decodes BF16→f32 and accumulates with
/// [`F32x16::mul_add`] (FMA), so it rides the polyfill's per-arch escalation —
/// AVX-512 `VFMADD231PS` where available, the emulated `(F32x8, F32x8)` pair on
/// AVX2, NEON on aarch64, scalar otherwise. The `F32x16` wrapper owns the
/// dispatch; there is no AMX intrinsic and no external BLAS backend.
///
/// Accumulates into `c` (`+=`), matching BLAS `C := A·B + C`. Inputs are
/// decoded BF16→f32, so the result matches an f32-accumulated scalar reference
/// up to BF16 input precision (~2⁻⁸ per multiply, O(√K) accumulated).
///
/// # Panics
/// Panics if `k % 32 != 0`, `a_bf16.len() != 16 * k`, `b_bf16.len() != k * 16`,
/// or `c.len() != 256`.
///
/// # Examples
/// ```
/// use ndarray::simd::{bf16_tile_gemm_16x16, f32_to_bf16_scalar};
/// let k = 32;
/// let a = vec![f32_to_bf16_scalar(1.0); 16 * k];
/// let b = vec![f32_to_bf16_scalar(1.0); k * 16];
/// let mut c = vec![0.0f32; 256];
/// bf16_tile_gemm_16x16(&a, &b, &mut c, k);
/// assert!((c[0] - k as f32).abs() < 1e-3); // Σ_{p<k} 1·1 = k
/// ```
pub fn bf16_tile_gemm_16x16(a_bf16: &[u16], b_bf16: &[u16], c: &mut [f32], k: usize) {
    assert_eq!(k % 32, 0, "K must be a multiple of 32");
    assert_eq!(a_bf16.len(), 16 * k, "A must be 16xK BF16 row-major");
    assert_eq!(b_bf16.len(), k * 16, "B must be Kx16 BF16 row-major");
    assert_eq!(c.len(), 16 * 16, "C must be 16x16 f32 row-major");

    // Decode BF16 -> f32 once; the batch decode rides the polyfill too.
    let mut a_f32 = vec![0.0f32; a_bf16.len()];
    let mut b_f32 = vec![0.0f32; b_bf16.len()];
    bf16_to_f32_batch(a_bf16, &mut a_f32);
    bf16_to_f32_batch(b_bf16, &mut b_f32);

    // C[i,j] += dot(A row i, B col j) via F32x16 + FMA on 16-wide chunks.
    // K is a multiple of 32, so chunks_exact(16) tiles A's row / B's column
    // with no remainder.
    for i in 0..16 {
        let a_row = &a_f32[i * k..i * k + k];
        for j in 0..16 {
            // Gather B column j (row-major B is strided by 16) into a buffer so
            // the dot product hits the contiguous chunks_exact(16) fast path.
            let mut col = vec![0.0f32; k];
            for (kk, slot) in col.iter_mut().enumerate() {
                *slot = b_f32[kk * 16 + j];
            }
            let mut acc = F32x16::splat(0.0);
            for (ra, rb) in a_row.chunks_exact(16).zip(col.chunks_exact(16)) {
                acc = F32x16::from_slice(ra).mul_add(F32x16::from_slice(rb), acc);
            }
            c[i * 16 + j] += acc.reduce_sum();
        }
    }
}

#[cfg(test)]
mod bf16_tile_gemm_tests {
    use super::*;
    use crate::simd::{bf16_to_f32_batch, f32_to_bf16_batch, f32_to_bf16_scalar};

    /// Scalar BF16 reference (f32-accumulated, `+=`) — the correctness anchor.
    fn ref_gemm(a: &[f32], b: &[f32], c: &mut [f32], k: usize) {
        for i in 0..16 {
            for j in 0..16 {
                let mut s = 0.0f32;
                for kk in 0..k {
                    s += a[i * k + kk] * b[kk * 16 + j];
                }
                c[i * 16 + j] += s;
            }
        }
    }

    #[test]
    fn matches_scalar_reference_k64() {
        let k = 64;
        let mut a_f32 = vec![0.0f32; 16 * k];
        let mut b_f32 = vec![0.0f32; k * 16];
        for (i, v) in a_f32.iter_mut().enumerate() {
            *v =
                (((i as i32).wrapping_mul(1103515245).wrapping_add(12345) >> 8) as f32 / 2147483648.0).clamp(-1.0, 1.0);
        }
        for (i, v) in b_f32.iter_mut().enumerate() {
            *v = (((i as i32).wrapping_mul(69069).wrapping_add(1) >> 8) as f32 / 2147483648.0).clamp(-1.0, 1.0);
        }
        let mut a_bf16 = vec![0u16; a_f32.len()];
        let mut b_bf16 = vec![0u16; b_f32.len()];
        f32_to_bf16_batch(&a_f32, &mut a_bf16);
        f32_to_bf16_batch(&b_f32, &mut b_bf16);

        // Reference consumes the same BF16-truncated inputs the kernel sees.
        let mut a_back = vec![0.0f32; a_f32.len()];
        let mut b_back = vec![0.0f32; b_f32.len()];
        bf16_to_f32_batch(&a_bf16, &mut a_back);
        bf16_to_f32_batch(&b_bf16, &mut b_back);
        let mut c_ref = vec![0.0f32; 256];
        ref_gemm(&a_back, &b_back, &mut c_ref, k);

        let mut c = vec![0.0f32; 256];
        bf16_tile_gemm_16x16(&a_bf16, &b_bf16, &mut c, k);

        let max_err = c
            .iter()
            .zip(&c_ref)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "polyfill vs scalar ref max_err = {max_err}");
    }

    #[test]
    fn accumulates_into_c() {
        // C is preloaded with 5.0; the kernel must ADD, not overwrite.
        let k = 32;
        let a = vec![f32_to_bf16_scalar(1.0); 16 * k];
        let b = vec![f32_to_bf16_scalar(1.0); k * 16];
        let mut c = vec![5.0f32; 256];
        bf16_tile_gemm_16x16(&a, &b, &mut c, k);
        // 5 (preload) + Σ_{p<32} 1·1 = 37
        for v in &c {
            assert!((v - 37.0).abs() < 1e-3, "got {v}");
        }
    }
}

#[cfg(test)]
mod array_chunks_tests {
    use super::*;

    #[test]
    fn array_chunks_4_over_16() {
        let data: Vec<u8> = (0u8..16).collect();
        let windows: Vec<&[u8; 4]> = array_chunks::<u8, 4>(&data).collect();
        assert_eq!(windows.len(), 4);
        assert_eq!(windows[0], &[0, 1, 2, 3]);
        assert_eq!(windows[1], &[4, 5, 6, 7]);
        assert_eq!(windows[2], &[8, 9, 10, 11]);
        assert_eq!(windows[3], &[12, 13, 14, 15]);
    }

    #[test]
    fn array_chunks_drops_tail() {
        let data: Vec<u8> = (0u8..7).collect();
        let windows: Vec<&[u8; 4]> = array_chunks::<u8, 4>(&data).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0], &[0, 1, 2, 3]);
    }

    #[test]
    fn array_chunks_checked_rejects_mismatch() {
        assert!(array_chunks_checked::<u8, 4>(&[0u8; 7]).is_err());
        assert!(array_chunks_checked::<u8, 4>(&[0u8; 5]).is_err());
        assert!(array_chunks_checked::<u8, 4>(&[0u8; 1]).is_err());
    }

    #[test]
    fn array_chunks_checked_accepts_aligned() {
        let data = [0u8; 16];
        let it = array_chunks_checked::<u8, 4>(&data).expect("16 is a multiple of 4");
        assert_eq!(it.count(), 4);
    }

    #[test]
    fn array_chunks_empty_buffer() {
        assert_eq!(array_chunks::<u8, 4>(&[]).count(), 0);
        let it = array_chunks_checked::<u8, 4>(&[]).expect("0 % 4 == 0, should be Ok");
        assert_eq!(it.count(), 0);
    }

    #[test]
    fn array_chunks_checked_rejects_zero_n() {
        // P2 review fix on PR #167: N == 0 must surface as Err, not panic
        // via `data.len() % 0`.
        assert!(array_chunks_checked::<u8, 0>(&[]).is_err());
        assert!(array_chunks_checked::<u8, 0>(&[0u8; 8]).is_err());
        assert!(array_chunks_checked::<u32, 0>(&[1u32, 2, 3]).is_err());
    }

    // ── array_windows (overlapping) ──────────────────────────────────

    #[test]
    fn array_windows_4_over_6() {
        let data: Vec<u8> = (0u8..6).collect();
        let ws: Vec<&[u8; 4]> = array_windows::<u8, 4>(&data).collect();
        assert_eq!(ws.len(), 3);
        assert_eq!(ws[0], &[0, 1, 2, 3]);
        assert_eq!(ws[1], &[1, 2, 3, 4]);
        assert_eq!(ws[2], &[2, 3, 4, 5]);
    }

    #[test]
    fn array_windows_short_buffer_empty_iter() {
        let data: Vec<u8> = (0u8..3).collect();
        let ws: Vec<&[u8; 4]> = array_windows::<u8, 4>(&data).collect();
        assert_eq!(ws.len(), 0);
    }

    #[test]
    fn array_windows_exact_n_yields_one() {
        let data: [u8; 4] = [10, 20, 30, 40];
        let ws: Vec<&[u8; 4]> = array_windows::<u8, 4>(&data).collect();
        assert_eq!(ws.len(), 1);
        assert_eq!(ws[0], &[10, 20, 30, 40]);
    }

    #[test]
    fn array_windows_empty_buffer() {
        let data: [u8; 0] = [];
        let ws: Vec<&[u8; 4]> = array_windows::<u8, 4>(&data).collect();
        assert_eq!(ws.len(), 0);
    }

    #[test]
    fn array_windows_n_zero_yields_empty() {
        // N == 0 returns empty iter from the unchecked variant — matches
        // slice::windows(0) panic-avoidance contract documented above.
        let data: [u8; 8] = [1; 8];
        let ws: Vec<&[u8; 0]> = array_windows::<u8, 0>(&data).collect();
        assert_eq!(ws.len(), 0);
    }

    #[test]
    fn array_windows_checked_accepts_long_enough() {
        let data: [u8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let it = array_windows_checked::<u8, 4>(&data).expect("8 >= 4");
        let ws: Vec<&[u8; 4]> = it.collect();
        assert_eq!(ws.len(), 5);
        assert_eq!(ws[4], &[4, 5, 6, 7]);
    }

    #[test]
    fn array_windows_checked_rejects_short_buffer() {
        assert!(array_windows_checked::<u8, 4>(&[0u8; 3]).is_err());
        assert!(array_windows_checked::<u8, 4>(&[0u8; 0]).is_err());
    }

    #[test]
    fn array_windows_checked_rejects_zero_n() {
        assert!(array_windows_checked::<u8, 0>(&[0u8; 8]).is_err());
        assert!(array_windows_checked::<u8, 0>(&[]).is_err());
    }
}

// ════════════════════════════════════════════════════════════════════
// FMA-into-accumulator tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod add_mul_tests {
    use super::*;

    fn ref_add_mul_f32(acc: &[f32], a: &[f32], b: &[f32]) -> Vec<f32> {
        acc.iter()
            .zip(a)
            .zip(b)
            .map(|((&c, &x), &y)| x.mul_add(y, c))
            .collect()
    }

    fn ref_add_mul_f64(acc: &[f64], a: &[f64], b: &[f64]) -> Vec<f64> {
        acc.iter()
            .zip(a)
            .zip(b)
            .map(|((&c, &x), &y)| x.mul_add(y, c))
            .collect()
    }

    #[test]
    fn add_mul_f32_aligned() {
        let mut acc = vec![1.0f32; 32];
        let a = vec![2.0f32; 32];
        let b = vec![3.0f32; 32];
        let expected = ref_add_mul_f32(&acc, &a, &b);
        add_mul_f32(&mut acc, &a, &b);
        for (got, want) in acc.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-5, "got={got}, want={want}");
        }
    }

    #[test]
    fn add_mul_f32_tail() {
        // 17 elements — exercises 16-wide chunk + 1-element scalar tail.
        for &len in &[0usize, 1, 15, 16, 17, 31, 32, 33, 100] {
            let mut acc: Vec<f32> = (0..len).map(|i| i as f32 + 0.5).collect();
            let a: Vec<f32> = (0..len).map(|i| (i as f32) * 0.1).collect();
            let b: Vec<f32> = (0..len).map(|i| (i as f32) * 0.2 - 0.3).collect();
            let expected = ref_add_mul_f32(&acc, &a, &b);
            add_mul_f32(&mut acc, &a, &b);
            for (i, (got, want)) in acc.iter().zip(expected.iter()).enumerate() {
                assert!((got - want).abs() < 1e-4, "len={len} idx={i}: got={got} want={want}");
            }
        }
    }

    #[test]
    fn add_mul_f32_mismatched_lengths_takes_min() {
        let mut acc = vec![0.0f32; 5];
        let a = vec![1.0f32; 10];
        let b = vec![2.0f32; 3];
        add_mul_f32(&mut acc, &a, &b);
        // First 3 lanes accumulate; remaining 2 unchanged.
        assert_eq!(&acc[..3], &[2.0, 2.0, 2.0]);
        assert_eq!(&acc[3..], &[0.0, 0.0]);
    }

    #[test]
    fn add_mul_f64_aligned_and_tail() {
        for &len in &[0usize, 1, 7, 8, 9, 16, 17, 50] {
            let mut acc: Vec<f64> = (0..len).map(|i| i as f64 + 0.25).collect();
            let a: Vec<f64> = (0..len).map(|i| (i as f64) * 0.5).collect();
            let b: Vec<f64> = (0..len).map(|i| (i as f64) * 0.7 - 1.0).collect();
            let expected = ref_add_mul_f64(&acc, &a, &b);
            add_mul_f64(&mut acc, &a, &b);
            for (i, (got, want)) in acc.iter().zip(expected.iter()).enumerate() {
                assert!((got - want).abs() < 1e-10, "len={len} idx={i}: got={got} want={want}");
            }
        }
    }
}
