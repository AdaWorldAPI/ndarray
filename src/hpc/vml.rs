//! VML: Vectorized Math Library.
//!
//! Element-wise transcendental and arithmetic operations over [`ArrayView`] /
//! [`ArrayViewMut`] inputs. SIMD-accelerated via `F32x16` / `F64x8` compat
//! types on the contiguous fast path; stride-aware [`Zip`] fallback on the
//! cold path.
//!
//! Each `pub fn` is generic over dimension `D: Dimension`, so callers can
//! pass 1-D, 2-D, or N-D views interchangeably. When all inputs share a
//! contiguous memory order, the kernel forwards a flat slice to the SIMD
//! primitive layer (`src/simd_*.rs`); otherwise it falls back to a
//! `Zip::from(...).for_each(...)` walk that respects strides.
//!
//! # Bridge pattern (W2-2a, see `.claude/knowledge/w2-arrayview-migration.md`)
//!
//! - Public surface: ArrayView / ArrayViewMut (kernel layer).
//! - Private `*_slice` helpers: typed-lane SIMD over flat `&[T]` / `&mut [T]`
//!   (primitive layer — kept slice-based on purpose, packed data has no
//!   shape).
//! - Pure Rust implementations; MKL-accelerated versions behind `intel-mkl`
//!   feature gate.

use crate::simd::{simd_exp_f32, simd_ln_f32, F32x16, F64x8};
use crate::{ArrayView, ArrayViewMut, Dimension, Zip};

// ===========================================================================
// Private slice-based SIMD primitives (preserved from the pre-W2-2a impls).
// These stay slice-based on purpose: SIMD primitives consume flat packed
// data and the typed lanes (`F32x16`, `F64x8`) take `&[T]`.
// ===========================================================================

fn vsexp_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        simd_exp_f32(F32x16::from_slice(x_chunk)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.exp();
    }
}

fn vdexp_slice(x: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v.exp();
    }
}

fn vsln_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        simd_ln_f32(F32x16::from_slice(x_chunk)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.ln();
    }
}

fn vdln_slice(x: &[f64], out: &mut [f64]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) {
        *o = v.ln();
    }
}

fn vssqrt_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        F32x16::from_slice(x_chunk).sqrt().copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.sqrt();
    }
}

fn vdsqrt_slice(x: &[f64], out: &mut [f64]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
        F64x8::from_slice(x_chunk).sqrt().copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 8;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.sqrt();
    }
}

fn vsabs_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        F32x16::from_slice(x_chunk).abs().copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.abs();
    }
}

fn vdabs_slice(x: &[f64], out: &mut [f64]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(8).zip(out.chunks_exact_mut(8)) {
        F64x8::from_slice(x_chunk).abs().copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 8;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.abs();
    }
}

fn vsadd_slice(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let (a, b, out) = (&a[..n], &b[..n], &mut out[..n]);
    for ((a_chunk, b_chunk), out_chunk) in a
        .chunks_exact(16)
        .zip(b.chunks_exact(16))
        .zip(out.chunks_exact_mut(16))
    {
        (F32x16::from_slice(a_chunk) + F32x16::from_slice(b_chunk)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for ((&av, &bv), o) in a[tail_start..]
        .iter()
        .zip(b[tail_start..].iter())
        .zip(out[tail_start..].iter_mut())
    {
        *o = av + bv;
    }
}

fn vsmul_slice(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let (a, b, out) = (&a[..n], &b[..n], &mut out[..n]);
    for ((a_chunk, b_chunk), out_chunk) in a
        .chunks_exact(16)
        .zip(b.chunks_exact(16))
        .zip(out.chunks_exact_mut(16))
    {
        (F32x16::from_slice(a_chunk) * F32x16::from_slice(b_chunk)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for ((&av, &bv), o) in a[tail_start..]
        .iter()
        .zip(b[tail_start..].iter())
        .zip(out[tail_start..].iter_mut())
    {
        *o = av * bv;
    }
}

fn vsdiv_slice(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let (a, b, out) = (&a[..n], &b[..n], &mut out[..n]);
    for ((a_chunk, b_chunk), out_chunk) in a
        .chunks_exact(16)
        .zip(b.chunks_exact(16))
        .zip(out.chunks_exact_mut(16))
    {
        (F32x16::from_slice(a_chunk) / F32x16::from_slice(b_chunk)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for ((&av, &bv), o) in a[tail_start..]
        .iter()
        .zip(b[tail_start..].iter())
        .zip(out[tail_start..].iter_mut())
    {
        *o = av / bv;
    }
}

fn vssin_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        let v = F32x16::from_slice(x_chunk);
        let arr = v.to_array();
        let mut res = [0.0f32; 16];
        for j in 0..16 {
            res[j] = arr[j].sin();
        }
        F32x16::from_array(res).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.sin();
    }
}

fn vscos_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let (x, out) = (&x[..n], &mut out[..n]);
    for (x_chunk, out_chunk) in x.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
        let v = F32x16::from_slice(x_chunk);
        let arr = v.to_array();
        let mut res = [0.0f32; 16];
        for j in 0..16 {
            res[j] = arr[j].cos();
        }
        F32x16::from_array(res).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
        *o = v.cos();
    }
}

fn vspow_slice(a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len().min(b.len()).min(out.len());
    let (a, b, out) = (&a[..n], &b[..n], &mut out[..n]);
    for ((a_chunk, b_chunk), out_chunk) in a
        .chunks_exact(16)
        .zip(b.chunks_exact(16))
        .zip(out.chunks_exact_mut(16))
    {
        let va = F32x16::from_slice(a_chunk);
        let vb = F32x16::from_slice(b_chunk);
        // a^b = exp(b * ln(a))
        simd_exp_f32(vb * simd_ln_f32(va)).copy_to_slice(out_chunk);
    }
    let tail_start = n - n % 16;
    for ((&av, &bv), o) in a[tail_start..]
        .iter()
        .zip(b[tail_start..].iter())
        .zip(out[tail_start..].iter_mut())
    {
        *o = av.powf(bv);
    }
}

fn vstanh_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    let two = F32x16::splat(2.0);
    let one = F32x16::splat(1.0);
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        // tanh(x) = 2·sigmoid(2x) - 1
        let two_x = v * two;
        // sigmoid(z) = 1/(1+exp(-z))
        let neg_two_x = F32x16::splat(0.0) - two_x;
        let exp_neg = simd_exp_f32(neg_two_x);
        let sigmoid = one / (exp_neg + one);
        let tanh_v = sigmoid.mul_add(two, F32x16::splat(-1.0));
        tanh_v.copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = x[i].tanh();
    }
}

fn vsfloor_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        v.floor().copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = x[i].floor();
    }
}

fn vsceil_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        // ceil = -floor(-x)
        let neg = F32x16::splat(0.0) - v;
        let floored = neg.floor();
        let ceiled = F32x16::splat(0.0) - floored;
        ceiled.copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = x[i].ceil();
    }
}

fn vsround_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        v.round().copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = x[i].round_ties_even();
    }
}

fn vsneg_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    let zero = F32x16::splat(0.0);
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        (zero - v).copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = -x[i];
    }
}

fn vstrunc_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        let abs_v = v.abs();
        let floored = abs_v.floor();
        let zero = F32x16::splat(0.0);
        let mask = v.simd_lt(zero); // true where x < 0
        let pos_result = floored;
        let neg_result = zero - floored;
        let result = mask.select(neg_result, pos_result);
        result.copy_to_slice(&mut out[off..off + 16]);
    }
    for i in (chunks * 16)..n {
        out[i] = x[i].trunc();
    }
}

// ===========================================================================
// Internal helpers for the contiguous fast path.
// ===========================================================================

/// Run the SIMD slice primitive when every view shares a contiguous memory
/// order (any axis order — C, F, or arbitrary, as long as strides walk the
/// underlying buffer sequentially). Returns `true` if dispatched.
#[inline]
fn dispatch_unary_contig<T, D, F>(x: ArrayView<'_, T, D>, out: &mut ArrayViewMut<'_, T, D>, f: F) -> bool
where
    D: Dimension,
    F: FnOnce(&[T], &mut [T]),
{
    if x.strides() != out.strides() {
        return false;
    }
    let xs = match x.as_slice_memory_order() {
        Some(s) => s,
        None => return false,
    };
    let os = match out.as_slice_memory_order_mut() {
        Some(s) => s,
        None => return false,
    };
    f(xs, os);
    true
}

#[inline]
fn dispatch_binary_contig<T, D, F>(
    a: ArrayView<'_, T, D>, b: ArrayView<'_, T, D>, out: &mut ArrayViewMut<'_, T, D>, f: F,
) -> bool
where
    D: Dimension,
    F: FnOnce(&[T], &[T], &mut [T]),
{
    if a.strides() != b.strides() || a.strides() != out.strides() {
        return false;
    }
    let as_ = match a.as_slice_memory_order() {
        Some(s) => s,
        None => return false,
    };
    let bs_ = match b.as_slice_memory_order() {
        Some(s) => s,
        None => return false,
    };
    let os_ = match out.as_slice_memory_order_mut() {
        Some(s) => s,
        None => return false,
    };
    f(as_, bs_, os_);
    true
}

// ===========================================================================
// Public ArrayView-first API (W2-2a). 16 unary + 4 binary.
// ===========================================================================

/// Element-wise `exp`: `out[i] = x[i].exp()`.
///
/// Hot path (all inputs contiguous, same memory order) dispatches to the
/// F32x16 SIMD polynomial via `simd_exp_f32`. Cold path uses a stride-aware
/// `Zip` walk.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
///
/// # Example
/// ```
/// use ndarray::{arr1, hpc::vml::vsexp};
/// let x = arr1(&[0.0_f32, 1.0]);
/// let mut out = arr1(&[0.0_f32; 2]);
/// vsexp(x.view(), out.view_mut());
/// assert!((out[0] - 1.0).abs() < 1e-5);
/// ```
pub fn vsexp<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vsexp: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vsexp_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.exp());
}

/// Element-wise `exp` (f64). Scalar loop — no SIMD polynomial for f64 exp yet.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vdexp<D: Dimension>(x: ArrayView<'_, f64, D>, mut out: ArrayViewMut<'_, f64, D>) {
    assert_eq!(x.shape(), out.shape(), "vdexp: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vdexp_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.exp());
}

/// Element-wise natural log: `out[i] = x[i].ln()`.
///
/// Hot path uses the F32x16 SIMD `simd_ln_f32` polynomial.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vsln<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vsln: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vsln_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.ln());
}

/// Element-wise natural log (f64). Scalar loop.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vdln<D: Dimension>(x: ArrayView<'_, f64, D>, mut out: ArrayViewMut<'_, f64, D>) {
    assert_eq!(x.shape(), out.shape(), "vdln: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vdln_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.ln());
}

/// Element-wise `sqrt`. Hot path uses `F32x16::sqrt()` (16-wide).
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vssqrt<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vssqrt: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vssqrt_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.sqrt());
}

/// Element-wise `sqrt` (f64). Hot path uses `F64x8::sqrt()` (8-wide).
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vdsqrt<D: Dimension>(x: ArrayView<'_, f64, D>, mut out: ArrayViewMut<'_, f64, D>) {
    assert_eq!(x.shape(), out.shape(), "vdsqrt: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vdsqrt_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.sqrt());
}

/// Element-wise `abs`. Hot path uses `F32x16::abs()`.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vsabs<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vsabs: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vsabs_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.abs());
}

/// Element-wise `abs` (f64). Hot path uses `F64x8::abs()`.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vdabs<D: Dimension>(x: ArrayView<'_, f64, D>, mut out: ArrayViewMut<'_, f64, D>) {
    assert_eq!(x.shape(), out.shape(), "vdabs: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vdabs_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.abs());
}

/// Element-wise add: `out[i] = a[i] + b[i]`. Hot path uses F32x16 operator
/// overload (16 elements per iteration).
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
pub fn vsadd<D: Dimension>(a: ArrayView<'_, f32, D>, b: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(a.shape(), b.shape(), "vsadd: a/b shape mismatch");
    assert_eq!(a.shape(), out.shape(), "vsadd: a/out shape mismatch");
    if dispatch_binary_contig(a.view(), b.view(), &mut out, vsadd_slice) {
        return;
    }
    Zip::from(&mut out)
        .and(&a)
        .and(&b)
        .for_each(|o, &x, &y| *o = x + y);
}

/// Element-wise mul: `out[i] = a[i] * b[i]`. Hot path uses F32x16 operator
/// overload.
///
/// # Panics
/// Panics if `a`, `b`, `out` do not all have the same shape.
pub fn vsmul<D: Dimension>(a: ArrayView<'_, f32, D>, b: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(a.shape(), b.shape(), "vsmul: a/b shape mismatch");
    assert_eq!(a.shape(), out.shape(), "vsmul: a/out shape mismatch");
    if dispatch_binary_contig(a.view(), b.view(), &mut out, vsmul_slice) {
        return;
    }
    Zip::from(&mut out)
        .and(&a)
        .and(&b)
        .for_each(|o, &x, &y| *o = x * y);
}

/// Element-wise div: `out[i] = a[i] / b[i]`. Hot path uses F32x16 operator
/// overload.
///
/// # Panics
/// Panics if `a`, `b`, `out` do not all have the same shape.
pub fn vsdiv<D: Dimension>(a: ArrayView<'_, f32, D>, b: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(a.shape(), b.shape(), "vsdiv: a/b shape mismatch");
    assert_eq!(a.shape(), out.shape(), "vsdiv: a/out shape mismatch");
    if dispatch_binary_contig(a.view(), b.view(), &mut out, vsdiv_slice) {
        return;
    }
    Zip::from(&mut out)
        .and(&a)
        .and(&b)
        .for_each(|o, &x, &y| *o = x / y);
}

/// Element-wise `sin`. Hot path loads 16 lanes at a time, applies scalar
/// `f32::sin` per lane, stores back; amortizes load/store overhead.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vssin<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vssin: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vssin_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.sin());
}

/// Element-wise `cos`. Hot path loads 16 lanes at a time, applies scalar
/// `f32::cos` per lane.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vscos<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vscos: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vscos_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.cos());
}

/// Element-wise `pow`: `out[i] = a[i] ^ b[i]`, via `exp(b * ln(a))`.
///
/// **Domain restriction**: the SIMD path requires `a[i] > 0`. Negative bases
/// produce NaN (since `ln(negative)` is undefined), unlike scalar `powf`
/// which handles some negative-base cases. The scalar tail (len < 16) and
/// the cold path use `powf` and therefore handle negative bases.
///
/// # Panics
/// Panics if `a`, `b`, `out` do not all have the same shape.
pub fn vspow<D: Dimension>(a: ArrayView<'_, f32, D>, b: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(a.shape(), b.shape(), "vspow: a/b shape mismatch");
    assert_eq!(a.shape(), out.shape(), "vspow: a/out shape mismatch");
    if dispatch_binary_contig(a.view(), b.view(), &mut out, vspow_slice) {
        return;
    }
    Zip::from(&mut out)
        .and(&a)
        .and(&b)
        .for_each(|o, &x, &y| *o = x.powf(y));
}

/// Element-wise `tanh` via the identity `tanh(x) = 2·sigmoid(2x) − 1`. Hot
/// path reuses the F32x16 sigmoid built from `simd_exp_f32`.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vstanh<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vstanh: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vstanh_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.tanh());
}

/// Element-wise `floor`. Hot path uses `F32x16::floor()` (VRNDSCALEPS on
/// AVX-512).
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vsfloor<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vsfloor: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vsfloor_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.floor());
}

/// Element-wise `ceil`. Hot path uses `-floor(-x)` on `F32x16`.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vsceil<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vsceil: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vsceil_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.ceil());
}

/// Element-wise round (ties to even). Hot path uses `F32x16::round()`.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vsround<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vsround: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vsround_slice) {
        return;
    }
    Zip::from(&mut out)
        .and(&x)
        .for_each(|o, &v| *o = v.round_ties_even());
}

/// Element-wise negate: `out[i] = -x[i]`.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vsneg<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vsneg: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vsneg_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = -v);
}

/// Element-wise `trunc` (round toward zero).
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vstrunc<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vstrunc: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vstrunc_slice) {
        return;
    }
    Zip::from(&mut out).and(&x).for_each(|o, &v| *o = v.trunc());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arr1, arr2, s, Array1, Array2};

    // -----------------------------------------------------------------------
    // vsexp
    // -----------------------------------------------------------------------

    #[test]
    fn test_vsexp_contig() {
        let x = arr1(&[0.0f32, 1.0, 2.0]);
        let mut out = Array1::<f32>::zeros(3);
        vsexp(x.view(), out.view_mut());
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - std::f32::consts::E).abs() < 1e-5);
        assert!((out[2] - (2.0_f32).exp()).abs() < 1e-5);
    }

    #[test]
    fn test_vsexp_strided() {
        let full = arr1(&[0.0f32, 9.0, 1.0, 9.0, 2.0, 9.0]);
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(3);
        vsexp(strided, out.view_mut());
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - std::f32::consts::E).abs() < 1e-5);
    }

    #[test]
    fn test_vsexp_2d() {
        let x = arr2(&[[0.0f32, 1.0], [2.0, 3.0]]);
        let mut out = Array2::<f32>::zeros((2, 2));
        vsexp(x.view(), out.view_mut());
        for ((r, c), &v) in out.indexed_iter() {
            assert!((v - x[[r, c]].exp()).abs() < 1e-4);
        }
    }

    // -----------------------------------------------------------------------
    // vdexp
    // -----------------------------------------------------------------------

    #[test]
    fn test_vdexp_contig() {
        let x = arr1(&[0.0f64, 1.0, 2.0]);
        let mut out = Array1::<f64>::zeros(3);
        vdexp(x.view(), out.view_mut());
        for i in 0..3 {
            assert!((out[i] - x[i].exp()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_vdexp_strided() {
        let full = arr1(&[0.0f64, 99.0, 1.0, 99.0, 2.0, 99.0]);
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f64>::zeros(3);
        vdexp(strided, out.view_mut());
        assert!((out[0] - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // vsln
    // -----------------------------------------------------------------------

    #[test]
    fn test_vsln_contig() {
        let x = arr1(&[1.0f32, std::f32::consts::E]);
        let mut out = Array1::<f32>::zeros(2);
        vsln(x.view(), out.view_mut());
        assert!(out[0].abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vsln_simd_path() {
        // exercise the >=16 SIMD path
        let x: Array1<f32> = (1..=20).map(|i| i as f32).collect();
        let mut out = Array1::<f32>::zeros(20);
        vsln(x.view(), out.view_mut());
        for i in 0..20 {
            assert!((out[i] - ((i + 1) as f32).ln()).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn test_vsln_strided() {
        let full: Array1<f32> = (0..32).map(|i| (i + 1) as f32).collect();
        let strided = full.slice(s![..;2]); // 16 elements, non-contig
        let mut out = Array1::<f32>::zeros(16);
        vsln(strided.view(), out.view_mut());
        for i in 0..16 {
            let expected = ((2 * i + 1) as f32).ln();
            assert!((out[i] - expected).abs() < 1e-5, "mismatch at {i}");
        }
    }

    // -----------------------------------------------------------------------
    // vdln
    // -----------------------------------------------------------------------

    #[test]
    fn test_vdln_contig() {
        let x = arr1(&[1.0f64, std::f64::consts::E]);
        let mut out = Array1::<f64>::zeros(2);
        vdln(x.view(), out.view_mut());
        assert!(out[0].abs() < 1e-10);
        assert!((out[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_vdln_strided() {
        let full = arr1(&[1.0f64, 0.0, std::f64::consts::E, 0.0]);
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f64>::zeros(2);
        vdln(strided, out.view_mut());
        assert!((out[1] - 1.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // vssqrt
    // -----------------------------------------------------------------------

    #[test]
    fn test_vssqrt_contig() {
        let x = arr1(&[4.0f32, 9.0, 16.0]);
        let mut out = Array1::<f32>::zeros(3);
        vssqrt(x.view(), out.view_mut());
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 3.0).abs() < 1e-5);
        assert!((out[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_vssqrt_strided() {
        let full = arr1(&[4.0f32, 0.0, 9.0, 0.0, 16.0, 0.0]);
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(3);
        vssqrt(strided, out.view_mut());
        assert!((out[1] - 3.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // vdsqrt
    // -----------------------------------------------------------------------

    #[test]
    fn test_vdsqrt_simd() {
        let x: Array1<f64> = (1..=12).map(|i| (i * i) as f64).collect();
        let mut out = Array1::<f64>::zeros(12);
        vdsqrt(x.view(), out.view_mut());
        for i in 0..12 {
            assert!((out[i] - (i + 1) as f64).abs() < 1e-10, "mismatch at {i}");
        }
    }

    #[test]
    fn test_vdsqrt_strided() {
        let full: Array1<f64> = (0..16).map(|i| (i + 1) as f64).collect();
        let strided = full.slice(s![..;2]); // 8 elements
        let mut out = Array1::<f64>::zeros(8);
        vdsqrt(strided, out.view_mut());
        for i in 0..8 {
            let expected = ((2 * i + 1) as f64).sqrt();
            assert!((out[i] - expected).abs() < 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // vsabs
    // -----------------------------------------------------------------------

    #[test]
    fn test_vsabs_contig() {
        let x = arr1(&[-1.0f32, 2.0, -3.0]);
        let mut out = Array1::<f32>::zeros(3);
        vsabs(x.view(), out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vsabs_strided_2d() {
        let m = arr2(&[[-1.0f32, 2.0], [3.0, -4.0]]);
        let mut out = Array2::<f32>::zeros((2, 2));
        vsabs(m.view(), out.view_mut());
        for ((r, c), &v) in out.indexed_iter() {
            assert_eq!(v, m[[r, c]].abs());
        }
    }

    // -----------------------------------------------------------------------
    // vdabs
    // -----------------------------------------------------------------------

    #[test]
    fn test_vdabs_simd() {
        let x: Array1<f64> = (-5..5).map(|i| i as f64).collect();
        let mut out = Array1::<f64>::zeros(10);
        vdabs(x.view(), out.view_mut());
        for i in 0..10 {
            assert_eq!(out[i], (x[i]).abs());
        }
    }

    #[test]
    fn test_vdabs_strided() {
        let full: Array1<f64> = (-10..10).map(|i| i as f64).collect();
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f64>::zeros(10);
        vdabs(strided.view(), out.view_mut());
        for i in 0..10 {
            assert_eq!(out[i], strided[i].abs());
        }
    }

    // -----------------------------------------------------------------------
    // vsadd
    // -----------------------------------------------------------------------

    #[test]
    fn test_vsadd_contig() {
        let a = arr1(&[1.0f32, 2.0, 3.0]);
        let b = arr1(&[4.0f32, 5.0, 6.0]);
        let mut out = Array1::<f32>::zeros(3);
        vsadd(a.view(), b.view(), out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vsadd_strided() {
        let a_full = arr1(&[1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0]);
        let b_full = arr1(&[4.0f32, 0.0, 5.0, 0.0, 6.0, 0.0]);
        let a = a_full.slice(s![..;2]);
        let b = b_full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(3);
        vsadd(a, b, out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vsadd_2d() {
        let a = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
        let b = arr2(&[[10.0f32, 20.0], [30.0, 40.0]]);
        let mut out = Array2::<f32>::zeros((2, 2));
        vsadd(a.view(), b.view(), out.view_mut());
        assert_eq!(out[[0, 0]], 11.0);
        assert_eq!(out[[1, 1]], 44.0);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn test_vsadd_panics_on_shape_mismatch() {
        let a = arr1(&[1.0f32, 2.0]);
        let b = arr1(&[1.0f32, 2.0, 3.0]);
        let mut out = Array1::<f32>::zeros(2);
        vsadd(a.view(), b.view(), out.view_mut());
    }

    // -----------------------------------------------------------------------
    // vsmul
    // -----------------------------------------------------------------------

    #[test]
    fn test_vsmul_contig() {
        let a = arr1(&[1.0f32, 2.0, 3.0]);
        let b = arr1(&[4.0f32, 5.0, 6.0]);
        let mut out = Array1::<f32>::zeros(3);
        vsmul(a.view(), b.view(), out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_vsmul_strided() {
        let a_full = arr1(&[2.0f32, 0.0, 3.0, 0.0]);
        let b_full = arr1(&[5.0f32, 0.0, 7.0, 0.0]);
        let a = a_full.slice(s![..;2]);
        let b = b_full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(2);
        vsmul(a, b, out.view_mut());
        assert_eq!(out[0], 10.0);
        assert_eq!(out[1], 21.0);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn test_vsmul_panics_on_shape_mismatch() {
        let a = arr1(&[1.0f32, 2.0]);
        let b = arr1(&[1.0f32]);
        let mut out = Array1::<f32>::zeros(2);
        vsmul(a.view(), b.view(), out.view_mut());
    }

    // -----------------------------------------------------------------------
    // vsdiv
    // -----------------------------------------------------------------------

    #[test]
    fn test_vsdiv_contig() {
        let a = arr1(&[10.0f32, 20.0, 30.0]);
        let b = arr1(&[2.0f32, 4.0, 5.0]);
        let mut out = Array1::<f32>::zeros(3);
        vsdiv(a.view(), b.view(), out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_vsdiv_strided() {
        let a_full = arr1(&[10.0f32, 0.0, 20.0, 0.0]);
        let b_full = arr1(&[2.0f32, 0.0, 4.0, 0.0]);
        let a = a_full.slice(s![..;2]);
        let b = b_full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(2);
        vsdiv(a, b, out.view_mut());
        assert_eq!(out[0], 5.0);
        assert_eq!(out[1], 5.0);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn test_vsdiv_panics_on_shape_mismatch() {
        let a = arr1(&[1.0f32, 2.0, 3.0]);
        let b = arr1(&[1.0f32, 2.0]);
        let mut out = Array1::<f32>::zeros(3);
        vsdiv(a.view(), b.view(), out.view_mut());
    }

    // -----------------------------------------------------------------------
    // vssin / vscos
    // -----------------------------------------------------------------------

    #[test]
    fn test_vssin_contig() {
        let x = arr1(&[0.0f32, core::f32::consts::FRAC_PI_2]);
        let mut out = Array1::<f32>::zeros(2);
        vssin(x.view(), out.view_mut());
        assert!(out[0].abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vssin_simd_batch() {
        let x: Array1<f32> = (0..32).map(|i| (i as f32) * 0.1).collect();
        let mut out = Array1::<f32>::zeros(32);
        vssin(x.view(), out.view_mut());
        for i in 0..32 {
            assert!((out[i] - x[i].sin()).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn test_vssin_strided() {
        let full: Array1<f32> = (0..64).map(|i| (i as f32) * 0.05).collect();
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(32);
        vssin(strided.view(), out.view_mut());
        for i in 0..32 {
            assert!((out[i] - strided[i].sin()).abs() < 1e-5);
        }
    }

    #[test]
    fn test_vscos_contig() {
        let x = arr1(&[0.0f32, core::f32::consts::PI]);
        let mut out = Array1::<f32>::zeros(2);
        vscos(x.view(), out.view_mut());
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] + 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_vscos_strided() {
        let full: Array1<f32> = (0..40).map(|i| (i as f32) * 0.1).collect();
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(20);
        vscos(strided.view(), out.view_mut());
        for i in 0..20 {
            assert!((out[i] - strided[i].cos()).abs() < 1e-5);
        }
    }

    // -----------------------------------------------------------------------
    // vspow
    // -----------------------------------------------------------------------

    #[test]
    fn test_vspow_contig() {
        let a = arr1(&[2.0f32, 3.0, 4.0]);
        let b = arr1(&[3.0f32, 2.0, 0.5]);
        let mut out = Array1::<f32>::zeros(3);
        vspow(a.view(), b.view(), out.view_mut());
        assert!((out[0] - 8.0).abs() < 1e-3, "2^3 = {}", out[0]);
        assert!((out[1] - 9.0).abs() < 1e-3, "3^2 = {}", out[1]);
        assert!((out[2] - 2.0).abs() < 1e-3, "4^0.5 = {}", out[2]);
    }

    #[test]
    fn test_vspow_strided() {
        let a_full = arr1(&[2.0f32, 0.0, 3.0, 0.0]);
        let b_full = arr1(&[3.0f32, 0.0, 2.0, 0.0]);
        let a = a_full.slice(s![..;2]);
        let b = b_full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(2);
        vspow(a, b, out.view_mut());
        assert!((out[0] - 8.0).abs() < 1e-3);
        assert!((out[1] - 9.0).abs() < 1e-3);
    }

    #[test]
    #[should_panic(expected = "shape mismatch")]
    fn test_vspow_panics_on_shape_mismatch() {
        let a = arr1(&[1.0f32, 2.0]);
        let b = arr1(&[1.0f32, 2.0, 3.0]);
        let mut out = Array1::<f32>::zeros(2);
        vspow(a.view(), b.view(), out.view_mut());
    }

    // -----------------------------------------------------------------------
    // vstanh
    // -----------------------------------------------------------------------

    #[test]
    fn test_vstanh_contig() {
        let x = arr1(&[0.0f32, 1.0, -1.0]);
        let mut out = Array1::<f32>::zeros(3);
        vstanh(x.view(), out.view_mut());
        assert!(out[0].abs() < 1e-5);
        assert!((out[1] - 1.0f32.tanh()).abs() < 1e-4);
        assert!((out[2] - (-1.0f32).tanh()).abs() < 1e-4);
    }

    #[test]
    fn test_vstanh_strided() {
        let full: Array1<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(16);
        vstanh(strided.view(), out.view_mut());
        for i in 0..16 {
            assert!((out[i] - strided[i].tanh()).abs() < 1e-4);
        }
    }

    // -----------------------------------------------------------------------
    // vsfloor / vsceil / vsround / vsneg / vstrunc
    // -----------------------------------------------------------------------

    #[test]
    fn test_vsfloor_contig() {
        let x = arr1(&[1.7f32, -1.2, 0.5]);
        let mut out = Array1::<f32>::zeros(3);
        vsfloor(x.view(), out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[1.0, -2.0, 0.0]);
    }

    #[test]
    fn test_vsfloor_strided() {
        let full = arr1(&[1.7f32, 0.0, -1.2, 0.0, 0.5, 0.0]);
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(3);
        vsfloor(strided, out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[1.0, -2.0, 0.0]);
    }

    #[test]
    fn test_vsceil_contig() {
        let x = arr1(&[1.2f32, -1.7, 0.5]);
        let mut out = Array1::<f32>::zeros(3);
        vsceil(x.view(), out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[2.0, -1.0, 1.0]);
    }

    #[test]
    fn test_vsceil_strided() {
        let full = arr1(&[1.2f32, 0.0, -1.7, 0.0, 0.5, 0.0]);
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(3);
        vsceil(strided, out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[2.0, -1.0, 1.0]);
    }

    #[test]
    fn test_vsround_contig() {
        let x = arr1(&[1.4f32, 1.6, -1.4, -1.6]);
        let mut out = Array1::<f32>::zeros(4);
        vsround(x.view(), out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[1.0, 2.0, -1.0, -2.0]);
    }

    #[test]
    fn test_vsround_strided() {
        let full = arr1(&[1.4f32, 0.0, 1.6, 0.0, -1.4, 0.0, -1.6, 0.0]);
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(4);
        vsround(strided, out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[1.0, 2.0, -1.0, -2.0]);
    }

    #[test]
    fn test_vsneg_contig() {
        let x = arr1(&[1.0f32, -2.0, 3.5]);
        let mut out = Array1::<f32>::zeros(3);
        vsneg(x.view(), out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[-1.0, 2.0, -3.5]);
    }

    #[test]
    fn test_vsneg_strided_2d() {
        let m = arr2(&[[1.0f32, -2.0], [-3.0, 4.0]]);
        let mut out = Array2::<f32>::zeros((2, 2));
        vsneg(m.view(), out.view_mut());
        assert_eq!(out[[0, 0]], -1.0);
        assert_eq!(out[[1, 1]], -4.0);
    }

    #[test]
    fn test_vstrunc_contig() {
        let x = arr1(&[1.9f32, -1.9, 0.5, -0.5]);
        let mut out = Array1::<f32>::zeros(4);
        vstrunc(x.view(), out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[1.0, -1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_vstrunc_strided() {
        let full = arr1(&[1.9f32, 0.0, -1.9, 0.0, 0.5, 0.0, -0.5, 0.0]);
        let strided = full.slice(s![..;2]);
        let mut out = Array1::<f32>::zeros(4);
        vstrunc(strided, out.view_mut());
        assert_eq!(out.as_slice().unwrap(), &[1.0, -1.0, 0.0, 0.0]);
    }
}
