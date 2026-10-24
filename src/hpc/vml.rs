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

/// 16-lane `tanh`, computed on `a = |x|` and negated where `x < 0`, so the
/// function is exactly odd. Lanes where `x == 0` return `x` itself, which keeps
/// `tanh(-0.0) = -0.0`. (Float compares and `select` only: the NEON `U32x16`
/// has no `&`/`|`, so the sign is not restored by bit masking.)
///
/// Two branches, chosen per lane:
/// - `|x| < 0.625`: `a + a·z·P(z)` with `z = a²` (the Cephes `tanhf`
///   minimax polynomial). It has no cancellation, and for subnormal or tiny
///   `a` the correction vanishes and the result is `a` itself.
/// - otherwise: `1 − 2/(exp(2a) + 1)`. `exp` overflows to `+inf` for large
///   `a`, which gives exactly `1.0`; a NaN input stays NaN.
///
/// The earlier `2·sigmoid(2x) − 1` cancelled for small `|x|`: below about
/// `1e-7` the sigmoid rounds to `0.5` and the result was `0`.
#[inline(always)]
fn tanh_f32x16(v: F32x16) -> F32x16 {
    let zero = F32x16::splat(0.0);
    let a = v.abs();
    let z = a * a;
    let p = F32x16::splat(-5.704_988_727_45e-3)
        .mul_add(z, F32x16::splat(2.063_908_879_54e-2))
        .mul_add(z, F32x16::splat(-5.373_971_555_31e-2))
        .mul_add(z, F32x16::splat(1.333_144_220_36e-1))
        .mul_add(z, F32x16::splat(-3.333_328_194_22e-1));
    let small = (p * z).mul_add(a, a);
    let e = simd_exp_f32(a + a);
    let large = F32x16::splat(1.0) - F32x16::splat(2.0) / (e + F32x16::splat(1.0));
    let r = a.simd_lt(F32x16::splat(0.625)).select(small, large);
    let r = v.simd_lt(zero).select(-r, r);
    v.simd_eq(zero).select(v, r)
}

fn vstanh_slice(x: &[f32], out: &mut [f32]) {
    let n = x.len().min(out.len());
    let chunks = n / 16;
    for i in 0..chunks {
        let off = i * 16;
        tanh_f32x16(F32x16::from_slice(&x[off..off + 16])).copy_to_slice(&mut out[off..off + 16]);
    }
    // The tail goes through the same 16-lane kernel, so a value gives the same
    // bits wherever it sits in the array.
    let rest = n - chunks * 16;
    if rest > 0 {
        let mut buf = [0.0f32; 16];
        buf[..rest].copy_from_slice(&x[chunks * 16..n]);
        let r = tanh_f32x16(F32x16::from_slice(&buf));
        r.copy_to_slice(&mut buf);
        out[chunks * 16..n].copy_from_slice(&buf[..rest]);
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
        // ceil = -floor(-x). `Neg` flips the sign bit; `0.0 - v` would turn
        // ceil(-0.5) and ceil(-0.0) into +0.0.
        (-(-v).floor()).copy_to_slice(&mut out[off..off + 16]);
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
    for i in 0..chunks {
        let off = i * 16;
        let v = F32x16::from_slice(&x[off..off + 16]);
        // `Neg` flips the sign bit, so -(+0.0) is -0.0; `0.0 - v` gives +0.0.
        (-v).copy_to_slice(&mut out[off..off + 16]);
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
        // Negative lanes take -floor(-x), the rest floor(x). -0.0 is not
        // < 0, so it takes floor(-0.0) = -0.0 and keeps its sign.
        let negative = v.simd_lt(F32x16::splat(0.0));
        negative
            .select(-(-v).floor(), v.floor())
            .copy_to_slice(&mut out[off..off + 16]);
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

/// Element-wise `tanh`, within 2 ULP of the correctly rounded result over
/// every finite `f32`. `tanh(±0) = ±0`, subnormals map to themselves,
/// `tanh(±inf) = ±1` and NaN stays NaN.
///
/// Every element, contiguous or strided, goes through the same 16-lane
/// kernel, so a value's result does not depend on its position or layout.
///
/// # Panics
/// Panics if `x` and `out` shapes differ.
pub fn vstanh<D: Dimension>(x: ArrayView<'_, f32, D>, mut out: ArrayViewMut<'_, f32, D>) {
    assert_eq!(x.shape(), out.shape(), "vstanh: x/out shape mismatch");
    if dispatch_unary_contig(x.view(), &mut out, vstanh_slice) {
        return;
    }
    Zip::from(&mut out)
        .and(&x)
        .for_each(|o, &v| *o = tanh_f32x16(F32x16::splat(v)).to_array()[0]);
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
    use crate::{arr1, arr2, s, Array1, Array2, ArrayView1, ArrayViewMut1};

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

    /// Distance in units in the last place between two `f32`, with `-0.0`
    /// and `+0.0` one apart so a lost sign of zero counts as an error.
    fn ulp_distance(a: f32, b: f32) -> u32 {
        if a.is_nan() || b.is_nan() {
            return if a.is_nan() && b.is_nan() { 0 } else { u32::MAX };
        }
        let key = |x: f32| {
            let i = x.to_bits() as i32;
            if i < 0 {
                -((i & i32::MAX) as i64) - 1
            } else {
                i as i64
            }
        };
        (key(a) - key(b)).unsigned_abs().min(u32::MAX as u64) as u32
    }

    /// Correctly rounded reference: `tanh` in `f64`, rounded once to `f32`.
    fn tanh_ref(x: f32) -> f32 {
        (x as f64).tanh() as f32
    }

    fn vstanh_vec(xs: &[f32]) -> Vec<f32> {
        let x = Array1::from(xs.to_vec());
        let mut out = Array1::<f32>::zeros(xs.len());
        vstanh(x.view(), out.view_mut());
        out.to_vec()
    }

    #[test]
    fn test_vstanh_within_2_ulp_across_magnitudes() {
        // Log-spaced magnitudes in each band, both signs. The length is not a
        // multiple of 16, so the last values take the tail path.
        let bands = [
            (1.0e-38f32, 1.0e-10f32),
            (1.0e-10, 1.0e-4),
            (1.0e-4, 0.1),
            (0.1, 0.625),
            (0.625, 1.0),
            (1.0, 4.0),
            (4.0, 10.0),
            (10.0, 100.0),
        ];
        for (lo, hi) in bands {
            let n = 4099;
            let mut xs = Vec::with_capacity(2 * n + 1);
            for i in 0..n {
                let t = i as f64 / (n - 1) as f64;
                let v = ((lo as f64).ln() + t * ((hi as f64).ln() - (lo as f64).ln())).exp() as f32;
                xs.push(v);
                xs.push(-v);
            }
            xs.push(hi);
            assert_ne!(xs.len() % 16, 0, "fixture must reach the tail path");
            let ys = vstanh_vec(&xs);
            let (worst, at) = xs
                .iter()
                .zip(&ys)
                .map(|(&x, &y)| (ulp_distance(y, tanh_ref(x)), x))
                .max_by_key(|&(u, _)| u)
                .unwrap();
            assert!(worst <= 2, "band [{lo:e}, {hi:e}]: {worst} ulp at x = {at:e}");
        }
    }

    #[test]
    fn test_vstanh_special_values_in_chunk_and_tail() {
        let sub = f32::from_bits(1);
        let cases: [(f32, f32); 9] = [
            (0.0, 0.0),
            (-0.0, -0.0),
            (sub, sub),
            (-sub, -sub),
            (f32::MIN_POSITIVE, f32::MIN_POSITIVE),
            (-f32::MIN_POSITIVE, -f32::MIN_POSITIVE),
            (f32::INFINITY, 1.0),
            (f32::NEG_INFINITY, -1.0),
            (1.0e-8, 1.0e-8),
        ];
        // 16 values fill one SIMD chunk; the same 9 again land in the tail.
        let mut xs: Vec<f32> = cases.iter().map(|c| c.0).collect();
        xs.resize(16, 0.5);
        xs.extend(cases.iter().map(|c| c.0));
        let ys = vstanh_vec(&xs);
        for (k, &(x, want)) in cases.iter().enumerate() {
            for (path, y) in [("chunk", ys[k]), ("tail", ys[16 + k])] {
                assert_eq!(y.to_bits(), want.to_bits(), "{path}: vstanh({x:e}) = {y:e}, want {want:e}");
            }
        }
        assert!(vstanh_vec(&[f32::NAN; 17]).iter().all(|v| v.is_nan()));
    }

    #[test]
    fn test_vstanh_tail_equals_chunk() {
        // The same input gives the same bits in a full chunk and in the tail,
        // so a result never depends on where in the array it sits.
        let vals: Vec<f32> = (0..15).map(|i| (i as f32 - 7.0) * 0.37 + 1.0e-3).collect();
        let mut xs = vals.clone();
        xs.push(0.0);
        xs.extend(&vals);
        let ys = vstanh_vec(&xs);
        for k in 0..vals.len() {
            assert_eq!(ys[k].to_bits(), ys[16 + k].to_bits(), "x = {}", vals[k]);
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

    // -----------------------------------------------------------------------
    // IEEE edge cases through the 16-lane SIMD body, not just the scalar tail.
    // Every length from 0 to 40 puts each value both in a SIMD chunk and in
    // the tail, so a body that disagrees with its own tail cannot pass.
    // -----------------------------------------------------------------------

    const EDGES: [f32; 12] = [
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        -0.25,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::NAN,
    ];

    /// Checks `kernel` against `scalar` bit for bit (NaN only as NaN) for
    /// every length 0..=40, cycling through `EDGES`.
    fn assert_bitwise_like_scalar(
        name: &str, kernel: fn(ArrayView1<'_, f32>, ArrayViewMut1<'_, f32>), scalar: fn(f32) -> f32,
    ) {
        for n in 0..=40usize {
            let x = Array1::from_iter((0..n).map(|i| EDGES[i % EDGES.len()]));
            let mut out = Array1::<f32>::zeros(n);
            kernel(x.view(), out.view_mut());
            for i in 0..n {
                let (got, want) = (out[i], scalar(x[i]));
                if want.is_nan() {
                    assert!(got.is_nan(), "{name}({}) n={n} i={i}: got {got}", x[i]);
                } else {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{name}({}) n={n} i={i}: got {got:?}, want {want:?}",
                        x[i]
                    );
                }
            }
        }
    }

    #[test]
    fn vsneg_flips_the_sign_of_zero_in_the_simd_body() {
        assert_bitwise_like_scalar("vsneg", vsneg, |v| -v);
    }

    #[test]
    fn vsceil_keeps_negative_zero_in_the_simd_body() {
        assert_bitwise_like_scalar("vsceil", vsceil, f32::ceil);
    }

    #[test]
    fn vstrunc_keeps_negative_zero_in_the_simd_body() {
        assert_bitwise_like_scalar("vstrunc", vstrunc, f32::trunc);
    }

    /// FAILS IF: the SIMD body breaks ties differently from the scalar tail
    /// (`round_ties_even`). On the AVX2, scalar and nightly arms
    /// `F32x16::round` used `f32::round`, which rounds ties away from zero.
    #[test]
    fn vsround_breaks_ties_to_even_in_the_simd_body() {
        assert_bitwise_like_scalar("vsround", vsround, f32::round_ties_even);
    }

    #[test]
    fn vsexp_maps_infinities_like_the_scalar_tail() {
        for n in [1usize, 16, 17, 32] {
            for (v, want) in [(f32::INFINITY, f32::INFINITY), (f32::NEG_INFINITY, 0.0)] {
                let x = Array1::from_elem(n, v);
                let mut out = Array1::<f32>::zeros(n);
                vsexp(x.view(), out.view_mut());
                for i in 0..n {
                    assert_eq!(out[i].to_bits(), want.to_bits(), "vsexp({v}) n={n} i={i}: got {:?}", out[i]);
                }
            }
        }
    }

    /// FAILS IF: any finite input in [-110, 88.72] is off by more than the
    /// polynomial's own error. Measured against f64 `exp`. Results below the
    /// normal range may also be off by one subnormal step.
    #[test]
    fn vsexp_is_accurate_up_to_the_overflow_and_down_into_subnormals() {
        let xs: Vec<f32> = (0..=19872)
            .map(|i| -110.0 + i as f32 * 0.01)
            .filter(|&x| x <= 88.72)
            .collect();
        let x = Array1::from(xs);
        let mut out = Array1::<f32>::zeros(x.len());
        vsexp(x.view(), out.view_mut());
        for (i, &v) in x.iter().enumerate() {
            let want = f64::from(v).exp();
            let got = f64::from(out[i]);
            if want >= f64::from(f32::MIN_POSITIVE) {
                assert!(((got - want) / want).abs() < 4e-6, "vsexp({v}): got {got:e}, want {want:e}");
            } else {
                // the polynomial's relative error plus one subnormal step
                let step = f64::from(f32::from_bits(1));
                assert!((got - want).abs() <= 4e-6 * want + step, "vsexp({v}): got {got:e}, want {want:e}");
            }
        }
    }
}
