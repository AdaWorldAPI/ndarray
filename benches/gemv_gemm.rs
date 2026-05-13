#![allow(clippy::many_single_char_names, clippy::deref_addrof, clippy::unreadable_literal)]

use criterion::{criterion_group, criterion_main, Criterion};
use num_complex::Complex;
use num_traits::{Float, One, Zero};

use ndarray::prelude::*;

use ndarray::linalg::general_mat_mul;
use ndarray::linalg::general_mat_vec_mul;
use ndarray::LinalgScalar;

fn gemv_64_64c(c: &mut Criterion)
{
    let a = Array::zeros((64, 64));
    let (m, n) = a.dim();
    let x = Array::zeros(n);
    let mut y = Array::zeros(m);
    c.bench_function("gemv_64_64c", |b| {
        b.iter(|| {
            general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y);
        });
    });
}

fn gemv_64_64f(c: &mut Criterion)
{
    let a = Array::zeros((64, 64).f());
    let (m, n) = a.dim();
    let x = Array::zeros(n);
    let mut y = Array::zeros(m);
    c.bench_function("gemv_64_64f", |b| {
        b.iter(|| {
            general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y);
        });
    });
}

fn gemv_64_32(c: &mut Criterion)
{
    let a = Array::zeros((64, 32));
    let (m, n) = a.dim();
    let x = Array::zeros(n);
    let mut y = Array::zeros(m);
    c.bench_function("gemv_64_32", |b| {
        b.iter(|| {
            general_mat_vec_mul(1.0, &a, &x, 1.0, &mut y);
        });
    });
}

fn cgemm_100(c: &mut Criterion)
{
    cgemm_bench::<f32>("cgemm_100", 100, c);
}

fn zgemm_100(c: &mut Criterion)
{
    cgemm_bench::<f64>("zgemm_100", 100, c);
}

fn cgemm_bench<A>(name: &str, size: usize, c: &mut Criterion)
where A: LinalgScalar + Float
{
    let (m, k, n) = (size, size, size);
    let a = Array::<Complex<A>, _>::zeros((m, k));

    let x = Array::zeros((k, n));
    let mut y = Array::zeros((m, n));
    c.bench_function(name, |b| {
        b.iter(|| {
            general_mat_mul(Complex::one(), &a, &x, Complex::zero(), &mut y);
        });
    });
}

criterion_group!(benches, gemv_64_64c, gemv_64_64f, gemv_64_32, cgemm_100, zgemm_100);
criterion_main!(benches);
