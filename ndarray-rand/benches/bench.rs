use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::Array;
use ndarray_rand::RandomExt;
use rand_distr::Normal;
use rand_distr::Uniform;

fn uniform_f32(c: &mut Criterion)
{
    let m = 100;
    c.bench_function("uniform_f32", |b| {
        b.iter(|| Array::random((m, m), Uniform::new(-1f32, 1.).unwrap()));
    });
}

fn norm_f32(c: &mut Criterion)
{
    let m = 100;
    c.bench_function("norm_f32", |b| {
        b.iter(|| Array::random((m, m), Normal::new(0f32, 1.).unwrap()));
    });
}

fn norm_f64(c: &mut Criterion)
{
    let m = 100;
    c.bench_function("norm_f64", |b| {
        b.iter(|| Array::random((m, m), Normal::new(0f64, 1.).unwrap()));
    });
}

criterion_group!(benches, uniform_f32, norm_f32, norm_f64);
criterion_main!(benches);
