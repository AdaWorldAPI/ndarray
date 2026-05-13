use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;
use ndarray::NdProducer;

fn chunk2x2_iter_sum(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((256, 256));
    let chunksz = (2, 2);
    let mut sum = Array::zeros(a.exact_chunks(chunksz).raw_dim());
    c.bench_function("chunk2x2_iter_sum", |b| {
        b.iter(|| {
            azip!((a in a.exact_chunks(chunksz), sum in &mut sum) {
                *sum = a.iter().sum::<f32>();
            });
        });
    });
}

fn chunk2x2_sum(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((256, 256));
    let chunksz = (2, 2);
    let mut sum = Array::zeros(a.exact_chunks(chunksz).raw_dim());
    c.bench_function("chunk2x2_sum", |b| {
        b.iter(|| {
            azip!((a in a.exact_chunks(chunksz), sum in &mut sum) {
                *sum = a.sum();
            });
        });
    });
}

fn chunk2x2_sum_get1(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((256, 256));
    let chunksz = (2, 2);
    let mut sum = Array::<f32, _>::zeros(a.exact_chunks(chunksz).raw_dim());
    c.bench_function("chunk2x2_sum_get1", |b| {
        b.iter(|| {
            let (m, n) = a.dim();
            for i in 0..m {
                for j in 0..n {
                    sum[[i / 2, j / 2]] += a[[i, j]];
                }
            }
        });
    });
}

fn chunk2x2_sum_uget1(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((256, 256));
    let chunksz = (2, 2);
    let mut sum = Array::<f32, _>::zeros(a.exact_chunks(chunksz).raw_dim());
    c.bench_function("chunk2x2_sum_uget1", |b| {
        b.iter(|| {
            let (m, n) = a.dim();
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        *sum.uget_mut([i / 2, j / 2]) += *a.uget([i, j]);
                    }
                }
            }
        });
    });
}

#[allow(clippy::identity_op)]
fn chunk2x2_sum_get2(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((256, 256));
    let chunksz = (2, 2);
    let mut sum = Array::<f32, _>::zeros(a.exact_chunks(chunksz).raw_dim());
    c.bench_function("chunk2x2_sum_get2", |b| {
        b.iter(|| {
            let (m, n) = sum.dim();
            for i in 0..m {
                for j in 0..n {
                    sum[[i, j]] += a[[i * 2 + 0, j * 2 + 0]];
                    sum[[i, j]] += a[[i * 2 + 0, j * 2 + 1]];
                    sum[[i, j]] += a[[i * 2 + 1, j * 2 + 1]];
                    sum[[i, j]] += a[[i * 2 + 1, j * 2 + 0]];
                }
            }
        });
    });
}

criterion_group!(
    benches,
    chunk2x2_iter_sum,
    chunk2x2_sum,
    chunk2x2_sum_get1,
    chunk2x2_sum_uget1,
    chunk2x2_sum_get2
);
criterion_main!(benches);
