use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;

fn push_reserve(c: &mut Criterion) {
    let ones: Array<f32, _> = array![1f32];
    c.bench_function("push_reserve", |b| {
        b.iter(|| {
            let mut a: Array<f32, Ix1> = array![];
            a.reserve(Axis(0), 100).unwrap();
            for _ in 0..100 {
                a.append(Axis(0), ones.view()).unwrap();
            }
        });
    });
}

fn push_no_reserve(c: &mut Criterion) {
    let ones: Array<f32, _> = array![1f32];
    c.bench_function("push_no_reserve", |b| {
        b.iter(|| {
            let mut a: Array<f32, Ix1> = array![];
            for _ in 0..100 {
                a.append(Axis(0), ones.view()).unwrap();
            }
        });
    });
}

criterion_group!(benches, push_reserve, push_no_reserve);
criterion_main!(benches);
