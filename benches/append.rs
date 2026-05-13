use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;

fn select_axis0(c: &mut Criterion) {
    let a = Array::<f32, _>::zeros((256, 256));
    let selectable = vec![0, 1, 2, 0, 1, 3, 0, 4, 16, 32, 128, 147, 149, 220, 221, 255, 221, 0, 1];
    c.bench_function("select_axis0", |b| {
        b.iter(|| black_box(&a).select(Axis(0), black_box(&selectable)))
    });
}

fn select_axis1(c: &mut Criterion) {
    let a = Array::<f32, _>::zeros((256, 256));
    let selectable = vec![0, 1, 2, 0, 1, 3, 0, 4, 16, 32, 128, 147, 149, 220, 221, 255, 221, 0, 1];
    c.bench_function("select_axis1", |b| {
        b.iter(|| black_box(&a).select(Axis(1), black_box(&selectable)))
    });
}

fn select_1d(c: &mut Criterion) {
    let a = Array::<f32, _>::zeros(1024);
    let mut selectable = (0..a.len()).step_by(17).collect::<Vec<_>>();
    selectable.extend(selectable.clone().iter().rev());
    c.bench_function("select_1d", |b| {
        b.iter(|| black_box(&a).select(Axis(0), black_box(&selectable)))
    });
}

criterion_group!(benches, select_axis0, select_axis1, select_1d);
criterion_main!(benches);
