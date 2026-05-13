#![allow(clippy::many_single_char_names, clippy::deref_addrof, clippy::unreadable_literal)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;

fn default_f64(c: &mut Criterion) {
    c.bench_function("default_f64", |b| {
        b.iter(|| Array::<f64, _>::default(black_box((128, 128))))
    });
}

fn zeros_f64(c: &mut Criterion) {
    c.bench_function("zeros_f64", |b| {
        b.iter(|| Array::<f64, _>::zeros(black_box((128, 128))))
    });
}

#[cfg(feature = "std")]
fn map_regular(c: &mut Criterion) {
    let a = Array::linspace(0.0..=127.0, 128)
        .into_shape_with_order((8, 16))
        .unwrap();
    c.bench_function("map_regular", |b| {
        b.iter(|| black_box(&a).map(|&x| 2. * x))
    });
}

#[cfg(feature = "std")]
fn map_stride(c: &mut Criterion) {
    let a = Array::linspace(0.0..=127.0, 256)
        .into_shape_with_order((8, 32))
        .unwrap();
    let av = a.slice(s![.., ..;2]);
    c.bench_function("map_stride", |b| {
        b.iter(|| black_box(&av).map(|&x| 2. * x))
    });
}

#[cfg(feature = "std")]
criterion_group!(benches, default_f64, zeros_f64, map_regular, map_stride);
#[cfg(not(feature = "std"))]
criterion_group!(benches, default_f64, zeros_f64);
criterion_main!(benches);
