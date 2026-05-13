use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;
use ndarray::Order;

fn to_shape2_1(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    c.bench_function("to_shape2_1", |b| {
        b.iter(|| view.to_shape(4 * 5).unwrap());
    });
}

fn to_shape2_2_same(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    c.bench_function("to_shape2_2_same", |b| {
        b.iter(|| view.to_shape((4, 5)).unwrap());
    });
}

fn to_shape2_2_flip(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    c.bench_function("to_shape2_2_flip", |b| {
        b.iter(|| view.to_shape((5, 4)).unwrap());
    });
}

fn to_shape2_3(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((4, 5));
    let view = a.view();
    c.bench_function("to_shape2_3", |b| {
        b.iter(|| view.to_shape((2, 5, 2)).unwrap());
    });
}

fn to_shape3_1(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    c.bench_function("to_shape3_1", |b| {
        b.iter(|| view.to_shape(3 * 4 * 5).unwrap());
    });
}

fn to_shape3_2_order(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    c.bench_function("to_shape3_2_order", |b| {
        b.iter(|| view.to_shape((12, 5)).unwrap());
    });
}

fn to_shape3_2_outoforder(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    c.bench_function("to_shape3_2_outoforder", |b| {
        b.iter(|| view.to_shape((4, 15)).unwrap());
    });
}

fn to_shape3_3c(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    c.bench_function("to_shape3_3c", |b| {
        b.iter(|| view.to_shape((3, 4, 5)).unwrap());
    });
}

fn to_shape3_3f(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((3, 4, 5).f());
    let view = a.view();
    c.bench_function("to_shape3_3f", |b| {
        b.iter(|| view.to_shape(((3, 4, 5), Order::F)).unwrap());
    });
}

fn to_shape3_4c(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((3, 4, 5));
    let view = a.view();
    c.bench_function("to_shape3_4c", |b| {
        b.iter(|| view.to_shape(((2, 3, 2, 5), Order::C)).unwrap());
    });
}

fn to_shape3_4f(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((3, 4, 5).f());
    let view = a.view();
    c.bench_function("to_shape3_4f", |b| {
        b.iter(|| view.to_shape(((2, 3, 2, 5), Order::F)).unwrap());
    });
}

criterion_group!(
    benches,
    to_shape2_1,
    to_shape2_2_same,
    to_shape2_2_flip,
    to_shape2_3,
    to_shape3_1,
    to_shape3_2_order,
    to_shape3_2_outoforder,
    to_shape3_3c,
    to_shape3_3f,
    to_shape3_4c,
    to_shape3_4f
);
criterion_main!(benches);
