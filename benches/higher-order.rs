#![allow(clippy::many_single_char_names, clippy::deref_addrof, clippy::unreadable_literal)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;

const N: usize = 1024;
const X: usize = 64;
const Y: usize = 16;

#[cfg(feature = "std")]
fn map_regular(c: &mut Criterion)
{
    let a = Array::linspace(0.0..=127.0, N)
        .into_shape_with_order((X, Y))
        .unwrap();
    c.bench_function("map_regular", |b| {
        b.iter(|| a.map(|&x| 2. * x));
    });
}

pub fn double_array(mut a: ArrayViewMut2<'_, f64>)
{
    a *= 2.0;
}

#[cfg(feature = "std")]
fn map_stride_double_f64(c: &mut Criterion)
{
    let mut a = Array::linspace(0.0..=127.0, N * 2)
        .into_shape_with_order([X, Y * 2])
        .unwrap();
    let mut av = a.slice_mut(s![.., ..;2]);
    c.bench_function("map_stride_double_f64", |b| {
        b.iter(|| {
            double_array(av.view_mut());
        });
    });
}

#[cfg(feature = "std")]
fn map_stride_f64(c: &mut Criterion)
{
    let a = Array::linspace(0.0..=127.0, N * 2)
        .into_shape_with_order([X, Y * 2])
        .unwrap();
    let av = a.slice(s![.., ..;2]);
    c.bench_function("map_stride_f64", |b| {
        b.iter(|| av.map(|&x| 2. * x));
    });
}

#[cfg(feature = "std")]
fn map_stride_u32(c: &mut Criterion)
{
    let a = Array::linspace(0.0..=127.0, N * 2)
        .into_shape_with_order([X, Y * 2])
        .unwrap();
    let b = a.mapv(|x| x as u32);
    let av = b.slice(s![.., ..;2]);
    c.bench_function("map_stride_u32", |b| {
        b.iter(|| av.map(|&x| 2 * x));
    });
}

#[cfg(feature = "std")]
fn fold_axis(c: &mut Criterion)
{
    let a = Array::linspace(0.0..=127.0, N * 2)
        .into_shape_with_order([X, Y * 2])
        .unwrap();
    c.bench_function("fold_axis", |b| {
        b.iter(|| a.fold_axis(Axis(0), 0., |&acc, &elt| acc + elt));
    });
}

const MA: usize = 64;
const MASZ: usize = MA * MA;

fn map_axis_0(c: &mut Criterion)
{
    let a = Array::from_iter(0..MASZ as i32)
        .into_shape_with_order([MA, MA])
        .unwrap();
    c.bench_function("map_axis_0", |b| {
        b.iter(|| a.map_axis(Axis(0), black_box));
    });
}

fn map_axis_1(c: &mut Criterion)
{
    let a = Array::from_iter(0..MASZ as i32)
        .into_shape_with_order([MA, MA])
        .unwrap();
    c.bench_function("map_axis_1", |b| {
        b.iter(|| a.map_axis(Axis(1), black_box));
    });
}

#[cfg(feature = "std")]
criterion_group!(
    benches,
    map_regular,
    map_stride_double_f64,
    map_stride_f64,
    map_stride_u32,
    fold_axis,
    map_axis_0,
    map_axis_1
);
#[cfg(not(feature = "std"))]
criterion_group!(benches, map_axis_0, map_axis_1);
criterion_main!(benches);
