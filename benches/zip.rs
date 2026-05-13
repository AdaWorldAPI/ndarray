use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::s;
use ndarray::IntoNdProducer;
use ndarray::{Array3, ShapeBuilder, Zip};

pub fn zip_copy<'a, A, P, Q>(data: P, out: Q)
where
    P: IntoNdProducer<Item = &'a A>,
    Q: IntoNdProducer<Item = &'a mut A, Dim = P::Dim>,
    A: Copy + 'a,
{
    Zip::from(data).and(out).for_each(|&i, o| {
        *o = i;
    });
}

pub fn zip_copy_split<'a, A, P, Q>(data: P, out: Q)
where
    P: IntoNdProducer<Item = &'a A>,
    Q: IntoNdProducer<Item = &'a mut A, Dim = P::Dim>,
    A: Copy + 'a,
{
    let z = Zip::from(data).and(out);
    let (z1, z2) = z.split();
    let (z11, z12) = z1.split();
    let (z21, z22) = z2.split();
    let f = |&i: &A, o: &mut A| *o = i;
    z11.for_each(f);
    z12.for_each(f);
    z21.for_each(f);
    z22.for_each(f);
}

pub fn zip_indexed(data: &Array3<f32>, out: &mut Array3<f32>) {
    Zip::indexed(data).and(out).for_each(|idx, &i, o| {
        let _ = black_box(idx);
        *o = i;
    });
}

// array size in benchmarks
const SZ3: (usize, usize, usize) = (100, 110, 100);

fn zip_cc(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim());
    c.bench_function("zip_cc", |b| b.iter(|| zip_copy(&data, &mut out)));
}

fn zip_cf(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim().f());
    c.bench_function("zip_cf", |b| b.iter(|| zip_copy(&data, &mut out)));
}

fn zip_fc(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim());
    c.bench_function("zip_fc", |b| b.iter(|| zip_copy(&data, &mut out)));
}

fn zip_ff(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim().f());
    c.bench_function("zip_ff", |b| b.iter(|| zip_copy(&data, &mut out)));
}

fn zip_indexed_cc(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim());
    c.bench_function("zip_indexed_cc", |b| b.iter(|| zip_indexed(&data, &mut out)));
}

fn zip_indexed_ff(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim().f());
    c.bench_function("zip_indexed_ff", |b| b.iter(|| zip_indexed(&data, &mut out)));
}

fn slice_zip_cc(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim());
    let data = data.slice(s![1.., 1.., 1..]);
    let mut out = out.slice_mut(s![1.., 1.., 1..]);
    c.bench_function("slice_zip_cc", |b| b.iter(|| zip_copy(&data, &mut out)));
}

fn slice_zip_ff(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim().f());
    let data = data.slice(s![1.., 1.., 1..]);
    let mut out = out.slice_mut(s![1.., 1.., 1..]);
    c.bench_function("slice_zip_ff", |b| b.iter(|| zip_copy(&data, &mut out)));
}

fn slice_split_zip_cc(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros(SZ3);
    let mut out = Array3::zeros(data.dim());
    let data = data.slice(s![1.., 1.., 1..]);
    let mut out = out.slice_mut(s![1.., 1.., 1..]);
    c.bench_function("slice_split_zip_cc", |b| b.iter(|| zip_copy_split(&data, &mut out)));
}

fn slice_split_zip_ff(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros(SZ3.f());
    let mut out = Array3::zeros(data.dim().f());
    let data = data.slice(s![1.., 1.., 1..]);
    let mut out = out.slice_mut(s![1.., 1.., 1..]);
    c.bench_function("slice_split_zip_ff", |b| b.iter(|| zip_copy_split(&data, &mut out)));
}

criterion_group!(
    benches, zip_cc, zip_cf, zip_fc, zip_ff, zip_indexed_cc, zip_indexed_ff, slice_zip_cc, slice_zip_ff,
    slice_split_zip_cc, slice_split_zip_ff
);
criterion_main!(benches);
