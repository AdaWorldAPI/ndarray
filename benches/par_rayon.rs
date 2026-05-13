#![cfg(feature = "rayon")]

use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Zip;

const EXP_N: usize = 256;
const ADDN: usize = 512;

fn set_threads() {
    // Consider setting a fixed number of threads here, for example to avoid
    // oversubscribing on hyperthreaded cores.
    // let n = 4;
    // let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
}

fn map_exp_regular(c: &mut Criterion) {
    let mut a = Array2::<f64>::zeros((EXP_N, EXP_N));
    a.swap_axes(0, 1);
    c.bench_function("map_exp_regular", |b| {
        b.iter(|| {
            a.mapv_inplace(|x| x.exp());
        });
    });
}

fn rayon_exp_regular(c: &mut Criterion) {
    set_threads();
    let mut a = Array2::<f64>::zeros((EXP_N, EXP_N));
    a.swap_axes(0, 1);
    c.bench_function("rayon_exp_regular", |b| {
        b.iter(|| {
            a.view_mut().into_par_iter().for_each(|x| *x = x.exp());
        });
    });
}

const FASTEXP: usize = EXP_N;

#[inline]
fn fastexp(x: f64) -> f64 {
    let x = 1. + x / 1024.;
    x.powi(1024)
}

fn map_fastexp_regular(c: &mut Criterion) {
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    c.bench_function("map_fastexp_regular", |b| b.iter(|| a.mapv_inplace(fastexp)));
}

fn rayon_fastexp_regular(c: &mut Criterion) {
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    c.bench_function("rayon_fastexp_regular", |b| {
        b.iter(|| {
            a.view_mut().into_par_iter().for_each(|x| *x = fastexp(*x));
        });
    });
}

fn map_fastexp_cut(c: &mut Criterion) {
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    let mut a = a.slice_mut(s![.., ..-1]);
    c.bench_function("map_fastexp_cut", |b| b.iter(|| a.mapv_inplace(fastexp)));
}

fn rayon_fastexp_cut(c: &mut Criterion) {
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    let mut a = a.slice_mut(s![.., ..-1]);
    c.bench_function("rayon_fastexp_cut", |b| {
        b.iter(|| {
            a.view_mut().into_par_iter().for_each(|x| *x = fastexp(*x));
        });
    });
}

fn map_fastexp_by_axis(c: &mut Criterion) {
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    c.bench_function("map_fastexp_by_axis", |b| {
        b.iter(|| {
            for mut sheet in a.axis_iter_mut(Axis(0)) {
                sheet.mapv_inplace(fastexp)
            }
        });
    });
}

fn rayon_fastexp_by_axis(c: &mut Criterion) {
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    c.bench_function("rayon_fastexp_by_axis", |b| {
        b.iter(|| {
            a.axis_iter_mut(Axis(0))
                .into_par_iter()
                .for_each(|mut sheet| sheet.mapv_inplace(fastexp));
        });
    });
}

fn rayon_fastexp_zip(c: &mut Criterion) {
    set_threads();
    let mut a = Array2::<f64>::zeros((FASTEXP, FASTEXP));
    c.bench_function("rayon_fastexp_zip", |b| {
        b.iter(|| {
            Zip::from(&mut a)
                .into_par_iter()
                .for_each(|(elt,)| *elt = fastexp(*elt));
        });
    });
}

fn add(c: &mut Criterion) {
    let mut a = Array2::<f64>::zeros((ADDN, ADDN));
    let b_arr = Array2::<f64>::zeros((ADDN, ADDN));
    let c_arr = Array2::<f64>::zeros((ADDN, ADDN));
    let d = Array2::<f64>::zeros((ADDN, ADDN));
    c.bench_function("add", |bench| {
        bench.iter(|| {
            azip!((a in &mut a, &b in &b_arr, &c in &c_arr, &d in &d) {
                *a += b.exp() + c.exp() + d.exp();
            });
        });
    });
}

fn rayon_add(c: &mut Criterion) {
    set_threads();
    let mut a = Array2::<f64>::zeros((ADDN, ADDN));
    let b_arr = Array2::<f64>::zeros((ADDN, ADDN));
    let c_arr = Array2::<f64>::zeros((ADDN, ADDN));
    let d = Array2::<f64>::zeros((ADDN, ADDN));
    c.bench_function("rayon_add", |bench| {
        bench.iter(|| {
            par_azip!((a in &mut a, b in &b_arr, c in &c_arr, d in &d) {
                *a += b.exp() + c.exp() + d.exp();
            });
        });
    });
}

const COLL_STRING_N: usize = 64;
const COLL_F64_N: usize = 128;

fn vec_string_collect(c: &mut Criterion) {
    let v = vec![""; COLL_STRING_N * COLL_STRING_N];
    c.bench_function("vec_string_collect", |b| {
        b.iter(|| v.iter().map(|s| s.to_owned()).collect::<Vec<_>>());
    });
}

fn array_string_collect(c: &mut Criterion) {
    let v = Array::from_elem((COLL_STRING_N, COLL_STRING_N), "");
    c.bench_function("array_string_collect", |b| {
        b.iter(|| Zip::from(&v).par_map_collect(|s| s.to_owned()));
    });
}

fn vec_f64_collect(c: &mut Criterion) {
    let v = vec![1.; COLL_F64_N * COLL_F64_N];
    c.bench_function("vec_f64_collect", |b| {
        b.iter(|| v.iter().map(|s| s + 1.).collect::<Vec<_>>());
    });
}

fn array_f64_collect(c: &mut Criterion) {
    let v = Array::from_elem((COLL_F64_N, COLL_F64_N), 1.);
    c.bench_function("array_f64_collect", |b| {
        b.iter(|| Zip::from(&v).par_map_collect(|s| s + 1.));
    });
}

criterion_group!(
    benches, map_exp_regular, rayon_exp_regular, map_fastexp_regular, rayon_fastexp_regular, map_fastexp_cut,
    rayon_fastexp_cut, map_fastexp_by_axis, rayon_fastexp_by_axis, rayon_fastexp_zip, add, rayon_add,
    vec_string_collect, array_string_collect, vec_f64_collect, array_f64_collect
);
criterion_main!(benches);
