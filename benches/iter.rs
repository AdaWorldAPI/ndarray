// `s![1..-1, 1..-1]` — ndarray's `s!` macro uses Python-style negative
// indexing where `-1` means "from end". Clippy can't see through the macro
// and reads `1..-1` as a plain (empty) Rust range; the actual semantics are
// correct.
#![allow(
    clippy::many_single_char_names, clippy::deref_addrof, clippy::unreadable_literal, clippy::reversed_empty_ranges
)]

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rawpointer::PointerExt;

use ndarray::prelude::*;
use ndarray::Slice;
use ndarray::{FoldWhile, Zip};

fn iter_sum_2d_regular(c: &mut Criterion) {
    let a = Array::<i32, _>::zeros((64, 64));
    c.bench_function("iter_sum_2d_regular", |b| b.iter(|| a.iter().sum::<i32>()));
}

fn iter_sum_2d_cutout(c: &mut Criterion) {
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = av;
    c.bench_function("iter_sum_2d_cutout", |b| b.iter(|| a.iter().sum::<i32>()));
}

fn iter_all_2d_cutout(c: &mut Criterion) {
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = av;
    c.bench_function("iter_all_2d_cutout", |b| b.iter(|| a.iter().all(|&x| x >= 0)));
}

fn iter_sum_2d_transpose(c: &mut Criterion) {
    let a = Array::<i32, _>::zeros((66, 66));
    let a = a.t();
    c.bench_function("iter_sum_2d_transpose", |b| b.iter(|| a.iter().sum::<i32>()));
}

#[cfg(feature = "std")]
fn iter_filter_sum_2d_u32(c: &mut Criterion) {
    let a = Array::linspace(0.0..=1.0, 256)
        .into_shape_with_order((16, 16))
        .unwrap();
    let b = a.mapv(|x| (x * 100.) as u32);
    c.bench_function("iter_filter_sum_2d_u32", |bn| bn.iter(|| b.iter().filter(|&&x| x < 75).sum::<u32>()));
}

#[cfg(feature = "std")]
fn iter_filter_sum_2d_f32(c: &mut Criterion) {
    let a = Array::linspace(0.0..=1.0, 256)
        .into_shape_with_order((16, 16))
        .unwrap();
    let b = a * 100.;
    c.bench_function("iter_filter_sum_2d_f32", |bn| bn.iter(|| b.iter().filter(|&&x| x < 75.).sum::<f32>()));
}

#[cfg(feature = "std")]
fn iter_filter_sum_2d_stride_u32(c: &mut Criterion) {
    let a = Array::linspace(0.0..=1.0, 256)
        .into_shape_with_order((16, 16))
        .unwrap();
    let b = a.mapv(|x| (x * 100.) as u32);
    let b = b.slice(s![.., ..;2]);
    c.bench_function("iter_filter_sum_2d_stride_u32", |bn| bn.iter(|| b.iter().filter(|&&x| x < 75).sum::<u32>()));
}

#[cfg(feature = "std")]
fn iter_filter_sum_2d_stride_f32(c: &mut Criterion) {
    let a = Array::linspace(0.0..=1.0, 256)
        .into_shape_with_order((16, 16))
        .unwrap();
    let b = a * 100.;
    let b = b.slice(s![.., ..;2]);
    c.bench_function("iter_filter_sum_2d_stride_f32", |bn| bn.iter(|| b.iter().filter(|&&x| x < 75.).sum::<f32>()));
}

#[cfg(feature = "std")]
fn iter_rev_step_by_contiguous(c: &mut Criterion) {
    let a = Array::linspace(0.0..=1.0, 512);
    c.bench_function("iter_rev_step_by_contiguous", |bn| {
        bn.iter(|| {
            a.iter().rev().step_by(2).for_each(|x| {
                black_box(x);
            })
        })
    });
}

#[cfg(feature = "std")]
fn iter_rev_step_by_discontiguous(c: &mut Criterion) {
    let mut a = Array::linspace(0.0..=1.0, 1024);
    a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    c.bench_function("iter_rev_step_by_discontiguous", |bn| {
        bn.iter(|| {
            a.iter().rev().step_by(2).for_each(|x| {
                black_box(x);
            })
        })
    });
}

const ZIPSZ: usize = 10_000;

fn sum_3_std_zip1(c: &mut Criterion) {
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c_vec = vec![1; ZIPSZ];
    c.bench_function("sum_3_std_zip1", |bn| {
        bn.iter(|| {
            a.iter()
                .zip(b.iter().zip(&c_vec))
                .fold(0, |acc, (&a, (&b, &c))| acc + a + b + c)
        })
    });
}

fn sum_3_std_zip2(c: &mut Criterion) {
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c_vec = vec![1; ZIPSZ];
    c.bench_function("sum_3_std_zip2", |bn| {
        bn.iter(|| {
            a.iter()
                .zip(b.iter())
                .zip(&c_vec)
                .fold(0, |acc, ((&a, &b), &c)| acc + a + b + c)
        })
    });
}

fn sum_3_std_zip3(c: &mut Criterion) {
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c_vec = vec![1; ZIPSZ];
    c.bench_function("sum_3_std_zip3", |bn| {
        bn.iter(|| {
            let mut s = 0;
            for ((&a, &b), &c) in a.iter().zip(b.iter()).zip(&c_vec) {
                s += a + b + c
            }
            s
        })
    });
}

fn vector_sum_3_std_zip(c: &mut Criterion) {
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c_vec = vec![1.; ZIPSZ];
    c.bench_function("vector_sum_3_std_zip", |bn| {
        bn.iter(|| {
            for ((&a, &b), c) in a.iter().zip(b.iter()).zip(&mut c_vec) {
                *c += a + b;
            }
        })
    });
}

fn sum_3_azip(c: &mut Criterion) {
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c_vec = vec![1; ZIPSZ];
    c.bench_function("sum_3_azip", |bn| {
        bn.iter(|| {
            let mut s = 0;
            azip!((&a in &a, &b in &b, &c in &c_vec) {
                s += a + b + c;
            });
            s
        })
    });
}

fn sum_3_azip_fold(c: &mut Criterion) {
    let a = vec![1; ZIPSZ];
    let b = vec![1; ZIPSZ];
    let c_vec = vec![1; ZIPSZ];
    c.bench_function("sum_3_azip_fold", |bn| {
        bn.iter(|| {
            Zip::from(&a)
                .and(&b)
                .and(&c_vec)
                .fold_while(0, |acc, &a, &b, &c| FoldWhile::Continue(acc + a + b + c))
                .into_inner()
        })
    });
}

fn vector_sum_3_azip(c: &mut Criterion) {
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c_vec = vec![1.; ZIPSZ];
    c.bench_function("vector_sum_3_azip", |bn| {
        bn.iter(|| {
            azip!((&a in &a, &b in &b, c in &mut c_vec) {
                *c += a + b;
            });
        })
    });
}

fn vector_sum3_unchecked(a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..c.len() {
        unsafe {
            *c.get_unchecked_mut(i) += *a.get_unchecked(i) + *b.get_unchecked(i);
        }
    }
}

fn vector_sum_3_zip_unchecked(c: &mut Criterion) {
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c_vec = vec![1.; ZIPSZ];
    c.bench_function("vector_sum_3_zip_unchecked", |bn| {
        bn.iter(|| {
            vector_sum3_unchecked(&a, &b, &mut c_vec);
        })
    });
}

fn vector_sum_3_zip_unchecked_manual(c: &mut Criterion) {
    let a = vec![1.; ZIPSZ];
    let b = vec![1.; ZIPSZ];
    let mut c_vec = vec![1.; ZIPSZ];
    c.bench_function("vector_sum_3_zip_unchecked_manual", |bn| {
        bn.iter(|| unsafe {
            let mut ap = a.as_ptr();
            let mut bp = b.as_ptr();
            let mut cp = c_vec.as_mut_ptr();
            let cend = cp.add(c_vec.len());
            while cp != cend {
                *cp.post_inc() += *ap.post_inc() + *bp.post_inc();
            }
        })
    });
}

// index iterator size
const ISZ: usize = 16;
const I2DSZ: usize = 64;

fn indexed_iter_1d_ix1(c: &mut Criterion) {
    let mut a = Array::<f64, _>::zeros(I2DSZ * I2DSZ);
    for (i, elt) in a.indexed_iter_mut() {
        *elt = i as _;
    }

    c.bench_function("indexed_iter_1d_ix1", |bn| {
        bn.iter(|| {
            for (i, &_elt) in a.indexed_iter() {
                black_box(i);
            }
        })
    });
}

fn indexed_zip_1d_ix1(c: &mut Criterion) {
    let mut a = Array::<f64, _>::zeros(I2DSZ * I2DSZ);
    for (i, elt) in a.indexed_iter_mut() {
        *elt = i as _;
    }

    c.bench_function("indexed_zip_1d_ix1", |bn| {
        bn.iter(|| {
            Zip::indexed(&a).for_each(|i, &_elt| {
                black_box(i);
            });
        })
    });
}

fn indexed_iter_2d_ix2(c: &mut Criterion) {
    let mut a = Array::<f64, _>::zeros((I2DSZ, I2DSZ));
    for ((i, j), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j) as _;
    }

    c.bench_function("indexed_iter_2d_ix2", |bn| {
        bn.iter(|| {
            for (i, &_elt) in a.indexed_iter() {
                black_box(i);
            }
        })
    });
}

fn indexed_zip_2d_ix2(c: &mut Criterion) {
    let mut a = Array::<f64, _>::zeros((I2DSZ, I2DSZ));
    for ((i, j), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j) as _;
    }

    c.bench_function("indexed_zip_2d_ix2", |bn| {
        bn.iter(|| {
            Zip::indexed(&a).for_each(|i, &_elt| {
                black_box(i);
            });
        })
    });
}

fn indexed_iter_3d_ix3(c: &mut Criterion) {
    let mut a = Array::<f64, _>::zeros((ISZ, ISZ, ISZ));
    for ((i, j, k), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j + 10000 * k) as _;
    }

    c.bench_function("indexed_iter_3d_ix3", |bn| {
        bn.iter(|| {
            for (i, &_elt) in a.indexed_iter() {
                black_box(i);
            }
        })
    });
}

fn indexed_zip_3d_ix3(c: &mut Criterion) {
    let mut a = Array::<f64, _>::zeros((ISZ, ISZ, ISZ));
    for ((i, j, k), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j + 10000 * k) as _;
    }

    c.bench_function("indexed_zip_3d_ix3", |bn| {
        bn.iter(|| {
            Zip::indexed(&a).for_each(|i, &_elt| {
                black_box(i);
            });
        })
    });
}

fn indexed_iter_3d_dyn(c: &mut Criterion) {
    let mut a = Array::<f64, _>::zeros((ISZ, ISZ, ISZ));
    for ((i, j, k), elt) in a.indexed_iter_mut() {
        *elt = (i + 100 * j + 10000 * k) as _;
    }
    let a = a.into_shape_with_order(&[ISZ; 3][..]).unwrap();

    c.bench_function("indexed_iter_3d_dyn", |bn| {
        bn.iter(|| {
            for (i, &_elt) in a.indexed_iter() {
                black_box(i);
            }
        })
    });
}

fn iter_sum_1d_strided_fold(c: &mut Criterion) {
    let mut a = Array::<u64, _>::ones(10240);
    a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    c.bench_function("iter_sum_1d_strided_fold", |bn| bn.iter(|| a.iter().sum::<u64>()));
}

fn iter_sum_1d_strided_rfold(c: &mut Criterion) {
    let mut a = Array::<u64, _>::ones(10240);
    a.slice_axis_inplace(Axis(0), Slice::new(0, None, 2));
    c.bench_function("iter_sum_1d_strided_rfold", |bn| bn.iter(|| a.iter().rfold(0, |acc, &x| acc + x)));
}

fn iter_axis_iter_sum(c: &mut Criterion) {
    let a = Array::<f32, _>::zeros((64, 64));
    c.bench_function("iter_axis_iter_sum", |bn| bn.iter(|| a.axis_iter(Axis(0)).map(|plane| plane.sum()).sum::<f32>()));
}

fn iter_axis_chunks_1_iter_sum(c: &mut Criterion) {
    let a = Array::<f32, _>::zeros((64, 64));
    c.bench_function("iter_axis_chunks_1_iter_sum", |bn| {
        bn.iter(|| {
            a.axis_chunks_iter(Axis(0), 1)
                .map(|plane| plane.sum())
                .sum::<f32>()
        })
    });
}

fn iter_axis_chunks_5_iter_sum(c: &mut Criterion) {
    let a = Array::<f32, _>::zeros((64, 64));
    c.bench_function("iter_axis_chunks_5_iter_sum", |bn| {
        bn.iter(|| {
            a.axis_chunks_iter(Axis(0), 5)
                .map(|plane| plane.sum())
                .sum::<f32>()
        })
    });
}

pub fn zip_mut_with(data: &Array3<f32>, out: &mut Array3<f32>) {
    out.zip_mut_with(data, |o, &i| {
        *o = i;
    });
}

fn zip_mut_with_cc(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros((ISZ, ISZ, ISZ));
    let mut out = Array3::zeros(data.dim());
    c.bench_function("zip_mut_with_cc", |b| b.iter(|| zip_mut_with(&data, &mut out)));
}

fn zip_mut_with_ff(c: &mut Criterion) {
    let data: Array3<f32> = Array3::zeros((ISZ, ISZ, ISZ).f());
    let mut out = Array3::zeros(data.dim().f());
    c.bench_function("zip_mut_with_ff", |b| b.iter(|| zip_mut_with(&data, &mut out)));
}

#[cfg(feature = "std")]
criterion_group!(
    benches_std, iter_filter_sum_2d_u32, iter_filter_sum_2d_f32, iter_filter_sum_2d_stride_u32,
    iter_filter_sum_2d_stride_f32, iter_rev_step_by_contiguous, iter_rev_step_by_discontiguous,
);

criterion_group!(
    benches,
    iter_sum_2d_regular,
    iter_sum_2d_cutout,
    iter_all_2d_cutout,
    iter_sum_2d_transpose,
    sum_3_std_zip1,
    sum_3_std_zip2,
    sum_3_std_zip3,
    vector_sum_3_std_zip,
    sum_3_azip,
    sum_3_azip_fold,
    vector_sum_3_azip,
    vector_sum_3_zip_unchecked,
    vector_sum_3_zip_unchecked_manual,
    indexed_iter_1d_ix1,
    indexed_zip_1d_ix1,
    indexed_iter_2d_ix2,
    indexed_zip_2d_ix2,
    indexed_iter_3d_ix3,
    indexed_zip_3d_ix3,
    indexed_iter_3d_dyn,
    iter_sum_1d_strided_fold,
    iter_sum_1d_strided_rfold,
    iter_axis_iter_sum,
    iter_axis_chunks_1_iter_sum,
    iter_axis_chunks_5_iter_sum,
    zip_mut_with_cc,
    zip_mut_with_ff,
);

#[cfg(feature = "std")]
criterion_main!(benches, benches_std);

#[cfg(not(feature = "std"))]
criterion_main!(benches);
