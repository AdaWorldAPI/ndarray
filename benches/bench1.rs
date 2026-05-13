// `s![1..-1, 1..-1]` — ndarray's `s!` macro uses Python-style negative
// indexing. Clippy can't see through the macro and reads `1..-1` as a
// plain (empty) Rust range; the actual semantics are correct.
#![allow(unused_imports)]
#![allow(clippy::many_single_char_names, clippy::deref_addrof, clippy::unreadable_literal, clippy::reversed_empty_ranges)]

use std::mem::MaybeUninit;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::{arr0, arr1, arr2, azip, s};
use ndarray::{Array, Array1, Array2, Axis, Ix, Zip};
use ndarray::{Array3, Array4, ShapeBuilder};
use ndarray::{Ix1, Ix2, Ix3, Ix5, IxDyn};

fn iter_sum_1d_regular(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros(64 * 64);
    let a = black_box(a);
    c.bench_function("iter_sum_1d_regular", |bn| {
        bn.iter(|| {
            let mut sum = 0;
            for &elt in a.iter() {
                sum += elt;
            }
            sum
        })
    });
}

fn iter_sum_1d_raw(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros(64 * 64);
    let a = black_box(a);
    c.bench_function("iter_sum_1d_raw", |bn| {
        bn.iter(|| {
            let mut sum = 0;
            for &elt in a.as_slice_memory_order().unwrap() {
                sum += elt;
            }
            sum
        })
    });
}

fn iter_sum_2d_regular(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    c.bench_function("iter_sum_2d_regular", |bn| {
        bn.iter(|| {
            let mut sum = 0;
            for &elt in a.iter() {
                sum += elt;
            }
            sum
        })
    });
}

fn iter_sum_2d_by_row(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    c.bench_function("iter_sum_2d_by_row", |bn| {
        bn.iter(|| {
            let mut sum = 0;
            for row in a.rows() {
                for &elt in row {
                    sum += elt;
                }
            }
            sum
        })
    });
}

fn iter_sum_2d_raw(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    c.bench_function("iter_sum_2d_raw", |bn| {
        bn.iter(|| {
            let mut sum = 0;
            for &elt in a.as_slice_memory_order().unwrap() {
                sum += elt;
            }
            sum
        })
    });
}

fn iter_sum_2d_cutout(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    c.bench_function("iter_sum_2d_cutout", |bn| {
        bn.iter(|| {
            let mut sum = 0;
            for &elt in a.iter() {
                sum += elt;
            }
            sum
        })
    });
}

fn iter_sum_2d_cutout_by_row(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    c.bench_function("iter_sum_2d_cutout_by_row", |bn| {
        bn.iter(|| {
            let mut sum = 0;
            for row in 0..a.shape()[0] {
                for &elt in a.row(row) {
                    sum += elt;
                }
            }
            sum
        })
    });
}

fn iter_sum_2d_cutout_outer_iter(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    c.bench_function("iter_sum_2d_cutout_outer_iter", |bn| {
        bn.iter(|| {
            let mut sum = 0;
            for row in a.rows() {
                for &elt in row {
                    sum += elt;
                }
            }
            sum
        })
    });
}

fn iter_sum_2d_transpose_regular(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((64, 64));
    a.swap_axes(0, 1);
    let a = black_box(a);
    c.bench_function("iter_sum_2d_transpose_regular", |bn| {
        bn.iter(|| {
            let mut sum = 0;
            for &elt in a.iter() {
                sum += elt;
            }
            sum
        })
    });
}

fn iter_sum_2d_transpose_by_row(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((64, 64));
    a.swap_axes(0, 1);
    let a = black_box(a);
    c.bench_function("iter_sum_2d_transpose_by_row", |bn| {
        bn.iter(|| {
            let mut sum = 0;
            for row in 0..a.shape()[0] {
                for &elt in a.row(row) {
                    sum += elt;
                }
            }
            sum
        })
    });
}

fn sum_2d_regular(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((64, 64));
    let a = black_box(a);
    c.bench_function("sum_2d_regular", |bn| bn.iter(|| a.sum()));
}

fn sum_2d_cutout(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    c.bench_function("sum_2d_cutout", |bn| bn.iter(|| a.sum()));
}

fn sum_2d_float(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let a = black_box(a.view());
    c.bench_function("sum_2d_float", |bn| bn.iter(|| a.sum()));
}

fn sum_2d_float_cutout(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    c.bench_function("sum_2d_float_cutout", |bn| bn.iter(|| a.sum()));
}

fn sum_2d_float_t_cutout(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]).reversed_axes();
    let a = black_box(av);
    c.bench_function("sum_2d_float_t_cutout", |bn| bn.iter(|| a.sum()));
}

fn fold_sum_i32_2d_regular(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((64, 64));
    c.bench_function("fold_sum_i32_2d_regular", |bn| {
        bn.iter(|| a.fold(0, |acc, &x| acc + x))
    });
}

fn fold_sum_i32_2d_cutout(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let av = a.slice(s![1..-1, 1..-1]);
    let a = black_box(av);
    c.bench_function("fold_sum_i32_2d_cutout", |bn| {
        bn.iter(|| a.fold(0, |acc, &x| acc + x))
    });
}

fn fold_sum_i32_2d_stride(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((64, 128));
    let av = a.slice(s![.., ..;2]);
    let a = black_box(av);
    c.bench_function("fold_sum_i32_2d_stride", |bn| {
        bn.iter(|| a.fold(0, |acc, &x| acc + x))
    });
}

fn fold_sum_i32_2d_transpose(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((64, 64));
    let a = a.t();
    c.bench_function("fold_sum_i32_2d_transpose", |bn| {
        bn.iter(|| a.fold(0, |acc, &x| acc + x))
    });
}

fn fold_sum_i32_2d_cutout_transpose(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((66, 66));
    let mut av = a.slice(s![1..-1, 1..-1]);
    av.swap_axes(0, 1);
    let a = black_box(av);
    c.bench_function("fold_sum_i32_2d_cutout_transpose", |bn| {
        bn.iter(|| a.fold(0, |acc, &x| acc + x))
    });
}

const ADD2DSZ: usize = 64;

fn add_2d_regular(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = b.view();
    c.bench_function("add_2d_regular", |bn| {
        bn.iter(|| {
            a += &bv;
        })
    });
}

fn add_2d_zip(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    c.bench_function("add_2d_zip", |bn| {
        bn.iter(|| {
            Zip::from(&mut a).and(&b).for_each(|a, &b| *a += b);
        })
    });
}

fn add_2d_alloc_plus(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    c.bench_function("add_2d_alloc_plus", |bn| bn.iter(|| &a + &b));
}

fn add_2d_alloc_zip_uninit(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    c.bench_function("add_2d_alloc_zip_uninit", |bn| {
        bn.iter(|| unsafe {
            let mut c_arr = Array::<i32, _>::uninit(a.dim());
            azip!((&a in &a, &b in &b, c in c_arr.raw_view_mut().cast::<i32>())
                c.write(a + b)
            );
            c_arr
        })
    });
}

fn add_2d_alloc_zip_collect(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    c.bench_function("add_2d_alloc_zip_collect", |bn| {
        bn.iter(|| Zip::from(&a).and(&b).map_collect(|&x, &y| x + y))
    });
}

fn vec_string_collect(c: &mut Criterion)
{
    let v = vec![""; 10240];
    c.bench_function("vec_string_collect", |bn| {
        bn.iter(|| v.iter().map(|s| s.to_owned()).collect::<Vec<_>>())
    });
}

fn array_string_collect(c: &mut Criterion)
{
    let v = Array::from(vec![""; 10240]);
    c.bench_function("array_string_collect", |bn| {
        bn.iter(|| Zip::from(&v).map_collect(|s| s.to_owned()))
    });
}

fn vec_f64_collect(c: &mut Criterion)
{
    let v = vec![1.; 10240];
    c.bench_function("vec_f64_collect", |bn| {
        bn.iter(|| v.iter().map(|s| s + 1.).collect::<Vec<_>>())
    });
}

fn array_f64_collect(c: &mut Criterion)
{
    let v = Array::from(vec![1.; 10240]);
    c.bench_function("array_f64_collect", |bn| {
        bn.iter(|| Zip::from(&v).map_collect(|s| s + 1.))
    });
}

fn add_2d_assign_ops(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = b.view();
    c.bench_function("add_2d_assign_ops", |bn| {
        bn.iter(|| {
            let mut x = a.view_mut();
            x += &bv;
            black_box(x);
        })
    });
}

fn add_2d_cutout(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ + 2, ADD2DSZ + 2));
    let mut acut = a.slice_mut(s![1..-1, 1..-1]);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = b.view();
    c.bench_function("add_2d_cutout", |bn| {
        bn.iter(|| {
            acut += &bv;
        })
    });
}

fn add_2d_zip_cutout(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ + 2, ADD2DSZ + 2));
    let mut acut = a.slice_mut(s![1..-1, 1..-1]);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    c.bench_function("add_2d_zip_cutout", |bn| {
        bn.iter(|| {
            Zip::from(&mut acut).and(&b).for_each(|a, &b| *a += b);
        })
    });
}

#[allow(clippy::identity_op)]
fn add_2d_cutouts_by_4(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let b = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let chunksz = (4, 4);
    c.bench_function("add_2d_cutouts_by_4", |bn| {
        bn.iter(|| {
            Zip::from(a.exact_chunks_mut(chunksz))
                .and(b.exact_chunks(chunksz))
                .for_each(|mut a, b| a += &b);
        })
    });
}

#[allow(clippy::identity_op)]
fn add_2d_cutouts_by_16(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let b = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let chunksz = (16, 16);
    c.bench_function("add_2d_cutouts_by_16", |bn| {
        bn.iter(|| {
            Zip::from(a.exact_chunks_mut(chunksz))
                .and(b.exact_chunks(chunksz))
                .for_each(|mut a, b| a += &b);
        })
    });
}

#[allow(clippy::identity_op)]
fn add_2d_cutouts_by_32(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let b = Array::<i32, _>::zeros((64 * 1, 64 * 1));
    let chunksz = (32, 32);
    c.bench_function("add_2d_cutouts_by_32", |bn| {
        bn.iter(|| {
            Zip::from(a.exact_chunks_mut(chunksz))
                .and(b.exact_chunks(chunksz))
                .for_each(|mut a, b| a += &b);
        })
    });
}

fn add_2d_broadcast_1_to_2(c: &mut Criterion)
{
    let mut a = Array2::<i32>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array1::<i32>::zeros(ADD2DSZ);
    let bv = b.view();
    c.bench_function("add_2d_broadcast_1_to_2", |bn| {
        bn.iter(|| {
            a += &bv;
        })
    });
}

fn add_2d_broadcast_0_to_2(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<i32, _>::zeros(());
    let bv = b.view();
    c.bench_function("add_2d_broadcast_0_to_2", |bn| {
        bn.iter(|| {
            a += &bv;
        })
    });
}

fn scalar_toowned(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((64, 64));
    c.bench_function("scalar_toowned", |bn| bn.iter(|| a.to_owned()));
}

fn scalar_add_1(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let n = 1.;
    c.bench_function("scalar_add_1", |bn| bn.iter(|| &a + n));
}

fn scalar_add_2(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let n = 1.;
    c.bench_function("scalar_add_2", |bn| bn.iter(|| n + &a));
}

fn scalar_add_strided_1(c: &mut Criterion)
{
    let a = Array::from_shape_fn((64, 64 * 2), |(i, j)| (i * 64 + j) as f32).slice_move(s![.., ..;2]);
    let n = 1.;
    c.bench_function("scalar_add_strided_1", |bn| bn.iter(|| &a + n));
}

fn scalar_add_strided_2(c: &mut Criterion)
{
    let a = Array::from_shape_fn((64, 64 * 2), |(i, j)| (i * 64 + j) as f32).slice_move(s![.., ..;2]);
    let n = 1.;
    c.bench_function("scalar_add_strided_2", |bn| bn.iter(|| n + &a));
}

fn scalar_sub_1(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let n = 1.;
    c.bench_function("scalar_sub_1", |bn| bn.iter(|| &a - n));
}

fn scalar_sub_2(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let n = 1.;
    c.bench_function("scalar_sub_2", |bn| bn.iter(|| n - &a));
}

fn add_2d_0_to_2_iadd_scalar(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let n = black_box(0);
    c.bench_function("add_2d_0_to_2_iadd_scalar", |bn| {
        bn.iter(|| {
            a += n;
        })
    });
}

fn add_2d_strided(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ * 2));
    let mut a = a.slice_mut(s![.., ..;2]);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = b.view();
    c.bench_function("add_2d_strided", |bn| {
        bn.iter(|| {
            a += &bv;
        })
    });
}

fn add_2d_regular_dyn(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros(&[ADD2DSZ, ADD2DSZ][..]);
    let b = Array::<i32, _>::zeros(&[ADD2DSZ, ADD2DSZ][..]);
    let bv = b.view();
    c.bench_function("add_2d_regular_dyn", |bn| {
        bn.iter(|| {
            a += &bv;
        })
    });
}

fn add_2d_strided_dyn(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros(&[ADD2DSZ, ADD2DSZ * 2][..]);
    let mut a = a.slice_mut(s![.., ..;2]);
    let b = Array::<i32, _>::zeros(&[ADD2DSZ, ADD2DSZ][..]);
    let bv = b.view();
    c.bench_function("add_2d_strided_dyn", |bn| {
        bn.iter(|| {
            a += &bv;
        })
    });
}

fn add_2d_zip_strided(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ * 2));
    let mut a = a.slice_mut(s![.., ..;2]);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    c.bench_function("add_2d_zip_strided", |bn| {
        bn.iter(|| {
            Zip::from(&mut a).and(&b).for_each(|a, &b| *a += b);
        })
    });
}

fn add_2d_one_transposed(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    c.bench_function("add_2d_one_transposed", |bn| {
        bn.iter(|| {
            a += &b;
        })
    });
}

fn add_2d_zip_one_transposed(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    c.bench_function("add_2d_zip_one_transposed", |bn| {
        bn.iter(|| {
            Zip::from(&mut a).and(&b).for_each(|a, &b| *a += b);
        })
    });
}

fn add_2d_both_transposed(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let mut b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    b.swap_axes(0, 1);
    c.bench_function("add_2d_both_transposed", |bn| {
        bn.iter(|| {
            a += &b;
        })
    });
}

fn add_2d_zip_both_transposed(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let mut b = Array::<i32, _>::zeros((ADD2DSZ, ADD2DSZ));
    b.swap_axes(0, 1);
    c.bench_function("add_2d_zip_both_transposed", |bn| {
        bn.iter(|| {
            Zip::from(&mut a).and(&b).for_each(|a, &b| *a += b);
        })
    });
}

fn add_2d_f32_regular(c: &mut Criterion)
{
    let mut a = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let b = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = b.view();
    c.bench_function("add_2d_f32_regular", |bn| {
        bn.iter(|| {
            a += &bv;
        })
    });
}

const ADD3DSZ: usize = 16;

fn add_3d_strided(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros((ADD3DSZ, ADD3DSZ, ADD3DSZ * 2));
    let mut a = a.slice_mut(s![.., .., ..;2]);
    let b = Array::<i32, _>::zeros(a.dim());
    let bv = b.view();
    c.bench_function("add_3d_strided", |bn| {
        bn.iter(|| {
            a += &bv;
        })
    });
}

fn add_3d_strided_dyn(c: &mut Criterion)
{
    let mut a = Array::<i32, _>::zeros(&[ADD3DSZ, ADD3DSZ, ADD3DSZ * 2][..]);
    let mut a = a.slice_mut(s![.., .., ..;2]);
    let b = Array::<i32, _>::zeros(a.dim());
    let bv = b.view();
    c.bench_function("add_3d_strided_dyn", |bn| {
        bn.iter(|| {
            a += &bv;
        })
    });
}

const ADD1D_SIZE: usize = 64 * 64;

fn add_1d_regular(c: &mut Criterion)
{
    let mut a = Array::<f32, _>::zeros(ADD1D_SIZE);
    let b = Array::<f32, _>::zeros(a.dim());
    c.bench_function("add_1d_regular", |bn| {
        bn.iter(|| {
            a += &b;
        })
    });
}

fn add_1d_strided(c: &mut Criterion)
{
    let mut a = Array::<f32, _>::zeros(ADD1D_SIZE * 2);
    let mut av = a.slice_mut(s![..;2]);
    let b = Array::<f32, _>::zeros(av.dim());
    c.bench_function("add_1d_strided", |bn| {
        bn.iter(|| {
            av += &b;
        })
    });
}

fn iadd_scalar_2d_regular(c: &mut Criterion)
{
    let mut a = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ));
    c.bench_function("iadd_scalar_2d_regular", |bn| {
        bn.iter(|| {
            a += 1.;
        })
    });
}

fn iadd_scalar_2d_strided(c: &mut Criterion)
{
    let mut a = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ * 2));
    let mut a = a.slice_mut(s![.., ..;2]);
    c.bench_function("iadd_scalar_2d_strided", |bn| {
        bn.iter(|| {
            a += 1.;
        })
    });
}

fn iadd_scalar_2d_regular_dyn(c: &mut Criterion)
{
    let mut a = Array::<f32, _>::zeros(vec![ADD2DSZ, ADD2DSZ]);
    c.bench_function("iadd_scalar_2d_regular_dyn", |bn| {
        bn.iter(|| {
            a += 1.;
        })
    });
}

fn iadd_scalar_2d_strided_dyn(c: &mut Criterion)
{
    let mut a = Array::<f32, _>::zeros(vec![ADD2DSZ, ADD2DSZ * 2]);
    let mut a = a.slice_mut(s![.., ..;2]);
    c.bench_function("iadd_scalar_2d_strided_dyn", |bn| {
        bn.iter(|| {
            a += 1.;
        })
    });
}

fn scaled_add_2d_f32_regular(c: &mut Criterion)
{
    let mut av = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let bv = Array::<f32, _>::zeros((ADD2DSZ, ADD2DSZ));
    let scalar = std::f32::consts::PI;
    c.bench_function("scaled_add_2d_f32_regular", |bn| {
        bn.iter(|| {
            av.scaled_add(scalar, &bv);
        })
    });
}

fn assign_scalar_2d_corder(c: &mut Criterion)
{
    let a = Array::zeros((ADD2DSZ, ADD2DSZ));
    let mut a = black_box(a);
    let s = 3.;
    c.bench_function("assign_scalar_2d_corder", |bn| bn.iter(|| a.fill(s)));
}

fn assign_scalar_2d_cutout(c: &mut Criterion)
{
    let mut a = Array::zeros((66, 66));
    let a = a.slice_mut(s![1..-1, 1..-1]);
    let mut a = black_box(a);
    let s = 3.;
    c.bench_function("assign_scalar_2d_cutout", |bn| bn.iter(|| a.fill(s)));
}

fn assign_scalar_2d_forder(c: &mut Criterion)
{
    let mut a = Array::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let mut a = black_box(a);
    let s = 3.;
    c.bench_function("assign_scalar_2d_forder", |bn| bn.iter(|| a.fill(s)));
}

fn assign_zero_2d_corder(c: &mut Criterion)
{
    let a = Array::zeros((ADD2DSZ, ADD2DSZ));
    let mut a = black_box(a);
    c.bench_function("assign_zero_2d_corder", |bn| bn.iter(|| a.fill(0.)));
}

fn assign_zero_2d_cutout(c: &mut Criterion)
{
    let mut a = Array::zeros((66, 66));
    let a = a.slice_mut(s![1..-1, 1..-1]);
    let mut a = black_box(a);
    c.bench_function("assign_zero_2d_cutout", |bn| bn.iter(|| a.fill(0.)));
}

fn assign_zero_2d_forder(c: &mut Criterion)
{
    let mut a = Array::zeros((ADD2DSZ, ADD2DSZ));
    a.swap_axes(0, 1);
    let mut a = black_box(a);
    c.bench_function("assign_zero_2d_forder", |bn| bn.iter(|| a.fill(0.)));
}

fn bench_iter_diag(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((1024, 1024));
    c.bench_function("bench_iter_diag", |bn| {
        bn.iter(|| {
            for elt in a.diag() {
                black_box(elt);
            }
        })
    });
}

fn bench_row_iter(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((1024, 1024));
    let it = a.row(17);
    c.bench_function("bench_row_iter", |bn| {
        bn.iter(|| {
            for elt in it {
                black_box(elt);
            }
        })
    });
}

fn bench_col_iter(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((1024, 1024));
    let it = a.column(17);
    c.bench_function("bench_col_iter", |bn| {
        bn.iter(|| {
            for elt in it {
                black_box(elt);
            }
        })
    });
}

macro_rules! mat_mul {
    ($modname:ident, $ty:ident, $(($name:ident, $m:expr, $n:expr, $k:expr))+) => {
        mod $modname {
            use criterion::{black_box, Criterion};
            use ndarray::Array;
            $(
            pub fn $name(c: &mut Criterion) {
                let a = Array::<$ty, _>::zeros(($m, $n));
                let b = Array::<$ty, _>::zeros(($n, $k));
                let a = black_box(a.view());
                let b = black_box(b.view());
                let bench_name = concat!(stringify!($modname), "::", stringify!($name));
                c.bench_function(bench_name, |bn| bn.iter(|| a.dot(&b)));
            }
            )+
            pub fn group(c: &mut Criterion) {
                $( $name(c); )+
            }
        }
    };
}

mat_mul! {mat_mul_f32, f32,
    (m004, 4, 4, 4)
    (m007, 7, 7, 7)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
    (m127, 127, 127, 127)
    (mix16x4, 32, 4, 32)
    (mix32x2, 32, 2, 32)
    (mix10000, 128, 10000, 128)
}

mat_mul! {mat_mul_f64, f64,
    (m004, 4, 4, 4)
    (m007, 7, 7, 7)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
    (m127, 127, 127, 127)
    (mix16x4, 32, 4, 32)
    (mix32x2, 32, 2, 32)
    (mix10000, 128, 10000, 128)
}

mat_mul! {mat_mul_i32, i32,
    (m004, 4, 4, 4)
    (m007, 7, 7, 7)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
    (m127, 127, 127, 127)
}

fn create_iter_4d(c: &mut Criterion)
{
    let mut a = Array::from_elem((4, 5, 3, 2), 1.0);
    a.swap_axes(0, 1);
    a.swap_axes(2, 1);
    let v = black_box(a.view());

    c.bench_function("create_iter_4d", |bn| bn.iter(|| v.into_iter()));
}

fn bench_to_owned_n(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((32, 32));
    c.bench_function("bench_to_owned_n", |bn| bn.iter(|| a.to_owned()));
}

fn bench_to_owned_t(c: &mut Criterion)
{
    let mut a = Array::<f32, _>::zeros((32, 32));
    a.swap_axes(0, 1);
    c.bench_function("bench_to_owned_t", |bn| bn.iter(|| a.to_owned()));
}

fn bench_to_owned_strided(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((32, 64));
    let a = a.slice(s![.., ..;2]);
    c.bench_function("bench_to_owned_strided", |bn| bn.iter(|| a.to_owned()));
}

fn equality_i32(c: &mut Criterion)
{
    let a = Array::<i32, _>::zeros((64, 64));
    let b = Array::<i32, _>::zeros((64, 64));
    c.bench_function("equality_i32", |bn| bn.iter(|| a == b));
}

fn equality_f32(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let b = Array::<f32, _>::zeros((64, 64));
    c.bench_function("equality_f32", |bn| bn.iter(|| a == b));
}

fn equality_f32_mixorder(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros((64, 64));
    let b = Array::<f32, _>::zeros((64, 64).f());
    c.bench_function("equality_f32_mixorder", |bn| bn.iter(|| a == b));
}

fn dot_f32_16(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(16);
    let b = Array::<f32, _>::zeros(16);
    c.bench_function("dot_f32_16", |bn| bn.iter(|| a.dot(&b)));
}

fn dot_f32_20(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(20);
    let b = Array::<f32, _>::zeros(20);
    c.bench_function("dot_f32_20", |bn| bn.iter(|| a.dot(&b)));
}

fn dot_f32_32(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(32);
    let b = Array::<f32, _>::zeros(32);
    c.bench_function("dot_f32_32", |bn| bn.iter(|| a.dot(&b)));
}

fn dot_f32_256(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(256);
    let b = Array::<f32, _>::zeros(256);
    c.bench_function("dot_f32_256", |bn| bn.iter(|| a.dot(&b)));
}

fn dot_f32_1024(c: &mut Criterion)
{
    let av = Array::<f32, _>::zeros(1024);
    let bv = Array::<f32, _>::zeros(1024);
    c.bench_function("dot_f32_1024", |bn| bn.iter(|| av.dot(&bv)));
}

fn dot_f32_10e6(c: &mut Criterion)
{
    let n = 1_000_000;
    let av = Array::<f32, _>::zeros(n);
    let bv = Array::<f32, _>::zeros(n);
    c.bench_function("dot_f32_10e6", |bn| bn.iter(|| av.dot(&bv)));
}

fn dot_extended(c: &mut Criterion)
{
    let m = 10;
    let n = 33;
    let k = 10;
    let av = Array::<f32, _>::zeros((m, n));
    let bv = Array::<f32, _>::zeros((n, k));
    let mut res = Array::<f32, _>::zeros((m, k));
    c.bench_function("dot_extended", |bn| {
        bn.iter(|| {
            for i in 0..m {
                for j in 0..k {
                    unsafe {
                        *res.uget_mut((i, j)) = av.row(i).dot(&bv.column(j));
                    }
                }
            }
        })
    });
}

const MEAN_SUM_N: usize = 127;

#[cfg(feature = "std")]
fn range_mat(m: Ix, n: Ix) -> Array2<f32>
{
    assert!(m * n != 0);
    Array::linspace(0.0..=(m * n - 1) as f32, m * n)
        .into_shape_with_order((m, n))
        .unwrap()
}

#[cfg(feature = "std")]
fn mean_axis0(c: &mut Criterion)
{
    let a = range_mat(MEAN_SUM_N, MEAN_SUM_N);
    c.bench_function("mean_axis0", |bn| bn.iter(|| a.mean_axis(Axis(0))));
}

#[cfg(feature = "std")]
fn mean_axis1(c: &mut Criterion)
{
    let a = range_mat(MEAN_SUM_N, MEAN_SUM_N);
    c.bench_function("mean_axis1", |bn| bn.iter(|| a.mean_axis(Axis(1))));
}

#[cfg(feature = "std")]
fn sum_axis0(c: &mut Criterion)
{
    let a = range_mat(MEAN_SUM_N, MEAN_SUM_N);
    c.bench_function("sum_axis0", |bn| bn.iter(|| a.sum_axis(Axis(0))));
}

#[cfg(feature = "std")]
fn sum_axis1(c: &mut Criterion)
{
    let a = range_mat(MEAN_SUM_N, MEAN_SUM_N);
    c.bench_function("sum_axis1", |bn| bn.iter(|| a.sum_axis(Axis(1))));
}

fn into_dimensionality_ix1_ok(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(Ix1(10));
    let a = a.view();
    c.bench_function("into_dimensionality_ix1_ok", |bn| {
        bn.iter(|| a.into_dimensionality::<Ix1>())
    });
}

fn into_dimensionality_ix3_ok(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(Ix3(10, 10, 10));
    let a = a.view();
    c.bench_function("into_dimensionality_ix3_ok", |bn| {
        bn.iter(|| a.into_dimensionality::<Ix3>())
    });
}

fn into_dimensionality_ix3_err(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(Ix3(10, 10, 10));
    let a = a.view();
    c.bench_function("into_dimensionality_ix3_err", |bn| {
        bn.iter(|| a.into_dimensionality::<Ix2>())
    });
}

fn into_dimensionality_dyn_to_ix3(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(IxDyn(&[10, 10, 10]));
    let a = a.view();
    c.bench_function("into_dimensionality_dyn_to_ix3", |bn| {
        bn.iter(|| a.clone().into_dimensionality::<Ix3>())
    });
}

fn into_dimensionality_dyn_to_dyn(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(IxDyn(&[10, 10, 10]));
    let a = a.view();
    c.bench_function("into_dimensionality_dyn_to_dyn", |bn| {
        bn.iter(|| a.clone().into_dimensionality::<IxDyn>())
    });
}

fn into_dyn_ix3(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(Ix3(10, 10, 10));
    let a = a.view();
    c.bench_function("into_dyn_ix3", |bn| bn.iter(|| a.into_dyn()));
}

fn into_dyn_ix5(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(Ix5(2, 2, 2, 2, 2));
    let a = a.view();
    c.bench_function("into_dyn_ix5", |bn| bn.iter(|| a.into_dyn()));
}

fn into_dyn_dyn(c: &mut Criterion)
{
    let a = Array::<f32, _>::zeros(IxDyn(&[10, 10, 10]));
    let a = a.view();
    c.bench_function("into_dyn_dyn", |bn| bn.iter(|| a.clone().into_dyn()));
}

fn broadcast_same_dim(c: &mut Criterion)
{
    let s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let s = Array4::from_shape_vec((2, 2, 3, 2), s.to_vec()).unwrap();
    let a = s.slice(s![.., ..;-1, ..;2, ..]);
    let b = s.slice(s![.., .., ..;2, ..]);
    c.bench_function("broadcast_same_dim", |bn| bn.iter(|| &a + &b));
}

fn broadcast_one_side(c: &mut Criterion)
{
    let s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let s2 = [1, 2, 3, 4, 5, 6];
    let a = Array4::from_shape_vec((4, 1, 3, 2), s.to_vec()).unwrap();
    let b = Array3::from_shape_vec((1, 3, 2), s2.to_vec()).unwrap();
    c.bench_function("broadcast_one_side", |bn| bn.iter(|| &a + &b));
}

criterion_group!(
    iter_benches,
    iter_sum_1d_regular,
    iter_sum_1d_raw,
    iter_sum_2d_regular,
    iter_sum_2d_by_row,
    iter_sum_2d_raw,
    iter_sum_2d_cutout,
    iter_sum_2d_cutout_by_row,
    iter_sum_2d_cutout_outer_iter,
    iter_sum_2d_transpose_regular,
    iter_sum_2d_transpose_by_row,
);

criterion_group!(
    sum_benches,
    sum_2d_regular,
    sum_2d_cutout,
    sum_2d_float,
    sum_2d_float_cutout,
    sum_2d_float_t_cutout,
    fold_sum_i32_2d_regular,
    fold_sum_i32_2d_cutout,
    fold_sum_i32_2d_stride,
    fold_sum_i32_2d_transpose,
    fold_sum_i32_2d_cutout_transpose,
);

criterion_group!(
    add_benches,
    add_2d_regular,
    add_2d_zip,
    add_2d_alloc_plus,
    add_2d_alloc_zip_uninit,
    add_2d_alloc_zip_collect,
    vec_string_collect,
    array_string_collect,
    vec_f64_collect,
    array_f64_collect,
    add_2d_assign_ops,
    add_2d_cutout,
    add_2d_zip_cutout,
    add_2d_cutouts_by_4,
    add_2d_cutouts_by_16,
    add_2d_cutouts_by_32,
    add_2d_broadcast_1_to_2,
    add_2d_broadcast_0_to_2,
);

criterion_group!(
    scalar_benches,
    scalar_toowned,
    scalar_add_1,
    scalar_add_2,
    scalar_add_strided_1,
    scalar_add_strided_2,
    scalar_sub_1,
    scalar_sub_2,
    add_2d_0_to_2_iadd_scalar,
);

criterion_group!(
    add_strided_benches,
    add_2d_strided,
    add_2d_regular_dyn,
    add_2d_strided_dyn,
    add_2d_zip_strided,
    add_2d_one_transposed,
    add_2d_zip_one_transposed,
    add_2d_both_transposed,
    add_2d_zip_both_transposed,
    add_2d_f32_regular,
    add_3d_strided,
    add_3d_strided_dyn,
    add_1d_regular,
    add_1d_strided,
);

criterion_group!(
    iadd_scalar_benches,
    iadd_scalar_2d_regular,
    iadd_scalar_2d_strided,
    iadd_scalar_2d_regular_dyn,
    iadd_scalar_2d_strided_dyn,
    scaled_add_2d_f32_regular,
    assign_scalar_2d_corder,
    assign_scalar_2d_cutout,
    assign_scalar_2d_forder,
    assign_zero_2d_corder,
    assign_zero_2d_cutout,
    assign_zero_2d_forder,
);

criterion_group!(
    iter_misc_benches,
    bench_iter_diag,
    bench_row_iter,
    bench_col_iter,
    create_iter_4d,
    bench_to_owned_n,
    bench_to_owned_t,
    bench_to_owned_strided,
    equality_i32,
    equality_f32,
    equality_f32_mixorder,
);

criterion_group!(
    dot_benches,
    dot_f32_16,
    dot_f32_20,
    dot_f32_32,
    dot_f32_256,
    dot_f32_1024,
    dot_f32_10e6,
    dot_extended,
);

criterion_group!(
    dimensionality_benches,
    into_dimensionality_ix1_ok,
    into_dimensionality_ix3_ok,
    into_dimensionality_ix3_err,
    into_dimensionality_dyn_to_ix3,
    into_dimensionality_dyn_to_dyn,
    into_dyn_ix3,
    into_dyn_ix5,
    into_dyn_dyn,
    broadcast_same_dim,
    broadcast_one_side,
);

criterion_group!(
    matmul_benches,
    mat_mul_f32::group,
    mat_mul_f64::group,
    mat_mul_i32::group,
);

#[cfg(feature = "std")]
criterion_group!(
    std_benches,
    mean_axis0,
    mean_axis1,
    sum_axis0,
    sum_axis1,
);

#[cfg(feature = "std")]
criterion_main!(
    iter_benches,
    sum_benches,
    add_benches,
    scalar_benches,
    add_strided_benches,
    iadd_scalar_benches,
    iter_misc_benches,
    dot_benches,
    dimensionality_benches,
    matmul_benches,
    std_benches,
);

#[cfg(not(feature = "std"))]
criterion_main!(
    iter_benches,
    sum_benches,
    add_benches,
    scalar_benches,
    add_strided_benches,
    iadd_scalar_benches,
    iter_misc_benches,
    dot_benches,
    dimensionality_benches,
    matmul_benches,
);
