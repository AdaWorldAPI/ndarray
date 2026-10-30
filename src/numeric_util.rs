// Copyright 2014-2016 bluss and ndarray developers.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp;

use crate::LinalgScalar;

/// Fold over the manually unrolled `xs` with `f`
pub fn unrolled_fold<A, I, F>(mut xs: &[A], init: I, f: F) -> A
where
    A: Clone,
    I: Fn() -> A,
    F: Fn(A, A) -> A,
{
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let mut acc = init();
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (init(), init(), init(), init(), init(), init(), init(), init());
    while xs.len() >= 8 {
        p0 = f(p0, xs[0].clone());
        p1 = f(p1, xs[1].clone());
        p2 = f(p2, xs[2].clone());
        p3 = f(p3, xs[3].clone());
        p4 = f(p4, xs[4].clone());
        p5 = f(p5, xs[5].clone());
        p6 = f(p6, xs[6].clone());
        p7 = f(p7, xs[7].clone());

        xs = &xs[8..];
    }
    acc = f(acc.clone(), f(p0, p4));
    acc = f(acc.clone(), f(p1, p5));
    acc = f(acc.clone(), f(p2, p6));
    acc = f(acc.clone(), f(p3, p7));

    // make it clear to the optimizer that this loop is short
    // and can not be autovectorized.
    for (i, x) in xs.iter().enumerate() {
        if i >= 7 {
            break;
        }
        acc = f(acc.clone(), x.clone())
    }
    acc
}

/// Compute the dot product.
///
/// `xs` and `ys` must be the same length
pub fn unrolled_dot<A>(xs: &[A], ys: &[A]) -> A
where
    A: LinalgScalar,
{
    debug_assert_eq!(xs.len(), ys.len());
    // eightfold unrolled so that floating point can be vectorized
    // (even with strict floating point accuracy semantics)
    let len = cmp::min(xs.len(), ys.len());
    let mut xs = &xs[..len];
    let mut ys = &ys[..len];
    let mut sum = A::zero();
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (A::zero(), A::zero(), A::zero(), A::zero(), A::zero(), A::zero(), A::zero(), A::zero());
    while xs.len() >= 8 {
        p0 = p0 + xs[0] * ys[0];
        p1 = p1 + xs[1] * ys[1];
        p2 = p2 + xs[2] * ys[2];
        p3 = p3 + xs[3] * ys[3];
        p4 = p4 + xs[4] * ys[4];
        p5 = p5 + xs[5] * ys[5];
        p6 = p6 + xs[6] * ys[6];
        p7 = p7 + xs[7] * ys[7];

        xs = &xs[8..];
        ys = &ys[8..];
    }
    sum = sum + (p0 + p4);
    sum = sum + (p1 + p5);
    sum = sum + (p2 + p6);
    sum = sum + (p3 + p7);

    for (i, (&x, &y)) in xs.iter().zip(ys).enumerate() {
        if i >= 7 {
            break;
        }
        sum = sum + x * y;
    }
    sum
}

/// Compute pairwise equality
///
/// `xs` and `ys` must be the same length
pub fn unrolled_eq<A, B>(xs: &[A], ys: &[B]) -> bool
where
    A: PartialEq<B>,
{
    debug_assert_eq!(xs.len(), ys.len());
    // eightfold unrolled for performance (this is not done by llvm automatically)
    let len = cmp::min(xs.len(), ys.len());
    let mut xs = &xs[..len];
    let mut ys = &ys[..len];

    while xs.len() >= 8 {
        if (xs[0] != ys[0])
            | (xs[1] != ys[1])
            | (xs[2] != ys[2])
            | (xs[3] != ys[3])
            | (xs[4] != ys[4])
            | (xs[5] != ys[5])
            | (xs[6] != ys[6])
            | (xs[7] != ys[7])
        {
            return false;
        }
        xs = &xs[8..];
        ys = &ys[8..];
    }

    for i in 0..xs.len() {
        if xs[i] != ys[i] {
            return false;
        }
    }

    true
}

/// The f32 summation order behind `sum`, `sum_axis` and `mean_axis` over a
/// contiguous lane is the eight interleaved partial sums of
/// [`unrolled_fold`], combined as `(p0+p4)+(p1+p5)+(p2+p6)+(p3+p7)`, then the
/// remaining elements in order. It is written out in Rust, has no fused
/// multiply-add, and LLVM may not reassociate IEEE adds, so it is the same on
/// every CPU tier. `examples/float_accum_probe.rs` measured identical bits on
/// AVX-512, FMA+AVX2, AVX, SSE2 and NEON.
#[cfg(test)]
mod accumulation_order_tests {
    use crate::{Array2, Axis};

    fn fixture() -> Array2<f32> {
        Array2::from_shape_fn((64, 1027), |(i, j)| {
            (((i * 2_654_435_761 + j * 40_503) % 100_003) as f32 / 100_003.0 - 0.5) * 3.7
        })
    }

    /// The order spelled out by hand, independent of `unrolled_fold`.
    fn eight_way(xs: &[f32]) -> f32 {
        let mut p = [0.0f32; 8];
        let full = xs.len() / 8 * 8;
        for c in xs[..full].chunks_exact(8) {
            for (pk, &x) in p.iter_mut().zip(c) {
                *pk += x;
            }
        }
        let mut acc = 0.0f32;
        for (a, b) in [(0, 4), (1, 5), (2, 6), (3, 7)] {
            acc += p[a] + p[b];
        }
        for &x in &xs[full..] {
            acc += x;
        }
        acc
    }

    #[test]
    fn sum_and_mean_axis_follow_the_documented_eight_way_order() {
        let a = fixture();
        // 1027 per row: 128 full groups of 8 plus a 3-element remainder.
        for (r, row) in a.rows().into_iter().enumerate() {
            let s = row.as_slice().unwrap();
            assert_eq!(row.sum().to_bits(), eight_way(s).to_bits(), "row {r} sum");
        }
        let mean = a.mean_axis(Axis(1)).unwrap();
        for (r, row) in a.rows().into_iter().enumerate() {
            let want = eight_way(row.as_slice().unwrap()) / 1027.0;
            assert_eq!(mean[r].to_bits(), want.to_bits(), "row {r} mean");
        }
        let all = a.as_slice().unwrap();
        assert_eq!(a.sum().to_bits(), eight_way(all).to_bits(), "whole-array sum");
        // Anti-vacuity: a plain left-to-right sum differs on this fixture, so
        // the comparison above can tell the two orders apart.
        let seq: f32 = all.iter().sum();
        assert_ne!(seq.to_bits(), a.sum().to_bits(), "fixture cannot separate the orders");
    }
}
