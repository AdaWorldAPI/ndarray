use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::prelude::*;

const N: usize = 1024;
const X: usize = 64;
const Y: usize = 16;

#[cfg(feature = "std")]
fn clip(c: &mut Criterion) {
    let mut a = Array::linspace(0.0..=127.0, N * 2)
        .into_shape_with_order([X, Y * 2])
        .unwrap();
    let min = 2.;
    let max = 5.;
    c.bench_function("clip", |b| {
        b.iter(|| {
            black_box(&mut a).mapv_inplace(|mut x| {
                if x < min {
                    x = min
                }
                if x > max {
                    x = max
                }
                x
            })
        })
    });
}

#[cfg(feature = "std")]
criterion_group!(benches, clip);
#[cfg(not(feature = "std"))]
criterion_group!(benches,);
criterion_main!(benches);
