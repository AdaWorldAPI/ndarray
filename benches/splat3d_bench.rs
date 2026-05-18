//! Criterion benches for `ndarray::hpc::splat3d` kernels.
//!
//! Per-PR bench growth:
//! - PR 1: `spd3::sandwich` scalar vs `sandwich_x16` SIMD (target ≥10×
//!   on AVX-512), `Spd3::eig` Smith-1961 closed-form throughput,
//!   `Spd3::from_scale_quat` (the 3DGS canonical builder).
//! - PR 2: `gaussian::GaussianBatch::covariance_x16`, `sh::sh_eval_deg3_x16`.
//! - PR 3+: `project_batch`, tile binning, per-tile rasterize.
//!
//! Hardware specs and absolute timings live in `benches/RESULTS.md`,
//! updated per-PR. The bench output committed to RESULTS.md is the
//! gate against regression — a >5% slowdown on any kernel blocks
//! merge per the sprint discipline.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::hpc::splat3d::{sandwich, sandwich_x16, Spd3};

fn bench_spd3_sandwich_scalar_loop(c: &mut Criterion) {
    let m = Spd3::from_scale_quat([1.3, 0.9, 0.6], [0.7071068, 0.0, 0.7071068, 0.0]);
    let n = Spd3::from_scale_quat([0.8, 1.1, 1.4], [0.9238795, 0.3826834, 0.0, 0.0]);
    let ms = [m; 16];
    let ns = [n; 16];

    c.bench_function("spd3_sandwich_scalar_x16_loop", |b| {
        b.iter(|| {
            let mut acc = Spd3::ZERO;
            for i in 0..16 {
                let r = sandwich(&ms[i], &ns[i]);
                acc.a11 += r.a11;
                acc.a22 += r.a22;
                acc.a33 += r.a33;
            }
            black_box(acc);
        });
    });
}

fn bench_spd3_sandwich_simd_x16(c: &mut Criterion) {
    let m = Spd3::from_scale_quat([1.3, 0.9, 0.6], [0.7071068, 0.0, 0.7071068, 0.0]);
    let n = Spd3::from_scale_quat([0.8, 1.1, 1.4], [0.9238795, 0.3826834, 0.0, 0.0]);
    let ms = [m; 16];
    let ns = [n; 16];
    let mut out = [Spd3::ZERO; 16];

    c.bench_function("spd3_sandwich_simd_x16", |b| {
        b.iter(|| {
            sandwich_x16(black_box(&ms), black_box(&ns), &mut out);
            black_box(&out);
        });
    });
}

fn bench_spd3_eig(c: &mut Criterion) {
    let s = Spd3::from_scale_quat([2.0, 1.5, 0.8], [0.8660254, 0.5, 0.0, 0.0]);
    c.bench_function("spd3_eig_smith_1961", |b| {
        b.iter(|| {
            let r = black_box(&s).eig();
            black_box(r);
        });
    });
}

fn bench_spd3_from_scale_quat(c: &mut Criterion) {
    let scale = [1.3f32, 0.9, 0.6];
    let quat = [0.7071068f32, 0.0, 0.7071068, 0.0];
    c.bench_function("spd3_from_scale_quat", |b| {
        b.iter(|| {
            let s = Spd3::from_scale_quat(black_box(scale), black_box(quat));
            black_box(s);
        });
    });
}

criterion_group!(
    spd3,
    bench_spd3_sandwich_scalar_loop,
    bench_spd3_sandwich_simd_x16,
    bench_spd3_eig,
    bench_spd3_from_scale_quat,
);
criterion_main!(spd3);
