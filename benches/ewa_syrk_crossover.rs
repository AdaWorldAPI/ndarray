//! EWA-SYRK crossover bench — kill-or-justify the "3DGS projection is a
//! BLAS workload in disguise" premise of `3DGS-EWA-SYRK-BLAS-MKL-plan.md`.
//!
//! # The question
//!
//! The plan proposes routing the EWA covariance sandwich `Σ' = M·Σ·Mᵀ`
//! through a batched SYRK/GEMM backend (native / MKL / OpenBLAS / AMX). But
//! the sandwich matrices are **3×3** (world→camera) and **2×3** (camera→
//! image), and `splat3d::project` already runs them 16-wide via the
//! `crate::simd` SoA polyfill. Batched *tiny* GEMM is exactly the regime
//! where CPU BLAS loses to a fused SIMD unroll (per-call overhead dominates;
//! there is no efficient CPU batched-3×3 SYRK — that pattern is a GPU one).
//!
//! This bench measures it instead of asserting it.
//!
//! # What it compares (same `M·N·Mᵀ` math, three kernel shapes)
//!
//! - `scalar`     — the hand-unrolled upper-triangle `spd3::sandwich`, per element.
//! - `simd_x16`   — the shipped `sandwich_x16` SoA path (the renderer's kernel).
//! - `gemm_shape` — two dense 3×3 matmuls per element (`M·N` then `·Mᵀ`),
//!                  with no SoA fusion. This models the **shape a CPU BLAS
//!                  backend imposes** — one small dense GEMM per matrix.
//!
//! `gemm_shape` runs **in-process, with no FFI**, so it is a *lower bound*
//! on a real `cblas`/MKL path: a genuine BLAS call adds argument marshalling
//! + dispatch overhead on top. If `gemm_shape` already loses to `simd_x16`,
//! the BLAS backend loses by more.
//!
//! Plus `project_batch` end-to-end throughput at the same batch sizes, so
//! the sandwich cost is seen in proportion to the full pipeline.
//!
//! Sweep: 1 024 / 100 000 / 1 000 000 (all exact multiples of 16, no tail).
//!
//! Caveat (honest scope): this measures the *per-element* kernel shape. The
//! one genuinely batchable step is `W·Σ·Wᵀ`, where `W` (the camera view
//! rotation) is shared across all gaussians — a shared-`W` batched GEMM is a
//! steelman left as a follow-up. The per-image Jacobian `J` differs per
//! gaussian, so `J·Σ·Jᵀ` does not batch that way.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::hpc::splat3d::{project_batch, sandwich, sandwich_x16, Camera, Gaussian3D, GaussianBatch, ProjectedBatch, Spd3};

const SIZES: [usize; 3] = [1_024, 100_000, 1_000_000];

#[inline]
fn rng(s: &mut u32) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    (*s as f32) / (u32::MAX as f32)
}

/// `n` distinct SPD pairs via `from_scale_quat` (entry-wise different across
/// all 6 SoA channels so the SIMD transpose does real work).
fn build_spd_pairs(n: usize) -> (Vec<Spd3>, Vec<Spd3>) {
    let mut st = 0x1234_5678u32;
    let mut ms = Vec::with_capacity(n);
    let mut ns = Vec::with_capacity(n);
    for _ in 0..n {
        let sm = [0.3 + 1.5 * rng(&mut st), 0.3 + 1.5 * rng(&mut st), 0.3 + 1.5 * rng(&mut st)];
        let sn = [0.3 + 1.5 * rng(&mut st), 0.3 + 1.5 * rng(&mut st), 0.3 + 1.5 * rng(&mut st)];
        let mut qm = [rng(&mut st) - 0.5, rng(&mut st) - 0.5, rng(&mut st) - 0.5, rng(&mut st) - 0.5];
        let mut qn = [rng(&mut st) - 0.5, rng(&mut st) - 0.5, rng(&mut st) - 0.5, rng(&mut st) - 0.5];
        normalize_quat(&mut qm);
        normalize_quat(&mut qn);
        ms.push(Spd3::from_scale_quat(sm, qm));
        ns.push(Spd3::from_scale_quat(sn, qn));
    }
    (ms, ns)
}

fn normalize_quat(q: &mut [f32; 4]) {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().max(1e-12);
    for v in q.iter_mut() {
        *v /= n;
    }
}

/// `M·N·Mᵀ` via two dense 3×3 matmuls — the shape a per-matrix BLAS call
/// imposes (no upper-triangle fusion, no SoA). Output upper triangle, off-
/// diagonals averaged to match `sandwich`'s symmetrisation.
#[inline]
fn sandwich_gemm_shape(m: &Spd3, n: &Spd3) -> Spd3 {
    let a = m.to_rows();
    let b = n.to_rows();
    let mut t = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    // R = T · Mᵀ  (Mᵀ[k][j] = A[j][k])
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = t[i][0] * a[j][0] + t[i][1] * a[j][1] + t[i][2] * a[j][2];
        }
    }
    Spd3::new(r[0][0], 0.5 * (r[0][1] + r[1][0]), 0.5 * (r[0][2] + r[2][0]), r[1][1], 0.5 * (r[1][2] + r[2][1]), r[2][2])
}

fn build_gaussians(n: usize) -> GaussianBatch {
    let mut st = 0x9E37_79B9u32;
    let mut batch = GaussianBatch::with_capacity(n);
    for _ in 0..n {
        let mut g = Gaussian3D::unit();
        g.mean = [4.0 * rng(&mut st) - 2.0, 4.0 * rng(&mut st) - 2.0, 1.0 + 5.0 * rng(&mut st)];
        g.scale = [0.05 + 0.2 * rng(&mut st); 3];
        g.quat = [1.0, 0.0, 0.0, 0.0];
        g.opacity = 0.5 + 0.5 * rng(&mut st);
        batch.push(g);
    }
    batch
}

fn bench_sandwich_paths(c: &mut Criterion) {
    let mut grp = c.benchmark_group("ewa_sandwich_paths");
    grp.sample_size(20);
    for &n in &SIZES {
        let (ms, ns) = build_spd_pairs(n);
        grp.throughput(Throughput::Elements(n as u64));

        grp.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, &n| {
            let mut out = vec![Spd3::ZERO; n];
            b.iter(|| {
                for i in 0..n {
                    out[i] = sandwich(black_box(&ms[i]), black_box(&ns[i]));
                }
                black_box(&out[n - 1]);
            });
        });

        grp.bench_with_input(BenchmarkId::new("simd_x16", n), &n, |b, &n| {
            let mut out = vec![Spd3::ZERO; n];
            b.iter(|| {
                let mc = ms.chunks_exact(16);
                let nc = ns.chunks_exact(16);
                let oc = out.chunks_exact_mut(16);
                for ((mm, nn), oo) in mc.zip(nc).zip(oc) {
                    let ma: &[Spd3; 16] = mm.try_into().unwrap();
                    let na: &[Spd3; 16] = nn.try_into().unwrap();
                    let oa: &mut [Spd3; 16] = oo.try_into().unwrap();
                    sandwich_x16(black_box(ma), black_box(na), oa);
                }
                black_box(&out[n - 1]);
            });
        });

        grp.bench_with_input(BenchmarkId::new("gemm_shape", n), &n, |b, &n| {
            let mut out = vec![Spd3::ZERO; n];
            b.iter(|| {
                for i in 0..n {
                    out[i] = sandwich_gemm_shape(black_box(&ms[i]), black_box(&ns[i]));
                }
                black_box(&out[n - 1]);
            });
        });
    }
    grp.finish();
}

fn bench_project_batch(c: &mut Criterion) {
    let mut grp = c.benchmark_group("ewa_project_batch");
    grp.sample_size(10);
    let cam = Camera::identity_at_origin(1920, 1080);
    for &n in &SIZES {
        let batch = build_gaussians(n);
        let mut out = ProjectedBatch::with_capacity(batch.capacity);
        grp.throughput(Throughput::Elements(n as u64));
        grp.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                project_batch(black_box(&batch), black_box(&cam), &mut out);
                black_box(out.len);
            });
        });
    }
    grp.finish();
}

criterion_group!(ewa_syrk, bench_sandwich_paths, bench_project_batch);
criterion_main!(ewa_syrk);
