//! Edge-codec flavor comparison — measure ALL flavors, validate/invalidate.
//!
//! For each data regime, encodes the same vectors with the three edge-codec
//! flavors and reports the full reliability suite (Pearson r, Spearman ρ,
//! ICC(2,1), Cronbach α) on DISTANCE PRESERVATION (true pairwise L2 vs
//! reconstructed pairwise L2 — the metric that matters for nearest-neighbour
//! order), plus per-vector reconstruction rel-L2 and cosine.
//!
//!   CoarseOnly    1 B/vec   palette index (the EdgeBlock byte as-is)
//!   CoarseResidue 1 + D/2   palette + value-slab signed-4-bit residue
//!   Pq32x4        16 B      32 subquantizers × 4-bit (the edge block as PQ)
//!
//! The point: a class/schema picks the flavor by its fidelity/byte tradeoff, and
//! these are the numbers that justify the pick. The deterministic part (nearest
//! centroid) is also shown running on the AMX `matmul_i8_to_i32` tile path,
//! bit-checked against the scalar assignment.
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example edge_codec_compare

use std::time::Instant;

use ndarray::hpc::edge_codec::{reconstruct_coarse, CoarseResidueCodec, Codebook, ProductQuantizer};
use ndarray::hpc::reliability::FidelityReport;
use ndarray::simd::{amx_available, matmul_i8_to_i32};
use ndarray::{ArrayView2, ArrayViewMut2};

fn splitmix(s: &mut u64) -> f32 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 40) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0 // [-1, 1)
}

/// Clustered data: each vector = a random centroid + noise (the regime where a
/// coarse code is meaningful and the residue captures the within-cell offset).
fn gen_blobs(n: usize, dim: usize, k: usize, noise: f32, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let centers: Vec<f32> = (0..k * dim).map(|_| splitmix(&mut s)).collect();
    let mut data = vec![0.0f32; n * dim];
    for i in 0..n {
        let c = (splitmix(&mut s).abs() * k as f32) as usize % k;
        for d in 0..dim {
            data[i * dim + d] = centers[c * dim + d] + noise * splitmix(&mut s);
        }
    }
    data
}

/// Continuous high-dimensional data (no cluster structure): the regime where a
/// coarse codebook can't tile the space and product quantization pulls ahead.
fn gen_continuous(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n * dim).map(|_| splitmix(&mut s)).collect()
}

fn l2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    if na < 1e-24 || nb < 1e-24 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Deterministic candidate pairs (i, j) for the distance-preservation metric.
fn sample_pairs(n: usize, m: usize, seed: u64) -> Vec<(usize, usize)> {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z ^= z >> 31;
        z
    };
    (0..m)
        .map(|_| {
            let i = (next() as usize) % n;
            let mut j = (next() as usize) % n;
            if j == i {
                j = (j + 1) % n;
            }
            (i, j)
        })
        .collect()
}

/// One flavor's measured row: byte cost + reconstruction + distance fidelity.
fn report_flavor(
    name: &str, bytes_per_vec: f64, data: &[f32], recon: &[f32], n: usize, dim: usize, pairs: &[(usize, usize)],
) {
    // Per-vector reconstruction quality.
    let mut rel_num = 0.0;
    let mut rel_den = 0.0;
    let mut cos_sum = 0.0;
    for i in 0..n {
        let v = &data[i * dim..(i + 1) * dim];
        let r = &recon[i * dim..(i + 1) * dim];
        rel_num += l2(v, r);
        rel_den += v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        cos_sum += cosine(v, r);
    }
    let recon_rel = rel_num / rel_den.max(1e-12);
    let recon_cos = cos_sum / n as f64;

    // Distance preservation: true vs reconstructed pairwise L2.
    let true_d: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| l2(&data[i * dim..(i + 1) * dim], &data[j * dim..(j + 1) * dim]))
        .collect();
    let rec_d: Vec<f64> = pairs
        .iter()
        .map(|&(i, j)| l2(&recon[i * dim..(i + 1) * dim], &recon[j * dim..(j + 1) * dim]))
        .collect();
    let f = FidelityReport::compute(&true_d, &rec_d);

    println!(
        "  {name:<14} {bytes_per_vec:>6.1} B  | recon rel-L2 {recon_rel:.4} cos {recon_cos:.4} | dist: r {:.4} ρ {:.4} ICC {:.4} α {:.4}",
        f.pearson, f.spearman, f.icc, f.cronbach
    );
}

/// AMX vs scalar assignment agreement + throughput (the deterministic part).
fn amx_assign_demo(data: &[f32], cb: &Codebook, n: usize, dim: usize) {
    if !amx_available() {
        println!("  (AMX unavailable — deterministic assign runs scalar)");
        return;
    }
    let k = cb.k;
    let q = |x: &[f32]| -> (Vec<i8>, f32) {
        let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-12);
        let sc = 127.0 / amax;
        (
            x.iter()
                .map(|&v| (v * sc).round().clamp(-127.0, 127.0) as i8)
                .collect(),
            sc,
        )
    };
    let (v_i8, _) = q(data);
    let (cb_i8, _) = q(&cb.centroids);
    let mut cbt = vec![0i8; dim * k]; // D×K transpose
    for c in 0..k {
        for d in 0..dim {
            cbt[d * k + c] = cb_i8[c * dim + d];
        }
    }
    let mut g = vec![0i32; n * k];
    let t0 = Instant::now();
    matmul_i8_to_i32(
        ArrayView2::from_shape((n, dim), &v_i8[..]).unwrap(),
        ArrayView2::from_shape((dim, k), &cbt[..]).unwrap(),
        ArrayViewMut2::from_shape((n, k), &mut g[..]).unwrap(),
    )
    .unwrap();
    let ns = t0.elapsed().as_nanos() as f64;
    let cnorm: Vec<i32> = (0..k)
        .map(|c| (0..dim).map(|d| (cb_i8[c * dim + d] as i32).pow(2)).sum())
        .collect();
    let mut agree = 0usize;
    for i in 0..n {
        let mut best = i32::MIN;
        let mut bj = 0u32;
        for j in 0..k {
            let score = 2 * g[i * k + j] - cnorm[j];
            if score > best {
                best = score;
                bj = j as u32;
            }
        }
        if bj == cb.assign(&data[i * dim..(i + 1) * dim]) {
            agree += 1;
        }
    }
    let macs = (n * k * dim) as f64;
    println!(
        "  AMX assign: {:.0} ns ({:.1} GMAC/s), agrees with scalar on {:.1}% of vectors",
        ns,
        macs / ns,
        100.0 * agree as f64 / n as f64
    );
}

fn run(label: &str, data: &[f32], n: usize, dim: usize, k: usize) {
    println!("\n== {label}  (N={n} D={dim} K={k}) ==");
    let pairs = sample_pairs(n, 4096, 0xF00D);

    // Flavor 1: coarse only.
    let cb = Codebook::train(data, n, dim, k, 12, 1);
    let mut recon_coarse = vec![0.0f32; n * dim];
    for i in 0..n {
        let idx = cb.assign(&data[i * dim..(i + 1) * dim]);
        recon_coarse[i * dim..(i + 1) * dim].copy_from_slice(&reconstruct_coarse(&cb, idx));
    }
    report_flavor("CoarseOnly", 1.0, data, &recon_coarse, n, dim, &pairs);

    // Flavor 2: coarse + per-dim 4-bit residue.
    let crc = CoarseResidueCodec::fit(data, n, dim, k, 12, 1);
    let mut recon_res = vec![0.0f32; n * dim];
    for i in 0..n {
        let code = crc.encode(&data[i * dim..(i + 1) * dim]);
        recon_res[i * dim..(i + 1) * dim].copy_from_slice(&crc.reconstruct(&code));
    }
    report_flavor("CoarseResidue", 1.0 + dim as f64 / 2.0, data, &recon_res, n, dim, &pairs);

    // Flavor 3: product quantizer 32×4-bit (16 B).
    if dim.is_multiple_of(32) {
        let pq = ProductQuantizer::fit(data, n, dim, 32, 12, 2);
        let mut recon_pq = vec![0.0f32; n * dim];
        for i in 0..n {
            let code = pq.encode(&data[i * dim..(i + 1) * dim]);
            recon_pq[i * dim..(i + 1) * dim].copy_from_slice(&pq.reconstruct(&code));
        }
        report_flavor("Pq32x4", 16.0, data, &recon_pq, n, dim, &pairs);
    } else {
        println!("  Pq32x4         (skipped — D={dim} not divisible by 32)");
    }

    amx_assign_demo(data, &cb, n, dim);
}

fn main() {
    println!("== Edge-codec flavor comparison (measure all, validate/invalidate) ==");
    println!("amx_available() = {}", amx_available());
    println!("metrics: recon rel-L2/cosine (per-vector) · dist r/ρ/ICC/α (pairwise L2 preservation)");

    let (n, dim, k) = (4096, 128, 256);
    run("blobs σ=0.15", &gen_blobs(n, dim, k, 0.15, 0x1111), n, dim, k);
    run("blobs σ=0.30", &gen_blobs(n, dim, k, 0.30, 0x2222), n, dim, k);
    run("continuous", &gen_continuous(n, dim, 0x3333), n, dim, k);
}
