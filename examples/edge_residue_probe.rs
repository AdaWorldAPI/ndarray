//! AMX edge-residue probe — "stream pairwise into AMX-turbovec as a cheap edge
//! residue that complements the 16×8bit=128bit coarse code."
//!
//! The holy-grail pipeline, measured end-to-end (anti-theater):
//!
//!   vectors (GGUF stand-in)
//!     → AMX int8 GEMM assigns each to a 256-entry palette  (1 byte COARSE)
//!     → 4-bit TurboQuant of the residue                    (D/2 bytes FINE)
//!     → reconstruct, measure error: coarse-only vs coarse+residue
//!
//! The AMX `matmul_i8_to_i32` (this session's 197 GMAC/s kernel) does the
//! pairwise vector×codebook inner products — the "stream pairwise into AMX"
//! step. The 1-byte palette index is the per-edge coarse code (16 edges =
//! 16 bytes = the node's 128-bit EdgeBlock); the 4-bit residue is the
//! turbovec/turboquant complement.
//!
//! PASS: coarse+residue reconstruction error ≪ coarse-only, at a small,
//! fixed extra byte cost, with the assignment running on the AMX tile path.
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example edge_residue_probe

use std::time::Instant;

use ndarray::simd::{amx_available, matmul_i8_to_i32};
use ndarray::{ArrayView2, ArrayViewMut2};

fn splitmix(s: &mut u64) -> f32 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32) / (1u32 << 24) as f32 * 2.0 - 1.0 // [-1, 1)
}

/// Symmetric per-tensor i8 quantization: scale so max|x| → 127.
fn quantize_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-12);
    let scale = 127.0 / amax;
    (
        x.iter()
            .map(|&v| (v * scale).round().clamp(-127.0, 127.0) as i8)
            .collect(),
        scale,
    )
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

fn run(n: usize, d: usize, k: usize, noise: f32) {
    let mut s = 0x1234_5678 ^ (d as u64);
    // Codebook: K centroids, D dims, in [-1,1].
    let cb: Vec<f32> = (0..k * d).map(|_| splitmix(&mut s)).collect();
    // Vectors: each = a random centroid + noise (so palette assignment is real
    // and the residue is the recoverable structure).
    let mut vecs = vec![0.0f32; n * d];
    let mut truth_idx = vec![0u32; n];
    for i in 0..n {
        let c = (splitmix(&mut s).abs() * k as f32) as usize % k;
        truth_idx[i] = c as u32;
        for j in 0..d {
            vecs[i * d + j] = cb[c * d + j] + noise * splitmix(&mut s);
        }
    }

    // ── AMX pairwise: G[i][j] = <v_i, c_j> via int8 GEMM (V[N×D] · Cᵀ[D×K]) ──
    let (v_i8, _) = quantize_i8(&vecs);
    let (cb_i8, _) = quantize_i8(&cb);
    // Transpose codebook to D×K.
    let mut cbt_i8 = vec![0i8; d * k];
    for c in 0..k {
        for j in 0..d {
            cbt_i8[j * k + c] = cb_i8[c * d + j];
        }
    }
    let mut g = vec![0i32; n * k];
    let av = ArrayView2::from_shape((n, d), &v_i8[..]).unwrap();
    let bv = ArrayView2::from_shape((d, k), &cbt_i8[..]).unwrap();
    let t0 = Instant::now();
    matmul_i8_to_i32(av, bv, ArrayViewMut2::from_shape((n, k), &mut g[..]).unwrap()).unwrap();
    let amx_ns = t0.elapsed().as_nanos() as f64;

    // ||c_j||² in the i8 domain (same scale as v) for the argmin.
    let cnorm: Vec<i32> = (0..k)
        .map(|c| (0..d).map(|j| (cb_i8[c * d + j] as i32).pow(2)).sum())
        .collect();
    // idx[i] = argmax_j (2·G[i][j] − ||c_j||²)  ≡  argmin_j ||v_i − c_j||².
    let mut idx = vec![0u32; n];
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
        idx[i] = bj;
    }
    let assign_acc = idx.iter().zip(&truth_idx).filter(|(a, b)| a == b).count() as f64 / n as f64;

    // ── Coarse recon (palette only) + 4-bit TurboQuant residue (the complement) ──
    let mut coarse_err = 0.0f64;
    let mut fine_err = 0.0f64;
    let mut vnorm_sum = 0.0f64;
    for i in 0..n {
        let v = &vecs[i * d..i * d + d];
        let c = &cb[idx[i] as usize * d..idx[i] as usize * d + d];
        // residue + 4-bit signed (−8..7) per-vector uniform quant → D/2 bytes.
        let res: Vec<f32> = v.iter().zip(c).map(|(a, b)| a - b).collect();
        let rmax = res.iter().fold(0.0f32, |a, &x| a.max(x.abs())).max(1e-12);
        let rs = rmax / 7.0;
        let fine: Vec<f32> = res
            .iter()
            .zip(c)
            .map(|(&r, &cc)| cc + ((r / rs).round().clamp(-8.0, 7.0)) * rs)
            .collect();
        let vn = v.iter().map(|x| x * x).sum::<f32>().sqrt() as f64;
        coarse_err += l2(v, c) as f64;
        fine_err += l2(v, &fine) as f64;
        vnorm_sum += vn;
    }
    let coarse_rel = coarse_err / vnorm_sum;
    let fine_rel = fine_err / vnorm_sum;

    let macs = (n * k * d) as f64;
    println!(
        "  N={n} D={d} K={k} noise={noise:.2}:\n\
         \x20   AMX assign: {:.0} ns ({:.1} GMAC/s), accuracy {:.1}%\n\
         \x20   coarse(1B/edge): rel-err {:.4}   →   +turbovec 4-bit ({} B): rel-err {:.4}  ({:.1}× better, +{} B/vec)",
        amx_ns,
        macs / amx_ns,
        100.0 * assign_acc,
        coarse_rel,
        d / 2,
        fine_rel,
        coarse_rel / fine_rel.max(1e-12),
        d / 2,
    );
}

fn main() {
    println!("== AMX edge-residue probe (palette assign on AMX + turbovec 4-bit residue) ==");
    println!("amx_available() = {}\n", amx_available());
    // D multiple of 64, K & N multiples of 16 → AMX tile path.
    for noise in [0.15f32, 0.30] {
        run(4096, 128, 256, noise);
    }
    run(8192, 64, 256, 0.20);
}
