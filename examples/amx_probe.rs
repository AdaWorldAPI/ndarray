//! AMX correctness validation — int8_tile_gemm_16x16 (raw u8×i8) and the full
//! matmul_i8_to_i32 (i8×i8 with the +128/bias trick) vs the scalar reference,
//! across single/multi K-block and single/multi-tile shapes.
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example amx_probe

use ndarray::hpc::amx_matmul::matmul_f32;
use ndarray::hpc::int8_tile_gemm::int8_tile_gemm_16x16;
use ndarray::simd::{amx_available, amx_report, cpu_model, matmul_i8_to_i32};
use ndarray::{ArrayView2, ArrayViewMut2};

fn ref_u8_i8_16(a: &[u8], b: &[i8], k: usize) -> Vec<i32> {
    let mut c = vec![0i32; 256];
    for i in 0..16 {
        for j in 0..16 {
            let mut s = 0i32;
            for kk in 0..k {
                s += a[i * k + kk] as i32 * b[kk * 16 + j] as i32;
            }
            c[i * 16 + j] = s;
        }
    }
    c
}

fn ref_i8_i8(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let av = a[i * k + kk] as i32;
            for j in 0..n {
                c[i * n + j] += av * b[kk * n + j] as i32;
            }
        }
    }
    c
}

fn first_mismatch(got: &[i32], exp: &[i32]) -> Option<(usize, i32, i32)> {
    got.iter()
        .zip(exp)
        .enumerate()
        .find(|(_, (g, e))| g != e)
        .map(|(i, (g, e))| (i, *g, *e))
}

fn test_tile_16(k: usize) {
    let a: Vec<u8> = (0..16 * k).map(|i| ((i * 31 + 7) % 256) as u8).collect();
    let b: Vec<i8> = (0..k * 16)
        .map(|i| ((i * 17 + 3) % 256) as u8 as i8)
        .collect();
    let exp = ref_u8_i8_16(&a, &b, k);
    let mut got = vec![0i32; 256];
    int8_tile_gemm_16x16(&a, &b, &mut got, k);
    match first_mismatch(&got, &exp) {
        None => println!("  int8_tile_gemm_16x16  K={k:<4}  CORRECT"),
        Some((i, g, e)) => println!("  int8_tile_gemm_16x16  K={k:<4}  WRONG  first@{i}: got {g} exp {e}"),
    }
}

fn test_matmul(m: usize, n: usize, k: usize) {
    let a: Vec<i8> = (0..m * k)
        .map(|i| ((i * 31 + 7) % 256) as u8 as i8)
        .collect();
    let b: Vec<i8> = (0..k * n)
        .map(|i| ((i * 17 + 3) % 256) as u8 as i8)
        .collect();
    let exp = ref_i8_i8(&a, &b, m, n, k);
    let mut got = vec![0i32; m * n];
    matmul_i8_to_i32(
        ArrayView2::from_shape((m, k), &a[..]).unwrap(),
        ArrayView2::from_shape((k, n), &b[..]).unwrap(),
        ArrayViewMut2::from_shape((m, n), &mut got[..]).unwrap(),
    )
    .unwrap();
    match first_mismatch(&got, &exp) {
        None => println!("  matmul_i8_to_i32  {m:>4}x{k:>4}x{n:>4}  CORRECT"),
        Some((i, g, e)) => println!("  matmul_i8_to_i32  {m:>4}x{k:>4}x{n:>4}  WRONG  first@{i}: got {g} exp {e}"),
    }
}

/// matmul_f32 routes through the BF16 TDPBF16PS tile path on AMX hosts, so we
/// check against an f32 scalar reference with a relative tolerance (BF16 has
/// ~8 mantissa bits → a few ×1e-2 relative error on accumulated sums is fine).
fn test_matmul_f32(m: usize, n: usize, k: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32 - 3.0) * 0.2).collect();
    let mut exp = vec![0.0f32; m * n];
    for i in 0..m {
        for kk in 0..k {
            let av = a[i * k + kk];
            for j in 0..n {
                exp[i * n + j] += av * b[kk * n + j];
            }
        }
    }
    let mut got = vec![0.0f32; m * n];
    matmul_f32(
        ArrayView2::from_shape((m, k), &a[..]).unwrap(),
        ArrayView2::from_shape((k, n), &b[..]).unwrap(),
        ArrayViewMut2::from_shape((m, n), &mut got[..]).unwrap(),
    )
    .unwrap();
    // True relative metric: L2 relative error ‖got−exp‖ / ‖exp‖ (robust to
    // small individual outputs — the previous `|e|.max(1.0)` denominator turned
    // every |e|<1 cell into an absolute-error test) plus the max absolute error.
    let mut sq_err = 0.0f64;
    let mut sq_ref = 0.0f64;
    let mut max_abs = 0.0f32;
    for (g, e) in got.iter().zip(&exp) {
        let d = g - e;
        sq_err += (d as f64) * (d as f64);
        sq_ref += (*e as f64) * (*e as f64);
        max_abs = max_abs.max(d.abs());
    }
    let rel_l2 = (sq_err.sqrt() / sq_ref.sqrt().max(1e-12)) as f32;
    let verdict = if rel_l2 < 0.02 { "CORRECT" } else { "WRONG  " };
    println!("  matmul_f32  {m:>4}x{k:>4}x{n:>4}  {verdict}  rel-L2 {rel_l2:.4}  max-abs {max_abs:.4}");
}

fn main() {
    println!("{}", amx_report());
    println!("cpu_model() = {:?}  has_amx() = {}", cpu_model(), cpu_model().has_amx());
    println!("amx_available() = {}\n", amx_available());

    println!("== matmul_f32 (BF16 TDPBF16PS tile path, ~1% BF16 tolerance) ==");
    test_matmul_f32(16, 16, 32);
    test_matmul_f32(32, 32, 64);
    test_matmul_f32(64, 48, 128);
    test_matmul_f32(128, 128, 256);

    println!("\n== int8_tile_gemm_16x16 (raw u8×i8 tile kernel) ==");
    test_tile_16(64);
    test_tile_16(128);
    test_tile_16(256);

    println!("\n== matmul_i8_to_i32 (signed i8×i8 with +128/bias + multi-tile) ==");
    test_matmul(16, 16, 64);
    test_matmul(16, 16, 128);
    test_matmul(32, 16, 64);
    test_matmul(16, 32, 64);
    test_matmul(32, 32, 128); // exactly one 2×2 register block
    test_matmul(64, 48, 192); // full blocks + right strip (n ≡ 16 mod 32)
    test_matmul(48, 48, 128); // full block + BOTH strips + corner (m,n ≡ 16 mod 32)
    test_matmul(96, 80, 128); // 3×2 full blocks + both strips
    test_matmul(256, 256, 256);
}
