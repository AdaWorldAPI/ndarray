//! AMX correctness validation — int8_tile_gemm_16x16 (raw u8×i8) and the full
//! matmul_i8_to_i32 (i8×i8 with the +128/bias trick) vs the scalar reference,
//! across single/multi K-block and single/multi-tile shapes.
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example amx_probe

use ndarray::hpc::int8_tile_gemm::int8_tile_gemm_16x16;
use ndarray::simd::{amx_available, matmul_i8_to_i32};
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

fn main() {
    println!("amx_available() = {}\n", amx_available());

    println!("== int8_tile_gemm_16x16 (raw u8×i8 tile kernel) ==");
    test_tile_16(64);
    test_tile_16(128);
    test_tile_16(256);

    println!("\n== matmul_i8_to_i32 (signed i8×i8 with +128/bias + multi-tile) ==");
    test_matmul(16, 16, 64);
    test_matmul(16, 16, 128);
    test_matmul(32, 16, 64);
    test_matmul(16, 32, 64);
    test_matmul(32, 32, 128);
    test_matmul(64, 48, 192);
    test_matmul(256, 256, 256);
}
