//! AMX 2×2 register-block encoding validator. Exercises the NEW tile encodings
//! (tmm3-7 loads, tmm1-3 stores, `for_dpbusd_8`, `tile_dpbusd_2x2`) by computing
//! one 32×32 output block via the register-blocked sequence and comparing to a
//! scalar u8×i8 reference. Front-loads the encoding risk before any driver.
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example amx_rb_probe

use ndarray::hpc::amx_matmul::{
    tile_dpbusd_2x2, tile_load, tile_loadconfig, tile_release, tile_store, tile_zero, vnni_pack_i8, TileConfig,
};
use ndarray::simd::amx_available;

/// Scalar 32×32 u8×i8 → i32 reference: C[i][j] = Σ_k A[i][k]·B[k][j].
fn ref_32(a: &[u8], b: &[i8], k: usize) -> Vec<i32> {
    let mut c = vec![0i32; 32 * 32];
    for i in 0..32 {
        for j in 0..32 {
            let mut s = 0i32;
            for kk in 0..k {
                s += a[i * k + kk] as i32 * b[kk * 32 + j] as i32;
            }
            c[i * 32 + j] = s;
        }
    }
    c
}

/// 32×32 = A(32×k u8) · B(k×32 i8) via the 2×2 register-blocked AMX kernel.
fn rb_32(a: &[u8], b: &[i8], k: usize) -> Vec<i32> {
    // Pack the two 16-wide B column bands into VNNI quads.
    let mut b0 = vec![0i8; k * 16];
    let mut b1 = vec![0i8; k * 16];
    {
        let mut band = vec![0i8; k * 16];
        for kk in 0..k {
            band[kk * 16..(kk + 1) * 16].copy_from_slice(&b[kk * 32..kk * 32 + 16]);
        }
        vnni_pack_i8(&band, &mut b0, k, 16);
        for kk in 0..k {
            band[kk * 16..(kk + 1) * 16].copy_from_slice(&b[kk * 32 + 16..kk * 32 + 32]);
        }
        vnni_pack_i8(&band, &mut b1, k, 16);
    }
    let mut c = vec![0i32; 32 * 32];
    let k_blocks = k / 64;
    // SAFETY: amx_available() checked by caller; 8 tiles configured; every
    // load/store is a 16×64 tile within the a/b0/b1/c allocations; stores use
    // the 32-wide C row pitch (128 bytes) into the four quadrants.
    unsafe {
        tile_loadconfig(&TileConfig::for_dpbusd_8());
        tile_zero(0);
        tile_zero(1);
        tile_zero(2);
        tile_zero(3);
        for kb in 0..k_blocks {
            tile_load(4, a.as_ptr().add(kb * 64), k); // A0 = rows 0..16
            tile_load(5, a.as_ptr().add(16 * k + kb * 64), k); // A1 = rows 16..32
            tile_load(6, b0.as_ptr().add(kb * 16 * 64) as *const u8, 64); // B0 vnni
            tile_load(7, b1.as_ptr().add(kb * 16 * 64) as *const u8, 64); // B1 vnni
            tile_dpbusd_2x2();
        }
        let cp = c.as_mut_ptr();
        let pitch = 32 * 4; // bytes per C row (32 i32)
        tile_store(0, cp as *mut u8, pitch); // C[0..16][0..16]
        tile_store(1, cp.add(16) as *mut u8, pitch); // C[0..16][16..32]
        tile_store(2, cp.add(16 * 32) as *mut u8, pitch); // C[16..32][0..16]
        tile_store(3, cp.add(16 * 32 + 16) as *mut u8, pitch); // C[16..32][16..32]
        tile_release();
    }
    c
}

fn check(k: usize) {
    let a: Vec<u8> = (0..32 * k).map(|i| ((i * 31 + 7) % 256) as u8).collect();
    let b: Vec<i8> = (0..k * 32)
        .map(|i| ((i * 17 + 3) % 256) as u8 as i8)
        .collect();
    let exp = ref_32(&a, &b, k);
    let got = rb_32(&a, &b, k);
    match got.iter().zip(&exp).enumerate().find(|(_, (g, e))| g != e) {
        None => println!("  2x2 register block  K={k:<4}  CORRECT"),
        Some((i, (g, e))) => {
            println!("  2x2 register block  K={k:<4}  WRONG  first@(row {},col {}): got {g} exp {e}", i / 32, i % 32)
        }
    }
}

fn main() {
    println!("amx_available() = {}\n", amx_available());
    if !amx_available() {
        return;
    }
    println!("== 2×2 register-blocked TDPBUSD (new encodings: tmm3-7, for_dpbusd_8) ==");
    check(64); // single K-block
    check(128); // two K-blocks (accumulate across blocks in 4 C tiles)
    check(256); // four K-blocks
}
