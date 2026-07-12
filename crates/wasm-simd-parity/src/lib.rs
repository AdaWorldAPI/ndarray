//! Run-under-node parity gate for the wasm SIMD tier.
//!
//! `#[no_mangle] selfcheck()` runs each wasm SIMD lane's arithmetic against the
//! scalar `u32`/`f32`/`i8` reference **in the same module**, returning `0` iff
//! every lane is bit-identical. The companion `run.mjs` instantiates the
//! `.wasm` and asserts `selfcheck() == 0`; `scripts/wasm-parity.sh` wires the
//! build + node run, and the CI `wasm_simd` job runs the script. This is the
//! standing version of the one-off node check from commit 500d57e6 — extend the
//! per-lane blocks below whenever a new `ndarray::simd` lane lands.
//!
//! The whole crate is gated to `wasm32 + simd128` so it compiles to nothing on
//! any other target (it is workspace-excluded and only built by the CI job).
#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use ndarray::simd::{F32x16, I8x16, U32x16};

/// Distinct nonzero return codes make a CI failure point at the exact lane+op.
#[no_mangle]
pub extern "C" fn selfcheck() -> u32 {
    if let Err(code) = check_u32x16() {
        return code;
    }
    if let Err(code) = check_f32x16() {
        return code;
    }
    if let Err(code) = check_i8x16() {
        return code;
    }
    0
}

/// `U32x16` ARX triple (Add / BitXor / rotate_left) — the ChaCha20/BLAKE lane.
fn check_u32x16() -> Result<(), u32> {
    let a_arr: [u32; 16] = [
        0x0000_0000, 0xFFFF_FFFF, 0x0000_0001, 0x8000_0000, 0x1234_5678, 0x9ABC_DEF0, 0xDEAD_BEEF, 0xCAFE_BABE,
        0x0F0F_0F0F, 0xF0F0_F0F0, 0x5555_5555, 0xAAAA_AAAA, 0x0000_00FF, 0xFF00_0000, 0x0101_0101, 0x8080_8080,
    ];
    let b_arr: [u32; 16] = [
        0x9E37_79B9, 0x1111_1111, 0xDEAD_C0DE, 0x0BAD_F00D, 0x7FFF_FFFF, 0x0000_0000, 0xFFFF_FFFF, 0x1357_9BDF,
        0x2468_ACE0, 0xFEDC_BA98, 0x0000_0010, 0x0000_001F, 0xABCD_EF01, 0x1020_4080, 0x0F0F_F0F0, 0xC0DE_CAFE,
    ];
    // from_array / to_array roundtrip (lane ordering).
    if U32x16::from_array(a_arr).to_array() != a_arr {
        return Err(10);
    }
    let a = U32x16::from_array(a_arr);
    let b = U32x16::from_array(b_arr);
    let add = (a + b).to_array();
    let xor = (a ^ b).to_array();
    for i in 0..16 {
        if add[i] != a_arr[i].wrapping_add(b_arr[i]) {
            return Err(11);
        }
        if xor[i] != a_arr[i] ^ b_arr[i] {
            return Err(12);
        }
    }
    // ARX rotate — ChaCha20 uses 16/12/8/7; edges included.
    for &n in &[0u32, 1, 7, 8, 12, 16, 24, 31] {
        let r = a.rotate_left(n).to_array();
        for i in 0..16 {
            if r[i] != a_arr[i].rotate_left(n) {
                return Err(13);
            }
        }
    }
    Ok(())
}

/// `F32x16` — the float hot-path lane (splat / roundtrip / add / reduce_sum).
fn check_f32x16() -> Result<(), u32> {
    let data: [f32; 16] = [
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
    ];
    if F32x16::from_array(data).to_array() != data {
        return Err(20);
    }
    let a = F32x16::from_array(data);
    let b = F32x16::splat(2.0);
    let sum = (a + b).to_array();
    for i in 0..16 {
        if sum[i] != data[i] + 2.0 {
            return Err(21);
        }
    }
    // 0+1+…+15 = 120.
    if F32x16::from_array(data).reduce_sum() != 120.0 {
        return Err(22);
    }
    Ok(())
}

/// `I8x16` — the byte lane (roundtrip / add).
fn check_i8x16() -> Result<(), u32> {
    let a_arr: [i8; 16] = [-128, -1, 0, 1, 127, 2, -2, 3, -3, 42, -42, 100, -100, 7, -7, 64];
    let b_arr: [i8; 16] = [1, 1, 1, 1, 1, -1, -1, -1, 5, -5, 10, -10, 25, -25, 63, -64];
    if I8x16::from_array(a_arr).to_array() != a_arr {
        return Err(30);
    }
    let sum = I8x16::from_array(a_arr).add(I8x16::from_array(b_arr)).to_array();
    for i in 0..16 {
        if sum[i] != a_arr[i].wrapping_add(b_arr[i]) {
            return Err(31);
        }
    }
    Ok(())
}
