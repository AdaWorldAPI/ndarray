//! Parity for plan `simd-fruits-integration-v1` F6: `U32x16::mul_widen_even`
//! (exact u32×u32→u64 of the even lanes) and `U16x32::mul_high` /
//! `I16x32::mul_high` (high 16 bits of the 32-bit product), against `u64` /
//! `u32` / `i32` arithmetic computed here, so the file grades whichever arm
//! the build compiled.
#![cfg(feature = "std")]

use ndarray::simd::{I16x32, U16x32, U32x16};

/// Small xorshift so the inputs cover the full range, including the extremes.
fn xs(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn simd_widen_mul_report_arm() {
    println!(
        "simd_widen_mul: avx512f={} avx512bw={} avx2={} avx={} neon={} simd128={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx512bw"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
    );
}

#[test]
fn mul_widen_even_matches_u64() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let edge = [0u32, 1, 2, u32::MAX, u32::MAX - 1, 0x8000_0000, 0xFFFF, 0x1_0000];
    for round in 0..500 {
        let a: [u32; 16] = core::array::from_fn(|i| {
            if round < 8 {
                edge[(i + round) % 8]
            } else {
                xs(&mut s) as u32
            }
        });
        let b: [u32; 16] = core::array::from_fn(|i| {
            if round < 8 {
                edge[(i * 3 + round) % 8]
            } else {
                xs(&mut s) as u32
            }
        });
        let got = U32x16::from_array(a)
            .mul_widen_even(U32x16::from_array(b))
            .to_array();
        let want: [u64; 8] = core::array::from_fn(|k| u64::from(a[2 * k]) * u64::from(b[2 * k]));
        assert_eq!(got, want, "mul_widen_even a={a:?} b={b:?}");
    }
}

#[test]
fn mul_high_matches_u32_and_i32() {
    let mut s = 0xD1B5_4A32_D192_ED03u64;
    let ue = [0u16, 1, 2, u16::MAX, 0x8000, 0x7FFF, 0x0100, 0x00FF];
    let ie = [0i16, 1, -1, i16::MIN, i16::MAX, -2, 256, -256];
    for round in 0..500 {
        let a: [u16; 32] = core::array::from_fn(|i| {
            if round < 8 {
                ue[(i + round) % 8]
            } else {
                xs(&mut s) as u16
            }
        });
        let b: [u16; 32] = core::array::from_fn(|i| {
            if round < 8 {
                ue[(i * 5 + round) % 8]
            } else {
                xs(&mut s) as u16
            }
        });
        let got = U16x32::from_array(a)
            .mul_high(U16x32::from_array(b))
            .to_array();
        let want: [u16; 32] = core::array::from_fn(|i| ((u32::from(a[i]) * u32::from(b[i])) >> 16) as u16);
        assert_eq!(got, want, "U16x32::mul_high a={a:?} b={b:?}");

        let c: [i16; 32] = core::array::from_fn(|i| {
            if round < 8 {
                ie[(i + round) % 8]
            } else {
                xs(&mut s) as i16
            }
        });
        let d: [i16; 32] = core::array::from_fn(|i| {
            if round < 8 {
                ie[(i * 3 + round) % 8]
            } else {
                xs(&mut s) as i16
            }
        });
        let got = I16x32::from_array(c)
            .mul_high(I16x32::from_array(d))
            .to_array();
        let want: [i16; 32] = core::array::from_fn(|i| ((i32::from(c[i]) * i32::from(d[i])) >> 16) as i16);
        assert_eq!(got, want, "I16x32::mul_high c={c:?} d={d:?}");
    }
}
