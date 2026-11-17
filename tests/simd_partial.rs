//! Parity for plan `simd-fruits-integration-v1` F4: `load_partial` /
//! `store_partial` on the seven chunk types, for every tail length
//! `0..=LANES` (and one longer slice), on whichever arm the build compiled.
//!
//! The store test also checks that nothing past `min(len, LANES)` is written:
//! the destination slice is a window into a larger, sentinel-filled buffer.
#![cfg(feature = "std")]

use ndarray::simd::{F32x16, F32x8, F64x4, F64x8, I32x16, U32x16, U8x64};

#[test]
fn simd_partial_report_arm() {
    println!(
        "simd_partial: avx512f={} avx512bw={} avx2={} avx={} neon={} simd128={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx512bw"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
    );
}

macro_rules! partial_parity {
    ($vec:ident, $elem:ty, $lanes:expr, $val:expr, $sentinel:expr, $zero:expr) => {{
        let src: Vec<$elem> = (0usize..$lanes + 3).map($val).collect();
        for len in 0..=$lanes + 1 {
            // load: lanes < len come from the slice, the rest are zero
            let got = $vec::load_partial(&src[..len]).to_array();
            for i in 0..$lanes {
                let want = if i < len { src[i] } else { $zero };
                assert_eq!(got[i], want, "{}::load_partial len {} lane {}", stringify!($vec), len, i);
            }
            // store: exactly min(len, LANES) elements written, neighbours untouched
            let mut buf: Vec<$elem> = vec![$sentinel; $lanes + 6];
            $vec::from_array(core::array::from_fn(|i| src[i])).store_partial(&mut buf[2..2 + len]);
            let n = len.min($lanes);
            for (j, &x) in buf.iter().enumerate() {
                let want = if j >= 2 && j < 2 + n { src[j - 2] } else { $sentinel };
                assert_eq!(x, want, "{}::store_partial len {} index {}", stringify!($vec), len, j);
            }
        }
    }};
}

#[test]
fn load_store_partial_every_tail_length() {
    partial_parity!(F32x16, f32, 16, |i| i as f32 + 0.5, -7.0f32, 0.0f32);
    partial_parity!(F64x8, f64, 8, |i| i as f64 * 1.25, -7.0f64, 0.0f64);
    partial_parity!(F32x8, f32, 8, |i| 100.0 - i as f32, -7.0f32, 0.0f32);
    partial_parity!(F64x4, f64, 4, |i| i as f64 - 3.0, -7.0f64, 0.0f64);
    partial_parity!(U8x64, u8, 64, |i| (i as u8).wrapping_mul(37).wrapping_add(1), 0xEEu8, 0u8);
    partial_parity!(I32x16, i32, 16, |i| i as i32 * -1000 - 1, i32::MIN, 0i32);
    partial_parity!(U32x16, u32, 16, |i| 0xF000_0000u32 | i as u32, 0xDEAD_BEEFu32, 0u32);
}

#[test]
fn tail_loop_sums_unaligned_lengths() {
    // The use the API exists for: a chunked reduction with a ragged tail.
    for n in [0usize, 1, 15, 16, 17, 31, 33, 100, 255] {
        let data: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
        let mut acc = F32x16::splat(0.0);
        let (chunks, tail) = data.as_chunks::<16>();
        for c in chunks {
            acc += F32x16::from_array(*c);
        }
        acc += F32x16::load_partial(tail);
        assert_eq!(acc.reduce_sum(), data.iter().sum::<f32>(), "n = {n}");
    }
}
