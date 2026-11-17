//! Parity for plan `simd-fruits-integration-v1` F7: `U8x64::mask_blend`,
//! `saturating_add` and `saturating_sub` against per-byte `u8` arithmetic,
//! on whichever arm the build compiled (the AVX2 arm realizes them as two
//! `U8x32` halves and a shuffle-free bit-to-byte mask expansion since F7).

use ndarray::simd::U8x64;

fn xs(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn simd_u8x64_blend_report_arm() {
    println!(
        "simd_u8x64_blend: avx512f={} avx512bw={} avx2={} avx={} neon={} simd128={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx512bw"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
    );
}

#[test]
fn mask_blend_takes_b_where_bit_set() {
    let mut s = 0x1234_5678_9ABC_DEF1u64;
    let mut masks: Vec<u64> = vec![0, u64::MAX, 1, 1 << 63, 0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA];
    // every single bit, and every byte-aligned 0xFF window
    masks.extend((0..64).map(|i| 1u64 << i));
    masks.extend((0..8).map(|q| 0xFFu64 << (8 * q)));
    masks.extend((0..400).map(|_| xs(&mut s)));
    let a: [u8; 64] = core::array::from_fn(|i| i as u8);
    let b: [u8; 64] = core::array::from_fn(|i| 200u8.wrapping_add(i as u8));
    for &m in &masks {
        let got = U8x64::mask_blend(m, U8x64::from_array(a), U8x64::from_array(b)).to_array();
        let want: [u8; 64] = core::array::from_fn(|i| if (m >> i) & 1 == 1 { b[i] } else { a[i] });
        assert_eq!(got, want, "mask_blend mask={m:#018x}");
    }
}

#[test]
fn saturating_add_sub_match_u8() {
    let mut s = 0xDEAD_BEEF_0BAD_F00Du64;
    for round in 0..600 {
        let a: [u8; 64] = core::array::from_fn(|i| match round {
            0 => 255,
            1 => 0,
            2 => i as u8 * 4,
            _ => xs(&mut s) as u8,
        });
        let b: [u8; 64] = core::array::from_fn(|i| match round {
            0 => 1,
            1 => 1,
            2 => 255 - i as u8,
            _ => xs(&mut s) as u8,
        });
        let (va, vb) = (U8x64::from_array(a), U8x64::from_array(b));
        let add: [u8; 64] = core::array::from_fn(|i| a[i].saturating_add(b[i]));
        let sub: [u8; 64] = core::array::from_fn(|i| a[i].saturating_sub(b[i]));
        assert_eq!(va.saturating_add(vb).to_array(), add, "saturating_add a={a:?} b={b:?}");
        assert_eq!(va.saturating_sub(vb).to_array(), sub, "saturating_sub a={a:?} b={b:?}");
    }
}
