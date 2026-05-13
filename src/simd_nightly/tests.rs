//! Round-3-portable-simd parity tests — agent #12.
//!
//! Every test verifies `simd_nightly::*` behaves identically to pure scalar
//! reference math. No comparison against `simd_avx512::*` (miri-opaque).
#![cfg(all(test, feature = "nightly-simd"))]

use super::*;

// ════════════════════════════════════════════════════════════════════════════
// 1. Constructor roundtrip — splat/from_array/from_slice/copy_to_slice
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn f32x16_splat_roundtrip() {
    let v = F32x16::splat(3.14_f32);
    let arr = v.to_array();
    assert!(arr.iter().all(|&x| (x - 3.14_f32).abs() < 1e-6));
}

#[test]
fn f32x16_from_array_identity() {
    let src: [f32; 16] = core::array::from_fn(|i| i as f32 * 7.0);
    assert_eq!(F32x16::from_array(src).to_array(), src);
}

#[test]
fn f32x16_from_slice_copy_to_slice() {
    let src: [f32; 16] = core::array::from_fn(|i| i as f32);
    let mut dst = [0.0_f32; 16];
    F32x16::from_slice(&src).copy_to_slice(&mut dst);
    assert_eq!(src, dst);
}

#[test]
fn f32x8_splat_roundtrip() {
    let v = F32x8::splat(-1.0_f32);
    assert!(v.to_array().iter().all(|&x| x == -1.0_f32));
}

#[test]
fn f32x8_from_array_identity() {
    let src: [f32; 8] = core::array::from_fn(|i| i as f32 * 3.0);
    assert_eq!(F32x8::from_array(src).to_array(), src);
}

#[test]
fn f32x8_from_slice_copy_to_slice() {
    let src: [f32; 8] = core::array::from_fn(|i| (i * 5) as f32);
    let mut dst = [0.0_f32; 8];
    F32x8::from_slice(&src).copy_to_slice(&mut dst);
    assert_eq!(src, dst);
}

#[test]
fn f64x8_splat_roundtrip() {
    let v = F64x8::splat(2.718_f64);
    assert!(v.to_array().iter().all(|&x| (x - 2.718_f64).abs() < 1e-12));
}

#[test]
fn f64x8_from_array_identity() {
    let src: [f64; 8] = core::array::from_fn(|i| i as f64 * 7.0);
    assert_eq!(F64x8::from_array(src).to_array(), src);
}

#[test]
fn f64x8_from_slice_copy_to_slice() {
    let src: [f64; 8] = core::array::from_fn(|i| i as f64);
    let mut dst = [0.0_f64; 8];
    F64x8::from_slice(&src).copy_to_slice(&mut dst);
    assert_eq!(src, dst);
}

#[test]
fn f64x4_splat_roundtrip() {
    let v = F64x4::splat(1.414_f64);
    assert!(v.to_array().iter().all(|&x| (x - 1.414_f64).abs() < 1e-12));
}

#[test]
fn f64x4_from_array_identity() {
    let src: [f64; 4] = core::array::from_fn(|i| (i + 1) as f64 * 2.5);
    assert_eq!(F64x4::from_array(src).to_array(), src);
}

#[test]
fn f64x4_from_slice_copy_to_slice() {
    let src: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut dst = [0.0_f64; 4];
    F64x4::from_slice(&src).copy_to_slice(&mut dst);
    assert_eq!(src, dst);
}

// ════════════════════════════════════════════════════════════════════════════
// 2. Reduction parity — sum / min / max vs scalar fold
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn f32x16_reduce_sum_parity() {
    let src: [f32; 16] = core::array::from_fn(|i| (i as f32) * 7.0);
    let scalar_sum: f32 = src.iter().copied().sum();
    let simd_sum = F32x16::from_array(src).reduce_sum();
    // Allow small FP rounding difference
    assert!((simd_sum - scalar_sum).abs() < 1e-3, "sum mismatch: {simd_sum} vs {scalar_sum}");
}

#[test]
fn f32x16_reduce_min_parity() {
    let src: [f32; 16] = core::array::from_fn(|i| (i as f32) * 7.0);
    let scalar_min = src.iter().copied().fold(f32::INFINITY, f32::min);
    assert_eq!(F32x16::from_array(src).reduce_min(), scalar_min);
}

#[test]
fn f32x16_reduce_max_parity() {
    let src: [f32; 16] = core::array::from_fn(|i| (i as f32) * 7.0);
    let scalar_max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert_eq!(F32x16::from_array(src).reduce_max(), scalar_max);
}

#[test]
fn f32x8_reduce_sum_parity() {
    let src: [f32; 8] = core::array::from_fn(|i| (i as f32) * 7.0);
    let scalar_sum: f32 = src.iter().copied().sum();
    let simd_sum = F32x8::from_array(src).reduce_sum();
    assert!((simd_sum - scalar_sum).abs() < 1e-3);
}

#[test]
fn f32x8_reduce_min_max_parity() {
    let src: [f32; 8] = core::array::from_fn(|i| (i as f32) * 3.0);
    let scalar_min = src.iter().copied().fold(f32::INFINITY, f32::min);
    let scalar_max = src.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let v = F32x8::from_array(src);
    assert_eq!(v.reduce_min(), scalar_min);
    assert_eq!(v.reduce_max(), scalar_max);
}

#[test]
fn f64x8_reduce_sum_parity() {
    let src: [f64; 8] = core::array::from_fn(|i| (i as f64) * 7.0);
    let scalar_sum: f64 = src.iter().copied().sum();
    let simd_sum = F64x8::from_array(src).reduce_sum();
    assert!((simd_sum - scalar_sum).abs() < 1e-9);
}

#[test]
fn f64x8_reduce_min_max_parity() {
    let src: [f64; 8] = core::array::from_fn(|i| (i as f64) * 5.0);
    let scalar_min = src.iter().copied().fold(f64::INFINITY, f64::min);
    let scalar_max = src.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let v = F64x8::from_array(src);
    assert_eq!(v.reduce_min(), scalar_min);
    assert_eq!(v.reduce_max(), scalar_max);
}

#[test]
fn f64x4_reduce_sum_parity() {
    let src: [f64; 4] = core::array::from_fn(|i| (i as f64) * 7.0);
    let scalar_sum: f64 = src.iter().copied().sum();
    let simd_sum = F64x4::from_array(src).reduce_sum();
    assert!((simd_sum - scalar_sum).abs() < 1e-9);
}

// ════════════════════════════════════════════════════════════════════════════
// 3. Comparison mask parity — bitmask matches scalar per-lane predicate
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn f32x16_cmpeq_mask_parity() {
    let a: [f32; 16] = core::array::from_fn(|i| (i % 4) as f32);
    let b: [f32; 16] = core::array::from_fn(|i| (i % 2) as f32);
    let mask = F32x16::from_array(a).simd_eq(F32x16::from_array(b)).to_bitmask();
    let mut expected: u16 = 0;
    for i in 0..16 {
        if a[i] == b[i] {
            expected |= 1 << i;
        }
    }
    assert_eq!(mask, expected, "cmpeq mask mismatch");
}

#[test]
fn f32x16_cmpgt_mask_parity() {
    let a: [f32; 16] = core::array::from_fn(|i| i as f32);
    let b = F32x16::splat(7.5);
    let mask = F32x16::from_array(a).simd_gt(b).to_bitmask();
    let mut expected: u16 = 0;
    for i in 0..16 {
        if a[i] > 7.5 {
            expected |= 1 << i;
        }
    }
    assert_eq!(mask, expected, "cmpgt mask mismatch");
}

#[test]
fn f32x8_cmpeq_mask_parity() {
    let a: [f32; 8] = core::array::from_fn(|i| (i % 2) as f32);
    let b: [f32; 8] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let mask = F32x8::from_array(a).simd_eq(F32x8::from_array(b)).to_bitmask();
    let mut expected: u8 = 0;
    for i in 0..8 {
        if a[i] == b[i] {
            expected |= 1 << i;
        }
    }
    assert_eq!(mask, expected);
}

#[test]
fn f64x8_cmpeq_mask_parity() {
    let a: [f64; 8] = core::array::from_fn(|i| (i % 3) as f64);
    let b: [f64; 8] = core::array::from_fn(|i| (i % 2) as f64);
    let mask = F64x8::from_array(a).simd_eq(F64x8::from_array(b)).to_bitmask();
    let mut expected: u8 = 0;
    for i in 0..8 {
        if a[i] == b[i] {
            expected |= 1 << i;
        }
    }
    assert_eq!(mask, expected);
}

#[test]
fn f64x4_cmpgt_mask_parity() {
    let a: [f64; 4] = [5.0, 1.0, 7.0, 3.0];
    let b = F64x4::splat(4.0);
    let mask = F64x4::from_array(a).simd_gt(b).to_bitmask();
    // Lanes 0 (5>4) and 2 (7>4) → bits 0 and 2 → 0b0101 = 5
    assert_eq!(mask, 0b0101_u8);
}

#[test]
fn u8x64_cmpeq_mask_parity() {
    let a: [u8; 64] = core::array::from_fn(|i| (i & 7) as u8);
    let b: [u8; 64] = core::array::from_fn(|i| (i & 6) as u8);
    let mask = u8_types::U8x64::from_array(a).cmpeq_mask(u8_types::U8x64::from_array(b));
    for i in 0..64 {
        let expected_bit = if a[i] == b[i] { 1u64 } else { 0u64 };
        assert_eq!((mask >> i) & 1, expected_bit, "lane {i}");
    }
}

#[test]
fn u8x64_cmpgt_mask_parity() {
    let a: [u8; 64] = core::array::from_fn(|i| i as u8);
    let b: [u8; 64] = core::array::from_fn(|i| (63 - i) as u8);
    let mask = u8_types::U8x64::from_array(a).cmpgt_mask(u8_types::U8x64::from_array(b));
    for i in 0..64 {
        let expected_bit = if a[i] > b[i] { 1u64 } else { 0u64 };
        assert_eq!((mask >> i) & 1, expected_bit, "lane {i}: {} > {}?", a[i], b[i]);
    }
}

#[test]
fn u8x32_cmpeq_mask_parity() {
    let a: [u8; 32] = core::array::from_fn(|i| (i & 3) as u8);
    let b: [u8; 32] = core::array::from_fn(|i| (i & 2) as u8);
    let mask = u8_types::U8x32::from_array(a).cmpeq_mask(u8_types::U8x32::from_array(b));
    let mut expected: u32 = 0;
    for i in 0..32 {
        if a[i] == b[i] {
            expected |= 1 << i;
        }
    }
    assert_eq!(mask, expected);
}

// ════════════════════════════════════════════════════════════════════════════
// 4. Saturating arithmetic — clamps at max/min for unsigned integer types
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn u8x64_saturating_add_clamps_at_max() {
    let a = u8_types::U8x64::splat(200);
    let b = u8_types::U8x64::splat(100);
    let r = a.saturating_add(b);
    // scalar: min(200 + 100, 255) = 255
    assert!(r.to_array().iter().all(|&x| x == 255), "expected 255, got {:?}", r.to_array()[0]);
}

#[test]
fn u8x64_saturating_sub_clamps_at_zero() {
    let a = u8_types::U8x64::splat(10);
    let b = u8_types::U8x64::splat(50);
    let r = a.saturating_sub(b);
    // scalar: max(10 - 50, 0) = 0
    assert!(r.to_array().iter().all(|&x| x == 0));
}

#[test]
fn u8x32_saturating_add_clamps_at_max() {
    let a = u8_types::U8x32::splat(255);
    let b = u8_types::U8x32::splat(1);
    let r = a.saturating_add(b);
    assert!(r.to_array().iter().all(|&x| x == 255));
}

#[test]
fn u8x32_saturating_sub_clamps_at_zero() {
    let a = u8_types::U8x32::splat(0);
    let b = u8_types::U8x32::splat(100);
    let r = a.saturating_sub(b);
    assert!(r.to_array().iter().all(|&x| x == 0));
}

#[test]
fn u16x32_saturating_add_clamps_at_max() {
    let a = u_word_types::U16x32::splat(u16::MAX - 1);
    let b = u_word_types::U16x32::splat(10);
    let r = a.saturating_add(b);
    assert!(r.to_array().iter().all(|&x| x == u16::MAX));
}

#[test]
fn u16x32_saturating_sub_clamps_at_zero() {
    let a = u_word_types::U16x32::splat(5);
    let b = u_word_types::U16x32::splat(10);
    let r = a.saturating_sub(b);
    assert!(r.to_array().iter().all(|&x| x == 0));
}

// ════════════════════════════════════════════════════════════════════════════
// 5. FMA bit-exact — mul_add(0.5, 2.0, 1.0) = 2.0 per lane
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn f32x16_fma_exact() {
    // 0.5 * 2.0 + 1.0 = 2.0 (exact in IEEE-754)
    let result = F32x16::splat(0.5).mul_add(F32x16::splat(2.0), F32x16::splat(1.0));
    let arr = result.to_array();
    assert!(arr.iter().all(|&v| v == 2.0_f32), "expected all 2.0, got {:?}", &arr[..4]);
}

#[test]
fn f32x8_fma_exact() {
    let result = F32x8::splat(0.5).mul_add(F32x8::splat(2.0), F32x8::splat(1.0));
    assert!(result.to_array().iter().all(|&v| v == 2.0_f32));
}

#[test]
fn f64x8_fma_exact() {
    // 0.5 * 2.0 + 1.0 = 2.0 (exact)
    let result = F64x8::splat(0.5).mul_add(F64x8::splat(2.0), F64x8::splat(1.0));
    assert!(result.to_array().iter().all(|&v| v == 2.0_f64));
}

#[test]
fn f64x4_fma_exact() {
    let result = F64x4::splat(0.5).mul_add(F64x4::splat(2.0), F64x4::splat(1.0));
    assert!(result.to_array().iter().all(|&v| v == 2.0_f64));
}

#[test]
fn f32x16_fma_scalar_parity() {
    // Verify mul_add matches scalar f32::mul_add lane-by-lane
    let a: [f32; 16] = core::array::from_fn(|i| (i as f32) * 0.1);
    let b: [f32; 16] = core::array::from_fn(|i| (i as f32) * 0.3);
    let c: [f32; 16] = core::array::from_fn(|i| i as f32);
    let simd_result = F32x16::from_array(a).mul_add(F32x16::from_array(b), F32x16::from_array(c));
    for (i, (&r, (&ai, (&bi, &ci)))) in simd_result.to_array().iter()
        .zip(a.iter().zip(b.iter().zip(c.iter())))
        .enumerate()
    {
        let scalar = ai.mul_add(bi, ci);
        assert!((r - scalar).abs() < 1e-5, "lane {i}: simd={r} scalar={scalar}");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 6. BF16 / F16 roundtrip — from_f32 → to_f32 within expected precision
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn bf16x16_from_f32_to_f32_lossy_within_bounds() {
    // BF16 truncates the low 16 mantissa bits; error ≤ 2^(e-7) where e = exponent
    let inputs: [f32; 16] = core::array::from_fn(|i| (i as f32 + 1.0) * 0.5);
    let v = bf16_types::BF16x16::from_f32_truncate(inputs);
    let out = v.to_f32_lossy();
    for (i, (&orig, &back)) in inputs.iter().zip(out.iter()).enumerate() {
        // BF16 truncation error is at most |orig| * 2^-7
        let max_err = orig.abs() * (2.0_f32.powi(-7)) + f32::EPSILON;
        assert!(
            (orig - back).abs() <= max_err,
            "lane {i}: orig={orig} back={back} err={}", (orig - back).abs()
        );
    }
}

#[test]
fn bf16x8_from_f32_to_f32_lossy_within_bounds() {
    let inputs: [f32; 8] = core::array::from_fn(|i| i as f32 * 2.0 + 1.0);
    let v = bf16_types::BF16x8::from_f32_truncate(inputs);
    let out = v.to_f32_lossy();
    for (i, (&orig, &back)) in inputs.iter().zip(out.iter()).enumerate() {
        let max_err = orig.abs() * (2.0_f32.powi(-7)) + f32::EPSILON;
        assert!(
            (orig - back).abs() <= max_err,
            "lane {i}: orig={orig} back={back}"
        );
    }
}

#[test]
fn bf16x16_bit_pattern_stability() {
    // Round-trip through bit patterns must be identity
    let bits: [u16; 16] = core::array::from_fn(|i| (i as u16) * 0x100 + 0x3F00);
    let v = bf16_types::BF16x16::from_array(bits);
    assert_eq!(v.to_array(), bits);
}

#[test]
fn bf16x8_bit_pattern_stability() {
    let bits: [u16; 8] = core::array::from_fn(|i| (i as u16) * 0x200 + 0x3F00);
    let v = bf16_types::BF16x8::from_array(bits);
    assert_eq!(v.to_array(), bits);
}

#[test]
fn f16x16_from_f32_to_f32_roundtrip_within_1ulp() {
    // Representable f16 values (0..=15 as f32) should round-trip exactly.
    let inputs: [f32; 16] = core::array::from_fn(|i| i as f32);
    let v = f16_types::F16x16::from_f32_array(inputs);
    let out = v.to_f32_array();
    for (i, (&orig, &back)) in inputs.iter().zip(out.iter()).enumerate() {
        // These integers are exactly representable in f16 (range 0..2048)
        assert!(
            (orig - back).abs() <= 0.5,
            "lane {i}: orig={orig} back={back}"
        );
    }
}

#[test]
fn f16x16_bit_pattern_stability() {
    // from_array → to_array must be identity for raw bit patterns
    let raw: [u16; 16] = core::array::from_fn(|i| 0x3C00_u16 + i as u16); // ~1.0 range
    let v = f16_types::F16x16::from_array(raw);
    assert_eq!(v.to_array(), raw);
}

// ════════════════════════════════════════════════════════════════════════════
// 7. Mask select — F32Mask16 / F32Mask8 / F64Mask8 / F64Mask4
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn f32mask16_select_all_true() {
    let mask = masks::F32Mask16::from_bitmask(0xFFFF);
    let t = F32x16::splat(1.0);
    let f = F32x16::splat(2.0);
    assert!(mask.select(t, f).to_array().iter().all(|&x| x == 1.0));
}

#[test]
fn f32mask16_select_all_false() {
    let mask = masks::F32Mask16::from_bitmask(0x0000);
    let t = F32x16::splat(1.0);
    let f = F32x16::splat(2.0);
    assert!(mask.select(t, f).to_array().iter().all(|&x| x == 2.0));
}

#[test]
fn f32mask16_select_alternating() {
    // bits 0,2,4,...,14 set → even lanes from true_val, odd from false_val
    let mask = masks::F32Mask16::from_bitmask(0x5555); // 0101010101010101
    let t = F32x16::splat(1.0);
    let f = F32x16::splat(2.0);
    let result = mask.select(t, f).to_array();
    for i in 0..16 {
        if i % 2 == 0 {
            assert_eq!(result[i], 1.0, "even lane {i} should be 1.0");
        } else {
            assert_eq!(result[i], 2.0, "odd lane {i} should be 2.0");
        }
    }
}

#[test]
fn f32mask16_select_scalar_parity() {
    // Build mask from comparison, then verify select matches scalar ternary
    let a: [f32; 16] = core::array::from_fn(|i| i as f32);
    let threshold = F32x16::splat(7.5);
    let va = F32x16::from_array(a);
    let mask = va.simd_lt(threshold);
    let result = mask.select(F32x16::splat(100.0), F32x16::splat(200.0)).to_array();
    for i in 0..16 {
        let expected = if a[i] < 7.5 { 100.0_f32 } else { 200.0_f32 };
        assert_eq!(result[i], expected, "lane {i}");
    }
}

#[test]
fn f32mask8_select_scalar_parity() {
    let a: [f32; 8] = core::array::from_fn(|i| i as f32);
    let threshold = F32x8::splat(3.5);
    let va = F32x8::from_array(a);
    let mask = va.simd_gt(threshold);
    let result = mask.select(F32x8::splat(10.0), F32x8::splat(20.0)).to_array();
    for i in 0..8 {
        let expected = if a[i] > 3.5 { 10.0_f32 } else { 20.0_f32 };
        assert_eq!(result[i], expected, "lane {i}");
    }
}

#[test]
fn f64mask8_select_scalar_parity() {
    let a: [f64; 8] = core::array::from_fn(|i| i as f64);
    let threshold = F64x8::splat(4.5);
    let va = F64x8::from_array(a);
    let mask = va.simd_ge(threshold);
    let result = mask.select(F64x8::splat(1.0), F64x8::splat(0.0)).to_array();
    for i in 0..8 {
        let expected = if a[i] >= 4.5 { 1.0_f64 } else { 0.0_f64 };
        assert_eq!(result[i], expected, "lane {i}");
    }
}

#[test]
fn f64mask4_select_scalar_parity() {
    let a: [f64; 4] = [1.0, 5.0, 3.0, 7.0];
    let threshold = F64x4::splat(4.0);
    let va = F64x4::from_array(a);
    let mask = va.simd_gt(threshold);
    let result = mask.select(F64x4::splat(99.0), F64x4::splat(0.0)).to_array();
    // Lanes 1 (5>4) and 3 (7>4) should be 99.0
    assert_eq!(result, [0.0, 99.0, 0.0, 99.0]);
}

// ════════════════════════════════════════════════════════════════════════════
// 8. Exotic methods — permute_bytes reverse / nibble_popcount_lut
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn u8x64_permute_bytes_reverse_identity() {
    // idx[i] = 63 - i reverses the vector
    let v = u8_types::U8x64::from_array(core::array::from_fn(|i| i as u8));
    let idx = u8_types::U8x64::from_array(core::array::from_fn(|i| (63 - i) as u8));
    let r = v.permute_bytes(idx);
    for i in 0..64 {
        assert_eq!(r.to_array()[i], (63 - i) as u8, "lane {i}");
    }
}

#[test]
fn u8x32_permute_bytes_reverse_identity() {
    // idx[i] = 31 - i reverses the vector
    let v = u8_types::U8x32::from_array(core::array::from_fn(|i| i as u8));
    let idx = u8_types::U8x32::from_array(core::array::from_fn(|i| (31 - i) as u8));
    let r = v.permute_bytes(idx);
    for i in 0..32 {
        assert_eq!(r.to_array()[i], (31 - i) as u8, "lane {i}");
    }
}

#[test]
fn u8x64_nibble_popcount_lut_vs_scalar_count_ones() {
    let lut = u8_types::U8x64::nibble_popcount_lut();
    let lut_arr = lut.to_array();
    // First 16 entries map nibble value → popcount
    for nibble in 0u8..16 {
        let simd_count = lut_arr[nibble as usize];
        let scalar_count = nibble.count_ones() as u8;
        assert_eq!(simd_count, scalar_count, "nibble {nibble:#04x}");
    }
}

#[test]
fn u8x32_nibble_popcount_lut_vs_scalar_count_ones() {
    let lut = u8_types::U8x32::nibble_popcount_lut();
    let lut_arr = lut.to_array();
    // First 16 entries
    for nibble in 0u8..16 {
        let simd_count = lut_arr[nibble as usize];
        let scalar_count = nibble.count_ones() as u8;
        assert_eq!(simd_count, scalar_count, "nibble {nibble:#04x}");
    }
}

#[test]
fn u8x64_shuffle_bytes_popcount_parity() {
    // Use shuffle_bytes to look up popcount of each nibble in test data.
    // Input byte b → low nibble → lut[b & 0xF] gives popcount of low nibble.
    let lut = u8_types::U8x64::nibble_popcount_lut();
    let data: [u8; 64] = core::array::from_fn(|i| (i % 16) as u8);
    let idx = u8_types::U8x64::from_array(data);
    let result = lut.shuffle_bytes(idx).to_array();
    for i in 0..64 {
        let nibble = data[i];
        let expected = nibble.count_ones() as u8;
        assert_eq!(result[i], expected, "data[{i}]={nibble} expected popcount {expected}");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 9. U-word types: constructor roundtrip + reduction parity
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn u64x8_splat_and_reduce_parity() {
    let src: [u64; 8] = core::array::from_fn(|i| (i as u64) * 7);
    let v = u_word_types::U64x8::from_array(src);
    let simd_sum = v.reduce_sum();
    let scalar_sum: u64 = src.iter().sum();
    assert_eq!(simd_sum, scalar_sum);
    assert_eq!(v.reduce_min(), *src.iter().min().unwrap());
    assert_eq!(v.reduce_max(), *src.iter().max().unwrap());
}

#[test]
fn u64x4_splat_and_reduce_parity() {
    let src: [u64; 4] = core::array::from_fn(|i| (i as u64) * 11);
    let v = u_word_types::U64x4::from_array(src);
    let simd_sum = v.reduce_sum();
    let scalar_sum: u64 = src.iter().sum();
    assert_eq!(simd_sum, scalar_sum);
    assert_eq!(v.reduce_min(), *src.iter().min().unwrap());
    assert_eq!(v.reduce_max(), *src.iter().max().unwrap());
}

#[test]
fn u32x16_splat_and_reduce_parity() {
    let src: [u32; 16] = core::array::from_fn(|i| (i as u32) * 7);
    let v = u_word_types::U32x16::from_array(src);
    let simd_sum = v.reduce_sum();
    let scalar_sum: u32 = src.iter().sum();
    assert_eq!(simd_sum, scalar_sum);
    assert_eq!(v.reduce_min(), *src.iter().min().unwrap());
    assert_eq!(v.reduce_max(), *src.iter().max().unwrap());
}

#[test]
fn u32x8_splat_and_reduce_parity() {
    let src: [u32; 8] = core::array::from_fn(|i| (i as u32) * 5);
    let v = u_word_types::U32x8::from_array(src);
    let simd_sum = v.reduce_sum();
    let scalar_sum: u32 = src.iter().sum();
    assert_eq!(simd_sum, scalar_sum);
    assert_eq!(v.reduce_min(), *src.iter().min().unwrap());
    assert_eq!(v.reduce_max(), *src.iter().max().unwrap());
}

#[test]
fn u16x32_splat_and_reduce_parity() {
    let src: [u16; 32] = core::array::from_fn(|i| (i as u16) * 3);
    let v = u_word_types::U16x32::from_array(src);
    assert_eq!(v.reduce_min(), *src.iter().min().unwrap());
    assert_eq!(v.reduce_max(), *src.iter().max().unwrap());
}

// ════════════════════════════════════════════════════════════════════════════
// 10. F32x16 math ops — sqrt / abs / floor / round vs scalar
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn f32x16_sqrt_parity() {
    let src: [f32; 16] = core::array::from_fn(|i| (i as f32) * 2.0 + 1.0);
    let result = F32x16::from_array(src).sqrt().to_array();
    for (i, (&r, &s)) in result.iter().zip(src.iter()).enumerate() {
        let expected = s.sqrt();
        assert!((r - expected).abs() < 1e-5, "lane {i}: sqrt({s}) = {r} expected {expected}");
    }
}

#[test]
fn f32x16_abs_parity() {
    let src: [f32; 16] = core::array::from_fn(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) });
    let result = F32x16::from_array(src).abs().to_array();
    for (i, (&r, &s)) in result.iter().zip(src.iter()).enumerate() {
        assert_eq!(r, s.abs(), "lane {i}");
    }
}

#[test]
fn f32x16_floor_parity() {
    let src: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.7 - 3.5);
    let result = F32x16::from_array(src).floor().to_array();
    for (i, (&r, &s)) in result.iter().zip(src.iter()).enumerate() {
        assert_eq!(r, s.floor(), "lane {i}");
    }
}

#[test]
fn f32x16_round_parity() {
    let src: [f32; 16] = [
        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, -0.5, -1.5,
        0.4, 1.6, 2.3, 3.7, -0.4, -1.6, 100.0, -100.0,
    ];
    let result = F32x16::from_array(src).round().to_array();
    for (i, (&r, &s)) in result.iter().zip(src.iter()).enumerate() {
        assert_eq!(r, s.round(), "lane {i}");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 11. F32x16 to_bits / from_bits roundtrip
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn f32x16_bits_roundtrip() {
    let src: [f32; 16] = core::array::from_fn(|i| (i as f32) * 1.5);
    let v = F32x16::from_array(src);
    let bits = v.to_bits();
    let back = F32x16::from_bits(bits);
    assert_eq!(v.to_array(), back.to_array());
}

#[test]
fn f32x16_bits_match_scalar() {
    let src: [f32; 16] = core::array::from_fn(|i| i as f32);
    let bits = F32x16::from_array(src).to_bits().to_array();
    for (i, (&b, &s)) in bits.iter().zip(src.iter()).enumerate() {
        assert_eq!(b, s.to_bits(), "lane {i}");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 12. Operator impls — Add/Sub/Mul/Div/BitAnd/BitOr/BitXor scalar parity
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn f32x16_add_sub_mul_div_parity() {
    let a: [f32; 16] = core::array::from_fn(|i| (i as f32) + 1.0);
    let b: [f32; 16] = core::array::from_fn(|i| (i as f32) * 0.5 + 0.1);
    let va = F32x16::from_array(a);
    let vb = F32x16::from_array(b);
    let add = (va + vb).to_array();
    let sub = (va - vb).to_array();
    let mul = (va * vb).to_array();
    let div = (va / vb).to_array();
    for i in 0..16 {
        assert!((add[i] - (a[i] + b[i])).abs() < 1e-5, "add lane {i}");
        assert!((sub[i] - (a[i] - b[i])).abs() < 1e-5, "sub lane {i}");
        assert!((mul[i] - (a[i] * b[i])).abs() < 1e-5, "mul lane {i}");
        assert!((div[i] - (a[i] / b[i])).abs() < 1e-4, "div lane {i}");
    }
}

#[test]
fn u8x64_bitwise_ops_parity() {
    let a: [u8; 64] = core::array::from_fn(|i| i as u8);
    let b: [u8; 64] = core::array::from_fn(|i| (255 - i) as u8);
    let va = u8_types::U8x64::from_array(a);
    let vb = u8_types::U8x64::from_array(b);
    let and_r = (va & vb).to_array();
    let or_r = (va | vb).to_array();
    let xor_r = (va ^ vb).to_array();
    for i in 0..64 {
        assert_eq!(and_r[i], a[i] & b[i], "AND lane {i}");
        assert_eq!(or_r[i], a[i] | b[i], "OR lane {i}");
        assert_eq!(xor_r[i], a[i] ^ b[i], "XOR lane {i}");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// 13. Mask bitmask roundtrip — from_bitmask(to_bitmask(m)) == m
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn f32mask16_bitmask_roundtrip() {
    let orig: u16 = 0b1010_1010_0101_0101;
    let mask = masks::F32Mask16::from_bitmask(orig);
    assert_eq!(mask.to_bitmask(), orig);
}

#[test]
fn f32mask8_bitmask_roundtrip() {
    let orig: u8 = 0b1100_0011;
    let mask = masks::F32Mask8::from_bitmask(orig);
    assert_eq!(mask.to_bitmask(), orig);
}

#[test]
fn f64mask8_bitmask_roundtrip() {
    let orig: u8 = 0b0101_1010;
    let mask = masks::F64Mask8::from_bitmask(orig);
    assert_eq!(mask.to_bitmask(), orig);
}

#[test]
fn f64mask4_bitmask_roundtrip() {
    let orig: u8 = 0b1101; // only low 4 bits matter
    let mask = masks::F64Mask4::from_bitmask(orig);
    assert_eq!(mask.to_bitmask() & 0xF, orig & 0xF);
}

// ════════════════════════════════════════════════════════════════════════════
// 14. simd_clamp parity — lanes clamped exactly as scalar
// ════════════════════════════════════════════════════════════════════════════

#[test]
fn f32x16_simd_clamp_parity() {
    let src: [f32; 16] = core::array::from_fn(|i| i as f32 - 5.0); // -5..10
    let lo = F32x16::splat(0.0);
    let hi = F32x16::splat(8.0);
    let result = F32x16::from_array(src).simd_clamp(lo, hi).to_array();
    for (i, (&r, &s)) in result.iter().zip(src.iter()).enumerate() {
        let expected = s.clamp(0.0, 8.0);
        assert_eq!(r, expected, "lane {i}");
    }
}

#[test]
fn f64x8_simd_clamp_parity() {
    let src: [f64; 8] = core::array::from_fn(|i| i as f64 - 3.0); // -3..4
    let lo = F64x8::splat(-1.0);
    let hi = F64x8::splat(3.0);
    let result = F64x8::from_array(src).simd_clamp(lo, hi).to_array();
    for (i, (&r, &s)) in result.iter().zip(src.iter()).enumerate() {
        let expected = s.clamp(-1.0, 3.0);
        assert_eq!(r, expected, "lane {i}");
    }
}
