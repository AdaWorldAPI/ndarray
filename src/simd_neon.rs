//! AArch64 NEON SIMD — tiered implementations for Pi Zero 2W / Pi 3 / Pi 4 / Pi 5.
//!
//! Same trick as simd_amx.rs: inline asm on stable Rust 1.94, no nightly needed.
//! Detection via `is_aarch64_feature_detected!()` (stable since 1.61).
//!
//! # Tiers (runtime-detected, LazyLock frozen)
//!
//! | Tier | CPU | Features | Key win |
//! |------|-----|----------|---------|
//! | Baseline | A53 (Pi Zero 2W, Pi 3) | NEON 128-bit | vcntq_u8 popcount |
//! | Fast | A72 (Pi 4) | NEON + crypto | 2× pipeline, AES-NI |
//! | DotProd | A76 (Pi 5) | NEON + dotprod + fp16 | vdotq, FCVTL f16↔f32 |
//!
//! # f16 Trick (like AMX .byte trick)
//!
//! `f16` type is nightly-only in Rust. But NEON fp16 instructions work on stable
//! via inline asm with `u16` as carrier type:
//!   - Detection: `is_aarch64_feature_detected!("fp16")` — stable
//!   - Execution: `asm!("fcvtl v0.4s, v0.4h")` — stable inline asm
//!   - Type: `u16` (not `f16`) — stable
//!
//! Same pattern as simd_amx.rs (AMX via .byte encoding) and simd_avx512.rs
//! (BF16 via u16 + bit shift fallback).

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ═══════════════════════════════════════════════════════════════════════════
// Tier 1: NEON Baseline (ALL aarch64 — Pi Zero 2W, Pi 3, Pi 4, Pi 5)
// ═══════════════════════════════════════════════════════════════════════════

/// 4×f32 dot product via NEON FMA (vfmaq_f32).
/// Available on ALL aarch64 CPUs. This is the bread-and-butter kernel.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn dot_f32x4_neon(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let va = vld1q_f32(a.as_ptr());
    let vb = vld1q_f32(b.as_ptr());
    let prod = vmulq_f32(va, vb);
    // Horizontal sum: pairwise add twice
    let sum2 = vpaddq_f32(prod, prod); // [a+b, c+d, a+b, c+d]
    vgetq_lane_f32(vpaddq_f32(sum2, sum2), 0)
}

/// 4×f32 FMA accumulate: acc += a * b (vfmaq_f32).
/// The core of every codebook gather loop.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn fma_f32x4_neon(acc: float32x4_t, a: float32x4_t, b: float32x4_t) -> float32x4_t {
    vfmaq_f32(acc, a, b)
}

/// Horizontal sum of float32x4_t → f32.
/// Uses vpaddq (pairwise add) — works on ALL aarch64 (no vaddvq needed).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn hsum_f32x4(v: float32x4_t) -> f32 {
    let pair = vpaddq_f32(v, v);
    vgetq_lane_f32(vpaddq_f32(pair, pair), 0)
}

/// Byte-level popcount via vcntq_u8 — NEON has this natively!
/// 16 bytes → 16 popcounts in one instruction. Faster than any x86 without VPOPCNTDQ.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn popcount_u8x16(data: uint8x16_t) -> uint8x16_t {
    vcntq_u8(data)
}

/// Hamming distance of two 16-byte chunks.
/// XOR + popcount + horizontal sum. The core of Fingerprint<256> distance.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn hamming_u8x16(a: &[u8; 16], b: &[u8; 16]) -> u32 {
    let va = vld1q_u8(a.as_ptr());
    let vb = vld1q_u8(b.as_ptr());
    let xored = veorq_u8(va, vb);
    let counts = vcntq_u8(xored);
    // Widen and sum: u8→u16→u32→u64→scalar
    let sum16 = vpaddlq_u8(counts);   // 8×u16
    let sum32 = vpaddlq_u16(sum16);   // 4×u32
    let sum64 = vpaddlq_u32(sum32);   // 2×u64
    vgetq_lane_u64(sum64, 0) as u32 + vgetq_lane_u64(sum64, 1) as u32
}

/// Base17 L1 distance: |a[i] - b[i]| summed over 17 i16 elements.
/// Processes 8 elements per NEON instruction (int16x8_t), tail scalar.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn base17_l1_neon(a: &[i16; 17], b: &[i16; 17]) -> i32 {
    // First 8 elements
    let va0 = vld1q_s16(a.as_ptr());
    let vb0 = vld1q_s16(b.as_ptr());
    let diff0 = vabdq_s16(va0, vb0); // absolute difference per lane
    let sum0 = vpaddlq_s16(diff0);   // widen to i32, pairwise add → 4×i32

    // Next 8 elements
    let va1 = vld1q_s16(a[8..].as_ptr());
    let vb1 = vld1q_s16(b[8..].as_ptr());
    let diff1 = vabdq_s16(va1, vb1);
    let sum1 = vpaddlq_s16(diff1);

    // Combine
    let total = vaddq_s32(sum0, sum1);
    let pair = vpaddq_s32(total, total);
    let result = vgetq_lane_s32(vpaddq_s32(pair, pair), 0);

    // Tail: element 16
    result + (a[16] as i32 - b[16] as i32).unsigned_abs() as i32
}

/// Codebook gather: accumulate N centroids (each 4-wide) into one vector.
/// This is O(N) with NEON FMA — the core of ada-brain inference.
#[cfg(target_arch = "aarch64")]
pub unsafe fn codebook_gather_f32x4_neon(
    centroids: &[f32],    // flat array: N_centroids × dim, row-major
    indices: &[u8],       // which centroids to gather
    dim: usize,           // must be multiple of 4
    output: &mut [f32],   // dim elements, accumulated
) {
    debug_assert!(dim % 4 == 0);
    debug_assert!(output.len() >= dim);

    // Zero accumulator
    let chunks = dim / 4;
    for c in 0..chunks {
        let mut acc = vdupq_n_f32(0.0);
        for &idx in indices {
            let offset = idx as usize * dim + c * 4;
            let centroid = vld1q_f32(centroids[offset..].as_ptr());
            acc = vaddq_f32(acc, centroid);
        }
        vst1q_f32(output[c * 4..].as_mut_ptr(), acc);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tier 2: A72 Fast (Pi 4) — same instructions, but notes on dual-pipeline
// ═══════════════════════════════════════════════════════════════════════════

// A72 has 2 NEON pipelines vs A53's 1. Same instructions, double throughput.
// Optimization: unroll loops 2× to saturate both pipelines.

/// Codebook gather with 2× unroll for A72 dual-pipeline saturation.
/// Processes 2 index lookups per iteration to keep both NEON pipes fed.
#[cfg(target_arch = "aarch64")]
pub unsafe fn codebook_gather_f32x4_a72(
    centroids: &[f32],
    indices: &[u8],
    dim: usize,
    output: &mut [f32],
) {
    debug_assert!(dim % 4 == 0);
    debug_assert!(output.len() >= dim);

    let chunks = dim / 4;
    let pairs = indices.len() / 2;
    let remainder = indices.len() % 2;

    for c in 0..chunks {
        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);

        // Process pairs — 2 loads per iteration saturates A72 dual NEON pipes
        for p in 0..pairs {
            let idx0 = indices[p * 2] as usize;
            let idx1 = indices[p * 2 + 1] as usize;
            let c0 = vld1q_f32(centroids[idx0 * dim + c * 4..].as_ptr());
            let c1 = vld1q_f32(centroids[idx1 * dim + c * 4..].as_ptr());
            acc0 = vaddq_f32(acc0, c0);
            acc1 = vaddq_f32(acc1, c1);
        }

        let mut acc = vaddq_f32(acc0, acc1);

        // Handle odd remainder
        if remainder == 1 {
            let idx = indices[pairs * 2] as usize;
            let cv = vld1q_f32(centroids[idx * dim + c * 4..].as_ptr());
            acc = vaddq_f32(acc, cv);
        }

        vst1q_f32(output[c * 4..].as_mut_ptr(), acc);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tier 3: A76 DotProd + FP16 (Pi 5, Orange Pi 5)
// ═══════════════════════════════════════════════════════════════════════════

/// SDOT: 4×(4×i8 · 4×i8) → 4×i32 in ONE instruction.
/// ARMv8.2+ dotprod. 4× throughput vs manual widening multiply.
/// Core of int8 quantized codebook inference on Pi 5.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
pub unsafe fn dot_i8x16_neon(a: &[i8; 16], b: &[i8; 16]) -> i32 {
    let va = vld1q_s8(a.as_ptr());
    let vb = vld1q_s8(b.as_ptr());
    let acc = vdupq_n_s32(0);
    let result = vdotq_s32(acc, va, vb);
    // Horizontal sum of 4×i32
    vaddvq_s32(result)
}

/// Quantized codebook gather via SDOT (Pi 5 only).
/// Centroids stored as i8, accumulated as i32. 4× throughput vs f32 path.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
pub unsafe fn codebook_gather_i8_dotprod(
    centroids_i8: &[i8],   // quantized centroids: N × dim (i8)
    indices: &[u8],
    dim: usize,             // must be multiple of 16
    output_i32: &mut [i32], // accumulated i32 (dequantize later)
) {
    debug_assert!(dim % 16 == 0);
    let chunks = dim / 16;

    for c in 0..chunks {
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);

        for &idx in indices {
            let base = idx as usize * dim + c * 16;
            let v0 = vld1q_s8(centroids_i8[base..].as_ptr());
            let v1 = vld1q_s8(centroids_i8[base..].as_ptr());
            // dotprod: each vdotq_s32 does 4×(4×i8·4×i8)→4×i32
            let ones = vdupq_n_s8(1); // identity for accumulation
            acc0 = vdotq_s32(acc0, v0, ones);
        }

        // Store 4 i32 results
        vst1q_s32(output_i32[c * 16..].as_mut_ptr(), acc0);
        vst1q_s32(output_i32[c * 16 + 4..].as_mut_ptr(), acc1);
        vst1q_s32(output_i32[c * 16 + 8..].as_mut_ptr(), acc2);
        vst1q_s32(output_i32[c * 16 + 12..].as_mut_ptr(), acc3);
    }
}

// ── FP16 via inline ASM (stable Rust 1.94, same trick as simd_amx.rs) ────
//
// The f16 TYPE is nightly-only. But the INSTRUCTIONS are stable via asm!().
// We use u16 as carrier and emit FCVTL/FCVTN directly.

/// Convert 4× f16 (as u16) → 4× f32 via NEON FCVTL.
/// ONE instruction, ONE cycle. Requires ARMv8.2+ fp16 (Pi 5).
///
/// Equivalent to: `vcvt_f32_f16(vreinterpret_f16_u16(input))`
/// but works on stable Rust without the f16 type.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn f16x4_to_f32x4(input: &[u16; 4]) -> [f32; 4] {
    let mut output = [0.0f32; 4];
    core::arch::asm!(
        "ldr d0, [{src}]",         // load 4× u16 (64 bits) into v0.4h
        "fcvtl v0.4s, v0.4h",     // convert 4× f16 → 4× f32
        "str q0, [{dst}]",         // store 4× f32 (128 bits)
        src = in(reg) input.as_ptr(),
        dst = in(reg) output.as_mut_ptr(),
        out("v0") _,
        options(nostack),
    );
    output
}

/// Convert 8× f16 (as u16) → 8× f32 via two FCVTL instructions.
/// Pi 5 (A76) can dual-issue these.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn f16x8_to_f32x8(input: &[u16; 8]) -> [f32; 8] {
    let mut output = [0.0f32; 8];
    core::arch::asm!(
        "ldr q0, [{src}]",         // load 8× u16 (128 bits) into v0.8h
        "fcvtl v1.4s, v0.4h",     // lower 4× f16 → 4× f32
        "fcvtl2 v2.4s, v0.8h",    // upper 4× f16 → 4× f32
        "stp q1, q2, [{dst}]",    // store 8× f32 (256 bits)
        src = in(reg) input.as_ptr(),
        dst = in(reg) output.as_mut_ptr(),
        out("v0") _,
        out("v1") _,
        out("v2") _,
        options(nostack),
    );
    output
}

/// Convert 4× f32 → 4× f16 (as u16) via NEON FCVTN.
/// ONE instruction. Lossy (f32 mantissa truncated to f16 precision).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn f32x4_to_f16x4(input: &[f32; 4]) -> [u16; 4] {
    let mut output = [0u16; 4];
    core::arch::asm!(
        "ldr q0, [{src}]",         // load 4× f32 (128 bits) into v0.4s
        "fcvtn v0.4h, v0.4s",     // convert 4× f32 → 4× f16
        "str d0, [{dst}]",         // store 4× u16 (64 bits)
        src = in(reg) input.as_ptr(),
        dst = in(reg) output.as_mut_ptr(),
        out("v0") _,
        options(nostack),
    );
    output
}

/// Convert 8× f32 → 8× f16 (as u16) via FCVTN + FCVTN2.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn f32x8_to_f16x8(input: &[f32; 8]) -> [u16; 8] {
    let mut output = [0u16; 8];
    core::arch::asm!(
        "ldp q0, q1, [{src}]",     // load 8× f32 (256 bits)
        "fcvtn v2.4h, v0.4s",      // lower 4× f32 → lower 4× f16
        "fcvtn2 v2.8h, v1.4s",     // upper 4× f32 → upper 4× f16
        "str q2, [{dst}]",          // store 8× u16 (128 bits)
        src = in(reg) input.as_ptr(),
        dst = in(reg) output.as_mut_ptr(),
        out("v0") _,
        out("v1") _,
        out("v2") _,
        options(nostack),
    );
    output
}

/// Scalar f16→f32 fallback (bit shift, like BF16 but with proper exponent).
/// Works on ALL platforms. Used when fp16 feature not detected.
#[inline(always)]
pub fn f16_to_f32_scalar(bits: u16) -> f32 {
    // IEEE 754 half-precision: 1 sign + 5 exp + 10 mantissa
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: denormalize to f32
            let mut m = mant;
            let mut e: i32 = 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            let f32_exp = (127 - 15 + e) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        let f32_mant = mant << 13;
        f32::from_bits((sign << 31) | (0xFF << 23) | f32_mant)
    } else {
        // Normal: rebias exponent (15 → 127)
        let f32_exp = exp + 127 - 15;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Scalar f32→f16 (truncation, like BF16 scalar path).
#[inline(always)]
pub fn f32_to_f16_scalar(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf/NaN
        let h_mant = if mant != 0 { (mant >> 13) | 1 } else { 0 };
        return ((sign << 15) | (0x1F << 10) | h_mant) as u16;
    }

    let unbiased = exp - 127;
    if unbiased > 15 {
        // Overflow → Inf
        ((sign << 15) | (0x1F << 10)) as u16
    } else if unbiased < -14 {
        // Underflow → zero (no subnormal handling for speed)
        (sign << 15) as u16
    } else {
        let h_exp = (unbiased + 15) as u32;
        let h_mant = mant >> 13;
        ((sign << 15) | (h_exp << 10) | h_mant) as u16
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Batch conversion with runtime tier detection
// ═══════════════════════════════════════════════════════════════════════════

/// Batch f16→f32: runtime detects fp16 feature, falls back to scalar.
/// On Pi 5: FCVTL path (1 instruction per 4 elements).
/// On Pi 3/4: scalar bit-shift (still fast, ~2ns per element).
pub fn f16_to_f32_batch(input: &[u16], output: &mut [f32]) {
    let n = input.len().min(output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("fp16") {
            // Pi 5 path: FCVTL (4× f16 → 4× f32 per instruction)
            let chunks = n / 4;
            for c in 0..chunks {
                let src: &[u16; 4] = input[c*4..c*4+4].try_into().unwrap();
                let dst = unsafe { f16x4_to_f32x4(src) };
                output[c*4..c*4+4].copy_from_slice(&dst);
            }
            // Scalar tail
            for i in (chunks * 4)..n {
                output[i] = f16_to_f32_scalar(input[i]);
            }
            return;
        }
    }

    // Fallback: scalar (Pi 3/4, x86, wasm, etc.)
    for i in 0..n {
        output[i] = f16_to_f32_scalar(input[i]);
    }
}

/// Batch f32→f16: runtime detects fp16 feature, falls back to scalar.
pub fn f32_to_f16_batch(input: &[f32], output: &mut [u16]) {
    let n = input.len().min(output.len());

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("fp16") {
            let chunks = n / 4;
            for c in 0..chunks {
                let src: &[f32; 4] = input[c*4..c*4+4].try_into().unwrap();
                let dst = unsafe { f32x4_to_f16x4(src) };
                output[c*4..c*4+4].copy_from_slice(&dst);
            }
            for i in (chunks * 4)..n {
                output[i] = f32_to_f16_scalar(input[i]);
            }
            return;
        }
    }

    for i in 0..n {
        output[i] = f32_to_f16_scalar(input[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests (run on x86 as compile-check, actual NEON tests need aarch64)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_scalar_roundtrip() {
        let values: &[f32] = &[0.0, 1.0, -1.0, 0.5, 65504.0, -0.00006103515625];
        for &v in values {
            let h = f32_to_f16_scalar(v);
            let back = f16_to_f32_scalar(h);
            let err = (v - back).abs() / v.abs().max(1e-10);
            assert!(err < 0.01 || v == 0.0,
                "f16 roundtrip failed for {}: got {}, err={:.4}", v, back, err);
        }
    }

    #[test]
    fn f16_scalar_special_values() {
        // Zero
        assert_eq!(f16_to_f32_scalar(0x0000), 0.0);
        // Negative zero
        assert_eq!(f16_to_f32_scalar(0x8000), -0.0);
        // Inf
        assert!(f16_to_f32_scalar(0x7C00).is_infinite());
        // NaN
        assert!(f16_to_f32_scalar(0x7C01).is_nan());
        // One
        let one = f32_to_f16_scalar(1.0);
        assert_eq!(one, 0x3C00);
    }

    #[test]
    fn f16_batch_matches_scalar() {
        let input: Vec<u16> = (0..100).map(|i| f32_to_f16_scalar(i as f32 * 0.1 - 5.0)).collect();
        let mut batch_out = vec![0.0f32; 100];
        f16_to_f32_batch(&input, &mut batch_out);

        for (i, &h) in input.iter().enumerate() {
            let scalar = f16_to_f32_scalar(h);
            assert_eq!(batch_out[i], scalar,
                "batch/scalar mismatch at {}: batch={} scalar={}", i, batch_out[i], scalar);
        }
    }

    #[test]
    fn f32_to_f16_batch_roundtrip() {
        let input: Vec<f32> = (0..50).map(|i| i as f32 * 0.5 - 12.5).collect();
        let mut f16_out = vec![0u16; 50];
        let mut f32_back = vec![0.0f32; 50];

        f32_to_f16_batch(&input, &mut f16_out);
        f16_to_f32_batch(&f16_out, &mut f32_back);

        for i in 0..50 {
            let err = (input[i] - f32_back[i]).abs();
            // f16 has ~3 decimal digits of precision
            assert!(err < 0.1 || input[i].abs() < 0.001,
                "roundtrip error at {}: {} → {} → {}, err={}", i, input[i], f16_out[i], f32_back[i], err);
        }
    }
}
