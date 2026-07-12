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
    let sum16 = vpaddlq_u8(counts); // 8×u16
    let sum32 = vpaddlq_u16(sum16); // 4×u32
    let sum64 = vpaddlq_u32(sum32); // 2×u64
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
    let sum0 = vpaddlq_s16(diff0); // widen to i32, pairwise add → 4×i32

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
    centroids: &[f32],  // flat array: N_centroids × dim, row-major
    indices: &[u8],     // which centroids to gather
    dim: usize,         // must be multiple of 4
    output: &mut [f32], // dim elements, accumulated
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
pub unsafe fn codebook_gather_f32x4_a72(centroids: &[f32], indices: &[u8], dim: usize, output: &mut [f32]) {
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

/// int8 dot product of two 16-lane chunks → i32.
///
/// Stable-Rust widening path: `vmull_s8` (8×(i8·i8)→i16 per half) then
/// `vpaddlq_s16` (pairwise widen i16→i32) and a horizontal `vaddvq_s32`.
/// Bit-identical result to the ARMv8.2 `SDOT` instruction, but compiles on
/// stable (the `vdotq_s32` intrinsic is nightly-only, issue #117224) and runs
/// on **all** aarch64 — no `dotprod` feature required. Max |product| = 16384,
/// pairwise sums stay well inside i32, so no overflow for any i8 input.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn dot_i8x16_neon(a: &[i8; 16], b: &[i8; 16]) -> i32 {
    let va = vld1q_s8(a.as_ptr());
    let vb = vld1q_s8(b.as_ptr());
    // 8×i16 products for the low and high halves.
    let plo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
    let phi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
    // Widen i16→i32 (pairwise) so the accumulation never overflows, then reduce.
    vaddvq_s32(vaddq_s32(vpaddlq_s16(plo), vpaddlq_s16(phi)))
}

/// Quantized codebook gather: element-wise widen-accumulate the selected i8
/// centroids into an i32 output of length `dim` (`dim` a multiple of 16).
///
/// The i32 counterpart of [`codebook_gather_f32x4_neon`]: for each output lane
/// `k`, `output_i32[k] = Σ_idx centroids_i8[idx*dim + k]`, widened i8→i32 via
/// `vmovl_s8` + `vaddw_s16`. Stable on all aarch64 (no `dotprod` intrinsic).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub unsafe fn codebook_gather_i8_dotprod(
    centroids_i8: &[i8], // quantized centroids: N × dim (i8)
    indices: &[u8],
    dim: usize,             // must be multiple of 16
    output_i32: &mut [i32], // accumulated i32 (dequantize later)
) {
    debug_assert!(dim % 16 == 0);
    debug_assert!(output_i32.len() >= dim);
    let chunks = dim / 16;

    for c in 0..chunks {
        // Four i32x4 accumulators cover the 16 i8 lanes of this chunk.
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);
        let mut acc2 = vdupq_n_s32(0);
        let mut acc3 = vdupq_n_s32(0);

        for &idx in indices {
            let base = idx as usize * dim + c * 16;
            let v = vld1q_s8(centroids_i8[base..].as_ptr()); // 16 i8
            let lo = vmovl_s8(vget_low_s8(v)); // lanes 0..8  → i16
            let hi = vmovl_s8(vget_high_s8(v)); // lanes 8..16 → i16
            acc0 = vaddw_s16(acc0, vget_low_s16(lo));
            acc1 = vaddw_s16(acc1, vget_high_s16(lo));
            acc2 = vaddw_s16(acc2, vget_low_s16(hi));
            acc3 = vaddw_s16(acc3, vget_high_s16(hi));
        }

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
                let src: &[u16; 4] = input[c * 4..c * 4 + 4].try_into().unwrap();
                let dst = unsafe { f16x4_to_f32x4(src) };
                output[c * 4..c * 4 + 4].copy_from_slice(&dst);
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
                let src: &[f32; 4] = input[c * 4..c * 4 + 4].try_into().unwrap();
                let dst = unsafe { f32x4_to_f16x4(src) };
                output[c * 4..c * 4 + 4].copy_from_slice(&dst);
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
// NEON-backed F32x16 / F64x8 — paired loads, NOT scalar fallback
// ═══════════════════════════════════════════════════════════════════════════
//
// Burn parity item 9 (verified 2026-04-30, agent A7): on aarch64, `F32x16`
// previously dispatched to `simd::scalar` mod (element-wise [f32;16] loop).
// This module provides a real NEON implementation backed by 4× `float32x4_t`
// for `F32x16` and 4× `float64x2_t` for `F64x8`. Hot-path ops (add, sub, mul,
// div, mul_add via `vfmaq_f32`/`vfmaq_f64`, splat, vld1q_*, vst1q_*) compile
// to a single NEON instruction per pair. `simd.rs` re-exports these for
// `target_arch = "aarch64"` ahead of the scalar fallback module.
//
// API matches `simd_avx2::F32x16` (the "dual-tuple" pattern). Methods that
// don't have a direct NEON counterpart (comparisons, reduce_min/max,
// to_bits/from_bits, cast_i32) round-trip through `to_array` — same shape
// as the AVX2 polyfill, so consumer code on aarch64 gets the same
// correctness with vectorized arithmetic kernels.

#[cfg(target_arch = "aarch64")]
pub mod aarch64_simd {
    use super::*;
    use core::fmt;
    use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    // Integer types come from the scalar fallback in simd.rs — they aren't on
    // the perf-critical f32 BLAS-1 / VML path that this module accelerates.
    // `U32x16` is the exception: it carries the ARX vocabulary (Add/BitXor/
    // rotate_left) the ChaCha20 lane needs, so it is the native `[U32x4; 4]`
    // defined at the top of this file (mirroring `simd_wasm::wasm32_simd`).
    pub use super::U32x16;
    pub use crate::simd::scalar::{I32x16, U64x8};

    /// 16×f32 backed by 4× NEON `float32x4_t` registers (paired loads).
    #[derive(Copy, Clone)]
    #[repr(align(64))]
    pub struct F32x16(pub [float32x4_t; 4]);

    impl F32x16 {
        pub const LANES: usize = 16;

        #[inline(always)]
        pub fn splat(v: f32) -> Self {
            unsafe {
                let s = vdupq_n_f32(v);
                Self([s, s, s, s])
            }
        }

        #[inline(always)]
        pub fn from_slice(s: &[f32]) -> Self {
            assert!(s.len() >= 16);
            unsafe {
                let p = s.as_ptr();
                Self([vld1q_f32(p), vld1q_f32(p.add(4)), vld1q_f32(p.add(8)), vld1q_f32(p.add(12))])
            }
        }

        #[inline(always)]
        pub fn from_array(a: [f32; 16]) -> Self {
            Self::from_slice(&a)
        }

        #[inline(always)]
        pub fn to_array(self) -> [f32; 16] {
            let mut out = [0.0f32; 16];
            self.copy_to_slice(&mut out);
            out
        }

        #[inline(always)]
        pub fn copy_to_slice(self, s: &mut [f32]) {
            assert!(s.len() >= 16);
            unsafe {
                let p = s.as_mut_ptr();
                vst1q_f32(p, self.0[0]);
                vst1q_f32(p.add(4), self.0[1]);
                vst1q_f32(p.add(8), self.0[2]);
                vst1q_f32(p.add(12), self.0[3]);
            }
        }

        #[inline(always)]
        pub fn reduce_sum(self) -> f32 {
            unsafe {
                let s01 = vaddq_f32(self.0[0], self.0[1]);
                let s23 = vaddq_f32(self.0[2], self.0[3]);
                vaddvq_f32(vaddq_f32(s01, s23))
            }
        }

        #[inline(always)]
        pub fn reduce_min(self) -> f32 {
            self.to_array()
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min)
        }

        #[inline(always)]
        pub fn reduce_max(self) -> f32 {
            self.to_array()
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            unsafe { Self([vabsq_f32(self.0[0]), vabsq_f32(self.0[1]), vabsq_f32(self.0[2]), vabsq_f32(self.0[3])]) }
        }

        #[inline(always)]
        pub fn sqrt(self) -> Self {
            unsafe {
                Self([vsqrtq_f32(self.0[0]), vsqrtq_f32(self.0[1]), vsqrtq_f32(self.0[2]), vsqrtq_f32(self.0[3])])
            }
        }

        #[inline(always)]
        pub fn round(self) -> Self {
            unsafe {
                Self([vrndnq_f32(self.0[0]), vrndnq_f32(self.0[1]), vrndnq_f32(self.0[2]), vrndnq_f32(self.0[3])])
            }
        }

        #[inline(always)]
        pub fn floor(self) -> Self {
            unsafe {
                Self([vrndmq_f32(self.0[0]), vrndmq_f32(self.0[1]), vrndmq_f32(self.0[2]), vrndmq_f32(self.0[3])])
            }
        }

        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            unsafe {
                Self([
                    vfmaq_f32(c.0[0], self.0[0], b.0[0]),
                    vfmaq_f32(c.0[1], self.0[1], b.0[1]),
                    vfmaq_f32(c.0[2], self.0[2], b.0[2]),
                    vfmaq_f32(c.0[3], self.0[3], b.0[3]),
                ])
            }
        }

        #[inline(always)]
        pub fn simd_min(self, other: Self) -> Self {
            unsafe {
                Self([
                    vminq_f32(self.0[0], other.0[0]),
                    vminq_f32(self.0[1], other.0[1]),
                    vminq_f32(self.0[2], other.0[2]),
                    vminq_f32(self.0[3], other.0[3]),
                ])
            }
        }

        #[inline(always)]
        pub fn simd_max(self, other: Self) -> Self {
            unsafe {
                Self([
                    vmaxq_f32(self.0[0], other.0[0]),
                    vmaxq_f32(self.0[1], other.0[1]),
                    vmaxq_f32(self.0[2], other.0[2]),
                    vmaxq_f32(self.0[3], other.0[3]),
                ])
            }
        }

        #[inline(always)]
        pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
            self.simd_max(lo).simd_min(hi)
        }

        #[inline(always)]
        pub fn simd_lt(self, other: Self) -> F32Mask16 {
            let a = self.to_array();
            let b = other.to_array();
            let mut bits: u16 = 0;
            for i in 0..16 {
                if a[i] < b[i] {
                    bits |= 1 << i;
                }
            }
            F32Mask16(bits)
        }
        #[inline(always)]
        pub fn simd_le(self, other: Self) -> F32Mask16 {
            let a = self.to_array();
            let b = other.to_array();
            let mut bits: u16 = 0;
            for i in 0..16 {
                if a[i] <= b[i] {
                    bits |= 1 << i;
                }
            }
            F32Mask16(bits)
        }
        #[inline(always)]
        pub fn simd_gt(self, other: Self) -> F32Mask16 {
            other.simd_lt(self)
        }
        #[inline(always)]
        pub fn simd_ge(self, other: Self) -> F32Mask16 {
            other.simd_le(self)
        }
        #[inline(always)]
        pub fn simd_eq(self, other: Self) -> F32Mask16 {
            let a = self.to_array();
            let b = other.to_array();
            let mut bits: u16 = 0;
            for i in 0..16 {
                if a[i] == b[i] {
                    bits |= 1 << i;
                }
            }
            F32Mask16(bits)
        }
        #[inline(always)]
        pub fn simd_ne(self, other: Self) -> F32Mask16 {
            let a = self.to_array();
            let b = other.to_array();
            let mut bits: u16 = 0;
            for i in 0..16 {
                if a[i] != b[i] {
                    bits |= 1 << i;
                }
            }
            F32Mask16(bits)
        }

        #[inline(always)]
        pub fn to_bits(self) -> U32x16 {
            let a = self.to_array();
            let mut o = [0u32; 16];
            for i in 0..16 {
                o[i] = a[i].to_bits();
            }
            U32x16::from_array(o)
        }
        #[inline(always)]
        pub fn from_bits(bits: U32x16) -> Self {
            let b = bits.to_array();
            let mut o = [0.0f32; 16];
            for i in 0..16 {
                o[i] = f32::from_bits(b[i]);
            }
            Self::from_array(o)
        }
        #[inline(always)]
        pub fn cast_i32(self) -> I32x16 {
            let a = self.to_array();
            let mut o = [0i32; 16];
            for i in 0..16 {
                o[i] = a[i] as i32;
            }
            I32x16(o)
        }
    }

    impl Add for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            unsafe {
                Self([
                    vaddq_f32(self.0[0], rhs.0[0]),
                    vaddq_f32(self.0[1], rhs.0[1]),
                    vaddq_f32(self.0[2], rhs.0[2]),
                    vaddq_f32(self.0[3], rhs.0[3]),
                ])
            }
        }
    }
    impl Sub for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            unsafe {
                Self([
                    vsubq_f32(self.0[0], rhs.0[0]),
                    vsubq_f32(self.0[1], rhs.0[1]),
                    vsubq_f32(self.0[2], rhs.0[2]),
                    vsubq_f32(self.0[3], rhs.0[3]),
                ])
            }
        }
    }
    impl Mul for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            unsafe {
                Self([
                    vmulq_f32(self.0[0], rhs.0[0]),
                    vmulq_f32(self.0[1], rhs.0[1]),
                    vmulq_f32(self.0[2], rhs.0[2]),
                    vmulq_f32(self.0[3], rhs.0[3]),
                ])
            }
        }
    }
    impl Div for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            unsafe {
                Self([
                    vdivq_f32(self.0[0], rhs.0[0]),
                    vdivq_f32(self.0[1], rhs.0[1]),
                    vdivq_f32(self.0[2], rhs.0[2]),
                    vdivq_f32(self.0[3], rhs.0[3]),
                ])
            }
        }
    }
    impl AddAssign for F32x16 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }
    impl SubAssign for F32x16 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }
    impl MulAssign for F32x16 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }
    impl DivAssign for F32x16 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }
    impl Neg for F32x16 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            unsafe { Self([vnegq_f32(self.0[0]), vnegq_f32(self.0[1]), vnegq_f32(self.0[2]), vnegq_f32(self.0[3])]) }
        }
    }
    impl fmt::Debug for F32x16 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "F32x16({:?})", self.to_array())
        }
    }
    impl PartialEq for F32x16 {
        fn eq(&self, other: &Self) -> bool {
            self.to_array() == other.to_array()
        }
    }
    impl Default for F32x16 {
        fn default() -> Self {
            Self::splat(0.0)
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub struct F32Mask16(pub u16);
    impl F32Mask16 {
        #[inline(always)]
        pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
            let t = true_val.to_array();
            let f = false_val.to_array();
            let mut o = [0.0f32; 16];
            for i in 0..16 {
                o[i] = if (self.0 >> i) & 1 == 1 { t[i] } else { f[i] };
            }
            F32x16::from_array(o)
        }
    }

    /// 8×f64 backed by 4× NEON `float64x2_t` registers (paired loads).
    #[derive(Copy, Clone)]
    #[repr(align(64))]
    pub struct F64x8(pub [float64x2_t; 4]);

    impl F64x8 {
        pub const LANES: usize = 8;

        #[inline(always)]
        pub fn splat(v: f64) -> Self {
            unsafe {
                let s = vdupq_n_f64(v);
                Self([s, s, s, s])
            }
        }

        #[inline(always)]
        pub fn from_slice(s: &[f64]) -> Self {
            assert!(s.len() >= 8);
            unsafe {
                let p = s.as_ptr();
                Self([vld1q_f64(p), vld1q_f64(p.add(2)), vld1q_f64(p.add(4)), vld1q_f64(p.add(6))])
            }
        }

        #[inline(always)]
        pub fn from_array(a: [f64; 8]) -> Self {
            Self::from_slice(&a)
        }

        #[inline(always)]
        pub fn to_array(self) -> [f64; 8] {
            let mut out = [0.0f64; 8];
            self.copy_to_slice(&mut out);
            out
        }

        #[inline(always)]
        pub fn copy_to_slice(self, s: &mut [f64]) {
            assert!(s.len() >= 8);
            unsafe {
                let p = s.as_mut_ptr();
                vst1q_f64(p, self.0[0]);
                vst1q_f64(p.add(2), self.0[1]);
                vst1q_f64(p.add(4), self.0[2]);
                vst1q_f64(p.add(6), self.0[3]);
            }
        }

        #[inline(always)]
        pub fn reduce_sum(self) -> f64 {
            unsafe {
                let s01 = vaddq_f64(self.0[0], self.0[1]);
                let s23 = vaddq_f64(self.0[2], self.0[3]);
                vaddvq_f64(vaddq_f64(s01, s23))
            }
        }

        #[inline(always)]
        pub fn reduce_min(self) -> f64 {
            self.to_array()
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
        }

        #[inline(always)]
        pub fn reduce_max(self) -> f64 {
            self.to_array()
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max)
        }

        #[inline(always)]
        pub fn abs(self) -> Self {
            unsafe { Self([vabsq_f64(self.0[0]), vabsq_f64(self.0[1]), vabsq_f64(self.0[2]), vabsq_f64(self.0[3])]) }
        }

        #[inline(always)]
        pub fn sqrt(self) -> Self {
            unsafe {
                Self([vsqrtq_f64(self.0[0]), vsqrtq_f64(self.0[1]), vsqrtq_f64(self.0[2]), vsqrtq_f64(self.0[3])])
            }
        }

        #[inline(always)]
        pub fn round(self) -> Self {
            unsafe {
                Self([vrndnq_f64(self.0[0]), vrndnq_f64(self.0[1]), vrndnq_f64(self.0[2]), vrndnq_f64(self.0[3])])
            }
        }

        #[inline(always)]
        pub fn floor(self) -> Self {
            unsafe {
                Self([vrndmq_f64(self.0[0]), vrndmq_f64(self.0[1]), vrndmq_f64(self.0[2]), vrndmq_f64(self.0[3])])
            }
        }

        #[inline(always)]
        pub fn mul_add(self, b: Self, c: Self) -> Self {
            unsafe {
                Self([
                    vfmaq_f64(c.0[0], self.0[0], b.0[0]),
                    vfmaq_f64(c.0[1], self.0[1], b.0[1]),
                    vfmaq_f64(c.0[2], self.0[2], b.0[2]),
                    vfmaq_f64(c.0[3], self.0[3], b.0[3]),
                ])
            }
        }

        #[inline(always)]
        pub fn simd_min(self, other: Self) -> Self {
            unsafe {
                Self([
                    vminq_f64(self.0[0], other.0[0]),
                    vminq_f64(self.0[1], other.0[1]),
                    vminq_f64(self.0[2], other.0[2]),
                    vminq_f64(self.0[3], other.0[3]),
                ])
            }
        }

        #[inline(always)]
        pub fn simd_max(self, other: Self) -> Self {
            unsafe {
                Self([
                    vmaxq_f64(self.0[0], other.0[0]),
                    vmaxq_f64(self.0[1], other.0[1]),
                    vmaxq_f64(self.0[2], other.0[2]),
                    vmaxq_f64(self.0[3], other.0[3]),
                ])
            }
        }

        #[inline(always)]
        pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
            self.simd_max(lo).simd_min(hi)
        }

        #[inline(always)]
        pub fn simd_ge(self, other: Self) -> F64Mask8 {
            let a = self.to_array();
            let b = other.to_array();
            let mut bits: u8 = 0;
            for i in 0..8 {
                if a[i] >= b[i] {
                    bits |= 1 << i;
                }
            }
            F64Mask8(bits)
        }
        #[inline(always)]
        pub fn simd_le(self, other: Self) -> F64Mask8 {
            let a = self.to_array();
            let b = other.to_array();
            let mut bits: u8 = 0;
            for i in 0..8 {
                if a[i] <= b[i] {
                    bits |= 1 << i;
                }
            }
            F64Mask8(bits)
        }

        #[inline(always)]
        pub fn to_bits(self) -> U64x8 {
            let a = self.to_array();
            let mut o = [0u64; 8];
            for i in 0..8 {
                o[i] = a[i].to_bits();
            }
            U64x8(o)
        }
        #[inline(always)]
        pub fn from_bits(bits: U64x8) -> Self {
            let mut o = [0.0f64; 8];
            for i in 0..8 {
                o[i] = f64::from_bits(bits.0[i]);
            }
            Self::from_array(o)
        }
    }

    impl Add for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {
            unsafe {
                Self([
                    vaddq_f64(self.0[0], rhs.0[0]),
                    vaddq_f64(self.0[1], rhs.0[1]),
                    vaddq_f64(self.0[2], rhs.0[2]),
                    vaddq_f64(self.0[3], rhs.0[3]),
                ])
            }
        }
    }
    impl Sub for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {
            unsafe {
                Self([
                    vsubq_f64(self.0[0], rhs.0[0]),
                    vsubq_f64(self.0[1], rhs.0[1]),
                    vsubq_f64(self.0[2], rhs.0[2]),
                    vsubq_f64(self.0[3], rhs.0[3]),
                ])
            }
        }
    }
    impl Mul for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn mul(self, rhs: Self) -> Self {
            unsafe {
                Self([
                    vmulq_f64(self.0[0], rhs.0[0]),
                    vmulq_f64(self.0[1], rhs.0[1]),
                    vmulq_f64(self.0[2], rhs.0[2]),
                    vmulq_f64(self.0[3], rhs.0[3]),
                ])
            }
        }
    }
    impl Div for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn div(self, rhs: Self) -> Self {
            unsafe {
                Self([
                    vdivq_f64(self.0[0], rhs.0[0]),
                    vdivq_f64(self.0[1], rhs.0[1]),
                    vdivq_f64(self.0[2], rhs.0[2]),
                    vdivq_f64(self.0[3], rhs.0[3]),
                ])
            }
        }
    }
    impl AddAssign for F64x8 {
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {
            *self = *self + rhs;
        }
    }
    impl SubAssign for F64x8 {
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {
            *self = *self - rhs;
        }
    }
    impl MulAssign for F64x8 {
        #[inline(always)]
        fn mul_assign(&mut self, rhs: Self) {
            *self = *self * rhs;
        }
    }
    impl DivAssign for F64x8 {
        #[inline(always)]
        fn div_assign(&mut self, rhs: Self) {
            *self = *self / rhs;
        }
    }
    impl Neg for F64x8 {
        type Output = Self;
        #[inline(always)]
        fn neg(self) -> Self {
            unsafe { Self([vnegq_f64(self.0[0]), vnegq_f64(self.0[1]), vnegq_f64(self.0[2]), vnegq_f64(self.0[3])]) }
        }
    }
    impl fmt::Debug for F64x8 {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "F64x8({:?})", self.to_array())
        }
    }
    impl PartialEq for F64x8 {
        fn eq(&self, other: &Self) -> bool {
            self.to_array() == other.to_array()
        }
    }
    impl Default for F64x8 {
        fn default() -> Self {
            Self::splat(0.0)
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub struct F64Mask8(pub u8);
    impl F64Mask8 {
        #[inline(always)]
        pub fn select(self, true_val: F64x8, false_val: F64x8) -> F64x8 {
            let t = true_val.to_array();
            let f = false_val.to_array();
            let mut o = [0.0f64; 8];
            for i in 0..8 {
                o[i] = if (self.0 >> i) & 1 == 1 { t[i] } else { f[i] };
            }
            F64x8::from_array(o)
        }
    }

    // Lowercase aliases (consumer-API parity)
    #[allow(non_camel_case_types)]
    pub type f32x16 = F32x16;
    #[allow(non_camel_case_types)]
    pub type f64x8 = F64x8;
}

#[cfg(all(target_arch = "aarch64", test))]
mod neon_pair_tests {
    use super::aarch64_simd::*;

    #[test]
    fn f32x16_neon_load_add_store() {
        let a: [f32; 16] = core::array::from_fn(|i| i as f32);
        let b: [f32; 16] = core::array::from_fn(|i| (i * 10) as f32);
        let va = F32x16::from_slice(&a);
        let vb = F32x16::from_slice(&b);
        let vc = va + vb;
        let mut out = [0.0f32; 16];
        vc.copy_to_slice(&mut out);
        for i in 0..16 {
            assert_eq!(out[i], (i + i * 10) as f32);
        }
    }

    #[test]
    fn f32x16_neon_mul_add() {
        let a = F32x16::splat(2.0);
        let b = F32x16::splat(3.0);
        let c = F32x16::splat(1.0);
        let r = a.mul_add(b, c).to_array();
        for &v in &r {
            assert_eq!(v, 7.0);
        }
    }

    #[test]
    fn f32x16_neon_reduce_sum() {
        let v = F32x16::from_array(core::array::from_fn(|i| (i + 1) as f32));
        // sum 1..=16 = 136
        assert_eq!(v.reduce_sum(), 136.0);
    }

    #[test]
    fn f64x8_neon_load_add_store() {
        let a: [f64; 8] = core::array::from_fn(|i| i as f64);
        let b: [f64; 8] = core::array::from_fn(|i| (i * 10) as f64);
        let va = F64x8::from_slice(&a);
        let vb = F64x8::from_slice(&b);
        let vc = va + vb;
        let mut out = [0.0f64; 8];
        vc.copy_to_slice(&mut out);
        for i in 0..8 {
            assert_eq!(out[i], (i + i * 10) as f64);
        }
    }

    #[test]
    fn f64x8_neon_mul_add_reduce() {
        let a = F64x8::splat(2.0);
        let b = F64x8::splat(3.0);
        let c = F64x8::splat(1.0);
        let r = a.mul_add(b, c);
        // 8 lanes × 7.0 = 56.0
        assert_eq!(r.reduce_sum(), 56.0);
    }
}

// I8/I16 SIMD vector types — NEON 128-bit native + scalar polyfills.
//
// Native 128-bit shapes:
//   • I8x16  ← int8x16_t   (vaddq_s8 / vminq_s8 / vcgtq_s8 …)
//   • I16x8  ← int16x8_t   (vaddq_s16 / vcgtq_s16 …)
//
// Polyfills (scalar arrays) for cross-tier API parity:
//   • I8x32  = [i8; 32]
//   • I8x64  = [i8; 64]
//   • I16x16 = [i16; 16]
//   • I16x32 = [i16; 32]
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I8x16(pub int8x16_t);

#[cfg(target_arch = "aarch64")]
impl I8x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: i8) -> Self {
        Self(unsafe { vdupq_n_s8(v) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { vdupq_n_s8(0) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[i8]) -> Self {
        assert!(s.len() >= 16);
        Self(unsafe { vld1q_s8(s.as_ptr()) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i8; 16]) -> Self {
        Self(unsafe { vld1q_s8(arr.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i8; 16] {
        let mut arr = [0i8; 16];
        unsafe { vst1q_s8(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i8]) {
        assert!(s.len() >= 16);
        unsafe { vst1q_s8(s.as_mut_ptr(), self.0) };
    }

    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { vaddq_s8(self.0, other.0) })
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { vsubq_s8(self.0, other.0) })
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_s8(self.0, other.0) })
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_s8(self.0, other.0) })
    }

    /// Compare-greater-than: returns 16-bit mask. Bit i set where self[i] > other[i].
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u16 {
        unsafe {
            let cmp = vcgtq_s8(self.0, other.0); // uint8x16_t, 0xFF where true
            let arr: [u8; 16] = core::mem::transmute(cmp);
            let mut m: u16 = 0;
            for i in 0..16 {
                if arr[i] != 0 {
                    m |= 1u16 << i;
                }
            }
            m
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl core::fmt::Debug for I8x16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I8x16({:?})", self.to_array())
    }
}
#[cfg(target_arch = "aarch64")]
impl PartialEq for I8x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I16x8(pub int16x8_t);

#[cfg(target_arch = "aarch64")]
impl I16x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        Self(unsafe { vdupq_n_s16(v) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { vdupq_n_s16(0) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[i16]) -> Self {
        assert!(s.len() >= 8);
        Self(unsafe { vld1q_s16(s.as_ptr()) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i16; 8]) -> Self {
        Self(unsafe { vld1q_s16(arr.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 8] {
        let mut arr = [0i16; 8];
        unsafe { vst1q_s16(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i16]) {
        assert!(s.len() >= 8);
        unsafe { vst1q_s16(s.as_mut_ptr(), self.0) };
    }

    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { vaddq_s16(self.0, other.0) })
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { vsubq_s16(self.0, other.0) })
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_s16(self.0, other.0) })
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_s16(self.0, other.0) })
    }

    /// Compare-greater-than: returns 8-bit mask. Bit i set where self[i] > other[i].
    #[inline(always)]
    pub fn cmp_gt(self, other: Self) -> u8 {
        unsafe {
            let cmp = vcgtq_s16(self.0, other.0); // uint16x8_t, 0xFFFF where true
            let arr: [u16; 8] = core::mem::transmute(cmp);
            let mut m: u8 = 0;
            for i in 0..8 {
                if arr[i] != 0 {
                    m |= 1u8 << i;
                }
            }
            m
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl core::fmt::Debug for I16x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "I16x8({:?})", self.to_array())
    }
}
#[cfg(target_arch = "aarch64")]
impl PartialEq for I16x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// W3-B: NEON integer wrapper types (item 8 of burn parity list)
// ─ U8x16, U16x8, U32x4, U64x2, I32x4, I64x2 ─
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U8x16(pub uint8x16_t);

#[cfg(target_arch = "aarch64")]
impl U8x16 {
    pub const LANES: usize = 16;
    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        Self(unsafe { vdupq_n_u8(v) })
    }
    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { vdupq_n_u8(0) })
    }
    #[inline(always)]
    pub fn from_slice(s: &[u8]) -> Self {
        assert!(s.len() >= 16);
        Self(unsafe { vld1q_u8(s.as_ptr()) })
    }
    #[inline(always)]
    pub fn from_array(arr: [u8; 16]) -> Self {
        Self(unsafe { vld1q_u8(arr.as_ptr()) })
    }
    #[inline(always)]
    pub fn to_array(self) -> [u8; 16] {
        let mut arr = [0u8; 16];
        unsafe { vst1q_u8(arr.as_mut_ptr(), self.0) };
        arr
    }
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u8]) {
        assert!(s.len() >= 16);
        unsafe { vst1q_u8(s.as_mut_ptr(), self.0) };
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { vaddq_u8(self.0, other.0) })
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { vsubq_u8(self.0, other.0) })
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_u8(self.0, other.0) })
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_u8(self.0, other.0) })
    }
}

#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U16x8(pub uint16x8_t);

#[cfg(target_arch = "aarch64")]
impl U16x8 {
    pub const LANES: usize = 8;
    #[inline(always)]
    pub fn splat(v: u16) -> Self {
        Self(unsafe { vdupq_n_u16(v) })
    }
    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { vdupq_n_u16(0) })
    }
    #[inline(always)]
    pub fn from_slice(s: &[u16]) -> Self {
        assert!(s.len() >= 8);
        Self(unsafe { vld1q_u16(s.as_ptr()) })
    }
    #[inline(always)]
    pub fn from_array(arr: [u16; 8]) -> Self {
        Self(unsafe { vld1q_u16(arr.as_ptr()) })
    }
    #[inline(always)]
    pub fn to_array(self) -> [u16; 8] {
        let mut arr = [0u16; 8];
        unsafe { vst1q_u16(arr.as_mut_ptr(), self.0) };
        arr
    }
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u16]) {
        assert!(s.len() >= 8);
        unsafe { vst1q_u16(s.as_mut_ptr(), self.0) };
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { vaddq_u16(self.0, other.0) })
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { vsubq_u16(self.0, other.0) })
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_u16(self.0, other.0) })
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_u16(self.0, other.0) })
    }
}

// Lowercase alias (consumer-API parity — `simd.rs` re-exports `u16x8`).
#[cfg(target_arch = "aarch64")]
#[allow(non_camel_case_types)]
pub type u16x8 = U16x8;

#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U32x4(pub uint32x4_t);

#[cfg(target_arch = "aarch64")]
impl U32x4 {
    pub const LANES: usize = 4;
    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self(unsafe { vdupq_n_u32(v) })
    }
    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { vdupq_n_u32(0) })
    }
    #[inline(always)]
    pub fn from_slice(s: &[u32]) -> Self {
        assert!(s.len() >= 4);
        Self(unsafe { vld1q_u32(s.as_ptr()) })
    }
    #[inline(always)]
    pub fn from_array(arr: [u32; 4]) -> Self {
        Self(unsafe { vld1q_u32(arr.as_ptr()) })
    }
    #[inline(always)]
    pub fn to_array(self) -> [u32; 4] {
        let mut arr = [0u32; 4];
        unsafe { vst1q_u32(arr.as_mut_ptr(), self.0) };
        arr
    }
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u32]) {
        assert!(s.len() >= 4);
        unsafe { vst1q_u32(s.as_mut_ptr(), self.0) };
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { vaddq_u32(self.0, other.0) })
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { vsubq_u32(self.0, other.0) })
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_u32(self.0, other.0) })
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_u32(self.0, other.0) })
    }
    /// Lane-wise XOR — the ARX `⊕` (`veorq_u32`).
    #[inline(always)]
    pub fn bitxor(self, other: Self) -> Self {
        Self(unsafe { veorq_u32(self.0, other.0) })
    }
    /// Lane-wise left-rotate by `n` bits — the ARX rotate (matches
    /// `u32::rotate_left`). NEON has no rotate op, so this is the shift-or via
    /// the variable-count `vshlq_u32` (a signed per-lane count: `+n` shifts
    /// left by `n`, `n-32 < 0` shifts logical-right by `32-n`). The `n % 32 == 0`
    /// early-return avoids the ambiguous full-width shift. Rotate amount is a
    /// public ARX constant.
    #[inline(always)]
    pub fn rotate_left(self, n: u32) -> Self {
        let n = n % 32;
        if n == 0 {
            return self;
        }
        unsafe {
            let l = vshlq_u32(self.0, vdupq_n_s32(n as i32));
            let r = vshlq_u32(self.0, vdupq_n_s32(n as i32 - 32));
            Self(vorrq_u32(l, r))
        }
    }
}

/// 16×u32 as `[U32x4; 4]` — the NEON-native 16-wide ARX lane (ChaCha20 /
/// BLAKE). `U32x4` is the dispatched native unit (`uint32x4_t`); `U32x16` fans
/// each op over the 4 sub-lanes. Consumer API (`Add` / `BitXor` / `rotate_left`)
/// matches `simd_avx512::U32x16` exactly, so the ChaCha20 backend compiles
/// unchanged on every tier (the wasm arm uses the identical `[U32x4; 4]` shape).
#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
#[repr(align(64))]
pub struct U32x16(pub [U32x4; 4]);

#[cfg(target_arch = "aarch64")]
impl U32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self([U32x4::splat(v); 4])
    }

    #[inline(always)]
    pub fn from_array(a: [u32; 16]) -> Self {
        Self([
            U32x4::from_array([a[0], a[1], a[2], a[3]]),
            U32x4::from_array([a[4], a[5], a[6], a[7]]),
            U32x4::from_array([a[8], a[9], a[10], a[11]]),
            U32x4::from_array([a[12], a[13], a[14], a[15]]),
        ])
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 16] {
        let mut o = [0u32; 16];
        for i in 0..4 {
            o[i * 4..i * 4 + 4].copy_from_slice(&self.0[i].to_array());
        }
        o
    }

    /// Lane-wise left-rotate by `n` bits (ARX rotate), fanned over 4 lanes.
    #[inline(always)]
    pub fn rotate_left(self, n: u32) -> Self {
        Self([
            self.0[0].rotate_left(n),
            self.0[1].rotate_left(n),
            self.0[2].rotate_left(n),
            self.0[3].rotate_left(n),
        ])
    }
}

#[cfg(target_arch = "aarch64")]
impl core::ops::Add for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, r: Self) -> Self {
        Self([self.0[0].add(r.0[0]), self.0[1].add(r.0[1]), self.0[2].add(r.0[2]), self.0[3].add(r.0[3])])
    }
}

#[cfg(target_arch = "aarch64")]
impl core::ops::BitXor for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, r: Self) -> Self {
        Self([
            self.0[0].bitxor(r.0[0]),
            self.0[1].bitxor(r.0[1]),
            self.0[2].bitxor(r.0[2]),
            self.0[3].bitxor(r.0[3]),
        ])
    }
}

#[cfg(target_arch = "aarch64")]
impl core::fmt::Debug for U32x16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "U32x16({:?})", self.to_array())
    }
}

#[cfg(target_arch = "aarch64")]
impl PartialEq for U32x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(non_camel_case_types)]
pub type u32x16 = U32x16;

#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U64x2(pub uint64x2_t);

#[cfg(target_arch = "aarch64")]
impl U64x2 {
    pub const LANES: usize = 2;
    #[inline(always)]
    pub fn splat(v: u64) -> Self {
        Self(unsafe { vdupq_n_u64(v) })
    }
    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { vdupq_n_u64(0) })
    }
    #[inline(always)]
    pub fn from_slice(s: &[u64]) -> Self {
        assert!(s.len() >= 2);
        Self(unsafe { vld1q_u64(s.as_ptr()) })
    }
    #[inline(always)]
    pub fn from_array(arr: [u64; 2]) -> Self {
        Self(unsafe { vld1q_u64(arr.as_ptr()) })
    }
    #[inline(always)]
    pub fn to_array(self) -> [u64; 2] {
        let mut arr = [0u64; 2];
        unsafe { vst1q_u64(arr.as_mut_ptr(), self.0) };
        arr
    }
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u64]) {
        assert!(s.len() >= 2);
        unsafe { vst1q_u64(s.as_mut_ptr(), self.0) };
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { vaddq_u64(self.0, other.0) })
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { vsubq_u64(self.0, other.0) })
    }
    // NEON has no vminq_u64 / vmaxq_u64 — scalar fallback
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        Self::from_array([a[0].min(b[0]), a[1].min(b[1])])
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        Self::from_array([a[0].max(b[0]), a[1].max(b[1])])
    }
}

#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I32x4(pub int32x4_t);

#[cfg(target_arch = "aarch64")]
impl I32x4 {
    pub const LANES: usize = 4;
    #[inline(always)]
    pub fn splat(v: i32) -> Self {
        Self(unsafe { vdupq_n_s32(v) })
    }
    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { vdupq_n_s32(0) })
    }
    #[inline(always)]
    pub fn from_slice(s: &[i32]) -> Self {
        assert!(s.len() >= 4);
        Self(unsafe { vld1q_s32(s.as_ptr()) })
    }
    #[inline(always)]
    pub fn from_array(arr: [i32; 4]) -> Self {
        Self(unsafe { vld1q_s32(arr.as_ptr()) })
    }
    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
        let mut arr = [0i32; 4];
        unsafe { vst1q_s32(arr.as_mut_ptr(), self.0) };
        arr
    }
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i32]) {
        assert!(s.len() >= 4);
        unsafe { vst1q_s32(s.as_mut_ptr(), self.0) };
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { vaddq_s32(self.0, other.0) })
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { vsubq_s32(self.0, other.0) })
    }
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self(unsafe { vminq_s32(self.0, other.0) })
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self(unsafe { vmaxq_s32(self.0, other.0) })
    }
}

#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I64x2(pub int64x2_t);

#[cfg(target_arch = "aarch64")]
impl I64x2 {
    pub const LANES: usize = 2;
    #[inline(always)]
    pub fn splat(v: i64) -> Self {
        Self(unsafe { vdupq_n_s64(v) })
    }
    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { vdupq_n_s64(0) })
    }
    #[inline(always)]
    pub fn from_slice(s: &[i64]) -> Self {
        assert!(s.len() >= 2);
        Self(unsafe { vld1q_s64(s.as_ptr()) })
    }
    #[inline(always)]
    pub fn from_array(arr: [i64; 2]) -> Self {
        Self(unsafe { vld1q_s64(arr.as_ptr()) })
    }
    #[inline(always)]
    pub fn to_array(self) -> [i64; 2] {
        let mut arr = [0i64; 2];
        unsafe { vst1q_s64(arr.as_mut_ptr(), self.0) };
        arr
    }
    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i64]) {
        assert!(s.len() >= 2);
        unsafe { vst1q_s64(s.as_mut_ptr(), self.0) };
    }
    #[inline(always)]
    pub fn add(self, other: Self) -> Self {
        Self(unsafe { vaddq_s64(self.0, other.0) })
    }
    #[inline(always)]
    pub fn sub(self, other: Self) -> Self {
        Self(unsafe { vsubq_s64(self.0, other.0) })
    }
    // NEON has no vminq_s64 / vmaxq_s64 — scalar fallback
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        Self::from_array([a[0].min(b[0]), a[1].min(b[1])])
    }
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        let a = self.to_array();
        let b = other.to_array();
        Self::from_array([a[0].max(b[0]), a[1].max(b[1])])
    }
}

// ── Polyfills for wider lanes (scalar arrays) ─────────────────────────────

#[allow(unused_macros)]
macro_rules! neon_int_polyfill {
    ($name:ident, $elem:ty, $lanes:expr, $zero:expr, $mask:ty) => {
        #[derive(Copy, Clone)]
        #[repr(align(64))]
        pub struct $name(pub [$elem; $lanes]);

        impl $name {
            pub const LANES: usize = $lanes;
            #[inline(always)]
            pub fn splat(v: $elem) -> Self {
                Self([v; $lanes])
            }
            #[inline(always)]
            pub fn zero() -> Self {
                Self([$zero; $lanes])
            }
            #[inline(always)]
            pub fn from_slice(s: &[$elem]) -> Self {
                assert!(s.len() >= $lanes);
                let mut a = [$zero; $lanes];
                a.copy_from_slice(&s[..$lanes]);
                Self(a)
            }
            #[inline(always)]
            pub fn from_array(a: [$elem; $lanes]) -> Self {
                Self(a)
            }
            #[inline(always)]
            pub fn to_array(self) -> [$elem; $lanes] {
                self.0
            }
            #[inline(always)]
            pub fn copy_to_slice(self, s: &mut [$elem]) {
                assert!(s.len() >= $lanes);
                s[..$lanes].copy_from_slice(&self.0);
            }
            #[inline(always)]
            pub fn add(self, other: Self) -> Self {
                let mut o = [$zero; $lanes];
                for i in 0..$lanes {
                    o[i] = self.0[i].wrapping_add(other.0[i]);
                }
                Self(o)
            }
            #[inline(always)]
            pub fn sub(self, other: Self) -> Self {
                let mut o = [$zero; $lanes];
                for i in 0..$lanes {
                    o[i] = self.0[i].wrapping_sub(other.0[i]);
                }
                Self(o)
            }
            #[inline(always)]
            pub fn min(self, other: Self) -> Self {
                let mut o = [$zero; $lanes];
                for i in 0..$lanes {
                    o[i] = self.0[i].min(other.0[i]);
                }
                Self(o)
            }
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {
                let mut o = [$zero; $lanes];
                for i in 0..$lanes {
                    o[i] = self.0[i].max(other.0[i]);
                }
                Self(o)
            }
            #[inline(always)]
            pub fn cmp_gt(self, other: Self) -> $mask {
                let mut m: $mask = 0;
                for i in 0..$lanes {
                    if self.0[i] > other.0[i] {
                        m |= (1 as $mask) << i;
                    }
                }
                m
            }
        }
        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, concat!(stringify!($name), "({:?})"), &self.0[..])
            }
        }
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
neon_int_polyfill!(I8x32, i8, 32, 0i8, u32);
#[cfg(target_arch = "aarch64")]
neon_int_polyfill!(I8x64, i8, 64, 0i8, u64);
#[cfg(target_arch = "aarch64")]
neon_int_polyfill!(I16x16, i16, 16, 0i16, u16);
#[cfg(target_arch = "aarch64")]
neon_int_polyfill!(I16x32, i16, 32, 0i16, u32);

#[cfg(target_arch = "aarch64")]
#[allow(non_camel_case_types)]
pub type i8x16 = I8x16;
#[cfg(target_arch = "aarch64")]
#[allow(non_camel_case_types)]
pub type i16x8 = I16x8;
#[cfg(target_arch = "aarch64")]
#[allow(non_camel_case_types)]
pub type i8x32 = I8x32;
#[cfg(target_arch = "aarch64")]
#[allow(non_camel_case_types)]
pub type i8x64 = I8x64;
#[cfg(target_arch = "aarch64")]
#[allow(non_camel_case_types)]
pub type i16x16 = I16x16;
#[cfg(target_arch = "aarch64")]
#[allow(non_camel_case_types)]
pub type i16x32 = I16x32;

// ============================================================================
// W1a SIMD primitives — NEON backend
// ============================================================================

// ── W1a-#1: I8x16::from_i4_packed_u64 + lane_i8 (NEON) ──────────────────────

#[cfg(target_arch = "aarch64")]
impl I8x16 {
    /// Unpack 16 signed i4 nibbles from a `u64` into 16 sign-extended `i8` lanes.
    ///
    /// Nibble layout: `lane[i] = sign_extend_i4((packed >> (4*i)) & 0xf)`.
    /// Values `0x0..=0x7` → `0..=7`; values `0x8..=0xf` → `-8..=-1`.
    ///
    /// On NEON this is implemented as a scalar loop (the shift+mask approach
    /// with `vshl_n_s8` would require byte-level load + nibble split across
    /// two registers, but the scalar approach is simpler and correct).
    ///
    /// # Example
    /// ```rust,ignore
    /// let neg = I8x16::from_i4_packed_u64(0xffff_ffff_ffff_ffff);
    /// assert_eq!(neg.lane_i8::<0>(), -1);
    /// ```
    #[inline(always)]
    pub fn from_i4_packed_u64(packed: u64) -> Self {
        let mut lanes = [0i8; 16];
        for i in 0..16 {
            let nibble = ((packed >> (4 * i)) & 0xf) as i8;
            lanes[i] = if nibble > 7 { nibble - 16 } else { nibble };
        }
        // SAFETY: vld1q_s8 loads 16 bytes from a valid aligned stack array.
        Self(unsafe { core::arch::aarch64::vld1q_s8(lanes.as_ptr()) })
    }

    /// Extract lane `N` as an `i8`.
    ///
    /// `N` must be in `0..16`.
    #[inline(always)]
    pub fn lane_i8<const N: usize>(self) -> i8 {
        self.to_array()[N]
    }

    // ── W1a-#2: saturating_abs (NEON) ────────────────────────────────────────

    /// Lane-wise saturating absolute value.
    ///
    /// `saturating_abs(i8::MIN) == i8::MAX` (127).  Uses NEON `vqabsq_s8`
    /// which is hardware-saturating (the `q` suffix denotes saturating
    /// semantics), unlike `vabsq_s8` which wraps.
    ///
    /// # Example
    /// ```rust,ignore
    /// let v = I8x16::splat(i8::MIN);
    /// assert!(v.saturating_abs().to_array().iter().all(|&x| x == i8::MAX));
    /// ```
    #[inline(always)]
    pub fn saturating_abs(self) -> Self {
        // SAFETY: vqabsq_s8 is available on all aarch64 targets; it is a
        // saturating absolute value — `vqabsq_s8(int8x16_t(-128))` returns 127.
        Self(unsafe { core::arch::aarch64::vqabsq_s8(self.0) })
    }
}

// ── W1a-#2: I8x32::saturating_abs (NEON polyfill) ─────────────────────────────

/// `I8x32` on NEON is a scalar polyfill (neon_int_polyfill! array).
/// We add saturating_abs via the scalar path as there is no 256-bit NEON reg.
#[cfg(target_arch = "aarch64")]
impl I8x32 {
    /// Lane-wise saturating absolute value (scalar polyfill on NEON).
    ///
    /// `saturating_abs(i8::MIN) == i8::MAX`.  All 32 lanes processed via
    /// `i8::saturating_abs` in a fused loop.
    ///
    /// # Example
    /// ```rust,ignore
    /// let v = I8x32::splat(i8::MIN);
    /// assert!(v.saturating_abs().to_array().iter().all(|&x| x == i8::MAX));
    /// ```
    #[inline(always)]
    pub fn saturating_abs(self) -> Self {
        let mut o = [0i8; 32];
        for i in 0..32 {
            o[i] = self.0[i].saturating_abs();
        }
        Self(o)
    }
}

// ── W1a-#3: U16x8::gather_u16 + palette_lookup_u8x8 (NEON) ──────────────────

#[cfg(target_arch = "aarch64")]
impl U16x8 {
    /// Gather 8 `u16` values from `table` at the indices in `self`.
    ///
    /// NEON has no native gather instruction; this is a scalar loop over
    /// 8 lanes which is still significantly faster than a cache-miss-bound
    /// random-access loop in typical use because 8 sequential indirections
    /// fit in NEON register pressure.
    ///
    /// In debug builds panics if any index `>= table.len()`.  In release
    /// builds falls back to `table.get(i).copied().unwrap_or(0)`.
    ///
    /// # Example
    /// ```rust,ignore
    /// let table = [10u16, 20, 30, 40, 50, 60, 70, 80];
    /// let idx = U16x8::from_array([0, 2, 4, 6, 1, 3, 5, 7]);
    /// let result = U16x8::gather_u16(idx, &table);
    /// assert_eq!(result.to_array(), [10, 30, 50, 70, 20, 40, 60, 80]);
    /// ```
    #[inline(always)]
    pub fn gather_u16(indices: U16x8, table: &[u16]) -> Self {
        let idx = indices.to_array();
        #[cfg(debug_assertions)]
        for &i in &idx {
            assert!((i as usize) < table.len(), "gather_u16: index {} out of bounds (len={})", i, table.len());
        }
        let mut out = [0u16; 8];
        for k in 0..8 {
            out[k] = table.get(idx[k] as usize).copied().unwrap_or(0);
        }
        Self::from_array(out)
    }

    /// Extract lane `k` as a `u16`.
    #[inline(always)]
    pub fn lane(self, k: usize) -> u16 {
        self.to_array()[k]
    }
}

// ── W1a-#3: U8x8 + palette_lookup_u8x8 (NEON) ───────────────────────────────

/// 8-lane `u8` vector for the NEON backend (scalar-storage polyfill).
/// Used as the return type of `palette_lookup_u8x8`.
#[cfg(target_arch = "aarch64")]
#[derive(Copy, Clone, PartialEq)]
#[repr(align(8))]
pub struct U8x8(pub [u8; 8]);

#[cfg(target_arch = "aarch64")]
impl U8x8 {
    pub const LANES: usize = 8;

    /// Broadcast a single `u8` to all 8 lanes.
    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        Self([v; 8])
    }

    /// Load from a fixed-size array.
    #[inline(always)]
    pub fn from_array(arr: [u8; 8]) -> Self {
        Self(arr)
    }

    /// Extract all 8 lanes as an array.
    #[inline(always)]
    pub fn to_array(self) -> [u8; 8] {
        self.0
    }
}

#[cfg(target_arch = "aarch64")]
impl core::fmt::Debug for U8x8 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "U8x8({:?})", &self.0[..])
    }
}

/// Look up 8 bytes from a `u8` LUT by `u16` indices (NEON backend).
///
/// Scalar loop over 8 lanes (NEON has no native gather).
///
/// Bounds: panics in debug if any index `>= lut.len()`; returns 0 safely in
/// release for out-of-range indices.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn palette_lookup_u8x8(idx_v: U16x8, lut: &[u8]) -> U8x8 {
    let idx = idx_v.to_array();
    #[cfg(debug_assertions)]
    for &i in &idx {
        assert!((i as usize) < lut.len(), "palette_lookup_u8x8: index {} OOB (len={})", i, lut.len());
    }
    let mut out = [0u8; 8];
    for k in 0..8 {
        out[k] = lut.get(idx[k] as usize).copied().unwrap_or(0);
    }
    U8x8(out)
}

// ── W1a-#4: prefetch_read_t0/t1/t2 (NEON / aarch64) ──────────────────────────

/// Hint that `ptr` will be read soon; load into L1 (T0) cache.
///
/// On aarch64 emits `prfm pldl1keep, [ptr]` via inline asm.  `ptr` may be
/// invalid (unmapped): the PRFM instruction is a hint that the CPU can silently
/// drop per the ARM architecture reference.  No assertion is made.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn prefetch_read_t0(ptr: *const u8) {
    // SAFETY: PRFM is a hint instruction; an invalid ptr simply makes the
    // prefetch a no-op.  The pointer is never dereferenced.
    // UNVERIFIED: inline asm syntax for `prfm` on Rust stable 1.94 aarch64 —
    // believed correct per ARM ISA but not verified against an aarch64 builder.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, readonly),
        );
    }
}

/// Hint to load into L2 (T1) cache on aarch64.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn prefetch_read_t1(ptr: *const u8) {
    // SAFETY: same as prefetch_read_t0 — PRFM hint, no fault on invalid ptr.
    // UNVERIFIED: pldl2keep is the correct ARM PRFM operand for L2 hint.
    unsafe {
        core::arch::asm!(
            "prfm pldl2keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, readonly),
        );
    }
}

/// Hint to load into L3 (T2) cache on aarch64.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn prefetch_read_t2(ptr: *const u8) {
    // SAFETY: same as prefetch_read_t0 — PRFM hint, no fault on invalid ptr.
    // UNVERIFIED: pldl3keep is the correct ARM PRFM operand for L3 hint.
    unsafe {
        core::arch::asm!(
            "prfm pldl3keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, readonly),
        );
    }
}

// ── W1a-#5: U64x8 / U64x4 popcnt (NEON) ──────────────────────────────────────
// The NEON aarch64_simd::U64x8 is actually re-exported from simd_scalar.rs
// (see `pub use crate::simd::scalar::{…U64x8}`).  popcnt / xor_popcount are
// added to the scalar U64x8 in simd_scalar.rs and are thereby visible through
// both x86_64 and aarch64 dispatch paths.
//
// U64x4 on the NEON backend is also the scalar polyfill (neon_int_polyfill!).
// Its popcnt is also in simd_scalar.rs for the same reason.

// ── W1a-#1: batch_packed_i4_16 (NEON backend) ────────────────────────────────

/// Closure-parameterised batch over packed i4 data (NEON backend).
///
/// See the x86_64 version in `simd_avx512.rs` for full documentation.
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn batch_packed_i4_16<E, F>(packed: &[u64], aux: &[i8], out: &mut [E], f: F)
where
    F: Fn(I8x16, i8) -> E + Sync + Send,
    E: Copy,
{
    assert_eq!(packed.len(), aux.len(), "batch_packed_i4_16: packed and aux must be same length");
    let n = packed.len().min(out.len());
    for i in 0..n {
        let lanes = I8x16::from_i4_packed_u64(packed[i]);
        out[i] = f(lanes, aux[i]);
    }
}

// ── Aliases ──────────────────────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
#[allow(non_camel_case_types)]
pub type u8x8 = U8x8;

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
            assert!(err < 0.01 || v == 0.0, "f16 roundtrip failed for {}: got {}, err={:.4}", v, back, err);
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
        let input: Vec<u16> = (0..100)
            .map(|i| f32_to_f16_scalar(i as f32 * 0.1 - 5.0))
            .collect();
        let mut batch_out = vec![0.0f32; 100];
        f16_to_f32_batch(&input, &mut batch_out);

        for (i, &h) in input.iter().enumerate() {
            let scalar = f16_to_f32_scalar(h);
            assert_eq!(
                batch_out[i], scalar,
                "batch/scalar mismatch at {}: batch={} scalar={}",
                i, batch_out[i], scalar
            );
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
            assert!(
                err < 0.1 || input[i].abs() < 0.001,
                "roundtrip error at {}: {} → {} → {}, err={}",
                i,
                input[i],
                f16_out[i],
                f32_back[i],
                err
            );
        }
    }
}
