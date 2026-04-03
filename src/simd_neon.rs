//! AArch64 NEON SIMD — scaffolding for future implementation.
//!
//! Mirrors simd_avx512.rs type API. Currently all methods are unimplemented.
//! When needed: fill in with core::arch::aarch64 intrinsics.
//!
//! Reference: macerator's aarch64 backend (tracel-ai/burn, wingertge/macerator)
//! Key intrinsics:
//!   float32x4_t  — 4 × f32 (128-bit NEON register)
//!   float64x2_t  — 2 × f64
//!   uint8x16_t   — 16 × u8
//!   int32x4_t    — 4 × i32
//!   uint64x2_t   — 2 × u64
//!
//! NEON is 128-bit — widest register is 4 × f32.
//! For F32x16 (16 lanes): use 4 × float32x4_t.
//! For F64x8 (8 lanes): use 4 × float64x2_t.
//!
//! Key operations from macerator's NEON backend:
//!   vaddq_f32, vsubq_f32, vmulq_f32, vdivq_f32  — arithmetic
//!   vfmaq_f32                                     — fused multiply-add
//!   vminq_f32, vmaxq_f32                          — min/max
//!   vceqq_f32, vcgeq_f32, vcgtq_f32              — comparison → mask
//!   vld1q_f32, vst1q_f32                          — load/store
//!   vaddvq_f32                                    — horizontal sum (ARMv8.2+)
//!   vpaddq_f32                                    — pairwise add (reduction)
//!   vdupq_n_f32                                   — broadcast (splat)
//!   veorq_u8                                      — XOR (for Hamming)
//!   vcntq_u8                                      — popcount per byte
//!   vpaddlq_u8 / vpaddlq_u16 / vpaddlq_u32       — widening pairwise add (for popcount reduction)

// #[cfg(target_arch = "aarch64")]
// use core::arch::aarch64::*;

// ============================================================================
// F32x16 — 16 × f32 via 4 × float32x4_t (128-bit NEON)
// ============================================================================

// #[derive(Copy, Clone)]
// pub struct F32x16(pub float32x4_t, pub float32x4_t, pub float32x4_t, pub float32x4_t);
//
// impl F32x16 {
//     pub const LANES: usize = 16;
//
//     pub fn splat(v: f32) -> Self {
//         let q = unsafe { vdupq_n_f32(v) };
//         Self(q, q, q, q)
//     }
//
//     pub fn from_slice(s: &[f32]) -> Self {
//         assert!(s.len() >= 16);
//         unsafe {
//             Self(
//                 vld1q_f32(s.as_ptr()),
//                 vld1q_f32(s[4..].as_ptr()),
//                 vld1q_f32(s[8..].as_ptr()),
//                 vld1q_f32(s[12..].as_ptr()),
//             )
//         }
//     }
//
//     pub fn reduce_sum(self) -> f32 {
//         unsafe {
//             let sum01 = vaddq_f32(self.0, self.1);
//             let sum23 = vaddq_f32(self.2, self.3);
//             let sum = vaddq_f32(sum01, sum23);
//             vaddvq_f32(sum)  // ARMv8.2+ horizontal sum
//         }
//     }
//
//     pub fn mul_add(self, b: Self, c: Self) -> Self {
//         unsafe {
//             Self(
//                 vfmaq_f32(c.0, self.0, b.0),  // a*b + c
//                 vfmaq_f32(c.1, self.1, b.1),
//                 vfmaq_f32(c.2, self.2, b.2),
//                 vfmaq_f32(c.3, self.3, b.3),
//             )
//         }
//     }
// }

// ============================================================================
// F64x8 — 8 × f64 via 4 × float64x2_t
// ============================================================================

// #[derive(Copy, Clone)]
// pub struct F64x8(pub float64x2_t, pub float64x2_t, pub float64x2_t, pub float64x2_t);
//
// impl F64x8 {
//     pub const LANES: usize = 8;
//     // ... same pattern: 4 × 2-lane operations
// }

// ============================================================================
// U8x64 — 64 × u8 via 4 × uint8x16_t (for Hamming / byte ops)
// ============================================================================

// #[derive(Copy, Clone)]
// pub struct U8x64(pub uint8x16_t, pub uint8x16_t, pub uint8x16_t, pub uint8x16_t);
//
// impl U8x64 {
//     pub const LANES: usize = 64;
//
//     pub fn splat(v: u8) -> Self {
//         let q = unsafe { vdupq_n_u8(v) };
//         Self(q, q, q, q)
//     }
//
//     // Hamming distance via vcntq_u8 (per-byte popcount) + widening sum
//     pub fn popcount_sum(self) -> u32 {
//         unsafe {
//             let c0 = vcntq_u8(self.0);  // popcount per byte
//             let c1 = vcntq_u8(self.1);
//             let c2 = vcntq_u8(self.2);
//             let c3 = vcntq_u8(self.3);
//             // Widen: u8 → u16 → u32 → u64 → scalar
//             let sum = vaddvq_u8(c0) as u32
//                     + vaddvq_u8(c1) as u32
//                     + vaddvq_u8(c2) as u32
//                     + vaddvq_u8(c3) as u32;
//             sum
//         }
//     }
// }

// ============================================================================
// I32x16 — 16 × i32 via 4 × int32x4_t (for Base17 L1 distance)
// ============================================================================

// #[derive(Copy, Clone)]
// pub struct I32x16(pub int32x4_t, pub int32x4_t, pub int32x4_t, pub int32x4_t);
//
// impl I32x16 {
//     pub const LANES: usize = 16;
//
//     pub fn from_i16_slice(s: &[i16]) -> Self {
//         // vmovl_s16: sign-extend 4 × i16 → 4 × i32
//         // Need to load 16 × i16 (32 bytes) → 4 × int32x4_t
//         unsafe {
//             let lo8 = vld1q_s16(s.as_ptr());            // 8 × i16
//             let hi8 = vld1q_s16(s[8..].as_ptr());       // 8 × i16
//             Self(
//                 vmovl_s16(vget_low_s16(lo8)),             // first 4
//                 vmovl_s16(vget_high_s16(lo8)),            // next 4
//                 vmovl_s16(vget_low_s16(hi8)),             // next 4
//                 vmovl_s16(vget_high_s16(hi8)),            // last 4
//             )
//         }
//     }
//
//     pub fn abs(self) -> Self {
//         unsafe {
//             Self(vabsq_s32(self.0), vabsq_s32(self.1),
//                  vabsq_s32(self.2), vabsq_s32(self.3))
//         }
//     }
//
//     pub fn reduce_sum(self) -> i32 {
//         unsafe {
//             let sum01 = vaddq_s32(self.0, self.1);
//             let sum23 = vaddq_s32(self.2, self.3);
//             let sum = vaddq_s32(sum01, sum23);
//             vaddvq_s32(sum)  // ARMv8.2+ horizontal sum
//         }
//     }
// }

// ============================================================================
// BF16 conversion on NEON (ARMv8.6+ has native BF16 instructions)
// ============================================================================

// ARMv8.6-A adds:
//   vcvtq_f32_bf16  — 8 BF16 → 8 f32 (via bfcvt instruction)
//   vcvtq_bf16_f32  — 8 f32 → 8 BF16
//
// Fallback (ARMv8.0-8.5): same bit-shift as x86 scalar:
//   f32::from_bits((bf16_bits as u32) << 16)
//
// pub fn bf16_to_f32_batch_neon(input: &[u16], output: &mut [f32]) {
//     // ARMv8.6+ path:
//     //   let bf16x8 = vld1q_bf16(input.as_ptr());
//     //   let f32x4_lo = vcvtq_low_f32_bf16(bf16x8);
//     //   let f32x4_hi = vcvtq_high_f32_bf16(bf16x8);
//     //
//     // Fallback: scalar bit shift
//     for (src, dst) in input.iter().zip(output.iter_mut()) {
//         *dst = f32::from_bits((*src as u32) << 16);
//     }
// }
