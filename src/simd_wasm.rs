//! WebAssembly SIMD128 — scaffolding for future implementation.
//!
//! Mirrors simd_avx512.rs type API. Currently all methods are unimplemented.
//! When needed: fill in with core::arch::wasm32 intrinsics.
//!
//! Reference: macerator's wasm32 backend (wingertge/macerator)
//!
//! WASM SIMD128 provides one 128-bit register type: v128
//! All operations are 128-bit wide:
//!   f32x4  — 4 × f32
//!   f64x2  — 2 × f64
//!   i8x16  — 16 × i8 / u8
//!   i16x8  — 8 × i16 / u16
//!   i32x4  — 4 × i32 / u32
//!   i64x2  — 2 × i64 / u64
//!
//! Key intrinsics (core::arch::wasm32):
//!   f32x4_add, f32x4_sub, f32x4_mul           — arithmetic
//!   f32x4_min, f32x4_max                       — min/max
//!   f32x4_splat                                 — broadcast
//!   v128_load, v128_store                       — memory
//!   f32x4_extract_lane                          — lane access
//!   i8x16_popcnt                                — popcount per byte (Relaxed SIMD)
//!   v128_xor, v128_and, v128_or                 — bitwise
//!   i16x8_extend_low_i8x16                      — sign-extend (for Base17)
//!   i32x4_extend_low_i16x8                      — sign-extend i16→i32
//!
//! For F32x16 (16 lanes): use 4 × v128 (f32x4 interpretation).
//! For F64x8 (8 lanes): use 4 × v128 (f64x2 interpretation).
//! Same 4-register pattern as NEON.
//!
//! WASM Relaxed SIMD (proposal, not yet standard):
//!   f32x4_fma                                   — fused multiply-add
//!   i8x16_relaxed_swizzle                       — byte shuffle
//!   These are NOT universally available yet.

// #[cfg(target_arch = "wasm32")]
// use core::arch::wasm32::*;

// ============================================================================
// F32x16 — 16 × f32 via 4 × v128 (f32x4 interpretation)
// ============================================================================

// #[derive(Copy, Clone)]
// pub struct F32x16(pub v128, pub v128, pub v128, pub v128);
//
// impl F32x16 {
//     pub const LANES: usize = 16;
//
//     pub fn splat(v: f32) -> Self {
//         let q = f32x4_splat(v);
//         Self(q, q, q, q)
//     }
//
//     pub fn from_slice(s: &[f32]) -> Self {
//         assert!(s.len() >= 16);
//         unsafe {
//             Self(
//                 v128_load(s.as_ptr() as *const v128),
//                 v128_load(s[4..].as_ptr() as *const v128),
//                 v128_load(s[8..].as_ptr() as *const v128),
//                 v128_load(s[12..].as_ptr() as *const v128),
//             )
//         }
//     }
//
//     pub fn reduce_sum(self) -> f32 {
//         // No horizontal sum instruction in WASM SIMD128.
//         // Manual: extract all 16 lanes + sum.
//         let sum01 = f32x4_add(self.0, self.1);
//         let sum23 = f32x4_add(self.2, self.3);
//         let sum = f32x4_add(sum01, sum23);
//         // Pairwise reduction within v128:
//         // shuffle high pair to low, add, extract lane 0
//         let hi = i32x4_shuffle::<2, 3, 0, 1>(sum, sum);
//         let sum2 = f32x4_add(sum, hi);
//         let hi2 = i32x4_shuffle::<1, 0, 3, 2>(sum2, sum2);
//         let sum1 = f32x4_add(sum2, hi2);
//         f32x4_extract_lane::<0>(sum1)
//     }
//
//     // FMA: requires Relaxed SIMD proposal
//     // pub fn mul_add(self, b: Self, c: Self) -> Self {
//     //     Self(
//     //         f32x4_relaxed_madd(self.0, b.0, c.0),
//     //         f32x4_relaxed_madd(self.1, b.1, c.1),
//     //         f32x4_relaxed_madd(self.2, b.2, c.2),
//     //         f32x4_relaxed_madd(self.3, b.3, c.3),
//     //     )
//     // }
//     // Fallback without Relaxed SIMD:
//     // pub fn mul_add(self, b: Self, c: Self) -> Self {
//     //     Self(
//     //         f32x4_add(f32x4_mul(self.0, b.0), c.0),
//     //         f32x4_add(f32x4_mul(self.1, b.1), c.1),
//     //         f32x4_add(f32x4_mul(self.2, b.2), c.2),
//     //         f32x4_add(f32x4_mul(self.3, b.3), c.3),
//     //     )
//     // }
// }

// ============================================================================
// U8x64 — 64 × u8 via 4 × v128 (i8x16 interpretation, for Hamming)
// ============================================================================

// #[derive(Copy, Clone)]
// pub struct U8x64(pub v128, pub v128, pub v128, pub v128);
//
// impl U8x64 {
//     pub const LANES: usize = 64;
//
//     // Popcount: i8x16_popcnt requires Relaxed SIMD proposal.
//     // Fallback: XOR → byte-level LUT popcount via i8x16_swizzle.
//     //
//     // Alternative: extract bytes to scalar and use count_ones().
// }

// ============================================================================
// I32x16 — 16 × i32 via 4 × v128 (i32x4 interpretation, for Base17)
// ============================================================================

// #[derive(Copy, Clone)]
// pub struct I32x16(pub v128, pub v128, pub v128, pub v128);
//
// impl I32x16 {
//     pub const LANES: usize = 16;
//
//     pub fn from_i16_slice(s: &[i16]) -> Self {
//         // i32x4_extend_low_i16x8: sign-extend lower 4 × i16 → 4 × i32
//         // Need: load 16 × i16 (32 bytes) → 4 passes of extend
//         // let v0 = v128_load(s.as_ptr() as *const v128);        // 8 × i16
//         // let v1 = v128_load(s[8..].as_ptr() as *const v128);   // 8 × i16
//         // Self(
//         //     i32x4_extend_low_i16x8(v0),    // first 4
//         //     i32x4_extend_high_i16x8(v0),   // next 4
//         //     i32x4_extend_low_i16x8(v1),    // next 4
//         //     i32x4_extend_high_i16x8(v1),   // last 4
//         // )
//     }
// }

// ============================================================================
// BF16 conversion on WASM (no hardware support — scalar only)
// ============================================================================

// WASM has no BF16 instructions. Use the universal scalar fallback:
//   f32::from_bits((bf16_bits as u32) << 16)
//
// pub fn bf16_to_f32_batch_wasm(input: &[u16], output: &mut [f32]) {
//     for (src, dst) in input.iter().zip(output.iter_mut()) {
//         *dst = f32::from_bits((*src as u32) << 16);
//     }
// }

// ============================================================================
// PREFERRED_LANES for WASM (128-bit only)
// ============================================================================

// pub const PREFERRED_F32_LANES: usize = 4;   // v128 = 4 × f32
// pub const PREFERRED_F64_LANES: usize = 2;   // v128 = 2 × f64
// pub const PREFERRED_U64_LANES: usize = 2;   // v128 = 2 × u64
// pub const PREFERRED_I16_LANES: usize = 8;   // v128 = 8 × i16
