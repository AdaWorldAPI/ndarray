//! Runtime-dispatched batch cast trampolines.
//!
//! Wraps the four existing half-precision cast surfaces. Each
//! underlying function already has internal runtime dispatch (F16C on
//! `cast_*_f16_*`, avx512bf16 + AVX-512F bit-shift on
//! `bf16_to_f32_batch`, AVX-512F-only RNE on `f32_to_bf16_batch_rne`),
//! shipped in PR #183. The wrappers here are `#[inline(always)]` so
//! they collapse to one call to the inner dispatcher at the consumer
//! site.
//!
//! All four are bit-exact per IEEE 754:
//! - F16 ↔ f32: `_mm256_cvtph_ps` / `_mm256_cvtps_ph::<0>` on F16C hosts,
//!   scalar `F16::to_f32` / `F16::from_f32_rounded` elsewhere.
//! - BF16 → f32: `_mm512_cvtpbh_ps` on avx512bf16 hosts, AVX-512F
//!   bit-shift on avx512f hosts, scalar bit-shift otherwise.
//! - f32 → BF16 RNE: pure AVX-512F bit-fiddle (byte-exact vs
//!   `_mm512_cvtneps_pbh`), scalar IEEE RNE elsewhere.

use crate::hpc::quantized::{BF16, F16};

/// F16 → f32 batch (lossless, IEEE 754 widening).
#[inline(always)]
pub fn cast_f16_to_f32_batch(src: &[F16], dst: &mut [f32]) {
    crate::simd_half::cast_f16_to_f32_batch(src, dst)
}

/// f32 → F16 batch (IEEE 754 RNE, bit-exact vs `F16::from_f32_rounded`).
#[inline(always)]
pub fn cast_f32_to_f16_batch(src: &[f32], dst: &mut [F16]) {
    crate::simd_half::cast_f32_to_f16_batch(src, dst)
}

/// BF16 → f32 batch (lossless bit-shift).
///
/// The underlying `crate::simd::bf16_to_f32_batch` takes `&[u16]`; the
/// trampoline accepts `&[BF16]` for symmetry with the F16 surface and
/// reinterprets via repr(transparent) layout equivalence.
#[inline(always)]
pub fn bf16_to_f32_batch(src: &[BF16], dst: &mut [f32]) {
    // SAFETY: BF16 is `#[repr(transparent)] struct BF16(pub u16)`.
    let src_u16: &[u16] = unsafe { core::slice::from_raw_parts(src.as_ptr() as *const u16, src.len()) };
    crate::simd::bf16_to_f32_batch(src_u16, dst)
}

/// f32 → BF16 batch (RNE, byte-exact vs `_mm512_cvtneps_pbh`).
///
/// Underlying `crate::simd::f32_to_bf16_batch_rne` takes `&mut [u16]`;
/// trampoline accepts `&mut [BF16]` and reinterprets.
#[inline(always)]
pub fn f32_to_bf16_batch_rne(src: &[f32], dst: &mut [BF16]) {
    // SAFETY: BF16 is repr(transparent) over u16.
    let dst_u16: &mut [u16] = unsafe { core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u16, dst.len()) };
    crate::simd::f32_to_bf16_batch_rne(src, dst_u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip_via_runtime_trampolines() {
        let inputs: Vec<F16> = (0..33).map(|i| F16::from_f32(i as f32 * 0.75)).collect();
        let mut f32_buf = vec![0.0f32; 33];
        cast_f16_to_f32_batch(&inputs, &mut f32_buf);
        let mut back = vec![F16::ZERO; 33];
        cast_f32_to_f16_batch(&f32_buf, &mut back);
        for i in 0..33 {
            assert_eq!(back[i], inputs[i], "F16 roundtrip mismatch at {i}");
        }
    }

    #[test]
    fn bf16_roundtrip_via_runtime_trampolines() {
        let inputs: Vec<BF16> = (0..33)
            .map(|i| BF16::from_f32_rounded(i as f32 * 0.75))
            .collect();
        let mut f32_buf = vec![0.0f32; 33];
        bf16_to_f32_batch(&inputs, &mut f32_buf);
        let mut back = vec![BF16::ZERO; 33];
        f32_to_bf16_batch_rne(&f32_buf, &mut back);
        for i in 0..33 {
            assert_eq!(back[i], inputs[i], "BF16 roundtrip mismatch at {i}");
        }
    }
}
