//! Half-precision SIMD vector types: `BF16x16` and `F16x16`.
//!
//! Portable scalar implementations — 16 lanes, operate by upcasting to f32,
//! performing the operation, then downcasting back. This is correct and
//! portable; SIMD-accelerated fast paths (AVX2, AVX-512-BF16, NEON fp16)
//! are a follow-up.
//!
//! Slice-level helpers (`add_bf16_inplace`, `cast_bf16_to_f32_batch`, etc.)
//! use `chunks_exact(16)` for the vectorized path with a scalar tail.

use crate::hpc::quantized::{BF16, F16};

// ============================================================================
// BF16x16 — 16-lane BFloat16 vector
// ============================================================================

/// 16 × BF16 packed into a scalar array.
///
/// All arithmetic operates via f32 upcast → op → BF16 downcast (round-to-nearest-even).
#[derive(Clone, Copy, Debug)]
pub struct BF16x16([u16; 16]);

impl BF16x16 {
    /// Number of lanes.
    pub const LANES: usize = 16;

    /// Load 16 BF16 values from a slice (must have `len >= 16`).
    #[inline]
    pub fn from_slice(s: &[BF16]) -> Self {
        assert!(s.len() >= 16, "BF16x16::from_slice: need >= 16 elements, got {}", s.len());
        let mut arr = [0u16; 16];
        for i in 0..16 {
            arr[i] = s[i].0;
        }
        BF16x16(arr)
    }

    /// Store 16 BF16 values into a slice (must have `len >= 16`).
    #[inline]
    pub fn copy_to_slice(self, s: &mut [BF16]) {
        assert!(s.len() >= 16, "BF16x16::copy_to_slice: need >= 16 elements, got {}", s.len());
        for i in 0..16 {
            s[i] = BF16(self.0[i]);
        }
    }

    /// Broadcast a single BF16 value across all 16 lanes.
    #[inline]
    pub fn splat(v: BF16) -> Self {
        BF16x16([v.0; 16])
    }

    /// Lane-wise addition: `self[i] + other[i]` for each lane.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        let mut out = [0u16; 16];
        for i in 0..16 {
            let a = BF16(self.0[i]).to_f32();
            let b = BF16(other.0[i]).to_f32();
            out[i] = BF16::from_f32_rounded(a + b).0;
        }
        BF16x16(out)
    }

    /// Lane-wise subtraction: `self[i] - other[i]` for each lane.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        let mut out = [0u16; 16];
        for i in 0..16 {
            let a = BF16(self.0[i]).to_f32();
            let b = BF16(other.0[i]).to_f32();
            out[i] = BF16::from_f32_rounded(a - b).0;
        }
        BF16x16(out)
    }

    /// Lane-wise multiplication: `self[i] * other[i]` for each lane.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        let mut out = [0u16; 16];
        for i in 0..16 {
            let a = BF16(self.0[i]).to_f32();
            let b = BF16(other.0[i]).to_f32();
            out[i] = BF16::from_f32_rounded(a * b).0;
        }
        BF16x16(out)
    }

    /// Fused multiply-add: `self[i] * b[i] + c[i]` for each lane.
    ///
    /// Uses f32 FMA for maximum precision in the intermediate result.
    #[inline]
    pub fn fma(self, b: Self, c: Self) -> Self {
        let mut out = [0u16; 16];
        for i in 0..16 {
            let av = BF16(self.0[i]).to_f32();
            let bv = BF16(b.0[i]).to_f32();
            let cv = BF16(c.0[i]).to_f32();
            out[i] = BF16::from_f32_rounded(av.mul_add(bv, cv)).0;
        }
        BF16x16(out)
    }

    /// Upcast all 16 BF16 lanes to f32.
    #[inline]
    pub fn to_f32x16(self) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = BF16(self.0[i]).to_f32();
        }
        out
    }
}

// ============================================================================
// F16x16 — 16-lane IEEE 754 half-precision vector
// ============================================================================

/// 16 × F16 (IEEE 754 binary16) packed into a scalar array.
///
/// # Scalar-perf disclosure (TD-SIMD-8)
///
/// **This is a scalar polyfill, not a SIMD type.** Storage is plain
/// `[u16; 16]` (no `__m256i` / `__m256bh` / `float16x8_t`). Every
/// arithmetic op upcasts to f32, computes lane-by-lane, downcasts back
/// to f16 with round-to-nearest-even — same path on every backend
/// (AVX-512, AVX2, NEON, scalar). Consumers in hot loops should NOT
/// reach for `crate::simd::F16x16` expecting SIMD throughput.
///
/// The hardware-native paths exist on x86 via `_mm256_cvtph_ps` /
/// `_mm256_cvtps_ph` (F16C; Ivy Bridge+) and on aarch64 via
/// `vfmaq_f16` (ARMv8.2-A `+fp16`; Pi 5, Apple, modern Snapdragons).
/// Wiring those into `F16x16` is tracked as TD-SIMD-8 in
/// `.claude/knowledge/simd-dispatch-architecture.md`. Until then, hot
/// loops on f16 should use `core::simd::f16x16` under the `nightly-simd`
/// feature (real `core::simd::*` codegen) or stay in f32 and convert
/// at storage boundaries.
///
/// Not to be confused with `simd_avx2::F16Scaler` — that's a *scaling
/// context* for range-normalizing values before f16 encoding (so the
/// dynamic range maps to f16's `[-65504, 65504]` window without
/// clipping), not a SIMD lane type.
#[derive(Clone, Copy, Debug)]
pub struct F16x16([u16; 16]);

impl F16x16 {
    /// Number of lanes.
    pub const LANES: usize = 16;

    /// Load 16 F16 values from a slice (must have `len >= 16`).
    #[inline]
    pub fn from_slice(s: &[F16]) -> Self {
        assert!(s.len() >= 16, "F16x16::from_slice: need >= 16 elements, got {}", s.len());
        let mut arr = [0u16; 16];
        for i in 0..16 {
            arr[i] = s[i].0;
        }
        F16x16(arr)
    }

    /// Store 16 F16 values into a slice (must have `len >= 16`).
    #[inline]
    pub fn copy_to_slice(self, s: &mut [F16]) {
        assert!(s.len() >= 16, "F16x16::copy_to_slice: need >= 16 elements, got {}", s.len());
        for i in 0..16 {
            s[i] = F16(self.0[i]);
        }
    }

    /// Broadcast a single F16 value across all 16 lanes.
    #[inline]
    pub fn splat(v: F16) -> Self {
        F16x16([v.0; 16])
    }

    /// Lane-wise addition: `self[i] + other[i]` for each lane.
    #[inline]
    pub fn add(self, other: Self) -> Self {
        let mut out = [0u16; 16];
        for i in 0..16 {
            let a = F16(self.0[i]).to_f32();
            let b = F16(other.0[i]).to_f32();
            out[i] = F16::from_f32_rounded(a + b).0;
        }
        F16x16(out)
    }

    /// Lane-wise subtraction: `self[i] - other[i]` for each lane.
    #[inline]
    pub fn sub(self, other: Self) -> Self {
        let mut out = [0u16; 16];
        for i in 0..16 {
            let a = F16(self.0[i]).to_f32();
            let b = F16(other.0[i]).to_f32();
            out[i] = F16::from_f32_rounded(a - b).0;
        }
        F16x16(out)
    }

    /// Lane-wise multiplication: `self[i] * other[i]` for each lane.
    #[inline]
    pub fn mul(self, other: Self) -> Self {
        let mut out = [0u16; 16];
        for i in 0..16 {
            let a = F16(self.0[i]).to_f32();
            let b = F16(other.0[i]).to_f32();
            out[i] = F16::from_f32_rounded(a * b).0;
        }
        F16x16(out)
    }

    /// Fused multiply-add: `self[i] * b[i] + c[i]` for each lane.
    ///
    /// Uses f32 FMA for maximum precision in the intermediate result.
    #[inline]
    pub fn fma(self, b: Self, c: Self) -> Self {
        let mut out = [0u16; 16];
        for i in 0..16 {
            let av = F16(self.0[i]).to_f32();
            let bv = F16(b.0[i]).to_f32();
            let cv = F16(c.0[i]).to_f32();
            out[i] = F16::from_f32_rounded(av.mul_add(bv, cv)).0;
        }
        F16x16(out)
    }

    /// Upcast all 16 F16 lanes to f32.
    #[inline]
    pub fn to_f32x16(self) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = F16(self.0[i]).to_f32();
        }
        out
    }
}

// ============================================================================
// Slice-level ops — BF16
// ============================================================================

/// Element-wise in-place addition: `dst[i] += src[i]`.
///
/// Uses BF16x16 for chunks of 16, scalar tail for remainder.
pub fn add_bf16_inplace(dst: &mut [BF16], src: &[BF16]) {
    let n = dst.len().min(src.len());
    let (dst, src) = (&mut dst[..n], &src[..n]);

    let chunks = n / 16;
    for c in 0..chunks {
        let off = c * 16;
        let a = BF16x16::from_slice(&dst[off..]);
        let b = BF16x16::from_slice(&src[off..]);
        a.add(b).copy_to_slice(&mut dst[off..]);
    }
    // Scalar tail
    for i in (chunks * 16)..n {
        let sum = dst[i].to_f32() + src[i].to_f32();
        dst[i] = BF16::from_f32_rounded(sum);
    }
}

/// Element-wise in-place multiplication: `dst[i] *= src[i]`.
///
/// Uses BF16x16 for chunks of 16, scalar tail for remainder.
pub fn mul_bf16_inplace(dst: &mut [BF16], src: &[BF16]) {
    let n = dst.len().min(src.len());
    let (dst, src) = (&mut dst[..n], &src[..n]);

    let chunks = n / 16;
    for c in 0..chunks {
        let off = c * 16;
        let a = BF16x16::from_slice(&dst[off..]);
        let b = BF16x16::from_slice(&src[off..]);
        a.mul(b).copy_to_slice(&mut dst[off..]);
    }
    // Scalar tail
    for i in (chunks * 16)..n {
        let prod = dst[i].to_f32() * src[i].to_f32();
        dst[i] = BF16::from_f32_rounded(prod);
    }
}

// ============================================================================
// Slice-level ops — F16
// ============================================================================

/// Element-wise in-place addition: `dst[i] += src[i]`.
///
/// Uses F16x16 for chunks of 16, scalar tail for remainder.
pub fn add_f16_inplace(dst: &mut [F16], src: &[F16]) {
    let n = dst.len().min(src.len());
    let (dst, src) = (&mut dst[..n], &src[..n]);

    let chunks = n / 16;
    for c in 0..chunks {
        let off = c * 16;
        let a = F16x16::from_slice(&dst[off..]);
        let b = F16x16::from_slice(&src[off..]);
        a.add(b).copy_to_slice(&mut dst[off..]);
    }
    // Scalar tail
    for i in (chunks * 16)..n {
        let sum = dst[i].to_f32() + src[i].to_f32();
        dst[i] = F16::from_f32_rounded(sum);
    }
}

/// Element-wise in-place multiplication: `dst[i] *= src[i]`.
///
/// Uses F16x16 for chunks of 16, scalar tail for remainder.
pub fn mul_f16_inplace(dst: &mut [F16], src: &[F16]) {
    let n = dst.len().min(src.len());
    let (dst, src) = (&mut dst[..n], &src[..n]);

    let chunks = n / 16;
    for c in 0..chunks {
        let off = c * 16;
        let a = F16x16::from_slice(&dst[off..]);
        let b = F16x16::from_slice(&src[off..]);
        a.mul(b).copy_to_slice(&mut dst[off..]);
    }
    // Scalar tail
    for i in (chunks * 16)..n {
        let prod = dst[i].to_f32() * src[i].to_f32();
        dst[i] = F16::from_f32_rounded(prod);
    }
}

// ============================================================================
// Batch cast operations
// ============================================================================

/// Batch convert BF16 → f32.
///
/// Uses BF16x16 for chunks of 16, scalar tail for remainder.
pub fn cast_bf16_to_f32_batch(src: &[BF16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    let chunks = n / 16;
    for c in 0..chunks {
        let off = c * 16;
        let v = BF16x16::from_slice(&src[off..]);
        let f = v.to_f32x16();
        dst[off..off + 16].copy_from_slice(&f);
    }
    // Scalar tail
    for i in (chunks * 16)..n {
        dst[i] = src[i].to_f32();
    }
}

/// Batch convert F16 → f32.
///
/// Uses F16x16 for chunks of 16, scalar tail for remainder.
pub fn cast_f16_to_f32_batch(src: &[F16], dst: &mut [f32]) {
    let n = src.len().min(dst.len());
    let chunks = n / 16;
    for c in 0..chunks {
        let off = c * 16;
        let v = F16x16::from_slice(&src[off..]);
        let f = v.to_f32x16();
        dst[off..off + 16].copy_from_slice(&f);
    }
    // Scalar tail
    for i in (chunks * 16)..n {
        dst[i] = src[i].to_f32();
    }
}

/// Batch convert f32 → BF16 (round-to-nearest-even).
pub fn cast_f32_to_bf16_batch(src: &[f32], dst: &mut [BF16]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        dst[i] = BF16::from_f32_rounded(src[i]);
    }
}

/// Batch convert f32 → F16 (round-to-nearest-even).
pub fn cast_f32_to_f16_batch(src: &[f32], dst: &mut [F16]) {
    let n = src.len().min(dst.len());
    for i in 0..n {
        dst[i] = F16::from_f32_rounded(src[i]);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── BF16x16 tests ───────────────────────────────────────────────

    #[test]
    fn bf16x16_add_matches_scalar() {
        let a_vals: Vec<BF16> = (0..16)
            .map(|i| BF16::from_f32_rounded(i as f32 * 0.5))
            .collect();
        let b_vals: Vec<BF16> = (0..16)
            .map(|i| BF16::from_f32_rounded(i as f32 * 0.25 + 1.0))
            .collect();

        let va = BF16x16::from_slice(&a_vals);
        let vb = BF16x16::from_slice(&b_vals);
        let result = va.add(vb);

        let mut out = vec![BF16::ZERO; 16];
        result.copy_to_slice(&mut out);

        for i in 0..16 {
            let expected = BF16::from_f32_rounded(a_vals[i].to_f32() + b_vals[i].to_f32());
            assert_eq!(out[i], expected, "BF16x16 add mismatch at lane {}", i);
        }
    }

    #[test]
    fn bf16x16_sub_matches_scalar() {
        let a_vals: Vec<BF16> = (0..16)
            .map(|i| BF16::from_f32_rounded(10.0 + i as f32))
            .collect();
        let b_vals: Vec<BF16> = (0..16)
            .map(|i| BF16::from_f32_rounded(i as f32 * 0.5))
            .collect();

        let result = BF16x16::from_slice(&a_vals).sub(BF16x16::from_slice(&b_vals));
        let mut out = vec![BF16::ZERO; 16];
        result.copy_to_slice(&mut out);

        for i in 0..16 {
            let expected = BF16::from_f32_rounded(a_vals[i].to_f32() - b_vals[i].to_f32());
            assert_eq!(out[i], expected, "BF16x16 sub mismatch at lane {}", i);
        }
    }

    #[test]
    fn bf16x16_mul_matches_scalar() {
        let a_vals: Vec<BF16> = (0..16)
            .map(|i| BF16::from_f32_rounded(i as f32 * 0.5 + 0.1))
            .collect();
        let b_vals: Vec<BF16> = (0..16)
            .map(|i| BF16::from_f32_rounded(i as f32 * 0.3 + 0.2))
            .collect();

        let result = BF16x16::from_slice(&a_vals).mul(BF16x16::from_slice(&b_vals));
        let mut out = vec![BF16::ZERO; 16];
        result.copy_to_slice(&mut out);

        for i in 0..16 {
            let expected = BF16::from_f32_rounded(a_vals[i].to_f32() * b_vals[i].to_f32());
            assert_eq!(out[i], expected, "BF16x16 mul mismatch at lane {}", i);
        }
    }

    #[test]
    fn bf16x16_fma_matches_scalar() {
        let a: Vec<BF16> = (0..16)
            .map(|i| BF16::from_f32_rounded(i as f32 + 1.0))
            .collect();
        let b: Vec<BF16> = (0..16)
            .map(|i| BF16::from_f32_rounded(0.5 * i as f32))
            .collect();
        let c: Vec<BF16> = (0..16)
            .map(|i| BF16::from_f32_rounded(i as f32 * 0.1))
            .collect();

        let result = BF16x16::from_slice(&a).fma(BF16x16::from_slice(&b), BF16x16::from_slice(&c));
        let mut out = vec![BF16::ZERO; 16];
        result.copy_to_slice(&mut out);

        for i in 0..16 {
            let expected = BF16::from_f32_rounded(a[i].to_f32().mul_add(b[i].to_f32(), c[i].to_f32()));
            assert_eq!(out[i], expected, "BF16x16 fma mismatch at lane {}", i);
        }
    }

    #[test]
    fn bf16x16_to_f32x16_roundtrip() {
        let vals: Vec<BF16> = (0..16)
            .map(|i| BF16::from_f32_rounded(i as f32 * 1.5))
            .collect();
        let v = BF16x16::from_slice(&vals);
        let f32s = v.to_f32x16();

        for i in 0..16 {
            assert_eq!(f32s[i], vals[i].to_f32(), "BF16x16::to_f32x16 mismatch at lane {}", i);
        }
    }

    #[test]
    fn bf16x16_splat() {
        let v = BF16x16::splat(BF16::from_f32_rounded(3.14));
        let f32s = v.to_f32x16();
        let expected = BF16::from_f32_rounded(3.14).to_f32();
        for i in 0..16 {
            assert_eq!(f32s[i], expected, "BF16x16 splat mismatch at lane {}", i);
        }
    }

    // ── F16x16 tests ────────────────────────────────────────────────

    #[test]
    fn f16x16_add_matches_scalar() {
        let a_vals: Vec<F16> = (0..16)
            .map(|i| F16::from_f32_rounded(i as f32 * 0.5))
            .collect();
        let b_vals: Vec<F16> = (0..16)
            .map(|i| F16::from_f32_rounded(i as f32 * 0.25 + 1.0))
            .collect();

        let result = F16x16::from_slice(&a_vals).add(F16x16::from_slice(&b_vals));
        let mut out = vec![F16::ZERO; 16];
        result.copy_to_slice(&mut out);

        for i in 0..16 {
            let expected = F16::from_f32_rounded(a_vals[i].to_f32() + b_vals[i].to_f32());
            assert_eq!(out[i], expected, "F16x16 add mismatch at lane {}", i);
        }
    }

    #[test]
    fn f16x16_mul_matches_scalar() {
        let a_vals: Vec<F16> = (0..16)
            .map(|i| F16::from_f32_rounded(i as f32 * 0.5 + 0.1))
            .collect();
        let b_vals: Vec<F16> = (0..16)
            .map(|i| F16::from_f32_rounded(i as f32 * 0.3 + 0.2))
            .collect();

        let result = F16x16::from_slice(&a_vals).mul(F16x16::from_slice(&b_vals));
        let mut out = vec![F16::ZERO; 16];
        result.copy_to_slice(&mut out);

        for i in 0..16 {
            let expected = F16::from_f32_rounded(a_vals[i].to_f32() * b_vals[i].to_f32());
            assert_eq!(out[i], expected, "F16x16 mul mismatch at lane {}", i);
        }
    }

    #[test]
    fn f16x16_sub_matches_scalar() {
        let a_vals: Vec<F16> = (0..16)
            .map(|i| F16::from_f32_rounded(10.0 + i as f32))
            .collect();
        let b_vals: Vec<F16> = (0..16)
            .map(|i| F16::from_f32_rounded(i as f32 * 0.5))
            .collect();

        let result = F16x16::from_slice(&a_vals).sub(F16x16::from_slice(&b_vals));
        let mut out = vec![F16::ZERO; 16];
        result.copy_to_slice(&mut out);

        for i in 0..16 {
            let expected = F16::from_f32_rounded(a_vals[i].to_f32() - b_vals[i].to_f32());
            assert_eq!(out[i], expected, "F16x16 sub mismatch at lane {}", i);
        }
    }

    #[test]
    fn f16x16_fma_matches_scalar() {
        let a: Vec<F16> = (0..16)
            .map(|i| F16::from_f32_rounded(i as f32 + 1.0))
            .collect();
        let b: Vec<F16> = (0..16)
            .map(|i| F16::from_f32_rounded(0.5 * i as f32))
            .collect();
        let c: Vec<F16> = (0..16)
            .map(|i| F16::from_f32_rounded(i as f32 * 0.1))
            .collect();

        let result = F16x16::from_slice(&a).fma(F16x16::from_slice(&b), F16x16::from_slice(&c));
        let mut out = vec![F16::ZERO; 16];
        result.copy_to_slice(&mut out);

        for i in 0..16 {
            let expected = F16::from_f32_rounded(a[i].to_f32().mul_add(b[i].to_f32(), c[i].to_f32()));
            assert_eq!(out[i], expected, "F16x16 fma mismatch at lane {}", i);
        }
    }

    #[test]
    fn f16x16_to_f32x16_roundtrip() {
        let vals: Vec<F16> = (0..16)
            .map(|i| F16::from_f32_rounded(i as f32 * 1.5))
            .collect();
        let v = F16x16::from_slice(&vals);
        let f32s = v.to_f32x16();

        for i in 0..16 {
            assert_eq!(f32s[i], vals[i].to_f32(), "F16x16::to_f32x16 mismatch at lane {}", i);
        }
    }

    #[test]
    fn f16x16_splat() {
        let v = F16x16::splat(F16::from_f32_rounded(2.71));
        let f32s = v.to_f32x16();
        let expected = F16::from_f32_rounded(2.71).to_f32();
        for i in 0..16 {
            assert_eq!(f32s[i], expected, "F16x16 splat mismatch at lane {}", i);
        }
    }

    // ── Slice-level ops: add_bf16_inplace with various tail lengths ──

    #[test]
    fn add_bf16_inplace_tail_15() {
        let n = 15;
        let mut dst: Vec<BF16> = (0..n).map(|i| BF16::from_f32_rounded(i as f32)).collect();
        let src: Vec<BF16> = (0..n)
            .map(|i| BF16::from_f32_rounded(i as f32 * 0.5))
            .collect();
        let expected: Vec<BF16> = (0..n)
            .map(|i| BF16::from_f32_rounded(i as f32 + i as f32 * 0.5))
            .collect();

        add_bf16_inplace(&mut dst, &src);
        for i in 0..n {
            assert_eq!(dst[i], expected[i], "add_bf16_inplace tail=15 mismatch at {}", i);
        }
    }

    #[test]
    fn add_bf16_inplace_tail_17() {
        let n = 17;
        let mut dst: Vec<BF16> = (0..n).map(|i| BF16::from_f32_rounded(i as f32)).collect();
        let src: Vec<BF16> = (0..n)
            .map(|i| BF16::from_f32_rounded(i as f32 * 0.5))
            .collect();
        let expected: Vec<BF16> = (0..n)
            .map(|i| BF16::from_f32_rounded(i as f32 + i as f32 * 0.5))
            .collect();

        add_bf16_inplace(&mut dst, &src);
        for i in 0..n {
            assert_eq!(dst[i], expected[i], "add_bf16_inplace tail=17 mismatch at {}", i);
        }
    }

    #[test]
    fn add_bf16_inplace_tail_31() {
        let n = 31;
        let mut dst: Vec<BF16> = (0..n).map(|i| BF16::from_f32_rounded(i as f32)).collect();
        let src: Vec<BF16> = (0..n).map(|i| BF16::from_f32_rounded(1.0)).collect();
        let expected: Vec<BF16> = (0..n)
            .map(|i| BF16::from_f32_rounded(i as f32 + 1.0))
            .collect();

        add_bf16_inplace(&mut dst, &src);
        for i in 0..n {
            assert_eq!(dst[i], expected[i], "add_bf16_inplace tail=31 mismatch at {}", i);
        }
    }

    #[test]
    fn add_bf16_inplace_tail_33() {
        let n = 33;
        let mut dst: Vec<BF16> = (0..n).map(|i| BF16::from_f32_rounded(i as f32)).collect();
        let src: Vec<BF16> = (0..n).map(|i| BF16::from_f32_rounded(2.0)).collect();
        let expected: Vec<BF16> = (0..n)
            .map(|i| BF16::from_f32_rounded(i as f32 + 2.0))
            .collect();

        add_bf16_inplace(&mut dst, &src);
        for i in 0..n {
            assert_eq!(dst[i], expected[i], "add_bf16_inplace tail=33 mismatch at {}", i);
        }
    }

    // ── Batch cast round-trip ────────────────────────────────────────

    #[test]
    fn cast_bf16_f32_roundtrip() {
        let bf16_vals: Vec<BF16> = (0..33)
            .map(|i| BF16::from_f32_rounded(i as f32 * 0.75))
            .collect();
        let mut f32_buf = vec![0.0f32; 33];
        let mut bf16_buf = vec![BF16::ZERO; 33];

        cast_bf16_to_f32_batch(&bf16_vals, &mut f32_buf);
        cast_f32_to_bf16_batch(&f32_buf, &mut bf16_buf);

        for i in 0..33 {
            assert_eq!(bf16_buf[i], bf16_vals[i], "BF16 cast roundtrip mismatch at {}", i);
        }
    }

    #[test]
    fn cast_f16_f32_roundtrip() {
        // Use small values to stay within F16 range
        let f16_vals: Vec<F16> = (0..33)
            .map(|i| F16::from_f32_rounded(i as f32 * 0.5))
            .collect();
        let mut f32_buf = vec![0.0f32; 33];
        let mut f16_buf = vec![F16::ZERO; 33];

        cast_f16_to_f32_batch(&f16_vals, &mut f32_buf);
        cast_f32_to_f16_batch(&f32_buf, &mut f16_buf);

        for i in 0..33 {
            assert_eq!(f16_buf[i], f16_vals[i], "F16 cast roundtrip mismatch at {}", i);
        }
    }

    // ── mul_bf16_inplace ─────────────────────────────────────────────

    #[test]
    fn mul_bf16_inplace_basic() {
        let n = 17;
        let mut dst: Vec<BF16> = (0..n)
            .map(|i| BF16::from_f32_rounded(i as f32 + 1.0))
            .collect();
        let src: Vec<BF16> = (0..n).map(|_| BF16::from_f32_rounded(2.0)).collect();
        let expected: Vec<BF16> = (0..n)
            .map(|i| BF16::from_f32_rounded((i as f32 + 1.0) * 2.0))
            .collect();

        mul_bf16_inplace(&mut dst, &src);
        for i in 0..n {
            assert_eq!(dst[i], expected[i], "mul_bf16_inplace mismatch at {}", i);
        }
    }

    // ── add_f16_inplace ──────────────────────────────────────────────

    #[test]
    fn add_f16_inplace_tail_17() {
        let n = 17;
        let mut dst: Vec<F16> = (0..n).map(|i| F16::from_f32_rounded(i as f32)).collect();
        let src: Vec<F16> = (0..n)
            .map(|i| F16::from_f32_rounded(i as f32 * 0.5))
            .collect();
        let expected: Vec<F16> = (0..n)
            .map(|i| F16::from_f32_rounded(i as f32 + i as f32 * 0.5))
            .collect();

        add_f16_inplace(&mut dst, &src);
        for i in 0..n {
            assert_eq!(dst[i], expected[i], "add_f16_inplace tail=17 mismatch at {}", i);
        }
    }

    // ── mul_f16_inplace ──────────────────────────────────────────────

    #[test]
    fn mul_f16_inplace_basic() {
        let n = 17;
        let mut dst: Vec<F16> = (0..n)
            .map(|i| F16::from_f32_rounded(i as f32 + 1.0))
            .collect();
        let src: Vec<F16> = (0..n).map(|_| F16::from_f32_rounded(2.0)).collect();
        let expected: Vec<F16> = (0..n)
            .map(|i| F16::from_f32_rounded((i as f32 + 1.0) * 2.0))
            .collect();

        mul_f16_inplace(&mut dst, &src);
        for i in 0..n {
            assert_eq!(dst[i], expected[i], "mul_f16_inplace mismatch at {}", i);
        }
    }
}
