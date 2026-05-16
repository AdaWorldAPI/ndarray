//! Half-precision dot products: `dot_bf16` and `dot_f16`.
//!
//! Both functions accept raw bit-pattern slices (`&[u16]`) and accumulate the
//! inner product in f32 for full numerical range.
//!
//! ## Implementation strategy (v1 — "widen then dot")
//!
//! 1. Interpret each `u16` as a BF16 / F16 bit pattern.
//! 2. Convert both slices to temporary `Vec<f32>` via the
//!    `cast_bf16_to_f32_batch` / `cast_f16_to_f32_batch` helpers from
//!    `simd_half`.  Those helpers already use `BF16x16` / `F16x16` for
//!    chunks-of-16 throughput with a scalar tail.
//! 3. Call `<f32 as BlasFloat>::backend_dot` which dispatches to the fastest
//!    available f32 dot (AVX-512, AVX2, NEON, or scalar) at runtime.
//!
//! An AVX-512 BF16 `vdpbf16ps` fast path (no widening needed) is deferred;
//! see `// TODO(ws1)` below.
//!
//! Ported from candle's per-arch `vec_dot_{bf16,f16}` in
//! `candle-core/src/cpu/mod.rs` (AVX2 FMA on packed half-precision lanes).

use crate::backend::BlasFloat;
use crate::hpc::quantized::{BF16, F16};
use crate::simd_half::{cast_bf16_to_f32_batch, cast_f16_to_f32_batch};

// TODO(ws1): Wire AVX-512 BF16 `vdpbf16ps` fast path.
//
// When `target_feature = "avx512bf16"` is available at compile time, we can
// compute the dot product directly on the u16 bit patterns without widening to
// f32 first.  The intrinsic accumulates into a 32-bit register with twice the
// throughput of the widen-then-dot path.
//
// Deferred because: (a) the fork's `simd_avx512.rs` does not yet expose a
// `vdpbf16ps` wrapper, and (b) the widen path already gives correct results
// and shares the backend's runtime dispatch.  Wire this once WS-2/WS-5 land
// the AVX-512 BF16 intrinsic wrapper.

// ============================================================================
// Public API
// ============================================================================

/// Dot product of two BF16 slices, accumulated in f32.
///
/// Each element of `a` and `b` is a raw BF16 bit pattern (the upper 16 bits
/// of the corresponding IEEE 754 binary32 value).  If the slices differ in
/// length, only the first `min(a.len(), b.len())` elements are used.
///
/// Accumulation is performed in f32, so the result has full f32 range even
/// when individual BF16 values are large.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::dot_half::dot_bf16;
/// use ndarray::hpc::quantized::BF16;
///
/// // [1.0, 2.0] · [3.0, 4.0] = 11.0
/// let a: Vec<u16> = [1.0f32, 2.0f32].iter().map(|&v| BF16::from_f32(v).0).collect();
/// let b: Vec<u16> = [3.0f32, 4.0f32].iter().map(|&v| BF16::from_f32(v).0).collect();
/// let result = dot_bf16(&a, &b);
/// assert!((result - 11.0).abs() < 0.1, "got {result}");
/// ```
pub fn dot_bf16(a: &[u16], b: &[u16]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    // Reinterpret raw u16 bit patterns as BF16 values.
    // SAFETY: BF16 is #[repr(transparent)] over u16, so &[u16] and &[BF16]
    // have the same layout, size, and alignment.  We are only reading the
    // memory through the new type, never writing, so this is safe.
    let a_bf16: &[BF16] = unsafe { core::slice::from_raw_parts(a.as_ptr() as *const BF16, n) };
    let b_bf16: &[BF16] = unsafe { core::slice::from_raw_parts(b.as_ptr() as *const BF16, n) };

    let mut a_f32 = vec![0.0f32; n];
    let mut b_f32 = vec![0.0f32; n];
    cast_bf16_to_f32_batch(a_bf16, &mut a_f32);
    cast_bf16_to_f32_batch(b_bf16, &mut b_f32);

    <f32 as BlasFloat>::backend_dot(&a_f32, &b_f32)
}

/// Dot product of two F16 slices, accumulated in f32.
///
/// Each element of `a` and `b` is a raw IEEE 754 binary16 bit pattern.  If
/// the slices differ in length, only the first `min(a.len(), b.len())`
/// elements are used.
///
/// Accumulation is performed in f32.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::dot_half::dot_f16;
/// use ndarray::hpc::quantized::F16;
///
/// // [1.0, 2.0] · [3.0, 4.0] = 11.0
/// let a: Vec<u16> = [1.0f32, 2.0f32].iter().map(|&v| F16::from_f32(v).0).collect();
/// let b: Vec<u16> = [3.0f32, 4.0f32].iter().map(|&v| F16::from_f32(v).0).collect();
/// let result = dot_f16(&a, &b);
/// assert!((result - 11.0).abs() < 0.1, "got {result}");
/// ```
pub fn dot_f16(a: &[u16], b: &[u16]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    // Reinterpret raw u16 bit patterns as F16 values.
    // SAFETY: F16 is #[repr(transparent)] over u16, so &[u16] and &[F16]
    // have the same layout, size, and alignment.  We are only reading the
    // memory through the new type, never writing, so this is safe.
    let a_f16: &[F16] = unsafe { core::slice::from_raw_parts(a.as_ptr() as *const F16, n) };
    let b_f16: &[F16] = unsafe { core::slice::from_raw_parts(b.as_ptr() as *const F16, n) };

    let mut a_f32 = vec![0.0f32; n];
    let mut b_f32 = vec![0.0f32; n];
    cast_f16_to_f32_batch(a_f16, &mut a_f32);
    cast_f16_to_f32_batch(b_f16, &mut b_f32);

    <f32 as BlasFloat>::backend_dot(&a_f32, &b_f32)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::quantized::{BF16, F16};

    // ── helpers ──────────────────────────────────────────────────────────────

    fn bf16_bits(v: f32) -> u16 {
        BF16::from_f32_rounded(v).0
    }

    fn f16_bits(v: f32) -> u16 {
        F16::from_f32_rounded(v).0
    }

    fn bf16_vec(vals: &[f32]) -> Vec<u16> {
        vals.iter().map(|&v| bf16_bits(v)).collect()
    }

    fn f16_vec(vals: &[f32]) -> Vec<u16> {
        vals.iter().map(|&v| f16_bits(v)).collect()
    }

    // ── 1. zero-length slices ─────────────────────────────────────────────

    #[test]
    fn dot_bf16_empty() {
        assert_eq!(dot_bf16(&[], &[]), 0.0f32);
    }

    #[test]
    fn dot_f16_empty() {
        assert_eq!(dot_f16(&[], &[]), 0.0f32);
    }

    // ── 2. single-element ─────────────────────────────────────────────────

    #[test]
    fn dot_bf16_single() {
        let a = bf16_vec(&[3.0]);
        let b = bf16_vec(&[4.0]);
        let got = dot_bf16(&a, &b);
        // BF16 has ~2 decimal digits of precision; allow 1% tolerance
        assert!((got - 12.0).abs() < 0.2, "expected 12.0, got {got}");
    }

    #[test]
    fn dot_f16_single() {
        let a = f16_vec(&[3.0]);
        let b = f16_vec(&[4.0]);
        let got = dot_f16(&a, &b);
        assert!((got - 12.0).abs() < 0.05, "expected 12.0, got {got}");
    }

    // ── 3. lane-aligned (exactly 16 elements) ────────────────────────────

    #[test]
    fn dot_bf16_16_elements() {
        // [1,2,...,16] · [1,1,...,1] = 136
        let a_vals: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let b_vals = vec![1.0f32; 16];
        let a = bf16_vec(&a_vals);
        let b = bf16_vec(&b_vals);
        let got = dot_bf16(&a, &b);
        let expected = 136.0f32;
        assert!((got - expected).abs() < 1.0, "expected {expected}, got {got}");
    }

    #[test]
    fn dot_f16_16_elements() {
        let a_vals: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let b_vals = vec![1.0f32; 16];
        let a = f16_vec(&a_vals);
        let b = f16_vec(&b_vals);
        let got = dot_f16(&a, &b);
        let expected = 136.0f32;
        assert!((got - expected).abs() < 0.5, "expected {expected}, got {got}");
    }

    // ── 4. tail sizes (17, 31, 100 elements) ─────────────────────────────

    #[test]
    fn dot_bf16_tail_17() {
        // [1.0; 17] · [2.0; 17] = 34.0
        let a = bf16_vec(&vec![1.0f32; 17]);
        let b = bf16_vec(&vec![2.0f32; 17]);
        let got = dot_bf16(&a, &b);
        assert!((got - 34.0).abs() < 0.5, "expected 34.0, got {got}");
    }

    #[test]
    fn dot_bf16_tail_31() {
        let a = bf16_vec(&vec![1.0f32; 31]);
        let b = bf16_vec(&vec![1.0f32; 31]);
        let got = dot_bf16(&a, &b);
        assert!((got - 31.0).abs() < 0.5, "expected 31.0, got {got}");
    }

    #[test]
    fn dot_bf16_tail_100() {
        // [0.1; 100] · [0.1; 100] = 1.0
        let a = bf16_vec(&vec![0.1f32; 100]);
        let b = bf16_vec(&vec![0.1f32; 100]);
        let got = dot_bf16(&a, &b);
        assert!((got - 1.0).abs() < 0.05, "expected 1.0, got {got}");
    }

    #[test]
    fn dot_f16_tail_17() {
        let a = f16_vec(&vec![1.0f32; 17]);
        let b = f16_vec(&vec![2.0f32; 17]);
        let got = dot_f16(&a, &b);
        assert!((got - 34.0).abs() < 0.5, "expected 34.0, got {got}");
    }

    #[test]
    fn dot_f16_tail_31() {
        let a = f16_vec(&vec![1.0f32; 31]);
        let b = f16_vec(&vec![1.0f32; 31]);
        let got = dot_f16(&a, &b);
        assert!((got - 31.0).abs() < 0.5, "expected 31.0, got {got}");
    }

    #[test]
    fn dot_f16_tail_100() {
        let a = f16_vec(&vec![0.1f32; 100]);
        let b = f16_vec(&vec![0.1f32; 100]);
        let got = dot_f16(&a, &b);
        assert!((got - 1.0).abs() < 0.05, "expected 1.0, got {got}");
    }

    // ── 5. scalar cross-check (random-ish fixed values) ──────────────────

    #[test]
    fn dot_bf16_scalar_crosscheck() {
        // Use a deterministic "random" vector: powers of small primes mod a
        // small range so values stay representable in BF16.
        let n = 64usize;
        let a_f32: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 17) as f32 * 0.25).collect();
        let b_f32: Vec<f32> = (0..n).map(|i| ((i * 11 + 5) % 13) as f32 * 0.125).collect();

        // Reference: compute in f32 from BF16-rounded inputs (matches what the
        // function will see after widening).
        let a_bf16: Vec<f32> = a_f32.iter().map(|&v| BF16::from_f32_rounded(v).to_f32()).collect();
        let b_bf16: Vec<f32> = b_f32.iter().map(|&v| BF16::from_f32_rounded(v).to_f32()).collect();
        let reference: f32 = a_bf16.iter().zip(b_bf16.iter()).map(|(x, y)| x * y).sum();

        let a = bf16_vec(&a_f32);
        let b = bf16_vec(&b_f32);
        let got = dot_bf16(&a, &b);

        // BF16 accumulation in f32 — allow 0.5% relative error
        let tol = reference.abs() * 0.005 + 1e-4;
        assert!((got - reference).abs() <= tol, "cross-check failed: ref={reference}, got={got}");
    }

    #[test]
    fn dot_f16_scalar_crosscheck() {
        let n = 64usize;
        let a_f32: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 17) as f32 * 0.25).collect();
        let b_f32: Vec<f32> = (0..n).map(|i| ((i * 11 + 5) % 13) as f32 * 0.125).collect();

        // Reference: compute in f32 from F16-rounded inputs.
        let a_f16: Vec<f32> = a_f32.iter().map(|&v| F16::from_f32_rounded(v).to_f32()).collect();
        let b_f16: Vec<f32> = b_f32.iter().map(|&v| F16::from_f32_rounded(v).to_f32()).collect();
        let reference: f32 = a_f16.iter().zip(b_f16.iter()).map(|(x, y)| x * y).sum();

        let a = f16_vec(&a_f32);
        let b = f16_vec(&b_f32);
        let got = dot_f16(&a, &b);

        let tol = reference.abs() * 0.002 + 1e-5;
        assert!((got - reference).abs() <= tol, "cross-check failed: ref={reference}, got={got}");
    }

    // ── 6. mismatched lengths — use shorter slice ─────────────────────────

    #[test]
    fn dot_bf16_mismatched_lengths() {
        // a has 3 elements, b has 5 — only first 3 used: [1·1, 2·2, 3·3] = 14
        let a = bf16_vec(&[1.0, 2.0, 3.0]);
        let b = bf16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let got = dot_bf16(&a, &b);
        assert!((got - 14.0).abs() < 0.2, "expected 14.0, got {got}");
    }

    #[test]
    fn dot_f16_mismatched_lengths() {
        let a = f16_vec(&[1.0, 2.0, 3.0]);
        let b = f16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let got = dot_f16(&a, &b);
        assert!((got - 14.0).abs() < 0.05, "expected 14.0, got {got}");
    }

    // ── 7. orthogonal vectors → zero ─────────────────────────────────────

    #[test]
    fn dot_bf16_orthogonal() {
        // [1, -1, 1, -1, ...] · [1, 1, 1, 1, ...] = 0
        let vals_a: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let vals_b = vec![1.0f32; 16];
        let a = bf16_vec(&vals_a);
        let b = bf16_vec(&vals_b);
        let got = dot_bf16(&a, &b);
        assert!(got.abs() < 0.1, "expected 0.0, got {got}");
    }

    #[test]
    fn dot_f16_orthogonal() {
        let vals_a: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let vals_b = vec![1.0f32; 16];
        let a = f16_vec(&vals_a);
        let b = f16_vec(&vals_b);
        let got = dot_f16(&a, &b);
        assert!(got.abs() < 0.05, "expected 0.0, got {got}");
    }
}
