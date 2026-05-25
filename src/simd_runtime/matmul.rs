//! Runtime-dispatched matmul trampolines.
//!
//! These are thin wrappers over the existing public matmul surfaces
//! that ALREADY have internal runtime dispatch (PR #182 / #184 / #185
//! shipped the per-tier kernels and the dispatch helpers). The
//! trampolines here exist for two reasons:
//!
//! 1. **Consistent surface under `crate::simd_runtime::*`** —
//!    consumers using the runtime-dispatch path get a uniform import
//!    site for every matmul + every vector op.
//! 2. **Inline-elision opportunity** — these wrappers are
//!    `#[inline(always)]` so the call collapses to the inner dispatch
//!    call without any extra indirection at the consumer site (the
//!    matmul entry points themselves are NOT `#[inline]` because
//!    they're large; the wrapper is one branch).
//!
//! Backend chains (all already in tree, this module adds nothing new):
//!
//! - `matmul_bf16_to_f32`: AMX TDPBF16PS → VDPBF16PS → scalar (PR #182).
//! - `matmul_f32` (BF16 compute on AMX hosts): same chain (PR #182).
//! - `matmul_i8_to_i32`: AMX TDPBUSD → VPDPBUSD-zmm → VPDPBUSD-ymm → scalar (PR #184/#185).
//! - `gemm_u8_i8`: AMX TDPBUSD → compile-time avx512vnni / avxvnni / scalar (PR #185).
//!
//! Cost: zero on top of what the underlying functions already pay.

use crate::{ArrayView2, ArrayViewMut2};

/// BF16 × BF16 → f32 matmul. Runtime-dispatched.
///
/// Delegates to [`crate::hpc::amx_matmul::matmul_bf16_to_f32`], which
/// already runtime-dispatches AMX TDPBF16PS → VDPBF16PS → scalar.
#[inline(always)]
pub fn matmul_bf16_to_f32(
    lhs: ArrayView2<'_, crate::hpc::quantized::BF16>, rhs: ArrayView2<'_, crate::hpc::quantized::BF16>,
    out: ArrayViewMut2<'_, f32>,
) -> Result<(), crate::hpc::amx_matmul::MatmulError> {
    crate::hpc::amx_matmul::matmul_bf16_to_f32(lhs, rhs, out)
}

/// f32 × f32 → f32 matmul (BF16 compute on AMX hosts).
/// Runtime-dispatched per the underlying tier chain.
#[inline(always)]
pub fn matmul_f32(
    lhs: ArrayView2<'_, f32>, rhs: ArrayView2<'_, f32>, out: ArrayViewMut2<'_, f32>,
) -> Result<(), crate::hpc::amx_matmul::MatmulError> {
    crate::hpc::amx_matmul::matmul_f32(lhs, rhs, out)
}

/// i8 × i8 → i32 matmul. Runtime-dispatched to AMX TDPBUSD → VPDPBUSD-zmm →
/// VPDPBUSD-ymm → scalar with the sign-shift bias trick.
#[inline(always)]
pub fn matmul_i8_to_i32(
    lhs: ArrayView2<'_, i8>, rhs: ArrayView2<'_, i8>, out: ArrayViewMut2<'_, i32>,
) -> Result<(), crate::hpc::amx_matmul::MatmulError> {
    crate::hpc::amx_matmul::matmul_i8_to_i32(lhs, rhs, out)
}

/// `C = A · B` where A is M×K u8, B is K×N i8, C is M×N i32 (overwrite).
///
/// Delegates to [`crate::simd_int_ops::gemm_u8_i8`]. Tier 0 (runtime
/// AMX detection) was added by PR #185; tiers 1-3 (compile-time
/// avx512vnni / avxvnni / scalar) come from PR #182.
#[inline(always)]
pub fn gemm_u8_i8(a: &[u8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    crate::simd_int_ops::gemm_u8_i8(a, b, c, m, n, k)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Array2;

    #[test]
    fn matmul_bf16_trampoline_works() {
        use crate::hpc::quantized::BF16;
        let a: Array2<BF16> = Array2::from_shape_fn((16, 64), |(_i, _j)| BF16::from_f32(1.0));
        let b: Array2<BF16> = Array2::from_shape_fn((64, 16), |(_i, _j)| BF16::from_f32(0.5));
        let mut c: Array2<f32> = Array2::zeros((16, 16));
        matmul_bf16_to_f32(a.view(), b.view(), c.view_mut()).unwrap();
        for &v in c.iter() {
            // 64 lanes × 1.0 × 0.5 = 32.0; BF16 of 1.0 and 0.5 are exact.
            assert!((v - 32.0).abs() < 1e-3, "expected ~32.0, got {v}");
        }
    }

    #[test]
    fn gemm_u8_i8_trampoline_matches_scalar() {
        let m = 16;
        let n = 16;
        let k = 64;
        let a: Vec<u8> = (0..m * k).map(|i| ((i * 7 + 3) % 256) as u8).collect();
        let b: Vec<i8> = (0..k * n)
            .map(|i| ((i * 11 + 5) % 256) as u8 as i8)
            .collect();
        let mut c = vec![0i32; m * n];
        gemm_u8_i8(&a, &b, &mut c, m, n, k);
        // Spot-check c[0]: sum over k of a[k] * b[k*n]
        let expected_c0: i32 = (0..k).map(|kk| a[kk] as i32 * b[kk * n] as i32).sum();
        assert_eq!(c[0], expected_c0, "c[0] mismatch");
    }
}
