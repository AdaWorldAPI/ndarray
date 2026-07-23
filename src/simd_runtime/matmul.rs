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

/// i8 × i8 → i32 matmul. Runtime-dispatched.
///
/// - **x86_64**: delegates unchanged to
///   [`crate::hpc::amx_matmul::matmul_i8_to_i32`] — AMX TDPBUSD →
///   VPDPBUSD-zmm → VPDPBUSD-ymm → scalar, with the sign-shift bias trick
///   documented there.
/// - **wasm32 + simd128**: dispatches to
///   `crate::simd_wasm::wasm32_simd::matmul_i8_to_i32_wasm` — a v128
///   i16-widen + i32-pairwise-reduce dot product. No sign-shift bias
///   trick is needed there: signed×signed multiply is native to that
///   widening path (unlike AMX/VPDPBUSD, which are unsigned×signed).
///   (Plain backtick, not an intra-doc link: that function only exists
///   under wasm32+simd128, narrower than this doc comment's own
///   `not(target_arch = "x86_64")` gate, so a `[...]` link would be
///   "broken" on e.g. aarch64 or plain wasm32 builds.)
/// - **Everywhere else off x86_64** (wasm32 without simd128, other
///   non-x86_64 targets): a portable scalar reference, bit-identical to
///   the x86_64 tier's scalar fallback.
///
/// `crate::hpc::amx_matmul` is `#[cfg(target_arch = "x86_64")]`-gated in
/// its entirety (it carries `std::arch::asm!` AMX tile code that only
/// makes sense there), so its `MatmulError` type is unreachable off
/// x86_64. The portable path below returns a distinct, same-shaped
/// `MatmulError` defined in this module instead — the two never collide
/// because exactly one of the two `matmul_i8_to_i32` definitions here is
/// ever compiled for a given target. (Plain backtick, not a link: that
/// `MatmulError` is itself `#[cfg(not(target_arch = "x86_64"))]`-gated,
/// so it doesn't exist in the x86_64 compilation this very doc comment
/// belongs to — a `[...]` link here would be "broken" on x86_64.)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn matmul_i8_to_i32(
    lhs: ArrayView2<'_, i8>, rhs: ArrayView2<'_, i8>, out: ArrayViewMut2<'_, i32>,
) -> Result<(), crate::hpc::amx_matmul::MatmulError> {
    crate::hpc::amx_matmul::matmul_i8_to_i32(lhs, rhs, out)
}

/// Portable (off-x86_64) error type for [`matmul_i8_to_i32`]. Mirrors the
/// shape of `crate::hpc::amx_matmul::MatmulError` (unreachable here — see
/// that function's doc comment) minus the x86_64-only `AmxUnavailable`
/// variant, which has no meaning off x86_64.
#[cfg(not(target_arch = "x86_64"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatmulError {
    /// Shapes don't satisfy `lhs:(M,K), rhs:(K,N), out:(M,N)`.
    ShapeMismatch {
        /// Shape of the LHS view, `(rows, cols)`.
        lhs: (usize, usize),
        /// Shape of the RHS view, `(rows, cols)`.
        rhs: (usize, usize),
        /// Shape of the output view, `(rows, cols)`.
        out: (usize, usize),
    },
    /// Output tensor is not row-contiguous (column stride != 1).
    NonContiguousOutput,
}

#[cfg(not(target_arch = "x86_64"))]
impl std::fmt::Display for MatmulError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatmulError::ShapeMismatch { lhs, rhs, out } => write!(
                f,
                "shape mismatch: lhs={:?} rhs={:?} out={:?}; expected lhs:(M,K), rhs:(K,N), out:(M,N)",
                lhs, rhs, out
            ),
            MatmulError::NonContiguousOutput => f.write_str("output must be row-contiguous (col stride = 1)"),
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
impl std::error::Error for MatmulError {}

/// Validate `lhs:(M,K) × rhs:(K,N) → out:(M,N)` and that `out` is
/// row-contiguous. Portable mirror of the private `check_shapes` helper
/// in `hpc::amx_matmul` (that module is x86_64-only).
#[cfg(not(target_arch = "x86_64"))]
fn check_shapes_portable(
    lhs: &ArrayView2<'_, i8>, rhs: &ArrayView2<'_, i8>, out: &ArrayViewMut2<'_, i32>,
) -> Result<(usize, usize, usize), MatmulError> {
    let (m, k) = lhs.dim();
    let (kr, n) = rhs.dim();
    let (mo, no) = out.dim();
    if k != kr || m != mo || n != no {
        return Err(MatmulError::ShapeMismatch {
            lhs: (m, k),
            rhs: (kr, n),
            out: (mo, no),
        });
    }
    // Output must be row-stride-1 (writes are linear per row).
    let strides = out.strides();
    if strides[1] != 1 {
        return Err(MatmulError::NonContiguousOutput);
    }
    Ok((m, n, k))
}

/// Copy a possibly-strided 2-D `i8` view into a contiguous row-major Vec.
/// Portable mirror of the private `pack_contig` helper in `hpc::amx_matmul`.
#[cfg(not(target_arch = "x86_64"))]
fn pack_contig_i8(view: &ArrayView2<'_, i8>) -> Vec<i8> {
    let (rows, cols) = view.dim();
    let mut buf = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            buf.push(view[[r, c]]);
        }
    }
    buf
}

/// Scalar i8×i8→i32 GEMM reference — bit-identical math to
/// `hpc::amx_matmul::matmul_i8_to_i32`'s tier-4 scalar fallback (plain
/// `i32` accumulation, no sign-shift bias). That fallback is inline
/// inside an x86_64-only function, so this is a portable copy of the
/// same loop rather than a cross-module call.
#[cfg(all(
    not(target_arch = "x86_64"),
    not(all(target_arch = "wasm32", target_feature = "simd128"))
))]
fn matmul_i8_scalar_reference(a: &[i8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p] as i32;
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j] as i32;
            }
        }
    }
}

/// i8 × i8 → i32 matmul — portable (off-x86_64) path. wasm32+simd128
/// dispatches to the v128 dot-product kernel in
/// `crate::simd_wasm::wasm32_simd::matmul_i8_to_i32_wasm`; everywhere
/// else (wasm32 without simd128, other non-x86_64 targets) falls through
/// to `matmul_i8_scalar_reference`. Both are plain backtick references,
/// not intra-doc links: each is `#[cfg]`-gated narrower than this
/// function itself, so a `[...]` link would be "broken" whichever one
/// isn't compiled for the current target. See the x86_64 sibling
/// definition's doc comment for the full tier rationale.
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn matmul_i8_to_i32(
    lhs: ArrayView2<'_, i8>, rhs: ArrayView2<'_, i8>, mut out: ArrayViewMut2<'_, i32>,
) -> Result<(), MatmulError> {
    let (m, n, k) = check_shapes_portable(&lhs, &rhs, &out)?;
    let a = pack_contig_i8(&lhs);
    let b = pack_contig_i8(&rhs);
    let mut c = vec![0i32; m * n];

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        crate::simd_wasm::wasm32_simd::matmul_i8_to_i32_wasm(&a, &b, &mut c, m, n, k);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        matmul_i8_scalar_reference(&a, &b, &mut c, m, n, k);
    }

    // Write i32 result back into the (possibly strided) output.
    let (rows, cols) = out.dim();
    debug_assert_eq!(c.len(), rows * cols);
    for r in 0..rows {
        for col in 0..cols {
            out[[r, col]] = c[r * cols + col];
        }
    }
    Ok(())
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

    /// Portable-path parity test for `matmul_i8_to_i32`: on wasm32+simd128
    /// this exercises `crate::simd_wasm::wasm32_simd::matmul_i8_to_i32_wasm`
    /// end-to-end through the public `ArrayView2` entry point (the
    /// self-contained kernel-level parity test lives in
    /// `simd_wasm.rs::wasm32_simd::tests::wasm_matmul_i8_matches_scalar`;
    /// this one instead proves the trampoline wiring itself). `K = 20` is
    /// deliberately not a multiple of 16, exercising the scalar tail.
    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn matmul_i8_to_i32_portable_matches_scalar() {
        let m = 5;
        let n = 4;
        let k = 20;
        let a = Array2::<i8>::from_shape_fn((m, k), |(i, j)| (((i * 7 + j * 3) as i32 % 13) - 6) as i8);
        let b = Array2::<i8>::from_shape_fn((k, n), |(i, j)| (((i * 5 + j * 11) as i32 % 17) - 8) as i8);
        let mut out = Array2::<i32>::zeros((m, n));
        matmul_i8_to_i32(a.view(), b.view(), out.view_mut()).expect("portable i8 matmul");

        // Scalar reference — identical math to `hpc::amx_matmul`'s tier-4.
        let mut expect = Array2::<i32>::zeros((m, n));
        for i in 0..m {
            for p in 0..k {
                let av = a[[i, p]] as i32;
                for j in 0..n {
                    expect[[i, j]] += av * b[[p, j]] as i32;
                }
            }
        }
        assert_eq!(out, expect);
    }
}
