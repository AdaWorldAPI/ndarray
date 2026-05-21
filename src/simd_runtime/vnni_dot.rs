//! Runtime-dispatched `vnni_dot_u8_i8` — `u8 × i8 → i32` dot product.
//!
//! Three backends, selected once at first call via `LazyLock`:
//!
//! 1. **AVX-512 VNNI** (`avx512f + avx512vnni`) → `simd_amx::vnni_dot_u8_i8`
//!    wrapped with a scalar tail loop. The existing kernel processes
//!    `n - (n%64)` lanes and silently drops the K-tail (its current
//!    matvec caller pre-aligns rows so the tail is never the bug;
//!    a general-purpose dispatch surface cannot assume that).
//! 2. **AVX-VNNI ymm** (`avx2 + avxvnniint8`) → `simd_amx::vnni2_dot_u8_i8`
//!    which already includes its own tail handling.
//! 3. **Scalar** → `simd_amx::vnni_dot_u8_i8_scalar` (always correct,
//!    no SIMD, the safety floor).
//!
//! On this host (Sapphire Rapids: AMX + AVX-512 + VNNI), tier 1 fires.
//! On Arrow Lake / Meteor Lake U (AVX-VNNI but no AVX-512), tier 2.
//! On Pi 4 / pre-2013 x86 / any non-x86 target, tier 3.

use std::sync::LazyLock;

type VnniDotFn = unsafe fn(&[u8], &[i8]) -> i32;

/// Static dispatch pointer. Set once on first call to
/// [`vnni_dot_u8_i8`]; every subsequent call is one indirect call
/// through this pointer.
static VNNI_DOT_U8_I8_DISPATCH: LazyLock<VnniDotFn> = LazyLock::new(|| {
    let _caps = crate::hpc::simd_caps::simd_caps();
    #[cfg(target_arch = "x86_64")]
    {
        if _caps.avx512f && _caps.avx512vnni {
            return vnni_dot_u8_i8_avx512_with_tail as VnniDotFn;
        }
        if _caps.avx2 && _caps.avxvnniint8 {
            return vnni2_dot_u8_i8_safe_wrapper as VnniDotFn;
        }
    }
    vnni_dot_u8_i8_scalar_safe_wrapper as VnniDotFn
});

/// `u8 × i8 → i32` dot product, runtime-dispatched to the best
/// available backend on the current CPU.
///
/// Returns the i32 sum of `a[i] * b[i]` for `i in 0..min(a.len(), b.len())`.
/// Wrapping arithmetic on accumulation; `127 × 255 × n` fits in i32 for
/// `n < ~32K`, so this is safe for any realistic vector length.
///
/// # Performance
/// First call: ~1µs (LazyLock initialization + CPUID).
/// Subsequent calls: ~2-3ns overhead + the chosen backend's runtime
/// (typically ~1 cycle per 4-8 lanes on the SIMD tiers).
#[inline]
pub fn vnni_dot_u8_i8(a: &[u8], b: &[i8]) -> i32 {
    // SAFETY: the dispatch closure picked a function whose
    // `#[target_feature]` requirement is satisfied by the runtime CPU
    // caps it checked. Each backend handles its own slice boundary.
    unsafe { (*VNNI_DOT_U8_I8_DISPATCH)(a, b) }
}

// ────────────────────────────────────────────────────────────────────────
// Tier 1 — AVX-512 VNNI with scalar tail
// ────────────────────────────────────────────────────────────────────────

/// Wraps `simd_amx::vnni_dot_u8_i8` to handle the K-tail.
///
/// The underlying kernel computes `chunks = n / 64` and reduces the
/// chunked sum, silently dropping `n % 64` lanes. That's fine for its
/// matvec caller (which pre-aligns rows), but a general-purpose
/// dispatch surface needs the tail too.
///
/// # Safety
/// Caller must have feature-detected `avx512f + avx512vnni` at runtime.
/// The dispatch closure above checks this before installing this
/// function as the dispatch target.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni")]
unsafe fn vnni_dot_u8_i8_avx512_with_tail(a: &[u8], b: &[i8]) -> i32 {
    let n = a.len().min(b.len());
    let aligned = n - (n % 64);
    // SAFETY: avx512f + avx512vnni enabled on this function via
    // target_feature, satisfying simd_amx::vnni_dot_u8_i8's contract.
    // `aligned` is a multiple of 64 by construction; slicing both
    // operands to `aligned` length stays within their bounds.
    let mut total: i32 = if aligned > 0 {
        crate::simd_amx::vnni_dot_u8_i8(&a[..aligned], &b[..aligned])
    } else {
        0
    };
    for i in aligned..n {
        total = total.wrapping_add((a[i] as i32) * (b[i] as i32));
    }
    total
}

// ────────────────────────────────────────────────────────────────────────
// Tier 2 — AVX-VNNI 256-bit
// ────────────────────────────────────────────────────────────────────────

/// Thin safe wrapper around `simd_amx::vnni2_dot_u8_i8` — that kernel
/// already handles its K-tail (chunked at 32 + scalar remainder), so
/// no additional logic needed here. The wrapper exists only to give
/// the dispatch table a `VnniDotFn` pointer with the right signature.
///
/// # Safety
/// Caller must have feature-detected `avx2 + avxvnniint8` at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnniint8")]
unsafe fn vnni2_dot_u8_i8_safe_wrapper(a: &[u8], b: &[i8]) -> i32 {
    crate::simd_amx::vnni2_dot_u8_i8(a, b)
}

// ────────────────────────────────────────────────────────────────────────
// Tier 3 — Scalar fallback (always correct)
// ────────────────────────────────────────────────────────────────────────

/// Thin wrapper around `simd_amx::vnni_dot_u8_i8_scalar` for dispatch
/// table type-compatibility (the scalar function is `fn`, not `unsafe
/// fn`, so we wrap to match the `VnniDotFn` pointer type).
///
/// # Safety
/// No actual unsafety — the wrapped function is safe. The `unsafe`
/// qualifier exists only because the dispatch table's pointer type is
/// `unsafe fn` (to accommodate the AVX-512 / AVX-VNNI arms above which
/// genuinely need it).
unsafe fn vnni_dot_u8_i8_scalar_safe_wrapper(a: &[u8], b: &[i8]) -> i32 {
    crate::simd_amx::vnni_dot_u8_i8_scalar(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the runtime trampoline produces the same result as the
    /// scalar reference for a few representative shapes — aligned and
    /// non-aligned K (which exercises the AVX-512 wrapper's scalar
    /// tail handling).
    #[test]
    fn vnni_dot_matches_scalar_reference() {
        // Aligned K = 64: exercises the chunked-only path on tier 1.
        let a: Vec<u8> = (0..64).map(|i| ((i * 7 + 3) % 256) as u8).collect();
        let b: Vec<i8> = (0..64).map(|i| ((i * 11 + 5) % 256) as u8 as i8).collect();
        let got = vnni_dot_u8_i8(&a, &b);
        let expected: i32 = (0..64).map(|i| (a[i] as i32) * (b[i] as i32)).sum();
        assert_eq!(got, expected, "aligned K=64 mismatch");

        // K = 100 (k % 64 = 36): exercises the scalar tail wrapper on
        // tier 1, and the existing tail handling on tier 2.
        let a: Vec<u8> = (0..100).map(|i| ((i * 13 + 1) % 256) as u8).collect();
        let b: Vec<i8> = (0..100).map(|i| ((i * 17 + 9) % 256) as u8 as i8).collect();
        let got = vnni_dot_u8_i8(&a, &b);
        let expected: i32 = (0..100).map(|i| (a[i] as i32) * (b[i] as i32)).sum();
        assert_eq!(got, expected, "K=100 (tail) mismatch");

        // Small K = 7: less than any SIMD chunk, exercises the tail
        // path entirely (or the scalar tier directly).
        let a: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7];
        let b: Vec<i8> = vec![-1, 2, -3, 4, -5, 6, -7];
        let got = vnni_dot_u8_i8(&a, &b);
        let expected: i32 = (0..a.len()).map(|i| a[i] as i32 * b[i] as i32).sum();
        assert_eq!(got, expected, "small K=7 mismatch");
    }

    /// Confirm the LazyLock fires exactly once — subsequent calls
    /// reuse the same dispatch target.
    #[test]
    fn dispatch_target_stable() {
        // Force first call.
        let _ = vnni_dot_u8_i8(&[0u8; 1], &[0i8; 1]);
        let ptr1 = *VNNI_DOT_U8_I8_DISPATCH as *const ();
        let _ = vnni_dot_u8_i8(&[1u8; 1], &[1i8; 1]);
        let ptr2 = *VNNI_DOT_U8_I8_DISPATCH as *const ();
        assert_eq!(ptr1, ptr2, "dispatch target changed between calls");
    }
}
