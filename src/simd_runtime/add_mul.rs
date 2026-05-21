//! Runtime-dispatched `add_mul_f32` / `add_mul_f64` — slice-level FMA.
//!
//! Semantics: `acc[i] += a[i] * b[i]` for each lane, single-rounded
//! per IEEE 754 FMA. Three backends per type, selected once at first
//! call via `LazyLock`:
//!
//! 1. **AVX-512** (`avx512f`) → 16-wide / 8-wide `_mm512_fmadd_ps` /
//!    `_mm512_fmadd_pd` (AVX-512 implies FMA on every CPU that ships
//!    it; no separate FMA feature check needed).
//! 2. **AVX2 + FMA** (`avx2 + fma`) → 8-wide / 4-wide
//!    `_mm256_fmadd_ps` / `_mm256_fmadd_pd`.
//! 3. **NEON** (aarch64 + neon) → 4-wide / 2-wide `vfmaq_f32` /
//!    `vfmaq_f64`.
//! 4. **Scalar** → `f32::mul_add` / `f64::mul_add` (always correct,
//!    always single-rounded per IEEE).
//!
//! Unlike the other surfaces in this module, the FMA backends are NEW
//! kernel code — no pre-existing slice-level FMA primitive in tree to
//! delegate to. The compile-time `crate::simd_ops::add_mul_f32` from
//! PR #182 polyfills through the `F32x16` lane wrapper; the runtime
//! version skips that indirection for one more inlined intrinsic per
//! chunk.

use std::sync::LazyLock;

type AddMulF32Fn = unsafe fn(&mut [f32], &[f32], &[f32]);
type AddMulF64Fn = unsafe fn(&mut [f64], &[f64], &[f64]);

static ADD_MUL_F32_DISPATCH: LazyLock<AddMulF32Fn> = LazyLock::new(|| {
    let _caps = crate::hpc::simd_caps::simd_caps();
    #[cfg(target_arch = "x86_64")]
    {
        if _caps.avx512f {
            return add_mul_f32_avx512 as AddMulF32Fn;
        }
        if _caps.avx2 && _caps.fma {
            return add_mul_f32_avx2_fma as AddMulF32Fn;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if _caps.neon {
            return add_mul_f32_neon as AddMulF32Fn;
        }
    }
    add_mul_f32_scalar as AddMulF32Fn
});

static ADD_MUL_F64_DISPATCH: LazyLock<AddMulF64Fn> = LazyLock::new(|| {
    let _caps = crate::hpc::simd_caps::simd_caps();
    #[cfg(target_arch = "x86_64")]
    {
        if _caps.avx512f {
            return add_mul_f64_avx512 as AddMulF64Fn;
        }
        if _caps.avx2 && _caps.fma {
            return add_mul_f64_avx2_fma as AddMulF64Fn;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if _caps.neon {
            return add_mul_f64_neon as AddMulF64Fn;
        }
    }
    add_mul_f64_scalar as AddMulF64Fn
});

/// `acc[i] += a[i] * b[i]` (f32, single-rounded FMA).
///
/// Slice lengths truncated to the shortest. Wrapping behaviour
/// matches `f32::mul_add` lane-by-lane (NaN propagates, infinities
/// handled per IEEE 754).
#[inline]
pub fn add_mul_f32(acc: &mut [f32], a: &[f32], b: &[f32]) {
    // SAFETY: dispatch closure verified the runtime CPU caps satisfy
    // the selected function's `#[target_feature]` requirements.
    unsafe { (*ADD_MUL_F32_DISPATCH)(acc, a, b) }
}

/// `acc[i] += a[i] * b[i]` (f64, single-rounded FMA).
#[inline]
pub fn add_mul_f64(acc: &mut [f64], a: &[f64], b: &[f64]) {
    unsafe { (*ADD_MUL_F64_DISPATCH)(acc, a, b) }
}

// ────────────────────────────────────────────────────────────────────────
// AVX-512 backends (16 / 8 lanes per FMA instruction)
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn add_mul_f32_avx512(acc: &mut [f32], a: &[f32], b: &[f32]) {
    use core::arch::x86_64::{_mm512_fmadd_ps, _mm512_loadu_ps, _mm512_storeu_ps};
    let n = acc.len().min(a.len()).min(b.len());
    let chunks = n / 16;
    for c in 0..chunks {
        let off = c * 16;
        let va = _mm512_loadu_ps(a.as_ptr().add(off));
        let vb = _mm512_loadu_ps(b.as_ptr().add(off));
        let vc = _mm512_loadu_ps(acc.as_ptr().add(off));
        let r = _mm512_fmadd_ps(va, vb, vc);
        _mm512_storeu_ps(acc.as_mut_ptr().add(off), r);
    }
    for i in (chunks * 16)..n {
        acc[i] = a[i].mul_add(b[i], acc[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn add_mul_f64_avx512(acc: &mut [f64], a: &[f64], b: &[f64]) {
    use core::arch::x86_64::{_mm512_fmadd_pd, _mm512_loadu_pd, _mm512_storeu_pd};
    let n = acc.len().min(a.len()).min(b.len());
    let chunks = n / 8;
    for c in 0..chunks {
        let off = c * 8;
        let va = _mm512_loadu_pd(a.as_ptr().add(off));
        let vb = _mm512_loadu_pd(b.as_ptr().add(off));
        let vc = _mm512_loadu_pd(acc.as_ptr().add(off));
        let r = _mm512_fmadd_pd(va, vb, vc);
        _mm512_storeu_pd(acc.as_mut_ptr().add(off), r);
    }
    for i in (chunks * 8)..n {
        acc[i] = a[i].mul_add(b[i], acc[i]);
    }
}

// ────────────────────────────────────────────────────────────────────────
// AVX2 + FMA backends (8 / 4 lanes per FMA instruction)
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_mul_f32_avx2_fma(acc: &mut [f32], a: &[f32], b: &[f32]) {
    use core::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_storeu_ps};
    let n = acc.len().min(a.len()).min(b.len());
    let chunks = n / 8;
    for c in 0..chunks {
        let off = c * 8;
        let va = _mm256_loadu_ps(a.as_ptr().add(off));
        let vb = _mm256_loadu_ps(b.as_ptr().add(off));
        let vc = _mm256_loadu_ps(acc.as_ptr().add(off));
        let r = _mm256_fmadd_ps(va, vb, vc);
        _mm256_storeu_ps(acc.as_mut_ptr().add(off), r);
    }
    for i in (chunks * 8)..n {
        acc[i] = a[i].mul_add(b[i], acc[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn add_mul_f64_avx2_fma(acc: &mut [f64], a: &[f64], b: &[f64]) {
    use core::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_storeu_pd};
    let n = acc.len().min(a.len()).min(b.len());
    let chunks = n / 4;
    for c in 0..chunks {
        let off = c * 4;
        let va = _mm256_loadu_pd(a.as_ptr().add(off));
        let vb = _mm256_loadu_pd(b.as_ptr().add(off));
        let vc = _mm256_loadu_pd(acc.as_ptr().add(off));
        let r = _mm256_fmadd_pd(va, vb, vc);
        _mm256_storeu_pd(acc.as_mut_ptr().add(off), r);
    }
    for i in (chunks * 4)..n {
        acc[i] = a[i].mul_add(b[i], acc[i]);
    }
}

// ────────────────────────────────────────────────────────────────────────
// NEON backends (4 / 2 lanes per vfmaq instruction)
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_mul_f32_neon(acc: &mut [f32], a: &[f32], b: &[f32]) {
    use core::arch::aarch64::{vfmaq_f32, vld1q_f32, vst1q_f32};
    let n = acc.len().min(a.len()).min(b.len());
    let chunks = n / 4;
    for c in 0..chunks {
        let off = c * 4;
        let va = vld1q_f32(a.as_ptr().add(off));
        let vb = vld1q_f32(b.as_ptr().add(off));
        let vc = vld1q_f32(acc.as_ptr().add(off));
        let r = vfmaq_f32(vc, va, vb);
        vst1q_f32(acc.as_mut_ptr().add(off), r);
    }
    for i in (chunks * 4)..n {
        acc[i] = a[i].mul_add(b[i], acc[i]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_mul_f64_neon(acc: &mut [f64], a: &[f64], b: &[f64]) {
    use core::arch::aarch64::{vfmaq_f64, vld1q_f64, vst1q_f64};
    let n = acc.len().min(a.len()).min(b.len());
    let chunks = n / 2;
    for c in 0..chunks {
        let off = c * 2;
        let va = vld1q_f64(a.as_ptr().add(off));
        let vb = vld1q_f64(b.as_ptr().add(off));
        let vc = vld1q_f64(acc.as_ptr().add(off));
        let r = vfmaq_f64(vc, va, vb);
        vst1q_f64(acc.as_mut_ptr().add(off), r);
    }
    for i in (chunks * 2)..n {
        acc[i] = a[i].mul_add(b[i], acc[i]);
    }
}

// ────────────────────────────────────────────────────────────────────────
// Scalar fallback (always correct, IEEE 754 FMA via core)
// ────────────────────────────────────────────────────────────────────────

unsafe fn add_mul_f32_scalar(acc: &mut [f32], a: &[f32], b: &[f32]) {
    let n = acc.len().min(a.len()).min(b.len());
    for i in 0..n {
        acc[i] = a[i].mul_add(b[i], acc[i]);
    }
}

unsafe fn add_mul_f64_scalar(acc: &mut [f64], a: &[f64], b: &[f64]) {
    let n = acc.len().min(a.len()).min(b.len());
    for i in 0..n {
        acc[i] = a[i].mul_add(b[i], acc[i]);
    }
}

// ────────────────────────────────────────────────────────────────────────
// CpuOps DTO entry points — pub(super) wrappers for cpu_ops.rs to
// reference the tier-specific kernels by name in static const decls.
// Each one has the safety invariant guaranteed by the cpu_ops()
// LazyLock that installed the parent &'static CpuOps.
// ────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn add_mul_f32_avx512_safe(acc: &mut [f32], a: &[f32], b: &[f32]) {
    add_mul_f32_avx512(acc, a, b)
}

#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn add_mul_f64_avx512_safe(acc: &mut [f64], a: &[f64], b: &[f64]) {
    add_mul_f64_avx512(acc, a, b)
}

#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn add_mul_f32_avx2_fma_safe(acc: &mut [f32], a: &[f32], b: &[f32]) {
    add_mul_f32_avx2_fma(acc, a, b)
}

#[cfg(target_arch = "x86_64")]
pub(super) unsafe fn add_mul_f64_avx2_fma_safe(acc: &mut [f64], a: &[f64], b: &[f64]) {
    add_mul_f64_avx2_fma(acc, a, b)
}

#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn add_mul_f32_neon_safe(acc: &mut [f32], a: &[f32], b: &[f32]) {
    add_mul_f32_neon(acc, a, b)
}

#[cfg(target_arch = "aarch64")]
pub(super) unsafe fn add_mul_f64_neon_safe(acc: &mut [f64], a: &[f64], b: &[f64]) {
    add_mul_f64_neon(acc, a, b)
}

pub(super) unsafe fn add_mul_f32_scalar_safe(acc: &mut [f32], a: &[f32], b: &[f32]) {
    add_mul_f32_scalar(acc, a, b)
}

pub(super) unsafe fn add_mul_f64_scalar_safe(acc: &mut [f64], a: &[f64], b: &[f64]) {
    add_mul_f64_scalar(acc, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_add_mul_f32(acc: &[f32], a: &[f32], b: &[f32]) -> Vec<f32> {
        let n = acc.len().min(a.len()).min(b.len());
        (0..n).map(|i| a[i].mul_add(b[i], acc[i])).collect()
    }

    fn ref_add_mul_f64(acc: &[f64], a: &[f64], b: &[f64]) -> Vec<f64> {
        let n = acc.len().min(a.len()).min(b.len());
        (0..n).map(|i| a[i].mul_add(b[i], acc[i])).collect()
    }

    #[test]
    fn add_mul_f32_matches_scalar() {
        // Multiple sizes spanning aligned + tail for each backend
        // (16/8/4 lane widths → 16, 17, 25 hit boundaries).
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 64, 100] {
            let acc_init: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
            let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5) - 1.0).collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3) + 0.7).collect();
            let expected = ref_add_mul_f32(&acc_init, &a, &b);
            let mut acc = acc_init.clone();
            add_mul_f32(&mut acc, &a, &b);
            for (i, (got, want)) in acc.iter().zip(expected.iter()).enumerate() {
                assert_eq!(got.to_bits(), want.to_bits(), "n={n} i={i}: got {got} want {want}");
            }
        }
    }

    #[test]
    fn add_mul_f64_matches_scalar() {
        for n in [0, 1, 3, 4, 5, 7, 8, 16, 32, 100] {
            let acc_init: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
            let a: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5) - 1.0).collect();
            let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3) + 0.7).collect();
            let expected = ref_add_mul_f64(&acc_init, &a, &b);
            let mut acc = acc_init.clone();
            add_mul_f64(&mut acc, &a, &b);
            for (i, (got, want)) in acc.iter().zip(expected.iter()).enumerate() {
                assert_eq!(got.to_bits(), want.to_bits(), "n={n} i={i}: got {got} want {want}");
            }
        }
    }
}
