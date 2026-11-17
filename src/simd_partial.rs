//! Masked tail load/store (`load_partial` / `store_partial`) for the chunk types — plan
//! `.claude/plans/simd-fruits-integration-v1.md` F4.
//!
//! Every chunk loop over a slice ends with a tail shorter than one register.
//! `load_partial` reads lanes `0..min(s.len(), LANES)` and zero-fills the rest;
//! `store_partial` writes only lanes `0..min(s.len(), LANES)` and touches
//! nothing else of the slice.
//!
//! Two realizations, selected by `cfg`:
//!
//! * **A, portable** (every arm unless B applies): the tail is staged through a
//!   zeroed stack array with one `copy_from_slice` memcpy, then
//!   `from_array` / `to_array`.
//! * **B, native masked memory ops** (fault-free: masked-off lanes are never
//!   read or written):
//!   * AVX-512F: `F32x16`, `F64x8`, `I32x16`, `U32x16` use `maskz_loadu` /
//!     `mask_storeu` with a low-`n`-bits `k` register; `U8x64` additionally
//!     needs AVX-512BW.
//!   * AVX: `F32x8` and `F64x4` use `vmaskmov` with a lane-index compare mask.
//!
//! The `nightly-simd` backend always takes realization A.

use crate::simd::{F32x16, F32x8, F64x4, F64x8, I32x16, U32x16, U8x64};

/// Generates `load_partial` / `store_partial` for one type. The load and store
/// bodies are supplied as blocks so both realizations share docs and framing.
macro_rules! partial_impl {
    (
        $ty:ident, $elem:ty, $lanes:expr,
        [$one:literal $two:literal $three:literal $zero:literal $nine:literal $five:literal],
        load($s:ident, $n:ident) $load:block,
        store($this:ident, $t:ident, $m:ident) $store:block
    ) => {
        impl $ty {
            #[doc = concat!(
                                        "Loads lanes `0..min(s.len(), LANES)` from `s` and zeroes the rest.\n\n",
                                        "Never reads past the end of `s`.\n\n",
                                        "```\n",
                                        "use ndarray::simd::", stringify!($ty), ";\n",
                                        "let v = ", stringify!($ty), "::load_partial(&[", stringify!($one), ", ",
                                        stringify!($two), ", ", stringify!($three), "]);\n",
                                        "assert_eq!(v.to_array()[..4], [", stringify!($one), ", ",
                                        stringify!($two), ", ", stringify!($three), ", ", stringify!($zero), "]);\n",
                                        "```"
                                    )]
            #[inline(always)]
            pub fn load_partial($s: &[$elem]) -> Self {
                let $n = $s.len().min($lanes);
                $load
            }

            #[doc = concat!(
                                        "Writes lanes `0..min(s.len(), LANES)` into `s`; nothing else is touched.\n\n",
                                        "Elements of `s` at index `LANES` and beyond are left unchanged.\n\n",
                                        "```\n",
                                        "use ndarray::simd::", stringify!($ty), ";\n",
                                        "let mut out = [", stringify!($nine), "; 3];\n",
                                        stringify!($ty), "::splat(", stringify!($five), ").store_partial(&mut out);\n",
                                        "assert_eq!(out, [", stringify!($five), "; 3]);\n",
                                        "```"
                                    )]
            #[inline(always)]
            pub fn store_partial(self, $t: &mut [$elem]) {
                let $this = self;
                let $m = $t.len().min($lanes);
                $store
            }
        }
    };
}

/// Realization A: stage the tail through a zeroed stack array (one memcpy).
#[allow(unused_macros)] // the AVX-512 arm realizes every type natively
macro_rules! partial_via_buffer {
    ($ty:ident, $elem:ty, $lanes:expr, $doc:tt) => {
        partial_impl!(
            $ty, $elem, $lanes, $doc,
            load(s, n) {
                let mut buf = [<$elem as Default>::default(); $lanes];
                buf[..n].copy_from_slice(&s[..n]);
                Self::from_array(buf)
            },
            store(this, t, n) {
                t[..n].copy_from_slice(&this.to_array()[..n]);
            }
        );
    };
}

// ---------------------------------------------------------------------------
// Native helpers (private): low-n-bit k masks.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", not(feature = "nightly-simd")))]
mod k_masks {
    use core::arch::x86_64::{__mmask16, __mmask8};

    /// Low `n` bits set (`n <= 16`), without overflow at `n == 16`.
    #[inline(always)]
    pub(super) fn mask16(n: usize) -> __mmask16 {
        if n >= 16 {
            0xFFFF
        } else {
            (1u16 << n) - 1
        }
    }

    /// Low `n` bits set (`n <= 8`), without overflow at `n == 8`.
    #[inline(always)]
    pub(super) fn mask8(n: usize) -> __mmask8 {
        if n >= 8 {
            0xFF
        } else {
            (1u8 << n) - 1
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    not(feature = "nightly-simd")
))]
mod k_masks64 {
    use core::arch::x86_64::__mmask64;

    /// Low `n` bits set (`n <= 64`), without overflow at `n == 64`.
    #[inline(always)]
    pub(super) fn mask64(n: usize) -> __mmask64 {
        if n >= 64 {
            u64::MAX
        } else {
            (1u64 << n) - 1
        }
    }
}

// ---------------------------------------------------------------------------
// F32x16 (AVX-512F masked move, else buffer)
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", not(feature = "nightly-simd")))]
partial_impl!(
    F32x16, f32, 16, [1.0 2.0 3.0 0.0 9.0 5.0],
    load(s, n) {
        // SAFETY: the pointer comes from a live slice; only the first n <= s.len()
        // lanes are enabled in k, and AVX-512 masked loads do not access (or fault
        // on) disabled lanes; loadu needs no alignment.
        unsafe { Self(core::arch::x86_64::_mm512_maskz_loadu_ps(k_masks::mask16(n), s.as_ptr())) }
    },
    store(this, t, n) {
        // SAFETY: the pointer comes from a live mutable slice; only the first
        // n <= t.len() lanes are enabled in k, and AVX-512 masked stores do not
        // access (or fault on) disabled lanes; storeu needs no alignment.
        unsafe {
            core::arch::x86_64::_mm512_mask_storeu_ps(t.as_mut_ptr(), k_masks::mask16(n), this.0)
        }
    }
);
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", not(feature = "nightly-simd"))))]
partial_via_buffer!(F32x16, f32, 16, [1.0 2.0 3.0 0.0 9.0 5.0]);

// ---------------------------------------------------------------------------
// F64x8
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", not(feature = "nightly-simd")))]
partial_impl!(
    F64x8, f64, 8, [1.0 2.0 3.0 0.0 9.0 5.0],
    load(s, n) {
        // SAFETY: the pointer comes from a live slice; only the first n <= s.len()
        // lanes are enabled in k, and AVX-512 masked loads do not access (or fault
        // on) disabled lanes; loadu needs no alignment.
        unsafe { Self(core::arch::x86_64::_mm512_maskz_loadu_pd(k_masks::mask8(n), s.as_ptr())) }
    },
    store(this, t, n) {
        // SAFETY: the pointer comes from a live mutable slice; only the first
        // n <= t.len() lanes are enabled in k, and AVX-512 masked stores do not
        // access (or fault on) disabled lanes; storeu needs no alignment.
        unsafe {
            core::arch::x86_64::_mm512_mask_storeu_pd(t.as_mut_ptr(), k_masks::mask8(n), this.0)
        }
    }
);
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", not(feature = "nightly-simd"))))]
partial_via_buffer!(F64x8, f64, 8, [1.0 2.0 3.0 0.0 9.0 5.0]);

// ---------------------------------------------------------------------------
// I32x16
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", not(feature = "nightly-simd")))]
partial_impl!(
    I32x16, i32, 16, [1 2 3 0 9 5],
    load(s, n) {
        // SAFETY: the pointer comes from a live slice; only the first n <= s.len()
        // lanes are enabled in k, and AVX-512 masked loads do not access (or fault
        // on) disabled lanes; loadu needs no alignment.
        unsafe {
            Self(core::arch::x86_64::_mm512_maskz_loadu_epi32(k_masks::mask16(n), s.as_ptr().cast()))
        }
    },
    store(this, t, n) {
        // SAFETY: the pointer comes from a live mutable slice; only the first
        // n <= t.len() lanes are enabled in k, and AVX-512 masked stores do not
        // access (or fault on) disabled lanes; storeu needs no alignment.
        unsafe {
            core::arch::x86_64::_mm512_mask_storeu_epi32(
                t.as_mut_ptr().cast(),
                k_masks::mask16(n),
                this.0,
            )
        }
    }
);
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", not(feature = "nightly-simd"))))]
partial_via_buffer!(I32x16, i32, 16, [1 2 3 0 9 5]);

// ---------------------------------------------------------------------------
// U32x16
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", not(feature = "nightly-simd")))]
partial_impl!(
    U32x16, u32, 16, [1 2 3 0 9 5],
    load(s, n) {
        // SAFETY: the pointer comes from a live slice; only the first n <= s.len()
        // lanes are enabled in k, and AVX-512 masked loads do not access (or fault
        // on) disabled lanes; loadu needs no alignment.
        unsafe {
            Self(core::arch::x86_64::_mm512_maskz_loadu_epi32(k_masks::mask16(n), s.as_ptr().cast()))
        }
    },
    store(this, t, n) {
        // SAFETY: the pointer comes from a live mutable slice; only the first
        // n <= t.len() lanes are enabled in k, and AVX-512 masked stores do not
        // access (or fault on) disabled lanes; storeu needs no alignment.
        unsafe {
            core::arch::x86_64::_mm512_mask_storeu_epi32(
                t.as_mut_ptr().cast(),
                k_masks::mask16(n),
                this.0,
            )
        }
    }
);
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f", not(feature = "nightly-simd"))))]
partial_via_buffer!(U32x16, u32, 16, [1 2 3 0 9 5]);

// ---------------------------------------------------------------------------
// U8x64 (AVX-512F + AVX-512BW masked byte move, else buffer)
// ---------------------------------------------------------------------------

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    not(feature = "nightly-simd")
))]
partial_impl!(
    U8x64, u8, 64, [1 2 3 0 9 5],
    load(s, n) {
        // SAFETY: the pointer comes from a live slice; only the first n <= s.len()
        // lanes are enabled in k, and AVX-512BW masked loads do not access (or
        // fault on) disabled lanes; loadu needs no alignment.
        unsafe {
            Self(core::arch::x86_64::_mm512_maskz_loadu_epi8(k_masks64::mask64(n), s.as_ptr().cast()))
        }
    },
    store(this, t, n) {
        // SAFETY: the pointer comes from a live mutable slice; only the first
        // n <= t.len() lanes are enabled in k, and AVX-512BW masked stores do not
        // access (or fault on) disabled lanes; storeu needs no alignment.
        unsafe {
            core::arch::x86_64::_mm512_mask_storeu_epi8(
                t.as_mut_ptr().cast(),
                k_masks64::mask64(n),
                this.0,
            )
        }
    }
);
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    not(feature = "nightly-simd")
)))]
partial_via_buffer!(U8x64, u8, 64, [1 2 3 0 9 5]);

// ---------------------------------------------------------------------------
// F32x8 (AVX vmaskmov, else buffer)
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx", not(feature = "nightly-simd")))]
partial_impl!(
    F32x8, f32, 8, [1.0 2.0 3.0 0.0 9.0 5.0],
    load(s, n) {
        use core::arch::x86_64::*;
        // SAFETY: the pointer comes from a live slice; lane i is active iff
        // i < n <= s.len(), and vmaskmov suppresses faults and does not access
        // masked-off elements; no alignment needed.
        unsafe {
            let lt = _mm256_cmp_ps::<_CMP_LT_OQ>(
                _mm256_setr_ps(0., 1., 2., 3., 4., 5., 6., 7.),
                _mm256_set1_ps(n as f32),
            );
            Self(_mm256_maskload_ps(s.as_ptr(), _mm256_castps_si256(lt)))
        }
    },
    store(this, t, n) {
        use core::arch::x86_64::*;
        // SAFETY: the pointer comes from a live mutable slice; lane i is active
        // iff i < n <= t.len(), and vmaskmov suppresses faults and does not
        // access masked-off elements; no alignment needed.
        unsafe {
            let lt = _mm256_cmp_ps::<_CMP_LT_OQ>(
                _mm256_setr_ps(0., 1., 2., 3., 4., 5., 6., 7.),
                _mm256_set1_ps(n as f32),
            );
            _mm256_maskstore_ps(t.as_mut_ptr(), _mm256_castps_si256(lt), this.0)
        }
    }
);
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx", not(feature = "nightly-simd"))))]
partial_via_buffer!(F32x8, f32, 8, [1.0 2.0 3.0 0.0 9.0 5.0]);

// ---------------------------------------------------------------------------
// F64x4 (AVX vmaskmov, else buffer)
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx", not(feature = "nightly-simd")))]
partial_impl!(
    F64x4, f64, 4, [1.0 2.0 3.0 0.0 9.0 5.0],
    load(s, n) {
        use core::arch::x86_64::*;
        // SAFETY: the pointer comes from a live slice; lane i is active iff
        // i < n <= s.len(), and vmaskmov suppresses faults and does not access
        // masked-off elements; no alignment needed.
        unsafe {
            let lt = _mm256_cmp_pd::<_CMP_LT_OQ>(
                _mm256_setr_pd(0., 1., 2., 3.),
                _mm256_set1_pd(n as f64),
            );
            Self(_mm256_maskload_pd(s.as_ptr(), _mm256_castpd_si256(lt)))
        }
    },
    store(this, t, n) {
        use core::arch::x86_64::*;
        // SAFETY: the pointer comes from a live mutable slice; lane i is active
        // iff i < n <= t.len(), and vmaskmov suppresses faults and does not
        // access masked-off elements; no alignment needed.
        unsafe {
            let lt = _mm256_cmp_pd::<_CMP_LT_OQ>(
                _mm256_setr_pd(0., 1., 2., 3.),
                _mm256_set1_pd(n as f64),
            );
            _mm256_maskstore_pd(t.as_mut_ptr(), _mm256_castpd_si256(lt), this.0)
        }
    }
);
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx", not(feature = "nightly-simd"))))]
partial_via_buffer!(F64x4, f64, 4, [1.0 2.0 3.0 0.0 9.0 5.0]);
