#![forbid(unsafe_code)]
//! Which SIMD intrinsics are callable from SAFE code on rustc 1.98.1, per arch.
#[cfg(target_arch = "aarch64")]
pub mod a64 {
    use core::arch::aarch64::*;
    /// A: plain fn, baseline-feature intrinsic (neon).
    #[cfg(probe_a)]
    pub fn plain_neon(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t { vandq_u32(a, b) }
    /// B: annotated fn, same call.
    #[target_feature(enable = "neon")]
    pub fn annotated_neon(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t { vandq_u32(a, b) }
    /// C: plain fn calling the annotated SAFE fn.
    #[cfg(probe_c)]
    pub fn plain_calls_annotated(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t { annotated_neon(a, b) }
}
#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use core::arch::x86_64::*;
    /// A: plain fn, sse2 (baseline for x86_64).
    #[cfg(probe_a)]
    pub fn plain_sse2(a: __m128i, b: __m128i) -> __m128i { _mm_and_si128(a, b) }
    /// A2: plain fn, avx2 (baseline only under -Ctarget-cpu=x86-64-v3).
    #[cfg(probe_a2)]
    pub fn plain_avx2(a: __m256i, b: __m256i) -> __m256i { _mm256_and_si256(a, b) }
    /// A3: plain fn, avx512f ternarylogic (baseline only under x86-64-v4).
    #[cfg(probe_a3)]
    pub fn plain_avx512(a: __m512i, b: __m512i, c: __m512i) -> __m512i { _mm512_ternarylogic_epi64::<0x96>(a, b, c) }
    /// B: annotated fn.
    #[target_feature(enable = "avx2")]
    pub fn annotated_avx2(a: __m256i, b: __m256i) -> __m256i { _mm256_and_si256(a, b) }
    /// C: plain fn calling annotated safe fn.
    #[cfg(probe_c)]
    pub fn plain_calls_annotated(a: __m256i, b: __m256i) -> __m256i { annotated_avx2(a, b) }
}
#[cfg(target_arch = "wasm32")]
pub mod w {
    use core::arch::wasm32::*;
    /// A: plain fn, simd128 intrinsic.
    #[cfg(probe_a)]
    pub fn plain_wasm(a: v128, b: v128) -> v128 { v128_and(a, b) }
}
