use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(chacha20_force_soft)] {
        pub(crate) mod soft;
    } else if #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))] {
        // AdaWorldAPI matryoshka: on an AVX-512 build the keystream double-round
        // rides `ndarray::simd::U32x16` (the polyfill lowers it to `__m512i`).
        // Takes precedence over the runtime-detected AVX2/SSE2 path below.
        pub(crate) mod ndarray_simd;
    } else if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        cfg_if! {
            if #[cfg(chacha20_force_avx2)] {
                pub(crate) mod avx2;
            } else if #[cfg(chacha20_force_sse2)] {
                pub(crate) mod sse2;
            } else {
                pub(crate) mod soft;
                pub(crate) mod avx2;
                pub(crate) mod sse2;
            }
        }
    } else if #[cfg(all(chacha20_force_neon, target_arch = "aarch64", target_feature = "neon"))] {
        pub(crate) mod neon;
    } else {
        pub(crate) mod soft;
    }
}
