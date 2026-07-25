#![cfg(any(feature = "cipher", feature = "rng"))]

use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(chacha20_backend = "soft")] {
        pub(crate) mod soft;
    } else if #[cfg(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            not(chacha20_backend = "avx2"),
            not(chacha20_backend = "sse2"),
            not(chacha20_backend = "avx512"),
        ),
        all(target_arch = "wasm32", target_feature = "simd128"),
    ))] {
        // The AdaWorldAPI matryoshka: replaces the intrinsic vector backends
        // with one expressed over `ndarray::simd::U32x16`. Compile-time
        // selected — `ndarray` lowers U32x16 to the AVX-512 lane on x86_64 and
        // to `[U32x4; 4]` on wasm32.
        pub(crate) mod ndarray_simd;
        // The RNG path still rides upstream's backends; only the cipher
        // keystream is polyfilled here. `soft` stays compiled for it.
        pub(crate) mod soft;
        #[cfg(all(feature = "rng", any(target_arch = "x86", target_arch = "x86_64")))]
        pub(crate) mod avx2;
    } else if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        cfg_if! {
            if #[cfg(all(chacha20_avx512, chacha20_backend = "avx512"))] {
                pub(crate) mod avx512;
                // AVX-2 backend needed for RNG if enabled
                #[cfg(feature = "rng")]
                pub(crate) mod avx2;
            } else if #[cfg(chacha20_backend = "avx2")] {
                pub(crate) mod avx2;
            } else if #[cfg(chacha20_backend = "sse2")] {
                pub(crate) mod sse2;
            } else {
                pub(crate) mod soft;
                #[cfg(chacha20_avx512)]
                pub(crate) mod avx512;
                pub(crate) mod avx2;
                pub(crate) mod sse2;
            }
        }
    } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
        pub(crate) mod neon;
    } else {
        pub(crate) mod soft;
    }
}
