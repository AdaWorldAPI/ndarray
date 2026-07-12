use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(chacha20_force_soft)] {
        pub(crate) mod soft;
    } else if #[cfg(any(
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            not(chacha20_force_avx2),
            not(chacha20_force_sse2),
        ),
        all(target_arch = "wasm32", target_feature = "simd128"),
    ))] {
        // AdaWorldAPI matryoshka: the keystream double-round rides
        // `ndarray::simd::U32x16` — the polyfill lowers it to `__m512i` on an
        // AVX-512 build (server) and to the native `[U32x4; 4]` lane on a
        // wasm32+simd128 build (browser). Same source, two hardware lanes.
        // AUTO-selection only: `chacha20_force_avx2`/`_sse2` still win on x86_64
        // (they fall through to the force subtree below) so A/B benchmarks and a
        // rollback stay possible on v4 builds; the force cfgs are documented as
        // ignored on non-x86, so the wasm arm is unguarded.
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
