//! CPU feature detection for JIT compilation.
//!
//! Maps runtime-detected CPU features to capabilities relevant for
//! instruction selection. Pure Rust — no Cranelift dependency.

/// Detected CPU capabilities relevant for JIT compilation.
///
/// This is a `Copy` type: small, stack-allocated, passed by value.
/// Follows the "Reasoning" data-flow pattern from data-flow.md.
#[derive(Debug, Clone, Copy)]
pub struct CpuCaps {
    /// AVX2 (256-bit integer SIMD).
    pub has_avx2: bool,
    /// AVX-512 Foundation.
    pub has_avx512f: bool,
    /// AVX-512 Byte/Word.
    pub has_avx512bw: bool,
    /// AVX-512 Vector Length extensions.
    pub has_avx512vl: bool,
    /// AVX-512 VPOPCNTDQ (hardware popcount on 512-bit).
    pub has_avx512vpopcntdq: bool,
    /// FMA (fused multiply-add).
    pub has_fma: bool,
    /// BMI2 (bit manipulation).
    pub has_bmi2: bool,
}

impl CpuCaps {
    /// Detect CPU capabilities at runtime.
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        Self {
            has_avx2: std::arch::is_x86_feature_detected!("avx2"),
            has_avx512f: std::arch::is_x86_feature_detected!("avx512f"),
            has_avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
            has_avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
            has_avx512vpopcntdq: std::arch::is_x86_feature_detected!("avx512vpopcntdq"),
            has_fma: std::arch::is_x86_feature_detected!("fma"),
            has_bmi2: std::arch::is_x86_feature_detected!("bmi2"),
        }
    }

    /// Detect CPU capabilities at runtime (non-x86: all false).
    #[cfg(not(target_arch = "x86_64"))]
    pub fn detect() -> Self {
        Self {
            has_avx2: false,
            has_avx512f: false,
            has_avx512bw: false,
            has_avx512vl: false,
            has_avx512vpopcntdq: false,
            has_fma: false,
            has_bmi2: false,
        }
    }

    /// Returns true if at least AVX2 is available (minimum for SIMD paths).
    pub fn has_simd_minimum(&self) -> bool {
        self.has_avx2
    }

    /// Returns true if the full AVX-512 popcount path is available.
    pub fn has_avx512_popcnt(&self) -> bool {
        self.has_avx512f && self.has_avx512vpopcntdq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_does_not_panic() {
        let caps = CpuCaps::detect();
        // On any platform, detect() should succeed without panic.
        // The booleans are valid regardless of CPU.
        let _ = caps.has_avx2;
        let _ = caps.has_avx512f;
    }

    #[test]
    fn cpu_caps_is_copy() {
        let caps = CpuCaps::detect();
        let caps2 = caps; // Copy
        let caps3 = caps; // Still valid — Copy, not moved
        assert_eq!(caps2.has_avx2, caps3.has_avx2);
    }

    #[test]
    fn convenience_methods() {
        let caps = CpuCaps {
            has_avx2: true,
            has_avx512f: true,
            has_avx512bw: false,
            has_avx512vl: false,
            has_avx512vpopcntdq: true,
            has_fma: false,
            has_bmi2: false,
        };
        assert!(caps.has_simd_minimum());
        assert!(caps.has_avx512_popcnt());

        let no_caps = CpuCaps {
            has_avx2: false,
            has_avx512f: false,
            has_avx512bw: false,
            has_avx512vl: false,
            has_avx512vpopcntdq: false,
            has_fma: false,
            has_bmi2: false,
        };
        assert!(!no_caps.has_simd_minimum());
        assert!(!no_caps.has_avx512_popcnt());
    }
}
