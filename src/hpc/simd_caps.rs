//! SIMD capability singleton — detect once, dispatch forever.
//!
//! Replaces per-call `is_x86_feature_detected!` (hidden `AtomicU8` load each time)
//! with a single `LazyLock<SimdCaps>` detected at first access. Every HPC module
//! calls `simd_caps()` which is one pointer deref to a frozen `Copy` struct.
//!
//! ```text
//! is_x86_feature_detected!("avx512f")  →  ~3ns (atomic load + branch)
//! simd_caps().avx512f                  →  ~1ns (LazyLock deref + bool read)
//! ```

use std::sync::LazyLock;

/// Detected SIMD capabilities, frozen at first access.
///
/// This is a `Copy` type: 8 bools packed into 8 bytes. Passed by value,
/// lives in registers after the first `LazyLock` deref.
#[derive(Debug, Clone, Copy)]
pub struct SimdCaps {
    /// AVX2 (256-bit integer/FP SIMD).
    pub avx2: bool,
    /// AVX-512 Foundation (512-bit).
    pub avx512f: bool,
    /// AVX-512 Byte/Word operations.
    pub avx512bw: bool,
    /// AVX-512 Vector Length extensions.
    pub avx512vl: bool,
    /// AVX-512 VPOPCNTDQ (hardware popcount on 512-bit).
    pub avx512vpopcntdq: bool,
    /// SSE 4.1.
    pub sse41: bool,
    /// SSE2 (baseline on x86_64, but explicit for clarity).
    pub sse2: bool,
    /// FMA (fused multiply-add).
    pub fma: bool,
}

/// Global singleton — detected once at first access via `LazyLock`.
static CAPS: LazyLock<SimdCaps> = LazyLock::new(SimdCaps::detect);

/// Get the detected SIMD capabilities. First call detects; all subsequent
/// calls are a single pointer deref with no atomic operations.
#[inline(always)]
pub fn simd_caps() -> SimdCaps {
    *CAPS
}

impl SimdCaps {
    /// Detect CPU capabilities at runtime.
    #[cfg(target_arch = "x86_64")]
    fn detect() -> Self {
        Self {
            avx2: is_x86_feature_detected!("avx2"),
            avx512f: is_x86_feature_detected!("avx512f"),
            avx512bw: is_x86_feature_detected!("avx512bw"),
            avx512vl: is_x86_feature_detected!("avx512vl"),
            avx512vpopcntdq: is_x86_feature_detected!("avx512vpopcntdq"),
            sse41: is_x86_feature_detected!("sse4.1"),
            sse2: is_x86_feature_detected!("sse2"),
            fma: is_x86_feature_detected!("fma"),
        }
    }

    /// Non-x86: all false.
    #[cfg(not(target_arch = "x86_64"))]
    fn detect() -> Self {
        Self {
            avx2: false,
            avx512f: false,
            avx512bw: false,
            avx512vl: false,
            avx512vpopcntdq: false,
            sse41: false,
            sse2: false,
            fma: false,
        }
    }

    /// True if AVX-512 Foundation + VPOPCNTDQ are both available.
    #[inline(always)]
    pub fn has_avx512_popcnt(self) -> bool {
        self.avx512f && self.avx512vpopcntdq
    }

    /// True if AVX-512 BW + VPOPCNTDQ are both available.
    #[inline(always)]
    pub fn has_avx512_bw_popcnt(self) -> bool {
        self.avx512bw && self.avx512vpopcntdq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_does_not_panic() {
        let caps = simd_caps();
        // On any platform, simd_caps() should succeed.
        let _ = caps.avx2;
        let _ = caps.avx512f;
    }

    #[test]
    fn simd_caps_is_copy() {
        let a = simd_caps();
        let b = a; // Copy
        let c = a; // Still valid
        assert_eq!(a.avx2, b.avx2);
        assert_eq!(b.avx512f, c.avx512f);
    }

    #[test]
    fn simd_caps_deterministic() {
        let a = simd_caps();
        let b = simd_caps();
        assert_eq!(a.avx2, b.avx2);
        assert_eq!(a.avx512f, b.avx512f);
        assert_eq!(a.avx512bw, b.avx512bw);
        assert_eq!(a.avx512vpopcntdq, b.avx512vpopcntdq);
        assert_eq!(a.sse41, b.sse41);
    }

    #[test]
    fn convenience_methods() {
        let caps = simd_caps();
        // Just verify these don't panic and return consistent values.
        let _ = caps.has_avx512_popcnt();
        let _ = caps.has_avx512_bw_popcnt();
    }
}
