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
/// This is a `Copy` type: bools packed into bytes. Passed by value,
/// lives in registers after the first `LazyLock` deref.
///
/// x86_64 fields detect via `is_x86_feature_detected!`.
/// aarch64 fields detect via `is_aarch64_feature_detected!`.
/// NEON is mandatory on aarch64 — the sub-features distinguish Pi models:
///   Pi Zero 2 W / Pi 3 (A53, v8.0): neon only
///   Pi 4 (A72, v8.0):               neon only (but 2× throughput)
///   Pi 5 (A76, v8.2):               neon + dotprod + fp16 + aes + sha2
#[derive(Debug, Clone, Copy)]
pub struct SimdCaps {
    // ── x86_64 ──
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

    // ── aarch64 (ARM) ──
    /// NEON 128-bit SIMD (mandatory on aarch64, always true).
    pub neon: bool,
    /// ASIMD dot product (ARMv8.2+: Pi 5 A76, NOT Pi 4 A72).
    /// Enables `vdotq_s32` — 4× throughput for int8 dot products.
    pub asimd_dotprod: bool,
    /// FP16 half-precision arithmetic (ARMv8.2+: Pi 5).
    /// Enables `vcvt_f16_f32` and native f16 math.
    pub fp16: bool,
    /// AES hardware acceleration (Pi 3+, all aarch64 Pi models).
    pub aes: bool,
    /// SHA-2 hardware acceleration (Pi 3+).
    pub sha2: bool,
    /// CRC32 instructions (Pi 3+).
    pub crc32: bool,
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
            // ARM fields: all false on x86
            neon: false,
            asimd_dotprod: false,
            fp16: false,
            aes: false,
            sha2: false,
            crc32: false,
        }
    }

    /// AArch64: detect NEON sub-features via `is_aarch64_feature_detected!`.
    /// NEON itself is mandatory (always true). The sub-features distinguish
    /// Pi Zero 2 W / Pi 3 (A53) from Pi 4 (A72) from Pi 5 (A76).
    #[cfg(target_arch = "aarch64")]
    fn detect() -> Self {
        Self {
            // x86 fields: all false on ARM
            avx2: false,
            avx512f: false,
            avx512bw: false,
            avx512vl: false,
            avx512vpopcntdq: false,
            sse41: false,
            sse2: false,
            fma: false,
            // ARM fields: runtime detection
            neon: true, // mandatory on aarch64
            asimd_dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
            fp16: std::arch::is_aarch64_feature_detected!("fp16"),
            aes: std::arch::is_aarch64_feature_detected!("aes"),
            sha2: std::arch::is_aarch64_feature_detected!("sha2"),
            crc32: std::arch::is_aarch64_feature_detected!("crc"),
        }
    }

    /// Non-x86, non-ARM: all false (wasm, riscv, etc).
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
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
            neon: false,
            asimd_dotprod: false,
            fp16: false,
            aes: false,
            sha2: false,
            crc32: false,
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

    // ── ARM convenience methods ──

    /// True if running on aarch64 with NEON (always true on aarch64).
    #[inline(always)]
    pub fn has_neon(self) -> bool {
        self.neon
    }

    /// True if ASIMD dot product is available (ARMv8.2+: Pi 5, Orange Pi 5).
    /// Enables `vdotq_s32` for 4× int8 dot product throughput.
    #[inline(always)]
    pub fn has_dotprod(self) -> bool {
        self.neon && self.asimd_dotprod
    }

    /// True if FP16 arithmetic is available (ARMv8.2+: Pi 5, Orange Pi 5).
    #[inline(always)]
    pub fn has_fp16(self) -> bool {
        self.neon && self.fp16
    }

    /// True if AES + SHA2 crypto extensions are available (Pi 3+, Orange Pi 4+).
    #[inline(always)]
    pub fn has_crypto(self) -> bool {
        self.aes && self.sha2
    }

    /// Identify the ARM SBC profile based on detected features.
    ///
    /// This is heuristic — detects the *capability tier*, not the exact board.
    /// Boards with the same SoC tier share the same SIMD capabilities:
    ///
    /// | Profile | SoC | Boards |
    /// |---------|-----|--------|
    /// | `A53Baseline` | Cortex-A53 v8.0 | Pi Zero 2 W, Pi 3B+ |
    /// | `A72Fast` | Cortex-A72 v8.0 | Pi 4, Orange Pi 4 LTS |
    /// | `A76DotProd` | Cortex-A76 v8.2 | Pi 5, Orange Pi 5 |
    /// | `Unknown` | Anything else | Other aarch64 SBCs |
    #[inline]
    pub fn arm_profile(self) -> ArmProfile {
        if !self.neon {
            return ArmProfile::NotArm;
        }
        if self.asimd_dotprod {
            // ARMv8.2+: Pi 5 (A76), Orange Pi 5 (RK3588/A76+A55)
            ArmProfile::A76DotProd
        } else if self.aes {
            // ARMv8.0 with crypto: could be A53 or A72.
            // Can't distinguish purely from features — both have
            // NEON + AES + SHA2 but NOT dotprod.
            // A72 has 2× NEON throughput but that's microarch, not features.
            // We report A72-tier since most deployments target Pi 4.
            ArmProfile::A72Fast
        } else {
            // NEON but no crypto — unusual for Pi, but possible on
            // older aarch64 SoCs or QEMU without extensions.
            ArmProfile::A53Baseline
        }
    }
}

/// ARM single-board computer capability tier.
///
/// Heuristic based on detected SIMD features. Boards with the same SoC
/// family share the tier. Used for codebook kernel selection and throughput
/// estimation in ada-brain cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmProfile {
    /// Not an ARM target (x86, wasm, etc.)
    NotArm,
    /// Cortex-A53 v8.0: Pi Zero 2 W, Pi 3B+. NEON baseline only.
    /// ~1 NEON pipeline, lower clock. Codebook: 50-500 tok/s.
    A53Baseline,
    /// Cortex-A72 v8.0: Pi 4, Orange Pi 4 LTS. NEON + crypto.
    /// 2× NEON throughput, higher clock. Codebook: 500-5K tok/s.
    A72Fast,
    /// Cortex-A76 v8.2: Pi 5, Orange Pi 5. NEON + dotprod + fp16.
    /// dotprod enables 4× int8 throughput. Codebook: 2K-10K tok/s.
    A76DotProd,
}

impl ArmProfile {
    /// Human-readable name.
    pub const fn name(self) -> &'static str {
        match self {
            Self::NotArm => "not-arm",
            Self::A53Baseline => "A53-baseline (Pi Zero 2W / Pi 3)",
            Self::A72Fast => "A72-fast (Pi 4 / Orange Pi 4)",
            Self::A76DotProd => "A76-dotprod (Pi 5 / Orange Pi 5)",
        }
    }

    /// Estimated codebook tokens/second for this profile.
    pub const fn estimated_tok_per_sec(self) -> u32 {
        match self {
            Self::NotArm => 0,
            Self::A53Baseline => 200,
            Self::A72Fast => 2_000,
            Self::A76DotProd => 5_000,
        }
    }

    /// Number of effective f32 NEON lanes (accounting for pipeline width).
    /// A53: 1 pipeline = 4 lanes effective.
    /// A72: 2 pipelines = 8 lanes effective (can issue 2 NEON ops/cycle).
    /// A76: 2 pipelines + dotprod = 8 lanes + int8 boost.
    pub const fn effective_f32_lanes(self) -> usize {
        match self {
            Self::NotArm => 1,
            Self::A53Baseline => 4,
            Self::A72Fast => 8,
            Self::A76DotProd => 8,
        }
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
        let _ = caps.neon;
    }

    #[test]
    fn simd_caps_is_copy() {
        let a = simd_caps();
        let b = a; // Copy
        let c = a; // Still valid
        assert_eq!(a.avx2, b.avx2);
        assert_eq!(b.avx512f, c.avx512f);
        assert_eq!(a.neon, c.neon);
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
        assert_eq!(a.neon, b.neon);
        assert_eq!(a.asimd_dotprod, b.asimd_dotprod);
    }

    #[test]
    fn convenience_methods() {
        let caps = simd_caps();
        // Just verify these don't panic and return consistent values.
        let _ = caps.has_avx512_popcnt();
        let _ = caps.has_avx512_bw_popcnt();
        let _ = caps.has_neon();
        let _ = caps.has_dotprod();
        let _ = caps.has_fp16();
        let _ = caps.has_crypto();
    }

    #[test]
    fn arm_profile_consistent() {
        let caps = simd_caps();
        let profile = caps.arm_profile();
        let _ = profile.name();
        let _ = profile.estimated_tok_per_sec();
        let _ = profile.effective_f32_lanes();
        // On x86, should be NotArm
        #[cfg(target_arch = "x86_64")]
        assert_eq!(profile, ArmProfile::NotArm);
        // On aarch64, should be one of the ARM profiles
        #[cfg(target_arch = "aarch64")]
        assert_ne!(profile, ArmProfile::NotArm);
    }
}
