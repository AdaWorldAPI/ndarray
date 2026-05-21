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
///
/// `#[non_exhaustive]` per codex P2 on PR #143: future capability fields
/// can be added without source-breaking downstream crates that construct
/// `SimdCaps` directly via struct literal (e.g. mocks, tests, custom
/// capability values). Downstream code must use `simd_caps()` or the
/// public constructor instead of struct-literal init.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
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
    /// AVX-512 VNNI (VPDPBUSD — u8×i8→i32 dot product of 4-element groups).
    /// Present on Ice Lake, Sapphire Rapids, Zen 4 (with AVX-512), Tiger Lake.
    pub avx512vnni: bool,
    /// AVX-512 VBMI (`_mm512_permutexvar_epi8` — full-width byte permute).
    /// Present on Ice Lake, Tiger Lake, Sapphire Rapids, Zen 4. ABSENT on
    /// Skylake-X / Cascade Lake / Ice Lake-SP — calling VBMI intrinsics on
    /// those CPUs SIGILLs even though `avx512f` is true.
    pub avx512vbmi: bool,
    /// AMX-TILE: tile register file present (CPUID.07H.0H:EDX bit 24).
    /// Sapphire Rapids, Granite Rapids, Meteor Lake, Arrow Lake.
    pub amx_tile: bool,
    /// AMX-INT8: `TDPBUSD` u8×i8→i32 tile dot product (CPUID.07H.0H:EDX bit 25).
    pub amx_int8: bool,
    /// AMX-BF16: `TDPBF16PS` BF16×BF16→f32 tile dot product (CPUID.07H.0H:EDX bit 22).
    pub amx_bf16: bool,
    /// AVX-512 BF16: `VCVTNE2PS2BF16` / `VDPBF16PS` 512-bit BF16 math
    /// (`is_x86_feature_detected!("avx512bf16")`).
    /// Present on Cooper Lake, Sapphire Rapids, Zen 4.
    pub avx512bf16: bool,
    /// AVX-VNNI-INT8: 256-bit `VPDPBSSD`/`VPDPBUUD` (non-AVX-512) VNNI
    /// (`is_x86_feature_detected!("avxvnniint8")`).
    /// Present on Arrow Lake, Lunar Lake, NUC 14 (Meteor Lake-H).
    pub avxvnniint8: bool,
    /// AVX-512 FP16 arithmetic (CPUID.07H.0H:EDX bit 23). Native
    /// `__m512h` operations (`_mm512_*_ph`). Present on Sapphire Rapids,
    /// Granite Rapids, Zen 4+. Bit is exposed for downstream substrate
    /// kernels and dispatch ladders; no consumer-facing dispatch axis
    /// is built on top of it.
    pub avx512fp16: bool,
    /// AVX-512 VP2INTERSECT (CPUID.07H.0H:EDX bit 8). Present only on
    /// Tiger Lake mobile silicon; absent from Ice Lake-SP and every
    /// later server part. Useful for future intersection-heavy
    /// primitives (set ops on bitmaps); exposed for completeness.
    pub avx512vp2intersect: bool,
    /// AMX-FP16 (CPUID.07H.1H:EAX bit 21). `TDPFP16PS` FP16 tile dot
    /// product, present on Granite Rapids only. Lives at CPUID leaf
    /// 7,1 (subleaf 1), not leaf 7,0 — separate `__cpuid_count(7, 1)`
    /// call required. The leaf 7,1 read is gated on leaf 7,0's EAX
    /// max-subleaf field being ≥ 1; on older silicon that field is 0
    /// and we never query leaf 7,1.
    pub amx_fp16: bool,

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
    /// Miri-only: CPUID inline asm is unsupported by Miri (it can't simulate
    /// CPU feature detection). Return an all-scalar capability set so any
    /// test reaching this LazyLock under Miri exercises the scalar fallback
    /// paths instead of aborting on the `__cpuid_count` call. Scoped to
    /// `cfg(miri)` — production builds and stable CI use the real detection
    /// below.
    #[cfg(miri)]
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
            avx512vnni: false,
            avx512vbmi: false,
            amx_tile: false,
            amx_int8: false,
            amx_bf16: false,
            avx512bf16: false,
            avxvnniint8: false,
            avx512fp16: false,
            avx512vp2intersect: false,
            amx_fp16: false,
            neon: false,
            asimd_dotprod: false,
            fp16: false,
            aes: false,
            sha2: false,
            crc32: false,
        }
    }

    /// Detect CPU capabilities at runtime.
    #[cfg(all(target_arch = "x86_64", not(miri)))]
    fn detect() -> Self {
        // `__cpuid_count` is safe on x86_64 (Rust 1.87+): CPUID is always
        // available on x86_64 (guaranteed by the ABI) and has no side effects
        // beyond reading CPU registers.
        let cpuid7 = core::arch::x86_64::__cpuid_count(7, 0);
        let amx_tile = (cpuid7.edx >> 24) & 1 == 1;
        let amx_int8 = (cpuid7.edx >> 25) & 1 == 1;
        let amx_bf16 = (cpuid7.edx >> 22) & 1 == 1;
        let avx512fp16 = (cpuid7.edx >> 23) & 1 == 1;
        let avx512vp2intersect = (cpuid7.edx >> 8) & 1 == 1;

        // Leaf 7,1 EAX bit 21 = AMX-FP16. Leaf 7,1 only exists when
        // leaf 7,0 EAX (max-subleaf) is at least 1; on older silicon
        // this returns 0 and the answer is correctly false.
        let amx_fp16 = if cpuid7.eax >= 1 {
            let cpuid7_1 = core::arch::x86_64::__cpuid_count(7, 1);
            (cpuid7_1.eax >> 21) & 1 == 1
        } else {
            false
        };

        Self {
            avx2: is_x86_feature_detected!("avx2"),
            avx512f: is_x86_feature_detected!("avx512f"),
            avx512bw: is_x86_feature_detected!("avx512bw"),
            avx512vl: is_x86_feature_detected!("avx512vl"),
            avx512vpopcntdq: is_x86_feature_detected!("avx512vpopcntdq"),
            sse41: is_x86_feature_detected!("sse4.1"),
            sse2: is_x86_feature_detected!("sse2"),
            fma: is_x86_feature_detected!("fma"),
            avx512vnni: is_x86_feature_detected!("avx512vnni"),
            avx512vbmi: is_x86_feature_detected!("avx512vbmi"),
            amx_tile,
            amx_int8,
            amx_bf16,
            avx512bf16: is_x86_feature_detected!("avx512bf16"),
            avxvnniint8: is_x86_feature_detected!("avxvnniint8"),
            avx512fp16,
            avx512vp2intersect,
            amx_fp16,
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
    #[cfg(all(target_arch = "aarch64", not(miri)))]
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
            avx512vnni: false,
            avx512vbmi: false,
            amx_tile: false,
            amx_int8: false,
            amx_bf16: false,
            avx512bf16: false,
            avxvnniint8: false,
            avx512fp16: false,
            avx512vp2intersect: false,
            amx_fp16: false,
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
            avx512vnni: false,
            avx512vbmi: false,
            amx_tile: false,
            amx_int8: false,
            amx_bf16: false,
            avx512bf16: false,
            avxvnniint8: false,
            avx512fp16: false,
            avx512vp2intersect: false,
            amx_fp16: false,
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

    /// True if AVX-512 VNNI is available (VPDPBUSD on zmm registers).
    /// Present on Ice Lake, Tiger Lake, Sapphire Rapids, Zen 4.
    #[inline(always)]
    pub fn has_avx512_vnni(self) -> bool {
        self.avx512f && self.avx512vnni
    }

    /// True if AMX is available at the CPUID level (`amx_tile && amx_int8`).
    ///
    /// Note: CPUID presence does **not** guarantee OS enablement. The full
    /// OS-level check (XCR0 bits 17+18, prctl ARCH_REQ_XCOMP_PERM) lives in
    /// `simd_amx::amx_available()`. This method is a lightweight CPUID-only
    /// probe suitable for capability reporting and coarse dispatch decisions.
    #[inline(always)]
    pub fn has_amx(self) -> bool {
        self.amx_tile && self.amx_int8
    }

    /// True if AVX-512 BF16 is available (`VCVTNE2PS2BF16` / `VDPBF16PS`).
    /// Present on Cooper Lake, Sapphire Rapids, Zen 4.
    #[inline(always)]
    pub fn has_avx512_bf16(self) -> bool {
        self.avx512bf16
    }

    /// True if AVX-VNNI-INT8 (256-bit `VPDPBSSD`/`VPDPBUUD`) is available.
    /// Present on Arrow Lake, Lunar Lake, NUC 14 (Meteor Lake-H).
    /// This is the non-AVX-512 VNNI path — does NOT require `avx512f`.
    #[inline(always)]
    pub fn has_avxvnniint8(self) -> bool {
        self.avxvnniint8
    }

    /// True if AVX-512 FP16 (`__m512h`) is available. Distinguishes
    /// SapphireRapids-class silicon (and Zen 4+) from the CascadeLake /
    /// IceLakeSp / SkylakeX baseline that lacks native `__m512h` math.
    #[inline(always)]
    pub fn has_avx512_fp16(self) -> bool {
        self.avx512fp16
    }

    /// True if AMX-FP16 (`TDPFP16PS`) is available. Only Granite Rapids
    /// advertises this bit. Requires both the CPUID 7,1 bit AND
    /// AMX-TILE (defense-in-depth: a CPU advertising AMX-FP16 without
    /// AMX-TILE is contradictory but the check stays cheap).
    #[inline(always)]
    pub fn has_amx_fp16(self) -> bool {
        self.amx_fp16 && self.amx_tile
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
        // New AMX / BF16 / VNNI fields must also be accessible without panic.
        let _ = caps.amx_tile;
        let _ = caps.amx_int8;
        let _ = caps.amx_bf16;
        let _ = caps.avx512bf16;
        let _ = caps.avxvnniint8;
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
    fn new_amx_bf16_vnni_convenience_methods_do_not_panic() {
        let caps = simd_caps();
        let amx = caps.has_amx();
        let bf16 = caps.has_avx512_bf16();
        let vnni = caps.has_avxvnniint8();
        // Semantic invariants: has_amx() requires both tile and int8.
        assert_eq!(amx, caps.amx_tile && caps.amx_int8);
        // has_avx512_bf16() mirrors the raw field.
        assert_eq!(bf16, caps.avx512bf16);
        // has_avxvnniint8() mirrors the raw field.
        assert_eq!(vnni, caps.avxvnniint8);
    }

    #[test]
    fn amx_fields_false_on_non_x86() {
        // On non-x86_64, all AMX and BF16 fields must be false because
        // the detect() fallback / aarch64 branch sets them to false.
        #[cfg(not(target_arch = "x86_64"))]
        {
            let caps = simd_caps();
            assert!(!caps.amx_tile);
            assert!(!caps.amx_int8);
            assert!(!caps.amx_bf16);
            assert!(!caps.avx512bf16);
            assert!(!caps.avxvnniint8);
            assert!(!caps.has_amx());
            assert!(!caps.has_avx512_bf16());
            assert!(!caps.has_avxvnniint8());
        }
        // On x86_64 we can only check that the call doesn't panic; the
        // actual values depend on the hardware running the test.
        #[cfg(target_arch = "x86_64")]
        {
            let caps = simd_caps();
            let _ = caps.has_amx();
            let _ = caps.has_avx512_bf16();
            let _ = caps.has_avxvnniint8();
        }
    }

    #[test]
    fn simd_caps_deterministic_new_fields() {
        let a = simd_caps();
        let b = simd_caps();
        assert_eq!(a.amx_tile, b.amx_tile);
        assert_eq!(a.amx_int8, b.amx_int8);
        assert_eq!(a.amx_bf16, b.amx_bf16);
        assert_eq!(a.avx512bf16, b.avx512bf16);
        assert_eq!(a.avxvnniint8, b.avxvnniint8);
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

    /// New CPUID 7,0 EDX bits and the CPUID 7,1 leaf read must surface
    /// without crashing on every host. Field values are host-dependent;
    /// we just exercise the readers and the convenience methods.
    #[test]
    fn cpuid_extended_bits_smoke() {
        let caps = simd_caps();
        let _ = caps.avx512fp16;
        let _ = caps.avx512vp2intersect;
        let _ = caps.amx_fp16;
        let _ = caps.has_avx512_fp16();
        let _ = caps.has_amx_fp16();
    }

    /// `has_amx_fp16()` defense-in-depth: even if `amx_fp16` were
    /// spuriously true without `amx_tile`, the convenience method must
    /// require both. Matches the pattern used by `has_amx_bf16` in
    /// `simd_amx::amx_available()`.
    #[test]
    fn has_amx_fp16_requires_amx_tile() {
        let synthetic = SimdCaps {
            avx2: false,
            avx512f: false,
            avx512bw: false,
            avx512vl: false,
            avx512vpopcntdq: false,
            sse41: false,
            sse2: false,
            fma: false,
            avx512vnni: false,
            avx512vbmi: false,
            amx_tile: false,
            amx_int8: false,
            amx_bf16: false,
            avx512bf16: false,
            avxvnniint8: false,
            avx512fp16: false,
            avx512vp2intersect: false,
            amx_fp16: true,
            neon: false,
            asimd_dotprod: false,
            fp16: false,
            aes: false,
            sha2: false,
            crc32: false,
        };
        assert!(
            !synthetic.has_amx_fp16(),
            "amx_fp16 without amx_tile must report false"
        );
    }

    /// On non-x86 builds the x86 capability bits MUST all read false —
    /// the platform-specific zero-defaults must not regress when new
    /// fields are added to `SimdCaps`.
    #[cfg(not(target_arch = "x86_64"))]
    #[test]
    fn x86_extended_bits_are_false_on_non_x86() {
        let caps = simd_caps();
        assert!(!caps.avx512fp16);
        assert!(!caps.avx512vp2intersect);
        assert!(!caps.amx_fp16);
        assert!(!caps.has_avx512_fp16());
        assert!(!caps.has_amx_fp16());
    }
}
