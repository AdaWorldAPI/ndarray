//! `SimdProfile` — silicon-grained dispatch identity.
//!
//! Where `SimdCaps` reports individual feature bits (AVX-512F, AMX-TILE,
//! VBMI, etc.), `SimdProfile` collapses a *combination* of those bits into
//! a single enum variant naming the silicon generation. One profile = one
//! best set of primitives. Consumers branch on `simd_profile()` once at
//! startup; subsequent dispatch is a `match` over the variant or an index
//! into a `*Dispatch` static table.
//!
//! Authoritative feature mapping lives in
//! `.claude/knowledge/td-simd-cpu-dispatch-matrix.md`. The detection ladder
//! below implements the decision tree at lines 271-305 of that document.
//! Critical invariants from § "Detection invariants":
//!
//! 1. `GraniteRapids` must be checked **before** `SapphireRapids` — GNR
//!    has every SPR bit plus AMX-FP16. Checking SPR first would route GNR
//!    silicon as SPR and leave the AMX-FP16 tile path unused.
//! 2. `Zen4Avx512` vs `SapphireRapids` discriminate on `amx_tile` (SPR
//!    yes, Zen 4 no) — both have F + VBMI + BF16 + FP16.
//! 3. `CooperLake` vs `IceLakeSp` are mutually exclusive bit patterns:
//!    CPL has BF16 without VBMI; ICX has VBMI without BF16.
//! 4. `TigerLakeU` vs `IceLakeSp` share an ISA except for VP2INTERSECT
//!    (TigerLake mobile only). Sole discriminator is `avx512vp2intersect`.

use crate::hpc::simd_caps::{simd_caps, ArmProfile};
use std::sync::LazyLock;

/// Silicon-grained dispatch identity. One variant = one set of best
/// primitives. See `td-simd-cpu-dispatch-matrix.md` for the authoritative
/// feature table per variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimdProfile {
    // ── x86_64 ──
    /// Granite Rapids. Sapphire-Rapids feature set plus AMX-FP16.
    GraniteRapids,
    /// Sapphire Rapids / Emerald Rapids (ISA-identical). AVX-512 superset
    /// with AMX-TILE/INT8/BF16 and AVX-512-FP16. No AMX-FP16.
    SapphireRapids,
    /// Zen 4 / Zen 5 (Genoa / Turin). AVX-512 superset matching SPR's
    /// non-AMX feature set; no AMX of any kind.
    Zen4Avx512,
    /// Cooper Lake. AVX-512F + VNNI + BF16, no VBMI/FP16/AMX. The unique
    /// "BF16 without VBMI" combination.
    CooperLake,
    /// Tiger Lake mobile. Ice-Lake-class AVX-512 + VP2INTERSECT, no
    /// BF16/FP16/AMX. Distinguished from Ice Lake-SP by VP2INTERSECT only.
    TigerLakeU,
    /// Ice Lake-SP server. AVX-512F + VBMI + VNNI + IFMA, no
    /// BF16/FP16/AMX/VP2INTERSECT.
    IceLakeSp,
    /// Cascade Lake. AVX-512F + VNNI, no VBMI/BF16/FP16/AMX.
    CascadeLake,
    /// Skylake-X / SP / W. AVX-512F baseline only (no VNNI).
    SkylakeX,
    /// Arrow Lake / Lunar Lake / Meteor Lake-H. AVX2 + AVX-VNNI-INT8 +
    /// AVX-IFMA + AVX-NE-CONVERT, no AVX-512.
    ArrowLake,
    /// Haswell through Coffee Lake / Zen 1-3. AVX2 + FMA, no AVX-512.
    HaswellAvx2,

    // ── aarch64 ──
    /// Cortex-A76+ (Pi 5, RK3588 big cores, Apple M-series). NEON +
    /// dotprod + fp16 + bf16/i8mm where present.
    A76DotProd,
    /// Cortex-A72 / A53-with-crypto. ARMv8.0 + NEON + crypto, no dotprod.
    /// HWCAP cannot distinguish A72 (Pi 4) from A53-with-crypto (Pi 3)
    /// silicon — both share the same dispatch tables at the ISA level.
    A72Fast,
    /// Cortex-A53 without crypto. Rare in the wild; QEMU / minimal aarch64.
    A53Baseline,

    // ── Fallback ──
    /// wasm32, riscv, x86 baseline, or anything else without a recognised
    /// SIMD ISA.
    Scalar,
}

impl SimdProfile {
    /// Resolve the current silicon to one of the enum variants.
    ///
    /// Reads `simd_caps()` once and walks the decision tree from
    /// `td-simd-cpu-dispatch-matrix.md` § "SimdProfile::detect() mapping".
    /// The order below is load-bearing: GNR before SPR, the BF16/VBMI
    /// mutex for CPL/ICX, VP2INTERSECT for TigerLakeU vs IceLakeSp.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            let caps = simd_caps();
            // Risk #3 from the integration plan ("Detection robustness
            // across hypervisors"): CPUID may report AMX-TILE while the
            // OS has not enabled the tile XSAVE state. In that case AMX
            // instructions SIGILL despite the CPUID bit being set.
            // `simd_amx::amx_available()` runs the full 4-step gate
            // (CPUID + OSXSAVE + XCR0 bits 17/18 + arch_prctl
            // XCOMP_PERM). Demote to the no-AMX dispatch branch when
            // the OS check fails — typically resolves as Zen4Avx512 on
            // SPR-class CPUID with locked-down hypervisor XSAVE state.
            let amx_usable = caps.amx_tile && crate::simd_amx::amx_available();
            // GraniteRapids: AMX-FP16 (CPUID 7,1 EAX bit 21). Must be
            // checked first because GNR is a strict superset of SPR.
            if amx_usable && caps.amx_fp16 {
                return SimdProfile::GraniteRapids;
            }
            // SapphireRapids / EmeraldRapids: AMX-TILE + AMX-BF16 +
            // AVX-512-FP16. EmeraldRapids has identical ISA — same variant.
            if amx_usable && caps.amx_bf16 && caps.avx512fp16 {
                return SimdProfile::SapphireRapids;
            }
            // Zen4 / Zen5: AVX-512 + VBMI + BF16 + FP16, but no usable
            // AMX. The `!amx_usable` guard also catches OS-demoted SPR
            // silicon so it resolves here instead of as SapphireRapids.
            if caps.avx512f
                && caps.avx512vbmi
                && caps.avx512bf16
                && caps.avx512fp16
                && !amx_usable
            {
                return SimdProfile::Zen4Avx512;
            }
            // CooperLake: AVX-512 + VNNI + BF16, but no VBMI. Mutually
            // exclusive bit pattern with IceLakeSp (which has VBMI but
            // no BF16). Order vs ICX is irrelevant.
            if caps.avx512f && caps.avx512vnni && caps.avx512bf16 && !caps.avx512vbmi {
                return SimdProfile::CooperLake;
            }
            // TigerLakeU vs IceLakeSp: same feature set EXCEPT
            // VP2INTERSECT (TigerLake mobile only). Discriminate here.
            if caps.avx512f && caps.avx512vbmi && caps.avx512vnni && !caps.avx512bf16 {
                return if caps.avx512vp2intersect {
                    SimdProfile::TigerLakeU
                } else {
                    SimdProfile::IceLakeSp
                };
            }
            // CascadeLake: AVX-512 + VNNI, no VBMI/BF16.
            if caps.avx512f && caps.avx512vnni {
                return SimdProfile::CascadeLake;
            }
            // SkylakeX: AVX-512F only, no VNNI.
            if caps.avx512f {
                return SimdProfile::SkylakeX;
            }
            // ArrowLake / Lunar Lake / Meteor Lake-H: no AVX-512 but
            // AVX-VNNI-INT8 present.
            if caps.avxvnniint8 {
                return SimdProfile::ArrowLake;
            }
            // Haswell..Coffee Lake / Zen 1-3: AVX2 + FMA.
            if caps.avx2 && caps.fma {
                return SimdProfile::HaswellAvx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // Reuse the in-tree heuristic from `simd_caps::arm_profile()`.
            // It already encodes the A72-vs-A53-crypto deployment-pragmatic
            // decision and is the canonical ARM dispatch helper.
            return match simd_caps().arm_profile() {
                ArmProfile::A76DotProd => SimdProfile::A76DotProd,
                ArmProfile::A72Fast => SimdProfile::A72Fast,
                ArmProfile::A53Baseline => SimdProfile::A53Baseline,
                ArmProfile::NotArm => SimdProfile::Scalar,
            };
        }
        SimdProfile::Scalar
    }

    /// Human-readable name. Stable across versions; safe to use in logs,
    /// telemetry, and bench output.
    pub const fn name(self) -> &'static str {
        match self {
            Self::GraniteRapids => "GraniteRapids",
            Self::SapphireRapids => "SapphireRapids",
            Self::Zen4Avx512 => "Zen4Avx512",
            Self::CooperLake => "CooperLake",
            Self::TigerLakeU => "TigerLakeU",
            Self::IceLakeSp => "IceLakeSp",
            Self::CascadeLake => "CascadeLake",
            Self::SkylakeX => "SkylakeX",
            Self::ArrowLake => "ArrowLake",
            Self::HaswellAvx2 => "HaswellAvx2",
            Self::A76DotProd => "A76DotProd",
            Self::A72Fast => "A72Fast",
            Self::A53Baseline => "A53Baseline",
            Self::Scalar => "Scalar",
        }
    }

    /// True iff this profile sits on the x86_64 silicon family. Useful
    /// for branching in dispatch tables that share a single AMX/AVX-512
    /// kernel across multiple variants.
    pub const fn is_x86(self) -> bool {
        matches!(
            self,
            Self::GraniteRapids
                | Self::SapphireRapids
                | Self::Zen4Avx512
                | Self::CooperLake
                | Self::TigerLakeU
                | Self::IceLakeSp
                | Self::CascadeLake
                | Self::SkylakeX
                | Self::ArrowLake
                | Self::HaswellAvx2
        )
    }

    /// True iff this profile sits on the aarch64 silicon family.
    pub const fn is_aarch64(self) -> bool {
        matches!(
            self,
            Self::A76DotProd | Self::A72Fast | Self::A53Baseline
        )
    }

    /// True iff this profile has AVX-512 Foundation. False for ArrowLake,
    /// HaswellAvx2, aarch64, and Scalar. Use to route AVX-512 dispatch
    /// without re-querying `SimdCaps`.
    pub const fn has_avx512(self) -> bool {
        matches!(
            self,
            Self::GraniteRapids
                | Self::SapphireRapids
                | Self::Zen4Avx512
                | Self::CooperLake
                | Self::TigerLakeU
                | Self::IceLakeSp
                | Self::CascadeLake
                | Self::SkylakeX
        )
    }

    /// True iff this profile has any AMX tile capability (AMX-TILE +
    /// AMX-INT8 + AMX-BF16 at minimum). Only SPR-class and GNR.
    pub const fn has_amx(self) -> bool {
        matches!(self, Self::GraniteRapids | Self::SapphireRapids)
    }
}

// ────────────────────────────────────────────────────────────────────
// Phase 3 T3.2 — compile-time pinning via `cpu-*` cargo features.
//
// When any `cpu-<codename>` feature is set, `PINNED_PROFILE` is `Some(_)`
// and `simd_profile()` folds to the const at compile time — no LazyLock
// initialisation, no branch, no atomics. Without any feature set,
// behaviour matches the original runtime LazyLock detection.
//
// Mutual exclusion: at most ONE cpu-* feature may be enabled. The
// `_PIN_COUNT` const assert below fires at compile time if more than
// one is active, mirroring integration-plan risk #1 ("if a user sets
// cpu-spr AND has runtime detection on Zen 4, the binary SIGILLs").
// ────────────────────────────────────────────────────────────────────

const _PIN_COUNT: u32 = 0
    + cfg!(feature = "cpu-gnr") as u32
    + cfg!(feature = "cpu-spr") as u32
    + cfg!(feature = "cpu-zen4") as u32
    + cfg!(feature = "cpu-cpl") as u32
    + cfg!(feature = "cpu-tigerlake") as u32
    + cfg!(feature = "cpu-icx") as u32
    + cfg!(feature = "cpu-clx") as u32
    + cfg!(feature = "cpu-skx") as u32
    + cfg!(feature = "cpu-arrowlake") as u32
    + cfg!(feature = "cpu-haswell") as u32
    + cfg!(feature = "cpu-a76") as u32
    + cfg!(feature = "cpu-a72") as u32
    + cfg!(feature = "cpu-a53") as u32;

const _: () = assert!(
    _PIN_COUNT <= 1,
    "cpu-* cargo features are mutually exclusive: enable at most one (cpu-gnr, cpu-spr, cpu-zen4, cpu-cpl, cpu-tigerlake, cpu-icx, cpu-clx, cpu-skx, cpu-arrowlake, cpu-haswell, cpu-a76, cpu-a72, cpu-a53)"
);

// ────────────────────────────────────────────────────────────────────
// target_arch guards (codex P2 closure).
//
// Each `cpu-<codename>` pin is only valid on its native silicon
// family. Without these guards `--features cpu-a76` on an x86_64
// build would silently route `simd_profile()` to `A76DotProd`,
// breaking the `is_x86()` / `is_aarch64()` partitioning and routing
// callers into the wrong dispatch family. Fail fast at compile time
// instead.
// ────────────────────────────────────────────────────────────────────

#[cfg(all(
    not(target_arch = "x86_64"),
    any(
        feature = "cpu-gnr",
        feature = "cpu-spr",
        feature = "cpu-zen4",
        feature = "cpu-cpl",
        feature = "cpu-tigerlake",
        feature = "cpu-icx",
        feature = "cpu-clx",
        feature = "cpu-skx",
        feature = "cpu-arrowlake",
        feature = "cpu-haswell",
    )
))]
compile_error!(
    "x86 cpu-* pinning features (cpu-gnr, cpu-spr, cpu-zen4, cpu-cpl, cpu-tigerlake, cpu-icx, cpu-clx, cpu-skx, cpu-arrowlake, cpu-haswell) require target_arch = \"x86_64\""
);

#[cfg(all(
    not(target_arch = "aarch64"),
    any(feature = "cpu-a76", feature = "cpu-a72", feature = "cpu-a53",)
))]
compile_error!(
    "ARM cpu-* pinning features (cpu-a76, cpu-a72, cpu-a53) require target_arch = \"aarch64\""
);

/// The compile-time pinned profile, or `None` when runtime detection is in
/// effect. `Some(_)` exactly when one of the `cpu-*` cargo features is
/// enabled; mutually exclusive features are enforced by the `_PIN_COUNT`
/// const assert above.
///
/// Consumers wanting branch-free dispatch on pinned builds can match on
/// this const directly — the optimiser folds `Some(SimdProfile::X)` into
/// the call site and the `None`-arm runtime path is eliminated. Returned
/// from a `const fn` so call sites in const contexts (e.g. `const` array
/// initialisers for dispatch tables) work as well.
pub const fn pinned_profile() -> Option<SimdProfile> {
    #[cfg(feature = "cpu-gnr")]
    {
        return Some(SimdProfile::GraniteRapids);
    }
    #[cfg(feature = "cpu-spr")]
    {
        return Some(SimdProfile::SapphireRapids);
    }
    #[cfg(feature = "cpu-zen4")]
    {
        return Some(SimdProfile::Zen4Avx512);
    }
    #[cfg(feature = "cpu-cpl")]
    {
        return Some(SimdProfile::CooperLake);
    }
    #[cfg(feature = "cpu-tigerlake")]
    {
        return Some(SimdProfile::TigerLakeU);
    }
    #[cfg(feature = "cpu-icx")]
    {
        return Some(SimdProfile::IceLakeSp);
    }
    #[cfg(feature = "cpu-clx")]
    {
        return Some(SimdProfile::CascadeLake);
    }
    #[cfg(feature = "cpu-skx")]
    {
        return Some(SimdProfile::SkylakeX);
    }
    #[cfg(feature = "cpu-arrowlake")]
    {
        return Some(SimdProfile::ArrowLake);
    }
    #[cfg(feature = "cpu-haswell")]
    {
        return Some(SimdProfile::HaswellAvx2);
    }
    #[cfg(feature = "cpu-a76")]
    {
        return Some(SimdProfile::A76DotProd);
    }
    #[cfg(feature = "cpu-a72")]
    {
        return Some(SimdProfile::A72Fast);
    }
    #[cfg(feature = "cpu-a53")]
    {
        return Some(SimdProfile::A53Baseline);
    }
    #[allow(unreachable_code)]
    None
}

/// `true` when a `cpu-*` cargo feature has pinned the profile at compile
/// time, `false` when runtime detection is in use. Equivalent to
/// `pinned_profile().is_some()` but spelled out for grep-ability.
pub const fn is_pinned() -> bool {
    pinned_profile().is_some()
}

// The LazyLock only exists when no cpu-* feature is set. With pinning,
// linking the LazyLock would defeat the purpose — we want every code
// path that touches `simd_profile()` to fold to a const.
#[cfg(not(any(
    feature = "cpu-gnr",
    feature = "cpu-spr",
    feature = "cpu-zen4",
    feature = "cpu-cpl",
    feature = "cpu-tigerlake",
    feature = "cpu-icx",
    feature = "cpu-clx",
    feature = "cpu-skx",
    feature = "cpu-arrowlake",
    feature = "cpu-haswell",
    feature = "cpu-a76",
    feature = "cpu-a72",
    feature = "cpu-a53",
)))]
static PROFILE: LazyLock<SimdProfile> = LazyLock::new(SimdProfile::detect);

/// Get the resolved silicon profile.
///
/// **Default (no `cpu-*` feature):** detected once at first access via
/// `LazyLock`; subsequent calls are a single pointer deref to a `Copy`
/// enum — no atomics, no branching.
///
/// **Pinned (one `cpu-*` feature set):** returns the pinned const
/// directly. The compiler folds this call into the matching variant at
/// every call site; the LazyLock is not linked into the binary.
///
/// Pair with `*Dispatch` static tables to make per-call dispatch a single
/// indirect call after monomorphisation (runtime path) or to fold to a
/// direct call (pinned path).
#[cfg(not(any(
    feature = "cpu-gnr",
    feature = "cpu-spr",
    feature = "cpu-zen4",
    feature = "cpu-cpl",
    feature = "cpu-tigerlake",
    feature = "cpu-icx",
    feature = "cpu-clx",
    feature = "cpu-skx",
    feature = "cpu-arrowlake",
    feature = "cpu-haswell",
    feature = "cpu-a76",
    feature = "cpu-a72",
    feature = "cpu-a53",
)))]
#[inline(always)]
pub fn simd_profile() -> SimdProfile {
    *PROFILE
}

/// Get the resolved silicon profile (pinned variant).
///
/// A `cpu-*` cargo feature is active: this returns the pinned constant
/// directly, foldable at every call site. The runtime LazyLock is not
/// linked into the binary. See the documentation on the runtime variant
/// for the call-site contract.
#[cfg(any(
    feature = "cpu-gnr",
    feature = "cpu-spr",
    feature = "cpu-zen4",
    feature = "cpu-cpl",
    feature = "cpu-tigerlake",
    feature = "cpu-icx",
    feature = "cpu-clx",
    feature = "cpu-skx",
    feature = "cpu-arrowlake",
    feature = "cpu-haswell",
    feature = "cpu-a76",
    feature = "cpu-a72",
    feature = "cpu-a53",
))]
#[inline(always)]
pub const fn simd_profile() -> SimdProfile {
    // SAFETY of the unwrap: the cfg gate above guarantees at least one
    // cpu-* feature is set, and `pinned_profile()` returns Some(_) under
    // any of those gates. Const-evaluable since the inner cfg cascade is
    // resolved at compile time.
    match pinned_profile() {
        Some(p) => p,
        None => SimdProfile::Scalar, // unreachable; const fn can't panic cleanly
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_a_valid_profile() {
        let p = simd_profile();
        // Smoke: the LazyLock must not panic and must return *some*
        // variant. `name()` exhausts the match, which would fail to
        // compile if a new variant were added without a name.
        let _ = p.name();
    }

    #[test]
    fn determinism() {
        // Two consecutive calls must agree — the LazyLock guarantees
        // the closure runs exactly once.
        assert_eq!(simd_profile(), simd_profile());
    }

    #[test]
    fn arch_partitioning_is_consistent() {
        let p = simd_profile();
        if p.is_x86() {
            assert!(!p.is_aarch64(), "{:?} flagged both x86 and aarch64", p);
        }
        if p.is_aarch64() {
            assert!(!p.is_x86(), "{:?} flagged both x86 and aarch64", p);
        }
        // On unsupported architectures the only legal answer is Scalar.
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert_eq!(p, SimdProfile::Scalar);
    }

    #[test]
    fn x86_target_lands_inside_x86_family() {
        // Pinning overrides hardware detection — the test only describes
        // the default (runtime-detection) path.
        if is_pinned() {
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            let p = simd_profile();
            // Scalar is a valid x86_64 answer (no AVX2/FMA available),
            // but any non-Scalar result must be inside the x86 family.
            if p != SimdProfile::Scalar {
                assert!(
                    p.is_x86(),
                    "x86_64 silicon resolved as non-x86 profile {:?}",
                    p
                );
            }
        }
    }

    #[test]
    fn aarch64_target_lands_inside_aarch64_family() {
        if is_pinned() {
            return;
        }
        #[cfg(target_arch = "aarch64")]
        {
            let p = simd_profile();
            assert!(p.is_aarch64(), "aarch64 silicon resolved as {:?}", p);
        }
    }

    #[test]
    fn pinning_default_is_off() {
        // The default build (no cpu-* feature) must NOT be pinned, so
        // downstream consumers don't get surprised by compile-time
        // dispatch they didn't opt into.
        #[cfg(not(any(
            feature = "cpu-gnr",
            feature = "cpu-spr",
            feature = "cpu-zen4",
            feature = "cpu-cpl",
            feature = "cpu-tigerlake",
            feature = "cpu-icx",
            feature = "cpu-clx",
            feature = "cpu-skx",
            feature = "cpu-arrowlake",
            feature = "cpu-haswell",
            feature = "cpu-a76",
            feature = "cpu-a72",
            feature = "cpu-a53",
        )))]
        {
            assert!(!is_pinned());
            assert_eq!(pinned_profile(), None);
        }
    }

    #[test]
    fn pinning_consistency() {
        // When pinning is in effect, simd_profile() must equal the
        // pinned const — hardware detection is bypassed entirely.
        if let Some(pinned) = pinned_profile() {
            assert!(is_pinned());
            assert_eq!(simd_profile(), pinned);
        } else {
            assert!(!is_pinned());
        }
    }

    #[test]
    fn has_avx512_is_subset_of_is_x86() {
        for &p in &[
            SimdProfile::GraniteRapids,
            SimdProfile::SapphireRapids,
            SimdProfile::Zen4Avx512,
            SimdProfile::CooperLake,
            SimdProfile::TigerLakeU,
            SimdProfile::IceLakeSp,
            SimdProfile::CascadeLake,
            SimdProfile::SkylakeX,
            SimdProfile::ArrowLake,
            SimdProfile::HaswellAvx2,
            SimdProfile::A76DotProd,
            SimdProfile::A72Fast,
            SimdProfile::A53Baseline,
            SimdProfile::Scalar,
        ] {
            if p.has_avx512() {
                assert!(p.is_x86(), "{:?} reports AVX-512 but is not x86", p);
            }
            if p.has_amx() {
                assert!(p.has_avx512(), "{:?} reports AMX but not AVX-512", p);
                assert!(p.is_x86(), "{:?} reports AMX but is not x86", p);
            }
        }
    }

    #[test]
    fn names_are_stable_and_unique() {
        let all = [
            SimdProfile::GraniteRapids,
            SimdProfile::SapphireRapids,
            SimdProfile::Zen4Avx512,
            SimdProfile::CooperLake,
            SimdProfile::TigerLakeU,
            SimdProfile::IceLakeSp,
            SimdProfile::CascadeLake,
            SimdProfile::SkylakeX,
            SimdProfile::ArrowLake,
            SimdProfile::HaswellAvx2,
            SimdProfile::A76DotProd,
            SimdProfile::A72Fast,
            SimdProfile::A53Baseline,
            SimdProfile::Scalar,
        ];
        let names: Vec<&'static str> = all.iter().map(|p| p.name()).collect();
        for i in 0..names.len() {
            for j in i + 1..names.len() {
                assert_ne!(
                    names[i], names[j],
                    "profile names must be unique: {:?} == {:?}",
                    all[i], all[j]
                );
            }
        }
    }
}
