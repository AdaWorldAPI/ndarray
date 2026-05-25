//! Per-CPU operations DTO — the *third* dispatch pattern.
//!
//! **Pattern 1** (`crate::simd::*`): compile-time `#[cfg(target_feature)]`
//! cascade. Direct monomorphized call. No runtime branch.
//!
//! **Pattern 2** (`crate::simd_runtime::vnni_dot::vnni_dot_u8_i8`,
//! `crate::simd_runtime::add_mul::*`): per-op `LazyLock<fn ptr>`
//! trampoline. One atomic-load + CPUID per op the first time it's
//! called.
//!
//! **Pattern 3 (this module)**: per-CPU [`CpuOps`] DTO selected once
//! at first access. Every op is a field/method on the struct. Consumers
//! that touch N different SIMD ops pay ONE LazyLock load total, not
//! N. The OpenBLAS / MKL dispatch model — also what GCC's own libgcc
//! uses for its multi-versioned function tables.
//!
//! # Why three patterns?
//!
//! - **Pattern 1** wins on bench / fixed-target builds — direct call
//!   sites, no indirection, full inlining.
//! - **Pattern 2** wins for sparse-op consumers — pay LazyLock cost
//!   only for ops you actually call.
//! - **Pattern 3** wins for dense-op consumers (linear-algebra
//!   pipelines that touch every BLAS-1 + BLAS-2 + GEMM kernel) — the
//!   single LazyLock load amortizes across all calls; cache-resident
//!   `&'static CpuOps` keeps every op's fn-ptr in L1.
//!
//! All three coexist. Consumers pick by import path:
//!
//! ```ignore
//! // Pattern 1: same call works on every target_feature config
//! crate::simd_ops::add_mul_f32(acc, a, b);
//!
//! // Pattern 2: one runtime CPUID per op, first call
//! crate::simd_runtime::add_mul_f32(acc, a, b);
//!
//! // Pattern 3: one runtime CPUID total, then per-op fn-ptr deref
//! let ops = crate::simd_runtime::cpu_ops();
//! unsafe { (ops.add_mul_f32)(acc, a, b); }
//! ```
//!
//! # The naughty bit — data-driven lookup
//!
//! The [`CpuOps`] selection is driven by data scraped from GCC's
//! `aarch64-cores.def` and `i386.h` (per the matrix doc § M). The
//! per-tier static `CpuOps` instances are baked in below; the
//! [`cpu_ops_for_tier`] lookup table maps a tier name → `&'static
//! CpuOps` for debugging and explicit-tier-pinning ("I want to force
//! the AVX2 arm even though AMX is available, to measure overhead").
//!
//! Future work: code-gen the table from a build.rs that fetches GCC's
//! latest core list. For now the table is hand-rolled from the scrape
//! recorded in `.claude/knowledge/agnostic-surface-cpu-matrix.md § M`.

use std::sync::LazyLock;

/// Per-CPU operations DTO — fn-pointer table for every op currently
/// in `crate::simd_runtime::*`.
///
/// `Copy + 'static`. Held by reference (`&'static CpuOps`) by every
/// consumer — the lifetime is fine because each instance is a static
/// const baked at compile time. Cache-resident after first read.
#[derive(Copy, Clone)]
pub struct CpuOps {
    /// Human-readable tier name (`"amx_int8"`, `"avx512vnni"`,
    /// `"avxvnni"`, `"avx2_fma"`, `"neon_bf16"`, `"neon_dotprod"`,
    /// `"neon"`, `"scalar"`).
    pub tier: &'static str,

    /// Architecture string (`"x86_64"`, `"aarch64"`, `"scalar"`).
    pub arch: &'static str,

    /// `u8 × i8 → i32` dot product.
    pub vnni_dot_u8_i8: unsafe fn(&[u8], &[i8]) -> i32,

    /// `acc[i] += a[i] * b[i]` (f32, single-rounded FMA).
    pub add_mul_f32: unsafe fn(&mut [f32], &[f32], &[f32]),

    /// `acc[i] += a[i] * b[i]` (f64, single-rounded FMA).
    pub add_mul_f64: unsafe fn(&mut [f64], &[f64], &[f64]),
}

// ────────────────────────────────────────────────────────────────────────
// Per-tier static instances
// ────────────────────────────────────────────────────────────────────────
//
// Each `CPU_OPS_*` is a `&'static CpuOps` baked at compile time. The
// fn ptrs reference the existing trampolines in
// `crate::simd_runtime::{vnni_dot, add_mul}`. We don't duplicate kernel
// code here; this module is pure dispatch glue.

#[cfg(target_arch = "x86_64")]
static CPU_OPS_AMX_INT8: CpuOps = CpuOps {
    tier: "amx_int8",
    arch: "x86_64",
    // VNNI dot prefers AVX-512 VNNI on AMX hosts (matmul uses TDPBUSD;
    // for slice-level dot the AVX-512 path is still the right primitive
    // since TDPBUSD operates on tile registers, not single rows).
    vnni_dot_u8_i8: super::vnni_dot::vnni_dot_u8_i8_avx512_with_tail_safe,
    add_mul_f32: super::add_mul::add_mul_f32_avx512_safe,
    add_mul_f64: super::add_mul::add_mul_f64_avx512_safe,
};

#[cfg(target_arch = "x86_64")]
static CPU_OPS_AVX512_VNNI: CpuOps = CpuOps {
    tier: "avx512vnni",
    arch: "x86_64",
    vnni_dot_u8_i8: super::vnni_dot::vnni_dot_u8_i8_avx512_with_tail_safe,
    add_mul_f32: super::add_mul::add_mul_f32_avx512_safe,
    add_mul_f64: super::add_mul::add_mul_f64_avx512_safe,
};

#[cfg(target_arch = "x86_64")]
static CPU_OPS_AVX512F: CpuOps = CpuOps {
    tier: "avx512f",
    arch: "x86_64",
    // No VNNI on this tier — falls back to scalar wrapper.
    vnni_dot_u8_i8: super::vnni_dot::vnni_dot_u8_i8_scalar_wrapper,
    add_mul_f32: super::add_mul::add_mul_f32_avx512_safe,
    add_mul_f64: super::add_mul::add_mul_f64_avx512_safe,
};

#[cfg(target_arch = "x86_64")]
static CPU_OPS_AVXVNNI: CpuOps = CpuOps {
    tier: "avxvnni",
    arch: "x86_64",
    vnni_dot_u8_i8: super::vnni_dot::vnni2_dot_u8_i8_safe,
    add_mul_f32: super::add_mul::add_mul_f32_avx2_fma_safe,
    add_mul_f64: super::add_mul::add_mul_f64_avx2_fma_safe,
};

#[cfg(target_arch = "x86_64")]
static CPU_OPS_AVX2_FMA: CpuOps = CpuOps {
    tier: "avx2_fma",
    arch: "x86_64",
    vnni_dot_u8_i8: super::vnni_dot::vnni_dot_u8_i8_scalar_wrapper,
    add_mul_f32: super::add_mul::add_mul_f32_avx2_fma_safe,
    add_mul_f64: super::add_mul::add_mul_f64_avx2_fma_safe,
};

#[cfg(target_arch = "aarch64")]
static CPU_OPS_NEON: CpuOps = CpuOps {
    tier: "neon",
    arch: "aarch64",
    vnni_dot_u8_i8: super::vnni_dot::vnni_dot_u8_i8_scalar_wrapper,
    add_mul_f32: super::add_mul::add_mul_f32_neon_safe,
    add_mul_f64: super::add_mul::add_mul_f64_neon_safe,
};

/// Universal scalar fallback. Always available on every target.
static CPU_OPS_SCALAR: CpuOps = CpuOps {
    tier: "scalar",
    arch: "scalar",
    vnni_dot_u8_i8: super::vnni_dot::vnni_dot_u8_i8_scalar_wrapper,
    add_mul_f32: super::add_mul::add_mul_f32_scalar_safe,
    add_mul_f64: super::add_mul::add_mul_f64_scalar_safe,
};

// ────────────────────────────────────────────────────────────────────────
// Selection
// ────────────────────────────────────────────────────────────────────────

/// Lazily-selected per-CPU ops table for the current host.
///
/// First call: ~1µs (LazyLock initialization + CPUID via
/// [`simd_caps`]). Subsequent calls: one atomic-acquire load that's
/// cache-resident; ~1-2 ns. Every op-method on the returned
/// `&'static CpuOps` is then ONE indirect call (no further LazyLock).
///
/// Compared to the per-op LazyLocks in
/// [`crate::simd_runtime::vnni_dot::vnni_dot_u8_i8`] etc., this
/// pattern wins when a consumer touches many different SIMD ops in
/// one critical section — they amortize a single LazyLock load
/// across all ops instead of paying once per op.
///
/// [`simd_caps`]: crate::hpc::simd_caps::simd_caps
pub fn cpu_ops() -> &'static CpuOps {
    static SELECTED: LazyLock<&'static CpuOps> = LazyLock::new(|| {
        let _caps = crate::hpc::simd_caps::simd_caps();

        #[cfg(target_arch = "x86_64")]
        {
            // AMX tier selection: CPUID-reports-AMX is necessary but
            // not sufficient. A hypervisor may mask XCR0 bits 17/18
            // (the tile XSAVE state) or the OS may not have honoured
            // `arch_prctl(XCOMP_PERM, 18)` on Linux 5.19+. In either
            // case AMX instructions SIGILL despite the CPUID bit
            // being set. `simd_amx::amx_available()` runs the full
            // four-step gate (CPUID + OSXSAVE + XCR0 + arch_prctl);
            // demote to the AVX-512 path when the OS-check fails.
            if _caps.amx_int8 && crate::simd_amx::amx_available() {
                return &CPU_OPS_AMX_INT8;
            }
            if _caps.avx512f && _caps.avx512vnni {
                return &CPU_OPS_AVX512_VNNI;
            }
            if _caps.avx512f {
                return &CPU_OPS_AVX512F;
            }
            if _caps.avx2 && _caps.avxvnniint8 {
                return &CPU_OPS_AVXVNNI;
            }
            if _caps.avx2 && _caps.fma {
                return &CPU_OPS_AVX2_FMA;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if _caps.neon {
                return &CPU_OPS_NEON;
            }
        }

        &CPU_OPS_SCALAR
    });
    *SELECTED
}

/// Lookup `&'static CpuOps` by tier name string. Used for explicit-
/// tier-pinning (forcing AVX2 when AMX is available, e.g. for
/// benchmarking) and for debug reporting of which tier the
/// auto-detection landed on.
///
/// Returns `None` for unknown tier names. The known tier strings are
/// the values you can read from `cpu_ops().tier`:
///
///   x86_64: `"amx_int8"`, `"avx512vnni"`, `"avx512f"`, `"avxvnni"`,
///           `"avx2_fma"`
///   aarch64: `"neon"`
///   universal: `"scalar"`
///
/// Future expansion: NEON BF16 / dotprod tiers will land here once
/// the asm-byte arms ship (Phase 3b in the matrix doc).
pub fn cpu_ops_for_tier(name: &str) -> Option<&'static CpuOps> {
    match name {
        #[cfg(target_arch = "x86_64")]
        "amx_int8" => Some(&CPU_OPS_AMX_INT8),
        #[cfg(target_arch = "x86_64")]
        "avx512vnni" => Some(&CPU_OPS_AVX512_VNNI),
        #[cfg(target_arch = "x86_64")]
        "avx512f" => Some(&CPU_OPS_AVX512F),
        #[cfg(target_arch = "x86_64")]
        "avxvnni" => Some(&CPU_OPS_AVXVNNI),
        #[cfg(target_arch = "x86_64")]
        "avx2_fma" => Some(&CPU_OPS_AVX2_FMA),
        #[cfg(target_arch = "aarch64")]
        "neon" => Some(&CPU_OPS_NEON),
        "scalar" => Some(&CPU_OPS_SCALAR),
        _ => None,
    }
}

/// Lookup a [`CpuOps`] by GCC CPU codename (e.g. `"sapphirerapids"`,
/// `"neoverse-v2"`, `"apple-m2"`) on the **current build host**.
///
/// Returns `Some(&'static CpuOps)` only when the named CPU's tier is
/// reachable from the current `target_arch` (e.g. an x86_64 CPU name
/// on an x86_64 build, an aarch64 CPU name on an aarch64 build).
/// Cross-arch lookups — e.g. `cpu_ops_for_cpu("apple-m2")` on an
/// x86_64 build — return `None` because the underlying NEON kernel
/// fn pointers are compiled out and there is no honest `CpuOps` to
/// return.
///
/// For pure introspection ("what tier would this CPU pick?", with no
/// intent to call kernels), use [`cpu_tier_for_cpu`] instead — it is
/// `cfg`-free and works on any build host.
///
/// Returns `None` for unknown CPU names. Only modern (V8.2-A+ on
/// aarch64, AVX-512+ or AVX-VNNI+ on x86_64) names are mapped — older
/// silicon falls through to `cpu_ops_for_tier("scalar")` by
/// convention if you really need it.
pub fn cpu_ops_for_cpu(name: &str) -> Option<&'static CpuOps> {
    cpu_ops_for_tier(cpu_tier_for_cpu(name)?)
}

/// Lookup the dispatch tier name (e.g. `"amx_int8"`, `"avx512vnni"`,
/// `"neon"`) for a GCC CPU codename. Data from the scrape recorded
/// in `.claude/knowledge/agnostic-surface-cpu-matrix.md` § M
/// (aarch64) plus the GCC i386 cpu definitions for x86_64.
///
/// `cfg`-free — works on any build host regardless of `target_arch`.
/// This is the right entry point for cross-target introspection:
/// deployment-planning tools, cross-compilation reports, integration
/// tests that assert "apple-m2 lands at the neon tier" without
/// actually building for that silicon.
///
/// Returns `None` for unknown CPU names.
pub fn cpu_tier_for_cpu(cpu: &str) -> Option<&'static str> {
    Some(match cpu {
        // x86_64 — AMX-INT8 hosts
        "sapphirerapids" | "graniterapids" | "graniterapids-d" | "emeraldrapids" => "amx_int8",

        // x86_64 — AVX-512 + VNNI (no AMX)
        "cascadelake" | "cooperlake" | "icelake-client" | "icelake-server" | "tigerlake" | "rocketlake" | "znver4"
        | "znver5" => "avx512vnni",

        // x86_64 — AVX-512F only (no VNNI)
        "skylake-avx512" => "avx512f",

        // x86_64 — AVX-VNNI (no AVX-512)
        "alderlake" | "raptorlake" | "meteorlake" | "arrowlake" | "arrowlake-s" | "lunarlake" | "pantherlake"
        | "sierraforest" | "grandridge" => "avxvnni",

        // x86_64 — plain AVX2 + FMA (Haswell baseline, Zen 1-3)
        "haswell" | "broadwell" | "skylake" | "kaby-lake" | "comet-lake" | "znver1" | "znver2" | "znver3" => "avx2_fma",

        // aarch64 — V8.6+/V9 (would map to neon_bf16 once that tier
        // lands; today both V8.6+ and the V8.2 baseline land at the
        // generic NEON tier).
        "apple-m1" | "apple-m2" | "apple-m3" | "apple-m4" | "oryon-1" | "ampere1" | "ampere1a" | "ampere1b"
        | "cortex-a76" | "cortex-a78" | "cortex-a510" | "cortex-a520" | "cortex-a710" | "cortex-a715"
        | "cortex-a720" | "cortex-a725" | "cortex-x1" | "cortex-x2" | "cortex-x3" | "cortex-x4" | "cortex-x925"
        | "neoverse-n1" | "neoverse-n2" | "neoverse-n3" | "neoverse-v1" | "neoverse-v2" | "neoverse-v3" | "grace"
        | "cortex-a72" | "cortex-a53" => "neon",

        // Unknown CPU
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_ops_resolves_on_this_host() {
        let ops = cpu_ops();
        // Tier name is non-empty.
        assert!(!ops.tier.is_empty(), "tier name is empty");
        // Arch is one of the known classes.
        assert!(matches!(ops.arch, "x86_64" | "aarch64" | "scalar"));
        eprintln!("cpu_ops() resolved tier={} arch={}", ops.tier, ops.arch);
    }

    #[test]
    fn cpu_ops_stable_across_calls() {
        let a = cpu_ops() as *const _ as usize;
        let b = cpu_ops() as *const _ as usize;
        assert_eq!(a, b, "cpu_ops() returned different pointers across calls");
    }

    #[test]
    fn cpu_ops_for_tier_known_names() {
        assert!(cpu_ops_for_tier("scalar").is_some());
        #[cfg(target_arch = "x86_64")]
        {
            assert!(cpu_ops_for_tier("avx2_fma").is_some());
            assert!(cpu_ops_for_tier("amx_int8").is_some());
            assert!(cpu_ops_for_tier("avx512vnni").is_some());
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert!(cpu_ops_for_tier("neon").is_some());
        }
        assert!(cpu_ops_for_tier("nonsense_tier").is_none());
    }

    #[test]
    fn cpu_tier_for_cpu_data_driven_lookup() {
        // Spot-check the GCC-scraped mapping (matrix doc § M). This
        // function is cfg-free — every assertion must hold on every
        // build host, regardless of target_arch.
        assert_eq!(cpu_tier_for_cpu("sapphirerapids"), Some("amx_int8"));
        assert_eq!(cpu_tier_for_cpu("graniterapids"), Some("amx_int8"));
        assert_eq!(cpu_tier_for_cpu("cascadelake"), Some("avx512vnni"));
        assert_eq!(cpu_tier_for_cpu("znver4"), Some("avx512vnni"));
        assert_eq!(cpu_tier_for_cpu("znver5"), Some("avx512vnni"));
        assert_eq!(cpu_tier_for_cpu("alderlake"), Some("avxvnni"));
        assert_eq!(cpu_tier_for_cpu("arrowlake"), Some("avxvnni"));
        assert_eq!(cpu_tier_for_cpu("haswell"), Some("avx2_fma"));
        assert_eq!(cpu_tier_for_cpu("znver3"), Some("avx2_fma"));

        assert_eq!(cpu_tier_for_cpu("apple-m2"), Some("neon"));
        assert_eq!(cpu_tier_for_cpu("neoverse-v2"), Some("neon"));
        assert_eq!(cpu_tier_for_cpu("oryon-1"), Some("neon"));
        assert_eq!(cpu_tier_for_cpu("grace"), Some("neon"));

        assert_eq!(cpu_tier_for_cpu("totally-fake-cpu"), None);
    }

    /// Regression for the cross-arch-introspection bug Codex flagged
    /// on PR #187: `cpu_tier_for_cpu` MUST return the same Some-string
    /// regardless of the build host. Previously, ARM CPU names like
    /// `"apple-m2"` would fall to `None` on an x86_64 build because the
    /// lookup piped through the cfg-gated `cpu_ops_for_tier`.
    #[test]
    fn cpu_tier_for_cpu_is_cross_arch() {
        // These four must resolve on EVERY build host (x86_64, aarch64,
        // wasm, etc.) — no cfg gating on this surface.
        assert_eq!(cpu_tier_for_cpu("apple-m2"), Some("neon"));
        assert_eq!(cpu_tier_for_cpu("sapphirerapids"), Some("amx_int8"));
        assert_eq!(cpu_tier_for_cpu("neoverse-v2"), Some("neon"));
        assert_eq!(cpu_tier_for_cpu("alderlake"), Some("avxvnni"));
    }

    #[test]
    fn cpu_ops_call_through_dto() {
        // The whole point: call ops through the DTO and verify it
        // produces correct results. Exercises the indirect-call
        // through the fn ptr without going through the per-op LazyLock.
        let ops = cpu_ops();
        let a: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();
        let b: Vec<i8> = (0..100).map(|i| ((i * 3) % 256) as u8 as i8).collect();
        let got = unsafe { (ops.vnni_dot_u8_i8)(&a, &b) };
        let expected: i32 = (0..100).map(|i| a[i] as i32 * b[i] as i32).sum();
        assert_eq!(got, expected, "vnni_dot via CpuOps DTO mismatch");

        let mut acc = vec![1.0f32; 32];
        let xa = vec![2.0f32; 32];
        let xb = vec![3.0f32; 32];
        unsafe { (ops.add_mul_f32)(&mut acc, &xa, &xb) };
        for &v in &acc {
            assert!((v - 7.0).abs() < 1e-6, "add_mul_f32 via DTO: got {v}");
        }
    }
}
