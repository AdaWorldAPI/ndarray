//! Build-vs-CPU guard: turn "built for a newer CPU" from a SIGILL into a message.
//!
//! The default build is `-Ctarget-cpu=native` (`.cargo/config.toml`), so the
//! binary is compiled for the instruction set of the machine that BUILT it, and
//! LLVM may emit any of those instructions anywhere, not only inside
//! `crate::simd`. Running such a binary on a CPU that lacks one of them dies
//! on the first such instruction with `SIGILL` (illegal instruction), with no
//! hint of why. That happens when the build host and the run host differ: a
//! release asset built on a CI runner, a Docker image built on one machine and
//! deployed on another.
//!
//! This module compares the target features the crate was COMPILED with
//! (`cfg!(target_feature = ...)`) against what the running CPU REPORTS
//! (CPUID and XCR0 read directly, see below), and names every feature that is
//! missing. x86_64 only for now; aarch64 is documented below as not covered.
//!
//! It also enforces one floor that is NOT a compile-time feature: **AVX2 on
//! every x86_64 build that uses the AVX2 realization.** `crate::simd` selects
//! that realization whenever AVX-512 is absent, except on the AVX-without-AVX2
//! arm (`simd_avx.rs`), and it calls AVX2 intrinsics without a runtime check,
//! so a baseline build compiled without `avx2` still needs AVX2 the moment it
//! touches the SIMD types. The floor is checked once, here, at startup, and
//! never per call: dispatch stays compile-time (operator, 2026-10-10, after
//! codex review on PR #348).
//!
//! # What it does not do
//!
//! It never changes how anything is built. Cross-building (for example
//! `--config .cargo/config-v4.toml` on a runner without AVX-512) is unaffected:
//! the check runs when the finished program STARTS, never at compile time. It
//! only ever fires for a binary that would otherwise have crashed with SIGILL
//! on this CPU, so there is no override switch to forget.
//!
//! # When it runs
//!
//! * Automatically, before `main`, on Linux / Android / FreeBSD (`.init_array`),
//!   macOS / iOS (`__mod_init_func`) and Windows (`.CRT$XCU`), for every binary
//!   that links this crate. On failure it prints the missing features to stderr
//!   and exits with status 132 (`128 + SIGILL`), the status the crash would
//!   have produced.
//! * On demand, through [`check_build_cpu`] (returns the mismatch) or
//!   [`assert_build_cpu`] (panics with the same message), for targets without a
//!   pre-`main` hook or for callers that want to report it themselves.
//!
//! # Limitation (measured)
//!
//! The guard is compiled with the build's own target features: Rust cannot
//! compile one function for a lower target than the rest of the crate. So the
//! guard itself needs whatever instruction ENCODING the build uses.
//!
//! * Covered: a build for AVX-512 (or for AVX-512 extensions such as VBMI,
//!   BF16, FP16, VNNI) run on a CPU with AVX/AVX2. The guard's own code is
//!   VEX-encoded, which such CPUs run. Measured: a `x86-64-v4` build under
//!   `qemu-x86_64 -cpu Haswell|Skylake-Server|Icelake-Server` prints the
//!   missing features and exits 132 (qemu emulates no AVX-512 at all).
//! * Not covered, by decision: a `x86-64-v3`/`v4`/`native` build run on a CPU
//!   WITHOUT AVX (pre-2011, e.g. Nehalem). There even scalar code is
//!   VEX-encoded, and the guard faults inside itself. Measured: `x86-64-v3`
//!   under `-cpu Nehalem` still dies with SIGILL in `guard_before_main`.
//!   Covering these CPUs is an OPTIONAL to-do, postponed (operator,
//!   2026-10-10); see `.claude/blackboard.md`, "Optional to-do: pre-AVX
//!   guard".
//! * Silent when it should be: the same `x86-64-v3` build under `-cpu Haswell`
//!   runs normally (exit 0).
//! * AVX2 floor: a baseline build (no `target-cpu`, CI's flags) under
//!   `-cpu Nehalem|SandyBridge|IvyBridge` prints the AVX2 message and exits
//!   132; under `-cpu Haswell|max` it runs normally. Such a build is not
//!   VEX-encoded, so on a pre-AVX CPU the guard runs and reports instead of
//!   faulting.

use std::fmt;

/// One target feature the crate was compiled with but the running CPU lacks.
pub type MissingFeature = &'static str;

/// The running CPU lacks target features this build was compiled to use.
///
/// Returned by [`check_build_cpu`]. Its `Display` names every missing
/// feature and how to rebuild.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildCpuMismatch {
    /// Features the build needs that the running CPU does not report: those
    /// enabled at compile time, plus the x86_64 AVX2 floor.
    pub missing: Vec<MissingFeature>,
}

impl fmt::Display for BuildCpuMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "this binary needs CPU features [{}], which this CPU does not support. \
             Running it would crash with SIGILL (illegal instruction).",
            self.missing.join(", ")
        )?;
        if self.missing.contains(&"avx2") {
            // The AVX2 realization was compiled in; only a different build helps.
            write!(
                f,
                " This build uses ndarray's AVX2 SIMD backend. For a CPU with AVX but \
                 not AVX2, rebuild with `target-cpu=native` on it or with \
                 `--config .cargo/config-avx.toml`; a CPU without AVX is not supported."
            )
        } else {
            write!(
                f,
                " Rebuild for this machine (the default `target-cpu=native` does that when \
                 built here), or for a common baseline such as \
                 `--config .cargo/config-v3.toml` (x86-64-v3, AVX2)."
            )
        }
    }
}

impl std::error::Error for BuildCpuMismatch {}

/// `(name, compiled-in, present-on-this-CPU)`.
type FeatureRow = (&'static str, bool, fn() -> bool);

/// Builds the table. `cfg!(target_feature = ...)` needs a string literal, so
/// the rows are generated rather than looped.
#[cfg(target_arch = "x86_64")]
macro_rules! feature_table {
    ($($name:tt => $present:expr),* $(,)?) => {
        &[$(($name, cfg!(target_feature = $name), (|| $present) as fn() -> bool),)*]
    };
}

// Why not `is_x86_feature_detected!`: that macro returns `true` WITHOUT asking
// the CPU whenever the feature is enabled at compile time (it expands to
// `cfg!(target_feature = ..) || runtime_check`). For the one question this
// module asks, "was it compiled in AND is it missing?", it therefore always
// answers "present". Measured: a `x86-64-v4` build under
// `qemu-x86_64 -cpu Haswell` passed that check and then died with SIGILL on
// its first EVEX instruction in `main`. So the CPU is asked directly.
#[cfg(target_arch = "x86_64")]
mod cpuid {
    use core::arch::x86_64::{__cpuid, __cpuid_count, CpuidResult};

    fn leaf(eax: u32, ecx: u32) -> CpuidResult {
        // CPUID exists on every x86_64 CPU, so `__cpuid_count` is a safe fn.
        // `bit` checks the maximum supported leaf before asking for one.
        __cpuid_count(eax, ecx)
    }

    fn max_basic() -> u32 {
        __cpuid(0).eax
    }

    fn max_ext() -> u32 {
        __cpuid(0x8000_0000).eax
    }

    /// Bit `b` of a register of leaf `(eax, ecx)`, or false when the leaf is
    /// beyond what the CPU reports.
    pub(super) fn bit(eax: u32, ecx: u32, reg: char, b: u32) -> bool {
        let supported = if eax >= 0x8000_0000 {
            max_ext() >= eax
        } else {
            max_basic() >= eax
        };
        if !supported || (eax == 7 && ecx > 0 && leaf(7, 0).eax < ecx) {
            return false;
        }
        let r = leaf(eax, ecx);
        let v = match reg {
            'a' => r.eax,
            'b' => r.ebx,
            'c' => r.ecx,
            _ => r.edx,
        };
        v >> b & 1 == 1
    }

    /// XCR0: which register states the OS saves on a context switch. AVX and
    /// AVX-512 instructions fault (`#UD`, i.e. SIGILL) unless these are set,
    /// even when the CPUID feature bit is.
    fn xcr0() -> u64 {
        if !bit(1, 0, 'c', 27) {
            return 0; // OSXSAVE clear: XGETBV itself would fault.
        }
        let (lo, hi): (u32, u32);
        // SAFETY: OSXSAVE (CPUID.1:ECX bit 27) is set, so XGETBV is enabled
        // and XCR0 (ECX = 0) is readable at every privilege level.
        unsafe {
            core::arch::asm!("xgetbv", in("ecx") 0u32, out("eax") lo, out("edx") hi,
                             options(nomem, nostack, preserves_flags));
        }
        (hi as u64) << 32 | lo as u64
    }

    /// The OS saves XMM + YMM state.
    pub(super) fn os_avx() -> bool {
        xcr0() & 0b110 == 0b110
    }

    /// The OS saves XMM + YMM + opmask + ZMM state.
    ///
    /// On Apple targets this is assumed: Darwin saves the AVX-512 context
    /// lazily, on first use, so XCR0 does not show it until then. LLVM's own
    /// host detection (`llvm/lib/TargetParser/Host.cpp`, `HasAVX512Save`)
    /// makes the same exception; without it this guard would refuse to start
    /// a correct AVX-512 build on an AVX-512 Mac.
    pub(super) fn os_avx512() -> bool {
        cfg!(target_vendor = "apple") || xcr0() & 0b1110_0110 == 0b1110_0110
    }
}

// Bit positions and OS-state gating mirror LLVM's `getHostCPUFeatures`
// (`llvm/lib/TargetParser/Host.cpp`, checked 2026-10-10), which is what
// `-Ctarget-cpu=native` itself consults, so "supported" here means what it
// means to the compiler that produced the build.
#[cfg(target_arch = "x86_64")]
const FEATURES: &[FeatureRow] = {
    use cpuid::{bit, os_avx as avx_os, os_avx512 as z};
    feature_table!(
        "sse3" => bit(1, 0, 'c', 0),
        "pclmulqdq" => bit(1, 0, 'c', 1),
        "ssse3" => bit(1, 0, 'c', 9),
        "fma" => bit(1, 0, 'c', 12) && avx_os(),
        "cmpxchg16b" => bit(1, 0, 'c', 13),
        "sse4.1" => bit(1, 0, 'c', 19),
        "sse4.2" => bit(1, 0, 'c', 20),
        "movbe" => bit(1, 0, 'c', 22),
        "popcnt" => bit(1, 0, 'c', 23),
        "aes" => bit(1, 0, 'c', 25),
        "xsave" => bit(1, 0, 'c', 26) && avx_os(),
        "avx" => bit(1, 0, 'c', 28) && avx_os(),
        "f16c" => bit(1, 0, 'c', 29) && avx_os(),
        "rdrand" => bit(1, 0, 'c', 30),
        "fxsr" => bit(1, 0, 'd', 24),
        "bmi1" => bit(7, 0, 'b', 3),
        "avx2" => bit(7, 0, 'b', 5) && avx_os(),
        "bmi2" => bit(7, 0, 'b', 8),
        "avx512f" => bit(7, 0, 'b', 16) && z(),
        "avx512dq" => bit(7, 0, 'b', 17) && z(),
        "rdseed" => bit(7, 0, 'b', 18),
        "adx" => bit(7, 0, 'b', 19),
        "avx512ifma" => bit(7, 0, 'b', 21) && z(),
        "avx512cd" => bit(7, 0, 'b', 28) && z(),
        "sha" => bit(7, 0, 'b', 29),
        "avx512bw" => bit(7, 0, 'b', 30) && z(),
        "avx512vl" => bit(7, 0, 'b', 31) && z(),
        "avx512vbmi" => bit(7, 0, 'c', 1) && z(),
        "avx512vbmi2" => bit(7, 0, 'c', 6) && z(),
        "gfni" => bit(7, 0, 'c', 8),
        "vaes" => bit(7, 0, 'c', 9) && avx_os(),
        "vpclmulqdq" => bit(7, 0, 'c', 10) && avx_os(),
        "avx512vnni" => bit(7, 0, 'c', 11) && z(),
        "avx512bitalg" => bit(7, 0, 'c', 12) && z(),
        "avx512vpopcntdq" => bit(7, 0, 'c', 14) && z(),
        "avx512vp2intersect" => bit(7, 0, 'd', 8) && z(),
        "avx512fp16" => bit(7, 0, 'd', 23) && z(),
        "avxvnni" => bit(7, 1, 'a', 4) && avx_os(),
        "avx512bf16" => bit(7, 1, 'a', 5) && z(),
        "xsaveopt" => bit(0xD, 1, 'a', 0) && avx_os(),
        "xsavec" => bit(0xD, 1, 'a', 1) && avx_os(),
        "xsaves" => bit(0xD, 1, 'a', 3) && avx_os(),
        "lzcnt" => bit(0x8000_0001, 0, 'c', 5),
        "sse4a" => bit(0x8000_0001, 0, 'c', 6),
        "tbm" => bit(0x8000_0001, 0, 'c', 21),
        "kl" => bit(7, 0, 'c', 23),
        "widekl" => bit(7, 0, 'c', 23) && bit(0x19, 0, 'b', 2),
        // LLVM gates SHA512/SM3/SM4 on the CPUID bit only (no XCR0 check);
        // mirrored as is.
        "sha512" => bit(7, 1, 'a', 0),
        "sm3" => bit(7, 1, 'a', 1),
        "sm4" => bit(7, 1, 'a', 2),
        "avxifma" => bit(7, 1, 'a', 23) && avx_os(),
        "avxvnniint8" => bit(7, 1, 'd', 4) && avx_os(),
        "avxneconvert" => bit(7, 1, 'd', 5) && avx_os(),
        "avxvnniint16" => bit(7, 1, 'd', 10) && avx_os(),
    )
};

/// The floor of `crate::simd`'s AVX2 realization (see the module docs):
/// required on every x86_64 build that compiles that realization in, whatever
/// its target features. The one x86_64 build that does not is the
/// AVX-without-AVX2 arm (`simd_avx.rs`, `.cargo/config-avx.toml`), whose
/// `avx` requirement the compiled-feature table already enforces.
#[cfg(target_arch = "x86_64")]
const BACKEND_FLOOR: &[FeatureRow] =
    &[("avx2", cfg!(not(all(target_feature = "avx", not(target_feature = "avx2")))), || {
        cpuid::bit(7, 0, 'b', 5) && cpuid::os_avx()
    })];

#[cfg(not(target_arch = "x86_64"))]
const BACKEND_FLOOR: &[FeatureRow] = &[];

// aarch64 is NOT covered yet, deliberately: `is_aarch64_feature_detected!`
// has the same compile-time short-circuit, so a table built on it could never
// fire, and a guard that cannot fire is worse than none (it reads as coverage).
// Covering it needs the kernel's HWCAP bits (`getauxval(AT_HWCAP)`), which std
// does not expose. Until then the explicit API returns Ok on aarch64.
#[cfg(not(target_arch = "x86_64"))]
const FEATURES: &[FeatureRow] = &[];

/// Target features the crate was compiled with that the running CPU lacks.
///
/// Empty on a CPU that supports the build, and always empty on architectures
/// this guard does not cover yet (anything but x86_64).
pub fn missing_build_features() -> Vec<MissingFeature> {
    let mut missing = missing_in(FEATURES);
    for f in missing_in(BACKEND_FLOOR) {
        if !missing.contains(&f) {
            missing.push(f);
        }
    }
    missing
}

fn missing_in(table: &[FeatureRow]) -> Vec<MissingFeature> {
    table
        .iter()
        .filter(|&&(_, compiled, detected)| compiled && !detected())
        .map(|&(name, _, _)| name)
        .collect()
}

/// Checks that the running CPU supports every target feature this build uses.
///
/// # Examples
///
/// ```
/// // A build made on the machine that runs it always passes.
/// assert!(ndarray::cpu_guard::check_build_cpu().is_ok());
/// ```
pub fn check_build_cpu() -> Result<(), BuildCpuMismatch> {
    let missing = missing_build_features();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(BuildCpuMismatch { missing })
    }
}

/// Panics with a readable message if the running CPU cannot run this build.
///
/// # Examples
///
/// ```
/// ndarray::cpu_guard::assert_build_cpu();
/// ```
pub fn assert_build_cpu() {
    if let Err(mismatch) = check_build_cpu() {
        panic!("{mismatch}");
    }
}

/// The pre-`main` hook. Prints and exits instead of panicking: unwinding out
/// of a loader-invoked `extern "C"` function would abort with no message.
extern "C" fn guard_before_main() {
    if let Err(mismatch) = check_build_cpu() {
        use std::io::Write;
        let _ = writeln!(std::io::stderr(), "error: {mismatch}");
        std::process::exit(132);
    }
}

// The loader calls every function pointer in these sections before `main`.
// `#[used]` keeps the static even though nothing references it.
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
#[used]
#[link_section = ".init_array"]
static GUARD_BEFORE_MAIN: extern "C" fn() = guard_before_main;

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[used]
#[link_section = "__DATA,__mod_init_func"]
static GUARD_BEFORE_MAIN: extern "C" fn() = guard_before_main;

#[cfg(target_os = "windows")]
#[used]
#[link_section = ".CRT$XCU"]
static GUARD_BEFORE_MAIN: extern "C" fn() = guard_before_main;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_host_runs_its_own_build() {
        // Built here, run here: nothing may be missing.
        assert_eq!(missing_build_features(), Vec::<MissingFeature>::new());
        assert!(check_build_cpu().is_ok());
    }

    #[test]
    fn the_table_sees_the_features_this_build_was_compiled_with() {
        // Anti-vacuity: on x86_64 every build has at least SSE2-era features,
        // and a native/v3 build has AVX2, so the "compiled" column must not be
        // all false (that would make the guard unable to fire).
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        assert!(FEATURES
            .iter()
            .any(|&(n, compiled, _)| n == "avx2" && compiled));
    }

    /// Every x86_64 target feature that some `-Ctarget-cpu` model enables on
    /// rustc 1.99, minus `sse`/`sse2` (the x86_64 baseline). A feature missing
    /// from `FEATURES` is one the guard cannot see, so a build that uses it on
    /// a CPU lacking it SIGILLs instead of reporting (codex review, PR #348).
    /// Regenerate on a toolchain bump:
    /// `for c in $(rustc --print target-cpus | awk 'NR>1{print $1}'); do
    ///  rustc --print cfg -Ctarget-cpu=$c; done | grep target_feature | sort -u`
    #[cfg(target_arch = "x86_64")]
    const RUSTC_CPU_MODEL_FEATURES: &[&str] = &[
        "adx", "aes", "avx", "avx2", "avx512bf16", "avx512bitalg", "avx512bw", "avx512cd", "avx512dq", "avx512f",
        "avx512fp16", "avx512ifma", "avx512vbmi", "avx512vbmi2", "avx512vl", "avx512vnni", "avx512vp2intersect",
        "avx512vpopcntdq", "avxifma", "avxneconvert", "avxvnni", "avxvnniint16", "avxvnniint8", "bmi1", "bmi2",
        "cmpxchg16b", "f16c", "fma", "fxsr", "gfni", "kl", "lzcnt", "movbe", "pclmulqdq", "popcnt", "rdrand", "rdseed",
        "sha", "sha512", "sm3", "sm4", "sse3", "sse4.1", "sse4.2", "sse4a", "ssse3", "tbm", "vaes", "vpclmulqdq",
        "widekl", "xsave", "xsavec", "xsaveopt", "xsaves",
    ];

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn the_table_covers_every_feature_a_cpu_model_can_enable() {
        let missing: Vec<_> = RUSTC_CPU_MODEL_FEATURES
            .iter()
            .filter(|f| !FEATURES.iter().any(|&(n, _, _)| n == **f))
            .collect();
        assert!(missing.is_empty(), "guard cannot see: {missing:?}");
        assert!(RUSTC_CPU_MODEL_FEATURES.len() > 40, "anti-vacuity");
    }

    #[test]
    fn a_compiled_feature_the_cpu_lacks_is_reported() {
        // Can-fire: a synthetic table, because the host supports its own build.
        let table: &[FeatureRow] = &[
            ("present", true, || true),
            ("absent", true, || false),
            ("not-compiled", false, || false),
            ("also-absent", true, || false),
        ];
        assert_eq!(missing_in(table), vec!["absent", "also-absent"]);
    }

    #[test]
    fn a_feature_not_compiled_in_is_never_reported() {
        // Can-stay-silent: a CPU lacking a feature the build does not use is fine.
        let table: &[FeatureRow] = &[("a", false, || false), ("b", true, || true)];
        assert!(missing_in(table).is_empty());
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_is_required_unless_the_avx_arm_is_compiled_in() {
        // A baseline build (no `avx2` compiled in) still reaches the AVX2 SIMD
        // backend, so the floor applies to it; only the AVX-without-AVX2 arm
        // is exempt, because it never selects that backend.
        let avx_arm = cfg!(all(target_feature = "avx", not(target_feature = "avx2")));
        assert!(BACKEND_FLOOR
            .iter()
            .any(|&(n, compiled, _)| n == "avx2" && compiled == !avx_arm));
    }

    #[test]
    fn the_message_explains_the_avx2_floor_only_when_avx2_is_missing() {
        let floor = BuildCpuMismatch { missing: vec!["avx2"] }.to_string();
        assert!(floor.contains("AVX2 SIMD backend"), "{floor}");
        assert!(floor.contains("config-avx.toml"), "{floor}");
        let other = BuildCpuMismatch {
            missing: vec!["avx512f"],
        }
        .to_string();
        assert!(!other.contains("AVX2 SIMD backend"), "{other}");
        assert!(other.contains("Rebuild"), "{other}");
    }

    #[test]
    fn the_message_names_every_missing_feature() {
        let m = BuildCpuMismatch {
            missing: vec!["avx512f", "avx512bw"],
        };
        let text = m.to_string();
        assert!(text.contains("avx512f, avx512bw"), "{text}");
        assert!(text.contains("SIGILL"), "{text}");
    }
}
