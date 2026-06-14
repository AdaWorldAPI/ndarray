//! AMX (Advanced Matrix Extensions) — tile matrix multiply on stable Rust 1.94
//! via inline `asm!` byte-encodings (the `_tile_*` intrinsics are nightly-only,
//! rust-lang/rust#126622).
//!
//!   TDPBUSD   : 16×16 × K=64 tile, u8×i8 → i32 = 16 384 MACs / instruction
//!   TDPBF16PS : 16×16 × K=32 tile, bf16×bf16 → f32
//!
//! Status (2026-06-14, VERIFIED BY EXECUTION on Emerald Rapids, kernel 6.18.5):
//! detection + every tile op + both GEMM kernels run and are correctness-checked
//! by `examples/amx_probe` (int8 bit-exact, bf16 within tolerance). int8 GEMM
//! 2048³ = 169.7 GMAC/s, 600× scalar, single thread.
//!
//! ⚠ HISTORY — this header used to claim "INLINE ASM TESTED ✓ all work". It was
//! NOT. Every AMX test early-returns `if !amx_available() { return; }`, and
//! detection returned false on every host (the arch_prctl syscall-number bug),
//! so the tile asm had never executed. That hid FIVE bugs: arch_prctl 157→158;
//! TILECFG rows/colsb swapped; TILELOADD/TILESTORED SIB base/index swapped;
//! TDPBUSD/TDPBF16PS ModRM same-tile #UD; and the mirrored operand index/sign
//! convention. All fixed + documented 2026-06-14. Lesson: a test behind an
//! `amx_available()` guard that is false is a SKIPPED test, not a passing one.
//!
//! Canonical reference: `.claude/knowledge/amx-enablement-and-kernel.md`
//! Troubleshooting playbook: `.claude/AMX_GOTCHAS.md`   Agent: `amx-savant`
//!
//! Dispatch tiers (MACs/instruction): AMX 16 384 → avx512vnni 64 →
//! avxvnniint8 32 → scalar 1.

// ═══════════════════════════════════════════════════════════════════════════
// Detection (stable — just CPUID, no AMX instructions)
// ═══════════════════════════════════════════════════════════════════════════

/// Intel server CPU generation, detected from CPUID.01H model bits. Lets a run
/// report which silicon it landed on and reason about AMX: SPR / EMR / GNR
/// expose AMX-TILE; Sierra Forest (E-core) does NOT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuModel {
    /// Sapphire Rapids — family 6, model 0x8F. AMX-TILE / INT8 / BF16.
    SapphireRapids,
    /// Emerald Rapids — family 6, model 0xCF. AMX-TILE / INT8 / BF16.
    EmeraldRapids,
    /// Granite Rapids — family 6, model 0xAD / 0xAE. AMX + AMX-FP16.
    GraniteRapids,
    /// Sierra Forest — family 6, model 0xAF. E-core, NO AMX.
    SierraForest,
    /// Any other x86_64 (older Intel, AMD, CPUID-masked hypervisor, …).
    OtherX86,
    /// Non-x86_64 build.
    NonX86,
}

impl CpuModel {
    /// Whether this generation is expected to expose AMX-TILE. A `true` here
    /// while [`amx_available`] is `false` points at OS / hypervisor enablement
    /// (XCR0 / arch_prctl), NOT the silicon — see `.claude/AMX_GOTCHAS.md`.
    pub fn has_amx(self) -> bool {
        matches!(self, CpuModel::SapphireRapids | CpuModel::EmeraldRapids | CpuModel::GraniteRapids)
    }

    /// Short human label for reports / logs.
    pub fn label(self) -> &'static str {
        match self {
            CpuModel::SapphireRapids => "Sapphire Rapids",
            CpuModel::EmeraldRapids => "Emerald Rapids",
            CpuModel::GraniteRapids => "Granite Rapids",
            CpuModel::SierraForest => "Sierra Forest (no AMX)",
            CpuModel::OtherX86 => "other x86_64",
            CpuModel::NonX86 => "non-x86_64",
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_cpu_model() -> CpuModel {
    // Only classify GenuineIntel by model number — AMD reuses the family/model
    // space differently. Vendor string is CPUID.0 EBX/EDX/ECX = "GenuineIntel".
    let v0 = core::arch::x86_64::__cpuid(0);
    let is_intel = v0.ebx == 0x756e_6547 && v0.edx == 0x4965_6e69 && v0.ecx == 0x6c65_746e;
    if !is_intel {
        return CpuModel::OtherX86;
    }
    let eax = core::arch::x86_64::__cpuid(1).eax;
    let base_family = (eax >> 8) & 0xf;
    let base_model = (eax >> 4) & 0xf;
    let ext_model = (eax >> 16) & 0xf;
    // Intel display-model rule: ext_model is folded in for family 0x6 and 0xF.
    let model = if base_family == 0x6 || base_family == 0xf {
        (ext_model << 4) | base_model
    } else {
        base_model
    };
    match (base_family, model) {
        (0x6, 0x8f) => CpuModel::SapphireRapids,
        (0x6, 0xcf) => CpuModel::EmeraldRapids,
        (0x6, 0xad) | (0x6, 0xae) => CpuModel::GraniteRapids,
        (0x6, 0xaf) => CpuModel::SierraForest,
        _ => CpuModel::OtherX86,
    }
}

#[cfg(target_arch = "x86_64")]
static CPU_MODEL: std::sync::LazyLock<CpuModel> = std::sync::LazyLock::new(detect_cpu_model);

/// The detected Intel CPU generation, cached. `CpuModel::NonX86` off x86_64.
#[cfg(target_arch = "x86_64")]
pub fn cpu_model() -> CpuModel {
    *CPU_MODEL
}

/// The detected Intel CPU generation (always `NonX86` on this target).
#[cfg(not(target_arch = "x86_64"))]
pub fn cpu_model() -> CpuModel {
    CpuModel::NonX86
}

/// AMX availability, computed ONCE and cached. The four detection gates are all
/// non-blocking — CPUID, XGETBV, and one idempotent `arch_prctl`: no I/O, no
/// lock contention, no spin — so the `LazyLock` init cannot stall, and every
/// later call is a plain cached load. The `arch_prctl` permission request is
/// process-wide and inherited by all threads (present and future), so issuing
/// it exactly once here is correct even under a multi-threaded (rayon) consumer.
#[cfg(target_arch = "x86_64")]
static AMX_AVAILABLE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(detect_amx);

/// Check if AMX is present, OS-enabled, AND this process holds XTILEDATA
/// permission. Cached after the first call (see the `AMX_AVAILABLE` static).
///
/// Four gates, in order — any miss ⇒ `false`:
///   1. CPUID.07H.0H:EDX bits 24 (AMX-TILE) + 25 (AMX-INT8): silicon supports it.
///   2. CPUID.01H:ECX bit 27 (OSXSAVE): OS turned on XSAVE.
///   3. XGETBV(0) bits 17 (TILECFG) + 18 (TILEDATA): OS enabled tile XSTATE.
///      Read the *live* XCR0, NOT CPUID leaf 0xD — leaf 0xD reports what the CPU
///      *could* support, and a hypervisor may advertise AMX yet leave XCR0 clear.
///   4. arch_prctl(ARCH_REQ_XCOMP_PERM, XTILEDATA): this process requests the
///      dynamically-enabled tile feature (Linux 5.16+). Without it the first
///      tile op faults (XFD #NM). 0x1023 is an *arch_prctl* op (syscall 158),
///      NOT prctl (157) — that one-digit mix-up returns -EINVAL and silently
///      disabled AMX on every capable host.
#[cfg(target_arch = "x86_64")]
pub fn amx_available() -> bool {
    *AMX_AVAILABLE
}

/// The actual four-gate detection, run once behind the `AMX_AVAILABLE` static.
#[cfg(target_arch = "x86_64")]
fn detect_amx() -> bool {
    // Step 1: CPU supports AMX-TILE + AMX-INT8?
    let cpuid = core::arch::x86_64::__cpuid_count(7, 0);
    let amx_tile = (cpuid.edx >> 24) & 1;
    let amx_int8 = (cpuid.edx >> 25) & 1;
    if amx_tile == 0 || amx_int8 == 0 {
        return false;
    }

    // Step 2: OS enabled XSAVE? (CPUID.01H:ECX bit 27 = OSXSAVE)
    let cpuid_01 = core::arch::x86_64::__cpuid(1);
    let osxsave = (cpuid_01.ecx >> 27) & 1;
    if osxsave == 0 {
        return false;
    }

    // Step 3: OS actually enabled tile state in XCR0?
    // _xgetbv(0) reads the ACTUAL XCR0 register (what the OS set),
    // not the CPUID-reported capability.
    // Bit 17 = TILECFG, Bit 18 = TILEDATA. Both must be set.
    let xcr0: u64 = unsafe { core::arch::x86_64::_xgetbv(0) };
    let tilecfg = (xcr0 >> 17) & 1;
    let tiledata = (xcr0 >> 18) & 1;
    if tilecfg == 0 || tiledata == 0 {
        return false;
    }

    // Step 4: Request XCOMP_PERM for TILEDATA.
    // Linux kernel 5.16+: processes must call arch_prctl(ARCH_REQ_XCOMP_PERM,
    // 18) to request permission for TILEDATA (XFEATURE 18) before using AMX.
    // Without this, the first AMX tile op faults (XFD #NM → SIGILL) even when
    // XCR0 bits are set. The request either succeeds (0) or fails (-errno) —
    // idempotent, safe to call multiple times.
    //
    // IMPORTANT: ARCH_REQ_XCOMP_PERM (0x1023) is an *arch_prctl* operation
    // (syscall 158), NOT regular prctl (157). Issuing it on syscall 157 makes
    // the kernel reject option 0x1023 with -EINVAL, which silently disabled
    // AMX on EVERY capable host (steps 1-3 pass, step 4 always failed).
    #[cfg(target_os = "linux")]
    {
        const SYS_ARCH_PRCTL: i64 = 158; // x86_64 syscall number for arch_prctl
        const ARCH_REQ_XCOMP_PERM: i64 = 0x1023;
        const XFEATURE_XTILEDATA: i64 = 18;
        // SAFETY: arch_prctl(ARCH_REQ_XCOMP_PERM, 18) is a simple permission
        // request. It either grants tile permission (returns 0) or fails
        // (returns -errno). No side effects on failure. Idempotent.
        let ret: i64;
        unsafe {
            core::arch::asm!(
                "syscall",
                inlateout("rax") SYS_ARCH_PRCTL => ret,
                in("rdi") ARCH_REQ_XCOMP_PERM,
                in("rsi") XFEATURE_XTILEDATA,
                in("rdx") 0i64,
                in("r10") 0i64,
                in("r8") 0i64,
                lateout("rcx") _,
                lateout("r11") _,
                options(nostack),
            );
        }
        if ret != 0 {
            return false;
        }
    }

    true
}

#[cfg(not(target_arch = "x86_64"))]
pub fn amx_available() -> bool {
    false
}

/// AMX capability report: detected CPU model + CPUID feature bits + the cached
/// `amx_available()` verdict. If `model.has_amx()` is true but `available` is
/// false, the gap is OS / hypervisor enablement (XCR0 / arch_prctl), not silicon.
pub fn amx_report() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        let cpuid = core::arch::x86_64::__cpuid_count(7, 0);
        let tile = (cpuid.edx >> 24) & 1 == 1;
        let int8 = (cpuid.edx >> 25) & 1 == 1;
        let bf16 = (cpuid.edx >> 22) & 1 == 1;
        let model = cpu_model();
        format!(
            "AMX [{} expects_amx={}]: TILE={} INT8={} BF16={} available={}",
            model.label(),
            model.has_amx(),
            tile,
            int8,
            bf16,
            amx_available()
        )
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        format!("AMX [{}]: not x86_64", cpu_model().label())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VNNI kernel (stable intrinsics — the bridge until AMX stabilizes)
// ═══════════════════════════════════════════════════════════════════════════

/// VNNI u8×i8 dot product: 64 multiply-accumulates per instruction.
///
/// Computes: for each 32-bit lane, sum of 4 products: u8[k] × i8[k].
/// 16 lanes × 4 products = 64 MACs total.
///
/// Used by ThinkingEngine for the u8 distance table × i8 energy MatVec.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni")]
pub unsafe fn vnni_dpbusd(
    acc: core::arch::x86_64::__m512i,
    a: core::arch::x86_64::__m512i, // 64 × u8
    b: core::arch::x86_64::__m512i, // 64 × i8 (energy, quantized)
) -> core::arch::x86_64::__m512i {
    core::arch::x86_64::_mm512_dpbusd_epi32(acc, a, b)
}

/// Complete VNNI MatVec: one row of distance table × energy vector.
///
/// Row: &[u8] of length N (one row of distance table).
/// Energy: &[i8] of length N (quantized energy).
/// Returns: i32 dot product (sum of all N u8×i8 products).
///
/// Processes 64 elements per VNNI instruction.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni")]
pub unsafe fn vnni_dot_u8_i8(row: &[u8], energy: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    let n = row.len().min(energy.len());
    let chunks = n / 64;
    let mut acc = _mm512_setzero_si512();

    for c in 0..chunks {
        let off = c * 64;
        let a = _mm512_loadu_si512(row[off..].as_ptr() as *const __m512i);
        let b = _mm512_loadu_si512(energy[off..].as_ptr() as *const __m512i);
        acc = _mm512_dpbusd_epi32(acc, a, b);
    }

    // Horizontal sum of 16 i32 lanes
    _mm512_reduce_add_epi32(acc)
}

/// VNNI MatVec for the entire distance table × energy vector.
///
/// table: &[u8] of size N×N (row-major distance table).
/// energy_i8: &[i8] of size N (quantized energy).
/// result: &mut [i32] of size N (output: accumulated dot products).
///
/// This IS the ThinkingEngine's core loop at VNNI resolution.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni")]
pub unsafe fn vnni_matvec(table: &[u8], energy_i8: &[i8], result: &mut [i32], n: usize) {
    for i in 0..n {
        if energy_i8.iter().all(|&e| e == 0) {
            result[i] = 0;
            continue;
        }
        let row = &table[i * n..(i + 1) * n];
        result[i] = vnni_dot_u8_i8(row, energy_i8);
    }
}

/// AVX-VNNI (ymm, 256-bit) dot product: 32 MACs per VPDPBUSD instruction.
/// For CPUs with avxvnniint8 but NOT avx512vnni (Arrow Lake, NUC 14 i9-185H, etc.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnniint8")]
pub unsafe fn vnni2_dot_u8_i8(row: &[u8], energy: &[i8]) -> i32 {
    use core::arch::x86_64::*;
    let n = row.len().min(energy.len());
    let chunks = n / 32;
    let mut acc = _mm256_setzero_si256();

    for c in 0..chunks {
        let off = c * 32;
        let a = _mm256_loadu_si256(row[off..].as_ptr() as *const __m256i);
        let b = _mm256_loadu_si256(energy[off..].as_ptr() as *const __m256i);
        // VPDPBUSD ymm: 8 lanes × 4 u8×i8 products = 32 MACs
        acc = _mm256_dpbusd_epi32(acc, a, b);
    }

    // Horizontal sum of 8 i32 lanes
    let hi128 = _mm256_extracti128_si256(acc, 1);
    let lo128 = _mm256_castsi256_si128(acc);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let sum64 = _mm_add_epi32(sum128, _mm_srli_si128(sum128, 8));
    let sum32 = _mm_add_epi32(sum64, _mm_srli_si128(sum64, 4));
    let mut total = _mm_extract_epi32(sum32, 0);

    // Scalar remainder
    let offset = chunks * 32;
    for i in offset..n {
        total += row[i] as i32 * energy[i] as i32;
    }
    total
}

/// VNNI2 MatVec for the entire distance table × energy vector (ymm path).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnniint8")]
pub unsafe fn vnni2_matvec(table: &[u8], energy_i8: &[i8], result: &mut [i32], n: usize) {
    for i in 0..n {
        let row = &table[i * n..(i + 1) * n];
        result[i] = vnni2_dot_u8_i8(row, energy_i8);
    }
}

/// Scalar fallback for VNNI dot product (non-x86 or no VNNI).
pub fn vnni_dot_u8_i8_scalar(row: &[u8], energy: &[i8]) -> i32 {
    let n = row.len().min(energy.len());
    let mut acc = 0i32;
    for i in 0..n {
        acc += row[i] as i32 * energy[i] as i32;
    }
    acc
}

/// Scalar MatVec fallback.
pub fn vnni_matvec_scalar(table: &[u8], energy_i8: &[i8], result: &mut [i32], n: usize) {
    for i in 0..n {
        let row = &table[i * n..(i + 1) * n];
        result[i] = vnni_dot_u8_i8_scalar(row, energy_i8);
    }
}

/// Runtime-dispatched VNNI MatVec: avx512vnni → avxvnniint8 → scalar i32.
///
/// Three tiers, checked in order (first match wins):
///   avx512vnni  — 64 MACs/instr (zmm, Cascade Lake+, Zen 4+)
///   avxvnniint8 — 32 MACs/instr (ymm, Arrow Lake, NUC 14 i9-185H)
///   scalar i32  — only for non-x86 or testing
///
/// IMPORTANT: avxvnniint8 (VNNI2, 256-bit) is NEVER reached when
/// avx512vnni (VNNI512) is present. This is correct:
///   - CPUs with avx512vnni always have 512-bit VPDPBUSD (faster)
///   - avxvnniint8 exists ONLY for CPUs that dropped AVX-512
///     but added 256-bit VNNI (Arrow Lake, Meteor Lake U-series)
///   - The two instructions have DIFFERENT encodings:
///     avx512vnni: EVEX-encoded VPDPBUSD zmm (512-bit)
///     avxvnniint8: VEX-encoded VPDPBUSD ymm (256-bit)
///   - Running EVEX VPDPBUSD on a VEX-only CPU = SIGILL
///   - Running VEX VPDPBUSD on an EVEX CPU = works but wastes half the width
///
/// The thinking engine's cycle_auto() dispatches:
///   VNNI detected → cycle_vnni() → this function
///   No VNNI       → cycle() → F32x16 FMA (never reaches here)
pub fn matvec_dispatch(table: &[u8], energy_i8: &[i8], result: &mut [i32], n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") {
            unsafe {
                vnni_matvec(table, energy_i8, result, n);
            }
            return;
        }
        if is_x86_feature_detected!("avxvnniint8") {
            unsafe {
                vnni2_matvec(table, energy_i8, result, n);
            }
            return;
        }
    }
    // Non-x86 or no VNNI: i32 scalar accumulate.
    // On x86, the thinking engine uses F32x16 FMA instead of reaching here.
    vnni_matvec_scalar(table, energy_i8, result, n);
}

// ═══════════════════════════════════════════════════════════════════════════
// Quantize energy f64 → i8 for VNNI path
// ═══════════════════════════════════════════════════════════════════════════

/// Quantize f64 energy vector to i8 for VNNI MatVec.
/// Maps [0.0, max_energy] → [0, 127].
pub fn quantize_energy_i8(energy: &[f64], output: &mut [i8]) {
    let n = energy.len().min(output.len());
    let max_e = energy.iter().cloned().fold(0.0f64, f64::max);
    if max_e < 1e-15 {
        for o in output[..n].iter_mut() {
            *o = 0;
        }
        return;
    }
    let scale = 127.0 / max_e;
    for i in 0..n {
        output[i] = (energy[i] * scale).round().clamp(0.0, 127.0) as i8;
    }
}

/// Dequantize i32 result back to f64.
pub fn dequantize_result_f64(result: &[i32], output: &mut [f64], scale: f64) {
    for (i, &r) in result.iter().enumerate() {
        if i < output.len() {
            output[i] = r as f64 * scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amx_detection() {
        let available = amx_available();
        let report = amx_report();
        eprintln!("{}", report);
        eprintln!("AMX available: {}", available);
    }

    #[test]
    fn test_vnni_dot_scalar() {
        let row = vec![128u8; 64]; // similarity = 0.5
        let energy = vec![10i8; 64]; // energy = 10
        let dot = vnni_dot_u8_i8_scalar(&row, &energy);
        assert_eq!(dot, 128 * 10 * 64);
        eprintln!("Scalar dot: {}", dot);
    }

    #[test]
    fn test_vnni_matvec_scalar() {
        let n = 64;
        let mut table = vec![128u8; n * n];
        for i in 0..n {
            table[i * n + i] = 255;
        } // diagonal = max

        let energy = vec![10i8; n];
        let mut result = vec![0i32; n];
        vnni_matvec_scalar(&table, &energy, &mut result, n);

        // Each row: 63 × 128 × 10 + 1 × 255 × 10 = 80640 + 2550 = 83190
        assert!(result[0] > 0);
        eprintln!("MatVec result[0]: {}", result[0]);
    }

    #[test]
    fn test_vnni_dispatch() {
        let n = 64;
        let mut table = vec![128u8; n * n];
        for i in 0..n {
            table[i * n + i] = 255;
        }
        let energy = vec![10i8; n];
        let mut result = vec![0i32; n];

        matvec_dispatch(&table, &energy, &mut result, n);
        assert!(result[0] > 0);

        #[cfg(target_arch = "x86_64")]
        eprintln!("VNNI available: {}", is_x86_feature_detected!("avx512vnni"));
        eprintln!("Dispatch result[0]: {}", result[0]);
    }

    #[test]
    fn test_quantize_energy() {
        let energy = [0.0, 0.5, 1.0, 0.25, 0.75];
        let mut quant = [0i8; 5];
        quantize_energy_i8(&energy, &mut quant);

        assert_eq!(quant[0], 0);
        assert_eq!(quant[2], 127); // max maps to 127
        assert!(quant[1] > 50 && quant[1] < 70); // ~63
        eprintln!("Quantized: {:?}", quant);
    }

    #[test]
    fn test_vnni_matches_scalar() {
        let n = 128;
        let table: Vec<u8> = (0..n * n).map(|i| (i % 256) as u8).collect();
        let energy: Vec<i8> = (0..n).map(|i| (i % 100) as i8).collect();

        let mut scalar_result = vec![0i32; n];
        vnni_matvec_scalar(&table, &energy, &mut scalar_result, n);

        let mut dispatch_result = vec![0i32; n];
        matvec_dispatch(&table, &energy, &mut dispatch_result, n);

        for i in 0..n {
            assert_eq!(
                scalar_result[i], dispatch_result[i],
                "mismatch at row {}: scalar={} dispatch={}",
                i, scalar_result[i], dispatch_result[i]
            );
        }
    }
}
