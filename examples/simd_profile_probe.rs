//! `simd_profile_probe` — boot-on-silicon diagnostic for the dispatch matrix.
//!
//! Step 1 of the TEST-promotion checklist from
//! `.claude/knowledge/td-simd-cpu-dispatch-matrix.md` § "TEST verification
//! checklist": *"Boot the binary on the silicon and confirm `simd_profile()`
//! returns the expected variant."*
//!
//! Prints every CPUID-derived capability bit plus the resolved `SimdProfile`
//! variant. Used to verify silicon → profile mapping when promoting DOC
//! cells in the dispatch matrix to TEST.
//!
//! Usage:
//! ```sh
//! # Runtime detection (default — same binary on any silicon):
//! cargo run --example simd_profile_probe --release
//!
//! # Compile-time pinned (the LazyLock is not linked in):
//! cargo run --example simd_profile_probe --release --features cpu-spr
//! ```

use ndarray::hpc::simd_caps::{simd_caps, ArmProfile, SimdCaps};
use ndarray::hpc::simd_profile::{is_pinned, pinned_profile, simd_profile, SimdProfile};

fn main() {
    let caps = simd_caps();
    let profile = simd_profile();

    println!("ndarray simd-profile probe");
    println!("==========================");
    println!();

    // ── Dispatch identity ───────────────────────────────────────────
    println!("Resolved profile: {}", profile.name());
    println!("  is_x86:        {}", profile.is_x86());
    println!("  is_aarch64:    {}", profile.is_aarch64());
    println!("  has_avx512:    {}", profile.has_avx512());
    println!("  has_amx:       {}", profile.has_amx());
    println!();

    // ── Pinning status ─────────────────────────────────────────────
    println!("Compile-time pinning: {}", if is_pinned() { "ACTIVE" } else { "off (runtime detection)" });
    if let Some(p) = pinned_profile() {
        println!("  Pinned variant:  {}", p.name());
    }
    println!();

    // ── Raw capability bits ────────────────────────────────────────
    println!("SimdCaps (raw bits):");
    print_caps(&caps);
    println!();

    // ── ARM-specific sub-profile (heuristic; deployment-pragmatic) ──
    let arm = caps.arm_profile();
    if !matches!(arm, ArmProfile::NotArm) {
        println!("ARM profile (heuristic):  {}", arm.name());
        println!("  est. tok/sec:  {}", arm.estimated_tok_per_sec());
        println!("  eff. f32 lanes:{}", arm.effective_f32_lanes());
        println!();
    }

    // ── Build configuration ─────────────────────────────────────────
    println!("Build:");
    println!("  target_arch:  {}", std::env::consts::ARCH);
    println!("  target_os:    {}", std::env::consts::OS);
    #[cfg(target_feature = "avx512f")]
    println!("  -Ctarget-feature avx512f: yes (compile-time)");
    #[cfg(not(target_feature = "avx512f"))]
    println!("  -Ctarget-feature avx512f: no (compile-time)");
    #[cfg(target_feature = "avx2")]
    println!("  -Ctarget-feature avx2:    yes (compile-time)");
    #[cfg(not(target_feature = "avx2"))]
    println!("  -Ctarget-feature avx2:    no (compile-time)");
    println!();

    // ── TEST promotion guidance ────────────────────────────────────
    println!("Matrix-doc cells affected by this CPU:");
    matrix_cell_summary(profile);

    // Sanity invariant: simd_profile() and pinned_profile() must agree
    // when pinning is active. This is the same check that
    // `pinning_consistency` runs as a unit test; we re-run it here so a
    // probe binary deployed on real silicon flags any future regression
    // in the cfg cascade.
    if let Some(p) = pinned_profile() {
        assert_eq!(
            profile, p,
            "INVARIANT VIOLATION: pinned_profile()={:?} disagrees with simd_profile()={:?}",
            p, profile
        );
    }
}

fn print_caps(c: &SimdCaps) {
    let bits: &[(&str, bool)] = &[
        ("avx2", c.avx2),
        ("avx512f", c.avx512f),
        ("avx512bw", c.avx512bw),
        ("avx512vl", c.avx512vl),
        ("avx512vnni", c.avx512vnni),
        ("avx512vbmi", c.avx512vbmi),
        ("avx512vpopcntdq", c.avx512vpopcntdq),
        ("avx512bf16", c.avx512bf16),
        ("avx512fp16", c.avx512fp16),
        ("avx512vp2intersect", c.avx512vp2intersect),
        ("avxvnniint8", c.avxvnniint8),
        ("amx_tile", c.amx_tile),
        ("amx_int8", c.amx_int8),
        ("amx_bf16", c.amx_bf16),
        ("amx_fp16", c.amx_fp16),
        ("fma", c.fma),
        ("sse41", c.sse41),
        ("sse2", c.sse2),
        ("neon", c.neon),
        ("asimd_dotprod", c.asimd_dotprod),
        ("fp16 (arm)", c.fp16),
        ("aes", c.aes),
        ("sha2", c.sha2),
        ("crc32", c.crc32),
    ];
    for (name, present) in bits {
        println!("  [{}] {}", if *present { "x" } else { " " }, name);
    }
}

fn matrix_cell_summary(p: SimdProfile) {
    // Lifted from `td-simd-cpu-dispatch-matrix.md` § "Master matrix"
    // for each x86 profile. The summary is intentionally terse — the
    // matrix doc is the source of truth and should be consulted before
    // promoting any DOC cell to TEST.
    let summary: &[&str] = match p {
        SimdProfile::GraniteRapids => &[
            "F+CD+VL+DQ+BW+IFMA+VBMI+VBMI2+VNNI+BF16+FP16",
            "VPOPCNTDQ+BITALG+GFNI+VAES+VPCLMUL",
            "AMX-TILE+INT8+BF16+FP16 (FP16 is the GNR discriminator)",
        ],
        SimdProfile::SapphireRapids => &[
            "F+CD+VL+DQ+BW+IFMA+VBMI+VBMI2+VNNI+BF16+FP16",
            "VPOPCNTDQ+BITALG+GFNI+VAES+VPCLMUL",
            "AMX-TILE+INT8+BF16 (no AMX-FP16 — that's GNR)",
        ],
        SimdProfile::Zen4Avx512 => &[
            "F+CD+VL+DQ+BW+IFMA+VBMI+VBMI2+VNNI+BF16+FP16",
            "No AMX of any kind; 256-bit FPU double-pumped on Zen4, native 512-bit on Zen5",
        ],
        SimdProfile::CooperLake => &[
            "F+CD+VL+DQ+BW+VNNI+BF16",
            "No VBMI, no FP16, no AMX — unique 'BF16 without VBMI'",
        ],
        SimdProfile::TigerLakeU => &[
            "F+CD+VL+DQ+BW+IFMA+VBMI+VBMI2+VNNI+VP2INTERSECT",
            "VP2INTERSECT is the sole discriminator vs IceLakeSp",
        ],
        SimdProfile::IceLakeSp => &[
            "F+CD+VL+DQ+BW+IFMA+VBMI+VBMI2+VNNI",
            "No BF16, no FP16, no AMX, no VP2INTERSECT",
        ],
        SimdProfile::CascadeLake => &["F+CD+VL+DQ+BW+VNNI", "First Xeon with VNNI; no VBMI/BF16/FP16/AMX"],
        SimdProfile::SkylakeX => &["F+CD+VL+DQ+BW", "Founding AVX-512 baseline; everything since adds on top"],
        SimdProfile::ArrowLake => &[
            "No AVX-512 (hybrid CPU design)",
            "AVX-VNNI-INT8 + AVX-IFMA + AVX-NE-CONVERT (256-bit / VEX forms)",
        ],
        SimdProfile::HaswellAvx2 => &["AVX2 + FMA + F16C + BMI1/2", "Haswell..Coffee Lake / Zen 1-3"],
        SimdProfile::A76DotProd => &[
            "NEON + dotprod + fp16 + bf16+ + i8mm",
            "Pi 5 (BCM2712), Orange Pi 5 (RK3588), Apple M1+",
        ],
        SimdProfile::A72Fast => &[
            "NEON 128-bit + crypto (AES/SHA-2/CRC32)",
            "Pi 4 (BCM2711), Pi 3-with-crypto, Orange Pi 4 — HWCAP cannot distinguish A72 from A53-with-crypto",
        ],
        SimdProfile::A53Baseline => &[
            "NEON 128-bit baseline",
            "Rare in the wild — QEMU / minimal aarch64 without crypto",
        ],
        SimdProfile::Scalar => &["No SIMD ISA recognised", "Fallback: scalar reference kernels"],
    };
    for line in summary {
        println!("  - {}", line);
    }
}
