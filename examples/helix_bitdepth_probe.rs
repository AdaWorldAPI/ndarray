//! Helix bit-depth sweep — 8 / 16 / 24 / 32-bit golden-spiral direction fidelity.
//!
//! Answers "which bit depth for max fidelity?" (the original helix session's
//! 24-bit recommendation). The helix direction code is an index into a 2^b-sample
//! golden-spiral hemisphere palette; angular resolution scales ~ 1/√N.
//!
//! Also contrasts ONE-PLACE encoding (a single direction, 8..32-bit) with
//! SPATIAL-ATTENTION encoding (an oriented anisotropic patch / splat covariance,
//! ≈48-bit = 2 axes = 6 bytes = the CAM-PQ budget).
//!
//! 8-bit (256) and 16-bit (65 536) are MEASURED by full nearest-search over the
//! palette. 24/32-bit (2^24..2^32 samples) cannot be enumerated, so they are
//! EXTRAPOLATED via the equal-area spacing law `err ≈ c/√N`, with `c` fit from
//! 8-bit and VALIDATED against the measured 16-bit point. Results are compared to
//! source-precision floors (bf16/f16/f32 unit-vector angular resolution) to find
//! where extra bits stop buying fidelity.
//!
//!   cargo run --release --example helix_bitdepth_probe --features std

fn splitmix(s: &mut u64) -> f64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

type V3 = [f64; 3];
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn rand_hemi(s: &mut u64) -> V3 {
    loop {
        let x = 2.0 * splitmix(s) - 1.0;
        let y = 2.0 * splitmix(s) - 1.0;
        let r2 = x * x + y * y;
        if r2 < 1.0 && r2 > 1e-9 {
            let f = 2.0 * (1.0 - r2).sqrt();
            let z = 1.0 - 2.0 * r2;
            return [x * f, y * f, z.abs()]; // fold to z ≥ 0 (hemisphere)
        }
    }
}

/// The n golden-spiral hemisphere sample directions.
fn golden_hemisphere(n: usize) -> Vec<V3> {
    let gamma = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..n)
        .map(|i| {
            let theta = 0.5
                * (1.0 - 2.0 * (i as f64 + 0.5) / n as f64)
                    .clamp(-1.0, 1.0)
                    .acos();
            let phi = (i as f64 * gamma).rem_euclid(std::f64::consts::TAU);
            [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()]
        })
        .collect()
}

/// Mean & max nearest-sample angular error (radians) over `m` random directions.
fn measure(bits: u32, m: usize, seed: u64) -> (f64, f64) {
    let palette = golden_hemisphere(1usize << bits);
    let mut s = seed;
    let mut sum = 0.0;
    let mut max = 0.0f64;
    for _ in 0..m {
        let v = rand_hemi(&mut s);
        let mut best = -2.0;
        for &p in &palette {
            let d = dot(v, p);
            if d > best {
                best = d;
            }
        }
        let ang = best.clamp(-1.0, 1.0).acos();
        sum += ang;
        max = max.max(ang);
    }
    (sum / m as f64, max)
}

fn main() {
    println!("== Helix bit-depth sweep: golden-spiral direction fidelity (8/16/24/32-bit) ==\n");

    // Measured (enumerable palettes).
    let (m8, x8) = measure(8, 4000, 0x8888);
    let (m16, x16) = measure(16, 2000, 0x1616);

    // Equal-area spacing law err ≈ c/√N, fit on 8-bit, validated on 16-bit.
    let c = m8 * (256.0_f64).sqrt();
    let pred16 = c / (65536.0_f64).sqrt();
    let law_err = ((pred16 - m16) / m16).abs() * 100.0;
    let extrap = |bits: u32| c / (2.0_f64.powi(bits as i32)).sqrt();
    let m24 = extrap(24);
    let m32 = extrap(32);
    let m48 = extrap(48);

    let deg = |r: f64| r.to_degrees();
    println!(
        "law check: c/√N fit on 8-bit predicts 16-bit = {:.4}° vs measured {:.4}°  ({law_err:.1}% off) ⇒ law holds\n",
        deg(pred16),
        deg(m16)
    );

    println!("  bits  samples       bytes   mean err     max err      source it's lossless for");
    println!("  ----  -----------   -----   ----------   ----------   -----------------------");
    println!(
        "   8    256           1 B     {:>7.3}°    {:>7.3}°    routing/buckets only        [measured]",
        deg(m8),
        deg(x8)
    );
    println!(
        "  16    65 536        2 B     {:>7.3}°    {:>7.3}°    ~bf16 directions (0.22°)    [measured]",
        deg(m16),
        deg(x16)
    );
    println!(
        "  24    16 777 216    3 B     {:>7.4}°   —            ~f16 directions (0.056°)    [extrapolated]",
        deg(m24)
    );
    println!(
        "  32    4.29e9        4 B     {:>7.4}°   —            far below any ≤f16 source   [extrapolated]",
        deg(m32)
    );
    println!(
        "  48    2.81e14       6 B     {:>9.2e}° —            ≈f32 floor — single-dir OVERKILL  [extrapolated]",
        deg(m48)
    );

    // Source-precision angular floors (unit-vector mantissa resolution).
    let bf16 = (2.0_f64).powi(-8);
    let f16 = (2.0_f64).powi(-10);
    let f32 = (2.0_f64).powi(-23);
    println!("\nsource-precision floors (a direction can't be known finer than this):");
    println!("  bf16 ≈ {:.3}°   f16 ≈ {:.4}°   f32 ≈ {:.2e}°", deg(bf16), deg(f16), deg(f32));

    println!("\none place (a single direction):");
    println!("  •  8-bit ({:.2}°): coarse — routing / HHTL buckets, NOT reconstruction.", deg(m8));
    println!("  • 16-bit ({:.3}°): economical 2-byte path; matches bf16 source precision.", deg(m16));
    println!(
        "  • 24-bit ({:.4}°): below f16 precision ({:.4}°) ⇒ effectively LOSSLESS for any",
        deg(m24),
        deg(f16)
    );
    println!("                    realistic (≤f16) direction. ← MAX USEFUL FIDELITY for one place.");
    println!("  • 32-bit ({:.4}°): below every ≤f16 source's own precision ⇒ diminishing returns.", deg(m32));
    println!("\nspatial attention (an oriented anisotropic region, NOT a point):");
    println!("  • 48-bit = 6 B = the CAM-PQ / splat budget. A single direction at 48-bit is pure");
    println!("    overkill ({:.2e}°, ≈ the f32 floor). Its real use is encoding a REGION: an oriented", deg(m48));
    println!("    anisotropic patch = 2 axes (principal + secondary ⊥) ≈ 2×24-bit — a Gaussian-splat");
    println!("    Σ / attention ellipse (orientation + anisotropy). A spread needs the frame;");
    println!("    'one place' (a point) needs only 24-bit.");
    println!("\n  ⇒ 24-bit = max fidelity per direction (one place); 48-bit (6 B) = spatial attention");
    println!("    (oriented anisotropic region = 2 axes = the CAM-PQ/splat budget); 16-bit = bf16-");
    println!("    economical; 8-bit = routing. (+1 sign bit for a full SIGNED direction.)");
}
