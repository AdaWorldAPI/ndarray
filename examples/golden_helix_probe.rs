//! Golden-helix anti-theater probe — does the irrational (golden-angle) sampling
//! and Fisher-z percentile rank earn their keep, or is the 2×2/4×4 perturbation
//! just "eigenvalue theater"?
//!
//! Two load-bearing claims from the architecture, each measured against a null:
//!
//! 1. COLLAPSE-AVOIDANCE (the Fujifilm-X-Trans / golden-ratio point).
//!    The helix places nodes on a hemisphere via the golden angle
//!    `γ = π(3−√5)`, `θ = ½·arccos(1 − 2(n+0.5)/N)`, `φ = n·γ`. An irrational
//!    stride is a low-discrepancy sampler: it should MAXIMISE the minimum
//!    nearest-neighbour gap (no two nodes collapse together) and keep the
//!    nearest-neighbour distances uniform (low coefficient of variation),
//!    BEATING both a regular (θ,φ) grid (which clumps at the pole) and uniform
//!    random (which clumps everywhere — Poisson). If golden does NOT beat both,
//!    the irrational stride is theater. Measured: min-gap (bigger = better) and
//!    NN-distance CoV (smaller = more uniform).
//!
//! 2. NO-COSINE NORMALISED KEY (palette256 Fisher-z Prozentrang).
//!    `fisher_z(s) = ½·ln((1+s)/(1-s)) = arctanh(s)` is strictly monotone in the
//!    cosine `s`, and a percentile rank of a monotone transform is monotone in
//!    `s` too — so the rank preserves EVERY pairwise similarity ordering
//!    (Spearman = 1) while being a normalised [0,1] key you can compare directly
//!    without ever re-materialising a cosine. Fisher-z additionally stretches
//!    the rim (high-|s|) so equal rank steps carry equal discriminability.
//!    Measured: ordering preservation, and rim-vs-centre resolution gain.
//!
//!   cargo run --release --example golden_helix_probe

// √5 to full f64 precision; the literal is intentionally exact (π(3−√5)).
#[allow(clippy::excessive_precision)]
const GAMMA: f64 = std::f64::consts::PI * (3.0 - 2.2360679774997896); // π(3−√5), golden angle

/// Golden-spiral hemisphere unit vectors (the helix node directions).
fn golden_hemisphere(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let theta = 0.5 * (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos(); // polar ∈ [0, π/2]
            let phi = (i as f64 * GAMMA) % (2.0 * std::f64::consts::PI);
            [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()]
        })
        .collect()
}

/// Regular (θ,φ) grid on the hemisphere — the "rational stride" null (clumps at pole).
fn regular_hemisphere(n: usize) -> Vec<[f64; 3]> {
    let side = (n as f64).sqrt().round() as usize;
    let mut v = Vec::with_capacity(side * side);
    for a in 0..side {
        for b in 0..side {
            let theta = 0.5 * std::f64::consts::PI * (a as f64 + 0.5) / side as f64;
            let phi = 2.0 * std::f64::consts::PI * (b as f64 + 0.5) / side as f64;
            v.push([theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()]);
        }
    }
    v
}

/// Uniform-random hemisphere (area-correct) — the Poisson-clumping null.
fn random_hemisphere(n: usize, seed: &mut u64) -> Vec<[f64; 3]> {
    let mut u = || {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let z = u(); // cosθ uniform in [0,1] ⇒ area-uniform on the hemisphere
            let phi = 2.0 * std::f64::consts::PI * u();
            let r = (1.0 - z * z).sqrt();
            [r * phi.cos(), r * phi.sin(), z]
        })
        .collect()
}

/// (min nearest-neighbour angle, CoV of nearest-neighbour angles). Angle = great-circle.
fn nn_stats(pts: &[[f64; 3]]) -> (f64, f64) {
    let n = pts.len();
    let mut nn = vec![f64::INFINITY; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dot = (pts[i][0] * pts[j][0] + pts[i][1] * pts[j][1] + pts[i][2] * pts[j][2]).clamp(-1.0, 1.0);
            let ang = dot.acos();
            if ang < nn[i] {
                nn[i] = ang;
            }
            if ang < nn[j] {
                nn[j] = ang;
            }
        }
    }
    let min = nn.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = nn.iter().sum::<f64>() / n as f64;
    let var = nn.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / n as f64;
    (min, var.sqrt() / mean) // (min gap, coefficient of variation)
}

fn fisher_z(s: f64) -> f64 {
    let s = s.clamp(-1.0 + 1e-9, 1.0 - 1e-9);
    0.5 * ((1.0 + s) / (1.0 - s)).ln()
}

fn main() {
    println!("== Golden-helix anti-theater probe ==\n");

    println!("[1] Collapse-avoidance — min NN gap (rad, BIGGER better) + CoV (SMALLER better):");
    println!("    N    golden(min/CoV)     regular(min/CoV)     random(min/CoV)   golden wins?");
    let mut seed = 0xABCDEF;
    for &n in &[16usize, 64, 256, 1024] {
        let (gm, gc) = nn_stats(&golden_hemisphere(n));
        let (rm, rc) = nn_stats(&regular_hemisphere(n));
        let (xm, xc) = nn_stats(&random_hemisphere(n, &mut seed));
        // "Wins" = golden has the largest min-gap AND the lowest CoV.
        let wins = gm >= rm && gm >= xm && gc <= rc && gc <= xc;
        println!(
            "  {n:>5}   {gm:.4}/{gc:.3}        {rm:.4}/{rc:.3}        {xm:.4}/{xc:.3}     {}",
            if wins { "YES" } else { "no" }
        );
    }

    println!("\n[2] Fisher-z percentile rank as a no-cosine normalised key:");
    // A deterministic spread of cosine similarities in (−1, 1).
    let mut sims: Vec<f64> = (0..1000)
        .map(|i| -0.999 + 1.998 * (i as f64 + 0.5) / 1000.0)
        .collect();
    // Percentile rank of fisher_z(s). Both fisher_z and ranking are monotone in s,
    // so the rank order must equal the cosine order — verify (Spearman == 1).
    let mut idx: Vec<usize> = (0..sims.len()).collect();
    idx.sort_by(|&a, &b| fisher_z(sims[a]).partial_cmp(&fisher_z(sims[b])).unwrap());
    let inversions = idx.windows(2).filter(|w| sims[w[0]] > sims[w[1]]).count();
    println!(
        "    rank-order vs cosine-order inversions: {inversions} (0 ⇒ ordering fully preserved, no cosine needed)"
    );

    // Rim-stretch: resolution (Δz per unit Δs) near the rim vs the centre.
    sims.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let res = |s: f64| (fisher_z(s + 0.01) - fisher_z(s - 0.01)) / 0.02;
    let centre = res(0.0);
    let rim = res(0.9);
    println!(
        "    Fisher-z resolution: centre(s=0.0) = {centre:.2}/unit, rim(s=0.9) = {rim:.2}/unit  → rim gets {:.1}× more bits",
        rim / centre
    );
    println!("    ⇒ percentile rank ∈ [0,1] is a normalised similarity key; compare ranks directly,");
    println!("      never re-materialising cosine, with extra resolution where similarity is high.");
}
