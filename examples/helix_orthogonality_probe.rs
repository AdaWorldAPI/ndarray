//! Helix ⊥ HHTL probe — the load-bearing claim of the 16-byte spec:
//! `PLACE (HHTL) + RESIDUE (HELIX) = REALITY`, with the two declared ORTHOGONAL.
//!
//! Tests a spherical/polar decomposition of a point around its cluster centroid:
//! PLACE = which centroid (the HHTL semantic basin); RADIAL = ‖point − centroid‖
//! (depth from the basin); HELIX = the residue direction, encoded to the nearest
//! of 256 golden-spiral hemisphere samples (the spec's PALETTE256) + a 1-bit sign.
//!
//! Four questions on the reliability instrument:
//!
//! - ORTHOGONAL: is the helix direction independent of PLACE and RADIAL? (ρ≈0)
//! - ADDS INFO: does centroid+radial+helix reconstruct far better than centroid alone?
//! - SEPARATES: do golden hemisphere samples spread better than random? (min gap)
//! - RELIABLE: is the encoded direction stable under observation noise? (ICC)
//!
//! Finding (measured): helix is a HEMISPHERE code (axis, ±v equivalent). A full
//! SIGNED direction needs the 256-sample index PLUS a 1-bit sign — included here;
//! WITHOUT it, reconstruction fails for half the sphere (fine_err ≈ coarse_err).
//!
//!   cargo run --release --example helix_orthogonality_probe --features std

use ndarray::hpc::reliability::{icc_a1, spearman};

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
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn unit(a: V3) -> V3 {
    let n = norm(a).max(1e-12);
    [a[0] / n, a[1] / n, a[2] / n]
}
fn rand_unit(s: &mut u64) -> V3 {
    // Marsaglia: uniform on the sphere.
    loop {
        let x = 2.0 * splitmix(s) - 1.0;
        let y = 2.0 * splitmix(s) - 1.0;
        let r2 = x * x + y * y;
        if r2 < 1.0 && r2 > 1e-9 {
            let f = 2.0 * (1.0 - r2).sqrt();
            return [x * f, y * f, 1.0 - 2.0 * r2];
        }
    }
}
/// Fold to the upper hemisphere (helix encodes an axis, not a signed vector).
fn fold(v: V3) -> V3 {
    if v[2] < 0.0 {
        [-v[0], -v[1], -v[2]]
    } else {
        v
    }
}

/// The 256 golden-spiral hemisphere sample directions (the spec's PALETTE256).
fn golden_hemisphere(n: usize) -> Vec<V3> {
    let gamma = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt()); // golden angle
    (0..n)
        .map(|i| {
            // θ = ½·arccos(1 − 2(i+0.5)/N) → equal-area on the hemisphere.
            let theta = 0.5
                * (1.0 - 2.0 * (i as f64 + 0.5) / n as f64)
                    .clamp(-1.0, 1.0)
                    .acos();
            let phi = (i as f64 * gamma).rem_euclid(std::f64::consts::TAU);
            [theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()]
        })
        .collect()
}

/// Encode a hemisphere direction to the nearest golden sample index.
fn encode(dir: V3, palette: &[V3]) -> usize {
    let mut best = -2.0;
    let mut bi = 0;
    for (i, &p) in palette.iter().enumerate() {
        let d = dot(dir, p);
        if d > best {
            best = d;
            bi = i;
        }
    }
    bi
}

fn min_angular_gap(dirs: &[V3]) -> f64 {
    let mut min = std::f64::consts::PI;
    for i in 0..dirs.len() {
        for j in (i + 1)..dirs.len() {
            let ang = dot(dirs[i], dirs[j]).clamp(-1.0, 1.0).acos();
            if ang < min {
                min = ang;
            }
        }
    }
    min
}

fn main() {
    println!("== Helix ⊥ HHTL probe: PLACE + RADIAL + HELIX(direction) spherical decomposition ==\n");
    let mut s = 0x4E11_5EED_u64;
    let palette = golden_hemisphere(256);

    let k = 16usize; // PLACE: number of cluster centroids
    let centroids: Vec<V3> = (0..k)
        .map(|_| {
            [
                6.0 * (2.0 * splitmix(&mut s) - 1.0),
                6.0 * (2.0 * splitmix(&mut s) - 1.0),
                6.0 * (2.0 * splitmix(&mut s) - 1.0),
            ]
        })
        .collect();
    let r_max = 2.0;
    let noise = 0.10;
    let n = 6000usize;

    let (mut f_place, mut f_radial, mut m_helix) = (vec![], vec![], vec![]);
    let mut coarse_err = 0.0;
    let mut fine_err = 0.0;
    let (mut dir_clean, mut dir_noisy) = (vec![], vec![]);
    let mut ang_quant_err = 0.0;

    for _ in 0..n {
        let c = (splitmix(&mut s) * k as f64) as usize % k;
        let radial = splitmix(&mut s) * r_max;
        let orient = rand_unit(&mut s); // INDEPENDENT of place & radial by construction
        let point = [
            centroids[c][0] + radial * orient[0],
            centroids[c][1] + radial * orient[1],
            centroids[c][2] + radial * orient[2],
        ];

        // Decode the spherical code from the point: hemisphere index + 1-bit sign.
        let residue = sub(point, centroids[c]);
        let rmag = norm(residue);
        let ud = unit(residue);
        let sign_z = if ud[2] < 0.0 { -1.0 } else { 1.0 }; // the 1-bit sign
        let dir = fold(ud); // hemisphere (z ≥ 0)
        let hidx = encode(dir, &palette);
        let h = palette[hidx];
        let recon_dir = [sign_z * h[0], sign_z * h[1], sign_z * h[2]]; // re-apply sign

        // Reconstruction: centroid (coarse) vs centroid + radial·helix-dir (fine).
        let recon = [
            centroids[c][0] + rmag * recon_dir[0],
            centroids[c][1] + rmag * recon_dir[1],
            centroids[c][2] + rmag * recon_dir[2],
        ];
        coarse_err += rmag; // ‖point − centroid‖
        fine_err += norm(sub(point, recon));
        ang_quant_err += dot(ud, recon_dir).clamp(-1.0, 1.0).acos();

        // Reliability: re-encode under observation noise.
        let pn = [
            point[0] + noise * (2.0 * splitmix(&mut s) - 1.0),
            point[1] + noise * (2.0 * splitmix(&mut s) - 1.0),
            point[2] + noise * (2.0 * splitmix(&mut s) - 1.0),
        ];
        let ud2 = unit(sub(pn, centroids[c]));
        let sign2 = if ud2[2] < 0.0 { -1.0 } else { 1.0 };
        let h2 = palette[encode(fold(ud2), &palette)];
        let recon_dir2 = [sign2 * h2[0], sign2 * h2[1], sign2 * h2[2]];
        for t in 0..3 {
            dir_clean.push(recon_dir[t]);
            dir_noisy.push(recon_dir2[t]);
        }

        f_place.push(c as f64);
        f_radial.push(radial);
        m_helix.push(hidx as f64);
    }

    // Orthogonality: helix index vs PLACE and vs RADIAL (expect ρ≈0).
    let rho_place = spearman(&f_place, &m_helix);
    let rho_radial = spearman(&f_radial, &m_helix);
    let coarse_rel = coarse_err / n as f64;
    let fine_rel = fine_err / n as f64;
    let icc_dir = icc_a1(&[&dir_clean, &dir_noisy]);
    let g_gap = min_angular_gap(&palette);
    let rand_dirs: Vec<V3> = (0..256).map(|_| fold(rand_unit(&mut s))).collect();
    let r_gap = min_angular_gap(&rand_dirs);

    println!("orthogonality (helix direction vs the other two axes):");
    println!("  ρ(PLACE,  helix) = {rho_place:+.3}      ρ(RADIAL, helix) = {rho_radial:+.3}   (≈0 ⇒ orthogonal)");
    println!("\nadds info (mean reconstruction error to the true point):");
    println!("  coarse PLACE-only        {coarse_rel:.4}");
    println!(
        "  + RADIAL + HELIX(dir)    {fine_rel:.4}   ({:.1}× lower; mean angular quant {:.1}°)",
        coarse_rel / fine_rel.max(1e-9),
        (ang_quant_err / n as f64).to_degrees()
    );
    println!(
        "\nseparation (min angular gap of 256 samples):  golden={:.4} rad  random={:.4} rad  ({:.1}× better)",
        g_gap,
        r_gap,
        g_gap / r_gap.max(1e-9)
    );
    println!("reliability (ICC of reconstructed direction @ noise {noise}):  {icc_dir:.3}");

    let orthogonal = rho_place.abs() <= 0.1 && rho_radial.abs() <= 0.1;
    let adds_info = fine_rel < 0.5 * coarse_rel;
    let separates = g_gap > 1.5 * r_gap;
    let reliable = icc_dir >= 0.7;
    let mark = |b: bool| if b { "PASS" } else { "FAIL" };
    println!("\nVERDICT:");
    println!("  orthogonal (helix ⊥ place, ⊥ radial) ............... {}", mark(orthogonal));
    println!("  adds info (helix recovers the residual direction) .. {}", mark(adds_info));
    println!("  golden separation > random ......................... {}", mark(separates));
    println!("  reliable under noise (ICC ≥ 0.7) ................... {}", mark(reliable));
    let all = orthogonal && adds_info && separates && reliable;
    println!(
        "\n  ⇒ {}",
        if all {
            "PLACE ⊥ RADIAL ⊥ HELIX confirmed — helix is a real orthogonal residue axis on top of HHTL (spec claim holds)"
        } else {
            "spec claim PARTIAL — see failed cells"
        }
    );
}
