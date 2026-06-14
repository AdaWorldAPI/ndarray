//! Measurement-instrument hypothesis probe (MTMM) — syntax=angle, semantics=
//! location-residue, episodic-basin=phase. Tests the THREE as an instrument, not
//! as assumed fact, via the reliability suite (Spearman + ICC).
//!
//! The hypothesis: these are three ORTHOGONAL measurement axes. The probe builds
//! a synthetic population where three latent factors vary independently — `syn`
//! (a pairwise relation = the angle imposed between s and o), `sem` (a semantic
//! location = residue magnitude from a centroid), and `epi` (an episode index →
//! golden phase) — then measures each axis from the fingerprints and asks the
//! reliability suite four multitrait-multimethod questions:
//!
//! - CONVERGENT: does each measure track its own factor? (diagonal high)
//! - DISCRIMINANT: does it stay flat on the others? (off-diagonal ~0)
//! - RELIABLE: is it stable under observation noise? (ICC test-retest)
//! - PHASE SEPARATES: does the golden phase spread episodes? (anti-collapse)
//!
//! A clean positive = the reframing is a real instrument. Any failed cell = the
//! operationalization conflates and must be corrected (not assumed).
//!
//!   cargo run --release --example instrument_mtmm_probe --features std

use ndarray::hpc::reliability::{icc_a1, spearman};

const D: usize = 24;
const K: usize = 8;

fn splitmix(s: &mut u64) -> f64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn randn(s: &mut u64) -> f64 {
    // Box-Muller.
    let u1 = splitmix(s).max(1e-12);
    let u2 = splitmix(s);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}
fn unit(s: &mut u64) -> Vec<f64> {
    let v: Vec<f64> = (0..D).map(|_| randn(s)).collect();
    let n = norm(&v).max(1e-12);
    v.iter().map(|x| x / n).collect()
}

fn angle_between(a: &[f64], b: &[f64]) -> f64 {
    (dot(a, b) / (norm(a) * norm(b)).max(1e-12))
        .clamp(-1.0, 1.0)
        .acos()
}

fn residue_to_nearest(v: &[f64], centroids: &[Vec<f64>]) -> f64 {
    centroids
        .iter()
        .map(|c| {
            v.iter()
                .zip(c)
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

/// Min circular gap of phases in [0, 2π) — the basin-separation metric.
fn min_circular_gap(phases: &[f64]) -> f64 {
    let mut p: Vec<f64> = phases.to_vec();
    p.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut min = f64::INFINITY;
    for i in 0..p.len() {
        let gap = if i + 1 < p.len() {
            p[i + 1] - p[i]
        } else {
            p[0] + std::f64::consts::TAU - p[i]
        };
        min = min.min(gap);
    }
    min
}

fn main() {
    println!("== Measurement-instrument MTMM probe: syntax=angle · semantics=residue · basin=phase ==\n");
    let mut s = 0x5EED_4A2B_u64;
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());

    // Well-separated semantic centroids (spacing >> residue range R so the
    // nearest-centroid stays the generating one — keeps residue = a clean
    // location measure).
    let centroids: Vec<Vec<f64>> = (0..K)
        .map(|_| unit(&mut s).iter().map(|x| x * 6.0).collect())
        .collect();
    let r_max = 1.5;
    let noise = 0.18; // observation noise for the test-retest reliability leg

    let n = 6000usize;
    let (mut f_syn, mut f_sem, mut f_epi) = (vec![], vec![], vec![]);
    let (mut m_angle, mut m_res, mut m_phase) = (vec![], vec![], vec![]);
    let (mut m_angle2, mut m_res2) = (vec![], vec![]); // noisy retest
    let mut golden_phase = vec![];
    let mut random_phase = vec![];

    for i in 0..n {
        let c = (splitmix(&mut s) * K as f64) as usize % K;
        let sem = splitmix(&mut s) * r_max; // semantic residue magnitude
        let syn = 0.15 + splitmix(&mut s) * (std::f64::consts::PI - 0.30); // relation angle
        let epi = i as f64;

        // s_fp = centroid + sem·dir  (its residue from the nearest centroid = sem).
        let dir = unit(&mut s);
        let s_fp: Vec<f64> = centroids[c]
            .iter()
            .zip(&dir)
            .map(|(cc, d)| cc + sem * d)
            .collect();
        // o_fp imposes EXACTLY angle `syn` to s_fp, independent of s_fp's location:
        // o = cos(syn)·s + sin(syn)·|s|·perp,  perp ⊥ s.
        let rnd = unit(&mut s);
        let proj = dot(&rnd, &s_fp) / dot(&s_fp, &s_fp).max(1e-12);
        let mut perp: Vec<f64> = rnd.iter().zip(&s_fp).map(|(r, sf)| r - proj * sf).collect();
        let pn = norm(&perp).max(1e-12);
        for x in perp.iter_mut() {
            *x /= pn;
        }
        let sn = norm(&s_fp);
        let o_fp: Vec<f64> = s_fp
            .iter()
            .zip(&perp)
            .map(|(sf, pp)| syn.cos() * sf + syn.sin() * sn * pp)
            .collect();

        // Clean measures.
        m_angle.push(angle_between(&s_fp, &o_fp));
        m_res.push(residue_to_nearest(&s_fp, &centroids));
        let ph = (epi * golden).rem_euclid(std::f64::consts::TAU);
        m_phase.push(ph);

        // Noisy retest (observation noise on both fingerprints).
        let s2: Vec<f64> = s_fp.iter().map(|x| x + noise * randn(&mut s)).collect();
        let o2: Vec<f64> = o_fp.iter().map(|x| x + noise * randn(&mut s)).collect();
        m_angle2.push(angle_between(&s2, &o2));
        m_res2.push(residue_to_nearest(&s2, &centroids));

        f_syn.push(syn);
        f_sem.push(sem);
        f_epi.push(epi);
        golden_phase.push(ph);
        random_phase.push(splitmix(&mut s) * std::f64::consts::TAU);
    }

    // ── MTMM Spearman matrix (factor × measure) ──
    let sa = spearman(&f_syn, &m_angle);
    let sr = spearman(&f_syn, &m_res);
    let ea = spearman(&f_sem, &m_angle);
    let er = spearman(&f_sem, &m_res);
    let pa = spearman(&f_epi, &m_angle);
    let pr = spearman(&f_epi, &m_res);

    println!("MTMM Spearman ρ (factor ↓ × measure →):   [convergent=diagonal, discriminant=off-diagonal]");
    println!("                    angle        residue");
    println!("  syn (relation)    {sa:+.3}       {sr:+.3}");
    println!("  sem (location)    {ea:+.3}       {er:+.3}");
    println!("  epi (episode)     {pa:+.3}       {pr:+.3}");

    // Phase independence + reliability + separation.
    let ph_syn = spearman(&f_syn, &m_phase);
    let ph_sem = spearman(&f_sem, &m_phase);
    let icc_angle = icc_a1(&[&m_angle, &m_angle2]);
    let icc_res = icc_a1(&[&m_res, &m_res2]);
    let prefix = 64usize;
    let g_gap = min_circular_gap(&golden_phase[..prefix]);
    let r_gap = min_circular_gap(&random_phase[..prefix]);

    println!("\nphase independence:  ρ(syn,phase)={ph_syn:+.3}  ρ(sem,phase)={ph_sem:+.3}   (≈0 expected)");
    println!("reliability (ICC test-retest @ noise {noise}):  angle={icc_angle:.3}  residue={icc_res:.3}  phase=1.000 (deterministic)");
    println!(
        "phase separation (min circular gap, first {prefix}):  golden={g_gap:.4}  random={r_gap:.4}  ({:.1}× better)",
        g_gap / r_gap.max(1e-9)
    );

    // ── Verdict ──
    let convergent = sa >= 0.9 && er >= 0.9;
    let discriminant = ea.abs() <= 0.2 && sr.abs() <= 0.2;
    let phase_indep = ph_syn.abs() <= 0.2 && ph_sem.abs() <= 0.2;
    let reliable = icc_angle >= 0.7 && icc_res >= 0.7;
    let separates = g_gap > 1.5 * r_gap;
    let mark = |b: bool| if b { "PASS" } else { "FAIL" };
    println!("\nVERDICT:");
    println!("  convergent (each measure tracks its factor) ......... {}", mark(convergent));
    println!("  discriminant (axes do not leak into each other) ..... {}", mark(discriminant));
    println!("  phase independent of content .........................{}", mark(phase_indep));
    println!("  reliable under noise (ICC ≥ 0.7) .................... {}", mark(reliable));
    println!("  golden phase separates episodes ..................... {}", mark(separates));
    let all = convergent && discriminant && phase_indep && reliable && separates;
    println!(
        "\n  ⇒ instrument {}",
        if all {
            "VALIDATED — syntax=angle ⊥ semantics=residue ⊥ basin=phase is a real 3-axis basis"
        } else {
            "NEEDS CORRECTION — at least one axis conflates; see failed cells"
        }
    );
}
