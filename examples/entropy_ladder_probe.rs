//! Entropy-ladder probe — Staunen↔Wisdom over a synthetic NARS edge population.
//!
//! Shows the ladder coordinate doing real work:
//!   * the syntax/semantics/pragmatics rung partition,
//!   * the entropy×energy quadrant partition (Staunen/Confusion/Boredom/Wisdom),
//!   * the validation number — Spearman ρ(entropy, empirical accuracy) — proving
//!     entropy is a reliability proxy (more negative = stronger), and
//!   * Pearl-2³ SPO decomposition over the 3×palette-256 indices.
//!
//!   cargo run --release --example entropy_ladder_probe --features std

use ndarray::hpc::entropy_ladder::{decompose_spo, entropy_class, nars_entropy, EntropyRung, Quadrant};
use ndarray::hpc::reliability::spearman;

fn splitmix(s: &mut u64) -> f64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

fn main() {
    println!("== Entropy ladder: Staunen↔Wisdom over a synthetic NARS edge population ==\n");

    let mut s = 0x5EED_1ADD_u64.wrapping_add(0xA11CE);
    let n = 20_000usize;
    let max_obs = 48u32;

    let mut entropies = Vec::with_capacity(n);
    let mut accuracies = Vec::with_capacity(n);
    let mut rungs = [0usize; 3]; // Syntax, Semantics, Pragmatics
    let mut quads = [0usize; 4]; // Staunen, Confusion, Boredom, Wisdom
    let mut classes = [0usize; 4]; // 3-spare-bit reliability classes

    for _ in 0..n {
        let p = splitmix(&mut s); // true Bernoulli rate (ground truth)
        let n_obs = 1 + (splitmix(&mut s) * (max_obs - 1) as f64) as u32;
        let pos = (0..n_obs).filter(|_| splitmix(&mut s) < p).count() as f64;
        let f = pos / n_obs as f64;
        let c = n_obs as f64 / (n_obs as f64 + 1.0); // NARS horizon-1 confidence
        let energy = n_obs as f64 / max_obs as f64; // observation effort = energy proxy

        let h = nars_entropy(f, c);
        let e = c * (f - 0.5) + 0.5;
        let predict_true = e > 0.5;
        let m = 64;
        let hits = (0..m)
            .filter(|_| (splitmix(&mut s) < p) == predict_true)
            .count() as f64;

        entropies.push(h);
        accuracies.push(hits / m as f64);
        rungs[EntropyRung::from_entropy(h) as usize] += 1;
        quads[Quadrant::classify(h, energy) as usize] += 1;
        classes[entropy_class(h) as usize] += 1;
    }

    let rho = spearman(&entropies, &accuracies);
    let pct = |x: usize| 100.0 * x as f64 / n as f64;

    println!("VALIDATION  ρ(entropy, empirical accuracy) = {rho:.4}  (more negative ⇒ stronger reliability proxy)\n");

    println!("Rung partition (syntax = high entropy / stimulus → pragmatics = low entropy / fact):");
    println!("  Syntax     {:6.1}%  ({} edges)", pct(rungs[0]), rungs[0]);
    println!("  Semantics  {:6.1}%  ({} edges)", pct(rungs[1]), rungs[1]);
    println!("  Pragmatics {:6.1}%  ({} edges)", pct(rungs[2]), rungs[2]);

    println!("\nQuadrant partition (entropy × energy, energy = observation effort):");
    println!("  Staunen   (hi-H, lo-E) {:6.1}%", pct(quads[0]));
    println!("  Confusion (hi-H, hi-E) {:6.1}%", pct(quads[1]));
    println!("  Boredom   (lo-H, lo-E) {:6.1}%", pct(quads[2]));
    println!("  Wisdom    (lo-H, hi-E) {:6.1}%", pct(quads[3]));

    println!("\n2-bit reliability class for CausalEdge64 spare bits [63:61]:");
    println!(
        "  class 0 (Wisdom) {:5.1}% | 1 {:5.1}% | 2 {:5.1}% | 3 (Staunen) {:5.1}%",
        pct(classes[0]),
        pct(classes[1]),
        pct(classes[2]),
        pct(classes[3])
    );

    println!("\nPearl-2³ SPO decomposition (3×palette-256 indices inside CausalEdge64, no re-quant):");
    let demo: [(u8, u8, u8, u8, f64, f64, &str); 4] = [
        (10, 20, 30, 0b111, 0.97, 0.95, "SPO crystalline fact"),
        (10, 20, 30, 0b101, 0.55, 0.20, "S_O fragment, raw stimulus"),
        (44, 88, 0, 0b011, 0.90, 0.80, "SP settled, O absent"),
        (7, 0, 0, 0b001, 0.50, 0.50, "S only, maximally ambiguous"),
    ];
    for (si, pi, oi, mask, f, c, label) in demo {
        let pt = decompose_spo(si, pi, oi, mask, f, c);
        println!(
            "  {label:<28} active={:?} H={:.3} {:<11?} basin_key=0x{:08X} class={}",
            pt.active, pt.entropy, pt.rung, pt.basin_key, pt.class
        );
    }
}
