//! Diagnostic only: is the W1 gap a FORMULA difference or f32 accumulation?
use ndarray::hpc::pillar::signature::signature_d2_deg3;
use sigker::signature_truncated;

const NAMES: [&str; 15] = [
    "s0", "1x", "1y", "2xx", "2xy", "2yx", "2yy", "3xxx", "3xxy", "3xyx", "3xyy", "3yxx", "3yxy", "3yyx", "3yyy",
];

fn cmp(tag: &str, flat: &[f32], n: usize) {
    let hw = signature_d2_deg3(flat, n);
    let pts: Vec<Vec<f64>> = (0..n)
        .map(|k| vec![flat[2 * k] as f64, flat[2 * k + 1] as f64])
        .collect();
    let refr: Vec<f64> = signature_truncated(&pts, 3)
        .levels
        .iter()
        .flat_map(|l| l.iter().copied())
        .collect();
    println!("--- {tag} (n={n}) ---");
    for i in 0..15 {
        let (h, r) = (hw[i] as f64, refr[i]);
        let d = (h - r).abs();
        let rel = if r.abs() > 1e-12 { d / r.abs() } else { d };
        if rel > 1e-6 {
            println!("  {:>5}: hw {:+.9}  ref {:+.9}  rel {:.3e}  <== DIFFERS", NAMES[i], h, r, rel);
        }
    }
}

fn main() {
    // Single segment: closed form, zero accumulation — any gap here is FORMULA.
    cmp("one segment", &[0.0, 0.0, 1.0, 0.5], 2);
    // Two segments: Chen composition enters.
    cmp("two segments", &[0.0, 0.0, 1.0, 0.5, 1.3, -0.2], 3);
    // Three, exact small values (representable in f32) — still formula-only.
    cmp("three segments", &[0.0, 0.0, 0.5, 0.25, 0.75, -0.5, 0.25, 0.125], 4);
}
