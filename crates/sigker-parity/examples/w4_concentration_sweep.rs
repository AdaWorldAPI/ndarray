//! Does depth-inf self-kernel concentration shrink like 1/sqrt(N) (a sample-
//! size effect) or plateau (a genuine heavy tail)? Measure, do not assume.
use ndarray::hpc::pillar::signature::brownian_path_d2;
use ndarray::hpc::pillar::SplitMix64;
use sigker::{signature_kernel_pde, signature_truncated};

const SEED: u64 = 0x5EED_1111_5164_A7AB;
const N_STEPS: usize = 50;

fn pool(n: usize) -> Vec<Vec<Vec<f64>>> {
    let mut rng = SplitMix64::new(SEED);
    (0..n)
        .map(|_| {
            let p = brownian_path_d2(&mut rng, N_STEPS);
            (0..=N_STEPS)
                .map(|k| vec![p[2 * k] as f64, p[2 * k + 1] as f64])
                .collect()
        })
        .collect()
}

fn stats(v: &[f64]) -> (f64, f64, f64) {
    let n = v.len();
    let h = n / 2;
    let m1 = v[..h].iter().sum::<f64>() / h as f64;
    let m2 = v[h..].iter().sum::<f64>() / (n - h) as f64;
    let mean = v.iter().sum::<f64>() / n as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    // half-mean gap, and the coefficient of variation that predicts it
    ((m1 - m2).abs() / mean, var.sqrt() / mean, mean)
}

fn main() {
    println!("depth-INFINITY (Goursat PDE)");
    println!("{:>6} {:>12} {:>10} {:>14} {:>12}", "N", "concentr", "CV", "predicted", "mean K");
    for &n in &[64usize, 128, 256, 512, 1000] {
        let p = pool(n);
        let k: Vec<f64> = p.iter().map(|x| signature_kernel_pde(x, x)).collect();
        let (c, cv, mean) = stats(&k);
        // For independent samples the expected half-mean gap ~ CV * sqrt(8/(pi*N))
        let pred = cv * (8.0 / (core::f64::consts::PI * n as f64)).sqrt();
        println!("{n:>6} {c:>12.4} {cv:>10.3} {pred:>14.4} {mean:>12.4e}");
    }
    println!("\ndepth-3 TRUNCATED (the existing battery's kernel, f64 reference)");
    println!("{:>6} {:>12} {:>10} {:>14} {:>12}", "N", "concentr", "CV", "predicted", "mean K");
    for &n in &[64usize, 1000] {
        let p = pool(n);
        let k: Vec<f64> = p
            .iter()
            .map(|x| {
                let s = signature_truncated(x, 3);
                s.levels
                    .iter()
                    .flat_map(|l| l.iter())
                    .map(|v| v * v)
                    .sum::<f64>()
            })
            .collect();
        let (c, cv, mean) = stats(&k);
        let pred = cv * (8.0 / (core::f64::consts::PI * n as f64)).sqrt();
        println!("{n:>6} {c:>12.4} {cv:>10.3} {pred:>14.4} {mean:>12.4e}");
    }
}
