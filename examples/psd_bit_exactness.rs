//! Is the Pillar-11 PSD gate bit-exact, and therefore safe to gate in CI?
//!
//! Prints a fingerprint over the RAW BIT PATTERNS of every Gram entry, plus
//! the Cholesky verdict. Two runs on one machine answer determinism; two runs
//! under different `-C target-cpu` answer whether autovectorisation/FMA moves
//! the low bits — which decides whether CI may pin exact values or only
//! verdicts. `.cargo/config.toml` pins `target-cpu=x86-64-v4`, so this is not
//! hypothetical.
use ndarray::hpc::lapack::LapackOps;
use ndarray::hpc::pillar::signature::PILLAR_11_SEED;
use ndarray::hpc::pillar::signature::{brownian_path_d2, sigker_hl, signature_d2_deg3};
use ndarray::hpc::pillar::SplitMix64;
use ndarray::Array2;

const N_STEPS: usize = 50;
const SUBSET: usize = 50;
const JITTER: f64 = 1e-4;

/// FNV-1a over raw bits — any single-bit change moves it.
fn fnv(acc: &mut u64, bits: u64) {
    for b in bits.to_le_bytes() {
        *acc ^= b as u64;
        *acc = acc.wrapping_mul(0x100_0000_01b3);
    }
}

fn main() {
    let mut rng = SplitMix64::new(PILLAR_11_SEED);
    let s: Vec<[f32; 15]> = (0..SUBSET)
        .map(|_| {
            let p = brownian_path_d2(&mut rng, N_STEPS);
            signature_d2_deg3(&p, N_STEPS + 1)
        })
        .collect();

    let mut sig_fp = 0xcbf2_9ce4_8422_2325u64;
    for v in &s {
        for x in v {
            fnv(&mut sig_fp, x.to_bits() as u64);
        }
    }

    let mut g = Array2::<f64>::zeros((SUBSET, SUBSET));
    let mut k_fp = 0xcbf2_9ce4_8422_2325u64;
    for i in 0..SUBSET {
        for j in 0..SUBSET {
            let v = sigker_hl(&s[i], &s[j]);
            fnv(&mut k_fp, v.to_bits() as u64);
            g[[i, j]] = v as f64;
        }
    }

    let mean_diag = (0..SUBSET).map(|i| g[[i, i]]).sum::<f64>() / SUBSET as f64;
    let eps = JITTER * mean_diag.abs().max(1.0);
    let mut j = g.clone();
    for i in 0..SUBSET {
        j[[i, i]] += eps;
    }
    let chol = j.cholesky();
    let mut l_fp = 0xcbf2_9ce4_8422_2325u64;
    for v in chol.factor.iter() {
        fnv(&mut l_fp, v.to_bits());
    }

    println!("signature bits  fnv1a = {sig_fp:#018x}");
    println!("kernel bits     fnv1a = {k_fp:#018x}");
    println!("cholesky L bits fnv1a = {l_fp:#018x}");
    println!("cholesky info         = {}", chol.info);
    println!("mean diag             = {mean_diag:.17e}");
}
