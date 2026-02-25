//! Standalone test: 16 dims + 1-bit intensity, BF16-aligned.
//!
//! Layout: 4 × u16 words (16 nibbles) + 1 bit (intensity: causing/caused)
//! = exactly BF16-aligned. Zero waste.
//!
//! Like Quintenzirkel: relative intervals only.

use serde::Deserialize;
use std::collections::HashMap;

// ============================================================================
// BF16 structured distance (kept for comparison)
// ============================================================================

const W_SIGN: u32 = 64;
const W_EXP: u32 = 4;
const W_MANT: u32 = 1;
const EXP_GATE: u32 = 2;

#[inline(always)]
fn bf16_element_distance(a: u16, b: u16) -> u32 {
    let sa = (a >> 15) & 1;
    let sb = (b >> 15) & 1;
    let ea = ((a >> 7) & 0xFF) as u8;
    let eb = ((b >> 7) & 0xFF) as u8;

    let sign_diff = sa ^ sb;
    let exp_delta = ea.abs_diff(eb);
    let mut score = W_SIGN * sign_diff as u32 + W_EXP * exp_delta as u32;

    if sign_diff == 0 && (exp_delta as u32) <= EXP_GATE {
        let ma = a & 0x007F;
        let mb = b & 0x007F;
        score += W_MANT * (ma ^ mb).count_ones();
    }
    score
}

fn bf16_distance_u16(a: &[u16], b: &[u16]) -> u32 {
    a.iter().zip(b).map(|(&x, &y)| bf16_element_distance(x, y)).sum()
}

fn f32_to_bf16_naive(val: f32) -> u16 { (val.to_bits() >> 16) as u16 }

const BIAS_OFFSET: f32 = 2.0;
fn qualia_to_bf16(val: f32) -> u16 { ((val + BIAS_OFFSET).to_bits() >> 16) as u16 }

// ============================================================================
// NIB4: 4-bit per dimension (0..F), 16 dims + 1-bit intensity
// ============================================================================

const NIB4_LEVELS: u8 = 15;
const QUALIA_DIMS: usize = 16; // excluding resolution_hunger

struct Nib4Codebook {
    bounds: Vec<(f32, f32)>,
}

impl Nib4Codebook {
    fn from_corpus(vectors: &[Vec<f32>]) -> Self {
        let ndims = vectors[0].len();
        let mut bounds = Vec::with_capacity(ndims);
        for d in 0..ndims {
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for v in vectors {
                if v[d] < mn { mn = v[d]; }
                if v[d] > mx { mx = v[d]; }
            }
            if (mx - mn).abs() < 1e-9 { mx = mn + 1.0; }
            bounds.push((mn, mx));
        }
        Self { bounds }
    }

    fn encode_dim(&self, dim: usize, val: f32) -> u8 {
        let (mn, mx) = self.bounds[dim];
        let t = (val - mn) / (mx - mn);
        (t * NIB4_LEVELS as f32).round().clamp(0.0, NIB4_LEVELS as f32) as u8
    }

    fn decode_dim(&self, dim: usize, nib: u8) -> f32 {
        let (mn, mx) = self.bounds[dim];
        mn + (nib as f32 / NIB4_LEVELS as f32) * (mx - mn)
    }

    fn encode_vec(&self, vals: &[f32]) -> Vec<u8> {
        vals.iter().enumerate().map(|(d, &v)| self.encode_dim(d, v)).collect()
    }
}

fn nib4_distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x.abs_diff(y) as u32).sum()
}

fn nib4_to_hex(nibs: &[u8]) -> String {
    nibs.iter().map(|n| format!("{:X}", n & 0xF)).collect::<Vec<_>>().join(":")
}

/// Encode intensity as 1-bit: high shame (>= median) = caused/CMYK, low = causing/RGB
fn encode_intensity(val: f32, threshold: f32) -> bool {
    val >= threshold
}

// ============================================================================
// Reference distances
// ============================================================================

fn euclidean_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn cosine_sim_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 { return 0.0; }
    dot / (na * nb)
}

// ============================================================================
// JSON parsing
// ============================================================================

#[derive(Deserialize)]
struct QualiaItem {
    id: String,
    #[allow(dead_code)]
    label: String,
    family: String,
    vector: HashMap<String, f64>,
    #[allow(dead_code)]
    neighbors: Option<Vec<String>>,
}

/// 16 nibble dims — canonical names → JSON keys.
/// New names: glow, valence, rooting, agency, resonance, clarity, social,
///            gravity, reverence, volition, dissonance, staunen, loss,
///            optimism, friction, equilibrium
const DIMS_16_JSON: &[&str] = &[
    "brightness", "valence", "dominance", "arousal",
    "warmth", "clarity", "social", "nostalgia",
    "sacredness", "desire", "tension", "awe",
    "grief", "hope", "edge", "resolution_hunger",
];
const DIMS_16_NAMES: &[&str] = &[
    "glow", "valence", "rooting", "agency",
    "resonance", "clarity", "social", "gravity",
    "reverence", "volition", "dissonance", "staunen",
    "loss", "optimism", "friction", "equilibrium",
];

/// The +1 intensity bit: shame (excluded from nibbles, becomes causing/caused flag)
const DIM_INTENSITY: &str = "shame";

fn extract_16(item: &QualiaItem) -> Vec<f32> {
    DIMS_16_JSON.iter().map(|d| *item.vector.get(*d).unwrap_or(&0.0) as f32).collect()
}

fn extract_intensity_val(item: &QualiaItem) -> f32 {
    *item.vector.get(DIM_INTENSITY).unwrap_or(&0.0) as f32
}

/// Full 17-dim for reference distances
const DIMS_17: &[&str] = &[
    "valence", "arousal", "dominance", "warmth", "brightness",
    "tension", "clarity", "social", "nostalgia", "sacredness",
    "desire", "grief", "awe", "shame", "hope", "edge", "resolution_hunger",
];

fn extract_17(item: &QualiaItem) -> Vec<f32> {
    DIMS_17.iter().map(|d| *item.vector.get(*d).unwrap_or(&0.0) as f32).collect()
}

fn main() {
    let json_str = include_str!("qualia_219.json");
    let items: Vec<QualiaItem> = serde_json::from_str(json_str).expect("parse JSON");
    let n = items.len();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  NIB4 × 16 + 1-bit intensity    │  BF16-aligned container  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  16 dims × 4b = 64b = 4 × u16  │  +1 bit at BF16 sign    ║");
    println!("║  I=0: RGB/causing               │  I=1: CMYK/caused       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Loaded {} items, {} nibble dims + 1 intensity bit", n, QUALIA_DIMS);

    // Extract vectors
    let f32_17: Vec<Vec<f32>> = items.iter().map(|i| extract_17(i)).collect();
    let f32_16: Vec<Vec<f32>> = items.iter().map(|i| extract_16(i)).collect();
    let intensity_vals: Vec<f32> = items.iter().map(|i| extract_intensity_val(i)).collect();

    // BF16 encodings (naive + biased, 17 dims for comparison)
    let naive_vecs: Vec<Vec<u16>> = f32_17.iter().map(|v| v.iter().map(|&x| f32_to_bf16_naive(x)).collect()).collect();
    let biased_vecs: Vec<Vec<u16>> = f32_17.iter().map(|v| v.iter().map(|&x| qualia_to_bf16(x)).collect()).collect();

    // Nib4 codebook (16 dims only)
    let codebook = Nib4Codebook::from_corpus(&f32_16);
    let nib4_vecs: Vec<Vec<u8>> = f32_16.iter().map(|v| codebook.encode_vec(v)).collect();

    // Brightness threshold: median of resolution_hunger values
    let mut intensity_sorted = intensity_vals.clone();
    intensity_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let intensity_threshold = intensity_sorted[n / 2];
    let intensity_bits: Vec<bool> = intensity_vals.iter()
        .map(|&v| encode_intensity(v, intensity_threshold)).collect();

    println!("  Intensity threshold (median shame): {:.2}", intensity_threshold);
    println!("  I=1 (caused/CMYK): {} / {}", intensity_bits.iter().filter(|&&b| b).count(), n);

    // ========================================================================
    // CHECK 1: ENCODING FIDELITY
    // ========================================================================
    println!("\n=== CHECK 1: ENCODING FIDELITY ===");
    {
        let mut max_err: f32 = 0.0;
        let mut total_err: f64 = 0.0;
        let mut count = 0u64;
        for (f32v, nib4v) in f32_16.iter().zip(&nib4_vecs) {
            for (d, (&orig, &enc)) in f32v.iter().zip(nib4v.iter()).enumerate() {
                let decoded = codebook.decode_dim(d, enc);
                let err = (orig - decoded).abs();
                if err > max_err { max_err = err; }
                total_err += err as f64;
                count += 1;
            }
        }
        println!("  Nib4 (16 dims): max_err={:.4}  mean_err={:.4}", max_err, total_err / count as f64);
    }

    // Codebook bounds
    println!("\n  Codebook (16 nibble dims):");
    for (d, dim) in DIMS_16_NAMES.iter().enumerate() {
        let (mn, mx) = codebook.bounds[d];
        println!("    {:<14} [{:>6.2}, {:>5.2}]  step={:.3}", dim, mn, mx, (mx - mn) / 15.0);
    }

    // Sample encodings
    println!("\n  Sample F:F encodings:");
    for i in [0, 1, 50, 100, 150, 218].iter().copied().filter(|&i| i < n) {
        let hex = nib4_to_hex(&nib4_vecs[i]);
        let b = if intensity_bits[i] { "I=1" } else { "I=0" };
        println!("    {:<35} → {} [{}]", items[i].id, hex, b);
    }

    // ========================================================================
    // CHECK 2: DISTANCE COMPARISON ON KNOWN PAIRS
    // ========================================================================
    println!("\n=== CHECK 2: DISTANCE COMPARISON ===");

    let test_pairs = [
        ("devotion_stay_when_ugly", "devotion_gentle_guardian", "same-fam"),
        ("devotion_stay_when_ugly", "devotion_hurt_but_here", "same-fam"),
        ("grief_private_weight", "grief_sacred_mourning", "same-fam"),
        ("anger_feral_burst", "stillness_safe_void", "opposite"),
        ("desire_want_with_teeth", "surrender_soft_collapse", "opposite"),
        ("endurance_ritual_breathing", "devotion_worship_without_church", "neighbor"),
        ("letting_go_complete", "anger_feral_burst", "extreme"),
        ("presence_enough", "loss_control_overwhelm", "extreme"),
    ];

    let id_map: HashMap<&str, usize> = items.iter().enumerate()
        .map(|(i, item)| (item.id.as_str(), i)).collect();

    println!("  {:<12} {:<22} ↔ {:<22} {:>5} {:>5} {:>6} {:>4} {:>3}",
        "Type", "A", "B", "Naive", "Bias", "Euclid", "Nib4", "ΔI");

    for (a_id, b_id, label) in &test_pairs {
        if let (Some(&ai), Some(&bi)) = (id_map.get(a_id), id_map.get(b_id)) {
            let dn = bf16_distance_u16(&naive_vecs[ai], &naive_vecs[bi]);
            let db = bf16_distance_u16(&biased_vecs[ai], &biased_vecs[bi]);
            let euc = euclidean_f32(&f32_17[ai], &f32_17[bi]);
            let d4 = nib4_distance(&nib4_vecs[ai], &nib4_vecs[bi]);
            let b_diff = if intensity_bits[ai] != intensity_bits[bi] { "!" } else { "=" };

            let short_a: String = a_id.chars().take(22).collect();
            let short_b: String = b_id.chars().take(22).collect();
            println!("  {:<12} {:<22} ↔ {:<22} {:>5} {:>5} {:>6.3} {:>4} {:>3}",
                label, short_a, short_b, dn, db, euc, d4, b_diff);
        }
    }

    // Per-dim breakdown for one pair
    if let (Some(&ai), Some(&bi)) = (id_map.get("devotion_stay_when_ugly"), id_map.get("anger_feral_burst")) {
        println!("\n  Per-dim: {} ↔ {}", "devotion_stay_when_ugly", "anger_feral_burst");
        println!("    {:>14}  {:>3} {:>3}  {:>3}", "Dimension", "A", "B", "|Δ|");
        for (d, dim) in DIMS_16_NAMES.iter().enumerate() {
            let na = nib4_vecs[ai][d];
            let nb = nib4_vecs[bi][d];
            println!("    {:>14}  {:>3X} {:>3X}  {:>3}", dim, na, nb, na.abs_diff(nb));
        }
        let total = nib4_distance(&nib4_vecs[ai], &nib4_vecs[bi]);
        let b_diff = intensity_bits[ai] != intensity_bits[bi];
        println!("    {:>14}         {:>3}  + intensity={}", "TOTAL", total, if b_diff { "FLIP" } else { "same" });
        println!("    Normalized: {}/0xF0 = {:.3}", total, total as f32 / 240.0);
    }

    // ========================================================================
    // CHECK 3: RANKING COHERENCE (the critical test)
    // ========================================================================
    println!("\n=== CHECK 3: RANKING COHERENCE (top-5 neighbors) ===");

    let k = 5;
    let mut jaccard_naive = 0.0f64;
    let mut jaccard_biased = 0.0f64;
    let mut jaccard_nib4 = 0.0f64;
    let mut jaccard_nib4b = 0.0f64; // nib4 + intensity penalty
    let mut fam_naive = 0u64;
    let mut fam_biased = 0u64;
    let mut fam_nib4 = 0u64;
    let mut fam_nib4b = 0u64;
    let mut fam_euc = 0u64;
    let mut fam_cos = 0u64;
    let mut total_neighbors = 0u64;

    let bright_penalty: u32 = 16; // one full dimension worth

    for i in 0..n {
        // Naive BF16 (17 dims)
        let mut naive_dists: Vec<(usize, u32)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, bf16_distance_u16(&naive_vecs[i], &naive_vecs[j]))).collect();
        naive_dists.sort_by_key(|&(_, d)| d);

        // Biased BF16 (17 dims)
        let mut biased_dists: Vec<(usize, u32)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, bf16_distance_u16(&biased_vecs[i], &biased_vecs[j]))).collect();
        biased_dists.sort_by_key(|&(_, d)| d);

        // Nib4 (16 dims, no intensity)
        let mut nib4_dists: Vec<(usize, u32)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, nib4_distance(&nib4_vecs[i], &nib4_vecs[j]))).collect();
        nib4_dists.sort_by_key(|&(_, d)| d);

        // Nib4 + intensity penalty
        let mut nib4b_dists: Vec<(usize, u32)> = (0..n).filter(|&j| j != i)
            .map(|j| {
                let mut d = nib4_distance(&nib4_vecs[i], &nib4_vecs[j]);
                if intensity_bits[i] != intensity_bits[j] { d += bright_penalty; }
                (j, d)
            }).collect();
        nib4b_dists.sort_by_key(|&(_, d)| d);

        // Euclidean (17 dims)
        let mut euc_dists: Vec<(usize, f32)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, euclidean_f32(&f32_17[i], &f32_17[j]))).collect();
        euc_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Cosine (17 dims)
        let mut cos_dists: Vec<(usize, f32)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, cosine_sim_f32(&f32_17[i], &f32_17[j]))).collect();
        cos_dists.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let naive_top: Vec<usize> = naive_dists.iter().take(k).map(|&(j, _)| j).collect();
        let biased_top: Vec<usize> = biased_dists.iter().take(k).map(|&(j, _)| j).collect();
        let nib4_top: Vec<usize> = nib4_dists.iter().take(k).map(|&(j, _)| j).collect();
        let nib4b_top: Vec<usize> = nib4b_dists.iter().take(k).map(|&(j, _)| j).collect();
        let euc_top: Vec<usize> = euc_dists.iter().take(k).map(|&(j, _)| j).collect();
        let cos_top: Vec<usize> = cos_dists.iter().take(k).map(|&(j, _)| j).collect();

        // Jaccard vs Euclidean
        let j_n = naive_top.iter().filter(|x| euc_top.contains(x)).count();
        let j_b = biased_top.iter().filter(|x| euc_top.contains(x)).count();
        let j_4 = nib4_top.iter().filter(|x| euc_top.contains(x)).count();
        let j_4b = nib4b_top.iter().filter(|x| euc_top.contains(x)).count();
        jaccard_naive += j_n as f64 / (2 * k - j_n) as f64;
        jaccard_biased += j_b as f64 / (2 * k - j_b) as f64;
        jaccard_nib4 += j_4 as f64 / (2 * k - j_4) as f64;
        jaccard_nib4b += j_4b as f64 / (2 * k - j_4b) as f64;

        let family = &items[i].family;
        for idx in 0..k {
            total_neighbors += 1;
            if items[naive_top[idx]].family == *family { fam_naive += 1; }
            if items[biased_top[idx]].family == *family { fam_biased += 1; }
            if items[nib4_top[idx]].family == *family { fam_nib4 += 1; }
            if items[nib4b_top[idx]].family == *family { fam_nib4b += 1; }
            if items[euc_top[idx]].family == *family { fam_euc += 1; }
            if items[cos_top[idx]].family == *family { fam_cos += 1; }
        }
    }

    println!("  {:>30} {:>10} {:>10} {:>10} {:>10}", "", "Naive BF16", "Bias BF16", "Nib4×16", "Nib4+B");
    println!("  {:>30} {:>10.4} {:>10.4} {:>10.4} {:>10.4}", "Jaccard vs Euclid:",
        jaccard_naive / n as f64, jaccard_biased / n as f64, jaccard_nib4 / n as f64, jaccard_nib4b / n as f64);
    println!("  {:>30} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.1}%", "Family coherence:",
        100.0 * fam_naive as f64 / total_neighbors as f64,
        100.0 * fam_biased as f64 / total_neighbors as f64,
        100.0 * fam_nib4 as f64 / total_neighbors as f64,
        100.0 * fam_nib4b as f64 / total_neighbors as f64);
    println!();
    println!("  {:>30} {:>9.1}%", "Euclid family coherence:", 100.0 * fam_euc as f64 / total_neighbors as f64);
    println!("  {:>30} {:>9.1}%", "Cosine family coherence:", 100.0 * fam_cos as f64 / total_neighbors as f64);

    // ========================================================================
    // CHECK 4: DISTANCE DISTRIBUTION
    // ========================================================================
    println!("\n=== CHECK 4: NIB4 DISTANCE DISTRIBUTION ===");

    let mut same_dists = Vec::new();
    let mut diff_dists = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = nib4_distance(&nib4_vecs[i], &nib4_vecs[j]);
            if items[i].family == items[j].family { same_dists.push(d); }
            else { diff_dists.push(d); }
        }
    }

    let mean = |v: &[u32]| -> f64 { v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64 };
    let median = |v: &mut Vec<u32>| -> u32 { v.sort(); v[v.len() / 2] };
    let mut same_s = same_dists.clone();
    let mut diff_s = diff_dists.clone();

    println!("  {:>15} {:>6} {:>5} {:>6} {:>6} {:>5}", "", "Count", "Min", "Med", "Mean", "Max");
    println!("  {:>15} {:>6} {:>5} {:>6} {:>6.1} {:>5}", "Same family", same_dists.len(),
        same_dists.iter().min().unwrap(), median(&mut same_s), mean(&same_dists), same_dists.iter().max().unwrap());
    println!("  {:>15} {:>6} {:>5} {:>6} {:>6.1} {:>5}", "Diff family", diff_dists.len(),
        diff_dists.iter().min().unwrap(), median(&mut diff_s), mean(&diff_dists), diff_dists.iter().max().unwrap());
    println!("  Separation ratio: {:.3} (higher = better)", mean(&diff_dists) / mean(&same_dists));

    // ========================================================================
    // CHECK 5: SPEED
    // ========================================================================
    println!("\n=== CHECK 5: SPEED ===");

    let start = std::time::Instant::now();
    let mut cs = 0u64;
    for i in 0..n { for j in 0..n { cs += nib4_distance(&nib4_vecs[i], &nib4_vecs[j]) as u64; } }
    let nib4_t = start.elapsed();

    let start = std::time::Instant::now();
    let mut cb = 0u64;
    for i in 0..n { for j in 0..n { cb += bf16_distance_u16(&biased_vecs[i], &biased_vecs[j]) as u64; } }
    let bf16_t = start.elapsed();

    let pairs = (n * n) as f64;
    println!("  Nib4 ×16:      {:?} ({:.0} ns/pair)", nib4_t, nib4_t.as_nanos() as f64 / pairs);
    println!("  Biased BF16:   {:?} ({:.0} ns/pair)", bf16_t, bf16_t.as_nanos() as f64 / pairs);
    println!("  Speedup: {:.1}x", bf16_t.as_nanos() as f64 / nib4_t.as_nanos() as f64);

    // ========================================================================
    // CHECK 6: PSYCHOMETRIC ITEM ANALYSIS
    // ========================================================================
    println!("\n=== CHECK 6: PSYCHOMETRIC ITEM ANALYSIS ===");

    // 6a. Item Discrimination (Cohen's d per dimension)
    // For each dimension: compute mean abs_diff for same-family vs diff-family pairs.
    // Cohen's d = (M_diff - M_same) / pooled_SD
    println!("\n  --- 6a. Item Discrimination (Cohen's d) ---");
    println!("  Higher d = better separation between families on that dimension.\n");
    println!("  {:<14}  {:>6} {:>6}  {:>6} {:>6}  {:>6}",
        "Dimension", "M_same", "M_diff", "SD_s", "SD_d", "d");

    let mut dim_discriminations: Vec<(usize, f64)> = Vec::new();

    for d in 0..QUALIA_DIMS {
        let mut same_deltas: Vec<f64> = Vec::new();
        let mut diff_deltas: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let delta = nib4_vecs[i][d].abs_diff(nib4_vecs[j][d]) as f64;
                if items[i].family == items[j].family {
                    same_deltas.push(delta);
                } else {
                    diff_deltas.push(delta);
                }
            }
        }

        let mean_s = same_deltas.iter().sum::<f64>() / same_deltas.len() as f64;
        let mean_d = diff_deltas.iter().sum::<f64>() / diff_deltas.len() as f64;
        let var_s = same_deltas.iter().map(|x| (x - mean_s).powi(2)).sum::<f64>() / same_deltas.len() as f64;
        let var_d = diff_deltas.iter().map(|x| (x - mean_d).powi(2)).sum::<f64>() / diff_deltas.len() as f64;
        let sd_pooled = ((var_s + var_d) / 2.0).sqrt();
        let cohens_d = if sd_pooled > 1e-9 { (mean_d - mean_s) / sd_pooled } else { 0.0 };

        dim_discriminations.push((d, cohens_d));

        println!("  {:<14}  {:>6.2} {:>6.2}  {:>6.2} {:>6.2}  {:>6.3}",
            DIMS_16_NAMES[d], mean_s, mean_d, var_s.sqrt(), var_d.sqrt(), cohens_d);
    }

    dim_discriminations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!("\n  Worst discriminators:");
    for &(d, cd) in dim_discriminations.iter().take(3) {
        println!("    {:<14} d={:.3} — weakest separation", DIMS_16_NAMES[d], cd);
    }
    println!("  Best discriminators:");
    for &(d, cd) in dim_discriminations.iter().rev().take(3) {
        println!("    {:<14} d={:.3} — strongest separation", DIMS_16_NAMES[d], cd);
    }

    // 6b. Item-Total Correlation (Pearson r: dim distance vs total distance)
    println!("\n  --- 6b. Item-Total Correlation (Pearson r) ---");
    println!("  How much each dimension tracks the total Nib4 distance.\n");

    let mut dim_correlations: Vec<(usize, f64)> = Vec::new();

    for d in 0..QUALIA_DIMS {
        let mut dim_vals: Vec<f64> = Vec::new();
        let mut total_vals: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                dim_vals.push(nib4_vecs[i][d].abs_diff(nib4_vecs[j][d]) as f64);
                total_vals.push(nib4_distance(&nib4_vecs[i], &nib4_vecs[j]) as f64);
            }
        }
        let n_pairs = dim_vals.len() as f64;
        let mean_x = dim_vals.iter().sum::<f64>() / n_pairs;
        let mean_y = total_vals.iter().sum::<f64>() / n_pairs;
        let cov = dim_vals.iter().zip(&total_vals).map(|(x, y)| (x - mean_x) * (y - mean_y)).sum::<f64>() / n_pairs;
        let sd_x = (dim_vals.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>() / n_pairs).sqrt();
        let sd_y = (total_vals.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>() / n_pairs).sqrt();
        let r = if sd_x > 1e-9 && sd_y > 1e-9 { cov / (sd_x * sd_y) } else { 0.0 };

        dim_correlations.push((d, r));
    }

    dim_correlations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!("  {:<14}  {:>8}", "Dimension", "r(item,total)");
    for &(d, r) in &dim_correlations {
        let flag = if r < 0.3 { " ⚠ LOW" } else if r > 0.6 { " ✓" } else { "" };
        println!("  {:<14}  {:>8.4}{}", DIMS_16_NAMES[d], r, flag);
    }

    // 6c. Cronbach's Alpha (internal consistency)
    println!("\n  --- 6c. Cronbach's Alpha (internal consistency) ---");
    {
        // Alpha = (k/(k-1)) * (1 - sum(var_i) / var_total)
        // Treat each dim's abs_diff as an "item score" for each pair
        let num_pairs = n * (n - 1) / 2;
        let k = QUALIA_DIMS as f64;

        // Compute total distances for all pairs
        let mut total_scores: Vec<f64> = Vec::with_capacity(num_pairs);
        let mut dim_scores: Vec<Vec<f64>> = vec![Vec::with_capacity(num_pairs); QUALIA_DIMS];
        for i in 0..n {
            for j in (i + 1)..n {
                total_scores.push(nib4_distance(&nib4_vecs[i], &nib4_vecs[j]) as f64);
                for d in 0..QUALIA_DIMS {
                    dim_scores[d].push(nib4_vecs[i][d].abs_diff(nib4_vecs[j][d]) as f64);
                }
            }
        }

        let var_total = {
            let m = total_scores.iter().sum::<f64>() / total_scores.len() as f64;
            total_scores.iter().map(|x| (x - m).powi(2)).sum::<f64>() / total_scores.len() as f64
        };

        let mut sum_var_items = 0.0;
        let mut dim_vars: Vec<(usize, f64)> = Vec::new();
        for d in 0..QUALIA_DIMS {
            let m = dim_scores[d].iter().sum::<f64>() / dim_scores[d].len() as f64;
            let v = dim_scores[d].iter().map(|x| (x - m).powi(2)).sum::<f64>() / dim_scores[d].len() as f64;
            sum_var_items += v;
            dim_vars.push((d, v));
        }

        let alpha = (k / (k - 1.0)) * (1.0 - sum_var_items / var_total);
        println!("  Cronbach's α = {:.4}  (k={} dimensions, {} pairs)", alpha, QUALIA_DIMS, num_pairs);

        if alpha >= 0.7 { println!("  Interpretation: acceptable (≥0.70)"); }
        else if alpha >= 0.6 { println!("  Interpretation: questionable (0.60-0.69)"); }
        else { println!("  Interpretation: poor (<0.60) — dimensions may be too heterogeneous"); }

        // Alpha-if-removed
        println!("\n  Alpha-if-removed (drop one dimension):");
        println!("  {:<14}  {:>8}  {:>8}", "Drop dim", "α′", "Δα");
        for d in 0..QUALIA_DIMS {
            let k2 = k - 1.0;
            let sum_var2 = sum_var_items - dim_vars[d].1;
            // Need var of total minus this dim
            let mut total2: Vec<f64> = Vec::with_capacity(num_pairs);
            let mut idx = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    total2.push(total_scores[idx] - dim_scores[d][idx]);
                    idx += 1;
                }
            }
            let m2 = total2.iter().sum::<f64>() / total2.len() as f64;
            let var_total2 = total2.iter().map(|x| (x - m2).powi(2)).sum::<f64>() / total2.len() as f64;
            let alpha2 = (k2 / (k2 - 1.0)) * (1.0 - sum_var2 / var_total2);
            let delta = alpha2 - alpha;
            let flag = if delta > 0.005 { " ← remove improves α" } else { "" };
            println!("  {:<14}  {:>8.4}  {:>+8.4}{}", DIMS_16_NAMES[d], alpha2, delta, flag);
        }
    }

    // 6d. Split-Half Reliability
    println!("\n  --- 6d. Split-Half Reliability ---");
    {
        // Split items into odd/even indexed, compute rankings on each half,
        // measure correlation of per-item family-coherence scores
        let mut scores_odd: Vec<f64> = Vec::with_capacity(n);
        let mut scores_even: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let mut odd_same = 0u32;
            let mut even_same = 0u32;
            let mut odd_dists: Vec<(usize, u32)> = Vec::new();
            let mut even_dists: Vec<(usize, u32)> = Vec::new();
            for j in 0..n {
                if j == i { continue; }
                let mut d_odd = 0u32;
                let mut d_even = 0u32;
                for d in (0..QUALIA_DIMS).step_by(2) {
                    d_even += nib4_vecs[i][d].abs_diff(nib4_vecs[j][d]) as u32;
                }
                for d in (1..QUALIA_DIMS).step_by(2) {
                    d_odd += nib4_vecs[i][d].abs_diff(nib4_vecs[j][d]) as u32;
                }
                odd_dists.push((j, d_odd));
                even_dists.push((j, d_even));
            }
            odd_dists.sort_by_key(|&(_, d)| d);
            even_dists.sort_by_key(|&(_, d)| d);

            let fam = &items[i].family;
            for idx in 0..k {
                if items[odd_dists[idx].0].family == *fam { odd_same += 1; }
                if items[even_dists[idx].0].family == *fam { even_same += 1; }
            }
            scores_odd.push(odd_same as f64 / k as f64);
            scores_even.push(even_same as f64 / k as f64);
        }

        let n_f = n as f64;
        let m_odd = scores_odd.iter().sum::<f64>() / n_f;
        let m_even = scores_even.iter().sum::<f64>() / n_f;
        let cov = scores_odd.iter().zip(&scores_even).map(|(a, b)| (a - m_odd) * (b - m_even)).sum::<f64>() / n_f;
        let sd_o = (scores_odd.iter().map(|x| (x - m_odd).powi(2)).sum::<f64>() / n_f).sqrt();
        let sd_e = (scores_even.iter().map(|x| (x - m_even).powi(2)).sum::<f64>() / n_f).sqrt();
        let r_half = if sd_o > 1e-9 && sd_e > 1e-9 { cov / (sd_o * sd_e) } else { 0.0 };
        // Spearman-Brown correction: r_full = 2*r_half / (1 + r_half)
        let r_full = 2.0 * r_half / (1.0 + r_half);

        println!("  Split: even dims (8) vs odd dims (8)");
        println!("  Half-test r     = {:.4}", r_half);
        println!("  Spearman-Brown  = {:.4}  (corrected full-test reliability)", r_full);
        println!("  Even-half coherence: {:.1}%", 100.0 * m_even);
        println!("  Odd-half coherence:  {:.1}%", 100.0 * m_odd);
    }

    // ========================================================================
    // CHECK 7: OUTLIER ANALYSIS — worst-distinguished items
    // ========================================================================
    println!("\n=== CHECK 7: OUTLIER ANALYSIS ===");

    // 7a. Per-item: what fraction of top-5 neighbors are same-family?
    let mut item_scores: Vec<(usize, f64, usize, String)> = Vec::new(); // (idx, score, same_count, top_families)
    for i in 0..n {
        let mut dists: Vec<(usize, u32)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, nib4_distance(&nib4_vecs[i], &nib4_vecs[j]))).collect();
        dists.sort_by_key(|&(_, d)| d);

        let fam = &items[i].family;
        let mut same = 0usize;
        let mut neighbor_fams: Vec<&str> = Vec::new();
        for idx in 0..k {
            let j = dists[idx].0;
            if items[j].family == *fam { same += 1; }
            neighbor_fams.push(&items[j].family);
        }
        let top_fams = neighbor_fams.join(", ");
        item_scores.push((i, same as f64 / k as f64, same, top_fams));
    }

    // Sort by score ascending (worst first)
    item_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("\n  --- 7a. Worst-distinguished items (0/5 or 1/5 same-family neighbors) ---\n");
    println!("  {:<40} {:>5} {:>7}  {}", "Item", "Score", "Family", "Top-5 neighbor families");
    let mut outlier_count = 0;
    for &(idx, score, same, ref fams) in &item_scores {
        if same > 1 { break; }
        outlier_count += 1;
        println!("  {:<40} {}/{:<4} {:>7}  [{}]",
            &items[idx].id, same, k, &items[idx].family, fams);
    }
    if outlier_count == 0 { println!("  (no items with ≤1/5 same-family neighbors)"); }

    // 7b. Per-family coherence breakdown
    println!("\n  --- 7b. Per-family coherence ---\n");

    let mut families: Vec<String> = items.iter().map(|i| i.family.clone()).collect();
    families.sort();
    families.dedup();

    println!("  {:<20} {:>4} {:>8} {:>12}", "Family", "N", "Coherence", "Mean dist");

    let mut family_stats: Vec<(String, f64, f64, usize)> = Vec::new(); // (name, coherence, mean_intra_dist, count)

    for fam in &families {
        let fam_indices: Vec<usize> = items.iter().enumerate()
            .filter(|(_, it)| it.family == *fam).map(|(i, _)| i).collect();
        let fam_n = fam_indices.len();

        if fam_n < 2 { continue; }

        // Family coherence: for each member, what fraction of top-k are same family?
        let mut fam_coherence = 0.0;
        for &i in &fam_indices {
            let &(_, score, _, _) = item_scores.iter().find(|&&(idx, _, _, _)| idx == i).unwrap();
            fam_coherence += score;
        }
        fam_coherence /= fam_n as f64;

        // Mean intra-family distance
        let mut intra_sum = 0u64;
        let mut intra_count = 0u64;
        for (a_idx, &i) in fam_indices.iter().enumerate() {
            for &j in fam_indices.iter().skip(a_idx + 1) {
                intra_sum += nib4_distance(&nib4_vecs[i], &nib4_vecs[j]) as u64;
                intra_count += 1;
            }
        }
        let mean_intra = if intra_count > 0 { intra_sum as f64 / intra_count as f64 } else { 0.0 };

        family_stats.push((fam.clone(), fam_coherence, mean_intra, fam_n));
    }

    family_stats.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    for (fam, coh, dist, count) in &family_stats {
        let flag = if *coh < 0.2 { " ⚠ POOR" } else { "" };
        println!("  {:<20} {:>4} {:>7.1}% {:>12.1}{}",
            fam, count, coh * 100.0, dist, flag);
    }

    // 7c. Confusion matrix: which families get confused with which?
    println!("\n  --- 7c. Most common confusions (wrong-family neighbors) ---\n");

    let mut confusions: HashMap<(String, String), usize> = HashMap::new();
    for i in 0..n {
        let mut dists: Vec<(usize, u32)> = (0..n).filter(|&j| j != i)
            .map(|j| (j, nib4_distance(&nib4_vecs[i], &nib4_vecs[j]))).collect();
        dists.sort_by_key(|&(_, d)| d);

        let fam = &items[i].family;
        for idx in 0..k {
            let j = dists[idx].0;
            if items[j].family != *fam {
                let pair = if *fam < items[j].family {
                    (fam.clone(), items[j].family.clone())
                } else {
                    (items[j].family.clone(), fam.clone())
                };
                *confusions.entry(pair).or_insert(0) += 1;
            }
        }
    }

    let mut conf_vec: Vec<((String, String), usize)> = confusions.into_iter().collect();
    conf_vec.sort_by(|a, b| b.1.cmp(&a.1));

    println!("  {:<20} ↔ {:<20} {:>6}", "Family A", "Family B", "Confusions");
    for ((a, b), count) in conf_vec.iter().take(15) {
        println!("  {:<20} ↔ {:<20} {:>6}", a, b, count);
    }

    // 7d. Per-dimension: which dim contributes most to confusion?
    println!("\n  --- 7d. Per-dim confusion contribution ---");
    println!("  For the most-confused pair: which dims are too similar?\n");

    if let Some(((fam_a, fam_b), _)) = conf_vec.first() {
        let idx_a: Vec<usize> = items.iter().enumerate()
            .filter(|(_, it)| it.family == *fam_a).map(|(i, _)| i).collect();
        let idx_b: Vec<usize> = items.iter().enumerate()
            .filter(|(_, it)| it.family == *fam_b).map(|(i, _)| i).collect();

        println!("  Most confused pair: {} ↔ {}\n", fam_a, fam_b);
        println!("  {:<14}  {:>6} {:>6} {:>6}  {:>6}", "Dimension", "M_A", "M_B", "|Δ|", "Overlap");

        for d in 0..QUALIA_DIMS {
            let mean_a: f64 = idx_a.iter().map(|&i| nib4_vecs[i][d] as f64).sum::<f64>() / idx_a.len() as f64;
            let mean_b: f64 = idx_b.iter().map(|&i| nib4_vecs[i][d] as f64).sum::<f64>() / idx_b.len() as f64;
            let sd_a: f64 = (idx_a.iter().map(|&i| (nib4_vecs[i][d] as f64 - mean_a).powi(2)).sum::<f64>() / idx_a.len() as f64).sqrt();
            let sd_b: f64 = (idx_b.iter().map(|&i| (nib4_vecs[i][d] as f64 - mean_b).powi(2)).sum::<f64>() / idx_b.len() as f64).sqrt();
            let delta = (mean_a - mean_b).abs();
            // Overlap: % of combined SD that eats into the gap
            let overlap = if delta > 1e-9 { ((sd_a + sd_b) / delta).min(9.99) } else { 9.99 };
            let flag = if overlap > 2.0 { " ⚠" } else { "" };
            println!("  {:<14}  {:>6.1} {:>6.1} {:>6.2}  {:>5.2}x{}",
                DIMS_16_NAMES[d], mean_a, mean_b, delta, overlap, flag);
        }
    }

    // ========================================================================
    // CHECK 8: VALIDITY — does each item sit in its family's meaning space?
    // ========================================================================
    println!("\n=== CHECK 8: CONSTRUCT VALIDITY (centroid analysis) ===");

    // Compute family centroids in nib4 space
    let mut centroids: HashMap<String, Vec<f64>> = HashMap::new();
    let mut fam_counts: HashMap<String, usize> = HashMap::new();
    for i in 0..n {
        let fam = items[i].family.clone();
        let entry = centroids.entry(fam.clone()).or_insert_with(|| vec![0.0; QUALIA_DIMS]);
        for d in 0..QUALIA_DIMS {
            entry[d] += nib4_vecs[i][d] as f64;
        }
        *fam_counts.entry(fam).or_insert(0) += 1;
    }
    for (fam, centroid) in centroids.iter_mut() {
        let count = fam_counts[fam] as f64;
        for d in 0..QUALIA_DIMS {
            centroid[d] /= count;
        }
    }

    // Per-family SD (for z-score)
    let mut fam_sds: HashMap<String, Vec<f64>> = HashMap::new();
    for fam in centroids.keys() {
        let mut sds = vec![0.0f64; QUALIA_DIMS];
        let members: Vec<usize> = items.iter().enumerate()
            .filter(|(_, it)| it.family == *fam).map(|(i, _)| i).collect();
        let centroid = &centroids[fam];
        for d in 0..QUALIA_DIMS {
            let var: f64 = members.iter().map(|&i| (nib4_vecs[i][d] as f64 - centroid[d]).powi(2)).sum::<f64>() / members.len() as f64;
            sds[d] = var.sqrt().max(0.5); // floor at 0.5 to avoid div-by-zero
        }
        fam_sds.insert(fam.clone(), sds);
    }

    // 8a. For each item: distance to own centroid vs nearest foreign centroid
    println!("\n  --- 8a. Items closer to foreign centroid than own ---\n");
    println!("  {:<40} {:>7} {:>8} {:>8} {:>8} {:<12}",
        "Item", "Family", "d(own)", "d(near)", "Ratio", "Nearest");

    let nib4_centroid_dist = |nibs: &[u8], centroid: &[f64]| -> f64 {
        nibs.iter().enumerate().map(|(d, &v)| (v as f64 - centroid[d]).abs()).sum()
    };

    let mut misplaced = Vec::new();

    for i in 0..n {
        let own_fam = &items[i].family;
        let d_own = nib4_centroid_dist(&nib4_vecs[i], &centroids[own_fam]);

        let mut nearest_fam = String::new();
        let mut d_nearest = f64::INFINITY;
        for (fam, centroid) in &centroids {
            if fam == own_fam { continue; }
            let d = nib4_centroid_dist(&nib4_vecs[i], centroid);
            if d < d_nearest {
                d_nearest = d;
                nearest_fam = fam.clone();
            }
        }

        let ratio = d_own / d_nearest; // >1.0 means closer to foreign
        if ratio > 0.85 {
            misplaced.push((i, d_own, d_nearest, ratio, nearest_fam.clone()));
        }
    }

    misplaced.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    for &(i, d_own, d_near, ratio, ref nearest) in misplaced.iter().take(20) {
        let flag = if ratio > 1.0 { " ⚠ MISPLACED" } else { "" };
        println!("  {:<40} {:>7} {:>8.1} {:>8.1} {:>8.3} {:<12}{}",
            items[i].id, items[i].family, d_own, d_near, ratio, nearest, flag);
    }
    println!("  Total items with ratio > 0.85: {} / {}", misplaced.len(), n);
    println!("  Total MISPLACED (ratio > 1.0): {} / {}", misplaced.iter().filter(|x| x.3 > 1.0).count(), n);

    // 8b. Per-item z-score from family centroid (Mahalanobis-like)
    println!("\n  --- 8b. Items with extreme z-scores (>2σ from family centroid) ---\n");

    let mut z_outliers: Vec<(usize, f64, usize)> = Vec::new(); // (item_idx, max_z, dim_of_max_z)
    for i in 0..n {
        let fam = &items[i].family;
        let centroid = &centroids[fam];
        let sds = &fam_sds[fam];
        let mut max_z = 0.0f64;
        let mut max_d = 0usize;
        for d in 0..QUALIA_DIMS {
            let z = (nib4_vecs[i][d] as f64 - centroid[d]).abs() / sds[d];
            if z > max_z { max_z = z; max_d = d; }
        }
        if max_z > 2.0 {
            z_outliers.push((i, max_z, max_d));
        }
    }

    z_outliers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("  {:<40} {:>7} {:>6} {:>12}",
        "Item", "Family", "Max z", "Deviant dim");
    for &(idx, z, d) in z_outliers.iter().take(15) {
        println!("  {:<40} {:>7} {:>6.2} {:>12}",
            items[idx].id, items[idx].family, z, DIMS_16_NAMES[d]);
    }
    println!("  Items with z > 2.0: {} / {}", z_outliers.len(), n);

    // 8c. Criterion validity: correlation of Nib4 ranking with Euclidean ranking
    println!("\n  --- 8c. Criterion Validity (Spearman rank correlation vs Euclidean) ---");
    {
        let mut rank_corrs: Vec<f64> = Vec::new();
        for i in 0..n {
            let mut nib_dists: Vec<(usize, u32)> = (0..n).filter(|&j| j != i)
                .map(|j| (j, nib4_distance(&nib4_vecs[i], &nib4_vecs[j]))).collect();
            nib_dists.sort_by_key(|&(_, d)| d);

            let mut euc_dists: Vec<(usize, f32)> = (0..n).filter(|&j| j != i)
                .map(|j| (j, euclidean_f32(&f32_17[i], &f32_17[j]))).collect();
            euc_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Assign ranks
            let m = n - 1;
            let mut nib_rank = vec![0.0f64; n];
            let mut euc_rank = vec![0.0f64; n];
            for (rank, &(j, _)) in nib_dists.iter().enumerate() {
                nib_rank[j] = rank as f64;
            }
            for (rank, &(j, _)) in euc_dists.iter().enumerate() {
                euc_rank[j] = rank as f64;
            }

            // Spearman: 1 - 6*sum(d^2) / (n*(n^2-1))
            let sum_d2: f64 = (0..n).filter(|&j| j != i)
                .map(|j| (nib_rank[j] - euc_rank[j]).powi(2)).sum();
            let rho = 1.0 - 6.0 * sum_d2 / (m as f64 * (m as f64 * m as f64 - 1.0));
            rank_corrs.push(rho);
        }

        let mean_rho = rank_corrs.iter().sum::<f64>() / rank_corrs.len() as f64;
        let min_rho = rank_corrs.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_rho = rank_corrs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        rank_corrs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med_rho = rank_corrs[rank_corrs.len() / 2];

        println!("  Spearman ρ (Nib4 rank vs Euclidean rank, per item):");
        println!("    Mean: {:.4}  Median: {:.4}  Min: {:.4}  Max: {:.4}", mean_rho, med_rho, min_rho, max_rho);
        if mean_rho >= 0.7 { println!("    Interpretation: strong criterion validity"); }
        else if mean_rho >= 0.5 { println!("    Interpretation: moderate criterion validity"); }
        else { println!("    Interpretation: weak criterion validity — Nib4 ranking diverges from Euclidean"); }
    }

    // ========================================================================
    // CHECK 9: DIMENSIONAL INDEPENDENCE — field modes, not word drift
    // ========================================================================
    println!("\n=== CHECK 9: DIMENSIONAL INDEPENDENCE ===");
    println!("  Are these dynamical field parameters, or redundant word labels?\n");

    // 9a. Inter-dimension Pearson correlation matrix (on raw f32 values)
    println!("  --- 9a. Inter-dimension correlation matrix (f32 corpus) ---\n");

    // Compute means and SDs per dimension
    let mut dim_means = vec![0.0f64; QUALIA_DIMS];
    let mut dim_sds = vec![0.0f64; QUALIA_DIMS];
    for d in 0..QUALIA_DIMS {
        dim_means[d] = f32_16.iter().map(|v| v[d] as f64).sum::<f64>() / n as f64;
    }
    for d in 0..QUALIA_DIMS {
        dim_sds[d] = (f32_16.iter().map(|v| (v[d] as f64 - dim_means[d]).powi(2)).sum::<f64>() / n as f64).sqrt();
    }

    // Correlation matrix
    let mut corr = vec![vec![0.0f64; QUALIA_DIMS]; QUALIA_DIMS];
    for a in 0..QUALIA_DIMS {
        for b in 0..QUALIA_DIMS {
            if a == b { corr[a][b] = 1.0; continue; }
            let cov: f64 = f32_16.iter()
                .map(|v| (v[a] as f64 - dim_means[a]) * (v[b] as f64 - dim_means[b]))
                .sum::<f64>() / n as f64;
            corr[a][b] = if dim_sds[a] > 1e-9 && dim_sds[b] > 1e-9 {
                cov / (dim_sds[a] * dim_sds[b])
            } else { 0.0 };
        }
    }

    // Print compact correlation matrix (upper triangle)
    print!("  {:>12}", "");
    for d in 0..QUALIA_DIMS {
        print!(" {:>5}", &DIMS_16_NAMES[d][..5.min(DIMS_16_NAMES[d].len())]);
    }
    println!();
    for a in 0..QUALIA_DIMS {
        print!("  {:>12}", DIMS_16_NAMES[a]);
        for b in 0..QUALIA_DIMS {
            if b < a {
                print!("      ");
            } else {
                let r = corr[a][b];
                let marker = if b != a && r.abs() > 0.7 { "!" } else { " " };
                print!(" {:>4.2}{}", r, marker);
            }
        }
        println!();
    }

    // 9b. Suspected redundancy pairs
    println!("\n  --- 9b. Suspected redundancy pairs ---\n");
    let suspects = [
        (1, 13, "valence ↔ optimism"),
        (3, 9,  "agency ↔ volition"),
        (4, 6,  "resonance ↔ social"),
        (0, 5,  "glow ↔ clarity"),
        (10, 14, "dissonance ↔ friction"),
        (7, 12, "gravity ↔ loss"),
    ];
    println!("  {:<26}  {:>6}  {}", "Pair", "r", "Verdict");
    for (a, b, label) in &suspects {
        let r = corr[*a][*b];
        let verdict = if r.abs() > 0.85 { "REDUNDANT — collapse candidate" }
                      else if r.abs() > 0.70 { "HIGH — possible compression" }
                      else if r.abs() > 0.50 { "moderate — independent enough" }
                      else { "independent ✓" };
        println!("  {:<26}  {:>+6.3}  {}", label, r, verdict);
    }

    // All high correlations (|r| > 0.5)
    println!("\n  All pairs with |r| > 0.50:");
    let mut high_corrs: Vec<(usize, usize, f64)> = Vec::new();
    for a in 0..QUALIA_DIMS {
        for b in (a + 1)..QUALIA_DIMS {
            if corr[a][b].abs() > 0.50 {
                high_corrs.push((a, b, corr[a][b]));
            }
        }
    }
    high_corrs.sort_by(|x, y| y.2.abs().partial_cmp(&x.2.abs()).unwrap());
    for (a, b, r) in &high_corrs {
        println!("    {:<14} ↔ {:<14}  r={:>+.3}", DIMS_16_NAMES[*a], DIMS_16_NAMES[*b], r);
    }
    if high_corrs.is_empty() { println!("    (none — all dimensions independent)"); }

    // 9c. PCA via power iteration on correlation matrix
    // Compute eigenvalues to see how many dimensions actually matter
    println!("\n  --- 9c. PCA eigenvalue decomposition ---");
    println!("  How many dimensions carry independent signal?\n");

    // Simple iterative eigenvalue extraction (deflation method)
    let mut work_mat = corr.clone();
    let mut eigenvalues: Vec<f64> = Vec::new();

    for _ev in 0..QUALIA_DIMS {
        // Power iteration to find largest eigenvalue
        let mut v = vec![1.0f64; QUALIA_DIMS];
        let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
        for x in v.iter_mut() { *x /= norm; }

        for _iter in 0..200 {
            let mut new_v = vec![0.0f64; QUALIA_DIMS];
            for i in 0..QUALIA_DIMS {
                for j in 0..QUALIA_DIMS {
                    new_v[i] += work_mat[i][j] * v[j];
                }
            }
            let norm = (new_v.iter().map(|x| x * x).sum::<f64>()).sqrt();
            if norm < 1e-12 { break; }
            for x in new_v.iter_mut() { *x /= norm; }
            v = new_v;
        }

        // Eigenvalue = v^T * M * v
        let mut mv = vec![0.0f64; QUALIA_DIMS];
        for i in 0..QUALIA_DIMS {
            for j in 0..QUALIA_DIMS {
                mv[i] += work_mat[i][j] * v[j];
            }
        }
        let lambda: f64 = v.iter().zip(&mv).map(|(a, b)| a * b).sum();
        eigenvalues.push(lambda.max(0.0));

        // Deflate: M = M - lambda * v * v^T
        for i in 0..QUALIA_DIMS {
            for j in 0..QUALIA_DIMS {
                work_mat[i][j] -= lambda * v[i] * v[j];
            }
        }
    }

    let total_var: f64 = eigenvalues.iter().sum();
    let mut cum_var = 0.0;
    println!("  {:>4}  {:>10}  {:>10}  {:>10}  {}", "PC", "Eigenvalue", "% Var", "Cum %", "Bar");
    for (i, &ev) in eigenvalues.iter().enumerate() {
        let pct = 100.0 * ev / total_var;
        cum_var += pct;
        let bar_len = (pct * 0.5).round() as usize;
        let bar: String = std::iter::repeat('█').take(bar_len).collect();
        let marker = if cum_var >= 90.0 && cum_var - pct < 90.0 { " ← 90%" } else { "" };
        println!("  PC{:<2}  {:>10.4}  {:>9.1}%  {:>9.1}%  {}{}",
            i + 1, ev, pct, cum_var, bar, marker);
    }

    // Count dimensions needed for 90% variance
    cum_var = 0.0;
    let mut dims_90 = QUALIA_DIMS;
    for (i, &ev) in eigenvalues.iter().enumerate() {
        cum_var += 100.0 * ev / total_var;
        if cum_var >= 90.0 { dims_90 = i + 1; break; }
    }
    println!("\n  Dimensions for 90% variance: {} / {}", dims_90, QUALIA_DIMS);
    if dims_90 <= 12 { println!("  → {} dims are near-redundant, could compress to {}", QUALIA_DIMS - dims_90, dims_90); }
    else { println!("  → All dimensions carry meaningful variance"); }

    // 9d. Per-family dimension signature
    println!("\n  --- 9d. Per-family dimension signatures ---");
    println!("  Which dims define each family? (z-score of family centroid vs global mean)\n");

    // Global mean per dim
    let global_mean: Vec<f64> = (0..QUALIA_DIMS).map(|d| {
        f32_16.iter().map(|v| v[d] as f64).sum::<f64>() / n as f64
    }).collect();
    let global_sd: Vec<f64> = (0..QUALIA_DIMS).map(|d| {
        (f32_16.iter().map(|v| (v[d] as f64 - global_mean[d]).powi(2)).sum::<f64>() / n as f64).sqrt()
    }).collect();

    // Show top-5 most distinctive families
    let mut fam_signatures: Vec<(String, Vec<(usize, f64)>)> = Vec::new();
    for fam in &families {
        let members: Vec<usize> = items.iter().enumerate()
            .filter(|(_, it)| it.family == *fam).map(|(i, _)| i).collect();
        if members.len() < 3 { continue; }
        let fam_mean: Vec<f64> = (0..QUALIA_DIMS).map(|d| {
            members.iter().map(|&i| f32_16[i][d] as f64).sum::<f64>() / members.len() as f64
        }).collect();
        let mut zscores: Vec<(usize, f64)> = (0..QUALIA_DIMS).map(|d| {
            let z = if global_sd[d] > 1e-9 { (fam_mean[d] - global_mean[d]) / global_sd[d] } else { 0.0 };
            (d, z)
        }).collect();
        zscores.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        fam_signatures.push((fam.clone(), zscores));
    }
    fam_signatures.sort_by(|a, b| {
        b.1[0].1.abs().partial_cmp(&a.1[0].1.abs()).unwrap()
    });

    for (fam, zscores) in fam_signatures.iter().take(10) {
        let top3: Vec<String> = zscores.iter().take(3).map(|&(d, z)| {
            format!("{}={:+.1}", DIMS_16_NAMES[d], z)
        }).collect();
        println!("  {:<20}  {}", fam, top3.join("  "));
    }

    // 9e. Antonym consistency: do known opposites maximize expected dimension?
    println!("\n  --- 9e. Antonym consistency ---");
    println!("  Do contrast pairs peak on the expected dynamical axis?\n");

    let antonym_pairs = [
        ("grief_private_weight", "play_childlike_energy", "valence", 1usize),
        ("anger_feral_burst", "stillness_safe_void", "agency", 3),
        ("devotion_total_presence", "letting_go_complete", "volition", 9),
        ("power_commanding_voice", "surrender_soft_collapse", "rooting", 2),
        ("curiosity_hungry_focus", "grief_sacred_mourning", "optimism", 13),
        ("awe_starfield_collapse", "endurance_ritual_breathing", "staunen", 11),
    ];

    println!("  {:<22} ↔ {:<22}  {:>10}  {:>4}  {}", "Item A", "Item B", "Expected", "Rank", "Top dim");
    for (a_id, b_id, expected_dim, expected_idx) in &antonym_pairs {
        if let (Some(&ai), Some(&bi)) = (id_map.get(a_id), id_map.get(b_id)) {
            // Find which dimension has largest absolute difference
            let mut dim_diffs: Vec<(usize, u8)> = (0..QUALIA_DIMS)
                .map(|d| (d, nib4_vecs[ai][d].abs_diff(nib4_vecs[bi][d])))
                .collect();
            dim_diffs.sort_by(|a, b| b.1.cmp(&a.1));

            let rank = dim_diffs.iter().position(|&(d, _)| d == *expected_idx)
                .map(|r| r + 1).unwrap_or(99);

            let top_dim = DIMS_16_NAMES[dim_diffs[0].0];
            let short_a: String = a_id.chars().take(22).collect();
            let short_b: String = b_id.chars().take(22).collect();
            let flag = if rank <= 3 { " ✓" } else { " ⚠" };
            println!("  {:<22} ↔ {:<22}  {:>10}  {:>4}  {}{}",
                short_a, short_b, expected_dim, rank, top_dim, flag);
        }
    }

    // ========================================================================
    // CHECK 10: SINGLE-AXIS ISOLATION — find pairs that isolate one mode
    // ========================================================================
    println!("\n=== CHECK 10: SINGLE-AXIS PAIR ISOLATION ===");
    println!("  For each dimension: find pairs where |Δd| is max, Σ|Δk≠d| is min.\n");

    // 10a. Best isolating pair per dimension
    println!("  --- 10a. Best isolating pair per dimension ---\n");
    println!("  {:<14}  {:<22} ↔ {:<22}  {:>5}  {:>5}  {:>5}",
        "Dimension", "Item A", "Item B", "|Δd|", "Σother", "Ratio");

    for target_d in 0..QUALIA_DIMS {
        let mut best_ratio = 0.0f64;
        let mut best_pair = (0usize, 0usize);
        let mut best_delta = 0u8;
        let mut best_other = 0u32;

        for i in 0..n {
            for j in (i + 1)..n {
                let delta_d = nib4_vecs[i][target_d].abs_diff(nib4_vecs[j][target_d]);
                if delta_d < 5 { continue; } // must be a real separation

                let other_sum: u32 = (0..QUALIA_DIMS)
                    .filter(|&d| d != target_d)
                    .map(|d| nib4_vecs[i][d].abs_diff(nib4_vecs[j][d]) as u32)
                    .sum();

                let ratio = delta_d as f64 / (other_sum as f64 + 1.0);
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best_pair = (i, j);
                    best_delta = delta_d;
                    best_other = other_sum;
                }
            }
        }

        let (ai, bi) = best_pair;
        let short_a: String = items[ai].id.chars().take(22).collect();
        let short_b: String = items[bi].id.chars().take(22).collect();
        let quality = if best_ratio > 0.5 { " ✓ clean" }
                      else if best_ratio > 0.2 { "" }
                      else { " ⚠ muddy" };
        println!("  {:<14}  {:<22} ↔ {:<22}  {:>5}  {:>5}  {:>5.2}{}",
            DIMS_16_NAMES[target_d], short_a, short_b,
            best_delta, best_other, best_ratio, quality);
    }

    // 10b. Find "ache vs pain" style mode-flip pairs
    // Pairs with SAME total magnitude but different causal stance
    // (similar Nib4 distance from center, but max distance from each other on 1-2 dims)
    println!("\n  --- 10b. Mode-flip candidates (same magnitude, different stance) ---");
    println!("  Pairs where total nibble sum is similar but specific dims flip.\n");

    // Compute nibble sum (total "magnitude") for each item
    let nib_sums: Vec<u32> = nib4_vecs.iter()
        .map(|v| v.iter().map(|&x| x as u32).sum())
        .collect();

    let mut mode_flips: Vec<(usize, usize, u32, u32, usize, u8)> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let mag_diff = (nib_sums[i] as i64 - nib_sums[j] as i64).unsigned_abs() as u32;
            if mag_diff > 15 { continue; } // similar total magnitude

            // Find the dimension with largest flip
            let mut max_flip_d = 0usize;
            let mut max_flip = 0u8;
            let mut total_dist = 0u32;
            for d in 0..QUALIA_DIMS {
                let delta = nib4_vecs[i][d].abs_diff(nib4_vecs[j][d]);
                total_dist += delta as u32;
                if delta > max_flip { max_flip = delta; max_flip_d = d; }
            }

            if max_flip >= 8 && total_dist < 60 { // strong single-dim flip, low total noise
                mode_flips.push((i, j, mag_diff, total_dist, max_flip_d, max_flip));
            }
        }
    }

    mode_flips.sort_by(|a, b| b.5.cmp(&a.5).then(a.3.cmp(&b.3)));

    println!("  {:<22} ↔ {:<22}  {:>6}  {:>4}  {:>4}  {:<14}",
        "Item A", "Item B", "MagΔ", "Dist", "Flip", "Flip dim");
    for &(i, j, mag_d, total_d, flip_d, flip_v) in mode_flips.iter().take(15) {
        let short_a: String = items[i].id.chars().take(22).collect();
        let short_b: String = items[j].id.chars().take(22).collect();
        println!("  {:<22} ↔ {:<22}  {:>6}  {:>4}  {:>4}  {:<14}",
            short_a, short_b, mag_d, total_d, flip_v, DIMS_16_NAMES[flip_d]);
    }

    // 10c. Proposed minimal orthogonal core (from PCA + correlation analysis)
    println!("\n  --- 10c. Redundancy clusters (from correlation matrix) ---\n");

    // Build clusters: dims with |r| > 0.7 are in same cluster
    let mut visited = vec![false; QUALIA_DIMS];
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for seed in 0..QUALIA_DIMS {
        if visited[seed] { continue; }
        let mut cluster = vec![seed];
        visited[seed] = true;
        let mut frontier = vec![seed];
        while let Some(curr) = frontier.pop() {
            for other in 0..QUALIA_DIMS {
                if !visited[other] && corr[curr][other].abs() > 0.70 {
                    visited[other] = true;
                    cluster.push(other);
                    frontier.push(other);
                }
            }
        }
        clusters.push(cluster);
    }

    for (ci, cluster) in clusters.iter().enumerate() {
        let names: Vec<&str> = cluster.iter().map(|&d| DIMS_16_NAMES[d]).collect();
        let leader = cluster[0]; // first dim = representative
        if cluster.len() > 1 {
            println!("  Cluster {}: [{}]", ci + 1, names.join(", "));
            println!("    → representative: {}", DIMS_16_NAMES[leader]);
            println!("    → collapsed dims: {}", names[1..].join(", "));
        } else {
            println!("  Singleton: {}", DIMS_16_NAMES[leader]);
        }
    }

    let core_count = clusters.len();
    println!("\n  Minimal orthogonal core: {} axes from {} dimensions", core_count, QUALIA_DIMS);
    if core_count <= 8 {
        println!("  → Fits in {} nibbles = {} bits = {} u16 words",
            core_count, core_count * 4, (core_count * 4 + 15) / 16);
    }

    // ========================================================================
    // CHECK 11: EQUILIBRIUM vs PEACE — vector vs state
    // ========================================================================
    println!("\n=== CHECK 11: EQUILIBRIUM vs PEACE (vector vs state) ===");
    println!("  Equilibrium = tension-aware stillness-seeking (a vector)");
    println!("  Peace = stillness attained (a state)\n");

    // 11a. Theoretical nibble profiles based on causal structure
    // Dim indices: 0=glow, 1=valence, 2=rooting, 3=agency,
    //              4=resonance, 5=clarity, 6=social, 7=gravity,
    //              8=reverence, 9=volition, 10=dissonance, 11=staunen,
    //              12=loss, 13=optimism, 14=friction, 15=equilibrium
    println!("  --- 11a. Theoretical nibble profiles ---\n");

    // Equilibrium: the ache for silence without the silence yet
    // - gravity HIGH (pull inward)
    // - dissonance MODERATE (felt imbalance)
    // - volition HIGH (desire for resolution)
    // - agency LOW (until balance restored)
    // - friction MODERATE
    // - equilibrium HIGH (by definition)
    // - glow LOW-MODERATE
    // - resonance MODERATE (seeking but not yet found)
    // Mixed mode, CMYK-leaning
    let equilibrium_profile: Vec<f32> = vec![
        0.3,  // glow: low-moderate (not emitting, searching)
        0.4,  // valence: slightly below neutral (uncomfortable awareness)
        0.3,  // rooting: low (allegiance not primary driver)
        0.25, // agency: low (haven't found balance yet)
        0.4,  // resonance: moderate (seeking attunement)
        0.5,  // clarity: moderate (aware of imbalance)
        0.3,  // social: low (inward-focused)
        0.8,  // gravity: HIGH (pull toward center)
        0.4,  // reverence: moderate (the search has weight)
        0.7,  // volition: HIGH (desire for resolution)
        0.6,  // dissonance: MODERATE-HIGH (felt imbalance)
        0.3,  // staunen: low (no wonder, just need)
        0.3,  // loss: low-moderate (not grief but ache)
        0.4,  // optimism: moderate (believes resolution possible)
        0.5,  // friction: MODERATE (the ripples resisting stillness)
        0.85, // equilibrium: HIGH (resolution_hunger is the whole point)
    ];

    // Peace: the lake at dawn
    // - dissonance LOW (resolved)
    // - friction LOW (no resistance)
    // - equilibrium LOW (no hunger — already there)
    // - resonance HIGH (stable attunement)
    // - agency MODERATE (restored, at rest)
    // - glow MODERATE (warm but not burning)
    // CMYK mode — absorbing, not emitting
    let peace_profile: Vec<f32> = vec![
        0.5,  // glow: moderate (warm glow, not burning)
        0.8,  // valence: HIGH (positive state)
        0.4,  // rooting: moderate (grounded but not striving)
        0.5,  // agency: moderate (capacity restored, at rest)
        0.7,  // resonance: HIGH (stable harmony)
        0.6,  // clarity: moderate-high (clear, undisturbed)
        0.5,  // social: moderate
        0.3,  // gravity: LOW (no pull needed, already centered)
        0.5,  // reverence: moderate (stillness has quiet weight)
        0.2,  // volition: LOW (no desire — content)
        0.1,  // dissonance: VERY LOW (resolved)
        0.4,  // staunen: moderate (peace can hold wonder)
        0.1,  // loss: VERY LOW
        0.6,  // optimism: moderate-high (at ease)
        0.1,  // friction: VERY LOW (smooth)
        0.1,  // equilibrium: LOW (no hunger — already satisfied)
    ];

    // Encode theoretical profiles
    let eq_nibs = codebook.encode_vec(&equilibrium_profile);
    let peace_nibs = codebook.encode_vec(&peace_profile);

    println!("  {:>14}  {:>5}  {:>5}  {:>5}", "Dimension", "Equil", "Peace", "|Δ|");
    let mut eq_peace_total_delta = 0u32;
    for d in 0..QUALIA_DIMS {
        let delta = eq_nibs[d].abs_diff(peace_nibs[d]);
        eq_peace_total_delta += delta as u32;
        let marker = if delta >= 4 { " ←KEY" } else { "" };
        println!("  {:>14}  {:>5X}  {:>5X}  {:>5}{}",
            DIMS_16_NAMES[d], eq_nibs[d], peace_nibs[d], delta, marker);
    }
    println!("  {:>14}         TOTAL  {:>5}", "", eq_peace_total_delta);
    println!("  Theoretical separation: {}/240 = {:.1}%",
        eq_peace_total_delta, 100.0 * eq_peace_total_delta as f64 / 240.0);

    // 11b. Find corpus items closest to each theoretical profile
    println!("\n  --- 11b. Corpus items closest to each profile ---\n");

    let mut eq_dists: Vec<(usize, u32)> = (0..n)
        .map(|i| (i, nib4_distance(&nib4_vecs[i], &eq_nibs)))
        .collect();
    eq_dists.sort_by_key(|&(_, d)| d);

    let mut peace_dists: Vec<(usize, u32)> = (0..n)
        .map(|i| (i, nib4_distance(&nib4_vecs[i], &peace_nibs)))
        .collect();
    peace_dists.sort_by_key(|&(_, d)| d);

    println!("  Top-10 EQUILIBRIUM candidates (closest to theoretical profile):");
    println!("  {:<40} {:>5} {:>7}  {}", "Item", "Dist", "Family", "I-bit");
    for &(idx, dist) in eq_dists.iter().take(10) {
        let mode = if intensity_bits[idx] { "CMYK" } else { "RGB" };
        println!("  {:<40} {:>5} {:>7}  {}",
            items[idx].id, dist, items[idx].family, mode);
    }

    println!("\n  Top-10 PEACE candidates (closest to theoretical profile):");
    println!("  {:<40} {:>5} {:>7}  {}", "Item", "Dist", "Family", "I-bit");
    for &(idx, dist) in peace_dists.iter().take(10) {
        let mode = if intensity_bits[idx] { "CMYK" } else { "RGB" };
        println!("  {:<40} {:>5} {:>7}  {}",
            items[idx].id, dist, items[idx].family, mode);
    }

    // 11c. Overlap check: how many items appear in both top-20?
    let eq_top20: Vec<usize> = eq_dists.iter().take(20).map(|&(i, _)| i).collect();
    let peace_top20: Vec<usize> = peace_dists.iter().take(20).map(|&(i, _)| i).collect();
    let overlap: Vec<usize> = eq_top20.iter().filter(|i| peace_top20.contains(i)).cloned().collect();
    println!("\n  Top-20 overlap: {} items in both neighborhoods", overlap.len());
    if !overlap.is_empty() {
        println!("  Ambiguous items (could be either):");
        for &idx in &overlap {
            println!("    {:<40} {:>7}", items[idx].id, items[idx].family);
        }
    }

    // 11d. Dimension-wise separation: which dims best distinguish eq from peace in corpus?
    println!("\n  --- 11c. Empirical separation (eq-like vs peace-like items) ---\n");

    let eq_idx: Vec<usize> = eq_dists.iter().take(15).map(|&(i, _)| i).collect();
    let peace_idx: Vec<usize> = peace_dists.iter().take(15).map(|&(i, _)| i).collect();

    println!("  {:<14}  {:>6}  {:>6}  {:>6}  {:>6}", "Dimension", "M_eq", "M_pce", "|Δ|", "Predicted");
    for d in 0..QUALIA_DIMS {
        let mean_eq: f64 = eq_idx.iter().map(|&i| nib4_vecs[i][d] as f64).sum::<f64>() / eq_idx.len() as f64;
        let mean_peace: f64 = peace_idx.iter().map(|&i| nib4_vecs[i][d] as f64).sum::<f64>() / peace_idx.len() as f64;
        let delta = (mean_eq - mean_peace).abs();
        // Compare to theoretical prediction
        let pred_eq = eq_nibs[d] as f64;
        let pred_peace = peace_nibs[d] as f64;
        let pred_dir = if pred_eq > pred_peace { "eq>pce" } else if pred_eq < pred_peace { "pce>eq" } else { "equal" };
        let actual_dir = if mean_eq > mean_peace { "eq>pce" } else if mean_eq < mean_peace { "pce>eq" } else { "equal" };
        let match_flag = if pred_dir == actual_dir { " ✓" } else { " ✗" };
        println!("  {:<14}  {:>6.1}  {:>6.1}  {:>6.2}  {:>6}{}",
            DIMS_16_NAMES[d], mean_eq, mean_peace, delta, pred_dir, match_flag);
    }

    // 11e. Distance between equilibrium-cluster centroid and peace-cluster centroid
    let eq_centroid: Vec<f64> = (0..QUALIA_DIMS).map(|d| {
        eq_idx.iter().map(|&i| nib4_vecs[i][d] as f64).sum::<f64>() / eq_idx.len() as f64
    }).collect();
    let peace_centroid: Vec<f64> = (0..QUALIA_DIMS).map(|d| {
        peace_idx.iter().map(|&i| nib4_vecs[i][d] as f64).sum::<f64>() / peace_idx.len() as f64
    }).collect();
    let centroid_dist: f64 = eq_centroid.iter().zip(&peace_centroid)
        .map(|(a, b)| (a - b).abs()).sum();
    println!("\n  Centroid distance (eq vs peace): {:.1} nibble-steps", centroid_dist);
    println!("  As fraction of max: {:.1}% (>10% = distinguishable)",
        100.0 * centroid_dist / 240.0);

    // Key structural insight
    println!("\n  Key structural markers of equilibrium vs peace:");
    let mut dim_markers: Vec<(usize, f64)> = (0..QUALIA_DIMS).map(|d| {
        (d, eq_centroid[d] - peace_centroid[d])
    }).collect();
    dim_markers.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    for &(d, delta) in dim_markers.iter().take(6) {
        let dir = if delta > 0.0 { "eq↑" } else { "peace↑" };
        println!("    {:<14}  Δ={:>+.1}  ({})", DIMS_16_NAMES[d], delta, dir);
    }

    // ========================================================================
    // CHECK 12: RGB/CMYK FLOW POLARITY — prove the mode bit is latent
    // ========================================================================
    println!("\n=== CHECK 12: RGB/CMYK FLOW POLARITY MODE BIT ===");
    println!("  RGB  = causing / emitting / outward charge");
    println!("  CMYK = caused / absorbing / inward reception\n");

    // 12a. Hand-label items as RGB or CMYK based on MODE of experiencing
    // NOT "which emotion" but "how is it operating?"
    // RGB  = expressive causation: I act, I radiate, I move outward
    // CMYK = perceptual reweighting: the world feels different from inside
    // Same noun can live in either mode. Joy-as-laughter=RGB, Joy-as-warmth=CMYK.
    // Ache is CMYK (field-shaped, retroactive gravity). Pain is RGB (event-shaped).
    println!("  --- 12a. Manual RGB/CMYK labels (mode of experiencing) ---\n");

    let rgb_items = [
        // Outward emission, expressive causation, "what can be seen"
        "anger_feral_burst",          // outward explosion
        "anger_righteous_heat",       // projecting moral force
        "anger_controlled_cut",       // precise outward strike
        "anger_clean_no",             // boundary declaration outward
        "desire_want_with_teeth",     // active reaching outward
        "desire_unapologetic_pull",   // radiating want
        "desire_body_first",          // body as emitter
        "power_commanding_voice",     // projecting authority
        "power_crowned_after_cost",   // visible claiming
        "power_rooted_presence",      // radiating groundedness
        "devotion_bold_yes",          // public vow, outward commitment
        "devotion_total_presence",    // active giving of attention
        "play_childlike_energy",      // energy radiating outward
        "play_rule_break_smile",      // defiance as emission
        "curiosity_hungry_focus",     // reaching outward to grasp
        "curiosity_wide_eyes",        // active intake = still outward
        "letting_go_complete",        // active release (expelling)
        "repair_touch_returns",       // action toward restoration
        "longing_defiant_reach",      // reaching outward despite
        "liberation_bare_spine",      // emergence, radiating freedom
    ];

    let cmyk_items = [
        // Inward saturation, perceptual reweighting, "what it is like"
        "grief_private_weight",       // gravity in the belly
        "grief_sacred_mourning",      // ink soaking paper
        "grief_memory_flash",         // past floods present
        "grief_soft_acceptance",      // world recolored by loss
        "grief_unfinished_sentence",  // interior incompleteness
        "surrender_soft_collapse",    // dissolving inward
        "surrender_exhale_finally",   // release as absorption
        "surrender_into_unknown",     // falling inward
        "stillness_safe_void",        // interior space expands
        "stillness_edge_of_sleep",    // consciousness narrowing inward
        "stillness_witnessing",       // receptive interior observation
        "awe_cosmic_smallness",       // self shrinks, world floods in
        "awe_breath_taken",           // overwhelmed from inside
        "awe_vast_open_sky",          // perception expands inward
        "loss_control_overwhelm",     // interior flooding
        "longing_snowglobe_memory",   // memory reweights the present
        "longing_warm_regret",        // past colors everything
        "longing_body_memory",        // body remembers, ache-shaped
        "endurance_ritual_breathing",  // interior sustaining
        "presence_enough",            // being held, not acting
    ];

    // Look up indices
    let rgb_idx: Vec<usize> = rgb_items.iter()
        .filter_map(|id| id_map.get(id).copied())
        .collect();
    let cmyk_idx: Vec<usize> = cmyk_items.iter()
        .filter_map(|id| id_map.get(id).copied())
        .collect();

    println!("  Labeled: {} RGB + {} CMYK = {} total", rgb_idx.len(), cmyk_idx.len(),
        rgb_idx.len() + cmyk_idx.len());

    // 12b. Mean nibble profile per mode
    println!("\n  --- 12b. Per-mode nibble profiles ---\n");
    println!("  {:<14}  {:>6}  {:>6}  {:>6}  {}", "Dimension", "M_RGB", "M_CMYK", "|Δ|", "Direction");

    let mut rgb_means = vec![0.0f64; QUALIA_DIMS];
    let mut cmyk_means = vec![0.0f64; QUALIA_DIMS];
    for d in 0..QUALIA_DIMS {
        rgb_means[d] = rgb_idx.iter().map(|&i| nib4_vecs[i][d] as f64).sum::<f64>() / rgb_idx.len() as f64;
        cmyk_means[d] = cmyk_idx.iter().map(|&i| nib4_vecs[i][d] as f64).sum::<f64>() / cmyk_idx.len() as f64;
    }

    let mut mode_deltas: Vec<(usize, f64)> = Vec::new();
    for d in 0..QUALIA_DIMS {
        let delta = rgb_means[d] - cmyk_means[d];
        mode_deltas.push((d, delta));
        let dir = if delta > 0.5 { "RGB↑" }
                 else if delta < -0.5 { "CMYK↑" }
                 else { "~same" };
        let flag = if delta.abs() > 1.5 { " ← STRONG" } else { "" };
        println!("  {:<14}  {:>6.1}  {:>6.1}  {:>6.2}  {}{}",
            DIMS_16_NAMES[d], rgb_means[d], cmyk_means[d], delta.abs(), dir, flag);
    }

    // 12c. Linear separator: can we predict RGB/CMYK from the 16 nibble dims?
    // Simple approach: compute a weight vector from the mean difference
    // w_d = (mean_RGB_d - mean_CMYK_d), threshold at 0
    println!("\n  --- 12c. Linear separator (mean-diff classifier) ---\n");

    // Weight vector = direction from CMYK centroid to RGB centroid
    let weights: Vec<f64> = (0..QUALIA_DIMS).map(|d| rgb_means[d] - cmyk_means[d]).collect();
    let w_norm = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
    let w_unit: Vec<f64> = weights.iter().map(|w| w / w_norm).collect();

    // Project all labeled items onto this axis
    let midpoint: Vec<f64> = (0..QUALIA_DIMS).map(|d| (rgb_means[d] + cmyk_means[d]) / 2.0).collect();

    let mut rgb_projections: Vec<f64> = Vec::new();
    let mut cmyk_projections: Vec<f64> = Vec::new();

    for &i in &rgb_idx {
        let proj: f64 = (0..QUALIA_DIMS).map(|d| {
            (nib4_vecs[i][d] as f64 - midpoint[d]) * w_unit[d]
        }).sum();
        rgb_projections.push(proj);
    }
    for &i in &cmyk_idx {
        let proj: f64 = (0..QUALIA_DIMS).map(|d| {
            (nib4_vecs[i][d] as f64 - midpoint[d]) * w_unit[d]
        }).sum();
        cmyk_projections.push(proj);
    }

    // Classification accuracy at threshold=0
    let rgb_correct = rgb_projections.iter().filter(|&&p| p > 0.0).count();
    let cmyk_correct = cmyk_projections.iter().filter(|&&p| p <= 0.0).count();
    let total_labeled = rgb_idx.len() + cmyk_idx.len();
    let accuracy = (rgb_correct + cmyk_correct) as f64 / total_labeled as f64;

    println!("  Weight vector (top contributing dims):");
    let mut w_sorted: Vec<(usize, f64)> = w_unit.iter().enumerate().map(|(d, &w)| (d, w)).collect();
    w_sorted.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    for &(d, w) in w_sorted.iter().take(6) {
        let dir = if w > 0.0 { "→RGB" } else { "→CMYK" };
        println!("    {:<14}  w={:>+.3}  ({})", DIMS_16_NAMES[d], w, dir);
    }

    println!("\n  Classification accuracy: {}/{} = {:.1}%",
        rgb_correct + cmyk_correct, total_labeled, 100.0 * accuracy);
    println!("    RGB correct:  {}/{}", rgb_correct, rgb_idx.len());
    println!("    CMYK correct: {}/{}", cmyk_correct, cmyk_idx.len());

    if accuracy > 0.85 {
        println!("  → MODE BIT IS LATENT: 16 nibble dims encode RGB/CMYK polarity");
    } else if accuracy > 0.70 {
        println!("  → Mode signal present but noisy — needs refinement");
    } else {
        println!("  → Mode bit NOT clearly latent in current dims");
    }

    // 12d. Project ALL 219 items — what's the overall RGB/CMYK split?
    println!("\n  --- 12d. Full corpus polarity projection ---\n");

    let mut all_projections: Vec<(usize, f64)> = (0..n).map(|i| {
        let proj: f64 = (0..QUALIA_DIMS).map(|d| {
            (nib4_vecs[i][d] as f64 - midpoint[d]) * w_unit[d]
        }).sum();
        (i, proj)
    }).collect();
    all_projections.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let n_rgb = all_projections.iter().filter(|&&(_, p)| p > 0.0).count();
    let n_cmyk = n - n_rgb;
    println!("  Full corpus: {} RGB ({:.0}%) / {} CMYK ({:.0}%)",
        n_rgb, 100.0 * n_rgb as f64 / n as f64,
        n_cmyk, 100.0 * n_cmyk as f64 / n as f64);

    // Compare with shame-based intensity bit
    let mut agree = 0;
    for &(i, proj) in &all_projections {
        let derived_cmyk = proj <= 0.0;
        let shame_cmyk = intensity_bits[i];
        if derived_cmyk == shame_cmyk { agree += 1; }
    }
    println!("  Agreement with shame-based I-bit: {}/{} = {:.1}%",
        agree, n, 100.0 * agree as f64 / n as f64);

    // Show strongest RGB and CMYK items
    println!("\n  Strongest RGB items (most emitting/causing):");
    for &(idx, proj) in all_projections.iter().rev().take(8) {
        let shame_mode = if intensity_bits[idx] { "I=1" } else { "I=0" };
        println!("    {:<40} proj={:>+5.1}  {:>7}  {}",
            items[idx].id, proj, items[idx].family, shame_mode);
    }
    println!("\n  Strongest CMYK items (most absorbing/caused):");
    for &(idx, proj) in all_projections.iter().take(8) {
        let shame_mode = if intensity_bits[idx] { "I=1" } else { "I=0" };
        println!("    {:<40} proj={:>+5.1}  {:>7}  {}",
            items[idx].id, proj, items[idx].family, shame_mode);
    }

    // 12e. Corpus bias check — is the drama of connection over-represented?
    println!("\n  --- 12e. Corpus polarity balance ---");
    // Check per-family RGB/CMYK balance
    println!("  {:<20}  {:>4}  {:>4}  {:>7}", "Family", "RGB", "CMYK", "Balance");
    let mut fam_balance: Vec<(String, usize, usize)> = Vec::new();
    for fam in &families {
        let members: Vec<usize> = items.iter().enumerate()
            .filter(|(_, it)| it.family == *fam).map(|(i, _)| i).collect();
        let fam_rgb = members.iter().filter(|&&i| {
            all_projections.iter().find(|&&(idx, _)| idx == i)
                .map(|&(_, p)| p > 0.0).unwrap_or(false)
        }).count();
        let fam_cmyk = members.len() - fam_rgb;
        let balance = if fam_rgb + fam_cmyk > 0 {
            format!("{:.0}%R", 100.0 * fam_rgb as f64 / (fam_rgb + fam_cmyk) as f64)
        } else { "N/A".to_string() };
        fam_balance.push((fam.clone(), fam_rgb, fam_cmyk));
        println!("  {:<20}  {:>4}  {:>4}  {:>7}", fam, fam_rgb, fam_cmyk, balance);
    }

    // ========================================================================
    // CHECK 13: MISSING STATES — ecstatic joy, happiness, excitement
    // ========================================================================
    println!("\n=== CHECK 13: MISSING STATES — THE QUIET QUADRANT ===");
    println!("  The corpus models the drama of connection. These states don't strain.\n");

    // 13a. Theoretical profiles for the missing states
    // Based on causal structure and resolution policy:
    //   Happiness:     tension never fully formed. Resolves and stays.
    //   Ecstatic joy:  structural overflow. Resolution embraced repeatedly.
    //   Excitement:    predictive pleasure. Delays resolution deliberately.
    //   Gentle joy:    stable warm field. Quiet positive.
    println!("  --- 13a. Theoretical nibble profiles ---\n");

    // Happiness: friction absence. Breath that doesn't catch.
    // Resolution policy: resolves early, stays. No hunger.
    // Not expansion, not explosion, not transcendence. Enoughness.
    // CMYK-leaning (interior climate, not observable behavior)
    let happiness_profile: Vec<f32> = vec![
        0.6,  // glow: moderate (warm but not burning)
        0.75, // valence: HIGH (positive state)
        0.4,  // rooting: moderate (grounded, not striving)
        0.5,  // agency: moderate (capacity present, at ease)
        0.6,  // resonance: moderate-high (in tune)
        0.6,  // clarity: moderate-high (clear mind)
        0.65, // social: moderate-high (open)
        0.2,  // gravity: LOW (no pull, no weight)
        0.3,  // reverence: low (no sacred weight)
        0.2,  // volition: LOW (nothing missing)
        0.1,  // dissonance: VERY LOW (no imbalance)
        0.3,  // staunen: low (wonder not primary)
        0.05, // loss: NEAR ZERO
        0.7,  // optimism: HIGH (natural positivity)
        0.05, // friction: NEAR ZERO (friction absence IS the state)
        0.1,  // equilibrium: LOW (no resolution hunger)
    ];

    // Ecstatic joy: structural overflow. Light that overexposes the frame.
    // Resolution policy: resolution embraced repeatedly. Lands and leaps again.
    // Breaks containment. Collapses separation. Suspends self-reference.
    // Mode: RGB outward burst (emission) — but tips to CMYK when the body floods
    let ecstasy_profile: Vec<f32> = vec![
        0.95, // glow: MAXIMUM (overexposure)
        0.9,  // valence: VERY HIGH
        0.5,  // rooting: moderate (allegiance present but dissolving)
        0.3,  // agency: LOW (ecstasy dissolves agency — liberation has it, ecstasy doesn't)
        0.8,  // resonance: HIGH (everything resonates)
        0.7,  // clarity: HIGH (paradoxically clear even while dissolved)
        0.7,  // social: HIGH (boundaries gone)
        0.1,  // gravity: VERY LOW (gravity drops — this is the key distinction from awe)
        0.4,  // reverence: moderate
        0.2,  // volition: LOW (not wanting — overflowing)
        0.15, // dissonance: LOW (but not zero — there's productive vibration)
        0.5,  // staunen: moderate (wonder present but not dominant like awe)
        0.05, // loss: NEAR ZERO
        0.85, // optimism: VERY HIGH
        0.15, // friction: LOW (no chromatic friction)
        0.15, // equilibrium: LOW (doesn't hunger for resolution — generates it)
    ];

    // Excitement: anticipation with pulse. Something good is coming.
    // Resolution policy: delays resolution deliberately. Enjoys the V chord humming.
    // Predictive pleasure. Lives slightly ahead of present.
    // RGB mode (leaning forward, emitting toward what's coming)
    let excitement_profile: Vec<f32> = vec![
        0.8,  // glow: HIGH (lit up)
        0.8,  // valence: HIGH
        0.5,  // rooting: moderate (committed to the approaching thing)
        0.7,  // agency: HIGH (primed, ready)
        0.6,  // resonance: moderate-high
        0.6,  // clarity: moderate-high (focused ahead)
        0.5,  // social: moderate
        0.2,  // gravity: LOW (light on feet)
        0.2,  // reverence: low (not sacred, kinetic)
        0.6,  // volition: MODERATE-HIGH (wanting what's coming)
        0.3,  // dissonance: moderate (productive tension)
        0.3,  // staunen: low-moderate
        0.05, // loss: NEAR ZERO
        0.8,  // optimism: HIGH (positive anticipation)
        0.3,  // friction: moderate (the edge of anticipation)
        0.5,  // equilibrium: MODERATE (wants fulfillment — key difference from happiness)
    ];

    // Gentle joy: the hearth. Steady warmth. "This is good."
    // Different from happiness (which is climate). Joy has lift.
    // CMYK mode (quiet inner warmth, not expressed outward)
    let gentle_joy_profile: Vec<f32> = vec![
        0.7,  // glow: moderate-high (warm inner light)
        0.8,  // valence: HIGH
        0.5,  // rooting: moderate (connected, grounded)
        0.4,  // agency: moderate (not acting, being)
        0.7,  // resonance: HIGH (in harmony)
        0.6,  // clarity: moderate-high
        0.6,  // social: moderate-high (open but not pushing)
        0.15, // gravity: LOW
        0.4,  // reverence: moderate (joy has quiet weight)
        0.15, // volition: LOW (content)
        0.05, // dissonance: VERY LOW
        0.5,  // staunen: moderate (wonder present — "this is good")
        0.05, // loss: NEAR ZERO
        0.75, // optimism: HIGH
        0.05, // friction: VERY LOW
        0.05, // equilibrium: VERY LOW (nothing to hunger for)
    ];

    // Encode all profiles
    let happiness_nibs = codebook.encode_vec(&happiness_profile);
    let ecstasy_nibs = codebook.encode_vec(&ecstasy_profile);
    let excitement_nibs = codebook.encode_vec(&excitement_profile);
    let gentle_joy_nibs = codebook.encode_vec(&gentle_joy_profile);

    println!("  {:>14}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}  {:>5}",
        "Dimension", "Hpyns", "Ecst", "Excit", "G.Joy", "Equil", "Peace");
    for d in 0..QUALIA_DIMS {
        println!("  {:>14}  {:>5X}  {:>5X}  {:>5X}  {:>5X}  {:>5X}  {:>5X}",
            DIMS_16_NAMES[d],
            happiness_nibs[d], ecstasy_nibs[d], excitement_nibs[d],
            gentle_joy_nibs[d], eq_nibs[d], peace_nibs[d]);
    }

    // 13b. Pairwise distances between ALL theoretical profiles
    println!("\n  --- 13b. Pairwise distances between theoretical profiles ---\n");

    let profiles: Vec<(&str, &[u8])> = vec![
        ("Happiness", &happiness_nibs),
        ("Ecstasy", &ecstasy_nibs),
        ("Excitement", &excitement_nibs),
        ("Gentle Joy", &gentle_joy_nibs),
        ("Equilibrium", &eq_nibs),
        ("Peace", &peace_nibs),
    ];

    print!("  {:>12}", "");
    for (name, _) in &profiles {
        print!("  {:>8}", &name[..name.len().min(8)]);
    }
    println!();

    for (i, (name_a, nibs_a)) in profiles.iter().enumerate() {
        print!("  {:>12}", name_a);
        for (j, (_, nibs_b)) in profiles.iter().enumerate() {
            if j <= i {
                print!("  {:>8}", "");
            } else {
                let d = nib4_distance(nibs_a, nibs_b);
                print!("  {:>8}", d);
            }
        }
        println!();
    }

    // 13c. Resolution policy axis — the dimension that defines these states
    println!("\n  --- 13c. Resolution policy (the defining axis) ---\n");
    println!("  State          eq_dim  friction  volition  dissonance  Resolution policy");
    for (name, nibs) in &profiles {
        let eq_val = nibs[15];
        let friction_val = nibs[14];
        let volition_val = nibs[9];
        let diss_val = nibs[10];
        let policy = match *name {
            "Equilibrium" => "delayed seeking (ache for resolution)",
            "Peace" => "already resolved (no hunger)",
            "Happiness" => "tension never formed (friction absence)",
            "Ecstasy" => "resolves repeatedly (lands and leaps)",
            "Excitement" => "delays deliberately (enjoys the V chord)",
            "Gentle Joy" => "softly resolved (this is good)",
            _ => "",
        };
        println!("  {:<14}  {:>6X}  {:>8X}  {:>8X}  {:>10X}  {}",
            name, eq_val, friction_val, volition_val, diss_val, policy);
    }

    // 13d. Where do these land among existing corpus items?
    println!("\n  --- 13d. Nearest corpus items to each theoretical profile ---\n");

    for (name, nibs) in &profiles {
        let mut dists: Vec<(usize, u32)> = (0..n)
            .map(|i| (i, nib4_distance(&nib4_vecs[i], nibs)))
            .collect();
        dists.sort_by_key(|&(_, d)| d);

        println!("  {} — top 5 nearest:", name);
        for &(idx, dist) in dists.iter().take(5) {
            let mode = if intensity_bits[idx] { "CMYK" } else { "RGB" };
            println!("    d={:>3}  {:<40} {:>12}  {}",
                dist, items[idx].id, items[idx].family, mode);
        }
        println!();
    }

    // 13e. Coverage gap: minimum distance from each theoretical profile to corpus
    println!("  --- 13e. Coverage gaps (min dist to nearest corpus item) ---\n");
    println!("  {:<14}  {:>8}  {:>12}  {}", "Profile", "Min dist", "Nearest item", "Gap?");
    for (name, nibs) in &profiles {
        let mut dists: Vec<(usize, u32)> = (0..n)
            .map(|i| (i, nib4_distance(&nib4_vecs[i], nibs)))
            .collect();
        dists.sort_by_key(|&(_, d)| d);
        let (nearest_idx, nearest_dist) = dists[0];
        let gap = if nearest_dist > 35 { " ← GAP" }
                 else if nearest_dist > 25 { " ← sparse" }
                 else { " ← covered" };
        println!("  {:<14}  {:>8}  {:<35}{}",
            name, nearest_dist, &items[nearest_idx].id, gap);
    }

    // 13f. Four ecstasy modes — how do they differ in the nibble space?
    println!("\n  --- 13f. Four ecstasy modes ---\n");
    println!("  Each shares overflow, but differs in direction.\n");

    // Ecstasy-Embodied: earthly body joy, gravity drops, social stays high
    let ecstasy_embodied: Vec<f32> = vec![
        0.9, 0.85, 0.5, 0.3, 0.8, 0.6, 0.8, 0.1,  // high social, very low gravity
        0.3, 0.2, 0.1, 0.4, 0.05, 0.8, 0.1, 0.1,   // low everything else
    ];
    // Ecstasy-Sacred: reverence+staunen spike, agency zeroes
    let ecstasy_sacred: Vec<f32> = vec![
        0.8, 0.8, 0.3, 0.1, 0.7, 0.5, 0.4, 0.3,  // low agency, moderate gravity
        0.9, 0.1, 0.1, 0.95, 0.05, 0.7, 0.1, 0.05, // reverence+staunen MAX
    ];
    // Ecstasy-Erotic: resonance+rooting max, glow max, friction moderate
    let ecstasy_erotic: Vec<f32> = vec![
        0.95, 0.85, 0.9, 0.6, 0.9, 0.5, 0.5, 0.15, // rooting+resonance MAX
        0.2, 0.5, 0.2, 0.3, 0.05, 0.8, 0.4, 0.2,   // friction moderate
    ];
    // Ecstasy-Communal: social maxes, individuation dissolves
    let ecstasy_communal: Vec<f32> = vec![
        0.85, 0.9, 0.6, 0.3, 0.9, 0.6, 0.95, 0.1, // social+resonance MAX
        0.4, 0.15, 0.05, 0.5, 0.05, 0.85, 0.05, 0.05, // very low everything else
    ];

    let e_embodied_nibs = codebook.encode_vec(&ecstasy_embodied);
    let e_sacred_nibs = codebook.encode_vec(&ecstasy_sacred);
    let e_erotic_nibs = codebook.encode_vec(&ecstasy_erotic);
    let e_communal_nibs = codebook.encode_vec(&ecstasy_communal);

    let ecstasy_modes: Vec<(&str, &[u8], &str)> = vec![
        ("Embodied", &e_embodied_nibs, "body floods, CMYK (interior expansion)"),
        ("Sacred", &e_sacred_nibs, "numinous colonizes, CMYK (vertical surrender)"),
        ("Erotic", &e_erotic_nibs, "radiating toward, RGB (horizontal charge)"),
        ("Communal", &e_communal_nibs, "emit AND absorb (mode boundary)"),
    ];

    // Show key distinguishing dims
    println!("  {:>14}  {:>6}  {:>6}  {:>6}  {:>6}", "Dimension", "Embod", "Sacred", "Erotic", "Commun");
    for d in 0..QUALIA_DIMS {
        let vals = [e_embodied_nibs[d], e_sacred_nibs[d], e_erotic_nibs[d], e_communal_nibs[d]];
        let max_v = *vals.iter().max().unwrap();
        let min_v = *vals.iter().min().unwrap();
        let spread = max_v - min_v;
        let marker = if spread >= 5 { " ← DISTINGUISHING" } else { "" };
        println!("  {:>14}  {:>6X}  {:>6X}  {:>6X}  {:>6X}{}",
            DIMS_16_NAMES[d], vals[0], vals[1], vals[2], vals[3], marker);
    }

    // Pairwise distances between ecstasy modes
    println!("\n  Pairwise distances between ecstasy modes:");
    for (i, (name_a, nibs_a, _)) in ecstasy_modes.iter().enumerate() {
        for (j, (name_b, nibs_b, _)) in ecstasy_modes.iter().enumerate() {
            if j > i {
                let d = nib4_distance(nibs_a, nibs_b);
                println!("    {} ↔ {}: {}", name_a, name_b, d);
            }
        }
    }

    // RGB/CMYK mode of each ecstasy type
    println!("\n  Ecstasy mode polarity (projected onto RGB/CMYK axis):");
    for (name, nibs, expected) in &ecstasy_modes {
        let proj: f64 = (0..QUALIA_DIMS).map(|d| {
            (nibs[d] as f64 - midpoint[d]) * w_unit[d]
        }).sum();
        let actual = if proj > 0.0 { "RGB" } else { "CMYK" };
        println!("    {:<10}  proj={:>+5.1}  → {}  (expected: {})", name, proj, actual, expected);
    }

    // 13g. Excitement variants — same voltage, different direction
    println!("\n  --- 13g. Excitement variants ---\n");
    println!("  All share anticipatory voltage. Differ in target direction.\n");

    // Excitement-Playful: social+glow high, friction low, light
    let excite_playful: Vec<f32> = vec![
        0.8, 0.85, 0.4, 0.7, 0.6, 0.5, 0.8, 0.15,
        0.15, 0.5, 0.2, 0.3, 0.05, 0.8, 0.2, 0.4,
    ];
    // Excitement-Erotic: resonance+glow max, rooting high
    let excite_erotic: Vec<f32> = vec![
        0.9, 0.8, 0.8, 0.7, 0.8, 0.4, 0.4, 0.2,
        0.15, 0.7, 0.3, 0.2, 0.05, 0.7, 0.4, 0.5,
    ];
    // Excitement-Creative: clarity spikes, staunen rises
    let excite_creative: Vec<f32> = vec![
        0.8, 0.7, 0.5, 0.8, 0.5, 0.9, 0.3, 0.15,
        0.2, 0.7, 0.3, 0.5, 0.05, 0.75, 0.3, 0.5,
    ];
    // Excitement-Sacred: reverence+staunen rise, anticipation of the numinous
    let excite_sacred: Vec<f32> = vec![
        0.7, 0.7, 0.4, 0.5, 0.6, 0.5, 0.3, 0.3,
        0.8, 0.5, 0.3, 0.7, 0.05, 0.7, 0.3, 0.5,
    ];

    let ep_nibs = codebook.encode_vec(&excite_playful);
    let ee_nibs = codebook.encode_vec(&excite_erotic);
    let ec_nibs = codebook.encode_vec(&excite_creative);
    let es_nibs = codebook.encode_vec(&excite_sacred);

    let excite_variants = [
        ("Playful", &ep_nibs, "emberglow + woodwarm"),
        ("Erotic", &ee_nibs, "emberglow + steelwind"),
        ("Creative", &ec_nibs, "steelwind + emberglow"),
        ("Sacred", &es_nibs, "velvetpause + steelwind"),
    ];

    println!("  {:>14}  {:>6}  {:>6}  {:>6}  {:>6}", "Dimension", "Play", "Erotic", "Create", "Sacred");
    for d in 0..QUALIA_DIMS {
        println!("  {:>14}  {:>6X}  {:>6X}  {:>6X}  {:>6X}",
            DIMS_16_NAMES[d], ep_nibs[d], ee_nibs[d], ec_nibs[d], es_nibs[d]);
    }

    println!("\n  Nearest corpus item per excitement variant:");
    for (name, nibs, qualia) in &excite_variants {
        let mut dists: Vec<(usize, u32)> = (0..n)
            .map(|i| (i, nib4_distance(&nib4_vecs[i], *nibs)))
            .collect();
        dists.sort_by_key(|&(_, d)| d);
        let (idx, dist) = dists[0];
        println!("    {:<10}  d={:>3}  {:<35}  ({})", name, dist, items[idx].id, qualia);
    }

    // 13h. The anxiety-excitement razor: same engine, sign bit flips
    println!("\n  --- 13h. The anxiety-excitement razor ---");
    println!("  Same engine (high arousal + uncertainty). Valence sign flips.\n");

    // Anxiety: same as excitement but valence drops, loss rises, optimism drops
    let anxiety_profile: Vec<f32> = vec![
        0.4,  // glow: LOW (dimmed)
        0.2,  // valence: LOW (negative)
        0.3,  // rooting: low
        0.3,  // agency: low (out of control)
        0.4,  // resonance: moderate (heightened awareness)
        0.3,  // clarity: low (foggy)
        0.3,  // social: low
        0.7,  // gravity: HIGH (weight, pull)
        0.3,  // reverence: low
        0.5,  // volition: moderate (wants escape)
        0.7,  // dissonance: HIGH
        0.2,  // staunen: low
        0.5,  // loss: moderate-high
        0.2,  // optimism: LOW
        0.7,  // friction: HIGH
        0.8,  // equilibrium: HIGH (desperately wants balance)
    ];

    let anxiety_nibs = codebook.encode_vec(&anxiety_profile);

    println!("  {:>14}  {:>8}  {:>8}  {:>6}", "Dimension", "Excite", "Anxiety", "|Δ|");
    let mut same_count = 0;
    let mut diff_count = 0;
    for d in 0..QUALIA_DIMS {
        let delta = excitement_nibs[d].abs_diff(anxiety_nibs[d]);
        let marker = if delta >= 5 { " ← SIGN FLIP" } else if delta <= 1 { " ≈ same engine" } else { "" };
        if delta <= 2 { same_count += 1; } else { diff_count += 1; }
        println!("  {:>14}  {:>8X}  {:>8X}  {:>6}{}",
            DIMS_16_NAMES[d], excitement_nibs[d], anxiety_nibs[d], delta, marker);
    }
    println!("\n  Same engine dims (|Δ|≤2): {}/{}  — confirms shared arousal base", same_count, QUALIA_DIMS);
    println!("  Sign-flip dims (|Δ|>2):  {}/{}  — the valence inversion", diff_count, QUALIA_DIMS);

    let ea_dist = nib4_distance(&excitement_nibs, &anxiety_nibs);
    println!("  Total distance: {}/240 = {:.1}% — close but distinguishable",
        ea_dist, 100.0 * ea_dist as f64 / 240.0);

    // ========================================================================
    // VERDICT
    // ========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                       FINAL LAYOUT                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                            ║");
    println!("║  word 0: [glow|valence|rooting|agency]         = 4 × F     ║");
    println!("║  word 1: [resonance|clarity|social|gravity]    = 4 × F     ║");
    println!("║  word 2: [reverence|volition|dissonance|staunen] = 4 × F   ║");
    println!("║  word 3: [loss|optimism|friction|equilibrium]  = 4 × F     ║");
    println!("║  word 4: [I|000...0] intensity bit at BF16 sign pos        ║");
    println!("║          I=0: RGB/causing  I=1: CMYK/caused                ║");
    println!("║  word 5..1023: topology (16,304 bits)                      ║");
    println!("║                                                            ║");
    println!("║  Distance: Manhattan over 16 nibbles ∈ [0, 0xF0]           ║");
    println!("║  + intensity penalty if causality direction flips          ║");
    println!("║                                                            ║");
    println!("║  Equilibrium ≠ Peace.                                      ║");
    println!("║  Equilibrium = tension-aware stillness-seeking (vector)    ║");
    println!("║  Peace = stillness attained (state, low eq dim)            ║");
    println!("║  Mode bit: derived from 16 dims, not extra scalar          ║");
    println!("║                                                            ║");
    println!("║  Resolution policy defines state class:                    ║");
    println!("║    happiness  = tension never formed                       ║");
    println!("║    ecstasy    = resolves repeatedly (overflow)             ║");
    println!("║    excitement = delays deliberately (anticipation)         ║");
    println!("║    gentle joy = softly resolved (this is good)             ║");
    println!("║    longing    = resolution delayed                         ║");
    println!("║    grief      = resolution withheld                        ║");
    println!("║    equilibrium = resolution sought (ache for balance)      ║");
    println!("║                                                            ║");
    println!("║  Quintenzirkel: relative intervals, not absolute values    ║");
    println!("║                                                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
