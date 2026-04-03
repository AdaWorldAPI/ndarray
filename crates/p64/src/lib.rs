//! # p64 — Palette64
//!
//! A 64×64 BNN attention matrix built from 8 phyllotactic HEEL planes.
//!
//! ## Architecture
//!
//! ```text
//! 8 HEEL planes (u64 each = 64 bits)
//!     ↓ phyllotactic expansion (8 octaves × golden rotation)
//! 64 rows × 64 columns = 4096 bits = 512 bytes = 8 L1 cache lines
//! ```
//!
//! ## Usage
//!
//! - **BNN Attention**: `Q AND K >> Gamma` — binary attention in 1 instruction
//! - **MoE Fanout**: 8 HEELs are expert gates; palette rows are expert outputs
//! - **Palette Lookup**: nearest entry by Hamming distance (POPCNT)
//! - **Stable Diffusion**: the matrix IS the cross-attention pattern
//!
//! ## Why 64×64
//!
//! - 512 bytes = 8 cache lines = fully L1-resident
//! - 64 = 8 × 8: natural tiling of 8 SIMD lanes across 8 octaves
//! - Hamming distance on u64: single POPCNT instruction
//! - 8 HEELs: matches MoE expert count (Maverick = 128 experts / 16 groups = 8)

// Golden ratio used conceptually — GOLDEN_SHIFT_64 derived from it.

// ============================================================================
// Core Types
// ============================================================================

/// A 64×64 binary attention/palette matrix.
///
/// Each row is a u64 (64-bit pattern). 64 rows = 512 bytes.
/// Fully L1-resident. All operations are POPCNT + AND.
#[derive(Debug, Clone, Copy)]
#[repr(align(64))] // cache-line aligned
pub struct Palette64 {
    /// 64 rows, each 64 bits wide.
    pub rows: [u64; 64],
}

/// 8 HEEL planes — the seed from which the full palette is expanded.
///
/// Each HEEL is a 64-bit pattern representing one MoE expert's
/// activation fingerprint. The 8 HEELs fan out to 64 rows via
/// golden-angle rotation across 8 octaves.
#[derive(Debug, Clone, Copy)]
pub struct HeelPlanes {
    pub planes: [u64; 8],
}

/// Result of a palette attention query.
#[derive(Debug, Clone, Copy)]
pub struct AttentionResult {
    /// Index of the best-matching row (0..63).
    pub best_idx: u8,
    /// Hamming distance to best match (0..64).
    pub distance: u8,
    /// Full 64-entry score vector (popcount of Q AND row[i]).
    pub scores: [u8; 64],
    /// Bitmask: which rows exceed the Gamma threshold.
    pub fires: u64,
}

/// MoE gate decision: which experts contribute and how.
#[derive(Debug, Clone, Copy)]
pub struct MoeGate {
    /// Which of the 8 HEELs are active (bitmask).
    pub active: u8,
    /// Per-expert activation strength (popcount match).
    pub strength: [u8; 8],
    /// Combined output: OR of active expert planes.
    pub combined: u64,
}

// ============================================================================
// Golden rotation for palette expansion
// ============================================================================

/// Rotate a u64 by `shift` bits (circular).
#[inline]
const fn rotate_left(val: u64, shift: u32) -> u64 {
    val.rotate_left(shift)
}

/// Golden shift for 64-bit words: round(64 / φ) = 40.
/// gcd(40, 64) = 8 — NOT coprime. So we use round(64 / φ²) = 24.
/// But gcd(24, 64) = 8 — same problem. 64 = 2⁶, so any even shift aliases.
///
/// Solution: 39 (nearest odd to 64/φ). gcd(39, 64) = 1. Coprime.
const GOLDEN_SHIFT_64: u32 = 39;

/// Expand 8 HEEL planes into a full 64-row palette.
///
/// Row `n` = HEEL[n % 8] rotated by `(n / 8) × GOLDEN_SHIFT_64`.
/// 8 octaves × 8 HEELs = 64 rows.
/// Golden rotation ensures no two octaves of the same HEEL alias.
impl HeelPlanes {
    /// Create from 8 raw u64 planes.
    pub const fn new(planes: [u64; 8]) -> Self {
        Self { planes }
    }

    /// Create from a 34-byte CLAM seed using the 7+1 decomposition.
    ///
    /// The 7 payload slices become 7 HEEL planes.
    /// The contradiction becomes HEEL[7] (the anti-expert).
    pub fn from_clam_seed(data: &[i8; 34]) -> Self {
        let mut planes = [0u64; 8];

        // 7 payload planes: 4 bytes each → expand to 64 bits via golden stepping
        for i in 0..7 {
            let start = 1 + (i * 4);
            let bytes = [
                data[start] as u8,
                data[start + 1] as u8,
                data[start + 2] as u8,
                data[start + 3] as u8,
            ];
            let val = u32::from_le_bytes(bytes);
            // Spread 32 bits across 64 bits: each input bit controls 2 output bits
            // via golden-step interleave
            planes[i] = spread_32_to_64(val);
        }

        // Plane 7: contradiction (4 bytes from data[29..33])
        let contra_bytes = [
            data[29] as u8,
            data[30] as u8,
            data[31] as u8,
            data[32] as u8,
        ];
        let contra_val = u32::from_le_bytes(contra_bytes);
        planes[7] = spread_32_to_64(contra_val);

        Self { planes }
    }

    /// Expand to full 64-row palette via golden rotation.
    pub fn expand(&self) -> Palette64 {
        let mut rows = [0u64; 64];
        for octave in 0..8u32 {
            let rotation = octave * GOLDEN_SHIFT_64;
            for heel in 0..8 {
                let row_idx = octave as usize * 8 + heel;
                rows[row_idx] = rotate_left(self.planes[heel], rotation);
            }
        }
        Palette64 { rows }
    }
}

/// Spread 32 bits to 64 bits using golden-step interleave.
///
/// Bit `i` of input → bit `(i × 39) % 64` of output.
/// This ensures maximum dispersion with no adjacent clustering.
#[inline]
fn spread_32_to_64(val: u32) -> u64 {
    let mut out = 0u64;
    for i in 0..32 {
        if val & (1 << i) != 0 {
            let target = ((i as u32) * GOLDEN_SHIFT_64) % 64;
            out |= 1u64 << target;
        }
    }
    out
}

// ============================================================================
// Multi-versioned attend kernel: AVX-512 → AVX2 → scalar.
// ============================================================================

/// Return type for attend kernel: (best_idx, distance, scores, fires).
type AttendFn = unsafe fn(&[u64; 64], u64, u8) -> (u8, u8, [u8; 64], u64);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn attend_avx512(rows: &[u64; 64], query: u64, gamma: u8) -> (u8, u8, [u8; 64], u64) {
    let mut best_idx = 0u8;
    let mut best_score = 0u8;
    let mut scores = [0u8; 64];
    let mut fires = 0u64;

    // Process 8 rows per chunk, 8 chunks = 64 rows
    // (scalar array ops — LLVM auto-vectorizes with target-cpu=x86-64-v4)
    for chunk in 0..8 {
        let base = chunk * 8;
        let mut vals = [0u64; 8];
        for j in 0..8 {
            vals[j] = rows[base + j] & query;
        }
        for j in 0..8 {
            let score = vals[j].count_ones() as u8;
            let idx = base + j;
            scores[idx] = score;
            if score > best_score {
                best_score = score;
                best_idx = idx as u8;
            }
            if score >= gamma {
                fires |= 1u64 << idx;
            }
        }
    }
    (best_idx, 64 - best_score, scores, fires)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn attend_avx2(rows: &[u64; 64], query: u64, gamma: u8) -> (u8, u8, [u8; 64], u64) {
    let mut best_idx = 0u8;
    let mut best_score = 0u8;
    let mut scores = [0u8; 64];
    let mut fires = 0u64;

    // Process 4 rows per chunk, 16 chunks = 64 rows
    // (scalar array ops — LLVM auto-vectorizes with target-cpu=x86-64-v4)
    for chunk in 0..16 {
        let base = chunk * 4;
        let mut vals = [0u64; 4];
        for j in 0..4 {
            vals[j] = rows[base + j] & query;
        }
        for j in 0..4 {
            let score = vals[j].count_ones() as u8;
            let idx = base + j;
            scores[idx] = score;
            if score > best_score {
                best_score = score;
                best_idx = idx as u8;
            }
            if score >= gamma {
                fires |= 1u64 << idx;
            }
        }
    }
    (best_idx, 64 - best_score, scores, fires)
}

fn attend_scalar(rows: &[u64; 64], query: u64, gamma: u8) -> (u8, u8, [u8; 64], u64) {
    let mut best_idx = 0u8;
    let mut best_score = 0u8;
    let mut scores = [0u8; 64];
    let mut fires = 0u64;
    for i in 0..64 {
        let score = (query & rows[i]).count_ones() as u8;
        scores[i] = score;
        if score > best_score {
            best_score = score;
            best_idx = i as u8;
        }
        if score >= gamma {
            fires |= 1u64 << i;
        }
    }
    (best_idx, 64 - best_score, scores, fires)
}

static ATTEND_KERNEL: std::sync::LazyLock<AttendFn> = std::sync::LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return attend_avx512 as AttendFn;
        }
        if is_x86_feature_detected!("avx2") {
            return attend_avx2 as AttendFn;
        }
    }
    attend_scalar as AttendFn
});

// ============================================================================
// Multi-versioned nearest_k kernel: AVX-512 → AVX2 → scalar.
// ============================================================================

/// Compute all 64 Hamming distances in one pass.
type NearestKFn = unsafe fn(&[u64; 64], u64) -> [u8; 64];

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn nearest_k_avx512(rows: &[u64; 64], query: u64) -> [u8; 64] {
    let mut dists = [0u8; 64];
    // Scalar array ops — LLVM auto-vectorizes with target-cpu=x86-64-v4
    for chunk in 0..8 {
        let base = chunk * 8;
        let mut vals = [0u64; 8];
        for j in 0..8 {
            vals[j] = rows[base + j] ^ query;
        }
        for j in 0..8 {
            dists[base + j] = vals[j].count_ones() as u8;
        }
    }
    dists
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nearest_k_avx2(rows: &[u64; 64], query: u64) -> [u8; 64] {
    let mut dists = [0u8; 64];
    // Scalar array ops — LLVM auto-vectorizes with target-cpu=x86-64-v4
    for chunk in 0..16 {
        let base = chunk * 4;
        let mut vals = [0u64; 4];
        for j in 0..4 {
            vals[j] = rows[base + j] ^ query;
        }
        for j in 0..4 {
            dists[base + j] = vals[j].count_ones() as u8;
        }
    }
    dists
}

fn nearest_k_scalar(rows: &[u64; 64], query: u64) -> [u8; 64] {
    let mut dists = [0u8; 64];
    for i in 0..64 {
        dists[i] = (query ^ rows[i]).count_ones() as u8;
    }
    dists
}

static NEAREST_K_KERNEL: std::sync::LazyLock<NearestKFn> = std::sync::LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return nearest_k_avx512 as NearestKFn;
        }
        if is_x86_feature_detected!("avx2") {
            return nearest_k_avx2 as NearestKFn;
        }
    }
    nearest_k_scalar as NearestKFn
});

// ============================================================================
// Multi-versioned moe_gate kernel: AVX-512 → AVX2 → scalar.
// ============================================================================

/// Return type: (active_mask, strength[8], combined).
type MoeGateFn = unsafe fn(&[u64; 8], u64, u8) -> (u8, [u8; 8], u64);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn moe_gate_avx512(planes: &[u64; 8], query: u64, threshold: u8) -> (u8, [u8; 8], u64) {
    // Scalar array ops — LLVM auto-vectorizes with target-cpu=x86-64-v4
    let mut vals = [0u64; 8];
    for i in 0..8 {
        vals[i] = planes[i] & query;
    }

    let mut active = 0u8;
    let mut strength = [0u8; 8];
    let mut combined = 0u64;
    for i in 0..8 {
        let score = vals[i].count_ones() as u8;
        strength[i] = score;
        if score >= threshold {
            active |= 1 << i;
            combined |= planes[i];
        }
    }
    (active, strength, combined)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn moe_gate_avx2(planes: &[u64; 8], query: u64, threshold: u8) -> (u8, [u8; 8], u64) {
    // Scalar array ops — LLVM auto-vectorizes with target-cpu=x86-64-v4
    let mut active = 0u8;
    let mut strength = [0u8; 8];
    let mut combined = 0u64;

    // Process 4 planes at a time, 2 chunks = 8 planes
    for chunk in 0..2 {
        let base = chunk * 4;
        let mut vals = [0u64; 4];
        for j in 0..4 {
            vals[j] = planes[base + j] & query;
        }
        for j in 0..4 {
            let score = vals[j].count_ones() as u8;
            let idx = base + j;
            strength[idx] = score;
            if score >= threshold {
                active |= 1 << idx;
                combined |= planes[idx];
            }
        }
    }
    (active, strength, combined)
}

fn moe_gate_scalar(planes: &[u64; 8], query: u64, threshold: u8) -> (u8, [u8; 8], u64) {
    let mut active = 0u8;
    let mut strength = [0u8; 8];
    let mut combined = 0u64;
    for i in 0..8 {
        let score = (query & planes[i]).count_ones() as u8;
        strength[i] = score;
        if score >= threshold {
            active |= 1 << i;
            combined |= planes[i];
        }
    }
    (active, strength, combined)
}

static MOE_GATE_KERNEL: std::sync::LazyLock<MoeGateFn> = std::sync::LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return moe_gate_avx512 as MoeGateFn;
        }
        if is_x86_feature_detected!("avx2") {
            return moe_gate_avx2 as MoeGateFn;
        }
    }
    moe_gate_scalar as MoeGateFn
});

// ============================================================================
// BNN Attention
// ============================================================================

impl Palette64 {
    /// Create an all-zero palette.
    pub const fn zero() -> Self {
        Self { rows: [0u64; 64] }
    }

    /// BNN attention: find best-matching row for query.
    ///
    /// Score = popcount(query AND row[i]).
    /// Higher score = more bits in common = better match.
    /// Gamma threshold: rows below this score don't "fire."
    ///
    /// Runtime dispatch via LazyLock: AVX-512 → AVX2 → scalar.
    #[inline]
    pub fn attend(&self, query: u64, gamma: u8) -> AttentionResult {
        // SAFETY: LazyLock guarantees the selected kernel matches CPU features.
        let (best_idx, distance, scores, fires) =
            unsafe { ATTEND_KERNEL(&self.rows, query, gamma) };
        AttentionResult {
            best_idx,
            distance,
            scores,
            fires,
        }
    }

    /// Multi-head attention: query with K separate query patterns.
    ///
    /// Returns the OR-combined firing mask across all heads.
    #[inline]
    pub fn multi_head_attend(&self, queries: &[u64], gamma: u8) -> u64 {
        let mut combined_fires = 0u64;
        for &q in queries {
            let result = self.attend(q, gamma);
            combined_fires |= result.fires;
        }
        combined_fires
    }

    /// Palette lookup: find the K nearest rows by Hamming distance.
    ///
    /// Returns (row_index, hamming_distance) sorted ascending.
    ///
    /// Runtime dispatch via LazyLock: AVX-512 → AVX2 → scalar.
    pub fn nearest_k(&self, query: u64, k: usize) -> Vec<(u8, u8)> {
        // SAFETY: LazyLock guarantees the selected kernel matches CPU features.
        let dists = unsafe { NEAREST_K_KERNEL(&self.rows, query) };
        let mut pairs: Vec<(u8, u8)> = (0..64u8).map(|i| (i, dists[i as usize])).collect();
        pairs.sort_by_key(|&(_, d)| d);
        pairs.truncate(k);
        pairs
    }

    /// Row density: popcount of each row. Sparse rows = abstract; dense = concrete.
    pub fn density(&self) -> [u8; 64] {
        let mut d = [0u8; 64];
        for i in 0..64 {
            d[i] = self.rows[i].count_ones() as u8;
        }
        d
    }

    /// Inter-row Hamming distance matrix (upper triangle).
    /// Returns mean and max distances for uniformity analysis.
    pub fn distance_stats(&self) -> (f64, u8) {
        let mut total = 0u64;
        let mut max_dist = 0u8;
        let mut count = 0u64;

        for i in 0..64 {
            for j in (i + 1)..64 {
                let d = (self.rows[i] ^ self.rows[j]).count_ones() as u8;
                total += d as u64;
                if d > max_dist {
                    max_dist = d;
                }
                count += 1;
            }
        }

        let mean = total as f64 / count as f64;
        (mean, max_dist)
    }
}

// ============================================================================
// MoE Fanout
// ============================================================================

impl HeelPlanes {
    /// MoE gate: query against the 8 expert planes.
    ///
    /// Each HEEL plane is an expert. The query's match against each expert
    /// determines which experts activate and with what strength.
    ///
    /// Runtime dispatch via LazyLock: AVX-512 → AVX2 → scalar.
    #[inline]
    pub fn moe_gate(&self, query: u64, threshold: u8) -> MoeGate {
        // SAFETY: LazyLock guarantees the selected kernel matches CPU features.
        let (active, strength, combined) =
            unsafe { MOE_GATE_KERNEL(&self.planes, query, threshold) };
        MoeGate {
            active,
            strength,
            combined,
        }
    }

    /// Soft MoE: combine expert planes weighted by match strength.
    ///
    /// Returns a u64 where bit `b` is set if the weighted vote exceeds 50%.
    /// Weight = popcount(query AND plane[i]) for active experts.
    #[inline]
    pub fn soft_moe(&self, query: u64, threshold: u8) -> u64 {
        let gate = self.moe_gate(query, threshold);
        let active_count = gate.active.count_ones();

        if active_count == 0 {
            return 0;
        }

        // Weighted majority vote per bit position
        let mut votes = [0u16; 64];
        let mut total_weight = 0u16;

        for i in 0..8 {
            if gate.active & (1 << i) != 0 {
                let weight = gate.strength[i] as u16;
                total_weight += weight;
                for bit in 0..64 {
                    if self.planes[i] & (1u64 << bit) != 0 {
                        votes[bit] += weight;
                    }
                }
            }
        }

        // Majority threshold: > 50% of total weight
        let half = total_weight / 2;
        let mut result = 0u64;
        for bit in 0..64 {
            if votes[bit] > half {
                result |= 1u64 << bit;
            }
        }

        result
    }

    /// Cross-expert interference: Hamming distance between all expert pairs.
    ///
    /// Returns the 28 pairwise distances (8 choose 2).
    /// High mean distance = experts are diverse (good).
    /// Low mean distance = experts are redundant (wasteful).
    pub fn expert_diversity(&self) -> ([u8; 28], f64) {
        let mut dists = [0u8; 28];
        let mut idx = 0;
        let mut total = 0u32;

        for i in 0..8 {
            for j in (i + 1)..8 {
                let d = (self.planes[i] ^ self.planes[j]).count_ones() as u8;
                dists[idx] = d;
                total += d as u32;
                idx += 1;
            }
        }

        let mean = total as f64 / 28.0;
        (dists, mean)
    }
}

// ============================================================================
// Stable Diffusion connection: denoising via palette
// ============================================================================

impl Palette64 {
    /// Single denoising step: project noisy latent onto nearest palette entry.
    ///
    /// This is the BNN equivalent of the cross-attention in a diffusion U-Net:
    /// - `noisy`: the current latent state (64 bits)
    /// - `noise_level`: controls the Gamma threshold (higher = more selective)
    /// - Returns: denoised latent (the best-matching palette row)
    #[inline]
    pub fn denoise_step(&self, noisy: u64, noise_level: u8) -> u64 {
        let result = self.attend(noisy, noise_level);
        self.rows[result.best_idx as usize]
    }

    /// Iterative denoising: apply `steps` rounds of palette projection.
    ///
    /// Each step projects the current state onto the nearest palette entry,
    /// progressively reducing noise. The noise_level decreases linearly.
    pub fn denoise(&self, initial: u64, steps: usize, max_noise: u8) -> u64 {
        let mut state = initial;
        for step in 0..steps {
            // Noise schedule: high threshold early (coarse), low later (fine)
            let noise = max_noise.saturating_sub(((step * max_noise as usize) / steps) as u8);
            state = self.denoise_step(state, noise);
        }
        state
    }

    /// Check if denoising converges: does the state reach a fixed point?
    pub fn convergence_test(&self, initial: u64, max_steps: usize) -> (u64, usize, bool) {
        let mut state = initial;
        for step in 0..max_steps {
            let next = self.denoise_step(state, 0); // gamma=0 = pure nearest lookup
            if next == state {
                return (state, step + 1, true);
            }
            state = next;
        }
        (state, max_steps, false)
    }
}

// ============================================================================
// Palette3D: 8-layer dynamic attention with thinking style modulation
// ============================================================================

/// 8 relationship layers stacked into a 3D attention volume.
///
/// ```text
/// Z: 8 layers    (relationship types / predicates / semirings)
/// Y: 64 rows     (source archetypes, SPO "S")
/// X: 64 columns  (target archetypes, SPO "O")
///
/// 8 × 64 × 64 = 32,768 bits = 4,096 bytes = 4KB = L1 resident
/// ```
///
/// Each layer is a `Palette64`. The Z-axis traversal IS the inference.
#[derive(Debug, Clone)]
#[repr(align(64))]
pub struct Palette3D {
    /// The 8 relationship layers.
    pub layers: [Palette64; 8],
    /// Current thinking style (controls layer gating and combination logic).
    pub style: ThinkingStyle,
    /// Inference step counter (for temporal dynamics).
    pub step: u64,
}

/// Predicate layer indices — the Z-axis semantics.
pub mod predicate {
    pub const CAUSES: usize = 0;
    pub const ENABLES: usize = 1;
    pub const SUPPORTS: usize = 2;
    pub const CONTRADICTS: usize = 3;
    pub const REFINES: usize = 4;
    pub const ABSTRACTS: usize = 5;
    pub const GROUNDS: usize = 6;
    pub const BECOMES: usize = 7;
}

/// Thinking style modulation parameters.
///
/// Each style controls:
/// - Which layers are active (layer_mask: 8-bit)
/// - How layers combine (mode: AND/OR/MAJORITY)
/// - Whether contradiction suppresses or amplifies (contra_mode)
/// - Attention density target (sparsity pressure)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThinkingStyle {
    /// Which of the 8 layers participate (bitmask).
    pub layer_mask: u8,
    /// How active layers are combined.
    pub combine: CombineMode,
    /// How the contradiction layer (3) interacts.
    pub contra: ContraMode,
    /// Target density for dynamic pruning (0.0 = maximally sparse, 1.0 = dense).
    pub density_target: f32,
    /// Name for diagnostics.
    pub name: &'static str,
}

/// How multiple active layers are combined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CombineMode {
    /// OR: any active layer can contribute (expansive, creative).
    Union,
    /// AND: all active layers must agree (conservative, analytical).
    Intersection,
    /// Majority: >50% of active layers must agree (balanced).
    Majority,
    /// Weighted: layer weights from MoE gate strength.
    Weighted,
}

/// How the contradiction layer (predicate::CONTRADICTS) modulates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContraMode {
    /// AND NOT: contradiction kills connections (default for analytical).
    Suppress,
    /// Contradiction is ignored (default for creative — let it flow).
    Ignore,
    /// Contradiction INVERTS: blocked paths become active (divergent thinking).
    Invert,
    /// Contradiction slows but doesn't kill (adds to "tension" metric).
    Tension,
}

/// Pre-defined thinking styles.
impl ThinkingStyle {
    /// Analytical: tight focus, all layers must agree, contradiction kills.
    pub const ANALYTICAL: Self = Self {
        layer_mask: 0b0111_0111, // CAUSES, ENABLES, SUPPORTS, REFINES, ABSTRACTS, GROUNDS
        combine: CombineMode::Intersection,
        contra: ContraMode::Suppress,
        density_target: 0.05,
        name: "analytical",
    };

    /// Creative: wide association, any layer contributes, contradiction ignored.
    pub const CREATIVE: Self = Self {
        layer_mask: 0b1111_1111, // all layers
        combine: CombineMode::Union,
        contra: ContraMode::Ignore,
        density_target: 0.40,
        name: "creative",
    };

    /// Focused: single causal chain, strict contradiction.
    pub const FOCUSED: Self = Self {
        layer_mask: 0b0000_0011, // only CAUSES + ENABLES
        combine: CombineMode::Intersection,
        contra: ContraMode::Suppress,
        density_target: 0.02,
        name: "focused",
    };

    /// Integrative: cross-layer synthesis, majority vote.
    pub const INTEGRATIVE: Self = Self {
        layer_mask: 0b1111_1111,
        combine: CombineMode::Majority,
        contra: ContraMode::Tension,
        density_target: 0.15,
        name: "integrative",
    };

    /// Divergent: contradiction as fuel — blocked paths become active.
    pub const DIVERGENT: Self = Self {
        layer_mask: 0b1000_1001, // CAUSES + CONTRADICTS + BECOMES
        combine: CombineMode::Union,
        contra: ContraMode::Invert,
        density_target: 0.30,
        name: "divergent",
    };

    /// Meta: observes the observation — ABSTRACTS + GROUNDS + BECOMES.
    pub const META: Self = Self {
        layer_mask: 0b1110_0000,
        combine: CombineMode::Majority,
        contra: ContraMode::Tension,
        density_target: 0.10,
        name: "meta",
    };
}

/// Result of a single inference step.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Combined attention mask for this query (64 bits = which targets fire).
    pub attention: u64,
    /// Contradiction tension (how many contradicted bits were encountered).
    pub tension: u32,
    /// Number of layers that contributed.
    pub active_layers: u8,
    /// Number of new connections discovered (dynamic mask growth).
    pub new_connections: u32,
    /// Current density after inference.
    pub density: f32,
}

impl Palette3D {
    /// Create from 8 individual palette layers.
    pub fn new(layers: [Palette64; 8], style: ThinkingStyle) -> Self {
        Self {
            layers,
            style,
            step: 0,
        }
    }

    /// Create from HeelPlanes: same expansion for all layers (then differentiate).
    pub fn from_heels_uniform(heels: &HeelPlanes, style: ThinkingStyle) -> Self {
        let base = heels.expand();
        Self {
            layers: [base; 8],
            style,
            step: 0,
        }
    }

    /// Switch thinking style. Does NOT reset the palette state.
    pub fn modulate(&mut self, style: ThinkingStyle) {
        self.style = style;
    }

    /// Total density across all active layers.
    pub fn density(&self) -> f32 {
        let mut total_bits = 0u32;
        let mut total_possible = 0u32;
        for z in 0..8 {
            if self.style.layer_mask & (1 << z) == 0 {
                continue;
            }
            for row in &self.layers[z].rows {
                total_bits += row.count_ones();
            }
            total_possible += 64 * 64;
        }
        if total_possible == 0 {
            0.0
        } else {
            total_bits as f32 / total_possible as f32
        }
    }

    // ── Core inference ─────────────────────────────────────────────────

    /// Single inference step: fan-out from query_row through active layers.
    pub fn infer(&mut self, query_row: usize) -> InferenceResult {
        let block_row = query_row.min(63);
        self.step += 1;

        // 1. Gather attention from each active layer
        let mut per_layer = [0u64; 8];
        let mut active_count = 0u8;
        for z in 0..8 {
            if self.style.layer_mask & (1 << z) == 0 {
                continue;
            }
            per_layer[z] = self.layers[z].rows[block_row];
            active_count += 1;
        }

        // 2. Combine according to thinking style
        let combined = self.combine_layers(&per_layer, active_count);

        // 3. Apply contradiction modulation
        let contra_mask = per_layer[predicate::CONTRADICTS];
        let (attention, tension) = self.apply_contradiction(combined, contra_mask);

        // 4. Dynamic mask growth: deduce new connections
        let new_connections = self.deduce_and_grow(block_row, &per_layer);

        let density = self.density();

        InferenceResult {
            attention,
            tension,
            active_layers: active_count,
            new_connections,
            density,
        }
    }

    /// Combine layer outputs according to current style.
    fn combine_layers(&self, per_layer: &[u64; 8], active_count: u8) -> u64 {
        match self.style.combine {
            CombineMode::Union => {
                let mut result = 0u64;
                for z in 0..8 {
                    if self.style.layer_mask & (1 << z) != 0 {
                        result |= per_layer[z];
                    }
                }
                result
            }
            CombineMode::Intersection => {
                let mut result = u64::MAX;
                let mut first = true;
                for z in 0..8 {
                    if self.style.layer_mask & (1 << z) != 0 {
                        if first {
                            result = per_layer[z];
                            first = false;
                        } else {
                            result &= per_layer[z];
                        }
                    }
                }
                if first { 0 } else { result }
            }
            CombineMode::Majority => {
                // Per-bit vote: set if >50% of active layers agree
                let threshold = (active_count / 2) + 1;
                let mut result = 0u64;
                for bit in 0..64 {
                    let mut votes = 0u8;
                    for z in 0..8 {
                        if self.style.layer_mask & (1 << z) != 0
                            && per_layer[z] & (1u64 << bit) != 0
                        {
                            votes += 1;
                        }
                    }
                    if votes >= threshold {
                        result |= 1u64 << bit;
                    }
                }
                result
            }
            CombineMode::Weighted => {
                // Weight by layer position: lower layers have more weight
                let mut votes = [0u16; 64];
                let mut total_weight = 0u16;
                for z in 0..8 {
                    if self.style.layer_mask & (1 << z) == 0 {
                        continue;
                    }
                    let weight = (8 - z) as u16; // CAUSES=8, BECOMES=1
                    total_weight += weight;
                    for bit in 0..64 {
                        if per_layer[z] & (1u64 << bit) != 0 {
                            votes[bit] += weight;
                        }
                    }
                }
                let half = total_weight / 2;
                let mut result = 0u64;
                for bit in 0..64 {
                    if votes[bit] > half {
                        result |= 1u64 << bit;
                    }
                }
                result
            }
        }
    }

    /// Apply contradiction layer modulation.
    fn apply_contradiction(&self, combined: u64, contra_mask: u64) -> (u64, u32) {
        let tension = (combined & contra_mask).count_ones();

        let attention = match self.style.contra {
            ContraMode::Suppress => combined & !contra_mask,
            ContraMode::Ignore => combined,
            ContraMode::Invert => combined ^ contra_mask,
            ContraMode::Tension => {
                // Keep the bits but record the tension
                // High tension = the result is contested
                combined
            }
        };

        (attention, tension)
    }

    // ── Deduction engine ───────────────────────────────────────────────

    /// Boolean SpMM deduction: follow active connections transitively.
    ///
    /// "A CAUSES B" AND "B ENABLES C" → discover "A indirectly ENABLES C"
    /// New connections are written into the ENABLES layer (dynamic growth).
    fn deduce_and_grow(&mut self, block_row: usize, per_layer: &[u64; 8]) -> u32 {
        let mut new = 0u32;

        // Deduction rule 1: CAUSES × ENABLES → expand ENABLES
        if self.style.layer_mask & (1 << predicate::CAUSES) != 0
            && self.style.layer_mask & (1 << predicate::ENABLES) != 0
        {
            let caused = per_layer[predicate::CAUSES];
            let mut transitive_enables = 0u64;
            let mut bits = caused;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if j < 64 {
                    transitive_enables |= self.layers[predicate::ENABLES].rows[j];
                }
            }
            // New bits = bits in transitive result that weren't already there
            let fresh = transitive_enables & !self.layers[predicate::ENABLES].rows[block_row];
            new += fresh.count_ones();
            self.layers[predicate::ENABLES].rows[block_row] |= transitive_enables;
        }

        // Deduction rule 2: SUPPORTS × SUPPORTS → expand GROUNDS
        if self.style.layer_mask & (1 << predicate::SUPPORTS) != 0
            && self.style.layer_mask & (1 << predicate::GROUNDS) != 0
        {
            let supported = per_layer[predicate::SUPPORTS];
            let mut mutual_support = 0u64;
            let mut bits = supported;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if j < 64 {
                    // If j also supports things the query supports → grounded
                    mutual_support |=
                        self.layers[predicate::SUPPORTS].rows[j] & supported;
                }
            }
            let fresh = mutual_support & !self.layers[predicate::GROUNDS].rows[block_row];
            new += fresh.count_ones();
            self.layers[predicate::GROUNDS].rows[block_row] |= mutual_support;
        }

        // Deduction rule 3: REFINES chains collapse
        if self.style.layer_mask & (1 << predicate::REFINES) != 0 {
            let refined = per_layer[predicate::REFINES];
            let mut chain = 0u64;
            let mut bits = refined;
            while bits != 0 {
                let j = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                if j < 64 {
                    chain |= self.layers[predicate::REFINES].rows[j];
                }
            }
            let fresh = chain & !self.layers[predicate::REFINES].rows[block_row];
            new += fresh.count_ones();
            self.layers[predicate::REFINES].rows[block_row] |= chain;
        }

        new
    }

    // ── Multi-step reasoning ───────────────────────────────────────────

    /// Run N inference steps, allowing deduction to propagate.
    /// Returns the final attention mask and accumulated tension.
    pub fn reason(&mut self, query_row: usize, steps: usize) -> InferenceResult {
        let mut last = InferenceResult {
            attention: 0,
            tension: 0,
            active_layers: 0,
            new_connections: 0,
            density: 0.0,
        };

        for _ in 0..steps {
            last = self.infer(query_row);
            if last.new_connections == 0 {
                break; // fixed point — deduction converged
            }
        }

        last
    }

    /// Style transition: smoothly shift between two styles over N steps.
    ///
    /// Useful for: analytical → creative (when stuck),
    /// creative → focused (when solution emerges).
    pub fn transition(
        &mut self,
        from: ThinkingStyle,
        to: ThinkingStyle,
        query_row: usize,
        steps: usize,
    ) -> Vec<InferenceResult> {
        let mut results = Vec::with_capacity(steps);

        for step in 0..steps {
            let t = step as f32 / steps.max(1) as f32;

            // Interpolate layer mask: union of both, gradually shifting
            let mask = if t < 0.5 {
                from.layer_mask | (to.layer_mask & self.random_mask(t))
            } else {
                to.layer_mask | (from.layer_mask & self.random_mask(1.0 - t))
            };

            // Combine mode snaps at midpoint
            let combine = if t < 0.5 { from.combine } else { to.combine };
            let contra = if t < 0.5 { from.contra } else { to.contra };
            let density_target =
                from.density_target * (1.0 - t) + to.density_target * t;

            self.style = ThinkingStyle {
                layer_mask: mask,
                combine,
                contra,
                density_target,
                name: if t < 0.5 { from.name } else { to.name },
            };

            results.push(self.infer(query_row));
        }

        results
    }

    /// Deterministic "random" mask based on step counter (no actual RNG).
    /// Uses golden ratio fractional parts for quasi-random bit selection.
    fn random_mask(&self, density: f32) -> u8 {
        let threshold = (density * 8.0) as u32;
        let mut mask = 0u8;
        for i in 0..8 {
            // Golden ratio hash: each bit has pseudo-independent activation
            let hash = ((self.step.wrapping_mul(2654435761)) >> (i * 4)) & 0x7;
            if hash < threshold as u64 {
                mask |= 1 << i;
            }
        }
        mask
    }

    // ── Diagnostics ────────────────────────────────────────────────────

    /// Snapshot of layer-wise density for style analysis.
    pub fn layer_densities(&self) -> [f32; 8] {
        let mut d = [0.0f32; 8];
        for z in 0..8 {
            let bits: u32 = self.layers[z].rows.iter().map(|r| r.count_ones()).sum();
            d[z] = bits as f32 / 4096.0;
        }
        d
    }

    /// Contradiction ratio: how much of the current attention is contested.
    pub fn contradiction_ratio(&self, query_row: usize) -> f32 {
        let block_row = query_row.min(63);
        let combined = self.infer_readonly(block_row);
        let contra = self.layers[predicate::CONTRADICTS].rows[block_row];
        let contested = (combined & contra).count_ones();
        let total = combined.count_ones();
        if total == 0 {
            0.0
        } else {
            contested as f32 / total as f32
        }
    }

    /// Read-only inference (no mutation) for diagnostics.
    fn infer_readonly(&self, block_row: usize) -> u64 {
        let mut per_layer = [0u64; 8];
        let mut active_count = 0u8;
        for z in 0..8 {
            if self.style.layer_mask & (1 << z) == 0 {
                continue;
            }
            per_layer[z] = self.layers[z].rows[block_row];
            active_count += 1;
        }
        self.combine_layers(&per_layer, active_count)
    }

    /// Memory footprint.
    pub fn size_bytes() -> usize {
        8 * 512 // 8 layers × 512 bytes
    }
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// Sparse256: 256×256 interaction matrix via CLAM triangle inequality
// ============================================================================

pub mod sparse256 {
    use super::Palette64;

    /// Pre-computed leaf cluster data for palette construction.
    /// p64 does NOT depend on ndarray/ClamTree — this struct accepts
    /// pre-computed distances and radii from any source.
    #[derive(Debug, Clone)]
    pub struct LeafCluster {
        /// Cluster index (0..255).
        pub id: u8,
        /// Cluster radius (Hamming distance from center to farthest member).
        pub radius: u64,
        /// HEEL group (0..7): which expert this cluster belongs to.
        pub heel_group: u8,
    }

    /// Pairwise center-to-center distances between leaf clusters.
    /// Stored as flat array: `dist[i * n + j]` for i,j in 0..n.
    pub struct PairwiseDistances {
        pub n: usize,
        pub dists: Vec<u64>,
    }

    impl PairwiseDistances {
        pub fn new(n: usize) -> Self {
            Self {
                n,
                dists: vec![0u64; n * n],
            }
        }

        #[inline]
        pub fn get(&self, i: usize, j: usize) -> u64 {
            self.dists[i * self.n + j]
        }

        #[inline]
        pub fn set(&mut self, i: usize, j: usize, d: u64) {
            self.dists[i * self.n + j] = d;
            self.dists[j * self.n + i] = d;
        }
    }

    /// Build a Palette64 from CLAM leaf cluster bounds.
    ///
    /// Triangle inequality pruning:
    /// ```text
    /// Two clusters A, B interact if:  d(c_A, c_B) <= r_A + r_B
    /// Two clusters DON'T interact if: d(c_A, c_B) >  r_A + r_B
    /// ```
    ///
    /// The 256 leaves are grouped into 64 blocks of 4.
    /// Block (I, J) is set to 1 if ANY leaf-pair across blocks can interact.
    /// This is conservative: zero false negatives, possible false positives.
    pub fn from_clam_leaves(
        leaves: &[LeafCluster],
        distances: &PairwiseDistances,
    ) -> (Palette64, SparsityStats) {
        let n = leaves.len().min(256);
        let n_blocks = (n + 3) / 4; // ceil(n/4), max 64

        let mut palette = Palette64::zero();
        let mut interactions = 0u64;
        let mut pruned = 0u64;

        for bi in 0..n_blocks {
            for bj in 0..n_blocks {
                let mut block_active = false;

                // Check all 4×4 pairs across block boundaries
                for li in 0..4 {
                    let i = bi * 4 + li;
                    if i >= n {
                        break;
                    }
                    for lj in 0..4 {
                        let j = bj * 4 + lj;
                        if j >= n {
                            break;
                        }

                        let d = distances.get(i, j);
                        let r_sum = leaves[i].radius.saturating_add(leaves[j].radius);

                        if d <= r_sum {
                            // Clusters CAN interact — set block bit
                            block_active = true;
                            interactions += 1;
                        } else {
                            pruned += 1;
                        }
                    }
                }

                if block_active && bi < 64 && bj < 64 {
                    palette.rows[bi] |= 1u64 << bj;
                }
            }
        }

        let total = (n_blocks * n_blocks) as u64;
        let stats = SparsityStats {
            total_blocks: total,
            active_blocks: interactions,
            pruned_blocks: pruned,
            density: if total > 0 {
                palette.rows.iter().map(|r| r.count_ones() as u64).sum::<u64>() as f64
                    / (n_blocks * n_blocks) as f64
            } else {
                0.0
            },
        };

        (palette, stats)
    }

    /// Statistics about CLAM-derived sparsity.
    #[derive(Debug, Clone)]
    pub struct SparsityStats {
        pub total_blocks: u64,
        pub active_blocks: u64,
        pub pruned_blocks: u64,
        /// Fraction of 64×64 bits that are set (0.0 .. 1.0).
        pub density: f64,
    }

    // ── BSR SpMV ───────────────────────────────────────────────────────

    impl Palette64 {
        /// Interpret as 256×256 sparse matrix in BSR format (4×4 blocks).
        /// Perform SpMV: y = A × x.
        ///
        /// Each set bit at (block_row, block_col) represents a non-zero 4×4 block.
        /// The block content is the identity (uniform weight) by default.
        /// For weighted blocks, use `spmv_256_weighted`.
        pub fn spmv_256(&self, x: &[f32; 256], y: &mut [f32; 256]) {
            y.fill(0.0);
            for block_row in 0..64 {
                let mask = self.rows[block_row];
                if mask == 0 {
                    continue;
                }

                let base_row = block_row * 4;
                let mut bits = mask;
                while bits != 0 {
                    let block_col = bits.trailing_zeros() as usize;
                    bits &= bits - 1;

                    let base_col = block_col * 4;
                    // 4×4 identity block: y[r] += x[c] for matching sub-indices
                    for k in 0..4 {
                        if base_row + k < 256 && base_col + k < 256 {
                            y[base_row + k] += x[base_col + k];
                        }
                    }
                }
            }
        }

        /// SpMV with per-block weights from CLAM LFD.
        ///
        /// `block_weights[block_row][block_col]` scales the 4×4 contribution.
        /// Weights typically come from inverse LFD (local fractal dimension):
        /// high LFD clusters get lower weight (they're spread out).
        pub fn spmv_256_weighted(
            &self,
            x: &[f32; 256],
            y: &mut [f32; 256],
            block_weights: &[[f32; 64]; 64],
        ) {
            y.fill(0.0);
            for block_row in 0..64 {
                let mask = self.rows[block_row];
                if mask == 0 {
                    continue;
                }

                let base_row = block_row * 4;
                let mut bits = mask;
                while bits != 0 {
                    let block_col = bits.trailing_zeros() as usize;
                    bits &= bits - 1;

                    let w = block_weights[block_row][block_col];
                    let base_col = block_col * 4;
                    for k in 0..4 {
                        if base_row + k < 256 && base_col + k < 256 {
                            y[base_row + k] += x[base_col + k] * w;
                        }
                    }
                }
            }
        }

        /// Count non-zero blocks (set bits). Each block = 4×4 = 16 entries.
        pub fn nnz_blocks(&self) -> u32 {
            self.rows.iter().map(|r| r.count_ones()).sum()
        }

        /// Effective FLOPs for SpMV vs dense 256×256.
        pub fn flop_ratio(&self) -> f64 {
            let sparse_flops = self.nnz_blocks() as f64 * 4.0; // 4 ops per identity block
            let dense_flops = 256.0 * 256.0;
            sparse_flops / dense_flops
        }
    }

    // ── HHTL Cascade via Palette ──────────────────────────────────────

    /// HHTL cascade levels mapped onto the palette.
    ///
    /// ```text
    /// HEEL: which 8×8 super-block?     (row/8, col/8) → 8×8 grid
    /// HIP:  which 64×64 bit is set?    (the palette itself)
    /// TWIG: which 4×4 sub-position?    (within the active block)
    /// LEAF: exact distance/weight       (ZeckF8 or BF16)
    /// ```
    #[derive(Debug, Clone, Copy)]
    pub struct HhtlAddress {
        /// Super-block position (0..7, 0..7) — the HEEL level.
        pub heel: (u8, u8),
        /// Block position within super-block (0..7, 0..7) — the HIP level.
        pub hip: (u8, u8),
        /// Sub-position within 4×4 block (0..3, 0..3) — the TWIG level.
        pub twig: (u8, u8),
        /// Fine-grain value — the LEAF level.
        pub leaf: u8,
    }

    impl HhtlAddress {
        /// Decompose a 256×256 matrix coordinate into HHTL levels.
        pub fn from_256(row: u8, col: u8) -> Self {
            Self {
                heel: (row / 32, col / 32),
                hip: ((row / 4) % 8, (col / 4) % 8),
                twig: (row % 4, col % 4),
                leaf: 0,
            }
        }

        /// Check if this address is active in the palette (HIP level).
        pub fn is_active(&self, palette: &Palette64) -> bool {
            let block_row = self.heel.0 as usize * 8 + self.hip.0 as usize;
            let block_col = self.heel.1 as usize * 8 + self.hip.1 as usize;
            if block_row >= 64 || block_col >= 64 {
                return false;
            }
            palette.rows[block_row] & (1u64 << block_col) != 0
        }

        /// Flatten to 256×256 coordinate.
        pub fn to_256(&self) -> (u8, u8) {
            let row = self.heel.0 * 32 + self.hip.0 * 4 + self.twig.0;
            let col = self.heel.1 * 32 + self.hip.1 * 4 + self.twig.1;
            (row, col)
        }
    }

    /// HHTL cascade search: early termination via palette sparsity.
    ///
    /// Instead of scanning all 256 entries, check the palette first:
    /// - Skip entire 32-element HEEL blocks if the 8×8 super-block is empty
    /// - Skip 4-element TWIG blocks if the palette bit is 0
    /// - Only compute exact distance for active palette entries
    ///
    /// `score_fn`: callback that computes the actual score for a (row, col) pair.
    /// This is where the LEAF level lives — LanceDB vector search, DistanceMatrix
    /// lookup, or BF16 dot product. The cascade doesn't know or care which.
    pub fn hhtl_cascade_search<F: Fn(u8, u8) -> f32>(
        palette: &Palette64,
        query_row: u8,
        scores: &mut [f32; 256],
        score_fn: F,
    ) -> usize {
        let heel_row = query_row / 32;
        let hip_row = (query_row / 4) % 8;
        let block_row = heel_row as usize * 8 + hip_row as usize;

        if block_row >= 64 {
            return 0;
        }

        let mask = palette.rows[block_row];
        if mask == 0 {
            return 0;
        }

        let mut computed = 0usize;
        let mut bits = mask;
        while bits != 0 {
            let block_col = bits.trailing_zeros() as usize;
            bits &= bits - 1;

            let base_col = block_col * 4;
            for k in 0..4 {
                let col = base_col + k;
                if col < 256 {
                    scores[col] = score_fn(query_row, col as u8);
                    computed += 1;
                }
            }
        }

        computed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_heels() -> HeelPlanes {
        // 8 diverse expert patterns
        HeelPlanes::new([
            0xAAAA_AAAA_AAAA_AAAA, // alternating 10
            0x5555_5555_5555_5555, // alternating 01
            0xFFFF_0000_FFFF_0000, // 16-bit blocks
            0x0000_FFFF_0000_FFFF, // inverse blocks
            0xFF00_FF00_FF00_FF00, // 8-bit blocks
            0x00FF_00FF_00FF_00FF, // inverse 8-bit
            0xF0F0_F0F0_F0F0_F0F0, // 4-bit blocks
            0x0F0F_0F0F_0F0F_0F0F, // inverse 4-bit
        ])
    }

    // ── Basic construction ─────────────────────────────────────────────

    #[test]
    fn palette_from_heels() {
        let heels = make_test_heels();
        let palette = heels.expand();

        // 64 rows should be populated
        let non_zero = palette.rows.iter().filter(|&&r| r != 0).count();
        assert_eq!(non_zero, 64, "All 64 rows should be non-zero");

        // Row 0 = HEEL[0] rotated by 0
        assert_eq!(palette.rows[0], heels.planes[0]);
        // Row 8 = HEEL[0] rotated by GOLDEN_SHIFT_64
        assert_eq!(
            palette.rows[8],
            heels.planes[0].rotate_left(GOLDEN_SHIFT_64)
        );

        eprintln!("Palette constructed: 64 rows × 64 bits = {} bytes", 64 * 8);
    }

    #[test]
    fn palette_from_clam_seed() {
        let mut seed = [0i8; 34];
        seed[0] = 42; // HEEL
        seed[33] = 7; // GAMMA
        for i in 1..33 {
            seed[i] = (i as i8).wrapping_mul(17).wrapping_add(3);
        }

        let heels = HeelPlanes::from_clam_seed(&seed);
        let palette = heels.expand();

        let non_zero = palette.rows.iter().filter(|&&r| r != 0).count();
        assert!(non_zero > 0, "Palette should have non-zero rows");

        eprintln!("CLAM seed → palette: {non_zero}/64 non-zero rows");
    }

    // ── Attention ──────────────────────────────────────────────────────

    #[test]
    fn bnn_attention_basic() {
        let heels = make_test_heels();
        let palette = heels.expand();

        // Query that matches HEEL[0] (alternating 10) perfectly
        let query = 0xAAAA_AAAA_AAAA_AAAA;
        let result = palette.attend(query, 16);

        eprintln!("Query: {:016X}", query);
        eprintln!("Best match: row {} (distance {})", result.best_idx, result.distance);
        eprintln!("Fires: {} rows exceed gamma=16", result.fires.count_ones());

        // Row 0 = 0xAAAA... has 32 ones. Overlap = popcount(0xAAAA & 0xAAAA) = 32.
        // Distance = 64 - 32 = 32 (overlap-based, not Hamming).
        assert_eq!(result.best_idx, 0);
        assert_eq!(result.scores[0], 32, "Perfect overlap with 32-bit pattern");
        // Row 1 = 0x5555 (complement) should have zero overlap
        assert_eq!(result.scores[1], 0, "Complement pattern has zero overlap");
    }

    #[test]
    fn bnn_attention_noisy() {
        let heels = make_test_heels();
        let palette = heels.expand();

        // Query = HEEL[0] with 8 low bits flipped
        let query = 0xAAAA_AAAA_AAAA_AAAA ^ 0xFF;
        let result = palette.attend(query, 16);

        eprintln!("Noisy query (8 bits flipped): score={}, distance={}",
            result.scores[result.best_idx as usize], result.distance);
        // Original: 32 matching bits. Flip 8 bits: ~4 matches lost, ~4 false matches gained.
        // Row 0 should still be best or near-best.
        assert!(
            result.scores[0] >= 24,
            "Row 0 should still score high despite noise (score={})", result.scores[0]
        );
    }

    // ── MoE Fanout ─────────────────────────────────────────────────────

    #[test]
    fn moe_gate_basic() {
        let heels = make_test_heels();

        // Query matching HEEL[0] pattern
        let gate = heels.moe_gate(0xAAAA_AAAA_AAAA_AAAA, 20);

        eprintln!("MoE gate:");
        for i in 0..8 {
            let active = if gate.active & (1 << i) != 0 {
                "ACTIVE"
            } else {
                "      "
            };
            eprintln!("  Expert {i}: strength={:2} {active}", gate.strength[i]);
        }
        eprintln!("Active experts: {}", gate.active.count_ones());
        eprintln!("Combined output: {:016X}", gate.combined);

        // HEEL[0] should be the strongest match
        assert!(gate.strength[0] >= gate.strength[1]);
        assert!(gate.active & 1 != 0, "Expert 0 should be active");
    }

    #[test]
    fn soft_moe_majority_vote() {
        let heels = make_test_heels();

        let query = 0xAAAA_AAAA_AAAA_AAAA;
        let hard = heels.moe_gate(query, 20).combined;
        let soft = heels.soft_moe(query, 20);

        eprintln!("Hard MoE (OR): {:016X}", hard);
        eprintln!("Soft MoE (vote): {:016X}", soft);
        eprintln!(
            "Hard density: {}, Soft density: {}",
            hard.count_ones(),
            soft.count_ones()
        );

        // Soft should be sparser than hard (majority vs OR)
        assert!(
            soft.count_ones() <= hard.count_ones(),
            "Soft MoE should be sparser than hard MoE"
        );
    }

    // ── Expert diversity ───────────────────────────────────────────────

    #[test]
    fn expert_diversity_check() {
        let heels = make_test_heels();
        let (dists, mean) = heels.expert_diversity();

        eprintln!("Expert diversity:");
        eprintln!("  Mean pairwise Hamming: {mean:.1}");
        eprintln!("  Min: {}", dists.iter().min().unwrap());
        eprintln!("  Max: {}", dists.iter().max().unwrap());

        // With our orthogonal test patterns, diversity should be high
        assert!(
            mean > 20.0,
            "Expert diversity should be high for orthogonal patterns"
        );
    }

    // ── Palette statistics ─────────────────────────────────────────────

    #[test]
    fn distance_stats() {
        let heels = make_test_heels();
        let palette = heels.expand();
        let (mean_dist, max_dist) = palette.distance_stats();

        eprintln!("Palette inter-row stats:");
        eprintln!("  Mean Hamming distance: {mean_dist:.2}");
        eprintln!("  Max Hamming distance:  {max_dist}");

        // Ideal: mean ≈ 32 (half of 64 = maximally dispersed)
        assert!(
            mean_dist > 20.0,
            "Palette rows should be well-dispersed (mean={mean_dist})"
        );
    }

    #[test]
    fn golden_shift_coprime() {
        // Verify gcd(39, 64) = 1 — full coverage
        assert_eq!(gcd(GOLDEN_SHIFT_64 as usize, 64), 1);

        // Verify that 8 rotations of any non-zero pattern produce 8 distinct values
        let pattern = 0xDEAD_BEEF_CAFE_BABEu64;
        let mut seen = std::collections::HashSet::new();
        for octave in 0..8u32 {
            let rotated = pattern.rotate_left(octave * GOLDEN_SHIFT_64);
            assert!(seen.insert(rotated), "Octave {octave} produced duplicate");
        }

        eprintln!("Golden shift {GOLDEN_SHIFT_64}: coprime with 64, 8 unique rotations verified");
    }

    // ── Denoising / Stable Diffusion ───────────────────────────────────

    #[test]
    fn denoising_convergence() {
        let heels = make_test_heels();
        let palette = heels.expand();

        // Start from random noise
        let noisy = 0x1234_5678_9ABC_DEF0u64;
        let (final_state, steps, converged) = palette.convergence_test(noisy, 100);

        eprintln!("Denoising convergence:");
        eprintln!("  Input:     {:016X}", noisy);
        eprintln!("  Output:    {:016X}", final_state);
        eprintln!("  Steps:     {steps}");
        eprintln!("  Converged: {converged}");

        // Should converge to a fixed point (a palette row)
        assert!(converged, "Denoising should converge to a fixed point");
        assert!(steps <= 3, "Should converge quickly (steps={steps})");

        // Final state should be exactly a palette row
        let is_palette_row = palette.rows.contains(&final_state);
        assert!(
            is_palette_row,
            "Final state should be a palette entry"
        );
    }

    #[test]
    fn denoising_schedule() {
        let heels = make_test_heels();
        let palette = heels.expand();

        let noisy = 0x1234_5678_9ABC_DEF0u64;
        let denoised = palette.denoise(noisy, 10, 32);

        eprintln!("Scheduled denoising:");
        eprintln!("  Input:    {:016X}", noisy);
        eprintln!("  Output:   {:016X}", denoised);

        // Output should be a palette row (fully denoised)
        let is_row = palette.rows.contains(&denoised);
        assert!(is_row, "Denoised output should be a palette entry");
    }

    // ── Cache/memory analysis ──────────────────────────────────────────

    #[test]
    fn memory_layout() {
        let palette = Palette64::zero();
        let size = std::mem::size_of_val(&palette);
        let align = std::mem::align_of_val(&palette);

        eprintln!("Palette64 memory:");
        eprintln!("  Size:  {} bytes ({} cache lines)", size, size / 64);
        eprintln!("  Align: {} bytes", align);
        eprintln!("  L1 fit: {} (< 32KB)", size < 32768);

        assert_eq!(size, 512, "Must be exactly 512 bytes");
        assert_eq!(align, 64, "Must be cache-line aligned");
    }

    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 {
            a
        } else {
            gcd(b, a % b)
        }
    }

    // ── Sparse256: CLAM-derived palette ────────────────────────────────

    #[test]
    fn sparse256_from_clam_leaves() {
        use super::sparse256::*;

        // 256 leaves with known radii and distances
        let mut leaves: Vec<LeafCluster> = (0..256)
            .map(|i| LeafCluster {
                id: i as u8,
                radius: 10, // uniform radius
                heel_group: (i / 32) as u8,
            })
            .collect();

        // Pairwise distances: clusters in same HEEL group are close,
        // clusters across groups are far apart.
        let mut distances = PairwiseDistances::new(256);
        for i in 0..256 {
            for j in (i + 1)..256 {
                let same_group = (i / 32) == (j / 32);
                let d = if same_group { 5 } else { 100 }; // 5 < 10+10, 100 > 10+10
                distances.set(i, j, d);
            }
        }

        let (palette, stats) = from_clam_leaves(&leaves, &distances);

        eprintln!("\n=== Sparse256 from CLAM leaves ===");
        eprintln!("Total blocks:  {}", stats.total_blocks);
        eprintln!("Active blocks: {}", stats.active_blocks);
        eprintln!("Pruned blocks: {}", stats.pruned_blocks);
        eprintln!("Density:       {:.4}", stats.density);
        eprintln!("NNZ blocks:    {}", palette.nnz_blocks());
        eprintln!("FLOP ratio:    {:.4}", palette.flop_ratio());

        // 8 HEEL groups × 8 blocks each = 64 blocks
        // Intra-group: distance=5, r+r=20 → 5<=20 → interact (1)
        // Inter-group: distance=100, r+r=20 → 100>20 → pruned (0)
        // So only 8 diagonal blocks of 8×8 should be active = 512 bits of 4096
        assert!(
            stats.density < 0.20,
            "Density should be low due to triangle inequality pruning"
        );
    }

    #[test]
    fn spmv_256_basic() {
        use super::sparse256::*;

        let heels = make_test_heels();
        let palette = heels.expand();

        let mut x = [0.0f32; 256];
        let mut y = [0.0f32; 256];
        // Set input to 1.0 everywhere
        x.fill(1.0);

        palette.spmv_256(&x, &mut y);

        // y should be non-zero where palette rows are non-zero
        let non_zero_y = y.iter().filter(|&&v| v > 0.0).count();
        eprintln!("SpMV: {} non-zero entries in y (of 256)", non_zero_y);

        assert!(non_zero_y > 0, "SpMV should produce non-zero output");
    }

    #[test]
    fn hhtl_address_roundtrip() {
        use super::sparse256::*;

        for row in [0u8, 31, 32, 127, 128, 255] {
            for col in [0u8, 15, 63, 128, 200, 255] {
                let addr = HhtlAddress::from_256(row, col);
                let (r, c) = addr.to_256();
                assert_eq!(
                    (r, c),
                    (row, col),
                    "HHTL roundtrip failed for ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn hhtl_cascade_pruning() {
        use super::sparse256::*;

        // Sparse palette: only diagonal blocks active
        let mut palette = Palette64::zero();
        for i in 0..64 {
            palette.rows[i] = 1u64 << i; // only self-block
        }

        let mut scores = [0.0f32; 256];
        let score_fn = |row: u8, col: u8| -> f32 { 1.0 - (row as f32 - col as f32).abs() / 256.0 };
        let computed = hhtl_cascade_search(&palette, 0, &mut scores, &score_fn);

        eprintln!("HHTL cascade: computed {} of 256 scores", computed);

        // Only 4 scores should be computed (one block of 4)
        assert_eq!(computed, 4, "Should only compute active block entries");

        // Row 128 → block_row = 128/32*8 + (128/4)%8 = 4*8 + 0 = 32
        let computed2 = hhtl_cascade_search(&palette, 128, &mut scores, &score_fn);
        assert_eq!(computed2, 4);
    }

    // ── Palette3D: thinking style modulation ───────────────────────────

    fn make_test_palette3d() -> Palette3D {
        let heels = make_test_heels();
        // Start all layers from same expansion, then differentiate
        let mut p3d = Palette3D::from_heels_uniform(&heels, ThinkingStyle::ANALYTICAL);

        // Make CONTRADICTS layer sparser — only block-diagonal
        p3d.layers[predicate::CONTRADICTS] = Palette64::zero();
        for i in 0..64 {
            p3d.layers[predicate::CONTRADICTS].rows[i] = 1u64 << ((i + 32) % 64);
        }

        p3d
    }

    #[test]
    fn palette3d_memory_layout() {
        let size = Palette3D::size_bytes();
        eprintln!("Palette3D: {} bytes ({} KB, {} cache lines)", size, size / 1024, size / 64);
        assert_eq!(size, 4096, "Must be exactly 4KB");
    }

    #[test]
    fn thinking_style_analytical() {
        let mut p3d = make_test_palette3d();
        p3d.modulate(ThinkingStyle::ANALYTICAL);

        let result = p3d.infer(0);

        eprintln!("\n=== Analytical Style ===");
        eprintln!("Attention:    {:016X} ({} bits)", result.attention, result.attention.count_ones());
        eprintln!("Tension:      {}", result.tension);
        eprintln!("Active layers: {}", result.active_layers);
        eprintln!("New connections: {}", result.new_connections);
        eprintln!("Density:      {:.4}", result.density);

        // Analytical uses Intersection → should be sparser than Union
        let analytical_bits = result.attention.count_ones();

        p3d.modulate(ThinkingStyle::CREATIVE);
        let creative_result = p3d.infer(0);
        let creative_bits = creative_result.attention.count_ones();

        eprintln!("\n=== Creative Style ===");
        eprintln!("Attention:    {:016X} ({} bits)", creative_result.attention, creative_bits);

        assert!(
            analytical_bits <= creative_bits,
            "Analytical (Intersection) should be sparser than Creative (Union): {analytical_bits} vs {creative_bits}"
        );
    }

    #[test]
    fn thinking_style_contradiction_modes() {
        let mut p3d = make_test_palette3d();
        let query = 0;

        // Suppress mode: contradiction kills bits
        p3d.modulate(ThinkingStyle::ANALYTICAL); // uses Suppress
        let suppress_result = p3d.infer(query);

        // Ignore mode: contradiction has no effect
        p3d.modulate(ThinkingStyle::CREATIVE); // uses Ignore
        let ignore_result = p3d.infer(query);

        // Invert mode: contradiction flips bits
        p3d.modulate(ThinkingStyle::DIVERGENT); // uses Invert
        let invert_result = p3d.infer(query);

        eprintln!("\n=== Contradiction Modes ===");
        eprintln!("Suppress: {} bits, tension={}", suppress_result.attention.count_ones(), suppress_result.tension);
        eprintln!("Ignore:   {} bits, tension={}", ignore_result.attention.count_ones(), ignore_result.tension);
        eprintln!("Invert:   {} bits, tension={}", invert_result.attention.count_ones(), invert_result.tension);

        // Suppress should have fewer bits than Ignore (bits get killed)
        assert!(
            suppress_result.attention.count_ones() <= ignore_result.attention.count_ones(),
            "Suppress should produce fewer or equal bits than Ignore"
        );
    }

    #[test]
    fn deduction_grows_connections() {
        let mut p3d = make_test_palette3d();
        p3d.modulate(ThinkingStyle::INTEGRATIVE);

        let before_density = p3d.density();
        let result = p3d.reason(0, 5);
        let after_density = p3d.density();

        eprintln!("\n=== Deduction Growth ===");
        eprintln!("Density before: {:.4}", before_density);
        eprintln!("Density after:  {:.4}", after_density);
        eprintln!("New connections: {}", result.new_connections);
        eprintln!("Steps to convergence: {}", p3d.step);

        // Deduction should either grow or stay same (never shrink)
        assert!(
            after_density >= before_density,
            "Deduction should grow or maintain density"
        );
    }

    #[test]
    fn style_transition() {
        let mut p3d = make_test_palette3d();

        let results = p3d.transition(
            ThinkingStyle::ANALYTICAL,
            ThinkingStyle::CREATIVE,
            0,
            8,
        );

        eprintln!("\n=== Style Transition: Analytical → Creative ===");
        for (i, r) in results.iter().enumerate() {
            eprintln!(
                "Step {i}: {} bits, tension={}, layers={}, density={:.4}",
                r.attention.count_ones(),
                r.tension,
                r.active_layers,
                r.density,
            );
        }

        assert_eq!(results.len(), 8);
        // Trend: bit count should generally increase from analytical to creative
        let first_bits = results.first().unwrap().attention.count_ones();
        let last_bits = results.last().unwrap().attention.count_ones();
        eprintln!("First: {first_bits} bits → Last: {last_bits} bits");
    }

    #[test]
    fn layer_densities_diagnostic() {
        let p3d = make_test_palette3d();
        let densities = p3d.layer_densities();

        eprintln!("\n=== Layer Densities ===");
        let names = [
            "CAUSES", "ENABLES", "SUPPORTS", "CONTRADICTS",
            "REFINES", "ABSTRACTS", "GROUNDS", "BECOMES",
        ];
        for (i, name) in names.iter().enumerate() {
            eprintln!("  {:<12} {:.4}", name, densities[i]);
        }

        // CONTRADICTS should be sparser (we set it to diagonal)
        assert!(
            densities[predicate::CONTRADICTS] < densities[predicate::CAUSES],
            "CONTRADICTS should be sparser than CAUSES"
        );
    }

    #[test]
    fn focused_vs_creative_fan_out() {
        let mut p3d = make_test_palette3d();

        // Focused: only CAUSES + ENABLES, tight intersection
        p3d.modulate(ThinkingStyle::FOCUSED);
        let focused = p3d.infer(10);

        // Creative: all layers, wide union
        p3d.modulate(ThinkingStyle::CREATIVE);
        let creative = p3d.infer(10);

        eprintln!("\n=== Fan-out comparison ===");
        eprintln!("Focused:  {} targets from query 10", focused.attention.count_ones());
        eprintln!("Creative: {} targets from query 10", creative.attention.count_ones());

        assert!(
            focused.attention.count_ones() <= creative.attention.count_ones(),
            "Focused should fan out to fewer targets than Creative"
        );
    }
}
