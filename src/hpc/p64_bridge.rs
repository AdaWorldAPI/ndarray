//! p64 bridge — wires the p64 crate into ndarray's HPC infrastructure.
//!
//! Sections:
//! 1. SIMD manifold expansion
//! 2. SIMD palette attention
//! 3. NARS bridge
//! 4. CausalEdge64 compat (no dep on causal-edge crate)
//! 5. ThinkingStyle LazyLock cache
//! 6. Semiring mapping
//! 7. DeepNSM palette builder

use std::collections::HashMap;
use std::sync::LazyLock;

use crate::simd::{F64x8, U64x8};
use crate::hpc::nars::NarsTruth;
use crate::hpc::simd_caps::simd_caps;

// Re-export p64 types for consumers.
pub use p64::{
    AttentionResult, CombineMode, ContraMode, HeelPlanes, Palette3D, Palette64, ThinkingStyle,
    predicate,
};
pub use phyllotactic_manifold::consts as manifold_consts;

// ============================================================================
// Section 1: SIMD manifold expansion
// ============================================================================

/// Expand 8 payload slices onto the 7+1 phyllotactic spiral using SIMD.
///
/// Applies `(slice + heel + gamma) * SPIRAL7_{X,Y}` across all 8 lanes.
/// Lane 7 is zeroed (contradiction channel does not participate in the spiral).
///
/// # Returns
///
/// `(x, y)` coordinate vectors on the manifold.
///
/// # Example
///
/// ```
/// use ndarray::hpc::p64_bridge::expand_manifold_simd;
///
/// let slices = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0];
/// let (x, y) = expand_manifold_simd(slices, 0.5, 0.25);
/// // Lane 7 is always zeroed on the manifold
/// assert_eq!(x.to_array()[7], 0.0);
/// assert_eq!(y.to_array()[7], 0.0);
/// ```
pub fn expand_manifold_simd(slices: [f64; 8], heel: f64, gamma: f64) -> (F64x8, F64x8) {
    let bias = F64x8::splat(heel + gamma);
    let vals = F64x8::from_array(slices) + bias;

    let spiral_x = F64x8::from_array(manifold_consts::SPIRAL7_X);
    let spiral_y = F64x8::from_array(manifold_consts::SPIRAL7_Y);

    let x = vals * spiral_x;
    let y = vals * spiral_y;

    (x, y)
}

/// Resonance check on SIMD manifold coordinates.
///
/// Returns a 7-bit mask: bit `i` is set if `x[i]^2 + y[i]^2 >= threshold^2`.
/// Lane 7 is excluded (contradiction channel).
///
/// # Example
///
/// ```
/// use ndarray::hpc::p64_bridge::{expand_manifold_simd, resonance_simd};
///
/// let slices = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 0.0];
/// let (x, y) = expand_manifold_simd(slices, 1.0, 1.0);
/// let mask = resonance_simd(&x, &y, 10.0);
/// // With large values, most lanes should fire
/// assert!(mask.count_ones() > 0);
/// ```
pub fn resonance_simd(x: &F64x8, y: &F64x8, threshold: f64) -> u8 {
    let xa = x.to_array();
    let ya = y.to_array();
    let t2 = threshold * threshold;
    let mut mask = 0u8;
    for i in 0..7 {
        if xa[i] * xa[i] + ya[i] * ya[i] >= t2 {
            mask |= 1 << i;
        }
    }
    mask
}

// ============================================================================
// Section 2: SIMD palette attention
// ============================================================================

/// Batch popcount-attention for 8 palette rows against a single query.
///
/// Returns `popcount(row[i] AND query)` for each of the 8 rows.
/// Uses AVX-512 VPOPCNTDQ fast path when available, scalar fallback otherwise.
///
/// # Example
///
/// ```
/// use ndarray::hpc::p64_bridge::attend_batch_8;
///
/// let rows = [u64::MAX; 8];
/// let query = 0xFF00_FF00_FF00_FF00u64;
/// let scores = attend_batch_8(&rows, query);
/// assert_eq!(scores[0], 32); // 32 bits set in query
/// ```
pub fn attend_batch_8(rows_8: &[u64; 8], query: u64) -> [u8; 8] {
    if simd_caps().avx512vpopcntdq {
        // Fast path: SIMD AND + popcount on 8 lanes
        let rows_vec = U64x8::from_array(*rows_8);
        let query_vec = U64x8::splat(query);
        let anded = rows_vec & query_vec;
        let arr = anded.to_array();
        let mut scores = [0u8; 8];
        for i in 0..8 {
            scores[i] = arr[i].count_ones() as u8;
        }
        scores
    } else {
        // Scalar fallback
        let mut scores = [0u8; 8];
        for i in 0..8 {
            scores[i] = (rows_8[i] & query).count_ones() as u8;
        }
        scores
    }
}

/// Full palette attention with SIMD acceleration.
///
/// Processes the 64 rows in 8 batches of 8, using `attend_batch_8` for each.
///
/// # Example
///
/// ```
/// use ndarray::hpc::p64_bridge::{attend_simd, Palette64};
///
/// let palette = Palette64::zero();
/// let result = attend_simd(&palette, 0xDEAD_BEEF, 16);
/// assert_eq!(result.distance, 64); // all-zero palette => no bits match
/// ```
pub fn attend_simd(palette: &Palette64, query: u64, gamma: u8) -> AttentionResult {
    let mut scores = [0u8; 64];
    let mut best_idx = 0u8;
    let mut best_score = 0u8;
    let mut fires = 0u64;

    for batch in 0..8 {
        let start = batch * 8;
        let chunk: [u64; 8] = [
            palette.rows[start],
            palette.rows[start + 1],
            palette.rows[start + 2],
            palette.rows[start + 3],
            palette.rows[start + 4],
            palette.rows[start + 5],
            palette.rows[start + 6],
            palette.rows[start + 7],
        ];
        let batch_scores = attend_batch_8(&chunk, query);
        for j in 0..8 {
            let idx = start + j;
            scores[idx] = batch_scores[j];
            if batch_scores[j] > best_score {
                best_score = batch_scores[j];
                best_idx = idx as u8;
            }
            if batch_scores[j] >= gamma {
                fires |= 1u64 << idx;
            }
        }
    }

    AttentionResult {
        best_idx,
        distance: 64 - best_score,
        scores,
        fires,
    }
}

// ============================================================================
// Section 3: NARS bridge
// ============================================================================

/// Convert a 7-bit resonance mask and contradiction value to a NARS truth value.
///
/// Uses the phyllotactic manifold's `nars_truth()` function:
/// - frequency = fraction of 7 lanes that fire
/// - confidence = inverse of normalized contradiction magnitude
///
/// # Example
///
/// ```
/// use ndarray::hpc::p64_bridge::resonance_to_nars;
///
/// let tv = resonance_to_nars(0b0111_1111, 0.0, 100.0);
/// assert!((tv.frequency - 1.0).abs() < 1e-6);
/// assert!((tv.confidence - 1.0).abs() < 0.01);
/// ```
pub fn resonance_to_nars(resonance_7bit: u8, contradiction: f64, max_contra: f64) -> NarsTruth {
    let (f, c) =
        phyllotactic_manifold::seven_plus_one::nars_truth(resonance_7bit, contradiction, max_contra);
    NarsTruth::new(f as f32, c as f32)
}

/// Encode a NARS truth value into a single branch byte (CLAM B2 format).
///
/// ```text
/// Bits [6:0] = frequency quantized to 0..127
/// Bit  [7]   = confidence > 0.5 flag
/// ```
///
/// # Example
///
/// ```
/// use ndarray::hpc::p64_bridge::nars_to_branch_byte;
/// use ndarray::hpc::nars::NarsTruth;
///
/// let tv = NarsTruth::new(1.0, 0.9);
/// let b = nars_to_branch_byte(&tv);
/// assert_eq!(b & 0x80, 0x80); // confident
/// assert_eq!(b & 0x7F, 127);  // max frequency
/// ```
pub fn nars_to_branch_byte(truth: &NarsTruth) -> u8 {
    let freq_bits = (truth.frequency.clamp(0.0, 1.0) * 127.0) as u8;
    let conf_bit = if truth.confidence > 0.5 { 0x80 } else { 0x00 };
    freq_bits | conf_bit
}

// ============================================================================
// Section 4: CausalEdge64 compat (NO dep on causal-edge crate)
// ============================================================================

/// Bit-layout constants compatible with lance-graph's CausalEdge64.
///
/// ```text
/// [63:58] source archetype (6 bits → 0..63)
/// [57:52] target archetype (6 bits → 0..63)
/// [51:44] NARS frequency   (8 bits → 0..255, maps to 0.0..1.0)
/// [43:36] NARS confidence  (8 bits → 0..255, maps to 0.0..1.0)
/// [35:33] predicate layer  (3 bits → 0..7)
/// [32:0]  payload / hash   (33 bits)
/// ```
pub mod causal_edge_compat {
    use super::*;

    pub const SRC_SHIFT: u32 = 58;
    pub const SRC_MASK: u64 = 0x3F << 58;
    pub const TGT_SHIFT: u32 = 52;
    pub const TGT_MASK: u64 = 0x3F << 52;
    pub const FREQ_SHIFT: u32 = 44;
    pub const FREQ_MASK: u64 = 0xFF << 44;
    pub const CONF_SHIFT: u32 = 36;
    pub const CONF_MASK: u64 = 0xFF << 36;
    pub const LAYER_SHIFT: u32 = 33;
    pub const LAYER_MASK: u64 = 0x07 << 33;

    /// Extract source and target archetype indices from an edge.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::p64_bridge::causal_edge_compat::edge_to_palette_address;
    ///
    /// // Source=5, Target=10 packed into bits [63:52]
    /// let edge = (5u64 << 58) | (10u64 << 52);
    /// let (src, tgt) = edge_to_palette_address(edge);
    /// assert_eq!(src, 5);
    /// assert_eq!(tgt, 10);
    /// ```
    #[inline]
    pub fn edge_to_palette_address(edge: u64) -> (usize, usize) {
        let src = ((edge & SRC_MASK) >> SRC_SHIFT) as usize;
        let tgt = ((edge & TGT_MASK) >> TGT_SHIFT) as usize;
        (src, tgt)
    }

    /// Extract NARS frequency and confidence from an edge as `(f32, f32)`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::p64_bridge::causal_edge_compat::edge_nars;
    ///
    /// let edge = (200u64 << 44) | (150u64 << 36);
    /// let (f, c) = edge_nars(edge);
    /// assert!((f - 200.0 / 255.0).abs() < 0.01);
    /// assert!((c - 150.0 / 255.0).abs() < 0.01);
    /// ```
    #[inline]
    pub fn edge_nars(edge: u64) -> (f32, f32) {
        let f_raw = ((edge & FREQ_MASK) >> FREQ_SHIFT) as f32;
        let c_raw = ((edge & CONF_MASK) >> CONF_SHIFT) as f32;
        (f_raw / 255.0, c_raw / 255.0)
    }

    /// Extract the 3-bit predicate layer index from an edge.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::p64_bridge::causal_edge_compat::edge_to_layer_mask;
    ///
    /// let edge = 3u64 << 33; // layer 3 = CONTRADICTS
    /// let mask = edge_to_layer_mask(edge);
    /// assert_eq!(mask, 1 << 3);
    /// ```
    #[inline]
    pub fn edge_to_layer_mask(edge: u64) -> u8 {
        let layer = ((edge & LAYER_MASK) >> LAYER_SHIFT) as u8;
        1u8 << layer
    }

    /// Build a Palette64 from a slice of CausalEdge64-format edges.
    ///
    /// For each edge, sets `palette.rows[src] |= (1 << tgt)`.
    /// Multiple edges accumulate via OR (union of connections).
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::p64_bridge::causal_edge_compat::palette_from_edges;
    ///
    /// let edge = (2u64 << 58) | (5u64 << 52);
    /// let palette = palette_from_edges(&[edge]);
    /// assert_ne!(palette.rows[2] & (1 << 5), 0);
    /// ```
    pub fn palette_from_edges(edges: &[u64]) -> Palette64 {
        let mut palette = Palette64::zero();
        for &edge in edges {
            let (src, tgt) = edge_to_palette_address(edge);
            if src < 64 && tgt < 64 {
                palette.rows[src] |= 1u64 << tgt;
            }
        }
        palette
    }

    /// Build a Palette3D from edges, routing each edge to its predicate layer.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::p64_bridge::causal_edge_compat::palette3d_from_edges;
    /// use ndarray::hpc::p64_bridge::ThinkingStyle;
    ///
    /// let edge = (1u64 << 58) | (2u64 << 52) | (0u64 << 33); // layer 0 = CAUSES
    /// let p3d = palette3d_from_edges(&[edge], ThinkingStyle::ANALYTICAL);
    /// assert_ne!(p3d.layers[0].rows[1] & (1 << 2), 0);
    /// ```
    pub fn palette3d_from_edges(edges: &[u64], style: ThinkingStyle) -> Palette3D {
        let mut layers = [Palette64::zero(); 8];
        for &edge in edges {
            let (src, tgt) = edge_to_palette_address(edge);
            let layer = ((edge & LAYER_MASK) >> LAYER_SHIFT) as usize;
            if src < 64 && tgt < 64 && layer < 8 {
                layers[layer].rows[src] |= 1u64 << tgt;
            }
        }
        Palette3D::new(layers, style)
    }
}

// ============================================================================
// Section 5: ThinkingStyle LazyLock cache
// ============================================================================

/// All defined ThinkingStyle variants, indexed by ordinal.
static STYLES: LazyLock<[ThinkingStyle; 6]> = LazyLock::new(|| {
    [
        ThinkingStyle::ANALYTICAL,
        ThinkingStyle::CREATIVE,
        ThinkingStyle::FOCUSED,
        ThinkingStyle::INTEGRATIVE,
        ThinkingStyle::DIVERGENT,
        ThinkingStyle::META,
    ]
});

/// Name-to-ordinal lookup map.
static STYLE_MAP: LazyLock<HashMap<&'static str, usize>> = LazyLock::new(|| {
    let styles = &*STYLES;
    let mut map = HashMap::new();
    for (i, s) in styles.iter().enumerate() {
        map.insert(s.name, i);
    }
    map
});

/// Get a ThinkingStyle by ordinal (wraps around if out of range).
///
/// # Example
///
/// ```
/// use ndarray::hpc::p64_bridge::thinking_style;
///
/// let s = thinking_style(0);
/// assert_eq!(s.name, "analytical");
/// ```
pub fn thinking_style(ordinal: usize) -> ThinkingStyle {
    STYLES[ordinal % STYLES.len()]
}

/// Get a ThinkingStyle by name. Falls back to ANALYTICAL if not found.
///
/// # Example
///
/// ```
/// use ndarray::hpc::p64_bridge::thinking_style_from_name;
///
/// let s = thinking_style_from_name("creative");
/// assert_eq!(s.name, "creative");
///
/// let fallback = thinking_style_from_name("nonexistent");
/// assert_eq!(fallback.name, "analytical");
/// ```
pub fn thinking_style_from_name(name: &str) -> ThinkingStyle {
    match STYLE_MAP.get(name) {
        Some(&idx) => STYLES[idx],
        None => ThinkingStyle::ANALYTICAL,
    }
}

// ============================================================================
// Section 6: Semiring mapping
// ============================================================================

/// Map a semiring name to the corresponding CombineMode.
///
/// | Semiring      | CombineMode  |
/// |---------------|--------------|
/// | "boolean"     | Union        |
/// | "tropical"    | Intersection |
/// | "viterbi"     | Majority     |
/// | "fuzzy"       | Weighted     |
/// | _             | Union        |
///
/// # Example
///
/// ```
/// use ndarray::hpc::p64_bridge::{semiring_to_combine, CombineMode};
///
/// assert_eq!(semiring_to_combine("boolean"), CombineMode::Union);
/// assert_eq!(semiring_to_combine("tropical"), CombineMode::Intersection);
/// ```
pub fn semiring_to_combine(name: &str) -> CombineMode {
    match name {
        "boolean" => CombineMode::Union,
        "tropical" => CombineMode::Intersection,
        "viterbi" => CombineMode::Majority,
        "fuzzy" => CombineMode::Weighted,
        _ => CombineMode::Union,
    }
}

/// Map a semiring name to the corresponding ContraMode.
///
/// | Semiring      | ContraMode |
/// |---------------|------------|
/// | "boolean"     | Suppress   |
/// | "tropical"    | Suppress   |
/// | "viterbi"     | Tension    |
/// | "fuzzy"       | Tension    |
/// | "creative"    | Ignore     |
/// | "divergent"   | Invert     |
/// | _             | Suppress   |
///
/// # Example
///
/// ```
/// use ndarray::hpc::p64_bridge::{semiring_to_contra, ContraMode};
///
/// assert_eq!(semiring_to_contra("boolean"), ContraMode::Suppress);
/// assert_eq!(semiring_to_contra("divergent"), ContraMode::Invert);
/// ```
pub fn semiring_to_contra(name: &str) -> ContraMode {
    match name {
        "boolean" | "tropical" => ContraMode::Suppress,
        "viterbi" | "fuzzy" => ContraMode::Tension,
        "creative" => ContraMode::Ignore,
        "divergent" => ContraMode::Invert,
        _ => ContraMode::Suppress,
    }
}

// ============================================================================
// Section 7: DeepNSM palette builder
// ============================================================================

/// Build a Palette64 from DeepNSM pairwise distances.
///
/// For each pair `(i, j)` where `i, j < n.min(64)`, sets bit `j` in row `i`
/// if `distances[i * n + j] <= radius`. This converts a distance matrix into
/// the binary adjacency / attention matrix that p64 operates on.
///
/// # Arguments
///
/// * `distances` - flat `n x n` distance matrix (row-major)
/// * `n` - side length of the distance matrix
/// * `radius` - connectivity threshold: pairs within this distance are connected
///
/// # Example
///
/// ```
/// use ndarray::hpc::p64_bridge::palette_from_deepnsm_distances;
///
/// // 4 nodes, all within radius of each other
/// let distances = vec![0, 1, 2, 3,
///                      1, 0, 1, 2,
///                      2, 1, 0, 1,
///                      3, 2, 1, 0];
/// let palette = palette_from_deepnsm_distances(&distances, 4, 2);
/// // Node 0 connects to nodes 0,1,2 (distance <= 2)
/// assert_ne!(palette.rows[0] & (1 << 1), 0);
/// assert_ne!(palette.rows[0] & (1 << 2), 0);
/// assert_eq!(palette.rows[0] & (1 << 3), 0); // distance 3 > radius 2
/// ```
pub fn palette_from_deepnsm_distances(distances: &[u16], n: usize, radius: u16) -> Palette64 {
    let cap = n.min(64);
    let mut palette = Palette64::zero();
    for i in 0..cap {
        for j in 0..cap {
            if distances[i * n + j] <= radius {
                palette.rows[i] |= 1u64 << j;
            }
        }
    }
    palette
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_manifold_simd_lane7_zero() {
        let slices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 99.0];
        let (x, y) = expand_manifold_simd(slices, 1.0, 0.5);
        // Lane 7 of SPIRAL7_X and SPIRAL7_Y is 0.0, so output must be 0.0
        assert_eq!(x.to_array()[7], 0.0);
        assert_eq!(y.to_array()[7], 0.0);
        // Other lanes should be non-zero (large input values)
        assert_ne!(x.to_array()[0], 0.0);
        assert_ne!(y.to_array()[1], 0.0);
    }

    #[test]
    fn test_resonance_simd_threshold() {
        let slices = [100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 100.0, 0.0];
        let (x, y) = expand_manifold_simd(slices, 0.0, 0.0);
        // With threshold 0, all non-zero lanes should fire
        let mask = resonance_simd(&x, &y, 0.0);
        // Lanes 0,2,4,6 have value 100, lanes 1,3,5 have value 0
        // Even zero lanes get multiplied by spiral coords, so check > 0
        // Actually lane 0 has SPIRAL7_X[0] ≈ 0.76, so 100*0.76 ≈ 76 → fires
        assert!(mask.count_ones() >= 4);

        // With a very high threshold, nothing should fire
        let mask_high = resonance_simd(&x, &y, 1e12);
        assert_eq!(mask_high, 0);
    }

    #[test]
    fn test_attend_batch_8_all_ones() {
        let rows = [u64::MAX; 8];
        let query = 0xAAAA_AAAA_AAAA_AAAAu64;
        let scores = attend_batch_8(&rows, query);
        let expected = query.count_ones() as u8; // 32
        for &s in &scores {
            assert_eq!(s, expected);
        }
    }

    #[test]
    fn test_attend_simd_matches_scalar() {
        // Build a palette with known patterns
        let mut palette = Palette64::zero();
        for i in 0..64 {
            palette.rows[i] = (i as u64).wrapping_mul(0x0101_0101_0101_0101);
        }
        let query = 0xFFFF_0000_FFFF_0000u64;
        let gamma = 8u8;

        let simd_result = attend_simd(&palette, query, gamma);
        let scalar_result = palette.attend(query, gamma);

        assert_eq!(simd_result.best_idx, scalar_result.best_idx);
        assert_eq!(simd_result.distance, scalar_result.distance);
        assert_eq!(simd_result.fires, scalar_result.fires);
        assert_eq!(simd_result.scores, scalar_result.scores);
    }

    #[test]
    fn test_resonance_to_nars() {
        // All 7 lanes fire, no contradiction
        let tv = resonance_to_nars(0b0111_1111, 0.0, 100.0);
        assert!((tv.frequency - 1.0).abs() < 1e-5);
        assert!((tv.confidence - 1.0).abs() < 0.01);

        // No lanes fire, max contradiction
        let tv2 = resonance_to_nars(0, 100.0, 100.0);
        assert!((tv2.frequency - 0.0).abs() < 1e-5);
        assert!((tv2.confidence - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_nars_to_branch_byte() {
        let tv = NarsTruth::new(1.0, 0.9);
        let b = nars_to_branch_byte(&tv);
        assert_eq!(b & 0x80, 0x80); // confident
        assert_eq!(b & 0x7F, 127);

        let tv_low = NarsTruth::new(0.0, 0.1);
        let b_low = nars_to_branch_byte(&tv_low);
        assert_eq!(b_low & 0x80, 0); // not confident
        assert_eq!(b_low & 0x7F, 0);
    }

    #[test]
    fn test_causal_edge_roundtrip() {
        use causal_edge_compat::*;

        let src = 15usize;
        let tgt = 42usize;
        let freq = 200u64;
        let conf = 150u64;
        let layer = 3u64; // CONTRADICTS

        let edge = ((src as u64) << SRC_SHIFT)
            | ((tgt as u64) << TGT_SHIFT)
            | (freq << FREQ_SHIFT)
            | (conf << CONF_SHIFT)
            | (layer << LAYER_SHIFT);

        let (s, t) = edge_to_palette_address(edge);
        assert_eq!(s, src);
        assert_eq!(t, tgt);

        let (f, c) = edge_nars(edge);
        assert!((f - 200.0 / 255.0).abs() < 0.01);
        assert!((c - 150.0 / 255.0).abs() < 0.01);

        let mask = edge_to_layer_mask(edge);
        assert_eq!(mask, 1 << 3);
    }

    #[test]
    fn test_thinking_style_cache() {
        let s = thinking_style(0);
        assert_eq!(s.name, "analytical");
        let s1 = thinking_style(1);
        assert_eq!(s1.name, "creative");
        // Wrap around
        let sw = thinking_style(6);
        assert_eq!(sw.name, "analytical"); // 6 % 6 = 0

        let sn = thinking_style_from_name("meta");
        assert_eq!(sn.name, "meta");
        let fallback = thinking_style_from_name("unknown");
        assert_eq!(fallback.name, "analytical");
    }

    #[test]
    fn test_semiring_mapping() {
        assert_eq!(semiring_to_combine("boolean"), CombineMode::Union);
        assert_eq!(semiring_to_combine("tropical"), CombineMode::Intersection);
        assert_eq!(semiring_to_combine("viterbi"), CombineMode::Majority);
        assert_eq!(semiring_to_combine("fuzzy"), CombineMode::Weighted);
        assert_eq!(semiring_to_combine("???"), CombineMode::Union);

        assert_eq!(semiring_to_contra("boolean"), ContraMode::Suppress);
        assert_eq!(semiring_to_contra("divergent"), ContraMode::Invert);
        assert_eq!(semiring_to_contra("creative"), ContraMode::Ignore);
    }

    #[test]
    fn test_palette_from_deepnsm_distances() {
        let n = 4;
        let distances = vec![0, 1, 2, 3, 1, 0, 1, 2, 2, 1, 0, 1, 3, 2, 1, 0u16];
        let palette = palette_from_deepnsm_distances(&distances, n, 2);

        // Node 0: connects to 0 (d=0), 1 (d=1), 2 (d=2), not 3 (d=3)
        assert_ne!(palette.rows[0] & (1 << 0), 0);
        assert_ne!(palette.rows[0] & (1 << 1), 0);
        assert_ne!(palette.rows[0] & (1 << 2), 0);
        assert_eq!(palette.rows[0] & (1 << 3), 0);

        // Node 3: connects to 2 (d=1), 3 (d=0), not 0 (d=3)
        assert_eq!(palette.rows[3] & (1 << 0), 0);
        assert_ne!(palette.rows[3] & (1 << 2), 0);
        assert_ne!(palette.rows[3] & (1 << 3), 0);
    }

    #[test]
    fn test_palette3d_from_edges() {
        use causal_edge_compat::*;

        let edge_causes = (1u64 << SRC_SHIFT) | (2u64 << TGT_SHIFT) | (0u64 << LAYER_SHIFT);
        let edge_contra =
            (3u64 << SRC_SHIFT) | (4u64 << TGT_SHIFT) | (3u64 << LAYER_SHIFT);

        let p3d =
            palette3d_from_edges(&[edge_causes, edge_contra], ThinkingStyle::ANALYTICAL);

        // CAUSES layer: row 1 has bit 2 set
        assert_ne!(p3d.layers[0].rows[1] & (1 << 2), 0);
        // CONTRADICTS layer: row 3 has bit 4 set
        assert_ne!(p3d.layers[3].rows[3] & (1 << 4), 0);
        // Other layers untouched
        assert_eq!(p3d.layers[1].rows[1], 0);
    }

    // ================================================================
    // GPT-2 → P64 rehydration: prove attention reconstruction works
    // ================================================================

    #[test]
    fn test_gpt2_palette_to_p64_rehydration() {
        use crate::hpc::jina::runtime::GPT2;

        let gpt2 = &*GPT2;
        eprintln!("GPT-2 vocab: {} tokens", gpt2.vocab_size());

        // Flatten the 256×256 distance table
        let dt = &gpt2.palette.distance_table;
        let mut flat = vec![0u16; 256 * 256];
        for i in 0..256 {
            for j in 0..256 {
                flat[i * 256 + j] = dt[i][j];
            }
        }

        // Find a reasonable interaction radius from the distance distribution
        // Use median distance as threshold — roughly 50% density
        let mut all_dists: Vec<u16> = Vec::with_capacity(256 * 255 / 2);
        for i in 0..256 {
            for j in (i + 1)..256 {
                all_dists.push(dt[i][j]);
            }
        }
        all_dists.sort();
        let median = all_dists[all_dists.len() / 2];
        // Use 25th percentile for sparse palette (~12.5% density)
        let p25 = all_dists[all_dists.len() / 4];
        eprintln!("Distance stats: median={}, p25={}, min={}, max={}",
            median, p25, all_dists[0], all_dists.last().unwrap());

        // Build Palette64 from GPT-2's learned distance table
        let palette = palette_from_deepnsm_distances(&flat, 256, p25);

        // Check it's not empty or full
        let density: u32 = palette.rows.iter().map(|r| r.count_ones()).sum();
        let total_bits = 64 * 64;
        let pct = density as f64 / total_bits as f64 * 100.0;
        eprintln!("Palette density: {}/{} bits ({:.1}%)", density, total_bits, pct);
        assert!(density > 100, "palette too sparse: {density}");
        assert!(density < 3500, "palette too dense: {density}");

        // Build Palette3D — same topology for all layers (GPT-2 is one model)
        let mut p3d_analytical = Palette3D::new([palette; 8], ThinkingStyle::ANALYTICAL);
        let mut p3d_creative = Palette3D::new([palette; 8], ThinkingStyle::CREATIVE);

        // Infer from archetype 42 through both styles
        let r_analytical = p3d_analytical.infer(42);
        let r_creative = p3d_creative.infer(42);

        eprintln!("Analytical: attention={:064b}, tension={}, active_layers={}, new={}",
            r_analytical.attention, r_analytical.tension,
            r_analytical.active_layers, r_analytical.new_connections);
        eprintln!("Creative:   attention={:064b}, tension={}, active_layers={}, new={}",
            r_creative.attention, r_creative.tension,
            r_creative.active_layers, r_creative.new_connections);

        // KEY ASSERTION: different styles produce different fan-out
        // Creative (Union, all layers, density 0.40) should activate MORE targets
        // than Analytical (Intersection, 6 layers, density 0.05)
        let analytical_popcount = r_analytical.attention.count_ones();
        let creative_popcount = r_creative.attention.count_ones();
        eprintln!("Fan-out: analytical={}, creative={}", analytical_popcount, creative_popcount);

        assert!(creative_popcount >= analytical_popcount,
            "Creative should have wider fan-out than Analytical: {} vs {}",
            creative_popcount, analytical_popcount);

        // Verify attention is non-trivial
        assert!(analytical_popcount > 0, "Analytical should fire something");
        assert!(creative_popcount > 0, "Creative should fire something");

        // Check that the palette-based similarity correlates with GPT-2's actual distances
        // Pick two tokens that the palette says interact (bit set) and two that don't
        let mut interacting = None;
        let mut non_interacting = None;
        for i in 0..64 {
            for j in 0..64 {
                if i == j { continue; }
                if palette.rows[i] & (1 << j) != 0 && interacting.is_none() {
                    interacting = Some((i, j));
                }
                if palette.rows[i] & (1 << j) == 0 && non_interacting.is_none() {
                    non_interacting = Some((i, j));
                }
                if interacting.is_some() && non_interacting.is_some() { break; }
            }
            if interacting.is_some() && non_interacting.is_some() { break; }
        }

        if let (Some((ia, ib)), Some((na, nb))) = (interacting, non_interacting) {
            // Interacting pair should have LOWER distance than non-interacting
            let d_interact = flat[ia * 256 + ib];
            let d_non = flat[na * 256 + nb];
            eprintln!("Interacting ({},{}) distance={}, Non-interacting ({},{}) distance={}",
                ia, ib, d_interact, na, nb, d_non);
            assert!(d_interact <= d_non,
                "Interacting pair should be closer: {} vs {}", d_interact, d_non);
        }

        eprintln!("GPT-2 → P64 rehydration: PASS");
        eprintln!("  50K tokens → 256 archetypes → 64×64 palette → 8-layer Palette3D");
        eprintln!("  Thinking style modulates fan-out: Analytical={}, Creative={}",
            analytical_popcount, creative_popcount);
    }
}
