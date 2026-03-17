//! Binary Neural Network (BNN) inference primitives for HDC graph memory.
//!
//! Implements BNN convolution and inference using XNOR + popcount operations,
//! directly leveraging the existing 16,384-bit `Fingerprint<256>` containers
//! and `GraphHV` 3-channel hypervectors.
//!
//! The fundamental operation: `dot(a, w) = 2 * popcount(XNOR(a, w)) - total_bits`
//!
//! ## Architecture
//!
//! - Binary weights and activations (1-bit): stored as `Fingerprint<256>`
//! - XNOR + popcount = binary dot product (the Hamming-distance dual)
//! - Rich information flow via amplitude correction
//! - Winner-take-all competition in each layer
//! - Plastic channel for continual learning without backprop

use super::cam_index::{CamIndex, GraphHV, SplitMix64};
use super::fingerprint::Fingerprint;
use super::kernels::{kernel_pipeline, EnergyConflict, KernelStage, PipelineStats, SliceGate, SKU_16K_WORDS};

/// Result of a binary convolution (XNOR + popcount).
#[derive(Clone, Copy, Debug)]
pub struct BnnDotResult {
    /// Raw popcount of XNOR(activation, weight) — number of matching bits.
    pub match_count: u32,
    /// Total bits compared.
    pub total_bits: u32,
    /// Normalized score in [-1.0, 1.0] (bipolar interpretation).
    pub score: f32,
}

/// Binary dot product via XNOR + popcount on a single channel.
#[inline]
pub fn bnn_dot(activation: &Fingerprint<256>, weight: &Fingerprint<256>) -> BnnDotResult {
    let total_bits = Fingerprint::<256>::BITS as u32;
    let xor_popcount =
        super::bitwise::hamming_distance_raw(activation.as_bytes(), weight.as_bytes()) as u32;
    let match_count = total_bits - xor_popcount;
    let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
    BnnDotResult {
        match_count,
        total_bits,
        score,
    }
}

/// Binary dot product across all 3 channels of a `GraphHV`.
pub fn bnn_dot_3ch(activation: &GraphHV, weight: &GraphHV) -> BnnDotResult {
    let total_bits = (Fingerprint::<256>::BITS * 3) as u32;
    let mut xor_total = 0u32;
    for ch in 0..3 {
        xor_total += super::bitwise::hamming_distance_raw(
            activation.channels[ch].as_bytes(),
            weight.channels[ch].as_bytes(),
        ) as u32;
    }
    let match_count = total_bits - xor_total;
    let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
    BnnDotResult {
        match_count,
        total_bits,
        score,
    }
}

/// A binary neuron in the graph BNN.
pub struct BnnNeuron {
    /// 3-channel state: [activation, weights, plastic].
    pub state: GraphHV,
    /// Bias (amplitude correction).
    pub bias: f32,
    /// Activation threshold.
    pub threshold: f32,
}

impl BnnNeuron {
    /// Create a new neuron with random weights.
    pub fn random(rng: &mut SplitMix64) -> Self {
        Self {
            state: GraphHV::random(rng),
            bias: 0.0,
            threshold: 0.0,
        }
    }

    /// Create from a pre-existing weight pattern.
    pub fn from_weights(weights: Fingerprint<256>, rng: &mut SplitMix64) -> Self {
        let mut state = GraphHV::zero();
        state.channels[1] = weights;
        state.channels[2] = state.channels[1].clone();
        let mut words = [0u64; 256];
        for w in words.iter_mut() {
            *w = rng.next_u64();
        }
        state.channels[0] = Fingerprint::from_words(words);
        Self {
            state,
            bias: 0.0,
            threshold: 0.0,
        }
    }

    /// Forward pass: compute binary activation from input.
    pub fn forward(
        &mut self,
        input: &Fingerprint<256>,
        learn: bool,
        learning_rate: f64,
        rng: &mut SplitMix64,
    ) -> f32 {
        let dot = bnn_dot(input, &self.state.channels[1]);
        let pre_activation = dot.score + self.bias;

        if pre_activation > self.threshold {
            self.state.channels[0] = input.clone();
        } else {
            self.state.channels[0] = !(input);
        }

        if learn {
            // Simple plastic bundling: majority-vote merge
            let plastic = &self.state.channels[2];
            let mut new_words = [0u64; 256];
            let rate_bits = (learning_rate * 64.0).round() as u32;
            for i in 0..256 {
                let mask = random_mask(rng, rate_bits);
                new_words[i] = (input.words[i] & mask) | (plastic.words[i] & !mask);
            }
            self.state.channels[2] = Fingerprint::from_words(new_words);
        }

        pre_activation
    }

    /// Get the current binary activation.
    #[inline]
    pub fn activation(&self) -> &Fingerprint<256> {
        &self.state.channels[0]
    }

    /// Get the weight pattern.
    #[inline]
    pub fn weights(&self) -> &Fingerprint<256> {
        &self.state.channels[1]
    }

    /// Get the plastic (learned) prototype.
    #[inline]
    pub fn plastic(&self) -> &Fingerprint<256> {
        &self.state.channels[2]
    }
}

/// Generate a random u64 bitmask with approximately `bits_set` bits set.
fn random_mask(rng: &mut SplitMix64, bits_set: u32) -> u64 {
    if bits_set >= 64 {
        return u64::MAX;
    }
    if bits_set == 0 {
        return 0;
    }
    let threshold = ((bits_set as u64) << 58) / 64;
    let mut mask = 0u64;
    for bit in 0..64 {
        if (rng.next_u64() >> 6) < threshold {
            mask |= 1u64 << bit;
        }
    }
    mask
}

/// A binary layer: multiple neurons processing the same input.
pub struct BnnLayer {
    pub neurons: Vec<BnnNeuron>,
}

impl BnnLayer {
    /// Create a layer with `n` neurons, randomly initialized.
    pub fn random(n: usize, rng: &mut SplitMix64) -> Self {
        Self {
            neurons: (0..n).map(|_| BnnNeuron::random(rng)).collect(),
        }
    }

    /// Forward pass: compute all neurons' activations.
    pub fn forward(
        &mut self,
        input: &Fingerprint<256>,
        learn: bool,
        learning_rate: f64,
        rng: &mut SplitMix64,
    ) -> Vec<f32> {
        self.neurons
            .iter_mut()
            .map(|n| n.forward(input, learn, learning_rate, rng))
            .collect()
    }

    /// Find the neuron with the highest activation score (winner-take-all).
    pub fn winner(&self, input: &Fingerprint<256>) -> (usize, BnnDotResult) {
        let mut best_idx = 0;
        let mut best_result = bnn_dot(input, &self.neurons[0].state.channels[1]);
        for (i, neuron) in self.neurons.iter().enumerate().skip(1) {
            let result = bnn_dot(input, &neuron.state.channels[1]);
            if result.score > best_result.score {
                best_idx = i;
                best_result = result;
            }
        }
        (best_idx, best_result)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// Build a CAM index over all neuron weight patterns.
    pub fn build_cam_index(&self, seed: u64) -> CamIndex {
        let mut cam = CamIndex::with_defaults(seed);
        for neuron in &self.neurons {
            let w = &neuron.state.channels[1];
            let hv = GraphHV {
                channels: [w.clone(), w.clone(), w.clone()],
            };
            cam.insert(hv);
        }
        cam
    }

    /// Winner-take-all using CAM index: O(log N) instead of O(N).
    pub fn winner_cam(
        &self,
        input: &Fingerprint<256>,
        cam: &CamIndex,
        shortlist_size: usize,
    ) -> Option<(usize, BnnDotResult)> {
        let query_hv = GraphHV {
            channels: [input.clone(), input.clone(), input.clone()],
        };

        let hits = cam.query(&query_hv, shortlist_size);
        if hits.is_empty() {
            return None;
        }

        let mut best_idx = hits[0].index;
        let mut best_result = bnn_dot(input, &self.neurons[hits[0].index].state.channels[1]);

        for hit in hits.iter().skip(1) {
            if hit.index < self.neurons.len() {
                let result = bnn_dot(input, &self.neurons[hit.index].state.channels[1]);
                if result.score > best_result.score {
                    best_idx = hit.index;
                    best_result = result;
                }
            }
        }

        Some((best_idx, best_result))
    }
}

/// Batch XNOR+popcount: compute binary dot products for multiple candidates.
pub fn bnn_batch_dot(
    query: &Fingerprint<256>,
    weights: &[Fingerprint<256>],
    top_k: usize,
) -> Vec<(usize, BnnDotResult)> {
    let mut results: Vec<(usize, BnnDotResult)> = weights
        .iter()
        .enumerate()
        .map(|(i, w)| (i, bnn_dot(query, w)))
        .collect();

    results.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
    results
}

/// A multi-layer binary neural network.
pub struct BnnNetwork {
    pub layers: Vec<BnnLayer>,
}

impl BnnNetwork {
    pub fn new(layer_sizes: &[usize], rng: &mut SplitMix64) -> Self {
        Self {
            layers: layer_sizes
                .iter()
                .map(|&n| BnnLayer::random(n, rng))
                .collect(),
        }
    }

    /// Forward pass through all layers.
    pub fn forward(
        &mut self,
        input: &Fingerprint<256>,
        learn: bool,
        learning_rate: f64,
        rng: &mut SplitMix64,
    ) -> (usize, BnnDotResult) {
        let mut current_input = input.clone();

        for layer in self.layers.iter_mut() {
            layer.forward(&current_input, learn, learning_rate, rng);
            let (winner_idx, _) = layer.winner(&current_input);
            current_input = layer.neurons[winner_idx].activation().clone();
        }

        self.layers.last().unwrap().winner(&current_input)
    }

    /// Predict without learning.
    pub fn predict(&self, input: &Fingerprint<256>) -> Vec<(usize, BnnDotResult)> {
        let mut current_input = input.clone();
        let mut layer_results = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let (winner_idx, result) = layer.winner(&current_input);
            layer_results.push((winner_idx, result));
            current_input = layer.neurons[winner_idx].activation().clone();
        }

        layer_results
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.layers.len()
    }
}

// ─── Cascade-accelerated BNN search ──────────────────

/// Result of a cascade-accelerated BNN search.
#[derive(Clone, Debug)]
pub struct BnnCascadeResult {
    pub matches: Vec<(usize, BnnDotResult)>,
    pub stats: PipelineStats,
}

/// Cascade-accelerated BNN batch search using K0/K1/K2 pipeline.
pub fn bnn_cascade_search(
    query: &Fingerprint<256>,
    weights: &[Fingerprint<256>],
    top_k: usize,
    gate: &SliceGate,
) -> BnnCascadeResult {
    if weights.is_empty() {
        return BnnCascadeResult {
            matches: Vec::new(),
            stats: PipelineStats::default(),
        };
    }

    let n_candidates = weights.len();
    let mut db_words = Vec::with_capacity(n_candidates * SKU_16K_WORDS);
    for w in weights {
        db_words.extend_from_slice(&w.words);
    }

    let (kernel_matches, stats) =
        kernel_pipeline(&query.words, &db_words, n_candidates, SKU_16K_WORDS, gate);

    let total_bits = Fingerprint::<256>::BITS as u32;
    let mut results: Vec<(usize, BnnDotResult)> = kernel_matches
        .iter()
        .map(|kr| {
            let match_count = total_bits - kr.distance;
            let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
            (
                kr.index,
                BnnDotResult {
                    match_count,
                    total_bits,
                    score,
                },
            )
        })
        .collect();

    results.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);

    BnnCascadeResult {
        matches: results,
        stats,
    }
}

/// Extended result including energy/conflict decomposition.
#[derive(Clone, Debug)]
pub struct BnnEnergyResult {
    pub index: usize,
    pub dot: BnnDotResult,
    pub energy: EnergyConflict,
    pub stage: KernelStage,
}

/// Like `bnn_cascade_search` but also returns EnergyConflict decomposition.
pub fn bnn_cascade_search_with_energy(
    query: &Fingerprint<256>,
    weights: &[Fingerprint<256>],
    top_k: usize,
    gate: &SliceGate,
) -> (Vec<BnnEnergyResult>, PipelineStats) {
    if weights.is_empty() {
        return (Vec::new(), PipelineStats::default());
    }

    let n_candidates = weights.len();
    let mut db_words = Vec::with_capacity(n_candidates * SKU_16K_WORDS);
    for w in weights {
        db_words.extend_from_slice(&w.words);
    }

    let (kernel_matches, stats) =
        kernel_pipeline(&query.words, &db_words, n_candidates, SKU_16K_WORDS, gate);

    let total_bits = Fingerprint::<256>::BITS as u32;
    let mut results: Vec<BnnEnergyResult> = kernel_matches
        .iter()
        .map(|kr| {
            let match_count = total_bits - kr.distance;
            let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
            BnnEnergyResult {
                index: kr.index,
                dot: BnnDotResult {
                    match_count,
                    total_bits,
                    score,
                },
                energy: kr.energy,
                stage: kr.stage,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.dot
            .score
            .partial_cmp(&a.dot.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);

    (results, stats)
}

// ─── HDR cascade search ──────────────────────────────

/// HDR-cascade-accelerated BNN search.
pub fn bnn_hdr_search(
    query: &Fingerprint<256>,
    weights: &[Fingerprint<256>],
    threshold: u64,
    top_k: usize,
) -> Vec<(usize, BnnDotResult)> {
    use super::cascade::Cascade;

    if weights.is_empty() {
        return Vec::new();
    }

    let vec_bytes = Fingerprint::<256>::BITS / 8;
    let query_bytes = query.as_bytes();

    let mut db_bytes = Vec::with_capacity(weights.len() * vec_bytes);
    for w in weights {
        db_bytes.extend_from_slice(w.as_bytes());
    }

    let cascade = Cascade::from_threshold(threshold, vec_bytes);
    let hdr_results = cascade.query(query_bytes, &db_bytes, vec_bytes, weights.len());

    let total_bits = Fingerprint::<256>::BITS as u32;
    let mut results: Vec<(usize, BnnDotResult)> = hdr_results
        .iter()
        .map(|hr| {
            let match_count = total_bits - hr.hamming as u32;
            let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
            (
                hr.index,
                BnnDotResult {
                    match_count,
                    total_bits,
                    score,
                },
            )
        })
        .collect();

    results.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
    results
}

// ─── Binary convolution ──────────────────────────────

/// 1D binary convolution: slide a kernel over a sequence of Fingerprints.
pub fn bnn_conv1d(
    input: &[Fingerprint<256>],
    kernel: &Fingerprint<256>,
    stride: usize,
) -> Vec<BnnDotResult> {
    let stride = stride.max(1);
    (0..input.len())
        .step_by(stride)
        .map(|i| bnn_dot(&input[i], kernel))
        .collect()
}

/// 1D binary convolution over 3-channel GraphHV sequences.
pub fn bnn_conv1d_3ch(input: &[GraphHV], kernel: &GraphHV, stride: usize) -> Vec<BnnDotResult> {
    let stride = stride.max(1);
    (0..input.len())
        .step_by(stride)
        .map(|i| bnn_dot_3ch(&input[i], kernel))
        .collect()
}

/// Cascade-accelerated 1D convolution.
pub fn bnn_conv1d_cascade(
    input: &[Fingerprint<256>],
    kernel: &Fingerprint<256>,
    stride: usize,
    gate: &SliceGate,
) -> BnnCascadeResult {
    let stride = stride.max(1);
    let positions: Vec<Fingerprint<256>> = (0..input.len())
        .step_by(stride)
        .map(|i| input[i].clone())
        .collect();

    bnn_cascade_search(kernel, &positions, positions.len(), gate)
}

// ─── Tests ──────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> SplitMix64 {
        SplitMix64::new(42)
    }

    fn random_fp(rng: &mut SplitMix64) -> Fingerprint<256> {
        let mut words = [0u64; 256];
        for w in words.iter_mut() {
            *w = rng.next_u64();
        }
        Fingerprint::from_words(words)
    }

    #[test]
    fn test_bnn_dot_identical() {
        let mut rng = make_rng();
        let fp = random_fp(&mut rng);
        let result = bnn_dot(&fp, &fp);
        assert_eq!(result.match_count, 16_384);
        assert!((result.score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bnn_dot_opposite() {
        let a = Fingerprint::<256>::ones();
        let b = Fingerprint::<256>::zero();
        let result = bnn_dot(&a, &b);
        assert_eq!(result.match_count, 0);
        assert!((result.score - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bnn_dot_random_near_zero() {
        let mut rng = make_rng();
        let a = random_fp(&mut rng);
        let b = random_fp(&mut rng);
        let result = bnn_dot(&a, &b);
        assert!(
            result.score.abs() < 0.05,
            "Expected ~0.0 score for random, got {:.4}",
            result.score
        );
    }

    #[test]
    fn test_bnn_dot_3ch() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let result = bnn_dot_3ch(&a, &a);
        assert_eq!(result.match_count, 49_152);
        assert!((result.score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bnn_neuron_forward() {
        let mut rng = make_rng();
        let mut neuron = BnnNeuron::random(&mut rng);
        let input = random_fp(&mut rng);
        let score = neuron.forward(&input, false, 0.03, &mut rng);
        assert!(score.abs() < 2.0, "Score out of range: {}", score);
    }

    #[test]
    fn test_bnn_neuron_plasticity() {
        let mut rng = make_rng();
        let mut neuron = BnnNeuron::random(&mut rng);
        let input = random_fp(&mut rng);
        let initial_plastic = neuron.plastic().clone();

        for _ in 0..50 {
            neuron.forward(&input, true, 0.1, &mut rng);
        }

        assert_ne!(
            *neuron.plastic(),
            initial_plastic,
            "Plastic channel should change after learning"
        );
    }

    #[test]
    fn test_bnn_layer_winner() {
        let mut rng = make_rng();
        let layer = BnnLayer::random(10, &mut rng);
        let input = random_fp(&mut rng);
        let (winner_idx, _) = layer.winner(&input);
        assert!(winner_idx < 10);
    }

    #[test]
    fn test_bnn_batch_dot_ordering() {
        let mut rng = make_rng();
        let query = random_fp(&mut rng);
        let weights: Vec<_> = (0..20).map(|_| random_fp(&mut rng)).collect();

        let results = bnn_batch_dot(&query, &weights, 5);
        assert_eq!(results.len(), 5);

        for i in 1..results.len() {
            assert!(results[i].1.score <= results[i - 1].1.score);
        }
    }

    #[test]
    fn test_bnn_layer_forward() {
        let mut rng = make_rng();
        let mut layer = BnnLayer::random(5, &mut rng);
        assert_eq!(layer.len(), 5);

        let input = random_fp(&mut rng);
        let scores = layer.forward(&input, false, 0.03, &mut rng);
        assert_eq!(scores.len(), 5);
        for &s in &scores {
            assert!(s.abs() < 2.0, "Score out of range: {}", s);
        }
    }

    #[test]
    fn test_bnn_dot_matches_scalar() {
        let mut rng = make_rng();
        let mut words_a = [0u64; 256];
        let mut words_b = [0u64; 256];
        for i in 0..256 {
            words_a[i] = rng.next_u64();
            words_b[i] = rng.next_u64();
        }
        let a = Fingerprint::from_words(words_a);
        let b = Fingerprint::from_words(words_b);

        let scalar_xor_pop: u32 = words_a
            .iter()
            .zip(words_b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum();
        let scalar_match = 16_384 - scalar_xor_pop;
        let scalar_score = (2.0 * scalar_match as f32 / 16_384.0) - 1.0;

        let result = bnn_dot(&a, &b);
        assert_eq!(result.match_count, scalar_match);
        assert!((result.score - scalar_score).abs() < f32::EPSILON);
    }

    #[test]
    fn test_batch_dot_finds_self() {
        let mut rng = make_rng();
        let target = random_fp(&mut rng);

        let mut weights: Vec<_> = (0..50).map(|_| random_fp(&mut rng)).collect();
        weights[25] = target.clone();

        let results = bnn_batch_dot(&target, &weights, 1);
        assert_eq!(results[0].0, 25);
        assert!((results[0].1.score - 1.0).abs() < f32::EPSILON);
    }

    // ─── Cascade search tests ──────────────────────────

    #[test]
    fn test_cascade_search_finds_exact_match() {
        let mut rng = make_rng();
        let query = random_fp(&mut rng);

        let mut weights: Vec<_> = (0..100).map(|_| random_fp(&mut rng)).collect();
        weights[42] = query.clone();

        let gate = SliceGate::sku_16k();
        let result = bnn_cascade_search(&query, &weights, 5, &gate);

        assert!(
            result
                .matches
                .iter()
                .any(|(idx, dot)| *idx == 42 && (dot.score - 1.0).abs() < f32::EPSILON),
            "Exact match at index 42 not found"
        );
    }

    #[test]
    fn test_cascade_search_zero_false_negatives() {
        let mut rng = make_rng();
        let mut query_words = [0u64; 256];
        for w in query_words.iter_mut() {
            *w = rng.next_u64();
        }
        let query = Fingerprint::from_words(query_words);

        let mut weights: Vec<_> = (0..200).map(|_| random_fp(&mut rng)).collect();
        weights[10] = query.clone();
        let mut near1 = query_words;
        near1[50] ^= 1;
        weights[50] = Fingerprint::from_words(near1);
        let mut near2 = query_words;
        near2[100] ^= 1;
        weights[150] = Fingerprint::from_words(near2);

        let gate = SliceGate::sku_16k();
        let cascade = bnn_cascade_search(&query, &weights, 200, &gate);

        assert!(cascade.matches.iter().any(|(i, _)| *i == 10));
        assert!(cascade.matches.iter().any(|(i, _)| *i == 50));
        assert!(cascade.matches.iter().any(|(i, _)| *i == 150));
    }

    #[test]
    fn test_cascade_search_with_energy() {
        let mut rng = make_rng();
        let query = random_fp(&mut rng);

        let mut weights: Vec<_> = (0..50).map(|_| random_fp(&mut rng)).collect();
        weights[25] = query.clone();

        let gate = SliceGate::sku_16k();
        let (results, stats) = bnn_cascade_search_with_energy(&query, &weights, 10, &gate);

        let exact = results.iter().find(|r| r.index == 25);
        assert!(exact.is_some(), "Exact match not found in energy results");
        let exact = exact.unwrap();
        assert_eq!(exact.energy.conflict, 0);
        assert_eq!(exact.energy.agreement, exact.energy.energy_a);
        assert_eq!(stats.total, 50);
    }

    #[test]
    fn test_cascade_search_empty() {
        let query = Fingerprint::<256>::zero();
        let gate = SliceGate::sku_16k();
        let result = bnn_cascade_search(&query, &[], 10, &gate);
        assert!(result.matches.is_empty());
    }

    // ─── Network tests ──────────────────────────────────

    #[test]
    fn test_network_creation() {
        let mut rng = make_rng();
        let net = BnnNetwork::new(&[10, 5, 3], &mut rng);
        assert_eq!(net.depth(), 3);
        assert_eq!(net.layers[0].len(), 10);
        assert_eq!(net.layers[1].len(), 5);
        assert_eq!(net.layers[2].len(), 3);
    }

    #[test]
    fn test_network_forward() {
        let mut rng = make_rng();
        let mut net = BnnNetwork::new(&[8, 4], &mut rng);
        let input = random_fp(&mut rng);

        let (winner_idx, result) = net.forward(&input, false, 0.03, &mut rng);
        assert!(winner_idx < 4);
        assert!(result.score.abs() <= 1.0 + f32::EPSILON);
    }

    #[test]
    fn test_network_predict() {
        let mut rng = make_rng();
        let net = BnnNetwork::new(&[10, 5, 3], &mut rng);
        let input = random_fp(&mut rng);

        let layer_results = net.predict(&input);
        assert_eq!(layer_results.len(), 3);
        assert!(layer_results[0].0 < 10);
        assert!(layer_results[1].0 < 5);
        assert!(layer_results[2].0 < 3);
    }

    #[test]
    fn test_network_learning() {
        let mut rng = make_rng();
        let mut net = BnnNetwork::new(&[5, 3], &mut rng);
        let input = random_fp(&mut rng);
        let before = net.layers[0].neurons[0].plastic().clone();

        for _ in 0..50 {
            net.forward(&input, true, 0.1, &mut rng);
        }

        let after = net.layers[0].neurons[0].plastic().clone();
        assert_ne!(before, after, "Network should learn from repeated input");
    }

    // ─── CAM index tests ──────────────────────────────

    #[test]
    fn test_build_cam_index() {
        let mut rng = make_rng();
        let layer = BnnLayer::random(50, &mut rng);
        let cam = layer.build_cam_index(42);
        assert_eq!(cam.len(), 50);
    }

    #[test]
    fn test_winner_cam_finds_match() {
        let mut rng = make_rng();
        let mut layer = BnnLayer::random(100, &mut rng);
        let target = random_fp(&mut rng);
        layer.neurons[42] = BnnNeuron::from_weights(target.clone(), &mut rng);

        let cam = layer.build_cam_index(123);
        let result = layer.winner_cam(&target, &cam, 20);
        assert!(result.is_some());
        let (_, dot) = result.unwrap();
        assert!(
            dot.score > 0.5,
            "Expected high score for planted match, got {:.4}",
            dot.score
        );
    }

    // ─── Convolution tests ──────────────────────────────

    #[test]
    fn test_conv1d_basic() {
        let mut rng = make_rng();
        let sequence: Vec<_> = (0..10).map(|_| random_fp(&mut rng)).collect();
        let kernel = sequence[5].clone();
        let results = bnn_conv1d(&sequence, &kernel, 1);

        assert_eq!(results.len(), 10);
        assert!(
            (results[5].score - 1.0).abs() < f32::EPSILON,
            "Expected 1.0 at pos 5, got {:.4}",
            results[5].score
        );
    }

    #[test]
    fn test_conv1d_stride() {
        let mut rng = make_rng();
        let sequence: Vec<_> = (0..20).map(|_| random_fp(&mut rng)).collect();
        let kernel = sequence[0].clone();

        assert_eq!(bnn_conv1d(&sequence, &kernel, 1).len(), 20);
        assert_eq!(bnn_conv1d(&sequence, &kernel, 3).len(), 7);
        assert_eq!(bnn_conv1d(&sequence, &kernel, 5).len(), 4);
    }

    #[test]
    fn test_conv1d_3ch() {
        let mut rng = make_rng();
        let sequence: Vec<_> = (0..5).map(|_| GraphHV::random(&mut rng)).collect();
        let kernel = sequence[2].clone();

        let results = bnn_conv1d_3ch(&sequence, &kernel, 1);
        assert_eq!(results.len(), 5);
        assert!((results[2].score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_conv1d_cascade() {
        let mut rng = make_rng();
        let sequence: Vec<_> = (0..100).map(|_| random_fp(&mut rng)).collect();
        let kernel = sequence[50].clone();
        let gate = SliceGate::sku_16k();

        let result = bnn_conv1d_cascade(&sequence, &kernel, 1, &gate);
        assert!(
            result
                .matches
                .iter()
                .any(|(idx, dot)| *idx == 50 && (dot.score - 1.0).abs() < f32::EPSILON),
            "Exact match at position 50 not found"
        );
    }

    // ─── HDR search test ──────────────────────────────

    #[test]
    fn test_hdr_search_finds_exact() {
        let mut rng = make_rng();
        let query = random_fp(&mut rng);
        let mut weights: Vec<_> = (0..50).map(|_| random_fp(&mut rng)).collect();
        weights[25] = query.clone();

        let results = bnn_hdr_search(&query, &weights, 16384, 10);
        assert!(
            results.iter().any(|(i, _)| *i == 25),
            "HDR search should find exact match at 25"
        );
    }
}
