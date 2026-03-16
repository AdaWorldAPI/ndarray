//! Binary Neural Network (BNN) inference primitives.
//!
//! Implements BNN convolution and inference using XNOR + popcount operations,
//! directly leveraging `Fingerprint<256>` containers.
//!
//! The fundamental operation: `dot(a, w) = 2 * popcount(XNOR(a, w)) - total_bits`

use super::fingerprint::Fingerprint;

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

/// Binary dot product via XNOR + popcount on a single fingerprint.
///
/// `dot(a, w) = 2 * popcount(XNOR(a, w)) - total_bits`
#[inline]
pub fn bnn_dot(activation: &Fingerprint<256>, weight: &Fingerprint<256>) -> BnnDotResult {
    let total_bits = Fingerprint::<256>::BITS as u32; // 16,384
    let xor_popcount = super::bitwise::hamming_distance_raw(activation.as_bytes(), weight.as_bytes()) as u32;
    let match_count = total_bits - xor_popcount;
    let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
    BnnDotResult { match_count, total_bits, score }
}

/// Binary dot product across 3 fingerprints (SPO channels).
pub fn bnn_dot_3ch(
    a: [&Fingerprint<256>; 3],
    w: [&Fingerprint<256>; 3],
) -> BnnDotResult {
    let total_bits = (Fingerprint::<256>::BITS * 3) as u32;
    let mut xor_total = 0u32;
    for ch in 0..3 {
        xor_total += super::bitwise::hamming_distance_raw(a[ch].as_bytes(), w[ch].as_bytes()) as u32;
    }
    let match_count = total_bits - xor_total;
    let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
    BnnDotResult { match_count, total_bits, score }
}

/// A binary neuron.
pub struct BnnNeuron {
    /// Binary weight pattern.
    pub weights: Fingerprint<256>,
    /// Current activation.
    pub activation: Fingerprint<256>,
    /// Plastic running average (learned prototype).
    pub plastic: Fingerprint<256>,
    /// Bias (amplitude correction).
    pub bias: f32,
    /// Activation threshold.
    pub threshold: f32,
}

impl BnnNeuron {
    /// Create with given weights.
    pub fn from_weights(weights: Fingerprint<256>) -> Self {
        Self {
            plastic: weights.clone(),
            activation: Fingerprint::zero(),
            weights,
            bias: 0.0,
            threshold: 0.0,
        }
    }

    /// Forward pass: XNOR+popcount → threshold → update activation.
    pub fn forward(&mut self, input: &Fingerprint<256>) -> f32 {
        let dot = bnn_dot(input, &self.weights);
        let pre_activation = dot.score + self.bias;
        if pre_activation > self.threshold {
            self.activation = input.clone();
        } else {
            self.activation = !(input);
        }
        pre_activation
    }
}

/// A layer of BNN neurons.
pub struct BnnLayer {
    pub neurons: Vec<BnnNeuron>,
}

impl BnnLayer {
    /// Create a layer with `n` neurons, each with random weights from a seed.
    pub fn new(n: usize, seed: u64) -> Self {
        let mut neurons = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            let mut words = [0u64; 256];
            for w in words.iter_mut() {
                state = state.wrapping_add(0x9E3779B97F4A7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
                *w = z ^ (z >> 31);
            }
            neurons.push(BnnNeuron::from_weights(Fingerprint::from_words(words)));
        }
        Self { neurons }
    }

    /// Forward pass: compute all neuron activations.
    pub fn forward(&mut self, input: &Fingerprint<256>) -> Vec<f32> {
        self.neurons.iter_mut().map(|n| n.forward(input)).collect()
    }
}

/// Simple BNN network (stack of layers).
pub struct BnnNetwork {
    pub layers: Vec<BnnLayer>,
}

impl BnnNetwork {
    pub fn new(layer_sizes: &[usize], seed: u64) -> Self {
        let layers = layer_sizes.iter().enumerate()
            .map(|(i, &size)| BnnLayer::new(size, seed.wrapping_add(i as u64 * 1000)))
            .collect();
        Self { layers }
    }

    /// Forward pass through all layers.
    /// Returns the final layer's activation scores.
    pub fn forward(&mut self, input: &Fingerprint<256>) -> Vec<f32> {
        let mut current = input.clone();
        let mut scores = Vec::new();
        for layer in &mut self.layers {
            scores = layer.forward(&current);
            // Use first neuron's activation as next layer input
            if !layer.neurons.is_empty() {
                current = layer.neurons[0].activation.clone();
            }
        }
        scores
    }
}

/// Batch BNN dot products: compute scores for multiple candidates.
pub fn bnn_batch_dot(
    query: &Fingerprint<256>,
    candidates: &[Fingerprint<256>],
) -> Vec<BnnDotResult> {
    candidates.iter().map(|c| bnn_dot(query, c)).collect()
}

/// Cascade search using BNN dot scores.
pub fn bnn_cascade_search(
    query: &Fingerprint<256>,
    database: &[Fingerprint<256>],
    threshold: f32,
    top_k: usize,
) -> Vec<(usize, BnnDotResult)> {
    let mut results: Vec<(usize, BnnDotResult)> = database.iter()
        .enumerate()
        .map(|(i, c)| (i, bnn_dot(query, c)))
        .filter(|(_, r)| r.score >= threshold)
        .collect();
    results.sort_unstable_by(|a, b| b.1.score.partial_cmp(&a.1.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(top_k);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bnn_dot_identical() {
        let fp = Fingerprint::<256>::ones();
        let result = bnn_dot(&fp, &fp);
        assert_eq!(result.match_count, 16384);
        assert!((result.score - 1.0).abs() < 0.001);
    }

    #[test]
    fn bnn_dot_opposite() {
        let a = Fingerprint::<256>::ones();
        let b = Fingerprint::<256>::zero();
        let result = bnn_dot(&a, &b);
        assert_eq!(result.match_count, 0);
        assert!((result.score + 1.0).abs() < 0.001);
    }

    #[test]
    fn bnn_dot_3ch_identical() {
        let fp = Fingerprint::<256>::ones();
        let result = bnn_dot_3ch([&fp, &fp, &fp], [&fp, &fp, &fp]);
        assert!((result.score - 1.0).abs() < 0.001);
    }

    #[test]
    fn bnn_neuron_forward() {
        let weights = Fingerprint::<256>::ones();
        let mut neuron = BnnNeuron::from_weights(weights);
        let input = Fingerprint::<256>::ones();
        let score = neuron.forward(&input);
        assert!(score > 0.0);
    }

    #[test]
    fn bnn_layer_forward() {
        let mut layer = BnnLayer::new(4, 42);
        let input = Fingerprint::<256>::ones();
        let scores = layer.forward(&input);
        assert_eq!(scores.len(), 4);
    }

    #[test]
    fn bnn_network_forward() {
        let mut net = BnnNetwork::new(&[4, 2], 42);
        let input = Fingerprint::<256>::ones();
        let scores = net.forward(&input);
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn bnn_cascade_search_finds_match() {
        let query = Fingerprint::<256>::ones();
        let database = vec![
            Fingerprint::<256>::zero(),
            Fingerprint::<256>::ones(),
            Fingerprint::<256>::zero(),
        ];
        let results = bnn_cascade_search(&query, &database, 0.5, 10);
        assert!(results.iter().any(|(i, _)| *i == 1));
    }

    #[test]
    fn bnn_batch_dot_len() {
        let query = Fingerprint::<256>::ones();
        let candidates = vec![Fingerprint::<256>::zero(); 5];
        let results = bnn_batch_dot(&query, &candidates);
        assert_eq!(results.len(), 5);
    }
}
