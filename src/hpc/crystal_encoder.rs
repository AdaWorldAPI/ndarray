//! Crystal Encoding Pipeline: SPO Crystal Encoding for holographic cognition.
//!
//! Three-phase pipeline:
//! 1. **External Embeddings** — project dense float vectors to binary fingerprints,
//!    then absorb into S/P/O planes of a [`Node`].
//! 2. **Distillation** — gradient-free knowledge distillation from teacher nodes
//!    to a student encoder using Hamming loss + causal divergence.
//! 3. **Pure Crystal Encoding** — hash-based word encoding via a 65-entry
//!    NSM semantic-primes codebook, bundled into SPO nodes.

use super::fingerprint::Fingerprint;
use super::node::Node;
use super::plane::Plane;

// ============================================================================
// Role enum
// ============================================================================

/// SPO role for absorbing a fingerprint into a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// Subject plane.
    Subject,
    /// Predicate plane.
    Predicate,
    /// Object plane.
    Object,
}

// ============================================================================
// SplitMix64 — deterministic RNG
// ============================================================================

/// Simple SplitMix64 RNG for deterministic random generation.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        // Returns a value in [-1.0, 1.0)
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

// ============================================================================
// Phase 1: CrystalEncoder — External Embeddings → Projection → Node Absorb
// ============================================================================

/// Number of u64 words in the standard fingerprint (256 words = 16384 bits).
const FP_WORDS: usize = 256;

/// Crystal encoder: projects dense embeddings to binary fingerprints and
/// absorbs them into S/P/O node planes.
///
/// Holds a random projection matrix seeded deterministically. The matrix
/// has `BITS` rows x `embedding_dim` columns, stored as flat `Vec<f32>`.
pub struct CrystalEncoder {
    /// Projection weights: `[BITS * embedding_dim]` flat row-major.
    /// Row `i` is the hyperplane normal for output bit `i`.
    projection: Vec<f32>,
    /// Dimensionality of the input embedding.
    pub embedding_dim: usize,
    /// Seed used to generate the projection matrix.
    seed: u64,
}

/// Total bits in a `Fingerprint<256>`.
const TOTAL_BITS: usize = FP_WORDS * 64; // 16384

impl CrystalEncoder {
    /// Create a new encoder with a random projection matrix.
    ///
    /// # Arguments
    /// * `embedding_dim` — dimensionality of the input dense embedding.
    /// * `seed` — deterministic seed for the projection matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::crystal_encoder::CrystalEncoder;
    ///
    /// let enc = CrystalEncoder::new(128, 42);
    /// assert_eq!(enc.embedding_dim, 128);
    /// ```
    pub fn new(embedding_dim: usize, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let total = TOTAL_BITS * embedding_dim;
        let mut projection = Vec::with_capacity(total);
        for _ in 0..total {
            projection.push(rng.next_f32());
        }
        Self {
            projection,
            embedding_dim,
            seed,
        }
    }

    /// Project a dense float embedding to a binary fingerprint via sign of
    /// random projection.
    ///
    /// For each output bit `i`, computes `dot(projection[i], embedding)`.
    /// If the dot product is positive, the bit is set to 1; otherwise 0.
    ///
    /// # Panics
    /// Panics if `embedding.len() != self.embedding_dim`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::crystal_encoder::CrystalEncoder;
    ///
    /// let enc = CrystalEncoder::new(4, 42);
    /// let fp = enc.encode_embedding(&[1.0, -0.5, 0.3, 0.8]);
    /// assert!(!fp.is_zero()); // should have some bits set
    /// ```
    pub fn encode_embedding(&self, embedding: &[f32]) -> Fingerprint<FP_WORDS> {
        assert_eq!(
            embedding.len(),
            self.embedding_dim,
            "embedding length {} != encoder dim {}",
            embedding.len(),
            self.embedding_dim
        );

        let mut words = [0u64; FP_WORDS];
        let dim = self.embedding_dim;

        for bit_idx in 0..TOTAL_BITS {
            let row_start = bit_idx * dim;
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += self.projection[row_start + d] * embedding[d];
            }
            if dot > 0.0 {
                let word = bit_idx / 64;
                let bit = bit_idx % 64;
                words[word] |= 1u64 << bit;
            }
        }

        Fingerprint::from_words(words)
    }

    /// Absorb a fingerprint into the appropriate S/P/O plane of a node.
    ///
    /// This feeds the fingerprint as evidence into the plane's i8 accumulator,
    /// reinforcing or weakening bits according to the plane's encounter logic.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::crystal_encoder::{CrystalEncoder, Role};
    /// use ndarray::hpc::node::Node;
    ///
    /// let enc = CrystalEncoder::new(4, 42);
    /// let fp = enc.encode_embedding(&[1.0, -0.5, 0.3, 0.8]);
    /// let mut node = Node::new();
    /// CrystalEncoder::absorb_into_node(&fp, &mut node, Role::Subject);
    /// assert!(node.s.encounters() > 0);
    /// ```
    pub fn absorb_into_node(
        fingerprint: &Fingerprint<FP_WORDS>,
        node: &mut Node,
        role: Role,
    ) {
        let plane: &mut Plane = match role {
            Role::Subject => &mut node.s,
            Role::Predicate => &mut node.p,
            Role::Object => &mut node.o,
        };
        plane.encounter_bits(fingerprint);
    }

    /// Return the seed used to generate the projection matrix.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Flip a single weight in the projection matrix.
    /// Used by the distillation optimizer.
    fn flip_weight(&mut self, index: usize) {
        self.projection[index] = -self.projection[index];
    }

    /// Encode an embedding, absorb it into a node under the given role, and
    /// return both the fingerprint and the mutated node.
    ///
    /// Convenience wrapper that chains `encode_embedding` and `absorb_into_node`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::crystal_encoder::{CrystalEncoder, Role};
    /// use ndarray::hpc::node::Node;
    ///
    /// let enc = CrystalEncoder::new(4, 42);
    /// let mut node = Node::new();
    /// let fp = enc.encode_and_absorb(&[1.0, -0.5, 0.3, 0.8], &mut node, Role::Subject);
    /// assert!(node.s.encounters() > 0);
    /// assert!(!fp.is_zero());
    /// ```
    pub fn encode_and_absorb(
        &self,
        embedding: &[f32],
        node: &mut Node,
        role: Role,
    ) -> Fingerprint<FP_WORDS> {
        let fp = self.encode_embedding(embedding);
        Self::absorb_into_node(&fp, node, role);
        fp
    }

    /// Search a database of nodes for the most similar to a query node.
    ///
    /// Computes SPO Hamming distance between the query node and each database
    /// node, returning `(index, distance)` pairs sorted by distance ascending,
    /// limited to `top_k` results.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::crystal_encoder::{CrystalEncoder, Role};
    /// use ndarray::hpc::node::Node;
    ///
    /// let enc = CrystalEncoder::new(4, 42);
    /// let mut query = Node::new();
    /// enc.encode_and_absorb(&[1.0, 0.0, 0.0, 0.0], &mut query, Role::Subject);
    /// enc.encode_and_absorb(&[0.0, 1.0, 0.0, 0.0], &mut query, Role::Predicate);
    /// enc.encode_and_absorb(&[0.0, 0.0, 1.0, 0.0], &mut query, Role::Object);
    ///
    /// let mut db = vec![Node::random(1), Node::random(2), query.clone()];
    /// let results = CrystalEncoder::search_similar(&mut query, &mut db, 2);
    /// assert!(!results.is_empty());
    /// assert_eq!(results[0].1, 0); // exact match should have distance 0
    /// ```
    pub fn search_similar(
        query: &mut Node,
        database: &mut [Node],
        top_k: usize,
    ) -> Vec<(usize, u32)> {
        let mut results: Vec<(usize, u32)> = database
            .iter_mut()
            .enumerate()
            .map(|(i, db_node)| {
                let d = spo_hamming(query, db_node);
                (i, d)
            })
            .collect();

        results.sort_unstable_by_key(|&(_, d)| d);
        results.truncate(top_k);
        results
    }
}

/// Full pipeline: encode three embeddings (S/P/O) into a node, then search
/// a database for the closest matches.
///
/// This wires the complete flow: projection -> node absorption -> search.
///
/// # Arguments
/// * `encoder` — the CrystalEncoder with a loaded projection matrix.
/// * `subject_emb` — dense float embedding for the Subject plane.
/// * `predicate_emb` — dense float embedding for the Predicate plane.
/// * `object_emb` — dense float embedding for the Object plane.
/// * `database` — mutable slice of database nodes to search against.
/// * `top_k` — maximum number of results to return.
///
/// # Returns
/// A tuple of `(query_node, results)` where results are `(index, distance)` pairs.
///
/// # Example
///
/// ```
/// use ndarray::hpc::crystal_encoder::{CrystalEncoder, pipeline_encode_search};
/// use ndarray::hpc::node::Node;
///
/// let enc = CrystalEncoder::new(4, 42);
/// let mut db = vec![Node::random(1), Node::random(2)];
/// let (query, results) = pipeline_encode_search(
///     &enc,
///     &[1.0, 0.0, 0.0, 0.0],
///     &[0.0, 1.0, 0.0, 0.0],
///     &[0.0, 0.0, 1.0, 0.0],
///     &mut db,
///     5,
/// );
/// assert!(query.s.encounters() > 0);
/// assert!(!results.is_empty());
/// ```
pub fn pipeline_encode_search(
    encoder: &CrystalEncoder,
    subject_emb: &[f32],
    predicate_emb: &[f32],
    object_emb: &[f32],
    database: &mut [Node],
    top_k: usize,
) -> (Node, Vec<(usize, u32)>) {
    let mut query_node = Node::new();
    encoder.encode_and_absorb(subject_emb, &mut query_node, Role::Subject);
    encoder.encode_and_absorb(predicate_emb, &mut query_node, Role::Predicate);
    encoder.encode_and_absorb(object_emb, &mut query_node, Role::Object);

    let results = CrystalEncoder::search_similar(&mut query_node, database, top_k);
    (query_node, results)
}

// ============================================================================
// Phase 2: Distillation
// ============================================================================

/// Compute the total SPO Hamming distance between two nodes.
///
/// Sums the Hamming distance of the bits fingerprints across all three planes.
fn spo_hamming(a: &mut Node, b: &mut Node) -> u32 {
    let mut total = 0u32;
    total += a.s.bits().hamming_distance(b.s.bits());
    total += a.p.bits().hamming_distance(b.p.bits());
    total += a.o.bits().hamming_distance(b.o.bits());
    total
}

/// Distillation loss: Hamming(student_SPO, teacher_SPO) + lambda * causal_divergence.
///
/// Causal divergence is approximated as the asymmetry between S-plane and O-plane
/// Hamming distances (causal direction should be preserved by the student).
fn distillation_loss(student: &mut Node, teacher: &mut Node, lambda: f32) -> f32 {
    let hamming = spo_hamming(student, teacher) as f32;

    // Causal divergence: difference between subject and object plane distances.
    // If teacher has a causal asymmetry (S != O), student should preserve it.
    let d_s = student.s.bits().hamming_distance(teacher.s.bits()) as f32;
    let d_o = student.o.bits().hamming_distance(teacher.o.bits()) as f32;

    // Teacher's own S/O asymmetry as reference
    let teacher_s_pop = teacher.s.bits().popcount() as f32;
    let teacher_o_pop = teacher.o.bits().popcount() as f32;
    let teacher_asym = (teacher_s_pop - teacher_o_pop).abs();

    let student_asym = (d_s - d_o).abs();
    let causal_div = (student_asym - teacher_asym).abs();

    hamming + lambda * causal_div
}

/// Run gradient-free distillation from teacher nodes into a student encoder.
///
/// For each epoch, iterates over teacher nodes and attempts binary weight
/// flips that reduce the combined Hamming + causal divergence loss.
///
/// # Arguments
/// * `teacher_nodes` — slice of teacher nodes to distill from.
/// * `student_encoder` — the student encoder whose projection weights are optimized.
/// * `epochs` — number of distillation passes.
///
/// # Returns
/// A vector of per-epoch average loss values for convergence tracking.
///
/// # Example
///
/// ```
/// use ndarray::hpc::crystal_encoder::{CrystalEncoder, Role, distill};
/// use ndarray::hpc::node::Node;
///
/// let teacher = Node::random(42);
/// let mut student = CrystalEncoder::new(8, 99);
/// let losses = distill(&[teacher], &mut student, 3);
/// assert_eq!(losses.len(), 3);
/// ```
pub fn distill(
    teacher_nodes: &[Node],
    student_encoder: &mut CrystalEncoder,
    epochs: usize,
) -> Vec<f32> {
    let lambda = 0.1f32;
    let dim = student_encoder.embedding_dim;
    let total_weights = student_encoder.projection.len();

    // Create a fixed set of pseudo-embeddings from teacher nodes (one per teacher).
    // We use the teacher's S-plane accumulator as the embedding source.
    let embeddings: Vec<Vec<f32>> = teacher_nodes
        .iter()
        .map(|t| {
            let acc = t.s.acc();
            (0..dim).map(|i| acc[i % acc.len()] as f32 / 127.0).collect()
        })
        .collect();

    let mut losses = Vec::with_capacity(epochs);
    let mut rng = SplitMix64::new(student_encoder.seed.wrapping_add(0xDEAD));

    for _epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;

        for (t_idx, teacher) in teacher_nodes.iter().enumerate() {
            let emb = &embeddings[t_idx];

            // Current student encoding
            let fp_s = student_encoder.encode_embedding(emb);
            let fp_p = {
                // Slightly rotated embedding for predicate
                let shifted: Vec<f32> = emb.iter().enumerate()
                    .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
                    .collect();
                student_encoder.encode_embedding(&shifted)
            };
            let fp_o = {
                // Reversed embedding for object
                let rev: Vec<f32> = emb.iter().rev().copied().collect();
                student_encoder.encode_embedding(&rev)
            };

            let mut student_node = Node::new();
            CrystalEncoder::absorb_into_node(&fp_s, &mut student_node, Role::Subject);
            CrystalEncoder::absorb_into_node(&fp_p, &mut student_node, Role::Predicate);
            CrystalEncoder::absorb_into_node(&fp_o, &mut student_node, Role::Object);

            let mut teacher_clone = teacher.clone();
            let current_loss = distillation_loss(&mut student_node, &mut teacher_clone, lambda);
            epoch_loss += current_loss;

            // Try a random weight flip and keep it if loss improves
            let num_trials = (total_weights / 100).clamp(1, 50);
            for _ in 0..num_trials {
                let idx = (rng.next_u64() as usize) % total_weights;
                student_encoder.flip_weight(idx);

                let fp_s2 = student_encoder.encode_embedding(emb);
                let shifted: Vec<f32> = emb.iter().enumerate()
                    .map(|(i, &v)| if i % 2 == 0 { v } else { -v })
                    .collect();
                let fp_p2 = student_encoder.encode_embedding(&shifted);
                let rev: Vec<f32> = emb.iter().rev().copied().collect();
                let fp_o2 = student_encoder.encode_embedding(&rev);

                let mut candidate = Node::new();
                CrystalEncoder::absorb_into_node(&fp_s2, &mut candidate, Role::Subject);
                CrystalEncoder::absorb_into_node(&fp_p2, &mut candidate, Role::Predicate);
                CrystalEncoder::absorb_into_node(&fp_o2, &mut candidate, Role::Object);

                let mut teacher_clone2 = teacher.clone();
                let new_loss = distillation_loss(&mut candidate, &mut teacher_clone2, lambda);

                if new_loss >= current_loss {
                    // Revert the flip
                    student_encoder.flip_weight(idx);
                }
            }
        }

        let n = teacher_nodes.len().max(1) as f32;
        losses.push(epoch_loss / n);
    }

    losses
}

// ============================================================================
// Phase 3: Pure Crystal Encoding — VerbCodebook + word/sentence encoding
// ============================================================================

/// The 65 NSM (Natural Semantic Metalanguage) semantic primes.
///
/// These are the universal semantic primitives proposed by Anna Wierzbicka.
const NSM_PRIMES: [&str; 65] = [
    // Substantives
    "I", "you", "someone", "something", "people", "body",
    // Determiners
    "this", "the same", "other", "else",
    // Quantifiers
    "one", "two", "some", "all", "much", "many",
    // Evaluators
    "good", "bad",
    // Descriptors
    "big", "small",
    // Mental predicates
    "think", "know", "want", "feel", "see", "hear",
    // Speech
    "say", "words", "true",
    // Actions/events/movement
    "do", "happen", "move",
    // Existence/possession
    "there is", "have",
    // Life/death
    "live", "die",
    // Time
    "when", "now", "before", "after", "a long time", "a short time", "for some time", "moment",
    // Space
    "where", "here", "above", "below", "far", "near", "side", "inside", "touch",
    // Logical concepts
    "not", "maybe", "can", "because", "if",
    // Intensifier/augmentor
    "very", "more",
    // Similarity
    "like", "as",
    // Taxonomy/partonomy
    "kind of", "part of",
    // Relational
    "way",
];

/// Codebook of 65 NSM semantic primes as binary fingerprints.
///
/// Each entry is a deterministic `Fingerprint<256>` generated from a seeded
/// RNG keyed to the prime's index. This gives each semantic prime a unique,
/// reproducible binary representation in 16384-bit space.
///
/// # Example
///
/// ```
/// use ndarray::hpc::crystal_encoder::NsmCodebook;
///
/// let cb = NsmCodebook::new();
/// assert_eq!(cb.len(), 65);
/// let fp = cb.get_prime(0); // "I"
/// assert!(!fp.is_zero());
/// ```
pub struct NsmCodebook {
    /// 65 fingerprints, one per NSM prime.
    entries: Vec<Fingerprint<FP_WORDS>>,
}

impl NsmCodebook {
    /// Build the codebook. Each prime gets a deterministic fingerprint
    /// seeded from `0xNSM_CODEBOOK + prime_index`.
    pub fn new() -> Self {
        let base_seed: u64 = 0x4E53_4D5F_C0DE;
        let entries: Vec<Fingerprint<FP_WORDS>> = (0..NSM_PRIMES.len())
            .map(|idx| {
                let seed = base_seed.wrapping_add(idx as u64);
                let mut rng = SplitMix64::new(seed);
                let mut words = [0u64; FP_WORDS];
                for w in words.iter_mut() {
                    *w = rng.next_u64();
                }
                Fingerprint::from_words(words)
            })
            .collect();
        Self { entries }
    }

    /// Number of entries (always 65).
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the codebook is empty (always false).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the fingerprint for a specific prime by index.
    ///
    /// # Panics
    /// Panics if `index >= 65`.
    pub fn get_prime(&self, index: usize) -> &Fingerprint<FP_WORDS> {
        &self.entries[index]
    }

    /// Get the prime name by index.
    ///
    /// # Panics
    /// Panics if `index >= 65`.
    pub fn prime_name(&self, index: usize) -> &'static str {
        NSM_PRIMES[index]
    }

    /// Find the closest codebook entry to a given fingerprint (by Hamming distance).
    pub fn nearest(&self, fp: &Fingerprint<FP_WORDS>) -> (usize, u32) {
        let mut best_idx = 0;
        let mut best_dist = u32::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            let d = fp.hamming_distance(entry);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        (best_idx, best_dist)
    }
}

impl Default for NsmCodebook {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode a word to a binary fingerprint using hash-based mapping.
///
/// The word is hashed (blake3) to select a codebook entry, then the entry
/// is XOR-permuted with a hash-derived fingerprint to produce a unique
/// encoding that retains similarity to the base prime.
///
/// # Example
///
/// ```
/// use ndarray::hpc::crystal_encoder::{NsmCodebook, encode_word};
///
/// let cb = NsmCodebook::new();
/// let fp = encode_word("hello", &cb);
/// assert!(!fp.is_zero());
/// ```
pub fn encode_word(word: &str, codebook: &NsmCodebook) -> Fingerprint<FP_WORDS> {
    // Hash the word to get a deterministic seed
    let hash = blake3::hash(word.as_bytes());
    let hash_bytes = hash.as_bytes();

    // Select codebook entry: first 8 bytes of hash mod 65
    let selector = u64::from_le_bytes([
        hash_bytes[0], hash_bytes[1], hash_bytes[2], hash_bytes[3],
        hash_bytes[4], hash_bytes[5], hash_bytes[6], hash_bytes[7],
    ]);
    let prime_idx = (selector % NSM_PRIMES.len() as u64) as usize;
    let base = codebook.get_prime(prime_idx).clone();

    // Generate a word-specific permutation fingerprint from the hash
    let mut hasher = blake3::Hasher::new();
    hasher.update(word.as_bytes());
    hasher.update(&[0xFF; 4]); // domain separator
    let mut xof = hasher.finalize_xof();
    let mut perm_bytes = vec![0u8; FP_WORDS * 8];
    xof.fill(&mut perm_bytes);
    let permutation = Fingerprint::<FP_WORDS>::from_bytes(&perm_bytes);

    // XOR the base prime with the word-specific permutation
    base ^ permutation
}

/// Encode a sentence (list of words) into an SPO node.
///
/// The first word is mapped to the Subject plane, the second to Predicate,
/// the third to Object. Additional words are cyclically distributed across
/// the three planes. Each word's fingerprint is absorbed as evidence.
///
/// # Example
///
/// ```
/// use ndarray::hpc::crystal_encoder::{NsmCodebook, encode_sentence};
///
/// let cb = NsmCodebook::new();
/// let node = encode_sentence(&["the", "cat", "sat"], &cb);
/// assert!(node.s.encounters() > 0);
/// assert!(node.p.encounters() > 0);
/// assert!(node.o.encounters() > 0);
/// ```
pub fn encode_sentence(words: &[&str], codebook: &NsmCodebook) -> Node {
    let roles = [Role::Subject, Role::Predicate, Role::Object];
    let mut node = Node::new();

    for (i, word) in words.iter().enumerate() {
        let fp = encode_word(word, codebook);
        let role = roles[i % 3];
        CrystalEncoder::absorb_into_node(&fp, &mut node, role);
    }

    node
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::node::SPO;

    // -- Phase 1 tests -------------------------------------------------------

    #[test]
    fn encoder_deterministic() {
        let enc1 = CrystalEncoder::new(16, 42);
        let enc2 = CrystalEncoder::new(16, 42);
        let emb = vec![1.0f32; 16];
        let fp1 = enc1.encode_embedding(&emb);
        let fp2 = enc2.encode_embedding(&emb);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn encoder_different_seeds_differ() {
        let enc1 = CrystalEncoder::new(8, 1);
        let enc2 = CrystalEncoder::new(8, 2);
        let emb = vec![1.0f32; 8];
        let fp1 = enc1.encode_embedding(&emb);
        let fp2 = enc2.encode_embedding(&emb);
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn encoder_nonzero_output() {
        let enc = CrystalEncoder::new(4, 42);
        let fp = enc.encode_embedding(&[1.0, -0.5, 0.3, 0.8]);
        assert!(!fp.is_zero());
    }

    #[test]
    fn encoder_opposite_embeddings_differ() {
        let enc = CrystalEncoder::new(8, 42);
        let pos = vec![1.0f32; 8];
        let neg = vec![-1.0f32; 8];
        let fp_pos = enc.encode_embedding(&pos);
        let fp_neg = enc.encode_embedding(&neg);
        // Opposite embeddings should produce very different fingerprints
        let dist = fp_pos.hamming_distance(&fp_neg);
        // They should differ in roughly half the bits (random projection property)
        assert!(dist > TOTAL_BITS as u32 / 4);
    }

    #[test]
    fn absorb_into_node_subject() {
        let enc = CrystalEncoder::new(4, 42);
        let fp = enc.encode_embedding(&[1.0, 0.0, 0.0, 0.0]);
        let mut node = Node::new();
        CrystalEncoder::absorb_into_node(&fp, &mut node, Role::Subject);
        assert_eq!(node.s.encounters(), 1);
        assert_eq!(node.p.encounters(), 0);
        assert_eq!(node.o.encounters(), 0);
    }

    #[test]
    fn absorb_into_node_predicate() {
        let enc = CrystalEncoder::new(4, 42);
        let fp = enc.encode_embedding(&[0.0, 1.0, 0.0, 0.0]);
        let mut node = Node::new();
        CrystalEncoder::absorb_into_node(&fp, &mut node, Role::Predicate);
        assert_eq!(node.s.encounters(), 0);
        assert_eq!(node.p.encounters(), 1);
        assert_eq!(node.o.encounters(), 0);
    }

    #[test]
    fn absorb_into_node_object() {
        let enc = CrystalEncoder::new(4, 42);
        let fp = enc.encode_embedding(&[0.0, 0.0, 0.0, 1.0]);
        let mut node = Node::new();
        CrystalEncoder::absorb_into_node(&fp, &mut node, Role::Object);
        assert_eq!(node.s.encounters(), 0);
        assert_eq!(node.p.encounters(), 0);
        assert_eq!(node.o.encounters(), 1);
    }

    #[test]
    fn absorb_multiple_encounters() {
        let enc = CrystalEncoder::new(4, 42);
        let fp = enc.encode_embedding(&[1.0, -0.5, 0.3, 0.8]);
        let mut node = Node::new();
        for _ in 0..5 {
            CrystalEncoder::absorb_into_node(&fp, &mut node, Role::Subject);
        }
        assert_eq!(node.s.encounters(), 5);
    }

    // -- Phase 2 tests -------------------------------------------------------

    #[test]
    fn distill_returns_correct_epoch_count() {
        let teacher = Node::random(42);
        let mut student = CrystalEncoder::new(8, 99);
        let losses = distill(&[teacher], &mut student, 5);
        assert_eq!(losses.len(), 5);
    }

    #[test]
    fn distill_loss_values_finite() {
        let teacher = Node::random(42);
        let mut student = CrystalEncoder::new(8, 99);
        let losses = distill(&[teacher], &mut student, 3);
        for loss in &losses {
            assert!(loss.is_finite(), "loss should be finite, got {}", loss);
            assert!(*loss >= 0.0, "loss should be non-negative, got {}", loss);
        }
    }

    #[test]
    fn distill_multiple_teachers() {
        let teachers: Vec<Node> = (0..3).map(|i| Node::random(i + 10)).collect();
        let mut student = CrystalEncoder::new(8, 99);
        let losses = distill(&teachers, &mut student, 2);
        assert_eq!(losses.len(), 2);
        for loss in &losses {
            assert!(loss.is_finite());
        }
    }

    #[test]
    fn distill_empty_teachers() {
        let mut student = CrystalEncoder::new(8, 99);
        let losses = distill(&[], &mut student, 3);
        assert_eq!(losses.len(), 3);
        for loss in &losses {
            assert_eq!(*loss, 0.0);
        }
    }

    #[test]
    fn spo_hamming_self_is_zero() {
        let mut a = Node::random(42);
        let mut b = a.clone();
        assert_eq!(spo_hamming(&mut a, &mut b), 0);
    }

    // -- Phase 3 tests -------------------------------------------------------

    #[test]
    fn codebook_has_65_entries() {
        let cb = NsmCodebook::new();
        assert_eq!(cb.len(), 65);
        assert!(!cb.is_empty());
    }

    #[test]
    fn codebook_entries_nonzero() {
        let cb = NsmCodebook::new();
        for i in 0..65 {
            assert!(!cb.get_prime(i).is_zero(), "prime {} should be nonzero", i);
        }
    }

    #[test]
    fn codebook_entries_distinct() {
        let cb = NsmCodebook::new();
        for i in 0..65 {
            for j in (i + 1)..65 {
                assert_ne!(
                    cb.get_prime(i),
                    cb.get_prime(j),
                    "primes {} and {} should differ",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn codebook_deterministic() {
        let cb1 = NsmCodebook::new();
        let cb2 = NsmCodebook::new();
        for i in 0..65 {
            assert_eq!(cb1.get_prime(i), cb2.get_prime(i));
        }
    }

    #[test]
    fn codebook_nearest_finds_self() {
        let cb = NsmCodebook::new();
        for i in 0..65 {
            let (idx, dist) = cb.nearest(cb.get_prime(i));
            assert_eq!(idx, i);
            assert_eq!(dist, 0);
        }
    }

    #[test]
    fn codebook_prime_names() {
        let cb = NsmCodebook::new();
        assert_eq!(cb.prime_name(0), "I");
        assert_eq!(cb.prime_name(1), "you");
        assert_eq!(cb.prime_name(64), "way");
    }

    #[test]
    fn encode_word_deterministic() {
        let cb = NsmCodebook::new();
        let fp1 = encode_word("hello", &cb);
        let fp2 = encode_word("hello", &cb);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn encode_word_different_words_differ() {
        let cb = NsmCodebook::new();
        let fp1 = encode_word("hello", &cb);
        let fp2 = encode_word("world", &cb);
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn encode_word_nonzero() {
        let cb = NsmCodebook::new();
        let fp = encode_word("test", &cb);
        assert!(!fp.is_zero());
    }

    #[test]
    fn encode_sentence_populates_all_planes() {
        let cb = NsmCodebook::new();
        let node = encode_sentence(&["the", "cat", "sat"], &cb);
        assert_eq!(node.s.encounters(), 1); // "the" -> Subject
        assert_eq!(node.p.encounters(), 1); // "cat" -> Predicate
        assert_eq!(node.o.encounters(), 1); // "sat" -> Object
    }

    #[test]
    fn encode_sentence_cyclic_distribution() {
        let cb = NsmCodebook::new();
        let node = encode_sentence(&["a", "b", "c", "d", "e", "f"], &cb);
        // a,d -> Subject (2), b,e -> Predicate (2), c,f -> Object (2)
        assert_eq!(node.s.encounters(), 2);
        assert_eq!(node.p.encounters(), 2);
        assert_eq!(node.o.encounters(), 2);
    }

    #[test]
    fn encode_sentence_single_word() {
        let cb = NsmCodebook::new();
        let node = encode_sentence(&["hello"], &cb);
        assert_eq!(node.s.encounters(), 1);
        assert_eq!(node.p.encounters(), 0);
        assert_eq!(node.o.encounters(), 0);
    }

    #[test]
    fn encode_sentence_empty() {
        let cb = NsmCodebook::new();
        let node = encode_sentence(&[], &cb);
        assert_eq!(node.s.encounters(), 0);
        assert_eq!(node.p.encounters(), 0);
        assert_eq!(node.o.encounters(), 0);
    }

    #[test]
    fn encode_sentence_deterministic() {
        let cb = NsmCodebook::new();
        let n1 = encode_sentence(&["the", "cat", "sat"], &cb);
        let n2 = encode_sentence(&["the", "cat", "sat"], &cb);
        assert_eq!(n1.s.acc(), n2.s.acc());
        assert_eq!(n1.p.acc(), n2.p.acc());
        assert_eq!(n1.o.acc(), n2.o.acc());
    }

    #[test]
    fn similar_sentences_closer_than_different() {
        let cb = NsmCodebook::new();
        let mut n1 = encode_sentence(&["I", "see", "you"], &cb);
        let mut n2 = encode_sentence(&["I", "see", "someone"], &cb);
        let mut n3 = encode_sentence(&["big", "bad", "body"], &cb);

        let d12 = n1.distance(&mut n2, SPO);
        let d13 = n1.distance(&mut n3, SPO);

        // n1 and n2 share "I" and "see", so should be closer than n1 and n3
        match (d12, d13) {
            (
                super::super::plane::Distance::Measured { disagreement: d1, .. },
                super::super::plane::Distance::Measured { disagreement: d2, .. },
            ) => {
                assert!(d1 < d2, "similar sentences should be closer: {} vs {}", d1, d2);
            }
            _ => panic!("expected Measured distances"),
        }
    }

    // -- Integration test: full pipeline ------------------------------------

    // -- Pipeline wiring tests -----------------------------------------------

    #[test]
    fn encode_and_absorb_returns_fingerprint() {
        let enc = CrystalEncoder::new(4, 42);
        let mut node = Node::new();
        let fp = enc.encode_and_absorb(&[1.0, -0.5, 0.3, 0.8], &mut node, Role::Subject);
        assert!(!fp.is_zero());
        assert_eq!(node.s.encounters(), 1);
        assert_eq!(node.p.encounters(), 0);
    }

    #[test]
    fn search_similar_finds_exact_match() {
        let enc = CrystalEncoder::new(4, 42);
        let emb = [1.0f32, -0.5, 0.3, 0.8];

        let mut query = Node::new();
        enc.encode_and_absorb(&emb, &mut query, Role::Subject);
        enc.encode_and_absorb(&emb, &mut query, Role::Predicate);
        enc.encode_and_absorb(&emb, &mut query, Role::Object);

        // Clone query into database
        let mut db = vec![Node::random(1), query.clone(), Node::random(2)];
        let results = CrystalEncoder::search_similar(&mut query, &mut db, 5);
        assert!(!results.is_empty());
        // Best match should be the clone at index 1 with distance 0
        assert_eq!(results[0].0, 1);
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn search_similar_respects_top_k() {
        let mut query = Node::random(42);
        let mut db: Vec<Node> = (0..10).map(|i| Node::random(i + 100)).collect();
        let results = CrystalEncoder::search_similar(&mut query, &mut db, 3);
        assert!(results.len() <= 3);
    }

    #[test]
    fn pipeline_encode_search_basic() {
        let enc = CrystalEncoder::new(4, 42);
        let mut db = vec![Node::random(1), Node::random(2), Node::random(3)];
        let (query, results) = pipeline_encode_search(
            &enc,
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
            &mut db,
            5,
        );
        assert_eq!(query.s.encounters(), 1);
        assert_eq!(query.p.encounters(), 1);
        assert_eq!(query.o.encounters(), 1);
        assert!(!results.is_empty());
        // Results should be sorted by distance
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn pipeline_encode_search_empty_db() {
        let enc = CrystalEncoder::new(4, 42);
        let mut db: Vec<Node> = vec![];
        let (query, results) = pipeline_encode_search(
            &enc,
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
            &mut db,
            5,
        );
        assert_eq!(query.s.encounters(), 1);
        assert!(results.is_empty());
    }

    // -- Phase 1 full integration test (original) --------------------------

    #[test]
    fn full_pipeline_encode_absorb_measure() {
        let enc = CrystalEncoder::new(16, 42);

        // Encode two similar embeddings
        let emb1 = vec![1.0f32; 16];
        let mut emb2 = vec![1.0f32; 16];
        emb2[0] = -1.0;

        let fp1 = enc.encode_embedding(&emb1);
        let fp2 = enc.encode_embedding(&emb2);

        // Absorb into nodes
        let mut node1 = Node::new();
        let mut node2 = Node::new();
        for _ in 0..3 {
            CrystalEncoder::absorb_into_node(&fp1, &mut node1, Role::Subject);
            CrystalEncoder::absorb_into_node(&fp1, &mut node1, Role::Predicate);
            CrystalEncoder::absorb_into_node(&fp1, &mut node1, Role::Object);
            CrystalEncoder::absorb_into_node(&fp2, &mut node2, Role::Subject);
            CrystalEncoder::absorb_into_node(&fp2, &mut node2, Role::Predicate);
            CrystalEncoder::absorb_into_node(&fp2, &mut node2, Role::Object);
        }

        // Measure distance
        let d = node1.distance(&mut node2, SPO);
        match d {
            super::super::plane::Distance::Measured { overlap, .. } => {
                assert!(overlap > 0, "similar embeddings should have overlap");
            }
            super::super::plane::Distance::Incomparable => {
                panic!("nodes with encounters should be comparable");
            }
        }
    }
}
