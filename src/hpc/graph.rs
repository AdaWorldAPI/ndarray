//! Graph operations: VerbCodebook for HDC-based knowledge graphs.
//!
//! Encodes directed edges as XOR bindings: edge = src ⊕ verb ⊕ tgt.
//! Supports causality checking and verb inference.

use crate::imp_prelude::*;
use super::hdc::HdcOps;
use super::bitwise::BitwiseOps;

/// A VerbCodebook maps verb names to binary hypervectors.
///
/// Each verb is represented by rotating a base vector by an offset.
///
/// # Example
///
/// ```
/// use ndarray::hpc::graph::VerbCodebook;
///
/// let cb = VerbCodebook::default_codebook();
/// assert!(cb.offset("causes").is_some());
/// ```
pub struct VerbCodebook {
    verbs: Vec<(String, usize)>,
    base_dim: usize,
}

impl VerbCodebook {
    /// Create the default codebook with common verbs.
    pub fn default_codebook() -> Self {
        Self::new(vec![
            ("causes", 1),
            ("enables", 2),
            ("prevents", 3),
            ("requires", 4),
            ("implies", 5),
            ("contains", 6),
            ("part_of", 7),
            ("is_a", 8),
            ("has", 9),
            ("relates_to", 10),
        ])
    }

    /// Create a new codebook from (verb, offset) pairs.
    pub fn new(verbs: Vec<(&str, usize)>) -> Self {
        Self {
            verbs: verbs.into_iter().map(|(v, o)| (v.to_string(), o)).collect(),
            base_dim: 4096,
        }
    }

    /// Get the offset for a verb.
    pub fn offset(&self, verb: &str) -> Option<usize> {
        self.verbs.iter().find(|(v, _)| v == verb).map(|(_, o)| *o)
    }

    /// List all verbs with their offsets.
    pub fn verbs(&self) -> Vec<(&str, usize)> {
        self.verbs.iter().map(|(v, o)| (v.as_str(), *o)).collect()
    }

    /// Generate a verb vector by rotating a fixed base pattern by the verb's offset.
    fn verb_vector(&self, verb: &str) -> Option<Array<u8, Ix1>> {
        let offset = self.offset(verb)?;
        // Create a deterministic pattern based on verb offset
        let mut v = Array::zeros(self.base_dim);
        let mut state = (offset as u64).wrapping_mul(2654435761);
        for byte in v.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *byte = (state >> 33) as u8;
        }
        Some(v)
    }

    /// Encode an edge: edge = permute(src) ⊕ verb_vec ⊕ tgt
    ///
    /// Uses permutation on src to break commutativity (causality).
    ///
    /// # Errors
    /// Returns `Err` if the verb is not in the codebook.
    pub fn try_encode_edge(
        &self,
        src: &Array<u8, Ix1>,
        verb: &str,
        tgt: &Array<u8, Ix1>,
    ) -> Result<Array<u8, Ix1>, &'static str> {
        let verb_vec = self.verb_vector(verb).ok_or("Unknown verb")?;
        let offset = self.offset(verb).unwrap_or(1);
        let src_permuted = src.hdc_permute(offset);
        Ok(src_permuted.hdc_bind(&verb_vec).hdc_bind(tgt))
    }

    /// Encode an edge (panics on unknown verb).
    pub fn encode_edge(
        &self,
        src: &Array<u8, Ix1>,
        verb: &str,
        tgt: &Array<u8, Ix1>,
    ) -> Array<u8, Ix1> {
        self.try_encode_edge(src, verb, tgt).unwrap()
    }

    /// Decode target: tgt = edge ⊕ permute(src) ⊕ verb_vec
    pub fn try_decode_target(
        &self,
        edge: &Array<u8, Ix1>,
        src: &Array<u8, Ix1>,
        verb: &str,
    ) -> Result<Array<u8, Ix1>, &'static str> {
        let verb_vec = self.verb_vector(verb).ok_or("Unknown verb")?;
        let offset = self.offset(verb).unwrap_or(1);
        let src_permuted = src.hdc_permute(offset);
        Ok(edge.hdc_bind(&src_permuted).hdc_bind(&verb_vec))
    }

    /// Decode target (panics on unknown verb).
    pub fn decode_target(
        &self,
        edge: &Array<u8, Ix1>,
        src: &Array<u8, Ix1>,
        verb: &str,
    ) -> Array<u8, Ix1> {
        self.try_decode_target(edge, src, verb).unwrap()
    }

    /// Causality asymmetry: measures how well edge(src→tgt) differs from edge(tgt→src).
    ///
    /// Returns a value between 0 (symmetric) and 1 (fully asymmetric).
    pub fn causality_asymmetry(
        &self,
        src: &Array<u8, Ix1>,
        verb: &str,
        tgt: &Array<u8, Ix1>,
    ) -> f64 {
        let forward = self.encode_edge(src, verb, tgt);
        let backward = self.encode_edge(tgt, verb, src);
        let dist = forward.hamming_distance(&backward);
        let max_bits = (forward.len() * 8) as f64;
        dist as f64 / max_bits
    }

    /// Full causality check: returns (forward_edge, backward_edge, asymmetry).
    pub fn causality_check(
        &self,
        src: &Array<u8, Ix1>,
        verb: &str,
        tgt: &Array<u8, Ix1>,
    ) -> (Array<u8, Ix1>, Array<u8, Ix1>, f64) {
        let forward = self.encode_edge(src, verb, tgt);
        let backward = self.encode_edge(tgt, verb, src);
        let asymmetry = {
            let dist = forward.hamming_distance(&backward);
            let max_bits = (forward.len() * 8) as f64;
            dist as f64 / max_bits
        };
        (forward, backward, asymmetry)
    }

    /// Find edges with low causality asymmetry (potentially non-causal).
    pub fn find_non_causal_edges(
        &self,
        edges: &[(Array<u8, Ix1>, &str, Array<u8, Ix1>)],
        threshold: f64,
    ) -> Vec<(usize, f64)> {
        edges
            .iter()
            .enumerate()
            .filter_map(|(i, (src, verb, tgt))| {
                let asym = self.causality_asymmetry(src, verb, tgt);
                if asym < threshold {
                    Some((i, asym))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Infer verb: given an edge and source, find which verb best explains it.
    ///
    /// Returns (verb_name, verb_offset, hamming_distance).
    pub fn infer_verb(
        &self,
        edge: &Array<u8, Ix1>,
        src: &Array<u8, Ix1>,
        candidates: &[Array<u8, Ix1>],
    ) -> Option<(String, usize, u64)> {
        if candidates.is_empty() {
            return None;
        }

        let mut best: Option<(String, usize, u64)> = None;

        for (verb_name, &offset) in self.verbs.iter().map(|(v, o)| (v, o)) {
            if let Ok(decoded_tgt) = self.try_decode_target(edge, src, verb_name) {
                for tgt in candidates {
                    let dist = decoded_tgt.hamming_distance(tgt);
                    if best.is_none() || dist < best.as_ref().unwrap().2 {
                        best = Some((verb_name.clone(), offset, dist));
                    }
                }
            }
        }
        best
    }
}

/// Encode an edge using explicit verb vector (no codebook needed).
pub fn encode_edge_explicit(
    src: &Array<u8, Ix1>,
    verb_vec: &Array<u8, Ix1>,
    tgt: &Array<u8, Ix1>,
) -> Array<u8, Ix1> {
    src.hdc_bind(verb_vec).hdc_bind(tgt)
}

/// Decode target using explicit verb vector.
pub fn decode_target_explicit(
    edge: &Array<u8, Ix1>,
    src: &Array<u8, Ix1>,
    verb_vec: &Array<u8, Ix1>,
) -> Array<u8, Ix1> {
    edge.hdc_bind(src).hdc_bind(verb_vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vec(seed: u64, len: usize) -> Array<u8, Ix1> {
        let mut v = Array::zeros(len);
        let mut state = seed;
        for byte in v.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *byte = (state >> 33) as u8;
        }
        v
    }

    #[test]
    fn test_default_codebook() {
        let cb = VerbCodebook::default_codebook();
        assert!(cb.offset("causes").is_some());
        assert!(cb.offset("enables").is_some());
        assert!(cb.offset("nonexistent").is_none());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let cb = VerbCodebook::default_codebook();
        let src = random_vec(42, 4096);
        let tgt = random_vec(99, 4096);

        let edge = cb.encode_edge(&src, "causes", &tgt);
        let decoded = cb.decode_target(&edge, &src, "causes");

        // XOR is its own inverse, so decoded should exactly equal tgt
        assert_eq!(decoded, tgt);
    }

    #[test]
    fn test_causality_asymmetry() {
        let cb = VerbCodebook::default_codebook();
        let src = random_vec(12345, 4096);
        let tgt = random_vec(67890, 4096);

        let asym = cb.causality_asymmetry(&src, "causes", &tgt);
        // Forward and backward edges should differ (asymmetry > 0)
        assert!(asym > 0.0, "Asymmetry should be non-zero: got {}", asym);
    }

    #[test]
    fn test_explicit_roundtrip() {
        let src = random_vec(10, 100);
        let tgt = random_vec(20, 100);
        let verb = random_vec(30, 100);

        let edge = encode_edge_explicit(&src, &verb, &tgt);
        let decoded = decode_target_explicit(&edge, &src, &verb);
        assert_eq!(decoded, tgt);
    }
}
