//! Node: the cognitive atom. Three Planes (S/P/O), separately addressable via Mask.

use super::fingerprint::Fingerprint;
use super::plane::{Distance, Plane, Truth};

/// Attention mask over S/P/O planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mask {
    pub s: bool,
    pub p: bool,
    pub o: bool,
}

pub const SPO: Mask = Mask { s: true, p: true, o: true };
pub const SP_: Mask = Mask { s: true, p: true, o: false };
pub const S_O: Mask = Mask { s: true, p: false, o: true };
pub const _PO: Mask = Mask { s: false, p: true, o: true };
pub const S__: Mask = Mask { s: true, p: false, o: false };
pub const _P_: Mask = Mask { s: false, p: true, o: false };
pub const __O: Mask = Mask { s: false, p: false, o: true };
pub const ___: Mask = Mask { s: false, p: false, o: false };

impl Mask {
    #[inline]
    pub fn count(&self) -> u32 {
        self.s as u32 + self.p as u32 + self.o as u32
    }
}

/// The cognitive atom. Three planes, separately addressable.
pub struct Node {
    pub s: Plane,
    pub p: Plane,
    pub o: Plane,
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            s: self.s.clone(),
            p: self.p.clone(),
            o: self.o.clone(),
        }
    }
}

/// Simple SplitMix64 RNG for deterministic random node generation.
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
}

impl Node {
    pub fn new() -> Self {
        Self {
            s: Plane::new(),
            p: Plane::new(),
            o: Plane::new(),
        }
    }

    /// Random node for testing. Seed → deterministic.
    pub fn random(seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut node = Self::new();
        for plane in [&mut node.s, &mut node.p, &mut node.o] {
            let mut words = [0u64; 256];
            for w in words.iter_mut() {
                *w = rng.next_u64();
            }
            let fp = Fingerprint::<256>::from_words(words);
            plane.encounter_bits(&fp);
            plane.encounter_bits(&fp);
            plane.encounter_bits(&fp);
        }
        node
    }

    pub fn distance(&mut self, other: &mut Node, mask: Mask) -> Distance {
        let mut total_disagreement = 0u32;
        let mut total_overlap = 0u32;
        let mut total_penalty = 0u32;
        let mut any_measured = false;

        macro_rules! add_plane {
            ($self_plane:expr, $other_plane:expr, $active:expr) => {
                if $active {
                    match $self_plane.distance(&mut $other_plane) {
                        Distance::Measured { disagreement, overlap, penalty } => {
                            total_disagreement += disagreement;
                            total_overlap += overlap;
                            total_penalty += penalty;
                            any_measured = true;
                        }
                        Distance::Incomparable => {}
                    }
                }
            };
        }

        add_plane!(self.s, other.s, mask.s);
        add_plane!(self.p, other.p, mask.p);
        add_plane!(self.o, other.o, mask.o);

        if !any_measured || total_overlap == 0 {
            Distance::Incomparable
        } else {
            Distance::Measured {
                disagreement: total_disagreement,
                overlap: total_overlap,
                penalty: total_penalty,
            }
        }
    }

    /// Project all 7 non-empty mask combinations and return distances.
    ///
    /// Returns distances in order: `[S, P, O, SP, SO, PO, SPO]`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::node::Node;
    ///
    /// let mut a = Node::random(42);
    /// let mut b = Node::random(43);
    /// let projections = a.project_all(&mut b);
    /// assert_eq!(projections.len(), 7);
    /// ```
    pub fn project_all(&mut self, other: &mut Node) -> [Distance; 7] {
        [
            self.distance(other, S__),
            self.distance(other, _P_),
            self.distance(other, __O),
            self.distance(other, SP_),
            self.distance(other, S_O),
            self.distance(other, _PO),
            self.distance(other, SPO),
        ]
    }

    pub fn truth(&mut self, mask: Mask) -> Truth {
        let mut total_freq = 0u64;
        let mut total_conf = 0u64;
        let mut total_evidence = 0u32;
        let mut count = 0u32;

        macro_rules! add_truth {
            ($plane:expr, $active:expr) => {
                if $active {
                    let t = $plane.truth();
                    total_freq += t.frequency as u64;
                    total_conf += t.confidence as u64;
                    total_evidence += t.evidence;
                    count += 1;
                }
            };
        }

        add_truth!(self.s, mask.s);
        add_truth!(self.p, mask.p);
        add_truth!(self.o, mask.o);

        if count == 0 {
            return Truth {
                frequency: 32768,
                confidence: 0,
                evidence: 0,
            };
        }

        Truth {
            frequency: (total_freq / count as u64) as u16,
            confidence: (total_conf / count as u64) as u16,
            evidence: total_evidence,
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_new_is_empty() {
        let n = Node::new();
        assert_eq!(n.s.encounters(), 0);
        assert_eq!(n.p.encounters(), 0);
        assert_eq!(n.o.encounters(), 0);
    }

    #[test]
    fn mask_count() {
        assert_eq!(SPO.count(), 3);
        assert_eq!(SP_.count(), 2);
        assert_eq!(S__.count(), 1);
        assert_eq!(___.count(), 0);
    }

    #[test]
    fn mask_skips_planes() {
        let mut a = Node::random(42);
        let mut b = Node::random(43);
        let d_spo = a.distance(&mut b, SPO);
        let d_s = a.distance(&mut b, S__);
        match (d_spo, d_s) {
            (
                Distance::Measured { overlap: o_spo, .. },
                Distance::Measured { overlap: o_s, .. },
            ) => {
                assert!(o_spo >= o_s);
            }
            _ => panic!("expected Measured for random nodes"),
        }
    }

    #[test]
    fn empty_mask_is_incomparable() {
        let mut a = Node::random(42);
        let mut b = Node::random(43);
        let d = a.distance(&mut b, ___);
        assert!(matches!(d, Distance::Incomparable));
    }

    #[test]
    fn node_truth_empty_mask() {
        let mut n = Node::new();
        let t = n.truth(___);
        assert_eq!(t.frequency, 32768);
        assert_eq!(t.confidence, 0);
        assert_eq!(t.evidence, 0);
    }

    #[test]
    fn project_all_returns_seven_distances() {
        let mut a = Node::random(42);
        let mut b = Node::random(43);
        let projections = a.project_all(&mut b);
        assert_eq!(projections.len(), 7);
        // All should be Measured for random nodes with encounters
        for (i, d) in projections.iter().enumerate() {
            match d {
                Distance::Measured { overlap, .. } => {
                    assert!(*overlap > 0, "projection {} should have overlap", i);
                }
                Distance::Incomparable => panic!("projection {} should be Measured", i),
            }
        }
    }

    #[test]
    fn project_all_spo_matches_direct() {
        let mut a = Node::random(42);
        let mut b = Node::random(43);
        let projections = a.project_all(&mut b);
        let direct_spo = a.distance(&mut b, SPO);
        // SPO is the last element (index 6)
        match (projections[6], direct_spo) {
            (
                Distance::Measured { disagreement: d1, overlap: o1, penalty: p1 },
                Distance::Measured { disagreement: d2, overlap: o2, penalty: p2 },
            ) => {
                assert_eq!(d1, d2);
                assert_eq!(o1, o2);
                assert_eq!(p1, p2);
            }
            _ => panic!("expected both Measured"),
        }
    }

    #[test]
    fn project_all_single_planes_match_direct() {
        let mut a = Node::random(100);
        let mut b = Node::random(200);
        let projections = a.project_all(&mut b);

        // S__ is index 0
        let d_s = a.distance(&mut b, S__);
        match (projections[0], d_s) {
            (
                Distance::Measured { disagreement: d1, .. },
                Distance::Measured { disagreement: d2, .. },
            ) => assert_eq!(d1, d2),
            _ => panic!("expected Measured"),
        }
    }

    #[test]
    fn node_random_deterministic() {
        let a = Node::random(99);
        let b = Node::random(99);
        assert_eq!(a.s.acc(), b.s.acc());
        assert_eq!(a.p.acc(), b.p.acc());
        assert_eq!(a.o.acc(), b.o.acc());
    }
}
