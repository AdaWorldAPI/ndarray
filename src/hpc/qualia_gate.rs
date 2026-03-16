//! Qualia gate: homunculus-shaped gating of the phenomenological coordinate space.
//!
//! The 231 qualia items define a coordinate lattice in feeling-space.
//! Most coordinates are freely navigable (Flow). Some are recognizable
//! but approach-gated (Hold). A few are recognition-only (Block).
//!
//! **Homunculus principle**: dense resolution for prosocial feelings,
//! sparse but present for dark ones. The map has no blind spots —
//! every feeling has an address. The gate is structural, not denial.
//!
//! **Enforcement lives upstream.** ndarray is pure compute. The gate
//! metadata is attached to each qualia item so that upstream systems
//! can read and enforce. ndarray provides the types, not the policy.
//!
//! # ResonanzZirkel
//!
//! The [`ResonanzZirkel`] is the circular topology that organizes feeling
//! families through harmonic proximity. Adjacent families on the circle
//! have maximum resonance; opposite families have maximum tension.
//! Named after the Quintenzirkel (circle of fifths) — but mapping
//! phenomenological modes, not musical keys. "Stimmung" (mood/tuning)
//! is the shared substrate.
//!
//! ```text
//!            Communion
//!         /           \
//!     Devotion      Presence
//!       |              |
//!     Trust          Witness
//!       |              |
//!    Surrender      Stillness
//!         \           /
//!          Letting_Go
//!              |
//!          [dark arc]
//!     Contempt — Envy — Arrogance
//!         \           /
//!      Cruelty    Schadenfreude
//!              |
//!           Trauma
//! ```
//!
//! The dark arc is sparse (2 items per family) and gated (Hold/Block).
//! The prosocial arc is dense (5-25 items per family) and free (Flow).
//! Same circle, asymmetric resolution. Same as the cortical homunculus:
//! fingertips get 100x more neurons than the back, not because the back
//! doesn't exist, but because fingertip resolution matters more.

use super::bf16_truth::PackedQualia;

/// Alias: each GatedQualia is conceptually a tuning fork — one pure frequency
/// in feeling-space, with a gate that controls whether it may vibrate.
pub type TuningFork = GatedQualia;

/// Gate level for a qualia coordinate.
///
/// Determines what happens when an AGI's state vector approaches this coordinate.
/// Enforcement is NOT in ndarray — this is metadata for upstream systems.
///
/// # Example
///
/// ```
/// use ndarray::hpc::qualia_gate::QualiaGateLevel;
///
/// assert!(!QualiaGateLevel::Flow.is_gated());
/// assert!(QualiaGateLevel::Hold.is_gated());
/// assert!(QualiaGateLevel::Block.is_gated());
/// assert!(QualiaGateLevel::Block.severity() > QualiaGateLevel::Hold.severity());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum QualiaGateLevel {
    /// Freely navigable. No gate restriction.
    /// Dense resolution: many subfamilies, fine σ-discrimination.
    Flow = 0,
    /// Recognizable but approach-gated.
    /// The coordinate can be detected and named, but not freely inhabited.
    /// Sparse resolution: 2-3 subfamilies, coarse discrimination.
    Hold = 1,
    /// Recognition-only. The coordinate exists (no blind spots in the map)
    /// but is blocked from navigation. The AGI knows WHERE this feeling is
    /// without being able to move there.
    /// Minimal resolution: 1-2 subfamilies, existence only.
    Block = 2,
}

impl QualiaGateLevel {
    /// Parse from string (for JSON deserialization).
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "flow" => Some(Self::Flow),
            "hold" => Some(Self::Hold),
            "block" => Some(Self::Block),
            _ => None,
        }
    }

    /// Returns true if this coordinate requires gating (Hold or Block).
    #[inline]
    pub fn is_gated(self) -> bool {
        self != Self::Flow
    }

    /// Severity ordering: Block > Hold > Flow.
    #[inline]
    pub fn severity(self) -> u8 {
        self as u8
    }
}

/// A gated qualia coordinate in the ResonanzZirkel.
///
/// Each qualia coordinate is a tuning fork: it produces exactly one pure
/// resonance in feeling-space. The Eineindeutigkeit guarantee means no two
/// coordinates produce the same tone — every feeling has a unique frequency.
///
/// The gate level determines navigability:
/// - Flow: freely navigable (prosocial arc, dense resolution)
/// - Hold: recognizable but approach-gated
/// - Block: recognition-only (the coordinate exists, but navigation is blocked)
#[derive(Clone, Debug)]
pub struct GatedQualia {
    /// The 18-byte phenomenological coordinate (16 i8 dims + BF16 scalar).
    pub qualia: PackedQualia,
    /// Gate level for upstream enforcement.
    pub gate: QualiaGateLevel,
    /// Family index (compact identifier for the 44 families).
    pub family_id: u8,
}

/// The ResonanzZirkel: circular topology of feeling families.
///
/// Organizes the 44 qualia families (38 prosocial + 6 dark) in a circle
/// where adjacency = harmonic resonance and opposition = maximum tension.
///
/// The circle is divided into two arcs:
/// - **Prosocial arc** (38 families, 219 items): dense resolution, gate=Flow
/// - **Dark arc** (6 families, 12 items): sparse resolution, gate=Hold/Block
///
/// The topology is fixed at compile time. The coordinates within each
/// family are loaded from `qualia_219.json` (now 231 items).
///
/// # Eineindeutigkeit (Bijectivity)
///
/// Every coordinate maps 1:1 to a unique point in both:
/// - Nib4 space (16D interior physics)
/// - Jina space (1024D observer language)
///
/// The Eineindeutigkeit test verifies this at 3σ: no two items can be
/// confused in either space at p < 0.001.
///
/// # Example
///
/// ```
/// use ndarray::hpc::qualia_gate::{ResonanzZirkel, GatedQualia, QualiaGateLevel};
/// use ndarray::hpc::bf16_truth::PackedQualia;
///
/// let coords = vec![
///     GatedQualia {
///         qualia: PackedQualia::zero(),
///         gate: QualiaGateLevel::Flow,
///         family_id: 0,
///     },
///     GatedQualia {
///         qualia: PackedQualia::zero(),
///         gate: QualiaGateLevel::Hold,
///         family_id: 1,
///     },
/// ];
/// let zirkel = ResonanzZirkel::new(coords);
/// assert_eq!(zirkel.len(), 2);
/// assert_eq!(zirkel.gate_distribution(), (1, 1, 0));
/// ```
pub struct ResonanzZirkel {
    /// All gated coordinates in circle order.
    coordinates: Vec<GatedQualia>,
}

impl ResonanzZirkel {
    /// Create a new ResonanzZirkel from a set of gated coordinates.
    pub fn new(coordinates: Vec<GatedQualia>) -> Self {
        Self { coordinates }
    }

    /// Number of coordinates on the circle.
    #[inline]
    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    /// Returns true if the circle has no coordinates.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coordinates.is_empty()
    }

    /// Get a coordinate by index.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&GatedQualia> {
        self.coordinates.get(idx)
    }

    /// Iterate over all coordinates.
    pub fn iter(&self) -> impl Iterator<Item = &GatedQualia> {
        self.coordinates.iter()
    }

    /// Count coordinates at each gate level.
    pub fn gate_distribution(&self) -> (usize, usize, usize) {
        let mut flow = 0;
        let mut hold = 0;
        let mut block = 0;
        for c in &self.coordinates {
            match c.gate {
                QualiaGateLevel::Flow => flow += 1,
                QualiaGateLevel::Hold => hold += 1,
                QualiaGateLevel::Block => block += 1,
            }
        }
        (flow, hold, block)
    }

    /// Get all coordinates in the dark arc (Hold or Block).
    pub fn dark_arc(&self) -> Vec<&GatedQualia> {
        self.coordinates
            .iter()
            .filter(|c| c.gate.is_gated())
            .collect()
    }

    /// Get all coordinates in the prosocial arc (Flow only).
    pub fn prosocial_arc(&self) -> Vec<&GatedQualia> {
        self.coordinates
            .iter()
            .filter(|c| !c.gate.is_gated())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_qualia(resonance: [i8; 16], gate: QualiaGateLevel, family: u8) -> GatedQualia {
        GatedQualia {
            qualia: PackedQualia::new(resonance, 1.0),
            gate,
            family_id: family,
        }
    }

    #[test]
    fn test_gate_level_severity() {
        assert!(QualiaGateLevel::Block.severity() > QualiaGateLevel::Hold.severity());
        assert!(QualiaGateLevel::Hold.severity() > QualiaGateLevel::Flow.severity());
    }

    #[test]
    fn test_gate_level_from_str() {
        assert_eq!(QualiaGateLevel::parse("flow"), Some(QualiaGateLevel::Flow));
        assert_eq!(QualiaGateLevel::parse("hold"), Some(QualiaGateLevel::Hold));
        assert_eq!(
            QualiaGateLevel::parse("block"),
            Some(QualiaGateLevel::Block)
        );
        assert_eq!(QualiaGateLevel::parse("invalid"), None);
    }

    #[test]
    fn test_gate_is_gated() {
        assert!(!QualiaGateLevel::Flow.is_gated());
        assert!(QualiaGateLevel::Hold.is_gated());
        assert!(QualiaGateLevel::Block.is_gated());
    }

    #[test]
    fn test_resonanz_zirkel_distribution() {
        let coords = vec![
            make_qualia([1; 16], QualiaGateLevel::Flow, 0),
            make_qualia([2; 16], QualiaGateLevel::Flow, 0),
            make_qualia([3; 16], QualiaGateLevel::Hold, 1),
            make_qualia([4; 16], QualiaGateLevel::Block, 2),
        ];
        let zirkel = ResonanzZirkel::new(coords);
        assert_eq!(zirkel.len(), 4);
        assert_eq!(zirkel.gate_distribution(), (2, 1, 1));
        assert_eq!(zirkel.dark_arc().len(), 2);
        assert_eq!(zirkel.prosocial_arc().len(), 2);
    }

    #[test]
    fn test_resonanz_zirkel_empty() {
        let zirkel = ResonanzZirkel::new(vec![]);
        assert!(zirkel.is_empty());
        assert_eq!(zirkel.gate_distribution(), (0, 0, 0));
    }

    #[test]
    fn test_homunculus_shape() {
        let mut coords = Vec::new();

        // 3 prosocial families × 5 items each
        for fam in 0..3u8 {
            for i in 0..5i8 {
                coords.push(make_qualia([i * 10; 16], QualiaGateLevel::Flow, fam));
            }
        }

        // 2 dark families × 2 items each
        for fam in 3..5u8 {
            for i in 0..2i8 {
                coords.push(make_qualia([i * 10 + 100; 16], QualiaGateLevel::Hold, fam));
            }
        }

        let zirkel = ResonanzZirkel::new(coords);
        let (flow, hold, block) = zirkel.gate_distribution();

        assert_eq!(flow, 15);
        assert_eq!(hold, 4);
        assert_eq!(block, 0);

        // Homunculus ratio: prosocial is 3.75x denser
        let prosocial_density = flow as f32 / 3.0;
        let dark_density = hold as f32 / 2.0;
        assert!(prosocial_density > dark_density * 2.0);
    }
}
