#![allow(clippy::assign_op_pattern, clippy::too_many_arguments, clippy::manual_range_contains, clippy::needless_range_loop)]

//! Cross-Plane Partial Binding Algebra for 3D SPO Inference.
//!
//! When a cascade filters candidates at the sigma-2/sigma-3 boundary, a
//! cross-plane orthogonal projection vote reveals each survivor's **plane
//! membership signature**: which of the 3 planes (S, P, O) independently
//! consider it a survivor. This produces exactly 6 non-trivial halo types
//! (excluding noise=0-of-3 and core=3-of-3):
//!
//! ```text
//! TYPE  PLANES  MEANING                      BINDING STATUS
//! S     S only  Entity as potential subject   Unbound (free variable)
//! P     P only  Action/relation detected      Unbound (free variable)
//! O     O only  Entity as potential object    Unbound (free variable)
//! SP    S + P   Subject performing action     Partial binding (missing O)
//! SO    S + O   Two entities related          Partial binding (missing P)
//! PO    P + O   Action applied to target      Partial binding (missing S)
//! ```
//!
//! These 6 types form the face lattice of a 2-simplex (triangle):
//!
//! ```text
//!          SPO (core)
//!         / | \
//!       SP  SO  PO
//!       |\ /\ /|
//!       S  P  O
//!         \|/
//!        noise
//! ```
//!
//! The lattice drives:
//! - **Typed queries**: SP-type -> "who does what, but to whom?"
//! - **DN tree growth**: incremental composition from fragments
//! - **Resonator warm-start**: partial evidence pre-fills known slots
//! - **Inference**: forward, backward, abductive, and analogical reasoning
//!
//! ## References
//!
//! - Kanerva (1988): Sparse Distributed Memory -- multi-field partial matching
//! - Plate (1995): HRR -- circular convolution binding/unbinding noise
//! - Smolensky (1990): Tensor Product Representations -- role-filler decomposition
//! - Kleyko et al. (2021): VSA survey (arXiv:2111.06077) -- resonator networks
//! - Fillmore (1968): Case grammar -- deep semantic roles
//! - Czegel et al. (2021): Darwinian neurodynamics -- error thresholds

use super::fingerprint::Fingerprint;
use super::kernels::{score_sigma, EnergyConflict, SigmaGate, SigmaScore, SignificanceLevel};

// ============================================================================
// CollapseGate (locally defined, matches rustynum-core layer_stack)
// ============================================================================

/// Gate decision for superposition collapse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CollapseGate {
    /// Gate opens. Superposition collapses into ground truth.
    Flow,
    /// Gate stays closed. Superposition persists across cycles.
    Hold,
    /// Contradiction. Superposition discarded. Ground truth unchanged.
    Block,
}

// ============================================================================
// SplitMix64 RNG (locally defined, matches rustynum-core rng)
// ============================================================================

/// SplitMix64 PRNG for deterministic random fingerprint generation.
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// Create a new SplitMix64 with the given seed.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Generate the next `u64` value.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

// ============================================================================
// Halo Types -- the 8 plane membership signatures
// ============================================================================

/// Plane membership signature from cross-plane orthogonal projection vote.
///
/// The 3 planes (S, P, O) each independently classify a candidate as
/// survivor or non-survivor. The 2^3 = 8 combinations form the power set
/// lattice, which is isomorphic to the face lattice of a 2-simplex.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HaloType {
    /// 0-of-3: no plane membership. Discard.
    Noise = 0,
    /// S only: entity exists as potential subject, no relational context.
    S = 1,
    /// P only: action/relation detected, no actor or patient.
    P = 2,
    /// O only: entity exists as potential object/patient.
    O = 3,
    /// S + P: subject performing action, but target unknown.
    SP = 4,
    /// S + O: two entities related, but relationship undefined.
    SO = 5,
    /// P + O: action applied to target, but agent unknown.
    PO = 6,
    /// 3-of-3: full SPO triple. Promote to core.
    Core = 7,
}

impl HaloType {
    /// Number of agreeing planes (0-3).
    #[inline]
    pub fn plane_count(self) -> u8 {
        match self {
            HaloType::Noise => 0,
            HaloType::S | HaloType::P | HaloType::O => 1,
            HaloType::SP | HaloType::SO | HaloType::PO => 2,
            HaloType::Core => 3,
        }
    }

    /// Which planes are active: (s, p, o).
    #[inline]
    pub fn planes(self) -> (bool, bool, bool) {
        match self {
            HaloType::Noise => (false, false, false),
            HaloType::S => (true, false, false),
            HaloType::P => (false, true, false),
            HaloType::O => (false, false, true),
            HaloType::SP => (true, true, false),
            HaloType::SO => (true, false, true),
            HaloType::PO => (false, true, true),
            HaloType::Core => (true, true, true),
        }
    }

    /// Lattice level: 0=noise, 1=free variable, 2=partial pair, 3=core.
    #[inline]
    pub fn lattice_level(self) -> u8 {
        self.plane_count()
    }

    /// Whether this type has an open slot that can be filled.
    #[inline]
    pub fn has_open_slot(self) -> bool {
        self.plane_count() < 3 && self.plane_count() > 0
    }

    /// Classify from raw plane membership bits.
    #[inline]
    pub fn from_membership(s: bool, p: bool, o: bool) -> Self {
        match (s, p, o) {
            (false, false, false) => HaloType::Noise,
            (true, false, false) => HaloType::S,
            (false, true, false) => HaloType::P,
            (false, false, true) => HaloType::O,
            (true, true, false) => HaloType::SP,
            (true, false, true) => HaloType::SO,
            (false, true, true) => HaloType::PO,
            (true, true, true) => HaloType::Core,
        }
    }

    /// The inference mode this halo type enables (for partial types only).
    pub fn inference_mode(self) -> Option<InferenceMode> {
        match self {
            HaloType::SP => Some(InferenceMode::Forward),
            HaloType::PO => Some(InferenceMode::Backward),
            HaloType::SO => Some(InferenceMode::Abduction),
            HaloType::S | HaloType::P | HaloType::O => Some(InferenceMode::Analogy),
            _ => None,
        }
    }
}

// ============================================================================
// Cross-Plane Vote -- bitwise extraction of 7 disjoint halo masks
// ============================================================================

/// Result of cross-plane voting on 3 per-plane survivor bitmasks.
///
/// Each field is a bitmask where bit i = 1 means codebook entry i has that
/// halo type. The 8 masks are disjoint and partition the full population.
///
/// Cost: 7 AND + 7 NOT per u64 word. Essentially free on AVX-512.
#[derive(Clone, Debug)]
pub struct CrossPlaneVote {
    /// 3-of-3: full SPO agreement. Promote to finer-grained search.
    pub core: Vec<u64>,
    /// S + P only: subject performing action, object unknown.
    pub sp: Vec<u64>,
    /// S + O only: entities related, predicate unknown.
    pub so: Vec<u64>,
    /// P + O only: action on target, agent unknown.
    pub po: Vec<u64>,
    /// S only: unbound subject (free variable).
    pub s_only: Vec<u64>,
    /// P only: unbound predicate (free variable).
    pub p_only: Vec<u64>,
    /// O only: unbound object (free variable).
    pub o_only: Vec<u64>,
    /// 0-of-3: no plane membership. Noise.
    pub noise: Vec<u64>,
    /// Number of codebook entries.
    pub n_entries: usize,
    /// Number of u64 words in each mask.
    pub n_words: usize,
}

impl CrossPlaneVote {
    /// Extract 8 disjoint halo type masks from 3 per-plane survivor masks.
    ///
    /// Each input mask has `n_words` u64 words. Bit i = 1 means entry i
    /// survived the sigma-gated filter in that plane.
    ///
    /// All operations are bitwise AND/NOT -- no branches, no floats.
    pub fn extract(s_mask: &[u64], p_mask: &[u64], o_mask: &[u64], n_entries: usize) -> Self {
        let n_words = s_mask.len();
        assert_eq!(p_mask.len(), n_words);
        assert_eq!(o_mask.len(), n_words);

        let mut core = vec![0u64; n_words];
        let mut sp = vec![0u64; n_words];
        let mut so = vec![0u64; n_words];
        let mut po = vec![0u64; n_words];
        let mut s_only = vec![0u64; n_words];
        let mut p_only = vec![0u64; n_words];
        let mut o_only = vec![0u64; n_words];
        let mut noise = vec![0u64; n_words];

        for i in 0..n_words {
            let s = s_mask[i];
            let p = p_mask[i];
            let o = o_mask[i];
            let ns = !s;
            let np = !p;
            let no = !o;

            core[i] = s & p & o;
            sp[i] = s & p & no;
            so[i] = s & np & o;
            po[i] = ns & p & o;
            s_only[i] = s & np & no;
            p_only[i] = ns & p & no;
            o_only[i] = ns & np & o;
            noise[i] = ns & np & no;
        }

        Self {
            core,
            sp,
            so,
            po,
            s_only,
            p_only,
            o_only,
            noise,
            n_entries,
            n_words,
        }
    }

    /// Count entries of each halo type.
    pub fn distribution(&self) -> HaloDistribution {
        HaloDistribution {
            core: popcount_mask(&self.core, self.n_entries),
            sp: popcount_mask(&self.sp, self.n_entries),
            so: popcount_mask(&self.so, self.n_entries),
            po: popcount_mask(&self.po, self.n_entries),
            s_only: popcount_mask(&self.s_only, self.n_entries),
            p_only: popcount_mask(&self.p_only, self.n_entries),
            o_only: popcount_mask(&self.o_only, self.n_entries),
            noise: popcount_mask(&self.noise, self.n_entries),
            total: self.n_entries,
        }
    }

    /// Get the mask for a specific halo type.
    pub fn mask_for(&self, halo: HaloType) -> &[u64] {
        match halo {
            HaloType::Noise => &self.noise,
            HaloType::S => &self.s_only,
            HaloType::P => &self.p_only,
            HaloType::O => &self.o_only,
            HaloType::SP => &self.sp,
            HaloType::SO => &self.so,
            HaloType::PO => &self.po,
            HaloType::Core => &self.core,
        }
    }

    /// Iterate over set bit indices in a halo type mask.
    pub fn entries_of(&self, halo: HaloType) -> Vec<usize> {
        let mask = self.mask_for(halo);
        let mut entries = Vec::new();
        for (word_idx, &word) in mask.iter().enumerate() {
            let mut w = word;
            while w != 0 {
                let bit = w.trailing_zeros() as usize;
                let idx = word_idx * 64 + bit;
                if idx < self.n_entries {
                    entries.push(idx);
                }
                w &= w - 1; // clear lowest set bit
            }
        }
        entries
    }
}

/// Population counts per halo type.
#[derive(Clone, Debug)]
pub struct HaloDistribution {
    pub core: usize,
    pub sp: usize,
    pub so: usize,
    pub po: usize,
    pub s_only: usize,
    pub p_only: usize,
    pub o_only: usize,
    pub noise: usize,
    pub total: usize,
}

impl HaloDistribution {
    /// Fraction of entries at each lattice level.
    pub fn level_fractions(&self) -> [f32; 4] {
        let t = self.total.max(1) as f32;
        [
            self.noise as f32 / t,
            (self.s_only + self.p_only + self.o_only) as f32 / t,
            (self.sp + self.so + self.po) as f32 / t,
            self.core as f32 / t,
        ]
    }
}

// ============================================================================
// Inference Modes -- typed queries from partial bindings
// ============================================================================

/// Inference mode determined by the open slot(s) in a partial binding.
///
/// Each mode corresponds to a specific cognitive query:
/// - Forward: known SP, find O ("who does what to WHOM?")
/// - Backward: known PO, find S ("WHO does this to whom?")
/// - Abduction: known SO, find P ("how ARE they related?")
/// - Analogy: single plane known, find the other two
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceMode {
    /// SP known -> find O. "Jan creates ???"
    Forward,
    /// PO known -> find S. "??? creates Ada"
    Backward,
    /// SO known -> find P. "Jan ??? Ada"
    Abduction,
    /// Single plane -> find the other two. Analogical transfer.
    Analogy,
}

/// A typed query on the cross-plane lattice.
///
/// Encodes which slots are known and which are open, plus the known
/// fingerprints for initialization.
#[derive(Clone, Debug)]
pub struct TypedQuery {
    pub mode: InferenceMode,
    /// Known S-plane fingerprint (if available).
    pub subject: Option<Fingerprint<256>>,
    /// Known P-plane fingerprint (if available).
    pub predicate: Option<Fingerprint<256>>,
    /// Known O-plane fingerprint (if available).
    pub object: Option<Fingerprint<256>>,
}

impl TypedQuery {
    /// Create a forward query: known S and P, find O.
    pub fn forward(subject: Fingerprint<256>, predicate: Fingerprint<256>) -> Self {
        Self {
            mode: InferenceMode::Forward,
            subject: Some(subject),
            predicate: Some(predicate),
            object: None,
        }
    }

    /// Create a backward query: known P and O, find S.
    pub fn backward(predicate: Fingerprint<256>, object: Fingerprint<256>) -> Self {
        Self {
            mode: InferenceMode::Backward,
            subject: None,
            predicate: Some(predicate),
            object: Some(object),
        }
    }

    /// Create an abductive query: known S and O, find P.
    pub fn abduction(subject: Fingerprint<256>, object: Fingerprint<256>) -> Self {
        Self {
            mode: InferenceMode::Abduction,
            subject: Some(subject),
            predicate: None,
            object: Some(object),
        }
    }

    /// Create an analogical query: hold one plane, resonate others.
    pub fn analogy_from_predicate(predicate: Fingerprint<256>) -> Self {
        Self {
            mode: InferenceMode::Analogy,
            subject: None,
            predicate: Some(predicate),
            object: None,
        }
    }

    /// Number of known slots (1 for analogy, 2 for forward/backward/abduction).
    pub fn known_count(&self) -> usize {
        self.subject.is_some() as usize
            + self.predicate.is_some() as usize
            + self.object.is_some() as usize
    }
}

// ============================================================================
// Partial Binding -- a candidate with its halo type and confidence
// ============================================================================

/// A candidate entry annotated with its plane membership and confidence.
#[derive(Clone, Debug)]
pub struct PartialBinding {
    /// Codebook index of this entry.
    pub entry_index: usize,
    /// Which planes agree on this candidate.
    pub halo_type: HaloType,
    /// Confidence from plane agreement: sum of per-plane Hamming similarities.
    /// Range: [0.0, 3.0] for core, [0.0, 2.0] for pairs, [0.0, 1.0] for free vars.
    pub confidence: f32,
    /// Per-plane Hamming distances (u32::MAX if plane does not agree).
    pub plane_distances: [u32; 3],
    /// Per-plane sigma-significance levels (Noise if plane does not agree).
    ///
    /// Each plane's Hamming distance is independently scored against the
    /// noise floor sigma = sqrt(D/4). This gives per-plane statistical significance
    /// that feeds into the B_3 lattice halo classification.
    pub plane_sigma: [SignificanceLevel; 3],
}

impl PartialBinding {
    /// NARS-style truth value: (frequency, confidence).
    ///
    /// Frequency comes from Hamming similarity in agreeing planes.
    /// Confidence is proportional to plane count / 3.
    pub fn nars_truth(&self) -> (f32, f32) {
        let n_planes = self.halo_type.plane_count() as f32;
        // Confidence: proportional to agreeing plane count
        let conf = n_planes / 3.0;
        // Frequency: average similarity in agreeing planes
        let mut sum_sim = 0.0f32;
        let mut count = 0;
        for &d in &self.plane_distances {
            if d != u32::MAX {
                // Similarity = 1 - distance/total_bits
                sum_sim += 1.0 - d as f32 / Fingerprint::<256>::BITS as f32;
                count += 1;
            }
        }
        let freq = if count > 0 {
            sum_sim / count as f32
        } else {
            0.0
        };
        (freq, conf)
    }
}

// ============================================================================
// DN Growth -- lattice climbing from fragments to full triples
// ============================================================================

/// One of 6 growth paths from free variable through partial pair to full triple.
///
/// Each path corresponds to a different cognitive/linguistic strategy:
/// - SubjectFirst (S->SP->SPO): "Jan does something to someone" (SVO languages)
/// - SubjectObject (S->SO->SPO): "Jan and Ada relate somehow" (implicit relation)
/// - ObjectAction (O->PO->SPO): "something is done to Ada" (passive, OVS)
/// - ActionFirst (P->SP->SPO): "creation happens, by whom?" (VSO)
/// - ActionObject (P->PO->SPO): "something creates Ada" (VOS)
/// - ObjectSubject (O->SO->SPO): "Ada and Jan, what relation?" (OSV)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrowthPath {
    SubjectFirst,
    SubjectObject,
    ObjectAction,
    ActionFirst,
    ActionObject,
    ObjectSubject,
}

impl GrowthPath {
    /// The sequence of halo types along this growth path.
    pub fn stages(self) -> [HaloType; 3] {
        match self {
            GrowthPath::SubjectFirst => [HaloType::S, HaloType::SP, HaloType::Core],
            GrowthPath::SubjectObject => [HaloType::S, HaloType::SO, HaloType::Core],
            GrowthPath::ObjectAction => [HaloType::O, HaloType::PO, HaloType::Core],
            GrowthPath::ActionFirst => [HaloType::P, HaloType::SP, HaloType::Core],
            GrowthPath::ActionObject => [HaloType::P, HaloType::PO, HaloType::Core],
            GrowthPath::ObjectSubject => [HaloType::O, HaloType::SO, HaloType::Core],
        }
    }
}

/// DN mutation operator -- replace one or two slots in an SPO triple.
///
/// Single-slot mutations (S, P, O) are conservative (exploitation).
/// Double-slot mutations (SP, SO, PO) are radical (exploration).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutationOp {
    /// Replace subject, keep predicate + object.
    MutateS,
    /// Replace predicate, keep subject + object.
    MutateP,
    /// Replace object, keep subject + predicate.
    MutateO,
    /// Replace subject + predicate, keep object (radical).
    MutateSP,
    /// Replace subject + object, keep predicate (radical).
    MutateSO,
    /// Replace predicate + object, keep subject (radical).
    MutatePO,
}

impl MutationOp {
    /// Number of slots being replaced (1 or 2).
    #[inline]
    pub fn slot_count(self) -> u8 {
        match self {
            MutationOp::MutateS | MutationOp::MutateP | MutationOp::MutateO => 1,
            MutationOp::MutateSP | MutationOp::MutateSO | MutationOp::MutatePO => 2,
        }
    }

    /// Whether this is a conservative (single-slot) mutation.
    #[inline]
    pub fn is_conservative(self) -> bool {
        self.slot_count() == 1
    }
}

/// Tracks partial bindings as they compose into full triples over time.
///
/// The lattice climber monitors the distribution across lattice levels
/// and detects when partial hypotheses compose into full triples (core).
/// This implements the staged error threshold from DN theory: each lattice
/// level requires progressively more cross-plane agreement.
#[derive(Clone, Debug)]
pub struct LatticeClimber {
    /// Free variables (lattice level 1): S, P, O types.
    pub free_vars: Vec<PartialBinding>,
    /// Partial pairs (lattice level 2): SP, SO, PO types.
    pub partial_pairs: Vec<PartialBinding>,
    /// Full triples (lattice level 3): Core type. Ready for DN insertion.
    pub full_triples: Vec<PartialBinding>,
    /// Growth path tracking: which paths are active and at which stage.
    pub active_paths: Vec<(GrowthPath, u8)>,
}

impl LatticeClimber {
    pub fn new() -> Self {
        Self {
            free_vars: Vec::new(),
            partial_pairs: Vec::new(),
            full_triples: Vec::new(),
            active_paths: Vec::new(),
        }
    }

    /// Ingest new partial bindings from a cross-plane vote, sorted into
    /// the appropriate lattice level.
    pub fn ingest(&mut self, bindings: &[PartialBinding]) {
        for b in bindings {
            match b.halo_type.lattice_level() {
                1 => self.free_vars.push(b.clone()),
                2 => self.partial_pairs.push(b.clone()),
                3 => self.full_triples.push(b.clone()),
                _ => {} // noise (0) discarded
            }
        }
    }

    /// Ingest candidates from raw per-plane Hamming distances with sigma-classification.
    ///
    /// This is the sigma-grounded alternative to `ingest()`: instead of receiving
    /// pre-built PartialBindings, it builds them from raw distances using
    /// `classify_with_sigma()`. Each plane's distance is scored against the
    /// noise floor, and the resulting per-plane significance levels determine
    /// the B_3 halo type.
    pub fn ingest_with_sigma(
        &mut self,
        candidates: &[(usize, [u32; 3])],
        sigma_gate: &SigmaGate,
        min_level: SignificanceLevel,
    ) {
        for &(entry_index, plane_distances) in candidates {
            let binding = classify_with_sigma(entry_index, plane_distances, sigma_gate, min_level);
            match binding.halo_type.lattice_level() {
                1 => self.free_vars.push(binding),
                2 => self.partial_pairs.push(binding),
                3 => self.full_triples.push(binding),
                _ => {} // noise (0) discarded
            }
        }
    }

    /// Attempt to compose partial bindings at adjacent lattice levels.
    ///
    /// Looks for free variables that can pair with existing free variables
    /// to form partial pairs, and partial pairs that can combine with free
    /// variables to form full triples. Uses XOR binding for composition.
    ///
    /// Returns newly promoted bindings.
    pub fn try_compose(
        &mut self,
        codebook_s: &[Fingerprint<256>],
        codebook_p: &[Fingerprint<256>],
        codebook_o: &[Fingerprint<256>],
        threshold: u32,
    ) -> Vec<PartialBinding> {
        let mut promoted = Vec::new();

        // Try to promote partial pairs -> full triples
        // SP + O -> Core, SO + P -> Core, PO + S -> Core
        let pairs = self.partial_pairs.clone();
        let fvs = self.free_vars.clone();

        for pair in &pairs {
            for fv in &fvs {
                let composed = try_compose_pair_and_free(
                    pair, fv, codebook_s, codebook_p, codebook_o, threshold,
                );
                if let Some(full) = composed {
                    promoted.push(full);
                }
            }
        }

        // Move promoted to full_triples
        for p in &promoted {
            self.full_triples.push(p.clone());
        }

        promoted
    }

    /// Evaluate the current state as a CollapseGate decision.
    ///
    /// - FLOW if we have full triples with high confidence
    /// - HOLD if we have partial pairs (hypothesis under construction)
    /// - BLOCK if only noise or conflicting free variables
    pub fn gate_decision(&self) -> CollapseGate {
        if !self.full_triples.is_empty() {
            // Check average confidence of full triples
            let avg_conf: f32 = self.full_triples.iter().map(|t| t.confidence).sum::<f32>()
                / self.full_triples.len() as f32;
            if avg_conf > 1.5 {
                return CollapseGate::Flow;
            }
        }
        if !self.partial_pairs.is_empty() {
            return CollapseGate::Hold;
        }
        if !self.free_vars.is_empty() {
            return CollapseGate::Hold;
        }
        CollapseGate::Block
    }

    /// Current distribution across lattice levels: [noise, free, pair, core].
    pub fn level_counts(&self) -> [usize; 4] {
        [
            0, // noise is never stored
            self.free_vars.len(),
            self.partial_pairs.len(),
            self.full_triples.len(),
        ]
    }
}

impl Default for LatticeClimber {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// SPO Triple -- a resolved full binding
// ============================================================================

/// A fully resolved SPO triple with its constituent fingerprints.
#[derive(Clone, Debug)]
pub struct SpoTriple {
    pub subject: Fingerprint<256>,
    pub predicate: Fingerprint<256>,
    pub object: Fingerprint<256>,
    /// Confidence: average similarity across all 3 planes.
    pub confidence: f32,
}

impl SpoTriple {
    /// Apply a mutation operator, replacing slot(s) with new fingerprint(s).
    pub fn mutate(
        &self,
        op: MutationOp,
        replacement: &Fingerprint<256>,
        rng: &mut SplitMix64,
    ) -> Self {
        let random_fp = random_fingerprint(rng);
        match op {
            MutationOp::MutateS => SpoTriple {
                subject: replacement.clone(),
                predicate: self.predicate.clone(),
                object: self.object.clone(),
                confidence: self.confidence * 0.5,
            },
            MutationOp::MutateP => SpoTriple {
                subject: self.subject.clone(),
                predicate: replacement.clone(),
                object: self.object.clone(),
                confidence: self.confidence * 0.5,
            },
            MutationOp::MutateO => SpoTriple {
                subject: self.subject.clone(),
                predicate: self.predicate.clone(),
                object: replacement.clone(),
                confidence: self.confidence * 0.5,
            },
            MutationOp::MutateSP => SpoTriple {
                subject: replacement.clone(),
                predicate: random_fp,
                object: self.object.clone(),
                confidence: self.confidence * 0.25,
            },
            MutationOp::MutateSO => SpoTriple {
                subject: replacement.clone(),
                predicate: self.predicate.clone(),
                object: random_fp,
                confidence: self.confidence * 0.25,
            },
            MutationOp::MutatePO => SpoTriple {
                subject: self.subject.clone(),
                predicate: replacement.clone(),
                object: random_fp,
                confidence: self.confidence * 0.25,
            },
        }
    }

    /// XOR-encode into 3D crystal (S^P, P^O, S^O).
    pub fn encode(&self) -> [Fingerprint<256>; 3] {
        [
            &self.subject ^ &self.predicate,
            &self.predicate ^ &self.object,
            &self.subject ^ &self.object,
        ]
    }
}

// ============================================================================
// Resonator Warm-Start from Partial Bindings
// ============================================================================

/// Warm-start configuration for a resonator from partial binding evidence.
///
/// Instead of random initialization, pre-fills known slots from the
/// halo type, so the resonator only needs to find the open slot(s).
/// Expected speedup: ~K x where K = number of pre-filled planes.
#[derive(Clone, Debug)]
pub struct WarmStart {
    /// Initial estimate for S-plane (None = random init).
    pub s_init: Option<Fingerprint<256>>,
    /// Initial estimate for P-plane (None = random init).
    pub p_init: Option<Fingerprint<256>>,
    /// Initial estimate for O-plane (None = random init).
    pub o_init: Option<Fingerprint<256>>,
    /// Number of pre-filled planes (1 or 2).
    pub prefilled: u8,
}

impl WarmStart {
    /// Create warm-start from a typed query.
    pub fn from_query(query: &TypedQuery) -> Self {
        Self {
            s_init: query.subject.clone(),
            p_init: query.predicate.clone(),
            o_init: query.object.clone(),
            prefilled: query.known_count() as u8,
        }
    }

    /// Fill any None slots with random fingerprints.
    pub fn fill_random(&mut self, rng: &mut SplitMix64) {
        if self.s_init.is_none() {
            self.s_init = Some(random_fingerprint(rng));
        }
        if self.p_init.is_none() {
            self.p_init = Some(random_fingerprint(rng));
        }
        if self.o_init.is_none() {
            self.o_init = Some(random_fingerprint(rng));
        }
    }
}

// ============================================================================
// Inference Engine -- forward, backward, abductive, analogical
// ============================================================================

/// Result of an inference query.
#[derive(Clone, Debug)]
pub struct InferenceResult {
    /// The inferred fingerprint for the open slot.
    pub inferred: Fingerprint<256>,
    /// Confidence of the inference (Hamming similarity to best match).
    pub confidence: f32,
    /// Codebook index of the best match (if resolved).
    pub best_match: Option<usize>,
    /// Which inference mode was used.
    pub mode: InferenceMode,
}

/// Run inference by finding the best codebook entry for the open slot.
///
/// Uses the XOR self-inverse property of binding:
/// - Forward (SP->O): if crystal_y = P^O, then O = crystal_y ^ P
/// - Backward (PO->S): if crystal_x = S^P, then S = crystal_x ^ P
/// - Abduction (SO->P): if crystal_x = S^P, then P = crystal_x ^ S
///
/// The query provides known slots; the codebook provides candidates for
/// the open slot. Returns the best-matching candidate.
pub fn infer(
    query: &TypedQuery,
    crystal: &[Fingerprint<256>; 3], // [S^P, P^O, S^O]
    codebook: &[Fingerprint<256>],
) -> Option<InferenceResult> {
    if codebook.is_empty() {
        return None;
    }

    match query.mode {
        InferenceMode::Forward => {
            // Known: S, P. Find O.
            // crystal[1] = P^O, so O = crystal[1] ^ P
            let p = query.predicate.as_ref()?;
            let o_estimate = crystal[1].clone() ^ p.clone();
            find_best_match(&o_estimate, codebook, InferenceMode::Forward)
        }
        InferenceMode::Backward => {
            // Known: P, O. Find S.
            // crystal[0] = S^P, so S = crystal[0] ^ P
            let p = query.predicate.as_ref()?;
            let s_estimate = crystal[0].clone() ^ p.clone();
            find_best_match(&s_estimate, codebook, InferenceMode::Backward)
        }
        InferenceMode::Abduction => {
            // Known: S, O. Find P.
            // crystal[0] = S^P, so P = crystal[0] ^ S
            let s = query.subject.as_ref()?;
            let p_estimate = crystal[0].clone() ^ s.clone();
            find_best_match(&p_estimate, codebook, InferenceMode::Abduction)
        }
        InferenceMode::Analogy => {
            // Known: one plane (e.g., P). Find the other two via crystal decoding.
            // This is a 2-slot inference -- find best combined (S, O) for fixed P.
            if let Some(p) = query.predicate.as_ref() {
                let s_estimate = crystal[0].clone() ^ p.clone();
                find_best_match(&s_estimate, codebook, InferenceMode::Analogy)
            } else if let Some(s) = query.subject.as_ref() {
                let p_estimate = crystal[0].clone() ^ s.clone();
                find_best_match(&p_estimate, codebook, InferenceMode::Analogy)
            } else if let Some(o) = query.object.as_ref() {
                let p_estimate = crystal[2].clone() ^ o.clone();
                find_best_match(&p_estimate, codebook, InferenceMode::Analogy)
            } else {
                None
            }
        }
    }
}

// ============================================================================
// Sigma-Gated Cross-Plane Vote -- per-plane sigma-significance thresholds
// ============================================================================

/// Per-plane sigma-significance summary for a candidate entry.
///
/// Records the sigma-score from each plane independently. This is the
/// per-plane decomposition of the aggregate Hamming distance, giving
/// 2^3 = 8 halo types with statistical grounding instead of ad-hoc thresholds.
#[derive(Clone, Debug)]
pub struct PlaneSignificance {
    /// sigma-significance for S-plane distance.
    pub s: SigmaScore,
    /// sigma-significance for P-plane distance.
    pub p: SigmaScore,
    /// sigma-significance for O-plane distance.
    pub o: SigmaScore,
}

impl PlaneSignificance {
    /// Classify halo type from per-plane sigma-significance.
    ///
    /// A plane is considered a survivor if its sigma-level meets the minimum
    /// threshold. The resulting HaloType is the B_3 lattice element from
    /// the 3-bit membership vector.
    pub fn halo_type(&self, min_level: SignificanceLevel) -> HaloType {
        let s = self.s.level >= min_level;
        let p = self.p.level >= min_level;
        let o = self.o.level >= min_level;
        HaloType::from_membership(s, p, o)
    }

    /// Minimum sigma-level across agreeing planes.
    pub fn min_level(&self) -> SignificanceLevel {
        self.s.level.min(self.p.level).min(self.o.level)
    }

    /// Maximum sigma-level across any plane (strongest match).
    pub fn max_level(&self) -> SignificanceLevel {
        self.s.level.max(self.p.level).max(self.o.level)
    }

    /// Convert to per-plane SignificanceLevel array [S, P, O].
    pub fn levels(&self) -> [SignificanceLevel; 3] {
        [self.s.level, self.p.level, self.o.level]
    }
}

/// Build `PartialBinding`s from per-plane Hamming distances with sigma-significance.
///
/// This replaces the binary survivor mask approach with sigma-grounded classification.
/// Each plane's distance is scored against its noise floor, and the resulting
/// per-plane significance levels determine the B_3 halo type.
pub fn classify_with_sigma(
    entry_index: usize,
    plane_distances: [u32; 3],
    sigma_gate: &SigmaGate,
    min_level: SignificanceLevel,
) -> PartialBinding {
    // Score each plane independently
    let levels: [SignificanceLevel; 3] = plane_distances.map(|dist| {
        let ec = EnergyConflict {
            conflict: dist,
            energy_a: 0,
            energy_b: 0,
            agreement: 0,
        };
        score_sigma(&ec, sigma_gate).level
    });

    let halo = HaloType::from_membership(
        levels[0] >= min_level,
        levels[1] >= min_level,
        levels[2] >= min_level,
    );

    // Confidence from sigma-significance: sum of per-plane sigma values
    let sigmas: [f32; 3] = plane_distances.map(|dist| {
        let ec = EnergyConflict {
            conflict: dist,
            energy_a: 0,
            energy_b: 0,
            agreement: 0,
        };
        score_sigma(&ec, sigma_gate).sigma
    });
    let confidence: f32 = sigmas.iter().filter(|s| **s >= 0.0).sum();

    PartialBinding {
        entry_index,
        halo_type: halo,
        confidence,
        plane_distances,
        plane_sigma: levels,
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Count set bits in a bitmask, capped at n_entries.
fn popcount_mask(mask: &[u64], n_entries: usize) -> usize {
    let full_words = n_entries / 64;
    let remaining_bits = n_entries % 64;
    let mut count: usize = 0;
    for &word in mask.iter().take(full_words) {
        count += word.count_ones() as usize;
    }
    if remaining_bits > 0 && full_words < mask.len() {
        let tail_mask = (1u64 << remaining_bits) - 1;
        count += (mask[full_words] & tail_mask).count_ones() as usize;
    }
    count
}

/// Find best matching codebook entry by Hamming distance.
fn find_best_match(
    estimate: &Fingerprint<256>,
    codebook: &[Fingerprint<256>],
    mode: InferenceMode,
) -> Option<InferenceResult> {
    let mut best_idx = 0;
    let mut best_dist = u32::MAX;
    for (i, entry) in codebook.iter().enumerate() {
        let d = estimate.hamming_distance(entry);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    let similarity = 1.0 - best_dist as f32 / Fingerprint::<256>::BITS as f32;
    Some(InferenceResult {
        inferred: codebook[best_idx].clone(),
        confidence: similarity,
        best_match: Some(best_idx),
        mode,
    })
}

/// Generate a random Fingerprint<256> from the RNG.
fn random_fingerprint(rng: &mut SplitMix64) -> Fingerprint<256> {
    let mut words = [0u64; 256];
    for word in &mut words {
        *word = rng.next_u64();
    }
    Fingerprint::from_words(words)
}

/// Try to compose a partial pair with a free variable into a full triple.
///
/// Checks if the free variable fills the open slot in the partial pair,
/// and if the resulting triple has Hamming distance below threshold on
/// the newly filled plane.
fn try_compose_pair_and_free(
    pair: &PartialBinding,
    fv: &PartialBinding,
    _codebook_s: &[Fingerprint<256>],
    _codebook_p: &[Fingerprint<256>],
    _codebook_o: &[Fingerprint<256>],
    threshold: u32,
) -> Option<PartialBinding> {
    // SP + O -> Core
    if pair.halo_type == HaloType::SP && fv.halo_type == HaloType::O {
        let o_dist = fv.plane_distances[2]; // O-plane distance
        if o_dist != u32::MAX && o_dist < threshold {
            return Some(PartialBinding {
                entry_index: pair.entry_index,
                halo_type: HaloType::Core,
                confidence: pair.confidence + fv.confidence,
                plane_distances: [pair.plane_distances[0], pair.plane_distances[1], o_dist],
                plane_sigma: [pair.plane_sigma[0], pair.plane_sigma[1], fv.plane_sigma[2]],
            });
        }
    }
    // SO + P -> Core
    if pair.halo_type == HaloType::SO && fv.halo_type == HaloType::P {
        let p_dist = fv.plane_distances[1];
        if p_dist != u32::MAX && p_dist < threshold {
            return Some(PartialBinding {
                entry_index: pair.entry_index,
                halo_type: HaloType::Core,
                confidence: pair.confidence + fv.confidence,
                plane_distances: [pair.plane_distances[0], p_dist, pair.plane_distances[2]],
                plane_sigma: [pair.plane_sigma[0], fv.plane_sigma[1], pair.plane_sigma[2]],
            });
        }
    }
    // PO + S -> Core
    if pair.halo_type == HaloType::PO && fv.halo_type == HaloType::S {
        let s_dist = fv.plane_distances[0];
        if s_dist != u32::MAX && s_dist < threshold {
            return Some(PartialBinding {
                entry_index: pair.entry_index,
                halo_type: HaloType::Core,
                confidence: pair.confidence + fv.confidence,
                plane_distances: [s_dist, pair.plane_distances[1], pair.plane_distances[2]],
                plane_sigma: [fv.plane_sigma[0], pair.plane_sigma[1], pair.plane_sigma[2]],
            });
        }
    }
    None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> SplitMix64 {
        SplitMix64::new(42)
    }

    /// Helper: create a mask with specific bits set.
    fn mask_from_bits(bits: &[usize], n_words: usize) -> Vec<u64> {
        let mut mask = vec![0u64; n_words];
        for &b in bits {
            mask[b / 64] |= 1u64 << (b % 64);
        }
        mask
    }

    #[test]
    fn test_halo_type_plane_count() {
        assert_eq!(HaloType::Noise.plane_count(), 0);
        assert_eq!(HaloType::S.plane_count(), 1);
        assert_eq!(HaloType::P.plane_count(), 1);
        assert_eq!(HaloType::O.plane_count(), 1);
        assert_eq!(HaloType::SP.plane_count(), 2);
        assert_eq!(HaloType::SO.plane_count(), 2);
        assert_eq!(HaloType::PO.plane_count(), 2);
        assert_eq!(HaloType::Core.plane_count(), 3);
    }

    #[test]
    fn test_halo_type_from_membership() {
        assert_eq!(HaloType::from_membership(true, true, true), HaloType::Core);
        assert_eq!(HaloType::from_membership(true, true, false), HaloType::SP);
        assert_eq!(HaloType::from_membership(true, false, true), HaloType::SO);
        assert_eq!(HaloType::from_membership(false, true, true), HaloType::PO);
        assert_eq!(HaloType::from_membership(true, false, false), HaloType::S);
        assert_eq!(HaloType::from_membership(false, true, false), HaloType::P);
        assert_eq!(HaloType::from_membership(false, false, true), HaloType::O);
        assert_eq!(
            HaloType::from_membership(false, false, false),
            HaloType::Noise
        );
    }

    #[test]
    fn test_cross_plane_vote_disjoint_partition() {
        // 128 entries = 2 u64 words
        let n = 128;
        let nw = 2;

        let s_mask = vec![0xAAAA_AAAA_AAAA_AAAAu64, 0x5555_5555_5555_5555u64];
        let p_mask = vec![0xCCCC_CCCC_CCCC_CCCCu64, 0x3333_3333_3333_3333u64];
        let o_mask = vec![0xF0F0_F0F0_F0F0_F0F0u64, 0x0F0F_0F0F_0F0F_0F0Fu64];

        let vote = CrossPlaneVote::extract(&s_mask, &p_mask, &o_mask, n);

        // Every entry should be in exactly one category
        for i in 0..nw {
            let union = vote.core[i]
                | vote.sp[i]
                | vote.so[i]
                | vote.po[i]
                | vote.s_only[i]
                | vote.p_only[i]
                | vote.o_only[i]
                | vote.noise[i];
            assert_eq!(union, u64::MAX, "word {}: not all bits covered", i);

            // Pairwise disjoint: AND of any two should be 0
            let masks = [
                vote.core[i],
                vote.sp[i],
                vote.so[i],
                vote.po[i],
                vote.s_only[i],
                vote.p_only[i],
                vote.o_only[i],
                vote.noise[i],
            ];
            for a in 0..8 {
                for b in (a + 1)..8 {
                    assert_eq!(
                        masks[a] & masks[b],
                        0,
                        "word {}: masks {} and {} overlap",
                        i,
                        a,
                        b
                    );
                }
            }
        }
    }

    #[test]
    fn test_cross_plane_vote_all_three_is_core() {
        let n = 64;
        // Entry 0 survives in all 3 planes
        let s_mask = vec![0x01u64];
        let p_mask = vec![0x01u64];
        let o_mask = vec![0x01u64];

        let vote = CrossPlaneVote::extract(&s_mask, &p_mask, &o_mask, n);
        assert_eq!(vote.core[0] & 1, 1);
        assert_eq!(vote.sp[0] & 1, 0);
        assert_eq!(vote.noise[0] & 1, 0);
    }

    #[test]
    fn test_cross_plane_vote_sp_type() {
        let n = 64;
        // Entry 5 survives in S and P but not O
        let s_mask = vec![1u64 << 5];
        let p_mask = vec![1u64 << 5];
        let o_mask = vec![0u64];

        let vote = CrossPlaneVote::extract(&s_mask, &p_mask, &o_mask, n);
        assert_eq!(vote.sp[0] & (1 << 5), 1 << 5);
        assert_eq!(vote.core[0] & (1 << 5), 0);
    }

    #[test]
    fn test_cross_plane_vote_distribution() {
        let n = 64;
        // S: entries 0-31, P: entries 16-47, O: entries 32-63
        let s_mask = vec![0x0000_0000_FFFF_FFFFu64];
        let p_mask = vec![0x0000_FFFF_FFFF_0000u64];
        let o_mask = vec![0xFFFF_FFFF_0000_0000u64];

        let vote = CrossPlaneVote::extract(&s_mask, &p_mask, &o_mask, n);
        let dist = vote.distribution();

        // Core = S & P & O = entries 32-47 would need all 3... let's check
        // S: bits 0-31, P: bits 16-47, O: bits 32-63
        // Core = bits 32-47 ... no, S only has 0-31. So no core entries.
        // Actually: S=0-31 (bits 0-31), P=16-47 (bits 16-47), O=32-63 (bits 32-63)
        // Core = S & P & O = nothing (S stops at 31, O starts at 32)
        assert_eq!(dist.core, 0);
        // SP = S & P & !O = bits 16-31
        assert_eq!(dist.sp, 16);
        // S only = bits 0-15
        assert_eq!(dist.s_only, 16);
        // PO = !S & P & O = bits 32-47
        assert_eq!(dist.po, 16);
        // O only = bits 48-63
        assert_eq!(dist.o_only, 16);
        // P only = none (P overlaps with either S or O everywhere)
        assert_eq!(dist.p_only, 0);
    }

    #[test]
    fn test_entries_of_returns_correct_indices() {
        let n = 128;
        let s_mask = mask_from_bits(&[3, 7, 100], 2);
        let p_mask = mask_from_bits(&[3, 100], 2);
        let o_mask = mask_from_bits(&[7, 100], 2);

        let vote = CrossPlaneVote::extract(&s_mask, &p_mask, &o_mask, n);

        let core_entries = vote.entries_of(HaloType::Core);
        assert_eq!(core_entries, vec![100]); // all 3 planes

        let sp_entries = vote.entries_of(HaloType::SP);
        assert_eq!(sp_entries, vec![3]); // S + P, not O

        let so_entries = vote.entries_of(HaloType::SO);
        assert_eq!(so_entries, vec![7]); // S + O, not P
    }

    #[test]
    fn test_halo_inference_mode() {
        assert_eq!(HaloType::SP.inference_mode(), Some(InferenceMode::Forward));
        assert_eq!(HaloType::PO.inference_mode(), Some(InferenceMode::Backward));
        assert_eq!(
            HaloType::SO.inference_mode(),
            Some(InferenceMode::Abduction)
        );
        assert_eq!(HaloType::S.inference_mode(), Some(InferenceMode::Analogy));
        assert_eq!(HaloType::Core.inference_mode(), None);
        assert_eq!(HaloType::Noise.inference_mode(), None);
    }

    #[test]
    fn test_typed_query_forward() {
        let mut rng = make_rng();
        let s = random_fingerprint(&mut rng);
        let p = random_fingerprint(&mut rng);
        let q = TypedQuery::forward(s, p);
        assert_eq!(q.mode, InferenceMode::Forward);
        assert_eq!(q.known_count(), 2);
        assert!(q.subject.is_some());
        assert!(q.predicate.is_some());
        assert!(q.object.is_none());
    }

    #[test]
    fn test_inference_forward_recovers_object() {
        let mut rng = make_rng();
        let s = random_fingerprint(&mut rng);
        let p = random_fingerprint(&mut rng);
        let o = random_fingerprint(&mut rng);

        // Encode SPO triple into crystal
        let crystal = [
            &s ^ &p, // crystal[0] = S^P
            &p ^ &o, // crystal[1] = P^O
            &s ^ &o, // crystal[2] = S^O
        ];

        // Codebook contains the original O plus some random entries
        let mut codebook = vec![random_fingerprint(&mut rng); 10];
        codebook[3] = o.clone();

        let query = TypedQuery::forward(s, p);
        let result = infer(&query, &crystal, &codebook).unwrap();

        // Should find entry 3 (the correct object)
        assert_eq!(result.best_match, Some(3));
        assert_eq!(result.confidence, 1.0); // exact match
        assert_eq!(result.mode, InferenceMode::Forward);
    }

    #[test]
    fn test_inference_backward_recovers_subject() {
        let mut rng = make_rng();
        let s = random_fingerprint(&mut rng);
        let p = random_fingerprint(&mut rng);
        let o = random_fingerprint(&mut rng);

        let crystal = [&s ^ &p, &p ^ &o, &s ^ &o];

        let mut codebook = vec![random_fingerprint(&mut rng); 10];
        codebook[7] = s.clone();

        let query = TypedQuery::backward(p, o);
        let result = infer(&query, &crystal, &codebook).unwrap();

        assert_eq!(result.best_match, Some(7));
        assert_eq!(result.confidence, 1.0);
        assert_eq!(result.mode, InferenceMode::Backward);
    }

    #[test]
    fn test_inference_abduction_recovers_predicate() {
        let mut rng = make_rng();
        let s = random_fingerprint(&mut rng);
        let p = random_fingerprint(&mut rng);
        let o = random_fingerprint(&mut rng);

        let crystal = [&s ^ &p, &p ^ &o, &s ^ &o];

        let mut codebook = vec![random_fingerprint(&mut rng); 10];
        codebook[5] = p.clone();

        let query = TypedQuery::abduction(s, o);
        let result = infer(&query, &crystal, &codebook).unwrap();

        assert_eq!(result.best_match, Some(5));
        assert_eq!(result.confidence, 1.0);
        assert_eq!(result.mode, InferenceMode::Abduction);
    }

    #[test]
    fn test_spo_triple_encode_decode() {
        let mut rng = make_rng();
        let s = random_fingerprint(&mut rng);
        let p = random_fingerprint(&mut rng);
        let o = random_fingerprint(&mut rng);

        let triple = SpoTriple {
            subject: s.clone(),
            predicate: p.clone(),
            object: o.clone(),
            confidence: 1.0,
        };

        let crystal = triple.encode();

        // Recover S from crystal[0] ^ P
        let s_recovered = crystal[0].clone() ^ p.clone();
        assert_eq!(s_recovered, s);

        // Recover O from crystal[1] ^ P
        let o_recovered = crystal[1].clone() ^ p.clone();
        assert_eq!(o_recovered, o);

        // Recover P from crystal[0] ^ S
        let p_recovered = crystal[0].clone() ^ s.clone();
        assert_eq!(p_recovered, p);
    }

    #[test]
    fn test_mutation_conservative_vs_radical() {
        assert!(MutationOp::MutateS.is_conservative());
        assert!(MutationOp::MutateP.is_conservative());
        assert!(MutationOp::MutateO.is_conservative());
        assert!(!MutationOp::MutateSP.is_conservative());
        assert!(!MutationOp::MutateSO.is_conservative());
        assert!(!MutationOp::MutatePO.is_conservative());
    }

    #[test]
    fn test_mutation_preserves_kept_slots() {
        let mut rng = make_rng();
        let s = random_fingerprint(&mut rng);
        let p = random_fingerprint(&mut rng);
        let o = random_fingerprint(&mut rng);
        let replacement = random_fingerprint(&mut rng);

        let triple = SpoTriple {
            subject: s.clone(),
            predicate: p.clone(),
            object: o.clone(),
            confidence: 1.0,
        };

        // MutateS: should keep P and O
        let mutated = triple.mutate(MutationOp::MutateS, &replacement, &mut rng);
        assert_eq!(mutated.subject, replacement);
        assert_eq!(mutated.predicate, p);
        assert_eq!(mutated.object, o);
        assert!(mutated.confidence < triple.confidence);
    }

    #[test]
    fn test_lattice_climber_ingest_and_levels() {
        let mut climber = LatticeClimber::new();

        let bindings = vec![
            PartialBinding {
                entry_index: 0,
                halo_type: HaloType::S,
                confidence: 0.8,
                plane_distances: [1000, u32::MAX, u32::MAX],
                plane_sigma: [
                    SignificanceLevel::Discovery,
                    SignificanceLevel::Noise,
                    SignificanceLevel::Noise,
                ],
            },
            PartialBinding {
                entry_index: 1,
                halo_type: HaloType::SP,
                confidence: 1.5,
                plane_distances: [1000, 2000, u32::MAX],
                plane_sigma: [
                    SignificanceLevel::Discovery,
                    SignificanceLevel::Discovery,
                    SignificanceLevel::Noise,
                ],
            },
            PartialBinding {
                entry_index: 2,
                halo_type: HaloType::Core,
                confidence: 2.5,
                plane_distances: [1000, 2000, 3000],
                plane_sigma: [
                    SignificanceLevel::Discovery,
                    SignificanceLevel::Discovery,
                    SignificanceLevel::Discovery,
                ],
            },
        ];

        climber.ingest(&bindings);

        let counts = climber.level_counts();
        assert_eq!(counts[1], 1); // 1 free var (S)
        assert_eq!(counts[2], 1); // 1 partial pair (SP)
        assert_eq!(counts[3], 1); // 1 full triple (Core)
    }

    #[test]
    fn test_lattice_climber_gate_decisions() {
        let mut climber = LatticeClimber::new();

        // Empty -> Block
        assert_eq!(climber.gate_decision(), CollapseGate::Block);

        // Free vars only -> Hold
        climber.free_vars.push(PartialBinding {
            entry_index: 0,
            halo_type: HaloType::S,
            confidence: 0.5,
            plane_distances: [1000, u32::MAX, u32::MAX],
            plane_sigma: [
                SignificanceLevel::Discovery,
                SignificanceLevel::Noise,
                SignificanceLevel::Noise,
            ],
        });
        assert_eq!(climber.gate_decision(), CollapseGate::Hold);

        // Full triple with high confidence -> Flow
        climber.full_triples.push(PartialBinding {
            entry_index: 1,
            halo_type: HaloType::Core,
            confidence: 2.5,
            plane_distances: [500, 600, 700],
            plane_sigma: [
                SignificanceLevel::Discovery,
                SignificanceLevel::Discovery,
                SignificanceLevel::Discovery,
            ],
        });
        assert_eq!(climber.gate_decision(), CollapseGate::Flow);
    }

    #[test]
    fn test_nars_truth_value() {
        let binding = PartialBinding {
            entry_index: 0,
            halo_type: HaloType::SP,
            confidence: 1.5,
            plane_distances: [
                1000, // S-plane: 1000 / 16384 ~ 6.1% distance -> 93.9% similarity
                2000, // P-plane: 2000 / 16384 ~ 12.2% distance -> 87.8% similarity
                u32::MAX,
            ],
            plane_sigma: [
                SignificanceLevel::Discovery,
                SignificanceLevel::Discovery,
                SignificanceLevel::Noise,
            ],
        };

        let (freq, conf) = binding.nars_truth();
        assert!((conf - 2.0 / 3.0).abs() < 0.01); // 2 of 3 planes
        assert!(freq > 0.8); // high similarity in agreeing planes
    }

    #[test]
    fn test_warm_start_from_forward_query() {
        let mut rng = make_rng();
        let s = random_fingerprint(&mut rng);
        let p = random_fingerprint(&mut rng);

        let query = TypedQuery::forward(s, p);
        let ws = WarmStart::from_query(&query);

        assert!(ws.s_init.is_some());
        assert!(ws.p_init.is_some());
        assert!(ws.o_init.is_none());
        assert_eq!(ws.prefilled, 2);
    }

    #[test]
    fn test_warm_start_fill_random() {
        let mut rng = make_rng();
        let s = random_fingerprint(&mut rng);
        let p = random_fingerprint(&mut rng);

        let query = TypedQuery::forward(s, p);
        let mut ws = WarmStart::from_query(&query);
        ws.fill_random(&mut rng);

        assert!(ws.s_init.is_some());
        assert!(ws.p_init.is_some());
        assert!(ws.o_init.is_some()); // now filled
    }

    #[test]
    fn test_growth_path_stages() {
        let stages = GrowthPath::SubjectFirst.stages();
        assert_eq!(stages, [HaloType::S, HaloType::SP, HaloType::Core]);

        let stages = GrowthPath::ObjectAction.stages();
        assert_eq!(stages, [HaloType::O, HaloType::PO, HaloType::Core]);
    }

    #[test]
    fn test_popcount_mask_exact() {
        // 100 entries in 2 words
        let mask = vec![u64::MAX, (1u64 << 36) - 1]; // 64 + 36 = 100 bits
        assert_eq!(popcount_mask(&mask, 100), 100);

        let mask2 = vec![0u64, 0u64];
        assert_eq!(popcount_mask(&mask2, 100), 0);
    }

    #[test]
    fn test_lattice_climber_compose_sp_plus_o() {
        let mut climber = LatticeClimber::new();
        let mut rng = make_rng();

        // SP pair at entry 0
        climber.partial_pairs.push(PartialBinding {
            entry_index: 0,
            halo_type: HaloType::SP,
            confidence: 1.5,
            plane_distances: [1000, 1200, u32::MAX],
            plane_sigma: [
                SignificanceLevel::Discovery,
                SignificanceLevel::Discovery,
                SignificanceLevel::Noise,
            ],
        });

        // O free var at entry 1
        climber.free_vars.push(PartialBinding {
            entry_index: 1,
            halo_type: HaloType::O,
            confidence: 0.7,
            plane_distances: [u32::MAX, u32::MAX, 800],
            plane_sigma: [
                SignificanceLevel::Noise,
                SignificanceLevel::Noise,
                SignificanceLevel::Discovery,
            ],
        });

        // Create dummy codebooks
        let codebook_s = vec![random_fingerprint(&mut rng)];
        let codebook_p = vec![random_fingerprint(&mut rng)];
        let codebook_o = vec![random_fingerprint(&mut rng)];

        // Threshold generous enough to allow composition
        let promoted = climber.try_compose(&codebook_s, &codebook_p, &codebook_o, 5000);

        assert_eq!(promoted.len(), 1);
        assert_eq!(promoted[0].halo_type, HaloType::Core);
        assert_eq!(promoted[0].confidence, 1.5 + 0.7);
    }
}
