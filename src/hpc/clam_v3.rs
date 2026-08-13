//! V3 distances for [`super::clam`] — read the row's REGIONS, never the whole row.
//!
//! # Why the default is wrong for a V3 row
//!
//! `ClamTree::build` / `build_with_config` route to `hamming_inline`, which
//! Hammings the **whole** point. Correct for flat fingerprints; a category
//! error for a V3 SoA row, which is three regions with three meanings:
//!
//! ```text
//! 512-byte V3 row
//!   0..16    key      classid(4) | facet payload(12)
//!  16..32    edges    EdgeBlock — reading is per-class (edge_codec_flavor)
//!  32..512   value    tenant slab
//! ```
//!
//! **The key is an ADDRESS.** Hamming between addresses clusters on address
//! noise, and it looks plausible because every number is well-formed. That is
//! the same defect as putting a count on the wire as a radius — a value read
//! outside the register that gives it meaning.
//!
//! # The facet register: 6×(u8:u8), axis-paired — NOT 3×u16
//!
//! The 12-byte payload is a content-blind register the classid's ClassView
//! projects. Its rail carving is **six levels of an axis PAIR** —
//! `X:Y` per level, e.g. `part_of:is_a` — two separate bytes, **never
//! widened to u16** (the flat-u24/u16 tail is the retired V1 shape: a widened
//! word has no axis and cannot carry a rail). A byte value is `1 + index`;
//! `0` means *no such level*.
//!
//! So the tree read is: pick ONE axis of the pair, walk the six levels,
//! stop at the first zero. Depth is a count, LCA is a leading-agreement
//! count, and the tree geodesic
//!
//! ```text
//! d(a,b) = depth(a) + depth(b) - 2·depth(lca(a,b))
//! ```
//!
//! costs a handful of byte compares. Position IS the information: byte i of
//! the walked axis is level i, and which bytes are live is a field-mask
//! question, not a parsing question.
//!
//! # The reading comes from the ClassView — so it is a PARAMETER here
//!
//! ndarray is the foundation crate; it cannot (and must not) resolve a
//! classid to its ClassView. What it can do is refuse to hardcode one
//! reading: [`RailSpec`] is the byte-range + axis handle a caller derives
//! from its ClassView / WideFieldMask and passes in. The unconfigured
//! convenience ([`RailSpec::v3_facet`]) is only the zero-fallback default —
//! primary register at `4..16`, axis 0, no continuation.
//!
//! **Stacking:** a class that needs more than six levels does not widen a
//! byte — it stacks a second register, e.g. into the edge lane, and chains
//! it (`RailSpec::stacked`). Depth then runs 0..=12 over two registers, same
//! hole rule, same arithmetic. Continuation is a spec decision made where
//! the ClassView lives, never assumed here.
//!
//! # Honest limits
//!
//! 1. **Bounded depth ⇒ pseudometric.** Past the stored levels, distinct
//!    nodes with an identical walked prefix measure 0 apart. Triangle
//!    inequality holds (CLAM pruning stays sound); identity of
//!    indiscernibles does not.
//! 2. **A DAG is not a tree.** The rail linearises through one chosen
//!    parent; pairs related only through a non-chosen parent read far. The
//!    linearisation question belongs to the bake, not to this function.
//! 3. **No distance over the edge block without its flavor.** `16..32` may
//!    be a continuation register (if the spec says so) — but a *similarity*
//!    over edge bytes needs `edge_codec_flavor` resolved first and is
//!    deliberately absent here.

use super::clam::Distance;

/// V3 key region: `classid(4) | facet payload(12)`.
pub const V3_KEY_LEN: usize = 16;
/// V3 edge block, following the key.
pub const V3_EDGE_LEN: usize = 16;
/// First byte a CONTENT distance may read.
pub const V3_VALUE_OFF: usize = V3_KEY_LEN + V3_EDGE_LEN;
/// Levels per rail register: six (u8:u8) pairs in the 12-byte facet payload.
pub const RAIL_PAIRS: usize = 6;

/// Which byte of each `(u8:u8)` level pair a walk reads.
///
/// The pair is an axis pair (`X:Y`, e.g. `part_of:is_a`) — two SEPARATE
/// bytes. An enum rather than a `u8` so a third axis cannot be conjured by
/// arithmetic: the pair has two lanes, and that is a property of the carving,
/// not a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RailAxis {
    /// Byte 0 of each pair (the `X` of `X:Y`).
    Lo,
    /// Byte 1 of each pair (the `Y` of `X:Y`).
    Hi,
}

/// The byte-range + axis handle for reading a row's rail register(s).
///
/// This is the ClassView's reading made portable: the caller resolves WHICH
/// bytes carry the register and which axis is walked from its
/// ClassView / WideFieldMask, and hands the result here. ndarray never
/// resolves a classid — it only refuses to guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RailSpec {
    /// Byte offset of the primary register (6 pairs = 12 bytes).
    pub reg: usize,
    /// Optional stacked continuation register (6 more pairs), e.g. carved
    /// into the edge lane when the ClassView says so. `None` = 6 levels max.
    pub cont: Option<usize>,
    /// Which byte of each level pair the walk reads.
    pub axis: RailAxis,
}

impl RailSpec {
    /// The zero-fallback default: primary register at `4..16` (the facet
    /// payload), axis `Lo`, no continuation. Correct for a class whose
    /// ClassView has not said otherwise — and ONLY for such a class.
    #[must_use]
    pub const fn v3_facet() -> Self {
        Self { reg: 4, cont: None, axis: RailAxis::Lo }
    }

    /// The same spec with a continuation register stacked at `at` —
    /// e.g. `V3_KEY_LEN` when the ClassView carves it into the edge lane.
    #[must_use]
    pub const fn stacked(self, at: usize) -> Self {
        Self { cont: Some(at), ..self }
    }

    /// Maximum representable depth under this spec.
    #[must_use]
    pub const fn max_depth(&self) -> u32 {
        if self.cont.is_some() { 2 * RAIL_PAIRS as u32 } else { RAIL_PAIRS as u32 }
    }

    /// The walked-axis byte of level `i` (0-based), or 0 if out of range.
    /// Position is the information: byte index IS the level.
    #[inline]
    fn level(&self, row: &[u8], i: usize) -> u8 {
        let (base, k) = if i < RAIL_PAIRS {
            (self.reg, i)
        } else {
            match self.cont {
                Some(c) => (c, i - RAIL_PAIRS),
                None => return 0,
            }
        };
        let at = base + 2 * k + match self.axis { RailAxis::Lo => 0, RailAxis::Hi => 1 };
        if at < row.len() { row[at] } else { 0 }
    }

    /// Depth = occupied leading levels. Stops at the first zero: a hole ends
    /// the chain, and a value after a hole is not ancestry (`[1,0,7]` is
    /// depth 1, never 2).
    #[must_use]
    pub fn depth(&self, row: &[u8]) -> u32 {
        let mut d = 0;
        for i in 0..self.max_depth() as usize {
            if self.level(row, i) == 0 {
                break;
            }
            d += 1;
        }
        d
    }

    /// Depth of the lowest common ancestor: leading levels that are occupied
    /// and equal on the walked axis.
    #[must_use]
    pub fn lca_depth(&self, a: &[u8], b: &[u8]) -> u32 {
        let mut d = 0;
        for i in 0..self.max_depth() as usize {
            let (x, y) = (self.level(a, i), self.level(b, i));
            if x == 0 || x != y {
                break;
            }
            d += 1;
        }
        d
    }

    /// The tree geodesic on the linearisation:
    /// `depth(a) + depth(b) - 2·lca`.
    #[must_use]
    pub fn geodesic(&self, a: &[u8], b: &[u8]) -> u64 {
        let (da, db, l) = (self.depth(a), self.depth(b), self.lca_depth(a, b));
        u64::from(da - l) + u64::from(db - l)
    }
}

/// [`Distance`] over the rail register — the TREE distance, ClassView-shaped.
///
/// `is_metric()` returns `true` because CLAM's pruning only needs the
/// triangle inequality, which the tree geodesic satisfies. Read the module
/// doc before trusting it further: bounded depth makes this a
/// **pseudometric** (distinct deep nodes can measure 0), and on a DAG it
/// measures the linearisation, not the full relation.
#[derive(Debug, Clone, Copy)]
pub struct V3RailGeodesic(pub RailSpec);

impl Distance for V3RailGeodesic {
    type Point = [u8];
    fn distance(&self, a: &[u8], b: &[u8]) -> u64 {
        self.0.geodesic(a, b)
    }
    fn is_metric(&self) -> bool {
        true
    }
}

/// [`Distance`] over the value slab only — the CONTENT distance.
///
/// Reads `32..` and nothing else: two rows with identical tenants and
/// different addresses are distance 0, which is the point — the address is
/// where a thing lives, not what it is.
#[derive(Debug, Clone, Copy, Default)]
pub struct V3ValueHamming;

impl Distance for V3ValueHamming {
    type Point = [u8];
    fn distance(&self, a: &[u8], b: &[u8]) -> u64 {
        v3_value_hamming(a, b)
    }
    fn is_metric(&self) -> bool {
        true
    }
}

/// The content distance as a bare fn, signature-compatible with
/// [`super::clam::DistanceFn`] so it plugs straight into
/// `ClamTree::build_with_fn(rows, 512, …, v3_value_hamming)`.
///
/// A row shorter than the value offset contributes nothing — an honest 0
/// beats a panic in a distance callback, and a truncated row is a loader
/// bug this function cannot repair.
#[must_use]
pub fn v3_value_hamming(a: &[u8], b: &[u8]) -> u64 {
    if a.len() <= V3_VALUE_OFF || b.len() <= V3_VALUE_OFF {
        return 0;
    }
    super::clam::hamming_inline(&a[V3_VALUE_OFF..], &b[V3_VALUE_OFF..])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 512-byte row with the given rail levels written on the chosen axis
    /// of the facet payload, and a value slab filled with `fill`.
    fn row(levels: &[u8], axis: RailAxis, fill: u8) -> Vec<u8> {
        let mut r = vec![0u8; 512];
        for (i, &v) in levels.iter().enumerate().take(RAIL_PAIRS) {
            let at = 4 + 2 * i + match axis { RailAxis::Lo => 0, RailAxis::Hi => 1 };
            r[at] = v;
        }
        for b in &mut r[V3_VALUE_OFF..] {
            *b = fill;
        }
        r
    }

    #[test]
    fn depth_stops_at_the_first_hole() {
        let spec = RailSpec::v3_facet();
        assert_eq!(spec.depth(&row(&[1, 2, 3], RailAxis::Lo, 0)), 3);
        // a value AFTER a hole is not ancestry
        assert_eq!(spec.depth(&row(&[1, 0, 7], RailAxis::Lo, 0)), 1);
        assert_eq!(spec.depth(&row(&[], RailAxis::Lo, 0)), 0);
    }

    #[test]
    fn geodesic_is_the_tree_distance_on_the_linearisation() {
        let spec = RailSpec::v3_facet();
        let parent = row(&[3, 5], RailAxis::Lo, 0);
        let child = row(&[3, 5, 2], RailAxis::Lo, 0);
        let sibling = row(&[3, 5, 4], RailAxis::Lo, 0);
        let stranger = row(&[9], RailAxis::Lo, 0);
        assert_eq!(spec.geodesic(&parent, &child), 1);
        assert_eq!(spec.geodesic(&child, &sibling), 2);
        assert_eq!(spec.geodesic(&child, &child), 0);
        // via the root: depth 3 + depth 1, no shared prefix
        assert_eq!(spec.geodesic(&child, &stranger), 4);
        // symmetry
        assert_eq!(spec.geodesic(&child, &parent), spec.geodesic(&parent, &child));
    }

    /// The pair is TWO SEPARATE BYTES — a walk on `Lo` must not see `Hi`.
    /// This is the test a widened u16 could not pass.
    #[test]
    fn the_axes_of_a_pair_are_independent() {
        let spec_lo = RailSpec::v3_facet();
        let spec_hi = RailSpec { axis: RailAxis::Hi, ..spec_lo };
        let mut r = vec![0u8; 512];
        r[4] = 2; // level 0, Lo
        r[5] = 7; // level 0, Hi — different chain entirely
        r[6] = 3; // level 1, Lo
        assert_eq!(spec_lo.depth(&r), 2);
        assert_eq!(spec_hi.depth(&r), 1);
        let mut s = r.clone();
        s[5] = 9; // change ONLY the Hi axis
        assert_eq!(spec_lo.geodesic(&r, &s), 0, "Lo darf Hi nicht sehen");
        assert!(spec_hi.geodesic(&r, &s) > 0);
    }

    /// Stacking a continuation register (e.g. into the edge lane) extends the
    /// walk past six levels with the same hole rule.
    #[test]
    fn a_stacked_register_extends_depth_past_six() {
        let plain = RailSpec::v3_facet();
        let stacked = plain.stacked(V3_KEY_LEN);
        let mut r = vec![0u8; 512];
        for i in 0..RAIL_PAIRS {
            r[4 + 2 * i] = 1; // primary full
        }
        r[V3_KEY_LEN] = 1; // continuation level 6
        r[V3_KEY_LEN + 2] = 2; // continuation level 7
        assert_eq!(plain.depth(&r), 6, "ohne Spec endet die Welt bei 6");
        assert_eq!(stacked.depth(&r), 8);
        assert_eq!(plain.max_depth(), 6);
        assert_eq!(stacked.max_depth(), 12);
    }

    /// The content distance reads the VALUE and only the value: identical
    /// tenants at different addresses are distance 0. The address is where a
    /// thing lives, not what it is.
    #[test]
    fn value_hamming_never_reads_the_address() {
        let a = row(&[1, 2, 3], RailAxis::Lo, 0xAA);
        let b = row(&[9, 8, 7], RailAxis::Lo, 0xAA);
        assert_eq!(v3_value_hamming(&a, &b), 0);
        let c = row(&[1, 2, 3], RailAxis::Lo, 0xAB); // 1 bit per value byte
        assert_eq!(v3_value_hamming(&a, &c), (512 - V3_VALUE_OFF) as u64);
        // truncated row: honest 0, no panic
        assert_eq!(v3_value_hamming(&a[..16], &b), 0);
    }

    /// Triangle inequality spot-check over a small closed set — the property
    /// CLAM's pruning actually relies on.
    #[test]
    fn the_geodesic_satisfies_the_triangle_inequality() {
        let spec = RailSpec::v3_facet();
        let rows = [
            row(&[1], RailAxis::Lo, 0),
            row(&[1, 2], RailAxis::Lo, 0),
            row(&[1, 2, 3], RailAxis::Lo, 0),
            row(&[1, 4], RailAxis::Lo, 0),
            row(&[6], RailAxis::Lo, 0),
            row(&[], RailAxis::Lo, 0),
        ];
        for a in &rows {
            for b in &rows {
                for c in &rows {
                    assert!(
                        spec.geodesic(a, c) <= spec.geodesic(a, b) + spec.geodesic(b, c),
                        "Dreiecksungleichung verletzt"
                    );
                }
            }
        }
    }

    /// The pseudometric limit, PINNED rather than hidden: distinct nodes
    /// deeper than the spec's reach measure 0 apart. Whoever needs identity
    /// of indiscernibles must refine over stored ancestry, not trust this.
    #[test]
    fn distinct_nodes_past_the_stored_depth_measure_zero() {
        let spec = RailSpec::v3_facet();
        let mut a = vec![0u8; 512];
        let mut b = vec![0u8; 512];
        for i in 0..RAIL_PAIRS {
            a[4 + 2 * i] = 1;
            b[4 + 2 * i] = 1;
        }
        // different VALUES — genuinely distinct nodes, same 6-level prefix
        a[V3_VALUE_OFF] = 0xFF;
        assert_eq!(spec.geodesic(&a, &b), 0, "das ist die dokumentierte Grenze");
        assert!(v3_value_hamming(&a, &b) > 0, "der Inhalt unterscheidet sie");
    }
}
