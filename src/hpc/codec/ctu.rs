//! CTU (Coding Tree Unit) carrier + quad-tree split/merge (PR-X12 A1).
//!
//! One CTU corresponds to one `BlockedGrid` L1 block (64×64 cognitive
//! cells). A CTU may be split recursively into CUs (Coding Units) via a
//! quad-tree: 64×64 → 32×32 → 16×16 → 8×8, depth 0..=3 (4 levels).
//!
//! # Arena allocation
//!
//! The quad-tree is stored in a pre-allocated arena (`CtuArena`) of at
//! most 85 nodes (1 + 4 + 16 + 64 — the closed-form sum for depth 3).
//! Split nodes hold `[NodeIdx; 4]` indices into the arena instead of
//! `Box<...>`-style heap pointers, per PR-X10 invariant 1 (zero-cost
//! abstractions, no `Box<dyn>` in hot paths). This keeps the entire
//! per-CTU tree co-located in one cache-friendly allocation.
//!
//! # Mode taxonomy (x265-shaped)
//!
//! Each leaf CU carries a 2-bit `CellMode`:
//!
//! | Mode   | Bits | Description                                     |
//! |--------|------|-------------------------------------------------|
//! | Skip   | `00` | Cell exactly matches its basin (δ = 0)          |
//! | Merge  | `01` | Inherits δ from N/E/W/S neighbour               |
//! | Delta  | `10` | Owns an 8-bit perturbation (`δ_u8`)             |
//! | Escape | `11` | Falls back to full 64-bit value in escape vector |
//!
//! Per-mode payload fields on `LeafCu` are `Option`-typed: only the
//! field for the current mode is present.
//!
//! # Design reference
//!
//! `.claude/knowledge/pr-x12-codec-x265-design.md` § "Core types"
//! (lines 49-98).

use core::num::NonZeroU16;

// ════════════════════════════════════════════════════════════════════
// Arena capacity bounds
// ════════════════════════════════════════════════════════════════════

/// Maximum quad-tree split depth (inclusive). A CTU has at most this
/// many levels of recursive 4-way splits below the root.
///
/// At depth 3 the leaf CU side is `64 / 2³ = 8` cells.
pub const MAX_SPLIT_DEPTH: u8 = 3;

/// Maximum number of nodes (leaves + interior) across a fully-split
/// CTU quad-tree. Closed-form `1 + 4 + 16 + 64 = 85`.
///
/// Used as the arena pre-allocation capacity in [`CtuArena::new`] so
/// no further heap allocation can happen during split operations.
pub const MAX_QUAD_TREE_NODES: usize = 1 + 4 + 16 + 64;

/// Strongly-typed index into a [`CtuArena`]. Wraps `u16` to keep the
/// arena footprint at 2 bytes per child reference (matches the 85-node
/// cap; `u16` indices stay in `[0, 84]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct NodeIdx(pub u16);

impl NodeIdx {
    /// The root node index — guaranteed to be `NodeIdx(0)` after
    /// [`CtuArena::new`].
    pub const ROOT: Self = NodeIdx(0);
}

// ════════════════════════════════════════════════════════════════════
// Mode taxonomy
// ════════════════════════════════════════════════════════════════════

/// 2-bit mode tag per leaf CU. Bit-pack-friendly: discriminants match
/// the on-wire 2-bit code in the eventual ANS bytestream.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellMode {
    /// Cell exactly matches basin; on-wire 2-bit code `00`.
    Skip = 0b00,
    /// Inherits delta from N/E/W/S neighbour; on-wire code `01`.
    Merge = 0b01,
    /// Carries an 8-bit perturbation; on-wire code `10`.
    Delta = 0b10,
    /// Falls back to 64-bit escape vector; on-wire code `11`.
    Escape = 0b11,
}

/// 2-bit cardinal direction for [`CellMode::Merge`]. Bit-packed on the
/// wire alongside the mode tag.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MergeDir {
    /// On-wire 2-bit code `00`.
    North = 0,
    /// On-wire 2-bit code `01`.
    East = 1,
    /// On-wire 2-bit code `10`.
    West = 2,
    /// On-wire 2-bit code `11`.
    South = 3,
}

// ════════════════════════════════════════════════════════════════════
// Leaf payload
// ════════════════════════════════════════════════════════════════════

/// Payload at a leaf CU. Per-mode `Option` fields keep each leaf at
/// its semantic minimum — only the field for the current `mode` is
/// present.
///
/// On the wire (PR-X12 A7 rANS encoder, not in this PR) the
/// `Option`-tagged dynamic shape collapses to a fixed per-mode bit
/// budget; the in-memory representation here optimises for clarity
/// over storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LeafCu {
    /// Mode discriminant; selects which of the optional payload fields
    /// below is present.
    pub mode: CellMode,
    /// 12-bit codebook index into the per-frame basin table. High 4
    /// bits reserved for the encoder (see design doc § "ANS"). Always
    /// present, including in `Escape` mode (the basin ref the escape
    /// is relative to).
    pub basin_idx: u16,
    /// Present iff `mode == CellMode::Delta`. 8-bit perturbation
    /// applied to `basin_idx` to recover the cell.
    pub delta: Option<u8>,
    /// Present iff `mode == CellMode::Merge`. Direction of the
    /// neighbour whose delta is inherited.
    pub merge_dir: Option<MergeDir>,
    /// Present iff `mode == CellMode::Escape`. Index into the
    /// per-frame escape vector holding the full 64-bit cell value.
    pub escape_idx: Option<u32>,
}

impl LeafCu {
    /// Construct a `Skip`-mode leaf. The 4 mode-payload fields are all
    /// `None` since `Skip` carries no per-cell delta.
    pub fn skip(basin_idx: u16) -> Self {
        Self {
            mode: CellMode::Skip,
            basin_idx,
            delta: None,
            merge_dir: None,
            escape_idx: None,
        }
    }

    /// Construct a `Delta`-mode leaf carrying an 8-bit perturbation.
    pub fn delta(basin_idx: u16, delta: u8) -> Self {
        Self {
            mode: CellMode::Delta,
            basin_idx,
            delta: Some(delta),
            merge_dir: None,
            escape_idx: None,
        }
    }

    /// Construct a `Merge`-mode leaf inheriting the delta from `dir`.
    pub fn merge(basin_idx: u16, dir: MergeDir) -> Self {
        Self {
            mode: CellMode::Merge,
            basin_idx,
            delta: None,
            merge_dir: Some(dir),
            escape_idx: None,
        }
    }

    /// Construct an `Escape`-mode leaf referring to a per-frame escape
    /// vector entry.
    pub fn escape(basin_idx: u16, escape_idx: u32) -> Self {
        Self {
            mode: CellMode::Escape,
            basin_idx,
            delta: None,
            merge_dir: None,
            escape_idx: Some(escape_idx),
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Quad-tree partition node
// ════════════════════════════════════════════════════════════════════

/// One node in the CTU quad-tree. Either a leaf carrying a [`LeafCu`]
/// payload, or a 4-way split whose children live at the indicated
/// [`NodeIdx`]es in the same [`CtuArena`].
///
/// The variants intentionally do not box children — see the module
/// header for the arena rationale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CtuPartition {
    /// Leaf CU — terminal node carrying a [`LeafCu`].
    Leaf(LeafCu),
    /// 4-way split — four child indices into the arena. By convention
    /// the order is `[NW, NE, SW, SE]` (Z-order).
    Split([NodeIdx; 4]),
}

// ════════════════════════════════════════════════════════════════════
// Arena
// ════════════════════════════════════════════════════════════════════

/// Pre-allocated arena holding all quad-tree nodes for one CTU.
///
/// Capacity is fixed at construction time at [`MAX_QUAD_TREE_NODES`]
/// (= 85), so no further heap allocation can happen during split /
/// merge operations. All node references inside the arena go through
/// [`NodeIdx`] (a `u16` index), not raw pointers.
#[derive(Debug, Clone)]
pub struct CtuArena {
    nodes: Vec<CtuPartition>,
}

impl CtuArena {
    /// Construct a fresh arena pre-allocated to [`MAX_QUAD_TREE_NODES`].
    /// The root leaf must be inserted via [`Self::push`] before the
    /// arena is meaningful; [`Ctu::new_skip`] does this for the common
    /// "fresh skip-mode CTU" entry point.
    pub fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(MAX_QUAD_TREE_NODES),
        }
    }

    /// Insert a node and return its [`NodeIdx`].
    ///
    /// # Panics
    ///
    /// Panics if the arena is at [`MAX_QUAD_TREE_NODES`] capacity. This
    /// is a structural invariant — depth ≤ 3 and 4-way branching can
    /// never produce more than 85 nodes — so a panic here indicates a
    /// caller bug, not a runtime resource shortage.
    pub fn push(&mut self, node: CtuPartition) -> NodeIdx {
        assert!(
            self.nodes.len() < MAX_QUAD_TREE_NODES,
            "CtuArena: structural cap of {} nodes exceeded",
            MAX_QUAD_TREE_NODES
        );
        let idx = NodeIdx(self.nodes.len() as u16);
        self.nodes.push(node);
        idx
    }

    /// Borrow the node at `idx`.
    pub fn get(&self, idx: NodeIdx) -> &CtuPartition {
        &self.nodes[idx.0 as usize]
    }

    /// Mutably borrow the node at `idx`.
    pub fn get_mut(&mut self, idx: NodeIdx) -> &mut CtuPartition {
        &mut self.nodes[idx.0 as usize]
    }

    /// Number of nodes currently in the arena (≤ [`MAX_QUAD_TREE_NODES`]).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the arena holds no nodes (no root inserted yet).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl Default for CtuArena {
    fn default() -> Self {
        Self::new()
    }
}

// ════════════════════════════════════════════════════════════════════
// CTU
// ════════════════════════════════════════════════════════════════════

/// One CTU = one `BlockedGrid` L1 block (64×64 cognitive cells).
///
/// Partitionable into CUs via the quad-tree rooted at
/// `arena.get(NodeIdx::ROOT)`. The split depth at the root is tracked
/// implicitly by recursive traversal; `split_depth` on the `Ctu`
/// struct is the **maximum** depth observed in the current tree (used
/// to short-circuit the depth check in [`Ctu::split`]).
#[derive(Debug, Clone)]
pub struct Ctu {
    /// Position of this CTU in the parent BlockedGrid (block-row).
    pub block_row: u16,
    /// Position of this CTU in the parent BlockedGrid (block-col).
    pub block_col: u16,
    /// Cascade tier this CTU belongs to (1..=4, matches splat4d).
    pub tier: NonZeroU16,
    /// Maximum quad-tree depth currently realised in the arena
    /// (0..=[`MAX_SPLIT_DEPTH`]).
    pub split_depth: u8,
    /// Backing arena holding every node in this CTU's quad-tree.
    pub arena: CtuArena,
}

impl Ctu {
    /// Construct a fresh CTU containing a single `Skip`-mode root leaf.
    ///
    /// # Panics
    ///
    /// Panics if `tier` is outside `1..=4`. The cascade tiers are 1-indexed
    /// L1..L4; values ≥ 5 are rejected because they would let invalid tier
    /// metadata enter the codec state (P2 codex review on PR #170 line 306).
    pub fn new_skip(block_row: u16, block_col: u16, tier: u8, basin_idx: u16) -> Self {
        assert!((1..=4).contains(&tier), "Ctu::new_skip: tier must be in 1..=4 (got {tier})");
        let tier = NonZeroU16::new(tier as u16).expect("post-assert tier > 0");
        let mut arena = CtuArena::new();
        arena.push(CtuPartition::Leaf(LeafCu::skip(basin_idx)));
        Self {
            block_row,
            block_col,
            tier,
            split_depth: 0,
            arena,
        }
    }

    /// Walk the arena from root to find the depth at which `target` lives.
    ///
    /// Returns `None` if `target` is not reachable from `NodeIdx::ROOT`
    /// (e.g. orphaned by a prior `merge` — see the GC note on `merge`).
    /// Used by `split` to enforce `MAX_SPLIT_DEPTH` against the actual
    /// tree state rather than a caller-supplied claim (P2 codex review
    /// on PR #170 line 333).
    ///
    /// Complexity: O(N) where N ≤ [`MAX_QUAD_TREE_NODES`] = 85.
    pub fn depth_of(&self, target: NodeIdx) -> Option<u8> {
        if target == NodeIdx::ROOT {
            return Some(0);
        }
        let mut frontier: Vec<(NodeIdx, u8)> = vec![(NodeIdx::ROOT, 0)];
        while let Some((node, d)) = frontier.pop() {
            if let CtuPartition::Split(children) = self.arena.get(node) {
                for &c in children {
                    if c == target {
                        return Some(d + 1);
                    }
                    frontier.push((c, d + 1));
                }
            }
        }
        None
    }

    /// Split the leaf at `node_idx` into four child leaves, all inheriting
    /// the parent's basin (and `Skip` mode by default).
    ///
    /// Returns the new `[NW, NE, SW, SE]` child indices on success.
    ///
    /// # Depth limit
    ///
    /// The depth of `node_idx` is computed from the tree itself via
    /// [`Self::depth_of`], not from caller input — passing a stale claim
    /// can no longer trigger the arena overflow `assert!` (P2 codex
    /// review on PR #170 line 333). Splits past [`MAX_SPLIT_DEPTH`]
    /// return [`SplitError::MaxSplitDepthReached`]; targets unreachable
    /// from `NodeIdx::ROOT` (e.g. orphaned post-merge) return
    /// [`SplitError::NodeNotReachable`].
    pub fn split(&mut self, node_idx: NodeIdx) -> Result<[NodeIdx; 4], SplitError> {
        let current_depth = self
            .depth_of(node_idx)
            .ok_or(SplitError::NodeNotReachable)?;
        if current_depth >= MAX_SPLIT_DEPTH {
            return Err(SplitError::MaxSplitDepthReached(MaxSplitDepthReached {
                depth: current_depth,
                cap: MAX_SPLIT_DEPTH,
            }));
        }
        // Snapshot the parent leaf so the children can inherit its
        // basin_idx; bail if the target node is already a Split.
        let parent_basin = match self.arena.get(node_idx) {
            CtuPartition::Leaf(leaf) => leaf.basin_idx,
            CtuPartition::Split(_) => return Err(SplitError::NotALeaf),
        };
        // Allocate four child leaves first, then patch the parent in
        // place. Doing the patch last keeps the arena valid even if a
        // `push` panicked midway (which it won't given the structural
        // cap, but the order makes the invariant explicit).
        let nw = self
            .arena
            .push(CtuPartition::Leaf(LeafCu::skip(parent_basin)));
        let ne = self
            .arena
            .push(CtuPartition::Leaf(LeafCu::skip(parent_basin)));
        let sw = self
            .arena
            .push(CtuPartition::Leaf(LeafCu::skip(parent_basin)));
        let se = self
            .arena
            .push(CtuPartition::Leaf(LeafCu::skip(parent_basin)));
        let children = [nw, ne, sw, se];
        *self.arena.get_mut(node_idx) = CtuPartition::Split(children);
        let new_depth = current_depth + 1;
        if new_depth > self.split_depth {
            self.split_depth = new_depth;
        }
        Ok(children)
    }

    /// Merge a 4-way split back into a single leaf, IF all four children
    /// are themselves leaves with **identical `LeafCu` payloads** (same
    /// `mode`, `basin_idx`, **and** per-mode payload — `delta`,
    /// `merge_dir`, or `escape_idx` as appropriate).
    ///
    /// The merged leaf takes the (now-unique) `LeafCu` of the children.
    /// Heterogeneous children (any field differs) are rejected with
    /// [`MergeError::ChildrenDiverge`]; non-leaf children are rejected
    /// with [`MergeError::ChildNotLeaf`].
    ///
    /// The full-payload equality is the P1 codex fix on PR #170 line
    /// 393: the prior implementation compared only `mode` + `basin_idx`,
    /// which silently dropped per-mode payload when the four children
    /// carried different `delta` / `merge_dir` / `escape_idx` values.
    ///
    /// Note: this method does NOT compact the arena — the child nodes
    /// remain allocated and orphaned. A full GC pass (out of scope for
    /// A1) would compact between encode passes.
    pub fn merge(&mut self, node_idx: NodeIdx) -> Result<(), MergeError> {
        let children = match self.arena.get(node_idx) {
            CtuPartition::Split(c) => *c,
            CtuPartition::Leaf(_) => return Err(MergeError::NotASplit),
        };
        let mut merged: Option<LeafCu> = None;
        for &child_idx in &children {
            let leaf = match self.arena.get(child_idx) {
                CtuPartition::Leaf(l) => *l,
                CtuPartition::Split(_) => return Err(MergeError::ChildNotLeaf),
            };
            match merged {
                None => merged = Some(leaf),
                Some(prev) => {
                    // Full LeafCu equality (mode + basin_idx + per-mode
                    // payload). `PartialEq` on LeafCu compares every
                    // field; any divergence in delta / merge_dir /
                    // escape_idx therefore surfaces as ChildrenDiverge.
                    if prev != leaf {
                        return Err(MergeError::ChildrenDiverge);
                    }
                }
            }
        }
        // SAFETY of unwrap: `children` has 4 entries, so the loop ran at
        // least once and merged is `Some`. Compiler can't prove that
        // statically; the unwrap is the explicit assertion.
        let unified = merged.expect("4-child loop guarantees Some");
        *self.arena.get_mut(node_idx) = CtuPartition::Leaf(unified);
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════
// Errors
// ════════════════════════════════════════════════════════════════════

/// Reasons a [`Ctu::split`] call may fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitError {
    /// The target node was already a 4-way split, not a leaf. Splits
    /// only operate on leaves; recurse into the children to split deeper.
    NotALeaf,
    /// The target node is not reachable from `NodeIdx::ROOT` (e.g. an
    /// orphaned node left behind by a prior `merge`). Returned by
    /// [`Ctu::split`] when [`Ctu::depth_of`] can't locate the target.
    NodeNotReachable,
    /// The split would push the quad-tree past [`MAX_SPLIT_DEPTH`].
    MaxSplitDepthReached(MaxSplitDepthReached),
}

/// Carries the depth at which the split was attempted and the cap.
/// Used by callers to report "depth 4 exceeds cap 3" without parsing
/// strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxSplitDepthReached {
    /// Depth at which the split was attempted.
    pub depth: u8,
    /// Hard cap ([`MAX_SPLIT_DEPTH`]).
    pub cap: u8,
}

/// Reasons a [`Ctu::merge`] call may fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeError {
    /// The target node was a leaf, not a split. Merging only operates
    /// on 4-way split nodes.
    NotASplit,
    /// One of the four children was itself a split. Merge can only
    /// collapse a layer of leaves; nested splits must be merged first.
    ChildNotLeaf,
    /// The four children disagree on `mode` and/or `basin_idx` —
    /// collapsing them into one leaf would lose information.
    ChildrenDiverge,
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_skip_creates_root_leaf() {
        let ctu = Ctu::new_skip(2, 3, 1, 42);
        assert_eq!(ctu.block_row, 2);
        assert_eq!(ctu.block_col, 3);
        assert_eq!(ctu.tier.get(), 1);
        assert_eq!(ctu.split_depth, 0);
        assert_eq!(ctu.arena.len(), 1);
        match ctu.arena.get(NodeIdx::ROOT) {
            CtuPartition::Leaf(leaf) => {
                assert_eq!(leaf.mode, CellMode::Skip);
                assert_eq!(leaf.basin_idx, 42);
                assert!(leaf.delta.is_none());
                assert!(leaf.merge_dir.is_none());
                assert!(leaf.escape_idx.is_none());
            }
            CtuPartition::Split(_) => panic!("root should be leaf after new_skip"),
        }
    }

    #[test]
    fn split_root_yields_four_children() {
        let mut ctu = Ctu::new_skip(0, 0, 1, 7);
        let children = ctu.split(NodeIdx::ROOT).expect("split should succeed");
        assert_eq!(children.len(), 4);
        assert_eq!(ctu.split_depth, 1);
        assert_eq!(ctu.arena.len(), 5); // root + 4 children
        for &c in &children {
            match ctu.arena.get(c) {
                CtuPartition::Leaf(leaf) => {
                    assert_eq!(leaf.mode, CellMode::Skip);
                    assert_eq!(leaf.basin_idx, 7, "children inherit parent basin");
                }
                CtuPartition::Split(_) => panic!("fresh child should be leaf"),
            }
        }
        // Root is now a Split
        match ctu.arena.get(NodeIdx::ROOT) {
            CtuPartition::Split(c) => assert_eq!(*c, children),
            CtuPartition::Leaf(_) => panic!("root should now be Split"),
        }
    }

    #[test]
    fn split_at_max_depth_rejects() {
        // Navigate the tree down to a depth-3 leaf, then try to split it.
        // Depth is computed from the tree (P2 codex fix), so callers can't
        // bypass the cap with a stale depth claim.
        let mut ctu = Ctu::new_skip(0, 0, 1, 0);
        let l1 = ctu.split(NodeIdx::ROOT).unwrap()[0];
        let l2 = ctu.split(l1).unwrap()[0];
        let l3 = ctu.split(l2).unwrap()[0];
        assert_eq!(ctu.depth_of(l3), Some(3));
        let err = ctu.split(l3).unwrap_err();
        match err {
            SplitError::MaxSplitDepthReached(info) => {
                assert_eq!(info.depth, MAX_SPLIT_DEPTH);
                assert_eq!(info.cap, MAX_SPLIT_DEPTH);
            }
            _ => panic!("expected MaxSplitDepthReached, got {err:?}"),
        }
        assert_eq!(ctu.split_depth, MAX_SPLIT_DEPTH);
    }

    #[test]
    fn split_unreachable_node_rejects() {
        // A node still inside the arena but not reachable from root (e.g.
        // orphaned post-merge) yields NodeNotReachable instead of panicking
        // on the arena overflow. P2 codex fix.
        let mut ctu = Ctu::new_skip(0, 0, 1, 0);
        let children = ctu.split(NodeIdx::ROOT).unwrap();
        // Merge collapses root back to a leaf — the four child entries
        // remain in the arena but are no longer linked from root.
        ctu.merge(NodeIdx::ROOT).unwrap();
        let orphan = children[0];
        let err = ctu.split(orphan).unwrap_err();
        assert_eq!(err, SplitError::NodeNotReachable);
    }

    #[test]
    fn split_already_split_node_rejects() {
        let mut ctu = Ctu::new_skip(0, 0, 1, 0);
        ctu.split(NodeIdx::ROOT).unwrap();
        let err = ctu.split(NodeIdx::ROOT).unwrap_err();
        assert_eq!(err, SplitError::NotALeaf);
    }

    #[test]
    fn merge_homogeneous_children_collapses() {
        let mut ctu = Ctu::new_skip(0, 0, 1, 13);
        ctu.split(NodeIdx::ROOT).unwrap();
        ctu.merge(NodeIdx::ROOT).expect("homogeneous merge ok");
        match ctu.arena.get(NodeIdx::ROOT) {
            CtuPartition::Leaf(leaf) => {
                assert_eq!(leaf.mode, CellMode::Skip);
                assert_eq!(leaf.basin_idx, 13);
            }
            CtuPartition::Split(_) => panic!("root should be merged back to leaf"),
        }
    }

    #[test]
    fn merge_heterogeneous_children_rejects() {
        let mut ctu = Ctu::new_skip(0, 0, 1, 5);
        let children = ctu.split(NodeIdx::ROOT).unwrap();
        // Swap one child to a different mode
        *ctu.arena.get_mut(children[2]) = CtuPartition::Leaf(LeafCu::delta(5, 7));
        let err = ctu.merge(NodeIdx::ROOT).unwrap_err();
        assert_eq!(err, MergeError::ChildrenDiverge);
    }

    #[test]
    fn merge_split_child_rejects() {
        let mut ctu = Ctu::new_skip(0, 0, 1, 0);
        let children = ctu.split(NodeIdx::ROOT).unwrap();
        // Re-split one of the children
        ctu.split(children[0]).unwrap();
        let err = ctu.merge(NodeIdx::ROOT).unwrap_err();
        assert_eq!(err, MergeError::ChildNotLeaf);
    }

    #[test]
    fn merge_leaf_rejects() {
        let mut ctu = Ctu::new_skip(0, 0, 1, 0);
        let err = ctu.merge(NodeIdx::ROOT).unwrap_err();
        assert_eq!(err, MergeError::NotASplit);
    }

    #[test]
    fn leaf_constructors_set_correct_fields() {
        let skip = LeafCu::skip(11);
        assert_eq!(skip.mode, CellMode::Skip);
        assert_eq!(skip.basin_idx, 11);
        assert!(skip.delta.is_none() && skip.merge_dir.is_none() && skip.escape_idx.is_none());

        let delta = LeafCu::delta(22, 0xAB);
        assert_eq!(delta.mode, CellMode::Delta);
        assert_eq!(delta.delta, Some(0xAB));
        assert!(delta.merge_dir.is_none() && delta.escape_idx.is_none());

        let merge = LeafCu::merge(33, MergeDir::East);
        assert_eq!(merge.mode, CellMode::Merge);
        assert_eq!(merge.merge_dir, Some(MergeDir::East));
        assert!(merge.delta.is_none() && merge.escape_idx.is_none());

        let escape = LeafCu::escape(44, 0x1234_5678);
        assert_eq!(escape.mode, CellMode::Escape);
        assert_eq!(escape.escape_idx, Some(0x1234_5678));
        assert!(escape.delta.is_none() && escape.merge_dir.is_none());
    }

    #[test]
    fn arena_capacity_bound_85() {
        // Fully split to depth 3 and verify total node count = 85.
        // depth 0 → 1 node; +4 = 5 (d1); +16 = 21 (d2); +64 = 85 (d3).
        // depth_of is consulted internally per split, so this exercises
        // both the depth-cap check and the arena cap together.
        let mut ctu = Ctu::new_skip(0, 0, 1, 0);
        fn recursive_split(ctu: &mut Ctu, node: NodeIdx) {
            let d = ctu.depth_of(node).expect("reachable");
            if d >= MAX_SPLIT_DEPTH {
                return;
            }
            let children = ctu.split(node).expect("split ok");
            for &c in &children {
                recursive_split(ctu, c);
            }
        }
        recursive_split(&mut ctu, NodeIdx::ROOT);
        assert_eq!(ctu.arena.len(), MAX_QUAD_TREE_NODES);
        assert_eq!(ctu.split_depth, MAX_SPLIT_DEPTH);
    }

    #[test]
    #[should_panic(expected = "tier must be in 1..=4")]
    fn new_skip_rejects_tier_5() {
        // P2 codex fix: tier > 4 panics at construction instead of
        // silently entering the codec state. Pinning the message text
        // ensures the assert remains explicit.
        let _ = Ctu::new_skip(0, 0, 5, 0);
    }

    #[test]
    #[should_panic(expected = "tier must be in 1..=4")]
    fn new_skip_rejects_tier_0() {
        let _ = Ctu::new_skip(0, 0, 0, 0);
    }

    #[test]
    fn merge_diverging_delta_payloads_rejects() {
        // P1 codex fix: prior implementation compared only mode +
        // basin_idx. Now Delta children with different δ values are
        // rejected; the previously-silent payload loss is gone.
        let mut ctu = Ctu::new_skip(0, 0, 1, 9);
        let children = ctu.split(NodeIdx::ROOT).unwrap();
        // Same mode (Delta) and basin_idx, but different δ values.
        *ctu.arena.get_mut(children[0]) = CtuPartition::Leaf(LeafCu::delta(9, 0x11));
        *ctu.arena.get_mut(children[1]) = CtuPartition::Leaf(LeafCu::delta(9, 0x22));
        *ctu.arena.get_mut(children[2]) = CtuPartition::Leaf(LeafCu::delta(9, 0x33));
        *ctu.arena.get_mut(children[3]) = CtuPartition::Leaf(LeafCu::delta(9, 0x44));
        let err = ctu.merge(NodeIdx::ROOT).unwrap_err();
        assert_eq!(err, MergeError::ChildrenDiverge);
    }

    #[test]
    fn merge_diverging_merge_dirs_rejects() {
        let mut ctu = Ctu::new_skip(0, 0, 1, 5);
        let children = ctu.split(NodeIdx::ROOT).unwrap();
        *ctu.arena.get_mut(children[0]) = CtuPartition::Leaf(LeafCu::merge(5, MergeDir::North));
        *ctu.arena.get_mut(children[1]) = CtuPartition::Leaf(LeafCu::merge(5, MergeDir::East));
        *ctu.arena.get_mut(children[2]) = CtuPartition::Leaf(LeafCu::merge(5, MergeDir::West));
        *ctu.arena.get_mut(children[3]) = CtuPartition::Leaf(LeafCu::merge(5, MergeDir::South));
        assert_eq!(ctu.merge(NodeIdx::ROOT).unwrap_err(), MergeError::ChildrenDiverge);
    }

    #[test]
    fn merge_diverging_escape_idx_rejects() {
        let mut ctu = Ctu::new_skip(0, 0, 1, 1);
        let children = ctu.split(NodeIdx::ROOT).unwrap();
        *ctu.arena.get_mut(children[0]) = CtuPartition::Leaf(LeafCu::escape(1, 100));
        *ctu.arena.get_mut(children[1]) = CtuPartition::Leaf(LeafCu::escape(1, 101));
        *ctu.arena.get_mut(children[2]) = CtuPartition::Leaf(LeafCu::escape(1, 102));
        *ctu.arena.get_mut(children[3]) = CtuPartition::Leaf(LeafCu::escape(1, 103));
        assert_eq!(ctu.merge(NodeIdx::ROOT).unwrap_err(), MergeError::ChildrenDiverge);
    }

    #[test]
    fn merge_identical_delta_payloads_collapses() {
        // Symmetric to the divergence tests above: identical Delta
        // children DO merge, preserving the unified payload.
        let mut ctu = Ctu::new_skip(0, 0, 1, 3);
        let children = ctu.split(NodeIdx::ROOT).unwrap();
        for &c in &children {
            *ctu.arena.get_mut(c) = CtuPartition::Leaf(LeafCu::delta(3, 0x77));
        }
        ctu.merge(NodeIdx::ROOT).expect("identical-delta merge ok");
        match ctu.arena.get(NodeIdx::ROOT) {
            CtuPartition::Leaf(leaf) => {
                assert_eq!(leaf.mode, CellMode::Delta);
                assert_eq!(leaf.basin_idx, 3);
                assert_eq!(leaf.delta, Some(0x77));
            }
            CtuPartition::Split(_) => panic!("should be merged"),
        }
    }

    #[test]
    fn cell_mode_discriminants_match_wire_codes() {
        assert_eq!(CellMode::Skip as u8, 0b00);
        assert_eq!(CellMode::Merge as u8, 0b01);
        assert_eq!(CellMode::Delta as u8, 0b10);
        assert_eq!(CellMode::Escape as u8, 0b11);
    }

    #[test]
    fn merge_dir_discriminants_match_wire_codes() {
        assert_eq!(MergeDir::North as u8, 0);
        assert_eq!(MergeDir::East as u8, 1);
        assert_eq!(MergeDir::West as u8, 2);
        assert_eq!(MergeDir::South as u8, 3);
    }

    #[test]
    fn node_idx_root_is_zero() {
        assert_eq!(NodeIdx::ROOT, NodeIdx(0));
    }
}
