//! `hlod` (group C) — HLOD (Hierarchical LOD) tree traversal with refine
//! `ADD` vs `REPLACE` semantics, gated by SSE.
//!
//! Implements the OGC 3D Tiles 1.1 tile refinement logic used by Cesium to
//! walk the tile tree and decide which tiles to render at any camera position.
//!
//! # Refinement modes (grounded against OGC 3D Tiles 1.1 §7.5)
//!
//! | Mode      | Meaning                                                       |
//! |-----------|---------------------------------------------------------------|
//! | `ADD`     | Child tiles are *added to* the parent's geometry.  The parent |
//! |           | is always rendered; children supplement it when refined.      |
//! | `REPLACE` | Child tiles *replace* the parent.  When refined, the parent   |
//! |           | is NOT rendered; children take over completely.               |
//!
//! # Traversal algorithm (depth-first, greedy)
//!
//! ```text
//! fn traverse(tile, camera, frustum, max_sse, result):
//!     sse = sse_for_tile(tile.geometric_error, dist(camera, tile), frustum)
//!     if sse <= max_sse OR tile is leaf:
//!         // This tile's LOD is sufficient — add to render list.
//!         result.push(tile.content_uri)
//!         return
//!     // SSE too large — must refine.
//!     match tile.refine:
//!         REPLACE:
//!             // Do NOT render the parent — recurse into children only.
//!             for child in tile.children: traverse(child, ...)
//!         ADD:
//!             // Render the parent AND recurse into children.
//!             result.push(tile.content_uri)
//!             for child in tile.children: traverse(child, ...)
//! ```
//!
//! # UNVERIFIED items
//! - Whether Cesium performs frustum culling before or after SSE testing.
//!   (Likely before, but the exact pass order in `Cesium3DTileset._update` is
//!   not confirmed here.)
//! - Whether `geometricError == 0` on leaf tiles is guaranteed by the spec or
//!   just a convention.
//! - Exact behaviour when `tile.children` is empty but `sse > max_sse`
//!   (spec says treat as leaf, but Cesium may log a warning).
//!
//! # Commented scaffold only
//! All impl is `//`-commented until reviewed live by Opus + CodeRabbit.
//! Compilable stub types and unit tests are live (std-only).

// ─────────────────────────────────────────────────────────────────────────────
// Compilable stub types (std-only)
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum recursion depth for [`traverse_tile`].
///
/// 3D-Tiles tile trees are typically very shallow (< 20 levels in practice).
/// This cap prevents infinite loops on cyclic child-index graphs produced by
/// malformed or adversarial input.
const MAX_HLOD_DEPTH: usize = 64;

/// Tile refinement strategy (OGC 3D Tiles 1.1 §7.5).
///
/// # Examples
///
/// ```
/// use cesium::hlod::Refine;
///
/// let r = Refine::Replace;
/// assert_ne!(r, Refine::Add);
///
/// // Refine is Copy — can be used after a move.
/// let r2 = r;
/// assert_eq!(r, r2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refine {
    /// Children ADD to parent — parent is always rendered when visible.
    Add,
    /// Children REPLACE parent — parent is NOT rendered when refined.
    Replace,
}

/// A node in the HLOD tile tree (stub representation).
///
/// In a live implementation this would reference the `tileset.json` content
/// and bounding-volume structures; here all geometry is reduced to the minimum
/// information needed for traversal decisions.
///
/// # Examples
///
/// ```
/// use cesium::hlod::{HlodTile, Refine};
///
/// let tile = HlodTile {
///     geometric_error: 100.0,
///     bounding_sphere_center: [0.0, 0.0, 0.0],
///     bounding_sphere_radius: 10.0,
///     refine: Refine::Replace,
///     content_uri: Some("root.glb".into()),
///     children: vec![],
/// };
/// assert!(tile.is_leaf());
/// ```
#[derive(Clone, Debug)]
pub struct HlodTile {
    /// Tile's `geometricError` from `tileset.json` (world units).
    pub geometric_error: f32,
    /// World-space centre of the tile's bounding sphere.
    pub bounding_sphere_center: [f32; 3],
    /// Radius of the tile's bounding sphere (world units).
    pub bounding_sphere_radius: f32,
    /// Refinement strategy for this tile.
    pub refine: Refine,
    /// URI of this tile's content (e.g. `"0/0/0.glb"`).
    /// `None` for implicit/empty tiles that have no renderable content.
    pub content_uri: Option<String>,
    /// Child tile indices (into the same flat `HlodTree::tiles` Vec).
    pub children: Vec<usize>,
}

impl HlodTile {
    /// True if this tile has no children (it is a leaf by structure).
    ///
    /// # Examples
    ///
    /// ```
    /// use cesium::hlod::{HlodTile, Refine};
    ///
    /// let leaf = HlodTile {
    ///     geometric_error: 0.0,
    ///     bounding_sphere_center: [1.0, 2.0, 3.0],
    ///     bounding_sphere_radius: 5.0,
    ///     refine: Refine::Replace,
    ///     content_uri: Some("leaf.glb".into()),
    ///     children: vec![],
    /// };
    /// assert!(leaf.is_leaf());
    ///
    /// let parent = HlodTile { children: vec![1], ..leaf.clone() };
    /// assert!(!parent.is_leaf());
    /// ```
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// Approximate distance from `camera` to the bounding-sphere surface.
    ///
    /// Returns 0.0 if the camera is inside the sphere.
    ///
    /// # Examples
    ///
    /// ```
    /// use cesium::hlod::{HlodTile, Refine};
    ///
    /// let tile = HlodTile {
    ///     geometric_error: 0.0,
    ///     bounding_sphere_center: [0.0, 0.0, 0.0],
    ///     bounding_sphere_radius: 1.0,
    ///     refine: Refine::Replace,
    ///     content_uri: None,
    ///     children: vec![],
    /// };
    ///
    /// // Camera on the surface — distance is 0.
    /// assert_eq!(tile.distance_to_camera([1.0, 0.0, 0.0]), 0.0);
    ///
    /// // Camera 5 units away from centre, radius 1 → surface distance = 4.
    /// let d = tile.distance_to_camera([5.0, 0.0, 0.0]);
    /// assert!((d - 4.0).abs() < 1e-5);
    /// ```
    pub fn distance_to_camera(&self, camera: [f32; 3]) -> f32 {
        let dx = camera[0] - self.bounding_sphere_center[0];
        let dy = camera[1] - self.bounding_sphere_center[1];
        let dz = camera[2] - self.bounding_sphere_center[2];
        let dist_center = (dx * dx + dy * dy + dz * dz).sqrt();
        (dist_center - self.bounding_sphere_radius).max(0.0)
    }
}

/// A flat HLOD tile tree (arena-allocated for cache-friendly traversal).
///
/// # Arena layout and root invariant
///
/// Tiles are stored in a plain `Vec` and referenced by index.  **Index 0 is
/// the root by construction.**  [`push`](HlodTree::push) returns the arena
/// index of each newly inserted tile; callers MUST push the root tile first
/// (before any child tiles) so that the root occupies index 0.  Violating
/// this invariant causes [`traverse_hlod`] to start at the wrong tile.
///
/// # Examples
///
/// ```
/// use cesium::hlod::{HlodTile, HlodTree, Refine};
///
/// let mut tree = HlodTree::new();
///
/// // Push the root first — it gets index 0.
/// let root_idx = tree.push(HlodTile {
///     geometric_error: 50.0,
///     bounding_sphere_center: [0.0, 0.0, 0.0],
///     bounding_sphere_radius: 20.0,
///     refine: Refine::Replace,
///     content_uri: Some("root.glb".into()),
///     children: vec![],   // no children yet
/// });
/// assert_eq!(root_idx, 0);
/// assert_eq!(tree.tiles.len(), 1);
/// ```
#[derive(Clone, Debug, Default)]
pub struct HlodTree {
    /// All tiles in the tree.  Root is at index 0.
    pub tiles: Vec<HlodTile>,
}

impl HlodTree {
    /// Construct an empty tree.
    ///
    /// # Examples
    ///
    /// ```
    /// use cesium::hlod::HlodTree;
    ///
    /// let tree = HlodTree::new();
    /// assert!(tree.tiles.is_empty());
    /// ```
    pub fn new() -> Self {
        Self { tiles: Vec::new() }
    }

    /// Push a tile and return its index.
    ///
    /// **Callers must push the root tile first** (index 0) before any child
    /// tiles; see the [`HlodTree`] type documentation for the root invariant.
    ///
    /// # Examples
    ///
    /// ```
    /// use cesium::hlod::{HlodTile, HlodTree, Refine};
    ///
    /// let mut tree = HlodTree::new();
    /// let idx = tree.push(HlodTile {
    ///     geometric_error: 0.0,
    ///     bounding_sphere_center: [0.0, 0.0, 0.0],
    ///     bounding_sphere_radius: 1.0,
    ///     refine: Refine::Replace,
    ///     content_uri: Some("leaf.glb".into()),
    ///     children: vec![],
    /// });
    /// assert_eq!(idx, 0);
    /// ```
    pub fn push(&mut self, tile: HlodTile) -> usize {
        let idx = self.tiles.len();
        self.tiles.push(tile);
        idx
    }
}

/// Result of one HLOD traversal: the set of content URIs to render.
///
/// # Examples
///
/// ```
/// use cesium::hlod::TraversalResult;
///
/// let mut r = TraversalResult::default();
/// assert!(r.render_list.is_empty());
/// r.render_list.push("tile.glb".into());
/// assert_eq!(r.render_list.len(), 1);
/// ```
#[derive(Clone, Debug, Default)]
pub struct TraversalResult {
    /// Content URIs of tiles selected for rendering (no duplicates guaranteed
    /// only if the tree structure itself has no cycles).
    pub render_list: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Live traversal stub (calls the commented-out recursive impl via shim)
// ─────────────────────────────────────────────────────────────────────────────

/// Traverse the HLOD tree from the root (index 0) and collect content URIs
/// to render given a camera position and SSE parameters.
///
/// **Stub.** The real recursive body is `//`-commented below; this entry point
/// delegates to the stub shim so callers and tests compile.
///
/// # Root invariant
///
/// Traversal always starts at index 0, which is the root **by construction**.
/// Callers must build the tree by pushing the root tile first (index 0); see
/// [`HlodTree`] for the full invariant.  If `tree.tiles` is empty this
/// function returns an empty [`TraversalResult`] immediately.
///
/// # Defensive safety
///
/// `traverse_tile` skips any child index that is out of range for the current
/// `tiles` slice (silent ignore — scaffold behaviour).  It also caps recursion
/// at [`MAX_HLOD_DEPTH`] (64) to prevent infinite loops on cyclic child-index
/// graphs produced by malformed input.  These guards do not affect well-formed
/// trees.
///
/// # Arguments
/// - `tree` — the HLOD tile tree (root at index 0).
/// - `camera` — world-space camera position.
/// - `sse_denominator` — pre-computed value from
///   [`crate::sse::SseFrustum::sse_denominator`].
/// - `maximum_sse` — accept threshold (Cesium default 16.0 pixels).
///
/// # Returns
/// [`TraversalResult`] containing the render list.
///
/// # Examples
///
/// ```
/// use cesium::hlod::{HlodTile, HlodTree, Refine, traverse_hlod};
/// use cesium::sse::SseFrustum;
///
/// // Build a tiny two-tile tree: root (Replace) → one leaf child.
/// let mut tree = HlodTree::new();
///
/// // Push root first (index 0).
/// let _root = tree.push(HlodTile {
///     geometric_error: 100.0,
///     bounding_sphere_center: [0.0, 0.0, 0.0],
///     bounding_sphere_radius: 10.0,
///     refine: Refine::Replace,
///     content_uri: Some("root.glb".into()),
///     children: vec![1],
/// });
/// // Push child (index 1).
/// let _child = tree.push(HlodTile {
///     geometric_error: 0.0,
///     bounding_sphere_center: [0.0, 0.0, 0.0],
///     bounding_sphere_radius: 1.0,
///     refine: Refine::Replace,
///     content_uri: Some("child.glb".into()),
///     children: vec![],
/// });
///
/// let denom = SseFrustum {
///     viewport_height_px: 1080.0,
///     fovy_rad: std::f32::consts::FRAC_PI_3,
/// }
/// .sse_denominator();
///
/// // Camera very far away → root SSE tiny → root accepted, no child.
/// let result = traverse_hlod(&tree, [0.0, 0.0, 1_000_000.0], denom, 16.0);
/// assert!(result.render_list.contains(&"root.glb".to_string()));
/// assert!(!result.render_list.contains(&"child.glb".to_string()));
///
/// // Camera at origin → root SSE enormous → Replace refines to child.
/// let result2 = traverse_hlod(&tree, [0.0, 0.0, 0.0], denom, 16.0);
/// assert!(!result2.render_list.contains(&"root.glb".to_string()));
/// assert!(result2.render_list.contains(&"child.glb".to_string()));
/// ```
pub fn traverse_hlod(tree: &HlodTree, camera: [f32; 3], sse_denominator: f32, maximum_sse: f32) -> TraversalResult {
    let mut result = TraversalResult::default();
    if tree.tiles.is_empty() {
        return result;
    }
    traverse_tile(tree, 0, camera, sse_denominator, maximum_sse, &mut result, 0);
    result
}

/// Recursive inner traversal (live stub — ADD/REPLACE logic is real).
///
/// `depth` tracks the current recursion depth.  When `depth >= MAX_HLOD_DEPTH`
/// the function returns immediately, preventing infinite loops on cyclic input.
/// Any child index that is out of range for `tree.tiles` is silently skipped.
fn traverse_tile(
    tree: &HlodTree, idx: usize, camera: [f32; 3], sse_denominator: f32, maximum_sse: f32,
    result: &mut TraversalResult, depth: usize,
) {
    // Depth guard — prevents infinite recursion on cyclic child-index graphs.
    if depth >= MAX_HLOD_DEPTH {
        return;
    }

    let tile = &tree.tiles[idx];
    let distance = tile.distance_to_camera(camera);
    let sse = crate::sse::sse_for_tile(tile.geometric_error, distance, sse_denominator);
    let refine_needed = sse > maximum_sse && !tile.is_leaf();

    match tile.refine {
        Refine::Replace => {
            if refine_needed {
                // REPLACE + must refine → skip parent, recurse into children.
                // Clone children indices to avoid borrow conflict on `tree`.
                let children: Vec<usize> = tile.children.clone();
                for &child_idx in &children {
                    // Bounds guard — skip out-of-range indices silently.
                    if child_idx >= tree.tiles.len() {
                        continue;
                    }
                    traverse_tile(tree, child_idx, camera, sse_denominator, maximum_sse, result, depth + 1);
                }
            } else {
                // REPLACE + SSE satisfied (or leaf) → render this tile.
                if let Some(uri) = &tile.content_uri {
                    result.render_list.push(uri.clone());
                }
            }
        }
        Refine::Add => {
            // ADD → always render this tile (if it has content).
            if let Some(uri) = &tile.content_uri {
                result.render_list.push(uri.clone());
            }
            if refine_needed {
                // ADD + must refine → also recurse into children.
                let children: Vec<usize> = tile.children.clone();
                for &child_idx in &children {
                    // Bounds guard — skip out-of-range indices silently.
                    if child_idx >= tree.tiles.len() {
                        continue;
                    }
                    traverse_tile(tree, child_idx, camera, sse_denominator, maximum_sse, result, depth + 1);
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// COMMENTED-OUT full implementation notes
// ─────────────────────────────────────────────────────────────────────────────
//
// When wiring to the live tileset parser (`crate::tileset`) and the SSE module:
//
// ```rust
// // Integration sketch (not yet live):
// //
// // use crate::sse::SseFrustum;
// // use crate::tileset::Tileset;
// //
// // pub fn render_list_from_tileset(
// //     tileset: &Tileset,
// //     camera: [f32; 3],
// //     frustum: SseFrustum,
// //     maximum_sse: f32,
// // ) -> TraversalResult {
// //     let tree = hlod_tree_from_tileset(tileset); // UNVERIFIED: exact mapping
// //     let denom = frustum.sse_denominator();
// //     traverse_hlod(&tree, camera, denom, maximum_sse)
// // }
// //
// // Frustum-culling pass (UNVERIFIED: exact Cesium pass order):
// //   Before calling traverse_tile, test whether tile bounding sphere
// //   intersects the view frustum.  If outside frustum → skip entirely
// //   (neither render nor recurse).
// //
// // UNVERIFIED: Cesium also applies `tileset.dynamicScreenSpaceError` for
// // tiles that are both far away and have high density — a progressive
// // density-based demotion.  Not implemented here.
// ```

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (std-only, live)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::SseFrustum;

    fn default_frustum() -> f32 {
        SseFrustum {
            viewport_height_px: 1080.0,
            fovy_rad: std::f32::consts::FRAC_PI_3,
        }
        .sse_denominator()
    }

    /// Build a minimal two-level tree: root → one child.
    fn two_level_tree(root_refine: Refine, ge_root: f32, ge_child: f32) -> HlodTree {
        let mut tree = HlodTree::new();
        let child_idx = tree.push(HlodTile {
            geometric_error: ge_child,
            bounding_sphere_center: [0.0, 0.0, 0.0],
            bounding_sphere_radius: 1.0,
            refine: Refine::Replace,
            content_uri: Some("child.glb".into()),
            children: vec![],
        });
        tree.push(HlodTile {
            geometric_error: ge_root,
            bounding_sphere_center: [0.0, 0.0, 0.0],
            bounding_sphere_radius: 10.0,
            refine: root_refine,
            content_uri: Some("root.glb".into()),
            children: vec![child_idx],
        });
        // Swap so root is at index 0 (last push is not root; rebuild correctly)
        // Actually we must re-build: root must be index 0.
        // Rebuild with correct order:
        let mut t2 = HlodTree::new();
        // root first (idx 0), child will be idx 1
        let _root = t2.push(HlodTile {
            geometric_error: ge_root,
            bounding_sphere_center: [0.0, 0.0, 0.0],
            bounding_sphere_radius: 10.0,
            refine: root_refine,
            content_uri: Some("root.glb".into()),
            children: vec![1], // child is index 1
        });
        let _child = t2.push(HlodTile {
            geometric_error: ge_child,
            bounding_sphere_center: [0.0, 0.0, 0.0],
            bounding_sphere_radius: 1.0,
            refine: Refine::Replace,
            content_uri: Some("child.glb".into()),
            children: vec![],
        });
        let _ = tree; // discard the mis-built one
        t2
    }

    #[test]
    fn empty_tree_returns_empty_list() {
        let tree = HlodTree::new();
        let result = traverse_hlod(&tree, [0.0, 0.0, 100.0], default_frustum(), 16.0);
        assert!(result.render_list.is_empty());
    }

    #[test]
    fn single_leaf_always_rendered() {
        let mut tree = HlodTree::new();
        tree.push(HlodTile {
            geometric_error: 0.0,
            bounding_sphere_center: [0.0, 0.0, 0.0],
            bounding_sphere_radius: 1.0,
            refine: Refine::Replace,
            content_uri: Some("leaf.glb".into()),
            children: vec![],
        });
        // Camera very far away — SSE is 0 (ge=0) → accepted
        let r = traverse_hlod(&tree, [0.0, 0.0, 1_000_000.0], default_frustum(), 16.0);
        assert_eq!(r.render_list, vec!["leaf.glb"]);
    }

    #[test]
    fn replace_refine_renders_child_when_sse_too_large() {
        // Root has huge geometric error close up → must refine.
        // Camera at origin, root bounding sphere at origin r=1 → distance ≈ 0 → enormous SSE.
        let tree = two_level_tree(Refine::Replace, 10_000.0, 0.0);
        let r = traverse_hlod(&tree, [0.0, 0.0, 0.0], default_frustum(), 16.0);
        // REPLACE + refine → parent NOT rendered, child rendered.
        assert!(!r.render_list.contains(&"root.glb".to_string()), "root must not appear on REPLACE refine");
        assert!(r.render_list.contains(&"child.glb".to_string()), "child must appear");
    }

    #[test]
    fn replace_refine_renders_root_when_sse_small() {
        // Camera very far away → SSE tiny → root accepted, no refinement.
        let tree = two_level_tree(Refine::Replace, 1.0, 0.0);
        let r = traverse_hlod(&tree, [0.0, 0.0, 1_000_000.0], default_frustum(), 16.0);
        assert!(r.render_list.contains(&"root.glb".to_string()));
        assert!(!r.render_list.contains(&"child.glb".to_string()), "child must NOT appear when root accepted");
    }

    #[test]
    fn add_refine_renders_both_when_sse_too_large() {
        // ADD: parent always rendered; children also rendered when SSE demands it.
        let tree = two_level_tree(Refine::Add, 10_000.0, 0.0);
        let r = traverse_hlod(&tree, [0.0, 0.0, 0.0], default_frustum(), 16.0);
        assert!(r.render_list.contains(&"root.glb".to_string()), "ADD root must always render");
        assert!(r.render_list.contains(&"child.glb".to_string()), "ADD child must render when refining");
    }

    #[test]
    fn add_refine_renders_only_root_when_sse_small() {
        let tree = two_level_tree(Refine::Add, 1.0, 0.0);
        let r = traverse_hlod(&tree, [0.0, 0.0, 1_000_000.0], default_frustum(), 16.0);
        assert!(r.render_list.contains(&"root.glb".to_string()));
        // SSE satisfied → no refinement → child not included for ADD either
        assert!(!r.render_list.contains(&"child.glb".to_string()));
    }

    #[test]
    fn tile_without_content_uri_does_not_pollute_render_list() {
        let mut tree = HlodTree::new();
        // Root with no content_uri but two children with content.
        tree.push(HlodTile {
            geometric_error: 100.0,
            bounding_sphere_center: [0.0, 0.0, 0.0],
            bounding_sphere_radius: 10.0,
            refine: Refine::Replace,
            content_uri: None, // no content
            children: vec![1, 2],
        });
        tree.push(HlodTile {
            geometric_error: 0.0,
            bounding_sphere_center: [0.0, 0.0, 0.0],
            bounding_sphere_radius: 1.0,
            refine: Refine::Replace,
            content_uri: Some("a.glb".into()),
            children: vec![],
        });
        tree.push(HlodTile {
            geometric_error: 0.0,
            bounding_sphere_center: [0.0, 0.0, 0.0],
            bounding_sphere_radius: 1.0,
            refine: Refine::Replace,
            content_uri: Some("b.glb".into()),
            children: vec![],
        });
        // Camera close → root SSE enormous → refine → children rendered.
        let r = traverse_hlod(&tree, [0.0, 0.0, 0.0], default_frustum(), 16.0);
        assert!(!r.render_list.iter().any(|s| s.is_empty()), "no empty URI strings");
        assert!(r.render_list.contains(&"a.glb".to_string()));
        assert!(r.render_list.contains(&"b.glb".to_string()));
        assert_eq!(r.render_list.len(), 2);
    }

    #[test]
    fn refine_enum_semantics_documented() {
        assert_ne!(Refine::Add, Refine::Replace);
        // Confirm Copy semantics
        let r = Refine::Add;
        let r2 = r; // Copy
        assert_eq!(r, r2);
    }

    #[test]
    fn out_of_range_child_index_is_skipped() {
        // Root references a child index (99) that does not exist.
        let mut tree = HlodTree::new();
        tree.push(HlodTile {
            geometric_error: 10_000.0,
            bounding_sphere_center: [0.0, 0.0, 0.0],
            bounding_sphere_radius: 10.0,
            refine: Refine::Replace,
            content_uri: Some("root.glb".into()),
            children: vec![99], // out of range
        });
        // Must not panic; render list is empty (REPLACE + refine, but no valid child).
        let r = traverse_hlod(&tree, [0.0, 0.0, 0.0], default_frustum(), 16.0);
        assert!(r.render_list.is_empty());
    }

    #[test]
    fn depth_guard_prevents_cycle_infinite_loop() {
        // Tile 0 points to tile 1 which points back to tile 0 — a cycle.
        let mut tree = HlodTree::new();
        tree.push(HlodTile {
            geometric_error: 10_000.0,
            bounding_sphere_center: [0.0, 0.0, 0.0],
            bounding_sphere_radius: 10.0,
            refine: Refine::Add,
            content_uri: None,
            children: vec![1],
        });
        tree.push(HlodTile {
            geometric_error: 10_000.0,
            bounding_sphere_center: [0.0, 0.0, 0.0],
            bounding_sphere_radius: 10.0,
            refine: Refine::Add,
            content_uri: None,
            children: vec![0], // cycle back to root
        });
        // Must terminate (depth guard fires at MAX_HLOD_DEPTH).
        let r = traverse_hlod(&tree, [0.0, 0.0, 0.0], default_frustum(), 16.0);
        // Both tiles have no content_uri — render list empty, but no panic/hang.
        let _ = r;
    }
}
