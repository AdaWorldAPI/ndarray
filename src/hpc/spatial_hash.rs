//! 3D spatial hash grid for efficient proximity queries.
//!
//! Entities are hashed into axis-aligned cells by position. Radius and KNN
//! queries only visit cells that overlap the search volume, giving O(1)
//! amortised lookup for uniformly distributed entities.
//!
//! The grid itself stores only `(cell_key -> [entity_id])`. Actual positions
//! are passed in by reference at query time so the caller keeps ownership.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn sq_dist_f32(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

// ---------------------------------------------------------------------------
// SpatialHash
// ---------------------------------------------------------------------------

/// 3D spatial hash. Entities are hashed into cells by position.
pub struct SpatialHash {
    cell_size: f32,
    inv_cell_size: f32,
    grid: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl SpatialHash {
    /// Create a new spatial hash with the given cell size.
    ///
    /// # Panics
    /// Panics if `cell_size` is not positive and finite.
    pub fn new(cell_size: f32) -> Self {
        assert!(cell_size > 0.0 && cell_size.is_finite(), "cell_size must be positive and finite");
        Self {
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            grid: HashMap::new(),
        }
    }

    /// Insert an entity at the given position.
    pub fn insert(&mut self, id: u32, x: f32, y: f32, z: f32) {
        let key = self.cell_key(x, y, z);
        self.grid.entry(key).or_default().push(id);
    }

    /// Remove an entity from its cell. Returns `true` if found and removed.
    pub fn remove(&mut self, id: u32, x: f32, y: f32, z: f32) -> bool {
        let key = self.cell_key(x, y, z);
        if let Some(cell) = self.grid.get_mut(&key) {
            if let Some(pos) = cell.iter().position(|&eid| eid == id) {
                cell.swap_remove(pos);
                if cell.is_empty() {
                    self.grid.remove(&key);
                }
                return true;
            }
        }
        false
    }

    /// Move an entity from `old` position to `new` position.
    pub fn update(&mut self, id: u32, old: [f32; 3], new: [f32; 3]) {
        let old_key = self.cell_key(old[0], old[1], old[2]);
        let new_key = self.cell_key(new[0], new[1], new[2]);
        if old_key != new_key {
            self.remove(id, old[0], old[1], old[2]);
            self.insert(id, new[0], new[1], new[2]);
        }
    }

    /// Remove all entities from the grid.
    pub fn clear(&mut self) {
        self.grid.clear();
    }

    /// Total number of entity entries across all cells.
    pub fn len(&self) -> usize {
        self.grid.values().map(|v| v.len()).sum()
    }

    /// Whether the grid contains zero entities.
    pub fn is_empty(&self) -> bool {
        self.grid.is_empty()
    }

    /// Find all entities within `radius` of `(x, y, z)`.
    ///
    /// `positions` maps entity id to its `[x, y, z]`. Only entities present
    /// in `positions` are considered. Returns `(entity_id, squared_distance)`
    /// sorted ascending by distance.
    pub fn query_radius(
        &self,
        x: f32,
        y: f32,
        z: f32,
        radius: f32,
        positions: &HashMap<u32, [f32; 3]>,
    ) -> Vec<(u32, f32)> {
        let radius_sq = radius * radius;
        let query = [x, y, z];

        // Determine cell range to search.
        let min_cx = ((x - radius) * self.inv_cell_size).floor() as i32;
        let max_cx = ((x + radius) * self.inv_cell_size).floor() as i32;
        let min_cy = ((y - radius) * self.inv_cell_size).floor() as i32;
        let max_cy = ((y + radius) * self.inv_cell_size).floor() as i32;
        let min_cz = ((z - radius) * self.inv_cell_size).floor() as i32;
        let max_cz = ((z + radius) * self.inv_cell_size).floor() as i32;

        let mut results: Vec<(u32, f32)> = Vec::new();

        for cx in min_cx..=max_cx {
            for cy in min_cy..=max_cy {
                for cz in min_cz..=max_cz {
                    if let Some(cell) = self.grid.get(&(cx, cy, cz)) {
                        for &eid in cell {
                            if let Some(&pos) = positions.get(&eid) {
                                let d2 = sq_dist_f32(query, pos);
                                if d2 <= radius_sq {
                                    results.push((eid, d2));
                                }
                            }
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        results
    }

    /// Find K nearest entities to `(x, y, z)`.
    ///
    /// Uses expanding-ring search: starts at the cell containing the query
    /// point and expands outward until at least K candidates are found, then
    /// refines. Returns `(entity_id, squared_distance)` sorted ascending.
    pub fn query_knn(
        &self,
        x: f32,
        y: f32,
        z: f32,
        k: usize,
        positions: &HashMap<u32, [f32; 3]>,
    ) -> Vec<(u32, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let query = [x, y, z];

        // Expand ring from 0 until we have enough candidates.
        let mut candidates: Vec<(u32, f32)> = Vec::new();
        let (cx, cy, cz) = self.cell_key(x, y, z);

        let mut ring = 0i32;
        let max_ring = 64; // safety cap

        loop {
            // Collect candidates from all cells in this ring shell.
            for dx in -ring..=ring {
                for dy in -ring..=ring {
                    for dz in -ring..=ring {
                        // Only visit cells on the shell (at least one coord at
                        // the ring boundary) to avoid re-visiting interior.
                        if dx.abs() != ring && dy.abs() != ring && dz.abs() != ring {
                            continue;
                        }
                        let key = (cx + dx, cy + dy, cz + dz);
                        if let Some(cell) = self.grid.get(&key) {
                            for &eid in cell {
                                if let Some(&pos) = positions.get(&eid) {
                                    let d2 = sq_dist_f32(query, pos);
                                    candidates.push((eid, d2));
                                }
                            }
                        }
                    }
                }
            }

            if ring >= max_ring {
                break;
            }

            // If we have at least k candidates, check whether the k-th best
            // is closer than the nearest possible point in the next ring.
            // If so, no further ring can improve the result.
            if candidates.len() >= k {
                candidates.sort_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal)
                });
                let worst = candidates[k - 1].1;
                // The nearest point in ring+1 is at least (ring * cell_size) away.
                let next_ring_min = (ring as f32) * self.cell_size;
                if worst <= next_ring_min * next_ring_min {
                    break;
                }
            }

            ring += 1;
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));
        candidates.truncate(k);
        candidates
    }

    /// Compute the cell key for a world-space coordinate.
    fn cell_key(&self, x: f32, y: f32, z: f32) -> (i32, i32, i32) {
        (
            (x * self.inv_cell_size).floor() as i32,
            (y * self.inv_cell_size).floor() as i32,
            (z * self.inv_cell_size).floor() as i32,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_positions(pts: &[(u32, [f32; 3])]) -> HashMap<u32, [f32; 3]> {
        pts.iter().copied().collect()
    }

    // -- insert / remove --

    #[test]
    fn test_insert_and_len() {
        let mut sh = SpatialHash::new(10.0);
        assert!(sh.is_empty());
        sh.insert(0, 1.0, 2.0, 3.0);
        sh.insert(1, 11.0, 2.0, 3.0);
        assert_eq!(sh.len(), 2);
        assert!(!sh.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, 1.0, 2.0, 3.0);
        assert!(sh.remove(0, 1.0, 2.0, 3.0));
        assert!(sh.is_empty());
    }

    #[test]
    fn test_remove_not_found() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, 1.0, 2.0, 3.0);
        assert!(!sh.remove(99, 1.0, 2.0, 3.0));
        assert!(!sh.remove(0, 999.0, 999.0, 999.0));
        assert_eq!(sh.len(), 1);
    }

    // -- update --

    #[test]
    fn test_update_same_cell() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, 1.0, 1.0, 1.0);
        sh.update(0, [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]); // same cell
        assert_eq!(sh.len(), 1);
    }

    #[test]
    fn test_update_different_cell() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, 1.0, 1.0, 1.0);
        sh.update(0, [1.0, 1.0, 1.0], [100.0, 100.0, 100.0]); // different cell
        assert_eq!(sh.len(), 1);
        // Old cell should be empty now
        assert!(sh.remove(0, 100.0, 100.0, 100.0));
        assert!(sh.is_empty());
    }

    // -- clear --

    #[test]
    fn test_clear() {
        let mut sh = SpatialHash::new(10.0);
        for i in 0..50 {
            sh.insert(i, i as f32, 0.0, 0.0);
        }
        assert_eq!(sh.len(), 50);
        sh.clear();
        assert!(sh.is_empty());
    }

    // -- radius query --

    #[test]
    fn test_query_radius_basic() {
        let mut sh = SpatialHash::new(10.0);
        let pts = vec![
            (0u32, [0.0f32, 0.0, 0.0]),
            (1, [5.0, 0.0, 0.0]),
            (2, [20.0, 0.0, 0.0]),
            (3, [100.0, 0.0, 0.0]),
        ];
        for &(id, pos) in &pts {
            sh.insert(id, pos[0], pos[1], pos[2]);
        }
        let positions = make_positions(&pts);
        let result = sh.query_radius(0.0, 0.0, 0.0, 10.0, &positions);
        let ids: Vec<u32> = result.iter().map(|&(id, _)| id).collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(!ids.contains(&2)); // dist=20 > radius=10
        assert!(!ids.contains(&3));
    }

    #[test]
    fn test_query_radius_sorted_by_distance() {
        let mut sh = SpatialHash::new(5.0);
        let pts = vec![
            (0u32, [10.0f32, 0.0, 0.0]),
            (1, [3.0, 0.0, 0.0]),
            (2, [1.0, 0.0, 0.0]),
        ];
        for &(id, pos) in &pts {
            sh.insert(id, pos[0], pos[1], pos[2]);
        }
        let positions = make_positions(&pts);
        let result = sh.query_radius(0.0, 0.0, 0.0, 20.0, &positions);
        // Should be sorted: id=2 (d=1), id=1 (d=9), id=0 (d=100)
        assert_eq!(result[0].0, 2);
        assert_eq!(result[1].0, 1);
        assert_eq!(result[2].0, 0);
    }

    #[test]
    fn test_query_radius_empty() {
        let sh = SpatialHash::new(10.0);
        let positions: HashMap<u32, [f32; 3]> = HashMap::new();
        let result = sh.query_radius(0.0, 0.0, 0.0, 100.0, &positions);
        assert!(result.is_empty());
    }

    // -- knn --

    #[test]
    fn test_knn_basic() {
        let mut sh = SpatialHash::new(10.0);
        let pts = vec![
            (0u32, [30.0f32, 0.0, 0.0]),
            (1, [10.0, 0.0, 0.0]),
            (2, [20.0, 0.0, 0.0]),
            (3, [5.0, 0.0, 0.0]),
        ];
        for &(id, pos) in &pts {
            sh.insert(id, pos[0], pos[1], pos[2]);
        }
        let positions = make_positions(&pts);
        let result = sh.query_knn(0.0, 0.0, 0.0, 2, &positions);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 3); // dist=25
        assert_eq!(result[1].0, 1); // dist=100
    }

    #[test]
    fn test_knn_vs_brute_force() {
        let mut sh = SpatialHash::new(5.0);
        let pts: Vec<(u32, [f32; 3])> = (0..50)
            .map(|i| {
                let v = i as f32 * 2.0;
                (i as u32, [v, v * 0.5, v * 0.3])
            })
            .collect();
        for &(id, pos) in &pts {
            sh.insert(id, pos[0], pos[1], pos[2]);
        }
        let positions = make_positions(&pts);
        let k = 5;
        let result = sh.query_knn(10.0, 5.0, 3.0, k, &positions);

        // Brute-force reference
        let mut brute: Vec<(u32, f32)> = pts
            .iter()
            .map(|&(id, pos)| (id, sq_dist_f32([10.0, 5.0, 3.0], pos)))
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        brute.truncate(k);

        assert_eq!(result.len(), brute.len());
        // Compare distances (not IDs — ties may break differently)
        for (r, b) in result.iter().zip(brute.iter()) {
            assert!(
                (r.1 - b.1).abs() < 1e-3,
                "knn dist mismatch: spatial_hash=({},{:.2}) brute=({},{:.2})",
                r.0, r.1, b.0, b.1
            );
        }
    }

    #[test]
    fn test_knn_k_zero() {
        let sh = SpatialHash::new(10.0);
        let positions: HashMap<u32, [f32; 3]> = HashMap::new();
        let result = sh.query_knn(0.0, 0.0, 0.0, 0, &positions);
        assert!(result.is_empty());
    }

    #[test]
    fn test_knn_k_larger_than_count() {
        let mut sh = SpatialHash::new(10.0);
        sh.insert(0, 1.0, 2.0, 3.0);
        let positions = make_positions(&[(0, [1.0, 2.0, 3.0])]);
        let result = sh.query_knn(0.0, 0.0, 0.0, 100, &positions);
        assert_eq!(result.len(), 1);
    }

    // -- negative coordinates --

    #[test]
    fn test_negative_coordinates() {
        let mut sh = SpatialHash::new(10.0);
        let pts = vec![
            (0u32, [-5.0f32, -5.0, -5.0]),
            (1, [5.0, 5.0, 5.0]),
        ];
        for &(id, pos) in &pts {
            sh.insert(id, pos[0], pos[1], pos[2]);
        }
        let positions = make_positions(&pts);
        let result = sh.query_radius(0.0, 0.0, 0.0, 20.0, &positions);
        assert_eq!(result.len(), 2);
    }

    #[test]
    #[should_panic]
    fn test_invalid_cell_size() {
        SpatialHash::new(0.0);
    }
}
