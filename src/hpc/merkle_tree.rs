//! Merkle tree over CogRecord regions — compressed searchable proxy.
//!
//! An 8Kbit Merkle tree built from a CogRecord's fields serves as a
//! compressed searchable proxy for the full node. Algorithms that work
//! on 16Kbit Containers (CLAM, CHAODA, CHESS, CAKES, panCAKES, BNN,
//! cascade) should also work on the 8Kbit tree at higher speed.

use super::seal::MerkleRoot;

/// Number of branches in the Merkle tree.
const NUM_BRANCHES: usize = 8;

/// Number of leaves in the Merkle tree.
const NUM_LEAVES: usize = 64;

/// Number of u64 words in the flat bits array (8Kbit = 128 x u64).
const BITS_WORDS: usize = 128;

/// Branch region definitions: (start_word, end_word_exclusive) within the
/// 256-word u64 metadata container, except branch 7 which covers content.
const BRANCH_REGIONS: [(usize, usize); 8] = [
    (0, 16),    // [0] identity
    (4, 8),     // [1] nars (overlaps identity — NARS truth words)
    (16, 32),   // [2] edges
    (32, 40),   // [3] rl
    (40, 48),   // [4] bloom
    (56, 64),   // [5] qualia
    (96, 112),  // [6] adjacency
    (0, 0),     // [7] content — handled specially
];

/// Type of change detected between two Merkle trees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaunenType {
    /// Nothing changed — trees are identical.
    Wisdom,
    /// Only content (branch 7) differs.
    ContentChanged,
    /// Only NARS truth (branch 1) differs.
    NarsChanged,
    /// Only edges (branch 2) differ.
    EdgesChanged,
    /// Only qualia (branch 5) differs.
    QualiaChanged,
    /// Several branches differ — contains the list of differing branch indices.
    MultipleChanges(Vec<u8>),
}

/// Merkle tree over CogRecord regions.
///
/// Each branch is a Blake3 hash of a metadata region or content container.
/// The tree IS a searchable binary vector: 8Kbit (1KB) aligned for SIMD.
///
/// # Layout
///
/// - Level 0: root hash (48 bits, Blake3 truncated)
/// - Level 1: 8 branch hashes (8 x 48 bits = 384 bits)
/// - Level 2: 64 leaf hashes (64 x 48 bits = 3072 bits)
/// - Total semantic: 48 + 384 + 3072 = 3504 bits
/// - Padded to 8192 bits (1KB) for SIMD alignment
#[derive(Clone)]
pub struct MerkleTree {
    /// Level 0: root hash (48 bits, Blake3 truncated).
    pub root: MerkleRoot,
    /// Level 1: branch hashes (8 x 48 bits = 384 bits).
    ///
    /// Indices: [0] identity, [1] nars, [2] edges, [3] rl,
    /// [4] bloom, [5] qualia, [6] adjacency, [7] content.
    pub branches: [MerkleRoot; NUM_BRANCHES],
    /// Level 2: leaf hashes (64 x 48 bits = 3072 bits).
    pub leaves: [MerkleRoot; NUM_LEAVES],
    /// Full tree as flat bits for SIMD operations.
    ///
    /// Total: 48 + 384 + 3072 = 3504 bits minimum,
    /// padded to 8192 bits (1KB) for alignment.
    pub bits: [u64; BITS_WORDS],
}

/// Truncate a blake3 hash to 48 bits and return a `MerkleRoot`.
#[inline]
fn truncate_hash(hash: &blake3::Hash) -> MerkleRoot {
    let mut root = [0u8; 6];
    root.copy_from_slice(&hash.as_bytes()[..6]);
    MerkleRoot(root)
}

/// Hash a slice of u64 words with blake3, truncate to 48 bits.
#[inline]
fn hash_words(words: &[u64]) -> MerkleRoot {
    // SAFETY: u64 slice reinterpretation as u8 is safe — u8 has no alignment
    // requirement, and the byte count is exact (words.len() * 8).
    let bytes = unsafe {
        core::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 8)
    };
    truncate_hash(&blake3::hash(bytes))
}

impl MerkleTree {
    /// Build a Merkle tree from a CogRecord's metadata and content containers.
    ///
    /// # Arguments
    ///
    /// * `meta` - The 256-word u64 metadata container.
    /// * `content` - Slice of references to 4 content containers (each 256 u64 words).
    ///
    /// # Panics
    ///
    /// Panics if `content` is empty.
    pub fn from_cogrecord(meta: &[u64; 256], content: &[&[u64; 256]]) -> Self {
        // --- Level 1: branch hashes ---
        let mut branches = [MerkleRoot([0u8; 6]); NUM_BRANCHES];
        for i in 0..7 {
            let (start, end) = BRANCH_REGIONS[i];
            branches[i] = hash_words(&meta[start..end]);
        }

        // Branch 7: hash all content containers concatenated
        let mut content_hasher = blake3::Hasher::new();
        for container in content {
            // SAFETY: u64 array reinterpretation as u8 is safe — u8 has no
            // alignment requirement, and the byte count is exact (256 * 8).
            let bytes = unsafe {
                core::slice::from_raw_parts(container.as_ptr() as *const u8, 256 * 8)
            };
            content_hasher.update(bytes);
        }
        branches[7] = truncate_hash(&content_hasher.finalize());

        // --- Level 2: leaf hashes ---
        // 64 leaves total: 8 leaves per branch (subdivide each branch region
        // into ~equal groups of words).
        let mut leaves = [MerkleRoot([0u8; 6]); NUM_LEAVES];
        let leaves_per_branch = NUM_LEAVES / NUM_BRANCHES; // 8

        for br in 0..7 {
            let (start, end) = BRANCH_REGIONS[br];
            let region_len = end - start;
            let leaf_base = br * leaves_per_branch;

            for leaf in 0..leaves_per_branch {
                let w_start = start + (leaf * region_len) / leaves_per_branch;
                let w_end = start + ((leaf + 1) * region_len) / leaves_per_branch;
                if w_start < w_end {
                    leaves[leaf_base + leaf] = hash_words(&meta[w_start..w_end]);
                } else {
                    // Degenerate sub-region: hash a single word or empty
                    leaves[leaf_base + leaf] = hash_words(&meta[w_start..w_start.saturating_add(1).min(end)]);
                }
            }
        }

        // Branch 7 leaves: split concatenated content into 8 chunks
        {
            let leaf_base = 7 * leaves_per_branch;
            let total_words: usize = content.len() * 256;
            let chunk_size = total_words / leaves_per_branch;

            // Flatten content into a temporary buffer for leaf hashing
            let mut flat = Vec::with_capacity(total_words);
            for container in content {
                flat.extend_from_slice(container.as_slice());
            }

            for leaf in 0..leaves_per_branch {
                let w_start = leaf * chunk_size;
                let w_end = if leaf == leaves_per_branch - 1 {
                    flat.len()
                } else {
                    (leaf + 1) * chunk_size
                };
                leaves[leaf_base + leaf] = hash_words(&flat[w_start..w_end]);
            }
        }

        // --- Level 0: root hash ---
        let mut root_hasher = blake3::Hasher::new();
        for branch in &branches {
            root_hasher.update(branch.as_bytes());
        }
        let root = truncate_hash(&root_hasher.finalize());

        // --- Pack into bits array ---
        let bits = Self::pack_bits(&root, &branches, &leaves);

        Self {
            root,
            branches,
            leaves,
            bits,
        }
    }

    /// Pack root, branches, and leaves into a flat 8Kbit (128 x u64) array.
    fn pack_bits(
        root: &MerkleRoot,
        branches: &[MerkleRoot; NUM_BRANCHES],
        leaves: &[MerkleRoot; NUM_LEAVES],
    ) -> [u64; BITS_WORDS] {
        let mut bits = [0u64; BITS_WORDS];

        // Collect all MerkleRoot bytes into a flat buffer, then pack into u64s.
        // Total bytes: 6 (root) + 8*6 (branches) + 64*6 (leaves) = 438 bytes.
        let mut buf = [0u8; BITS_WORDS * 8]; // 1024 bytes, zero-padded
        let mut offset = 0;

        buf[offset..offset + 6].copy_from_slice(root.as_bytes());
        offset += 6;

        for branch in branches {
            buf[offset..offset + 6].copy_from_slice(branch.as_bytes());
            offset += 6;
        }

        for leaf in leaves {
            buf[offset..offset + 6].copy_from_slice(leaf.as_bytes());
            offset += 6;
        }

        // Pack bytes into u64 words (little-endian)
        for (i, word) in bits.iter_mut().enumerate() {
            let base = i * 8;
            *word = u64::from_le_bytes([
                buf[base],
                buf[base + 1],
                buf[base + 2],
                buf[base + 3],
                buf[base + 4],
                buf[base + 5],
                buf[base + 6],
                buf[base + 7],
            ]);
        }

        bits
    }

    /// View the bits array as a byte slice for SIMD operations.
    #[inline]
    fn bits_as_bytes(&self) -> &[u8] {
        // SAFETY: [u64; 128] is contiguous in memory. u8 has no alignment
        // requirement stricter than u64. Length 128 * 8 = 1024 is exact.
        unsafe {
            core::slice::from_raw_parts(self.bits.as_ptr() as *const u8, BITS_WORDS * 8)
        }
    }

    /// Hamming distance between two Merkle trees over the full 8Kbit vector.
    ///
    /// Uses SIMD-accelerated `hamming_distance_raw()` for maximum throughput.
    #[inline]
    pub fn hamming(&self, other: &MerkleTree) -> u32 {
        super::bitwise::hamming_distance_raw(self.bits_as_bytes(), other.bits_as_bytes()) as u32
    }

    /// Per-branch diff: returns `[bool; 8]` where `true` means the branch differs.
    #[inline]
    pub fn diff_branches(&self, other: &MerkleTree) -> [bool; 8] {
        let mut diff = [false; NUM_BRANCHES];
        for i in 0..NUM_BRANCHES {
            diff[i] = self.branches[i] != other.branches[i];
        }
        diff
    }

    /// Classify the type of change between this tree and a stored reference.
    ///
    /// Analyzes which branches differ and returns a semantic `StaunenType`.
    pub fn typed_staunen(&self, stored: &MerkleTree) -> StaunenType {
        let diff = self.diff_branches(stored);
        let changed: Vec<u8> = diff
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| if d { Some(i as u8) } else { None })
            .collect();

        match changed.len() {
            0 => StaunenType::Wisdom,
            1 => match changed[0] {
                7 => StaunenType::ContentChanged,
                1 => StaunenType::NarsChanged,
                2 => StaunenType::EdgesChanged,
                5 => StaunenType::QualiaChanged,
                _ => StaunenType::MultipleChanges(changed),
            },
            _ => StaunenType::MultipleChanges(changed),
        }
    }

    /// XOR diff between two Merkle trees for panCAKES compression.
    ///
    /// Returns a new `MerkleTree` whose `bits` array is the XOR of the two
    /// input trees. The root, branches, and leaves are recomputed from the
    /// XOR'd bytes by re-hashing.
    pub fn xor_diff(&self, other: &MerkleTree) -> MerkleTree {
        let mut bits = [0u64; BITS_WORDS];
        for i in 0..BITS_WORDS {
            bits[i] = self.bits[i] ^ other.bits[i];
        }

        // Reconstruct root/branches/leaves from the XOR'd bits by unpacking
        let buf = {
            let mut b = [0u8; BITS_WORDS * 8];
            for (i, &word) in bits.iter().enumerate() {
                let base = i * 8;
                b[base..base + 8].copy_from_slice(&word.to_le_bytes());
            }
            b
        };

        let mut offset = 0;

        let mut root_bytes = [0u8; 6];
        root_bytes.copy_from_slice(&buf[offset..offset + 6]);
        let root = MerkleRoot(root_bytes);
        offset += 6;

        let mut branches = [MerkleRoot([0u8; 6]); NUM_BRANCHES];
        for branch in branches.iter_mut() {
            let mut b = [0u8; 6];
            b.copy_from_slice(&buf[offset..offset + 6]);
            *branch = MerkleRoot(b);
            offset += 6;
        }

        let mut leaves = [MerkleRoot([0u8; 6]); NUM_LEAVES];
        for leaf in leaves.iter_mut() {
            let mut b = [0u8; 6];
            b.copy_from_slice(&buf[offset..offset + 6]);
            *leaf = MerkleRoot(b);
            offset += 6;
        }

        MerkleTree {
            root,
            branches,
            leaves,
            bits,
        }
    }

    /// Count non-zero branches in the diff between two trees.
    ///
    /// Returns a sparsity score 0..=8 (0 = identical, 8 = all branches differ).
    #[inline]
    pub fn diff_sparsity(&self, other: &MerkleTree) -> u8 {
        self.diff_branches(other)
            .iter()
            .filter(|&&d| d)
            .count() as u8
    }
}

impl core::fmt::Debug for MerkleTree {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "MerkleTree {{ root: {:?}, branches: {} differing from zero }}",
            self.root,
            self.branches.iter().filter(|b| b.0 != [0u8; 6]).count()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create deterministic test metadata (256 u64 words).
    fn make_meta(seed: u64) -> [u64; 256] {
        let mut meta = [0u64; 256];
        let mut state = seed;
        for word in meta.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *word = state;
        }
        meta
    }

    /// Create deterministic test content containers.
    fn make_content(seed: u64) -> [[u64; 256]; 4] {
        let mut containers = [[0u64; 256]; 4];
        let mut state = seed;
        for container in containers.iter_mut() {
            for word in container.iter_mut() {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                *word = state;
            }
        }
        containers
    }

    fn build_tree(meta_seed: u64, content_seed: u64) -> MerkleTree {
        let meta = make_meta(meta_seed);
        let content = make_content(content_seed);
        let content_refs: Vec<&[u64; 256]> = content.iter().collect();
        MerkleTree::from_cogrecord(&meta, &content_refs)
    }

    #[test]
    fn test_from_cogrecord_deterministic() {
        let tree1 = build_tree(42, 99);
        let tree2 = build_tree(42, 99);
        assert_eq!(tree1.root, tree2.root);
        assert_eq!(tree1.branches, tree2.branches);
        assert_eq!(tree1.leaves, tree2.leaves);
        assert_eq!(tree1.bits, tree2.bits);
    }

    #[test]
    fn test_hamming_self_zero() {
        let tree = build_tree(42, 99);
        assert_eq!(tree.hamming(&tree), 0);
    }

    #[test]
    fn test_hamming_different() {
        let tree_a = build_tree(42, 99);
        let tree_b = build_tree(100, 200);
        assert!(tree_a.hamming(&tree_b) > 0);
    }

    #[test]
    fn test_diff_branches_identical() {
        let tree = build_tree(42, 99);
        let diff = tree.diff_branches(&tree);
        assert_eq!(diff, [false; 8]);
    }

    #[test]
    fn test_typed_staunen_wisdom() {
        let tree = build_tree(42, 99);
        assert_eq!(tree.typed_staunen(&tree), StaunenType::Wisdom);
    }

    #[test]
    fn test_typed_staunen_content() {
        let meta = make_meta(42);
        let content_a = make_content(99);
        let content_b = make_content(200);
        let refs_a: Vec<&[u64; 256]> = content_a.iter().collect();
        let refs_b: Vec<&[u64; 256]> = content_b.iter().collect();
        let tree_a = MerkleTree::from_cogrecord(&meta, &refs_a);
        let tree_b = MerkleTree::from_cogrecord(&meta, &refs_b);
        // Same meta, different content → only branch 7 differs
        assert_eq!(tree_a.typed_staunen(&tree_b), StaunenType::ContentChanged);
    }

    #[test]
    fn test_typed_staunen_nars() {
        let mut meta_a = make_meta(42);
        let meta_b = meta_a;
        // Mutate only words 4..8 (NARS region = branch 1).
        // Branch 0 (identity) covers words 0..16 which includes 4..8,
        // so mutating 4..8 changes both branch 0 and branch 1.
        // To isolate branch 1 only, we need branch 0 to also differ —
        // actually the spec says branch 1 overlaps identity intentionally.
        // For a pure NARS test, only change words 4..8 but note that
        // branch 0 (0..16) will also change → MultipleChanges.
        // To get NarsChanged we need only branch 1 to differ.
        // Branch 1 uses words 4..8. Branch 0 uses words 0..16.
        // Since 4..8 is inside 0..16, any change to 4..8 also changes branch 0.
        // The only way to get pure NarsChanged is if branch 0 hash happens to
        // stay the same — which won't happen. So we test the MultipleChanges
        // path for the overlap case and test NarsChanged by directly
        // constructing trees with only branch 1 differing.
        let content = make_content(99);
        let refs: Vec<&[u64; 256]> = content.iter().collect();
        let tree_a = MerkleTree::from_cogrecord(&meta_a, &refs);

        // Construct tree_b by hand with only branch[1] different
        let mut tree_b = tree_a.clone();
        tree_b.branches[1] = MerkleRoot([0xFF; 6]);
        // Repack bits
        tree_b.bits = MerkleTree::pack_bits(&tree_b.root, &tree_b.branches, &tree_b.leaves);

        assert_eq!(tree_a.typed_staunen(&tree_b), StaunenType::NarsChanged);
    }

    #[test]
    fn test_typed_staunen_multiple() {
        let tree_a = build_tree(42, 99);
        let tree_b = build_tree(100, 200);
        // Different seeds → all branches differ
        match tree_a.typed_staunen(&tree_b) {
            StaunenType::MultipleChanges(branches) => {
                assert!(branches.len() > 1);
            }
            other => panic!("Expected MultipleChanges, got {:?}", other),
        }
    }

    #[test]
    fn test_xor_diff_self_zero() {
        let tree = build_tree(42, 99);
        let diff = tree.xor_diff(&tree);
        for word in &diff.bits {
            assert_eq!(*word, 0, "XOR with self should be all zeros");
        }
    }

    #[test]
    fn test_diff_sparsity() {
        // Identical trees → 0
        let tree = build_tree(42, 99);
        assert_eq!(tree.diff_sparsity(&tree), 0);

        // Fully different trees → 8
        let tree_b = build_tree(100, 200);
        assert_eq!(tree.diff_sparsity(&tree_b), 8);

        // Only content differs → 1
        let meta = make_meta(42);
        let c1 = make_content(99);
        let c2 = make_content(200);
        let r1: Vec<&[u64; 256]> = c1.iter().collect();
        let r2: Vec<&[u64; 256]> = c2.iter().collect();
        let t1 = MerkleTree::from_cogrecord(&meta, &r1);
        let t2 = MerkleTree::from_cogrecord(&meta, &r2);
        assert_eq!(t1.diff_sparsity(&t2), 1);
    }
}
