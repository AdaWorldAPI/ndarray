#![allow(missing_docs)]

//! Rotary Position Embedding (RoPE) — Llama / Mistral / Qwen3 style.
//!
//! # Algorithm
//!
//! RoPE encodes token position by rotating consecutive pairs of dimensions in
//! the query and key vectors.  For position `p` and dimension pair `(2i, 2i+1)`:
//!
//! ```text
//! θᵢ = 1 / theta^(2i / head_dim)
//! [q₂ᵢ, q₂ᵢ₊₁] ← [q₂ᵢ·cos(p·θᵢ) − q₂ᵢ₊₁·sin(p·θᵢ),
//!                   q₂ᵢ·sin(p·θᵢ) + q₂ᵢ₊₁·cos(p·θᵢ)]
//! ```
//!
//! The same rotation is applied to the key vector.
//!
//! # Cache strategy
//!
//! [`RopeCache`] pre-computes `cos(p·θᵢ)` and `sin(p·θᵢ)` for every position
//! `p ∈ [0, max_seq_len)` and every pair index `i ∈ [0, head_dim/2)` at
//! construction time, trading memory (2 × max_seq_len × head_dim/2 floats) for
//! zero transcendental cost on the forward pass.
//!
//! # References
//!
//! Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position
//! Embedding". <https://arxiv.org/abs/2104.09864>

/// Pre-computed cosine / sine tables for Rotary Position Embedding.
///
/// Build once with [`RopeCache::build`] and then call [`RopeCache::apply_qk_f32`]
/// for every forward pass.
///
/// # Layout
///
/// Both `cos_table` and `sin_table` are stored in row-major order with shape
/// `[max_seq_len, head_dim / 2]`.  Entry `[pos, i]` stores the value for
/// position `pos` and dimension pair `i`.
pub struct RopeCache {
    /// Cosine table, shape `[max_seq_len, head_dim / 2]` (row-major).
    pub cos_table: Vec<f32>,
    /// Sine table, shape `[max_seq_len, head_dim / 2]` (row-major).
    pub sin_table: Vec<f32>,
    /// Number of dimensions per attention head (must be even).
    pub head_dim: usize,
    /// Maximum supported sequence length.
    pub max_seq_len: usize,
}

impl RopeCache {
    /// Pre-compute cosine and sine tables.
    ///
    /// # Arguments
    ///
    /// * `head_dim`    — dimensions per head; **must be even**.
    /// * `max_seq_len` — maximum number of sequence positions to cache.
    /// * `theta`       — base for the geometric frequency sequence (Llama: 10000.0,
    ///                   Qwen3: 1 000 000.0).
    ///
    /// # Panics
    ///
    /// Panics if `head_dim` is zero or odd.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::hpc::linalg::rope::RopeCache;
    ///
    /// let cache = RopeCache::build(64, 2048, 10000.0);
    /// assert_eq!(cache.cos_table.len(), 2048 * 32);
    /// ```
    pub fn build(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        assert!(head_dim > 0 && head_dim % 2 == 0, "head_dim must be a positive even number");
        let half = head_dim / 2;
        let capacity = max_seq_len * half;
        let mut cos_table = vec![0.0_f32; capacity];
        let mut sin_table = vec![0.0_f32; capacity];

        for pos in 0..max_seq_len {
            for i in 0..half {
                // θᵢ = 1 / theta^(2i / head_dim)
                let freq = 1.0_f32 / theta.powf((2 * i) as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                cos_table[pos * half + i] = angle.cos();
                sin_table[pos * half + i] = angle.sin();
            }
        }

        Self { cos_table, sin_table, head_dim, max_seq_len }
    }

    /// Apply RoPE in-place to query and key tensors.
    ///
    /// # Tensor layout (flat, row-major)
    ///
    /// Both `q` and `k` are expected in `[batch, seq, heads, head_dim]` order.
    ///
    /// # Arguments
    ///
    /// * `q`         — query tensor; mutated in place.
    /// * `k`         — key tensor; mutated in place.
    /// * `positions` — token positions, shape `[batch, seq]` (flat).  Each
    ///                 value must be `< self.max_seq_len`.
    /// * `batch`     — batch size B.
    /// * `seq`       — sequence length S.
    /// * `heads`     — number of query/key heads H.
    ///
    /// # Panics
    ///
    /// Panics if any position exceeds `max_seq_len - 1` or if the slice
    /// lengths do not match `batch * seq * heads * head_dim`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::hpc::linalg::rope::RopeCache;
    ///
    /// let cache = RopeCache::build(4, 16, 10000.0);
    /// let mut q = vec![1.0_f32; 1 * 4 * 2 * 4]; // batch=1, seq=4, heads=2, head_dim=4
    /// let mut k = vec![1.0_f32; 1 * 4 * 2 * 4];
    /// let positions: Vec<u32> = (0..4).collect();
    /// cache.apply_qk_f32(&mut q, &mut k, &positions, 1, 4, 2);
    /// ```
    pub fn apply_qk_f32(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        positions: &[u32],
        batch: usize,
        seq: usize,
        heads: usize,
    ) {
        let half = self.head_dim / 2;
        let expected_qk = batch * seq * heads * self.head_dim;
        let expected_pos = batch * seq;
        assert_eq!(q.len(), expected_qk, "q length mismatch");
        assert_eq!(k.len(), expected_qk, "k length mismatch");
        assert_eq!(positions.len(), expected_pos, "positions length mismatch");

        for b in 0..batch {
            for s in 0..seq {
                let pos = positions[b * seq + s] as usize;
                assert!(pos < self.max_seq_len, "position {pos} >= max_seq_len {}", self.max_seq_len);
                let cos_row = &self.cos_table[pos * half..(pos + 1) * half];
                let sin_row = &self.sin_table[pos * half..(pos + 1) * half];

                for h in 0..heads {
                    let base = (b * seq * heads + s * heads + h) * self.head_dim;
                    rotate_pairs(&mut q[base..base + self.head_dim], cos_row, sin_row);
                    rotate_pairs(&mut k[base..base + self.head_dim], cos_row, sin_row);
                }
            }
        }
    }
}

/// Apply RoPE rotation to a single vector of length `head_dim` (even).
///
/// Operates on consecutive pairs `(x[2i], x[2i+1])` using the supplied
/// cosine and sine slices.  No allocations; purely in-place arithmetic.
#[inline(always)]
fn rotate_pairs(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    debug_assert_eq!(x.len(), 2 * half);
    for i in 0..half {
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i]     = x0 * cos[i] - x1 * sin[i];
        x[2 * i + 1] = x0 * sin[i] + x1 * cos[i];
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: L∞ distance between two slices.
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
    }

    /// Applying RoPE at position 0 must be the identity transformation
    /// (cos(0)=1, sin(0)=0).
    #[test]
    fn test_rope_position_zero_is_identity() {
        let cache = RopeCache::build(8, 64, 10000.0);
        let original = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut q = original.clone();
        let mut k = original.clone();
        let positions = vec![0u32];
        cache.apply_qk_f32(&mut q, &mut k, &positions, 1, 1, 1);
        assert!(max_abs_diff(&q, &original) < 1e-6, "position-0 q not identity: {q:?}");
        assert!(max_abs_diff(&k, &original) < 1e-6, "position-0 k not identity: {k:?}");
    }

    /// Applying RoPE at position `p` then at position `-p` must recover the
    /// original vector (rotation is orthogonal / invertible).
    ///
    /// RoPE at position `p` followed by RoPE at position `−p` ≡ rotation by
    /// `+angle` then `−angle`, which is the identity.  We model this by using
    /// `apply_qk_f32` twice: once at position `p` (theta scale chosen so
    /// that `p` and `max_seq_len - p` cancel numerically).
    ///
    /// A simpler, exact formulation: rotation(p) ∘ rotation(-p) = Id.
    /// We test this directly on the `rotate_pairs` helper.
    #[test]
    fn test_rope_double_rotation_cancels() {
        let head_dim = 8;
        let max_seq_len = 128;
        let theta = 10000.0_f32;
        let half = head_dim / 2;

        // Build forward (pos=7) and inverse (pos=7 negated analytically) rows.
        let pos = 7usize;
        let mut cos_fwd = vec![0.0_f32; half];
        let mut sin_fwd = vec![0.0_f32; half];
        let mut cos_inv = vec![0.0_f32; half];
        let mut sin_inv = vec![0.0_f32; half];
        for i in 0..half {
            let freq = 1.0_f32 / theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            cos_fwd[i] = angle.cos();
            sin_fwd[i] = angle.sin();
            // cos(-angle) = cos(angle); sin(-angle) = -sin(angle)
            cos_inv[i] = angle.cos();
            sin_inv[i] = -angle.sin();
        }

        let original = vec![1.0_f32, -2.0, 0.5, 3.0, -1.5, 0.25, 7.0, -3.0];
        let mut v = original.clone();

        // Forward rotation at +pos
        rotate_pairs(&mut v, &cos_fwd, &sin_fwd);
        // Inverse rotation at -pos
        rotate_pairs(&mut v, &cos_inv, &sin_inv);

        assert!(
            max_abs_diff(&v, &original) < 1e-5,
            "double rotation did not cancel: original={original:?} result={v:?}"
        );

        // Ensure the build path is exercised (suppress unused warning on max_seq_len).
        let _ = RopeCache::build(head_dim, max_seq_len, theta);
    }

    /// Rotations are independent across heads — applying the cache on two
    /// separate heads should give the same result as applying independently.
    #[test]
    fn test_rope_multi_head_independence() {
        let cache = RopeCache::build(4, 32, 10000.0);
        // batch=1, seq=2, heads=2, head_dim=4
        let mut q = vec![
            // s=0, h=0
            1.0_f32, 2.0, 3.0, 4.0,
            // s=0, h=1
            5.0, 6.0, 7.0, 8.0,
            // s=1, h=0
            0.5, -0.5, 1.5, -1.5,
            // s=1, h=1
            2.5, -2.5, 3.5, -3.5,
        ];
        let mut k = q.clone();
        let positions = vec![0u32, 1];
        cache.apply_qk_f32(&mut q, &mut k, &positions, 1, 2, 2);

        // The two heads at the same sequence position must receive the same rotation.
        // Verify by extracting rotation at pos=0: expect identity.
        assert!(
            (q[0] - 1.0).abs() < 1e-6 && (q[1] - 2.0).abs() < 1e-6,
            "position-0 head-0 mismatch: {:?}", &q[..4]
        );
        assert!(
            (q[4] - 5.0).abs() < 1e-6 && (q[5] - 6.0).abs() < 1e-6,
            "position-0 head-1 mismatch: {:?}", &q[4..8]
        );
    }
}
