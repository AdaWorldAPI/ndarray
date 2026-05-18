#![allow(missing_docs)]

//! Multi-head attention: naive O(N²) and flash-attention-style O(N) memory variant.
//!
//! # Tensor layout
//!
//! All tensors are in **row-major `[batch, seq, heads, head_dim]`** order.
//! The flat index for element `[b, s, h, d]` is:
//!
//! ```text
//! b * (seq * heads * head_dim) + s * (heads * head_dim) + h * head_dim + d
//! ```
//!
//! # Algorithms
//!
//! * [`attention_f32`] — Classic softmax(Q·Kᵀ/√d + mask)·V.  Materialises
//!   the full `[batch, heads, seq, seq]` score matrix; memory is O(N²).
//!
//! * [`flash_attention_f32`] — Tile-wise online-softmax following Dao (2022).
//!   Memory is O(N · block_size); numerics match the naive path within 1e-5.
//!
//! # References
//!
//! Dao et al. (2022) "FlashAttention: Fast and Memory-Efficient Exact Attention
//! with IO-Awareness". <https://arxiv.org/abs/2205.14135>

use crate::hpc::activations::softmax_f32;
use crate::{Array1, ArrayView1, ArrayViewMut1};

// ============================================================================
// Public types
// ============================================================================

/// Configuration shared by all attention variants.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::linalg::attention::AttentionConfig;
///
/// let cfg = AttentionConfig { num_heads: 8, head_dim: 64, causal_mask: true };
/// ```
pub struct AttentionConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimensionality of each head.
    pub head_dim: usize,
    /// If `true`, apply a causal (lower-triangular) mask so token `i` only
    /// attends to tokens `j ≤ i`.
    pub causal_mask: bool,
}

// ============================================================================
// Naive O(N²) attention
// ============================================================================

/// Naive multi-head attention: `softmax(Q·Kᵀ/√d + mask) · V`.
///
/// The full `[seq, seq]` score matrix is materialised per (batch, head).
/// This is correct but requires O(N²) memory.
///
/// # Tensor layout
///
/// `q`, `k`, `v`, `out` are all in `[batch, seq, heads, head_dim]` row-major
/// order (flat `&[f32]`).
///
/// # Arguments
///
/// * `q`      — query tensor.
/// * `k`      — key tensor.
/// * `v`      — value tensor.
/// * `out`    — output tensor; written in place (same shape as `q`).
/// * `config` — head count, head dim, and causal-mask flag.
/// * `batch`  — batch size B.
/// * `seq`    — sequence length S.
///
/// # Panics
///
/// Panics if any slice length does not match `batch * seq * heads * head_dim`.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::linalg::attention::{attention_f32, AttentionConfig};
///
/// // All seq positions share the same constant Q, K, V row → uniform weights
/// // → output equals V (softmax of constants identity).
/// let b = 1; let s = 4; let h = 1; let d = 4;
/// let row = vec![1.0_f32, 0.5, 0.25, 0.125];
/// let mut v = vec![0.0_f32; b * s * h * d];
/// for si in 0..s { v[si * d..][..d].copy_from_slice(&row); }
/// let q = v.clone(); let k = v.clone();
/// let mut out = vec![0.0_f32; b * s * h * d];
/// let cfg = AttentionConfig { num_heads: h, head_dim: d, causal_mask: false };
/// attention_f32(&q, &k, &v, &mut out, &cfg, b, s);
/// for (o, vi) in out.iter().zip(v.iter()) {
///     assert!((o - vi).abs() < 1e-5, "output differs from V");
/// }
/// ```
pub fn attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    config: &AttentionConfig,
    batch: usize,
    seq: usize,
) {
    let AttentionConfig { num_heads, head_dim, causal_mask } = *config;
    let token_stride = num_heads * head_dim;
    let total = batch * seq * token_stride;
    assert_eq!(q.len(), total, "q length mismatch");
    assert_eq!(k.len(), total, "k length mismatch");
    assert_eq!(v.len(), total, "v length mismatch");
    assert_eq!(out.len(), total, "out length mismatch");

    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Temporary buffers per (batch, head) pass — seq×seq scores and softmax.
    let mut scores  = vec![0.0_f32; seq * seq];
    let mut attn_buf = vec![0.0_f32; seq * seq];

    for b in 0..batch {
        for h in 0..num_heads {
            // ----------------------------------------------------------------
            // 1. Compute scores[i, j] = dot(Q[b,i,h,:], K[b,j,h,:]) * scale
            // ----------------------------------------------------------------
            for i in 0..seq {
                let qi_base = (b * seq * num_heads + i * num_heads + h) * head_dim;
                for j in 0..seq {
                    let kj_base = (b * seq * num_heads + j * num_heads + h) * head_dim;
                    let dot: f32 = q[qi_base..qi_base + head_dim]
                        .iter()
                        .zip(k[kj_base..kj_base + head_dim].iter())
                        .map(|(a, bv)| a * bv)
                        .sum();
                    let s = dot * scale;
                    scores[i * seq + j] =
                        if causal_mask && j > i { f32::NEG_INFINITY } else { s };
                }
            }

            // ----------------------------------------------------------------
            // 2. Softmax each row of the score matrix.
            // ----------------------------------------------------------------
            for i in 0..seq {
                let row: Array1<f32> = Array1::from(scores[i * seq..(i + 1) * seq].to_vec());
                let row_view: ArrayView1<f32> = row.view();
                let attn_slice = &mut attn_buf[i * seq..(i + 1) * seq];
                let mut attn_arr: Array1<f32> = Array1::zeros(seq);
                {
                    let attn_view_mut: ArrayViewMut1<f32> = attn_arr.view_mut();
                    softmax_f32(row_view, attn_view_mut);
                }
                attn_slice.copy_from_slice(attn_arr.as_slice().unwrap());
            }

            // ----------------------------------------------------------------
            // 3. Output[b, i, h, :] = Σⱼ attn[i,j] · V[b, j, h, :]
            // ----------------------------------------------------------------
            for i in 0..seq {
                let out_base = (b * seq * num_heads + i * num_heads + h) * head_dim;
                let out_slice = &mut out[out_base..out_base + head_dim];
                out_slice.fill(0.0);
                for j in 0..seq {
                    let vj_base = (b * seq * num_heads + j * num_heads + h) * head_dim;
                    let a = attn_buf[i * seq + j];
                    for d in 0..head_dim {
                        out_slice[d] += a * v[vj_base + d];
                    }
                }
            }
        }
    }
}

// ============================================================================
// Flash-attention O(N·block) memory variant
// ============================================================================

/// Flash-attention-style tiled attention with O(N) memory.
///
/// Uses Dao (2022)'s online-softmax tile scheme to avoid materialising the
/// full `seq × seq` score matrix.  Numerics match [`attention_f32`] within
/// 1e-5 for typical inputs.
///
/// # Tile scheme
///
/// For each query block `Bᵢ` of size `block_size`:
///  1. For each key/value block `Bⱼ` of size `block_size`:
///     * Compute tile scores `S = Q_block · K_blockᵀ / √d`.
///     * Apply causal mask within the tile.
///     * Update the running online-softmax statistics `(m, l)` and accumulate
///       the output numerator.
///  2. Divide the accumulated numerator by the final denominator `l` to get
///     the normalised output.
///
/// # Arguments
///
/// Same as [`attention_f32`], plus:
///
/// * `block_size` — tile size (number of tokens per block).  Typical value: 64.
///   Must be ≥ 1.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::linalg::attention::{flash_attention_f32, AttentionConfig};
///
/// let b = 1; let s = 8; let h = 1; let d = 4;
/// let q = vec![1.0_f32; b * s * h * d];
/// let k = q.clone();
/// let v = q.clone();
/// let mut out = vec![0.0_f32; b * s * h * d];
/// let cfg = AttentionConfig { num_heads: h, head_dim: d, causal_mask: false };
/// flash_attention_f32(&q, &k, &v, &mut out, &cfg, b, s, 4);
/// for (o, vi) in out.iter().zip(v.iter()) {
///     assert!((o - vi).abs() < 1e-5, "flash output differs from V");
/// }
/// ```
pub fn flash_attention_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    config: &AttentionConfig,
    batch: usize,
    seq: usize,
    block_size: usize,
) {
    assert!(block_size >= 1, "block_size must be >= 1");
    let AttentionConfig { num_heads, head_dim, causal_mask } = *config;
    let token_stride = num_heads * head_dim;
    let total = batch * seq * token_stride;
    assert_eq!(q.len(), total, "q length mismatch");
    assert_eq!(k.len(), total, "k length mismatch");
    assert_eq!(v.len(), total, "v length mismatch");
    assert_eq!(out.len(), total, "out length mismatch");

    let scale = 1.0_f32 / (head_dim as f32).sqrt();

    // Per-tile scratch: score tile is at most block_size×block_size.
    let tile_area = block_size * block_size;
    let mut score_tile = vec![0.0_f32; tile_area];

    for b in 0..batch {
        for h in 0..num_heads {
            // Iterate over query blocks.
            let mut qi_start = 0;
            while qi_start < seq {
                let qi_end = (qi_start + block_size).min(seq);
                let qi_len = qi_end - qi_start;

                // Running online-softmax state per query row in the block:
                //   m[i]   = running maximum (initialised to −∞)
                //   l[i]   = running sum of exp(s − m[i])
                //   acc[i] = unnormalised output accumulator (head_dim values)
                let mut m   = vec![f32::NEG_INFINITY; qi_len];
                let mut l   = vec![0.0_f32; qi_len];
                let mut acc = vec![0.0_f32; qi_len * head_dim];

                // Iterate over key/value blocks.
                let mut kj_start = 0;
                while kj_start < seq {
                    let kj_end = (kj_start + block_size).min(seq);
                    let kj_len = kj_end - kj_start;

                    // -----------------------------------------------------------
                    // Compute score tile S[qi, kj] = Q[qi] · K[kj]ᵀ * scale.
                    // -----------------------------------------------------------
                    for qi in 0..qi_len {
                        let global_qi = qi_start + qi;
                        let q_base =
                            (b * seq * num_heads + global_qi * num_heads + h) * head_dim;
                        for kj in 0..kj_len {
                            let global_kj = kj_start + kj;
                            let k_base =
                                (b * seq * num_heads + global_kj * num_heads + h) * head_dim;
                            let dot: f32 = q[q_base..q_base + head_dim]
                                .iter()
                                .zip(k[k_base..k_base + head_dim].iter())
                                .map(|(a, bv)| a * bv)
                                .sum();
                            let s = dot * scale;
                            score_tile[qi * block_size + kj] =
                                if causal_mask && global_kj > global_qi {
                                    f32::NEG_INFINITY
                                } else {
                                    s
                                };
                        }
                    }

                    // -----------------------------------------------------------
                    // Online softmax update (Dao 2022, Algorithm 1).
                    //
                    // For each query row qi:
                    //   m_new = max(m_old, row_max)
                    //   l_new = exp(m_old − m_new) * l_old + Σⱼ exp(s[j] − m_new)
                    //   acc_new = exp(m_old − m_new)*acc_old + Σⱼ exp(s[j]−m_new)*v[j]
                    // -----------------------------------------------------------
                    for qi in 0..qi_len {
                        let row_max: f32 = (0..kj_len)
                            .map(|kj| score_tile[qi * block_size + kj])
                            .fold(f32::NEG_INFINITY, f32::max);

                        let m_new = m[qi].max(row_max);
                        let correction = (m[qi] - m_new).exp();
                        let mut l_new = l[qi] * correction;

                        let acc_base = qi * head_dim;
                        for d in 0..head_dim {
                            acc[acc_base + d] *= correction;
                        }

                        for kj in 0..kj_len {
                            let global_kj = kj_start + kj;
                            let s = score_tile[qi * block_size + kj];
                            if s.is_infinite() && s < 0.0 {
                                continue; // causal-masked out
                            }
                            let e = (s - m_new).exp();
                            l_new += e;
                            let v_base =
                                (b * seq * num_heads + global_kj * num_heads + h) * head_dim;
                            for d in 0..head_dim {
                                acc[acc_base + d] += e * v[v_base + d];
                            }
                        }

                        m[qi] = m_new;
                        l[qi] = l_new;
                    }

                    kj_start = kj_end;
                }

                // -----------------------------------------------------------
                // Finalise: divide by l to normalise.
                // -----------------------------------------------------------
                for qi in 0..qi_len {
                    let global_qi = qi_start + qi;
                    let out_base =
                        (b * seq * num_heads + global_qi * num_heads + h) * head_dim;
                    let acc_base = qi * head_dim;
                    let l_val = l[qi].max(1e-30);
                    for d in 0..head_dim {
                        out[out_base + d] = acc[acc_base + d] / l_val;
                    }
                }

                qi_start = qi_end;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
    }

    // -------------------------------------------------------------------------
    // Gate 1: Identity — uniform Q=K=V → output equals V (softmax of constants).
    // -------------------------------------------------------------------------

    /// When all token vectors in Q and K are **identical** (same constant
    /// row repeated S times), the score matrix is a constant matrix and
    /// softmax yields uniform weights `1/S`.  The output is then the
    /// uniform average of V rows.  If every V row is also the same constant
    /// vector, the average equals that vector — i.e., output = V.
    ///
    /// This is the "softmax of constants" identity stated in the design doc.
    #[test]
    fn test_naive_identity_qkv() {
        let (b, s, h, d) = (1, 4, 2, 8);
        // One representative row per head (constant across all seq positions).
        let row_h0 = vec![0.5_f32, 0.3, 0.7, 0.1, 0.9, 0.2, 0.4, 0.6];
        let row_h1 = vec![1.0_f32, 0.0, 0.5, 0.5, 0.25, 0.75, 0.1, 0.9];

        // Build full tensors: layout [batch=1, seq, heads, head_dim]
        let mut v = vec![0.0_f32; b * s * h * d];
        for si in 0..s {
            // head 0
            v[(si * h + 0) * d..][..d].copy_from_slice(&row_h0);
            // head 1
            v[(si * h + 1) * d..][..d].copy_from_slice(&row_h1);
        }
        let q = v.clone(); // same as V — all rows identical within each head
        let k = v.clone();
        let mut out = vec![0.0_f32; b * s * h * d];
        let cfg = AttentionConfig { num_heads: h, head_dim: d, causal_mask: false };
        attention_f32(&q, &k, &v, &mut out, &cfg, b, s);
        // With uniform weights and identical V rows, output = V exactly.
        assert!(
            max_abs_diff(&out, &v) < 1e-5,
            "naive identity failed: max_err={}", max_abs_diff(&out, &v)
        );
    }

    #[test]
    fn test_flash_identity_qkv() {
        let (b, s, h, d) = (1, 8, 1, 4);
        let row = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut v = vec![0.0_f32; b * s * h * d];
        for si in 0..s {
            v[si * d..][..d].copy_from_slice(&row);
        }
        let q = v.clone();
        let k = v.clone();
        let mut out = vec![0.0_f32; b * s * h * d];
        let cfg = AttentionConfig { num_heads: h, head_dim: d, causal_mask: false };
        flash_attention_f32(&q, &k, &v, &mut out, &cfg, b, s, 4);
        assert!(
            max_abs_diff(&out, &v) < 1e-5,
            "flash identity failed: max_err={}", max_abs_diff(&out, &v)
        );
    }

    // -------------------------------------------------------------------------
    // Gate 2: Causal mask — token i only attends to tokens ≤ i.
    // -------------------------------------------------------------------------

    /// Build Q = all ones (uniform Q·K) and V[j, 0] = j+1.
    /// With causal attention, token i's output[i, 0] = average(1..=i+1).
    #[test]
    fn test_naive_causal_mask() {
        let (b, s, h, d) = (1, 6, 1, 4);
        let q = vec![1.0_f32; b * s * h * d];
        let k = q.clone();
        let mut v = vec![0.0_f32; b * s * h * d];
        for j in 0..s {
            v[(j * h + 0) * d + 0] = (j + 1) as f32;
        }
        let mut out = vec![0.0_f32; b * s * h * d];
        let cfg = AttentionConfig { num_heads: h, head_dim: d, causal_mask: true };
        attention_f32(&q, &k, &v, &mut out, &cfg, b, s);
        for i in 0..s {
            let expected = (1..=(i + 1)).map(|x| x as f32).sum::<f32>() / (i + 1) as f32;
            let actual = out[(i * h + 0) * d + 0];
            assert!(
                (actual - expected).abs() < 1e-4,
                "naive causal: token {i} expected {expected:.4} got {actual:.4}"
            );
        }
    }

    #[test]
    fn test_flash_causal_mask() {
        let (b, s, h, d) = (1, 6, 1, 4);
        let q = vec![1.0_f32; b * s * h * d];
        let k = q.clone();
        let mut v = vec![0.0_f32; b * s * h * d];
        for j in 0..s {
            v[(j * h + 0) * d + 0] = (j + 1) as f32;
        }
        let mut out = vec![0.0_f32; b * s * h * d];
        let cfg = AttentionConfig { num_heads: h, head_dim: d, causal_mask: true };
        flash_attention_f32(&q, &k, &v, &mut out, &cfg, b, s, 3);
        for i in 0..s {
            let expected = (1..=(i + 1)).map(|x| x as f32).sum::<f32>() / (i + 1) as f32;
            let actual = out[(i * h + 0) * d + 0];
            assert!(
                (actual - expected).abs() < 1e-4,
                "flash causal: token {i} expected {expected:.4} got {actual:.4}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Gate 3: Parity — naive vs flash agree within 1e-5 (seq=8, head_dim=4).
    // -------------------------------------------------------------------------

    #[test]
    fn test_naive_vs_flash_parity() {
        let (b, s, h, d) = (1, 8, 2, 4);
        let total = b * s * h * d;
        let mut seed = 42u64;
        let mut next = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let q: Vec<f32> = (0..total).map(|_| next()).collect();
        let k: Vec<f32> = (0..total).map(|_| next()).collect();
        let v: Vec<f32> = (0..total).map(|_| next()).collect();

        let mut out_naive = vec![0.0_f32; total];
        let mut out_flash = vec![0.0_f32; total];

        attention_f32(&q, &k, &v, &mut out_naive,
            &AttentionConfig { num_heads: h, head_dim: d, causal_mask: false }, b, s);
        flash_attention_f32(&q, &k, &v, &mut out_flash,
            &AttentionConfig { num_heads: h, head_dim: d, causal_mask: false }, b, s, 4);

        let err = max_abs_diff(&out_naive, &out_flash);
        assert!(err < 1e-5, "naive vs flash parity > 1e-5: max_err={err}");
    }

    #[test]
    fn test_naive_vs_flash_parity_causal() {
        let (b, s, h, d) = (2, 8, 2, 4);
        let total = b * s * h * d;
        let mut seed = 99u64;
        let mut next = || -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };
        let q: Vec<f32> = (0..total).map(|_| next()).collect();
        let k: Vec<f32> = (0..total).map(|_| next()).collect();
        let v: Vec<f32> = (0..total).map(|_| next()).collect();

        let mut out_naive = vec![0.0_f32; total];
        let mut out_flash = vec![0.0_f32; total];

        attention_f32(&q, &k, &v, &mut out_naive,
            &AttentionConfig { num_heads: h, head_dim: d, causal_mask: true }, b, s);
        flash_attention_f32(&q, &k, &v, &mut out_flash,
            &AttentionConfig { num_heads: h, head_dim: d, causal_mask: true }, b, s, 4);

        let err = max_abs_diff(&out_naive, &out_flash);
        assert!(err < 1e-5, "causal naive vs flash parity > 1e-5: max_err={err}");
    }
}
