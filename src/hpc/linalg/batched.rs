//! Batched GEMM — loops over batch / head axes calling the backend GEMM kernel.
//!
//! # Layout conventions
//!
//! All matrices are **row-major** (`C` order).  Slices are flat, packed:
//! - 3-D tensor: `[batch, m, k]` for `x`, `[batch, k, n]` for `y`, `[batch, m, n]` for `out`.
//! - 4-D tensor: `[batch, heads, m, k]` for `x`, `[batch, heads, k, n]` for `y`,
//!   `[batch, heads, m, n]` for `out` — the attention `[B, H, seq, dim]` shape.
//!
//! # Complexity
//!
//! Both variants are `O(batch * m * k * n)` (or `O(batch * heads * …)` for 4D).
//! Each inner call delegates to `crate::backend::BlasFloat::backend_gemm` which
//! dispatches to the active BLAS backend (native / OpenBLAS / MKL).
//!
//! # Example
//!
//! ```rust
//! use ndarray::hpc::linalg::batched::batched_gemm_f32;
//!
//! // 2 batches of (2×3) · (3×2) = (2×2)
//! let x = vec![1f32, 2., 3., 4., 5., 6.,   // batch 0: [[1,2,3],[4,5,6]]
//!              1f32, 0., 0., 0., 1., 0.];   // batch 1: [[1,0,0],[0,1,0]]
//! let y = vec![1f32, 0., 0., 1., 0., 0.,   // batch 0: [[1,0],[0,1],[0,0]]
//!              2f32, 3., 4., 5., 6., 7.];   // batch 1: [[2,3],[4,5],[6,7]]
//! let mut out = vec![0f32; 2 * 2 * 2];
//! batched_gemm_f32(&x, &y, &mut out, 2, 2, 3, 2, 1.0, 0.0);
//! // batch 0 out[0..4]: [[1,2],[4,5]]  (first 2 rows of x · first two cols of y)
//! assert!((out[0] - 1.0).abs() < 1e-5);
//! ```

use crate::backend::BlasFloat;

// ─────────────────────────────────────────────────────────────────────────────
// 3-D batched GEMM
// ─────────────────────────────────────────────────────────────────────────────

/// Batched matrix multiply: `out[b] = alpha * x[b] * y[b] + beta * out[b]`.
///
/// Tensors are flat row-major slices with the following strides:
/// - `x`   : shape `[batch, m, k]`, stride `m*k` per batch item.
/// - `y`   : shape `[batch, k, n]`, stride `k*n` per batch item.
/// - `out` : shape `[batch, m, n]`, stride `m*n` per batch item.
///
/// # Panics
///
/// Panics if any slice is shorter than required by the shape parameters.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::linalg::batched::batched_gemm_f32;
///
/// let x   = vec![1f32, 0., 0., 1.];          // batch=1, m=2, k=2 identity
/// let y   = vec![3f32, 4., 5., 6.];          // batch=1, k=2, n=2
/// let mut out = vec![0f32; 4];
/// batched_gemm_f32(&x, &y, &mut out, 1, 2, 2, 2, 1.0, 0.0);
/// assert!((out[0] - 3.0).abs() < 1e-5);
/// ```
pub fn batched_gemm_f32(
    x: &[f32], y: &[f32], out: &mut [f32], batch: usize, m: usize, k: usize, n: usize, alpha: f32, beta: f32,
) {
    let x_stride = m * k;
    let y_stride = k * n;
    let o_stride = m * n;

    assert!(
        x.len() >= batch * x_stride,
        "batched_gemm_f32: x too short (need {}, got {})",
        batch * x_stride,
        x.len()
    );
    assert!(
        y.len() >= batch * y_stride,
        "batched_gemm_f32: y too short (need {}, got {})",
        batch * y_stride,
        y.len()
    );
    assert!(
        out.len() >= batch * o_stride,
        "batched_gemm_f32: out too short (need {}, got {})",
        batch * o_stride,
        out.len()
    );

    for b in 0..batch {
        let a_slice = &x[b * x_stride..(b + 1) * x_stride];
        let b_slice = &y[b * y_stride..(b + 1) * y_stride];
        let c_slice = &mut out[b * o_stride..(b + 1) * o_stride];
        // Row-major: leading dims are k (for A) and n (for B and C).
        f32::backend_gemm(m, n, k, alpha, a_slice, k, b_slice, n, beta, c_slice, n);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4-D batched GEMM  (attention layout: [batch, heads, seq, dim])
// ─────────────────────────────────────────────────────────────────────────────

/// 4-D batched GEMM for multi-head attention: `out[b,h] = alpha * x[b,h] * y[b,h] + beta * out[b,h]`.
///
/// Tensors are flat row-major slices with shape:
/// - `x`   : `[batch, heads, m, k]`
/// - `y`   : `[batch, heads, k, n]`
/// - `out` : `[batch, heads, m, n]`
///
/// # Panics
///
/// Panics if any slice is shorter than required by the shape parameters.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::linalg::batched::batched_gemm_4d_f32;
///
/// // 1 batch, 2 heads, (1×2)·(2×1) = (1×1)
/// let x   = vec![1f32, 2.,   3., 4.];   // [1,2,1,2]
/// let y   = vec![5f32, 6.,   7., 8.];   // [1,2,2,1]
/// let mut out = vec![0f32; 2];           // [1,2,1,1]
/// batched_gemm_4d_f32(&x, &y, &mut out, 1, 2, 1, 2, 1, 1.0, 0.0);
/// // head0: [1,2]·[5,6]^T = 17;  head1: [3,4]·[7,8]^T = 53
/// assert!((out[0] - 17.0).abs() < 1e-5);
/// assert!((out[1] - 53.0).abs() < 1e-5);
/// ```
pub fn batched_gemm_4d_f32(
    x: &[f32], y: &[f32], out: &mut [f32], batch: usize, heads: usize, m: usize, k: usize, n: usize, alpha: f32,
    beta: f32,
) {
    let x_head = m * k;
    let y_head = k * n;
    let o_head = m * n;
    let x_batch = heads * x_head;
    let y_batch = heads * y_head;
    let o_batch = heads * o_head;

    assert!(x.len() >= batch * x_batch, "batched_gemm_4d_f32: x too short");
    assert!(y.len() >= batch * y_batch, "batched_gemm_4d_f32: y too short");
    assert!(out.len() >= batch * o_batch, "batched_gemm_4d_f32: out too short");

    for b in 0..batch {
        for h in 0..heads {
            let ax = b * x_batch + h * x_head;
            let ay = b * y_batch + h * y_head;
            let ao = b * o_batch + h * o_head;

            let a_slice = &x[ax..ax + x_head];
            let b_slice = &y[ay..ay + y_head];
            let c_slice = &mut out[ao..ao + o_head];

            f32::backend_gemm(m, n, k, alpha, a_slice, k, b_slice, n, beta, c_slice, n);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: single-GEMM reference for one 2D slice using a plain loop.
    fn ref_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    #[test]
    fn batched_gemm_matches_loop_of_single_gemm() {
        // 3 batches of (2×3)·(3×2) = (2×2)
        let batch = 3;
        let (m, k, n) = (2, 3, 2);

        // Different values per batch to catch index bugs.
        let x: Vec<f32> = (0..batch * m * k).map(|i| i as f32 + 1.0).collect();
        let y: Vec<f32> = (0..batch * k * n).map(|i| (i as f32 * 0.5) + 0.1).collect();
        let mut out = vec![0f32; batch * m * n];

        batched_gemm_f32(&x, &y, &mut out, batch, m, k, n, 1.0, 0.0);

        for b in 0..batch {
            let a_sl = &x[b * m * k..(b + 1) * m * k];
            let b_sl = &y[b * k * n..(b + 1) * k * n];
            let expected = ref_gemm(a_sl, b_sl, m, k, n);
            let got = &out[b * m * n..(b + 1) * m * n];
            for (e, g) in expected.iter().zip(got.iter()) {
                assert!((e - g).abs() < 1e-4, "batch {b}: expected {e}, got {g}");
            }
        }
    }

    #[test]
    fn batched_gemm_4d_matches_loop() {
        // 2 batches, 2 heads, (2×3)·(3×2) = (2×2)
        let (batch, heads, m, k, n) = (2, 2, 2, 3, 2);
        let x: Vec<f32> = (0..batch * heads * m * k).map(|i| i as f32 + 1.0).collect();
        let y: Vec<f32> = (0..batch * heads * k * n)
            .map(|i| (i as f32 * 0.3) + 0.2)
            .collect();
        let mut out4d = vec![0f32; batch * heads * m * n];

        batched_gemm_4d_f32(&x, &y, &mut out4d, batch, heads, m, k, n, 1.0, 0.0);

        // Compare against batched_gemm_f32 treating (batch*heads) as the flat batch.
        let flat_batch = batch * heads;
        let mut out3d = vec![0f32; flat_batch * m * n];
        batched_gemm_f32(&x, &y, &mut out3d, flat_batch, m, k, n, 1.0, 0.0);

        for (a, b) in out4d.iter().zip(out3d.iter()) {
            assert!((a - b).abs() < 1e-5, "4d vs 3d mismatch: {a} vs {b}");
        }
    }
}
