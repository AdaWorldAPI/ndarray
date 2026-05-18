//! Convolution kernels — Conv1D, Conv2D, 3×3/5×5 direct, and im2col+GEMM.
//!
//! # Hierarchy
//!
//! | Function | Method | Notes |
//! |---|---|---|
//! | [`conv1d_f32`] | Direct sliding-window | stride, zero-padding |
//! | [`conv2d_f32`] | Direct O(out·kh·kw·Cin) | general dispatcher |
//! | [`conv2d_3x3_f32`] | Unrolled 3×3 inner loop | fastest for kh=kw=3 |
//! | [`conv2d_5x5_f32`] | Unrolled 5×5 inner loop | fastest for kh=kw=5 |
//! | [`conv2d_im2col_f32`] | im2col + [`crate::backend::gemm_f32`] | large kernels |
//!
//! # Layout conventions
//!
//! - **Conv1D**: `input[i]`, `kernel[j]`, `out[o]` — 1-D flat slices.
//! - **Conv2D input**: channel-first flat, shape `[in_channels, in_h, in_w]`
//!   (`in_channels` contiguous rows of `in_h × in_w`).
//! - **Conv2D kernel**: shape `[out_channels, in_channels, kh, kw]` — row-major.
//! - **Conv2D output**: `[out_channels, out_h, out_w]` — row-major.
//!
//! Output spatial dimensions are computed as:
//! ```text
//! out_h = (in_h + 2*pad_h - kh) / stride_h + 1
//! out_w = (in_w + 2*pad_w - kw) / stride_w + 1
//! ```
//!
//! # Out-of-scope hard boundary
//!
//! No SIMD primitives — those live in `crate::simd`. No `#[target_feature]`.

#![allow(missing_docs)]

use crate::backend::gemm_f32;

// ───────────────────────────────────────────────────────────────────────────
// Conv1D
// ───────────────────────────────────────────────────────────────────────────

/// 1-D convolution — `out[o] = Σ_j input[o*stride + j] * kernel[j]`.
///
/// Zero-padding of `padding` elements is applied symmetrically on both ends
/// of the input before sliding the kernel.
///
/// # Panics
///
/// Panics if `out` is not exactly
/// `(input.len() + 2*padding - kernel.len()) / stride + 1` elements long.
///
/// # Examples
///
/// ```rust
/// use ndarray::hpc::linalg::conv::conv1d_f32;
/// let input  = [1.0_f32, 2.0, 3.0, 4.0];
/// let kernel = [1.0_f32, 0.0, 0.0];
/// let mut out = vec![0.0_f32; 2];
/// conv1d_f32(&input, &kernel, 1, 0, &mut out);
/// assert!((out[0] - 1.0).abs() < 1e-6);
/// assert!((out[1] - 2.0).abs() < 1e-6);
/// ```
pub fn conv1d_f32(input: &[f32], kernel: &[f32], stride: usize, padding: usize, out: &mut [f32]) {
    let in_len = input.len();
    let klen = kernel.len();
    assert!(stride >= 1, "stride must be >= 1");
    assert!(klen >= 1, "kernel must be non-empty");
    let out_len = (in_len + 2 * padding).saturating_sub(klen) / stride + 1;
    assert_eq!(
        out.len(),
        out_len,
        "conv1d_f32: out has wrong length (expected {out_len}, got {})",
        out.len()
    );

    for o in 0..out_len {
        let mut acc = 0.0_f32;
        for j in 0..klen {
            let idx = o * stride + j; // index into padded input
            let padded_val = if idx < padding || idx >= in_len + padding {
                0.0_f32
            } else {
                input[idx - padding]
            };
            acc += padded_val * kernel[j];
        }
        out[o] = acc;
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Internal helpers: output-size computation
// ───────────────────────────────────────────────────────────────────────────

#[inline]
fn out_spatial(in_size: usize, k: usize, stride: usize, pad: usize) -> usize {
    (in_size + 2 * pad).saturating_sub(k) / stride + 1
}

// ───────────────────────────────────────────────────────────────────────────
// Conv2D — general direct path
// ───────────────────────────────────────────────────────────────────────────

/// General 2-D convolution (direct algorithm, channel-first layout).
///
/// - `in_shape`  = `(in_channels, in_h, in_w)`
/// - `kernel_shape` = `(out_channels, in_channels, kh, kw)`
/// - `stride`    = `(stride_h, stride_w)`
/// - `padding`   = `(pad_h, pad_w)` — symmetric zero-padding
/// - `out` must have exactly `out_channels * out_h * out_w` elements.
///
/// For the common 3×3 or 5×5 cases prefer [`conv2d_3x3_f32`] /
/// [`conv2d_5x5_f32`]. For large kernels prefer [`conv2d_im2col_f32`].
///
/// # Examples
///
/// ```rust
/// use ndarray::hpc::linalg::conv::conv2d_f32;
/// // 1-channel 4×4 input, 1-channel 3×3 all-ones kernel, stride=1, pad=0
/// let input  = vec![1.0_f32; 16]; // 4×4
/// let kernel = vec![1.0_f32; 9];  // 3×3 all-ones → sums 3×3 patches
/// let mut out = vec![0.0_f32; 4]; // 2×2
/// conv2d_f32(&input, (1, 4, 4), &kernel, (1, 1, 3, 3), (1, 1), (0, 0), &mut out);
/// for v in &out { assert!((*v - 9.0).abs() < 1e-5); }
/// ```
pub fn conv2d_f32(
    input: &[f32],
    in_shape: (usize, usize, usize),
    kernel: &[f32],
    kernel_shape: (usize, usize, usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    out: &mut [f32],
) {
    let (cin, in_h, in_w) = in_shape;
    let (cout, _cin_k, kh, kw) = kernel_shape;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    debug_assert_eq!(_cin_k, cin, "kernel in_channels must match input in_channels");

    let out_h = out_spatial(in_h, kh, sh, ph);
    let out_w = out_spatial(in_w, kw, sw, pw);

    assert_eq!(
        out.len(),
        cout * out_h * out_w,
        "conv2d_f32: out buffer length mismatch"
    );

    // Kernel element k[oc, ic, ky, kx] at flat index:
    //   oc*(cin*kh*kw) + ic*(kh*kw) + ky*kw + kx
    let k_ic_stride = kh * kw;
    let k_oc_stride = cin * k_ic_stride;

    // Output element [oc, oh, ow]:
    let out_oc_stride = out_h * out_w;

    for oc in 0..cout {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = 0.0_f32;
                for ic in 0..cin {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let ih = oh * sh + ky; // index into padded input
                            let iw = ow * sw + kx;
                            let val = if ih < ph || iw < pw || ih - ph >= in_h || iw - pw >= in_w {
                                0.0_f32
                            } else {
                                input[ic * (in_h * in_w) + (ih - ph) * in_w + (iw - pw)]
                            };
                            acc += val * kernel[oc * k_oc_stride + ic * k_ic_stride + ky * kw + kx];
                        }
                    }
                }
                out[oc * out_oc_stride + oh * out_w + ow] = acc;
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Conv2D — 3×3 specialized direct path
// ───────────────────────────────────────────────────────────────────────────

/// Specialized 2-D convolution for a **3×3** kernel (direct, fully unrolled inner loop).
///
/// Signature is identical to [`conv2d_f32`]; `kernel_shape` must have
/// `kh = kw = 3`. Panics otherwise.
///
/// The unrolled 9-multiply inner loop allows the compiler to schedule
/// independent FMAs and produces better throughput than the general loop.
///
/// # Examples
///
/// ```rust
/// use ndarray::hpc::linalg::conv::conv2d_3x3_f32;
/// let input  = vec![1.0_f32; 25]; // 5×5
/// let kernel = vec![1.0_f32; 9];  // 3×3 all-ones
/// let mut out = vec![0.0_f32; 9]; // 3×3
/// conv2d_3x3_f32(&input, (1, 5, 5), &kernel, (1, 1, 3, 3), (1, 1), (0, 0), &mut out);
/// for v in &out { assert!((*v - 9.0).abs() < 1e-5); }
/// ```
pub fn conv2d_3x3_f32(
    input: &[f32],
    in_shape: (usize, usize, usize),
    kernel: &[f32],
    kernel_shape: (usize, usize, usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    out: &mut [f32],
) {
    let (cin, in_h, in_w) = in_shape;
    let (cout, _cin_k, kh, kw) = kernel_shape;
    assert_eq!(kh, 3, "conv2d_3x3_f32: kernel height must be 3");
    assert_eq!(kw, 3, "conv2d_3x3_f32: kernel width must be 3");
    debug_assert_eq!(_cin_k, cin, "kernel in_channels must match input in_channels");

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let out_h = out_spatial(in_h, 3, sh, ph);
    let out_w = out_spatial(in_w, 3, sw, pw);

    assert_eq!(out.len(), cout * out_h * out_w, "conv2d_3x3_f32: out buffer length mismatch");

    let k_ic_stride = 9_usize; // kh*kw = 9
    let k_oc_stride = cin * k_ic_stride;
    let out_oc_stride = out_h * out_w;
    let in_hw = in_h * in_w;

    for oc in 0..cout {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = 0.0_f32;
                for ic in 0..cin {
                    let k_base = oc * k_oc_stride + ic * k_ic_stride;
                    let k = &kernel[k_base..k_base + 9];
                    let in_base = ic * in_hw;

                    // Inline helper: sample padded input
                    let sample = |dy: usize, dx: usize| -> f32 {
                        let ih = oh * sh + dy;
                        let iw = ow * sw + dx;
                        if ih < ph || iw < pw || ih - ph >= in_h || iw - pw >= in_w {
                            0.0_f32
                        } else {
                            input[in_base + (ih - ph) * in_w + (iw - pw)]
                        }
                    };

                    // Unrolled 3×3
                    acc += sample(0, 0) * k[0];
                    acc += sample(0, 1) * k[1];
                    acc += sample(0, 2) * k[2];
                    acc += sample(1, 0) * k[3];
                    acc += sample(1, 1) * k[4];
                    acc += sample(1, 2) * k[5];
                    acc += sample(2, 0) * k[6];
                    acc += sample(2, 1) * k[7];
                    acc += sample(2, 2) * k[8];
                }
                out[oc * out_oc_stride + oh * out_w + ow] = acc;
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Conv2D — 5×5 specialized direct path
// ───────────────────────────────────────────────────────────────────────────

/// Specialized 2-D convolution for a **5×5** kernel (direct, fully unrolled inner loop).
///
/// Signature is identical to [`conv2d_f32`]; `kernel_shape` must have
/// `kh = kw = 5`. Panics otherwise.
///
/// # Examples
///
/// ```rust
/// use ndarray::hpc::linalg::conv::conv2d_5x5_f32;
/// let input  = vec![1.0_f32; 49]; // 7×7
/// let kernel = vec![1.0_f32; 25]; // 5×5 all-ones
/// let mut out = vec![0.0_f32; 9]; // 3×3
/// conv2d_5x5_f32(&input, (1, 7, 7), &kernel, (1, 1, 5, 5), (1, 1), (0, 0), &mut out);
/// for v in &out { assert!((*v - 25.0).abs() < 1e-4); }
/// ```
pub fn conv2d_5x5_f32(
    input: &[f32],
    in_shape: (usize, usize, usize),
    kernel: &[f32],
    kernel_shape: (usize, usize, usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    out: &mut [f32],
) {
    let (cin, in_h, in_w) = in_shape;
    let (cout, _cin_k, kh, kw) = kernel_shape;
    assert_eq!(kh, 5, "conv2d_5x5_f32: kernel height must be 5");
    assert_eq!(kw, 5, "conv2d_5x5_f32: kernel width must be 5");
    debug_assert_eq!(_cin_k, cin, "kernel in_channels must match input in_channels");

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let out_h = out_spatial(in_h, 5, sh, ph);
    let out_w = out_spatial(in_w, 5, sw, pw);

    assert_eq!(out.len(), cout * out_h * out_w, "conv2d_5x5_f32: out buffer length mismatch");

    let k_ic_stride = 25_usize; // kh*kw = 25
    let k_oc_stride = cin * k_ic_stride;
    let out_oc_stride = out_h * out_w;
    let in_hw = in_h * in_w;

    for oc in 0..cout {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = 0.0_f32;
                for ic in 0..cin {
                    let k_base = oc * k_oc_stride + ic * k_ic_stride;
                    let k = &kernel[k_base..k_base + 25];
                    let in_base = ic * in_hw;

                    let sample = |dy: usize, dx: usize| -> f32 {
                        let ih = oh * sh + dy;
                        let iw = ow * sw + dx;
                        if ih < ph || iw < pw || ih - ph >= in_h || iw - pw >= in_w {
                            0.0_f32
                        } else {
                            input[in_base + (ih - ph) * in_w + (iw - pw)]
                        }
                    };

                    // Unrolled 5×5 = 25 FMAs
                    acc += sample(0, 0) * k[0];
                    acc += sample(0, 1) * k[1];
                    acc += sample(0, 2) * k[2];
                    acc += sample(0, 3) * k[3];
                    acc += sample(0, 4) * k[4];
                    acc += sample(1, 0) * k[5];
                    acc += sample(1, 1) * k[6];
                    acc += sample(1, 2) * k[7];
                    acc += sample(1, 3) * k[8];
                    acc += sample(1, 4) * k[9];
                    acc += sample(2, 0) * k[10];
                    acc += sample(2, 1) * k[11];
                    acc += sample(2, 2) * k[12];
                    acc += sample(2, 3) * k[13];
                    acc += sample(2, 4) * k[14];
                    acc += sample(3, 0) * k[15];
                    acc += sample(3, 1) * k[16];
                    acc += sample(3, 2) * k[17];
                    acc += sample(3, 3) * k[18];
                    acc += sample(3, 4) * k[19];
                    acc += sample(4, 0) * k[20];
                    acc += sample(4, 1) * k[21];
                    acc += sample(4, 2) * k[22];
                    acc += sample(4, 3) * k[23];
                    acc += sample(4, 4) * k[24];
                }
                out[oc * out_oc_stride + oh * out_w + ow] = acc;
            }
        }
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Conv2D — im2col + GEMM path
// ───────────────────────────────────────────────────────────────────────────

/// General 2-D convolution via **im2col + GEMM** (delegates to
/// [`crate::backend::gemm_f32`]).
///
/// im2col reshapes the input patches into a matrix `col` of shape
/// `[cin*kh*kw, out_h*out_w]`, then computes:
/// ```text
/// out_mat = kernel_mat × col
/// ```
/// where `kernel_mat` is `[cout, cin*kh*kw]`.
///
/// This path allocates a temporary `col` buffer on the heap
/// (`cin * kh * kw * out_h * out_w` f32 values). It is most efficient when
/// the kernel is large relative to the output spatial size.
///
/// # Examples
///
/// ```rust
/// use ndarray::hpc::linalg::conv::conv2d_im2col_f32;
/// let input  = vec![1.0_f32; 16]; // 1×4×4
/// let kernel = vec![1.0_f32; 9];  // 1×1×3×3
/// let mut out = vec![0.0_f32; 4]; // 1×2×2
/// conv2d_im2col_f32(&input, (1, 4, 4), &kernel, (1, 1, 3, 3), (1, 1), (0, 0), &mut out);
/// for v in &out { assert!((*v - 9.0).abs() < 1e-5); }
/// ```
pub fn conv2d_im2col_f32(
    input: &[f32],
    in_shape: (usize, usize, usize),
    kernel: &[f32],
    kernel_shape: (usize, usize, usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    out: &mut [f32],
) {
    let (cin, in_h, in_w) = in_shape;
    let (cout, _cin_k, kh, kw) = kernel_shape;
    debug_assert_eq!(_cin_k, cin);

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let out_h = out_spatial(in_h, kh, sh, ph);
    let out_w = out_spatial(in_w, kw, sw, pw);
    let n_patches = out_h * out_w; // number of output spatial positions
    let patch_len = cin * kh * kw; // number of values per patch column

    assert_eq!(out.len(), cout * out_h * out_w, "conv2d_im2col_f32: out buffer length mismatch");

    // ── im2col: build col matrix [patch_len, n_patches] (row-major) ──────────
    //
    // col[c_patch * n_patches + p] = input pixel at position (ic, ih-ph, iw-pw)
    // for patch p = (oh, ow) and channel-row-col (ic, ky, kx).
    let col_len = patch_len * n_patches;
    let mut col = vec![0.0_f32; col_len];

    let in_hw = in_h * in_w;
    for ic in 0..cin {
        for ky in 0..kh {
            for kx in 0..kw {
                let row = ic * kh * kw + ky * kw + kx; // row in col matrix
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let col_idx = oh * out_w + ow; // column in col matrix
                        let ih = oh * sh + ky;
                        let iw = ow * sw + kx;
                        let val = if ih < ph || iw < pw || ih - ph >= in_h || iw - pw >= in_w {
                            0.0_f32
                        } else {
                            input[ic * in_hw + (ih - ph) * in_w + (iw - pw)]
                        };
                        col[row * n_patches + col_idx] = val;
                    }
                }
            }
        }
    }

    // ── GEMM: out_mat[cout, n_patches] = kernel_mat[cout, patch_len] × col[patch_len, n_patches]
    //
    // m = cout, n = n_patches, k = patch_len
    // lda = patch_len (kernel rows), ldb = n_patches (col rows), ldc = n_patches
    if cout == 0 || n_patches == 0 || patch_len == 0 {
        return;
    }
    gemm_f32(
        cout,
        n_patches,
        patch_len,
        1.0,
        kernel,
        patch_len,
        &col,
        n_patches,
        0.0,
        out,
        n_patches,
    );
}

// ───────────────────────────────────────────────────────────────────────────
// Tests
// ───────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    // Gate 1: conv1d identity — kernel=[1,0,0] shifts input left by 0
    #[test]
    fn test_conv1d_identity_kernel() {
        // kernel [1, 0, 0]: out[o] = input[o]*1 + input[o+1]*0 + input[o+2]*0
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let kernel = [1.0_f32, 0.0, 0.0];
        // out_len = (5 + 0 - 3)/1 + 1 = 3
        let mut out = vec![0.0_f32; 3];
        conv1d_f32(&input, &kernel, 1, 0, &mut out);
        assert!((out[0] - 1.0).abs() < EPS, "out[0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < EPS, "out[1]={}", out[1]);
        assert!((out[2] - 3.0).abs() < EPS, "out[2]={}", out[2]);
    }

    // Gate 2: conv2d_3x3 all-ones kernel sums 3×3 neighbourhood
    #[test]
    fn test_conv2d_3x3_sum_kernel() {
        // 1-channel 5×5 input filled with 1.0; 3×3 all-ones kernel → each out = 9.0
        let input = vec![1.0_f32; 25];
        let kernel = vec![1.0_f32; 9];
        let mut out = vec![0.0_f32; 9]; // 3×3 output
        conv2d_3x3_f32(&input, (1, 5, 5), &kernel, (1, 1, 3, 3), (1, 1), (0, 0), &mut out);
        for (i, v) in out.iter().enumerate() {
            assert!((v - 9.0).abs() < EPS, "out[{i}]={v}");
        }
    }

    // Gate 3: conv2d_im2col vs conv2d_3x3 parity on a 3×3 kernel
    #[test]
    fn test_conv2d_im2col_vs_direct_3x3_parity() {
        // 2-channel 6×6 input, 3-channel 3×3 kernel, stride=1, pad=0 → 3×4×4 output
        let cin = 2;
        let cout = 3;
        let in_h = 6;
        let in_w = 6;
        let kh = 3;
        let kw = 3;
        let n_input = cin * in_h * in_w;
        let n_kernel = cout * cin * kh * kw;
        let out_h = out_spatial(in_h, kh, 1, 0);
        let out_w = out_spatial(in_w, kw, 1, 0);
        let n_out = cout * out_h * out_w;

        // Deterministic fill using index mod 7
        let input: Vec<f32> = (0..n_input).map(|i| (i % 7) as f32 * 0.5).collect();
        let kernel: Vec<f32> = (0..n_kernel).map(|i| ((i + 1) % 5) as f32 * 0.3 - 0.3).collect();

        let mut out_direct = vec![0.0_f32; n_out];
        let mut out_im2col = vec![0.0_f32; n_out];

        conv2d_3x3_f32(
            &input,
            (cin, in_h, in_w),
            &kernel,
            (cout, cin, kh, kw),
            (1, 1),
            (0, 0),
            &mut out_direct,
        );
        conv2d_im2col_f32(
            &input,
            (cin, in_h, in_w),
            &kernel,
            (cout, cin, kh, kw),
            (1, 1),
            (0, 0),
            &mut out_im2col,
        );

        for (i, (a, b)) in out_direct.iter().zip(out_im2col.iter()).enumerate() {
            assert!(
                (a - b).abs() < EPS,
                "mismatch at index {i}: direct={a} im2col={b}"
            );
        }
    }

    // Gate 4: conv2d stride=2 produces half-size spatial output
    #[test]
    fn test_conv2d_stride2_output_size() {
        // 1-channel 8×8 input, 3×3 kernel, stride=2, pad=0 → 3×3 output
        let in_h = 8_usize;
        let in_w = 8_usize;
        let kh = 3_usize;
        let kw = 3_usize;
        let stride = (2_usize, 2_usize);
        let padding = (0_usize, 0_usize);
        let out_h = out_spatial(in_h, kh, stride.0, padding.0);
        let out_w = out_spatial(in_w, kw, stride.1, padding.1);
        assert_eq!(out_h, 3);
        assert_eq!(out_w, 3);

        let input = vec![1.0_f32; in_h * in_w];
        let kernel = vec![1.0_f32; kh * kw]; // all-ones → each out = 9.0
        let mut out = vec![0.0_f32; out_h * out_w];
        conv2d_f32(
            &input,
            (1, in_h, in_w),
            &kernel,
            (1, 1, kh, kw),
            stride,
            padding,
            &mut out,
        );
        for v in &out {
            assert!((v - 9.0).abs() < EPS, "v={v}");
        }
    }

    // Gate 5: conv2d padding=1 preserves spatial dims for 3×3 kernel
    #[test]
    fn test_conv2d_padding1_preserves_spatial_dims() {
        let in_h = 5_usize;
        let in_w = 5_usize;
        let out_h = out_spatial(in_h, 3, 1, 1);
        let out_w = out_spatial(in_w, 3, 1, 1);
        assert_eq!(out_h, in_h, "padding=1 should preserve height");
        assert_eq!(out_w, in_w, "padding=1 should preserve width");

        let input = vec![2.0_f32; in_h * in_w]; // constant 2.0 input
        let kernel = vec![1.0_f32; 9]; // all-ones 3×3
        let mut out = vec![0.0_f32; out_h * out_w];
        conv2d_3x3_f32(
            &input,
            (1, in_h, in_w),
            &kernel,
            (1, 1, 3, 3),
            (1, 1),
            (1, 1),
            &mut out,
        );
        // Interior pixels sum 9 neighbours × 2.0 = 18.0
        let interior = out[1 * out_w + 1]; // oh=1, ow=1
        assert!((interior - 18.0).abs() < EPS, "interior={interior}");
        // Corner pixel (0,0) has only 4 valid neighbours → 4*2=8.0
        let corner = out[0];
        assert!((corner - 8.0).abs() < EPS, "corner={corner}");
    }

    // Gate 6: conv2d_5x5 all-ones kernel sums 5×5 neighbourhood
    #[test]
    fn test_conv2d_5x5_sum_kernel() {
        // 1-channel 7×7 input filled with 1.0; 5×5 all-ones → interior = 25.0
        let input = vec![1.0_f32; 49];
        let kernel = vec![1.0_f32; 25];
        let out_h = out_spatial(7, 5, 1, 0);
        let out_w = out_spatial(7, 5, 1, 0);
        assert_eq!(out_h, 3);
        assert_eq!(out_w, 3);
        let mut out = vec![0.0_f32; 9];
        conv2d_5x5_f32(&input, (1, 7, 7), &kernel, (1, 1, 5, 5), (1, 1), (0, 0), &mut out);
        for (i, v) in out.iter().enumerate() {
            assert!((v - 25.0).abs() < 1e-4, "out[{i}]={v}");
        }
    }
}
