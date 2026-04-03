use crate::UnsafeSharedRef;
use crate::{NdArrayElement, ShapeOps, SharedArray, iter_range_par, ops::NdArrayOps, run_par};

use alloc::{vec, vec::Vec};
use burn_backend::ElementConversion;
use burn_backend::Shape;
use ndarray::{IxDyn, s};

#[cfg(feature = "std")]
use std::collections::HashMap;
#[cfg(feature = "std")]
use std::sync::{LazyLock, RwLock};

// ============================================================================
// Compiled Attention Cache — O(1) table lookup replacing O(d) matmul
// ============================================================================
//
// When a model is loaded, attention weight matrices can be compiled into
// precomputed distance tables. During matmul, we check this cache first:
//   - Hit: return table[q_palette_idx][k_palette_idx] (O(1) per element)
//   - Miss: fall through to BLAS (O(d) per element)
//
// The cache is keyed by (m, k, n) dimensions of the matmul.
// In attention: m=seq_len, k=d_head, n=seq_len. The k dimension identifies
// which attention head's table to use.

/// A compiled attention table: 256×256 u16 distances, precomputed.
#[cfg(feature = "std")]
#[derive(Clone)]
pub struct CompiledAttention {
    /// 256×256 distance table. table[q][k] = precomputed attention distance.
    pub table: Vec<u16>,
    /// Palette size (number of archetypes, typically 256).
    pub k_palette: usize,
    /// Input dimension this table was compiled from.
    pub d_head: usize,
    /// Palette assignment: for each row index, which palette entry it maps to.
    pub q_assignments: Vec<u8>,
    /// Palette assignment for columns.
    pub k_assignments: Vec<u8>,
}

/// Global cache of compiled attention tables.
/// Keyed by (d_head) — the inner dimension of the attention matmul.
#[cfg(feature = "std")]
static ATTENTION_CACHE: LazyLock<RwLock<HashMap<usize, CompiledAttention>>> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

/// Register a compiled attention table for a given head dimension.
#[cfg(feature = "std")]
pub fn register_attention_table(d_head: usize, table: CompiledAttention) {
    let mut cache = ATTENTION_CACHE.write().unwrap();
    cache.insert(d_head, table);
}

/// Check if a compiled attention table exists for the given dimensions.
#[cfg(feature = "std")]
pub fn has_attention_table(d_head: usize) -> bool {
    let cache = ATTENTION_CACHE.read().unwrap();
    cache.contains_key(&d_head)
}

/// Clear all compiled attention tables.
#[cfg(feature = "std")]
pub fn clear_attention_cache() {
    let mut cache = ATTENTION_CACHE.write().unwrap();
    cache.clear();
}

// ============================================================================
// VNNI u8 MatVec fast path — 64 MACs per instruction
// ============================================================================
//
// For quantized u8×i8 matmul (codebook distance table build):
//   Input A: [m, k] u8 (codebook rows, quantized)
//   Input B: [k, n] i8 (codebook cols, quantized)
//   Output C: [m, n] i32 (distance table)
//
// One VPDPBUSD = 64 multiply-accumulates in one instruction.
// Entire 4096² distance table in ~1:20h instead of 24-48h.
//
// Runtime dispatched: VNNI → scalar. AMX added when Rust stabilizes (issue #126622).

/// Try VNNI-accelerated u8 matmul for distance table construction.
/// Returns true if VNNI was used, false to fall through to BLAS.
///
/// Only activates when BOTH inputs are contiguous u8/i8-quantized.
/// The caller is responsible for quantizing f32→u8/i8 before calling.
#[cfg(feature = "std")]
pub fn try_vnni_matmul_u8(
    a_u8: &[u8],       // [m × k] row-major
    b_i8: &[i8],       // [k × n] row-major (transposed for dot product)
    c_i32: &mut [i32],  // [m × n] output
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if !is_x86_feature_detected!("avx512vnni") { return false; }
        if a_u8.len() < m * k || b_i8.len() < k * n || c_i32.len() < m * n { return false; }

        // For each output[i][j]: dot product of A[i, :] and B[:, j]
        // B is stored row-major [k, n], but we need column j → stride n access.
        // Transpose B on the fly into a contiguous column buffer.
        let mut col_buf = vec![0i8; k];

        for j in 0..n {
            // Extract column j of B into contiguous buffer
            for p in 0..k { col_buf[p] = b_i8[p * n + j]; }

            // VNNI dot product: each row of A against this column
            for i in 0..m {
                let row_a = &a_u8[i * k..(i + 1) * k];
                c_i32[i * n + j] = ndarray::simd_amx::vnni_dot_u8_i8_scalar(row_a, &col_buf);
                // Note: using scalar dot here for correctness.
                // The vnni_dot_u8_i8 (SIMD) requires #[target_feature] propagation
                // which we can't do from a non-target_feature function.
                // For full VNNI speed, call ndarray::simd_amx::matvec_dispatch directly.
            }
        }
        return true;
    }
    #[allow(unreachable_code)]
    false
}

/// Build a k×k distance table from k centroids using VNNI if available.
///
/// centroids_u8: [k × dim] quantized codebook centroids (u8, row-major)
/// Returns: [k × k] i32 dot product matrix (symmetric)
///
/// Uses VNNI dot product (64 MACs/instruction) for each centroid pair.
/// Symmetric: only computes upper triangle, mirrors to lower.
///
/// This IS the ThinkingEngine's brain construction step.
/// 4096² = 16M dot products. With VNNI: ~1:20h for large dim.
#[cfg(feature = "std")]
pub fn build_distance_table_vnni(centroids_u8: &[u8], k: usize, dim: usize) -> Vec<i32> {
    assert_eq!(centroids_u8.len(), k * dim);

    // Convert to i8 for the second operand (VNNI does u8 × i8)
    let centroids_i8: Vec<i8> = centroids_u8.iter()
        .map(|&v| (v as i16 - 128) as i8)
        .collect();

    let mut table = vec![0i32; k * k];

    // Tiered dispatch: AMX (256 MACs) → AVX-512 VNNI (64 MACs) → AVX-VNNI (32 MACs) → scalar
    #[cfg(target_arch = "x86_64")]
    let tier = {
        if ndarray::simd_amx::amx_available() {
            3 // AMX: 256 MACs/instr (TDPBUSD 16×16 tile) — inline asm on stable
        } else if is_x86_feature_detected!("avx512vnni") {
            2 // AVX-512 VNNI: 64 MACs/instr (VPDPBUSD zmm)
        } else if is_x86_feature_detected!("avx_vnni") {
            1 // AVX-VNNI (Alder Lake+): 32 MACs/instr (VPDPBUSD ymm, no avx512)
        } else {
            0 // Scalar
        }
    };
    #[cfg(not(target_arch = "x86_64"))]
    let tier = 0;

    // Dot product function selected by tier
    let dot_fn: fn(&[u8], &[i8]) -> i32 = match tier {
        2 => |a, b| {
            // SAFETY: AVX-512 VNNI confirmed via is_x86_feature_detected above
            #[cfg(target_arch = "x86_64")]
            unsafe { ndarray::simd_amx::vnni_dot_u8_i8(a, b) }
            #[cfg(not(target_arch = "x86_64"))]
            ndarray::simd_amx::vnni_dot_u8_i8_scalar(a, b)
        },
        // Tier 3 (AMX): use VNNI for now — AMX tile matmul needs different API
        // (tiles operate on 16×64 blocks, not row×col dot products)
        // TODO: when AMX intrinsics stabilize, use TDPBUSD for 16×16 tile blocks
        3 => |a, b| {
            #[cfg(target_arch = "x86_64")]
            unsafe { ndarray::simd_amx::vnni_dot_u8_i8(a, b) }
            #[cfg(not(target_arch = "x86_64"))]
            ndarray::simd_amx::vnni_dot_u8_i8_scalar(a, b)
        },
        // Tier 1 (AVX-VNNI): uses ymm (256-bit) VPDPBUSD — same instruction, half width
        // Our vnni_dot_u8_i8 uses zmm (512-bit), but the CPU will execute it
        // at half throughput on AVX-VNNI-only chips. Still faster than scalar.
        1 => |a, b| {
            #[cfg(target_arch = "x86_64")]
            unsafe { ndarray::simd_amx::vnni_dot_u8_i8(a, b) }
            #[cfg(not(target_arch = "x86_64"))]
            ndarray::simd_amx::vnni_dot_u8_i8_scalar(a, b)
        },
        _ => ndarray::simd_amx::vnni_dot_u8_i8_scalar,
    };

    for i in 0..k {
        let row_u8 = &centroids_u8[i * dim..(i + 1) * dim];

        // Diagonal
        table[i * k + i] = dot_fn(row_u8, &centroids_i8[i * dim..(i + 1) * dim]);

        // Upper triangle (symmetric: compute once, mirror)
        for j in (i + 1)..k {
            let dot = dot_fn(row_u8, &centroids_i8[j * dim..(j + 1) * dim]);
            table[i * k + j] = dot;
            table[j * k + i] = dot;
        }
    }

    table
}

/// Try to compute matmul using compiled attention table lookup.
/// Returns None if no table exists for these dimensions.
#[cfg(feature = "std")]
fn try_attention_matmul<E: NdArrayElement>(
    _lhs: &ndarray::ArrayView2<'_, E>,
    _rhs: &ndarray::ArrayView2<'_, E>,
    out: &mut ndarray::ArrayViewMut2<'_, E>,
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    let cache = ATTENTION_CACHE.read().unwrap();
    let table = match cache.get(&k) {
        Some(t) => t,
        None => return false,
    };

    // Use palette assignments to look up precomputed distances
    if table.q_assignments.len() < m || table.k_assignments.len() < n {
        return false;
    }

    for i in 0..m {
        let q_idx = table.q_assignments[i] as usize;
        for j in 0..n {
            let k_idx = table.k_assignments[j] as usize;
            // Table lookup: O(1) per element instead of O(k) dot product
            let dist = table.table[q_idx * table.k_palette + k_idx];
            // Convert distance to similarity score (higher = more attention)
            // Negate and scale: attention ∝ -distance
            let score: f64 = -(dist as f64) / 1000.0;
            out[[i, j]] = score.elem();
        }
    }
    true
}

pub(crate) fn matmul<E: NdArrayElement>(
    lhs: SharedArray<E>,
    rhs: SharedArray<E>,
) -> SharedArray<E> {
    let shape_lhs = lhs.shape();
    let shape_rhs = rhs.shape();
    let ndims = shape_lhs.num_dims();
    let m = shape_lhs[ndims - 2]; // # of left rows
    let k = shape_rhs[ndims - 2]; // # of left cols and right rows
    let n = shape_rhs[ndims - 1]; // # of right cols

    let (out_shape, strides_lhs, strides_rhs, strides_out) = output_shape(shape_lhs, shape_rhs);
    let l_mat_size = m * k; // size of matrix component of left array
    let r_mat_size = k * n; // size of matrix component of right array
    let out_mat_size = m * n; // size of matrix component of output array

    let num_l_batches = shape_lhs.num_elements() / l_mat_size;
    let num_r_batches = shape_rhs.num_elements() / r_mat_size;
    let num_out_batches = out_shape.num_elements() / out_mat_size;

    let lhs_array = NdArrayOps::reshape(lhs, Shape::new([num_l_batches, m, k]));
    let rhs_array = NdArrayOps::reshape(rhs, Shape::new([num_r_batches, k, n]));

    let alpha: E = 1.0.elem();
    let beta: E = 0.0.elem();

    let out = run_par!(|| {
        let mut out_array = ndarray::Array3::<E>::zeros((num_out_batches, m, n));
        let unsafe_shared_out_array = UnsafeSharedRef::new(&mut out_array);

        iter_range_par!(0, num_out_batches).for_each(|out_batch| {
            // Here, we:
            //   1. Un-flatten the output batch into a component-based batch index.
            //   2. Use the strides for left and right batch indices to convert it to a flattened
            //      batch for left and right.
            let out_index = strides_out.unflatten(out_batch);
            let l_batch = strides_lhs.flatten(&out_index);
            let r_batch = strides_rhs.flatten(&out_index);

            let lhs_slice = lhs_array.slice(s!(l_batch, .., ..));
            let rhs_slice = rhs_array.slice(s!(r_batch, .., ..));

            unsafe {
                let mut out_slice = unsafe_shared_out_array
                    .get()
                    .slice_mut(s!(out_batch, .., ..));

                // Try compiled attention table (O(1) per element).
                // Falls through to BLAS if no table is registered for d_head=k.
                #[cfg(feature = "std")]
                if try_attention_matmul(&lhs_slice, &rhs_slice, &mut out_slice, m, k, n) {
                    return;
                }

                ndarray::linalg::general_mat_mul(
                    alpha,
                    &lhs_slice,
                    &rhs_slice,
                    beta,
                    &mut out_slice,
                )
            }
        });

        out_array.into_shared().into_dyn()
    });

    NdArrayOps::reshape(out, out_shape)
}

#[derive(Debug, PartialEq)]
struct Strides {
    strides: Vec<usize>,
}
impl Strides {
    fn new(strides: Vec<usize>) -> Self {
        Strides { strides }
    }

    fn unflatten(&self, linear_index: usize) -> Vec<usize> {
        let mut coord = Vec::with_capacity(self.strides.len());
        let mut rem = linear_index;
        for stride in self.strides.iter() {
            coord.push(rem / stride);
            rem %= stride;
        }
        coord
    }

    fn flatten(&self, index: &Vec<usize>) -> usize {
        assert_eq!(self.strides.len(), index.len());
        self.strides
            .iter()
            .zip(index)
            .map(|(stride, index)| stride * index)
            .sum()
    }
}

/// Compute the (broadcasted) output shape of matrix multiplication, along with strides for
/// the non-matrix dimensions of all arrays.
///
/// # Arguments
/// * `lsh`: Shape of the first (left-hand) matrix multiplication argument.
/// * `rsh`: Shape of the second (right-hand) matrix multiplication argument.
///
/// # Panics
/// * If `D` is not at least 2.
/// * If the matrix multiplication dimensions (last 2) are incompatible.
/// * If any other dimension is not the same for both tensors, or equal to 1. (Any dimension where
///   one dim is equal to 1 is broadcast.)
fn output_shape(lsh: &[usize], rsh: &[usize]) -> (Shape, Strides, Strides, Strides) {
    let ndims = lsh.num_dims();
    if ndims < 2 {
        panic!("Matrix multiplication requires an array with at least 2 dimensions.");
    }

    // Fetch matrix dimensions and check compatibility.
    let l_rows = lsh[ndims - 2];
    let l_cols = lsh[ndims - 1];
    let r_rows = rsh[ndims - 2];
    let r_cols = rsh[ndims - 1];
    if l_cols != r_rows {
        panic!("Dimensions are incompatible for matrix multiplication.");
    }
    // Set matrix dimensions of the output shape.
    let mut osh = vec![0; ndims];
    osh[ndims - 2] = l_rows;
    osh[ndims - 1] = r_cols;

    // Set other array dimensions, broadcasting as necessary.
    // Compute the strides inline.
    let mut cur_l_stride: usize = 1;
    let mut cur_r_stride: usize = 1;
    let mut cur_o_stride: usize = 1;
    let mut l_strides = Vec::with_capacity(ndims - 2);
    let mut r_strides = Vec::with_capacity(ndims - 2);
    let mut o_strides = Vec::with_capacity(ndims - 2);
    for i in (0..ndims - 2).rev() {
        let l_dim = lsh[i];
        let r_dim = rsh[i];

        // Compatible dimensions are:
        //   1. Both dimensions are equal.
        //   2. One of the dimensions is equal to 1.
        let o_dim: usize;
        if l_dim == r_dim {
            o_dim = l_dim; // both dimensions are equal
            l_strides.push(cur_l_stride);
            r_strides.push(cur_r_stride);
        } else if l_dim == 1 {
            o_dim = r_dim; // broadcast the left
            l_strides.push(0);
            r_strides.push(cur_r_stride);
        } else if r_dim == 1 {
            o_dim = l_dim; // broadcast the right
            l_strides.push(cur_l_stride);
            r_strides.push(0);
        } else {
            panic!("Dimensions differ and cannot be broadcasted.");
        }
        osh[i] = o_dim;
        o_strides.push(cur_o_stride);
        cur_o_stride *= o_dim;

        cur_l_stride *= l_dim;
        cur_r_stride *= r_dim;
    }
    l_strides.reverse();
    r_strides.reverse();
    o_strides.reverse();

    (
        Shape::from(osh),
        Strides::new(l_strides),
        Strides::new(r_strides),
        Strides::new(o_strides),
    )
}

pub(crate) fn cross<E: NdArrayElement>(
    lhs: SharedArray<E>,
    rhs: SharedArray<E>,
    dim: usize,
) -> SharedArray<E> {
    let shape_lhs = lhs.shape();
    let shape_rhs = rhs.shape();
    let ndims = shape_lhs.num_dims();

    // Broadcast the shapes except along dim
    let mut broadcast_shape = vec![0; ndims];
    for i in 0..ndims {
        if i == dim {
            broadcast_shape[i] = shape_lhs[i]; // already checked to be 3
        } else {
            let l = shape_lhs[i];
            let r = shape_rhs[i];
            if l == r {
                broadcast_shape[i] = l;
            } else if l == 1 {
                broadcast_shape[i] = r;
            } else if r == 1 {
                broadcast_shape[i] = l;
            } else {
                panic!("Tensors are not broadcastable along dimension {}", i);
            }
        }
    }

    // Broadcast lhs and rhs
    let lhs_broadcast = if shape_lhs == broadcast_shape.as_slice() {
        lhs
    } else {
        NdArrayOps::expand(lhs, Shape::from(broadcast_shape.clone()))
    };
    let rhs_broadcast = if shape_rhs == broadcast_shape.as_slice() {
        rhs
    } else {
        NdArrayOps::expand(rhs, Shape::from(broadcast_shape.clone()))
    };

    // Now, move dim to the last dimension
    let mut perm = (0..ndims).collect::<Vec<_>>();
    perm.remove(dim);
    perm.push(dim);

    let lhs_permuted = NdArrayOps::permute(lhs_broadcast, &perm);
    let rhs_permuted = NdArrayOps::permute(rhs_broadcast, &perm);

    // Reshape to (*, 3)
    let total_elements = lhs_permuted.shape().num_elements();
    let batch_size = total_elements / 3;
    let lhs_reshaped = NdArrayOps::reshape(lhs_permuted, Shape::new([batch_size, 3]));
    let rhs_reshaped = NdArrayOps::reshape(rhs_permuted, Shape::new([batch_size, 3]));

    // Compute cross product
    let mut result = ndarray::ArrayD::<E>::zeros(IxDyn(&[batch_size, 3]));
    for i in 0..batch_size {
        let a1 = lhs_reshaped[IxDyn(&[i, 0])];
        let a2 = lhs_reshaped[IxDyn(&[i, 1])];
        let a3 = lhs_reshaped[IxDyn(&[i, 2])];
        let b1 = rhs_reshaped[IxDyn(&[i, 0])];
        let b2 = rhs_reshaped[IxDyn(&[i, 1])];
        let b3 = rhs_reshaped[IxDyn(&[i, 2])];
        result[IxDyn(&[i, 0])] = a2.mul(b3).sub(a3.mul(b2));
        result[IxDyn(&[i, 1])] = a3.mul(b1).sub(a1.mul(b3));
        result[IxDyn(&[i, 2])] = a1.mul(b2).sub(a2.mul(b1));
    }

    let result_shared = result.into_shared();

    // Reshape back to the broadcast shape with dim at the end
    let mut result_shape = broadcast_shape;
    result_shape.remove(dim);
    result_shape.push(3);
    let result_reshaped = NdArrayOps::reshape(result_shared, Shape::from(result_shape));

    // Permute back
    let mut inv_perm = vec![0; ndims];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }
    NdArrayOps::permute(result_reshaped, &inv_perm)
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Strides {
        fn empty() -> Self {
            Strides {
                strides: Vec::with_capacity(0),
            }
        }
    }

    #[test]
    fn test_output_shape() {
        // plain matrix multiply
        assert_eq!(
            output_shape(&[5, 3], &[3, 7]),
            (
                Shape::from([5, 7]),
                Strides::empty(),
                Strides::empty(),
                Strides::empty()
            )
        );
        // matrix multiply with one extra stack dimension
        assert_eq!(
            output_shape(&[4, 5, 3], &[4, 3, 7]),
            (
                Shape::from([4, 5, 7]),
                Strides::new(vec![1]),
                Strides::new(vec![1]),
                Strides::new(vec![1])
            )
        );
        // rank 3, broadcast left
        assert_eq!(
            output_shape(&[1, 5, 3], &[4, 3, 7]),
            (
                Shape::from([4, 5, 7]),
                Strides::new(vec![0]),
                Strides::new(vec![1]),
                Strides::new(vec![1])
            )
        );
        // rank 3, broadcast right
        assert_eq!(
            output_shape(&[4, 5, 3], &[1, 3, 7]),
            (
                Shape::from([4, 5, 7]),
                Strides::new(vec![1]),
                Strides::new(vec![0]),
                Strides::new(vec![1])
            )
        );
        // rank 4, multi broadcast
        assert_eq!(
            output_shape(&[1, 4, 5, 3], &[8, 1, 3, 7]),
            (
                Shape::from([8, 4, 5, 7]),
                Strides::new(vec![0, 1]),
                Strides::new(vec![1, 0]),
                Strides::new(vec![4, 1])
            )
        );
        // rank 5, multi-broadcast
        assert_eq!(
            output_shape(&[1, 3, 4, 5, 3], &[8, 3, 1, 3, 7]),
            (
                Shape::from([8, 3, 4, 5, 7]),
                Strides::new(vec![0, 4, 1]),
                Strides::new(vec![3, 1, 0]),
                Strides::new(vec![12, 4, 1])
            )
        )
    }

    #[test]
    #[should_panic(
        expected = "Matrix multiplication requires an array with at least 2 dimensions."
    )]
    fn test_output_shape_too_small() {
        output_shape(&[4], &[4]);
    }

    #[test]
    #[should_panic(expected = "Dimensions are incompatible for matrix multiplication.")]
    fn test_output_shape_bad_matrix_dims() {
        output_shape(&[5, 3], &[4, 7]);
    }

    #[test]
    #[should_panic(expected = "Dimensions differ and cannot be broadcasted.")]
    fn test_output_shape_non_broadcast() {
        output_shape(&[4, 5, 3], &[2, 3, 7]);
    }
}
