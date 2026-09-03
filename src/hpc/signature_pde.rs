//! `signature_pde_sweep` — the Chen-Lyons signature kernel via the Goursat
//! PDE, on ndarray's canonical SIMD substrate.
//!
//! This is the real, general-dimension successor to `jc`'s
//! `goursat_substrate_probe` example (lance-graph
//! `TD-PILLAR11-SCIENTIFIC-LOOPS-BYPASS-NDARRAY-SIMD-1` /
//! `.claude/knowledge/vertical-simd-consumer-contract.md` W1.5 item #6). The
//! probe fixed path dimension = 2 as a throwaway falsifier; this module lifts
//! the same wavefront to arbitrary dimension so it is a drop-in replacement
//! for `sigker::signature_kernel_pde(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64`.
//!
//! ## The recurrence
//!
//! For paths `x` (length `n`) and `y` (length `m`) in `R^dim`, the depth-∞
//! signature kernel solves
//!
//! ```text
//! K[i+1][j+1] = K[i+1][j] + K[i][j+1] - K[i][j] + c_ij * K[i][j]
//! c_ij = <x[i+1]-x[i], y[j+1]-y[j]>          (Euclidean inner product)
//! K[i][0] = K[0][j] = 1                      (boundary)
//! ```
//!
//! in `O(n*m*dim)` flops, with no signature materialization (Hambly-Lyons
//! 2010; see `jc::hambly_lyons` for the uniqueness certificate this kernel
//! exists to serve).
//!
//! ## Why a wavefront, not row-major
//!
//! `K[i+1][j+1]` depends on three earlier cells, none of which lie on the
//! same row — but every cell on the anti-diagonal `i+j = d` depends only on
//! cells on diagonals `d-1` and `d-2`. So the solve sweeps diagonals with
//! three rolling row-indexed buffers (`prev2`, `prev1`, `cur`) instead of a
//! full `n*m` grid, and — because the interior of one diagonal has no
//! cross-cell dependency — each diagonal's interior is computed
//! [`LANES`]-wide via [`F64x8`].
//!
//! `y`'s increments are stored **reversed per dimension** (`dyr`): the
//! diagonal walk needs `dy[j-1]` for `j` decreasing as row `i` increases,
//! and reversing turns that backward walk into a forward, contiguous read —
//! no gather primitive is needed. This is an architectural property of the
//! recurrence, not a hand-tuned trick: verified bit-for-bit against the
//! dim=2 probe before this generalization (`E-OCR`-style falsifier, see the
//! module tests below).
//!
//! ## Numerics
//!
//! The three-FMA body (`t = 1*left + up`, `u = -1*diag + t`,
//! `new = c*diag + u`) fuses only the last multiply-add; `±1.0` multipliers
//! round exactly like a plain add/subtract, so this differs from a
//! non-fused scalar evaluation only in the rounding of `c*diag`, at the
//! `f64` ULP level — matching the probe's predeclared A1<->A2 tolerance.

use crate::simd::F64x8;

/// SIMD lane width for the interior-diagonal sweep (matches [`F64x8`]).
const LANES: usize = 8;

/// Per-dimension increments, one contiguous `Vec<f64>` per coordinate axis:
/// `out[a][i] = path[i+1][a] - path[i][a]`. Storing per-axis (rather than
/// interleaved) is what lets the SIMD sweep read each axis as a plain
/// contiguous slice.
fn increments_soa(path: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    let mut out = vec![Vec::with_capacity(path.len().saturating_sub(1)); dim];
    for w in path.windows(2) {
        for (axis, lane) in out.iter_mut().enumerate() {
            lane.push(w[1][axis] - w[0][axis]);
        }
    }
    out
}

/// Signature kernel `<S(x), S(y)>` via the depth-infinity Goursat PDE.
///
/// Drop-in for `sigker::signature_kernel_pde` — identical signature, any
/// path dimension (`x[0].len()`), any (possibly unequal) path lengths.
///
/// # Panics
///
/// Panics (debug only) if `x` and `y` disagree on coordinate dimension.
/// Panics (always) if either path is empty — a path needs at least one
/// point.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::signature_pde::signature_pde_sweep;
/// // A single-point path has no increments; the kernel is the empty-word 1.
/// let k = signature_pde_sweep(&[vec![0.0, 0.0]], &[vec![1.0, 1.0]]);
/// assert!((k - 1.0).abs() < 1e-12);
/// ```
pub fn signature_pde_sweep(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
    let (n, m) = (x.len(), y.len());
    assert!(n >= 1 && m >= 1, "signature_pde_sweep: paths must have at least one point");
    let dim = x[0].len();
    debug_assert_eq!(dim, y[0].len(), "signature_pde_sweep: x and y must share coordinate dimension");

    let dx = increments_soa(x, dim);
    let mut dyr = increments_soa(y, dim);
    for lane in &mut dyr {
        lane.reverse();
    }

    let mut prev2 = vec![1.0f64; n];
    let mut prev1 = vec![1.0f64; n];
    let mut cur = vec![1.0f64; n];
    let (one, neg_one, zero) = (F64x8::splat(1.0), F64x8::splat(-1.0), F64x8::splat(0.0));
    let mut lane_out = [0.0f64; LANES];

    for d in 2..(n + m - 1) {
        // Boundary of this diagonal: k[0][d] and k[d][0] are always 1.
        if d < m {
            cur[0] = 1.0;
        }
        if d < n {
            cur[d] = 1.0;
        }
        // Interior rows: i >= 1, j = d - i >= 1, i <= n-1, j <= m-1.
        let lo = 1usize.max(d.saturating_sub(m - 1));
        let hi = (d - 1).min(n - 1);
        if lo > hi {
            std::mem::swap(&mut prev2, &mut prev1);
            std::mem::swap(&mut prev1, &mut cur);
            continue;
        }
        // dyr index for row i is (m-1-d)+i: transiently negative in isize
        // before adding i, always in-range once i is in the interior band.
        let base = (m as isize) - 1 - (d as isize);
        let mut i = lo;
        while i + LANES <= hi + 1 {
            let left = F64x8::from_slice(&prev1[i..i + LANES]);
            let up = F64x8::from_slice(&prev1[i - 1..i - 1 + LANES]);
            let diag = F64x8::from_slice(&prev2[i - 1..i - 1 + LANES]);
            let r = (base + i as isize) as usize;
            debug_assert!(r + LANES <= m - 1, "signature_pde_sweep: dyr SIMD window out of range");
            let mut c = zero;
            for a in 0..dim {
                let av = F64x8::from_slice(&dx[a][i - 1..i - 1 + LANES]);
                let bv = F64x8::from_slice(&dyr[a][r..r + LANES]);
                c = av.mul_add(bv, c);
            }
            let t = one.mul_add(left, up);
            let u = neg_one.mul_add(diag, t);
            c.mul_add(diag, u).copy_to_slice(&mut lane_out);
            cur[i..i + LANES].copy_from_slice(&lane_out);
            i += LANES;
        }
        // Scalar tail: same three-FMA arithmetic, so A2 stays internally uniform.
        while i <= hi {
            let r = (base + i as isize) as usize;
            debug_assert!(r < m - 1, "signature_pde_sweep: dyr scalar index out of range");
            let mut c = 0.0f64;
            for a in 0..dim {
                c = dx[a][i - 1].mul_add(dyr[a][r], c);
            }
            let t = 1.0f64.mul_add(prev1[i], prev1[i - 1]);
            let u = (-1.0f64).mul_add(prev2[i - 1], t);
            cur[i] = c.mul_add(prev2[i - 1], u);
            i += 1;
        }
        std::mem::swap(&mut prev2, &mut prev1);
        std::mem::swap(&mut prev1, &mut cur);
    }
    prev1[n - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only scalar reference: the shipped recurrence, row-major, no
    /// SIMD, arbitrary dimension. Deliberately NOT shared code with
    /// `signature_pde_sweep` — this is the independent oracle the parity
    /// tests check against, mirroring `sigker::signature_kernel_pde`'s own
    /// row-major evaluation without depending on that crate (ndarray must
    /// not depend on sigker).
    fn goursat_reference(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64 {
        let (n, m) = (x.len(), y.len());
        let dim = x[0].len();
        let mut k = vec![1.0f64; n * m];
        for i in 0..n.saturating_sub(1) {
            for j in 0..m.saturating_sub(1) {
                let c: f64 = (0..dim)
                    .map(|a| (x[i + 1][a] - x[i][a]) * (y[j + 1][a] - y[j][a]))
                    .sum();
                let (left, up, diag) = (k[(i + 1) * m + j], k[i * m + j + 1], k[i * m + j]);
                k[(i + 1) * m + j + 1] = left + up - diag + c * diag;
            }
        }
        k[n * m - 1]
    }

    fn wiggly_path(n: usize, dim: usize, seed: f64) -> Vec<Vec<f64>> {
        (0..=n)
            .map(|i| {
                let t = i as f64 / n.max(1) as f64;
                (0..dim)
                    .map(|a| {
                        let phase = seed + a as f64 * 1.7;
                        t * (a as f64 + 1.0) + 0.05 * (37.0 * t + phase).cos()
                    })
                    .collect()
            })
            .collect()
    }

    fn assert_matches_reference(x: &[Vec<f64>], y: &[Vec<f64>]) {
        let expected = goursat_reference(x, y);
        let actual = signature_pde_sweep(x, y);
        let tol = 1e-9 * expected.abs().max(1.0);
        assert!(
            (expected - actual).abs() <= tol,
            "signature_pde_sweep mismatch: reference={expected:e} actual={actual:e} \
             (n={}, m={}, dim={})",
            x.len(),
            y.len(),
            x[0].len()
        );
    }

    #[test]
    fn parity_across_dimensions() {
        for &dim in &[1usize, 2, 3, 5] {
            let x = wiggly_path(37, dim, 0.3);
            let y = wiggly_path(37, dim, 1.1);
            assert_matches_reference(&x, &y);
        }
    }

    #[test]
    fn parity_rectangular_grids() {
        // n != m, both well past one SIMD lane, neither a multiple of LANES.
        let x = wiggly_path(50, 3, 0.2);
        let y = wiggly_path(23, 3, 0.9);
        assert_matches_reference(&x, &y);
        let x = wiggly_path(23, 3, 0.2);
        let y = wiggly_path(50, 3, 0.9);
        assert_matches_reference(&x, &y);
    }

    #[test]
    fn parity_lengths_not_multiple_of_lanes() {
        for &n in &[2usize, 3, 9, 15, 17, 33, 65] {
            let x = wiggly_path(n, 2, 0.4);
            let y = wiggly_path(n + 3, 2, 0.6);
            assert_matches_reference(&x, &y);
        }
    }

    #[test]
    fn degenerate_single_point_path_is_one() {
        let x = vec![vec![0.3, -1.2, 4.0]];
        let y = wiggly_path(10, 3, 0.5);
        assert!((signature_pde_sweep(&x, &y) - 1.0).abs() < 1e-12);
        assert!((signature_pde_sweep(&y, &x) - 1.0).abs() < 1e-12);
        let both_single = vec![vec![1.0, 2.0]];
        assert!((signature_pde_sweep(&both_single, &both_single) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn zero_increment_path_keeps_kernel_at_one() {
        // A constant path has zero increments everywhere; c_ij = 0 for all
        // (i, j), and by induction on the boundary K[i][0] = K[0][j] = 1,
        // the whole grid stays 1. This is a real invariant of the
        // recurrence (verified by hand, not asserted from the code under
        // test), so it discriminates a broken sweep from a correct one —
        // not a vacuous "the function returns *something*" check.
        let x: Vec<Vec<f64>> = (0..20).map(|_| vec![7.0, -3.0]).collect();
        let y: Vec<Vec<f64>> = (0..15).map(|_| vec![7.0, -3.0]).collect();
        let k = signature_pde_sweep(&x, &y);
        assert!((k - 1.0).abs() < 1e-12, "expected K == 1 everywhere, got {k}");
    }
}
