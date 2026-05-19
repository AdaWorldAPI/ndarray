//! `MatN<const N: usize>` — generic row-major N×N f32 carrier.
//!
//! Also provides concrete type aliases (`Mat2`, `Mat3`, `Mat4`) and
//! the SPD-cone primitives `Spd2` / `Spd3`.
//!
//! The Smith-1961 eigendecomposition for `Spd3` is foreshadowed here
//! but lives in `crate::hpc::splat3d::spd3` until PR-X10 A4 migrates it.

// ═══════════════════════════════════════════════════════════════════════════
// MatN<const N: usize>
// ═══════════════════════════════════════════════════════════════════════════

/// Row-major N×N f32 matrix carrier.
///
/// `#[repr(C, align(64))]` keeps the backing store on a single AVX-512
/// cache line for N ≤ 4 and ensures correct alignment for 64-byte SIMD
/// loads on larger N.
///
/// # Examples
///
/// ```rust
/// # use ndarray::hpc::linalg::{MatN, Mat3};
/// let id: Mat3 = MatN::identity();
/// assert_eq!(id.get(0, 0), 1.0);
/// assert_eq!(id.get(0, 1), 0.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(64))]
pub struct MatN<const N: usize> {
    data: [[f32; N]; N],
}

impl<const N: usize> MatN<N> {
    /// All-zero matrix.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Mat3;
    /// let z = Mat3::zero();
    /// assert_eq!(z.get(1, 2), 0.0);
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self { data: [[0.0; N]; N] }
    }

    /// N×N identity matrix.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Mat4;
    /// let id = Mat4::identity();
    /// assert_eq!(id.get(2, 2), 1.0);
    /// assert_eq!(id.get(1, 3), 0.0);
    /// ```
    #[inline]
    pub fn identity() -> Self {
        let mut m = Self::zero();
        for i in 0..N {
            m.data[i][i] = 1.0;
        }
        m
    }

    /// Construct from a row-major 2D array.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Mat3;
    /// let m = Mat3::from_array([[1.0, 2.0, 3.0],
    ///                            [4.0, 5.0, 6.0],
    ///                            [7.0, 8.0, 9.0]]);
    /// assert_eq!(m.get(1, 2), 6.0);
    /// ```
    #[inline]
    pub fn from_array(arr: [[f32; N]; N]) -> Self {
        Self { data: arr }
    }

    /// Construct from a closure `f(row, col) -> f32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ndarray::hpc::linalg::Mat3;
    /// let m = Mat3::from_fn(|i, j| (i * 3 + j) as f32);
    /// assert_eq!(m.get(2, 1), 7.0);
    /// ```
    #[inline]
    pub fn from_fn<F: Fn(usize, usize) -> f32>(f: F) -> Self {
        let mut data = [[0.0f32; N]; N];
        for i in 0..N {
            for j in 0..N {
                data[i][j] = f(i, j);
            }
        }
        Self { data }
    }

    /// Get element at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `row >= N` or `col >= N`.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row][col]
    }

    /// Set element at `(row, col)`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `row >= N` or `col >= N`.
    #[inline]
    pub fn set(&mut self, row: usize, col: usize, v: f32) {
        self.data[row][col] = v;
    }

    /// Return a copy of row `i` as a `[f32; N]` array.
    #[inline]
    pub fn row(&self, i: usize) -> [f32; N] {
        self.data[i]
    }

    /// Return a copy of column `j` as a `[f32; N]` array.
    #[inline]
    pub fn col(&self, j: usize) -> [f32; N] {
        let mut out = [0.0f32; N];
        for i in 0..N {
            out[i] = self.data[i][j];
        }
        out
    }

    /// Trace (sum of diagonal elements).
    #[inline]
    pub fn trace(&self) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..N {
            sum += self.data[i][i];
        }
        sum
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Concrete type aliases
// ═══════════════════════════════════════════════════════════════════════════

/// 2×2 row-major f32 matrix.
pub type Mat2 = MatN<2>;

/// 3×3 row-major f32 matrix.
pub type Mat3 = MatN<3>;

/// 4×4 row-major f32 matrix.
pub type Mat4 = MatN<4>;

// ═══════════════════════════════════════════════════════════════════════════
// Spd2 — symmetric 2×2 positive-definite matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Symmetric 2×2 SPD matrix stored as the upper triangle.
///
/// ```text
///   [ a11  a12 ]
///   [ a12  a22 ]
/// ```
///
/// `#[repr(C, align(32))]` — 12 B of payload + 20 B trailing pad = 32 B total.
/// The pad keeps the struct at an exact power-of-two size for array alignment.
///
/// # Examples
///
/// ```rust
/// # use ndarray::hpc::linalg::Spd2;
/// let id = Spd2::I;
/// assert_eq!(id.trace(), 2.0);
/// assert_eq!(id.det(), 1.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(32))]
pub struct Spd2 {
    /// (1,1) entry.
    pub a11: f32,
    /// (1,2) = (2,1) entry.
    pub a12: f32,
    /// (2,2) entry.
    pub a22: f32,
    _pad: [u8; 20],
}

impl Spd2 {
    /// 2×2 identity (unit isotropic).
    pub const I: Self = Self {
        a11: 1.0,
        a12: 0.0,
        a22: 1.0,
        _pad: [0; 20],
    };

    /// All-zero matrix. Not SPD; used only as an accumulator initialiser.
    pub const ZERO: Self = Self {
        a11: 0.0,
        a12: 0.0,
        a22: 0.0,
        _pad: [0; 20],
    };

    /// Construct from three explicit upper-triangle entries.
    #[inline]
    pub const fn new(a11: f32, a12: f32, a22: f32) -> Self {
        Self {
            a11,
            a12,
            a22,
            _pad: [0; 20],
        }
    }

    /// Trace = a11 + a22.
    #[inline]
    pub fn trace(&self) -> f32 {
        self.a11 + self.a22
    }

    /// Determinant = a11·a22 − a12².
    #[inline]
    pub fn det(&self) -> f32 {
        self.a11 * self.a22 - self.a12 * self.a12
    }

    /// Frobenius norm squared: a11² + a22² + 2·a12².
    #[inline]
    pub fn frobenius_sq(&self) -> f32 {
        self.a11 * self.a11 + self.a22 * self.a22 + 2.0 * self.a12 * self.a12
    }

    /// True when all eigenvalues exceed `eps` (Sylvester criterion).
    ///
    /// Exact for 2×2: the two leading principal minors are a11 and det.
    #[inline]
    pub fn is_symmetric_pd(&self, eps: f32) -> bool {
        self.a11 > eps && self.det() > eps
    }

    /// Expand to a `Mat2`.
    #[inline]
    pub fn to_mat_n(&self) -> Mat2 {
        Mat2::from_array([[self.a11, self.a12], [self.a12, self.a22]])
    }

    /// Build from the upper triangle of a `Mat2`.
    ///
    /// The lower triangle of `m` is ignored; symmetry is enforced by
    /// reading only `m.get(0,1)` for the off-diagonal entry.
    #[inline]
    pub fn from_mat_n_symmetric(m: &Mat2) -> Self {
        Self::new(m.get(0, 0), m.get(0, 1), m.get(1, 1))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Spd3 — re-export from splat3d for backward compatibility
// ═══════════════════════════════════════════════════════════════════════════

// The canonical implementation of Spd3 (including Smith-1961 eig, sandwich,
// sandwich_x16, sqrt, pow, log_spd, from_scale_quat) lives in
// `crate::hpc::splat3d::spd3` and is re-exported here so that PR-X10
// consumers get a stable `crate::hpc::linalg::Spd3` path today.
//
// In PR-X10 A4 (eig_sym), the implementation will migrate here and
// `splat3d` will become the downstream re-exporter. Until then, this
// conditional re-export preserves zero API breakage for both paths.
#[cfg(feature = "splat3d")]
pub use crate::hpc::splat3d::spd3::Spd3;

/// Symmetric 3×3 SPD matrix stored as the upper triangle.
///
/// This is a stand-alone definition used when the `splat3d` feature is
/// disabled. When `splat3d` is enabled, `Spd3` is re-exported from
/// `crate::hpc::splat3d::spd3` instead so the Smith-1961 eigendecomp
/// and sandwich kernels remain available.
///
/// ```text
///   [ a11  a12  a13 ]
///   [ a12  a22  a23 ]
///   [ a13  a23  a33 ]
/// ```
///
/// `#[repr(C, align(32))]` — 24 B payload + 8 B trailing pad = 32 B total.
/// Two consecutive `Spd3` instances fit one 64-B cache line.
///
/// # Examples
///
/// ```rust
/// # use ndarray::hpc::linalg::Spd3;
/// let id = Spd3::I;
/// assert_eq!(id.trace(), 3.0);
/// assert_eq!(id.det(), 1.0);
/// ```
#[cfg(not(feature = "splat3d"))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C, align(32))]
pub struct Spd3 {
    /// (1,1) entry.
    pub a11: f32,
    /// (1,2) = (2,1) entry.
    pub a12: f32,
    /// (1,3) = (3,1) entry.
    pub a13: f32,
    /// (2,2) entry.
    pub a22: f32,
    /// (2,3) = (3,2) entry.
    pub a23: f32,
    /// (3,3) entry.
    pub a33: f32,
    /// Explicit trailing pad — keeps `size_of::<Spd3>() == 32` stable.
    /// Never read.
    _pad: [u8; 8],
}

#[cfg(not(feature = "splat3d"))]
impl Spd3 {
    /// 3×3 identity covariance.
    pub const I: Self = Self {
        a11: 1.0,
        a12: 0.0,
        a13: 0.0,
        a22: 1.0,
        a23: 0.0,
        a33: 1.0,
        _pad: [0; 8],
    };

    /// All-zero matrix. Not SPD; used only as an accumulator initialiser.
    pub const ZERO: Self = Self {
        a11: 0.0,
        a12: 0.0,
        a13: 0.0,
        a22: 0.0,
        a23: 0.0,
        a33: 0.0,
        _pad: [0; 8],
    };

    /// Construct from six explicit upper-triangle entries.
    #[inline]
    pub const fn new(a11: f32, a12: f32, a13: f32, a22: f32, a23: f32, a33: f32) -> Self {
        Self {
            a11,
            a12,
            a13,
            a22,
            a23,
            a33,
            _pad: [0; 8],
        }
    }

    /// Trace = a11 + a22 + a33.
    #[inline]
    pub fn trace(&self) -> f32 {
        self.a11 + self.a22 + self.a33
    }

    /// Determinant of the symmetric 3×3.
    #[inline]
    pub fn det(&self) -> f32 {
        let Self {
            a11,
            a12,
            a13,
            a22,
            a23,
            a33,
            ..
        } = *self;
        a11 * (a22 * a33 - a23 * a23) - a12 * (a12 * a33 - a13 * a23) + a13 * (a12 * a23 - a13 * a22)
    }

    /// Frobenius norm squared: sum of all 9 squared entries (off-diags ×2).
    #[inline]
    pub fn frobenius_sq(&self) -> f32 {
        self.a11 * self.a11
            + self.a22 * self.a22
            + self.a33 * self.a33
            + 2.0 * (self.a12 * self.a12 + self.a13 * self.a13 + self.a23 * self.a23)
    }

    /// Sylvester-criterion SPD check: all three leading principal minors
    /// must exceed `eps`.
    ///
    /// O(1). Do **not** call in a per-pixel inner loop.
    #[inline]
    pub fn is_symmetric_pd(&self, eps: f32) -> bool {
        if self.a11 <= eps {
            return false;
        }
        let m22 = self.a11 * self.a22 - self.a12 * self.a12;
        if m22 <= eps {
            return false;
        }
        self.det() > eps
    }

    /// Expand to a `Mat3` (lower triangle mirrored from upper).
    #[inline]
    pub fn to_mat_n(&self) -> Mat3 {
        Mat3::from_array([
            [self.a11, self.a12, self.a13],
            [self.a12, self.a22, self.a23],
            [self.a13, self.a23, self.a33],
        ])
    }

    /// Build from the upper triangle of a `Mat3`.
    ///
    /// The lower triangle is ignored; mismatched entries are silently
    /// discarded (only the upper triangle is read).
    #[inline]
    pub fn from_mat_n_symmetric(m: &Mat3) -> Self {
        Self::new(m.get(0, 0), m.get(0, 1), m.get(0, 2), m.get(1, 1), m.get(1, 2), m.get(2, 2))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Spd3 — methods available when splat3d is enabled
// ═══════════════════════════════════════════════════════════════════════════

// When `splat3d` is enabled, the Spd3 type is the one from splat3d::spd3.
// That type already has trace(), det(), frobenius_sq(), is_spd(), and the
// full Smith-1961 eig/sandwich/pow/sqrt/log_spd API. We add the linalg-style
// helpers (is_symmetric_pd, to_mat_n, from_mat_n_symmetric) as an extension
// trait so they are available on the re-exported type.

/// Extension trait providing `linalg`-style helpers on `Spd3` when the
/// `splat3d` feature is enabled and the type is re-exported from there.
#[cfg(feature = "splat3d")]
pub trait Spd3LinalgExt {
    /// Sylvester SPD check (alias for `is_spd` with the same semantics).
    fn is_symmetric_pd(&self, eps: f32) -> bool;
    /// Expand to a `Mat3`.
    fn to_mat_n(&self) -> Mat3;
    /// Build from the upper triangle of a `Mat3`.
    fn from_mat_n_symmetric(m: &Mat3) -> Self;
}

#[cfg(feature = "splat3d")]
impl Spd3LinalgExt for Spd3 {
    #[inline]
    fn is_symmetric_pd(&self, eps: f32) -> bool {
        self.is_spd(eps)
    }

    #[inline]
    fn to_mat_n(&self) -> Mat3 {
        Mat3::from_array([
            [self.a11, self.a12, self.a13],
            [self.a12, self.a22, self.a23],
            [self.a13, self.a23, self.a33],
        ])
    }

    #[inline]
    fn from_mat_n_symmetric(m: &Mat3) -> Self {
        Self::new(m.get(0, 0), m.get(0, 1), m.get(0, 2), m.get(1, 1), m.get(1, 2), m.get(2, 2))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── MatN basics ──────────────────────────────────────────────────────

    #[test]
    fn matn_zero_all_zeros() {
        let z = Mat3::zero();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(z.get(i, j), 0.0, "zero[{i}][{j}]");
            }
        }
    }

    #[test]
    fn matn_identity_diagonal_and_off_diagonal() {
        let id = Mat4::identity();
        for i in 0..4 {
            assert_eq!(id.get(i, i), 1.0, "identity diag [{i}]");
            for j in 0..4 {
                if i != j {
                    assert_eq!(id.get(i, j), 0.0, "identity off-diag [{i}][{j}]");
                }
            }
        }
    }

    #[test]
    fn mat3_from_array_round_trip() {
        let arr = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let m = Mat3::from_array(arr);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(m.get(i, j), arr[i][j], "[{i}][{j}]");
            }
        }
    }

    #[test]
    fn mat3_from_fn_produces_expected_pattern() {
        const N: usize = 3;
        let m = MatN::<N>::from_fn(|i, j| (i * N + j) as f32);
        for i in 0..N {
            for j in 0..N {
                assert_eq!(m.get(i, j), (i * N + j) as f32, "[{i}][{j}]");
            }
        }
    }

    #[test]
    fn matn5_identity_trace() {
        // Exercises the general N path (N > 4, beyond the concrete aliases).
        let id = MatN::<5>::identity();
        let t = id.trace();
        assert_eq!(t, 5.0);
    }

    // ── Spd2 ─────────────────────────────────────────────────────────────

    #[test]
    fn spd2_identity_trace() {
        assert_eq!(Spd2::I.trace(), 2.0);
    }

    #[test]
    fn spd2_identity_det() {
        assert_eq!(Spd2::I.det(), 1.0);
    }

    #[test]
    fn spd2_size_alignment() {
        assert_eq!(core::mem::size_of::<Spd2>(), 32);
        assert_eq!(core::mem::align_of::<Spd2>(), 32);
    }

    #[test]
    fn spd2_is_symmetric_pd() {
        assert!(Spd2::I.is_symmetric_pd(1e-7));
        // Off-diagonal too large → not PD.
        assert!(!Spd2::new(1.0, 2.0, 1.0).is_symmetric_pd(1e-7));
    }

    // ── Spd3 ─────────────────────────────────────────────────────────────

    #[test]
    fn spd3_identity_trace() {
        assert_eq!(Spd3::I.trace(), 3.0);
    }

    #[test]
    fn spd3_identity_det() {
        // det(I) = 1.
        let d = Spd3::I.det();
        assert!((d - 1.0).abs() < 1e-7, "det(I) = {d}");
    }

    #[cfg(not(feature = "splat3d"))]
    #[test]
    fn spd3_size_alignment() {
        assert_eq!(core::mem::size_of::<Spd3>(), 32);
        assert_eq!(core::mem::align_of::<Spd3>(), 32);
    }

    #[cfg(not(feature = "splat3d"))]
    #[test]
    fn spd3_from_mat3_identity_is_symmetric_pd() {
        let m = Mat3::identity();
        let s = Spd3::from_mat_n_symmetric(&m);
        assert!(s.is_symmetric_pd(1e-7));
    }

    #[cfg(feature = "splat3d")]
    #[test]
    fn spd3_from_mat3_identity_is_symmetric_pd_splat3d() {
        use super::Spd3LinalgExt;
        let m = Mat3::identity();
        let s = Spd3::from_mat_n_symmetric(&m);
        assert!(s.is_symmetric_pd(1e-7));
    }
}
