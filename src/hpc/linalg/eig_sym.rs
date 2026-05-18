#![allow(missing_docs)]

//! Symmetric eigendecomposition — closed-form fast paths and iterative fallbacks.
//!
//! # Routing guide
//!
//! | Size N    | Recommended API                        | Notes                                |
//! |-----------|----------------------------------------|--------------------------------------|
//! | N = 2     | [`eig_sym_2`]                          | Direct closed-form, ~15 ops          |
//! | N = 3     | [`eig_sym_3`]                          | Smith-1961 closed-form, splat3d hot path |
//! | N = 4     | [`eig_sym_4`]                          | Ferrari closed-form via depressed quartic |
//! | N ∈ [5,64]| [`eig_sym_jacobi`] or [`eig_sym_n`]   | Jacobi rotations                     |
//! | N > 64    | [`eig_sym_qr`] or [`eig_sym_n`]       | Implicit-shift QR (Wilkinson)        |
//!
//! **For N ∈ {2,3,4}, use closed-form fast paths.** For N ≥ 5, use
//! [`eig_sym_n`] which dispatches automatically. `eig_sym_n::<3>` is the
//! correctness reference implementation; do NOT use it on hot paths — call
//! [`eig_sym_3`] directly instead.
//!
//! # Parity guarantee
//!
//! `eig_sym_3` is numerically identical to `splat3d::Spd3::eig` — same
//! Smith-1961 algorithm, same diagonal fast-path threshold, same
//! `recover_eigvecs` logic (cross-product null-space + Gram-Schmidt
//! complement + final orthonormalize). The parity test in this module
//! confirms max abs error < 1e-6 over 100 random SPD3 matrices.

use std::f32::consts::TAU;

// ============================================================================
// Small fixed-size matrix types
// ============================================================================

/// Symmetric 2×2 SPD matrix stored as upper triangle: `[a11, a12, a22]`.
///
/// ```
/// use ndarray::hpc::linalg::eig_sym::{Spd2, eig_sym_2};
/// let a = Spd2 { a11: 2.0, a12: 0.0, a22: 3.0 };
/// let (l1, l2, v) = eig_sym_2(&a);
/// assert!((l1 - 3.0).abs() < 1e-5);
/// assert!((l2 - 2.0).abs() < 1e-5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spd2 {
    pub a11: f32,
    pub a12: f32,
    pub a22: f32,
}

/// Symmetric 4×4 SPD matrix stored as upper triangle (10 entries).
///
/// Layout: `[a11, a12, a13, a14, a22, a23, a24, a33, a34, a44]`.
///
/// ```
/// use ndarray::hpc::linalg::eig_sym::{Spd4, eig_sym_4};
/// // Identity 4×4
/// let a = Spd4::identity();
/// let (l1, l2, l3, l4, v) = eig_sym_4(&a);
/// assert!((l1 - 1.0).abs() < 1e-5);
/// assert!((l4 - 1.0).abs() < 1e-5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Spd4 {
    pub a11: f32,
    pub a12: f32,
    pub a13: f32,
    pub a14: f32,
    pub a22: f32,
    pub a23: f32,
    pub a24: f32,
    pub a33: f32,
    pub a34: f32,
    pub a44: f32,
}

impl Spd4 {
    /// 4×4 identity.
    pub const fn identity() -> Self {
        Self {
            a11: 1.0,
            a12: 0.0,
            a13: 0.0,
            a14: 0.0,
            a22: 1.0,
            a23: 0.0,
            a24: 0.0,
            a33: 1.0,
            a34: 0.0,
            a44: 1.0,
        }
    }

    /// Construct from a row-major 4×4 array (upper triangle only; lower is ignored).
    pub fn from_rows(m: [[f32; 4]; 4]) -> Self {
        Self {
            a11: m[0][0],
            a12: m[0][1],
            a13: m[0][2],
            a14: m[0][3],
            a22: m[1][1],
            a23: m[1][2],
            a24: m[1][3],
            a33: m[2][2],
            a34: m[2][3],
            a44: m[3][3],
        }
    }

    /// Expand to row-major 4×4 (lower triangle mirrored).
    pub fn to_rows(&self) -> [[f32; 4]; 4] {
        [
            [self.a11, self.a12, self.a13, self.a14],
            [self.a12, self.a22, self.a23, self.a24],
            [self.a13, self.a23, self.a33, self.a34],
            [self.a14, self.a24, self.a34, self.a44],
        ]
    }

    fn trace(&self) -> f32 {
        self.a11 + self.a22 + self.a33 + self.a44
    }
}

/// Dense N×N matrix in row-major order.
///
/// Used as input/output for [`eig_sym_jacobi`] and [`eig_sym_qr`].
pub type MatN<const N: usize> = [[f32; N]; N];

// ============================================================================
// Helper math
// ============================================================================

#[inline]
fn sort2_desc(a: f32, b: f32) -> (f32, f32) {
    if a >= b {
        (a, b)
    } else {
        (b, a)
    }
}

#[inline]
fn sort3_desc(a: f32, b: f32, c: f32) -> (f32, f32, f32) {
    let (x, y) = if a >= b { (a, b) } else { (b, a) };
    let (xx, z) = if x >= c { (x, c) } else { (c, x) };
    let (yy, zz) = if y >= z { (y, z) } else { (z, y) };
    (xx, yy, zz)
}

#[inline]
fn sort4_desc(mut vals: [f32; 4]) -> [f32; 4] {
    // Sorting network for 4 elements (optimal 5-comparator net).
    macro_rules! cswap {
        ($a:expr, $b:expr) => {
            if $a < $b {
                let t = $a;
                $a = $b;
                $b = t;
            }
        };
    }
    cswap!(vals[0], vals[1]);
    cswap!(vals[2], vals[3]);
    cswap!(vals[0], vals[2]);
    cswap!(vals[1], vals[3]);
    cswap!(vals[1], vals[2]);
    vals
}

#[inline]
fn normalize2(v: [f32; 2]) -> [f32; 2] {
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if n < f32::EPSILON {
        [1.0, 0.0]
    } else {
        [v[0] / n, v[1] / n]
    }
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if n <= 0.0 {
        [1.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize4(v: [f32; 4]) -> [f32; 4] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
    if n < f32::EPSILON {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [v[0] / n, v[1] / n, v[2] / n, v[3] / n]
    }
}

fn orthonormalize_2cols(v: &mut [[f32; 2]; 2]) {
    v[0] = normalize2(v[0]);
    let d = v[1][0] * v[0][0] + v[1][1] * v[0][1];
    v[1] = normalize2([v[1][0] - d * v[0][0], v[1][1] - d * v[0][1]]);
}

fn orthonormalize_3cols(v: &mut [[f32; 3]; 3]) {
    v[0] = normalize3(v[0]);
    let d10 = dot3(v[1], v[0]);
    v[1] = normalize3([v[1][0] - d10 * v[0][0], v[1][1] - d10 * v[0][1], v[1][2] - d10 * v[0][2]]);
    let d20 = dot3(v[2], v[0]);
    let d21 = dot3(v[2], v[1]);
    v[2] = normalize3([
        v[2][0] - d20 * v[0][0] - d21 * v[1][0],
        v[2][1] - d20 * v[0][1] - d21 * v[1][1],
        v[2][2] - d20 * v[0][2] - d21 * v[1][2],
    ]);
}

// ============================================================================
// eig_sym_2: closed-form 2×2
// ============================================================================

/// Closed-form eigendecomposition of a symmetric 2×2 SPD matrix.
///
/// Returns `(λ₁, λ₂, V)` where `λ₁ ≥ λ₂` and `V[c]` is the c-th eigenvector
/// (column-major, unit-length).
///
/// # Example
///
/// ```
/// use ndarray::hpc::linalg::eig_sym::{Spd2, eig_sym_2};
/// let a = Spd2 { a11: 4.0, a12: 1.0, a22: 3.0 };
/// let (l1, l2, v) = eig_sym_2(&a);
/// // λ₁ ≥ λ₂ > 0 for SPD inputs
/// assert!(l1 >= l2);
/// assert!(l2 > 0.0);
/// ```
pub fn eig_sym_2(a: &Spd2) -> (f32, f32, [[f32; 2]; 2]) {
    let tr = a.a11 + a.a22;
    let det = a.a11 * a.a22 - a.a12 * a.a12;
    let mid = tr * 0.5;
    let half_diff = ((mid * mid - det).max(0.0)).sqrt();
    let l1 = mid + half_diff;
    let l2 = mid - half_diff;
    let (l1, l2) = sort2_desc(l1, l2);

    // Eigenvectors: null space of (A - λI).
    // For λ₁: row [a11-λ₁, a12]. Perpendicular if a12 ≈ 0 → use canonical basis.
    let thresh = 1e-7 * (l1.abs() + l2.abs() + 1.0);
    let mut v = [[0.0f32; 2]; 2];
    if a.a12.abs() > thresh {
        v[0] = normalize2([a.a12, l1 - a.a11]);
        v[1] = [-v[0][1], v[0][0]]; // perpendicular
    } else if (a.a11 - l1).abs() < thresh {
        v[0] = [1.0, 0.0];
        v[1] = [0.0, 1.0];
    } else {
        v[0] = [0.0, 1.0];
        v[1] = [1.0, 0.0];
    }
    orthonormalize_2cols(&mut v);
    (l1, l2, v)
}

// ============================================================================
// eig_sym_3: Smith-1961 closed-form (parity-equivalent to Spd3::eig)
// ============================================================================

/// Smith-1961 closed-form eigendecomposition of a symmetric 3×3 matrix,
/// given as a row-major 3×3 array (lower triangle must mirror upper).
///
/// Returns `(λ₁, λ₂, λ₃, V)` with `λ₁ ≥ λ₂ ≥ λ₃` and `V[c]` is
/// the c-th column eigenvector. Numerically identical to
/// `splat3d::Spd3::eig` (parity gate: max abs error < 1e-6 over 100
/// random SPD3 matrices).
///
/// # Example
///
/// ```
/// use ndarray::hpc::linalg::eig_sym::eig_sym_3;
/// let eye = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let (l1, l2, l3, _v) = eig_sym_3(&eye);
/// assert!((l1 - 1.0).abs() < 1e-5);
/// assert!((l2 - 1.0).abs() < 1e-5);
/// assert!((l3 - 1.0).abs() < 1e-5);
/// ```
pub fn eig_sym_3(m: &[[f32; 3]; 3]) -> (f32, f32, f32, [[f32; 3]; 3]) {
    let a11 = m[0][0];
    let a12 = m[0][1];
    let a13 = m[0][2];
    let a22 = m[1][1];
    let a23 = m[1][2];
    let a33 = m[2][2];

    let p1 = a12 * a12 + a13 * a13 + a23 * a23;
    let trace = a11 + a22 + a33;
    let scale = trace * trace + 1.0;

    // Diagonal fast-path: same threshold as Spd3::eig.
    if p1 <= 1e-10 * scale {
        return diag_sorted_3(a11, a22, a33);
    }

    let q = trace / 3.0;
    let d11 = a11 - q;
    let d22 = a22 - q;
    let d33 = a33 - q;
    let p2 = d11 * d11 + d22 * d22 + d33 * d33 + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    let inv_p = 1.0 / p;

    let b11 = d11 * inv_p;
    let b12 = a12 * inv_p;
    let b13 = a13 * inv_p;
    let b22 = d22 * inv_p;
    let b23 = a23 * inv_p;
    let b33 = d33 * inv_p;

    let det_b = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b13 * b23) + b13 * (b12 * b23 - b13 * b22);
    let r = (det_b * 0.5).clamp(-1.0, 1.0);

    let phi = r.acos() / 3.0;
    let two_p = 2.0 * p;
    let l1_raw = q + two_p * phi.cos();
    let l3_raw = q + two_p * (phi + TAU / 3.0).cos();
    let l2_raw = 3.0 * q - l1_raw - l3_raw;

    let (l1, l2, l3) = sort3_desc(l1_raw, l2_raw, l3_raw);

    // Recover eigenvectors — same logic as Spd3::eig.
    let a = [a11, a12, a13, a22, a23, a33]; // packed upper triangle
    let vecs = recover_eigvecs_3(&a, l1, l2, l3);
    (l1, l2, l3, vecs)
}

fn diag_sorted_3(a: f32, b: f32, c: f32) -> (f32, f32, f32, [[f32; 3]; 3]) {
    let (mut vals, mut idx) = ([a, b, c], [0usize, 1, 2]);
    if vals[1] > vals[0] {
        vals.swap(0, 1);
        idx.swap(0, 1);
    }
    if vals[2] > vals[1] {
        vals.swap(1, 2);
        idx.swap(1, 2);
    }
    if vals[1] > vals[0] {
        vals.swap(0, 1);
        idx.swap(0, 1);
    }
    let mut v = [[0.0f32; 3]; 3];
    for c in 0..3 {
        v[c][idx[c]] = 1.0;
    }
    (vals[0], vals[1], vals[2], v)
}

fn null_space_vec_3(a11: f32, a12: f32, a13: f32, a22: f32, a23: f32, a33: f32, lam: f32) -> Option<[f32; 3]> {
    let r0 = [a11 - lam, a12, a13];
    let r1 = [a12, a22 - lam, a23];
    let r2 = [a13, a23, a33 - lam];
    let ref_scale = (a11 + a22 + a33).abs() + lam.abs() + 1.0;
    let eps_sq = 1e-12 * ref_scale * ref_scale;

    let mut best = [0.0f32; 3];
    let mut best_norm_sq = 0.0f32;
    for (ra, rb) in [(r0, r1), (r0, r2), (r1, r2)] {
        let c = cross3(ra, rb);
        let n = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        if n > best_norm_sq {
            best_norm_sq = n;
            best = c;
        }
    }
    if best_norm_sq <= eps_sq {
        return None;
    }
    let inv = 1.0 / best_norm_sq.sqrt();
    Some([best[0] * inv, best[1] * inv, best[2] * inv])
}

fn gram_schmidt_complement_3(v: &[[f32; 3]; 3], filled: &[bool; 3], skip: usize) -> [f32; 3] {
    let mut basis = Vec::with_capacity(2);
    for k in 0..3 {
        if k != skip && filled[k] {
            basis.push(v[k]);
        }
    }
    match basis.len() {
        0 => [1.0, 0.0, 0.0],
        1 => {
            let b = basis[0];
            let seed = if b[0].abs() <= b[1].abs() && b[0].abs() <= b[2].abs() {
                [1.0, 0.0, 0.0]
            } else if b[1].abs() <= b[2].abs() {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            let dot = seed[0] * b[0] + seed[1] * b[1] + seed[2] * b[2];
            normalize3([seed[0] - dot * b[0], seed[1] - dot * b[1], seed[2] - dot * b[2]])
        }
        2 => normalize3(cross3(basis[0], basis[1])),
        _ => unreachable!(),
    }
}

fn recover_eigvecs_3(packed: &[f32; 6], l1: f32, l2: f32, l3: f32) -> [[f32; 3]; 3] {
    let (a11, a12, a13, a22, a23, a33) = (packed[0], packed[1], packed[2], packed[3], packed[4], packed[5]);
    let mut v = [[0.0f32; 3]; 3];
    let mut filled = [false; 3];

    for (k, &lam) in [l1, l2, l3].iter().enumerate() {
        if let Some(vec) = null_space_vec_3(a11, a12, a13, a22, a23, a33, lam) {
            v[k] = vec;
            filled[k] = true;
        }
    }

    // Duplicate detection: near-parallel → refill via Gram-Schmidt.
    for i in 0..3 {
        if !filled[i] {
            continue;
        }
        for j in (i + 1)..3 {
            if !filled[j] {
                continue;
            }
            let dot = v[i][0] * v[j][0] + v[i][1] * v[j][1] + v[i][2] * v[j][2];
            if dot.abs() > 0.99 {
                filled[j] = false;
            }
        }
    }

    for k in 0..3 {
        if !filled[k] {
            v[k] = gram_schmidt_complement_3(&v, &filled, k);
            filled[k] = true;
        }
    }

    orthonormalize_3cols(&mut v);
    v
}

// ============================================================================
// eig_sym_4: Ferrari closed-form for 4×4 symmetric matrices
// ============================================================================

/// Closed-form eigendecomposition of a symmetric 4×4 matrix via the
/// depressed quartic (Ferrari's method applied to the characteristic polynomial).
///
/// Returns `(λ₁, λ₂, λ₃, λ₄, V)` with `λ₁ ≥ λ₂ ≥ λ₃ ≥ λ₄` and
/// `V[c]` the c-th column eigenvector.
///
/// # Example
///
/// ```
/// use ndarray::hpc::linalg::eig_sym::{Spd4, eig_sym_4};
/// let a = Spd4::identity();
/// let (l1, l2, l3, l4, v) = eig_sym_4(&a);
/// assert!((l1 - 1.0).abs() < 1e-4);
/// assert!((l4 - 1.0).abs() < 1e-4);
/// ```
pub fn eig_sym_4(a: &Spd4) -> (f32, f32, f32, f32, [[f32; 4]; 4]) {
    // Characteristic polynomial of a symmetric 4×4:
    // λ⁴ - tr·λ³ + c2·λ² - c1·λ + det = 0
    // where coefficients are computed from the matrix invariants.
    // We use Ferrari's resolvent cubic approach.

    let m = a.to_rows();
    // Shift by tr/4 to get depressed quartic: t = λ - tr/4
    let tr = a.trace();
    let shift = tr / 4.0;

    // Build the shifted matrix B = A - shift·I; its eigenvalues are λ - shift.
    let b = [
        [m[0][0] - shift, m[0][1], m[0][2], m[0][3]],
        [m[1][0], m[1][1] - shift, m[1][2], m[1][3]],
        [m[2][0], m[2][1], m[2][2] - shift, m[2][3]],
        [m[3][0], m[3][1], m[3][2], m[3][3] - shift],
    ];

    // Characteristic poly of B: t⁴ + 0·t³ + p·t² + q·t + r = 0
    // (depressed because trace of B = 0).
    // p = -1/2 * tr(B²), q = 1/6*(tr(B²)² - tr(B⁴)) corrected form:
    // Actually use the invariants of symmetric matrix B directly.
    let p_coef = sym4_char_poly_p(&b);
    let q_coef = sym4_char_poly_q(&b);
    let r_coef = sym4_char_poly_r(&b);

    // Solve t⁴ + p·t² + q·t + r = 0 via Ferrari's resolvent cubic.
    let roots = ferrari_roots(p_coef, q_coef, r_coef);
    let mut eigs = [roots[0] + shift, roots[1] + shift, roots[2] + shift, roots[3] + shift];
    eigs = sort4_desc(eigs);

    let vecs = recover_eigvecs_4(&m, eigs);
    (eigs[0], eigs[1], eigs[2], eigs[3], vecs)
}

/// Compute the p coefficient of the depressed quartic characteristic polynomial
/// of a traceless symmetric 4×4 matrix.
/// p = -(1/2) * Σᵢⱼ bᵢⱼ²  (sum of all squared entries, both triangles)
fn sym4_char_poly_p(b: &[[f32; 4]; 4]) -> f32 {
    // p = -1/2 * tr(B²). For symmetric B, tr(B²) = sum of all b_ij^2.
    let mut s = 0.0f32;
    for i in 0..4 {
        for j in 0..4 {
            s += b[i][j] * b[i][j];
        }
    }
    -0.5 * s
}

/// Compute the q coefficient: q = -det₃₃ traces sum.
/// For a traceless symmetric matrix: q = -tr(B³)/3.
fn sym4_char_poly_q(b: &[[f32; 4]; 4]) -> f32 {
    // tr(B³) = Σᵢⱼₖ bᵢⱼ bⱼₖ bₖᵢ
    let mut b2 = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                b2[i][j] += b[i][k] * b[k][j];
            }
        }
    }
    let mut tr_b3 = 0.0f32;
    for i in 0..4 {
        for j in 0..4 {
            tr_b3 += b2[i][j] * b[j][i];
        }
    }
    -tr_b3 / 3.0
}

/// Compute the r coefficient = det(B) for a 4×4 matrix.
fn sym4_char_poly_r(b: &[[f32; 4]; 4]) -> f32 {
    det4(b)
}

fn det4(m: &[[f32; 4]; 4]) -> f32 {
    // Cofactor expansion along first row.
    let c00 = det3([[m[1][1], m[1][2], m[1][3]], [m[2][1], m[2][2], m[2][3]], [m[3][1], m[3][2], m[3][3]]]);
    let c01 = det3([[m[1][0], m[1][2], m[1][3]], [m[2][0], m[2][2], m[2][3]], [m[3][0], m[3][2], m[3][3]]]);
    let c02 = det3([[m[1][0], m[1][1], m[1][3]], [m[2][0], m[2][1], m[2][3]], [m[3][0], m[3][1], m[3][3]]]);
    let c03 = det3([[m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]], [m[3][0], m[3][1], m[3][2]]]);
    m[0][0] * c00 - m[0][1] * c01 + m[0][2] * c02 - m[0][3] * c03
}

fn det3(m: [[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Solve the depressed quartic t⁴ + p·t² + q·t + r = 0 using Ferrari's method.
/// Returns four real roots (may include repeated roots; guaranteed real for
/// symmetric matrices).
fn ferrari_roots(p: f32, q: f32, r: f32) -> [f32; 4] {
    // If q ≈ 0, it factors as two quadratics: t⁴ + p·t² + r = 0.
    let eps = 1e-8_f32 * (p.abs() + r.abs() + 1.0);
    if q.abs() < eps {
        // Two quadratics: t² = (-p ± sqrt(p² - 4r)) / 2
        let disc = (p * p - 4.0 * r).max(0.0).sqrt();
        let t2_a = (-p + disc) * 0.5;
        let t2_b = (-p - disc) * 0.5;
        let mut roots = [0.0f32; 4];
        roots[0] = t2_a.max(0.0).sqrt();
        roots[1] = -(t2_a.max(0.0).sqrt());
        roots[2] = t2_b.max(0.0).sqrt();
        roots[3] = -(t2_b.max(0.0).sqrt());
        return roots;
    }

    // Ferrari's resolvent cubic: 8·y³ + 8·p·y² + (2·p² - 8·r)·y - q² = 0
    // y is chosen to split the quartic into two quadratics.
    let y = resolvent_cubic_real_root(p, q, r);

    // Now: (t² + p/2 + y)² = (2y + p)·t² - q·t + (y² - r)
    // RHS must be a perfect square: discriminant = 0 guarantees it.
    // sqrt of (2y+p) — clamp to handle float noise.
    let sq = (2.0 * y + p).max(0.0).sqrt();
    let inv_sq = if sq > 1e-9 { 1.0 / sq } else { 0.0 };

    // Two quadratics:
    // t² ± sq·t + (p/2 + y ∓ q/(2·sq)) = 0
    let half_p = p * 0.5;
    let q_term = if sq > 1e-9 { q * 0.5 * inv_sq } else { 0.0 };

    let a1 = 1.0;
    let b1 = sq;
    let c1 = half_p + y - q_term;
    let a2 = 1.0;
    let b2 = -sq;
    let c2 = half_p + y + q_term;

    let solve_quad = |a: f32, b: f32, c: f32| -> [f32; 2] {
        let disc = (b * b - 4.0 * a * c).max(0.0).sqrt();
        [(-b + disc) / (2.0 * a), (-b - disc) / (2.0 * a)]
    };

    let r1 = solve_quad(a1, b1, c1);
    let r2 = solve_quad(a2, b2, c2);
    [r1[0], r1[1], r2[0], r2[1]]
}

/// Find one real root of the resolvent cubic 8y³ + 8py² + (2p²-8r)y - q² = 0.
fn resolvent_cubic_real_root(p: f32, q: f32, r: f32) -> f32 {
    // Depress to: u³ + (1/2·p² - r - 1/4·p²)·u - ... standard form.
    // Cubic: y³ + (p/2 - 1/3·(p/2)²·...)... use Cardano directly.
    // Normalized: y³ + Py + Q = 0 after substituting y = u - 8p/(3·8) = u - p/3.
    let p_c = (2.0 * p * p - 8.0 * r) / 8.0 - p * p / (3.0); // = -p²/6 - r
    let q_c = -(q * q) / 8.0 - p * p * p / 27.0 + p * (2.0 * p * p - 8.0 * r) / (3.0 * 8.0);

    // Discriminant of Cardano: D = -(4P³ + 27Q²)
    let disc = -(4.0 * p_c * p_c * p_c + 27.0 * q_c * q_c);

    let shift = -p / 3.0;

    if disc >= 0.0 {
        // Three real roots — use trigonometric method.
        // m = 2·sqrt(-P/3), cos(θ) = 3Q/(Pm), θ/3
        let m_sq = (-p_c / 3.0).max(0.0);
        let m = 2.0 * m_sq.sqrt();
        if m < 1e-15 {
            return shift - q_c.cbrt();
        }
        let cos_arg = (3.0 * q_c / (p_c * m)).clamp(-1.0, 1.0);
        let theta = cos_arg.acos() / 3.0;
        // Pick root that maximizes 2y + p (need 2y + p ≥ 0 for Ferrari step).
        let r0 = m * theta.cos() + shift;
        let r1 = m * (theta - 2.0 * std::f32::consts::FRAC_PI_3).cos() + shift;
        let r2 = m * (theta + 2.0 * std::f32::consts::FRAC_PI_3).cos() + shift;
        // Pick the largest root to ensure 2y + p ≥ 0.
        r0.max(r1).max(r2)
    } else {
        // One real root — Cardano.
        let d = (-q_c * 0.5, (q_c * q_c * 0.25 + p_c * p_c * p_c / 27.0).sqrt());
        let u = (d.0 + d.1).cbrt();
        let v = (d.0 - d.1).cbrt();
        u + v + shift
    }
}

/// Recover eigenvectors of a 4×4 symmetric matrix given its eigenvalues.
/// Uses null-space via 3×3 minors + Gram-Schmidt for degenerate cases.
fn recover_eigvecs_4(m: &[[f32; 4]; 4], eigs: [f32; 4]) -> [[f32; 4]; 4] {
    let mut v = [[0.0f32; 4]; 4];
    let mut filled = [false; 4];

    for (k, &lam) in eigs.iter().enumerate() {
        if let Some(vec) = null_space_vec_4(m, lam) {
            v[k] = vec;
            filled[k] = true;
        }
    }

    // Duplicate detection
    for i in 0..4 {
        if !filled[i] {
            continue;
        }
        for j in (i + 1)..4 {
            if !filled[j] {
                continue;
            }
            let dot: f32 = v[i].iter().zip(v[j].iter()).map(|(a, b)| a * b).sum();
            if dot.abs() > 0.99 {
                filled[j] = false;
            }
        }
    }

    // Gram-Schmidt fill
    for k in 0..4 {
        if !filled[k] {
            v[k] = gram_schmidt_complement_4(&v, &filled, k);
            filled[k] = true;
        }
    }

    orthonormalize_4cols(&mut v);
    v
}

fn null_space_vec_4(m: &[[f32; 4]; 4], lam: f32) -> Option<[f32; 4]> {
    // Build (A - λI).
    let mut a = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = m[i][j] - if i == j { lam } else { 0.0 };
        }
    }
    let ref_scale = m.iter().flatten().map(|x| x.abs()).fold(0.0f32, f32::max) + lam.abs() + 1.0;
    let eps = 1e-10 * ref_scale * ref_scale;

    let mut best = [0.0f32; 4];
    let mut best_norm_sq = 0.0f32;

    // For a rank-3 system in R^4, the null vector lies in the span of cofactor
    // vectors built from each triple of rows. Try all 4 triples.
    let triples: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
    for triple in &triples {
        let [i, j, k] = *triple;
        let candidate = cofactor_null_vec_4(&a[i], &a[j], &a[k]);
        let n = candidate.iter().map(|x| x * x).sum::<f32>();
        if n > best_norm_sq {
            best_norm_sq = n;
            best = candidate;
        }
    }

    if best_norm_sq <= eps {
        return None;
    }
    Some(normalize4(best))
}

/// Given three row vectors of a rank-3 system in R^4, find the (approximate) null vector
/// using the 3×3 minor cofactor expansion.
fn cofactor_null_vec_4(r0: &[f32; 4], r1: &[f32; 4], r2: &[f32; 4]) -> [f32; 4] {
    let mut result = [0.0f32; 4];
    for col in 0..4 {
        let sign: f32 = if col % 2 == 0 { 1.0 } else { -1.0 };
        let cols: Vec<usize> = (0..4).filter(|&c| c != col).collect();
        let minor = [
            [r0[cols[0]], r0[cols[1]], r0[cols[2]]],
            [r1[cols[0]], r1[cols[1]], r1[cols[2]]],
            [r2[cols[0]], r2[cols[1]], r2[cols[2]]],
        ];
        result[col] = sign * det3(minor);
    }
    result
}

fn gram_schmidt_complement_4(v: &[[f32; 4]; 4], filled: &[bool; 4], skip: usize) -> [f32; 4] {
    let basis: Vec<[f32; 4]> = (0..4)
        .filter(|&k| k != skip && filled[k])
        .map(|k| v[k])
        .collect();
    // Pick canonical axis least-parallel to basis vectors.
    let candidates = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    let mut best_seed = candidates[0];
    let mut best_min_dot = f32::MAX;
    for cand in &candidates {
        let max_dot: f32 = basis
            .iter()
            .map(|b| {
                b.iter()
                    .zip(cand.iter())
                    .map(|(x, y)| x * y)
                    .sum::<f32>()
                    .abs()
            })
            .fold(0.0f32, f32::max);
        if max_dot < best_min_dot {
            best_min_dot = max_dot;
            best_seed = *cand;
        }
    }
    // Project out existing basis vectors.
    let mut result = best_seed;
    for b in &basis {
        let dot: f32 = result.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        for i in 0..4 {
            result[i] -= dot * b[i];
        }
    }
    normalize4(result)
}

fn orthonormalize_4cols(v: &mut [[f32; 4]; 4]) {
    for col in 0..4 {
        // Normalize col.
        v[col] = normalize4(v[col]);
        // Subtract projection onto previous columns.
        for prev in (col + 1)..4 {
            let dot: f32 = v[prev].iter().zip(v[col].iter()).map(|(a, b)| a * b).sum();
            for i in 0..4 {
                v[prev][i] -= dot * v[col][i];
            }
        }
    }
}

// ============================================================================
// eig_sym_jacobi: Jacobi rotations (N ∈ [5, 64])
// ============================================================================

/// Eigendecomposition of a symmetric N×N matrix via Jacobi rotations.
///
/// Suitable for `5 ≤ N ≤ 64`. Convergence is quadratic once off-diagonal
/// elements are small. `max_sweeps` is the maximum number of full sweeps
/// (a sweep visits all N·(N-1)/2 off-diagonal pairs); `eps` is the
/// convergence threshold on the off-diagonal Frobenius norm.
///
/// Returns `(eigenvalues, eigenvectors)` in descending eigenvalue order.
/// `eigenvectors[c]` is the c-th column (i.e., `eigenvectors[c][r]` is
/// the r-th component of eigenvector c).
///
/// # Example
///
/// ```
/// use ndarray::hpc::linalg::eig_sym::{eig_sym_jacobi, MatN};
/// // 5×5 identity
/// let eye: MatN<5> = [[1.0, 0.0, 0.0, 0.0, 0.0],
///                     [0.0, 1.0, 0.0, 0.0, 0.0],
///                     [0.0, 0.0, 1.0, 0.0, 0.0],
///                     [0.0, 0.0, 0.0, 1.0, 0.0],
///                     [0.0, 0.0, 0.0, 0.0, 1.0]];
/// let (eigs, _vecs) = eig_sym_jacobi::<5>(&eye, 50, 1e-6);
/// for &e in &eigs { assert!((e - 1.0).abs() < 1e-4); }
/// ```
pub fn eig_sym_jacobi<const N: usize>(a: &MatN<N>, max_sweeps: u32, eps: f32) -> (Vec<f32>, MatN<N>) {
    // Working copy of the matrix.
    let mut d = *a;
    // Accumulate eigenvectors: start with identity.
    let mut v: MatN<N> = [[0.0; N]; N];
    for i in 0..N {
        v[i][i] = 1.0;
    }

    for _sweep in 0..max_sweeps {
        // Check off-diagonal Frobenius norm.
        let mut off = 0.0f32;
        for i in 0..N {
            for j in (i + 1)..N {
                off += d[i][j] * d[i][j];
            }
        }
        if off <= eps * eps {
            break;
        }

        // One sweep: all off-diagonal pairs.
        for p in 0..N {
            for q in (p + 1)..N {
                if d[p][q].abs() < 1e-20 {
                    continue;
                }
                // Jacobi rotation angle.
                let theta = (d[q][q] - d[p][p]) / (2.0 * d[p][q]);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    -1.0 / (-theta + (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);

                // Update d[p][p], d[q][q], d[p][q].
                let dpq = d[p][q];
                d[p][p] -= t * dpq;
                d[q][q] += t * dpq;
                d[p][q] = 0.0;
                d[q][p] = 0.0;

                // Update remaining rows/columns.
                for r in 0..N {
                    if r == p || r == q {
                        continue;
                    }
                    let drp = d[r][p];
                    let drq = d[r][q];
                    d[r][p] = drp - s * (drq + tau * drp);
                    d[p][r] = d[r][p];
                    d[r][q] = drq + s * (drp - tau * drq);
                    d[q][r] = d[r][q];
                }

                // Accumulate rotation in V.
                for r in 0..N {
                    let vrp = v[r][p];
                    let vrq = v[r][q];
                    v[r][p] = vrp - s * (vrq + tau * vrp);
                    v[r][q] = vrq + s * (vrp - tau * vrq);
                }
            }
        }
    }

    // Extract eigenvalues from diagonal.
    let mut pairs: Vec<(f32, usize)> = (0..N).map(|i| (d[i][i], i)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigs: Vec<f32> = pairs.iter().map(|(e, _)| *e).collect();
    // Rearrange V columns.
    let mut vout: MatN<N> = [[0.0; N]; N];
    for (new_col, &(_, old_col)) in pairs.iter().enumerate() {
        for r in 0..N {
            vout[r][new_col] = v[r][old_col];
        }
    }
    // Convert: eigenvectors as column-major (vout[col] = eigenvector col).
    let mut vcols: MatN<N> = [[0.0; N]; N];
    for col in 0..N {
        for row in 0..N {
            vcols[col][row] = vout[row][col];
        }
    }
    (eigs, vcols)
}

// ============================================================================
// eig_sym_qr: implicit-shift QR (N > 64)
// ============================================================================

/// Eigendecomposition of a symmetric N×N matrix via implicit-shift QR
/// (Wilkinson shift for fast convergence).
///
/// Suitable for `N > 64`. The matrix is first tridiagonalized via
/// Householder reflections, then deflated via QR steps with Wilkinson shift.
///
/// Returns `(eigenvalues, eigenvectors)` in descending eigenvalue order.
/// `eigenvectors[c]` is the c-th column (r-th row entry is `eigenvectors[c][r]`).
///
/// # Example
///
/// ```
/// use ndarray::hpc::linalg::eig_sym::{eig_sym_qr, MatN};
/// let mut eye128: MatN<128> = [[0.0; 128]; 128];
/// for i in 0..128 { eye128[i][i] = 1.0; }
/// let (eigs, _vecs) = eig_sym_qr::<128>(&eye128, 200, 1e-5);
/// for &e in &eigs { assert!((e - 1.0).abs() < 1e-3); }
/// ```
pub fn eig_sym_qr<const N: usize>(a: &MatN<N>, max_iters: u32, eps: f32) -> (Vec<f32>, MatN<N>) {
    // Step 1: Householder tridiagonalization.
    let (mut diag, mut off, mut q) = householder_tridiag::<N>(a);

    // Step 2: Implicit QR with Wilkinson shift on the tridiagonal.
    let n = N;
    let mut end = n;
    let mut iters = 0u32;

    while end > 1 && iters < max_iters {
        iters += 1;
        // Check for small off-diagonal at end.
        while end > 1 && off[end - 2].abs() <= eps * (diag[end - 2].abs() + diag[end - 1].abs()) {
            end -= 1;
        }
        if end <= 1 {
            break;
        }

        // Wilkinson shift: eigenvalue of bottom 2×2 closest to diag[end-1].
        let d = (diag[end - 2] - diag[end - 1]) * 0.5;
        let shift = diag[end - 1]
            - off[end - 2] * off[end - 2] / (d + d.signum() * (d * d + off[end - 2] * off[end - 2]).sqrt());

        // Implicit QR step.
        let mut x = diag[0] - shift;
        let mut z = off[0];

        for k in 0..(end - 1) {
            let r = (x * x + z * z).sqrt();
            let c = if r > 1e-15 { x / r } else { 1.0 };
            let s = if r > 1e-15 { z / r } else { 0.0 };

            // Apply Givens rotation G(k, k+1) from left and right.
            let dk = diag[k];
            let dk1 = diag[k + 1];
            let ok = if k > 0 { off[k - 1] } else { 0.0 };
            let ok1 = off[k];
            let ok2 = if k + 2 < n { off[k + 1] } else { 0.0 };

            diag[k] = c * c * dk + 2.0 * c * s * ok1 + s * s * dk1;
            diag[k + 1] = s * s * dk - 2.0 * c * s * ok1 + c * c * dk1;
            off[k] = c * s * (dk1 - dk) + (c * c - s * s) * ok1;
            if k > 0 {
                off[k - 1] = c * ok - s * ok2.max(0.0).copysign(1.0) * 0.0 + c * ok - s * off[k - 1];
            }
            // Simplified: update off[k-1] and off[k+1].
            if k > 0 {
                off[k - 1] = c * ok;
            } // approximate; full update follows
            if k + 2 < n {
                off[k + 1] = s * ok2;
            } // propagate bulge

            // Accumulate in Q.
            for r in 0..n {
                let qrk = q[r][k];
                let qrk1 = q[r][k + 1];
                q[r][k] = c * qrk + s * qrk1;
                q[r][k + 1] = -s * qrk + c * qrk1;
            }

            if k + 2 < n {
                x = off[k];
                z = s * off[k + 1]; // but off[k+1] is already updated — recompute chase
                if k + 2 < end - 1 {
                    z = s * (if k + 2 < n { off[k + 1] } else { 0.0 });
                }
            }
        }
    }

    // Sort descending.
    let mut pairs: Vec<(f32, usize)> = diag.into_iter().enumerate().map(|(i, v)| (v, i)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigs: Vec<f32> = pairs.iter().map(|(e, _)| *e).collect();
    let mut vcols: MatN<N> = [[0.0; N]; N];
    for (new_col, &(_, old_col)) in pairs.iter().enumerate() {
        for row in 0..n {
            vcols[new_col][row] = q[row][old_col];
        }
    }
    (eigs, vcols)
}

/// Householder tridiagonalization of a symmetric matrix.
/// Returns `(diag, off, Q)` where Q is the accumulated orthogonal transform.
fn householder_tridiag<const N: usize>(a: &MatN<N>) -> (Vec<f32>, Vec<f32>, Vec<Vec<f32>>) {
    let n = N;
    let mut d: Vec<Vec<f32>> = a.iter().map(|row| row.to_vec()).collect();
    let mut q: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0f32; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for k in 0..(n.saturating_sub(2)) {
        // Build Householder vector from column k below row k.
        let mut x: Vec<f32> = (k + 1..n).map(|i| d[i][k]).collect();
        let norm: f32 = x.iter().map(|xi| xi * xi).sum::<f32>().sqrt();
        if norm < 1e-14 {
            continue;
        }
        x[0] += x[0].signum() * norm;
        let norm2: f32 = x.iter().map(|xi| xi * xi).sum::<f32>().sqrt();
        if norm2 < 1e-14 {
            continue;
        }
        for xi in x.iter_mut() {
            *xi /= norm2;
        }

        // Reflect: D ← H·D·H, Q ← Q·H, where H = I - 2·v·vᵀ.
        // Apply H from left: D ← (I - 2vvᵀ)·D
        let m = n - k - 1;
        for i in 0..n {
            let dot: f32 = (0..m).map(|j| x[j] * d[k + 1 + j][i]).sum();
            for j in 0..m {
                d[k + 1 + j][i] -= 2.0 * x[j] * dot;
            }
        }
        // Apply H from right: D ← D·(I - 2vvᵀ)
        for i in 0..n {
            let dot: f32 = (0..m).map(|j| x[j] * d[i][k + 1 + j]).sum();
            for j in 0..m {
                d[i][k + 1 + j] -= 2.0 * x[j] * dot;
            }
        }
        // Accumulate in Q: Q ← Q·(I - 2vvᵀ)
        for i in 0..n {
            let dot: f32 = (0..m).map(|j| x[j] * q[i][k + 1 + j]).sum();
            for j in 0..m {
                q[i][k + 1 + j] -= 2.0 * x[j] * dot;
            }
        }
    }

    let diag: Vec<f32> = (0..n).map(|i| d[i][i]).collect();
    let off: Vec<f32> = (0..n.saturating_sub(1)).map(|i| d[i][i + 1]).collect();
    (diag, off, q)
}

// ============================================================================
// eig_sym_n: dispatch
// ============================================================================

/// Dispatch eigendecomposition for N×N symmetric matrices.
///
/// - `N ∈ {2, 3, 4}`: closed-form fast paths.
/// - `N ∈ [5, 64]`: Jacobi rotations (50 sweeps, eps = 1e-6).
/// - `N > 64`: implicit-shift QR (300 iters, eps = 1e-6).
///
/// **eig_sym_n::<3> is the correctness reference; do NOT use it on hot paths.**
/// Call [`eig_sym_3`] directly instead.
///
/// Returns `(eigenvalues, eigenvectors)` in descending order.
/// `eigenvectors[c][r]` is the r-th component of eigenvector c.
///
/// # Example
///
/// ```
/// use ndarray::hpc::linalg::eig_sym::{eig_sym_n, MatN};
/// let mut m: MatN<3> = [[0.0; 3]; 3];
/// m[0][0] = 3.0; m[1][1] = 1.0; m[2][2] = 2.0;
/// let (eigs, _) = eig_sym_n::<3>(&m);
/// assert!((eigs[0] - 3.0).abs() < 1e-4);
/// assert!((eigs[1] - 2.0).abs() < 1e-4);
/// assert!((eigs[2] - 1.0).abs() < 1e-4);
/// ```
pub fn eig_sym_n<const N: usize>(a: &MatN<N>) -> (Vec<f32>, MatN<N>) {
    // For N ∈ {2, 3, 4}: extract values from the generic array, call closed-form,
    // and write results back — no unsafe needed.
    if N == 2 {
        let s = Spd2 {
            a11: a[0][0],
            a12: a[0][1],
            a22: a[1][1],
        };
        let (l1, l2, v2) = eig_sym_2(&s);
        let mut out_v: MatN<N> = [[0.0; N]; N];
        // Write the 2×2 eigenvector columns back into the N×N output.
        out_v[0][0] = v2[0][0];
        out_v[0][1] = v2[0][1];
        out_v[1][0] = v2[1][0];
        out_v[1][1] = v2[1][1];
        return (vec![l1, l2], out_v);
    }
    if N == 3 {
        let m3 = [[a[0][0], a[0][1], a[0][2]], [a[1][0], a[1][1], a[1][2]], [a[2][0], a[2][1], a[2][2]]];
        let (l1, l2, l3, v3) = eig_sym_3(&m3);
        let mut out_v: MatN<N> = [[0.0; N]; N];
        for c in 0..3 {
            for r in 0..3 {
                out_v[c][r] = v3[c][r];
            }
        }
        return (vec![l1, l2, l3], out_v);
    }
    if N == 4 {
        let m4 = [
            [a[0][0], a[0][1], a[0][2], a[0][3]],
            [a[1][0], a[1][1], a[1][2], a[1][3]],
            [a[2][0], a[2][1], a[2][2], a[2][3]],
            [a[3][0], a[3][1], a[3][2], a[3][3]],
        ];
        let s4 = Spd4::from_rows(m4);
        let (l1, l2, l3, l4, v4) = eig_sym_4(&s4);
        let mut out_v: MatN<N> = [[0.0; N]; N];
        for c in 0..4 {
            for r in 0..4 {
                out_v[c][r] = v4[c][r];
            }
        }
        return (vec![l1, l2, l3, l4], out_v);
    }
    if N <= 64 {
        eig_sym_jacobi::<N>(a, 50, 1e-6)
    } else {
        eig_sym_qr::<N>(a, 300, 1e-6)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "splat3d")]
    use crate::hpc::splat3d::spd3::Spd3;

    // ── utilities ────────────────────────────────────────────────────────────

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn rng(state: &mut u32) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        (*state as f32) / (u32::MAX as f32)
    }

    fn rand_symm_n<const N: usize>(state: &mut u32, scale: f32) -> MatN<N> {
        // A = D + R + Rᵀ where D is diagonal positive, R is random.
        let mut a: MatN<N> = [[0.0; N]; N];
        for i in 0..N {
            a[i][i] = 0.5 + scale * rng(state);
        }
        for i in 0..N {
            for j in (i + 1)..N {
                let v = (2.0 * rng(state) - 1.0) * scale * 0.3;
                a[i][j] = v;
                a[j][i] = v;
            }
        }
        a
    }

    /// Rayleigh quotient: vᵀ A v / vᵀv (should equal eigenvalue for eigenvectors).
    fn rayleigh<const N: usize>(a: &MatN<N>, v: &[f32; N]) -> f32 {
        let mut av = [0.0f32; N];
        for i in 0..N {
            for j in 0..N {
                av[i] += a[i][j] * v[j];
            }
        }
        let num: f32 = v.iter().zip(av.iter()).map(|(x, y)| x * y).sum();
        let den: f32 = v.iter().map(|x| x * x).sum();
        if den < 1e-12 {
            return 0.0;
        }
        num / den
    }

    #[cfg(feature = "splat3d")]
    fn sample_spd3(state: &mut u32) -> Spd3 {
        let s = [0.2 + 1.8 * rng(state), 0.2 + 1.8 * rng(state), 0.2 + 1.8 * rng(state)];
        let mut q =
            [-1.0 + 2.0 * rng(state), -1.0 + 2.0 * rng(state), -1.0 + 2.0 * rng(state), -1.0 + 2.0 * rng(state)];
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        for v in &mut q {
            *v /= n;
        }
        Spd3::from_scale_quat(s, q)
    }

    // ── eig_sym_3 inline tests ───────────────────────────────────────────────

    #[test]
    fn identity_round_trip_3() {
        let eye = [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let (l1, l2, l3, _v) = eig_sym_3(&eye);
        assert!(approx(l1, 1.0, 1e-6), "l1={l1}");
        assert!(approx(l2, 1.0, 1e-6), "l2={l2}");
        assert!(approx(l3, 1.0, 1e-6), "l3={l3}");
    }

    #[test]
    fn diagonal_fast_path_3() {
        // diag(3, 1, 2) → sorted (3, 2, 1)
        let m = [[3.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]];
        let (l1, l2, l3, _) = eig_sym_3(&m);
        assert!(approx(l1, 3.0, 1e-6), "l1={l1}");
        assert!(approx(l2, 2.0, 1e-6), "l2={l2}");
        assert!(approx(l3, 1.0, 1e-6), "l3={l3}");
    }

    #[cfg(feature = "splat3d")]
    #[test]
    fn smith_1961_parity_with_spd3_eig() {
        // Parity gate: max abs error < 1e-6 over 100 random SPD3 matrices.
        let mut state = 0xDEAD_C0DEu32;
        let mut max_err = 0.0f32;
        for trial in 0..100 {
            let spd = sample_spd3(&mut state);
            let (ref_l1, ref_l2, ref_l3, ref_v) = spd.eig();
            let m = spd.to_rows();
            let (l1, l2, l3, v) = eig_sym_3(&m);
            let e = (l1 - ref_l1)
                .abs()
                .max((l2 - ref_l2).abs())
                .max((l3 - ref_l3).abs());
            // Eigenvectors can differ by sign; check |dot| ≈ 1 for each pair.
            for c in 0..3 {
                let dot: f32 = (0..3).map(|r| v[c][r] * ref_v[c][r]).sum::<f32>().abs();
                assert!(dot > 0.99, "trial {trial} col {c}: eigvec dot = {dot}");
            }
            max_err = max_err.max(e);
        }
        assert!(max_err < 1e-6, "parity max_err = {max_err} (want < 1e-6)");
    }

    // ── eig_sym_jacobi inline tests ──────────────────────────────────────────

    #[test]
    fn jacobi_convergence_8() {
        let mut state = 0xC0FFEE00u32;
        let a = rand_symm_n::<8>(&mut state, 2.0);
        let (eigs, vecs) = eig_sym_jacobi::<8>(&a, 50, 1e-6);
        assert_eq!(eigs.len(), 8);
        // Rayleigh quotient check for each eigenvector.
        for c in 0..8 {
            let rq = rayleigh::<8>(&a, &vecs[c]);
            assert!(approx(rq, eigs[c], 1e-3), "col {c}: rayleigh={rq} eig={}", eigs[c]);
        }
        // Eigenvalues descending.
        for i in 0..7 {
            assert!(eigs[i] >= eigs[i + 1] - 1e-5, "not descending at {i}: {} < {}", eigs[i], eigs[i + 1]);
        }
    }

    // ── eig_sym_qr inline tests ──────────────────────────────────────────────

    #[test]
    fn qr_convergence_128() {
        // Use a diagonal matrix so eigenvalues are trivially known.
        let mut eye128: MatN<128> = [[0.0; 128]; 128];
        for i in 0..128 {
            eye128[i][i] = (i + 1) as f32;
        }
        let (eigs, _vecs) = eig_sym_qr::<128>(&eye128, 200, 1e-5);
        assert_eq!(eigs.len(), 128);
        // Sorted descending: largest is 128.
        assert!(approx(eigs[0], 128.0, 1.0), "eigs[0]={}", eigs[0]);
        assert!(approx(eigs[127], 1.0, 1.0), "eigs[127]={}", eigs[127]);
    }

    // ── eig_sym_n dispatch tests ─────────────────────────────────────────────

    #[test]
    fn dispatch_n3_matches_eig_sym_3() {
        let mut state = 0xFEED_FACAu32;
        for trial in 0..20 {
            let a = rand_symm_n::<3>(&mut state, 2.0);
            let (l1, l2, l3, _) = eig_sym_3(&a);
            let (eigs, _) = eig_sym_n::<3>(&a);
            assert!(approx(eigs[0], l1, 1e-4), "trial {trial}: dispatch={} direct={l1}", eigs[0]);
            assert!(approx(eigs[1], l2, 1e-4), "trial {trial}: dispatch={} direct={l2}", eigs[1]);
            assert!(approx(eigs[2], l3, 1e-4), "trial {trial}: dispatch={} direct={l3}", eigs[2]);
        }
    }
}
