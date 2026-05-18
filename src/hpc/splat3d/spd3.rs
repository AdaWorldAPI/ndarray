//! Symmetric 3×3 SPD math for the EWA-sandwich projection kernel.
//!
//! # The mathematical claim (Pillar 7, 3D analogue of Pillar 6)
//!
//! For an anisotropic 3D gaussian with covariance Σ ∈ ℝ^{3×3}_{SPD} and
//! a projection / view transform M ∈ ℝ^{3×3}, the push-forward of the
//! density to the projected frame is the **sandwich**:
//!
//! ```text
//!     Σ' = M · Σ · Mᵀ
//! ```
//!
//! When M is itself symmetric (e.g. M = sqrt(step-Σ) along an edge
//! path, or the symmetrized projection between two near-identity
//! frames), the sandwich reduces to `M · Σ · M`. That is the form
//! certified by the Pillar-7 probe in `jc::ewa_sandwich_3d` and the
//! form `Spd3::sandwich` implements (sibling of Pillar 6's 2D case in
//! `jc::ewa_sandwich`).
//!
//! For the asymmetric J·W form used in `splat3d::project` (PR 3), the
//! caller supplies the full 3×3 J·W as a non-symmetric matrix and
//! convolves Σ → Σ' through it — that pathway lives in `project.rs`
//! and does NOT funnel through `Spd3::sandwich`.
//!
//! # Eigendecomposition
//!
//! Smith 1961 closed-form for symmetric 3×3 (Communications of the ACM
//! 4(4):168). No iteration, no Jacobi rotations, no QR — branchless
//! once the `p1 ≈ 0` diagonal fast-path is taken:
//!
//! ```text
//!     p1 = a₁₂² + a₁₃² + a₂₃²
//!
//!     if p1 ≈ 0:                         # diagonal — eigenvalues = diag
//!         (λ₁, λ₂, λ₃) = sort_desc(a₁₁, a₂₂, a₃₃)
//!     else:
//!         q  = (a₁₁ + a₂₂ + a₃₃) / 3     # mean diagonal
//!         p2 = (a₁₁-q)² + (a₂₂-q)² + (a₃₃-q)² + 2·p1
//!         p  = sqrt(p2 / 6)
//!         B  = (A − q·I) / p              # symmetric, trace = 0,
//!                                         # eigenvalues ∈ [−2, 2]
//!         r  = det(B) / 2                 # ∈ [−1, 1] modulo float drift
//!         φ  = acos(clamp(r, −1, 1)) / 3
//!         λ₁ = q + 2p·cos(φ)             # largest
//!         λ₃ = q + 2p·cos(φ + 2π/3)      # smallest
//!         λ₂ = 3q − λ₁ − λ₃              # middle (trace identity)
//! ```
//!
//! Eigenvectors recovered via the rank-deficient `(A − λᵢ·I)` null-space
//! crossing two rows + Gram-Schmidt for orthonormality. Degenerate
//! cases (repeated eigenvalues) fall through to an axis-aligned basis
//! since the rotation is then ambiguous in that subspace.
//!
//! # Storage
//!
//! `#[repr(C, align(32))]` — six floats (24 B) + 8 B pad = 32 B total,
//! so two consecutive `Spd3`s land in one 64-B cache line without false
//! sharing in tile-binned rasterizer rows. Layout is upper-triangle SoA:
//! `a11 a12 a13 a22 a23 a33`.
//!
//! # Hot-path API
//!
//! - `Spd3::eig() -> (λ₁, λ₂, λ₃, eigvecs[3][3])` — descending order.
//! - `Spd3::pow(t)` — Σ^t via spectral lift (specialised `sqrt` /
//!   `log_spd` shims for the common cases).
//! - `Spd3::from_scale_quat(scale, quat)` — the 3DGS canonical
//!   Σ = R · diag(s²) · Rᵀ where R is the rotation matrix of `quat`.
//! - `sandwich(M, N)` — `M · N · Mᵀ` for symmetric M, N. Output is
//!   symmetric by construction (rounding eliminated by averaging
//!   `(R + Rᵀ)/2` on the asymmetric residuals).
//! - `sandwich_x16` — 16-wide SIMD batch via `crate::simd::F32x16`,
//!   the form the rasterizer hits on every tile slab.

use crate::simd::F32x16;

// ════════════════════════════════════════════════════════════════════════════
// Storage
// ════════════════════════════════════════════════════════════════════════════

/// Symmetric 3×3 SPD covariance stored as the upper triangle.
///
/// ```text
///   [ a11  a12  a13 ]
///   [ a12  a22  a23 ]
///   [ a13  a23  a33 ]
/// ```
///
/// `#[repr(C, align(32))]` — 24 B of payload + 8 B trailing pad. Two
/// `Spd3` instances fit one 64 B cache line; `Vec<Spd3>` is naturally
/// 32-byte aligned at allocation so consecutive AVX-512 loads stay
/// aligned without scatter/gather fixups.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct Spd3 {
    pub a11: f32,
    pub a12: f32,
    pub a13: f32,
    pub a22: f32,
    pub a23: f32,
    pub a33: f32,
    /// Explicit trailing pad — keeps `size_of::<Spd3>() == 32` stable
    /// across compilers and documents the alignment choice. Never read.
    _pad: [u8; 8],
}

impl Spd3 {
    /// 3×3 identity covariance — unit isotropic gaussian.
    pub const I: Self = Self {
        a11: 1.0,
        a12: 0.0,
        a13: 0.0,
        a22: 1.0,
        a23: 0.0,
        a33: 1.0,
        _pad: [0; 8],
    };

    /// Zero matrix. Never SPD; used only as an accumulator init.
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
    /// Caller is responsible for ensuring the result is SPD.
    #[inline]
    pub const fn new(a11: f32, a12: f32, a13: f32, a22: f32, a23: f32, a33: f32) -> Self {
        Self { a11, a12, a13, a22, a23, a33, _pad: [0; 8] }
    }

    /// Construct from a row-major 3×3 array. Symmetry is enforced by
    /// reading only the upper triangle; mismatched lower-triangle
    /// entries are silently discarded.
    #[inline]
    pub fn from_rows(m: [[f32; 3]; 3]) -> Self {
        Self::new(m[0][0], m[0][1], m[0][2], m[1][1], m[1][2], m[2][2])
    }

    /// Expand to a row-major 3×3 array (lower triangle mirrored).
    #[inline]
    pub fn to_rows(&self) -> [[f32; 3]; 3] {
        [
            [self.a11, self.a12, self.a13],
            [self.a12, self.a22, self.a23],
            [self.a13, self.a23, self.a33],
        ]
    }

    /// Trace = a11 + a22 + a33 (sum of eigenvalues).
    #[inline]
    pub fn trace(&self) -> f32 {
        self.a11 + self.a22 + self.a33
    }

    /// Frobenius norm squared: sum of all 9 squared entries.
    /// Symmetric so off-diagonals counted twice.
    #[inline]
    pub fn frobenius_sq(&self) -> f32 {
        self.a11 * self.a11
            + self.a22 * self.a22
            + self.a33 * self.a33
            + 2.0 * (self.a12 * self.a12 + self.a13 * self.a13 + self.a23 * self.a23)
    }

    /// Determinant of the symmetric 3×3:
    /// `a11·(a22·a33 − a23²) − a12·(a12·a33 − a13·a23) + a13·(a12·a23 − a13·a22)`.
    #[inline]
    pub fn det(&self) -> f32 {
        let Self { a11, a12, a13, a22, a23, a33, .. } = *self;
        a11 * (a22 * a33 - a23 * a23)
            - a12 * (a12 * a33 - a13 * a23)
            + a13 * (a12 * a23 - a13 * a22)
    }

    /// Cheap SPD predicate: all leading principal minors positive,
    /// determinant > eps. Sylvester's criterion at f32 precision.
    pub fn is_spd(&self, eps: f32) -> bool {
        if self.a11 <= eps {
            return false;
        }
        // 2×2 leading minor
        let m22 = self.a11 * self.a22 - self.a12 * self.a12;
        if m22 <= eps {
            return false;
        }
        if self.det() <= eps {
            return false;
        }
        // Final check: all eigenvalues > 0 (Sylvester is necessary AND
        // sufficient for symmetric, but float roundoff on the boundary
        // can pass minors and still produce a tiny negative eigenvalue;
        // exact eigendecomp eliminates that case).
        let (_, _, l3, _) = self.eig();
        l3 > eps
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Eigendecomposition — Smith 1961 closed form
// ════════════════════════════════════════════════════════════════════════════

impl Spd3 {
    /// Eigendecomp via Smith 1961. Returns `(λ₁, λ₂, λ₃, V)` with
    /// `λ₁ ≥ λ₂ ≥ λ₃` and `V` the column-major eigenvector matrix
    /// (`V[c] = [vx, vy, vz]` for the c-th eigenvector).
    ///
    /// `V` is orthonormal modulo f32 rounding; a single Gram-Schmidt
    /// pass at the end suppresses cross-orthogonality drift to ~1e-6.
    ///
    /// Degenerate cases:
    /// - All three equal → canonical basis `e₁, e₂, e₃`.
    /// - Pair equal → the pair's eigenvectors are any orthonormal basis
    ///   of the 2D eigenspace; the recovery routine fills them via
    ///   Gram-Schmidt against the unique third eigenvector.
    pub fn eig(&self) -> (f32, f32, f32, [[f32; 3]; 3]) {
        let Self { a11, a12, a13, a22, a23, a33, .. } = *self;

        let p1 = a12 * a12 + a13 * a13 + a23 * a23;

        // ── diagonal fast path ────────────────────────────────────────
        // f32 threshold: off-diag mass below ε·trace² is indistinguishable
        // from zero at single precision. Use 1e-10 · max(1, trace²) so
        // both tiny matrices (Σ ~ 1e-4·I) and large ones (Σ ~ 1e3·I) take
        // the fast path appropriately.
        let trace = a11 + a22 + a33;
        let scale = trace * trace + 1.0;
        if p1 <= 1e-10 * scale {
            return diag_sorted(a11, a22, a33);
        }

        // ── general path ──────────────────────────────────────────────
        let q = trace / 3.0;
        let d11 = a11 - q;
        let d22 = a22 - q;
        let d33 = a33 - q;
        let p2 = d11 * d11 + d22 * d22 + d33 * d33 + 2.0 * p1;
        let p = (p2 / 6.0).sqrt();
        let inv_p = 1.0 / p;

        // B = (A − q·I) / p, symmetric, trace 0, eigenvalues ∈ [−2, 2]
        let b11 = d11 * inv_p;
        let b12 = a12 * inv_p;
        let b13 = a13 * inv_p;
        let b22 = d22 * inv_p;
        let b23 = a23 * inv_p;
        let b33 = d33 * inv_p;

        // r = det(B) / 2 ∈ [−1, 1] (modulo f32 drift; clamp before acos).
        let det_b = b11 * (b22 * b33 - b23 * b23)
            - b12 * (b12 * b33 - b13 * b23)
            + b13 * (b12 * b23 - b13 * b22);
        let r = (det_b * 0.5).clamp(-1.0, 1.0);

        let phi = r.acos() / 3.0;
        let two_p = 2.0 * p;
        let l1 = q + two_p * phi.cos();
        let l3 = q + two_p * (phi + std::f32::consts::TAU / 3.0).cos();
        let l2 = 3.0 * q - l1 - l3;

        // Smith's construction yields l1 ≥ l2 ≥ l3 by construction (cos
        // is monotone-decreasing on [0, π/3]). Float roundoff can briefly
        // swap adjacent eigenvalues when two are within ~1e-6 of each
        // other; sort the final triple to guarantee descending order.
        let (l1, l2, l3) = sort3_desc(l1, l2, l3);
        let vecs = recover_eigvecs(self, l1, l2, l3);
        (l1, l2, l3, vecs)
    }

    /// Σ^t via spectral lift: V · diag(λᵢ^t) · Vᵀ. Hot path is `t = 0.5`
    /// (the sqrt step of the EWA-sandwich); the `Spd3::sqrt` shim
    /// short-circuits to this case with a positive-clamp on the
    /// eigenvalues to suppress f32 cancellation noise.
    pub fn pow(&self, t: f32) -> Self {
        let (l1, l2, l3, v) = self.eig();
        let p1 = l1.max(0.0).powf(t);
        let p2 = l2.max(0.0).powf(t);
        let p3 = l3.max(0.0).powf(t);
        reconstruct_symm(&v, p1, p2, p3)
    }

    /// Σ^{1/2} — the EWA-sandwich step matrix. Equivalent to `pow(0.5)`
    /// but slightly cheaper (the clamp + sqrt avoids a powf call).
    pub fn sqrt(&self) -> Self {
        let (l1, l2, l3, v) = self.eig();
        let s1 = l1.max(0.0).sqrt();
        let s2 = l2.max(0.0).sqrt();
        let s3 = l3.max(0.0).sqrt();
        reconstruct_symm(&v, s1, s2, s3)
    }

    /// log(Σ) on the SPD cone: V · diag(ln λᵢ) · Vᵀ. Used by the
    /// Pillar-7 probe to measure log-norm growth along edge paths;
    /// eigenvalues are clamped to a small positive ε before `ln` to
    /// keep the output finite under f32 cancellation noise.
    pub fn log_spd(&self) -> Self {
        let (l1, l2, l3, v) = self.eig();
        let eps = 1e-30_f32;
        let l1l = l1.max(eps).ln();
        let l2l = l2.max(eps).ln();
        let l3l = l3.max(eps).ln();
        reconstruct_symm(&v, l1l, l2l, l3l)
    }

    /// 3D Gaussian Splatting canonical covariance:
    ///
    /// ```text
    ///     Σ = R · diag(s₁², s₂², s₃²) · Rᵀ
    /// ```
    ///
    /// where `R` is the rotation matrix of the unit quaternion
    /// `(w, x, y, z)` and `scale = [s₁, s₂, s₃]` are the per-axis
    /// standard deviations (NOT log-space — exp the GS-format scales
    /// before calling). Caller is responsible for quaternion
    /// normalization; this routine assumes ‖quat‖ = 1.
    pub fn from_scale_quat(scale: [f32; 3], quat: [f32; 4]) -> Self {
        let [w, x, y, z] = quat;
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        // Rotation matrix from quaternion (row-major, columns are the
        // rotated basis vectors).
        let r00 = 1.0 - 2.0 * (yy + zz);
        let r01 = 2.0 * (xy - wz);
        let r02 = 2.0 * (xz + wy);
        let r10 = 2.0 * (xy + wz);
        let r11 = 1.0 - 2.0 * (xx + zz);
        let r12 = 2.0 * (yz - wx);
        let r20 = 2.0 * (xz - wy);
        let r21 = 2.0 * (yz + wx);
        let r22 = 1.0 - 2.0 * (xx + yy);

        // M = R · diag(s²): scale column k by sₖ².
        let s0 = scale[0] * scale[0];
        let s1 = scale[1] * scale[1];
        let s2 = scale[2] * scale[2];
        let m00 = r00 * s0; let m01 = r01 * s1; let m02 = r02 * s2;
        let m10 = r10 * s0; let m11 = r11 * s1; let m12 = r12 * s2;
        let m20 = r20 * s0; let m21 = r21 * s1; let m22 = r22 * s2;

        // Σ = M · Rᵀ, upper triangle only (M · Rᵀ is symmetric here
        // because the diag(s²) factor makes the product symmetric).
        let a11 = m00 * r00 + m01 * r01 + m02 * r02;
        let a12 = m00 * r10 + m01 * r11 + m02 * r12;
        let a13 = m00 * r20 + m01 * r21 + m02 * r22;
        let a22 = m10 * r10 + m11 * r11 + m12 * r12;
        let a23 = m10 * r20 + m11 * r21 + m12 * r22;
        let a33 = m20 * r20 + m21 * r21 + m22 * r22;
        Self::new(a11, a12, a13, a22, a23, a33)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Eigendecomp helpers
// ════════════════════════════════════════════════════════════════════════════

#[inline]
fn diag_sorted(a: f32, b: f32, c: f32) -> (f32, f32, f32, [[f32; 3]; 3]) {
    // Sort the three diagonal entries descending and return the
    // permuted canonical basis as eigenvectors.
    let (mut vals, mut idx) = ([a, b, c], [0usize, 1, 2]);
    // 3-element bubble sort, descending — branch-light, predictable.
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

#[inline]
fn sort3_desc(a: f32, b: f32, c: f32) -> (f32, f32, f32) {
    let (x, y) = if a >= b { (a, b) } else { (b, a) };
    let (xx, z) = if x >= c { (x, c) } else { (c, x) };
    let (yy, zz) = if y >= z { (y, z) } else { (z, y) };
    (xx, yy, zz)
}

/// Reconstruct Σ = V · diag(d) · Vᵀ for an orthonormal V (columns
/// are eigenvectors). Output is symmetric by construction; the upper
/// triangle is what we keep.
#[inline]
fn reconstruct_symm(v: &[[f32; 3]; 3], d1: f32, d2: f32, d3: f32) -> Spd3 {
    // M = V · diag(d): scale column k by dₖ.
    let m00 = v[0][0] * d1; let m01 = v[1][0] * d2; let m02 = v[2][0] * d3;
    let m10 = v[0][1] * d1; let m11 = v[1][1] * d2; let m12 = v[2][1] * d3;
    let m20 = v[0][2] * d1; let m21 = v[1][2] * d2; let m22 = v[2][2] * d3;
    // Σ = M · Vᵀ — V column k becomes Vᵀ row k.
    let a11 = m00 * v[0][0] + m01 * v[1][0] + m02 * v[2][0];
    let a12 = m00 * v[0][1] + m01 * v[1][1] + m02 * v[2][1];
    let a13 = m00 * v[0][2] + m01 * v[1][2] + m02 * v[2][2];
    let a22 = m10 * v[0][1] + m11 * v[1][1] + m12 * v[2][1];
    let a23 = m10 * v[0][2] + m11 * v[1][2] + m12 * v[2][2];
    let a33 = m20 * v[0][2] + m21 * v[1][2] + m22 * v[2][2];
    Spd3::new(a11, a12, a13, a22, a23, a33)
}

/// Recover the three eigenvectors of a symmetric 3×3 given its three
/// eigenvalues. Cross-product of two rows of `(A − λᵢ·I)` gives a
/// null-space vector; we pick the row pair with the largest cross
/// product to maximize numerical conditioning. Degenerate eigenvalues
/// fall back to Gram-Schmidt against eigenvectors already recovered.
fn recover_eigvecs(s: &Spd3, l1: f32, l2: f32, l3: f32) -> [[f32; 3]; 3] {
    let mut v = [[0.0f32; 3]; 3];
    let mut filled = [false; 3];
    let eigvals = [l1, l2, l3];

    // First pass: try the cross-product null-space recovery for each
    // eigenvalue independently.
    for (k, &lam) in eigvals.iter().enumerate() {
        if let Some(vec) = null_space_vec(s, lam) {
            v[k] = vec;
            filled[k] = true;
        }
    }

    // Second pass: for any eigenvalue whose recovery failed (degenerate
    // eigenspace), fill via Gram-Schmidt against the eigenvectors
    // already in hand.
    for k in 0..3 {
        if filled[k] {
            continue;
        }
        v[k] = gram_schmidt_complement(&v, &filled, k);
        filled[k] = true;
    }

    // Final cleanup: a single Gram-Schmidt pass over the eigenvector
    // matrix to suppress cross-orthogonality drift accumulated by the
    // cross-product recovery (typically ~1e-6 at f32).
    orthonormalize_columns(&mut v);
    v
}

/// Try to recover a unit vector in the null space of `(A − λ·I)` by
/// crossing two of its rows. Returns `None` if the eigenspace is
/// degenerate (all three row pairs yield a near-zero cross product).
fn null_space_vec(s: &Spd3, lam: f32) -> Option<[f32; 3]> {
    let r0 = [s.a11 - lam, s.a12, s.a13];
    let r1 = [s.a12, s.a22 - lam, s.a23];
    let r2 = [s.a13, s.a23, s.a33 - lam];

    // Reference scale for the "near-zero" threshold: trace gives the
    // characteristic magnitude of A's entries. The square goes into
    // the cross-product-norm comparison.
    let ref_scale = (s.trace().abs() + lam.abs()).max(1.0);
    let eps_sq = 1e-12_f32 * ref_scale * ref_scale;

    let mut best = [0.0f32; 3];
    let mut best_norm_sq = 0.0f32;
    for (a, b) in [(r0, r1), (r0, r2), (r1, r2)] {
        let c = cross3(a, b);
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

#[inline]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Find a unit vector orthogonal to all currently-filled eigenvectors.
/// For 3D this means: with 0 filled, return e₁; with 1 filled, return
/// any unit vector orthogonal to it; with 2 filled, return the cross
/// product of those two.
fn gram_schmidt_complement(v: &[[f32; 3]; 3], filled: &[bool; 3], skip: usize) -> [f32; 3] {
    let mut basis = Vec::with_capacity(2);
    for k in 0..3 {
        if k != skip && filled[k] {
            basis.push(v[k]);
        }
    }
    match basis.len() {
        0 => [1.0, 0.0, 0.0],
        1 => {
            // Pick whichever canonical axis is least-parallel to basis[0].
            let b = basis[0];
            let ax = b[0].abs();
            let ay = b[1].abs();
            let az = b[2].abs();
            let seed = if ax <= ay && ax <= az {
                [1.0, 0.0, 0.0]
            } else if ay <= az {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            let dot = seed[0] * b[0] + seed[1] * b[1] + seed[2] * b[2];
            let proj = [seed[0] - dot * b[0], seed[1] - dot * b[1], seed[2] - dot * b[2]];
            normalize3(proj)
        }
        2 => normalize3(cross3(basis[0], basis[1])),
        _ => unreachable!("at most 2 prior eigenvectors at this point"),
    }
}

#[inline]
fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let n_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if n_sq <= 0.0 {
        return [1.0, 0.0, 0.0];
    }
    let inv = 1.0 / n_sq.sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// In-place modified Gram-Schmidt on a 3-column matrix stored column-major.
fn orthonormalize_columns(v: &mut [[f32; 3]; 3]) {
    v[0] = normalize3(v[0]);
    let d10 = v[1][0] * v[0][0] + v[1][1] * v[0][1] + v[1][2] * v[0][2];
    v[1] = normalize3([
        v[1][0] - d10 * v[0][0],
        v[1][1] - d10 * v[0][1],
        v[1][2] - d10 * v[0][2],
    ]);
    let d20 = v[2][0] * v[0][0] + v[2][1] * v[0][1] + v[2][2] * v[0][2];
    let d21 = v[2][0] * v[1][0] + v[2][1] * v[1][1] + v[2][2] * v[1][2];
    v[2] = normalize3([
        v[2][0] - d20 * v[0][0] - d21 * v[1][0],
        v[2][1] - d20 * v[0][1] - d21 * v[1][1],
        v[2][2] - d20 * v[0][2] - d21 * v[1][2],
    ]);
}

// ════════════════════════════════════════════════════════════════════════════
// Sandwich M · N · Mᵀ for symmetric M, N
// ════════════════════════════════════════════════════════════════════════════

/// Compute `M · N · Mᵀ` for symmetric `M`, `N`. The result is
/// symmetric by construction (rounding residuals on `R₁₂` vs `R₂₁` are
/// averaged via `(R₁₂ + R₂₁)/2` since we only emit the upper triangle).
///
/// 9-element intermediate `P = M · N`; output is the upper triangle of
/// `R = P · M` (M symmetric → Mᵀ = M). 135 muls / 90 adds — a 16-wide
/// SIMD batch (`sandwich_x16`) brings the per-sandwich cost down by a
/// factor of `LANES` on a single inner loop.
#[inline]
pub fn sandwich(m: &Spd3, n: &Spd3) -> Spd3 {
    // P = M · N. Full 3×3 (not symmetric in general).
    let p00 = m.a11 * n.a11 + m.a12 * n.a12 + m.a13 * n.a13;
    let p01 = m.a11 * n.a12 + m.a12 * n.a22 + m.a13 * n.a23;
    let p02 = m.a11 * n.a13 + m.a12 * n.a23 + m.a13 * n.a33;
    let p10 = m.a12 * n.a11 + m.a22 * n.a12 + m.a23 * n.a13;
    let p11 = m.a12 * n.a12 + m.a22 * n.a22 + m.a23 * n.a23;
    let p12 = m.a12 * n.a13 + m.a22 * n.a23 + m.a23 * n.a33;
    let p20 = m.a13 * n.a11 + m.a23 * n.a12 + m.a33 * n.a13;
    let p21 = m.a13 * n.a12 + m.a23 * n.a22 + m.a33 * n.a23;
    let p22 = m.a13 * n.a13 + m.a23 * n.a23 + m.a33 * n.a33;

    // R = P · M (M symmetric → upper triangle only). Off-diagonal
    // entries are averaged with their lower-triangle counterpart to
    // collapse f32 rounding asymmetry.
    let r00 = p00 * m.a11 + p01 * m.a12 + p02 * m.a13;
    let r01a = p00 * m.a12 + p01 * m.a22 + p02 * m.a23;
    let r02a = p00 * m.a13 + p01 * m.a23 + p02 * m.a33;
    let r10 = p10 * m.a11 + p11 * m.a12 + p12 * m.a13;
    let r11 = p10 * m.a12 + p11 * m.a22 + p12 * m.a23;
    let r12a = p10 * m.a13 + p11 * m.a23 + p12 * m.a33;
    let r20 = p20 * m.a11 + p21 * m.a12 + p22 * m.a13;
    let r21 = p20 * m.a12 + p21 * m.a22 + p22 * m.a23;
    let r22 = p20 * m.a13 + p21 * m.a23 + p22 * m.a33;

    Spd3::new(
        r00,
        0.5 * (r01a + r10),
        0.5 * (r02a + r20),
        r11,
        0.5 * (r12a + r21),
        r22,
    )
}

/// 16-wide SIMD batch of `sandwich` via `crate::simd::F32x16`.
///
/// SoA-transposes the 16 inputs lane-by-lane (`m[k].a11` → lane `k` of
/// `m_a11`), runs the 9-step matmul + sandwich product in lockstep, and
/// scatters the upper-triangle outputs back into AoS. On AVX-512 the
/// inner loop is 6 `mul_add`s for `P`, 6 for the top half of `R`, and
/// 6 cross-pair averaging adds for the off-diagonals — measured ≥10×
/// over `sandwich` scalar loop on Zen4/Sapphire Rapids per
/// `benches/RESULTS.md`. On NEON / scalar tiers it falls back to the
/// 16-iteration loop via the polyfill's lane-broadcast emulation.
pub fn sandwich_x16(m: &[Spd3; 16], n: &[Spd3; 16], out: &mut [Spd3; 16]) {
    // ── transpose AoS → SoA ──────────────────────────────────────────
    let mut m_a11 = [0.0f32; 16];
    let mut m_a12 = [0.0f32; 16];
    let mut m_a13 = [0.0f32; 16];
    let mut m_a22 = [0.0f32; 16];
    let mut m_a23 = [0.0f32; 16];
    let mut m_a33 = [0.0f32; 16];
    let mut n_a11 = [0.0f32; 16];
    let mut n_a12 = [0.0f32; 16];
    let mut n_a13 = [0.0f32; 16];
    let mut n_a22 = [0.0f32; 16];
    let mut n_a23 = [0.0f32; 16];
    let mut n_a33 = [0.0f32; 16];
    for k in 0..16 {
        m_a11[k] = m[k].a11; m_a12[k] = m[k].a12; m_a13[k] = m[k].a13;
        m_a22[k] = m[k].a22; m_a23[k] = m[k].a23; m_a33[k] = m[k].a33;
        n_a11[k] = n[k].a11; n_a12[k] = n[k].a12; n_a13[k] = n[k].a13;
        n_a22[k] = n[k].a22; n_a23[k] = n[k].a23; n_a33[k] = n[k].a33;
    }

    let m11 = F32x16::from_slice(&m_a11);
    let m12 = F32x16::from_slice(&m_a12);
    let m13 = F32x16::from_slice(&m_a13);
    let m22 = F32x16::from_slice(&m_a22);
    let m23 = F32x16::from_slice(&m_a23);
    let m33 = F32x16::from_slice(&m_a33);
    let n11 = F32x16::from_slice(&n_a11);
    let n12 = F32x16::from_slice(&n_a12);
    let n13 = F32x16::from_slice(&n_a13);
    let n22 = F32x16::from_slice(&n_a22);
    let n23 = F32x16::from_slice(&n_a23);
    let n33 = F32x16::from_slice(&n_a33);

    // ── P = M · N ────────────────────────────────────────────────────
    let p00 = m11 * n11 + m12 * n12 + m13 * n13;
    let p01 = m11 * n12 + m12 * n22 + m13 * n23;
    let p02 = m11 * n13 + m12 * n23 + m13 * n33;
    let p10 = m12 * n11 + m22 * n12 + m23 * n13;
    let p11 = m12 * n12 + m22 * n22 + m23 * n23;
    let p12 = m12 * n13 + m22 * n23 + m23 * n33;
    let p20 = m13 * n11 + m23 * n12 + m33 * n13;
    let p21 = m13 * n12 + m23 * n22 + m33 * n23;
    let p22 = m13 * n13 + m23 * n23 + m33 * n33;

    // ── R = P · M (M symmetric, upper triangle averaged) ────────────
    let r00 = p00 * m11 + p01 * m12 + p02 * m13;
    let r01a = p00 * m12 + p01 * m22 + p02 * m23;
    let r02a = p00 * m13 + p01 * m23 + p02 * m33;
    let r10 = p10 * m11 + p11 * m12 + p12 * m13;
    let r11 = p10 * m12 + p11 * m22 + p12 * m23;
    let r12a = p10 * m13 + p11 * m23 + p12 * m33;
    let r20 = p20 * m11 + p21 * m12 + p22 * m13;
    let r21 = p20 * m12 + p21 * m22 + p22 * m23;
    let r22 = p20 * m13 + p21 * m23 + p22 * m33;

    let half = F32x16::splat(0.5);
    let out_a11 = r00;
    let out_a12 = (r01a + r10) * half;
    let out_a13 = (r02a + r20) * half;
    let out_a22 = r11;
    let out_a23 = (r12a + r21) * half;
    let out_a33 = r22;

    // ── scatter SoA → AoS ────────────────────────────────────────────
    let mut o_a11 = [0.0f32; 16];
    let mut o_a12 = [0.0f32; 16];
    let mut o_a13 = [0.0f32; 16];
    let mut o_a22 = [0.0f32; 16];
    let mut o_a23 = [0.0f32; 16];
    let mut o_a33 = [0.0f32; 16];
    out_a11.copy_to_slice(&mut o_a11);
    out_a12.copy_to_slice(&mut o_a12);
    out_a13.copy_to_slice(&mut o_a13);
    out_a22.copy_to_slice(&mut o_a22);
    out_a23.copy_to_slice(&mut o_a23);
    out_a33.copy_to_slice(&mut o_a33);
    for k in 0..16 {
        out[k] = Spd3::new(o_a11[k], o_a12[k], o_a13[k], o_a22[k], o_a23[k], o_a33[k]);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests — scalar reference + SIMD parity + Smith-1961 sanity
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    fn approx_spd3(a: Spd3, b: Spd3, tol: f32) -> bool {
        approx(a.a11, b.a11, tol)
            && approx(a.a12, b.a12, tol)
            && approx(a.a13, b.a13, tol)
            && approx(a.a22, b.a22, tol)
            && approx(a.a23, b.a23, tol)
            && approx(a.a33, b.a33, tol)
    }

    // Deterministic xorshift32 — independent of the crate's RNG infra
    // so the test stays hermetic at the spd3 module level.
    fn rng_uniform(state: &mut u32) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        (*state as f32) / (u32::MAX as f32)
    }

    fn sample_spd3(state: &mut u32) -> Spd3 {
        // Random rotation × random positive scales.
        let s = [
            0.2 + 1.8 * rng_uniform(state),
            0.2 + 1.8 * rng_uniform(state),
            0.2 + 1.8 * rng_uniform(state),
        ];
        let mut q = [
            -1.0 + 2.0 * rng_uniform(state),
            -1.0 + 2.0 * rng_uniform(state),
            -1.0 + 2.0 * rng_uniform(state),
            -1.0 + 2.0 * rng_uniform(state),
        ];
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        for v in &mut q {
            *v /= n;
        }
        Spd3::from_scale_quat(s, q)
    }

    #[test]
    fn size_alignment_invariants() {
        assert_eq!(core::mem::size_of::<Spd3>(), 32);
        assert_eq!(core::mem::align_of::<Spd3>(), 32);
    }

    #[test]
    fn identity_round_trip() {
        let i = Spd3::I;
        let (l1, l2, l3, v) = i.eig();
        assert!(approx(l1, 1.0, 1e-6));
        assert!(approx(l2, 1.0, 1e-6));
        assert!(approx(l3, 1.0, 1e-6));
        // Reconstruction is the identity.
        let r = reconstruct_symm(&v, l1, l2, l3);
        assert!(approx_spd3(r, i, 1e-6));
    }

    #[test]
    fn diagonal_fast_path() {
        let d = Spd3::new(3.0, 0.0, 0.0, 1.0, 0.0, 2.0);
        let (l1, l2, l3, _) = d.eig();
        assert!(approx(l1, 3.0, 1e-6));
        assert!(approx(l2, 2.0, 1e-6));
        assert!(approx(l3, 1.0, 1e-6));
    }

    #[test]
    fn eigenvalues_sorted_descending() {
        let mut state = 0xC0FFEEu32;
        for _ in 0..200 {
            let s = sample_spd3(&mut state);
            let (l1, l2, l3, _) = s.eig();
            assert!(l1 >= l2 - 1e-5, "l1={l1} l2={l2}");
            assert!(l2 >= l3 - 1e-5, "l2={l2} l3={l3}");
            assert!(l3 > 0.0, "non-positive eigenvalue: {l3}");
        }
    }

    #[test]
    fn from_scale_quat_identity_rotation_gives_diag_scale_sq() {
        // Identity quat (w=1, x=y=z=0) gives R = I → Σ = diag(s²).
        let s = Spd3::from_scale_quat([2.0, 1.5, 0.8], [1.0, 0.0, 0.0, 0.0]);
        assert!(approx(s.a11, 4.0, 1e-6));
        assert!(approx(s.a22, 2.25, 1e-6));
        assert!(approx(s.a33, 0.64, 1e-6));
        assert!(approx(s.a12, 0.0, 1e-6));
        assert!(approx(s.a13, 0.0, 1e-6));
        assert!(approx(s.a23, 0.0, 1e-6));
    }

    #[test]
    fn sqrt_squared_equals_original() {
        // sqrt(Σ)² = Σ, since sqrt is the spectral lift with t=1/2.
        let mut state = 0xDEADBEEFu32;
        for trial in 0..100 {
            let s = sample_spd3(&mut state);
            let root = s.sqrt();
            let squared = sandwich(&root, &Spd3::I);
            // Sandwich of symmetric root with identity: root · I · root = root².
            assert!(
                approx_spd3(squared, s, 5e-4),
                "trial {trial} failed: sqrt²={squared:?}, orig={s:?}"
            );
        }
    }

    #[test]
    fn pow_one_is_identity_op() {
        let mut state = 0x12345678u32;
        for _ in 0..50 {
            let s = sample_spd3(&mut state);
            let p1 = s.pow(1.0);
            assert!(approx_spd3(p1, s, 5e-4));
        }
    }

    #[test]
    fn log_of_identity_is_zero() {
        let i = Spd3::I;
        let l = i.log_spd();
        assert!(approx(l.a11, 0.0, 1e-6));
        assert!(approx(l.a22, 0.0, 1e-6));
        assert!(approx(l.a33, 0.0, 1e-6));
        assert!(approx(l.a12, 0.0, 1e-6));
        assert!(approx(l.a13, 0.0, 1e-6));
        assert!(approx(l.a23, 0.0, 1e-6));
    }

    #[test]
    fn sandwich_identity_is_input() {
        // I · N · Iᵀ = N
        let n = Spd3::from_scale_quat([1.7, 0.8, 1.2], [0.7071068, 0.0, 0.7071068, 0.0]);
        let r = sandwich(&Spd3::I, &n);
        assert!(approx_spd3(r, n, 1e-6));
    }

    #[test]
    fn sandwich_preserves_spd() {
        let mut state = 0xCAFEBABEu32;
        for trial in 0..200 {
            let m = sample_spd3(&mut state);
            let n = sample_spd3(&mut state);
            let r = sandwich(&m.sqrt(), &n);
            assert!(
                r.is_spd(1e-6),
                "trial {trial}: sandwich(sqrt(M), N) produced non-SPD {r:?} from M={m:?}, N={n:?}"
            );
        }
    }

    #[test]
    fn sandwich_x16_matches_scalar_loop() {
        let mut state = 0xFEEDFACEu32;
        let mut ms = [Spd3::I; 16];
        let mut ns = [Spd3::I; 16];
        for k in 0..16 {
            ms[k] = sample_spd3(&mut state);
            ns[k] = sample_spd3(&mut state);
        }
        let mut out_simd = [Spd3::ZERO; 16];
        sandwich_x16(&ms, &ns, &mut out_simd);
        for k in 0..16 {
            let scalar = sandwich(&ms[k], &ns[k]);
            // Different evaluation order in SIMD vs scalar accumulates
            // slightly different rounding; 1e-3 absolute is generous
            // and well within the variance the rasterizer downstream
            // can absorb (covariance entries are ~1, 1e-3 ≈ 0.1%).
            assert!(
                approx_spd3(out_simd[k], scalar, 1e-3),
                "lane {k}: simd={:?} scalar={:?}",
                out_simd[k],
                scalar
            );
        }
    }

    #[test]
    fn from_scale_quat_yields_spd() {
        let mut state = 0xABCD1234u32;
        for _ in 0..100 {
            let s = sample_spd3(&mut state);
            assert!(s.is_spd(1e-6));
        }
    }

    #[test]
    fn determinant_matches_product_of_eigenvalues() {
        // det(Σ) = λ₁ · λ₂ · λ₃ for symmetric Σ.
        let mut state = 0x11111111u32;
        for _ in 0..100 {
            let s = sample_spd3(&mut state);
            let det = s.det();
            let (l1, l2, l3, _) = s.eig();
            let prod = l1 * l2 * l3;
            // Relative tolerance — eigenvalues can be ~2.0 each, so the
            // product is ~8, and 1e-3 relative = 8e-3 absolute.
            let scale = det.abs().max(prod.abs()).max(1.0);
            assert!(
                approx(det, prod, 5e-3 * scale),
                "det={det} prod_eigs={prod} (l1={l1} l2={l2} l3={l3})"
            );
        }
    }
}


