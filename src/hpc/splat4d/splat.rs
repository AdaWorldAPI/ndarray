//! Splat primitives for the cognitive-spacetime cascade (PR-X4 A3).
//!
//! A [`Splat<D>`] is one anisotropic Gaussian in `D`-dimensional inquiry
//! space (D=2 screen / cell space, D=3 scene space). It carries a typed
//! cognitive [`SplatCell`] payload and a NARS [`Splat::confidence`] sort
//! key. The binner ([`super::tile`]/[`super::bin`]) places each splat into
//! the L1–L4 tiles its 3σ footprint covers; the compositor
//! ([`super::compose`]) blends the cells.
//!
//! # No-float scope note
//!
//! `Splat` uses `f32` for `pos` / covariance / `confidence`. This is the
//! **graphics lane** (sibling of `hpc::splat3d`), NOT the cognitive
//! distance kernel. The substrate's no-float discipline applies to
//! similarity/distance (`cam_pq` ADC, popcount) — not to splat geometry.
//! The cognitive *payload* ([`SplatCell`]) is integer-only (q8/INT4).
//!
//! # Distance-typing guardrail
//!
//! Per `.claude/knowledge/cognitive-distance-typing.md`, there is no
//! `Splat::distance_to` / `enum DistanceMetric` here. Cognitive distance
//! between splats is a typed metric that lives elsewhere; this module is
//! pure geometry + payload.
//!
//! # Dimension bound
//!
//! v1 supports `D ∈ {1, 2, 3}` (the binner additionally requires `D ≥ 2`).
//! The [`SplatCovariance::Cholesky`] factor is backed by a fixed
//! `[f32; 6]` (the `D ≤ 3` lower-triangular budget) because a
//! `[f32; { D*(D+1)/2 }]` length needs the still-unstable
//! `generic_const_exprs` feature. Constructors enforce `D ≤ 3` at compile
//! time via an inline `const` assertion.

/// Maximum supported inquiry dimension in v1. The packed Cholesky factor
/// `[f32; LT_LEN_MAX]` holds the lower triangle for `D ≤ MAX_SPLAT_DIM`.
pub const MAX_SPLAT_DIM: usize = 3;

/// Lower-triangular element budget for `D ≤ MAX_SPLAT_DIM`
/// (`3 * 4 / 2 = 6`).
pub const LT_LEN_MAX: usize = MAX_SPLAT_DIM * (MAX_SPLAT_DIM + 1) / 2;

// ════════════════════════════════════════════════════════════════════
// SplatCell — typed cognitive payload
// ════════════════════════════════════════════════════════════════════

/// Typed cognitive cell state carried by a splat (forward-compatible with
/// PR-X7's `cognitive_shader!` macro). Integer-only: no floats here.
///
/// Layout (`repr(C, align(8))`): `edge` (8) + `thinking` (16) + `qualia`
/// (8) + `vocab` (2) → 34 bytes, rounded up to **40** by the 8-byte
/// alignment.
///
/// ```
/// use ndarray::hpc::splat4d::SplatCell;
/// assert_eq!(core::mem::size_of::<SplatCell>(), 40);
/// assert_eq!(core::mem::align_of::<SplatCell>(), 8);
/// ```
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplatCell {
    /// CausalEdge64 mantissa — the collapsed truth / particle state.
    pub edge: u64,
    /// 32-dim thinking-style vector, INT4-packed (2 nibbles/byte → 16 B).
    pub thinking: [u8; 16],
    /// 16-dim qualia vector, INT4-packed (8 B).
    pub qualia: [u8; 8],
    /// 12-bit codebook index into the 4096-atom CAM vocabulary.
    pub vocab: u16,
}

impl SplatCell {
    /// Construct a cell from its four fields.
    pub fn new(edge: u64, thinking: [u8; 16], qualia: [u8; 8], vocab: u16) -> Self {
        Self {
            edge,
            thinking,
            qualia,
            vocab,
        }
    }
}

impl Default for SplatCell {
    fn default() -> Self {
        Self {
            edge: 0,
            thinking: [0; 16],
            qualia: [0; 8],
            vocab: 0,
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// SplatCovariance — anisotropic footprint shape
// ════════════════════════════════════════════════════════════════════

/// Anisotropic covariance of a splat's footprint, in three representations.
///
/// All three are `Copy` (no heap), keeping [`Splat`] cheap to pass by
/// value. The [`Cholesky`](SplatCovariance::Cholesky) factor is a packed
/// lower triangle in row-major order: for `D = 3` the six entries are
/// `[L00, L10, L11, L20, L21, L22]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SplatCovariance<const D: usize> {
    /// Isotropic: identity scaled by `sigma2` (variance, σ²) on every axis.
    Isotropic {
        /// Per-axis variance σ².
        sigma2: f32,
    },
    /// Axis-aligned anisotropy: per-axis variance.
    Diagonal {
        /// Per-axis variance σ²ᵢ.
        diag: [f32; D],
    },
    /// Full anisotropy as a lower-triangular Cholesky factor `L`
    /// (covariance = `L · Lᵀ`). Only the first `D*(D+1)/2` entries of
    /// `lt` are used; the rest are ignored. Capped at `D ≤ MAX_SPLAT_DIM`.
    Cholesky {
        /// Packed lower triangle of `L`, row-major (`lt[i*(i+1)/2 + j]` =
        /// `L[i][j]` for `j ≤ i`).
        lt: [f32; LT_LEN_MAX],
    },
}

/// Packed lower-triangular index: `L[i][j]` (with `j ≤ i`) lives at this
/// offset in the row-major lower-triangle array.
#[inline]
const fn lt_index(i: usize, j: usize) -> usize {
    i * (i + 1) / 2 + j
}

impl<const D: usize> SplatCovariance<D> {
    /// Isotropic covariance (variance σ² on every axis).
    pub fn isotropic(sigma2: f32) -> Self {
        SplatCovariance::Isotropic { sigma2 }
    }

    /// Diagonal (axis-aligned) covariance.
    pub fn diagonal(diag: [f32; D]) -> Self {
        SplatCovariance::Diagonal { diag }
    }

    /// Full Cholesky-factor covariance. `lt` holds the packed lower
    /// triangle (`L[i][j]` at `lt[i*(i+1)/2 + j]`).
    ///
    /// `D ≤ MAX_SPLAT_DIM` is enforced at compile time.
    pub fn cholesky(lt: [f32; LT_LEN_MAX]) -> Self {
        const { assert!(D <= MAX_SPLAT_DIM, "SplatCovariance::cholesky: D must be <= MAX_SPLAT_DIM (3)") };
        SplatCovariance::Cholesky { lt }
    }

    /// The 3σ half-extent of the footprint along `axis` (the bounding-box
    /// radius used for tile binning). Always non-negative.
    ///
    /// For `Cholesky`, the per-axis variance is `(L·Lᵀ)[axis][axis] =
    /// Σ_{j ≤ axis} L[axis][j]²`.
    ///
    /// ```
    /// use ndarray::hpc::splat4d::SplatCovariance;
    /// let cov = SplatCovariance::<2>::isotropic(4.0); // σ² = 4 → σ = 2
    /// assert_eq!(cov.sigma3_extent(0), 6.0);          // 3σ = 6
    /// assert_eq!(cov.sigma3_extent(1), 6.0);
    /// ```
    pub fn sigma3_extent(&self, axis: usize) -> f32 {
        let var = match self {
            SplatCovariance::Isotropic { sigma2 } => *sigma2,
            SplatCovariance::Diagonal { diag } => diag[axis],
            SplatCovariance::Cholesky { lt } => {
                // `.get(..).unwrap_or(0.0)` keeps this panic-free even if a
                // caller bypasses the `cholesky()` D≤3 guard by constructing
                // the (pub) variant directly with `D > MAX_SPLAT_DIM` — the
                // out-of-budget triangle entries read as 0. For the supported
                // D ≤ 3 every index is < LT_LEN_MAX, so this is exact.
                let mut v = 0.0f32;
                let mut j = 0;
                while j <= axis {
                    let l = lt.get(lt_index(axis, j)).copied().unwrap_or(0.0);
                    v += l * l;
                    j += 1;
                }
                v
            }
        };
        3.0 * var.max(0.0).sqrt()
    }
}

// ════════════════════════════════════════════════════════════════════
// Splat<D>
// ════════════════════════════════════════════════════════════════════

/// One Gaussian splat in `D`-dimensional cognitive-spacetime.
///
/// `pos[0]` is the row coordinate and `pos[1]` the column coordinate in
/// cell space (the binner uses axes 0 and 1). Higher axes (D=3: depth /
/// scene-Z) are carried for scene-space operations but not used by the 2-D
/// tile binner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Splat<const D: usize> {
    /// Position in cognitive-spacetime (cell space for axes 0,1).
    pub pos: [f32; D],
    /// Anisotropic footprint covariance.
    pub cov: SplatCovariance<D>,
    /// Typed cognitive payload.
    pub cell: SplatCell,
    /// NARS truth projection — the tile-list sort key (highest composited
    /// first).
    pub confidence: f32,
}

impl<const D: usize> Splat<D> {
    /// Construct a splat. Enforces `1 ≤ D ≤ MAX_SPLAT_DIM` at compile time.
    pub fn new(pos: [f32; D], cov: SplatCovariance<D>, cell: SplatCell, confidence: f32) -> Self {
        const { assert!(D >= 1 && D <= MAX_SPLAT_DIM, "Splat: D must be in 1..=MAX_SPLAT_DIM (3)") };
        Self {
            pos,
            cov,
            cell,
            confidence,
        }
    }

    /// Axis-aligned bounding box of the 3σ footprint, in cell space:
    /// returns `(min, max)` where `min[i] = pos[i] - 3σᵢ` and
    /// `max[i] = pos[i] + 3σᵢ`.
    ///
    /// ```
    /// use ndarray::hpc::splat4d::{Splat, SplatCovariance, SplatCell};
    /// let s = Splat::<2>::new([10.0, 20.0], SplatCovariance::isotropic(4.0), SplatCell::default(), 1.0);
    /// let (mn, mx) = s.aabb_3sigma();
    /// assert_eq!(mn, [4.0, 14.0]);  // 10-6, 20-6
    /// assert_eq!(mx, [16.0, 26.0]); // 10+6, 20+6
    /// ```
    pub fn aabb_3sigma(&self) -> ([f32; D], [f32; D]) {
        let mut mn = [0.0f32; D];
        let mut mx = [0.0f32; D];
        for axis in 0..D {
            let ext = self.cov.sigma3_extent(axis);
            mn[axis] = self.pos[axis] - ext;
            mx[axis] = self.pos[axis] + ext;
        }
        (mn, mx)
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splat_cell_is_40_bytes_aligned_8() {
        assert_eq!(core::mem::size_of::<SplatCell>(), 40);
        assert_eq!(core::mem::align_of::<SplatCell>(), 8);
    }

    #[test]
    fn splat_cell_default_is_zero() {
        let c = SplatCell::default();
        assert_eq!(c.edge, 0);
        assert_eq!(c.thinking, [0; 16]);
        assert_eq!(c.qualia, [0; 8]);
        assert_eq!(c.vocab, 0);
    }

    #[test]
    fn isotropic_extent_is_3sigma() {
        let cov = SplatCovariance::<3>::isotropic(9.0); // σ = 3
        for axis in 0..3 {
            assert_eq!(cov.sigma3_extent(axis), 9.0); // 3σ = 9
        }
    }

    #[test]
    fn diagonal_extent_per_axis() {
        let cov = SplatCovariance::<2>::diagonal([4.0, 16.0]); // σ = 2, 4
        assert_eq!(cov.sigma3_extent(0), 6.0);
        assert_eq!(cov.sigma3_extent(1), 12.0);
    }

    #[test]
    fn cholesky_extent_matches_diagonal_when_off_diagonal_zero() {
        // L = diag(2, 4) → covariance = diag(4, 16); same as the diagonal case.
        // Packed lower-tri (D=2): [L00, L10, L11] = [2, 0, 4].
        let mut lt = [0.0f32; LT_LEN_MAX];
        lt[lt_index(0, 0)] = 2.0;
        lt[lt_index(1, 0)] = 0.0;
        lt[lt_index(1, 1)] = 4.0;
        let cov = SplatCovariance::<2>::cholesky(lt);
        assert_eq!(cov.sigma3_extent(0), 6.0); // 3·√4
        assert_eq!(cov.sigma3_extent(1), 12.0); // 3·√16
    }

    #[test]
    fn cholesky_extent_includes_off_diagonal() {
        // L row 1 = [L10=3, L11=4] → variance axis1 = 9 + 16 = 25 → σ = 5.
        let mut lt = [0.0f32; LT_LEN_MAX];
        lt[lt_index(0, 0)] = 1.0;
        lt[lt_index(1, 0)] = 3.0;
        lt[lt_index(1, 1)] = 4.0;
        let cov = SplatCovariance::<2>::cholesky(lt);
        assert_eq!(cov.sigma3_extent(1), 15.0); // 3·√25
    }

    #[test]
    fn aabb_centered_on_pos() {
        let s = Splat::<2>::new([100.0, 50.0], SplatCovariance::isotropic(1.0), SplatCell::default(), 0.5);
        let (mn, mx) = s.aabb_3sigma();
        assert_eq!(mn, [97.0, 47.0]); // 3σ = 3
        assert_eq!(mx, [103.0, 53.0]);
    }

    #[test]
    fn cholesky_extent_is_panic_free_when_variant_bypasses_d_guard() {
        // P2 hardening: the Cholesky variant is pub, so a caller can build
        // SplatCovariance::<4>::Cholesky directly (skipping cholesky()'s D≤3
        // const guard). sigma3_extent on an out-of-budget axis must NOT panic
        // — out-of-range triangle entries read as 0 via `.get`.
        let cov = SplatCovariance::<4>::Cholesky { lt: [1.0; LT_LEN_MAX] };
        let _ = cov.sigma3_extent(3); // lt_index(3,3)=9 > 5 → no panic
    }

    #[test]
    fn splat_is_copy() {
        // Compile-time proof that Splat<D> is Copy (no heap, cheap by-value).
        fn assert_copy<T: Copy>() {}
        assert_copy::<Splat<2>>();
        assert_copy::<Splat<3>>();
        assert_copy::<SplatCell>();
        assert_copy::<SplatCovariance<3>>();
    }
}
