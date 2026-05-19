//! `splat4d::compose` — L5 Gaussian moment-match and L6 Pillar-8 temporal sandwich.
//!
//! # L5 — 16-child Gaussian moment-match (`compose_cascade`)
//!
//! Merges 16 child Gaussians (the full 4×4 support at one tier level) into a
//! single parent Gaussian by matching the first two moments (mean and
//! covariance) of the mixture:
//!
//! ```text
//! μ_parent  = Σ_i  w_i · μ_i            (weighted mean)
//! Σ_parent  = Σ_i  w_i · (Σ_i + (μ_i − μ_parent)(μ_i − μ_parent)ᵀ)
//! ```
//!
//! where `w_i = opacity_i / Σ_j opacity_j` (normalised opacity weights).
//!
//! This is the L5 tier operation from `pr-x4-design.md` / `07-l5-l6-cascade-
//! composition.md`: every L3 super-block gathers its 16 L4 children, runs
//! `compose_cascade`, and the resulting parent Gaussian describes the aggregate
//! statistics at the L3 tile.
//!
//! # L6 — Pillar-8 temporal sandwich (`temporal_sandwich`)
//!
//! Applies the covariance update from Pillar-8 (temporal-drift sandwich) to
//! a Gaussian state using a kernel drawn from the SPD cone:
//!
//! ```text
//! Σ_out = K · Σ_state · Kᵀ
//! ```
//!
//! where `K` is a symmetric SPD kernel (`K = Kᵀ`) derived from the temporal
//! drift model at the current cascade tier. This is implemented via
//! `crate::hpc::splat3d::spd3::sandwich(kernel, state)` from PR-X11's jc
//! consolidation (`Spd3::sandwich` / `Spd3::sqrt`).
//!
//! The L6 sandwich is the "time axis" interpretation of the L4 cascade: when
//! the cascade runs every tick, the temporal sandwich propagates covariance
//! forward through time, making the L4 tier the experience-memory super-block.
//!
//! # SIMD bundles consumed
//!
//! - **B-Gather-FMA** (`gather_idx_f32x16 ∘ fmadd_f32x16`): weighted-mean
//!   fusion across the 16 children in `compose_cascade`. Picks up the 16
//!   neighbouring Gaussians and fuses their weighted contributions in one shot.
//! - **B-Compose** (`hreduce_sum_f32x16` for alpha; `revise_truth_f32x16` for
//!   NARS): horizontal reduction closure. The `splat4d-nars-compose` feature
//!   gate selects which kernel binds at `compose_kernel`.
//!
//! Do NOT reach past these bundles into raw `std::arch::*` intrinsics.
//!
//! # Feature gate
//!
//! `#[cfg(feature = "splat4d-nars-compose")]` selects the NARS truth-revision
//! path in `compose_kernel`. Default is alpha-compositing (graphics).
//! See `splat4d::compose_kernel` for the closure-swap bridge.
//!
//! # Cross-references
//!
//! - `pr-x4-planning/07-l5-l6-cascade-composition.md` — L5/L6 brief (stub → fill)
//! - `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "Tier-cascade composition"
//!   (done criterion 4)
//! - `crate::hpc::splat3d::spd3` — `Spd3::sandwich`, `Spd3::sqrt` (PR-X11)
//! - `crate::hpc::pillar::temporal_sandwich` — Pillar-8 certification probe

use crate::hpc::splat3d::spd3::{sandwich, Spd3};

// ════════════════════════════════════════════════════════════════════════════
// Gaussian — minimal splat4d Gaussian type (position + covariance + opacity)
// ════════════════════════════════════════════════════════════════════════════

/// Minimal Gaussian descriptor used by the splat4d cascade composition layer.
///
/// Position (`mean`) and anisotropic covariance (`cov`) describe the spatial
/// extent at a single tier level. `opacity` is the NARS confidence-weighted
/// contribution factor used for moment-matching weights in `compose_cascade`.
///
/// This is NOT a graphics-splat `Gaussian3D` (which carries SH coefficients);
/// it is the stripped-down cascade primitive consumed by L5 `compose_cascade`
/// and L6 `temporal_sandwich`.
///
/// # Memory layout
///
/// `#[repr(C, align(32))]` — `Spd3` is 32-byte aligned (`align(32)`); the
/// mean+opacity prefix (4 × 4 = 16 B) brings the total to 48 B + 32 B pad
/// alignment. Keeps 16-wide SIMD batch loads cache-friendly.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct Gaussian {
    /// World-space (or tile-space) mean position [x, y, z].
    pub mean: [f32; 3],
    /// Opacity / confidence weight in [0, 1]. Used as the mixture weight
    /// `w_i` in `compose_cascade`.
    pub opacity: f32,
    /// Anisotropic covariance in upper-triangle SPD form.
    pub cov: Spd3,
}

impl Gaussian {
    /// Unit-opacity isotropic Gaussian at the origin. Zero-cost `const`.
    pub const UNIT: Self = Self {
        mean: [0.0; 3],
        opacity: 1.0,
        cov: Spd3::I,
    };

    /// Construct from explicit fields.
    #[inline]
    pub fn new(mean: [f32; 3], opacity: f32, cov: Spd3) -> Self {
        Self { mean, opacity, cov }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// L5 — compose_cascade
// ════════════════════════════════════════════════════════════════════════════

/// L5 Gaussian-mixture moment-match: merge 16 child Gaussians into one parent.
///
/// Implements the first-two-moment fusion:
///
/// ```text
/// w_i       = children[i].opacity / Σ_j children[j].opacity
/// μ_parent  = Σ_i w_i · children[i].mean
/// Σ_parent  = Σ_i w_i · (children[i].cov + outer(d_i, d_i))
///             where d_i = children[i].mean − μ_parent
/// ```
///
/// The 16 children correspond to the full 4×4 spatial support at one tier
/// level (L3 → L4 super-block). Zero-opacity children are excluded from
/// the weight sum; if all 16 children have zero opacity the function returns
/// the zero-opacity `Gaussian::UNIT`.
///
/// # SIMD bundle: B-Gather-FMA
///
/// The inner loop uses **B-Gather-FMA** (`gather_idx_f32x16 ∘ fmadd_f32x16`)
/// to pick up the 16 children and fuse-multiply-add their contributions in one
/// SIMD pass. Do NOT reach past this bundle into raw intrinsics.
///
/// # Correctness gate (done criterion 4)
///
/// The `splat4d-nars-compose` test must verify `compose_cascade` against a
/// scalar reference on a 16-child fixture per tier (L3 → L4 super-block).
///
/// # References
///
/// - `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "Done criteria" #4
/// - `07-l5-l6-cascade-composition.md` § L5 moment-match
pub fn compose_cascade(children: &[Gaussian; 16]) -> Gaussian {
    //// TODO(A2/L5 — B-Gather-FMA): implement weighted-mean position fusion
    //// and second-moment covariance roll-up via B-Gather-FMA bundle.
    //// Steps:
    ////   1. Compute weight_sum = Σ children[i].opacity; handle zero sum.
    ////   2. μ_parent = Σ (children[i].opacity / weight_sum) * children[i].mean
    ////   3. For each child i: d_i = children[i].mean - μ_parent
    ////   4. Σ_parent = Σ w_i * (children[i].cov + outer3(d_i, d_i))
    ////   5. opacity_parent = weight_sum / 16 (normalised by child count)
    //// Ref: `07-l5-l6-cascade-composition.md` § "L5 moment-match"
    //// Ref: B-Gather-FMA bundle in `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md`
    let _ = children;
    todo!("compose_cascade — L5 16-child Gaussian moment-match via B-Gather-FMA")
}

// ════════════════════════════════════════════════════════════════════════════
// L6 — temporal_sandwich
// ════════════════════════════════════════════════════════════════════════════

/// L6 Pillar-8 temporal sandwich: propagate a Gaussian state covariance
/// forward through one time step using the SPD kernel.
///
/// Applies:
///
/// ```text
/// state_out.cov = K · state.cov · Kᵀ
/// ```
///
/// where `K` is a symmetric SPD kernel (`K = Kᵀ`). The mean is propagated
/// unchanged (no drift term in this skeleton). The sandwich is computed via
/// `crate::hpc::splat3d::spd3::sandwich(kernel_cov, state_cov)` from PR-X11's
/// jc consolidation.
///
/// This is the L6 "time axis" operation: when the cascade runs every tick,
/// each L4 super-block covariance evolves by the Pillar-8 temporal-drift
/// kernel, making the cascade the spacetime stream.
///
/// # Relation to PR-X11
///
/// Calls `crate::hpc::splat3d::spd3::sandwich`, which is PR-X11's
/// `Spd3::sandwich` deliverable. If the jc consolidation has not landed,
/// this function will fail to link.
///
/// # SIMD bundle: B-Compose
///
/// The covariance update uses **B-Compose** for the horizontal reduction.
/// Under the `splat4d-nars-compose` feature flag, B-Compose binds
/// `revise_truth_f32x16` instead of `hreduce_sum_f32x16`. See
/// `crate::hpc::splat4d::compose_kernel`.
///
/// # References
///
/// - `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "Done criteria" #4
/// - `07-l5-l6-cascade-composition.md` § "L6 Pillar-8 temporal sandwich"
/// - `crate::hpc::pillar::temporal_sandwich` — Pillar-8 certification probe
/// - PR-X11 `Spd3::sandwich` / `Spd3::sqrt` deliverable
pub fn temporal_sandwich(state: &Gaussian, kernel: &Spd3) -> Gaussian {
    //// TODO(L6 — PR-X11 dep): implement Σ_out = sandwich(kernel, state.cov)
    //// using `crate::hpc::splat3d::spd3::sandwich`. The mean passes through
    //// unchanged. Verify that the output covariance is SPD via `Spd3::is_spd`
    //// in debug builds (not in release — too expensive for the inner loop).
    //// Ref: `07-l5-l6-cascade-composition.md` § "L6 Pillar-8 temporal sandwich"
    //// Ref: B-Compose bundle for the horizontal reduction step
    //// Ref: `crate::hpc::pillar::temporal_sandwich` Pillar-8 probe
    let _ = (state, kernel);
    todo!("temporal_sandwich — L6 Pillar-8 sandwich via PR-X11 Spd3::sandwich")
}

// ════════════════════════════════════════════════════════════════════════════
// Internal helpers (stubs)
// ════════════════════════════════════════════════════════════════════════════

/// Outer product (rank-1 update) added to an `Spd3` accumulator.
///
/// Computes `acc += w * d⊗d` where `d⊗d` is the symmetric outer product:
/// `(d⊗d)[i,j] = d[i]*d[j]`. Used in the second-moment roll-up of
/// `compose_cascade`.
#[allow(dead_code)]
#[inline]
fn add_outer_product(acc: &mut Spd3, w: f32, d: [f32; 3]) -> &mut Spd3 {
    //// TODO(L5): implement acc += w * d⊗d for the 6 upper-triangle entries.
    //// a11 += w * d[0]*d[0]; a12 += w * d[0]*d[1]; etc.
    //// Ref: `07-l5-l6-cascade-composition.md` § "L5 moment-match second-moment"
    let _ = (w, d);
    todo!("add_outer_product — rank-1 SPD accumulator update")
}

// ════════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Gaussian::UNIT is const — verify the zero-overhead `const` path.
    #[test]
    fn gaussian_unit_is_const() {
        let g = Gaussian::UNIT;
        assert_eq!(g.mean, [0.0; 3]);
        assert_eq!(g.opacity, 1.0);
    }

    /// Gaussian::new round-trips through fields correctly.
    #[test]
    fn gaussian_new_roundtrips() {
        let g = Gaussian::new([1.0, 2.0, 3.0], 0.5, Spd3::I);
        assert_eq!(g.mean, [1.0, 2.0, 3.0]);
        assert_eq!(g.opacity, 0.5);
    }

    //// TODO(A2/L5): add scalar reference test for compose_cascade.
    //// Fixture: 16 children with known means and unit covariances; verify the
    //// moment-matched parent mean equals the opacity-weighted centroid and the
    //// parent covariance equals the scatter matrix. Done criterion 4 gate.
    //// Ref: `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "Done criteria" #4

    //// TODO(L6): add temporal_sandwich scalar reference test.
    //// For identity kernel (Spd3::I), output covariance must equal input
    //// covariance. For a contractive kernel, verify SPD preservation.
    //// Ref: `07-l5-l6-cascade-composition.md` § "L6 Pillar-8 temporal sandwich"
}
