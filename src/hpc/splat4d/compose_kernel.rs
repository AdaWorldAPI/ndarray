//! `splat4d::compose_kernel` — closure-swappable composition kernel bridge.
//!
//! # Design
//!
//! The `B-Compose` SIMD bundle is closure-swappable: the same outer loop
//! (cascade sort + horizontal reduction) drives two algebraically distinct
//! inner kernels:
//!
//! | Kernel | Algebra | Default? | Gate |
//! |---|---|---|---|
//! | `AlphaCompose` | `C_out = α·C_splat + (1−α)·C_accum` | ✓ | always |
//! | `NarsRevise` | NARS revision on `(freq, conf)` tuple | ✗ | `splat4d-nars-compose` |
//!
//! The `ComposeKernel` trait is a zero-cost abstraction: `const T_SATURATION_EPS`
//! lives on the trait so each kernel can tune its own convergence threshold;
//! the `compose` method is inlined by the monomorphised caller.
//!
//! # Feature-flag selection
//!
//! The kernel is selected at compile time via a const-generic `<C: ComposeKernel>`
//! on `frame_pipeline_with` — no `Box<dyn>`, no vtable, no branch in the hot
//! path. The `splat4d-nars-compose` feature flag only gates the *NarsRevise impl
//! being compiled in*; the `AlphaCompose` impl always compiles.
//!
//// TODO(compose-swap-W7): At W7's PR-X4 closure-swap sprint, flip the
//// default kernel in `pipeline::frame_pipeline` from AlphaCompose to NarsRevise
//// by changing the #[cfg] guard. Do NOT flip here (forbidden constraint #5).
//!
//! # NARS revision math
//!
//! ```text
//! revise(T1, T2) = (
//!   freq:  (f1·c1 + f2·c2) / (c1 + c2),
//!   conf:  (c1 + c2) / (c1 + c2 + k),    where k = 1 (NARS convention)
//! )
//! ```
//!
//! Precision note: `fast_exp_x16` carries 3% relative error. The G4 audit
//! (A5 worker) must confirm `worst_conf_diff < 1e-3` over 10K random revisions
//! before SG4 is considered valid. If the audit fails, a `precise_exp_x16`
//! follow-up PR against PR-X10 A6 must be opened before W7 closure swap.
//!
//! # Cross-references
//!
//! - `pipeline.rs` — `frame_pipeline_with<C: ComposeKernel>` consumer
//! - `revise.rs` — G3 NARS truth-revision kernel + G4 fast_exp_x16 audit
//! - `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "G3" and § "Gates" SG4
//! - `hpc::nars` — scalar `nars_revision` reference impl PR-X4 tests against

// ════════════════════════════════════════════════════════════════════════════
// TruthValue
// ════════════════════════════════════════════════════════════════════════════

/// A single NARS-style truth value used as the composition unit inside
/// the `B-Compose` bundle.
///
/// Both the graphics interpretation (alpha-compositing) and the cognitive
/// interpretation (NARS revision) are encoded as `(value, weight)` pairs:
///
/// | Interpretation | `value` | `weight` |
/// |---|---|---|
/// | Alpha-composite | pixel channel value ∈ [0, 1] | opacity α ∈ [0, 1] |
/// | NARS revision | frequency f ∈ [0, 1] | confidence c ∈ [0, 1) |
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct TruthValue {
    /// Primary scalar: frequency (NARS) or channel value (alpha).
    pub value: f32,
    /// Weight scalar: confidence (NARS) or opacity (alpha).
    pub weight: f32,
}

impl TruthValue {
    /// Construct a `TruthValue` from its components.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let tv = TruthValue::new(0.9, 0.8);  // freq=0.9, conf=0.8
    /// ```
    #[inline]
    pub const fn new(value: f32, weight: f32) -> Self {
        Self { value, weight }
    }

    /// Total ignorance: value=0.5, weight=0.0 (NARS convention).
    #[inline]
    pub const fn ignorance() -> Self {
        Self { value: 0.5, weight: 0.0 }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// ComposeKernel trait
// ════════════════════════════════════════════════════════════════════════════

/// Closure-swappable composition kernel for the `B-Compose` SIMD bundle.
///
/// Implementors provide:
/// - `compose`: combine an incoming splat truth-value `t1` with the
///   running accumulator `t2` to produce the updated accumulator.
/// - `T_SATURATION_EPS`: early-exit threshold — when the accumulator's
///   weight falls below this value, the cascade tile can stop compositing.
///
/// # Contract
///
/// - `compose` must be pure (no `&mut self`, per the data-flow invariant
///   in `.claude/rules/data-flow.md`).
/// - `T_SATURATION_EPS > 0.0` — zero would disable the early-exit and
///   regress SG2 p95 latency.
///
/// # Example (alpha composition algebra)
///
/// ```ignore
/// use ndarray::hpc::splat4d::compose_kernel::{AlphaCompose, ComposeKernel, TruthValue};
/// let splat = TruthValue::new(0.8, 0.6);
/// let accum = TruthValue::new(0.3, 0.0);
/// let out = AlphaCompose::compose(splat, accum);
/// ```
pub trait ComposeKernel {
    /// Combine the incoming splat truth-value `t1` with the running
    /// accumulator `t2` and return the updated accumulator.
    ///
    /// Both algebras have the same call signature so the outer sort-and-
    /// reduce loop is polymorphic over `C: ComposeKernel` with zero overhead.
    fn compose(t1: TruthValue, t2: TruthValue) -> TruthValue;

    /// Early-exit threshold.  When the accumulator `weight` drops below
    /// this value, subsequent splats contribute negligibly and the tile
    /// loop may stop.
    ///
    /// `T_SATURATION_EPS` is parameterised here (rather than being a
    /// module-level constant) so each kernel can tune its own convergence
    /// bound — graphics alpha can tolerate 1e-4; NARS confidence may
    /// require a tighter floor until the G4 fast_exp_x16 audit passes.
    const T_SATURATION_EPS: f32;
}

// ════════════════════════════════════════════════════════════════════════════
// AlphaCompose — default graphics kernel (always compiled)
// ════════════════════════════════════════════════════════════════════════════

/// Alpha-compositing kernel (default, always enabled).
///
/// Implements the standard Porter-Duff over-operator:
///
/// ```text
/// C_out = α_splat · C_splat + (1 − α_splat) · C_accum
/// α_out = α_splat + (1 − α_splat) · α_accum
/// ```
///
/// Encoded as `TruthValue { value: C, weight: α }`.
/// Early-exit when the remaining transmittance `T = (1 − α_accum)` drops
/// below `T_SATURATION_EPS` — same threshold as the v1 rasterizer in
/// `splat3d::raster::T_SATURATION_EPS`.
///
/// # Cross-reference
///
/// `crate::hpc::splat3d::raster::T_SATURATION_EPS` (= 1e-4) — the v1
/// rasterizer uses the same convergence floor; AlphaCompose mirrors it
/// for back-compat during the v1→v2 side-by-side period.
pub struct AlphaCompose;

impl ComposeKernel for AlphaCompose {
    /// Porter-Duff over: `C_out = α·C_splat + (1−α)·C_accum`.
    #[inline]
    fn compose(t1: TruthValue, t2: TruthValue) -> TruthValue {
        todo!(
            "A6/A5: implement AlphaCompose::compose — \
             C_out = t1.weight * t1.value + (1 - t1.weight) * t2.value; \
             w_out = t1.weight + (1 - t1.weight) * t2.weight"
        )
    }

    /// Matches `splat3d::raster::T_SATURATION_EPS` for back-compat.
    const T_SATURATION_EPS: f32 = 1e-4;
}

// ════════════════════════════════════════════════════════════════════════════
// NarsRevise — cognitive kernel (gated on feature flag)
// ════════════════════════════════════════════════════════════════════════════

/// NARS truth-revision kernel.
///
/// Gated on `#[cfg(feature = "splat4d-nars-compose")]` — **default OFF**
/// per forbidden constraint #5 in the PR-X4 pre-sprint prompt.
///
/// # Revision formula
///
/// ```text
/// revise(T1, T2) = (
///   freq:  (f1·c1 + f2·c2) / (c1 + c2),
///   conf:  (c1 + c2) / (c1 + c2 + k),    k = 1 (NARS convention)
/// )
/// ```
///
/// Where `TruthValue { value: freq, weight: conf }`.
///
/// # Precision gate (SG4 dependency)
///
/// `T_SATURATION_EPS` is set to `1e-3` (tighter than alpha's `1e-4`) pending
/// the G4 `fast_exp_x16` precision audit.  The A5 worker must confirm
/// `worst_conf_diff < 1e-3`; if the audit fails, this const may need to
/// be relaxed and `precise_exp_x16` shipped before W7 closure swap.
///
//// SG4-marker: NarsRevise::compose is the SG4 gate kernel.
//// A measurement of SG4 latency vs SG1-SG3 under AlphaCompose is required
//// before the W7 closure-swap sprint proceeds. If SG4 fails (same envelope
//// not achieved), the NARS B-Compose path must be re-staged — see
//// hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md § "Gates" row SG4.
#[cfg(feature = "splat4d-nars-compose")]
pub struct NarsRevise;

#[cfg(feature = "splat4d-nars-compose")]
impl ComposeKernel for NarsRevise {
    /// NARS revision: `(f1·c1 + f2·c2)/(c1+c2)` ; `(c1+c2)/(c1+c2+1)`.
    #[inline]
    fn compose(t1: TruthValue, t2: TruthValue) -> TruthValue {
        todo!(
            "A5: implement NarsRevise::compose — \
             freq_out = (t1.value*t1.weight + t2.value*t2.weight) / (t1.weight + t2.weight); \
             conf_out = (t1.weight + t2.weight) / (t1.weight + t2.weight + 1.0); \
             handle zero-weight denominator gracefully"
        )
    }

    /// Tighter than alpha's 1e-4 — pending G4 fast_exp_x16 audit verdict.
    //// TODO(G4-audit): relax to 1e-4 if the A5 audit confirms
    //// worst_conf_diff < 1e-3 for the full NARS confidence range.
    const T_SATURATION_EPS: f32 = 1e-3;
}
