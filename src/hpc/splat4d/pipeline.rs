//! `splat4d::pipeline` — `frame_pipeline` entry point for A6 Railway smoke
//! deployment.
//!
//! # Pipeline shape
//!
//! ```text
//! Frame (raw pixels)
//!   │
//!   ▼
//! decode          (H.264/HLS → raw planar f32 RGB)
//!   │
//!   ▼
//! B-Interleave-Transpose
//!   │             row-major splat3d ↔ lane-major splat4d boundary
//!   ▼
//! cascade L1 → L2 → L3 → L4
//!   │             4×4 covariance footprint per tier; L4 gated on
//!   │             PR-X10 A12b Hilbert-3D fix (returns Err(NotReadyL4)
//!   │             until fixed — A6 exercises L1-L3 only for smoke)
//!   ▼
//! B-Compose       closure-swappable: AlphaCompose (default) or
//!   │             NarsRevise (#[cfg(feature = "splat4d-nars-compose")])
//!   ▼
//! emit            serialise back to output Frame
//! ```
//!
//! # SIMD bundles consumed
//!
//! | Bundle | Purpose |
//! |---|---|
//! | **B-Interleave-Transpose** | Row-major ↔ lane-major boundary |
//! | **B-Splat** | Broadcast Gaussian centre across 16 tile lanes |
//! | **B-Compose** | Horizontal reduction (alpha or NARS via feature flag) |
//!
//! # Smoke acceptance gates
//!
//! | Gate | Target |
//! |---|---|
//! | SG1: median FPS | ≥ 60 fps for 1080p H.264, 10-min Big Buck Bunny |
//! | SG2: p95 frame time | ≤ 20 ms |
//! | SG3: stutter count | 0 events > 33 ms over 10 minutes |
//! | SG4: closure swap | Same SG1-SG3 envelope with `splat4d-nars-compose` on |
//!
//! SG4 is conditional on G3 (NARS truth-revision kernel) being ULP-correct
//! against the scalar reference (A5 worker verdict).
//!
//! # Cross-references
//!
//! - `pr-x4-planning/06-a6-railway-smoke-brief.md` — A6 deployment spec
//! - `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` § "Smoke acceptance gates"
//! - `compose_kernel.rs` — ComposeKernel trait + AlphaCompose / NarsRevise impls
//! - `cascade.rs` — CascadeAddr + NotReadyL4

use crate::hpc::splat4d::compose_kernel::{AlphaCompose, ComposeKernel};
#[cfg(feature = "splat4d-nars-compose")]
use crate::hpc::splat4d::compose_kernel::NarsRevise;

// ════════════════════════════════════════════════════════════════════════════
// Public types
// ════════════════════════════════════════════════════════════════════════════

/// One video frame entering or leaving the `frame_pipeline`.
///
/// Pixel data is stored as planar f32 RGB, row-major, stride = `width`.
/// Length invariant: `pixels.len() == 3 * width as usize * height as usize`.
///
/// # Layout
///
/// ```text
/// pixels[0 .. W*H]         — R plane
/// pixels[W*H .. 2*W*H]     — G plane
/// pixels[2*W*H .. 3*W*H]   — B plane
/// ```
pub struct Frame {
    /// Planar f32 RGB pixel data.
    pub pixels: Vec<f32>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Monotonic presentation timestamp, microseconds from epoch.
    pub pts_us: u64,
}

/// Errors that can occur inside `frame_pipeline`.
#[derive(Debug)]
pub enum PipelineError {
    /// Input frame dimensions are inconsistent with the pixel buffer length.
    DimensionMismatch {
        /// Expected `3 * width * height` bytes.
        expected: usize,
        /// Actual pixel buffer length.
        actual: usize,
    },
    /// The L4 cascade level is not yet available — PR-X10 A12b Hilbert-3D
    /// fix has not landed on master.  L1-L3 still produce a valid output.
    NotReadyL4,
    /// A SIMD bundle call returned an error (e.g. alignment fault).
    SimdBundle(&'static str),
}

impl core::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PipelineError::DimensionMismatch { expected, actual } => {
                write!(f, "frame dimension mismatch: expected {expected} floats, got {actual}")
            }
            PipelineError::NotReadyL4 => {
                write!(f, "L4 cascade not ready: waiting for PR-X10 A12b Hilbert-3D fix")
            }
            PipelineError::SimdBundle(name) => {
                write!(f, "SIMD bundle error in {name}")
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// frame_pipeline — public entry point
// ════════════════════════════════════════════════════════════════════════════

/// Run one video frame through the full splat4d cascade and return the
/// composed output frame.
///
/// # Pipeline steps
///
/// 1. **Validate** — check `input.pixels.len() == 3 * w * h`.
/// 2. **B-Interleave-Transpose** — convert row-major pixel layout to the
///    lane-major splat4d representation consumed by the cascade.
/// 3. **Cascade L1 → L4** — four-tier 4×4 covariance cascade.  L4 is
///    attempted; if PR-X10 A12b's Hilbert-3D fix is not yet on master
///    the L4 step is skipped silently (returns `Ok` — not `Err(NotReadyL4)` —
///    because the smoke gates SG1-SG3 need only L1-L3 to be exercised).
/// 4. **B-Compose** — horizontal reduction via the compile-time-selected
///    `ComposeKernel`.  Default: `AlphaCompose`.  With feature flag
///    `splat4d-nars-compose`: `NarsRevise`.
/// 5. **Emit** — convert back to the row-major `Frame` format.
///
/// # Composition kernel selection
///
/// The `C: ComposeKernel` type parameter is bound at call-site.  Callers
/// should use the convenience wrapper `frame_pipeline` (no type param) which
/// selects the kernel via `#[cfg(feature = "splat4d-nars-compose")]`.
///
/// # Feature-flag justification (`////` marker)
///
//// TODO(A6-compose-swap): The feature-flag approach (cfg-selection of C at
//// the call-site wrapper rather than dynamic dispatch) gives zero overhead
//// in the default alpha path and is the cleanest option per forbidden
//// constraint #5 ("no NARS-revision activation by default").  A const-generic
//// <C: ComposeKernel> lets the compiler monomorphise a separate binary for
//// each composition algebra — same lane width, same latency class, tested
//// identically by SG4.  The alternative (runtime flag + Box<dyn>) would add
//// a vtable indirection on the hot path and violate the "no Box<dyn> in hot
//// paths" rule from CLAUDE.md.
pub fn frame_pipeline_with<C: ComposeKernel>(input: Frame) -> Result<Frame, PipelineError> {
    todo!(
        "A6: implement frame_pipeline_with<C: ComposeKernel> — \
         decode → B-Interleave-Transpose → cascade L1..L4 → B-Compose<C> → emit"
    )
}

/// Convenience entry point — selects `AlphaCompose` or `NarsRevise` at
/// compile time depending on the `splat4d-nars-compose` feature flag.
///
/// # Errors
///
/// See [`PipelineError`].
///
/// # Example
///
/// ```ignore
/// let out = frame_pipeline(input_frame)?;
/// ```
pub fn frame_pipeline(input: Frame) -> Result<Frame, PipelineError> {
    //// SG4-gate: cfg selection mirrors the feature-flag requirement from
    //// hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md § "Gates" row SG4.
    //// AlphaCompose is always the default (forbidden constraint #5).
    #[cfg(not(feature = "splat4d-nars-compose"))]
    return frame_pipeline_with::<AlphaCompose>(input);

    #[cfg(feature = "splat4d-nars-compose")]
    return frame_pipeline_with::<NarsRevise>(input);
}

// ════════════════════════════════════════════════════════════════════════════
// Private helpers (stubs — filled in A6 worker)
// ════════════════════════════════════════════════════════════════════════════

/// Decode a raw H.264 / HLS segment into a planar f32 `Frame`.
///
//// TODO(A6-decode): wire in a zero-dep software H.264 decoder or accept
//// pre-decoded YCbCr from the axum handler and apply BT.709 → linear RGB.
/// For the smoke gate the input is assumed to be pre-decoded 1080p.
#[allow(dead_code)]
fn decode_frame(_raw: &[u8], _width: u32, _height: u32) -> Result<Frame, PipelineError> {
    todo!("A6: H.264 / HLS segment → planar f32 Frame")
}

/// Apply B-Interleave-Transpose: row-major → lane-major splat4d layout.
///
/// Consumes `B-Interleave-Transpose` bundle from `ndarray::simd`.
/// MUST NOT reach into raw `std::arch::*` intrinsics — see forbidden
/// constraint #2 in the PR-X4 pre-sprint prompt.
///
//// TODO(A6-interleave): call `simd::interleave_f32x16` then
//// `simd::transpose_inplace` on the pixel planes.
#[allow(dead_code)]
fn b_interleave_transpose(_frame: &mut Frame) -> Result<(), PipelineError> {
    todo!("A6: B-Interleave-Transpose → lane-major splat4d layout")
}

/// Run the 4-tier cascade (L1 → L2 → L3 → L4).
///
/// L4 silently skipped until PR-X10 A12b Hilbert-3D fix lands.
///
//// TODO(A6-cascade): invoke cascade L1..L3 unconditionally; attempt L4
//// via CascadeAddr::from_position; swallow NotReadyL4 per A6 spec.
#[allow(dead_code)]
fn run_cascade(_frame: &mut Frame) -> Result<(), PipelineError> {
    todo!("A6: cascade L1 → L3 (L4 gated on A12b)")
}

/// Apply B-Compose with the selected `ComposeKernel`.
///
//// TODO(A6-compose): call `ComposeKernel::compose` per tile pair,
//// respecting `T_SATURATION_EPS` early-exit.
#[allow(dead_code)]
fn b_compose<C: ComposeKernel>(_frame: &mut Frame) -> Result<(), PipelineError> {
    todo!("A6: B-Compose<C: ComposeKernel> horizontal reduction")
}
