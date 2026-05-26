//! `oracle` (group D) — parity harness: load a reference, render via `splat3d`,
//! diff (SSIM / PSNR).
//!
//! # Purpose
//!
//! This is a **measurement instrument**, not an assertion engine.  It quantifies
//! how close our `splat3d` render is to a reference image — it never asserts
//! "must be above N dB" at runtime.  Threshold gates live in [`crate::fixtures`].
//!
//! # Reference pipeline (first reference = Inria `.ply`)
//!
//! ```text
//! Inria .ply file
//!     │  read_ply::<BufReader<File>>()          ← splat3d::ply::read_ply
//!     ▼
//! GaussianBatch (SoA)                           ← splat3d::gaussian::GaussianBatch
//!     │  (project → tile → rasterize)           ← splat3d::raster::rasterize_frame
//!     ▼
//! framebuffer: Vec<f32>  (interleaved RGB, 3·W·H)
//!     │  ParityMetrics::compute(&ref_fb, &our_fb)
//!     ▼
//! ParityMetrics { ssim, psnr }
//! ```
//!
//! # Named risk: sort-order divergence
//!
//! `splat3d::tile::TileBinning::from_projected` sorts instances by packed u64
//! key `(tile_id << 32) | depth_bits` using `sort_unstable_by_key`, giving
//! **front-to-back** depth order within each tile.  The rasterizer in
//! `raster::rasterize_tile` then performs a **front-to-back alpha-blend** with
//! early-out at `T < T_SATURATION_EPS = 1e-4`.
//!
//! The reference renderers (brush, Inria `diff-gaussian-rasterization`) may
//! sort **back-to-front** (painter's-algorithm) instead.  When two or more
//! Gaussians have nearly equal depth, floating-point accumulation order is
//! reversed between the two approaches, producing per-pixel colour differences
//! of order `O(alpha²)`.  For high-opacity scenes this can be several dB of
//! PSNR.  Any SSIM/PSNR delta unexplained by numerical noise should be
//! investigated for sort-order as the primary suspect.
//!
//! # Reference renderer — same binary, NO FFI
//!
//! There is **no Cesium/cesium-native FFI** and there never will be.  The
//! oracle's "reference" is one of the existing in-scope 3DGS renderers wired
//! as an ordinary workspace dependency in the **same binary** (A-to-Z, no
//! `extern "C"`, no browser/Node, no C++ ABI):
//!
//! - **`brush`** (`AdaWorldAPI/brush`, Rust + `wgpu`) — runnable in-binary on
//!   CPU/any-GPU with no CUDA toolchain; the default reference path.
//! - **Inria `gaussian-splatting`** + **`diff-gaussian-rasterization`**
//!   (`AdaWorldAPI/*`, CUDA) — the ground-truth rasterizer when a CUDA device
//!   is present; same binary, linked as a normal Rust crate, not via FFI.
//!
//! All three speak our CAM SoA (`splat3d::gaussian::GaussianBatch`) at the
//! boundary, so the oracle compares like-for-like: load → render(ours) vs
//! render(reference) → SSIM/PSNR.  None of these is a foreign binary; the
//! wiring is `[dependencies]` + a direct call, gated by `cfg`/feature, once
//! `ndarray` is re-enabled in `Cargo.toml`.
//!
//! # Implementation status
//!
//! **Commented scaffold only.**  `ndarray` is commented out in `Cargo.toml`;
//! the stub types (`ParityMetrics`, `OracleError`) are `std`-only and compile
//! clean.  All impl blocks that touch `splat3d` symbols are commented out.
//!
//! # Source grounding
//!
//! Symbols cited from the local codebase (confirmed by reading before writing):
//! - `read_ply` — `src/hpc/splat3d/ply.rs` line 110: `pub fn read_ply<R: Read>(reader: R) -> Result<GaussianBatch, PlyError>`
//! - `rasterize_frame` — `src/hpc/splat3d/raster.rs` line 213: `pub fn rasterize_frame(binning: &TileBinning, projected: &ProjectedBatch, framebuffer: &mut [f32], width: u32, height: u32, background: [f32; 3])`
//! - `TileBinning::from_projected` — `src/hpc/splat3d/tile.rs` line 105: sort by `(tile_id << 32) | depth_bits`, front-to-back
//! - `GaussianBatch` — `src/hpc/splat3d/gaussian.rs`: SoA fields `mean_x`, `mean_y`, `mean_z`, `len`
//! - Framebuffer layout — `raster.rs` doc: interleaved RGB `[R0,G0,B0,R1,…]`, len = `3·width·height`

// ─────────────────────────────────────────────────────────────────────────────
// Stub types (live — std-only, no ndarray dep)
// ─────────────────────────────────────────────────────────────────────────────

/// Errors the oracle can return.
#[derive(Debug)]
pub enum OracleError {
    /// I/O error reading a reference file.
    Io(std::io::Error),
    /// PLY parse failed.  Payload is a human-readable description.
    PlyParse(String),
    /// Framebuffer dimensions mismatch between reference and candidate.
    DimensionMismatch { ref_len: usize, candidate_len: usize },
    /// The render pipeline returned an empty framebuffer (width or height = 0).
    EmptyFramebuffer,
    // DEFERRED: ReferenceRender(String) — reserved for failures from the
    // in-binary reference renderer (brush / Inria), wired as a workspace
    // dependency once `ndarray` is re-enabled in Cargo.toml. NOT an FFI path.
    // ReferenceRender(String),
}

impl std::fmt::Display for OracleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OracleError::Io(e) => write!(f, "oracle I/O error: {e}"),
            OracleError::PlyParse(msg) => write!(f, "oracle PLY parse error: {msg}"),
            OracleError::DimensionMismatch { ref_len, candidate_len } => {
                write!(f, "oracle dimension mismatch: ref={ref_len} candidate={candidate_len}")
            }
            OracleError::EmptyFramebuffer => write!(f, "oracle: empty framebuffer (width or height = 0)"),
        }
    }
}

impl From<std::io::Error> for OracleError {
    fn from(e: std::io::Error) -> Self {
        OracleError::Io(e)
    }
}

/// SSIM / PSNR parity metrics between a reference and a candidate framebuffer.
///
/// Both metrics are computed over the full `3·W·H` interleaved RGB buffer
/// (not per-channel).  Channel weighting and luminance-only variants are
/// `// DEFERRED:` until the fixture suite validates a specific convention.
///
/// # SSIM
///
/// Structural Similarity Index Measure in `[−1, 1]`; 1.0 = identical.
/// A coarse approximation (global mean/variance, not 11×11 Gaussian window)
/// is acceptable for the parity gate; the fixture thresholds in
/// [`crate::fixtures`] are set conservatively to hide the approximation error.
///
/// # PSNR
///
/// Peak Signal-to-Noise Ratio in dB: `10·log10(MAX² / MSE)` where `MAX=1.0`
/// (framebuffer in `[0,1]`).  `f32::INFINITY` when the buffers are identical
/// (MSE = 0).  Practically ≥ 30 dB = visually acceptable; ≥ 40 dB = excellent.
///
/// # Sort-order risk
///
/// See module-level doc.  Any SSIM < 0.98 or PSNR < 35 dB for a scene with
/// depth-ordered overlapping Gaussians should be attributed to sort-order
/// divergence before tuning thresholds.
#[derive(Debug, Clone, PartialEq)]
pub struct ParityMetrics {
    /// SSIM in `[−1, 1]`.  1.0 = identical.
    pub ssim: f32,
    /// PSNR in dB.  `f32::INFINITY` when buffers are identical.
    pub psnr: f32,
    /// Mean squared error (raw, before log).
    pub mse: f32,
    /// Number of f32 samples compared (`3 · width · height`).
    pub sample_count: usize,
}

impl ParityMetrics {
    /// Compute SSIM and PSNR between two equal-length interleaved RGB
    /// framebuffers.  Both must have the same `len = 3 · width · height`.
    ///
    /// Returns `Err(OracleError::DimensionMismatch)` if lengths differ, or
    /// `Err(OracleError::EmptyFramebuffer)` if either is empty.
    ///
    /// # Algorithm (coarse approximation — acceptable for parity gating)
    ///
    /// ```text
    /// mu_r  = mean(ref),   mu_c = mean(candidate)
    /// var_r = var(ref),    var_c = var(candidate)
    /// cov   = covariance(ref, candidate)
    /// C1    = (0.01)²,     C2 = (0.03)²    [standard luminance/contrast constants]
    /// SSIM  = (2·mu_r·mu_c + C1)(2·cov + C2)
    ///       / ((mu_r² + mu_c² + C1)(var_r + var_c + C2))
    /// MSE   = mean((ref[i] - cand[i])²)
    /// PSNR  = 10·log10(1.0 / MSE)   (MAX=1.0)
    /// ```
    pub fn compute(reference: &[f32], candidate: &[f32]) -> Result<Self, OracleError> {
        let n = reference.len();
        if n == 0 || candidate.is_empty() {
            return Err(OracleError::EmptyFramebuffer);
        }
        if n != candidate.len() {
            return Err(OracleError::DimensionMismatch {
                ref_len: n,
                candidate_len: candidate.len(),
            });
        }

        let nf = n as f64;

        // Means
        let mu_r: f64 = reference.iter().map(|&v| v as f64).sum::<f64>() / nf;
        let mu_c: f64 = candidate.iter().map(|&v| v as f64).sum::<f64>() / nf;

        // Variances and covariance (single-pass after mean)
        let mut var_r = 0.0f64;
        let mut var_c = 0.0f64;
        let mut cov = 0.0f64;
        let mut mse = 0.0f64;
        for (&r, &c) in reference.iter().zip(candidate.iter()) {
            let dr = r as f64 - mu_r;
            let dc = c as f64 - mu_c;
            var_r += dr * dr;
            var_c += dc * dc;
            cov += dr * dc;
            let diff = r as f64 - c as f64;
            mse += diff * diff;
        }
        var_r /= nf;
        var_c /= nf;
        cov /= nf;
        mse /= nf;

        // SSIM constants (standard Wang et al. 2004 values for [0,1] range)
        const C1: f64 = 0.01 * 0.01;
        const C2: f64 = 0.03 * 0.03;
        let num = (2.0 * mu_r * mu_c + C1) * (2.0 * cov + C2);
        let den = (mu_r * mu_r + mu_c * mu_c + C1) * (var_r + var_c + C2);
        let ssim = if den.abs() < 1e-30 { 1.0f32 } else { (num / den) as f32 };

        let psnr = if mse < 1e-30 {
            f32::INFINITY
        } else {
            (10.0 * (1.0 / mse).log10()) as f32
        };

        Ok(ParityMetrics {
            ssim,
            psnr,
            mse: mse as f32,
            sample_count: n,
        })
    }

    /// Returns `true` if this result passes the given SSIM and PSNR thresholds.
    ///
    /// The oracle measures parity — it does not assert it.  Callers in
    /// [`crate::fixtures`] decide whether to gate on these results.
    pub fn passes(&self, min_ssim: f32, min_psnr_db: f32) -> bool {
        self.ssim >= min_ssim && (self.psnr >= min_psnr_db || self.psnr.is_infinite())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Commented scaffold — live impl blocked on ndarray dep
// ─────────────────────────────────────────────────────────────────────────────

// DEFERRED: wire the full pipeline once `ndarray = { workspace = true }` is
// re-enabled and the module passes Opus + CodeRabbit review.
//
// use std::fs::File;
// use std::io::BufReader;
//
// // ── splat3d symbols (confirmed from local source, not fabricated) ────────────
// // read_ply     — ply.rs:110  pub fn read_ply<R: Read>(reader: R) -> Result<GaussianBatch, PlyError>
// // rasterize_frame — raster.rs:213  pub fn rasterize_frame(binning: &TileBinning, projected: &ProjectedBatch, framebuffer: &mut [f32], width: u32, height: u32, background: [f32; 3])
// // TileBinning::from_projected — tile.rs:105  sorts front-to-back by (tile_id<<32)|depth_bits
// // GaussianBatch — gaussian.rs: SoA with len, mean_x/y/z, scale_x/y/z, quat_w/x/y/z, opacity, sh
// // Framebuffer: interleaved RGB [R0,G0,B0,R1,…], len = 3·width·height (raster.rs doc)
// //
// use ndarray_hpc::hpc::splat3d::ply::{read_ply, PlyError};
// use ndarray_hpc::hpc::splat3d::gaussian::GaussianBatch;
// use ndarray_hpc::hpc::splat3d::raster::rasterize_frame;
// use ndarray_hpc::hpc::splat3d::tile::TileBinning;
// use ndarray_hpc::hpc::splat3d::project::{Camera, ProjectedBatch, project_batch};
//
// impl From<PlyError> for OracleError {
//     fn from(e: PlyError) -> Self {
//         OracleError::PlyParse(format!("{e:?}"))
//     }
// }
//
// /// Load an Inria `.ply` file and return a `GaussianBatch`.
// ///
// /// Entry point for the first parity reference.  Applies the canonical
// /// activation transforms inline (sigmoid(opacity), exp(scale), normalize(quat))
// /// as documented in `ply.rs`.
// ///
// /// # Sort-order risk (see module doc)
// ///
// /// The batch returned here is unsorted.  `TileBinning::from_projected`
// /// will sort instances front-to-back via the packed u64 key.  If the
// /// reference render was produced with back-to-front order, the PSNR
// /// comparison will reflect the sort-order delta.
// pub fn load_ply_batch(path: &std::path::Path) -> Result<GaussianBatch, OracleError> {
//     let file = File::open(path)?;
//     let reader = BufReader::new(file);
//     let batch = read_ply(reader)?;  // ply.rs::read_ply
//     Ok(batch)
// }
//
// /// Render a `GaussianBatch` via the `splat3d` pipeline and return the
// /// framebuffer as `Vec<f32>` (interleaved RGB, len = `3·width·height`).
// ///
// /// # Named risk: sort-order
// ///
// /// `rasterize_frame` (raster.rs:213) consumes instances in the order
// /// produced by `TileBinning::from_projected` (tile.rs:105), which is
// /// **front-to-back** (ascending depth).  If the reference was rendered
// /// back-to-front, every pixel with overlapping Gaussians will differ.
// ///
// /// # UNVERIFIED: camera parameters
// ///
// /// The `Camera` struct (project.rs) takes `(width, height, fovy_rad,
// /// near, far)`.  The identity camera `Camera::identity_at_origin` is used
// /// here as a placeholder; real parity comparisons need matched intrinsics.
// // UNVERIFIED: Camera::identity_at_origin exists in project.rs — confirm
// // when wiring this function.
// pub fn render_batch(
//     batch: &GaussianBatch,
//     camera: &Camera,
//     background: [f32; 3],
// ) -> Result<Vec<f32>, OracleError> {
//     let w = camera.width;
//     let h = camera.height;
//     if w == 0 || h == 0 {
//         return Err(OracleError::EmptyFramebuffer);
//     }
//     let mut projected = ProjectedBatch::with_capacity(batch.len);
//     project_batch(batch, camera, &mut projected);            // UNVERIFIED: project_batch fn name
//     let binning = TileBinning::from_projected(&projected, camera); // tile.rs:105
//     let mut framebuffer = vec![0.0f32; 3 * w as usize * h as usize];
//     rasterize_frame(&binning, &projected, &mut framebuffer, w, h, background); // raster.rs:213
//     Ok(framebuffer)
// }
//
// /// Full oracle run: load `.ply`, render, compare to a reference framebuffer.
// ///
// /// Returns `ParityMetrics`; caller decides whether to gate on the result.
// pub fn run_ply_oracle(
//     ply_path: &std::path::Path,
//     reference_fb: &[f32],
//     camera: &Camera,
//     background: [f32; 3],
// ) -> Result<ParityMetrics, OracleError> {
//     let batch = load_ply_batch(ply_path)?;
//     let candidate_fb = render_batch(&batch, camera, background)?;
//     ParityMetrics::compute(reference_fb, &candidate_fb)
// }

// ─────────────────────────────────────────────────────────────────────────────
// DEFERRED: in-binary reference renderer (brush / Inria) — NO FFI
// ─────────────────────────────────────────────────────────────────────────────

// The cross-renderer reference is an existing in-scope 3DGS renderer wired as
// an ordinary workspace dependency in the SAME binary — no `extern "C"`, no
// `#[link]`, no browser/Node, no foreign ABI. It consumes the same CAM SoA
// `GaussianBatch` we feed `splat3d`, renders to the same interleaved-RGB
// framebuffer layout, and we diff the two with `ParityMetrics::compute`.
//
// DEFERRED until `ndarray` (and the chosen reference crate) are re-enabled in
// Cargo.toml and the module passes Opus + CodeRabbit review:
//
// // Default reference: brush (Rust + wgpu, runnable without CUDA).
// // UNVERIFIED: brush's public render entry point + camera/SoA adapter shape;
// // confirm against AdaWorldAPI/brush before wiring.
// fn render_reference_brush(
//     batch: &GaussianBatch,
//     camera: &Camera,
//     background: [f32; 3],
// ) -> Result<Vec<f32>, OracleError> {
//     // brush::render(...) → Vec<f32> interleaved RGB, len = 3·W·H
//     todo!("wire brush as same-binary workspace dep")
// }
//
// // Ground truth (when a CUDA device is present): Inria gaussian-splatting /
// // diff-gaussian-rasterization, linked as a normal Rust crate (NOT FFI),
// // gated behind a `cuda` cfg/feature.
// // UNVERIFIED: the Rust-facing entry point exposed by the Inria crates.
// #[cfg(feature = "cuda")]
// fn render_reference_inria(
//     batch: &GaussianBatch,
//     camera: &Camera,
//     background: [f32; 3],
// ) -> Result<Vec<f32>, OracleError> {
//     todo!("wire diff-gaussian-rasterization as same-binary workspace dep")
// }

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests (live — std-only)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_fb(values: &[f32]) -> Vec<f32> {
        values.to_vec()
    }

    #[test]
    fn parity_metrics_identical_buffers_gives_inf_psnr_and_ssim_one() {
        let fb = make_fb(&[0.5f32, 0.3, 0.8, 0.1, 0.9, 0.2]);
        let m = ParityMetrics::compute(&fb, &fb).unwrap();
        assert!(m.psnr.is_infinite(), "identical buffers must give infinite PSNR");
        assert!((m.ssim - 1.0).abs() < 1e-5, "identical buffers must give SSIM = 1.0, got {}", m.ssim);
        assert!(m.mse < 1e-10, "identical buffers MSE must be 0, got {}", m.mse);
    }

    #[test]
    fn parity_metrics_zero_vs_one_gives_finite_psnr() {
        let ref_fb = vec![1.0f32; 48];
        let cand_fb = vec![0.0f32; 48];
        let m = ParityMetrics::compute(&ref_fb, &cand_fb).unwrap();
        assert!(m.psnr.is_finite(), "PSNR must be finite for non-identical buffers, got {}", m.psnr);
        // MSE = 1.0, PSNR = 10*log10(1/1) = 0 dB
        assert!((m.psnr - 0.0).abs() < 0.01, "PSNR for all-zero vs all-one should be 0 dB, got {}", m.psnr);
        assert!((m.mse - 1.0).abs() < 1e-5, "MSE must be 1.0, got {}", m.mse);
    }

    #[test]
    fn parity_metrics_passes_respects_thresholds() {
        let ref_fb = vec![0.5f32; 12];
        let cand_fb = ref_fb.clone();
        let m = ParityMetrics::compute(&ref_fb, &cand_fb).unwrap();
        // Identical → passes any threshold
        assert!(m.passes(0.99, 40.0), "identical buffers must pass any threshold");
        // Fabricate a bad metrics to test failure path
        let bad = ParityMetrics {
            ssim: 0.5,
            psnr: 20.0,
            mse: 0.01,
            sample_count: 12,
        };
        assert!(!bad.passes(0.99, 40.0), "bad metrics must fail tight threshold");
        assert!(bad.passes(0.4, 15.0), "bad metrics must pass loose threshold");
    }

    #[test]
    fn parity_metrics_dimension_mismatch_returns_error() {
        let a = vec![0.0f32; 12];
        let b = vec![0.0f32; 9];
        match ParityMetrics::compute(&a, &b) {
            Err(OracleError::DimensionMismatch {
                ref_len: 12,
                candidate_len: 9,
            }) => {}
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn parity_metrics_empty_returns_error() {
        match ParityMetrics::compute(&[], &[]) {
            Err(OracleError::EmptyFramebuffer) => {}
            other => panic!("expected EmptyFramebuffer, got {other:?}"),
        }
    }

    #[test]
    fn oracle_error_display_is_human_readable() {
        let e = OracleError::DimensionMismatch {
            ref_len: 100,
            candidate_len: 99,
        };
        let s = format!("{e}");
        assert!(s.contains("100") && s.contains("99"), "display must include lens: {s}");
    }
}
