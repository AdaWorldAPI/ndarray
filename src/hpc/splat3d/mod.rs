//! CPU-SIMD 3D Gaussian Splatting forward renderer (Kerbl et al. 2023).
//!
//! # Mission
//!
//! Render anisotropic 3D gaussians to a 2D image plane via the
//! Elliptical-Weighted-Average (EWA) splatting pipeline of Zwicker 2001 /
//! Kerbl 2023. All on the CPU, all in SIMD, no GPU, no wgpu, no new
//! top-level dependencies. Target: ≥30 fps at 1080p with 500K gaussians
//! on an 8-core AVX-512 box; NEON and AVX2 paths as graceful fallbacks.
//!
//! # The pipeline (forward pass)
//!
//! ```text
//!  GaussianBatch                Camera
//!  (μ, scale, quat, opacity,    (V, K, near/far,
//!   SH coefficients)             image dims)
//!         │                          │
//!         └───────────┬──────────────┘
//!                     ▼
//!              project_batch  ── J·W·Σ·Wᵀ·Jᵀ (EWA-sandwich, Pillar 7)
//!                     │           depth + 2D conic + 3σ radius + SH→RGB
//!                     ▼
//!              ProjectedBatch (SoA)
//!                     │
//!                     ▼
//!              TileBinning   ── 16×16 tile grid, AABB intersection,
//!                     │         radix-sort (tile_id, depth)
//!                     ▼
//!              raster_frame  ── per-tile alpha-blend front-to-back,
//!                     │         F32x16-wide pixels per inner loop
//!                     ▼
//!              framebuffer: Vec<f32>  (RGB, length = 3 · W · H)
//! ```
//!
//! # Architectural invariants — DO NOT VIOLATE
//!
//! 1. **Zero-dep on hot path.** No `serde`, no `tokio`, no `glam`. Use
//!    `crate::simd::{F32x16, PREFERRED_F32_LANES}` for all SIMD.
//! 2. **SoA, 64-byte aligned, padded to `PREFERRED_F32_LANES`.** Every
//!    buffer length is `pad_to_lanes(n, L)`. No scalar tails.
//! 3. **Click P-1 method discipline.** Operations on carriers:
//!    `frame.bin_tile(g)`, not `bin_tile(frame, g)`.
//! 4. **`#[repr(C, align(N))]` on cross-FFI structs, `#[repr(u8)]` on
//!    enums.** No `#[derive(Serialize)]`.
//! 5. **Per-tier SIMD via `crate::simd` polyfill.** Same pattern as
//!    `hpc::vsa`. Compile-time routes to AVX-512 / AVX2 / NEON /
//!    scalar — never hand-write intrinsics here.
//! 6. **Module docs lead with the math.** Every `.rs` opens with `//!`
//!    stating the equation it implements and citing the paper section.
//! 7. **The cognitive `splat.rs` is sacred.** `lance_graph_contract::splat`
//!    is the cognitive splat (CAM-plane deposition); this `splat3d` is
//!    the graphics splat. They are siblings, not parent/child.
//!
//! # Module layout (PRs landing in order)
//!
//! - [`spd3`] — symmetric 3×3 SPD math: Smith-1961 eigendecomp,
//!   `Spd3::pow(t)`, `sqrt`, `log_spd`, `from_scale_quat`,
//!   `sandwich(M, N)` + `sandwich_x16`. **PR 1.**
//! - [`gaussian`] — `GaussianBatch` SoA storage + `Gaussian3D`
//!   convenience constructor. **PR 2.**
//! - [`sh`] — degree-3 spherical-harmonics evaluator (RGB color
//!   from view direction). **PR 2.**
//! - `project` — EWA projection kernel. **PR 3.**
//! - `tile` — frustum cull + tile binning + radix sort. **PR 4.**
//! - `raster` — depth-sorted alpha-blend with `F32x16` pixel rows. **PR 5.**
//! - `frame` — `SplatFrame` double-buffer (sibling of
//!   `hpc::renderer::RenderFrame`). **PR 6.**
//!
//! # SIMD dispatch
//!
//! All SIMD goes through `crate::simd::F32x16`. The polyfill picks the
//! native width at compile time: AVX-512 (1× __m512), AVX2 (2× __m256),
//! NEON (4× float32x4_t), or scalar `[f32; 16]`. Consumer code never
//! mentions the tier — write once, run everywhere the workspace builds.
//!
//! ```ignore
//! use ndarray::simd::F32x16;
//!
//! let a = F32x16::splat(2.0);
//! let b = F32x16::from_slice(&[1.0; 16]);
//! let c = a.mul_add(b, F32x16::splat(0.5));  // a*b + 0.5, lanewise
//! ```
//!
//! # PR 1 surface (this commit)
//!
//! Only [`spd3`] is populated; the rest are placeholder declarations
//! that will fill in subsequent PRs. The Pillar-7 probe certifying the
//! EWA-sandwich math lives in `lance-graph/crates/jc/src/ewa_sandwich_3d.rs`
//! and runs against an independent f64 reference implementation — that
//! shared math claim is the contract these kernels must honor.

pub mod spd3;
pub mod gaussian;

pub use spd3::{sandwich, sandwich_x16, Spd3};
pub use gaussian::{GaussianBatch, Gaussian3D, SH_DEGREE, SH_COEFFS_PER_CHANNEL, SH_COEFFS_PER_GAUSSIAN};
