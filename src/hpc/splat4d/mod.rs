//! `splat4d` — 4-tier cascade Gaussian splatting pipeline (PR-X4).
//!
//! # Overview
//!
//! This module promotes `splat3d` from a bespoke 16×16 tile binner to a
//! *typed multi-resolution cognitive evolution operator* structured as a
//! (4×4)×(4×4)×(4×4)×(4×4) Gaussian-covariance cascade:
//!
//! ```text
//! L4 (16384²)  framebuffer / experience memory   ←  4×4 super-grid of L3
//! L3 (4096²)   scene aggregation                  ←  16×16 super-grid of L2
//! L2 (256²)    regional resonance                  ←  4×4 super-grid of L1
//! L1 (64²)     per-cell context                    ←  4×4 covariance footprint per splat
//! ```
//!
//! Verbatim load-bearing claim from `pr-x4-design.md`:57-72 (`c8f4af68`):
//! each tier covers 16× more area than the previous (uniform area-branching),
//! giving the L4/L1 area ratio = 16⁴ = 65,536. This matches the cell-count
//! ratio 16384²/64² = 65,536 exactly. Departing from 4×4 at any tier breaks
//! the structural identity between the graphics splat and the cognitive shader
//! (forbidden constraint #4 from the PR-X4 pre-sprint prompt).
//!
//! # SIMD bundles consumed (not extended)
//!
//! PR-X4 consumes the following bundles from `ndarray::simd` and MUST NOT
//! reach past them into raw `std::arch::*` intrinsics:
//!
//! | Bundle | Module | Purpose |
//! |---|---|---|
//! | **B-Cascade-Permute** | `cascade` | Cross-tier rotation L_k → L_{k+1} |
//! | **B-Compose** | `compose` | Horizontal reduction (alpha or NARS depending on feature flag) |
//! | **B-Gather-FMA** | `compose` | Weighted-mean fusion across 16 children |
//! | **B-Splat** | `pipeline` | Broadcast a Gaussian center across 16 tile lanes |
//! | **B-Pack-Dot** | `packed_dot` | INT4×32 packed dot (G2, thinking-style signatures) |
//! | **B-Interleave-Transpose** | `pipeline` | Row-major splat3d ↔ lane-major splat4d boundary |
//!
//! # Feature gates
//!
//! * `splat4d-nars-compose` (default OFF): swap the `B-Compose` closure from
//!   alpha-compositing to NARS truth-revision. W7's PR-X4 closure-swap sprint
//!   is the canonical activation point. Do NOT flip on by default in PR-X4.
//!
//! # Module layout
//!
//! - [`cascade`] — `CascadeAddr(u16)` 4-nibble addressing; L4 path gated on
//!   PR-X10 A12b Hilbert-3D fix (returns `Err(NotReadyL4)` until then).
//! - [`compose`] — L5 Gaussian moment-match (`compose_cascade`) and L6
//!   Pillar-8 temporal sandwich (`temporal_sandwich`).
//! - [`sh`] — G1 SH deg-3 inquiry-direction evaluation (A3 worker).
//! - [`packed_dot`] — G2 INT4×32 packed dot, three backends (A4 worker).
//! - [`revise`] — G3 NARS truth-revision kernel + G4 `fast_exp_x16` audit
//!   (A5 worker).
//! - [`pipeline`] — `frame_pipeline` entry point wired for Railway A6 smoke
//!   deployment. Exercisable at L1-L3 before A12b L4 fix lands.
//! - [`compose_kernel`] — closure-swappable composition kernel bridge:
//!   alpha-compositing (default) or NARS-revision (`splat4d-nars-compose`).
//!
//! # Cross-references
//!
//! - `pr-x4-design.md` (merged PR #162, `c8f4af68`) — master design doc
//! - `pr-x4-planning/02-a2-cascadeaddr-brief.md` — A2 CascadeAddr spec
//! - `pr-x4-planning/07-l5-l6-cascade-composition.md` — L5/L6 composition brief
//! - `hhtl-pr-x4-splat-cascade-pre-sprint-prompt.md` — W4-W5 sprint prompt
//! - `pp13-brutally-honest-tester-verdict.md` § P0-4 — L4 Hilbert-3D bug

pub mod cascade;
pub mod compose;
pub mod sh;
pub mod packed_dot;
pub mod revise;
pub mod pipeline;
pub mod compose_kernel;

pub use cascade::{CascadeAddr, NotReadyL4};
pub use compose::{compose_cascade, temporal_sandwich};
