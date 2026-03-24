//! # jitson — config values compiled to native code via Cranelift JIT
//!
//! This module provides JIT compilation of scan parameters into native
//! function pointers. Threshold comparisons become CMP immediates,
//! focus masks become VPANDQ bitmasks, branch weights become branch hints.
//!
//! ## Feature gating
//!
//! The IR types ([`ScanParams`], [`PhilosopherIR`], [`RecipeIR`], [`JitError`])
//! and CPU detection ([`CpuCaps`]) are always available — they are pure Rust
//! with no heavy dependencies.
//!
//! The JIT compilation engine ([`JitEngine`], [`ScanKernel`]) requires the
//! `jit` feature flag, which pulls in Cranelift:
//!
//! ```toml
//! [dependencies]
//! ndarray = { version = "0.17", features = ["jit"] }
//! ```

pub mod ir;
pub mod detect;

#[cfg(feature = "jit")]
pub mod engine;
#[cfg(feature = "jit")]
pub mod scan;

pub use ir::*;
pub use detect::CpuCaps;
#[cfg(feature = "jit")]
pub use engine::{JitEngine, JitEngineBuilder};
#[cfg(feature = "jit")]
pub use scan::ScanKernel;
