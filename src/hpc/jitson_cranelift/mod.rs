//! # jitson_cranelift — Cranelift JIT compilation backend
//!
//! This module is **only compiled** with the `jit-native` feature flag.
//! It provides native code generation from scan parameters via Cranelift:
//!
//! - [`ir`]: `ScanParams`, `PhilosopherIR`, `RecipeIR`, `JitError`
//! - [`detect`]: `CpuCaps` — runtime CPU feature detection
//! - [`engine`]: `JitEngine`, `JitEngineBuilder` — Cranelift JIT module + kernel cache
//! - [`scan_jit`]: `ScanKernel`, `build_scan_ir()` — Cranelift IR codegen
//!
//! ```toml
//! [dependencies]
//! ndarray = { version = "0.17", features = ["jit-native"] }
//! ```

pub mod ir;
pub mod detect;
pub mod engine;
pub mod scan_jit;
pub mod noise_jit;

pub use ir::*;
pub use detect::CpuCaps;
pub use engine::{JitEngine, JitEngineBuilder};
pub use scan_jit::ScanKernel;
pub use noise_jit::{NoiseKernel, NoiseKernelParams};
