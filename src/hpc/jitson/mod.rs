//! # jitson — JSON config → native code via Cranelift JIT
//!
//! This module provides:
//!
//! ## Always available (no feature flags)
//! - IR types: [`ScanParams`], [`PhilosopherIR`], [`RecipeIR`], [`JitError`]
//! - CPU detection: [`CpuCaps`]
//!
//! ## `jitson` feature (no Cranelift dependency)
//! - [`parser`]: no_std JSON parser with bracket recovery
//! - [`validator`]: Schema validation for JITSON templates
//! - [`template`]: [`JitsonTemplate`], conversion, precompile queue
//! - [`scan_config`]: [`ScanConfig`], SIMD trampolines, non-JIT scan path
//!
//! ## `jit` feature (Cranelift JIT compilation)
//! - [`JitEngine`], [`JitEngineBuilder`]: compile scan params to native code
//! - [`ScanKernel`]: compiled native function pointer
//!
//! ```toml
//! [dependencies]
//! # Parser + validator + template + scan (no Cranelift)
//! ndarray = { version = "0.17", features = ["jitson"] }
//! # Full JIT compilation via Cranelift
//! ndarray = { version = "0.17", features = ["jit"] }
//! ```

// Always available: pure Rust, no heavy dependencies
pub mod ir;
pub mod detect;

// Behind "jitson" feature: parser, validator, template, scan_config
// No Cranelift dependency — works in no_std/WASM contexts.
pub mod parser;
pub mod validator;
pub mod template;
pub mod scan_config;

// Behind "jit" feature: Cranelift JIT compilation engine
#[cfg(feature = "jit")]
pub mod engine;
#[cfg(feature = "jit")]
pub mod scan;

// Re-exports: always available
pub use ir::*;
pub use detect::CpuCaps;

// Re-exports: jitson layer
pub use parser::{parse_json, JsonValue, ParseError};
pub use validator::{validate, ValidationError};
pub use template::{
    from_json, check_pipeline_features, template_hash,
    JitsonTemplate, PipelineStage, BackendConfig, JitsonError,
    PrecompileQueue, PrecompileEntry, CompileState,
};
pub use scan_config::{
    ScanConfig, ScanResult, SimdKernelRegistry, DefaultKernelRegistry,
    scan_hamming, jit_symbol_table,
};

// Re-exports: JIT engine (Cranelift)
#[cfg(feature = "jit")]
pub use engine::{JitEngine, JitEngineBuilder};
#[cfg(feature = "jit")]
pub use scan::ScanKernel;
