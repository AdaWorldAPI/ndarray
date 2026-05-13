//! # jitson — JSON config → native code via Cranelift JIT
//!
//! ## Always available (no feature flags)
//! - [`parser`]: no_std JSON parser with bracket recovery
//! - [`validator`]: Schema validation, instruction→feature mapping
//! - [`template`]: [`JitsonTemplate`], `from_json()`, `template_hash()`
//! - [`precompile`]: WAL precompile queue, prefetch addressing
//! - [`scan_config`]: [`ScanConfig`], SIMD kernel trampolines, non-JIT scan
//! - [`packed`]: Re-export of [`crate::hpc::packed`] (PackedDatabase)
//!
//! ## `jit-native` feature (Cranelift JIT compilation)
//! See [`crate::hpc::jitson_cranelift`] for:
//! - `ScanParams`, `PhilosopherIR`, `RecipeIR`, `JitError`
//! - `CpuCaps` — CPU feature detection
//! - `JitEngine`, `JitEngineBuilder` — compile scan params to native code
//! - `ScanKernel` — compiled native function pointer
//!
//! ```toml
//! [dependencies]
//! # Parser + validator + template + scan (no Cranelift)
//! ndarray = { version = "0.17" }
//! # Full JIT compilation via Cranelift
//! ndarray = { version = "0.17", features = ["jit-native"] }
//! ```

pub mod parser;
pub mod validator;
pub mod template;
pub mod precompile;
pub mod scan_config;
pub mod packed;
pub mod noise;

// Re-exports: parser layer
pub use parser::{parse_json, JsonValue, ParseError};
pub use validator::{validate, ValidationError};

// Re-exports: template layer
pub use template::{
    check_pipeline_features, from_json, template_hash, BackendConfig, JitsonError, JitsonTemplate, PipelineStage,
};

// Re-exports: precompile queue
pub use precompile::{CompileState, PrecompileEntry, PrecompileQueue};

// Re-exports: scan config + SIMD trampolines
pub use scan_config::{
    jit_symbol_table, scan_hamming, DefaultKernelRegistry, ScanConfig, ScanResult, SimdKernelRegistry,
};

// Re-exports: noise parameters + terrain templates
pub use noise::{simple_noise_3d, CompiledNoiseConfig, NoiseParams, TerrainFillParams, GRAD3};
