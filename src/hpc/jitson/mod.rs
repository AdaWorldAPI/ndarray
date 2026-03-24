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
    from_json, check_pipeline_features, template_hash,
    JitsonTemplate, PipelineStage, BackendConfig, JitsonError,
};

// Re-exports: precompile queue
pub use precompile::{PrecompileQueue, PrecompileEntry, CompileState};

// Re-exports: scan config + SIMD trampolines
pub use scan_config::{
    ScanConfig, ScanResult, SimdKernelRegistry, DefaultKernelRegistry,
    scan_hamming, jit_symbol_table,
};

// Re-exports: noise parameters + terrain templates
pub use noise::{NoiseParams, GRAD3, simple_noise_3d, CompiledNoiseConfig, TerrainFillParams};
