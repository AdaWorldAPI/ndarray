//! Noise function JIT specialization.
//!
//! Generates a native multi-octave noise evaluation function where:
//! - `frequencies[i]` -> F64 immediate operand in FMUL (no memory fetch)
//! - `amplitudes[i]` -> F64 immediate operand in FMA (no memory fetch)
//! - `normalization` -> F64 immediate final scale (no memory fetch)
//! - `num_octaves` -> unrolled loop (straight-line code, no branches)
//!
//! The base noise function is external, called via `FuncRef`
//! (registered via `JitEngineBuilder::register_fn()`).

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, Signature, UserFuncName};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::JITModule;
use cranelift_module::{Linkage, Module};

use super::ir::JitError;

/// IR parameters for noise kernel compilation.
///
/// Each field maps to a specific instruction encoding:
/// - `num_octaves` -> unrolled loop bound (no branch, no counter)
/// - `frequencies[i]` -> F64 immediate operand in FMUL per octave
/// - `amplitudes[i]` -> F64 immediate operand in FMA per octave
/// - `normalization` -> F64 immediate in final FMUL
///
/// Cannot be `Copy` because `frequencies` and `amplitudes` are `Vec<f64>`.
#[derive(Debug, Clone)]
pub struct NoiseKernelParams {
    /// Number of octaves — baked as loop unroll count (no branch).
    pub num_octaves: u32,

    /// Per-octave frequency scale (baked as F64 immediates).
    /// Length must equal `num_octaves`.
    pub frequencies: Vec<f64>,

    /// Per-octave amplitude scale (baked as F64 immediates).
    /// Length must equal `num_octaves`.
    pub amplitudes: Vec<f64>,

    /// Final normalization scale factor (baked as F64 immediate).
    pub normalization: f64,
}

/// A compiled noise kernel — holds the native function pointer
/// and the params that generated it (for introspection).
pub struct NoiseKernel {
    /// The compiled noise function.
    /// Signature: `fn(x: f64, y: f64, z: f64) -> f64`
    fn_ptr: *const u8,

    /// Parameters that were baked into this kernel.
    pub params: NoiseKernelParams,
}

// SAFETY: The compiled code is immutable and thread-safe.
// Function pointers point to finalized Cranelift code pages
// which are never modified after compilation.
unsafe impl Send for NoiseKernel {}
// SAFETY: The compiled code is immutable and thread-safe.
// No mutable state is accessed through shared references.
unsafe impl Sync for NoiseKernel {}

impl NoiseKernel {
    /// Wrap a raw function pointer as a `NoiseKernel`.
    pub(crate) fn from_raw(ptr: *const u8, params: NoiseKernelParams) -> Self {
        Self {
            fn_ptr: ptr,
            params,
        }
    }

    /// Evaluate the compiled noise function at the given coordinates.
    ///
    /// # Safety
    ///
    /// - `self.fn_ptr` must point to a valid Cranelift-compiled function
    ///   with the signature `fn(f64, f64, f64) -> f64`.
    /// - The base noise function registered during compilation must still
    ///   be valid (not unloaded or freed).
    pub unsafe fn evaluate(&self, x: f64, y: f64, z: f64) -> f64 {
        // SAFETY: caller guarantees fn_ptr validity; fn_ptr was compiled
        // by Cranelift with the matching signature (f64, f64, f64) -> f64.
        let func: unsafe extern "C" fn(f64, f64, f64) -> f64 =
            std::mem::transmute(self.fn_ptr);
        func(x, y, z)
    }

    /// Get the raw function pointer (for benchmarking/introspection).
    pub fn as_fn_ptr(&self) -> *const u8 {
        self.fn_ptr
    }
}

/// Build a `NoiseKernelParams` from a `CompiledNoiseConfig`.
///
/// Maps the config's precomputed per-octave arrays directly into
/// the IR parameter struct for Cranelift code generation.
///
/// # Examples
///
/// ```ignore
/// use ndarray::hpc::jitson::noise::{NoiseParams, CompiledNoiseConfig};
/// use ndarray::hpc::jitson_cranelift::noise_jit::from_compiled_config;
///
/// let params = NoiseParams::perlin(4, 2.0, 0.5);
/// let config = CompiledNoiseConfig::from_params(&params, 42);
/// let kernel_params = from_compiled_config(&config);
/// assert_eq!(kernel_params.num_octaves, 4);
/// ```
pub fn from_compiled_config(
    config: &super::super::jitson::noise::CompiledNoiseConfig,
) -> NoiseKernelParams {
    NoiseKernelParams {
        num_octaves: config.frequencies.len() as u32,
        frequencies: config.frequencies.clone(),
        amplitudes: config.amplitudes.clone(),
        normalization: config.normalization,
    }
}

/// Build the Cranelift IR for a multi-octave noise function with baked-in parameters.
///
/// Generates a function with signature `fn(x: f64, y: f64, z: f64) -> f64`
/// that evaluates multi-octave noise by calling an external base noise function.
///
/// The octave loop is fully unrolled — each octave becomes straight-line code
/// with F64 immediates for frequency and amplitude. No branches in the hot path.
///
/// Generated pseudo-code:
/// ```text
/// fn noise(x: f64, y: f64, z: f64) -> f64:
///     value = 0.0
///     // Octave 0 (unrolled):
///     value += AMP_0 * base_noise(x * FREQ_0, y * FREQ_0, z * FREQ_0)
///     // Octave 1 (unrolled):
///     value += AMP_1 * base_noise(x * FREQ_1, y * FREQ_1, z * FREQ_1)
///     // ... (one block per octave, no loop)
///     value *= NORMALIZATION
///     return value
/// ```
pub fn build_noise_ir(
    func: &mut Function,
    params: &NoiseKernelParams,
    base_noise_ref: cranelift_codegen::ir::FuncRef,
) -> Result<(), JitError> {
    // Validate params
    let n = params.num_octaves as usize;
    if params.frequencies.len() != n {
        return Err(JitError::InvalidParams(format!(
            "frequencies length {} != num_octaves {}",
            params.frequencies.len(),
            n
        )));
    }
    if params.amplitudes.len() != n {
        return Err(JitError::InvalidParams(format!(
            "amplitudes length {} != num_octaves {}",
            params.amplitudes.len(),
            n
        )));
    }

    let mut fbc = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(func, &mut fbc);

    // Variable for accumulating noise value across octaves.
    let v_value = Variable::from_u32(0);
    builder.declare_var(v_value, types::F64);

    // Entry block — function signature: fn(x: f64, y: f64, z: f64) -> f64
    let entry = builder.create_block();
    builder.append_block_params_for_function_params(entry);
    builder.switch_to_block(entry);
    builder.seal_block(entry);

    // Get function parameters
    let x = builder.block_params(entry)[0];
    let y = builder.block_params(entry)[1];
    let z = builder.block_params(entry)[2];

    // Initialize accumulator: value = 0.0
    let zero_f64 = builder.ins().f64const(0.0);
    builder.def_var(v_value, zero_f64);

    // Unrolled octave loop — each iteration is straight-line code with
    // frequency/amplitude baked as F64 immediates.
    for i in 0..n {
        let freq_imm = builder.ins().f64const(params.frequencies[i]);
        let amp_imm = builder.ins().f64const(params.amplitudes[i]);

        // Scaled coordinates: sx = x * freq, sy = y * freq, sz = z * freq
        let sx = builder.ins().fmul(x, freq_imm);
        let sy = builder.ins().fmul(y, freq_imm);
        let sz = builder.ins().fmul(z, freq_imm);

        // CALL base_noise(sx, sy, sz)
        let call = builder.ins().call(base_noise_ref, &[sx, sy, sz]);
        let noise_val = builder.inst_results(call)[0];

        // value += amp * noise_val
        // FMA: fma(amp, noise_val, accum) = amp * noise_val + accum
        let accum = builder.use_var(v_value);
        let new_accum = builder.ins().fma(amp_imm, noise_val, accum);
        builder.def_var(v_value, new_accum);
    }

    // Final normalization: value *= normalization
    let normalization_imm = builder.ins().f64const(params.normalization);
    let accum = builder.use_var(v_value);
    let result = builder.ins().fmul(accum, normalization_imm);

    builder.ins().return_(&[result]);

    builder.finalize();
    Ok(())
}

/// Noise function signature: `fn(x: f64, y: f64, z: f64) -> f64`
fn noise_signature(module: &JITModule) -> Signature {
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::F64)); // x
    sig.params.push(AbiParam::new(types::F64)); // y
    sig.params.push(AbiParam::new(types::F64)); // z
    sig.returns.push(AbiParam::new(types::F64)); // result
    sig
}

/// Base noise function signature (external): `fn(f64, f64, f64) -> f64`
fn base_noise_signature(module: &JITModule) -> Signature {
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::F64)); // x
    sig.params.push(AbiParam::new(types::F64)); // y
    sig.params.push(AbiParam::new(types::F64)); // z
    sig.returns.push(AbiParam::new(types::F64)); // result
    sig
}

/// Hash noise kernel params + base noise name for cache lookup.
fn noise_params_hash(params: &NoiseKernelParams, base_noise_name: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    params.num_octaves.hash(&mut hasher);
    for f in &params.frequencies {
        f.to_bits().hash(&mut hasher);
    }
    for a in &params.amplitudes {
        a.to_bits().hash(&mut hasher);
    }
    params.normalization.to_bits().hash(&mut hasher);
    base_noise_name.hash(&mut hasher);
    hasher.finish()
}

/// Cached noise kernel entry (internal to engine).
pub(crate) struct CachedNoiseKernel {
    /// Compiled function pointer.
    fn_ptr: *const u8,
    /// Parameters baked into this kernel.
    params: NoiseKernelParams,
}

impl super::engine::JitEngine {
    /// Compile a noise kernel and add it to the cache.
    ///
    /// The `base_noise_name` must be a symbol registered via
    /// `JitEngineBuilder::register_fn()` with signature `fn(f64, f64, f64) -> f64`.
    ///
    /// Only works during BUILD phase (before sharing via `Arc`).
    ///
    /// Returns a cache hash that can be used with `get_noise()`.
    pub fn compile_noise(
        &mut self,
        params: NoiseKernelParams,
        base_noise_name: &str,
    ) -> Result<u64, JitError> {
        let cache_key = noise_params_hash(&params, base_noise_name);

        // Already compiled? Return existing hash.
        if self.noise_cache.contains_key(&cache_key) {
            return Ok(cache_key);
        }

        // Declare the noise function
        let func_name = format!("noise_{cache_key:x}");
        let sig = noise_signature(&self.module);

        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Local, &sig)
            .map_err(|e| JitError::Module(e.to_string()))?;

        // Declare the base noise function as an import
        let base_noise_sig = base_noise_signature(&self.module);
        let base_noise_id = self
            .module
            .declare_function(base_noise_name, Linkage::Import, &base_noise_sig)
            .map_err(|e| JitError::Module(e.to_string()))?;

        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        // Get a FuncRef for the base noise function
        let base_noise_ref = self
            .module
            .declare_func_in_func(base_noise_id, &mut ctx.func);

        // Generate the noise IR
        build_noise_ir(&mut ctx.func, &params, base_noise_ref)?;

        // Compile
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| JitError::Codegen(format!("{e:?}")))?;

        let code_ptr = self.module.get_finalized_function(func_id);

        self.noise_cache.insert(
            cache_key,
            CachedNoiseKernel {
                fn_ptr: code_ptr,
                params: params.clone(),
            },
        );

        Ok(cache_key)
    }

    /// Look up a compiled noise kernel by hash. Zero-cost after freeze.
    /// Returns `None` if the kernel wasn't compiled during BUILD.
    pub fn get_noise(&self, hash: u64) -> Option<NoiseKernel> {
        self.noise_cache
            .get(&hash)
            .map(|k| NoiseKernel::from_raw(k.fn_ptr, k.params.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::jitson::noise::{CompiledNoiseConfig, NoiseParams};

    #[test]
    fn test_noise_kernel_params_from_config() {
        let noise_params = NoiseParams::perlin(4, 2.0, 0.5);
        let config = CompiledNoiseConfig::from_params(&noise_params, 42);

        let kernel_params = from_compiled_config(&config);

        assert_eq!(kernel_params.num_octaves, 4);
        assert_eq!(kernel_params.frequencies.len(), 4);
        assert_eq!(kernel_params.amplitudes.len(), 4);

        // Verify frequencies roundtrip
        for i in 0..4 {
            assert!(
                (kernel_params.frequencies[i] - config.frequencies[i]).abs() < 1e-10,
                "frequency mismatch at octave {i}"
            );
        }

        // Verify amplitudes roundtrip
        for i in 0..4 {
            assert!(
                (kernel_params.amplitudes[i] - config.amplitudes[i]).abs() < 1e-10,
                "amplitude mismatch at octave {i}"
            );
        }

        // Verify normalization roundtrip
        assert!(
            (kernel_params.normalization - config.normalization).abs() < 1e-10,
            "normalization mismatch"
        );
    }

    #[test]
    fn test_build_noise_ir_compiles() {
        use cranelift_codegen::ir::{AbiParam, UserFuncName};
        use cranelift_codegen::isa::CallConv;
        use cranelift_codegen::settings;

        let params = NoiseKernelParams {
            num_octaves: 3,
            frequencies: vec![1.0, 2.0, 4.0],
            amplitudes: vec![1.0, 0.5, 0.25],
            normalization: 1.0 / 1.75,
        };

        // Build a minimal Function with the correct signature
        let call_conv = CallConv::SystemV;

        let mut sig = cranelift_codegen::ir::Signature::new(call_conv);
        sig.params.push(AbiParam::new(types::F64)); // x
        sig.params.push(AbiParam::new(types::F64)); // y
        sig.params.push(AbiParam::new(types::F64)); // z
        sig.returns.push(AbiParam::new(types::F64)); // result

        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);

        // Declare the external base noise function signature
        let mut base_sig = cranelift_codegen::ir::Signature::new(call_conv);
        base_sig.params.push(AbiParam::new(types::F64));
        base_sig.params.push(AbiParam::new(types::F64));
        base_sig.params.push(AbiParam::new(types::F64));
        base_sig.returns.push(AbiParam::new(types::F64));

        let base_noise_ref = func.import_function(cranelift_codegen::ir::ExtFuncData {
            name: cranelift_codegen::ir::ExternalName::user(0, 1),
            signature: func.import_signature(base_sig),
            colocated: false,
        });

        // Build the IR — should not error
        let result = build_noise_ir(&mut func, &params, base_noise_ref);
        assert!(result.is_ok(), "build_noise_ir failed: {result:?}");
    }

    #[test]
    fn test_noise_kernel_params_clone() {
        let params = NoiseKernelParams {
            num_octaves: 4,
            frequencies: vec![1.0, 2.0, 4.0, 8.0],
            amplitudes: vec![1.0, 0.5, 0.25, 0.125],
            normalization: 1.0 / 1.875,
        };

        let cloned = params.clone();
        assert_eq!(cloned.num_octaves, params.num_octaves);
        assert_eq!(cloned.frequencies.len(), params.frequencies.len());
        assert_eq!(cloned.amplitudes.len(), params.amplitudes.len());
        assert!((cloned.normalization - params.normalization).abs() < 1e-10);

        for i in 0..4 {
            assert!((cloned.frequencies[i] - params.frequencies[i]).abs() < 1e-10);
            assert!((cloned.amplitudes[i] - params.amplitudes[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_noise_kernel_send_sync() {
        /// Compile-time assertion that `T` implements Send + Sync.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<NoiseKernel>();
    }

    #[test]
    fn test_build_noise_ir_rejects_mismatched_frequencies() {
        use cranelift_codegen::ir::{AbiParam, UserFuncName};
        use cranelift_codegen::isa::CallConv;

        let params = NoiseKernelParams {
            num_octaves: 3,
            frequencies: vec![1.0, 2.0], // only 2, but num_octaves says 3
            amplitudes: vec![1.0, 0.5, 0.25],
            normalization: 1.0,
        };

        let call_conv = CallConv::SystemV;
        let mut sig = cranelift_codegen::ir::Signature::new(call_conv);
        sig.params.push(AbiParam::new(types::F64));
        sig.params.push(AbiParam::new(types::F64));
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));

        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);

        let mut base_sig = cranelift_codegen::ir::Signature::new(call_conv);
        base_sig.params.push(AbiParam::new(types::F64));
        base_sig.params.push(AbiParam::new(types::F64));
        base_sig.params.push(AbiParam::new(types::F64));
        base_sig.returns.push(AbiParam::new(types::F64));

        let base_noise_ref = func.import_function(cranelift_codegen::ir::ExtFuncData {
            name: cranelift_codegen::ir::ExternalName::user(0, 1),
            signature: func.import_signature(base_sig),
            colocated: false,
        });

        let result = build_noise_ir(&mut func, &params, base_noise_ref);
        assert!(
            result.is_err(),
            "should reject mismatched num_octaves vs frequencies"
        );
    }

    #[test]
    fn test_build_noise_ir_rejects_mismatched_amplitudes() {
        use cranelift_codegen::ir::{AbiParam, UserFuncName};
        use cranelift_codegen::isa::CallConv;

        let params = NoiseKernelParams {
            num_octaves: 2,
            frequencies: vec![1.0, 2.0],
            amplitudes: vec![1.0], // only 1, but num_octaves says 2
            normalization: 1.0,
        };

        let call_conv = CallConv::SystemV;
        let mut sig = cranelift_codegen::ir::Signature::new(call_conv);
        sig.params.push(AbiParam::new(types::F64));
        sig.params.push(AbiParam::new(types::F64));
        sig.params.push(AbiParam::new(types::F64));
        sig.returns.push(AbiParam::new(types::F64));

        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);

        let mut base_sig = cranelift_codegen::ir::Signature::new(call_conv);
        base_sig.params.push(AbiParam::new(types::F64));
        base_sig.params.push(AbiParam::new(types::F64));
        base_sig.params.push(AbiParam::new(types::F64));
        base_sig.returns.push(AbiParam::new(types::F64));

        let base_noise_ref = func.import_function(cranelift_codegen::ir::ExtFuncData {
            name: cranelift_codegen::ir::ExternalName::user(0, 1),
            signature: func.import_signature(base_sig),
            colocated: false,
        });

        let result = build_noise_ir(&mut func, &params, base_noise_ref);
        assert!(
            result.is_err(),
            "should reject mismatched num_octaves vs amplitudes"
        );
    }
}
