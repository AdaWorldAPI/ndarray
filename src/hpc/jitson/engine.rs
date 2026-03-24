//! JIT compilation engine — Cranelift infrastructure.
//!
//! The `JitEngine` manages Cranelift's JIT module, compiles IR to native code,
//! and caches compiled kernels by their parameter hash.
//!
//! ## Data-flow compliance
//!
//! `compile_scan` takes `&self` (not `&mut self`) — the kernel cache uses
//! `RwLock` for interior mutability. The compute path never requires
//! exclusive access to the engine.

use std::collections::HashMap;
use std::sync::RwLock;

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, Signature, UserFuncName};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use super::detect::CpuCaps;
use super::ir::{JitError, ScanParams};
use super::scan::ScanKernel;

/// Builder for creating a JIT engine with registered external functions.
///
/// External functions must be registered before engine creation because
/// Cranelift's JIT module resolves symbols at link time.
pub struct JitEngineBuilder {
    symbols: HashMap<String, *const u8>,
}

// SAFETY: function pointers are safe to send across threads.
unsafe impl Send for JitEngineBuilder {}

impl JitEngineBuilder {
    /// Create a new builder with no registered symbols.
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }

    /// Register an external function that JIT code can call by name.
    ///
    /// # Safety
    ///
    /// The function pointer must remain valid for the lifetime of the engine.
    pub unsafe fn register_fn(mut self, name: &str, ptr: *const u8) -> Self {
        self.symbols.insert(name.to_string(), ptr);
        self
    }

    /// Build the JIT engine with all registered symbols.
    pub fn build(self) -> Result<JitEngine, JitError> {
        let caps = CpuCaps::detect();

        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        let isa_builder = cranelift_codegen::isa::lookup(target_lexicon::Triple::host())
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register external symbols so JIT code can call them.
        // Convert *const u8 to usize to satisfy Send requirements,
        // then convert back in the lookup closure.
        let symbols: HashMap<String, usize> = self
            .symbols
            .into_iter()
            .map(|(k, v)| (k, v as usize))
            .collect();
        builder.symbol_lookup_fn(Box::new(move |name| {
            symbols.get(name).map(|&addr| addr as *const u8)
        }));

        let module = JITModule::new(builder);

        Ok(JitEngine {
            module: RwLock::new(module),
            caps,
            scan_cache: RwLock::new(HashMap::new()),
        })
    }
}

impl Default for JitEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The JIT compilation engine.
///
/// Holds a Cranelift `JITModule` and a cache of compiled kernels.
/// Thread-safe: compiled function pointers can be shared across threads.
///
/// **No `&mut self` during computation.** The kernel cache and module
/// use `RwLock` for interior mutability — `compile_scan` takes `&self`.
pub struct JitEngine {
    /// Cranelift JIT module — owns the compiled code pages.
    /// Behind RwLock: only locked for writes during compilation (cache miss).
    module: RwLock<JITModule>,

    /// CPU capabilities detected at engine creation.
    pub caps: CpuCaps,

    /// Compiled scan kernel cache: params hash -> (fn_ptr, FuncId).
    /// RwLock: read-locked for cache hits, write-locked for cache misses.
    scan_cache: RwLock<HashMap<u64, (*const u8, FuncId)>>,
}

// SAFETY: JITModule's compiled code pages are immutable after finalization.
// Function pointers are safe to call from any thread.
// The RwLock provides synchronized access to mutable state.
unsafe impl Send for JitEngine {}
unsafe impl Sync for JitEngine {}

impl JitEngine {
    /// Create a new JIT engine with auto-detected CPU features.
    /// For engines with external functions, use `JitEngineBuilder` instead.
    pub fn new() -> Result<Self, JitError> {
        JitEngineBuilder::new().build()
    }

    /// Compile a scan kernel with the given parameters baked as immediates.
    ///
    /// Returns a `ScanKernel` whose `scan()` method is a native function
    /// pointer with `threshold`, `prefetch_ahead`, `record_size` etc.
    /// compiled as immediate operands.
    ///
    /// Takes `&self` — cache misses acquire a write lock internally.
    pub fn compile_scan(&self, params: ScanParams) -> Result<ScanKernel, JitError> {
        self.compile_scan_with_distance(params, None)
    }

    /// Compile a hybrid scan kernel that calls an external distance function.
    ///
    /// The `distance_fn_name` must be a symbol registered via
    /// `JitEngineBuilder::register_fn()`.
    /// Signature: `fn(a: *const u8, b: *const u8, len: u64) -> u64`
    ///
    /// Takes `&self` — cache misses acquire a write lock internally.
    pub fn compile_hybrid_scan(
        &self,
        params: ScanParams,
        distance_fn_name: &str,
    ) -> Result<ScanKernel, JitError> {
        self.compile_scan_with_distance(params, Some(distance_fn_name))
    }

    fn compile_scan_with_distance(
        &self,
        params: ScanParams,
        distance_fn: Option<&str>,
    ) -> Result<ScanKernel, JitError> {
        // Fast path: check cache with read lock
        let cache_key = params_hash(&params, distance_fn);
        {
            let cache = self
                .scan_cache
                .read()
                .map_err(|e| JitError::Module(format!("cache lock poisoned: {e}")))?;
            if let Some(&(ptr, _)) = cache.get(&cache_key) {
                return Ok(ScanKernel::from_raw(ptr, params));
            }
        }

        // Slow path: acquire write locks for compilation
        let mut module = self
            .module
            .write()
            .map_err(|e| JitError::Module(format!("module lock poisoned: {e}")))?;

        // Double-check: another thread may have compiled while we waited
        {
            let cache = self
                .scan_cache
                .read()
                .map_err(|e| JitError::Module(format!("cache lock poisoned: {e}")))?;
            if let Some(&(ptr, _)) = cache.get(&cache_key) {
                return Ok(ScanKernel::from_raw(ptr, params));
            }
        }

        // Build the scan function
        let func_name = format!("scan_{cache_key:x}");
        let sig = scan_signature(&module);

        let func_id = module
            .declare_function(&func_name, Linkage::Local, &sig)
            .map_err(|e| JitError::Module(e.to_string()))?;

        // If using a distance function, declare it as an import
        let dist_func_id = if let Some(dist_name) = distance_fn {
            let dist_sig = distance_signature(&module);
            let id = module
                .declare_function(dist_name, Linkage::Import, &dist_sig)
                .map_err(|e| JitError::Module(e.to_string()))?;
            Some(id)
        } else {
            None
        };

        let mut ctx = module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        // If hybrid mode, get a FuncRef for the distance function
        let dist_func_ref = if let Some(fid) = dist_func_id {
            Some(module.declare_func_in_func(fid, &mut ctx.func))
        } else {
            None
        };

        // Generate the scan loop IR
        super::scan::build_scan_ir(&mut ctx.func, &params, dist_func_ref)?;

        // Compile
        module
            .define_function(func_id, &mut ctx)
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        module.clear_context(&mut ctx);
        module
            .finalize_definitions()
            .map_err(|e| JitError::Codegen(format!("{e:?}")))?;

        let code_ptr = module.get_finalized_function(func_id);

        // Insert into cache
        {
            let mut cache = self
                .scan_cache
                .write()
                .map_err(|e| JitError::Module(format!("cache lock poisoned: {e}")))?;
            cache.insert(cache_key, (code_ptr, func_id));
        }

        Ok(ScanKernel::from_raw(code_ptr, params))
    }

    /// Get the number of cached kernels.
    pub fn cached_count(&self) -> usize {
        self.scan_cache.read().map(|c| c.len()).unwrap_or(0)
    }
}

/// Scan function signature:
/// `fn(query: *const u8, field: *const u8, field_len: u64,
///     record_size: u64, candidates_out: *mut u64) -> u64`
fn scan_signature(module: &JITModule) -> Signature {
    let ptr_type = module.target_config().pointer_type();
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type)); // query
    sig.params.push(AbiParam::new(ptr_type)); // field
    sig.params.push(AbiParam::new(types::I64)); // field_len
    sig.params.push(AbiParam::new(types::I64)); // record_size
    sig.params.push(AbiParam::new(ptr_type)); // candidates_out
    sig.returns.push(AbiParam::new(types::I64)); // num_candidates
    sig
}

/// Distance function signature (for hybrid scan):
/// `fn(a: *const u8, b: *const u8, len: u64) -> u64`
fn distance_signature(module: &JITModule) -> Signature {
    let ptr_type = module.target_config().pointer_type();
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(ptr_type)); // a
    sig.params.push(AbiParam::new(ptr_type)); // b
    sig.params.push(AbiParam::new(types::I64)); // len
    sig.returns.push(AbiParam::new(types::I64)); // distance
    sig
}

/// Hash scan params + optional distance fn name for cache lookup.
fn params_hash(params: &ScanParams, dist_fn: Option<&str>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    params.threshold.hash(&mut hasher);
    params.top_k.hash(&mut hasher);
    params.prefetch_ahead.hash(&mut hasher);
    params.record_size.hash(&mut hasher);
    if let Some(ref mask) = params.focus_mask {
        mask.hash(&mut hasher);
    }
    if let Some(name) = dist_fn {
        name.hash(&mut hasher);
    }
    hasher.finish()
}
