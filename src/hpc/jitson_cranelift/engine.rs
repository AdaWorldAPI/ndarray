//! JIT compilation engine — two-phase lifecycle via `LazyLock::get_mut`.
//!
//! ## Lifecycle
//!
//! ```text
//! Phase 1 — BUILD (&mut self, single-threaded):
//!   compile() / compile_hybrid() / compile_batch()
//!   LazyLock::get_mut → &mut KernelCache
//!   No locks. No contention.
//!
//! Phase 2 — RUN (&self via Arc, zero-cost reads):
//!   get() → Option<ScanKernel>
//!   LazyLock deref → &KernelCache (HashMap::get)
//!   No lock. No atomic. No compilation.
//! ```
//!
//! ## Data-flow compliance
//!
//! `compile()` takes `&mut self` — the BUILD phase.
//! `get()` takes `&self` — the RUN phase.
//! Once shared via `Arc<JitEngine>`, `&mut self` is unreachable,
//! so the cache is frozen by Rust's ownership system.

use std::collections::HashMap;
use std::sync::LazyLock;

use cranelift_codegen::ir::types;
use cranelift_codegen::ir::{AbiParam, Signature, UserFuncName};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use super::detect::CpuCaps;
use super::ir::{JitError, ScanParams};
use super::scan_jit::ScanKernel;

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
        let symbols: HashMap<String, usize> = self
            .symbols
            .into_iter()
            .map(|(k, v)| (k, v as usize))
            .collect();
        builder.symbol_lookup_fn(Box::new(move |name| {
            symbols.get(name).map(|&addr| addr as *const u8)
        }));

        let module = JITModule::new(builder);

        // Initialize the LazyLock immediately so get_mut() works during BUILD.
        fn empty_cache() -> KernelCache {
            KernelCache {
                map: HashMap::new(),
                prefetch_chain: Vec::new(),
            }
        }
        let cache = LazyLock::new(empty_cache as fn() -> KernelCache);
        LazyLock::force(&cache);

        Ok(JitEngine { module, caps, cache })
    }
}

impl Default for JitEngineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// The frozen kernel registry. Array-indexed for hot path.
struct KernelCache {
    /// Hash → compiled kernel. Immutable after freeze.
    map: HashMap<u64, CachedKernel>,
    /// Ordered list for prefetch chain (WAL precompile queue order).
    prefetch_chain: Vec<(u64, *const u8)>,
}

struct CachedKernel {
    fn_ptr: *const u8,
    #[allow(dead_code)] // retained for future hot-swap / eviction by FuncId
    func_id: FuncId,
    params: ScanParams,
}

// SAFETY: compiled code pages are immutable. Function pointers are Send+Sync.
unsafe impl Send for KernelCache {}
unsafe impl Sync for KernelCache {}

/// JIT compilation engine with two-phase lifecycle:
///
/// 1. **BUILD** — compile kernels via `compile()` (`&mut self`)
/// 2. **RUN** — lookup kernels via `get()` (`&self`, zero-cost)
///
/// After wrapping in `Arc<JitEngine>`, `&mut self` is unreachable:
/// the cache is frozen by Rust's ownership system. No locks needed.
///
/// ```text
/// RwLock cache hit:    ~25ns (atomic read, memory barrier)
/// LazyLock frozen get:  ~5ns (plain HashMap::get, no synchronization)
/// ```
pub struct JitEngine {
    /// Cranelift JIT module — owns the compiled code pages.
    /// Only accessed during BUILD phase (&mut self).
    module: JITModule,

    /// CPU capabilities detected at engine creation.
    pub caps: CpuCaps,

    /// Kernel cache. Mutable during BUILD (via get_mut), frozen during RUN.
    cache: LazyLock<KernelCache>,
}

// SAFETY: JITModule's compiled code pages are immutable after finalization.
// Function pointers are safe to call from any thread.
// During RUN phase, module is never accessed — only cached fn pointers are used.
unsafe impl Send for JitEngine {}
unsafe impl Sync for JitEngine {}

impl JitEngine {
    /// Create a new JIT engine with auto-detected CPU features.
    /// For engines with external functions, use `JitEngineBuilder` instead.
    pub fn new() -> Result<Self, JitError> {
        JitEngineBuilder::new().build()
    }

    // ── Phase 1: BUILD (mutable, single-threaded) ─────────────

    /// Compile a scan kernel and add it to the cache.
    /// Only works during BUILD phase (before sharing via Arc).
    ///
    /// Panics if called after the cache is frozen (shouldn't be possible
    /// through safe code — `&mut self` is unreachable after `Arc::new()`).
    pub fn compile(&mut self, params: ScanParams) -> Result<u64, JitError> {
        self.compile_inner(params, None)
    }

    /// Compile a hybrid scan kernel that calls an external distance function.
    ///
    /// The `distance_fn_name` must be a symbol registered via
    /// `JitEngineBuilder::register_fn()`.
    pub fn compile_hybrid(
        &mut self,
        params: ScanParams,
        distance_fn_name: &str,
    ) -> Result<u64, JitError> {
        self.compile_inner(params, Some(distance_fn_name))
    }

    /// Compile all kernels from a slice of params (batch BUILD).
    pub fn compile_batch(&mut self, queue: &[ScanParams]) -> Result<Vec<u64>, JitError> {
        queue.iter().map(|p| self.compile(p.clone())).collect()
    }

    fn compile_inner(
        &mut self,
        params: ScanParams,
        distance_fn: Option<&str>,
    ) -> Result<u64, JitError> {
        let cache_key = params_hash(&params, distance_fn);

        let cache = LazyLock::get_mut(&mut self.cache)
            .expect("JitEngine: cannot compile after freeze — cache is immutable");

        // Already compiled? Return existing hash.
        if cache.map.contains_key(&cache_key) {
            return Ok(cache_key);
        }

        // Build the scan function
        let func_name = format!("scan_{cache_key:x}");
        let sig = scan_signature(&self.module);

        let func_id = self
            .module
            .declare_function(&func_name, Linkage::Local, &sig)
            .map_err(|e| JitError::Module(e.to_string()))?;

        // If using a distance function, declare it as an import
        let dist_func_id = if let Some(dist_name) = distance_fn {
            let dist_sig = distance_signature(&self.module);
            let id = self
                .module
                .declare_function(dist_name, Linkage::Import, &dist_sig)
                .map_err(|e| JitError::Module(e.to_string()))?;
            Some(id)
        } else {
            None
        };

        let mut ctx = self.module.make_context();
        ctx.func.signature = sig;
        ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        // If hybrid mode, get a FuncRef for the distance function
        let dist_func_ref = if let Some(fid) = dist_func_id {
            Some(self.module.declare_func_in_func(fid, &mut ctx.func))
        } else {
            None
        };

        // Generate the scan loop IR
        super::scan_jit::build_scan_ir(&mut ctx.func, &params, dist_func_ref)?;

        // Compile
        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| JitError::Codegen(e.to_string()))?;

        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| JitError::Codegen(format!("{e:?}")))?;

        let code_ptr = self.module.get_finalized_function(func_id);

        // Insert into cache and prefetch chain
        cache.map.insert(
            cache_key,
            CachedKernel {
                fn_ptr: code_ptr,
                func_id,
                params: params.clone(),
            },
        );
        cache.prefetch_chain.push((cache_key, code_ptr));

        Ok(cache_key)
    }

    // ── Phase 2: RUN (frozen, zero-cost) ──────────────────────

    /// Look up a compiled kernel by hash. Zero-cost after freeze.
    /// Returns None if the kernel wasn't compiled during BUILD.
    #[inline(always)]
    pub fn get(&self, hash: u64) -> Option<ScanKernel> {
        // Deref through LazyLock → &KernelCache. No lock. No atomic.
        self.cache
            .map
            .get(&hash)
            .map(|k| ScanKernel::from_raw(k.fn_ptr, k.params.clone()))
    }

    /// Prefetch the NEXT kernel's code page.
    /// Call while executing kernel N to warm L1 for kernel N+1.
    #[inline(always)]
    pub fn prefetch_next(&self, current_hash: u64) {
        let chain = &self.cache.prefetch_chain;
        if let Some(idx) = chain.iter().position(|(h, _)| *h == current_hash) {
            if let Some(&(_, next_ptr)) = chain.get(idx + 1) {
                // SAFETY: _mm_prefetch is a CPU hint — no UB for any pointer value.
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    #[cfg(target_feature = "sse")]
                    core::arch::x86_64::_mm_prefetch::<
                        { core::arch::x86_64::_MM_HINT_T0 },
                    >(next_ptr as *const i8);
                }
                let _ = next_ptr; // suppress unused warning on non-x86
            }
        }
    }

    /// Number of compiled kernels.
    pub fn kernel_count(&self) -> usize {
        self.cache.map.len()
    }

    /// Is the cache initialized? (Always true after build().)
    /// The real "freeze" is ownership-based: once in Arc, &mut self is gone.
    pub fn is_frozen(&self) -> bool {
        LazyLock::get(&self.cache).is_some()
    }

    // ── Backward-compat shims ─────────────────────────────────

    /// Compile a scan kernel (legacy API — delegates to `compile()`).
    ///
    /// Unlike the old `&self` version, this requires `&mut self`.
    /// If you need the old lazy-compile-on-miss behavior, compile
    /// all expected kernels during BUILD phase instead.
    pub fn compile_scan(&mut self, params: ScanParams) -> Result<ScanKernel, JitError> {
        let hash = self.compile(params.clone())?;
        Ok(self.get(hash).expect("just compiled"))
    }

    /// Compile a hybrid scan kernel (legacy API).
    pub fn compile_hybrid_scan(
        &mut self,
        params: ScanParams,
        distance_fn_name: &str,
    ) -> Result<ScanKernel, JitError> {
        let hash = self.compile_hybrid(params.clone(), distance_fn_name)?;
        Ok(self.get(hash).expect("just compiled"))
    }

    /// Get the number of cached kernels (legacy name).
    pub fn cached_count(&self) -> usize {
        self.kernel_count()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_build_and_compile() {
        let mut engine = JitEngine::new().unwrap();
        assert_eq!(engine.kernel_count(), 0);

        let params = ScanParams::default();
        let hash = engine.compile(params.clone()).unwrap();
        assert_eq!(engine.kernel_count(), 1);

        // Dedup: same params → same hash, no new kernel
        let hash2 = engine.compile(params).unwrap();
        assert_eq!(hash, hash2);
        assert_eq!(engine.kernel_count(), 1);
    }

    #[test]
    fn engine_get_after_compile() {
        let mut engine = JitEngine::new().unwrap();
        let params = ScanParams {
            threshold: 100,
            top_k: 10,
            prefetch_ahead: 2,
            focus_mask: None,
            record_size: 256,
        };
        let hash = engine.compile(params).unwrap();

        let kernel = engine.get(hash);
        assert!(kernel.is_some());
        assert_eq!(kernel.unwrap().params.threshold, 100);
    }

    #[test]
    fn engine_get_missing_returns_none() {
        let engine = JitEngine::new().unwrap();
        assert!(engine.get(0xDEAD).is_none());
    }

    #[test]
    fn engine_compile_batch() {
        let mut engine = JitEngine::new().unwrap();
        let params_list = vec![
            ScanParams { threshold: 100, ..ScanParams::default() },
            ScanParams { threshold: 200, ..ScanParams::default() },
            ScanParams { threshold: 300, ..ScanParams::default() },
        ];
        let hashes = engine.compile_batch(&params_list).unwrap();
        assert_eq!(hashes.len(), 3);
        assert_eq!(engine.kernel_count(), 3);

        // All lookups work
        for hash in &hashes {
            assert!(engine.get(*hash).is_some());
        }
    }

    #[test]
    fn engine_legacy_compile_scan() {
        let mut engine = JitEngine::new().unwrap();
        let kernel = engine.compile_scan(ScanParams::default()).unwrap();
        assert_eq!(kernel.params.threshold, 500);
        assert_eq!(engine.cached_count(), 1);
    }

    #[test]
    fn engine_is_frozen() {
        let engine = JitEngine::new().unwrap();
        // After build(), LazyLock is force-initialized → is_frozen() returns true
        assert!(engine.is_frozen());
    }

    #[test]
    fn engine_prefetch_chain_order() {
        let mut engine = JitEngine::new().unwrap();
        let h1 = engine
            .compile(ScanParams { threshold: 10, ..ScanParams::default() })
            .unwrap();
        let h2 = engine
            .compile(ScanParams { threshold: 20, ..ScanParams::default() })
            .unwrap();
        let _h3 = engine
            .compile(ScanParams { threshold: 30, ..ScanParams::default() })
            .unwrap();

        // Verify prefetch chain is populated
        let chain = &LazyLock::get(&engine.cache).unwrap().prefetch_chain;
        assert_eq!(chain.len(), 3);
        assert_eq!(chain[0].0, h1);
        assert_eq!(chain[1].0, h2);
    }

    #[test]
    fn scan_correctness_inline() {
        let mut engine = JitEngine::new().unwrap();
        let params = ScanParams {
            threshold: 10,
            top_k: 5,
            prefetch_ahead: 0,
            focus_mask: None,
            record_size: 8,
        };
        let hash = engine.compile(params).unwrap();
        let kernel = engine.get(hash).unwrap();

        // 4 records of 8 bytes each. Query = 0x00..00.
        // Record 0: 0x01 (popcnt=1, < threshold 10) → match
        // Record 1: 0xFF (popcnt=8, < threshold 10) → match
        // Record 2: all 0xFF bytes (popcnt>10) → depends on inline POC (only first 8 bytes)
        let query = [0u8; 8];
        let field: Vec<u8> = vec![
            0x01, 0, 0, 0, 0, 0, 0, 0, // record 0: popcount=1
            0xFF, 0, 0, 0, 0, 0, 0, 0, // record 1: popcount=8
            0xFF, 0xFF, 0, 0, 0, 0, 0, 0, // record 2: popcount=16 (exceeds threshold)
            0x03, 0, 0, 0, 0, 0, 0, 0, // record 3: popcount=2
        ];
        let mut candidates = [0u64; 5];

        // SAFETY: test data is valid, candidates buffer is large enough.
        let count = unsafe {
            kernel.scan(
                query.as_ptr(),
                field.as_ptr(),
                4,  // 4 records
                8,  // record_size (ignored by JIT — baked as immediate)
                candidates.as_mut_ptr(),
            )
        };

        // Records 0, 1, 3 should match (popcount < 10)
        // Record 2 has popcount=16, rejected
        assert_eq!(count, 3);
        assert_eq!(candidates[0], 0); // record 0
        assert_eq!(candidates[1], 1); // record 1
        assert_eq!(candidates[2], 3); // record 3
    }

    #[test]
    fn kernel_caching_dedup() {
        let mut engine = JitEngine::new().unwrap();
        let p1 = ScanParams { threshold: 42, ..ScanParams::default() };
        let p2 = ScanParams { threshold: 42, ..ScanParams::default() };
        let p3 = ScanParams { threshold: 99, ..ScanParams::default() };

        let h1 = engine.compile(p1).unwrap();
        let h2 = engine.compile(p2).unwrap();
        let h3 = engine.compile(p3).unwrap();

        assert_eq!(h1, h2, "same params should produce same hash");
        assert_ne!(h1, h3, "different params should produce different hash");
        assert_eq!(engine.kernel_count(), 2);
    }
}
