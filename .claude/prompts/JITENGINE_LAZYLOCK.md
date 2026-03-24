# JitEngine Refactor — RwLock → LazyLock::get_mut (Pre-populate-then-freeze)

## FIRST: Read .claude/rules/borrow-strategy.md (from q2, same principle applies)

## The Problem

JitEngine currently uses `RwLock<HashMap<u64, ScanKernel>>` for the kernel cache.
Every query during runtime takes a read-lock, and cache misses take a write-lock
to compile. This means:

- Lock contention under concurrent access
- Unpredictable latency spikes when a new config triggers compilation mid-tick
- RwLock overhead on EVERY cache hit (atomic operations, memory barriers)
- Compilation (521µs) happening during gameplay / graph queries / replay

## The Fix

Two-phase architecture using `LazyLock::get_mut` (stable in Rust 1.94):

```
Phase 1 — BUILD (mutable, single-threaded):
  LazyLock::get_mut(&mut engine.cache) → &mut HashMap
  Compile ALL kernels upfront. 100 kernels × 521µs = 52ms.
  No locks. No contention. Runs during startup/loading.

Phase 2 — RUN (frozen, zero-cost reads):
  &LazyLock → &HashMap (immutable reference)
  Function pointer lookup = HashMap::get(). No lock. No atomic.
  Zero contention. Zero compilation. Zero latency spikes.
  
  get_mut() returns None after first deref — the cache is frozen.
  Any attempt to compile during Phase 2 is a compile error, not a
  runtime error. The type system enforces the freeze.
```

## Implementation

### Current code (src/hpc/jitson_cranelift/engine.rs):

```rust
// CURRENT — wrong
pub struct JitEngine {
    module: JITModule,
    pub caps: CpuCaps,
    scan_cache: RwLock<HashMap<u64, (*const u8, FuncId)>>,  // ← lock on every access
}

impl JitEngine {
    pub fn compile_scan(&self, params: ScanParams) -> Result<ScanKernel, JitError> {
        // read-lock to check cache
        // write-lock to compile on miss
        // contention, latency spike
    }
}
```

### New code:

```rust
use std::sync::LazyLock;
use std::collections::HashMap;

/// JIT compilation engine with two-phase lifecycle:
/// 1. BUILD: compile kernels via `populate()` — mutable access
/// 2. RUN: lookup kernels via `get()` — immutable, zero-cost
///
/// After the first immutable access, `get_mut()` returns `None` and
/// no more kernels can be compiled. The cache is frozen.
pub struct JitEngine {
    module: JITModule,
    pub caps: CpuCaps,
    /// Kernel cache. Mutable during BUILD, frozen during RUN.
    cache: LazyLock<KernelCache>,
}

/// The frozen kernel registry. Array-indexed for hot path.
struct KernelCache {
    /// Hash → function pointer. Immutable after freeze.
    map: HashMap<u64, CachedKernel>,
    /// Ordered list for prefetch chain (WAL precompile queue order).
    prefetch_chain: Vec<(u64, *const u8)>,
}

struct CachedKernel {
    fn_ptr: *const u8,
    func_id: FuncId,
    params: ScanParams,
}

// Safety: compiled code pages are immutable. Function pointers are Send+Sync.
unsafe impl Send for KernelCache {}
unsafe impl Sync for KernelCache {}

impl JitEngine {
    pub fn new() -> Result<Self, JitError> {
        JitEngineBuilder::new().build()
    }

    // ── Phase 1: BUILD (mutable) ──────────────────────────────

    /// Compile a scan kernel and add it to the cache.
    /// Only works during BUILD phase (before any `get()` call).
    /// Panics if called after freeze.
    pub fn compile(&mut self, params: ScanParams) -> Result<u64, JitError> {
        let cache = LazyLock::get_mut(&mut self.cache)
            .expect("JitEngine: cannot compile after freeze — cache is immutable");
        
        let hash = params_hash(&params, None);
        if cache.map.contains_key(&hash) {
            return Ok(hash);  // already compiled
        }

        let (fn_ptr, func_id) = self.compile_inner(&params, None)?;
        cache.map.insert(hash, CachedKernel { fn_ptr, func_id, params: params.clone() });
        cache.prefetch_chain.push((hash, fn_ptr));
        Ok(hash)
    }

    /// Compile a hybrid scan kernel (JIT loop + external SIMD function).
    pub fn compile_hybrid(&mut self, params: ScanParams, distance_fn: &str) -> Result<u64, JitError> {
        let cache = LazyLock::get_mut(&mut self.cache)
            .expect("JitEngine: cannot compile after freeze");

        let hash = params_hash(&params, Some(distance_fn));
        if cache.map.contains_key(&hash) {
            return Ok(hash);
        }

        let (fn_ptr, func_id) = self.compile_inner(&params, Some(distance_fn))?;
        cache.map.insert(hash, CachedKernel { fn_ptr, func_id, params: params.clone() });
        cache.prefetch_chain.push((hash, fn_ptr));
        Ok(hash)
    }

    /// Compile all kernels from a precompile queue (batch BUILD).
    pub fn compile_batch(&mut self, queue: &[ScanParams]) -> Result<Vec<u64>, JitError> {
        queue.iter().map(|p| self.compile(p.clone())).collect()
    }

    /// Compile all palette kernels (bits_per_index 1-8).
    pub fn compile_palette_kernels(&mut self) -> Result<(), JitError> {
        for bits in 1..=8 {
            self.compile(ScanParams {
                threshold: u32::MAX,  // palette unpack doesn't threshold
                top_k: 4096,         // full section
                prefetch_ahead: 4,
                focus_mask: None,
                record_size: bits as u32,
            })?;
        }
        Ok(())
    }

    // ── Phase 2: RUN (frozen, zero-cost) ──────────────────────

    /// Look up a compiled kernel by hash. Zero-cost after freeze.
    /// Returns None if the kernel wasn't compiled during BUILD.
    #[inline(always)]
    pub fn get(&self, hash: u64) -> Option<ScanKernel> {
        // First access to self.cache freezes it via LazyLock::deref()
        // After this, get_mut() returns None — no more compilation possible
        self.cache.map.get(&hash).map(|k| ScanKernel::from_raw(k.fn_ptr, k.params.clone()))
    }

    /// Prefetch the NEXT kernel's code page.
    /// Call while executing kernel N to warm L1 for kernel N+1.
    #[inline(always)]
    pub fn prefetch_next(&self, current_hash: u64) {
        let chain = &self.cache.prefetch_chain;
        if let Some(idx) = chain.iter().position(|(h, _)| *h == current_hash) {
            if let Some((_, next_ptr)) = chain.get(idx + 1) {
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    core::arch::x86_64::_mm_prefetch(
                        *next_ptr as *const i8,
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
        }
    }

    /// Number of compiled kernels.
    pub fn kernel_count(&self) -> usize {
        self.cache.map.len()
    }

    /// Is the cache frozen? (Has any `get()` been called?)
    pub fn is_frozen(&self) -> bool {
        // If get_mut returns None on a &mut self, it's frozen.
        // But we can't call get_mut without &mut self.
        // After first deref, LazyLock is initialized → frozen.
        LazyLock::get(&self.cache).is_some()
    }
}
```

## Usage Pattern

### Pumpkin Server Startup

```rust
fn main() {
    // ── Phase 1: BUILD (during "Loading..." screen) ──
    let mut jit = JitEngine::new().unwrap();
    
    // Palette kernels (7 bit widths)
    jit.compile_palette_kernels().unwrap();
    
    // Noise kernels (per biome)
    for biome in &world.biomes {
        jit.compile(biome.noise_params.to_scan_params()).unwrap();
    }
    
    // Property mask kernels
    jit.compile(waterlogged_mask_params()).unwrap();
    jit.compile(tick_eligible_params()).unwrap();
    
    // Distance threshold kernels
    for radius in [16.0, 32.0, 64.0, 128.0] {
        jit.compile(radius_scan_params(radius)).unwrap();
    }
    
    println!("JIT: {} kernels compiled in BUILD phase", jit.kernel_count());
    // "JIT: 97 kernels compiled in BUILD phase" — took ~50ms
    
    // ── Phase 2: RUN (frozen, zero-cost) ──
    // First .get() call freezes the cache via LazyLock::deref()
    let kernel = jit.get(palette_hash_4bit).unwrap();
    // From here: .compile() would panic. Cache is immutable.
    // Every .get() is a HashMap lookup. No lock. No atomic. No contention.
    
    // Share across threads
    let jit = Arc::new(jit);  // Arc, not Arc<RwLock> — already frozen
    
    // Tick loop — zero-cost kernel access
    loop {
        let kernel = jit.get(current_palette_hash).unwrap();
        unsafe { kernel.scan(query, field, len, size, out) };
        jit.prefetch_next(current_palette_hash);  // warm next kernel
    }
}
```

### q2 Cockpit Replay

```rust
fn start_replay(engine: &mut VizEngine, jit: &mut JitEngine, versions: &[PathBuf]) {
    // ── Phase 1: BUILD — compile all version kernels before play ──
    for (i, version_file) in versions.iter().enumerate() {
        let params = version_scan_params(i, version_file);
        jit.compile(params).unwrap();
    }
    println!("Replay: {} kernels pre-compiled", jit.kernel_count());
    
    // ── Phase 2: RUN — play button starts, zero compilation during playback ──
    for (i, version_file) in versions.iter().enumerate() {
        let hash = version_hash(i);
        let kernel = jit.get(hash).unwrap();  // frozen, instant
        // Process version through thinking graph...
        // No latency spikes. No lock contention. Smooth playback.
    }
}
```

## Why This Matters

```
RwLock cache hit:       ~25ns (atomic read, memory barrier)
LazyLock frozen get:    ~5ns  (plain HashMap::get, no synchronization)

RwLock cache miss:      521µs + write-lock contention
LazyLock cache miss:    panic (compile error if you try after freeze)

100 kernel lookups/tick × 20ns saved = 2µs/tick saved
At 20 TPS = 40µs/second saved on lock overhead alone
```

The real win isn't nanoseconds. It's DETERMINISM. The tick loop never
stalls for compilation. The replay never hiccups. The demo never stutters.
Every kernel that will ever be needed is compiled before the first tick.
If you forgot one, you get a panic at startup, not a lag spike in front
of an audience.

## The Amiga Parallel

Amiga demo coders pre-computed copper lists during the vertical blank
interval. When the display beam reached the visible area, every list
was ready. No computation during rendering. The beam just read addresses.

`LazyLock::get_mut` IS the vertical blank interval:
- BUILD = VBI (compile everything, nobody's watching)
- RUN = visible area (just read function pointers, zero computation)

The freeze is not a limitation. It's a GUARANTEE.

## What NOT to do

- Do NOT keep RwLock as a fallback "just in case"
- Do NOT add a `compile_if_missing()` method that works during RUN
- Do NOT use `OnceCell` or `OnceLock` (they don't have the get_mut → freeze semantic)
- Do NOT make the panic optional — if a kernel is missing during RUN, that's a bug
- SIMD stays on slices. This refactor doesn't touch SIMD paths.
