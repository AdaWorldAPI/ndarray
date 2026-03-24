# JITSON Migration — rustynum → ndarray (URGENT)

## FIRST: Read this completely before touching any code

jitson is a JSON-to-native-code JIT compiler stuck in AdaWorldAPI/rustynum
which is being retired. If rustynum dies, jitson dies. Nobody else has this.

JSON config values become CPU immediates:
- `threshold: 500` → `CMP reg, 500` (not LOAD + CMP)
- `focus_mask: [47, 193]` → VPANDQ bitmask as immediate data
- `prefetch_ahead: 4` → `PREFETCHT0 [ptr + 4 * RECORD_SIZE]`

Cold compile: 521µs. Cache hit: 455ns. Engine init: 47µs.

## Source files (in AdaWorldAPI/rustynum)

```
rustynum-core/src/jitson.rs     — no_std JSON parser, bracket recovery, schema validator,
                                   AVX-512 instruction→feature mapping, WAL precompile queue,
                                   prefetch addressing, JitsonTemplate conversion
                                   
rustynum-core/src/jit_scan.rs   — ScanConfig, SIMD kernel registry (hamming/cosine/dot),
                                   C-ABI trampolines, scan_hamming fallback, jit_symbol_table()
                                   
rustynum-core/src/packed.rs     — PackedDatabase stroke-aligned layout, 3-stroke cascade,
                                   90% rejection per stroke, 11.3x bandwidth reduction

jitson/src/ir.rs                — ScanParams, PhilosopherIR, CollapseParams, RecipeIR,
                                   CollapseBias (Flow/Hold), VotingStrategy
                                   
jitson/src/engine.rs            — JitEngine, JitEngineBuilder, Cranelift JIT module,
                                   kernel cache by params hash, CPU feature detection
                                   
jitson/src/scan_jit.rs          — build_scan_ir() Cranelift codegen (the actual IR builder),
                                   ScanKernel native fn pointer wrapper
                                   
jitson/src/detect.rs            — CpuCaps (AVX2/AVX-512F/BW/VL/VPOPCNTDQ/FMA/BMI2)
```

## Target structure in ndarray

```
src/hpc/jitson/
  mod.rs          ← re-exports
  parser.rs       ← from jitson.rs (no_std JSON parser + bracket recovery)
  validator.rs    ← from jitson.rs (schema validation + instruction→feature map)
  template.rs     ← from jitson.rs (JitsonTemplate, from_json(), template_hash)
  precompile.rs   ← from jitson.rs (PrecompileQueue, WAL, prefetch addressing)
  scan_config.rs  ← from jit_scan.rs (ScanConfig, SimdKernelRegistry, trampolines)
  packed.rs       ← from packed.rs (PackedDatabase, stroke layout, cascade)

src/hpc/jitson_cranelift/  ← ONLY compiled with feature "jit-native"
  mod.rs
  ir.rs           ← ScanParams, PhilosopherIR, RecipeIR
  engine.rs       ← JitEngine, JitEngineBuilder, Cranelift module
  scan_jit.rs     ← build_scan_ir(), ScanKernel
  detect.rs       ← CpuCaps → Cranelift ISA
```

## Cargo.toml additions

```toml
[features]
jitson = []
jit-native = ["jitson", "dep:cranelift-codegen", "dep:cranelift-frontend",
              "dep:cranelift-jit", "dep:cranelift-module", "dep:target-lexicon"]

[dependencies]
cranelift-codegen = { git = "https://github.com/AdaWorldAPI/wasmtime.git", branch = "main", optional = true }
cranelift-frontend = { git = "https://github.com/AdaWorldAPI/wasmtime.git", branch = "main", optional = true }
cranelift-jit = { git = "https://github.com/AdaWorldAPI/wasmtime.git", branch = "main", optional = true }
cranelift-module = { git = "https://github.com/AdaWorldAPI/wasmtime.git", branch = "main", optional = true }
target-lexicon = { version = "0.13", optional = true }
```

## Rust 1.94 unlocks to apply during migration

- `array_windows::<N>()` — SIMD read path: `&[u8; 128]` windows over packed stroke data
- `element_offset` — pointer offset math in PackedDatabase without manual arithmetic
- `LazyLock::get_mut` — JitEngine kernel cache: mutable during build, immutable during compute
- `f32::mul_add` const — compile-time threshold constants with FMA precision

## Rules

- SIMD stays on SLICES. Never copy data for SIMD. PackedDatabase is contiguous for the prefetcher.
- The no_std JSON parser MUST remain no_std. It works in embedded/WASM.
- Cranelift is ALWAYS behind `jit-native` feature gate. Never mandatory.
- The trampolines call ndarray's own SIMD functions (not rustynum's). Rewire:
  `crate::simd::hamming_distance` not `rustynum_core::simd::hamming_distance`
- Tests must pass: scan_correctness, kernel_caching, compile_latency, hybrid_scan

## What NOT to do

- Do NOT rewrite the Cranelift IR builder (scan_jit.rs). It works. Copy it.
- Do NOT change the PackedDatabase stroke sizes (128B / 384B / 1536B). They're tuned.
- Do NOT remove the WAL precompile queue. It enables prefetch addressing.
- Do NOT make Cranelift a hard dependency. Feature-gate it.
