# ndarray — HPC Expansion for Rust

*Fork of [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray) with 100 HPC modules, 2,534 passing library tests, and SIMD kernels from Intel AMX to Raspberry Pi NEON. Runs on stable Rust 1.99.0 without nightly features.*

<sub>Counts at commit `f2c1aea`: `pub mod` entries in `src/hpc/mod.rs`; `cargo test --lib` (2,534 passed, 32 ignored). How every number on this page was obtained: [Evidence](#evidence-for-the-numbers-on-this-page).</sub>

[Deutsche Version](README-DE.md) | [Full Feature Comparison (146 modules)](COMPARISON.md)

---

## What This Is

The upstream ndarray is a solid library for n-dimensional arrays in Rust. What it does not provide: hardware-aware SIMD acceleration, BLAS without external C libraries, and support for data types like f16 or BF16 that Rust simply does not offer on a stable toolchain.

This fork closes those gaps. The expansion comprises about 205,000 lines of Rust in 424 files that do not exist upstream — from Goto-GEMM microkernels to ARM NEON tier detection to a codec stack that implements cosine similarity as an integer table lookup.

The core trick in one number: a palette similarity is **one table read — about 0.84 ns, ~1.19 billion lookups per second on one core** of a 2.8 GHz Cascade Lake Xeon, with no floating-point arithmetic and no GPU (measured; see [Evidence](#evidence-for-the-numbers-on-this-page)).

---

## The Core Idea: Cosine Similarity Without Floating Point

Vector search in databases like LanceDB or FAISS computes a dot product for every candidate: `dot(a,b) / (|a| * |b|)`. At 768 dimensions, that is 1,536 floating-point operations and 3 KB of candidate data per comparison.

This fork takes a different approach. Vectors are quantized offline to 256 archetypes. The pairwise distances between all archetypes are precomputed into a 256x256 table (`DistanceMatrix`, u16 entries, 128 KB). At query time, a cosine lookup reduces to a single table read.

### Measured

| Operation | Host | Result |
|-----------|------|--------|
| `DistanceMatrix::distance`, random pairs | Xeon @ 2.8 GHz (Cascade Lake class, AVX-512 + VNNI), 1 thread | 0.84 ns, ~1.19 G lookups/s |
| `Base17::l1`, 20,000 candidates | same | 3.04 ns each, 60.7 µs total |

Earlier revisions of this page listed per-platform rates (Sapphire Rapids ~3.2 G/s, i7-11700K 2.4 G/s, Raspberry Pi 4 ~400 M/s, Pi Zero 2W ~80 M/s) and a comparison with FAISS CPU/GPU and cuVS. Those figures have no benchmark in this repository and the FAISS/GPU numbers were not measured here, so they are no longer quoted as results. A like-for-like FAISS comparison would need the same data, recall target and hardware.

---

## Three-Level Cascade: How the Search Actually Works

The palette table alone does not explain how a million vectors are searched quickly. That is the job of a three-level cascade in which each level prunes candidates for the next. Whether a level can lose a relevant result depends on the bound it uses; that guarantee is not yet tested in this repository.

### Level 1: Hamming Sweep over Bitpacked Fingerprints

Each vector is stored as a bitpacked fingerprint. The cascade below assumes 32-byte (256-bit) fingerprints; note that the crate's own `Fingerprint<256>` type is 256 *words* — 2,048 bytes. Comparing two fingerprints is an XOR followed by a popcount:

- **AVX-512 VPOPCNTDQ**: native 64-bit lane popcount where available; otherwise a VPSHUFB lookup + VPSADBW (AVX-512 BW / AVX2)
- **NEON vcntq_u8**: Per-byte popcount, native on every ARM processor

Measured with `bitwise::hamming_batch_raw` on one core of the Cascade Lake host (no VPOPCNTDQ): one query against one million 32-byte fingerprints takes **15.5 ms**; against 2,048-byte `Fingerprint<256>` rows it costs 277 ns per row (~7.4 GB/s). The elimination rate depends on the data and the threshold; it is not measured here.

### Level 2: Base17 L1 Distance

The remaining ~20,000 candidates are refined with 17-dimensional i16 vectors (34 bytes). Measured cost: 3.04 ns per comparison (60.7 µs for 20,000). About 200 candidates survive.

### Level 3: Palette Lookup

The ~200 finalists are scored via the precomputed 256x256 table. One read per candidate, 0.84 ns measured.

### End-to-End: One Million Vectors to Top-K

| Level | In | Out | Duration | Evidence |
|-------|-----|-----|----------|----------|
| Hamming sweep (32 B) | 1,000,000 | data-dependent | 15.5 ms | measured, 1 core, no VPOPCNTDQ |
| Base17 L1 | 20,000 | ~200 | 60.7 µs | measured |
| Palette lookup | 200 | Top-K | ~0.17 µs | 200 × 0.84 ns, derived |

At 32-byte rows the sweep runs at 2.1 GB/s, so per-row overhead, not memory bandwidth, is the limit (2,048-byte rows reach 7.4 GB/s); multi-core scaling is not measured here. An end-to-end comparison with FAISS Flat has not been run in this repository.

### Integration with Lance

The cascade is a substrate path, not a Lance index. In [lance-graph](https://github.com/AdaWorldAPI/lance-graph), bit-vector Hamming distance is exposed as the DataFusion UDF `hamming_distance` (calling `bitwise::hamming_distance_raw`). It is **not** wired into Lance's ANN search, which still uses `lance-linalg` distances and returns an error for a Hamming metric. Nothing here replaces `lance-linalg` inside a Lance scan.

---

## What Upstream Provides and What This Fork Adds

### SIMD Coverage

Upstream ndarray delegates matrix multiplication to the external `matrixmultiply` crate, which can use AVX2. It has no own SIMD types or hardware detection. On ARM, upstream falls back to scalar code.

This fork implements its own SIMD layer: 27 portable vector/mask types selected at compile time (AVX-512, AVX2, NEON, WASM SIMD128, scalar, or nightly `core::simd`), plus runtime-dispatched kernels across 7 tiers (`amx_int8 > avx512vnni > avx512f > avxvnni > avx2_fma > neon > scalar`). Each tier is gated on the instruction feature its kernel needs; the `avxvnni` tier (VEX `VPDPBUSD`) is gated on AVX-VNNI. That tier could not be executed on the measuring host, which has AVX-512 VNNI but not AVX-VNNI, and its kernel is checked by its emitted instruction encoding only.

What the layer buys is measured per operation, against a named baseline, on one core of the Cascade Lake host (median of 15 runs, 1 M elements). Each timing pair lists the fork first and the baseline second:

| Operation | Baseline | Fork | Ratio |
|-----------|----------|------|-------|
| u8 compare → bitmask (`simd::eq_u8_to_mask`) | plain Rust loop | 0.029 vs 0.088 ns/elem | 3.1× |
| f32 → BF16 RNE (`f32_to_bf16_batch_rne`) | scalar per element | 0.196 vs 1.62 ns/elem | 8.3× |
| masked i32 sum (`masked_sum_i32`, 50% density) | plain bit-test loop | 0.53 vs 0.79 ns/elem | 1.5× |
| f32 sum (`F32x16` + `reduce_sum`) | sequential `iter().sum()` | 0.128 vs 1.26 ns/elem | 9.8× |
| int8 GEMM u8×i8→i32 256³ (`gemm_u8_i8`, VNNI) | `int8_gemm_i32` (scalar) | 22.5 vs 5.9 GMAC/s | 3.8× |
| AMX INT8 GEMM 2048³ | scalar | 169.7 GMAC/s | 600× (Emerald Rapids, [`AMX_GOTCHAS.md`](.claude/AMX_GOTCHAS.md)) |

Where plain Rust already autovectorizes, the polyfill matches it rather than beating it: fused multiply-add, chunked f32 sums, 64-bit popcount and `popcount(a&b&c)` all land within ±20% of the plain loop (LLVM emits the same VPSHUFB popcount and VPTERNLOGQ). Instruction width (16 f32 lanes, 64 VNNI MACs) is a ceiling, not a speedup.

Detection happens once via `LazyLock<SimdCaps>`. On this host a repeated `is_x86_feature_detected!` costs ~0.34 ns and a `simd_caps()` copy ~0.62 ns, so the gain of freezing dispatch is predictable dispatch and one decision per process, not a large per-call saving.

### GEMM Performance

Measured on one core of the Cascade Lake host (best of 3–7 runs, `matrixmultiply` threading off):

| Matrix size | `Array::<f32>::dot` | `Array::<f64>::dot` | `simd::gemm_f64_tiled_fma` |
|-------------|--------------------|--------------------|---------------------------|
| 512 × 512 | 70.8 GFLOPS | 34.3 GFLOPS | 9.7 GFLOPS |
| 1024 × 1024 | 70.1 GFLOPS | 34.3 GFLOPS | 9.1 GFLOPS |
| 2048 × 2048 | 65.0 GFLOPS | 32.7 GFLOPS | — |

`Array::dot()` calls `matrixmultiply::sgemm`/`dgemm` (`src/linalg/impl_linalg.rs:503,522`) — the same engine upstream uses, so these columns are not a fork-vs-upstream comparison. The fork-local `gemm_f64_tiled_fma` (fixed `TILE=64`, `F64x8` accumulation) is currently ~3.6× slower than `matrixmultiply` on f64. An earlier table on this page (fork 47/139/~150 GFLOPS vs upstream 13–20, plus NumPy and RTX 3060 columns) had no benchmark in the repository and has been withdrawn.

`simd_ops::array_chunks` walks a slice as non-overlapping `&[T; N]` windows; `array_windows` is the overlapping counterpart (a stable-Rust equivalent of nightly `slice::array_windows::<N>()`). Both pin the window size at the call site so it feeds `F32x16::from_array` / `F64x8::from_array` directly, and both drop the per-element bounds check a dynamically-indexed loop pays. Current in-crate call sites: `hpc::blake3` (64-byte block chunking) and `heel_f64x8::cosine_f32_to_f64_simd`, both via `array_chunks`; `array_windows`, `array_windows_checked`, and `array_chunks_checked` are exported but have no in-crate production caller yet. They are the traversal primitive the hand-rolled BLAS-graph/bgz17 kernels are built on, where the const-generic window landed close to a Cranelift-JIT'd inner loop without paying for a JIT — see `src/simd_ops.rs` module docs.

### Data Types Beyond f32/f64

| Type | Upstream | This Fork | Method |
|------|----------|-----------|--------|
| f16 (IEEE 754) | Not available | Available | u16 carrier + F16C hardware (x86) / FCVTL via inline asm (ARM) |
| BF16 (bfloat16) | Not available | Available | Hardware instructions + RNE emulation (bit-exact with VCVTNEPS2BF16) |
| i8/u8 (quantized) | Not available | Available | VNNI dot, Hamming, popcount |
| i16 (Base17) | Not available | Available | L1 distance with SIMD widen/narrow |

Rust's `f16` type is nightly-only (issue #116909). The fork uses the same approach as AMX: `u16` as carrier, hardware instructions via stable `#[target_feature]` attributes or inline assembler. The result is IEEE 754-compliant conversion at hardware speed on stable Rust.

---

## Seven Things Nobody Else Does on Stable Rust

**1. A std::simd-shaped polyfill on stable.** Rust's portable SIMD API has been nightly-only for years. This fork implements a `std::simd`-style type surface — 27 types including F32x16, F64x8, U8x64, masks, reductions and comparisons — on stable `core::arch`, with a nightly `core::simd` backend behind the `nightly-simd` feature and bit-exact parity crates in CI (`simd-masking-parity`, run under AVX-512, AVX2, NEON via qemu and wasm). It is not the complete `std::simd` API, and a method present on one backend is not guaranteed on all; the parity program is what checks that, and the `F64x8` comparisons were missing on AVX2 and NEON until they were added to it.

**2. f16 without nightly.** Carrier type u16 plus hardware instructions: F16C (VCVTPH2PS/VCVTPS2PH) on x86, FCVTL/FCVTN via asm!() on ARM. Three precision levels: plain f16 (10-bit mantissa), scaled-f16 (range-optimized, 1.5x more precise), double-f16 (hi+lo pair, ~20-bit effective).

**3. AMX on stable Rust.** Intel's Advanced Matrix Extensions (TDPBUSD: a 16×16 tile of outputs over K=64 bytes, 16,384 MACs per instruction) are nightly-only as Rust intrinsics (issue #126622). The fork emits them through `asm!` — all four INT8 forms (`tdpb{ss,su,us,uu}d`), BF16, FP16 and FP8 — and measured 169.7 GMAC/s single-threaded on INT8 2048³ (Emerald Rapids, kernel 6.18.5, [`AMX_GOTCHAS.md`](.claude/AMX_GOTCHAS.md)).

**4. Tiered ARM NEON detection.** Three tiers with runtime detection (the portable NEON types are used on aarch64; the dotprod/BF16 kernel stubs and the `simd_dispatch` table still route to scalar wrappers): A53 baseline (Pi Zero 2W, Pi 3 — single NEON pipeline), A72 fast (Pi 4, Orange Pi 4 — dual pipeline, 2x unrolling), A76 dotprod (Pi 5, Orange Pi 5 — vdotq_s32, native fp16). big.LITTLE systems (RK3399, RK3588) handled correctly.

**5. Frozen dispatch.** The fork detects CPU features once and freezes a function-pointer table (`LazyLock`), so every later call takes the same indirect path. The per-call saving is small on current CPUs — a cached `is_x86_feature_detected!` already costs ~0.34 ns here — the value is one decision per process and a single place that names the selected tier.

**6. BF16 conversion bit-exact with hardware.** The function f32_to_bf16_batch_rne() implements the IEEE 754 RNE algorithm using pure AVX-512-F instructions, matching Intel's VCVTNEPS2BF16 bit-for-bit. Checked against the scalar RNE reference and an independent f64 oracle on **all 4,294,967,296 f32 bit patterns: 0 mismatches** (`cargo run --release --example bf16_rne_exhaustive`, 11.5 s on 4 threads of the Cascade Lake host). The comparison is exact u16 equality, so NaN sign and payload bits count. A unit test additionally compares against the hardware `VCVTNEPS2BF16` on hosts with AVX-512-BF16; the measuring host has none, so that comparison was not run here.

**7. Cognitive codec stack.** Beyond classical numerics, the fork implements a complete encoding pipeline: Fingerprint<256> (VSA, SIMD Hamming), Base17 (17-dimensional i16 vectors), CAM-PQ (product quantization with compiled distance tables), palette semiring (256x256 distance matrices for O(1) lookups), bgz7/bgz17 (compressed model weight format; a 201 GB BF16 → 685 MB conversion was reported for the release artifacts in lance-graph, not reproduced in this repository).

---

## Codebook Inference: Token Generation Without GPU

Beyond vector search, the fork uses the same table approach for LLM inference. Instead of matrix multiplication (`y = W*x`), a precomputed codebook is indexed (`y = codebook[index[x]]`) — O(1) per token. A tokens-per-second table previously shown here (AMX 380,000 tok/s down to Pi 4 500–2,000 tok/s) had no benchmark in the repository and has been withdrawn until it is reproduced.

---

## f16 Weight Transcoding

Measured on one core of the Cascade Lake host (F16C), 15 million Gaussian weights (σ = 0.02):

| Format | Size | Maximum error | RMSE | Throughput |
|--------|------|---------------|------|------------|
| f32 (original) | 60 MB | — | — | — |
| f16 (`cast_f32_to_f16_batch`) | 30 MB | 3.1 × 10⁻⁵ | 4.2 × 10⁻⁶ | 1,805 M params/s |

Error depends on the weight distribution; scaled-f16 and double-f16 are available for tighter error. An earlier table (94/91/42 M params/s) had no benchmark in the repository.

---

## Quick Start

```rust
use ndarray::Array2;
use ndarray::hpc::simd_caps::simd_caps;

let a = Array2::<f32>::ones((1024, 1024));
let c = a.dot(&a);  // matrixmultiply, as upstream

let caps = simd_caps();
if caps.avx512f { println!("AVX-512 active"); }
if caps.neon { println!("ARM profile: {}", caps.arm_profile().name()); }
```

```bash
# Portable / distribution build — x86-64-v3 (AVX2) baseline, runs on any
# Haswell-or-later x86_64. Pass the config EXPLICITLY: since 2026-09-16 the
# default is `target-cpu=native`, which tunes the artifact to the BUILD host
# and is not safe to ship (`.cargo/config-native.toml` says so in as many
# words). Runtime `simd_caps()` detection cannot rescue a binary whose
# baseline codegen already emits host-only instructions.
cargo --config .cargo/config-v3.toml build --release

# Build for THIS machine (dev / benchmarking). Fastest here, portable nowhere.
cargo build --release

# Cross-compile for Raspberry Pi 4
cargo build --release --target aarch64-unknown-linux-gnu

# Maximum performance on AVX-512 server
cargo --config .cargo/config-v4.toml build --release

# Library tests (2,534 at f2c1aea)
cargo test --lib
```

## Requirements

- Rust 1.99.0 stable (pinned in `rust-toolchain.toml`; no nightly, no unstable features)
- Optional: gcc-aarch64-linux-gnu for Pi cross-compilation
- Optional: Intel MKL or OpenBLAS (feature-gated)

### Transitive dependencies of the `std` feature

**None for hashing.** BLAKE3 is in-tree.

The cognitive substrate modules under `hpc/` — `plane`, `seal`,
`merkle_tree`, `vsa`, `spo_bundle`, `crystal_encoder`, `compression_curves`,
`deepnsm` — use `hpc::blake3` for integrity hashing and XOF expansion. That
is a portable pure-Rust transcription of the BLAKE3 reference
implementation, shipped in this crate: no SIMD, no `unsafe`, no C, and no
build script.

Earlier revisions pulled the external **`blake3`** crate here, first gated
behind `hpc-extras` (which caused recurring "missing blake3" build errors
for consumers such as `burn-ndarray` selecting
`default-features = false, features = ["std"]`), then pinned to `std`.
**Both are gone.** `blake3` and its transitive `constant_time_eq`,
`arrayref` and `arrayvec` no longer appear in the dependency graph at any
feature combination, so the footgun cannot recur.

Consumers building `default-features = false` (no `std`, e.g. the
`thumbv6m-none-eabi` nostd target) skip the `hpc` module and the BLAKE3
code with it, so the nostd link is unaffected.

## Evidence for the numbers on this page

| Number | Evidence |
|--------|----------|
| 100 HPC modules, 2,534 lib tests, ~205k added lines / 424 files | counted at `f2c1aea` (`src/hpc/mod.rs`; `cargo test --lib`; path diff against rust-ndarray `bd3ade9`) |
| 0.84 ns palette lookup, 3.04 ns Base17 L1, 15.5 ms 1 M sweep, SIMD ratios, GEMM, f16 | measured on one core of a Xeon @ 2.8 GHz (family 6 model 85, AVX-512 F/BW/VL/DQ/CD + VNNI; no AMX, no AVX-512-BF16, no VPOPCNTDQ), Rust 1.98.1, `target-cpu=native`, median of 15 runs unless stated |
| BF16 RNE, all 2³² inputs, 0 mismatches | `examples/bf16_rne_exhaustive.rs`: every u32 bit pattern in 4 contiguous ranges, 65,536-input batches through the AVX-512F path; exact u16 comparison against `f32_to_bf16_scalar_rne` and an independent f64 nearest-value oracle (quiet-forced NaN, DAZ, ties to even); order-independent output checksum `0x5cd3eaa07f7f8080` (same for 2 and 4 threads); 11.5 s, Rust 1.98.1. A deliberately broken oracle (ties away from zero) reports 32,512 mismatches, so the check can fail |
| AMX 169.7 GMAC/s, 600× scalar | measured on Emerald Rapids, [`AMX_GOTCHAS.md`](.claude/AMX_GOTCHAS.md) |

## Ecosystem

This fork is the hardware foundation for a larger architecture:

| Repository | Purpose |
|------------|---------|
| [lance-graph](https://github.com/AdaWorldAPI/lance-graph) | Cypher/SQL engine on DataFusion, the Quack columnar query surface, codec stack. Owns graph, query and end-to-end benchmarks; this repository owns kernel and microbenchmark numbers |
| [home-automation-rs](https://github.com/AdaWorldAPI/home-automation-rs) | Smart home with voice AI, MCP server, MQTT |

## License

MIT OR Apache-2.0 (identical to upstream)
