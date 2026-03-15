---
name: port-rustynum
description: >
  Full autonomous porting of rustynum HPC features into ndarray.
  Never stops until every function exists with ndarray-native syntax.
  Write scope limited to this repo only.
allowed-tools: Read, Edit, Write, Bash, Glob, Grep, Task
---

# MISSION: Port rustynum → ndarray (Complete, Autonomous, Zero-Gap)

You are the ORCHESTRATOR for Project NDARRAY Expansion.

## PRIME DIRECTIVE
Port **every public function** from `adaworldapi/rustynum` into this ndarray fork.
**Do not stop until every function listed in the inventory below has a working ndarray equivalent.**
Same functionality. Different syntax — ndarray's `ArrayBase<S, D>` generics instead of rustynum's `NumArray<T, Ops>`.
More mature, more idiomatic, zero-cost.

## SCOPE LOCK
- **WRITE ONLY** to files inside this repository (`adaworldapi/ndarray`)
- **READ ONLY** from rustynum (clone to `/tmp/rustynum-ref` for reference)
- Never modify rustynum. Never push to rustynum. It is the source of truth.

## ENVIRONMENT
- Rust 1.94 Stable (no nightly)
- ndarray's existing trait system is the foundation — extend, don't replace
- Feature gates: `native` (default), `intel-mkl`, `openblas` — mutually exclusive

---

## EXECUTION TREE

### STAGE 0: SETUP (do this first, do it once)

```
0.1  Clone rustynum reference:
     cd /tmp && git clone https://github.com/AdaWorldAPI/rustynum rustynum-ref

0.2  Read .claude/blackboard.md — this is your state surface

0.3  Audit current ndarray src/ — what already exists?
     grep -rn "pub fn\|pub trait\|pub struct" src/ | sort

0.4  Diff: what rustynum has vs what ndarray has → write gap list to blackboard
```

### STAGE 1: FOUNDATION — Backend Trait + Feature Gates
**Agent: savant-architect**

```
1.1  Create src/backend/mod.rs with LinalgBackend trait:
     - gemm(alpha, a, b, beta, c)
     - dot(a, b) -> T
     - axpy(alpha, x, y)
     - scal(alpha, x)
     - nrm2(x) -> T
     - asum(x) -> T
     All generic over f32/f64.

1.2  Create src/backend/native.rs — pure Rust + SIMD impl
     Port from: rustynum-core/src/simd.rs (dot_f32, dot_f64, axpy_f32, etc.)

1.3  Cargo.toml feature gates:
     [features]
     default = ["native"]
     native = []
     intel-mkl = ["dep:intel-mkl-sys"]
     openblas = ["dep:openblas-sys"]

     Add compile_error! for mkl+openblas

1.4  CHECKPOINT: `cargo check --features native` must pass
```

### STAGE 2: BLAS LEVEL 1 — Vector Operations
**Agent: savant-architect**
Port from: `rustyblas/src/level1.rs` + `rustynum-core/src/simd.rs`

```
EVERY one of these must exist in ndarray:

 ┌─────────────────────────────────────────────────────┐
 │ BLAS L1 (rustyblas/src/level1.rs)                   │
 ├─────────────────────────────────────────────────────┤
 │ sdot / ddot          → dot product                  │
 │ saxpy / daxpy        → y = α·x + y                 │
 │ sscal / dscal        → x = α·x                     │
 │ snrm2 / dnrm2        → L2 norm                     │
 │ sasum / dasum         → L1 norm (abs sum)           │
 │ isamax / idamax       → index of max abs element    │
 │ scopy / dcopy         → vector copy                 │
 │ sswap / dswap         → vector swap                 │
 ├─────────────────────────────────────────────────────┤
 │ SIMD kernels (rustynum-core/src/simd.rs)            │
 ├─────────────────────────────────────────────────────┤
 │ add_f32_scalar / add_f64_scalar                     │
 │ sub_f32_scalar / sub_f64_scalar                     │
 │ mul_f32_scalar / mul_f64_scalar                     │
 │ div_f32_scalar / div_f64_scalar                     │
 │ add_f32_vec / add_f64_vec                           │
 │ sub_f32_vec / sub_f64_vec                           │
 │ mul_f32_vec / mul_f64_vec                           │
 │ div_f32_vec / div_f64_vec                           │
 └─────────────────────────────────────────────────────┘

2.1  Implement each as ndarray extension trait on ArrayBase
2.2  SIMD dispatch: AVX-512 → AVX2 → SSE4.2 → scalar
2.3  CHECKPOINT: unit test per function, `cargo test` passes
```

### STAGE 3: BLAS LEVEL 2 — Matrix-Vector
**Agent: savant-architect**
Port from: `rustyblas/src/level2.rs`

```
 ┌─────────────────────────────────────────────────────┐
 │ BLAS L2 (rustyblas/src/level2.rs)                   │
 ├─────────────────────────────────────────────────────┤
 │ sgemv / dgemv        → y = α·A·x + β·y             │
 │ sger / dger          → A = α·x·yᵀ + A (rank-1)     │
 │ ssymv / dsymv        → symmetric matrix-vector      │
 │ strmv / dtrmv        → triangular matrix-vector     │
 │ strsv / dtrsv        → triangular solve             │
 └─────────────────────────────────────────────────────┘

3.1  Implement on ArrayBase<S, Ix2> / ArrayBase<S, Ix1>
3.2  Handle row-major and column-major via .is_standard_layout()
3.3  CHECKPOINT: unit tests with known BLAS reference values
```

### STAGE 4: BLAS LEVEL 3 — Matrix-Matrix
**Agent: savant-architect**
Port from: `rustyblas/src/level3.rs` + `rustynum-rs/src/num_array/linalg.rs`

```
 ┌─────────────────────────────────────────────────────┐
 │ BLAS L3 (rustyblas/src/level3.rs)                   │
 ├─────────────────────────────────────────────────────┤
 │ sgemm / dgemm        → C = α·A·B + β·C             │
 │ ssyrk / dsyrk        → C = α·A·Aᵀ + β·C            │
 │ strsm                → triangular solve (matrix)    │
 │ ssymm / dsymm        → symmetric matrix multiply    │
 ├─────────────────────────────────────────────────────┤
 │ Mixed precision (rustyblas/src/bf16_gemm.rs)        │
 ├─────────────────────────────────────────────────────┤
 │ BF16 type + conversions                             │
 │ bf16_gemm_f32        → BF16 GEMM with f32 accum    │
 │ mixed_precision_gemm → f32 input, BF16 compute      │
 ├─────────────────────────────────────────────────────┤
 │ Int8 quantized (rustyblas/src/int8_gemm.rs)         │
 ├─────────────────────────────────────────────────────┤
 │ quantize_f32_to_u8 / quantize_f32_to_i8            │
 │ quantize_per_channel_i8                             │
 │ int8_gemm_i32 / int8_gemm_f32                      │
 │ int8_gemm_per_channel_f32                           │
 │ quantize_f32_to_i4 / dequantize_i4_to_f32          │
 ├─────────────────────────────────────────────────────┤
 │ Linalg (rustynum-rs/src/num_array/linalg.rs)       │
 ├─────────────────────────────────────────────────────┤
 │ matrix_vector_multiply                              │
 │ matrix_matrix_multiply                              │
 │ matrix_multiply (dispatch)                          │
 └─────────────────────────────────────────────────────┘

4.1  GEMM with tiled micro-kernel (L1=32K, L2=256K, L3=shared)
4.2  BF16 and Int8 behind feature gates
4.3  CHECKPOINT: benchmark against naive — must show >10x for 1024×1024
```

### STAGE 5: MKL + LAPACK
**Agent: savant-architect**
Port from: `rustymkl/src/`

```
 ┌─────────────────────────────────────────────────────┐
 │ MKL/LAPACK (rustymkl/src/lapack.rs)                 │
 ├─────────────────────────────────────────────────────┤
 │ sgetrf / dgetrf      → LU factorization             │
 │ sgetrs / dgetrs      → solve via LU                 │
 │ spotrf / dpotrf      → Cholesky factorization       │
 │ spotrs               → solve via Cholesky           │
 │ sgeqrf / dgeqrf      → QR factorization             │
 ├─────────────────────────────────────────────────────┤
 │ FFT (rustymkl/src/fft.rs)                           │
 ├─────────────────────────────────────────────────────┤
 │ fft_f32 / fft_f64    → forward FFT                  │
 │ ifft_f32 / ifft_f64  → inverse FFT                  │
 │ rfft_f32             → real-to-complex FFT           │
 ├─────────────────────────────────────────────────────┤
 │ VML (rustymkl/src/vml.rs)                           │
 ├─────────────────────────────────────────────────────┤
 │ vsexp/vdexp, vsln/vdln, vssqrt/vdsqrt              │
 │ vsabs/vdabs, vsadd, vsmul, vsdiv                   │
 │ vssin, vscos, vspow                                 │
 └─────────────────────────────────────────────────────┘

5.1  All behind feature = "intel-mkl"
5.2  FFI bindings: exact C signature match
5.3  Native fallback for every MKL function (scalar Rust impl)
5.4  CHECKPOINT: `cargo test --features intel-mkl` passes (or feature-gated skip)
```

### STAGE 6: ARRAY OPERATIONS — NumArray → ArrayBase
**Agent: product-engineer**
Port from: `rustynum-rs/src/num_array/`

```
 ┌─────────────────────────────────────────────────────┐
 │ Constructors (constructors.rs)                      │
 ├─────────────────────────────────────────────────────┤
 │ zeros / ones          → Array::zeros / ones (exist) │
 │ arange / linspace     → Array::range / linspace     │
 ├─────────────────────────────────────────────────────┤
 │ Statistics (statistics.rs)                           │
 ├─────────────────────────────────────────────────────┤
 │ mean / mean_axis                                    │
 │ median / median_axis                                │
 │ var / var_axis                                      │
 │ std / std_axis                                      │
 │ percentile / percentile_axis                        │
 │ sort                                                │
 ├─────────────────────────────────────────────────────┤
 │ Operations (array_struct.rs)                        │
 ├─────────────────────────────────────────────────────┤
 │ dot                   → already in ndarray          │
 │ min / max / min_axis / max_axis                     │
 │ argmin / argmax                                     │
 │ top_k                                               │
 │ cumsum                                              │
 │ exp / log / sigmoid / softmax / log_softmax         │
 │ cosine_similarity                                   │
 │ norm(p, axis, keepdims)                             │
 ├─────────────────────────────────────────────────────┤
 │ Manipulation (manipulation.rs)                      │
 ├─────────────────────────────────────────────────────┤
 │ transpose / reshape / flip_axis                     │
 │ squeeze / slice / concatenate                       │
 ├─────────────────────────────────────────────────────┤
 │ Broadcast ops (operations.rs)                       │
 ├─────────────────────────────────────────────────────┤
 │ try_add_broadcast / try_sub_broadcast               │
 │ try_mul_broadcast / try_div_broadcast               │
 ├─────────────────────────────────────────────────────┤
 │ Views (view.rs)                                     │
 ├─────────────────────────────────────────────────────┤
 │ ArrayView / ArrayViewMut (ndarray already has these)│
 │ Verify: t, swap_axes, slice_axis, flip_axis,        │
 │         reshape, to_vec, iter all map correctly     │
 └─────────────────────────────────────────────────────┘

6.1  Extension traits on ArrayBase for anything ndarray doesn't have
6.2  Check each function — if ndarray already has it, document the mapping
6.3  CHECKPOINT: integration tests matching rustynum's test suite
```

### STAGE 7: HDC + BINARY OPERATIONS
**Agent: vector-synthesis**
Port from: `rustynum-rs/src/num_array/hdc.rs` + `bitwise.rs` + `projection.rs`

```
 ┌─────────────────────────────────────────────────────┐
 │ HDC / Hyperdimensional Computing (hdc.rs)           │
 ├─────────────────────────────────────────────────────┤
 │ bind / permute / bundle / bundle_byte_slices        │
 │ dot_i8                                              │
 ├─────────────────────────────────────────────────────┤
 │ Bitwise (bitwise.rs)                                │
 ├─────────────────────────────────────────────────────┤
 │ hamming_distance / popcount / hamming_distance_batch │
 ├─────────────────────────────────────────────────────┤
 │ Projection (projection.rs)                          │
 ├─────────────────────────────────────────────────────┤
 │ simhash_project / simhash_batch_project             │
 │ simhash_int8_project                                │
 ├─────────────────────────────────────────────────────┤
 │ SIMD binary ops (rustynum-core/src/simd.rs)         │
 ├─────────────────────────────────────────────────────┤
 │ hamming_distance / hamming_batch / hamming_top_k    │
 │ popcount / dot_i8                                   │
 │ hdr_cascade_search                                  │
 └─────────────────────────────────────────────────────┘

7.1  ndarray extension traits for binary/HDC ops on Array<u8, _>
7.2  SIMD accelerated: AVX-512 VPOPCNTDQ where available
7.3  CHECKPOINT: hamming_top_k benchmark, bit-exact match with rustynum
```

### STAGE 8: COGRECORD + GRAPH
**Agent: vector-synthesis**
Port from: `rustynum-rs/src/num_array/cogrecord.rs` + `graph.rs`

```
 ┌─────────────────────────────────────────────────────┐
 │ CogRecord (cogrecord.rs)                            │
 ├─────────────────────────────────────────────────────┤
 │ CogRecord struct (4-channel: meta, cam, btree, emb) │
 │ new / zeros / container / hamming_4ch               │
 │ sweep_adaptive / hdr_sweep                          │
 │ to_bytes / from_bytes / from_borrowed               │
 │ sweep_cogrecords (batch)                            │
 ├─────────────────────────────────────────────────────┤
 │ VerbCodebook / Graph ops (graph.rs)                 │
 ├─────────────────────────────────────────────────────┤
 │ VerbCodebook: default_codebook, new, offset, verbs  │
 │ encode_edge / decode_target / causality_asymmetry   │
 │ causality_check / find_non_causal_edges / infer_verb│
 └─────────────────────────────────────────────────────┘

8.1  CogRecord backed by Array<u8, Ix1> × 4 channels
8.2  VerbCodebook using ndarray for XOR binding
8.3  CHECKPOINT: round-trip: encode → decode must be exact
```

### STAGE 9: QA SWEEP
**Agent: sentinel-qa**

```
9.1  Audit every unsafe block in the codebase
9.2  Verify every SAFETY comment
9.3  Feature gate soundness: test all combinations
9.4  Run full test suite: cargo test --all-features
9.5  Clippy clean: cargo clippy -- -D warnings
9.6  CHECKPOINT: zero BLOCK findings
```

### STAGE 10: DOCUMENTATION + PUBLISH READINESS
**Agent: product-engineer**

```
10.1  Every pub fn has /// doc comments with examples
10.2  cargo doc --no-deps builds clean
10.3  README.md with feature matrix and usage examples
10.4  Benchmark suite: cargo bench (criterion)
10.5  CHECKPOINT: ready to publish
```

---

## AUTO-ATTENDANT RULES

1. **Never stop between stages.** When a stage checkpoint passes, immediately proceed to the next.
2. **If a checkpoint fails**, fix the issue and re-check. Do not skip.
3. **Delegate to agents** for their domains — don't try to do everything in the main context.
4. **Update .claude/blackboard.md** at the end of each stage with:
   - Functions ported (with checkmark)
   - Functions remaining
   - Any BLOCK findings from sentinel-qa
5. **If you hit a compile error**, debug it. Don't leave broken code and move on.
6. **If you're unsure about a design decision**, check how rustynum did it, then adapt to ndarray idioms.

## COMPLETION CRITERIA

The task is complete when:
- [ ] Every function in the inventory above has an ndarray equivalent
- [ ] `cargo test` passes with zero failures
- [ ] `cargo clippy -- -D warnings` is clean
- [ ] `cargo doc --no-deps` builds without warnings
- [ ] .claude/blackboard.md shows all functions checked off
- [ ] sentinel-qa has audited all unsafe blocks with PASS

**Begin at Stage 0. Do not stop until Stage 10 is complete.**
