# SESSION: Rustynum → ndarray Migration Inventory

## Mission

Produce a precise, file-by-file inventory of what's been ported, what's missing,
what diverged, and what should be dropped. The ndarray blackboard (Epoch 4) claims
core types are done but the actual `hpc/` directory has 51 files / 35K lines that
go far beyond what the blackboard tracks. Rustynum has evolved since the migration
started. This session reconciles reality.

## READ FIRST

```bash
# ndarray current state
cd <ndarray-repo>
cat .claude/blackboard.md                    # Claims vs reality
find src/hpc -name "*.rs" | wc -l            # Should be 51
find src/hpc -name "*.rs" -exec wc -l {} +   # Should be ~35K total
cat src/hpc/mod.rs                           # What's actually wired

# The SIMD architecture (NOT debt — properly decomposed)
cat src/backend/mod.rs                       # BlasFloat trait + dispatch
cat src/backend/native.rs                    # Goto GEMM tiling + dispatch! macro
cat src/backend/kernels_avx512.rs            # 962 lines: ALL AVX-512 intrinsics
cat src/hpc/bitwise.rs                       # Hamming dispatch: VPOPCNTDQ→AVX-512BW→AVX2→scalar

# rustynum current state
cd <rustynum-repo>
find rustynum-core/src -name "*.rs" | wc -l  # Should be 42
find rustynum-rs/src -name "*.rs" | wc -l    # Should be 20
find rustyblas/src -name "*.rs" | wc -l      # Should be 6
cat .claude/SESSION_M_NDARRAY_MIGRATION.md   # Original migration plan

# All existing migration prompts in rustynum (most may be done)
ls <rustynum-repo>/.claude/prompts/
```

## PRE-EXISTING KNOWLEDGE: What Is NOT Debt

The BLAS files shrank because ndarray DECOMPOSED them, not because it dropped them.
Verify this but do NOT flag these as debt without checking the backend:

```
rustynum (monolithic):        ndarray (decomposed):
  rustyblas/level1.rs  578      hpc/blas_level1.rs (trait)       278
  rustyblas/level2.rs 1521      hpc/blas_level2.rs (trait)       321
  rustyblas/level3.rs 1942      hpc/blas_level3.rs (trait)       345
                                backend/mod.rs (dispatch)        165
                                backend/native.rs (Goto tiling)  747
                                backend/kernels_avx512.rs        962
  ─────────────────────         ──────────────────────────────
  4,041 lines                   2,818 lines

The traits are thin: blas_level3.rs calls A::backend_gemm() via BlasFloat.
backend/native.rs does Goto-style 6×16 f32 / 6×8 f64 microkernels with
dispatch! macro routing to AVX-512 → AVX2 → scalar.
kernels_avx512.rs has: dot, axpy, scal, nrm2, asum, iamax,
  16 element-wise ops (add/sub/mul/div × scalar/vec × f32/f64),
  sgemm_blocked, dgemm_blocked, hamming_distance, popcount, dot_i8, hamming_batch.

SIMD for Hamming lives in bitwise.rs:
  dispatch_hamming: VPOPCNTDQ → AVX-512BW → AVX2 → scalar
  dispatch_popcount: same tier chain
  hamming_batch_raw: query-vs-database batch
  hamming_top_k: top-k by Hamming

This is the CORRECT architecture. The monolithic rustynum SIMD files
(simd.rs 1092, simd_avx2.rs 600, simd_avx512.rs 2643 = 4,335 lines)
are properly replaced by the decomposed backend/ + bitwise.rs.

HOWEVER: rustynum's simd_avx512.rs (2643 lines) is NOT just raw intrinsics.
It's a std::simd COMPATIBILITY LAYER — portable SIMD types backed by
stable core::arch intrinsics:

    pub struct F32x16(pub __m512);    // mimics std::simd::f32x16
    pub struct F64x8(pub __m512d);
    pub struct U8x64(pub __m512i);
    pub struct I32x16(pub __m512i);
    ... (11 types total)

    impl Add, Sub, Mul for F32x16    // operator overloading
    impl SimdFloat for F32x16        // reduce_sum, reduce_max, etc.

ndarray skipped this layer and calls _mm512_* directly in every kernel.
This works today but has consequences:

  - Every kernel is x86_64-only. Aarch64/NEON needs SEPARATE kernel files.
  - When std::simd stabilizes, every kernel must be rewritten.
  - Adding new operations requires raw intrinsics knowledge, not SIMD math.

The compat layer costs zero runtime (everything inlines to the same
instructions). It costs 2643 lines of compile-time boilerplate.
It saves: write-once kernels for all architectures, clean std::simd
migration path, simpler kernel authoring.

RECOMMENDATION: Port the compat layer as P1. Put it in
    src/backend/simd_compat.rs
Wire kernels_avx512.rs to use F32x16 etc. instead of raw __m512.
Add #[cfg(target_arch = "aarch64")] backing using NEON intrinsics later.
Kernels become architecture-portable without rewriting.
```

## KNOWN REAL DEBT (files that actually lost functionality)

```
FILE                RUSTYNUM  NDARRAY  LOSS   WHY IT'S DEBT
──────────          ────────  ───────  ────   ─────────────
hdc.rs              1553      178      89%    bind/permute/bundle/simhash MISSING
                                              These are VSA core ops.
                                              bgz17 Base17::xor_bind needs these.
                                              NOT handled by ndarray natively.

cogrecord.rs        511       238      53%    CogRecord is domain-specific.
                                              4-channel struct, hamming_4ch, sweep.
                                              No ndarray equivalent.

statistics.rs       865       325      62%    median, var, std, percentile with
                                              axis variants MISSING.
                                              ndarray has sum/mean only.

graph_hv.rs         840       282      66%    VerbCodebook, encode_edge, decode_target,
(→ graph.rs)                                  causality_asymmetry, find_non_causal_edges.
                                              Verify what was kept vs dropped.

bf16_hamming.rs     1510      680*     55%    *Renamed to bf16_truth.rs.
(→ bf16_truth.rs)                             Verify: are BF16 Hamming kernels present
                                              or only truth encoding?

bf16_gemm.rs        536       416*     22%    *Possibly in quantized.rs.
(→ quantized.rs?)                             VERIFY: does quantized.rs contain
                                              actual bf16_gemm_f32, mixed_precision_gemm?

int8_gemm.rs        940       416*     56%    *Possibly in quantized.rs.
                                              VERIFY: quantize_f32_to_u8/i8/i4,
                                              int8_gemm_i32/f32, per_channel variants?
```

## KNOWN REAL GAPS (files not ported at all)

```
FILE                    LINES   PRI   WHAT IT ACTUALLY IS
──────────────          ─────   ───   ──────────────────
hybrid.rs               2032    P1    3-stage scoring pipeline: K0 probe (64-bit, reject 55%)
                                      → K1 stats (512-bit, reject 90%) → K2 exact → BF16 tail.
                                      Bridges kernels.rs + bf16_hamming.rs + awareness substrate.
                                      THE hot-path orchestrator for the Cascade.

tail_backend.rs          884    P1    TailBackend trait (libCEED pattern): trait boundary between
                                      safe orchestration (hybrid.rs) and unsafe SIMD/FFI backends.
                                      PopcntBackend, XsmmBackend, FallbackBackend.
                                      Check overlap with backend/native.rs dispatch.

soaking.rs               407    P1    Int8 10000D transient accumulation layer.
                                      dot_i8_10k, binary_to_int8, int8_to_binary (crystallize),
                                      AttentionMask (σ-2/3 focus lens with project/classify).
                                      Check overlap with arrow_bridge.rs SoakingBuffer.

layer_stack.rs           328    P1    Collapse gate (Luftschleuse): Flow/Hold/Block airlock
                                      between superposition (delta layers) and ground truth.
                                      Multi-writer concurrent state without mutation.

delta.rs                 237    P1    XOR delta layer: borrow-free holographic overlay.
                                      effective = ground_truth XOR delta. No RefCell, no UnsafeCell.
                                      XOR's self-inverse property handles isolation algebraically.
                                      layer_stack.rs depends on this.

spatial_resonance.rs     758    P2    BF16 3D spatial resonance (Crystal4K axis model).
                                      Three orthogonal BF16 projections (X/Y/Z) with
                                      sign/exp/man decomposition per axis. SPO grammar.

compute.rs               265    P2    Tiered compute dispatch: INT8 VNNI → BF16 → FP32 → scalar.
                                      Check overlap with backend/mod.rs Tier enum + dispatch! macro.

scalar_fns.rs            302    P2    Scalar fallback for every SIMD op (dot, axpy, scal, etc.).
                                      Check if backend/native.rs scalar paths cover these.
                                      If fully covered, DROP.

jitson.rs               1620    P2    Cranelift JIT — DUAL PURPOSE:
                                      (a) Scan: param baking as immediates. Partially obsoleted by
                                          Rust 1.94 array_windows (const-generic → autovectorize).
                                      (b) Graph-to-native: compile graph topology → flat instruction
                                          stream for rs-graph-llm LangGraph port. THE real future.
                                      Depends on AdaWorldAPI/wasmtime fork (AVX-512).

jit_scan.rs              316    P2    Hybrid JIT scan (Cranelift outer loop + SIMD inner kernel).
                                      Companion to jitson.rs. For scan path, use array_windows
                                      first. JIT adds value for graph orchestration.

mkl_ffi.rs               472    DROP  Replaced by backend/mkl.rs (237 lines).
rng.rs                   117    DROP  Inline SplitMix64 already in node.rs.
parallel.rs              109    DROP  ndarray has par_azip, rayon integration.
layout.rs                 75    DROP  ndarray handles memory layout natively.
```

## bgz17 INTEGRATION REQUIREMENTS

### NaN Guard for Empty Palette Clusters

When bgz17 runs k-means with k=256 on a scope with <256 unique patterns,
some clusters end up empty. Empty centroid = mean of nothing = NaN.
NaN propagates through the 256×256 distance matrix. Any tree (CLAM,
archetype tree) built on NaN distances produces garbage at the root.

**The fix belongs in bgz17's palette.rs, not in ndarray.** But ndarray's
CLAM implementation must also guard against NaN inputs:

```rust
// bgz17/palette.rs: after k-means convergence
// Remove empty clusters, compact codebook, adjust k
let non_empty: Vec<_> = centroids.iter()
    .enumerate()
    .filter(|(_, c)| c.member_count > 0)
    .collect();
let effective_k = non_empty.len();
// Renumber assignments, rebuild distance matrix at effective_k × effective_k
// NO NaN possible: every centroid has at least one member

// ndarray/hpc/clam.rs: guard at tree build
assert!(!distances.iter().any(|d| d.is_nan()),
    "CLAM tree build received NaN distance — palette has empty clusters");
```

### SIMD Palette Distance for ndarray

bgz17's `batch_palette_distance` (AVX-512 VGATHERDPS for 16 palette
lookups per instruction) should integrate with ndarray's tier dispatch:

```rust
// ndarray/src/backend/kernels_avx512.rs — add:
pub fn batch_palette_distance_avx512(
    distance_matrix: &[u16],  // 256×256 flat
    query_idx: u8,
    candidate_indices: &[u8],
    distances_out: &mut [u16],
) { ... }

// ndarray/src/hpc/bitwise.rs — add dispatch:
pub fn batch_palette_distance(dm: &[u16], query: u8, candidates: &[u8], out: &mut [u16]) {
    // Same tier dispatch as hamming: VGATHERDPS → scalar
}
```

This is Session C's DELIVERABLE 8 (SIMD dispatch). The ndarray inventory
should identify WHERE in the backend this should be wired.

### HDC / VSA Operations (the hdc.rs debt)

bgz17's Base17::xor_bind, bundle, permute map directly to rustynum's
hdc.rs: bind, bundle, permute. ndarray's hdc.rs (178 lines) is 89% smaller.
The missing functions are:

```
NEEDED BY bgz17          RUSTYNUM hdc.rs              ndarray hdc.rs
──────────────            ───────────────              ──────────────
Base17::xor_bind          bind(a, b) → XOR             ???
Base17::bundle            bundle(vecs) → majority      ???
Base17::permute           permute(v, shift) → rotate   ???
compose_table build       bind + nearest               ???
PaletteSemiring           bind + distance              ???
```

The audit must verify: which of these 3 operations exist in ndarray's
178-line hdc.rs, and which are missing?

## DEMAND SIDE: Pumpkin (Minecraft Rust) SIMD Wishlist

A separate session identified 10 ndarray features that would drop Pumpkin's
server tick from 1.5 CPU to 0.2 CPU. Cross-referenced against existing code:

```
#   FEATURE                  WHAT EXISTS                        GAP
──  ───────────────────────  ─────────────────────────────────  ──────────────
1   simd_map (lane-native)   (nothing)                          NEW API
    arr.simd_map::<f32x16>   NEEDS: compat layer (F32x16 type)  BLOCKED ON
    contractual vectorize    from rustynum simd_avx512.rs       compat layer

2   SpatialArray3 (CAM)      cam_index.rs (478 lines) is LSH    DIFFERENT THING
    O(1) spatial insert      CAM, not spatial array. cam_index   NEW TYPE needed
    region() → SIMD slice    is hash-based, not coordinate.     for 3D chunks

3   xor_diff (change detect) merkle_tree.rs has xor_diff()      PARTIAL
    simd_xor_diff::<u64x8>   delta.rs in rustynum (NOT ported)  delta.rs = P1
    nonzero_iter via mask    is the XOR overlay algebra.         port delta.rs
                             NEEDS: _mm512_test_epi64_mask      + SIMD sparse iter

4   gather_scatter           kernels_avx512.rs HAS VPGATHERDD   NOT EXPOSED
    VPGATHERDD / VGATHERDPS  in sgemm_blocked (internal use).   needs user-facing
    for permutation tables   bgz17 needs this too for palette.  API on Array

5   Arrow columnar_view      arrow_bridge.rs (931 lines) has    PARTIAL
    zero-copy RecordBatch    ThreePlaneFingerprintBuffer,        missing: generic
    → ArrayView              SoakingBuffer, PlaneBuffer.         columnar_view()
                             Not zero-copy into ArrayView yet.   for arbitrary cols

6   Zip::simd_apply          (nothing)                          NEW API
    multi-array fused SIMD   NEEDS: compat layer for portable   BLOCKED ON
    kernel over N arrays     SIMD types in the Zip combinator   compat layer

7   runtime_dispatch         backend/native.rs HAS Tier enum    EXISTS internally
    Array-level dispatch     + LazyLock<Tier> detection.         NOT exposed as
    .with_dispatch(Auto)     dispatch! macro routes to tiers.    user-facing API

8   stencil (neighbor SIMD)  (nothing)                          NEW API
    VonNeumann3D / Moore3D   Common in HPC (structured grids).  needs Array3
    64 blocks per AVX-512    Would use prefetch + u8x64.        stencil iterator

9   compact_palette          bgz17 palette.rs IS this           EXISTS in bgz17
    bit-packed SIMD          for 8-bit palette indices.          NOT in ndarray
    VPMOVZX + VPSHUFB        PaletteEdge, distance_matrix.      needs Array wrapper
    unpack/repack            Minecraft uses 4-15 bit palette.    for variable bits

10  prefetch + stream_store  packed.rs has stroke-aligned        PARTIAL
    _mm_prefetch, VMOVNTPS   layout for prefetch-friendly scan.  not user-facing
    memory hierarchy ctrl    bgz17 prefetch.rs has _mm_prefetch. needs Array API
```

### What This Reveals About Priorities

The compat layer (rustynum simd_avx512.rs → `src/backend/simd_compat.rs`)
is the FOUNDATION for items 1, 3, 4, 6, 8. Without portable F32x16/U8x64
types, none of the user-facing SIMD APIs can be implemented portably.
This reinforces simd_compat as P1 — it unblocks both bgz17 integration
AND the Pumpkin feature set.

Items that already have internal implementations but lack user-facing API:
  4 (gather — in kernels_avx512.rs), 7 (dispatch — in backend/native.rs),
  10 (prefetch — in packed.rs + bgz17 prefetch.rs).
These need thin wrapper traits on ArrayBase, not new kernels.

Items that connect to existing bgz17 work:
  9 (compact_palette — bgz17 palette.rs IS the palette codec)
  4 (gather — bgz17 batch_palette_distance needs VGATHERDPS)
  3 (xor_diff — delta.rs XOR overlay, same algebra)

Items that are genuinely new:
  2 (SpatialArray3 — new type for coordinate-indexed 3D arrays)
  8 (stencil — new iteration pattern for structured grids)

## CURRENT STATE (as of March 2026)

### Repos

```
rustynum (AdaWorldAPI/rustynum):
  rustynum-core/src/   42 files   ~22K lines   Kernels, SIMD, cognitive types
  rustynum-rs/src/     20 files   ~12K lines   NumArray container, ops, statistics
  rustyblas/src/        6 files   ~5.6K lines  BLAS L1/L2/L3, BF16/Int8 GEMM
  rustynum-bnn/src/     6 files   ~5.9K lines  BNN inference + search
  rustynum-clam/src/    6 files   ~5.9K lines  CLAM tree, search, compress
  rustynum-arrow/src/   9 files   ~5K lines    Arrow bridge, soaking buffer
  qualia_xor/src/       4 files   ~5.2K lines  Qualia XOR layer
  rustynum-holo/src/    9 files   ~8.8K lines  Wave substrate (experimental)
  rustynum-oracle/src/ 17 files  ~11.3K lines  Oracle layer (experimental)
  rustymkl/src/         4 files   ~3.5K lines  MKL FFI bindings
  jitson/src/           5 files   ~748 lines   JIT JSON parsing

ndarray (AdaWorldAPI/ndarray, branch: master):
  src/hpc/             51 files   ~35K lines   Ported cognitive + BLAS + extensions
  src/backend/          5 files   ~2.3K lines  SIMD dispatch + Goto GEMM + AVX-512 kernels
```

### Cross-Reference: rustynum-core → ndarray

```
STATUS  rustynum-core FILE           LINES   ndarray FILE              LINES   NOTES
──────  ─────────────────────        ─────   ──────────────────        ─────   ─────
✅      blackboard.rs                  757   hpc/blackboard.rs           781
✅      cam_index.rs                   420   hpc/cam_index.rs            478
✅      causality.rs                   619   hpc/causality.rs            468
✅      dn_tree.rs                     598   hpc/dn_tree.rs              739
✅      fingerprint.rs                 407   hpc/fingerprint.rs          394
✅      kernels.rs                    1589   hpc/kernels.rs             1589
✅      node.rs                        358   hpc/node.rs                 312
✅      organic.rs                     905   hpc/organic.rs              783
✅      packed.rs                      466   hpc/packed.rs               355
✅      plane.rs                       840   hpc/plane.rs                758
✅      prefilter.rs                   403   hpc/prefilter.rs            448
✅      qualia_gate.rs                 303   hpc/qualia_gate.rs          328
✅      seal.rs                        122   hpc/seal.rs                  99
✅      substrate.rs                   933   hpc/substrate.rs            933

⚠️      bf16_hamming.rs               1510   hpc/bf16_truth.rs           680   RENAMED + SHRUNK
⚠️      hdr.rs                         556   hpc/cascade.rs              758   RENAMED + GREW
⚠️      qualia_cam.rs                  501   hpc/qualia.rs               613   RENAMED + GREW
⚠️      graph_hv.rs                    840   hpc/graph.rs                282   SHRUNK 66%

❌      hybrid.rs                     2032   (none)                              P1: 3-stage pipeline (K0/K1/K2 → BF16 tail)
❌      spatial_resonance.rs           758   (none)                              P2: BF16 3D axis model (Crystal4K)
❌      tail_backend.rs                884   (none)                              P1: TailBackend trait (libCEED pattern)
                                                                                Check overlap with backend/native.rs
❌      soaking.rs                     407   (none)                              P1: int8 10KD accumulation + crystallization
                                                                                Check overlap with arrow_bridge.rs
❌      layer_stack.rs                 328   (none)                              P1: collapse gate (Flow/Hold/Block)
❌      delta.rs                       237   (none)                              P1: XOR delta layer (borrow-free overlay)
❌      compute.rs                     265   (none)                              P2: tiered compute dispatch (INT8→BF16→FP32)
                                                                                Check overlap with backend/mod.rs Tier
❌      jitson.rs                     1620   (none)                              P2: Cranelift JIT (graph-to-native for rs-graph-llm)
❌      jit_scan.rs                    316   (none)                              P2: hybrid JIT scan (companion to jitson)
❌      scalar_fns.rs                  302   (none)                              Check vs backend/native.rs scalar paths
❌      mkl_ffi.rs                     472   (none)                              DROP: replaced by backend/mkl.rs (237 lines)
❌      rng.rs                         117   (none)                              DROP: inline SplitMix64 already in node.rs
❌      parallel.rs                    109   (none)                              DROP: ndarray has par_azip, rayon integration
❌      layout.rs                       75   (none)                              DROP: ndarray handles memory layout

SIMD: DECOMPOSED (not missing), but COMPAT LAYER not ported:
        simd.rs (1092)         → backend/native.rs dispatch! macro    747
        simd_avx2.rs (600)     → (AVX2 paths in native.rs fallback)
        simd_avx512.rs (2643)  → backend/kernels_avx512.rs           962
                                  ↑ MISSING: compat layer types (F32x16 etc.)
                                    kernels use raw __m512 instead
                                    Port as src/backend/simd_compat.rs (P1)
        simd_isa.rs (215)      → backend/mod.rs Tier enum             165
        simd_compat.rs (4)     → (not needed)
```

### Cross-Reference: rustynum-rs → ndarray

```
STATUS  rustynum-rs FILE             LINES   ndarray FILE              LINES   NOTES
──────  ─────────────────────        ─────   ──────────────────        ─────   ─────
✅      binding_matrix.rs              373   hpc/binding_matrix.rs       416
✅      bitwise.rs                     471   hpc/bitwise.rs              639   GREW (+ dispatch)
⚠️      cogrecord.rs                   511   hpc/cogrecord.rs            238   DEBT: 53% smaller
⚠️      graph.rs                       407   hpc/graph.rs                282   DEBT: 31% smaller
⚠️      hdc.rs                        1553   hpc/hdc.rs                  178   DEBT: 89% smaller
⚠️      projection.rs                  296   hpc/projection.rs           143   DEBT: 52% smaller
⚠️      statistics.rs                  865   hpc/statistics.rs           325   DEBT: 62% smaller

DROP    array_struct.rs               2203   ndarray IS the container
DROP    constructors.rs                223   ndarray has these
DROP    impl_clone_from.rs             101   ndarray handles
DROP    linalg.rs                      263   ndarray-linalg crate
DROP    manipulation.rs                562   ndarray native (partial)
DROP    operations.rs                  833   ndarray ops (partial)
DROP    view.rs                        747   ndarray ArrayView
```

### Cross-Reference: rustyblas → ndarray

```
STATUS  rustyblas FILE               LINES   ndarray FILE              LINES   NOTES
──────  ─────────────────────        ─────   ──────────────────        ─────   ─────
✅      level1.rs                      578   hpc/blas_level1.rs          278   TRAIT only
                                             backend/kernels_avx512.rs         + 12 AVX-512 fns
✅      level2.rs                     1521   hpc/blas_level2.rs          321   TRAIT + fallback
                                             backend/native.rs                 + gemv dispatch
✅      level3.rs                     1942   hpc/blas_level3.rs          345   TRAIT + fallback
                                             backend/native.rs                 + Goto GEMM
                                             backend/kernels_avx512.rs         + sgemm_blocked
⚠️      bf16_gemm.rs                   536   hpc/quantized.rs?           416   VERIFY contents
⚠️      int8_gemm.rs                   940   hpc/quantized.rs?           416   VERIFY contents
```

### Only in ndarray (NEW — not from rustynum)

```
FILE                       LINES   SOURCE
──────────────────         ─────   ──────
activations.rs                86   New
arrow_bridge.rs              931   From rustynum-arrow (partial)
bnn_causal_trajectory.rs    2116   New composition
bnn_cross_plane.rs          1631   New composition
clam_compress.rs             707   From rustynum-clam
clam_search.rs               612   From rustynum-clam
compression_curves.rs       1733   New
crystal_encoder.rs           883   New
cyclic_bundle.rs             741   New
deepnsm.rs                   845   New
fft.rs                       209   Partial (from rustymkl?)
lapack.rs                    310   Partial
merkle_tree.rs               521   Extended from seal.rs
nars.rs                      747   From rustynum-oracle
qualia.rs                    613   From qualia_xor
quantized.rs                 416   From rustyblas bf16/int8 (VERIFY)
spo_bundle.rs               1514   New composition
surround_metadata.rs        1283   New
tekamolo.rs                  502   New
udf_kernels.rs               789   New
vml.rs                       154   Partial
vsa.rs                       727   New
```

## DELIVERABLE: Migration Inventory Document

Produce `MIGRATION_INVENTORY.md` with these sections:

### 1. Reconciliation Table

For EVERY ✅ and ⚠️ file: compare pub fn signatures between rustynum and ndarray.

```bash
diff <(grep "pub fn" <rustynum-file> | sort) <(grep "pub fn" <ndarray-file> | sort)
```

Flag functions that exist in rustynum but are missing from ndarray port.
Flag functions where signatures diverged (different types, different args).

### 2. DEBT Quantification

For each ⚠️ SHRUNK file, produce an exact missing-function list:

```
FILE            RUSTYNUM pub fns    NDARRAY pub fns     MISSING
──────          ────────────────    ───────────────     ───────
hdc.rs          bind                ???                 ???
                permute             ???                 ???
                bundle              ???                 ???
                ...
```

Repeat for cogrecord.rs, statistics.rs, graph.rs (→graph_hv.rs),
projection.rs, bf16_truth.rs (→bf16_hamming.rs).

### 3. Backend SIMD Verification

Confirm that the BLAS decomposition is complete:

```bash
# Every pub fn in rustyblas/level1.rs should map to either:
# (a) a BlasFloat method in backend/mod.rs, OR
# (b) a function in kernels_avx512.rs, OR
# (c) a trait method in blas_level1.rs
# Any fn in (a) that is NOT in (b) means AVX-512 acceleration is missing.

grep "pub fn" <rustynum>/rustyblas/src/level1.rs | sort
grep "pub fn" <ndarray>/src/backend/kernels_avx512.rs | grep -v "gemm\|hamming" | sort
```

Repeat for L2 and L3.

Also audit the compat layer gap:

```bash
# What types does rustynum's compat layer define?
grep "pub struct" <rustynum>/rustynum-core/src/simd_avx512.rs

# What traits does it implement?
grep "impl.*for F32x16\|impl.*for F64x8\|impl.*for U8x64\|trait Simd" \
     <rustynum>/rustynum-core/src/simd_avx512.rs | head -20

# How many operator impls?
grep "impl.*Add\|impl.*Sub\|impl.*Mul\|impl.*Div\|impl.*BitXor\|impl.*BitAnd\|impl.*BitOr" \
     <rustynum>/rustynum-core/src/simd_avx512.rs | wc -l

# Verify kernels_avx512.rs could be rewritten with compat types:
# Count raw __m512/__m512d/__m512i usage
grep -c "__m512\|__m256\|__mmask" <ndarray>/src/backend/kernels_avx512.rs
```

### 4. Quantized GEMM Verification

The most likely debt. Check whether `quantized.rs` (416 lines) actually contains:

```
FROM rustyblas/bf16_gemm.rs (536 lines):
  bf16_gemm_f32()           — BF16 input, F32 accumulation
  mixed_precision_gemm()    — mixed BF16/F32
  BF16 type + conversions

FROM rustyblas/int8_gemm.rs (940 lines):
  quantize_f32_to_u8()
  quantize_f32_to_i8()
  quantize_f32_to_i4()
  int8_gemm_i32()
  int8_gemm_f32()
  per_channel_quantize()
  per_channel_dequantize()
  int8_gemm_per_channel()
```

If `quantized.rs` doesn't have these, that's ~1,476 lines of real debt
in the most performance-critical path (quantized inference).

### 5. NaN Guard Audit

Check ALL division and mean operations across ndarray hpc/ for NaN risk:

```bash
grep -n "/ \|\.div\|/ n\|/ len\|/ count\|mean()\|\.avg" src/hpc/*.rs | grep -v test | grep -v "//"
```

For each: is there a guard for n=0 / empty input?

Specific files to check:
- `clam.rs` — LFD computation (parent_radius / child_radius)
- `statistics.rs` — mean, var, std (divide by n)
- `bf16_truth.rs` — truth normalization
- `cascade.rs` — sigma computation (divide by count)

bgz17-specific: palette k-means with k > unique_patterns produces
empty clusters → NaN centroids → NaN distances → NaN at tree root.
Check if ndarray CLAM guards against NaN distance inputs.

### 6. Existing Prompt Status

Rustynum has 21 prompts in `.claude/prompts/`. Many may already be done.
For each prompt, check:

```bash
for f in <rustynum>/.claude/prompts/*.md; do
    echo "=== $(basename $f) ==="
    grep -i "file\|crate\|module\|output\|branch" "$f" | head -5
done
```

Classify each as: DONE, PARTIAL, NOT STARTED, SUPERSEDED, N/A.

### 7. bgz17 Integration Gaps

What does lance-graph Session C need from ndarray that isn't there?

```
NEEDED FOR SESSION C         WHERE IN NDARRAY          STATUS
────────────────────         ────────────────          ──────
NdarrayFingerprint↔Base17    hpc/fingerprint.rs        Fingerprint exists, Base17 bridge NOT
CLAM with palette distance   hpc/clam.rs               CLAM exists, palette distance fn NOT
batch_palette_distance SIMD  backend/kernels_avx512.rs AVX-512 kernels exist, palette NOT
TruthGate on containers      hpc/bf16_truth.rs         Truth exists, container read NOT
Cascade stride-16 benefit    hpc/cascade.rs            Cascade exists, container NOT
HDC bind/bundle/permute      hpc/hdc.rs                178 lines — VERIFY contents
```

### 8. Blackboard Update Draft

Produce updated blackboard reflecting actual state:
- Which files are TRULY done (pub fn parity with rustynum)
- Which are partial (ported but missing functions)
- Which are new (not from rustynum, created fresh in ndarray)
- Updated test count and line count
- SIMD backend correctly documented: decomposed (not missing),
  but compat layer (F32x16 etc.) flagged as P1 port target

### 9. Action Plan

Ordered list based on:
1. **P0 — bgz17 blockers:** hdc.rs debt (bind/permute/bundle), NaN guards,
   palette distance in kernels_avx512.rs
2. **P1 — SIMD compat layer:** port rustynum simd_avx512.rs type system
   (F32x16, F64x8, U8x64, I32x16, SimdFloat trait) into
   `src/backend/simd_compat.rs`. Refactor kernels_avx512.rs to use
   compat types instead of raw `__m512`. Zero runtime cost. Unlocks:
   aarch64/NEON support, std::simd migration, simpler kernel authoring.
   ALSO UNBLOCKS Pumpkin items 1,3,4,6,8 (simd_map, xor_diff, gather,
   Zip::simd_apply, stencil) — all need portable SIMD types.
3. **P1 — hot-path pipeline:** hybrid.rs (2032 lines, K0/K1/K2 → BF16 tail),
   tail_backend.rs (884 lines, TailBackend trait)
4. **P1 — superposition algebra:** delta.rs (237 lines, XOR overlay) +
   layer_stack.rs (328 lines, collapse gate Flow/Hold/Block).
   delta.rs first — layer_stack depends on it.
5. **P1 — soaking layer:** soaking.rs (407 lines, int8 10KD accumulation).
   Check arrow_bridge.rs overlap first.
6. **P1 — function parity:** missing pub fns in ⚠️ files
   (hdc.rs, statistics.rs, cogrecord.rs, graph.rs, projection.rs)
7. **P1 — quantized GEMM:** verify quantized.rs covers bf16_gemm + int8_gemm
8. **P2 — spatial/compute:** spatial_resonance.rs, compute.rs, scalar_fns.rs
9. **P2 — JIT infrastructure:** jitson.rs + jit_scan.rs (defer until
   rs-graph-llm LangGraph port needs graph-to-native compilation)
10. **P2 — Pumpkin user-facing APIs** (after compat layer lands):
    - `Array::simd_gather()` — expose VPGATHERDD already in kernels_avx512.rs
    - `Array::runtime_dispatch()` — expose Tier enum already in backend/native.rs
    - `Array::prefetch_region()` — expose _mm_prefetch already in packed.rs/bgz17
    These are thin wrappers on existing internals, not new kernels.
11. **P3 — Pumpkin new types** (significant design work):
    - `SpatialArray3<T>` — coordinate-indexed 3D array (not CAM hash)
    - `Array3::stencil()` — Von Neumann/Moore neighbor iterator + SIMD
    - `PaletteArray<T, BITS>` — variable-width bit-packed SIMD unpack/repack
12. **DROP — confirmed unnecessary:** mkl_ffi, rng, parallel, layout

## OUTPUT

```
<ndarray-repo>/.claude/MIGRATION_INVENTORY.md  — the full inventory
<ndarray-repo>/.claude/blackboard.md           — updated with accurate state
```

Do NOT modify any Rust code. This is a READ-ONLY audit session.
Run `grep`, `diff`, `wc -l`. No `cargo`, no edits.
