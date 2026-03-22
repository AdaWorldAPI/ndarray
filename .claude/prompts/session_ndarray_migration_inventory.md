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

# rustynum current state
cd <rustynum-repo>
find rustynum-core/src -name "*.rs" | wc -l  # Should be 42
find rustynum-rs/src -name "*.rs" | wc -l    # Should be 20
find rustyblas/src -name "*.rs" | wc -l      # Should be 6
cat .claude/SESSION_M_NDARRAY_MIGRATION.md   # Original migration plan
```

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
  src/kernels/          native GEMM microkernels (Goto-style)
```

### Cross-Reference: rustynum-core → ndarray hpc

```
STATUS  rustynum-core FILE           LINES   ndarray hpc FILE          LINES
──────  ─────────────────────        ─────   ──────────────────        ─────
✅ PORT blackboard.rs                  757   blackboard.rs               781
✅ PORT cam_index.rs                   420   cam_index.rs                478
✅ PORT causality.rs                   619   causality.rs                468
✅ PORT dn_tree.rs                     598   dn_tree.rs                  739
✅ PORT fingerprint.rs                 407   fingerprint.rs              394
✅ PORT kernels.rs                    1589   kernels.rs                 1589
✅ PORT node.rs                        358   node.rs                     312
✅ PORT organic.rs                     905   organic.rs                  783
✅ PORT packed.rs                      466   packed.rs                   355
✅ PORT plane.rs                       840   plane.rs                    758
✅ PORT prefilter.rs                   403   prefilter.rs                448
✅ PORT qualia_gate.rs                 303   qualia_gate.rs              328
✅ PORT seal.rs                        122   seal.rs                      99
✅ PORT substrate.rs                   933   substrate.rs                933

❌ MISS bf16_hamming.rs               1510   → bf16_truth.rs?            680   DIVERGED: ndarray has bf16_truth, not bf16_hamming
❌ MISS compute.rs                     265   → (none)
❌ MISS delta.rs                       237   → (none)
❌ MISS graph_hv.rs                    840   → graph.rs?                 282   SMALLER: ndarray version may be subset
❌ MISS hdr.rs                         556   → cascade.rs?               758   DIVERGED: ndarray version is larger
❌ MISS hybrid.rs                     2032   → (none)                          2K lines, biggest gap
❌ MISS jit_scan.rs                    316   → (none)
❌ MISS jitson.rs                     1620   → (none)                          Separate crate in rustynum
❌ MISS layer_stack.rs                 328   → (none)
❌ MISS layout.rs                       75   → (none)
❌ MISS mkl_ffi.rs                     472   → (none)                          MKL-specific, may not need
❌ MISS parallel.rs                    109   → (none)
❌ MISS qualia_cam.rs                  501   → qualia.rs?                613   DIVERGED: different names
❌ MISS rng.rs                         117   → (none)                          Inline SplitMix64 in node.rs?
❌ MISS scalar_fns.rs                  302   → (none)
❌ MISS simd.rs                       1092   → bitwise.rs?               639   DIVERGED: ndarray dispatches differently
❌ MISS simd_avx2.rs                   600   → (none)                          Inlined into bitwise.rs?
❌ MISS simd_avx512.rs                2643   → (none)                          Inlined into bitwise.rs?
❌ MISS simd_isa.rs                    215   → (none)
❌ MISS soaking.rs                     407   → (none)                          Arrow soaking? In arrow_bridge.rs?
❌ MISS spatial_resonance.rs           758   → (none)
❌ MISS tail_backend.rs                884   → (none)
```

### Cross-Reference: rustynum-rs → ndarray hpc

```
STATUS  rustynum-rs FILE             LINES   ndarray hpc FILE          LINES
──────  ─────────────────────        ─────   ──────────────────        ─────
✅ PORT binding_matrix.rs              373   binding_matrix.rs           416
✅ PORT bitwise.rs                     471   bitwise.rs                  639
✅ PORT cogrecord.rs                   511   cogrecord.rs                238   SMALLER: ndarray stripped?
✅ PORT graph.rs                       407   graph.rs                    282   SMALLER
✅ PORT hdc.rs                        1553   hdc.rs                      178   MUCH SMALLER: 1553→178
✅ PORT projection.rs                  296   projection.rs               143   SMALLER
✅ PORT statistics.rs                  865   statistics.rs               325   SMALLER

❌ MISS array_struct.rs               2203   → ndarray IS the array          Not needed
❌ MISS constructors.rs                223   → ndarray has these              Not needed
❌ MISS impl_clone_from.rs             101   → ndarray handles                Not needed
❌ MISS linalg.rs                      263   → ndarray-linalg crate          External dep
❌ MISS manipulation.rs                562   → ndarray has reshape/etc        Partially not needed
❌ MISS operations.rs                  833   → ndarray ops                    Partially not needed
❌ MISS view.rs                        747   → ndarray ArrayView              Not needed
```

### Cross-Reference: rustyblas → ndarray hpc

```
STATUS  rustyblas FILE               LINES   ndarray hpc FILE          LINES
──────  ─────────────────────        ─────   ──────────────────        ─────
⚠️ PART level1.rs                     578   blas_level1.rs              278   SMALLER: only basics?
⚠️ PART level2.rs                    1521   blas_level2.rs              321   MUCH SMALLER: 1521→321
⚠️ PART level3.rs                    1942   blas_level3.rs              345   MUCH SMALLER: 1942→345
❌ MISS bf16_gemm.rs                   536   → quantized.rs?             416   DIVERGED
❌ MISS int8_gemm.rs                   940   → quantized.rs?             416   DIVERGED
```

### Only in ndarray (NEW — not from rustynum)

```
FILE                       LINES   PURPOSE
──────────────────         ─────   ───────
activations.rs                86   sigmoid, relu, softmax
arrow_bridge.rs              931   ThreePlaneFingerprintBuffer (from rustynum-arrow)
bnn_causal_trajectory.rs    2116   BNN + causality combined (NEW composition)
bnn_cross_plane.rs          1631   Cross-plane BNN ops (NEW)
clam_compress.rs             707   CLAM tree compression (from rustynum-clam)
clam_search.rs               612   CLAM search ops (from rustynum-clam)
compression_curves.rs       1733   Compression analysis (NEW)
crystal_encoder.rs           883   Crystal encoding pipeline (NEW)
cyclic_bundle.rs             741   Cyclic VSA bundling (NEW)
deepnsm.rs                   845   DeepNSM integration (NEW)
fft.rs                       209   FFT (partial, from rustymkl?)
lapack.rs                    310   LU/Cholesky/QR (partial)
merkle_tree.rs               521   Merkle tree (from rustynum-core seal?)
nars.rs                      747   NARS reasoning (from rustynum-oracle)
qualia.rs                    613   Qualia layer (from qualia_xor?)
spo_bundle.rs               1514   SPO bundling (NEW composition)
surround_metadata.rs        1283   Surround metadata encoding (NEW)
tekamolo.rs                  502   TEKAMOLO grammar (NEW)
udf_kernels.rs               789   User-defined kernels (NEW)
vml.rs                       154   Vector math library (partial)
vsa.rs                       727   VSA operations (NEW)
```

## DELIVERABLE: Migration Inventory Document

Produce `MIGRATION_INVENTORY.md` with these sections:

### 1. Reconciliation Table

For EVERY ✅ PORT file: compare pub fn signatures between rustynum and ndarray.
Flag any functions that exist in rustynum but are missing from the ndarray port.
Flag any functions where the signature diverged (different types, different args).

```bash
# For each ported file, extract pub fn signatures from both repos
diff <(grep "pub fn" <rustynum-file> | sort) <(grep "pub fn" <ndarray-file> | sort)
```

### 2. Gap Priority Matrix

For each ❌ MISS file, classify:

```
PRIORITY  CATEGORY          RATIONALE
──────    ────────          ─────────
P0        Must port         Used by lance-graph or bgz17, blocking integration
P1        Should port       Useful for production, has tests in rustynum
P2        Nice to have      Research/experimental, can defer
DROP      Not needed        Replaced by ndarray native functionality
```

Specific questions to answer per file:
- Is it imported by any other rustynum crate? (`grep -r "use rustynum_core::$module"`)
- Is it referenced by lance-graph? (`grep -r "$module" lance-graph/`)
- Does ndarray have native equivalent? (ndarray docs)
- Does it have tests? How many pass?

### 3. Size Gap Analysis

Several ported files are MUCH SMALLER in ndarray:

```
hdc.rs:        1553 → 178 lines (89% smaller)
cogrecord.rs:   511 → 238 lines (53% smaller)
statistics.rs:  865 → 325 lines (62% smaller)
blas_level2.rs: 1521 → 321 lines (79% smaller)
blas_level3.rs: 1942 → 345 lines (82% smaller)
```

For each: are the missing lines (a) dropped intentionally because ndarray
handles them, (b) deferred for later porting, or (c) accidentally lost?
Check by comparing `pub fn` signatures.

### 4. Divergence Map

Files where names or structure changed:

```
rustynum                    ndarray                  QUESTION
──────                      ──────                   ────────
bf16_hamming.rs (1510)  →   bf16_truth.rs (680)      Same content? What's missing?
hdr.rs (556)            →   cascade.rs (758)         ndarray version is LARGER — what was added?
qualia_cam.rs (501)     →   qualia.rs (613)          Renamed + extended?
simd.rs (1092)          →   bitwise.rs (639)         Different dispatch architecture?
simd_avx2.rs (600)      →   (inlined?)               Where did the AVX2 kernels go?
simd_avx512.rs (2643)   →   (inlined?)               Where did the AVX-512 kernels go?
graph_hv.rs (840)       →   graph.rs (282)           What was dropped?
```

### 5. Tier 2-3 Crate Status

```
CRATE              LINES   ndarray COVERAGE                    PRIORITY
─────              ─────   ───────────────                    ────────
rustynum-bnn       5917    bnn.rs (942) + bnn_causal (2116)   Partial
                           + bnn_cross_plane (1631)
rustynum-clam      5869    clam.rs (2593) + compress (707)    Partial
                           + search (612)
rustynum-arrow     5010    arrow_bridge.rs (931)              Partial
qualia_xor         5201    qualia.rs (613)                    Partial
rustynum-holo      8821    (none)                             Not started
rustynum-oracle   11337    nars.rs (747) + organic.rs (783)   Partial
rustymkl           3547    (none — MKL-specific)              Maybe DROP
jitson              748    (none)                             Low priority
```

### 6. bgz17 / lance-graph Integration Gaps

What does lance-graph need from ndarray that isn't ported yet?

```bash
# Check what lance-graph imports from ndarray (if any)
grep -r "ndarray\|hpc::" <lance-graph-repo>/crates/ | grep -v test | grep -v target
# Check what bgz17 would need
grep -r "Fingerprint\|Plane\|Node\|Cascade\|CLAM" <lance-graph-repo>/crates/bgz17/
```

Specific needs for Session C (ndarray ← bgz17 dual-path):
- `NdarrayFingerprint ↔ Base17` conversion (ndarray_bridge.rs)
- CLAM tree build with palette distance
- Cascade benefit from container W112 (stride-16)
- TruthGate integration (reads container W4-7)

### 7. Blackboard Update Draft

Produce an updated blackboard section reflecting actual state:
- Which files are TRULY done (pub fn parity with rustynum)
- Which are partial (ported but missing functions)
- Which are new (not from rustynum, created fresh in ndarray)
- Updated test count
- Updated line count

### 8. Action Plan

Ordered list of what to port next, based on:
1. lance-graph/bgz17 integration blockers (P0)
2. Function parity gaps in already-ported files (P1)
3. Missing files needed by downstream crates (P1)
4. Everything else by line count / complexity (P2)

## OUTPUT

```
<ndarray-repo>/.claude/MIGRATION_INVENTORY.md  — the full inventory
<ndarray-repo>/.claude/blackboard.md           — updated with accurate state
```

Do NOT modify any Rust code. This is a READ-ONLY audit session.
Run `grep`, `diff`, `wc -l`. No `cargo`, no edits.
