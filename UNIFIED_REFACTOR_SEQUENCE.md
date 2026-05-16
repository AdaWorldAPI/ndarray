# Unified Refactor Sequence: ndarray Fork

> Integrates four analysis passes into one executable sequence:
> 1. **REFACTOR_HPC_INTEGRATION.md** — Type bridges & extension traits
> 2. **SOA_KERNEL_ARCHITECTURE.md** — Columnar cascade & field-separated storage
> 3. **Transformer session feedback** — API conventions, namespace, codegen macros
> 4. **lance-graph W1a consumer contract** — 5 SIMD primitives blocking 158-violation remediation
>
> Each wave is self-contained. Later waves depend on earlier ones.
> Within a wave, items are independent and can run in parallel.

---

## Wave 0 — Conventions & Foundations (3 days)

Unlocks everything else. No code changes to hot paths. Pure contract + tooling.

| ID | Item | Source | Effort | Why First |
|----|------|--------|--------|-----------|
| W0.1 | **Dual-form signature convention** (`_into` + Vec wrapper + `_ptr`) | R1.1 | 1d | Fixes root cause of PP-15 WS-1/2/4 ❌. All future kernels follow this shape. |
| W0.2 | **Unified HpcError enum** (`src/hpc/error.rs`) | R1.2 | 4h | Replaces 4 error conventions with 1. |
| W0.3 | **`#![deny(warnings)]` → targeted denies** | R1.3 | 1h | Unblocks doctest authoring for all subsequent waves. |
| W0.4 | **Feature flag rename** (`hpc-extras` → `research`, `backend-*` prefix) | R4.1 | 4h | Clean slate for feature-gated additions. |
| W0.5 | **Prelude module** (`hpc::prelude::*`) | R1.4 | 1h | Single discovery surface for downstream consumers. |

**Gate**: All downstream crates (burn, candle, tract, ort, lance-graph) still compile unchanged.
The dual-form is additive; existing call sites continue working.

---

## Wave 1 — SIMD Consumer Primitives (3 days) ← lance-graph P0

From `lance-graph` W1a consumer contract. **Five primitives blocking 158 raw-intrinsic
violations across 5 lance-graph crates.** Each is a tight-scope PR. Parallel review.

The lance-graph `simd-savant` agent runs PRE-MERGE against each PR.

| ID | Item | Source | Effort | Consumer Site |
|----|------|--------|--------|---------------|
| W1.1 | **`I8x16::from_i4_packed_u64()`** + `batch_packed_i4_16()` | W1a-#1 | 1d | `lance-graph-contract/src/mul.rs::i4_eval::batch` — 5 batch fns over `QualiaI4_16D(u64)`. Closes PR #398 codex P1 (NEON OOB at `len==2`). |
| W1.2 | **`I8x16::saturating_abs()`** + `I8x32::saturating_abs()` | W1a-#2 | 4h | `lance-graph-contract/src/mul.rs` — Direction-B fix. `abs(i8::MIN)` must return `i8::MAX`, not wrap. See §VPABSB correction. |
| W1.3 | **`U16x8::gather_u16()`** + `palette_lookup_u8x8()` | W1a-#3 | 4h | `bgz17/src/simd.rs:88` — currently inlines `_mm256_i32gather_epi32` (AP-SIMD-1 violation). |
| W1.4 | **`prefetch_read_t0/t1/t2()`** | W1a-#4 | 2h | `bgz17/src/prefetch.rs:96,100` — currently inlines `_mm_prefetch` directly. |
| W1.5 | **`U64x8::popcnt()`** + `U64x8::xor_popcount()` | W1a-#5 | 4h | `holograph/hamming.rs` + `lance-graph/graph/blasgraph/types.rs` — inline `_mm512_popcnt_epi64`. |

### Per-primitive implementation contract

Every PR MUST:
1. **Three backends**: AVX-512, NEON, scalar. Missing scalar = P0 reject.
2. **Edge-case semantics documented** in doc-comment (`i8::MIN`, empty slices, OOB indices).
3. **Parity test**: all backends produce identical output on randomized corpus including edge cases.
4. **Bench against scalar**: record AVX-512/NEON speedup ratios in PR body.
5. **`// SAFETY:` on every `unsafe` block**.
6. **No new `is_x86_feature_detected!`** outside `simd_caps.rs`.
7. **Consumer site cited** in PR description.

### W1.1 — I8x16::from_i4_packed_u64 (nibble unpack + sign extend)

```rust
impl I8x16 {
    /// Unpack 16 signed i4 nibbles from a u64 into 16 i8 lanes (sign-extended).
    /// Nibble layout: lane[i] = sign_extend_4((packed >> (4*i)) & 0xf).
    pub fn from_i4_packed_u64(packed: u64) -> Self;

    /// Const-folded lane extract.
    pub fn lane_i8<const N: usize>(self) -> i8;
}

/// Closure-parameterized batch: run `f` over each (unpacked_i8x16, aux[i]) pair.
/// Bounds-aware tail handling; scalar fallback on unsupported arch.
pub fn batch_packed_i4_16<E, F>(packed: &[u64], aux: &[i8], out: &mut [E], f: F)
where F: Fn(I8x16, i8) -> E + Sync + Send, E: Copy;
```

**AVX-512**: `_mm_cvtsi64_si128` + nibble shuffle via VPSHUFB mask LUT, then sign-extend.
Alt: PDEP (`_pdep_u64` × 2) into two u64 halves → load → `vpmovsxbw`. Bench both on Zen4 + SPR.
**NEON**: `vld1_u8` 8 bytes → `vshl_n_s8(v, 4)` + `vshr_n_s8(v, 4)` for nibble split + sign-extend.
**Scalar**: `((packed >> (4*i)) & 0xf) as i8`, with `if x > 7 { x - 16 } else { x }` for sign-extend.

The `batch_packed_i4_16` closure-batch is the foundation for W1.5-#7 (randomized signatures).

### W1.2 — I8x16::saturating_abs (VPABSB correction)

**Critical**: `_mm512_abs_epi8` does NOT saturate `i8::MIN`. Returns `0x80` (= -128).

```rust
impl I8x16 { pub fn saturating_abs(self) -> Self; }
impl I8x32 { pub fn saturating_abs(self) -> Self; }
```

**AVX-512**: `_mm512_min_epu8(_mm512_abs_epi8(x), _mm512_set1_epi8(0x7f))`.
**NEON**: `vqabsq_s8(x)` — hardware-saturating (the `q` suffix).
**Scalar**: `i8::saturating_abs()` (stdlib, well-defined).

**Mandatory test**:
```rust
let input = I8x16::splat(i8::MIN);
let result = input.saturating_abs();
assert_eq!(result.lane_i8::<0>(), i8::MAX);  // 127, NOT -128
```

### W1.3 — U16x8::gather_u16 (palette lookup)

```rust
impl U16x8 {
    /// Gather 8 u16 values from `table` at given indices.
    /// Debug: panics if max(indices) >= table.len(). Release: scalar fallback.
    pub fn gather_u16(indices: U16x8, table: &[u16]) -> Self;
}
pub fn palette_lookup_u8x8(idx_v: U16x8, lut: &[u8]) -> U8x8;
```

**AVX2**: `_mm256_i32gather_epi32` with index widening + downcast.
**NEON / Scalar**: loop `(0..8).map(|i| table[indices.lane(i)])`.

### W1.4 — Prefetch hints (cross-arch)

```rust
pub fn prefetch_read_t0(ptr: *const u8);  // L1
pub fn prefetch_read_t1(ptr: *const u8);  // L2
pub fn prefetch_read_t2(ptr: *const u8);  // L3
```

**x86**: `_mm_prefetch(ptr, _MM_HINT_T0/T1/T2)`.
**aarch64**: `prfm pldl1keep/pldl2keep/pldl3keep`.
**Other**: no-op (prefetch is a hint; silent no-op is correct per ISA).

### W1.5 — U64x8::popcnt (lane-wise popcount)

```rust
impl U64x8 {
    /// Lane-wise population count. Each lane returns its u64 bit-count (0..=64).
    pub fn popcnt(self) -> Self;
    /// XOR + lane-wise popcount + horizontal sum. Optimized for Hamming distance.
    pub fn xor_popcount(self, other: Self) -> u64;
}
impl U64x4 { pub fn popcnt(self) -> Self; }  // AVX2 parity
```

**AVX-512 VPOPCNTDQ**: `_mm512_popcnt_epi64` directly (feature `avx512vpopcntdq`).
**AVX-512 without VPOPCNTDQ**: Mula's algorithm via `VPSHUFB` + `VPSADBW` per-byte LUT.
**NEON**: `vcntq_u8` → `vpaddlq_u8` cascade to sum within each u64.
**Scalar**: `u64::count_ones()` fused loop.

### W1.5+ — Deferred primitives (gated on sigker certification)

Three more primitives queued behind `lance-graph:crates/sigker` certification
(Hambly-Lyons 2010 path-signature uniqueness). **No code needed today.** Listed
so W1a additions are designed broad enough to compose with these later.

| ID | Primitive | Gate |
|----|-----------|------|
| W1.5-#6 | `signature_pde_sweep()` — Goursat PDE for `〈S(X), S(Y)〉` | `jc Pillar 11` activation |
| W1.5-#7 | Randomized projection (Cuchiero-Schmocker-Teichmann 2021) | sigker bench at production widths |
| W1.5-#8 | Lyndon-pack log-signature (7-13× compression) on `I16x16` | sigker bench at production widths |

The closure-batch shape introduced in W1.1 is the foundation for W1.5-#7.

**Gate**: All 5 primitives pass parity tests across AVX-512/NEON/scalar.
`lance-graph` `simd-savant` agent verifies compliance PRE-MERGE.
Consumer-side migration PRs (#398-#402) unblocked.

---

## Wave 2 — Codegen Macros (2 days)

Eliminates copy-paste before adding new code. Every subsequent wave benefits.
Wave 1 primitives become the first consumers of these macros (retrofit optional).

| ID | Item | Source | Effort | What It Produces |
|----|------|--------|--------|------------------|
| W2.1 | **Dtype-parity macro** (`reductions_for!`) | R3.1 | 1d | One line = 7 reductions for a dtype. Cuts 700→150 lines. |
| W2.2 | **Per-arch dispatch macro** (`simd_dispatch!`) | R3.2 | 4h | Eliminates dispatch skeleton copy-paste. |
| W2.3 | **Reduction kernel template** (`reduce_simd()`) | R3.3 | 4h | Generic chunk-loop; sum/max/nrm2 become 5-line callers. |
| W2.4 | **Dual-form fusion** (`kernel_simd_dual!`) | R3.4 | 1d | One body → `_into`, Vec, `_ptr`, all arch variants. |

**Gate**: `cargo test -p ndarray` passes. Macro output matches existing hand-rolled functions.
Run benches to verify no regression.

---

## Wave 3 — Type Bridges (2 days)

From REFACTOR_HPC_INTEGRATION.md Tier 1. Makes domain types ndarray-native.

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W3.1 | **Fingerprint<N> ↔ ArrayView1<u64>** | Tier 1.1 | 4h | Zero-copy bridge, unblocks Zip/broadcast on fingerprints |
| W3.2 | **VsaVector ↔ ArrayView1<u64>** | Tier 1.2 | 2h | Same pattern |
| W3.3 | **BF16/F16 → BlasFloat** | Tier 1.3 | 4h | Enables `Array<BF16>.dot()`, unlocks BLAS dispatch |
| W3.4 | **CogRecord channel_as_words()** | Tier 1.4 | 2h | ArrayView1<u64> per channel |
| W3.5 | **Arrow bridge → ArrayView factories** | Tier 4.1 | 2h | `binary_column_view()` returns ArrayView2 |

**Gate**: New `From`/`Into` impls compile. Existing code unchanged. Add tests for each bridge.

---

## Wave 4 — Extension Traits (3 days)

From REFACTOR_HPC_INTEGRATION.md Tier 2. hpc operations feel native on ndarray types.

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W4.1 | **HdcOps trait** (K0/K1/K2 on Array1<u64>) | Tier 2.1 | 4h | `query.cascade_distance(&candidate, &gate)` |
| W4.2 | **Quantize trait** (Array → BF16/I8 with Array output) | Tier 2.2 | 4h | `array.to_bf16()`, `array.to_i8_symmetric()` |
| W4.3 | **Prefilter on ArrayView2** | Tier 2.3 | 4h | Eliminates `(slice, rows, cols)` pattern |
| W4.4 | **ActivationsNd** (softmax_axis, log_softmax_axis) | Tier 2.4 | 4h | `batch.softmax_axis(Axis(1))` for inference |
| W4.5 | **SimdMath trait** (VML on any Array<f32>) | Tier 2.5 | 4h | `array.simd_exp()` — 10-50x for transcendentals |
| W4.6 | **Int8Matmul trait** (Array2<u8> × Array2<i8> → Array2<i32>) | Tier 3.3 | 4h | Quantized inference via ndarray types |

**Gate**: Each trait has tests matching the raw-slice function's test suite.
Benchmark: trait overhead vs. direct raw-slice call ≤ 1%.

---

## Wave 5 — Backend Wiring (2 days)

From REFACTOR_HPC_INTEGRATION.md Tier 3. Core ndarray operations silently accelerate.

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W5.1 | **Unified SIMD detection** (merge simd_caps + simd_dispatch into core) | Tier 3.2 | 4h | Deletes 877 lines of duplication |
| W5.2 | **Core sum/mean → SIMD dispatch** | Tier 3.1 | 4h | 16x faster `.sum()` on contiguous f32/f64 |
| W5.3 | **SIMD axis reductions** (sum_axis with SIMD lanes) | Tier 6.1 | 1d | ML training hot path |

**Gate**: `cargo bench` shows measurable improvement on contiguous arrays.
Non-contiguous arrays unchanged (fallback to generic fold).

---

## Wave 6 — SoA Kernel Architecture (1 week)

From SOA_KERNEL_ARCHITECTURE.md. The genius-level structural change.
Wave 1's `U64x8::popcnt()` and `U64x8::xor_popcount()` are direct prerequisites
for the columnar XOR+popcount scan in W6.2.

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W6.1 | **TieredDatabase struct** (K0/K1/K2 as F-order Array2) | SoA §5 | 1d | Physical separation matching cache hierarchy |
| W6.2 | **k0_columnar_simd** (scan N u64s with broadcast+XOR+popcount) | SoA §1 | 1d | Uses `U64x8::xor_popcount` from W1.5. 8 candidates per cycle. |
| W6.3 | **Bitmask survivor propagation** (mask narrowing, no branching) | SoA §2 | 4h | Replace per-candidate `continue` with mask ops |
| W6.4 | **k1_accumulate_columnar** (8-column scan with mask) | SoA §1 | 4h | K1 on contiguous columns |
| W6.5 | **k2_exact_masked** (row extraction for survivors only) | SoA §1 | 4h | Stack-copy 2KB per survivor |
| W6.6 | **BF16FieldDatabase** (sign/exp/mantissa separated at ingest) | SoA §6 | 1d | Awareness without runtime decomposition |
| W6.7 | **QualiaColumns** (16 × Array1<i8>) | SoA §7 | 4h | 18x dimension scan throughput. Uses `I8x16::from_i4_packed_u64` from W1.1. |
| W6.8 | **Arrow native path** (RecordBatch columns → columnar scan) | SoA §4 | 4h | Zero-copy from Lance/Arrow |
| W6.9 | **Benchmark: AoS vs SoA** on 1M SKU-16K containers | SoA §perf | 4h | Proof of 4-8x claim |

**Gate**: SoA cascade produces identical results to AoS cascade on full test suite.
Benchmark confirms ≥3x throughput improvement on 1M containers.

---

## Wave 7 — Namespace Restructure (3 days)

From transformer feedback R2.1. Enforces the architecture rule from CLAUDE.md.

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W7.1 | **Split hpc/ → hpc/ + cog/ + ext/ + io/** | R2.1 | 2-3d | 30 numeric modules in hpc/, 35 cognitive in cog/, 20 experimental in ext/, 6 I/O in io/ |
| W7.2 | **SIMD directory consolidation** (simd_*.rs → src/simd/) | R2.2 | 1d | One directory for all SIMD code. W1 primitives land in new locations. |
| W7.3 | **Quantized module split** (quantized.rs → hpc/quant/) | R2.3 | 4h | BlockQ4_0 packed struct for candle compat |

**Gate**: `pub use` deprecation shims for all moved modules.
All existing `use ndarray::hpc::*` paths still resolve (with deprecation warning).

---

## Wave 8 — Test & Bench Infrastructure (2 days)

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W8.1 | **Extract integration tests** to `tests/hpc/`, `tests/cog/` | R4.2 | 2d | Module files lose 30-50% of line count |
| W8.2 | **HPC bench harness** (dot, reductions, softmax, quantized) | R4.5 | 1d | Reproducible perf claims |
| W8.3 | **Cross-comparison bench** (ndarray vs candle vs tract kernel) | R4.5 ext | 4h | The whole reason for B1 |
| W8.4 | **W1a primitive parity bench** (AVX-512 vs NEON vs scalar speedup) | lance-graph | 4h | Acceptance criteria §4 |

---

## Wave 9 — Version Bump & Downstream Pin (1 day)

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W9.1 | **Tag v0.18.0** | R4.4 | 2h | Clean cut |
| W9.2 | **CHANGELOG with migration guide** | R4.4 | 2h | Downstream knows what changed |
| W9.3 | **Downstream Cargo.toml pins** (burn, candle, tract, ort, lance-graph) | R4.4 | 2h | tag = "v0.18.0" |
| W9.4 | **Remove deprecation shims** (schedule for v0.19) | R2.1 | — | Future cleanup |

---

## Dependency Graph

```
Wave 0 (conventions)
  │
  ├──→ Wave 1 (SIMD primitives) ←── P0 for lance-graph
  │       │
  │       ├──→ Wave 6.2 (k0_columnar uses U64x8::xor_popcount from W1.5)
  │       ├──→ Wave 6.7 (QualiaColumns uses I8x16::from_i4_packed from W1.1)
  │       └──→ Wave 8.4 (primitive parity benches)
  │
  ├──→ Wave 2 (codegen macros)
  │       │
  │       └──→ Wave 5 (backend wiring uses macros)
  │
  ├──→ Wave 3 (type bridges)
  │       │
  │       ├──→ Wave 4 (extension traits need bridges)
  │       ├──→ Wave 5 (backend dispatch needs BlasFloat for BF16)
  │       └──→ Wave 6 (SoA needs Fingerprint↔Array, Arrow views)
  │
  ├──→ Wave 7 (namespace) — independent of 1-6
  │
  └──→ Wave 8 (tests/benches) — after 1+5+6
          │
          └──→ Wave 9 (release)
```

### Critical Path

**For lance-graph (P0 consumer)**: 0 → 1 = **6 days**. Consumer migration PRs unblocked.

**For full release**: 0 → 1 → 3 → 5 → 6 → 8 → 9 = **18 days** serial.
With parallelism (1∥2, 3∥4, 6∥7, 8∥backlog): **~12 working days**.

---

## How Wave 1 Feeds Everything Downstream

The 5 SIMD primitives aren't isolated additions — they're load-bearing for later waves:

| Primitive | Immediate consumer (lance-graph) | Later consumer (ndarray) |
|-----------|----------------------------------|--------------------------|
| `I8x16::from_i4_packed_u64` | `mul.rs::i4_eval::batch` (5 batch fns) | **W6.7 QualiaColumns** — unpack i4 qualia from packed storage without scalar loop |
| `I8x16::saturating_abs` | Direction-B fix, ValleyOfDespair classifier | **W4.2 Quantize trait** — safe abs in quantization error metrics |
| `U16x8::gather_u16` | `bgz17/simd.rs` palette lookup | **W6.6 BF16FieldDatabase** — gather exponent fields from lookup table |
| `prefetch_read_t0/t1/t2` | `bgz17/prefetch.rs` tile prefetch | **W6.2 k0_columnar_simd** — prefetch next column chunk during K0 scan |
| `U64x8::popcnt` + `xor_popcount` | `holograph/hamming.rs` + `blasgraph/types.rs` | **W6.2-W6.5 entire SoA cascade** — columnar XOR+popcount is THE operation |

The SoA cascade (Wave 6) is fundamentally built on `U64x8::xor_popcount()`. Without Wave 1,
the SoA implementation would need to drop back to `hpc::bitwise::hamming_distance_raw()` on
slices, losing the typed-wrapper discipline and duplicating dispatch logic.

---

## The SoA Integration with All Waves

| Wave 6 needs | Provided by |
|--------------|-------------|
| `U64x8::xor_popcount()` for columnar scan | **Wave 1.5** (SIMD primitives) |
| `I8x16::from_i4_packed_u64()` for qualia | **Wave 1.1** (SIMD primitives) |
| `prefetch_read_t0()` for column prefetch | **Wave 1.4** (SIMD primitives) |
| F-order Array2<u64> database | **Wave 3** (type bridges — Fingerprint converts to Array) |
| Dispatch macro for k0_columnar_simd | **Wave 2** (codegen macros) |
| `_into` form for columnar kernels | **Wave 0** (signature convention) |
| Extension trait: `database.cascade_soa(&query, &gate)` | **Wave 4** (HdcOps trait extended) |
| BF16FieldDatabase uses Quantize trait | **Wave 4** (Quantize extension) |
| Arrow columns → direct scan | **Wave 3** (Arrow view factories) |
| Unified SIMD detection for dispatch | **Wave 5** (backend wiring) |
| Benchmark harness to prove 4-8x | **Wave 8** (bench infrastructure) |

---

## The Meta-Level SoA Observation

The SoA insight goes beyond just "transpose the database." It's a design principle
that applies recursively across the entire surface:

### Level 1: Database layout (SOA_KERNEL_ARCHITECTURE.md)
- Fingerprint database: column-per-word → cascade becomes column scan
- BF16 fields: separated sign/exp/mantissa → awareness is native layout
- Qualia: column-per-dimension → per-dim queries are sequential scans

### Level 2: API surface
- The **dual-form convention** (W0.1) IS SoA thinking applied to function signatures:
  - AoS API: `fn foo(input) -> output` (allocates inside, hides layout)
  - SoA API: `fn foo_into(input, output)` (caller controls memory layout)

### Level 3: SIMD primitives (lance-graph W1a)
- The **typed-wrapper-as-method** convention IS SoA for the SIMD surface:
  - AoS: raw intrinsics scattered across consumer crates (158 violations)
  - SoA: struct methods on typed wrappers, consumers import ONE namespace
- The **closure-batch pattern** (W1.1 `batch_packed_i4_16`) IS SoA for computation:
  - AoS: each consumer writes its own unpack+process+pack loop
  - SoA: one batch primitive accepts a closure, owns chunking/tailing/dispatch

### Level 4: Module structure (W7.1 namespace split)
- AoS: one module (`hpc/`) contains everything → grep to find anything
- SoA: separated by concern (`hpc/`, `cog/`, `ext/`, `io/`) → contiguous columns

### Level 5: ndarray's type system
- `Array2<u64>` in F-order IS the SoA database
- `.column(i)` IS the field-access pattern
- `.row(i)` IS the record-access pattern
- Strides encode the duality without duplication

### Level 6: Arrow + Lance alignment
- Arrow IS columnar storage (SoA by definition)
- Lance stores Arrow columns on disk
- The SoA kernel reads Arrow columns directly — no ETL, no transpose, no copy
- Storage format = memory layout = compute pattern = cache access order

### Level 7: The consumer contract itself
- lance-graph declares what it needs (W1a primitives), ndarray provides
- The contract is columnar: one primitive per concern, not a monolithic "SIMD library"
- Each primitive is independently testable, deployable, and benchmarkable

**The architecture collapses all intermediate transformations at every level.**

---

## Effort Summary

| Wave | Days | Parallel? | Key Outcome |
|------|------|-----------|-------------|
| 0 | 3 | Yes | Conventions locked |
| 1 | 3 | Yes (after 0) | **lance-graph unblocked** (P0) |
| 2 | 2 | ∥ with 1 | Codegen macros ready |
| 3 | 2 | After 0 | Domain types ↔ Array |
| 4 | 3 | After 3 | Extension traits live |
| 5 | 2 | After 2+3 | Core ops accelerated |
| 6 | 5 | After 1+3+5 | SoA cascade (the big win) |
| 7 | 3 | ∥ with 4-6 | Namespace clean |
| 8 | 2 | After 1+6 | Proof via benchmarks |
| 9 | 1 | After all | Release |

**lance-graph unblocked**: Wave 0+1 = **6 days**.
**Full release critical path**: 18 days serial, **~12 days** with parallelism.

---

## What NOT to Do

1. **Don't delete hpc/ modules** — raw-slice functions stay (FFI, embedded, zero-overhead)
2. **Don't premature-v1.0** — surface is too young; candle/tract wiring will shake it again
3. **Don't gate SoA behind a feature flag** — it's the default hot path, not optional
4. **Don't couple SoA with module restructure** — they're independent; merge separately
5. **Don't break downstream in one shot** — deprecation shims for one release minimum
6. **Don't ship W1a primitives without parity tests** — the codex P2 i8::MIN divergence on PR #398 happened because no such test existed
7. **Don't add `is_x86_feature_detected!` outside simd_caps.rs** — dispatch through the singleton
8. **Don't implement W1.5+ (deferred primitives) until sigker certification** — they're gated
