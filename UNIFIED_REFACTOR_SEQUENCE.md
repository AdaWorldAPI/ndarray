# Unified Refactor Sequence: ndarray Fork

> Integrates three analysis passes into one executable sequence:
> 1. **REFACTOR_HPC_INTEGRATION.md** — Type bridges & extension traits
> 2. **SOA_KERNEL_ARCHITECTURE.md** — Columnar cascade & field-separated storage
> 3. **Transformer session feedback** — API conventions, namespace, codegen macros
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

**Gate**: All downstream crates (burn, candle, tract, ort) still compile unchanged.
The dual-form is additive; existing call sites continue working.

---

## Wave 1 — Codegen Macros (2 days)

Eliminates copy-paste before adding new code. Every subsequent wave benefits.

| ID | Item | Source | Effort | What It Produces |
|----|------|--------|--------|------------------|
| W1.1 | **Dtype-parity macro** (`reductions_for!`) | R3.1 | 1d | One line = 7 reductions for a dtype. Cuts 700→150 lines. |
| W1.2 | **Per-arch dispatch macro** (`simd_dispatch!`) | R3.2 | 4h | Eliminates dispatch skeleton copy-paste. |
| W1.3 | **Reduction kernel template** (`reduce_simd()`) | R3.3 | 4h | Generic chunk-loop; sum/max/nrm2 become 5-line callers. |
| W1.4 | **Dual-form fusion** (`kernel_simd_dual!`) | R3.4 | 1d | One body → `_into`, Vec, `_ptr`, all arch variants. |

**Gate**: `cargo test -p ndarray` passes. Macro output matches existing hand-rolled functions.
Run benches to verify no regression.

---

## Wave 2 — Type Bridges (2 days)

From REFACTOR_HPC_INTEGRATION.md Tier 1. Makes domain types ndarray-native.

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W2.1 | **Fingerprint<N> ↔ ArrayView1<u64>** | Tier 1.1 | 4h | Zero-copy bridge, unblocks Zip/broadcast on fingerprints |
| W2.2 | **VsaVector ↔ ArrayView1<u64>** | Tier 1.2 | 2h | Same pattern |
| W2.3 | **BF16/F16 → BlasFloat** | Tier 1.3 | 4h | Enables `Array<BF16>.dot()`, unlocks BLAS dispatch |
| W2.4 | **CogRecord channel_as_words()** | Tier 1.4 | 2h | ArrayView1<u64> per channel |
| W2.5 | **Arrow bridge → ArrayView factories** | Tier 4.1 | 2h | `binary_column_view()` returns ArrayView2 |

**Gate**: New `From`/`Into` impls compile. Existing code unchanged. Add tests for each bridge.

---

## Wave 3 — Extension Traits (3 days)

From REFACTOR_HPC_INTEGRATION.md Tier 2. hpc operations feel native on ndarray types.

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W3.1 | **HdcOps trait** (K0/K1/K2 on Array1<u64>) | Tier 2.1 | 4h | `query.cascade_distance(&candidate, &gate)` |
| W3.2 | **Quantize trait** (Array → BF16/I8 with Array output) | Tier 2.2 | 4h | `array.to_bf16()`, `array.to_i8_symmetric()` |
| W3.3 | **Prefilter on ArrayView2** | Tier 2.3 | 4h | Eliminates `(slice, rows, cols)` pattern |
| W3.4 | **ActivationsNd** (softmax_axis, log_softmax_axis) | Tier 2.4 | 4h | `batch.softmax_axis(Axis(1))` for inference |
| W3.5 | **SimdMath trait** (VML on any Array<f32>) | Tier 2.5 | 4h | `array.simd_exp()` — 10-50x for transcendentals |
| W3.6 | **Int8Matmul trait** (Array2<u8> × Array2<i8> → Array2<i32>) | Tier 3.3 | 4h | Quantized inference via ndarray types |

**Gate**: Each trait has tests matching the raw-slice function's test suite.
Benchmark: trait overhead vs. direct raw-slice call ≤ 1%.

---

## Wave 4 — Backend Wiring (2 days)

From REFACTOR_HPC_INTEGRATION.md Tier 3. Core ndarray operations silently accelerate.

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W4.1 | **Unified SIMD detection** (merge simd_caps + simd_dispatch into core) | Tier 3.2 | 4h | Deletes 877 lines of duplication |
| W4.2 | **Core sum/mean → SIMD dispatch** | Tier 3.1 | 4h | 16x faster `.sum()` on contiguous f32/f64 |
| W4.3 | **SIMD axis reductions** (sum_axis with SIMD lanes) | Tier 6.1 | 1d | ML training hot path |

**Gate**: `cargo bench` shows measurable improvement on contiguous arrays.
Non-contiguous arrays unchanged (fallback to generic fold).

---

## Wave 5 — SoA Kernel Architecture (1 week)

From SOA_KERNEL_ARCHITECTURE.md. The genius-level structural change.

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W5.1 | **TieredDatabase struct** (K0/K1/K2 as F-order Array2) | SoA §5 | 1d | Physical separation matching cache hierarchy |
| W5.2 | **k0_columnar_simd** (scan N u64s with broadcast+XOR+popcount) | SoA §1 | 1d | 8 candidates per cycle, sequential memory |
| W5.3 | **Bitmask survivor propagation** (mask narrowing, no branching) | SoA §2 | 4h | Replace per-candidate `continue` with mask ops |
| W5.4 | **k1_accumulate_columnar** (8-column scan with mask) | SoA §1 | 4h | K1 on contiguous columns |
| W5.5 | **k2_exact_masked** (row extraction for survivors only) | SoA §1 | 4h | Stack-copy 2KB per survivor |
| W5.6 | **BF16FieldDatabase** (sign/exp/mantissa separated at ingest) | SoA §6 | 1d | Awareness without runtime decomposition |
| W5.7 | **QualiaColumns** (16 × Array1<i8>) | SoA §7 | 4h | 18x dimension scan throughput |
| W5.8 | **Arrow native path** (RecordBatch columns → columnar scan) | SoA §4 | 4h | Zero-copy from Lance/Arrow |
| W5.9 | **Benchmark: AoS vs SoA** on 1M SKU-16K containers | SoA §perf | 4h | Proof of 4-8x claim |

**Gate**: SoA cascade produces identical results to AoS cascade on full test suite.
Benchmark confirms ≥3x throughput improvement on 1M containers.

---

## Wave 6 — Namespace Restructure (3 days)

From transformer feedback R2.1. Enforces the architecture rule from CLAUDE.md.

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W6.1 | **Split hpc/ → hpc/ + cog/ + ext/ + io/** | R2.1 | 2-3d | 30 numeric modules in hpc/, 35 cognitive in cog/, 20 experimental in ext/, 6 I/O in io/ |
| W6.2 | **SIMD directory consolidation** (simd_*.rs → src/simd/) | R2.2 | 1d | One directory for all SIMD code |
| W6.3 | **Quantized module split** (quantized.rs → hpc/quant/) | R2.3 | 4h | BlockQ4_0 packed struct for candle compat |

**Gate**: `pub use` deprecation shims for all moved modules.
All existing `use ndarray::hpc::*` paths still resolve (with deprecation warning).

---

## Wave 7 — Test & Bench Infrastructure (2 days)

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W7.1 | **Extract integration tests** to `tests/hpc/`, `tests/cog/` | R4.2 | 2d | Module files lose 30-50% of line count |
| W7.2 | **HPC bench harness** (dot, reductions, softmax, quantized) | R4.5 | 1d | Reproducible perf claims |
| W7.3 | **Cross-comparison bench** (ndarray vs candle vs tract kernel) | R4.5 ext | 4h | The whole reason for B1 |

---

## Wave 8 — Version Bump & Downstream Pin (1 day)

| ID | Item | Source | Effort | Impact |
|----|------|--------|--------|--------|
| W8.1 | **Tag v0.18.0** | R4.4 | 2h | Clean cut |
| W8.2 | **CHANGELOG with migration guide** | R4.4 | 2h | Downstream knows what changed |
| W8.3 | **Downstream Cargo.toml pins** (burn, candle, tract, ort) | R4.4 | 2h | tag = "v0.18.0" |
| W8.4 | **Remove deprecation shims** (schedule for v0.19) | R2.1 | — | Future cleanup |

---

## The SoA Integration with Conventions

The SoA architecture (Wave 5) benefits from every preceding wave:

| Wave 5 needs | Provided by |
|--------------|-------------|
| F-order Array2<u64> database | Wave 2 (type bridges — Fingerprint converts to Array) |
| SIMD column scanning | Wave 4 (unified SIMD detection, core dispatch) |
| Bitmask as Array1<u64> | Wave 2 (ArrayView factory pattern) |
| `_into` form for columnar kernels | Wave 0 (signature convention) |
| Dispatch macro for k0_columnar_simd | Wave 1 (codegen macros) |
| Extension trait: `database.cascade_soa(&query, &gate)` | Wave 3 (HdcOps trait extended) |
| BF16FieldDatabase uses Quantize trait | Wave 3 (Quantize extension) |
| Arrow columns → direct scan | Wave 2 (Arrow view factories) |
| Benchmark harness to prove 4-8x | Wave 7 (bench infrastructure) |

This is why the waves are ordered this way. The SoA kernel is the most impactful
change (~4-8x cascade throughput), but it needs the convention/macro/bridge
foundation to land cleanly. Without Wave 0-4, the SoA code would be yet another
raw-slice module that doesn't integrate with ndarray — repeating the original sin.

---

## The Meta-Level SoA Observation

The SoA insight goes beyond just "transpose the database." It's a design principle
that applies recursively across the entire hpc/ surface:

### Level 1: Database layout (SOA_KERNEL_ARCHITECTURE.md)
- Fingerprint database: column-per-word → cascade becomes column scan
- BF16 fields: separated sign/exp/mantissa → awareness is native layout
- Qualia: column-per-dimension → per-dim queries are sequential scans

### Level 2: API surface (this document)
- The **dual-form convention** (W0.1) IS SoA thinking applied to function signatures:
  - AoS API: `fn foo(input) -> output` (allocates inside, hides layout)
  - SoA API: `fn foo_into(input, output)` (caller controls memory layout)
  - The `_into` form lets the caller decide whether output is contiguous, chunked, or interleaved

### Level 3: Module structure (W6.1 namespace split)
- AoS: one module (`hpc/`) contains everything → grep to find anything
- SoA: separated by concern (`hpc/`, `cog/`, `ext/`, `io/`) → each namespace is a contiguous "column" of related functionality that you scan sequentially

### Level 4: ndarray's type system
- `Array2<u64>` in F-order IS the SoA database. No wrapper type needed.
- `.column(i)` IS the field-access pattern. No custom iterator needed.
- `.row(i)` IS the record-access pattern. No conversion needed.
- Strides encode the duality. The same allocation serves both patterns.

### Level 5: The cascade pipeline itself
- K0: ONE field (word 0) across ALL records → column scan
- K1: EIGHT fields (words 0-7) across SURVIVORS → multi-column scan
- K2: ALL fields across FEW records → row reconstruction

This is a **widening read pattern**: start narrow (1 column), widen as population shrinks.
SoA makes the narrow-start cheap (contiguous) and tolerates the wide-end being strided
(because so few records reach it).

### Level 6: Arrow + Lance alignment
- Arrow IS columnar storage (SoA by definition)
- Lance stores Arrow columns on disk
- The SoA kernel reads Arrow columns directly — no ETL, no transpose, no copy
- The database layout on disk IS the query execution layout in memory
- This is the zero-copy nirvana: mmap → column pointer → SIMD scan → result

**The architecture collapses all intermediate transformations.**
Storage format = memory layout = compute pattern = cache access order.
One decision (F-order Array2) propagates from disk through cache to SIMD lanes.

---

## Effort Summary

| Wave | Days | Parallel? | Key Outcome |
|------|------|-----------|-------------|
| 0 | 3 | Yes | Conventions locked |
| 1 | 2 | Yes | Codegen macros ready |
| 2 | 2 | Yes (after 0) | Domain types ↔ Array |
| 3 | 3 | Yes (after 2) | Extension traits live |
| 4 | 2 | After 1+2 | Core ops accelerated |
| 5 | 5 | After 2+4 | SoA cascade (the big win) |
| 6 | 3 | After 0 | Namespace clean |
| 7 | 2 | After 5 | Proof via benchmarks |
| 8 | 1 | After all | Release |

**Critical path**: 0 → 2 → 4 → 5 → 7 → 8 = **15 days** serial.
With parallel waves (1∥2, 3∥4, 6∥5, 7∥8): **~10 working days**.

---

## What NOT to Do

1. **Don't delete hpc/ modules** — raw-slice functions stay (FFI, embedded, zero-overhead)
2. **Don't premature-v1.0** — surface is too young; candle/tract wiring will shake it again
3. **Don't gate SoA behind a feature flag** — it's the default hot path, not optional
4. **Don't couple SoA with module restructure** — they're independent; merge separately
5. **Don't break downstream in one shot** — deprecation shims for one release minimum
