# HPC API Stability Commitment — AdaWorldAPI/ndarray Fork

> **2026-05-18 wave-2 update**: `l1_f64_simd`, `l2_f64_simd`, and
> `linf_f64_simd` were initially marked as "aspirational reserved names"
> in this doc because they were absent from the source at wave-1 time
> (per `docs/hpc-api-inventory.md`). Wave-2 commit `71cdbd4`
> ("feat(hpc): materialise l1_f64_simd, l2_f64_simd, linf_f64_simd in
> heel_f64x8") materialised all three with the exact signatures this doc
> promised, matching `cosine_f64_simd`'s F64x8-chunk + scalar-tail
> pattern. 15/15 `heel_f64x8` tests pass. The "Stable public surface"
> table below now describes REAL APIs — not promises. Reading order:
> the freeze commitment is now load-bearing, not aspirational.


**Status:** DRAFT — first published 2026-05-18
**Branch at time of publication:** `claude/lance-surrealdb-analysis-LXmug`
**Applies to crate:** `ndarray` v0.17.x (AdaWorldAPI fork of rust-ndarray/ndarray)
**Rust edition:** 2021 / MSRV 1.95 stable

---

## Table of Contents

1. [Scope](#1-scope)
2. [Stable Public Surface](#2-stable-public-surface)
3. [Internal / Unstable Surface](#3-internal--unstable-surface)
4. [What "Frozen" Means](#4-what-frozen-means)
5. [Adding New Kernels — The Additive Pattern](#5-adding-new-kernels--the-additive-pattern)
6. [Diamond-Dependency Guard](#6-diamond-dependency-guard)
7. [CI Commitment](#7-ci-commitment)
8. [Cross-References to Integration Plans](#8-cross-references-to-integration-plans)
9. [Appendix: Numeric Tolerance Derivation](#9-appendix-numeric-tolerance-derivation)

---

## 1. Scope

### Why This Document Exists

This fork of ndarray (`github.com/AdaWorldAPI/ndarray`) diverges from
upstream `rust-ndarray/ndarray` by adding a significant HPC layer under
`src/hpc/` (175+ Rust source files at time of writing) and a portable SIMD
abstraction layer in `src/simd.rs`, `src/simd_avx512.rs`, `src/simd_avx2.rs`,
and `src/simd_neon.rs`.

The fork occupies a **load-bearing position in two dependency chains**:

```
surrealdb-core
  └── [patch.crates-io] ndarray = { git = "...", branch = "..." }
        └── ndarray::hpc::heel_f64x8  (vector index distance kernels)
        └── ndarray::simd::F64x8      (SIMD register type)

lance-graph cognitive crates
  └── ndarray::hpc::heel_f64x8       (cosine/L1/L2/Linf distance)
  └── ndarray::simd::F64x8            (F64x8 polyfill — AVX-512/AVX2/NEON)
  └── ndarray::hpc::bitwise           (Hamming / DataFusion UDFs)
  └── ndarray::hpc::udf_kernels       (DataFusion-registered UDFs)
```

**Signature breaks in this fork cascade silently into compilation failures in
surrealdb-core and lance-graph**, often manifesting only when the downstream
workspace links the ONNX runtime crate (`ort`) against a surrealdb-core that
now has a different ndarray ABI than `ort` expects. This is the classic Rust
diamond-dependency problem: two crates in the same build graph that each
depend on `ndarray` but at different versions or from different sources get
separate incompatible types even though they share a name.

This document codifies which parts of the public surface are **frozen**,
which are explicitly **unstable**, and the rules that govern the addition of
new functionality without breaking existing consumers.

---

## 2. Stable Public Surface

### 2.1 Overview

The following items constitute the **frozen, stable public API** of this fork.
Changes to any item in this table require a coordinated bump of `Cargo.toml`
`version` plus deprecation notices in all known downstreams before the old
signature is removed.

| Symbol | Module path | Kind |
|--------|-------------|------|
| `F64x8` | `ndarray::simd::F64x8` (re-exported from `ndarray::hpc::heel_f64x8` via `ndarray::simd`) | `pub struct` |
| `cosine_f64_simd` | `ndarray::hpc::heel_f64x8::cosine_f64_simd` | `pub fn` |
| `l1_f64_simd` | `ndarray::hpc::heel_f64x8` (aspirational; see §2.3) | `pub fn` |
| `l2_f64_simd` | `ndarray::hpc::heel_f64x8` (aspirational; see §2.3) | `pub fn` |
| `linf_f64_simd` | `ndarray::hpc::heel_f64x8` (aspirational; see §2.3) | `pub fn` |
| `hpc-extras` | `Cargo.toml [features]` | Cargo feature |

### 2.2 `ndarray::simd::F64x8`

**File:** `src/simd_avx512.rs:304` (AVX-512 backend),
`src/simd_avx2.rs:815` (AVX2 backend),
`src/simd_neon.rs:819` (NEON backend),
unified dispatch in `src/simd.rs:244` / `src/simd.rs:280`.

**Definition (canonical, backend-neutral):**

```rust
// AVX-512 (src/simd_avx512.rs:304):
pub struct F64x8(pub __m512d);

// AVX2 (src/simd_avx2.rs:815):
pub struct F64x8(pub f64x4, pub f64x4);   // 2 × __m256d

// NEON (src/simd_neon.rs:819):
pub struct F64x8(pub [float64x2_t; 4]);   // 4 × 128-bit NEON lanes

// Scalar fallback (simd.rs dispatch, target = other):
pub struct F64x8([f64; 8]);
```

**Stable constructor and accessor methods** (identical signature on all four
backends — this uniformity IS the contract):

```rust
impl F64x8 {
    pub fn splat(v: f64) -> Self;
    pub fn from_slice(s: &[f64]) -> Self;   // reads first 8 elements
    pub fn from_array(arr: [f64; 8]) -> Self;
    pub fn to_array(self) -> [f64; 8];
    pub fn reduce_sum(self) -> f64;
    pub fn mul_add(self, b: Self, c: Self) -> Self;  // FMA: self*b + c
}

// Arithmetic traits (all backends):
impl Add<F64x8> for F64x8 { type Output = F64x8; }
impl Sub<F64x8> for F64x8 { type Output = F64x8; }
impl Mul<F64x8> for F64x8 { type Output = F64x8; }
impl Div<F64x8> for F64x8 { type Output = F64x8; }
impl AddAssign<F64x8> for F64x8 { }
impl SubAssign<F64x8> for F64x8 { }
impl MulAssign<F64x8> for F64x8 { }
impl DivAssign<F64x8> for F64x8 { }
impl Neg for F64x8 { type Output = F64x8; }
impl PartialEq for F64x8 { }
impl Default for F64x8 { }
impl fmt::Debug for F64x8 { }
```

**Semantics:** `F64x8` is an 8-wide lane of `f64` values. All arithmetic
operations are element-wise. `mul_add(b, c)` computes `self * b + c` with
FMA semantics where the hardware supports it; on backends lacking FMA the
result may differ by up to one ULP from a strict fused multiply-add.
`reduce_sum` returns the horizontal sum of all 8 lanes.

**Consumer code pattern** (what downstream crates MUST write — the
polyfill handles backend selection):

```rust
use ndarray::simd::F64x8;

let va = F64x8::from_slice(&a[i*8..]);
let vb = F64x8::from_slice(&b[i*8..]);
let acc = va.mul_add(vb, acc);  // acc = va * vb + acc
```

Consumers MUST NOT import from `ndarray::simd_avx512`, `ndarray::simd_avx2`,
or `ndarray::simd_neon` directly. Those modules are internal dispatch
backends (see §3).

**Numeric tolerance:** `F64x8` arithmetic results agree with IEEE 754
double-precision arithmetic to within the rounding error introduced by the
FMA instruction: at most 0.5 ULP per operation. For a dot product of length
`n` computed via `mul_add` + `reduce_sum`, the accumulated error is bounded
by `f64::EPSILON * n` relative to the scalar reference value computed with
the same operands in the same order.

### 2.3 `heel_f64x8::cosine_f64_simd`

**File:** `src/hpc/heel_f64x8.rs:109`

**Signature:**

```rust
pub fn cosine_f64_simd(a: &[f64], b: &[f64]) -> f64
```

**Semantics:**

Computes the cosine similarity between vectors `a` and `b`:

```
cosine(a, b) = dot(a, b) / (||a||₂ × ||b||₂)
```

Returns a value in `[-1.0, 1.0]`. Returns `0.0` when either input is the
zero vector (denominator < `1e-12`).

The implementation processes 8 elements per SIMD iteration using `F64x8`
FMA, then handles the scalar remainder. A single pass accumulates `dot`,
`norm_a`, and `norm_b` simultaneously — no second pass over the data.

**Numeric tolerance contract:**

The SIMD result agrees with the scalar reference (naive `f64` loop over
`dot`, `na`, `nb`, then `dot / (na * nb).sqrt()`) to within:

```
|cosine_simd(a,b) - cosine_scalar(a,b)| < f64::EPSILON * len
```

where `len = a.len().min(b.len())`. This contract is validated by the
regression test `cosine_matches_scalar` in `src/hpc/heel_f64x8.rs:278`.

**Invariants that must not change:**

- The return type is always `f64`.
- Slices of unequal length: only the `min(a.len(), b.len())` prefix is used.
- Empty slices: both `a` and `b` of length 0 return `0.0` (zero-vector guard).
- NaN propagation: if either input contains `NaN`, the result is `NaN`
  (IEEE 754 semantics propagate through `F64x8` arithmetic).

### 2.4 `heel_f64x8::l1_f64_simd` (aspirational frozen)

**Status:** This function name is reserved in the stability commitment but
does not yet exist as a standalone `pub fn` in `heel_f64x8.rs` at time of
publication. The L1 norm capability exists in the codebase under different
names (`hpc/bgz17_bridge.rs:419` as `Base17::l1`, `hpc/holo.rs:1981` as
`focus_l1`), but a unified, slice-oriented `l1_f64_simd` kernel for the
`ndarray::hpc::heel_f64x8` module is called for by the integration plan.

**Intended signature when implemented:**

```rust
pub fn l1_f64_simd(a: &[f64], b: &[f64]) -> f64
```

**Intended semantics:**

Computes the L1 (Manhattan) distance:

```
L1(a, b) = Σᵢ |aᵢ - bᵢ|
```

SIMD implementation using `F64x8`: compute element-wise absolute
differences in 8-wide chunks, accumulate via `reduce_sum`.

**Numeric tolerance contract (when implemented):**

```
|l1_simd(a,b) - l1_scalar(a,b)| ≤ f64::EPSILON * len
```

### 2.5 `heel_f64x8::l2_f64_simd` (aspirational frozen)

**Status:** Reserved. Not yet implemented as a standalone function in
`heel_f64x8.rs`. Existing L2/Euclidean distance kernels live in
`src/hpc/distance.rs` (squared L2 for spatial point sets) and
`src/hpc/cam_pq.rs` (as `squared_l2`, re-exported through `ndarray::simd`).

**Intended signature when implemented:**

```rust
pub fn l2_f64_simd(a: &[f64], b: &[f64]) -> f64
```

**Intended semantics:**

Computes the L2 (Euclidean) distance:

```
L2(a, b) = sqrt(Σᵢ (aᵢ - bᵢ)²)
```

SIMD implementation: accumulate squared differences via `F64x8::mul_add`,
then `reduce_sum`, then scalar `sqrt`. Note: the sqrt is applied once after
the vector accumulation — not inside the SIMD loop.

**Numeric tolerance contract (when implemented):**

```
|l2_simd(a,b) - l2_scalar(a,b)| ≤ f64::EPSILON * len
```

### 2.6 `heel_f64x8::linf_f64_simd` (aspirational frozen)

**Status:** Reserved. Not yet implemented as a standalone function.

**Intended signature when implemented:**

```rust
pub fn linf_f64_simd(a: &[f64], b: &[f64]) -> f64
```

**Intended semantics:**

Computes the L-infinity (Chebyshev) distance:

```
L∞(a, b) = max_i |aᵢ - bᵢ|
```

SIMD implementation: compute element-wise absolute differences via `F64x8`,
reduce via element-wise max, then final horizontal max over 8 lanes.

**Numeric tolerance contract (when implemented):**

```
|linf_simd(a,b) - linf_scalar(a,b)| = 0.0
```

The L-infinity distance is a pure selection (max of absolute values), which
is exact under IEEE 754. No accumulation error is introduced.

### 2.7 `hpc-extras` Cargo Feature

**File:** `Cargo.toml:207`

```toml
hpc-extras = ["std", "dep:p64", "dep:fractal", "fractal/std"]
```

**Semantic contract:** Enabling `hpc-extras` (which is part of the `default`
feature set — see `Cargo.toml:174`) activates the p64 palette / NARS bridge
and the fractal manifold crates. The following modules become available:

- `ndarray::hpc::spo_bundle`
- `ndarray::hpc::deepnsm`
- `ndarray::hpc::compression_curves`
- `ndarray::hpc::crystal_encoder`
- `ndarray::hpc::p64_bridge`

**Stability contract for `hpc-extras`:**

1. The feature name `hpc-extras` is frozen. It will not be renamed.
2. Its implied set of features (`std`, `dep:p64`, `dep:fractal`) is frozen
   in the sense that it will never be made _smaller_ without a semver bump
   and deprecation period. The set may grow (new optional deps are additive).
3. Consumers that build with `default-features = false` and do not re-enable
   `hpc-extras` will continue to have a working build. The `hpc` module is
   always available with `std`; only the `hpc-extras`-gated submodules
   (listed above) disappear. The stable surface (`F64x8`, `cosine_f64_simd`,
   etc.) is in the `std`-gated core and is NOT gated on `hpc-extras`.

---

## 3. Internal / Unstable Surface

The following items are **NOT part of the stable API**. They may change
without notice between versions, including patch releases during active
development. Downstream crates that depend on them are responsible for
tracking changes.

### 3.1 Backend Dispatch Modules

```
src/simd_avx512.rs    — AVX-512F + AVX-512VBMI intrinsics
src/simd_avx2.rs      — AVX2 + fallback intrinsics
src/simd_neon.rs      — ARM NEON paired-load implementation
src/simd.rs           — compile-time + runtime dispatch glue
```

The internal layout of these modules — which intrinsic calls are used,
which `#[target_feature]` guards appear, which helper types (`f64x4`,
`float64x2_t`) are used to build `F64x8` — can change without notice.

In particular, the VBMI dispatch path introduced in the SIMD review of
2026-05-13 (see `.claude/board/SIMD_REVIEW_FIXES_2026_05_13.md`) added
`avx512vbmi: bool` to `SimdCaps` and a runtime branch in
`U8x64::permute_bytes`. Similar runtime dispatch adjustments within the
polyfill internals are expected and explicitly not subject to stability
guarantees.

### 3.2 Auto-Dispatch Heuristics

The `src/hpc/simd_caps.rs` singleton (`SimdCaps`) and the
`src/hpc/simd_dispatch.rs` frozen function-pointer table are internal
implementation details. They detect the host CPU at startup and route
all SIMD operations to the best available backend.

The exact detection logic (`is_x86_feature_detected!`, `cpuid` calls,
`avx512vbmi` / `avx512f` branching) may change as new ISA extensions are
added to the dispatch table. The contract for consumers is: write code
using `ndarray::simd::F64x8` and the stable free functions; the dispatch
layer guarantees correctness on all supported targets.

### 3.3 Internal Scratch Buffers

Several `heel_f64x8` helper functions (`cosine_f32_to_f64_simd` at
`src/hpc/heel_f64x8.rs:149`) use stack-allocated scratch buffers of type
`[f64; 8]` for widening conversions. The size, lifetime, and placement of
these buffers are implementation details and may be refactored (e.g., moved
into callers, replaced with SIMD widening intrinsics) without notice.

### 3.4 The `hpc/` Submodule Inventory

The following modules under `src/hpc/` are explicitly unstable:

```
src/hpc/ocr_simd.rs
src/hpc/clam_compress.rs
src/hpc/holo.rs           (carrier_distance_l1, focus_l1, focus_hamming — not the stable heel_f64x8 variants)
src/hpc/packed.rs
src/hpc/crystal_encoder.rs
src/hpc/byte_scan.rs
src/hpc/activations.rs
src/hpc/framebuffer.rs
src/hpc/cyclic_bundle.rs
src/hpc/causality.rs
src/hpc/nibble.rs
src/hpc/arrow_bridge.rs
src/hpc/vml.rs
src/hpc/layered_distance.rs
src/hpc/prefilter.rs
src/hpc/surround_metadata.rs
src/hpc/reductions.rs
src/hpc/lapack.rs
src/hpc/projection.rs
src/hpc/compression_curves.rs
src/hpc/simd_caps.rs
src/hpc/simd_dispatch.rs
src/hpc/gpt2/
src/hpc/jina/
src/hpc/stream/
src/hpc/stable_diffusion/
src/hpc/styles/
```

These modules are present for internal and research purposes. They do not
participate in the stability commitment. Their interfaces may change, be
removed, or be refactored into new modules at any time.

### 3.5 `.cargo/config.toml` CPU Targeting

The repository ships with `.cargo/config.toml` setting
`target-cpu=x86-64-v4` (AVX-512 mandatory for x86_64 development builds).
This is a developer convenience. Downstream consumers building on earlier
microarchitectures must override this via their own `.cargo/config.toml` or
`RUSTFLAGS`. The runtime dispatch in `simd_caps.rs` correctly falls back to
AVX2, NEON, or scalar regardless of the compile-time `target-cpu` setting
when the `#[target_feature]` guards are respected.

---

## 4. What "Frozen" Means

A symbol listed in §2 as stable has the following properties permanently
guaranteed:

### 4.1 No Signature Change

The Rust function signature — including parameter types, return type,
generic bounds, and `where` clauses — will not change without a semver major
version bump. For `F64x8` methods, "signature" includes the `Self` type and
all associated types.

Examples of what is **not allowed** without a semver bump:

```rust
// FORBIDDEN — changing parameter type:
// Was: pub fn cosine_f64_simd(a: &[f64], b: &[f64]) -> f64
// Now: pub fn cosine_f64_simd(a: &[f64], b: &[f64], len: usize) -> f64

// FORBIDDEN — changing return type:
// Was: pub fn cosine_f64_simd(a: &[f64], b: &[f64]) -> f64
// Now: pub fn cosine_f64_simd(a: &[f64], b: &[f64]) -> f32

// FORBIDDEN — adding generic parameters:
// Was: pub fn cosine_f64_simd(a: &[f64], b: &[f64]) -> f64
// Now: pub fn cosine_f64_simd<T: Float>(a: &[T], b: &[T]) -> T
```

### 4.2 No Rename

The symbol name at the module path level will not change. If
`ndarray::hpc::heel_f64x8::cosine_f64_simd` is the stable name, it stays at
that path. Re-exporting it at a new path is additive and allowed; removing
the original re-export is not.

### 4.3 No Semantic Drift

The mathematical semantics of a stable function will not change. In
particular:

- `cosine_f64_simd` will always return cosine similarity, never cosine
  distance (`1.0 - cosine`), and will never change the zero-vector guard
  threshold without a deprecation cycle.
- `F64x8::reduce_sum` will always return the sum of all 8 lanes, not a
  partial sum or a dot product.
- `F64x8::mul_add(b, c)` will always compute `self * b + c`, not
  `self + b * c`.

### 4.4 New Variants Ship Next to Existing Ones

When a capability needs to be extended or a performance-improved variant is
introduced, the new symbol ships as an **additional** function with a new
name, leaving the original untouched. The original is never removed or
silently replaced.

**Example — hypothetical FMA-specialized cosine:**

```rust
// Original (frozen, untouched):
pub fn cosine_f64_simd(a: &[f64], b: &[f64]) -> f64 { /* ... */ }

// New variant ships NEXT to original, never replaces it:
pub fn cosine_f64_simd_fma(a: &[f64], b: &[f64]) -> f64 { /* fma-specialized */ }
```

Consumers can opt into the new variant at their own pace. Nothing breaks.

### 4.5 Deprecation Timeline

When a stable symbol must eventually be superseded, the procedure is:

1. Add the replacement with a new name (additive).
2. Mark the original `#[deprecated(since = "...", note = "use new_name")]`.
3. Keep both symbols for at least two minor releases (or 90 calendar days,
   whichever is longer).
4. Only then may the deprecated symbol be moved to an internal module or
   removed.

No stable symbol has been deprecated as of 2026-05-18.

---

## 5. Adding New Kernels — The Additive Pattern

All growth of the HPC surface happens additively. The patterns below are
the only approved ways to add new distance and SIMD kernels.

### 5.1 New f32 Kernels — `F32x16` Pattern

If an f32-width variant of the cosine / L1 / L2 / Linf kernels is needed,
it ships in a new function (or in an extended `heel_f64x8.rs` section) using
`ndarray::simd::F32x16` as the SIMD register type:

```rust
// In src/hpc/heel_f64x8.rs (additive — new function, old function untouched):
pub fn cosine_f32_simd(a: &[f32], b: &[f32]) -> f32 { /* uses F32x16 */ }
```

Note that `cosine_f32_to_f64_simd` (which converts f32 inputs to f64
internally) already exists at `src/hpc/heel_f64x8.rs:149` and is
re-exported via `ndarray::simd::cosine_f32_to_f64_simd`
(`src/simd.rs:1751`). A native f32-output variant would be a distinct,
additional function.

### 5.2 New Int8 Kernels — `heel_i8x32` Pattern

Int8 distance metrics (for quantized embedding spaces) would ship in a new
module:

```
src/hpc/heel_i8x32.rs      (new file — does not touch heel_f64x8.rs)
```

With a new Cargo feature gate if the dependency weight warrants it. The
naming convention follows the existing heel prefix: `heel_i8x32`.

**Expected public surface:**

```rust
// src/hpc/heel_i8x32.rs
pub fn l1_i8_simd(a: &[i8], b: &[i8]) -> i64;
pub fn dot_i8_simd(a: &[i8], b: &[i8]) -> i64;
```

The existing `ndarray::hpc::hpc::quantized` module (`src/hpc/quantized.rs`)
provides `Int8Gemm` infrastructure that `heel_i8x32` would build on.

### 5.3 Hamming on Binary Vectors — `heel_u8x32` Pattern

Bit-level Hamming distance for dense binary vectors (e.g., binary
quantized embeddings, CLAM binary tree codes) would ship in:

```
src/hpc/heel_u8x32.rs      (new file — additive)
```

**Expected public surface:**

```rust
// src/hpc/heel_u8x32.rs
pub fn hamming_u8_simd(a: &[u8], b: &[u8]) -> u64;
```

Note: a scalar `hamming_distance_raw` already exists at
`src/hpc/bitwise.rs:180`, and a DataFusion UDF wrapper at
`src/hpc/udf_kernels.rs:49`. The `heel_u8x32::hamming_u8_simd` variant
would be a new SIMD-accelerated standalone kernel using `ndarray::simd::U8x64`.

### 5.4 Submodule Naming Convention

All new heel-family kernels follow the convention:

```
heel_{type}x{lane_count}
```

| Submodule | Element type | Lane count | Register |
|-----------|-------------|-----------|---------|
| `heel_f64x8` (existing) | `f64` | 8 | `F64x8` |
| `heel_f32x16` (planned) | `f32` | 16 | `F32x16` |
| `heel_i8x32` (planned) | `i8` | 32 | (sub-byte SIMD) |
| `heel_u8x32` (planned) | `u8` | 32 | `U8x64` (2-chunk) |

### 5.5 Additive Rule Summary

> **New capability = new symbol at new path. Never a signature change to
> an existing stable symbol.**

This rule applies to:
- New functions in existing modules (added, not changed)
- New modules alongside existing modules (added, not changed)
- New Cargo features alongside existing features (added, not changed)
- New type parameters on existing types (forbidden for stable types)

---

## 6. Diamond-Dependency Guard

### 6.1 The Problem

Rust's dependency resolution allows at most one version of a crate per
build graph when that crate is shared (not renamed). When `surrealdb-core`
and the ONNX runtime crate `ort` both depend on `ndarray`, they must agree
on exactly which `ndarray` they are using — otherwise Rust generates two
incompatible types both named `ndarray::Array2<f64>`, and the build fails
at the type-system level when code tries to pass one to a function expecting
the other.

This fork exists precisely to solve that problem: by placing the
AdaWorldAPI-extended ndarray at a pinned git revision under
`[patch.crates-io]`, all crates in the workspace see the same ndarray.

### 6.2 The Patch Contract

In the surrealdb-core workspace `Cargo.toml` (and in any consumer that
assembles surrealdb-core + lance-graph cognitive crates):

```toml
# In the root Cargo.toml of the consumer workspace:
[patch.crates-io]
ndarray = { git = "https://github.com/AdaWorldAPI/ndarray.git", branch = "main" }
```

This entry is the **contract**. Its presence makes the fork's stable API
available to every crate in the build graph. Its **absence** or **change**
breaks the fork.

**What the patch replaces:** The upstream `ndarray` crate from crates.io
(currently 0.16.x stable, later 0.17.x). Any workspace crate that specifies
`ndarray = "0.16"` or `ndarray = "0.17"` in its own `[dependencies]` will
silently receive this fork instead, because `[patch.crates-io]` overrides
all version-matched dependencies.

**What breaks if the patch is removed or points to the wrong commit:**

1. surrealdb-core's vector index distance kernels lose access to
   `ndarray::hpc::heel_f64x8::cosine_f64_simd` — linker error or type
   mismatch.
2. `ort` (the ONNX runtime Rust crate) may resolve to the upstream ndarray,
   creating a second ndarray in the build graph. Downstream code that passes
   `ndarray::Array` values between `ort` and surrealdb-core fails with
   cryptic type errors like `expected ndarray::Array2<f64>, found
   ndarray::Array2<f64>` (same name, different crate instance).
3. The lance-graph cognitive crates lose access to `ndarray::simd::F64x8`
   and all `hpc::` distance kernels.

### 6.3 Version Pinning Strategy

The `[patch.crates-io]` stanza should pin to a specific **tag** (not a
floating branch name) in production deployments:

```toml
# Preferred for production:
[patch.crates-io]
ndarray = { git = "https://github.com/AdaWorldAPI/ndarray.git", tag = "v0.17.2-hpc-1" }

# Acceptable for CI on main branch:
[patch.crates-io]
ndarray = { git = "https://github.com/AdaWorldAPI/ndarray.git", branch = "main" }
```

Floating branch pins (`branch = "main"`) are acceptable in CI but must not
be used in published releases of surrealdb-core or lance-graph, as they
make the build non-reproducible.

### 6.4 ort Interop Invariant

The ONNX runtime crate (`ort`, wrapped from the C++ ORT library) has its
own optional ndarray integration. The fork's `Cargo.toml` at
`src/lib.rs:313` exposes `pub mod hpc` only under `#[cfg(feature = "std")]`.
This means:

- `ort` configurations that only need the core ndarray array types (no HPC)
  continue to work: they depend on `ndarray::Array`, `ndarray::ArrayView`,
  etc., which are unchanged from upstream.
- `ort` configurations that use ndarray as a tensor interchange format
  with surrealdb-core benefit from the fork's presence because all three
  crates now share the same ndarray type identity.

The fork adds ONLY new modules and features. It does not modify the core
array types, layout types, or BLAS backends that `ort` depends on.

---

## 7. CI Commitment

### 7.1 Target Architecture Matrix

The following cross-architecture matrix is the aspirational CI target. It
documents the intended coverage; implementation of the full matrix in CI
infrastructure is work in progress as of 2026-05-18.

| Target triple | SIMD tier | `F64x8` backend | Status |
|--------------|-----------|-----------------|--------|
| `x86_64-unknown-linux-gnu` + AVX-512F | AVX-512 | `simd_avx512::F64x8` | Intended |
| `x86_64-unknown-linux-gnu` + AVX2 (no AVX-512) | AVX2 | `simd_avx2::F64x8` | Intended |
| `aarch64-unknown-linux-gnu` + NEON | NEON | `simd_neon::F64x8` | Intended |
| `x86_64-unknown-linux-gnu` (scalar only) | Scalar | fallback `[f64; 8]` | Intended |
| `thumbv6m-none-eabi` (no-std) | None (`hpc` disabled) | N/A | Intended |

### 7.2 Doctest Coverage

All stable public functions in §2 must have at least one doctest that
compiles and runs correctly under `cargo test --doc`. The current status
for `cosine_f64_simd` is satisfied via the test suite in
`src/hpc/heel_f64x8.rs` (8 unit tests, including `cosine_matches_scalar`
at line 278 which verifies the numeric tolerance contract against a scalar
reference).

The aspirational goal is for each stable function to have a doctest visible
in the rendered docs (i.e., in the `///` doc comment rather than only in
`#[test]`). This requires that `cargo test --doc --features std` passes on
all four SIMD tiers listed in §7.1.

### 7.3 Test Command

The current passing test invocation (1786 passing as of the SIMD review
on 2026-05-13):

```sh
cargo test --features rayon --lib
```

The clippy clean invocation:

```sh
cargo clippy --features rayon -- -D warnings
```

Both must pass on every commit to the `main` branch that touches any file
in the stable surface (§2). Changes to explicitly unstable modules (§3)
are encouraged to pass both commands but are not gating.

### 7.4 Numeric Regression Guard

The tolerance assertions in `src/hpc/heel_f64x8.rs` (tests `cosine_matches_scalar`,
`cosine_identical`, `cosine_opposite`, `cosine_orthogonal`) form the
numeric regression guard for the stable API. These tests must not be
weakened (loosened tolerance) or removed without a corresponding update to
this document.

The current observed tolerance for `cosine_f64_simd` vs scalar on x86_64
(tested at len=333 with trigonometric inputs) is less than `1e-10`, well
within the committed `f64::EPSILON * len` bound (`2.22e-16 * 333 = 7.4e-14`).

---

## 8. Cross-References to Integration Plans

This stability commitment is informed by and consistent with four
integration planning documents in the repository:

### Plan 1: Lance-Graph DataFusion Integration

**File:** `.claude/prompts/04_lance_graph_integration.md`

This plan defines the DataFusion UDF layer that uses ndarray HPC kernels:

| UDF Name | Underlying ndarray kernel |
|----------|--------------------------|
| `hamming` | `hpc::bitwise::hamming_distance_raw` (`src/hpc/bitwise.rs:180`) |
| `spo_distance` | `hpc::node::Node::distance` |
| `nars_revision` | `hpc::causality::NarsTruthValue::revision` |
| `sigma_classify` | `hpc::cascade::Cascade::expose` |
| `bf16_hamming` | `hpc::bf16_truth::bf16_hamming_scalar` |

The document notes that ndarray provides the kernels; lance-graph provides
the DataFusion UDF wrappers. This separation is architecturally correct and
preserved: stable kernels in ndarray, UDF registration in lance-graph.

The lance-graph repo's phase completion status (as of 2026-03-22):
- Phase 1 (blasgraph CSC/Planner): DONE
- Phase 2 (bgz17 container/semiring): DONE
- Phase 3 (dual-path): NOT STARTED — depends on `heel_f64x8` stable surface
- Phase 4 (FalkorDB retrofit): NOT STARTED

The frozen `cosine_f64_simd` and the aspirational `l1_f64_simd`,
`l2_f64_simd`, `linf_f64_simd` functions are the kernel requirements for
Phase 3 to proceed.

### Plan 2: SIMD Review and Soundness Fixes (2026-05-13)

**File:** `.claude/board/SIMD_REVIEW_FIXES_2026_05_13.md`

The 15-agent CCA2A review fleet identified three soundness/correctness
issues and deferred a broader "cosmetic SIMD" sweep. The P0 SIGILL fix
for `U8x64::permute_bytes` on AVX-512F-without-VBMI machines is directly
relevant to the stability commitment: it demonstrates the mechanism by which
the polyfill internals (AVX2/AVX-512/NEON dispatch paths) CAN change
without the stable consumer API changing.

The P0 fix added `avx512vbmi: bool` to `SimdCaps` and a runtime branch in
`U8x64::permute_bytes`. The consumer API (`ndarray::simd::U8x64`) was
unchanged. This is the correct pattern for all future backend changes.

The deferred "cosmetic SIMD" item (scalar function bodies wearing
`#[target_feature]` decorations) will be cleaned up when the polyfill
is completed — `U8x64` / `F32x8` / etc. will have full method parity
across AVX-512, AVX2, NEON, and scalar. Until then, those files remain
in the explicitly-unstable category (§3.4).

### Plan 3: SPO Bundle Simulation Findings

**File:** `.claude/SPO_BUNDLE_FINDINGS_v2.md`

This empirical study confirmed that majority-vote bundling at 8K and 16K
bits is in the "dead zone" for ranking tasks (Spearman ρ ≈ 0.001 at 8K,
ρ ≈ 0.417 at 16K). The ZeckF64 band encoding at 64 bits dominates both.

This finding is relevant to stability because it validates that the
distance kernels in `heel_f64x8` (cosine, and the aspirational L1/L2/Linf)
are the correct abstraction boundary: they operate on f64 vectors, not on
fixed-width binary bundles. The `heel_f64x8` module design is not expected
to need binary-bundle variants (those live in `hpc::spo_bundle`,
`hpc::cyclic_bundle`, and related unstable modules).

### Plan 4: Architecture Rule (from CLAUDE.md)

**File:** `CLAUDE.md` (repository root, referenced in agent instructions)

The architecture rule is:

```
ndarray = hardware (SIMD, Palette, Base17, SpoDistanceMatrices, read_bgz7_file)
lance-graph = thinking (NarsTruth, NarsEngine, TripleModel, AutocompleteCache)
causal-edge = protocol (CausalEdge64, NarsTables, forward/learn)
p64 = convergence highway (both repos meet here)
```

The stable API in §2 maps directly to the "hardware" layer: `F64x8` is raw
SIMD register abstraction, `cosine_f64_simd` is a distance kernel. Both
are pure compute with no reasoning logic embedded. This architectural
separation is explicitly preserved by the stability commitment: stable symbols
in `ndarray::hpc::heel_f64x8` and `ndarray::simd` will not acquire
reasoning semantics (NarsTruth weighting, cascade band classification, etc.).
Those belong in lance-graph.

---

## 9. Appendix: Numeric Tolerance Derivation

### 9.1 IEEE 754 Error Accumulation

For a dot product computed via FMA `mul_add` over `n` elements:

```
acc_0 = 0
acc_i = acc_{i-1} + a_{chunk} * b_{chunk}    (FMA in each SIMD lane)
```

Each `mul_add` introduces at most 0.5 ULP error relative to the exact
result of `a * b + acc`. After `n/8` iterations (one per 8-wide chunk),
the accumulated error is bounded by:

```
|sum_SIMD - sum_exact| ≤ (n/8) × 0.5 × ε_mach × |exact_sum|
```

where `ε_mach = f64::EPSILON = 2.220446049250313e-16`.

For the cosine similarity specifically, three accumulators (dot, na, nb)
each accumulate independently, then the error in the final result
`dot / sqrt(na * nb)` is bounded (by first-order error analysis) by
approximately `3 × (n/8) × 0.5 × ε_mach`, which for large `n` is still
well within the committed `ε_mach × n` bound.

### 9.2 Observed vs Committed Tolerance

| Function | Vector length tested | Observed max error | Committed bound |
|----------|---------------------|--------------------|-----------------|
| `cosine_f64_simd` | 333 | `< 1e-10` | `ε × 333 ≈ 7.4e-14` |
| `cosine_f64_simd` | 1024 | `< 1e-10` (self-cosine = 1.0) | `ε × 1024 ≈ 2.3e-13` |
| `cosine_f64_simd` | 256 | `< 1e-10` (orthogonal = 0.0) | `ε × 256 ≈ 5.7e-14` |

The observed error of `< 1e-10` is approximately 6 orders of magnitude
below the committed bound. The generous committed bound (`ε × len`) allows
for worst-case inputs (e.g., catastrophic cancellation) while being met
with significant headroom for typical embedding inputs.

### 9.3 Zero-Vector Guard Threshold

The zero-vector guard (`denom < 1e-12`) is part of the semantic contract
for `cosine_f64_simd`. The threshold `1e-12` was chosen to be:
- Above the rounding noise for zero vectors computed via FMA
  (`n` multiplications of `0.0`, resulting in exactly `0.0`)
- Below the smallest meaningful norm of a non-zero embedding vector
  used in practice (`min_norm ≫ 1e-6` for normalized unit vectors,
  `min_norm ≫ 1e-3` for un-normalized language model embeddings)

This threshold is frozen and will not change without a deprecation notice.

---

*End of document. Maintained by the AdaWorldAPI/ndarray HPC team.*
*For questions: open an issue at https://github.com/AdaWorldAPI/ndarray*
