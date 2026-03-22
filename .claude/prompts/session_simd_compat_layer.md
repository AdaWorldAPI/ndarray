# SESSION: Port SIMD Compat Layer to ndarray

## Mission

Port rustynum's `simd_avx512.rs` (2643 lines) portable SIMD type system into
ndarray as `src/backend/simd_compat.rs`. Then refactor `kernels_avx512.rs` to
use the compat types instead of raw `__m512` intrinsics.

Zero runtime cost. Same instructions. Unlocks: aarch64/NEON, std::simd migration,
simpler kernel authoring, and 5 of 10 Pumpkin SIMD features.

## READ FIRST

```bash
# The source — rustynum's compat layer
cat <rustynum>/rustynum-core/src/simd_avx512.rs   # 2643 lines, 11 types, 60 impl blocks

# The target — ndarray's current raw intrinsics
cat <ndarray>/src/backend/kernels_avx512.rs        # 962 lines, raw __m512 everywhere
cat <ndarray>/src/backend/native.rs                # dispatch! macro, Tier enum
cat <ndarray>/src/backend/mod.rs                   # BlasFloat trait
cat <ndarray>/src/hpc/bitwise.rs                   # Hamming dispatch, also raw intrinsics
```

## What Exists in rustynum

11 types wrapping stable `core::arch::x86_64` intrinsics:

```
TYPE            BACKING           WIDTH   LANES   PURPOSE
──────────      ──────────        ─────   ─────   ───────
F32x16          __m512            512     16      Float SIMD (GEMM, dot, exp, etc.)
F32Mask16       __mmask16         16      16      Comparison results for F32x16
F64x8           __m512d           512     8       Double SIMD (dgemm, ddot, etc.)
F64Mask8        __mmask8          8       8       Comparison results for F64x8
U8x64           __m512i           512     64      Byte SIMD (Hamming, popcount, BNN)
I32x16          __m512i           512     16      Int32 SIMD (gather indices, quantized)
I64x8           __m512i           512     8       Int64 SIMD (addresses, counters)
U32x16          __m512i           512     16      Uint32 SIMD (bit ops, shifts)
U64x8           __m512i           512     8       Uint64 SIMD (fingerprint words)
F32x8           __m256            256     8       AVX2 fallback float
F64x4           __m256d           256     4       AVX2 fallback double
```

Key methods on F32x16 (representative):
```
splat(v) → F32x16              broadcast scalar to all lanes
from_array([f32; 16]) → Self   load from array
to_array(self) → [f32; 16]     store to array
copy_to_slice(self, &mut [f32]) store to slice
reduce_sum(self) → f32         horizontal sum
reduce_min/max(self) → f32     horizontal min/max
simd_min/max(self, other)      lane-wise min/max
simd_clamp(self, lo, hi)       lane-wise clamp
mul_add(self, b, c) → Self     fused multiply-add (FMA)
sqrt/round/floor/abs(self)     lane-wise math
to_bits(self) → U32x16         reinterpret as uint
cast_i32(self) → I32x16        convert to int
simd_eq/ne/lt/le/gt/ge         comparison → mask
select(mask, true, false)      masked select (blend)
```

Operator overloads: Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign.
For integer types: also BitXor, BitAnd, BitOr, Not, Shr, Shl.

## DELIVERABLE 1: simd_compat.rs (new file)

Create `src/backend/simd_compat.rs`:

```rust
//! Portable SIMD types backed by stable core::arch intrinsics.
//!
//! Mirrors std::simd API surface (types, operators, methods) using
//! stable #[target_feature] functions. Zero runtime cost — everything
//! inlines to the same instructions as raw intrinsics.
//!
//! When std::simd stabilizes, replace this file with re-exports.
//! When aarch64 support is needed, add #[cfg(target_arch = "aarch64")]
//! backing using NEON intrinsics (F32x16 → 4× float32x4_t).
```

Port from `rustynum-core/src/simd_avx512.rs`. Changes from rustynum:
- Remove `use rustynum_core::` dependencies (none exist — it's self-contained)
- Keep all 11 types, all operator impls, all methods
- Add `#[cfg(target_arch = "x86_64")]` gate on the module
- Add scalar fallback stubs for non-x86 (`#[cfg(not(target_arch = "x86_64"))]`)
  that use `[f32; 16]` arrays instead of `__m512` — correct but slow
- Wire into `backend/mod.rs` as `pub(crate) mod simd_compat;`

## DELIVERABLE 2: Refactor kernels_avx512.rs

Replace raw `__m512` with compat types. Example:

```rust
// BEFORE (raw intrinsics):
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let mut acc0 = _mm512_setzero_ps();
    // ...
    while i + 64 <= n {
        unsafe {
            acc0 = _mm512_fmadd_ps(
                _mm512_loadu_ps(x[i..].as_ptr()),
                _mm512_loadu_ps(y[i..].as_ptr()),
                acc0
            );
        }
        i += 16;
    }
    _mm512_reduce_add_ps(acc0 + acc1 + acc2 + acc3)
}

// AFTER (compat types):
pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
    let mut acc0 = F32x16::splat(0.0);
    // ...
    while i + 64 <= n {
        acc0 = F32x16::from_slice(&x[i..]).mul_add(
            F32x16::from_slice(&y[i..]),
            acc0
        );
        i += 16;
    }
    (acc0 + acc1 + acc2 + acc3).reduce_sum()
}
```

Same instructions emitted. Reads like math, not like intrinsics.

Refactor all 962 lines. Count of `__m512`/`__m256`/`__mmask` references to replace:
```bash
grep -c "__m512\|__m256\|__mmask" src/backend/kernels_avx512.rs
# Should be ~100-150 references → all become compat type names
```

## DELIVERABLE 3: Refactor bitwise.rs SIMD paths

`bitwise.rs` has 3 inline SIMD functions using raw intrinsics:
- `hamming_avx2()` — uses `__m256i`, `_mm256_*`
- `hamming_avx512bw()` — uses `__m512i`, `_mm512_*`
- `popcount_avx512bw()` — uses `__m512i`, `_mm512_*`

Refactor to use `U8x64` (for AVX-512) and a future `U8x32` (for AVX2).
The dispatch functions (`dispatch_hamming`, `dispatch_popcount`) stay as-is —
they just call the refactored functions.

## DELIVERABLE 4: Scalar Fallback Stubs

For non-x86 architectures, provide array-backed fallbacks:

```rust
#[cfg(not(target_arch = "x86_64"))]
pub struct F32x16([f32; 16]);

#[cfg(not(target_arch = "x86_64"))]
impl F32x16 {
    pub fn splat(v: f32) -> Self { Self([v; 16]) }
    pub fn reduce_sum(self) -> f32 { self.0.iter().sum() }
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        let mut r = [0.0f32; 16];
        for i in 0..16 { r[i] = self.0[i] * b.0[i] + c.0[i]; }
        Self(r)
    }
    // ... etc
}
```

This lets ALL kernel code compile on aarch64/riscv — just slowly.
NEON-accelerated backing comes later as a separate PR.

## DELIVERABLE 5: Wire activations.rs + vml.rs Through Backend

Currently scalar:
```rust
// activations.rs
fn sigmoid(&self) -> Array<A, Ix1> {
    self.mapv(|v| A::one() / (A::one() + (-v).exp()))  // SCALAR
}

// vml.rs
pub fn vsexp(x: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(x.iter()) { *o = v.exp(); }  // SCALAR
}
```

After compat layer, wire through backend:
```rust
// vml.rs (SIMD-dispatched)
pub fn vsexp(x: &[f32], out: &mut [f32]) {
    // Process 16 elements at a time via F32x16
    let mut i = 0;
    while i + 16 <= x.len() {
        let v = F32x16::from_slice(&x[i..]);
        // Polynomial approximation of exp() using mul_add chains
        let result = simd_exp_f32(v);
        result.copy_to_slice(&mut out[i..]);
        i += 16;
    }
    // Scalar tail
    for j in i..x.len() { out[j] = x[j].exp(); }
}
```

Blueprint: rustynum array_struct.rs has 35 `Ops::` dispatch calls showing
exactly which operations need SIMD paths (dot, exp, log, sigmoid, softmax,
cosine_similarity, norm, min, max, sum, l1/l2_norm, add/sub/mul/div_scalar).

## TESTS

1. Every compat type: `from_array → to_array` roundtrip = identity
2. `F32x16::splat(x).reduce_sum() == x * 16.0`
3. `F32x16 a + b == F32x16::from_array(element_wise_add(a, b))`
4. `mul_add` matches `a * b + c` within 1 ULP (FMA rounding)
5. Operator overloads: `a + b`, `a * b`, `a - b`, `a / b` all correct
6. Mask operations: `simd_lt` + `select` produces correct blend
7. Refactored `dot_f32` produces identical results to pre-refactor
8. Refactored `sgemm_blocked` produces identical results
9. Refactored `hamming_avx512bw` produces identical results
10. Scalar fallback: same results as SIMD (on x86, test both paths)
11. `vsexp` SIMD matches scalar within 2 ULP for range [-10, 10]

## CONSTRAINTS

1. **Zero runtime cost.** Profile before/after. GEMM throughput must not regress.
2. **All tests pass.** `cargo test` must produce identical results.
3. **No new dependencies.** The compat layer uses only `core::arch` (stable).
4. **std::simd API alignment.** Use the same method names as nightly `std::simd`
   so future migration is a file swap, not a rewrite.
5. **`#[inline(always)]` on every method.** The compiler MUST inline these.
   A function call boundary on a SIMD type defeats the purpose.

## OUTPUT

Branch: `feat/simd-compat-layer`
Files created: `src/backend/simd_compat.rs`
Files modified: `src/backend/kernels_avx512.rs`, `src/backend/mod.rs`,
               `src/hpc/bitwise.rs`, `src/hpc/vml.rs`, `src/hpc/activations.rs`
Run: `cargo test && cargo bench` — verify no regression
