# SESSION: Cleanup simd_compat Hallucination + Move Real Polyfill

## Background

A previous session wrote a prompt describing ndarray's SIMD architecture
WITHOUT READING THE SOURCE. That hallucinated prompt was executed by another
session, creating `src/backend/simd_compat.rs` (1072 lines) and rewiring
4 files to import from it. The created code is NOT the real polyfill.

The REAL `std::simd` polyfill lives in rustynum and has been running in
production with tests passing. This session cleans up the mess and replaces
it with the real thing.

## STEP 1: Read Everything Before Touching Anything

```bash
# The DAMAGE (what was hallucinated)
cat src/backend/simd_compat.rs              # 1072 lines, WRONG
grep -rn "simd_compat" src/ --include="*.rs"  # 4 files import from it

# The REAL polyfill (in rustynum, working)
cat <rustynum>/rustynum-core/src/simd.rs           # Router
cat <rustynum>/rustynum-core/src/simd_avx512.rs    # 2643 lines, 11 types
cat <rustynum>/rustynum-core/src/simd_avx2.rs      # AVX2 tier

# Verify: what tests exist for simd_compat?
grep -n "#\[test\]" src/backend/simd_compat.rs
```

## STEP 2: Audit the Damage

The hallucinated `simd_compat.rs` has these problems:

### Missing 7 of 11 types
```
HAS:     F32x16, F32Mask16, F64x8, F64Mask8          (4 types)
MISSING: U8x64, I32x16, I64x8, U32x16, U64x8,       (7 types)
         F32x8, F64x4

U8x64  = Hamming/popcount/BNN — CRITICAL for bgz17
I32x16 = gather indices — needed for palette lookup
U64x8  = fingerprint words — CRITICAL for Hamming
F32x8  = AVX2 float — the ENTIRE AVX2 tier
F64x4  = AVX2 double
```

### No router
```
simd_compat: #[cfg(target_arch = "x86_64")] at compile time
             → AVX2-only machines get SCALAR, not AVX2
             → No runtime CPU detection

rustynum:    One-time CPU detect → picks avx512 OR avx2
             → AVX2 machines get AVX2 types
```

### No AVX2 tier at all
The hallucinated file has AVX-512 + scalar. No AVX2.
Machines without AVX-512 fall directly to scalar loops.

### Missing F32x16 methods
```
rustynum has:    to_bits(), from_bits(), cast_i32()
simd_compat:     missing all three
```

### Bespoke simd_exp_f32 / pow2n_from_int
These were invented, not copied from rustynum.
May produce different numerical results.

### Files that were modified
```
src/backend/simd_compat.rs    ← CREATED (1072 lines, wrong)
src/backend/mod.rs            ← MODIFIED (added pub(crate) mod simd_compat)
src/backend/kernels_avx512.rs ← REWRITTEN: BLAS-1 + element-wise now use
                                 F32x16/F64x8 from simd_compat.
                                 GEMM + Hamming UNTOUCHED (still raw intrinsics).
src/hpc/activations.rs        ← MODIFIED: added sigmoid_f32, softmax_f32,
                                 log_softmax_f32 using simd_compat types.
src/hpc/vml.rs                ← REWRITTEN: vsexp, vssqrt, vsabs, vsadd, vsmul,
                                 vsdiv now use F32x16 from simd_compat.
```

## STEP 3: Move Real Polyfill

Copy from rustynum to ndarray at crate root level:

```
rustynum-core/src/simd.rs           → ndarray/src/simd/mod.rs
rustynum-core/src/simd_avx512.rs    → ndarray/src/simd/avx512.rs
rustynum-core/src/simd_avx2.rs      → ndarray/src/simd/avx2.rs
```

Wire in `src/lib.rs`: `pub mod simd;`

Adjust internal imports in the moved files:
- Remove any `use rustynum_core::` or `use crate::` that point to rustynum internals
- The simd module should be self-contained (it already is in rustynum)

## STEP 4: Rewire Imports

Change all 4 files from hallucinated path to real path:

```
BEFORE:  use crate::backend::simd_compat::{F32x16, F64x8};
AFTER:   use crate::simd::{F32x16, F64x8};

BEFORE:  use crate::backend::simd_compat::{simd_exp_f32, F32x16};
AFTER:   use crate::simd::{F32x16};
         (simd_exp_f32 may need to come from rustynum's polyfill or be
          re-implemented using the real types — check if rustynum has it)
```

Files to change:
1. `src/backend/kernels_avx512.rs` line 17
2. `src/hpc/activations.rs` line 5
3. `src/hpc/vml.rs` line 7
4. `src/backend/mod.rs` line 16 (remove `pub(crate) mod simd_compat;`)

## STEP 5: Delete Hallucinated File

```bash
rm src/backend/simd_compat.rs
```

Verify no remaining references:
```bash
grep -rn "simd_compat" src/ --include="*.rs"
# Must return ZERO results
```

## STEP 6: Verify Kernel Code Still Works

The BLAS-1 and element-wise kernels in `kernels_avx512.rs` were rewritten
to use F32x16/F64x8 from simd_compat. The API surface of the hallucinated
types is SIMILAR to rustynum's real types (same method names for the methods
that exist). Check each kernel function:

```bash
# Extract all F32x16/F64x8 method calls in kernels_avx512.rs
grep -o "F32x16::[a-z_]*\|F64x8::[a-z_]*\|\.reduce_sum\|\.mul_add\|\.copy_to_slice\|\.from_slice\|\.splat\|\.sqrt\|\.abs" \
    src/backend/kernels_avx512.rs | sort -u
```

Verify each of these methods exists in rustynum's `simd_avx512.rs`.
If any are missing, add them or rewrite the kernel to not use them.

## STEP 7: Handle simd_exp_f32

The hallucinated file invented `simd_exp_f32()` and `pow2n_from_int()`.
These are used by `activations.rs` and `vml.rs`.

Check if rustynum has equivalent:
```bash
grep "exp\|pow2" <rustynum>/rustynum-core/src/simd_avx512.rs
```

If not: the `simd_exp_f32` from simd_compat is actually reasonable code
(Remez polynomial approximation). Extract it into a separate file
`src/simd/math.rs` that uses the real F32x16 type. Don't lose the exp
implementation — just make it use the right types.

## STEP 8: Verify GEMM + Hamming Are Untouched

The GEMM (sgemm_blocked, dgemm_blocked) and Hamming (hamming_distance,
popcount, dot_i8, hamming_batch) functions in kernels_avx512.rs still
use raw `core::arch::x86_64::*` intrinsics. They were NOT converted.

Verify they still compile and produce correct results.
These will be converted to use `crate::simd::` types LATER,
in a SEPARATE session, AFTER the polyfill is verified working.

## STEP 9: Run Tests

```bash
cargo test 2>&1 | tail -20
# Must pass. The 820 tests that passed with simd_compat
# should still pass with the real polyfill.
```

If tests fail, the failure is in the import rewiring (Step 4)
or in API differences between hallucinated and real types (Step 6).

## OUTPUT

```
DELETED:  src/backend/simd_compat.rs
CREATED:  src/simd/mod.rs, src/simd/avx512.rs, src/simd/avx2.rs
MODIFIED: src/lib.rs (add pub mod simd)
MODIFIED: src/backend/mod.rs (remove simd_compat reference)
MODIFIED: src/backend/kernels_avx512.rs (change import path)
MODIFIED: src/hpc/activations.rs (change import path)
MODIFIED: src/hpc/vml.rs (change import path)
MAYBE:    src/simd/math.rs (if simd_exp_f32 needs a home)
```

When `std::simd` stabilizes in core:
  `use crate::simd::` → `use std::simd::`
  Delete `src/simd/`. Done. Every kernel unchanged.
