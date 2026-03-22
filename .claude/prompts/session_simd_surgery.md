# SESSION: Open Heart Surgery — simd_compat → Real std::simd Polyfill

## Context

ndarray has `src/backend/simd_compat.rs` (1072 lines) that was created
by a session working from a prompt that didn't read the source architecture.
The code works (820 tests pass) but may not match the intended design.

The REAL `std::simd` polyfill lives in rustynum and has been running in
production. It has a different architecture (router, AVX2 tier, 11 types).

Your job: read all three codebases, save snapshots, and produce a CORRECT
third version that combines the working new code with the real architecture.

## STEP 1: Save Snapshots (before ANY changes)

```bash
# Save "AFTER" — current ndarray state (the working code under review)
mkdir -p /tmp/simd_surgery/after
cp src/backend/simd_compat.rs     /tmp/simd_surgery/after/
cp src/backend/kernels_avx512.rs  /tmp/simd_surgery/after/
cp src/backend/mod.rs             /tmp/simd_surgery/after/
cp src/hpc/activations.rs         /tmp/simd_surgery/after/
cp src/hpc/vml.rs                 /tmp/simd_surgery/after/

# Save "BEFORE" — pre-simd_compat state from git
PRE_COMMIT="22bfb7a"  # commit before simd_compat was introduced
mkdir -p /tmp/simd_surgery/before
git show ${PRE_COMMIT}:src/backend/kernels_avx512.rs  > /tmp/simd_surgery/before/kernels_avx512.rs
git show ${PRE_COMMIT}:src/backend/mod.rs             > /tmp/simd_surgery/before/mod.rs
git show ${PRE_COMMIT}:src/hpc/activations.rs         > /tmp/simd_surgery/before/activations.rs
git show ${PRE_COMMIT}:src/hpc/vml.rs                 > /tmp/simd_surgery/before/vml.rs

# Save "REAL" — rustynum's actual polyfill (the source of truth for architecture)
mkdir -p /tmp/simd_surgery/real
cp <rustynum>/rustynum-core/src/simd.rs          /tmp/simd_surgery/real/
cp <rustynum>/rustynum-core/src/simd_avx512.rs   /tmp/simd_surgery/real/
cp <rustynum>/rustynum-core/src/simd_avx2.rs     /tmp/simd_surgery/real/
```

## STEP 2: Read All Three (before writing ANYTHING)

Read these files IN FULL. Do not skim. Do not summarize from memory.

```bash
# The REAL polyfill (rustynum) — this is the architectural source of truth
cat /tmp/simd_surgery/real/simd.rs
cat /tmp/simd_surgery/real/simd_avx512.rs
cat /tmp/simd_surgery/real/simd_avx2.rs

# The BEFORE state (ndarray before simd_compat)
cat /tmp/simd_surgery/before/kernels_avx512.rs
cat /tmp/simd_surgery/before/activations.rs
cat /tmp/simd_surgery/before/vml.rs

# The AFTER state (ndarray with simd_compat, currently working, 820 tests)
cat /tmp/simd_surgery/after/simd_compat.rs
cat /tmp/simd_surgery/after/kernels_avx512.rs
cat /tmp/simd_surgery/after/activations.rs
cat /tmp/simd_surgery/after/vml.rs
```

## STEP 3: Answer These Questions (in writing, before any code changes)

### About the REAL polyfill (rustynum):
1. How many types does simd_avx512.rs define? List them all.
2. How does simd.rs (the router) work? What does it do at init?
3. What does simd_avx2.rs provide? Same types? Different backing?
4. What happens on an AVX2-only machine?
5. What is the module structure? How does `crate::simd::F32x16` resolve?

### About simd_compat.rs (the working but possibly wrong code):
6. How many types does it define? Which are missing vs rustynum?
7. Does it have a router? How does it pick AVX-512 vs fallback?
8. Does it have an AVX2 tier?
9. What functions does it add that rustynum doesn't have? (simd_exp_f32? others?)
10. Are there API differences for the types it DOES have vs rustynum's versions?

### About the BEFORE kernels_avx512.rs:
11. What SIMD approach did it use? Raw intrinsics? Wrapper types?
12. What functions does it contain? (BLAS-1, element-wise, GEMM, Hamming?)

### About the AFTER kernels_avx512.rs:
13. Which functions were changed to use simd_compat types?
14. Which functions were left with raw intrinsics? Why?
15. Do the changed functions produce the same results?

### About activations.rs BEFORE vs AFTER:
16. What existed before? What was added?
17. Is the added code (sigmoid_f32, softmax_f32, etc.) correct?
18. Does rustynum have equivalent SIMD activation functions?

### About vml.rs BEFORE vs AFTER:
19. What was scalar before? What got SIMD paths?
20. Are the SIMD paths correct?

### The synthesis question:
21. If the original prompt had correctly described rustynum's architecture,
    what would the result look like? Specifically:
    - Where would the polyfill live? (src/backend/? src/simd/? other?)
    - Would it be one file or multiple?
    - Would it have a router?
    - Would the kernels use the polyfill types or raw intrinsics?
    - Would activations.rs and vml.rs be wired through it?
    - What would the import path be?

## STEP 4: Produce the Correct Third Version

Based on your answers to Step 3, write the code that SHOULD exist.

Rules:
- The polyfill must use the SAME type names and method names as rustynum's.
  When `std::simd` stabilizes, `crate::simd::` → `std::simd::` must work
  with zero code changes in any file that uses the types.
- The router must do ONE-TIME CPU detection, not per-call dispatch.
- AVX2 tier must exist (not just AVX-512 + scalar).
- All 11 types from rustynum must be present (not just F32x16/F64x8).
- The SIMD activation functions (sigmoid_f32, softmax_f32, log_softmax_f32)
  that were added in the AFTER state should be KEPT if they're correct,
  but rewired to use the correct polyfill types.
- The SIMD vml.rs paths should be KEPT if correct, rewired same way.
- GEMM and Hamming in kernels should be left as raw intrinsics FOR NOW
  (converting them is a separate future session).
- Everything must compile and pass the existing tests.

## STEP 5: Verify

```bash
# Must pass
cargo test 2>&1 | tail -20

# Must have zero references to simd_compat
grep -rn "simd_compat" src/ --include="*.rs"

# Must have the polyfill at the correct location
ls -la src/simd/ 2>/dev/null || echo "Check: where did you put the polyfill?"
```

## Output

Write your Step 3 answers to `.claude/SIMD_SURGERY_REPORT.md`.
The code changes go directly into the repo.
Commit message must reference which files came from rustynum,
which were kept from simd_compat, and what was rewritten.
