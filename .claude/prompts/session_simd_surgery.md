# SESSION: Fix SIMD — Previous Session Got It Wrong

## What Happened

A session created `src/backend/simd_compat.rs` and rewired `kernels_avx512.rs`,
`activations.rs`, and `vml.rs` to import from it. The prompt that session
worked from was written without reading the actual source. The code landed
in the wrong location, with the wrong name, missing types, and no router.

The REAL std::simd polyfill already exists in rustynum and works:
```
rustynum-core/src/simd.rs
rustynum-core/src/simd_avx512.rs
rustynum-core/src/simd_avx2.rs
```

These belong in ndarray's `src/` root (same position as rustynum), not
inside `src/backend/`. Consumers use `crate::simd::`. When `std::simd`
stabilizes: `crate::simd` → `std::simd`, delete the files. One word change.

5 other prompt files reference `simd_compat.rs` by name (24 total references),
which is why other sessions keep recreating it.

## What To Do

1. Read rustynum's `simd.rs`, `simd_avx512.rs`, `simd_avx2.rs`
2. Read ndarray's current `simd_compat.rs`, `kernels_avx512.rs`, `activations.rs`, `vml.rs`
3. Read what those ndarray files looked like BEFORE (`git show 22bfb7a:<path>`)
4. Make a plan to fix it
5. **Show me the plan. Don't execute yet.**
