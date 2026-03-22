# SIMD Surgery

`src/backend/simd_compat.rs` was created from a hallucinated prompt that
never read the source. It landed in a completely wrong location with the
wrong name. The real polyfill is in rustynum: `simd.rs`, `simd_avx512.rs`,
`simd_avx2.rs` — sitting in `src/` root, not `src/backend/`.

5 prompt files (24 references) keep pointing sessions to `simd_compat.rs`,
which is why it keeps getting recreated.

Read what was before (`git show 22bfb7a`), what is now, and what rustynum
actually has. Make a plan how to fix it.

**Don't execute. Show me the plan.**
