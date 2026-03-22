# SESSION: Deep Audit — simd_compat.rs in ndarray

## Context

A previous session created `src/backend/simd_compat.rs` and modified several
files to import from it. The session that wrote the prompt for this work
admits it did NOT read the source code before writing the prompt. The prompt
may have described the wrong architecture. The resulting code may be wrong.

This audit determines: what was created, what was changed, what's broken,
and what the correct state should be.

## READ FIRST — The Two Codebases

Before answering ANY question below, read BOTH of these in full:

```bash
# 1. What was CREATED in ndarray (the thing being audited)
cat src/backend/simd_compat.rs
grep -rn "simd_compat" src/ --include="*.rs"

# 2. What ALREADY EXISTS in rustynum (the supposed source of truth)
cat <rustynum>/rustynum-core/src/simd.rs
cat <rustynum>/rustynum-core/src/simd_avx512.rs
cat <rustynum>/rustynum-core/src/simd_avx2.rs

# 3. What ndarray's kernels looked like BEFORE the changes
git log --oneline src/backend/kernels_avx512.rs | head -10
git show <commit-before-simd_compat>:src/backend/kernels_avx512.rs | head -50

# 4. What ndarray's activations.rs and vml.rs looked like BEFORE
git log --oneline src/hpc/activations.rs | head -10
git log --oneline src/hpc/vml.rs | head -10
```

## Questions to Answer

### Q1: Type Inventory
How many types does `simd_compat.rs` define?
How many types does rustynum's `simd_avx512.rs` define?
Which types are in rustynum but missing from `simd_compat.rs`?
Do the missing types matter? What are they used for?

### Q2: API Surface
For each type that exists in BOTH files:
  - Are the method signatures identical?
  - Are there methods in rustynum's version that are missing from `simd_compat.rs`?
  - Are there methods in `simd_compat.rs` that DON'T exist in rustynum?
  - Do the operator impls match (Add, Sub, Mul, Div, BitXor, Neg, etc.)?

### Q3: Architecture — Router
Rustynum has `simd.rs` which does one-time CPU detection at init.
Does `simd_compat.rs` have any runtime CPU detection?
How does `simd_compat.rs` choose between AVX-512 and non-AVX-512?
What happens on an x86_64 machine that has AVX2 but NOT AVX-512?
  - In rustynum?
  - In the ndarray simd_compat version?

### Q4: AVX2 Tier
Does rustynum have an AVX2 implementation? (Check `simd_avx2.rs`)
Does `simd_compat.rs` have an AVX2 implementation?
If a machine only has AVX2, what code path does each version take?

### Q5: Files Modified
Which files in ndarray import from `simd_compat`?
For each file: what did it look like BEFORE the simd_compat changes?
Use `git log` and `git show` to compare before/after.
Were the changes correct? Do they improve or degrade the code?

### Q6: kernels_avx512.rs Changes
What functions in `kernels_avx512.rs` were changed to use simd_compat types?
What functions were left with raw intrinsics?
Is the split intentional or accidental?
Do the changed functions produce the same numerical results as the originals?

### Q7: activations.rs Changes
What was added to `activations.rs`?
Was it there before, or is it entirely new code?
Does the new code match what rustynum's `array_struct.rs` does for the same operations?
Is the `simd_exp_f32` function correct? Compare against rustynum's exp implementation if one exists.

### Q8: vml.rs Changes
Same questions as Q7 but for `vml.rs`.
Were the scalar loop implementations replaced or augmented?
Do the SIMD versions produce the same results as the scalar versions?

### Q9: The Correct Target State
Based on reading rustynum's actual SIMD architecture:
  - Where should the polyfill live in ndarray? (backend/? simd/? root?)
  - What module structure does rustynum use?
  - How does rustynum's router work?
  - What is the intended relationship between the polyfill and `std::simd`?
  - When `std::simd` stabilizes, what needs to change?

### Q10: Salvageable vs Must-Redo
For each change made by the simd_compat session:
  - Is the code CORRECT (produces right results)?
  - Is the code COMPLETE (covers what rustynum covers)?
  - Is the code in the RIGHT PLACE (matches intended architecture)?
  - Should it be kept, fixed, or reverted?

### Q11: Test Coverage
How many tests does `simd_compat.rs` have?
How many tests does rustynum's `simd_avx512.rs` have?
Do the simd_compat tests cover the same cases?
Are there failure modes that simd_compat tests miss?

### Q12: Build Status
Does ndarray currently build? If not, is simd_compat the cause?
Do all existing tests pass?
Were any existing tests broken by the simd_compat changes?

## Deliverable

Produce a report answering each question with evidence (file paths, line numbers,
git commits, code snippets). No assumptions. No "probably" or "likely."

Then produce a RECOMMENDED ACTION for each file:
  KEEP AS-IS / FIX (describe what) / REVERT TO PRE-CHANGE / REPLACE WITH RUSTYNUM VERSION

The report goes in `.claude/SIMD_AUDIT_REPORT.md`.
