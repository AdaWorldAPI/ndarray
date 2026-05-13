# SIMD review fixes — 2026-05-13

> **Branch:** `claude/ndarray-simd-review-S0zXK`
> **Driver:** 15-agent CCA2A fleet review (12 file-scoped + meta + brutal-reviewer + this PR).
> **Fleet log:** [`AGENT_LOG.md`](./AGENT_LOG.md)

## What this PR fixes

Three soundness/correctness bugs surfaced by the review fleet and confirmed
real by the brutally-honest reviewer (which built the workspace and ran
`cargo clippy --features rayon -- -D warnings` clean and `cargo test
--features rayon --lib` 1783-pass before any change). Most other findings
were either already-clean (project_ortho saturating-cast was already
defined behavior post-Rust-1.45) or deferred (cosmetic-SIMD sweep, polyfill
completion).

| # | Bug | Severity | Fix |
|---|---|---|---|
| 1 | `simd_avx512::permute_bytes` calls `_mm512_permutexvar_epi8` (AVX-512VBMI) as safe `pub fn` with no gate. SIGILL on Skylake-X / Cascade Lake / Ice Lake-SP (which have AVX-512F but **not** VBMI). The doc comment claimed a fallback existed; none did. | **P0 SIGILL** | Added `avx512vbmi: bool` to `SimdCaps`. `permute_bytes` now runtime-branches via the singleton: VBMI hosts use the hardware intrinsic (gated `#[target_feature(enable = "avx512vbmi")]` inner unsafe leaf, Rust language requirement); non-VBMI AVX-512F hosts use a scalar fallback (mirrors the AVX2-tier fallback at `simd_avx2.rs:1435`). |
| 2 | `simd_exp_f32(+Inf)` silently returned ~0.5 in release / panicked in debug. `pow2n_from_int` saturated `f32::INFINITY as i32` to `i32::MAX`, then `(i32::MAX + 127) as u32` wrapped, producing an arbitrary IEEE bit pattern via `f32::from_bits` that combined with the polynomial to `~0.5`. | **P1 silent-wrong-output** | Pre-clamp input domain to `[-87.336, 88.722]` in `simd_exp_f32` (the safe range where exp() is f32-representable). Defense in depth: `pow2n_from_int` also clamps `ni` to `[-126, 127]` before the +127 bias. NaN propagates naturally through the polynomial. Three regression tests added: `+Inf`, `-Inf`, and large-positive (`x=200`) — all assert finite output. |
| 3 | `framebuffer::project_ortho` cast `(neg_f32) as usize` directly. **Reviewer correction:** Rust 1.45+ saturates float→int casts (NaN→0, <MIN→0, >MAX→MAX), so this was already defined behavior. The original commit message overstated it as "UB fix"; it's actually a clarity improvement that clamps in float domain so the intent is visible at the call site. Same observable behavior. | **clarity** | Pre-fix in float domain via `.clamp(0.0, screen_dim as f32 - 1)` before the cast. Functionally equivalent to the prior code; just makes the bounds explicit. |

## What this PR does NOT fix (intentional)

The reviewer flagged that the broader fleet over-alarmed. These were
considered and explicitly deferred:

- **"Cosmetic SIMD" sweep.** ~6 files (`byte_scan::byte_find_all_avx2`,
  `palette_codec::pack_generic_avx512`, `aabb::aabb_intersect_batch_sse41`,
  `renderer::apply_uniform_force`, `simd_ln_f32`) wear `#[target_feature]`
  decorations on scalar bodies. Real but the reviewer judged: not
  Bevy-blocking, real perf-only fix is to complete the polyfill (`U8x64`
  has 25 methods on AVX-512, 0 in `simd_avx2.rs`, 3 in scalar fallback).
  That's the keystone for a future hpc/* rewrite — separate work.
- **AMX detection duplication.** `simd_amx::amx_available()` re-implements
  CPUID + XCR0 + Linux prctl detection that should fold into `SimdCaps`.
  The user explicitly asked to keep this PR surgical and not touch AMX
  byte-call tricks. Deferred.
- **SAFETY-comment audit on `simd_avx512.rs`** (200-deficit). Reviewer
  judged: macro-generated, share one safety contract, adding 200 inline
  comments catches zero bugs. Defer.

## Changes by file

### `src/hpc/simd_caps.rs`
- Added `avx512vbmi: bool` field to `SimdCaps` (previously absent — the
  reviewer's #1 missing-field finding).
- Added `is_x86_feature_detected!("avx512vbmi")` to the x86_64 detect
  branch; `false` in the aarch64 + non-x86 stubs.
- Strictly additive: every existing field unchanged.

### `src/simd_avx512.rs`
- `U8x64::permute_bytes`: rewrote to runtime-dispatch via
  `simd_caps().avx512vbmi`. VBMI path delegates to a new `unsafe fn
  permute_bytes_vbmi` leaf marked `#[target_feature(enable =
  "avx512vbmi")]` (Rust requires this attribute to call VBMI intrinsics
  from a function not compiled with VBMI globally — there is no other
  legal way).
- AVX-512F-without-VBMI path: scalar fallback via `to_array` →
  permute → `from_array`. Same algorithm as `simd_avx2.rs:1435`.
- Inner leaf `permute_bytes_vbmi` documented with explicit SAFETY
  contract referencing the `simd_caps()` gate.
- No other intrinsic touched. AMX inline-asm encodings, `_mm512_*` calls
  in other methods, and the existing `#[target_feature]` annotations are
  all unchanged.

### `src/simd.rs`
- `simd_exp_f32`: pre-clamp input via `simd_clamp(splat(-87.336),
  splat(88.722))` before range reduction. Comment explains the bound is
  the f32-representable domain of exp().
- `pow2n_from_int`: clamp `ni` to `[-126, 127]` before bias addition.
  Defense in depth — caller already pre-clamps but this prevents future
  regressions if the caller's clamp is removed or bypassed.
- Three new tests: `simd_exp_f32_handles_positive_infinity`,
  `simd_exp_f32_handles_negative_infinity`,
  `simd_exp_f32_handles_large_positive`. All assert finite, plausibly-
  scaled output. Pre-fix these would have shown garbage bit patterns
  (release) or panicked (debug).

### `src/hpc/framebuffer.rs`
- `project_ortho`: clamp coords in float domain before `as usize` cast.
  Functionally equivalent to the prior code (Rust 1.45+ saturates), but
  the bound is now visible at the call site rather than relying on the
  cast's saturating behavior + post-cast `.min`.

### `.claude/board/AGENT_LOG.md`
- New file. CCA2A file-blackboard for the 15-agent fleet review that
  produced this PR. APPEND-ONLY. Includes the fleet manifest and 13
  agent entries (12 file-scoped + meta-orchestrator + brutally-honest
  reviewer).

### `.claude/board/SIMD_REVIEW_FIXES_2026_05_13.md`
- This file. PR documentation per request.

## Test surface

```
$ cargo test --features rayon --lib
test result: ok. 1786 passed; 0 failed; 36 ignored; 0 measured

$ cargo clippy --features rayon -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) — 0 warnings
```

Pre-PR: 1783 passing. Post-PR: 1786 passing (+3 simd_exp_f32 regression
tests). No existing tests modified or removed.

## Hardware test matrix

| Target | Pre-PR `permute_bytes` | Post-PR `permute_bytes` |
|---|---|---|
| Sapphire Rapids (avx512f + avx512vbmi) | works (VBMI hardware path) | works (same VBMI path, now via dispatch) |
| Skylake-X / Cascade Lake / Ice Lake-SP (avx512f, no VBMI) | **SIGILL** | works (scalar fallback) |
| Pre-AVX-512 (avx2 only) | type unavailable (cfg-gated out) | type unavailable (unchanged) |
| ARM aarch64 | type unavailable (unchanged) | type unavailable (unchanged) |

`simd_exp_f32` regression tests cover any host capable of running the
test suite — the bug was in the f32 cast logic, not the SIMD intrinsics.

## Review fleet output

15 agents, all entries in `.claude/board/AGENT_LOG.md`:
- Agents #1-12: file-scoped reviews (Sonnet, parallel)
- Agent M: meta-orchestrator synthesis (Opus)
- Agent R: brutally-honest reviewer (Opus, ran the build)

Pattern observed by the fleet but deferred: many `hpc/*` files use
`#[target_feature(enable = "...")]` decorations on scalar code bodies
("cosmetic SIMD"). Real perf work, but per the brutally-honest reviewer
not Bevy-blocking. The keystone fix is completing the polyfill — every
method on `U8x64` / `F32x8` / etc. that exists on AVX-512 must also
exist on AVX2 and scalar, so consumers can write
`crate::simd::U8x64::cmpeq_mask()` and have it work on any CPU. Then
the cosmetic-SIMD wrappers can be deleted in favor of polyfill calls.
That's the next session.
