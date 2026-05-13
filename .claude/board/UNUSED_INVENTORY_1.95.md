# Unused / Allow-suppressed Inventory — 1.95 sweep

> **Captured:** 2026-05-13 while bumping ndarray master to Rust 1.95.0.
> **Source:** `grep -rn "#\[allow(dead_code\|#\[allow(unused" src/` (lib only).
> **Purpose:** signal mining — each entry below is a piece of code that
> compiles but is allowed to be unused. Some are legitimate scaffolding
> (rayon-gated, no_std polyfill, hot-swap reserves), some are actionable
> (Rust-1.64 compatibility imports that 1.95 likely doesn't need, phantom
> tier variants flagged by the round-1 audit fleet).

## Categorization

### A — Likely actionable (real "API drift" candidates)

| # | Site | Suppression | Why interesting |
|---|---|---|---|
| A1 | `src/impl_owned_array.rs:7` | `unused_imports // Needed for Rust 1.64` | We just bumped to 1.95. The 1.64-era workaround is almost certainly stale. **Action:** remove the suppression, see if the import is still needed; if not, drop it. |
| A2 | `src/iterators/mod.rs:24` | `unused_imports // Needed for Rust 1.64` | Same as A1. |
| A3 | `src/hpc/simd_dispatch.rs:47` | `dead_code` on `SimdTier::WasmSimd128` (and likely `Sse2`) | **Flagged by round-1 audit agent #6**: `SimdTier::Sse2` is never selected in `detect()`; no SSE2 wrapper functions exist. Phantom variants. **Action:** delete or wire. |
| A4 | `src/hpc/clam.rs:928` | `dead_code` | CLAM cluster code. **Action:** verify if the suppressed fn is reachable through any public CLAM call path; if not, delete. |
| A5 | `src/hpc/clam_compress.rs:141` | `dead_code` | Same family as A4. |
| A6 | `src/hpc/jitson/scan_config.rs:144` | `dead_code` | jitson scan-config. Probably scaffolding for an unfinished feature; needs PR-author review. |
| A7 | `src/impl_ref_types.rs:364` | `dead_code` | ndarray ref type helper. |
| A8 | `src/backend/native.rs:376`, `:423` | `dead_code` (×2) | Backend kernel helpers possibly only used by feature-gated paths. |
| A9 | `src/hpc/packed.rs:29`, `:42` | `unused_variables` (×2) | Variables defined but unused. Either rename to `_x` for explicit "ignored", or wire them in. |

### B — Documented scaffolding (intentional, leave)

| # | Site | Suppression | Justification |
|---|---|---|---|
| B1 | `src/simd.rs:15,31,48,96,110,126,1612,1658,1672` (×9) | `dead_code` | Tier enum variants for cross-arch builds (only some platforms use all variants) + no_std `LazyLock` polyfill helpers. |
| B2 | `src/split_at.rs:11` | `dead_code // used only when Rayon support is enabled` | Self-documented. |
| B3 | `src/hpc/jitson_cranelift/engine.rs:130` | `dead_code // retained for future hot-swap / eviction by FuncId` | Self-documented. |
| B4 | `src/data_repr.rs:11`, `src/data_traits.rs:11`, `src/impl_constructors.rs:37`, `src/impl_methods.rs:13`, `src/free_functions.rs:13`, `src/impl_views/conversions.rs:10` | `unused_imports` | Conditional-cfg imports needed under some feature combinations. |
| B5 | `src/parallel/mod.rs:120,122` | `unused_imports // used by rustdoc links` | Self-documented. |
| B6 | `src/simd_neon.rs:1721` | `unused_macros` | NEON macros only used on aarch64. |
| B7 | `src/zip/mod.rs:424,715` | `dead_code` / `unused` | Zip iterator scaffolding. |
| B8 | `src/backend/native.rs:20` | `dead_code` | Backend kernel scaffold. |
| B9 | `src/doc/ndarray_for_numpy_users/mod.rs:750` | `unused_imports` | Doc example. |
| B10 | `src/itertools.rs:90` | (inside a doc-comment example) | Not real code. |

## Suggested follow-ups (post-1.95 bump)

- **A1, A2 (Rust 1.64 compat sweep):** remove the suppression, attempt build, drop the import if no longer needed. Likely a 2-line change × 2 files.
- **A3 (phantom SimdTier variants):** the round-1 audit fleet already flagged this. Pair with the round-3 cosmetic-SIMD consumer fleet — they'll touch `simd_dispatch.rs` anyway.
- **A4, A5 (clam dead_code):** needs domain owner review. Worth flagging in a CLAM-related PR; not urgent.
- **A6 (jitson scan_config):** check INTEGRATION_PLANS — if jitson is still active, wire; if archived, delete.
- **A7, A8, A9:** small-scope per-site review; aggregate into a "dead-code cleanup" PR.

## What this PR (1.95 bump) does NOT touch

This file is **signal mining only**. The 1.95 bump PR itself:
- Bumps `rust-toolchain.toml` → 1.95.0
- Bumps `Cargo.toml` `rust-version` → "1.95"
- Bumps `MSRV` / `BLAS_MSRV` env in ci.yaml → 1.95.0
- Fixes the ONE 1.95 clippy lint that fires (`clippy::manual_checked_ops`
  in `impl_owned_array.rs::into_scalar`)

The 36 suppression sites are pre-existing and persist after the bump.
Cleaning them up is the follow-up work surfaced by this inventory.
