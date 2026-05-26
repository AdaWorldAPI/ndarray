# W1a SIMD Primitives — Integration Plan & Per-Agent Assignment

> Date: 2026-05-26 · Branch: `claude/splat3d-cpu-simd-renderer-MAOO0`
> Companions: `vertical-simd-consumer-contract.md` (the W1a spec),
> `simd-dispatch-architecture.md` (the dispatch model).
> Purpose: pin assignments, sequencing, and the **brutal-surprise register**
> so the parallel draft → Opus integrate → PR → bot-loop pipeline lands
> without a SIGILL / won't-compile / unrunnable-test surprise.

## 0. Binding constraints (from this session)

- **Compile-time dispatch ONLY.** New primitives surface through `simd.rs`
  via `#[cfg(target_feature=…)]` / `#[cfg(target_arch=…)]` arms — the existing
  cascade (nightly-simd → avx512f → avx2 → aarch64 → scalar). **No runtime
  dispatch**: no `LazyLock<CpuCaps>`, no `is_*_feature_detected!`, no runtime
  `simd_caps()` routing. Runtime-dispatch versions are **DEFERRED** (was Phase 5).
- **Scalar is the mandatory correctness anchor** (W1a acceptance #1).
- **One PR**, per-workstream commits; subscribe to PR activity for the
  CodeRabbit/Codex fix loop.
- **Clear, file-disjoint agent boundaries** (no two agents touch the same file).

## 1. Per-agent assignment (file-exclusive slots)

| Agent | Exclusive files | Scope | Status |
|---|---|---|---|
| **W1** | `src/simd.rs`, `simd_ops.rs`, `simd_int_ops.rs`, `simd_avx512.rs`, `simd_avx2.rs`, `simd_neon.rs`, `simd_scalar.rs` | 5 W1a primitives (below) | drafting |
| **W2** | `src/hpc/activations.rs`, `src/hpc/reductions.rs` | axis variants (`softmax_axis_f32`, `log_softmax_axis_f32`, maybe `sum_axis_f32`) | drafting |
| **W3** | `src/hpc/soa.rs` | P2 polish: `#[inline]`, `Clone/Debug`, `iter_rows()`, `SoaBatch?` | **done** (no `SoaBatch` — not in design doc) |
| **W4** | `src/hpc/bulk.rs` | P2: un-gate integration test, `bulk_for_each` (+ deprecated `bulk_scan` alias), `#[inline]` | drafting |

No file appears in two rows → zero write-collision by construction.

### W1a primitive sub-slots (all in W1's file set, one agent, sequential)
1. `I8x16::from_i4_packed_u64` + `lane_i8::<N>` + free `batch_packed_i4_16`.
2. `I8x16::saturating_abs` + `I8x32::saturating_abs` (VPABSB correction).
3. `U16x8::gather_u16` + free `palette_lookup_u8x8`.
4. free `prefetch_read_t0/t1/t2` (sanctioned free-fn exception — hints).
5. `U64x8::popcnt` + `xor_popcount` + `U64x4::popcnt`.

## 2. Dependency / entanglement map

- **Missing wrapper types (W1 must define, not assume):** the parity matrix in
  `simd-dispatch-architecture.md §4` shows `I8x16` and `U16x8` are **aarch64-native
  narrow only** — they have **no x86 home**. `U8x8` (palette_lookup output) likewise.
  W1a-#1/#2/#3 therefore require W1 to FIRST add minimal `I8x16`/`U16x8`/`U8x8`
  definitions to `simd_avx512.rs` + `simd_avx2.rs` + `simd_scalar.rs` (native
  `__m128i` on SSE2 where sane, else scalar-storage matching the existing `🟠`
  polyfill pattern). This is W1a's own work — **not** blocked on TD-SIMD-2.
- `U64x8` exists as a `🟠` scalar polyfill on the AVX2 default and native on
  AVX-512 → `popcnt` is implementable on both today (no new type needed).
- W2/W3/W4 are independent of W1 and of each other (only `hpc/mod.rs` is shared,
  and it already declares `soa`/`bulk` — no edit needed).
- **Downstream:** the 5 lance-graph consumer PRs wait on this PR merging.

## 3. BRUTAL-SURPRISE REGISTER (read before integrating)

1. **Only ONE backend compiles per cargo config.** Default `.cargo/config.toml`
   is `x86-64-v3` (AVX2) → the `#[cfg(target_feature="avx512f")]` arms and all
   `_mm512_*` code are **not even compiled** by default `cargo check`. ⇒ Opus
   MUST compile-check BOTH configs: default **and**
   `--config .cargo/config-avx512.toml`. A green default build does NOT prove
   the AVX-512 intrinsics compile.
2. **AVX-512 binaries SIGILL on non-AVX-512 silicon** (TD-SIMD-1). We can
   *compile* the avx512 config here but may not be able to *run* its tests on
   this runner. Treat avx512 as compile-checked, run-deferred to a capable CI job.
3. **"All backends agree" is NOT testable in one binary** under compile-time
   dispatch (the other backends' types are `#[cfg]`-ed out). ⇒ Parity tests
   assert the **active compiled backend == an inline plain-Rust scalar reference**
   computed in the test itself. Cross-backend agreement is then guaranteed by
   running the same test suite under EACH config (v3 / avx512 / aarch64-cross /
   nightly-simd), not by comparing two backends in one process.
4. **VPABSB does not saturate `i8::MIN`** — binding. AVX-512 must be
   `_mm512_min_epu8(_mm512_abs_epi8(x), set1(0x7f))`; NEON `vqabsq_s8`; scalar
   `i8::saturating_abs`. The `saturating_abs(i8::MIN)==i8::MAX` test is mandatory.
5. **`gather` has no NEON instruction** → scalar loop on aarch64 is the correct
   impl, not a stopgap. Bounds-validate `max(idx) < table.len()` before any x86
   gather (debug panic; release scalar-safe).
6. **`prefetch` on an invalid ptr is allowed** (silent CPU drop) — no `assert!`.
   `_mm_prefetch` on x86; `prfm pld…` asm on aarch64; **no-op** elsewhere.
7. **`avx512vpopcntdq` is a sub-feature** of avx512f. `_mm512_popcnt_epi64`
   needs `#[cfg(target_feature="avx512vpopcntdq")]`; provide the
   VPSHUFB/Mula fallback for plain-avx512f and the byte-LUT path otherwise.
8. **Doc-examples are doctests** (CodeRabbit enforces `///` + example on every
   `pub fn`). They must compile on the DEFAULT config — keep examples backend-
   agnostic (call the `crate::simd::*` surface, not a specific backend type).

## 4. Central verification gates (Opus, once, shared target/)

In order, after all drafts land:
1. `cargo fmt --all`
2. `cargo clippy -p ndarray --all-targets` (default v3) → `-D warnings`
3. `cargo test -p ndarray` (default v3: exercises AVX2 + scalar arms + doctests)
4. `cargo clippy --config .cargo/config-avx512.toml -p ndarray` (compile-check
   the AVX-512 arms — surprise #1/#2). Run its tests only if the runner has
   AVX-512; otherwise note "compile-checked, run-deferred".
5. (best-effort) `cargo check --target aarch64-unknown-linux-gnu` if a cross
   toolchain is present — else rely on the NEON CI job.

## 5. Integration order

W3 (done) → fold W2/W4 (low-risk, isolated) → fold W1 last and hardest:
reconcile missing-type definitions, fix every `// UNVERIFIED:`, run gate §4
(both configs), then commit per-workstream. PR. Subscribe. Bot-loop hardens.

## 6. PR

Single PR off `claude/splat3d-cpu-simd-renderer-MAOO0`. Body cites each W1a
consumer site (`lance-graph:crates/lance-graph-contract/src/mul.rs`,
`bgz17/src/simd.rs`, `holograph/hamming.rs`) per acceptance #7, and states the
compile-time-only / runtime-deferred posture explicitly so reviewers don't flag
the absent runtime dispatch as a gap.
