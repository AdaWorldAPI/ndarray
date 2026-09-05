# gemm-ternlog-mask-consolidation-v1 — one GEMM entry per dtype, masks as the prefilter, ternlog as the mask ALU

> **Status:** DRAFT v1 (2026-09-05). Source-first: every "exists" row cites `file:line`
> at ndarray `claude/great-curie-d2ufyl` HEAD (PR #303 + this doc). Every "proposed"
> row carries a falsifier. Nothing in §5 is built. No kernel, no ABI symbol.
>
> **Operator ask (2026-09-05, verbatim in spirit):** *consolidate gemm, ternlogq chaining,
> and cached mask reuse; MKL and gemm and blasgraph are all reverse-engineered; MKL could
> benefit from ternlogq.* Plus the W0 evidence that started it: the fork's AMX f32 GEMM
> silently computed in bf16 (burn#9 → ndarray#303), and the benchmark that decided the
> fix showed **AMX loses to the hand-rolled F32x16 kernel on both axes at every size**.
>
> **Scope fence, read first.** *Cached mask reuse* already has an authoritative,
> council-corrected owner: lance-graph-java `.claude/plans/mask-risc-lowering-v1.md`
> (v4.2 — the memo, rail-group factoring, voxelmasking, "focus of awareness are cached
> masks"). This plan does NOT re-plan it. It consumes its decisions (§2.3) and adds the
> two things nobody owns: **(a)** the GEMM surface itself, and **(b)** the seam where a
> cached mask becomes a GEMM row/col prefilter — the only place "ternlogq chaining" and
> "GEMM" actually meet in code today (`hpc/prefilter.rs:189`).

## §0 — The model, in one picture

```
   T2  where / hop / plan_eval        Mask × WideFieldMask → Mask         (lgj-abi exports)
        │  cached masks (mask-risc-lowering-v1: memo, rail groups)
        ▼
   T1  ternlog chain  m = ternlog<IMM>(a, b, c)  …  one VPTERNLOGQ per 512 bits
        │  (ndarray::simd::mask_ternlog_assign; lgj kernels::simd_mask_ternlog_assign)
        ▼
   T1  mask → GEMM prefilter          pruned_gemm_rows(mask, …)          (hpc/prefilter.rs)
        │
        ▼
   T0/T1  ONE gemm per dtype          gemm_f32 / gemm_f64 / gemm_bf16 / gemm_i8   (backend::mod.rs)
                                       ↓ dispatch: F32x16 sgemm_blocked | matrixmultiply | AMX(bf16,i8 only)
```

Three claims, each falsifiable in §6:
1. **There is one GEMM per dtype at the facade; everything else is a backend.** Today there are ~50 `pub fn *gemm*|*matmul*` across 12 files (§1.1).
2. **A mask is the prefilter, and it arrives already chained.** The T2 tier hands T1 a mask that ternlog has already conjoined; GEMM never recomputes a predicate.
3. **AMX is a BF16/INT8 unit, not an f32 unit.** Measured (§1.3). No f32 path routes through it by default, ever.

## §1 — What EXISTS (documentation register — every row cites source)

### §1.1 — The GEMM inventory (the consolidation target)

`grep -rn 'pub fn [a-z_0-9]*\(gemm\|matmul\)' src/` at HEAD, grouped by what they are:

| group | entry points | file:line | role today |
|---|---|---|---|
| **Facade, unified** | `cblas_sgemm`, `cblas_dgemm`, `gemm_i8`, `gemm_bf16` | `backend/mod.rs:149,158,186,211` | the ONLY dtype-unified surface; `cblas_gemm_s8s8s32`, `cblas_gemm_bf16bf16f32` at `:268,:274` are aliases |
| Native f32/f64 | `gemm_f32`, `gemm_f64`, `gemm_f32_tiled`, `sgemm_nr/mr`, `dgemm_nr/mr` | `backend/native.rs:210,260,425,50-77` | `gemm_f32` = `matrixmultiply::sgemm` (crates.io), NOT the hand-rolled kernel |
| **Hand-rolled F32x16** | `sgemm_blocked`, `dgemm_blocked` | `backend/kernels_avx512.rs:665,833` | packed panels MR=6/NR=16/KC=256/MC=72/NC=256 (`:538-542`); **fastest exact path measured**; `pub(crate)`, unreachable from outside |
| Hand-rolled AVX2 twin | `sgemm_blocked`, `dgemm_blocked` | `simd_avx2.rs:462,479` | second copy, portable backend |
| Reverse-engineered "MKL" | `gemm_f32`, `gemm_f64`, `sgemm`, `dgemm`, `sgemm_bf16`, `sgemm_int8` | `backend/mkl.rs:195,219,384,429,479,531` | CBLAS-shaped API names over pure Rust; feature `intel-mkl` |
| OpenBLAS-shaped | `gemm_f32`, `gemm_f64` | `backend/openblas.rs:90,116` | feature `openblas` |
| AMX (ArrayView API) | `matmul_bf16_to_f32`, `matmul_i8_to_i32`, `matmul_f32`, `matmul_f32_bf16_fast`, `matmul_f32_amx_split` | `hpc/amx_matmul.rs:531,887,781,748,810` | post-#303: `matmul_f32` is exact (delegates to native); the two AMX-f32 variants are named opt-ins with the bench table in their docs |
| AMX tiles | `bf16_tile_gemm_16x16`, `_packed`, `_tier` | `hpc/bf16_tile_gemm.rs:45,203,66` | the TDPBF16PS 16×16 tile |
| AMX tiles (dup) | `bf16_tile_gemm_16x16` | `simd_ops.rs:587` | **duplicate name of the above** — verify same body (W0) |
| INT8 | `int8_tile_gemm_16x16`, `int8_gemm_amx_tiled`, `int8_gemm_vnni`, `gemm_u8_i8` | `hpc/int8_tile_gemm.rs:47,358`, `hpc/vnni_gemm.rs:46`, `simd_int_ops.rs:253` | AMX → VNNI zmm → VNNI ymm → scalar |
| Quantized refs | `bf16_gemm_f32`, `mixed_precision_gemm`, `int8_gemm_i32`, `int8_gemm_f32`, `int8_gemm_per_channel_f32` | `hpc/quantized.rs:444,484,618,633,655` | scalar references + per-channel scaling |
| f64 tiled | `gemm_f64_tiled`, `gemm_f64_tiled_fma` | `simd_ops.rs:952,1003` | separate f64 blocking, not `dgemm_blocked` |
| Batched | `batched_gemm_f32`, `batched_gemm_4d_f32` | `hpc/linalg/batched.rs:60,124` | loops over 2-D GEMM |
| **Mask-prefiltered** | `pruned_gemm_rows` | `hpc/prefilter.rs:189` | the ONLY existing mask→GEMM bridge |
| Runtime re-exports | `matmul_f32`, `matmul_bf16_to_f32`, `matmul_i8_to_i32` (×2), `gemm_u8_i8` | `simd_runtime/matmul.rs:43,33,80,193,227` | feature `runtime-dispatch` mirror of the AMX API |
| Misc | `matmul_vec`, `matmul_i8_to_i32_wasm` | `hpc/models/layers.rs:174`, `simd_wasm.rs:1474` | model layer GEMV; wasm arm |

**Counted, not estimated:** 54 entry points, 12 files, **4 unified.** `hpc/blas_level3.rs` (named in CLAUDE.md as "BLAS L3 gemm/syrk/trsm/symm") returned **zero** `pub fn` hits in this grep — W0 must resolve whether that module is empty, macro-generated, or misnamed.

### §1.2 — ternlog: already a T1 primitive, already chained once

- Facade: `U64x8::ternlog<const IMM: i32>` / `U32x16::ternlog<IMM>` — `simd_avx2.rs:3559,3618` (portable backend; W1a-#9), with the truth-table constants module `simd.rs:570` (`AND3 0x80, AND2_ANDNOT 0x40, AND_ANDNOT2 0x10, OR2_AND 0xA8, XOR3 0x96, MAJ3 0xE8, AND2 0xC0, OR3 0xFE`). `mask_ternlog` / `mask_ternlog_assign` re-exported at `simd.rs:733`.
- lgj-abi consumes it correctly by tier: `kernels.rs:100` `pub use ndarray::simd::ternlog;`, `kernels.rs:111` `simd_mask_ternlog_assign<IMM>` → `ndarray::simd::mask_ternlog_assign::<IMM>`. `exports.rs` names `kernels::ternlog::AND3`, never `ndarray::simd` (membrane-tiers.md:37).
- **It is already the hop conjunction:** lgj LATEST_STATE:10-13 — the two-AND conjunction became ONE `mask_ternlog_assign::<AND3>` pass, one `VPTERNLOGQ` per 512 bits.
- **And it is NOT where the time goes:** LATEST_STATE:27-29 — "the ternlog wire is one of three mask passes and cannot account for 5×"; the bulk was `eq_u32_strided_to_mask` at `stride_bytes == 4`. This plan inherits that measurement: **ternlog chaining is a correctness/shape win first, a speed win only where a measurement says so.**
- jitson already whitelists `vpternlogd/vpternlogq` (`hpc/jitson/validator.rs:39,327`) — the JIT tier can emit it.

### §1.3 — The W0 evidence this plan is built on (measured 2026-09-05, PR #303)

`gemm_paths_bench` (`hpc/amx_matmul.rs`, `#[ignore]`, `--release`), Xeon w/ AMX + AVX-512, square f32, irrational inputs:

| size | `native::gemm_f32` (matrixmultiply) | **F32x16 `sgemm_blocked`** | AMX 3-pass split | AMX 1-pass bf16 |
|---|---|---|---|---|
| 256³ | 0.42 ms · 2.9e-7 | **0.37 ms · 2.9e-7** | 2.78 ms · 1.4e-6 | 0.77 ms · 3.7e-4 |
| 512³ | 3.51 ms · 1.2e-6 | **3.22 ms · 1.2e-6** | 20.9 ms · 1.4e-6 | 6.37 ms · 2.2e-4 |
| 1024³ | 28.8 ms · 1.4e-6 | **27.3 ms · 1.4e-6** | 159 ms · 1.5e-6 | 45.9 ms · 1.6e-4 |

**Re-measured past the cache-resident regime (operator objection: "256/512/1024 are undersized"):**

| shape | matrixmultiply | F32x16 `sgemm_blocked` | AMX 1-pass | AMX 3-pass |
|---|---|---|---|---|
| 2048³ | 218.7 ms | **216.7 ms** | 371.2 ms · 1.2e-4 | 1099 ms · 2.3e-6 |
| 4096³ | 1737 ms | **1616 ms** | 2826 ms · 1.1e-4 | 8106 ms · 2.9e-6 |
| 1024×4096×1024 (deep K) | 115.0 ms | **109.6 ms** | 206.6 ms | 580.7 ms |
| 256×8192×256 (skinny) | **14.4 ms** | 17.1 ms | 33.2 ms | 99.3 ms |
| 64×2048×8192 (wide N) | **35.6 ms** | 39.3 ms | 148.5 ms | 369.6 ms |

AMX does not converge with size — 1-pass is 1.75× slower than F32x16 at 4096³, 3-pass 5×. "Undersized" was not hiding an AMX win. What size DID expose: on the two skinny rectangles `matrixmultiply` beats `sgemm_blocked` by 10–19% — the hand-rolled MC=72/NC=256 blocking is tuned for square-ish operands. **D-GTM-F1 therefore needs the shape guard D-GTM-0c anticipated**, not a blanket default.

Three consequences, all frozen in §4: the hand-rolled kernel is the exact default *for square-ish shapes, with a measured rectangle guard*; AMX is BF16/INT8-only; the "one rounding + f32 accumulate" discipline was correctly implemented in the AMX path and was still insufficient for an f32 contract — the missing half was operand splitting, and even that half loses to not using AMX.

### §1.4 — Mask caching: owned elsewhere, consumed here

- lgj-abi `registry.rs:842-849` — `set_cached_carving` / `cached_carving` on a mask (per-generation, invalidated on close).
- Java `Mask.java:43,185-191` — the packed-bit word lane is resolved once and cached, re-validated per facade call (the `epoch` recheck; `ISS-LGJ-CACHED-DESCRIPTOR-CROSS-THREAD-WINDOW` still open).
- `NativePattern.java:54` — `scratchMask` reused as the destination of terminal ops.
- **The plan that owns all of this:** `lance-graph-java/.claude/plans/mask-risc-lowering-v1.md` v4.2 — §5 rail-group factoring (why the memo MAY hit), §14 voxelmasking (the vertical axis is enumerated, not cached), §3c reuse map (OSM native; weather splits). Its frozen decisions and gates D-MRL-* are **inputs** here, never re-decided.

## §2 — Three seams, one rule each

### §2.1 — GEMM facade rule: **one name per (dtype-in, dtype-out); backends are `pub(crate)`**

`backend::mod.rs:149-215` is already that shape for four signatures. The rule makes it total: every row in §1.1 that is not one of `gemm_f32 / gemm_f64 / gemm_bf16 / gemm_i8` (+ the batched wrappers) either (a) becomes the backend one of those dispatches to, (b) is a named opt-in with a measured reason to exist, or (c) is deleted after W0 proves it a duplicate. The `simd_runtime/matmul.rs` mirror is (a). The MKL/OpenBLAS-shaped modules are (a) behind their features. The two AMX-f32 variants are (b), already documented. `simd_ops.rs:587` vs `hpc/bf16_tile_gemm.rs:45` is (c) pending W0.

### §2.2 — ternlog chaining rule: **a predicate is composed at T2, lowered to ONE ternlog chain at T1, and never re-evaluated below**

A 3-input `IMM` is the whole truth table (`simd.rs:561-563`), so any Boolean over three masks is one pass; an n-input predicate is ⌈(n−1)/2⌉ chained passes. The chain is *emitted* by the T2 planner (mask-risc-lowering-v1 Wave 1/2), *executed* by `mask_ternlog_assign`, and its RESULT is what reaches the GEMM prefilter. GEMM never sees a predicate, only a mask. This is the "chaining" the operator named; the *cache* of the intermediate masks is mask-risc-lowering-v1's memo (§1.4), not ours.

**Where MKL benefits (the operator's hunch, scoped honestly):** in a GEMM, ternlog pays at the *edges* — the K-tail / N-tail lane masks, sign/abs prologues, and beta==0/1 select — not in the FMA core, which is arithmetic. In the reverse-engineered `backend/mkl.rs` the same tail handling appears per-kernel (`sgemm`, `sgemm_bf16`, `sgemm_int8`). Consolidating those tails onto `mask_ternlog` is a shape win (one tail idiom) and *possibly* a speed win; W0-3 measures before anyone claims it.

### §2.3 — Mask→GEMM seam rule: **the mask is the row/col fetch list; the prefilter is bulk, not per-row**

`hpc/prefilter.rs:189 pruned_gemm_rows` is the seed. The rule from mask-risc-lowering-v1 §14.7 ("the mask IS the fetch list") applies verbatim: a cached mask selects which rows of A (or cols of B) enter the packed panel; the pack loop consumes the mask by word (`popcount` + `pdep`-style expansion or a compacted index list built ONCE per mask generation), never by testing bits inside the micro-kernel. The mask's `cached_carving` (`registry.rs:842`) is the natural home for that compacted index list — **that is the one place this plan asks mask-risc-lowering-v1 for something**: a carving slot whose payload is a GEMM row-index list, keyed by mask generation.

## §3 — Non-goals (each with why)

- **Not** re-deciding anything in mask-risc-lowering-v1 (memo policy, rail groups, voxelmasking). Consumed as inputs.
- **Not** an f32 AMX path of any kind. Measured; §1.3. Re-open only with a new measurement on a host with a true f32 tile op.
- **Not** touching `blasgraph` (lance-graph) — that is HDR/16384-bit semiring algebra with its own sparse formats and (per its own plan doc) an *unwritten* tropical GEMM. It is a consumer of §2.1's facade once one exists, not part of the consolidation.
- **Not** a new SIMD backend, a new crate, or any `#[cfg(target_arch)]` above T1 (simd-savant / kernel-membrane-warden).
- **Not** `array_windows` for GEMM: sliding windows overlap; packed panels do not. Recorded so it is not re-proposed.

## §4 — Frozen decisions (the council attacks these, not the prose)

| id | decision | evidence |
|---|---|---|
| D-GTM-F1 | `gemm_f32` default = hand-rolled F32x16 `sgemm_blocked` on `avx512f` hosts for square-ish shapes; `matrixmultiply` for skinny/wide rectangles (threshold from D-GTM-0c) and on other hosts. Exact on both. | §1.3: wins 256³–4096³ (up to 7%); LOSES 10–19% at 256×8192×256 and 64×2048×8192 |
| D-GTM-F2 | AMX serves `gemm_bf16` and `gemm_i8` ONLY. Any f32 AMX path is a named opt-in carrying its measured table. | §1.3; ndarray#303 |
| D-GTM-F3 | The facade is `backend::{gemm_f32, gemm_f64, gemm_bf16, gemm_i8}` (+ batched). Every other GEMM symbol is `pub(crate)`, a documented opt-in, or deleted. | §1.1 count 54→4 |
| D-GTM-F4 | Mask→GEMM prefilter consumes a compacted row-index list built once per mask generation, stored as a mask carving. Never bit-tests inside the micro-kernel. | §2.3; mask-risc §14.7 |
| D-GTM-F5 | Every accuracy test in this surface uses inputs whose significands exceed 8 bits, and tolerances at f32 grade (1e-5) for f32 APIs. | the vacuous `(i+j)*0.5` test that hid the bf16 loss (#303) |
| D-GTM-F6 | Every new kernel lands with a two-sided pin: the fast path must beat the reference by a stated factor AND the reference must still be measurably slower — so a regression in either direction fails. | `three_pass_split_beats_one_bf16_pass` pattern |

## §5 — What is PROPOSED (plan register), substrate-first

### Wave 0 — measure before minting (no production code; all probes `#[ignore]` + `--release`)

| id | probe | falsifier / gate |
|---|---|---|
| D-GTM-0a | Diff `simd_ops.rs:587 bf16_tile_gemm_16x16` vs `hpc/bf16_tile_gemm.rs:45`; diff `simd_avx2.rs:462 sgemm_blocked` vs `kernels_avx512.rs:665`. | byte-identical bodies → delete one; divergent → record which is canonical and why |
| D-GTM-0b | Resolve `hpc/blas_level3.rs`: what does it export, if anything? | zero `pub fn` → either delete the CLAUDE.md claim or find the macro |
| D-GTM-0c | Extend `gemm_paths_bench` to f64 (`dgemm_blocked` vs `gemm_f64_tiled_fma` vs matrixmultiply) and to non-square / K-tail shapes (e.g. 1000×1000×1000, 17×33×15 scaled). | if `sgemm_blocked` loses on any tail shape, D-GTM-F1 gains a shape guard, not a revert |
| D-GTM-0d | **The MKL-ternlog hunch:** in `backend/mkl.rs` `sgemm` (`:384`), time the tail-lane handling with the current idiom vs one `mask_ternlog` select, K∈{255,257,1023,1025}. | < 3% end-to-end → record as shape-only win, no speed claim; ≥ 3% → W1 item |
| D-GTM-0e | `pruned_gemm_rows` (`prefilter.rs:189`): measure per-row bit-test vs compacted-index pack at 10%/50%/90% mask density. | the crossover density decides D-GTM-F4's threshold, or proves compaction always wins |
| D-GTM-0f | Count real call sites of every §1.1 symbol across ndarray, lance-graph, lance-graph-java, burn (`grep -rn`). | symbols with zero external callers are `pub(crate)` candidates for W1 with no consumer wave |

### Wave 1 — ndarray (the facade and the backends)

- D-GTM-1: `backend::gemm_f32` → runtime `avx512f` check → `sgemm_blocked` (one `unsafe` call with a SAFETY comment; the `#[target_feature]` boundary stays inside `kernels_avx512.rs`), else matrixmultiply. `dgemm` per D-GTM-0c. Two-sided pin per D-GTM-F6.
- D-GTM-2: demote every non-facade GEMM symbol per D-GTM-F3, in one commit per file, each with the W0-0f caller count in its message. `simd_runtime/matmul.rs` becomes a thin re-export of the facade.
- D-GTM-3: `hpc/amx_matmul::matmul_f32` (the ArrayView API) delegates to `backend::gemm_f32` — already does post-#303; make the delegation explicit in the doc and drop `pack_contig` when the views are already contiguous.
- D-GTM-4 (gated on D-GTM-0d ≥ 3%): tail-lane selects in `mkl.rs` `sgemm`/`sgemm_bf16`/`sgemm_int8` become `mask_ternlog` idioms.

### Wave 2 — the mask→GEMM seam (ndarray + lgj-abi, one PR each)

- D-GTM-5 (ndarray): `pruned_gemm_rows` takes a `&[u32]` compacted index list, not a mask; a `mask_to_row_indices(&[u64]) -> Vec<u32>` T1 primitive builds it (popcount + expand, one pass). Density crossover per D-GTM-0e.
- D-GTM-6 (lgj-abi, **filed against mask-risc-lowering-v1, not built here**): a carving kind whose payload is that index list, keyed by mask generation, invalidated with the mask. This is the single ask of the other plan.

### Wave 3 — consumers (last, by the STOP rule)

- burn-ndarray `simd_dispatch::matmul_2d` → `backend::gemm_f32` once D-GTM-1 lands; delete the `amx-f32` feature (burn#9 follow-up).
- blasgraph / bgz-tensor / jc: no change unless W0-0f finds a caller of a demoted symbol.

## §6 — Pre-registered gates (decided before any worker runs)

- G1: after W1, `grep -rn 'pub fn [a-z_0-9]*\(gemm\|matmul\)' src/` outside `backend/mod.rs` + batched + documented opt-ins returns **0**. Falsifier: the grep.
- G2: `gemm_paths_bench` f32 default row is `sgemm_blocked`'s number ±5% on the same host; error ≤ 1e-5 rel at every shape. Falsifier: the bench.
- G3: burn tensor suite stays 1826/1826 against the W1 fork. Falsifier: `cargo test -p burn-backend-tests --features ndarray --test tensor`.
- G4: no new `#[cfg(target_arch)]`, `_mm*`, or `core::arch` above `simd_*.rs` / `kernels_avx512.rs`. Falsifier: `simd-savant` grep.
- G5: every new test passes D-GTM-F5's input rule — a reviewer can grep the fixture for `* 0.5` / `* 0.25` / `as f32` on small ints and find none in an accuracy test.

## §7 — Worker allocation (by role, per the workspace model policy)

- **Opus:** this plan; W0 result synthesis (D-GTM-0a..0f read together decide W1's shape); every disable-run adjudication; central `fmt`/`clippy -D warnings`/tests; all commits/pushes.
- **Sonnet:** each W1 demotion commit (one file, one shape: "make these symbols `pub(crate)`, cite the caller count"); the f64 bench extension; the `mask_to_row_indices` primitive against a written spec. Edit-only; no `cargo build/check` per `.claude/rules/agent-cargo-hygiene.md`; the orchestrator compiles once.
- **Haiku:** only the guarded-executor role — run a pre-written `#[ignore]` bench card and paste the tail.

## §8 — Open questions (answered by W0, not by discussion)

1. Is `sgemm_blocked`'s 12% lead over matrixmultiply stable across shapes and hosts, or an artifact of square 2ⁿ sizes? (D-GTM-0c)
2. Does the reverse-engineered MKL path have any caller today, or is it a second facade with zero consumers? (D-GTM-0f decides whether W1 demotes it or deletes it)
3. At what mask density does compacted-index packing beat per-row bit tests? (D-GTM-0e)
4. Is there a real shape where an f32 AMX path wins? Nothing measured says yes; the opt-ins exist so the question stays cheap to re-ask.
