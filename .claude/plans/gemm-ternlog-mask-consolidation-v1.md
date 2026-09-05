# gemm-ternlog-mask-consolidation-v1 — one GEMM entry per dtype, masks as the prefilter, ternlog as the mask ALU

> **Status:** DRAFT v1.3 (2026-09-05) — §11.10 strengthens the invariant to
> `substrate == mask geometry == projection surface`: the 3-D field (diamond tracts, the
> "cube") is never allocated — it is a mask-address projection of the 2-D 6×2×8 surface,
> recovered only when a question requires it. "Holographic" gets a falsifiable definition
> (recoverability, D-GTM-0l) paired with non-allocation (D-GTM-0k). §11.9's diamond flag
> resolved [S]→[H]: bonds are implied, not walked, so no arity conflict with the trie.
>
> **Status:** DRAFT v1.2 (2026-09-05) — §11 folds in the operator's grey/white-matter
> statement: hex (grey, 6×2×8 rails, local permeability) and trie (white, 3×2 path
> prefixes, routing) are ONE 12-byte register read two ways; `substrate == selection ==
> routing` is the V3 4+12 facet doctrine as a compute model. §9 R1 re-graded [S]→[H];
> D-GTM-5 corrected a THIRD time (pack consumes mask words, zero index materialization);
> K0..K7 = the SPO 2³ triadic projections [H]; six operator falsifiers D-GTM-0g..0l with
> the E-Q8 degree-ablation control mandatory.
>
> **Status:** DRAFT v1.1 (2026-09-05) — §9 folds in the operator's Mississippi Queen
> metaphor: M1 CORRECTS D-GTM-F4/D-GTM-5 (panel-ahead expansion, not a whole-board
> prologue), M1b names where the cache lives, M2 re-shapes D-GTM-0e into a ladder,
> M3 turns the T2→T1 prohibition into a budget. R1 (the hexagon) is marked rhyme.
>
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

**Counted, not estimated:** 54 entry points, 12 files, **4 unified.** ⊘ **§10 D-GTM-0b corrects this count's blind spot:** `hpc/blas_level3.rs` returned zero `pub fn` hits because its surface is a **trait** (`BlasLevel3<A>`, six methods, blanket impl, re-exported at `simd.rs:656`) dispatching through `BlasFloat::backend_gemm` — a dtype-generic facade that already exists beside the four free functions. A `pub fn` grep cannot see a method; any re-inventory must grep `fn`.

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
| D-GTM-F4 | ⊘ **AMENDED AGAIN by §11.4.** The pack CONSUMES MASK WORDS directly (tzcnt / vpcompress inside the panel window) — **no index list exists at any point**, not a `Vec<u32>` prologue (v1), not a panel-ahead `ArrayVec<u32>` (v1.1). Cache key stays `(mask generation, panel index)`; the reusable object generalizes to a permeability codebook entry (§11.6). | `substrate == mask geometry == projection surface` (§11.10, strengthening §11.4) |
| D-GTM-F5 | Every accuracy test in this surface uses inputs whose significands exceed 8 bits, and tolerances at f32 grade (1e-5) for f32 APIs. | the vacuous `(i+j)*0.5` test that hid the bf16 loss (#303) |
| D-GTM-F6 | Every new kernel lands with a two-sided pin: the fast path must beat the reference by a stated factor AND the reference must still be measurably slower — so a regression in either direction fails. | `three_pass_split_beats_one_bf16_pass` pattern |

## §5 — What is PROPOSED (plan register), substrate-first

### Wave 0 — measure before minting (no production code; all probes `#[ignore]` + `--release`)

| id | probe | falsifier / gate |
|---|---|---|
| D-GTM-0a | ✅ **RUN — §10.** Both pairs divergent. `bf16_tile_gemm_16x16` = polyfill vs dispatcher, legitimate, facade already renames one `_amx`. `sgemm_blocked` in `simd_avx2.rs` = a naive scalar triple loop (neither AVX2 nor blocked) — a naming defect. | v1 expected "byte-identical → delete"; neither pair is |
| D-GTM-0b | ✅ **RUN — §10.** A `BlasLevel3<A>` trait, 6 methods, blanket impl, → `BlasFloat::backend_gemm` (f32/f64 only). CLAUDE.md was right; the grep was blind. **Re-frames D-GTM-F3: two facades already exist.** | found the surface; the macro hypothesis was wrong |
| D-GTM-0c | Extend `gemm_paths_bench` to f64 (`dgemm_blocked` vs `gemm_f64_tiled_fma` vs matrixmultiply) and to non-square / K-tail shapes (e.g. 1000×1000×1000, 17×33×15 scaled). | if `sgemm_blocked` loses on any tail shape, D-GTM-F1 gains a shape guard, not a revert |
| D-GTM-0d | **The MKL-ternlog hunch:** in `backend/mkl.rs` `sgemm` (`:384`), time the tail-lane handling with the current idiom vs one `mask_ternlog` select, K∈{255,257,1023,1025}. | < 3% end-to-end → record as shape-only win, no speed claim; ≥ 3% → W1 item |
| D-GTM-0e | ⊘ **RE-SHAPED by §9 M2 into a LADDER:** `pruned_gemm_rows` (`prefilter.rs:189`) measured over lookahead ∈ {1,2,4,8} panels × density ∈ {10%,50%,90%}, per-row bit-test as the floor. | a single crossover cannot express a ±1-adaptive depth; the ladder decides the step size, or proves depth inert |
| D-GTM-0f | ✅ **RUN — §10.** `pruned_gemm_rows` and `mixed_precision_gemm` have **0 callers anywhere**; `blas_gemm` 0 external; only `bf16_tile_gemm_16x16` has real external consumers (5), reached by two paths that resolve to two different bodies (one an iron-rule violation, reported). | D-GTM-5 is a FIRST WRITER, not a migration |

### Wave 1 — ndarray (the facade and the backends)

- D-GTM-1: `backend::gemm_f32` → runtime `avx512f` check → `sgemm_blocked` (one `unsafe` call with a SAFETY comment; the `#[target_feature]` boundary stays inside `kernels_avx512.rs`), else matrixmultiply. `dgemm` per D-GTM-0c. Two-sided pin per D-GTM-F6.
- D-GTM-2: demote every non-facade GEMM symbol per D-GTM-F3, in one commit per file, each with the W0-0f caller count in its message. `simd_runtime/matmul.rs` becomes a thin re-export of the facade.
- D-GTM-3: `hpc/amx_matmul::matmul_f32` (the ArrayView API) delegates to `backend::gemm_f32` — already does post-#303; make the delegation explicit in the doc and drop `pack_contig` when the views are already contiguous.
- D-GTM-4 (gated on D-GTM-0d ≥ 3%): tail-lane selects in `mkl.rs` `sgemm`/`sgemm_bf16`/`sgemm_int8` become `mask_ternlog` idioms.

### Wave 2 — the mask→GEMM seam (ndarray + lgj-abi, one PR each)

- D-GTM-5 (ndarray): ⊘ corrected three times, see §11.4 — `pack_a_masked_f32(a, lda, mask, row_cursor, kc, k_start, buf) -> rows_packed` consumes mask words directly; **no index list, no `Vec<u32>`, no `ArrayVec<u32>`**. `pruned_gemm_rows` has zero callers (§10 0f) so this is a first writer. Ladder per D-GTM-0e; bytes-materialized per D-GTM-0k must be ≈ 0.
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

## §9 — The Mississippi Queen shape (operator metaphor, 2026-09-05) — v1.1 AMENDMENT

**The game.** A paddle-steamer race whose river board does not exist in advance:
hex tiles are laid a few ahead of the lead boat, never the whole course. Each turn
you set speed by **±1 only** and commit it *before* moving that many hexes. One
direction change is free; extras are paid from a fixed **coal** budget. Several
boats race the same river, and a tile laid for the leader is free for everyone
behind.

Graded per the workspace rule (`cross-domain-synthesizer`: shared MECHANISM is
transferable, mere rhyme is decorative and must be labelled). Three mechanisms,
one rhyme.

### M1 — Reveal ahead of the cursor; never lay the whole river [G] — **CORRECTS D-GTM-F4 / D-GTM-5**

D-GTM-F4 as written says the compacted row-index list is *"built once per mask
generation"*. That is laying the entire river before any boat moves: `O(n_rows)`
memory, and it pays for rows the GEMM may never reach — a pruned GEMM can stop
early, and a blocked GEMM only ever needs the panel under its cursor.

The premise that makes the correction concrete is already in the code:
`pack_a_f32` (`backend/kernels_avx512.rs:552`) walks
`while ii + SGEMM_MR <= mc`, addressing `a[(i_start + ii + ir) * lda + …]`.
**Packing already has a cursor.** The mask→index expansion belongs at that same
cursor, in that same loop, one panel ahead — not in a prologue.

Consequence, replacing D-GTM-5's signature:

```rust
// WAS (v1): whole-board prologue, heap, pays for unreached rows
fn mask_to_row_indices(mask: &[u64]) -> Vec<u32>

// IS (v1.1): one panel ahead of the pack cursor, stack, no hot-loop alloc
fn next_panel_indices(mask: &[u64], cursor: usize, mr: usize) -> ArrayVec<u32, SGEMM_MR>
```

This also satisfies `.claude/rules/data-flow.md` §1 ("never allocate inside a hot
loop — slice into pre-allocated storage"), which the `Vec<u32>` version quietly
violated.

### M1b — A tile serves every boat behind it [G] — this is *the* amortization

Several boats race one river. The leader pays to reveal a tile; everyone behind
crosses it free. That is the amortization the operator named, and the game says
**where the cache lives**: on the *tile*, not on the *boat*.

So the cache key is `(mask generation, panel index)` — **not** the call. Two GEMMs
against the same mask generation reuse the same expanded panels; a new mask
generation invalidates them wholesale (the registry already does exactly this to
`cached_carving`, `lgj-abi/registry.rs:842-849`). A per-call cache would re-lay the
river for every boat and amortize nothing.

### M2 — Speed changes by ±1 and is committed before the move [H]

Lookahead depth (how many panels ahead the expansion runs) is a state variable
that moves **one step at a time** and is **committed before the panel is entered**.
Two properties, both load-bearing:

- *Hysteresis* — it cannot be re-derived per row, which is what stops a
  per-call heuristic from thrashing between depths on adjacent panels.
- *Commit-ahead* — you cannot discover mid-panel that you needed a deeper
  lookahead; by then the pack loop is already running.

**Changes D-GTM-0e:** it measured a single crossover density. It now measures a
**ladder** — lookahead ∈ {1, 2, 4, 8} panels × density ∈ {10%, 50%, 90%} — because
a single crossover cannot express a ±1-adaptive depth. Graded [H]: the ladder shape
is argued, the step size is not yet measured.

### M3 — Coal: a bounded budget for extra maneuvers [H]

Re-chaining the ternlog predicate mid-stream (the mask gains a conjunct, or changes
shape) is a maneuver paid from a fixed budget. When the budget is spent you commit
to the mask you hold rather than re-deriving it.

This is what keeps *chaining* from degenerating into *re-evaluate the predicate per
panel* — the exact T2→T1 violation `membrane-tiers.md:105` already forbids as a
prohibition. The game supplies the better form: **a budget rather than a ban**, so
the legitimate mid-stream re-chain stays possible and the pathological one runs out
of coal.

### R1 — The hexagon itself is rhyme, pending one operator word [S]

Six is conspicuous on both sides: the game moves on 6-neighbour hexes; this
substrate carves the 12-byte V3 facet as `6×(u8:u8)` and the HHTL path as 6 bytes
= CAM-PQ `6×256`. **They are not obviously the same six.** The game's six is
*adjacency* (which cell may I move to next); the substrate's six is *field carving*
(which byte pair means what). Adjacency and carving are different mechanisms that
happen to share a cardinality — the textbook rhyme signature.

Left [S] and unbuilt. If the operator meant the **rails** specifically, this is
promoted and gets its own section; nothing in M1-M3 depends on it either way.

### What this amendment does NOT touch

The AMX verdict (§1.3, measured), the facade consolidation (D-GTM-F1/F3), and the
W0 static probes (0a, 0b, 0f) are unaffected — the metaphor is about *when work is
done and who pays for it*, not about which kernel is fastest or how many entry
points exist.

## §10 — WAVE 0 RESULTS (run 2026-09-05, static probes 0a / 0b / 0f)

Three of the six W0 probes are measurement-free (greps and body diffs) and are run
here. Each corrected something in §1.1's inventory — which is the point of running
them before building anything.

### D-GTM-0a — the two duplicate names: BOTH divergent, only ONE is a defect

| pair | `simd_*` body | `hpc/` or `backend/` body | verdict |
|---|---|---|---|
| `bf16_tile_gemm_16x16` | `simd_ops.rs`, 37 lines — decode BF16→f32, then F32x16 + FMA. The **polyfill**. | `hpc/bf16_tile_gemm.rs`, 17 lines — `amx_available() \|\| avx512bf16` → VNNI-pack → tile tiers. The **dispatcher**. | **NOT a defect.** Backend vs dispatcher, legitimately distinct. The facade already disambiguates: `simd.rs:714` re-exports the hpc one **renamed** `bf16_tile_gemm_16x16_amx`; `simd.rs:744` exports the polyfill under the plain name. |
| `sgemm_blocked` | `simd_avx2.rs:462`, 14 lines — a **naive scalar triple loop** (`for i / for j / for p { sum += … }`). | `backend/kernels_avx512.rs:665`, 52 lines — the real packed-panel MR=6/NR=16 kernel. | **DEFECT, naming.** The `simd_avx2.rs` body is neither AVX2 nor blocked; the file name, the function name, and the body disagree three ways. |

⊘ **Corrects v1's D-GTM-0a**, which anticipated "byte-identical bodies → delete
one". Neither pair is byte-identical and neither should be deleted. The real
finding is narrower and different: one legitimate tier pair (already handled by a
facade rename) and one mislabelled scalar fallback.

### D-GTM-0b — `blas_level3.rs` is not empty; the inventory grep was blind

393 lines, **zero `pub fn`** — because the surface is method-shaped:

```rust
pub trait BlasLevel3<A> {
    fn blas_gemm(&self, alpha: A, b: &Self, beta: A) -> Array<A, Ix2>;
    fn blas_gemm_into(&self, alpha: A, b: &Self, beta: A, c: &mut Array<A, Ix2>);
    fn blas_syrk (&self, uplo: Uplo, alpha: A, beta: A, c_init: Option<&Self>) -> Array<A, Ix2>;
    fn blas_symm (&self, side: Side, uplo: Uplo, alpha: A, b: &Self, beta: A, c_init: Option<&Self>) -> …;
    fn blas_trmm (&self, side: Side, uplo: Uplo, alpha: A, a_tri: &Self) -> Array<A, Ix2>;
    fn blas_trsm (&self, side: Side, uplo: Uplo, alpha: A, b: &Self) -> Array<A, Ix2>;
}
impl<A, S> BlasLevel3<A> for ArrayBase<S, Ix2> where A: BlasFloat + Float + AddAssign, S: Data<Elem = A>
```

Re-exported at `simd.rs:656`. CLAUDE.md's "BLAS L3 (gemm, syrk, trsm, symm)" claim
was **accurate all along**; §1.1's `grep 'pub fn …gemm'` simply could not see a
trait. **Any future inventory of this crate must grep `fn`, not `pub fn`.**

**And it dispatches through a facade that already exists.** `blas_gemm`'s body is
`A::backend_gemm(m, n, k, alpha, …)` — a method on `BlasFloat` (`backend/mod.rs:75`),
implemented for **f32 and f64 only** (`:83`, `:110`), with `f32::backend_gemm` →
`gemm_f32`.

⊘ **This materially re-frames D-GTM-F3.** v1 said "the facade is
`backend::{gemm_f32, gemm_f64, gemm_bf16, gemm_i8}`; consolidate everything else
under it." In fact **two facades already exist side by side**:

| | shape | dtypes | callers (0f) |
|---|---|---|---|
| `BlasFloat::backend_gemm` | dtype-**generic** trait method | f32, f64 | reached only via `BlasLevel3` |
| `backend::{cblas_sgemm, cblas_dgemm, gemm_i8, gemm_bf16}` | four **free functions** | f32, f64, i8, bf16 | 10 / 2 / 6 in-crate |

The consolidation question is therefore **not** "build one facade" but "**which of
the two existing facades is canonical**". The constraint that decides it is already
in the source: `BlasFloat`'s impl bounds require `num_traits::Float`, so **i8 and
bf16 cannot join it** without a second trait or a bound change. A generic facade
that structurally excludes half the dtypes is not the canonical one. Recorded as
the first thing W1 must settle; no verdict claimed here.

### D-GTM-0f — caller census: two of the plan's own load-bearing symbols are DEAD

| symbol | in-crate | external | note |
|---|---|---|---|
| `bf16_tile_gemm_16x16` | 21 | **5** | the only symbol with real external consumers |
| `int8_gemm_vnni` | 12 | 0 | |
| `batched_gemm_f32` | 11 | 0 | |
| `cblas_sgemm` | 10 | 0 | |
| `gemm_f64_tiled_fma` | 9 | 0 | |
| `gemm_bf16` | 6 | 0 | |
| `sgemm_blocked` | 3 | 0 | `pub(crate)`-reachable only |
| `gemm_i8` | 2 | 0 | |
| `blas_gemm` | 2 | 0 | decl + impl; the trait facade is **unused** |
| **`pruned_gemm_rows`** | **0** | **0** | ⚠ |
| **`mixed_precision_gemm`** | **0** | **0** | ⚠ |

**`pruned_gemm_rows` has zero callers.** §2.3 called it "the seed" and "the ONLY
existing mask→GEMM bridge", and §9 M1 rewrote its signature — all of that was
reasoning about **dead code**. It is still the right *shape* to build on, but the
plan must stop describing it as an existing integration: nothing integrates it.
D-GTM-5 is therefore a **first writer**, not a migration.

### A live iron-rule violation, found incidentally by 0f

The five external `bf16_tile_gemm_16x16` references reach it by **two different
paths**, and because 0a proved the bodies divergent, they are calling **different
functions**:

- `lance-graph/crates/symbiont/src/domino.rs:27` — `use ndarray::simd::{amx_available, amx_report, bf16_tile_gemm_16x16}` → resolves to the **polyfill** (F32x16 decode), *not* a tile op. Its own module doc at `:8` reads *"which `bf16_tile_gemm_16x16` calls before any tile op"*, and it imports `amx_available` alongside — so the call site appears to expect the AMX dispatcher and receives the polyfill. `..._amx` is the name it wants.
- `lance-graph/crates/thinking-engine/examples/amx_bf16_probe.rs:15` — `use ndarray::hpc::bf16_tile_gemm::bf16_tile_gemm_16x16`, reaching **past the facade into `hpc::`**. That is the exact form the workspace iron rule forbids (*"all SIMD from `ndarray::simd`, never `ndarray::hpc::*`, never raw intrinsics"* — lance-graph-java `CLAUDE.md`, `abi.md` §8). It gets the dispatcher, correctly, by an illegal route.

Both are **lance-graph** call sites, not ndarray's, and `symbiont` is
⊘ DEPRECATED (operator no-go, 2026-08-18) — so this is **reported, not fixed**, and
belongs to whoever next touches those files. ndarray's own facade is correct: the
rename at `simd.rs:714` is precisely the disambiguation these call sites failed to
use.

### W0 remaining

`0c` (f64 + tail-shape bench), `0d` (the MKL-ternlog tail hunch), `0e` (the M2
lookahead × density **ladder**) are measurement probes and are not run here. Note
0e's target is now known to be dead code — the ladder measures a kernel with no
consumers, which is fine for a probe and must not be described as measuring
production behaviour.

## §11 — v1.2: the hex/trie field is ONE packed-address substrate read two ways (operator statement, 2026-09-05)

### 11.1 The statement, compressed to its load-bearing claims

> *Treat the hexagon field as digital grey/white matter over one packed-address
> substrate, not as two separately materialized graphs.*

1. **Grey = local hex state.** Each cell is a fixed-width state carrier — the
   existing **96-bit 6×2×8** geometry. The same geometry is substrate AND mask.
   Local learning changes *admissibility/permeability masks*, never an external
   pointer structure.
2. **White = trie routing through packed location.** Long-range edges are never
   object lists. Location is hierarchical *in the address bits*; each prefix is a
   trie level/region; navigation is successive prefix refinement. Every prefix is
   itself a mask: `ADDRESS & PREFIX_MASK == PREFIX`. A tract is
   `(prefix, mask, learned transition)`, not a materialized path.
3. **TERNLOGQ is the membrane algebra.** `U' = ternlog(U, local_membrane_mask,
   trie_route_mask)`; the surviving bits ARE the lawful next hex / prefix
   refinement. Cached masks (`current_state`, `local_hex_mask`,
   `trie_prefix_mask`, `learned_relation_mask`, `attention/focus_mask`) chain in
   the same native bit geometry.
4. **Learning is a codebook, not addresses.** `context mask + relation atom +
   packed prefix delta → permeability`. After exposure, most structure resolves
   through learned vocabulary; only residual novelty alters the membrane. The
   R2IL/C64 experiment is the model.
5. **The invariant: `substrate == selection geometry == routing geometry`.**
   Expanding a mask into IDs, materializing a neighbour list, or converting the
   trie into an edge table *for the hot path* is the loss condition.
6. **The hypothesis is NOT "TERNLOGQ replaces GEMM."** GEMM is attractive when
   information is dense; a hex/trie field may win when cognition is mostly
   *successive elimination of possibility*. Grey squeezes locally; white moves
   the constraint field cheaply across distance.
7. **Underlined:** white matter is not another data structure — it is an
   *interpretation of packed location prefixes*. Hexagon supplies neighbourhood;
   trie supplies scale; TERNLOGQ supplies permeability.

### 11.2 Why the measured hex record (Q6 / Q7 / Q8) does NOT close this — and what it DOES bind

Three board entries measured "hex" and found it wanting:
`E-Q6-HEX-FAILS-CONTENT-ADDRESSING-…-1` (learns less *and* interferes more),
`E-Q7-…-COMPLEMENTARY-NOT-COMPETING-1` (frequency sizing rescues learning, not
interference), `E-Q8-THE-SIX-DOES-NO-WORK-…-1` (degree-1 ablation: identical
completion at 5.5× less memory — *"the information is in the PAIR, not in the
neighbourhood's shape"*). The #1023 audit found **zero** hex adjacency by grep
anywhere in lance-graph / ndarray / OGAR.

**Those experiments tested a different claim.** Their B-arm was a *learned
association overlay* — a co-occurrence neighbourhood graph for macro recall,
scored on completion / false resonance / interference. §11.1 is a *compute and
memory* claim: propagation cost and bytes materialized under successive
elimination, against GEMM. That is precisely the bar `r2il-machine-semantic-
contract-v1` §7.2 already sharpened: *"if hex wins, it wins as a COMPUTE
topology — never retroactively as an explanation of the 96-bit register."*
§11.1 claims exactly and only that.

**What the record binds, non-negotiably:**
- **E-Q8's lesson becomes a mandatory control.** *"A locality claim needs a
  DEGREE ablation, not only a wiring null."* Every probe in §11.7 runs the hex
  arm at degree 6 **and** degree 1; any advantage that survives degree 1 is not
  hexagonal and must not be reported as such.
- The ratified demarcation is retained verbatim: *"White Matter ist Wahrheit und
  Zwang. Grey Matter ist Hypothese und Plastizität. … Hexagon ist noch gar
  nichts außer einem Kandidaten für lokale Rechengeometrie."* Grey is
  hypothesis; a learned permeability mask is never a second truth
  (`E-*-a-macro-never-becomes-a-second-truth`, the B4 invariant).

⊘ **§9 R1 is therefore RE-GRADED, [S] → [H]-with-falsifier.** R1 said the game's
six (adjacency) and the substrate's six (carving) were different mechanisms.
That distinction stands — and §11.1 resolves it by assigning them to the two
tissues: adjacency is *grey* (the six neighbours, an unproven compute topology),
carving is *white* (the six rails, the packed address). Not rhyme, not the same
six — two readings of one register (11.3).

### 11.3 6×2×8 and 3×2 are the SAME twelve bytes — this is already canon

| reading | what the 12-byte V3 payload means | tissue |
|---|---|---|
| **rails** — `6×(u8:u8)` = 6×2×8 = 96 bits | six `palette256:palette256` pairs; the colon carries the distribution (`E-PALETTE256-IS-A-NEEDLE-THE-COLON-IS-THE-DISTRIBUTION-1`) | **grey** — local state |
| **path** — `HEEL:HIP:TWIG` = 3 tiers × 2 axes = 6 bytes | the CAM-PQ `6×256` code; `path distance = 3 tier-table lookups, O(1)`; longest-prefix binding; `is_ancestor_of` = centroid-tree containment (OGAR `CLAUDE.md` §Tier interpretation) | **white** — routing |

`le-contract.md` §3 already says it: *"the 12B is a dumb byte register the
ClassView projects — it holds every sanctioned reading at once."* So the
operator's invariant `substrate == selection == routing` is **the V3
content-blind 4+12 facet doctrine restated as a compute model.** Nothing new is
laid out; the proposal is a way of *executing over* the existing register. That
is what makes it admissible under the STOP rule — no new tissue.

The L0…L5 ladder (`region → basin → tract → bundle → hex → local state`) is the
nibble-tree: 1 hex digit = 1 level of the 16-ary tree, tier-of-level = `level >> 2`
(a shift, never a branch). The trie already exists; §11.1 asks that it be *read as
a mask* rather than *walked as a structure*.

### 11.4 Where the tree currently VIOLATES the invariant — including my own §9

The loss condition is materializing IDs on the hot path. Census:

| site | what it materializes | verdict |
|---|---|---|
| `lance-graph/…/blasgraph/heel_hip_twig_leaf.rs` — `heel_search` → `Vec<SearchHit>` = `Vec<(usize, u32)>`, `k = 50` survivors **per tier**, sorted + truncated | a gathered row set, four times per query | **violates** — the semantic HHTL arm is the anti-pattern §11.1 names |
| `ndarray/src/hpc/splat3d/depth_cascade.rs` — `cascade_blocks` → per-block `BlockDepthDecision { block_index, tier_reached, action, … }` | an ID-carrying decision per block | **violates** (spatial HHTL arm) — though `HhtlAction::{Reject, KeepCoarse, Refine, ProjectExact, RenderExact}` is already a 5-valued *"lawful next refinement"*, exactly §11.1's surviving-bits semantics wearing an enum |
| `ndarray/src/hpc/splat3d/tile.rs` — packed `u64` key `(tile_id << 32) \| depth_bits`, sorted tile-major | **nothing** — routes by packed-key prefix | **conforms** — the renderer already does white-matter routing by address prefix |
| **this plan, §9 M1** — `next_panel_indices(...) -> ArrayVec<u32, SGEMM_MR>` | a per-panel index list | **violates.** Smaller than v1's `Vec<u32>`, still an ID expansion on the hot path. |

⊘ **D-GTM-5 is corrected a THIRD time.** v1: whole-board `Vec<u32>`. v1.1 (§9):
panel-ahead `ArrayVec<u32>`. v1.2: **the pack consumes mask words directly** —
iterate set bits with `tzcnt` inside the panel window (or `vpcompress` on
AVX-512), gathering rows straight into the packed panel buffer, **zero index
materialization**:

```rust
// v1.2 — the pack reads the mask; no index list exists at any point
fn pack_a_masked_f32(a: &[f32], lda: usize, mask: &[u64], row_cursor: usize,
                     kc: usize, k_start: usize, buf: &mut [f32]) -> usize /* rows packed */
```

Each revision removed one more layer of materialization; the operator's
invariant names the fixed point.

### 11.5 K0…K7 — the eight "terniating" masks are the SPO 2³ triadic projections [H]

A 3-input `ternlog` immediate is an 8-entry truth table indexed by
`(a<<2)|(b<<1)|c` (`simd.rs:559-563`). Eight is also **exactly** the query
grammar the workspace already carries:

```rust
// lance-graph/.claude/knowledge/cam-codebook-resonance-projection.md §SPO 2^3
pub enum TriadicProjection { Abc, AbAskC, AcAskB, BcAskA, AOnly, BOnly, COnly, Background }
```

*"For a triad (A, B, C), there are 2³ observation/query masks … The 2³
structure is not decorative. It is the query grammar for the CAM field."* So
`K0…K7` reads as: the eight presence-patterns of `(S, P, O)`, each selecting
which of the triadic pressures (`S×P → O`, `S×O → P`, `P×O → S`) applies, and
"terniating" = iterating the ternlog truth table over them. **[H] pending one
operator word** — the mapping is exact in cardinality and in role, but the name
`K0..K7` itself is not in the tree.

### 11.6 What amortizes, precisely

§9 M1b said the cache key is `(mask generation, panel index)`. §11.1 generalizes
the *tile*: the reusable object is a **codebook entry**
`(context mask, relation atom, packed prefix delta) → permeability mask`. It is
laid once — by whichever traversal first needs it — and every later traversal
that hits the same `(prefix, relation)` crosses it for free. That is the
Mississippi Queen tile at the level of *learned transitions*, and it is what
keeps learned semantics low-entropy (§11.7 probe 6 measures exactly whether it
stays that way).

### 11.7 The falsification program — six operator probes, plus the E-Q8 control

Each is a W0 probe; none is production code. Numbering continues §5.

| id | probe (operator's words) | the measurement | the E-Q8 control |
|---|---|---|---|
| D-GTM-0g | mask/trie propagation vs dense GEMM for an equivalent sparse learned transition | wall time + result equality, same transition matrix realized both ways | hex arm at degree 6 **and** 1 |
| D-GTM-0h | does chained VPTERNLOGQ stay register/cache resident as depth increases | `perf` L1/L2 miss rate vs chain depth 1…32; the knee is the finding | — |
| D-GTM-0i | sweep unknown density almost-empty → almost-full | density ∈ {1,5,10,25,50,75,90,99}% | — |
| D-GTM-0j | the crossover where GEMM wins | from 0g × 0i: a density-vs-size frontier, not a single number | — |
| D-GTM-0k | **bytes materialized per inference step** | count every heap/stack byte that is not the mask itself; the hex/trie path must approach **zero** | this is the invariant's own falsifier |
| D-GTM-0l | can packed-prefix routing express all required long-range transitions without exploding codebook entropy | entries needed vs transitions covered, on the R2IL/C64 ore; the C64 vocabulary-resolution rate is the reference | — |

**Pre-registered kill condition (one, stated before any run):** if at every
density in 0i the GEMM arm is both faster AND materializes fewer bytes than the
hex/trie arm, §11.1's hypothesis is false *for this substrate* and is recorded
as such — the same way Q6 was.

### 11.8 Restated non-goals

- Not "TERNLOGQ replaces GEMM" (§11.1 pt 6 — the operator's own fence).
- Not a new layout, a new crate, or a second graph. The register is the V3
  facet; the trie is the nibble tree; both exist.
- Not a re-run of Q6/Q7/Q8's association task. Different claim, different
  metrics, same degree-ablation discipline.

### 11.9 The Panela primitive, and hex + diamond as the two lattices (operator infographic, 2026-09-05)

**Panela** — the DDR construction toy: one flat piece with an E-E comb profile that
interlocks with its own kind. *"One shape. Many worlds. Positive shape =
connectivity. Negative space = admissibility."* That is the §11.1 invariant as a
physical object: the same piece IS the structure and IS the selection, and the
infographic draws the 96-bit `6×2×8` register (six rows L0…L5 × byte 0 / byte 1)
**as** the E-profile. Substrate S and mask M are the same silhouette; ternlog
combines them without either ever leaving the lane.

**Two lattices, one per tissue** — "from planar to space-filling":

| lattice | coordination | character | tissue |
|---|---|---|---|
| **hexagonal** (Bienenwaben) | 6, isotropic, planar | efficient packing, natural for local reasoning | **grey** — dense local recurrence |
| **diamond** (Bindungen) | 4, tetrahedral, 3-D | strong directional bonds, efficient long-range structure | **white** — cross-scale tracts |

A hex sheet per level, diamond links between levels: the cortical sheet with its
tracts. This is the first *geometric* statement of white matter in this plan —
§11.1 gave it as an *interpretation of prefixes*; the diamond gives it a shape.

**One mismatch to flag, not resolve.** A diamond lattice has coordination **4**;
the canon's trie is **16-ary** (1 hex digit = 1 level, `FAN_OUT=16`, tier-of-level
= `level >> 2`). A tetrahedral tract therefore does not map one-to-one onto a
nibble level — it maps onto **two bits** of one. Either the diamond is a
*visualization* of the four sub-branches a nibble refines through (in which case
it is rhyme, and harmless), or it is a claim that white-matter routing is 4-ary
at each step (in which case it conflicts with the 3×4 vs 4×3 standing watch and
needs its own probe). **Graded [S] until the operator says which.** Nothing in
§11.7's six probes depends on it.

**The neuron reading** (dendrites = incoming masks, membrane = permeability mask,
axon = outgoing mask, learning = mask update, state = the 96-bit substrate) and
**constraint soaking** (`U₁ = U₀ ⊗ M₁`, `U₂ = U₁ ⊗ M₂`, … until residual) are §11.1
points 1 and 6 restated; recorded as vocabulary, not as new claims.

### 11.10 The amended invariant: `substrate == mask geometry == projection surface` (operator, 2026-09-05)

**What was missing, in the operator's words.** *"Make the 96-bit object
holographic"* risked being metaphor, because a volumetric cube wants a huge
number of independently addressable states — the 3-D geometry existed with no
economical way of specifying its interior. The Panela / photolithographic layer
closes that: **the cube is never stored voxel-by-voxel.**

```
apparent 3-D field  (diamond bonds, tracts, the "cube")   ← NOT materialized
        ⇅ projection
planar bit geometry: EE EE EE …  = 6×2×8 register, masks + packed location
```

**Holographic, defined falsifiably:** not "every voxel exists" but *the
information required to reconstruct / address the relevant 3-D relation is
distributed through the 2-D representation.* You don't fill the hologram with
bits; you fill the surface with enough invariants that the volume can be
recovered *when something asks a question that requires it.*

**Assignment of parts** (replaces §11.1 pt 7):

| supplies | from |
|---|---|
| depth | packed location |
| local curvature | hex adjacency |
| scale | trie prefixes |
| permeability | masks |
| the dumb physics | VPTERNLOGQ |

**The invariant, strengthened.** §11.1's `substrate == selection geometry ==
routing geometry` becomes

> **`substrate == mask geometry == projection surface`**

The hardware inversion is the point: not *3-D problem → flatten awkwardly onto
silicon*, but *embrace the 2-D silicon-like representation and make the
higher-dimensional object a mask-address projection of it.* VPTERNLOGQ stays
wonderfully boring — three bit fields, one Boolean function, no knowledge of
neurons, cubes, hexagons or ontology. **The meaning lives in how the fields are
laid out, not in the instruction executing them** — which is `membrane-tiers.md`'s
T1 rule (a primitive is dumb; the tier above it carries the meaning) stated from
the other side.

⊘ **§11.9's diamond flag is RESOLVED, [S] → [H].** The mismatch I flagged — a
tetrahedral tract (coordination 4) against the 16-ary nibble trie — assumed the
diamond bonds were *walked*. They are not: they are *implied* by address + masks
and recovered on demand. Nothing is 4-ary per step because nothing steps. The
diamond is the shape of the projection, not a level of the trie. No conflict with
the 3×4 standing watch; nothing to probe about arity.

**What IS now falsifiable, and where it already sits in §11.7:**

- *Holographic recoverability* = **D-GTM-0l**: can packed-prefix routing +
  masks + codebook express every required long-range relation without codebook
  entropy exploding? If the surface's invariants are insufficient, the volume
  cannot be recovered and the claim fails there — measured on the R2IL/C64 ore.
- *The volume is never allocated* = **D-GTM-0k**: bytes materialized per
  inference step → 0. A "projection" that secretly allocates the cube shows up
  as bytes.

Those two probes together are the hologram's test: recover the relation (0l)
without allocating it (0k). Nothing else in the program changes.
