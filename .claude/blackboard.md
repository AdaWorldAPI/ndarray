# Current epoch (2026-05-26) — splat / palette / pillar / 3DGS

> **Read this first.** The "Polyglot Notebook" architecture below is a
> separate/older program, not the current epoch.

## 2026-07-06 (later) — `ndarray::simd::gemm_f64_tiled` surfaced (operator directive)

The crate-native tiled f64 GEMM graduated from dead code
(`backend/native.rs` private `mod scalar`, zero callers) to the canonical
simd surface: `src/simd_ops.rs::gemm_f64_tiled`, re-exported
**unconditionally** in `src/simd.rs` (alloc-free; `pub mod simd` itself is
std-gated in lib.rs — see reviewer note below if that changed).

- **Bit-exactness contract (documented on the fn):** every C[i,j] gets
  `c = c + (α·A[i,p])·B[p,j]` ascending-p, mul and add UNFUSED — the
  `*`/`+` operators on F64x8 lower to plain mul/add intrinsics on ALL
  five backends (AVX-512 `_mm512_mul/add_pd`, AVX2 per-half, NEON
  `vmulq/vaddq_f64`, WASM `f64x2_mul/add`, scalar) and Rust never
  FP-contracts explicit intrinsics → bit-identical across backends; at
  α=1 β=0 bit-identical to the naive triple-loop reference.
- Innermost j-loop vectorized on dispatched `F64x8` (one source, every
  backend, per the simd_ops polyfill model); TILE=64 blocking preserved
  verbatim from the original.
- [MEASURED] vs naive scalar triple loop, single thread, this EMR VM:
  128³ 2.23×, 256³ 2.54×, 512³ 6.74× (4.6 GFLOP/s), **bit-equal: true**
  at every size (W1a bench criterion; well above the 0.5× reject line).
- W1a compliance: parity tests = 7 new tests in
  `simd_ops::gemm_f64_tiled_tests` (fixed-seed splitmix64 corpus, 13
  shape sweep incl. multi-tile 70³, strided lda/ldb/ldc with
  sentinel-padding assert, α/β semantics, β=0-over-NaN, k=0, m/n=0,
  denormals/−0.0) — all bit-equality (`to_bits`), not tolerance. Zero
  `unsafe`. No new feature detection. Consumer sites named: the
  `direct_matmul` f64 ground-truth in `examples/subpel_tap_tile.rs` +
  `examples/gridlake_field_tile.rs`, both REWIRED to it (bit-identical
  swap — subpel prints identical numbers pre/post).
- Free-fn shape note: matches the existing simd-surface GEMM family
  (`bf16_tile_gemm_16x16`, `matmul_i8_to_i32`) — operator-directed, not
  a speculative W1a-queue addition.
- Dead code removed: `native.rs` `gemm_f64_tiled` deleted (pointer
  comment left); `gemm_f32_tiled` stays dead in `mod scalar` pending the
  same treatment (UNUSED_INVENTORY thread).
- Gates: clippy `-D warnings` clean (lib + both examples), fmt clean,
  **2182/2182 lib tests**, doctest green, `--no-default-features` build
  green.

[NOTE] `--tests` clippy surfaces 3 PRE-EXISTING test-code lints
(property_mask.rs:426 unusual_byte_groupings, bitwise.rs:637 identity_op,
palette_codec.rs:806 needless_range_loop) — not touched here; the house
gate (`cargo clippy -- -D warnings`, no --tests) is clean.

[OBSERVED, then MEASURED] `amx_available()` flipped true→false between
runs two hours apart (subpel ran AMX TDPBF16PS earlier, F32x16 polyfill
later; identical BF16-class errors either way — tier ladder correct).
**Diagnosis via `examples/amx_probe` per AMX_GOTCHAS discipline** (initial
"enablement drift / Gotcha 14 adjacent" guess was WRONG): CPUID leaf-7
TILE/INT8/BF16 bits all false, `cpu_model() = OtherX86`, `has_amx() =
false` — the **silicon identity itself changed** (morning: EMR 0xCF with
AMX). The session container was rescheduled onto non-AMX/CPUID-masked
silicon. NOT Gotcha 4 (that signature is has_amx()==true with
available==false) and NOT Gotcha 14 (corruption while available==true).
Gotcha 9's always-print-the-tier discipline is what surfaced the flip.
Consequence for remote sessions: the host under an ephemeral container
can change mid-session — re-run `amx_probe`/`amx_report()` before any
AMX-tier claim, never carry `amx_available()` results across runs.

[LOOSE END] subpel_tap_tile findings 1-3 from the same-day review still
queued (Gotcha-14 assert guard, PackedBf16B throughput leg, positive-
operand comment fix). Adversarial 3-angle verify workflow on this diff
was in flight at commit time; findings (if any) land as follow-up.

**[VERIFY OUTCOME, same day]** 3-angle adversarial review (IEEE/bit-
exactness, W1a compliance, regression-trace): **PASS / PASS / PASS**.
One convergent P1 fixed in the follow-up commit: the simd.rs re-export
comment claimed no_std availability, but `pub mod simd`/`simd_ops` are
std-gated in lib.rs — the no_std build passes because the code is
compiled OUT (comment corrected; un-gating simd_ops for a genuinely
no_std kernel is possible future work, not this diff). P2s folded in:
cross-backend bit-identity scoped to non-NaN inputs (NaN payloads are
backend-defined, WASM may canonicalize); `alpha == 0.0` non-short-
circuit documented (0·Inf=NaN propagates, −0.0 can flip, unlike BLAS
quick-return); Panics doc states which checks are skipped (m/n==0, k==0);
length-extent asserts now overflow-checked (`checked_mul`); parity corpus
widened to 70+ invocations (full 13-shape sweep × 4 α/β combos) for the
W1a "50+" letter; free-fn-shape precedent sentence added to the doc
(GEMM family: bf16_tile_gemm_16x16, matmul_i8_to_i32). Deferred as
noise: x87-only i586 excess-precision footnote (tier-2, pre-SSE2).

## 2026-07-06 — Review pass: health check green + 10 findings on subpel_tap_tile (#235)

Review-only session (no code changes). **Health check:** `cargo fmt --check`
clean, `cargo clippy -p ndarray --lib -- -D warnings` clean, clippy on the new
example clean, **2175/2175 lib tests pass** (30 ignored). The stale top-of-
CLAUDE.md "build fails (exit 101)" note again does NOT reproduce. The example
`subpel_tap_tile` runs green end-to-end on this EMR host (tier = AMX TDPBF16PS,
rel err 0.157% / 0.215%, asserts pass).

**Findings on `examples/subpel_tap_tile.rs` (PR #235), most severe first:**
1. Lines 208-209 hard-assert rel err < 0.05 on AMX-dispatched results with no
   contention guard — flakes on oversubscribed VMs per Gotcha 14 (precedent:
   `#[ignore]` gating in bf16_tile_gemm.rs:572).
2. Throughput leg (3) times per-call allocs + f32→bf16 of BOTH operands + the
   kernel's per-call VNNI pack of the CONSTANT H — printed 1.00 M/s measures
   wrapper overhead, not the tile op; use `PackedBf16B` +
   `bf16_tile_gemm_16x16_packed` (the API built for exactly this).
3. Line 167 comment "(fits u8 and i8; positive operand)" is false (x reaches
   ≈ −19.6 for r ≥ 11) — porting hazard toward the u8×i8 int8 path.
4. Reuse: `mix()` is the 5th example-local splitmix64 copy (public bit-identical
   `hpc::cam_index::SplitMix64` exists); `direct_matmul` duplicates
   `backend::native::gemm_f64` — **[MEASURED this session] bit-exact, 0/256
   lanes differ** vs the naive loop on the probe's exact operands (operator
   correction: earlier "independence" refutation was wrong — the BF16 tile
   kernel shares zero code with gemm_f64).
5. Cleanups: Vec<f32> C + copy (kernel takes &mut [f32] — stack array works),
   to_bf16 double-alloc, b_pad identical-index loop = prefix copy, direct_hv
   duplicated FIR pass, padded-16→32 scaffold now copy-pasted across 2 probes
   (suggest one public padded helper on ndarray::simd before probe #3).

**Refuted during verify:** .gitattributes deletion is the deliberate,
documented revert PR #236 (union merge mangles [[example]] blocks — no residue
found); clippy needless_range_loop claim (empirically clean under -D warnings);
f32 checksum absorption (print-only anti-DCE); transpose_matrix reuse (net
loss — Vec round-trips vs 8-line stack helper).

[LOOSE END] Findings 1-3 are worth a small follow-up PR (contention guard or
warning, packed-B throughput leg, comment fix); none applied here (review-only).

**[ADDENDUM, same day — operator challenge "our gemm, not the stoneage
external":** the first bit-exactness probe compared naive-loop vs
`backend::native::gemm_f64`, which delegates to the EXTERNAL
`matrixmultiply::dgemm` (native.rs:249; registry dep, Cargo.toml:154). Re-ran
against the crate's OWN scalar `gemm_f64_tiled` (native.rs:473, verbatim):
**bit-exact too — 0/256 lanes differ** on the probe's operands AND on random
f64 at K=64. All three (naive / own tiled / matrixmultiply) agree bit-for-bit
on these shapes. **Structural finding the challenge surfaced:** the crate's
entire PUBLIC f64 GEMM surface (`gemm_f64`, `BlasLevel3::blas_gemm` via
`backend_gemm`) is external-backed, while the own-Rust `gemm_f64_tiled` sits
dead in the private `mod scalar` with zero callers. If the policy is
own-reverse-engineered-Rust-only, the right fix for finding #4 is to surface
`gemm_f64_tiled` (or route the scalar tier of `backend_gemm` through it) and
point probes at THAT — files under the UNUSED_INVENTORY dead-code thread.]**

## 2026-07-02 (later) — bf16 tile GEMM: VDPBF16PS middle tier + PackedBf16B (loose end closed)

Closed the [LOOSE END] from the 1BRC entry below. `hpc/bf16_tile_gemm.rs`
is now a three-tier ladder — **AMX TDPBF16PS → AVX-512 VDPBF16PS →
decode+FMA polyfill** — with the polyfill kernel (`simd_ops.rs`) untouched:

- **VDPBF16PS tier** (`avx512bf16_path`, private): bf16 pairs multiplied
  natively per zmm (no bf16→f32 decode), f32 lane accumulators, SAME VNNI
  operand layout as the AMX tile → one packed buffer serves both tile
  tiers. `_mm512_dpbf16_ps` verified stable on Rust 1.94. Runtime
  `is_x86_feature_detected!("avx512bf16")` (EMR box has it).
- **`PackedBf16B`** + **`bf16_tile_gemm_16x16_packed`**: VNNI pack (and
  its per-call allocation) hoisted out of hot loops; `vnni_index(row,col)
  = (row/2)·32 + 2·col + (row&1)` supports staging B DIRECTLY in VNNI
  layout (zero pack cost — the right shape for one-hot/sparse staging).
- **`bf16_tile_gemm_tier()`**: names the tier that will run (Gotcha 9
  reporting). Re-exports via `ndarray::simd::*` (W1a surface).
- **Exactness boundary preserved (operator condition):** bit-exact across
  ALL tiers for bf16-exact integer operands with accumulation < 2^24 —
  asserted with `assert_eq!` in the new parity tests (vnni_index vs
  vnni_pack_bf16; packed==unpacked==i64 reference; VDPBF16PS exact +
  tolerance-parity vs polyfill on floats; accumulate semantics). Gotcha-14
  contention parity test included as `#[ignore]` (fails on oversubscribed
  VMs BY DESIGN; run `--ignored` on dedicated silicon).

[MEASURED] onebrc probe GEMM leg with direct-VNNI staging: **3.6 → 21.3
Mrows/s (5.9×), 23.7 → 141.9 GMAC/s** (single thread — near the 169.7
GMAC/s int8 AMX anchor in AMX_GOTCHAS). 413/413 stations still EXACT;
8/8 lib tests + 2 doctests green; clippy/fmt clean.

[NOTE] Dispatch-behavior change signed off by operator: the row-major
entry `bf16_tile_gemm_16x16` now routes avx512bf16-without-AMX hosts
through VDPBF16PS instead of decode+FMA (bit-exact within the integer
boundary; BF16-precision-class accumulation-order differences on general
floats, same as any tier change).

[ADDED, same day] **LE byte contract on `PackedBf16B`** (operator "Go" —
first brick of the SoA-Morton batch-writer / write-hiding design):
`as_le_bytes()` (zero-cost reinterpret; LE by construction — the module
is x86_64-only) + `from_le_bytes()` (endian-correct anywhere, plain copy
on LE). This is the persistence/mailbox face per lance-graph's
SoaEnvelope discipline (envelope bytes LE from creation to tombstone).
Test `le_byte_view_roundtrips_and_is_truly_le` asserts byte 2i = low
byte of lane i AND that a GEMM over the roundtripped buffer stays
bit-exact. 9/9 lib tests green. Next bricks (lance-graph side): batch
writer flushing tile buffers as envelope tenants; write-hiding = stage
morsel N+1's VNNI writes while morsel N's tiles compute.

## 2026-07-02 — 1BRC-on-substrate probe (`examples/onebrc_cascade_probe.rs`)

1BRC workload (min/mean/max per station) restated on the substrate, as a
sibling of `morton_cascade_probe`. Branch `claude/1brc-lance-graph-xfx5tu`.
Three paths certified bit-for-bit against a scalar integer reference
(413 stations, integer tenths → exact in f32/f64 by construction):

- **Morton scatter**: stations minted as cells on a 64×64 Morton grid
  (4×4 tile = one F32x16), morsel-batched (64K rows) scatter into
  L1-resident SoA accumulators, (min,max,Σ,n) monoid fold.
- **AMX BF16 tile-GEMM group-by**: (Σ,n) as `C += A[16×K]·B[K×16]` via
  the NEW `ndarray::simd::bf16_tile_gemm_16x16_amx` re-export (W1a: the
  AMX-dispatching hpc wrapper surfaced through the canonical polyfill,
  same pattern as `matmul_i8_to_i32`; the `_amx` suffix disambiguates
  from the pure-FMA `simd::bf16_tile_gemm_16x16`) — B = per-row one-hot
  station indicator (26 column-blocks of 16), A rows = {1, hi(t), lo(t),
  bf16-RNE(t)} with the exactness split `hi=(t/256)·256, lo=t−hi` (both
  bf16-exact; f32 tile accumulation exact for K=4096). Clear-by-undo
  keeps B staging O(rows). AMX **actually ran** (amx_available()==true
  printed per Gotcha 9 discipline; EMR-class Xeon, kernel 6.18.5).
- **Aggregate pyramid** over the tile grid: hierarchical (min,mean,max)
  per tile/region/root in the same pass + band-prune queries
  (Belichtungsmesser on the MIN channel).

[MEASURED] 10M rows, 4-core Xeon EMR VM, single thread:
reference 453 Mrows/s | morton scatter 443 Mrows/s (**substrate tax ≈ 2%**)
| tile-GEMM 3.6 Mrows/s = 23.7 GMAC/s (dense one-hot indicator = the
honest price of group-by-as-matmul; per-call `vnni_pack_bf16` alloc in
`bf16_tile_gemm_16x16` is a visible overhead) | pyramid fold 0.02 ms |
band query prune 90.2%. All 413 stations EXACT on both paths; PASS.
Also EXACT at 100M rows (idle). **"Is BF16 precise enough?" — measured:**
the naive bf16-RNE row through the same tile gives max per-station
|Δmean| = 0.0123 tenths (0.0012 °C, N≈24k/station — quantization bias
averages out); single readings off by ≤ 2 tenths (half-ulp of bf16 at
|t|∈[512,1024)). Verdict: bf16-direct fine for means, hi/lo split (free —
spare A rows) required for min/max + exactness certification.

[FINDING → **Gotcha 14**, `.claude/AMX_GOTCHAS.md`] On this oversubscribed
VM, **AMX tile state silently corrupts under host CPU contention**: idle
= 413/413 exact at 100M rows; with 4 busy-loop competitors = 89-152/413
(whole rows lost, no fault); guest-side core pinning does NOT mitigate
(124/413); AVX-512 scatter path in the same run stays exact → isolated
to TMM state; suspected host-vCPU-switch XTILEDATA loss. Consequences
written into the gotcha: never certify AMX numerics on shared VMs; parity
tests must also run under deliberate load (Gotcha 9 extension); short
tile residency = harm reduction only.

[CROSS-REPO] Algebraic certification (partition/regroup invariance of the
monoid fold, bf16 hi/lo decomposition exactness) lands as a diagnostic
probe in `lance-graph/crates/jc` (`onebrc_agg`) — kernels here, proof
there, per the architecture rule (ndarray = hardware, jc = proof).

[LOOSE END] AMX has no min/max tile op → min/max stay on the scatter
path by construction. `bf16_tile_gemm_16x16` allocates + VNNI-packs B on
every call — a pre-packed-B variant would lift the GEMM leg
substantially; file under W1-adjacent if the group-by-as-GEMM shape
recurs. Text-ingest leg (SWAR/SIMD parse of the 13 GB file) deliberately
NOT probed here — separate probe if pursued (would exercise
`byte_scan.rs`).

## 2026-06-28 — WASM SIMD128 backend filled in (`src/simd_wasm.rs`)

Replaced the commented-out scaffolding in `src/simd_wasm.rs` with a real
`core::arch::wasm32` SIMD128 backend, mirroring `simd_neon::aarch64_simd`'s
proven split (native v128 for the float/byte hot path, scalar fallback for
the long tail). Branch `claude/ndarray-wasm-scalar-zr9n46`.

**`src/simd_wasm.rs::wasm32_simd`** (gated `#[cfg(all(target_arch="wasm32",
target_feature="simd128"))]`):
- `F32x16` / `F64x8` as `[v128;4]` + `F32Mask16` / `F64Mask8` — full API
  parity with the scalar macro (splat/from_slice/from_array/to_array/
  copy_to_slice/reduce_{sum,min,max}/abs/sqrt/round/floor/mul_add/
  simd_{min,max,clamp,lt,le,gt,ge,eq,ne}/to_bits/from_bits/cast_i32 +
  Add/Sub/Mul/Div/*Assign/Neg/Debug/PartialEq/Default + Mask::select).
- `I8x16` (one `v128`) = UNION of the scalar + NEON method sets
  (add/sub/min/max/cmp_gt + from_i4_packed_u64/lane_i8/saturating_abs)
  so consumers are portable across every backend.
- Free hot-kernels (v128 counterparts to the NEON kernels):
  `dot_f32x4_wasm`, `popcount_u8x16_wasm`, `hamming_u8x16_wasm`,
  `hamming_u8x64_wasm` (Fingerprint<256> distance via `i8x16_popcnt`),
  `base17_l1_wasm`, `codebook_gather_f32x4_wasm`, `bf16_to_f32_batch_wasm`.
- `mul_add`: `f32x4_relaxed_madd` under `+relaxed-simd`, else mul+add
  (base simd128 has no FMA). `round()` = `f32x4_nearest` (ties-even, =NEON).
  NaN in simd_min/max follows IEEE (NaN-propagating, =NEON); the existing
  `simd_exp_f32` NaN save/restore already absorbs this. All documented.

**Dispatch (`src/simd.rs`):** new `target_arch="wasm32" + target_feature=
"simd128"` arm re-exports the 8 native names from `wasm32_simd` and the
remainder from `scalar`; the "Other non-x86" arm now excludes that case
(wasm-without-simd128 + riscv etc. stay full-scalar). Added wasm32
`PREFERRED_*_LANES` arms (F32=4/F64=2/U64=2/I16=8, 128-bit widths) and a
`.cargo/config-wasm.toml` (`-Ctarget-feature=+simd128`).

**Unblocked the wasm build (pre-existing x86 leaks, not SIMD-scaffolding):**
the crate did NOT compile for wasm at all — `src/simd.rs` re-exported the
x86-only `amx_matmul` / `simd_amx` modules unconditionally, and
`backend::gemm_bf16` called `amx_matmul::matmul_bf16_to_f32` directly.
Gated both re-exports to `#[cfg(target_arch="x86_64")]`; split `gemm_bf16`
into the IDENTICAL x86 AMX path + a non-x86 branch routing through the
portable `hpc::quantized::bf16_gemm_f32(.., 1.0, 0.0)` (the same scalar
reference the AMX dispatcher itself falls back to → bit-equivalent). x86
behavior is untouched by construction (the original block now lives under
`cfg(target_arch="x86_64")`).

[VERIFICATION] (1) `cargo build -p ndarray --lib` for wasm32 **+simd128**
(native) AND **without** simd128 (scalar) AND **--no-default-features**
(no_std) AND x86_64 default — all green. (2) A standalone faithful copy of
`wasm32_simd` built to wasm32+simd128 and run under **node**: 51 numeric
checks (incl. exact mask bit-patterns, saturating_abs(i8::MIN)=127,
Hamming=512, Base17 vs scalar incl. a pathological |a-b|=60000 overflow
case, bf16 shift) all PASS. (3) x86 regression: 217 SIMD tests + 85
backend/bf16 tests pass; `clippy -p ndarray --lib -- -D warnings` clean;
`fmt --check` clean. Harness: `/tmp/.../scratchpad/wasmverify`.

[ADVERSARIAL REVIEW] Ran a 3-angle Opus review (cfg-gating / intrinsic-
semantics / x86-regression). x86-regression = PASS (x86 path byte-identical;
non-x86 bf16 fallback bit-equivalent). Two findings resolved: (P0 cfg-gating
"no_std arm break") = **false positive** — `pub mod simd` is itself
`#[cfg(feature="std")]` (lib.rs:239), so the native wasm arm is transitively
std-gated; `--no-default-features` wasm build is clean (empirically
confirmed). (P1 base17 i16 wrap) = **real, fixed** — `base17_l1_wasm` now
sign-extends i16→i32 via `i32x4_extend_{low,high}_i16x8` BEFORE the subtract,
so `|a-b|` is computed in i32 and matches the scalar reference for the full
i16 range (the prior i16-domain abs-diff, like NEON's `vabdq_s16`, wrapped at
|a-b|>i16::MAX). Doc nits (mul_add ULP wording, reduce_sum order, Tier-enum
comment) also tightened.

[NOTE] The stale top-of-CLAUDE.md "Build currently fails (exit 101)" no
longer reproduces — x86 lib builds clean this turn.

[LOOSE END] Full-crate (workspace) wasm build still blocked by `getrandom
0.3` (via `ndarray-rand`/`numeric-tests`, members that depend ON ndarray)
needing the `wasm_js` backend — orthogonal to this work; `-p ndarray --lib`
is the correct wasm surface and it is green. `bf16_to_f32_batch_wasm` is
provided + tested but NOT wired into the `bf16_to_f32_batch` dispatch (left
scalar to keep the BF16 path untouched); wire it if a wasm BF16 hot path
appears. Native U8x64/I32x16/U64x8 stay scalar on wasm (same as NEON keeps
them scalar) — the free Hamming/Base17 kernels cover those hot paths.

## 2026-06-17 — DECISION: HHTL fork ladder coded in `hpc::entropy_ladder` (CONJECTURE)

Reified the operator's standing idea — *if the orthogonal (helix/CAM-PQ)
leaf residue is strong enough, free energy forks into another domain
(HHTL shift = new exploration)* — as pure functions beside the existing
entropy/quadrant code. Unifies four vocabularies as one 2-axis structure:
`entropy_ladder::Quadrant(entropy,energy)` ≡ `lance-graph-contract::mul::
FlowState(challenge,skill)` (Csikszentmihalyi) ≡ Friston model-vs-surprise
≡ Staunen↔Wisdom.

- `residue_surprise(mag, noise_floor, sigma_k) → [0,1]` — orthogonal residue
  magnitude (prediction error the in-domain centroid codebook fails to
  explain) → challenge axis. Below floor = quantization (≈0); linear ramp
  over `sigma_k·noise_floor`. Threshold provenance per `I-NOISE-FLOOR-JIRAK`
  (Berry-Esseen wrong under CAM-PQ weak dependence); ramp is an honest proxy
  pending Jirak calibration, **not** a claimed bound.
- `ForkAction {Commit, DescendDeeper, ForkBasin, ForkDomain}` + `fork_decision`
  — bands challenge−skill on the shipped `mul::flow_state_from` boundaries
  (Anxiety δ>0.2, Boredom δ<-0.2; the matched middle |δ|≤0.2, which
  flow_state_from splits into Flow/Transition, collapses to one in-domain
  branch here); HHTL depth decides descend-vs-
  fork. `ForkDomain` (mint a new classid = the Friston model-switch) requires
  BOTH leaf depth AND challenge≫skill — the "strong enough AT THE LEAF" invariant.

Layering kept honest: the `FlowState` *assessment* stays in lance-graph
(thinking); the fork *math* lives here in ndarray (substrate, where residue +
energy physically are) — per the Architecture Rule. Pure fns + one enum, no
struct/layer, composes with `Quadrant`. 5 lib + 2 doctests, clippy clean.
Branch `claude/jirak-math-theorems-harvest-rfii13`.

**Loose ends (CONJECTURE → gated):** (a) feed the *real* `edge_codec::
CoarseResidue` magnitude from the live codec into `fork_decision` (currently
caller-supplied); (b) `ForkDomain` vs `ForkBasin` should be arbitrated by
residue *orthogonality* (⊥ all in-domain centroids = genuinely new), not the
depth+delta proxy; (c) Jirak-derived σ threshold to replace the `sigma_k`
proxy. This driver-side wire merges with lance-graph `materialize`'s
`ThoughtCtx::from_live` step (same call-site).


## 2026-06-10 — DECISION: GUID prefix→shape routing crystallized (docs-only)

The operator-pinned canonical GUID (`OGAR/CLAUDE.md`: hex dash-groups =
`classid(8)-HEEL(4)-HIP(4)-TWIG(4)-[basin·leaf+id]`; 3×4 tiers, `>> 2`)
now has its ndarray-side contract at
`.claude/knowledge/guid-prefix-shape-routing.md`: ndarray = MECHANISM
(layout-only `PrefixShapeTable`, opaque `ShapeId(u16)`, longest-prefix,
L1/L2-resident, no distance API — no-umbrella honored), consumer =
POLICY (lance-graph registers the table). GridLake continuation: key
selects grid family + pyramid level; value stays one byte-store
(column-substrate identity). φ-quorum anti-eigenvalue-theater contract
pinned with the PP-13 casebook as failure catalog; probes named
(ROUTE-1, QUORUM-1, PHI-1, PYR-1, CODEBOOK-44; HILBERT-L4 = existing
P0-4 blocker for any L4 cascade claim). CONJECTURE until coded — no
.rs touched in this commit.

## Evidence model (binding — from PR #200)
- **L0** = source · passing tests · ratified standards (ground truth).
- **L1** = `.claude/PR-X12-docs-audit.md` (#200) + `.claude/knowledge/plans-alignment-triage.md` — claims-about-source; **spot-check, never inherit**.
- **L2** = `.claude/plans/*` + `pr-x12-*` perspective docs = **inspiration, NOT evidence**.
- **Whole-file reads only** — no `grep`/`sed`/`head`/`tail` (`ls` to locate).
- Build/bench locally at `target-cpu=x86-64-v4`; committed `.cargo/config.toml` stays **v3** (GitHub/CI).

## Settled architecture (grounded this epoch, whole-read)
- **Cognitive similarity/cosine = Palette256 + Fisher-z**, integer: `hpc::cam_pq`
  squared-L2 ADC (u8 codebook indices) gated by θ = `distance::similarity_z` (atanh).
  Validated 10k×10k @ θ≈cos-0.90. **No float MAC in the distance kernel.**
- The cognitive **"splat"** = `lance-graph-contract::splat::CamPlaneSplat` (q8) →
  `AwarenessPlane16K` (16 384-bit OR deposition). **Sibling of, not the same as,** the
  graphics `splat3d` EWA renderer (per `splat3d/mod.rs`).
- EWA float-Σ sandwich (`splat3d::spd3`, Pillar 6/7) = uncertainty propagation +
  certification, **not** similarity. Pillar suite (6–17) certifies the substrate;
  **Pflug-10 certifies the CAM-PQ palette**.
- Typed distance (`cognitive-distance-typing.md`): one named fn per metric, newtype
  outputs, **no `fn distance<T>` umbrella**, conversions explicit. `palette→fisher→
  cosine→hamming` roundtrip is the named anti-pattern.

## Outstanding (per triage + #200)
- **#4** pr-x12 doc-fixes + evidence-policy + archive fabrication-heavy plans (Geo/Gov).
- **#5** ASG-leaf canon spec (Gov) — prerequisite for #7.
- **#7** ASG-leaf impl (Kernel) — must **extend `CamPlaneSplat`**, not reinvent; trails #5.
- `cam-pq-production-wiring` (UNOWNED, lance-graph) — route `cam_pq` through `CamCodecContract`.
- `UNUSED_INVENTORY_1.95` A1–A9 dead-code (phantom `SimdTier::{Sse2,WasmSimd128}`, stale 1.64 imports).

## Consolidation-sprint debt (PR-X program; ground-truthed `ls src/hpc/` 2026-05-27)
> Shipped-state vs `pr-master-consolidation.md`. Landed: ✅ **PR-X10** `linalg/`,
> ✅ **PR-X11** `pillar/`, ✅ **PR-X13** `ogit_bridge/`, ✅ **PR-X3** `blocked_grid/`.
- **PR-X12 codec ⚠️ v1 NEAR-COMPLETE** — `ctu/mode/predict` + now **`rdo` (A6, λ-RDO,
  integer fixed-point λ_q8 — no float) + `ans` (A7, static-table rANS over the 4-symbol
  mode alphabet, bit-exact round-trip)**. Remaining: `transform` (A4, deferred to v2 per
  design Q2) + `stream` (A8, framing over `ans`). 81 lib + 20 doctests green, clippy clean.
- **PR-X4 splat4d ❌ OUTSTANDING** — no `src/hpc/splat4d/`. Unbuilt.
- **PR-X9 cognitive ❌ OUTSTANDING** — no `src/hpc/cognitive/`. Unbuilt; must **consume**
  `lance-graph-contract::splat::CamPlaneSplat` (q8), never redefine it (contract is sacred).

## Merged / closed this epoch
- ✅ #201 triage · #205 `3dgs-tiles` (cesium tileset) · #206 + #208 render-depth cert.
- ❌ #207 EWA-SYRK bench **closed** (wrong regime — category error).
- 🗑 `phi_spiral.rs` abandoned (float, wrong manifold). Net new usable code this
  session = 0 (see `board/EPIPHANIES.md` grounding-discipline entry).

---

# Polyglot Notebook — Single Binary Architecture

> Separate/older program — NOT the current epoch (see top of file).

## The Binary

One `cargo build`. Ships as one executable. Contains:

```
reactive runtime     (transcoded from marimo Python)
graph query engines  (transcoded from graph-notebook Python)
kernel protocol      (Rust-native ZMQ, from kernel-protocol spec)
document publisher   (transcoded from quarto TS/Deno)
local graph database (lance-graph, already Rust)
SIMD kernels         (ndarray, already Rust)
graph compiler       (rs-graph-llm, already Rust)
web frontend         (marimo's JS/React, served by the binary)
```

External process: R only (Bardioc/almato). Speaks Arrow IPC to the binary.

## Repos → Crates

| Repo (source) | Becomes | Work |
|------|---------|------|
| marimo | `crate::runtime` + `crate::server` | Transcode Python→Rust |
| graph-notebook | `crate::query::{cypher,gremlin,sparql,nars}` | Transcode Python→Rust |
| kernel-protocol | `crate::kernel` | Implement from spec in Rust |
| quarto | `crate::publish` | Transcode TS→Rust |
| quarto-r | external R process | Stays R, Arrow IPC bridge |
| lance-graph | `crate::graph` | Already Rust, integrate |
| ndarray | `crate::simd` + `crate::linalg` | Already Rust, integrate |
| rs-graph-llm | `crate::compiler` | Already Rust, fix build |

## Scopes (parallel, non-overlapping)

### SCOPE A: Reactive Runtime (marimo → Rust)
Transcode marimo's reactive cell execution model to Rust.
The core insight: cells have dependencies, when a cell's input changes,
downstream cells re-execute. That's a DAG scheduler — natural in Rust.

### SCOPE B: Query Engines (graph-notebook → Rust)
Transcode graph-notebook's Cypher/Gremlin/SPARQL executors to Rust.
Bolt protocol client, WebSocket client, HTTP client — all Rust-native.
Add local path: Cypher → lance-graph semiring (no network).

### SCOPE C: Kernel Protocol (kernel-protocol spec → Rust)
Implement Jupyter kernel wire protocol in Rust.
Only needed for R (IRkernel) — everything else runs in-process.
ZMQ via zeromq-rs. Connection file parsing. Message ser/de.

### SCOPE D: Publisher (quarto TS → Rust)
Transcode Quarto's document rendering pipeline to Rust.
Pandoc AST manipulation. Markdown → PDF/HTML.
Custom graph visualization extension.

### SCOPE E: Integration (lance-graph + ndarray + rs-graph-llm)
Wire the existing Rust crates into the binary.
Fix rs-graph-llm build. SIMD kernels for graph ops.
This is mostly Cargo.toml workspace wiring + API surface.

## Decisions
[DECISION] One binary, no Python runtime
[DECISION] marimo's JS frontend served by Rust HTTP server (axum/actix)
[DECISION] R is the ONLY external process (Arrow IPC bridge)
[DECISION] Cypher executes locally via lance-graph semiring by default
[DECISION] Remote DB connections (Neo4j, FalkorDB) via native Bolt client
[DECISION] vis.js graph rendering served as static assets by the binary

## Architecture Decisions

### 2026-06-13 — GEMM-dispatch routing fixes (savant-architect)
Branch `claude/wonderful-hawking-lodtql`. Three public GEMM entry points
were not routing to the accelerated kernels.

- **`backend::gemm_bf16` (src/backend/mod.rs)** — ALREADY FIXED in the
  working tree this session. Now routes to
  `hpc::amx_matmul::matmul_bf16_to_f32` (AMX `TDPBF16PS` → AVX-512
  `VDPBF16PS` → scalar). Slice→ArrayView2 wrapping mirrors the call shape
  in `simd_runtime::matmul`; inputs sliced to exact `m*k`/`k*n`/`m*n`.
  Bit-equivalent on non-AMX/non-AVX512BF16 hosts because the dispatcher's
  scalar fallback is the same `quantized::bf16_gemm_f32(a,b,c,m,n,k,1.0,0.0)`
  the old direct call used (alpha=1, beta=0 preserved).
- **`backend::gemm_i8` (src/backend/mod.rs)** — ALREADY FIXED in the
  working tree this session. Routes to `simd_int_ops::gemm_u8_i8`
  (4-tier: AMX `TDPBUSD` → VNNI-zmm → AVX-VNNI-ymm → scalar).
  [DECISION] Deliberately NOT routed to `amx_matmul::matmul_i8_to_i32` as
  the literal task text asked: `gemm_i8` is **u8×i8→i32**, but
  `matmul_i8_to_i32` is **i8×i8→i32** and would reinterpret A-bytes ≥128
  as negative — NOT bit-equivalent. `gemm_u8_i8`'s scalar fallback is the
  same `quantized::int8_gemm_i32` the old `vnni_gemm::int8_gemm_vnni`
  used → bit-identical on scalar hosts; VNNI-zmm arm calls the same
  `int8_gemm_vnni_avx512` kernel as before. All tiers integer-exact.
- **`native::gemv_f32` / `gemv_f64` (src/backend/native.rs)** — FIXED
  THIS TURN (was calling `scalar::gemv_*` unconditionally). Now matches
  on `tier()`: Scalar tier → unchanged `scalar::gemv_*` (byte-identical);
  Avx2/Avx512 tiers → per-row `dot_f32`/`dot_f64` (the existing
  dispatched, parity-tested SIMD dot). GEMV = stack of row dots; each A
  row is row-major-contiguous so contiguous `dot_*` loads apply. Leading
  `n` of each `lda`-wide row taken via `&a[i*lda..i*lda+n]`; no new bounds
  requirement vs scalar ref. SIMD tiers carry the module's documented
  1-2 ULP reduce-order drift (within BLAS tol; `test_gemv_f32` uses 1e-5,
  no byte-exact consumer asserts gemv).

[UNSAFE-AUDIT] gemv fix added **zero** new `unsafe` — it reuses the
already-audited `dot_*` kernels. No new sentinel-qa surface from this turn.
The two mod.rs fixes contain `unsafe` repr(transparent) slice reinterprets
(BF16/u16) that were landed earlier this session and warrant the standard
sentinel-qa pass if not already covered.

[LOOSE END] Repo references modules that exist on disk but the Glob/Grep
index was transiently stale this session (returned empty for
`simd_int_ops.rs`, `vnni_gemm.rs`, `bf16_gemm_f32`); Bash ground-truth
confirmed all present. Orchestrator should `cargo fmt`/`clippy`/`test`
centrally (edits were edit-only, no compile performed here).
