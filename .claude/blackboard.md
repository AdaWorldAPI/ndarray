# Current epoch (2026-05-26) — splat / palette / pillar / 3DGS

> **Read this first.** The "Polyglot Notebook" architecture below is a
> separate/older program, not the current epoch.

## 2026-07-06 (latest) — F64 GEMM completed: FMA tier + register residency + native-engine swap

Operator: "complete the F64 gemm… and pr". Three moves, all on the entry
below's foundation:

1. **`gemm_f64_tiled_fma`** — fast fused tier via const-generic
   `gemm_f64_tiled_impl<const FMA: bool>` (monomorphized, no runtime
   branch). Per-element ascending-p order preserved; fused step
   `c = fma(α·a, b, c)`. Bit-identical to the reference tier on
   integer-valued operands (products+sums < 2^53 — asserted with
   `assert_eq!` on the full shape sweep); last-ulp-per-step differences
   on general floats (tolerance test scaled k·ε to SUMMAND magnitude —
   the initial result-scaled tolerance was wrong for cancellation-heavy
   elements and failed honestly). Cross-backend caveat documented (WASM
   without relaxed-simd has unfused vector lanes + fused scalar tail).
2. **Register-resident C row-block** (both tiers): C block loaded into
   `[F64x8; TILE/LANES]` accumulators + scalar tail array ONCE per
   (kk,ii,jj,i), whole kb-loop accumulates in registers, one store.
   f64→f64 store/reload never rounds ⇒ per-element op sequence
   unchanged ⇒ every bit-equality test green untouched. [MEASURED]
   ref 4.6→10.0 GF, fma 4.7→10.7 GF (2.2×).
3. **Engine swap:** `backend::native::gemm_f64` now routes to the
   crate-native tiled kernel — the f64 GEMM behind
   `BlasFloat::backend_gemm` / `hpc::blas_level3::blas_gemm` / batched
   linalg is entirely own Rust; matrixmultiply remains only in gemm_f32
   and upstream `Array::dot` (impl_linalg.rs, untouched at ~33 GF).
   **[REVISED post-verify] Engine = the UNFUSED reference tier**, not
   fma: the verify pass surfaced a cliff — the AVX2-polyfill/scalar
   `mul_add` lowers to a libm `fma()` call on baseline x86-64 builds
   (consumers do NOT inherit this repo's `.cargo` target-cpu pin; CI
   lands exactly there). The unfused tier has no libm dependence on any
   backend, costs only ~7% vs fused on pinned builds (10.0 vs 10.7 GF),
   and makes the backend engine bit-identical to the certification
   reference. `gemm_f64_tiled_fma` stays public for FMA-pinned
   consumers. New panic contract documented on `gemm_f64` (# Panics —
   checked preconditions vs the old wrapper's silent-UB on short
   slices; matches CBLAS xerbla behavior).

[MEASURED, 3-engine, this VM (v3 compile → AVX2 arm, PREFERRED_F64_LANES=4;
host runtime has avx512f but committed .cargo config is v3)]:
256³/512³/1024³ — ref 11.2/10.0/9.6 GF | fma 11.9/10.7/10.3 GF |
matrixmultiply 34.1/32.3/33.7 GF | max|fma−mm| ≤ 1.4e-13.
**Own-engine gap: ~3.1×** (was 6.6× pre-restructure). Trade accepted per
operator priority (own reverse-engineered Rust in the path, auditable
numerics); blast radius = hpc BLAS surface only. 2185/2185 lib tests
green WITH the swap.

[LOOSE END → next rung] Closing the 3× needs a real microkernel: B-panel
register reuse (i-tiling IR=2..4 × narrower j-block), then A/B packing —
the matrixmultiply Goto recipe, own-Rust edition. Also: `gemm_f32_tiled`
still dead in native.rs `mod scalar` (f32 sibling completion);
avx512f compile arm untested on CI (v3 config) — the F64x8=__m512d arm
runs only on local v4 builds.

**[VERIFY OUTCOME]** 3-angle adversarial pass on the completion diff:
numerics PASS / swap-trace PASS / docs FAIL→fixed. Substantive P1 acted
on: baseline-x86 libm-fma cliff → engine revised to the unfused
reference tier (see #3 REVISED above). Doc P1s fixed: two stale
"gemm_f64 delegates to matrixmultiply" claims (simd.rs comment +
gemm_f64_tiled rustdoc) contradicted the swap in the same diff. P2s
fixed: fma determinism scoped to per-(build,runtime) (wasm relaxed-simd
fusion is implementation-defined); AVX2 vfmadd naming corrected (per-
lane f64::mul_add polyfill, fused semantics); integer-corpus bound
comment corrected (k_max=128, ≈2.1e4). All gates re-run green after
fixes: clippy -D warnings, 2185/2185 lib tests, 4 gemm doctests.

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

---

## 2026-07-11 — U32x16 ARX lane + ChaCha20 matryoshka (see .claude/CHACHA20_MATRYOSHKA_PLAN.md)

**DONE:** `ndarray::simd::U32x16` full ARX triple (Add/BitXor/**rotate_left**) on
every tier — avx512 native `_mm512_rolv_epi32`, native wasm `[U32x4;4]`, avx2/
scalar/nightly. Node-run wasm parity CI gate (`wasm_simd` job + `scripts/wasm-parity.sh`
+ `crates/wasm-simd-parity/`, workspace-excluded, tests the real types, no drift).
Interim `src/simd_crypto.rs` (chacha20 scalar+avx512+wasm128, RustCrypto-parity-proven)
is superseded by the matryoshka once the fork lands.

**TO-DO (deferred, token limit):**
1. Native neon `U32x16 = [U32x4;4]` (extend `U32x4(uint32x4_t)` w/ xor+rotl, compose,
   fix F32x16 to_bits, swap simd.rs aarch64 arm).
2. aarch64 cross (qemu) parity CI job — generalize `wasm-simd-parity` to a shared
   `simd-parity` harness; closes NEON's x86-suite blind spot.
3. avx2 native `U32x16` (2×__m256i, TD-SIMD-3) — optional.
4. **Matryoshka execution:** fork `chacha20`, clone `avx2.rs` backend → rewire over
   `ndarray::simd::U32x16`, `[patch]` into the encryption stack (transitive accel),
   gate vs RustCrypto `soft`, retire `simd_crypto.rs`. Full plan in the doc above.

---

## 2026-07-12 — PR #240 CI fully green (no_std/MSRV fix chain closed)

Tip `5a914c37`. All 20 checks pass; the three previously-red failures are resolved:

- **261f736a** — `simd_crypto` dispatcher: `is_x86_feature_detected!` → compile-time
  `#[cfg(all(target_arch="x86_64", target_feature="avx512f"))]` (no runtime detection;
  the workspace SIMD-dispatch rule). Unblocked `blas-msrv` + `nostd/thumbv6m`.
- **5a914c37** — three no_std/bare-`--no-default-features` test-build fixes:
  - `simd_crypto` tests: `vec![[0u8;64]; n]` → fixed arrays `[[0u8;64]; 40]` / `[[0u8;64]; 16]`
    (no `vec!` under no_std).
  - `tests/chacha20_rustcrypto_parity.rs`: added `#![cfg(feature = "std")]` (imports
    std-gated `ndarray::simd`; no-ops under `--no-default-features`).
  - `src/tri.rs`: `Array2::<i32>::zeros(...)` (serde_json dep-drift added
    `impl PartialEq<Value> for i32`, making bare `Array2::zeros` element-type ambiguous).

Green jobs of note: `tests/{stable,beta,1.95.0}`, `blas-msrv`, `nostd/thumbv6m-none-eabi`,
`clippy/1.95.0`, `format/stable`, `native-backend/stable`, `tier4-avx512-check`,
`wasm-simd/parity-node` (new gate), `hpc-stream-parallel/rayon`, CodeRabbit.

Deferred (task #30, token-limit call): native neon `U32x16=[U32x4;4]` + aarch64 cross
parity CI + the matryoshka chacha20 fork. Plan in `.claude/CHACHA20_MATRYOSHKA_PLAN.md`.

---

## 2026-07-12 — Matryoshka finalized + NEON cross-CI (deferred #30 CLOSED)

- **Native NEON `U32x16 = [U32x4;4]` ARX lane** (commit 06a61bf9): bitxor/
  rotate_left on U32x4, composed U32x16 Add/BitXor/rotate_left; also fixed the
  pre-existing aarch64 stable-compile breakage (`u16x8` alias; nightly-only
  `vdotq_s32` → stable widening NEON). aarch64 now `cargo check`-clean on stable.
- **ChaCha20 matryoshka + `simd_crypto.rs` RETIRED** (commit 20dc6c3f):
  `vendor/chacha20/` fork (name/version kept, own [workspace]); the ONE delta is
  `backends/ndarray_simd.rs` — the transpose block16 over `ndarray::simd::U32x16`
  (pure +/^/rotate_left, no intrinsics, no unsafe), compile-time-selected under
  cfg(x86_64+avx512f), `[patch.crates-io]`-folded under encryption. Triple parity
  gate GREEN (fork RFC 8439 vectors through ndarray_simd @ v4; encryption 23 AEAD
  tests @ v3+v4). Deleted src/simd_crypto.rs + the parity test + the chacha20 dev-dep
  + the ndarray::simd::chacha20_* surface.
- **NEON cross parity CI**: `crates/neon-simd-parity` (excluded bin) +
  `scripts/neon-parity.sh` (cross-build aarch64 + run under qemu-aarch64-static) +
  CI `neon_simd` job (added to conclusion needs). Runtime-verifies U32x16 ARX /
  F32x16 / I8x16 == scalar on real aarch64. Green locally under qemu.

Follow-ups (documented in `.claude/CHACHA20_MATRYOSHKA_PLAN.md`): wasm matryoshka
backend (simd128 branch); cross-repo `[patch]` for MedCare-rs; the workspace
default is x86-64-v3 (avx2) so ndarray_simd activates on avx512 builds only.

---

## 2026-07-16 — PR-X12 x265/x266 plan review: audit applied + H.267 standards grounding

- **PR-X12-docs-audit corrections finally APPLIED** (they had sat unapplied
  since 2026-05-22): fabricated symbols marked ([PLANNED] `batched_ssd_search`;
  `blasgraph::tropical_gemm` / `bgz17::tropical_spmv` removed — real min-plus is
  method `ScalarCsr::spmv_min_plus`, lossy sibling); blasgraph restored as
  bit-exact canon over bgz17; per-arch DCT crossovers tagged [UNCALIBRATED];
  false `signature_kernel_pde` Goursat-bug claim withdrawn (its convergence
  tests pass); the R-11 unit is leaves-at-8×8 not CTUs, and the count itself
  was corrected to 129,600 exact (padded 130,560; the old 132,710 was
  ungrounded) → ~129 ns/leaf budget; §9 falsifiability matrix tagged
  FORWARD-CONDITIONAL. Tier-1 docs (`woa-multiarch-orchestration`,
  `bgz-jc-substrate-synergies`) ⛔ QUARANTINED pending rewrite.
- **NEW: `.claude/knowledge/pr-x12-h266-h267-standards-landscape.md`** —
  sourced public-standards anchor: H.266/VVC (2020, ~40-50% over HEVC, dec
  1.5-2×/enc ~10×), ECM-16.1 (~27% over VTM, complexity flagged impractical),
  NNVC v7 (NN in-loop ≈9% RA each — the antithesis of our anti-neural rule),
  H.267 (CfP Jul 2026 → submissions Nov 2026 → evaluation Jan 2027 → finalize
  ~2028; requirement ≥40% over VVC Main 10 at 4K+). "x266" in our docs =
  PR-X12 3DGS scene codec, never H.266.
- x266 lens doc got §12 reality-check addendum + F-3b falsifier
  (conventional-plus-neural acceptance risk); capstone + 3DGS plan index got
  standards-watch sections. Watch dates: Nov 2026, Jan 2027.
- Status note: `src/hpc/codec/` now has `ans.rs` + `rdo.rs` (A7/A6 debts
  D-CODEC-2/-3 have code); still no `ndarray-codec` crate (Plan H open).

---

## 2026-07-16 (2) — H.268 codename + graded Morton/wgpu synergy matrix

- **Codename ruling:** the "x266" placeholder (PR-X12 3DGS scene codec) is
  internally codenamed **H.268** — INTERNAL ONLY, never an ITU designation
  (H.267 itself is still prospective). Registered in the x266 lens header,
  landscape doc, capstone, 3DGS plan index.
- **NEW: `.claude/knowledge/pr-x12-h268-morton-wgpu-synergies.md`** — the
  "industry-impractical vs realistically-achievable" matrix, every claim
  adversarially verified (workflow wf_6c6fb99a-cb4, 15 agents, file:line
  receipts): 1× FEASIBLE-NOW (the scoping row), 2× NEEDS-PROBE, 7×
  OVERCLAIM-CORRECTED. Load-bearing findings: wasm SIMD128 lane IS real +
  CI-parity-verified (simd_wasm.rs; wasm-simd/parity-node); bgz17 256×256
  tables are texture-isomorphic (dense u16, R16Uint-ready) but zero GPU-LUT
  code exists; ctu.rs is an ARENA tree (no Morton in codec dir) — flat
  Morton SoA is an unimplemented refactor; D-PHASE/D-WHP are [H] with unrun
  probes (J2 kill: dither-only); a2ui-paint wgpu = untested quad demo,
  `webgl` feature unwired; ndarray deliberately "no GPU, no wgpu".
- **Probe queue established:** PROBE-GPU-LUT, PROBE-MORTON-CTU,
  PROBE-RANS-INTERLEAVE (new names), + OGAR PHASE-1/PERT-RHO/PYR-1,
  WHP-1..4, Plan E bits/Gaussian, a2ui N2 — each with pass/kill conditions.

---

## 2026-07-16 (3) — H.268 addendum: comma closure + 96-bit carving + kernel-shape rule + replayable-tile synergies

- **`pr-x12-h268-morton-wgpu-synergies.md` extended §7-§10** (old §6
  Cross-references renumbered to §11), per
  `.claude/plans/H268-comma-96bit-replayable-addendum-v1.md`.
- **§7 comma closure:** Pythagorean-comma/X-Trans anti-moiré framing;
  `CurveRuler` stride-4-over-17 as the coprime-integer surrogate.
  D-QUANTGATE rationale restated to its three real legs (libm
  non-portability, WGSL floats not IEEE-pinned, bijective closure) —
  the "floats round differently" leg is explicitly withdrawn, with
  receipts (`std::f64::consts::{GOLDEN_RATIO,EULER_GAMMA}` compile
  bit-exact on 1.94/1.95; no `std::simd::const::*`; `gemm_f64_tiled`
  five-backend bit-identical). φ-PLACES/walk-QUANTIZES/γ-CORRECTS
  division of labor stated as a rule.
- **§8 96-bit facet carving:** CAM-PQ 48b + helix `ResidueEdge` 24b +
  turbovec 24b = 96 bit = the V3 12-byte content-blind payload identity;
  `Signed360` (48b) is the out-of-row alternate carving. Three flavours
  of 256 (post-review correction + operator refinement): CAM-PQ =
  6×256² compressed to per-query 6×256 f32 ADC rows (6KB,
  `cam_pq.rs:76-84`); bgz17 = the explicit materialized 256² u16 (+ k×k
  u8 compose; 388KB benchmark = 3 S/P/O planes × 128KB); V3 facet =
  explicit 6×256² as codec-agnostic ADDRESS (6×(u8:u8) rails = 96 bit;
  classid→ClassView switches which codec's 256² family each rail
  indexes).
- **§9 kernel-shape rule:** VNNI/AMX for matmul-shaped ops, LUT/texture
  for lookup-shaped ops — turbovec NativeLut measured **11.4×** faster
  than the VPDPBUSD GEMM polyfill (n=20k/dim=512/4-bit, FINDING). ITU
  claim scoped to compute kernels only (not CABAC/conformance/ECM count).
- **§10 replayable-tile synergies:** 4×4 Morton tile as the shared
  object between H.268 (phase-side seekability — entropy-level seek
  still A8-gated; seekable grain; C6-scoped native tiling) and cognitive
  shaders (RNG-free exploration, replayable thinking on the CPU/wasm
  integer path, anti-confabulation [H, needs correlation-spectrum
  probe], cache-native 192B working set) — all nine consequences stay
  **probe-gated** (D-MTS-1..3, PHASE-1/PERT-RHO/PYR-1, WHP-1..4, L4
  doc-lock); no kill condition weakened.
- **§8 fourth-mode follow-up (post-#243):** the Hambly–Lyons anchor is
  in-workspace, not external — THIS repo `src/hpc/pillar/signature.rs`
  (Pillar-11 B7: sig transform + sig-kernel, Gram PSD, 1000-Lévy-path
  certification) + lance-graph jc Pillar 11 (`hambly_lyons.rs`, feature
  `hambly-lyons` → sigker). Only the ladder→signature MAPPING stays
  [S]; probe builds on the Pillar-11 harnesses. §8 sentence amended.
- **§10 forward synthesis (operator):** two candidate adoptions for the
  tile pyramid, probe-gated — [H] one WH family for both pyramid sides
  (OGAR sign side is already WH-of-the-address-tree; bgz-tensor's
  hadamard_rotate = same family as magnitude-side preconditioner;
  PROBE-WH-MAG = WHT₁₆+i4/i2 vs direct on real tile magnitudes) and
  [S] signature-as-trajectory-checksum (tree-like equivalence = the
  digest's null space, the formal "which detours leave no comma";
  PROBE-SIG-CHECKSUM on the Pillar-11 harnesses). Neither adds stored
  tile fields.

---

## 2026-07-16 (4) — h268-probe-wave-v1 RESULTS (adjudicated)

- **PROBE-WH-MAG → NEUTRAL; bare-tile leg CLOSED NOT-TRANSFERRING.**
  B/A 0.929/1.317/1.869 — WHT₁₆ spreads outlier energy tile-wide,
  inflating the per-cell quantization floor; the row-level win needs the
  passthrough escape + centroid residual the probe deliberately omits.
  Shipped row codec untouched. PROBE-WH-MAG-2 named, deferred.
- **PROBE-SIG-CHECKSUM → PASS** with the depth-2 bound: parallel-chord
  interior displacement is EXACTLY signature-invisible — null space
  exceeds tree-like equivalence; mitigate via depth 3 or paired digest.
- **PROBE-WALK-SPECTRUM → KILL** of §10(g)'s "decorrelated by
  construction" (walk lattice |R| 0.875 vs PRNG 0.0205 = 42.7×;
  C(13)=−15 sidelobe, coprimality ≠ decorrelation); "known period-17
  structure" half CONFIRMED (R(17m)=1−8m/N). D-QUANTGATE unaffected.
- Doc updates: §10(g) corrected, §10 forward-synthesis RESULT lines,
  §5 results sub-table. Canonical verdicts: lance-graph
  E-H268-PROBE-WAVE-1-RESULTS + plan h268-probe-wave-v1.md Results.
  Probes: bgz-tensor probe_wh_mag / jc sig_checksum / helix
  walk_spectrum (all suites green).

## 2026-07-16 (5) — sprite amortization spec'd + two standing corrections

- **PROBE-SPRITE-REPLAY spec'd** (plan `x265-sprite-replay-probe-v1.md`,
  §5 row added): moving object = HHTL-anchored splat sprite + helix
  motion code, mapped onto the x265 I/P/B grammar (I = splat set at
  anchor; P = one helix code per sprite, replacing per-block MV search;
  B = parametric interpolation along the helical path). Scope guards:
  NOT H.268, NOT x265 bit-parity — GOP-grammar replay on our primitives;
  CPU/wasm carries the bit-exactness claims, wgpu is render-grade (C9).
  Amortizations: motion search → address arithmetic in the Morton
  cascade; the minimal wgpu harness doubles as PROBE-GPU-LUT's missing
  harness. KILL: helix object-motion collapsing back into a dense MV
  field.
- **§10(i) honesty amendment**: the 3-cache-line tile claim holds only
  under the analytic Fisher-z canon (materialized 256² u16 = 128KB =
  L2-resident); analytic drops table residency to 8B and makes rail
  reads |Δi8| arithmetic (four tiles/lane per AVX-512 register).
- **PROBE-WH-MAG-2 deferral weakened**: the Skip/Merge/Delta/Escape
  mode grammar already IS the per-tile escape tier; WH-MAG-2 = WH under
  the mode grammar, not a wait for new machinery.

## 2026-07-25 — encryption: the KDF cost fields are acted on before they are authenticated

**Status:** FINDING (reproducer in `envelope::tests::every_single_bit_flip_is_refused_and_none_of_them_are_expensive`)

Found by a downstream consumer building a password-sealed record POC on
`encryption::envelope`. An exhaustive single-bit-flip sweep over a sealed
blob did not fail — it **aborted the test process**. One flipped bit in the
`m_cost_kib` header field asks Argon2id for a 4 TiB allocation; the
allocation fails, and a failed allocation in Rust aborts rather than
unwinding.

The header IS authenticated (it is the AEAD's associated data), and that
was the reasoning behind not checking it. But verifying the tag needs the
key, and deriving the key means first running Argon2id **with the
parameters the blob just supplied**. So there is a window, before anything
is proven, where an attacker-chosen cost decides how much memory this
process reserves. Tamper detection works exactly as designed and the
process still dies before reaching it. *Authenticated-but-only-later is not
the same as trusted.*

`KdfParams::validate()` now gates m/t/p **before any allocation**, in
`derive_key` and in `decode_header`. The tests assert the refusal is
**cheap** — an expensive rejection is itself the attack.

**Codex P1 on the first cut, and it was right:** the initial ceiling
(1 GiB / 64 passes) was chosen as "below Argon2's 4 TiB roof", which is a
rounding error, not a limit — 1 GiB × 64 passes pre-authentication is
equally fatal on a browser tab or a small container, and a few concurrent
requests exhaust the host. Replaced by `CostLimits`, a caller-supplied
budget: `DEFAULT` = 128 MiB / 4 / 2 (twice the memory and one pass more
than the heaviest shipped preset, so a cost bump still opens old and new
blobs), `SHIPPED_PRESETS_ONLY` = exactly 64 MiB / 3 / 1 for services that
mint every blob they open. `open_within` / `derive_key_within` take the
budget explicitly.

Measured worst case the default admits: **414 ms, 128 MiB** (release, this
box; the `#[ignore]`d `worst_admitted_cost_is_within_the_documented_budget`
prints it). The bit-flip sweep dropped 13.5 s → 2.3 s once the tighter cap
started refusing the flips it used to honour — the sweep had itself been
running multi-hundred-MiB derivations.

**Not done — an allowlist of known profiles** (Codex's alternative) would
break cost bumps in the other direction: a reader shipped before the writer
would reject the new profile. A bounded budget keeps the forward
compatibility the header format exists for.
