# ndarray — Epiphanies (append-only)

## 2026-07-29 — BLAKE3 needs a method surface, not intrinsics (measured)
**Status:** FINDING
**Scope:** @simd-savant domain:codec
**Cross-ref:** PR #264, PR #265, `.claude/knowledge/blake3-on-ndarray-simd.md`,
`.claude/knowledge/simd-codegen-oracle/`

C is out of `blake3` (`features = ["pure"]`, #264 — 33 `.o` objects and
`libblake3_avx512_assembly.a` gone). What remained was a second SIMD surface:
2,910 lines of raw `core::arch` in its Rust backends.

Census + four probes settle what a `ndarray::simd` backend would cost.
`rust_avx2.rs` uses 15 distinct intrinsics; **twelve already exist on
`U32x16`**. The other three families (`unpack{lo,hi}_epi32`,
`unpack{lo,hi}_epi64`, `permute2x128` — 18 call sites) serve one purpose:
`hash_many`'s transpose.

| probe | packed | scalar on lane data |
|---|---|---|
| `blake3_g_u32x16` | 72 | 0 |
| `interleave_lo_u32x16` | 11 | 0 |
| `transpose_16x16_u32` (index loop) | **0** | 1 |
| `transpose_16x16_composed` (4 stages, correctness-checked) | 79 | 0 |

The compression core is free. The interleave *primitive* is free — a scalar
index loop compiles to `vpermd` + `vpblendd`, a real two-source cross-lane
permute. Only the **monolithic index-loop spelling** of the transpose fails;
composed from `exchange<G>` at G = 1/2/4/8 it emits `vpunpcklqdq`, `vpermq`,
`vpermpd`, `vinserti128` — the same family the intrinsic backend hand-writes.

**So no intrinsic override is earned.** The gap is one const-generic method,
~15 lines, no `unsafe` and no `core::arch`. That is the third consecutive
time this oracle has turned "obviously needs intrinsics" into "already free"
(after `saturating_abs`, `widening`, and `cross_lane_reverse`).

**The methodological catch is the part worth keeping.** The first version of
this finding rested on `transpose_stage_u32x16` — ONE 32-bit stage whose
helpers, by their own doc comment, do not compose into a transpose. Codex and
CodeRabbit independently flagged it. It was an inference wearing a
measurement's clothes, published in the document that cites TD-T22 as the
reason not to do exactly that. Two guards added in response:

- the composed transpose is **checked against a naive reference and aborts on
  mismatch**, control-tested by deleting `stage::<8>` to confirm it fires;
- `run.sh` now **executes** the probe binary, because `--emit asm` links
  nothing and the assertion had never actually run. A claim of verification
  that does not verify is the same defect one level up.

Still unmeasured, and stated as such: throughput vs `rust_avx2.rs`. "Emits
packed shuffles" is not "is faster."


## 2026-07-29 — Reading one config file is not reading the build
**Status:** FINDING
**Scope:** @simd-savant @truth-architect domain:build-tiers
**Cross-ref:** PR #265, `.claude/knowledge/chacha20-vendoring-blast-radius.md`,
`.claude/knowledge/td-t22-asm-investigation.md`

This repo has **three** build tiers, and an audit that reads only
`.cargo/config.toml` sees one of them and concludes the opposite of the truth.

| tier | target-cpu | purpose |
|---|---|---|
| CI (`ci.yaml`) | none globally; v4 in `tier4-avx512-check` | one binary, all ISAs, runtime `LazyLock<Tier>`; the `cross_test` matrix spans i686 and s390x, so a global pin is impossible |
| `Dockerfile` | `x86-64-v3` | the portable image |
| `Dockerfile.avx512` | `x86-64-v4` | the AVX-512 image |

Operator's formulation: *cargo is CI is github needs V3; dockerfile is V4.*

**Two conclusions I published were wrong because of this**, both in the same
week, both stated with measurement-backed confidence:

1. *"No default build of any repo runs `vendor/chacha20`'s `ndarray_simd`
   backend."* True of `cargo build`, false as a claim about what ships —
   `Dockerfile.avx512` is a v4 build of this workspace and compiles it. The
   backend is the deployed path on AVX-512 silicon, not dead code.
2. *"CI has been testing different machine code than anyone reviews"*, filed
   as a ⚠ defect. It is the design, and `ci.yaml:17-22` says so in prose I
   had not read: a global pin collides with the non-x86 cross_test matrix and
   contradicts the runtime-dispatch intent.

**The pattern to name:** each measurement was individually correct. The error
was in the *scope quantifier* — "no build", "CI" — attached to evidence drawn
from one file. `rustc --print cfg` told me what v3 lacks; it could not tell me
which tiers exist. A claim quantified over "every build" needs an enumeration
of builds, and `find . -iname 'Dockerfile*'` is that enumeration.

Consequence, concretely: **before any claim of the form "no build does X" or
"CI does Y", enumerate the tiers** — `.cargo/config*.toml`, `Dockerfile*`, and
the workflow `env:` blocks including per-job overrides. Three greps.

A second-order note worth keeping: ndarray's own SIMD upgrades itself at run
time via `LazyLock<Tier>` even in a v3 build, but `vendor/chacha20`'s gate is
`#[cfg(target_feature = "avx512f")]` — compile-time, no runtime fallback. So a
v3 image on AVX-512 silicon has ndarray's kernels upgrading while the chacha20
keystream stays on RustCrypto's backends. Compile-time and runtime dispatch
living in one binary is not a contradiction, but it means "this build is v3"
does not settle what any given subsystem selects.


## 2026-04-19 — Prompt↔PR ledger is 10⁷× cheaper than code grep
**Status:** FINDING
**Scope:** @workspace-primer domain:bookkeeping

To answer "what did we ship for topic X":

- **Grep across code:** ~100 MB of Rust across N crates, ~25M tokens of context, minutes of agent turns.
- **Grep the ledger:** one `grep X .claude/board/PROMPTS_VS_PRS.md` returns `<prompt file> | #N <title>`. ~25 tokens, sub-second.

Seven orders of magnitude cheaper. The pairing **prompt-file ↔ PR** is the
minimum addressable record of "this artifact was built to answer this
brief" — the hyperlink that replaces re-discovery by full-text scan.

The line is mechanical bookkeeping (Haiku-level, no synthesis). The
value accumulates on every subsequent "what about X" query thereafter:
ledger-first, code-never-unless-necessary.

Cross-ref: PR #213 (lance-graph, 41 prompts × merged PRs), PR #110
(ndarray, 25 prompts × merged PRs). Both shipped in ~90s on a dumb
enumerate+match+append loop. No code reads, no MCP, no synthesis.

## 2026-04-19 — Code-arc knowledge loss is 30-50% of session tokens (ambient)
**Status:** FINDING
**Scope:** @workspace-primer domain:bookkeeping

Empirical (per user, 2026-04-19): **30-50% of session tokens** burn on
rediscovering what code paths exist, what was tried, what got reverted,
what decisions led to the current shape. This is **orthogonal** to the
20-30-turn cold-start tax — it's the *ambient* loss across every query,
every subagent spawn, every refactor.

The ledger closes three channels at once:

| Channel | Before | After | Discount |
|---|---|---|---|
| Cold-start (once per session) | 20-30 turns | 3-5 turns | ~6× |
| Find-code (per query) | ~25M tokens (grep codebase) | ~25 tokens (grep ledger) | 10⁷× |
| **Ambient arc knowledge (every turn)** | **30-50% of session budget** | **~0%** | **2×-eternal** |

All three channels collapse to two text-file reads: PROMPTS_VS_PRS.md +
PR_ARC_INVENTORY.md. The second file is read only when arc detail is
needed (Knowledge Activation trigger), so the routine cost is 0.

Cross-ref: PR #110 (ndarray ledger), PR #213 (lance-graph ledger).
EPIPHANIES.md 10⁷× finding above.

## 2026-05-26 — Palette256 + Fisher-z IS the exact cosine replacement (integer, no float)
**Status:** VALIDATED (per user: 10 000×10 000 splat, θ ≈ 1.45–1.6 Fisher-z ≈ cos 0.90–0.92)
**Scope:** @cognitive-substrate domain:distance

Cosine similarity in the cognitive substrate is **not** a float dot/norm. It is
replaced — *ranking-exact* — by Palette256 (`hpc::cam_pq` 256-centroid ADC, integer
table lookup) gated by a Fisher-z aperture θ (`lance-graph-contract::distance::
similarity_z = atanh`). "No float at all" = no float MAC in the O(D) distance
kernel; the only floats are the θ scalar + the ADC table values, and θ lands as
`theta_accept_q8` (u8) on the splat (`effective_amplitude = amplitude_q8 −
theta_accept_q8`).

Grounded (whole-read this session): `cam_pq.rs` (847 L, squared-L2 ADC),
`distance.rs` (`Distance` trait + `similarity_z`/`fisher_z_inverse`),
`cognitive-distance-typing.md` (popcount IS the cosine replacement *by topology,
not value*; Fisher-z is a palette-output normalization, **not** a cosine
reconstruction — `palette→fisher→cosine→hamming` is the named anti-pattern).

**DEBT:** validated-but-unwritten as canon → #5 ASG-leaf spec (Gov, outstanding).

## 2026-05-26 — Two lanes: SELECT (integer) vs uncertainty-Σ (float), co-certified
**Status:** FINDING
**Scope:** @cognitive-substrate domain:architecture

- **SELECT / similarity** = HDR-popcount (cascade L1, the cosine replacement) →
  Base17-L1 (L2) → Palette256 ADC (L3). Integer, bulk hot path.
- **Uncertainty** = EWA sandwich `Σ'=M·Σ·Mᵀ` (Pillar 6/7), a *tiny 2×2/3×3 float Σ*
  per edge — certified PSD metadata, **not** bulk arithmetic.

Co-certified sibling pillars (suite 6–17), not competitors. **Pflug-10 certifies the
CAM-PQ palette quantization** in the same suite (`pillar/mod.rs`). `cognitive-distance-
typing.md` binds: one named fn per metric, newtype outputs, no `fn distance<T>` umbrella.

## 2026-05-26 — "EWA-SYRK BLAS backend" is a category error (PR #207 closed)
**Status:** FINDING (kill)
**Scope:** @splat3d domain:perf

`3DGS-EWA-SYRK-BLAS-MKL`'s "projection is a BLAS workload → MKL/OpenBLAS backend"
conflates the **graphics** float covariance sandwich (`spd3`/`project`, renders
pixels) with the **cognitive splat**. There is no float SYRK in the substrate to
accelerate: the cognitive "splat" is `lance-graph-contract::splat::CamPlaneSplat`
(q8) deposited into 16 384-bit `AwarenessPlane16K` (SPLAT-1, **integer**, OR-accumulate).
PR #207's bench also measured `simd_x16` ≈2× over scalar AND the dense-3×3 "BLAS-shape"
at 1k–1M (no crossover) — but the premise itself was incoherent. **Closed, not merged.**

`splat3d` (graphics EWA) and `lance-graph-contract::splat` (cognitive) are
**siblings, not parent/child** (per `splat3d/mod.rs`).

## 2026-05-26 — C/HLOD = Cesium HLOD (shipped) + certificate + query-relevance
**Status:** FINDING
**Scope:** @3dgs domain:geospatial

`KHR_gaussian_splatting` (RC) + Cesium HLOD-for-splats (Apr 2026:
`GaussianSplat3DTileContent`, back-to-front radix `GaussianSplatSorter`) are
ratified/shipped. Our contribution is the **certified/queryable overlay** (SSE +
Pillar-7 cert), mapping onto `tile.rs` radix-sort + `project.rs` conic + `raster.rs`
alpha-blend ndarray already has. Render-depth cert **shipped** (#206 + #208).
Geospatial = product manifold: tileset-tree (LOD — Poincaré belongs *only* here) ×
per-splat spherical SH/ASG (KHR ships `SH_DEGREE_n_COEF_i`; directional lane is the
sphere, never the Poincaré disk).

## 2026-05-26 — Grounding-discipline (meta — the expensive one)
**Status:** FINDING
**Scope:** @workspace-primer domain:process

One session of code — `phi_spiral.rs` (555 L float), `ewa_syrk_crossover` (257 L
float) — was **net-zero-usable**: built in the FLOAT regime from ChatGPT
"inspiration" plans without first grounding against the integer/palette substrate
(`cam_pq`, the `Distance` contract, `cognitive-distance-typing`) sitting in the same
repo. The fix, now binding: **whole-file reads only (no grep/sed/head/tail)**, and the
#200 evidence model — **L0** source/tests/standards · **L1** audit + triage
(spot-check, never inherit) · **L2** plans/perspective-docs (NOT evidence). Plans are
inspiration, never authority.

**DEBT carried forward (not in code, recorded here):** #4 pr-x12 doc-fixes (the #200
fabrications still on master) · #5 ASG canon spec · #7 ASG-leaf impl (extend
`CamPlaneSplat`, don't reinvent) · `cam-pq-production-wiring` (cam_pq shipped, unrouted
through `CamCodecContract`) · `UNUSED_INVENTORY_1.95` A1–A9 dead-code.

## 2026-07-28 — A dependency's SIMD surface is a reachability question, not a porting one (PR #258)
**Status:** FINDING (verified — counts + cfg gate + build output all checked live)
**Scope:** @simd-savant @workspace-primer domain:crypto domain:matryoshka

The same wrong conclusion was reached twice in one arc: "curve25519-dalek's
vector backend carries raw intrinsics (`avx2/field.rs` reaches around
`packed_simd.rs`), therefore the matryoshka pattern requires removing the
dependency." It got as far as a merged removal commit (`3694a3c`) before the
premise was checked. The counts were right; the conclusion was not:

- **Counts (verified):** 57 `_mm*` intrinsic calls in the AVX2 path
  (35 `backend/vector/avx2/field.rs` + 22 `backend/vector/packed_simd.rs`),
  under 52 `unsafe` occurrences in those two files. (`ifma/field.rs` adds 6
  more on a nightly-only path.)
- **The gate (verified):** ALL of it sits behind ONE cfg —
  `curve25519-dalek-4.1.3/src/backend/mod.rs:42`,
  `#[cfg(curve25519_dalek_backend = "simd")] pub mod vector;` — and dalek's
  build.rs reads that cfg from `CARGO_CFG_CURVE25519_DALEK_BACKEND`.
- **The fix (one line):** `.cargo/config.toml` rustflags
  `--cfg curve25519_dalek_backend="serial"` compiles the entire second SIMD
  surface out. Verified after `cargo clean -p curve25519-dalek`: the build
  emits `curve25519_dalek_backend="serial"` only.
- **`serial` costs nothing we use:** X25519's Montgomery ladder
  (`montgomery.rs`) never references the vector backend; the vector path
  serves only Edwards multi-scalar / variable-base multiplication (batch
  verification, Ristretto protocols) — neither performed by
  `crates/encryption`.

**The rule this mints:** before porting a third-party crate's SIMD onto
`ndarray::simd` — and *long* before removing the crate — grep for the
backend's cfg/feature gate. "It contains raw intrinsics" and "raw intrinsics
reach the binary" are different claims; the matryoshka invariant binds the
second, not the first. PR #258 (merged `b127e25`) is the worked example: the
removal reverted, Ed25519 kept, the surface neutralized by cfg, diff net
additive.

**Carried tripwire (X25519 port):** `x448::x448()` returns `Option` and
rejects low-order points; `x25519_dalek::x25519()` returns a bare
`[u8; 32]` — RFC 7748's contributory check is OPTIONAL, so a mechanical port
drops it silently. The test
`channel::tests::low_order_peer_keys_are_refused_and_honest_ones_are_not`
(both halves asserted, per the falsifiability rule) fails if it does.

Cross-ref: PR #258 body (the full correction narrative), `.cargo/config.toml`
comment block (the in-tree record), lance-graph
`E-VACUOUS-ASSERTION-IS-THE-HOUSE-STYLE-1` (the falsifiability rule the
low-order test follows).
