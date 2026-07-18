# PROBE-SPRITE-REPLAY v1 — x265 I/P/B grammar over HHTL sprites with helix motion

> Date: 2026-07-16. Status: SPEC'D (operator-directed; execution wave is
> the next plateau after TD-BGZ-TENSOR-PRE-LANE-REVIEW lands).
> Scope guard, operator's words: the initial test is NOT reaching for the
> stars (H.268) — it is **a simple replay of x265's GOP grammar** using
> our own primitives, on our own hardware tiers.

## The thesis (the new amortization)

**A moving object = an HHTL-addressed spatial sprite** — a Gaussian-splat
set anchored at an HHTL address — **with a helix direction code as its
motion primitive**, mapping directly onto x265's basic frame operations:

| x265 op | Sprite equivalent | Primitive |
|---|---|---|
| **I-frame** | the sprite's full splat set at its anchor address | splat3d EWA set + HHTL anchor (HEEL\|HIP\|TWIG) |
| **P-frame** | ONE helix motion code per sprite (object-level), replacing the per-block MV field | helix `ResidueEdge` (24-bit hemisphere) or `Signed360` (48-bit signed full-sphere) |
| **B-frame** | parametric interpolation along the helical path between two anchors | evaluate the sprite at t ∈ (0,1); bidirectional weights |

The amortization stack:
- **Motion search dies.** x265's dominant encoder cost (per-block MV
  search) becomes address arithmetic: the sprite's anchor moves; its
  splats ride along. The nested stacked-inverse-pyramid 4×4 ergonomics
  re-rasterize the sprite footprint through the Morton cascade at
  whatever LOD the certificate demands (depth_cascade actions).
- **One substrate, both consumers** (per E-H268-REPLAYABLE-TILE-1): the
  same tile cascade serves codec rasterization and shader dispatch.
- **The wgpu harness is shared.** The minimal render harness this probe
  needs is exactly the harness PROBE-GPU-LUT has been gated on — build
  once, both probes consume it.
- **Fisher-z canon applies** (E-FISHERZ-CANONICAL-COSINE-REPLACEMENT-1):
  similarity/direction-adjacent reads carry as normalized i8; helix runs
  the 2z rung of the same analytic family (`batch_fisher_z` exists).

## Scope guards (graded, non-negotiable)

- **NOT H.268.** No scene-codec claims, no beyond-VVC claims.
- **NOT bitstream/byte parity with x265.** "Replay of x265" = replaying
  the **I/P/B operational grammar** with our motion primitive; x265
  itself is at most an optional external reference point (bitrate/PSNR
  context on the rasterized sequence), never a parity gate.
- **GPU is render-grade only** (C5/C9 discipline): the bit-exactness
  claims live on the CPU/wasm integer/pinned-math path (sprite STATES:
  helix codes → positions); raster output compares to tolerance, GPU
  raster is a visual tier.
- All claims [S/H] until the probe runs; this plan is the spec, not a
  result.

## Test spec (minimal, deterministic)

1. **Scene:** N=8 sprites × K=64 Gaussians each (seeded), moving on
   ground-truth helical paths (the helix codes ARE the ground truth —
   encode direction as `ResidueEdge` AND `Signed360`, measure both
   widths' quantization error).
2. **Encode:** GOP = I B B P B B P … (classic pattern): I = splat dump +
   anchors; P = per-sprite helix delta codes; B = no stored motion —
   derived by parametric interpolation between surrounding anchors.
3. **Decode tiers:** (a) CPU native (ndarray `splat3d` EWA rasterizer);
   (b) wasm (same code — the shipped parity-CI pattern); (c) wgpu quad/
   splat raster (a2ui-paint tier; N2 gate applies).
4. **Pass criteria:**
   - **Replay determinism:** decoded sprite states (positions from helix
     codes, pinned unfused math) bit-identical CPU native vs wasm.
   - **B-consistency:** a B frame decoded forward-from-I and
     backward-from-P agrees with the parametric midpoint to a stated
     tolerance (the bidirectional check).
   - **Motion fidelity:** helix-coded direction reproduces ground-truth
     paths within the register's quantization bound (report 24-bit vs
     48-bit error curves).
   - **KILL:** if object-level helix motion cannot express the test
     paths without per-splat residual fields (i.e. the "one code per
     sprite" claim collapses back into a dense MV field), the sprite
     amortization dies as stated and the finding is the honest record.
5. **Optional context (not a gate):** run actual x265 over the CPU
   raster PNG sequence; report bits/frame + PSNR as an external anchor.

## Standing corrections folded from the probe wave (same doc pass)

- §10(i) cache-honesty: the 192 B/3-cache-line tile claim is fully
  honest only under the analytic Fisher-z canon (a materialized 256² u16
  table is 128 KB — L2-resident, not L1D); the analytic path drops table
  residency to 8 B.
- PROBE-WH-MAG-2's deferral condition is weaker than recorded: the
  codec's 2-bit mode grammar (Skip/Merge/Delta/**Escape**) already IS
  the per-tile escape tier the probe lacked; re-running WH under the
  mode grammar is the natural PROBE-WH-MAG-2 home.

## Execution model

Same wave pattern: drafters (grindwork: scene generator, GOP encoder,
CPU/wasm decode, harness), filigree adjudication vs the pass/KILL bands,
central gates, PR, autonomous merge (standing authority). The wgpu tier
may land as a second commit gated on the shared harness; if it slips,
the CPU/wasm probe stands alone.

## Results (2026-07-16 — PROBE-SPRITE-REPLAY-CORE, reviewer-adjudicated)

**Verdict: PASS-AT-SIGNED360 (scoped) / ResidueEdge-24bit INSUFFICIENT. The KILL band did NOT fire.**

Probe: `lance-graph crates/helix/src/sprite_replay.rs` (`probe_sprite_replay_core`),
CPU-native core only (splat3d EWA raster + wgpu tiers deferred per the scope
guard); N=8 sprites, TOTAL=240, 6 anchors (I+5P), GOP `I B B P ...`. Deterministic
(SplitMix64); the test asserts determinism + finiteness + counts, never a verdict.

| Metric (world-unit Euclidean position error) | ResidueEdge (24-bit) | Signed360 (48-bit) |
|---|---|---|
| I+P anchor motion fidelity — mean / max | 9.98 / 42.43 | **0.0 / 0.0** |
| Sign::Pos sprites — mean / max | 3.55 / 17.60 | 0.0 / 0.0 |
| Sign::Neg sprites — mean / max | 16.41 / 42.43 | 0.0 / 0.0 |
| B-frame bidir-delta — mean / max | 6.04 / 22.50 | 0.0 / 0.0 |
| B-frame vs-truth — mean / max | 10.45 / 41.57 | 0.0 / 0.0 |

- **KILL did not fire** (plan §4): object-level helix motion does NOT need a
  dense per-splat residual field — one `Signed360` code per sprite per P-frame
  reconstructs motion + parametric B-frames exactly.
- **PASS is scoped (the load-bearing caveat).** Ground-truth motion is drawn
  from helix's OWN `signed_lift` template, so the 0.0 Signed360 result proves
  CAPACITY (16-bit azimuth indexes n∈0..240 trivially) + ROUND-TRIP (the
  hand-built public-primitive inverse recovers n and sign exactly) + SIGN
  carriage — NOT that arbitrary independently-captured object motion fits the
  helix manifold. "P-frame = one Signed360 code" is FEASIBLE at 48-bit for
  **helix-manifold** motion [proven]; arbitrary-motion generality stays [H],
  gated on a follow-up probe with independent ground truth.
- **ResidueEdge (24-bit) INSUFFICIENT — Signed360-only ruling.** Two
  independent failures: (1) hemisphere-blind (no sign bit → Neg sprites mirror
  wrong: mean 16.4 / max 42.4); (2) rank-adjacency hazard (8-bit `end_idx`
  bucket; a ±1 rank error under golden-spiral ~137.5° spacing lands a distant
  point — even Pos sprites: mean 3.55 / max 17.6). The sprite motion primitive
  should be Signed360; ResidueEdge is a residue-edge codec, not a motion code.

**Follow-up (named, deferred):** an arbitrary-independent-ground-truth motion
probe (a captured 3-D trajectory the encoder did not generate) to promote the
helix-manifold [proven] result toward arbitrary-motion [H].

## Results (2026-07-18 — PROBE-GPU-LUT, the shared wgpu decode-tier harness, main-thread-adjudicated)

**Verdict: HARNESS-REAL / CPU-primitive GREEN / GPU-exec COMPILED+SHIPPED, execution-parity adapter-deferred. The KILL ("GPU lane abandoned for LUTs") did NOT fire.**

Probe: `a2ui-rs crates/a2ui-paint/src/gpu_lut_probe.rs`. The wgpu decode tier
(§Decode tiers c, the `a2ui N2` queue row) was gated on a "shared PROBE-GPU-LUT
harness" — operator ruling this session pinned that harness to a2ui-paint's real
`wgpu = "22"` seam (WebGPU + WebGL2), the one in-scope GPU path (q2 `sculpt` +
ndarray `splat3d` both deliberately opt OUT of GPU; measured this session).

- **CPU-reference leg (ran here, PASS):** the 256²-u16 palette-distance LUT
  texture-gather (`textureLoad(lut,(q,k)).r` == row-major `lut[q*256+k]`) is
  bit-exact over all 65536 entries; table is symmetric + zero-diagonal +
  deterministic (SplitMix64); 256² u16 = 128 KiB (the §10(i) materialized-table
  figure). **This is the falsifiable core** — the arithmetic is what could be
  wrong; the GPU only executes it.
- **GPU-exec leg (COMPILED + SHIPPED, adapter-deferred):** the full
  R16Uint-LUT → fragment `textureLoad` → R32Uint target → readback →
  full-table-parity path compiles clean under wgpu 22 (WebGPU + WebGL2 via
  `glow`; `clippy --features wgpu -D warnings` clean, fmt clean) and
  **SKIPS-green** in this sandbox — measured: `libvulkan` loader present but
  **0 ICDs** installed → `request_adapter()` returns `None`. It runs the real
  65536/65536 parity wherever a WebGPU/WebGL2 adapter exists (lavapipe CI, a
  browser). Integer sampled texture + `textureLoad` + integer render target are
  all WebGL2-core, so the one shader covers both backends.
- **KILL did not fire** (§Decode tiers c / `a2ui N2`): the bgz17 256²-u16 table
  IS gatherable through a real in-scope wgpu texture, so the GPU LUT lane is not
  abandoned — the harness capability is proven buildable.
- **HONEST CAVEAT:** the GPU-exec *execution* was NOT run on silicon here (no
  adapter). "GPU-exec green" = COMPILES + SKIPS-cleanly + is the shipped WGSL,
  NOT "65536 texels compared on a GPU in this session." The CPU-reference is the
  leg that actually ran. Runtime-execution parity is the one piece that awaits an
  adapter environment (the `a2ui N2` render-parity-headless-vs-browser bar).
- **Boundary kept clean:** no bgz17 dep in a2ui-paint (charter: no consumer
  crate deps) — the 256² table is built deterministically with bgz17's table
  STRUCTURE (symmetric u16, zero diagonal); this is a HARNESS-CAPABILITY probe,
  not a bgz17 integration. Test-only `pollster` dev-dep for the async block.

**Consequence (scoped — corrected per codex P2 on ndarray #249):** only §Decode
tiers **(c) the wgpu harness** is structurally un-gated — the shared harness it
waited on is real and the LUT-gather compiles + CPU-proves. The remaining
deferral is narrow: run the GPU-exec parity in an adapter environment (lavapipe
CI or browser) to close the `a2ui N2` render-parity bar on silicon.

**Tier (b) — the wasm tier — is NOT un-gated by this wave.** Its gate is a
distinct check: **CPU-native vs wasm32 replay-determinism** of the decoded
sprite states (plan §4). PROBE-GPU-LUT recorded only a CPU-reference run + an
adapter-skipped GPU leg — **no wasm result**. Tier (b) still requires its own
CPU-vs-wasm parity run before it can be called un-gated; PROBE-GPU-LUT does not
touch it.

## Results (2026-07-18 — HEVC external anchor, §5 optional context, RUN + VISUAL)

The plan §5 optional anchor ("run actual x265 over the CPU raster sequence;
report bits/frame + PSNR") is now RUN — and made visual.

- **Scene:** 8 gaussian sprites tracing φ-spiral (golden-angle hemisphere)
  paths, alternating hemispheres by index parity — the sprite-replay scene
  (NUM_SPRITES=8, TOTAL=240) rendered to pixels. 320×240, 240 frames, a faint
  panning background so P/B-frames have global motion to track.
- **Encoder:** x265 3.5, preset medium, `--psnr`. x265 ran its OWN I/P/B GOP
  over our moving scene (the arc's "replay x265's GOP grammar" made literal):
  1 I, 56 P, 183 B, up to 5 consecutive B-frames.
- **Numbers:** raw Y4M 27,649,483 B → HEVC 43,115 B = **641×**; **1437.2
  bits/frame** (180 B/frame); Global **PSNR 60.94 dB** (Y 47.5–52.0 by
  slice-type; chroma neutral). Encode 316 fps. (Re-run after the CodeRabbit #738
  fix: the reproducer now uses `sprite_replay`'s canonical draw sequence +
  signed-z hemisphere projection; the earlier 578×/60.79 dB figures were the
  pre-fix scene where `sign` was inert.)
- **Roundtrip visual:** frames decoded back OUT of the `.265` bitstream (ffmpeg)
  into a 5-frame motion montage + animated GIF — the sprites visibly at
  different positions/sizes across time.
- **Reproducer:** `lance-graph crates/helix/examples/hevc_moving_scene.rs`
  (std-only, deterministic SplitMix64) → Y4M; `x265 --input scene.y4m --y4m
  --preset medium --psnr -o scene.265`; `ffmpeg -i scene.265 … montage/gif`.

**Reading (honest):** this is an EXTERNAL ANCHOR (plan §5, explicitly "not a
gate"), NOT a claim about our codec. The 578× / 60.79 dB are **x265's** numbers
on a smooth-gaussian synthetic scene that compresses easily — they anchor "what
a stock HEVC encoder does with this content," not "our primitives beat x265."
The arc's own sprite-replay motion coding (E-SPRITE-IPB-HELIX-1: one Signed360
code per sprite per P-frame) is the thing being contextualized; the
bitrate-comparison study (our object-level motion codes vs x265's per-block MV
field on the SAME scene) is a NAMED follow-up, not done here.

## Results (2026-07-18 — HEAD-TO-HEAD: our helix motion codes vs x265 MV field)

The head-to-head the HEVC anchor set up: our object-level helix motion codes vs
x265's per-block MV search, on the SAME φ-spiral scene. Reproducer:
`lance-graph crates/helix/examples/hevc_headtohead.rs` (uses the REAL helix
public API — `ResidueEncoder::encode_signed` / `Signed360` / azimuth-roundtrip
decode — not a re-inline; 3 tests, clippy-clean).

| Lane | Bytes | bits/frame | PSNR | Codec |
|---|---|---|---|---|
| A — x265 3.5 medium | 43,115 | 1437.2 | 60.94 dB | general block-MV, blind to scene structure |
| B — our helix `Signed360` | **432** (192 appearance + 240 motion) | 14.4 | **∞ (bit-exact)** | object-motion: 1× 6-byte code / sprite / P-frame |

**Verdict: TAUTOLOGICAL-WIN / MODEL-MATCHED LOWER BOUND (KILL did not fire; NOT a
general-codec win).** Lane B was handed the EXACT generative model of a
self-generated φ-scene; the ∞ PSNR is the decoder re-running the same generator.
Legitimately establishes the object-motion amortization's concrete bit-cost — one
6-byte `Signed360` per sprite per P-frame = 240 B of motion for the whole
240-frame GOP — replacing x265's entire per-block MV field, bit-exact on
model-matched content. Does NOT beat x265 in general: arbitrary non-φ-manifold
motion needs stored residuals (PROBE-SPRITE-REPLAY's [H] gate), the named
follow-up. Board: lance-graph `E-X265-HEADTOHEAD-1`.

## Follow-up probe queued — PROBE-SPLAT-μ-HYDRATION-RHO (the spatial cousin)

The operator's 3DGS↔blasgraph↔gridlake presumption, adjudicated by the
convergence-architect (lance-graph `E-3DGS-MU-HYDRATION-1`): `splat3d::GaussianBatch`
is ALREADY SoA, so the only net-new "gridlake" claim is **address-derived gaussian
μ** (WORTH-EXPLORING on the φ-manifold — the SAME [H] gate as this head-to-head's
temporal tautology). "EWA = blasgraph semiring" DROPPED (depth-sorted Porter-Duff
*over* is non-commutative → axiom break). Named probe **PROBE-SPLAT-μ-HYDRATION-RHO**
runs on a real trained Inria `.ply` (non-φ input via `splat3d::ply`): μ-hydration
ratio + semiring-matches-EWA (commutative-⊕ vs sorted-over) + SoA-sweep throughput.
PASS→[H] / KILL→DROP bands in the board entry.
