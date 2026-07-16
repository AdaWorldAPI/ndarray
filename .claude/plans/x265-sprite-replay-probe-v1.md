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
