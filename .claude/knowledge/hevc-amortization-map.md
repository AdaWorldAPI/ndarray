# HEVC / H.265 amortization map — onto the BF16 16×16 AMX tile substrate

> READ BY: cognitive-architect, savant-architect, amx-savant, truth-architect,
> l3-strategist, product-engineer
>
> Status ledger. Every row is tagged **[MEASURED #PR]** (probe run) /
> **[SHIPPED]** (code exists + tested, not yet wired into the codec pipeline) /
> **[MECHANISM — unmeasured, probe named]** / **[PARTIAL]** / **[DOESN'T FIT]**.
> No unmarked conjecture. Companion to
> `.claude/knowledge/pr-x12-codec-cognitive-substrate-mapping.md` (the E-7/H-7
> plan this measures against).

## The thesis (operator, 2026-07-04)

> Use the BF16 16×16 AMX GEMM / blasgraph neighborhood / Morton-tile pyramid /
> perturbation-shader cascade (+ AMX masking or other algorithms for certain
> stages) **to amortize H.265 / HEVC.**

Amortize = the codec's per-block marginal cost collapses to **a gather (bit
shift) + one 16×16 BF16 AMX tile op** (`hpc::bf16_tile_gemm::bf16_tile_gemm_16x16`
→ `TDPBF16PS`), routed over a Morton/HHTL tile pyramid whose codebooks/bases are
**deterministic and built once** (the helix "template is free, regenerable —
only the endpoint costs" principle), then reused across every block and every
frame. One tile machine + one pyramid + one place/residue codec, not N bespoke
HEVC kernels.

## The one primitive, three layers

| layer | what it is | shipped in |
|---|---|---|
| **address (bit shift)** | integer motion / tile position / CTU quad-tree = pure addressing, zero arithmetic (`VPGATHERDD`) | `hpc::cascade` (Morton), blasgraph `heel_hip_twig_leaf` / `clam_neighborhood` (lance-graph) |
| **couple (tile op)** | a 16×16 BF16 GEMM `C = A·B` = one `TDPBF16PS` — every linear field coupling / transform / covariance projection | `hpc::bf16_tile_gemm::bf16_tile_gemm_16x16`, `hpc::splat3d::spd3::sandwich_x16` |
| **code (place/residue)** | deterministic pyramid PLACE (free) + coded RESIDUE magnitude; the perturbation phase IS the Walsh–Hadamard sign transform of the address tree | `hpc::splat3d::helix_orient`, `crates/helix` (lance-graph), OGAR `guid-prefix-shape-routing.md §4b` |
| **entropy (palette pack)** | small-alphabet indices → variable-width SIMD bit-pack (VPSHUFB, 16 idx/cycle) — the parallel entropy layer that replaces serial CABAC | `ndarray::palette_codec` / `nibble` / `byte_scan` (Pumpkin/Minecraft port; 28+23+18 tests) |

The perturbation-shader cascade and the codec transform are **the same object**:
OGAR canon states "the perturbation pyramid becomes the Walsh–Hadamard transform
of the address tree" (`OGAR/CLAUDE.md`, Bipolar-phase pyramid), and PR #232
measured WHT (the ±1 add/subtract sign transform) as the codec transform,
matching the DCT within 2%. Same ±1 sign butterfly, two names.

## HEVC stage → substrate primitive

| HEVC stage | reduces to | status |
|---|---|---|
| **Motion estimation** (block match / SAD→SSD) | `A·B` middle term of SSD = i8/bf16 GEMM | **[MEASURED #230]** — ME argmin == direct SSD, 256/256, via `int8_gemm_i32` |
| **Motion compensation** (integer-pel) | gather at MV offset + residual add | **[MEASURED #230]** — bit-exact `recon = shifted_ref + residual` |
| **Inverse transform** (integer DCT/DST 4–32) | separable `M·X` = tile GEMM; WHT (sign-only) suffices | **[MEASURED #232]** — WHT tracks DCT ≤2%; transform is a *no-op* when motion whitens the residual (dct/dir → 1) |
| **Intra prediction** (planar/DC/33 angular) | directional extrapolation from neighbors = linear field op = tile GEMM; the angle is a helix direction code | **[MEASURED shape #233]** — field coupling `Xᵀ·X` / `Mᵀ·Σ·M` = tile op @ 0.1–0.4% Frobenius. Full intra-mode probe: **unmeasured** |
| **Sub-pel interpolation** (8-tap luma / 4-tap chroma) | separable FIR = small GEMM per block = tile op | **[MECHANISM — unmeasured]** probe: `subpel_tap_tile` (8-tap separable == `bf16_tile_gemm_16x16`, bit-close to the reference filter) |
| **Reference-sample smoothing / covariance** | Gaussian/covariance projection = the EWA sandwich | **[MEASURED #233]** — `sandwich_x16` shape @ 0.4% |
| **CTU quad-tree partition** (64→32→16→8) | Morton/HHTL tier cascade = the address pyramid | **[MECHANISM — shipped]** `hpc::cascade`, blasgraph HHTL; codec `Ctu` quad-tree. Routing-parity probe unmeasured |
| **Deblocking + SAO** (in-loop filters) | conditional edge filter + offset LUT — clip/branch-heavy, *not pure GEMM* | **[PARTIAL — AMX masking]** the operator's "AMX masking": the filter is a masked tile op (predicated add), SAO is a gather-LUT. Fits with masking, not plain GEMM. Unmeasured |
| **Entropy — palette-pack module** (parallel, small-alphabet) | small alphabet (≤256 basin/centroid indices) → **variable-width SIMD palette pack** (VPSHUFB, 16 idx/cycle) | **[SHIPPED module / PROPOSED codec wiring]** `ndarray::palette_codec` (Pumpkin/Minecraft port, 28 tests): `pack_indices_simd`/`unpack_indices_simd` bit-exact all 1–8 widths; `transcode` = per-block bit-width; `nibble`+`byte_scan` complete the byte layer. **Not yet wired as the codec's entropy stage** — the shipped codec (`hpc::codec::ans`) currently does static per-block rANS over the 4 `CellMode` tags. Using palette-index packing *as* the codec's entropy layer is the proposed change. |
| **CABAC entropy** (context-adaptive binary arithmetic — `.265` compat only) | serial, bit-level, data-dependent arithmetic coding | **[DOESN'T FIT — and the substrate can route around it]** CABAC is a serial dependency chain, not a GEMM; AMX/tiles do not help. It is required ONLY to decode an *existing* `.265` bitstream (the M2 serial front-end). The substrate quantises to a small palette, so its entropy need is met by the SIMD-parallel palette-pack row above (SHIPPED module; PROPOSED as the codec's stage) — with, at most, a light static-rANS pass over the packed indices for the last few %. So "route around CABAC" = replace context-adaptive arithmetic coding with parallel palette-pack (+ optional non-adaptive rANS), NOT "zero entropy coding." The boundary is `.265`-compat, not the substrate's own entropy needs. |

## What amortizes, precisely

- **Build once, reuse per block.** The transform basis (WHT ±1 sign matrix), the
  interpolation taps, the Morton pyramid, the palette/centroid codebooks, and the
  golden-spiral place template are **deterministic** — computed once, never
  stored per-block (helix principle). The 64×64 gridlake = 4×4 = 16 tile positions.
- **BUT: dispatch at the R-5 batch crossover — the tile path is NOT unconditional.**
  Per the companion plan's **R-5** rule (`pr-x12-codec-cognitive-substrate-mapping.md`
  §2.2; `pr-x12-substrate-canon-resolutions.md` §R-5): a single 32×32 DCT via
  butterflies is ~80 ops vs ~32K for the GEMM form — **per-block, butterflies win
  by ~400×**. The BF16 tile GEMM only wins **above** the batch crossover, where
  hardware fusion amortises across many blocks. Crossover is per-arch:
  **SPR = 64, ICX = 32, Zen4 = 96, Apple M = 256, Graviton = 128** blocks. So the
  amortized dispatch is: **per-CTU / low-latency / small-batch → per-block WHT
  butterflies; per-frame / large-batch (N ≥ crossover) → batched BF16 tile GEMM.**
  A ledger that read "every block pays a tile op" would mis-route small-CTU work
  onto the tile path and invalidate the measurement — ship both, dispatch on N.
- **One kernel family, many stages.** ME, inverse transform, intra prediction,
  sub-pel, and covariance are all the same *algebra* (`M·X` / `Xᵀ·X` / `Mᵀ·Σ·M`),
  realised as **butterflies below the crossover, `bf16_tile_gemm_16x16` above it**.
  The substrate is the zero-FP cognitive shader (gather + tile/butterfly ops), so
  "the codec IS the substrate" holds down to the instruction (`TDPBF16PS` in the
  batched regime).
- **The transform mostly vanishes.** Per #232, good motion whitens the residual
  and the transform's advantage → 1 (a no-op); where it survives it is a
  multiply-free WHT. So the amortized codec spends tile ops on prediction, not on
  transform-coding a well-predicted residual.

## The two honest boundaries

1. **CABAC is not a GEMM — but the substrate routes around it.** CABAC entropy
   (de)coding is inherently serial arithmetic coding; the tile substrate does not
   amortize it, and a real `.265` decoder still needs it (the M2 serial
   front-end). BUT the substrate does not *need* CABAC: it quantises to a small
   palette (≤256 basin/centroid indices), so its entropy need can be met by
   **variable-width SIMD palette packing** (`ndarray::palette_codec`, the
   Pumpkin/Minecraft port — `unpack_indices_simd`, VPSHUFB, 16 idx/cycle,
   `transcode` for per-block bit-width). This module is **shipped + tested** but
   **not yet wired as the codec's entropy stage** (the shipped codec uses
   `hpc::codec::ans` static rANS over the 4 `CellMode` tags) — the wiring is the
   proposed change. The point stands: a substrate-native pipeline has **no
   context-adaptive serial stage** — `gather + tile op + palette-pack` — with, at
   most, a *non-adaptive* static-rANS pass over the packed indices (order-0,
   interleavable, not a CABAC-style dependency chain). The boundary is `.265`
   *compatibility*, not the substrate's own entropy needs. (Palette packing is a
   rate/parallelism trade vs an optimal arithmetic coder — near-optimal when the
   palette is well-chosen.)
2. **The golden-spiral is for directional/sparse data, not dense scalars.** Per
   #231 (measured negative), coding a *dense scalar* residual with sparse
   golden-spiral anchors loses to local DPCM; the golden-spiral / helix codec
   belongs on directions (surfel normals, plausibly MV fields), not the dense
   luma residual. Use DPCM/WHT there.

## Probe queue (next falsifiable steps, in order)

1. **`subpel_tap_tile`** — HEVC 8-tap separable interpolation == `bf16_tile_gemm_16x16`,
   Frobenius-close to the reference FIR. (Cleanest next measurement; mechanism
   already clear.)
2. **`intra_angular_tile`** — the 33 angular intra modes as directional tile-op
   extrapolations; measure PSNR vs the HEVC reference predictor.
3. **`deblock_masked_tile`** — deblocking as a predicated (AMX-masked) tile op;
   measure vs the HEVC clip-based filter (tests the operator's "AMX masking").
4. **CTU quad-tree routing parity** — Morton/HHTL cascade partition == HEVC CTU
   split decisions on a real frame.
5. **M2 (separate track)** — CABAC serial front-end to extract MVs+residuals from
   a real `.265` bitstream. Not tile-amortizable; the honest hard part.

## Anchors

PRs #230 (ME=GEMM, MC bit-exact), #231 (residue upscaling — measured negative for
dense scalar), #232 (transform no-op × WHT), #233 (field coupling = AMX tile op,
on real `TDPBF16PS`). Shipped: `hpc::bf16_tile_gemm`, `hpc::splat3d::spd3`,
`hpc::cascade`, `hpc::codec`; the Pumpkin/Minecraft entropy layer
`palette_codec` / `nibble` / `byte_scan` (69 tests); lance-graph blasgraph HHTL;
OGAR perturbation pyramid = WHT.
