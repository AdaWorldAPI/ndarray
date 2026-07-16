# PR-X12 / H.268 — Morton-Cascade × Perturbation-Pyramid × wgpu/wasm Synergies (graded)

> READ BY: savant-architect, codec-architect, l3-strategist, truth-architect,
> sentinel-qa, product-engineer, vector-synthesis.
>
> Date: 2026-07-16
> Status: **graded synthesis** — every row below carries a verdict from an
> adversarial verify pass (10 claims × skeptic agents with file:line receipts,
> run under the PR-X12-docs-audit discipline). Verdicts: **FEASIBLE-NOW**
> (shipped or trivially composable from shipped pieces), **NEEDS-PROBE**
> (plausible, gated on a named unrun probe/bench), **OVERCLAIM-CORRECTED**
> (the naive claim failed against source; the corrected wording is what this
> doc carries). Nothing here is presented above its evidence grade.

---

## 0. Codename ruling — H.268 (INTERNAL)

**"H.268" is the internal codename for what the doc set previously called
"x266"** — the PR-X12 3DGS scene codec / deterministic beyond-standards
track (`pr-x12-x266-3dgs-spacetime-upscaling.md`).

- INTERNAL ONLY. It is **not** an ITU designation and must never be
  presented as one: H.267 itself is still a prospective standardization
  effort (CfP July 2026, finalize ~2028), and a real ITU "H.268" would be
  a 2030s artifact. The codename encodes intent — *the generation after
  the one currently being standardized* — nothing more.
- Filenames keep the historical `x266` slug (link stability); prose says
  "H.268 (internal codename)" on first use.
- Never in commit messages to public-facing artifacts as a standards claim;
  never in marketing copy without the INTERNAL qualifier.

---

## 1. The hand we actually hold (verified FINDINGs, whole-file receipts)

| Substrate | Status | Receipt |
|---|---|---|
| **wasm SIMD128 backend** — `F32x16=[v128;4]`, `F64x8`, `I8x16` (incl. W1a primitives), `U32x16` ARX; kernels `hamming_u8x64_wasm` (Fingerprint<256>), `base17_l1_wasm`, `codebook_gather_f32x4_wasm`, `dot_f32x4_wasm`; optional relaxed-simd madd | FINDING — CI bit-parity-verified under Node 22 (`wasm-simd/parity-node`) | `src/simd_wasm.rs:65,87,742,1060,1108`; `.github/workflows/ci.yaml:115-142`; `crates/wasm-simd-parity/src/lib.rs:13-67` |
| **Browser crypto lane** — vendored chacha20 `ndarray_simd` backend lowers the same `U32x16` to AVX-512 on server and v128 on browser | FINDING (shipped 2026-07-12; CI compile-guarded) | `vendor/chacha20/src/backends.rs:11-18` |
| **Codec wire format** — 16-bit header: 12-bit basin + 2-bit mode (Skip/Merge/Delta/Escape) + 2 reserved; `pack_leaf` 2/3/3/6 bytes; integer-only rANS mode coder (static per-block tables, no float, no unsafe) | FINDING | `src/hpc/codec/mode.rs:15-41,94,212-295`; `src/hpc/codec/ans.rs:1-56` |
| **256×256 tables are texture-isomorphic** — bgz17 `PaletteDistanceTable` is a dense, zero-padded row-major `vec![0u16; 256*256]` (≡ one R16Uint texture); compose = k×k u8 (≡ R8Uint); bgz-tensor Attention/Compose tables mirror; lookups exact, no filtering semantics | FINDING (shape); GPU realization ABSENT | `lance-graph crates/bgz17/src/palette.rs:77-88`, `palette_semiring.rs:20-43` |
| **Morton 2bit×2bit primitives exist** — lance-graph `FacetTier::morton` (4⁴ nibble ancestry), symbiont `morton4`, perturbation-sim cascade keys; OGAR 3×4 canon pins tier-of-level = `>>2` shift [G] | FINDING (primitives); the CTU codec does NOT use them (see §3 row 1) | lance-graph facet/symbiont sources; OGAR `CLAUDE.md` §3×4 |
| **Certificate-gated cascade** — splat3d `depth_cascade.rs`/`depth_cert.rs`: HEEL→HIP→TWIG→LEAF preselection with Reject/KeepCoarse/Refine/ProjectExact/RenderExact actions | FINDING — but it gates **render** work on decoded data, not decode work | `src/hpc/splat3d/depth_cascade.rs:1-65` |
| **Deterministic phase generator** — helix `CurveRuler` stride-4-over-17 coprime walk (bit-exact integer, full-17 permutation tests) | FINDING (the generator leg of the pyramid) | lance-graph `crates/helix` |
| **wgpu in the workspace** — a2ui-paint only: **off-by-default** feature, constant-color quad pipeline, **no textures/bind groups**, wgpu's `webgl` cargo feature NOT enabled (WebGL2 is doc-comment-only), untested, repo has no CI. ndarray itself: **zero wgpu by design** ("no GPU, no wgpu", splat3d) | FINDING (negative) | a2ui-rs `a2ui-paint` manifest+src; ndarray `Cargo.lock` (0 matches), `src/hpc/splat3d/mod.rs:7-9` |
| **Perturbation pyramid + two-algebra rule** — D-PHASE and D-WHP are operator-pinned **[H] hypotheses with named unrun probes** (PHASE-1, PERT-RHO, PYR-1; WHP-1..4), CANON-pin not CODED; losslessness fenced to *synthesis* (dither/anti-moiré grade) with quorum-certificate escalation | CONJECTURE [H], probe-gated | OGAR `docs/DISCOVERY-MAP.md:249-250`, `CLAUDE.md` fences; J2 kill condition |

## 2. The industry walls (from `pr-x12-h266-h267-standards-landscape.md`, sourced)

W1 — **Complexity spiral:** ECM-16.1 buys ~27% over VTM-11 at complexity the
industry itself calls impractical. W2 — **NN-in-loop decoder cost:** NNVC
filters ≈9% RA each at kMAC/pixel costs; mobile vendors weight decoder
complexity above bitrate. W3 — **No browser story:** no post-VVC codec has a
credible wasm/WebGL deployment path; VVC browser decode is effectively absent.
W4 — **Upscaler drift:** un-standardized AI upscaling (DLSS/RIFE-class) is
non-reproducible across model versions; standards-track upscaling is 2D RPR
only. W5 — **3DGS delivery weight:** stock web viewers ship tens of MB per
scene. W6 — **Serial entropy:** CABAC's per-bit context serialization resists
GPU/SIMD decode.

## 3. The graded synergy matrix

| # | Industry wall | Naive synergy claim | **Verdict** | What is true (corrected wording) | Gate |
|---|---|---|---|---|---|
| 1 | W1 (tree-walk partitioning) | "Morton 2bit×2bit makes the CTU quad-tree flat address arithmetic" | **OVERCLAIM-CORRECTED** | Shipped Morton primitives prove a 4-level quadtree address CAN be pure shift/interleave math — but the shipped `ctu.rs` stores an **arena tree with `[NodeIdx;4]` child links and O(N) BFS walks**, zero Morton code in the codec dir. The flat 85-slot SoA re-plumb (1+4+16+64, index computable from address) is a plausible, unimplemented refactor with unmeasured GPU/wasm advantage | **PROBE-MORTON-CTU** (new): flat Morton-addressed arena vs pointer arena, bench partition sweep |
| 2 | W3 (browser LUT decode) | "256-palettes are textures; decode = one texelFetch on WebGL2/WebGPU" | **NEEDS-PROBE** | The data shapes are texture-isomorphic **today** (dense 256×256 u16 / k×k u8, exact unfiltered lookups — R16Uint/R8Uint + `texelFetch`/`textureLoad` is standard practice); a full SPO distance = 3 fetches, CAM-PQ path = 6, multi-hop compose = dependent-fetch chain. **Zero GPU-LUT code exists**; a2ui-paint has no textures and no `webgl` feature | **PROBE-GPU-LUT**: upload bgz17 tables, fragment-shader distance parity vs `batch_palette_distance`, on wgpu gles (WebGL2) + WebGPU |
| 3 | W4 (deterministic upscaling) | "Perturbation pyramid = synthesis-lossless reconstruction at arbitrary density, zero NN" | **OVERCLAIM-CORRECTED** | D-PHASE [H]: magnitude-only storage + address-derived deterministic phase is the *design*; the shipped leg is the `CurveRuler` coprime generator. Losslessness is fenced to **synthesis (dither/anti-moiré grade)** — treating it as *content* reconstruction is exactly falsification joint **J2** ("lossless-for-synthesis scope drift"; kill = all codec-savings claims struck) | OGAR **PHASE-1** (file absent), **PERT-RHO** (escalation rate must be reported), **PYR-1** |
| 4 | W2 (decoder complexity wall) | "Certificate-gated skip makes decoder work scale with the viewed region" | **OVERCLAIM-CORRECTED** | The cascade-skip pattern is shipped **for render preselection** on decoded data. Decode-side region scaling needs the unshipped A8 stream framing (region-addressable CTU markers), a codec region index (explicitly deferred), and a reconstruction-error certificate (depth_cert certifies a different quantity). Skip rates (95/60/30) are unmeasured doc claims | A8 + **PROBE-REGION-DECODE**; measure skip rates |
| 5 | W3 (two-tier browser lane) | "wasm SIMD128 + wgpu = CPU decode + GPU raster in browser, today" | **OVERCLAIM-CORRECTED** | The **CPU-wasm foundation is real and CI-parity-verified**; the codec itself is scalar (F32x16 CTU sweep is a documented follow-up) and the GPU tier is absent (a2ui-paint = untested quad demo, WebGL2 unwired). Two-tier is a **roadmap**: (a) SIMD-batched CTU sweep, (b) wasm32-tested wgpu raster with `webgl` + texture upload (a2ui N2), (c) a bridge crate that doesn't exist | codec SIMD sweep (predict.rs:55-58) + a2ui N2 + bridge |
| 6 | W6 (serial entropy) | "rANS + 2-bit alphabet = GPU-parallel entropy decode; 4×4 = workgroup tiling" | **OVERCLAIM-CORRECTED** | Shipped rANS is **single-state scalar**; but each CTU's tag stream is self-contained (count + table in header), so **CTU-granular parallel decode works today on any host incl. wasm**. Interleaved N-state lane rANS (the genuinely GPU/SIMD form) is unimplemented. The 4×4 ergonomics belong to the 3DGS/L4 substrate, NOT this codec (64×64 CTU / 8×8 leaf) — drop that fusion | **PROBE-RANS-INTERLEAVE** (new) |
| 7 | W2 (NN governance) | "Anti-neural discipline achieves reproducibility at browser complexity" | **OVERCLAIM-CORRECTED** | The discipline is a design rule, correctly stated in *qualified* form (standardized NN decode can be bit-exact; our differentiator is **no model artifact to pin/version/govern**). The shipped substrate is *consistent* with it (integer-only codec fragment, exact LUTs) but **no end-to-end decoder exists** (A3 ships encoder direction only; A4/A8 deferred; zero GEMM in the shipped codec) — you cannot claim a decoder property for a decoder that doesn't exist | end-to-end decode path (A4/A8) then complexity measurement |
| 8 | W5 (3DGS web streaming) | "~4 bits/Gaussian + wgpu raster = web-streamable scenes now" | **OVERCLAIM-CORRECTED** | R-10 is paper math with an **assumed** 60/20/15/5 mode mix and an unrun bench gate; IF Plan E confirms ~4 bits/Gaussian, 1M-Gaussian scenes ≈ 500 KB (~100× under stock PLY viewers). Today: mode/rANS coder is for cognitive cells (not splats), the splat codec is absent, and the renderer is deliberately CPU-only | **Plan E bench** (bits/Gaussian on Mip-NeRF 360) |
| 9 | GPU determinism | "Two-algebra rule (XOR/bundle) ports bit-exact to WGSL" | **OVERCLAIM-CORRECTED** | IF a GPU port were built: sign side = u32 xor (order-independent, trivially bit-exact); magnitude side needs 8-bit lanes **packed into u32 with emulated saturation** (WGSL has no u8) and is bit-exact **only with pinned accumulation order** (saturating add is non-associative). Integer LUTs port bit-exactly by construction. Float EWA is never bit-exact across GPUs — keep it outside all bit-exactness claims. D-WHP itself is [H], probes WHP-1..4 unrun, CPU-only parity scope | WHP-1..4 (add a GPU arm to WHP-2 if the port is attempted) |
| 10 | (scoping) | "We don't race ECM/H.267 BD-rate; no browser encode; WebGL2 prefix-scan limits acknowledged" | **FEASIBLE-NOW** | All three exclusions verified against shipped code; positive claims stay scoped to predictable-codebook signals + 3DGS/tensor loads, and even there only to shipped mechanism — no BD-rate, fps, or skip-rate figure is measured yet | — (this row IS the discipline) |

## 4. So what CAN we honestly say vs the industry? (the synthesis)

**Where the industry wall is real and our position is genuinely differentiated:**

1. **The browser foundation exists and theirs doesn't.** A CI-bit-parity-
   verified wasm SIMD128 kernel lane (Hamming-on-Fingerprint, Base17,
   codebook gather, ARX) plus table shapes that are already
   texture-isomorphic is a *foundation* no post-VVC codec effort has. What's
   missing on our side is composition (SIMD CTU sweep, GPU LUT probe, bridge
   crate) — engineering, not research.
2. **Decoder-side determinism-by-construction remains the moat** — in the
   qualified form: no model artifact to pin, version, or audit; integer
   tables + closed-form math. The industry's own CfE record (every response
   NN-flavored) plus the mobile decoder-complexity pushback makes this the
   one axis where the "impractical" judgment *helps* us: they declared the
   tables-only path insufficient for BD-rate racing, and we never entered
   that race (row 10).
3. **The complexity inversion is a hypothesis we can actually test cheaply.**
   The industry wall is "gains cost decoder complexity." Our counter-shape —
   address-arithmetic partitioning (row 1), region-scaled decode (row 4),
   CTU-granular then lane-parallel rANS (row 6) — is fully probe-able on
   shipped substrate within single-PR-sized efforts, unlike ECM tool
   ablations.

**Where "impractical" stays true for us too (do not drift):** general-video
BD-rate competition, browser-side encoding/3DGS fitting, content-grade (vs
dither-grade) pyramid reconstruction until PHASE-1/PERT-RHO/PYR-1 run, and
any float-path GPU bit-exactness claim.

## 5. Probe queue (ordered, with kill conditions)

| Probe | Question | Pass | Kill |
|---|---|---|---|
| PROBE-GPU-LUT | bgz17 256×256 tables as R16Uint/R8Uint; fragment-shader distance == `batch_palette_distance`? | bit-parity on wgpu gles + WebGPU | GPU lane abandoned for LUTs; CPU-wasm only |
| PROBE-MORTON-CTU | flat Morton-addressed 85-slot SoA vs shipped arena tree | ≥2× partition-sweep throughput, code no larger | keep arena; Morton stays address-canon only |
| PROBE-RANS-INTERLEAVE | N-state interleaved rANS, wasm SIMD128 lanes | ≥4× decode throughput vs scalar at equal ratio | CTU-granular parallelism declared sufficient |
| OGAR PHASE-1 / PERT-RHO / PYR-1 | phase determinism; escalation rate; pyramid roundtrip | per OGAR canon | **J2**: D-PHASE stays dither-only; all codec-savings claims struck |
| WHP-1..4 (+GPU arm) | two-algebra pyramid parity | per OGAR canon | magnitude side stays CPU |
| Plan E bench | bits/Gaussian on Mip-NeRF 360 | ≤4 bits | R-10 re-derived; web-streaming claim withdrawn |
| a2ui N2 | wgpu `webgl` feature + texture upload, wasm32-tested | render parity headless vs browser | GPU raster tier deferred; CPU raster only |

## 6. Cross-references

- `pr-x12-h266-h267-standards-landscape.md` — the industry walls, sourced
- `pr-x12-x266-3dgs-spacetime-upscaling.md` — the H.268 lens body (+ §12)
- `PR-X12-docs-audit.md` — the discipline this doc's verdicts ran under
- OGAR `CLAUDE.md` (perturbation, bipolar pyramid, 256×256 tile, 3×4) +
  `docs/DISCOVERY-MAP.md` D-PHASE/D-WHP — the [H] canon and its fences
- lance-graph `le-contract.md` §3 (L4 palette256 tenant), bgz17/bgz-tensor
  table sources; a2ui-rs `a2ui-paint` (the only wgpu in the workspace)

_Last edit: 2026-07-16. Verdicts from workflow run wf_6c6fb99a-cb4 (15 agents,
whole-file receipts; journal retained in session transcript dir)._
