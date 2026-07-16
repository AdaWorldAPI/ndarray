# PR-X12 — x266 / Next-Gen Codec via 3DGS Space-Time Upscaling

> **Naming + standards anchor (2026-07-16):** "x266" in this doc means the
> **PR-X12 3DGS scene codec** — NOT H.266/VVC (finalized 2020, shipping) and
> not the real H.267 (JVET's beyond-VVC standard, CfP running July 2026,
> target ~2028). For the grounded public landscape — VVC facts, ECM, NNVC,
> H.267 requirement + dates — read
> `pr-x12-h266-h267-standards-landscape.md` alongside this doc, and see
> §12 below for the dated reality-check addendum.
>
> Date: 2026-05-22
> Status: **speculative perspective doc** — explores what becomes possible when the codec substrate (PR-X12) is extended one step beyond HEVC compatibility, into territory that subsumes both AI-frame-interpolation and AI-super-resolution as codec-native deterministic operations. Companion to `pr-x12-x265-blasgraph-gemm.md`.
>
> Status caveat: nothing in this document is committed as PR-X12 scope. It's the future shape that PR-X12's substrate makes obvious. Plan E + Plan G prerequisites must land first.
>
> Premise: in-loop reference-frame reconstruction in HEVC is a 2D pixel-grid render. In a 3DGS-augmented codec, it's a re-rasterization from a 3D Gaussian scene model. Same trait (`Basis<T>`), different impl. The decoder becomes responsible for (resolution, frame-rate) at playback time, not (encoder, capture).

---

## 0. One-sentence thesis

**HEVC's in-loop filter is a `Basis<T>::apply` call whose output happens to be a 2D pixel array.** Replace that with an EWA-splat `Basis<T>::apply` whose output is a 2D rasterization of a 3D Gaussian scene at a parameter-controlled (resolution, time), and *the same encoder produces a free space-time upscalable bitstream* — no AI frame interpolation, no neural super-resolution, just deterministic re-rasterization from a scene model that already lives in the wire format.

---

## 1. The capability gap PR-X12 closes

Current state-of-the-art for high-quality playback at non-native rate:

- **Frame interpolation:** DAIN, RIFE, FILM — learned optical flow models that hallucinate intermediate frames. Per-frame inference cost ~30-100 ms on a GPU. Non-deterministic across model versions. No codec integration.
- **Super-resolution:** ESRGAN, Real-ESRGAN, DLSS-FG — learned upscalers. Per-frame cost similar. Same non-determinism and integration gap.
- **Codec-native upscaling:** lanczos/bicubic — deterministic but low-quality; H.266/VVC adds Reference Picture Resampling, but it's still a 2D resample, not a 3D-scene-aware reconstruction.

**The PR-X12 substrate exposes a third option:** ship a 3D scene model in the bitstream, and let the decoder render at arbitrary (res, fps). The 3D scene model is *the reference frame*, not a precomputed 2D image. This isn't novel as research (3DGS papers from 2023-2025) — it's novel as a *codec primitive*, because no codec has been able to express "the in-loop filter is a basis swap, swap it" cleanly. PR-X12 can.

---

## 2. 3DGS as a `Basis<T>` impl — the trait shape

Recall (R-1, M:E-A): `LinearReduce<T>` decomposes into `Basis<T>` + reduction. The basis is data; the reduction is the inner loop. The codec's transform path calls `basis.apply(src, dst)`; Plan E's EWA splat rasterizer calls the same.

```rust
pub trait Basis<T: Copy> {
    /// Apply this basis to a source array, writing into a destination.
    /// For DCT: src = pixel block, dst = coefficient block.
    /// For EWA: src = 3DGS scene params, dst = rasterized 2D pixel frame.
    fn apply<R: Reducer<T>>(
        &self,
        src: &[T],
        dst: &mut [T],
        params: &Self::Params,  // basis-specific: viewport, time, etc.
        reducer: R,
    );

    type Params;
}

// Existing (R-1, R-5):
impl<const N: usize> Basis<i16> for DctIIBasis<N> {
    type Params = ();
    fn apply<R: Reducer<i16>>(&self, src: &[i16], dst: &mut [i16], _: &(), r: R) {
        // batched DCT-II via bf16_tile_gemm at N >= 64
    }
}

// Future (Plan E, then x266):
impl Basis<f16> for EwaSplatBasis {
    type Params = ViewportTime;  // camera pose + timestamp
    fn apply<R: Reducer<f16>>(
        &self,
        gaussians: &[GaussianRecord],   // 3DGS scene (5-7 KB per cell, see §6)
        out_frame: &mut [f16],          // 2D pixel buffer at target res
        vp: &ViewportTime,              // (W, H, t) — chosen by decoder
        r: R,
    ) {
        // Rasterize 3DGS scene at (W, H, t)
        // Same per-tile GEMM pattern as ndarray-image's existing EWA path
    }
}

struct ViewportTime {
    width: u32,
    height: u32,
    time_ms: u64,           // frame timestamp; 3DGS scene is continuous in t
    camera_pose: Mat3x4f,   // identity for monoscopic; non-trivial for VR
}
```

**The crucial property:** the codec body (`ndarray-codec`) doesn't know whether it's calling `DctIIBasis` or `EwaSplatBasis`. It dispatches via the trait. The bitstream header (`Ctu` header bits, see M:E-J) selects which basis is in play.

This is exactly the kind of substrate flexibility R-1 was designed to provide. Without R-1, this paragraph is fantasy; with R-1, it's a 6-week engineering effort to land Plan E and wire the trait.

---

## 3. The encoder problem: fitting a Gaussian scene model to a clip

Encoding a video clip with a 3DGS scene anchor means: given N input frames at known camera pose (or estimated pose), find a 3DGS scene S such that rendering S at each frame's (pose, time) reproduces the input frames to within a target PSNR.

This is a standard 3DGS fitting problem (Kerbl et al. 2023, Mip-Splatting 2024). The relevant fact for PR-X12:

```text
Input: N frames @ (1080p, 24 fps) for 10 seconds = 240 frames
Output: scene S = ~100K-500K anisotropic Gaussians
        ~32 bytes per Gaussian (position 3×f16, scale 3×f16,
        rotation quaternion 4×f16, color 3×f16, opacity 1×f16 = 28 B)
        + quantized SH coefficients for view-dependent color (~8-16 B)
        Total: ~40-50 bytes per Gaussian × 200K = 8-10 MB per scene anchor

Compare to:
        240 frames × 3 MB (Bbb 1080p I-frame at HEVC Q=20) = ~720 MB raw I-frame
        HEVC encoded @ ~5 Mbps = ~6.3 MB for the whole clip
```

So the scene-anchor encoding is the same order of magnitude as standard HEVC encoding *for one anchor period*. The win comes from:

1. **Re-rasterization is free** — render at 4K, 8K, 60 fps, 120 fps, all from the same 8 MB scene model
2. **Anchor periods stretch** — if motion is low, one anchor lasts 10+ seconds; HEVC has to re-encode I-frames every ~2 sec for random-access seek
3. **View interpolation** — for VR/stereo, render two views from one scene; HEVC needs to encode two streams

**The encoder pipeline:**

```text
Anchor frame n:
    1. Estimate camera pose from frame n+0 through n+anchor_period (~240 frames)
    2. Initialize Gaussian cloud from frame n's depth estimate
    3. Optimize cloud via gradient descent: minimize Σ |render(S, t_k) - frame_k|²
       (This is k-means-like; uses cam_pq infrastructure)
    4. Quantize to scene-anchor format (see §5)

Per-frame delta n+1, n+2, ...:
    Standard HEVC inter-prediction against the 3DGS-rendered ref frame.
    The 3DGS-rendered ref is computed by the decoder too, so the delta is
    in the same algebraic space as HEVC.
```

The clever part: **the decoder's reference frame for inter-prediction is the 3DGS render at the previous frame's (pose, t)**. So the per-frame delta is small — most motion is already captured in the scene model.

---

## 4. The decoder problem: rasterizing at arbitrary (res, fps)

The decoder receives:

- Scene anchor: scene S (8-10 MB) at clip start, then refreshes every ~250 frames
- Per-frame deltas: standard HEVC-like residual, against the 3DGS-rendered ref

At playback time:

```text
For each output frame at (W_target, H_target, t_target):
    1. Render scene S at (W_target, H_target, t_target) via EwaSplatBasis::apply
       Output: ref_frame in pixel buffer
    2. Decode per-frame delta against ref_frame
    3. Apply standard HEVC in-loop filtering (deblock + SAO)
    4. Emit pixel buffer
```

**Key observation:** step 1 is parametrised in (W, H, t). The encoder shipped a 1080p @ 24 fps clip; the decoder renders at 4K @ 60 fps by choosing different (W_target, H_target, t_target) tuples. The scene model is continuous in (W, H, t); the rasterizer interpolates.

This is **codec-native space-time upscaling**, deterministic across decoder implementations because the math (EWA splat rasterization) is well-specified. Same scene model, same camera pose, same t → same pixels. No model versioning. No "frame interpolation v3 hallucinates differently than v2."

**Cost per frame:** EWA splat raster of 200K Gaussians at 4K → ~5-15 ms on a modern GPU; ~50-100 ms on CPU. Tight for real-time decode at 60 fps on CPU; comfortable at 24-30 fps. R-11 latency assertion applies — Plan G's decoder lane must hit real-time at the target playback rate.

---

## 5. Wire format: scene-anchor frames + per-frame deltas

Building on M:E-J's 16-bit header layout (header_kind ∈ {Skip, Merge, Delta, Escape}), x266 needs one new header_kind: **SceneAnchor**.

```text
HEVC-compatible PR-X12 header (16 bits, R-2):
    bits 0-1:   header_kind {Skip, Merge, Delta, Escape}
    bits 2-13:  basin_index (12 bits, M:E-J)
    bit  14:    CONSUMER-TYPED (semantic per frame-header `ConsumerProfile`;
                cognitive: Pearl-rung high bit; video: reserved=0;
                splat: LOD-cascade-source flag; gradient: worker-shard parity)
    bit  15:    UNIVERSAL "has inter-tier reference" (A3-inter); identical
                across all four consumers
    NOTE: leaf-size (8/16/32/64) is encoded structurally via `Ctu<const N>`
    (M:E-G) at the type level, not via header bits.

x266 extension (NOT in PR-X12 scope, future):
    bits 0-1:   header_kind, now 4 variants
                  00 = Skip (HEVC-compatible)
                  01 = Merge (HEVC-compatible)
                  10 = Delta (HEVC-compatible)
                  11 = Escape OR SceneAnchor (escape bit at byte boundary disambiguates)
    bits 2-15:  basin_index (12 bits) + scene_anchor_id (2 bits) when in anchor mode
```

**Anchor frame payload** (after the 16-bit header):

```text
SceneAnchorFrame:
    scene_id: u8                        // which anchor in the GOP
    num_gaussians: u24                  // typically 50K - 500K
    cam_pose_keyframes: u8              // number of pose anchors
    [GaussianRecord; N]:                // 40-50 bytes each, quantized
        position: [u16; 3]              // q15 fixed-point per axis
        scale_log: [u8; 3]              // log-quantized
        rot_quat: [u8; 4]               // quantized to 8-bit
        sh_coeffs: [u8; 27]             // 9 coefs per channel × 3 channels, q7
        opacity: u8
    pose_keyframes: [(t_ms: u32, Mat3x4f); cam_pose_keyframes]
```

Per-frame deltas after the anchor are standard HEVC-like, with one difference: the reference frame is derived by rasterizing the anchor scene at the frame's (pose, t), not by decoding a prior I-frame.

**Bitstream compatibility:** an HEVC-spec decoder that doesn't understand `SceneAnchor` headers can fall back to displaying the inter-frame deltas as zero-padded macroblocks (visibly broken, but won't crash). A PR-X12 decoder with EwaSplatBasis loaded plays back at native quality.

---

## 6. Bandwidth math: when does this beat HEVC?

Rough rule (calibrated against published 3DGS papers):

```text
Clip:   10 seconds, 1080p, 30 fps, modest motion (e.g. Bbb sample)

HEVC reference (5 Mbps avg, hardware encoded):
    bytes = 5 × 10⁶ × 10 / 8 = 6.25 MB

PR-X12 + 3DGS anchor (single anchor for the clip):
    anchor: 200K Gaussians × 40 B = 8 MB
    deltas: ~300 frames × 1 KB avg = 300 KB
    Total: 8.3 MB

→ HEVC wins by ~25% for native (1080p, 30 fps) playback.

BUT for 4K @ 60 fps playback:
    HEVC: re-encode at 4K/60fps target = 4 (res) × 2 (fps) × 6.25 = 50 MB
            (4× pixel scaling × 2× framerate scaling × 6.25 MB native bitrate;
             or super-res upscaling at decode = 6.25 MB + neural inference)
    PR-X12 + 3DGS: same 8.3 MB
            decoder rasterizes at (4K, 60 fps); the math is in the scene

→ PR-X12 wins by ~6× for high-resolution playback,
   AND playback is deterministic (no neural model versioning).
```

**Where the crossover sits:** PR-X12 + 3DGS becomes a win when the playback target (W × H × fps) exceeds the encode target by ~1.3× (the point at which HEVC's re-encoded size crosses the fixed 8.3 MB PR-X12 budget). At 1× (native), HEVC is a hair cheaper. At 8× pixel-bandwidth (4K@60 from 1080p@30), PR-X12 dominates by ~6×.

This matches the intuition that **3DGS is a scene model**, not a frame model — its compression ratio improves with resolution, while HEVC's degrades.

---

## 7. The "free upscaling" insight — why this isn't AI

Critics will read §6 and say "this is just AI upscaling rebranded." The distinction is sharper than it sounds.

**AI upscaling** (DLSS, ESRGAN, Real-ESRGAN, RIFE, DAIN, FILM):
- Input: 2D pixel array at low res
- Model: learned NN with millions of parameters; non-deterministic across versions
- Output: 2D pixel array at high res, with hallucinated detail
- Failure mode: hallucinates wrong detail (e.g. wrong text on a sign)
- Latency: per-frame ~30-100 ms on a GPU
- Codec integration: zero

**PR-X12 + 3DGS rasterization** (this doc):
- Input: 3D Gaussian scene + camera pose
- Model: closed-form EWA splat formula (Zwicker et al. 2001, refined in 3DGS papers)
- Output: 2D pixel array at any res, computed deterministically
- Failure mode: misses detail that wasn't in the scene model — but never hallucinates
- Latency: per-frame ~5-15 ms on a GPU; ~50-100 ms on CPU
- Codec integration: full, basis trait dispatch

The 3DGS scene captures the actual 3D geometry of what was in front of the camera. Rasterizing at higher resolution doesn't invent detail — it *samples the 3D scene more finely*. If the encoder couldn't fit a detail (e.g. the text on a small sign), the decoder can't recover it. That's a **failure of completeness**, not a failure of fidelity. Compare to AI upscaling, which has both modes and can't tell you which is happening.

For high-stakes video (legal evidence, medical imaging, scientific recording), this distinction matters. PR-X12 + 3DGS is **legally and scientifically defensible** in a way no learned upscaler can be.

---

## 8. PR-X12 prerequisites

Nothing in this doc is in PR-X12 scope. What it requires from PR-X12:

| Requirement | Source | Status |
|---|---|---|
| `Basis<T>` trait with parametric `apply` | R-1, M:E-A | **canon-fixed** (R-1 trait shape committed); **implementation** scheduled in Plan A4 |
| EWA splat rasterizer as `Basis<f16>` impl | Plan E | scheduled |
| Codec body decoupled from specific basis | M:H-NEW-2 LoC envelope | enforced via R-3 audit rule (doc commitment; CI check pending) |
| Header byte stable across basis swaps | R-2, M:E-J bits 0-1 | **canon-fixed** (R-2 commits bits 0-1 = `header_kind`); wire-format implementation in Plan A8 |
| Plan G video lane validates per-arch latency | R-4, R-11 | scheduled |
| Federated codebook policy for scene anchors | R-13 | **canon-fixed** (R-13 commits Option A: per-shard codebook for Plan F v1); implementation in Plan F |

**"Canon-fixed"** = the resolution doc commits the design; **"scheduled"** = the implementation has a named plan card. None of the above have shipping code today.

The path to x266-like capability is:

1. Land PR-X12 (HEVC-compatible, no 3DGS). Plan A4 → Plan H.
2. Plan E ships EWA splat as `Basis<f16>`.
3. New crate `ndarray-codec-scene` (or extension within `ndarray-codec`) adds `SceneAnchor` header kind + scene encoder/decoder pipelines.
4. Bench against AI upscaling pipelines (RIFE / Real-ESRGAN) on quality and latency.
5. Standardise the wire format extension (separate spec, not HEVC-compatible).

Conservative estimate: **24-36 months from PR-X12 merge**, assuming Plan E lands on schedule and 3DGS encoder math is taken from existing research (no novel algorithms required).

---

## 9. Falsifiers

What kills this path? Be specific:

**F-1: Encoder math doesn't converge for general video.** 3DGS papers focus on static scenes with controlled camera motion. Real video has occlusion, transparency, fast motion. If 3DGS fitting can't hit PSNR ≥ 35 dB on motion-heavy clips (e.g. sports footage) within reasonable encode time, the substrate is decorative. **Mitigation:** restrict scope to slow-camera-motion content (talking heads, drone footage, security cameras); HEVC stays the fallback for sports.

**F-2: Decoder rasterization too slow.** If EwaSplatBasis::apply can't hit real-time at 4K @ 60 fps on a 2026-class CPU, the codec is server-side only. **Mitigation:** PR-X12's R-11 latency assertion catches this in CI; if the CPU path fails, the codec emits a GPU-required flag in the bitstream.

**F-3: Wire format ossifies.** If HEVC stays dominant and x266 adoption is slow (the H.266/VVC story so far — 2020 release, still <5% market share in 2026), the SceneAnchor extension never sees a standards body. **Mitigation:** ship it as a non-standard extension first, in an open-source decoder; let market traction force standardisation.

**F-4: Patents.** 3DGS-as-codec-primitive may sit in a patent thicket. Some 3DGS rendering optimisations (tile binning, depth sorting) are likely patented. **Mitigation:** the basis trait is general; if Gaussian splats are patented, swap to another basis (TensoRF, NeRF compaction, point cloud + bilinear) — same architecture, different math.

None of these falsifiers invalidate PR-X12 itself. They only constrain the post-PR-X12 path.

---

## 10. Why this lens matters now, for PR-X12 scoping

The temptation in scoping PR-X12 is to optimise for HEVC compatibility only — strip out anything that doesn't directly serve the x265-replacement story. **The basis trait (R-1) and the EWA-splat schedule (Plan E) survive that pruning** because they were independently motivated. This doc makes the case that they were also the right call by another measure: they're the substrate that lets x266 happen at all.

Concretely:

- **Do not** weaken `Basis<T>` to be DCT-only "for now." The generality has zero LoC cost (the trait is the same) and unlocks 3DGS later.
- **Do** keep Plan E on the roadmap even if Plan H/codec-fast-path pressure tries to defer it. EWA splat is the first non-DCT basis and validates the trait shape.
- **Do** keep the codec body free of basis-specific code. M:H-NEW-2's "ratchet on codec LoC at the basis boundary" already enforces this; the x266 lens is why it matters.

---

## 11. Cross-references

- **Substrate dependencies:** R-1, R-2, R-3, R-11, R-13 in `pr-x12-canon-resolutions-delta.md`
- **Basis trait architecture:** §M:E-A in `pr-x12-substrate-merged-canon.md`
- **EWA splat planning:** Plan E in `pr-arc-inventory.md`
- **Codec foundation:** `pr-x12-codec-x265-design.md`
- **GEMM lens:** `pr-x12-x265-blasgraph-gemm.md`
- **Bandwidth comparison reading list:** 3DGS (Kerbl et al. SIGGRAPH 2023), Mip-Splatting (Yu et al. 2024), 4DGS (Wu et al. 2024)

---

## 12. Standards reality-check addendum (2026-07-16)

Written against the live JVET landscape; sources and full detail in
`pr-x12-h266-h267-standards-landscape.md`.

**What the standards world did since this doc was drafted:**

- The H.267 **Call for Evidence was evaluated October 2025** (40th JVET,
  Geneva). Best responses reached ~**30% bitrate reduction over VVC** — and
  *every* response incorporated NN tools ("conventional-plus-neural" is the
  emergent architecture). The final **Call for Proposals is running now
  (July 2026)**, submissions November 2026, evaluation **January 2027**,
  finalization targeted **2028** (deployment historically lags to ~2034+).
- The H.267 requirement is ≥**40% bitrate reduction vs VVC Main 10 at
  4K-and-above**, with explicit decoder-complexity/power constraints —
  mobile vendors (Samsung at the ITU 2025 workshop) now weight decoder
  complexity above raw bitrate.
- ECM-16.1 sits at ~**27% BD-rate over VTM-11** (RA), up to ~40% for screen
  content — at a complexity the industry itself calls impractical. NNVC is
  at software v7 with NN in-loop filters worth ~9% RA each.

**What this changes in this doc — and what it doesn't:**

1. §1's characterization of the field **holds**: codec-native upscaling in
   the standards track is still 2D resampling (VVC RPR); the learned
   track (NNVC, DLSS-class) is non-deterministic in exactly the way §7
   argues against. No CfE response ships a scene-model reference frame.
   The 3DGS-as-`Basis<T>` bet remains unoccupied territory.
2. §7's determinism argument is now **stronger**, not weaker: JVET putting
   NNs into the decode loop makes "same scene, same pose, same t → same
   pixels, forever" a differentiator no H.267 candidate can match. For
   legal/medical/scientific video this is a moat.
3. §8's "24-36 months from PR-X12 merge" estimate lands comfortably inside
   the H.267 deployment gap (finalize 2028, deploy 2034+). The scene codec
   does not race H.267; it races NN-upscaler pipelines.
4. **F-3 update:** the "H.266/VVC <5% market share" premise remains
   directionally true in 2026. **F-3b (new falsifier):** if the January
   2027 CfP evaluation converges on a conventional-plus-neural design whose
   decoder complexity mobile vendors accept, the determinism differentiator
   weakens for consumer video (it survives for evidentiary/medical/
   scientific recording regardless).
5. **Watch calendar:** re-check the landscape doc after the November 2026
   CfP submissions and the January 2027 evaluation — those two events fix
   H.267's architecture and decide F-3b.

_Last edit: 2026-07-16 (addendum §12 + naming anchor; body unchanged from 2026-05-22)._
_Status: speculative — explores what's possible after PR-X12 lands; not in PR-X12 scope._
