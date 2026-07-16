# PR-X12 — H.266/VVC · ECM · NNVC · H.267 Public-Standards Landscape

> READ BY: savant-architect, codec-architect, l3-strategist, sentinel-qa,
> product-engineer, truth-architect.
>
> Date: 2026-07-16
> Status: **research doc — externally sourced.** Every load-bearing number
> here carries a source (§8). This doc grounds the PR-X12 "x265/x266" lens
> cluster in the *actual* JVET/MPEG standards trajectory, so the workspace's
> codec claims are benchmarked against the public state of the art rather
> than against a 2013-era HEVC strawman.
>
> **Naming disambiguation (read first):** the workspace doc
> `pr-x12-x266-3dgs-spacetime-upscaling.md` uses "x266" loosely for
> "our next-gen 3DGS scene codec." That is NOT H.266/VVC (finalized 2020,
> already shipping) and NOT the real x266 encoder project (an x265-style
> VVC encoder effort). Where precision matters, say **"PR-X12 3DGS scene
> codec"** for ours and **H.266/VVC**, **ECM**, **NNVC**, **H.267** for the
> standards-track artifacts. This doc is the reference for the latter four.

---

## 1. H.266 / VVC — the shipped baseline (finalized July 2020)

| Fact | Value | Confidence |
|---|---|---|
| Finalized | July 2020 (ITU-T H.266 / ISO/IEC 23090-3) | [G] |
| Bitrate savings vs HEVC | **up to ~50% subjective** at equal quality; objective (PSNR/BD-rate) studies report **~31-40%**, ~36% average for VTM | [G] |
| Decoder complexity vs HEVC | **~1.5-2×** (150-200% depending on configuration) | [G] |
| Encoder complexity vs HEVC | **up to ~10×** (VTM vs HM reference) | [G] |
| Market adoption (2026) | still small; the "H.266/VVC <5% market share" claim in the x266 lens doc's F-3 falsifier remains directionally true | [H] |

Two VVC facts matter directly to the PR-X12 lens set:

1. **Reference Picture Resampling (RPR)** — VVC already ships codec-native
   resolution switching, but as a *2D resample*. This is the weak sibling of
   the PR-X12 3DGS-lens claim (`Basis<T>` swap → re-rasterization from a 3D
   scene model). The lens doc's §1 characterization ("H.266/VVC adds RPR, but
   it's still a 2D resample, not a 3D-scene-aware reconstruction") is
   **confirmed accurate** against public sources.
2. **The complexity asymmetry** (10× encode, 2× decode) is the pattern
   PR-X12's GEMM thesis targets: the encode-side inner loops (ME, transform,
   RDO) are exactly the stages `pr-x12-x265-blasgraph-gemm.md` maps to BLAS.
   VVC's toolset was frozen against ~2018 hardware; the same
   "hardware-floor inversion" argument the GEMM lens makes for HEVC/2013
   applies to VVC/2020 (no AMX, VNNI barely shipping).

## 2. ECM — Enhanced Compression Model (the conventional H.267 track)

JVET's post-VVC exploration software, the likely foundation of H.267:

| Fact | Value | Confidence |
|---|---|---|
| ECM-16.1 BD-rate gain vs VTM-11 | **~27%** (Random Access) | [G] |
| ECM-13..15 typical figures | ~25% RA; **up to ~40% for screen content** | [G] |
| ECM-1.0 (2021 start) | ~12% RA — gains accreted tool-by-tool over ~15 versions | [G] |
| Complexity | high on both sides; industry commentary: "**far too complex**" for practical deployment as-is | [H] |
| Benchmark position | best overall coding performance vs AVM (+16.1%) and learned codecs (DCVC-FM +11%) in low-delay benchmarking | [G] |

**Architectural reading for PR-X12:** ECM is the *tool-accretion* strategy —
more modes, more context models, more per-block branching. It buys ~27% at a
complexity cost the industry itself flags as impractical. PR-X12's bet is
orthogonal: keep the mode taxonomy small (Skip/Merge/Delta/Escape) and win
the *implementation* dimension by reformulating inner loops as GEMM against
the 2026 hardware floor (VNNI/AMX/BF16). These are complementary, not
competing: an ECM-class toolset running on per-pixel scalar loops and a
4-mode codec running on tile GEMM are different points on the
(compression-ratio × watts × latency) surface. PR-X12 should **benchmark
against x265/VTM as ratio baseline but against ECM's complexity trajectory
as the cautionary tale** — every tool added to the codec body must survive
the R-3 LoC envelope precisely because ECM shows where unbounded tool
accretion lands.

## 3. NNVC — Neural Network Video Coding (the learned H.267 track)

| Fact | Value | Confidence |
|---|---|---|
| First common software (NCS-1.0) | two NN in-loop filter tools, **8.71% / 9.44%** RA gain each | [G] |
| Current software | NNVC at **version 7+** of algorithm/software spec | [G] |
| Adopted tool classes | NN intra prediction + NN in-loop filtering | [G] |
| Canonical reference | arXiv:2309.05846 "Designs and Implementations in Neural Network-based Video Coding" | [G] |
| Deployment blocker | decoder-side kMAC/pixel cost; industry (Samsung at ITU 2025 workshop): **low decoder complexity, especially mobile, matters more than bitrate alone** | [H] |

**Architectural reading for PR-X12:** NNVC is the *direct antithesis* of the
workspace's anti-neural rule (`pr-x12-anti-neural-lookup-inversion.md`:
"NNs may train tables; NNs must not sit in the codec hot loop"). JVET is
putting NNs *in* the decode loop; PR-X12 compiles learned structure into
frozen lookup tables (k-means basins, distance LUTs, Gaussian-tail rANS)
so the hot loop stays deterministic table-lookup + GEMM. The public record
now supplies evidence for both directions:

- *For NNVC:* every single CfE response for H.267 incorporated some form of
  NN tool — the field has voted.
- *For the inversion:* the decoder-complexity/power pushback (Samsung et
  al.) is precisely the failure mode the anti-neural rule predicts. The
  reproducibility argument needs stating precisely: a *standardized*
  NN-in-loop codec can be bit-exact (JVET conformance requires fixed-point
  decode), so "NN = nondeterministic" is wrong as a universal claim. The
  PR-X12 advantage is **reproducibility-by-construction and governance**:
  closed-form math with no model artifact to pin, version, or govern —
  versus NN tools whose reproducibility depends on disciplined model
  pinning and whose behavior is not analytically auditable (x266 lens §7).
  Un-standardized AI-upscaler pipelines (DLSS-class, RIFE-class) DO drift
  across model versions; that comparison stands unqualified.

This — stated in the qualified form above — is the sharpest external
differentiator the PR-X12 line has. Keep it, and keep it precise.

## 4. H.267 — the process, the requirement, the dates

| Milestone | Date | Status (2026-07-16) |
|---|---|---|
| Joint CfE issued (ITU-T SG21 + ISO/IEC MPEG WG04) | 2024-2025 | done |
| CfE responses evaluated (40th JVET meeting, Geneva) | **October 2025** | done — best responses ~**30% over VVC**, all with NN tools; "conventional-plus-neural" is the emergent architecture |
| **Final Call for Proposals** | **July 2026 — NOW** | in flight |
| CfP submissions due | November 2026 | upcoming |
| Proposal evaluation / collaborative-phase launch | **January 2027** (key milestone) | upcoming |
| Target finalization | July-October **2028** (some projections ~2030) | projected |
| Meaningful deployment (historical lag) | ~2034-2036 | projected |

**The requirement (JVET, July 2024):** ≥**40% bitrate reduction vs VVC
Main 10** for 4K-and-above at similar subjective quality, while remaining
implementable in real-world encoders/decoders with controlled decoder
complexity and power. Scope explicitly includes HDR, 8K, gaming, and
user-generated content.

**What this gives PR-X12, concretely:**

1. **An external stretch anchor.** R-4's video threshold is anchored to
   x265 ultrafast; the H.267 requirement (VVC −40%) defines the public
   2028 frontier. A Plan G stretch row "ratio vs VTM" places PR-X12 on the
   same axis the standards world uses.
2. **A timing window.** H.267 finalizes ~2028 and deploys ~2034+. The x266
   lens doc's conservative estimate (24-36 months from PR-X12 merge to a
   3DGS scene codec) lands *years* inside the deployment gap — a
   deterministic scene codec does not have to outrun H.267 adoption, only
   NN-upscaler pipelines.
3. **A falsifier update.** The lens doc's F-3 ("wire format ossifies; VVC
   story — <5% share in 2026") stays valid; add F-3b: *if the January 2027
   CfP evaluation converges on a conventional-plus-neural design whose
   decoder complexity is accepted by the mobile vendors, the
   "determinism-as-differentiator" argument weakens for consumer video* —
   though it survives untouched for legal/medical/scientific recording.

## 5. Side activities worth one line each

- **MPEG-AI / VCM (Video Coding for Machines):** coding for machine
  consumption (surveillance, autonomous vehicles) — adjacent to the
  PR-X12 cognitive-cell load, where the "viewer" is also a machine.
- **MPAI-EEV (v4):** end-to-end neural video coding outside JVET — the
  fully-learned pole of the spectrum; no commercial readiness reported.
- **AVM (AOMedia Video Model):** AV1's successor exploration; ECM
  currently leads it by ~16% in JVET-side benchmarking [H — single-source].

## 6. Hardening actions applied to the PR-X12 plan set (2026-07-16)

| # | Action | Where |
|---|---|---|
| H-1 | Naming disambiguation — "PR-X12 3DGS scene codec" vs H.266/H.267 | this doc header; x266 lens addendum |
| H-2 | ECM complexity trajectory adopted as the R-3 LoC-envelope cautionary anchor | §2 |
| H-3 | NNVC named as the explicit antithesis of the anti-neural rule; determinism differentiator sharpened with Samsung complexity evidence | §3 |
| H-4 | H.267 requirement (VVC −40% @ 4K+) recorded as external stretch anchor for Plan G | §4 |
| H-5 | Standards-watch calendar: **Jul 2026 CfP (now) → Nov 2026 submissions → Jan 2027 evaluation → 2028 finalization** | §4 table |
| H-6 | F-3b falsifier added to the x266 lens (conventional-plus-neural acceptance risk) | §4; x266 lens addendum |
| H-7 | Audit Tier-2 corrections applied to the canon docs in the same PR (fabricated symbols marked, uncalibrated numbers tagged, PDE-bug claim withdrawn) | `PR-X12-docs-audit.md` addendum |

**Standing watch rule:** any session citing "H.267" or "beyond-VVC" numbers
re-checks §4's table against the JVET document registry if more than ~6
months have passed since the last-edit date below. The CfE→CfP→evaluation
cadence moves roughly twice a year (JVET meets quarterly).

## 7. Reading list (arXiv / primary)

- arXiv:2309.05846 — *Designs and Implementations in Neural Network-based
  Video Coding* (the NNVC reference)
- arXiv:2404.07872 — *Video Compression Beyond VVC: Quantitative Analysis
  of Intra Coding Tools in ECM* (per-tool gain attribution)
- arXiv:2503.18679 — *Merge Mode for Template-based Intra Mode Derivation
  (TIMD) in ECM* (example of ECM tool-accretion granularity)
- arXiv:2408.05042 — *Benchmarking Conventional and Learned Video Codecs
  with a Low-Delay Configuration* (ECM vs AVM vs DCVC-FM numbers)
- arXiv:2005.10801 — *Complexity Analysis of Next-Generation VVC Encoding
  and Decoding* (the 10×/2× VVC complexity source)
- arXiv:2310.13093 — *Video Quality Assessment and Coding Complexity of
  the VVC Standard* (the ~31-40% objective-gain source)
- JVET-AA0006 — *AHG report: ECM software development* (complexity
  reporting home)
- Kerbl et al. SIGGRAPH 2023 (3DGS), Zwicker et al. 2001 (EWA) — carried
  over from the x266 lens doc's list; the scene-codec math is unchanged.

## 8. Sources (web, retrieved 2026-07-16)

- Ofinno, "The Next-Generation Video Coding Race Heats Up" (40th JVET
  meeting readout: CfE evaluation Oct 2025, ~30%-over-VVC best responses,
  all-NN-flavored, Jan 2027 CfP milestone, ~2030 collaborative completion,
  decoder complexity/power requirement)
- Streaming Learning Center, "AI Video Compression Standards: Who's Doing
  What and When" (ECM-15 ~25% RA / 40% SC; NNVC v7; H.267 finalization
  Jul-Oct 2028; deployment 2034-2036; MPEG-AI/VCM/MPAI-EEV; Samsung
  decoder-complexity position)
- Streaming Media, "H.267: A Codec for (One Possible) Future" + Rethink,
  "H.267 — VVC's heir officially proposed" (requirement: ≥40% vs VVC
  Main 10 at 4K+, July 2024 JVET timeline document)
- themoonlight.io review of arXiv:2509.25668 (ECM-16.1 = 27.06% BD-rate
  vs VTM-11.0)
- kpubs/KIBME, Choi, "Neural Network based Video Coding in JVET"
  (NCS-1.0 filter gains 8.71%/9.44% RA)
- Elecard, "A review of the VVC codec" + Spin Digital VVC page (VVC
  decoder 150-200% of HEVC; up-to-50% subjective savings; ~36% objective)
- Wikipedia, "Versatile Video Coding" (finalization date, profile facts)

_Last edit: 2026-07-16._
