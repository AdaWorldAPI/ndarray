# N-FACET-STRIDE-BAG-FOLD-1: Positional Bagop execution

**Status:** PROPOSED, documentation only. **Date:** 2026-11-05. **Owner:** ndarray. **Dependency:** P0 exclusive holographic Bag plan in lance-graph.

**Requirement:** Every XVERWEIS must lower through the holographic Bagop folding interface and observe the positional discipline of [src/simd_stride.rs](https://github.com/AdaWorldAPI/ndarray/blob/master/src/simd_stride.rs). ndarray owns mechanics, not Excel semantics or PSCustomObject objects.

## Existing facts and boundaries

FACET STRIDE states that a loaded 128-bit facet is sixteen u8 positions read where they sit. Its current exports **lut_u8** and **lut9** are position-addressed LUT transforms. **Neither is an XVERWEIS, ordered first-match fold, join or Bagop interpreter.**

The [facet-stride guard](https://github.com/AdaWorldAPI/ndarray/blob/master/.claude/hooks/facet-stride-guard.sh) enforces R1 no shuffle-as-compute, R2 no scalar loop in non-scalar SIMD backend, R3 no u8/i8 sign confusion. Genuine byte moves and scalar backend execution are distinct, explicitly controlled cases. [simd_masking_ops.rs](https://github.com/AdaWorldAPI/ndarray/blob/master/src/simd_masking_ops.rs) and mask-risc are the appropriate existing operator route. DR512 [P×M law](https://github.com/AdaWorldAPI/ndarray/blob/master/.claude/knowledge/dr512-epistemic-pm-invariant.md) does not authorize interpreting P as user permission or Excel Blank.

    GridCraft / PowerShell expression
      → lance-graph Bagop semantics and DT64 binding
      → admitted Quack / RelAlg / mask-risc
      → ndarray::simd positional masks and folds
      → canonical borrowed Bag representation

ndarray shall not learn customer keys, tenants, sheet cells, search modes or source-specific object shapes. Do not call a direct lookup of a cross-row position a legal Gather without upstream admission.

## Required operator inventory

| Need | Owner | Planned investigation |
|---|---|---|
| Equality over aligned typed lanes | ndarray/mask-risc | Compare + mask available; exact type and state gating |
| Count/existence/masked reduction | ndarray/mask-risc | Existing word/lane operations, accumulator width |
| First/last by source order | Bagop owner P2 | Prove witnessed order selection; not implicit mask index |
| Property projection | Bagop/DT64 | Co-addressed borrowed plane; refuse unadmitted value-derived cross-table hop |
| Global-to-local mapping | DR owner M2 | Route once and reuse, not an application HashMap |
| Strided source | ndarray | Correct scalar oracle and end-to-end SIMD-cost test |

## Falsifier program

Fixture lengths 0, 1, 15, 16, 17, 63, 64, 65, 255, 256, 257 and 65,536. Test noncontiguous strides, tails with poisoned bytes, duplicate equal keys, first/last ordering, reversed and permuted positions, signed/zero/absence boundaries and value revision.

1. Scalar oracle must implement semantics independently from production LUTs and SIMD.
2. Compare scalar, AVX2, AVX-512, NEON and WASM where runnable; unrun targets are UNVERIFIED.
3. Tests must fail when order witness is removed, when multiplicities are collapsed, when a tail is read and when a shuffle-as-lookup is introduced.
4. Record per-shape bind time, mask formation, positional reads, fold time, bytes allocated, cold and warmed performance. The LUT benchmark is not a lookup benchmark.
5. Never replace a missing primitive with a second SIMD engine in GridCraft or report-rs.

## Decision / ownership matrix

| Requirement | Source | Gate |
|---|---|---|
| Positional read law | simd_stride.rs + guard | Implemented discipline, prove kernel adherence |
| Ordered Bag fold | P2 Bagop admission | OPEN, not granted by this document |
| Lifetime, frame layout | DRISA M1/M2 | OPEN owner decisions |
| Cross-table hop | ADMISSION B/Gather law | REFUSE until admitted |
| Consumer façade | P3 GridCraft, P4 lance-graph-java | Not ndarray's execution semantics |

**Decision log 2026-11-05:** plan only. **Next implementation PR:** only a minimal missing kernel established by source census and owner approval, with independent parity, disable controls and measured backend benefit. Do not modify src/simd_stride.rs in this plan PR.
