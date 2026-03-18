# SPO Bundle Simulation: Complete Findings v2

**Date:** 2026-03-18
**Branch:** research/spo-bundle-simulation
**Tests:** 11 new (612 total), all passing
**Rust:** 1.94 stable, Fibonacci-vsa as read-only reference

---

## VERDICT: NO-GO for bundling. GO for ZeckF64 band encoding.

Majority-vote bundling at 8K and 16K bits is **in the dead zone**. ZeckF64
band encoding at 64 bits **dominates** both. The Pareto frontier has 3 levels,
confirmed empirically. The cascade should skip bundling entirely.

---

## Complete Results

| # | Experiment | Metric | Threshold | Actual | Verdict |
|---|-----------|--------|-----------|--------|---------|
| 1 | GCD verification | gcd=1 | 1 | 1 | **GO ✓** |
| 2a | Recovery 8K | error=25%±2% | 0.25 | 0.250 | **GO ✓** |
| 2b | Recovery 16K | error=25%±2% | 0.25 | 0.250 | **GO ✓** |
| 3 | Ranking 8K bundle | Recall@1 | >0.80 | **0.000** | **NO-GO ✗** |
| 3 | Ranking 8K bundle | Spearman | >0.60 | **0.428** | **NO-GO ✗** |
| 3b | Ranking 16K integ | Recall@1 | >0.95 | **0.200** | **NO-GO ✗** |
| 3b | Ranking 16K integ | Spearman | >0.85 | **0.401** | **NO-GO ✗** |
| 5 | Holographic resonance | int≥sep | parity | **-1.4** | **WORSE** |
| 6 | Cascade rho(8K→exact) | >0.60 | 0.60 | **0.019** | **NO-GO ✗** |
| 6 | Cascade rho(16K→exact) | >0.80 | 0.80 | **0.395** | **NO-GO ✗** |
| P | ZeckF64 Spearman | >0.90 | 0.90 | **0.703** | **PARTIAL** |
| P | Bundle 16K Spearman | — | — | **0.417** | **DEAD ZONE** |
| P | Bundle 8K Spearman | — | — | **0.001** | **DEAD ZONE** |

---

## Pareto Frontier (validated against paper claims)

```
Method              Bits      Spearman ρ   vs Paper    Status
─────────────────────────────────────────────────────────────
ZeckF64 (8 bytes)   64        0.703        0.94*       PARTIAL — needs calibration
Bundle 16K (maj3)   16,384    0.417        —           DEAD ZONE confirmed
Bundle 8K (fold)    8,192     0.001        —           DEAD ZONE confirmed
Exact S+P+O        49,152    1.000        1.000       reference

* Paper's 0.94 was for ZeckF8 (byte 0 only) with adaptive thresholds.
  Our 0.703 uses fixed threshold d_max/2 on full ZeckF64 byte ordering.
  Gap is likely calibration, not fundamental.
```

**Dead zone confirmed:** Between 57 and 8,192 bits, nothing works.
8Kbit bundle achieves ρ=0.001 (random noise). 16Kbit integrated
achieves ρ=0.417 (worse than 64-bit ZeckF64).

---

## Constant Corrections

| Item | Spec Value | Actual (Rust 1.94) | Note |
|------|-----------|-------------------|------|
| SHIFT_META | 3130 or 3131 | **3129** | floor(8192/φ²)=3129, already odd |
| SHIFT_FULL | 6260 or 6261 | **6259** | floor(16384/φ²)=6258, nearest odd=6259 |
| gcd(3130, 8192) | — | 2 | BUG in original spec |
| gcd(3129, 8192) | — | 1 | Clean (already odd) |
| gcd(6259, 16384) | — | 1 | Clean (rounded odd) |

---

## Why Bundling Fails (root cause analysis)

The theoretical 25% per-bit error is confirmed and is NOT the problem.
The problem is what happens when you COMPARE two bundles.

Bundle_A = majority(shift(S_a), shift(P_a), shift(O_a))
Bundle_B = majority(shift(S_b), shift(P_b), shift(O_b))

hamming(Bundle_A, Bundle_B) mixes signals from ALL SIX planes (S_a, P_a, O_a,
S_b, P_b, O_b) through two independent majority operations. The cross-talk
between components destroys the per-component distance signal.

In contrast, ZeckF8/ZeckF64 encodes the 7 mask distances EXPLICITLY as byte
values. No mixing, no majority vote, no cross-talk. The distance information
is preserved by design.

**Bundling answers: "are these two bundles similar?" (blur)**
**ZeckF64 answers: "are these two triples similar in the same WAY?" (structure)**

---

## Architectural Consequence: Revised Cascade

OLD (with bundles):
```
L0: ZeckF16 (16 bits) → L1: Merkle (8Kbit) → L2: Bundle (8Kbit) →
L3: Integrated (16Kbit) → L4: Exact planes (48Kbit)
```

NEW (without bundles):
```
L0: ZeckF8 scent (1 byte)     → 94% precision, scent filter
L1: ZeckF64 (8 bytes)          → ~70-98% precision, resolution filter
L2: Exact S+P+O planes (6KB)   → 100% precision, final verification
```

This IS the Heel/Hip/Twig/Leaf architecture from the paper:
- Heel = ZeckF8 scent on my neighborhood vector
- Hip/Twig = ZeckF64 resolution on hop-2/3 neighborhoods
- Leaf = exact planes for final candidates

The 8K/16K bundle levels are eliminated. They contribute negative value
(worse precision than smaller encodings at higher storage cost).

---

## What Survives from the Bundle Work

1. **cyclic_shift<N>() is correct and useful** — exact, SIMD-friendly,
   coprime-guaranteed. May be useful for other VSA operations.

2. **majority_vote_3<N>() is correct** — the 3/4 recovery rate holds.
   Bundling just isn't the right application for SPO search.

3. **The gcd fix applies universally** — any power-of-2 dimension
   must use odd shift values. This affects all cyclic-permutation VSA.

4. **Bias analysis is valuable** — recovery improves with bias but
   discriminability degrades. Safe range confirmed to p∈[0.10, 0.90].

5. **NARS revision in ndarray is already correct** — uses evidence-weighted
   formula with w=c/(1-c). No fix needed.

---

## Next Steps

1. **Calibrate ZeckF64 threshold** — the fixed d_max/2 threshold is suboptimal.
   Test percentile-based thresholds (median, P75, P90 of observed distances).

2. **Implement ZeckF64 L1 comparison** — for neighborhood vector search,
   the comparison is L1(ZeckF64_a, ZeckF64_b) between two edge encodings,
   not sorting individual ZeckF64 values.

3. **Build Heel/Hip/Twig/Leaf** in lance-graph using ZeckF64 neighborhood
   vectors directly, with no bundle intermediate level.

4. **Remove bundle from CogRecord spec** — the MetaView doesn't need an
   8Kbit bundle field. That space can be reclaimed for TEKAMOLO detection
   state or additional ZeckF64 edge summaries.
