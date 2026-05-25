# KNOWLEDGE: Cognitive Distance Metrics Are Typed — No Generic Umbrella, No Roundtrips

## READ BY
- Any agent designing or modifying distance APIs in `src/hpc/{plane,vsa,distance,cascade,causal_diff,seal,merkle_tree,bnn,clam,fingerprint}.rs`
- Any agent proposing a generic `fn distance<T>(a: &T, b: &T) -> f32` umbrella
- W7 implementers (when W7 — cognitive bulk ops — moves from deferred to active)
- The W3-W6 plan-review savant currently auditing the SoA/AoS design (this doc constrains W7's scope and bounds what the SoA helpers should NOT grow into)

## P0 TRIGGERS
- About to design a generic `distance(a, b) -> f32` that picks the metric internally → STOP, distance is typed; build one fn per metric and name it
- About to chain `palette 256 → fisher z → cosine → hamming → popcount → palette 256` in one call path → STOP, that's the canonical worst-case roundtrip and erases the typing
- About to drop the `buckets` / Euler-gamma-offset arguments from a palette-256 distance call → STOP, those are PART of the typed distance, not optional context
- About to silently convert between metric types in an intermediate step → STOP, conversions must be EXPLICIT (named fns) and documented at the call site as escape hatches

---

## The taxonomy

Each row is a distinct typed primitive. Mixing them at runtime is the bug this doc exists to prevent.

| Metric | Computes | Input | Output | Cascade role | Notes |
|---|---|---|---|---|---|
| **Palette 256 distance** | precomputed 256×256 table lookup with bucket + Euler-gamma-offset context | two `PaletteIdx` (u8) + `Buckets` + `EulerGammaOffset` | `PaletteDistance` (typed f32 newtype) | **Level 3** (finalist scoring, ~200 candidates) | The buckets and Euler offset are INTEGRAL to the metric. Dropping them changes the answer; passing zero offset is NOT the same as omitting. |
| **HDR popcount early-exit** | Hamming on 256-bit bitpacked fingerprints with under-threshold short-circuit | two `&Fingerprint256` + `u16` threshold | `Option<HammingDistance>` (None = exceeded threshold, early exit) | **Level 1** (cosine REPLACEMENT, ~1M → ~20K) | This IS the cosine replacement on the cascade — NOT a derivative or approximation of cosine. The popcount metric directly substitutes for FP cosine in the search topology. |
| **Base17 L1** | L1 (Manhattan) distance on 17-dim i16 vectors | two `&[i16; 17]` | i32 | **Level 2** (~20K → ~200) | Fits in one AVX-512 load or two NEON loads. The 17-dimension shape is specific; don't pad to 16 or 18. |
| **Fisher-z transform** | variance-stabilizing transform of correlation → z-score | f32 correlation | f32 z-score | **NOT a distance** — a normalization applied to palette 256 OUTPUT when distance distributions need comparison across heterogeneous buckets | Calling Fisher-z on a non-correlation value is a category error. |
| **BF16 mantissa exact transformation** | direct palette-256 → palette-256 mapping using BF16 mantissa context | `PaletteIdx` + `EulerGammaOffset` (+ mantissa context) | new `PaletteIdx` | **bypasses the cascade entirely** when the direct mapping is known | The fast path: when you already have a palette index and need the transformed palette index under a known offset, this is one typed hop in palette space. No metric translation, no cascade levels. |

---

## The roundtrip anti-pattern (worst case — do not write this code path)

```
  palette 256 distance        [Level 3 typed]
        │
        ▼
   fisher z normalize         [valid: variance-stabilize a Level-3 result]
        │
        ▼
   "treated as cosine"        ←── BUG: popcount IS cosine replacement,
        │                          fisher-z of palette ≠ cosine input
        ▼
   hamming distance           [Level 1 typed — wrong scale, wrong topology]
        │
        ▼
   HDR popcount preheat       [Level 1 detail — re-derived from wrong source]
        │
        ▼
   early exit                 [decisions made on un-rounded round-trip]
        │
        ▼
   palette 256 distance       [back to start, BUT: buckets + Euler offset lost
                               somewhere in the chain — answer differs from
                               the original Level-3 result by quantization noise
                               + loss of bucket assignment]
```

Each arrow:
- pays an arithmetic / cache cost
- loses the typed-distance identity (the type system stops protecting the call site)
- introduces conversion error that compounds along the chain
- can converge back to "approximately the same number" — which makes the bug invisible in unit tests but wrong in deployment

## The direct path (preferred)

```
  palette 256 distance ──[Euler gamma offset known]──▶ palette 256 BF16 mantissa exact transformation
                                                                    │
                                                                    ▼
                                                              new PaletteIdx
                                                              (stays in palette space)
```

ONE typed step. Stays in palette-256 type space. Preserves bucket + offset throughout. No cascade traversal, no metric translation, no conversion noise.

When the BF16-mantissa direct path is applicable (caller has a `PaletteIdx` and an `EulerGammaOffset`), use it. The cascade exists for the case where you don't yet know which palette band the target lives in.

---

## API design rule (binding)

1. **One fn per metric. Named.**
   ```rust
   pub fn palette256_distance(
       a: PaletteIdx, b: PaletteIdx,
       buckets: &Buckets, offset: EulerGammaOffset,
   ) -> PaletteDistance;

   pub fn hdr_popcount_early_exit(
       a: &Fingerprint256, b: &Fingerprint256, threshold: u16,
   ) -> Option<HammingDistance>;

   pub fn base17_l1(a: &[i16; 17], b: &[i16; 17]) -> i32;

   pub fn palette256_bf16_mantissa_transform(
       p: PaletteIdx, offset: EulerGammaOffset, mantissa: BF16MantissaCtx,
   ) -> PaletteIdx;
   ```

2. **Conversions are explicit and named.** When a caller must cross metric boundaries (escape-hatch case):
   ```rust
   pub fn hamming_distance_to_palette_index_estimate(d: HammingDistance) -> PaletteIdx;
   //                                  ^-- name says "estimate" so callers can't mistake it for exact
   ```
   The call site MUST carry a comment naming WHY the conversion is happening (not the default path).

3. **No `Box<dyn Distance>` / no `enum DistanceMetric { Palette, Hamming, Base17, … }` / no `fn distance<T: HasMetric>(a, b) -> f32` umbrella.**
   The type system distinguishes the metrics for a reason.

4. **Newtype the output.** `PaletteDistance(f32)`, `HammingDistance(u16)`, `Base17L1(i32)` — different output types prevent accidental cross-metric arithmetic. `let d = palette256_distance(...) + hamming_dist;` should not compile.

5. **The buckets and Euler-gamma-offset arguments to palette-256 fns are REQUIRED, not optional.** Default values for those parameters are domain-meaningful (changing them changes the metric); a `Default::default()` impl on `EulerGammaOffset` is acceptable only with strong documentation that the default is a deliberate calibration constant, not a sentinel.

---

## What this means for the SoA/AoS sprint (W3-W6) and beyond

### W3-W6 (currently in-flight on `claude/w3-w6-soa-aos-helpers`)
- `SoaVec<T, N>`, `soa_struct!`, `aos_to_soa`, `soa_to_aos`, `bulk_apply`, `bulk_scan` are **generic over T**. They do NOT bake in any distance metric.
- **DO NOT** during the sprint or in post-review extend any of these primitives toward distance computation. Distance stays out of the helper layer.
- If a worker is tempted to add `fn bulk_distance<T>(...)` to `bulk.rs` → STOP, that's the umbrella anti-pattern.

### W7 (deferred — cognitive bulk ops)
- When W7 lands, EACH metric gets its own bulk primitive named for the metric:
  ```rust
  pub fn bulk_hdr_popcount_early_exit(
      query: &Fingerprint256, db: &[Fingerprint256], threshold: u16,
  ) -> Vec<Option<HammingDistance>>;

  pub fn bulk_palette256_distance(
      query: PaletteIdx, db: &[PaletteIdx],
      buckets: &Buckets, offset: EulerGammaOffset,
  ) -> Vec<PaletteDistance>;

  pub fn bulk_palette256_mantissa_transform(
      palettes: &[PaletteIdx], offset: EulerGammaOffset, mantissa: BF16MantissaCtx,
  ) -> Vec<PaletteIdx>;
  ```
- Underneath, the bulk fns MAY use `SoaVec` / `bulk_apply` from W3/W4 for layout staging. That's fine — those are layout helpers, not distance helpers.
- The cascade orchestrator in `hpc/cascade.rs` calls each Level's typed bulk primitive directly. It does NOT internally translate Level-1 outputs to Level-3 inputs by passing through Fisher-z.

### Bench harness (prereq for W7's SIMD acceleration)
- Per-metric benches: one per typed primitive, no umbrella `bench_distance` macro.
- The bench output should report the typed primitive's name in the column header so regressions are attributable to the specific metric.

---

## Cross-references

- `CLAUDE.md` § "Three-Level Cascade: How the Search Actually Works" — describes L1 Hamming sweep, L2 Base17 L1, L3 Palette lookup. The Levels are typed distance bands, NOT tiers of a generic distance abstraction.
- `src/hpc/cascade.rs` — the orchestrator. Inspect before adding any new distance code; the call chain there is the canonical correct example.
- `src/hpc/distance.rs` — distance utilities. Audit for any `fn distance<T>` umbrella that may have crept in; refactor to typed fns if found.
- `src/hpc/plane.rs`, `src/hpc/vsa.rs` — produce values that feed the cascade. Their output types must be the typed distance newtypes, not raw `f32`.
- `.claude/knowledge/w3-w6-soa-aos-design.md` — the W3-W6 helper design (this doc constrains what those helpers must NOT grow into).
- `.claude/knowledge/vertical-simd-consumer-contract.md` — the layering rule (user → crate::simd → simd_{type}). Same family: layering and typing are both about preserving identity across abstraction levels.

---

## TL;DR for an agent reading this in 30 seconds

1. Palette-256 distance ≠ Hamming popcount ≠ Base17 L1. Don't put them under one API.
2. Palette-256 distance carries `buckets + EulerGammaOffset` always. Don't drop them.
3. The fast path inside palette space is the BF16-mantissa direct transform — one hop, no cascade.
4. The cascade is THREE typed levels in sequence, NOT a generic distance pipeline with intermediate conversions.
5. Conversions between metric types must be explicit, named, documented per call site.
6. No `Box<dyn Distance>`, no umbrella `fn distance<T>(...)`, no `enum DistanceMetric`.
