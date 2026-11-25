# Research prompt: Adaptive Numeric Binding — P × M, slice + seal, ordinal bookmarks

Date: 2026-11-24
Status: HYPOTHESIS / research only. No production API, codec, ABI, layout or dispatch change is authorized by this document.
Baseline inspected: ndarray master c790bd9b8ac3cd722138b6319e812858913ffda3.

## 0. Operator hypothesis

Instead of INT4 meaning four bytes, can an exact integer be a borrowed magnitude slice whose width and meaning follow from a sealed layout, and an existing mathematical method select its SIMD realization at binding time?

The physical type is a method parameter, not a new semantic type. Conceptually:

    {op}::{params}::{DR<slice, seal>} -> ndarray::simd::{op}::<physical_width>

This is a notation for the hypothesis, NOT an assertion that this Rust API exists. Investigate actual facade words and ownership before adding any API. Prefer a configuration of existing methods over a new runtime, numerical IR, type registry or interpreter.

Two distinct ideas:
1. Exact adaptive width: omit redundant high magnitude bytes, never change numeric value.
2. Adaptive precision: change quantization scale/resolution with an explicit accuracy contract. Defer this second idea until exact width is understood; promotion alone must never silently change scale.

Research PM9, PM17, PM25, PM33 (tentative names only). For PM25, five 24-bit magnitudes + five P bits = 125 bits; three bits remain in a 128-bit packing. Exact magnitude range 0..16,777,215; signed range ±16,777,215 with +0 and one untouched code. Four i32 values also occupy 128 bits. Five vs four means 25% more values/register or 20% less packed storage/value including padding, before metadata. This is a capacity calculation, not native five-lane SIMD, not measured throughput.

## 1. Reuse census first

Read and reconcile:
- src/hpc/codec9.rs, src/hpc/codec8.rs, src/hpc/abakus/pm.rs
- .claude/knowledge/dr512-epistemic-pm-invariant.md
- .claude/knowledge/add-pm-probe.md, u9-abacus-probe.md, u9-domino-probe.md
- .claude/knowledge/codec-soa-facet-map.md, simd-one-spec-design.md, simd-dispatch-architecture.md, vertical-simd-consumer-contract.md
- tests/dr512_epistemic_contract.rs, tests/add_pm.rs and scripts/add-pm-backends.sh
- Existing SIMD facade, U8/U16/U32 kernels, bitplane operations, carry, packed-lane/SWAR primitives, dispatch, slice/stride and codebook/range-index facilities. Search historical branches/docs for 5×U24; no corresponding implementation was identified in the current master filename scan. Do not assume it exists.
- Cross-repo context (read only): [lance-graph #2160](https://github.com/AdaWorldAPI/lance-graph/pull/2160), [EXO-330](https://linear.app/exo-red/issue/EXO-330), [EXO-332](https://linear.app/exo-red/issue/EXO-332), [EXO-294](https://linear.app/exo-red/issue/EXO-294), [EXO-296](https://linear.app/exo-red/issue/EXO-296), [lance-graph #2174](https://github.com/AdaWorldAPI/lance-graph/pull/2174).

Return a source-backed table: existing contract / source / reused verb / insufficiency / candidate / oracle. The ndarray crate should not depend on lance-graph. DR/DT64 owner and frame integration is a later consumer decision; use a local test witness only if needed. Reuse the actual existing seal/lease semantics where applicable; a cognitive Seal enum is not automatically a layout witness.

## 2. Freeze Codec9; generalize the law only if justified

For every supported width, hypothesize:
- P=0,M=0: untouched (⊥); P=0,M>0: negative.
- P=1,M=0: touched zero; P=1,M>0: positive.
- Valid = P OR (M != 0). No stored second validity plane.
- Touched quantization: TiesAway or TiesEven, same positive scale, rounded zero canonicalized to +0; never fabricate ⊥.
- Abakus add_pm/sub_pm/neg_pm are reference semantics: ⊥ is identity for the specified addition, not for every operation; ⊥ decodes to NaN. Preserve the order of arithmetic then decode. Exact overflow escape, not silent saturation.
- Separate raw polarity, relational presence, numerical validity, selection and write authority. No universal per-op presence prefilter; no numerical codec dependency in relational mask arithmetic. Audit EXO-296 raw Present operand versus numerical validity, without changing lance-graph here.
- Reuse typed rounding; do not infer that every wider variant is already implemented. No changes to Codec9 or DR512 naming. DR512 means codewords, not positions.

## 3. Candidate physical geometries

A. Fixed native i32/i64 with equivalent touched/overflow semantics (fair baseline); fixed existing PM9; native-width unsigned magnitude + P as additional control.
B. Five U24 magnitudes + five P bits in 128-bit packed block. Compare to four i32 values; measure explicit mask, cross-field carry isolation, shifts and shuffles. Compare packing with 3×U8 byteplanes + P and widened SIMD. Packed capacity is not computational lane count.
C. Homogeneous width segments, one shared width/scale/layout descriptor per segment; bind existing typed methods once per segment, not per value.
D. Ordered width groups with ordinal boundary bookmarks (e.g. b1,b2,b3), no per-value type tag or BTree lookup in the hot path. Consider a cold codebook/BTree for mapping when necessary, then compare direct boundary arithmetic, sorted runs and binary search. Do not add BTree by default.
E. Nested magnitude byteplanes: B0 for all N values, Bj only for ordinal >= bj; missing upper bytes are zero. For k bytes, derive address in plane j as i-bj if i>=bj, otherwise implicit zero. All P bits share the same ordinal order. Compare against segmented AoS packing and homogeneous SoA, not assume equivalence of physical costs.
F. Stable original ordinals with sparse high-byte exceptions or block-local width changes, versus global width sorting. Existing exact carry escape may be the best solution; compare rather than replace.

For fixed scale and equal width classes, a prefix/segment descriptor can derive offset from ordinal and boundaries without storing a per-number slice header. Borrow slices per segment. No per-value pointer, seal, allocation or type tag on the candidate fast path. Account explicitly for padding, length, tail, alignment, endian, checked offsets, and empty ranges. No transmute or unaligned reads without proof.

Ordered width groups can change physical row order. Do not change logical identities. Multiple independently sorted operands may force remapping/gather. Charge permutations, inverse mappings, staging, sort/reorder, cache misses and invalidation to the candidate, including when no such mapping existed on the baseline. A width bookmark is not an identity map. Compare stable-order block layouts before choosing a global sort.

## 4. Binding and operations

Ask how the existing ndarray facade can prepare/dispatch add, sub, neg, compare, range/threshold, count/any, modulo (ordinal vs numeric distinguished), min/max and sum for homogeneous spans. Start with add/sub/neg and exact count/validity only; others are follow-up questions. Do not invent a SIMD type for U24 merely to name the hypothesis.

For binary ops with different widths, intersect compatible segment extents if the frame and row order match; otherwise charge/measure alignment or refuse. Resolve widths and backend outside the per-value loop. Existing caller-held slices, mask words and typed operations first. No mandatory decode-to-i32, serialization, per-value branch, new universal filter, or tuple/selection population. Widening baseline is still required.

For sealed layout: document width, scale, row order, extent, owner/generation and borrow lifetime; test wrong frame, stale promotion, mixed order and overflowed address. Seal is a witness to layout, not authority or arithmetic truth. Do not add a new shared ABI or mint a cross-repo seal.

Carry permits exact escape; compare:
- fixed narrow + exact wide tally/side path,
- promotion at block/segment boundary,
- deferred promotion/repack at a new frame (with hysteresis),
- fixed wide control.
Do not silently write saturated carry bytes as a result. Demotion only if all relevant magnitudes fit at unchanged scale and outstanding readers remain valid. No width mutation underneath a borrowed slice. Promotion may be slower than escape; measure break-even.

## 5. Scientific precedent, not novelty by assertion

Survey Patched Frame of Reference/FastPFOR (Lemire & Boytsov), SIMD bitpacking/BP128, Stream VByte, Bit-Sliced Index Arithmetic (Rinfret et al.), MLWeaving and Stripes. Locate primary papers and code, identify which are compression, directly executable arithmetic or hardware-specific variable precision. Verify references and versions; compare their invariants and update/scan costs, and distinguish our exact-width CPU hypothesis from lossy adaptive precision and custom accelerators. Investigate prior sign-magnitude and SWAR multiword arithmetic. No claim of novelty until prior art and measured differentiator are established.

## 6. Falsification and measurement plan

Stage 0: reuse table and prior art, no code.
Stage 1: isolated example/probe only; no public ABI; independent scalar i64/i128 or checked wider oracle with Option for untouched (not floating NaN).
Stage 2: one narrow workload end-to-end; report negative results before broader work.

Correctness:
- Preserve all 512 existing codewords and existing exhaustive pair tests, both rounding policies and scale semantics.
- At widths 8/16/24/32: ±0 (touched vs untouched), ±(2^w-1), first carry, cancellation, cross-width addition/subtraction, opposite sign, signed modulo convention, carry chains, exact escape, empty spans, tail/padding, bookmark boundaries, mixed widths, stale/wrong seals, stable identity after promotion. Randomized differential tests; no exhaustive 2^25-pair claim.
- Rounding boundaries at ±0.5 and nearby f32 representable values with unit/non-unit scale. No precision loss during exact width changes.
- Disable controls: remove zero normalization, ignore high bytes/carry, off-by-one bookmark, wrong identity mapping, ignore stale witness; each must fail a meaningful test.

Workloads: synthetic skew (95/70/30% narrow), uniform full width, adversarial alternating widths, independent layouts of two operands, carry frequencies from zero to dense, sequential scan vs sparse/random read, update/repack churn. Realistic degree deltas/counters and sparse graph-derived statistics; use non-sensitive fixtures. Optional fixed-point geometry/sensors later. Measure cold binding + hot runs and full pipeline, including allocation, payload + all metadata, mapping, escape, reorder and memory bandwidth. Report ns/value, values/s, cycles/value, allocations, peak bytes, loads/stores, shuffle/branch/carry counts, p50/p95, compiler flags, CPU/ISA, data sizes and seeds. L1/L2/LLC and streaming; warm and cold. Inspect actual assembly and test backend parity.

Baseline add-pm-probe reports AVX-512 11.8–11.9 ns/64 vs decode-i32-encode 89.7 ns; AVX2 284.7–285.3 vs 133.5–134.7 ns. These are recorded historical probe results, not rerun here, and are not a direct equal-semantics native i8 benchmark. Operator reported ~24% CPU savings elsewhere: retrieve exact provenance, denominator, workload and ISA before reusing the number. Do not generalize across hardware. Native SIMD signed and unsigned ordinary add often share an instruction; compare complete equivalent operations. Prioritize real AVX2/AVX-512 and AArch64 NEON, ideally Raspberry Pi 3 1 GB; QEMU can support correctness, not hardware timing. Include scalar/wasm fallback as appropriate.

## 7. Return / decision gates

1. Reuse table and prior-art matrix with verified sources.
2. Exact-width geometry diagram and byte/metadata cost model for each layout, including stable mapping.
3. Actual facade call paths and smallest missing primitive, if any.
4. Independent correctness and disable-control evidence; reproducible benchmark command and raw results.
5. Break-even by distribution, carry, access pattern, update rate and ISA. Compare against fixed width, existing PM, escape, conventional packed compression + decode and directly executable alternatives.
6. Explicit recommendation: reuse as-is, retain as lab, reject, or propose smallest new contract with evidence.

Go only if semantics survive and an end-to-end workload shows a meaningful measured benefit versus equal-semantics controls. A storage-only win may justify a cold representation, not a hot arithmetic claim. A loss is an acceptable research result. No production migration, cross-repo code changes, new dispatch layer, or broad implementation without separate admission.

Research thesis: the mathematical method is fixed; the physical width is a bound parameter. Precision is not an independently materialized type population. Store the magnitude bytes needed, retain the polarity law, derive the slice from the coordinate, and bind the cheapest lawful existing operation.
