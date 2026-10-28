# DR512 / Codec9 universal epistemic P × magnitude law

**Operator ruling 2026-10-28.** This is the one DR512 law, not a numerical mode
selected by a consumer. It supersedes the signed-negative-bit reading in the
historical `palette512-signed-magnitude9-experiment-20261027.md` and the
two-plane probe in ndarray #381. This is a proposed lab CI gate, not
permission for BGZ wire migration, Quack source authorization, DR64³ layout,
or DT64 production dispatch.

| Presence P | Magnitude M | Meaning |
|---|---|---|
| 0 | 0 | **-0 / NaN / epistemically NEVER TOUCHED** |
| 0 | 1..255 | negative numeric value -1..-255 |
| 1 | 0 | valid **+0** |
| 1 | 1..255 | positive numeric value +1..+255 |

One u8 magnitude plus **one** presence bit per numeric row:
64 magnitudes + one u64 presence word = 72 bytes per 64 rows;
512 distinct codewords, exactly 511 signed numeric values and one
never-touched code. Numeric polarity P is NOT access control or alpha
write authority. Negative numeric evidence also is not a source-attested
Boolean negation unless such a mapping is established separately.

## The ONLY never-touched WHERE

```text
WHERE (P == 0) AND (M == 0)
```

No additional 8-KiB sentinel mask is required. A consumer can derive a
bitmap ON DEMAND if its vector engine needs one, but absence cannot be
inferred from P alone: P=0,M>0 carries a valid negative value.

## Touched write: normalize zero to +0

A write/calculation already admitted by its source owner obtains a **touched**
numeric value. Encode with `Codec9::encode_touched`. If the quantized
magnitude is zero, emit P=1,M=0 unconditionally. Pre-existing untouched
P=0,M=0 rows are not touched, not claimed and must remain untouched.
This holds especially for -0.49999, -0.4 and -0.0:

- TiesEven: -0.5 rounds to **valid +0**; -0.50001 → -1.
- TiesAway: -0.5 → -1; -0.4 → **valid +0**.
- Explicitly touched NaN/infinity is rejected, not converted into the
  never-touched sentinel. Finite out-of-range values saturate at ±255.
- LUT9 preserves an *existing* -0/NaN; when a nonzero negative LUT input
  becomes magnitude zero, it emits +0 and never fabricates absence.
- No numerical threshold or range filter `(-0.5,0)` is required.
  The determinant is the touched operation and final quantized zero.

## Existing compile-time owner patterns

`Codec9<R, C>` intentionally reuses ndarray
`codec8::{Rounding, TiesEven, TiesAway, Calibration}` and fixes the
other semantic parameters by construction. Asym, Floor and
`NoNaN` must **not** select a second DR512 bit interpretation. Codebook
enums with 512 arbitrary tokens are different from DR512 signed numeric
codes: they must retain their own typed IDs and source mapping. Never
reinterpret `bgz-tensor::WeightPalette` u8 archetypes as signed magnitudes.

## Cross-repo ownership and CI tripwires

1. **ndarray:** `src/hpc/codec9.rs`, `src/simd_stride.rs`,
   `examples/palette512_probe.rs`, `examples/sm9_presence_probe.rs`,
   `examples/lut9_bench.rs`, `crates/simd-masking-parity/src/lib.rs`.
   Exhaustive 512-codeword oracle, negative-zero ceiling test,
   backend/shape parity, WHERE 64K geometry.
2. **lance-graph DT64/Quack/DR64:** keep owner-issued source versions,
   source authorization, K3 predicates, sparse write/snapshot and alpha
   claims distinct. A numerical P=0 is **not** whitelist, permission or
   proof that an ontology relationship is absent.
3. **Odoo, OGIT/OGAR and crai:** use pinned source codebook IDs;
   guard incomplete registry, unresolved IDs and source-attested WHEN
   / ON / DO. Test 255/256/257 codebooks explicitly.
4. **MedCare:** no physical clinical-route promotion from a numeric
   sentinel or unsigned-magnitude sign bit without its independent
   compilation, ACL and source evidence.

This document does not grant a merge or production ABI change. Every
subsequent consumer PR must either prove exact P×M parity using the
independent oracle or refuse the integration. Operator-rejected
lance-graph #1812 stays quarantined and is NOT implementation evidence.
