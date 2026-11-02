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

**Runtime scale (operator, 2026-10-30).** The goal is both exact
`LUT[a, b]` work and learning. `Codec9Runtime<R>` takes the scale as a
value, checked once (positive, finite), and calls the same encoder and
decoder as `Codec9<R, C>`: for equal scales the codewords are identical
(tested over a 6,000-value sweep plus every codeword). Rounding stays a type
parameter restricted to nearest; storage, polarity, Sym and NaN stay fixed.
Use `Codec9` where a table over codewords must stay valid for everything
encoded with it; use `Codec9Runtime` where the scale is learned or
calibrated. Its `encode_touched_batch` / `decode_batch` read and write the
magnitude lane plus the presence plane, refuse a non-finite value before
writing anything, and clear presence bits past the last lane. A table built
under one scale is invalid under another: a consumer that moves its scale
rebuilds its tables. The batch loops are scalar today (not measured).

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


## [-a,+b] LUT distance efficiency, independent cold gate

For **valid numeric** DR512 pairs `x`, `y` with identical positive
magnitude step, distance need not use a 512×512 LUT:

```text
same P:      abs(m_x - m_y)
opposite P:  m_x + m_y
```

The result is `u16` (maximum **510**), **not** `u8`. The
512-codeword space has exactly one forbidden -0 sentinel that must
refuse distance instead of being treated as 0. Unit tests independently
decode all **511² = 261,121** valid numeric pairs and compare distances.

For a separately proposed calibrated asymmetric range [-a,+b], with
negative step a/255 and positive step b/255:

```text
both negative: abs(m_x-m_y) * a/255
both positive: abs(m_x-m_y) * b/255
opposite:      m_negative * a/255 + m_positive * b/255
```

The `tests/dr512_epistemic_contract.rs` independent oracle checks the
[-3,+7] case over all 511² valid pairs. **This is a distance-law
analysis, not an approved alternative P×M codec, nor a measured
speedup.** A full pairwise distance table has 512² entries, or **512 KiB**
at u16; even separate same/opposite 256² u16 tables total **256 KiB**.
The arithmetic is a subtract/abs or an add and should be the baseline
before claiming LUT efficiency. Any real SIMD/L1/L2-cache comparison
requires a pinned host/ISA benchmark with end-to-end memory costs.

## CausalEdge64 extension: DEFERRED, do not implement in this PR

Current CausalEdge64 bits 0..23 are THREE 8-bit SPO palette codes;
24..39 are NARS truth, 40..58 Pearl rungs, 59..63 epistemic/reasoning.
If each SPO code were extended from 256 to 512 choices, THREE extra
bits are required per edge (one per S/P/O), not an available 3-bit
space in the existing 64-bit contract. A 3-bit sidecar might be
theoretically sufficient but needs all THREE original palette identities
to be associated with the same pinned source/codebook generation and
the sidecar to be versioned and snapshot-attested with them.
Cross-page splits, stale codebooks, incomplete witnesses and partial
writes must refuse reconstruction. Do NOT steal CausalEdge64 truth,
Pearl or Epi5 bits, alter current codecs, or promote a physical layout.
Evaluate independently, under lance-graph's CausalEdge64/DT64/DR64
owner, only after evidence of correctness and benefit. **DEFERRED**.
