# N-FACET-STRIDE-TILE-1: Positional execution beneath coordinate-routed SoA tiles

**Status:** PROPOSED / documentation only. **Date:** 2026-11-05. **Owner:** ndarray. No semantics or new kernel is admitted by this plan.

## Boundary: what ndarray actually owns

**DR routes where computation looks; DT validates structural and behavioral meaning; Quack lowers a lawful operation; mask-RISC and ndarray execute it.** ndarray never decides that a 128-bit GUID is a mailbox, that a property is `XVERWEIS`, or that a row in one table may be used to index a foreign lane.

The consumer's SAP / PSCustomObject-shaped logical tile is a normalized, one-key ordinal hashtable **in meaning**, not a HashMap or an array of host-language objects. After a scoped identity correspondence is prepared, a reference to the matched row, class basis and field ordinal permits lazy property reads. Each source SoA stays where it is.

## Source truth

- [`src/simd_stride.rs`](https://github.com/AdaWorldAPI/ndarray/blob/master/src/simd_stride.rs) defines the **FACET STRIDE** positional reading law: one 128-bit facet is sixteen byte positions loaded in place; `lut_u8` and `lut9` transform positional bytes. They **do not** implement an arbitrary 128-bit ID join.
- [`lance-graph-sap::query`](https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-sap/src/query.rs) already executes typed CATS lane folds via Quack and mask-RISC without allocating one source DTO per row.
- [`ogar-dir-core::correspond`](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-dir-core/src/correspond.rs) first builds sorted `(scope, u128 key, row)` indexes, then folds GUID evidence numerically. A cold index is not a hot-path violation or an independent authority.
- [`ogar-dir-core::dn128`](https://github.com/AdaWorldAPI/OGAR/blob/main/crates/ogar-dir-core/src/dn128.rs) uses a 16-byte hierarchy coordinate suitable for strided prefix matching. This is **not** `Guid128` identity, despite the equal width.

**The promised optimization is chained positional references after one admitted identity bind.** Don't represent this as an already shipped `simd_stride` GUID-join primitive or claim that preparing correspondence requires no allocation.

## Admitted versus open operations

| Shape | Execution precedent | Boundary |
|---|---|---|
| Typed scalar compare / resident-plane filtering | Quack → mask-RISC | Field type, scope and missing-state semantics supplied by owner |
| Count, group sum, survivor mask | Existing Quack and SAP consumer | Correct result under independent scalar oracle |
| 16-byte in-place prefix over `Dn128` | Dir-sim strided pattern | Scope + depth separate from bytes |
| `Guid128` cold correspondence | OGAR sorted index + numeric fold | Not equivalent to arbitrary SIMD byte-LUT join |
| Warm property read via chained ordinal references | Proposed binding and witness shape | Must be co-addressed or admitted; no automatic cross-table Gather |
| XVERWEIS-like operation pointer | DT64 `MethodRef` / `AlgebraRef`, Bagop owner | May produce coordinate, scalar or mask according to method contract; ndarray does not own Excel return semantics |
| Composite cross-table operations | OP-14 / NotASide / ADMISSION B | STOP when missing lawful operation, no hidden Gather or computed-mask reuse |

A 128-bit GUID is not truncated to 16 bits to fit a row coordinate; ordinal numbers exist only within a snapshot/frame. `ClassView`/codebooks give field ordinals, not ndarray data lanes.

## Minimum falsifier

Construct a tiny AD → Entra → EXO correspondence with one clean `User_{GUID}` backsync, a second duplicate, a foreign tenant, and one CATS-like numeric field. Bind the correspondence once under scope and layout witnesses; evaluate repeated property probes and one admitted group/sum fold. Reorder the local row ordinals while keeping the same canonical GUIDs. The result must either rebind to the new layout or refuse. Never accept a stale positional answer.

Record **cold** decode/index/bind cost apart from **warm** field access and fold. Count all allocations in the hot path, full output bytes at egress, and mask scratch separately. Compare independent scalar and available SIMD backends for correct tails, alignment, null/unknown states and byte order. Do not claim AVX2/AVX-512/NEON/WASM verification when the target was not run.

**STOP:** if source pointers are computed by a foreign lane without the required authorization, if reuse of a derived mask crosses an unadmitted program boundary, or if DR frame and SoA layout are not attested (DRISA M1/M2), report `NOT ADMITTED`. No local workaround operator in ndarray.

**POC critical path:** P0 architectural law + existing OGAR / dir-sim + SAP fold + one verified binding. **Downstream only:** GridCraft, Excel formulas, F### formula mapping, report-rs workbook ingestion, Java, Codegraph. No code modified by this PR.
