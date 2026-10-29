# D-POF-ANT-0: Position × Order × Law, the first antennas

READ BY: savant-architect, simd-savant, vector-synthesis, the Perm64 owner, the
DR/DT owners (lance-graph #1835), the Thought-Git owner (#1848).

Scope: lab code and docs only.
- Code: `examples/pof_antennas/{position,exposure,order_law,meta,scale,main}.rs`
  and `tests/pof_antennas.rs`.
- No facade word, backend, dispatch, storage owner, cache, registry, mask
  minting, Thought-Git runtime or public trait was added.
- #393 (E5) is untouched and its verdict stands: E2 below is a new row, not a
  revision.
- lance-graph was read at `origin/main` 61720fa (#1848 merged) and not
  modified.

Method:
- The tests are exact oracles plus falsifiers: 41 tests, 14 disable runs, all
  red.
- Timings are medians of 2001 runs (L1) or 5–9 runs (streamed). Each tier was
  run 3 times; the tables give min–max of the medians.
- Tiers: native = AVX-512 + VNNI; v4 = `config-v4.toml`, AVX-512 without VNNI;
  v3 = `config-v3.toml`, AVX2. Every binary prints its tier on the first line.
- Loop bodies come from `#[inline(never)]` copies, disassembled with objdump.

## The answer in one paragraph

Sharing a coordinate basis is useful without sharing a mask, but only under
two conditions:

1. **Invariant results** (counts, sums, the exact U9 dot) need **no** move
   under a rebase.
2. **Equivariant results** (lane-local selects) need **one** carry of the
   **result**, not one per input.

Under both, a rebase costs at most one `conjugate_mask`. A rebase is "free"
only if conjugation is cheap. Today it is not: `conjugate_mask` expands the
mask to bytes with a 64-lane scalar loop, which costs about 4.4 ns on
AVX-512. That is more than all three operations together.

The laws that license reuse are already in the stack:
- ndarray: the markers `PermInvariant` and `PermEquivariant`;
- lance-graph mask-risc: commutativity read from the ternlog table, a merge
  law per terminal, `ExtentUnsupported` refusals, and stated row bounds;
- the witnesses `OrderedLaneWitness` and `SealedFacetLane::validate`.

None of them needs a new trait. The smallest evidence-carrying fold result
has six fields: result, the law it was folded under, basis/frame, order
witness, the dependency versions, and UNKNOWN kept as its own value.

## A0: census of what already exists

Classification: IMPLEMENTED (code exists), MEASURED (timed by this probe or a
cited PR), PROPOSED (plan/decision only), MISSING.

| primitive | file | owner | in → out | position | order | law | frame/version | oracle/tests | cost measured | status | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Perm64::{then, inverse, relative_to}` | `src/hpc/perm.rs` | ndarray (#336) | map × map → map | a coordinate map | composition order (`then` left to right) | group law | **none**: a map carries no basis id or frame | perm.rs, 24 tests | — | IMPLEMENTED | basis identity; A1 adds the frame outside the map |
| `Perm64::materialize` | perm.rs:283 | ndarray | 64 B → 64 B | the only data move | n/a | equivariant | none | perm.rs | 9.6 ns for 4 planes (VNNI), 57 ns (AVX2) | MEASURED | — |
| `Perm64::conjugate_mask` | perm.rs:265 | ndarray | `u64` → `u64` | carries a mask through the map without moving data | n/a | equivariant | none | perm.rs; A1, A6 | **4.4 ns each** (VNNI/v4), **~60 ns** (AVX2) | MEASURED | the mask→bytes expansion is a scalar `from_fn` |
| `LaneFold` + `PermInvariant` | perm.rs:766 | ndarray | 64 lanes → `Out` | ignored | any | **marker only**, unchecked | none | perm.rs; A4 negative `IndexWeightedSum` | — | IMPLEMENTED | a wrong `impl` is not detected; A3/A4 tests are the guard |
| `LaneOp<N>` + `PermEquivariant<N>` | perm.rs:781 | ndarray | N × 64 → 64 | preserved | any | marker only | none | perm.rs; A4 negative `PrefixSum` | — | IMPLEMENTED | same |
| `InBasis`, `combine_in_basis` | perm.rs:876, 910 | ndarray | two bases → one | relative map | n/a | requires equivariant | basis = the map value | perm.rs; A3 | — | IMPLEMENTED | — |
| `PermTable12`, `Exposure`, `count_where` | perm.rs:338, 957, 1011 | ndarray | 4096 cells → a 64×64 `u16` kernel | the whole field at once | none | invariant counts only | none | perm.rs; A6 | **7.5 µs per field**, 1 allocation (8 KiB) | MEASURED | — |
| `PermChain`, `PermField`, `PermBatch`, `Schedule` | perm.rs | ndarray | — | — | — | — | none | perm.rs | — | IMPLEMENTED, not used here | — |
| mask-RISC masking ops (`eq/lt_*_to_mask`, masked reductions, `ternlog`, `mask_shift_morton`) | `src/simd_masking_ops.rs` | ndarray (#306) | lanes → mask, mask → fold | strided read in place | traversal order free | per-op doc | none | 181 tests | per #306 | IMPLEMENTED | — |
| `U64x8::sum8_u8_add_masked` | `src/simd_avx512.rs:2186` | ndarray (#390) | `U8x64` + keep → 8 × `u64` | vertical | free | exact, 255·8 per lane per call | none | #390, #393 | #393 | MEASURED | — |
| `I32x16::dot4_u8_i8_add(_masked)` | simd_avx512.rs | ndarray | `U8x64` × `I8x64` → 16 × `i32` | vertical | free | exact within the flush bound | none | #388 | #388 | MEASURED | — |
| `U8x64::mask_blend` | simd_avx512.rs, simd_avx2.rs | ndarray | mask × 2 → 1 | lane-local | — | equivariant | — | — | **one `vpblendmb`/`vmovdqu8{z}` on AVX-512; a per-lane loop on AVX2** (A2 E2: 73 vs 10 ns) | MEASURED | AVX2 realization (already on the facet-stride R2 list) |
| `U64x8::ternlog::<IMM>` | simd_avx512.rs:5851 | ndarray | 3 → 1 | lane-local | — | the immediate is the truth table | — | — | LLVM already fuses `(a^b)&c` to one `vpternlogq` | MEASURED (A4) | — |
| U9 kernels B, E3–E5 | `examples/u9_*` | lab (#388–#393) | 2 × U9 block → `i64` | vertical | free | exact, flushed | none | #388, #389, #393 | #393 | MEASURED | — |
| W1a consumer contract | `.claude/knowledge/vertical-simd-consumer-contract.md` | ndarray | — | — | — | defines saturation/overflow docs | — | — | — | IMPLEMENTED (doc) | — |
| `OrderedLaneWitness`, `SealedFacetLane::{seal, validate, verify}` | lance-graph `lance-graph-contract/src/ordered_lane.rs:51, 193, 264, 294` | lance-graph | keys × version → witness | numeric projection order | **sealed sort order** | O(1) validation, O(n) re-attestation | `LanceVersion`, n_rows, digest, lens | ordered_lane tests | — | IMPLEMENTED | — |
| mask-risc `Program`, `Terminal`, `GroupFold::merge`, `ExtentUnsupported`, `MASKED_SUM_I32_MAX_ROWS`, ternlog commutativity | lance-graph `lance-graph-mask-risc/src/{ir,exec,value}.rs` | lance-graph | planes → terminal | borrowed planes | straight-line ops, one terminal | **merge law per terminal; commutativity read from the table (`exec.rs:427`); partial-extent refusal; stated row bounds** | none | mask-risc tests (e.g. `exec.rs:2460`) | per mask-risc PRs | IMPLEMENTED | — |
| thought-git-probe `BaseRef`, `VersionStore::{seal, commit_cycle}` | lance-graph `thought-git-probe/src/receipt.rs:56, 128, 153` | lab | thoughts × base → commit witness | — | cycle | — | `BaseRef{dataset, branch, version}` | probe tests | — | IMPLEMENTED (lab) | `DurableRef`, `ThoughtCommitId` are PROPOSED (D-CPG); not in crates |
| DR = HHTL16 addressing, DT = `ClassId → schema + methods`, DR #4 method routing | lance-graph `.claude/plans/d-drdt-dr-dt-ownership-amendment-v1.md` (#1835) | lance-graph | — | DR owns coordinate maps | DT owns ordered effects and method identity | — | `FrameStore` (lab) | — | `LocalMap` 1,050 ns cold, 7.7 ns per lookup | PROPOSED (decisions) | bit-matrix column read (#1835 §3) |
| witness-gated reuse; 9 stale cases X1–X9 | lance-graph `.claude/plans/recursive-perspective-rebinding-v1.md` §6 (#1622) | lance-graph | prepared frame → reuse or refuse | — | — | — | `Witness` | the probe implements the refusal; X1–X9 not run | — | PARTIAL | X1–X9 |
| epistemic basin B0–B11 (four spaces kept apart; duplicate roots do not inflate support; UNKNOWN never becomes zero or false) | lance-graph `docs/research/epistemic-awareness/2026-10-29-epistemic-basin-metathought-pr-scratchpad.md` (#1848) | lance-graph | — | — | — | — | — | — | — | PROPOSED | all |

The Haiku maps behind this table were spot-checked against the source at
their stated lines: `ordered_lane.rs:193` and `:264`,
`simd_avx512.rs:2186`, `simd_masking_ops.rs:6053`, and 10 `perm.rs` items.
Every one matched.

## A1: position antenna

Fixture: 64 positions with four byte planes (magnitude, polarity, predicate,
selector), one `Perm64` basis, and a frame number. Three operations:
- `count(pred & positive)`;
- a masked signed sum (`sum8_u8_add_masked` on each side);
- a select `(sel & (pred ^ pos)) | (!sel & (pred & pos))`.

| ns per block (64 positions) | native L1 | v4 L1 | v3 L1 | native streamed |
|---|---|---|---|---|
| P0 separate preparation per operation | 3.75 – 4.69 | 3.77 – 5.08 | 8.33 – 8.38 | 13.9 – 16.4 |
| P1 shared basis: prepare + run | 3.50 – 4.22 | 3.50 – 4.08 | **42.4 – 42.7** | 10.4 – 12.4 |
| ↳ setup only | 1.77 – 2.73 | 1.70 – 2.53 | 2.78 – 2.81 | 9.7 – 11.4 |
| ↳ execution only | 2.47 – 3.63 | 2.50 – 3.44 | 5.23 – 5.27 | 8.8 – 10.4 |
| P2 eager rebase: prepare + 3 × `conjugate_mask` + run | 16.4 – 17.6 | 15.2 – 17.3 | 246 – 261 | 20.6 – 21.6 |
| ↳ rebase only (3 × `conjugate_mask`) | 13.6 | 13.3 – 15.5 | 179 | 14.6 – 17.0 |
| **P2-lazy: run in source, conjugate the select result once** | **6.34 – 7.05** | 6.36 – 9.03 | 53.4 – 55.9 | 10.4 – 11.3 |
| ↳ 3 × blended conjugation (`mask_blend` + `materialize`) | **2.42 – 2.58** | 4.31 – 8.11 | 453 – 466 | 7.2 – 7.7 |
| reference: materialize all 4 planes | 9.64 – 10.9 | 15.3 – 21.5 | 57.1 – 58.2 | 11.1 – 11.7 |

Findings:

- **Sharing the basis is useful without sharing a mask** (PROVEN; WIN on
  AVX-512). P1 prepares three masks once and every operation reuses them: 0.93×
  of P0 in L1 and 0.79× streamed. That is preparation sharing; no mask is
  shared across operations of different domains.
- **A rebase never needs to move the magnitudes** (PROVEN). Count and the
  signed sum are permutation-invariant, so the re-based basis evaluates them in
  the source basis. The falsifier `mixing_bases_changes_the_result` shows that
  re-based masks over unmoved magnitudes give a wrong sum, so "evaluate in
  whichever basis is consistent" is load-bearing.
- **Defer the rebase to the one equivariant result** (PROVEN; WIN). Lazy
  rebase: `p∘select(a,b,c) == select(p∘a, p∘b, p∘c)` because select is
  lane-local. That turns 3 conjugations into 1, so 6.3 ns instead of 16.4 ns
  on VNNI. The negative, `lazy_rebase_is_refused_for_a_positional_result`,
  shows that the same deferral over a prefix-parity result is wrong.
- **`conjugate_mask` is the cost, not the algebra** (MEASURED; fruit, not
  fixed here). Its body expands the mask to bytes with a 64-lane
  `core::array::from_fn` (perm.rs:266) before the `permute_bytes`. With the
  expansion replaced by one `U8x64::mask_blend`, the same map costs 0.81 ns
  per conjugation instead of 4.5 on VNNI (equal bit for bit on 5 masks for each
  of 39 maps, 195 pairs). On AVX2 `mask_blend` is a per-lane loop and the blended
  form is 2.5× **worse**. So the fix is per tier, and it belongs to `perm.rs`'s
  owner, not to this lab.
- **AVX2 P1 anomaly** (OPEN). Setup + execution measured apart cost 8 ns, but
  together 42 ns. The array-backed AVX2 `U8x64` in a by-value `Prepared` is the
  likely cause. Not root-caused.
- Control bits follow their positions (`selector_must_follow_its_positions`);
  a stale frame is refused with both frames named (`a_stale_basis_is_refused`);
  two domains meet only through the relative map
  (`two_domains_meet_through_the_relative_map`).

## A2: exposure antenna

- E0 = B and E1 = E5 are unchanged.
- E2 prepares the sign-routed operand once: `a_pos = maskz(a, !neg)`,
  `a_neg = maskz(a, neg)`. The `low7` dots then run unmasked on it, and the
  anchor narrows the same vectors by bit 7.
- Reuse law: `maskz(maskz(a, s), t) == maskz(a, s & t)`. It is proven per byte
  by `a2_narrowing_law_holds_and_is_not_identity`, and E2 is exact on all
  511 × 511 pairs and on worst-case runs across three flush bounds.

| ns per 64 values | native L1 | v4 L1 | v3 L1 | native streamed | v4 streamed |
|---|---|---|---|---|---|
| E0 = B end-to-end | **2.89 – 2.91** | 5.63 | **10.0** | 8.2 – 9.8 | 12.3 – 13.2 |
| E1 = E5 end-to-end | 3.52 – 3.53 | 5.11 | 12.5 | 9.1 – 10.7 | 12.0 – 13.1 |
| **E2 end-to-end** | 3.19 | **4.69** | 73.6 – 84.6 | 8.5 – 10.1 | **11.1 – 12.9** |
| B / E5 / E2 arithmetic only | 2.39 / 2.75 / 2.53 | 5.00 / 4.44 / **4.17** | 9.2 / 10.9 / 73.6 | — | — |
| B / E5 / E2 scan | 2.66 / 3.28 / 2.92 | 4.77 / 4.64 / **4.29** | 9.2 / 11.6 / 74.8 | 3.2–5.1 / 3.6–5.6 / 3.5–5.1 | — |
| E3 two consumers, two passes | 3.03 – 3.09 | 5.61 – 5.64 | 11.4 | 24.5 – 28.4 | 33.4 – 39.5 |
| **E3 fused, one pass** | **2.42 – 2.44** (0.79×) | 5.28 (0.94×) | 9.7 – 10.2 (0.87×) | 15.8 – 17.4 (0.6–0.7×) | 22.9 – 23.4 (0.6–0.7×) |

As in #389/#393, the streamed "arithmetic only" rows read precomputed
operands that are larger than the compact layout, so they measure memory and
are left out.

Steady-state loop bodies (instructions; no stack operands on any AVX-512
body):

| tier | B | E5 | **E2** | E3 fused | positive-|a| alone |
|---|---|---|---|---|---|
| VNNI | 18: 4 `vpdpbusd`, 2 `vmovdqu8{z}` | 23: 2 `vpdpbusd`, 4 `{z}`, 2 `vpsadbw`, 2 `vpaddq`, `kandq`, `kandnq` | **21**: same, **without `kandq`/`kandnq`** | 20: B + 1 `vpsadbw` + 1 `vpaddq`, **still 2 `{z}`** | 9 |
| v4 | 40 | 36 | **34** | 42 | 9 |
| AVX2 | 122, 22 stack operands | 145, 18 | **510, 134** (per-lane `mask_blend`) | 133 | 28 |

Findings:

- **E2's reuse is exact, and its lowering removes the mask ALU, not the
  moves** (PROVEN, verified in the assembly). LLVM does **not** fold the nested
  `maskz` into one masked move: E2 still has 4 zero-masked moves, like E5. What
  disappears is E5's `kandq`/`kandnq` (the `s & t` mask), 2 instructions per
  block.
- **E2 is the best U9 candidate on AVX-512 without VNNI** (WIN). L1
  end-to-end 4.69 vs E5 5.11 vs B 5.63; arithmetic 4.17; scan 4.29. Every one
  of three runs and every row beats E5 and B on that tier. Streamed, the
  ranges overlap with E5 (host noise, as in #393).
- **E2 still loses to B on VNNI** (LOSS, 1.10×). It beats E5 there (3.19 vs
  3.53). The cause is unchanged from #393: B's fourth `vpdpbusd` reuses the
  sign's zero-masked copy, while the anchor needs a second masked move per
  sign.
- **E2 is refused on AVX2** (LOSS, 7×). `U8x64::mask_blend` is a per-lane
  loop there. On that tier `sum8_u8_add_masked` builds its keep mask in
  registers, so E5 is the better shape.
- **Two consumers fused share the preparation, not the accumulators** (WIN on
  every tier). E3 keeps B's `i32` lanes with their flush law and the
  positive-|a| sum's `u64` lanes with none. In the VNNI body the second
  consumer reuses B's existing zero-masked copy of `a` (still 2 `{z}`) and
  adds 2 instructions instead of a 9-instruction loop. The win is 0.79× in L1
  and 0.6–0.7× streamed (one read of `a` instead of two).

## A3 and A4: order and law

`order_law.rs`, 18 tests. A3 keeps three orders separate:
- traversal order: no effect on invariant folds;
- coordinate order: the basis; no effect on invariant folds, carried by
  equivariant ops;
- effect order: `Set`/`Add`/`Mul` sequences; `x=1; x+=1` ≠ `x+=1; x=1`.

A3 cases:
- Two readers of one snapshot see the same version.
- A sequential dependency must see the first result.
- `check_reorder` refuses a reorder of two non-commuting effects and names
  both.

A4 checks each licensed optimization against a negative example. Two of
them (reorder and defer) also have disable runs, X7 and X8.

| optimization | licensed by | negative example | correct | asm vs direct baseline |
|---|---|---|---|---|
| reorder a fold | `PermInvariant` (count, sum, min, max) | `IndexWeightedSum` is a `LaneFold` and not invariant; reordering changes it | PROVEN | count with the map skipped: **3 instructions** (`vptestmb`, `kmovq`, `popcnt`) vs ~200 with `materialize` (VNNI) |
| fuse two lane ops | both `PermEquivariant` and lane-local | `PrefixSum ∘ AND` is not fusable into one lane op | PROVEN | `(a^b)&c` is **already one `vpternlogq` (0x48)** in the direct form; the explicit `ternlog::<0x28>` adds nothing (**TIE**) |
| share coordinate preparation | the same frame and the same basis | a stale frame or a different basis is refused (A1) | PROVEN | A1 P1 vs P0 |
| defer a reduction | associative and commutative, **plus an exact bound** | flushing at bound + 1 = 16,449 blocks overflows `i32`; bound = ⌊i32::MAX / 130,560⌋ = 16,448, computed in the test | PROVEN | #393 (`B_FLUSH_BLOCKS` = 16,384 sits under the bound) |
| propagate a compact result | invariant: no carry; equivariant: one carry of the result | a positional result (prefix parity) cannot be carried after the fact (A1) | PROVEN | A1 P2-lazy |
| refuse an invalid composition | `check_reorder`; mask-risc `ExtentUnsupported` | `Set` across `Add` | PROVEN | — |

`LAW_TABLE` (10 rows) records shape, lane-locality, associativity,
commutativity, invariance, equivariance, ordered effects, exactness, overflow,
snapshot and frame for every operation used. It is a **lab table**, not a
trait. The equivalent production knowledge already lives in mask-risc:
commutativity read from the ternlog truth table, a merge law per terminal, and
stated row bounds. No new trait restates it.

## A5: meta-SoA antenna

`meta.rs`, 9 tests, a pure scalar oracle. The fixture:
- 1 sealed WORLD (id 7, version 1, 64 Kleene facts, 6 UNKNOWN);
- 2 THOUGHT overlays;
- 3 methods (AllOf, AnyOf, AtLeast-3);
- 2 rules (OpenWorld and ClosedWorld);
- 1 meta-evaluation.

Each `Receipt` records:
- frame/version;
- basis id;
- method id;
- order witness (FNV-1a over the step list);
- policy (the law);
- result and evidence;
- typed dependencies `{kind, id, version}`.

The meta fold (`meta_evaluate`, `which_changed`, `aggregate`, `may_reuse`)
reads receipts only. `meta_fold_reads_receipts_only` evaluates it after the
world has been dropped.

| negative case | detected by | disable run |
|---|---|---|
| stale source | `may_reuse` → `StaleDep{kind, id, recorded, current}` | X10 red |
| policy change with unchanged evidence | `compare` → `rule_changed` | X13 red |
| different thoughts treated as the same | `which_thought_changed_the_result` | — |
| UNKNOWN treated as FALSE | `unknown_is_not_false` (only ClosedWorld coerces) | X9 red |
| method order changing the effects | order witness → `OrderChanged` | X11 red |
| provenance lost in aggregation | `aggregate` keeps every child and the deduplicated dependency union | X12 red |

`aggregate` deduplicates dependencies, so a dependency named twice does not
count twice. That is the receipt-level half of #1848's "duplicate roots do not
inflate support"; the support half (votes) is not modelled. OPEN, it belongs
to #1848 B4.

## A6: scale antenna

The field is the 64×64 = 4,096 cells of a `PermTable12`, and one field-wide
question: `Σ #{ i : A[i] ∧ B[cell[i]] }`. No dense 64³ structure is
allocated; the largest object is the 8 KiB exposure kernel.

| µs per field | native | v4 | v3 | compositions | bytes written | allocations |
|---|---|---|---|---|---|---|
| per cell, materialized, row-major | 51.4 | 54.5 | 165 – 168 | 4,096 | 262,144 | 0 |
| per cell, materialized, Morton | 51.4 | 54.5 | 165 | 4,096 | 262,144 | 0 |
| per cell, conjugated mask, row-major | **5.2 – 5.7** | 9.6 – 11.0 | 1,138 – 1,150 | 4,096 | **0** | 0 |
| per cell, conjugated mask, Morton | 5.2 – 5.8 | 9.6 – 10.6 | 1,132 – 1,143 | 4,096 | 0 | 0 |
| whole field through `Exposure` | 7.6 – 8.0 | **7.5** | **6.1** | 0 | 8,192 | 1 (8,192 B) |

Findings:
- **Exposure folding is the only tier-independent form** (WIN on v3 by 190×,
  TIE on AVX-512).
- **Per-cell conjugation is fastest on VNNI**, at about 1.3 ns per cell, 0
  bytes written. On AVX2 it is the worst, because `conjugate_mask` there is the
  scalar expansion plus the array-backed `permute_bytes`.
- **Row-major vs Morton: TIE on every tier.** The table's working set is 8 KiB
  and stays in L1, so traversal order does not matter at this size. The
  traversal law holds: both orders give the same answer
  (`morton_visits_every_cell_once`).
- Cache-miss counters were not available in this container; working set,
  bytes written and allocations come from the counting allocator and the
  `Traffic` counters.

## Discovery matrix

| antenna | reusable substrate | new law found | missing element | correctness | cost | verdict |
|---|---|---|---|---|---|---|
| A1 shared basis | `Perm64`, masking ops | preparation sharing is independent of mask sharing | basis/frame identity is not on `Perm64` | PROVEN | WIN (0.79–0.93×) AVX-512; LOSS v3 (OPEN anomaly) | ADOPT-LAB |
| A1 rebase | `conjugate_mask`, `PermInvariant`/`Equivariant` | invariant: no carry; equivariant: carry the result once | cheap mask→bytes expansion in `conjugate_mask` | PROVEN | P2-lazy WIN 2.6× over eager; eager LOSS vs materialize | ADOPT-LAB (lazy); fruit for the `perm.rs` owner |
| A2 E2 | `mask_blend`, `dot4_u8_i8_add`, `sum8_u8_add_masked` | nested maskz = maskz of the AND; lowering removes the mask ALU, not the moves | AVX2 `mask_blend` | PROVEN | WIN v4; LOSS VNNI vs B (WIN vs E5); LOSS AVX2 | ADOPT-LAB on v4; #393 unchanged |
| A2 E3 | same | two consumers share preparation and keep separate accumulator laws | — | PROVEN | WIN all tiers (0.63–0.94×) | ADOPT-LAB |
| A3 order | `Perm64`, snapshot by value | three orders are independent; effects need a partial order | — | PROVEN | UNMEASURED (semantic) | PROVEN |
| A4 law | markers + mask-risc laws | every admitted rewrite has a counterexample | marker traits are unchecked | PROVEN | count-skip 3 vs ~200 instructions; ternlog fusion TIE (LLVM already does it) | PROVEN |
| A5 meta | mask-risc witnesses, thought-git-probe receipts | a receipt-only second fold is sufficient | `DurableRef`, `ThoughtCommitId` (PROPOSED in D-CPG) | PROVEN (scalar oracle) | UNMEASURED | PARTIAL (support counting OPEN) |
| A6 scale | `PermTable12::exposure` | the field fold is tier-independent; traversal is free at 8 KiB | — | PROVEN | Exposure TIE/WIN; per-cell conj WIN VNNI, LOSS AVX2 | ADOPT-LAB |

## The distilled contract: already represented, nothing new prescribed

| element | where it already lives | gap |
|---|---|---|
| **POSITION** | `Perm64` (map), `InBasis` (basis = map value), `conjugate_mask` (masks move without data), `Morton8x8` / HHTL16 (DR addressing, #1835) | a map carries no basis id or frame; the frame comes from WITNESS |
| **ORDER** | `Perm64::then` (composition order); `OrderedLaneWitness` (sealed key order); mask-risc `Program` (straight-line ops, one terminal); DT method schema (ordered effects, #1835) | the effect-order refusal exists only in this lab (`check_reorder`) and in DT's PROPOSED schema |
| **LAW** | `PermInvariant`, `PermEquivariant` (markers); mask-risc `GroupFold::merge`, commutativity from the ternlog table, `ExtentUnsupported`, `MASKED_SUM_I32_MAX_ROWS`; W1a overflow docs | markers are unchecked; exactness bounds are per-PR prose (B_FLUSH_BLOCKS) |
| **EXPOSURE** | `PermTable12::exposure` / `Exposure::count_where` (the field); the prepared operand (A2 E2, E3) | — |
| **WITNESS** | `OrderedLaneWitness` + `SealedFacetLane::validate` (O(1)) / `verify` (O(n)); RPR witness-gated reuse; thought-git-probe `VersionStore::seal` | `DurableRef` / `ThoughtCommitId` PROPOSED; the RPR stale cases X1–X9 not run |

No new struct or trait is needed: all five elements exist, and they live in
ndarray (POSITION, LAW markers, EXPOSURE) and lance-graph (ORDER, LAW tables,
WITNESS). That split follows the repo rule: ndarray = hardware, lance-graph =
meaning.

## Q1–Q3

**Q1. What can be reused without materialization?**

1. A **prepared position basis**: the masks built once from byte planes,
   shared by every operation that reads them (A1 P1).
2. The **result of an invariant fold** across any rebase or traversal order:
   no carry at all (A1, A4, A6).
3. The **result of an equivariant op**, carried once by `conjugate_mask`
   instead of carrying its inputs (A1 P2-lazy).
4. A **prepared operand** across consumers with different accumulator types
   (A2 E2 within one kernel; A2 E3 across two consumers).
5. A **field of maps** as one 64×64 exposure kernel instead of 4,096
   compositions (A6).
6. A **receipt**: the second fold reads receipts only (A5).

Materialization was never required. It was only the slowest correct form.

**Q2. Which laws make that reuse correct?**

- Permutation invariance licenses skipping or reordering the map for folds.
- Lane-local permutation equivariance licenses carrying one result instead of
  its inputs, and fusing lane ops.
- Associativity and commutativity **with a stated exactness bound** license
  deferring a reduction (16,448 blocks, not "large").
- Frame and basis equality license sharing preparation.
- Dependency-version equality and order-witness equality license reusing a
  receipt.
- `maskz(maskz(a, s), t) = maskz(a, s & t)` licenses narrowing a prepared
  operand.

Each license has a counterexample in the tests, and removing each one turns a
test red (14 disables).

**Q3. What is the smallest fold result that still carries its evidence?**

`(result, law, basis/frame, order witness, dependency versions)`, with UNKNOWN
kept as its own value of `result`:
- Without the law, a policy change with identical evidence is invisible.
- Without the frame, a stale source is reused.
- Without the order witness, reordered effects are reused.
- Without the dependencies, aggregation loses provenance.
- Collapsing UNKNOWN into FALSE changes AnyOf/AtLeast answers.

A5 has one failing test for each omission. Evidence is optional for
correctness; it is what makes the result explainable, not reusable.

## Correctness record

`tests/pof_antennas.rs` + the module tests: 41 tests. They pass on:
- native (AVX-512 VNNI), v4 and v3;
- AVX (qemu SandyBridge) and v2 (qemu Nehalem);
- aarch64 (qemu);
- nightly-simd.

Clippy `-D warnings` and rustfmt are clean. The facet-stride guard reports
nothing on the added lines; the lab uses `Perm64`'s API for every byte move.

Disable runs (commit first, patch with an asserted anchor, run, `git
checkout`), each red:

| id | disable | red tests |
|---|---|---|
| X1 | A1 selector left in the source basis | 3 |
| X2 | A1 stale frame accepted | 2 |
| X3 | A1 lazy rebase drops the result conjugation | 1 |
| X4 | A2 E2 anchor narrows the raw operand | 2 |
| X5 | A2 E2 sign routing inverted | 2 |
| X6 | A2 E3 second consumer reads the wrong sign | 1 |
| X7 | A3 every reorder admitted | 5 |
| X8 | A4 deferred reduction never flushes | 1 |
| X9 | A5 UNKNOWN coerced to FALSE under every policy | 3 |
| X10 | A5 reuse ignores dependency versions | 2 |
| X11 | A5 reuse ignores the order witness | 1 |
| X12 | A5 aggregation drops provenance | 1 |
| X13 | A5 policy not compared | 1 |
| X14 | A6 moving mask not conjugated | 1 |

## Fruits (not done here)

1. `Perm64::conjugate_mask`: expand the mask with one masked select on
   AVX-512 (5.6× in this probe), keep the current path on AVX2. This changes
   production code, so it is owned by `perm.rs` and needs its own PR with
   parity tests.
2. E2 as the AVX-512-without-VNNI U9 kernel: it beats E5 there in every row.
   It would join #393's ADOPT-LAB row; it is not integrated.
3. The AVX2 `U8x64::mask_blend` per-lane loop (already on the facet-stride
   R2 list) is what refuses E2 and the blended conjugation on v3.
4. The AVX2 P1 anomaly (8 ns apart, 42 ns together): OPEN.
5. The lance-graph RPR stale cases X1–X9 and #1848 B0–B11 remain theirs; A5
   is a scalar oracle that a B0 fixture could reuse.
