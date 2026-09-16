# Masking ops — what is DONE, what is PENDING, and what is only POTENTIAL

> READ BY: any agent about to add, extend, or cite a `*_to_mask` / `mask_*` /
> `masked_*` primitive; any session picking up the DuckDB→V3 translation
> matrix; any consumer (lance-graph, lance-graph-java) that thinks it needs a
> new masking capability. **Read this before proposing a primitive** — three of
> the seven named gaps are deliberately unbuilt, and one is already shipped
> with its own falsifier fired.
>
> Status: **FINDING for the DONE rows** (each measured, dated, gated).
> **CONJECTURE for OUTLOOK.** POTENTIAL is explicitly not a plan.

## Where this sits in the stack

Operator ruling, 2026-09-16 — the allocation the whole arc serves:

| what | lives in | membrane |
|---|---|---|
| **thinking** | lance-graph | **Panama** (computation never lives in Java) |
| **SIMD** | **ndarray** ← this repo | the `ndarray::simd` facade |
| **storage** | lance-graph | **Valhalla** (storage never lives in Java) |

So: **every masking op belongs here, and only here.** A consumer that
hand-rolls one has found a gap in this file, not a licence. That is the
`simd-savant` / missing-capability STOP rule, and it is why gaps are tracked
as gaps rather than worked around downstream.

The consumer-visible point of it (operator, same day): *"java is the low-code
intake glove around the lance-graph spine — lance-graph-java just happens to
offer the menu to the table in a pleasing way, using masking ops, offering 5
star for the price of a blink."* The masking ops ARE what makes the menu cheap.

## The gap list is not ours — it is the DuckDB→V3 matrix's own §3

`lance-graph/.claude/plans/duckdb-to-v3-translation-matrix-v1.md` §3 enumerates
G1-G7, each **verified absent** at the time with a pre-registered falsifier.
This file tracks their state.

## DONE — shipped, gated, and (where claimed) measured

| gap | what | state |
|---|---|---|
| **G6** | `mask_set_range(dst, lo, hi)` | **SHIPPED** (`33716b9`, fills vectorized `347875e`). Two consumers were working around its absence. `word_range_mask` built from two bits-below-N masks so `hi == 64` never computes `1u64 << 64`. |
| **G1** | `{eq,ne,gt,ge,lt,le}_u8_to_mask` | **SHIPPED** (`a8e7d7d`). The width where **packing is free**: 64 lanes, 64-bit word, one chunk is one whole word, no shift. **Falsifier ANSWERED — see below.** |
| **G2** | `{eq,ne,gt,ge,lt,le}_u64_to_mask` + `U64x8::{cmpeq_mask,cmpgt_mask}` on all six realizations | **SHIPPED** (`e05afbd`). Packing is NOT free here — eight groups share a word. Only avx512 (`epu64`) and NEON (`cmhi`) have the instruction; wasm has **no** unsigned ordered 64-bit compare and uses the sign-bias trick; scalar/avx2 are flat polyfills. **Measurement OPEN.** |
| **G4** | `mask_shift_morton` | **SHIPPED** (`255c36d`) — **and its own falsifier FIRED.** See OUTLOOK. |

Also shipped alongside, not a numbered gap: `simd_avx2`'s `U8x64::{cmpeq_mask,
cmpgt_mask}` were 64-iteration **scalar loops** sitting beside an already-
vectorized `U8x32` that solved them (`3a5da8c`).

### G1's falsifier — ANSWERED, and then corrected

Pre-registered: *"build `gt_u8_to_mask`, re-run the probes; if neither moves,
the widening was not the cost and G1 drops in priority."* It moved.

| tier | M1b: 6 masks | coal: one re-chain |
|---|---|---|
| v4 / AVX-512 | 41268 → **5973 ns** (**6.91×**) | 6319 → **1082 ns** (**5.84×**) |
| v3 / AVX2 | 38840 → 6084 ns (**6.38×**) | 6060 → 935 ns (**6.48×**) |

Nearly tier-independent ⇒ a **WIDTH** effect, not an ISA one: 4× fewer
instructions AND 4× less memory, plus (conjecture) the vanished packing.

⊘ **The first published figures (8.06× / 6.75×) were INFLATED by dead-store
elimination** — the timed closures carried no `black_box`, and the exposure was
asymmetric: the i32 arms were accidentally protected by a later read, the u8
arms had none. Corrected in `daed0fd`/`95ac06f`. **The conclusion survived, the
number did not** — which is the reason to state both.

## PENDING — the measurement half, and it is not optional

**Existing code is not a moved measurement.** Two pre-registered falsifiers are
still unrun, and both run through the same probe:

- **G1's second half** — PR #308's `r2il_column_scan_probe` crossover.
- **G2's only half** — its `find_ram_in_range` re-expressed against the real
  `u64` family instead of the hi32/lo32 bucket split it used to dodge the
  missing primitive.

**Blocked, not skipped.** That probe needs a column dump from `r2sleigh-lift`'s
`win32_census`, which needs a **Win32 PE binary**; none exists in this
container. A synthetic dump would produce a number shaped like the answer
without being it, against that probe's own *"a real lift rather than a
synthetic stream"*.

**G1's answer does not transfer to G2.** Part of G1's win is the packing
vanishing — a property of u8 alone. At u64 eight groups share a word and each
needs a shift, so the u64 family starts from a structurally weaker position.

Also pending, and each is a deliberate non-build rather than an oversight:

| not built | why | what would change it |
|---|---|---|
| `_under` (care-masked) siblings for u8/u64 | the i32/u32 families have them; the shape is mechanical; **no caller needs one** | a caller |
| `u16` compare-to-mask (half of G1) | no consumer compares u16 lanes | a consumer |
| `i64` ordered compare (half of G2) | the named consumer is an unsigned address window | a signed consumer |

A speculative family is surface with no falsifier attached to it. That is the
rule these three rows apply.

## OUTLOOK — named, grounded, not yet built

**G3 — `masked_argmin_i32` / `masked_argmax_i32`.** `masked_min/max` return the
**value**; *"the row where x is minimal"* is unanswerable without a second full
pass, and is ambiguous on ties. Pre-registered measurement: two-pass
find-value-then-find-row vs a fused single-pass argmin at selectivity
{0.01, 0.5}. **Under 2× the gap is real but low priority.** Either way the
tie-break rule (first index wins) must be pinned in the API, because a fold has
no natural one.

**G4 — shipped, and the model was half wrong.** `mask_shift_morton` landed and
its falsifier fired: over the full field the word op recovers **−14%** (14.5 vs
17.0 µs), not the modelled ~98%, and it **loses** to the NNUE delta arm
(9.3 µs). The **−66%** came from restricting the op to the trie node's own word
span — a mechanism the gap row never named. Verdict `[H]`: *the fitted cost
model was right about the COST of `n` and wrong about the REMEDY.* Worth
re-reading before trusting any other fitted model on this path.

**G5 — masked compaction (`vpcompress`-shaped), and the standing order is DO
NOT BUILD IT YET.** Every ELIMINATE verdict in the matrix rests on "the mask
travels instead of the data" — true *inside* a plan, false *at the membrane*.
Egress needs the index list exactly once. It is also **the most expensive gap
on the list**: AVX-512 has `vpcompressd`; AVX2/NEON/WASM/scalar need generated
bodies. The gate is a count, not an opinion — egress points per query in the
intended consumers. One per query ⇒ scalar compaction is fine. The matrix says
it in capitals: *"Do not build it before that count exists."*

**G7 — lane-vs-lane compare is DELIBERATELY ABSENT.** Every predicate here
takes a **scalar** threshold. DuckDB's surface is inherently binary; V3's is
unary-with-constant, and that narrowing is exactly the register in which the
`ConstantVector` ELIMINATE holds. If all intended predicates compare against a
constant, record G7 as *deliberately absent* **in the IR's docs**, so a future
session does not "fix" it.

**The nightly arm is AHEAD of the stable arms, and it is the contract
reference.** `src/simd_nightly/` carries **18 compare-to-mask pairs across
every width**; the stable arms have a subset. So N2/N3 were not adding a
capability — they were bringing the stable arms up to a contract the validation
arm already stated. Consequence: `scripts/masking-parity.sh nightly` is a real
cross-realization differential, not a same-author self-check, and the nightly
bodies are what a new width should be matched against.

## THE MENU — who serves these ops to whom (operator framing, 2026-09-16)

The masking ops are not a library looking for a caller; they are the kitchen
behind a menu, and the courses are at different stages:

| course | surface | state |
|---|---|---|
| **starter** | **SQL, via `lance-graph-quack`** — DuckDB→V3, zero-copy masked ops | **SHIPPED.** *"duckdb > quack is a nice proof-of-concept surface to offer SQL zero-copy masked ops, handed in the menu as a starter."* Plane leaf + survivor-skip gate, IN, projection, fused lowering, two-phase GROUP BY, each with a per-row oracle; and `plan_lower` is pinned EQUAL to quack's lowering by a differential, so there is one lowering rather than two that agree by luck. |
| **main** | **the Java glove** — `view.where(..).hop(..).count()` | **SHIPPED** (ABI minor 11). Reads as ordinary Java, costs a blink, because the work and the data are both elsewhere. |
| **another course** | **Gremlin / TinkerPop** | POTENTIAL, see below. |

Why the starter matters beyond being a demo: **SQL is the surface where "zero
copy" is checkable by a stranger.** A `SELECT … WHERE` either returns the right
rows or it does not, and the per-row oracle says which — so the proof-of-concept
is also the cheapest available falsifier for the whole mask-travels-instead-of-
the-data claim. That is a better reason to keep it than novelty.

## POTENTIAL — strategic, explicitly not a plan

These are the operator's outlook statements. **None has a falsifier, none is
scheduled, and none should be started without one.** Recorded so the shape of
the opportunity is not lost.

- **TinkerPop on lance-graph, replacing OGIT / JanusGraph / Cassandra**
  (operator, 2026-09-16). This is the masking surface's natural upper bound: a
  Gremlin `GraphTraversal` is `where`/`hop` composition, which is already
  `Mask × ClassView → Mask`. The honest question to answer FIRST is which
  Gremlin steps lower onto masks and which genuinely need G5 (compaction at an
  egress membrane) or G7 (lane-vs-lane) — i.e. this outlook is partly a
  **consumer census that would settle two open gaps**, which makes it cheaper
  to scope than it looks.
- **r2sleigh as the ghidra backbone for the extended menu** (operator, same
  day), **and the mechanism is a SHAPE match, not just a data source.**
  Ghidra's p-code addresses **varnodes, which carry types**. r2sleigh's r2il
  drops the typing and lands on a shape *similar to the V3 ABI* — which is the
  interesting part for this file, because the V3 register is
  `classid + 12 content-blind bytes` whose reading the ClassView picks per
  read. An untyped IL whose operands are addresses-plus-bytes is already in the
  register masks operate on; a typed varnode IR is not, and every type it
  carries is one more thing to strip before a mask can see it.

  Concretely for the gaps here: predicates over an untyped r2il operand are
  `lane + scalar` by construction — exactly G7's deliberate unary-with-constant
  narrowing — whereas a typed varnode surface invites the lane-vs-lane form.
  So this outlook, if pursued, is evidence FOR the narrowing rather than
  against it.

  It is also the immediate practical blocker: r2sleigh is the source of the one
  real-data probe this arc depends on (`win32_census` →
  `r2il_column_scan_probe`), which is what PENDING is waiting on. **So the
  outlook and the blocked measurement are the same dependency seen from two
  ends: a PE binary in reach unblocks the measurement half of two gaps.**
- **Bitpacked mask representation.** Not raised as a gap; noted because
  `strip_borders`-style binary morphology in sibling repos measured
  representation as the first-order lever and SIMD as second-order. If a
  masking consumer ever profiles hot, check representation before intrinsics.

## The one procedural rule this arc earned

**A benchmark's correctness is not only about what it MEASURES — it is about
whether the code under test is still THERE.** `timed()` in
`examples/hex_tenant_mq_probe.rs` now carries the `black_box` requirement in
its own doc comment, with the incident as provenance. Asymmetric protection is
the dangerous case: if neither arm is guarded both get deleted and the ratio
looks absurd; one arm accidentally guarded by an unrelated later read produces
a **plausible wrong answer**.
