# Synergy map 2026-10-29 — since lance-graph #1591 and ndarray #378

**Status:** OPEN — exploration map. It decides nothing; the operator chooses.

**Base:** lance-graph `.claude/board/entries/2026-10-16-synergy-map-since-1579.md`
(the last detailed mapping). That entry revised the 2026-10-11 fleet listing
(`coresearch-2026-10-11-simd-fleet.md`: A–F, X1–X19). This map revises every
point of it and adds what the merges since then closed or opened.

**Delta read:**
- ndarray: #378–#383, all merged (`cadb572` … `2cbe131`), plus this PR.
- lance-graph `main` `b272a1c`: 114 merges since 2026-10-16 18:00. Only the
  rows this listing tracks were re-read.
- Fleet default branches, fetched 2026-10-29: lichess-rs `79af1ab`,
  tesseract-rs `7cbc969`, lance-graph-java `d8cdcc1`, OGAR `2a080a9`,
  MedCare-rs `fb829b6`, q2 `66281e7`, ladybug-rs `7f2cc38`, surrealdb
  `ef75a05`.

**Method:** Haiku cards did the harvests (§7). Every row marked CLOSED or
LOOSE below rests on a line that was re-read on the main thread; rows marked
"not re-read" are carried over unchanged.

## R2. Revision after ndarray #385–#393 and lance-graph #1847–#1848

Second pass, same day. ndarray `master` `870e733` (9 PRs, 39 files,
+8,597/−31). lance-graph `main` `61720fa` (2 PRs, docs only: the
`palette512-turbovec9-nsl-worksteal` board entry, the DR³ session
scratchpad, the epistemic-basin scratchpad). Sections 0–7 below are the
first pass and stay as written; this section revises their rows. The
facet-stride counts were re-taken: still 166 (7 R1, 157 R2, 2 R3). The new
code added no hit, and its probe that times shuffle lookups on purpose
carries a reasoned R1 exclusion.

### R2.1 What the nine ndarray PRs did

| PR | added | verdict recorded |
|---|---|---|
| #385 | `U16x32`: `zero`, `simd_min/max`, `saturating_add/sub`, `cmpeq/cmpgt_mask`, `reduce_min/max`, `to_f32x16_lo/hi` on AVX-512, AVX2, AVX, scalar, nightly; `reduce_sum` → `u16` everywhere; parity module `u16x32_facade_parity` | first parity coverage for the type found two older AVX-512 bugs: `shl`/`shr` returned 0 except for counts 1, 2, 4 and 8; `pack_saturate_u8` used `packus_epi16`. Both fixed |
| #386 | `I32x16::dot4_u8_i8_add` (`vpdpbusd` with VNNI, widen-then-madd otherwise), `U8x64::low_nibbles/high_nibbles` (GFNI when compiled in), every backend; contract note "`vpmaddubsw` is NOT an emulation of `vpdpbusd`" | triggered by turbovec's raw `vpdpbusd`/`SDOT` kernels |
| #387 | `I32x16::dot4_u8_i8_add_masked`; crate-private `BIT_IN_BYTE`, `mask_byte_spread`; `u9_cost_probe` | stored SM9 grid 46.0 dB at 1.29–1.32 ns/64; 512 Lloyd-Max IDs 49.9 dB but the in-scan lookup costs ~5× the arithmetic; no 512-entry facade word |
| #388 | head-to-head probe | B (U8×I8 byte-dot) wins every row; contract split Arithmetic511 vs Codebook512 |
| #389 | abacus probe | C (bit planes) REFUSE; D (16-bit carry column) ADOPT-LAB as A's f32 replacement; E (anchor at 128) no adoption; `carry_save3` ADOPT-LAB |
| #390 | `U64x8::sum8_u8_add_masked`, every backend | 0.687 ns/64 B on AVX-512, 1.99 ns on AVX2 |
| #391 | domino probe | F (eager flip) ADOPT-LAB; P, L, PL REFUSE |
| #392 | two EPIPHANIES entries; U1–U7 fruit list | — |
| #393 | E5 probe | ADOPT-LAB only on AVX-512 without VNNI; DEFER as B replacement; closes U1 |

### R2.2 Rows of this map that moved

| row | was | now | evidence |
|---|---|---|---|
| H1 parity for 16 unnamed types | LOOSE | **PARTLY CLOSED, and its value is now measured** | #385 covered `U16x32` in `src/simd.rs` test modules and found two real AVX-512 bugs on the first type tried. The 16-type count in `simd-masking-parity` is unchanged (the new tests live in lib test modules) |
| H3 AVX2 popcount: scalar `popcnt` vs nibble table | LOOSE | **REFRAMED** | `u9-abacus-probe.md:275`: "Without VPOPCNTDQ the popcount fold is a byte shuffle. On v4 and AVX2, `u64::count_ones` lowers to the `vpshufb` nibble table". Rewriting the 7 R1 hits as `count_ones` would silence the source-level guard and leave the machine code identical. The honest options are an exclusion with a measured reason, or a per-u64 `popcnt` that LLVM is prevented from vectorizing, which is a measurement question, not an edit |
| N6 D-RPF-4 "LUT-weighted strided sum" | PROBE | PROBE, sharpened | #387 measured per-position table reads (`lut_u8` shape) at 38–40 ns/64 against ~1.3 ns for the arithmetic they feed. A LUT-weighted sum built from `lut_u8` is therefore lookup-bound; the probe must time the LUT step separately |
| N7/N8 burn-down | one job | **priority set by measurement** | `U8x64::mask_blend`'s AVX2 realization is a per-lane loop (an R2 hit) and dominates every A/E3 row on AVX2: 75–81 ns vs B 10 ns per 64 (`turbovec9-u9-head-to-head.md:227`). Fix it first of the 157 R2 hits |
| N4 runtime-scale batch Codec9 | ruling needed | ruling needed, **second requester** | lance-graph W3 compares f32/DR512/i4/U9 graph loss and gradient; DR512 values for that come from a runtime-calibrated encode, the same need NSL stated |
| L4 `lut9` doc | LOOSE | LOOSE | `simd_stride.rs` untouched by the delta |
| H2 aarch64 guard | LOOSE | LOOSE | `src/cpu_guard.rs:318` and `:461-465` unchanged (the path in my census brief was wrong; the file is `src/cpu_guard.rs`) |
| H5 NaN erasers | LOOSE | LOOSE | 14 references in ndarray; 4 in lance-graph, all in `bgz-tensor/src/{adaptive_codec,had_cascade}.rs` |
| L5 runtime-detect census | LOOSE | LOOSE | 96 sites in 17 files, unchanged |

### R2.3 New dots

**N10 — two 9-bit contracts, kept apart on both sides.** ndarray's
head-to-head (`:244-250`) and lance-graph's entry (`:22-24`, scratchpad
`:21`) record the same split: Arithmetic511 (DR512: sign + magnitude, one
never-touched code) and Codebook512 (TurboVec-U9: 512 centroid IDs, all
valid). Both say never reinterpret one as the other; lance-graph adds "never
silently turn `lut9` into a generic U9 lookup" (`:34-35`), and ndarray added
no 512-entry facade word. Consistent; nothing to do.

**N11 — the two repos measured different things and point opposite ways.**
lance-graph's TurboVec9 numbers are recall: scalar i9 0.984 vs i4 0.788
Recall@10 on synthetic unit vectors (`:518`), using 512 Lloyd-Max centroid
IDs. ndarray's numbers are cost and signal: the IDs reach 49.9 dB but need a
512-entry lookup in the scan (5.4–5.7 ns/64); a stored uniform 9-bit grid
reaches 46.0 dB at 1.3 ns/64, the same per-dimension L1 cost as i4
(`turbovec9-u9-substrate.md:111-123`). Neither side measured recall for the
stored-grid layout. The owner experiment U7 (and lance-graph W1/W6) should
include the stored SM9 arm next to the ID arm, or the faster layout is never
compared on the metric that decides.

**N12 — the masked byte-dot hides shuffles on AVX2.**
`dot4_u8_i8_add_masked` on AVX2 lowers to 4 `vpshufb` (LLVM's spread of the
mask), and E5 on AVX2 to 8 (`u9-e5-probe.md:111`). The facet-stride guard
reads source, so it cannot see this. U2 (AVX2 streamed B 27–33 ns vs 9–10 in
L1) may be these shuffles rather than bandwidth; that is exactly the probe
U2 names. `scripts/codegen-witness.sh` is the existing tool that reads
assembly; adding a "no `vpshufb` in the masked dot on AVX2" assertion
there would make the rule cover what LLVM emits too.

**N13 — W4 is half done, by the right route.** lance-graph's W4 asks ndarray
for `dot4_u8_i8_add` + nibble ops "ONLY if needed by a proven consumer"
(`:564`). #386 added exactly those, triggered by turbovec's raw `vpdpbusd`
and `SDOT` kernels. The remaining half is turbovec moving onto the facade,
which is turbovec's repository (not in this session's scope) and lance-graph
W5 (the bridge is "likely non-building", `:57-61`).

**N14 — the U8→I8 bit view and R3 are one ruling.** U6 asks for a
`U8x64 → I8x64` bit view (today a store/load, 1.16–1.27 ns per block). R3
forbids mixing u8 and i8, and the 2 existing R3 hits are the same kind of
crossing (u8 `splat` cast to an `i8` intrinsic parameter). A ruling that
permits a named, zero-cost reinterpret — and still forbids arithmetic on
the reinterpreted value — would settle both.

**N15 — CI does not run the new lib tests on NEON or wasm.** #386's own note:
"CI never builds aarch64/wasm lib tests, only the two parity programs". The
`u16x32_facade_parity` and `dot4_nibble_facade_parity` modules run on x86 CI
only; NEON and wasm are covered through codes 0x500–0x507 in the parity
programs. The head-to-head tests also ran on the other tiers locally only
(`turbovec9-u9-head-to-head.md:135-136`).

### R2.4 Fruit, revised

Closed this pass: U1 (#393); W4's ndarray half (#386); the `U16x32` part of
H1 (#385).

Excluded, do not reopen without new evidence (from #392): bit-plane
multiplication as a B replacement; lazy EMPTY/FULL flags in bit-sliced
counters.

| # | item | size | why first |
|---|---|---|---|
| L6 | add the AVX2 "no `vpshufb` in masked dot" check to the codegen witness, initially as an expected-fail record | low | N12; makes the hidden shuffles visible |
| L7 | U2 probe: AVX2 streamed B, bandwidth vs the 4 `vpshufb` | low | the number is unexplained |
| L3, L4, L1, L2, L5 | as in §5 | low | unchanged |
| H9 | AVX2 `mask_blend` without the per-lane loop (U5), then U4 (sign plane → ±1 bytes) on top of it | high | biggest measured AVX2 cost (75–81 vs 10 ns/64); R2 burn-down item #1 |
| H1' | parity for the remaining 15 facade types, `U16x32` as the template | high | #385 shows it finds real bugs |
| H3' | AVX2 popcount: measure scalar-`popcnt` (vectorization blocked) against the nibble table; exclusion with the number if the table wins | high | N8 reframed |
| H5, H6, H2, H4, H7, H8 | as in §5 | high | unchanged; H5 still first for data safety |
| V9 | U7 / W1 / W6: recall for stored SM9 vs Lloyd-Max IDs on a real index | heavy | N11; TurboVec owner |
| V10 | the N14 ruling (named u8→i8 reinterpret) | heavy | operator |
| V11 | fused single-widen SM9 op for non-VNNI tiers; AVX-VNNI 256-bit and NEON `USDOT` arms; signed i8×i8 facade word | heavy | recorded loose ends of #386/#387, each needs a consumer |
| V1–V8 | as in §5 | heavy | unchanged |

### R2.5 Harvest ledger, this pass

| card | task | result | spot check |
|---|---|---|---|
| substrate-mapper | five U9 knowledge docs + board diff | DONE, all read | `u9-e5-probe.md:16-24`, blackboard U1–U7 rows re-read |
| substrate-mapper | `src/` diff (2,433 lines) + parity diff | DONE, 83 added `pub fn` | AVX2 `sum8_u8_add_masked` body re-read: `for h in 0..2` is the two-half split, not a per-lane loop |
| substrate-mapper | three lance-graph docs | DONE, all read | entry `:28-36`, `:555-572` re-read |
| census-counter | 13 counts | DONE | R2 = 157 re-run; `cpu_guard` path miss was my brief's error |

Misses: one, mine (wrong path in a brief). No card concluded or miscounted.

## R3. Revision after ndarray #395 (lance-graph unchanged)

ndarray `master` `03b3299`: one new PR, #395 (D-POF-ANT-0, "Position ×
Order × Law antennas", 11 files, +3,737, lab only: `examples/pof_antennas/`,
`tests/pof_antennas.rs` with 41 tests on 7 realizations and 14 disable runs,
report `.claude/knowledge/pof-antennas-probe.md`). lance-graph `main` is still
`61720fa` (#1848); nothing to revise there. Facet-stride guard: still 166.

### R3.1 What #395 found (verdicts as recorded in the report, `:301-310`)

| antenna | verdict | number |
|---|---|---|
| A1 shared basis | ADOPT-LAB | sharing a prepared basis wins without sharing a mask: 0.79–0.93× on AVX-512; loses on AVX2 (anomaly, OPEN) |
| A1 rebase | ADOPT-LAB (lazy) | invariant folds need no carry; an equivariant result needs ONE `conjugate_mask` of the result: 6.3 vs 16.4 ns on VNNI |
| A2 E2 | ADOPT-LAB on AVX-512 without VNNI | L1 end-to-end 4.69 vs E5 5.11 vs B 5.63 ns/64; 1.10× B on VNNI; 7× worse on AVX2 |
| A2 E3 | ADOPT-LAB | two consumers share preparation, keep separate accumulator laws: 0.63–0.94× on every tier |
| A6 scale | ADOPT-LAB | the field fold is tier-independent (6.1–8.0 µs/field); Morton equals row-major at 8 KiB |

It adds no facade word, trait or storage. Its contract section (`:312-324`)
records that LAW already lives in lance-graph mask-risc (ternlog
commutativity, per-terminal merge laws, `ExtentUnsupported`, row bounds) and
WITNESS in `OrderedLaneWitness` / `SealedFacetLane::validate`; nothing new is
prescribed.

### R3.2 Rows that moved

| row | was | now | evidence |
|---|---|---|---|
| H9 AVX2 `mask_blend` | priority #1 of R2 | **priority #1, third independent measurement** | #395 `:68`: "a per-lane loop on AVX2 (A2 E2: 73 vs 10 ns)"; it refuses E2 and the blended conjugation on v3 (`:417-418`). Three probes (#388, #389, #395) now lose on AVX2 to the same function |
| U9 kernel choice per tier | B everywhere; E5 ADOPT-LAB on v4 | **v4 has a better candidate** | E2 beats E5 and B on AVX-512 without VNNI in every row (`:415-416`); B stays the kernel on VNNI and AVX2 |
| P4 / lance-graph RPR X1–X9, #1848 B0–B11 | lance-graph's, unrun | lance-graph's, **with a ready oracle** | `:420`: "A5 is a scalar oracle that a B0 fixture could reuse"; A5 also pins #1848 B4 ("duplicate roots do not inflate support") at receipt level (`:267-269`) |
| everything else in R2.4 | — | unchanged | no other file in the delta |

### R3.3 New dots

**N16 — `conjugate_mask`'s slow step already has a fast part in the same
crate.** `Perm64::conjugate_mask` (`src/hpc/perm.rs:265`) builds its byte
flags with a scalar `core::array::from_fn` before one `permute_bytes`; #395
measured it at 4.4 ns (AVX-512) and ~60 ns (AVX2), and found a masked select
5.6× faster on AVX-512. The mask→bytes expansion #387 added for the masked
byte-dot (`simd_masking_ops::mask_byte_spread` + `BIT_IN_BYTE`, crate-private,
`:5311`) is exactly this step, and `perm.rs` is in the same crate. The fix
(fruit P1) should reuse it, not add a third expansion. Note `perm.rs` is
excluded from the facet-stride R1 rule for its `permute_bytes` (it is a real
move); the `from_fn` expansion is not covered by that exclusion's reason.

**N17 — the per-tier kernel choice now has three rows, which is the point to
write it down once.** VNNI: B. AVX-512 without VNNI: E2 (beats E5, which beat
B). AVX2: B, until `mask_blend` is fixed, after which E2 and E3 must be
re-measured there. Today this choice lives in three probe reports; whoever
integrates U9 (TurboVec owner, or lance-graph W1/W3) needs it as one table.

**N18 — #395 confirms the cross-repo split rather than adding to it.** Its
census (`:59-77`) places POSITION / LAW markers / EXPOSURE in ndarray and
ORDER / LAW tables / WITNESS in lance-graph, matching "ndarray = hardware,
lance-graph = thinking". It cites lance-graph #1835 (DR/DT ownership) and
#1848 without changing them. The open lance-graph items it names (`DurableRef`,
`ThoughtCommitId` PROPOSED; X1–X9 not run) are theirs.

### R3.4 Fruit, revised

Closed this pass: none (lab probe; nothing integrated).

| # | item | size | change |
|---|---|---|---|
| L8 | P1: `conjugate_mask` expansion via one masked select on AVX-512, reusing `mask_byte_spread`/`BIT_IN_BYTE`; keep the current path on AVX2 until measured | low | new (N16); production code in `perm.rs`, needs parity on all realizations |
| L9 | P3: AVX2 P1 anomaly (8 ns apart, 42 ns together): read the asm of the combined path | low | new |
| L10 | write the per-tier U9 kernel table (N17) into `turbovec9-u9-substrate.md` | low | new, doc only |
| H9 | AVX2 `mask_blend` without the per-lane loop | high | unchanged rank #1; now blocks E2/E3 on AVX2 too |
| H10 | P2: E2 as the v4 kernel, integrated where U9 is integrated | high | new; owner is the integrator, not ndarray |
| V12 | P4: lance-graph X1–X9 and #1848 B0 using #395's A5 oracle | heavy | new; lance-graph owners |
| all other L/H/V | as in R2.4 | — | unchanged |

### R3.5 Harvest ledger, this pass

No card dispatched: the delta was one PR, and its blackboard entry, verdict
table (`:301-310`), fruit list (`:409-421`) and `perm.rs:255-290` were read
directly on the main thread, which was cheaper than briefing a card. The
facet-stride count was re-run (166).

## R4. Two rulings implemented (operator, 2026-10-30)

| ruling | implemented as | closes / changes |
|---|---|---|
| u8→i8 only "as in explicit calling on intentional explicit parameter" | `ReinterpretBits` (zero-sized marker) and `U8x64::reinterpret_as_i8(ReinterpretBits)` / `I8x64::reinterpret_as_u8(ReinterpretBits)` in `src/simd.rs`; a bit reinterpret, no conversion; FACET STRIDE rule 3 amended in `CLAUDE.md` | V10/N14 decided. U6's facade gap is closed; the probe's `kernels::bits_as_i8` stand-in is not migrated (its timings stay reproducible). The 2 existing R3 hits are `u8` splats cast to an `i8` intrinsic parameter, not lane data, so they are not covered by this word and stay on the burn-down list |
| Codec9 runtime scale + batch encoder, "to enable both extreme accuracy LUT[a,b] and learning" | `Codec9Runtime<R>` in `src/hpc/codec9.rs`: checked scale, same encoder/decoder as `Codec9<R, C>` (equal scales give identical codewords), `encode_touched_batch` / `decode_batch` over magnitudes + presence plane; DR512 law doc amended | V1 and N4 decided. Compile-time `Codec9` stays the form for codeword tables; `Codec9Runtime` is the form for learned scales. Open: the batch loops are scalar (a SIMD batch is the next step once a consumer is timed); NSL has no ndarray dependency, so it either takes one or keeps its mirror behind a pairing test against these codewords (L3); lance-graph `scent_learn.rs:345-356` can replace its hand-built presence loop with `encode_touched_batch` (N5) |

Tests: 8 new in `codec9` (12 total), 4 in `simd::reinterpret_bits_parity`;
5 disable runs, each red (scale ignored, tail bits not cleared, write before
the finiteness check, saturating reinterpret, scale check removed). Lib suite
2,676 passed; AVX2 tier, aarch64 build and the wasm/NEON parity programs pass.

## R5. lance-graph #1850 + #1851: the ONNX graph through the DT64 registry

lance-graph `main` `f2bbd27`. Two merges, both in the new lab crate
`crates/tract-harvest-probe` (its own workspace, excluded, ~2.1k lines): an
ONNX graph laid onto one 64×64 DR#3 tile and each node resolved through an
ON × WHEN × DO registry (one lab classid per op signature, `0x0F01..0x0F51`).
#1851 adds two guards (run inputs checked against the resolved facts; a `Pow`
exponent may not grow the base). No ONNX PR is open. The rows below were read
in `registry.rs`, `value.rs`, `plan.rs`, `Cargo.toml` and
`lance-graph-onnx/src/lib.rs`; ndarray symbols were checked in this tree.

**Closed by #1850 (no ndarray work):**

- **N19 — every DO is an existing ndarray word.** The registry adds no
  kernel. Each DO is a `lance-graph-onnx` alias, and every alias names an
  ndarray function: `Greater` → `simd::gt_i32_to_mask`
  (`simd_masking_ops.rs:344`), `Where` → `simd::blend_i32` (`:5827`), `Abs`/
  `Sqrt` → `hpc::vml::vsabs`/`vssqrt` (`vml.rs:509`, `:485`), `ReduceMean` →
  `ArrayBase::mean_axis`. The dependency direction is right: the consumer
  names ndarray words and grows none.
- **N20 — `tensor(bool)` is the ndarray mask word.** `value.rs:3-6` decodes a
  source bool tensor once into "bit `i % 64` of word `i / 64`", the layout every
  `*_to_mask` kernel writes. A comparison output feeds `blend_i32` with no
  repacking. This is the second consumer (after `lance-graph-onnx`) to adopt
  the convention by name.
- **N21 — one ndarray package in a Tract graph.** Tract's workspace patches
  crates.io `ndarray` to this fork; the probe repeats the `[patch]`
  (`Cargo.toml:31-35`), so Tract and the DOs share one `ndarray` package
  identity. Note it pins `branch = "master"`; a change here reaches it on its
  next `cargo update`.

**Loose ends, with the ndarray word that would close each:**

The probe's measured coverage on Jina v5 small: 3,751 of 7,620 nodes resolve;
3,700 have no registration, the largest groups being Unsqueeze, Shape, Gather,
Concat and f32 MatMul.

| id | gap in the probe | existing ndarray word | ndarray work |
|---|---|---|---|
| L11 | `Unsqueeze`, `Shape` unregistered | `insert_axis`, `.shape()` (views, no copy) | none |
| L12 | `Concat` unregistered | `ndarray::concatenate` | none |
| L13 | `Gather` unregistered (axis gather, int indices) | `ArrayBase::select(axis, &[usize])` | none; the i64 → usize index check and negative-index wrap belong in the WHEN |
| L14 | f32 `MatMul` unregistered | `ArrayBase::dot` / the backend `sgemm` | none; but nothing pins `dot` to ORT's summation order, so until measured it registers as Bounded, like `ReduceMean` |
| L15 | f16 `Cast` refused as `Unregistered` | `simd::cast_f16_to_f32_batch` / `cast_f32_to_f16_batch` (`simd.rs:760-762`) | none; RNE on the narrowing cast must match ORT (`f32_to_f16_batch_ieee754_rne` exists) |
| L16 | `Erf` refused | none: `hpc::vml` has no `vserf` | **H11** below |
| L17 | i32 DOs copy their input (`x.iter().copied().collect()`) and `Where` with a scalar Y fills a full `Vec` | the kernels already take `&[i32]`; `as_slice()` on a standard-layout `ArrayD` borrows | lance-graph only; a `blend_i32_scalar(mask, a, b: i32, dst)` would remove the fill (**L18**) |
| L19 | `ReduceMean` is Bounded (max 4 ULP, 9,259 / 10,240 exact on the RMSNorm slice) | `mean_axis` reassociates | a sum with a pinned order that matches ORT would make it exact; probe first (**V13**) |

**Not a synergy, recorded so nobody tries it:** ONNX
`QuantizeLinear`/`DequantizeLinear` are not Codec9. ONNX int8 is
`[-128, 127]` with an affine zero point and no "never touched" state; Codec9 is
a 9-bit symmetric magnitude with presence. `Codec9Runtime` matches neither
range nor absence. The contract's `CodeSpace::from_zero_point` (#1839) is the
place where an ONNX quant annotation selects a code space; it already refuses
a 9-bit asymmetric lane. If the probe registers `DequantizeLinear`, its DO is a
plain affine decode, and an int8 tensor reaching a `U8x64` kernel crosses with
`reinterpret_as_u8(ReinterpretBits)`, never `as`.

**Fruit:**

- **Low.** L11–L13 and L15: registrations over existing words, in
  lance-graph. Each needs a WHEN (rank and shape facts) and an ORT parity
  test; no ndarray change. L17 is a lance-graph cleanup under the zero-copy law.
- **L18 (low, ndarray).** `blend_i32_scalar`: `Where(C, X, scalar)` is the
  common ONNX shape (the tiny fixture's `Where(c, X, 1)`). One masked blend
  against a broadcast register; every backend; parity test against
  `blend_i32` with a filled vector. Only if a second consumer wants it.
- **H11 (high, ndarray).** `vserf` / `vserf_slice` in `hpc::vml`: GELU-family
  models use `Erf`, and today the probe refuses it. Needs an accuracy bound
  stated in ULP against a reference (as `vsexp` does), the `±0`, `±inf` and
  NaN cases, and every backend.
- **L14 (high, lance-graph).** f32 MatMul is the largest unregistered compute
  op. Registering it over `dot` is easy; making it exact against ORT is not,
  so it registers as Bounded with a measured ULP bound.
- **V13 (heavy, probe).** Does ORT's `ReduceMean` (and MatMul) use a fixed
  summation order that an ndarray reduction could reproduce? If yes, an
  order-pinned reduction turns two Bounded methods exact. Measure ORT's order
  first; do not build the reduction before that.

**Haiku cards for the grindwork (§7 rules):** `census-counter` for the
unregistered op histogram per model (the probe's `examples/stages` output is
the source; count, never interpret); `harvest-scout` for every `ndarray::`
and `lance_graph_onnx::` name in `registry.rs` against `simd.rs` /
`hpc/vml.rs` exports, so a renamed ndarray word is caught before the probe's
`cargo update`; `quote-harvester` for ORT's reduction kernels when V13 starts.
Spot-check: re-read every hit used, re-run one count.

## R6. H11 and L18 implemented (2026-10-30)

The two ndarray-side gaps R5 named for the ONNX registry probe:

- **H11 → CLOSED.** `hpc::vml::vserf` (ONNX `Erf`): the Cephes `erff`
  polynomial for `|x| < 1`, `1 − exp(−x²)/x · P(1/x²)` up to 4, and `1.0`
  from 4. At most 2 ULP from the correctly rounded value over `[1e-38, 100]`,
  both signs, chunk and tail (tested against an `f64` series). It is exactly
  odd, keeps `erf(−0) = −0` and maps `±inf` to `±1`; NaN stays NaN. The Cephes
  `erfcf` polynomial for `x ≥ 2` turned out unnecessary for `erf` (a disable
  run that dropped it stayed green, and a prototype measured the same 2 ULP
  with one polynomial), so it was removed.
- **L18 → CLOSED.** `simd::blend_i32_scalar(mask, a, b: i32, dst)`: ONNX
  `Where(C, X, y)` with a scalar `y`, without the caller filling a `b`
  vector. Pinned against `blend_i32` with a filled vector over 131 lanes
  (two full words, a partial word, stray bits past the end).

Disable runs, each restored from the commit: sign restore removed (3 tests
red), NaN clamped by a `< 4` select (special-values test red), small branch
on `x` instead of `|x|` (2 red), scalar ignored in `blend_i32_scalar` (red).

What is left for the probe is lance-graph work: register `Erf` over `vserf`,
switch its `Where` DO to `blend_i32_scalar` when `Y` is a scalar (this also
closes half of L17), plus L11–L15. V13 stays a probe.

## 0. What the six ndarray PRs did

| PR | closed | opened or left |
|---|---|---|
| #378 | wasm `F32x16`/`F64x8::mul_add` round once: certified fast domain, local repair, exact fallback. `+relaxed-simd` tested and identical. Proof obligations in `wasm-fma-proof-obligations.md` | `reduce_sum` order still differs per realization; no mechanised proof |
| #379 | NaN-exclusion mindmap and premise audit (document) | S2 masked float fold, S3 `n_excluded` |
| #380 | Palette512 / SM9 scalar probe; note corrected (tests run, VPDPBUUD exists) | its separate SIMD gate |
| #381 | float lanes: presence IS NaN, held as a bijection (`not_nan_*_to_mask`, `nan_where_clear_*`, `presence_nan_mismatches_*`); SQL TRUE planes `{lt,…,ne}_{f64,f32}_to_mask`; `mask_not_under`; SM9 probe with the 3σ gate | the two-plane SM9 layout (superseded by #383) |
| #382 | FACET STRIDE rule, guard and exclusions; `simd_stride::{lut_u8, lut9}` | burn-down list: 166 ndarray hits, 10 lance-graph hits |
| #383 | DR512 P×M law: ONE presence/polarity bit (P=0,M=0 is the only NaN); `hpc::codec9` (`EpiLane9`, `Codec9::encode_touched`); `lut9` moved onto one plane; CI job `dr512/epistemic-presence` runs the 512-codeword, distance, WHERE-64K and ceiling oracles plus our probes | CausalEdge64 512-palette sidecar DEFERRED; LUT distance speed unmeasured |
| this PR | guard `--all` skipped nothing on dangling symlinks (34 `crates/burn/src/*.rs` → absent `../upstream`), now skips them (count unchanged at 166); `wasm-simd-consumer-guide.md:206` said wasm `mul_add` rounds twice; mindmap records #383 | — |

## 1. The base listing, point by point

### A. `fearless_simd` vs `ndarray::simd`

| point | 10-16 | now | evidence |
|---|---|---|---|
| compile-time dispatch, no per-kernel runtime dispatch, v4-on-AVX2 SIGILL, 512-shaped names, SSE tiers, AVX2 integer polyfills | CLOSED | CLOSED | no change |
| aarch64 guard | LOOSE | LOOSE | `cpu_guard.rs:318` still says it needs `getauxval(AT_HWCAP)`; the only `getauxval` call reads `AT_PHDR` (`:461-465`) |
| generated backends | PARTIAL | PARTIAL | — |
| per-level parity, 18 types | LOOSE | LOOSE, refined | of the 18, 16 never occur in `crates/simd-masking-parity/src/lib.rs`; `I16x32` occurs 7 times, `I8x64` once (counted, not judged as coverage) |
| MSRV prose | LOOSE | LOOSE | lance-graph `ISSUES.md:472` `ISS-NDARRAY-CANONICAL-COORDINATE-COUPLES-FLEET-MSRV`: **Status: OPEN** |
| one binary everywhere | EXCLUDED | EXCLUDED | — |

### B. "Three places the CPU leaks back"

| point | now | evidence |
|---|---|---|
| the build cares; runtime dispatch | CLOSED | — |
| consumers inherit no config | CLOSED for lance-graph; LOOSE for lance-graph-java | lgj `native/lgj-abi/.cargo/config.toml:35` still says "one known host"; `check_build_cpu` has 0 hits on lgj `main`. The v4 hard-code was not re-read |

### C. Accelerate split — PARK, unchanged. ### E/F (SSE slow paths) — unchanged.

### D. Unsafe inventory

Unchanged: missing `// SAFETY:` comments, CONSOLIDATE / SAFE-API rows, the
SAFE-CRATE (`bytemuck`/`zerocopy`) question. `is_x86_feature_detected!`:
96 sites in 17 files (counted today), still unclassified into
startup / once / per-call.

### Findings X1–X19 and TD rows

| id | 10-16 | now | evidence |
|---|---|---|---|
| X1, X7, X18 | CLOSED | CLOSED | X18 re-read: OGAR takes `encryption` from ndarray `branch = "master"` (`ogar-encryption/Cargo.toml:15`), the named-branch form the substrate law allows |
| X2–X4 ladybug | LOOSE | **UNKNOWN** | a search on ladybug-rs `main` for `grep avx512f`, `x86-64-v2`, `config-v2` found nothing. Search is not evidence of a fix; read the image selector before closing |
| X6 v4 Railway images | DECIDED | DECIDED | q2 `Dockerfile:227` still `-C target-cpu=x86-64-v4`; MedCare-rs: no `x86-64-v4` string found in Dockerfiles/toml/yml (not read; may have moved) |
| X11 `mul_add` | LOOSE (doc pair contradicted) | **CLOSED** | `simd.rs:1319` "fused on every arm" now matches `simd_wasm.rs:37-40` (emulated, rounds once). The last stale doc, `wasm-simd-consumer-guide.md:206`, is fixed in this PR. lance-graph CP-3 (harvest 10-27) asked for a "parity statement": parity group `check_mul_add_single_rounding` plus the proof document is that statement |
| X16 consumer raw intrinsics | LOOSE | LOOSE, **merged with the FACET STRIDE lance-graph hits** | lance-graph CP-2 (10-27) ADOPTED: `ndarray_bridge.rs` raw blocks sit under `cfg(not(feature = "ndarray-hpc"))` (13 occurrences), a non-default fallback. `TD-SIMD-SWEEP-W2` HALF DONE: `types.rs` migrated, `ndarray_bridge.rs` 66 raw lines left. The guard's 10 lance-graph R1 hits are exactly those blocks (8) plus holograph `hamming.rs:593-594` (2, live) — one burn-down, not two |
| X17 forbidden remedies in records | PARTLY CLOSED | PARTLY CLOSED | still open: lichess-rs `CLAUDE.md:32,60`, `README.md:93-97`, `engine-adapter-stockfish/Cargo.toml:14,16` say `+1.95`; tesseract-rs `rust.yml:60` "ndarray's rust-version = 1.95 floor"; the MSRV issue |
| X19 lgj | LOOSE | LOOSE | see B |
| TD-NDARRAY-SIMD-POPCNT-NATIVE | (b) CLOSED | (b) CLOSED | (a) objdump and (c) the `Cargo.toml` comment not re-read |

## 2. The 10-16 dots D1–D8

| dot | 10-16 | now | evidence |
|---|---|---|---|
| D1 one set algebra, four spellings | LOOSE, hinge D-RA-2 | **CLOSED by decision and code** | D-RA-2 DECISION 2026-10-20 option A; D-CMR-1 shipped (#1644, relalg `normalize` delegates to `contract::set_algebra`); D-CMR-6 shipped (#1654, relalg `symbolic` delegates to `contract::set_rules`). crepe's join-body rewrites stay, by the decision |
| D2 one preimage op | CLOSED AS VIOLATION | CLOSED | — |
| D3 read masks × clustering × prefilter | LOOSE | not re-read | — |
| D4 per-row truth lane terminal | LOOSE | LOOSE | D-NALP-4 "Partial, shipped (#1548)": scalar law exact on 2³² rows; whether a lane terminal is needed is open |
| D5 one `Binder::code` | LOOSE | LOOSE | D-ASM-1b Queued, not authorized |
| D6 SAG-5 | LOOSE | LOOSE | D-SAG-5-0 … -10 Queued / docs |
| D7 two ndarray ids | CLOSED | CLOSED | — |
| D8 doc drift | PARTLY CLOSED | PARTLY CLOSED | closed: X11 pair, consumer guide (this PR). Open: X19 lgj comment, MSRV issue option, lichess/tesseract comments |

## 3. The 10-16 synergies S1–S8

| | 10-16 | now | evidence |
|---|---|---|---|
| S1 one register, five reasoners | LOOSE | LOOSE, **unblocked** | its prerequisite D-CGW-1 shipped (#1594); D-CGW-13 Queued |
| S2 wire gate = admission-as-mask | LOOSE | LOOSE, **new ndarray hook (N8)** | D-RPF-4 "Blocked (4 decisions; T1 LUT-weighted strided sum gap)" |
| S3 three devices, one PCS | LOOSE | LOOSE | `ISS-QUALIA-A2B-CLUT-UNMEASURED` open; D-CGW-14 Queued (modulators only) |
| S4 orientation, two readers | LOOSE | **CLOSED** | #1594: `causal_edge::Orientation` on 43..45; Moore `lift_ce64` refuses an oriented edge, so there is one table |
| S5 Binder is the one bind surface | LOOSE | LOOSE | D-ASM-1b Queued |
| S6 one address minting path | LOOSE | not re-read | — |
| S7 `syllogize` writes 50..52 as `0b111` | LOOSE | LOOSE | the D-CGW-1 row says "46..49 (D-CGW-9) and 50..52 still open"; `syllogize` now lives in `edge_v3.rs` / `isa.rs`; whether 50..52 is still written as `0b111` was not read |
| S8 triad readers to migrate | LOOSE | **CLOSED** | #1594 removed the S/P/O triad ("no production writer") |

## 4. New synergies since then (the dots between arcs)

**N1 — `lut9` and Codec9 have their first consumer, and it adds no kernel.**
lance-graph `planner/src/nars/scent_learn.rs:49-50` imports
`ndarray::hpc::codec9::EpiLane9` and `ndarray::simd::lut9` (D-EPI-COT C3c,
TEST-PINNED). Its learning rate is the `lut9` table; its delta is built in
code space with the same zero rule `lut9` applies to its output. The
dependency direction is the right one: the consumer names ndarray words and
grows none.

**N2 — one P×M rule, three homes, two with a pairing test.** The rule
"absent = P=0 AND M=0; 9-bit P is also the sign; touched zero is +0" lives in
ndarray `codec9.rs`, in lance-graph-contract `lane_codec.rs` (`EpiLaneSpec`,
presence plane mandatory, a 9-bit lane must be Sym), and in the
thought-layers probe (`write_epi`: a NaN lane is not written, merge on read
keeps the prior). The contract tests that it "agrees with `codec9` on all 512
codes" (`lane_codec.rs:993`). The third copy, `lance-graph-nsl/src/eval.rs:3-7`,
mirrors Codec9 with a runtime scale and has no such test (the crate is
dependency-free by design).

**N3 — the float bridge is one mask expression.** A DR512 lane's float
presence is `P | ne_u8_to_mask(M, 0)`; its complement is "never touched".
That is exactly the plane #381's `nan_where_clear_f32` consumes, so decoding a
DR512 lane to f32 keeps #381's bijection with no new primitive. The probe test
`decode_satisfies_the_float_bijection` already pins this end to end.

**N4 — NSL wants a runtime-scale, batch Codec9.** D-NSL-W0W1 kept "SM9 as a
storage format for neighbor input features" as a PROBE (9-bit 2–4× better
than int8) that "needs Codec9 runtime scale, batch encode, and a Cora-scale
run first". ndarray's Codec9 is scalar and fixes the scale at compile time on
purpose (`codec9.rs:18-21`: "Storage, polarity, Sym and NaN policy are
intentionally NOT configurable … the only compile-time knobs are … rounding
mode and scale calibration"). A runtime scale is therefore a ruling, not a
patch. NSL also REJECTED SM9 edge weights as sort keys (top-1 changes for
7–21% of seeds); that verdict stays.

**N5 — the "touched presence" word.** `scent_learn.rs:345-356` builds
`P = present & (positive | (M == 0))` with a scalar loop over its features;
`lut9` applies the same rule internally. With a batch encoder (N4) this would
be the second writer of the same word; the facade could then name it
(`present & (positive | eq_u8_to_mask(M, 0))`, two mask ops). One consumer
today, so not yet.

**N6 — D-RPF-4's "T1 LUT-weighted strided sum gap" may already compose.**
ndarray now has `lut_u8` (facet stride) and `masked_strided_group_sum`. A
LUT-weighted strided sum is `lut_u8` into a weight lane, then the strided
group sum. Whether that meets D-RPF-4's need is unread; a probe answers it
before anyone builds a fused primitive.

**N7 — X16 and the FACET STRIDE burn-down are one job.** The 8 lance-graph
R1 hits are the dead fallback blocks of `TD-SIMD-SWEEP-W2`; deleting or
migrating them (to `U64x8::popcnt` / `ndarray::hpc::bitwise`) closes both
lists at once. holograph `hamming.rs:593-594` is the only live one.

**N8 — the popcount question the rule raises.** All 7 ndarray R1 hits and all
10 lance-graph R1 hits are the nibble-table popcount (`pshufb` over two 4-bit
halves). On AVX-512 VPOPCNTDQ replaces it; on AVX2 there is no vector
popcount, and the nibble table is the published fast method (Muła et al.).
The FACET STRIDE rule forbids a lookup built from shuffles. The options are a
per-u64 scalar `popcnt` (four per 256-bit register), or an exclusion with a
reason. This needs a measurement on an AVX2 host before either is chosen.

**N9 — the NaN erasers stay the open hazard.** `quantize_f32_to_u8`
(`hpc/quantized.rs:517`) and `quantize_f32_to_i8` (`:545`) still map NaN
through a saturating cast; 14 call sites in ndarray, 4 in lance-graph. Under
both rulings (float presence IS NaN, DR512 never-touched) they silently turn
NULL into a value.

## 5. Closed ends, loose ends, fruit

### Closed since 10-16

1. X11 and its doc pair (#374, #378, this PR).
2. D1 (D-RA-2 option A; #1644; #1654).
3. S4 and S8 (#1594, D-CGW-1).
4. H2 superseded: D-CGW-9 replaced by D-HL-1 O1 (46..49 = activation i4).
5. NaN exclusion on float lanes (#381) and integer lanes (#383, one plane).
6. FACET STRIDE rule, guard, audit (#382); guard symlink crash (this PR).
7. `lut9` / Codec9 adopted downstream without a new kernel (N1).
8. Palette512 probe corrected and kept as a CI tripwire (#380, #383).

### Excluded or rejected (do not reopen without new evidence)

- One binary everywhere (operator).
- Runtime dispatch of core types.
- Gather / preimage designs (D2, D-ASM-11/11A/11B, D-QMB-0…5: VIOLATION banners).
- SM9 edge weights as sort keys (D-NSL-W0W1).
- A second DR512 bit interpretation (Asym, Floor, NoNaN) for Codec9 (#383).

### Low-hanging (≤ 1 hour each)

| # | item | repo | check |
|---|---|---|---|
| L1 | lichess-rs `+1.95` (5 places) and tesseract-rs `rust.yml:60` comment → 1.99 | lichess-rs, tesseract-rs | doc only; needs push in those repos |
| L2 | lgj: name the artifact tier, drop "one known host", call `check_build_cpu()` in the handshake | lance-graph-java | needs push approval |
| L3 | NSL: a pairing test between its SM9 mirror and Codec9 semantics (512 codes, the -0.4/-0.5 ceiling cases), kept in NSL's own test data since the crate is dependency-free | lance-graph | the contract's `lane_codec.rs:993` test is the template |
| L4 | restore the facet-stride paragraph in `lut9`'s doc (lost in #383's rewrite) | ndarray | doc only |
| L5 | classify the 96 `is_x86_feature_detected!` sites (startup / once / per-call) as an inventory column | ndarray | grindwork for a Haiku census + Sonnet classification |

### High-hanging (a session each)

| # | item | why | gate |
|---|---|---|---|
| H1 | parity groups for the 16 never-named facade types | A | one flipped scalar op per group goes red |
| H2 | aarch64 guard via `AT_HWCAP` | A | dotprod build under `qemu-aarch64 -cpu cortex-a53` reports, no SIGILL |
| H3 | AVX2 popcount: scalar `popcnt` vs nibble table, measured, then migrate the 7 ndarray R1 hits or exclude them with the number as the reason | N8 | the faster one on a named AVX2 host; parity unchanged |
| H4 | lance-graph `ndarray_bridge.rs` dead blocks deleted, holograph `hamming.rs` onto `ndarray::simd` | X16, N7 | guard reports 0 for lance-graph; holograph Hamming tests unchanged |
| H5 | `quantize_f32_to_{u8,i8}`: refuse NaN or document the mapping; parity test that a NULL=NaN lane survives `Codec8 NaNAtMin` with zero mismatches | N9 | `presence_nan_mismatches == 0` after the round trip |
| H6 | probe: does `lut_u8` + `masked_strided_group_sum` cover D-RPF-4's T1 gap | N6 | equals D-RPF-4's reference fold on its fixture |
| H7 | D-CGW-13 first fixture (ONA + Java OpenNARS on one NAL fixture) | S1, now unblocked | two disagreeing oracles never project to equal edges |
| H8 | S7: read whether `syllogize` still writes 50..52 as `0b111`; if so, give 50..52 its one meaning | S7 | field isolation on 50..52 |

### Heavy-hanging (multi-session, owner or ruling needed)

| # | item | why |
|---|---|---|
| V1 | runtime-scale, batch Codec9 (SIMD encode into magnitudes + presence plane) | N4: a ruling on #383's "only compile-time knobs" |
| V2 | masked float fold with a fixed order, carrier `n, mean, M2` (mindmap S2), plus `n_excluded` (S3) | determinism ladder steps 2–3 |
| V3 | DR512 comparisons without decode (`cmp` on P×M as masks) | waits for a consumer; lance-graph lance13 map X14 parks u8 `Pred` until one asks |
| V4 | CausalEdge64 512-palette sidecar | DEFERRED by #383; lance-graph CE64 owner |
| V5 | per-classid label codebook behind `Binder::code` (D-ASM-1/1b) | D5, not authorized |
| V6 | D-NALP-5…7 | D4 |
| V7 | SAG-5 on D-CMR | D6 |
| V8 | Accelerate split, Kani, `bytemuck`/`zerocopy` | PARK |

## 6. Not searched

ladybug-rs image selector (search empty, not read); MedCare-rs deploy CPU
flags (search empty, not read); lgj `.cargo/config.toml` lines 1–34 and 36+;
S6 minting path; S7 `syllogize` body; D3; the 114 lance-graph merges beyond
the rows above; stockfish-rs, HubSPO-rs, spear, a2ui-rs, surrealdb (fetched,
not searched for these items; surrealdb's `ndarray = "0.17.2"` from crates.io
is unchanged).

## 7. Harvest ledger (Haiku cards)

| card | task | result | spot check on the main thread |
|---|---|---|---|
| substrate-mapper | map of #383's six files | 172 entries, DONE | `codec9.rs` read in full; `:56` `NEVER_TOUCHED`, `:148` `encode_touched` match |
| substrate-mapper | lance-graph consumers of SM9/Codec9 | 151 entries, PARTIAL (STATUS_BOARD too large to read whole; searched) | `scent_learn.rs:49-50` imports and `:345-356` zero rule re-read |
| harvest-scout | NaN / presence / shuffle candidates in lance-graph | DONE, but ran on a stale local checkout (128 commits behind `main`) | its 10 `shuffle_epi8` hits match the guard's 10 lance-graph hits |
| census-counter | guard counts, lance-graph grep counts | DONE; exposed the guard's symlink crash | crash reproduced; fixed in this PR |
| census-counter | X-item status across 9 repos (18 commands) | DONE | `simd.rs:1319`, consumer guide `:206`, `cpu_guard.rs:318` re-read |
| quote-harvester | six lance-graph board entries | DONE, 6/6 | NSL `:20-23` decisions re-read |
| census-counter | board status rows for 20 IDs | DONE | `STATUS_BOARD.md:364,372,494,594`, `ISSUES.md:472`, `TECH_DEBT.md:1947,2274` re-read |

Misses: one (the stale-checkout scout; my brief named the local checkout
before I had fetched `main`). No card miscounted or concluded.
