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
