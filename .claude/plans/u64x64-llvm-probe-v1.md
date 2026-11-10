# D-U64X64-LLVM-1 — whole-tile compiler and Abakus probe (plan v1)

**Status:** PLAN. Nothing is built yet. **Question:** can LLVM produce a faster
exact kernel for a whole 64×64 tile whose values are already locally
addressable? **Coordination:** lance-graph DR/DT64 (the tile is a DR64 tile);
the Foundry P1/G1 algebra work stays in its own session and is only read here
for value ranges and the G1 operation.

**Constraint:** no public `U64x64` and no `src/` edit. A type is admitted only
once a consumer and a repeatable advantage exist.

Prior art reused: `examples/microtile/`, `.claude/knowledge/microtile-probe.md`,
`.claude/knowledge/binary-abacus-polyfill-probe.md`, `src/simd_stride.rs`,
`src/simd_masking_ops.rs`, `ndarray::simd::{U64x8, U8x64, half_add_mask, full_add_mask}`.

## 0. Facts the plan depends on (read in source on 2026-11-10)

- Tests are auto-discovered. `tests/microtile.rs` includes example code with
  `#[path = "../examples/microtile/tile.rs"] mod tile;` behind
  `#![cfg(feature = "std")]`. Examples are registered in `Cargo.toml`
  (`[[example]]` with `required-features = ["std"]`). Edition 2021, so plain
  `#[no_mangle]` is accepted (`examples/bounds_check_elim.rs:35`).
- Toolchain: rustc 1.99.0 (LLVM 23). `/usr/bin/llvm-objdump` and
  `/usr/bin/llvm-mca` are LLVM 18.1.3, a version behind. `llvm-tools` is not
  installed. valgrind 3.22.0 is present. `perf` is absent and there is no `cpu`
  PMU, so hardware counters are unavailable.
- Host: Xeon @ 2.10 GHz, 4 vCPU; L1d 48 KiB/core, L2 2 MiB/core, LLC 260 MiB;
  15 GiB RAM; avx512bw/vl, avx512_vpopcntdq, bmi2, popcnt.
- Tiers: `.cargo/config-v2.toml` selects `simd_scalar.rs`
  (`src/simd.rs:409-417`, `sse4.2 && !avx`); config-v3 gives AVX2; config-v4
  gives AVX-512 and sets `-Dwarnings`.
- Facade APIs (AVX-512 arm, `src/simd_avx512.rs`):
  - `U8x64`: `splat`, `from_slice`, `to_array`, `cmpeq_mask`, `cmpgt_mask`
    (unsigned), `mask_blend(mask,a,b)` (b where bit set), `simd_min`,
    `sum_bytes_u64`, `Add`/`BitAnd`/`BitOr`/`BitXor`.
  - `U16x32`: `splat`, `zero`, `from_slice`, `to_array`, `copy_to_slice`,
    `from_u8x64_lo/hi` (zero-extend), `shr`/`shl` (runtime count), `mullo`,
    `simd_min`, `cmpeq_mask`, `cmpgt_mask` (unsigned), bit ops, `Not`, `Add`,
    `Sub`.
  - `U64x8`: `cmpgt_mask` (unsigned), `mul_lo32`, `sum8_u8_add_masked`,
    `popcnt` (vpopcntdq only under `cfg(target_feature="avx512vpopcntdq")`;
    the fallback is UNVERIFIED).
  - Masking ops: `half_add_mask`, `full_add_mask`
    (`src/simd_masking_ops.rs`, re-exported in `src/simd.rs`),
    `ge_i32_to_mask`, `mask_and`, `masked_min_i32` (reduces over rows, so it
    cannot do a per-row min across categories).
- Facade gaps, recorded as findings and **not fixed** here: no
  `U16x32::mask_blend`; no u16→u32 widening; no `mulhi` at any width;
  `U32x16` has no `simd_min` and no `cmpgt`.
- On AVX2, `U16x32` is array-backed: `shr`/`shl` are `for i in 0..32` loops
  (`src/simd_avx2.rs:2787-2800`). Route C on v3 is therefore partly a facade
  loop plus LLVM, and the results must say so.
- Two existing laws are reused by copying, because their modules cannot be
  included: `carry_after(low,p,sum) = ((low & p) | ((low | p) & !sum)).shr(15)`
  (`examples/u9_abacus/carry16.rs:25`) and `Abacus::try_add_at`
  (`examples/u9_domino/domino.rs:330`).
- Foundry value ranges (`lance-graph crates/foundry-parity-lab/src/fixture.rs`):
  stock i32 20..79 with a STALE value of 1000 in unknown slots; `unit_cost`
  50..249 cents; demand 1..3 on ≤3 categories; `per_staff` 4..8; staff 0..3;
  q 1..=128.

## 1. File layout (a lab fixture only)

| file | responsibility |
|---|---|
| `Cargo.toml` (+4 lines) | `[[example]] name = "u64x64" path = "examples/u64x64/main.rs" required-features = ["std"]` |
| `examples/u64x64/fixture.rs` | value and output types, the lab `Source`, the frame `Witness`, `bind()` and its refusals, deterministic generators, the falsifier corpora, the Foundry-shaped i32 source |
| `examples/u64x64/oracle.rs` | independent scalar oracle for every op: u64/u128 arithmetic, one bit at a time, imports only fixture types |
| `examples/u64x64/route_a.rs` | A: runtime-dimension plain Rust over borrowed planes; A0: existing mask-RISC primitives over the i32 source |
| `examples/u64x64/route_b.rs` | B: bit-sliced — local `Abakus16`, CSA vertical counter, MSB-first comparator, bitsliced constant multiply, min/argmin, capacity |
| `examples/u64x64/route_c.rs` | C: SoA over `U8x64`, `U16x32`, `U64x8`, with a local `Col16` on the `carry_after` law |
| `examples/u64x64/route_d.rs` | D: const-generic specialization — `loop`, `u8`, `full` (unrolled 64), `const_sparse`, `const_dense` |
| `examples/u64x64/route_e.rs` | E: `Prepared<T>` with a full witness — E1 narrows i32→u16, E2 builds u16 bit planes; invalidation; the forbidden-memo stub used by a disable run |
| `examples/u64x64/harness.rs` | median timing, cache sweep, counting `#[global_allocator]`, bytes model, tier report, single-kernel mode for callgrind |
| `examples/u64x64/main.rs` | modes `check`, `bench <scenario>`, `one <symbol> <n>`, `sizes`; an entry table built from each route's `pub const ENTRIES` |
| `tests/u64x64.rs` | exactness of every entry against the oracle on every corpus, the refusals, the identities, the disable-run anchors |
| `scripts/u64x64-inspect.sh` | read-only analysis of built `.s`/`.ll`/binary into per-symbol count tables; no cargo |
| `.claude/knowledge/u64x64-llvm-probe.md` | results, written after measurement |

Every kernel is a monomorphic `#[no_mangle] #[inline(never)] pub fn
u64x64_<op>_<route>[_variant]` with signature `(&Bound, &Params, &mut Out) ->
Result<(), Refusal>`. `Out` is preallocated by the caller.

## 2. Semantics

**Geometries.** 64×64 bits = `BitTile = [u64; 64]`: word k is category k, bit
r is row r (512 B). 64×64 values = `ValueTile<u16> = [[u16; 64]; 64]` for
stock, `[k][r]` (8 KiB; one plane = 128 B = 2 `U16x32` loads). Cost is
`ValueTile<u8>` (4 KiB; one plane = one `U8x64`). One orientation for both,
the Foundry source orientation.

**Widths.** u16 for stock (STALE 1000 and q·d ≤ 384 do not fit u8); u8 for
cost (50..249). The lab `Source` owns u16/u8 planes, so A–D read them with
zero copy. The i32 source exists for A0 and E1; narrowing it is a
*preparation* that refuses negatives and values > 65535.

**Admission bounds** (identical for every route, enforced by
`bind`/`Params::new`): `d_k ≤ 255`, `staff_r ≤ 255`, `u ∈ 1..=255`,
`QMAX = 128`, `q ∈ 1..=128`, `t_k ∈ 0..=65536`. Recorded as compromises (C's
restoring division needs `d·cand ≤ 255·255`); every Foundry value fits.

**Frame and refusals.** `Witness { source_id, version, row_order: [u16;64]
(+hash), shape, fields }`. `bind(&Source, &Expect)` refuses `Stale`,
`RowOrder`, `Shape`, `Field`, `ForeignSource`, `DemandOutOfRange` before any
plane is read. Kernels take `&Bound<'a>`, borrowed and never copied.

**Operations** (the oracle for each is literal per-element code over (k, r)):

- **O1 masked count/intersection (bits).** A, B: BitTile; live L.
  `plane_cnt[k] = popcnt(A[k]&B[k]&L)`; `row_cnt[r] = #{k : bit r of A[k]&B[k]}`
  for r ∈ L else 0; `total = Σ plane_cnt`.
- **O2 exact threshold.** `ge[k]` bit r = `S[k][r] ≥ t[k]` for k ∈ D;
  `all_ge = ⋀_{k∈D} ge[k]` (all ones if D is empty); unsigned; t = 65536 means
  never.
- **O3 presence-aware min/argmin per row.** Min over k ∈ D with `P[k]_r`;
  `arg` is the smallest k attaining it; `none` → `min = 0, arg = 0xFF`;
  `unknown` bit r iff ∃k∈D with ¬P[k]_r. STALE slots are never read as values.
- **O4 weighted row sum.** `sum[r] = Σ_k w_k·C[k][r]` as u32;
  `total = Σ_{r∈L} sum[r]` as u64.
- **O4a checked add.** `Z = (X+Y) mod 2^16`; `ovf[k]` bit r = `X+Y ≥ 2^16`.
- **O5 repeated Abakus updates.** 64×64 counters of 16 bits. Update
  `(k, b, m)`: b ≥ 16 refuses `SignificanceOutOfRange`; if ∃r∈m with
  `c[k][r] + 2^b ≥ 2^16`, refuse `Overflow{lanes}` and leave the state
  unchanged; else add. Stream of 4096 updates. O5r = readout.
- **O6 CanCook capacity (Foundry G1).** Per row r: gate G_r = 0 → Denied;
  else ∃k∈D with ¬P[k]_r → Unknown (`unknown_k` = smallest); else Known with
  `cap = min(min_{k∈D} ⌊S/d_k⌋, staff·u, 128)` and `limit` = first of
  (k ascending, STAFF=64, CEILING=65) equal to cap.
  `CanCook(q)_r ⇔ Known_r ∧ cap_r ≥ q`. O6n: `NetworkCapacity = Σ_Known cap_r`,
  asserted equal to `Σ_{q=1}^{128} Count(CanCook(q))`;
  `cost_r(q) = Σ_{k∈D} d_k·q·C[k][r]`; `network_cost(q) = Σ_{r∈CanCook(q)} cost_r(q)`.

## 3. How each route expresses each op

| op | A native | B bitslice | C SIMD SoA | D LLVM-specialized | E prepared |
|---|---|---|---|---|---|
| O1 | `count_ones`; row_cnt by bit tests | CSA tree of 64 words (`full_add_mask`/`half_add_mask`) into 7 planes, readout via `U8x64::mask_blend` | `acc += U8x64::mask_blend(A&B, 0, 1)`; plane_cnt via `U64x8` & `popcnt` | A with `const R,K`, const L | NOT-ADMITTED (input is already minimal) |
| O2 | u32 loops | MSB-first comparator over E2 planes | `!(splat(t).cmpgt_mask(v))` per half, `lo \| hi << 32` | const t, D | E2+B on reuse; E1+C on i32 |
| O3 | loops, strict `<` | bitsliced pairwise min | absent → 0xFFFF via `mask_blend` + widening; `upd = m.cmpgt_mask(v')`; `arg = mask_blend(upd, arg, k)`; `m = m.simd_min(v')` | const dims (pre-registered: LLVM will not vectorize first-index argmin) | NOT-ADMITTED (D is a program parameter) |
| O4 | u32 loops | add C plane at b+j per set bit of w_k into a 32-plane Abakus | two `Col16` accumulators on `mullo` of w's low and high byte | const w | NOT-ADMITTED |
| O4a | loops | 16-plane `full_add_mask` ripple | `sum = x+y`; `carry_after`; `cmpeq_mask(1)` | const dims | NOT-ADMITTED |
| O5 | check, then add over set bits | `Abakus16` ripple with transactional undo | `inc = widen(mask_blend(m,0,1)).shl(b)`; commit only if no carry | const BITS, dims | NOT-ADMITTED (mutation) |
| O6 | runtime `/`, min, ordered limit | bit-serial cap over E2 planes, shift-add multiply, MSB-first compare | restoring quotient b=7..0 with `mullo` and a borrow bit; then O3's min/argmin; staffcap = `staff.mullo(u).simd_min(128)`; cost via O4 | `loop/u8/full` with runtime recipe; `const_sparse` / `const_dense` via `trait Recipe { const ID; const D: [u16;64]; }` | E2 planes reused across recipes and q |
| baseline | **A0**: `ge_i32_to_mask` + `mask_and` on i32 planes; O6 by the Foundry layer-cake (128 programs) | — | — | — | — |

Rules for every route: no `shuffle_bytes`/`permute_bytes`/`transpose8` (FACET
STRIDE R1); no i8; planes read in place; only demanded categories are
iterated, so bytes read scale with |D|.

## 4. LLVM inspection (command cards)

The orchestrator builds; Haiku runs only the read-only steps 3–5.

```sh
# 0. setup (UNVERIFIED until run)
rustup component add llvm-tools
rustc -vV; llvm-mca --version; llvm-objdump --version; valgrind --version; lscpu
rustc --print cfg -C target-cpu=x86-64-v4 | grep target_feature
llvm-mca -mcpu=help 2>&1 | grep -i sapphire

# 1. one tier at a time, shared target/, never in parallel
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0
for T in v2 v3 v4; do
  rm -f target/release/examples/u64x64-*.s target/release/examples/u64x64-*.ll
  touch examples/u64x64/main.rs
  env -u RUSTFLAGS cargo --config .cargo/config-$T.toml rustc --release --example u64x64 \
      --features std -- --emit=llvm-ir,asm,link -C debuginfo=0
  # copy binary, .s, .ll out to $SCRATCH/u64x64/$T, then run `check`
done
# 2. vectorizer remarks (v3, v4)
#    -C remark=loop-vectorize,slp-vectorizer,loop-unroll
# 3. per symbol: llvm-objdump --disassemble-symbols=...; llvm-nm -S --size-sort;
#    awk the .s and .ll bodies
# 4. llvm-mca -mcpu=sapphirerapids -iterations=100 on the hot loop body
```

`scripts/u64x64-inspect.sh <dir> <symbol>` prints per symbol: instruction
count; code bytes; zmm/ymm/xmm op counts; GPR ALU ops; `popcnt`; `div`/`idiv`
(expected only in A's O6); `vpmulhuw`/`imul`/`mulx`; gathers (expected 0);
byte-shuffle family (each hit classified reduction-internal or compute — a
compute hit violates FACET R1); mask-register ops; memory loads/stores;
spills; backward branches; `call` (must be 0); distinct plane base offsets
(for `const_sparse` this must be 3). IR is checked for `<32 x i16>`,
`<64 x i8>`, `<16 x i32>`, `udiv`, `@llvm.umin.v*`, `@llvm.ctpop.*`, loop
metadata.

An mca result is VALIDATED for a kernel when `cycles/iter × iters / 2.1 GHz`
is within ±30% of the hot measured ns, else NOT-VALIDATED (the frequency is
nominal in a VM). mca never decides a verdict.

## 5. Benchmark harness

- **Timing.** One sample is R calls (R=1000 hot, R=1 cold), `black_box` on
  input and output. Median of 31 samples (41 cold), 3 processes per cell.
  Ranges across processes, in ns per tile and ns per row-answer (/64); O5 per
  update. Timer overhead calibrated with an empty entry; < 3× overhead is
  marked LOW-RES. Pinned with `taskset -c 2` if available.
- **Scenarios.** S1 hot; S2 cold-L2 (16 MiB sweep); S3 cold-LLC (640 MiB
  sweep, labelled "≥L2-cold" if the VM's LLC is not real); S4 repeated-query
  (N ∈ {1, 8, 128}); S5 changed-input (one cell mutated, version bumped, E
  re-prepares). Working sets 1 / 64 / 1024 tiles.
- **Verification** outside the timed region against the oracle; a mismatch
  aborts.
- **End-to-end** = bind + prep + kernel, each reported separately, plus
  invalidation cost.
- **Allocations**: counting `#[global_allocator]`; 0 inside every timed region
  is asserted; prep allocations reported.
- **Bytes**: analytic model per entry (planes actually read + scratch +
  spills), cross-checked with callgrind.
- **Counters fallback**: `valgrind --tool=callgrind --cache-sim=yes
  --branch-sim=yes` on v2 and v3 binaries only (SIMULATED; AVX-512 is
  expected to SIGILL under valgrind 3.22, UNVERIFIED). v4 uses static counts
  plus mca.
- **Code size** from `llvm-nm -S`.

## 6. Falsifiers and disable runs (`tests/u64x64.rs`, on v2, v3 and v4)

**Corpora:** zeros; all-ones; sparse single bits at rows 0/31/32/63 and
categories 0/63; random (4096 tiles); carry propagation (counter at 0xFFFF−2^b,
Col16 near 65535, row_cnt = 64); overflow (unchanged state on refusal,
0xFFFF+1, w = 65535 with C = 255); presence (STALE 1000, unknown in a
non-demanded category ignored, Foundry "store 09"); signedness (negative i32
refused by E1; 0x8000..0xFFFF ordered unsigned); row order (permuted source
refused; after re-bind `out(perm) == perm(out)`); invalid frames (stale
version, 63/65 rows, missing field, foreign source, d=256, u=0, q=0, q=129);
ties (equal values give arg 0; category vs staff; CEILING only when nothing
else equals 128); thresholds t ∈ {0, s−1, s, s+1, 65535, 65536}; capacity
(s < d gives 0; empty D gives `min(staff·u, 128)`); identities
(NetworkCapacity = Σ_q Count(CanCook(q)); network_cost(q) = oracle for
q ∈ {1, 2, 3, 64, 128}).

**Disable runs** (each must turn the suite red; restored with
`git checkout -- <file>` after committing):

| id | break | red test |
|---|---|---|
| B1 | CSA carry := 0 | row_cnt all-ones |
| B2 | Abakus16 no undo | overflow unchanged-state |
| B3 | comparator ge := gt | t = s |
| C1 | hi half mask not `<< 32` | sparse rows 32/63 |
| C2 | argmin `≥` | ties |
| C3 | skip absent substitution | presence/STALE |
| C4 | Col16 carry dropped | carry corpus |
| C5 | restoring borrow inverted | s = d·cand |
| C6 | drop `hi` accumulator | w = 600 |
| D1 | skip `Recipe::ID` check | mismatched-recipe refusal |
| E1 | skip version check | stale frame |
| E2 | skip row_order check | permuted source |
| E3 | prep key without field id | cost planes as stock |
| E4 | memo O6 keyed on frame only | P1(q=2) then P2(q=3) equal the oracle |
| W1 | bind skips shape | 63-row source |
| W2 | E1 skips sign check | negative stock |
| P1 | unknown demanded treated as known | store 09 |

## 7. Work breakdown

Haiku never runs cargo or benchmarks. Sonnet edits one file per chunk and
never compiles. The orchestrator compiles, measures and decides.

1. Commit this plan; add the `Cargo.toml` `[[example]]` block.
2. Haiku H0 (census-counter): run the §4 step-0 commands, report verbatim.
3. Haiku H1 (quote-harvester): for every method in §3, on the avx512, avx2
   and scalar arms, report `file:line` and whether the body is an intrinsic or
   an array loop. Any method missing on any tier makes its cell NOT-ADMITTED.
4. Sonnet S1 `fixture.rs`; S2 `oracle.rs`; S3 `harness.rs`; S4 `main.rs`;
   each checked by the orchestrator before the next.
5. Sonnet S5–S9: one route file each, order A (+A0), D, C, B, E; `check` on
   v4, v3, v2 after each.
6. Sonnet S10 `tests/u64x64.rs`; run on v2, v3, v4.
7. Sonnet S11 `scripts/u64x64-inspect.sh`; checked against one symbol by hand.
8. Disable runs (spec-editor applies each verbatim edit).
9. M1: builds and inspection for 3 tiers; Haiku H2 fills tables, H3 runs mca.
10. M2: benchmarks, 3 processes × scenarios × tiers; callgrind on v2/v3.
11. Verdicts by §8; knowledge file; blackboard entry (card runs, items
    checked, items wrong).

## 8. Pre-registered expectations (direction only) and decision rule

**Expectations:** O1 row_cnt C ≤ B < A on v4, B ≤ C on v3; plane_cnt
NO-MEASURED-ADVANTAGE. O2 C ≈ D; B on prepared planes wins only in S4 above
some N*. O3 C wins; LLVM does not vectorize D's argmin; B slowest. O4 D ≤ C.
O4a C ≈ D. O5 B wins on dense and single-lane masks (~4× precedent); B ≈ C on
random masks. O6 runtime recipe: C wins; all ≪ the 15 µs Foundry layer-cake;
A0 ≥ 10× C. O6 `const_sparse`: D fastest by a wide margin, labelled
**compile-time recipe only** and not compared with runtime-recipe rows.
D-full ≤ 15% faster than D-loop, ≥ 4× larger, spills on v3 for O3/O6. S3
shrinks differences toward bytes read. E wins nowhere except possibly O2 at
large N.

**Decision rule**, per (op, geometry, working set, scenario, tier):

- Candidates are admitted entries that passed exactness; a failed entry is
  FAILED-EXACTNESS and never a winner.
- Class: A scalar; B bitslice; C SIMD; E prepared; D is SIMD iff its hot loop
  has more vector than scalar data ops in the asm (the route tag stays in the
  cell, e.g. "SIMD-WINS (D)").
- W = lowest median of medians; R = best entry of a different class.
  **X-WINS** iff max(W across processes) < min(R across processes) and
  median(W) ≤ 0.85·median(R); else NO-MEASURED-ADVANTAGE.
- **PREPARED-REUSE-WINS** iff in S4 at stated N, (prep + N·query)/N passes the
  same test against the best unprepared entry; report N* or "never". In S5 E
  must lose; if it wins, that is a bug.
- **NOT-ADMITTED** cells are fixed now by §3 and cannot change after measuring.
- All cells are re-run together or not at all.
- A verdict holds for v3 and v4 only if it holds on each.

Verdict classes: SCALAR-WINS, BITSLICE-WINS, SIMD-WINS, PREPARED-REUSE-WINS,
NO-MEASURED-ADVANTAGE, NOT-ADMITTED.

## 9. Non-goals and risks

**Non-goals:** a public `U64x64`; any `src/` edit; new facade methods (gaps
go into the knowledge file); Foundry or lance-graph changes;
multithreading; NEON/wasm timing; per-category invalidation; IAM, scenarios,
writeback; memoization across programs (E4 is a disable run).

**Risks:** AVX2 `U16x32` is array-backed; llvm-mca 18 vs LLVM 23 asm; no
AVX-512 under valgrind and no PMU, so counters are simulated; VM noise and
unknown real frequency; `Instant` resolution at R=1; virtualized LLC;
`-Dwarnings` on v4 needs `#![allow(dead_code)]` for unused entries;
compile-time recipes do not exist in production; the global d ≤ 255 bound is
imposed by C; B on O3/O6 is expected to be very slow, which is a result, not a
reason to drop it; per-tier rebuilds cost time and disk, so artifacts are
copied out and tiers are never built in parallel.

## 10. Results, routes A / A0 / C / D (2026-11-10)

**Status:** MEASURED for A, A0, C and D on v2, v3 and v4. B and E are not
built yet, so nothing here is a verdict against them.

Method: `check` (31 entries, 23,585 cases, all exact on every tier), then
`bench hot` (R = 1000, 31 samples) and `bench l2` (R = 1, 41 samples, 16 MiB
sweep), three processes each, pinned with `taskset -c 2`, one tile. Builds
use `--emit=llvm-ir,asm,link`. Disassembly uses
`scripts/u64x64-inspect.sh <binary>`. The decision rule of §8 is
`scripts/u64x64-decide.py <dir> <scenario>`, which reads D's class from the
asm (vector data ops > scalar data ops).

### 10.1 Hot, decision per op (median of medians, ns)

| op | v4 | v3 | v2 |
|---|---|---|---|
| O1 | SIMD-WINS C 63.6 vs A 487 | SCALAR-WINS A 509 vs D 601 | SCALAR-WINS A 427 vs D 1239 |
| O2 | NO-ADV C 8.6 vs A 9.1 | NO-ADV | SIMD-WINS C 7.6 vs A0 30.2 |
| O3 | SIMD-WINS C 5.6 vs A 118.5 | NO-ADV C 101 vs A 115 | NO-ADV |
| O4 | NO-ADV D-loop 276 vs A 310 | NO-ADV | SIMD-WINS D-loop 504 vs A 612 |
| O4a | SIMD-WINS C 168 vs A 578 | SIMD-WINS C 336 vs A 833 | NO-ADV |
| O5 | SIMD-WINS C 4.9 vs A 106 | SIMD-WINS C 41 vs A 105 | SIMD-WINS D-loop 87 vs A 104 |
| O6 | NO-ADV C 153 vs A 170 | SCALAR-WINS A 177 vs C 555 | SCALAR-WINS A 187 vs C 928 |
| O6c (compile-time recipe only) | D const_dense 105 | D const_dense 102 | D const_dense 122 |

Cold L2 on v4 keeps SIMD-WINS for O1 and O3 only; every other op is
NO-MEASURED-ADVANTAGE, as pre-registered.

**A0, the independent baseline:** O6 1,968 / 2,200 / 3,529 ns on v4 / v3 / v2,
i.e. 12.9× C and 11.6× A on v4. The expectation "A0 ≥ 10× C" holds on v4. A0
is not a candidate on any op.

### 10.1b Cold LLC (640 MiB sweep, R = 1, 11 samples; labelled "≥L2-cold", the VM's LLC is not verified)

| op | v4 | v3 | v2 |
|---|---|---|---|
| O1 | SIMD-WINS C 1406 vs A 2275 | SIMD-WINS D-loop 1944 vs A 2299 | NO-ADV |
| O2 | NO-ADV | NO-ADV | SIMD-WINS D-u8 1082 vs A 1776 |
| O3 | SIMD-WINS C 1185 vs A 1554 | SCALAR-WINS A 1632 vs C 2703 | SCALAR-WINS A |
| O4 | NO-ADV | SCALAR-WINS A 1831 vs D-loop 2430 | SCALAR-WINS A |
| O4a | SIMD-WINS C 2647 vs A 3126 | NO-ADV | NO-ADV |
| O5 | NO-ADV | SCALAR-WINS A | SCALAR-WINS A |
| O6 | NO-ADV A 1854 vs C 1896 | SCALAR-WINS A 1738 vs C 7466 | SCALAR-WINS A 1957 vs C 6165 |

Cold, every kernel costs 1-5 µs and the v4 differences shrink to within 2%
except O1, O3 and O4a. A0 on O6: 4959 / 5060 / 6455 ns.

### 10.2 The rest of the hot path

| step | v4 ns |
|---|---|
| `bind` (witness, row-order hash, five checks) | 249–276 |
| O6 `Params::new` | 26–28 |

`bind` costs more than most kernels. It recomputes the 64-entry row-order
hash and compares the order on every call. So a per-query bind plus a
kernel is bind-bound: O3 C costs about 5.6 + 260 ns. The kernel numbers above
only hold for a binding resolved once and reused (the DR/DT64 locality
contract). A cheaper bind (a cached hash in the witness) is a separate
question and is not measured.

### 10.3 Disassembly findings (v4 unless stated)

- No kernel has a hot-loop `call`. The calls present are cold panic paths
  (`panic_bounds_check`, `panic_fmt`, div-by-zero), plus two real calls:
  `memset` in O6 A0 and `bcmp` in both O6c `const_*` entries.
- Byte shuffles, classified:
  - O1 C: nibble-table popcount (`vpshufb` + `vpsadbw`) inside the facade's
    popcount on a CPU without VPOPCNTDQ. Reduction-internal, the known
    FACET R1 burn-down class.
  - O3 C, O6 C: `lane16` (`x | x << 8` after a widen) lowered by LLVM to a
    byte-duplicating `vpshufb`. Cause: the facade has no
    `U16x32::mask_blend`, so the mask is built as bytes and widened.
  - O4 C: 32 xmm `vpshufb` from the scalar u16 → u32 recombination tail,
    auto-vectorized. Cause: the facade has no u16 → u32 widening.
- D-full against D-loop: O2 2.4× faster at 56× the code (32.8 KB vs 586 B);
  O6 11% faster at 16× the code. The pre-registered "≤ 15% faster" holds for
  O6 and fails for O2.
- LLVM does not vectorize D's O3 argmin (D-loop 2.2 µs, zero vector ops), as
  pre-registered. O4a D-loop compiles to the same code as A (868 B, identical
  counts), and A auto-vectorizes O2, O4 and O4a.
- v3 and v2: C loses O1 and O6 because AVX2's `U16x32` is array-backed and v2
  is the scalar backend. O6 C grows from 2.2 KB (v4) to 16.3 KB (v3), with
  533 scalar data ops. This is a facade realization result, not a property
  of route C.

### 10.4 Open

- Routes B and E; `tests/u64x64.rs` with the D1/E/W/P disables; llvm-mca
  validation; callgrind on v2/v3; the `llc` scenario; working sets 64 and
  1024.
- Facade gaps named in §10.3: `U16x32::mask_blend(u32, a, b)` and a u16 → u32
  widening. Both need the W1a contract on every backend.
  Checked against every backend file (src/simd_{avx512,avx2,scalar,neon,wasm}.rs):
  `mask_blend` exists only on `U8x64` (avx512:699, avx2:1561, scalar:1932) and
  `U8x32`; no `from_u16x32_lo/hi` exists anywhere. `simd_neon.rs` and
  `simd_wasm.rs` define neither `U8x64` nor `U16x32`; which realization
  `ndarray::simd` re-exports on those targets is not checked here, and route C
  has not been run on NEON or wasm.
- No winner is declared across routes until B and E are measured.
