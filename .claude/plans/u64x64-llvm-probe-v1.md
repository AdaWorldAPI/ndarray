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

## 11. Results, routes B and E (2026-11-10)

**Status:** MEASURED, all five routes on v2, v3 and v4 (hot, cold L2, cold
LLC; three processes each). `check`: 38 entries, 27,960 cases, plus 2,539
route E cases, all exact on every tier. Disable runs B1–B3 and E1–E4 each go
red on the intended op (E2 needed its patch fixed for operator precedence;
E4's test was rewritten because O6 does not read `q`, so the planned
q=2/q=3 sequence could not catch a memo that ignores its parameters).

### 11.1 Route B (bit-sliced), hot

| op | v4 | v3 | v2 |
|---|---|---|---|
| O1 | runner-up: C 64.4 vs B 86.0 | B best (325 vs A 530) but NO-ADV: B's max 497 > A's min 445 | **BITSLICE-WINS** B 323 vs A 428 |
| O3 | C 5.9 vs B 38.5 | — | — |
| O5 | C 5.4 vs B 55.4 | — | — |
| O4, O4a | B 2.1 µs and 3.4 µs: the slowest entries | | |

B rebuilds bit planes on every call. It is competitive only on O1, where the
inputs are already bits, and wins only where SIMD is weakest (v2). Cold, it
wins nothing. The pre-registered "O5 B wins on dense masks" is not met.

### 11.2 Route E (prepared), against the best unprepared entry

| | v4 hot | v3 hot | v2 hot | v4 l2 | v3 l2 | v2 l2 | any llc |
|---|---|---|---|---|---|---|---|
| O2 on prepared planes | never | never | never | never | never | never | never |
| O6 on prepared planes | never | never | never | never | never | never | never |
| O6 memo, identical query | **from N = 128** | **from N = 128** | from N = 1024 | from N = 128 | from N = 128 | from N = 1024 | never |

Preparing all 64 categories costs 1.9 µs (v4) to 21 µs (v2), and a query on
prepared planes is no cheaper than B rebuilding only the demanded categories
(v4: O2 29 vs 27 ns, O6 240 vs 254 ns), so plane reuse cannot pay. Only
repeating an identical query through the memo does: a hit is 26 ns on v4.
A stale reuse is refused in 0.5 ns (`E_STALE_CHECK`), so S5's cost is that
check plus a fresh prepare. E1 + C on i32 input costs 2.46 µs per call (v4),
dominated by the narrowing and the bind.

### 11.3 Verdict

Across all five routes, per op and tier:
- **SIMD (C) is the default on AVX-512**: it wins O1, O3, O4a and O5 hot on v4
  and is never beaten there.
- **Scalar (A) is the default on v3 and v2 for O3 and O6**, because AVX2's
  `U16x32` is array-backed and v2 has no SIMD backend; a cold cache widens A's
  lead.
- **Bit-slicing (B) wins one cell**: O1 on v2, hot.
- **Prepared reuse (E) wins only as a memo of identical queries**, from
  N = 128 on v4/v3 and N = 1024 on v2, and never with a cold LLC.
- **A0 is never a candidate**: on O6 it is 13× C on v4.

No route is a universal winner; the choice is per op and per tier, and a
cold cache erases most differences. Open: `tests/u64x64.rs`, llvm-mca
validation, callgrind, working sets 64/1024.

## 12. Second full run after two review fixes (2026-11-11)

**Status:** MEASURED. A second set of all 27 runs (v2/v3/v4 × hot/l2/llc × 3
processes), after two Codex fixes on #419:

- Every route-E query timing now black-boxes its output (`o2`/`o6`), as A–D
  do. The memo hit did not move: 21.6–34.6 ns, against 24.8–34.2 before.
- `scripts/u64x64-decide.py` no longer lets A0 be the winner, the reference,
  or the prepared-reuse comparison point. It was the reference for O2 in
  several cells of set 1 (e.g. v2 hot "C vs A0").

`check` stayed exact on every tier (38 entries, 27,960 cases; 2,539 route E
cases). All nine decision files were regenerated by the orchestrator,
byte-identical to the executor's.

### 12.1 Decisions, set 2 (✱ = verdict differs from set 1, §10–§11)

| op | v4 hot | v3 hot | v2 hot | v4 llc | v3 llc | v2 llc |
|---|---|---|---|---|---|---|
| O1 | NO-ADV B 83.0 vs C 88.7 ✱ | BITSLICE-WINS B 309 vs A 448 ✱ | BITSLICE-WINS B 317 vs A 645 | SIMD-WINS C 1334 vs B 1615 | NO-ADV ✱ | NO-ADV |
| O2 | NO-ADV C 7.7 vs A 7.8 | SIMD-WINS C 8.9 vs A 13.2 ✱ | SIMD-WINS C 6.8 vs A 41.6 | NO-ADV | SIMD-WINS C 1116 vs B 1344 ✱ | SIMD-WINS D-loop 1295 vs A 1850 |
| O3 | SIMD-WINS C 8.7 vs B 37.4 | NO-ADV | NO-ADV | NO-ADV ✱ | SCALAR-WINS A 1447 vs C 1909 | NO-ADV ✱ |
| O4 | NO-ADV | NO-ADV | NO-ADV ✱ | NO-ADV | NO-ADV ✱ | SCALAR-WINS A 2351 vs D 2900 |
| O4a | SIMD-WINS C 260 vs A 518 | SIMD-WINS C 333 vs A 805 | NO-ADV | NO-ADV ✱ | SIMD-WINS C 2674 vs A 3269 ✱ | NO-ADV |
| O5 | SIMD-WINS C 7.9 vs B 53.8 | SIMD-WINS C 44.0 vs A 97.1 | NO-ADV ✱ | NO-ADV | SCALAR-WINS A 1099 vs C 1377 | NO-ADV ✱ |
| O6 | NO-ADV A 170 vs C 199 | SCALAR-WINS A 173 vs C 517 | SCALAR-WINS A 164 vs C 862 | NO-ADV | SCALAR-WINS A 1941 vs C 5545 | SCALAR-WINS A 1651 vs C 4353 |

Route E, set 2:

| | v4 hot | v3 hot | v2 hot | v4 llc | v3 llc | v2 llc |
|---|---|---|---|---|---|---|
| O2 / O6 on prepared planes | never | never | never | never | never | never |
| O6 memo, identical query | from N = 128 | from N = 128 | from N = 1024 | from N = 128 ✱ | from N = 128 ✱ | never |

v4 hot route E: prepare 1.65–2.22 µs, O2 17–91 ns, O6 206–401 ns, memo hit
21.5–51.8 ns, stale refusal 0.5 ns, `bind` 239–354 ns.

### 12.2 What survives both sets

A verdict that flips between two identical runs is not a result. Stable in
both:

- C wins O3, O4a and O5 hot on v4, and O4a and O5 hot on v3.
- A wins O6 on v3 and v2, hot and cold.
- B wins O1 on v2 hot.
- E's prepared planes never win; the memo of an identical query wins hot
  from N = 128 on v4 and v3 (N = 1024 on v2).
- A0 is never a candidate (now enforced by the script).

Everything marked ✱ moved between sets, including both v4 hot O1 (C won in
set 1, B is best but not admitted in set 2) and the cold-LLC memo verdict.
§11.3's "C is never beaten on AVX-512" does not hold for O1 and O6 in set 2
and is withdrawn for those two ops. Per-cell ranges overlap the decision
rule's margin often enough that a third set, or a quieter host, is needed
before any ✱ cell is cited.

## 13. Route F: the 64×64 as an X,Y shader grid (2026-11-11, PLANNED)

**Question (operator):** run the 64×64 as an ISA contract over X,Y, with no
materialization and no Abakus, like a shader grid, with any building done
through `simd_stride`.

**Reading.** X = row r (64 cells, the lanes), Y = category k (64 rows of the
grid). The grid is the source's own `ValueTile<u16>`, read in place. A shader
is a per-cell function of `(y, x-row, uniforms)`; the dispatcher maps it over
the demanded Y and folds the result. Nothing is built from source values: no
bit planes (B, E) and no Abakus. What a shader may prepare is a **method
constant** keyed by its parameters, never by source data: here, one
`simd_stride::lut_u8` table.

**What F tests, and what it does not.** Route C already executes the same
instructions as a hand-written loop over Y with X in lanes. F changes only the
contract: one generic dispatcher plus small shaders, monomorphized. So:

- **H-F1 (zero cost):** for O2, O3 and O6, F's median is within the decision
  rule's margin of C's on v2, v3 and v4, and `scripts/u64x64-inspect.sh`
  reports the same instruction classes (vdata, sdata, popcnt, shuf, call = 0)
  within ±10% of C's counts. **Falsified** if F is slower beyond the margin
  on any tier, or adds a hot-loop `call`.
- **H-F2 (stride building):** O6 with its X-uniform cap
  `staffcap[r] = min(staff[r]·u, 128)` built as a 256-entry `lut_u8` table
  per query (`f/lut`) is not faster than C's `mullo` + `simd_min` (`f/-`).
  Pre-registered direction: LUT ties or loses on v4/v3 (a 256-byte build per
  query against two vector ops), may win on v2. This is the only place in the
  six ops where a byte-to-byte per-cell function exists: `⌊S/d⌋` takes a u16
  S, and `d·q·C` needs a u16 output, so neither fits `lut_u8`.

**Not built, recorded:** a native-width X tile on v3 (four `U16x16` per row,
not the array-backed `U16x32`) would be the shader grid's real advantage over
C on AVX2. It is blocked by the facade: on AVX2 `U16x16` has `splat`,
`from_slice`, `mullo`, `shl`/`shr`, bit ops and `reduce_sum`, but no
`cmpgt_mask`, `cmpeq_mask`, `simd_min` or `mask_blend` (read in
`src/simd_avx2.rs`, `impl U16x16`). Adding them is `src/` work under the W1a
contract on every backend, outside a lab PR.

### 13.1 Spec (verbatim for the drafting worker)

New file `examples/u64x64/route_f.rs`. Visibility change only in
`route_c.rs`: `P16`, `join`, `load16`, `widen_u8`, `lane16`, `quot8` become
`pub(crate)`. F reuses C's per-cell arithmetic on purpose, so the measured
difference is the contract alone.

```rust
/// One Y row of the grid: 64 X cells as two U16x32 halves.
pub(crate) type Row = route_c::P16;

/// A per-cell function folded over the demanded Y rows.
pub(crate) trait Shader {
    type Acc: Copy;
    fn init(&self) -> Self::Acc;
    /// `row` is grid[y], loaded in place. Returns the folded accumulator.
    fn cell(&self, y: usize, row: Row, acc: Self::Acc) -> Self::Acc;
}

/// Maps `s` over every set bit of `ys` in ascending y, reading
/// `grid[y]` in place.
#[inline(always)]
pub(crate) fn dispatch<S: Shader>(grid: &ValueTile<u16>, mut ys: u64, s: &S) -> S::Acc {
    let mut acc = s.init();
    while ys != 0 {
        let y = ys.trailing_zeros() as usize;
        ys &= ys - 1; // F1: every demanded y is visited exactly once.
        acc = s.cell(y, route_c::load16(&grid[y]), acc);
    }
    acc
}
```

Kernels (`#[no_mangle] #[inline(never)]`, signatures identical to route C):

- O2 writes one output word per visited y, so it uses `ShaderOut`:

```rust
/// A shader that also writes one output word per visited y.
pub(crate) trait ShaderOut: Shader {
    fn out(&self, y: usize, row: Row, acc: Self::Acc) -> (Self::Acc, u64);
}

#[inline(always)]
pub(crate) fn dispatch_out<S: ShaderOut>(
    grid: &ValueTile<u16>, mut ys: u64, s: &S, out: &mut [u64; 64],
) -> S::Acc {
    let mut acc = s.init();
    while ys != 0 {
        let y = ys.trailing_zeros() as usize;
        ys &= ys - 1; // F1
        let (a, w) = s.out(y, route_c::load16(&grid[y]), acc);
        acc = a;
        out[y] = w;
    }
    acc
}
```

  O2 implements `ShaderOut` with `Acc = u64` (`all`, init `u64::MAX`);
  `out` computes `ge` exactly as `u64x64_o2_c` (t > 65535 → 0), returns
  `(acc & ge, ge)`. The kernel sets `out.ge = [0; 64]` first, then
  `out.all_ge = dispatch_out(&bound.src.stock, demanded, &sh, &mut out.ge)`.

- `u64x64_o3_f(bound, demanded, out)`: `Shader` with
  `Acc = (P16 m, U8x64 arg, u64 have, u64 unknown)`, init and per-cell body
  copied from `u64x64_o3_c`'s loop body (presence read from
  `bound.src.present[y]`), finish identical to C after `dispatch`.
- `u64x64_o6_f(bound, params, out)`: `Shader` with
  `Acc = (P16 cap, u64 missing, U8x64 first_miss, U8x64 lim, u64 limset)`;
  init computes `staff_raw` and `cap` exactly as C; `cell` is C's loop body
  with `k = y` and `s = row`; the Y set is `params.recipe.demanded()`
  (ascending, the same order as C's `demanded_list`). Finish
  identical to C. `staff_raw` is a shader field.
- `u64x64_o6_f_lut(bound, params, out)`: identical to `o6_f` except the
  initial `cap`. Build `cap8: [u8; 256]` with `cap8[s] = min(s·u, QMAX)` (fits
  u8, QMAX = 128), call `ndarray::simd::lut_u8(&src.staff, &cap8, &mut c8)`
  into a `[u8; 64]`, and set `cap = widen_u8(U8x64::from_slice(&c8))`.
  `staff_raw` stays C's `mullo` (a u16, up to 255·255), because the `STAFF`
  limit test compares against it.
  F2 anchor: `// F2: the table entry is min(s*u, QMAX), not s*u.`

`entries()`: route `"f"`, class `Class::Simd`; variants `""` for o2/o3/o6
and `"lut"` for `o6_f_lut`. Registration: `mod route_f;` and
`v.extend(route_f::entries());` in `main.rs`.

**Worker rules:** no `cargo` of any kind, no git; no byte shuffle, no
`as i8`, no scalar loop except the 256-entry table build and the existing
C finish loop (FACET STRIDE); no allocation; `// F1` and `// F2` anchors
exactly as written. Do not claim it compiles.

### 13.2 Orchestrator steps (not the worker's)

1. `decide.py`: class `'f'` → Simd, and a direct contract line per op:
   `F/C ratio` with both ranges, verdict `ZERO-COST` iff the ranges overlap or
   the medians are within 10% of each other, else `CONTRACT-COST` (F slower)
   or `F-FASTER`. (F and C share a
   class, so the ordinary rule never compares them.)
2. Build v2/v3/v4, `check` exact; disable runs F1 (after `ys &= ys - 1`,
   add `if ys.count_ones() == 1 { ys = 0; }`: the last demanded y is never
   visited) and F2 (table entry `s*u` without the min with QMAX) each turn
   `check` red.
3. `inspect.sh` counts for `o2/o3/o6_{c,f}`; 27 bench runs (Haiku
   executor); decisions regenerated by the orchestrator.
4. Results as §14; PR body table.

### 13.3 The question, sharpened (operator, 2026-11-11)

> Can direct, coordinate-aware cell evaluation outperform route C without
> constructing bit-planes, once preparation and reduction costs are counted?

§13.1's F reuses C's per-cell arithmetic, so it can at best tie C. It stays
as the **contract control** (F0): it measures what the X,Y contract itself
costs, so any gain below is not the dispatch form. The arm that answers the
question is **F-coord**:

- **Coordinate used:** the Y uniform `d[y]`. Per y, decided once per y and
  never per cell, a power of two `d = 2^p` takes `q = min(s >> p, 255)`, which
  equals `quot8`'s largest `q ≤ 255` with `q·d ≤ s` exactly; any other `d`
  takes `quot8` (8 restoring steps). Anchor F3.
- **Preparation counted:** the class decision runs inside the kernel on every
  call. No per-recipe table is cached, so nothing is hidden outside the timer.
- **Reduction counted:** the min/argmin fold and the finish are the same as
  C's and inside the timed kernel.
- **No bit planes, no Abakus,** values read in place.

**Isolation:** the gain depends on the share of power-of-two `d`, so it is
measured at both extremes as well as on the bench recipe:

| row | recipe | expectation |
|---|---|---|
| `ISO_POW2_{C,FCOORD}` | d = 1, 2, 4 | the largest gain F-coord can show |
| `ISO_GEN_{C,FCOORD}` | d = 3, 5, 7 | F-coord ≈ C: only the class test is added |
| `O6 f/coord` vs `O6 c` | `random_recipe(1)`, d ∈ 1..3 | in between |

- **H-F3:** on v4 hot, `ISO_POW2_FCOORD` ≤ 0.85 × `ISO_POW2_C` with
  non-overlapping ranges, and `ISO_GEN_FCOORD` is ZERO-COST against
  `ISO_GEN_C`. **Falsified** if the power-of-two case does not clear the
  margin (coordinate awareness does not pay even where it can), or the
  general case is CONTRACT-COST (the class test costs more than it saves).
- **What a win would and would not mean:** the gain comes from a per-y
  uniform that C could also branch on. A win shows that evaluating with the
  coordinate's uniform beats a coordinate-blind uniform kernel. It does not
  show a property of the shader-grid contract; F0 measures that separately.

## 14. D-U64X64-SHADER-F-0: separated cell evaluation (2026-11-11, PLANNED)

**Question (operator):** can a direct per-cell function `f(x, y, v, params)`
outperform route C without constructing bit planes, with exact semantics and
with reductions separated from cell evaluation? NO-ADVANTAGE is an allowed
outcome.

**Baseline.** #419 merged at `9683379`. Its post-review corrections (A0 never
a candidate; route-E outputs consumed) are in #420, **not yet on master**, so
this work is stacked on #420's branch and is not valid against bare master.

**Three phases, timed separately.**

| phase | what | runs |
|---|---|---|
| preparation | method constants keyed by their parameters (the cap LUT; the per-y division class) | once per parameter set; amortised over N |
| cell | read the value in place, compute, write a caller-owned output grid | per run |
| reduction | fold the cell grid into the answer | per run |

The cell grid is caller-owned output (stack, no heap). Its size per op is
stated below and is the only new memory. Nothing is copied from the source.

### 14.1 Per-op classification

| op | reads per cell | class | cell output (bytes) | reduction |
|---|---|---|---|---|
| O1 | A[y], B[y], uniform L | CELL + REDUCTION | BitTile `I = A&B&L`, 512 | per-y popcount, per-x vertical count, total |
| O2 | S[y], uniform t_y | CELL + REDUCTION | `ge[y]` = `out.ge`, 512 | AND over demanded y |
| O3 | S[y], P[y] | CELL + REDUCTION | `v[y] = S[y] \| !P[y]` for demanded y, 128·|D| | min/argmin over y (strictly smaller updates; ties keep the smaller y), `have`, `unknown` (reads P again: stated dependency) |
| O4 | C[y], uniform w_y | CELL + REDUCTION | two u16 product rows `C·(w&0xFF)`, `C·(w>>8)` per y with w ≠ 0, 256·|W| | C's `Col16` carry sums over y, then the u32 finish (C's scalar loop: the facade has no u16→u32 widening, §10.3) |
| O4a | X[y], Y[y] | PURE CELL | `z`, `ovf` (the output itself) | none; F ≡ C, not duplicated |
| O5 | counters[k], uniform (b, m) | CELL + REDUCTION + TRANSACTION, one Y row | the summed row, 128 | any-carry over x; write back only if none |
| O6 | S[y], P[y], uniform d_y; X uniforms staff, gate | CELL + REDUCTION | quotient rows `q[y]` for demanded y, 128·|D| | cap = min over y with the limit tie rule, missing/first-miss, finish |

### 14.2 Variants (route `f`, all registered in `entries()` so `check` covers them)

- `split` for O1, O2, O3, O4, O5, O6: `cell` into the caller grid, then
  `reduce`. Both are separate `pub(crate)` functions; the entry kernel
  calls them in order with a stack grid.
- `split_coord` for O6: the cell uses the per-y class (power-of-two `d`
  shifts, others `quot8`), §13.3.
- `split_lut` for O6: the reduction's initial cap comes from a prepared
  `CapLut`. The entry kernel prepares it per call (N = 1); the bench rows
  time preparation and use separately.
- The fused variants of §13 (`""`, `coord`, `lut`) stay as the comparison
  for "reduction inside the cell".

**LUT eligibility.** Only `staffcap = min(staff·u, 128)`: u8 in, u8 out
exactly. Not eligible: `⌊S/d⌋` (u16 S), `d·q·C` (u16 out), O3/O4 values
(u16). Nothing is truncated to fit.

`CapLut { u: u32, table: [u8; 256] }`: `prepare(u)`, and
`cap0(&self, staff: &[u8; 64], u: u32) -> Result<P16, LutRefusal>` refusing
`LutRefusal::ParamMismatch { prepared, asked }` when `u != self.u` (anchor F4).

### 14.3 Falsifiers (beyond `check`, which already pins parity, presence,
permutation, overflow, ties, rollback, tails and sparse masks for every
registered entry)

- `cap_lut_equals_mullo_min_for_every_u_and_staff` (exhaustive 255 × 256).
- `cap_lut_prepared_for_one_u_refuses_another` (F4 disable turns it red).
- `split_cell_writes_only_demanded_rows`: a grid filled with a sentinel;
  after `cell`, exactly the demanded rows differ, and each equals the per-cell
  arithmetic (no other row is materialised).
- Disable anchors: F4 (LUT parameter check), F5 (O3 reduction: strictly
  smaller update), F6 (O5 write-back only if no lane carries).

### 14.4 Bench rows (Fixed rows, every timed closure consumes its output)

`FS_O1_CELL`, `FS_O1_RED`, `FS_O2_CELL`, `FS_O2_RED`, `FS_O3_CELL`,
`FS_O3_RED`, `FS_O4_CELL`, `FS_O4_RED`, `FS_O6_CELL`, `FS_O6_CELL_COORD`,
`FS_O6_RED`, `FS_O6_RED_LUT` (with a LUT prepared outside the timer),
`LUT_PREP`. End-to-end per variant comes from the ordinary entry rows.
`decide.py` reports F-vs-C for every route-f variant, and the LUT
amortisation `(LUT_PREP + N·(FS_O6_CELL + FS_O6_RED_LUT)) / N` against O6 C
for N ∈ {1, 2, 8, 32, 128, 1024}.

### 14.5 Decision

GO only if a split variant beats C on at least one op and tier under the
§8 rule in both of two full sets, with parity and every disable red. LAB
ONLY if it ties or wins only in one set. NO GO if it loses everywhere;
then the measured counterexample is the result.

## 15. Results, route F (2026-11-11)

**Status:** MEASURED, two full sets (2 × 27 runs: v2/v3/v4 × hot/l2/llc × 3
processes, `taskset -c 2`). `check`: 51 entries, 38,428 cases, all exact on
every tier, plus 2,539 route-E cases. 23 example tests pass. Disable runs
F1 (both dispatchers), F2, F3 (fused and split), F4 (unit test), F5 and F6
each turn their check red. The first F1 attempt was vacuous: its inserted
code landed inside a `//` comment. It was caught and redone, never counted.
All 18 decision files were regenerated by the orchestrator, byte-identical.

### 15.1 F against C (ratio set 1 / set 2; F = F-FASTER, = ZERO-COST, C = CONTRACT-COST)

| variant | v4 hot | v4 l2 | v4 llc | v3 hot | v3 l2 | v3 llc | v2 hot | v2 l2 | v2 llc |
|---|---|---|---|---|---|---|---|---|---|
| O2/- (contract control) | = 1.08/1.00 | = | = | = | = | = | = | = | = |
| O3/- (contract control) | = 1.06/0.96 | = | = | = | = | = | = 0.99/0.90 | = | = |
| O6/- (contract control) | = 0.95/0.95 | = | = | = | = | = | = | = | = |
| **O6/coord** | **F 0.56/0.63** | **F 0.81/0.80** | = 0.87/0.89 | **F 0.36/0.38** | **F 0.68/0.66** | **F 0.57/0.58** | **F 0.44/0.44** | **F 0.69/0.71** | **F 0.72/0.74** |
| O6/lut | = / C 1.10/1.12 | = | = | = | C / = | = | = | = | = |
| O1/split | **F 0.81/0.81** | = | = / F | = | = | = | = | = | = |
| O2/split | = | = | = | = | = | = | = | = | = |
| O3/split | C 9.63/9.61 | C | C | C | = / C | C | C | = / C | C |
| O4/split | C 1.58/1.59 | C / = | C | = | = | C | C | = / C | C |
| O5/split | = | = | = | = | = | = | = | = | = |
| O6/split | C 1.11/1.29 | C | C | = | C / = | = / C | C | = | = |
| O6/split_coord | F / = 0.84/0.95 | C | C | F 0.45/0.49 | F | F | F 0.55/0.57 | F | = |
| O6/split_lut | C 1.42/1.60 | C | C | = / C | C / = | = / C | C | C | C |
| ISO POW2 (d = 1, 2, 4) | **W 0.49/0.49** | = / W | W | W 0.30/0.30 | W | W | W 0.30/0.29 | W | W |
| ISO GEN (d = 3, 5, 7) | = 0.95/0.96 | = | = | = | = | = | = | = | = |

### 15.2 Against the best existing route (the §8 rule)

| | v4 hot | v4 l2 | v4 llc | v3, v2 (all three) |
|---|---|---|---|---|
| O6 | **F-coord 81.3 vs A 173–177 vs C 145, both sets** | NO-ADV (A ≈ F-coord) | NO-ADV | **A wins**: v3 hot 159–162 vs F-coord 219–220; v2 hot 174–179 vs 380 |
| O1 | **F-split 50.6 vs B 88–90 vs C 62.5, both sets** | NO-ADV | set 2 only (not stable) | B or A wins |

### 15.3 Phases (hot, set 1, ns)

| | v4 cell / reduce | v3 cell / reduce | v2 cell / reduce |
|---|---|---|---|
| O1 | 6.3 / 46.0 | 8.5 / 2276 | 10.1 / 1840 |
| O2 | 1.8 / 0.9 | 4.6 / 1.0 | 4.7 / 1.0 |
| O3 | 2.5 / 3.7 | 33.6 / 66.9 | 31.2 / 86.2 |
| O4 | 146 / 228 | 138 / 971 | 229 / 679 |
| O6 (quot8 cell) | 36.9 / 86.8 | 315 / 220 | 546 / 385 |
| O6 (coord cell) | **8.3** / 86.8 | **12.5** / 220 | **50.0** / 385 |

LUT: `LUT_PREP` costs 11.1 ns (v4) to 19 ns (v2). Applying the LUT is about
13 ns slower than C's `mullo` + `simd_min` (v4: `FS_O6_RED_LUT` 92.5 vs
`FS_O6_RED` 79.9). Amortised over N = 1–1024 it never reaches C or the best
route.

### 15.4 Findings

- **The X,Y contract costs nothing.** The O2/O3/O6 controls are ZERO-COST in
  every cell. O2 and O3 compile to C's machine code byte for byte (v4:
  242/59 and 329/66 bytes/instructions).
- **Coordinate awareness pays where a Y uniform admits a cheaper exact
  formula.**
  - On v4 the O6 cell drops from 36.9 to 8.3 ns once a power-of-two `d`
    shifts instead of dividing.
  - With no power-of-two `d` it costs nothing (ISO GEN ZERO-COST in all 18
    cells).
  - The gain is bounded by the share of power-of-two divisors: d ∈ 1..3 in
    the bench recipe.
  - This is a specialization of the quotient, not a property of the shader
    form. Routes A and C could adopt the same per-row branch; A with it was
    not measured.
- **Separating the reduction from the cell costs, except where the output is
  the grid.**
  - O3/O4/O6 split variants are slower than both their fused forms and C.
  - The O3 split entry is 49 ns, against phases summing to 6.2 ns. The entry
    zeroes an 8 KiB stack grid with `memset` (the disassembly shows the
    call), and that zeroing dominates.
  - O2 (output = `ge`) and O5 (one row) are ZERO-COST.
  - O1 split wins on v4 hot (0.81, both sets) and nowhere else.
- **The cap LUT never pays.** It is the only `lut_u8`-eligible computation,
  and two vector ops beat a table build plus a scalar-position lookup.
- **On v3/v2, scalar A remains the O6 winner.** F-coord beats C there, but
  not A.
- **Disassembly:** every F kernel on v4 uses zmm, with no hot-loop call
  except the split entries' `memset`. On v3 F inherits C's array-backed
  `U16x32`.

### 15.5 Decision

- **LAB ONLY** for route F as a route.
- **NO GO** for:
  - a shader-grid ISA or type: the contract is free but adds nothing;
  - separating reductions by default;
  - the cap LUT.
- **Promotable, pending a consumer measurement:** one operation, the
  Y-uniform divisor class inside the O6 quotient (power-of-two `d` → shift).
  It is no-regret on the measured tiers and halves O6 on AVX-512 hot.
  - Consumer: Foundry G6 `CanCook` (O6 is its G1 capacity).
  - Gate: measure the actual recipe divisor distribution first, since the
    gain is proportional to it.
  - Owner: route C's quotient (`quot8`). It should not get a new API.

**R5** is filed separately as lance-graph #1998, with a red-first test on
branch `ccr-8062bcc7-l7lkz9-r5-mapframe`.

## 16. D-QUOT8-ADMISSION-0: does the power-of-two quotient earn admission? (2026-11-11)

**Question (operator):** does the per-row power-of-two specialization that
route F measured (§15) give a measurable end-to-end benefit for Foundry G1's
actual recipe populations and workloads? Route F's design is not reopened.

**G1 vs G6.** G1 is capacity (`NetworkCapacity`, per-store max, the
min/argmin limiting resource). G6 is `CanCook(q)` executed from a registered
`MethodRef`. In lance-graph's `foundry-parity-lab` both execute as threshold
sweeps: `arm_b::cook_ops` compares stock against `d·q` (`Pred::GeI32 { t: d *
q }`), and `max_quantities` / `network_capacity` sweep q = 1..Q_MAX. **No
quotient is evaluated on the executable path.** `S / d` occurs only in the
oracle (`oracle.rs:62`) and the materialized null model (`arm_a.rs:75`).

### 16.1 Counterfactual arms (lab only, `examples/u64x64`)

- `c/pow2`: `u64x64_o6_c` with one change. Per demanded row y, a
  power-of-two `d` takes `min(S >> p, 255)`; any other `d` takes `quot8`.
  This is F-coord's selection inside C's own loop, with no shader dispatch.
  Anchor F7.
- `a/pow2`: `u64x64_o6_a` with the same per-row choice in the scalar
  division (`s >> p` for a power of two, `s / d` otherwise). Anchor F8.
- Both are registered entries, so `check` pins exact parity. Disable F7/F8
  (test `d > 0` instead of `is_power_of_two`) must turn `check` red.

### 16.2 Populations (never combined)

- **Hypothetical sensitivity:** 8 demanded categories at
  y ∈ {3, 11, 19, 27, 35, 43, 51, 59}. The first k rows take power-of-two
  `d` from (1, 2, 4) cycling, the rest non-powers from (3, 5, 7) cycling,
  for k = 0..8, i.e. 0%–100% in 12.5% steps. Rows `SENS_{C,CP,A,AP}_{k}`.
- **Foundry fixture (synthetic):** the 8 menu items of
  `foundry-parity-lab::fixture::recipe(m)`, demands mapped one to one (category
  index, `d`). Rows `FND_{C,CP,A,AP}_{m}`. This is the only recipe population
  that exists, and it is synthetic: PRODUCTION_DISTRIBUTION = UNKNOWN.

Every timed closure consumes its output. Recipes are built outside the
timer.
