## 2026-09-16 (5) — why worker-side cargo is prohibited ABSOLUTELY: residue is the cost, BACKEND POLLUTION is the correctness failure

Operator, verbatim: *"Worker are prohibited from running cargo"*, and the reason:
*"Würdest du mit jedem worker kompilieren hättest du target residue backend
pollution."* The second half was not written down anywhere and is the stronger
argument, so it now sits in `.claude/rules/agent-cargo-hygiene.md` beside the
residue one.

The shared `target/` holds ONE realization at a time. A target-cpu change
invalidates the cache, so a worker's plain `cargo test` — which takes
`.cargo/config.toml`, i.e. **v3/AVX2** — run after the orchestrator built
`--config .cargo/config-v4.toml` **replaces** the v4 artifacts rather than
adding to them. The next probe reports whichever tier compiled last and says
nothing about which.

**This is plan §17's defect one scale down, and the severity differs.** On the
orchestrator's own runs a tier is merely UNLABELLED — a re-run under a named
config repairs it, which is exactly what happened today (273.9×–300.9× v3 →
110.2×–123.2× v4). With N workers compiling on their own schedule it becomes
UNATTRIBUTABLE: the interleaving is gone and no number can be traced to a
backend afterwards.

One shared `target/`, one realization, one compiler — the orchestrator. Workers
edit. Both worker briefs dispatched today carry the prohibition verbatim as
rule 1, with no carve-out for `test` or `clippy`.

Separately, operator correction to my own framing: **clippy is not compiling.**
I had read the rule file's "tests yes, compile no" as contradicting the absolute
prohibition on the grounds that clippy compiles. It does not — it type-checks and
lints. The rule file was left unedited on that point; only the missing second
reason was appended. Practical consequence for the orchestrator's gates: a green
`cargo clippy` proves types and lints, NOT that a runnable artifact builds. The
evidence for today's landings was the `cargo run --release` probe and parity
runs, not the lint.

## 2026-09-16 (4) — the AVX2 tier's `U8x64` byte compares were scalar loops beside a vectorized `U8x32` that already solved them

Found while scoping T1 gap G1 (the `u8` compare-to-mask family): on the v3/AVX2
arm `U8x64::{cmpeq_mask, cmpgt_mask}` built the `u64` one bit at a time, under a
comment calling itself a "scalar fallback", while `U8x32` in the SAME file
already carried `_mm256_cmpeq_epi8` + `_mm256_movemask_epi8` and an UNSIGNED
`cmpgt_mask` with the sign-bias XOR that AVX2 needs (it has only the signed
`_mm256_cmpgt_epi8`). Both now compose two halves: `(lo) | (hi << 32)`.

**Scope, and it decides where this is exercised:** this body compiles ONLY on
the v3 arm. Under x86-64-v4 `U8x64` is the native `__m512i` from
`simd_avx512.rs` and none of it is reached. Per the operator ruling recorded in
entry (3), v3 is not where this workspace MEASURES — it is where the
realization matrix requires every backend to stay bit-exact, which is the whole
reason to fix it rather than leave it.

Disable run (after the commit `3a5da8c`, so the restore could not eat it):
swapping the `lo`/`hi` halves fails 2 of 3 tests — the scalar-oracle test and
the lane-position test at indices 0/31/32/63. The sign-boundary test correctly
stays GREEN under that disable, because it tests signedness, not composition;
each of the three pins its own property and none of them is redundant.

Gates on the v3 arm: clippy `-D warnings` clean, fmt clean, 70 tests across the
two touched modules, masking parity 11 groups bit-identical.

## 2026-09-16 (3) — ⊘ every reveal-ratio number in this arc was v3/AVX2, including today's; the AVX-512 ratio is 110–123×, and CLAUDE.md was why

Operator ruling: *"It's unacceptable to use avx2 / of course you need to run
ndarray with x86 v4 avx512f."* Correct, and the reason I was not is a
documentation defect I have now fixed rather than worked around.

`CLAUDE.md:84` claimed `.cargo/config.toml` sets `target-cpu=x86-64-v4`
("AVX-512 mandatory"). It sets **v3**, and always has (`.cargo/config.toml:83`),
deliberately — v3 is the portable CI/distribution floor. So a plain `cargo run`
measures AVX2. Every number in §14, in the (5) STORNO, and in my own §16.2
entry earlier today is a v3 number that does not say so. CLAUDE.md now carries
the v4 invocation and the `env -u RUSTFLAGS` trap explicitly.

Same probe, same binary, same host (Xeon with the full AVX-512 set), verified by
the parity program's own `avx512f=` line:

| | v3 / AVX2 | **v4 / AVX-512** |
|---|---|---|
| TCAM sweep | 14 126–15 141 ns | **5 298–5 484 ns** |
| range write | 50–52 ns | 43–49 ns |
| reveal ratio | 273.9×–300.9× | **110.2×–123.2×** |
| ternlogq | 149.0 ns (0.146/word) | **111.6 ns (0.109/word)** |
| coal | 5 630 ns = 0.79 steps | **4 666 ns = 0.62 steps** |

**The generalizable finding: a ratio between two arms that vectorize
differently belongs to the pair AND the target-cpu.** The TCAM sweep gains 2.7×
from AVX-512; the range write is memory-bound and gains ~nothing. The ratio
therefore falls by ~2.5× *because the wider ISA helps the arm being beaten*.
§14's 291 ns/pass (0.285 ns/word) is 2.0× my v3 and 2.6× my v4 per word, so it
belongs to neither run and its config was never recorded — it cannot be placed
on this table at all.

What survives on the correct tier: the matrix's G6 re-scope trigger is "below
~10×", and 110× clears it by an order of magnitude, so the claim stands while
the headline number is more than halved. The SHAPE is config-independent and was
never the headline — the TCAM arm is flat in node size on both tiers, the range
arm tracks the node.

Standing rule: every timing published for this plan names its target-cpu.
`mask_set_range`'s 5/5 tests and the 11 parity groups are green under v4 as
well as v3.

## 2026-09-16 (2) — N1 / T1 gap G6 LANDED: `mask_set_range`, and the primitive changed the measured SHAPE, not only the code

`pub fn mask_set_range(out_words: &mut [u64], lo: usize, hi: usize)` — full
overwrite (bits `[lo, hi)` set, everything else zeroed), address-blind, no
per-bit loop, single-word case its own branch. Plan §16.6 N1; matrix §3 G6.
Consumer wired: `hex_tenant_mq_probe`'s `range_reveal` keeps its `(lo, hi)`
trie arithmetic and calls the primitive for the write.

**The measurement is a MIXED result and is recorded as one.** The probe's range
arm:

| level | rows/node | hand-rolled ns | with the primitive |
|---|---|---|---|
| 0 | 65 536 | 89 | 52 |
| 1 | 4 096 | 54 | 50 |
| 2 | 256 | 44 | 51 |
| 3 | 16 | 43 | 50 |
| 4 | 1 | 43 | 52 |

Faster at large ranges, **marginally slower at small ones**, flat overall.
Mechanism, and it is the reason to keep the primitive rather than the ratio:
the hand-rolled version zeroed the WHOLE buffer and then overwrote the
interior, paying 1024 redundant word writes at level 0 and none at level 4; the
primitive splits the zero-fill around the range, so it writes each word once.
Reveal-vs-TCAM band: **161.8×–343.5× → 273.9×–300.9×** — a higher floor and a
flatter curve. Both bands are one host, one fixture, one seed; §16.2's
portability caveat applies unchanged.

`n` read 6 556 ns with residual 2.2 % here against 7 678 / 15.9 % on the
immediately preceding run of the SAME binary. That spread is cross-run noise on
this host, not an effect of the change — recorded so neither number is cited as
a before/after.

**Disable runs, both red-then-green, both after the commit (`33716b9`) so the
restore could not eat the work:**

| disable | observed |
|---|---|
| drop the three zero-fills and make the single-word write an `\|=` (i.e. OR, not overwrite) | 3 of 5 unit tests FAIL, including `..._overwrites_a_dirty_destination_rather_than_oring`; the two `should_panic` guards correctly stay green |
| make the single-word branch unreachable so it falls into the two-edge path | 2 of 5 unit tests FAIL **and the parity program ABORTS** — so the new `0xBxx` group is non-vacuous, not decoration |

Gates: clippy `-D warnings` clean, `cargo fmt` clean, 5/5 unit tests, masking
parity **11** groups bit-identical (was 10), probe gate (range mask == TCAM
mask, popcount == node size) green at every level.

Not built, named: no OR-ing variant (no consumer, and a zero-caller function is
dead code here). Still open from §16.6: whether a nibble-aligned
`reveal(prefix, level)` beats the general `[lo, hi)` — the general form shipped
first as a decision, not a finding.

## 2026-09-16 (1) — census of this plan against the board and git: four governance defects, a fourth hex-tenant run, and the wave is the DuckDB matrix's own T1 gap list

Three read-only censuses (ndarray code, ndarray plan+board, lance-graph
consumers) plus my own verification of every decisive claim. Full text:
`.claude/plans/gemm-ternlog-mask-consolidation-v1.md` §16.

- **§14's numbers are stale against entry (5)'s STORNO, and the STORNO's own
  commit `e5a87e6` edited this plan without touching them** (verified by diff:
  one line changed, §15's tail law). §16.1 tabulates every correction.
- **A fourth hex-tenant run** (§16.2) puts the reveal ratio at 161.8×–343.5×,
  below the stated ~200× floor, so across four runs the honest range is
  ~160–490×. Its `ternlogq` is 152 ns/pass against the board's 280–300 — a
  different HOST, not a better estimate. Consequence worth keeping: **coal
  denominated in "maintained steps" is not a portable unit** (µs fell, steps
  rose, because the step got cheaper). The invariant across all four runs is the
  SHAPE: the TCAM arm is flat in node size, the range arm tracks it.
- **`D-GTM-0m` names two unrelated probes** nine days apart (`f1f4023`
  behavioural soak, `d9459f0` hex tenant). Not renumbered — that would break the
  commit messages carrying the results — but no new work may use the id.
- **`D-GTM-0n` is measured, committed, and in NO governance doc.** Its
  mask-vs-sparse-survivor crossover (0.1–0.8 % active) is directly relevant to
  §12.5 pt 2's "no mask-beats-sparse claim until a sparse arm exists", and that
  question could not be asked while the probe was invisible.
- **The wave is not new.** `duckdb-to-v3-translation-matrix-v1.md` §3 already
  lists G6 (`mask_set_range`), G1 (u8/u16 compare-to-mask) and G2 (ordered
  u64/i64) as verified-absent with pre-registered falsifiers. N1 = G6 (two
  consumers working around it, payoff measured). N2 = G1 (two fixtures measure
  the 4× widening cost; the matrix's falsifier stands unchanged). N3 = G2 —
  **PR #308 answers the matrix's own open question**: the ordered-u64 predicate
  count is ≥ 1, and 100 % of the real offsets exceed 2³², so narrowing is
  unsound. G2 is a gap.

## 2026-09-14 (5) — STORNO on entries (3) and (2), from the #307 council (measurement-skeptic, kernel-membrane, overclaim): same numbers, tighter words

The numbers stand; the wording ran past them. Corrections, each to the sentence it replaces:

- (3) *"full-field −14 %"* → **−5 % to −14 % across two runs** (14 517 vs 16 944 and 16 507 vs 17 462): within this density's cross-run noise, i.e. no reliable full-field win over the per-bit loop. The table pinned the better run.
- (3) *"node-span −66 % (n 17.0 → 5.7 µs)"* → **on this fixture (frontier confined to one 64-word level-1 tile, 1/16 of the field, 29 % of the tile span), restricting the shifts to the tile span cuts the x=0 STEP from 16.9 to 5.7 µs (−66 %, one run).** It is a step ratio, not an op ratio: every arm still carries the full-field housekeeping (scratch zeroing, `OR2_AND`, the reverse walk over 1024 words) AND the probe's own per-rep reset — the `rails_run` copy is **786 KiB** (65 536 × 12 B), not the "8 KiB" the probe comment said (fixed), ~1 µs per step amortized inside every `n`. A frontier straddling k nodes pays Σ spans.
- (3) *"the ratio survives the degree-1 ablation"* → **a −33 % gain survives** (4 487 vs 6 704); the ratio halves. What the ablation shows is that the word op's advantage is not hex-specific; the other half of the degree-6 gain scales with the per-bit loop's direction count.
- (3) *"remaining 5.7 µs is the field-wide housekeeping"* → housekeeping **plus** the reset amortization; neither is separated.
- (3) *"delta-frontier on top of the node span buys nothing (5.7 → 6.0)"* → no gain; the +0.4 µs is inside single-window noise, not a finding.
- (2)/(3) *"coal = 8.9 µs = 0.48 maintained steps"* → **8.8–13.7 µs across three runs = 0.47–0.76 steps at x=4** (single window each). *"ternlogq 281 ns/pass"* → 280–300 across runs, max residual ≤ 2.8 %; the "1.7 % at x=1" is fit-derived. *"228–462×"* range-reveal ratio → ~200–490× across three runs at N = 65 536; the range arm's floor is the 8 KiB output clear.
- (3) *"68/68 gates"* → 68/68 probe ROWS pass the three spread gates on one fixture and one seed. *"1.8 % density"* → 1.8 % of the field, **29 % of the tile span** — the denominator that explains why the word op loses full-field and wins node-span.
- Method: one 50 ms window per cell, no repeats, no variance reported; cross-run spread is visible only because three runs were banked (ladder x=0 spread 3 %, `shift` 14 %, coal 56 %). Single-fixture observations, all of them.
- Kernel-membrane, on `mask_shift_morton`: the slice IS the field — a sub-span shift is NOT the restriction of the full-field shift (no carry crosses the span edge). The probe's node arm was correct only because a level-1 tile is Morton-aligned and the source is confined to it; that law is now on the op's doc and the probe asserts `lo % len == 0` at the call site. The op is a lattice axis shift (four Cartesian moves; hex diagonals are two-call compositions), renamed as such in its docs.
- Codex P2 on the probe: the Hebbian reverse walk credited any predecessor in the accumulated `state`; under the delta-frontier arms only `delta` cells propagated, so an older active neighbour of a newly reached cell could be credited for a firing it did not carry. Fixed (attribution against the step's SOURCE frontier, `delta` published after the walk) and **measured inert on this fixture — all 68 rows' fired/step and survivors identical before and after** — because eligibility is static here: an older predecessor's target was already settled when that predecessor was itself in the delta. Real for dynamic eligibility, unobservable in this probe.

## 2026-09-14 (4) — D-MRX-0: `*_to_mask_under` — the gated predicate mask-risc's `Pred { under }` promised and T1 lacked

**Why now.** lance-graph #1225's kernel-membrane review (PR2 council) ruled the
executor may not compose a gated predicate from `*_to_mask` + `mask_and` (the
whole plane, twice) nor skip words itself (a compute path above the facade):
*"PR3 lands `*_to_mask_under` first."* This is that word.

**Shape.** ONE private engine, `pack_under::<T, L>(name, values, under, out,
group_bits)`: `out[w] = under[w] & pred(values)[w]`, the predicate evaluated
only where `under[w] != 0`. The skip is word-granular (64 rows) — the unit
the caller holds; a per-group skip is reachable and deliberately not done. An
executor may skip coarser (mask-risc speaks of 1024-row chunks) with a result
identical by construction (a skipped chunk is an all-zero gate) — not
exercised by any test. COMPARE cost ∝ live gate WORDS (a sparse frontier
spread across every word pays every compare); the per-word gate test and
zero store stay ∝ rows/64; no timing of the skip exists, only the call count
below. The predicate's own tail law makes a phantom gate bit past `n` vanish —
the AND conforms regardless, so a gate never needs cleaning before use
(surplus `under` words are never read). Ten public
members, one per `Pred` variant (`gt/lt/ge/le/eq/ne_i32`, `eq/ne_u32`,
`ternary_match_u32/u64`), each one `pack_under` call over the SAME lane op its
ungated sibling uses — no new backend semantics, so no backend file changed.

**Falsifiers.** (1) family ≡ `ungated & gate` word-for-word at ten lengths with
phantom gate bits present; (2) the skip is MEASURED: a counting closure sees
20 of 40 groups under an alternating gate, 40 under all-ones — disable-run
red (`left: 40, right: 20`) with the `gate == 0` early-out removed, green
restored (verified once, not re-runnable); (3) phantom bits do not leak at
n = 70; (4) short gate panics; (5) each public member is ONE `pack_under`
delegation, counted from the source, so the skip test on the private engine
covers all ten;
doctests 10/10. Parity harness group 10 (`0xAxx`, `check_predicates_under`):
alternating-zero gate with random phantom bits, reference reads the gate BIT
per row. Native (AVX2), nightly (`core::simd`), wasm simd128 and wasm scalar: all
10/10 bit-identical here; neon has no qemu in this sandbox, and the
`neon-simd/parity-qemu` CI job on #307 is the witness for that arm (green).

**Allocation.** Engine + first member by the orchestrator; nine members +
tests by a worker agent against a written spec (its Bash died on a full disk
mid-run — the tasks tmpfs and the checkout share one allowance — so it landed
Tasks A–C unverified and reported exactly that; the orchestrator gated
centrally after freeing 6.5 GB of stale scratch targets and landed the parity
arm itself). W1a deviation record as for `mask_andnot` / `mask_ternlog`:
free-fn family shape beside its siblings, not a struct method.

**Loose ends.** The strided operand family (12-in-16-byte register compares
for `LaneRef`) and `u8/u16/u64` compare-to-mask (DuckDB matrix G1/G2) remain
T1 gaps; neither is needed for PR3's first executor.

## 2026-09-14 (3) — D-GTM-1m MEASURED: `mask_shift_morton` lands; the win is in the NODE SPAN, not the op — n = 17.0 → 5.7 µs (−66 %)

`mask_shift_morton` (255c36d) is bit-exact (F1–F4, parity 9/9 on native/
nightly/wasm/wasm-scalar; neon-qemu absent here) — and over the FULL field it
barely moves `n`: 14.5–16.5 µs vs 17.0 µs for the per-bit loop. The worker
implementer named the mechanism correctly: the op is FIELD-size-bound (1024
words × ~8 passes per direction) while the per-bit loop is ACTIVE-bound, so at
1.8 % density they cost about the same and the op loses once gates thin the
frontier.

**The fix is the fixed-spatial-distribution dividend a second time.** A trie
node is a contiguous word span AND a square Morton sub-field (the level-1 tile =
64 words = a 64×64 field, `log2(64)` even), so the shifts run over the node's
own span with no correctness change (the source ⊆ tile, so no carry enters the
span; a carry leaving it is what `& tile` removes anyway). Measured, dirs = 6,
x = 0, same gates green (68/68 rows, 0 heap B/step):

| arm | ns/step | vs ladder |
|---|---|---|
| ladder (per-bit, full state) | 16,944 | — |
| nnue (per-bit, delta frontier) | 9,296 | −45 % |
| shift (word op, full field) | 14,517 | −14 % |
| **node (word op, tile span)** | **5,677** | **−66 %** |
| node+g / node+nn / node+nn+g | 5,722 / 6,044 / 5,774 | — |

Delta-frontier on top of the node span buys nothing (5.7 → 6.0), as predicted:
a span-bound op does not care how many bits are set. Degree-1 control: node
4.5 µs vs ladder 6.7 µs — the ratio survives the E-Q8 ablation, so the gain is
the word op, not the six. The ternlog fit is unchanged (281 ns/pass, n = 17.0 µs
on the ladder arm by construction). Remaining 5.7 µs is the field-wide
housekeeping (scratch zeroing, `OR2_AND`, the Hebbian reverse walk over 1024
words) — the next rung restricts THOSE to the node span too and re-measures;
not claimed here.

Rule extracted: **a word-level op pays for the span it is given; give it the
node, never the field.** The same statement as "top-down is a range, not a
compare" (2026-09-14 (2)), now on the grey side.

## 2026-09-14 (2) — D-GTM-0m: the hex TENANT — top-down traversal AND spread on ONE Morton-keyed SoA; `step = x·ternlogq + n` measured, and the chain is 1.7 % of it

Probe: `examples/hex_tenant_mq_probe.rs` (`--release`, committed; output banked
in the session scratchpad). Operator statement it builds (2026-09-14): *"static
traversal top down AND plasticity (spread) in the same substrate — SoA gets a
hex tenant with 6×2×8 bit and the field is a trie (fixed spatial distribution)."*
This is `gemm-ternlog-mask-consolidation-v1.md` §9 M1/M1b/M2/M3 and §11.10
(`substrate == mask geometry == projection surface`) executed on the merged #306
facade, not argued. Substrate: 65,536 rows = 256×256 axial hex cells, **row =
Morton(q, r)**, payload = the V3 12-byte register read `6×(u8:u8)` with rail `d`
= hex direction `d`, `u8:u8 = (permeability, strength)`. §9 R1 ("not obviously
the same six") is resolved by construction: adjacency and carving ARE the same
six once the rail index is the direction. Three gates, all green (32/32 cells):
range reveal == TCAM reveal at every level/prefix; Morton-arm spread == an
independent row-major axial BFS at every step (plasticity bytes compared too);
hot-path heap = **0 B/step** everywhere (counting allocator).

**White — top-down is a RANGE, not a compare.** A trie node at nibble level L is
`2^(16−4L)` CONTIGUOUS rows, so revealing it is a range write: **49–99 ns** vs
**22.4–22.8 µs** for the general `ternary_match_u32_to_mask` sweep over the
address column — **228–462×**. That is the fixed-spatial-distribution dividend
stated as a number: the TCAM op stays for addresses that are NOT laid out (the
D-GTM-0l linker case); a minted, Morton-keyed tenant never pays it.

**Grey — the cost model, fitted (identity gates, survivors held at 1175):**
`step = x·ternlogq + n` with **ternlogq = 291 ns/pass** (8 KiB masks, 0.285
ns/word) and **n = 17.3 µs**, max residual 2.8 % over x ∈ {0,1,2,4,8,16,32}.
So at x = 1 the chain is **1.7 %** of the step; even x = 32 only doubles it.
`n` is the ONE non-mask op on the path — the per-active-bit hex shift (dilated-
integer add per direction). **That is the missing substrate word**: a mask-level
neighbour shift on the Morton lattice (`mask_shift_hex(state, d, dst)` — within
a nibble a 4×4 block shift, carries across blocks), which would fold `n` into a
handful of word passes. Filed, not built. First rung without it: the **NNUE
reading** — spread from the DELTA frontier (`scratch & !state`), never from the
accumulated state — gives the identical closure (gate green) at **8.8 µs**
(−48 %), 8.4 µs with the real gates. The E-Q8 control is in the table (degree-1
arm: 7.0 µs, 301 survivors — a different closure, so a cost floor, not a cost
equivalence).

**Coal.** One re-chain (regenerate a resident mask from its column,
`gt_i32_to_mask` over 256 KiB) = **8.9 µs = 30.6 ternlogq passes = 0.48
maintained steps** at x = 4. M2 is linear: x → x±1 is one pass (291 ns), no
cliff. M1b generation (6 eligibility masks) = 75 µs once per mask generation.
Plasticity: 740 rows' strength bytes bumped IN the register (1,263 firings over
24 steps), on survivors only — the first version fired on every re-reached cell
the gates then removed, which is the "fire before you know it survived" bug the
reverse walk fixes.

**Stated limits.** The T1 compare is i32-wide, so the u8 permeability column is
widened 4× for `gt_i32_to_mask` — `n_gen` and coal are UPPER bounds; a u8/u16
compare-to-mask is a T1 addition. One fixture density (62 % permeable), one
tile size, timing floor 50 ms, no `perf`. No production caller; this is W0.

## 2026-09-14 — AVX2 arm of the mask family MEASURED, not rewritten: 6 of 10 shapes were already packed, 4 earned intrinsic realizations

**The pre-compaction plan was wrong, and the instrument said so before code
was written.** The five-flavour audit (entry below) scheduled a rewrite of
`simd_avx2.rs`'s `U64x8`/`I32x16` from `avx2_int_type!` array polyfills to
native `[__m256i; 2]` types. Before doing it I added the mask family to the
codegen oracle as Group F (`.claude/knowledge/simd-codegen-oracle/probes.rs`,
ten `#[inline(never)]` probes calling the SHIPPED library methods, each with
a runtime self-check against the bit-serial definition) and ran it on the
untouched polyfill at `-Ctarget-cpu=x86-64-v3`:

| shape | packed / scalar-lane-arith |
|---|---|
| `ternlog::<MAJ3>` / `::<0xCA>` u64x8 | 18 / 0 |
| `ternlog::<XOR_AND>` u32x16 | 8 / 0 |
| `andnot` u64x8 | 6 / 0 |
| `popcnt` u64x8 | 21 / 0 (vpshufb nibble LUT, not 8× popcntq) |
| `xor_popcount` u64x8 | 25 / 0 |
| **`rotate_left`** u64x8 | **0 / 8** `rolq` |
| **`reduce_max`** i32x16 | **0 / 17** `cmpl` |
| **`gt_bitmask`** i32x16 | **23 / 3** MIXED — lanes 0, 13–15 peeled to scalar |
| **`cmpge_zero_mask`** i32x16 | **17 / 11** MIXED — same peel |

The whole bit-logic half — the generated Shannon ladders, andnot, popcount
— was packed from scalar source, exactly the oracle README's standing
finding ("a recent PR hand-wrote ~700 lines of intrinsics to fix a gap that
did not exist"). A `[__m256i; 2]` rewrite would have re-implemented six
already-packed shapes and broken every `.0[i]` site in the file's seven
`U64x8`/`I32x16`/`U32x16` impl blocks for nothing.

**What shipped instead (backend-local, narrow `unsafe`, no `#[target_feature]`):**
`U64x8::rotate_left/right` → `vpsllq`+`vpsrlq`+`vpor` per 256-bit half
(uniform xmm count; 10 packed / 2 scalar — the 2 are `n % 64` / `64 - n`
count setup, not lane data); `I32x16::reduce_min/max` → `vpminsd`/`vpmaxsd`
tree 16→8→4→2→1 (8 / 0); `I32x16::gt_bitmask` → `vpcmpgtd` + movemask per
half (9 / 0); `I32x16::cmpge_zero_mask` → complemented sign-bit movemask
(10 / 0). SAFETY precondition on every block: this file is the x86-64-v3
backend, `.cargo/config.toml` pins the target-cpu for every x86_64 build
that selects the arm — the footing the native `U16x16` already stood on.
Oracle re-run: ALL PROBES MATCH; Group F rows now carry `expect =
"vectorized"` with 60 % floors and both runs' numbers in the notes.

**Two stale claims corrected in the same pass.** `I32x16::gt_bitmask`'s doc
comment said the oracle had measured a clean packed lowering for "exactly
this form" — it had not been probed; measured, it was the mixed peel above.
And `simd.rs`'s AVX2-arm comment claimed `simd_avx2.rs` carries
per-function `#[target_feature(enable = "avx,avx2,fma")]` — grep finds
zero, and by the operator's standing rule there must be none (one backend
file, one compile-time target). Both rewritten to what is true.

**Instrument findings, recorded not explained.** (a) `scripts/neon-asm-rung3.sh`
misreported `check_ternlog_all_tables` as scalarised on its first real run
because it counted LLVM's `.LBB*` basic-block labels as symbol boundaries —
684 vector ops fragmented into hundreds of 4-op stubs. Fixed (function
symbols only) and the scalar reference oracles excluded from the gate: rung
3 PASS at 794 vector / 33 scalar. (b) The oracle's untouched hand-written
`shiftor_rot_u64x8` probe flipped 0→10 packed in the same run that made the
library rotate an intrinsic, while its two siblings stayed scalar; mechanism
not established, noted on its baseline row as a finding about the instrument.

**Gates on the final tree:** v3 clippy `-D warnings` + full lib 2292/2292;
v4 clippy + 114 masking/simd tests; aarch64 check + rung 3 PASS; WASM parity
OK; oracle ALL MATCH. New falsifier
`i32x16_compare_bitmasks_and_reductions_at_lane_extremes` places the signed
extremes at lanes 0/7/8/15 and walks MIN/MAX through every lane, so a
wrong half order or a lane-dropping tree cannot pass it.

## 2026-09-13 (2) — FIVE-FLAVOUR AUDIT of the masking substrate: U64x8/I32x16 were SCALAR on NEON, WASM and (as array polyfills) AVX2

**Operator correction, verbatim in substance:** *re-read the dispatch
architecture before using `safe_intrinsic_probe` to establish policy.* There
are five execution flavours — (1) x86-64-v3 default/CI → `simd_avx2`;
(2) AVX-512/v4 → `simd_avx512`; (3) `target-cpu=native` → backend from the
build host's CPUID; (4) `nightly-simd` → `simd_nightly`/`core::simd`;
(5) `--features runtime-dispatch` → one LazyLock capability detection, then the
selected kernel. **`#[target_feature]` propagation is NOT the architecture.**
For every compile-time flavour the selected backend file IS the capability
proof and raw intrinsics stay at a narrow backend-local `unsafe` boundary; for
flavour 5 the LazyLock branch is the proof; nightly inherits no ISA contract.
**Never route a mask primitive through Scalar because rustc wants `unsafe` at
an intrinsic.** (Measured: PR #306 adds no `#[target_feature]` outside the
probe's two demonstration arms; the worker briefs forbid it verbatim.)

**The audit's finding, the one that mattered:** the mask family is built on
`U64x8` (all bulk algebra + ternlog) and `I32x16` (the signed-compare family),
and `simd.rs` resolved BOTH to the **scalar** backend on aarch64
(`:390-393`) and wasm32 (`:408-414`), while the v3 arm's are `avx2_int_type!`
array polyfills. Only the `U32x16` paths (`eq_u32`/`ne_u32`/`ternary_match_u32`)
reached NEON/v128. So the "424 NEON vector ops" rung-3 measurement was on the
one lane type the family barely uses, and every `mask_and`/`mask_ternlog`/
`gt_i32_to_mask` ran scalar loops on three of five flavours. The
`agnostic-surface-cpu-matrix.md` rows claiming NEON `4×uint64x2_t` /
`4×int32x4_t` were wrong (the dispatch-architecture matrix's ❌ was right).

**Fix (PR1 scope — "backend-local polyfill completion"):** native `U64x8`
(`[uint64x2_t;4]` / `[v128;4]` / `[__m256i;2]`) and `I32x16` (`[int32x4_t;4]` /
`[v128;4]` / `[__m256i;2]`) in `simd_neon.rs`, `simd_wasm.rs`, `simd_avx2.rs`
with the FULL scalar `impl_int_type!` surface (so nothing that compiled
against the scalar re-export breaks — census: no consumer constructs these
types or reads `.0`); generator arms for their `ternlog`; `simd.rs` re-export
flip; harness arms (`check_u64x8_algebra`, `check_i32x16_compare`) on NEON and
WASM; `scripts/neon-asm-rung3.sh` — the rung-3 count made symmetric
(same mnemonic set on v-regs and w/x-regs, incl. `orn`/`mvn`), attributed per
symbol, with a gate that every `ternlog` symbol is vector-dominant.

**Recorded limits, not fixed here:** flavour 4 — the mask family does not
compile under `nightly-simd` (`I32x16` has `cmpgt_mask`, not `gt_bitmask`;
`U64x8` has no `andnot`; `ternlog` is the 36-op minterm) — PRE-EXISTING (same
calls lived in `simd_int_ops` before #306; CI's nightly job is skipped).
> ⊘ SUPERSEDED within #306 (e730109 "nightly realization complete"): the
> nightly arm now carries `gt_bitmask`, `andnot`, `ternlog` and the W1a
> types; the parity program runs on it (`masking-parity.sh nightly`) and
> the matrix's nightly row exercises it. The limit above is history.
Flavour 5 — no mask trampolines in `simd_runtime`; a release binary runs the
v3-compiled mask kernels on every host. Both are separate decisions.

**Council corrections folded in (C2, overclaim audit):** "≤ 7 ops" → 7/8 per
vocabulary, now asserted by the generator; "113/113" → the filter is named;
the x86 256-table sweep gained a `U32x16` twin; "public surface unchanged" →
facade-preserved, module paths removed; `#![forbid(unsafe_code)]` is now
declared on `simd_masking_ops.rs` rather than claimed; the wasm harness
comment no longer claims the NEON equality it does not run; `mask_any` is
documented tail-blind (its pair `mask_all` takes `n_rows`); the generator gained
`--check` (regenerate-and-diff; a hand-edited body is now detectable).

## 2026-09-13 — `simd_masking_ops.rs` + generated backend-local `ternlog` bodies + the DuckDB-vector-execution primitive set (PR1 of the mask-RISC arc)

**Three-layer contract, operator-ruled this session — the architecture law
this entry exists to make durable:**

```text
consumers (lance-graph-mask-risc, lgj-abi kernels, planner)
        │  semantic ops only: TERNLOG<IMM>, AND, XOR, COUNT, eq→mask …
        ▼
simd_masking_ops.rs     slice/chunk/tail ergonomics, *_assign forms,
        │               mask composition, masked reductions — NEVER an ISA
        ▼
simd.rs                 architecture-agnostic lane types, compile-time selected
        ▼
simd_{avx512,avx2,neon,wasm,scalar}.rs   each owns its realization, as a PEER
```

- **POLYFILL LAW.** ndarray is the ISA membrane. Every public mask/SIMD
  primitive a consumer uses has compile-time implementations for AVX-512,
  AVX2, NEON, WASM SIMD and scalar. **Scalar is a peer backend, not a
  fallback.** No runtime ISA dispatch, no fallback chains. Hardware-specific
  optimisation — including truth-table specialisation of `ternlog` — lives
  entirely inside the corresponding backend file. Consumers never branch on
  ISA; `TERNLOG` stays semantic above the backends.
- **BACKEND LAW.** No shared generic/polyfill implementation body that the
  backends delegate into. Shared *tests* and shared *generated truth-table
  logic* are fine; a shared *runtime* body is not. The route to remove
  repeated source is code generation emitting backend-LOCAL bodies.

**What landed:**

> **AMX fill (2026-09-14):** `src/hpc/amx_ops.rs` — the surviving
> `X86InstrAMX.td` surface (amx-transpose is absent by design) as mnemonics
> with `const` tile operands (INT8×4, BF16, FP16, COMPLEX×2, FP8×4, MOVRS×2,
> AVX512 row ops ×6 under `avx512f` cfg, STTILECFG/TILELOADDT1, all 8 tiles)
> plus TF32 as hand-encoded raw bytes (nightly's LLVM 23 rejects the
> mnemonic; b80fd83). `amx_features()` per LLVM `Host.cpp` bits;
> `amx_report()` prints the tiers. Encoding falsifiers read each wrapper's
> bytes out of the test binary's ELF symtab on any x86 host: the four
> GEMM-tier sequences are pinned to the EMR-VALIDATED table, every other
> pinned op is pinned to LLVM's own emission (a drift guard, not silicon
> validation). The "mirrored operand convention" turned out to be a misread
> of that table (Gotcha 15). Extended tiers are assembler-verified only.

> **CI finding (2026-09-14, e730109 red on `tier4-avx512-check`):** `ci.yaml`'s
> workflow-global `RUSTFLAGS: "-D warnings"` REPLACES every `.cargo/config*`
> rustflags entry (cargo precedence: RUSTFLAGS > target.<triple> > target.<cfg>
> > build). Consequences, both measured: (a) the v4 job never had a
> target-cpu, with the env-var recipe (which loses to the joined cfg `v3`
> locally) or with `--config` (erased by the global env in CI) — the new
> vpternlog assertion caught it at 0; (b) EVERY x86 job in CI builds at the
> x86-64 baseline, without the v3 pin and without the dalek/poly1305 `--cfg`s
> that `.cargo/config.toml` exists to apply. (a) is fixed in this PR
> (`env -u RUSTFLAGS` + `--config .cargo/config-v4.toml`, `-Dwarnings` moved
> into that file). (b) is pre-existing and out of this PR's concern: the fix
> is a triple-scoped `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS` for the
> x86 jobs plus per-target flags for nostd/wasm/aarch64, its own PR.

1. **`src/simd_masking_ops.rs`** — the mask family moved out of
   `simd_int_ops.rs` wholesale (predicates→mask, mask algebra, ternlog,
   masked reductions, care-masked register match, blend) with its tests.
   `simd_int_ops.rs` is integer arithmetic/conversion again. Every moved
   `pub fn` (31/31) still re-exports through `ndarray::simd`, AND the 13
   mask functions that were public on master as `ndarray::simd_int_ops::<f>`
   (`simd_int_ops` is `pub mod`, so those were public paths — the first
   draft dropped them and called the surface "unchanged", C2; CodeRabbit
   round 2 caught the downstream break) are re-exported from `simd_int_ops`
   as a compatibility surface, verified complete by diffing master's
   `pub fn` list against HEAD's `pub fn` + `pub use` set (0 missing). The
   canonical path is the facade; `lance-graph-planner`
   `examples/dcr_w0_replay_budget.rs` moves to it in the lance-graph PR.
2. **`tools/gen_ternlog_bodies.py`** — Shannon-lowers each 8-bit table into
   two 2-input tables (`f = (!c & T0) | (c & T1)`), ≤ 7 ops (the naive
   minterm form was up to 36), self-checks all 256 tables in Python, and
   PRINTS each backend's body in that backend's own vocabulary between
   `GEN-TERNLOG` markers (worst case **7 ops** where the vocabulary has a
   native and-not — NEON `vbic`, WASM `v128.andnot` — and **8** where and-not
   is spelled `x & !y`, the avx2/scalar operator vocabularies; the generator
   ASSERTS these bounds on the emitted text, `count_ops`; the earlier "≤ 7 for
   any table" was wrong for two of four backends — C2 council finding): operator traits on the array lanes (avx2, scalar),
   per-`u32`-lane for NEON (`#[cfg(target_arch = "aarch64")]`-gated helper),
   `v128_*` intrinsics for WASM (helper inside the cfg-gated `wasm32_simd`
   module). AVX-512 keeps `_mm512_ternarylogic_epi64` untouched. The generic
   `simd_ternlog_lower.rs` that a first cut shared across four backends was
   DELETED — it violated the Backend Law and also blew the debug stack
   (`#[inline(always)]` × 256 tables in one test frame).
   Two generator traps recorded: a `const` item cannot read the enclosing
   fn's `IMM` (E0401) — the tables are `let`-bound and fold identically after
   monomorphisation; and the helper must be placed INSIDE the cfg-gated
   module or every host compiles it and fails to resolve the intrinsics.
3. **New primitives** (all through `ndarray::simd`): `lt/ge/le/ne/eq_i32_to_mask`,
   `ne_u32_to_mask`, `mask_not{,_assign}` (tail re-cleared against `n_rows`),
   `mask_xor{,_assign}` (its own primitive, lane `^` — NOT `ternlog::<XOR3>`,
   whose AVX2 minterm cost is pointless for a native op), `mask_any`,
   `mask_all`, `ternary_match_{u32,u64,strided}_to_mask` (care-masked
   register match via `ternlog::<XOR_AND>` + zero test — the TCAM shape of a
   V3 12-byte facet), `masked_min/max_i32`, `blend_i32`; immediates
   `XOR_AND = 0x28`, `AND2_OR = 0xEA`. Ordered compares derive from `gt`
   by complement, so they are exact at `i32::MIN`/`MAX` (threshold shifting
   underflows).

**Acceptance matrix, measured (not asserted):** for every IMM in 0..=255,
bit-serial reference == the compiled realisation —
x86 arms, `cargo test --lib -- simd_masking_ops::tests simd_int_ops::tests
simd::tests` (a FILTER — the lib suite is ~3,100 tests; 113 is the selected
set): v3 113/113, v4 (separate target dir) 113/113. On x86 the 256-table
sweep now runs on BOTH lane types (`U64x8` and, since the C2 finding, a
distinct `U32x16` sweep — the v3 arm carries two separate generated
ladders and v4 two different intrinsics, so one sweep proved nothing about
the other); **WASM: run for real under node** via
`scripts/wasm-parity.sh`, whose harness gained `check_ternlog_all_tables`
(256 tables × `U32x16` native v128 body × `U64x8` scalar body, two operand
triples each) — rc=0; **NEON: rungs 1 and 3 of the AArch64 ladder measured on this host** —
`cargo check --target aarch64-unknown-linux-gnu` of lib+tests and the
harness (all 256 monomorphisations) compile; the cross-compiled harness
assembly selects **424 NEON vector logical ops** (`and/orr/eor/bic/orn
v.16b` — LLVM fuses `orr(mvn)` into `orn`) against 41 scalar ops left in
harness scaffolding. The FIRST generated NEON body — a per-`u32`-lane loop
through `to_array()`/`from_array()` — SCALARIZED: 536 scalar vs 4 vector
ops. Same truth tables, same tests, rung 3 red. The body is now emitted
per 128-bit quad in the backend's own intrinsic vocabulary
(`vandq/vorrq/veorq/vbicq/vmvnq_u32` on `uint32x4_t`). Rung 2 (run under
qemu) is CI's `neon_simd` job — no cross linker / qemu on this host, stated
not assumed; rung 5 (Apple/AArch64 hardware) is a later performance gate,
never a blocker for authoring the backend. Both harness arms use the SAME check body — shared tests are
allowed, shared implementation is not.

**AArch64 acceptance ladder (operator, 2026-09-13)** — the LLVM/Clang
intrinsic corpus is the remote instruction catalogue for a backend the
author cannot run: (1) cross-target compile succeeds; (2) parity harness
compiles/runs under an emulator where sensible; (3) generated LLVM IR /
assembly contains the expected NEON operations and no unexpected
scalarisation; (4) truth-table / reference parity is exhaustive where
possible; (5) real hardware benchmarking is a LATER performance gate.
Three different proofs, kept sharp: LLVM says what lowering is available,
cross-compiled assembly says what LLVM actually selected, hardware says
whether the selection is fast.

**Two standing rules (operator, 2026-09-13), recorded where the next
backend author will look:**

- **97 % safe. `unsafe` only for byte-code asm (AMX-class inline asm).**
  Operator, on the intrinsic question: *"98 % of intrinsics are available in
  safe mode by rust 1.98.1 — if not, document where and why."* Measured on
  the pinned 1.98.1 with `tools/safe_intrinsic_probe` (re-run after every
  toolchain bump; the answer is a toolchain property):

  | arch / call shape | 1.98.1 |
  |---|---|
  | aarch64: plain fn → `vandq_u32` | **E0133** — caller must carry `#[target_feature(enable = "neon")]`; build-config `neon` "does not remove the requirement" |
  | aarch64: `#[target_feature(neon)]` fn → `vandq_u32` | OK (safe, no `unsafe`) |
  | aarch64: plain fn → that safe annotated fn | **E0133** — the requirement propagates up the chain |
  | x86_64: plain fn → `_mm_and_si128` (sse2, baseline) | **E0133** |
  | x86_64: plain fn → `_mm256_and_si256`, even with `-Ctarget-cpu=x86-64-v3` | **E0133** |
  | x86_64: plain fn → `_mm512_ternarylogic_epi64`, even with `-Ctarget-cpu=x86-64-v4` | **E0133** |
  | wasm32: plain fn → `v128_and`, with or without `+simd128` | **OK** |

  So the intrinsic *functions* are safe, but rustc only accepts a per-fn
  `#[target_feature]` as evidence — and in this repo that evidence is
  **illogical to state** (operator ruling): every `simd_{arch}.rs` is compiled
  for exactly one target CPU, selected by `cfg` at compile time, so the
  feature is already a property of the file. Annotating each fn would be a
  second, redundant declaration of the same fact, and it would propagate to
  every safe caller up to the pub boundary. rustc simply does not read the
  `cfg` as proof. Consequence: one expression-narrow `unsafe` at the
  intrinsic boundary per backend method, with a SAFETY line (the generated
  NEON body); the generated WASM body carries none; `simd_masking_ops.rs`
  and every consumer above it stay `forbid(unsafe_code)`. Follow-up, not
  this PR: the 20 pre-existing `unsafe` blocks in `simd_wasm.rs` are
  removable under this finding.
- **Conversions are bit-exact; rounding happens at most once.** F32 →
  BF16x16 rounds exactly once, through a fused `add_mul` — never a separate
  multiply then add, never a convert-then-convert. Mask primitives carry no
  floats, so this PR is unaffected; the rule binds the BF16 lanes and every
  future reduction that touches them.

**Loose ends:** `simd_masking_ops` still has only slice-level compositions
that consumers already needed; the ergonomic fused forms the mask-RISC
executor will want (`masked_count_where_eq`, chunked survivor-word skip
helpers) land with that consumer, backend-first. A `stride_bytes == 8` twin
of the contiguous `eq_u32_strided` fast path is still unbuilt (no caller).

## 2026-09-05 — D-GTM-0l MEASURED (prefix-tract coverage, R2IL 6502 ore)

The probe I flagged as decisive ran. `examples/prefix_tract_coverage_probe.rs`
against the Elite-rs R2IL harvest (427 long-range call edges, 2,647 packed addresses;
ore passed by path, never vendored — its provenance forbids redistribution).

- Prefix locality is REAL: k>=2 shared prefix is 15.69% of edges vs 1.52% under a
  degree-preserving null (50 shuffles). ~11x enrichment.
- The tract codebook does NOT compress: 7.62x at k=1 (4 KiB buckets, 16 of them on a
  64 KiB image — no resolution left), 1.66x at k=2, 1.08x at k=3, 1.00x at k=4.
- The non-local residual is DIFFUSE, not a hub set: 262 edges over 113 targets,
  entropy 6.36 bits vs 6.82 uniform. A k=1 tract plus explicit far edges is 318
  entries against 427 edges.

Verdict [G] on this ore: "white matter as an interpretation of packed location
prefixes" fails its own success condition — one tract per edge is an edge list.
Scope leg: these are PHYSICAL addresses from a 1986 linker, not semantic addresses
minted so the prefix carries meaning. The falsification is of the physical case only.
Next probe is the same instrument against an OGAR-minted classid space; the plan's
new §13 carries the full tables and the pass/fail condition for it.

# Current epoch (2026-05-26) — splat / palette / pillar / 3DGS

> **Read this first.** The "Polyglot Notebook" architecture below is a
> separate/older program, not the current epoch.

## 2026-09-05 (W0 MEASURED) — mask/trie vs GEMM: 0k passes, 0j falsifies the density framing

Probe `examples/hex_trie_vs_gemm_probe.rs` (committed, --release). N=4096, mask
512 B, dense relation 64 MB, correctness gate on survivor counts at every cell.

**D-GTM-0k ANSWERED, cleanly.** Mask hot path = **0 bytes/step** at every
density, every depth, both relation shapes — measured with a counting global
allocator, not asserted. GEMM = 73,728 B/step (packing buffer inside gemm_f32).
The invariant's own falsifier passes.

**D-GTM-0j FALSIFIES §11.1 pt 6.** "GEMM is attractive when information is
dense" — measured, there is NO density crossover: masks win 745x at 0.02%
relation density and 297x at 100%. Both costs are flat in density (GEMM O(N²)
FMAs; mask O(active·N/64) word ORs). Honest correction, a TYPE boundary not a
density one: **masks win whenever the relation is Boolean; a relation that
carries VALUES needs a value-aware algorithm** — which one (GEMM, CSR SpMV, other)
is the still-unrun weighted arm, not a conclusion of 0j. A bitmask is 32x denser
than f32 before any algorithm runs, so a 0/1 relation in f32 was never the right
representation.

**The headline numbers are explicitly NOT evidence** (§12.5): the dense-f32
baseline is mis-specified, and the missing arm is CSR SpMV (O(nnz) — at deg=1
that is ~4096 FMAs, same order as the mask arm's 11.7 µs, so it would plausibly
cross). The one internally fair comparison is PREFIX vs RANDOM inside the mask
arm: 21-79 ns flat vs 305-35,522 ns scaling with active bits — structure worth
~3 orders of magnitude, degrading exactly where there is nothing to exploit.

**The gate caught a bug in my own probe.** RANDOM failed immediately (912 vs
930): the GEMM arm computes `{i : srcs(i) ∩ active ≠ ∅}` while the mask arm was
unioning `srcs(i)` over active i — those agree only for a SYMMETRIC relation
(bucket membership is, random is not). The mask arm must union the TRANSPOSE.
A second flaw was caught by reading, not by any gate: mask timings at/below
timer resolution made every early "speedup" (25,940x…853,300x) a noise ratio;
both arms now run to a 50 ms floor and report ns/step.

**0h graded [S]:** no perf in this sandbox; residency inferred from timing
(PREFIX 70 ns at depth 1 → 21-27 ns at depths 8/32, no knee to 32) — consistent
with resident, not proof. **0l is now the decisive probe** — with the density
axis dead, the hypothesis stands or falls on prefix-routing coverage vs codebook
entropy on the R2IL/C64 ore. New probe implied by 12.4: the weighted arm.

## 2026-09-05 (v1.3) — the invariant strengthened: `substrate == mask geometry == projection surface`

Operator: "make the 96-bit object holographic" was metaphor while the cube wanted
voxel-by-voxel states. The Panela/photolithographic layer fixes it — **the cube is
never stored.** Holographic now means: *the information to reconstruct/address the
relevant 3-D relation is distributed through the 2-D 6×2×8 surface*, and the
volume appears only when a question requires it. Depth ← packed location; local
curvature ← hex adjacency; scale ← trie prefixes; permeability ← masks; dumb
physics ← VPTERNLOGQ (meaning lives in layout, never in the instruction —
`membrane-tiers.md` T1 from the other side). Hardware inversion: don't flatten
3-D onto silicon; make the higher-dimensional object a mask-address projection of
the 2-D surface. **Resolves the §11.9 diamond flag [S]→[H]:** bonds are implied
by address+masks, not walked, so coordination-4 vs the 16-ary trie is not a
conflict. The hologram's test already sits in the program: recover the relation
(D-GTM-0l, codebook entropy) without allocating it (D-GTM-0k, bytes/step → 0).

## 2026-09-05 (v1.2) — GEMM plan: grey/white matter over ONE packed register; D-GTM-5 corrected a third time

**Operator statement folded in as §11.** Hex field = digital grey/white matter over
one packed-address substrate, not two graphs. Grey = local hex state in the
existing 96-bit `6×2×8` register (same geometry is substrate AND mask; learning
changes permeability masks, never pointers). White = trie routing through packed
location (`ADDRESS & PREFIX_MASK == PREFIX`; a tract is `(prefix, mask, learned
transition)`, never a materialized path). TERNLOGQ is the membrane algebra.
**Invariant: `substrate == selection == routing`** — expanding a mask into IDs,
materializing a neighbour list, or converting the trie to an edge table on the hot
path is the loss condition. Hypothesis is NOT "ternlog replaces GEMM": GEMM wins
when dense; hex/trie may win when cognition is successive elimination.

**Reconciled with the measured hex record, not against it.** Q6/Q7/Q8 tested a
learned *association overlay* (recall/interference); §11 is a *compute + bytes*
claim — the bar r2il §7.2 already sharpened ("wins only as a COMPUTE topology").
E-Q8's degree-ablation lesson is now a MANDATORY control on every new probe. §9 R1
re-graded [S]→[H]: adjacency (grey, six neighbours) and carving (white, six rails)
are two readings of the same 12 bytes — `le-contract.md` §3 already says the
register "holds every sanctioned reading at once".

**The census bites both existing HHTL arms AND my own §9:** blasgraph
`heel_search → Vec<SearchHit>` (k=50 per tier) and splat3d `Vec<BlockDepthDecision>`
both materialize IDs; `splat3d/tile.rs`'s packed `(tile_id<<32 | depth)` key already
conforms. **D-GTM-5 corrected a THIRD time:** v1 `Vec<u32>` → v1.1 `ArrayVec<u32>`
→ v1.2 `pack_a_masked_f32` consumes mask words directly (tzcnt / vpcompress),
zero index materialization. Each revision removed one layer; the invariant is the
fixed point.

**K0..K7 [H]** = the SPO 2³ `TriadicProjection` masks (`cam-codebook-resonance-
projection.md`: 8 observation/query masks, "not decorative — the query grammar")
= the 8 rows of a ternlog truth table. **Six operator falsifiers** D-GTM-0g..0l
(mask/trie vs GEMM; VPTERNLOGQ residency vs depth; density sweep; crossover;
bytes-materialized/step → 0; prefix-routing codebook entropy) with one
pre-registered kill condition. **Panela + diamond (§11.9):** the toy's
positive/negative shape IS the invariant; diamond lattice (coord 4, tetrahedral)
as white-matter geometry is [S] — it conflicts with the 16-ary nibble trie unless
read as two bits of a level; flagged for one operator word.

## 2026-09-05 (later) — GEMM consolidation plan v1.1: Mississippi Queen amendment + Wave 0 static results

**Operator metaphor folded in (§9).** Mississippi Queen: the river board is laid a
few hex tiles ahead of the lead boat, speed changes ±1 and is committed before the
move, extra maneuvers cost from a fixed coal budget, and a tile laid for the leader
is free for every boat behind. Graded per the mechanism-vs-rhyme rule:

- **M1 [G] CORRECTS D-GTM-F4/D-GTM-5.** v1 said the compacted row-index list is
  "built once per mask generation" — laying the whole river before any boat moves.
  Wrong shape: `pack_a_f32` (`kernels_avx512.rs:552`) already walks a panel cursor,
  so mask→index expansion belongs ONE PANEL AHEAD of that cursor, in that loop.
  Signature changes `mask_to_row_indices(&[u64]) -> Vec<u32>` →
  `next_panel_indices(&[u64], cursor, mr) -> ArrayVec<u32, SGEMM_MR>` (stack, no
  hot-loop alloc — which also fixes a quiet `data-flow.md` §1 violation in v1).
- **M1b [G] — the amortization itself.** Many boats, one river: the cache key is
  `(mask generation, panel index)`, NOT the call. Per-call caching amortizes nothing.
- **M2 [H]** lookahead depth adapts ±1 and commits before the panel → D-GTM-0e
  becomes a LADDER (lookahead 1/2/4/8 × density 10/50/90%), not one crossover.
- **M3 [H]** coal = a bounded budget for mid-stream re-chains; replaces the
  T2→T1 prohibition (`membrane-tiers.md:105`) with a budget.
- **R1 [S] the hexagon is rhyme** pending one operator word: the game's six is
  ADJACENCY, the substrate's six (`6×(u8:u8)` facet rails, 6-byte HHTL path =
  CAM-PQ 6×256) is FIELD CARVING. Same cardinality, different mechanism. Unbuilt.

**Wave 0 static probes run (§10) — each corrected the inventory:**

- **0a:** both duplicate names DIVERGENT, only one a defect. `bf16_tile_gemm_16x16`
  = polyfill (`simd_ops.rs`, F32x16 decode) vs dispatcher (`hpc/`, AMX/VNNI) —
  legitimate, and `simd.rs:714` already renames the dispatcher `_amx`.
  `simd_avx2.rs:462 sgemm_blocked` is a **naive scalar triple loop** — neither AVX2
  nor blocked; file, name and body disagree three ways.
- **0b:** `blas_level3.rs` is NOT empty — 393 lines, zero `pub fn` because it is a
  **trait** (`BlasLevel3<A>`: gemm/gemm_into/syrk/symm/trmm/trsm, blanket impl,
  re-exported `simd.rs:656`) dispatching to `BlasFloat::backend_gemm`
  (`backend/mod.rs:75`, impl'd **f32/f64 only**). CLAUDE.md was right; my `pub fn`
  grep was blind. **Re-frames D-GTM-F3: TWO facades already exist** — the generic
  trait method and the four free functions — and `BlasFloat`'s `num_traits::Float`
  bound structurally excludes i8/bf16 from the generic one. W1's first question is
  which is canonical, not how to build one.
- **0f caller census:** `pruned_gemm_rows` **0 callers**, `mixed_precision_gemm`
  **0**, `blas_gemm` 0 external. §2.3 called `pruned_gemm_rows` "the ONLY existing
  mask→GEMM bridge" — it is dead code, so D-GTM-5 is a FIRST WRITER, not a
  migration. Only `bf16_tile_gemm_16x16` has real external consumers (5).

**Incidental find, reported not fixed (lance-graph call sites):** the 5 external
references reach `bf16_tile_gemm_16x16` by two paths that resolve to two different
bodies. `symbiont/src/domino.rs:27` imports it from `ndarray::simd` alongside
`amx_available` and its doc mentions tile ops — but that name is the POLYFILL; it
wants `bf16_tile_gemm_16x16_amx`. `thinking-engine/examples/amx_bf16_probe.rs:15`
imports from `ndarray::hpc::bf16_tile_gemm::*`, reaching past the facade — the exact
form the "all SIMD from `ndarray::simd`, never `hpc::*`" iron rule forbids. ndarray's
own facade is correct; both defects are consumer-side (and symbiont is deprecated).

**Not run:** 0c (f64/tail bench), 0d (MKL-ternlog tail hunch), 0e (the M2 ladder —
now known to measure a zero-caller kernel, fine for a probe, not production evidence).

## 2026-09-05 — AMX f32 GEMM was silently bf16; `matmul_f32` made exact; consolidation plan filed

**Finding (measured, PR #303):** `hpc::amx_matmul::matmul_f32` downcast both operands
to BF16 once and ran `TDPBF16PS` — ~1e-3 relative error under an f32 name, `Ok(())`
returned. Found via burn (burn#9): 50 linalg tests failed (25 qr / 13 lu / 7 svd /
3 det / 1 attention) vs 1826/1826 on the exact path. The one-rounding + f32-accumulate
discipline was correctly implemented and still insufficient: single rounding discards
16 of 24 significand bits. The existing test used `(i+j)*0.5` / `(i*3+j)*0.25` inputs —
all exactly BF16-representable — at 1% tolerance, so it could neither see nor fail on it.

**Decision:** `matmul_f32` delegates to `backend::native::gemm_f32` (exact, every host).
Both AMX-f32 variants (`matmul_f32_amx_split` 3-pass hi/lo split, `matmul_f32_bf16_fast`
1-pass) kept as NAMED opt-ins carrying the bench table. Why not the split: measured
`gemm_paths_bench` — AMX loses on BOTH axes at every size (1024³: F32x16 `sgemm_blocked`
27.3 ms / 1.4e-6; matrixmultiply 28.8 ms; AMX 3-pass 159 ms / 1.5e-6; AMX 1-pass
45.9 ms / 1.6e-4). AMX is a BF16/INT8 unit; `matmul_bf16_to_f32` / `matmul_i8_to_i32`
untouched.

**Plan filed:** `.claude/plans/gemm-ternlog-mask-consolidation-v1.md` (DRAFT v1) —
54 `gemm|matmul` entry points across 12 files, 4 unified; one facade per dtype
(D-GTM-F3), F32x16 `sgemm_blocked` as the exact default (D-GTM-F1), ternlog chaining
at T1 feeding a compacted-index GEMM prefilter (D-GTM-F4). Cached-mask reuse is
CONSUMED from lance-graph-java `mask-risc-lowering-v1` (v4.2), not re-planned. W0 is
six measurement probes; nothing in W1+ is built.

**Loose ends:** burn's `amx-f32` feature becomes dead once #303 merges — delete it
(burn#9 follow-up). `hpc/blas_level3.rs` shows zero `pub fn` in the inventory grep
(D-GTM-0b). `simd_ops.rs:587` duplicates `hpc/bf16_tile_gemm.rs:45` by name (D-GTM-0a).

## 2026-09-04 — W1.5 signature primitives: PR #293/#294/#295 landed, no board entry until now

Three merged PRs closing W1.5 signature-kernel work items went unrecorded on
this blackboard — corrected here.

**PR #293 — `hpc: signature_pde_sweep`** (merged), closes W1.5-#6
`TD-NDARRAY-SIMD-SIGNATURE-PDE-SWEEP`. Signature kernel `<S(X),S(Y)>` via the
Goursat PDE, anti-diagonal SIMD wavefront. **Correction it recorded:** the
consumer-contract doc sketched `f32`/`F32x16`; sigker is actually
`f64`/`Vec<f64>`. **Now wired:** lance-graph `crates/sigker/src/kernel.rs:35`
does `use ndarray::hpc::signature_pde::signature_pde_sweep;`.

**PR #294 — `hpc: randomized_signature_sweep`** (merged), closes W1.5-#7
`TD-NDARRAY-SIMD-RANDOMIZED-PROJECTION`. Cuchiero-Schmocker-Teichmann
randomized-signature recurrence on `F64x8`. New file
`src/hpc/randomized_signature.rs` + `examples/randomized_signature_bench.rs`.
Public API: `randomized_signature_sweep`, `randomized_signature_sweep_with`,
`randomized_signature_step`, `const INCREMENT_EPSILON = 1e-15`. Same
lane-type correction as #293 (doc sketched `F32x16`; real consumer is f64 →
`F64x8`); doc also wrongly sketched per-step re-derived Gaussians and a
single-register update — real shape is materialize-once buffers + runtime-k
`k×k` GEMV + axpy, `O(T·d·k²)`. Built only on already-parity-confirmed
`F64x8` methods (splat, from_slice, mul_add, reduce_sum, copy_to_slice) → zero
new arch-specific code, zero `unsafe`. [MEASURED] 2.00x @ k=32 up to 3.79x @
k=512 vs scalar; max rel err 1e-14; cross-backend contract is 1e-9 relative
tolerance, NOT bit-equality (reduce_sum order differs per backend). Second
commit `c129662` fixed a real bug CodeRabbit caught: the ragged-path guard
used `debug_assert_eq!`, which compiles out under `--release`; a wider later
path point would silently truncate and return a signature for the WRONG
path. Switched to `assert_eq!` + a `should_panic` test proving it fires in
release. **Loose end:** sigker's `RandomizedSignatureBuilder::encode` does
NOT yet delegate — still its own scalar loop
(lance-graph `crates/sigker/src/randomized.rs:95`). Wiring is in flight this
session.

**PR #295 — `hpc: docstring the randomized_signature tests, bench, and e2e
pipeline tests`** (merged, master `183c324`). Comments only, no behavioural
change. Motivation: CodeRabbit's docstring-coverage check on #294 reported
61.76% against an 80% threshold, scoped to functions touched by that diff (34
functions / 3 files) — gap was entirely test + bench code; all 3 public fns
and all 4 private compute helpers were already documented. Commit 1:
docstringed SplitMix64 methods, `wiggly_path`, `assert_matches_reference`, 10
of 11 test bodies, 8 of 9 bench fns in `src/hpc/randomized_signature.rs` +
`examples/randomized_signature_bench.rs`. Commit 2: docstringed the 7
`e2e_tests` pipeline tests in `src/hpc/mod.rs` (outside the check's scope —
#294's diff only added a `pub mod` line to that file — but the last
undocumented fns in the hpc surface). Result: 42/42 functions documented
across the three files. CI green: 13 non-skipped jobs incl.
tests/{stable,beta,1.97.1}, clippy/1.97.1, format/stable, wasm-simd,
neon-simd, tier4-avx512, nostd/thumbv6m. **Caveat worth recording:** the PR
was merged ~5s after leaving draft, so CodeRabbit never re-ran the coverage
check — the 100% figure is verified by a strict `///` scan, not by the bot's
own (looser) heuristic.

**Loose ends (all three PRs):**
- sigker randomized-signature wiring not yet landed (in flight, see #294).
- W1.5-#8 `TD-NDARRAY-SIMD-LYNDON-PACK` is the last unbuilt W1.5 primitive.
  Its gate (jc Pillar 11, Hambly-Lyons) IS activated — `jc/src/lib.rs:26`
  says "Pillar 11 activated 2026-05-07" — so #8 is unblocked, not deferred.
- The consumer-contract doc's W1.5 lane-type sketches have now been wrong
  twice (both #293 and #294). #8's `I16x16` sketch should be treated as
  unverified until checked against the real consumer.

## 2026-09-01 (latest) — Pillar-11 lattice lane: BIT-EXACT i128 lattice signature + Hambly–Lyons Thm 5/6 certificate

`src/hpc/pillar/lattice_signature.rs` (feature `pillar`). For unit-step
lattice walks every level-`k` signature coefficient is a rational with
denominator `k!`, so the lane stores `k!·S_k` as `i128` and the whole
computation is bit-exact — identity is `==`, no tolerance. Chen composition
with a unit step is a binomial convolution against a tensor supported on
`(a,…,a)` only, so each step costs `O(Σ_k d^k·k)`. Depth policy is INTEGER:
`theorem2_depth(L) = ⌈47917·L/10000⌉ ≥ ⌊2e·ln(1+√2)·L⌋` (the PUBLISHED
constant: Annals of Math 171 (2010) Theorem 5, `⌊2e·log(1+√2)·L⌋`; the
arXiv v2 preprint's Thm 2 states `e` — a version trap CodeRabbit caught on
lance-graph #1133 and the Annals PDF confirmed; the float never enters the
kernel), `theorem3_factor(d) = 2⌈log₃(d/2)⌉+3` by integer loop.
Measured (debug): 52/52 reduced `d=2` words of length ≤ 3 separated at the
theorem depth (`LATTICE_L_MAX = 3` — the doubled constant makes depth ⌊c·L⌋
grow to 19 at L=4, and `d^depth` coefficients per level exhaust memory); 64/64 tree-like words EXACTLY the identity at a fixed depth 12
(`LATTICE_TREELIKE_DEPTH`, since identity holds at every depth and the
theorem depth for length-6 words is 28); 64
reduced length-8 words share `S^(2) = 1` with the constant path (the
paper's §1.6 figure-of-8 class) and every one separates at level 3
(`3!·S_xxy = 6` for the canonical one); `d = 1` collapses the 64 length-6
words to exactly 7 tensors (net increment only — the `d ≥ 2` precondition
is now a pin, not prose). Parity pin against the existing f32 lane
`signature_d2_deg3` on every lattice word of length ≤ 6 (exact small
integers). Bit-exactness pin: FNV digest `0x7C9612A734212FC6` over all
words of length ≤ 3 at theorem depth. `prove_pillar_11_lattice()` reports
`psd_rate` = separated fraction (1.0), `n_paths` = 52, `n_hops` = 64
false merges, `lognorm_concentration` = deepest separation level (3).
**Disambiguation:** `signature.rs` stays the f32 depth-3 kernel-STABILITY
battery; this lane is the UNIQUENESS half, and it is the ndarray twin of
lance-graph `jc::hambly_lyons` W6 (PR #1133) with the f64 tolerance
replaced by integer equality. **SIMD:** scalar integer reference lane on
purpose; the W1.5 sigker vectorised lane (now unblocked) must reproduce
these `i128` tensors bit-for-bit. Loose ends: an `i128` lane in
`ndarray::simd` does not exist; the `d ≥ 3` arm of Theorem 3 is
implemented (depth formula) but not exercised by a test beyond the factor
pins; `crates/sigker-parity` should gain a W1b test comparing this lane
against `sigker::signature_truncated` on lattice words (exact ints vs f64).

## 2026-08-31 — W1a-#9 masking primitives SHIPPED on every dispatch arm (PR #285)

`U64x8`/`U32x16` gained `andnot` (set difference, `self & !other` — argument
order deliberately differs from the raw Intel intrinsic, same direction on
every backend) and `ternlog::<IMM>` (any 3-input boolean via the VPTERNLOG
truth-table immediate; single VPTERNLOGQ/D on avx512, const-folded minterm
composition elsewhere). Review round forced the completion that mattered:
the methods exist on ALL SIX arms (scalar, avx2, avx512, neon `U32x16`,
wasm `U32x16` via per-part `v128_andnot`, nightly via core::simd), the
named immediates live on the always-compiled facade as
`crate::simd::ternlog` (their first home in the scalar backend was compiled
out on x86 — three bots caught it independently), and every portable arm
carries the avx512-equivalent compile-time IMM domain guard. Doc examples
on all twelve method sites.

> **⊘ Correction (2026-08-31, codex P2 pair on the record PR — both
> accepted):** (a) the sentence that stood here claimed the avx512 doc
> examples "execute in this environment's doc-test run" — under the
> default v3 config the examples import the FACADE types, so what runs is
> default-facade coverage, not the avx512 backend; the claim is retired.
> The honest replacement is a MEASUREMENT: this host carries avx512f
> (cpuinfo), and all five w1a9 facade tests pass under
> `CARGO_BUILD_RUSTFLAGS='-Ctarget-cpu=x86-64-v4'` — that run IS the
> avx512 backend's runtime verification, hardware-executed, not inferred.
> (b) the scalar U64x8 ternlog was missing the IMM const guard the entry
> recorded as complete (U32x16 had it; U64x8 did not — `ternlog::<256>`
> compiled and silently truncated). Guard added with the same message as
> every other arm. Declined finding, reasons on the PR thread: extending the
pre-existing off-by-default nightly arm is REQUIRED (not extending it is
the E0599 hole), and the stable-only rule governs the default build graph,
which is untouched. Loose end, deliberate: whole-crate wasm compile-check
is blocked by the pre-existing getrandom wasm dependency gap — the
wasm-simd-parity sibling crate remains the verification home for that arm.
Consumer side: lance-graph's mask-algebra arc (D-MAR-1, #1099) is the
first caller; the codegen probe measured `vpternlogq $0x80` / `vandnps`
single-instruction lowering under v4.

## 2026-07-29 — blake3 dependency DROPPED: call sites swapped to the in-tree module

Operator: "go ahead" on the rung-3a swap, with the measured cost table in
hand. This closes the only rung of `the-simd-ladder.md` that carried a cargo
cycle (`blake3 → ndarray::simd`).

1. **15 call sites, 8 modules** (`seal`, `merkle_tree`, `plane`, `vsa`,
   `spo_bundle`, `crystal_encoder`, `compression_curves`, `deepnsm`) now
   reach `crate::hpc::blake3`. Implemented as one `use super::blake3;` per
   module rather than rewriting each `blake3::` path — the in-tree API is
   shape-compatible (`hash`, `Hasher::{new,new_keyed,new_derive_key,update,
   finalize,finalize_xof}`, `Hash::as_bytes → &[u8;32]`), so the diff is 10
   added lines, not 15 edited ones. `spo_bundle`'s use is inside
   `#[cfg(test)]`, so it takes `use crate::hpc::blake3;` there instead of
   leaning on the subtree's blanket `unused_imports` allow.
2. **Dependency gone.** `dep:blake3` off the `std` feature; the ~55-line
   Cargo.toml comment block replaced with a do-not-re-add note. `blake3`
   **and** its transitive `constant_time_eq` are both absent from
   `Cargo.lock`. nostd matrix untouched — every call site is under
   `pub mod hpc`, itself `#[cfg(feature = "std")]`.
3. **What this actually bought:** the second SIMD surface. Even at
   `default-features = false, features = ["pure"]` the crate shipped its own
   `rust_{sse2,sse41,avx2}.rs` intrinsics beside `ndarray::simd` — the exact
   thing the matryoshka pattern exists to prevent. The old Cargo.toml comment
   named this and said "Tracked, not done here."

**Cost, accepted with numbers in hand [MEASURED, not re-measurable]:**
1.25–1.39× across the input sizes this crate hashes (16 B 1.34–1.39×, 256 B
1.25–1.29×, 2 KB 1.30×); 4.7–4.9× at 64 KB, which no call site here touches.
The A/B bench needs both implementations present and is unbuildable from this
commit forward, by its own design note.

**Loose ends.**
- `AdaWorldAPI/BLAKE3` (v1.8.5) exists and is an **unmodified upstream
  mirror** — no ndarray wiring, all `ffi_*`/`rust_*` files intact. Master had
  been pinning `blake3 = { version = "1" }` from **crates.io** while that fork
  existed, i.e. the P0 fork rule was already broken. This change resolves it
  by deletion; wiring the fork instead would have been the other resolution
  and would NOT have closed the second-surface problem.
- Verified `src/hpc/blake3_test_vectors.json` is **byte-identical** (31,922 B)
  to that fork's `test_vectors.json`, so the correctness tests run against
  genuine upstream vectors.
- Rung 3b (`hash_many`) still absent; that is the whole 64 KB gap. Not worth
  building for ndarray's own call sites — nothing here hashes bulk.

## 2026-07-06 — F64 GEMM completed: FMA tier + register residency + native-engine swap

Operator: "complete the F64 gemm… and pr". Three moves, all on the entry
below's foundation:

1. **`gemm_f64_tiled_fma`** — fast fused tier via const-generic
   `gemm_f64_tiled_impl<const FMA: bool>` (monomorphized, no runtime
   branch). Per-element ascending-p order preserved; fused step
   `c = fma(α·a, b, c)`. Bit-identical to the reference tier on
   integer-valued operands (products+sums < 2^53 — asserted with
   `assert_eq!` on the full shape sweep); last-ulp-per-step differences
   on general floats (tolerance test scaled k·ε to SUMMAND magnitude —
   the initial result-scaled tolerance was wrong for cancellation-heavy
   elements and failed honestly). Cross-backend caveat documented (WASM
   without relaxed-simd has unfused vector lanes + fused scalar tail).
2. **Register-resident C row-block** (both tiers): C block loaded into
   `[F64x8; TILE/LANES]` accumulators + scalar tail array ONCE per
   (kk,ii,jj,i), whole kb-loop accumulates in registers, one store.
   f64→f64 store/reload never rounds ⇒ per-element op sequence
   unchanged ⇒ every bit-equality test green untouched. [MEASURED]
   ref 4.6→10.0 GF, fma 4.7→10.7 GF (2.2×).
3. **Engine swap:** `backend::native::gemm_f64` now routes to the
   crate-native tiled kernel — the f64 GEMM behind
   `BlasFloat::backend_gemm` / `hpc::blas_level3::blas_gemm` / batched
   linalg is entirely own Rust; matrixmultiply remains only in gemm_f32
   and upstream `Array::dot` (impl_linalg.rs, untouched at ~33 GF).
   **[REVISED post-verify] Engine = the UNFUSED reference tier**, not
   fma: the verify pass surfaced a cliff — the AVX2-polyfill/scalar
   `mul_add` lowers to a libm `fma()` call on baseline x86-64 builds
   (consumers do NOT inherit this repo's `.cargo` target-cpu pin; CI
   lands exactly there). The unfused tier has no libm dependence on any
   backend, costs only ~7% vs fused on pinned builds (10.0 vs 10.7 GF),
   and makes the backend engine bit-identical to the certification
   reference. `gemm_f64_tiled_fma` stays public for FMA-pinned
   consumers. New panic contract documented on `gemm_f64` (# Panics —
   checked preconditions vs the old wrapper's silent-UB on short
   slices; matches CBLAS xerbla behavior).

[MEASURED, 3-engine, this VM (v3 compile → AVX2 arm, PREFERRED_F64_LANES=4;
host runtime has avx512f but committed .cargo config is v3)]:
256³/512³/1024³ — ref 11.2/10.0/9.6 GF | fma 11.9/10.7/10.3 GF |
matrixmultiply 34.1/32.3/33.7 GF | max|fma−mm| ≤ 1.4e-13.
**Own-engine gap: ~3.1×** (was 6.6× pre-restructure). Trade accepted per
operator priority (own reverse-engineered Rust in the path, auditable
numerics); blast radius = hpc BLAS surface only. 2185/2185 lib tests
green WITH the swap.

[LOOSE END → next rung] Closing the 3× needs a real microkernel: B-panel
register reuse (i-tiling IR=2..4 × narrower j-block), then A/B packing —
the matrixmultiply Goto recipe, own-Rust edition. Also: `gemm_f32_tiled`
still dead in native.rs `mod scalar` (f32 sibling completion);
avx512f compile arm untested on CI (v3 config) — the F64x8=__m512d arm
runs only on local v4 builds.

**[VERIFY OUTCOME]** 3-angle adversarial pass on the completion diff:
numerics PASS / swap-trace PASS / docs FAIL→fixed. Substantive P1 acted
on: baseline-x86 libm-fma cliff → engine revised to the unfused
reference tier (see #3 REVISED above). Doc P1s fixed: two stale
"gemm_f64 delegates to matrixmultiply" claims (simd.rs comment +
gemm_f64_tiled rustdoc) contradicted the swap in the same diff. P2s
fixed: fma determinism scoped to per-(build,runtime) (wasm relaxed-simd
fusion is implementation-defined); AVX2 vfmadd naming corrected (per-
lane f64::mul_add polyfill, fused semantics); integer-corpus bound
comment corrected (k_max=128, ≈2.1e4). All gates re-run green after
fixes: clippy -D warnings, 2185/2185 lib tests, 4 gemm doctests.

## 2026-07-06 (later) — `ndarray::simd::gemm_f64_tiled` surfaced (operator directive)

The crate-native tiled f64 GEMM graduated from dead code
(`backend/native.rs` private `mod scalar`, zero callers) to the canonical
simd surface: `src/simd_ops.rs::gemm_f64_tiled`, re-exported
**unconditionally** in `src/simd.rs` (alloc-free; `pub mod simd` itself is
std-gated in lib.rs — see reviewer note below if that changed).

- **Bit-exactness contract (documented on the fn):** every C[i,j] gets
  `c = c + (α·A[i,p])·B[p,j]` ascending-p, mul and add UNFUSED — the
  `*`/`+` operators on F64x8 lower to plain mul/add intrinsics on ALL
  five backends (AVX-512 `_mm512_mul/add_pd`, AVX2 per-half, NEON
  `vmulq/vaddq_f64`, WASM `f64x2_mul/add`, scalar) and Rust never
  FP-contracts explicit intrinsics → bit-identical across backends; at
  α=1 β=0 bit-identical to the naive triple-loop reference.
- Innermost j-loop vectorized on dispatched `F64x8` (one source, every
  backend, per the simd_ops polyfill model); TILE=64 blocking preserved
  verbatim from the original.
- [MEASURED] vs naive scalar triple loop, single thread, this EMR VM:
  128³ 2.23×, 256³ 2.54×, 512³ 6.74× (4.6 GFLOP/s), **bit-equal: true**
  at every size (W1a bench criterion; well above the 0.5× reject line).
- W1a compliance: parity tests = 7 new tests in
  `simd_ops::gemm_f64_tiled_tests` (fixed-seed splitmix64 corpus, 13
  shape sweep incl. multi-tile 70³, strided lda/ldb/ldc with
  sentinel-padding assert, α/β semantics, β=0-over-NaN, k=0, m/n=0,
  denormals/−0.0) — all bit-equality (`to_bits`), not tolerance. Zero
  `unsafe`. No new feature detection. Consumer sites named: the
  `direct_matmul` f64 ground-truth in `examples/subpel_tap_tile.rs` +
  `examples/gridlake_field_tile.rs`, both REWIRED to it (bit-identical
  swap — subpel prints identical numbers pre/post).
- Free-fn shape note: matches the existing simd-surface GEMM family
  (`bf16_tile_gemm_16x16`, `matmul_i8_to_i32`) — operator-directed, not
  a speculative W1a-queue addition.
- Dead code removed: `native.rs` `gemm_f64_tiled` deleted (pointer
  comment left); `gemm_f32_tiled` stays dead in `mod scalar` pending the
  same treatment (UNUSED_INVENTORY thread).
- Gates: clippy `-D warnings` clean (lib + both examples), fmt clean,
  **2182/2182 lib tests**, doctest green, `--no-default-features` build
  green.

[NOTE] `--tests` clippy surfaces 3 PRE-EXISTING test-code lints
(property_mask.rs:426 unusual_byte_groupings, bitwise.rs:637 identity_op,
palette_codec.rs:806 needless_range_loop) — not touched here; the house
gate (`cargo clippy -- -D warnings`, no --tests) is clean.

[OBSERVED, then MEASURED] `amx_available()` flipped true→false between
runs two hours apart (subpel ran AMX TDPBF16PS earlier, F32x16 polyfill
later; identical BF16-class errors either way — tier ladder correct).
**Diagnosis via `examples/amx_probe` per AMX_GOTCHAS discipline** (initial
"enablement drift / Gotcha 14 adjacent" guess was WRONG): CPUID leaf-7
TILE/INT8/BF16 bits all false, `cpu_model() = OtherX86`, `has_amx() =
false` — the **silicon identity itself changed** (morning: EMR 0xCF with
AMX). The session container was rescheduled onto non-AMX/CPUID-masked
silicon. NOT Gotcha 4 (that signature is has_amx()==true with
available==false) and NOT Gotcha 14 (corruption while available==true).
Gotcha 9's always-print-the-tier discipline is what surfaced the flip.
Consequence for remote sessions: the host under an ephemeral container
can change mid-session — re-run `amx_probe`/`amx_report()` before any
AMX-tier claim, never carry `amx_available()` results across runs.

[LOOSE END] subpel_tap_tile findings 1-3 from the same-day review still
queued (Gotcha-14 assert guard, PackedBf16B throughput leg, positive-
operand comment fix). Adversarial 3-angle verify workflow on this diff
was in flight at commit time; findings (if any) land as follow-up.

**[VERIFY OUTCOME, same day]** 3-angle adversarial review (IEEE/bit-
exactness, W1a compliance, regression-trace): **PASS / PASS / PASS**.
One convergent P1 fixed in the follow-up commit: the simd.rs re-export
comment claimed no_std availability, but `pub mod simd`/`simd_ops` are
std-gated in lib.rs — the no_std build passes because the code is
compiled OUT (comment corrected; un-gating simd_ops for a genuinely
no_std kernel is possible future work, not this diff). P2s folded in:
cross-backend bit-identity scoped to non-NaN inputs (NaN payloads are
backend-defined, WASM may canonicalize); `alpha == 0.0` non-short-
circuit documented (0·Inf=NaN propagates, −0.0 can flip, unlike BLAS
quick-return); Panics doc states which checks are skipped (m/n==0, k==0);
length-extent asserts now overflow-checked (`checked_mul`); parity corpus
widened to 70+ invocations (full 13-shape sweep × 4 α/β combos) for the
W1a "50+" letter; free-fn-shape precedent sentence added to the doc
(GEMM family: bf16_tile_gemm_16x16, matmul_i8_to_i32). Deferred as
noise: x87-only i586 excess-precision footnote (tier-2, pre-SSE2).

## 2026-07-06 — Review pass: health check green + 10 findings on subpel_tap_tile (#235)

Review-only session (no code changes). **Health check:** `cargo fmt --check`
clean, `cargo clippy -p ndarray --lib -- -D warnings` clean, clippy on the new
example clean, **2175/2175 lib tests pass** (30 ignored). The stale top-of-
CLAUDE.md "build fails (exit 101)" note again does NOT reproduce. The example
`subpel_tap_tile` runs green end-to-end on this EMR host (tier = AMX TDPBF16PS,
rel err 0.157% / 0.215%, asserts pass).

**Findings on `examples/subpel_tap_tile.rs` (PR #235), most severe first:**
1. Lines 208-209 hard-assert rel err < 0.05 on AMX-dispatched results with no
   contention guard — flakes on oversubscribed VMs per Gotcha 14 (precedent:
   `#[ignore]` gating in bf16_tile_gemm.rs:572).
2. Throughput leg (3) times per-call allocs + f32→bf16 of BOTH operands + the
   kernel's per-call VNNI pack of the CONSTANT H — printed 1.00 M/s measures
   wrapper overhead, not the tile op; use `PackedBf16B` +
   `bf16_tile_gemm_16x16_packed` (the API built for exactly this).
3. Line 167 comment "(fits u8 and i8; positive operand)" is false (x reaches
   ≈ −19.6 for r ≥ 11) — porting hazard toward the u8×i8 int8 path.
4. Reuse: `mix()` is the 5th example-local splitmix64 copy (public bit-identical
   `hpc::cam_index::SplitMix64` exists); `direct_matmul` duplicates
   `backend::native::gemm_f64` — **[MEASURED this session] bit-exact, 0/256
   lanes differ** vs the naive loop on the probe's exact operands (operator
   correction: earlier "independence" refutation was wrong — the BF16 tile
   kernel shares zero code with gemm_f64).
5. Cleanups: Vec<f32> C + copy (kernel takes &mut [f32] — stack array works),
   to_bf16 double-alloc, b_pad identical-index loop = prefix copy, direct_hv
   duplicated FIR pass, padded-16→32 scaffold now copy-pasted across 2 probes
   (suggest one public padded helper on ndarray::simd before probe #3).

**Refuted during verify:** .gitattributes deletion is the deliberate,
documented revert PR #236 (union merge mangles [[example]] blocks — no residue
found); clippy needless_range_loop claim (empirically clean under -D warnings);
f32 checksum absorption (print-only anti-DCE); transpose_matrix reuse (net
loss — Vec round-trips vs 8-line stack helper).

[LOOSE END] Findings 1-3 are worth a small follow-up PR (contention guard or
warning, packed-B throughput leg, comment fix); none applied here (review-only).

**[ADDENDUM, same day — operator challenge "our gemm, not the stoneage
external":** the first bit-exactness probe compared naive-loop vs
`backend::native::gemm_f64`, which delegates to the EXTERNAL
`matrixmultiply::dgemm` (native.rs:249; registry dep, Cargo.toml:154). Re-ran
against the crate's OWN scalar `gemm_f64_tiled` (native.rs:473, verbatim):
**bit-exact too — 0/256 lanes differ** on the probe's operands AND on random
f64 at K=64. All three (naive / own tiled / matrixmultiply) agree bit-for-bit
on these shapes. **Structural finding the challenge surfaced:** the crate's
entire PUBLIC f64 GEMM surface (`gemm_f64`, `BlasLevel3::blas_gemm` via
`backend_gemm`) is external-backed, while the own-Rust `gemm_f64_tiled` sits
dead in the private `mod scalar` with zero callers. If the policy is
own-reverse-engineered-Rust-only, the right fix for finding #4 is to surface
`gemm_f64_tiled` (or route the scalar tier of `backend_gemm` through it) and
point probes at THAT — files under the UNUSED_INVENTORY dead-code thread.]**

## 2026-07-02 (later) — bf16 tile GEMM: VDPBF16PS middle tier + PackedBf16B (loose end closed)

Closed the [LOOSE END] from the 1BRC entry below. `hpc/bf16_tile_gemm.rs`
is now a three-tier ladder — **AMX TDPBF16PS → AVX-512 VDPBF16PS →
decode+FMA polyfill** — with the polyfill kernel (`simd_ops.rs`) untouched:

- **VDPBF16PS tier** (`avx512bf16_path`, private): bf16 pairs multiplied
  natively per zmm (no bf16→f32 decode), f32 lane accumulators, SAME VNNI
  operand layout as the AMX tile → one packed buffer serves both tile
  tiers. `_mm512_dpbf16_ps` verified stable on Rust 1.94. Runtime
  `is_x86_feature_detected!("avx512bf16")` (EMR box has it).
- **`PackedBf16B`** + **`bf16_tile_gemm_16x16_packed`**: VNNI pack (and
  its per-call allocation) hoisted out of hot loops; `vnni_index(row,col)
  = (row/2)·32 + 2·col + (row&1)` supports staging B DIRECTLY in VNNI
  layout (zero pack cost — the right shape for one-hot/sparse staging).
- **`bf16_tile_gemm_tier()`**: names the tier that will run (Gotcha 9
  reporting). Re-exports via `ndarray::simd::*` (W1a surface).
- **Exactness boundary preserved (operator condition):** bit-exact across
  ALL tiers for bf16-exact integer operands with accumulation < 2^24 —
  asserted with `assert_eq!` in the new parity tests (vnni_index vs
  vnni_pack_bf16; packed==unpacked==i64 reference; VDPBF16PS exact +
  tolerance-parity vs polyfill on floats; accumulate semantics). Gotcha-14
  contention parity test included as `#[ignore]` (fails on oversubscribed
  VMs BY DESIGN; run `--ignored` on dedicated silicon).

[MEASURED] onebrc probe GEMM leg with direct-VNNI staging: **3.6 → 21.3
Mrows/s (5.9×), 23.7 → 141.9 GMAC/s** (single thread — near the 169.7
GMAC/s int8 AMX anchor in AMX_GOTCHAS). 413/413 stations still EXACT;
8/8 lib tests + 2 doctests green; clippy/fmt clean.

[NOTE] Dispatch-behavior change signed off by operator: the row-major
entry `bf16_tile_gemm_16x16` now routes avx512bf16-without-AMX hosts
through VDPBF16PS instead of decode+FMA (bit-exact within the integer
boundary; BF16-precision-class accumulation-order differences on general
floats, same as any tier change).

[ADDED, same day] **LE byte contract on `PackedBf16B`** (operator "Go" —
first brick of the SoA-Morton batch-writer / write-hiding design):
`as_le_bytes()` (zero-cost reinterpret; LE by construction — the module
is x86_64-only) + `from_le_bytes()` (endian-correct anywhere, plain copy
on LE). This is the persistence/mailbox face per lance-graph's
SoaEnvelope discipline (envelope bytes LE from creation to tombstone).
Test `le_byte_view_roundtrips_and_is_truly_le` asserts byte 2i = low
byte of lane i AND that a GEMM over the roundtripped buffer stays
bit-exact. 9/9 lib tests green. Next bricks (lance-graph side): batch
writer flushing tile buffers as envelope tenants; write-hiding = stage
morsel N+1's VNNI writes while morsel N's tiles compute.

## 2026-07-02 — 1BRC-on-substrate probe (`examples/onebrc_cascade_probe.rs`)

1BRC workload (min/mean/max per station) restated on the substrate, as a
sibling of `morton_cascade_probe`. Branch `claude/1brc-lance-graph-xfx5tu`.
Three paths certified bit-for-bit against a scalar integer reference
(413 stations, integer tenths → exact in f32/f64 by construction):

- **Morton scatter**: stations minted as cells on a 64×64 Morton grid
  (4×4 tile = one F32x16), morsel-batched (64K rows) scatter into
  L1-resident SoA accumulators, (min,max,Σ,n) monoid fold.
- **AMX BF16 tile-GEMM group-by**: (Σ,n) as `C += A[16×K]·B[K×16]` via
  the NEW `ndarray::simd::bf16_tile_gemm_16x16_amx` re-export (W1a: the
  AMX-dispatching hpc wrapper surfaced through the canonical polyfill,
  same pattern as `matmul_i8_to_i32`; the `_amx` suffix disambiguates
  from the pure-FMA `simd::bf16_tile_gemm_16x16`) — B = per-row one-hot
  station indicator (26 column-blocks of 16), A rows = {1, hi(t), lo(t),
  bf16-RNE(t)} with the exactness split `hi=(t/256)·256, lo=t−hi` (both
  bf16-exact; f32 tile accumulation exact for K=4096). Clear-by-undo
  keeps B staging O(rows). AMX **actually ran** (amx_available()==true
  printed per Gotcha 9 discipline; EMR-class Xeon, kernel 6.18.5).
- **Aggregate pyramid** over the tile grid: hierarchical (min,mean,max)
  per tile/region/root in the same pass + band-prune queries
  (Belichtungsmesser on the MIN channel).

[MEASURED] 10M rows, 4-core Xeon EMR VM, single thread:
reference 453 Mrows/s | morton scatter 443 Mrows/s (**substrate tax ≈ 2%**)
| tile-GEMM 3.6 Mrows/s = 23.7 GMAC/s (dense one-hot indicator = the
honest price of group-by-as-matmul; per-call `vnni_pack_bf16` alloc in
`bf16_tile_gemm_16x16` is a visible overhead) | pyramid fold 0.02 ms |
band query prune 90.2%. All 413 stations EXACT on both paths; PASS.
Also EXACT at 100M rows (idle). **"Is BF16 precise enough?" — measured:**
the naive bf16-RNE row through the same tile gives max per-station
|Δmean| = 0.0123 tenths (0.0012 °C, N≈24k/station — quantization bias
averages out); single readings off by ≤ 2 tenths (half-ulp of bf16 at
|t|∈[512,1024)). Verdict: bf16-direct fine for means, hi/lo split (free —
spare A rows) required for min/max + exactness certification.

[FINDING → **Gotcha 14**, `.claude/AMX_GOTCHAS.md`] On this oversubscribed
VM, **AMX tile state silently corrupts under host CPU contention**: idle
= 413/413 exact at 100M rows; with 4 busy-loop competitors = 89-152/413
(whole rows lost, no fault); guest-side core pinning does NOT mitigate
(124/413); AVX-512 scatter path in the same run stays exact → isolated
to TMM state; suspected host-vCPU-switch XTILEDATA loss. Consequences
written into the gotcha: never certify AMX numerics on shared VMs; parity
tests must also run under deliberate load (Gotcha 9 extension); short
tile residency = harm reduction only.

[CROSS-REPO] Algebraic certification (partition/regroup invariance of the
monoid fold, bf16 hi/lo decomposition exactness) lands as a diagnostic
probe in `lance-graph/crates/jc` (`onebrc_agg`) — kernels here, proof
there, per the architecture rule (ndarray = hardware, jc = proof).

[LOOSE END] AMX has no min/max tile op → min/max stay on the scatter
path by construction. `bf16_tile_gemm_16x16` allocates + VNNI-packs B on
every call — a pre-packed-B variant would lift the GEMM leg
substantially; file under W1-adjacent if the group-by-as-GEMM shape
recurs. Text-ingest leg (SWAR/SIMD parse of the 13 GB file) deliberately
NOT probed here — separate probe if pursued (would exercise
`byte_scan.rs`).

## 2026-06-28 — WASM SIMD128 backend filled in (`src/simd_wasm.rs`)

Replaced the commented-out scaffolding in `src/simd_wasm.rs` with a real
`core::arch::wasm32` SIMD128 backend, mirroring `simd_neon::aarch64_simd`'s
proven split (native v128 for the float/byte hot path, scalar fallback for
the long tail). Branch `claude/ndarray-wasm-scalar-zr9n46`.

**`src/simd_wasm.rs::wasm32_simd`** (gated `#[cfg(all(target_arch="wasm32",
target_feature="simd128"))]`):
- `F32x16` / `F64x8` as `[v128;4]` + `F32Mask16` / `F64Mask8` — full API
  parity with the scalar macro (splat/from_slice/from_array/to_array/
  copy_to_slice/reduce_{sum,min,max}/abs/sqrt/round/floor/mul_add/
  simd_{min,max,clamp,lt,le,gt,ge,eq,ne}/to_bits/from_bits/cast_i32 +
  Add/Sub/Mul/Div/*Assign/Neg/Debug/PartialEq/Default + Mask::select).
- `I8x16` (one `v128`) = UNION of the scalar + NEON method sets
  (add/sub/min/max/cmp_gt + from_i4_packed_u64/lane_i8/saturating_abs)
  so consumers are portable across every backend.
- Free hot-kernels (v128 counterparts to the NEON kernels):
  `dot_f32x4_wasm`, `popcount_u8x16_wasm`, `hamming_u8x16_wasm`,
  `hamming_u8x64_wasm` (Fingerprint<256> distance via `i8x16_popcnt`),
  `base17_l1_wasm`, `codebook_gather_f32x4_wasm`, `bf16_to_f32_batch_wasm`.
- `mul_add`: `f32x4_relaxed_madd` under `+relaxed-simd`, else mul+add
  (base simd128 has no FMA). `round()` = `f32x4_nearest` (ties-even, =NEON).
  NaN in simd_min/max follows IEEE (NaN-propagating, =NEON); the existing
  `simd_exp_f32` NaN save/restore already absorbs this. All documented.

**Dispatch (`src/simd.rs`):** new `target_arch="wasm32" + target_feature=
"simd128"` arm re-exports the 8 native names from `wasm32_simd` and the
remainder from `scalar`; the "Other non-x86" arm now excludes that case
(wasm-without-simd128 + riscv etc. stay full-scalar). Added wasm32
`PREFERRED_*_LANES` arms (F32=4/F64=2/U64=2/I16=8, 128-bit widths) and a
`.cargo/config-wasm.toml` (`-Ctarget-feature=+simd128`).

**Unblocked the wasm build (pre-existing x86 leaks, not SIMD-scaffolding):**
the crate did NOT compile for wasm at all — `src/simd.rs` re-exported the
x86-only `amx_matmul` / `simd_amx` modules unconditionally, and
`backend::gemm_bf16` called `amx_matmul::matmul_bf16_to_f32` directly.
Gated both re-exports to `#[cfg(target_arch="x86_64")]`; split `gemm_bf16`
into the IDENTICAL x86 AMX path + a non-x86 branch routing through the
portable `hpc::quantized::bf16_gemm_f32(.., 1.0, 0.0)` (the same scalar
reference the AMX dispatcher itself falls back to → bit-equivalent). x86
behavior is untouched by construction (the original block now lives under
`cfg(target_arch="x86_64")`).

[VERIFICATION] (1) `cargo build -p ndarray --lib` for wasm32 **+simd128**
(native) AND **without** simd128 (scalar) AND **--no-default-features**
(no_std) AND x86_64 default — all green. (2) A standalone faithful copy of
`wasm32_simd` built to wasm32+simd128 and run under **node**: 51 numeric
checks (incl. exact mask bit-patterns, saturating_abs(i8::MIN)=127,
Hamming=512, Base17 vs scalar incl. a pathological |a-b|=60000 overflow
case, bf16 shift) all PASS. (3) x86 regression: 217 SIMD tests + 85
backend/bf16 tests pass; `clippy -p ndarray --lib -- -D warnings` clean;
`fmt --check` clean. Harness: `/tmp/.../scratchpad/wasmverify`.

[ADVERSARIAL REVIEW] Ran a 3-angle Opus review (cfg-gating / intrinsic-
semantics / x86-regression). x86-regression = PASS (x86 path byte-identical;
non-x86 bf16 fallback bit-equivalent). Two findings resolved: (P0 cfg-gating
"no_std arm break") = **false positive** — `pub mod simd` is itself
`#[cfg(feature="std")]` (lib.rs:239), so the native wasm arm is transitively
std-gated; `--no-default-features` wasm build is clean (empirically
confirmed). (P1 base17 i16 wrap) = **real, fixed** — `base17_l1_wasm` now
sign-extends i16→i32 via `i32x4_extend_{low,high}_i16x8` BEFORE the subtract,
so `|a-b|` is computed in i32 and matches the scalar reference for the full
i16 range (the prior i16-domain abs-diff, like NEON's `vabdq_s16`, wrapped at
|a-b|>i16::MAX). Doc nits (mul_add ULP wording, reduce_sum order, Tier-enum
comment) also tightened.

[NOTE] The stale top-of-CLAUDE.md "Build currently fails (exit 101)" no
longer reproduces — x86 lib builds clean this turn.

[LOOSE END] Full-crate (workspace) wasm build still blocked by `getrandom
0.3` (via `ndarray-rand`/`numeric-tests`, members that depend ON ndarray)
needing the `wasm_js` backend — orthogonal to this work; `-p ndarray --lib`
is the correct wasm surface and it is green. `bf16_to_f32_batch_wasm` is
provided + tested but NOT wired into the `bf16_to_f32_batch` dispatch (left
scalar to keep the BF16 path untouched); wire it if a wasm BF16 hot path
appears. Native U8x64/I32x16/U64x8 stay scalar on wasm (same as NEON keeps
them scalar) — the free Hamming/Base17 kernels cover those hot paths.

## 2026-06-17 — DECISION: HHTL fork ladder coded in `hpc::entropy_ladder` (CONJECTURE)

Reified the operator's standing idea — *if the orthogonal (helix/CAM-PQ)
leaf residue is strong enough, free energy forks into another domain
(HHTL shift = new exploration)* — as pure functions beside the existing
entropy/quadrant code. Unifies four vocabularies as one 2-axis structure:
`entropy_ladder::Quadrant(entropy,energy)` ≡ `lance-graph-contract::mul::
FlowState(challenge,skill)` (Csikszentmihalyi) ≡ Friston model-vs-surprise
≡ Staunen↔Wisdom.

- `residue_surprise(mag, noise_floor, sigma_k) → [0,1]` — orthogonal residue
  magnitude (prediction error the in-domain centroid codebook fails to
  explain) → challenge axis. Below floor = quantization (≈0); linear ramp
  over `sigma_k·noise_floor`. Threshold provenance per `I-NOISE-FLOOR-JIRAK`
  (Berry-Esseen wrong under CAM-PQ weak dependence); ramp is an honest proxy
  pending Jirak calibration, **not** a claimed bound.
- `ForkAction {Commit, DescendDeeper, ForkBasin, ForkDomain}` + `fork_decision`
  — bands challenge−skill on the shipped `mul::flow_state_from` boundaries
  (Anxiety δ>0.2, Boredom δ<-0.2; the matched middle |δ|≤0.2, which
  flow_state_from splits into Flow/Transition, collapses to one in-domain
  branch here); HHTL depth decides descend-vs-
  fork. `ForkDomain` (mint a new classid = the Friston model-switch) requires
  BOTH leaf depth AND challenge≫skill — the "strong enough AT THE LEAF" invariant.

Layering kept honest: the `FlowState` *assessment* stays in lance-graph
(thinking); the fork *math* lives here in ndarray (substrate, where residue +
energy physically are) — per the Architecture Rule. Pure fns + one enum, no
struct/layer, composes with `Quadrant`. 5 lib + 2 doctests, clippy clean.
Branch `claude/jirak-math-theorems-harvest-rfii13`.

**Loose ends (CONJECTURE → gated):** (a) feed the *real* `edge_codec::
CoarseResidue` magnitude from the live codec into `fork_decision` (currently
caller-supplied); (b) `ForkDomain` vs `ForkBasin` should be arbitrated by
residue *orthogonality* (⊥ all in-domain centroids = genuinely new), not the
depth+delta proxy; (c) Jirak-derived σ threshold to replace the `sigma_k`
proxy. This driver-side wire merges with lance-graph `materialize`'s
`ThoughtCtx::from_live` step (same call-site).


## 2026-06-10 — DECISION: GUID prefix→shape routing crystallized (docs-only)

The operator-pinned canonical GUID (`OGAR/CLAUDE.md`: hex dash-groups =
`classid(8)-HEEL(4)-HIP(4)-TWIG(4)-[basin·leaf+id]`; 3×4 tiers, `>> 2`)
now has its ndarray-side contract at
`.claude/knowledge/guid-prefix-shape-routing.md`: ndarray = MECHANISM
(layout-only `PrefixShapeTable`, opaque `ShapeId(u16)`, longest-prefix,
L1/L2-resident, no distance API — no-umbrella honored), consumer =
POLICY (lance-graph registers the table). GridLake continuation: key
selects grid family + pyramid level; value stays one byte-store
(column-substrate identity). φ-quorum anti-eigenvalue-theater contract
pinned with the PP-13 casebook as failure catalog; probes named
(ROUTE-1, QUORUM-1, PHI-1, PYR-1, CODEBOOK-44; HILBERT-L4 = existing
P0-4 blocker for any L4 cascade claim). CONJECTURE until coded — no
.rs touched in this commit.

## Evidence model (binding — from PR #200)
- **L0** = source · passing tests · ratified standards (ground truth).
- **L1** = `.claude/PR-X12-docs-audit.md` (#200) + `.claude/knowledge/plans-alignment-triage.md` — claims-about-source; **spot-check, never inherit**.
- **L2** = `.claude/plans/*` + `pr-x12-*` perspective docs = **inspiration, NOT evidence**.
- **Whole-file reads only** — no `grep`/`sed`/`head`/`tail` (`ls` to locate).
- Build/bench locally at `target-cpu=x86-64-v4`; committed `.cargo/config.toml` stays **v3** (GitHub/CI).

## Settled architecture (grounded this epoch, whole-read)
- **Cognitive similarity/cosine = Palette256 + Fisher-z**, integer: `hpc::cam_pq`
  squared-L2 ADC (u8 codebook indices) gated by θ = `distance::similarity_z` (atanh).
  Validated 10k×10k @ θ≈cos-0.90. **No float MAC in the distance kernel.**
- The cognitive **"splat"** = `lance-graph-contract::splat::CamPlaneSplat` (q8) →
  `AwarenessPlane16K` (16 384-bit OR deposition). **Sibling of, not the same as,** the
  graphics `splat3d` EWA renderer (per `splat3d/mod.rs`).
- EWA float-Σ sandwich (`splat3d::spd3`, Pillar 6/7) = uncertainty propagation +
  certification, **not** similarity. Pillar suite (6–17) certifies the substrate;
  **Pflug-10 certifies the CAM-PQ palette**.
- Typed distance (`cognitive-distance-typing.md`): one named fn per metric, newtype
  outputs, **no `fn distance<T>` umbrella**, conversions explicit. `palette→fisher→
  cosine→hamming` roundtrip is the named anti-pattern.

## Outstanding (per triage + #200)
- **#4** pr-x12 doc-fixes + evidence-policy + archive fabrication-heavy plans (Geo/Gov).
- **#5** ASG-leaf canon spec (Gov) — prerequisite for #7.
- **#7** ASG-leaf impl (Kernel) — must **extend `CamPlaneSplat`**, not reinvent; trails #5.
- `cam-pq-production-wiring` (UNOWNED, lance-graph) — route `cam_pq` through `CamCodecContract`.
- `UNUSED_INVENTORY_1.95` A1–A9 dead-code (phantom `SimdTier::{Sse2,WasmSimd128}`, stale 1.64 imports).

## Consolidation-sprint debt (PR-X program; ground-truthed `ls src/hpc/` 2026-05-27)
> Shipped-state vs `pr-master-consolidation.md`. Landed: ✅ **PR-X10** `linalg/`,
> ✅ **PR-X11** `pillar/`, ✅ **PR-X13** `ogit_bridge/`, ✅ **PR-X3** `blocked_grid/`.
- **PR-X12 codec ⚠️ v1 NEAR-COMPLETE** — `ctu/mode/predict` + now **`rdo` (A6, λ-RDO,
  integer fixed-point λ_q8 — no float) + `ans` (A7, static-table rANS over the 4-symbol
  mode alphabet, bit-exact round-trip)**. Remaining: `transform` (A4, deferred to v2 per
  design Q2) + `stream` (A8, framing over `ans`). 81 lib + 20 doctests green, clippy clean.
- **PR-X4 splat4d ❌ OUTSTANDING** — no `src/hpc/splat4d/`. Unbuilt.
- **PR-X9 cognitive ❌ OUTSTANDING** — no `src/hpc/cognitive/`. Unbuilt; must **consume**
  `lance-graph-contract::splat::CamPlaneSplat` (q8), never redefine it (contract is sacred).

## Merged / closed this epoch
- ✅ #201 triage · #205 `3dgs-tiles` (cesium tileset) · #206 + #208 render-depth cert.
- ❌ #207 EWA-SYRK bench **closed** (wrong regime — category error).
- 🗑 `phi_spiral.rs` abandoned (float, wrong manifold). Net new usable code this
  session = 0 (see `board/EPIPHANIES.md` grounding-discipline entry).

---

# Polyglot Notebook — Single Binary Architecture

> Separate/older program — NOT the current epoch (see top of file).

## The Binary

One `cargo build`. Ships as one executable. Contains:

```
reactive runtime     (transcoded from marimo Python)
graph query engines  (transcoded from graph-notebook Python)
kernel protocol      (Rust-native ZMQ, from kernel-protocol spec)
document publisher   (transcoded from quarto TS/Deno)
local graph database (lance-graph, already Rust)
SIMD kernels         (ndarray, already Rust)
graph compiler       (rs-graph-llm, already Rust)
web frontend         (marimo's JS/React, served by the binary)
```

External process: R only (Bardioc/almato). Speaks Arrow IPC to the binary.

## Repos → Crates

| Repo (source) | Becomes | Work |
|------|---------|------|
| marimo | `crate::runtime` + `crate::server` | Transcode Python→Rust |
| graph-notebook | `crate::query::{cypher,gremlin,sparql,nars}` | Transcode Python→Rust |
| kernel-protocol | `crate::kernel` | Implement from spec in Rust |
| quarto | `crate::publish` | Transcode TS→Rust |
| quarto-r | external R process | Stays R, Arrow IPC bridge |
| lance-graph | `crate::graph` | Already Rust, integrate |
| ndarray | `crate::simd` + `crate::linalg` | Already Rust, integrate |
| rs-graph-llm | `crate::compiler` | Already Rust, fix build |

## Scopes (parallel, non-overlapping)

### SCOPE A: Reactive Runtime (marimo → Rust)
Transcode marimo's reactive cell execution model to Rust.
The core insight: cells have dependencies, when a cell's input changes,
downstream cells re-execute. That's a DAG scheduler — natural in Rust.

### SCOPE B: Query Engines (graph-notebook → Rust)
Transcode graph-notebook's Cypher/Gremlin/SPARQL executors to Rust.
Bolt protocol client, WebSocket client, HTTP client — all Rust-native.
Add local path: Cypher → lance-graph semiring (no network).

### SCOPE C: Kernel Protocol (kernel-protocol spec → Rust)
Implement Jupyter kernel wire protocol in Rust.
Only needed for R (IRkernel) — everything else runs in-process.
ZMQ via zeromq-rs. Connection file parsing. Message ser/de.

### SCOPE D: Publisher (quarto TS → Rust)
Transcode Quarto's document rendering pipeline to Rust.
Pandoc AST manipulation. Markdown → PDF/HTML.
Custom graph visualization extension.

### SCOPE E: Integration (lance-graph + ndarray + rs-graph-llm)
Wire the existing Rust crates into the binary.
Fix rs-graph-llm build. SIMD kernels for graph ops.
This is mostly Cargo.toml workspace wiring + API surface.

## Decisions
[DECISION] One binary, no Python runtime
[DECISION] marimo's JS frontend served by Rust HTTP server (axum/actix)
[DECISION] R is the ONLY external process (Arrow IPC bridge)
[DECISION] Cypher executes locally via lance-graph semiring by default
[DECISION] Remote DB connections (Neo4j, FalkorDB) via native Bolt client
[DECISION] vis.js graph rendering served as static assets by the binary

## Architecture Decisions

### 2026-06-13 — GEMM-dispatch routing fixes (savant-architect)
Branch `claude/wonderful-hawking-lodtql`. Three public GEMM entry points
were not routing to the accelerated kernels.

- **`backend::gemm_bf16` (src/backend/mod.rs)** — ALREADY FIXED in the
  working tree this session. Now routes to
  `hpc::amx_matmul::matmul_bf16_to_f32` (AMX `TDPBF16PS` → AVX-512
  `VDPBF16PS` → scalar). Slice→ArrayView2 wrapping mirrors the call shape
  in `simd_runtime::matmul`; inputs sliced to exact `m*k`/`k*n`/`m*n`.
  Bit-equivalent on non-AMX/non-AVX512BF16 hosts because the dispatcher's
  scalar fallback is the same `quantized::bf16_gemm_f32(a,b,c,m,n,k,1.0,0.0)`
  the old direct call used (alpha=1, beta=0 preserved).
- **`backend::gemm_i8` (src/backend/mod.rs)** — ALREADY FIXED in the
  working tree this session. Routes to `simd_int_ops::gemm_u8_i8`
  (4-tier: AMX `TDPBUSD` → VNNI-zmm → AVX-VNNI-ymm → scalar).
  [DECISION] Deliberately NOT routed to `amx_matmul::matmul_i8_to_i32` as
  the literal task text asked: `gemm_i8` is **u8×i8→i32**, but
  `matmul_i8_to_i32` is **i8×i8→i32** and would reinterpret A-bytes ≥128
  as negative — NOT bit-equivalent. `gemm_u8_i8`'s scalar fallback is the
  same `quantized::int8_gemm_i32` the old `vnni_gemm::int8_gemm_vnni`
  used → bit-identical on scalar hosts; VNNI-zmm arm calls the same
  `int8_gemm_vnni_avx512` kernel as before. All tiers integer-exact.
- **`native::gemv_f32` / `gemv_f64` (src/backend/native.rs)** — FIXED
  THIS TURN (was calling `scalar::gemv_*` unconditionally). Now matches
  on `tier()`: Scalar tier → unchanged `scalar::gemv_*` (byte-identical);
  Avx2/Avx512 tiers → per-row `dot_f32`/`dot_f64` (the existing
  dispatched, parity-tested SIMD dot). GEMV = stack of row dots; each A
  row is row-major-contiguous so contiguous `dot_*` loads apply. Leading
  `n` of each `lda`-wide row taken via `&a[i*lda..i*lda+n]`; no new bounds
  requirement vs scalar ref. SIMD tiers carry the module's documented
  1-2 ULP reduce-order drift (within BLAS tol; `test_gemv_f32` uses 1e-5,
  no byte-exact consumer asserts gemv).

[UNSAFE-AUDIT] gemv fix added **zero** new `unsafe` — it reuses the
already-audited `dot_*` kernels. No new sentinel-qa surface from this turn.
The two mod.rs fixes contain `unsafe` repr(transparent) slice reinterprets
(BF16/u16) that were landed earlier this session and warrant the standard
sentinel-qa pass if not already covered.

[LOOSE END] Repo references modules that exist on disk but the Glob/Grep
index was transiently stale this session (returned empty for
`simd_int_ops.rs`, `vnni_gemm.rs`, `bf16_gemm_f32`); Bash ground-truth
confirmed all present. Orchestrator should `cargo fmt`/`clippy`/`test`
centrally (edits were edit-only, no compile performed here).

---

## 2026-07-11 — U32x16 ARX lane + ChaCha20 matryoshka (see .claude/CHACHA20_MATRYOSHKA_PLAN.md)

**DONE:** `ndarray::simd::U32x16` full ARX triple (Add/BitXor/**rotate_left**) on
every tier — avx512 native `_mm512_rolv_epi32`, native wasm `[U32x4;4]`, avx2/
scalar/nightly. Node-run wasm parity CI gate (`wasm_simd` job + `scripts/wasm-parity.sh`
+ `crates/wasm-simd-parity/`, workspace-excluded, tests the real types, no drift).
Interim `src/simd_crypto.rs` (chacha20 scalar+avx512+wasm128, RustCrypto-parity-proven)
is superseded by the matryoshka once the fork lands.

**TO-DO (deferred, token limit):**
1. Native neon `U32x16 = [U32x4;4]` (extend `U32x4(uint32x4_t)` w/ xor+rotl, compose,
   fix F32x16 to_bits, swap simd.rs aarch64 arm).
2. aarch64 cross (qemu) parity CI job — generalize `wasm-simd-parity` to a shared
   `simd-parity` harness; closes NEON's x86-suite blind spot.
3. avx2 native `U32x16` (2×__m256i, TD-SIMD-3) — optional.
4. **Matryoshka execution:** fork `chacha20`, clone `avx2.rs` backend → rewire over
   `ndarray::simd::U32x16`, `[patch]` into the encryption stack (transitive accel),
   gate vs RustCrypto `soft`, retire `simd_crypto.rs`. Full plan in the doc above.

---

## 2026-07-12 — PR #240 CI fully green (no_std/MSRV fix chain closed)

Tip `5a914c37`. All 20 checks pass; the three previously-red failures are resolved:

- **261f736a** — `simd_crypto` dispatcher: `is_x86_feature_detected!` → compile-time
  `#[cfg(all(target_arch="x86_64", target_feature="avx512f"))]` (no runtime detection;
  the workspace SIMD-dispatch rule). Unblocked `blas-msrv` + `nostd/thumbv6m`.
- **5a914c37** — three no_std/bare-`--no-default-features` test-build fixes:
  - `simd_crypto` tests: `vec![[0u8;64]; n]` → fixed arrays `[[0u8;64]; 40]` / `[[0u8;64]; 16]`
    (no `vec!` under no_std).
  - `tests/chacha20_rustcrypto_parity.rs`: added `#![cfg(feature = "std")]` (imports
    std-gated `ndarray::simd`; no-ops under `--no-default-features`).
  - `src/tri.rs`: `Array2::<i32>::zeros(...)` (serde_json dep-drift added
    `impl PartialEq<Value> for i32`, making bare `Array2::zeros` element-type ambiguous).

Green jobs of note: `tests/{stable,beta,1.95.0}`, `blas-msrv`, `nostd/thumbv6m-none-eabi`,
`clippy/1.95.0`, `format/stable`, `native-backend/stable`, `tier4-avx512-check`,
`wasm-simd/parity-node` (new gate), `hpc-stream-parallel/rayon`, CodeRabbit.

Deferred (task #30, token-limit call): native neon `U32x16=[U32x4;4]` + aarch64 cross
parity CI + the matryoshka chacha20 fork. Plan in `.claude/CHACHA20_MATRYOSHKA_PLAN.md`.

---

## 2026-07-12 — Matryoshka finalized + NEON cross-CI (deferred #30 CLOSED)

- **Native NEON `U32x16 = [U32x4;4]` ARX lane** (commit 06a61bf9): bitxor/
  rotate_left on U32x4, composed U32x16 Add/BitXor/rotate_left; also fixed the
  pre-existing aarch64 stable-compile breakage (`u16x8` alias; nightly-only
  `vdotq_s32` → stable widening NEON). aarch64 now `cargo check`-clean on stable.
- **ChaCha20 matryoshka + `simd_crypto.rs` RETIRED** (commit 20dc6c3f):
  `vendor/chacha20/` fork (name/version kept, own [workspace]); the ONE delta is
  `backends/ndarray_simd.rs` — the transpose block16 over `ndarray::simd::U32x16`
  (pure +/^/rotate_left, no intrinsics, no unsafe), compile-time-selected under
  cfg(x86_64+avx512f), `[patch.crates-io]`-folded under encryption. Triple parity
  gate GREEN (fork RFC 8439 vectors through ndarray_simd @ v4; encryption 23 AEAD
  tests @ v3+v4). Deleted src/simd_crypto.rs + the parity test + the chacha20 dev-dep
  + the ndarray::simd::chacha20_* surface.
- **NEON cross parity CI**: `crates/neon-simd-parity` (excluded bin) +
  `scripts/neon-parity.sh` (cross-build aarch64 + run under qemu-aarch64-static) +
  CI `neon_simd` job (added to conclusion needs). Runtime-verifies U32x16 ARX /
  F32x16 / I8x16 == scalar on real aarch64. Green locally under qemu.

Follow-ups (documented in `.claude/CHACHA20_MATRYOSHKA_PLAN.md`): wasm matryoshka
backend (simd128 branch); cross-repo `[patch]` for MedCare-rs; the workspace
default is x86-64-v3 (avx2) so ndarray_simd activates on avx512 builds only.

---

## 2026-07-16 — PR-X12 x265/x266 plan review: audit applied + H.267 standards grounding

- **PR-X12-docs-audit corrections finally APPLIED** (they had sat unapplied
  since 2026-05-22): fabricated symbols marked ([PLANNED] `batched_ssd_search`;
  `blasgraph::tropical_gemm` / `bgz17::tropical_spmv` removed — real min-plus is
  method `ScalarCsr::spmv_min_plus`, lossy sibling); blasgraph restored as
  bit-exact canon over bgz17; per-arch DCT crossovers tagged [UNCALIBRATED];
  false `signature_kernel_pde` Goursat-bug claim withdrawn (its convergence
  tests pass); the R-11 unit is leaves-at-8×8 not CTUs, and the count itself
  was corrected to 129,600 exact (padded 130,560; the old 132,710 was
  ungrounded) → ~129 ns/leaf budget; §9 falsifiability matrix tagged
  FORWARD-CONDITIONAL. Tier-1 docs (`woa-multiarch-orchestration`,
  `bgz-jc-substrate-synergies`) ⛔ QUARANTINED pending rewrite.
- **NEW: `.claude/knowledge/pr-x12-h266-h267-standards-landscape.md`** —
  sourced public-standards anchor: H.266/VVC (2020, ~40-50% over HEVC, dec
  1.5-2×/enc ~10×), ECM-16.1 (~27% over VTM, complexity flagged impractical),
  NNVC v7 (NN in-loop ≈9% RA each — the antithesis of our anti-neural rule),
  H.267 (CfP Jul 2026 → submissions Nov 2026 → evaluation Jan 2027 → finalize
  ~2028; requirement ≥40% over VVC Main 10 at 4K+). "x266" in our docs =
  PR-X12 3DGS scene codec, never H.266.
- x266 lens doc got §12 reality-check addendum + F-3b falsifier
  (conventional-plus-neural acceptance risk); capstone + 3DGS plan index got
  standards-watch sections. Watch dates: Nov 2026, Jan 2027.
- Status note: `src/hpc/codec/` now has `ans.rs` + `rdo.rs` (A7/A6 debts
  D-CODEC-2/-3 have code); still no `ndarray-codec` crate (Plan H open).

---

## 2026-07-16 (2) — H.268 codename + graded Morton/wgpu synergy matrix

- **Codename ruling:** the "x266" placeholder (PR-X12 3DGS scene codec) is
  internally codenamed **H.268** — INTERNAL ONLY, never an ITU designation
  (H.267 itself is still prospective). Registered in the x266 lens header,
  landscape doc, capstone, 3DGS plan index.
- **NEW: `.claude/knowledge/pr-x12-h268-morton-wgpu-synergies.md`** — the
  "industry-impractical vs realistically-achievable" matrix, every claim
  adversarially verified (workflow wf_6c6fb99a-cb4, 15 agents, file:line
  receipts): 1× FEASIBLE-NOW (the scoping row), 2× NEEDS-PROBE, 7×
  OVERCLAIM-CORRECTED. Load-bearing findings: wasm SIMD128 lane IS real +
  CI-parity-verified (simd_wasm.rs; wasm-simd/parity-node); bgz17 256×256
  tables are texture-isomorphic (dense u16, R16Uint-ready) but zero GPU-LUT
  code exists; ctu.rs is an ARENA tree (no Morton in codec dir) — flat
  Morton SoA is an unimplemented refactor; D-PHASE/D-WHP are [H] with unrun
  probes (J2 kill: dither-only); a2ui-paint wgpu = untested quad demo,
  `webgl` feature unwired; ndarray deliberately "no GPU, no wgpu".
- **Probe queue established:** PROBE-GPU-LUT, PROBE-MORTON-CTU,
  PROBE-RANS-INTERLEAVE (new names), + OGAR PHASE-1/PERT-RHO/PYR-1,
  WHP-1..4, Plan E bits/Gaussian, a2ui N2 — each with pass/kill conditions.

---

## 2026-07-16 (3) — H.268 addendum: comma closure + 96-bit carving + kernel-shape rule + replayable-tile synergies

- **`pr-x12-h268-morton-wgpu-synergies.md` extended §7-§10** (old §6
  Cross-references renumbered to §11), per
  `.claude/plans/H268-comma-96bit-replayable-addendum-v1.md`.
- **§7 comma closure:** Pythagorean-comma/X-Trans anti-moiré framing;
  `CurveRuler` stride-4-over-17 as the coprime-integer surrogate.
  D-QUANTGATE rationale restated to its three real legs (libm
  non-portability, WGSL floats not IEEE-pinned, bijective closure) —
  the "floats round differently" leg is explicitly withdrawn, with
  receipts (`std::f64::consts::{GOLDEN_RATIO,EULER_GAMMA}` compile
  bit-exact on 1.94/1.95; no `std::simd::const::*`; `gemm_f64_tiled`
  five-backend bit-identical). φ-PLACES/walk-QUANTIZES/γ-CORRECTS
  division of labor stated as a rule.
- **§8 96-bit facet carving:** CAM-PQ 48b + helix `ResidueEdge` 24b +
  turbovec 24b = 96 bit = the V3 12-byte content-blind payload identity;
  `Signed360` (48b) is the out-of-row alternate carving. Three flavours
  of 256 (post-review correction + operator refinement): CAM-PQ =
  6×256² compressed to per-query 6×256 f32 ADC rows (6KB,
  `cam_pq.rs:76-84`); bgz17 = the explicit materialized 256² u16 (+ k×k
  u8 compose; 388KB benchmark = 3 S/P/O planes × 128KB); V3 facet =
  explicit 6×256² as codec-agnostic ADDRESS (6×(u8:u8) rails = 96 bit;
  classid→ClassView switches which codec's 256² family each rail
  indexes).
- **§9 kernel-shape rule:** VNNI/AMX for matmul-shaped ops, LUT/texture
  for lookup-shaped ops — turbovec NativeLut measured **11.4×** faster
  than the VPDPBUSD GEMM polyfill (n=20k/dim=512/4-bit, FINDING). ITU
  claim scoped to compute kernels only (not CABAC/conformance/ECM count).
- **§10 replayable-tile synergies:** 4×4 Morton tile as the shared
  object between H.268 (phase-side seekability — entropy-level seek
  still A8-gated; seekable grain; C6-scoped native tiling) and cognitive
  shaders (RNG-free exploration, replayable thinking on the CPU/wasm
  integer path, anti-confabulation [H, needs correlation-spectrum
  probe], cache-native 192B working set) — all nine consequences stay
  **probe-gated** (D-MTS-1..3, PHASE-1/PERT-RHO/PYR-1, WHP-1..4, L4
  doc-lock); no kill condition weakened.
- **§8 fourth-mode follow-up (post-#243):** the Hambly–Lyons anchor is
  in-workspace, not external — THIS repo `src/hpc/pillar/signature.rs`
  (Pillar-11 B7: sig transform + sig-kernel, Gram PSD, 1000-Lévy-path
  certification) + lance-graph jc Pillar 11 (`hambly_lyons.rs`, feature
  `hambly-lyons` → sigker). Only the ladder→signature MAPPING stays
  [S]; probe builds on the Pillar-11 harnesses. §8 sentence amended.
- **§10 forward synthesis (operator):** two candidate adoptions for the
  tile pyramid, probe-gated — [H] one WH family for both pyramid sides
  (OGAR sign side is already WH-of-the-address-tree; bgz-tensor's
  hadamard_rotate = same family as magnitude-side preconditioner;
  PROBE-WH-MAG = WHT₁₆+i4/i2 vs direct on real tile magnitudes) and
  [S] signature-as-trajectory-checksum (tree-like equivalence = the
  digest's null space, the formal "which detours leave no comma";
  PROBE-SIG-CHECKSUM on the Pillar-11 harnesses). Neither adds stored
  tile fields.

---

## 2026-07-16 (4) — h268-probe-wave-v1 RESULTS (adjudicated)

- **PROBE-WH-MAG → NEUTRAL; bare-tile leg CLOSED NOT-TRANSFERRING.**
  B/A 0.929/1.317/1.869 — WHT₁₆ spreads outlier energy tile-wide,
  inflating the per-cell quantization floor; the row-level win needs the
  passthrough escape + centroid residual the probe deliberately omits.
  Shipped row codec untouched. PROBE-WH-MAG-2 named, deferred.
- **PROBE-SIG-CHECKSUM → PASS** with the depth-2 bound: parallel-chord
  interior displacement is EXACTLY signature-invisible — null space
  exceeds tree-like equivalence; mitigate via depth 3 or paired digest.
- **PROBE-WALK-SPECTRUM → KILL** of §10(g)'s "decorrelated by
  construction" (walk lattice |R| 0.875 vs PRNG 0.0205 = 42.7×;
  C(13)=−15 sidelobe, coprimality ≠ decorrelation); "known period-17
  structure" half CONFIRMED (R(17m)=1−8m/N). D-QUANTGATE unaffected.
- Doc updates: §10(g) corrected, §10 forward-synthesis RESULT lines,
  §5 results sub-table. Canonical verdicts: lance-graph
  E-H268-PROBE-WAVE-1-RESULTS + plan h268-probe-wave-v1.md Results.
  Probes: bgz-tensor probe_wh_mag / jc sig_checksum / helix
  walk_spectrum (all suites green).

## 2026-07-16 (5) — sprite amortization spec'd + two standing corrections

- **PROBE-SPRITE-REPLAY spec'd** (plan `x265-sprite-replay-probe-v1.md`,
  §5 row added): moving object = HHTL-anchored splat sprite + helix
  motion code, mapped onto the x265 I/P/B grammar (I = splat set at
  anchor; P = one helix code per sprite, replacing per-block MV search;
  B = parametric interpolation along the helical path). Scope guards:
  NOT H.268, NOT x265 bit-parity — GOP-grammar replay on our primitives;
  CPU/wasm carries the bit-exactness claims, wgpu is render-grade (C9).
  Amortizations: motion search → address arithmetic in the Morton
  cascade; the minimal wgpu harness doubles as PROBE-GPU-LUT's missing
  harness. KILL: helix object-motion collapsing back into a dense MV
  field.
- **§10(i) honesty amendment**: the 3-cache-line tile claim holds only
  under the analytic Fisher-z canon (materialized 256² u16 = 128KB =
  L2-resident); analytic drops table residency to 8B and makes rail
  reads |Δi8| arithmetic (four tiles/lane per AVX-512 register).
- **PROBE-WH-MAG-2 deferral weakened**: the Skip/Merge/Delta/Escape
  mode grammar already IS the per-tile escape tier; WH-MAG-2 = WH under
  the mode grammar, not a wait for new machinery.

## 2026-07-25 — encryption: the KDF cost fields are acted on before they are authenticated

**Status:** FINDING (reproducer in `envelope::tests::every_single_bit_flip_is_refused_and_none_of_them_are_expensive`)

Found by a downstream consumer building a password-sealed record POC on
`encryption::envelope`. An exhaustive single-bit-flip sweep over a sealed
blob did not fail — it **aborted the test process**. One flipped bit in the
`m_cost_kib` header field asks Argon2id for a 4 TiB allocation; the
allocation fails, and a failed allocation in Rust aborts rather than
unwinding.

The header IS authenticated (it is the AEAD's associated data), and that
was the reasoning behind not checking it. But verifying the tag needs the
key, and deriving the key means first running Argon2id **with the
parameters the blob just supplied**. So there is a window, before anything
is proven, where an attacker-chosen cost decides how much memory this
process reserves. Tamper detection works exactly as designed and the
process still dies before reaching it. *Authenticated-but-only-later is not
the same as trusted.*

`KdfParams::validate()` now gates m/t/p **before any allocation**, in
`derive_key` and in `decode_header`. The tests assert the refusal is
**cheap** — an expensive rejection is itself the attack.

**Codex P1 on the first cut, and it was right:** the initial ceiling
(1 GiB / 64 passes) was chosen as "below Argon2's 4 TiB roof", which is a
rounding error, not a limit — 1 GiB × 64 passes pre-authentication is
equally fatal on a browser tab or a small container, and a few concurrent
requests exhaust the host. Replaced by `CostLimits`, a caller-supplied
budget: `DEFAULT` = 128 MiB / 4 / 2 (twice the memory and one pass more
than the heaviest shipped preset, so a cost bump still opens old and new
blobs), `SHIPPED_PRESETS_ONLY` = exactly 64 MiB / 3 / 1 for services that
mint every blob they open. `open_within` / `derive_key_within` take the
budget explicitly.

Measured worst case the default admits: **414 ms, 128 MiB** (release, this
box; the `#[ignore]`d `worst_admitted_cost_is_within_the_documented_budget`
prints it). The bit-flip sweep dropped 13.5 s → 2.3 s once the tighter cap
started refusing the flips it used to honour — the sweep had itself been
running multi-hundred-MiB derivations.

**Not done — an allowlist of known profiles** (Codex's alternative) would
break cost bumps in the other direction: a reader shipped before the writer
would reject the new profile. A bounded budget keeps the forward
compatibility the header format exists for.

## 2026-08-18 — mask_andnot / mask_andnot_assign added for lance-graph-java D-LGJ-W8

Added `mask_andnot(a, b, dst)` (`dst = a & !b`) and `mask_andnot_assign(a, b)`
(`a &= !b`) to `src/simd_int_ops.rs`, re-exported through
`ndarray::simd::{mask_andnot, mask_andnot_assign}`. Consumer: lance-graph-java
wave D-LGJ-W8 — the mask-native navigation correction — `lgj_mask_andnot`
behind `Mask.minus`.

Both follow the existing mask-op family shape exactly: `U64x8`-chunked
polyfill dispatch with a scalar tail, panic-on-length-mismatch, and the
tail-bit semantics documented precisely (`dst`'s tail is zero whenever `a`'s
tail is zero, because `a & !b` is a bitwise subset of `a` — the same
pre-conforming-inputs contract `mask_or` already carries). Parity vs an
independent scalar reference, algebra identities
(`(a & !b) | (a & b) == a`, `(a & !b) & b == 0`), and a dedicated
tail-conformance falsifier (a conforming `a` against a maximally dirty `b`,
including a `b`-tail-only-dirty arm) are all in place.

**EXPLICIT W1a DEVIATION RECORD.** The pair follows the existing mask-op
*family* shape (free functions re-exported through `ndarray::simd`) rather
than the W1a struct-method litmus —
`.claude/knowledge/vertical-simd-consumer-contract.md:325-326` would reject
a free fn for a *new* primitive. Rationale: `mask_andnot` /
`mask_andnot_assign` are not a new primitive shape, they are the fifth and
sixth members of the existing `mask_and` / `mask_and_assign` / `mask_or` /
`mask_or_assign` free-function family; a lone struct-method member sitting
beside four free-fn siblings would fragment exactly the polyfill surface
the `simd.rs` re-export comment (:686-693) protects, not honor it. All
other W1a criteria hold in full:
parity vs scalar reference, tail-bit semantics documented, all backends
reached via the existing polyfill dispatch (`crate::simd::U64x8`, which
resolves to AVX-512 / AVX2 / NEON-scalar / wasm-scalar / portable-scalar
per target — every arm confirmed to carry `Not`). Deviation was
council-surfaced (5+3, S2-7) and operator-visible, not smuggled.

Loose ends: none.

## 2026-09-04 — mask_ternlog / mask_ternlog_assign + contiguous fast path in eq_u32_strided_to_mask (lance-graph-java lgj_hop)

Added `mask_ternlog::<IMM>(a, b, c, dst)` and `mask_ternlog_assign::<IMM>(a, b, c)`
to `src/simd_int_ops.rs`, re-exported through `ndarray::simd::{mask_ternlog,
mask_ternlog_assign}`. Consumer: lance-graph-java `lgj_hop`, whose selection
`selected = class_f ∧ src ∧ struct_f` was two `mask_and_assign` passes through a
scratch write and is now ONE `AND3` (0x80) pass — the rank-1 spelling of a
3-input mask op replaced by the op itself.

Shape: the general member of the mask-op family (`mask_and` is `AND2` with `c`
ignored, `mask_andnot` is `AND2_ANDNOT`). `U64x8::ternlog::<IMM>` polyfill
dispatch (one VPTERNLOGQ per 512 bits on AVX-512; AVX2/NEON/wasm/scalar arms
already carried `ternlog` from W1a-#9) with a bit-serial scalar tail
(`ternlog_word`) that doubles as the parity reference. Tail-bit contract stated
precisely: conforming inputs give a conforming `dst` **iff `IMM` is even** (the
all-zero row of the truth table); every named immediate in `simd::ternlog` is
even; the subset-shaped tables inherit `mask_andnot`'s stronger "subset of `a`"
guarantee. Tests: all 256 tables × 13 lengths against an independent bit-serial
reference, `AND3 == and∘and` (non-vacuous: both narrowers must contribute),
tail-conforms-iff-even with the odd (NOR3) can-it-fire arm, length-mismatch
panics for both forms.

**W1a deviation record — same as 2026-08-18 `mask_andnot`:** free-fn family
shape, not struct-method; a lone method beside six free-fn siblings would
fragment the `simd.rs` re-export surface. All other W1a criteria in full.

**Second change, same PR:** `eq_u32_strided_to_mask` at `stride_bytes == 4` was
gathering 16 bounds-checked scalar `u32` reads into a temporary array before the
`U32x16` compare — a copy where a cast belonged. That is the exact shape of
every facet lane in lance-graph-java's facet-major columnar store (ABI minor
10), so the store was contiguous and the kernel still read it as strided. Added
a contiguous path: one fixed `[u8; 64]` window per 16 elements (safe `try_into`,
bounds proven up front), which the compiler lowers to a single vector load. The
existing `eq_u32_strided_stride4_matches_contiguous_primitive` parity test
covers it. Measured through lance-graph-java's own `columnar_hop_bench`
(65 536 rows, 32 facets, both changes together): `lgj_hop` 6 342/6 408/7 547 µs
→ **1 203/1 101/1 851 µs** (classid / hop2 / all-rows frontiers), 4.1–5.8×,
equivalence asserted before timing. 16 MB of lane read in ~1.2 ms ≈ 13 GB/s,
up from 2.1 GB/s.

Loose ends: the general strided path still gathers scalar (correct — at row
strides ≥ a cache line a hardware gather buys nothing, per the doc); a
`stride_bytes == 8` twin for `u64` lanes does not exist yet because no caller
compares `u64` lanes.
