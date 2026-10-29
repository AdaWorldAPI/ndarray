# D-DOMINO-MOV-0: bit flip, positional substitution and lazy FULL carry

READ BY: savant-architect, vector-synthesis, simd-savant, the TurboVec-U9 owner. Follows `u9-abacus-probe.md` (#389).

Scope: branch `claude/u9-domino`, stacked on #389 (base `claude/u9-abacus`). Files: `examples/u9_domino/{domino.rs, main.rs}` and `tests/u9_domino.rs`. Plain Rust, no facade, backend or storage change, no intrinsics. #388 and #389 are untouched.

All numbers were measured on one AVX-512 VNNI Xeon. A range is the min - max of 3 runs of medians.

The tables are a second full measurement, taken after the PL fix below (review, Codex, PR #391): PL's ordinary ripple stored an all-ones result without marking the column FULL, unlike L, so a later all-ones carry read and wrote the word instead of passing it as FULL -> EMPTY. The fix makes PL's L1 dense row 6.4 - 9.6 ns (was 6.3 - 6.4) and its per-FULL-column cost about 2.2 ns (was 1.85); verdicts are unchanged. The second run also differs where no code changed: F on random masks reads 8.1 - 8.2 ns on VNNI (was 6.4), and the streamed rows sit higher for every counter alike, so those differences are host and code-layout variance, not the fix.

## Verdict

- **F (Eager): ADOPT-LAB.** A bit-plane counter for dense and sparse masks. At the 64K boundary it pays off only on AVX2, where D's lowering is poor, and even there it does not beat native u64 lanes.
- **P (Subst): REFUSE.** Assignment into an EMPTY column compiles to a store anyway, and the `empty` bitmap check costs more than the XOR/AND it avoids.
- **L (LazyFull): REFUSE.** The FULL bookkeeping costs more per column than the plain ripple it replaces.
- **PL (SubstLazy): REFUSE.** It pays both bookkeeping costs and is slower than F in every measured case.

Answer to the final research question: the bit flip crosses 64K exactly, and R1 (two u64 planes) and R2 (one u128) cost the same. But a two-plane abacus does not make carry escalation cheaper than native addition in general. LLVM already compiles native lane adds to vector code, and the lazy and substitution bookkeeping costs more than the XOR/AND it avoids. The measured exception is dense or single-lane masks fed one at a time, where F's short ripple beats 64 per-lane adds.

## The question

**H1, positional substitution.** When a carry lands in a column that is known to be empty, the result is just the carry mask. The column can be assigned without being read, and no further carry is produced.

**H2, lazy binary carry.** A column that an XOR leaves all ones is FULL. Adding a mask m to a FULL column gives `!m` and carries m, and an all-ones carry through a FULL column only changes its state to EMPTY, with no data read or written.

## Invariants

- Bit flip: `b + c = (b ^ c) + 2(b & c)` per lane.
- Substitution: EMPTY column + m = m, no carry.
- Lazy FULL: FULL + m = `!m`, carry m. FULL + all-ones = EMPTY, carry all-ones.
- Finding: with zero-initialized words, the physical word under an EMPTY or FULL flag is always 0 in PL. The EMPTY flag therefore carries information only when words can hold stale data (reused buffers). The tests also run P and PL over stale words (`0xDEAD_BEEF_F00D_CAFE`).

## Implementations

All in `domino.rs`.

- `flip(old, carry) = (old ^ carry, old & carry)`: a half adder on 64 lanes, bit j = lane j.
- `Counter<N>`: N columns of u64. Column k has significance 2^k for all 64 lanes. `add_at(k, m)` adds 2^k to the lanes in m. The carry mask moves to column k+1 unshifted.
- **F `Eager`:** loops `(col[k], m) = flip(col[k], m); k += 1` while m != 0. Returns `Err(Overflow(m))` if a carry leaves column N-1.
- **P `Subst`:** F plus an `empty: u128` bitmap. Adding into an EMPTY column assigns `col[k] = m` without reading it and stops. `Subst::stale(x)` builds a zero counter over words holding garbage x.
- **L `LazyFull`:** F plus a `full: u128` bitmap. A column becomes FULL when an XOR leaves it all ones. Adding m to FULL writes `!m` without reading and carries m.
- **PL `SubstLazy`:** both bitmaps. An all-ones carry through a FULL column is a state change FULL -> EMPTY only. EMPTY + m assigns, or sets FULL if m is all ones.
- **Native:** `[u64; 64]` per-lane integers, `lane += (m >> j & 1) << k`.
- **64K boundary:** low16 as two U16x32 (wrapping add). The carry vector comes from #389's `carry_after`, becomes a 64-lane mask with `U16x32::cmpeq_mask`, and is added at column 0 (= bit 16) of a bit-sliced counter. Exact value = sum(low lanes) + 2^16 * counter total.
- **R1 `Pair128 {lo, hi}`:** two significance planes (64 independent 2-bit counters). **R2:** the same planes in one u128 (low half = 2^0 plane).

## Coverage matrix of #388/#389

Audited from source: 15 tests in `tests/u9_abacus.rs`, 5 in `tests/u9_head_to_head.rs`.

| case | status | evidence |
|---|---|---|
| crossing 2^16 | PROVEN | `carry_rules_match_u32_for_every_low`: every low 0..=65535, p at !low, !low+1, !low-1, 0, 65025, random; D 511x511, full blocks, worst case |
| flipping the bit-16 position | MISSING | D keeps high as a u16 count (`high += carry`); no bit-16 plane exists or is tested |
| carry into bit 17 when bit 16 is set | PARTIAL | implied by D's worst-case run (high counts near 65k) and C's 2^20 run; never asserted at bit level |
| repeated carries must not cancel | PARTIAL | numeric totals only (D worst case, C 2^20 all-ones) |
| eager vs deferred normalization | PARTIAL | C timed both; correctness only for deferred; D deferred (flush 65,535) only |
| consecutive FULL columns | PARTIAL | `abacus_carries_through_many_columns`: up to ~20 columns, all-lane masks only; no 32/64 chains, no single-lane chains |
| assignment into a known-empty destination | PARTIAL | `Abacus::add` stores into an empty slot without reading; numerically covered, never isolated or measured |
| carry mask forwarded without moving lanes | PARTIAL | every C test compares only lane-blind totals (`value()` = sum of popcount * 2^w); a carry rotated into another lane would pass every C test |
| two 64-bit planes in one 128-bit carrier | MISSING | |
| read-after-write / observation between adds | PARTIAL | abacus test reads `value()` between 3 rounds; D only at finish |

D's `high += carry`: numeric correctness is PROVEN. The bit-flip formulation was never measured (confirmed).

The harvest was verified: 5 quoted lines were re-read at their stated places, and the counts 15 and 5 were re-counted.

## New tests

`tests/u9_domino.rs`, 8 tests, all pass on native (AVX-512 VNNI), v4, v3, v2 (qemu) and aarch64 (qemu); clippy `-D warnings` and the shuffle guard are clean. Test counters use 96 columns (oracle u128).

- `flip_is_a_half_adder_in_every_lane`: 4 input pairs, full and zero masks, single lanes 0/17/63, 1000 random words checked per lane.
- `random_adds_match_the_oracle_in_any_order`: 3000 adds, masks zero/full/one lane/random, significances 0, 1, 15, 16, 17, 31, 63, 64, 65. A per-lane u128 oracle with checked arithmetic is compared after every add on lanes 0, 1, 31, 32, 62, 63, for F, P, L, PL, P-stale and PL-stale. The same run in reverse order. All six give identical columns in both orders.
- `carries_run_through_long_full_chains`: chains of 1, 2, 8, 16, 32, 64 FULL columns, all-lane and single-lane (lane 37) masks, then two successive carries into column 0. Exact column contents are asserted (2^n + 1 in the mask lanes, 0 elsewhere, no other lane touched).
- `the_64k_boundary_flips_bit_16_then_carries_into_bit_17`: low16 in {0, 1, 510, 511, 512, 65000, 65534, 65535} x p in {0, 1, 511, 65024, 65025} x bit 16 preset or not. Value `low + 2^16*high` is exact in lanes 0, 31, 32, 63. Bit 16 flips exactly when crossing XOR preset. Bit 17 is set exactly when crossing AND preset.
- `product_sums_match_native_and_d`: 1, 300 random and 300, 70,000 dense (65,025) blocks. Native, D, F, P, L, PL, each with deferred and per-block normalization, all equal the u128 sum.
- `overflow_past_the_last_column_is_refused`: 4-column counters of every variant. A carry past the top returns `Err(Overflow(lanes))`. An add at k=N with m=0 is Ok.
- `xor_made_full_column_is_full_and_carries_by_state` (added after review): complementary masks at one significance leave a column all ones by XOR; L and PL both mark it FULL, and the next all-ones carry turns it EMPTY and lands in column 1 as FULL, lanes exact. `check_all` also asserts after every add that L, PL and PL-stale hold exactly the logically all-ones columns as FULL.
- `pair_and_packed_agree_and_roundtrip`: 10,000 random adds, R1 and R2 identical, roundtrip identity, per-lane 2-bit counter check (64 independent counters, not one integer).

## Disable runs

Each disable must turn tests red.

| id | disable | result |
|---|---|---|
| Y1 | F carry not propagated | RED (5 tests) |
| Y2 | PL lost the FULL->EMPTY transition | GREEN at first, RED after adding stale-word runs (2 tests) |
| Y3 | P assigns into a non-empty column | RED (4 tests) |
| Y4 | L treats FULL as unchanged after a partial carry | RED (3 tests) |
| Y5 | carry mask rotated one lane (R1) | RED (1 test) |
| Y6 | P reads the stale word instead of assigning | RED (2 tests) |
| Y7 | PL ordinary ripple does not mark an all-ones result FULL (the pre-review code) | RED (2 tests) |

**The Y2 story.** The first Y2 run was green. With zero-initialized words, the physical word under an EMPTY or FULL flag is always 0 in PL, so a lost FULL->EMPTY transition left the data identical to the correct result and nothing could tell the difference. Only after P and PL were also run over stale words (`0xDEAD_BEEF_F00D_CAFE`) did the same disable turn red, in 2 tests.

This shows two things. A green disable run on a lazy structure may mean the tests cannot see the state, not that the guard is unneeded. And the EMPTY flag is the only thing that makes stale data harmless, so a test that never feeds stale data cannot cover it.

## Measurements

How to run:

```sh
cargo run --release --example u9_domino --features std
cargo test --release --test u9_domino
```

### A. u16 products summed per lane across the 64K boundary (ns per 64 products)

L1 = 64 blocks. Streamed = 2^20 blocks (128 MiB). Dense = products 65,024-65,025 (a carry on almost every add). Sparse = products 0..=255 (a carry about once per 260 blocks per lane).

| L1 dense | AVX-512 VNNI | AVX-512 (v4) | AVX2 (v3) |
|---|---|---|---|
| native u64 lanes | 4.81 - 4.83 | 4.83 - 4.84 | 4.83 - 4.84 |
| D (#389) | 2.28 - 2.56 | 2.30 - 2.66 | 9.39 - 9.42 |
| F | 2.70 - 2.88 | 3.05 - 3.14 | 5.66 - 5.67 |
| P | 5.44 - 5.47 | 5.78 - 5.95 | 7.61 - 7.81 |
| L | 5.89 - 6.22 | 6.55 - 6.61 | 8.50 - 8.56 |
| PL | 6.36 - 9.58 | 6.67 - 6.91 | 8.80 - 8.98 |
| P normalized every block | 74.7 - 74.8 | 75.0 | 68.0 - 68.8 |
| PL normalized every block | 107 - 110 | 111 - 112 | 115 - 116 |

| L1 sparse | VNNI | v4 | v3 |
|---|---|---|---|
| native | 4.83 - 5.58 | 4.81 - 4.86 | 4.83 - 4.84 |
| D | 2.31 - 3.19 | 2.22 - 2.67 | 9.42 |
| F | 2.59 - 2.94 | 2.94 - 3.03 | 5.61 - 5.77 |
| P | 3.64 - 4.91 | 3.95 - 3.98 | 6.53 - 6.70 |
| L | 3.62 - 3.83 | 3.95 - 3.97 | 6.61 - 6.75 |
| PL | 3.92 - 4.00 | 4.20 - 4.36 | 7.05 - 7.06 |

| streamed dense | VNNI | v4 | v3 |
|---|---|---|---|
| native | 15.5 - 16.7 | 15.7 - 16.1 | 14.2 - 16.6 |
| D | 11.3 - 12.4 | 11.3 - 12.2 | 31.6 - 32.4 |
| F | 12.5 - 13.7 | 13.6 - 14.4 | 15.8 - 20.4 |
| P | 13.8 - 15.9 | 14.6 - 14.8 | 14.6 - 25.8 |
| L | 13.9 - 16.5 | 13.4 - 14.7 | 13.6 - 27.3 |
| PL | 13.3 - 15.2 | 13.8 - 15.1 | 12.2 - 26.2 |

| streamed sparse | VNNI | v4 | v3 |
|---|---|---|---|
| native | 15.5 - 16.5 | 15.5 - 16.5 | 16.0 - 16.7 |
| D | 10.7 - 12.1 | 10.2 - 11.4 | 31.4 - 32.5 |
| F | 10.3 - 13.2 | 10.2 - 12.9 | 17.2 - 17.8 |
| P | 11.2 - 12.9 | 9.90 - 13.0 | 16.8 - 19.0 |
| L | 10.9 - 12.1 | 10.2 - 12.7 | 16.0 - 19.6 |
| PL | 11.1 - 12.3 | 10.9 - 12.5 | 16.2 - 19.3 |

The streamed rows are noisy. The ranges overlap between F, P, L and PL.

### B. Counters alone, ns per add (4,096 adds, 64 lanes)

VNNI figures. v4 and v3 are similar unless noted.

| workload | native | F | P | L | PL |
|---|---|---|---|---|---|
| dense all-ones at 2^0 (binary counting) | 6.99 - 7.04 | 1.93 - 1.96 (v4 1.94, v3 1.95 - 1.98) | 2.92 - 2.93 | 3.42 - 3.49 | 3.22 - 3.29 |
| random masks at 2^0 | 6.99 | 8.12 - 8.21 (v4 8.15 - 10.1, v3 8.05 - 8.21) | 10.5 - 10.8 | 12.4 | 15.5 - 15.6 |
| sparse one lane at 2^0 | 6.99 | 5.77 - 6.41 (v4 5.79 - 6.21, v3 6.10 - 6.29) | 6.17 - 6.33 | 6.14 - 7.05 | 7.14 - 7.65 |
| weighted: random masks entering at 2^0, 2^16, 2^17, 2^32 | 6.99 | 7.57 - 7.71 (v4 7.73 - 8.83, v3 7.64 - 8.21) | 10.2 - 10.8 | 11.8 - 12.8 | 14.5 - 14.8 |

### B'. One add whose carry runs through n FULL columns

ns per rep = clone + add. Each cell is "clone-only / clone+add", so subtract the clone-only figure. VNNI; v4 and v3 are within a few percent.

| n | F | P | L | PL |
|---|---|---|---|---|
| 1 | 5.27 - 5.46 / 5.88 | 4.93 - 5.03 / 12.5 | 5.65 / 11.5 - 11.8 | 5.66 / 12.1 - 12.3 |
| 8 | 5.14 - 5.47 / 11.8 | 4.91 - 5.04 / 18.0 - 18.4 | 5.51 - 5.65 / 24.0 - 24.4 | 5.65 - 5.66 / 24.6 - 25.2 |
| 32 | 5.27 - 5.47 / 30.1 - 30.4 | 5.03 - 5.04 / 61.9 - 70.6 | 5.62 - 7.39 / 75.2 - 94.3 | 5.52 - 5.66 / 79.2 - 82.5 |
| 64 | 5.46 - 7.47 / 38.6 - 60.4 | 5.03 - 6.76 / 108 - 143 | 5.64 - 7.53 / 127 - 168 | 5.66 - 7.18 / 146 - 150 |

Per FULL column (64-column row minus clone, medians, divided by 64): F about 0.52 ns, P about 1.6 ns, L about 1.9 ns, PL about 2.2 ns.

"Constant state" rows (counter built from literals inside the timed loop, then add + total): LLVM did not fold the counter to a constant. F measured 101 ns and PL 347 - 349 ns at n=64 on VNNI. They include building the preset, so they are not comparable to the opaque rows above.

### C. R1 vs R2 (ns per add, 4,096 adds)

| | VNNI | v4 | v3 |
|---|---|---|---|
| R1 two u64 | 0.412 - 0.415 | 0.415 - 0.424 | 0.411 - 0.414 |
| R2 one u128 | 0.417 - 0.453 | 0.442 - 0.453 | 0.443 - 0.517 |

## Assembly

objdump of `#[inline(never)]` copies. Counts came from a Haiku census with the stated grep; 2 rows were re-run by hand and matched.

| function | instructions | notes |
|---|---|---|
| flip | 5 | mov, mov, xor, and, ret; scalar GPR, no ternlog |
| R1 `pair_add` and R2 `packed_add` | 10 each, identical | native and v3: mov, xor, and, mov, xor/and, and/xor, 3 stores, ret. LLVM sees through the u128 packing: no shifts, no extracts, no extra moves |
| `eager_add` (F) | 47 - 59 | 10 - 12 branches, loop over the column array in memory; logic 14 - 18; no ternlog, no shuffles |
| `subst_add` (P) | 99 - 123 | 13 - 16 branches, 12 - 16 shifts (u128 bitmap tests), 6 - 9 push/pop |
| `lazy_add` (L) | 51 - 88 | 11 - 15 branches, 6 - 8 compares |
| `pl_add` (PL) | 86 | 15 branches, 8 compares, 8 push/pop |
| `native_add` (64 lanes += bit << k) | 68 - 69 | vectorized: 16 adds, 16 shifts, 16 logic, 1 branch, 2 vpbroadcastq |
| 64K block, AVX-512 (native and v4): D | 20 | 4 vpternlogq (the carry rule), no shuffle, 1 branch |
| 64K block, AVX-512 (native and v4): F | 68 - 79 | 4 vpternlogq, 10 - 12 branches (the ripple loop), 27 - 33 moves incl. mask transfers |
| 64K block, AVX2 (v3): D | 107 | 13 shuffle-class instructions (3 vextracti128, 1 vinserti128, 5 vpbroadcastq, 1 vperm2i128, 1 vpermq, 1 vpunpcklqdq) from the array-backed U16x32 |
| 64K block, AVX2 (v3): F | 142 | 18 shuffle-class, 10 branches |

A first R1/R2 anchor was constant-propagated because it was called with a known-zero high half. It was fixed with an opaque input.

## Reading it

- **F vs D at the 64K boundary, AVX-512.** D is faster in L1 (2.2 - 2.6 vs 2.7 - 3.3 ns) because its whole high column is 4 vpternlogq plus adds, with no branch. F's ripple is a scalar loop with branches. Streamed, the two overlap (the run is memory-bound).
- **F vs D, AVX2.** F is 1.6 - 1.7x faster than D in L1 and about 2x streamed, because D's AVX2 U16x32 lowering moves halves around (13 shuffle-class ops). The native u64-per-lane sum (4.8 ns) still beats both F and D on AVX2 in L1.
- **The flip itself is cheap.** The cost is the data-dependent ripple loop and getting the carry out of the vector into a GPR mask.
- **P, L, PL are slower than F in every measured case.** Every metadata check is a branch plus a u128 shift/test. The gap grows with chain length: about 1.6 - 2.2 ns per FULL column against 0.5 ns for F. The deferred FULL->EMPTY transition skips a data read and write, but the flag read and write costs more.
- **Normalizing P or PL every block costs 70 - 120 ns per block** (normalize rewrites all 72 columns). Only the deferred form is usable, and even deferred it loses to F.
- **Positional substitution does not disappear in the compiler.** Assignment into an EMPTY column compiles to a store: the column array lives in memory indexed by a runtime k, so no register renaming happens. LLVM does not eliminate it.
- **R1 and R2 cost the same** (section C), and LLVM compiles them to identical code.
- **Limits.** One machine, one compiler build. The streamed rows are noisy. The "native beats F" statement on AVX2 is for the L1 rows (dense and sparse); streamed, the ranges overlap.

## Break-even

- No break-even for P, L or PL against F in any workload measured.
- F breaks even against native integer addition on dense and single-lane masks (3.6x faster on dense counting). It does not on random masks: about 16% slower on every tier (8.1 - 8.2 vs 7.0 ns).

## Not done

- NEON and wasm timings.
- Unrolled or vectorized ripple.
- Hand scheduling (out of scope by design).
