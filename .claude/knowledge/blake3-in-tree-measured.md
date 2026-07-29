# In-tree BLAKE3 — correct, and what it costs

> **Status: MEASURED, 2026-07-29.** Correctness against the official vectors;
> throughput against the external crate. Both numbers below are reproducible
> with the instruments in this directory.

## READ BY:
- Anyone about to drop the external `blake3` dependency
- Anyone continuing rung 3 of `the-simd-ladder.md`

## P0 TRIGGER
About to swap `blake3::` call sites onto `crate::hpc::blake3`? **The swap is
correct but costs 1.3× on typical inputs and ~5× at 64 KB. Read the table.**

---

## Why it exists

Root `ndarray` depends on `blake3`, so `blake3 → ndarray::simd` is a cargo
cycle — the only rung of the ladder that has one (`the-simd-ladder.md`).
Cutting it means ndarray owning BLAKE3 rather than consuming the crate.

Scoping finding that made this small: **ndarray's usage is entirely
single-input.** 14 call sites across 8 files use only `hash`,
`Hasher::{new, new_keyed, update, finalize, finalize_xof().fill()}`,
`Hash::as_bytes`, and `Hash` as a signature type. **No `hash_many`.** So the
serial core suffices, and it needs no SIMD at all.

## Correctness — proven

`src/hpc/blake3.rs`, 771 lines, transcribed from upstream's own
`reference_impl/reference_impl.rs` (the spec-referenced serial
implementation, BLAKE3 spec §5.1). No `unsafe`, no `core::arch`, no new
dependencies.

Against the official `test_vectors.json`, vendored to
`src/hpc/blake3_test_vectors.json`:

- **35/35** cases, unkeyed `hash` **and** keyed `keyed_hash`,
- each checked at **both** 32-byte length and the full extended length via
  `finalize_xof().fill()` — 140 assertions,
- input lengths 0 … 102 400.

Plus: streaming (one `update` vs many 37-byte `update`s over 102 400 bytes),
empty input, and incremental `fill` (7 bytes at a time vs one shot).

`derive_key` was included too — it is another `Hasher` invocation with
different flags, so it came free.

## Throughput — the cost, measured

`sh .claude/knowledge/blake3-ab-bench/run.sh`, release build, two runs:

| input | in-tree | `blake3` crate | ratio |
|---|---|---|---|
| 16 B (a word) | 134 ns | 100 ns | **1.34–1.39×** |
| 256 B (text) | 437 ns | 350 ns | **1.25–1.29×** |
| 2 KB (`VSA_BYTES`) | 3.4 µs | 2.6 µs | **1.30×** |
| 64 KB (bulk) | 109 µs | 23 µs | **4.7–4.9×** |

### The `array_chunks` fast path — operator's lead, measured

Hypothesis (operator, citing the blasgraph JIT-gap precedent): the gap might
be closed by proper use of the existing slice primitives rather than by new
SIMD. The staging path copies every byte **twice** — input → `self.block` →
`block_words` — and for a full block the first copy is pure overhead.

Implemented as a `crate::simd_ops::array_chunks::<u8, 64>` fast path in
`ChunkState::update`, guarded `input.len() > BLOCK_LEN` so a chunk's final
block is never compressed early (it carries `CHUNK_END`). **Measured, three
runs:**

| input | before | after | change |
|---|---|---|---|
| 16 B | 137 ns | 134 ns | — (never reaches the fast path) |
| 256 B | 445 ns | 437 ns | — |
| **2 KB** | **4322 ns** | **3421 ns** | **−21 %** |
| 64 KB | 114.8 µs | 109.0 µs | −5 % |

**Verdict: real, and bounded.** The double copy was costing ~21 % at the mid
sizes — not nothing, and free to remove. But it does **not** replace the two
structural gaps: inputs ≤ 1 block never reach the fast path at all, and at
64 KB the copy is noise beside the absent `hash_many`. The ratio at 2 KB
moved 1.34–1.60× → a stable 1.30×; the small-input 1.3× and the bulk 4.8×
both stand.

So the answer to "does it just need proper `array_chunks` use?" is **partly,
and the part it fixes is now fixed.** Rungs 3b (`hash_many`) and 3c (SIMD
single-compress) remain the load-bearing ones.

Correctness is gated, not assumed: the official vectors cover every boundary
the fast path turns on — 63/64/65 (the `> BLOCK_LEN` guard itself),
127/128/129, and 1023/1024/1025 (the chunk boundary).

**The two gaps have different causes, and only one is about `hash_many`.**

- **The 64 KB gap is `hash_many`.** Above one chunk (1024 B) the crate
  switches to its degree-8/16 parallel path with the transpose. We have none.
  This is exactly rung 3b, and the `U32x16` shuffle surface merged in #267 is
  what it would be built on.
- **The 1.3× small-input gap is NOT.** At 16 B there is a single compression
  and no parallelism to be had — the crate is still faster because it
  SIMD-accelerates *the single compress itself* (its sse41 backend). Closing
  that needs a `U32x4`-shaped compress, which is a rung the ladder plan does
  not currently have. Call it 3c.

So the honest shape is:

```
3a  in-tree core, correct          DONE, costs 1.3x typical / 5x bulk
3b  hash_many on U32x16            closes the bulk gap
3c  SIMD single-compress (U32x4)   closes the small-input gap
```

## What this means for the swap

**The cycle-cut is available now and is correct.** It removes a cargo cycle,
the C build question, and 2,910 lines of second-surface `core::arch`.

**It is not free**, and the previous framing ("removes things, needs no
benchmark to justify") was true about what it *removes* and silent about what
it *costs*. With the numbers in hand that framing is incomplete: this is a
trade, and which side wins depends on how hot ndarray's hashing actually is.

Where the call sites sit on the curve: `crystal_encoder` hashes a word
(16 B band), `vsa` XOF-expands to 2 KB, `merkle_tree`/`seal`/`spo_bundle`
hash small nodes, `deepnsm`/`compression_curves` small. So ndarray's real
exposure is the **1.3–1.6× band**, not the 5× one — but 1.3× on a hot encoder
path is a real cost, not a rounding error.

**Not swapped here.** The module lands and is tested; the call sites still
use the external crate. Flipping them is a decision with a measured price
tag, and it is the operator's.

## One deliberate deviation from upstream, and one restored

- **Deviated:** transcribed from `reference_impl.rs` rather than
  `portable.rs` + `lib.rs`. Upstream ships the reference implementation as
  the readable, algorithmically-identical serial version, which is what
  "take the serial branch everywhere" reduces to. Correctness is proven by
  the vectors. Part of the 1.3× is likely this choice rather than the SIMD
  gap — `portable.rs` avoids a per-block staging copy — and that has **not**
  been separated out.
- **Restored:** `Hash::eq` is **constant-time**. Upstream uses the
  `constant_time_eq` crate; the transcription initially used a plain `==`,
  which leaks match-prefix length through timing when a BLAKE3 output is used
  as a MAC. Rewritten as an XOR-fold with a `black_box` on the accumulator,
  no dependency added. No call site in this crate compares two `Hash` values
  today — `seal.rs` compares the truncated `MerkleRoot` — so this is a guard
  for future consumers, not a live-leak fix.

## Not claimed

- Not that the in-tree version should replace the crate. That is the open
  decision this document exists to inform.
- Not that 1.3× is attributable to any single cause. The reference-impl
  choice and the absent single-compress SIMD are confounded and were not
  separated.
- Not that the bench is rigorous. It is a warm-loop wall-clock A/B, adequate
  for a 1.3× vs 5× distinction and not for anything finer.
