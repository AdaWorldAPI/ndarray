# Sonnet clippy lint sweep — rust-toolchain 1.95.0 → 1.97.1

Branch: `claude/x265-x266-plans-review-h9osnl` (shared checkout, no worktree, no commits made).

## Pass 1 — `cargo clippy -p ndarray -- -D warnings` (lib)

3 errors, all `clippy::question_mark` on the identical `match self.<field> { None => return None, Some(ref ix) => ix.clone() }` pattern inside `Iterator::next` impls. Fixed by collapsing to `self.<field>.clone()?`.

- `src/indexes.rs:60` — `IndicesIter::next` — `clippy::question_mark` — replaced 4-line match with `let index = self.index.clone()?;`
- `src/iterators/mod.rs:491` — `IndexedIter::next` — `clippy::question_mark` — same fix, `self.0.inner.index.clone()?`
- `src/iterators/mod.rs:667` — `IndexedIterMut::next` — `clippy::question_mark` — same fix, `self.0.inner.index.clone()?`

Re-run: clean (`Finished` in 13.88s).

## Pass 2 — `cargo clippy -p ndarray --tests -- -D warnings`

3 errors, all in `#[cfg(test)]` modules:

- `src/property_mask.rs:426` — `clippy::unusual_byte_groupings` — `0b11111_1000` → `0b1_1111_1000` (regrouped into nibbles per clippy's own suggestion; value unchanged, still asserting the same bit pattern).
- `src/bitwise.rs:637` — `clippy::identity_op` — `assert_eq!(super::popcount_batch_u64(&words), 64 + 0 + 32)` had a no-op `+ 0`. Replaced with the literal `96` and kept the per-word breakdown as a comment (`// per-word contributions: 64 + 0 + 32`) so the documentational intent survives.
- `src/palette_codec.rs:810` — `clippy::needless_range_loop` in `test_bedrock_pack_section` — `for i in 0..4096 { states[i] = (i % 4) as u16; }` → `for (i, state) in states.iter_mut().enumerate() { *state = (i % 4) as u16; }`. Purely mechanical (index only used for indexing + the `% 4` value); no semantic change.

Re-run: clean (`Finished` in 36.14s).

## Pass 3 — `cargo fmt -p ndarray`

Applied. Reformatted only the touched hunks (rustfmt collapsed the shortened `next()` bodies and normalized the byte-grouping/comment line spacing). No behavioural diff beyond formatting.

## Nothing skipped

No `src/simd_*.rs` files were touched — none of the flagged lints landed there. No `unsafe` blocks were touched (none of the fixes were near unsafe code). No public API signatures changed.

## Final status

- `cargo clippy -p ndarray -- -D warnings` — **clean**
- `cargo clippy -p ndarray --tests -- -D warnings` — **clean**
- `cargo fmt -p ndarray` — **applied**, `git diff --stat`: 5 files changed, 8 insertions(+), 16 deletions(-)
- No `cargo test` run (disk guard tripped — see below).

## Disk guard note

Free space on `/home/user` dropped from 5.7 GB → 5.3 GB → **4.2 GB** over the course of this run (crossing the 5 GB floor right after the `cargo fmt` invocation, which had itself been preceded by a compliant 5.3 GB check). Per the hard rule I stopped immediately after `cargo fmt` and did **not** run any further cargo invocations (no targeted `cargo test`, no additional clippy passes). All fixes above are comment/mechanical-only changes (`.clone()?` collapse, digit regrouping, literal substitution, `iter_mut().enumerate()`), so behavioural risk is low, but the orchestrator should re-run `cargo test -p ndarray` centrally once disk space is confirmed healthy.
