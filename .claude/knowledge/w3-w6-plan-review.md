# W3-W6 Plan Review

Reviewer: plan-review savant
Branch: `claude/w3-w6-soa-aos-helpers`
Design under review: `/home/user/ndarray/.claude/knowledge/w3-w6-soa-aos-design.md`
Date: 2026-05-18

## Verdict

**READY-WITH-DOC-FIXES** — design is sound, layering rule respected, no irreducible ambiguity. One P0 doc fix (module placement), six P1 clarifications, and several P2 polish notes should land before workers spawn. Implementation can proceed in parallel once the doc patches commit.

## Findings

### Blockers (P0 — must fix before workers spawn)

- **P0-1 — `aos_to_soa` / `soa_to_aos` are wired to the wrong module.** Design doc §"W5 + W6" (lines 264–330) places them in `src/simd_ops.rs`. That module is *exclusively* SIMD-dispatch glue (every existing fn uses `F32x16` / `F64x8`; see `simd_ops.rs:1–10` doc header: "Slice-level elementwise ops built on the polyfill SIMD types"). Putting *pure-scalar, no-SIMD* helpers there:
  1. Contradicts the module's documented charter (`simd_ops.rs:1`).
  2. Violates the W1a consumer contract (`vertical-simd-consumer-contract.md:31`): "ndarray's SIMD surface is shaped to fit exactly what the Ada stack vertically needs — not as a generic library". Free-function `aos_to_soa` in `simd_ops.rs` is exactly the free-function shape the litmus tests at lines 315–317 say to reject.
  3. Forces the future SIMD swap to either (a) leave the scalar fn at the same module path while adding `_avx512` / `_neon` cousins in different files — semantically incoherent — or (b) rename, breaking callers.

  **Recommended fix in the design doc:** move both fns to `src/hpc/soa.rs` (same module as `SoaVec`). Doc example becomes:
  ```rust
  use ndarray::hpc::soa::{SoaVec, aos_to_soa, soa_to_aos};
  ```
  This co-locates types and their conversions and keeps `simd_ops.rs` pure-SIMD. When the future SIMD swap happens, the dispatcher at `hpc::soa::aos_to_soa` can grow per-arch arms internally (delegating to per-arch impls under `simd_*.rs` only for the SIMD bits, not for the user-visible API).

  Cross-ref: E2 below (answer: `hpc::soa`).

### Important (P1 — fix in doc or first commit)

- **P1-1 — `SoaVec::field(i)` runtime panic is avoidable for known indices.** Design doc lines 132–136. `N` is const-generic and `i: usize` is runtime — most call sites will use literal indices. Recommend adding a const-generic accessor alongside the runtime one:
  ```rust
  pub fn field_n<const I: usize>(&self) -> &[T] {
      const { assert!(I < N) };  // compile-time bounds check (Rust 1.79+)
      &self.fields[I]
  }
  ```
  Keep the runtime `field(i)` for dynamic-index callers. Doc-comment must state which to use when. Without this, every hot-path access pays an unnecessary bounds check that the optimizer often can't elide.

- **P1-2 — `T: Copy` bound is silent in `aos_to_soa` / `soa_to_aos`.** Design doc lines 290–298: `extract: F where F: Fn(&T) -> [f32; N]` — the closure returns `[f32; N]` by value, so the *return type* is `Copy`. `T` itself does NOT need `Copy` because `extract(item)` borrows via `&T`. But the doc-comment must say so explicitly. The current example uses `struct Item { a: f32, b: f32, c: f32 }` which is implicitly `Copy`; a consumer with `String` fields would be uncertain. Add a doc note: "`T` need not be `Copy`; only the extracted `[f32; N]` row is materialized."

- **P1-3 — `soa_struct!` macro has zero reserved-word / field-name-collision handling.** Design doc lines 202–253. The macro generates a `push(&mut self, $($field: $ty),*)` method — if a struct has a field named `len`, `clear`, `new`, `with_capacity`, `is_empty`, or `default`, the generated impl will fail to compile due to method-name collision (the macro generates `pub fn len(&self)` AND the field `pub len: Vec<...>`; user accessing `self.len` is ambiguous between the field and the method). The compile error will be cryptic. Doc must add a §"Reserved field names" section listing the six names that conflict and stating "the macro deliberately does not work around this; choose different field names."

- **P1-4 — D3 (pub-field invariant) is left undecided.** Design doc lines 211–212: `$vis struct $name { $($field_vis $field: ::std::vec::Vec<$ty>),* }`. The macro respects user-provided `pub` on fields, which means callers CAN do `batch.means_x.truncate(5)` and break the field-length invariant silently (release-mode debug_assert disabled). The doc never specifies whether the invariant is owned by the type or the caller. Worker will have to guess. See D3 recommendation below.

- **P1-5 — `aos_to_soa::<_, 3, _>` turbofish is awkward; inference is brittle.** Design doc line 285 example: `aos_to_soa::<_, 3, _>(&aos, |it| [it.a, it.b, it.c])`. Two `_` placeholders means callers MUST remember the positional order (T, N, F). The const-generic `N` can usually be inferred from the closure return type `[f32; 3]`, but Rust's inference for const generics from array literals is brittle. Doc must include an "if inference fails" note showing the alternative: `aos_to_soa(&aos, |it| -> [f32; 3] { [it.a, it.b, it.c] })`. Tag as "verified on Rust 1.94".

- **P1-6 — `bulk_apply` panic on `chunk_size == 0` is documented; `usize::MAX` is not.** Design doc lines 374–384. `slice::chunks_mut(usize::MAX)` on a non-empty slice returns a single-chunk iterator (stdlib behavior). Doc should add one line: "`chunk_size == usize::MAX` yields the entire slice as a single chunk." Trivial fix, prevents future confusion.

- **P1-7 — Module registration spec ambiguity in `src/lib.rs`.** Design doc lines 419–429: "the macro is already `#[macro_export]` which puts it at the crate root, so `ndarray::soa_struct!` works. No manual re-export needed." Technically correct, but workers must NOT also write `pub use crate::hpc::soa::soa_struct;` (would conflict). Add an explicit "do not re-export the macro; `#[macro_export]` handles it" line in the spec.

### Nice to have (P2 — workers may apply, codex can flag later)

- **P2-1 — `bulk_scan` is a misnomer.** Design doc lines 386–409. In Rust idiom (`Iterator::scan`), FP, APL/GPU, "scan" = prefix-sum / fold-with-state. The fn is a read-only `chunks` walker. Better names: `bulk_each`, `bulk_for_each`, `bulk_inspect`, or `bulk_apply_ref` to pair with the mut version. Doc should at least call out the divergence from convention if `bulk_scan` is kept.

- **P2-2 — `SoaVec` is missing `iter_rows()` and `iter_rows_mut()`.** Design doc lines 98–148. Chunked iterator is there, row iterator (yielding `[&T; N]` per row) is absent. Consumers will want it for `for row in soa.iter_rows() { ... }` patterns. Either add it to v0 or document explicitly: "row iteration is not provided; use `soa.chunks(1)` if needed".

- **P2-3 — `core::array::from_fn(|k| fields[k][i])` in `soa_to_aos` (line 325) is O(N) per row.** For N=2/3/4 it's fine. Add one sentence: "Complexity: O(N·len) where N is field count and len is row count." Settles the "should this be SIMD'd later?" question by making the cost visible.

- **P2-4 — `SoaVec` doesn't have a `Clone` derive (or any derives).** Design doc line 94. SoA containers commonly want `Clone` and sometimes `Debug`. The macro at line 211 captures `$(#[$meta:meta])*` for the struct attrs — verify that user `#[derive(Clone)]` on the macro invocation passes through correctly; if not, add a derive-passthrough fragment.

- **P2-5 — Test §"hpc::soa::push debug-assert fires" is hard to write for `SoaVec`.** Design doc line 262. For the *macro*-generated struct, fields are `pub`, so the test is straightforward. For `SoaVec`, fields are private, so the corrupt-state test is impossible without `unsafe`. Either drop the `debug_assert!` in `SoaVec::len()` (it can never fire from safe code), or keep it as defense-in-depth and skip the SoaVec test (only test the macro-generated case). Worker spec should pick one.

- **P2-6 — `SoaVec` lacks `pop`, `truncate`, `swap_remove`.** Symmetry with `Vec`. Not required for v0; add a future-work note in the module-level doc.

## Specific recommendation for each open question

- **D3 (macro: pub fields stay pub OR go private?):** **Keep fields `pub` (or whatever vis the user specifies), document the invariant as caller-owned for direct mutation.** Rationale: (1) the SoA layout's entire ergonomic win is direct `&[T]` access for SIMD-style loops — hiding fields behind getters forces `batch.means_x()` everywhere, ugly; (2) the existing `GaussianBatch` in `splat3d/gaussian.rs:93–110` already uses `pub` fields, so changing the pattern would create inconsistency; (3) the `debug_assert!` in `len()` catches mistakes during dev. **Add doc line:** "Fields are `pub` (or use the visibility you specify). Mutation through `push` / `clear` preserves the field-length invariant. If you mutate fields directly (e.g., `batch.means_x.truncate(5)`), you OWN the invariant — `len()` will debug-assert but release builds will return arbitrary results until you restore equality."

- **E2 (`simd_ops` vs `hpc::soa` for the helpers?):** **`hpc::soa`.** See P0-1. The helpers have no SIMD content; they live with `SoaVec`; `simd_ops.rs` module charter (line 1) is SIMD-only. This is the load-bearing change to the design doc.

- **F1 (future SIMD wave: rename `_scalar` or grow internal arms?):** **Grow internal arms.** Keep `pub fn aos_to_soa(...)` at `hpc::soa::aos_to_soa` as the stable user-facing entry. When SIMD lands, the dispatcher inside the fn calls `simd_dispatch::table().aos_to_soa_f32` (or similar function-pointer) which selects between `aos_to_soa_scalar`, `aos_to_soa_avx512`, `aos_to_soa_neon` at LazyLock startup. Public API is forever stable. Reason: per `vertical-simd-consumer-contract.md:31`, the consumer pattern is "closure-parameterized batch primitives that absorb the consumer's domain semantics" — the closure `Fn(&T) -> [f32; N]` IS the consumer's domain semantics, and the dispatch happens *inside* the entry.

  **F2 caveat acknowledged:** a future SIMD impl will want `(*const T, stride)` (or `field_offsets: [usize; N]`) not `Fn(&T) -> [f32; N]`, because the closure is per-row scalar and prevents true vectorization. The migration plan: when SIMD lands, the closure-API stays as fallback for "extract logic too complex to vectorize", and a NEW entry `aos_to_soa_strided(packed: &[T], field_offsets: [usize; N]) -> SoaVec<f32, N>` ships alongside for the "dense POD struct" case where stride/offset can drive a real gather. Two entries, both stable. No rename.

## Layering rule compliance summary

- **A1 (per-arch leakage):** Clean. Zero `#[target_feature]`, zero `is_x86_feature_detected!`, zero `cfg(target_arch)` in any design surface. Doc explicitly forbids them at lines 42–48.
- **A2 (const-generic forcing specialization):** No. `const N: usize` is a *value* generic, not a SIMD-feature generic. Future SIMD impls can dispatch on `N` (e.g., specialize N=3 for LD3 NEON, N=4 for AVX-512 gather-stride-4) inside the dispatcher without changing the public signature.
- **A3 (future SIMD swap):** Holds, conditional on E2 (`hpc::soa` placement) and F2 (strided alt-entry as the migration path).

## Confidence to proceed

**HIGH.** Design is implementable with the one P0 module-relocation patch. No irreducible ambiguity. Workers can be spawned after the doc absorbs P0-1, P1-3, P1-4 (D3), P1-7, and the explicit D3 / E2 / F1 picks above. P2 items are codex-audit territory.

## Action items before spawning workers

1. **Doc patch:** apply P0-1 — change all "src/simd_ops.rs" references in §"W5 + W6" to "src/hpc/soa.rs", update import paths in examples.
2. **Doc patch:** apply D3 and E2 decisions explicitly in the spec text.
3. **Doc patch:** add P1-1 (`field_n<const I>`), P1-3 (reserved-name section), P1-7 (no manual re-export).
4. **Optional doc patch:** add P1-2, P1-5, P1-6 clarifying notes.
5. Spawn W3+W5+W6 as **one** worker (now all in `hpc::soa.rs`), W4 worker separately.

## Sequencing concern flagged by E2

If the E2 recommendation is taken (helpers live in `hpc::soa`), W3 and W5+W6 both touch `src/hpc/soa.rs`. Options:
- **(a) Sequential:** W3 commits first (just `SoaVec` + macro), then W5+W6 commits the helpers on top.
- **(b) Single combined worker:** W3+W5+W6 spawn as one worker producing one file in one commit. W4 (`hpc/bulk.rs`) stays parallel.

Option (b) is cleaner — one commit per logical unit (SoA infrastructure), one PR. Recommend (b).
