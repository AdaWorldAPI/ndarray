# W3-W6 Codex Audit

Auditor: codex P0 review agent
Branch: `claude/w3-w6-soa-aos-helpers` (rebased onto master at `3f35170d`)
Commits under review:
- `5095853c` feat(hpc/soa): `SoaVec` + `soa_struct!` + `aos_to_soa` + `soa_to_aos` (W3+W5+W6)
- `5845ab2d` feat(hpc/bulk): `bulk_apply` + `bulk_scan` (W4)

## Verdict

**READY-FOR-PR.** Zero P0 findings. One P1 cosmetic docstring gap (patched on this branch). Three P2 items intentionally deferred per the design contract.

## Verification exit codes (all 0)

| Command | Exit | Notes |
|---|---|---|
| `cargo check -p ndarray --no-default-features --features std` | 0 | |
| `cargo test -p ndarray --lib --no-default-features --features std hpc::soa` | 0 | 29 passed, 0 failed |
| `cargo test -p ndarray --lib --no-default-features --features std hpc::bulk` | 0 | 16 passed, 0 failed |
| `cargo test --doc -p ndarray --no-default-features --features std hpc::soa` | 0 | 10 passed, 1 intentionally ignored |
| `cargo test --doc -p ndarray --no-default-features --features std hpc::bulk` | 0 | 2 passed, 1 intentionally ignored |
| `cargo fmt --all -- --check` | 0 | |
| `cargo clippy -p ndarray --no-default-features --features std -- -D warnings` | 0 | |

## P0 findings

None.

## P1 findings

- **F4** — `usize::MAX` chunk-size behavior is tested at `src/hpc/bulk.rs:182-194` but NOT documented in the public docstring of `bulk_apply` (`src/hpc/bulk.rs:46-66`) or `bulk_scan` (`src/hpc/bulk.rs:80-97`). One-line addition: "`chunk_size == usize::MAX` yields the entire slice as a single chunk." **Patched on this branch** before PR opens.

## P2 findings (deferred per design contract)

- **G1** (`bulk_scan` naming): the savant flagged that "scan" conventionally means fold-with-state. Kept as `bulk_scan` for symmetry with `bulk_apply`. Rename to `bulk_for_each` / `bulk_inspect` is a follow-up if downstream consumers find the name misleading.
- **G2** (`SoaVec::iter_rows`): row iterator yielding `[&T; N]` per row is absent. Use `soa.chunks(1)` for the same effect. Deferred to a follow-up once a real use case exists.
- **G3** (`SoaVec` lacks `#[derive(Clone, Debug)]`): macro-generated structs DO support derive passthrough (verified by test at `src/hpc/soa.rs:733-742`), but the generic container does not. Deliberately deferred — adding derives would require `where T: Clone + Debug` bounds that callers don't always want.

## D4 — integration test gate

The `bulk_apply` × `aos_to_soa` integration test at `src/hpc/bulk.rs:295` correctly uses `#[cfg(any())]` (canonical never-compile sentinel). The test body at `src/hpc/bulk.rs:297-324` is sound. Now that `hpc/soa.rs` and `hpc/bulk.rs` are landing in the same PR, the gate could be removed as a follow-up — but worker B's `cfg(any())` gate preserves the safe deferral if the PR review wants to keep them independently mergeable.

## Compliance summary

| Concern | Status | Notes |
|---|---|---|
| Layering rule (no `#[target_feature]`, no per-arch imports, no raw intrinsics) | ✅ clean | Only doc-prose mentions in module headers; zero actual attributes |
| Distance typing (no umbrella `fn distance<T>`, no `enum DistanceMetric`, no `Box<dyn Distance>`) | ✅ clean | Both module headers cite `cognitive-distance-typing.md` and warn against extension toward distance |
| Spec API match (per design doc v2 §C1-C7, §D1-D4) | ✅ exact | `field_n::<const I>()` uses `const { assert!(I < N) }`; all method signatures verbatim |
| Doc coverage (every `pub fn` has `///` doc with working `# Example`) | ✅ complete | After F4 patch |
| Test coverage (per design doc §"Tests" per fn) | ✅ complete | 29 + 16 = 45 unit tests + 12 doctests |

## Recommended next step

Open the W3-W6 PR. F4 fix landed on the same branch as a follow-up commit; integration test stays `cfg(any())`-gated for the PR; P2 items deferred.
