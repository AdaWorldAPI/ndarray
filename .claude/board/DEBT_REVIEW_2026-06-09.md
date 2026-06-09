# Debt Review — unsafe / clippy / unused (2026-06-09)

> **Scope:** `adaworldapi/ndarray` (this repo) + cross-ref to `adaworldapi/lance-graph`.
> **Branch:** `claude/quirky-volta-m2r6ak`. **Toolchain:** Rust 1.95.0 (pinned).
> **Method:** workspace clippy (`--workspace --all-targets`), a `--force-warn`
> pass to measure suppression-masked debt, `cargo-machete` for unused deps, and
> grep censuses for unsafe/SAFETY ratios + `#[allow]` suppressions. Numbers are
> measured. `ndarray` has no `TECH_DEBT.md`; this dated doc is the ledger
> artifact (same convention as `SIMD_REVIEW_FIXES_2026_05_13.md` /
> `UNUSED_INVENTORY_1.95.md`).

## Cross-repo headline

| | **ndarray** | **lance-graph** |
|---|---|---|
| Workspace clippy | 🔴 RED → 🟡 (one RED fixed this session; one pre-existing by-design `compile_error!` remains) | 🟢 GREEN (53 warnings, ~17 intentional deprecations) |
| Real issue | Green `-D warnings` gate is **suppression-masked** across the new code | Healthy; main debt is **unused deps** |
| Unsafe | Large undocumented gap in `hpc/`+`simd_*` (clippy can't see it) | Well-quarantined (heavy unsafe in *excluded* `holograph`) |

Full lance-graph detail: `lance-graph/.claude/board/DEBT_REVIEW_2026-06-09.md`.

---

## 1. Clippy

### 1a. FIXED this session — `cesium` `erasing_op` (was the workspace RED)
`crates/cesium/src/to_cam_soa.rs:518-519` had pedagogical `0 * 3` / `0 * 16`
literals in the `sh_repack_index_formula` test. `clippy::erasing_op`
(deny-by-default) flagged them as ERRORS, making
`cargo clippy --workspace --all-targets` fail (exit 101). Fixed by introducing
named `(k, ch)` indices so the formula reads `k*3+ch` / `i*sh_stride+ch*16+k`
(no suppression, intent preserved). Verified: `cargo clippy -p cesium
--all-targets` → exit 0; cesium absent from the workspace error list.

### 1b. Pre-existing, by-design (NOT fixed — out of scope)
`crates/blas-tests/src/lib.rs:2` has an intentional `compile_error!("Missing
backend…")` when no BLAS feature is selected. So `cargo clippy --workspace
--all-targets` still exits 101 *unless* a backend is chosen
(`--features openblas-system` / `intel-mkl` / `native`). Relevant to
recommendation #2 below: any move to widen CI clippy to `--workspace` must
select a backend or `--exclude blas-tests numeric-tests`.

### 1c. The real debt — the green gate is suppression-masked
CI runs clippy on the **root package only** (`.github/workflows/ci.yaml:59,60,172`:
`cargo clippy --features approx,serde,rayon -- -D warnings`, `--features native`,
`-p ndarray --features rayon --lib`). Never `--workspace`, never `--all-targets`
across members. Consequences:
- **Member crates are ungated**: `cesium` (above) and `blas-mock-tests`
  (4 `missing_safety_doc` on unsafe fns, `src/lib.rs:17,39,61,83`) accumulate
  invisibly.
- **The new code has clippy switched off.** `#![allow(clippy::all,
  unused_imports, dead_code)]` blankets the entire `hpc/` tree
  (`src/hpc/mod.rs:1` — 91 files / 16 sub-dirs) and **13** module decls in
  `src/lib.rs` carry `#[allow(clippy::all, missing_docs, dead_code,
  unused_variables, unused_imports)]` — covering all 12 `simd_*` modules + `hpc`.
  Plus `src/backend/kernels_avx512.rs:12` `#![allow(dead_code)]`.

**Measured hidden debt** (`cargo clippy -p ndarray --lib --features
approx,serde,rayon -- --force-warn clippy::all`): **397 default-group warnings**
in the lib (111 auto-fixable). Biggest cluster: `missing_transmute_annotations`
on SIMD transmutes (mechanical turbofish). With `--force-warn clippy::pedantic`
too: **6602** — but ~411 of those are `cast_possible_truncation` (expected
numeric-cast noise in SIMD code; fine to keep allowed).

## 2. Unsafe

`src/` totals: **790 `unsafe {`**, 233 `unsafe fn`, 93 `unsafe impl` vs **219
`// SAFETY:`** + 104 `# Safety` doc sections.

Split by where the audit concern lives:
- **New code** (`hpc/` + `simd_*`): **460 `unsafe {` vs 148 `// SAFETY:`** → a
  documentation gap of ~300 blocks (upper bound — some are covered by fn-level
  `# Safety` docs). The Hard Rule "*every unsafe block needs a `// SAFETY:`
  comment*" is materially unmet here, and because `clippy::missing_safety_doc`
  is suppressed across this tree (the §1c blanket allow + 10 explicit
  `#[allow(clippy::missing_safety_doc)]`), **the gap is unenforced** — tooling
  cannot see it.
- **Upstream-core** (313 `unsafe {`): inherited from upstream ndarray
  (iterators, raw views, dimension). Audited upstream; lower priority.

## 3. Unused

### 3a. Dead code / unused imports
`src/` `#[allow]` census: 20 `dead_code`, 12 `unused_imports`. Already triaged in
`UNUSED_INVENTORY_1.95.md` → **A1–A9 actionable** (stale Rust-1.64 compat imports
at `impl_owned_array.rs:7` / `iterators/mod.rs:24`; phantom
`SimdTier::{Sse2,WasmSimd128}` at `hpc/simd_dispatch.rs:47`; dead CLAM/jitson
helpers) vs B1–B10 intentional scaffolding. Not yet executed.

### 3b. Unused dependencies (`cargo-machete`)
The **main `ndarray` lib is clean**. Flagged deps are all in auxiliary crates:
- `crates/burn` → 14 unused (vendored crate — partial vendor, expected).
- `crates/blas-tests` → `{blis-src, netlib-src, openblas-src}` and
  `crates/numeric-tests` → `openblas-src`: **false positives** — `-src` linker
  crates used for their build-time link side-effect, not imported in Rust.
- `crates/p64` → `fractal` (1) — worth a look.

## Prioritized actions (ndarray)
1. ✅ **DONE** — cesium `erasing_op` (workspace clippy RED → cesium-clean).
2. Widen CI clippy beyond the root package (at least add `cesium` +
   `blas-mock-tests`; for `--workspace` select a BLAS feature or exclude the
   backend-guarded test crates) so member debt stops accruing invisibly.
3. Narrow the `hpc/`+`simd_*` blanket `#![allow(clippy::all)]` to the specific
   noisy lints (`cast_possible_truncation`, `missing_transmute_annotations`),
   then burn down the ~397 residual default-group warnings. **Critically, stop
   suppressing `missing_safety_doc`** so the ~300-block unsafe gap becomes
   visible and enforceable.
4. Execute `UNUSED_INVENTORY_1.95.md` A1–A9 (already triaged; ~1 small PR).

## Environment note
`lance-graph` workspace clippy required `protoc` (protobuf-compiler), which was
**absent** from this environment; installed `protoc 3.21.12` to complete the
review. Not ndarray-scoped, but the shared CI image must provide it.
