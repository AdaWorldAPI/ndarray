# SIMD Dispatch Architecture — design, parity, tech debt, integration plan

> Date: 2026-05-20 · Status: design v1 (post PR #170 PR-X12 A1 discussion).
> Companion to: `vertical-simd-consumer-contract.md` (W1a consumer contract),
> `databend-ndarray-simd-prompt.md`, `ndarray-simd-trojan-horse-prompt.md`.

## 1. Why this exists

`ndarray::simd::*` is the single public surface every cognitive-shader,
splat, codec, BLAS, and FFI consumer reaches for. The current dispatch
in `src/simd.rs` is **compile-time-only** with arms keyed off
`target_feature = "avx512f"` / `target_arch = "aarch64"` / scalar
fallback. `.cargo/config.toml` pins `target-cpu = x86-64-v4`, baking
AVX-512 into every compiled artifact.

The consequence surfaced on PR #170 (`tests/1.95.0` CI run
[26151746204/76920666348](https://github.com/AdaWorldAPI/ndarray/actions/runs/26151746204/job/76920666348)):
**38 tests in `simd_avx2`, `simd_amx`, `simd_ops`, `simd_soa` SIGILL** on
a GitHub runner without AVX-512 silicon, all timing out uniformly
~19 s — the symptom of "binary cannot execute" rather than assertion
failure. The same configuration also leaves `simd_nightly/*` (the
portable-SIMD polyfill backend) unreachable because no dispatch arm in
`simd.rs` re-exports from it.

This document pins the target architecture, captures the parity gaps,
ranks the technical debt, and sequences the integration.

## 2. Dispatch model — three build configs, one runtime mode

Each build mode is a **conscious cargo invocation** via a distinct
`.cargo/config*.toml`. No silent fallbacks, no surprise hardware
mismatch. Whoever builds with `v3` / `v4` / `native` / `nightly-simd`
chose it deliberately.

| Config file | `target-cpu` | Dispatch strategy | Default? | Use case |
|---|---|---|---|---|
| `.cargo/config.toml` | `x86-64-v3` (AVX2) | compile-time → `simd_avx2` | ✅ default, GitHub CI | portable artifact across all x86_64 silicon ≥ 2013 |
| `.cargo/config-avx512.toml` | `x86-64-v4` (AVX-512) | compile-time → `simd_avx512` | explicit | benchmarking, AVX-512 deployment |
| `.cargo/config-native.toml` | `native` | compile-time, build-machine CPUID resolved at rustc invocation → whatever arm matches the build host | explicit | developer machine builds |
| `.cargo/config-nightly.toml` (+ `--features nightly-simd`) | `x86-64-v3` (or any) | compile-time → `simd_nightly` (`std::simd::*` polyfill) | explicit | miri / cargo-careful / portable-SIMD experiments |

The aarch64 path is automatic: any `target_arch = "aarch64"` build
selects `simd_neon` regardless of the config above.

**Runtime LazyLock dispatch** is a separate, fifth opt-in mode used
when shipping a single release binary that must adapt at process
start across heterogeneous deployment silicon (one binary running on
AVX-512 + AVX2-only machines from the same artifact). It compiles all
backends in and uses `LazyLock<CpuCaps>` trampolines. Reserved for the
release-binary distribution path; never the dev / CI default.

### Dispatch precedence in `simd.rs`

Compile-time arms read like a cascade, **not** like priority overrides
— each cargo config sets exactly one `target_feature` / `feature` such
that exactly one arm matches. The order below is the source-of-truth
ranking the compiler walks:

```rust
// 1. Explicit portable-SIMD polyfill (nightly + opt-in feature).
//    No `target_arch` constraint — `core::simd` is portable, so this
//    arm is the one true backend on wasm32 / riscv / any other target
//    as soon as `nightly-simd` is on. Keeping it unconditional on
//    `feature = "nightly-simd"` is what makes the `not(feature =
//    "nightly-simd")` exclusion on every other arm sound.
#[cfg(feature = "nightly-simd")]
pub use crate::simd_nightly::{F32x16, F64x8, U8x32, U8x64, U16x32, U32x16, U64x8, I8x32, I8x64, I16x16, I16x32, I32x16, I64x8, F32Mask16, F64Mask8, BF16x16, BF16x8};

// 2. AVX-512 (target_feature = "avx512f"; set by `v4` and `native` configs on AVX-512 hosts)
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f", not(feature = "nightly-simd")))]
pub use crate::simd_avx512::{...};

// 3. AVX2 baseline (the v3 / GitHub-CI default)
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", not(target_feature = "avx512f"), not(feature = "nightly-simd")))]
pub use crate::simd_avx2::{...};

// 4. NEON (aarch64)
#[cfg(all(target_arch = "aarch64", not(feature = "nightly-simd")))]
pub use crate::simd_neon::aarch64_simd::{...};

// 5. Scalar fallback (everything else: wasm32, riscv, x86_64 without
//    AVX2, etc.). The predicate is the negation of arms 1-4 so that
//    *exactly one* arm matches on every (target, feature) pair.
#[cfg(not(any(
    feature = "nightly-simd",
    all(target_arch = "x86_64", target_feature = "avx2"),
    target_arch = "aarch64",
)))]
pub use scalar::{...};
```

Runtime dispatch via `LazyLock<CpuCaps>` lives in a separate
`simd_runtime` module (TBD per § 7.1) reached by a `--features runtime-dispatch`
flag, mutually exclusive with the compile-time arms above.

## 3. Module roles

```
crate::simd::*                ← user-facing key registry (re-exports only)
        │
        ├── simd.rs           = dispatch arms; no implementation, only `pub use`
        │
        ├── simd_ops.rs       = slice-level ops over crate::simd::* primitives
        │                       (add_f32, scale_f64, array_chunks, …)
        │
        ├── simd_avx512.rs    = __m512* values, native 512-bit registers
        ├── simd_avx2.rs      = __m256* values + (F32x16, F64x8) as two-half
        │                       wrappers (struct F32x16(pub f32x8, pub f32x8))
        ├── simd_neon.rs      = float32x4_t / uint64x2_t natives + larger shapes
        │                       composed as [float32x4_t; 4] etc.
        ├── simd_nightly/     = std::simd::* polyfill — portable, miri-executable
        │   ├── f32_types.rs    F32x16, F32x8
        │   ├── f64_types.rs    F64x8, F64x4
        │   ├── u8_types.rs     U8x64, U8x32
        │   ├── u_word_types.rs U16x32, U32x16, U64x8
        │   ├── i8_types.rs     I8x64, I8x32
        │   ├── i_word_types.rs I16x16, I16x32, I32x16, I64x8
        │   ├── bf16_types.rs   BF16x16, BF16x8
        │   ├── f16_types.rs    F16x16
        │   ├── masks.rs        F32Mask16, F32Mask8, F64Mask4, F64Mask8
        │   └── ops.rs          op impls
        └── scalar (inline `mod scalar` in simd.rs)
                              = pure-Rust fallback for unknown arch
```

Every `simd_<arch>.rs` is just a SOURCE of typed primitives. `simd.rs`
chooses the source; the cargo config chooses how `simd.rs` chooses.

## 4. Parity matrix — typed lane primitives per backend

Legend: ✅ native, 🟡 composed wrapper (two-half / four-quarter), 🟠
scalar polyfill (struct exists with full API but storage is `[$elem;
$lanes]` — no SIMD execution), 🔵 portable-SIMD polyfill via
`core::simd`, ❌ missing, ⛔ N/A for this arch.

(Reality check 2026-05-20: many AVX2 int rows previously marked ❌ are
actually 🟠 — `simd_avx2.rs` ships them via the `avx2_int_type!` macro
as scalar-storage structs that match the AVX-512 API surface. The
arithmetic is plain Rust under the hood; only the FLOAT wrappers in
this column are true two-half SIMD composites. Filling in real AVX2
vectorization for the int wrappers is its own piece of tech debt
tracked as TD-SIMD-3.)

| Lane type | `simd_avx512` (v4) | `simd_avx2` (v3) | `simd_neon` (aarch64) | `simd_nightly` | `scalar` |
|---|---|---|---|---|---|
| `F32x16` | ✅ `__m512` | 🟡 `(f32x8, f32x8)` | 🟡 `[float32x4_t; 4]` | 🔵 `core::simd::f32x16` | ✅ `[f32; 16]` |
| `F32x8` | ✅ `__m256` | ✅ `__m256` (in `simd_avx512`) | ⛔ | 🔵 | ✅ |
| `F64x8` | ✅ `__m512d` | 🟡 `(f64x4, f64x4)` | 🟡 `[float64x2_t; 4]` | 🔵 | ✅ |
| `F64x4` | ✅ `__m256d` | ✅ `__m256d` (in `simd_avx512`) | ⛔ | 🔵 | ✅ |
| `U8x64` | ✅ `__m512i` | 🟠 `[u8; 64]` polyfill | ❌ | 🔵 | ✅ |
| `U8x32` | ✅ `__m256i` | ✅ `__m256i` | ❌ | 🔵 | ✅ |
| `U16x32` | ✅ `__m512i` | 🟠 `[u16; 32]` polyfill | ❌ | 🔵 | ✅ |
| `U16x16` | ❌ | ❌ | ❌ | ❌ | ❌ |
| `U32x16` | ✅ `__m512i` | 🟠 `[u32; 16]` polyfill | ❌ | 🔵 | ✅ |
| `U32x8` | ❌ | ❌ | ❌ | 🔵 `core::simd::u32x8` | ❌ |
| `U64x8` | ✅ `__m512i` | 🟠 `[u64; 8]` polyfill | ❌ | 🔵 | ✅ |
| `U64x4` | ❌ | ❌ | ❌ | 🔵 `core::simd::u64x4` | ❌ |
| `I8x32` | ✅ `__m256i` | ✅ `__m256i` (in `simd_avx512`) | ❌ | 🔵 | ✅ |
| `I8x64` | ✅ `__m512i` | 🟠 `[i8; 64]` polyfill | ❌ | 🔵 | ✅ |
| `I16x16` | ✅ `__m256i` | ✅ `__m256i` (in `simd_avx512`) | ❌ | 🔵 | ✅ |
| `I16x32` | ✅ `__m512i` | 🟠 `[i16; 32]` polyfill | ❌ | 🔵 | ✅ |
| `I32x16` | ✅ `__m512i` | 🟠 `[i32; 16]` polyfill | ❌ | 🔵 | ✅ |
| `I32x8` | ❌ | ❌ | ❌ | ❌ | ❌ |
| `I64x8` | ✅ `__m512i` | 🟠 `[i64; 8]` polyfill | ❌ | 🔵 | ✅ |
| `I64x4` | ❌ | ❌ | ❌ | ❌ | ❌ |
| `BF16x8` | ✅ `__m128bh` | ❌ | ❌ | 🔵 | ✅ |
| `BF16x16` | ✅ `__m256bh` | ❌ | ❌ | 🔵 | ✅ |
| `F16x16` | ❌ | 🟠 `[u16; 16]` polyfill (`simd_half`) | 🟠 `[u16; 16]` polyfill (`simd_half`) | 🔵 | ✅ |
| `F32Mask16` | ✅ `__mmask16` | ✅ `u16` bitmask | ✅ `u16` bitmask | 🔵 | ✅ |
| `F32Mask8` | ❌ exposed (avail via `__mmask8`) | ❌ exposed (avail in `simd_scalar` as `F32Mask8Scalar`) | ❌ exposed | 🔵 | ✅ via `F32Mask8Scalar` |
| `F64Mask8` | ✅ `__mmask8` | ✅ `u8` bitmask | ✅ `u8` bitmask | 🔵 | ✅ |
| `F64Mask4` | ❌ exposed (avail via `__mmask8`) | ❌ exposed (avail in `simd_scalar` as `F64Mask4Scalar`) | ❌ exposed | 🔵 | ✅ via `F64Mask4Scalar` |

### Sub-byte lanes (not a SIMD wrapper anywhere)

**`I4` / `U4`** — 4-bit (nibble) lanes used by INT4 quantized inference
(GGUF Q4_0 / Q4_K, GPTQ, AWQ). No first-class wrapper exists or is
planned. Consumers pack 2× nibbles per byte and operate through
`U8x64` with `shr_epi16` + `& 0x0F` masks; the same trick gives them
the 128- and 64-byte shapes via the existing AVX-512 / AVX2 / NEON
paths. If a first-class `I4x128` were ever wanted, AVX-512 VBMI2's
`VPCOMPRESSB` + `VPEXPANDB` and AVX-512 IFMA's `VPMADD52` give the
hardware story; on aarch64 there's no native nibble support and the
shr+mask trick stays. Tracked as TD-SIMD-11 if a consumer files for it.

### Aarch64-native narrower types

Only useful directly when the consumer wants 128-bit shapes:
`I8x16`, `I16x8`, `U8x16`, `U16x8`, `U32x4`, `U64x2`, `I32x4`, `I64x2`.
These are not in the cross-arch parity surface — consumers requesting
256-bit / 512-bit shapes go through the composed wrappers.

### Gaps surfaced 2026-05-20

- **`F32x8` / `F64x4` are universal on x86**, even on the v3 / AVX2 path
  — they share the `__m256` / `__m256d` declarations exposed by
  `simd_avx512.rs` (AVX, not AVX-512; works on every host with AVX
  support, i.e. Sandy Bridge+). The previous matrix marked them `❌`
  in the v3 column — corrected above.
- **`U32x8` / `U64x4`** exist only in `simd_nightly` (via `core::simd`).
  No native or polyfill wrapper on x86 or aarch64. Add to `simd_avx512`
  + `simd_scalar` if a consumer needs them at 256-bit width.
- **`I32x8` / `I64x4` / `U16x16`** missing across every backend (incl.
  nightly). Theoretical 256-bit shapes that no consumer has reached for
  yet; add to backlog if needed.
- **`F32Mask8` / `F64Mask4`** are declared in `simd_scalar` as
  `F32Mask8Scalar` / `F64Mask4Scalar` (the rename came from a duplicate-
  decl conflict on i686 — see `src/simd_scalar.rs:340-345`). Not
  surfaced through `crate::simd::*`. If consumers want these mask
  widths, expose them and unify the name (drop the `Scalar` suffix on
  AVX-512 where `__mmask8` natively maps to F64Mask8 already; the
  256-bit f64 lane width needs a 4-bit mask which `__mmask8` can hold
  but isn't yet typed as `F64Mask4`).

### Read of the matrix

- **F32x16 + F64x8 are universal** — all four backends ship them. Hot
  paths can rely on these without branching.
- **`simd_avx2` is the bottleneck.** It only exposes `F32x16`, `F64x8`,
  `F32Mask16`, `F64Mask8`, `U8x32`, and an `F16Scaler`. Every other
  cross-arch lane is missing — making the v3 default config crash any
  consumer that reaches for `U64x8`, `I32x16`, `U16x32`, etc.
- **NEON is even sparser** at the 256/512-bit level.
- **`simd_nightly` is the most complete** but is unreachable today
  because `simd.rs` has no arm wiring `feature = "nightly-simd"` to its
  re-exports.
- **`scalar`** has comprehensive cover and is the safest fallback for
  any arch the others miss, but lives inline in `simd.rs` rather than
  in a dedicated `simd_scalar.rs`. Symmetry would help.

## 5. Technical debt matrix

Ranked by P0 (blocks current CI / consumers) → P3 (nice-to-have).

| ID | Severity | Description | Detection | Fix scope |
|---|---|---|---|---|
| **TD-SIMD-1** | **P0** | `.cargo/config.toml` defaults to `x86-64-v4` → every CI runner without AVX-512 silicon SIGILLs on the first SIMD op. 38 tests fail at 19 s timeout each on `tests/1.95.0`. | PR #170 CI run | Change default to `x86-64-v3`; add `.cargo/config-avx512.toml` for the opt-in AVX-512 path. ~5 LoC. |
| **TD-SIMD-2** | **P0** | `simd_avx2.rs` ships `F32x16`/`F64x8`/`U8x32` only. Consumers requesting `U64x8`, `I32x16`, `U16x32`, `BF16x16`, etc. fail to compile on the v3 path. | grep `pub use crate::simd_avx2::` then cross-ref against the parity matrix | Add the missing types as two-half wrappers (`U64x8(pub u64x4, pub u64x4)` etc.) over native `__m256i` halves. ~500 LoC. |
| **TD-SIMD-3** | **P1** | `simd.rs` has no dispatch arm for `#[cfg(feature = "nightly-simd")]` → the `simd_nightly` polyfill is unreachable. miri / cargo-careful jobs that should exercise the portable path fall through to whatever cfg cascade matches, never to `std::simd::*`. | grep `simd_nightly` in `simd.rs` (returns 0 dispatch arms) | Add the `feature = "nightly-simd"` arm at the top of the cascade per § 2. ~30 LoC. |
| **TD-SIMD-4** | **P1** | `simd_neon.rs` only ships `F32x16` / `F64x8` cross-arch shapes. Consumers reaching for `U8x64`, `U64x8`, `I32x16`, etc. on aarch64 have no path. | grep + parity matrix | Compose larger shapes from native NEON 128-bit lanes (`U8x64([uint8x16_t; 4])`, `U64x8([uint64x2_t; 4])`, etc.). ~400 LoC. |
| **TD-SIMD-5** | **P1** | Scalar fallback inline in `simd.rs` (`pub(crate) mod scalar`) makes symmetry hard — every other backend is its own file. | inspection | Promote to `src/simd_scalar.rs`; `simd.rs` becomes pure dispatch. ~mechanical refactor. |
| **TD-SIMD-6** | **P2** | No `runtime-dispatch` feature / `simd_runtime` module exists yet. Release-binary distribution to heterogeneous silicon requires recompile per target today. | `grep -r "LazyLock<CpuCaps>"` only matches reporting code in `simd.rs:52-55` | New module wiring per-op trampolines from the compiled-in backends. ~300 LoC + one new cargo feature. |
| **TD-SIMD-7** | **P2** | Compile-time arms in `simd.rs:153-194` are duplicated four times (one per type group: F32x16, F64x8, U8x32, BF16x16). Adding a new lane requires copy-pasting four `#[cfg(...)]` arms. | inspection | Single source-of-truth macro emitting the arms. ~one macro_rules!, 50 LoC. |
| **TD-SIMD-8** | **P2** | `F16x16` in `src/simd_half.rs:123` is a scalar `[u16; 16]` polyfill — every arithmetic op upcasts to f32, computes, downcasts. Consumers using `crate::simd::F16x16` get scalar perf even on AVX-512 hardware with `vcvtph2ps` / `vcvtps2ph`. (`F16Scaler` in `simd_avx2.rs:2566` is unrelated — it's a *scaling context* for range-normalizing values before f16 encoding, not the F16x16 SIMD type.) | inspection of `src/simd_half.rs:115-150` | (a) Replace the `[u16; 16]` storage with `__m256i` + `_mm256_cvtph_ps` / `_mm256_cvtps_ph` under `target_feature = "f16c"` (Sapphire Rapids+, all Skylake AVX-512). (b) Add an `F16x16Scalar` alias and route consumers explicitly. (c) Add a doc-warning at the type level pointing at the architecture doc. ~80 LoC. |
| **TD-SIMD-9** | **P3** | No CI matrix entry for the `nightly-simd` polyfill path. | `.github/workflows/ci.yaml` | Add a `nightly-simd-polyfill` job that builds with `--features nightly-simd` on nightly rustc. ~20 LoC YAML. |
| **TD-SIMD-10** | **P3** | No CI matrix entry for `.cargo/config-avx512.toml`. AVX-512 deployment path silently bit-rots between PRs. | `.github/workflows/ci.yaml` | Add an `avx-512-explicit` job using a runner with AVX-512 silicon. ~20 LoC YAML; runner availability TBD. |

## 6. Integration plan — sequenced sprints

Each phase is a single-PR worker (sized for one Sonnet impl-sprint per
the `.claude/EN/agents/worker-template.md` shape). Phases sequence so
each lands a green CI; the next phase depends only on shipped state.

### Phase 1 — Unblock CI (P0 fixes)

**Goal:** GitHub `tests/1.95.0` job green. The default `.cargo/config.toml`
build runs end-to-end on AVX2-only silicon.

| # | Worker | Scope | Files | Acceptance |
|---|---|---|---|---|
| 1.1 | flip baseline | Change `target-cpu` from `v4` → `v3`. Add `.cargo/config-avx512.toml` with the old `v4` value. | `.cargo/config.toml`, `.cargo/config-avx512.toml` | `cargo check` clean on default; `tests/1.95.0` no longer SIGILLs |
| 1.2 | AVX2 two-half wrappers — float | Add `U8x64`, `U64x8`, `U32x16`, `U16x32`, `I8x32`, `I8x64`, `I16x16`, `I16x32`, `I32x16`, `I64x8` as two-half wrappers over native AVX2 `__m256i` halves. | `src/simd_avx2.rs` | per-type parity test vs `simd_avx512` on AVX-512 host; per-type unit test on AVX2-only |
| 1.3 | simd.rs dispatch refresh | Add the AVX2 cfg arm wiring the new wrappers; tighten existing arms with the new precedence (per § 2). | `src/simd.rs` | `cargo check --features approx,serde,rayon` clean on default config; `cargo check` clean on `--config .cargo/config-avx512.toml` |

After Phase 1, PR #170 (PR-X12 A1) and any future consumer PR ships
green CI by default. AVX-512 testing becomes an explicit job.

### Phase 2 — Unblock the polyfill (P1: `nightly-simd`)

**Goal:** `cargo +nightly check --features nightly-simd` reaches
`simd_nightly/*` via `crate::simd::*`. miri can execute the portable
path.

| # | Worker | Scope | Files | Acceptance |
|---|---|---|---|---|
| 2.1 | nightly-simd dispatch arm | Add `#[cfg(feature = "nightly-simd")]` arms in `simd.rs` re-exporting every typed lane from `crate::simd_nightly::*`. | `src/simd.rs` | `crate::simd::F32x16` resolves to `core::simd::f32x16` under the feature |
| 2.2 | nightly-simd parity tests | Run the existing simd_ops / simd_soa test suite against the polyfill backend. | `src/simd_nightly/tests.rs` | all simd_ops + simd_soa tests pass under `--features nightly-simd` |
| 2.3 | CI matrix | Add `nightly-simd-polyfill` job to `.github/workflows/ci.yaml`. | `.github/workflows/ci.yaml` | job green on nightly rustc with the feature |

### Phase 3 — NEON parity (P1)

**Goal:** aarch64 build reaches the same cross-arch lane set as the v3
config.

| # | Worker | Scope | Files | Acceptance |
|---|---|---|---|---|
| 3.1 | NEON quartet wrappers | Compose `U8x64`, `U64x8`, `U32x16`, `U16x32`, `I8x32`, `I8x64`, `I16x16`, `I16x32`, `I32x16`, `I64x8` from native 128-bit NEON lanes. | `src/simd_neon.rs` | parity vs `simd_avx2` two-half wrappers on a 16-pair fixture |
| 3.2 | simd.rs aarch64 arms | Extend `aarch64` arms to re-export the new types. | `src/simd.rs` | `cargo check --target aarch64-unknown-linux-gnu` clean |

### Phase 4 — Symmetry + ergonomics (P1/P2)

| # | Worker | Scope | Files | Acceptance |
|---|---|---|---|---|
| 4.1 | scalar → file | Promote `mod scalar` to `src/simd_scalar.rs`. | `src/simd.rs`, new `src/simd_scalar.rs` | no behaviour change; `cargo check` clean on all configs |
| 4.2 | dispatch macro | Collapse the 4× duplicated `#[cfg(...)]` blocks into one macro. | `src/simd.rs` | adding a new lane type is one macro invocation |
| 4.3 | F16 honesty | Either rename `F16Scaler` or gate `F16x16` behind `f16c`. | `src/simd_avx2.rs` | scalar perf no longer surprises hot-path consumers |

### Phase 5 — Runtime dispatch (P2, opt-in)

**Goal:** ship-once binaries that adapt across heterogeneous deployment
silicon.

| # | Worker | Scope | Files | Acceptance |
|---|---|---|---|---|
| 5.1 | `simd_runtime` module | New module compiling all backends in and selecting per-op trampolines via `LazyLock<CpuCaps>`. | `src/simd_runtime.rs` | one binary runs on AVX-512 + AVX2-only hosts from the same artifact |
| 5.2 | feature flag | New `runtime-dispatch` cargo feature, mutually exclusive with `nightly-simd`. | `Cargo.toml`, `src/simd.rs` | `cargo check --features runtime-dispatch` clean on the v3 baseline |
| 5.3 | CI matrix | Add a `runtime-dispatch-portable` job. | `.github/workflows/ci.yaml` | job green |

### Phase 6 — CI matrix for explicit AVX-512 (P3)

| # | Worker | Scope | Files | Acceptance |
|---|---|---|---|---|
| 6.1 | AVX-512 explicit job | Add `avx-512-explicit` to `.github/workflows/ci.yaml` using `--config .cargo/config-avx512.toml`. Requires AVX-512-capable runner. | `.github/workflows/ci.yaml` | green on the AVX-512 runner |

## 7. Open questions

1. **Runtime trampoline cost class.** Phase 5's per-op indirection
   adds one indirect call per `F32x16::add(...)`. Acceptable for the
   typical 100+ cycle SIMD-op cost, but consumer benchmarks should
   sanity-check before declaring the path production-ready.
2. **`feature = "nightly-simd"` precedence.** § 2 puts it at the top
   of the cascade; alternative reading is "polyfill is for miri only,
   so put it BELOW the arch-specific arms and only fire on non-x86_64,
   non-aarch64 targets." The current proposal matches the user's
   "explicit opt-in wins" framing; revisit if there's a use case for
   `--features nightly-simd` on an AVX-512 host wanting the AVX-512
   path.
3. **AMX status.** `simd_amx.rs` (Sapphire Rapids+ tile ops) is
   x86_64-only and orthogonal to the F32x16 / U8x64 cross-arch surface.
   Out of scope for this document; tracked under PR-X10 A6
   (`linalg::distance`) follow-ups.

## 8. Cross-references

- `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a consumer
  contract every new SIMD primitive follows (struct methods on typed
  wrappers, three-backend parity test, saturating/overflow semantics
  documented).
- `.claude/knowledge/databend-ndarray-simd-prompt.md` — Databend
  integration consumer of `crate::simd::*`.
- `.claude/knowledge/ndarray-simd-trojan-horse-prompt.md` — ClickHouse +
  Tantivy injection plan; depends on Phase 1 + 2 landing.
- `src/simd.rs` lines 52-55 — existing `is_x86_feature_detected!`
  reporting (NOT dispatch) — repurpose for Phase 5 trampoline.
- `src/simd_nightly/mod.rs` lines 37-44 — complete `pub use` set
  ready to be wired into `simd.rs` dispatch (Phase 2).

## 9. TL;DR

Default cargo config drops to **`x86-64-v3`** (AVX2) → GitHub CI green by
default. **`.cargo/config-avx512.toml`** is the explicit AVX-512 path.
`simd_avx2.rs` needs ~10 missing two-half wrappers (P0, Phase 1).
`simd.rs` needs a `nightly-simd` dispatch arm so `simd_nightly/*`
becomes reachable (P1, Phase 2). NEON gets quartet wrappers (P1, Phase
3). Scalar / macros / runtime-dispatch / explicit-AVX-512 CI are
P2-P3 follow-ups (Phases 4-6). Each phase is one PR; landing
in order keeps every step green.
