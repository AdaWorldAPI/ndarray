# Plan: `simd_avx.rs` — AVX-without-AVX2 backend (v1)

Operator decision 2026-10-10 (after PR #348): Option A, a real backend for CPUs
with AVX but not AVX2 (Sandy Bridge / Ivy Bridge, AMD Bulldozer–Jaguar, VMs that
hide AVX2), selected at compile time. No runtime feature checks in the SIMD code.

## Selection

* Arm predicate: `all(target_arch = "x86_64", target_feature = "avx",
  not(target_feature = "avx2"))` — written out at each gate (no build.rs, no
  cfg-alias crate).
* Reached by: `target-cpu=native` on such a CPU, or the new
  `.cargo/config-avx.toml` (`-Ctarget-cpu=sandybridge`).
* Unchanged: AVX-512 builds, AVX2 builds, and BASELINE builds (no `avx` cfg).
  A baseline build keeps the AVX2 arm and the `cpu_guard` AVX2 startup floor —
  switching it to this arm would slow every baseline build on AVX2 hosts.
  (Open question for the operator; default is "unchanged".)
* `cpu_guard::BACKEND_FLOOR` applies only when the AVX2 arm is compiled in. On
  this arm the normal compiled-feature table already requires `avx`.

## Shape: impl blocks, not types

Measured: the AVX2-only intrinsics sit in impl blocks; the TYPES are shared and
referenced across the arm (`F32x16`→`I32x16`/`U32x16`, `F64x8`→`U64x8`,
`U16x32`→`U8x64`, `U8x32`→`U16x16`). Redefining types would cascade, so:

* Type definitions and `simd.rs` export lists do not change.
* In `simd_avx2.rs` / `simd_avx512.rs`, every method that calls an AVX2-only
  intrinsic moves into an impl block gated `not(<arm predicate>)`. Mixed blocks
  are split; their portable methods stay shared (no duplication).
* `simd_avx.rs` provides the same methods, same signatures, gated on the arm:
  two `__m128i` halves via SSE2/SSSE3/SSE4.1 (all present on every AVX CPU),
  plus AVX1 for loads/stores/float-domain bit ops.

Affected — ⊘ the first table here counted doc-comment mentions and listed
`U8x64`, `U32x16`, `U32x8`, `U16x8`, `palette_lookup_u8x8` and a 1-op `F32x8`
item. Re-measured with comments stripped (`tools/avx2_gate.py`): those are
scalar or comment-only and needed nothing. What actually needed a gate:

| item | gated in | notes |
|---|---|---|
| `U8x32` | `simd_avx2.rs` | 15 methods + 5 traits; biggest |
| `U16x16` | `simd_avx2.rs` | 8 methods + 12 traits |
| `U64x8` | `simd_avx2.rs` | `rotl_half`, `mul_lo32`, `transpose8`, `Shl`/`Shr<Self>` (per-lane: no SSE `sllv`) |
| `I32x16` | `simd_avx2.rs` | `reduce_min/max`, `gt_bitmask` |
| `popcount`, `dot_i8` (free fns) | `simd_avx2.rs` | re-exported from `simd_avx` on the arm |
| `I8x32`, `I16x16` | `simd_avx512.rs` | 6 + 5 methods, 4 traits each |
| `F32x8::mul_add` | `simd_avx512.rs` | FMA3, not AVX2 — missed by the AVX2-only scan, caught by sentinel-qa; now `cfg(target_feature = "fma")` with a per-lane `f32::mul_add` fallback |

49 gates in `simd_avx2.rs`, 19 in `simd_avx512.rs`. F16C sites in
`simd_avx512.rs` are runtime-detected (`is_x86_feature_detected!`) and need none.

## Work items

- [x] P0 TESTS FIRST: a `native-avx` arm in `crates/simd-masking-parity`
      (`--config .cargo/config-avx.toml`, run under `qemu-x86_64-static -cpu
      SandyBridge`), wired into `scripts/masking-parity.sh` and the
      `simd-matrix` CI workflow. Expect it to SIGILL before the backend exists.
- [x] P1 `.cargo/config-avx.toml`; arm predicate; `cpu_guard` floor gated.
- [x] P2 split the affected impl blocks (no behaviour change on other arms;
      codegen-witness avx2/avx512 must stay byte-for-byte green).
- [x] P3 `simd_avx.rs` per type, one type per chunk (U8x32, U16x16, U64x8,
      U32x8, U32x16, I32x16, U8x64, popcount/dot_i8, I8x32, I16x16, U16x8,
      palette_lookup_u8x8). Every `unsafe` gets SAFETY + sentinel-qa audit.
- [x] P4 gates: lib tests on the avx arm under qemu SandyBridge; parity on all
      six existing arms + the new one; v3/v4 clippy; disable-run per type.
- [x] P5 docs: CLAUDE.md realization list, contract doc, blackboard, unsafe
      inventory rows for the new file.

Estimate: ~1,000–1,500 lines, 2–3 sessions.
