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

Affected (measured, `src/simd_avx2.rs` + the always-compiled natives in
`src/simd_avx512.rs`):

| item | AVX2-only ops | notes |
|---|---|---|
| `U8x32` | 21 | biggest; `__m256i` native |
| `U16x16` | 12 | many 1-method blocks |
| `U64x8` | 10 | `sllv`/`srlv` epi64 have no SSE form → per-lane |
| `U32x8` | 6 | shuffles/unpacks |
| `U32x16` | 5 | blake3 transpose |
| `I32x16` | 4 | |
| `U8x64` | 3 | 3 of 20 methods |
| `popcount`, `dot_i8` (free fns) | 6, 4 | |
| `I8x32`, `I16x16` (simd_avx512.rs) | 8, 8 | `__m256i` native |
| `U16x8` gather, `palette_lookup_u8x8` | 1, 2 | |
| `F32x8` | 1 | `permute2x128` → `permute2f128` in place (exact) |

## Work items

- [ ] P0 TESTS FIRST: a `native-avx` arm in `crates/simd-masking-parity`
      (`--config .cargo/config-avx.toml`, run under `qemu-x86_64-static -cpu
      SandyBridge`), wired into `scripts/masking-parity.sh` and the
      `simd-matrix` CI workflow. Expect it to SIGILL before the backend exists.
- [ ] P1 `.cargo/config-avx.toml`; arm predicate; `cpu_guard` floor gated.
- [ ] P2 split the affected impl blocks (no behaviour change on other arms;
      codegen-witness avx2/avx512 must stay byte-for-byte green).
- [ ] P3 `simd_avx.rs` per type, one type per chunk (U8x32, U16x16, U64x8,
      U32x8, U32x16, I32x16, U8x64, popcount/dot_i8, I8x32, I16x16, U16x8,
      palette_lookup_u8x8). Every `unsafe` gets SAFETY + sentinel-qa audit.
- [ ] P4 gates: lib tests on the avx arm under qemu SandyBridge; parity on all
      six existing arms + the new one; v3/v4 clippy; disable-run per type.
- [ ] P5 docs: CLAUDE.md realization list, contract doc, blackboard, unsafe
      inventory rows for the new file.

Estimate: ~1,000–1,500 lines, 2–3 sessions.
