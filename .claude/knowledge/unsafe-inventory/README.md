# `unsafe` inventory — 2026-10-10 (master `6704865` + `cpu_guard`, rustc 1.99.0)

READ BY: sentinel-qa, savant-architect, simd-savant, anyone removing or adding `unsafe`.

Every code-level `unsafe` in `src/` (comment lines excluded): **1,301 sites**.
**903 were added by this fork, 398 are inherited from upstream ndarray** (the
file exists in `rust-ndarray/ndarray` master `bd3ade9`). Kinds: 904 blocks,
292 `unsafe fn`, 93 `unsafe impl`, 10 `unsafe trait`, 2 `unsafe extern`.

Per-site data: [`sites.tsv`](sites.tsv), one row per site:
`file line origin kind scope verdict safety flag workaround ops snippet`.

## How it was made, and how much to trust it

1. A regex pre-pass recorded each site's kind and the operations visible in its
   body (pointer loads, intrinsics, FFI, `asm!`, transmute, …).
2. A real `cargo clippy --lib` on the native (Sapphire Rapids) build with
   `undocumented_unsafe_blocks` + `multiple_unsafe_ops_per_block`.
3. Ten reviewers each read one slice of the code and gave every row a verdict.
   **These are reading-level verdicts, not compiled ones.** Spot-checks below.
4. The orchestrator corrected one verdict class with the compiler (next section)
   and re-read the four most serious flags at source (all four held).

`scope=test` is crude (everything after a file's first `#[cfg(test)]`); the NEON
reviewer found rows after `simd_neon.rs:2753` mis-tagged as test.

## The compiler fact that decides most of the SIMD rows

`tools/safe_intrinsic_probe`, re-run on **rustc 1.99.0**:

| probe | result |
|---|---|
| plain fn, `_mm_and_si128` (SSE2, x86_64 baseline) | E0133 |
| plain fn, AVX2 intrinsic, `-Ctarget-cpu=x86-64-v3` | E0133 |
| plain fn, AVX-512 intrinsic, `-Ctarget-cpu=x86-64-v4` | E0133 |
| plain fn, AVX-512 intrinsic, `-Ctarget-feature=+avx512f` | E0133 |
| plain fn calling a SAFE `#[target_feature]` fn (x86 and aarch64) | E0133 |
| plain fn, NEON intrinsic, aarch64 (NEON is baseline) | E0133 |
| plain fn, `v128_and`, wasm32 `+simd128` | **OK** |

Also confirmed in-tree: removing the `unsafe` around a lone `_mm512_srlv_epi32`
(`simd_avx512.rs:1705`) fails with E0133 under both v3 and v4.

So on x86_64 and aarch64 a value intrinsic is safe to call ONLY inside a function
that itself carries `#[target_feature(enable = ..)]`. Crate-wide target features
do not count, and calling such a function from a plain one is unsafe again. Two
reviewers had marked 212 value-intrinsic blocks `REMOVABLE` on the 1.87 rule;
they are re-labelled `NEEDS-TF` here. **There is no per-site removal for them on
1.99.** The minimum is the current shape: one `unsafe` inside each polyfill
wrapper, so consumers of `ndarray::simd` write none. Only wasm32 is different.

## Verdicts

| verdict | fork | upstream | meaning, and the workaround |
|---|---|---|---|
| `IRREDUCIBLE` | 279 | 259 | FFI (MKL/OpenBLAS), `asm!` / AMX tile config, `unsafe impl Send/Sync`, ndarray's raw-pointer core, `#[target_feature]` calls after a runtime check. Keep; document. |
| `NEEDS-TF` | 212 | 0 | Value intrinsics. See above: no per-site fix on 1.99. |
| `CONSOLIDATE` | 185 | 1 | Genuinely unsafe pointer loads/stores (`_mm*_loadu/storeu`, `vld1q/vst1q`). Route each type's slice methods through its existing `from_array`/`to_array` (fixed-size `&[T; N]`), leaving ONE unsafe load and store per type. Several sites bypass helpers that already exist (e.g. `simd_neon.rs:2431`, `:3182`). |
| `SAFE-API` | 107 | 32 | A named safe std API replaces it: `as_chunks`, `<[T;N]>::try_from`, `to_bits`/`from_ne_bytes`, `copy_from_slice`, `NonNull::from_ref`, `Vec::into_flattened`, checked `from_shape_vec`, `HashMap::get_disjoint_mut` (`hpc/blackboard.rs`, 5 sites), a `dyn Any` downcast for same-`'static`-type transmutes. |
| `UNSAFE-FN-API` | 41 | 105 | `unsafe fn` whose contract is the point (`uget`, `from_shape_ptr`, …). |
| `SAFE-CRATE` | 27 | 1 | Only `bytemuck` / `zerocopy` make it safe. Needs operator approval for the dependency. |
| `NIGHTLY-BY-DESIGN` | 10 | 0 | `src/simd_nightly/*`: validation backend over `core::simd`, nightly only, unsafe by design. No SAFETY-comment requirement; inline notes where they help (operator, 2026-10-10). |
| `REMOVABLE` | 42 | 0 | `unsafe fn` keywords on bodies with no unsafe op: scalar fallbacks and forwarders in `simd_runtime/*`, `bgz17_bridge.rs` (10), `aabb.rs`, `byte_scan.rs`, `bitwise.rs`, … Caveat: where the fn is `#[target_feature]`, the keyword can go but its CALLERS still need `unsafe` (probe row 5); dropping the ATTRIBUTE changes codegen on v3 builds, so measure first. |

## `// SAFETY:` comments (CLAUDE.md hard rule)

856 sites lack one (526 fork, 330 upstream; nightly rows are exempt). The clippy run counts 439 blocks and
77 impls on the native x86 build alone (NEON / wasm / nightly / feature-gated
files are not compiled there). Several macros (`simd_avx512.rs:50`, `:61`) cover
many expansions with one comment.

## Soundness leads (160 flags, 63 of them `feature:`)

A flag is a lead, not a verdict. The first four were re-read at source and hold.

**Confirmed: out-of-bounds reachable from safe code**
- `simd_avx2.rs:406` `dot_i8`: loop sized by `a.len()`, `b[base..]` checks only
  the start, `_mm256_loadu_si256` reads 32 bytes → read past `b`'s end when
  `b.len() < a.len()`. Also no AVX2 check in a safe `pub fn`.
- `hpc/blocked_grid/iter.rs:306` `GridBlockMut::row_mut`: both bounds checks are
  `debug_assert!`; its SAFETY comment cites "the bounds check above".
- `backend/kernels_avx512.rs:665/:694` `sgemm_blocked` (and `dgemm_blocked`
  `:783/:861`, `:1006`): `&mut c[ir*ldc..]` is start-checked only, then a full
  16-lane store; the stated safety contract covers AVX-512F only, not `c`'s length.
- `hpc/int8_tile_gemm.rs:358` `int8_gemm_amx_tiled`: lengths are asserted, but
  `amx_available()` and the m/n/k multiple-of-16/64 checks are `debug_assert!`;
  the reviewer reports OOB tile access at `:487`, `:538` when misaligned.

**Reported, not yet re-read**
- `backend/native.rs:219` passes raw pointers to `matrixmultiply` with no
  extent check against m/n/k/ld*.
- Lab-only (32 rows tagged `[lab-only]`): MKL / OpenBLAS are a comparison
  harness, not the production path, which is the native Rust BlasGraph GEMM.
  Their safe wrappers pass raw pointers unchecked and truncate dims with
  `as c_int` (`backend/mkl.rs:199/223/244/263`, `openblas.rs:96/120`); stride-0
  row views give `ld < cols` → `xerbla` (`mkl.rs:407/452/504/553`). Real, low
  priority.
- `simd_neon.rs:115/147/214` codebook gathers: start-only checks, output length
  `debug_assert!` only. `simd_wasm.rs:1416/1477`: same pattern.
- `simd_avx512.rs:3477/3700`: private fns rely on callers' asserts.
- `hpc/gguf.rs:229`: `Vec<u8>` pointer cast to 2-aligned BF16;
  `jitson/scan_config.rs:138` `from_raw_parts::<f32>` on arbitrary byte pointers.
- `hpc/blackboard.rs:295`, `blocked_grid/iter.rs:171`: raw-pointer `&mut`
  re-borrows invalid under Stacked Borrows.
- JIT kernels (`jitson_cranelift/noise_jit.rs:79`, `scan_jit.rs:49`) are not
  lifetime-tied to the engine that owns their code memory.
- Upstream: `linalg/impl_linalg.rs:316` `set_len` on uninitialised elements
  (`Array::uninit` + `assume_init` at `:368` avoids it);
  `impl_methods.rs:3301` reads with only a `size_of` check;
  `iterators/mod.rs:1459/1484` trusts `TrustedIterator` exactness (upstream FIXME).
- Correctness, not UB: `simd_amx.rs:367` AVX-512 VNNI dot drops the `n % 64`
  tail, so `matvec_dispatch` differs from the vnni2/scalar arms;
  `simd_neon.rs:90` `vabdq_s16` wraps above 32767; `simd_neon.rs:3126/3138`
  shift counts use only the low byte.

**ISA-gating (`feature:`)**
- AVX-512BW intrinsics (`I8x64`, `I16x32`, …) live behind an `avx512f`-only gate
  (`simd.rs`). Practical exposure: Knights Landing-class only.
- `simd_avx2` is compiled on every x86_64 build and its safe `pub fn`s use AVX2
  with no check. A pre-AVX2 x86_64 build reaching them SIGILLs. `cpu_guard`
  covers mismatched BUILDS, not this (the code is compiled into a baseline build).
- `simd_runtime/cpu_ops.rs:74`: the AMX rung checks only `amx_int8` + the OS
  grant but installs kernels needing `avx512f` + `avx512vnni`;
  `simd_int_ops.rs:315/320` check only the VNNI bit. AMX-BF16 (`TDPBF16PS`)
  never checks its own CPUID bit.

## Toolchain-gated candidates (need the floor raised to 1.99)

- `Vec::into_parts` / `Vec::from_parts` / `Box::into_non_null` are stable on
  1.99.0 and unstable on 1.98.1 (E0658 `box_vec_non_null`, probed). In
  `data_repr.rs`, `OwnedRepr::from` becomes `let (ptr, len, capacity) =
  v.into_parts();` with no `ManuallyDrop`/`nonnull_from_vec_data`;
  `take_as_vec` stays `unsafe` (`from_parts` is unsafe by contract).
- `u128` as an `xmm_reg` `asm!` operand (1.99.0; E0658 on 1.98.1, probed):
  for the hand-written asm kernels, see the `amx-savant` card.

## Suggested order

1. Fix the confirmed safe-code OOB paths (assert lengths, or make the fns
   `unsafe fn` with a real `# Safety`). Small and local.
2. Complete the runtime-check feature sets (`cpu_ops.rs:74`, `simd_int_ops.rs`,
   AMX-BF16).
3. `CONSOLIDATE` through `from_array`/`to_array`: shrinks the pointer-op count
   per type to two.
4. SAFETY comments, macro-first.
5. `SAFE-API` swaps; `SAFE-CRATE` only with approval.
