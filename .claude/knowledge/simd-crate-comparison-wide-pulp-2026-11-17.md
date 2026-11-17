# ndarray::simd vs `wide` vs `pulp` — inventory, tech debt, ideas to steal (2026-11-17)

Sources: `Lokathor/wide` @ `1b4559c` (v1.7.1, 2026-11-02), `sarah-quinones/pulp` @
`601de0c` (v0.23.0, 2026-09-08), this tree. Raw fact maps (Haiku `substrate-mapper`,
six runs) and the name sets behind every count below are in the session scratchpad
(`maps/*.md`, `sets/*.txt`); every claim with a `file:line` here was Read by the
orchestrator. Benchmark: `AdaWorldAPI/mandelbrot-simd` branch
`claude/practical-newton-u8lir8` — measured, see § 6.

## 1. Shape of each crate

| | `wide` | `pulp` | `ndarray::simd` |
|---|---|---|---|
| Model | concrete lane types (`f64x4`, `u8x16`, …), operator overloading | one `Simd` trait with associated types (`S::f64s`, `S::m64s`, `S::c64s`); you write a generic fn over `S` | concrete lane types (`F64x8`, `U8x64`, …), operator overloading |
| Tier choice | compile time (`cfg(target_feature)` inside every method) | **run time**: `Arch::new()` tries V4 → V3 → scalar (`pulp/src/x86.rs:317-336`), `dispatch(WithSimd)` enters a `#[target_feature]` trampoline (`vectorize`) | compile time (`simd.rs` module routing), `cpu_guard` refuses a binary the CPU can't run; `simd_caps` detects but does not dispatch |
| Width | fixed by the type name you pick | native per tier: `f64s` = `f64x4` on V3, `f64x8` on V4 (`x86/v3.rs:350`, `x86/v4.rs:476`) | fixed by type name; on AVX2 a 512-bit type is two 256-bit halves or a `[T; N]` array |
| Mask type | the vector itself (all-ones lanes) | **tier-native**: `m64s` = `m64x4` vector on V3, `b8` bitmask on V4 (`x86/v3.rs:357`, `x86/v4.rs:483`) | `F64Mask8(u8)` bitmask **on every tier** (`simd_avx512.rs:507` `__mmask8`, `simd_avx2.rs:1145` plain `u8`) |
| Types | 30 types: 128/256/512-bit × f32 f64 i8–i64 u8–u64 | 16 associated types per tier, incl. complex `c32s` `c64s` | 22 structs on the AVX-512 arm (`simd_avx512.rs`), incl. `BF16x16/8`, `F16x16`; 13 AVX2 integer types array-backed |
| Backends | SSE2/SSE4.1/AVX/AVX2/AVX-512 (f, vl, bw, ifma), NEON, wasm simd128, scalar | x86 V1–V4 (+ `V4_256`), aarch64 NEON (+ fcma), wasm (+ relaxed), aarch32, scalar | AVX-512, AVX2, AVX (no AVX2), v2→scalar, NEON (+dotprod, bf16, A72), wasm, wasm-scalar, AMX, nightly `core::simd` |
| Size | 357 distinct fn names (`wide/src/simd*.rs` + backends) | `Simd` trait 414 items incl. macro-generated (mapper, `lib.rs:608`), 1,042 backend pub fns | 175 type-method names + 248 in the op modules (160 in `simd_masking_ops.rs`) |

## 1b. ISA and architecture coverage

Counted from every `target_arch = "…"`, `target_feature = "…"` / `enable = "…"`,
`is_*_feature_detected!("…")` and pulp `f!("…")` string in each crate's `src/`.

| | `wide` | `pulp` | `ndarray` |
|---|---|---|---|
| x86 SSE2…SSE4.2, AVX, AVX2, FMA | yes | yes (V1, V2, V3) | yes; AVX-without-AVX2 has its own arm (`simd_avx.rs`), v2 is routed to scalar |
| AVX-512 F/BW/DQ/VL | yes | yes (V4, plus `V4_256`) | yes (`simd_avx512.rs`) |
| AVX-512 extras | ifma, vbmi | cd, ifma, vbmi2, **fp16** | **bf16, vnni, vpopcntdq, vbmi, gfni**, avxvnni, avxvnniint8 |
| **AMX** (tile, int8, bf16) | none | none | **yes**: `simd_amx.rs`, `hpc/amx_matmul.rs`, `hpc/bf16_tile_gemm.rs` (30 files mention it) |
| AVX10 | none | none | detection only (`cpu_guard.rs:663`) |
| aarch64 NEON | yes | yes (`Neon`, `NeonFcma`) | yes, by tier: baseline, **dotprod**, **bf16** (stub, see § 2b), A72/A76 profiles, fp16 |
| aarch64 extras detected | none | fcma, dotprod, i8mm, rdm, frintts, aes/sha3/sm4 | dotprod, bf16, fp16, crc, sha2 |
| **SVE / SVE2** | none | detection tokens only (`core_arch/mod.rs:227-248`), no `Simd` backend | none |
| **wasm32 simd128** | yes | yes | yes (`simd_wasm.rs`, 3,043 lines) plus a **wasm-scalar** arm |
| wasm relaxed-simd | `swizzle_relaxed` only | **yes**, feature `relaxed-simd` (on by default) | not used; `simd_wasm.rs:37` documents why (`relaxed_madd` may or may not fuse) |
| aarch32 (ARMv7 NEON) | none | yes (`aarch32.rs`) | none |
| riscv / powerpc / mips / wasm64 | none | `core::arch` re-export under `nightly` only, no backend | scalar fallback (`simd.rs:249,480`) |
| LoongArch LSX/LASX | none | none | none |
| Portable `core::simd` | none | `nightly` feature for bytemuck | `nightly-simd` backend (validation only) |
| Runtime CPU detection | none | yes (`Arch::new`, raw-cpuid) | yes: `simd_caps.rs` (26 detects), `cpu_guard` refuses an unsupported binary |
| Checks that backends agree | — | — | `masking-parity` arms: native, nightly, wasm, wasm-scalar, neon-qemu, avx-qemu, v2-qemu |

Coverage verdict: ndarray is widest on x86 server features (AMX, BF16, VNNI,
VPOPCNTDQ, GFNI) and is the only one that checks its backends agree. pulp is
widest across architectures (aarch32, relaxed-simd, fp16, i8mm detection). Nobody has SVE,
RISC-V V or LoongArch backends. Two pulp ideas are cheap to take here: AVX-512 FP16 (the host
this was written on has `avx512_fp16`) and wasm relaxed-simd behind a feature for the
non-FMA ops (min/max, dot, swizzle) where the result is deterministic.

## 2b. bf16 comparison

| | `wide` | `pulp` | `ndarray` |
|---|---|---|---|
| bf16 vector type | none | none (`Avx512bf16` and aarch64 `Bf16` are feature tokens, `core_arch/x86/mod.rs:363`) | `BF16x16(__m256bh)`, `BF16x8(__m128bh)` (`simd_avx512.rs:4135,4169`) |
| bf16 → f32 | do it yourself (widen u16, shift 16) | do it yourself | hardware `vcvtneebf162ps` + batch helpers; scalar shift on other tiers |
| f32 → bf16, round-to-nearest-even | none | none | `_mm512_cvtneps_pbh` + a software RNE certified byte-identical (`simd_avx512.rs:4378-4481`, `examples/bf16_rne_exhaustive.rs`) |
| bf16 dot / GEMM | none | none | `VDPBF16PS` and AMX `TDPBF16PS` paths (`hpc/bf16_tile_gemm.rs`, `hpc/amx_matmul.rs`, `simd_runtime/matmul.rs`) |
| aarch64 BFDOT | none | token only | **stub**: `BF16x8Stub` / `BF16x16Stub` (`simd_neon_bf16.rs:149,156`) |

bf16 tech debt (ndarray):
- **Two different `BF16x16` types.** `simd_avx512::BF16x16(__m256bh)` is hardware;
  `simd_half::BF16x16([u16; 16])` is scalar ("SIMD-accelerated fast paths … are a
  follow-up", `simd_half.rs:1-7`). The same name means two costs depending on the import.
- **`simd_half` arithmetic is upcast-per-op scalar.** `add_bf16_inplace` and
  `mul_bf16_inplace` (`simd_half.rs:245,266`) don't use the hardware conversions that
  exist one file over.
- **NEON bf16 is stubs** while `simd_caps` already detects `bf16`.

bf16 has no competitor: neither crate has a bf16 type, so "ndarray vs wide/pulp" here
means "ndarray vs what a wide/pulp user hand-writes" (widen u16→u32, shift, reinterpret).

AVX-512 bf16 benchmark, prepared and not run (cargo was denied in this session, in this
checkout too). Arms per 1 Mi elements: (a) hand-written widen+shift on `U16x32`/`U32x16` (the
wide/pulp equivalent), (b) `BF16x16::to_f32x16` (`vcvtneebf162ps`), (c) `f32_to_bf16_batch_rne`
software vs `_mm512_cvtneps_pbh`, (d) bf16 dot: decode-then-FMA vs `VDPBF16PS` vs AMX
`TDPBF16PS` 16×16 tile. Pinned `--config .cargo/config-avx512.toml` (Sapphire Rapids; the
host has `avx512_bf16 amx_bf16 amx_tile`).
Predictions: P5 (b) ≥ 1.5× (a); P6 (c) software RNE within 2× of hardware; P7 (d)
VDPBF16PS ≥ 2× decode-then-FMA; AMX ≥ 4× VDPBF16PS at K ≥ 512.

## 2. Feature inventory (concept → who has it)

Counts are distinct fn names matching the concept in each crate's name set
(lower bounds for pulp: much of its trait is macro-generated).

| Concept | wide | pulp | ndarray | Note |
|---|---|---|---|---|
| Transcendentals (sin/cos/exp/ln/pow/atan2/tanh/cbrt) | **40**, all float types | ~0 | `simd_exp_f32`, `simd_ln_f32` only (f32) | `hpc/vml.rs` exists but is not part of the type API |
| floor/ceil/trunc/round/round_ties_even | 28 | 12 (backend) | `floor`, `round` | |
| recip / recip_sqrt | yes | yes (backend) | none | |
| is_nan / is_finite / is_inf / copysign / flip_signs | yes | partial | none | |
| Mask any/all/none | `any`, `all`, `none` | `first_true_m*s` | only `to_bitmask() == 0` | |
| Masked / partial load-store (tails) | none (`from_slice_unaligned`) | **36**: `partial_load_*`, `partial_store_*`, `mask_load_ptr_*`, `mask_store_ptr_*`, `MemMask` | `mask_store` on one type; tail logic per caller | |
| Saturating / overflowing / wrapping ints | **43** incl. mixed signed/unsigned | wrapping only | 7 (`saturating_add/sub/abs`, `narrow_saturating_*`) | |
| Widening mul / mul-high | `widening_mul`, `mul_keep_high`, `mul_widen` | `widening_mul_u32s` | none at type level | `mul_lo32`, `mullo` only |
| Interleave / transpose | 7 | 4 (shfl) | 5 (`interleave_*_u32/u64`, `transpose8`) | |
| Gather | none | backend only | `gather`, `gather_u16`, codebook gathers | |
| Complex c32/c64 (mul_add, conj, abs2, swap_re_im) | none | **40** | none | relevant to `hpc/fft.rs` |
| Fast vs correct variants (`fast_min`, `fast_round_int`, …) | yes, explicit | `mul_add_e` (exact) vs `mul_add` | no split | |
| f16 / bf16 | none | fp16 intrinsics in `core_arch` | **full**: RNE conversions, `F16x16`, `BF16x16/8`, Kahan f16 dot | ndarray-only |
| Bit tricks: ternlog, popcount, hamming, nibble ops, GF2 affine | none | none | **yes** | ndarray-only |
| Int8 dot (VNNI `dot4_u8_i8`), fixed-point (`saturating_rounding_doubling_high_mul`) | none | none | yes | ndarray-only |
| AMX tile GEMM | none | none | yes (`simd_amx.rs`) | ndarray-only |
| Masking-op library (`*_to_mask`, `mask_*`, `masked_*`) | none | none | 160 fns | ndarray-only |
| Runtime dispatch, one binary for v3+v4 | no | **yes**, whole kernel: `Arch::dispatch(WithSimd)` enters a `#[target_feature]` trampoline once | **yes, opt-in**: `src/simd_runtime/` (feature `runtime-dispatch`), one `LazyLock` fn pointer **per op** (`vnni_dot`, `matmul`, `casts`, `add_mul`) | ⊘ corrected: the first revision said "no"; no mapper unit covered `simd_runtime/` |
| serde / bytemuck Pod | both | bytemuck | none found in `simd.rs` | |

## 3. ndarray tech debt found by this comparison

Ordered by how directly it costs speed in real loops.

1. **AVX2 `F64x8` compares are scalar loops.** `simd_le` and its siblings convert to arrays
   and compare lane by lane (`simd_avx2.rs:982`, the f32 twin at `:644`), and
   `F64Mask8::select` does a per-lane `if` (`simd_avx2.rs:1153`). wide does the same op
   with one `_mm256_cmp_pd` + `_mm256_blendv_pd` (`wide/src/f64x4_backend.rs:79,107`).
   This sits in the Mandelbrot inner loop. It is also an R2 facet-stride violation.
2. **The root cause is the mask type, not the compare.** `F64Mask8(u8)` is AVX-512's
   shape. On AVX2 the natural result of `vcmppd` is a vector mask, so a `u8` mask
   forces movemask out and per-lane expansion back in for every `select`. pulp avoids this by
   making the mask an associated type that differs per tier; wide avoids it by making
   the mask the vector type itself.
3. **`F64x4` has no compare, mask or select on any stable arm.** The AVX-512 arm's
   `impl F64x4` (`simd_avx512.rs:3078`) has load/store, `reduce_sum`, `abs`, `mul_add`
   and arithmetic only. The AVX2 arm reuses that type (`simd_avx2.rs:9`). `F64Mask4` is
   exported by `simd.rs:263` but defined only in `simd_nightly/masks.rs:148`. So the
   **backends offer different APIs**: code using `F64x4::simd_le` compiles on
   `nightly-simd` and on no stable arm. Fixing this also fixes item 1, because
   AVX2 `F64x8` is two `F64x4` halves.
4. **13 AVX2 integer types are `[T; N]` arrays with scalar ops** (`avx2_int_type!`,
   `simd_avx2.rs:1332-1705, 2439-2441`): `U8x64 I8x64 I16x32 I32x16 I64x8 U16x32 U32x16
   U64x8 U32x8 U64x4 I32x8 …`. This is already in the burn-down list (81 of its R2 hits); the
   comparison confirms that both wide and pulp realize the same widths as vector pairs.
5. **No mask predicates** (`any`/`all`/`none`): every caller writes `to_bitmask() == 0`.
6. **No float classification or reciprocal**: `is_nan`, `is_finite`, `copysign`, `recip`,
   `recip_sqrt` are missing everywhere.
7. **Transcendentals only for f32 exp/ln.** wide ships sin_cos/exp/exp2/ln/log2/pow/atan2/
   tanh/cbrt on f32 and f64 at every width.
8. **No generic tail handling.** pulp's `partial_load_*`/`partial_store_*` and
   `MemMask` give every type a masked head/tail; ndarray callers hand-roll remainders.
9. **No widening multiply or mul-high on the integer types.**
10. **No 128-bit float types on the x86 facade** (`F32x4`/`F64x2` are not exported by
    `simd.rs`; only the wasm path mentions f32x4 lanes, `simd.rs:198`).

## 4. Ideas to steal

| Idea | From | Why it fits ndarray | Cost |
|---|---|---|---|
| **Tier-native mask type**: AVX2 `F64Mask8` = two `__m256d` lane masks, AVX-512 = `__mmask8`; `to_bitmask()` stays as the common exit | pulp (`m64s`), wide | Removes items 1–2 at the root; R1/R2-clean (one `vcmppd`, one `vblendvpd`, no shuffles) | Medium: mask API is public; `to_bitmask`/`select` signatures stay |
| **Complete `F64x4`/`F32x8` compare + mask + min/max on every stable arm** | wide | Closes the gap between backends (item 3), and AVX2 512-bit types then compose from halves | Small |
| **`any()` / `all()` / `none()` on every mask** | wide | Readability; one `vmovmskpd`/`kortest` | Trivial |
| **`partial_load` / `partial_store` / `MemMask` per type** | pulp | Uniform tails using AVX-512 `k`-masked loads and AVX2 `vmaskmov`; most hand-rolled remainders go away | Medium |
| **Explicit fast/correct split** (`fast_min` vs `min`, `mul_add_e` vs `mul_add`) | wide, pulp | Matches the W1a contract's "document saturating/overflow semantics"; makes NaN/FMA choices visible at the call site | Small per op |
| **Integer semantics breadth**: `overflowing_*`, `saturating_*` incl. mixed-sign, `widening_mul`, `mul_keep_high` | wide | The W1a contract already requires documented overflow semantics; wide's names are a ready vocabulary | Medium |
| **f32/f64 transcendentals on the types** (sin_cos, exp, ln, pow, atan2) | wide | `hpc/vml.rs` and activations could call type methods instead of duplicating polynomial code | Medium |
| **Complex `c32s`/`c64s` ops** (mul_add, conj_mul, abs2, swap_re_im) | pulp | `hpc/fft.rs` (Cooley-Tukey) is the consumer | Medium |
| **Kernel-granular dispatch** (`WithSimd` + `#[target_feature]` trampoline) | pulp | ndarray already ships opt-in runtime dispatch (`simd_runtime/`), but per op: each call pays an indirect call (~2-3 ns, its own doc) and the op cannot inline into the caller's loop. pulp dispatches once per kernel and the whole loop body is compiled for the tier. Worth taking for the `runtime-dispatch` layer: a `with_tier(\|t\| ...)` entry that monomorphizes the closure per tier | Medium |

What **not** to copy: wide's general `shuffle`/`swizzle`/`swizzle_half` family and
pulp's `interleave_shfl`/`deinterleave_shfl` used as compute. Both are banned in this tree by the
FACET STRIDE rule R1. ndarray is ahead on quantized/bit-level work (VNNI, ternlog,
popcount, AMX, f16/bf16 RNE, the masking library) and on the number of backends it
covers and checks for parity against each other.

## 5. Mandelbrot benchmark — plan and pre-registered predictions (results in § 6)

Fork: `AdaWorldAPI/mandelbrot-simd`, branch `claude/practical-newton-u8lir8` (local,
uncommitted). Changes: stable toolchain 1.99.0; `std::simd` arm behind a `nightly`
feature; `wide` 0.7.28 → 1.7.1; `pulp` 0.20 → 0.23 with `x86-v4` (its default only has
v3, so without it pulp silently runs AVX2 on an AVX-512 host); new arms `wide8`
(f64x8) and `ndarray` (`ndarray::simd::F64x8`, `src/ndarray_par.rs`); in-process timer
`src/bin/bench.rs` that checks each arm against scalar before timing and prints the tier
(`avx512f=`, `avx2=`).

Blocked: `cargo build` in that checkout was denied by the session's permission
classifier (three attempts), and so was `cargo install hyperfine`.

Commands, one per tier (separate target dirs so the realizations don't overwrite each other):

```sh
cd mandelbrot-simd
RUSTFLAGS="-C target-cpu=x86-64-v4" CARGO_TARGET_DIR=target/v4 cargo run --release --bin bench -- 3200 3200 15 scalar,autovect,wide,wide8,pulp,ndarray
RUSTFLAGS="-C target-cpu=x86-64-v3" CARGO_TARGET_DIR=target/v3 cargo run --release --bin bench -- 3200 3200 15 scalar,autovect,wide,wide8,pulp,ndarray
RAYON_NUM_THREADS=1 ...   # same, single thread
```

Pre-registered predictions (from the code read in § 3; to be graded, not assumed):

- P1 v4: `ndarray` within 15% of `pulp` and `wide8`. All three lower to
  `vcmppd k` + masked blend on zmm.
- P2 v4: `wide` (f64x4) slower than `wide8` (256-bit vs 512-bit).
- P3 v3: `ndarray` slower than `wide` and `pulp`, by at least 2×: its compare and select
  are scalar loops (TD 1). `pulp` on v3 picks V3 at run time (`f64s = f64x4`).
- P4 v3: under the pinned v3 build pulp's `Arch::new()` still finds AVX-512 at run time
  and dispatches V4, so `pulp` should be roughly the same as on the v4 build. A
  run-time dispatcher ignores the compile-time pin.
- Kill condition for TD 1's priority: if P3 fails (ndarray within 20% of wide on v3),
  LLVM is re-vectorizing the scalar loop and item 1 drops in priority.

Reverse-engineering step after the run: `objdump -d --no-show-raw-insn` the
`get_count` of each arm per tier, then count `vcmppd`, `vblendvpd`/`vblendmpd`,
`kortest`/`vmovmskpd`, and scalar `vucomisd` in the loop body.

## 6. Mandelbrot benchmark — MEASURED 2026-11-17 (predictions graded)

Fork `AdaWorldAPI/mandelbrot-simd` @ `ccf4b81` (wide 1.7.1, pulp 0.22.3 `x86-v4`, ndarray master `09cdaa1`).
3200×3200, 50 iterations, in-process timer, median of 15 (min in brackets), ms. Host: Xeon with AVX-512,
4 cores. Every arm is first checked equal to the scalar output. Tier is from the binary's own header.

| arm | v4, 1 thread | v4, 4 threads | v3, 1 thread | v3, 4 threads |
|---|---|---|---|---|
| scalar | 639 (581) | 172 (161) | 645 (594) | 179 (157) |
| autovect | 355 (325) | 104 (90) | 348 (278) | 102 (93) |
| wide `f64x4` | 231 (206) | 58 (52) | 214 (196) | 56 (52) |
| wide `f64x8` (wide8) | 156 (151) | 44 (40) | 152 (139) | 39 (35) |
| pulp | 155 (131) | 42 (39) | 160 (150) | 41 (37) |
| **ndarray `F64x8`** | **170 (163)** | **47 (44)** | **199 (187)** | **57 (49)** |
| ndarray before PR #435 (`bf112d2`) | — | — | 229 (201) | 65 (51) |

Run-to-run noise on the control arms between the two v3 sessions was about 6% (wide8 152 vs 162).

Predictions (§ 5):
- **P1 holds.** v4: ndarray is 9% behind pulp and wide8 single-threaded (170 vs 155/156), 8–11% with 4 threads.
- **P2 holds.** v4: wide `f64x4` is 1.48× slower than wide8.
- **P3 FAILED, kill condition hit.** Before the PR, ndarray on v3 was 1.41–1.46× slower than wide8/pulp, not
  ≥ 2×. The pre-PR disassembly shows why: LLVM had already vectorized most of the "scalar" compare loop
  (`vcmplepd`, `vcmpnlepd`, `vblendvpd` present, with `vcmplesd` / `vucomisd` / 4× `sete` remnants). TD1 was
  real but much cheaper than I claimed.
- **P4 holds.** pulp is as fast on the v3 build as on v4 (160 vs 155): its code in the v3 binary contains
  `zmm` instructions (2 symbols), the wide and ndarray arms contain none.

Effect of F1/F2 (PR #435) on v3: 229 → 199 ms single-threaded (13%), 65 → 57 ms with 4 threads; hot loop
176 → 141 instructions, all scalar compares gone.

Why ndarray is still behind, from the disassembly of the hot closure:
- **v3 (1.24–1.30× behind):** the compare result is packed to the public bitmask and re-expanded for
  `select` — `vmovmskpd` ×2, then `vpbroadcastq` ×3 + `vpand` ×4 + `vcmpneqpd` ×2. wide8 keeps the lane mask
  in the register (`vcmple_oqpd` + `vandpd`). This is the cost of `F64Mask8(pub u8)` staying as it is.
- **v4 (≈ 9% behind):** compare and select are single mask-register ops, same as the others. The difference
  seen in the loop is the count conversion: the arm converts 8 lanes one at a time (`vcvttsd2si` ×8 with
  `vminsd`/`vmaxsd` saturation) because the facade has no vector `F64x8` → integer conversion; wide uses
  `round_int`, pulp counts in `u64` lanes with `select_u64s`. Not measured in isolation.
- wide's AVX2 `any()` lowers to 8–16 scalar `vucomisd` + `cmov` here, yet wide8 is still the fastest v3 arm.

## 7. Before / after PR #435, method by method — llvm-objdump + timing (2026-11-17)

Probe: one `#[inline(never)] #[no_mangle]` function per facade method, built against ndarray `bf112d2`
(before) and `09cdaa1` (after) at `-C target-cpu=x86-64-v3` and `-v4`, opt-level 3, rustc 1.99.0,
disassembled with `llvm-objdump -d`. Timing: 2,000,000 calls, best of 7 inside a run, best of 5 runs
interleaved before/after; every output is passed through `black_box` (a first harness without that
reported 0.3 ns for the out-pointer probes — the calls were being optimized away; those numbers were
discarded). The result fingerprint is identical in all four binaries (`0x4ffeb11112ae9530`).

```
== x86-64-v3: instructions (total / scalar-FP / branches) and ns per call (best of 5 runs)
method                    insns before   insns after  ns before  ns after   change
p_f64x8_simd_le              9/ 0/ 0      12/ 0/ 0        1.81      1.85    0.98x
p_f64x8_simd_lt              9/ 0/ 0      12/ 0/ 0        1.71      1.87    0.91x  SLOWER
p_f64x8_simd_eq              9/ 0/ 0      12/ 0/ 0        1.81      1.85    0.98x
p_f64x8_simd_ne              9/ 0/ 0      12/ 0/ 0        1.84      1.74    1.06x
p_f64x8_le_select           22/ 1/ 0      26/ 0/ 0        3.29      3.17    1.04x
p_f64x8_simd_min            14/ 0/ 0      14/ 0/ 0        2.23      2.06    1.08x  FASTER
p_f64x8_simd_max            14/ 0/ 0      14/ 0/ 0        2.30      2.12    1.08x  FASTER
p_f64x8_sqrt                 6/ 0/ 0       6/ 0/ 0        6.90      6.71    1.03x
p_f64x8_round                6/ 0/ 0       6/ 0/ 0        1.44      1.40    1.03x
p_f64x8_floor                6/ 0/ 0       6/ 0/ 0        1.41      1.27    1.11x  FASTER
p_f64x8_abs                  7/ 0/ 0       7/ 0/ 0        1.21      1.12    1.08x  FASTER
p_f64x8_reduce_min          21/10/ 0      27/ 4/ 0        2.15      4.03    0.53x  SLOWER
p_f64x8_reduce_max          21/10/ 0      27/ 4/ 0        2.11      4.00    0.53x  SLOWER
p_f32x16_simd_le            11/ 0/ 0      14/ 0/ 0        1.82      1.74    1.05x
p_f32x16_simd_lt            11/ 0/ 0      14/ 0/ 0        1.58      1.73    0.91x  SLOWER
p_f32x16_simd_eq            11/ 0/ 0      14/ 0/ 0        1.76      1.83    0.96x
p_f32x16_simd_ne            11/ 0/ 0      14/ 0/ 0        1.81      1.83    0.99x
p_f32x16_le_select          80/ 1/ 0      26/ 0/ 0        7.91      3.32    2.38x  FASTER
p_f32x16_simd_min           14/ 0/ 0      14/ 0/ 0        2.29      2.21    1.04x
p_f32x16_simd_max           14/ 0/ 0      14/ 0/ 0        2.27      2.22    1.02x
p_f32x16_sqrt                6/ 0/ 0       6/ 0/ 0        3.54      3.42    1.04x
p_f32x16_round               6/ 0/ 0       6/ 0/ 0        1.46      1.42    1.03x
p_f32x16_floor               6/ 0/ 0       6/ 0/ 0        1.47      1.40    1.05x
p_f32x16_reduce_min         53/26/ 0      35/ 4/ 0        9.49      5.81    1.63x  FASTER
p_f32x16_reduce_max         53/26/ 0      35/ 4/ 0        9.37      5.75    1.63x  FASTER
p_u8x64_saturating_add       8/ 0/ 0       8/ 0/ 0        1.64      1.57    1.04x
p_u8x64_saturating_sub       8/ 0/ 0       8/ 0/ 0        1.62      1.57    1.03x
p_u8x64_mask_blend         377/ 0/65      30/ 0/ 0       28.78      2.89    9.96x  FASTER
c_f64x8_add                  8/ 0/ 0       8/ 0/ 0        1.58      1.36    1.16x  FASTER
c_f64x8_mul_add             10/ 0/ 0      10/ 0/ 0        1.76      1.61    1.09x  FASTER

== x86-64-v4: instructions (total / scalar-FP / branches) and ns per call (best of 5 runs)
method                    insns before   insns after  ns before  ns after   change
p_f64x8_simd_le              5/ 0/ 0       5/ 0/ 0        1.76      1.67    1.05x
p_f64x8_simd_lt              5/ 0/ 0       5/ 0/ 0        1.61      1.62    0.99x
p_f64x8_simd_eq              5/ 0/ 0       5/ 0/ 0        1.61      1.60    1.01x
p_f64x8_simd_ne              5/ 0/ 0       5/ 0/ 0        1.59      1.41    1.13x  FASTER
p_f64x8_le_select            7/ 0/ 0       7/ 0/ 0        2.10      1.77    1.19x  FASTER
p_f64x8_simd_min             5/ 0/ 0       5/ 0/ 0        1.71      1.62    1.06x
p_f64x8_simd_max             5/ 0/ 0       5/ 0/ 0        1.70      1.76    0.97x
p_f64x8_sqrt                 4/ 0/ 0       4/ 0/ 0        7.36      7.24    1.02x
p_f64x8_round                4/ 0/ 0       4/ 0/ 0        1.49      1.51    0.99x
p_f64x8_floor                4/ 0/ 0       4/ 0/ 0        1.47      1.46    1.01x
p_f64x8_abs                  5/ 0/ 0       5/ 0/ 0        1.27      1.38    0.92x  SLOWER
p_f64x8_reduce_min           8/ 1/ 0       8/ 1/ 0        1.17      1.16    1.01x
p_f64x8_reduce_max           8/ 1/ 0       8/ 1/ 0        1.19      1.17    1.02x
p_f32x16_simd_le             5/ 0/ 0       5/ 0/ 0        1.37      1.34    1.02x
p_f32x16_simd_lt             5/ 0/ 0       5/ 0/ 0        1.38      1.39    0.99x
p_f32x16_simd_eq             5/ 0/ 0       5/ 0/ 0        1.53      1.36    1.12x  FASTER
p_f32x16_simd_ne             5/ 0/ 0       5/ 0/ 0        1.41      1.35    1.04x
p_f32x16_le_select           7/ 0/ 0       7/ 0/ 0        1.92      1.78    1.08x  FASTER
p_f32x16_simd_min            5/ 0/ 0       5/ 0/ 0        1.68      1.48    1.14x  FASTER
p_f32x16_simd_max            5/ 0/ 0       5/ 0/ 0        1.65      1.48    1.11x  FASTER
p_f32x16_sqrt                4/ 0/ 0       4/ 0/ 0        3.53      3.34    1.06x
p_f32x16_round               4/ 0/ 0       4/ 0/ 0        1.47      1.31    1.12x  FASTER
p_f32x16_floor               4/ 0/ 0       4/ 0/ 0        1.48      1.32    1.12x  FASTER
p_f32x16_reduce_min         10/ 1/ 0      10/ 1/ 0        1.41      1.16    1.22x  FASTER
p_f32x16_reduce_max         10/ 1/ 0      10/ 1/ 0        1.38      1.16    1.19x  FASTER
p_u8x64_saturating_add       5/ 0/ 0       5/ 0/ 0        1.63      1.40    1.16x  FASTER
p_u8x64_saturating_sub       5/ 0/ 0       5/ 0/ 0        1.65      1.40    1.18x  FASTER
p_u8x64_mask_blend           6/ 0/ 0       6/ 0/ 0        1.54      1.29    1.19x  FASTER
c_f64x8_add                  5/ 0/ 0       5/ 0/ 0        1.62      1.40    1.16x  FASTER
c_f64x8_mul_add              6/ 0/ 0       6/ 0/ 0        1.71      1.42    1.20x  FASTER
```

Reading it:
- **AVX-512 (v4): the PR changed no generated code.** All 30 probe symbols disassemble identically before
  and after. The v4 timing column is therefore pure noise, and it reaches ±20% on this host (see the
  untouched controls `c_f64x8_add` 1.16×, `c_f64x8_mul_add` 1.20×). Only v3 changes beyond that band count.
- **AVX2 (v3): LLVM had already vectorized the "scalar lane loops".** Before the PR, `F64x8::simd_le` was 9
  instructions with zero scalar FP ops (`vcmplepd` ×2, `vpackssdw`, `vpermq`, `vmovmskps`). sqrt / round /
  floor / abs and the U8x64 saturating ops are byte-identical before and after; min / max have the same
  instruction count. The hand-written halves bought nothing there.
- **The compares got 3 instructions longer** (12 vs 9, 14 vs 11): going through `F64Mask4::to_bitmask` adds
  `vpxor` + `vpcmpgtq` ×2, because LLVM re-derives the sign bits it can no longer see are a compare result.
  Timing is inside the noise band.
- **REGRESSION introduced by the PR: `F64x8::reduce_min` / `reduce_max` on AVX2 are 1.9× slower**
  (2.15 → 4.03 ns; 21 → 27 instructions). The NaN-ignoring tree (`vminpd` + `vcmpunordpd` + `vblendvpd` per
  level) is longer than what LLVM made of master's `fold(INFINITY, f64::min)` (a `vminsd` chain with two
  NaN fix-ups).
- **Real gains on AVX2:** `U8x64::mask_blend` 10.0× (377 instructions with 65 branches → 30, branch-free);
  `F32x16` compare + select 2.4× (80 → 26 instructions); `F32x16::reduce_min/max` 1.6× (53 → 35, scalar FP
  ops 26 → 4).

Net: on AVX2 the PR's measurable value is three things (mask_blend, f32 compare+select, f32 reductions) and
the new API; the rest of F2's rewrite is neutral, and the f64 reductions are worse. Lesson: read the
disassembly of the existing code BEFORE calling a lane loop "scalar" — the source shape is not the codegen.
