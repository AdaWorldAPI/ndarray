# The crypto lane — what is proven, what is missing

> **Status: MEASURED, 2026-07-28.** Every claim here is backed by an
> assembly probe or a grep across all six backends. No estimates.

## READ BY:
- Anyone building on `crates/encryption` (argon2 / BLAKE3 / ChaCha20 / AEAD)
- Anyone asked "do we need new SIMD work for cipher X?"

---

## The u32 ARX lane: PROVEN, at the instruction floor

`crate::simd::U32x16` Add / BitXor / `rotate_left` is the ChaCha20 and BLAKE3
mixing triple. TD-T22 measured it (`td-t22-asm-investigation.md`): the
scalar-storage polyfill compiles to **8 `vpaddd` for 64 u32 lanes — the AVX2
instruction floor** — with no scalar op touching lane data, and
`rotate_left(16)` strength-reduced to `vpshufb`, cheaper than the
`shl|shr|or` triple an intrinsic emits.

**Consequence: ChaCha20 and BLAKE3 need no new SIMD work.** The lane they
ride is already optimal on the default tier.

The float side is likewise done: `add_mul_f32` emits real `vfmadd213ps` —
one rounding, mantissa preserved — and `array_chunks` / `array_windows`
(+ `_checked`) already exist as the slice-level primitives in `simd_ops.rs`.

## The u64 ARX lane: DOES NOT EXIST

Measured across `simd_avx512`, `simd_avx2`, `simd_scalar`, `simd_neon`,
`simd_wasm`, and `simd_nightly`: **zero `rotate_left` or `rotate_right`
methods on `U64x8` or `U64x4`. On any backend.**

Whole-crate census of rotate methods — `grep -rhoE "fn rotate_(left|right)" src/`:

| method | count |
|---|---|
| `rotate_left` | 8 (all u32 lanes) |
| `rotate_right` | **0, at any width** |

*(Search validated by control: the same pattern finds `U32x16::rotate_left`
in all four backends that define it, so the empty u64 result is a true
negative and not a broken query.)*

Note the second row. BLAKE2b specifies **right** rotations. `rotr(n)` is
expressible as `rotl(64 - n)`, so this is a naming/API gap rather than a
mathematical one — but a caller writing BLAKE2b today has neither.

This matters because **BLAKE2b is a 64-bit ARX cipher**, and BLAKE2b is what
**argon2** uses. Its G-function is
`a+=b; d=(d^a).rotr(32); c+=d; b=(b^c).rotr(24); a+=b; d=(d^a).rotr(16); c+=d; b=(b^c).rotr(63)`
— four u64 rotates per mixing step, none of which the crate can express
today.

`crates/encryption` currently references exactly one SIMD type:
`simd::U32x16`.

**This is a real gap, unlike the u32 one.** The distinction is the whole
lesson of TD-T22: a missing *source-level* lowering is not a gap when LLVM
already emits the instruction; a missing *method* is a gap regardless of
what LLVM would do with it.

### ANSWERED by the oracle: no, it does not vectorize

Measured on x86_64 v3 via `.claude/knowledge/simd-codegen-oracle/`:

| probe | packed | scalar lane-arith | verdict |
|---|---|---|---|
| `rot_u64x8` | **0** | 8 | one scalar `rorq %cl` per lane |
| `rot_u64x4` | **0** | 4 | one scalar `rorq %cl` per lane |
| `blake2b_g_u64x8` | 22 | ~88 | leading add only |

`blake2b_g_u64x8` in detail: the opening `a = a + b` vectorizes (`vpaddq`
across both ymm halves). The moment a rotate is needed LLVM extracts every
lane to a GPR (`vmovq` / `vpextrq`) and stays scalar (`rorxq`/`addq`/`xorq`)
through all four rotate stages, going packed again only for the return
struct's reassembly.

Two things make this a genuine finding rather than a shrug:

1. **The byte-granular amounts stay scalar too.** Rotates by 32/24/16 are
   byte-aligned — the class LLVM folds to `vpshufb` for u32 — yet all four
   BLAKE2b amounts (32/24/16/63) lowered identically to scalar `rorxq`.
2. **The mechanism exists and is unused.** AVX2 has `vpsllq`/`vpsrlq`, and
   LLVM *does* use exactly that shift-or composition for u32's rotate-by-12
   and rotate-by-7. It has the tools and declines to apply them at 64-bit
   width.

**So the u64 ARX lane is the crate's first intrinsic override that meets the
entry criterion** (a probe proving the generic form fails). AVX-512:
`_mm512_rorv_epi64` / `VPROLVQ`, one instruction. AVX2 / NEON / wasm: write
the `vpsllq`/`vpsrlq`-shaped shift-or explicitly, since LLVM will not.

Contrast with the u32 lane, where hand-writing intrinsics *lost* to the
optimizer. Same crate, same week, opposite answers — which is the argument
for the oracle existing at all.

## Decision gates (operator, not engineering)

Neither of these is blocked on SIMD work:

1. **FIPS.** If any deployment needs FIPS-adjacent claims, BLAKE3 is out and
   SHA-384 stays the KDF hash. Settle before investing in a BLAKE3 lane.
2. **`x448` audit provenance.** Gates whether an X25519 port is worth it.
   The tripwire test already on master
   (`channel::tests::low_order_peer_keys_are_refused_and_honest_ones_are_not`)
   asserts both halves, so a mechanical port cannot silently drop RFC 7748's
   contributory check — `x448::x448()` returns `Option` where
   `x25519_dalek::x25519()` returns a bare `[u8; 32]`.

## Summary

| lane | status | blocks |
|---|---|---|
| u32 ARX (ChaCha20, BLAKE3) | **proven optimal** | nothing |
| f32 FMA (`add_mul`) | **proven fused** | nothing |
| slice chunking | **exists** | nothing |
| **u64 ARX (BLAKE2b → argon2)** | **absent on all 6 backends** | argon2 SIMD |
