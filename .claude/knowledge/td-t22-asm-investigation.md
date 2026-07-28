# TD-T22 — the AVX2 int-polyfill investigation (CLOSED, no gap)

## READ BY:
- Anyone about to "lower the AVX2 scalar polyfills to native `__m256i`"
- `simd-savant`, `truth-architect` — before accepting a SIMD-lowering proposal
- Any session reading a `🟠 scalar polyfill` cell in the parity matrix

## P0 TRIGGER
About to hand-write `__m256i` wrappers for `U32x8` / `U32x16` / `U16x32` /
`U64x4` / `I32x8` / `I64x4` because the parity matrix says "scalar polyfill"?
**Read this first. The matrix marks a SOURCE-level property, not a codegen
one, and the codegen gap does not exist.**

---

## The question TD-T22 actually asked

`td-simd-tier-audit.md:339` — `TD-T22 | – | – | Investigation only — needs
simd_avx2.rs read first`. `CHACHA20_MATRYOSHKA_PLAN.md:104` — *"avx2 native
`U32x16` — the TD-SIMD-3 lowering; **still optional**, the scalar polyfill is
correct meanwhile."*

The ticket asked whether the `avx2_int_type!` scalar-storage polyfills
actually cost anything. It did not ask for a lowering. This note answers it.

## Answer

**No gap. The "scalar polyfill" is not scalar in the emitted binary.**
`.cargo/config.toml` pins `-Ctarget-cpu=x86-64-v3` (AVX2+FMA) for **every**
x86_64 build including CI, so LLVM auto-vectorizes the macro's
`for i in 0..N { … }` bodies into packed AVX2. Measured below: **zero scalar
ALU instructions**, and the ARX loop hits the theoretical instruction minimum.

## Method

Branch at `1ff8171`. `U32x16` is `avx2_int_type!(U32x16, u32, 16, 0u32)`
(`simd_avx2.rs:1542`) — macro-generated `[u32; 16]` storage, confirmed: no
hand-written `pub struct U32x16` exists in that file.

```rust
// examples/td_t22_probe.rs — #[inline(never)] so each survives as a symbol
use ndarray::simd::U32x16;

#[inline(never)]
pub fn arx_quarter_round(a: U32x16, b: U32x16) -> U32x16 {
    let s = a + b;
    let x = s ^ b;
    x.rotate_left(16)
}

#[inline(never)]
pub fn arx_ten_rounds(mut st: [U32x16; 4]) -> [U32x16; 4] {
    for _ in 0..10 {
        st[0] = st[0] + st[1];
        st[3] = (st[3] ^ st[0]).rotate_left(16);
        st[2] = st[2] + st[3];
        st[1] = (st[1] ^ st[2]).rotate_left(12);
        st[0] = st[0] + st[1];
        st[3] = (st[3] ^ st[0]).rotate_left(8);
        st[2] = st[2] + st[3];
        st[1] = (st[1] ^ st[2]).rotate_left(7);
    }
    st
}

#[inline(never)]
pub fn red_sum(a: U32x16) -> u32 { a.reduce_sum() }
```

```sh
cargo rustc --release --example td_t22_probe -p ndarray -- --emit asm -C debuginfo=0
# then histogram the instructions between each symbol's .type/@function label
# and its retq, in target/release/examples/td_t22_probe-*.s
```

## Measured output

**`arx_quarter_round`** — `(a+b)^b`, then `rotate_left(16)`:

| instr | count |
|---|---|
| `vpaddd` | 2 |
| `vpxor` | 2 |
| `vpshufb` | 2 |
| `vmovdqa` | 5 |
| **scalar arithmetic on lane data** | **0** |

16 u32 lanes = 2 ymm registers, so 2 packed ops per lane-op is exactly one
instruction per half. LLVM strength-reduced `rotate_left(16)` into a **byte
shuffle** (`vpshufb`) — cheaper than the `shl|shr|or` triple a hand-written
intrinsic version would emit.

**`arx_ten_rounds`** — the full ChaCha20 double-round over `[U32x16; 4]`:

| instr | count |
|---|---|
| `vpaddd` | 8 |
| `vpxor` | 8 |
| `vpshufb` | 6 |
| `vpsrld` / `vpslld` / `vpor` | 4 / 4 / 4 |
| `jne` | 1 |
| **scalar arithmetic on lane data** | **0** |

The `jne` shows this is one rolled loop body, so the counts are per
iteration. **This is the instruction-count floor:** 4 `U32x16` adds = 64 u32
lanes; on 256-bit AVX2 that is 8 ymm adds minimum, and exactly 8 `vpaddd`
were emitted. Rotates by 16 and 8 fold to `vpshufb` (byte-granular); rotates
by 12 and 7 use the `vpsrld`/`vpslld`/`vpor` triple. **There is no headroom
for a hand-written version to recover.**

**`red_sum`** — `reduce_sum`:

| instr | count |
|---|---|
| `vpaddd` | 4 |
| `vpshufd` | 2 |
| `vextracti128` | 1 |
| `vmovd` | 1 |
| **scalar arithmetic on lane data** | **0** |

A textbook logarithmic horizontal-reduction tree, not the scalar
`wrapping_add` fold the source literally spells out.

**Precision on "0 scalar arithmetic on lane data".** A broad sweep of ALL
non-vector instructions across the three symbols returns exactly:
`3 retq`, `1 movl`, `1 jne`, `1 decl`. The `movl`/`decl`/`jne` are the
`arx_ten_rounds` loop counter and branch — loop control, not lane
arithmetic. So the honest claim is **no scalar op touches lane data**, not
"no scalar instruction exists": one `decl` does, and it decrements the trip
count. Every `u32` lane operation in all three probes is a packed AVX2
instruction.

## The same result holds for the float side

`F32x16::mul_add` (`simd_avx2.rs`) is likewise written as
`to_array()` → scalar loop → `from_array()`. `add_mul_f32` built on it emits
**`vfmadd213ps`** — real fused multiply-add, one rounding, mantissa
preserved. The doc comment at `simd_ops.rs:132` claiming "AVX2 + FMA:
`_mm256_fmadd_ps`" is correct about the *emitted code* despite the scalar
source. `array_chunks` / `array_windows` (+ `_checked`) already exist in
`simd_ops.rs` as the slice-level primitives.

## Consequences

1. **TD-T22 is CLOSED as "no gap".** The `🟠 scalar polyfill` cells in the
   parity matrix and the `⏳` in `agnostic-surface-cpu-matrix.md` mark a
   **source-level** property. They are accurate as written and must not be
   read as a performance defect.
2. **Do not hand-lower these types for codegen reasons.** A lowering can
   only be justified by properties LLVM does *not* give you:
   - `#[repr(align(64))]` cacheline guarantee (the polyfill HAS it; a
     `#[repr(transparent)]` `__m256i` wrapper LOSES it — measured:
     `U32x8` size 64→32, `U32x16` align 64→32),
   - ABI shape across a non-inlined boundary,
   - independence from `opt-level` (auto-vectorization does not hold at
     `-O0`/`-O1`) and from LLVM version drift.
   Those are real but small, and none of them is a speed argument.
3. **`U32x8` must not become `U32x16`'s building block.** Operator ruling,
   2026-07-28: a half-width type composing the lane the substrate actually
   uses is an absolute no-go — it splits the lane vocabulary for no gain.
4. **The guard that was missing is a codegen oracle, not a parity harness.**
   x86_64 is the CI host, so `cargo test` already runs the AVX2 tier
   natively and `simd.rs`'s `u32x16_arx_ops_match_scalar` already gates every
   backend's ARX triple against scalar. What did not exist — and what would
   have answered this ticket in twenty minutes — is an **asm-diff / bench**
   check. Adding one to CI is the sanctioned follow-up.

## Provenance

Investigated 2026-07-28 after a lowering PR (closed unmerged, ndarray #261)
shipped ~700 lines of hand-written intrinsics on the premise that these
polyfills executed scalar code. They do not. The PR was closed on the
operator's `U32x8`-composition ruling; this investigation is the deliverable
the ticket originally asked for.
