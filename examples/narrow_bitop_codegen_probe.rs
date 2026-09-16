//! Does the facade's NARROW u64 type already descend to one instruction?
//!
//! On the x86 backends `U64x4` is a scalar-storage polyfill — `pub struct
//! U64x4(pub [u64; 4])` (`simd_avx2.rs`'s `avx2_int_type!`), with `BitAnd`
//! implemented as a 4-iteration loop. The mask-algebra tail descent needs a
//! real 256-bit `and`, and the obvious reading of that struct is that the
//! facade cannot supply one without being retyped to `__m256i`.
//!
//! That reading is an ASSUMPTION about codegen, and this probe is here to
//! refuse it. `[u64; 4]` is `repr(align(64))` and the loop is trivially
//! unrollable, so LLVM may already be emitting a single `vpand ymm` — in
//! which case the tail descent needs a NAME on the facade and nothing more,
//! and the multi-backend retype is unnecessary work.
//!
//! Read the emitted assembly, not this comment:
//!
//! ```sh
//! cargo rustc --release --example narrow_bitop_codegen_probe -- --emit asm
//! ```
//!
//! and grep the four probe symbols for `vpand` / `vandps` / `vpternlog`.
//!
//! # The aarch64 half, and it is the one that matters for the tail decision
//!
//! `simd_masking_ops.rs` does NOT zero-pad its tails for speed. Its header
//! (`:102-107`) records the reason as codegen UNIFORMITY: with an exact-length
//! scalar tail, LLVM "fully unrolled it on aarch64 into 7 x (and, orr) on
//! GPRs ... in a facade op whose contract is `packed on every backend`", while
//! the same loop on AVX2 became `vpmaskmovq`. Padding made both arms one
//! shape. The same header is explicit that "no throughput comparison against
//! the old peel has been made".
//!
//! So a tail proposal has to clear TWO bars, and speed is only the first.
//! `mask_algebra_tail_probe` supplies the missing throughput comparison; this
//! probe answers the codegen one, on the backend the objection was about:
//!
//! ```sh
//! rustup target add aarch64-unknown-linux-gnu   # asm needs no linker, no qemu
//! rustc --target aarch64-unknown-linux-gnu -O --emit asm probe.rs
//! ```
//!
//! Measured 2026-09-16, a FIXED-width peel on aarch64:
//!
//! ```text
//! probe_fixed4_and:  ldp q0, q3, [x1] / and v0.16b, v0.16b, v1.16b
//!                    and v1.16b, v3.16b, v2.16b / stp q0, q1, [x2]
//! probe_fixed2_and:  ldr q0, [x0] / and v0.16b, v1.16b, v0.16b / str q0, [x2]
//! ```
//!
//! Two NEON `and`s on 128-bit `v` registers, and one. **Packed** — not the
//! GPR unroll the header warns about. The distinction the header's measurement
//! could not draw is between an EXACT-LENGTH tail, whose trip count is a
//! runtime value, and a FIXED-WIDTH step, whose trip count is a constant.
//! Only the first degenerates to GPRs.
//!
//! That is what makes fixed-width steps admissible rather than merely faster:
//! they PRESERVE the property padding was chosen to protect, on x86 and
//! aarch64 alike, while the padded form pays 8-20 ns for it.
//!
//! # Ternlog is NOT settled by the AND arms — measured, and it cuts both ways
//!
//! Every other arm here is two-input AND, which is not the question for
//! `mask_ternlog` (codex P1, #315): that is the one mask-algebra op whose
//! narrow descent would reach for `_mm256_ternarylogic_epi64`, a VL
//! instruction. `probe_fixed4_ternlog` asks what LLVM does with a fixed-step
//! arbitrary three-input truth table. Measured 2026-09-16:
//!
//! ```text
//! v4:  vpor %ymm0, %ymm2, %ymm3
//!      vpternlogq $32,  %ymm0, %ymm1, %ymm2
//!      vpternlogq $236, %ymm1, %ymm2, %ymm3      <- 3 logic ops, 2 of them VL
//! v3:  vandnps / vandps / vorps / vandps / vorps  <- 5 logic ops, packed, no VL
//! ```
//!
//! Two findings, and they point opposite ways:
//!
//! - **The `avx512vl` GATE is unnecessary even for ternlog.** LLVM emits
//!   `vpternlogq` on a 256-bit `ymm` from plain Rust when the target supports
//!   it, and degrades to packed boolean ops when it does not. Tier selection
//!   is the compiler's job; naming VL in our source would only duplicate it.
//! - **But an intrinsic descent would still be strictly better HERE.** One
//!   `_mm256_ternarylogic_epi64` is ONE instruction; LLVM used three. That is
//!   a real gap, and it exists for ternlog alone — the AND arms lower to a
//!   single `vandps`.
//!
//! So the throughput question for `mask_ternlog`'s tail is **OPEN**, and the
//! AND measurements must not be read as closing it. What IS closed: the other
//! ten algebra tails, and the gate.

use ndarray::simd::{U64x4, U64x8};

/// 256-bit AND through the facade's narrow type. Probe symbol.
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn probe_u64x4_and(a: &[u64; 4], b: &[u64; 4], out: &mut [u64; 4]) {
    let r = U64x4::from_array(*a) & U64x4::from_array(*b);
    *out = r.to_array();
}

/// 512-bit AND through the facade's wide type — the known-good reference.
/// If THIS does not emit a single wide op the probe is measuring the build,
/// not the type.
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn probe_u64x8_and(a: &[u64; 8], b: &[u64; 8], out: &mut [u64; 8]) {
    let r = U64x8::from_slice(a) & U64x8::from_slice(b);
    r.copy_to_slice(out);
}

/// The hand-written scalar the polyfill's loop is supposed to beat (or match).
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn probe_scalar4_and(a: &[u64; 4], b: &[u64; 4], out: &mut [u64; 4]) {
    for i in 0..4 {
        out[i] = a[i] & b[i];
    }
}

/// A fixed-step 4-lane arbitrary THREE-input truth table — the case that
/// decides whether `mask_ternlog`'s tail can drop AVX-512VL.
///
/// Every other probe here is two-input AND, and two-input AND is not the
/// question (codex P1, #315): `mask_ternlog` is the ONE mask-algebra op whose
/// narrow descent would reach for `_mm256_ternarylogic_epi64`, which IS a VL
/// instruction. If LLVM lowers this to several boolean ops instead of one
/// `vpternlogq`, then the VL gate survives for ternlog even though the AND /
/// OR / XOR / ANDNOT tails do not need it.
///
/// `IMM` is the 8-bit truth table, matching `mask_ternlog`'s own convention:
/// bit `(a<<2)|(b<<1)|c` of `IMM` is the output for that input triple. Written
/// as the canonical sum-of-minterms so nothing but the truth table is assumed.
///
/// `IMM = 0xE8` is bitwise MAJORITY — a bit is set where at least two of the
/// three inputs have it set:
///
/// ```text
/// a = 0xF0F0, b = 0x00FF, c = 0x0F0F
/// a&b = 0x00F0    a&c = 0x0000    b&c = 0x000F    ->   0x00FF
/// ```
///
/// This is deliberately NOT written as a rustdoc example. This file lives in
/// `examples/`, where rustdoc never runs, so a ``` block here would be an
/// untested assertion dressed as a verified one — the exact thing the rest of
/// this PR spent its time removing (coderabbit asked for a usage example,
/// #315). `main` asserts the triple above instead, so the claim is CHECKED on
/// every run rather than decorated.
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn probe_fixed4_ternlog(a: &[u64; 4], b: &[u64; 4], c: &[u64; 4], out: &mut [u64; 4]) {
    const IMM: u64 = 0xE8; // majority(a, b, c) — a table with no 2-input shortcut
    for i in 0..4 {
        let (x, y, z) = (a[i], b[i], c[i]);
        let mut r = 0u64;
        for m in 0..8u32 {
            if (IMM >> m) & 1 == 1 {
                let mx = if m & 4 != 0 { x } else { !x };
                let my = if m & 2 != 0 { y } else { !y };
                let mz = if m & 1 != 0 { z } else { !z };
                r |= mx & my & mz;
            }
        }
        out[i] = r;
    }
}

/// Two-lane scalar AND — the last rung of a descent, and the width at which
/// auto-vectorization is least likely to be worth LLVM's while.
#[inline(never)]
#[unsafe(no_mangle)]
pub extern "C" fn probe_scalar2_and(a: &[u64; 2], b: &[u64; 2], out: &mut [u64; 2]) {
    for i in 0..2 {
        out[i] = a[i] & b[i];
    }
}

fn main() {
    let a4 = [0xF0F0u64; 4];
    let b4 = [0x00FFu64; 4];
    let mut o4 = [0u64; 4];
    probe_u64x4_and(&a4, &b4, &mut o4);
    probe_scalar4_and(&a4, &b4, &mut o4);
    let a8 = [0xF0F0u64; 8];
    let b8 = [0x00FFu64; 8];
    let mut o8 = [0u64; 8];
    probe_u64x8_and(&a8, &b8, &mut o8);
    // Pin the ternlog arm's SEMANTICS, not just that it runs: `IMM = 0xE8` is
    // bitwise majority, so a bit survives where at least two inputs set it.
    // 0xF0F0/0x00FF/0x0F0F pairwise-AND to 0x00F0 | 0x0000 | 0x000F = 0x00FF.
    //
    // This assertion earned its place on the first run: it rejected 0x0FFF,
    // which is what I had written from doing the arithmetic in my head (I had
    // 0xF0F0 & 0x0F0F as 0x0F00 when the nibbles do not overlap at all, so it
    // is 0x0000). A truth table transcribed one bit off would still emit
    // plausible `vpternlogq` and survive a read of the assembly — and so would
    // a doc example nobody executes. This fails instead.
    let c4 = [0x0F0Fu64; 4];
    probe_fixed4_ternlog(&a4, &b4, &c4, &mut o4);
    assert_eq!(o4, [0x00FFu64; 4], "IMM=0xE8 must be bitwise majority: 0xF0F0/0x00FF/0x0F0F -> 0x00FF");
    let mut o2 = [0u64; 2];
    probe_scalar2_and(&[1, 2], &[3, 3], &mut o2);
    println!("probe: {o4:?} {o8:?} {o2:?}");
}
