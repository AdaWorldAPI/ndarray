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
    let mut o2 = [0u64; 2];
    probe_scalar2_and(&[1, 2], &[3, 3], &mut o2);
    println!("probe: {o4:?} {o8:?} {o2:?}");
}
