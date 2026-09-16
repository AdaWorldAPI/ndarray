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
