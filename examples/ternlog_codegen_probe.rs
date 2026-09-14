//! Codegen witness for the mask family — the tiny OPTIMIZED surface the SIMD
//! realization matrix inspects with `--emit=asm` (`scripts/codegen-witness.sh`).
//!
//! The semantic arms of the matrix run the masking tests at `opt-level = 0`,
//! which is cheap and proves bits. It proves nothing about machine code: LLVM
//! has not run the transformations we certify. So this example is built under
//! `[profile.ci-codegen]` (opt-level 3, no debuginfo, no LTO) and its symbols
//! are checked for the instruction each backend is REQUIRED to select:
//!
//! | arm | must contain | must not contain |
//! |---|---|---|
//! | AVX-512 (`-Ctarget-cpu=x86-64-v4`) | `vpternlogq` / `vpternlogd` | — |
//! | AVX2 (`x86-64-v3`, native without avx512f) | packed `vpand`/`vpxor`/… on ymm | `vpternlog*`, GPR logic on lane data |
//! | NEON (aarch64) | `and`/`eor`/`bic`/`orr` on `v*.16b` | GPR logic on lane data |
//!
//! Every probe is `#[inline(never)]` so it survives as its own symbol, takes
//! runtime-seeded inputs through `black_box` so nothing constant-folds, and is
//! self-checked against the bit-serial ternlog definition before the assembly
//! is trusted — a packed-but-wrong body must fail here, not read as a success.
//! Calls the SHIPPED `ndarray::simd` methods, never a look-alike.

use ndarray::simd::{mask_ternlog, ternlog, U32x16, U64x8};
use std::hint::black_box;

/// `U64x8::ternlog::<MAJ3>` — the general Shannon-ladder arm.
#[inline(never)]
fn probe_ternlog_u64x8(a: U64x8, b: U64x8, c: U64x8) -> U64x8 {
    a.ternlog::<{ ternlog::MAJ3 }>(b, c)
}

/// `U32x16::ternlog::<XOR_AND>` — the immediate `simd_masking_ops` uses.
#[inline(never)]
fn probe_ternlog_u32x16(a: U32x16, b: U32x16, c: U32x16) -> U32x16 {
    a.ternlog::<{ ternlog::XOR_AND }>(b, c)
}

/// `U64x8::andnot` — the mask set-difference.
#[inline(never)]
fn probe_andnot_u64x8(a: U64x8, b: U64x8) -> U64x8 {
    a.andnot(b)
}

/// The slice-level facade op over 64 words (8 full `U64x8` chunks) — the
/// shape a consumer actually calls; proves the ergonomic layer inlines down
/// to the backend's realization rather than adding a scalar detour.
#[inline(never)]
fn probe_mask_ternlog_slice(a: &[u64], b: &[u64], c: &[u64], dst: &mut [u64]) {
    mask_ternlog::<{ ternlog::AND2_OR }>(a, b, c, dst)
}

fn ref_ternlog(imm: i32, a: u64, b: u64, c: u64) -> u64 {
    let mut out = 0u64;
    for bit in 0..64 {
        let idx = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1);
        out |= (((imm as u64) >> idx) & 1) << bit;
    }
    out
}

struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn main() {
    // Runtime seed: wall clock XOR argc, so no input is a compile-time constant.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5EED);
    let mut rng = SplitMix64(black_box(nanos ^ (std::env::args().count() as u64)));
    let mut acc = 0u64;

    let (a, b, c) = (
        U64x8::from_array(std::array::from_fn(|_| rng.next())),
        U64x8::from_array(std::array::from_fn(|_| rng.next())),
        U64x8::from_array(std::array::from_fn(|_| rng.next())),
    );
    let got = probe_ternlog_u64x8(black_box(a), black_box(b), black_box(c));
    for i in 0..8 {
        assert_eq!(got.to_array()[i], ref_ternlog(ternlog::MAJ3, a.to_array()[i], b.to_array()[i], c.to_array()[i]));
    }
    acc ^= got.reduce_sum();

    let an = probe_andnot_u64x8(black_box(a), black_box(b));
    assert_eq!(an.to_array(), std::array::from_fn::<u64, 8, _>(|i| a.to_array()[i] & !b.to_array()[i]));
    acc ^= an.reduce_sum();

    let (ua, ub, uc) = (
        U32x16::from_array(std::array::from_fn(|_| rng.next() as u32)),
        U32x16::from_array(std::array::from_fn(|_| rng.next() as u32)),
        U32x16::from_array(std::array::from_fn(|_| rng.next() as u32)),
    );
    let ugot = probe_ternlog_u32x16(black_box(ua), black_box(ub), black_box(uc));
    for i in 0..16 {
        let want =
            ref_ternlog(ternlog::XOR_AND, ua.to_array()[i] as u64, ub.to_array()[i] as u64, uc.to_array()[i] as u64);
        assert_eq!(ugot.to_array()[i], want as u32);
    }
    acc ^= ugot.reduce_sum() as u64;

    let sa: Vec<u64> = (0..64).map(|_| rng.next()).collect();
    let sb: Vec<u64> = (0..64).map(|_| rng.next()).collect();
    let sc: Vec<u64> = (0..64).map(|_| rng.next()).collect();
    let mut sd = vec![0u64; 64];
    probe_mask_ternlog_slice(black_box(&sa), black_box(&sb), black_box(&sc), black_box(&mut sd));
    for i in 0..64 {
        assert_eq!(sd[i], ref_ternlog(ternlog::AND2_OR, sa[i], sb[i], sc[i]));
    }
    acc ^= sd.iter().fold(0, |s, &x| s ^ x);

    println!("ternlog_codegen_probe: self-check OK, checksum = {acc:#018x}");
}
