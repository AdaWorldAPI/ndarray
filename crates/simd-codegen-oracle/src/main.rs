//! SIMD codegen oracle — probe kernels for `scripts/codegen-oracle.sh`.
//!
//! # Why this exists
//!
//! `.cargo/config.toml` pins `-Ctarget-cpu=x86-64-v3`. Because of that, this
//! crate's "scalar polyfill" SIMD storage types — macro-generated `[T; N]`
//! arrays, e.g. `avx2_int_type!(U32x16, u32, 16, 0u32)` at
//! `src/simd_avx2.rs:1542` — already compile to packed AVX2 instructions:
//! LLVM's loop/SLP vectorizer sees a fixed-trip-count loop over an aligned
//! array with no cross-lane data dependency and lowers it to `vpaddd` /
//! `vpxor` / `vpsrld` etc. A recent PR hand-wrote ~700 lines of intrinsics to
//! "fix" a gap that did not exist. This binary makes that a TESTED invariant
//! (Group A below) instead of a fact someone has to rediscover by reading
//! disassembly.
//!
//! The oracle must discriminate in BOTH directions: it must also confirm the
//! *absence* of vectorization where the shape of the code makes it
//! impossible (Group B) — a tool that reports "everything vectorizes" is as
//! useless as one that reports nothing does.
//!
//! Group C is neither: it is the open empirical question that motivated
//! extending this oracle — whether a hand-written scalar `u64` rotate loop
//! (mirroring the *actual* library pattern used for `U32x16::rotate_left`)
//! gets the same free ride from LLVM that the `u32` lane does. No backend in
//! this crate (avx512 / avx2 / scalar / neon / wasm / nightly) currently
//! defines a `rotate_left`/`rotate_right` on any `u64` lane type, so there is
//! no existing library function to call here — the probes below are written
//! by hand, deliberately mirroring `U32x16::rotate_left`'s shape line for
//! line, so the comparison is apples-to-apples.
//!
//! # Rules every probe kernel follows
//!
//! - `#[inline(never)]` so it survives as its own labeled symbol in the
//!   emitted assembly (`scripts/codegen-oracle.sh` locates probes by name via
//!   the `.type <sym>,@function` label).
//! - Inputs arrive as parameters, never as `const`/literal-folded values —
//!   `main` builds every input from a runtime seed (wall-clock time XORed
//!   with argc, see `runtime_seed()` below) and pipes it through
//!   `std::hint::black_box` before the call, so LLVM cannot constant-fold
//!   into the callee even under an optimizer aggressive enough to see past
//!   `#[inline(never)]` at the call site.
//! - The return value is consumed (folded into the printed report), so nothing
//!   is dead-code-eliminated.

use ndarray::simd::{add_mul_f32, I8x32, U32x16, U64x4, U64x8, U8x64};
use std::hint::black_box;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Group A — expected to FULLY VECTORIZE
// ============================================================================
// The scalar-source-is-not-scalar-codegen claim. Every kernel here operates
// on a full ndarray::simd lane type (or a fixed-width slice loop through the
// library's own `add_mul_f32`), with no loop-carried dependency *between*
// lanes and a compile-time-fixed lane count. LLVM has everything it needs to
// lower these to packed AVX2 without any hand-written intrinsic.

/// The ChaCha/BLAKE ARX triple over `U32x16`: `(a+b)^b`, then `rotate_left(16)`.
/// `U32x16` is the `[u32; 16]` scalar-polyfill storage (`simd_avx2.rs`); its
/// `Add` / `BitXor` / `rotate_left` are themselves per-lane scalar loops in
/// the library source — this probe measures whether that scalar *source*
/// still lowers to packed AVX2 *codegen* at `-Ctarget-cpu=x86-64-v3`.
#[inline(never)]
pub fn arx_u32x16(a: U32x16, b: U32x16) -> U32x16 {
    ((a + b) ^ b).rotate_left(16)
}

/// Ten iterations of a ChaCha-style ARX double-round over four `U32x16`
/// lanes (16 independent ChaCha block-instances processed in parallel, one
/// per SIMD lane) — the actual production shape: a small, compile-time-fixed
/// trip count wrapped around several `U32x16` ARX ops per iteration.
#[inline(never)]
pub fn arx_rounds_u32x16(state: [U32x16; 4]) -> [U32x16; 4] {
    let [mut a, mut b, mut c, mut d] = state;
    for _ in 0..10 {
        a += b;
        d = (d ^ a).rotate_left(16);
        c += d;
        b = (b ^ c).rotate_left(12);
        a += b;
        d = (d ^ a).rotate_left(8);
        c += d;
        b = (b ^ c).rotate_left(7);
    }
    [a, b, c, d]
}

/// Fused multiply-add into an accumulator slice via the library's own
/// `ndarray::simd::add_mul_f32` — built on `F32x16::mul_add` (native
/// `vfmadd*ps` on AVX2+FMA hosts, which `x86-64-v3` guarantees).
#[inline(never)]
pub fn fma_f32x16(acc: &mut [f32], a: &[f32], b: &[f32]) {
    add_mul_f32(acc, a, b);
}

/// Horizontal reduce over `U32x16` — `avx2_int_type!`'s `reduce_sum`, itself
/// a per-lane `wrapping_add` fold loop in the library source.
#[inline(never)]
pub fn reduce_u32x16(v: U32x16) -> u32 {
    v.reduce_sum()
}

/// Byte-lane bitwise chain over `U8x64`: `(a ^ b) & a | b`.
#[inline(never)]
pub fn bitwise_u8x64(a: U8x64, b: U8x64) -> U8x64 {
    (a ^ b) & a | b
}

// ============================================================================
// Group B — expected to NOT fully vectorize
// ============================================================================
// This half is the point. Do NOT "fix" these if they come out scalar — a
// scalar verdict here is the expected, correct finding. Each kernel's shape
// defeats auto-vectorization for a distinct, structural reason (loop-carried
// serial dependency, gather-shaped indexing, or an inherently cross-lane
// permute), independent of whatever the library's storage type happens to be.

/// A loop where iteration `i` depends on iteration `i-1` through a
/// rotate+xor — an inherently serial ARX-shaped chain (as opposed to
/// `arx_rounds_u32x16` above, where the 10 "rounds" only chain full 16-lane
/// vector registers together, never individual scalar values). `n` is a
/// runtime parameter, not a compile-time constant, so LLVM cannot unroll and
/// then discover parallelism across iterations even if it wanted to — the
/// data dependency itself is nonlinear (rotate is not linear over XOR/add),
/// so there is no parallel prefix-scan trick available either.
#[inline(never)]
pub fn serial_dependent_chain(seed: u32, n: u32) -> u32 {
    let mut x = seed;
    for _ in 0..n {
        x = x.rotate_left(7) ^ x.wrapping_add(0x9E37_79B9);
    }
    x
}

/// Lane-wise `i8::saturating_abs` written as a scalar loop over
/// `I8x32::to_array()`. `abs(i8::MIN)` does not fit in `i8` (`+128`
/// overflows), which is exactly the correction recorded in
/// `.claude/knowledge/vertical-simd-consumer-contract.md` § "VPABSB
/// correction": `_mm512_abs_epi8` does NOT saturate `i8::MIN` by itself: the
/// real hardware-correct implementation needs `abs` + `min_epu8(_, 0x7f)`.
/// Whether LLVM's auto-vectorizer discovers that two-instruction idiom from
/// a scalar `saturating_abs()` call in a 32-iteration loop is exactly the
/// kind of thing this oracle exists to measure rather than assume.
#[inline(never)]
pub fn saturating_abs_i8x32(v: I8x32) -> I8x32 {
    let arr = v.to_array();
    let mut out = [0i8; 32];
    for i in 0..32 {
        out[i] = arr[i].saturating_abs();
    }
    I8x32::from_array(out)
}

/// Per-lane table lookup (gather-shaped: `table[idx[i]]`, no contiguous
/// load). LLVM's auto-vectorizer does not synthesize gather instructions
/// from a scalar indexed-load loop by default.
#[inline(never)]
pub fn gather_lookup_u8(table: &[u8; 256], idx: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..32 {
        out[i] = table[idx[i] as usize];
    }
    out
}

/// Zero-extend 16 × `u16` to `f32` via a scalar loop (`as` cast per lane).
#[inline(never)]
pub fn widening_u16_to_f32(v: [u16; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = v[i] as f32;
    }
    out
}

/// Reverse all 64 byte lanes of a `U8x64` via a scalar index loop — an
/// arbitrary cross-lane permute with no per-128-bit-lane locality (unlike
/// e.g. `unpack_lo_epi8` in `simd_avx2.rs`, which stays within 16-byte
/// sub-lanes and is realizable with `vpshufb`).
#[inline(never)]
pub fn cross_lane_reverse_u8x64(v: U8x64) -> U8x64 {
    let arr = v.to_array();
    let mut out = [0u8; 64];
    for i in 0..64 {
        out[i] = arr[63 - i];
    }
    U8x64::from_array(out)
}

// ============================================================================
// Group C — UNKNOWN. Do not pre-classify. This is the open question.
// ============================================================================
// No backend in this crate defines rotate_left/rotate_right on any u64 lane
// type today (measured: zero hits across avx512/avx2/scalar/neon/wasm/
// nightly). BLAKE2b — which argon2 uses — is a 64-bit ARX cipher, so this is
// a genuine gap, not a stylistic one. The question: does a scalar-shaped u64
// rotate loop, mirroring U32x16::rotate_left's own shape line for line, get
// the same free ride from LLVM that the u32 lane gets? If yes, the u64 ARX
// lane is free (no intrinsic needed, same as U32x16::rotate_left today). If
// no, this is the first legitimately justified case in this crate for a
// hand-written intrinsic override (AVX-512 has `_mm512_rorv_epi64` /
// VPROLVQ as a single instruction; AVX2 baseline has no direct equivalent).

/// Mirrors `U32x16::rotate_left`'s exact shape (`simd_avx2.rs:1553`) at u64
/// width, 8 lanes: `to_array()` → per-lane `u64::rotate_right` → `from_array()`.
/// `#[inline(always)]` so its expansion appears directly in the probe
/// symbols below (`rot_u64x8` / `blake2b_g_u64x8`), not behind a `call`.
#[inline(always)]
fn scalar_rotr_u64x8(v: U64x8, n: u32) -> U64x8 {
    let arr = v.to_array();
    let mut out = [0u64; 8];
    for i in 0..8 {
        out[i] = arr[i].rotate_right(n);
    }
    U64x8::from_array(out)
}

/// 256-bit sibling of [`scalar_rotr_u64x8`] — 4 lanes.
#[inline(always)]
fn scalar_rotr_u64x4(v: U64x4, n: u32) -> U64x4 {
    let arr = v.to_array();
    let mut out = [0u64; 4];
    for i in 0..4 {
        out[i] = arr[i].rotate_right(n);
    }
    U64x4::from_array(out)
}

/// Lane-wise `u64::rotate_right(n)` over `U64x8`, 8-wide (512-bit-equivalent
/// storage). The probe symbol itself, `#[inline(never)]`.
#[inline(never)]
pub fn rot_u64x8(v: U64x8, n: u32) -> U64x8 {
    scalar_rotr_u64x8(v, n)
}

/// Lane-wise `u64::rotate_right(n)` over `U64x4`, 4-wide (256-bit storage).
#[inline(never)]
pub fn rot_u64x4(v: U64x4, n: u32) -> U64x4 {
    scalar_rotr_u64x4(v, n)
}

/// One BLAKE2b G-function mixing step over `U64x8` — the u64 analogue of
/// `arx_rounds_u32x16` above, and the actual kernel an argon2/BLAKE2b lane
/// would need. Rotate amounts 32/24/16 are byte-granular (like ChaCha's 16
/// and 8, which fold to `vpshufb` on the u32 lane); 63 is not — the
/// instruction histogram may show a split between shuffle-lowered and
/// shift-or-lowered rotates, which is itself a finding worth reporting
/// separately, not averaging away.
#[inline(never)]
pub fn blake2b_g_u64x8(a: U64x8, b: U64x8, c: U64x8, d: U64x8) -> (U64x8, U64x8, U64x8, U64x8) {
    let mut a = a;
    let mut b = b;
    let mut c = c;
    let mut d = d;
    a += b;
    d = scalar_rotr_u64x8(d ^ a, 32);
    c += d;
    b = scalar_rotr_u64x8(b ^ c, 24);
    a += b;
    d = scalar_rotr_u64x8(d ^ a, 16);
    c += d;
    b = scalar_rotr_u64x8(b ^ c, 63);
    (a, b, c, d)
}

// ============================================================================
// Driver — runtime-derived inputs, every result consumed.
// ============================================================================

/// Wall-clock nanoseconds XORed with argc — a runtime value no build-time
/// constant folder can predict, used to seed a small PRNG for building probe
/// inputs. Piped through `black_box` at every call site below as well, so
/// nothing about a probe's actual argument values is visible to the
/// optimizer at compile time.
fn runtime_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let argc = std::env::args().count() as u64;
    black_box(nanos ^ argc.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// splitmix64 — deterministic given a seed, but the seed itself is
/// runtime-derived (see `runtime_seed`), so the sequence is not a
/// compile-time constant anywhere it is consumed.
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
    let mut rng = SplitMix64(runtime_seed());
    let mut acc: u64 = 0;

    // ---- Group A ----
    let a16: [u32; 16] = std::array::from_fn(|_| rng.next() as u32);
    let b16: [u32; 16] = std::array::from_fn(|_| rng.next() as u32);
    let r = arx_u32x16(black_box(U32x16::from_array(a16)), black_box(U32x16::from_array(b16)));
    acc ^= r.reduce_sum() as u64;

    let state: [U32x16; 4] = std::array::from_fn(|_| U32x16::from_array(std::array::from_fn(|_| rng.next() as u32)));
    let rounds = arx_rounds_u32x16(black_box(state));
    for lane in &rounds {
        acc ^= lane.reduce_sum() as u64;
    }

    let n = 64usize;
    let mut acc_v: Vec<f32> = (0..n).map(|_| (rng.next() as u32 as f32) * 1e-9).collect();
    let a_v: Vec<f32> = (0..n).map(|_| (rng.next() as u32 as f32) * 1e-9).collect();
    let b_v: Vec<f32> = (0..n).map(|_| (rng.next() as u32 as f32) * 1e-9).collect();
    fma_f32x16(black_box(&mut acc_v), black_box(&a_v), black_box(&b_v));
    acc ^= acc_v.iter().fold(0u64, |s, &x| s ^ x.to_bits() as u64);

    let rv = reduce_u32x16(black_box(U32x16::from_array(std::array::from_fn(|_| rng.next() as u32))));
    acc ^= rv as u64;

    let bw = bitwise_u8x64(
        black_box(U8x64::from_array(std::array::from_fn(|_| rng.next() as u8))),
        black_box(U8x64::from_array(std::array::from_fn(|_| rng.next() as u8))),
    );
    acc ^= bw.reduce_sum() as u64;

    // ---- Group B ----
    let sdc = serial_dependent_chain(black_box(rng.next() as u32), black_box(50 + (rng.next() % 8) as u32));
    acc ^= sdc as u64;

    let sat = saturating_abs_i8x32(black_box(I8x32::from_array(std::array::from_fn(|_| rng.next() as i8))));
    acc ^= sat
        .to_array()
        .iter()
        .map(|&x| x as i64 as u64)
        .fold(0u64, |s, x| s ^ x);

    let mut table = [0u8; 256];
    for (i, t) in table.iter_mut().enumerate() {
        *t = (i as u8).wrapping_mul(0x9B).wrapping_add(rng.next() as u8);
    }
    let idx: [u8; 32] = std::array::from_fn(|_| rng.next() as u8);
    let gathered = gather_lookup_u8(black_box(&table), black_box(&idx));
    acc ^= gathered.iter().fold(0u64, |s, &x| s ^ x as u64);

    let u16in: [u16; 16] = std::array::from_fn(|_| rng.next() as u16);
    let widened = widening_u16_to_f32(black_box(u16in));
    acc ^= widened.iter().fold(0u64, |s, &x| s ^ x.to_bits() as u64);

    let rev = cross_lane_reverse_u8x64(black_box(U8x64::from_array(std::array::from_fn(|_| rng.next() as u8))));
    acc ^= rev.reduce_sum() as u64;

    // ---- Group C ----
    let rot8 = rot_u64x8(
        black_box(U64x8::from_array(std::array::from_fn(|_| rng.next()))),
        black_box(1 + (rng.next() % 63) as u32),
    );
    acc ^= rot8.reduce_sum();

    let rot4 = rot_u64x4(
        black_box(U64x4::from_array(std::array::from_fn(|_| rng.next()))),
        black_box(1 + (rng.next() % 63) as u32),
    );
    acc ^= rot4.reduce_sum();

    let (ga, gb, gc, gd) = blake2b_g_u64x8(
        black_box(U64x8::from_array(std::array::from_fn(|_| rng.next()))),
        black_box(U64x8::from_array(std::array::from_fn(|_| rng.next()))),
        black_box(U64x8::from_array(std::array::from_fn(|_| rng.next()))),
        black_box(U64x8::from_array(std::array::from_fn(|_| rng.next()))),
    );
    acc ^= ga.reduce_sum() ^ gb.reduce_sum() ^ gc.reduce_sum() ^ gd.reduce_sum();

    println!("simd-codegen-oracle: probes executed, combined checksum = {acc:#018x}");
}
