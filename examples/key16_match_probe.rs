//! Is a contiguous fast path worth adding to `ternary_match_strided16_to_mask`?
//!
//! The shipped kernel gathers 32 scalar 8-byte loads into two stack arrays for
//! every group of 16 rows, then compares with `U64x8` ternlog. It takes that
//! path even at `stride_bytes == 16`, where keys are packed back to back and
//! one 64-byte load already holds four whole keys. This probe prices two
//! alternatives against it, in the probe only; the kernel is not touched
//! (plan `key16-contiguous-match-v1`, D-KEY16-2).
//!
//! | arm | what it is | applies at |
//! |---|---|---|
//! | A | `ndarray::simd::ternary_match_strided16_to_mask`, as shipped | every stride |
//! | B | contiguous: 8 `u64::from_le_bytes` over one `&[u8; 64]` chunk -> `U64x8::from_array` -> `ternlog::<XOR_AND>` against alternating `[plo, phi, ..]` / `[clo, chi, ..]` splats -> `cmpeq_mask(0)`, then `m & (m >> 1) & 0x55` and a 4-bit compaction | stride 16 |
//! | C | AVX-512 `_mm512_i64gather_epi64` (two gathers per 8 rows: bytes 0..8 and 8..16), then the same facade `ternlog` / `cmpeq_mask` | strides > 16, `cfg(target_feature = "avx512f")` only |
//!
//! Arm B's byte load: the facade has no `U64x8::from_le_bytes` today (the
//! plan adds one in W3), so B builds the `[u64; 8]` from eight
//! `u64::from_le_bytes` calls over a `&[u8; 64]` and calls `from_array`. That
//! is the scalar-equivalent spelling the plan names for backends without an
//! unaligned vector load; whether LLVM fuses it into one vector load is what
//! the timing includes. Arm C uses raw `core::arch` gathers because the facade
//! has no gather; that is allowed in a probe and nowhere else.
//!
//! All arms must return the SAME mask words (every word, surplus included);
//! the probe aborts before it times anything if they do not. It also checks a
//! misaligned first offset and a non-multiple-of-16 count once per case.
//!
//! Usage (the backend is compile-time; pin the tier and READ the `tier:`
//! line, never infer it):
//!
//! ```text
//! env -u RUSTFLAGS cargo --config .cargo/config-v3.toml run --release --example key16_match_probe
//! env -u RUSTFLAGS cargo --config .cargo/config-v4.toml run --release --example key16_match_probe
//! ```
//!
//! ## PROVISIONAL results
//!
//! **PROVISIONAL.** Measured 2026-10-11 on a 4-core Intel Xeon @ 2.10 GHz
//! virtual machine (Linux 6.18, rustc 1.99.0). The machine was SHARED and NOT
//! quiet: other builds ran on it during these runs (load average 0.7 to 3.0),
//! so absolute nanoseconds and single-run ratios are indicative only. Medians
//! of 101 runs; two full passes per tier, second pass tabulated, the first
//! pass's ratios quoted as a range. Not a decision input until re-run on a
//! quiet host; this comment does not say whether the W3 / W4 thresholds are
//! met.
//!
//! Tier `avx512f=false` (`config-v3`, AVX2; arm C not compiled), speedup of the
//! arm over A (A ns / arm ns), second pass:
//!
//! | N | stride | sel | A ns | A/B |
//! |---|---|---|---|---|
//! | 65 536 | 16 | 1 % | 110 826 | 2.46 |
//! | 65 536 | 16 | 50 % | 109 017 | 2.58 |
//! | 1 048 576 | 16 | 1 % | 1 868 717 | 1.79 |
//! | 1 048 576 | 16 | 50 % | 1 787 177 | 1.77 |
//!
//! First pass A/B: 2.13, 2.31, 1.64, 1.50 (same order). Arm B does not apply
//! at strides 32 / 512, so those rows are arm A alone (A ns at 65 536 rows:
//! 117-160 k at stride 32, 338-377 k at stride 512; at 1 048 576 rows:
//! 2.0-2.8 M and 12.0-14.5 M).
//!
//! Tier `avx512f=true` (`config-v4`), second pass:
//!
//! | N | stride | sel | A ns | B ns | C ns | A/B | A/C |
//! |---|---|---|---|---|---|---|---|
//! | 65 536 | 16 | 1 % | 116 592 | 40 077 | - | 2.91 | - |
//! | 65 536 | 16 | 50 % | 124 107 | 48 711 | - | 2.55 | - |
//! | 65 536 | 32 | 1 % | 109 982 | - | 35 533 | - | 3.10 |
//! | 65 536 | 32 | 50 % | 108 855 | - | 38 209 | - | 2.85 |
//! | 65 536 | 512 | 1 % | 351 340 | - | 380 185 | - | 0.92 |
//! | 65 536 | 512 | 50 % | 354 506 | - | 368 086 | - | 0.96 |
//! | 1 048 576 | 16 | 1 % | 1 682 974 | 836 625 | - | 2.01 | - |
//! | 1 048 576 | 16 | 50 % | 1 794 950 | 819 325 | - | 2.19 | - |
//! | 1 048 576 | 32 | 1 % | 2 053 836 | - | 1 187 172 | - | 1.73 |
//! | 1 048 576 | 32 | 50 % | 1 981 374 | - | 1 171 749 | - | 1.69 |
//! | 1 048 576 | 512 | 1 % | 11 164 272 | - | 8 714 383 | - | 1.28 |
//! | 1 048 576 | 512 | 50 % | 12 831 483 | - | 8 740 774 | - | 1.47 |
//!
//! First pass on the same tier: A/B 2.13, 2.32, 1.51, 2.21; A/C 2.63, 2.79,
//! 1.45, 1.64 at stride 32, and 1.00, 0.95, 1.41, 1.36 at stride 512 (same row
//! order as the table). The one pattern stable across both passes: at stride
//! 512 and 65 536 rows arm C is at or slightly below arm A (0.92 to 1.00), and
//! its advantage at 1 048 576 rows (1.28 to 1.64) is smaller than at stride
//! 32.

use std::hint::black_box;
use std::time::Instant;

use ndarray::simd::ternlog::XOR_AND;
use ndarray::simd::{ternary_match_strided16_to_mask, U64x8};

const REPS: usize = 101;
const WARMUP: usize = 5;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 11
}

fn words_for(count: usize) -> usize {
    count.div_ceil(64)
}

/// The same up-front validation the shipped kernel performs, so B and C fail
/// loudly rather than read out of bounds.
fn check_bounds(bytes: &[u8], off: usize, stride: usize, count: usize, out: &[u64]) {
    assert!(out.len() >= words_for(count));
    if count > 0 {
        let last_end = (count - 1) * stride + off + 16;
        assert!(last_end <= bytes.len(), "last element ends at {last_end} > {}", bytes.len());
    }
}

fn halves(pattern: &[u8; 16], care: &[u8; 16]) -> (u64, u64, u64, u64) {
    (
        u64::from_le_bytes(pattern[0..8].try_into().unwrap()),
        u64::from_le_bytes(pattern[8..16].try_into().unwrap()),
        u64::from_le_bytes(care[0..8].try_into().unwrap()),
        u64::from_le_bytes(care[8..16].try_into().unwrap()),
    )
}

/// Scalar tail shared by B and C: rows `from..count`, same arithmetic as the
/// shipped kernel's tail.
#[allow(clippy::too_many_arguments)]
fn scalar_tail(
    bytes: &[u8], off: usize, stride: usize, from: usize, count: usize, h: (u64, u64, u64, u64), out: &mut [u64],
) {
    let (plo, phi, clo, chi) = h;
    for i in from..count {
        let o = off + i * stride;
        let l = u64::from_le_bytes(bytes[o..o + 8].try_into().unwrap());
        let hi = u64::from_le_bytes(bytes[o + 8..o + 16].try_into().unwrap());
        if (l ^ plo) & clo == 0 && (hi ^ phi) & chi == 0 {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
}

/// Arm B: contiguous stride-16 path. Four keys per `U64x8`, 16 rows per group.
fn arm_b(bytes: &[u8], off: usize, count: usize, pattern: &[u8; 16], care: &[u8; 16], out: &mut [u64]) {
    check_bounds(bytes, off, 16, count, out);
    out.iter_mut().for_each(|w| *w = 0);
    let h = halves(pattern, care);
    let (plo, phi, clo, chi) = h;
    let vp = U64x8::from_array([plo, phi, plo, phi, plo, phi, plo, phi]);
    let vc = U64x8::from_array([clo, chi, clo, chi, clo, chi, clo, chi]);
    let zero = U64x8::splat(0);
    let groups = count / 16;
    for g in 0..groups {
        let mut bits = 0u16;
        for q in 0..4 {
            let o = off + g * 256 + q * 64;
            let chunk: &[u8; 64] = bytes[o..o + 64].try_into().unwrap();
            let arr: [u64; 8] =
                core::array::from_fn(|j| u64::from_le_bytes(chunk[j * 8..j * 8 + 8].try_into().unwrap()));
            let v = U64x8::from_array(arr).ternlog::<XOR_AND>(vp, vc);
            let m = v.cmpeq_mask(zero);
            // A key matches iff BOTH of its halves (bits 2j, 2j+1) are zero.
            let k = m & (m >> 1) & 0x55;
            let nib = (k & 1) | ((k >> 1) & 2) | ((k >> 2) & 4) | ((k >> 3) & 8);
            bits |= (nib as u16) << (q * 4);
        }
        out[g / 4] |= (bits as u64) << ((g % 4) * 16);
    }
    scalar_tail(bytes, off, 16, groups * 16, count, h, out);
}

/// Arm C: AVX-512 64-bit gather for strides > 16.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn arm_c(bytes: &[u8], off: usize, stride: usize, count: usize, pattern: &[u8; 16], care: &[u8; 16], out: &mut [u64]) {
    use core::arch::x86_64::{__m512i, _mm512_add_epi64, _mm512_i64gather_epi64, _mm512_set1_epi64, _mm512_setr_epi64};
    check_bounds(bytes, off, stride, count, out);
    out.iter_mut().for_each(|w| *w = 0);
    let h = halves(pattern, care);
    let (plo, phi, clo, chi) = h;
    let (vplo, vclo, vphi, vchi) = (U64x8::splat(plo), U64x8::splat(clo), U64x8::splat(phi), U64x8::splat(chi));
    let zero = U64x8::splat(0);
    let s = stride as i64;
    // SAFETY: avx512f is enabled for this build (the cfg above). Every index
    // below addresses a record start `off + i * stride` with `i < count`, and
    // `check_bounds` proved `(count - 1) * stride + off + 16 <= bytes.len()`,
    // so both 8-byte gathers per row (at +0 and +8) stay inside `bytes`.
    let step = unsafe { _mm512_setr_epi64(0, s, 2 * s, 3 * s, 4 * s, 5 * s, 6 * s, 7 * s) };
    let groups = count / 16;
    for g in 0..groups {
        let mut bits = 0u16;
        for half in 0..2usize {
            let first_row = g * 16 + half * 8;
            let base = (off + first_row * stride) as i64;
            // SAFETY: see above; the pointer arithmetic stays within `bytes`.
            let (lo, hi): ([u64; 8], [u64; 8]) = unsafe {
                let idx = _mm512_add_epi64(_mm512_set1_epi64(base), step);
                let lo = _mm512_i64gather_epi64::<1>(idx, bytes.as_ptr().cast::<i64>());
                let hi = _mm512_i64gather_epi64::<1>(idx, bytes.as_ptr().add(8).cast::<i64>());
                (core::mem::transmute::<__m512i, [u64; 8]>(lo), core::mem::transmute::<__m512i, [u64; 8]>(hi))
            };
            let rl = U64x8::from_array(lo).ternlog::<XOR_AND>(vplo, vclo);
            let rh = U64x8::from_array(hi).ternlog::<XOR_AND>(vphi, vchi);
            let m = (rl | rh).cmpeq_mask(zero);
            bits |= (m as u16) << (half * 8);
        }
        out[g / 4] |= (bits as u64) << ((g % 4) * 16);
    }
    scalar_tail(bytes, off, stride, groups * 16, count, h, out);
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
fn arm_c(
    _bytes: &[u8], _off: usize, _stride: usize, _count: usize, _pattern: &[u8; 16], _care: &[u8; 16], _out: &mut [u64],
) {
    unreachable!("arm C is compiled only with avx512f");
}

const C_AVAILABLE: bool = cfg!(all(target_arch = "x86_64", target_feature = "avx512f"));

/// `n` records of `stride` bytes; a fraction `p` of them match `pattern` on
/// the cared bits (noise on the don't-care bits), the rest are random.
fn build(n: usize, stride: usize, p: f64, pattern: &[u8; 16], care: &[u8; 16], seed: u64) -> Vec<u8> {
    let mut s = seed;
    let mut bytes = vec![0u8; n * stride + 16];
    let thresh = (p * (1u64 << 53) as f64) as u64;
    for i in 0..n {
        let rec = &mut bytes[i * stride..i * stride + 16];
        let hit = lcg(&mut s) < thresh;
        for k in 0..16 {
            let r = lcg(&mut s) as u8;
            rec[k] = if hit { pattern[k] ^ (!care[k] & r) } else { r };
        }
    }
    bytes
}

fn median_ns(mut f: impl FnMut()) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let mut ts: Vec<f64> = (0..REPS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64
        })
        .collect();
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts[REPS / 2]
}

fn main() {
    println!(
        "tier: arch={} avx2={} avx512f={} neon={} simd128={}",
        std::env::consts::ARCH,
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
    );
    println!("reps={REPS} (median), arm C available: {C_AVAILABLE}");

    let mut pattern = [0u8; 16];
    let mut care = [0xFFu8; 16];
    let mut s = 0x16u64;
    for b in pattern.iter_mut() {
        *b = lcg(&mut s) as u8;
    }
    care[5] = 0xF0; // one partially-cared byte, so the noise bits matter
    care[12] = 0xFF;
    care[15] = 0xFF;

    println!(
        "\n{:>9} {:>6} {:>5} | {:>10} {:>10} {:>10} | {:>7} {:>7}",
        "N", "stride", "sel", "A ns", "B ns", "C ns", "A/B", "A/C"
    );
    for &n in &[65_536usize, 1_048_576] {
        for &stride in &[16usize, 32, 512] {
            for &p in &[0.01f64, 0.5] {
                let bytes = build(n, stride, p, &pattern, &care, 0xC0FFEE ^ (stride as u64) ^ n as u64);
                let wlen = words_for(n) + 1;

                // Gate: bit-identical answers BEFORE any timing. Full count at
                // offset 0, then a misaligned offset and a ragged count.
                // The offset-5 view is the same records behind 5 leading bytes.
                let shifted: Vec<u8> = std::iter::repeat_n(0xA5u8, 5)
                    .chain(bytes.iter().copied())
                    .collect();
                for &(off, count, buf) in &[(0usize, n, &bytes), (5, n - 37, &shifted)] {
                    let mut a = vec![u64::MAX; wlen];
                    ternary_match_strided16_to_mask(buf, off, stride, count, &pattern, &care, &mut a);
                    let ones: u32 = a.iter().map(|w| w.count_ones()).sum();
                    assert!(ones > 0 && (ones as usize) < count, "degenerate fixture: {ones} hits of {count}");
                    if stride == 16 {
                        let mut b = vec![u64::MAX; wlen];
                        arm_b(buf, off, count, &pattern, &care, &mut b);
                        assert_eq!(a, b, "arm B != arm A (n={n} stride={stride} p={p} off={off} count={count})");
                    } else if C_AVAILABLE {
                        let mut c = vec![u64::MAX; wlen];
                        arm_c(buf, off, stride, count, &pattern, &care, &mut c);
                        assert_eq!(a, c, "arm C != arm A (n={n} stride={stride} p={p} off={off} count={count})");
                    }
                }

                drop(shifted);

                let mut out = vec![0u64; wlen];
                let a_ns = median_ns(|| {
                    ternary_match_strided16_to_mask(black_box(&bytes), 0, stride, n, &pattern, &care, &mut out);
                    black_box(&out);
                });
                let b_ns = (stride == 16).then(|| {
                    median_ns(|| {
                        arm_b(black_box(&bytes), 0, n, &pattern, &care, &mut out);
                        black_box(&out);
                    })
                });
                let c_ns = (stride > 16 && C_AVAILABLE).then(|| {
                    median_ns(|| {
                        arm_c(black_box(&bytes), 0, stride, n, &pattern, &care, &mut out);
                        black_box(&out);
                    })
                });
                let fmt = |v: Option<f64>| v.map_or("-".to_string(), |x| format!("{x:.0}"));
                let ratio = |v: Option<f64>| v.map_or("-".to_string(), |x| format!("{:.2}", a_ns / x));
                println!(
                    "{:>9} {:>6} {:>4.0}% | {:>10.0} {:>10} {:>10} | {:>7} {:>7}",
                    n,
                    stride,
                    p * 100.0,
                    a_ns,
                    fmt(b_ns),
                    fmt(c_ns),
                    ratio(b_ns),
                    ratio(c_ns)
                );
            }
        }
    }
    println!("\nall arms bit-identical to arm A at every case (checked before timing)");
}
