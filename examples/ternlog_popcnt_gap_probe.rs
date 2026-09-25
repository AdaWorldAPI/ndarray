//! Where does `mask_ternlog_popcount` lose to an inlined loop?
//!
//! Five arms count `(a & b) | c` (`AND2_OR`) over the same planes, all asserted
//! equal:
//!
//! | arm | what it is |
//! |---|---|
//! | P | production `mask_ternlog_popcount::<AND2_OR>` |
//! | T | the same loop with the popcount REMOVED (XOR-accumulate): the loop's floor |
//! | M | ternlog + a Mula popcount (VPSHUFB nibble LUT + VPSADBW) — LAB ARM, raw AVX-512BW intrinsics, bounds what a backend `popcnt` fix buys; never shipped from here |
//! | S | plain scalar `((a & b) \| c).count_ones()` over zipped slices (what LLVM makes of it) |
//!
//! A second table runs five planes `((a|b)&c)^(d&!e)` three ways: the
//! production two-pass shape (`g` into a 256-word chunk, then `h` folded by
//! popcount), a register-resident two-ternlog `U64x8` loop (also unrolled x4),
//! and the scalar loop.
//!
//! Measured 2026-09-25 (Xeon @ 2.8 GHz, AVX-512F/BW, no VPOPCNTDQ; median, 3 runs):
//!
//! - 3 planes: P == M at every size (the scalar-lane popcount fallback compiles
//!   as fast as a Mula LUT), and P is within ~1.15x of the no-popcount floor T.
//!   At 1024 words (one 64k-row V3 tile) P is 1.3x FASTER than S.
//!   **The kernel's loop shape is not the gap.**
//! - 5 planes, 4K-16K words: two-pass 13.3 us, register-resident 11.8 us,
//!   unrolled x4 11.9 us, scalar 9.0 us. Fusing the two tables in registers
//!   buys only 1.1x; the scalar loop is 1.2-1.5x ahead of every `U64x8` form.
//!   Unrolling does not explain it, and a `prefer-256-bit` on/off rebuild was
//!   inconclusive (the effect is inside run-to-run noise on this host). OPEN.
//!
//! Build: `RUSTFLAGS="-C target-cpu=native" cargo run --release --example ternlog_popcnt_gap_probe`

use std::time::Instant;

use ndarray::simd::ternlog::AND2_OR;
use ndarray::simd::{mask_ternlog_popcount, U64x8};

const L: usize = U64x8::LANES;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

fn median<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut ts = Vec::with_capacity(reps);
    let mut last = None;
    for _ in 0..reps {
        let t = Instant::now();
        last = Some(std::hint::black_box(f()));
        ts.push(t.elapsed().as_nanos() as f64);
    }
    ts.sort_by(|a, b| a.total_cmp(b));
    (ts[reps / 2], last.expect("reps > 0"))
}

/// T: the kernel's loop without the popcount. Not a count; the floor.
fn ternlog_only(a: &[u64], b: &[u64], c: &[u64]) -> u64 {
    let (ca, _) = a.as_chunks::<L>();
    let (cb, _) = b.as_chunks::<L>();
    let (cc, _) = c.as_chunks::<L>();
    let mut acc = U64x8::splat(0);
    for ((x, y), z) in ca.iter().zip(cb).zip(cc) {
        acc ^= U64x8::from_array(*x).ternlog::<AND2_OR>(U64x8::from_array(*y), U64x8::from_array(*z));
    }
    acc.to_array().iter().fold(0, |s, w| s ^ w)
}

/// M: ternlog + Mula popcount. LAB ARM (raw intrinsics).
#[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
fn ternlog_mula(a: &[u64], b: &[u64], c: &[u64]) -> u64 {
    use core::arch::x86_64::*;
    assert!(a.len().is_multiple_of(L));
    // SAFETY: avx512f/bw enabled at compile time (cfg); loads are unaligned
    // reads of 8 u64 inside the slices (len is a multiple of 8, asserted).
    unsafe {
        let lut = _mm512_set4_epi32(0x0403_0302, 0x0302_0201, 0x0302_0201, 0x0201_0100);
        let lo = _mm512_set1_epi8(0x0F);
        let mut acc = _mm512_setzero_si512();
        let mut i = 0;
        while i < a.len() {
            let va = _mm512_loadu_si512(a.as_ptr().add(i).cast());
            let vb = _mm512_loadu_si512(b.as_ptr().add(i).cast());
            let vc = _mm512_loadu_si512(c.as_ptr().add(i).cast());
            let t = _mm512_ternarylogic_epi64::<AND2_OR>(va, vb, vc);
            let l = _mm512_shuffle_epi8(lut, _mm512_and_si512(t, lo));
            let h = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(t), lo));
            acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_add_epi8(l, h), _mm512_setzero_si512()));
            i += L;
        }
        _mm512_reduce_add_epi64(acc) as u64
    }
}

/// S: the scalar loop as written.
fn scalar(a: &[u64], b: &[u64], c: &[u64]) -> u64 {
    a.iter()
        .zip(b)
        .zip(c)
        .map(|((x, y), z)| u64::from(((x & y) | z).count_ones()))
        .sum()
}

/// Five planes, `((a|b)&c) ^ (d&!e)` as `h(g(a,b,c), d, e)`:
/// g = (a|b)&c = 0xE0 (a=0xF0,b=0xCC,c=0xAA: (F0|CC)&AA = FC&AA = A8)
const G: i32 = 0xA8;
/// h(t,d,e) = t ^ (d & !e): t=0xF0, d=0xCC, e=0xAA -> F0 ^ (CC & 55) = F0 ^ 44 = B4
const H: i32 = 0xB4;

/// Production shape of `Tern2`: g into an L1 chunk, then h folded by popcount.
fn two_pass(p: [&[u64]; 5], chunk: usize) -> u64 {
    let mut t = vec![0u64; chunk];
    let mut n = 0;
    let mut i = 0;
    while i < p[0].len() {
        let j = (i + chunk).min(p[0].len());
        let t = &mut t[..j - i];
        ndarray::simd::mask_ternlog::<G>(&p[0][i..j], &p[1][i..j], &p[2][i..j], t);
        n += mask_ternlog_popcount::<H>(t, &p[3][i..j], &p[4][i..j]);
        i = j;
    }
    n
}

/// Register-resident: both tables per 8-word chunk, from existing U64x8 methods.
fn fused2(p: [&[u64]; 5]) -> u64 {
    let c: Vec<&[[u64; L]]> = p.iter().map(|s| s.as_chunks::<L>().0).collect();
    let mut acc = U64x8::splat(0);
    #[allow(clippy::needless_range_loop)] // k indexes five parallel slices
    for k in 0..c[0].len() {
        let v = |i: usize| U64x8::from_array(c[i][k]);
        let t = v(0).ternlog::<G>(v(1), v(2));
        acc += t.ternlog::<H>(v(3), v(4)).popcnt();
    }
    acc.reduce_sum()
}

/// `fused2`, unrolled x4 with four independent accumulators.
fn fused2_x4(p: [&[u64]; 5]) -> u64 {
    let c: Vec<&[[u64; L]]> = p.iter().map(|s| s.as_chunks::<L>().0).collect();
    let n = c[0].len();
    let mut acc = [U64x8::splat(0); 4];
    let mut k = 0;
    while k + 4 <= n {
        for (j, a) in acc.iter_mut().enumerate() {
            let v = |i: usize| U64x8::from_array(c[i][k + j]);
            *a += v(0)
                .ternlog::<G>(v(1), v(2))
                .ternlog::<H>(v(3), v(4))
                .popcnt();
        }
        k += 4;
    }
    while k < n {
        let v = |i: usize| U64x8::from_array(c[i][k]);
        acc[0] += v(0)
            .ternlog::<G>(v(1), v(2))
            .ternlog::<H>(v(3), v(4))
            .popcnt();
        k += 1;
    }
    acc.iter().map(|a| a.reduce_sum()).sum()
}

fn scalar5(p: [&[u64]; 5]) -> u64 {
    (0..p[0].len())
        .map(|i| u64::from((((p[0][i] | p[1][i]) & p[2][i]) ^ (p[3][i] & !p[4][i])).count_ones()))
        .sum()
}

fn five_planes() {
    println!("\n5 planes ((a|b)&c)^(d&!e)");
    println!(
        "{:>9} {:>11} {:>11} {:>11}   {:>8} {:>8}",
        "words", "2pass ns", "fused2 ns", "scalar ns", "2p/f2", "2p/sc"
    );
    let mut seed = 11;
    for words in [1024usize, 4096, 16384, 262_144] {
        let v: Vec<Vec<u64>> = (0..5)
            .map(|_| (0..words).map(|_| lcg(&mut seed)).collect())
            .collect();
        let p = [&v[0][..], &v[1][..], &v[2][..], &v[3][..], &v[4][..]];
        let reps = if words > 100_000 { 21 } else { 201 };
        let (t2, a) = median(reps, || two_pass(p, 256));
        let (tf, b) = median(reps, || fused2(p));
        let (tx, x) = median(reps, || fused2_x4(p));
        let (ts, c) = median(reps, || scalar5(p));
        assert_eq!(x, c);
        print!("  [x4 {tx:.0} ns, f2/x4 {:.2}x] ", tf / tx);
        assert_eq!(a, c);
        assert_eq!(b, c);
        println!("{words:>9} {t2:>11.0} {tf:>11.0} {ts:>11.0}   {:>7.2}x {:>7.2}x", t2 / tf, t2 / ts);
    }
}

fn main() {
    println!(
        "backend: avx512vpopcntdq={} avx512bw={}",
        cfg!(target_feature = "avx512vpopcntdq"),
        cfg!(target_feature = "avx512bw")
    );
    println!(
        "{:>9} {:>10} {:>10} {:>10} {:>10}   {:>6} {:>6}",
        "words", "P ns", "T ns", "M ns", "S ns", "P/M", "P/T"
    );
    let mut seed = 7;
    for words in [256usize, 512, 1024, 2048, 4096, 16384, 262_144] {
        let mk = |s: &mut u64| (0..words).map(|_| lcg(s)).collect::<Vec<u64>>();
        let (a, b, c) = (mk(&mut seed), mk(&mut seed), mk(&mut seed));
        let reps = if words > 100_000 { 21 } else { 201 };
        let (tp, vp) = median(reps, || mask_ternlog_popcount::<AND2_OR>(&a, &b, &c));
        let (tt, _) = median(reps, || ternlog_only(&a, &b, &c));
        let (ts, vs) = median(reps, || scalar(&a, &b, &c));
        assert_eq!(vp, vs);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512bw"))]
        let tm = {
            let (tm, vm) = median(reps, || ternlog_mula(&a, &b, &c));
            assert_eq!(vm, vp);
            tm
        };
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512bw")))]
        let tm = f64::NAN;
        println!("{words:>9} {tp:>10.0} {tt:>10.0} {tm:>10.0} {ts:>10.0}   {:>5.2}x {:>5.2}x", tp / tm, tp / tt);
    }
    five_planes();
}
