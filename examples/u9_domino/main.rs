//! D-DOMINO-MOV-0 probe: bit flip, positional substitution and lazy FULL
//! forwarding (`domino.rs`) against native integer addition and #389's D.
//! Every row is checked against an exact reference before it is timed.
//! `cargo run --release --example u9_domino --features std`.

#[path = "../u9_head_to_head/kernels.rs"]
mod kernels;
#[path = "../u9_abacus/carry16.rs"]
mod carry16;
mod domino;

use domino::*;
use ndarray::simd::U16x32;
use std::hint::black_box;
use std::time::Instant;

const N: usize = 72;

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512vnni") {
        "x86_64 AVX-512 + VNNI"
    } else if cfg!(target_feature = "avx512bw") {
        "x86_64 AVX-512 (no VNNI)"
    } else if cfg!(target_feature = "avx2") {
        "x86_64 AVX2"
    } else {
        "other"
    }
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Median ns per unit over `runs`.
fn time(name: &str, units: usize, runs: usize, unit: &str, base: Option<f64>, f: &dyn Fn() -> u128) -> f64 {
    black_box(f());
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let t0 = Instant::now();
            black_box(f());
            t0.elapsed().as_nanos() as f64 / units as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    let med = t[runs / 2];
    let rel = base
        .map(|b| format!("  {:6.2}x", med / b))
        .unwrap_or_default();
    println!("{name:<58} {med:9.3} ns/{unit}{rel}");
    med
}

/// Section A: u16 products summed exactly per lane across the 64K boundary.
fn boundary(label: &str, blocks: usize, runs: usize, dense: bool, seed: u64) {
    let mut s = seed;
    let raw: Vec<[u16; 64]> = (0..blocks)
        .map(|_| {
            core::array::from_fn(|_| {
                let a = xorshift(&mut s);
                if dense {
                    65_025 - (a & 1) as u16
                } else {
                    // Small products: a carry about once per 260 blocks per lane.
                    (a & 0xFF) as u16
                }
            })
        })
        .collect();
    let prods: Vec<[U16x32; 2]> = raw
        .iter()
        .map(|p| [U16x32::from_slice(&p[..32]), U16x32::from_slice(&p[32..])])
        .collect();
    let want: u128 = raw.iter().flatten().map(|&x| u128::from(x)).sum();
    let native = || sum_native(&raw);
    let d = || sum_d(&prods);
    let f = || sum_bitsliced::<Eager<N>, N>(&prods, false);
    let p = || sum_bitsliced::<Subst<N>, N>(&prods, false);
    let l = || sum_bitsliced::<LazyFull<N>, N>(&prods, false);
    let pl = || sum_bitsliced::<SubstLazy<N>, N>(&prods, false);
    let p_each = || sum_bitsliced::<Subst<N>, N>(&prods, true);
    let pl_each = || sum_bitsliced::<SubstLazy<N>, N>(&prods, true);
    for (name, g) in [
        ("native", &native as &dyn Fn() -> u128),
        ("D", &d),
        ("F", &f),
        ("P", &p),
        ("L", &l),
        ("PL", &pl),
        ("P each", &p_each),
        ("PL each", &pl_each),
    ] {
        assert_eq!(g(), want, "{name}");
    }
    println!(
        "-- A, {label}: {blocks} blocks x 64 u16 products, {}",
        if dense { "dense carries" } else { "sparse carries" }
    );
    let b = time("native: u64 per lane, plain Rust adds", blocks, runs, "64", None, &native);
    time("D (#389): wrapping low + carry rule + high += carry", blocks, runs, "64", Some(b), &d);
    time("F: low + carry mask -> bit-16 plane, eager flip ripple", blocks, runs, "64", Some(b), &f);
    time("P: F + EMPTY bitmap, assignment into empty planes", blocks, runs, "64", Some(b), &p);
    time("L: F + FULL bitmap, all-ones carry writes zero unread", blocks, runs, "64", Some(b), &l);
    time("PL: both bitmaps, all-ones carry through FULL = state", blocks, runs, "64", Some(b), &pl);
    time("P, normalized every block", blocks, runs, "64", Some(b), &p_each);
    time("PL, normalized every block", blocks, runs, "64", Some(b), &pl_each);
}

/// Section B: the counters alone, one add per step, against per-lane integers.
fn counters(label: &str, adds: &[(usize, u64)], runs: usize) {
    let mut want = [0u128; 64];
    for &(k, m) in adds {
        for (j, w) in want.iter_mut().enumerate() {
            *w += u128::from(m >> j & 1) << k;
        }
    }
    let want_total: u128 = want.iter().sum();
    fn run<C: Counter<N>>(adds: &[(usize, u64)]) -> u128 {
        let mut c = C::new();
        for &(k, m) in adds {
            c.add_at(k, black_box(m)).unwrap();
        }
        c.total()
    }
    let native = || {
        let mut n = Native::new();
        for &(k, m) in adds {
            n.add_at(k, black_box(m));
        }
        n.lanes.iter().map(|&x| u128::from(x)).sum()
    };
    let (f, p, l, pl) = (
        || run::<Eager<N>>(adds),
        || run::<Subst<N>>(adds),
        || run::<LazyFull<N>>(adds),
        || run::<SubstLazy<N>>(adds),
    );
    let fits = adds.iter().all(|&(k, _)| k < 48);
    if fits {
        assert_eq!(native(), want_total, "native");
    }
    for (name, g) in [("F", &f as &dyn Fn() -> u128), ("P", &p), ("L", &l), ("PL", &pl)] {
        assert_eq!(g(), want_total, "{name}");
    }
    println!("-- B, {label}: {} adds", adds.len());
    let b = if fits {
        Some(time("native: 64 lanes += bit << k", adds.len(), runs, "add", None, &native))
    } else {
        None
    };
    time("F eager", adds.len(), runs, "add", b, &f);
    time("P EMPTY substitution", adds.len(), runs, "add", b, &p);
    time("L lazy FULL", adds.len(), runs, "add", b, &l);
    time("PL both", adds.len(), runs, "add", b, &pl);
}

/// Section B': one add whose carry runs through `len` FULL columns. Each rep
/// clones a preset counter (opaque to the compiler) and adds once; a
/// clone-only row per variant gives the cost to subtract. Correctness is
/// checked outside the timed loop.
fn chains(len: usize, runs: usize) {
    const REPS: usize = 1000;
    fn preset<C: Counter<N>>(len: usize) -> C {
        let mut c = C::new();
        for k in 0..len {
            c.add_at(k, u64::MAX).unwrap();
        }
        c
    }
    fn check<C: Counter<N>>(len: usize) {
        let mut c: C = preset(len);
        c.add_at(0, u64::MAX).unwrap();
        assert_eq!(c.total(), 64u128 << len, "chain {len}");
    }
    fn add_rep<C: Counter<N> + Clone>(base: &C) -> u128 {
        for _ in 0..REPS {
            let mut c = black_box(base.clone());
            let _ = c.add_at(0, black_box(u64::MAX));
            black_box(&c);
        }
        0
    }
    fn clone_rep<C: Counter<N> + Clone>(base: &C) -> u128 {
        for _ in 0..REPS {
            let c = black_box(base.clone());
            black_box(&c);
        }
        0
    }
    fn constant_rep<C: Counter<N>>(len: usize) -> u128 {
        let mut acc = 0u128;
        for _ in 0..REPS {
            let mut c: C = preset(len);
            let _ = c.add_at(0, u64::MAX);
            acc += c.total() >> len;
        }
        acc
    }
    check::<Eager<N>>(len);
    check::<Subst<N>>(len);
    check::<LazyFull<N>>(len);
    check::<SubstLazy<N>>(len);
    assert_eq!(constant_rep::<Eager<N>>(len), 64 * REPS as u128);
    assert_eq!(constant_rep::<SubstLazy<N>>(len), 64 * REPS as u128);
    println!("-- B', one add carrying through {len} FULL columns (per rep: clone + add)");
    let (f, p, l, pl): (Eager<N>, Subst<N>, LazyFull<N>, SubstLazy<N>) =
        (preset(len), preset(len), preset(len), preset(len));
    for (name, g) in [
        ("F clone only", &(|| clone_rep(&f)) as &dyn Fn() -> u128),
        ("F clone + add", &|| add_rep(&f)),
        ("P clone only", &|| clone_rep(&p)),
        ("P clone + add", &|| add_rep(&p)),
        ("L clone only", &|| clone_rep(&l)),
        ("L clone + add", &|| add_rep(&l)),
        ("PL clone only", &|| clone_rep(&pl)),
        ("PL clone + add", &|| add_rep(&pl)),
        ("F constant state: build + add + total", &|| constant_rep::<Eager<N>>(len)),
        ("PL constant state: build + add + total", &|| constant_rep::<SubstLazy<N>>(len)),
    ] {
        time(name, REPS, runs, "rep", None, g);
    }
}

/// Section C: R1 (two words) against R2 (one u128), same planes.
fn planes128(runs: usize) {
    let mut s = 99u64;
    let masks: Vec<u64> = (0..4096).map(|_| xorshift(&mut s)).collect();
    let r1 = || {
        let (mut p, mut out) = (Pair128 { lo: 0, hi: 0 }, 0u64);
        for &m in &masks {
            let (np, o) = pair_add(p, m);
            p = np;
            out ^= o;
        }
        pack(p) ^ u128::from(out)
    };
    let r2 = || {
        let (mut x, mut out) = (0u128, 0u64);
        for &m in &masks {
            let (nx, o) = packed_add(x, m);
            x = nx;
            out ^= o;
        }
        x ^ u128::from(out)
    };
    assert_eq!(r1(), r2());
    println!("-- C, two planes in 128 bits: {} adds", masks.len());
    let b = time("R1: two u64 planes", masks.len(), runs, "add", None, &r1);
    time("R2: one u128 carrier", masks.len(), runs, "add", Some(b), &r2);
}

// Out-of-line copies for the disassembly.
#[inline(never)]
pub fn asm_flip(a: u64, b: u64) -> (u64, u64) {
    flip(a, b)
}
#[inline(never)]
pub fn asm_pair_add(p: Pair128, m: u64) -> (Pair128, u64) {
    pair_add(p, m)
}
#[inline(never)]
pub fn asm_packed_add(x: u128, m: u64) -> (u128, u64) {
    packed_add(x, m)
}
#[inline(never)]
pub fn asm_eager_add(c: &mut Eager<N>, m: u64) -> Result<(), Overflow> {
    c.add_at(0, m)
}
#[inline(never)]
pub fn asm_subst_add(c: &mut Subst<N>, m: u64) -> Result<(), Overflow> {
    c.add_at(0, m)
}
#[inline(never)]
pub fn asm_lazy_add(c: &mut LazyFull<N>, m: u64) -> Result<(), Overflow> {
    c.add_at(0, m)
}
#[inline(never)]
pub fn asm_pl_add(c: &mut SubstLazy<N>, m: u64) -> Result<(), Overflow> {
    c.add_at(0, m)
}
#[inline(never)]
pub fn asm_native_add(c: &mut Native, m: u64) {
    c.add_at(0, m)
}
#[inline(never)]
pub fn asm_boundary_f(low: &mut [U16x32; 2], hi: &mut Eager<N>, p: &[U16x32; 2]) {
    let s0 = low[0] + p[0];
    let s1 = low[1] + p[1];
    let m = carry_mask(carry16::carry_after(low[0], p[0], s0), carry16::carry_after(low[1], p[1], s1));
    *low = [s0, s1];
    hi.add_at(0, m).unwrap();
}
#[inline(never)]
pub fn asm_boundary_d(col: &mut [carry16::Col16; 2], p: &[U16x32; 2]) {
    col[0].add::<false>(p[0]);
    col[1].add::<false>(p[1]);
}

fn main() {
    println!("tier: {} (target-cpu as compiled)", tier());
    {
        let m = black_box(0x5555u64);
        let mut e = Eager::<N>::new();
        let mut su = Subst::<N>::new();
        let mut la = LazyFull::<N>::new();
        let mut pl = SubstLazy::<N>::new();
        let mut na = Native::new();
        black_box((
            asm_eager_add(&mut e, m).is_ok(),
            asm_subst_add(&mut su, m).is_ok(),
            asm_lazy_add(&mut la, m).is_ok(),
            asm_pl_add(&mut pl, m).is_ok(),
        ));
        asm_native_add(&mut na, m);
        let mut low = [U16x32::splat(black_box(9)); 2];
        asm_boundary_f(&mut low, &mut e, &[U16x32::splat(black_box(7)); 2]);
        let mut col = [carry16::Col16::zero(); 2];
        asm_boundary_d(&mut col, &[U16x32::splat(black_box(7)); 2]);
        black_box((
            asm_flip(m, m),
            asm_pair_add(black_box(Pair128 { lo: !m, hi: m }), m),
            asm_packed_add(black_box(u128::from(m) << 64 | u128::from(!m)), m),
            na.lanes[0],
            low[0].to_array()[0],
            col[0].total(),
            e.total(),
        ));
    }
    if std::env::args().any(|a| a == "quick") {
        return;
    }
    boundary("L1-resident", 64, 2001, true, 1);
    boundary("L1-resident", 64, 2001, false, 2);
    boundary("streamed", 1 << 20, 5, true, 3);
    boundary("streamed", 1 << 20, 5, false, 4);

    let mut s = 5u64;
    let dense: Vec<(usize, u64)> = (0..4096).map(|_| (0, u64::MAX)).collect();
    let random: Vec<(usize, u64)> = (0..4096).map(|_| (0, xorshift(&mut s))).collect();
    let sparse: Vec<(usize, u64)> = (0..4096)
        .map(|_| (0, 1 << (xorshift(&mut s) % 64)))
        .collect();
    let weighted: Vec<(usize, u64)> = (0..4096)
        .map(|t| ([0, 16, 17, 32][t % 4], xorshift(&mut s)))
        .collect();
    counters("dense: all-ones at 2^0 (binary counting, long FULL runs)", &dense, 2001);
    counters("random masks at 2^0", &random, 2001);
    counters("sparse: one lane at 2^0", &sparse, 2001);
    counters("weighted: random masks entering at 2^0, 2^16, 2^17, 2^32", &weighted, 2001);
    for len in [1, 8, 32, 64] {
        chains(len, 201);
    }
    planes128(2001);
}
