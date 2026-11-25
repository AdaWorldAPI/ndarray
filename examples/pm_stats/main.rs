//! D-PM-STATS: cost of `P × M` statistical folds, signed comparison and
//! sign-partitioned products, against conventional signed lanes with an
//! equal-semantics validity plane. Correctness lives in `tests/pm_stats.rs`.
//!
//!   cargo run --release --example pm_stats --features std
//!
//! Reports ns per 64 lanes (best of several timed repetitions). The tier is
//! whatever this build targets; the header prints it.
#[allow(dead_code)]
mod stats;

use std::hint::black_box;
use std::time::Instant;

use ndarray::hpc::abakus::pm::Pm;
use ndarray::simd::masked_sum_i32;
use stats::*;

struct Data {
    mags: Vec<u8>,
    pos: Vec<u64>,
    sel: Vec<u64>,
    /// The same values as `i16` and `i32`, with a validity plane already
    /// intersected with the selection (the baseline gets that for free).
    v16: Vec<i16>,
    v32: Vec<i32>,
    live: Vec<u64>,
}

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

#[derive(Clone, Copy)]
enum Mix {
    Mixed,
    Positive,
    Untouched,
}

fn data(blocks: usize, density_pct: u64, mix: Mix, seed: u64) -> Data {
    let mut s = seed;
    let n = blocks * 64;
    let (mut mags, mut v16, mut v32) = (vec![0u8; n], vec![0i16; n], vec![0i32; n]);
    let (mut pos, mut sel, mut live) = (vec![0u64; blocks], vec![0u64; blocks], vec![0u64; blocks]);
    for b in 0..blocks {
        for j in 0..64 {
            let i = b * 64 + j;
            let r = xorshift(&mut s);
            let (m, p) = match mix {
                Mix::Mixed => ((r % 256) as u8, r >> 9 & 1 == 1),
                Mix::Positive => ((r % 256) as u8, true),
                Mix::Untouched => (0, false),
            };
            let picked = xorshift(&mut s) % 100 < density_pct;
            mags[i] = m;
            pos[b] |= u64::from(p) << j;
            sel[b] |= u64::from(picked) << j;
            let touched = p || m != 0;
            let v = if p { i16::from(m) } else { -i16::from(m) };
            v16[i] = v;
            v32[i] = i32::from(v);
            live[b] |= u64::from(picked && touched) << j;
        }
    }
    Data {
        mags,
        pos,
        sel,
        v16,
        v32,
        live,
    }
}

/// Best ns per 64 lanes over `reps` timed passes of `f`.
fn time<R>(blocks: usize, mut f: impl FnMut() -> R) -> f64 {
    let passes = (1usize << 22).div_ceil(blocks * 64).max(1);
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = Instant::now();
        for _ in 0..passes {
            black_box(f());
        }
        let ns = t.elapsed().as_nanos() as f64 / (passes * blocks) as f64;
        best = best.min(ns);
    }
    best
}

// ---------- the arms ----------

fn s_split(d: &Data) -> i64 {
    let mut s = 0i64;
    for (b, &m) in d.sel.iter().enumerate() {
        let x = block_at(&d.mags, &d.pos, b);
        s += msum(m & x.pos, x.mag) as i64 - msum(m & !x.pos, x.mag) as i64;
    }
    s
}

fn s_total_minus_neg(d: &Data) -> i64 {
    let mut s = 0i64;
    for (b, &m) in d.sel.iter().enumerate() {
        let x = block_at(&d.mags, &d.pos, b);
        s += msum(m, x.mag) as i64 - 2 * msum(m & !x.pos, x.mag) as i64;
    }
    s
}

fn s_i32(d: &Data) -> i64 {
    masked_sum_i32(&d.v32, &d.live)
}

fn s_i16(d: &Data) -> i64 {
    let mut s = 0i64;
    for (b, &m) in d.live.iter().enumerate() {
        for j in 0..64 {
            s += i64::from(d.v16[b * 64 + j]) * (m >> j & 1) as i64;
        }
    }
    s
}

fn all_i16(d: &Data) -> Stats {
    let mut st = Stats::default();
    for (b, &m) in d.live.iter().enumerate() {
        for j in 0..64 {
            let k = (m >> j & 1) as i64;
            let v = i64::from(d.v16[b * 64 + j]);
            st.n += k as u64;
            st.s += v * k;
            st.q += (v * v * k) as u64;
            st.a += (v.abs() * k) as u64;
        }
    }
    st
}

fn lt_pm(d: &Data, e: &Data) -> u32 {
    let mut c = 0;
    for b in 0..d.sel.len() {
        c += lt(block_at(&d.mags, &d.pos, b), block_at(&e.mags, &e.pos, b))
            .is
            .count_ones();
    }
    c
}

fn lt_zero_pm(d: &Data) -> u32 {
    (0..d.sel.len())
        .map(|b| lt_const(block_at(&d.mags, &d.pos, b), 0).is.count_ones())
        .sum()
}

fn lt_i16(d: &Data, e: &Data) -> u32 {
    let mut c = 0;
    for b in 0..d.live.len() {
        let both = d.live[b] & e.live[b];
        let mut r = 0u64;
        for j in 0..64 {
            r |= u64::from(d.v16[b * 64 + j] < e.v16[b * 64 + j]) << j;
        }
        c += (r & both).count_ones();
    }
    c
}

/// The fair SIMD baseline: two `I16x32` compares per 64 lanes.
fn lt_i16_simd(d: &Data, e: &Data) -> u32 {
    use ndarray::simd::I16x32;
    let mut c = 0;
    for b in 0..d.live.len() {
        let at = |v: &[i16], h: usize| I16x32::from_slice(&v[b * 64 + h * 32..b * 64 + h * 32 + 32]);
        let lo = at(&e.v16, 0).cmp_gt(at(&d.v16, 0));
        let hi = at(&e.v16, 1).cmp_gt(at(&d.v16, 1));
        c += ((u64::from(lo) | u64::from(hi) << 32) & d.live[b] & e.live[b]).count_ones();
    }
    c
}

fn dot_pm(d: &Data, e: &Data) -> i64 {
    let mut s = 0i64;
    for b in 0..d.sel.len() {
        let (x, y) = (block_at(&d.mags, &d.pos, b), block_at(&e.mags, &e.pos, b));
        s += dot_total_minus_twice_opp(&[x], &[y], &[u64::MAX]);
    }
    s
}

fn dot_i16(d: &Data, e: &Data) -> i64 {
    let mut s = 0i64;
    for i in 0..d.v16.len() {
        s += i64::from(d.v16[i]) * i64::from(e.v16[i]);
    }
    s
}

fn main() {
    println!(
        "D-PM-STATS  avx512f={} avx2={}  (ns per 64 lanes, best of 7)",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2")
    );
    // Bytes per 64 lanes each arm reads.
    println!("bytes/64: P×M 72 (+8 sel) | i16 128 (+8 live) | i32 256 (+8 live)\n");

    println!("[1] signed sum S, 1024 blocks (64K lanes), by density and polarity mix");
    println!(
        "{:<10} {:>4}  {:>8} {:>8} {:>8} {:>8} {:>8}",
        "mix", "sel%", "split", "M-2neg", "foldS", "i32lib", "i16"
    );
    for (name, mix) in [("mixed", Mix::Mixed), ("positive", Mix::Positive), ("untouched", Mix::Untouched)] {
        for dens in [0, 1, 10, 50, 100] {
            let d = data(1024, dens, mix, 7);
            let want = s_i16(&d);
            assert_eq!((s_split(&d), s_total_minus_neg(&d), s_i32(&d)), (want, want, want));
            assert_eq!(fold_soa(&d.mags, &d.pos, &d.sel, Demand::SUM).s, want);
            println!(
                "{:<10} {:>4}  {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8.2}",
                name,
                dens,
                time(1024, || s_split(&d)),
                time(1024, || s_total_minus_neg(&d)),
                time(1024, || fold_soa(&d.mags, &d.pos, &d.sel, Demand::SUM).s),
                time(1024, || s_i32(&d)),
                time(1024, || s_i16(&d)),
            );
        }
    }

    println!("\n[1b] S and A together: horizontal per block vs vertical per tile, mixed, 50% selection");
    println!("{:<12} {:>10} {:>10}", "lanes", "horiz", "vertical");
    for blocks in [128usize, 1024, 262_144] {
        let d = data(blocks, 50, Mix::Mixed, 13);
        let h = fold_soa(
            &d.mags,
            &d.pos,
            &d.sel,
            Demand {
                n: false,
                s: true,
                q: false,
                a: true,
            },
        );
        assert_eq!(sum_abs_vertical(&d.mags, &d.pos, &d.sel), (h.s, h.a));
        println!(
            "{:<12} {:>10.2} {:>10.2}",
            blocks * 64,
            time(blocks, || fold_soa(
                &d.mags,
                &d.pos,
                &d.sel,
                Demand {
                    n: false,
                    s: true,
                    q: false,
                    a: true
                }
            )),
            time(blocks, || sum_abs_vertical(&d.mags, &d.pos, &d.sel)),
        );
    }

    println!("\n[1] all four statistics (N, S, Q, A), mixed, 50% selection, by working set");
    println!("{:<12} {:>10} {:>10}", "lanes", "fold", "i16");
    for blocks in [1usize, 1024, 262_144] {
        let d = data(blocks, 50, Mix::Mixed, 9);
        assert_eq!(fold_soa(&d.mags, &d.pos, &d.sel, Demand::ALL), all_i16(&d));
        println!(
            "{:<12} {:>10.2} {:>10.2}",
            blocks * 64,
            time(blocks, || fold_soa(&d.mags, &d.pos, &d.sel, Demand::ALL)),
            time(blocks, || all_i16(&d)),
        );
    }

    println!("\n[2] a < b and x < 0, mixed, 100% selection");
    println!("{:<12} {:>10} {:>10} {:>10} {:>10}", "lanes", "lt P×M", "lt i16simd", "lt i16loop", "x<0 P×M");
    for blocks in [1024usize, 262_144] {
        let (d, e) = (data(blocks, 100, Mix::Mixed, 3), data(blocks, 100, Mix::Mixed, 4));
        assert_eq!(lt_pm(&d, &e), lt_i16(&d, &e));
        assert_eq!(lt_pm(&d, &e), lt_i16_simd(&d, &e));
        println!(
            "{:<12} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            blocks * 64,
            time(blocks, || lt_pm(&d, &e)),
            time(blocks, || lt_i16_simd(&d, &e)),
            time(blocks, || lt_i16(&d, &e)),
            time(blocks, || lt_zero_pm(&d)),
        );
    }

    println!("\n[3] signed dot product, mixed, all lanes");
    println!("{:<12} {:>10} {:>10}", "lanes", "P×M", "i16");
    for blocks in [1024usize, 262_144] {
        let (d, e) = (data(blocks, 100, Mix::Mixed, 5), data(blocks, 100, Mix::Mixed, 6));
        assert_eq!(dot_pm(&d, &e), dot_i16(&d, &e));
        println!(
            "{:<12} {:>10.2} {:>10.2}",
            blocks * 64,
            time(blocks, || dot_pm(&d, &e)),
            time(blocks, || dot_i16(&d, &e)),
        );
    }
    let _: Option<Pm> = None;
}
