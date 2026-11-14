//! D-U64X64-NNUE-1 lab benchmark for route G (`nnue.rs`): full refresh
//! against incremental update, scalar against `I32x16`, on one bound
//! 64x64 tile. Exactness lives in `tests/u64x64_nnue.rs`; this binary only
//! re-checks parity once per workload before timing it.
//!
//! Run: `cargo run --release --example u64x64_nnue --features std`.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

mod fixture;
mod harness;
mod nnue;

use fixture::BitTile;
use harness::{assert_no_alloc, bb, measure, tier_line, timer_overhead};
use nnue::*;

#[global_allocator]
static A: harness::Counting = harness::Counting;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn weights(seed: u64) -> CellWeights {
    let mut s = seed | 1;
    let mut w = vec![[[0i16; H]; 64]; 64].into_boxed_slice();
    for row in w.iter_mut().flatten().flatten() {
        *row = ((xorshift(&mut s) % 65_535) as i32 - 32_767) as i16;
    }
    CellWeights::new([0; H], w).unwrap()
}

fn tile(s: &mut u64, num: u64, den: u64) -> BitTile {
    let mut t = [0u64; 64];
    for word in &mut t {
        for r in 0..64 {
            if xorshift(s) % den < num {
                *word |= 1u64 << r;
            }
        }
    }
    t
}

/// `n` distinct cells: scattered over the tile, or packed into word 17.
fn flips(s: &mut u64, n: usize, packed: bool) -> BitTile {
    let mut t = [0u64; 64];
    if packed {
        t[17] = if n == 64 { u64::MAX } else { (1u64 << n) - 1 };
        return t;
    }
    let mut placed = 0;
    while placed < n {
        let c = (xorshift(s) % 4096) as usize;
        if t[c / 64] >> (c % 64) & 1 == 0 {
            t[c / 64] |= 1u64 << (c % 64);
            placed += 1;
        }
    }
    t
}

fn xor(a: &BitTile, b: &BitTile) -> BitTile {
    let mut t = [0u64; 64];
    for k in 0..64 {
        t[k] = a[k] ^ b[k];
    }
    t
}

fn popcount(t: &BitTile) -> u32 {
    t.iter().map(|w| w.count_ones()).sum()
}

const REPS: usize = 2_000;
const SAMPLES: usize = 31;

fn main() {
    println!("{}", tier_line());
    let oh = timer_overhead(SAMPLES);
    println!("timer overhead: {:.2} ns per call", oh.median);
    println!(
        "resident: weights {} B (4,096 rows x {H} i16), tile 512 B, accumulator {} B; scratch 0 B",
        4096 * H * 2,
        H * 4
    );
    let w = weights(0x9E37_79B9_7F4A_7C15);
    let mut s = 0x1234_5678_9ABC_DEF1;

    println!(
        "\n## refresh (ns per tile, median of {SAMPLES} samples x {REPS} calls; the four variants interleaved per sample, order rotated)"
    );
    println!("| tile | set bits | scalar | I32x16 x1 | I32x16 x2 | I32x16 x4 | x1 / x2 | x2 / x4 |");
    println!("|---|---|---|---|---|---|---|---|");
    let mut half_refresh = (0.0, 0.0);
    let variants: [fn(&CellWeights, &BitTile) -> Acc; 4] =
        [nnue_refresh, nnue_refresh_simd, nnue_refresh_simd2, nnue_refresh_simd4];
    for (name, num, den) in
        [("empty", 0, 1), ("1/64", 1, 64), ("1/4", 1, 4), ("1/2", 1, 2), ("3/4", 3, 4), ("full", 1, 1)]
    {
        let t = tile(&mut s, num, den);
        for f in &variants[1..] {
            assert_eq!(nnue_refresh(&w, &t), f(&w, &t), "refresh parity");
        }
        assert_no_alloc("refresh", || {
            for f in &variants {
                bb(f(&w, &t));
            }
        });
        let mut samples = [[0f64; SAMPLES]; 4];
        for (n, sample) in (0..SAMPLES).enumerate() {
            for v in 0..4 {
                let v = (v + sample) % 4;
                let f = variants[v];
                let t0 = std::time::Instant::now();
                for _ in 0..REPS {
                    bb(f(bb(&w), bb(&t)));
                }
                samples[v][n] = t0.elapsed().as_nanos() as f64 / REPS as f64;
            }
        }
        let med = samples.map(|x| harness::median(&x));
        if name == "1/2" {
            half_refresh = (med[0], med[1]);
        }
        println!(
            "| {name} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {:.2} | {:.2} |",
            popcount(&t),
            med[0],
            med[1],
            med[2],
            med[3],
            med[1] / med[2],
            med[2] / med[3]
        );
    }

    println!("\n## update on a half-full tile (ns per update; one sample = {REPS} x (old->new, new->old), halved)");
    println!("| changed bits | placement | scalar | I32x16 | refresh / update (I32x16) |");
    println!("|---|---|---|---|---|");
    let base = tile(&mut s, 1, 2);
    for (n, packed) in [(1, false), (8, false), (8, true), (64, false), (64, true)] {
        let d = flips(&mut s, n, packed);
        let next = xor(&base, &d);
        let mut acc = nnue_refresh(&w, &base);
        nnue_update_simd(&w, &mut acc, &base, &next);
        assert_eq!(acc, nnue_refresh(&w, &next), "update parity");
        assert_no_alloc("update", || {
            nnue_update(&w, &mut acc, &next, &base);
            nnue_update_simd(&w, &mut acc, &base, &next);
        });
        let mut a1 = nnue_refresh(&w, &base);
        let a = measure(
            REPS,
            SAMPLES,
            || {},
            || {
                nnue_update(bb(&w), &mut a1, bb(&base), bb(&next));
                nnue_update(bb(&w), &mut a1, bb(&next), bb(&base));
            },
        );
        let mut a2 = nnue_refresh(&w, &base);
        let b = measure(
            REPS,
            SAMPLES,
            || {},
            || {
                nnue_update_simd(bb(&w), &mut a2, bb(&base), bb(&next));
                nnue_update_simd(bb(&w), &mut a2, bb(&next), bb(&base));
            },
        );
        bb((a1, a2));
        let (us, ub) = (a.median / 2.0, b.median / 2.0);
        println!(
            "| {n} | {} | {us:.1} | {ub:.1} | {:.0} |",
            if packed { "one word" } else { "scattered" },
            half_refresh.1 / ub
        );
    }

    // one cell toggled over and over
    let mut t = base;
    let mut acc = nnue_refresh(&w, &t);
    let r = measure(
        REPS,
        SAMPLES,
        || {},
        || {
            let mut next = t;
            next[42] ^= 1u64 << 7;
            nnue_update_simd(bb(&w), &mut acc, bb(&t), bb(&next));
            t = next;
        },
    );
    assert_eq!(acc, nnue_refresh(&w, &t), "repeated toggle parity");
    println!("| 1 | same cell, repeated | - | {:.1} | {:.0} |", r.median, half_refresh.1 / r.median);
    println!("\nhalf-tile refresh: scalar {:.1} ns, I32x16 {:.1} ns", half_refresh.0, half_refresh.1);
}
