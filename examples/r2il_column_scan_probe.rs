//! Does vectorizing r2sleigh's `OpColumns` scans pay, and where is the crossover?
//!
//! `r2il::columns` was laid out for `ndarray::simd`'s mask surface and says so
//! in its own module docs — while deliberately taking no `ndarray` dependency
//! and declining to claim the SIMD is worth it: *"whether that is worth doing
//! is a profiling question nobody has answered."* This is the consumer-side
//! answer, on a real lift rather than a synthetic stream.
//!
//! Input is a column dump from `r2sleigh-lift`'s `win32_census` example
//! (`WIN32_CENSUS_COLUMNS_OUT=<path>`): `u64 n`, `n×u8` tag, `n×u8` space,
//! `n×u64` offset. Run:
//!
//! ```text
//! cargo run --release --example r2il_column_scan_probe -- <dump>
//! ```
//!
//! # The two queries
//!
//! **Q1 — `find_tag`**: `tag == Call`. One predicate, the simplest scan.
//!
//! **Q2 — `find_ram_in_range`**: the census's IAT/prefetch classification,
//! `space == Ram && lo <= offset < hi`. Four expressible predicates once the
//! u64 offset is split (see below), so it is the conjunction question.
//!
//! # Two primitive gaps this probe had to work around, and they are findings
//!
//! 1. **No `u8` comparator.** `OpColumns::{tag,space}` are `Vec<u8>`; the
//!    facade's narrowest value type is `u32`. A consumer must keep a widened
//!    copy — 4× the bytes of the column it scans, which directly attacks the
//!    memory-traffic argument that motivates the layout.
//! 2. **No `u64` RANGE comparator.** The facade has `ternary_match_u64_to_mask`
//!    (exact, with don't-care bits) but no `ge/lt` over `u64`; the ordered
//!    family stops at `i32`. Measured on the real dump, **100 % of Ram-space
//!    offsets exceed 2³²** (they are image-based: `0x1_4000_105e` …
//!    `0x1_4000_8398`), so narrowing is not sound in general. The query is
//!    re-expressed EXACTLY here by splitting the offset into `hi32`/`lo32`
//!    columns — valid only because the chosen window lies inside one `hi32`
//!    bucket, which the probe asserts rather than assumes. A general range
//!    needs the primitive.
//!
//! # Arms (every arm must produce a bit-identical mask, or the run aborts)
//!
//! | arm | how the conjunction is formed | passes |
//! |---|---|---|
//! | `S` | scalar over the NATIVE `u8`/`u64` columns (no widening at all) | 1 |
//! | `AND` | 4 `*_to_mask` + 3 `mask_and_assign` | 7 |
//! | `TERN` | 4 `*_to_mask` + `mask_ternlog::<AND3>` + `mask_and_assign` | 6 |
//! | `UNDER` | the `_under` chain — each predicate narrows the live mask | 4 |
//!
//! `S` is the honest baseline precisely because it needs no widened columns:
//! charging the SIMD arms for the layout they require is the comparison a
//! consumer actually faces.
//!
//! # Measured 2026-09-14 — Xeon @ 2.10 GHz, avx512f/bw/vl, release, 3 runs
//!
//! Fixture: `probes/win32-census/legacy_app.exe` (PE32+ x86-64, `.text` 7 688 B),
//! **12 408 p-code ops**. Spans above that are the real stream TILED. Every arm
//! agreed bit-for-bit at every span; the numbers below are ns per op.
//!
//! | span | S | AND | TERN | UNDER | S/AND | S/TERN | S/UNDER |
//! |---:|---:|---:|---:|---:|---:|---:|---:|
//! | 1 024 | 0.78 | 0.51 | 0.50 | 0.60 | 1.54 | 1.56 | 1.30 |
//! | 4 096 | 0.76 | 0.54 | 0.53 | 0.57 | 1.42 | 1.45 | 1.33 |
//! | **12 408 (real)** | 0.70 | 0.52 | 0.53 | 0.49 | **1.35** | **1.32** | **1.42** |
//! | 49 632 T | 0.70 | 0.58 | 0.58 | 0.55 | 1.21 | 1.22 | 1.29 |
//! | 198 528 T | 0.87 | 0.89 | 0.88 | 0.67 | 0.98 | 0.99 | 1.31 |
//! | 794 112 T | 0.87 | 1.12 | 1.11 | 0.90 | **0.77** | **0.78** | 0.96 |
//! | 3 176 448 T | 0.91 | 1.22 | 1.13 | 0.95 | **0.75** | **0.80** | 0.95 |
//!
//! Q1 (`find_tag`, one predicate): 2.45× / 2.07× / 1.90× at 256 / 1 K / 4 K,
//! **1.76× at the real 12 408**, 1.58× at 49 K, and **1.00× at 794 K** — the
//! widened column's extra memory traffic eats the whole win.
//!
//! ## Four findings
//!
//! 1. **There is a crossover and it is low.** The mask arms win up to roughly
//!    50 K ops and LOSE from roughly 200 K. Nothing here is a memory-bandwidth
//!    surprise: `S` reads 9 B/op (`u8` + `u64`), the mask arms read 12 B/op of
//!    widened columns and write four mask buffers. The layout's own motivation
//!    — fewer bytes touched — is partly spent paying for the primitives' value
//!    types.
//! 2. **The ternlog fusion is not the lever for this query.** `TERN` and `AND`
//!    are within noise at every span (1.32 vs 1.35 at the real size). The cost
//!    is the four passes over the value columns, not the three mask combines
//!    the fusion removes. Fusion pays where a conjunction is over masks a
//!    caller ALREADY holds; here each predicate must first be computed.
//! 3. **`_under` is the arm that survives scale.** Narrowing the live mask in
//!    place needs no separate combine and no extra buffers, so it is the best
//!    arm at 12 408 (1.42×) and the only one still near parity at 3.2 M.
//! 4. **The ratio is favourable exactly where the absolute time is
//!    irrelevant.** A whole-census `find_ram_in_range` is 8.9 µs scalar and
//!    6.4 µs vectorized: **2.5 µs saved** on a binary whose SLEIGH lift costs
//!    milliseconds. Per the workspace's own rule — a word-level op pays for
//!    the span it is given — this span is not worth paying for.

use std::time::Instant;

use ndarray::simd::ternlog::AND3;
use ndarray::simd::{
    eq_i32_to_mask_under, eq_u32_to_mask, ge_i32_to_mask_under, lt_i32_to_mask, lt_i32_to_mask_under, mask_and_assign,
    mask_ternlog, popcount_batch_u64,
};

const RAM: u8 = 3;
const TAG_CALL: u8 = 4;
/// Window: `[image_base, image_base + 0x2000)` for this fixture's 0x1_4000_0000
/// base — the `.text` head, the shape the census's IAT walk asks about.
const WIN_LO: u64 = 0x1_4000_0000;
const WIN_HI: u64 = 0x1_4000_2000;

fn words(n: usize) -> usize {
    n.div_ceil(64)
}

struct Cols {
    tag: Vec<u8>,
    space: Vec<u8>,
    offset: Vec<u64>,
}

fn load(path: &str) -> Cols {
    let b = std::fs::read(path).expect("read column dump");
    let n = u64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
    let tag = b[8..8 + n].to_vec();
    let space = b[8 + n..8 + 2 * n].to_vec();
    let mut offset = Vec::with_capacity(n);
    for i in 0..n {
        let o = 8 + 2 * n + i * 8;
        offset.push(u64::from_le_bytes(b[o..o + 8].try_into().unwrap()));
    }
    Cols { tag, space, offset }
}

/// Tile the real stream to reach spans a single 130 KB binary cannot supply.
/// Labelled TILED in the output — it preserves the measured distribution but
/// is not more evidence about program shape, only about throughput.
fn take(c: &Cols, n: usize) -> Cols {
    let m = c.tag.len();
    Cols {
        tag: (0..n).map(|i| c.tag[i % m]).collect(),
        space: (0..n).map(|i| c.space[i % m]).collect(),
        offset: (0..n).map(|i| c.offset[i % m]).collect(),
    }
}

// ---- scalar reference arms (native column types, no widening) ----

fn q1_scalar(c: &Cols, out: &mut [u64]) {
    out.fill(0);
    for (i, &t) in c.tag.iter().enumerate() {
        if t == TAG_CALL {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
}

// Spelled exactly as `r2il::OpColumns::find_ram_in_range` spells it — this arm
// is the baseline BECAUSE it mirrors the shipped scalar, so it is not rewritten
// into `Range::contains` for the lint's sake.
#[expect(clippy::manual_range_contains)]
fn q2_scalar(c: &Cols, out: &mut [u64]) {
    out.fill(0);
    for i in 0..c.tag.len() {
        let o = c.offset[i];
        if c.space[i] == RAM && o >= WIN_LO && o < WIN_HI {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: <column-dump>");
    let base = load(&path);
    let real_n = base.tag.len();
    println!("dump: {real_n} ops from {path}\n");

    // Gap 2's soundness precondition, asserted rather than assumed: the window
    // must lie inside one hi32 bucket for the split re-expression to be exact.
    assert_eq!(
        WIN_LO >> 32,
        (WIN_HI - 1) >> 32,
        "window straddles a hi32 boundary; the split decomposition is not exact"
    );

    println!(
        "{:<10} {:>9}  {:>10} {:>10} {:>10} {:>10}   {:>8} {:>8} {:>8}",
        "span", "hits", "S ns/op", "AND", "TERN", "UNDER", "S/AND", "S/TERN", "S/UNDER"
    );

    let mut spans: Vec<(usize, bool)> = vec![256, 1024, 4096]
        .into_iter()
        .map(|n| (n, false))
        .collect();
    spans.push((real_n, false));
    for k in [4usize, 16, 64, 256] {
        spans.push((real_n * k, true));
    }

    for (n, tiled) in spans {
        let c = if n == real_n { load(&path) } else { take(&base, n) };
        let w = words(n);

        // The widened columns the facade's value types force on a consumer.
        let space32: Vec<u32> = c.space.iter().map(|&s| s as u32).collect();
        let tag32: Vec<u32> = c.tag.iter().map(|&t| t as u32).collect();
        let hi32: Vec<i32> = c.offset.iter().map(|&o| (o >> 32) as i32).collect();
        let lo32: Vec<i32> = c.offset.iter().map(|&o| (o as u32) as i32).collect();
        let want_hi = (WIN_LO >> 32) as i32;
        let lo_lo = (WIN_LO as u32) as i32;
        let lo_hi = (WIN_HI as u32) as i32;
        assert!(lo_lo >= 0 && lo_hi > lo_lo, "window lo32 must stay positive");

        let (mut s, mut a, mut t, mut u) = (vec![0u64; w], vec![0u64; w], vec![0u64; w], vec![0u64; w]);
        let (mut m0, mut m1, mut m2, mut m3) = (vec![0u64; w], vec![0u64; w], vec![0u64; w], vec![0u64; w]);

        let build = |m0: &mut [u64], m1: &mut [u64], m2: &mut [u64], m3: &mut [u64]| {
            eq_u32_to_mask(&space32, RAM as u32, m0);
            ndarray::simd::eq_i32_to_mask(&hi32, want_hi, m1);
            ndarray::simd::ge_i32_to_mask(&lo32, lo_lo, m2);
            lt_i32_to_mask(&lo32, lo_hi, m3);
        };

        let and_arm = |a: &mut Vec<u64>, m0: &mut Vec<u64>, m1: &mut Vec<u64>, m2: &mut Vec<u64>, m3: &mut Vec<u64>| {
            build(m0, m1, m2, m3);
            a.copy_from_slice(m0);
            mask_and_assign(a, m1);
            mask_and_assign(a, m2);
            mask_and_assign(a, m3);
        };
        let tern_arm =
            |t: &mut Vec<u64>, m0: &mut Vec<u64>, m1: &mut Vec<u64>, m2: &mut Vec<u64>, m3: &mut Vec<u64>| {
                build(m0, m1, m2, m3);
                mask_ternlog::<AND3>(m0, m1, m2, t);
                mask_and_assign(t, m3);
            };
        let under_arm = |u: &mut Vec<u64>, m0: &mut Vec<u64>, m1: &mut Vec<u64>| {
            eq_u32_to_mask(&space32, RAM as u32, u);
            eq_i32_to_mask_under(&hi32, want_hi, u, m0);
            ge_i32_to_mask_under(&lo32, lo_lo, m0, m1);
            lt_i32_to_mask_under(&lo32, lo_hi, m1, u);
        };

        // ---- correctness gate: every arm, bit for bit, before any timing ----
        q2_scalar(&c, &mut s);
        and_arm(&mut a, &mut m0, &mut m1, &mut m2, &mut m3);
        tern_arm(&mut t, &mut m0, &mut m1, &mut m2, &mut m3);
        under_arm(&mut u, &mut m0, &mut m1);
        assert_eq!(s, a, "AND arm disagrees at n={n}");
        assert_eq!(s, t, "TERN arm disagrees at n={n}");
        assert_eq!(s, u, "UNDER arm disagrees at n={n}");
        let hits: u64 = popcount_batch_u64(&s);

        // ---- timing ----
        let reps = (40_000_000 / n).max(20);
        macro_rules! bench {
            ($body:expr) => {{
                $body;
                let t0 = Instant::now();
                for _ in 0..reps {
                    $body;
                }
                t0.elapsed().as_secs_f64() / (reps as f64) / (n as f64) * 1e9
            }};
        }

        let ns_s = bench!(q2_scalar(&c, &mut s));
        let ns_a = bench!(and_arm(&mut a, &mut m0, &mut m1, &mut m2, &mut m3));
        let ns_t = bench!(tern_arm(&mut t, &mut m0, &mut m1, &mut m2, &mut m3));
        let ns_u = bench!(under_arm(&mut u, &mut m0, &mut m1));

        println!(
            "{:<10} {:>9}  {:>10.4} {:>10.4} {:>10.4} {:>10.4}   {:>8.2} {:>8.2} {:>8.2}",
            format!("{}{}", n, if tiled { "T" } else { "" }),
            hits,
            ns_s,
            ns_a,
            ns_t,
            ns_u,
            ns_s / ns_a,
            ns_s / ns_t,
            ns_s / ns_u
        );

        // Q1, the single-predicate scan, at the same span.
        let mut q1s = vec![0u64; w];
        let mut q1v = vec![0u64; w];
        q1_scalar(&c, &mut q1s);
        eq_u32_to_mask(&tag32, TAG_CALL as u32, &mut q1v);
        assert_eq!(q1s, q1v, "Q1 arms disagree at n={n}");
        let q1_hits = popcount_batch_u64(&q1s);
        let q1_ns_s = bench!(q1_scalar(&c, &mut q1s));
        let q1_ns_v = bench!(eq_u32_to_mask(&tag32, TAG_CALL as u32, &mut q1v));
        println!(
            "{:<10} {:>9}  {:>10.4} {:>10.4}   <- Q1 find_tag: scalar u8 vs eq_u32 over a widened column, ratio {:.2}",
            "",
            q1_hits,
            q1_ns_s,
            q1_ns_v,
            q1_ns_s / q1_ns_v
        );
    }
}
