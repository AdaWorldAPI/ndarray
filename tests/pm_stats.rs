#![cfg(feature = "std")]
#![allow(clippy::needless_range_loop)]
//! D-PM-STATS probes (`examples/pm_stats/stats.rs`) against independent
//! scalar oracles: every 512 × 512 codeword pair for comparison, selection and
//! products; random and boundary blocks for the statistical folds.

#[path = "../examples/pm_stats/stats.rs"]
#[allow(dead_code)]
mod stats;

use ndarray::hpc::abakus::pm::Pm;
use stats::*;

/// Every (a, b) codeword pair, 64 pairs to a block.
fn all_pairs() -> Vec<(Pm, Pm, [(u16, u16); 64])> {
    let codes: Vec<(u16, u16)> = (0..512u16).flat_map(|a| (0..512u16).map(move |b| (a, b))).collect();
    codes
        .chunks(64)
        .map(|ch| {
            let mut pairs = [(0u16, 0u16); 64];
            pairs.copy_from_slice(ch);
            let a: Vec<u16> = ch.iter().map(|p| p.0).collect();
            let b: Vec<u16> = ch.iter().map(|p| p.1).collect();
            (block_of(&a), block_of(&b), pairs)
        })
        .collect()
}

fn bit(m: u64, i: usize) -> bool {
    m >> i & 1 == 1
}

// ---------- Probe 2: comparison and selection, exhaustive ----------

#[test]
fn comparisons_match_the_oracle_on_every_codeword_pair() {
    let mut checked = 0usize;
    for (a, b, pairs) in all_pairs() {
        let (l, e, q) = (lt(a, b), eq(a, b), le(a, b));
        for i in 0..64 {
            let (va, vb) = (value_of(pairs[i].0), value_of(pairs[i].1));
            match (va, vb) {
                (Some(x), Some(y)) => {
                    assert!(!bit(l.unknown, i));
                    assert_eq!(bit(l.is, i), x < y, "lt {x} {y}");
                    assert_eq!(bit(e.is, i), x == y, "eq {x} {y}");
                    assert_eq!(bit(q.is, i), x <= y, "le {x} {y}");
                }
                _ => {
                    assert!(bit(l.unknown, i) && bit(e.unknown, i), "{va:?} {vb:?} must be unknown");
                    assert!(!bit(l.is, i) && !bit(e.is, i) && !bit(q.is, i));
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 512 * 512);
}

#[test]
fn min_and_max_match_the_oracle_on_every_codeword_pair() {
    for (a, b, pairs) in all_pairs() {
        let (lo, hi) = (min(a, b), max(a, b));
        for i in 0..64 {
            let (va, vb) = (value_of(pairs[i].0), value_of(pairs[i].1));
            let want = va.zip(vb);
            assert_eq!(lane(lo, i), want.map(|(x, y)| x.min(y)), "min {va:?} {vb:?}");
            assert_eq!(lane(hi, i), want.map(|(x, y)| x.max(y)), "max {va:?} {vb:?}");
        }
    }
}

/// A constant comparison takes its own path; it must agree with the general
/// two-operand one for every codeword and every constant in range.
#[test]
fn constant_comparison_agrees_with_the_oracle() {
    let codes: Vec<u16> = (0..512).collect();
    for c in -255i16..=255 {
        for ch in codes.chunks(64) {
            let x = block_of(ch);
            let r = lt_const(x, c);
            for (i, &code) in ch.iter().enumerate() {
                match value_of(code) {
                    Some(v) => {
                        assert!(!bit(r.unknown, i));
                        assert_eq!(bit(r.is, i), v < i32::from(c), "{v} < {c}");
                    }
                    None => assert!(bit(r.unknown, i) && !bit(r.is, i)),
                }
            }
        }
    }
}

/// Numerical equality is not codeword identity for untouched lanes: two
/// untouched operands are unknown, never equal.
#[test]
fn two_untouched_lanes_are_unknown_not_equal() {
    let u = block_of(&[0u16; 64]);
    let e = eq(u, u);
    assert_eq!((e.is, e.unknown), (0, u64::MAX));
    let z = block_of(&[256u16; 64]); // +0 everywhere
    assert_eq!(eq(z, z).is, u64::MAX, "+0 = +0");
}

// ---------- Probe 3: products, exhaustive ----------

#[test]
fn sign_partitioned_products_match_the_oracle_on_every_codeword_pair() {
    let blocks = all_pairs();
    let a: Vec<Pm> = blocks.iter().map(|t| t.0).collect();
    let b: Vec<Pm> = blocks.iter().map(|t| t.1).collect();
    let sel = vec![u64::MAX; a.len()];
    let want: i64 = blocks
        .iter()
        .flat_map(|t| t.2.iter())
        .filter_map(|&(x, y)| Some(i64::from(value_of(x)?) * i64::from(value_of(y)?)))
        .sum();
    assert_eq!(dot_split(&a, &b, &sel), want);
    assert_eq!(dot_total_minus_twice_opp(&a, &b, &sel), want);
    // Per block too: one-sided errors could cancel in the total.
    for t in &blocks {
        let w: i64 = t.2.iter().filter_map(|&(x, y)| Some(i64::from(value_of(x)?) * i64::from(value_of(y)?))).sum();
        assert_eq!(dot_split(&[t.0], &[t.1], &[u64::MAX]), w);
    }
}

#[test]
fn prod_sum_is_exact_at_the_magnitude_bound() {
    use ndarray::simd::U8x64;
    let m = U8x64::splat(255);
    assert_eq!(prod_sum(m, m), 64 * 255 * 255);
}

// ---------- Probe 1: statistical folds ----------

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn random_blocks(n: usize, seed: u64) -> (Vec<Pm>, Vec<u64>, Vec<[u16; 64]>) {
    let mut s = seed;
    let mut xs = Vec::new();
    let mut sel = Vec::new();
    let mut codes = Vec::new();
    for _ in 0..n {
        let mut c = [0u16; 64];
        for v in &mut c {
            *v = (xorshift(&mut s) % 512) as u16;
        }
        xs.push(block_of(&c));
        sel.push(xorshift(&mut s));
        codes.push(c);
    }
    (xs, sel, codes)
}

fn oracle(sel: &[u64], codes: &[[u16; 64]]) -> Stats {
    let mut st = Stats::default();
    for (m, c) in sel.iter().zip(codes) {
        for i in 0..64 {
            if bit(*m, i) {
                if let Some(v) = value_of(c[i]) {
                    st.n += 1;
                    st.s += i64::from(v);
                    st.q += (i64::from(v) * i64::from(v)) as u64;
                    st.a += u64::from(v.unsigned_abs());
                }
            }
        }
    }
    st
}

#[test]
fn statistics_match_the_oracle_on_random_blocks() {
    for seed in 1..=8u64 {
        let (xs, sel, codes) = random_blocks(257, seed);
        let want = oracle(&sel, &codes);
        assert_eq!(fold(&xs, &sel, Demand::ALL), want, "seed {seed}");
        assert_eq!(sum_split(&xs, &sel), want.s);
        assert_eq!(sum_total_minus_twice_neg(&xs, &sel), want.s);
    }
}

#[test]
fn fold_computes_only_what_is_demanded() {
    let (xs, sel, codes) = random_blocks(9, 3);
    let want = oracle(&sel, &codes);
    assert_eq!(fold(&xs, &sel, Demand::SUM), Stats { s: want.s, ..Stats::default() });
}

#[test]
fn boundary_populations() {
    let all = [u64::MAX];
    // Entirely untouched: no values, and variance is undefined, not zero.
    let u = block_of(&[0u16; 64]);
    let st = fold(&[u], &all, Demand::ALL);
    assert_eq!(st, Stats::default());
    assert_eq!(variance_parts(st), None);
    // Only established zeros: 64 values, sum 0, variance exactly 0.
    let z = block_of(&[256u16; 64]);
    let st = fold(&[z], &all, Demand::ALL);
    assert_eq!(st, Stats { n: 64, ..Stats::default() });
    assert_eq!(variance_parts(st), Some((0, 64 * 64)));
    // Full cancellation: +255 and -255 alternating.
    let codes: Vec<u16> = (0..64).map(|i| if i % 2 == 0 { 256 + 255 } else { 255 }).collect();
    let st = fold(&[block_of(&codes)], &all, Demand::ALL);
    assert_eq!((st.n, st.s, st.a, st.q), (64, 0, 64 * 255, 64 * 255 * 255));
    // Nearly constant: 63 × +200 and one +201; variance is 63/4096.
    let mut codes = vec![256 + 200u16; 64];
    codes[17] = 256 + 201;
    let st = fold(&[block_of(&codes)], &all, Demand::ALL);
    assert_eq!(variance_parts(st), Some((63, 4096)));
}

#[test]
fn selection_density_extremes() {
    let (xs, _, codes) = random_blocks(33, 11);
    for sel in [vec![0u64; 33], vec![u64::MAX; 33]] {
        assert_eq!(fold(&xs, &sel, Demand::ALL), oracle(&sel, &codes));
    }
}
