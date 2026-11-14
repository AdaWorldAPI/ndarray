#![allow(clippy::needless_range_loop)]
//! Integration tests for the masked arg-min / arg-max folds over i32 lanes.
//!
//! Contract under test: mask bit i is bit (i % 64) of word (i / 64); bits at or
//! past `values.len()` are ignored; the result is (position, value) of the
//! extreme among selected rows, `None` when no row is selected; among equal
//! extremes the lowest position wins (min and max alike).

use ndarray::simd::{masked_argmax_i32, masked_argmin_i32, masked_argminmax_i32, masked_max_i32, masked_min_i32};

type Pair = ((usize, i32), (usize, i32));

const LENS: [usize; 19] = [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1000, 65_536];

// ---------------------------------------------------------------- helpers

/// SplitMix64 deterministic generator.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Inclusive range.
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi as i64 - lo as i64 + 1) as u64;
        (lo as i64 + (self.next() % span) as i64) as i32
    }
    fn full_i32(&mut self) -> i32 {
        self.next() as u32 as i32
    }
}

fn mask_from(n: usize, pred: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut m = vec![0u64; n.div_ceil(64)];
    for i in 0..n {
        if pred(i) {
            m[i / 64] |= 1u64 << (i % 64);
        }
    }
    m
}

fn bit(mask: &[u64], i: usize) -> bool {
    (mask[i / 64] >> (i % 64)) & 1 == 1
}

/// Independent scalar oracle; never calls the functions under test.
fn oracle(values: &[i32], mask: &[u64]) -> Option<Pair> {
    let mut mn: Option<(usize, i32)> = None;
    let mut mx: Option<(usize, i32)> = None;
    for i in 0..values.len() {
        if (mask[i / 64] >> (i % 64)) & 1 == 1 {
            let v = values[i];
            match mn {
                None => mn = Some((i, v)),
                Some((_, h)) => {
                    if v < h {
                        mn = Some((i, v));
                    }
                }
            }
            match mx {
                None => mx = Some((i, v)),
                Some((_, h)) => {
                    if v > h {
                        mx = Some((i, v));
                    }
                }
            }
        }
    }
    match (mn, mx) {
        (Some(a), Some(b)) => Some((a, b)),
        _ => None,
    }
}

fn check(label: &str, values: &[i32], mask: &[u64]) {
    let want = oracle(values, mask);
    let n = values.len();
    assert_eq!(masked_argmin_i32(values, mask), want.map(|p| p.0), "argmin mismatch: n={n} case={label}");
    assert_eq!(masked_argmax_i32(values, mask), want.map(|p| p.1), "argmax mismatch: n={n} case={label}");
    assert_eq!(masked_argminmax_i32(values, mask), want, "argminmax mismatch: n={n} case={label}");
    assert_eq!(masked_min_i32(values, mask), want.map(|p| p.0 .1), "masked_min mismatch: n={n} case={label}");
    assert_eq!(masked_max_i32(values, mask), want.map(|p| p.1 .1), "masked_max mismatch: n={n} case={label}");
}

fn gen_values(n: usize, seed: u64) -> Vec<i32> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.range(-500, 500)).collect()
}

fn random_bits(n: usize, seed: u64, one_in: u64) -> Vec<bool> {
    let mut r = Rng::new(seed);
    (0..n).map(|_| r.next().is_multiple_of(one_in)).collect()
}

/// The standard family of masks used by tests 6 and 15.
fn mask_family(n: usize) -> Vec<(&'static str, Vec<u64>)> {
    let half = random_bits(n, 0xA11CE ^ n as u64, 2);
    let sparse = random_bits(n, 0xB0B ^ n as u64, 64);
    vec![
        ("all-ones", mask_from(n, |_| true)),
        ("all-zeros", mask_from(n, |_| false)),
        ("every-2nd", mask_from(n, |i| i % 2 == 0)),
        ("every-7th", mask_from(n, |i| i % 7 == 0)),
        ("every-97th", mask_from(n, |i| i % 97 == 0)),
        ("random-half", mask_from(n, |i| half[i])),
        ("random-1/64", mask_from(n, |i| sparse[i])),
        ("only-last", mask_from(n, |i| i == n - 1)),
        ("only-first", mask_from(n, |i| i == 0)),
    ]
}

/// Bits [base, base+len) of `mask`, re-based to start at bit 0.
fn slice_mask(mask: &[u64], base: usize, len: usize) -> Vec<u64> {
    let mut out = vec![0u64; len.div_ceil(64)];
    for i in 0..len {
        if bit(mask, base + i) {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    out
}

fn better_min(a: (usize, i32), b: (usize, i32)) -> (usize, i32) {
    if b.1 < a.1 || (b.1 == a.1 && b.0 < a.0) {
        b
    } else {
        a
    }
}

fn better_max(a: (usize, i32), b: (usize, i32)) -> (usize, i32) {
    if b.1 > a.1 || (b.1 == a.1 && b.0 < a.0) {
        b
    } else {
        a
    }
}

fn merge(a: Option<Pair>, b: Option<Pair>) -> Option<Pair> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => Some((better_min(x.0, y.0), better_max(x.1, y.1))),
    }
}

fn partials(values: &[i32], mask: &[u64], seg: usize) -> Vec<Option<Pair>> {
    let mut out = Vec::new();
    let mut base = 0;
    while base < values.len() {
        let len = seg.min(values.len() - base);
        let m = slice_mask(mask, base, len);
        let p = masked_argminmax_i32(&values[base..base + len], &m)
            .map(|((pa, va), (pb, vb))| ((pa + base, va), (pb + base, vb)));
        out.push(p);
        base += len;
    }
    out
}

fn merge_ltr(p: &[Option<Pair>]) -> Option<Pair> {
    p.iter().copied().fold(None, merge)
}

fn merge_rtl(p: &[Option<Pair>]) -> Option<Pair> {
    p.iter().rev().copied().fold(None, |acc, x| merge(x, acc))
}

fn merge_tree(p: &[Option<Pair>]) -> Option<Pair> {
    let mut cur: Vec<Option<Pair>> = p.to_vec();
    while cur.len() > 1 {
        cur = cur
            .chunks(2)
            .map(|c| if c.len() == 2 { merge(c[0], c[1]) } else { c[0] })
            .collect();
    }
    cur.first().copied().flatten()
}

fn assert_all_merges(values: &[i32], mask: &[u64], want: Option<Pair>, label: &str) {
    for seg in [1usize, 16, 17, 64, 100, 256, 4096, 65_536] {
        let p = partials(values, mask, seg);
        assert_eq!(merge_ltr(&p), want, "ltr merge: seg={seg} case={label}");
        assert_eq!(merge_rtl(&p), want, "rtl merge: seg={seg} case={label}");
        assert_eq!(merge_tree(&p), want, "tree merge: seg={seg} case={label}");
    }
}

// ------------------------------------------------------------------ tests

/// FAILS IF: an empty selection returns Some, or an empty/zero mask panics.
#[test]
fn empty_population_returns_none() {
    for n in [0usize, 1, 63, 64, 65, 257] {
        let values = gen_values(n, 11);
        let mask = vec![0u64; n.div_ceil(64)];
        assert_eq!(masked_argmin_i32(&values, &mask), None, "argmin n={n}");
        assert_eq!(masked_argmax_i32(&values, &mask), None, "argmax n={n}");
        assert_eq!(masked_argminmax_i32(&values, &mask), None, "argminmax n={n}");
    }
    let empty: [i32; 0] = [];
    let no_words: [u64; 0] = [];
    assert_eq!(masked_argmin_i32(&empty, &no_words), None);
    assert_eq!(masked_argmax_i32(&empty, &no_words), None);
    assert_eq!(masked_argminmax_i32(&empty, &no_words), None);
}

/// FAILS IF: a single selected row is reported at the wrong position or value.
#[test]
fn one_selected_value_is_returned_exactly() {
    for n in [1usize, 16, 17, 64, 65, 257] {
        let values = gen_values(n, 22);
        let positions: Vec<usize> = if n <= 65 {
            (0..n).collect()
        } else {
            vec![0, 15, 16, 63, 64, 255, 256]
        };
        for p in positions {
            let mask = mask_from(n, |i| i == p);
            assert_eq!(masked_argmin_i32(&values, &mask), Some((p, values[p])), "min n={n} p={p}");
            assert_eq!(masked_argmax_i32(&values, &mask), Some((p, values[p])), "max n={n} p={p}");
            check("single", &values, &mask);
        }
    }
}

/// FAILS IF: ties are not resolved to the lowest selected position, for min or max.
#[test]
fn equal_extremes_pick_lowest_position() {
    let n = 300;
    let mut r = Rng::new(33);
    let mut values: Vec<i32> = (0..n).map(|_| r.range(-500, 500)).collect();
    for p in [5usize, 21, 70, 200] {
        values[p] = -1000;
    }
    for p in [9usize, 33, 130, 250] {
        values[p] = 1000;
    }
    let full = mask_from(n, |_| true);
    assert_eq!(masked_argminmax_i32(&values, &full), Some(((5, -1000), (9, 1000))));
    check("ties-full", &values, &full);

    let no5 = mask_from(n, |i| i != 5);
    assert_eq!(masked_argmin_i32(&values, &no5), Some((21, -1000)));
    check("ties-no5", &values, &no5);

    let no9 = mask_from(n, |i| i != 9);
    assert_eq!(masked_argmax_i32(&values, &no9), Some((33, 1000)));
    check("ties-no9", &values, &no9);
}

/// FAILS IF: an all-equal lane does not resolve to the first selected position.
#[test]
fn all_identical_values_pick_first_selected() {
    for n in [1usize, 15, 16, 17, 64, 65, 300] {
        let values = vec![7i32; n];
        let full = mask_from(n, |_| true);
        assert_eq!(masked_argminmax_i32(&values, &full), Some(((0, 7), (0, 7))), "full n={n}");
        check("identical-full", &values, &full);
        if n > 20 {
            let tail = mask_from(n, |i| i >= 20);
            assert_eq!(masked_argminmax_i32(&values, &tail), Some(((20, 7), (20, 7))), "tail n={n}");
            check("identical-tail", &values, &tail);
        }
    }
}

/// FAILS IF: i32::MIN / i32::MAX / sign boundaries are mishandled (overflow, unsigned compare, sentinel collision).
#[test]
fn i32_extremes_across_group_boundaries() {
    let n = 100;
    let mut r = Rng::new(55);
    let mut values: Vec<i32> = (0..n).map(|_| r.full_i32()).collect();
    values[0] = i32::MIN;
    values[15] = i32::MAX;
    values[16] = -1;
    values[31] = 0;
    values[63] = 1;
    values[64] = i32::MIN;
    values[65] = i32::MAX;
    let full = mask_from(n, |_| true);
    assert_eq!(masked_argminmax_i32(&values, &full), Some(((0, i32::MIN), (15, i32::MAX))));
    check("extremes-full", &values, &full);
    check("extremes-even", &values, &mask_from(n, |i| i % 2 == 0));
    check("extremes-from-1", &values, &mask_from(n, |i| i >= 1));
    check("extremes-from-16", &values, &mask_from(n, |i| i >= 16));
    check("extremes-mid", &values, &mask_from(n, |i| (16..=65).contains(&i) && i != 64));

    let all_min = vec![i32::MIN; n];
    assert_eq!(masked_argminmax_i32(&all_min, &full), Some(((0, i32::MIN), (0, i32::MIN))));
    check("all-min", &all_min, &full);
    let all_max = vec![i32::MAX; n];
    assert_eq!(masked_argminmax_i32(&all_max, &full), Some(((0, i32::MAX), (0, i32::MAX))));
    check("all-max", &all_max, &full);
}

/// FAILS IF: any lane length x mask density combination disagrees with the oracle.
#[test]
fn sparse_and_dense_masks_match_oracle() {
    for &n in LENS.iter() {
        let values = gen_values(n, 66 + n as u64);
        for (label, mask) in mask_family(n) {
            check(label, &values, &mask);
        }
    }
}

/// FAILS IF: mask bits at or past values.len() (or surplus mask words) influence the result.
#[test]
fn tail_bits_past_length_are_ignored() {
    for n in [63usize, 64, 65, 255, 256, 257, 65_536] {
        let values = gen_values(n, 77 + n as u64);
        let exact = vec![u64::MAX; n.div_ceil(64)];
        check("tail-exact-words", &values, &exact);
        let surplus = vec![u64::MAX; n.div_ceil(64) + 1];
        check("tail-surplus-word", &values, &surplus);
    }
    let n = 65;
    let values = vec![5i32; n];
    let mask = vec![u64::MAX; 3];
    let got = masked_argminmax_i32(&values, &mask);
    assert_eq!(got, oracle(&values, &mask));
    let ((pa, _), (pb, _)) = got.expect("rows are selected");
    assert!(pa < n && pb < n, "positions must be < n: {pa} {pb}");
    assert_eq!((pa, pb), (0, 0));
}

/// FAILS IF: a masked-out i32::MIN / i32::MAX still wins, or a selected one fails to win.
#[test]
fn masked_out_extreme_cannot_win() {
    let n = 1000;
    let mut r = Rng::new(88);
    let mut values: Vec<i32> = (0..n).map(|_| r.range(-1_000_000, 1_000_000)).collect();
    values[500] = i32::MIN;
    values[501] = i32::MAX;
    let excl = mask_from(n, |i| i != 500 && i != 501);
    let got = masked_argminmax_i32(&values, &excl).expect("rows are selected");
    assert_ne!(got.0 .1, i32::MIN);
    assert_ne!(got.1 .1, i32::MAX);
    assert_ne!(got.0 .0, 500);
    assert_ne!(got.1 .0, 501);
    check("excluded", &values, &excl);

    let incl = mask_from(n, |_| true);
    assert_eq!(masked_argminmax_i32(&values, &incl), Some(((500, i32::MIN), (501, i32::MAX))));
    check("included", &values, &incl);
}

/// FAILS IF: folding a lane in pieces (any split, any merge order) differs from the single fold.
#[test]
fn segmented_folds_obey_merge_law() {
    let n = 65_536;
    let mut r = Rng::new(99);
    let values: Vec<i32> = (0..n).map(|_| (r.next() % 5) as i32 - 2).collect();
    let bits = random_bits(n, 0xFEED, 2);
    let mask = mask_from(n, |i| bits[i]);
    let want = oracle(&values, &mask);
    assert_eq!(masked_argminmax_i32(&values, &mask), want, "whole lane vs oracle");
    assert_all_merges(&values, &mask, want, "dup-heavy");
}

/// FAILS IF: tie winner depends on merge order (a later piece with an equal value displaces an earlier one).
#[test]
fn tie_winner_is_independent_of_merge_order() {
    let n = 65_536;
    let values = vec![3i32; n];
    let bits = random_bits(n, 0xC0FFEE, 2);
    let mask = mask_from(n, |i| bits[i]);
    let first = (0..n).find(|&i| bits[i]).expect("some row selected");
    let want = Some(((first, 3), (first, 3)));
    assert_eq!(oracle(&values, &mask), want);
    assert_eq!(masked_argminmax_i32(&values, &mask), want);
    assert_all_merges(&values, &mask, want, "all-equal");
}

/// FAILS IF: extreme values depend on physical row order, or positions do not name a row holding the extreme value.
#[test]
fn physical_reorder_keeps_values_and_documents_position_tie_rule() {
    let n = 500usize;
    let mut r = Rng::new(123);
    let value_of_key: Vec<i32> = (0..n).map(|_| r.range(-300, 300)).collect();
    let selected = |k: usize| !k.is_multiple_of(3);

    let keys_a: Vec<usize> = (0..n).collect();
    let mut keys_b = keys_a.clone();
    for i in (1..n).rev() {
        let j = (r.next() % (i as u64 + 1)) as usize;
        keys_b.swap(i, j);
    }
    let lane = |keys: &[usize]| -> (Vec<i32>, Vec<u64>) {
        (keys.iter().map(|&k| value_of_key[k]).collect(), mask_from(n, |i| selected(keys[i])))
    };
    let (va, ma) = lane(&keys_a);
    let (vb, mb) = lane(&keys_b);
    let ra = masked_argminmax_i32(&va, &ma).expect("selected");
    let rb = masked_argminmax_i32(&vb, &mb).expect("selected");
    assert_eq!(ra.0 .1, rb.0 .1, "min value differs across orders");
    assert_eq!(ra.1 .1, rb.1 .1, "max value differs across orders");
    for (res, keys) in [(ra, &keys_a), (rb, &keys_b)] {
        for (pos, val) in [res.0, res.1] {
            let k = keys[pos];
            assert!(selected(k), "position names an unselected key");
            assert_eq!(value_of_key[k], val, "position names a row without the extreme value");
        }
    }

    // Constructed tie: keys 1 and 3 share the min, keys 2 and 4 share the max.
    // The winner is the lowest POSITION in the lane passed, so the winning KEY
    // differs between the two orders while the value is equal (documented contract).
    let vk = [5, 1, 9, 1, 9, 2, 3, 4];
    let ka = [0usize, 1, 2, 3, 4, 5, 6, 7];
    let kb = [0usize, 3, 4, 1, 2, 5, 6, 7];
    let la: Vec<i32> = ka.iter().map(|&k| vk[k]).collect();
    let lb: Vec<i32> = kb.iter().map(|&k| vk[k]).collect();
    let full = mask_from(8, |_| true);
    let ta = masked_argminmax_i32(&la, &full).expect("selected");
    let tb = masked_argminmax_i32(&lb, &full).expect("selected");
    assert_eq!(ta.0 .1, tb.0 .1);
    assert_eq!(ta.1 .1, tb.1 .1);
    assert_eq!((ka[ta.0 .0], kb[tb.0 .0]), (1, 3), "min winning keys");
    assert_eq!((ka[ta.1 .0], kb[tb.1 .0]), (2, 4), "max winning keys");
    assert_ne!(ka[ta.0 .0], kb[tb.0 .0]);
    assert_ne!(ka[ta.1 .0], kb[tb.1 .0]);
}

/// FAILS IF: positions are carried in a narrow (u16/u17-ish) index type and wrap on large lanes.
#[test]
fn positions_beyond_2_pow_16_are_exact() {
    let n = 200_000;
    let mut r = Rng::new(321);
    let mut values: Vec<i32> = (0..n).map(|_| r.range(0, 1000)).collect();
    values[199_999] = -5;
    values[131_072] = 5000;
    let full = mask_from(n, |_| true);
    assert_eq!(masked_argminmax_i32(&values, &full), Some(((199_999, -5), (131_072, 5000))));
    assert_eq!(masked_argmin_i32(&values, &full), Some((199_999, -5)));
    assert_eq!(masked_argmax_i32(&values, &full), Some((131_072, 5000)));
    check("large-lane", &values, &full);
}

/// FAILS IF: monotone lanes (every row a new extreme, or never) are mishandled.
#[test]
fn monotone_lanes_descending_and_ascending() {
    let n = 4097;
    let desc: Vec<i32> = (0..n).map(|i| (n - i) as i32).collect();
    let asc: Vec<i32> = (0..n).map(|i| i as i32).collect();
    let full = mask_from(n, |_| true);
    assert_eq!(masked_argminmax_i32(&desc, &full), Some(((n - 1, 1), (0, n as i32))));
    assert_eq!(masked_argminmax_i32(&asc, &full), Some(((0, 0), (n - 1, (n - 1) as i32))));
    check("descending", &desc, &full);
    check("ascending", &asc, &full);
}

/// FAILS IF: masked_argmin_i32 reads past a too-short mask instead of panicking.
#[test]
#[should_panic]
fn short_mask_panics_argmin() {
    let values = vec![1i32; 65];
    let mask = vec![u64::MAX; 1];
    let _ = masked_argmin_i32(&values, &mask);
}

/// FAILS IF: masked_argmax_i32 reads past a too-short mask instead of panicking.
#[test]
#[should_panic]
fn short_mask_panics_argmax() {
    let values = vec![1i32; 65];
    let mask = vec![u64::MAX; 1];
    let _ = masked_argmax_i32(&values, &mask);
}

/// FAILS IF: masked_argminmax_i32 reads past a too-short mask instead of panicking.
#[test]
#[should_panic]
fn short_mask_panics_argminmax() {
    let values = vec![1i32; 65];
    let mask = vec![u64::MAX; 1];
    let _ = masked_argminmax_i32(&values, &mask);
}

/// FAILS IF: the value-only folds disagree with the value half of the arg folds.
#[test]
fn arg_folds_agree_with_value_only_folds() {
    for &n in LENS.iter() {
        let values = gen_values(n, 444 + n as u64);
        for (label, mask) in mask_family(n) {
            assert_eq!(
                masked_argmin_i32(&values, &mask).map(|p| p.1),
                masked_min_i32(&values, &mask),
                "min value: n={n} case={label}"
            );
            assert_eq!(
                masked_argmax_i32(&values, &mask).map(|p| p.1),
                masked_max_i32(&values, &mask),
                "max value: n={n} case={label}"
            );
        }
    }
}
