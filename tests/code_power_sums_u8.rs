#![cfg(feature = "std")]
//! Power sums and extrema of a `u8` lane from per-code counts.
//!
//! Predictions P1..P4 are pre-registered; each test names its prediction.
//! The oracle `direct` multiplies in `u128` row by row and calls nothing
//! under test.

use ndarray::simd::{
    code_power_sums_u8, masked_code_counts_u8, masked_first_row_of_code_u8, masked_group_bounded_power_sums_u8,
    widen_bounded_power_sums, CodePowerSums, PowerSums, BOUNDED_TILE_ROWS,
};

struct SplitMix64(u64);

impl SplitMix64 {
    fn new() -> Self {
        SplitMix64(0x9E37_79B9_7F4A_7C15)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

fn selected(mask: &[u64], i: usize) -> bool {
    (mask[i / 64] >> (i % 64)) & 1 == 1
}

/// Independent oracle: `(n, [Σx, Σx², Σx³, Σx⁴], min, max)` over selected rows.
fn direct(values: &[u8], mask: &[u64]) -> Option<(u64, [u128; 4], u8, u8)> {
    let mut n = 0u64;
    let mut sum = [0u128; 4];
    let mut min = 255u8;
    let mut max = 0u8;
    for (i, &v) in values.iter().enumerate() {
        if !selected(mask, i) {
            continue;
        }
        let x = u128::from(v);
        n += 1;
        sum[0] += x;
        sum[1] += x * x;
        sum[2] += x * x * x;
        sum[3] += x * x * x * x;
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    if n == 0 {
        None
    } else {
        Some((n, sum, min, max))
    }
}

fn make_mask(rng: &mut SplitMix64, len: usize, kind: usize) -> Vec<u64> {
    let words = len.div_ceil(64);
    (0..words)
        .map(|_| match kind {
            0 => u64::MAX,
            1 => rng.next_u64(),
            _ => {
                let mut w = u64::MAX;
                for _ in 0..6 {
                    w &= rng.next_u64();
                }
                w
            }
        })
        .collect()
}

fn make_values(rng: &mut SplitMix64, len: usize, kind: usize) -> Vec<u8> {
    (0..len)
        .map(|_| match kind {
            0 => (rng.next_u64() & 0xFF) as u8,
            1 => {
                if rng.next_u64() & 1 == 0 {
                    0
                } else {
                    255
                }
            }
            _ => 100 + (rng.next_u64() & 3) as u8,
        })
        .collect()
}

fn counted(values: &[u8], mask: &[u64]) -> Option<CodePowerSums> {
    let mut counts = [0u64; 256];
    masked_code_counts_u8(values, mask, &mut counts);
    code_power_sums_u8(&counts)
}

fn assert_matches_direct(values: &[u8], mask: &[u64], case: &str) {
    let got = counted(values, mask);
    let want = direct(values, mask);
    match (got, want) {
        (None, None) => {}
        (Some(g), Some((n, sum, min, max))) => {
            assert_eq!(g.n, n, "{case}: n");
            assert_eq!(g.sum, sum, "{case}: sums");
            assert_eq!(g.min, min, "{case}: min");
            assert_eq!(g.max, max, "{case}: max");
        }
        (g, w) => panic!("{case}: presence differs, got {g:?} want {w:?}"),
    }
}

fn pow(u: u8, k: u32) -> u128 {
    u128::from(u).pow(k)
}

/// P1: the map from counts to sums is linear, so exactness on every
/// single-bin histogram implies exactness on every table (absent overflow).
#[test]
fn exact_on_every_single_bin_histogram() {
    for u in 0..=255u8 {
        for &c in &[1u64, 2, 65_536, 1u64 << 40] {
            let mut counts = [0u64; 256];
            counts[usize::from(u)] = c;
            let s = code_power_sums_u8(&counts).expect("non-empty single-bin table");
            assert_eq!(s.n, c, "P1 code {u} count {c}: n");
            assert_eq!(s.min, u, "P1 code {u} count {c}: min");
            assert_eq!(s.max, u, "P1 code {u} count {c}: max");
            for k in 1..=4u32 {
                assert_eq!(s.sum[(k - 1) as usize], u128::from(c) * pow(u, k), "P1 code {u} count {c}: sum[{}]", k - 1);
            }
        }
    }
}

/// P1b: linearity on random tables, and large counts.
#[test]
fn linearity_on_random_tables() {
    let mut rng = SplitMix64::new();
    for case in 0..200 {
        let mut counts = [0u64; 256];
        for c in counts.iter_mut() {
            if rng.next_u64() & 1 == 0 {
                *c = rng.next_u64() & 0xFFFF_FFFF;
            }
        }
        if counts.iter().all(|&c| c == 0) {
            counts[(rng.next_u64() & 0xFF) as usize] = 1 + rng.below(1000);
        }
        let s = code_power_sums_u8(&counts).expect("non-empty table");
        let mut n = 0u64;
        let mut sum = [0u128; 4];
        for (u, &c) in counts.iter().enumerate() {
            n += c;
            for k in 1..=4u32 {
                sum[(k - 1) as usize] += u128::from(c) * pow(u as u8, k);
            }
        }
        let first = counts.iter().position(|&c| c != 0).unwrap() as u8;
        let last = counts.iter().rposition(|&c| c != 0).unwrap() as u8;
        assert_eq!(s.n, n, "P1b random table {case}: n");
        assert_eq!(s.sum, sum, "P1b random table {case}: sums");
        assert_eq!(s.min, first, "P1b random table {case}: min");
        assert_eq!(s.max, last, "P1b random table {case}: max");
    }

    let big = u64::MAX / 256;
    let counts = [big; 256];
    let s = code_power_sums_u8(&counts).expect("full table");
    let mut sum = [0u128; 4];
    for u in 0..=255u8 {
        for k in 1..=4u32 {
            sum[(k - 1) as usize] += u128::from(big) * pow(u, k);
        }
    }
    assert_eq!(s.n, big * 256, "P1b large counts: n");
    assert_eq!(s.sum, sum, "P1b large counts: sums");
    assert_eq!((s.min, s.max), (0, 255), "P1b large counts: extrema");
}

/// P2: the count-table route equals the row-by-row oracle.
#[test]
fn matches_direct_sums_over_selected_rows() {
    let mut rng = SplitMix64::new();
    let lens = [0usize, 1, 63, 64, 65, 127, 128, 1000, 4097, 65_536, 70_001];
    let mut non_empty = 0;
    for case in 0..300 {
        let len = lens[(rng.next_u64() % lens.len() as u64) as usize];
        let mk = (rng.next_u64() % 3) as usize;
        let vk = (rng.next_u64() % 3) as usize;
        let values = make_values(&mut rng, len, vk);
        let mask = make_mask(&mut rng, len, mk);
        if direct(&values, &mask).is_some() {
            non_empty += 1;
        }
        assert_matches_direct(&values, &mask, &format!("P2 case {case} len {len} mask kind {mk} value kind {vk}"));
    }
    assert!(non_empty >= 200, "P2 anti-vacuity: only {non_empty} of 300 cases had a selection");
}

/// P2b: bits at or past the lane are ignored, and a longer mask is harmless.
#[test]
fn bits_past_the_lane_are_ignored() {
    let mut rng = SplitMix64::new();
    let values = make_values(&mut rng, 70, 0);
    let mask = [rng.next_u64(), u64::MAX];
    let inside: Vec<u64> = vec![mask[0], (1u64 << 6) - 1];
    let want = direct(&values, &inside).expect("rows selected");
    let got = counted(&values, &mask).expect("rows selected");
    assert_eq!(got.n, want.0, "P2b past-lane bits: n");
    assert_eq!(got.sum, want.1, "P2b past-lane bits: sums");
    assert_eq!((got.min, got.max), (want.2, want.3), "P2b past-lane bits: extrema");

    let longer = [mask[0], mask[1], u64::MAX, u64::MAX];
    let got_long = counted(&values, &longer).expect("rows selected");
    assert_eq!(got_long, got, "P2b longer mask slice changes the result");
}

/// P3: agreement with the bounded 16-byte register fold.
#[test]
fn agrees_with_the_bounded_register_fold() {
    let mut rng = SplitMix64::new();
    let check = |values: &[u8], mask: &[u64], case: &str| {
        let keys = vec![0u32; values.len()];
        let mut regs = [[0u8; 16]; 1];
        masked_group_bounded_power_sums_u8(mask, &keys, values, &mut regs).expect("tile within bound");
        let reg = widen_bounded_power_sums(&regs[0]);
        match counted(values, mask) {
            Some(s) => {
                let ps = s.power_sums().expect("sum fits i64 on a bounded tile");
                assert_eq!(ps, reg, "{case}: record differs from the bounded fold");
            }
            None => assert_eq!(reg, PowerSums::default(), "{case}: empty selection must widen to default"),
        }
    };
    for &len in &[1usize, 64, 4096, BOUNDED_TILE_ROWS] {
        for mk in 0..3 {
            let values = make_values(&mut rng, len, 0);
            let mask = make_mask(&mut rng, len, mk);
            check(&values, &mask, &format!("P3 len {len} mask kind {mk}"));
        }
    }

    let worst = vec![255u8; BOUNDED_TILE_ROWS];
    let all = vec![u64::MAX; BOUNDED_TILE_ROWS / 64];
    check(&worst, &all, "P3 worst case 65536 x 255");
    let s = counted(&worst, &all).expect("selected");
    assert_eq!(s.n, BOUNDED_TILE_ROWS as u64, "P3 worst case: n");

    let values = make_values(&mut rng, 4096, 0);
    let none = vec![0u64; 64];
    assert!(counted(&values, &none).is_none(), "P3 empty selection: code route must be None");
    check(&values, &none, "P3 empty selection");
}

/// P4: counts of pieces add.
#[test]
fn counts_of_pieces_add() {
    let mut rng = SplitMix64::new();
    let len = 200_000usize;
    let values = make_values(&mut rng, len, 0);
    let mask = make_mask(&mut rng, len, 1);
    let cuts = [0usize, 64_000, 128_000, len];

    let mut whole = [0u64; 256];
    masked_code_counts_u8(&values, &mask, &mut whole);

    let mut chunked = [0u64; 256];
    let mut separate = [[0u64; 256]; 3];
    for p in 0..3 {
        let (a, b) = (cuts[p], cuts[p + 1]);
        let piece_mask = &mask[a / 64..b.div_ceil(64)];
        masked_code_counts_u8(&values[a..b], piece_mask, &mut chunked);
        masked_code_counts_u8(&values[a..b], piece_mask, &mut separate[p]);
    }
    assert_eq!(chunked, whole, "P4 three calls into one table differ from one call");

    let mut added = [0u64; 256];
    for t in &separate {
        for (a, &c) in added.iter_mut().zip(t.iter()) {
            *a += c;
        }
    }
    assert_eq!(added, whole, "P4 three separate tables added bin by bin differ from one call");

    let want = direct(&values, &mask).expect("selection");
    let s = code_power_sums_u8(&chunked).expect("selection");
    assert_eq!(s.n, want.0, "P4 n");
    assert_eq!(s.sum, want.1, "P4 sums");
    assert_eq!((s.min, s.max), (want.2, want.3), "P4 extrema");
}

/// P4b: counting twice doubles everything; extrema stay.
#[test]
fn counts_accumulate_not_overwrite() {
    let mut rng = SplitMix64::new();
    let values = make_values(&mut rng, 5000, 0);
    let mask = make_mask(&mut rng, 5000, 1);
    let mut once = [0u64; 256];
    masked_code_counts_u8(&values, &mask, &mut once);
    let mut twice = [0u64; 256];
    masked_code_counts_u8(&values, &mask, &mut twice);
    masked_code_counts_u8(&values, &mask, &mut twice);
    for u in 0..256 {
        assert_eq!(twice[u], 2 * once[u], "P4b bin {u} not doubled");
    }
    let a = code_power_sums_u8(&once).expect("selection");
    let b = code_power_sums_u8(&twice).expect("selection");
    assert_eq!(b.n, 2 * a.n, "P4b n");
    for k in 0..4 {
        assert_eq!(b.sum[k], 2 * a.sum[k], "P4b sum[{k}]");
    }
    assert_eq!((b.min, b.max), (a.min, a.max), "P4b extrema changed");
}

fn scan_first(values: &[u8], mask: &[u64], code: u8) -> Option<usize> {
    (0..values.len()).find(|&i| selected(mask, i) && values[i] == code)
}

#[test]
fn first_row_finds_the_lowest_selected_row_of_the_code() {
    let mut rng = SplitMix64::new();
    let lens = [0usize, 1, 63, 64, 65, 200, 5000];
    for case in 0..300 {
        let len = lens[(rng.next_u64() % lens.len() as u64) as usize];
        let mk = (rng.next_u64() % 3) as usize;
        let vk = (rng.next_u64() % 3) as usize;
        let values = make_values(&mut rng, len, vk);
        let mask = make_mask(&mut rng, len, mk);
        let mut codes = vec![(rng.next_u64() & 0xFF) as u8];
        if let Some((_, _, min, max)) = direct(&values, &mask) {
            codes.push(min);
            codes.push(max);
        }
        for code in codes {
            assert_eq!(
                masked_first_row_of_code_u8(&values, &mask, code),
                scan_first(&values, &mask, code),
                "first_row case {case} len {len} mask kind {mk} value kind {vk} code {code}"
            );
        }
    }

    // Occurs at an unselected lower row and a selected higher row.
    let mut values = vec![1u8; 10];
    values[2] = 9;
    values[7] = 9;
    let mask = [!(1u64 << 2)];
    assert_eq!(
        masked_first_row_of_code_u8(&values, &mask, 9),
        Some(7),
        "first_row: unselected lower row must be skipped"
    );

    // Only in the final partial block.
    let mut values = vec![1u8; 100];
    values[99] = 77;
    let mask = [u64::MAX, u64::MAX];
    assert_eq!(
        masked_first_row_of_code_u8(&values, &mask, 77),
        Some(99),
        "first_row: code only in the final partial block"
    );

    // Padding bytes are zero but are not rows.
    let values = vec![5u8; 70];
    let mask = [u64::MAX, u64::MAX];
    assert_eq!(masked_first_row_of_code_u8(&values, &mask, 0), None, "first_row: padding zeros must not be found");
    let mut values = vec![5u8; 70];
    values[3] = 0;
    let mask = [!(1u64 << 3), u64::MAX];
    assert_eq!(
        masked_first_row_of_code_u8(&values, &mask, 0),
        None,
        "first_row: unselected zero and padding zeros must not be found"
    );
}

#[test]
fn extremes_and_degenerates() {
    let empty = [0u64; 256];
    assert!(code_power_sums_u8(&empty).is_none(), "empty table must be None");
    let mut counts = [0u64; 256];
    masked_code_counts_u8(&[], &[], &mut counts);
    assert_eq!(counts, empty, "empty lane must leave the table zero");
    assert!(code_power_sums_u8(&counts).is_none(), "empty lane must be None");

    let values = [42u8, 7, 9];
    let mut counts = [0u64; 256];
    masked_code_counts_u8(&values, &[0b001], &mut counts);
    let s = code_power_sums_u8(&counts).expect("one row");
    assert_eq!(s.n, 1, "one row: n");
    assert_eq!((s.min, s.max), (42, 42), "one row: extrema");
    assert_eq!(s.sum[0], 42, "one row: sum");

    let rows = 65_536usize;
    let all255 = vec![255u8; rows];
    let mask = vec![u64::MAX; rows / 64];
    let s = counted(&all255, &mask).expect("selected");
    assert_eq!(s.sum[3], rows as u128 * 255u128.pow(4), "all 255: fourth power sum");
    assert_eq!((s.n, s.min, s.max), (rows as u64, 255, 255), "all 255: n and extrema");

    let zeros = vec![0u8; rows];
    let s = counted(&zeros, &mask).expect("selected");
    assert_eq!(s.sum, [0u128; 4], "all zeros: every sum zero");
    assert_eq!((s.n, s.min, s.max), (rows as u64, 0, 0), "all zeros: n and extrema");
}

#[test]
fn power_sums_record_matches() {
    let mut counts = [0u64; 256];
    counts[3] = 10;
    counts[200] = 4;
    let s = code_power_sums_u8(&counts).expect("non-empty");
    let ps = s.power_sums().expect("small sums fit");
    assert_eq!(ps.n, s.n, "record n");
    assert_eq!(i128::from(ps.sum), s.sum[0] as i128, "record sum is sum[0]");
    assert_eq!(ps.sum_sq, s.sum[1], "record sum_sq is sum[1]");

    let mut counts = [0u64; 256];
    counts[255] = u64::MAX / 2;
    let s = code_power_sums_u8(&counts).expect("table is non-empty");
    assert!(s.sum[0] > i64::MAX as u128, "fixture must exceed i64::MAX");
    assert!(s.power_sums().is_none(), "sum beyond i64 must give None from power_sums");
}
