//! D-LXC-29 measurement: bounded `u8` power sums into 128-bit registers vs the
//! wide `i32` kernels, univariate and bivariate, per tile size and over a
//! multi-tile population. Exactness is asserted on every run before timing is
//! reported. Prints the realized SIMD tier; a timing without it is an anecdote.
//!
//! ```sh
//! env -u RUSTFLAGS cargo run --release --example bounded_power_sums_bench
//! ```

use ndarray::simd::{
    fold_bounded_cross_power_sums_tiles, fold_bounded_power_sums_tiles, masked_group_bounded_cross_power_sums_u8,
    masked_group_bounded_power_sums_u8, masked_group_cross_power_sums_i32, masked_group_power_sums_i32,
    widen_bounded_cross_power_sums, widen_bounded_power_sums, CrossPowerSums, PowerSums, BOUNDED_TILE_ROWS,
};
use std::hint::black_box;
use std::time::Instant;

const GROUPS: usize = 16;

fn data(n: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>, Vec<u8>) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s
    };
    let mut mask = vec![0u64; n.div_ceil(64)];
    for (i, _) in (0..n).enumerate().filter(|(i, _)| i % 9 != 2) {
        mask[i / 64] |= 1 << (i % 64);
    }
    let keys = (0..n)
        .map(|_| (next() >> 40) as u32 % GROUPS as u32)
        .collect();
    let xs = (0..n).map(|_| (next() >> 56) as u8).collect();
    let ys = (0..n).map(|_| (next() >> 56) as u8).collect();
    (mask, keys, xs, ys)
}

/// Median ns/row over `reps` runs.
fn time(rows: usize, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_nanos() as f64 / rows as f64
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[reps / 2]
}

fn main() {
    #[cfg(target_arch = "x86_64")]
    println!("tier: avx512f={} avx2={}", cfg!(target_feature = "avx512f"), cfg!(target_feature = "avx2"));
    #[cfg(not(target_arch = "x86_64"))]
    println!("tier: arch={}", std::env::consts::ARCH);
    println!("groups={GROUPS}  median of 31 runs, ns/row (lower is better)\n");
    println!("{:<34}{:>10}{:>10}{:>9}", "case", "wide i32", "bounded", "ratio");

    for &tile in &[4_096usize, 16_384, BOUNDED_TILE_ROWS] {
        let (mask, keys, xs, ys) = data(tile);
        let wx: Vec<i32> = xs.iter().map(|&x| i32::from(x)).collect();
        let wy: Vec<i32> = ys.iter().map(|&y| i32::from(y)).collect();

        // exactness first
        let mut regs = [[0u8; 16]; GROUPS];
        let mut want = [PowerSums::default(); GROUPS];
        masked_group_bounded_power_sums_u8(&mask, &keys, &xs, &mut regs).unwrap();
        masked_group_power_sums_i32(&mask, &keys, &wx, &mut want);
        assert!(regs
            .iter()
            .map(widen_bounded_power_sums)
            .eq(want.iter().copied()));

        let w = time(tile, 31, || {
            let mut o = [PowerSums::default(); GROUPS];
            masked_group_power_sums_i32(black_box(&mask), black_box(&keys), black_box(&wx), &mut o);
            black_box(o);
        });
        let b = time(tile, 31, || {
            let mut r = [[0u8; 16]; GROUPS];
            masked_group_bounded_power_sums_u8(black_box(&mask), black_box(&keys), black_box(&xs), &mut r).unwrap();
            black_box(r);
        });
        println!("{:<34}{w:>10.3}{b:>10.3}{:>9.2}", format!("univariate tile={tile}"), w / b);

        let (mut r0, mut r1) = ([[0u8; 16]; GROUPS], [[0u8; 16]; GROUPS]);
        let mut want = [CrossPowerSums::default(); GROUPS];
        masked_group_bounded_cross_power_sums_u8(&mask, &keys, &xs, &ys, &mut r0, &mut r1).unwrap();
        masked_group_cross_power_sums_i32(&mask, &keys, &wx, &wy, &mut want);
        assert!(r0
            .iter()
            .zip(&r1)
            .map(|(a, b)| widen_bounded_cross_power_sums(a, b))
            .eq(want.iter().copied()));

        let w = time(tile, 31, || {
            let mut o = [CrossPowerSums::default(); GROUPS];
            masked_group_cross_power_sums_i32(
                black_box(&mask),
                black_box(&keys),
                black_box(&wx),
                black_box(&wy),
                &mut o,
            );
            black_box(o);
        });
        let b = time(tile, 31, || {
            let (mut a, mut c) = ([[0u8; 16]; GROUPS], [[0u8; 16]; GROUPS]);
            masked_group_bounded_cross_power_sums_u8(
                black_box(&mask),
                black_box(&keys),
                black_box(&xs),
                black_box(&ys),
                &mut a,
                &mut c,
            )
            .unwrap();
            black_box((a, c));
        });
        println!("{:<34}{w:>10.3}{b:>10.3}{:>9.2}", format!("bivariate tile={tile}"), w / b);
    }

    // Multi-tile population through the tiled drivers (widen + merge included).
    let n = 16 * BOUNDED_TILE_ROWS + 123;
    let (mask, keys, xs, ys) = data(n);
    let wx: Vec<i32> = xs.iter().map(|&x| i32::from(x)).collect();
    let wy: Vec<i32> = ys.iter().map(|&y| i32::from(y)).collect();
    let tiled_uni = || {
        let (mut regs, mut out) = ([[0u8; 16]; GROUPS], [PowerSums::default(); GROUPS]);
        fold_bounded_power_sums_tiles(n, &mut regs, &mut out, |t, regs| {
            masked_group_bounded_power_sums_u8(&mask[t.start / 64..], &keys[t.clone()], &xs[t], regs)
        })
        .unwrap();
        out
    };
    let mut want = [PowerSums::default(); GROUPS];
    masked_group_power_sums_i32(&mask, &keys, &wx, &mut want);
    assert_eq!(tiled_uni(), want);
    let w = time(n, 31, || {
        let mut o = [PowerSums::default(); GROUPS];
        masked_group_power_sums_i32(&mask, &keys, black_box(&wx), &mut o);
        black_box(o);
    });
    let b = time(n, 31, || {
        black_box(tiled_uni());
    });
    println!("{:<34}{w:>10.3}{b:>10.3}{:>9.2}", format!("univariate tiled n={n}"), w / b);

    let tiled_bi = || {
        let (mut a, mut c, mut out) = ([[0u8; 16]; GROUPS], [[0u8; 16]; GROUPS], [CrossPowerSums::default(); GROUPS]);
        fold_bounded_cross_power_sums_tiles(n, &mut a, &mut c, &mut out, |t, a, c| {
            masked_group_bounded_cross_power_sums_u8(
                &mask[t.start / 64..],
                &keys[t.clone()],
                &xs[t.clone()],
                &ys[t],
                a,
                c,
            )
        })
        .unwrap();
        out
    };
    let mut want = [CrossPowerSums::default(); GROUPS];
    masked_group_cross_power_sums_i32(&mask, &keys, &wx, &wy, &mut want);
    assert_eq!(tiled_bi(), want);
    let w = time(n, 31, || {
        let mut o = [CrossPowerSums::default(); GROUPS];
        masked_group_cross_power_sums_i32(&mask, &keys, black_box(&wx), black_box(&wy), &mut o);
        black_box(o);
    });
    let b = time(n, 31, || {
        black_box(tiled_bi());
    });
    println!("{:<34}{w:>10.3}{b:>10.3}{:>9.2}", format!("bivariate tiled n={n}"), w / b);
    println!("\ninput bytes/row: wide i32 = 4 per lane, bounded u8 = 1 per lane");
}
