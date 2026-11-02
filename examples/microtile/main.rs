//! D-SIMD-MICRO-MACRO-1 step 1 cost gate: field-isolated add and exact fold
//! over 64K microtiles, in four realizations. Every realization is checked
//! against the unpacked oracle before it is timed.
//! `cargo run --release --example microtile --features std`.

// The tests exercise sub and the stencil; the bench times add and fold only.
#[allow(dead_code)]
mod tile;

use std::hint::black_box;
use std::time::Instant;
use tile::*;

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512bw") {
        "x86_64 AVX-512"
    } else if cfg!(target_feature = "avx2") {
        "x86_64 AVX2"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64 NEON"
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

/// Median ns per tile over 31 runs.
fn time(name: &str, tiles: usize, base: Option<f64>, f: &mut dyn FnMut() -> u64) -> f64 {
    black_box(f());
    let mut t: Vec<f64> = (0..31)
        .map(|_| {
            let t0 = Instant::now();
            black_box(f());
            t0.elapsed().as_nanos() as f64 / tiles as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    let med = t[15];
    let rel = base
        .map(|b| format!("  {:6.2}x", med / b))
        .unwrap_or_default();
    println!("  {name:<44} {med:8.3} ns/tile{rel}");
    med
}

fn run<P: Packing>(label: &str, tiles: usize) {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let a: Vec<u128> = (0..tiles)
        .map(|_| (xorshift(&mut s) as u128) << 64 | xorshift(&mut s) as u128)
        .collect();
    let b: Vec<u128> = (0..tiles)
        .map(|_| (xorshift(&mut s) as u128) << 64 | xorshift(&mut s) as u128)
        .collect();
    let ((alo, ahi), (blo, bhi)) = (split(&a), split(&b));
    let (sa, sb) = (Soa { lo: &alo, hi: &ahi }, Soa { lo: &blo, hi: &bhi });
    let (wa, wb) = (widen::<P>(&a), widen::<P>(&b));
    let (pa, pb) = (to_planes::<P>(&a), to_planes::<P>(&b));
    let (n, w) = (P::N as usize, P::W as usize);

    // Exactness before timing.
    let want: Vec<u128> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| oracle_add::<P>(x, y).0)
        .collect();
    let mut out128 = vec![0u128; tiles];
    for t in 0..tiles {
        out128[t] = add::<P>(a[t], b[t]);
    }
    assert_eq!(out128, want, "{label} u128 add");
    let (mut ol, mut oh) = (vec![0u64; tiles], vec![0u64; tiles]);
    soa_add_slices::<P>(&sa, &sb, &mut ol, &mut oh);
    assert!((0..tiles).all(|t| (ol[t] as u128 | (oh[t] as u128) << 64) == want[t]), "{label} soa add");
    let mut ow = vec![vec![0u32; tiles]; n];
    for j in 0..n {
        widened_add::<P>(&wa[j], &wb[j], &mut ow[j]);
        assert!((0..tiles).all(|t| ow[j][t] == get::<P>(want[t], j as u32)), "{label} widened add");
    }
    let (fu, fs) = oracle_fold::<P>(&a);
    assert_eq!(soa_fold::<P>(&sa), (fu.clone(), fs.clone()), "{label} soa fold");
    for j in 0..n {
        assert_eq!(widened_fold::<P>(&wa[j]), fu[j], "{label} widened fold");
        assert_eq!(planes_fold(&pa[j], w), (fu[j], fs[j]), "{label} planes fold");
    }

    println!("{label}: {n} fields x {w} bits, {tiles} tiles, all realizations exact");
    println!("  storage: packed 16 B/tile, widened {} B/tile, planes 16 B/tile", 4 * n);
    let base = time("add  u128 (one 128-bit word)", tiles, None, &mut || {
        for t in 0..tiles {
            out128[t] = add::<P>(a[t], b[t]);
        }
        out128[tiles - 1] as u64
    });
    time("add  SoA U64x8 (8 tiles per op)", tiles, Some(base), &mut || {
        soa_add_slices::<P>(&sa, &sb, &mut ol, &mut oh);
        ol[tiles - 1]
    });
    time("add  widened U32x16 (one field per lane)", tiles, Some(base), &mut || {
        for j in 0..n {
            widened_add::<P>(&wa[j], &wb[j], &mut ow[j]);
        }
        ow[0][tiles - 1] as u64
    });
    let mut op = vec![0u64; w];
    time("add  bit planes, full_add_mask ripple", tiles, Some(base), &mut || {
        let mut acc = 0;
        for j in 0..n {
            for (ak, bk) in pa[j].chunks_exact(w).zip(pb[j].chunks_exact(w)) {
                acc ^= planes_add(ak, bk, &mut op);
            }
        }
        acc
    });
    let base = time("fold SoA U64x8 (unsigned + signed)", tiles, None, &mut || soa_fold::<P>(&sa).0[0]);
    time("fold widened U32x16 (unsigned)", tiles, Some(base), &mut || {
        (0..n).map(|j| widened_fold::<P>(&wa[j])).sum()
    });
    time("fold bit planes, popcount (unsigned + signed)", tiles, Some(base), &mut || {
        (0..n).map(|j| planes_fold(&pa[j], w).0).sum()
    });
}

fn main() {
    println!("D-SIMD-MICRO-MACRO-1 step 1, {}", tier());
    for tiles in [1 << 12, 1 << 16] {
        run::<U24x5>("U24x5", tiles);
        run::<U9x14>("U9x14", tiles);
    }
}
