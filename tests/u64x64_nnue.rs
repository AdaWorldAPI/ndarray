#![cfg(feature = "std")]
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]
//! D-U64X64-NNUE-1 exactness, refusals and allocation for lab route G
//! (`examples/u64x64/nnue.rs`), against an independent per-cell oracle.

#[path = "../examples/u64x64/fixture.rs"]
mod fixture;
#[path = "../examples/u64x64/nnue.rs"]
mod nnue;

use fixture::*;
use nnue::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

// ---------------------------------------------------------------- allocator

/// Counts allocations per thread, so parallel tests do not see each other.
struct PerThread;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for PerThread {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: layout is passed through unchanged to System.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr/layout come from a matching System allocation.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        // SAFETY: ptr/layout/new_size are passed through unchanged to System.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static A: PerThread = PerThread;

fn allocs() -> usize {
    ALLOCS.with(|c| c.get())
}

// ---------------------------------------------------------------- generators

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// CellWeights in the full admitted range `-(2^15 - 1) ..= 2^15 - 1`, bias in
/// `-2^15 ..= 2^15`.
fn weights(seed: u64) -> CellWeights {
    let mut s = seed | 1;
    let mut w = vec![[[0i16; H]; 64]; 64].into_boxed_slice();
    for k in 0..64 {
        for r in 0..64 {
            for j in 0..H {
                let v = (xorshift(&mut s) % 65_535) as i32 - 32_767;
                w[k][r][j] = v as i16;
            }
        }
    }
    let mut bias = [0i32; H];
    for b in &mut bias {
        *b = (xorshift(&mut s) % 65_537) as i32 - 32_768;
    }
    CellWeights::new(bias, w).unwrap()
}

/// Every weight equal to `v`, bias `b`.
fn uniform(v: i16, b: i32) -> CellWeights {
    CellWeights::new([b; H], vec![[[v; H]; 64]; 64].into_boxed_slice()).unwrap()
}

/// A tile where each cell is set with probability `num / den`.
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

/// A mask of exactly `n` distinct random cells.
fn cells(s: &mut u64, n: usize) -> BitTile {
    let mut t = [0u64; 64];
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

// ---------------------------------------------------------------- oracle

/// Independent oracle: all 4,096 cells, one bit at a time, sums in `i64`.
/// No `trailing_zeros`, no XOR, no SIMD.
fn oracle(w: &CellWeights, t: &BitTile) -> [i64; H] {
    let mut acc = [0i64; H];
    for j in 0..H {
        acc[j] = i64::from(w.bias()[j]);
    }
    for k in 0..64 {
        for r in 0..64 {
            if (t[k] >> r) & 1 == 1 {
                let row = w.row(k, r);
                for j in 0..H {
                    acc[j] += i64::from(row[j]);
                }
            }
        }
    }
    acc
}

fn widen(a: &Acc) -> [i64; H] {
    let mut out = [0i64; H];
    for j in 0..H {
        out[j] = i64::from(a[j]);
    }
    out
}

/// Every route on `t`: both refreshes, and both updates from the empty tile.
fn check_all(w: &CellWeights, t: &BitTile, what: &str) {
    let want = oracle(w, t);
    assert_eq!(widen(&nnue_refresh(w, t)), want, "{what}: refresh");
    assert_eq!(widen(&nnue_refresh_simd(w, t)), want, "{what}: refresh_simd");
    assert_eq!(widen(&nnue_refresh_simd2(w, t)), want, "{what}: refresh_simd2");
    assert_eq!(widen(&nnue_refresh_simd4(w, t)), want, "{what}: refresh_simd4");
    let empty = [0u64; 64];
    let mut a = w.bias();
    nnue_update(w, &mut a, &empty, t);
    assert_eq!(widen(&a), want, "{what}: update from empty");
    let mut b = w.bias();
    nnue_update_simd(w, &mut b, &empty, t);
    assert_eq!(widen(&b), want, "{what}: update_simd from empty");
}

// ---------------------------------------------------------------- P1

#[test]
fn every_single_cell_matches_the_oracle() {
    let w = weights(0x9E37_79B9_7F4A_7C15);
    let full = [u64::MAX; 64];
    for c in 0..4096 {
        let mut t = [0u64; 64];
        t[c / 64] = 1u64 << (c % 64);
        check_all(&w, &t, &format!("cell {c}"));
        // removing the one cell from the full tile, both directions
        let mut rest = full;
        rest[c / 64] &= !(1u64 << (c % 64));
        let mut a = nnue_refresh(&w, &full);
        nnue_update(&w, &mut a, &full, &rest);
        assert_eq!(widen(&a), oracle(&w, &rest), "cell {c}: full minus one");
        nnue_update_simd(&w, &mut a, &rest, &full);
        assert_eq!(widen(&a), oracle(&w, &full), "cell {c}: and back");
    }
}

#[test]
fn empty_full_and_random_tiles_match_the_oracle() {
    let w = weights(7);
    check_all(&w, &[0u64; 64], "empty");
    check_all(&w, &[u64::MAX; 64], "full");
    let mut s = 0x1234_5678_9ABC_DEF1;
    for (num, den) in [(1, 64), (1, 4), (1, 2), (3, 4)] {
        for i in 0..50 {
            let t = tile(&mut s, num, den);
            check_all(&w, &t, &format!("density {num}/{den} #{i}"));
        }
    }
}

/// The overflow bound is tight in both directions and still exact.
#[test]
fn the_extreme_admitted_weights_stay_exact() {
    for (v, b) in [(i16::MAX, 1 << 15), (-i16::MAX, -(1 << 15))] {
        let w = uniform(v, b);
        let full = [u64::MAX; 64];
        let want = i64::from(b) + 4096 * i64::from(v);
        assert!(want.unsigned_abs() < 1 << 28, "bound claimed in the module doc");
        assert_eq!(nnue_refresh(&w, &full), [want as i32; H]);
        assert_eq!(nnue_refresh_simd(&w, &full), [want as i32; H]);
        assert_eq!(nnue_refresh_simd2(&w, &full), [want as i32; H]);
        assert_eq!(nnue_refresh_simd4(&w, &full), [want as i32; H]);
    }
}

#[test]
fn weights_outside_the_bound_are_refused() {
    let mut w = vec![[[0i16; H]; 64]; 64].into_boxed_slice();
    w[17][33][5] = i16::MIN;
    assert_eq!(CellWeights::new([0; H], w).err(), Some(Refusal::TooLarge));
    let ok = vec![[[0i16; H]; 64]; 64].into_boxed_slice();
    assert!(CellWeights::new([1 << 15; H], ok.clone()).is_ok());
    assert!(CellWeights::new([-(1 << 15); H], ok.clone()).is_ok());
    let mut bias = [0; H];
    bias[3] = (1 << 15) + 1;
    assert_eq!(CellWeights::new(bias, ok).err(), Some(Refusal::TooLarge));
    let short = vec![[[0i16; H]; 64]; 63].into_boxed_slice();
    assert_eq!(CellWeights::new([0; H], short).err(), Some(Refusal::Shape));
}

// ---------------------------------------------------------------- P2

/// 1,000 updates of 1, 8 or 64 changed cells, mixed on and off, each checked
/// against a fresh refresh; then every change undone in reverse returns the
/// starting accumulator exactly.
#[test]
fn update_chains_match_a_fresh_refresh_and_undo_exactly() {
    let w = weights(99);
    let mut s = 0xDEAD_BEEF_CAFE_F00D;
    let start = tile(&mut s, 1, 2);
    let start_acc = nnue_refresh(&w, &start);
    let mut cur = start;
    let mut acc = start_acc;
    let mut acc_simd = start_acc;
    let mut history = Vec::new();
    let mut both_ways = 0;
    for step in 0..1000 {
        let n = [1, 8, 64][step % 3];
        let flip = cells(&mut s, n);
        let next = xor(&cur, &flip);
        if (0..64).any(|k| flip[k] & cur[k] != 0) && (0..64).any(|k| flip[k] & !cur[k] != 0) {
            both_ways += 1;
        }
        nnue_update(&w, &mut acc, &cur, &next);
        nnue_update_simd(&w, &mut acc_simd, &cur, &next);
        let want = oracle(&w, &next);
        assert_eq!(widen(&acc), want, "step {step}");
        assert_eq!(widen(&acc_simd), want, "step {step} simd");
        history.push(cur);
        cur = next;
    }
    assert!(both_ways > 300, "updates must turn cells on and off together: {both_ways}");
    for prev in history.into_iter().rev() {
        nnue_update(&w, &mut acc, &cur, &prev);
        nnue_update_simd(&w, &mut acc_simd, &cur, &prev);
        cur = prev;
    }
    assert_eq!(cur, start);
    assert_eq!(acc, start_acc);
    assert_eq!(acc_simd, start_acc);
}

/// Toggling one cell twice is the identity; the same cell many times in a
/// row stays exact.
#[test]
fn repeated_toggles_of_one_cell_return_to_the_start() {
    let w = weights(5);
    let mut s = 11;
    let t0 = tile(&mut s, 1, 4);
    let a0 = nnue_refresh(&w, &t0);
    let mut a = a0;
    let mut t = t0;
    for i in 0..101 {
        let mut next = t;
        next[42] ^= 1u64 << 7;
        nnue_update(&w, &mut a, &t, &next);
        t = next;
        assert_eq!(widen(&a), oracle(&w, &t), "toggle {i}");
    }
    assert_ne!(t, t0, "an odd number of toggles leaves the cell flipped");
    assert_ne!(a, a0);
}

// ---------------------------------------------------------------- P3

#[test]
fn refresh_and_update_do_not_allocate() {
    let w = weights(3);
    let mut s = 21;
    let old = tile(&mut s, 1, 2);
    let new = tile(&mut s, 1, 2);
    let mut acc = nnue_refresh(&w, &old);
    let before = allocs();
    let a = nnue_refresh(&w, &new);
    let b = nnue_refresh_simd(&w, &new);
    nnue_update(&w, &mut acc, &old, &new);
    nnue_update_simd(&w, &mut acc, &new, &old);
    let after = allocs();
    assert_eq!(after, before, "allocations inside refresh or update");
    assert_eq!(a, b);
    // the counter is live: an allocation in this thread moves it
    let v = std::hint::black_box(vec![0u8; 64]);
    assert!(allocs() > after, "the allocation counter must count");
    drop(v);
}

// ---------------------------------------------------------------- frames

fn source_with(id: u64, t: &BitTile) -> Source {
    let mut s = Source::blank(id);
    s.present = *t;
    s
}

#[test]
fn a_bound_frame_refreshes_and_updates_across_versions() {
    let w = weights(8);
    let mut seed = 31;
    let old_src = source_with(4, &tile(&mut seed, 1, 2));
    let mut new_src = old_src.bumped();
    new_src.set_present(9, 9, !(old_src.present[9] >> 9 & 1 == 1));
    new_src.set_present(63, 63, !(old_src.present[63] >> 63 & 1 == 1));
    let old = bind(&old_src, &Expect::of(&old_src)).unwrap();
    let new = bind(&new_src, &Expect::of(&new_src)).unwrap();
    let mut acc = refresh_bound(&w, &old).unwrap();
    assert_eq!(widen(&acc), oracle(&w, &old_src.present));
    update_bound(&w, &mut acc, &old, &new).unwrap();
    assert_eq!(widen(&acc), oracle(&w, &new_src.present));
    // the old frame still binds to its own expectation; only a consumer
    // expecting the old version is refused the new frame
    assert!(bind(&old_src, &Expect::of(&old_src)).is_ok());
    assert_eq!(bind(&new_src, &Expect::of(&old_src)).err(), Some(Refusal::Stale));
}

#[test]
fn incompatible_frames_are_refused_and_leave_the_accumulator_untouched() {
    let w = weights(9);
    let mut seed = 41;
    let a_src = source_with(1, &tile(&mut seed, 1, 2));
    let a = bind(&a_src, &Expect::of(&a_src)).unwrap();
    let start = refresh_bound(&w, &a).unwrap();

    let foreign_src = source_with(2, &tile(&mut seed, 1, 2));
    let foreign = bind(&foreign_src, &Expect::of(&foreign_src)).unwrap();

    let mut reordered_src = a_src.bumped();
    reordered_src.row_order.swap(0, 1);
    let reordered = bind(&reordered_src, &Expect::of(&reordered_src)).unwrap();

    let short_witness = Witness { rows: 63, ..a.witness };
    let short = Bound {
        src: &a_src,
        witness: short_witness,
    };

    let blind_src = a_src.without_field(F_PRESENT);
    let blind = bind(
        &blind_src,
        &Expect {
            fields: F_STOCK,
            ..Expect::of(&blind_src)
        },
    )
    .unwrap();

    for (other, why) in [
        (&foreign, Refusal::ForeignSource),
        (&reordered, Refusal::RowOrder),
        (&short, Refusal::Shape),
        (&blind, Refusal::Field),
    ] {
        let mut acc = start;
        assert_eq!(update_bound(&w, &mut acc, &a, other), Err(why));
        assert_eq!(acc, start, "{why:?}: accumulator changed on refusal");
    }
    assert_eq!(refresh_bound(&w, &blind), Err(Refusal::Field));
}

// ---------------------------------------------------------------- P6

/// A Boolean tile is a set. A bag of edges with a repeated `(k, r)` weighs
/// that cell once per edge; the tile weighs it once. Equal only without
/// repeats.
#[test]
fn a_tile_drops_multiplicity_that_a_bag_of_edges_keeps() {
    let w = weights(12);
    let bag_sum = |edges: &[(usize, usize)]| -> [i64; H] {
        let mut acc = widen(&w.bias());
        for &(k, r) in edges {
            for j in 0..H {
                acc[j] += i64::from(w.row(k, r)[j]);
            }
        }
        acc
    };
    let to_tile = |edges: &[(usize, usize)]| -> BitTile {
        let mut t = [0u64; 64];
        for &(k, r) in edges {
            t[k] |= 1u64 << r;
        }
        t
    };
    let distinct = [(0, 0), (5, 63), (63, 1)];
    assert_eq!(widen(&nnue_refresh(&w, &to_tile(&distinct))), bag_sum(&distinct));
    let repeated = [(0, 0), (5, 63), (5, 63), (63, 1)];
    assert_ne!(
        widen(&nnue_refresh(&w, &to_tile(&repeated))),
        bag_sum(&repeated),
        "the tile must not be read as carrying the repeated edge"
    );
    assert_eq!(to_tile(&repeated), to_tile(&distinct));
}
