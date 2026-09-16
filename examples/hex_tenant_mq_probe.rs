//! W0 probe D-GTM-0m — the hex TENANT: static top-down traversal AND
//! plasticity (spread) in ONE SoA substrate, and the Mississippi Queen cost
//! model `step = x·ternlogq + n`, coal = the re-chain budget.
//!
//! Plan: `.claude/plans/gemm-ternlog-mask-consolidation-v1.md` §9 (M1/M1b/M2/M3)
//! and §11.10 (`substrate == mask geometry == projection surface`). Operator
//! statement (2026-09-14): *"static traversal top down AND plasticity (spread)
//! in the same substrate — SoA gets a hex tenant with 6×2×8 bit and the field
//! is a trie (fixed spatial distribution)."*
//!
//! ## The substrate
//!
//! 65,536 rows = a 256×256 axial hex field `(q, r)`. **Row index = Morton(q, r)**
//! (8+8 bits interleaved), so the field IS the nibble trie: a trie node at
//! level `L` (4·L address bits) is a CONTIGUOUS run of `2^(16-4L)` rows. The
//! tenant payload is the V3 12-byte register read as `6×(u8:u8)`: rail `d` =
//! hex direction `d`, `u8:u8` = `(permeability, strength)`. Adjacency and
//! carving are the SAME six here — that is the claim §9 R1 left open and this
//! probe builds rather than argues.
//!
//! ## The two readings, one register
//!
//! * **white — top-down (static, replayable):** revealing a trie node is a
//!   RANGE mask (bits `[lo, hi)`), never a compare sweep. The general (non-
//!   fixed-distribution) path is `ternary_match_u32_to_mask` over an address
//!   column — a TCAM sweep of the whole column. Both are timed; both must
//!   yield the same mask.
//! * **grey — spread (plastic, winner-update):** `state' = (state ∪ ⋃_d
//!   shift_d(state ∧ elig_d)) ∧ tile`, chained through `x` resident gate masks
//!   with `mask_ternlog_assign::<AND3>`. Each cell newly reached via rail `d`
//!   fires that rail's `strength` byte (saturating) in place — Hebbian, one
//!   owner, no second structure. Degree-1 control per the E-Q8 rule: the
//!   same run with ONE direction; any advantage that survives it is not hex.
//!
//! ## The cost model under test
//!
//! Maintained speed `x` (chain depth) costs `x · t_ternlogq + n` per step,
//! where `n` is the reveal/spread term. A maneuver (re-chaining a resident
//! mask from its column) costs coal `c`. The probe fits `t_ternlogq` and `n`
//! from a depth ladder and reports `c` in units of maintained steps.
//!
//! ## Gates (a mismatch aborts the run)
//!
//! * range reveal == TCAM reveal, every level, every prefix tried;
//! * Morton-arm spread == an independent row-major axial BFS, every step;
//! * hot-path heap bytes == 0 (counting allocator).
//!
//!   cargo run --release --example hex_tenant_mq_probe --features std

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::simd::ternlog::{AND3, OR2_AND};
use ndarray::simd::{
    gt_i32_to_mask, mask_and, mask_set_range, mask_shift_morton, mask_ternlog_assign, popcount_batch_u64,
    ternary_match_u32_to_mask, MortonDir,
};

// ── counting allocator: the 0k instrument ────────────────────────────────────
static ALLOCED: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicUsize = AtomicUsize::new(0);

struct Counting;
// SAFETY: a pure pass-through allocator. Every call forwards the caller's own
// `Layout` (and, for `dealloc`, the pointer that `alloc` returned for that
// layout) to `System` unchanged, so `System` upholds the `GlobalAlloc`
// contract on our behalf; the only added work is a relaxed atomic counter
// that never touches the allocation.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) == 1 {
            ALLOCED.fetch_add(l.size(), Ordering::Relaxed);
        }
        // SAFETY: `l` is the layout the caller passed, forwarded verbatim.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` was returned by `System.alloc(l)` above for this same `l`.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

fn count_on() {
    ALLOCED.store(0, Ordering::Relaxed);
    COUNTING.store(1, Ordering::Relaxed);
}
fn count_off() -> usize {
    COUNTING.store(0, Ordering::Relaxed);
    ALLOCED.load(Ordering::Relaxed)
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ── the field: 256×256 axial, Morton-keyed ───────────────────────────────────
const SIDE: usize = 256;
const N: usize = SIDE * SIDE; // 65,536 rows
const WORDS: usize = N / 64; // 1,024 mask words = 8 KiB
const DIRS: usize = 6;
const X_BITS: u32 = 0x5555; // q lives in the even bit positions
const Y_BITS: u32 = 0xAAAA; // r lives in the odd bit positions

/// Axial hex neighbours: (dq, dr) for rails 0..6.
const HEX: [(i32, i32); DIRS] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];

fn dilate8(v: u32) -> u32 {
    let mut x = v & 0xFF;
    x = (x | (x << 4)) & 0x0F0F;
    x = (x | (x << 2)) & 0x3333;
    (x | (x << 1)) & 0x5555
}
fn morton(q: u32, r: u32) -> u32 {
    dilate8(q) | (dilate8(r) << 1)
}
fn compact8(v: u32) -> u32 {
    let mut x = v & 0x5555;
    x = (x | (x >> 1)) & 0x3333;
    x = (x | (x >> 2)) & 0x0F0F;
    (x | (x >> 4)) & 0x00FF
}
fn axial(m: u32) -> (u32, u32) {
    (compact8(m), compact8(m >> 1))
}

/// Neighbour of Morton cell `m` along rail `d`, by dilated-integer arithmetic —
/// no table, no deinterleave on the hot path. `None` at the field boundary.
#[inline]
fn neighbour(m: u32, d: usize) -> Option<u32> {
    let (dq, dr) = HEX[d];
    let mut x = m & X_BITS;
    let mut y = m & Y_BITS;
    match dq {
        1 => {
            if x == X_BITS {
                return None;
            }
            x = ((x | Y_BITS) + 1) & X_BITS;
        }
        -1 => {
            if x == 0 {
                return None;
            }
            x = (x - 1) & X_BITS;
        }
        _ => {}
    }
    match dr {
        1 => {
            if y == Y_BITS {
                return None;
            }
            y = ((y | X_BITS) + 1) & Y_BITS;
        }
        -1 => {
            if y == 0 {
                return None;
            }
            y = (y - 1) & Y_BITS;
        }
        _ => {}
    }
    Some(x | y)
}

#[inline]
fn bit(words: &[u64], i: usize) -> bool {
    (words[i >> 6] >> (i & 63)) & 1 == 1
}
#[inline]
fn set(words: &mut [u64], i: usize) {
    words[i >> 6] |= 1u64 << (i & 63);
}

/// A trie node at nibble level `level` (0..=4) with prefix `p`: rows
/// `[p << shift, (p+1) << shift)`. The fixed spatial distribution makes the
/// reveal a range write, not a compare — `ndarray::simd::mask_set_range` is
/// exactly that primitive (N1,
/// `.claude/plans/gemm-ternlog-mask-consolidation-v1.md` §16.6), so the
/// range write itself lives there now; this function only computes the
/// `(lo, hi)` node bounds.
fn range_reveal(level: u32, p: u32, out: &mut [u64]) -> (usize, usize) {
    let shift = 16 - 4 * level;
    let lo = (p as usize) << shift;
    let hi = ((p + 1) as usize) << shift;
    mask_set_range(out, lo, hi);
    (lo, hi)
}

fn tcam_reveal(addr: &[u32], level: u32, p: u32, out: &mut [u64]) {
    let shift = 16 - 4 * level;
    let care = if shift == 32 { 0 } else { !((1u32 << shift) - 1) };
    ternary_match_u32_to_mask(addr, p << shift, care, out);
}

/// One spread step on the Morton arm. Returns the number of rail firings.
/// `scratch` receives the shifted frontier; the chain then narrows in place.
/// The resident (white, replayable) inputs of one spread step: laid once per
/// mask generation, borrowed by every step — the M1b tile.
struct Resident<'a> {
    elig: &'a [Vec<u64>; DIRS],
    tile: &'a [u64],
    gates: &'a [Vec<u64>],
    x: usize,
    dirs: usize,
    from_delta: bool,
    /// `Some((lo, hi))` = run the word-level shifts over ONLY this word range
    /// (the trie node's own contiguous span — a square Morton sub-field), so
    /// the op's cost is bound by the NODE, not the field. `None` = full field.
    word_range: Option<(usize, usize)>,
}

#[inline(never)]
/// The Hebbian reverse walk shared by both spread arms: for every cell `j`
/// newly reached this step (`scratch & !state`), bump the strength byte of
/// each rail `d` whose predecessor `i` (j's neighbour along the opposite rail)
/// is BOTH in this step's `source` frontier AND eligible to leave along `d`.
/// Returns the number of firings. Reads `source` and `state`, writes only
/// `rails` — the caller publishes the new `delta` afterwards, because under
/// `from_delta` the source IS the old delta and must survive the walk.
fn hebbian_reverse_walk(
    scratch: &[u64], state: &[u64], source: &[u64], elig: &[Vec<u64>], dirs: usize, rails: &mut [[u8; 12]],
) -> usize {
    let mut fired = 0usize;
    for wi in 0..scratch.len() {
        let mut new_bits = scratch[wi] & !state[wi];
        while new_bits != 0 {
            let j = (wi << 6) | new_bits.trailing_zeros() as usize;
            new_bits &= new_bits - 1;
            for d in 0..dirs {
                // the rail that reaches j from i along d is d; i is j's neighbour along the reverse rail
                let rev = d ^ 1; // HEX pairs (0,1) (2,3) (4,5) are opposites
                if let Some(i) = neighbour(j as u32, rev) {
                    let i = i as usize;
                    if bit(source, i) && bit(&elig[d], i) {
                        rails[i][2 * d + 1] = rails[i][2 * d + 1].saturating_add(1);
                        fired += 1;
                    }
                }
            }
        }
    }
    fired
}

fn spread_step(
    state: &mut [u64], delta: &mut [u64], scratch: &mut [u64], res: &Resident<'_>, rails: &mut [[u8; 12]],
) -> usize {
    let Resident {
        elig,
        tile,
        gates,
        x,
        dirs,
        from_delta,
        word_range: _,
    } = *res;
    for w in scratch.iter_mut() {
        *w = 0;
    }
    // the ONLY non-mask op on the path: the hex shift, per active bit. This is
    // the `n` term — and the primitive the substrate does not have yet.
    // NNUE reading: spread from the DELTA frontier (cells reached last step),
    // never from the whole accumulated state — same closure, `n ∝ |delta|`.
    let source: &[u64] = if from_delta { delta } else { state };
    for (wi, &w) in source.iter().enumerate() {
        let mut bits = w;
        while bits != 0 {
            let i = (wi << 6) | bits.trailing_zeros() as usize;
            bits &= bits - 1;
            for (d, e) in elig.iter().enumerate().take(dirs) {
                if bit(e, i) {
                    if let Some(j) = neighbour(i as u32, d) {
                        set(scratch, j as usize);
                    }
                }
            }
        }
    }
    // scratch = (scratch | state) & tile, then the chain narrows scratch in
    // place: x resident gate masks, one AND3 pass each. `state` stays the
    // previous frontier until the survivors are known.
    mask_ternlog_assign::<OR2_AND>(scratch, state, tile);
    for k in 0..x {
        let g1 = &gates[k % gates.len()];
        let g2 = &gates[(k + 1) % gates.len()];
        mask_ternlog_assign::<AND3>(scratch, g1, g2);
    }
    // Hebbian, on SURVIVORS only: a rail fires when the cell it points at
    // actually became active after the chain — reverse walk from each newly
    // reached cell to the eligible neighbour that could have carried it.
    // The predecessor is tested against the SOURCE frontier of this step,
    // not the accumulated state: under `from_delta` only delta cells
    // propagated, so an older active neighbour of a newly reached cell did
    // not carry it and must not be credited (codex P2 on #307). `delta` is
    // therefore published only after attribution — it IS the source while
    // the walk runs.
    let fired = hebbian_reverse_walk(scratch, state, source, elig, dirs, rails);
    for wi in 0..scratch.len() {
        delta[wi] = scratch[wi] & !state[wi];
    }
    state.copy_from_slice(scratch);
    fired
}

/// The same step as [`spread_step`], but with the per-active-bit hex
/// neighbour loop — the `n` term §14/§15 name as the whole cost of a spread
/// step — replaced by [`mask_shift_morton`] over whole words. For each axis
/// rail `d`, the eligibility mask is applied to the SOURCE first (`elig[d]`
/// bit `i` means "cell `i` may leave along `d`", so `source & elig[d]`
/// selects the cells allowed to leave, THEN the shift moves them), and the
/// shifted result is OR-accumulated into `scratch`. The two diagonal rails
/// (4 = `+q-r`, 5 = `-q+r`) are two composed calls through `diag_scratch`
/// (§15: shift the masked source along the FIRST axis into `diag_scratch`,
/// then shift `diag_scratch` — never the masked source again — along the
/// SECOND axis into `scratch`). Everything from the tile/gate chain onward,
/// including the Hebbian reverse walk, is copied verbatim from
/// [`spread_step`] — this function changes only how `scratch` is built.
#[inline(never)]
fn spread_step_shift(
    state: &mut [u64], delta: &mut [u64], scratch: &mut [u64], res: &Resident<'_>, rails: &mut [[u8; 12]],
    masked: &mut [u64], diag_scratch: &mut [u64],
) -> usize {
    let Resident {
        elig,
        tile,
        gates,
        x,
        dirs,
        from_delta,
        word_range,
    } = *res;
    for w in scratch.iter_mut() {
        *w = 0;
    }
    let source: &[u64] = if from_delta { delta } else { state };
    // The word-level shifts run over the node's own span when one is given.
    // Correctness is unchanged: the source is a subset of the tile, so no
    // carry can arrive from outside the span, and a carry LEAVING the span
    // would be removed by the `& tile` below anyway.
    let (lo, hi) = word_range.unwrap_or((0, scratch.len()));
    // `mask_shift_morton` treats the slice AS the field (no carry across the
    // span boundary), which is only the restriction of the full-field shift
    // when the span is a Morton-aligned node: its origin is a multiple of
    // its own length. Enforced where it is known, not assumed.
    assert_eq!(lo % (hi - lo), 0, "word_range {lo}..{hi} is not a Morton-aligned node span");
    let src = &source[lo..hi];
    let masked = &mut masked[lo..hi];
    let diag_scratch = &mut diag_scratch[lo..hi];
    let out = &mut scratch[lo..hi];

    const AXES: [MortonDir; 4] = [MortonDir::PosQ, MortonDir::NegQ, MortonDir::PosR, MortonDir::NegR];
    for (d, &dir) in AXES.iter().enumerate().take(dirs.min(4)) {
        mask_and(src, &elig[d][lo..hi], masked);
        mask_shift_morton(masked, dir, out);
    }
    if dirs > 4 {
        // rail 4 = +q-r
        mask_and(src, &elig[4][lo..hi], masked);
        for w in diag_scratch.iter_mut() {
            *w = 0;
        }
        mask_shift_morton(masked, MortonDir::PosQ, diag_scratch);
        mask_shift_morton(diag_scratch, MortonDir::NegR, out);
    }
    if dirs > 5 {
        // rail 5 = -q+r (the mirror composition)
        mask_and(src, &elig[5][lo..hi], masked);
        for w in diag_scratch.iter_mut() {
            *w = 0;
        }
        mask_shift_morton(masked, MortonDir::NegQ, diag_scratch);
        mask_shift_morton(diag_scratch, MortonDir::PosR, out);
    }

    // scratch = (scratch | state) & tile, then the chain narrows scratch in
    // place: x resident gate masks, one AND3 pass each — verbatim from
    // `spread_step`.
    mask_ternlog_assign::<OR2_AND>(scratch, state, tile);
    for k in 0..x {
        let g1 = &gates[k % gates.len()];
        let g2 = &gates[(k + 1) % gates.len()];
        mask_ternlog_assign::<AND3>(scratch, g1, g2);
    }
    // Hebbian reverse walk — the same attribution as `spread_step`, against
    // the step's SOURCE frontier; `delta` is published after it.
    let fired = hebbian_reverse_walk(scratch, state, source, elig, dirs, rails);
    for wi in 0..scratch.len() {
        delta[wi] = scratch[wi] & !state[wi];
    }
    state.copy_from_slice(scratch);
    fired
}

/// Independent reference: row-major axial BFS with the same permeability,
/// tile and gate semantics, no Morton, no masks.
fn reference_step(
    state: &mut [bool], perm: &[[u8; 12]], thr: u8, tile: &[bool], gates: &[Vec<bool>], x: usize, dirs: usize,
) {
    let mut next = state.to_vec();
    for q in 0..SIDE as i32 {
        for r in 0..SIDE as i32 {
            let i = (q as usize) * SIDE + r as usize;
            if !state[i] {
                continue;
            }
            let m = morton(q as u32, r as u32) as usize;
            for (d, &(dq, dr)) in HEX.iter().enumerate().take(dirs) {
                if perm[m][2 * d] <= thr {
                    continue;
                }
                let (nq, nr) = (q + dq, r + dr);
                if nq < 0 || nr < 0 || nq >= SIDE as i32 || nr >= SIDE as i32 {
                    continue;
                }
                next[(nq as usize) * SIDE + nr as usize] = true;
            }
        }
    }
    for i in 0..N {
        let mut v = next[i] && tile[i];
        for k in 0..x {
            v = v && gates[k % gates.len()][i] && gates[(k + 1) % gates.len()][i];
        }
        state[i] = v;
    }
}

fn masks_equal_axial(mask: &[u64], reference: &[bool]) -> bool {
    (0..N).all(|i| {
        let (q, r) = axial(i as u32);
        bit(mask, i) == reference[(q as usize) * SIDE + r as usize]
    })
}

fn timed<F: FnMut()>(mut f: F) -> f64 {
    // run to a 50 ms floor; report ns per call
    let mut reps = 1usize;
    loop {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        let el = t.elapsed();
        if el.as_millis() >= 50 {
            return el.as_nanos() as f64 / reps as f64;
        }
        reps *= 2;
    }
}

fn main() {
    let mut seed = 0x5EED_0000_0000_0001u64;
    println!("hex tenant probe — N={N} rows (256×256 axial, Morton-keyed), {WORDS} words/mask");

    // ── the tenant: 12-byte register per row, 6×(perm:strength) ────────────
    let rails: Vec<[u8; 12]> = (0..N)
        .map(|_| {
            let mut r = [0u8; 12];
            for d in 0..DIRS {
                r[2 * d] = (splitmix(&mut seed) & 0xFF) as u8; // permeability
                r[2 * d + 1] = 0; // strength (plastic)
            }
            r
        })
        .collect();
    let thr: u8 = 96; // ~62% of rails permeable
                      // Column views for the SIMD compare. The T1 compare is i32-wide today; a
                      // u8 column compare is a T1 addition (stated, not hidden). Widening costs
                      // 4× the bandwidth, so `n_gen` below is an UPPER bound on the reveal term.
    let perm_cols: Vec<Vec<i32>> = (0..DIRS)
        .map(|d| rails.iter().map(|r| r[2 * d] as i32).collect())
        .collect();
    let addr: Vec<u32> = (0..N as u32).collect();

    // ── M1b: eligibility masks laid ONCE per mask generation ─────────────────
    let mut elig: [Vec<u64>; DIRS] = std::array::from_fn(|_| vec![0u64; WORDS]);
    let t_gen = timed(|| {
        for d in 0..DIRS {
            gt_i32_to_mask(&perm_cols[d], thr as i32, &mut elig[d]);
        }
    });
    println!(
        "M1b generation: 6 eligibility masks from 6 columns = {:.0} ns ({:.0} ns/mask, {:.2} ns/row)",
        t_gen,
        t_gen / DIRS as f64,
        t_gen / (DIRS * N) as f64
    );

    // ── white: top-down reveal, range vs TCAM, every level ───────────────────
    println!("\n[white] top-down reveal — fixed-distribution RANGE vs TCAM compare sweep");
    println!("level  rows/node  range ns   tcam ns   ratio   gate");
    let mut m_range = vec![0u64; WORDS];
    let mut m_tcam = vec![0u64; WORDS];
    for level in 0..=4u32 {
        let n_nodes = 1u32 << (4 * level);
        let mut ok = true;
        for p in [0u32, n_nodes / 2, n_nodes - 1] {
            range_reveal(level, p, &mut m_range);
            tcam_reveal(&addr, level, p, &mut m_tcam);
            ok &= m_range == m_tcam && popcount_batch_u64(&m_range) as usize == (1usize << (16 - 4 * level));
        }
        assert!(ok, "reveal gate FAILED at level {level}");
        let p = n_nodes / 2;
        let tr = timed(|| {
            range_reveal(level, p, &mut m_range);
        });
        let tt = timed(|| {
            tcam_reveal(&addr, level, p, &mut m_tcam);
        });
        println!("{level:>5}  {:>9}  {tr:>8.0}  {tt:>8.0}  {:>6.1}×  ok", 1usize << (16 - 4 * level), tt / tr);
    }

    // ── grey: spread inside a level-1 tile from a level-2 seed ───────────────
    // tile = level-1 node p=5 (4096 cells); seed = its first level-2 child.
    let mut tile = vec![0u64; WORDS];
    let (tlo, thi) = range_reveal(1, 5, &mut tile);
    let mut tile_ref = vec![false; N];
    for i in tlo..thi {
        let (q, r) = axial(i as u32);
        tile_ref[(q as usize) * SIDE + r as usize] = true;
    }
    // resident gate masks: 4 random 90%-dense masks (the "rung/tenant" gates)
    let n_gates = 4;
    let gates: Vec<Vec<u64>> = (0..n_gates)
        .map(|_| {
            (0..WORDS)
                .map(|_| {
                    let mut w = u64::MAX;
                    for _ in 0..6 {
                        w &= !(1u64 << (splitmix(&mut seed) & 63));
                    }
                    w
                })
                .collect()
        })
        .collect();
    let gates_ref: Vec<Vec<bool>> = gates
        .iter()
        .map(|g| {
            let mut v = vec![false; N];
            for i in 0..N {
                let (q, r) = axial(i as u32);
                v[(q as usize) * SIDE + r as usize] = bit(g, i);
            }
            v
        })
        .collect();

    let ones: Vec<Vec<u64>> = vec![vec![u64::MAX; WORDS]; n_gates];
    let ones_ref: Vec<Vec<bool>> = vec![vec![true; N]; n_gates];

    let steps = 24usize;
    println!("\n[grey] spread — Morton arm vs row-major axial BFS reference, {steps} steps, tile=4096 cells");
    println!("The LADDER arm uses identity gates so the survivor population is held fixed across x");
    println!("(otherwise the gates shrink the frontier and n moves with x — the fit measures nothing).");
    println!("The GATED arm uses the 4 real 90%-dense gates: what a rung/tenant chain does to the spread.");
    println!("The NNUE arm spreads from the DELTA frontier only (nnue+g = with the real gates).");
    println!(
        "The SHIFT arms use mask_shift_morton over the FULL field; the NODE arms over the tile's own 64-word span."
    );
    println!("The SHIFT arm replaces the per-bit hex loop with mask_shift_morton over whole words —");
    println!("the D-GTM-1m primitive; same identity/real-gate split as ladder/gated (shift+g).");
    println!("arm        dirs  x   ns/step   fired/step  heap B/step  gate");
    let mut fits: Vec<(usize, f64)> = Vec::new();
    // the tile is a level-1 node = one 64x64 Morton sub-field = 64 contiguous
    // words; `node` arms run the word-level shifts over exactly that span
    let tile_words = (tlo >> 6, thi >> 6);
    for (arm, gates, gates_ref, from_delta, use_shift, word_range) in [
        ("ladder", &ones, &ones_ref, false, false, None),
        ("gated", &gates, &gates_ref, false, false, None),
        ("nnue", &ones, &ones_ref, true, false, None),
        ("nnue+g", &gates, &gates_ref, true, false, None),
        ("shift", &ones, &ones_ref, false, true, None),
        ("shift+g", &gates, &gates_ref, false, true, None),
        ("node", &ones, &ones_ref, false, true, Some(tile_words)),
        ("node+g", &gates, &gates_ref, false, true, Some(tile_words)),
        ("node+nn", &ones, &ones_ref, true, true, Some(tile_words)),
        ("node+nn+g", &gates, &gates_ref, true, true, Some(tile_words)),
    ] {
        for dirs in [6usize, 1] {
            for x in [0usize, 1, 2, 4, 8, 16, 32] {
                if arm != "ladder" && !matches!(x, 0 | 4 | 32) {
                    continue;
                }
                // fresh state + reference, same seed node every time
                let mut seed_state = vec![0u64; WORDS];
                range_reveal(2, 5 * 16, &mut seed_state); // first level-2 child of tile 5
                let mut reference = vec![false; N];
                for i in 0..N {
                    if bit(&seed_state, i) {
                        let (q, r) = axial(i as u32);
                        reference[(q as usize) * SIDE + r as usize] = true;
                    }
                }
                let res = Resident {
                    elig: &elig,
                    tile: &tile,
                    gates,
                    x,
                    dirs,
                    from_delta,
                    word_range,
                };
                let mut state = vec![0u64; WORDS];
                let mut delta = vec![0u64; WORDS];
                let mut scratch = vec![0u64; WORDS];
                // only the `shift`/`shift+g` arms touch these; allocated
                // unconditionally (outside the timed region, negligible) so
                // the dispatch below stays a plain branch, not a second copy
                // of the buffer setup per arm.
                let mut masked = vec![0u64; WORDS];
                let mut diag_scratch = vec![0u64; WORDS];
                let mut rails_run = rails.clone();
                let mut fired_total = 0usize;
                let mut ok = true;
                // repeat the whole 24-step run to a 50 ms floor; the reset is an
                // reset is two 8 KiB copies PLUS the 786 KiB `rails_run` copy
                // (65 536 × 12 B), all inside the timed region (stated): it
                // amortizes to ~1 µs per step and sits inside every arm's `n`.
                let mut reps = 1usize;
                let (el, heap) = loop {
                    count_on();
                    let t = Instant::now();
                    for _ in 0..reps {
                        state.copy_from_slice(&seed_state);
                        delta.copy_from_slice(&seed_state);
                        rails_run.copy_from_slice(&rails);
                        fired_total = 0;
                        for _ in 0..steps {
                            fired_total += if use_shift {
                                spread_step_shift(
                                    &mut state, &mut delta, &mut scratch, &res, &mut rails_run, &mut masked,
                                    &mut diag_scratch,
                                )
                            } else {
                                spread_step(&mut state, &mut delta, &mut scratch, &res, &mut rails_run)
                            };
                        }
                    }
                    let e = t.elapsed();
                    let heap = count_off();
                    // The header names this as a gate, so it aborts like the
                    // other two rather than silently invalidating the ns/step.
                    assert_eq!(
                        heap, 0,
                        "heap gate FAILED arm={arm} dirs={dirs} x={x}: {heap} bytes allocated on the hot path"
                    );
                    if e.as_millis() >= 50 {
                        break (e.as_nanos() as f64 / (reps * steps) as f64, heap / reps);
                    }
                    reps *= 2;
                };
                // the correctness gate runs OUTSIDE the timed/counted region
                let mut state_chk = seed_state.clone();
                let mut delta_chk = seed_state.clone();
                let mut scratch_chk = vec![0u64; WORDS];
                let mut masked_chk = vec![0u64; WORDS];
                let mut diag_scratch_chk = vec![0u64; WORDS];
                let mut rails_chk = rails.clone();
                for _ in 0..steps {
                    if use_shift {
                        spread_step_shift(
                            &mut state_chk, &mut delta_chk, &mut scratch_chk, &res, &mut rails_chk, &mut masked_chk,
                            &mut diag_scratch_chk,
                        );
                    } else {
                        spread_step(&mut state_chk, &mut delta_chk, &mut scratch_chk, &res, &mut rails_chk);
                    }
                    reference_step(&mut reference, &rails, thr, &tile_ref, gates_ref, x, dirs);
                    ok &= masks_equal_axial(&state_chk, &reference);
                }
                assert!(ok, "spread gate FAILED arm={arm} dirs={dirs} x={x}");
                assert_eq!(state, state_chk, "timed run diverged from checked run");
                assert_eq!(rails_run, rails_chk, "plasticity diverged between runs");
                if arm == "ladder" && dirs == 6 {
                    fits.push((x, el));
                }
                println!(
                    "{arm:<10} {dirs:>4}  {x:>2}  {el:>8.0}   {:>10.1}  {heap:>11}  ok  (survivors {})",
                    fired_total as f64 / steps as f64,
                    popcount_batch_u64(&state)
                );
                if arm == "ladder" && dirs == 6 && x == 0 {
                    // plasticity landed in the register, not beside it
                    let bumped = rails_run
                        .iter()
                        .zip(rails.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                    println!("      plasticity: {bumped} rows' strength bytes changed in place, {fired_total} firings");
                }
            }
        }
    }

    // ── the fit: step = x·t + n over the degree-6 ladder ─────────────────────
    let k = fits.len() as f64;
    let sx: f64 = fits.iter().map(|(x, _)| *x as f64).sum();
    let sy: f64 = fits.iter().map(|(_, y)| *y).sum();
    let sxx: f64 = fits.iter().map(|(x, _)| (*x as f64).powi(2)).sum();
    let sxy: f64 = fits.iter().map(|(x, y)| *x as f64 * y).sum();
    let t_tern = (k * sxy - sx * sy) / (k * sxx - sx * sx);
    let n_term = (sy - t_tern * sx) / k;
    let resid = fits
        .iter()
        .map(|(x, y)| (y - (t_tern * *x as f64 + n_term)).abs() / y)
        .fold(0.0f64, f64::max);
    println!("\n[fit] step = x·ternlogq + n   ⇒  ternlogq = {t_tern:.1} ns/pass ({:.3} ns/word), n = {n_term:.0} ns, max rel residual {:.1}%",
        t_tern / WORDS as f64, resid * 100.0);

    // ── M3: coal — one maneuver = re-chain a resident mask from its column ──
    let mut m_new = vec![0u64; WORDS];
    let c = timed(|| gt_i32_to_mask(&perm_cols[0], thr as i32, &mut m_new));
    println!(
        "[coal] one re-chain (gt_i32 sweep over one column) = {c:.0} ns = {:.1} ternlogq passes = {:.2} maintained steps at x=4",
        c / t_tern,
        c / (4.0 * t_tern + n_term)
    );
    println!(
        "[M2]   speed change x→x±1 costs one ternlogq pass ({t_tern:.0} ns); x→x±k costs k passes — linear, no cliff"
    );
}
