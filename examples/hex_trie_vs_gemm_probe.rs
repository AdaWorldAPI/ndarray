//! W0 probes D-GTM-0g/0h/0i/0j/0k — the mask/trie field against dense GEMM.
//!
//! Plan: `.claude/plans/gemm-ternlog-mask-consolidation-v1.md` §11.7.
//! Hypothesis under test (§11.1 pt 6, operator's own fence): **NOT** "TERNLOGQ
//! replaces GEMM". GEMM is attractive when information is dense; a learned
//! hex/trie field may win when cognition is mostly successive elimination of
//! possibility. The invariant (§11.10): `substrate == mask geometry ==
//! projection surface` — expanding a mask into IDs or materializing a neighbour
//! list on the hot path is the loss condition.
//!
//! ## The task, identical for both arms
//!
//! `D` steps of `state = R(state) ∩ constraint_i` over `N` items — one hop
//! through a relation, then a filter. Both arms must return the same survivor
//! set; a mismatch aborts the run (a comparison of two different computations
//! measures nothing).
//!
//! ## Two relation shapes — this is what makes the probe two-sided
//!
//! * `Prefix` — `R(x)` = items sharing `x`'s prefix. The hypothesis's home
//!   turf: successors of a SET are the union of touched prefix buckets, and a
//!   bucket's member mask is a contiguous run of words, so propagation is a
//!   word scan with no ID list and no allocation.
//! * `Random` — an arbitrary sparse relation with the SAME edge count and no
//!   prefix structure. The mask arm has nothing to exploit and must OR one
//!   successor row per active bit. If the mask arm does not lose here, the
//!   probe is rigged.
//!
//! ## Metrics
//!
//! * time/step (0g, 0j) across density (0i) and chain depth (0h — a timing
//!   proxy; `perf` is unavailable in this sandbox, so residency is inferred
//!   from time/step vs depth, and that limit is stated rather than hidden).
//! * **bytes materialized per step (0k)** — a counting allocator, reported
//!   separately for one-time setup (amortized per §11.6) and the hot path. The
//!   invariant predicts the mask arm's hot path approaches zero.
//!
//!   cargo run --release --example hex_trie_vs_gemm_probe --features std

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// ── counting allocator: the 0k instrument ────────────────────────────────────
static ALLOCED: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicUsize = AtomicUsize::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) == 1 {
            ALLOCED.fetch_add(l.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
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

const N: usize = 4096;
const W: usize = N / 64; // 64 mask words = 512 bytes
const BUCKET: usize = 64; // prefix bucket = 1 word = 64 items

#[derive(Clone, Copy, PartialEq)]
enum Rel {
    Prefix,
    Random,
}

/// Dense 0/1 relation matrix, `N×N` f32 — what the GEMM arm needs materialized.
fn build_matrix_deg(rel: Rel, seed: u64, deg: usize) -> Vec<f32> {
    let mut m = vec![0.0f32; N * N];
    let mut s = seed;
    match rel {
        Rel::Prefix => {
            // successors of x = x's bucket. Edge count = N * BUCKET.
            for i in 0..N {
                let b = i / BUCKET;
                for j in b * BUCKET..(b + 1) * BUCKET {
                    m[i * N + j] = 1.0;
                }
            }
        }
        Rel::Random => {
            // SAME edge count, no prefix structure.
            for i in 0..N {
                for _ in 0..deg {
                    let j = (splitmix(&mut s) as usize) % N;
                    m[i * N + j] = 1.0;
                }
            }
        }
    }
    m
}

/// Forward masks — `fwd[j]` = the items `j` can activate, i.e. the TRANSPOSE of
/// the relation matrix's rows.
///
/// The transpose is load-bearing and the probe's own correctness gate is what
/// found it: the GEMM arm computes `{ i : srcs(i) ∩ active ≠ ∅ }`, so the mask
/// arm must union `fwd[j]` over active `j`, never `srcs(i)` over active `i`.
/// Those two agree only for a SYMMETRIC relation — true of bucket membership,
/// false of a random relation, which is exactly where the gate fired.
fn build_fwd_masks(matrix: &[f32]) -> Vec<u64> {
    let mut t = vec![0u64; N * W];
    for i in 0..N {
        for j in 0..N {
            if matrix[i * N + j] != 0.0 {
                t[j * W + i / 64] |= 1u64 << (i % 64);
            }
        }
    }
    t
}

fn dense_state(mask: &[u64]) -> Vec<f32> {
    (0..N)
        .map(|i| if mask[i / 64] >> (i % 64) & 1 == 1 { 1.0 } else { 0.0 })
        .collect()
}

/// GEMM arm: one hop = matvec through the dense relation, threshold, filter.
fn step_gemm(matrix: &[f32], state: &mut [f32], scratch: &mut [f32], constraint: &[f32]) {
    ndarray::backend::gemm_f32(N, 1, N, 1.0, matrix, N, state, 1, 0.0, scratch, 1);
    for i in 0..N {
        state[i] = if scratch[i] > 0.0 { constraint[i] } else { 0.0 };
    }
}

/// Mask arm, `Prefix`: propagation is a word scan — a non-empty bucket becomes
/// full. No ID list, no allocation, no gather.
fn step_mask_prefix(state: &mut [u64], c1: &[u64], c2: &[u64]) {
    for w in state.iter_mut() {
        if *w != 0 {
            *w = u64::MAX; // bucket == one word: the union of its member mask
        }
    }
    // filter: state &= c1 & c2 — one VPTERNLOGQ per 512 bits
    ndarray::simd::mask_ternlog_assign::<{ ndarray::simd::ternlog::AND3 }>(state, c1, c2);
}

/// Mask arm, `Random`: no structure to exploit — OR one successor row per
/// active bit. This is the fallback the invariant calls the loss condition.
fn step_mask_random(state: &mut [u64], fwd: &[u64], out: &mut [u64], c1: &[u64], c2: &[u64]) {
    out.fill(0);
    for (wi, &w0) in state.iter().enumerate() {
        let mut word = w0;
        while word != 0 {
            let b = word.trailing_zeros() as usize;
            word &= word - 1;
            let row = (wi * 64 + b) * W;
            for k in 0..W {
                out[k] |= fwd[row + k];
            }
        }
    }
    state.copy_from_slice(out);
    ndarray::simd::mask_ternlog_assign::<{ ndarray::simd::ternlog::AND3 }>(state, c1, c2);
}

fn popcnt(m: &[u64]) -> u32 {
    m.iter().map(|w| w.count_ones()).sum()
}

fn main() {
    println!("hex/trie vs GEMM — N={N}, mask={} B, matrix={} MB\n", W * 8, N * N * 4 / 1_048_576);

    for rel in [Rel::Prefix, Rel::Random] {
        let name = if rel == Rel::Prefix {
            "PREFIX (structured)"
        } else {
            "RANDOM (no structure)"
        };
        println!("═══ relation: {name} ═══");

        count_on();
        let matrix = build_matrix_deg(rel, 0xC0FFEE, BUCKET);
        let gemm_setup = count_off();
        count_on();
        let succ = build_fwd_masks(&matrix);
        let mask_setup = count_off();

        println!("  setup bytes (amortized, §11.6):  GEMM {:>10}   MASK {:>10}", gemm_setup, mask_setup);
        println!(
            "  {:>6} {:>7} {:>11} {:>11} {:>9} {:>13} {:>13}",
            "dens%", "depth", "gemm ns/st", "mask ns/st", "speedup", "gemm B/step", "mask B/step"
        );

        for &dens in &[1usize, 5, 10, 25, 50, 75, 90, 99] {
            for &depth in &[1usize, 8, 32] {
                // initial state at the requested density
                let mut s = 0x5EEDu64 ^ (dens as u64) << 8;
                let mut m0 = vec![0u64; W];
                for i in 0..N {
                    if (splitmix(&mut s) % 100) < dens as u64 {
                        m0[i / 64] |= 1u64 << (i % 64);
                    }
                }
                // constraints: two learned permeability masks per step
                let mut c1 = vec![0u64; W];
                let mut c2 = vec![0u64; W];
                for w in 0..W {
                    c1[w] = splitmix(&mut s) | splitmix(&mut s);
                    c2[w] = splitmix(&mut s) | splitmix(&mut s);
                }
                let (c1f, c2f) = (dense_state(&c1), dense_state(&c2));
                let dense_m0 = dense_state(&m0);

                // ── GEMM arm ── (repeat to a 50 ms floor; a sub-resolution
                // timing divided into a ratio is noise, not a speedup)
                let mut gs = dense_state(&m0);
                let mut scratch = vec![0.0f32; N];
                let mut reps = 0usize;
                count_on();
                let t = Instant::now();
                while t.elapsed().as_secs_f64() < 0.05 {
                    gs.copy_from_slice(&dense_m0);
                    for _ in 0..depth {
                        step_gemm(&matrix, &mut gs, &mut scratch, &c1f);
                        for i in 0..N {
                            gs[i] *= c2f[i];
                        }
                    }
                    reps += 1;
                }
                let el = t.elapsed().as_secs_f64();
                let gemm_b = count_off() / reps.max(1);
                let gemm_ns = el * 1e9 / (reps * depth) as f64;
                std::hint::black_box(&gs);

                // ── MASK arm ──
                let mut ms_ = m0.clone();
                let mut out = vec![0u64; W];
                let mut mreps = 0usize;
                count_on();
                let t = Instant::now();
                while t.elapsed().as_secs_f64() < 0.05 {
                    ms_.copy_from_slice(&m0);
                    for _ in 0..depth {
                        match rel {
                            Rel::Prefix => step_mask_prefix(&mut ms_, &c1, &c2),
                            Rel::Random => step_mask_random(&mut ms_, &succ, &mut out, &c1, &c2),
                        }
                    }
                    mreps += 1;
                }
                let el = t.elapsed().as_secs_f64();
                let mask_b = count_off() / mreps.max(1);
                let mask_ns = el * 1e9 / (mreps * depth) as f64;
                std::hint::black_box(&ms_);

                // ── correctness gate: same survivors, or the numbers mean nothing ──
                let gemm_pop = gs.iter().filter(|&&v| v > 0.0).count() as u32;
                let mask_pop = popcnt(&ms_);
                assert_eq!(
                    gemm_pop, mask_pop,
                    "ARMS DISAGREE at dens={dens} depth={depth} ({name}): gemm {gemm_pop} vs mask {mask_pop}"
                );

                println!(
                    "  {:>6} {:>7} {:>11.3} {:>11.3} {:>8.1}x {:>13} {:>13}",
                    dens,
                    depth,
                    gemm_ns,
                    mask_ns,
                    gemm_ns / mask_ns.max(1e-9),
                    gemm_b,
                    mask_b
                );
            }
        }
        println!();
    }

    // ── D-GTM-0i, the axis the first table missed: RELATION density ──────────
    // State density fixed at 50%, depth 8, RANDOM relation (no structure to
    // exploit), sweeping edges-per-row from 1 to N. At deg = N the relation is
    // fully dense — the regime §11.1 pt 6 concedes to GEMM.
    println!("═══ D-GTM-0i: RELATION density sweep (RANDOM, state 50%, depth 8) ═══");
    println!("  {:>8} {:>9} {:>12} {:>12} {:>9}", "deg", "rel dens%", "gemm ns/st", "mask ns/st", "speedup");
    for &deg in &[1usize, 16, 64, 256, 1024, 4096] {
        let matrix = build_matrix_deg(Rel::Random, 0xC0FFEE, deg);
        let fwd = build_fwd_masks(&matrix);
        let mut s = 0xBEEFu64;
        let mut m0 = vec![0u64; W];
        for i in 0..N {
            if splitmix(&mut s) % 100 < 50 {
                m0[i / 64] |= 1u64 << (i % 64);
            }
        }
        let (mut c1, mut c2) = (vec![0u64; W], vec![0u64; W]);
        for w in 0..W {
            c1[w] = splitmix(&mut s) | splitmix(&mut s);
            c2[w] = splitmix(&mut s) | splitmix(&mut s);
        }
        let (c1f, c2f) = (dense_state(&c1), dense_state(&c2));
        let dense_m0 = dense_state(&m0);

        let mut gs = dense_m0.clone();
        let mut scratch = vec![0.0f32; N];
        let (mut r, t) = (0usize, Instant::now());
        while t.elapsed().as_secs_f64() < 0.05 {
            gs.copy_from_slice(&dense_m0);
            for _ in 0..8 {
                step_gemm(&matrix, &mut gs, &mut scratch, &c1f);
                for i in 0..N {
                    gs[i] *= c2f[i];
                }
            }
            r += 1;
        }
        let gemm_ns = t.elapsed().as_secs_f64() * 1e9 / (r * 8) as f64;

        let mut ms_ = m0.clone();
        let mut out = vec![0u64; W];
        let (mut mr, t) = (0usize, Instant::now());
        while t.elapsed().as_secs_f64() < 0.05 {
            ms_.copy_from_slice(&m0);
            for _ in 0..8 {
                step_mask_random(&mut ms_, &fwd, &mut out, &c1, &c2);
            }
            mr += 1;
        }
        let mask_ns = t.elapsed().as_secs_f64() * 1e9 / (mr * 8) as f64;

        let gp = gs.iter().filter(|&&v| v > 0.0).count() as u32;
        assert_eq!(gp, popcnt(&ms_), "ARMS DISAGREE at deg={deg}");
        println!(
            "  {:>8} {:>8.2}% {:>12.0} {:>12.0} {:>8.1}x",
            deg,
            100.0 * deg as f64 / N as f64,
            gemm_ns,
            mask_ns,
            gemm_ns / mask_ns.max(1e-9)
        );
    }
}
