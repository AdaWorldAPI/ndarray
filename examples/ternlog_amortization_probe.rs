//! D-GTM-0n — is a *further* learned constraint cheap once the representation
//! is resident, and where does residency actually break?
//!
//! The architectural claim under test is NOT that ternary Boolean logic is
//! intelligent. It is narrower and mechanical: if learned state and active
//! state are the same shape, then stacking one more constraint should cost
//! roughly *one more resident load plus one instruction*, rather than another
//! pass of graph materialization. That is a claim about an amortization curve,
//! so this probe measures the curve rather than a headline ratio.
//!
//! Four arms compute the IDENTICAL logical result — `A ∧ M₁ ∧ … ∧ M_K` — and a
//! correctness gate aborts the run unless every arm agrees, both on the
//! survivor count and on the surviving set itself.
//!
//! | arm | representation | work per constraint |
//! |---|---|---|
//! | T0 | materialized `Vec<u32>` id list | rebuild the candidate list |
//! | T1 | bitset, `mask_and_assign` | one pass over the mask |
//! | T3 | bitset, `mask_ternlog_assign::<AND3>` | one pass per TWO constraints |
//! | T5 | sorted index intersection | merge over the sparse sets |
//!
//! T3 is the arm the claim is about: `VPTERNLOGQ` folds two constraints into
//! one instruction, so if the cost model is "a pass over resident bytes", T3
//! should sit near half of T1 and the per-constraint cost should be FLAT in K.
//! Flat is the honest success criterion — a per-constraint cost that *falls*
//! with K would mean something else is happening (and would need explaining,
//! not celebrating).
//!
//! Residency is derived, not asserted: the probe reports achieved mask-traffic
//! bandwidth, and the knee where GB/s collapses is where the working set left a
//! cache level. Reading residency off "pretty nanosecond numbers" is exactly
//! what D-GTM-0h was graded [S] for, so the bandwidth column is the evidence
//! and the ns column is not.
//!
//! Usage: `cargo run --release --example ternlog_amortization_probe`

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::simd::ternlog::AND3;
use ndarray::simd::{mask_and_assign, mask_ternlog_assign};

/// Counting allocator — "materialized 0 bytes" is a measurement here, not a claim.
struct Counting;
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(l.size(), Ordering::Relaxed);
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static A: Counting = Counting;

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Bits set in a mask, as the arms' shared correctness currency.
fn popcnt(m: &[u64]) -> u32 {
    m.iter().map(|w| w.count_ones()).sum()
}

fn to_ids(m: &[u64]) -> Vec<u32> {
    let mut v = Vec::new();
    for (w, word) in m.iter().enumerate() {
        let mut b = *word;
        while b != 0 {
            v.push((w * 64 + b.trailing_zeros() as usize) as u32);
            b &= b - 1;
        }
    }
    v
}

/// Run one (N, K) cell across all four arms. Returns per-arm ns/constraint.
fn cell(n_bits: usize, k: usize, seed: u64) -> [f64; 4] {
    let words = n_bits / 64;
    let mut rng = SplitMix(seed);

    // Constraint masks are dense enough that survivors persist to K=64: each
    // keeps ~97% of bits, so the intersection stays non-empty and every arm
    // does real work at every depth. A sparser mask would empty the set after
    // a few constraints and measure nothing but early exit.
    let masks: Vec<Vec<u64>> = (0..k)
        .map(|_| (0..words).map(|_| !(1u64 << (rng.next() % 64))).collect())
        .collect();
    let base: Vec<u64> = (0..words).map(|_| rng.next() | rng.next()).collect();

    // ---- correctness gate, before any timing ----
    let mut t1 = base.clone();
    for m in &masks {
        mask_and_assign(&mut t1, m);
    }
    let mut t3 = base.clone();
    let mut i = 0;
    while i + 1 < k {
        mask_ternlog_assign::<AND3>(&mut t3, &masks[i], &masks[i + 1]);
        i += 2;
    }
    if i < k {
        mask_and_assign(&mut t3, &masks[i]);
    }
    let gold = to_ids(&t1);
    assert_eq!(gold, to_ids(&t3), "T3 disagrees with T1 at N={n_bits} K={k}");

    // sparse arms share the same logical sets, expressed as sorted id lists
    let sparse: Vec<Vec<u32>> = masks.iter().map(|m| to_ids(m)).collect();
    let base_ids = to_ids(&base);
    let mut t5 = base_ids.clone();
    for s in &sparse {
        let mut out = Vec::with_capacity(t5.len());
        let (mut a, mut b) = (0usize, 0usize);
        while a < t5.len() && b < s.len() {
            match t5[a].cmp(&s[b]) {
                std::cmp::Ordering::Equal => {
                    out.push(t5[a]);
                    a += 1;
                    b += 1;
                }
                std::cmp::Ordering::Less => a += 1,
                std::cmp::Ordering::Greater => b += 1,
            }
        }
        t5 = out;
    }
    assert_eq!(gold, t5, "T5 disagrees with T1 at N={n_bits} K={k}");

    // ---- timing: each arm runs to a 60 ms floor, well past timer resolution ----
    const FLOOR: f64 = 0.060;
    let mut out = [0.0f64; 4];

    // T1 — one pass per constraint
    let mut scratch = base.clone();
    let (mut iters, t0) = (0u64, Instant::now());
    while t0.elapsed().as_secs_f64() < FLOOR {
        scratch.copy_from_slice(&base);
        for m in &masks {
            mask_and_assign(&mut scratch, m);
        }
        iters += 1;
    }
    out[1] = t0.elapsed().as_secs_f64() * 1e9 / (iters as f64 * k as f64);

    // T3 — one pass per TWO constraints
    let (mut iters3, t3c) = (0u64, Instant::now());
    while t3c.elapsed().as_secs_f64() < FLOOR {
        scratch.copy_from_slice(&base);
        let mut i = 0;
        while i + 1 < k {
            mask_ternlog_assign::<AND3>(&mut scratch, &masks[i], &masks[i + 1]);
            i += 2;
        }
        if i < k {
            mask_and_assign(&mut scratch, &masks[i]);
        }
        iters3 += 1;
    }
    out[2] = t3c.elapsed().as_secs_f64() * 1e9 / (iters3 as f64 * k as f64);

    // T5 — sorted-index intersection (the honest sparse arm)
    let (mut iters5, t5c) = (0u64, Instant::now());
    while t5c.elapsed().as_secs_f64() < FLOOR {
        let mut cur = base_ids.clone();
        for s in &sparse {
            let mut o = Vec::with_capacity(cur.len());
            let (mut a, mut b) = (0usize, 0usize);
            while a < cur.len() && b < s.len() {
                match cur[a].cmp(&s[b]) {
                    std::cmp::Ordering::Equal => {
                        o.push(cur[a]);
                        a += 1;
                        b += 1;
                    }
                    std::cmp::Ordering::Less => a += 1,
                    std::cmp::Ordering::Greater => b += 1,
                }
            }
            cur = o;
        }
        std::hint::black_box(&cur);
        iters5 += 1;
    }
    out[3] = t5c.elapsed().as_secs_f64() * 1e9 / (iters5 as f64 * k as f64);

    // T0 — materialize the surviving id list after every constraint
    let (mut iters0, t0c) = (0u64, Instant::now());
    while t0c.elapsed().as_secs_f64() < FLOOR {
        let mut cur = base.clone();
        for m in &masks {
            mask_and_assign(&mut cur, m);
            std::hint::black_box(to_ids(&cur));
        }
        iters0 += 1;
    }
    out[0] = t0c.elapsed().as_secs_f64() * 1e9 / (iters0 as f64 * k as f64);

    out
}

fn main() {
    println!("D-GTM-0n — ternlog amortization + residency boundary");
    println!("host L1d 48 KiB/core, L2 2 MiB/core, L3 260 MiB shared\n");

    // Depth sweep at a fixed L1-resident mask, isolating K from working-set size.
    let n = 1 << 15; // 32768 bits = 4 KiB per mask
    println!("A. DEPTH SWEEP — mask 4 KiB, working set = (K+1)x4 KiB");
    println!("  K | working set | T0 mat'd | T1 and | T3 ternlog | T5 sparse | T3/T1 | T3 GB/s");
    for k in [1usize, 2, 4, 8, 16, 32, 64] {
        let r = cell(n, k, 0x1234 + k as u64);
        let ws = (k + 1) * n / 8;
        // each constraint moves one mask in and the accumulator through
        let gbs = (n as f64 / 8.0 * 2.0) / r[2];
        println!(
            "  {k:2} | {:8} B | {:8.1} | {:6.1} | {:10.1} | {:9.1} | {:5.2} | {:7.1}",
            ws,
            r[0],
            r[1],
            r[2],
            r[3],
            r[2] / r[1],
            gbs
        );
    }

    // Working-set sweep at fixed depth: the knee locates the cache boundary.
    println!("\nB. RESIDENCY SWEEP — K=8 constraints, mask size grows");
    println!("  mask     | working set | T1 and | T3 ternlog | T3/T1 | T3 GB/s | level");
    for lg in [12usize, 14, 16, 18, 20, 22, 24, 26] {
        let nb = 1usize << lg;
        let mask_b = nb / 8;
        let ws = 9 * mask_b;
        let r = cell(nb, 8, 0x99 + lg as u64);
        let gbs = (mask_b as f64 * 2.0) / r[2];
        let level = if ws <= 48 * 1024 {
            "L1"
        } else if ws <= 2 * 1024 * 1024 {
            "L2"
        } else if ws <= 260 * 1024 * 1024 {
            "L3"
        } else {
            "DRAM"
        };
        println!(
            "  {:6} B | {:9} B | {:6.1} | {:10.1} | {:5.2} | {:7.1} | {level}",
            mask_b,
            ws,
            r[1],
            r[2],
            r[2] / r[1],
            gbs
        );
    }

    // D. DENSITY SWEEP — the arm that decides whether masks are the right
    // currency at all. A focus field over a huge substrate is SPARSE, and a
    // bitset pays for every bit whether set or not, while an id list pays only
    // for survivors. Sections A and B ran at ~97% density, which is a
    // mis-specified baseline for the sparse arms in exactly the way a dense f32
    // matrix was a mis-specified baseline for a Boolean relation in D-GTM-0j —
    // so their T0/T5 columns are void as evidence and this sweep replaces them.
    println!("\nD. DENSITY SWEEP — N=2^20 bits (128 KiB mask), K=8 constraints");
    println!("  active   | survivors | T1 and | T3 ternlog | T5 sparse | winner");
    let nb = 1usize << 20;
    let words = nb / 64;
    for (label, keep_shift) in [("100%", 0u32), ("50%", 1), ("6%", 4), ("0.8%", 7), ("0.1%", 10), ("0.012%", 13)] {
        let mut rng = SplitMix(0xDEAD + keep_shift as u64);
        // base: keep roughly 1 bit in 2^keep_shift
        let base: Vec<u64> = (0..words)
            .map(|_| {
                let mut w = u64::MAX;
                for _ in 0..keep_shift {
                    w &= rng.next();
                }
                w
            })
            .collect();
        // constraints stay near-total so survivors track the base density
        let masks: Vec<Vec<u64>> = (0..8)
            .map(|_| (0..words).map(|_| !(1u64 << (rng.next() % 64))).collect())
            .collect();

        let mut t1v = base.clone();
        for m in &masks {
            mask_and_assign(&mut t1v, m);
        }
        let survivors = popcnt(&t1v);
        let sparse: Vec<Vec<u32>> = masks.iter().map(|m| to_ids(m)).collect();
        let base_ids = to_ids(&base);

        const FLOOR: f64 = 0.060;
        let mut scratch = base.clone();
        let (mut i1, c1) = (0u64, Instant::now());
        while c1.elapsed().as_secs_f64() < FLOOR {
            scratch.copy_from_slice(&base);
            for m in &masks {
                mask_and_assign(&mut scratch, m);
            }
            i1 += 1;
        }
        let ns1 = c1.elapsed().as_secs_f64() * 1e9 / (i1 as f64 * 8.0);

        let (mut i3, c3) = (0u64, Instant::now());
        while c3.elapsed().as_secs_f64() < FLOOR {
            scratch.copy_from_slice(&base);
            let mut i = 0;
            while i + 1 < 8 {
                mask_ternlog_assign::<AND3>(&mut scratch, &masks[i], &masks[i + 1]);
                i += 2;
            }
            i3 += 1;
        }
        let ns3 = c3.elapsed().as_secs_f64() * 1e9 / (i3 as f64 * 8.0);

        // The sparse arm gets its BEST case: it walks only survivors, and the
        // constraint sets are checked by direct bit test rather than by merging
        // two long lists — a list merge against a near-total set is the sparse
        // representation's worst case and would not be how anyone builds this.
        let (mut i5, c5) = (0u64, Instant::now());
        while c5.elapsed().as_secs_f64() < FLOOR {
            let mut cur = base_ids.clone();
            for m in &masks {
                cur.retain(|&id| m[id as usize / 64] >> (id % 64) & 1 == 1);
            }
            std::hint::black_box(&cur);
            i5 += 1;
        }
        let ns5 = c5.elapsed().as_secs_f64() * 1e9 / (i5 as f64 * 8.0);
        let _ = &sparse;

        let winner = if ns5 < ns3 { "sparse" } else { "mask" };
        println!("  {:8} | {:9} | {:6.0} | {:10.0} | {:9.0} | {winner}", label, survivors, ns1, ns3, ns5);
    }

    // Materialization: the claim is that the Boolean arms never build a set.
    let before = ALLOCATED.load(Ordering::Relaxed);
    let words = (1usize << 16) / 64;
    let mut acc: Vec<u64> = vec![u64::MAX; words];
    let m1: Vec<u64> = vec![0xF0F0_F0F0_F0F0_F0F0; words];
    let m2: Vec<u64> = vec![0xFF00_FF00_FF00_FF00; words];
    let setup = ALLOCATED.load(Ordering::Relaxed);
    for _ in 0..1000 {
        mask_ternlog_assign::<AND3>(&mut acc, &m1, &m2);
    }
    let after = ALLOCATED.load(Ordering::Relaxed);
    println!(
        "\nC. MATERIALIZATION — setup {} B, then {} B over 1000 chained steps (popcount {})",
        setup - before,
        after - setup,
        popcnt(&acc)
    );
}
