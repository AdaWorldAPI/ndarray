//! D-POF-ANT-0: Position × Order × Law, the first antennas.
//!
//! Each antenna is a module with its own exact oracle and falsifiers
//! (`tests/pof_antennas.rs` runs them). This binary times the ones whose cost
//! is a question: A1 (position), A2 (exposure), A4's two assembly pairs and
//! A6 (scale). A3 and A5 are semantic antennas and have no timing.
//!
//! `cargo run --release --example pof_antennas --features std`;
//! the tier is whatever this build compiled (printed in the header).

#[path = "../u9_abacus/anchor.rs"]
mod anchor;
mod exposure;
#[path = "../u9_head_to_head/kernels.rs"]
mod kernels;
mod meta;
mod order_law;
mod position;
mod scale;

use anchor::{split_anchor, AccE5, QueryE};
use exposure::{AccDual, AccE2, AccPosMag};
use kernels::{check, oracle, prepare_b, AccB, Block, Operand, QueryB};
use ndarray::hpc::perm::Perm64;
use ndarray::simd::{I8x64, U64x8, U8x64};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Counts allocations and allocated bytes, so A6 reports them measured.
struct Counting;
static ALLOCS: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocs() -> (u64, u64) {
    (ALLOCS.load(Ordering::Relaxed), ALLOC_BYTES.load(Ordering::Relaxed))
}

fn tier() -> &'static str {
    if cfg!(target_feature = "avx512vnni") {
        "x86_64 AVX-512 + VNNI"
    } else if cfg!(target_feature = "avx512bw") {
        "x86_64 AVX-512 (no VNNI)"
    } else if cfg!(target_feature = "avx2") {
        "x86_64 AVX2"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64 NEON"
    } else {
        "other"
    }
}

/// Median of `runs` timings, in ns per `unit`.
fn time(name: &str, units: usize, unit: &str, runs: usize, base: Option<f64>, f: &dyn Fn() -> i64) -> f64 {
    black_box(f());
    let mut t: Vec<f64> = (0..runs)
        .map(|_| {
            let t0 = Instant::now();
            black_box(f());
            t0.elapsed().as_nanos() as f64 / units as f64
        })
        .collect();
    t.sort_by(f64::total_cmp);
    let med = t[runs / 2];
    let rel = base
        .map(|b| format!("  {:6.2}x", med / b))
        .unwrap_or_default();
    println!("{name:<62} {med:9.3} ns/{unit}{rel}");
    med
}

type Raw = ([u8; 64], u64);

/// Uniform over the 511 numeric U9 values (as in #388).
fn data(blocks: usize, seed: u64) -> (Vec<Raw>, Vec<Raw>) {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as u32
    };
    let mut mk = || {
        let mut m = [0u8; 64];
        let mut p = 0u64;
        for (i, mi) in m.iter_mut().enumerate() {
            let v = (next() % 511) as i32 - 255;
            *mi = v.unsigned_abs() as u8;
            p |= u64::from(v >= 0) << i;
        }
        (m, p)
    };
    let a = (0..blocks).map(|_| mk()).collect();
    let b = (0..blocks).map(|_| mk()).collect();
    (a, b)
}

fn perm(seed: u64) -> Perm64 {
    let mut idx: [u8; 64] = core::array::from_fn(|i| i as u8);
    let mut s = seed | 1;
    for i in (1..64).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        idx.swap(i, (s % (i as u64 + 1)) as usize);
    }
    Perm64::from_indices(idx).unwrap()
}

// ---- A1 ------------------------------------------------------------------

fn bench_a1(label: &str, n: usize, runs: usize) {
    use position::*;
    let raw: Vec<RawPlanes> = (0..n).map(|k| fixture(1, 0x9E37_79B9 ^ k as u64)).collect();
    let prepared: Vec<Prepared> = raw.iter().map(Prepared::new).collect();
    let p = perm(77);
    let rebased: Vec<Rebased> = prepared.iter().map(|x| x.rebase(p)).collect();
    for (r, x) in raw.iter().zip(&rebased) {
        assert_eq!(x.run(1).unwrap(), oracle(&permute_raw(r, p)));
    }
    let fold = |r: Results| r.count as i64 ^ r.signed_sum ^ r.select as i64;
    let p0 = || {
        raw.iter()
            .map(|r| fold(p0_separate(r)))
            .fold(0, i64::wrapping_add)
    };
    let p1 = || {
        raw.iter()
            .map(|r| fold(Prepared::new(r).run(1).unwrap()))
            .fold(0, i64::wrapping_add)
    };
    let setup = || {
        raw.iter()
            .map(|r| {
                black_box(Prepared::new(r));
                0
            })
            .sum()
    };
    let p1_exec = || {
        prepared
            .iter()
            .map(|x| fold(x.run(1).unwrap()))
            .fold(0, i64::wrapping_add)
    };
    let rebase = || {
        prepared
            .iter()
            .map(|x| {
                black_box(x.rebase(p));
                0
            })
            .sum()
    };
    let p2_exec = || {
        rebased
            .iter()
            .map(|x| fold(x.run(1).unwrap()))
            .fold(0, i64::wrapping_add)
    };
    let p2 = || {
        raw.iter()
            .map(|r| fold(Prepared::new(r).rebase(p).run(1).unwrap()))
            .fold(0, i64::wrapping_add)
    };
    let lazy = || {
        prepared
            .iter()
            .map(|x| fold(x.run_rebased_lazy(p, 1).unwrap()))
            .fold(0, i64::wrapping_add)
    };
    let blended = || {
        prepared
            .iter()
            .map(|x| {
                black_box(conjugate_mask_blended(&p, x.mask_pred()));
                black_box(conjugate_mask_blended(&p, x.mask_positive()));
                black_box(conjugate_mask_blended(&p, x.mask_sel()));
                0
            })
            .sum()
    };
    let materialize_all = || {
        raw.iter()
            .map(|r| {
                black_box(permute_raw(r, p));
                0
            })
            .sum()
    };
    println!("-- A1 position, {label}: {n} blocks of 64 positions (4 byte planes = 256 B read per block)");
    let b0 = time("P0 separate preparation per operation", n, "block", runs, None, &p0);
    time("P1 shared basis: prepare + run", n, "block", runs, Some(b0), &p1);
    time("  P1 setup only (4 planes -> 3 masks + magnitudes)", n, "block", runs, Some(b0), &setup);
    time("  P1 execution only (prepared basis given)", n, "block", runs, Some(b0), &p1_exec);
    time("P2 re-based: prepare + rebase + run", n, "block", runs, Some(b0), &p2);
    time("  P2 rebase only (3 conjugate_mask)", n, "block", runs, Some(b0), &rebase);
    time("  P2 execution only (re-based basis given)", n, "block", runs, Some(b0), &p2_exec);
    time(
        "P2-lazy: prepared, run in source, conjugate the select result once",
        n,
        "block",
        runs,
        Some(b0),
        &lazy,
    );
    time(
        "  rebase only, blended mask expansion (3 x mask_blend + materialize)",
        n,
        "block",
        runs,
        Some(b0),
        &blended,
    );
    time(
        "  reference: materialize all 4 planes through the map",
        n,
        "block",
        runs,
        Some(b0),
        &materialize_all,
    );
}

// ---- A2 ------------------------------------------------------------------

fn bench_a2(label: &str, n: usize, runs: usize, seed: u64) {
    let (ra, rb) = data(n, seed);
    let (ab, bb) = (Operand::new(&ra), Operand::new(&rb));
    let want = oracle(&ra, &rb).unwrap();
    let q = bb.block(0);
    let want_scan = oracle(&ra, &vec![rb[0]; n]).unwrap();
    let split: Vec<(u64, I8x64, I8x64)> = ab
        .blocks()
        .zip(bb.blocks())
        .map(|(a, b)| prepare_b(a, b))
        .collect();
    let anc: Vec<(u64, I8x64, u64)> = ab
        .blocks()
        .zip(bb.blocks())
        .map(|(a, b)| {
            let (l, t) = split_anchor(b.mag);
            (a.positive ^ b.positive, l, t)
        })
        .collect();

    let b_e2e = || {
        let mut acc = AccB::new();
        for (a, b) in ab.blocks().zip(bb.blocks()) {
            check(a, b).unwrap();
            acc.step(a, b);
        }
        acc.finish()
    };
    let e5_e2e = || {
        let mut acc = AccE5::new();
        for (a, b) in ab.blocks().zip(bb.blocks()) {
            check(a, b).unwrap();
            acc.step(a, b);
        }
        acc.finish()
    };
    let e2_e2e = || {
        let mut acc = AccE2::new();
        for (a, b) in ab.blocks().zip(bb.blocks()) {
            check(a, b).unwrap();
            let (l, t) = split_anchor(b.mag);
            acc.mac(a.mag, a.positive ^ b.positive, l, t);
        }
        acc.finish()
    };
    let b_ari = || asm_b_loop(&ab.mag, &split);
    let e5_ari = || asm_e5_loop(&ab.mag, &anc);
    let e2_ari = || asm_e2_loop(&ab.mag, &anc);
    let b_scan = || {
        let qb = QueryB::new(q);
        let mut acc = AccB::new();
        for x in ab.blocks() {
            check(x, q).unwrap();
            acc.scan_step(x, &qb);
        }
        acc.finish()
    };
    let e5_scan = || {
        let qe = QueryE::new(q);
        let mut acc = AccE5::new();
        for x in ab.blocks() {
            check(x, q).unwrap();
            acc.scan_step(x, &qe);
        }
        acc.finish()
    };
    let e2_scan = || {
        let qe = QueryE::new(q);
        let mut acc = AccE2::new();
        for x in ab.blocks() {
            check(x, q).unwrap();
            acc.scan_step(x, &qe);
        }
        acc.finish()
    };
    let dual_fused = || {
        let (t, m) = asm_dual_loop(&ab.mag, &split);
        t ^ m as i64
    };
    let dual_separate = || {
        let t = asm_b_loop(&ab.mag, &split);
        let m = asm_posmag_loop(&ab.mag, &split);
        t ^ m as i64
    };
    for (name, f, w) in [
        ("B e2e", &b_e2e as &dyn Fn() -> i64, want),
        ("E5 e2e", &e5_e2e, want),
        ("E2 e2e", &e2_e2e, want),
        ("B ari", &b_ari, want),
        ("E5 ari", &e5_ari, want),
        ("E2 ari", &e2_ari, want),
        ("B scan", &b_scan, want_scan),
        ("E5 scan", &e5_scan, want_scan),
        ("E2 scan", &e2_scan, want_scan),
    ] {
        assert_eq!(f(), w, "{name} disagrees with the oracle");
    }
    assert_eq!(dual_fused(), dual_separate());

    println!("-- A2 exposure, {label}: {n} blocks of 64 values per operand");
    let be = time("E0 = B end-to-end", n, "64", runs, None, &b_e2e);
    time("E1 = E5 end-to-end (unchanged, #393)", n, "64", runs, Some(be), &e5_e2e);
    time("E2 end-to-end: sign-routed operand once, anchor narrows it", n, "64", runs, Some(be), &e2_e2e);
    let ba = time("E0 = B arithmetic only", n, "64", runs, None, &b_ari);
    time("E1 = E5 arithmetic only", n, "64", runs, Some(ba), &e5_ari);
    time("E2 arithmetic only", n, "64", runs, Some(ba), &e2_ari);
    let bs = time("E0 = B scan", n, "64", runs, None, &b_scan);
    time("E1 = E5 scan", n, "64", runs, Some(bs), &e5_scan);
    time("E2 scan", n, "64", runs, Some(bs), &e2_scan);
    let sep = time("E3 two consumers, two passes (B + positive-|a| sum)", n, "64", runs, None, &dual_separate);
    time("E3 two consumers fused in one pass", n, "64", runs, Some(sep), &dual_fused);
}

#[inline(never)]
pub fn asm_b_loop(mags: &[U8x64], split: &[(u64, I8x64, I8x64)]) -> i64 {
    let mut acc = AccB::new();
    for (a, (ng, l, t)) in mags.iter().zip(split) {
        acc.mac(*a, *ng, *l, *t);
    }
    acc.finish()
}
#[inline(never)]
pub fn asm_e5_loop(mags: &[U8x64], anc: &[(u64, I8x64, u64)]) -> i64 {
    let mut acc = AccE5::new();
    for (a, (ng, l, t)) in mags.iter().zip(anc) {
        acc.mac(*a, *ng, *l, *t);
    }
    acc.finish()
}
#[inline(never)]
pub fn asm_e2_loop(mags: &[U8x64], anc: &[(u64, I8x64, u64)]) -> i64 {
    let mut acc = AccE2::new();
    for (a, (ng, l, t)) in mags.iter().zip(anc) {
        acc.mac(*a, *ng, *l, *t);
    }
    acc.finish()
}
#[inline(never)]
pub fn asm_dual_loop(mags: &[U8x64], split: &[(u64, I8x64, I8x64)]) -> (i64, u64) {
    let mut acc = AccDual::new();
    for (a, (ng, l, t)) in mags.iter().zip(split) {
        acc.mac(*a, *ng, *l, *t);
    }
    acc.finish()
}
#[inline(never)]
pub fn asm_posmag_loop(mags: &[U8x64], split: &[(u64, I8x64, I8x64)]) -> u64 {
    let mut acc = AccPosMag::new();
    for (a, (ng, _, _)) in mags.iter().zip(split) {
        acc.mac(*a, *ng);
    }
    acc.finish()
}

// ---- A4: the obvious optimized expression vs the direct baseline ---------

/// Count after moving the data through the map (direct baseline).
#[inline(never)]
pub fn asm_count_after_perm(p: &Perm64, a: &[u8; 64]) -> u32 {
    p.materialize(a).iter().filter(|&&v| v != 0).count() as u32
}
/// Count with the map skipped: admitted only because count is invariant.
#[inline(never)]
pub fn asm_count_skipping_perm(_p: &Perm64, a: &[u8; 64]) -> u32 {
    U8x64::from_array(*a)
        .cmpeq_mask(U8x64::splat(0))
        .count_zeros()
}
/// (a ^ b) & c as two lane operations.
#[inline(never)]
pub fn asm_xor_then_and(a: U64x8, b: U64x8, c: U64x8) -> U64x8 {
    (a ^ b) & c
}
/// (a ^ b) & c as one ternary-logic operation (immediate 0x28).
#[inline(never)]
pub fn asm_fused_ternlog(a: U64x8, b: U64x8, c: U64x8) -> U64x8 {
    a.ternlog::<0x28>(b, c)
}

// ---- A6 ------------------------------------------------------------------

fn bench_a6(runs: usize) {
    use scale::*;
    let t = table(0x51_7CC1_B727_220A);
    let (a, b) = (0xF0F0_1234_8000_0001u64, 0x0FF0_F00F_5555_AAAAu64);
    let (ab, bb) = (bytes_of(a), bytes_of(b));
    let (rm, mo) = (row_major(), morton());
    let want = oracle(&t, a, b);
    type Row<'a> = (&'a str, Box<dyn Fn() -> (u64, Traffic) + 'a>);
    let rows: [Row; 5] = [
        ("per cell, materialized, row-major", Box::new(|| per_cell_materialized(&t, &ab, &bb, &rm))),
        ("per cell, materialized, Morton", Box::new(|| per_cell_materialized(&t, &ab, &bb, &mo))),
        ("per cell, conjugated mask, row-major", Box::new(|| per_cell_conjugated(&t, a, b, &rm))),
        ("per cell, conjugated mask, Morton", Box::new(|| per_cell_conjugated(&t, a, b, &mo))),
        ("whole field through the exposure (64x64 u16 kernel)", Box::new(|| whole_field_exposure(&t, a, b))),
    ];
    println!("-- A6 scale: 64x64 field (4096 cells), one field-wide count; table working set 8 KiB");
    let mut base = None;
    for (name, f) in &rows {
        let (got, tr) = f();
        assert_eq!(got, want, "{name}");
        let (n0, b0) = allocs();
        black_box(f());
        let (n1, b1) = allocs();
        let med = time(name, 1, "field", runs, base, &|| f().0 as i64);
        base.get_or_insert(med);
        println!(
            "    compositions {}, bytes written {}, allocations {} ({} B)",
            tr.compositions,
            tr.bytes_written,
            n1 - n0,
            b1 - b0
        );
    }
}

fn main() {
    println!("tier: {} (target-cpu as compiled)", tier());
    {
        // Keep the out-of-line copies in the binary for the disassembly.
        let (a, b, c) = (black_box(U64x8::splat(3)), black_box(U64x8::splat(5)), black_box(U64x8::splat(9)));
        assert_eq!(asm_xor_then_and(a, b, c).to_array(), asm_fused_ternlog(a, b, c).to_array());
        let p = perm(5);
        let blk: [u8; 64] = core::array::from_fn(|i| (i % 3) as u8);
        assert_eq!(asm_count_after_perm(&p, &blk), asm_count_skipping_perm(&p, &blk));
        let _: Block;
    }
    bench_a1("L1-resident", 64, 2001);
    bench_a1("streamed", 1 << 18, 9);
    bench_a2("L1-resident", 64, 2001, 1);
    bench_a2("streamed", 1 << 20, 5, 2);
    bench_a6(201);
}
