//! D-U64X64-LLVM-1 lab benchmark harness (plan §5 and §8): counting allocator,
//! tier report, timing, cache sweeps, allocation guard, analytic bytes model and
//! the pre-registered decision rule.
//!
//! `main.rs` must install the allocator itself:
//! `#[global_allocator] static A: harness::Counting = harness::Counting;`
//! This module deliberately declares no `#[global_allocator]`.
//!
//! Lab fixture only: no public type, no production caller.
#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

/// Allocator that counts allocation events (alloc, alloc_zeroed, realloc) and
/// forwards to `System`. Install it from `main.rs`.
pub struct Counting;

// SAFETY: every method forwards unchanged to `System`, which upholds the
// `GlobalAlloc` contract; the only addition is a relaxed atomic increment.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: caller's layout contract is passed through to System.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr/layout come from a matching alloc through this allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: caller's layout contract is passed through to System.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: ptr/layout/new_size are passed through unchanged to System.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Allocation events counted so far (zero growth if `Counting` is not global).
pub fn allocs() -> usize {
    ALLOCS.load(Ordering::Relaxed)
}

/// Compile-time target tier, from the features this binary was built with.
pub fn tier_line() -> String {
    format!(
        "tier: avx512f={} avx512bw={} avx512vpopcntdq={} avx2={} sse4.2={} bmi2={} arch={}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx512bw"),
        cfg!(target_feature = "avx512vpopcntdq"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "sse4.2"),
        cfg!(target_feature = "bmi2"),
        std::env::consts::ARCH
    )
}

/// One timing sample.
pub struct Sample {
    pub ns_per_call: f64,
}

/// Summary of a set of samples (ns per call).
pub struct Stats {
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub samples: Vec<f64>,
}

/// Median of a slice (mean of the two middle values when even); 0.0 if empty.
pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

impl Stats {
    pub fn from_samples(samples: Vec<f64>) -> Stats {
        let min = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Stats { median: median(&samples), min, max, samples }
    }
}

/// Pass-through `black_box` for callers.
pub fn bb<T>(x: T) -> T {
    std::hint::black_box(x)
}

/// One sample = `before_each_sample()` untimed, then `reps` timed calls of `f`.
pub fn measure<F: FnMut()>(
    reps: usize,
    samples: usize,
    mut before_each_sample: impl FnMut(),
    mut f: F,
) -> Stats {
    let reps = reps.max(1);
    let mut out = Vec::with_capacity(samples);
    for _ in 0..samples {
        before_each_sample();
        let t0 = Instant::now();
        for _ in 0..reps {
            f();
        }
        let ns = t0.elapsed().as_nanos() as f64;
        out.push(ns / reps as f64);
    }
    Stats::from_samples(out)
}

/// Cost of the timing loop itself: an empty closure, 1000 reps per sample.
pub fn timer_overhead(samples: usize) -> Stats {
    measure(1000, samples, || {}, || {})
}

/// A kernel under 3x the timer overhead is marked LOW-RES.
pub fn low_res(kernel: f64, overhead: f64) -> bool {
    kernel < 3.0 * overhead
}

/// S2 sweep size (cold L2).
pub const L2_SWEEP: usize = 16 << 20;
/// S3 sweep size (cold LLC; label "≥L2-cold" if the VM's LLC is not real).
pub const LLC_SWEEP: usize = 640 << 20;

/// Buffer read at one byte per cache line to evict working sets.
pub struct Sweep {
    buf: Vec<u8>,
}

impl Sweep {
    pub fn new(bytes: usize) -> Sweep {
        let buf = (0..bytes).map(|i| (i as u8) | 1).collect();
        Sweep { buf }
    }

    /// Read every 64th byte; return the wrapping sum so callers can black_box it.
    pub fn run(&mut self) -> u64 {
        let mut acc = 0u64;
        for &b in self.buf.iter().step_by(64) {
            acc = acc.wrapping_add(b as u64);
        }
        acc
    }
}

/// Panic if `f` performed any counted allocation. Only meaningful when
/// `Counting` is the global allocator.
pub fn assert_no_alloc<R>(label: &str, f: impl FnOnce() -> R) -> R {
    let before = allocs();
    let r = f();
    let after = allocs();
    if before != after {
        panic!("{label}: {} allocation(s) inside a timed region", after - before);
    }
    r
}

/// Analytic bytes model per entry.
pub struct Bytes {
    pub source: u64,
    pub scratch: u64,
}

impl Bytes {
    pub fn total(&self) -> u64 {
        self.source + self.scratch
    }
}

/// Bytes of stock planes read for `k` units (plan formula: k * 128).
pub fn stock_planes(k: u64) -> u64 {
    k * 128
}
/// Bytes of cost planes read for `k` units (plan formula: k * 64).
pub fn cost_planes(k: u64) -> u64 {
    k * 64
}
/// Bytes of bit planes read for `k` units (plan formula: k * 8).
pub fn bit_planes(k: u64) -> u64 {
    k * 8
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Scalar,
    Bitslice,
    Simd,
    Prepared,
}

pub struct Cell {
    pub entry: &'static str,
    pub class: Class,
    pub process_medians: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Wins { entry: &'static str, class: Class },
    NoMeasuredAdvantage,
}

fn pmax(c: &Cell) -> f64 {
    c.process_medians.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
fn pmin(c: &Cell) -> f64 {
    c.process_medians.iter().copied().fold(f64::INFINITY, f64::min)
}

/// Plan §8: W = lowest median of per-process medians; R = best cell of a
/// different class. W wins iff max(W) < min(R) and median(W) <= 0.85*median(R).
pub fn decide(cells: &[Cell]) -> Verdict {
    let live: Vec<&Cell> = cells.iter().filter(|c| !c.process_medians.is_empty()).collect();
    let by_med = |a: &&Cell, b: &&Cell| {
        median(&a.process_medians).total_cmp(&median(&b.process_medians))
    };
    let Some(w) = live.iter().copied().min_by(by_med) else {
        return Verdict::NoMeasuredAdvantage;
    };
    let Some(r) = live.iter().copied().filter(|c| c.class != w.class).min_by(by_med) else {
        return Verdict::NoMeasuredAdvantage;
    };
    if pmax(w) < pmin(r) && median(&w.process_medians) <= 0.85 * median(&r.process_medians) {
        Verdict::Wins { entry: w.entry, class: w.class }
    } else {
        Verdict::NoMeasuredAdvantage
    }
}

/// N* = prep / (t_unprep - t_query); `None` ("never") when prepared never pays.
pub fn prepared_break_even(prep: f64, t_query: f64, t_unprep: f64) -> Option<f64> {
    if t_unprep > t_query {
        Some(prep / (t_unprep - t_query))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(entry: &'static str, class: Class, m: &[f64]) -> Cell {
        Cell { entry, class, process_medians: m.to_vec() }
    }

    #[test]
    fn decide_wins_when_separated() {
        let cells = [
            cell("A", Class::Scalar, &[100.0, 101.0, 102.0]),
            cell("C", Class::Simd, &[50.0, 51.0, 52.0]),
        ];
        assert_eq!(decide(&cells), Verdict::Wins { entry: "C", class: Class::Simd });
    }

    #[test]
    fn decide_fails_the_85_percent_rule() {
        // ranges separate (90 < 95) but 91 > 0.85 * 96.
        let cells = [
            cell("A", Class::Scalar, &[95.0, 96.0, 97.0]),
            cell("C", Class::Simd, &[89.0, 91.0, 90.0]),
        ];
        assert_eq!(decide(&cells), Verdict::NoMeasuredAdvantage);
    }

    #[test]
    fn decide_fails_overlapping_ranges() {
        // median(W)=50 <= 0.85*100 but max(W)=105 >= min(R)=99.
        let cells = [
            cell("A", Class::Scalar, &[99.0, 100.0, 101.0]),
            cell("C", Class::Simd, &[40.0, 50.0, 105.0]),
        ];
        assert_eq!(decide(&cells), Verdict::NoMeasuredAdvantage);
    }

    #[test]
    fn decide_single_class_is_no_advantage() {
        let cells = [
            cell("C1", Class::Simd, &[10.0, 11.0]),
            cell("C2", Class::Simd, &[50.0, 51.0]),
        ];
        assert_eq!(decide(&cells), Verdict::NoMeasuredAdvantage);
        assert_eq!(decide(&[]), Verdict::NoMeasuredAdvantage);
    }

    #[test]
    fn break_even_arithmetic() {
        assert_eq!(prepared_break_even(100.0, 5.0, 15.0), Some(10.0));
        assert_eq!(prepared_break_even(100.0, 15.0, 15.0), None);
        assert_eq!(prepared_break_even(100.0, 20.0, 15.0), None);
    }

    #[test]
    fn stats_median_odd_and_even() {
        let s = Stats::from_samples(vec![3.0, 1.0, 2.0]);
        assert_eq!((s.median, s.min, s.max), (2.0, 1.0, 3.0));
        let s = Stats::from_samples(vec![4.0, 1.0, 3.0, 2.0]);
        assert_eq!(s.median, 2.5);
    }

    #[test]
    fn allocs_is_monotone_and_guard_passes_without_allocation() {
        let a = allocs();
        let v: Vec<u8> = Vec::with_capacity(16);
        drop(v);
        assert!(allocs() >= a);
        let r = assert_no_alloc("noop", || 1 + 1);
        assert_eq!(r, 2);
    }

    // No #[should_panic] test for assert_no_alloc: it only fires when `Counting`
    // is the global allocator, which it is not in a plain unit-test build.

    #[test]
    fn sweep_and_overhead_smoke() {
        let mut s = Sweep::new(4096);
        assert_ne!(s.run(), 0);
        let o = timer_overhead(3);
        assert_eq!(o.samples.len(), 3);
        assert!(low_res(1.0, 1.0));
        assert!(!low_res(4.0, 1.0));
    }
}
