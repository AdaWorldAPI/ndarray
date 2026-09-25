//! Can a 2/3-input Boolean membership END in Count/Any without writing a mask?
//!
//! The question this probe answers is narrower than "is a fused primitive
//! faster". It is: **does the existing `U64x8` surface already compose the
//! fold register-only**, or is a new T1 primitive required? Every realization
//! of `U64x8` (avx512 / avx2-polyfill / scalar / neon / wasm) carries
//! `ternlog::<IMM>`, `popcnt`, `+` and `reduce_sum`, so the composition
//! `ternlog → popcnt → accumulate → reduce_sum` type-checks everywhere. What
//! the probe measures is whether that composition is a real win over the
//! materializing path, on each backend this workspace builds for.
//!
//! Three arms compute the IDENTICAL result for `(a & b) | c` (`AND2_OR`, the
//! mask-risc flagship immediate); a correctness gate aborts on disagreement:
//!
//! | arm | shape | derived words written |
//! |---|---|---|
//! | M | `mask_ternlog::<IMM>` into a buffer, then `popcount_batch_u64` | n |
//! | R | chunked `U64x8::ternlog → popcnt`, lane-wise accumulate, one `reduce_sum` | 0 |
//! | S | plain scalar `((a & b) \| c).count_ones()` over zipped slices | 0 |
//!
//! Any is measured the same way: M = materialize + `mask_any`; R =
//! OR-accumulate with a test once per block of 8 chunks; S = scalar `any`. Any
//! is timed on an all-zero result (the worst case: no early exit is possible).
//! A first R arm that tested every chunk LOST to M on avx2 at 16K words (0.91×)
//! — the per-chunk horizontal test cost more than the ternlog it guarded.
//!
//! Measured 2026-09-23 (median ns, avx2 = `config-v3`, avx512 = `config-v4`,
//! a host without `avx512vpopcntdq`, so avx512 `popcnt` is the scalar-lane
//! fallback, which LLVM compiles as fast as a Mula LUT; see
//! `ternlog_popcnt_gap_probe`):
//!
//! | backend | words | Count M/R | Any M/R |
//! |---|---|---|---|
//! | avx2 | 16 384 | 1.27 | 3.24 |
//! | avx2 | 262 144 | 1.50 | 6.18 |
//! | avx512 | 16 384 | 1.90 | 2.59 |
//! | avx512 | 262 144 | 1.94 | 4.39 |
//!
//! The scalar fused arm S is ~equal to R on avx2 and at the memory-bound size
//! on avx512 — the win here is NOT writing the mask, not SIMD per se.
//!
//! Usage (the backend is compile-time and the default config is
//! `target-cpu=native`, so pin the tier and read the `backend:` line):
//!
//! ```text
//! env -u RUSTFLAGS cargo --config .cargo/config-v3.toml run --release --example ternlog_fold_probe
//! env -u RUSTFLAGS cargo --config .cargo/config-v4.toml run --release --example ternlog_fold_probe
//! ```

use std::time::Instant;

use ndarray::simd::ternlog::AND2_OR;
use ndarray::simd::{mask_any, mask_ternlog, popcount_batch_u64, U64x8};

const L: usize = U64x8::LANES;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s
}

/// Arm R: the register-only composition over existing `U64x8` methods.
fn fold_count(a: &[u64], b: &[u64], c: &[u64]) -> u64 {
    let (ca, ta) = a.as_chunks::<L>();
    let (cb, tb) = b.as_chunks::<L>();
    let (cc, tc) = c.as_chunks::<L>();
    let mut acc = U64x8::splat(0);
    for ((x, y), z) in ca.iter().zip(cb).zip(cc) {
        let t = U64x8::from_array(*x).ternlog::<AND2_OR>(U64x8::from_array(*y), U64x8::from_array(*z));
        acc += t.popcnt();
    }
    let mut tail = 0u64;
    for ((x, y), z) in ta.iter().zip(tb).zip(tc) {
        tail += ((x & y) | z).count_ones() as u64;
    }
    acc.reduce_sum() + tail
}

fn fold_any(a: &[u64], b: &[u64], c: &[u64]) -> bool {
    let (ca, ta) = a.as_chunks::<L>();
    let (cb, tb) = b.as_chunks::<L>();
    let (cc, tc) = c.as_chunks::<L>();
    // OR-accumulate in a register and test once per block of chunks: a
    // per-chunk horizontal test costs more than the ternlog it guards.
    const BLOCK: usize = 8;
    let mut acc = U64x8::splat(0);
    for (i, ((x, y), z)) in ca.iter().zip(cb).zip(cc).enumerate() {
        acc |= U64x8::from_array(*x).ternlog::<AND2_OR>(U64x8::from_array(*y), U64x8::from_array(*z));
        if i % BLOCK == BLOCK - 1 && acc.to_array().iter().any(|&w| w != 0) {
            return true;
        }
    }
    if acc.to_array().iter().any(|&w| w != 0) {
        return true;
    }
    ta.iter()
        .zip(tb)
        .zip(tc)
        .any(|((x, y), z)| (x & y) | z != 0)
}

fn median<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut ts = Vec::with_capacity(reps);
    let mut last = None;
    for _ in 0..reps {
        let t = Instant::now();
        last = Some(std::hint::black_box(f()));
        ts.push(t.elapsed().as_nanos() as f64);
    }
    ts.sort_by(|x, y| x.total_cmp(y));
    (ts[reps / 2], last.expect("reps > 0"))
}

fn main() {
    let backend = if cfg!(target_feature = "avx512f") {
        "avx512"
    } else if cfg!(target_feature = "avx2") {
        "avx2-polyfill"
    } else {
        "scalar"
    };
    println!("backend: {backend}");
    println!(
        "{:>8} {:>6} {:>12} {:>12} {:>12} {:>8} {:>8}",
        "words", "term", "M_ns", "R_ns", "S_ns", "M/R", "S/R"
    );
    let mut seed = 0x7E4_u64;
    for &n in &[8usize, 128, 16_384, 262_144] {
        let a: Vec<u64> = (0..n).map(|_| lcg(&mut seed)).collect();
        let b: Vec<u64> = (0..n).map(|_| lcg(&mut seed)).collect();
        let c: Vec<u64> = (0..n).map(|_| lcg(&mut seed) & lcg(&mut seed)).collect();
        let zero = vec![0u64; n];
        let mut dst = vec![0u64; n];
        let reps = if n > 100_000 { 41 } else { 2001 };

        let (m, vm) = median(reps, || {
            mask_ternlog::<AND2_OR>(&a, &b, &c, &mut dst);
            popcount_batch_u64(&dst)
        });
        let (r, vr) = median(reps, || fold_count(&a, &b, &c));
        let (s, vs) = median(reps, || {
            a.iter()
                .zip(&b)
                .zip(&c)
                .map(|((x, y), z)| ((x & y) | z).count_ones() as u64)
                .sum::<u64>()
        });
        assert!(vm == vr && vr == vs, "count disagreement at n={n}: {vm} {vr} {vs}");
        println!("{n:>8} {:>6} {m:>12.0} {r:>12.0} {s:>12.0} {:>8.2} {:>8.2}", "Count", m / r, s / r);

        // Any over an all-zero result: a = b = c = 0, so no early exit.
        let (m, vm) = median(reps, || {
            mask_ternlog::<AND2_OR>(&zero, &zero, &zero, &mut dst);
            mask_any(&dst)
        });
        let (r, vr) = median(reps, || fold_any(&zero, &zero, &zero));
        let (s, vs) = median(reps, || {
            zero.iter()
                .zip(&zero)
                .zip(&zero)
                .any(|((x, y), z)| (x & y) | z != 0)
        });
        assert!(!vm && !vr && !vs, "any disagreement at n={n}");
        println!("{n:>8} {:>6} {m:>12.0} {r:>12.0} {s:>12.0} {:>8.2} {:>8.2}", "Any", m / r, s / r);
    }
}
