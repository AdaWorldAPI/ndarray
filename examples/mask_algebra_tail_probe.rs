//! Does the tail DESCENT pay for the mask-ALGEBRA ops, end to end?
//!
//! # The question
//!
//! `examples/ternlogq_tail_descent_probe.rs` measured the descent **tail only**
//! and found it 5-8x faster than zero-padding a `U64x8`. That is one operation.
//! It does NOT say whether the tail is a meaningful FRACTION of a whole
//! `mask_and` call: over `n` words the body runs `n/8` register ops and the
//! tail runs ONE, so at large `n` a faster tail should be invisible, while at
//! `n < 8` the tail IS the whole call. Somewhere between, it stops mattering.
//!
//! The answer decides whether a `U64x4`/`U64x2` facade surface across six
//! backend files is worth building.
//!
//! # The arms
//!
//! - **P — production.** `ndarray::simd::mask_and`, called directly. Not a
//!   re-implementation of it: the real function, so whatever `U64x8` resolves
//!   to on this build (one `_mm512_and_si512` under AVX-512, two 256-bit ops
//!   under v3) is what gets measured.
//! - **D — descent.** The SAME facade `U64x8` body, then the remainder walked
//!   greedily 4 -> 2 -> 1 with `_mm256_and_si256` / `_mm_and_si128` / scalar.
//!   Every lane live, no zero-init, no `copy_from_slice` of a padded array.
//!
//! Both are gated bit-identical against each other and against a scalar
//! reference before any timing, so a faster-but-wrong descent cannot be
//! reported as a win.
//!
//! # Why the descent needs no AVX-512VL
//!
//! `_mm256_and_si256` is AVX2 and `_mm_and_si128` is SSE2. Only
//! `_mm256_ternarylogic_epi64` is a VL instruction. So the bitwise descent runs
//! on the portable v3 baseline as well as on v4, and the result applies to 8 of
//! the 11 mask-algebra tail sites; only `mask_ternlog` needs the VL gate.
//!
//! # Three measurement errors this probe made, and what fixed them
//!
//! Recorded because each produced a plausible number that was not about the
//! variable under test. The first two were caught here, the last three in
//! review on #315.
//!
//! 1. **Confounded bodies.** The arms originally had DIFFERENT bodies
//!    (`as_chunks::<8>()` + a nested loop vs a flat `step_by(4)` loop) and the
//!    difference was reported as a tail result. It is not: at `n % 8 == 0`,
//!    with NO TAIL AT ALL, it still showed `D/P` of 0.84 / 0.76 / 0.72. A tail
//!    strategy cannot move a call that has no tail.
//! 2. **Cache order** was then tested as the explanation and RULED OUT — both
//!    orders agreed. So a residual remained that was neither tail nor order.
//! 3. **The padded arm was a STRAWMAN on v4** (codex, #315). It hand-rolled two
//!    256-bit ops while production `mask_and` issues one `_mm512_and_si512`, so
//!    the AVX-512 numbers compared the descent against something slower than
//!    the real thing. Fixed by calling the real `mask_and`, which also retires
//!    error 1 permanently: there is no hand-written body left to diverge.
//! 4. **Asymmetric noise handling** (codex, #315). A negative tail delta was
//!    reported as noise, while an equally noise-sized POSITIVE delta was
//!    accepted and divided into a ratio — which is where the old table's 23x,
//!    37x and 56x came from. Now a noise floor is MEASURED and applied to both
//!    signs alike.
//! 5. **The isolating section dropped the order control** (coderabbit, #315).
//!    The first table alternated orders; the difference-in-differences loop,
//!    which produces the headline, did not. Now every sample alternates.
//!
//! Run: `cargo run --release --example mask_algebra_tail_probe`

fn main() {
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("mask_algebra_tail_probe: x86_64 only (the descent arm uses AVX2/SSE2); skipping");
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("mask_algebra_tail_probe: needs AVX2 at runtime; skipping");
            return;
        }
        imp::run();
    }
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use ndarray::simd::{mask_and, U64x8};
    use std::arch::x86_64::*;
    use std::hint::black_box;
    use std::time::Instant;

    /// P — the PRODUCTION path, called not re-implemented.
    fn and_production(a: &[u64], b: &[u64], dst: &mut [u64]) {
        mask_and(a, b, dst);
    }

    /// D — the same facade `U64x8` body as production, then a 4 -> 2 -> 1
    /// descent over the remainder instead of a zero-padded full-width op.
    ///
    /// The body is written through the facade (`U64x8`) so it compiles to
    /// whatever backend this build selected, exactly as `mask_and`'s body does.
    /// Only the tail differs.
    fn and_descent(a: &[u64], b: &[u64], dst: &mut [u64]) {
        let n = a.len();
        let done = n & !7;
        for i in (0..done).step_by(8) {
            let va = U64x8::from_slice(&a[i..i + 8]);
            let vb = U64x8::from_slice(&b[i..i + 8]);
            (va & vb).copy_to_slice(&mut dst[i..i + 8]);
        }
        // SAFETY: avx2 was confirmed by the runtime check in `main`, so the
        // 256-bit and 128-bit intrinsics below are legal on this CPU. Every
        // access is bounded by the loop conditions against `n`, which equals
        // `a.len() == b.len() == dst.len()` (asserted by the caller).
        unsafe {
            let mut i = done;
            while n - i >= 4 {
                let va = _mm256_loadu_si256(a.as_ptr().add(i).cast());
                let vb = _mm256_loadu_si256(b.as_ptr().add(i).cast());
                _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), _mm256_and_si256(va, vb));
                i += 4;
            }
            while n - i >= 2 {
                let va = _mm_loadu_si128(a.as_ptr().add(i).cast());
                let vb = _mm_loadu_si128(b.as_ptr().add(i).cast());
                _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), _mm_and_si128(va, vb));
                i += 2;
            }
            while i < n {
                dst[i] = a[i] & b[i];
                i += 1;
            }
        }
    }

    type Arm = fn(&[u64], &[u64], &mut [u64]);

    /// Median-free single timing of one arm: warm up over an eighth of the
    /// budget, then time `iters` back-to-back calls and return ns/call.
    ///
    /// `black_box` wraps BOTH the inputs and one output word, so the optimizer
    /// can neither hoist the call out of the loop nor delete the write.
    fn time_once(f: Arm, a: &[u64], b: &[u64], dst: &mut [u64], iters: u32) -> f64 {
        for _ in 0..(iters / 8).max(1) {
            f(black_box(a), black_box(b), black_box(dst));
        }
        let t = Instant::now();
        for _ in 0..iters {
            f(black_box(a), black_box(b), black_box(dst));
        }
        let e = t.elapsed();
        black_box(&dst[0]);
        e.as_secs_f64() * 1e9 / f64::from(iters)
    }

    /// Median of a timing sample, the robust centre for this probe.
    ///
    /// Deliberately not the mean: a scheduler preemption lands as one huge
    /// outlier, which moves a mean by more than the effect being measured.
    fn median(v: &mut [f64]) -> f64 {
        v.sort_by(|x, y| x.partial_cmp(y).expect("timings are finite"));
        let m = v.len() / 2;
        if v.len() % 2 == 1 {
            v[m]
        } else {
            (v[m - 1] + v[m]) / 2.0
        }
    }

    /// Time both arms on the same input, ALTERNATING which runs first on every
    /// sample, and return each arm's median.
    ///
    /// Alternation is the control for two different biases at once: the arm
    /// timed second inherits a warm cache, and a fixed order lets clock drift
    /// accumulate into one arm. Both were real risks here — the first table's
    /// two-order columns exist because of them — and the isolating section
    /// below originally lacked this control entirely (coderabbit, #315).
    fn time_pair(a: &[u64], b: &[u64], iters: u32, samples: usize) -> (f64, f64) {
        let n = a.len();
        let mut dp = vec![0u64; n];
        let mut dd = vec![0u64; n];
        let mut ps = Vec::with_capacity(samples);
        let mut ds = Vec::with_capacity(samples);
        for s in 0..samples {
            if s % 2 == 0 {
                ps.push(time_once(and_production, a, b, &mut dp, iters));
                ds.push(time_once(and_descent, a, b, &mut dd, iters));
            } else {
                ds.push(time_once(and_descent, a, b, &mut dd, iters));
                ps.push(time_once(and_production, a, b, &mut dp, iters));
            }
        }
        (median(&mut ps), median(&mut ds))
    }

    /// Deterministic pseudo-random operand of `n` words.
    ///
    /// The odd Weyl multiplier gives every word a different bit pattern, so a
    /// tail group cannot accidentally be all-zero and time as free.
    fn mk(n: usize, seed: u64) -> Vec<u64> {
        (0..n)
            .map(|i| seed ^ (i as u64).wrapping_mul(0x9E37_79B9))
            .collect()
    }

    /// Correctness gate: both arms must be BIT-IDENTICAL to a scalar `&`
    /// reference at this width before any timing of that width is believed.
    ///
    /// A descent that silently skipped its tail would be faster and wrong; the
    /// gate is what stops the speed number from being reported anyway.
    fn gate(n: usize) {
        let a = mk(n, 0xF0F0_5555_AAAA_1111);
        let b = mk(n, 0x0FF0_1234_5678_9ABC);
        let mut dp = vec![0u64; n];
        let mut dd = vec![0u64; n];
        and_production(&a, &b, &mut dp);
        and_descent(&a, &b, &mut dd);
        let expect: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x & y).collect();
        assert_eq!(dp, expect, "n={n}: production mask_and must equal the scalar reference");
        assert_eq!(dd, expect, "n={n}: descent must equal the scalar reference");
    }

    pub fn run() {
        let avx512 = std::arch::is_x86_feature_detected!("avx512f");
        println!("mask_algebra_tail_probe: production `mask_and` vs a 4->2->1 tail descent");
        println!("arch=x86_64 avx2=true avx512f={avx512}");
        println!("P is the REAL `ndarray::simd::mask_and`; D shares its facade U64x8 body.\n");

        for n in 1..=16usize {
            gate(n);
        }
        for &n in &[31usize, 64, 65, 256, 1024] {
            gate(n);
        }

        // ── Step 1: measure the NOISE FLOOR ───────────────────────────────
        //
        // Both the baseline and the tail points are timings, so their
        // difference carries both their noise. Repeating the SAME measurement
        // and taking the spread of the differences gives the magnitude below
        // which a tail delta means nothing — in EITHER direction.
        //
        // The old version rejected only negative deltas and divided by the
        // positive ones however small, which manufactured ratios of 23x, 37x
        // and 56x out of noise (codex, #315). The floor is applied to |delta|.
        let base_probe = 64usize;
        let a0 = mk(base_probe, 0xF0F0_5555_AAAA_1111);
        let b0 = mk(base_probe, 0x0FF0_1234_5678_9ABC);
        let iters = 2_000_000u32;
        let mut deltas = Vec::new();
        for _ in 0..7 {
            let (p1, d1) = time_pair(&a0, &b0, iters, 2);
            let (p2, d2) = time_pair(&a0, &b0, iters, 2);
            deltas.push((p1 - p2).abs());
            deltas.push((d1 - d2).abs());
        }
        let floor = deltas.iter().cloned().fold(0.0f64, f64::max);
        println!("noise floor (max |repeat - repeat| over 14 same-input pairs): {floor:.2} ns");
        println!("A tail delta whose magnitude is below this is reported as `~noise`, either sign.\n");

        // ── Step 2: the isolating measurement ─────────────────────────────
        //
        // `t(base + k) - t(base)` at a fixed body size is the cost of a k-word
        // tail and nothing else: the body work is identical in both terms and
        // cancels per arm, whatever that arm's body compiled to.
        println!("── tail cost ISOLATED: t(base + k) - t(base), body held constant ──");
        println!("{:>6}  {:>3}  {:>11} {:>11}  {:>8}", "base", "k", "P tail ns", "D tail ns", "P/D");
        let mut wins = 0usize;
        let mut rows = 0usize;
        for &base in &[8usize, 64] {
            let ab = mk(base, 0xF0F0_5555_AAAA_1111);
            let bb = mk(base, 0x0FF0_1234_5678_9ABC);
            let (p0, d0) = time_pair(&ab, &bb, iters, 5);
            for k in 1..8usize {
                let n = base + k;
                let a = mk(n, 0xF0F0_5555_AAAA_1111);
                let b = mk(n, 0x0FF0_1234_5678_9ABC);
                let (pk, dk) = time_pair(&a, &b, iters, 5);
                let pt = pk - p0;
                let dt = dk - d0;
                rows += 1;
                let ps = if pt.abs() < floor {
                    format!("{pt:>8.2} ~n")
                } else {
                    format!("{pt:>11.2}")
                };
                let ds = if dt.abs() < floor {
                    format!("{dt:>8.2} ~n")
                } else {
                    format!("{dt:>11.2}")
                };
                // A ratio is only meaningful when BOTH terms clear the floor.
                // Below it, the honest answer is that the tail is too cheap to
                // measure — which is itself the finding, not a missing number.
                let rs = if pt.abs() < floor {
                    "     n/a".to_string()
                } else if dt.abs() < floor {
                    wins += 1;
                    "  D~noise".to_string()
                } else if dt > 0.0 {
                    if pt / dt > 2.0 {
                        wins += 1;
                    }
                    format!("{:>8.2}", pt / dt)
                } else {
                    "     n/a".to_string()
                };
                println!("{base:>6}  {k:>3}  {ps} {ds}  {rs}");
            }
        }
        println!("\n`~n` = magnitude below the noise floor, i.e. indistinguishable from no tail.");
        println!("`D~noise` = the padded tail is measurable and the descent's is not.");
        println!("A ratio is printed only when BOTH terms clear the floor.");
        println!("\ndescent clearly cheaper on {wins} of {rows} tail widths.");
    }
}
