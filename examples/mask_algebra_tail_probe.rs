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
//! # What the padded tail is actually FOR — read this before proposing a peel
//!
//! The zero-padded tail is not an oversight and not a speed choice. It was
//! chosen deliberately (`simd_masking_ops.rs:102-107`, codegen witness
//! 2026-09-14) for codegen UNIFORMITY: an exact-length scalar tail "fully
//! unrolled ... on aarch64 into 7 x (and, orr) on GPRs ... in a facade op
//! whose contract is `packed on every backend`", and became `vpmaskmovq` on
//! AVX2. Padding makes both arms one shape — packed body, packed tail.
//!
//! That header is also explicit that "no throughput comparison against the old
//! peel has been made". THAT is the gap this probe fills. So any tail proposal
//! clears two bars, and being faster is only the first:
//!
//! 1. **Throughput** — measured here. The padded tail costs 8-20 ns against a
//!    body of ~3.5 ns at 8 words, so it is routinely larger than the work it
//!    trails, and 7 of 8 mask sizes have one.
//! 2. **Packed on every backend** — measured in `narrow_bitop_codegen_probe`,
//!    including on the backend the objection was about. A FIXED-WIDTH peel
//!    emits `and v0.16b` on aarch64 and `vandps ymm`/`xmm` on x86. The
//!    distinction the 2026-09-14 witness could not draw is between an
//!    EXACT-LENGTH tail, whose trip count is a runtime value, and a
//!    FIXED-WIDTH step, whose trip count is a constant. Only the first
//!    degenerates to GPRs — which is also exactly why arm `S` below loses to
//!    arm `F`.
//!
//! So the recommendation is not "padding was wrong". Padding bought a real
//! property with a real measurement behind it; fixed-width steps keep that
//! property AND stop paying 8-20 ns for it.
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

    /// S — the same facade body, then the tail as a PLAIN SCALAR LOOP.
    ///
    /// The arm that could make the whole descent unnecessary. `D` beats `P`
    /// because padding to full width does real work on words nobody wants,
    /// not because intrinsics are magic — and a bounded scalar tail loop is
    /// not obviously slow: measured on this tree (`narrow_bitop_codegen_probe`),
    /// a fixed `for i in 0..4` over `&[u64; 4]` emits exactly
    /// `vmovups/vandps ymm/vmovups`, and the 2-lane form the `xmm` equivalent.
    /// LLVM already descends a trivially-bounded loop.
    ///
    /// If S ties D, the facade needs NO new narrow type, no `avx512vl` gate
    /// and no six-backend edit: the tail is a `while i < n` and portability
    /// comes free. That is the cheapest possible shape, so it has to be
    /// falsified before the expensive one is built.
    fn and_scalar_tail(a: &[u64], b: &[u64], dst: &mut [u64]) {
        let n = a.len();
        let done = n & !7;
        for i in (0..done).step_by(8) {
            let va = U64x8::from_slice(&a[i..i + 8]);
            let vb = U64x8::from_slice(&b[i..i + 8]);
            (va & vb).copy_to_slice(&mut dst[i..i + 8]);
        }
        for i in done..n {
            dst[i] = a[i] & b[i];
        }
    }

    /// F — the synthesis, and the arm that should make this whole question
    /// cheap: the same 4 -> 2 -> 1 descent as `D`, written in PLAIN RUST with
    /// FIXED-SIZE steps and not one intrinsic.
    ///
    /// The mechanism `S` was missing. `S`'s tail is `for i in done..n`, a
    /// loop whose trip count is unknown at compile time, so LLVM emits a
    /// scalar loop with a branch per word. `F`'s steps are `[u64; 4]` and
    /// `[u64; 2]` — trip counts fixed at compile time, which
    /// `narrow_bitop_codegen_probe` measured compiling to a single
    /// `vandps ymm` and `vandps xmm` respectively.
    ///
    /// If F ties D, the descent needs no narrow facade type, no `avx512vl`
    /// gate and no backend edits: it is a portable helper that every backend
    /// already compiles correctly, and it reaches the tail sites on NEON and
    /// wasm too, which an x86 intrinsic descent never could.
    fn and_fixed_descent(a: &[u64], b: &[u64], dst: &mut [u64]) {
        let n = a.len();
        let done = n & !7;
        for i in (0..done).step_by(8) {
            let va = U64x8::from_slice(&a[i..i + 8]);
            let vb = U64x8::from_slice(&b[i..i + 8]);
            (va & vb).copy_to_slice(&mut dst[i..i + 8]);
        }
        let mut i = done;
        if n - i >= 4 {
            for j in 0..4 {
                dst[i + j] = a[i + j] & b[i + j];
            }
            i += 4;
        }
        if n - i >= 2 {
            for j in 0..2 {
                dst[i + j] = a[i + j] & b[i + j];
            }
            i += 2;
        }
        if i < n {
            dst[i] = a[i] & b[i];
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
        // NaN, not a panic: an empty sample is a real outcome here, not a bug.
        // The qualification filter can reject every width in a pass when the
        // machine is noisy, and "nothing was resolvable" is the finding — a
        // probe that crashes instead of reporting it would look like broken
        // code rather than an unresolvable measurement.
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|x, y| x.partial_cmp(y).expect("timings are finite"));
        let m = v.len() / 2;
        if v.len() % 2 == 1 {
            v[m]
        } else {
            (v[m - 1] + v[m]) / 2.0
        }
    }

    /// Time all three arms on the same input, ROTATING which runs first on
    /// every sample, and return each arm's median.
    ///
    /// Alternation is the control for two different biases at once: the arm
    /// timed second inherits a warm cache, and a fixed order lets clock drift
    /// accumulate into one arm. Both were real risks here — the first table's
    /// two-order columns exist because of them — and the isolating section
    /// below originally lacked this control entirely (coderabbit, #315). With
    /// four arms the alternation became a rotation: a two-way swap would
    /// have pinned the other two arms to fixed positions.
    fn time_arms(a: &[u64], b: &[u64], iters: u32, samples: usize) -> [f64; 4] {
        let n = a.len();
        let mut d: [Vec<u64>; 4] = [vec![0; n], vec![0; n], vec![0; n], vec![0; n]];
        let arms: [Arm; 4] = [and_production, and_descent, and_scalar_tail, and_fixed_descent];
        let mut acc: [Vec<f64>; 4] = [
            Vec::with_capacity(samples),
            Vec::with_capacity(samples),
            Vec::with_capacity(samples),
            Vec::with_capacity(samples),
        ];
        for s in 0..samples {
            // Rotate which arm goes first. With more than two arms a swap is no
            // longer enough: a fixed order gives arm 0 a cold cache on every
            // sample and the last arm a warm one, forever.
            for step in 0..4 {
                let k = (s + step) % 4;
                let t = time_once(arms[k], a, b, &mut d[k], iters);
                acc[k].push(t);
            }
        }
        let [mut p, mut dd, mut ss, mut ff] = acc;
        [median(&mut p), median(&mut dd), median(&mut ss), median(&mut ff)]
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
        let mut ds = vec![0u64; n];
        and_scalar_tail(&a, &b, &mut ds);
        let mut df = vec![0u64; n];
        and_fixed_descent(&a, &b, &mut df);
        let expect: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x & y).collect();
        assert_eq!(dp, expect, "n={n}: production mask_and must equal the scalar reference");
        assert_eq!(dd, expect, "n={n}: descent must equal the scalar reference");
        assert_eq!(ds, expect, "n={n}: scalar-tail arm must equal the scalar reference");
        assert_eq!(df, expect, "n={n}: fixed-step descent must equal the scalar reference");
    }

    pub fn run() {
        // Two DIFFERENT facts, and conflating them is the trap this workspace
        // keeps paying for. `is_x86_feature_detected!` asks the HOST CPU what
        // it can do; `cfg!(target_feature)` reports what this BUILD was
        // compiled for. Under `--config .cargo/config-v3.toml` on an AVX-512
        // host the first says true and the second says false — so a probe that
        // prints only the runtime bit lets a v3 measurement be read as v4.
        let host_avx512 = std::arch::is_x86_feature_detected!("avx512f");
        let built_avx512 = cfg!(target_feature = "avx512f");
        println!("mask_algebra_tail_probe: production `mask_and` vs a 4->2->1 tail descent");
        println!("arch=x86_64  BUILT-FOR avx512f={built_avx512}  (host cpu avx512f={host_avx512})");
        println!("the BUILT-FOR bit is the tier these timings belong to, not the host bit.");
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
            let [p1, d1, s1, f1] = time_arms(&a0, &b0, iters, 2);
            let [p2, d2, s2, f2] = time_arms(&a0, &b0, iters, 2);
            deltas.push((p1 - p2).abs());
            deltas.push((d1 - d2).abs());
            deltas.push((s1 - s2).abs());
            deltas.push((f1 - f2).abs());
        }
        let floor = deltas.iter().cloned().fold(0.0f64, f64::max);
        println!("noise floor (max |repeat - repeat| over 28 same-input repeats): {floor:.2} ns");
        println!("A tail delta whose magnitude is below this is reported as `~noise`, either sign.\n");

        // ── Step 2: the isolating measurement ─────────────────────────────
        //
        // `t(base + k) - t(base)` at a fixed body size is the cost of a k-word
        // tail and nothing else: the body work is identical in both terms and
        // cancels per arm, whatever that arm's body compiled to.
        println!("── tail cost ISOLATED: t(base + k) - t(base), body held constant ──");
        println!(
            "{:>6}  {:>3}  {:>11} {:>11} {:>11} {:>11}",
            "base", "k", "P tail ns", "D tail ns", "S tail ns", "F tail ns"
        );
        let mut wins = 0usize;
        let mut rows = 0usize;
        // REPLICATE the whole sweep. A single pass gave D - F of -2.0, -8.2,
        // -2.3, -4.4 and -0.4 points on v4 but -1.6, +3.1 and +5.3 on v3: the
        // sign is not stable, so any verdict read off one pass is a draw, not
        // a result. Repeating inside the probe is what lets it report the
        // SPREAD instead of inviting the reader to trust whichever pass ran.
        const REPEATS: usize = 7;
        let mut ds_pts = Vec::with_capacity(REPEATS);
        let mut df_pts = Vec::with_capacity(REPEATS);
        let mut ds_gap = Vec::new();
        // Fraction of the padded tail's cost that each strategy removes. This
        // is the statistic the verdict rests on, because the obvious one — a
        // COUNT of widths where D beats S — is thresholded by the noise floor
        // and the floor is itself a draw: consecutive runs of this probe gave
        // 2 of 14 and 10 of 14 from floors of 1.35 ns and 0.64 ns. A count
        // that flips with the floor cannot decide a six-file change; the
        // magnitudes it was thresholding were stable the whole time.
        let mut s_frac = Vec::new();
        let mut d_frac = Vec::new();
        let mut f_frac = Vec::new();
        // How many widths were actually resolvable. The verdict is gated on
        // this: a tail costs ~1 ns and the floor on a busy machine is ~2 ns,
        // so on some runs NOTHING qualifies and the only honest output is to
        // say the question was not answered.
        let mut qualified = 0usize;
        // Absolute tail cost in ns on the qualified widths, per arm. Ratios and
        // shares can blow up when a denominator or numerator approaches zero;
        // a nanosecond cannot. These four numbers are the least-processed form
        // of the result and are what the verdict should be sanity-checked
        // against.
        let mut p_ns = Vec::new();
        let mut d_ns = Vec::new();
        let mut s_ns = Vec::new();
        let mut f_ns = Vec::new();
        for rep in 0..REPEATS {
            // Mark where this pass's contributions start. The vectors are NOT
            // cleared: the headline medians POOL every pass, and only the
            // spread line slices out one pass at a time. Clearing here — which
            // is what this code did until it was caught by re-reading the diff
            // — left the line labelled "pooled over all N passes" reading the
            // LAST pass alone, a false label on the headline number.
            let pass_start = s_frac.len();
            for &base in &[8usize, 64] {
                let ab = mk(base, 0xF0F0_5555_AAAA_1111);
                let bb = mk(base, 0x0FF0_1234_5678_9ABC);
                let [p0, d0, s0, f0] = time_arms(&ab, &bb, iters, 9);
                for k in 1..8usize {
                    let n = base + k;
                    let a = mk(n, 0xF0F0_5555_AAAA_1111);
                    let b = mk(n, 0x0FF0_1234_5678_9ABC);
                    let [pk, dk, sk, fk] = time_arms(&a, &b, iters, 9);
                    let pt = pk - p0;
                    let dt = dk - d0;
                    let st = sk - s0;
                    let ft = fk - f0;
                    rows += 1;
                    ds_gap.push(st - dt);
                    // A width contributes to the fraction statistics only
                    // when ALL FOUR of its tail costs are measurable AND
                    // positive. Without this, a tail that timed NEGATIVE (the
                    // longer array came out faster — pure noise) gives
                    // (pt - dt)/pt > 1, and pooling those produced a "D
                    // removes 121.3% of the padded cost", which is not a
                    // quantity. Same asymmetric-noise error codex caught in
                    // the ratio column, reintroduced one statistic later:
                    // rejecting a negative DENOMINATOR is not enough when a
                    // negative NUMERATOR inflates instead.
                    if pt >= floor && dt >= floor && st >= floor && ft >= floor {
                        qualified += 1;
                        p_ns.push(pt);
                        d_ns.push(dt);
                        s_ns.push(st);
                        f_ns.push(ft);
                        s_frac.push((pt - st) / pt);
                        d_frac.push((pt - dt) / pt);
                        f_frac.push((pt - ft) / pt);
                    }
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
                    let ss = if st.abs() < floor {
                        format!("{st:>8.2} ~n")
                    } else {
                        format!("{st:>11.2}")
                    };
                    let fs = if ft.abs() < floor {
                        format!("{ft:>8.2} ~n")
                    } else {
                        format!("{ft:>11.2}")
                    };
                    // "Cheaper than the padded tail" counts a width when the
                    // padded cost is measurable and the descent's is either below
                    // the floor or more than 2x smaller. A ratio is deliberately
                    // NOT printed: below the floor the denominator is noise, and a
                    // number there would invent precision the measurement lacks.
                    if pt.abs() >= floor && (dt.abs() < floor || (dt > 0.0 && pt / dt > 2.0)) {
                        wins += 1;
                    }
                    if rep == 0 {
                        println!("{base:>6}  {k:>3}  {ps} {ds} {ss} {fs}");
                    }
                }
            }
            let sp = median(&mut s_frac[pass_start..].to_vec()) * 100.0;
            let dp = median(&mut d_frac[pass_start..].to_vec()) * 100.0;
            let fp = median(&mut f_frac[pass_start..].to_vec()) * 100.0;
            if (dp - sp).is_finite() {
                ds_pts.push(dp - sp);
            }
            if (dp - fp).is_finite() {
                df_pts.push(dp - fp);
            }
            if rep == 0 {
                println!("\n(table above is pass 1 of {REPEATS}; the verdict uses all {REPEATS})");
            }
        }
        println!("\n`~n` = magnitude below the noise floor, i.e. indistinguishable from no tail.");
        println!("P = production padded tail, D = x86 intrinsic descent,");
        println!("S = variable-length scalar loop, F = portable fixed-step descent.");
        println!("\ndescent clearly cheaper than the PADDED tail on {wins} of {rows} tail widths.");

        // ── Step 3: the question that decides what gets BUILT ─────────────
        println!("\n── what does the tail actually need: intrinsics, or a fixed trip count? ──");
        let med_gap = median(&mut ds_gap);
        println!("median (S - D) within pass 1: {med_gap:.2} ns   [noise floor {floor:.2} ns]");

        let lo = |v: &[f64]| {
            if v.is_empty() {
                f64::NAN
            } else {
                v.iter().cloned().fold(f64::INFINITY, f64::min)
            }
        };
        let hi = |v: &[f64]| {
            if v.is_empty() {
                f64::NAN
            } else {
                v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            }
        };
        let s_all = median(&mut s_frac.clone()) * 100.0;
        let d_all = median(&mut d_frac.clone()) * 100.0;
        let f_all = median(&mut f_frac.clone()) * 100.0;
        let ds_med = d_all - s_all;
        let df_med = d_all - f_all;
        // NaN reaches these lines whenever the filter rejected every width, so
        // it prints as `n/a` rather than as a number: a bare `NaN%` reads like
        // a broken probe when it actually means "not resolvable on this run".
        let pc = |x: f64| {
            if x.is_finite() {
                format!("{x:.1}%")
            } else {
                "n/a".to_string()
            }
        };
        let pt_ = |x: f64| {
            if x.is_finite() {
                format!("{x:>5.1}")
            } else {
                "  n/a".to_string()
            }
        };
        println!(
            "\nmedian ABSOLUTE tail cost on the {qualified} qualified widths (ns):\n\
             \x20 P {:.2}   D {:.2}   S {:.2}   F {:.2}",
            median(&mut p_ns.clone()),
            median(&mut d_ns.clone()),
            median(&mut s_ns.clone()),
            median(&mut f_ns.clone())
        );
        // ⊘ The filter is a SELECTION on the very quantities being compared,
        // and the first version of this note had its direction backwards. It
        // said the filter is "conservative AGAINST D, S and F" because it
        // keeps the widths where their tails are most expensive, and that for
        // D-vs-F it "restricts the subset without tilting the difference"
        // because the requirement is symmetric. The second half is false
        // (codex P2, #315): when the true costs sit near the floor — exactly
        // this regime — an arm qualifies only on draws where its OWN
        // measurement error pushed it upward. That truncation compresses the
        // observed D-F gap toward zero, which systematically favours the
        // "fixed steps tie intrinsics" reading. Symmetry does not remove the
        // conditioning, it applies it to both arms.
        //
        // The filter stays because unfiltered is worse — negative tails gave
        // "D removes 121.3% of the padded cost" — but this is a CENSORED
        // sample and the D-F gap it yields is a lower bound on any true
        // difference. Resolving it properly needs precision, not filtering:
        // more iterations, a quiet machine, or modelling the censored points.
        println!("\npooled over all {REPEATS} passes -- S {}  D {}  F {}", pc(s_all), pc(d_all), pc(f_all));
        println!("\nShare of the padded tail's cost removed, differenced per pass.");
        println!("Percentage-point gaps, pooled, with the per-pass spread beside them:");
        println!(
            "  D - S = {} [{}, {}]   intrinsic descent vs a variable-length loop",
            pt_(ds_med),
            pt_(lo(&ds_pts)),
            pt_(hi(&ds_pts))
        );
        println!(
            "  D - F = {} [{}, {}]   intrinsic descent vs PORTABLE fixed steps",
            pt_(df_med),
            pt_(lo(&df_pts)),
            pt_(hi(&df_pts))
        );

        // The decision is not "does D ever win" — the question is whether what
        // intrinsics buy over PORTABLE fixed steps is worth a narrow type on
        // six backend files plus an `avx512vl` gate. Compare the two gaps: if
        // D - F is small beside D - S, the mechanism was the trip count and
        // not the instruction set, and the portable form is the one to build.
        // A verdict is only available when the baseline comparison itself is
        // positive and clear. If the intrinsic descent does not measurably
        // beat even a variable-length loop on this machine, nothing here can
        // adjudicate the finer D-vs-F question, and saying so is the result.
        let resolvable = qualified * 4 >= rows && ds_med.is_finite() && df_med.is_finite();
        println!(
            "widths where all four tails were measurable: {qualified} of {rows}               ({}resolvable)",
            if resolvable { "" } else { "NOT " }
        );
        if !resolvable {
            println!(
                "\nVERDICT: INCONCLUSIVE — and that is the honest result, not a failed run.\n\
                 A k-word tail costs on the order of 1 ns; the noise floor here measured\n\
                 {floor:.2} ns. Only {qualified} of {rows} widths had all four tail costs resolvable\n\
                 above it, so the D-vs-F comparison has no support in this data and any\n\
                 number printed for it would be arithmetic on noise.\n\
                 Re-run on a quiet machine, or with larger `iters`, before reading a\n\
                 verdict into the gaps below.\n\
                 What DOES survive, because it is an order of magnitude larger than the\n\
                 floor: every tail strategy removes most of the PADDED tail's 8-20 ns,\n\
                 which is the finding this probe was built to establish.\n\
                 \n\
                 This branch deliberately stops here. An earlier version went on to\n\
                 say D, S and F had landed \"within a point or two\" and to recommend F\n\
                 anyway — a measurement claim drawn from the data the same paragraph\n\
                 had just rejected, on a run where those gaps may be NaN or arbitrarily\n\
                 large (codex P2, #315). Nothing here establishes that closeness, so\n\
                 nothing here recommends an architecture."
            );
        } else if ds_med <= 2.0 {
            println!(
                "\nVERDICT: INCONCLUSIVE on this run. The intrinsic descent beat a plain\n\
                 variable-length tail loop by only {ds_med:.1} points, so this machine is too\n\
                 noisy right now to resolve the smaller D-vs-F gap ({df_med:.1} points).\n\
                 Re-run on a quiet machine before reading anything into either.\n\
                 What IS solid here and does not depend on that comparison: all three\n\
                 tail strategies remove ~80-95% of the PADDED tail's cost, which is the\n\
                 finding this probe was built for."
            );
        } else if df_med < ds_med / 2.0 {
            println!(
                "\nVERDICT: the mechanism is the TRIP COUNT, not the instruction set.\n\
                 Against a variable-length `for i in done..n` tail the intrinsic descent\n\
                 is worth a median {ds_med:.1} points; against fixed-width steps written in\n\
                 plain Rust it is worth {df_med:.1}. A `for j in 0..4` has a trip count known at\n\
                 compile time and compiles to a single `vandps ymm`\n\
                 (`narrow_bitop_codegen_probe`, where the facade's own `U64x4 &`\n\
                 emits assembly the assembler ALIASES to the hand-written loop);\n\
                 `done..n` does not and cannot.\n\
                 So the `U64x4`/`U64x2` facade surface and the `avx512vl` gate are NOT\n\
                 justified. Write the tail as fixed-width steps: no backend edits,\n\
                 no raw intrinsics — hence no exception to the all-SIMD-from-the-facade\n\
                 invariant — and it reaches the NEON, wasm and scalar tails, which an\n\
                 x86 intrinsic descent never could."
            );
        } else {
            println!(
                "\nVERDICT: intrinsics buy a median {df_med:.1} points over portable fixed steps,\n\
                 a real fraction of the {ds_med:.1} they buy over a variable-length loop.\n\
                 The facade narrow-type surface has measured support. Build phase 1."
            );
        }
    }
}
