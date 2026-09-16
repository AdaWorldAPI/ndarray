//! Does the tail DESCENT pay for the mask-ALGEBRA ops, end to end?
//!
//! # The question, and why it is not already answered
//!
//! `examples/ternlogq_tail_descent_probe.rs` measured the descent **tail only**
//! and found it 5-8x faster than zero-padding a `U64x8`. That result is real
//! and it is about one operation. What it does NOT say is whether the tail is a
//! meaningful FRACTION of a whole `mask_and` / `mask_or` / `mask_xor` call:
//! over `n` words the body runs `n/8` register ops and the tail runs ONE, so at
//! `n = 1024` a 5x-faster tail moves ~1/128 of the work and the end-to-end
//! effect should be invisible. At `n < 8` the tail IS the whole call.
//!
//! Somewhere between those two the descent stops mattering. This probe finds
//! where, so the decision to build a `U64x4`/`U64x2` facade surface across six
//! backend files rests on a measurement rather than on the tail-only number.
//!
//! # Arms
//!
//! - **P — padded**: exactly what `mask_and` does today. `as_chunks::<8>()`
//!   body, then both operands zero-padded into a `U64x8` and the first
//!   `tail.len()` lanes copied back out.
//! - **D — descent**: same body, then the remainder walked greedily 4 -> 2 -> 1
//!   with `_mm256_and_si256` / `_mm_and_si128` / scalar `&`. Every lane live,
//!   no zero-init, no `copy_from_slice` of a padded array.
//!
//! Both arms produce identical output; the probe asserts that before timing, so
//! a faster-but-wrong descent cannot be reported as a win.
//!
//! # The confound this probe had, and why the arms now share a body
//!
//! The first version gave the two arms DIFFERENT bodies — `as_chunks::<8>()`
//! with a nested 2x256-bit loop for P, a flat `step_by(4)` loop for D — and
//! then reported the difference as a tail result. It is not: at `n = 256`,
//! `1024` and `4096`, where `n % 8 == 0` and there is NO TAIL AT ALL, it still
//! showed `D/P` of 0.844, 0.763 and 0.716. A tail strategy cannot move a call
//! that has no tail, so those rows were measuring body shape.
//!
//! Both arms now call ONE shared `body()` and differ only after it. Any
//! remaining `D/P != 1` at `n % 8 == 0` would be noise or layout, never the
//! variable under test — which makes those rows the probe's own control.
//!
//! # Why this needs no AVX-512
//!
//! `_mm256_and_si256` is AVX2 and `_mm_and_si128` is SSE2. The bitwise descent
//! for and/or/xor/andnot needs **no AVX-512VL** at all — only the `ternlog`
//! descent does, because `_mm256_ternarylogic_epi64` is a VL instruction. So
//! this arm runs on the portable v3 baseline as well as on v4, and the result
//! applies to 8 of the 11 mask-algebra tail sites.
//!
//! Run: `cargo run --release --example mask_algebra_tail_probe`

fn main() {
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("mask_algebra_tail_probe: x86_64 only (uses AVX2/SSE2 intrinsics); skipping");
    }
    #[cfg(target_arch = "x86_64")]
    {
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("mask_algebra_tail_probe: needs AVX2 at runtime; skipping");
            return;
        }
        // SAFETY: avx2 confirmed present by the runtime check above; every
        // callee below is `#[target_feature(enable = "avx2")]` and every access
        // it makes is bounded by the slice lengths asserted equal in `run`.
        unsafe { imp::run() }
    }
}

#[cfg(target_arch = "x86_64")]
mod imp {
    use std::arch::x86_64::*;
    use std::hint::black_box;
    use std::time::Instant;

    /// The SHARED body, identical in both arms: full 4-word registers over the
    /// first `n & !7` words. Factored out deliberately — see the module note on
    /// the confound this removes.
    #[target_feature(enable = "avx2")]
    unsafe fn body(a: &[u64], b: &[u64], dst: &mut [u64], upto: usize) {
        for i in (0..upto).step_by(4) {
            // SAFETY: i + 4 <= upto <= n = a.len() = b.len() = dst.len().
            let va = _mm256_loadu_si256(a.as_ptr().add(i).cast());
            let vb = _mm256_loadu_si256(b.as_ptr().add(i).cast());
            _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), _mm256_and_si256(va, vb));
        }
    }

    /// P — shared body, then the CURRENT tail: both operands zero-padded into a
    /// `[u64; 8]`, one full-width op, first `t` lanes copied back.
    #[target_feature(enable = "avx2")]
    unsafe fn and_padded(a: &[u64], b: &[u64], dst: &mut [u64]) {
        let n = a.len();
        let done = n & !7;
        body(a, b, dst, done);
        let t = n - done;
        if t != 0 {
            let mut pa = [0u64; 8];
            let mut pb = [0u64; 8];
            pa[..t].copy_from_slice(&a[done..]);
            pb[..t].copy_from_slice(&b[done..]);
            let mut out = [0u64; 8];
            for k in 0..2 {
                // SAFETY: pa/pb/out are [u64; 8]; k in 0..2 so the 256-bit
                // access at u64 index 0 or 4 stays inside 64 bytes.
                let va = _mm256_loadu_si256(pa.as_ptr().add(k * 4).cast());
                let vb = _mm256_loadu_si256(pb.as_ptr().add(k * 4).cast());
                _mm256_storeu_si256(out.as_mut_ptr().add(k * 4).cast(), _mm256_and_si256(va, vb));
            }
            dst[done..].copy_from_slice(&out[..t]);
        }
    }

    /// D — the SAME shared body, then the remainder walked greedily 4 -> 2 -> 1.
    /// Every lane live, no zero-init, no padded copy.
    #[target_feature(enable = "avx2")]
    unsafe fn and_descent(a: &[u64], b: &[u64], dst: &mut [u64]) {
        let n = a.len();
        let done = n & !7;
        body(a, b, dst, done);
        let mut i = done;
        while n - i >= 4 {
            // SAFETY: loop condition guarantees i + 4 <= n.
            let va = _mm256_loadu_si256(a.as_ptr().add(i).cast());
            let vb = _mm256_loadu_si256(b.as_ptr().add(i).cast());
            _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), _mm256_and_si256(va, vb));
            i += 4;
        }
        while n - i >= 2 {
            // SAFETY: loop condition guarantees i + 2 <= n. SSE2, 128-bit.
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

    fn bench(f: unsafe fn(&[u64], &[u64], &mut [u64]), a: &[u64], b: &[u64], dst: &mut [u64], iters: u32) -> f64 {
        // SAFETY: caller only passes the two avx2 fns above, under the runtime
        // avx2 check in `main`; lengths are equal (asserted in `run`).
        unsafe {
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
    }

    pub unsafe fn run() {
        println!("mask_algebra_tail_probe: mask_and, padded tail vs 4->2->1 descent");
        println!("arch=x86_64 avx2=true (AVX2/SSE2 only — no AVX-512, no VL)\n");
        println!(
            "{:>7}  {:>6}  {:>9} {:>9}  {:>6}  {:>6}  verdict",
            "n", "tail", "P padded", "D descent", "D/P", "D/P'"
        );

        for &n in &[1usize, 3, 7, 8, 9, 15, 16, 31, 64, 256, 1024, 4096] {
            let a: Vec<u64> = (0..n)
                .map(|i| 0xF0F0_5555_AAAA_1111u64 ^ (i as u64).wrapping_mul(0x9E37_79B9))
                .collect();
            let b: Vec<u64> = (0..n)
                .map(|i| 0x0FF0_1234_5678_9ABCu64 ^ (i as u64).wrapping_mul(0x85EB_CA6B))
                .collect();
            let mut dp = vec![0u64; n];
            let mut dd = vec![0u64; n];
            assert_eq!(a.len(), b.len());

            // Gate on equality BEFORE timing: a faster wrong arm is not a win.
            and_padded(&a, &b, &mut dp);
            and_descent(&a, &b, &mut dd);
            assert_eq!(dp, dd, "n={n}: descent must be bit-identical to padded");
            let expect: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x & y).collect();
            assert_eq!(dp, expect, "n={n}: padded arm must equal the scalar reference");

            let iters = if n <= 64 { 2_000_000 } else { 200_000 };

            // BOTH ORDERS. The arm timed second inherits a warm cache from the
            // first, and at n >= 256 the three buffers exceed L1 — so a
            // single-order probe systematically favours whichever arm runs
            // last. Reporting both makes that bias visible instead of letting
            // it masquerade as the effect under test.
            let p1 = bench(and_padded, &a, &b, &mut dp, iters);
            let d1 = bench(and_descent, &a, &b, &mut dd, iters);
            let d2 = bench(and_descent, &a, &b, &mut dd, iters);
            let p2 = bench(and_padded, &a, &b, &mut dp, iters);

            let r1 = d1 / p1;
            let r2 = d2 / p2;
            let spread = (r1 - r2).abs();
            let verdict = if spread > 0.15 {
                "ORDER-DEPENDENT — not a result"
            } else if r1.max(r2) < 0.90 {
                "descent WINS"
            } else if r1.min(r2) > 1.10 {
                "descent LOSES"
            } else {
                "inert"
            };
            println!("{n:>7}  {:>6}  {p1:>9.2} {d1:>9.2}  {r1:>6.3}  {r2:>6.3}  {verdict}", n % 8);
        }
        // ── The isolating measurement: difference-in-differences ──────────
        //
        // The table above cannot be read as a tail result on its own. At
        // `n % 8 == 0` both arms execute the SAME code (shared body, then a
        // `t == 0` branch neither takes), yet they do not time the same — and
        // the both-orders columns rule out cache order as the cause. Something
        // about how the two functions compile differs, so every ratio above
        // carries that unknown alongside the tail.
        //
        // Differencing removes it. For a fixed body size `base` (a multiple of
        // 8), `t(base + k) - t(base)` is the cost of a k-word tail and nothing
        // else: the body work is identical in both terms and cancels, per arm,
        // whatever each arm's body compiled to.
        println!("\n── tail cost ISOLATED: t(base + k) - t(base), body held constant ──");
        println!("{:>6}  {:>3}  {:>10} {:>10}  {:>7}", "base", "k", "P tail ns", "D tail ns", "P/D");
        for &base in &[8usize, 64] {
            let mk = |n: usize, seed: u64| -> Vec<u64> {
                (0..n)
                    .map(|i| seed ^ (i as u64).wrapping_mul(0x9E37_79B9))
                    .collect()
            };
            let a0 = mk(base, 0xF0F0_5555_AAAA_1111);
            let b0 = mk(base, 0x0FF0_1234_5678_9ABC);
            let mut d0 = vec![0u64; base];
            let it = 2_000_000;
            let p0 = bench(and_padded, &a0, &b0, &mut d0, it);
            let q0 = bench(and_descent, &a0, &b0, &mut d0, it);
            for k in 1..8usize {
                let n = base + k;
                let a = mk(n, 0xF0F0_5555_AAAA_1111);
                let b = mk(n, 0x0FF0_1234_5678_9ABC);
                let mut d = vec![0u64; n];
                let pk = bench(and_padded, &a, &b, &mut d, it);
                let qk = bench(and_descent, &a, &b, &mut d, it);
                let pt = pk - p0;
                let dt = qk - q0;
                let ratio = if dt > 0.0 { pt / dt } else { f64::NAN };
                println!("{base:>6}  {k:>3}  {pt:>10.2} {dt:>10.2}  {ratio:>7.2}");
            }
        }
        println!("P/D > 1 means the padded tail costs that many times the descent's.");
        println!("A NEGATIVE D-tail is not an error: it means the descent's tail is BELOW the");
        println!("run-to-run noise of the body it was differenced against — indistinguishable");
        println!("from having no tail at all. The ratio is NaN there because the denominator is");
        println!("noise, and reporting a number would invent precision the measurement lacks.");

        println!("\nns per call (P and D columns are the P-first pass).");
        println!("D/P  = descent/padded with PADDED timed first.");
        println!("D/P' = the same ratio with DESCENT timed first.");
        println!("The two must agree; if they do not, the number is cache order, not the tail.");
        println!("tail = n % 8, the remainder the two arms handle differently.");
    }
}
