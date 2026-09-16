//! `VPTERNLOGQ` tail: pad to zmm, or descend zmm → ymm → xmm?
//!
//! `mask_ternlog` chunks over `U64x8::LANES` = **8 words = 512 rows**, then
//! ends on three `pad_tail`s — zero-fill three 8-word stack arrays, run the
//! full 512-bit op, copy the live prefix back out. Anything under 512 rows has
//! no full chunk at all, so the whole operation IS that padded tail
//! (`ogar-r2il`'s `CallMask` is `[u64; 3]`).
//!
//! The alternative (operator, 2026-09-16): **do not pad — descend.** `VPTERNLOGQ`
//! has ymm and xmm encodings under AVX512VL, so a 1..7-word remainder splits as
//! 4 + 2 + 1 with every lane live, no zero-fill, no prefix copy — and, unlike a
//! scalar peel, **it stays entirely in vector registers**. That distinction is
//! load-bearing: `scripts/codegen-witness.sh` caps GPR logic on lane data at
//! `SLICE_GPR_CAP=6` and records that an exact-length scalar tail measured 16 GPR
//! ops on aarch64 (LLVM unrolled it to 7×(and, orr)), which is what `pad_tail`
//! removed. A vector descent is not that peel and does not reintroduce it.
//!
//! The facade carries `ternlog` on `U64x8` and `U32x16` only — there is no
//! `U64x4`/`U64x2` wrapper — so the split arm calls the intrinsics such a wrapper
//! would hold. Measuring whether they are worth adding is the point of the probe.
//!
//! AVX-512 only, deliberately: v3 is the GitHub/distribution baseline, not a
//! deployment target.
//!
//! # Measured 2026-09-16 — Xeon @ 2.10 GHz, `avx512f=true`, release, 3 runs
//!
//! ## The crux: for a remainder of `t` words, which decomposition?
//!
//! Tail only, no body loop. ns per call.
//!
//! | t | shape | **P** padded zmm | **G** greedy | **X** all-xmm | winner |
//! |---:|---|---:|---:|---:|---|
//! | 1 | `1` | 13.1-13.8 | 1.78-1.90 | 1.49-1.60 | G = X, same instructions |
//! | 2 | `2` | 13.0-13.7 | **1.74-1.94** | 2.57-2.78 | **G** |
//! | 3 | `2+1` | 17.1-18.6 | **2.13-2.36** | 2.82-3.23 | **G** |
//! | 4 | `4` vs `2+2` | 12.8-13.6 | **1.59-1.76** | 2.92-3.00 | **G** |
//! | 5 | `4+1` vs `2+2+1` | 17.3-18.3 | **2.08-2.13** | 3.23-3.49 | **G** |
//! | **6** | **`4+2`** vs `2+2+2` | 17.0-18.3 | **2.26-2.74** | 3.29-3.68 | **G** |
//! | 7 | `4+2+1` vs `2+2+2+1` | 16.8-18.5 | **2.38-2.57** | 3.68-3.95 | **G** |
//!
//! **Greedy widest-first wins at every `t >= 2`** — 1.3-1.6x over all-xmm and
//! **5-8x over padding**. The `t = 6` case: `4+2` at 2.26-2.74 ns against
//! `2+2+2` at 3.29-3.68 ns — one ymm plus one xmm beats three xmm. Fewer wider
//! ops win; three 128-bit ops do not pay for avoiding one 256-bit one.
//!
//! At `t = 1` G and X compile to the SAME instructions (neither the ymm nor the
//! pair rung is reachable), so the 0.3 ns gap is layout noise. Recorded as a tie
//! rather than a ranking — a "winner" there is an artifact of the comparison.
//!
//! ## End to end, through the real chunk loop
//!
//! | words | rows | tail | padded | descend | pad/descend |
//! |---:|---:|---:|---:|---:|---:|
//! | 3 | 192 | 3 | 18.1 | 2.3 | **7.7-8.0x** |
//! | 1-7 | 64-448 | 1-7 | 13-23 | 2.0-2.6 | **5.9-8.0x** |
//! | 9 | 576 | 1 | 14.2 | 3.7 | **4.2x** |
//! | 11 | 704 | 3 | 21.9 | 3.6 | **5.4-5.6x** |
//! | 31 | 1984 | 7 | 23.6 | 6.8 | **2.5-3.3x** |
//! | 194 | 12416 | 2 | 41.4 | 26.5 | 1.3-1.4x |
//! | 8 / 16 / 24 / 64 | - | **none** | - | - | 1.0-1.4x |
//!
//! `ogar-r2il`'s `CallMask` is `[u64; 3]` — three words, zero full chunks — so
//! it sits at the 7.7-8.0x row: 18 ns to AND three words, because three 8-word
//! stack arrays are zeroed and partly filled to compute 192 live rows inside a
//! 512-row register.
//!
//! **What is NOT the tail:** the no-tail rows still show 1.0-1.4x. That is loop
//! shape (`as_chunks` + zip + `from_array`/`to_array` round-trips vs direct
//! `loadu`/`storeu`), so the tail-attributable factor is ~5-7x at small word
//! counts, not the full 8x. Subtract it before quoting these.
//!
//! ## The emitted code, checked rather than claimed
//!
//! `--emit=asm` on this probe: **33 zmm + 3 ymm + 6 xmm `vpternlogq`**, several
//! with folded memory operands (`vpternlogq $128, (%r13,%r10,8), %ymm0, %ymm1`),
//! and **zero** GPR `and`/`or`/`xor` on lane data. That is what separates a
//! descent from a scalar peel: `scripts/codegen-witness.sh` caps GPR logic at
//! `SLICE_GPR_CAP=6` and records an exact-length scalar tail measuring 16 GPR ops
//! on aarch64 — the thing `pad_tail` was introduced to remove. A ymm/xmm descent
//! is not that peel and does not reintroduce it.
//!
//! ```text
//! env -u RUSTFLAGS cargo --config .cargo/config-v4.toml run --release \
//!     --example ternlogq_tail_descent_probe
//! ```

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::hint::black_box;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::time::Instant;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use ndarray::simd::ternlog::AND3;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use ndarray::simd::U64x8;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f,avx512vl")]
/// **P** — pad the remainder into one zmm (today's shape).
unsafe fn tail_padded(a: &[u64], b: &[u64], c: &[u64], dst: &mut [u64]) {
    let t = dst.len();
    let (mut pa, mut pb, mut pc) = ([0u64; 8], [0u64; 8], [0u64; 8]);
    pa[..t].copy_from_slice(a);
    pb[..t].copy_from_slice(b);
    pc[..t].copy_from_slice(c);
    let v = U64x8::from_array(pa)
        .ternlog::<AND3>(U64x8::from_array(pb), U64x8::from_array(pc))
        .to_array();
    dst.copy_from_slice(&v[..t]);
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f,avx512vl")]
/// **G** — greedy widest-first: ymm, then xmm, then xmm-low.
unsafe fn tail_greedy(a: &[u64], b: &[u64], c: &[u64], dst: &mut [u64]) {
    let (t, mut i) = (dst.len(), 0usize);
    unsafe {
        if t - i >= 4 {
            let v = _mm256_ternarylogic_epi64::<AND3>(
                _mm256_loadu_si256(a.as_ptr().add(i).cast()),
                _mm256_loadu_si256(b.as_ptr().add(i).cast()),
                _mm256_loadu_si256(c.as_ptr().add(i).cast()),
            );
            _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), v);
            i += 4;
        }
        while t - i >= 2 {
            let v = _mm_ternarylogic_epi64::<AND3>(
                _mm_loadu_si128(a.as_ptr().add(i).cast()),
                _mm_loadu_si128(b.as_ptr().add(i).cast()),
                _mm_loadu_si128(c.as_ptr().add(i).cast()),
            );
            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), v);
            i += 2;
        }
        if t - i == 1 {
            let v = _mm_ternarylogic_epi64::<AND3>(
                _mm_loadl_epi64(a.as_ptr().add(i).cast()),
                _mm_loadl_epi64(b.as_ptr().add(i).cast()),
                _mm_loadl_epi64(c.as_ptr().add(i).cast()),
            );
            _mm_storel_epi64(dst.as_mut_ptr().add(i).cast(), v);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f,avx512vl")]
/// **X** — never widen past xmm: pairs, then the odd one.
unsafe fn tail_all_xmm(a: &[u64], b: &[u64], c: &[u64], dst: &mut [u64]) {
    let (t, mut i) = (dst.len(), 0usize);
    unsafe {
        while t - i >= 2 {
            let v = _mm_ternarylogic_epi64::<AND3>(
                _mm_loadu_si128(a.as_ptr().add(i).cast()),
                _mm_loadu_si128(b.as_ptr().add(i).cast()),
                _mm_loadu_si128(c.as_ptr().add(i).cast()),
            );
            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), v);
            i += 2;
        }
        if t - i == 1 {
            let v = _mm_ternarylogic_epi64::<AND3>(
                _mm_loadl_epi64(a.as_ptr().add(i).cast()),
                _mm_loadl_epi64(b.as_ptr().add(i).cast()),
                _mm_loadl_epi64(c.as_ptr().add(i).cast()),
            );
            _mm_storel_epi64(dst.as_mut_ptr().add(i).cast(), v);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
/// Today's shape: three zero-filled 8-word arrays, one zmm op, prefix copy out.
fn tern_padded(a: &[u64], b: &[u64], c: &[u64], dst: &mut [u64]) {
    const L: usize = U64x8::LANES;
    let (ca, ta) = a.as_chunks::<L>();
    let (cb, tb) = b.as_chunks::<L>();
    let (cc, tc) = c.as_chunks::<L>();
    let (cd, td) = dst.as_chunks_mut::<L>();
    for (((x, y), z), d) in ca.iter().zip(cb).zip(cc).zip(cd.iter_mut()) {
        *d = U64x8::from_array(*x)
            .ternlog::<AND3>(U64x8::from_array(*y), U64x8::from_array(*z))
            .to_array();
    }
    if !ta.is_empty() {
        let (mut pa, mut pb, mut pc) = ([0u64; L], [0u64; L], [0u64; L]);
        pa[..ta.len()].copy_from_slice(ta);
        pb[..tb.len()].copy_from_slice(tb);
        pc[..tc.len()].copy_from_slice(tc);
        let v = U64x8::from_array(pa)
            .ternlog::<AND3>(U64x8::from_array(pb), U64x8::from_array(pc))
            .to_array();
        td.copy_from_slice(&v[..td.len()]);
    }
}

/// zmm for the body, then ymm → xmm → xmm-low for the remainder. Every lane
/// live; no zero-fill, no prefix copy, and no GPR logic on lane data.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[target_feature(enable = "avx512f,avx512vl")]
unsafe fn tern_descend(a: &[u64], b: &[u64], c: &[u64], dst: &mut [u64]) {
    const L: usize = U64x8::LANES;
    let n = dst.len();
    let full = n / L;
    for g in 0..full {
        let (i, p) = (g * L, dst.as_mut_ptr());
        unsafe {
            let va = _mm512_loadu_si512(a.as_ptr().add(i).cast());
            let vb = _mm512_loadu_si512(b.as_ptr().add(i).cast());
            let vc = _mm512_loadu_si512(c.as_ptr().add(i).cast());
            _mm512_storeu_si512(p.add(i).cast(), _mm512_ternarylogic_epi64::<AND3>(va, vb, vc));
        }
    }
    let mut i = full * L;
    unsafe {
        // ymm rung — 4 live words.
        if n - i >= 4 {
            let va = _mm256_loadu_si256(a.as_ptr().add(i).cast());
            let vb = _mm256_loadu_si256(b.as_ptr().add(i).cast());
            let vc = _mm256_loadu_si256(c.as_ptr().add(i).cast());
            _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), _mm256_ternarylogic_epi64::<AND3>(va, vb, vc));
            i += 4;
        }
        // xmm rung — 2 live words.
        if n - i >= 2 {
            let va = _mm_loadu_si128(a.as_ptr().add(i).cast());
            let vb = _mm_loadu_si128(b.as_ptr().add(i).cast());
            let vc = _mm_loadu_si128(c.as_ptr().add(i).cast());
            _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), _mm_ternarylogic_epi64::<AND3>(va, vb, vc));
            i += 2;
        }
        // xmm-low rung — the final single word, still vector (64-bit load/store,
        // so it never reads past the slice).
        if n - i == 1 {
            let va = _mm_loadl_epi64(a.as_ptr().add(i).cast());
            let vb = _mm_loadl_epi64(b.as_ptr().add(i).cast());
            let vc = _mm_loadl_epi64(c.as_ptr().add(i).cast());
            _mm_storel_epi64(dst.as_mut_ptr().add(i).cast(), _mm_ternarylogic_epi64::<AND3>(va, vb, vc));
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
fn main() {
    // The descent rungs are `VPTERNLOGQ` ymm/xmm under AVX512VL, so this probe
    // has nothing to measure off v4 x86_64. It still has to BUILD on the
    // matrix's aarch64 / wasm32 / v3 rows, hence a running no-op rather than a
    // `compile_error!` or a constant assert (which clippy rejects anyway).
    println!("skipped: needs x86_64 + avx512f — run under .cargo/config-v4.toml");
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
fn main() {
    println!("realization: avx512f=true\n");

    for words in 1..=40usize {
        let a: Vec<u64> = (0..words)
            .map(|i| 0xF0F0_F0F0_0000_1111u64 ^ (i as u64 * 31))
            .collect();
        let b: Vec<u64> = (0..words)
            .map(|i| 0xFFFF_0000_FFFF_0000u64 ^ (i as u64 * 17))
            .collect();
        let c: Vec<u64> = (0..words)
            .map(|i| 0x0F0F_0F0F_1111_0000u64 ^ (i as u64 * 7))
            .collect();
        let (mut d1, mut d2) = (vec![0u64; words], vec![0u64; words]);
        tern_padded(&a, &b, &c, &mut d1);
        // SAFETY: avx512f+avx512vl asserted above; every load/store is within
        // the slices (the final rung uses a 64-bit load, not a 128-bit one).
        unsafe { tern_descend(&a, &b, &c, &mut d2) };
        assert_eq!(d1, d2, "arms disagree at words={words}");
    }
    println!("equivalence: 1..=40 words, bit-identical\n");

    macro_rules! bench {
        ($body:expr, $reps:expr) => {{
            $body;
            let t0 = Instant::now();
            for _ in 0..$reps {
                $body;
            }
            t0.elapsed().as_secs_f64() / ($reps as f64) * 1e9
        }};
    }

    // ---- the crux: for a tail of t words, which decomposition wins? ----
    // t = 6 is the interesting one: 1 ymm + 1 xmm, or 3 xmm, or one padded zmm?
    println!("== TAIL-ONLY, per remainder length: which split? (ns per call) ==");
    println!("{:>5}  {:>10} {:>10} {:>10}   {}", "t", "P padded", "G greedy", "X all-xmm", "winner");
    for t in 1..=7usize {
        let a: Vec<u64> = (0..t).map(|i| 0xF0F0_1111u64 ^ i as u64).collect();
        let b: Vec<u64> = (0..t).map(|i| 0xFF00_2222u64 ^ i as u64).collect();
        let c: Vec<u64> = (0..t).map(|i| 0x0F0F_4444u64 ^ i as u64).collect();
        let (mut dp, mut dg, mut dx) = (vec![0u64; t], vec![0u64; t], vec![0u64; t]);
        // SAFETY: avx512f+avx512vl hold on this arm; every access is in-slice.
        unsafe {
            tail_padded(&a, &b, &c, &mut dp);
            tail_greedy(&a, &b, &c, &mut dg);
            tail_all_xmm(&a, &b, &c, &mut dx);
        }
        assert_eq!(dp, dg, "greedy disagrees at t={t}");
        assert_eq!(dp, dx, "all-xmm disagrees at t={t}");
        let reps = 3_000_000;
        // SAFETY: as above.
        let p = bench!(
            {
                unsafe { tail_padded(black_box(&a), black_box(&b), black_box(&c), &mut dp) };
                black_box(&dp);
            },
            reps
        );
        let g = bench!(
            {
                unsafe { tail_greedy(black_box(&a), black_box(&b), black_box(&c), &mut dg) };
                black_box(&dg);
            },
            reps
        );
        let x = bench!(
            {
                unsafe { tail_all_xmm(black_box(&a), black_box(&b), black_box(&c), &mut dx) };
                black_box(&dx);
            },
            reps
        );
        // At t == 1 the G and X bodies are the SAME instructions (neither the
        // ymm nor the pair rung is reachable, both fall to xmm-low), so a gap
        // there is layout noise and a "winner" would be an artifact of this
        // comparison rather than a strategy difference. Say so, do not rank it.
        let best = if t == 1 {
            "G=X (identical code)"
        } else if p <= g && p <= x {
            "P"
        } else if g <= x {
            "G"
        } else {
            "X"
        };
        let shape = match t {
            1 => "1",
            2 => "2",
            3 => "2+1",
            4 => "4 | 2+2",
            5 => "4+1 | 2+2+1",
            6 => "4+2 | 2+2+2",
            _ => "4+2+1 | 2+2+2+1",
        };
        println!("{:>5}  {:>10.2} {:>10.2} {:>10.2}   {}   [{}]", t, p, g, x, best, shape);
    }
    println!();

    println!("{:>6} {:>7} {:>5} {:>11} {:>11} {:>10}", "words", "rows", "tail", "padded", "descend", "pad/desc");
    for &words in &[1usize, 2, 3, 4, 5, 6, 7, 8, 9, 11, 16, 24, 31, 64, 194] {
        let a: Vec<u64> = (0..words).map(|i| 0xF0F0u64 ^ i as u64).collect();
        let b: Vec<u64> = (0..words).map(|i| 0xFF00u64 ^ i as u64).collect();
        let c: Vec<u64> = (0..words).map(|i| 0x0F0Fu64 ^ i as u64).collect();
        let (mut d1, mut d2) = (vec![0u64; words], vec![0u64; words]);
        let reps = (3_000_000 / words.max(1)).max(2000);
        let p = bench!(
            {
                tern_padded(black_box(&a), black_box(&b), black_box(&c), &mut d1);
                black_box(&d1);
            },
            reps
        );
        let s = bench!(
            {
                // SAFETY: as above.
                unsafe { tern_descend(black_box(&a), black_box(&b), black_box(&c), &mut d2) };
                black_box(&d2);
            },
            reps
        );
        println!("{:>6} {:>7} {:>5} {:>11.2} {:>11.2} {:>10.2}", words, words * 64, words % 8, p, s, p / s);
    }
}
