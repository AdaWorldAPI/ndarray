//! Does vectorizing r2sleigh's `OpColumns` scans pay, and where is the crossover?
//!
//! `r2il::columns` was laid out for `ndarray::simd`'s mask surface and says so
//! in its own module docs — while deliberately taking no `ndarray` dependency
//! and declining to claim the SIMD is worth it: *"whether that is worth doing
//! is a profiling question nobody has answered."* This is the consumer-side
//! answer, on a real lift rather than a synthetic stream.
//!
//! Input is a column dump from `r2sleigh-lift`'s `win32_census` example
//! (`WIN32_CENSUS_COLUMNS_OUT=<path>`): `u64 n`, `n×u8` tag, `n×u8` space,
//! `n×u64` offset. Run:
//!
//! ```text
//! cargo run --release --example r2il_column_scan_probe -- <dump>
//! ```
//!
//! # The two queries
//!
//! **Q1 — `find_tag`**: `tag == Call`. One predicate, the simplest scan.
//!
//! **Q2 — `find_ram_in_range`**: the census's IAT/prefetch classification,
//! `space == Ram && lo <= offset < hi`. Four expressible predicates once the
//! u64 offset is split (see below), so it is the conjunction question.
//!
//! # Two primitive gaps this probe reported — both now CLOSED by ndarray #309
//!
//! Stated as the FIRST run found them, because the widened arms below still
//! exist and still need their rationale. What changed: `{eq,…}_u8_to_mask` and
//! `{ge,lt,…}_u64_to_mask` now ship, so the `NATIVE` arm spells the query
//! directly and the two workarounds are kept only as the comparison.
//!
//! 1. **No `u8` comparator.** `OpColumns::{tag,space}` are `Vec<u8>`; the
//!    facade's narrowest value type is `u32`. A consumer must keep a widened
//!    copy — 4× the bytes of the column it scans, which directly attacks the
//!    memory-traffic argument that motivates the layout.
//! 2. **No `u64` RANGE comparator.** The facade has `ternary_match_u64_to_mask`
//!    (exact, with don't-care bits) but no `ge/lt` over `u64`; the ordered
//!    family stops at `i32`. Measured on the real dump, **100 % of Ram-space
//!    offsets exceed 2³²** (they are image-based: `0x1_4000_105e` …
//!    `0x1_4000_8398`), so narrowing is not sound in general. The query is
//!    re-expressed EXACTLY here by splitting the offset into `hi32`/`lo32`
//!    columns — valid only because the chosen window lies inside one `hi32`
//!    bucket, which the probe asserts rather than assumes. A general range
//!    needed the primitive — `ge_u64_to_mask` / `lt_u64_to_mask` are it, and
//!    the `NATIVE` arm uses them with no split and no bucket assumption.
//!
//! # Arms (every arm must produce a bit-identical mask, or the run aborts)
//!
//! | arm | how the conjunction is formed | passes |
//! |---|---|---|
//! | `S` | scalar over the columns' own `u8`/`u64` types (no widening at all) | 1 |
//! | `AND` | 4 `*_to_mask` over WIDENED columns + 3 `mask_and_assign` | 7 |
//! | `TERN` | 4 `*_to_mask` over WIDENED columns + `mask_ternlog::<AND3>` + `mask_and_assign` | 6 |
//! | `UNDER` | the `_under` chain over WIDENED columns — each predicate narrows the live mask | 4 |
//! | `NATIVE` | 3 `*_to_mask` over the columns' OWN types (`eq_u8` + `ge_u64` + `lt_u64`) + one `mask_ternlog::<AND3>` | 4 |
//!
//! `NATIVE` is the arm ndarray #309 made possible: three predicates instead of
//! four (the `hi32 == 1` term disappears with the split), zero widened columns,
//! and no assumption that the window lies inside one `hi32` bucket. `S` is the
//! honest baseline precisely because it too needs no widened columns: charging
//! the widened arms for the layout they require is the comparison a consumer
//! actually faces.
//!
//! # Measured — and the two gaps this probe reported are now CLOSED
//!
//! The first run of this probe (2026-09-14) reported two missing primitives:
//! no `u8` comparator and no `u64` RANGE comparator. ndarray #309 shipped both
//! (`{eq,ne,gt,ge,lt,le}_{u8,u64}_to_mask`), so `find_ram_in_range` now has a
//! direct spelling — three predicates over the columns' own types, no widened
//! copies, no hi32/lo32 split, no assumption that the window sits inside one
//! `hi32` bucket. This is the re-run, and it answers the two falsifiers
//! `.claude/knowledge/masking-ops-state.md` had PENDING (G1's second half and
//! G2's only half).
//!
//! ## Two corrections to the first run, both of which changed a number
//!
//! 1. **It measured v3/AVX2 and the report implied AVX-512.** `.cargo/config.toml`
//!    is `x86-64-v3`; the host having `avx512f` says nothing about what was
//!    compiled. Every arm below is now run under BOTH configs and the program
//!    prints its own realization line.
//! 2. **No `black_box`.** ndarray's own G1 figures were published 8.06×/6.75×
//!    and corrected to 6.91×/5.84× for exactly this — asymmetric dead-store
//!    elimination. Inputs AND outputs of every arm are now protected, all or
//!    none.
//!
//! ## Q1 — `find_tag`, one predicate over a `u8` column (12 408 ops, real)
//!
//! | tier | scalar | widened `eq_u32` | NATIVE `eq_u8` |
//! |---|---:|---:|---:|
//! | v4 / AVX-512 | 1× | 3.96× | **34.4×** |
//! | v3 / AVX2 | 1× | 1.84× | **26.1×** |
//!
//! **G1 is decisively answered**: the native primitive is 8.7× (v4) / 14× (v3)
//! better than the widening workaround it replaced. The ~26-34× against scalar
//! is the expected lane count, not an anomaly — 32-64 bytes per instruction
//! against a 1-byte scalar loop; bandwidth confirms it (52 GB/s L1-resident vs
//! 2 GB/s scalar). Do NOT compare this to G1's own 6.91×: that figure is
//! u8-vectorized vs i32-vectorized, a different pair.
//!
//! ## Q2 — `find_ram_in_range`, the `u64` range query (ns/op)
//!
//! | span | S | AND | TERN | UNDER | NATIVE | S/NATIVE |
//! |---:|---:|---:|---:|---:|---:|---:|
//! | **12 408 v4** | 0.73 | 0.46 | 0.45 | 0.64 | **0.48** | **1.51×** |
//! | **12 408 v3** | 0.72 | 0.73 | 0.73 | 0.79 | **1.31** | **0.55×** |
//! | 3 176 448 T v4 | 1.24 | 1.73 | 1.68 | 1.58 | 1.87 | 0.66× |
//! | 3 176 448 T v3 | 1.44 | 2.27 | 2.35 | 1.77 | 2.59 | 0.55× |
//!
//! **G2 is answered, and the answer is TIER-DEPENDENT — the headline finding.**
//! The same native arm is a **1.5× win on AVX-512 and a 0.55× LOSS on AVX2**.
//! That is not noise: only avx512 (`epu64`) and NEON (`cmhi`) have an unsigned
//! ordered 64-bit compare; avx2 and scalar are flat polyfills, so on v3 the
//! "native" path is a scalar loop wearing a vector signature. A consumer on a
//! v3 baseline should keep the hi32/lo32 split; on v4 the native spelling is
//! both faster and general.
//!
//! ## The u8 widening is a 4× tax; the u64 "widening" is a DISCOUNT
//!
//! The first run blamed Q2's crossover on widened columns. Counted per arm,
//! the widened path is not merely no worse — it reads **less**:
//!
//! | arm | reads per logical row | total |
//! |---|---|---:|
//! | `NATIVE` | `space` 1 B + `offset` 8 B (`ge`) + `offset` 8 B (`lt`) | **17 B** |
//! | `AND` / `TERN` | `space32` 4 B + `hi32` 4 B + `lo32` 4 B + `lo32` 4 B | **16 B** |
//!
//! Offset-derived alone it is **16 B native vs 12 B split** — a 25 % saving,
//! because each split predicate touches only the half it needs (`hi32` once,
//! `lo32` twice) while every native `u64` predicate must pull all eight bytes.
//! So the split both halves the element width (what the vector units reward)
//! AND cuts traffic. The `u8` case is the opposite sign: widening `tag`/`space`
//! to `u32` is a real 4× tax with no compensating structure.
//!
//! ⊘ This corrects the first published version of this section, which claimed
//! `hi32`+`lo32` came to "also 16 B/op" and concluded the Q2 arms paid no tax
//! at all. `hi32` is read ONCE, not twice — 12 B, not 16. The **direction**
//! (widened beats native on v3, matches on v4) is unchanged and is what the
//! measured timings show; the equal-traffic mechanism offered for it was
//! wrong. The consumer guidance below follows the timings and the native
//! path's generality, never that retired claim.
//!
//! ## The crossover survives both corrections
//!
//! Every arm degrades to ≤ 1.0× above ~200 K ops on both tiers. And the
//! absolute stakes are unchanged: a whole-census `find_ram_in_range` is ~9.0 µs
//! scalar and ~5.9 µs native-on-v4 — **3 µs saved** on a binary whose SLEIGH
//! lift costs milliseconds. Per the workspace rule *a word-level op pays for
//! the span it is given*, this span still does not pay. What changed is the
//! primitive surface, not the verdict for this consumer.

use std::hint::black_box;
use std::time::Instant;

use ndarray::simd::ternlog::AND3;
use ndarray::simd::{
    eq_i32_to_mask_under, eq_u32_to_mask, eq_u8_to_mask, ge_i32_to_mask_under, ge_u64_to_mask, lt_i32_to_mask,
    lt_i32_to_mask_under, lt_u64_to_mask, mask_and_assign, mask_ternlog, popcount_batch_u64,
};

const RAM: u8 = 3;
const TAG_CALL: u8 = 4;
/// Window: `[image_base, image_base + 0x2000)` for this fixture's 0x1_4000_0000
/// base — the `.text` head, the shape the census's IAT walk asks about.
const WIN_LO: u64 = 0x1_4000_0000;
const WIN_HI: u64 = 0x1_4000_2000;

fn words(n: usize) -> usize {
    n.div_ceil(64)
}

struct Cols {
    tag: Vec<u8>,
    space: Vec<u8>,
    offset: Vec<u64>,
}

fn load(path: &str) -> Cols {
    let b = std::fs::read(path).expect("read column dump");
    let n = u64::from_le_bytes(b[0..8].try_into().unwrap()) as usize;
    let tag = b[8..8 + n].to_vec();
    let space = b[8 + n..8 + 2 * n].to_vec();
    let mut offset = Vec::with_capacity(n);
    for i in 0..n {
        let o = 8 + 2 * n + i * 8;
        offset.push(u64::from_le_bytes(b[o..o + 8].try_into().unwrap()));
    }
    Cols { tag, space, offset }
}

/// Tile the real stream to reach spans a single 130 KB binary cannot supply.
/// Labelled TILED in the output — it preserves the measured distribution but
/// is not more evidence about program shape, only about throughput.
fn take(c: &Cols, n: usize) -> Cols {
    let m = c.tag.len();
    Cols {
        tag: (0..n).map(|i| c.tag[i % m]).collect(),
        space: (0..n).map(|i| c.space[i % m]).collect(),
        offset: (0..n).map(|i| c.offset[i % m]).collect(),
    }
}

// ---- scalar reference arms (native column types, no widening) ----

fn q1_scalar(c: &Cols, out: &mut [u64]) {
    out.fill(0);
    for (i, &t) in c.tag.iter().enumerate() {
        if t == TAG_CALL {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
}

// Spelled exactly as `r2il::OpColumns::find_ram_in_range` spells it — this arm
// is the baseline BECAUSE it mirrors the shipped scalar, so it is not rewritten
// into `Range::contains` for the lint's sake.
#[expect(clippy::manual_range_contains)]
fn q2_scalar(c: &Cols, out: &mut [u64]) {
    out.fill(0);
    for i in 0..c.tag.len() {
        let o = c.offset[i];
        if c.space[i] == RAM && o >= WIN_LO && o < WIN_HI {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: <column-dump>");
    let base = load(&path);
    let real_n = base.tag.len();
    println!("dump: {real_n} ops from {path}");
    // Which realization actually ran. `.cargo/config.toml` is v3/AVX2; AVX-512
    // needs `env -u RUSTFLAGS cargo --config .cargo/config-v4.toml`, and a
    // RUSTFLAGS env var silently REPLACES the config's rustflags. A timing
    // without its tier is an anecdote, so the arm prints its own.
    println!(
        "realization: avx512f={} avx2={} neon={}\n",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "neon"),
    );

    // Gap 2's soundness precondition, asserted rather than assumed: the window
    // must lie inside one hi32 bucket for the split re-expression to be exact.
    assert_eq!(
        WIN_LO >> 32,
        (WIN_HI - 1) >> 32,
        "window straddles a hi32 boundary; the split decomposition is not exact"
    );

    println!(
        "{:<10} {:>9}  {:>10} {:>10} {:>10} {:>10} {:>10}   {:>8} {:>8} {:>9}",
        "span", "hits", "S ns/op", "AND", "TERN", "UNDER", "NATIVE", "S/AND", "S/TERN", "S/NATIVE"
    );

    let mut spans: Vec<(usize, bool)> = vec![256, 1024, 4096]
        .into_iter()
        .map(|n| (n, false))
        .collect();
    spans.push((real_n, false));
    for k in [4usize, 16, 64, 256] {
        spans.push((real_n * k, true));
    }

    for (n, tiled) in spans {
        let c = if n == real_n { load(&path) } else { take(&base, n) };
        let w = words(n);

        // The widened columns the facade's value types force on a consumer.
        let space32: Vec<u32> = c.space.iter().map(|&s| s as u32).collect();
        let tag32: Vec<u32> = c.tag.iter().map(|&t| t as u32).collect();
        let hi32: Vec<i32> = c.offset.iter().map(|&o| (o >> 32) as i32).collect();
        let lo32: Vec<i32> = c.offset.iter().map(|&o| (o as u32) as i32).collect();
        let want_hi = (WIN_LO >> 32) as i32;
        let lo_lo = (WIN_LO as u32) as i32;
        let lo_hi = (WIN_HI as u32) as i32;
        assert!(lo_lo >= 0 && lo_hi > lo_lo, "window lo32 must stay positive");

        let (mut s, mut a, mut t, mut u) = (vec![0u64; w], vec![0u64; w], vec![0u64; w], vec![0u64; w]);
        let (mut m0, mut m1, mut m2, mut m3) = (vec![0u64; w], vec![0u64; w], vec![0u64; w], vec![0u64; w]);

        let build = |m0: &mut [u64], m1: &mut [u64], m2: &mut [u64], m3: &mut [u64]| {
            eq_u32_to_mask(black_box(&space32), black_box(RAM as u32), m0);
            ndarray::simd::eq_i32_to_mask(black_box(&hi32), black_box(want_hi), m1);
            ndarray::simd::ge_i32_to_mask(black_box(&lo32), black_box(lo_lo), m2);
            lt_i32_to_mask(black_box(&lo32), black_box(lo_hi), m3);
        };

        let and_arm = |a: &mut Vec<u64>, m0: &mut Vec<u64>, m1: &mut Vec<u64>, m2: &mut Vec<u64>, m3: &mut Vec<u64>| {
            build(m0, m1, m2, m3);
            a.copy_from_slice(m0);
            mask_and_assign(a, m1);
            mask_and_assign(a, m2);
            mask_and_assign(a, m3);
        };
        let tern_arm =
            |t: &mut Vec<u64>, m0: &mut Vec<u64>, m1: &mut Vec<u64>, m2: &mut Vec<u64>, m3: &mut Vec<u64>| {
                build(m0, m1, m2, m3);
                mask_ternlog::<AND3>(m0, m1, m2, t);
                mask_and_assign(t, m3);
            };
        // The NATIVE arm — the two gaps this probe reported are now closed
        // (`eq_u8_to_mask` / `ge_u64_to_mask` / `lt_u64_to_mask`, ndarray #309),
        // so the query has a direct, general spelling: THREE predicates over the
        // columns' own types, no widened copies, no hi32/lo32 split, and no
        // assumption that the window lies inside one hi32 bucket.
        let native_arm = |n0: &mut Vec<u64>, m0: &mut Vec<u64>, m1: &mut Vec<u64>, out: &mut Vec<u64>| {
            eq_u8_to_mask(black_box(&c.space), black_box(RAM), n0);
            ge_u64_to_mask(black_box(&c.offset), black_box(WIN_LO), m0);
            lt_u64_to_mask(black_box(&c.offset), black_box(WIN_HI), m1);
            mask_ternlog::<AND3>(n0, m0, m1, out);
        };

        let under_arm = |u: &mut Vec<u64>, m0: &mut Vec<u64>, m1: &mut Vec<u64>| {
            eq_u32_to_mask(black_box(&space32), black_box(RAM as u32), u);
            eq_i32_to_mask_under(&hi32, want_hi, u, m0);
            ge_i32_to_mask_under(&lo32, lo_lo, m0, m1);
            lt_i32_to_mask_under(&lo32, lo_hi, m1, u);
        };

        // ---- correctness gate: every arm, bit for bit, before any timing ----
        q2_scalar(&c, &mut s);
        and_arm(&mut a, &mut m0, &mut m1, &mut m2, &mut m3);
        tern_arm(&mut t, &mut m0, &mut m1, &mut m2, &mut m3);
        under_arm(&mut u, &mut m0, &mut m1);
        let mut nat = vec![0u64; w];
        let mut nscratch = vec![0u64; w];
        native_arm(&mut nscratch, &mut m0, &mut m1, &mut nat);
        assert_eq!(s, nat, "NATIVE arm disagrees at n={n}");
        assert_eq!(s, a, "AND arm disagrees at n={n}");
        assert_eq!(s, t, "TERN arm disagrees at n={n}");
        assert_eq!(s, u, "UNDER arm disagrees at n={n}");
        let hits: u64 = popcount_batch_u64(&s);

        // ---- timing ----
        let reps = (40_000_000 / n).max(20);
        macro_rules! bench {
            ($body:expr) => {{
                $body;
                let t0 = Instant::now();
                for _ in 0..reps {
                    $body;
                }
                t0.elapsed().as_secs_f64() / (reps as f64) / (n as f64) * 1e9
            }};
        }

        // `black_box` on every arm's OUTPUT, not just some of them. The
        // asymmetry is the trap: ndarray's own G1 figures were published 8.06x
        // / 6.75x and corrected to 6.91x / 5.84x because one family's stores
        // were read later and the other's were eliminated. Protect all or none.
        let ns_s = bench!({
            q2_scalar(black_box(&c), &mut s);
            black_box(&s);
        });
        let ns_a = bench!({
            and_arm(&mut a, &mut m0, &mut m1, &mut m2, &mut m3);
            black_box(&a);
        });
        let ns_t = bench!({
            tern_arm(&mut t, &mut m0, &mut m1, &mut m2, &mut m3);
            black_box(&t);
        });
        let ns_u = bench!({
            under_arm(&mut u, &mut m0, &mut m1);
            black_box(&u);
        });
        let ns_n = bench!({
            native_arm(&mut nscratch, &mut m0, &mut m1, &mut nat);
            black_box(&nat);
        });

        println!(
            "{:<10} {:>9}  {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}   {:>8.2} {:>8.2} {:>9.2}",
            format!("{}{}", n, if tiled { "T" } else { "" }),
            hits,
            ns_s,
            ns_a,
            ns_t,
            ns_u,
            ns_n,
            ns_s / ns_a,
            ns_s / ns_t,
            ns_s / ns_n
        );

        // Q1, the single-predicate scan, at the same span.
        let mut q1s = vec![0u64; w];
        let mut q1v = vec![0u64; w];
        q1_scalar(&c, &mut q1s);
        eq_u32_to_mask(&tag32, TAG_CALL as u32, &mut q1v);
        assert_eq!(q1s, q1v, "Q1 arms disagree at n={n}");
        let q1_hits = popcount_batch_u64(&q1s);
        let mut q1n = vec![0u64; w];
        eq_u8_to_mask(&c.tag, TAG_CALL, &mut q1n);
        assert_eq!(q1s, q1n, "Q1 native arm disagrees at n={n}");
        let q1_ns_s = bench!({
            q1_scalar(black_box(&c), &mut q1s);
            black_box(&q1s);
        });
        let q1_ns_v = bench!({
            eq_u32_to_mask(black_box(&tag32), black_box(TAG_CALL as u32), &mut q1v);
            black_box(&q1v);
        });
        let q1_ns_n = bench!({
            eq_u8_to_mask(black_box(&c.tag), black_box(TAG_CALL), &mut q1n);
            black_box(&q1n);
        });
        println!(
            "{:<10} {:>9}  {:>10.4} {:>10.4} {:>10.4}              <- Q1: scalar / widened eq_u32 ({:.2}x) / NATIVE eq_u8 ({:.2}x)",
            "",
            q1_hits,
            q1_ns_s,
            q1_ns_v,
            q1_ns_n,
            q1_ns_s / q1_ns_v,
            q1_ns_s / q1_ns_n
        );
    }
}
