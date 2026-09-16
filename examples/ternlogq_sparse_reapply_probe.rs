//! Sparse frontier: re-apply the `VPTERNLOGQ` chain per chunk, or walk the bits?
//!
//! Follow-up to `ternlogq_tail_descent_probe` (the TAIL) for the other place a
//! 64×2 rung could matter: a **sparse frontier over a full-width mask**. The
//! MQ cost model (`hex_tenant_mq_probe`, plan §14) is `step = x·ternlogq + n`:
//! ternlogq = one full pass over every word (291 ns at 1 024 words), `n` = the
//! per-active-bit walk (17.3 µs). `lgj_hop` has the same two halves — three
//! full mask passes, then a `trailing_zeros` scatter over the selected set.
//!
//! The question (operator, 2026-09-16): *"would the gather vs re-apply be
//! faster with 64×2 instead of 64×8?"* At 1 024 words there is no tail, so the
//! descent can only pay if the re-apply is **gated per chunk on a non-empty
//! source** — then the chunk width sets how many dead rows each live bit drags
//! through the ALU: 512 (zmm), 256 (ymm), 128 (xmm), 64 (a GPR word). Against
//! that stands the per-bit gather, O(frontier) and blind to the mask geometry.
//!
//! Arms, all computing `dst = src ∧ gate ∧ elig` (AND3) and gated bit-identical:
//!
//! | arm | shape |
//! |---|---|
//! | `F8` | the shipped full pass, `mask_ternlog` over `U64x8` chunks |
//! | `S8` | zmm chunks, skip when the 8-word src chunk is all zero (`vptestmq` → kortest) |
//! | `S4` | ymm chunks, same skip at 4 words |
//! | `S2` | xmm chunks, same skip at 2 words |
//! | `S1` | one word: `if s != 0 { s & g & e }` — GPR, the shape `codegen-witness.sh` caps; here a FLOOR, not a candidate |
//! | `W`  | the gather: `trailing_zeros` walk of src, per bit one 8-byte payload read + one dst bit — O(frontier) |
//!
//! Two frontier shapes at each density, because a BFS frontier on a Morton
//! lattice is CLUSTERED, not uniform: `uniform` scatters the live bits over
//! all 1 024 words; `clustered` packs them into a contiguous run of words.
//!
//! # Measured 2026-09-16 — Xeon @ 2.10 GHz, `avx512f=true avx512vl=true`, release, 3 runs
//!
//! ns per call, 1 024 words (65 536 rows), operands ~62 % dense, best of 3 × 30 ms.
//!
//! | frontier | shape | live | F8 | S8 | S4 | S2 | S1 (gpr) | W (gather) |
//! |---|---|---:|---:|---:|---:|---:|---:|---:|
//! | 0.01 % | uniform | 7 | 143–164 | **121–141** | 148–168 | 239–277 | 192–233 | 404–435 |
//! | 0.01 % | clustered | 7 | 148–163 | **125–136** | 144–145 | 240–242 | 223 | 358–368 |
//! | 0.1 % | clustered | 66 | 127–161 | **103–132** | 137–145 | 227–240 | 198–223 | 394–421 |
//! | 1 % | uniform | 655 | **153–162** | 190–197 | 216–222 | 277–289 | 198–224 | 918–1013 |
//! | 1 % | clustered | 655 | 152–162 | **123–132** | 129–146 | 190–240 | 223–227 | 933–981 |
//! | 10 % | uniform | 6 554 | **153–162** | 188–200 | 213–222 | 314–334 | 216–224 | 9 987–10 849 |
//! | 10 % | clustered | 6 554 | 152–161 | **135–141** | 144–157 | 239–258 | 193–222 | 6 187–6 926 |
//! | 100 % | either | 65 536 | **145–161** | 172–205 | 194–222 | 313–339 | 201–229 | 52 351–58 329 |
//!
//! **Answer: no — 64×2 is not the faster re-apply at any density.** `S2` is
//! 1.5–2.1× SLOWER than the full zmm pass everywhere, `S4` never beats `S8`,
//! and the skip itself is worth at most **1.24×** (clustered frontiers; on a
//! uniform frontier at ≥ 1 % the branch mispredicts and `S8` LOSES to `F8`).
//! The full pass is 150 ns for 24 KiB of reads + 8 KiB of writes — L1-resident,
//! already at the bandwidth the chunk shape cannot improve on. Chunk width sets
//! wasted rows per live bit only in principle; at this population the pass is
//! bandwidth, not ALU, so narrower chunks just mean more loop iterations.
//!
//! **The gather never wins here, and the reason is its own loop.** `W` walks
//! all 1 024 source words before it can know they are empty, so it costs ≥ 360 ns
//! even at 7 live bits (2.6–3.5× the mask pass) and grows ~0.8 ns per live bit
//! (payload L1/L2-resident; lgj's 512-byte-strided rows would be ~25 ns/bit,
//! which is the number the operator quoted). A gather that walked an INDEX LIST
//! would be O(bits) and would win below ~0.2 % — and that list is exactly the
//! serialization of the population the mask doctrine forbids (lgj R1); it is not
//! a candidate, and the mask pass is within 1.5× of what it would cost at the
//! sparsest arm anyway.
//!
//! So for a full-width frontier the 64×2 rung is a TAIL instrument only
//! (`ternlogq_tail_descent_probe`: 5–8× on 1..7-word masks); on a 1 024-word
//! mask it is the wrong tool, and the one lever that shows is chunk-skip on a
//! clustered frontier, worth ≤ 1.24×.
//!
//! AVX-512 only (v4 config), gated on `avx512f` AND `avx512vl` AND `avx512dq`
//! (Codex P2 on #311): `S4`/`S2` are VL encodings and `vptestmq` is DQ; a
//! `#[target_feature]` attribute is a caller precondition, not a CPU check, so
//! an F-only target takes the no-op `main`.
//!
//! ```text
//! env -u RUSTFLAGS cargo --config .cargo/config-v4.toml run --release \
//!     --example ternlogq_sparse_reapply_probe
//! ```

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512dq"
))]
mod probe {
    use ndarray::simd::mask_ternlog;
    use ndarray::simd::ternlog::AND3;
    use std::arch::x86_64::*;
    use std::hint::black_box;
    use std::time::Instant;

    pub const ROWS: usize = 65_536;
    pub const WORDS: usize = ROWS / 64;

    fn splitmix(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `F8` — what ships.
    fn f8(src: &[u64], gate: &[u64], elig: &[u64], dst: &mut [u64]) {
        mask_ternlog::<AND3>(src, gate, elig, dst);
    }

    /// `S8` — zmm chunks, skipped when the src chunk is all zero.
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn s8(src: &[u64], gate: &[u64], elig: &[u64], dst: &mut [u64]) {
        let n = src.len();
        let mut i = 0;
        unsafe {
            while i + 8 <= n {
                let s = _mm512_loadu_si512(src.as_ptr().add(i).cast());
                if _mm512_test_epi64_mask(s, s) == 0 {
                    _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), _mm512_setzero_si512());
                } else {
                    let v = _mm512_ternarylogic_epi64::<AND3>(
                        s,
                        _mm512_loadu_si512(gate.as_ptr().add(i).cast()),
                        _mm512_loadu_si512(elig.as_ptr().add(i).cast()),
                    );
                    _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast(), v);
                }
                i += 8;
            }
        }
        debug_assert_eq!(i, n, "probe sizes are multiples of 8 words");
    }

    /// `S4` — ymm chunks, skipped when the 4-word src chunk is all zero.
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn s4(src: &[u64], gate: &[u64], elig: &[u64], dst: &mut [u64]) {
        let n = src.len();
        let mut i = 0;
        unsafe {
            while i + 4 <= n {
                let s = _mm256_loadu_si256(src.as_ptr().add(i).cast());
                if _mm256_test_epi64_mask(s, s) == 0 {
                    _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), _mm256_setzero_si256());
                } else {
                    let v = _mm256_ternarylogic_epi64::<AND3>(
                        s,
                        _mm256_loadu_si256(gate.as_ptr().add(i).cast()),
                        _mm256_loadu_si256(elig.as_ptr().add(i).cast()),
                    );
                    _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast(), v);
                }
                i += 4;
            }
        }
    }

    /// `S2` — xmm chunks, skipped when the 2-word src chunk is all zero.
    #[target_feature(enable = "avx512f,avx512vl")]
    unsafe fn s2(src: &[u64], gate: &[u64], elig: &[u64], dst: &mut [u64]) {
        let n = src.len();
        let mut i = 0;
        unsafe {
            while i + 2 <= n {
                let s = _mm_loadu_si128(src.as_ptr().add(i).cast());
                if _mm_test_epi64_mask(s, s) == 0 {
                    _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), _mm_setzero_si128());
                } else {
                    let v = _mm_ternarylogic_epi64::<AND3>(
                        s,
                        _mm_loadu_si128(gate.as_ptr().add(i).cast()),
                        _mm_loadu_si128(elig.as_ptr().add(i).cast()),
                    );
                    _mm_storeu_si128(dst.as_mut_ptr().add(i).cast(), v);
                }
                i += 2;
            }
        }
    }

    /// `S1` — the GPR floor. Not a candidate (fails the codegen witness); it
    /// bounds what "skip at the finest granularity" can buy.
    fn s1(src: &[u64], gate: &[u64], elig: &[u64], dst: &mut [u64]) {
        for (((&s, &g), &e), d) in src.iter().zip(gate).zip(elig).zip(dst.iter_mut()) {
            *d = if s == 0 { 0 } else { s & g & e };
        }
    }

    /// `W` — the gather. Walk src's set bits; per bit read one 8-byte payload
    /// (the hop's decode read, from a 65 536 × u64 column) and decide from it.
    /// `payload[row] & 1` is prearranged to equal `gate ∧ elig` at that row, so
    /// the walk computes the SAME predicate and gates bit-identical.
    fn w(src: &[u64], payload: &[u64], dst: &mut [u64]) {
        for (wi, (&sw, d)) in src.iter().zip(dst.iter_mut()).enumerate() {
            let mut bits = sw;
            let mut out = 0u64;
            while bits != 0 {
                let b = bits.trailing_zeros();
                bits &= bits - 1;
                let row = wi * 64 + b as usize;
                out |= (payload[row] & 1) << b;
            }
            *d = out;
        }
    }

    /// Time `f` to a 30 ms floor, best of 3 rounds; ns per call.
    fn time(mut f: impl FnMut()) -> f64 {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let mut reps = 0u64;
            let t0 = Instant::now();
            while t0.elapsed().as_millis() < 30 {
                f();
                reps += 1;
            }
            best = best.min(t0.elapsed().as_nanos() as f64 / reps as f64);
        }
        best
    }

    fn frontier(seed: &mut u64, live_bits: usize, clustered: bool) -> Vec<u64> {
        let mut m = vec![0u64; WORDS];
        let mut placed = 0;
        // clustered: live bits confined to the first `span` words, where span is
        // the smallest run that fits them at ~50% fill; uniform: anywhere.
        let span = if clustered {
            ((live_bits * 2) / 64).clamp(1, WORDS)
        } else {
            WORDS
        };
        while placed < live_bits {
            let r = splitmix(seed) as usize;
            let wi = r % span;
            let b = (r >> 20) % 64;
            if m[wi] >> b & 1 == 0 {
                m[wi] |= 1 << b;
                placed += 1;
            }
        }
        m
    }

    pub fn main() {
        println!(
            "realization: avx512f={} avx512vl={}  rows={ROWS} words={WORDS}",
            cfg!(target_feature = "avx512f"),
            cfg!(target_feature = "avx512vl")
        );
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        // Operands at ~62% density each (the MQ fixture's permeability).
        let dense = |seed: &mut u64| -> Vec<u64> {
            (0..WORDS)
                .map(|_| splitmix(seed) | splitmix(seed) & splitmix(seed))
                .collect()
        };
        let gate = dense(&mut seed);
        let elig = dense(&mut seed);
        // payload column: bit 0 = gate ∧ elig at that row, upper bits noise.
        let payload: Vec<u64> = (0..ROWS)
            .map(|r| (splitmix(&mut seed) & !1) | ((gate[r / 64] & elig[r / 64]) >> (r % 64) & 1))
            .collect();

        println!(
            "\n{:<9} {:<10} {:>8} | {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} | {:<12}",
            "frontier", "shape", "live", "F8", "S8", "S4", "S2", "S1(gpr)", "W(gather)", "best vector"
        );
        for &pct in &[0.01f64, 0.1, 1.0, 10.0, 100.0] {
            let live = ((ROWS as f64) * pct / 100.0).round().max(1.0) as usize;
            for clustered in [false, true] {
                let src = frontier(&mut seed, live, clustered);
                let mut d_ref = vec![0u64; WORDS];
                f8(&src, &gate, &elig, &mut d_ref);
                let mut d = vec![0u64; WORDS];
                // equivalence gate, every arm, before any timing
                unsafe { s8(&src, &gate, &elig, &mut d) };
                assert_eq!(d, d_ref, "S8");
                unsafe { s4(&src, &gate, &elig, &mut d) };
                assert_eq!(d, d_ref, "S4");
                unsafe { s2(&src, &gate, &elig, &mut d) };
                assert_eq!(d, d_ref, "S2");
                s1(&src, &gate, &elig, &mut d);
                assert_eq!(d, d_ref, "S1");
                w(&src, &payload, &mut d);
                assert_eq!(d, d_ref, "W");

                let t_f8 = time(|| f8(black_box(&src), black_box(&gate), black_box(&elig), black_box(&mut d)));
                let t_s8 =
                    time(|| unsafe { s8(black_box(&src), black_box(&gate), black_box(&elig), black_box(&mut d)) });
                let t_s4 =
                    time(|| unsafe { s4(black_box(&src), black_box(&gate), black_box(&elig), black_box(&mut d)) });
                let t_s2 =
                    time(|| unsafe { s2(black_box(&src), black_box(&gate), black_box(&elig), black_box(&mut d)) });
                let t_s1 = time(|| s1(black_box(&src), black_box(&gate), black_box(&elig), black_box(&mut d)));
                let t_w = time(|| w(black_box(&src), black_box(&payload), black_box(&mut d)));
                let best = [("S8", t_s8), ("S4", t_s4), ("S2", t_s2)]
                    .iter()
                    .fold(("F8", t_f8), |acc, &(n, t)| if t < acc.1 { (n, t) } else { acc });
                println!(
                    "{:<9} {:<10} {:>8} | {:>8.0} {:>8.0} {:>8.0} {:>8.0} {:>8.0} {:>9.0} | {} ({:.2}x F8, {:.2}x W)",
                    format!("{pct}%"),
                    if clustered { "clustered" } else { "uniform" },
                    live,
                    t_f8,
                    t_s8,
                    t_s4,
                    t_s2,
                    t_s1,
                    t_w,
                    best.0,
                    t_f8 / best.1,
                    t_w / best.1
                );
            }
        }
        println!("\nns per call; equivalence asserted for every arm before timing.");
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512dq"
))]
fn main() {
    probe::main();
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vl",
    target_feature = "avx512dq"
)))]
fn main() {
    println!("ternlogq_sparse_reapply_probe: AVX-512 only; nothing to measure on this realization.");
}
