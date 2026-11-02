//! Cost probe for 9-bit sign-magnitude (SM9) arithmetic and 512-entry lookup.
//!
//! Measurement only: the raw intrinsics here are candidates being timed, not
//! a realization. Compares, per 64 lanes, a plain u8 x i8 four-byte dot
//! (`vpdpbusd`), the signed i8 x i8 dot, three exact ways to multiply a
//! sign-magnitude value in -255..=255 by an i8 weight, and the ways to look a
//! 9-bit centroid ID up in a 512-entry table.
//!
//! Build with the host's features (`.cargo/config.toml` is `target-cpu=native`):
//! `cargo run --release --example u9_cost_probe --features std`.

#[cfg(all(target_arch = "x86_64", target_feature = "avx512vnni", target_feature = "avx512vbmi"))]
mod probe {
    use core::arch::x86_64::*;
    use std::hint::black_box;
    use std::time::Instant;

    /// Chunks of 64 lanes per pass: magnitudes + weights stay in L1.
    const CHUNKS: usize = 256;
    const PASSES: usize = 4000;

    struct Data {
        mag: Vec<[u8; 64]>,
        pos: Vec<u64>,
        w: Vec<[i8; 64]>,
        ids_lo: Vec<[u8; 64]>,
        ids_hi: Vec<u64>,
    }

    fn data() -> Data {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            s >> 11
        };
        Data {
            mag: (0..CHUNKS)
                .map(|_| core::array::from_fn(|_| next() as u8))
                .collect(),
            pos: (0..CHUNKS).map(|_| next() ^ (next() << 32)).collect(),
            w: (0..CHUNKS)
                .map(|_| core::array::from_fn(|_| i8::from_ne_bytes([next() as u8])))
                .collect(),
            ids_lo: (0..CHUNKS)
                .map(|_| core::array::from_fn(|_| next() as u8))
                .collect(),
            ids_hi: (0..CHUNKS).map(|_| next() ^ (next() << 32)).collect(),
        }
    }

    /// Runs `body` over every chunk PASSES times; returns (ns, TSC ticks) per 64 lanes.
    fn time(name: &str, mut body: impl FnMut() -> i64) -> (f64, f64) {
        black_box(body());
        let t0 = Instant::now();
        // SAFETY: rdtsc has no preconditions.
        let c0 = unsafe { _rdtsc() };
        let mut sink = 0i64;
        for _ in 0..PASSES {
            sink = sink.wrapping_add(body());
        }
        // SAFETY: as above.
        let c1 = unsafe { _rdtsc() };
        let ns = t0.elapsed().as_nanos() as f64;
        black_box(sink);
        let per = (PASSES * CHUNKS) as f64;
        let (n, c) = (ns / per, (c1 - c0) as f64 / per);
        println!("{name:<44} {n:7.3} ns/64 lanes  {c:6.2} TSC/64  {:6.3} TSC/elem", c / 64.0);
        (n, c)
    }

    #[inline(always)]
    unsafe fn ld(p: &[u8; 64]) -> __m512i {
        unsafe { _mm512_loadu_si512(p.as_ptr() as *const __m512i) }
    }
    #[inline(always)]
    unsafe fn ldi(p: &[i8; 64]) -> __m512i {
        unsafe { _mm512_loadu_si512(p.as_ptr() as *const __m512i) }
    }
    /// Times `body` only after it matched the exact reference once.
    fn checked(expected: i64, name: &str, mut body: impl FnMut() -> i64) -> (f64, f64) {
        assert_eq!(body(), expected, "{name} does not compute the exact SM9 dot");
        time(name, body)
    }

    #[inline(always)]
    unsafe fn hsum(v: __m512i) -> i64 {
        unsafe { _mm512_reduce_add_epi32(v) as i64 }
    }

    /// Codes streamed from DRAM (64 MiB of code bytes), query weights L1-resident:
    /// the shape of a scan over an index larger than cache.
    fn stream(d: &Data, mag_t: &[u8; 512]) {
        const BIG: usize = 1 << 20; // chunks of 64 code bytes
        let codes: Vec<[u8; 64]> = (0..BIG).map(|i| d.mag[i % CHUNKS]).collect();
        let plane: Vec<u64> = (0..BIG).map(|i| d.pos[i % CHUNKS] ^ i as u64).collect();
        println!("== streamed from DRAM: 64 MiB of code bytes, weights in L1 ==");
        let run = |name: &str, f: &dyn Fn() -> i64, dims: f64, bytes: f64| {
            black_box(f());
            let t0 = Instant::now();
            let mut sink = 0i64;
            for _ in 0..5 {
                sink = sink.wrapping_add(f());
            }
            let ns = t0.elapsed().as_nanos() as f64 / 5.0;
            black_box(sink);
            println!("{name:<44} {:7.4} ns/dim  {:6.2} GB/s", ns / dims, bytes / ns);
        };
        // SAFETY (both): avx512f/bw/vnni compiled in (module cfg); whole-array loads.
        run(
            "i4 permute-dot, 128 dims per 64 bytes",
            &|| unsafe {
                let lv = _mm512_broadcast_i32x4(_mm_loadu_si128(mag_t.as_ptr() as *const _));
                let mut acc = _mm512_setzero_si512();
                for (i, c) in codes.iter().enumerate() {
                    let codes = ld(c);
                    let w = ldi(&d.w[i % CHUNKS]);
                    let l = _mm512_shuffle_epi8(lv, _mm512_and_si512(codes, _mm512_set1_epi8(0x0F)));
                    let h =
                        _mm512_shuffle_epi8(lv, _mm512_and_si512(_mm512_srli_epi16(codes, 4), _mm512_set1_epi8(0x0F)));
                    acc = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(acc, l, w), h, w);
                }
                hsum(acc)
            },
            (BIG * 128) as f64,
            (BIG * 64) as f64,
        );
        run(
            "i9 stored SM9 C', 64 dims per 72 bytes",
            &|| unsafe {
                let (mut p, mut n) = (_mm512_setzero_si512(), _mm512_setzero_si512());
                for (i, (c, &pl)) in codes.iter().zip(&plane).enumerate() {
                    let (m, w) = (ld(c), ldi(&d.w[i % CHUNKS]));
                    p = _mm512_dpbusd_epi32(p, _mm512_maskz_mov_epi8(pl, m), w);
                    n = _mm512_dpbusd_epi32(n, _mm512_maskz_mov_epi8(!pl, m), w);
                }
                hsum(_mm512_sub_epi32(p, n))
            },
            (BIG * 64) as f64,
            (BIG * 72) as f64,
        );
    }

    pub fn run() {
        let d = data();
        // Independent i64 reference for the exact SM9 kernels: every kernel checked
        // against it must match before its time is reported.
        let sm9: i64 = (0..CHUNKS)
            .flat_map(|c| (0..64).map(move |i| (c, i)))
            .map(|(c, i)| {
                let v = i64::from(d.mag[c][i]);
                (if d.pos[c] >> i & 1 == 1 { v } else { -v }) * i64::from(d.w[c][i])
            })
            .sum();
        println!("== arithmetic (one accumulator chain per kernel, data in L1) ==");
        // SAFETY (all kernels): the build has avx512f/bw/vnni/vbmi compiled in
        // (module cfg); loads read whole 64-byte arrays.
        let base = time("u8 x i8 dot4 (vpdpbusd, #386)", || unsafe {
            let mut acc = _mm512_setzero_si512();
            for c in 0..CHUNKS {
                acc = _mm512_dpbusd_epi32(acc, ld(&d.mag[c]), ldi(&d.w[c]));
            }
            hsum(acc)
        });
        time("i8 x i8 dot4 (xor 0x80 bias, 1 dpbusd)", || unsafe {
            let bias = _mm512_set1_epi8(-128);
            let mut acc = _mm512_setzero_si512();
            for c in 0..CHUNKS {
                acc = _mm512_dpbusd_epi32(acc, _mm512_xor_si512(ld(&d.mag[c]), bias), ldi(&d.w[c]));
            }
            hsum(acc)
        });
        time("SM9 A: sign folded into w (w in -127..=127)", || unsafe {
            let mut acc = _mm512_setzero_si512();
            for c in 0..CHUNKS {
                let w = ldi(&d.w[c]);
                let wn = _mm512_sub_epi8(_mm512_setzero_si512(), w);
                let ws = _mm512_mask_blend_epi8(!d.pos[c], w, wn);
                acc = _mm512_dpbusd_epi32(acc, ld(&d.mag[c]), ws);
            }
            hsum(acc)
        });
        checked(sm9, "SM9 B: 2C limbs, dot(lo) - 256*dot(sign)", || unsafe {
            let one = _mm512_set1_epi8(1);
            let mut acc = _mm512_setzero_si512();
            for c in 0..CHUNKS {
                let w = ldi(&d.w[c]);
                let m = ld(&d.mag[c]);
                // Two's-complement -m is 256 * (-1) + ((256 - m) & 0xFF) only for m != 0:
                // a negative zero has no borrow, so the sign limb skips it.
                let neg = !d.pos[c] & _mm512_test_epi8_mask(m, m);
                let lo = _mm512_mask_sub_epi8(m, neg, _mm512_setzero_si512(), m);
                let a = _mm512_dpbusd_epi32(acc, lo, w);
                let t = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_maskz_mov_epi8(neg, one), w);
                acc = _mm512_sub_epi32(a, _mm512_slli_epi32(t, 8));
            }
            hsum(acc)
        });
        checked(sm9, "SM9 C: dot(m) - 2*dot(m & neg), one acc", || unsafe {
            let mut acc = _mm512_setzero_si512();
            for c in 0..CHUNKS {
                let (m, w) = (ld(&d.mag[c]), ldi(&d.w[c]));
                let a = _mm512_dpbusd_epi32(acc, m, w);
                let t = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_maskz_mov_epi8(!d.pos[c], m), w);
                acc = _mm512_sub_epi32(a, _mm512_add_epi32(t, t));
            }
            hsum(acc)
        });
        checked(sm9, "SM9 C': dot(m & pos) - dot(m & neg), two accs", || unsafe {
            let (mut p, mut n) = (_mm512_setzero_si512(), _mm512_setzero_si512());
            for c in 0..CHUNKS {
                let (m, w) = (ld(&d.mag[c]), ldi(&d.w[c]));
                p = _mm512_dpbusd_epi32(p, _mm512_maskz_mov_epi8(d.pos[c], m), w);
                n = _mm512_dpbusd_epi32(n, _mm512_maskz_mov_epi8(!d.pos[c], m), w);
            }
            hsum(_mm512_sub_epi32(p, n))
        });

        println!("== 512-entry lookup: 9-bit ID (lo byte + hi plane) -> SM9 (magnitude + positive plane) ==");
        let mag_t: [u8; 512] = core::array::from_fn(|i| (i as u32 * 97 % 256) as u8);
        let pos_t: [u8; 512] = core::array::from_fn(|i| (i >= 256) as u8);
        time("per-position reads (lut_u8 shape)", || {
            let mut sink = 0i64;
            for c in 0..CHUNKS {
                let (lo, hi) = (&d.ids_lo[c], d.ids_hi[c]);
                let mut out = [0u8; 64];
                let mut plane = 0u64;
                for (i, o) in out.iter_mut().enumerate() {
                    let idx = usize::from(lo[i]) | ((((hi >> i) & 1) as usize) << 8);
                    *o = mag_t[idx];
                    plane |= u64::from(pos_t[idx]) << i;
                }
                sink = sink.wrapping_add(i64::from(out[c & 63]) ^ plane as i64);
            }
            sink
        });
        time("vpermi2b x4 + blends (mag) + x4 (sign)", || unsafe {
            let t: [__m512i; 8] = core::array::from_fn(|k| _mm512_loadu_si512(mag_t.as_ptr().add(64 * k) as *const _));
            let s: [__m512i; 8] = core::array::from_fn(|k| _mm512_loadu_si512(pos_t.as_ptr().add(64 * k) as *const _));
            let mut sink = 0i64;
            for c in 0..CHUNKS {
                let lo = ld(&d.ids_lo[c]);
                let b7 = _mm512_movepi8_mask(lo);
                let hi = d.ids_hi[c];
                let pick = |tb: &[__m512i; 8]| {
                    let q0 = _mm512_permutex2var_epi8(tb[0], lo, tb[1]);
                    let q1 = _mm512_permutex2var_epi8(tb[2], lo, tb[3]);
                    let q2 = _mm512_permutex2var_epi8(tb[4], lo, tb[5]);
                    let q3 = _mm512_permutex2var_epi8(tb[6], lo, tb[7]);
                    let l = _mm512_mask_blend_epi8(b7, q0, q1);
                    let h = _mm512_mask_blend_epi8(b7, q2, q3);
                    _mm512_mask_blend_epi8(hi, l, h)
                };
                let m = pick(&t);
                let plane = _mm512_test_epi8_mask(pick(&s), pick(&s));
                sink = sink.wrapping_add(hsum(m) ^ plane as i64);
            }
            sink
        });
        time("vpermi2b x2 + blend, antisymmetric codebook", || unsafe {
            let t: [__m512i; 4] =
                core::array::from_fn(|k| _mm512_loadu_si512(mag_t.as_ptr().add(256 + 64 * k) as *const _));
            let mut sink = 0i64;
            for c in 0..CHUNKS {
                let hi = d.ids_hi[c];
                // negative half reads entry 255 - lo of the positive table
                let raw = ld(&d.ids_lo[c]);
                let lo = _mm512_mask_blend_epi8(!hi, raw, _mm512_xor_si512(raw, _mm512_set1_epi8(-1)));
                let b7 = _mm512_movepi8_mask(lo);
                let q0 = _mm512_permutex2var_epi8(t[0], lo, t[1]);
                let q1 = _mm512_permutex2var_epi8(t[2], lo, t[3]);
                let m = _mm512_mask_blend_epi8(b7, q0, q1);
                sink = sink.wrapping_add(hsum(m) ^ hi as i64);
            }
            sink
        });

        println!("== end to end per 64 lanes: lookup + SM9 dot vs i4 permute-dot ==");
        time("i4 permute-dot (vpshufb 16-level + dpbusd)", || unsafe {
            let lv = _mm512_broadcast_i32x4(_mm_loadu_si128(mag_t.as_ptr() as *const _));
            let mut acc = _mm512_setzero_si512();
            for c in 0..CHUNKS {
                let codes = ld(&d.ids_lo[c]);
                let l = _mm512_shuffle_epi8(lv, _mm512_and_si512(codes, _mm512_set1_epi8(0x0F)));
                let h = _mm512_shuffle_epi8(lv, _mm512_and_si512(_mm512_srli_epi16(codes, 4), _mm512_set1_epi8(0x0F)));
                acc = _mm512_dpbusd_epi32(acc, l, ldi(&d.w[c]));
                acc = _mm512_dpbusd_epi32(acc, h, ldi(&d.w[c]));
            }
            hsum(acc)
        });
        let e2e = time("i9 8+1: vpermi2b lookup + SM9 C'", || unsafe {
            let t: [__m512i; 8] = core::array::from_fn(|k| _mm512_loadu_si512(mag_t.as_ptr().add(64 * k) as *const _));
            let s: [__m512i; 8] = core::array::from_fn(|k| _mm512_loadu_si512(pos_t.as_ptr().add(64 * k) as *const _));
            let (mut p, mut n) = (_mm512_setzero_si512(), _mm512_setzero_si512());
            for c in 0..CHUNKS {
                let lo = ld(&d.ids_lo[c]);
                let b7 = _mm512_movepi8_mask(lo);
                let hi = d.ids_hi[c];
                let pick = |tb: &[__m512i; 8]| {
                    let q0 = _mm512_permutex2var_epi8(tb[0], lo, tb[1]);
                    let q1 = _mm512_permutex2var_epi8(tb[2], lo, tb[3]);
                    let q2 = _mm512_permutex2var_epi8(tb[4], lo, tb[5]);
                    let q3 = _mm512_permutex2var_epi8(tb[6], lo, tb[7]);
                    _mm512_mask_blend_epi8(hi, _mm512_mask_blend_epi8(b7, q0, q1), _mm512_mask_blend_epi8(b7, q2, q3))
                };
                let m = pick(&t);
                let pos = _mm512_test_epi8_mask(pick(&s), pick(&s));
                let w = ldi(&d.w[c]);
                p = _mm512_dpbusd_epi32(p, _mm512_maskz_mov_epi8(pos, m), w);
                n = _mm512_dpbusd_epi32(n, _mm512_maskz_mov_epi8(!pos, m), w);
            }
            hsum(_mm512_sub_epi32(p, n))
        });
        time("4+5 additive: vpshufb C16 + vpermb R32, 2 dot4", || unsafe {
            // value = C16[hi4] + R32[lo5], both stored +128-biased u8; the bias is
            // removed once per query (2*128*sum(w)), outside this loop.
            let c16 = _mm512_broadcast_i32x4(_mm_loadu_si128(mag_t.as_ptr() as *const _));
            let r32 = _mm512_loadu_si512(mag_t.as_ptr().add(64) as *const _);
            let mut acc = _mm512_setzero_si512();
            for c in 0..CHUNKS {
                let raw = ld(&d.ids_lo[c]);
                let hi4 = _mm512_and_si512(_mm512_srli_epi16(raw, 4), _mm512_set1_epi8(0x0F));
                let lo5 = _mm512_and_si512(raw, _mm512_set1_epi8(0x1F));
                let w = ldi(&d.w[c]);
                acc = _mm512_dpbusd_epi32(acc, _mm512_shuffle_epi8(c16, hi4), w);
                acc = _mm512_dpbusd_epi32(acc, _mm512_permutexvar_epi8(lo5, r32), w);
            }
            hsum(acc)
        });
        checked(sm9, "i9 values stored (no lookup) + SM9 C'", || unsafe {
            let (mut p, mut n) = (_mm512_setzero_si512(), _mm512_setzero_si512());
            for c in 0..CHUNKS {
                let (m, w) = (ld(&d.mag[c]), ldi(&d.w[c]));
                p = _mm512_dpbusd_epi32(p, _mm512_maskz_mov_epi8(d.pos[c], m), w);
                n = _mm512_dpbusd_epi32(n, _mm512_maskz_mov_epi8(!d.pos[c], m), w);
            }
            hsum(_mm512_sub_epi32(p, n))
        });
        stream(&d, &mag_t);
        println!(
            "note: i4 permute-dot scores 128 dims per 64 bytes, i9 scores 64 dims per 64 bytes + 8-byte plane; \
             per dimension the i9 end-to-end costs {:.2}x the u8 dot4 per 64 lanes",
            e2e.0 / base.0
        );
    }
}

/// Reconstruction error of the candidate 9-bit layouts against a unit Gaussian,
/// the coordinate distribution after TurboVec's rotation. The codebooks are
/// probe-local Lloyd-Max fits on a discretized density, not TurboVec's solver.
mod precision {
    const BINS: usize = 1 << 20;
    const LO: f64 = -7.0;
    const HI: f64 = 7.0;

    fn grid() -> (Vec<f64>, Vec<f64>) {
        let dx = (HI - LO) / BINS as f64;
        let xs: Vec<f64> = (0..BINS).map(|i| LO + (i as f64 + 0.5) * dx).collect();
        let mut ws: Vec<f64> = xs.iter().map(|x| (-0.5 * x * x).exp()).collect();
        let z: f64 = ws.iter().sum();
        ws.iter_mut().for_each(|w| *w /= z);
        (xs, ws)
    }

    /// Lloyd-Max levels for weighted samples (xs sorted ascending).
    fn lloyd(xs: &[f64], ws: &[f64], n: usize) -> Vec<f64> {
        let (mut c, mut acc, mut lv) = (0.0, 0.0, vec![0.0; n]);
        // Start at the high-resolution optimum: levels at the quantiles of the
        // density raised to 1/3 (Panter-Dite), so few iterations are needed.
        let w3: Vec<f64> = ws.iter().map(|w| w.cbrt()).collect();
        let z3: f64 = w3.iter().sum();
        let mut k = 0;
        for (x, w) in xs.iter().zip(&w3) {
            acc += w / z3;
            while k < n && acc >= (k as f64 + 0.5) / n as f64 {
                lv[k] = *x;
                k += 1;
            }
        }
        for _ in 0..1000 {
            let (mut sw, mut sx) = (vec![0.0; n], vec![0.0; n]);
            let mut j = 0;
            for (x, w) in xs.iter().zip(ws) {
                while j + 1 < n && *x > 0.5 * (lv[j] + lv[j + 1]) {
                    j += 1;
                }
                sw[j] += w;
                sx[j] += w * x;
            }
            let mut moved = 0.0f64;
            for i in 0..n {
                if sw[i] > 0.0 {
                    let v = sx[i] / sw[i];
                    moved = moved.max((v - lv[i]).abs());
                    lv[i] = v;
                }
            }
            c = moved;
            if moved < 1e-12 {
                break;
            }
        }
        let _ = c;
        lv
    }

    fn mse(xs: &[f64], ws: &[f64], levels: &[f64]) -> f64 {
        let mut l = levels.to_vec();
        l.sort_by(f64::total_cmp);
        let mut j = 0;
        let mut e = 0.0;
        for (x, w) in xs.iter().zip(ws) {
            while j + 1 < l.len() && *x > 0.5 * (l[j] + l[j + 1]) {
                j += 1;
            }
            e += w * (x - l[j]) * (x - l[j]);
        }
        e
    }

    fn snap(levels: &[f64], max_code: f64) -> (Vec<f64>, usize) {
        let s = max_code / levels.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let mut q: Vec<f64> = levels.iter().map(|v| (v * s).round() / s).collect();
        q.sort_by(f64::total_cmp);
        q.dedup();
        let n = q.len();
        (q, n)
    }

    pub fn run() {
        let (xs, ws) = grid();
        let db = |m: f64| 10.0 * (1.0 / m).log10();
        println!("== precision on N(0,1): MSE and SNR (probe-local Lloyd-Max, not TurboVec's solver) ==");
        let l16 = lloyd(&xs, &ws, 16);
        let (q16, _) = snap(&l16, 127.0);
        println!("i4  16 levels (f64)          MSE {:.3e}  {:5.2} dB", mse(&xs, &ws, &l16), db(mse(&xs, &ws, &l16)));
        println!("i4  16 levels on i8 grid     MSE {:.3e}  {:5.2} dB", mse(&xs, &ws, &q16), db(mse(&xs, &ws, &q16)));
        let l256 = lloyd(&xs, &ws, 256);
        let (q256, n256) = snap(&l256, 127.0);
        println!("i8 256 levels (f64)          MSE {:.3e}  {:5.2} dB", mse(&xs, &ws, &l256), db(mse(&xs, &ws, &l256)));
        println!(
            "i8 256 levels on i8 grid     MSE {:.3e}  {:5.2} dB  ({n256} distinct)",
            mse(&xs, &ws, &q256),
            db(mse(&xs, &ws, &q256))
        );
        let l512 = lloyd(&xs, &ws, 512);
        let (q512, n512) = snap(&l512, 255.0);
        println!("i9 512 levels (f64, ideal)   MSE {:.3e}  {:5.2} dB", mse(&xs, &ws, &l512), db(mse(&xs, &ws, &l512)));
        println!(
            "i9 512 levels on SM9 grid    MSE {:.3e}  {:5.2} dB  ({n512} distinct of 512)",
            mse(&xs, &ws, &q512),
            db(mse(&xs, &ws, &q512))
        );
        // 4+5 additive: coarse 16-level Lloyd, then one shared 32-level residual Lloyd.
        let coarse = l16.clone();
        let mut rx = Vec::with_capacity(BINS);
        {
            let mut j = 0;
            for x in &xs {
                while j + 1 < coarse.len() && *x > 0.5 * (coarse[j] + coarse[j + 1]) {
                    j += 1;
                }
                rx.push(x - coarse[j]);
            }
        }
        let mut order: Vec<usize> = (0..BINS).collect();
        order.sort_by(|a, b| rx[*a].total_cmp(&rx[*b]));
        let (rxs, rws): (Vec<f64>, Vec<f64>) = order.iter().map(|&i| (rx[i], ws[i])).unzip();
        let r32 = lloyd(&rxs, &rws, 32);
        let sum: Vec<f64> = coarse
            .iter()
            .flat_map(|c| r32.iter().map(move |r| c + r))
            .collect();
        // The stored-value layout is a uniform 9-bit grid, so the Lloyd IDs buy nothing
        // there; what matters is the clip. Sweep the step for the best uniform SM9 grid.
        let mut best = (f64::MAX, 0.0);
        for k in 0..200 {
            let clip = 2.5 + 0.01 * k as f64;
            let grid: Vec<f64> = (-255..=255).map(|m| m as f64 * clip / 255.0).collect();
            let e = mse(&xs, &ws, &grid);
            if e < best.0 {
                best = (e, clip);
            }
        }
        let mut best8 = (f64::MAX, 0.0);
        for k in 0..200 {
            let clip = 2.0 + 0.01 * k as f64;
            let grid: Vec<f64> = (-127..=127).map(|m| m as f64 * clip / 127.0).collect();
            let e = mse(&xs, &ws, &grid);
            if e < best8.0 {
                best8 = (e, clip);
            }
        }
        println!(
            "i8 uniform grid, best clip    MSE {:.3e}  {:5.2} dB  (clip {:.2} sigma, 255 values)",
            best8.0,
            db(best8.0),
            best8.1
        );
        println!(
            "SM9 uniform grid, best clip   MSE {:.3e}  {:5.2} dB  (clip {:.2} sigma, 511 values)",
            best.0,
            db(best.0),
            best.1
        );
        println!(
            "4+5 C16+R32 additive (f64)   MSE {:.3e}  {:5.2} dB  (residual codebook fit after the coarse split)",
            mse(&xs, &ws, &sum),
            db(mse(&xs, &ws, &sum))
        );
        // As executed: each table on its own i8 grid (two scales, two accumulators).
        let q8 = |v: &[f64]| {
            let sc = 127.0 / v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            v.iter()
                .map(|x| (x * sc).round() / sc)
                .collect::<Vec<f64>>()
        };
        let (cq, rq) = (q8(&coarse), q8(&r32));
        let sumq: Vec<f64> = cq
            .iter()
            .flat_map(|c| rq.iter().map(move |r| c + r))
            .collect();
        println!(
            "4+5 C16+R32 on two i8 grids   MSE {:.3e}  {:5.2} dB",
            mse(&xs, &ws, &sumq),
            db(mse(&xs, &ws, &sumq))
        );
    }
}

/// The same two kernels through the `ndarray::simd` facade, on whatever tier
/// this build compiled: what a consumer actually calls.
mod facade {
    use ndarray::simd::{I32x16, I8x64, U8x64};
    use std::hint::black_box;
    use std::time::Instant;

    pub fn run() {
        const CHUNKS: usize = 256;
        const PASSES: usize = 4000;
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            s >> 11
        };
        let mag: Vec<U8x64> = (0..CHUNKS)
            .map(|_| U8x64::from_array(core::array::from_fn(|_| next() as u8)))
            .collect();
        let w: Vec<I8x64> = (0..CHUNKS)
            .map(|_| I8x64::from_array(core::array::from_fn(|_| i8::from_ne_bytes([next() as u8]))))
            .collect();
        let pos: Vec<u64> = (0..CHUNKS).map(|_| next() ^ (next() << 32)).collect();
        let time = |name: &str, f: &dyn Fn() -> i32| {
            black_box(f());
            let t0 = Instant::now();
            let mut sink = 0i32;
            for _ in 0..PASSES {
                sink = sink.wrapping_add(f());
            }
            let ns = t0.elapsed().as_nanos() as f64 / (PASSES * CHUNKS) as f64;
            black_box(sink);
            println!("{name:<44} {ns:7.3} ns/64 lanes");
        };
        // Every lane feeds the result, so no backend can drop part of the work.
        let all = |v: I32x16| v.to_array().iter().fold(0i32, |s, x| s.wrapping_add(*x));
        let sm9: i64 = (0..CHUNKS)
            .flat_map(|c| (0..64).map(move |i| (c, i)))
            .map(|(c, i)| {
                let v = i64::from(mag[c].to_array()[i]);
                (if pos[c] >> i & 1 == 1 { v } else { -v }) * i64::from(w[c].to_array()[i])
            })
            .sum();
        println!("== facade (ndarray::simd, this build's tier) ==");
        time("I32x16::dot4_u8_i8_add", &|| {
            let mut acc = I32x16::splat(0);
            for c in 0..CHUNKS {
                acc = acc.dot4_u8_i8_add(mag[c], w[c]);
            }
            all(acc)
        });
        time("two dot4_u8_i8_add accumulators, no mask", &|| {
            let (mut p, mut n) = (I32x16::splat(0), I32x16::splat(0));
            for c in 0..CHUNKS {
                p = p.dot4_u8_i8_add(mag[c], w[c]);
                n = n.dot4_u8_i8_add(mag[(c + 1) % CHUNKS], w[c]);
            }
            all(p).wrapping_sub(all(n))
        });
        let masked = || {
            let (mut p, mut n) = (I32x16::splat(0), I32x16::splat(0));
            for c in 0..CHUNKS {
                p = p.dot4_u8_i8_add_masked(mag[c], pos[c], w[c]);
                n = n.dot4_u8_i8_add_masked(mag[c], !pos[c], w[c]);
            }
            all(p).wrapping_sub(all(n))
        };
        assert_eq!(i64::from(masked()), sm9, "facade SM9 does not compute the exact dot");
        time("SM9: two dot4_u8_i8_add_masked accumulators", &masked);
    }
}

fn main() {
    precision::run();
    facade::run();
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512vnni", target_feature = "avx512vbmi"))]
    probe::run();
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512vnni", target_feature = "avx512vbmi")))]
    eprintln!("u9_cost_probe needs avx512vnni + avx512vbmi compiled in (target-cpu=native on such a host)");
}
