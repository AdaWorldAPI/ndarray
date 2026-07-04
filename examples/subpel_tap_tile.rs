//! Sub-pel interpolation = a BF16 16×16 tile op — measured (HEVC 8-tap).
//!
//! Next probe from the HEVC amortization map (`.claude/knowledge/hevc-amortization-map.md`):
//! HEVC fractional-pel motion compensation applies a **separable 8-tap FIR** to a
//! block (half-pel luma taps `[-1, 4, -11, 40, 40, -11, 4, -1] / 64`). An FIR is a
//! **linear operator**, so filtering a 16×16 block is a matrix product — one
//! `bf16_tile_gemm_16x16` per pass, and the full separable H+V interpolation is
//! `H_v · X · H_h`: **the same two-tile "sandwich" shape as the splat covariance
//! projection** (`Mᵀ·Σ·M`, #233) and the codec transform (`M·X`, #232). This
//! probe measures that the tile-GEMM form matches the direct 8-tap FIR to BF16
//! precision, on the shipped kernel, and reports the tier.
//!
//! Fairness / honesty:
//!   • Both the tile path and the reference use the SAME 16×16 band operator `H`
//!     (8 taps per output column, reflected at the block edge), so this is an
//!     equal-result comparison — only BF16 rounding differs.
//!   • R-5 dispatch still applies: below the per-arch batch crossover (SPR 64,
//!     ICX 32, …) a per-block 8-tap butterfly wins; the tile GEMM wins in the
//!     batched regime. This probe shows the *equivalence*, not that the tile path
//!     is always the right dispatch for one block.
//!
//! The x86-only AMX/tile path is wrapped in a `cfg(target_arch = "x86_64")`
//! module with a non-x86 fallback `main` (see `gridlake_field_tile`).
//!
//! Run: `cargo run --release --example subpel_tap_tile --features std`

#[cfg(target_arch = "x86_64")]
mod amx {
    use ndarray::hpc::quantized::{f32_to_bf16_rounded, BF16};
    use ndarray::simd::{amx_available, bf16_tile_gemm_16x16_amx as bf16_tile_gemm_16x16, bf16_tile_gemm_tier};

    const M: usize = 16; // block edge
    const KPAD: usize = 32; // K padded to a multiple of 32 for the tile GEMM

    /// HEVC half-pel luma 8-tap (sum = 64).
    const TAPS: [f32; 8] = [-1.0, 4.0, -11.0, 40.0, 40.0, -11.0, 4.0, -1.0];

    fn mix(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn to_bf16(src: &[f32]) -> Vec<u16> {
        let mut b = vec![BF16(0); src.len()];
        f32_to_bf16_rounded(src, &mut b);
        b.iter().map(|x| x.0).collect()
    }

    /// Reflect an index into `[0, M)` (block-edge boundary handling).
    fn reflect(i: i32) -> usize {
        let mut i = i;
        while !(0..M as i32).contains(&i) {
            if i < 0 {
                i = -i - 1;
            } else if i >= M as i32 {
                i = 2 * M as i32 - 1 - i;
            }
        }
        i as usize
    }

    /// The 16×16 band operator `H`: `out[·][j] = Σ_t TAPS[t]/64 · in[·][reflect(j-3+t)]`,
    /// i.e. `H[k][j]` accumulates the tap weight of input column `k` for output
    /// column `j`. `out = X · H`. This IS the sub-pel FIR as a fixed matrix.
    fn band_operator() -> [f32; M * M] {
        let mut h = [0.0f32; M * M];
        for j in 0..M {
            for (t, &w) in TAPS.iter().enumerate() {
                let k = reflect(j as i32 - 3 + t as i32);
                h[k * M + j] += w / 64.0;
            }
        }
        h
    }

    /// `C[16×16] = A[16×16] · B[16×16]` via the shipped BF16 tile op (K padded 16→32).
    fn tile_matmul(a: &[f32; M * M], b: &[f32; M * M]) -> [f32; M * M] {
        let mut a_pad = [0.0f32; M * KPAD];
        let mut b_pad = [0.0f32; KPAD * M];
        for i in 0..M {
            for k in 0..M {
                a_pad[i * KPAD + k] = a[i * M + k];
                b_pad[k * M + i] = b[k * M + i];
            }
        }
        let (ab, bb) = (to_bf16(&a_pad), to_bf16(&b_pad));
        let mut c = vec![0.0f32; M * M];
        bf16_tile_gemm_16x16(&ab, &bb, &mut c, KPAD);
        let mut out = [0.0f32; M * M];
        out.copy_from_slice(&c);
        out
    }

    fn direct_matmul(a: &[f32; M * M], b: &[f32; M * M]) -> [f64; M * M] {
        let mut c = [0.0f64; M * M];
        for i in 0..M {
            for j in 0..M {
                let mut s = 0.0f64;
                for k in 0..M {
                    s += a[i * M + k] as f64 * b[k * M + j] as f64;
                }
                c[i * M + j] = s;
            }
        }
        c
    }

    fn transpose(a: &[f32; M * M]) -> [f32; M * M] {
        let mut t = [0.0f32; M * M];
        for i in 0..M {
            for j in 0..M {
                t[j * M + i] = a[i * M + j];
            }
        }
        t
    }

    /// Direct separable H+V 8-tap FIR reference (`f64`), reflected edges — the
    /// ground truth the tile-GEMM sandwich must match.
    fn direct_hv(x: &[f32; M * M]) -> [f64; M * M] {
        // horizontal
        let mut h = [0.0f64; M * M];
        for i in 0..M {
            for j in 0..M {
                let mut s = 0.0f64;
                for (t, &w) in TAPS.iter().enumerate() {
                    s += (w as f64 / 64.0) * x[i * M + reflect(j as i32 - 3 + t as i32)] as f64;
                }
                h[i * M + j] = s;
            }
        }
        // vertical
        let mut out = [0.0f64; M * M];
        for i in 0..M {
            for j in 0..M {
                let mut s = 0.0f64;
                for (t, &w) in TAPS.iter().enumerate() {
                    s += (w as f64 / 64.0) * h[reflect(i as i32 - 3 + t as i32) * M + j];
                }
                out[i * M + j] = s;
            }
        }
        out
    }

    fn frob_rel(tile: &[f32; M * M], direct: &[f64; M * M]) -> (f64, f64) {
        let (mut max_abs, mut num, mut den) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..M * M {
            let e = tile[i] as f64 - direct[i];
            max_abs = max_abs.max(e.abs());
            num += e * e;
            den += direct[i] * direct[i];
        }
        (max_abs, (num / den.max(1e-12)).sqrt())
    }

    pub fn run() {
        println!("Sub-pel interpolation = BF16 16×16 tile op — measured (HEVC 8-tap half-pel)");
        println!(
            "  shipped kernel tier on THIS host: {}  (amx_available = {})\n",
            bf16_tile_gemm_tier(),
            amx_available()
        );

        // 7-bit luma block (fits u8 and i8; positive operand for the tile GEMM)
        let mut x = [0.0f32; M * M];
        for r in 0..M {
            for c in 0..M {
                x[r * M + c] = ((mix((r as u64) << 8 | c as u64) & 0x7F) as f32) * 0.7 + 20.0 * (r as f32 * 0.3).sin();
            }
        }
        let h = band_operator();
        let ht = transpose(&h);

        // (1) horizontal half-pel = X · H  (one tile op)
        let horiz_tile = tile_matmul(&x, &h);
        let horiz_direct = direct_matmul(&x, &h);
        let (ha, hr) = frob_rel(&horiz_tile, &horiz_direct);
        println!("  (1) horizontal 8-tap   out = X·H            (one tile op)");
        println!("      max_abs_err={ha:.4}  frobenius_rel_err={:.3}%", hr * 100.0);

        // (2) full separable H+V = Hᵀ · (X · H)  (two tile ops — the sandwich shape)
        let hv_tile = tile_matmul(&ht, &horiz_tile);
        let hv_direct = direct_hv(&x);
        let (va, vr) = frob_rel(&hv_tile, &hv_direct);
        println!("  (2) separable H+V      out = Hᵀ·(X·H)       (two tile ops = Mᵀ·Σ·M sandwich)");
        println!("      max_abs_err={va:.4}  frobenius_rel_err={:.3}%", vr * 100.0);

        // (3) throughput of the sub-pel tile op
        let iters = 200_000usize;
        let t0 = std::time::Instant::now();
        let mut acc = 0.0f32;
        for it in 0..iters {
            let mut xv = x;
            xv[0] += (it & 0x7) as f32 * 0.01;
            let c = tile_matmul(&xv, &h);
            acc += c[0] + c[M * M - 1];
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "  (3) throughput: {iters} sub-pel tile ops in {:.1} ms → {:.2} M/s  (checksum {acc:.1})",
            dt * 1000.0,
            iters as f64 / dt / 1e6
        );

        assert!(hr < 0.05, "horizontal FIR tile-GEMM rel err too high for BF16 class");
        assert!(vr < 0.05, "separable H+V tile-GEMM rel err too high for BF16 class");

        println!(
            "\n  MEASURED: HEVC 8-tap sub-pel interpolation IS a tile GEMM — X·H (one op)\n\
             \x20 and the separable H+V is Hᵀ·(X·H), the SAME two-tile sandwich shape as the\n\
             \x20 splat covariance projection (#233) and the codec transform (#232). Matches\n\
             \x20 the direct 8-tap FIR to BF16 precision. Fractional-pel motion is therefore\n\
             \x20 'just another tile op', closing the MC story past integer-pel. (Dispatch\n\
             \x20 per R-5: per-block 8-tap butterfly below the batch crossover, batched tile\n\
             \x20 GEMM above it.)"
        );
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    amx::run();
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("subpel_tap_tile requires x86_64 (AMX TDPBF16PS / AVX-512 VDPBF16PS BF16 tile GEMM).");
}
