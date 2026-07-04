//! Gridlake field interdependency = one BF16 16×16 AMX tile op — measured.
//!
//! The operator's claim (2026-07-04): a **gridlake**'s cell *interdependency*,
//! treated as a *field operation*, reduces to a **BF16 16×16 AMX tile GEMM**
//! (`TDPBF16PS`) — the same shape as the **intercorrelation / perturbation in a
//! Gaussian splat** (the EWA covariance sandwich `Σ' = Mᵀ·Σ·M`, shipped as
//! `hpc::splat3d::spd3::sandwich_x16`).
//!
//! Why it's true, mechanically: a *field operation* whose output cell is a linear
//! combination of input cells is a linear operator `C = A·B` — a GEMM. The
//! "interdependency" / "intercorrelation" IS the coupling matrix (the off-diagonal
//! second-moment structure). At the gridlake tile granularity (16×16, one quarter
//! of the 64×64 gridlake in each axis → the AMX tile) that GEMM is exactly one
//! `hpc::bf16_tile_gemm::bf16_tile_gemm_16x16` call — one `TDPBF16PS` tile op.
//! This is the SAME primitive as:
//!   • the codec separable transform (#232: `M·X`, the WHT/DCT on a tile), and
//!   • the splat EWA covariance projection (`sandwich_x16`, `Mᵀ·Σ·M`).
//! One op, three names.
//!
//! This probe MEASURES the reduction on the shipped kernel (whatever tier the host
//! selects — AMX `TDPBF16PS` / AVX-512 `VDPBF16PS` / F32x16 polyfill), reports the
//! tier, and checks against a direct `f64` reference:
//!   1. **intercorrelation** `C = Xᵀ·X` of a gridlake field tile (the inter-cell
//!      coupling as a field op) — one tile GEMM.
//!   2. **covariance sandwich** `Σ' = Mᵀ·Σ·M` (the splat EWA projection shape) —
//!      two tile GEMMs.
//!   3. **bit-exactness** on bf16-exact integer operands (the module's guarantee),
//!      and BF16-precision relative error on a continuous field.
//!   4. **throughput** of the 16×16 tile op on this host.
//!
//! Run: `cargo run --release --example gridlake_field_tile --features std`

use ndarray::hpc::amx_matmul::amx_available;
use ndarray::hpc::bf16_tile_gemm::{bf16_tile_gemm_16x16, bf16_tile_gemm_tier};
use ndarray::hpc::quantized::{f32_to_bf16_rounded, BF16};

const M: usize = 16; // tile edge — the AMX tile / gridlake quadrant
const KPAD: usize = 32; // bf16_tile_gemm requires K a multiple of 32; pad 16→32 with zeros

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Pack a row-major `M×KPAD` (or `KPAD×M`) f32 matrix into the `&[u16]` bf16
/// operand the tile GEMM consumes (RNE, matching `VCVTNEPS2BF16`).
fn to_bf16(src: &[f32]) -> Vec<u16> {
    let mut b = vec![BF16(0); src.len()];
    f32_to_bf16_rounded(src, &mut b);
    b.iter().map(|x| x.0).collect()
}

/// `C[16×16] = A[16×16] · B[16×16]` via the SHIPPED BF16 tile op. A/B are padded
/// on K from 16→32 with zeros (the pad contributes nothing), so this is one
/// `bf16_tile_gemm_16x16` — one `TDPBF16PS` tile op on the AMX tier.
fn tile_matmul(a: &[f32; M * M], b: &[f32; M * M]) -> [f32; M * M] {
    let mut a_pad = [0.0f32; M * KPAD];
    let mut b_pad = [0.0f32; KPAD * M];
    for i in 0..M {
        for k in 0..M {
            a_pad[i * KPAD + k] = a[i * M + k]; // A[16×32], cols 16..31 = 0
            b_pad[k * M + i] = b[k * M + i]; // B[32×16], rows 16..31 = 0 (already)
        }
    }
    let (ab, bb) = (to_bf16(&a_pad), to_bf16(&b_pad));
    let mut c = vec![0.0f32; M * M];
    bf16_tile_gemm_16x16(&ab, &bb, &mut c, KPAD);
    let mut out = [0.0f32; M * M];
    out.copy_from_slice(&c);
    out
}

/// Direct `f64` reference GEMM `C = A·B` (16×16).
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

/// `(max_abs_err, frobenius_relative_err)` of a tile-GEMM result vs the f64
/// direct. Frobenius-relative `‖tile − direct‖_F / ‖direct‖_F` is the honest
/// aggregate precision — a per-element max-relative blows up on the near-zero
/// entries that a cancellation-heavy product (e.g. `Mᵀ·Σ·M`) naturally has,
/// which measures the cancellation, not the kernel.
fn errors(tile: &[f32; M * M], direct: &[f64; M * M]) -> (f64, f64) {
    let mut max_abs = 0.0f64;
    let mut num = 0.0f64; // ‖E‖_F²
    let mut den = 0.0f64; // ‖direct‖_F²
    for i in 0..M * M {
        let e = tile[i] as f64 - direct[i];
        max_abs = max_abs.max(e.abs());
        num += e * e;
        den += direct[i] * direct[i];
    }
    (max_abs, (num / den.max(1e-12)).sqrt())
}

fn main() {
    println!("Gridlake field interdependency = BF16 16×16 AMX tile op — measured");
    println!("  tile M=16 (the AMX tile; 64×64 gridlake = 4×4 = 16 of these), K padded 16→32");
    println!(
        "  shipped kernel tier on THIS host: {}  (amx_available = {})\n",
        bf16_tile_gemm_tier(),
        amx_available()
    );

    // ── field tile X: a correlated gridlake patch (smooth → strong interdependency)
    let mut x = [0.0f32; M * M];
    for r in 0..M {
        for c in 0..M {
            x[r * M + c] = 4.0 * (r as f32 * 0.4).sin() * (c as f32 * 0.35).cos() + 0.5 * (r as f32 - c as f32);
        }
    }

    // ── (1) intercorrelation C = Xᵀ·X — the inter-cell coupling as a field op ──
    let xt = transpose(&x);
    let corr_tile = tile_matmul(&xt, &x);
    let corr_direct = direct_matmul(&xt, &x);
    let (ca, cr) = errors(&corr_tile, &corr_direct);
    println!("  (1) intercorrelation  C = Xᵀ·X   (one tile op)");
    println!("      max_abs_err={ca:.4}  frobenius_rel_err={:.3}%  (BF16-precision class)", cr * 100.0);

    // ── (2) covariance sandwich Σ' = Mᵀ·Σ·M — the splat EWA projection shape ──
    // Σ = a symmetric PSD covariance (Xᵀ·X); M = a coupling/projection matrix.
    let sigma_f: [f32; M * M] = {
        let mut s = [0.0f32; M * M];
        for i in 0..M * M {
            s[i] = corr_direct[i] as f32;
        }
        s
    };
    let mut m_proj = [0.0f32; M * M];
    for r in 0..M {
        for c in 0..M {
            m_proj[r * M + c] = if r == c {
                1.0
            } else {
                0.15 * (r as f32 - c as f32).signum()
            };
        }
    }
    // tile path: T = Σ·M, then Σ' = Mᵀ·T  (two tile ops — the sandwich)
    let t_tile = tile_matmul(&sigma_f, &m_proj);
    let mt = transpose(&m_proj);
    let sand_tile = tile_matmul(&mt, &t_tile);
    // direct reference: Mᵀ·Σ·M in f64
    let t_direct = {
        let mut td = [0.0f32; M * M];
        let d = direct_matmul(&sigma_f, &m_proj);
        for i in 0..M * M {
            td[i] = d[i] as f32;
        }
        td
    };
    let sand_direct = direct_matmul(&mt, &t_direct);
    let (sa, sr) = errors(&sand_tile, &sand_direct);
    println!("  (2) covariance sandwich  Σ' = Mᵀ·Σ·M   (two tile ops = the splat EWA projection)");
    println!(
        "      max_abs_err={sa:.4}  frobenius_rel_err={:.3}%  (== hpc::splat3d::sandwich_x16 shape;",
        sr * 100.0
    );
    println!("      per-entry max-rel is meaningless here — Mᵀ·Σ·M cancels to near-zero entries)");

    // ── (3) bit-exactness on bf16-exact integer operands (module guarantee) ──
    let mut ai = [0.0f32; M * M];
    let mut bi = [0.0f32; M * M];
    for i in 0..M * M {
        ai[i] = ((mix(i as u64) % 16) as f32) - 8.0; // small ints, bf16-exact
        bi[i] = ((mix(i as u64 ^ 0x5555) % 16) as f32) - 8.0;
    }
    let int_tile = tile_matmul(&ai, &bi);
    let int_direct = direct_matmul(&ai, &bi);
    let bit_exact = (0..M * M).all(|i| int_tile[i] as f64 == int_direct[i]);
    println!("  (3) bit-exact on bf16-exact integer operands (|Σ| < 2^24): {bit_exact}");

    // ── (4) throughput of the 16×16 tile op on this host ──
    let iters = 200_000usize;
    let t0 = std::time::Instant::now();
    let mut acc = 0.0f32;
    for it in 0..iters {
        // vary the operand slightly so nothing is optimized away
        let mut xv = x;
        xv[0] += (it & 0x7) as f32 * 0.01;
        let c = tile_matmul(&xv, &m_proj);
        acc += c[0] + c[M * M - 1];
    }
    let dt = t0.elapsed().as_secs_f64();
    let tiles_s = iters as f64 / dt;
    println!(
        "  (4) throughput: {iters} tile ops in {:.1} ms → {:.2} M tile-ops/s  (checksum {acc:.1})",
        dt * 1000.0,
        tiles_s / 1e6
    );

    // hard gate: BF16-precision class (rel err small) + integer bit-exactness
    assert!(cr < 0.05, "intercorrelation rel err too high for BF16 class");
    assert!(sr < 0.05, "sandwich rel err too high for BF16 class");
    assert!(bit_exact, "bf16-exact integer tile op is NOT bit-exact vs direct");

    println!(
        "\n  MEASURED CONCLUSION: YES — a gridlake field interdependency reduces to a\n\
         \x20 BF16 16×16 tile op (one `bf16_tile_gemm_16x16`, i.e. one `TDPBF16PS` on\n\
         \x20 the AMX tier), matching the direct f64 reference to BF16 precision and\n\
         \x20 BIT-EXACT for bf16-exact integer operands. It IS the same primitive as:\n\
         \x20   • the codec separable transform (#232: M·X on a tile — WHT/DCT), and\n\
         \x20   • the Gaussian-splat EWA covariance sandwich Σ'=Mᵀ·Σ·M\n\
         \x20     (hpc::splat3d::spd3::sandwich_x16 — the 16-wide batched form).\n\
         \x20 Interdependency / intercorrelation / perturbation / transform / covariance\n\
         \x20 projection are ONE op: the 16×16 BF16 tile GEMM. The 64×64 gridlake is\n\
         \x20 4×4 = 16 of these tiles."
    );
}
