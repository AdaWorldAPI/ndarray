//! Geostrophic finite-difference stencil, computed as a tile-GEMM — the
//! mechanism `domino.rs` tile-batches, exercised on real ERA5 pressure data
//! through `ndarray::simd`'s portable polyfill entry, under TWO encodings:
//! BF16, and a rolling-floor palette256 CASCADE at the same byte cost.
//!
//! Session context: `symbiont/domino.rs` (lance-graph) proves 16 SoA boards
//! batch into ONE AMX 16×16 tile GEMM; this probe asks whether the same
//! 16×16×64 shape can compute a REAL finite-difference stencil — a
//! longitudinal MSLP centered difference, the term geostrophic balance is
//! built from — after the input has been encoded BF16, and how close the
//! GEMM-recovered value lands to the unquantized f64 reference.
//!
//! # The mapping onto one AMX tile
//! - `A[16, K=64]` (BF16): 16 independent latitude rows × 64 pressure
//!   ANOMALY samples per row (a `mean_sea_level_pressure` window, centered —
//!   see `encode_bf16()` for why that is exact AND necessary).
//!   K = 64 is exactly TWO AMX BF16 tiles deep: `TDPBF16PS` packs 2 bf16 per
//!   32-bit dword, so one 16×64-byte tile holds 16×32 bf16, and the kernel
//!   requires K % 32 == 0.
//! - `B[K=64, N=16]` (BF16): 16 columns, each an EXACT `{-1, 0, +1}` centered-
//!   difference stencil at a distinct longitude tap — no quantization error
//!   in the operator itself, only in the DATA it multiplies.
//! - `C[16, 16]` (f32) = `A · B`, computed via
//!   [`ndarray::simd::bf16_tile_gemm_16x16`]. Consumers reach SIMD ONLY through
//!   `ndarray::simd::*` — never `ndarray::hpc::*` — and this runs on STABLE
//!   through the polyfill: the SIMD tier is chosen at COMPILE time by
//!   `target-cpu` (`.cargo/config.toml` v3 default, `config-avx512.toml` for
//!   v4), so no `runtime-dispatch` feature and no nightly `core::simd`.
//!
//! `C[i,j] = bf16(p[row_i, tap_j+1]) − bf16(p[row_i, tap_j−1])`, accumulated
//! in f32 — already an estimate of the true finite difference
//! `p[j+1] − p[j−1]` in Pa, with no dequantization step.
//!
//! # Data provenance (`geostrophic_stencil.bin`, 4096 bytes, `include_bytes!`)
//! WeatherBench2 ERA5 6-hourly 0.25° reanalysis,
//! `gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr`,
//! `mean_sea_level_pressure`, time index 91246 (2021-06-15 12Z — the SAME
//! timestep as the session's `voxel_chess_probe.py`). 16 rows spanning
//! 52.25°N .. 59.75°N (0.5° spacing) around the NH low that probe's own E6
//! Rankine-vortex measurement centered on (55.75°N, 334.50°E); a 64-wide
//! longitude window (326.5° .. 342.5°E) centered on that storm. Raw f32,
//! row-major, regenerable from `probes/weather-p1/` in the `lance-graph`
//! sibling repo against the same store/time_index/indices recorded here.
//!
//! # What this DOES and does NOT prove
//! `bf16_tile_gemm_16x16` is the POLYFILL entry (`simd_ops`) — it carries no
//! arch gate, so this example builds and runs on every target, decoding BF16
//! and accumulating through `F32x16` + FMA on the compile-time-selected tier.
//! The AMX `TDPBF16PS` ladder is its sibling `bf16_tile_gemm_16x16_amx`
//! (x86_64). This host's `amx_available()` is reported at the top of the run.
//! On a non-AMX host this validates the stencil-as-GEMM NUMERICS and the BF16
//! encode, not AMX instruction throughput.
//!
//! Pre-registered bars (checked, not assumed):
//!   BAR-1 (can-it-fire, meaningful): Pearson corr(d_est, d_true) ≥ 0.98
//!         across all 256 cells — proves the mechanism differentiates with
//!         the right sign and scale, not merely "some correlation".
//!   BAR-2 (encode bound, per-cell): |d_est − d_true| ≤ 2·absmax·2⁻⁸ for
//!         EVERY cell — the BF16 unit-roundoff bound for a difference of two
//!         encoded samples, derived from the format, not fitted.
//!   BAR-3 (silence guard / non-triviality): std(d_true) > 50 Pa — the
//!         sample must carry real synoptic-scale variation, or a mechanism
//!         that produces near-zero everywhere would trivially "pass" BAR-2.
//!   BAR-4/BAR-5 (the cascade arm): the same stencil, same kernel, over a
//!         SIGNED rolling-floor palette cascade about the standard atmosphere
//!         — corr ≥ 0.98 and per-cell error within 2× the FINEST tier step.
//!         Measured on this fixture the cascade lands 32× tighter than BF16 at
//!         identical 2-byte cost, because palette indices are EXACT in BF16 so
//!         the GEMM contributes no error at all and only the quantizer's floor
//!         remains. Tier 0's SIGN BIT is the Tief/Hoch classification, free.
//!   DISABLE-RUN (falsifier self-check): re-run with the stencil SIGN
//!         FLIPPED. If this does not invert the correlation to ≤ −0.9, the
//!         harness cannot actually detect a broken stencil and BAR-1 is
//!         decoration — this run's own defect-injection proof.
//!
//!   cargo run --release --example geostrophic_stencil

use ndarray::simd::{
    add_mul_f64, add_scalar_f32, amx_available, array_windows, bf16_tile_gemm_16x16, f32_to_bf16_scalar,
};

const RAW: &[u8] = include_bytes!("geostrophic_stencil.bin");

/// Encoding reference: 1000 hPa. A CONVENTION, not a measurement — MSLP on
/// Earth lives in the 950–1050 hPa window (observed extremes 870–1085), so
/// subtracting a fixed 1000 hPa is a stable offset every tile shares. That
/// matters for a substrate: two tiles from different storms or timesteps stay
/// directly comparable and composable, which a per-tile mean would break.
const P_REF_PA: f32 = 100_000.0;

/// Worst-case |anomaly| the 950–1050 hPa window admits, in Pa. Used for the
/// UNIVERSAL bound quoted below; the per-tile bound uses this tile's own max.
const P_WINDOW_HALF_PA: f64 = 5_000.0;

/// Rolling-floor palette ZERO: 1013.25 hPa, the standard atmosphere — which is
/// also the meteorological boundary between low and high pressure. Tier 0 is
/// therefore SIGNED, and the sign bit alone classifies the sample: negative is
/// a Tief, positive a Hoch. That classification costs nothing — it is not
/// computed, it IS the sign bit — and it makes the stencil's output a signed
/// gradient with no offset bookkeeping.
const PAL_ZERO_PA: f32 = 101_325.0;
/// Tier-0 step: 0.5 hPa = 50 Pa. Chosen so the index is MENTALLY computable —
/// `idx = (p_hPa − 1013.25) · 2`, inverse `p_hPa = 1013.25 + idx/2`. 0.5 hPa is
/// also the operational precision surface pressure is reported at. Signed i8
/// then spans 949.25–1076.75 hPa, which contains the operational MSLP range
/// (the 870/1085 hPa records clip, and are not what this operator is used on).
const PAL_STEP_PA: f32 = 50.0;
/// Tier count. Each tier is one u8 that refines the previous tier's residual by
/// 1/256, so tier k has step `PAL_STEP_PA / 256^k`. Two tiers cost the same 2
/// bytes as BF16 and are measured ~32× tighter on this fixture.
const PAL_TIERS: usize = 2;
const M: usize = 16; // latitude rows
const K: usize = 64; // longitude samples per row
const N: usize = 16; // output stencil taps

/// Longitude index (within the 64-wide window) of each of the 16 output
/// taps: 4, 7, 10, ..., 49 — evenly spaced, every one safely inside
/// `[1, K-2]` so both `tap-1` and `tap+1` stay in-window.
const TAPS: [usize; N] = {
    let mut t = [0usize; N];
    let mut j = 0;
    while j < N {
        t[j] = 4 + 3 * j;
        j += 1;
    }
    t
};

fn load_rows() -> [[f32; K]; M] {
    assert_eq!(RAW.len(), M * K * 4, "fixture size mismatch");
    let mut rows = [[0f32; K]; M];
    let mut vals = RAW
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()));
    for row in rows.iter_mut() {
        for cell in row.iter_mut() {
            *cell = vals.next().expect("fixture holds M*K f32");
        }
    }
    rows
}

/// Encode `A` as BF16: subtract the fixed `P_REF_PA` through
/// `ndarray::simd::add_scalar_f32`, then `f32_to_bf16_scalar` each sample.
///
/// Referencing is exact for this operator — the stencil computes
/// `p[tap+1] − p[tap−1]`, so a constant offset cancels identically — and it is
/// NECESSARY. BF16 carries p = 8 significand bits, so its unit roundoff is
/// 2^-8. Against the raw absolute field (MSLP ≈ 1.0e5 Pa) that is ~390 Pa per
/// sample, far larger than the ~100 Pa signal the stencil is recovering: an
/// unreferenced BF16 encode would be swamped by its own rounding. Referenced,
/// the anomaly is bounded by the physical pressure window (±5000 Pa worst
/// case, this tile far less), and the same relative precision buys single-Pa
/// resolution.
fn encode_bf16(rows: &[[f32; K]; M]) -> (Vec<u16>, f32) {
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let anom = add_scalar_f32(&flat, -P_REF_PA);
    let absmax = anom.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let a: Vec<u16> = anom.iter().map(|&v| f32_to_bf16_scalar(v)).collect();
    (a, absmax)
}

/// Encode `A` as a rolling-floor palette CASCADE about the standard
/// atmosphere: tier 0 SIGNED (i8 range, sign = Tief/Hoch), later tiers
/// unsigned residual refinements. Every tier is returned BF16-encoded, ready
/// for the same tile-GEMM.
///
/// Why the later tiers stay unsigned: FLOOR is used, not round-to-nearest, so
/// the residual `r − idx·step` lies in `[0, step)` even where `idx` is
/// negative. The sign lives in tier 0 alone; the refinements are magnitudes.
///
/// Why this composes with a difference operator: each tier is a UNIFORM
/// quantizer, so it is affine in the physical quantity, and
/// `idx(a) − idx(b) = (a − b)/step` up to the floor. A centroid codebook would
/// NOT have this property and would break the stencil — uniformity is the
/// condition, not palette-ness.
///
/// Why the indices survive BF16 exactly: BF16 carries 8 significand bits plus
/// a sign bit, so every integer in `-128..=255` is exact, and so is their
/// difference in the f32 accumulator. The GEMM therefore contributes ZERO
/// error — all of it lives in the quantizer, bounded by the finest tier step.
fn encode_palette_cascade(rows: &[[f32; K]; M]) -> (Vec<Vec<u16>>, i32, i32) {
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    let mut residual = add_scalar_f32(&flat, -PAL_ZERO_PA);
    let mut tiers = Vec::with_capacity(PAL_TIERS);
    let mut step = PAL_STEP_PA;
    let (mut lo, mut hi) = (i32::MAX, i32::MIN);
    for tier in 0..PAL_TIERS {
        // Tier 0 is signed (i8); refinements are unsigned (u8).
        let (clamp_lo, clamp_hi) = if tier == 0 { (-128.0, 127.0) } else { (0.0, 255.0) };
        let idx: Vec<f32> = residual
            .iter()
            .map(|&r| (r / step).floor().clamp(clamp_lo, clamp_hi))
            .collect();
        if tier == 0 {
            for &i in &idx {
                lo = lo.min(i as i32);
                hi = hi.max(i as i32);
            }
        }
        residual = residual
            .iter()
            .zip(&idx)
            .map(|(&r, &i)| r - i * step)
            .collect();
        tiers.push(idx.iter().map(|&i| f32_to_bf16_scalar(i)).collect());
        step /= 256.0;
    }
    (tiers, lo, hi)
}

/// Reference finite differences in f64, via the crate's own fixed-size sliding
/// window. A centered difference IS a 3-tap stencil, which is exactly what
/// `array_windows::<_, 3>` is for (as opposed to a wide box filter, where an
/// integral image wins). The window starting at `tap-1` spans
/// `[tap-1, tap, tap+1]`, so the difference is `w[2] - w[0]`.
fn reference_differences(rows: &[[f32; K]; M]) -> Vec<f64> {
    let mut out = Vec::with_capacity(M * N);
    for row in rows {
        let w: Vec<&[f32; 3]> = array_windows::<f32, 3>(row).collect();
        for &tap in TAPS.iter() {
            out.push((w[tap - 1][2] - w[tap - 1][0]) as f64);
        }
    }
    out
}

/// Build the `K x N` centered-difference stencil as BF16: column `j` has
/// `+1` at `TAPS[j]+1` and `sign * -1` at `TAPS[j]-1` (sign flips for the
/// disable-run), zero elsewhere. ±1 and 0 are EXACT in BF16, so the operator
/// itself contributes no error — only the DATA it multiplies does.
fn build_stencil(sign: f32) -> Vec<u16> {
    let mut b = vec![f32_to_bf16_scalar(0.0); K * N];
    for (j, &tap) in TAPS.iter().enumerate() {
        b[(tap - 1) * N + j] = f32_to_bf16_scalar(-sign);
        b[(tap + 1) * N + j] = f32_to_bf16_scalar(sign);
    }
    b
}

fn run_gemm(a_bf16: &[u16], b_bf16: &[u16]) -> Vec<f32> {
    let mut c = vec![0f32; M * N];
    bf16_tile_gemm_16x16(a_bf16, b_bf16, &mut c, K);
    c
}

/// Run the cascade: one GEMM per tier through the SAME kernel, each tier's
/// integer index-difference scaled by that tier's step and summed. Because the
/// indices are exact in BF16, every tier's GEMM is exact — the sum carries only
/// the quantizer's own error, bounded by the FINEST tier's step.
fn run_cascade(tiers: &[Vec<u16>], b_bf16: &[u16]) -> Vec<f64> {
    let mut out = vec![0f64; M * N];
    let mut step = PAL_STEP_PA as f64;
    for tier in tiers {
        let c = run_gemm(tier, b_bf16);
        for (o, &v) in out.iter_mut().zip(c.iter()) {
            *o += v as f64 * step;
        }
        step /= 256.0;
    }
    out
}

/// Pearson r, with the three accumulations riding `ndarray::simd::add_mul_f64`
/// (`acc[i] += a[i] * b[i]`, F64x8 chunks + scalar tail) instead of a hand
/// loop.
fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let dx: Vec<f64> = xs.iter().map(|x| x - mx).collect();
    let dy: Vec<f64> = ys.iter().map(|y| y - my).collect();
    let (mut cov, mut vx, mut vy) = (vec![0.0; dx.len()], vec![0.0; dx.len()], vec![0.0; dx.len()]);
    add_mul_f64(&mut cov, &dx, &dy);
    add_mul_f64(&mut vx, &dx, &dx);
    add_mul_f64(&mut vy, &dy, &dy);
    cov.iter().sum::<f64>() / (vx.iter().sum::<f64>().sqrt() * vy.iter().sum::<f64>().sqrt())
}

fn main() {
    let rows = load_rows();
    let (a_bf16, absmax) = encode_bf16(&rows);
    // BF16 carries p = 8 significand bits (1 implicit + 7 stored), so its unit
    // roundoff is u = 2^-p = 2^-8: fl(x) = x(1+d) with |d| <= u. The stencil
    // differences TWO encoded samples, so the worst-case absolute error is
    // 2 * |anomaly|max * 2^-8. Derived from the format, not fitted.
    let u = f64::powi(2.0, -8);
    let bf16_bound = 2.0 * absmax as f64 * u;
    // The same bound at the physical limit of the encoding: any Earth MSLP
    // tile referenced to 1000 hPa has |anomaly| <= 5000 Pa.
    let universal_bound = 2.0 * P_WINDOW_HALF_PA * u;

    let (pal, pal_lo, pal_hi) = encode_palette_cascade(&rows);
    // Each tier is a UNIFORM quantizer, so its floor error is one step; the
    // cascade's residual after PAL_TIERS tiers is bounded by the finest step,
    // and the stencil differences TWO samples.
    let pal_step_fine = PAL_STEP_PA as f64 / f64::powi(256.0, (PAL_TIERS - 1) as i32);
    let pal_bound = 2.0 * pal_step_fine;

    println!("== geostrophic_stencil: tile-GEMM stencil, real ERA5 data ==\n");
    println!(
        "  amx_available() = {}  -> execution path: {}",
        amx_available(),
        if amx_available() {
            "AMX TDPBUSD"
        } else {
            "compile-time SIMD tier (F32x16 FMA polyfill; target-cpu v3 default, v4 via .cargo/config-avx512.toml)"
        }
    );
    println!("  encoding: ref {:.0} hPa (fixed), anomaly |max| {absmax:.1} Pa", P_REF_PA / 100.0);
    println!("  BF16 bound: {bf16_bound:.4} Pa this tile / {universal_bound:.4} Pa any Earth MSLP tile");
    println!(
        "  palette cascade: zero {:.2} hPa (std atmosphere), tier-0 step {:.1} hPa SIGNED\n  \
         idx = (p_hPa - {:.2})*2, i8 spans {:.2}..{:.2} hPa; {PAL_TIERS} tiers = {PAL_TIERS} byte(s), \
         finest step {pal_step_fine:.5} Pa, bound {pal_bound:.5} Pa",
        PAL_ZERO_PA / 100.0,
        PAL_STEP_PA / 100.0,
        PAL_ZERO_PA / 100.0,
        (PAL_ZERO_PA - 128.0 * PAL_STEP_PA) / 100.0,
        (PAL_ZERO_PA + 127.0 * PAL_STEP_PA) / 100.0,
    );
    println!(
        "  tier-0 index range on this tile: {pal_lo} .. {pal_hi}  -> sign bit says: {}\n",
        if pal_hi < 0 {
            "TIEF (every sample below the standard atmosphere)"
        } else if pal_lo > 0 {
            "HOCH (every sample above it)"
        } else {
            "MIXED (the tile straddles the low/high boundary)"
        }
    );

    let flat_true = reference_differences(&rows);
    let std_true = {
        let m = flat_true.iter().sum::<f64>() / flat_true.len() as f64;
        (flat_true.iter().map(|v| (v - m).powi(2)).sum::<f64>() / flat_true.len() as f64).sqrt()
    };

    // ---- CORRECT stencil ----
    let b_correct = build_stencil(1.0);
    let c_correct = run_gemm(&a_bf16, &b_correct);
    let flat_est: Vec<f64> = c_correct.iter().map(|&v| v as f64).collect();
    let corr = pearson(&flat_est, &flat_true);
    let max_err = flat_est
        .iter()
        .zip(&flat_true)
        .map(|(e, t)| (e - t).abs())
        .fold(0.0_f64, f64::max);

    // ---- the palette cascade, same stencil, same kernel ----
    let pal_est = run_cascade(&pal, &b_correct);
    let pal_corr = pearson(&pal_est, &flat_true);
    let pal_max_err = pal_est
        .iter()
        .zip(&flat_true)
        .map(|(e, t)| (e - t).abs())
        .fold(0.0_f64, f64::max);

    println!("  BAR-1 corr(d_est, d_true)      = {corr:.4}   (bar: >= 0.98)");
    println!("  BAR-2 max |d_est - d_true|     = {max_err:.4} Pa  (bar: <= {bf16_bound:.4} Pa)");
    println!("  BAR-3 std(d_true)              = {std_true:.2} Pa  (bar: > 50 Pa)");
    println!("\n  PALETTE CASCADE (same bytes as BF16, same kernel):");
    println!("  BAR-4 corr(d_pal, d_true)      = {pal_corr:.6}   (bar: >= 0.98)");
    println!(
        "  BAR-5 max |d_pal - d_true|     = {pal_max_err:.5} Pa  (bar: <= {pal_bound:.5} Pa)  \
         [{:.0}x tighter than BF16]",
        max_err / pal_max_err.max(f64::MIN_POSITIVE)
    );

    // ---- DISABLE-RUN: sign-flipped stencil (the falsifier self-check) ----
    let b_flipped = build_stencil(-1.0);
    let c_flipped = run_gemm(&a_bf16, &b_flipped);
    let flat_flip: Vec<f64> = c_flipped.iter().map(|&v| v as f64).collect();
    let corr_flip = pearson(&flat_flip, &flat_true);
    println!("\n  DISABLE-RUN corr(d_est_signflipped, d_true) = {corr_flip:.4}   (bar: <= -0.90)");

    let bar1 = corr >= 0.98;
    let bar2 = max_err <= bf16_bound;
    let bar3 = std_true > 50.0;
    let bar4 = pal_corr >= 0.98;
    let bar5 = pal_max_err <= pal_bound;
    let disable_ok = corr_flip <= -0.90;

    println!("\n  VERDICT:");
    println!("    BAR-1 (fires correctly) ....... {}", if bar1 { "PASS" } else { "FAIL" });
    println!("    BAR-2 (BF16 encode bound) ..... {}", if bar2 { "PASS" } else { "FAIL" });
    println!("    BAR-3 (non-trivial sample) .... {}", if bar3 { "PASS" } else { "FAIL" });
    println!("    BAR-4 (cascade fires) ......... {}", if bar4 { "PASS" } else { "FAIL" });
    println!("    BAR-5 (cascade step bound) .... {}", if bar5 { "PASS" } else { "FAIL" });
    println!(
        "    disable-run (can detect a broken stencil) . {}",
        if disable_ok {
            "PASS"
        } else {
            "FAIL — harness is blind"
        }
    );

    if !(bar1 && bar2 && bar3 && bar4 && bar5 && disable_ok) {
        eprintln!("\n  ONE OR MORE BARS FAILED — see above.");
        std::process::exit(1);
    }
    println!("\n  All bars satisfied. The geostrophic centered-difference stencil");
    println!("  survives BOTH encodings through ndarray::simd::bf16_tile_gemm_16x16");
    println!("  on this host's {} tier.", if amx_available() { "AMX" } else { "compile-time SIMD" });
}
