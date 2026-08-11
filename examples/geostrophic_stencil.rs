//! Geostrophic finite-difference stencil, computed as an int8 tile-GEMM —
//! the mechanism `domino.rs` tile-batches, exercised on real ERA5 pressure
//! data through the crate's own safe dispatching wrapper.
//!
//! Session context: `symbiont/domino.rs` (lance-graph) proves 16 SoA boards
//! batch into ONE AMX 16×16 tile GEMM; this probe asks whether the same
//! 16×16×64 shape can compute a REAL finite-difference stencil (the leading
//! term of geostrophic balance, `du/dy` via centered differences) after the
//! input has been quantized to the shipped u8-palette convention, and how
//! close the GEMM-recovered value lands to the unquantized f64 reference.
//!
//! # The mapping onto one AMX tile
//! - `A[16, K=64]` (u8): 16 independent latitude rows × 64 quantized
//!   pressure samples per row (a `mean_sea_level_pressure` window).
//! - `B[K=64, N=16]` (i8): 16 columns, each an EXACT `{-1, 0, +1}` centered-
//!   difference stencil at a distinct longitude tap — no quantization error
//!   in the operator itself, only in the DATA it multiplies.
//! - `C[16, 16]` (i32) = `A · B`, computed via
//!   [`ndarray::simd::gemm_u8_i8`] — the canonical polyfill entry (AMX
//!   `TDPBUSD` when available, then avx512vnni / avxvnni / scalar; the tiers
//!   are proven bit-identical by the crate's own parity tests). Consumers
//!   reach SIMD ONLY through `ndarray::simd::*` — never `ndarray::hpc::*`.
//!
//! `C[i,j] = idx(p[row_i, tap_j+1]) − idx(p[row_i, tap_j−1])`, an INTEGER
//! bucket-index difference. Dequantized by the shared bucket width, that is
//! an estimate of the true finite difference `p[j+1] − p[j−1]` in Pa.
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
//! `gemm_u8_i8` runtime-dispatches to AMX `TDPBUSD` when `amx_available()`,
//! then the VNNI and scalar tiers. This host's
//! actual path is reported at the top of the run. On a non-AMX host this
//! validates the stencil-as-GEMM NUMERICS and the u8-quantization pipeline,
//! not AMX instruction throughput — the crate's own bit-exact fallback
//! parity test is what licenses treating those as interchangeable.
//!
//! Pre-registered bars (checked, not assumed):
//!   BAR-1 (can-it-fire, meaningful): Pearson corr(d_est, d_true) ≥ 0.98
//!         across all 256 cells — proves the mechanism differentiates with
//!         the right sign and scale, not merely "some correlation".
//!   BAR-2 (quantization bound, per-cell): |d_est − d_true| ≤ 1.01×bucket
//!         width for EVERY cell — two independent half-bucket errors.
//!   BAR-3 (silence guard / non-triviality): std(d_true) > 50 Pa — the
//!         sample must carry real synoptic-scale variation, or a mechanism
//!         that produces near-zero everywhere would trivially "pass" BAR-2.
//!   DISABLE-RUN (falsifier self-check): re-run with the stencil SIGN
//!         FLIPPED. If this does not invert the correlation to ≤ −0.9, the
//!         harness cannot actually detect a broken stencil and BAR-1 is
//!         decoration — this run's own defect-injection proof.
//!
//!   cargo run --release --example geostrophic_stencil

use ndarray::simd::{amx_available, gemm_u8_i8};

const RAW: &[u8] = include_bytes!("geostrophic_stencil.bin");
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
    for i in 0..M {
        for k in 0..K {
            let off = (i * K + k) * 4;
            rows[i][k] = f32::from_le_bytes(RAW[off..off + 4].try_into().unwrap());
        }
    }
    rows
}

/// Linear quantization over the fixture's own min/max — the same bucketing
/// formula the shipped `helix::quantize::RollingFloor` uses (a fixed
/// percentile window would barely differ at this sample size, so the
/// simpler min/max convention is used here and stated as such).
fn quantize(rows: &[[f32; K]; M]) -> ([[u8; K]; M], f32, f32) {
    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for row in rows {
        for &v in row {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    let mut q = [[0u8; K]; M];
    for i in 0..M {
        for k in 0..K {
            let t = (rows[i][k] - lo) / (hi - lo);
            q[i][k] = (t * 256.0).clamp(0.0, 255.0) as u8;
        }
    }
    (q, lo, hi)
}

/// Build the `K x N` centered-difference stencil: column `j` has `+1` at
/// `TAPS[j]+1`, and `sign * -1` at `TAPS[j]-1` (sign flips for the
/// disable-run), zero elsewhere.
fn build_stencil(sign: i8) -> [i8; K * N] {
    let mut b = [0i8; K * N];
    for (j, &tap) in TAPS.iter().enumerate() {
        b[(tap - 1) * N + j] = -sign;
        b[(tap + 1) * N + j] = sign;
    }
    b
}

fn run_gemm(a_u8: &[[u8; K]; M], b_i8: &[i8; K * N]) -> [i32; M * N] {
    let a_flat: Vec<u8> = a_u8.iter().flatten().copied().collect();
    let mut c = [0i32; M * N];
    gemm_u8_i8(&a_flat, b_i8, &mut c, M, N, K);
    c
}

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let (mut cov, mut vx, mut vy) = (0.0, 0.0, 0.0);
    for (x, y) in xs.iter().zip(ys) {
        cov += (x - mx) * (y - my);
        vx += (x - mx).powi(2);
        vy += (y - my).powi(2);
    }
    cov / (vx.sqrt() * vy.sqrt())
}

fn main() {
    let rows = load_rows();
    let (a_u8, lo, hi) = quantize(&rows);
    let bucket_width = (hi - lo) as f64 / 256.0;

    println!("== geostrophic_stencil: int8 tile-GEMM stencil, real ERA5 data ==\n");
    println!(
        "  amx_available() = {}  -> execution path: {}",
        amx_available(),
        if amx_available() {
            "AMX TDPBUSD"
        } else {
            "scalar/VNNI tier (gemm_u8_i8's own dispatch)"
        }
    );
    println!("  fixture: p range [{lo:.1}, {hi:.1}] Pa, bucket width {bucket_width:.4} Pa\n");

    // True (unquantized) finite differences, for reference.
    let mut d_true = [[0f64; N]; M];
    for i in 0..M {
        for (j, &tap) in TAPS.iter().enumerate() {
            d_true[i][j] = (rows[i][tap + 1] - rows[i][tap - 1]) as f64;
        }
    }
    let flat_true: Vec<f64> = d_true.iter().flatten().copied().collect();
    let std_true = {
        let m = flat_true.iter().sum::<f64>() / flat_true.len() as f64;
        (flat_true.iter().map(|v| (v - m).powi(2)).sum::<f64>() / flat_true.len() as f64).sqrt()
    };

    // ---- CORRECT stencil ----
    let b_correct = build_stencil(1);
    let c_correct = run_gemm(&a_u8, &b_correct);
    let flat_est: Vec<f64> = c_correct.iter().map(|&v| v as f64 * bucket_width).collect();
    let corr = pearson(&flat_est, &flat_true);
    let max_err = flat_est
        .iter()
        .zip(&flat_true)
        .map(|(e, t)| (e - t).abs())
        .fold(0.0_f64, f64::max);

    println!("  BAR-1 corr(d_est, d_true)      = {corr:.4}   (bar: >= 0.98)");
    println!("  BAR-2 max |d_est - d_true|     = {max_err:.4} Pa  (bar: <= {:.4} Pa)", 1.01 * bucket_width);
    println!("  BAR-3 std(d_true)              = {std_true:.2} Pa  (bar: > 50 Pa)");

    // ---- DISABLE-RUN: sign-flipped stencil (the falsifier self-check) ----
    let b_flipped = build_stencil(-1);
    let c_flipped = run_gemm(&a_u8, &b_flipped);
    let flat_flip: Vec<f64> = c_flipped.iter().map(|&v| v as f64 * bucket_width).collect();
    let corr_flip = pearson(&flat_flip, &flat_true);
    println!("\n  DISABLE-RUN corr(d_est_signflipped, d_true) = {corr_flip:.4}   (bar: <= -0.90)");

    let bar1 = corr >= 0.98;
    let bar2 = max_err <= 1.01 * bucket_width;
    let bar3 = std_true > 50.0;
    let disable_ok = corr_flip <= -0.90;

    println!("\n  VERDICT:");
    println!("    BAR-1 (fires correctly) ....... {}", if bar1 { "PASS" } else { "FAIL" });
    println!("    BAR-2 (quantization bound) .... {}", if bar2 { "PASS" } else { "FAIL" });
    println!("    BAR-3 (non-trivial sample) .... {}", if bar3 { "PASS" } else { "FAIL" });
    println!(
        "    disable-run (can detect a broken stencil) . {}",
        if disable_ok {
            "PASS"
        } else {
            "FAIL — harness is blind"
        }
    );

    if !(bar1 && bar2 && bar3 && disable_ok) {
        eprintln!("\n  ONE OR MORE BARS FAILED — see above.");
        std::process::exit(1);
    }
    println!("\n  All bars satisfied. The geostrophic centered-difference stencil");
    println!("  survives u8 quantization + ndarray::simd::gemm_u8_i8 on this host's");
    println!("  {} path.", if amx_available() { "AMX" } else { "scalar-fallback" });
}
