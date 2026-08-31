//! W1 — the parity bridge (census F-4).
//!
//! `ndarray::…::signature_d2_deg3` (hardware: f32, fixed d=2/deg-3) and
//! `sigker::signature_truncated` (reference: f64, any d/depth) implement the
//! same iterated integrals in two repos with, until this file, zero
//! cross-checks.
//!
//! # Layout correspondence (the thing that makes the comparison meaningful)
//!
//! ndarray returns one flat `[f32; 15]`:
//! `[s0, s1x, s1y, s2xx, s2xy, s2yx, s2yy, s3xxx, s3xxy, s3xyx, s3xyy,
//!   s3yxx, s3yxy, s3yyx, s3yyy]`.
//!
//! sigker returns per-level flat storage, `levels[k]` of length `d^k`,
//! row-major in the index tuple. For d = 2 that is `[1]`, `[x, y]`,
//! `[xx, xy, yx, yy]`, `[xxx, xxy, xyx, xyy, yxx, yxy, yyx, yyy]` — the same
//! order, concatenated. So the comparison is index-for-index, and
//! `concat_levels` asserts the total length is 15 rather than assuming it.
//!
//! # Why the gate is normalized by LEVEL, not by coefficient
//!
//! The obvious gate — per-coefficient relative error — is the cancellation
//! trap this plan's §6 already names as law, one level down from where it was
//! first met (the D-SK kernel-scalar finding). Measured by
//! `examples/w1_sweep.rs` over 1000 paths at each length:
//!
//! ```text
//!      N  worst |abs|   worst /coeff  worst /levelmax
//!     16     8.741e-7        2.010e2         6.452e-6
//!     32     3.865e-6        1.827e3         2.439e-6
//!     64     1.727e-5        7.911e2         4.154e-6
//!    128     7.587e-5        7.779e2         4.070e-6
//!    256     2.988e-4        7.926e3         9.032e-6
//! ```
//!
//! Per-coefficient relative error swings over 2e2..8e3 with no trend — a
//! level-3 coefficient of a random walk passes through zero by cancellation,
//! and relative error against a near-zero denominator is unbounded no matter
//! how correct the implementation is. Absolute error is no better: it grows
//! ~N² with the signature's own scale. Normalized by the characteristic
//! magnitude of the coefficient's OWN LEVEL, the error is flat at
//! 2.4e-6..9.0e-6 across a 16x range of path length — that is a property of
//! the implementation, so that is what the gate binds on.
//!
//! `REL_TOL = 1e-4` therefore sits ~11x above the worst measured value. It is
//! a pre-registered bound with margin, not a fitted line; the sweep is
//! committed alongside so a future tightening has its evidence.
//!
//! And the formula itself is EXACT: on a single segment (closed form, zero
//! accumulation) the two implementations agree to the last f32 bit — see
//! `examples/w1_diagnose.rs`. What this gate bounds is f32 accumulation
//! drift, nothing else.

use ndarray::hpc::pillar::signature::{signature_d2_deg3, SIG_D2_DEG3_LEN};
use sigker::signature_truncated;

const N_PATHS: usize = 1000;
const N_POINTS: usize = 64;
/// Error bound, normalized by the characteristic magnitude of each
/// coefficient's own level. Pre-registered from the sweep above (worst
/// measured 9.03e-6 at N=256) with ~11x margin.
const REL_TOL: f64 = 1e-4;
/// Which signature level each of the 15 coefficients belongs to.
const LEVEL: [usize; SIG_D2_DEG3_LEN] = [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3];

/// Deterministic path generator — a tiny xorshift so the fixture needs no
/// dev-dep and reproduces byte-for-byte across runs and machines.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40; // 24 bits
        (v as f32 / (1u32 << 24) as f32) - 0.5
    }
}

/// One random d=2 walk, in both repos' input shapes.
fn make_path(rng: &mut Rng, n_points: usize) -> (Vec<f32>, Vec<Vec<f64>>) {
    let mut flat = Vec::with_capacity(n_points * 2);
    let mut pts = Vec::with_capacity(n_points);
    let (mut x, mut y) = (0.0f32, 0.0f32);
    for _ in 0..n_points {
        flat.push(x);
        flat.push(y);
        pts.push(vec![x as f64, y as f64]);
        x += rng.next_f32();
        y += rng.next_f32();
    }
    (flat, pts)
}

/// sigker's per-level storage, flattened into ndarray's single-array order.
fn concat_levels(sig: &sigker::Signature) -> Vec<f64> {
    let out: Vec<f64> = sig.levels.iter().flat_map(|l| l.iter().copied()).collect();
    assert_eq!(
        out.len(),
        SIG_D2_DEG3_LEN,
        "layout correspondence broken: sigker d=2/depth=3 must flatten to \
         exactly the {SIG_D2_DEG3_LEN} coefficients ndarray returns"
    );
    out
}

/// Worst level-normalized error over `n_paths` random walks.
///
/// The denominator is the largest reference coefficient WITHIN the same
/// level, so a coefficient that cancels toward zero is measured against the
/// scale its level actually carries rather than against its own vanishing
/// magnitude.
fn worst_level_error(n_paths: usize, seed: u64) -> (f64, usize) {
    let mut rng = Rng(seed);
    let mut worst = 0.0f64;
    let mut worst_idx = 0usize;
    for _ in 0..n_paths {
        let (flat, pts) = make_path(&mut rng, N_POINTS);
        let hw = signature_d2_deg3(&flat, N_POINTS);
        let refr = concat_levels(&signature_truncated(&pts, 3));
        for (i, (&h, &r)) in hw.iter().zip(refr.iter()).enumerate() {
            let err = (h as f64 - r).abs() / level_scale(&refr, LEVEL[i]);
            if err > worst {
                worst = err;
                worst_idx = i;
            }
        }
    }
    (worst, worst_idx)
}

/// Characteristic magnitude of one signature level, from the f64 reference.
fn level_scale(refr: &[f64], level: usize) -> f64 {
    refr.iter()
        .enumerate()
        .filter(|(i, _)| LEVEL[*i] == level)
        .map(|(_, v)| v.abs())
        .fold(0.0f64, f64::max)
        .max(1e-12)
}

#[test]
fn hardware_signature_matches_the_sigker_reference() {
    let (worst, idx) = worst_level_error(N_PATHS, 0x9E37_79B9_7F4A_7C15);
    assert!(
        worst < REL_TOL,
        "W1 parity FAILED: worst level-normalized error {worst:.3e} \
         (coefficient index {idx}) exceeds {REL_TOL:.0e} over {N_PATHS} paths"
    );
}

/// The margin is REPORTED, not merely asserted — a gate whose measured value
/// is invisible cannot be re-tightened later, and a silent 100x margin is
/// indistinguishable from a test that compares nothing.
#[test]
fn parity_margin_is_reported() {
    let (worst, idx) = worst_level_error(N_PATHS, 0xD1B5_4A32_D192_ED03);
    println!(
        "W1 parity margin: worst level-normalized err {worst:.3e} at coefficient {idx}, \
         bound {REL_TOL:.0e}, margin {:.1}x",
        REL_TOL / worst.max(f64::MIN_POSITIVE)
    );
    assert!(worst < REL_TOL, "second seed must hold the same bound");
}

// ════════════════════════════════════════════════════════════════════════════
// Anti-vacuity half — the gate must be able to FAIL.
//
// A parity assertion is only evidence if some reachable implementation
// violates it. `sabotaged_signature` is the same Chen accumulation with ONE
// term removed: the `½ dx_i dx_j` self-term in the level-2 update (and its
// level-3 companions). That is the single most plausible way to get this
// wrong — it is exactly the term that distinguishes the true iterated
// integral from the naive "outer product of increments" — and dropping it
// still produces finite, same-magnitude, plausible-looking coefficients.
// If REL_TOL still passed on that, the test above would be measuring nothing.
// ════════════════════════════════════════════════════════════════════════════

fn sabotaged_signature(path: &[f32], n_points: usize) -> [f32; SIG_D2_DEG3_LEN] {
    let (mut s1x, mut s1y) = (0.0f32, 0.0f32);
    let (mut s2xx, mut s2xy, mut s2yx, mut s2yy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut s3 = [0.0f32; 8];
    for k in 0..n_points - 1 {
        let dx = path[2 * k + 2] - path[2 * k];
        let dy = path[2 * k + 3] - path[2 * k + 1];
        // Level 3 — unchanged from the real update.
        s3[0] += s2xx * dx + 0.5 * s1x * dx * dx + (1.0 / 6.0) * dx * dx * dx;
        s3[1] += s2xx * dy + 0.5 * s1x * dx * dy + (1.0 / 6.0) * dx * dx * dy;
        s3[2] += s2xy * dx + 0.5 * s1x * dy * dx + (1.0 / 6.0) * dx * dy * dx;
        s3[3] += s2xy * dy + 0.5 * s1x * dy * dy + (1.0 / 6.0) * dx * dy * dy;
        s3[4] += s2yx * dx + 0.5 * s1y * dx * dx + (1.0 / 6.0) * dy * dx * dx;
        s3[5] += s2yx * dy + 0.5 * s1y * dx * dy + (1.0 / 6.0) * dy * dx * dy;
        s3[6] += s2yy * dx + 0.5 * s1y * dy * dx + (1.0 / 6.0) * dy * dy * dx;
        s3[7] += s2yy * dy + 0.5 * s1y * dy * dy + (1.0 / 6.0) * dy * dy * dy;
        // THE SABOTAGE: the `+ 0.5 * d_i * d_j` self-term is dropped here.
        s2xx += s1x * dx;
        s2xy += s1x * dy;
        s2yx += s1y * dx;
        s2yy += s1y * dy;
        s1x += dx;
        s1y += dy;
    }
    [1.0, s1x, s1y, s2xx, s2xy, s2yx, s2yy, s3[0], s3[1], s3[2], s3[3], s3[4], s3[5], s3[6], s3[7]]
}

#[test]
fn a_wrong_chen_accumulation_is_caught_by_the_same_bound() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut worst = 0.0f64;
    for _ in 0..N_PATHS {
        let (flat, pts) = make_path(&mut rng, N_POINTS);
        let bad = sabotaged_signature(&flat, N_POINTS);
        let refr = concat_levels(&signature_truncated(&pts, 3));
        for (i, (&h, &r)) in bad.iter().zip(refr.iter()).enumerate() {
            worst = worst.max((h as f64 - r).abs() / level_scale(&refr, LEVEL[i]));
        }
    }
    assert!(
        worst > REL_TOL,
        "ANTI-VACUITY FAILED: the sabotaged accumulation passed the parity \
         bound (worst level-normalized err {worst:.3e} < {REL_TOL:.0e}) — the \
         gate above therefore proves nothing"
    );
    println!(
        "W1 anti-vacuity: sabotaged accumulation reaches level-normalized err \
         {worst:.3e}, {:.0}x the bound {REL_TOL:.0e}",
        worst / REL_TOL
    );
}
