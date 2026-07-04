//! Residue upscaling — the q2 anatomy technique applied to a codec residual.
//!
//! The point (the operator's, 2026-07-04): the q2 FMA-anatomy pipeline upscales
//! **2000 surfels / 100k nodes → 4,000,000 vertices** by storing sparse anchors
//! with a *deterministic, regenerable* golden-spiral placement and coding only
//! the **residue** — the deviation the deterministic placement does not capture.
//! (`crates/helix`: "HHTL is the deterministic PLACE; helix is the RESIDUE …
//! 8K resolution at Super-8 cost — the curve is regenerated from a φ-spiral
//! template, not stored; the cost is only the endpoint pair.")
//!
//! M1 proved motion estimation *is* the shader's i8 GEMM and MC is bit-exact, but
//! it stored the residual **densely** (per-cell Delta, the measured ~3-byte
//! floor). This probe asks the operator's real question: can we get the residue
//! *cheaper* the way anatomy gets vertices cheaper — sparse anchors + a
//! deterministic FMA extrapolation as the PLACE, and only the small
//! residue-of-extrapolation coded?
//!
//! # Honest framing (no strawman, no Frankenstein)
//!
//! - **Baseline is entropy-coded, not raw bytes.** A real codec (rANS) approaches
//!   the order-0 Shannon entropy `H₀` of the residual. So the baseline here is
//!   `H₀(residual)·N/8` bytes — the bits an *ideal* entropy coder spends — not a
//!   naive 1 byte/cell. Beating that is a real win, not an accounting trick.
//! - **The mechanism is decorrelation.** Sparse-anchor FMA extrapolation is a
//!   *predictor* (like DPCM prediction or the HEVC transform): if it captures the
//!   field's low-frequency structure, `H₀(field − prediction) < H₀(field)`, so
//!   even an ideal coder spends fewer bits. The anchors cost extra; the win is
//!   real iff the entropy drop exceeds the anchor cost.
//! - **Anchor POSITIONS are free.** Both a regular grid (stride known) and the
//!   2-D golden-spiral sunflower (`k → (√((k+½)/A), k·golden_angle)`, regenerable)
//!   have *deterministic* positions — never stored, exactly the helix
//!   "template is free, only the endpoint costs" principle. Only anchor VALUES cost.
//! - **Scope / what this is NOT.** The anatomy residue is a *direction field on S²*
//!   where the golden-spiral RVQ (`hpc::splat3d::helix_orient`, `helix::Signed360`)
//!   is near-optimal; a codec luma residual is a *scalar field*, so the S² encoder
//!   does NOT apply and is deliberately not forced onto it (Frankenstein guard).
//!   What transfers is dimension-general: place/residue decorrelation + free
//!   deterministic anchor placement. The golden-spiral enters here as the 2-D
//!   *anchor placement* (the surfel distribution), not as an S² angle codec.
//!
//! # What it measures
//!
//! Over a smoothness sweep of residual-shaped fields, three codings, in bytes an
//! ideal entropy coder would spend:
//!   1. **dense**  — `H₀(field)` (the codec's current residual, entropy-coded).
//!   2. **grid**   — `A` grid anchors + bilinear FMA extrapolation → `H₀(residue)`.
//!   3. **spiral** — `A` golden-spiral anchors + IDW FMA extrapolation → `H₀(residue)`.
//!
//! Both upscalers reconstruct losslessly (residue is added back exact). The
//! per-cell residue also classifies into the shipped `hpc::codec::CellMode`
//! (Skip/Delta/Escape) — the codec's taxonomy IS the residue decision (H-7).
//!
//! Run: `cargo run --release --example residue_upscale --features codec`

use ndarray::hpc::codec::CellMode;

const W: usize = 128;
const H: usize = 128;
const N: usize = W * H;
const STRIDE: usize = 8; // grid anchor stride → (W/S)·(H/S) anchors
const NAX: usize = W / STRIDE; // grid anchors per row
const NAY: usize = H / STRIDE;
const A: usize = NAX * NAY; // anchor count (grid == spiral, fair comparison)
const KIDW: usize = 4; // nearest-K anchors for spiral IDW extrapolation

/// Golden angle `π·(3 − √5)` — same constant as `helix_orient`'s spherical
/// Fibonacci, here on the 2-D sunflower disk instead of S².
const GOLDEN_ANGLE: f64 = std::f64::consts::PI * 0.763_932_022_500_210_4;

fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Order-0 Shannon entropy of an i32 symbol stream, in **bytes** (`H₀·n/8`).
/// This is the floor an ideal entropy coder (rANS) approaches — the honest,
/// non-strawman cost of a symbol stream.
fn entropy_bytes(sym: &[i32]) -> f64 {
    use std::collections::HashMap;
    let mut hist: HashMap<i32, u64> = HashMap::new();
    for &s in sym {
        *hist.entry(s).or_insert(0) += 1;
    }
    let n = sym.len() as f64;
    let bits: f64 = hist
        .values()
        .map(|&c| {
            let p = c as f64 / n;
            -(c as f64) * p.log2()
        })
        .sum();
    bits / 8.0
}

/// Four residual-shaped fields at increasing spatial frequency (decreasing
/// smoothness). Zero-mean i32, so they stand in for an MC residual.
fn field(kind: usize) -> Vec<i32> {
    let mut f = vec![0i32; N];
    for y in 0..H {
        for x in 0..W {
            let (xf, yf) = (x as f64, y as f64);
            let v = match kind {
                0 => 90.0 * (xf * 0.024).sin() * (yf * 0.020).cos(), // very smooth
                1 => {
                    // "weather-like": the codec_mode_histogram smooth field, zero-mean
                    80.0 * (xf * 0.018).sin() * (yf * 0.018).cos() + 24.0 * (xf * 0.09).sin()
                }
                2 => 60.0 * (xf * 0.11).sin() * (yf * 0.13).cos() + 30.0 * (yf * 0.07).sin(), // mid
                _ => ((mix((y as u64) << 20 | x as u64) & 0xFF) as f64) - 128.0,              // noise
            };
            f[y * W + x] = v.round() as i32;
        }
    }
    f
}

/// Bilinear FMA extrapolation from a regular anchor grid (stride `STRIDE`).
/// Anchor value = the field sampled at the anchor cell. Returns the dense
/// prediction (rounded to i32) — the PLACE.
fn grid_predict(f: &[i32]) -> (Vec<i32>, Vec<i32>) {
    // anchor values sampled at grid cells
    let mut av = vec![0i32; A];
    for ay in 0..NAY {
        for ax in 0..NAX {
            let (px, py) = ((ax * STRIDE).min(W - 1), (ay * STRIDE).min(H - 1));
            av[ay * NAX + ax] = f[py * W + px];
        }
    }
    let at = |ax: usize, ay: usize| av[ay.min(NAY - 1) * NAX + ax.min(NAX - 1)] as f64;
    let mut pred = vec![0i32; N];
    for y in 0..H {
        for x in 0..W {
            let (gx, gy) = (x as f64 / STRIDE as f64, y as f64 / STRIDE as f64);
            let (x0, y0) = (gx.floor() as usize, gy.floor() as usize);
            let (tx, ty) = (gx - x0 as f64, gy - y0 as f64);
            // bilinear FMA (multiply-add of the four corner anchors)
            let top = at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx;
            let bot = at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx;
            pred[y * W + x] = (top * (1.0 - ty) + bot * ty).round() as i32;
        }
    }
    (pred, av)
}

/// 2-D golden-spiral (sunflower) anchor positions — deterministic, regenerable
/// from `k` alone (never stored), the 2-D analog of `helix_orient`'s spherical
/// Fibonacci. Fills the frame: radius `√((k+½)/A)`, angle `k·golden_angle`.
fn spiral_anchor(k: usize) -> (usize, usize) {
    let r = (((k as f64) + 0.5) / A as f64).sqrt();
    let a = k as f64 * GOLDEN_ANGLE;
    // map unit disk → frame (ellipse to fill W×H)
    let cx = (W as f64 - 1.0) * 0.5;
    let cy = (H as f64 - 1.0) * 0.5;
    let x = (cx + r * a.cos() * cx).round().clamp(0.0, W as f64 - 1.0) as usize;
    let y = (cy + r * a.sin() * cy).round().clamp(0.0, H as f64 - 1.0) as usize;
    (x, y)
}

/// Inverse-distance-weighted FMA extrapolation from golden-spiral anchors —
/// each cell is `Σ wᵢ·vᵢ / Σ wᵢ` over its `KIDW` nearest anchors (surfel splat,
/// accumulated). Anchor positions are free (regenerable); only values cost.
fn spiral_predict(f: &[i32]) -> (Vec<i32>, Vec<i32>) {
    let pos: Vec<(usize, usize)> = (0..A).map(spiral_anchor).collect();
    let av: Vec<i32> = pos.iter().map(|&(x, y)| f[y * W + x]).collect();
    let mut pred = vec![0i32; N];
    for y in 0..H {
        for x in 0..W {
            // K nearest anchors by squared pixel distance
            let mut best: [(u64, usize); KIDW] = [(u64::MAX, 0); KIDW];
            for (i, &(ax, ay)) in pos.iter().enumerate() {
                let dx = ax as i64 - x as i64;
                let dy = ay as i64 - y as i64;
                let d2 = (dx * dx + dy * dy) as u64;
                // insert into the small top-K (K=4, linear is fine)
                if d2 < best[KIDW - 1].0 {
                    let mut j = KIDW - 1;
                    while j > 0 && best[j - 1].0 > d2 {
                        best[j] = best[j - 1];
                        j -= 1;
                    }
                    best[j] = (d2, i);
                }
            }
            // exact hit → take the anchor; else IDW (weight 1/(d²+1))
            let (mut num, mut den) = (0.0f64, 0.0f64);
            let mut exact = None;
            for &(d2, i) in &best {
                if d2 == 0 {
                    exact = Some(av[i]);
                    break;
                }
                let w = 1.0 / (d2 as f64 + 1.0);
                num += w * av[i] as f64;
                den += w;
            }
            pred[y * W + x] = exact.unwrap_or_else(|| (num / den).round() as i32);
        }
    }
    (pred, av)
}

fn residue(f: &[i32], pred: &[i32]) -> Vec<i32> {
    f.iter().zip(pred).map(|(&a, &p)| a - p).collect()
}

/// Paeth (PNG) local predictor — the standard lossless spatial baseline. This is
/// the *fair* rival to the global anchor predictors: a real lossless codec
/// decorrelates with a cheap causal local model (left/up/up-left), storing ZERO
/// anchors. `pred = whichever of {left, up, up-left} is nearest to left+up−ul`.
fn paeth_residue(f: &[i32]) -> Vec<i32> {
    let mut res = vec![0i32; N];
    for y in 0..H {
        for x in 0..W {
            let a = if x > 0 { f[y * W + x - 1] } else { 0 }; // left
            let b = if y > 0 { f[(y - 1) * W + x] } else { 0 }; // up
            let c = if x > 0 && y > 0 { f[(y - 1) * W + x - 1] } else { 0 }; // up-left
            let p = a + b - c;
            let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
            let pred = if pa <= pb && pa <= pc {
                a
            } else if pb <= pc {
                b
            } else {
                c
            };
            res[y * W + x] = f[y * W + x] - pred;
        }
    }
    res
}

/// Classify a residue field into the shipped codec taxonomy (H-7): |r|=0 → Skip,
/// |r|≤127 → Delta, else Escape.
fn mode_hist(res: &[i32]) -> [usize; 3] {
    let mut h = [0usize; 3];
    for &r in res {
        let idx = if r == 0 {
            CellMode::Skip as usize
        } else if r.abs() <= 127 {
            CellMode::Delta as usize
        } else {
            CellMode::Escape as usize
        };
        // Skip=0, Merge=1, Delta=2, Escape=3 → fold Merge slot out (unused here)
        h[match idx {
            0 => 0,
            2 => 1,
            _ => 2,
        }] += 1;
    }
    h
}

fn main() {
    println!("Residue upscaling — anatomy technique on a codec residual");
    println!("  {W}×{H} field, {A} anchors (grid {NAX}×{NAY} stride {STRIDE} == spiral {A}), K={KIDW} IDW");
    println!("  bytes = order-0 Shannon entropy H₀ (what an ideal rANS coder spends)\n");
    println!(
        "  {:<14} {:>9} {:>9} {:>11} {:>11}   {:>7} {:>7}",
        "field", "dense H₀", "paeth", "grid a+res", "spiral a+res", "grid×", "spiral×"
    );
    println!(
        "  {:<14} {:>9} {:>9} {:>11} {:>11}   {:>7} {:>7}",
        "", "(no model)", "(local)", "(global)", "(global)", "/paeth", "/paeth"
    );

    let names = ["very-smooth", "weather-like", "mid-freq", "noise"];
    for (kind, name) in names.iter().enumerate() {
        let f = field(kind);
        let dense = entropy_bytes(&f);
        let paeth = entropy_bytes(&paeth_residue(&f));

        let (gp, gav) = grid_predict(&f);
        let gr = residue(&f, &gp);
        let grid_total = entropy_bytes(&gav) + entropy_bytes(&gr);

        let (sp, sav) = spiral_predict(&f);
        let sr = residue(&f, &sp);
        let spiral_total = entropy_bytes(&sav) + entropy_bytes(&sr);

        // the honest comparison is vs the LOCAL predictor (paeth), not the no-model dense.
        println!(
            "  {:<14} {:>9.0} {:>9.0} {:>11.0} {:>11.0}   {:>6.2}× {:>6.2}×",
            name,
            dense,
            paeth,
            grid_total,
            spiral_total,
            paeth / grid_total,
            paeth / spiral_total,
        );
    }

    // H-7 tie-in: on the smoothest field, show the spiral-residue mode split —
    // the residue collapses to mostly Skip (0) once the predictor captures it.
    let f0 = field(0);
    let (sp0, _) = spiral_predict(&f0);
    let sr0 = residue(&f0, &sp0);
    let mh = mode_hist(&sr0);
    println!("\n  [H-7] very-smooth spiral-residue CellMode split (residue → mode):");
    println!(
        "        skip={} delta={} escape={}  ({:.1}% skip after the golden-spiral predictor)",
        mh[0],
        mh[1],
        mh[2],
        100.0 * mh[0] as f64 / N as f64
    );

    println!(
        "\n  MEASURED CONCLUSION (fair fight = vs paeth local DPCM, the standard lossless\n\
         \x20 predictor, which stores ZERO anchors):\n\
         \x20 • Global sparse-anchor upscaling DOES decorrelate — grid a+res (3648) ≪ the\n\
         \x20   no-model order-0 dense (14807). The anatomy 2000→4M mechanism is real.\n\
         \x20 • BUT for a DENSE SCALAR luma residual it LOSES to local DPCM (grid 0.70×,\n\
         \x20   spiral 0.38× of paeth on very-smooth): a cheap causal local predictor\n\
         \x20   beats it, because the 256 anchor VALUES cost real bits and a sparse grid\n\
         \x20   can't model structure finer than its stride. For scalar-dense residuals\n\
         \x20   the analogy is RHYME, not a winning transfer — use DPCM / transform.\n\
         \x20 • grid > spiral here (bilinear is a better scalar interpolant than\n\
         \x20   scattered IDW). The golden-spiral's win is NOT scalar fields.\n\
         \x20 • Where the anatomy technique is ALREADY measured to dominate is its NATIVE\n\
         \x20   domain: sparse DIRECTION fields on S² (surfel normals) — helix_orient\n\
         \x20   measures 1–3 bytes at Pearson 0.9917, a regime with no dense causal\n\
         \x20   neighbor to DPCM against.\n\
         \x20 • CONJECTURE (NOT measured here — needs realistic continuous MVs from a real\n\
         \x20   decode, i.e. M2): a sparse continuous MOTION-VECTOR field is directional +\n\
         \x20   anchor-shaped like the surfel case, so helix/golden-spiral coding may beat\n\
         \x20   dense MV coding there. The M1 MVs (integer −2..2) are too quantised to\n\
         \x20   test it — do not treat this bullet as a result."
    );
}
