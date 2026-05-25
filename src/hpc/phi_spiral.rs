//! PhiSpiral256 — golden-spiral leaf-location codec.
//!
//! # What this is
//!
//! A fixed 256-entry codebook of points on the unit disk, placed on a
//! golden-angle (Vogel / phyllotactic) spiral. Each point is a *location
//! address*: the one-byte answer to "**where** does the unexplained local
//! residual live?" — after meaning and magnitude have already been
//! stripped by sibling lanes.
//!
//! ```text
//!   CAM_PQ       removes the meaning direction      → residual ⊥ meaning
//!   PolarQuant   projects onto magnitude/similarity → LOSES the perpendicular location
//!   PhiSpiral256 addresses that orthogonal location → recovers what PolarQuant discarded
//!   BGZ17        phase: golden/coprime cyclic step  → ≈ Quintenzirkel (circle of fifths)
//! ```
//!
//! PhiSpiral256 is **orthogonal to PolarQuant**: PolarQuant carries the
//! radial magnitude scalar and throws away angular position; PhiSpiral256
//! carries exactly that thrown-away position. It is *not* a meaning lane
//! (that is CAM_PQ) and *not* a magnitude lane (that is PolarQuant).
//!
//! # Two axes: BGZ17 phase × Poincaré clustering
//!
//! The golden spiral decomposes into exactly two independent axes, which
//! correspond to two distinct lanes — PhiSpiral256 does not replace either,
//! it *carries* both:
//!
//! - **Angle = BGZ17 phase.** BGZ17 *is* a phase encoding — a golden /
//!   coprime cyclic step akin to the **Quintenzirkel** (circle of fifths),
//!   where a coprime interval visits every position in a maximally-spread
//!   order. The golden angle `θ_k = 2π·frac(k/φ)` is the continuous limit
//!   of that step (the most-irrational rotation), so the spiral's **angle
//!   is the BGZ17 phase**. BGZ17's own coprime step is `GOLDEN_STEP = 11`
//!   over `BASE_DIM = 17` (verified in `hpc::bgz17_bridge`: the table
//!   `(i·11) mod 17` visits all 17 positions); the atom's two 2-bit `bgz`
//!   fields are plan-inspired placeholders, **not** those constants.
//! - **Radius + spread = Poincaré clustering.** Poincaré is its own thing:
//!   **spatial location clustering** on the hyperbolic radial chart (see
//!   [`SpiralChart::Poincare`]) — a hemispheric chart around the parent
//!   anchor rather than a flat plane. This is the axis the cluster lives on.
//!
//! # Design trajectory: splat, not point (the goal)
//!
//! This module currently implements the **anchor layer**: a 256-point
//! golden-spiral codebook with nearest-anchor lookup. The intended end
//! state, on the Poincaré (clustering) axis, is bigger:
//!
//! - **A Gaussian splat, not a point.** A leaf addresses a **cluster** —
//!   an anchor plus a 2D spread (covariance) — encoding *spatial location
//!   clustering* of the orthogonal residual, not a single location. This is
//!   the same object the graphics renderer already produces: the 2D conic
//!   `Σ_img` in `hpc::splat3d::project` and the SPD covariance in
//!   `hpc::splat3d::spd3` *are* the splat spread. The location splat reuses
//!   that math.
//! - **Maybe spatial magnitude.** The splat may also carry a weight /
//!   opacity, making the leaf a weighted cluster. Forward extension, not
//!   yet encoded.
//!
//! So the trajectory is: anchor codebook (here) → splat = anchor + 2D
//! covariance (reuse `splat3d` Spd/conic) → optional magnitude. The point
//! codebook below is deliberately the first, smallest, testable layer.
//!
//! # The geometry (golden angle + radius law)
//!
//! For `k ∈ 0..256` the center is `(r_k·cosθ_k, r_k·sinθ_k)` with
//!
//! ```text
//!   θ_k = 2π · frac(k / φ)            φ = (1+√5)/2     (golden angle, ≈137.5°)
//! ```
//!
//! The golden angle is the "most irrational" rotation, so the points never
//! settle into radial spokes — the disk is covered with minimal clumping
//! and no angular aliasing (the Vogel sunflower lattice). The same
//! most-irrational-spacing principle already ships in this crate
//! (`hpc::surround_metadata::GOLDEN_ANGLE`, `hpc::holo` Fibonacci carriers).
//!
//! The radius law is a chart choice:
//!
//! ```text
//!   Euclidean (equal area):       r_k = √( (k+0.5)/256 )
//!   Poincaré  (equal hyp. area):  ρ_k = arccosh(1 + u_k·(cosh ρ_max − 1)),
//!                                 r_k = tanh(ρ_k / 2),  u_k = (k+0.5)/256
//! ```
//!
//! # Pseudo-Poincaré: when the cheap chart is free
//!
//! The Euclidean spiral is a legitimate stand-in for a true Poincaré chart
//! **only near the origin**, because the Poincaré metric degenerates to
//! Euclidean there (`tanh(x) ≈ x` for small `x`). PhiSpiral256 encodes a
//! *small* orthogonal residual around a parent anchor, so it lives in that
//! near-origin regime where the two charts nearly coincide — that is why
//! "golden spiral as pseudo-Poincaré" is free here. Toward the disk
//! boundary the charts diverge sharply, and only a true Poincaré chart
//! preserves Möbius-equivariance (isometric recentering of the anchor); a
//! fixed Euclidean spiral does not. Callers that recenter via a
//! Möbius/Poincaré chart before addressing should build with
//! [`SpiralChart::Poincare`]; the Euclidean default is the correct first
//! mode per the design plan.
//!
//! # Precision ladder (location)
//!
//! BGZ17 is the *coarse* recoverable-sampling skeleton — Base17 with
//! `BASE_DIM = 17` and `GOLDEN_STEP = 11` (verified in `hpc::bgz17_bridge`;
//! the coprime golden step `(i·11) mod 17` visits all 17 positions),
//! reducing a 16384-bit plane to a **lossy** `[i16; 17]` by golden-step
//! octave averaging. When precise location is needed, the residual is
//! addressed at higher resolution than Base17:
//!
//! ```text
//!   BGZ17 / Base17    17 levels   coarse recoverable sampling skeleton
//!   PhiSpiral256      256 levels  one-byte spiral location address (spiral_id)
//!   i16 / i16 (Q15)   full        signed disk coordinates (PhiSpiralCenterQ15)
//! ```
//!
//! The packed atom carries the byte address (`spiral_id`) plus the BGZ17
//! offset/stride *family indices*; the precise signed `i16`/`i16` coordinate
//! lives in the center table ([`PhiSpiralCenterQ15`]).
//!
//! # Storage
//!
//! Centers are Q15 fixed-point (`i16`/`u16`), so the 256-entry table is
//! 256 × 6 = 1536 bytes — one small cache-resident codebook. Distance and
//! neighbor tables are computed on demand from the centers (a sqrt each);
//! materialized tables can be added when a consumer needs O(1) lookup, but
//! are intentionally not stored here (1536 B beats a 128 KB distance table
//! for the common nearest-center query).

const N: usize = 256;

/// Golden ratio φ = (1+√5)/2.
const PHI: f64 = 1.618_033_988_749_894_8;

/// Q15 fixed-point scale: a value in `[-1, 1]` maps to `[-32767, 32767]`.
const Q15: f64 = 32767.0;

// ════════════════════════════════════════════════════════════════════════════
// Chart selection
// ════════════════════════════════════════════════════════════════════════════

/// Radius law for the spiral centers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpiralChart {
    /// Equal-Euclidean-area Vogel disk: `r_k = √((k+0.5)/256)`. The correct
    /// first mode — flat local tangent coordinates, no Möbius recenter.
    Euclidean,
    /// Equal-hyperbolic-area Poincaré disk with maximum hyperbolic radius
    /// `rho_max`: `r_k = tanh(ρ_k/2)`, `ρ_k = arccosh(1 + u_k·(cosh ρ_max − 1))`.
    /// Use when the caller recenters the parent anchor via a Möbius chart.
    Poincare {
        /// Maximum hyperbolic radius ρ_max (> 0). Larger values push more
        /// resolution toward the disk boundary.
        rho_max: f32,
    },
}

// ════════════════════════════════════════════════════════════════════════════
// Center storage — Q15 fixed point
// ════════════════════════════════════════════════════════════════════════════

/// One spiral center in Q15 fixed point. `x_q15`/`y_q15` are the disk
/// coordinates in `[-1, 1]`; `radius_q15` is `‖(x, y)‖ ∈ [0, 1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct PhiSpiralCenterQ15 {
    /// X coordinate × 32767.
    pub x_q15: i16,
    /// Y coordinate × 32767.
    pub y_q15: i16,
    /// Radius × 32767 (always ≥ 0).
    pub radius_q15: u16,
}

impl PhiSpiralCenterQ15 {
    const ZERO: Self = Self {
        x_q15: 0,
        y_q15: 0,
        radius_q15: 0,
    };

    /// Decode to `(x, y)` floats in `[-1, 1]`.
    #[inline]
    pub fn xy(self) -> (f32, f32) {
        (self.x_q15 as f32 / Q15 as f32, self.y_q15 as f32 / Q15 as f32)
    }

    /// Decode the radius to a float in `[0, 1)`.
    #[inline]
    pub fn radius(self) -> f32 {
        self.radius_q15 as f32 / Q15 as f32
    }
}

#[inline]
fn q15_signed(v: f64) -> i16 {
    (v * Q15).round().clamp(-Q15, Q15) as i16
}

#[inline]
fn q15_unsigned(v: f64) -> u16 {
    // Radius is in [0, 1); Q15 keeps it inside u16 (max 32767).
    (v * Q15).round().clamp(0.0, Q15) as u16
}

// ════════════════════════════════════════════════════════════════════════════
// PhiSpiral256 — the 256-center codebook
// ════════════════════════════════════════════════════════════════════════════

/// A 256-entry golden-spiral location codebook.
#[derive(Clone, Debug)]
pub struct PhiSpiral256 {
    chart: SpiralChart,
    centers: [PhiSpiralCenterQ15; N],
}

impl PhiSpiral256 {
    /// Build the 256 centers for `chart`. Deterministic for a given chart
    /// (modulo platform `f64` transcendental rounding, which the Q15
    /// quantization largely absorbs).
    pub fn new(chart: SpiralChart) -> Self {
        let inv_phi = 1.0 / PHI;
        let mut centers = [PhiSpiralCenterQ15::ZERO; N];
        for (k, c) in centers.iter_mut().enumerate() {
            let kf = k as f64;
            // Golden angle: θ = 2π · frac(k/φ).
            let theta = std::f64::consts::TAU * (kf * inv_phi).fract();
            let u = (kf + 0.5) / N as f64;
            let r = match chart {
                SpiralChart::Euclidean => u.sqrt(),
                SpiralChart::Poincare { rho_max } => {
                    let rmax = rho_max as f64;
                    let rho = (1.0 + u * (rmax.cosh() - 1.0)).acosh();
                    (rho * 0.5).tanh()
                }
            };
            *c = PhiSpiralCenterQ15 {
                x_q15: q15_signed(r * theta.cos()),
                y_q15: q15_signed(r * theta.sin()),
                radius_q15: q15_unsigned(r),
            };
        }
        Self { chart, centers }
    }

    /// The chart this codebook was built with.
    #[inline]
    pub fn chart(&self) -> SpiralChart {
        self.chart
    }

    /// All 256 centers (Q15).
    #[inline]
    pub fn centers(&self) -> &[PhiSpiralCenterQ15; N] {
        &self.centers
    }

    /// Decode center `id` to `(x, y)` floats.
    #[inline]
    pub fn center_xy(&self, id: u8) -> (f32, f32) {
        self.centers[id as usize].xy()
    }

    /// Nearest spiral id to the query point `(x, y)` by Euclidean distance.
    /// Scalar reference; `O(256)` linear scan, no allocation.
    pub fn nearest(&self, x: f32, y: f32) -> u8 {
        let mut best = 0u8;
        let mut best_d2 = f32::INFINITY;
        for k in 0..N {
            let (cx, cy) = self.centers[k].xy();
            let dx = x - cx;
            let dy = y - cy;
            let d2 = dx * dx + dy * dy;
            if d2 < best_d2 {
                best_d2 = d2;
                best = k as u8;
            }
        }
        best
    }

    /// Euclidean distance between centers `i` and `j` (chord length on the
    /// disk, in `[0, 2]`).
    #[inline]
    pub fn distance(&self, i: u8, j: u8) -> f32 {
        let (ax, ay) = self.centers[i as usize].xy();
        let (bx, by) = self.centers[j as usize].xy();
        let dx = ax - bx;
        let dy = ay - by;
        (dx * dx + dy * dy).sqrt()
    }

    /// Fill `out` with the ids of the `out.len()` nearest centers to `id`
    /// (excluding `id` itself), nearest first. `out.len()` must be ≤ 255.
    pub fn k_nearest(&self, id: u8, out: &mut [u8]) {
        debug_assert!(out.len() < N, "k_nearest: K must be < 256");
        // Selection by repeated min over the 255 candidates — fine for the
        // small K (8/16) the codebook neighbor tables use.
        let mut used = [false; N];
        used[id as usize] = true;
        for slot in out.iter_mut() {
            let mut best = u8::MAX;
            let mut best_d = f32::INFINITY;
            for j in 0..N {
                if used[j] {
                    continue;
                }
                let d = self.distance(id, j as u8);
                if d < best_d {
                    best_d = d;
                    best = j as u8;
                }
            }
            used[best as usize] = true;
            *slot = best;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PhiSpiralLeafAtom16 — packed leaf atom
// ════════════════════════════════════════════════════════════════════════════

/// A packed 16-bit leaf atom.
///
/// ```text
///   bits  0..=7   phi_spiral_id   (location address, 0..255)
///   bits  8..=11  mag4            (magnitude band, 0..15)
///   bits 12..=13  bgz family lo   (plan-inspired 2-bit carrier, 0..3)
///   bits 14..=15  bgz family hi   (plan-inspired 2-bit carrier, 0..3)
/// ```
///
/// The two `bgz` fields are **plan-inspired placeholders**, not grounded
/// BGZ17 parameters. The real BGZ17 (`hpc::bgz17_bridge`) is Base17 with
/// `BASE_DIM = 17` and `GOLDEN_STEP = 11` (the coprime golden-step table
/// `(i·11) mod 17` visits all positions — a Quintenzirkel-like phase),
/// reducing a 16384-bit plane to a lossy `[i16; 17]` by golden-step octave
/// averaging. These 4 bits do not encode those constants; a real BGZ17
/// binding is a follow-up, not specified here. (The earlier "offset 17..27 /
/// stride 2,4" annotation was an unverified chat description — corrected.)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct PhiSpiralLeafAtom16(pub u16);

impl PhiSpiralLeafAtom16 {
    /// Pack the four fields. `mag4`, `offset2`, `stride2` are masked to
    /// their bit widths (4/2/2); the full `spiral_id` byte is kept.
    #[inline]
    pub fn new(spiral_id: u8, mag4: u8, offset2: u8, stride2: u8) -> Self {
        let v = (spiral_id as u16)
            | (((mag4 & 0x0F) as u16) << 8)
            | (((offset2 & 0x03) as u16) << 12)
            | (((stride2 & 0x03) as u16) << 14);
        Self(v)
    }

    /// The location address (0..255).
    #[inline]
    pub fn spiral_id(self) -> u8 {
        (self.0 & 0x00FF) as u8
    }

    /// The magnitude band (0..15).
    #[inline]
    pub fn mag4(self) -> u8 {
        ((self.0 >> 8) & 0x0F) as u8
    }

    /// The BGZ17 offset family (0..3).
    #[inline]
    pub fn offset2(self) -> u8 {
        ((self.0 >> 12) & 0x03) as u8
    }

    /// The BGZ17 stride family (0..3).
    #[inline]
    pub fn stride2(self) -> u8 {
        ((self.0 >> 14) & 0x03) as u8
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Tests — deterministic construction + golden invariants
// ════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn euclidean() -> PhiSpiral256 {
        PhiSpiral256::new(SpiralChart::Euclidean)
    }

    #[test]
    fn atom16_pack_unpack_roundtrip() {
        for spiral_id in [0u8, 1, 42, 137, 255] {
            for mag4 in 0u8..16 {
                for offset2 in 0u8..4 {
                    for stride2 in 0u8..4 {
                        let a = PhiSpiralLeafAtom16::new(spiral_id, mag4, offset2, stride2);
                        assert_eq!(a.spiral_id(), spiral_id);
                        assert_eq!(a.mag4(), mag4);
                        assert_eq!(a.offset2(), offset2);
                        assert_eq!(a.stride2(), stride2);
                    }
                }
            }
        }
    }

    #[test]
    fn atom16_masks_overflowing_fields() {
        // mag4/offset2/stride2 are masked to their widths; spiral_id is full.
        let a = PhiSpiralLeafAtom16::new(0xFF, 0xFF, 0xFF, 0xFF);
        assert_eq!(a.spiral_id(), 0xFF);
        assert_eq!(a.mag4(), 0x0F);
        assert_eq!(a.offset2(), 0x03);
        assert_eq!(a.stride2(), 0x03);
    }

    #[test]
    fn centers_are_deterministic() {
        let a = euclidean();
        let b = euclidean();
        assert_eq!(a.centers(), b.centers(), "same chart must yield identical centers");
    }

    #[test]
    fn all_centers_inside_unit_disk() {
        for chart in [SpiralChart::Euclidean, SpiralChart::Poincare { rho_max: 4.0 }] {
            let s = PhiSpiral256::new(chart);
            for k in 0..256u16 {
                let (x, y) = s.center_xy(k as u8);
                let r = (x * x + y * y).sqrt();
                assert!(r <= 1.0 + 1e-4, "{chart:?} center {k} outside disk: r={r}");
            }
        }
    }

    #[test]
    fn euclidean_radius_is_monotone_nondecreasing() {
        // r_k = sqrt((k+0.5)/256) increases with k. The stored radius_q15
        // must reflect that (modulo Q15 rounding ties).
        let s = euclidean();
        let mut prev = 0u16;
        for k in 0..256 {
            let r = s.centers()[k].radius_q15;
            assert!(r + 1 >= prev, "radius decreased at k={k}: {prev} -> {r}");
            prev = r;
        }
    }

    #[test]
    fn self_distance_is_zero() {
        let s = euclidean();
        for k in 0..256u16 {
            assert_eq!(s.distance(k as u8, k as u8), 0.0, "self-distance nonzero at {k}");
        }
    }

    #[test]
    fn distance_is_symmetric() {
        let s = euclidean();
        for i in [0u8, 7, 42, 200, 255] {
            for j in [1u8, 13, 99, 180, 254] {
                assert_eq!(s.distance(i, j), s.distance(j, i), "asymmetric distance ({i},{j})");
            }
        }
    }

    #[test]
    fn nearest_of_a_center_is_that_center() {
        // Querying exactly at center `id` must return `id` — the cleanest
        // sanity check on the nearest scan and the Q15 decode.
        let s = euclidean();
        for id in 0..256u16 {
            let (x, y) = s.center_xy(id as u8);
            assert_eq!(s.nearest(x, y), id as u8, "nearest(center[{id}]) != {id}");
        }
    }

    #[test]
    fn centers_are_distinct() {
        // The golden angle must not collide two centers onto the same Q15
        // cell (that would mean a wasted codebook slot / angular alias).
        let s = euclidean();
        let c = s.centers();
        for i in 0..256 {
            for j in (i + 1)..256 {
                assert_ne!(c[i], c[j], "duplicate centers at {i} and {j}");
            }
        }
    }

    #[test]
    fn k_nearest_excludes_self_and_is_ordered() {
        let s = euclidean();
        let mut out = [0u8; 8];
        s.k_nearest(42, &mut out);
        // none equals self
        for &n in &out {
            assert_ne!(n, 42, "k_nearest included self");
        }
        // distances are nondecreasing
        let mut prev = 0.0f32;
        for &n in &out {
            let d = s.distance(42, n);
            assert!(d >= prev - 1e-6, "k_nearest not ordered: {prev} then {d}");
            prev = d;
        }
        // ids are unique
        for i in 0..out.len() {
            for j in (i + 1)..out.len() {
                assert_ne!(out[i], out[j], "duplicate neighbor id");
            }
        }
    }

    #[test]
    fn poincare_front_loads_the_boundary() {
        // Grounded check of the chart distinction: equal-hyperbolic-area
        // packs MORE centers toward the boundary than equal-Euclidean-area,
        // because hyperbolic area density 1/(1−r²)² explodes as r → 1.
        //
        // Note the right discriminator is outer-ring *population*, not the
        // single outermost radius: the Poincaré max radius is capped at
        // tanh(ρ_max/2) (≈0.964 for ρ_max=4), below the Euclidean max ≈1.
        // It front-loads the boundary in count, not in reach.
        let e = PhiSpiral256::new(SpiralChart::Euclidean);
        let p = PhiSpiral256::new(SpiralChart::Poincare { rho_max: 4.0 });
        let outer = |s: &PhiSpiral256| (0..N).filter(|&k| s.centers()[k].radius() > 0.5).count();
        assert!(
            outer(&p) > outer(&e),
            "Poincaré should put more centers in the outer ring (r>0.5): poincaré={}, euclidean={}",
            outer(&p),
            outer(&e),
        );
    }

    #[test]
    fn near_origin_charts_nearly_coincide() {
        // The pseudo-Poincaré claim: near the origin the two charts agree
        // (tanh(x) ≈ x). The innermost center (k=0) should have nearly the
        // same radius under both laws.
        let e = PhiSpiral256::new(SpiralChart::Euclidean);
        let p = PhiSpiral256::new(SpiralChart::Poincare { rho_max: 4.0 });
        let e0 = e.centers()[0].radius();
        let p0 = p.centers()[0].radius();
        // Innermost radii are small; the absolute gap should be modest
        // relative to the disk. This documents the regime, not a tight bound.
        assert!((e0 - p0).abs() < 0.2, "near-origin charts diverge too much: e0={e0} p0={p0}");
    }
}
