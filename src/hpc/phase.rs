//! Phases in turns, and `(cos, sin)` without a transcendental call.
//!
//! A phase is a `u32` in **turns**: `2^32` is one full rotation. Wraparound is
//! the integer's own overflow, so `3θ` is `p.wrapping_mul(3)` and `θ + φ` is
//! `p.wrapping_add(q)`, both exact modulo one turn.
//!
//! [`PhaseLut`] maps a phase to `(cos, sin)` through a `2^bits`-entry table,
//! either at the nearest entry or by linear interpolation between the two
//! neighbours. Each lookup has a stated worst-case error,
//! [`PhaseLut::nearest_error_bound`] and [`PhaseLut::lerp_error_bound`], so a
//! caller can carry it into an exact decision instead of trusting the table.
//!
//! [`PhaseStep`] is an integer phase accumulator: the phase after `n` steps is
//! `start + n·step` modulo one turn, computed directly. It cannot drift, which
//! a floating-point complex recurrence `u ← u·e^{iΔθ}` does.
//!
//! Measured in the lance-graph `phasor_trig_probe` (x86-64-v3, release, 1M
//! random phases): interpolation at `2^12` entries matches f32 `sin_cos`
//! accuracy (max error 4.6e-7 relative against 5.1e-7) at about a quarter of
//! the time, and is about 12× faster than f64 `sin_cos`. The lookups here are
//! scalar; nothing in this module is a SIMD kernel.

use std::f64::consts::{PI, TAU};
use std::sync::OnceLock;

/// One full turn as a `f64`: the number of distinct phases.
const TURN: f64 = 4_294_967_296.0;

/// Convert an angle in radians to a phase in turns, modulo one turn.
///
/// Rounds to the nearest phase. Negative angles wrap. `NaN` maps to `0`. The
/// conversion is exact to the
/// precision of `rad` itself: for `|rad|` far beyond a few turns the `f64`
/// input already lost the low bits a `u32` phase can hold.
///
/// # Example
///
/// ```
/// use ndarray::hpc::phase::turns_from_radians;
/// use std::f64::consts::PI;
///
/// assert_eq!(turns_from_radians(0.0), 0);
/// assert_eq!(turns_from_radians(PI), 1 << 31);
/// assert_eq!(turns_from_radians(-PI / 2.0), 3 << 30);
/// ```
pub fn turns_from_radians(rad: f64) -> u32 {
    // rem_euclid, or the rounding, can reach exactly 2^32; the `u32` cast
    // then wraps it to 0, which is the correct phase.
    ((rad / TAU).rem_euclid(1.0) * TURN).round() as u64 as u32
}

/// Convert a phase in turns to an angle in radians, in `[0, 2π)`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::phase::radians_from_turns;
/// use std::f64::consts::PI;
///
/// assert_eq!(radians_from_turns(1 << 31), PI);
/// assert_eq!(radians_from_turns(0), 0.0);
/// ```
pub fn radians_from_turns(p: u32) -> f64 {
    f64::from(p) / TURN * TAU
}

/// A `(cos, sin)` table over `2^bits` equally spaced phases, `f32`.
///
/// Entries are stored as `[cos, sin]` pairs, so one lookup touches one cache
/// line. `2^12` entries are 32 KB.
///
/// # Example
///
/// ```
/// use ndarray::hpc::phase::{turns_from_radians, PhaseLut};
///
/// let lut = PhaseLut::new(12);
/// let p = turns_from_radians(1.0);
/// let (c, s) = lut.lerp(p);
/// assert!((f64::from(c) - 1.0f64.cos()).abs() <= lut.lerp_error_bound());
/// assert!((f64::from(s) - 1.0f64.sin()).abs() <= lut.lerp_error_bound());
/// ```
#[derive(Debug, Clone)]
pub struct PhaseLut {
    bits: u32,
    table: Box<[[f32; 2]]>,
}

impl PhaseLut {
    /// Smallest supported table: 4 entries.
    pub const MIN_BITS: u32 = 2;
    /// Largest supported table: `2^20` entries, 8 MB.
    pub const MAX_BITS: u32 = 20;

    /// Build a table of `2^bits` entries from `f64` `sin_cos`.
    ///
    /// # Panics
    ///
    /// If `bits` is outside [`Self::MIN_BITS`]`..=`[`Self::MAX_BITS`].
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseLut;
    ///
    /// let lut = PhaseLut::new(10);
    /// assert_eq!(lut.bits(), 10);
    /// assert_eq!(lut.len(), 1024);
    /// ```
    pub fn new(bits: u32) -> Self {
        assert!(
            (Self::MIN_BITS..=Self::MAX_BITS).contains(&bits),
            "PhaseLut bits must be in {}..={}, got {bits}",
            Self::MIN_BITS,
            Self::MAX_BITS
        );
        let n = 1usize << bits;
        let table = (0..n)
            .map(|i| {
                let (s, c) = (TAU * i as f64 / n as f64).sin_cos();
                [c as f32, s as f32]
            })
            .collect();
        PhaseLut { bits, table }
    }

    /// `log2` of the table size.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseLut;
    /// assert_eq!(PhaseLut::new(8).bits(), 8);
    /// ```
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Number of entries, `2^bits`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseLut;
    /// assert_eq!(PhaseLut::new(8).len(), 256);
    /// ```
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Always `false`: a table has at least [`Self::MIN_BITS`] bits.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseLut;
    /// assert!(!PhaseLut::new(2).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Bytes the table occupies.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseLut;
    /// assert_eq!(PhaseLut::new(12).bytes(), 32 * 1024);
    /// ```
    pub fn bytes(&self) -> usize {
        self.table.len() * core::mem::size_of::<[f32; 2]>()
    }

    /// `(cos, sin)` at the table entry nearest to `p`.
    ///
    /// Error: at most [`Self::nearest_error_bound`] per component.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseLut;
    ///
    /// let lut = PhaseLut::new(4);
    /// assert_eq!(lut.nearest(0), (1.0, 0.0));
    /// // One quarter turn is entry 4 of 16 exactly.
    /// let (c, s) = lut.nearest(1 << 30);
    /// assert!(c.abs() < 1e-7 && (s - 1.0).abs() < 1e-7);
    /// ```
    #[inline]
    pub fn nearest(&self, p: u32) -> (f32, f32) {
        let sh = 32 - self.bits;
        let i = (p.wrapping_add(1 << (sh - 1)) >> sh) as usize & (self.table.len() - 1);
        let [c, s] = self.table[i];
        (c, s)
    }

    /// `(cos, sin)` by linear interpolation between the two entries around `p`.
    ///
    /// Error: at most [`Self::lerp_error_bound`] per component. The result is
    /// not renormalised: it lies on the chord, slightly inside the unit circle.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::{radians_from_turns, PhaseLut};
    ///
    /// let lut = PhaseLut::new(10);
    /// let p = 0x1234_5678;
    /// let (c, s) = lut.lerp(p);
    /// let (st, ct) = radians_from_turns(p).sin_cos();
    /// assert!((f64::from(c) - ct).abs() <= lut.lerp_error_bound());
    /// assert!((f64::from(s) - st).abs() <= lut.lerp_error_bound());
    /// ```
    #[inline]
    pub fn lerp(&self, p: u32) -> (f32, f32) {
        let sh = 32 - self.bits;
        let m = self.table.len() - 1;
        let i = (p >> sh) as usize;
        let f = (p & ((1u32 << sh) - 1)) as f32 / (1u64 << sh) as f32;
        let [c0, s0] = self.table[i & m];
        let [c1, s1] = self.table[(i + 1) & m];
        (c0 + (c1 - c0) * f, s0 + (s1 - s0) * f)
    }

    /// [`Self::lerp`] over a slice of phases into two output slices.
    ///
    /// # Panics
    ///
    /// If the three slices differ in length.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseLut;
    ///
    /// let lut = PhaseLut::new(12);
    /// let phases = [0u32, 1 << 30, 1 << 31];
    /// let (mut c, mut s) = ([0f32; 3], [0f32; 3]);
    /// lut.lerp_batch(&phases, &mut c, &mut s);
    /// assert_eq!((c[2], s[2]), lut.lerp(1 << 31));
    /// ```
    pub fn lerp_batch(&self, phases: &[u32], cos: &mut [f32], sin: &mut [f32]) {
        assert!(
            phases.len() == cos.len() && phases.len() == sin.len(),
            "lerp_batch: {} phases, {} cos, {} sin",
            phases.len(),
            cos.len(),
            sin.len()
        );
        for ((p, c), s) in phases.iter().zip(cos.iter_mut()).zip(sin.iter_mut()) {
            (*c, *s) = self.lerp(*p);
        }
    }

    /// A per-component upper bound on the error of [`Self::nearest`].
    ///
    /// Half a table step of arc, `π / 2^bits`, plus the `f32` rounding of the
    /// stored entry.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseLut;
    /// let b = PhaseLut::new(12).nearest_error_bound();
    /// assert!(b > 7.6e-4 && b < 7.7e-4);
    /// ```
    pub fn nearest_error_bound(&self) -> f64 {
        PI / f64::from(1u32 << self.bits) + f64::from(f32::EPSILON)
    }

    /// A per-component upper bound on the error of [`Self::lerp`].
    ///
    /// For a step `h = 2π / 2^bits`, the chord between two neighbouring
    /// entries stays within `h² / 8` of the arc. The rest covers `f32`
    /// rounding of the entries, the fraction and the interpolation.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseLut;
    /// let b = PhaseLut::new(12).lerp_error_bound();
    /// assert!(b < 1e-6);
    /// ```
    pub fn lerp_error_bound(&self) -> f64 {
        let h = TAU / f64::from(1u32 << self.bits);
        h * h / 8.0 + 4.0 * f64::from(f32::EPSILON)
    }
}

/// A shared `2^12`-entry table (32 KB), built on first use.
///
/// # Example
///
/// ```
/// use ndarray::hpc::phase::phase_lut_4096;
///
/// let lut = phase_lut_4096();
/// assert_eq!(lut.len(), 4096);
/// assert!(std::ptr::eq(lut, phase_lut_4096()));
/// ```
pub fn phase_lut_4096() -> &'static PhaseLut {
    static LUT: OnceLock<PhaseLut> = OnceLock::new();
    LUT.get_or_init(|| PhaseLut::new(12))
}

/// An integer phase accumulator: a start phase and a per-step increment.
///
/// [`PhaseStep::at`] gives the phase after `n` steps directly, `start + n·step`
/// modulo one turn. There is no running state, so there is nothing to drift.
/// The chosen `step` is the nearest representable turn fraction to the
/// requested angle; that difference is a frequency choice, not accumulated
/// error, and [`PhaseStep::from_radians`] reports it.
///
/// # Example
///
/// ```
/// use ndarray::hpc::phase::PhaseStep;
///
/// let s = PhaseStep::new(0, 1 << 30); // a quarter turn per step
/// assert_eq!(s.at(0), 0);
/// assert_eq!(s.at(3), 3 << 30);
/// assert_eq!(s.at(4), 0); // one full turn
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhaseStep {
    start: u32,
    step: u32,
}

impl PhaseStep {
    /// A start phase and an increment, both in turns.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseStep;
    /// let s = PhaseStep::new(7, 3);
    /// assert_eq!((s.start(), s.step()), (7, 3));
    /// ```
    pub fn new(start: u32, step: u32) -> Self {
        PhaseStep { start, step }
    }

    /// Build from radians. Returns the accumulator and the difference, in
    /// radians per step, between the requested and the representable step.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseStep;
    ///
    /// let (s, off) = PhaseStep::from_radians(0.0, 0.001);
    /// assert!(off.abs() < 1e-9);
    /// assert_eq!(s.start(), 0);
    /// ```
    pub fn from_radians(start: f64, step: f64) -> (Self, f64) {
        let p = turns_from_radians(step);
        // The representable step as a signed angle in (-π, π].
        let got = f64::from(p as i32) / TURN * TAU;
        let want = (step + PI).rem_euclid(TAU) - PI;
        (
            PhaseStep {
                start: turns_from_radians(start),
                step: p,
            },
            got - want,
        )
    }

    /// The start phase.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseStep;
    /// assert_eq!(PhaseStep::new(5, 1).start(), 5);
    /// ```
    pub fn start(&self) -> u32 {
        self.start
    }

    /// The increment per step.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseStep;
    /// assert_eq!(PhaseStep::new(5, 1).step(), 1);
    /// ```
    pub fn step(&self) -> u32 {
        self.step
    }

    /// The phase after `n` steps: `start + n·step` modulo one turn, exact.
    ///
    /// Only `n mod 2^32` matters, because `step · 2^32 ≡ 0` modulo one turn.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::phase::PhaseStep;
    ///
    /// let s = PhaseStep::new(10, 3);
    /// assert_eq!(s.at(5), 25);
    /// assert_eq!(s.at(1 << 32), s.at(0));
    /// ```
    pub fn at(&self, n: u64) -> u32 {
        self.start.wrapping_add(self.step.wrapping_mul(n as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SplitMix64, for phases that are not a lattice.
    fn phases(n: usize, seed: u64) -> Vec<u32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = s;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                (z ^ (z >> 31)) as u32
            })
            .collect()
    }

    /// Max per-component error against `f64` `sin_cos`, over random phases plus
    /// the boundaries a wrong index or a missing wrap would break.
    fn max_error(lut: &PhaseLut, f: impl Fn(&PhaseLut, u32) -> (f32, f32)) -> f64 {
        let mut ps = phases(1 << 18, 0x5EED + u64::from(lut.bits()));
        let sh = 32 - lut.bits();
        ps.extend([0, u32::MAX, 1 << (sh - 1), (1 << sh) - 1, 1 << sh, u32::MAX - (1 << (sh - 1))]);
        ps.iter()
            .map(|&p| {
                let (c, s) = f(lut, p);
                let (st, ct) = radians_from_turns(p).sin_cos();
                (f64::from(c) - ct).abs().max((f64::from(s) - st).abs())
            })
            .fold(0.0, f64::max)
    }

    /// FAILS IF: a lookup exceeds its stated bound, OR the bound is so loose
    /// that it could not catch a real error (the measured error must reach
    /// at least half of it for nearest, a tenth for lerp).
    #[test]
    fn lookups_stay_within_their_stated_bounds_and_the_bounds_are_tight() {
        for bits in [4, 8, 10, 12, 16] {
            let lut = PhaseLut::new(bits);
            let en = max_error(&lut, PhaseLut::nearest);
            let el = max_error(&lut, PhaseLut::lerp);
            assert!(en <= lut.nearest_error_bound(), "bits {bits}: nearest {en} > {}", lut.nearest_error_bound());
            assert!(el <= lut.lerp_error_bound(), "bits {bits}: lerp {el} > {}", lut.lerp_error_bound());
            assert!(en > 0.5 * lut.nearest_error_bound(), "bits {bits}: nearest bound is vacuous");
            if bits <= 12 {
                // Above 12 bits the f32 rounding floor dominates the chord sag.
                assert!(el > 0.1 * lut.lerp_error_bound(), "bits {bits}: lerp bound is vacuous");
            }
            assert!(el < en, "bits {bits}: interpolation must beat the nearest entry");
        }
    }

    /// FAILS IF: interpolation at 2^12 entries is less accurate than f32
    /// `sin_cos`'s own error scale (the claim the module doc makes).
    #[test]
    fn interpolation_at_4096_entries_matches_f32_accuracy() {
        let el = max_error(phase_lut_4096(), PhaseLut::lerp);
        assert!(el < 1e-6, "{el}");
    }

    /// FAILS IF: the table's index wraps wrongly at the end of the turn: the
    /// last interpolation interval must run into entry 0, not past the table.
    #[test]
    fn the_last_interval_interpolates_into_entry_zero() {
        let lut = PhaseLut::new(8);
        let (c, s) = lut.lerp(u32::MAX);
        assert!((c - 1.0).abs() < 1e-6 && s.abs() < 1e-6, "({c}, {s})");
        assert_eq!(lut.nearest(u32::MAX), (1.0, 0.0));
    }

    /// FAILS IF: `3θ` as `wrapping_mul(3)` disagrees with `3θ mod 2π`.
    #[test]
    fn integer_phase_multiplication_is_angle_multiplication_mod_one_turn() {
        for p in phases(4096, 3) {
            let want = (3.0 * radians_from_turns(p)).rem_euclid(TAU);
            let got = radians_from_turns(p.wrapping_mul(3));
            let d = (want - got).abs();
            assert!(d < 1e-9 || (TAU - d) < 1e-9, "{p}: {want} vs {got}");
        }
    }

    /// FAILS IF: the conversions do not round-trip, or negative angles do not
    /// wrap, or NaN leaks into a phase.
    #[test]
    fn conversions_round_trip_and_wrap() {
        for p in phases(4096, 9) {
            assert_eq!(turns_from_radians(radians_from_turns(p)), p);
        }
        assert_eq!(turns_from_radians(-TAU), 0);
        assert_eq!(turns_from_radians(-PI / 2.0), 3 << 30);
        assert_eq!(turns_from_radians(5.0 * TAU + PI), 1 << 31);
        assert_eq!(turns_from_radians(f64::NAN), 0);
    }

    /// FAILS IF: the accumulator drifts. After 10^6 steps the phase must equal
    /// the closed form exactly, and the looked-up point must stay within the
    /// table's bound of the true point at that exact phase.
    #[test]
    fn the_accumulator_does_not_drift() {
        let (s, off) = PhaseStep::from_radians(0.3, TAU / 4096.0 * 1.618_033_988_749_895);
        assert!(off.abs() < 1e-9, "{off}");
        let lut = phase_lut_4096();
        let mut p = s.start();
        for n in 1..=1_000_000u64 {
            p = p.wrapping_add(s.step());
            if n % 997 == 0 || n == 1_000_000 {
                assert_eq!(p, s.at(n));
                let (c, sn) = lut.lerp(p);
                let (st, ct) = radians_from_turns(p).sin_cos();
                let e = (f64::from(c) - ct).abs().max((f64::from(sn) - st).abs());
                assert!(e <= lut.lerp_error_bound(), "step {n}: {e}");
            }
        }
    }

    /// FAILS IF: `from_radians` misreports the representable step, for a
    /// negative step and one past half a turn (both wrap to a signed step).
    #[test]
    fn from_radians_reports_the_step_it_actually_took() {
        for want in [-0.25, 0.1, PI - 1e-3, PI + 1e-3, -3.0] {
            let (s, off) = PhaseStep::from_radians(0.0, want);
            let got = f64::from(s.step() as i32) / TURN * TAU;
            let w = (want + PI).rem_euclid(TAU) - PI;
            assert!((got - w - off).abs() < 1e-15, "{want}");
            assert!(off.abs() <= PI / TURN + 1e-15, "{want}: {off}");
        }
    }

    /// FAILS IF: the batch form differs from the scalar lookup.
    #[test]
    fn batch_equals_scalar() {
        let lut = PhaseLut::new(12);
        let ps = phases(1000, 11);
        let (mut c, mut s) = (vec![0f32; ps.len()], vec![0f32; ps.len()]);
        lut.lerp_batch(&ps, &mut c, &mut s);
        for (i, &p) in ps.iter().enumerate() {
            assert_eq!((c[i], s[i]), lut.lerp(p));
        }
    }

    /// FAILS IF: a mismatched batch is accepted.
    #[test]
    #[should_panic(expected = "lerp_batch")]
    fn batch_rejects_mismatched_lengths() {
        PhaseLut::new(4).lerp_batch(&[0, 1], &mut [0.0; 2], &mut [0.0; 1]);
    }

    /// FAILS IF: an out-of-range table size is accepted.
    #[test]
    #[should_panic(expected = "PhaseLut bits")]
    fn rejects_bits_out_of_range() {
        let _ = PhaseLut::new(PhaseLut::MAX_BITS + 1);
    }
}
