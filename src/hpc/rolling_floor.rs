//! HDR rolling distribution: live statistics of popcount / Hamming
//! observations, and σ-lattice thresholds derived from them on demand.
//!
//! Harvested from lance-graph's `graph/blasgraph/hdr.rs`. Its constants
//! (reservoir capacity, cadence, minimum population, normality window, drift
//! rule) carry over unchanged; the arithmetic underneath is #327's exact,
//! mergeable [`MomentsU32`].
//!
//! # The accumulator owns facts, the query owns the view
//!
//! What is stored:
//!
//! * **Moments** — exact `(n, Σx, Σx²)`, folded on every observation. The
//!   current coordinates `(μ_t, σ_t)` follow from them at any time, from the
//!   first observation on: observation and use are concurrent, and more
//!   observations only sharpen the estimate.
//! * **Reservoir** — a deterministic Algorithm-R sample, the *evidence* for
//!   the periodic shape check.
//! * **Shape** — the *belief* about the distribution family: [`Shape::Gaussian`]
//!   (no data) or [`Shape::Empirical`] (a sorted sample and the `(μ_s, σ_s)`
//!   frame it was measured in).
//! * **Anchor** — the calibrated `(μ, σ)`, used only as the reference the drift
//!   rule measures against.
//!
//! What is derived, never stored: every threshold. A detector asks for a
//! [`SigmaLevel`] — a point on the integer σ-lattice, `k` quarter-σ below the
//! noise floor — and the active shape locates it in the current coordinates:
//!
//! * Gaussian: `μ_t − k·σ_t/4`.
//! * Empirical: the sample value at `k`'s Gaussian-equivalent tail rank,
//!   moved into the current frame, `μ_t + (x − μ_s)·σ_t/σ_s`.
//!
//! The percentile is not a second coordinate: it is fixed by `k` through
//! [`SigmaLevel::gaussian_tail_per_10000`]. A detector's shade of a response is
//! how far along its chosen lattice points the response survives
//! ([`RollingFloor::shade`]). Everything is integer.
//!
//! # Two cadences
//!
//! Moments and reservoir update on every observation. Every
//! [`RollingFloor::EVAL_CADENCE`] observations (after the first cadence) a
//! checkpoint checks drift and, when the parameters have not drifted, the
//! shape.
//!
//! # Parameter drift is not shape drift
//!
//! A drift alert means the running `(μ, σ)` left the anchor: the evidence now
//! spans two parameter regimes, so the shape is not judged from it at that
//! checkpoint. [`RollingFloor::recalibrate`] moves the anchor and forgets the
//! moments and the reservoir, but keeps the shape: a Gaussian whose `μ` and
//! `σ` moved is still Gaussian. The shape changes only at a drift-free
//! checkpoint.
//!
//! # What is and is not mergeable
//!
//! The moments are exact and partition-independent. The reservoir is
//! deterministic for a given observation *order* and has no merge law;
//! [`RollingFloor::observe_batch`] therefore reproduces the scalar stream
//! exactly rather than merging shards.

use super::statistics::{moments_u32, MomentsU32};

/// Floor of `√n`, integer Newton iteration. Exact for every `u32`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::rolling_floor::isqrt_u32;
/// assert_eq!(isqrt_u32(4095), 63);
/// assert_eq!(isqrt_u32(4096), 64);
/// assert_eq!(isqrt_u32(u32::MAX), 65535);
/// ```
pub fn isqrt_u32(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    // The start value must be >= floor(√n) so Newton descends monotonically.
    let mut x = 1u32 << ((33 - n.leading_zeros()) / 2);
    loop {
        let x1 = (x + n / x) / 2;
        if x1 >= x {
            return x;
        }
        x = x1;
    }
}

/// Rank `⌊per_10000 · len / 10000⌋`, clamped to the last index. `0` for an
/// empty sample. The integer rank rule for every empirical lookup.
///
/// # Example
///
/// ```
/// use ndarray::hpc::rolling_floor::rank_per_10000;
/// assert_eq!(rank_per_10000(1000, 1587), 158);
/// assert_eq!(rank_per_10000(1000, 10_000), 999);
/// assert_eq!(rank_per_10000(0, 5000), 0);
/// ```
pub fn rank_per_10000(len: usize, per_10000: u32) -> usize {
    if len == 0 {
        return 0;
    }
    ((u64::from(per_10000) * len as u64 / 10_000) as usize).min(len - 1)
}

/// Empirical quantile of an ascending slice at `per_10000 / 10000`, by
/// [`rank_per_10000`]. `0` for an empty slice.
///
/// # Example
///
/// ```
/// use ndarray::hpc::rolling_floor::quantile_of_sorted;
/// let s = [1, 2, 3, 4];
/// assert_eq!(quantile_of_sorted(&s, 0), 1);
/// assert_eq!(quantile_of_sorted(&s, 5000), 3);
/// assert_eq!(quantile_of_sorted(&s, 10_000), 4);
/// ```
pub fn quantile_of_sorted(sorted: &[u32], per_10000: u32) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[rank_per_10000(sorted.len(), per_10000)]
}

/// Deterministic reservoir sample of a `u32` stream (Vitter's Algorithm R).
///
/// Every element seen so far has the same chance of being held. The
/// replacement decision for the `k`-th element is a fixed hash of `k`, so an
/// identical stream always produces an identical reservoir. The result depends
/// on the order of the stream; there is no merge of two reservoirs.
///
/// # Example
///
/// ```
/// use ndarray::hpc::rolling_floor::ReservoirU32;
/// let mut r = ReservoirU32::new(4);
/// for d in [10, 20, 30] {
///     r.observe(d);
/// }
/// assert_eq!(r.len(), 3);
/// assert_eq!(r.quantile(5000), 20);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReservoirU32 {
    samples: Vec<u32>,
    capacity: usize,
    seen: u64,
}

impl ReservoirU32 {
    /// An empty reservoir holding at most `capacity` samples.
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
            seen: 0,
        }
    }

    /// Offer one value to the reservoir.
    #[inline]
    pub fn observe(&mut self, value: u32) {
        self.seen += 1;
        if self.samples.len() < self.capacity {
            self.samples.push(value);
        } else {
            // Replace slot j with probability capacity / seen.
            let j = Self::replacement_hash(self.seen) % self.seen;
            if (j as usize) < self.capacity {
                self.samples[j as usize] = value;
            }
        }
    }

    /// Maximum number of samples held.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of values offered so far.
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Number of samples currently held.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// `true` when no value has been offered yet.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The held samples, in slot order.
    pub fn samples(&self) -> &[u32] {
        &self.samples
    }

    /// The held samples, sorted ascending.
    pub fn sorted(&self) -> Vec<u32> {
        let mut s = self.samples.clone();
        s.sort_unstable();
        s
    }

    /// Empirical quantile at `per_10000 / 10000` (see [`rank_per_10000`]).
    /// O(len · log len); meant for the periodic shape path.
    pub fn quantile(&self, per_10000: u32) -> u32 {
        quantile_of_sorted(&self.sorted(), per_10000)
    }

    /// Pearson's second skewness `3(μ − median) / σ`, in integer division.
    /// Positive means right-skewed, `0` symmetric. `0` when `σ = 0` or the
    /// reservoir is empty.
    pub fn skewness(&self, mu: u32, sigma: u32) -> i32 {
        if sigma == 0 || self.samples.is_empty() {
            return 0;
        }
        skewness_from_median(mu, sigma, self.quantile(5000))
    }

    /// Kurtosis ×100: `100 · E[(X − μ)⁴] / σ⁴` over the reservoir, with the
    /// normal distribution at 300. Returns 300 when `σ = 0` or fewer than 4
    /// samples are held.
    pub fn kurtosis(&self, mu: u32, sigma: u32) -> u32 {
        if sigma == 0 || self.samples.len() < 4 {
            return 300;
        }
        let n = self.samples.len() as u128;
        // u128: a fourth power of a u32 difference is below 2^128.
        let m4: u128 = self
            .samples
            .iter()
            .map(|&d| {
                let diff = u128::from(d.abs_diff(mu));
                diff * diff * diff * diff
            })
            .sum::<u128>()
            / n;
        let s4 = u128::from(sigma).pow(4);
        u32::try_from(m4 * 100 / s4).unwrap_or(u32::MAX)
    }

    /// Deterministic splitmix64-style hash that drives replacement.
    fn replacement_hash(seed: u64) -> u64 {
        let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

fn skewness_from_median(mu: u32, sigma: u32, median: u32) -> i32 {
    // i64 so no difference of two u32 values can overflow.
    let s = 3 * (i64::from(mu) - i64::from(median)) / i64::from(sigma);
    s.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// `Φ(−k/4)`, the Gaussian lower-tail mass `k` quarter-σ below the mean, in
/// parts per 10 000 (rounded to nearest), for `k = 0..=16`.
const GAUSSIAN_TAIL_PER_10000: [u32; 17] =
    [5000, 4013, 3085, 2266, 1587, 1056, 668, 401, 228, 122, 62, 30, 13, 6, 2, 1, 0];

/// A point on the integer σ-lattice: `k` quarter-σ below the noise floor.
///
/// This is the identity of a sensitivity cut. A detector chooses which points
/// it asks for; the active [`Shape`] decides how each point is located in the
/// current distribution. `SigmaLevel(12)` is 3σ, `SigmaLevel(6)` is 1.5σ,
/// `SigmaLevel(0)` is the mean itself.
///
/// # Example
///
/// ```
/// use ndarray::hpc::rolling_floor::SigmaLevel;
/// assert_eq!(SigmaLevel(4).gaussian_tail_per_10000(), 1587); // 1σ
/// assert_eq!(SigmaLevel(12).gaussian_tail_per_10000(), 13); // 3σ
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SigmaLevel(pub u8);

impl SigmaLevel {
    /// The lattice coordinate `k`, in quarter-σ.
    pub const fn quarters(self) -> u32 {
        self.0 as u32
    }

    /// The Gaussian-equivalent lower-tail mass of this level, `Φ(−k/4)`, in
    /// parts per 10 000. This is how an empirical shape locates the same cut:
    /// the level fixes the rank, no caller supplies a percentile. Levels
    /// beyond 4σ (`k > 16`) have tail `0`, i.e. the sample minimum.
    pub const fn gaussian_tail_per_10000(self) -> u32 {
        let k = self.0 as usize;
        if k < GAUSSIAN_TAIL_PER_10000.len() {
            GAUSSIAN_TAIL_PER_10000[k]
        } else {
            0
        }
    }
}

/// The learned geometry of a non-Gaussian distribution: a sorted sample and
/// the `(μ_s, σ_s)` frame it was measured in.
///
/// It answers a [`SigmaLevel`] by the sample value at the level's tail rank,
/// moved from its own frame into the current coordinates.
///
/// # Example
///
/// ```
/// use ndarray::hpc::rolling_floor::{EmpiricalShape, SigmaLevel};
/// let shape = EmpiricalShape::from_sample(&[10, 20, 30, 40]).unwrap();
/// // In its own frame a level answers with the raw sample value.
/// let (mu, sigma) = (shape.mu(), shape.sigma());
/// assert_eq!(shape.locate(SigmaLevel(0), mu, sigma), 30);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmpiricalShape {
    sorted: Vec<u32>,
    mu: u32,
    sigma: u32,
}

impl EmpiricalShape {
    /// Learn the shape from a sample: sort it and record its own floor mean
    /// and `⌊√⌊M2/n⌋⌋` spread. `None` for an empty sample.
    pub fn from_sample(sample: &[u32]) -> Option<Self> {
        if sample.is_empty() {
            return None;
        }
        let mut sorted = sample.to_vec();
        sorted.sort_unstable();
        let m = moments_u32(&sorted);
        Some(Self {
            mu: saturate_u32(m.sum / u128::from(m.n)),
            sigma: isqrt_u32(saturate_u32(variance_floor(&m))),
            sorted,
        })
    }

    /// The sorted sample.
    pub fn sorted(&self) -> &[u32] {
        &self.sorted
    }

    /// The sample's floor mean.
    pub fn mu(&self) -> u32 {
        self.mu
    }

    /// The sample's spread.
    pub fn sigma(&self) -> u32 {
        self.sigma
    }

    /// Locate `level` in the frame `(mu, sigma)`:
    /// `mu + ⌊(x − μ_s)·sigma / σ_s⌋` with `x` the sample value at the level's
    /// tail rank, clamped to `u32`. When the sample has no spread (`σ_s = 0`)
    /// the offset `x − μ_s` is used unscaled.
    pub fn locate(&self, level: SigmaLevel, mu: u32, sigma: u32) -> u32 {
        let x = quantile_of_sorted(&self.sorted, level.gaussian_tail_per_10000());
        let delta = i128::from(x) - i128::from(self.mu);
        let offset = if self.sigma == 0 {
            delta
        } else {
            (delta * i128::from(sigma)).div_euclid(i128::from(self.sigma))
        };
        (i128::from(mu) + offset).clamp(0, i128::from(u32::MAX)) as u32
    }
}

/// The distribution family a [`RollingFloor`] currently believes in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Shape {
    /// Normal: levels are located analytically as `μ − k·σ/4`.
    Gaussian,
    /// Not normal: levels are located through a learned sample.
    Empirical(EmpiricalShape),
}

/// A detected drift of the running parameters away from the anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloorShift {
    /// Anchor mean before the shift.
    pub old_mu: u32,
    /// Running mean at the checkpoint that raised the shift.
    pub new_mu: u32,
    /// Anchor standard deviation before the shift.
    pub old_sigma: u32,
    /// Running standard deviation at that checkpoint.
    pub new_sigma: u32,
    /// Observation count at that checkpoint.
    pub observations: u64,
}

/// Live distribution of a stream of `u32` distances, answering σ-lattice
/// thresholds on demand.
///
/// # Example
///
/// ```
/// use ndarray::hpc::rolling_floor::{RollingFloor, SigmaLevel};
/// let mut floor = RollingFloor::for_width(16384);
/// // No observations yet: the binomial prior μ = 8192, σ = 64 answers.
/// assert_eq!(floor.threshold(SigmaLevel(12)), 8000);
/// // The first observation is already a valid current state.
/// floor.observe(8100);
/// assert_eq!(floor.coordinates(), Some((8100, 0)));
/// // Three sensitivities of one detector; the response's shade is how many
/// // of them it survives.
/// for d in 0..3000u32 {
///     if let Some(shift) = floor.observe(8192 + (d % 7)) {
///         floor.recalibrate(&shift);
///     }
/// }
/// let levels = [SigmaLevel(6), SigmaLevel(8), SigmaLevel(12)];
/// assert_eq!(floor.shade(0, &levels), 3);
/// assert_eq!(floor.shade(u32::MAX, &levels), 0);
/// ```
#[derive(Debug, Clone)]
pub struct RollingFloor {
    anchor_mu: u32,
    anchor_sigma: u32,
    moments: MomentsU32,
    reservoir: ReservoirU32,
    shape: Shape,
    skewness: i32,
    kurtosis: u32,
}

impl RollingFloor {
    /// Reservoir capacity.
    pub const RESERVOIR_CAP: usize = 1000;
    /// Checkpoints fall where the observation count is a multiple of this and
    /// larger than it.
    pub const EVAL_CADENCE: u64 = 1000;
    /// Minimum reservoir population before the shape is judged.
    pub const MIN_SHAPE_SAMPLES: usize = 100;
    /// Kurtosis ×100 of the normal distribution.
    pub const NORMAL_KURTOSIS: u32 = 300;

    /// A floor with only a prior `(μ, σ)`: Gaussian shape, no observations.
    pub fn from_params(mu: u32, sigma: u32) -> Self {
        Self {
            anchor_mu: mu,
            anchor_sigma: sigma,
            moments: MomentsU32::default(),
            reservoir: ReservoirU32::new(Self::RESERVOIR_CAP),
            shape: Shape::Gaussian,
            skewness: 0,
            kurtosis: Self::NORMAL_KURTOSIS,
        }
    }

    /// Resume from an anchor `(μ, σ)` and already-accumulated moments, with an
    /// empty reservoir and Gaussian shape.
    pub fn from_params_and_moments(mu: u32, sigma: u32, moments: MomentsU32) -> Self {
        let mut floor = Self::from_params(mu, sigma);
        floor.moments = moments;
        floor
    }

    /// Binomial prior for the Hamming distance of two random
    /// `total_bits`-bit vectors: `μ = bits/2`, `σ = max(1, ⌊√(bits/4)⌋)`.
    pub fn for_width(total_bits: u32) -> Self {
        Self::from_params(total_bits / 2, isqrt_u32(total_bits / 4).max(1))
    }

    /// Calibrate from a warm-up sample of at least two distances.
    ///
    /// The anchor is `μ = ⌊Σx/n⌋` and `σ = max(1, ⌊√⌊Σ(x − μ)²/n⌋⌋)`, spread
    /// around that integer mean exactly as the reference does. The sample
    /// seeds the moments and the reservoir; the shape starts Gaussian.
    ///
    /// # Panics
    ///
    /// If `sample` has fewer than two values.
    pub fn calibrate(sample: &[u32]) -> Self {
        assert!(sample.len() > 1, "need at least 2 samples to calibrate");
        let moments = moments_u32(sample);
        let mu = saturate_u32(moments.sum / u128::from(moments.n));
        let sigma = isqrt_u32(saturate_u32(centred_on_floor_mean(&moments) / u128::from(moments.n))).max(1);
        let mut floor = Self::from_params_and_moments(mu, sigma, moments);
        for &d in sample {
            floor.reservoir.observe(d);
        }
        floor
    }

    /// Fold one observation in. At a checkpoint, returns a shift when the
    /// running parameters have drifted from the anchor:
    /// `|μ_run − μ| > σ/2` or `|σ_run − σ| > σ/4`.
    #[inline]
    pub fn observe(&mut self, distance: u32) -> Option<FloorShift> {
        self.moments.observe(distance);
        self.reservoir.observe(distance);
        if self.at_checkpoint() {
            self.checkpoint()
        } else {
            None
        }
    }

    /// Fold a batch in, stopping right after the first checkpoint that
    /// raises a shift. Returns how many values were consumed and the shift.
    ///
    /// Feeding the unconsumed rest in after acting on the shift reproduces the
    /// scalar loop `if let Some(s) = observe(d) { recalibrate(&s) }` exactly,
    /// for any batching.
    pub fn observe_batch(&mut self, distances: &[u32]) -> (usize, Option<FloorShift>) {
        let mut consumed = 0;
        while consumed < distances.len() {
            let to_checkpoint = (Self::EVAL_CADENCE - self.moments.n % Self::EVAL_CADENCE) as usize;
            let take = to_checkpoint.min(distances.len() - consumed);
            let chunk = &distances[consumed..consumed + take];
            self.moments = self.moments.merge(moments_u32(chunk));
            for &d in chunk {
                self.reservoir.observe(d);
            }
            consumed += take;
            if take == to_checkpoint && self.at_checkpoint() {
                if let Some(shift) = self.checkpoint() {
                    return (consumed, Some(shift));
                }
            }
        }
        (consumed, None)
    }

    /// Adopt the shifted parameters as the new anchor and forget the
    /// accumulated evidence (moments and reservoir). The shape is kept:
    /// parameter drift is not shape drift. `σ` is floored at 1.
    pub fn recalibrate(&mut self, shift: &FloorShift) {
        self.anchor_mu = shift.new_mu;
        self.anchor_sigma = shift.new_sigma.max(1);
        self.moments = MomentsU32::default();
        self.reservoir = ReservoirU32::new(Self::RESERVOIR_CAP);
    }

    fn at_checkpoint(&self) -> bool {
        let n = self.moments.n;
        n.is_multiple_of(Self::EVAL_CADENCE) && n > Self::EVAL_CADENCE
    }

    /// The periodic path: drift first; the shape only when there is none.
    fn checkpoint(&mut self) -> Option<FloorShift> {
        let run_mu = saturate_u32(self.moments.sum / u128::from(self.moments.n));
        let run_sigma = isqrt_u32(saturate_u32(variance_floor(&self.moments))).max(1);

        let mu_drift = run_mu.abs_diff(self.anchor_mu);
        let sigma_drift = run_sigma.abs_diff(self.anchor_sigma);
        if mu_drift > self.anchor_sigma / 2 || sigma_drift > self.anchor_sigma / 4 {
            // The evidence spans two parameter regimes; do not read the shape
            // from it.
            return Some(FloorShift {
                old_mu: self.anchor_mu,
                new_mu: run_mu,
                old_sigma: self.anchor_sigma,
                new_sigma: run_sigma,
                observations: self.moments.n,
            });
        }

        if self.reservoir.len() >= Self::MIN_SHAPE_SAMPLES {
            let sorted = self.reservoir.sorted();
            self.skewness = skewness_from_median(run_mu, run_sigma, quantile_of_sorted(&sorted, 5000));
            self.kurtosis = self.reservoir.kurtosis(run_mu, run_sigma);
            self.shape = if self.shape_is_normal() {
                Shape::Gaussian
            } else {
                // `sorted` is non-empty here.
                EmpiricalShape::from_sample(&sorted).map_or(Shape::Gaussian, Shape::Empirical)
            };
        }
        None
    }

    /// The reference normality window: `|skew| < 2` and `200 < kurt < 500`.
    pub fn shape_is_normal(&self) -> bool {
        self.skewness.abs() < 2 && self.kurtosis > 200 && self.kurtosis < 500
    }

    /// Current coordinates `(μ_t, σ_t)` from the running moments:
    /// `⌊Σx/n⌋` and `⌊√⌊M2/n⌋⌋`. `None` only before the first observation;
    /// after one observation they are `(x, 0)`.
    pub fn coordinates(&self) -> Option<(u32, u32)> {
        if self.moments.n == 0 {
            return None;
        }
        Some((
            saturate_u32(self.moments.sum / u128::from(self.moments.n)),
            isqrt_u32(saturate_u32(variance_floor(&self.moments))),
        ))
    }

    /// The coordinates a threshold is located in: the running ones, or the
    /// anchor when nothing has been observed yet.
    fn frame(&self) -> (u32, u32) {
        self.coordinates()
            .unwrap_or((self.anchor_mu, self.anchor_sigma))
    }

    fn locate(&self, level: SigmaLevel, mu: u32, sigma: u32) -> u32 {
        match &self.shape {
            Shape::Gaussian => mu.saturating_sub(level.quarters() * sigma / 4),
            Shape::Empirical(e) => e.locate(level, mu, sigma),
        }
    }

    /// The threshold of one σ-lattice point in the current distribution.
    /// With `σ_t = 0` every Gaussian level sits at `μ_t`.
    pub fn threshold(&self, level: SigmaLevel) -> u32 {
        let (mu, sigma) = self.frame();
        self.locate(level, mu, sigma)
    }

    /// Thresholds of several lattice points, reading the coordinates once.
    pub fn thresholds<const N: usize>(&self, levels: &[SigmaLevel; N]) -> [u32; N] {
        let (mu, sigma) = self.frame();
        levels.map(|l| self.locate(l, mu, sigma))
    }

    /// How many of `levels` the response `x` survives: the number whose
    /// threshold lies strictly above `x` (lower distance is stronger). This is
    /// the response's shade on the detector's own lattice.
    pub fn shade<const N: usize>(&self, x: u32, levels: &[SigmaLevel; N]) -> usize {
        self.thresholds(levels).iter().filter(|&&t| x < t).count()
    }

    /// The current shape belief.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Whether the shape is empirical.
    pub fn is_empirical(&self) -> bool {
        matches!(self.shape, Shape::Empirical(_))
    }

    /// Anchor mean, the drift reference.
    pub fn mu(&self) -> u32 {
        self.anchor_mu
    }

    /// Anchor standard deviation, the drift reference.
    pub fn sigma(&self) -> u32 {
        self.anchor_sigma
    }

    /// Skewness at the last shape check (`0` before any).
    pub fn skewness(&self) -> i32 {
        self.skewness
    }

    /// Kurtosis ×100 at the last shape check (300 before any).
    pub fn kurtosis(&self) -> u32 {
        self.kurtosis
    }

    /// Exact running moments since calibration or the last recalibration.
    pub fn moments(&self) -> MomentsU32 {
        self.moments
    }

    /// Observation count since calibration or the last recalibration.
    pub fn observations(&self) -> u64 {
        self.moments.n
    }

    /// The reservoir, the shape check's evidence.
    pub fn reservoir(&self) -> &ReservoirU32 {
        &self.reservoir
    }
}

fn saturate_u32(x: u128) -> u32 {
    u32::try_from(x).unwrap_or(u32::MAX)
}

/// `Σ(x − q)²` with `q = ⌊Σx / n⌋`, exactly. With `r = Σx − n·q` this is
/// `Σx² − q·Σx − q·r`, and no intermediate goes negative. Requires `n > 0`.
fn centred_on_floor_mean(m: &MomentsU32) -> u128 {
    let n = u128::from(m.n);
    let q = m.sum / n;
    let r = m.sum % n;
    m.sum_sq - q * m.sum - q * r
}

/// `⌊M2 / n⌋` for the exact population variance, with `M2 = Σ(x − mean)²`.
/// Requires `n > 0`.
fn variance_floor(m: &MomentsU32) -> u128 {
    let n = u128::from(m.n);
    if let (Some(a), Some(b)) = (n.checked_mul(m.sum_sq), m.sum.checked_mul(m.sum)) {
        return (a - b) / (n * n);
    }
    variance_floor_centred(m)
}

/// The overflow-free form of [`variance_floor`], valid for every `n > 0`.
/// `n·M2 = C·n − r²`, with `C` centred on the floor mean. Write
/// `C = a·n + b`; then `⌊(C·n − r²)/n²⌋ = a − [b·n < r²]`, and both `b·n`
/// and `r²` are below `2^128`.
fn variance_floor_centred(m: &MomentsU32) -> u128 {
    let n = u128::from(m.n);
    let c = centred_on_floor_mean(m);
    let r = m.sum % n;
    let (a, b) = (c / n, c % n);
    if b * n >= r * r {
        a
    } else {
        a - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(n: usize, base: u32, spread: u32, mut s: u64) -> Vec<u32> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                base + (s % u64::from(spread)) as u32
            })
            .collect()
    }

    /// Roughly normal around `mu` with the given spread (sum of 12 uniforms).
    fn normalish(n: usize, mu: u32, sigma: u32, mut s: u64) -> Vec<u32> {
        (0..n)
            .map(|_| {
                let mut acc = 0i64;
                for _ in 0..12 {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    acc += (s % 1000) as i64;
                }
                // Sum of 12 U(0,1000) has sd ≈ 1000.
                (i64::from(mu) + (acc - 6000) * i64::from(sigma) / 1000).max(0) as u32
            })
            .collect()
    }

    /// Scalar reference loop: observe, recalibrate on every shift.
    fn run_scalar(mut f: RollingFloor, xs: &[u32]) -> (RollingFloor, Vec<FloorShift>) {
        let mut shifts = Vec::new();
        for &d in xs {
            if let Some(s) = f.observe(d) {
                f.recalibrate(&s);
                shifts.push(s);
            }
        }
        (f, shifts)
    }

    fn run_batched(mut f: RollingFloor, xs: &[u32], chunk: usize) -> (RollingFloor, Vec<FloorShift>) {
        let mut shifts = Vec::new();
        for c in xs.chunks(chunk) {
            let mut rest = c;
            while !rest.is_empty() {
                let (used, s) = f.observe_batch(rest);
                rest = &rest[used..];
                if let Some(s) = s {
                    f.recalibrate(&s);
                    shifts.push(s);
                }
            }
        }
        (f, shifts)
    }

    const LATTICE: [SigmaLevel; 9] = [
        SigmaLevel(0),
        SigmaLevel(4),
        SigmaLevel(6),
        SigmaLevel(7),
        SigmaLevel(8),
        SigmaLevel(9),
        SigmaLevel(10),
        SigmaLevel(11),
        SigmaLevel(12),
    ];

    fn same_state(a: &RollingFloor, b: &RollingFloor) {
        assert_eq!(a.moments(), b.moments());
        assert_eq!(a.reservoir(), b.reservoir());
        assert_eq!((a.mu(), a.sigma()), (b.mu(), b.sigma()));
        assert_eq!(a.shape(), b.shape());
        assert_eq!((a.skewness(), a.kurtosis()), (b.skewness(), b.kurtosis()));
        assert_eq!(a.thresholds(&LATTICE), b.thresholds(&LATTICE));
    }

    #[test]
    fn isqrt_is_floor_sqrt() {
        for n in (0..200_000u32).chain([u32::MAX, u32::MAX - 1, 65535 * 65535, 65536 * 65535]) {
            let r = isqrt_u32(n);
            assert!(u64::from(r) * u64::from(r) <= u64::from(n));
            assert!(u64::from(r + 1) * u64::from(r + 1) > u64::from(n));
        }
    }

    /// `Φ(−k/4)` at the quarter-σ lattice, per 10 000.
    #[test]
    fn gaussian_tail_table_is_phi() {
        let expect = [
            (0, 5000),
            (4, 1587),
            (6, 668),
            (7, 401),
            (8, 228),
            (9, 122),
            (10, 62),
            (11, 30),
            (12, 13),
            (16, 0),
            (40, 0),
        ];
        for (k, v) in expect {
            assert_eq!(SigmaLevel(k).gaussian_tail_per_10000(), v, "k {k}");
        }
        // Monotone: a deeper cut is rarer.
        for k in 0..16u8 {
            assert!(SigmaLevel(k).gaussian_tail_per_10000() >= SigmaLevel(k + 1).gaussian_tail_per_10000());
        }
    }

    /// The integer rank rule against the reference `f32` rule
    /// `⌊(q as f32)·len⌋`, for every reservoir length 1..=1000.
    ///
    /// The eight cascade levels match the reference exactly. The three old
    /// band percentiles (0.159, 0.023, 0.001) were coarse approximations of
    /// 1σ, 2σ and 3σ; unifying them onto the lattice ranks changes the sample
    /// index at exactly these many lengths, pinned so the delta is explicit.
    #[test]
    fn integer_rank_rule_against_the_reference() {
        let f32_rank = |q: f32, len: usize| ((q * len as f32) as usize).min(len - 1);
        let cascade = [
            (4u8, 0.1587f32),
            (6, 0.0668),
            (7, 0.0401),
            (8, 0.0228),
            (9, 0.0122),
            (10, 0.0062),
            (11, 0.0030),
            (12, 0.0013),
        ];
        for len in 1..=1000 {
            for (k, q) in cascade {
                assert_eq!(
                    rank_per_10000(len, SigmaLevel(k).gaussian_tail_per_10000()),
                    f32_rank(q, len),
                    "k {k} len {len}"
                );
            }
            assert_eq!(rank_per_10000(len, SigmaLevel(0).gaussian_tail_per_10000()), f32_rank(0.5, len));
        }
        let changed = |k: u8, q: f32| {
            (1..=1000)
                .filter(|&len| rank_per_10000(len, SigmaLevel(k).gaussian_tail_per_10000()) != f32_rank(q, len))
                .count()
        };
        assert_eq!(changed(4, 0.159), 150);
        assert_eq!(changed(8, 0.023), 98);
        assert_eq!(changed(12, 0.001), 230);
        // At the full reservoir: 1σ 159 → 158, 2σ 23 → 22, 3σ 1 → 1.
        assert_eq!((f32_rank(0.159, 1000), rank_per_10000(1000, 1587)), (159, 158));
        assert_eq!((f32_rank(0.023, 1000), rank_per_10000(1000, 228)), (23, 22));
        assert_eq!((f32_rank(0.001, 1000), rank_per_10000(1000, 13)), (1, 1));
    }

    #[test]
    fn scalar_equals_singleton_batch() {
        let xs = stream(5000, 8000, 300, 11);
        let mut a = RollingFloor::for_width(16384);
        let mut b = RollingFloor::for_width(16384);
        for &d in &xs {
            let sa = a.observe(d);
            let (used, sb) = b.observe_batch(&[d]);
            assert_eq!(used, 1);
            assert_eq!(sa, sb);
            if let Some(s) = sa {
                a.recalibrate(&s);
                b.recalibrate(&s);
            }
        }
        same_state(&a, &b);
    }

    /// Any batching reproduces the scalar loop, including shifts raised in
    /// the middle of a batch.
    #[test]
    fn arbitrary_batching_equals_scalar_stream() {
        let mut xs = normalish(4000, 8192, 64, 3);
        xs.extend(normalish(6000, 8900, 64, 4)); // forces mid-stream shifts
        let (s, s_shifts) = run_scalar(RollingFloor::for_width(16384), &xs);
        assert!(!s_shifts.is_empty(), "fixture must raise a shift");
        for chunk in [1, 7, 999, 1000, 1001, 2500, 10_000] {
            let (b, b_shifts) = run_batched(RollingFloor::for_width(16384), &xs, chunk);
            assert_eq!(s_shifts, b_shifts, "chunk {chunk}");
            same_state(&s, &b);
        }
    }

    #[test]
    fn checkpoint_cadence_is_every_1000_after_the_first() {
        // An anchor far from the data: every checkpoint must alert.
        let mut f = RollingFloor::from_params(100, 1);
        let mut at = Vec::new();
        for i in 1..=4000u64 {
            if f.observe(9000).is_some() {
                at.push(i);
            }
        }
        assert_eq!(at, [2000, 3000, 4000]);
    }

    #[test]
    fn drift_boundary_is_strict() {
        // Anchor σ = 8: shift iff |Δμ| > 4 or |Δσ| > 2.
        for (value, expect) in [(104u32, false), (105, true)] {
            let mut f = RollingFloor::from_params(100, 8);
            let mut hit = None;
            for i in 0..2000u32 {
                hit = f.observe(if i % 2 == 0 { value - 8 } else { value + 8 });
            }
            assert_eq!(hit.is_some(), expect, "mean {value}");
        }
        for (half, expect) in [(10u32, false), (11, true)] {
            let mut f = RollingFloor::from_params(100, 8);
            let mut hit = None;
            for i in 0..2000u32 {
                hit = f.observe(if i % 2 == 0 { 100 - half } else { 100 + half });
            }
            assert_eq!(hit.is_some(), expect, "σ {half}");
        }
    }

    /// The ballot box: no running coordinates before the first observation,
    /// a valid (noisy) state after it, and σ = 0 collapses every Gaussian cut
    /// onto μ rather than borrowing the anchor.
    #[test]
    fn coordinates_are_valid_from_the_first_observation() {
        let mut f = RollingFloor::from_params(5000, 50);
        assert_eq!(f.coordinates(), None);
        assert_eq!(f.threshold(SigmaLevel(8)), 4900, "prior answers before any observation");
        f.observe(7000);
        assert_eq!(f.coordinates(), Some((7000, 0)));
        assert_eq!(f.thresholds(&[SigmaLevel(0), SigmaLevel(4), SigmaLevel(12)]), [7000, 7000, 7000]);
        let levels = [SigmaLevel(4), SigmaLevel(8), SigmaLevel(12)];
        assert_eq!(f.shade(6999, &levels), 3);
        assert_eq!(f.shade(7000, &levels), 0);
        f.observe(7010);
        assert_eq!(f.coordinates(), Some((7005, 5)));
    }

    /// Gaussian thresholds follow the running coordinates on every
    /// observation, with no recalibration: exactly `μ_t − k·σ_t/4`.
    #[test]
    fn gaussian_thresholds_follow_current_moments() {
        let mut f = RollingFloor::calibrate(&normalish(1000, 5000, 40, 5));
        let anchored = f.thresholds(&LATTICE);
        let mut moved = false;
        // A slow drift, small enough never to raise a shift.
        for (i, d) in normalish(900, 5015, 40, 6).into_iter().enumerate() {
            assert!(f.observe(d).is_none());
            let (mu, s) = f.coordinates().unwrap();
            let want = LATTICE.map(|l| mu.saturating_sub(l.quarters() * s / 4));
            assert_eq!(f.thresholds(&LATTICE), want, "observation {i}");
            moved |= want != anchored;
        }
        assert!(moved, "the floor must move without recalibration");
    }

    /// Half-σ lattice points reproduce the integer `SigmaGate::custom` tiers
    /// bit for bit.
    #[test]
    fn half_sigma_levels_match_the_sigma_gate() {
        use crate::hpc::kernels::SigmaGate;
        for (mu, s) in [(8192u32, 64u32), (5000, 51), (1000, 7), (100, 1)] {
            let f = RollingFloor::from_params(mu, s);
            let g = SigmaGate::custom(mu, s);
            let t = f.thresholds(&[SigmaLevel(12), SigmaLevel(10), SigmaLevel(8), SigmaLevel(6)]);
            assert_eq!(t, [g.discovery, g.strong, g.evidence, g.hint], "mu {mu} sigma {s}");
        }
    }

    #[test]
    fn shade_counts_the_levels_a_response_survives() {
        let f = RollingFloor::from_params(8192, 64);
        let levels = [SigmaLevel(12), SigmaLevel(6), SigmaLevel(8)]; // order is free
                                                                     // Cuts at 8000 (3σ), 8096 (1.5σ), 8064 (2σ).
        assert_eq!(f.shade(7999, &levels), 3);
        assert_eq!(f.shade(8000, &levels), 2);
        assert_eq!(f.shade(8063, &levels), 2);
        assert_eq!(f.shade(8064, &levels), 1);
        assert_eq!(f.shade(8095, &levels), 1);
        assert_eq!(f.shade(8096, &levels), 0);
    }

    /// Recalibration forgets the evidence and moves the anchor, but keeps the
    /// shape.
    #[test]
    fn recalibration_forgets_evidence_and_keeps_shape() {
        let (lo, hi) = (normalish(1000, 7800, 20, 3), normalish(1000, 8600, 20, 4));
        let bimodal: Vec<u32> = lo.iter().zip(&hi).flat_map(|(&a, &b)| [a, b]).collect();
        let mut f = RollingFloor::calibrate(&bimodal[..1000]);
        f.observe_batch(&bimodal[1000..]);
        assert!(f.is_empirical());
        let shape = f.shape().clone();
        let shift = FloorShift {
            old_mu: f.mu(),
            new_mu: 9000,
            old_sigma: f.sigma(),
            new_sigma: 300,
            observations: f.observations(),
        };
        f.recalibrate(&shift);
        assert_eq!((f.mu(), f.sigma()), (9000, 300));
        assert_eq!(f.observations(), 0);
        assert!(f.reservoir().is_empty());
        assert_eq!(f.reservoir().capacity(), RollingFloor::RESERVOIR_CAP);
        assert_eq!(f.shape(), &shape);
    }

    /// A pure location and spread change of a normal stream is parameter
    /// drift only: the drift checkpoint does not read the mixed evidence as a
    /// new shape, and the shape stays Gaussian throughout.
    #[test]
    fn pure_parameter_shift_keeps_the_gaussian_shape() {
        let mut f = RollingFloor::calibrate(&normalish(1000, 5000, 40, 7));
        let shifted = normalish(8000, 5600, 90, 8);
        let (used, shift) = f.observe_batch(&shifted);
        let shift = shift.expect("parameters moved");
        assert_eq!(shift.observations, 2000);
        assert_eq!(f.shape(), &Shape::Gaussian, "mixed evidence must not relearn the shape");
        f.recalibrate(&shift);
        let (f, later) = run_scalar(f, &shifted[used..]);
        assert!(later.len() <= 1, "{later:?}");
        assert_eq!(f.shape(), &Shape::Gaussian);
        assert!(f.shape_is_normal(), "skew {} kurt {}", f.skewness(), f.kurtosis());
        let (mu, s) = f.coordinates().unwrap();
        assert!(mu.abs_diff(5600) <= 5 && s.abs_diff(90) <= 5, "{mu} {s}");
    }

    #[test]
    fn reservoir_is_deterministic_and_bounded() {
        let xs = stream(20_000, 0, 1 << 20, 5);
        let mut a = ReservoirU32::new(1000);
        let mut b = ReservoirU32::new(1000);
        xs.iter().for_each(|&d| a.observe(d));
        xs.iter().for_each(|&d| b.observe(d));
        assert_eq!(a, b);
        assert_eq!(a.len(), 1000);
        assert_eq!(a.seen(), 20_000);
        assert_ne!(a.samples(), &xs[..1000]);
    }

    #[test]
    fn normal_stream_stays_gaussian_and_bimodal_goes_empirical() {
        let mut f = RollingFloor::calibrate(&normalish(1000, 8192, 64, 1));
        f.observe_batch(&normalish(1000, 8192, 64, 2));
        assert!(f.shape_is_normal(), "skew {} kurt {}", f.skewness(), f.kurtosis());
        assert_eq!(f.shape(), &Shape::Gaussian);

        let (lo, hi) = (normalish(1000, 7800, 20, 3), normalish(1000, 8600, 20, 4));
        let bimodal: Vec<u32> = lo.iter().zip(&hi).flat_map(|(&a, &b)| [a, b]).collect();
        let mut g = RollingFloor::calibrate(&bimodal[..1000]);
        g.observe_batch(&bimodal[1000..]);
        assert!(g.is_empirical(), "skew {} kurt {}", g.skewness(), g.kurtosis());
    }

    /// An empirical shape answers with the raw sample quantile in its own
    /// frame, translates with μ and scales with σ.
    #[test]
    fn empirical_shape_projects_into_current_coordinates() {
        let (lo, hi) = (normalish(500, 7800, 20, 3), normalish(500, 8600, 20, 4));
        let sample: Vec<u32> = lo.into_iter().chain(hi).collect();
        let e = EmpiricalShape::from_sample(&sample).unwrap();
        let (mu, s) = (e.mu(), e.sigma());
        for l in LATTICE {
            let x = quantile_of_sorted(e.sorted(), l.gaussian_tail_per_10000());
            assert_eq!(e.locate(l, mu, s), x, "own frame, k {}", l.0);
            assert_eq!(e.locate(l, mu + 100, s), x + 100, "translation, k {}", l.0);
            let d = i64::from(x) - i64::from(mu);
            let doubled = i64::from(mu) + (d * 2 * i64::from(s)).div_euclid(i64::from(s));
            assert_eq!(i64::from(e.locate(l, mu, 2 * s)), doubled, "scale, k {}", l.0);
        }
        assert_eq!(EmpiricalShape::from_sample(&[]), None);
        // No spread: the offset is used unscaled.
        let flat = EmpiricalShape::from_sample(&[7, 7, 7]).unwrap();
        assert_eq!((flat.sigma(), flat.locate(SigmaLevel(12), 100, 50)), (0, 100));
    }

    /// Each kurtosis bound switches to the empirical shape on its own, with
    /// the skew inside the window.
    #[test]
    fn kurtosis_alone_switches_to_empirical() {
        let uniform = stream(2000, 8000, 400, 21);
        let mut u = RollingFloor::calibrate(&uniform[..1000]);
        u.observe_batch(&uniform[1000..]);
        assert!(u.skewness().abs() < 2 && u.kurtosis() <= 200, "skew {} kurt {}", u.skewness(), u.kurtosis());
        assert!(u.is_empirical());

        let (core, tail) = (normalish(2000, 8192, 10, 22), normalish(200, 8192, 120, 23));
        let mut mix = core;
        for (i, t) in tail.into_iter().enumerate() {
            mix[i * 9] = t;
        }
        let mut h = RollingFloor::calibrate(&mix[..1000]);
        h.observe_batch(&mix[1000..]);
        assert!(h.skewness().abs() < 2 && h.kurtosis() >= 500, "skew {} kurt {}", h.skewness(), h.kurtosis());
        assert!(h.is_empirical());
    }

    /// The normality window at each of its boundaries.
    #[test]
    fn normality_window_boundaries() {
        let mut f = RollingFloor::for_width(16384);
        for (skew, kurt, normal) in [
            (0, 300, true),
            (1, 300, true),
            (-1, 300, true),
            (2, 300, false),
            (-2, 300, false),
            (0, 200, false),
            (0, 201, true),
            (0, 499, true),
            (0, 500, false),
        ] {
            f.skewness = skew;
            f.kurtosis = kurt;
            assert_eq!(f.shape_is_normal(), normal, "skew {skew} kurt {kurt}");
        }
    }

    /// Usable from the first observation, refined by more of them without
    /// any reset.
    #[test]
    fn anytime_use_refines_with_population() {
        let mut f = RollingFloor::for_width(16384);
        let xs = normalish(50_000, 8192, 64, 12);
        let mut errs = Vec::new();
        for (i, &d) in xs.iter().enumerate() {
            assert!(f.observe(d).is_none(), "on-prior data must not drift");
            if [100, 1000, 50_000].contains(&(i + 1)) {
                errs.push(f.coordinates().unwrap().1.abs_diff(64));
            }
        }
        assert_eq!(f.observations(), 50_000);
        assert!(errs[2] <= errs[0], "{errs:?}");
    }

    #[test]
    fn large_population_variance_floor_is_exact() {
        let half = 1u128 << 32;
        let hi = u128::from(u32::MAX);
        let m = MomentsU32 {
            n: 1 << 33,
            sum: half * (hi + hi - 2),
            sum_sq: half * (hi * hi + (hi - 2) * (hi - 2)),
        };
        assert!(u128::from(m.n).checked_mul(m.sum_sq).is_none());
        assert_eq!(variance_floor(&m), 1);
        let m = MomentsU32 {
            n: 4,
            sum: 10,
            sum_sq: 30,
        };
        assert_eq!(variance_floor(&m), 1);
    }

    /// The overflow-free variance form agrees with the direct one on every
    /// small sample, including the fractional means that take the `a − 1`
    /// correction.
    #[test]
    fn centred_variance_floor_matches_the_direct_form() {
        let mut corrected = 0;
        for seed in 1..400u64 {
            let len = 2 + (seed % 37) as usize;
            let xs = stream(len, (seed * 97 % 5000) as u32, 1 + (seed % 60) as u32, seed);
            let m = moments_u32(&xs);
            let n = u128::from(m.n);
            let direct = (n * m.sum_sq - m.sum * m.sum) / (n * n);
            assert_eq!(variance_floor_centred(&m), direct, "{xs:?}");
            let c = centred_on_floor_mean(&m);
            corrected += usize::from((c % n) * n < (m.sum % n).pow(2));
        }
        assert!(corrected > 0, "fixture must exercise the correction branch");
    }

    #[test]
    fn calibrate_uses_spread_around_the_integer_mean() {
        let f = RollingFloor::calibrate(&[0, 0, 0, 1]);
        assert_eq!((f.mu(), f.sigma()), (0, 1));
        let f = RollingFloor::calibrate(&[100, 120, 100, 120]);
        assert_eq!((f.mu(), f.sigma()), (110, 10));
        assert_eq!(f.observations(), 4);
    }

    // ── The lance-graph reference, kept verbatim as an oracle ───────────
    struct LegacyWelford {
        n: u64,
        sum: u64,
        m2: u64,
    }
    impl LegacyWelford {
        fn observe(&mut self, d: u32) -> Option<(u32, u32)> {
            let d = u64::from(d);
            self.n += 1;
            self.sum += d;
            let old = if self.n > 1 { (self.sum - d) / (self.n - 1) } else { d };
            let new = self.sum / self.n;
            self.m2 = self
                .m2
                .wrapping_add(((d as i64 - old as i64) * (d as i64 - new as i64)) as u64);
            if self.n.is_multiple_of(1000) && self.n > 1000 {
                Some((new as u32, isqrt_u32((self.m2 / self.n) as u32).max(1)))
            } else {
                None
            }
        }
    }

    /// At every checkpoint, the running (μ, σ) the drift rule sees match the
    /// legacy integer Welford on these streams.
    #[test]
    fn checkpoint_parameters_match_legacy_welford() {
        let mut checked = 0;
        for (mu, sigma, seed) in [(8192, 64, 1), (5000, 50, 2), (100, 3, 3), (8192, 1, 4), (16384, 91, 5)] {
            let xs = normalish(20_000, mu, sigma, seed);
            let mut legacy = LegacyWelford { n: 0, sum: 0, m2: 0 };
            let mut m = MomentsU32::default();
            for &d in &xs {
                m.observe(d);
                if let Some((lmu, lsig)) = legacy.observe(d) {
                    let emu = saturate_u32(m.sum / u128::from(m.n));
                    let esig = isqrt_u32(saturate_u32(variance_floor(&m))).max(1);
                    assert_eq!((emu, esig), (lmu, lsig), "mu {mu} sigma {sigma} n {}", m.n);
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 5 * 19);
    }
}
