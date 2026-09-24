//! HDR rolling floor: an online distribution floor for popcount / Hamming
//! observations.
//!
//! This is the adaptive half of the HDR exposure meter, harvested from
//! lance-graph's `graph/blasgraph/hdr.rs`. That file stays the behavioural
//! reference: constants, cadence, drift rule and reset semantics are carried
//! over unchanged. What changed is the arithmetic underneath. The old
//! approximate integer Welford is replaced by the exact, mergeable
//! [`MomentsU32`].
//!
//! # Two layers, two cadences
//!
//! * **Parameters (continuous, cheap).** Every observation folds into exact
//!   `(n, Σx, Σx²)`. Location and spread are available at any time, so the
//!   floor can be used from the first observation and only gets better as the
//!   population grows. There is no training barrier.
//! * **Shape (periodic, amortised).** A deterministic Algorithm-R reservoir
//!   feeds median, empirical quantiles, skewness and kurtosis. They are
//!   evaluated once every [`RollingFloor::EVAL_CADENCE`] observations, never on
//!   the per-observation path.
//!
//! # Floors
//!
//! While the shape reads as normal, the floors are the analytical quantiles of
//! the calibrated `(μ, σ)`: `μ − kσ`, i.e. `μ + σ·Φ⁻¹(p)` at the percentiles
//! the empirical tables name. When the shape does not read as normal, the
//! floors are the reservoir's empirical quantiles at those same percentiles.
//!
//! # What is and is not mergeable
//!
//! The moments are exact and partition-independent: shards can be merged in
//! any order. The reservoir is not. It is deterministic for a given
//! observation *order* and has no merge law. [`RollingFloor::observe_batch`]
//! therefore reproduces the scalar stream exactly rather than merging shards.
//!
//! # Reference behaviour this preserves, including its limits
//!
//! * Floors move only when a drift alert is acted on
//!   ([`RollingFloor::recalibrate`]), not continuously with the running
//!   parameters.
//! * A parameter drift resets the shape layer too (reservoir, empirical mode,
//!   skewness, kurtosis). The reference does not separate parameter drift from
//!   shape drift; neither does this port.

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
/// assert_eq!(r.quantile(0.5), 20);
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

    /// The held samples, sorted ascending. Sort once and use
    /// [`quantile_of_sorted`] when several quantiles are needed.
    pub fn sorted(&self) -> Vec<u32> {
        let mut s = self.samples.clone();
        s.sort_unstable();
        s
    }

    /// Empirical quantile: the sorted sample at index `⌊q·len⌋`, clamped to
    /// the last element. `0` for an empty reservoir.
    ///
    /// O(len · log len); meant for the periodic shape path, not per
    /// observation.
    pub fn quantile(&self, q: f32) -> u32 {
        quantile_of_sorted(&self.sorted(), q)
    }

    /// Pearson's second skewness `3(μ − median) / σ`, in integer division.
    /// Positive means right-skewed, `0` symmetric. `0` when `σ = 0` or the
    /// reservoir is empty.
    pub fn skewness(&self, mu: u32, sigma: u32) -> i32 {
        if sigma == 0 || self.samples.is_empty() {
            return 0;
        }
        skewness_from_median(mu, sigma, self.quantile(0.5))
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

/// Empirical quantile of an ascending slice: the element at `⌊q·len⌋`,
/// clamped to the last element. `0` for an empty slice.
///
/// # Example
///
/// ```
/// use ndarray::hpc::rolling_floor::quantile_of_sorted;
/// let s = [1, 2, 3, 4];
/// assert_eq!(quantile_of_sorted(&s, 0.0), 1);
/// assert_eq!(quantile_of_sorted(&s, 0.5), 3);
/// assert_eq!(quantile_of_sorted(&s, 1.0), 4);
/// ```
pub fn quantile_of_sorted(sorted: &[u32], q: f32) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((q * sorted.len() as f32) as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn skewness_from_median(mu: u32, sigma: u32, median: u32) -> i32 {
    // i64 so no difference of two u32 values can overflow.
    let s = 3 * (i64::from(mu) - i64::from(median)) / i64::from(sigma);
    s.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// A detected drift of the running parameters away from the calibrated ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FloorShift {
    /// Calibrated mean before the shift.
    pub old_mu: u32,
    /// Running mean at the checkpoint that raised the shift.
    pub new_mu: u32,
    /// Calibrated standard deviation before the shift.
    pub old_sigma: u32,
    /// Running standard deviation at that checkpoint.
    pub new_sigma: u32,
    /// Observation count at that checkpoint.
    pub observations: u64,
}

/// The HDR rolling floor for a stream of `u32` distances.
///
/// It holds the calibrated `(μ, σ)`, the sigma floors derived from them, and
/// empirical floors from a reservoir. Running parameters come from exact
/// [`MomentsU32`]. Every [`EVAL_CADENCE`](Self::EVAL_CADENCE) observations
/// (after the first cadence) it re-reads the distribution shape, selects sigma
/// or empirical floors, and checks for drift.
///
/// The floors are lower-is-better thresholds, from strictest to loosest:
/// four band floors at `[μ−3σ, μ−2σ, μ−σ, μ]` and eight cascade floors at
/// quarter-σ steps from `μ−σ` to `μ−3σ`. How a caller turns them into bands
/// is its own policy.
///
/// # Example
///
/// ```
/// use ndarray::hpc::rolling_floor::RollingFloor;
/// let mut floor = RollingFloor::for_width(16384);
/// assert_eq!(floor.mu(), 8192);
/// assert_eq!(floor.sigma(), 64);
/// assert_eq!(floor.active_floors(), [8000, 8064, 8128, 8192]);
/// // Usable immediately; observations refine it as they arrive.
/// for d in 0..3000u32 {
///     if let Some(shift) = floor.observe(8192 + (d % 7)) {
///         floor.recalibrate(&shift);
///     }
/// }
/// ```
#[derive(Debug, Clone)]
pub struct RollingFloor {
    mu: u32,
    sigma: u32,
    sigma_floors: [u32; 4],
    sigma_cascade: [u32; 8],
    reservoir: ReservoirU32,
    empirical_floors: [u32; 4],
    empirical_cascade: [u32; 8],
    use_empirical: bool,
    skewness: i32,
    kurtosis: u32,
    moments: MomentsU32,
}

impl RollingFloor {
    /// Reservoir capacity.
    pub const RESERVOIR_CAP: usize = 1000;
    /// Shape and drift are evaluated when the observation count is a multiple
    /// of this, and larger than it.
    pub const EVAL_CADENCE: u64 = 1000;
    /// Minimum reservoir population before shape is evaluated.
    pub const MIN_SHAPE_SAMPLES: usize = 100;
    /// Kurtosis ×100 of the normal distribution.
    pub const NORMAL_KURTOSIS: u32 = 300;
    /// Percentiles of the empirical band floors: ≈3σ, 2σ, 1σ, median.
    pub const FLOOR_PERCENTILES: [f32; 4] = [0.001, 0.023, 0.159, 0.500];
    /// Percentiles of the empirical cascade floors: 1σ, 1.5σ, 1.75σ, 2σ,
    /// 2.25σ, 2.5σ, 2.75σ, 3σ below the mean.
    pub const CASCADE_PERCENTILES: [f32; 8] = [0.1587, 0.0668, 0.0401, 0.0228, 0.0122, 0.0062, 0.0030, 0.0013];

    /// Band floors `[μ−3σ, μ−2σ, μ−σ, μ]`, saturating at zero.
    pub fn sigma_floors_of(mu: u32, sigma: u32) -> [u32; 4] {
        [mu.saturating_sub(3 * sigma), mu.saturating_sub(2 * sigma), mu.saturating_sub(sigma), mu]
    }

    /// Cascade floors at `μ − kσ/4` for `k = 4, 6, 7, 8, 9, 10, 11, 12`,
    /// saturating at zero.
    pub fn sigma_cascade_of(mu: u32, sigma: u32) -> [u32; 8] {
        [4, 6, 7, 8, 9, 10, 11, 12].map(|k: u32| mu.saturating_sub(k * sigma / 4))
    }

    /// Floors that assume only a prior `(μ, σ)`, with no observations yet.
    pub fn from_params(mu: u32, sigma: u32) -> Self {
        let sigma_floors = Self::sigma_floors_of(mu, sigma);
        let sigma_cascade = Self::sigma_cascade_of(mu, sigma);
        Self {
            mu,
            sigma,
            sigma_floors,
            sigma_cascade,
            reservoir: ReservoirU32::new(Self::RESERVOIR_CAP),
            empirical_floors: sigma_floors,
            empirical_cascade: sigma_cascade,
            use_empirical: false,
            skewness: 0,
            kurtosis: Self::NORMAL_KURTOSIS,
            moments: MomentsU32::default(),
        }
    }

    /// Resume from calibrated `(μ, σ)` and already-accumulated running
    /// moments, with an empty reservoir. The next checkpoint follows from
    /// `moments.n`.
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
    /// `μ` is the floor of the sample mean and `σ` is
    /// `max(1, ⌊√⌊Σ(x − μ)² / n⌋⌋)`, spread measured around that integer
    /// mean, exactly as the reference does. The sample seeds the reservoir and
    /// the running moments; the floors start in sigma mode.
    ///
    /// # Panics
    ///
    /// If `sample` has fewer than two values.
    pub fn calibrate(sample: &[u32]) -> Self {
        assert!(sample.len() > 1, "need at least 2 samples to calibrate");
        let moments = moments_u32(sample);
        let mu = saturate_u32(moments.sum / u128::from(moments.n));
        let sigma = isqrt_u32(saturate_u32(centred_on_floor_mean(&moments) / u128::from(moments.n))).max(1);
        let mut floor = Self::from_params(mu, sigma);
        for &d in sample {
            floor.reservoir.observe(d);
        }
        let sorted = floor.reservoir.sorted();
        floor.empirical_floors = Self::FLOOR_PERCENTILES.map(|p| quantile_of_sorted(&sorted, p));
        floor.empirical_cascade = Self::CASCADE_PERCENTILES.map(|p| quantile_of_sorted(&sorted, p));
        floor.moments = moments;
        floor
    }

    /// Fold one observation in. Returns a shift when this observation lands
    /// on a checkpoint and the running parameters have drifted:
    /// `|μ_run − μ| > σ/2` or `|σ_run − σ| > σ/4`, against the calibrated
    /// `(μ, σ)`.
    #[inline]
    pub fn observe(&mut self, distance: u32) -> Option<FloorShift> {
        self.moments.observe(distance);
        self.reservoir.observe(distance);
        if self.at_checkpoint() {
            self.evaluate()
        } else {
            None
        }
    }

    /// Fold a batch in, stopping right after the first checkpoint that
    /// raises a shift.
    ///
    /// Returns how many values were consumed and the shift, if any. Feeding
    /// the unconsumed rest in after acting on the shift reproduces the scalar
    /// loop `if let Some(s) = observe(d) { recalibrate(&s) }` exactly, for any
    /// batching: the moments between checkpoints go through
    /// [`moments_u32`], the reservoir sees every value in stream order, and
    /// every checkpoint is evaluated on the same state the scalar loop sees.
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
                if let Some(shift) = self.evaluate() {
                    return (consumed, Some(shift));
                }
            }
        }
        (consumed, None)
    }

    /// Adopt the shifted parameters as the new calibration and restart
    /// observation from scratch: running moments, reservoir, empirical mode
    /// and shape diagnostics are all reset. `σ` is floored at 1.
    pub fn recalibrate(&mut self, shift: &FloorShift) {
        let capacity = self.reservoir.capacity();
        *self = Self::from_params(shift.new_mu, shift.new_sigma.max(1));
        self.reservoir = ReservoirU32::new(capacity);
    }

    fn at_checkpoint(&self) -> bool {
        let n = self.moments.n;
        n.is_multiple_of(Self::EVAL_CADENCE) && n > Self::EVAL_CADENCE
    }

    /// The periodic path: shape evaluation, floor selection, drift check.
    fn evaluate(&mut self) -> Option<FloorShift> {
        let run_mu = saturate_u32(self.moments.sum / u128::from(self.moments.n));
        let run_sigma = isqrt_u32(saturate_u32(variance_floor(&self.moments))).max(1);

        if self.reservoir.len() >= Self::MIN_SHAPE_SAMPLES {
            let sorted = self.reservoir.sorted();
            self.skewness = skewness_from_median(run_mu, run_sigma, quantile_of_sorted(&sorted, 0.5));
            self.kurtosis = self.reservoir.kurtosis(run_mu, run_sigma);
            if self.shape_is_normal() {
                self.use_empirical = false;
            } else {
                self.empirical_floors = Self::FLOOR_PERCENTILES.map(|p| quantile_of_sorted(&sorted, p));
                self.empirical_cascade = Self::CASCADE_PERCENTILES.map(|p| quantile_of_sorted(&sorted, p));
                self.use_empirical = true;
            }
        }

        let mu_drift = run_mu.abs_diff(self.mu);
        let sigma_drift = run_sigma.abs_diff(self.sigma);
        if mu_drift > self.sigma / 2 || sigma_drift > self.sigma / 4 {
            Some(FloorShift {
                old_mu: self.mu,
                new_mu: run_mu,
                old_sigma: self.sigma,
                new_sigma: run_sigma,
                observations: self.moments.n,
            })
        } else {
            None
        }
    }

    /// The reference normality window: `|skew| < 2` and `200 < kurt < 500`.
    pub fn shape_is_normal(&self) -> bool {
        self.skewness.abs() < 2 && self.kurtosis > 200 && self.kurtosis < 500
    }

    /// Calibrated mean.
    pub fn mu(&self) -> u32 {
        self.mu
    }

    /// Calibrated standard deviation.
    pub fn sigma(&self) -> u32 {
        self.sigma
    }

    /// Band floors derived from the calibrated `(μ, σ)`.
    pub fn sigma_floors(&self) -> [u32; 4] {
        self.sigma_floors
    }

    /// Cascade floors derived from the calibrated `(μ, σ)`.
    pub fn sigma_cascade(&self) -> [u32; 8] {
        self.sigma_cascade
    }

    /// Band floors from the reservoir's empirical quantiles.
    pub fn empirical_floors(&self) -> [u32; 4] {
        self.empirical_floors
    }

    /// Cascade floors from the reservoir's empirical quantiles.
    pub fn empirical_cascade(&self) -> [u32; 8] {
        self.empirical_cascade
    }

    /// The band floors in effect: empirical when the shape read as non-normal
    /// at the last checkpoint, sigma otherwise.
    pub fn active_floors(&self) -> [u32; 4] {
        if self.use_empirical {
            self.empirical_floors
        } else {
            self.sigma_floors
        }
    }

    /// The cascade floors in effect, chosen like [`active_floors`](Self::active_floors).
    pub fn active_cascade(&self) -> [u32; 8] {
        if self.use_empirical {
            self.empirical_cascade
        } else {
            self.sigma_cascade
        }
    }

    /// Whether the empirical floors are in effect.
    pub fn is_empirical(&self) -> bool {
        self.use_empirical
    }

    /// Skewness at the last shape evaluation (`0` before any).
    pub fn skewness(&self) -> i32 {
        self.skewness
    }

    /// Kurtosis ×100 at the last shape evaluation (300 before any).
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

    /// The reservoir behind the empirical floors.
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
    // n·M2 = C·n − r², with C centred on the floor mean. Write C = a·n + b;
    // then ⌊(C·n − r²)/n²⌋ = a − [b·n < r²], and b·n, r² < 2^128.
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

    fn same_state(a: &RollingFloor, b: &RollingFloor) {
        assert_eq!(a.moments(), b.moments());
        assert_eq!(a.reservoir(), b.reservoir());
        assert_eq!((a.mu(), a.sigma()), (b.mu(), b.sigma()));
        assert_eq!(a.active_floors(), b.active_floors());
        assert_eq!(a.active_cascade(), b.active_cascade());
        assert_eq!((a.is_empirical(), a.skewness(), a.kurtosis()), (b.is_empirical(), b.skewness(), b.kurtosis()));
    }

    #[test]
    fn isqrt_is_floor_sqrt() {
        for n in (0..200_000u32).chain([u32::MAX, u32::MAX - 1, 65535 * 65535, 65536 * 65535]) {
            let r = isqrt_u32(n);
            assert!(u64::from(r) * u64::from(r) <= u64::from(n));
            assert!(u64::from(r + 1) * u64::from(r + 1) > u64::from(n));
        }
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
        // A floor far from the data: every checkpoint must alert.
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
        // Calibrated σ = 8: shift iff |Δμ| > 4 or |Δσ| > 2.
        for (value, expect) in [(104u32, false), (105, true)] {
            let mut f = RollingFloor::from_params(100, 8);
            // Alternate value ± 8 so the running σ is exactly 8.
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

    #[test]
    fn recalibration_resets_the_running_state() {
        let mut f = RollingFloor::calibrate(&normalish(2000, 5000, 50, 9));
        let shift = f
            .observe_batch(&normalish(3000, 5400, 50, 10))
            .1
            .expect("shift");
        f.recalibrate(&shift);
        assert_eq!((f.mu(), f.sigma()), (shift.new_mu, shift.new_sigma));
        assert_eq!(f.observations(), 0);
        assert!(f.reservoir().is_empty());
        assert_eq!(f.reservoir().capacity(), RollingFloor::RESERVOIR_CAP);
        assert!(!f.is_empirical());
        assert_eq!((f.skewness(), f.kurtosis()), (0, 300));
        assert_eq!(f.active_floors(), RollingFloor::sigma_floors_of(f.mu(), f.sigma()));
        assert_eq!(f.empirical_cascade(), f.sigma_cascade());
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
        // Replacement actually happens: the held set is not the first 1000.
        assert_ne!(a.samples(), &xs[..1000]);
    }

    #[test]
    fn quantile_rule_is_floor_index_clamped() {
        let s: Vec<u32> = (0..1000).collect();
        assert_eq!(quantile_of_sorted(&s, 0.0013), 1);
        assert_eq!(quantile_of_sorted(&s, 0.159), 159);
        assert_eq!(quantile_of_sorted(&s, 0.5), 500);
        assert_eq!(quantile_of_sorted(&s, 1.0), 999);
        assert_eq!(quantile_of_sorted(&[], 0.5), 0);
    }

    #[test]
    fn normal_stream_stays_sigma_and_bimodal_goes_empirical() {
        let mut f = RollingFloor::calibrate(&normalish(1000, 8192, 64, 1));
        f.observe_batch(&normalish(1000, 8192, 64, 2));
        assert!(f.shape_is_normal(), "skew {} kurt {}", f.skewness(), f.kurtosis());
        assert!(!f.is_empirical());

        // Symmetric two-mode mixture: kurtosis ≈ 100, far below the window.
        let (lo, hi) = (normalish(1000, 7800, 20, 3), normalish(1000, 8600, 20, 4));
        let bimodal: Vec<u32> = lo.iter().zip(&hi).flat_map(|(&a, &b)| [a, b]).collect();
        let mut g = RollingFloor::calibrate(&bimodal[..1000]);
        g.observe_batch(&bimodal[1000..]);
        assert!(g.is_empirical(), "skew {} kurt {}", g.skewness(), g.kurtosis());
        let sorted = g.reservoir().sorted();
        assert_eq!(g.active_floors()[2], quantile_of_sorted(&sorted, 0.159));
    }

    /// Parameter drift and shape are separate readings. A pure location and
    /// spread change of a normal stream raises a parameter shift. At that
    /// checkpoint the reservoir still mixes the old and the new population,
    /// so the shape reads non-normal: this is the reference behaviour, kept
    /// as is. After recalibration the shape layer restarts and the same
    /// shifted stream reads normal again.
    #[test]
    fn parameter_drift_then_recalibration_restores_the_normal_shape() {
        let mut f = RollingFloor::calibrate(&normalish(1000, 5000, 40, 7));
        let shifted = normalish(5000, 5600, 90, 8);
        let (used, shift) = f.observe_batch(&shifted);
        let shift = shift.expect("parameters moved");
        assert!(!f.shape_is_normal(), "mixed reservoir at the drift checkpoint");
        assert_eq!(shift.observations, 2000, "first checkpoint still mixes both");
        f.recalibrate(&shift);
        // The first shift adopts the mixture; the next one settles on the
        // shifted stream's own parameters.
        let (f, later) = run_scalar(f, &shifted[used..]);
        assert_eq!(later.len(), 1, "{later:?}");
        assert!(f.mu().abs_diff(5600) <= 5 && f.sigma().abs_diff(90) <= 5, "{} {}", f.mu(), f.sigma());
        assert!(f.shape_is_normal(), "skew {} kurt {}", f.skewness(), f.kurtosis());
        assert!(!f.is_empirical());
    }

    /// Usable from the first observation, and refined by more of them without
    /// any reset.
    #[test]
    fn anytime_use_refines_with_population() {
        let mut f = RollingFloor::for_width(16384);
        assert_eq!(f.active_floors(), [8000, 8064, 8128, 8192]);
        let xs = normalish(50_000, 8192, 64, 12);
        let mut errs = Vec::new();
        for (i, &d) in xs.iter().enumerate() {
            assert!(f.observe(d).is_none(), "on-prior data must not drift");
            if [100, 1000, 50_000].contains(&(i + 1)) {
                let m = f.moments();
                errs.push((m.variance().sqrt() - 64.0).abs());
            }
        }
        assert_eq!(f.observations(), 50_000);
        assert!(errs[2] < errs[0], "{errs:?}");
    }

    /// Large-population running spread is still meaningful: the checkpoint
    /// variance comes from exact moments past the u128 product range.
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
        assert_eq!(variance_floor(&m), 1); // values ±1 around the mean
        let m = MomentsU32 {
            n: 4,
            sum: 10,
            sum_sq: 30,
        }; // 1,2,3,4: var 1.25
        assert_eq!(variance_floor(&m), 1);
    }

    #[test]
    fn calibrate_uses_spread_around_the_integer_mean() {
        // 0,0,0,1: mean 0.25, floor mean 0, Σ(x−0)² = 1, 1/4 = 0 -> σ 1 (floored).
        let f = RollingFloor::calibrate(&[0, 0, 0, 1]);
        assert_eq!((f.mu(), f.sigma()), (0, 1));
        let f = RollingFloor::calibrate(&[100, 120, 100, 120]);
        assert_eq!((f.mu(), f.sigma()), (110, 10));
        assert_eq!(f.observations(), 4);
    }

    // ── The lance-graph reference, kept verbatim as an oracle ───────────
    //
    // Old `hdr.rs` arithmetic: integer Welford with truncated means. Used
    // only to measure where exact moments change a floor decision.
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
