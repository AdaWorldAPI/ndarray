//! Reliability & validity statistics — Pearson r, Spearman ρ, Cronbach α, ICC.
//!
//! The measurement layer for *validate/invalidate* work: given a ground-truth
//! signal and a reconstructed/quantized estimate of it, quantify how faithfully
//! the estimate preserves the original. The four metrics answer complementary
//! questions and disagreeing is informative:
//!
//! * **Pearson r** — linear association (scale/offset invariant).
//! * **Spearman ρ** — *rank* preservation (the metric that matters for ANN /
//!   nearest-neighbour order; invariant to any monotone transform).
//! * **ICC(2,1)** — *absolute agreement* (two-way random, single rater): unlike
//!   Pearson it PENALISES a systematic offset, so `r == 1` with `ICC < 1` is the
//!   signature of a biased-but-correlated reconstruction.
//! * **Cronbach α** — internal consistency treating truth & estimate as two
//!   "items" measuring one construct (a reliability coefficient in [−∞, 1]).
//!
//! These are general primitives (`&[f64]` in, `f64` out) — they have no
//! dependency on the edge-codec work that motivated them and are reused by the
//! `jc` proof-pillars and any consumer needing fidelity numbers.
//!
//! All estimators use the *population* variance convention (divide by `n`); for
//! the ratio-based coefficients (α, ICC) the divisor cancels, and for Pearson /
//! Spearman it cancels in numerator vs denominator, so results match the usual
//! textbook definitions.

/// Pearson product-moment correlation between two equal-length samples.
///
/// Returns `0.0` for degenerate input (`n < 2` or zero variance in either
/// sample) rather than `NaN`, so it is safe to aggregate.
///
/// # Examples
/// ```
/// use ndarray::hpc::reliability::pearson;
/// let x = [1.0, 2.0, 3.0, 4.0];
/// // y = 2x + 1 ⇒ perfect positive linear association.
/// let y = [3.0, 5.0, 7.0, 9.0];
/// assert!((pearson(&x, &y) - 1.0).abs() < 1e-12);
/// assert!((pearson(&x, &[4.0, 3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
/// ```
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 || b.len() != n {
        return 0.0;
    }
    let inv = 1.0 / n as f64;
    let mean_a = a.iter().sum::<f64>() * inv;
    let mean_b = b.iter().sum::<f64>() * inv;
    let (mut cov, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va < 1e-12 || vb < 1e-12 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// Average-rank vector (ties share the mean of the ranks they span).
fn average_ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        v[i].partial_cmp(&v[j])
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        // Extend over the tie group of equal values.
        while j < n && v[idx[j]] == v[idx[i]] {
            j += 1;
        }
        // Mean of ranks [i, j) using 1-based ranks → ((i+1)+j)/2.
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

/// Spearman rank correlation: Pearson on the average-rank vectors (tie-aware).
///
/// # Examples
/// ```
/// use ndarray::hpc::reliability::spearman;
/// // Monotone but non-linear ⇒ Spearman 1.0 (Pearson would be < 1).
/// let x = [1.0, 2.0, 3.0, 4.0];
/// let y = [1.0, 4.0, 9.0, 16.0];
/// assert!((spearman(&x, &y) - 1.0).abs() < 1e-12);
/// ```
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 || b.len() != n {
        return 0.0;
    }
    pearson(&average_ranks(a), &average_ranks(b))
}

/// Cronbach's α over `k` items × `n` subjects (each inner slice is one item's
/// scores across the shared subjects; all slices must have the same length).
///
/// α = (k / (k−1)) · (1 − Σ Var(itemᵢ) / Var(Σ itemᵢ)). Returns `0.0` for
/// `k < 2`, mismatched lengths, `n < 2`, or zero total variance.
///
/// # Examples
/// ```
/// use ndarray::hpc::reliability::cronbach_alpha;
/// // Two identical items ⇒ perfectly consistent ⇒ α = 1.
/// let item = [1.0, 2.0, 3.0, 4.0, 5.0];
/// assert!((cronbach_alpha(&[&item, &item]) - 1.0).abs() < 1e-12);
/// ```
pub fn cronbach_alpha(items: &[&[f64]]) -> f64 {
    let k = items.len();
    if k < 2 {
        return 0.0;
    }
    let n = items[0].len();
    if n < 2 || items.iter().any(|it| it.len() != n) {
        return 0.0;
    }
    // Population variance of a slice.
    let var = |xs: &[f64]| -> f64 {
        let m = xs.iter().sum::<f64>() / n as f64;
        xs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / n as f64
    };
    let sum_item_var: f64 = items.iter().map(|it| var(it)).sum();
    let totals: Vec<f64> = (0..n).map(|j| items.iter().map(|it| it[j]).sum()).collect();
    let total_var = var(&totals);
    if total_var < 1e-12 {
        return 0.0;
    }
    (k as f64 / (k as f64 - 1.0)) * (1.0 - sum_item_var / total_var)
}

/// Intraclass correlation ICC(2,1) — two-way random effects, single rater,
/// **absolute agreement** (Shrout & Fleiss 1979; McGraw & Wong 1996).
///
/// `ratings` is `k` raters × `n` subjects (each inner slice is one rater's
/// scores across the shared subjects). For the common "ground-truth vs
/// reconstruction" case pass two slices. Unlike Pearson, this penalises a
/// constant offset between raters, so it detects systematic bias.
///
/// Returns `0.0` for degenerate input (`k < 2`, `n < 2`, mismatched lengths, or
/// a zero denominator).
///
/// # Examples
/// ```
/// use ndarray::hpc::reliability::icc_a1;
/// let truth = [1.0, 2.0, 3.0, 4.0, 5.0];
/// // Perfect agreement ⇒ ICC = 1.
/// assert!((icc_a1(&[&truth, &truth]) - 1.0).abs() < 1e-9);
/// // A constant +10 offset is perfectly *correlated* but not in *agreement*:
/// let biased: Vec<f64> = truth.iter().map(|x| x + 10.0).collect();
/// let icc = icc_a1(&[&truth, &biased]);
/// assert!(icc < 0.5, "absolute-agreement ICC must penalise the offset, got {icc}");
/// ```
pub fn icc_a1(ratings: &[&[f64]]) -> f64 {
    let k = ratings.len();
    if k < 2 {
        return 0.0;
    }
    let n = ratings[0].len();
    if n < 2 || ratings.iter().any(|r| r.len() != n) {
        return 0.0;
    }
    let (kf, nf) = (k as f64, n as f64);
    let total: f64 = ratings.iter().map(|r| r.iter().sum::<f64>()).sum();
    let grand = total / (kf * nf);

    // Between-subjects (rows) sum of squares: k · Σ_s (subject_mean − grand)².
    let mut ss_subj = 0.0;
    for s in 0..n {
        let sm = ratings.iter().map(|r| r[s]).sum::<f64>() / kf;
        ss_subj += (sm - grand) * (sm - grand);
    }
    ss_subj *= kf;

    // Between-raters (columns) sum of squares: n · Σ_r (rater_mean − grand)².
    let mut ss_rater = 0.0;
    for r in ratings {
        let rm = r.iter().sum::<f64>() / nf;
        ss_rater += (rm - grand) * (rm - grand);
    }
    ss_rater *= nf;

    // Total SS, then residual by subtraction.
    let mut ss_total = 0.0;
    for r in ratings {
        for &x in r.iter() {
            ss_total += (x - grand) * (x - grand);
        }
    }
    let ss_err = ss_total - ss_subj - ss_rater;

    let msr = ss_subj / (nf - 1.0); // mean square, subjects
    let msc = ss_rater / (kf - 1.0); // mean square, raters
    let mse = ss_err / ((nf - 1.0) * (kf - 1.0)); // mean square, residual

    let denom = msr + (kf - 1.0) * mse + (kf / nf) * (msc - mse);
    if denom.abs() < 1e-12 {
        return 0.0;
    }
    (msr - mse) / denom
}

/// Bundled fidelity of a reconstruction against ground truth.
///
/// One call computes all four coefficients plus the relative-L2 error and
/// cosine similarity, so a harness can print a row per codec/flavor without
/// recomputing means four times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FidelityReport {
    /// Pearson product-moment correlation (linear association).
    pub pearson: f64,
    /// Spearman rank correlation (order preservation).
    pub spearman: f64,
    /// ICC(2,1) absolute agreement (bias-sensitive).
    pub icc: f64,
    /// Cronbach α treating {truth, estimate} as two items.
    pub cronbach: f64,
    /// Relative L2 error ‖estimate − truth‖ / ‖truth‖.
    pub rel_l2: f64,
    /// Cosine similarity between truth and estimate.
    pub cosine: f64,
}

impl FidelityReport {
    /// Compute every coefficient for `(truth, estimate)`.
    ///
    /// Mismatched lengths are a caller error and yield a **degenerate** report
    /// (all coefficients `0.0`, `rel_l2 = 1.0`) rather than silently truncating to
    /// the shorter prefix — consistent with the module's return-degenerate (never
    /// panic, never hide) convention, so a dropped tail cannot masquerade as high
    /// fidelity.
    ///
    /// # Examples
    /// ```
    /// use ndarray::hpc::reliability::FidelityReport;
    /// let truth = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    /// let est = [0.1, 0.9, 2.1, 2.9, 4.2, 4.8];
    /// let r = FidelityReport::compute(&truth, &est);
    /// assert!(r.pearson > 0.99 && r.spearman > 0.99);
    /// assert!(r.rel_l2 < 0.1);
    /// // Mismatched lengths → degenerate (does NOT truncate-then-score):
    /// let bad = FidelityReport::compute(&[1.0, 2.0, 100.0], &[1.0, 2.0]);
    /// assert_eq!(bad.pearson, 0.0);
    /// assert_eq!(bad.rel_l2, 1.0);
    /// ```
    pub fn compute(truth: &[f64], estimate: &[f64]) -> Self {
        // Length mismatch is a caller bug: return a degenerate (all-zero,
        // max-error) report rather than truncating to the shorter prefix, which
        // could hide a dropped tail behind a falsely-high score.
        if truth.len() != estimate.len() {
            return FidelityReport {
                pearson: 0.0,
                spearman: 0.0,
                icc: 0.0,
                cronbach: 0.0,
                rel_l2: 1.0,
                cosine: 0.0,
            };
        }
        let n = truth.len();
        let (t, e) = (truth, estimate);
        let mut se = 0.0;
        let mut st = 0.0;
        let mut dot = 0.0;
        let mut ne = 0.0;
        for i in 0..n {
            let d = e[i] - t[i];
            se += d * d;
            st += t[i] * t[i];
            dot += t[i] * e[i];
            ne += e[i] * e[i];
        }
        let rel_l2 = if st > 1e-24 { se.sqrt() / st.sqrt() } else { 0.0 };
        let cosine = if st > 1e-24 && ne > 1e-24 {
            dot / (st.sqrt() * ne.sqrt())
        } else {
            0.0
        };
        FidelityReport {
            pearson: pearson(t, e),
            spearman: spearman(t, e),
            icc: icc_a1(&[t, e]),
            cronbach: cronbach_alpha(&[t, e]),
            rel_l2,
            cosine,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_perfect_and_anti() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v - 2.0).collect();
        assert!((pearson(&x, &y) - 1.0).abs() < 1e-12);
        let z: Vec<f64> = x.iter().rev().copied().collect();
        assert!((pearson(&x, &z) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_degenerate_is_zero_not_nan() {
        assert_eq!(pearson(&[1.0], &[2.0]), 0.0);
        assert_eq!(pearson(&[1.0, 1.0, 1.0], &[2.0, 3.0, 4.0]), 0.0);
        assert_eq!(pearson(&[1.0, 2.0], &[3.0, 4.0, 5.0]), 0.0);
    }

    #[test]
    fn spearman_monotone_nonlinear_is_one() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 8.0, 27.0, 64.0, 125.0]; // x³, monotone
        assert!((spearman(&x, &y) - 1.0).abs() < 1e-12);
        // Pearson is strictly below 1 for the same data.
        assert!(pearson(&x, &y) < 0.99);
    }

    #[test]
    fn spearman_handles_ties() {
        // Tie group in `a`; average-rank must keep ρ well-defined (no NaN).
        let a = [1.0, 1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        let rho = spearman(&a, &b);
        assert!(rho.is_finite() && rho > 0.7);
    }

    #[test]
    fn cronbach_identical_items_is_one() {
        let item = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!((cronbach_alpha(&[&item, &item]) - 1.0).abs() < 1e-12);
        assert!((cronbach_alpha(&[&item, &item, &item]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cronbach_uncorrelated_items_is_low() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [6.0, 1.0, 5.0, 2.0, 4.0, 3.0];
        // Poorly-consistent items → α well below the 0.7 conventional floor.
        assert!(cronbach_alpha(&[&a, &b]) < 0.7);
    }

    #[test]
    fn icc_perfect_agreement_is_one() {
        let t = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        assert!((icc_a1(&[&t, &t]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn icc_penalises_offset_pearson_does_not() {
        let t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let biased: Vec<f64> = t.iter().map(|x| x + 5.0).collect();
        // Correlated perfectly...
        assert!((pearson(&t, &biased) - 1.0).abs() < 1e-12);
        // ...but absolute-agreement ICC is dragged down by the systematic bias.
        let icc = icc_a1(&[&t, &biased]);
        assert!(icc < 0.7, "icc should penalise offset, got {icc}");
    }

    #[test]
    fn icc_scaled_agreement_below_perfect() {
        let t = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let scaled: Vec<f64> = t.iter().map(|x| x * 2.0).collect();
        let icc = icc_a1(&[&t, &scaled]);
        assert!(icc.is_finite() && icc < 1.0);
    }

    #[test]
    fn fidelity_report_clean_reconstruction() {
        let truth: Vec<f64> = (0..32).map(|i| (i as f64).sin()).collect();
        let est: Vec<f64> = truth.iter().map(|x| x + 0.001).collect();
        let r = FidelityReport::compute(&truth, &est);
        assert!(r.pearson > 0.999);
        assert!(r.spearman > 0.999);
        assert!(r.icc > 0.99);
        assert!(r.cosine > 0.999);
        assert!(r.rel_l2 < 0.05);
    }

    #[test]
    fn fidelity_report_degrades_with_noise() {
        let truth: Vec<f64> = (0..64).map(|i| ((i * 7 % 13) as f64) - 6.0).collect();
        let clean: Vec<f64> = truth.iter().map(|x| x + 0.01).collect();
        let noisy: Vec<f64> = truth
            .iter()
            .enumerate()
            .map(|(i, x)| x + ((i % 5) as f64 - 2.0))
            .collect();
        let rc = FidelityReport::compute(&truth, &clean);
        let rn = FidelityReport::compute(&truth, &noisy);
        // Every coefficient should rank the clean reconstruction above the noisy.
        assert!(rc.pearson > rn.pearson);
        assert!(rc.icc > rn.icc);
        assert!(rc.rel_l2 < rn.rel_l2);
    }
}
