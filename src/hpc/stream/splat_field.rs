//! SplatFieldStream — forward-iterator over Gaussian-splat field samples.
//! Per cognitive-substrate-convergence-v1.md §5 L-20 + .claude/knowledge/
//! splat-shader-rayon-struct-method-vision.md: vertical streaming over
//! the splat field for the D-CSV-12 splat op fleet.
//!
//! Each row = one Gaussian splat (mean, σ², energy). Pure iterator
//! scaffold; `par_splat_field_stream` rayon variant wired in sprint-13
//! (D-CSV-17) behind `#[cfg(feature = "rayon")]`.

// NOTE: SplatField is defined locally here — do NOT import lance-graph-contract
// (would create a circular dep; ndarray is a producer, contract is a consumer).

/// One Gaussian splat row: mean position, variance (σ²), accumulated energy,
/// and a generation/cycle stamp.
///
/// Layout: `repr(C, align(16))` — 4 × 4-byte fields = exactly 16 bytes.
/// `align(16)` matches the SSE/NEON minimum and is verified by
/// `test_splat_field_size_16b`. Four rows fit one 64-byte cache line.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct SplatField {
    /// Mean position in the field space (could be index, palette ID, or BindSpace row).
    pub mean: u32,
    /// σ² (variance) — controls splat spread.
    pub variance: f32,
    /// Accumulated energy at this splat.
    pub energy: f32,
    /// Generation/cycle stamp for the splat.
    pub generation: u32,
}

/// Forward-iterator over a borrowed `&[SplatField]` slice.
///
/// Yields `(row_index, &SplatField)` tuples in ascending index order.
///
/// # Example
///
/// ```
/// use ndarray::hpc::stream::splat_field::{SplatField, SplatFieldStream};
///
/// let rows = vec![
///     SplatField { mean: 0, variance: 1.0, energy: 0.5, generation: 1 },
///     SplatField { mean: 1, variance: 2.0, energy: 1.5, generation: 2 },
/// ];
/// let mut stream = SplatFieldStream::new(&rows);
/// let (idx, splat) = stream.next().unwrap();
/// assert_eq!(idx, 0);
/// assert_eq!(splat.mean, 0);
/// ```
pub struct SplatFieldStream<'a> {
    rows: &'a [SplatField],
    cursor: usize,
}

impl<'a> SplatFieldStream<'a> {
    /// Construct a new `SplatFieldStream` over `rows`.
    /// The cursor starts at index 0.
    #[inline]
    pub fn new(rows: &'a [SplatField]) -> Self {
        Self { rows, cursor: 0 }
    }

    /// Total number of rows in the backing slice (unchanged by iteration).
    #[inline]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// `true` if the backing slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Number of rows not yet yielded (decrements with each `next()` call).
    #[inline]
    pub fn remaining(&self) -> usize {
        self.rows.len().saturating_sub(self.cursor)
    }

    /// Reset the cursor to 0, allowing the stream to be re-iterated from the start.
    #[inline]
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Filter to only splats whose `energy` field is strictly above `threshold`.
    ///
    /// Consumes `self` (the `SplatFieldStream` is itself an `Iterator`) and
    /// returns a lazy `impl Iterator` — no allocation.
    pub fn filter_energy_above(self, threshold: f32) -> impl Iterator<Item = (usize, &'a SplatField)> {
        self.filter(move |(_, s)| s.energy > threshold)
    }
}

impl<'a> Iterator for SplatFieldStream<'a> {
    type Item = (usize, &'a SplatField);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.rows.len() {
            let i = self.cursor;
            self.cursor += 1;
            Some((i, &self.rows[i]))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.remaining();
        (rem, Some(rem))
    }
}

impl<'a> ExactSizeIterator for SplatFieldStream<'a> {
    /// Returns the number of rows not yet yielded.
    #[inline]
    fn len(&self) -> usize {
        self.remaining()
    }
}

// ─── Rayon-parallel variant (D-CSV-17, sprint-13) ─────────────────────────────

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Rayon-parallel forward-iterator over `&[SplatField]`.
///
/// Yields `(row_index, &SplatField)` tuples. Returns `impl IndexedParallelIterator`
/// so callers retain `enumerate()`, `zip()`, and order-preserving `collect()`.
///
/// `SplatField` is 16 bytes (`repr(C, align(16))`), so 4 rows fit one 64-byte
/// cache line. Callers folding into ordered structures should call
/// `.with_min_len(4)` for cache-line-aligned chunking (OQ-CSV-8 fixed chunk for
/// SplatField).
///
/// Used by the D-CSV-12 splat op fleet for parallel evaluation of
/// `splat_gaussian`, `score_hole_closure`, `replay_coherence`, and
/// `emit_if_epiphany` across the entire splat field.
///
/// # Determinism
///
/// See §6 of pr-sprint-13-rayon-streams.md. Order is NOT guaranteed ascending;
/// use `.with_min_len(rows.len())` to coerce serial execution for bit-exact
/// reproducibility when the downstream accumulator is non-commutative.
#[cfg(feature = "rayon")]
#[inline]
pub fn par_splat_field_stream(rows: &[SplatField]) -> impl IndexedParallelIterator<Item = (usize, &SplatField)> {
    rows.par_iter().enumerate()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{SplatField, SplatFieldStream};
    use std::mem;

    fn make_splat(mean: u32, variance: f32, energy: f32, generation: u32) -> SplatField {
        SplatField {
            mean,
            variance,
            energy,
            generation,
        }
    }

    /// Empty slice → stream yields nothing immediately.
    #[test]
    fn test_splat_stream_empty() {
        let rows: Vec<SplatField> = vec![];
        let mut stream = SplatFieldStream::new(&rows);
        assert!(stream.is_empty());
        assert_eq!(stream.len(), 0);
        assert_eq!(stream.remaining(), 0);
        assert_eq!(stream.next(), None);
    }

    /// Stream over N rows must yield exactly N items with matching indices.
    #[test]
    fn test_splat_stream_yields_all() {
        let rows = vec![make_splat(0, 1.0, 0.1, 1), make_splat(1, 2.0, 0.5, 2), make_splat(2, 0.5, 2.0, 3)];
        let stream = SplatFieldStream::new(&rows);
        let collected: Vec<(usize, &SplatField)> = stream.collect();
        assert_eq!(collected.len(), 3);
        for (idx, splat) in &collected {
            assert_eq!(splat.mean, *idx as u32);
        }
    }

    /// `filter_energy_above` must retain only splats strictly above the threshold.
    #[test]
    fn test_filter_energy_above() {
        let rows = vec![
            make_splat(0, 1.0, 0.1, 1),
            make_splat(1, 1.0, 0.5, 2),
            make_splat(2, 1.0, 1.0, 3),
            make_splat(3, 1.0, 2.0, 4),
        ];
        let stream = SplatFieldStream::new(&rows);
        let above: Vec<(usize, &SplatField)> = stream.filter_energy_above(0.5).collect();
        // Only rows with energy > 0.5: indices 2 (1.0) and 3 (2.0).
        assert_eq!(above.len(), 2);
        assert_eq!(above[0].0, 2);
        assert_eq!(above[1].0, 3);
    }

    /// `size_of::<SplatField>()` must be exactly 16 bytes — verifies `align(16)`
    /// and field packing (4 × 4-byte fields with no hidden padding).
    #[test]
    fn test_splat_field_size_16b() {
        assert_eq!(mem::size_of::<SplatField>(), 16, "SplatField must be exactly 16 bytes (4 × 4B fields, align(16))");
        assert_eq!(mem::align_of::<SplatField>(), 16, "SplatField alignment must be 16");
    }

    /// `remaining()` must decrement by 1 with each `next()` call.
    #[test]
    fn test_remaining_decrements() {
        let rows = vec![
            make_splat(0, 1.0, 1.0, 0),
            make_splat(1, 1.0, 1.0, 1),
            make_splat(2, 1.0, 1.0, 2),
            make_splat(3, 1.0, 1.0, 3),
        ];
        let mut stream = SplatFieldStream::new(&rows);
        assert_eq!(stream.remaining(), 4);
        let _ = stream.next();
        assert_eq!(stream.remaining(), 3);
        let _ = stream.next();
        assert_eq!(stream.remaining(), 2);
        // Exhaust remaining
        while stream.next().is_some() {}
        assert_eq!(stream.remaining(), 0);
        assert_eq!(stream.next(), None);
    }

    /// After `reset()`, the stream replays all rows from index 0.
    #[test]
    fn test_reset_restarts() {
        let rows = vec![make_splat(10, 1.0, 0.3, 1), make_splat(20, 2.0, 0.6, 2), make_splat(30, 3.0, 0.9, 3)];
        let mut stream = SplatFieldStream::new(&rows);
        // Consume everything
        while stream.next().is_some() {}
        assert_eq!(stream.remaining(), 0);
        // Reset and verify replay
        stream.reset();
        assert_eq!(stream.remaining(), 3);
        let first = stream.next();
        assert!(first.is_some());
        let (idx, splat) = first.unwrap();
        assert_eq!(idx, 0);
        assert_eq!(splat.mean, 10);
    }
}

// ─── Rayon par_* tests (D-CSV-17) ─────────────────────────────────────────────

#[cfg(all(test, feature = "rayon"))]
mod par_tests {
    use super::{par_splat_field_stream, SplatField, SplatFieldStream};
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn make_splat(mean: u32, variance: f32, energy: f32, generation: u32) -> SplatField {
        SplatField {
            mean,
            variance,
            energy,
            generation,
        }
    }

    /// T-P-S-1: par_splat_field_stream yields all N items.
    #[test]
    fn test_par_splat_yields_all() {
        let rows: Vec<SplatField> = (0u32..1024).map(|i| make_splat(i, 1.0, 0.1, 0)).collect();
        let count = par_splat_field_stream(&rows).count();
        assert_eq!(count, 1024);
    }

    /// T-P-S-2: par_splat_field_stream on empty slice yields zero items.
    #[test]
    fn test_par_splat_empty() {
        let rows: Vec<SplatField> = vec![];
        let count = par_splat_field_stream(&rows).count();
        assert_eq!(count, 0);
    }

    /// T-P-S-3: par_iter result equals serial iter result (as sorted sets).
    #[test]
    fn test_par_splat_matches_serial() {
        let rows: Vec<SplatField> = (0u32..256)
            .map(|i| make_splat(i, 1.0, i as f32 / 256.0, 0))
            .collect();
        // Use (index, mean) tuples — mean is unique per row, so sort gives canonical order.
        let mut par: Vec<(usize, u32)> = par_splat_field_stream(&rows)
            .map(|(i, s)| (i, s.mean))
            .collect();
        let mut ser: Vec<(usize, u32)> = SplatFieldStream::new(&rows)
            .map(|(i, s)| (i, s.mean))
            .collect();
        par.sort_unstable();
        ser.sort_unstable();
        assert_eq!(par, ser);
    }

    /// T-P-S-4: filter on energy field — parallel equivalent of
    /// SplatFieldStream::filter_energy_above.
    ///
    /// 256 rows with energy = i/256.0; energies > 0.5 occur for i in 129..256 → 127 items.
    #[test]
    fn test_par_splat_filter_energy() {
        let rows: Vec<SplatField> = (0u32..256)
            .map(|i| make_splat(i, 1.0, (i as f32) / 256.0, 0))
            .collect();
        let above = par_splat_field_stream(&rows)
            .filter(|(_, s)| s.energy > 0.5)
            .count();
        // i=128 → 128/256=0.5 (NOT > 0.5); i=129 → 129/256≈0.504 (> 0.5).
        // So i ∈ 129..256 → 127 items.
        assert_eq!(above, 127);
    }

    /// T-P-S-5: with_min_len(4) — cache-line alignment knob.
    /// 4 rows × 16 B = 64 B = one cache line (OQ-CSV-8 fixed chunk for SplatField).
    #[test]
    fn test_par_splat_min_len() {
        let rows: Vec<SplatField> = (0u32..1024).map(|i| make_splat(i, 1.0, 0.1, 0)).collect();
        let count = par_splat_field_stream(&rows).with_min_len(4).count();
        assert_eq!(count, 1024);
    }

    /// T-P-S-6: thread-safety — SplatField is Send + Sync; verified by
    /// mutating an AtomicUsize from the parallel for_each closure.
    #[test]
    fn test_par_splat_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SplatField>();
        let rows: Vec<SplatField> = (0u32..1024).map(|i| make_splat(i, 1.0, 0.1, 0)).collect();
        let counter = AtomicUsize::new(0);
        par_splat_field_stream(&rows).for_each(|_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 1024);
    }
}
