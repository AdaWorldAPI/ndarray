//! SplatFieldStream — forward-iterator over Gaussian-splat field samples.
//! Per cognitive-substrate-convergence-v1.md §5 L-20 + .claude/knowledge/
//! splat-shader-rayon-struct-method-vision.md: vertical streaming over
//! the splat field for the D-CSV-12 splat op fleet.
//!
//! Each row = one Gaussian splat (mean, σ², energy). Pure iterator
//! scaffold; `par_splat_stream` rayon variant is sprint-13+.

// NOTE: SplatField is defined locally here — do NOT import lance-graph-contract
// (would create a circular dep; ndarray is a producer, contract is a consumer).

/// One Gaussian splat row: mean position, variance (σ²), accumulated energy,
/// and a generation/cycle stamp.
///
/// Layout: `repr(C, align(16))` — 4 × 4-byte fields = exactly 16 bytes.
/// `align(16)` matches the SSE/NEON minimum and is verified by
/// `test_splat_field_size_16b`.
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
/// use crate::hpc::stream::splat_field::{SplatField, SplatFieldStream};
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
