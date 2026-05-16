//! QualiaStream — forward-iterator over a borrowed `&[QualiaI4Row]` slice.
//! Per cognitive-substrate-convergence-v1.md §5 L-20: vertical streaming
//! structs at the ndarray hardware-acceleration layer for sweep over the
//! QualiaColumn SoA layout introduced by D-CSV-5b.
//!
//! Yields `(row_index, &QualiaI4Row)` tuples. Pure iterator scaffold; the
//! `par_qualia_stream` rayon-parallel variant is sprint-13+ once rayon is
//! wired into the ndarray feature gate.

// NOTE: do NOT import lance-graph-contract here (would create circular dep
// since contract is *consumer* of ndarray). Define a minimal local mirror
// of the QualiaI4_16D shape — 8 bytes, `Copy`, hashable. Real coupling at
// the consumer boundary, not at the producer.

/// Local mirror of `lance_graph_contract::qualia::QualiaI4_16D`.
/// Bit-compatible: `repr(C, align(8))`, 8 bytes, 16 × 4-bit signed lanes.
/// Defined here to avoid a circular dependency with the contract crate.
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct QualiaI4Row(pub u64);

/// Forward-iterator over a borrowed `&[QualiaI4Row]` slice.
///
/// Yields `(row_index, &QualiaI4Row)` tuples in ascending index order.
///
/// # Example
///
/// ```
/// use ndarray::hpc::stream::qualia::{QualiaI4Row, QualiaStream};
///
/// let rows = vec![QualiaI4Row(1), QualiaI4Row(2), QualiaI4Row(3)];
/// let mut stream = QualiaStream::new(&rows);
/// assert_eq!(stream.next(), Some((0, &QualiaI4Row(1))));
/// assert_eq!(stream.next(), Some((1, &QualiaI4Row(2))));
/// assert_eq!(stream.next(), Some((2, &QualiaI4Row(3))));
/// assert_eq!(stream.next(), None);
/// ```
pub struct QualiaStream<'a> {
    rows: &'a [QualiaI4Row],
    cursor: usize,
}

impl<'a> QualiaStream<'a> {
    /// Construct a new `QualiaStream` over `rows`.
    /// The cursor starts at index 0.
    #[inline]
    pub fn new(rows: &'a [QualiaI4Row]) -> Self {
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

    /// Current cursor position (0-based index of the NEXT row to be yielded).
    #[inline]
    pub fn cursor(&self) -> usize {
        self.cursor
    }
}

impl<'a> Iterator for QualiaStream<'a> {
    type Item = (usize, &'a QualiaI4Row);

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

impl<'a> ExactSizeIterator for QualiaStream<'a> {
    /// Returns the number of rows not yet yielded.
    #[inline]
    fn len(&self) -> usize {
        self.remaining()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{QualiaI4Row, QualiaStream};

    /// Empty slice → stream yields nothing immediately.
    #[test]
    fn test_empty_stream() {
        let rows: Vec<QualiaI4Row> = vec![];
        let mut stream = QualiaStream::new(&rows);
        assert!(stream.is_empty());
        assert_eq!(stream.len(), 0);
        assert_eq!(stream.remaining(), 0);
        assert_eq!(stream.next(), None);
    }

    /// Stream over N rows must yield exactly N items.
    #[test]
    fn test_stream_yields_all_rows() {
        let rows = vec![QualiaI4Row(0xAA), QualiaI4Row(0xBB), QualiaI4Row(0xCC), QualiaI4Row(0xDD)];
        let stream = QualiaStream::new(&rows);
        let collected: Vec<(usize, &QualiaI4Row)> = stream.collect();
        assert_eq!(collected.len(), rows.len());
        for (i, row) in collected.iter() {
            assert_eq!(row.0, rows[*i].0);
        }
    }

    /// Each yielded index must equal the row's position in the slice.
    #[test]
    fn test_stream_indices_match() {
        let rows: Vec<QualiaI4Row> = (0u64..8).map(QualiaI4Row).collect();
        let mut stream = QualiaStream::new(&rows);
        let mut expected_idx = 0usize;
        while let Some((idx, row)) = stream.next() {
            assert_eq!(idx, expected_idx, "index mismatch at position {}", expected_idx);
            assert_eq!(row.0, expected_idx as u64, "row value mismatch at index {}", expected_idx);
            expected_idx += 1;
        }
        assert_eq!(expected_idx, rows.len());
    }

    /// `remaining()` must decrement by 1 with each `next()` call.
    #[test]
    fn test_remaining_decrements() {
        let rows: Vec<QualiaI4Row> = (0u64..5).map(QualiaI4Row).collect();
        let mut stream = QualiaStream::new(&rows);
        assert_eq!(stream.remaining(), 5);
        let _ = stream.next();
        assert_eq!(stream.remaining(), 4);
        let _ = stream.next();
        assert_eq!(stream.remaining(), 3);
        // Exhaust
        while stream.next().is_some() {}
        assert_eq!(stream.remaining(), 0);
    }

    /// After `reset()`, the stream must replay all rows from index 0.
    #[test]
    fn test_reset_restarts() {
        let rows = vec![QualiaI4Row(10), QualiaI4Row(20), QualiaI4Row(30)];
        let mut stream = QualiaStream::new(&rows);
        // Consume all
        while stream.next().is_some() {}
        assert_eq!(stream.remaining(), 0);
        // Reset and re-collect
        stream.reset();
        assert_eq!(stream.remaining(), 3);
        let first = stream.next();
        assert_eq!(first, Some((0, &QualiaI4Row(10))));
    }

    /// `ExactSizeIterator::len()` must equal `remaining()` at every step.
    #[test]
    fn test_exact_size_iterator() {
        let rows: Vec<QualiaI4Row> = (0u64..6).map(QualiaI4Row).collect();
        let mut stream = QualiaStream::new(&rows);
        // Before iteration
        assert_eq!(ExactSizeIterator::len(&stream), 6);
        assert_eq!(ExactSizeIterator::len(&stream), stream.remaining());
        // After each next()
        for expected_remaining in (0..6usize).rev() {
            let _ = stream.next();
            assert_eq!(ExactSizeIterator::len(&stream), expected_remaining);
            assert_eq!(ExactSizeIterator::len(&stream), stream.remaining());
        }
        assert_eq!(stream.next(), None);
        assert_eq!(ExactSizeIterator::len(&stream), 0);
    }
}
