//! InferenceStream — forward-iterator over a borrowed `&[InferenceRow]` slice.
//! Per cognitive-substrate-convergence-v1.md §5 L-20: vertical streaming
//! over the inference-mantissa lane of the EdgeColumn SoA. Used by the
//! integer-SIMD MUL evaluation hot path (D-CSV-8 sprint-12 SIMD vec).
//!
//! Pure iterator scaffold; `par_inference_stream` rayon variant wired in
//! sprint-13 (D-CSV-17) behind `#[cfg(feature = "rayon")]`.

// Local mirror of CausalEdge64 shape (bit-compatible with causal_edge::CausalEdge64).
// No cross-crate import: ndarray is the producer; causal-edge is the consumer.

/// A single row of the EdgeColumn SoA, bit-compatible with
/// `causal_edge::CausalEdge64` v2 layout.
///
/// Fields of interest for the inference-mantissa lane:
/// - bits 46-49: signed 4-bit inference mantissa (−8..+7)
/// - bits 53-58: W-slot corpus root handle (0..=63)
#[repr(C, align(8))]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct InferenceRow(pub u64);

impl InferenceRow {
    /// Read the 4-bit signed mantissa at bits 46-49 (matches causal-edge v2
    /// `inference_mantissa()` exactly — see `causal-edge/src/layout.rs`).
    ///
    /// Sign-extension: extract 4-bit unsigned value, then sign-extend to i8
    /// via arithmetic left-shift trick: `(raw << 4) >> 4`.
    #[inline]
    pub fn inference_mantissa(&self) -> i8 {
        let raw = ((self.0 >> 46) & 0xF) as i8;
        (raw << 4) >> 4 // sign-extend 4 → 8 bits
    }

    /// Read the W-slot at bits 53-58 (6 bits, 0..=63).
    ///
    /// The W-slot is the witness corpus root handle per CausalEdge64 v2 L-6.
    /// Returns 0 for zero-initialized rows.
    #[inline]
    pub fn w_slot(&self) -> u8 {
        ((self.0 >> 53) & 0x3F) as u8
    }
}

/// Forward-iterator over a borrowed slice of [`InferenceRow`] values.
///
/// Provides vertical streaming access to the inference-mantissa lane of the
/// EdgeColumn SoA. Yields `(index, &InferenceRow)` tuples so callers can
/// correlate back to the originating row without maintaining external counters.
///
/// # Example
/// ```rust
/// use ndarray::hpc::stream::inference::{InferenceRow, InferenceStream};
///
/// let rows = vec![InferenceRow(0), InferenceRow(1 << 46)];
/// let mut stream = InferenceStream::new(&rows);
/// assert_eq!(stream.len(), 2);
/// let (idx, row) = stream.next().unwrap();
/// assert_eq!(idx, 0);
/// ```
pub struct InferenceStream<'a> {
    rows: &'a [InferenceRow],
    cursor: usize,
}

impl<'a> InferenceStream<'a> {
    /// Construct a new stream over the given slice. The cursor starts at 0.
    pub fn new(rows: &'a [InferenceRow]) -> Self {
        Self { rows, cursor: 0 }
    }

    /// Total number of rows in the underlying slice (not remaining).
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns `true` if the underlying slice is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Number of rows not yet yielded by the iterator.
    pub fn remaining(&self) -> usize {
        self.rows.len().saturating_sub(self.cursor)
    }

    /// Reset the cursor to the beginning so the stream can be iterated again.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}

impl<'a> Iterator for InferenceStream<'a> {
    type Item = (usize, &'a InferenceRow);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.rows.len() {
            let i = self.cursor;
            self.cursor += 1;
            Some((i, &self.rows[i]))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.remaining();
        (rem, Some(rem))
    }
}

impl<'a> ExactSizeIterator for InferenceStream<'a> {
    fn len(&self) -> usize {
        self.remaining()
    }
}

// ─── Rayon-parallel variant (D-CSV-17, sprint-13) ─────────────────────────────

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Rayon-parallel forward-iterator over `&[InferenceRow]`.
///
/// Mirrors `par_qualia_stream` semantics. Returns `impl IndexedParallelIterator`
/// so callers retain `enumerate()`, `zip()`, and order-preserving `collect()`.
///
/// For `InferenceRow` (8 B/row, `repr(C, align(8))`), 8 rows fill one 64-byte
/// cache line; callers folding into ordered structures should call
/// `.with_min_len(8)` to align chunks (OQ-CSV-8 fixed chunk for InferenceRow).
///
/// Particularly useful for the integer-SIMD MUL evaluation hot path (D-CSV-8):
/// folding the inference mantissa lane across millions of EdgeColumn rows
/// benefits from work-stealing on multi-core hosts.
///
/// # Determinism
///
/// See §6 of pr-sprint-13-rayon-streams.md. Pattern A (order-insensitive folds
/// like `.sum()`, `.count()`) and Pattern B (`collect()`) are safe.
/// Pattern C (non-commutative accumulators) requires per-callsite analysis.
#[cfg(feature = "rayon")]
#[inline]
pub fn par_inference_stream(rows: &[InferenceRow]) -> impl IndexedParallelIterator<Item = (usize, &InferenceRow)> {
    rows.par_iter().enumerate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_stream_empty() {
        let rows: &[InferenceRow] = &[];
        let mut stream = InferenceStream::new(rows);
        assert!(stream.is_empty());
        assert_eq!(stream.len(), 0);
        assert_eq!(stream.remaining(), 0);
        assert!(stream.next().is_none());
    }

    #[test]
    fn test_inference_stream_yields_all() {
        let rows = vec![InferenceRow(0), InferenceRow(1), InferenceRow(2)];
        let stream = InferenceStream::new(&rows);
        let collected: Vec<_> = stream.collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].0, 0);
        assert_eq!(collected[1].0, 1);
        assert_eq!(collected[2].0, 2);
        assert_eq!(collected[0].1 as *const _, &rows[0] as *const _);
        assert_eq!(collected[2].1 as *const _, &rows[2] as *const _);
    }

    #[test]
    fn test_mantissa_signed_extraction() {
        // Pack bits 46-49 = 0b1111 = 15 (raw), which is -1 in 4-bit two's complement.
        let raw_bits: u64 = 0b1111u64 << 46;
        let row = InferenceRow(raw_bits);
        assert_eq!(row.inference_mantissa(), -1);

        // Pack bits 46-49 = 0b0111 = 7 (raw), positive maximum.
        let row_pos = InferenceRow(0b0111u64 << 46);
        assert_eq!(row_pos.inference_mantissa(), 7);

        // Pack bits 46-49 = 0b1000 = 8 (raw), which is -8 in 4-bit two's complement.
        let row_min = InferenceRow(0b1000u64 << 46);
        assert_eq!(row_min.inference_mantissa(), -8);

        // Zero mantissa.
        let row_zero = InferenceRow(0);
        assert_eq!(row_zero.inference_mantissa(), 0);
    }

    #[test]
    fn test_w_slot_extraction() {
        // Pack bits 53-58 = 0b111111 = 63 (maximum W-slot value).
        let raw_bits: u64 = 0b111111u64 << 53;
        let row = InferenceRow(raw_bits);
        assert_eq!(row.w_slot(), 63);

        // W-slot = 0 (zero row).
        let row_zero = InferenceRow(0);
        assert_eq!(row_zero.w_slot(), 0);

        // W-slot = 1.
        let row_one = InferenceRow(1u64 << 53);
        assert_eq!(row_one.w_slot(), 1);

        // W-slot = 32 (bit 58 set, bit 53 clear).
        let row_32 = InferenceRow(32u64 << 53);
        assert_eq!(row_32.w_slot(), 32);
    }

    #[test]
    fn test_remaining_decrements() {
        let rows = vec![InferenceRow(0); 4];
        let mut stream = InferenceStream::new(&rows);
        assert_eq!(stream.remaining(), 4);
        stream.next();
        assert_eq!(stream.remaining(), 3);
        stream.next();
        assert_eq!(stream.remaining(), 2);
        stream.next();
        assert_eq!(stream.remaining(), 1);
        stream.next();
        assert_eq!(stream.remaining(), 0);
        // Exhausted: remaining stays 0.
        stream.next();
        assert_eq!(stream.remaining(), 0);
    }

    #[test]
    fn test_reset_restarts() {
        let rows = vec![InferenceRow(10), InferenceRow(20)];
        let mut stream = InferenceStream::new(&rows);

        // Exhaust the stream.
        assert!(stream.next().is_some());
        assert!(stream.next().is_some());
        assert!(stream.next().is_none());
        assert_eq!(stream.remaining(), 0);

        // After reset, the stream yields from the beginning again.
        stream.reset();
        assert_eq!(stream.remaining(), 2);
        let first = stream.next().unwrap();
        assert_eq!(first.0, 0);
        assert_eq!(first.1 .0, 10);
    }
}

// ─── Rayon par_* tests (D-CSV-17) ─────────────────────────────────────────────

#[cfg(all(test, feature = "rayon"))]
mod par_tests {
    use super::{par_inference_stream, InferenceRow, InferenceStream};
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// T-P-I-1: par_inference_stream yields all N items.
    #[test]
    fn test_par_inference_yields_all() {
        let rows: Vec<InferenceRow> = (0u64..1024).map(InferenceRow).collect();
        let count = par_inference_stream(&rows).count();
        assert_eq!(count, 1024);
    }

    /// T-P-I-2: par_inference_stream on empty slice yields zero items.
    #[test]
    fn test_par_inference_empty() {
        let rows: Vec<InferenceRow> = vec![];
        let count = par_inference_stream(&rows).count();
        assert_eq!(count, 0);
    }

    /// T-P-I-3: par_iter result equals serial iter result (as sorted sets).
    #[test]
    fn test_par_inference_matches_serial() {
        let rows: Vec<InferenceRow> = (0u64..256).map(InferenceRow).collect();
        let mut par: Vec<u64> = par_inference_stream(&rows)
            .map(|(i, r)| (i as u64) ^ r.0)
            .collect();
        let mut ser: Vec<u64> = InferenceStream::new(&rows)
            .map(|(i, r)| (i as u64) ^ r.0)
            .collect();
        par.sort_unstable();
        ser.sort_unstable();
        assert_eq!(par, ser);
    }

    /// T-P-I-4: filter on inference_mantissa() — sign-extension behaves
    /// identically under parallel access.
    ///
    /// 256 rows with mantissa varying 0..16 cyclically via bits 46-49:
    ///   - raw values 0..7 → mantissa 0..7 (non-negative): 128 rows
    ///   - raw values 8..15 → mantissa -8..-1 (negative): 128 rows
    #[test]
    fn test_par_inference_filter_mantissa() {
        let rows: Vec<InferenceRow> = (0u64..256).map(|i| InferenceRow((i & 0xF) << 46)).collect();
        let neg = par_inference_stream(&rows)
            .filter(|(_, r)| r.inference_mantissa() < 0)
            .count();
        assert_eq!(neg, 128);
    }

    /// T-P-I-5: with_min_len(8) knob compiles and yields all items.
    /// 8 rows × 8 B = 64 B = one cache line (OQ-CSV-8 fixed chunk for InferenceRow).
    #[test]
    fn test_par_inference_min_len() {
        let rows: Vec<InferenceRow> = (0u64..1024).map(InferenceRow).collect();
        let count = par_inference_stream(&rows).with_min_len(8).count();
        assert_eq!(count, 1024);
    }

    /// T-P-I-6: thread-safety — InferenceRow is Send + Sync; verified by
    /// mutating an AtomicUsize from the parallel for_each closure.
    #[test]
    fn test_par_inference_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<InferenceRow>();
        let rows: Vec<InferenceRow> = (0u64..1024).map(InferenceRow).collect();
        let counter = AtomicUsize::new(0);
        par_inference_stream(&rows).for_each(|_| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), 1024);
    }
}
