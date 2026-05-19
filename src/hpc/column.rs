//! Multi-lane typed column view over a shared byte backing store (PR-X1).
//!
//! [`MultiLaneColumn`] wraps one `Arc<[u8]>` backing buffer and provides
//! zero-copy typed lane views at different SIMD widths. Consumers pick the
//! lane width per operation; the backing store is never copied.
//!
//! This module is **layout-only**. No `#[target_feature]`, no per-arch
//! imports, no raw intrinsics. The SIMD register load happens inside the
//! consumer's loop using `crate::simd::F32x16::from_array` etc.
//!
//! # Layering
//!
//! Lives in `hpc::column`; the `crate::simd::*` re-export lands in the PR-X1
//! re-export sweep (see `.claude/knowledge/pr-x1-design.md` § 4). Doctests in
//! this file therefore use the canonical `ndarray::hpc::column` path until
//! the sweep ships.
//!
//! # Distance typing
//!
//! This type is layout-only. No distance-aware API. See
//! `.claude/knowledge/cognitive-distance-typing.md` (no-umbrella rule).
//!
//! # Design reference
//!
//! `.claude/knowledge/pr-x1-design.md` § "1. `MultiLaneColumn`". The
//! `iter_*_bytes` family deliberately returns `&[u8; 64]` "shape" iterators
//! (the consumer applies the typed reinterpret at the call site) — this is
//! the maintainer-blessed deviation from the design doc's typed-iterator
//! sketch, centralising the one allowed `unsafe` cast at the consumer rather
//! than per-iterator here.

use std::sync::Arc;

/// Multi-lane (N-wide) typed column view over a shared `Arc<[u8]>` buffer.
///
/// Useful for SIMD-staged inner loops that view the same backing bytes as
/// different SIMD lane widths without copying. The caller allocates the
/// backing buffer once; `MultiLaneColumn` holds an `Arc` reference so the
/// column can be cloned cheaply for multi-consumer access.
///
/// The backing store must be a multiple of 64 bytes (the AVX-512 register
/// width and cache-line size). [`MultiLaneColumn::new`] returns `Err(())`
/// otherwise.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::column::MultiLaneColumn;
/// use std::sync::Arc;
///
/// let data: Arc<[u8]> = Arc::from(vec![0u8; 128]);
/// let col = MultiLaneColumn::new(data).unwrap();
/// assert_eq!(col.len_bytes(), 128);
/// assert_eq!(col.len_u8x64(), 2);
/// ```
#[derive(Clone)]
pub struct MultiLaneColumn {
    data: Arc<[u8]>,
}

impl MultiLaneColumn {
    /// Construct a `MultiLaneColumn` from a shared byte buffer.
    ///
    /// Returns `Err(())` if `data.len()` is not a multiple of 64.
    ///
    /// An empty buffer (`data.len() == 0`) is accepted —
    /// [`MultiLaneColumn::is_empty`] returns `true` and all iterators
    /// yield zero windows.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::hpc::column::MultiLaneColumn;
    /// use std::sync::Arc;
    ///
    /// let ok: Arc<[u8]> = Arc::from(vec![1u8; 64]);
    /// assert!(MultiLaneColumn::new(ok).is_ok());
    ///
    /// let bad: Arc<[u8]> = Arc::from(vec![0u8; 100]);
    /// assert!(MultiLaneColumn::new(bad).is_err());
    /// ```
    pub fn new(data: Arc<[u8]>) -> Result<Self, ()> {
        if data.len() % 64 != 0 {
            return Err(());
        }
        Ok(Self { data })
    }

    /// Total byte length of the backing store.
    pub fn len_bytes(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the column has zero bytes.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Number of 64-byte (`U8x64`) chunks in this column.
    pub fn len_u8x64(&self) -> usize {
        self.data.len() / 64
    }

    /// Number of `F32x16`-shaped (16 × f32 = 64-byte) chunks.
    pub fn len_f32x16(&self) -> usize {
        self.data.len() / 64
    }

    /// Number of `F64x8`-shaped (8 × f64 = 64-byte) chunks.
    pub fn len_f64x8(&self) -> usize {
        self.data.len() / 64
    }

    /// Number of `U64x8`-shaped (8 × u64 = 64-byte) chunks.
    pub fn len_u64x8(&self) -> usize {
        self.data.len() / 64
    }

    /// View the backing store as a raw byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Iterate the column as contiguous `&[u8; 64]` windows (`U8x64` shape).
    ///
    /// Each window is exactly 64 bytes — one AVX-512 `U8x64` register load.
    /// Zero-copy: each window is a reference into the backing store.
    ///
    /// Feed each window into `U8x64::from_array(*win)` or
    /// `crate::simd::U8x64::from_slice(win)` inside the consumer's loop.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::hpc::column::MultiLaneColumn;
    /// use std::sync::Arc;
    ///
    /// let data: Arc<[u8]> = Arc::from((0u8..128).collect::<Vec<_>>());
    /// let col = MultiLaneColumn::new(data).unwrap();
    /// let windows: Vec<&[u8; 64]> = col.iter_u8x64().collect();
    /// assert_eq!(windows.len(), 2);
    /// assert_eq!(windows[0][0], 0u8);
    /// assert_eq!(windows[1][0], 64u8);
    /// ```
    pub fn iter_u8x64(&self) -> impl Iterator<Item = &[u8; 64]> + '_ {
        self.data.as_chunks::<64>().0.iter()
    }

    /// Iterate the column as `&[u8; 64]` windows reinterpreted as `[f32; 16]`-shape.
    ///
    /// The bytes are NOT converted — same memory, different lane width.
    /// Consumer is responsible for using `F32x16::from_array(bytemuck::cast(*win))`
    /// or equivalent typed reinterpretation.
    pub fn iter_f32x16_bytes(&self) -> impl Iterator<Item = &[u8; 64]> + '_ {
        self.data.as_chunks::<64>().0.iter()
    }

    /// Iterate the column as `&[u8; 64]` windows reinterpreted as `[f64; 8]`-shape.
    pub fn iter_f64x8_bytes(&self) -> impl Iterator<Item = &[u8; 64]> + '_ {
        self.data.as_chunks::<64>().0.iter()
    }

    /// Iterate the column as `&[u8; 64]` windows reinterpreted as `[u64; 8]`-shape.
    pub fn iter_u64x8_bytes(&self) -> impl Iterator<Item = &[u8; 64]> + '_ {
        self.data.as_chunks::<64>().0.iter()
    }
}

// ============================================================================
// Tests — commented-out final form
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Construction with a 64-byte buffer succeeds; len_bytes round-trips.
    #[test]
    fn new_64byte_buffer_succeeds() {
        let col = MultiLaneColumn::new(Arc::from(vec![0u8; 64])).unwrap();
        assert_eq!(col.len_bytes(), 64);
        assert_eq!(col.len_u8x64(), 1);
        assert_eq!(col.len_f32x16(), 1);
        assert_eq!(col.len_f64x8(), 1);
        assert_eq!(col.len_u64x8(), 1);
    }

    /// Construction with a non-multiple-of-64 buffer returns Err.
    #[test]
    fn new_non_multiple_of_64_errors() {
        assert!(MultiLaneColumn::new(Arc::from(vec![0u8; 100])).is_err());
        assert!(MultiLaneColumn::new(Arc::from(vec![0u8; 63])).is_err());
        assert!(MultiLaneColumn::new(Arc::from(vec![0u8; 65])).is_err());
    }

    /// Empty buffer is accepted; is_empty == true; iterators yield 0 windows.
    #[test]
    fn empty_buffer_yields_zero_windows() {
        let col = MultiLaneColumn::new(Arc::from(vec![0u8; 0])).unwrap();
        assert!(col.is_empty());
        assert_eq!(col.len_bytes(), 0);
        assert_eq!(col.iter_u8x64().count(), 0);
        assert_eq!(col.iter_f32x16_bytes().count(), 0);
        assert_eq!(col.iter_f64x8_bytes().count(), 0);
        assert_eq!(col.iter_u64x8_bytes().count(), 0);
    }

    /// Two-chunk buffer yields exactly 2 windows of 64 bytes each.
    #[test]
    fn iter_u8x64_two_chunks() {
        let mut v = vec![0u8; 128];
        for i in 0..128 {
            v[i] = i as u8;
        }
        let col = MultiLaneColumn::new(Arc::from(v)).unwrap();
        let windows: Vec<&[u8; 64]> = col.iter_u8x64().collect();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0][0], 0u8);
        assert_eq!(windows[0][63], 63u8);
        assert_eq!(windows[1][0], 64u8);
        assert_eq!(windows[1][63], 127u8);
    }

    /// Clone shares the same backing Arc (no copy).
    #[test]
    fn clone_shares_backing() {
        let col = MultiLaneColumn::new(Arc::from(vec![0u8; 64])).unwrap();
        let col2 = col.clone();
        // Both columns reference the same underlying allocation: pointer equality
        // is the observable contract without accessing private Arc internals.
        assert_eq!(
            col.as_bytes().as_ptr(),
            col2.as_bytes().as_ptr(),
            "clone must share the same Arc backing, not copy"
        );
    }

    /// Bytes-shape iterators all yield the same chunk count and content as
    /// `iter_u8x64` — they are pure aliasing views, not separate buffers.
    #[test]
    fn bytes_shape_iterators_alias_u8x64() {
        let v: Vec<u8> = (0u8..192).collect();
        let col = MultiLaneColumn::new(Arc::from(v)).unwrap();

        let u8_wins: Vec<&[u8; 64]> = col.iter_u8x64().collect();
        let f32_wins: Vec<&[u8; 64]> = col.iter_f32x16_bytes().collect();
        let f64_wins: Vec<&[u8; 64]> = col.iter_f64x8_bytes().collect();
        let u64_wins: Vec<&[u8; 64]> = col.iter_u64x8_bytes().collect();

        assert_eq!(u8_wins.len(), 3);
        assert_eq!(f32_wins.len(), 3);
        assert_eq!(f64_wins.len(), 3);
        assert_eq!(u64_wins.len(), 3);

        // Each shape iterator yields references into the same backing bytes:
        // pointer equality across the four iterators on every chunk.
        for i in 0..3 {
            assert_eq!(u8_wins[i].as_ptr(), f32_wins[i].as_ptr());
            assert_eq!(u8_wins[i].as_ptr(), f64_wins[i].as_ptr());
            assert_eq!(u8_wins[i].as_ptr(), u64_wins[i].as_ptr());
            assert_eq!(u8_wins[i][0], (i as u8) * 64);
            assert_eq!(u8_wins[i][63], (i as u8) * 64 + 63);
        }
    }

    /// `as_bytes()` returns the full backing slice and aliases the Arc storage.
    #[test]
    fn as_bytes_returns_full_backing_slice() {
        let v: Vec<u8> = (0u8..64).collect();
        let arc: Arc<[u8]> = Arc::from(v);
        let arc_ptr = arc.as_ptr();
        let col = MultiLaneColumn::new(arc).unwrap();
        let bytes = col.as_bytes();
        assert_eq!(bytes.len(), 64);
        assert_eq!(bytes.as_ptr(), arc_ptr, "as_bytes must alias the Arc backing, not copy");
        for (i, &b) in bytes.iter().enumerate() {
            assert_eq!(b, i as u8);
        }
    }

    /// Static assertion: `MultiLaneColumn` is `Send + Sync`, so it can cross
    /// thread boundaries — required for cognitive-shader-stack multi-consumer
    /// access patterns.
    #[test]
    fn multilane_column_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MultiLaneColumn>();
    }
}
