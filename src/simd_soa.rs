//! SoA-shaped SIMD substrate primitives (PR-X1).
//!
//! Lives at the crate root under `crate::simd_soa::*` and is re-exported
//! through `crate::simd::*` per the W1a consumer contract. This is the
//! canonical home for SoA-of-bytes / multi-lane column carriers and the
//! const-size chunk-walking helpers that SIMD-staged inner loops use.
//!
//! # What lives here
//!
//! - [`MultiLaneColumn`] — `Arc<[u8]>` carrier with typed-width chunk iters
//! - [`array_chunks`] / [`array_chunks_checked`] — generic const-size
//!   non-overlapping `&[T] → impl Iterator<Item = &[T; N]>` helpers
//!
//! # Layering
//!
//! This module is **layout-only**. No `#[target_feature]`, no per-arch
//! imports, no raw intrinsics. The SIMD register load happens inside the
//! consumer's loop using `crate::simd::F32x16::from_array` etc. The
//! `simd.rs` dispatcher re-exports these primitives via `pub use
//! crate::simd_soa::{…};` so consumers always go through
//! `use ndarray::simd::*;`.
//!
//! # Distance typing
//!
//! These types are layout-only. No distance-aware API. See
//! `.claude/knowledge/cognitive-distance-typing.md` (no-umbrella rule).
//!
//! # Design references
//!
//! - `.claude/knowledge/pr-x1-design.md` § "1. `MultiLaneColumn`"
//! - `.claude/knowledge/pr-x1-design.md` § "3. `array_window`" (the
//!   singular-window sketch superseded by the iterator form here; the
//!   name landed as `array_chunks` to avoid collision with std's
//!   nightly overlapping `slice::array_windows`).

use std::sync::Arc;

// ════════════════════════════════════════════════════════════════════
// MultiLaneColumn — Arc<[u8]> carrier with typed lane-width chunk iters
// ════════════════════════════════════════════════════════════════════

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
/// use ndarray::simd::MultiLaneColumn;
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
    /// use ndarray::simd::MultiLaneColumn;
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
    /// use ndarray::simd::MultiLaneColumn;
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

// ════════════════════════════════════════════════════════════════════
// array_chunks — generic non-overlapping const-size chunk helpers
// ════════════════════════════════════════════════════════════════════
//
// Naming: `array_chunks` (NOT `array_windows`) because:
// - `std::slice::array_windows::<N>()` (nightly) is the **overlapping**
//   variant, already referenced in src/simd.rs comments.
// - These helpers are the **non-overlapping** variant, matching
//   `std::slice::ArrayChunks` / stable `slice::as_chunks`.

/// Walk `data` as a sequence of non-overlapping const-size windows.
///
/// Returns an iterator over `&[T; N]` references into `data`. The tail
/// (`data.len() % N` items) is discarded; use [`array_chunks_checked`] to
/// fail-fast when the length is not a multiple of `N`.
///
/// Zero-cost: this is a thin wrapper around [`slice::as_chunks`] that pins
/// the chunk size at the call site for type inference.
///
/// # Examples
///
/// ```
/// use ndarray::simd::array_chunks;
/// let data: Vec<u8> = (0..16).collect();
/// let windows: Vec<&[u8; 4]> = array_chunks::<u8, 4>(&data).collect();
/// assert_eq!(windows.len(), 4);
/// assert_eq!(windows[0], &[0, 1, 2, 3]);
/// assert_eq!(windows[3], &[12, 13, 14, 15]);
/// ```
///
/// # Examples — tail discarded
///
/// ```
/// use ndarray::simd::array_chunks;
/// let data: Vec<u8> = (0..7).collect();
/// let windows: Vec<&[u8; 4]> = array_chunks::<u8, 4>(&data).collect();
/// assert_eq!(windows.len(), 1);
/// ```
#[inline]
pub fn array_chunks<T, const N: usize>(data: &[T]) -> impl Iterator<Item = &[T; N]> + '_ {
    data.as_chunks::<N>().0.iter()
}

/// Walk `data` as `&[T; N]` windows, returning `Err(())` if `data.len()`
/// is not a multiple of `N`.
///
/// Strict variant of [`array_chunks`]: the consumer asserts up front that
/// the buffer is lane-aligned and wants the error surfaced rather than
/// silently truncating.
///
/// # Examples
///
/// ```
/// use ndarray::simd::array_chunks_checked;
/// let data: Vec<u8> = (0..16).collect();
/// let it = array_chunks_checked::<u8, 4>(&data).expect("16 is a multiple of 4");
/// assert_eq!(it.count(), 4);
///
/// let bad: Vec<u8> = (0..7).collect();
/// assert!(array_chunks_checked::<u8, 4>(&bad).is_err());
/// ```
#[inline]
pub fn array_chunks_checked<T, const N: usize>(
    data: &[T],
) -> Result<impl Iterator<Item = &[T; N]> + '_, ()> {
    if data.len() % N != 0 {
        return Err(());
    }
    Ok(array_chunks::<T, N>(data))
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MultiLaneColumn ----

    #[test]
    fn new_64byte_buffer_succeeds() {
        let col = MultiLaneColumn::new(Arc::from(vec![0u8; 64])).unwrap();
        assert_eq!(col.len_bytes(), 64);
        assert_eq!(col.len_u8x64(), 1);
        assert_eq!(col.len_f32x16(), 1);
        assert_eq!(col.len_f64x8(), 1);
        assert_eq!(col.len_u64x8(), 1);
    }

    #[test]
    fn new_non_multiple_of_64_errors() {
        assert!(MultiLaneColumn::new(Arc::from(vec![0u8; 100])).is_err());
        assert!(MultiLaneColumn::new(Arc::from(vec![0u8; 63])).is_err());
        assert!(MultiLaneColumn::new(Arc::from(vec![0u8; 65])).is_err());
    }

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

    #[test]
    fn clone_shares_backing() {
        let col = MultiLaneColumn::new(Arc::from(vec![0u8; 64])).unwrap();
        let col2 = col.clone();
        assert_eq!(
            col.as_bytes().as_ptr(),
            col2.as_bytes().as_ptr(),
            "clone must share the same Arc backing, not copy"
        );
    }

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

        for i in 0..3 {
            assert_eq!(u8_wins[i].as_ptr(), f32_wins[i].as_ptr());
            assert_eq!(u8_wins[i].as_ptr(), f64_wins[i].as_ptr());
            assert_eq!(u8_wins[i].as_ptr(), u64_wins[i].as_ptr());
            assert_eq!(u8_wins[i][0], (i as u8) * 64);
            assert_eq!(u8_wins[i][63], (i as u8) * 64 + 63);
        }
    }

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

    #[test]
    fn multilane_column_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MultiLaneColumn>();
    }

    // ---- array_chunks ----

    #[test]
    fn array_chunks_4_over_16() {
        let data: Vec<u8> = (0u8..16).collect();
        let windows: Vec<&[u8; 4]> = array_chunks::<u8, 4>(&data).collect();
        assert_eq!(windows.len(), 4);
        assert_eq!(windows[0], &[0, 1, 2, 3]);
        assert_eq!(windows[1], &[4, 5, 6, 7]);
        assert_eq!(windows[2], &[8, 9, 10, 11]);
        assert_eq!(windows[3], &[12, 13, 14, 15]);
    }

    #[test]
    fn array_chunks_drops_tail() {
        let data: Vec<u8> = (0u8..7).collect();
        let windows: Vec<&[u8; 4]> = array_chunks::<u8, 4>(&data).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0], &[0, 1, 2, 3]);
    }

    #[test]
    fn array_chunks_checked_rejects_mismatch() {
        assert!(array_chunks_checked::<u8, 4>(&[0u8; 7]).is_err());
        assert!(array_chunks_checked::<u8, 4>(&[0u8; 5]).is_err());
        assert!(array_chunks_checked::<u8, 4>(&[0u8; 1]).is_err());
    }

    #[test]
    fn array_chunks_checked_accepts_aligned() {
        let data = [0u8; 16];
        let it = array_chunks_checked::<u8, 4>(&data).expect("16 is a multiple of 4");
        assert_eq!(it.count(), 4);
    }

    #[test]
    fn array_chunks_empty_buffer() {
        assert_eq!(array_chunks::<u8, 4>(&[]).count(), 0);
        let it = array_chunks_checked::<u8, 4>(&[]).expect("0 % 4 == 0, should be Ok");
        assert_eq!(it.count(), 0);
    }
}
