//! Const-size sliding-window helpers over `&[T]` (PR-X1).
//!
//! Centralises the `&[T] → impl Iterator<Item = &[T; N]>` pattern that
//! SIMD-staged consumers use to walk a buffer N lanes at a time. Without
//! this helper, each consumer either calls `slice::as_chunks::<N>` directly
//! (fine but undiscoverable) or rolls a raw slice index without the
//! compile-time bounds check.
//!
//! # Semantics — non-overlapping (NOT std `array_windows`)
//!
//! Despite the plural name, this is the **non-overlapping** chunk variant —
//! consecutive windows do not share elements. The name follows std's
//! plural iterator-type convention ([`std::slice::ArrayWindows`]), but the
//! semantics match [`std::slice::ArrayChunks`] / `slice::as_chunks`.
//!
//! For an 8-element input walked at `N = 4`:
//! - `array_windows` (this module): yields `[0..4]`, `[4..8]` — **2 windows**.
//! - `std::slice::array_windows` (overlapping): would yield 5 windows.
//!
//! Non-overlapping is the correct shape for SIMD-staged inner loops where
//! each lane-register load consumes its N elements before advancing by N.
//!
//! # Layering
//!
//! Lives in `hpc::array_windows`, re-exported from `crate::simd::*` per the
//! W1a consumer contract at
//! `.claude/knowledge/vertical-simd-consumer-contract.md`.
//!
//! # Design reference
//!
//! `.claude/knowledge/pr-x1-design.md` § "3. `array_window`". This module
//! ships the **iterator-shape** variant (whole-buffer walk yielding all
//! const-size windows); the design doc's singular-window sketch
//! (`array_window(slice, offset) -> &[T; N]`) was superseded by the
//! iterator form here, which composes directly with SIMD-staged consumer
//! loops and avoids per-call panic surface in tight inner loops. The name
//! was pluralised to match the std iterator-type convention.

/// Walk `data` as a sequence of non-overlapping const-size windows.
///
/// Returns an iterator over `&[T; N]` references into `data`. The tail
/// (`data.len() % N` items) is discarded; use [`array_windows_checked`] to
/// fail-fast when the length is not a multiple of `N`.
///
/// Zero-cost: this is a thin wrapper around [`slice::as_chunks`] that pins
/// the chunk size at the call site for type inference.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::array_windows::array_windows;
/// let data: Vec<u8> = (0..16).collect();
/// let windows: Vec<&[u8; 4]> = array_windows::<u8, 4>(&data).collect();
/// assert_eq!(windows.len(), 4);
/// assert_eq!(windows[0], &[0, 1, 2, 3]);
/// assert_eq!(windows[3], &[12, 13, 14, 15]);
/// ```
///
/// # Examples — tail discarded
///
/// ```
/// use ndarray::hpc::array_windows::array_windows;
/// let data: Vec<u8> = (0..7).collect();
/// let windows: Vec<&[u8; 4]> = array_windows::<u8, 4>(&data).collect();
/// // 7 / 4 = 1 window; the trailing 3 items are dropped.
/// assert_eq!(windows.len(), 1);
/// ```
#[inline]
pub fn array_windows<T, const N: usize>(data: &[T]) -> impl Iterator<Item = &[T; N]> + '_ {
    data.as_chunks::<N>().0.iter()
}

/// Walk `data` as `&[T; N]` windows, returning `Err(())` if `data.len()`
/// is not a multiple of `N`.
///
/// This is the strict variant of [`array_windows`]: the consumer asserts up
/// front that the buffer is lane-aligned and the caller wants the error
/// surfaced rather than silently truncating.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::array_windows::array_windows_checked;
/// let data: Vec<u8> = (0..16).collect();
/// let it = array_windows_checked::<u8, 4>(&data).expect("16 is a multiple of 4");
/// assert_eq!(it.count(), 4);
///
/// let bad: Vec<u8> = (0..7).collect();
/// assert!(array_windows_checked::<u8, 4>(&bad).is_err());
/// ```
#[inline]
pub fn array_windows_checked<T, const N: usize>(
    data: &[T],
) -> Result<impl Iterator<Item = &[T; N]> + '_, ()> {
    if data.len() % N != 0 {
        return Err(());
    }
    Ok(array_windows::<T, N>(data))
}

// ============================================================================
// Tests — commented-out final form
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 16-element buffer yields four 4-wide windows.
    #[test]
    fn array_windows_4_over_16() {
        let data: Vec<u8> = (0u8..16).collect();
        let windows: Vec<&[u8; 4]> = array_windows::<u8, 4>(&data).collect();
        assert_eq!(windows.len(), 4);
        assert_eq!(windows[0], &[0, 1, 2, 3]);
        assert_eq!(windows[1], &[4, 5, 6, 7]);
        assert_eq!(windows[2], &[8, 9, 10, 11]);
        assert_eq!(windows[3], &[12, 13, 14, 15]);
    }

    /// Tail items are silently discarded by `array_windows`.
    #[test]
    fn array_windows_drops_tail() {
        let data: Vec<u8> = (0u8..7).collect();
        let windows: Vec<&[u8; 4]> = array_windows::<u8, 4>(&data).collect();
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0], &[0, 1, 2, 3]);
    }

    /// Mismatched length surfaces as Err in the checked variant.
    #[test]
    fn array_windows_checked_rejects_mismatch() {
        assert!(array_windows_checked::<u8, 4>(&[0u8; 7]).is_err());
        assert!(array_windows_checked::<u8, 4>(&[0u8; 5]).is_err());
        assert!(array_windows_checked::<u8, 4>(&[0u8; 1]).is_err());
    }

    /// Aligned length succeeds in the checked variant.
    #[test]
    fn array_windows_checked_accepts_aligned() {
        let data = [0u8; 16];
        let it = array_windows_checked::<u8, 4>(&data).expect("16 is a multiple of 4");
        assert_eq!(it.count(), 4);
    }

    /// Empty buffer yields zero windows (not an error in either variant).
    #[test]
    fn array_windows_empty_buffer() {
        assert_eq!(array_windows::<u8, 4>(&[]).count(), 0);
        let it = array_windows_checked::<u8, 4>(&[]).expect("0 % 4 == 0, should be Ok");
        assert_eq!(it.count(), 0);
    }
}
