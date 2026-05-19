//! Const-size sliding-window helpers over `&[T]` (PR-X1).
//!
//! Centralises the `&[T] → impl Iterator<Item = &[T; N]>` pattern that
//! SIMD-staged consumers use to walk a buffer N lanes at a time. Without
//! this helper, each consumer either calls `slice::as_chunks::<N>` directly
//! (fine but undiscoverable) or rolls a raw slice index without the
//! compile-time bounds check.
//!
//! # Layering
//!
//! Lives in `hpc::array_window`, re-exported from `crate::simd::*` per the
//! W1a consumer contract at
//! `.claude/knowledge/vertical-simd-consumer-contract.md`.
//!
//! # Design reference
//!
//! `.claude/knowledge/pr-x1-design.md` § "3. `array_window`" — verbatim API
//! surface; this file is the commented-out final form for the PR-X1 sprint.

/// Walk `data` as a sequence of non-overlapping const-size windows.
///
/// Returns an iterator over `&[T; N]` references into `data`. The tail
/// (`data.len() % N` items) is discarded; use [`array_window_checked`] to
/// fail-fast when the length is not a multiple of `N`.
///
/// Zero-cost: this is a thin wrapper around [`slice::as_chunks`] that pins
/// the chunk size at the call site for type inference.
///
/// # Examples
///
/// ```
/// use ndarray::simd::array_window;
/// let data: Vec<u8> = (0..16).collect();
/// let windows: Vec<&[u8; 4]> = array_window::<u8, 4>(&data).collect();
/// assert_eq!(windows.len(), 4);
/// assert_eq!(windows[0], &[0, 1, 2, 3]);
/// assert_eq!(windows[3], &[12, 13, 14, 15]);
/// ```
///
/// # Examples — tail discarded
///
/// ```
/// use ndarray::simd::array_window;
/// let data: Vec<u8> = (0..7).collect();
/// let windows: Vec<&[u8; 4]> = array_window::<u8, 4>(&data).collect();
/// // 7 / 4 = 1 window; the trailing 3 items are dropped.
/// assert_eq!(windows.len(), 1);
/// ```
#[inline]
pub fn array_window<T, const N: usize>(_data: &[T]) -> impl Iterator<Item = &[T; N]> + '_ {
    // Skeleton: `data.as_chunks::<N>().0.iter()` once that API stabilises,
    // or a manual chunks loop yielding `<&[T] as TryInto<&[T; N]>>::try_into`.
    // Implementation lands in the uncomment sprint.
    core::iter::empty::<&[T; N]>()
}

/// Walk `data` as `&[T; N]` windows, returning `Err(())` if `data.len()`
/// is not a multiple of `N`.
///
/// This is the strict variant of [`array_window`]: the consumer asserts up
/// front that the buffer is lane-aligned and the caller wants the error
/// surfaced rather than silently truncating.
///
/// # Examples
///
/// ```
/// use ndarray::simd::array_window_checked;
/// let data: Vec<u8> = (0..16).collect();
/// let it = array_window_checked::<u8, 4>(&data).expect("16 is a multiple of 4");
/// assert_eq!(it.count(), 4);
///
/// let bad: Vec<u8> = (0..7).collect();
/// assert!(array_window_checked::<u8, 4>(&bad).is_err());
/// ```
#[inline]
pub fn array_window_checked<T, const N: usize>(
    data: &[T],
) -> Result<impl Iterator<Item = &[T; N]> + '_, ()> {
    if data.len() % N != 0 {
        return Err(());
    }
    Ok(array_window::<T, N>(data))
}

// ============================================================================
// Tests — commented-out final form
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 16-element buffer yields four 4-wide windows.
    #[test]
    fn array_window_4_over_16() {
        unimplemented!("PR-X1 test: collect into Vec<&[u8;4]>; assert windows.len() == 4 and contents match 0..16")
    }

    /// Tail items are silently discarded by `array_window`.
    #[test]
    fn array_window_drops_tail() {
        unimplemented!("PR-X1 test: 7-element buffer over N=4 → 1 window; trailing 3 items dropped")
    }

    /// Mismatched length surfaces as Err in the checked variant.
    #[test]
    fn array_window_checked_rejects_mismatch() {
        unimplemented!("PR-X1 test: assert!(array_window_checked::<u8,4>(&[0u8;7]).is_err())")
    }

    /// Aligned length succeeds in the checked variant.
    #[test]
    fn array_window_checked_accepts_aligned() {
        unimplemented!("PR-X1 test: array_window_checked::<u8,4>(&[0u8;16]) returns Ok iterator yielding 4 windows")
    }

    /// Empty buffer yields zero windows (not an error in either variant).
    #[test]
    fn array_window_empty_buffer() {
        unimplemented!("PR-X1 test: array_window::<u8,4>(&[]).count() == 0; array_window_checked is Ok (0 % 4 == 0)")
    }
}
