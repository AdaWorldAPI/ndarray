//! Bulk traversal helpers for AoS slices.
//!
//! [`bulk_apply`] chunks a `&mut [T]` and invokes a closure with each chunk
//! plus its starting index. Useful when you want predictable cache behavior
//! (chunk_size matched to L1 working-set) or when staging chunks to SoA for
//! SIMD processing inside the closure.
//!
//! [`bulk_scan`] is the read-only sibling for non-mutating traversal.
//!
//! Both helpers are scalar wrappers — no `#[target_feature]`, no per-arch
//! dispatch. They are user-level code per the layering rule in
//! `.claude/knowledge/vertical-simd-consumer-contract.md`: only the dispatch
//! layer (`crate::simd`, `crate::simd_ops`) and per-tier impls
//! (`simd_avx512.rs`, `simd_avx2.rs`, `simd_neon.rs`) may carry SIMD
//! attributes. The public API here is forward-compatible: a future
//! bench-justified wave can swap in SIMD-accelerated chunk iteration via
//! the dispatch layer without breaking callers.
//!
//! # Composition with SoA staging
//!
//! `bulk_apply` composes naturally with `crate::hpc::soa::aos_to_soa` inside
//! the closure body when the caller wants per-chunk SoA staging (e.g. for
//! cache-blocked SIMD-style loops on each field):
//!
//! ```ignore
//! use ndarray::hpc::bulk::bulk_apply;
//! use ndarray::hpc::soa::aos_to_soa;
//! struct Item { a: f32, b: f32, c: f32 }
//! let mut items: Vec<Item> = (0..100)
//!     .map(|i| Item { a: i as f32, b: (i * 2) as f32, c: (i * 3) as f32 })
//!     .collect();
//! bulk_apply(&mut items, 16, |chunk, _start| {
//!     let soa = aos_to_soa::<_, 3, _>(chunk, |it| [it.a, it.b, it.c]);
//!     // ... per-field SIMD-style loops over soa.field(0), soa.field(1), ...
//!     let _ = soa;
//! });
//! ```
//!
//! # Out of scope — distance metrics
//!
//! These helpers stay generic over `T`. They MUST NOT grow toward distance
//! computation (no `bulk_distance<T>` umbrella, no `enum DistanceMetric`).
//! Distance metrics in this codebase are typed — one named fn per metric.
//! See `.claude/knowledge/cognitive-distance-typing.md` for the binding rule.

/// Apply `f` to consecutive chunks of `items`. Each invocation receives the
/// chunk slice and the absolute index of the chunk's first element.
///
/// The last chunk may be shorter than `chunk_size` when `chunk_size` does
/// not divide `items.len()`. A `chunk_size` of `usize::MAX` yields the
/// entire slice as a single chunk.
///
/// # Panics
/// Panics if `chunk_size == 0` (`chunks_mut(0)` would otherwise return an
/// iterator that does not make progress).
///
/// # Example
/// ```
/// use ndarray::hpc::bulk::bulk_apply;
/// let mut v: Vec<i32> = (0..10).collect();
/// bulk_apply(&mut v, 3, |chunk, start| {
///     for (i, x) in chunk.iter_mut().enumerate() {
///         *x = (start + i) as i32 * 10;
///     }
/// });
/// assert_eq!(v, vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
/// ```
pub fn bulk_apply<T, F>(items: &mut [T], chunk_size: usize, mut f: F)
where
    F: FnMut(&mut [T], usize),
{
    assert!(chunk_size > 0, "bulk_apply: chunk_size must be > 0");
    let mut start = 0;
    for chunk in items.chunks_mut(chunk_size) {
        let n = chunk.len();
        f(chunk, start);
        start += n;
    }
}

/// Read-only sibling of [`bulk_apply`]. Applies `f` to consecutive immutable
/// chunks of `items`, passing the absolute starting index of each chunk.
///
/// The last chunk may be shorter than `chunk_size`. A `chunk_size` of
/// `usize::MAX` yields the entire slice as a single chunk.
///
/// # Panics
/// Panics if `chunk_size == 0`.
///
/// # Example
/// ```
/// use ndarray::hpc::bulk::bulk_scan;
/// let v: Vec<i32> = (0..10).collect();
/// let mut sum = 0i32;
/// bulk_scan(&v, 4, |chunk, _start| {
///     sum += chunk.iter().sum::<i32>();
/// });
/// assert_eq!(sum, 45);
/// ```
pub fn bulk_scan<T, F>(items: &[T], chunk_size: usize, mut f: F)
where
    F: FnMut(&[T], usize),
{
    assert!(chunk_size > 0, "bulk_scan: chunk_size must be > 0");
    let mut start = 0;
    for chunk in items.chunks(chunk_size) {
        let n = chunk.len();
        f(chunk, start);
        start += n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----- bulk_apply -----

    #[test]
    fn bulk_apply_chunk_size_divides_len() {
        // len == 10, chunk_size == 5 → exactly 2 chunks of 5.
        let mut v: Vec<i32> = (0..10).collect();
        let mut sizes = Vec::new();
        bulk_apply(&mut v, 5, |chunk, _start| {
            sizes.push(chunk.len());
        });
        assert_eq!(sizes, vec![5, 5]);
    }

    #[test]
    fn bulk_apply_chunk_size_does_not_divide_len() {
        // len == 10, chunk_size == 3 → 3 + 3 + 3 + 1.
        let mut v: Vec<i32> = (0..10).collect();
        let mut sizes = Vec::new();
        bulk_apply(&mut v, 3, |chunk, _start| {
            sizes.push(chunk.len());
        });
        assert_eq!(sizes, vec![3, 3, 3, 1]);
    }

    #[test]
    fn bulk_apply_chunk_size_greater_than_len() {
        // chunk_size > len → a single chunk of len rows.
        let mut v: Vec<i32> = (0..10).collect();
        let mut sizes = Vec::new();
        bulk_apply(&mut v, 100, |chunk, start| {
            assert_eq!(start, 0);
            sizes.push(chunk.len());
        });
        assert_eq!(sizes, vec![10]);
    }

    #[test]
    fn bulk_apply_start_indices_3_3_3_1() {
        // The 3-3-3-1 chunking should produce starts [0, 3, 6, 9].
        let mut v: Vec<i32> = (0..10).collect();
        let mut start_indices: Vec<usize> = Vec::new();
        bulk_apply(&mut v, 3, |_chunk, start| {
            start_indices.push(start);
        });
        assert_eq!(start_indices, vec![0, 3, 6, 9]);
    }

    #[test]
    fn bulk_apply_mutates_via_start_plus_offset() {
        // The closure can compute each element's absolute index from
        // `start + i` and overwrite. Verifies the start index is correct.
        let mut v: Vec<i32> = vec![0; 10];
        bulk_apply(&mut v, 3, |chunk, start| {
            for (i, x) in chunk.iter_mut().enumerate() {
                *x = (start + i) as i32 * 10;
            }
        });
        assert_eq!(v, vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
    }

    #[test]
    #[should_panic(expected = "chunk_size must be > 0")]
    fn bulk_apply_panics_on_zero_chunk_size() {
        let mut v: Vec<i32> = (0..4).collect();
        bulk_apply(&mut v, 0, |_, _| {});
    }

    #[test]
    fn bulk_apply_chunk_size_usize_max_single_chunk() {
        // stdlib `chunks_mut(usize::MAX)` yields a single chunk equal to the
        // whole slice. Smoke-test: doesn't loop, doesn't panic, one chunk.
        let mut v: Vec<i32> = (0..4).collect();
        let mut count = 0;
        bulk_apply(&mut v, usize::MAX, |chunk, start| {
            count += 1;
            assert_eq!(start, 0);
            assert_eq!(chunk.len(), 4);
        });
        assert_eq!(count, 1);
    }

    #[test]
    fn bulk_apply_empty_slice() {
        // Empty input: closure never invoked.
        let mut v: Vec<i32> = Vec::new();
        let mut count = 0;
        bulk_apply(&mut v, 4, |_, _| {
            count += 1;
        });
        assert_eq!(count, 0);
    }

    // ----- bulk_scan -----

    #[test]
    fn bulk_scan_chunk_size_divides_len() {
        let v: Vec<i32> = (0..10).collect();
        let mut sizes = Vec::new();
        bulk_scan(&v, 5, |chunk, _start| {
            sizes.push(chunk.len());
        });
        assert_eq!(sizes, vec![5, 5]);
    }

    #[test]
    fn bulk_scan_chunk_size_does_not_divide_len() {
        let v: Vec<i32> = (0..10).collect();
        let mut sizes = Vec::new();
        bulk_scan(&v, 3, |chunk, _start| {
            sizes.push(chunk.len());
        });
        assert_eq!(sizes, vec![3, 3, 3, 1]);
    }

    #[test]
    fn bulk_scan_chunk_size_greater_than_len() {
        let v: Vec<i32> = (0..10).collect();
        let mut sizes = Vec::new();
        bulk_scan(&v, 100, |chunk, start| {
            assert_eq!(start, 0);
            sizes.push(chunk.len());
        });
        assert_eq!(sizes, vec![10]);
    }

    #[test]
    fn bulk_scan_start_indices_3_3_3_1() {
        let v: Vec<i32> = (0..10).collect();
        let mut start_indices: Vec<usize> = Vec::new();
        bulk_scan(&v, 3, |_chunk, start| {
            start_indices.push(start);
        });
        assert_eq!(start_indices, vec![0, 3, 6, 9]);
    }

    #[test]
    fn bulk_scan_sums_chunks() {
        let v: Vec<i32> = (0..10).collect();
        let mut sum = 0i32;
        bulk_scan(&v, 4, |chunk, _start| {
            sum += chunk.iter().sum::<i32>();
        });
        assert_eq!(sum, 45);
    }

    #[test]
    #[should_panic(expected = "chunk_size must be > 0")]
    fn bulk_scan_panics_on_zero_chunk_size() {
        let v: Vec<i32> = (0..4).collect();
        bulk_scan(&v, 0, |_, _| {});
    }

    #[test]
    fn bulk_scan_chunk_size_usize_max_single_chunk() {
        let v: Vec<i32> = (0..4).collect();
        let mut count = 0;
        bulk_scan(&v, usize::MAX, |chunk, start| {
            count += 1;
            assert_eq!(start, 0);
            assert_eq!(chunk.len(), 4);
        });
        assert_eq!(count, 1);
    }

    #[test]
    fn bulk_scan_empty_slice() {
        let v: Vec<i32> = Vec::new();
        let mut count = 0;
        bulk_scan(&v, 4, |_, _| {
            count += 1;
        });
        assert_eq!(count, 0);
    }

    // ----- integration with aos_to_soa -----
    //
    // hpc::soa and hpc::bulk co-merge in PR #156, so the worker-isolation
    // deferral is no longer needed. This test exercises the canonical
    // compose pattern: a `bulk_apply` outer chunk loop with `aos_to_soa`
    // staging inside the closure (SoA-stage-then-process pattern that a
    // SIMD consumer would use per-tile).
    #[test]
    fn bulk_apply_composes_with_aos_to_soa() {
        use crate::hpc::soa::aos_to_soa;

        struct Item {
            a: f32,
            b: f32,
            c: f32,
        }

        let mut items: Vec<Item> = (0..100)
            .map(|i| Item {
                a: i as f32,
                b: (i * 2) as f32,
                c: (i * 3) as f32,
            })
            .collect();

        let mut chunk_count = 0;
        bulk_apply(&mut items, 16, |chunk, start_idx| {
            let soa = aos_to_soa::<_, 3, _>(chunk, |it| [it.a, it.b, it.c]);
            assert_eq!(soa.len(), chunk.len());
            // First row of the chunk corresponds to absolute index start_idx.
            assert_eq!(soa.field(0)[0], start_idx as f32);
            chunk_count += 1;
        });
        // 100 / 16 = 6 full chunks of 16 + 1 tail of 4 = 7 chunks total.
        assert_eq!(chunk_count, 7);
    }
}
