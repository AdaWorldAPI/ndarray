//! SoA-shaped SIMD substrate carriers (PR-X1).
//!
//! Lives at the crate root under `crate::simd_soa::*` and is re-exported
//! through `crate::simd::*` per the W1a consumer contract. This is the
//! canonical home for SoA-of-bytes / multi-lane column carriers.
//!
//! Generic slice / chunk helpers (`array_chunks`, `array_chunks_checked`)
//! live in `crate::simd_ops` — they're operations, not carriers.
//!
//! # What lives here
//!
//! - [`MultiLaneColumn`] — `Arc<[u8]>` carrier with typed-width chunk iters
//!
//! # Layering
//!
//! This module is **layout-only**. No `#[target_feature]`, no per-arch
//! imports, no raw intrinsics. The SIMD register load happens inside the
//! consumer's loop using `crate::simd::F32x16::from_array` etc. The
//! `simd.rs` dispatcher re-exports these carriers via `pub use
//! crate::simd_soa::{…};` so consumers always go through
//! `use ndarray::simd::*;`.
//!
//! # Distance typing
//!
//! These types are layout-only. No distance-aware API. See
//! `.claude/knowledge/cognitive-distance-typing.md` (no-umbrella rule).
//!
//! # Design reference
//!
//! `.claude/knowledge/pr-x1-design.md` § "1. `MultiLaneColumn`".

use std::sync::Arc;

// Typed lane primitives — dispatched through `crate::simd::*`, which
// re-exports the right backend (AVX-512 / NEON / scalar) per `cfg`. Per
// the W1a layering rule, `simd_soa.rs` MUST go through `crate::simd::`
// rather than dipping into `simd_avx512` / `simd_neon` / `scalar` directly.
use crate::simd::{F32x16, F64x8, I32x16, I64x8, U32x16, U64x8, U8x64};

// Endian-correct `&[u8; 4]` → `f32` / `&[u8; 8]` → `f64`/`u64` helpers.
// `f32::from_le_bytes` is intrinsically optimised to a single load on
// little-endian targets (x86_64, aarch64, wasm32), so this scalar
// `from_fn` loop compiles to the same instruction stream as a
// `bytemuck::cast`-style reinterpret — without requiring a new workspace
// dep and without the alignment risk of pointer-casting `Arc<[u8]>`
// (which is only `u8`-aligned in stable Rust).
#[inline(always)]
fn f32x16_from_chunk(chunk: &[u8; 64]) -> F32x16 {
    let arr: [f32; 16] = core::array::from_fn(|i| {
        let off = i * 4;
        f32::from_le_bytes([chunk[off], chunk[off + 1], chunk[off + 2], chunk[off + 3]])
    });
    F32x16::from_array(arr)
}

#[inline(always)]
fn f64x8_from_chunk(chunk: &[u8; 64]) -> F64x8 {
    let arr: [f64; 8] = core::array::from_fn(|i| {
        let off = i * 8;
        f64::from_le_bytes([
            chunk[off],
            chunk[off + 1],
            chunk[off + 2],
            chunk[off + 3],
            chunk[off + 4],
            chunk[off + 5],
            chunk[off + 6],
            chunk[off + 7],
        ])
    });
    F64x8::from_array(arr)
}

#[inline(always)]
fn u64x8_from_chunk(chunk: &[u8; 64]) -> U64x8 {
    let arr: [u64; 8] = core::array::from_fn(|i| {
        let off = i * 8;
        u64::from_le_bytes([
            chunk[off],
            chunk[off + 1],
            chunk[off + 2],
            chunk[off + 3],
            chunk[off + 4],
            chunk[off + 5],
            chunk[off + 6],
            chunk[off + 7],
        ])
    });
    U64x8::from_array(arr)
}

#[inline(always)]
fn u32x16_from_chunk(chunk: &[u8; 64]) -> U32x16 {
    let arr: [u32; 16] = core::array::from_fn(|i| {
        let off = i * 4;
        u32::from_le_bytes([chunk[off], chunk[off + 1], chunk[off + 2], chunk[off + 3]])
    });
    U32x16::from_array(arr)
}

#[inline(always)]
fn i32x16_from_chunk(chunk: &[u8; 64]) -> I32x16 {
    let arr: [i32; 16] = core::array::from_fn(|i| {
        let off = i * 4;
        i32::from_le_bytes([chunk[off], chunk[off + 1], chunk[off + 2], chunk[off + 3]])
    });
    I32x16::from_array(arr)
}

#[inline(always)]
fn i64x8_from_chunk(chunk: &[u8; 64]) -> I64x8 {
    let arr: [i64; 8] = core::array::from_fn(|i| {
        let off = i * 8;
        i64::from_le_bytes([
            chunk[off],
            chunk[off + 1],
            chunk[off + 2],
            chunk[off + 3],
            chunk[off + 4],
            chunk[off + 5],
            chunk[off + 6],
            chunk[off + 7],
        ])
    });
    I64x8::from_array(arr)
}

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
    #[allow(clippy::result_unit_err)] // matches PR-X1 design § 1 `Result<Self, ()>` contract
    pub fn new(data: Arc<[u8]>) -> Result<Self, ()> {
        if !data.len().is_multiple_of(64) {
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

    /// Number of `U32x16`-shaped (16 × u32 = 64-byte) chunks.
    pub fn len_u32x16(&self) -> usize {
        self.data.len() / 64
    }

    /// Number of `I32x16`-shaped (16 × i32 = 64-byte) chunks.
    pub fn len_i32x16(&self) -> usize {
        self.data.len() / 64
    }

    /// Number of `I64x8`-shaped (8 × i64 = 64-byte) chunks.
    pub fn len_i64x8(&self) -> usize {
        self.data.len() / 64
    }

    /// View the backing store as a raw byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Iterate the column as typed [`U8x64`] values dispatched via
    /// `crate::simd::*` (AVX-512 / NEON / scalar per `cfg`).
    ///
    /// Each yielded value is one register-width load over a 64-byte chunk
    /// of the backing store. The construction is zero-cost on every backend:
    /// `U8x64::from_array(*chunk)` is a single move on AVX-512, a paired
    /// LD2 on NEON, and a memcpy on the scalar fallback.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::simd::{MultiLaneColumn, U8x64};
    /// use std::sync::Arc;
    ///
    /// let data: Arc<[u8]> = Arc::from((0u8..128).collect::<Vec<_>>());
    /// let col = MultiLaneColumn::new(data).unwrap();
    /// let lanes: Vec<U8x64> = col.iter_u8x64().collect();
    /// assert_eq!(lanes.len(), 2);
    /// assert_eq!(lanes[0].to_array()[0], 0u8);
    /// assert_eq!(lanes[1].to_array()[0], 64u8);
    /// ```
    pub fn iter_u8x64(&self) -> impl Iterator<Item = U8x64> + '_ {
        self.data
            .as_chunks::<64>()
            .0
            .iter()
            .map(|chunk| U8x64::from_array(*chunk))
    }

    /// Iterate the column as typed [`F32x16`] values dispatched via
    /// `crate::simd::*` (AVX-512 / NEON / scalar per `cfg`).
    ///
    /// Bytes are decoded little-endian. On LE targets (x86_64, aarch64,
    /// wasm32) the `f32::from_le_bytes` loop optimises to a register-width
    /// load equivalent to a `bytemuck::cast`-style reinterpret, without the
    /// alignment risk of pointer-casting `Arc<[u8]>` (which is `u8`-aligned).
    pub fn iter_f32x16(&self) -> impl Iterator<Item = F32x16> + '_ {
        self.data.as_chunks::<64>().0.iter().map(f32x16_from_chunk)
    }

    /// Iterate the column as typed [`F64x8`] values dispatched via
    /// `crate::simd::*`.
    pub fn iter_f64x8(&self) -> impl Iterator<Item = F64x8> + '_ {
        self.data.as_chunks::<64>().0.iter().map(f64x8_from_chunk)
    }

    /// Iterate the column as typed [`U64x8`] values dispatched via
    /// `crate::simd::*`.
    pub fn iter_u64x8(&self) -> impl Iterator<Item = U64x8> + '_ {
        self.data.as_chunks::<64>().0.iter().map(u64x8_from_chunk)
    }

    /// Iterate the column as typed [`U32x16`] values dispatched via
    /// `crate::simd::*`.
    ///
    /// Bytes are decoded little-endian (`u32::from_le_bytes`), the unsigned
    /// sibling of [`iter_i32x16`](Self::iter_i32x16) — the lane width an
    /// AoS-facet row store needs for classid columns: a 64-byte chunk of a
    /// 512-byte row holds four 16-byte facets, whose `u32` classids sit at
    /// `U32x16` positions 0/4/8/12, so one `U32x16::eq_bitmask` per chunk
    /// answers "which of these four facets carry classid X".
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::simd::MultiLaneColumn;
    /// use std::sync::Arc;
    ///
    /// let mut bytes = vec![0u8; 64];
    /// bytes[0..4].copy_from_slice(&7u32.to_le_bytes());
    /// bytes[60..64].copy_from_slice(&u32::MAX.to_le_bytes());
    /// let col = MultiLaneColumn::new(Arc::from(bytes)).unwrap();
    /// let lane = col.iter_u32x16().next().unwrap();
    /// assert_eq!(lane.to_array()[0], 7);
    /// assert_eq!(lane.to_array()[15], u32::MAX);
    /// ```
    pub fn iter_u32x16(&self) -> impl Iterator<Item = U32x16> + '_ {
        self.data.as_chunks::<64>().0.iter().map(u32x16_from_chunk)
    }

    /// Iterate the column as typed [`I32x16`] values dispatched via
    /// `crate::simd::*`.
    ///
    /// Bytes are decoded little-endian (`i32::from_le_bytes`), the signed
    /// sibling of [`iter_f32x16`](Self::iter_f32x16) — the lane width the
    /// gridlake batch SoA needs for integer min/max/sum tile columns (the
    /// consumer that could previously only view f32 min/max columns).
    pub fn iter_i32x16(&self) -> impl Iterator<Item = I32x16> + '_ {
        self.data.as_chunks::<64>().0.iter().map(i32x16_from_chunk)
    }

    /// Iterate the column as typed [`I64x8`] values dispatched via
    /// `crate::simd::*`.
    ///
    /// Bytes are decoded little-endian (`i64::from_le_bytes`), the signed
    /// sibling of [`iter_u64x8`](Self::iter_u64x8) — the lane width for
    /// 64-bit integer accumulator columns (running sums).
    pub fn iter_i64x8(&self) -> impl Iterator<Item = I64x8> + '_ {
        self.data.as_chunks::<64>().0.iter().map(i64x8_from_chunk)
    }
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
        assert_eq!(col.len_i32x16(), 1);
        assert_eq!(col.len_i64x8(), 1);
    }

    #[test]
    fn new_non_multiple_of_64_errors() {
        assert!(MultiLaneColumn::new(Arc::from(vec![0u8; 100])).is_err());
        assert!(MultiLaneColumn::new(Arc::from(vec![0u8; 63])).is_err());
        assert!(MultiLaneColumn::new(Arc::from(vec![0u8; 65])).is_err());
    }

    #[test]
    fn empty_buffer_yields_zero_lanes() {
        let col = MultiLaneColumn::new(Arc::from(vec![0u8; 0])).unwrap();
        assert!(col.is_empty());
        assert_eq!(col.len_bytes(), 0);
        assert_eq!(col.iter_u8x64().count(), 0);
        assert_eq!(col.iter_f32x16().count(), 0);
        assert_eq!(col.iter_f64x8().count(), 0);
        assert_eq!(col.iter_u64x8().count(), 0);
        assert_eq!(col.iter_i32x16().count(), 0);
        assert_eq!(col.iter_i64x8().count(), 0);
    }

    #[test]
    fn iter_u8x64_two_chunks() {
        let mut v = vec![0u8; 128];
        for i in 0..128 {
            v[i] = i as u8;
        }
        let col = MultiLaneColumn::new(Arc::from(v)).unwrap();
        let lanes: Vec<U8x64> = col.iter_u8x64().collect();
        assert_eq!(lanes.len(), 2);
        let a0 = lanes[0].to_array();
        let a1 = lanes[1].to_array();
        assert_eq!(a0[0], 0u8);
        assert_eq!(a0[63], 63u8);
        assert_eq!(a1[0], 64u8);
        assert_eq!(a1[63], 127u8);
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
    fn iter_u32x16_le_round_trip() {
        // 16 u32 values incl. 0 and u32::MAX so a signed misread would show.
        let src: [u32; 16] = core::array::from_fn(|i| match i {
            0 => 0,
            15 => u32::MAX,
            _ => (i as u32).wrapping_mul(0x9E37_79B9),
        });
        let mut bytes = vec![0u8; 64];
        for (i, &v) in src.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let col = MultiLaneColumn::new(Arc::from(bytes)).unwrap();
        let lane = col.iter_u32x16().next().expect("one lane");
        assert_eq!(lane.to_array(), src);
    }

    /// The facet-scan shape iter_u32x16 exists for: a 64-byte chunk viewed as
    /// four 16-byte facets, classids at u32 positions 0/4/8/12 — one
    /// `eq_bitmask` answers all four at once.
    #[test]
    fn iter_u32x16_supports_the_facet_classid_scan() {
        let mut bytes = vec![0u8; 64];
        // facet 0 and facet 2 carry classid 7; facets 1 and 3 carry 9.
        for (facet, class) in [(0usize, 7u32), (1, 9), (2, 7), (3, 9)] {
            bytes[facet * 16..facet * 16 + 4].copy_from_slice(&class.to_le_bytes());
        }
        let col = MultiLaneColumn::new(Arc::from(bytes)).unwrap();
        let lane = col.iter_u32x16().next().unwrap();
        let needle = U32x16::from_array([7u32; 16]);
        let m = lane.eq_bitmask(needle) & 0x1111; // classid positions only
        assert_eq!(m, (1 << 0) | (1 << 8), "facets 0 and 2 match");
    }

    #[test]
    fn iter_f32x16_le_round_trip() {
        // Build a buffer of 16 f32 values laid out little-endian, then
        // verify iter_f32x16 reads them back in order.
        let src: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.25 - 1.0);
        let mut bytes = vec![0u8; 64];
        for (i, &v) in src.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let col = MultiLaneColumn::new(Arc::from(bytes)).unwrap();
        let lane = col.iter_f32x16().next().expect("one lane");
        assert_eq!(lane.to_array(), src);
    }

    #[test]
    fn iter_f64x8_le_round_trip() {
        let src: [f64; 8] = core::array::from_fn(|i| (i as f64).sin());
        let mut bytes = vec![0u8; 64];
        for (i, &v) in src.iter().enumerate() {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        let col = MultiLaneColumn::new(Arc::from(bytes)).unwrap();
        let lane = col.iter_f64x8().next().expect("one lane");
        assert_eq!(lane.to_array(), src);
    }

    #[test]
    fn iter_u64x8_le_round_trip() {
        let src: [u64; 8] = core::array::from_fn(|i| (i as u64 + 1) * 0x0123_4567_89AB_CDEF);
        let mut bytes = vec![0u8; 64];
        for (i, &v) in src.iter().enumerate() {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        let col = MultiLaneColumn::new(Arc::from(bytes)).unwrap();
        let lane = col.iter_u64x8().next().expect("one lane");
        assert_eq!(lane.to_array(), src);
    }

    #[test]
    fn iter_i32x16_le_round_trip() {
        // Signed values incl. negatives, to prove sign-extension is
        // preserved by the LE decode (the point of the i32 lane).
        let src: [i32; 16] = core::array::from_fn(|i| (i as i32 - 8) * 0x0011_2233);
        let mut bytes = vec![0u8; 64];
        for (i, &v) in src.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let col = MultiLaneColumn::new(Arc::from(bytes)).unwrap();
        let lane = col.iter_i32x16().next().expect("one lane");
        assert_eq!(lane.to_array(), src);
    }

    #[test]
    fn iter_i64x8_le_round_trip() {
        let src: [i64; 8] = core::array::from_fn(|i| (i as i64 - 4) * 0x0123_4567_89AB_CDEF);
        let mut bytes = vec![0u8; 64];
        for (i, &v) in src.iter().enumerate() {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        let col = MultiLaneColumn::new(Arc::from(bytes)).unwrap();
        let lane = col.iter_i64x8().next().expect("one lane");
        assert_eq!(lane.to_array(), src);
    }

    #[test]
    fn typed_iters_yield_three_lanes_over_192_bytes() {
        let v: Vec<u8> = (0u8..192).collect();
        let col = MultiLaneColumn::new(Arc::from(v)).unwrap();
        assert_eq!(col.iter_u8x64().count(), 3);
        assert_eq!(col.iter_f32x16().count(), 3);
        assert_eq!(col.iter_f64x8().count(), 3);
        assert_eq!(col.iter_u64x8().count(), 3);
        assert_eq!(col.iter_i32x16().count(), 3);
        assert_eq!(col.iter_i64x8().count(), 3);
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
}
