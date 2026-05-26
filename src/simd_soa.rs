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
use crate::simd::{F32x16, F64x8, U64x8, U8x64};

// The SoA layout descriptor. `SoaColumns` binds one of these to a shared
// backing buffer; the descriptor stays the single source of truth in
// `crate::hpc::soa` and is composed here without modification.
use crate::hpc::soa::SoaContainerHeader;

// `CausalEdge64` is a `#[repr(transparent)]` newtype over `u64`. The
// baked-in edge-column accessor below reinterprets a `u64`-stride column's
// little-endian cells as `CausalEdge64` — a pure layout reinterpret (no
// distance / semantic operation), consistent with this module's
// layout-only rule.
use crate::hpc::causal_diff::CausalEdge64;

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
}

// ════════════════════════════════════════════════════════════════════
// SoaColumns — N-field SoA carrier over one shared backing (MailboxSoA shape)
// ════════════════════════════════════════════════════════════════════

/// Multi-column SoA lane carrier — the "richer MailboxSoA shape".
///
/// Binds a [`SoaContainerHeader<N>`] descriptor to a single shared
/// `Arc<[u8]>` backing and exposes **per-field, zero-copy** typed-lane
/// iterators (the same lane shapes [`MultiLaneColumn`] offers, one set per
/// field) plus a baked-in [`CausalEdge64`] edge-column accessor.
///
/// The `N` columns are the OGIT *custom fields* (embedding codebooks / role
/// catalogues / qualia / meta / edge). Which columns are "in focus" for a
/// given sweep is **not** decided here — that masking is the attention
/// layer's job (the sparse-rename / `AttentionMask` register file that reads
/// these columns). This carrier is layout-only: it lays the columns out and
/// hands back typed lanes; it does not compute distance, attention, or
/// semantics.
///
/// [`MultiLaneColumn`] is OGIT-inherited and frozen (additive-only); this is
/// the additive multi-column sibling that *composes* the descriptor rather
/// than modifying either type.
///
/// # Invariant
///
/// Each field column spans `row_capacity * elem_stride[i]` bytes, which MUST
/// be a multiple of 64 (the AVX-512 register / cache-line width) so per-field
/// lane iteration covers the column with no partial tail. Because the header
/// packs columns contiguously from offset 0, this single invariant also makes
/// every `field_offsets[i]` 64-byte aligned. [`SoaColumns::new`] returns
/// `Err` otherwise.
///
/// # Examples
///
/// ```
/// use ndarray::simd::SoaColumns;
/// use ndarray::hpc::soa::SoaContainerHeader;
/// use std::sync::Arc;
///
/// // 2 fields, 16 rows: field 0 = 16×4 = 64 B (f32 lane), field 1 = 16×8
/// // = 128 B (u64 / CausalEdge64 lane). Both 64-multiples.
/// let hdr: SoaContainerHeader<2> = SoaContainerHeader::new(16, [4, 8]);
/// let data: Arc<[u8]> = Arc::from(vec![0u8; hdr.backing_buffer_bytes()]);
/// let cols = SoaColumns::new(hdr, data).unwrap();
/// assert_eq!(cols.field_count(), 2);
/// assert_eq!(cols.iter_field_f32x16(0).count(), 1);   // 64 B / 64 = 1 lane
/// assert_eq!(cols.iter_field_causaledge64(1).count(), 16); // 128 B / 8 = 16 edges
/// ```
#[derive(Clone)]
pub struct SoaColumns<const N: usize> {
    header: SoaContainerHeader<N>,
    data: Arc<[u8]>,
}

impl<const N: usize> SoaColumns<N> {
    /// Bind a header to a shared backing buffer.
    ///
    /// Returns `Err` if the header fails [`SoaContainerHeader::validate`],
    /// the backing is shorter than
    /// [`backing_buffer_bytes()`](SoaContainerHeader::backing_buffer_bytes),
    /// or any field column (`row_capacity * elem_stride[i]`) is not a
    /// multiple of 64.
    pub fn new(header: SoaContainerHeader<N>, data: Arc<[u8]>) -> Result<Self, &'static str> {
        header.validate()?;
        if data.len() < header.backing_buffer_bytes() {
            return Err("SoaColumns: backing buffer shorter than header.backing_buffer_bytes()");
        }
        for i in 0..N {
            let col_bytes = (header.row_capacity as usize) * (header.elem_stride[i] as usize);
            if !col_bytes.is_multiple_of(64) {
                return Err("SoaColumns: field column (row_capacity * elem_stride[i]) is not a multiple of 64");
            }
        }
        Ok(Self { header, data })
    }

    /// The bound layout descriptor.
    pub fn header(&self) -> &SoaContainerHeader<N> {
        &self.header
    }

    /// Number of fields (columns) — always `N`.
    pub fn field_count(&self) -> usize {
        N
    }

    /// Current logical row count (from the header; `≤ row_capacity`).
    pub fn row_count(&self) -> u32 {
        self.header.row_count
    }

    /// Maximum row capacity (the per-column byte extent is `row_capacity *
    /// elem_stride[field]`).
    pub fn row_capacity(&self) -> u32 {
        self.header.row_capacity
    }

    /// The full backing buffer as bytes (aliases the `Arc`, never copies).
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Byte range `[start, end)` of field `field`'s column in the backing.
    ///
    /// # Panics
    ///
    /// Panics if `field >= N`.
    pub fn column_byte_range(&self, field: usize) -> (usize, usize) {
        assert!(field < N, "SoaColumns: field {field} out of range (N={N})");
        let start = self.header.field_offsets[field] as usize;
        let len = (self.header.row_capacity as usize) * (self.header.elem_stride[field] as usize);
        (start, start + len)
    }

    /// Field `field`'s column as a byte slice (aliases the `Arc`, never copies).
    ///
    /// # Panics
    ///
    /// Panics if `field >= N`.
    #[inline]
    fn column_bytes(&self, field: usize) -> &[u8] {
        let (start, end) = self.column_byte_range(field);
        &self.data[start..end]
    }

    /// Iterate field `field` as [`U8x64`] lanes (dispatched via `crate::simd::*`).
    ///
    /// # Panics
    ///
    /// Panics if `field >= N`.
    pub fn iter_field_u8x64(&self, field: usize) -> impl Iterator<Item = U8x64> + '_ {
        self.column_bytes(field)
            .as_chunks::<64>()
            .0
            .iter()
            .map(|chunk| U8x64::from_array(*chunk))
    }

    /// Iterate field `field` as [`F32x16`] lanes (little-endian decode).
    ///
    /// # Panics
    ///
    /// Panics if `field >= N`.
    pub fn iter_field_f32x16(&self, field: usize) -> impl Iterator<Item = F32x16> + '_ {
        self.column_bytes(field)
            .as_chunks::<64>()
            .0
            .iter()
            .map(f32x16_from_chunk)
    }

    /// Iterate field `field` as [`F64x8`] lanes (little-endian decode).
    ///
    /// # Panics
    ///
    /// Panics if `field >= N`.
    pub fn iter_field_f64x8(&self, field: usize) -> impl Iterator<Item = F64x8> + '_ {
        self.column_bytes(field)
            .as_chunks::<64>()
            .0
            .iter()
            .map(f64x8_from_chunk)
    }

    /// Iterate field `field` as [`U64x8`] lanes (little-endian decode).
    ///
    /// # Panics
    ///
    /// Panics if `field >= N`.
    pub fn iter_field_u64x8(&self, field: usize) -> impl Iterator<Item = U64x8> + '_ {
        self.column_bytes(field)
            .as_chunks::<64>()
            .0
            .iter()
            .map(u64x8_from_chunk)
    }

    /// Iterate the **baked-in edge column**: field `field`'s little-endian
    /// `u64` cells reinterpreted as [`CausalEdge64`] (one per 8 bytes).
    ///
    /// Intended for a `u64`-stride field (`elem_stride[field] == 8`) — the
    /// MailboxSoA `EdgeColumn`. The reinterpret is pure layout (`CausalEdge64`
    /// is `#[repr(transparent)]` over `u64`); no edge semantics are computed
    /// here.
    ///
    /// # Panics
    ///
    /// Panics if `field >= N`.
    pub fn iter_field_causaledge64(&self, field: usize) -> impl Iterator<Item = CausalEdge64> + '_ {
        self.column_bytes(field)
            .as_chunks::<8>()
            .0
            .iter()
            .map(|cell| CausalEdge64(u64::from_le_bytes(*cell)))
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
    fn typed_iters_yield_three_lanes_over_192_bytes() {
        let v: Vec<u8> = (0u8..192).collect();
        let col = MultiLaneColumn::new(Arc::from(v)).unwrap();
        assert_eq!(col.iter_u8x64().count(), 3);
        assert_eq!(col.iter_f32x16().count(), 3);
        assert_eq!(col.iter_f64x8().count(), 3);
        assert_eq!(col.iter_u64x8().count(), 3);
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

    // ---- SoaColumns ----

    use crate::hpc::soa::SoaContainerHeader;

    // Helper: a valid 2-field header (field 0 = 16×4 = 64 B f32 column,
    // field 1 = 16×8 = 128 B u64/CausalEdge64 column; both 64-multiples).
    fn valid_hdr_2() -> SoaContainerHeader<2> {
        SoaContainerHeader::new(16, [4, 8])
    }

    #[test]
    fn soa_columns_new_valid() {
        let hdr = valid_hdr_2();
        let data: Arc<[u8]> = Arc::from(vec![0u8; hdr.backing_buffer_bytes()]);
        let cols = SoaColumns::new(hdr, data).unwrap();
        assert_eq!(cols.field_count(), 2);
        assert_eq!(cols.row_capacity(), 16);
        assert_eq!(cols.row_count(), 0);
    }

    #[test]
    fn soa_columns_rejects_non_64_multiple_column() {
        // 10 rows × 4 bytes = 40 bytes — not a multiple of 64.
        let hdr: SoaContainerHeader<1> = SoaContainerHeader::new(10, [4]);
        let data: Arc<[u8]> = Arc::from(vec![0u8; hdr.backing_buffer_bytes()]);
        assert!(SoaColumns::new(hdr, data).is_err());
    }

    #[test]
    fn soa_columns_rejects_short_backing() {
        let hdr = valid_hdr_2();
        // One byte short of the required backing.
        let data: Arc<[u8]> = Arc::from(vec![0u8; hdr.backing_buffer_bytes() - 1]);
        assert!(SoaColumns::new(hdr, data).is_err());
    }

    #[test]
    fn soa_columns_column_byte_range() {
        let hdr = valid_hdr_2();
        let data: Arc<[u8]> = Arc::from(vec![0u8; hdr.backing_buffer_bytes()]);
        let cols = SoaColumns::new(hdr, data).unwrap();
        assert_eq!(cols.column_byte_range(0), (0, 64));
        assert_eq!(cols.column_byte_range(1), (64, 192));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn soa_columns_field_oob_panics() {
        let hdr = valid_hdr_2();
        let data: Arc<[u8]> = Arc::from(vec![0u8; hdr.backing_buffer_bytes()]);
        let cols = SoaColumns::new(hdr, data).unwrap();
        let _ = cols.column_byte_range(2);
    }

    #[test]
    fn soa_columns_iter_f32x16_field_round_trip() {
        let hdr = valid_hdr_2();
        let mut bytes = vec![0u8; hdr.backing_buffer_bytes()];
        // Field 0 occupies bytes [0, 64) = 16 f32 LE.
        let src: [f32; 16] = core::array::from_fn(|i| i as f32 * 0.5 - 2.0);
        for (i, &v) in src.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let cols = SoaColumns::new(hdr, Arc::from(bytes)).unwrap();
        let lane = cols.iter_field_f32x16(0).next().expect("one f32 lane");
        assert_eq!(lane.to_array(), src);
        assert_eq!(cols.iter_field_f32x16(0).count(), 1);
    }

    #[test]
    fn soa_columns_iter_causaledge64_field_round_trip() {
        let hdr = valid_hdr_2();
        let mut bytes = vec![0u8; hdr.backing_buffer_bytes()];
        // Field 1 (the edge column) occupies bytes [64, 192) = 16 u64 LE.
        let (start, _end) = (64usize, 192usize);
        let src: [u64; 16] = core::array::from_fn(|i| (i as u64 + 1).wrapping_mul(0x0123_4567_89AB_CDEF));
        for (i, &v) in src.iter().enumerate() {
            bytes[start + i * 8..start + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        let cols = SoaColumns::new(hdr, Arc::from(bytes)).unwrap();
        let edges: Vec<CausalEdge64> = cols.iter_field_causaledge64(1).collect();
        assert_eq!(edges.len(), 16);
        for (i, &v) in src.iter().enumerate() {
            assert_eq!(edges[i], CausalEdge64(v));
        }
    }

    #[test]
    fn soa_columns_clone_is_o1_indivisible_carry() {
        // The core SoA property: a whole cycle's column block (CausalEdge64
        // + all other fields) carries to the next cycle in O(1) — clone is an
        // Arc refcount bump sharing the same backing, never a byte copy.
        let hdr = valid_hdr_2();
        let data: Arc<[u8]> = Arc::from(vec![7u8; hdr.backing_buffer_bytes()]);
        let cycle_n = SoaColumns::new(hdr, data).unwrap();
        let cycle_next = cycle_n.clone();
        assert_eq!(
            cycle_n.as_bytes().as_ptr(),
            cycle_next.as_bytes().as_ptr(),
            "clone must carry the same Arc backing to the next cycle, not copy"
        );
    }

    #[test]
    fn soa_columns_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SoaColumns<4>>();
    }
}
