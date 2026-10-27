//! FACET STRIDE patterns: byte work that reads positions where they sit.
//!
//! `'{0}..{15}' -f $128.split()`: a 128-bit value IS sixteen u8 positions,
//! and the load that brought it in already reads all sixteen. Nothing here
//! moves a byte to read it: no shuffle, no permute, no table lookup
//! assembled from shuffles and masks (CLAUDE.md, FACET STRIDE).
//!
//! This module holds the reusable form of that shape so callers do not
//! rebuild it. Like [`crate::simd_masking_ops`] it names no ISA and owns no
//! backend; it is slice ergonomics over the position view.

/// Bytes per facet: one 128-bit value read as sixteen u8 positions.
pub const FACET_BYTES: usize = 16;

/// Maps every byte through a 256-entry table: `out[i] = table[values[i]]`.
///
/// Each run of [`FACET_BYTES`] values is one 128-bit facet; every position
/// addresses the table directly, so the lookup costs one table read per
/// byte and no byte is moved to compute it. This is the shape to reach for
/// instead of a byte shuffle used as a lookup.
///
/// `out` may be longer than `values`; bytes past `values.len()` are left
/// untouched.
///
/// # Panics
///
/// When `out` is shorter than `values`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::lut_u8;
///
/// let mut table = [0u8; 256];
/// for (i, t) in table.iter_mut().enumerate() {
///     *t = (255 - i) as u8;
/// }
/// let mut out = [0u8; 3];
/// lut_u8(&[0, 1, 255], &table, &mut out);
/// assert_eq!(out, [255, 254, 0]);
/// ```
pub fn lut_u8(values: &[u8], table: &[u8; 256], out: &mut [u8]) {
    let n = values.len();
    assert!(out.len() >= n, "lut_u8: out holds {} bytes, need {n}", out.len());
    let (facets, tail) = values.as_chunks::<FACET_BYTES>();
    let (out_facets, out_tail) = out[..n].as_chunks_mut::<FACET_BYTES>();
    for (positions, o) in facets.iter().zip(out_facets) {
        *o = positions.map(|v| table[usize::from(v)]);
    }
    for (&v, o) in tail.iter().zip(out_tail) {
        *o = table[usize::from(v)];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> [u8; 256] {
        // A non-monotone, non-affine table, so a position read at the wrong
        // index cannot land on the right value by accident.
        core::array::from_fn(|i| (i.wrapping_mul(73) ^ 0x5A) as u8)
    }

    #[test]
    fn lut_u8_matches_a_per_byte_lookup_at_every_length() {
        let t = table();
        for n in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 300] {
            let values: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            let mut out = vec![0xC3u8; n + 5];
            lut_u8(&values, &t, &mut out);
            for i in 0..n {
                assert_eq!(out[i], t[usize::from(values[i])], "n={n} i={i}");
            }
            assert!(out[n..].iter().all(|&b| b == 0xC3), "n={n}: wrote past values.len()");
        }
    }

    #[test]
    fn lut_u8_reads_every_table_entry() {
        let t = table();
        let values: Vec<u8> = (0..=255).collect();
        let mut out = vec![0u8; 256];
        lut_u8(&values, &t, &mut out);
        assert_eq!(out[..], t[..]);
    }

    #[test]
    #[should_panic(expected = "lut_u8: out holds")]
    fn lut_u8_rejects_a_short_output() {
        lut_u8(&[1, 2, 3], &table(), &mut [0u8; 2]);
    }
}
