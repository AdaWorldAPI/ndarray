//! FACET STRIDE patterns: byte work that reads positions where they sit.
//!
//! `'{0}..{15}' -f $128.split()`: a 128-bit value IS sixteen u8 positions,
//! and the load that brought it in already reads all sixteen. Nothing here
//! moves a byte to read it: no shuffle, no permute, no table lookup
//! assembled from shuffles and masks (CLAUDE.md, FACET STRIDE).
//!
//! This module holds the reusable forms of that shape: [`lut_u8`] for a
//! plain byte table and [`lut9`] for an odd function over a sign-magnitude
//! 9-bit (SM9) lane, u8 magnitude plus sign plane, so callers do not
//! rebuild them. Like [`crate::simd_masking_ops`] it names no ISA and owns
//! no backend; it is slice ergonomics over the position view.

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

// ────────────────────────────────────────────────────────────────────────
// lut9: an odd-function table over a sign-magnitude 9-bit (SM9) lane.
//
// A sign-magnitude 9-bit lane stores each value as a `u8` magnitude plus one
// bit in a `u64` sign plane per 64 rows; whether the row holds a value at all
// is a separate presence plane, and the writer stores `+0` (magnitude 0, sign
// clear) for zero and for missing rows. For an odd function
// (`f(-x) = -f(x)`, e.g. tanh, sin, x³, sign-preserving clamps) one table
// over the 256 magnitudes covers both signs: the magnitude is looked up and
// the sign plane passes through. The presence plane is not an input — a
// missing row is magnitude 0, `f(0) = 0`, so it stays `+0`.
// ────────────────────────────────────────────────────────────────────────

/// Applies an odd function, given as a 256-entry magnitude table, to a
/// sign-magnitude lane.
///
/// For every row `i < magnitudes.len()`:
///
/// * `out_magnitudes[i] = table[magnitudes[i]]`;
/// * the output sign bit is the input sign bit, cleared where a nonzero
///   input gives a zero result. A negative value whose result rounds to
///   zero is stored as `+0`, never `-0`.
/// * an input `-0` (magnitude 0, sign set) comes out as `-0`. A writer
///   that only stores `+0` never produces it, so under the plain numeric
///   reading this changes nothing. A class whose reading gives `-0` a
///   meaning of its own, such as "observed, value NaN" beside a clear
///   presence bit for "not observed", keeps that meaning through the
///   function, the way IEEE arithmetic carries NaN through. Clearing it
///   would turn that reading's NaN into the value 0.
///
/// Rows past `magnitudes.len()` are not written in `out_magnitudes`. In
/// `out_signs` the bits past `magnitudes.len()` and any surplus words are
/// written as `0`. The presence plane is not read or written: because
/// `table[0]` must be `0`, a missing row (stored as `+0`) stays `+0`.
///
/// Facet stride: every 16 magnitudes are one 128-bit value whose positions
/// `{0}..{15}` are read where they sit, and each position addresses the table
/// directly. No byte is shuffled. The sign plane is 64 rows per
/// [`crate::simd::U8x64::cmpeq_mask`]. The final partial chunk is padded with
/// magnitude 0 and goes through the same path, with the padding lanes masked
/// off the sign plane.
///
/// # Panics
///
/// * if `table[0] != 0`, since an odd function maps 0 to 0;
/// * if `out_magnitudes.len() < magnitudes.len()`;
/// * if `signs.len()` or `out_signs.len()` is less than
///   `magnitudes.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::lut9;
///
/// // f(m) = m / 2, an odd function on the signed value.
/// let mut table = [0u8; 256];
/// for (m, t) in table.iter_mut().enumerate() {
///     *t = (m / 2) as u8;
/// }
/// // Values: +10, -10, -1, +0
/// let magnitudes = [10u8, 10, 1, 0];
/// let signs = [0b0110u64];
/// let mut out = [0u8; 4];
/// let mut out_signs = [0u64; 1];
/// lut9(&magnitudes, &signs, &table, &mut out, &mut out_signs);
/// assert_eq!(out, [5, 5, 0, 0]);
/// // -1 / 2 rounds to zero and is stored as +0: only row 1 keeps its sign.
/// assert_eq!(out_signs[0], 0b0010);
/// ```
pub fn lut9(magnitudes: &[u8], signs: &[u64], table: &[u8; 256], out_magnitudes: &mut [u8], out_signs: &mut [u64]) {
    use crate::simd::U8x64;
    let n = magnitudes.len();
    let words = n.div_ceil(64);
    assert_eq!(table[0], 0, "lut9: an odd function maps 0 to 0, so table[0] must be 0");
    assert!(out_magnitudes.len() >= n, "lut9: out_magnitudes holds {} rows, need {n}", out_magnitudes.len());
    assert!(signs.len() >= words, "lut9: signs holds {} words, need {words}", signs.len());
    assert!(out_signs.len() >= words, "lut9: out_signs holds {} words, need {words}", out_signs.len());

    // FACET STRIDE: the magnitude lookup reads positions where they sit
    // ([`lut_u8`]); no byte is shuffled.
    let fill = |chunk: &[u8; 64], out: &mut [u8; 64]| lut_u8(chunk, table, out);
    let zero = U8x64::splat(0);
    let (in_chunks, tail) = magnitudes.as_chunks::<64>();
    let (out_chunks, out_tail) = out_magnitudes[..n].as_chunks_mut::<64>();
    for (w, (chunk, out)) in in_chunks.iter().zip(out_chunks).enumerate() {
        fill(chunk, out);
        let (m, r) = (U8x64::from_array(*chunk), U8x64::from_array(*out));
        // Keep the sign where the result is nonzero or the input is `-0`.
        out_signs[w] = signs[w] & (!r.cmpeq_mask(zero) | m.cmpeq_mask(zero));
    }
    let mut w = in_chunks.len();
    if !tail.is_empty() {
        let mut padded = [0u8; 64];
        padded[..tail.len()].copy_from_slice(tail);
        let mut looked_up = [0u8; 64];
        fill(&padded, &mut looked_up);
        out_tail.copy_from_slice(&looked_up[..tail.len()]);
        let (m, r) = (U8x64::from_array(padded), U8x64::from_array(looked_up));
        // Padding lanes look like input `-0` (magnitude 0), so the keep-`-0`
        // rule would pass their junk sign bits; mask them off.
        out_signs[w] =
            signs[w] & (!r.cmpeq_mask(zero) | m.cmpeq_mask(zero)) & crate::simd_masking_ops::live64(tail.len());
        w += 1;
    }
    for s in &mut out_signs[w..] {
        *s = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── lut9 (SM9 odd table) ──────────────────────────────────────────────────────────

    fn lut9_reference(mags: &[u8], signs: &[u64], table: &[u8; 256]) -> (Vec<u8>, Vec<u64>) {
        let out: Vec<u8> = mags.iter().map(|&m| table[m as usize]).collect();
        let mut out_signs = vec![0u64; mags.len().div_ceil(64)];
        for (i, &o) in out.iter().enumerate() {
            let neg = (signs[i / 64] >> (i % 64)) & 1 == 1;
            if neg && (o != 0 || mags[i] == 0) {
                out_signs[i / 64] |= 1 << (i % 64);
            }
        }
        (out, out_signs)
    }

    fn lut9_xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    /// FAILS IF: a quarter of the table is picked by the wrong magnitude bits,
    /// a sign bit survives on a zero result, the final partial chunk writes
    /// rows past the end, or bits/words past the last row are left set.
    #[test]
    fn lut9_matches_the_table_at_every_lane_and_length() {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        // Random table with table[0] = 0 and some zero entries elsewhere, so
        // the sign-clearing rule is exercised away from magnitude 0 as well.
        let mut table = [0u8; 256];
        for t in table.iter_mut().skip(1) {
            let r = lut9_xorshift(&mut s) as u8;
            *t = if r.is_multiple_of(7) { 0 } else { r };
        }
        for &n in &[0usize, 1, 63, 64, 65, 127, 128, 129, 255, 256, 300, 1024 + 17] {
            for shift in [0usize, 1, 37, 101] {
                // Every magnitude appears at every lane position across shifts.
                let mags: Vec<u8> = (0..n).map(|i| ((i * 37 + shift) % 256) as u8).collect();
                let words = n.div_ceil(64);
                let mut signs: Vec<u64> = (0..words).map(|_| lut9_xorshift(&mut s)).collect();
                // Junk sign bits past the last row must not leak into the output.
                if let Some(last) = signs.last_mut() {
                    *last |= !crate::simd_masking_ops::live64(n - (words - 1) * 64);
                }
                let (want, want_signs) = lut9_reference(&mags, &signs, &table);
                let mut out = vec![0xAAu8; n + 5];
                let mut out_signs = vec![u64::MAX; words + 2];
                lut9(&mags, &signs, &table, &mut out, &mut out_signs);
                assert_eq!(&out[..n], &want[..], "n={n} shift={shift}");
                assert!(out[n..].iter().all(|&b| b == 0xAA), "rows past the end written, n={n}");
                assert_eq!(&out_signs[..words], &want_signs[..], "n={n} shift={shift}");
                assert!(out_signs[words..].iter().all(|&w| w == 0), "surplus words, n={n}");
            }
        }
    }

    /// FAILS IF: a negative value whose result rounds to zero keeps its sign
    /// (a stored `-0`), or an input `-0` loses its sign. Silence half: a
    /// nonzero negative result keeps its sign, and `+0` stays `+0`.
    #[test]
    fn lut9_rounds_to_plus_zero_and_keeps_an_input_minus_zero() {
        let mut table = [0u8; 256];
        for (m, t) in table.iter_mut().enumerate() {
            *t = (m / 4) as u8;
        }
        // -0, -3 (rounds to 0), -4 (-> -1), +3, +200, +0
        let mags = [0u8, 3, 4, 3, 200, 0];
        let signs = [0b000111u64];
        let mut out = [0u8; 6];
        let mut out_signs = [u64::MAX; 1];
        lut9(&mags, &signs, &table, &mut out, &mut out_signs);
        assert_eq!(out, [0, 0, 1, 0, 50, 0]);
        assert_eq!(out_signs[0], 0b000101);
    }

    #[test]
    #[should_panic(expected = "table[0] must be 0")]
    fn lut9_refuses_a_table_that_moves_zero() {
        let mut table = [0u8; 256];
        table[0] = 1;
        lut9(&[0], &[0], &table, &mut [0], &mut [0]);
    }

    #[test]
    #[should_panic(expected = "signs holds 1 words, need 2")]
    fn lut9_refuses_a_short_sign_plane() {
        let table = [0u8; 256];
        lut9(&[0; 65], &[0], &table, &mut [0; 65], &mut [0; 2]);
    }

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
