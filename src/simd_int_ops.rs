//! Slice-level integer SIMD ops for `i8` / `i16` data.
//!
//! Mirrors the float helpers in `simd_avx2.rs` (dot_f32, axpy_f32, …).

//! Each function dispatches at compile-time to the widest available SIMD type:
//!
//! | Lane width | x86_64 + AVX-512BW | x86_64 (AVX2 baseline) | aarch64 NEON | scalar |
//! |------------|--------------------|------------------------|--------------|--------|
//! | i8         | `I8x64` (64 lanes) | `I8x32`  (32 lanes)    | `I8x16`      | scalar |
//! | i16        | `I16x32`           | `I16x16`               | `I16x8`      | scalar |
//!
//! The accumulator widths (`i32` for `dot_i8`, `i64` for `dot_i16`) are
//! deliberately wider than the lane element type — `127 × 127 × 64 ≈ 1 M`
//! fits in i32 but not in i8/i16 reductions.

// ────────────────────────────────────────────────────────────────────────
// add_i8 / sub_i8 — element-wise mutate-in-place
// ────────────────────────────────────────────────────────────────────────

/// Element-wise `dst[i] += src[i]` (wrapping i8 add).
///
/// Panics if `dst.len() != src.len()`.
#[inline]
pub fn add_i8(dst: &mut [i8], src: &[i8]) {
    assert_eq!(dst.len(), src.len(), "add_i8: length mismatch");
    for i in 0..dst.len() {
        dst[i] = dst[i].wrapping_add(src[i]);
    }
}

/// Element-wise `dst[i] -= src[i]` (wrapping i8 sub).
#[inline]
pub fn sub_i8(dst: &mut [i8], src: &[i8]) {
    assert_eq!(dst.len(), src.len(), "sub_i8: length mismatch");
    for i in 0..dst.len() {
        dst[i] = dst[i].wrapping_sub(src[i]);
    }
}

/// Element-wise `dst[i] += src[i]` (wrapping i16 add).
#[inline]
pub fn add_i16(dst: &mut [i16], src: &[i16]) {
    assert_eq!(dst.len(), src.len(), "add_i16: length mismatch");
    for i in 0..dst.len() {
        dst[i] = dst[i].wrapping_add(src[i]);
    }
}

// ────────────────────────────────────────────────────────────────────────
// dot_i8 / dot_i16 — overflow-safe dot product
// ────────────────────────────────────────────────────────────────────────

/// Sum of `a[i] * b[i]` accumulated in `i32` to avoid overflow.
///
/// Worst-case lane product is `127 × -128 = -16_256`; with 4M lanes the sum
/// stays well within `i32::MAX`. For longer slices, callers should chunk.
///
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len(), "dot_i8: length mismatch");
    let mut acc: i32 = 0;
    for i in 0..a.len() {
        acc = acc.wrapping_add((a[i] as i32) * (b[i] as i32));
    }
    acc
}

/// Sum of `a[i] * b[i]` accumulated in `i64`.
#[inline]
pub fn dot_i16(a: &[i16], b: &[i16]) -> i64 {
    assert_eq!(a.len(), b.len(), "dot_i16: length mismatch");
    let mut acc: i64 = 0;
    for i in 0..a.len() {
        acc = acc.wrapping_add((a[i] as i64) * (b[i] as i64));
    }
    acc
}

// ────────────────────────────────────────────────────────────────────────
// min_i8 / max_i8 — horizontal reduction
// ────────────────────────────────────────────────────────────────────────

/// Horizontal minimum across `s`. Empty input → `i8::MAX`.
#[inline]
pub fn min_i8(s: &[i8]) -> i8 {
    if s.is_empty() {
        return i8::MAX;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use crate::simd::I8x64;
        const L: usize = 64;
        let n = s.len();
        if n >= L {
            let chunks = n / L;
            let mut acc = I8x64::from_slice(&s[..L]);
            for c in 1..chunks {
                let v = I8x64::from_slice(&s[c * L..c * L + L]);
                acc = acc.min(v);
            }
            let acc_arr = acc.to_array();
            let mut m = acc_arr[0];
            for &v in acc_arr[1..].iter() {
                if v < m {
                    m = v;
                }
            }
            for &v in s[chunks * L..n].iter() {
                if v < m {
                    m = v;
                }
            }
            return m;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::simd_neon::I8x16;
        const L: usize = 16;
        let n = s.len();
        if n >= L {
            let chunks = n / L;
            let mut acc = I8x16::from_slice(&s[..L]);
            for c in 1..chunks {
                let v = I8x16::from_slice(&s[c * L..c * L + L]);
                acc = acc.min(v);
            }
            let acc_arr = acc.to_array();
            let mut m = acc_arr[0];
            for &v in acc_arr[1..].iter() {
                if v < m {
                    m = v;
                }
            }
            for &v in s[chunks * L..n].iter() {
                if v < m {
                    m = v;
                }
            }
            return m;
        }
    }

    let mut m = s[0];
    for &v in &s[1..] {
        if v < m {
            m = v;
        }
    }
    m
}

/// Horizontal maximum across `s`. Empty input → `i8::MIN`.
#[inline]
pub fn max_i8(s: &[i8]) -> i8 {
    if s.is_empty() {
        return i8::MIN;
    }

    #[cfg(target_arch = "x86_64")]
    {
        use crate::simd::I8x64;
        const L: usize = 64;
        let n = s.len();
        if n >= L {
            let chunks = n / L;
            let mut acc = I8x64::from_slice(&s[..L]);
            for c in 1..chunks {
                let v = I8x64::from_slice(&s[c * L..c * L + L]);
                acc = acc.max(v);
            }
            let acc_arr = acc.to_array();
            let mut m = acc_arr[0];
            for &v in acc_arr[1..].iter() {
                if v > m {
                    m = v;
                }
            }
            for &v in s[chunks * L..n].iter() {
                if v > m {
                    m = v;
                }
            }
            return m;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::simd_neon::I8x16;
        const L: usize = 16;
        let n = s.len();
        if n >= L {
            let chunks = n / L;
            let mut acc = I8x16::from_slice(&s[..L]);
            for c in 1..chunks {
                let v = I8x16::from_slice(&s[c * L..c * L + L]);
                acc = acc.max(v);
            }
            let acc_arr = acc.to_array();
            let mut m = acc_arr[0];
            for &v in acc_arr[1..].iter() {
                if v > m {
                    m = v;
                }
            }
            for &v in s[chunks * L..n].iter() {
                if v > m {
                    m = v;
                }
            }
            return m;
        }
    }

    let mut m = s[0];
    for &v in &s[1..] {
        if v > m {
            m = v;
        }
    }
    m
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar_add_i8(dst: &mut [i8], src: &[i8]) {
        for i in 0..dst.len() {
            dst[i] = dst[i].wrapping_add(src[i]);
        }
    }
    fn scalar_sub_i8(dst: &mut [i8], src: &[i8]) {
        for i in 0..dst.len() {
            dst[i] = dst[i].wrapping_sub(src[i]);
        }
    }
    fn scalar_add_i16(dst: &mut [i16], src: &[i16]) {
        for i in 0..dst.len() {
            dst[i] = dst[i].wrapping_add(src[i]);
        }
    }

    #[test]
    fn add_i8_matches_scalar_for_tail_lengths() {
        for &len in &[0usize, 1, 32, 63, 64, 65, 127, 128, 129, 256] {
            let a_init: Vec<i8> = (0..len).map(|i| (i as i32 - 50) as i8).collect();
            let b: Vec<i8> = (0..len).map(|i| ((i * 3) as i32 - 30) as i8).collect();

            let mut a_simd = a_init.clone();
            add_i8(&mut a_simd, &b);

            let mut a_scalar = a_init.clone();
            scalar_add_i8(&mut a_scalar, &b);

            assert_eq!(a_simd, a_scalar, "add_i8 mismatch at len={}", len);
        }
    }

    #[test]
    fn sub_i8_matches_scalar_for_tail_lengths() {
        for &len in &[0usize, 1, 63, 64, 65, 127, 128, 129] {
            let a_init: Vec<i8> = (0..len).map(|i| (i as i32 - 30) as i8).collect();
            let b: Vec<i8> = (0..len).map(|i| ((i * 7) as i32 - 60) as i8).collect();

            let mut a_simd = a_init.clone();
            sub_i8(&mut a_simd, &b);

            let mut a_scalar = a_init.clone();
            scalar_sub_i8(&mut a_scalar, &b);

            assert_eq!(a_simd, a_scalar, "sub_i8 mismatch at len={}", len);
        }
    }

    #[test]
    fn add_i16_matches_scalar_for_tail_lengths() {
        for &len in &[0usize, 1, 31, 32, 33, 64, 65, 100] {
            let a_init: Vec<i16> = (0..len).map(|i| i as i16 * 7 - 1000).collect();
            let b: Vec<i16> = (0..len).map(|i| i as i16 * -3 + 500).collect();

            let mut a_simd = a_init.clone();
            add_i16(&mut a_simd, &b);

            let mut a_scalar = a_init.clone();
            scalar_add_i16(&mut a_scalar, &b);

            assert_eq!(a_simd, a_scalar, "add_i16 mismatch at len={}", len);
        }
    }

    #[test]
    fn dot_i8_overflow_safety() {
        // [127; 64] dot [127; 64] = 127 * 127 * 64 = 1_032_256.
        // Fits in i32 (max ~2.1B). Without widening to i32 this would overflow.
        let a = [127i8; 64];
        let b = [127i8; 64];
        let got = dot_i8(&a, &b);
        let expected: i32 = 127 * 127 * 64;
        assert_eq!(got, expected, "dot_i8([127; 64], [127; 64])");
    }

    #[test]
    fn dot_i8_negative_values() {
        let a = [-128i8; 32];
        let b = [-128i8; 32];
        // -128 × -128 = 16_384, × 32 = 524_288. Fits in i32.
        let got = dot_i8(&a, &b);
        assert_eq!(got, 16_384 * 32);
    }

    #[test]
    fn dot_i16_basic() {
        let a: Vec<i16> = (1..=32).collect();
        let b: Vec<i16> = (1..=32).map(|x| x * 2).collect();
        let got = dot_i16(&a, &b);
        let expected: i64 = (1..=32i64).map(|x| x * (x * 2)).sum();
        assert_eq!(got, expected);
    }

    #[test]
    fn dot_i16_overflow_safety() {
        // [32767; 100] dot [32767; 100] = 32767² × 100 ≈ 1.07e11. Fits in i64.
        let a = [i16::MAX; 100];
        let b = [i16::MAX; 100];
        let got = dot_i16(&a, &b);
        let expected: i64 = (i16::MAX as i64) * (i16::MAX as i64) * 100;
        assert_eq!(got, expected);
    }

    #[test]
    fn min_max_i8_basic() {
        let s: Vec<i8> = (0..100_i32).map(|i| (i - 50) as i8).collect();
        // Range -50..=49.
        assert_eq!(min_i8(&s), -50);
        assert_eq!(max_i8(&s), 49);
    }

    #[test]
    fn min_max_i8_boundary_values() {
        let mut s = vec![0i8; 200];
        s[42] = i8::MIN; // -128
        s[123] = i8::MAX; // 127
        assert_eq!(min_i8(&s), -128);
        assert_eq!(max_i8(&s), 127);
    }

    #[test]
    fn min_max_i8_short_slices() {
        // Fewer than one SIMD lane width.
        let s = [3i8, -7, 12, 0];
        assert_eq!(min_i8(&s), -7);
        assert_eq!(max_i8(&s), 12);
    }

    #[test]
    fn min_max_i8_empty() {
        let s: [i8; 0] = [];
        assert_eq!(min_i8(&s), i8::MAX);
        assert_eq!(max_i8(&s), i8::MIN);
    }
}
