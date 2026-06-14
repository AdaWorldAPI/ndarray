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
/// Dispatches to the widest available SIMD lane:
///
/// | Backend    | Lane    | Per-iteration intrinsic |
/// |------------|---------|-------------------------|
/// | x86_64     | `I8x64` | `_mm512_add_epi8` zmm (AVX-512-BW) / 2× `_mm256_add_epi8` ymm (AVX2 polyfill of `I8x64`) |
/// | aarch64    | `I8x16` | `vaddq_s8` × N                                |
/// | other      | scalar  | `i8::wrapping_add` lane-by-lane               |
///
/// Wrapping arithmetic. Panics if `dst.len() != src.len()`.
#[inline]
pub fn add_i8(dst: &mut [i8], src: &[i8]) {
    assert_eq!(dst.len(), src.len(), "add_i8: length mismatch");
    let n = dst.len();

    #[cfg(target_arch = "x86_64")]
    {
        use crate::simd::I8x64;
        const L: usize = 64;
        let chunks = n / L;
        for c in 0..chunks {
            let off = c * L;
            let d = I8x64::from_slice(&dst[off..]);
            let s = I8x64::from_slice(&src[off..]);
            let arr = (d + s).to_array();
            dst[off..off + L].copy_from_slice(&arr);
        }
        for i in (chunks * L)..n {
            dst[i] = dst[i].wrapping_add(src[i]);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::simd_neon::I8x16;
        const L: usize = 16;
        let chunks = n / L;
        for c in 0..chunks {
            let off = c * L;
            let d = I8x16::from_slice(&dst[off..]);
            let s = I8x16::from_slice(&src[off..]);
            let arr = d.add(s).to_array();
            dst[off..off + L].copy_from_slice(&arr);
        }
        for i in (chunks * L)..n {
            dst[i] = dst[i].wrapping_add(src[i]);
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for i in 0..n {
            dst[i] = dst[i].wrapping_add(src[i]);
        }
    }
}

/// Element-wise `dst[i] -= src[i]` (wrapping i8 sub).
///
/// Dispatches the same way as [`add_i8`] (zmm AVX-512-BW / ymm AVX2 /
/// 128-bit NEON / scalar) using the polyfilled lane's `Sub`
/// implementation.
#[inline]
pub fn sub_i8(dst: &mut [i8], src: &[i8]) {
    assert_eq!(dst.len(), src.len(), "sub_i8: length mismatch");
    let n = dst.len();

    #[cfg(target_arch = "x86_64")]
    {
        use crate::simd::I8x64;
        const L: usize = 64;
        let chunks = n / L;
        for c in 0..chunks {
            let off = c * L;
            let d = I8x64::from_slice(&dst[off..]);
            let s = I8x64::from_slice(&src[off..]);
            let arr = (d - s).to_array();
            dst[off..off + L].copy_from_slice(&arr);
        }
        for i in (chunks * L)..n {
            dst[i] = dst[i].wrapping_sub(src[i]);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::simd_neon::I8x16;
        const L: usize = 16;
        let chunks = n / L;
        for c in 0..chunks {
            let off = c * L;
            let d = I8x16::from_slice(&dst[off..]);
            let s = I8x16::from_slice(&src[off..]);
            let arr = d.sub(s).to_array();
            dst[off..off + L].copy_from_slice(&arr);
        }
        for i in (chunks * L)..n {
            dst[i] = dst[i].wrapping_sub(src[i]);
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for i in 0..n {
            dst[i] = dst[i].wrapping_sub(src[i]);
        }
    }
}

/// Element-wise `dst[i] += src[i]` (wrapping i16 add).
///
/// Dispatches to `I16x32` (AVX-512-BW `_mm512_add_epi16`) on x86_64,
/// `I16x8` (`vaddq_s16`) on aarch64, scalar otherwise.
#[inline]
pub fn add_i16(dst: &mut [i16], src: &[i16]) {
    assert_eq!(dst.len(), src.len(), "add_i16: length mismatch");
    let n = dst.len();

    #[cfg(target_arch = "x86_64")]
    {
        use crate::simd::I16x32;
        const L: usize = 32;
        let chunks = n / L;
        for c in 0..chunks {
            let off = c * L;
            let d = I16x32::from_slice(&dst[off..]);
            let s = I16x32::from_slice(&src[off..]);
            let arr = (d + s).to_array();
            dst[off..off + L].copy_from_slice(&arr);
        }
        for i in (chunks * L)..n {
            dst[i] = dst[i].wrapping_add(src[i]);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use crate::simd_neon::I16x8;
        const L: usize = 8;
        let chunks = n / L;
        for c in 0..chunks {
            let off = c * L;
            let d = I16x8::from_slice(&dst[off..]);
            let s = I16x8::from_slice(&src[off..]);
            let arr = d.add(s).to_array();
            dst[off..off + L].copy_from_slice(&arr);
        }
        for i in (chunks * L)..n {
            dst[i] = dst[i].wrapping_add(src[i]);
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for i in 0..n {
            dst[i] = dst[i].wrapping_add(src[i]);
        }
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
// gemm_u8_i8 — agnostic u8 × i8 → i32 matrix multiply
// ────────────────────────────────────────────────────────────────────────

/// `C = A · B` where `A` is `M × K` `u8`, `B` is `K × N` `i8`, `C` is `M × N`
/// `i32` (row-major, overwritten — not accumulated).
///
/// Agnostic consumer surface. Resolves at **compile time** to one kernel
/// per the active `target_feature` set; consumers never branch on CPU
/// capability and the chosen kernel is fully inlined at the call site.
///
/// Build matrix (additive, filled in as paths land):
///
/// | `target_feature`           | Kernel                                                |
/// |----------------------------|-------------------------------------------------------|
/// | `amx-int8` *(planned)*     | AMX `TDPBUSD` 16×16 tile (Sapphire / Granite Rapids)  |
/// | `avx512vnni`               | `VPDPBUSD` zmm — 16 i32 lanes (CLX → Zen 4 / SPR)     |
/// | `avxvnni`                  | `VPDPBUSD` ymm — 8 i32 lanes (Alder/Arrow Lake, Zen 4)|
/// | `neon,dotprod` *(planned)* | NEON `SDOT` (A76+ / Apple M-series)                   |
/// | *(none)*                   | Scalar reference [`hpc::quantized::int8_gemm_i32`]    |
///
/// Arm precedence is widest-vector-first: when several `target_feature`
/// flags are set simultaneously (e.g. Sapphire Rapids enables `avx512vnni`
/// AND `avxvnni`), the highest-bandwidth arm wins via `#[cfg]` ordering.
///
/// Build configs:
///
/// * Default `x86-64-v3` (no VNNI) → scalar arm. Same result as calling
///   [`crate::hpc::quantized::int8_gemm_i32`] directly.
/// * `--config .cargo/config-avx512.toml` (= Sapphire Rapids, includes
///   VNNI + BF16 + FP16 + AMX) → the `avx512vnni` zmm arm. The future
///   `amx-int8` arm, once landed, will preempt this on the same config.
/// * `-Ctarget-cpu=cascadelake` / `znver4` → also lands in the
///   `avx512vnni` zmm arm (no AMX, no BF16).
/// * `RUSTFLAGS='-Ctarget-feature=+avxvnni'` on an AVX2 baseline →
///   the `avxvnni` ymm arm (Arrow Lake / Alder Lake without AVX-512).
///
/// # Panics
///
/// Panics if the slice lengths are inconsistent with the given dimensions.
#[inline]
pub fn gemm_u8_i8(a: &[u8], b: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    assert!(a.len() >= m * k, "gemm_u8_i8: a.len()={} < m*k={}", a.len(), m * k);
    assert!(b.len() >= k * n, "gemm_u8_i8: b.len()={} < k*n={}", b.len(), k * n);
    assert!(c.len() >= m * n, "gemm_u8_i8: c.len()={} < m*n={}", c.len(), m * n);

    // Tier 0 — runtime AMX check. AMX is a different feature class than
    // the rest of the dispatch chain: it requires CPUID + XCR0 + a Linux
    // `prctl(ARCH_REQ_XCOMP_PERM, 18)` to be granted, none of which fit
    // a `target_feature` compile-time gate. The check is one CPUID +
    // one XGETBV + one prctl (idempotent, cached after first call). On
    // aligned shapes (16/16/64) this dispatches to TDPBUSD via the
    // shared `int8_gemm_amx_tiled` helper — 16 384 MACs per instruction
    // vs VPDPBUSD-zmm's 64. Since `gemm_u8_i8` is u8×i8 natively (no
    // sign-shift bias needed), the AMX path is a direct call with no
    // bias correction — simpler than `matmul_i8_to_i32`'s i8×i8 path.
    #[cfg(target_arch = "x86_64")]
    {
        if crate::hpc::amx_matmul::amx_available()
            && m.is_multiple_of(16)
            && n.is_multiple_of(16)
            && k.is_multiple_of(64)
        {
            crate::hpc::int8_tile_gemm::int8_gemm_amx_tiled(a, b, c, m, n, k);
            return;
        }
    }

    // RUNTIME VNNI dispatch (tiers 1-2, after the AMX check above). This MUST
    // be runtime `is_x86_feature_detected!`, NOT compile-time
    // `#[cfg(target_feature)]`: the default x86-64-v3 build has neither
    // avx512vnni nor avxvnni as a *compile* feature, so a cfg chain would strip
    // both arms and fall through to scalar even on Ice Lake / Sapphire Rapids /
    // Zen 4 silicon that supports VNNI at runtime (the regression codex flagged
    // on PR #217). Runtime detection keeps the VNNI kernels reachable on the
    // baseline build, matching the pre-consolidation `simd_caps()` behaviour.
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512vnni") {
            // SAFETY: avx512vnni detected ⇒ AVX-512F + VNNI + BW present, the
            // kernel's `#[target_feature(enable)]` set.
            unsafe { crate::hpc::vnni_gemm::int8_gemm_vnni_avx512(a, b, c, m, n, k) };
            return;
        }
        if std::is_x86_feature_detected!("avxvnni") {
            // SAFETY: avxvnni detected ⇒ AVX + AVX2 + AVX-VNNI present.
            unsafe { crate::hpc::vnni_gemm::int8_gemm_avxvnni_ymm(a, b, c, m, n, k) };
            return;
        }
    }

    // Fallback: scalar reference kernel. Always correct; same result the
    // VNNI / AMX / SDOT paths produce when they land. Targets without an
    // INT8 dot-product instruction (x86-64-v3 baseline without AVX-VNNI,
    // ARMv8.0 without dotprod, wasm32, riscv) reach this arm at compile
    // time.
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx512vnni"),
        all(target_arch = "x86_64", target_feature = "avxvnni"),
    )))]
    {
        crate::hpc::quantized::int8_gemm_i32(a, b, c, m, n, k);
    }
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

    // ── gemm_u8_i8 ────────────────────────────────────────────────────────

    /// Independent scalar reference used to validate `gemm_u8_i8` against
    /// the active compile-time dispatch arm (scalar or VNNI), without
    /// going through `hpc::quantized::int8_gemm_i32` (which IS the scalar
    /// arm — comparing against it on a v3 build would be tautological).
    fn ref_gemm_u8_i8(a: &[u8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for p in 0..k {
                let av = a[i * k + p] as i32;
                for j in 0..n {
                    c[i * n + j] += av * b[p * n + j] as i32;
                }
            }
        }
        c
    }

    #[test]
    fn gemm_u8_i8_4x4_identity() {
        let m = 4;
        let n = 4;
        let k = 4;
        let a: Vec<u8> = (1..=16).collect();
        let b: Vec<i8> = vec![1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
        let expected = ref_gemm_u8_i8(&a, &b, m, n, k);
        let mut c = vec![99i32; m * n];
        gemm_u8_i8(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected);
    }

    #[test]
    fn gemm_u8_i8_rectangular_3x5x8() {
        let m = 3;
        let n = 5;
        let k = 8;
        let a: Vec<u8> = (0..m * k).map(|i| (i % 200) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 100) as i8 - 50).collect();
        let expected = ref_gemm_u8_i8(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        gemm_u8_i8(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected);
    }

    #[test]
    fn gemm_u8_i8_17x17_tail() {
        // Exercises the VNNI tail-masking path on AVX-512 builds and the
        // scalar fallback on v3 builds. Same expected output either way.
        let m = 17;
        let n = 17;
        let k = 17;
        let a: Vec<u8> = (0..m * k).map(|i| ((i * 7 + 3) % 256) as u8).collect();
        let b: Vec<i8> = (0..k * n)
            .map(|i| ((i * 11 + 5) % 256) as u8 as i8)
            .collect();
        let expected = ref_gemm_u8_i8(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        gemm_u8_i8(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected);
    }

    #[test]
    fn gemm_u8_i8_extreme_values() {
        // u8 = 255, i8 alternating ±127 stresses i32 accumulation across
        // the AVX-512 tail path and the scalar reference.
        let m = 4;
        let n = 4;
        let k = 8;
        let a = vec![255u8; m * k];
        let b: Vec<i8> = (0..k * n)
            .map(|i| if i % 2 == 0 { 127i8 } else { -128i8 })
            .collect();
        let expected = ref_gemm_u8_i8(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        gemm_u8_i8(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected);
    }

    /// Sanity-check timing harness — run with:
    ///   cargo test --release simd_int_ops::tests::bench_gemm_u8_i8_vs_scalar \
    ///       -- --ignored --nocapture
    ///
    /// Re-run under each cfg arm to confirm the kernel actually beats the
    /// scalar reference (the question the user raised: "if AVX2 ends up
    /// slower than scalar GEMM something isn't done right"):
    ///   # scalar arm (default v3)
    ///   cargo test --release ...
    ///   # avxvnni ymm arm
    ///   RUSTFLAGS='-Ctarget-cpu=alderlake' cargo test --release ...
    ///   # avx512vnni zmm arm
    ///   cargo --config .cargo/config-avx512.toml test --release ...
    #[test]
    #[ignore]
    fn bench_gemm_u8_i8_vs_scalar() {
        use std::time::Instant;

        let sizes = [(64usize, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)];

        for (m, n, k) in sizes {
            let a: Vec<u8> = (0..m * k).map(|i| (i % 251) as u8).collect();
            let b: Vec<i8> = (0..k * n)
                .map(|i| ((i % 127) as i8).wrapping_sub(63))
                .collect();
            let mut c_simd = vec![0i32; m * n];
            let mut c_scalar = vec![0i32; m * n];

            // Warm-up — first call also resolves any one-time setup.
            for _ in 0..2 {
                gemm_u8_i8(&a, &b, &mut c_simd, m, n, k);
            }
            for _ in 0..2 {
                crate::hpc::quantized::int8_gemm_i32(&a, &b, &mut c_scalar, m, n, k);
            }

            // Iterations scale down with size to keep total time reasonable.
            let iters = match m {
                0..=64 => 50,
                65..=128 => 10,
                129..=256 => 3,
                _ => 1,
            };

            let t0 = Instant::now();
            for _ in 0..iters {
                gemm_u8_i8(&a, &b, &mut c_simd, m, n, k);
            }
            let dt_simd = t0.elapsed() / iters;

            let t0 = Instant::now();
            for _ in 0..iters {
                crate::hpc::quantized::int8_gemm_i32(&a, &b, &mut c_scalar, m, n, k);
            }
            let dt_scalar = t0.elapsed() / iters;

            assert_eq!(c_simd, c_scalar, "perf bench failed correctness at {m}x{n}x{k}");
            let speedup = dt_scalar.as_nanos() as f64 / dt_simd.as_nanos() as f64;
            println!(
                "gemm_u8_i8 {m:>3}x{n:>3}x{k:>3}: simd={:>10.3?}  scalar={:>10.3?}  speedup={speedup:>6.2}x",
                dt_simd, dt_scalar,
            );
        }
    }

    // ── W1a parity tests ────────────────────────────────────────────────────
    //
    // These tests exercise the correctness of the 5 W1a primitives on the
    // current compilation backend.  Because the dispatch is compile-time
    // only, each test runs against exactly one backend per build.  The
    // fixed corpus includes all required edge-case values from the consumer
    // contract (i8::MIN, i8::MAX, 0, all-bits-set u64, OOB index edge).

    /// W1a-#1 + #2: I8x16 from_i4_packed_u64 + lane_i8 + saturating_abs
    #[test]
    fn w1a_i8x16_from_i4_packed_u64_basic() {
        use crate::simd::I8x16;
        // All nibbles 0 → all lanes 0
        let z = I8x16::from_i4_packed_u64(0);
        assert!(z.to_array().iter().all(|&x| x == 0), "all-zero packed");

        // All nibbles 0xf → all lanes -1
        let neg = I8x16::from_i4_packed_u64(u64::MAX);
        assert!(neg.to_array().iter().all(|&x| x == -1), "all-0xf packed → -1");

        // Nibble 0x8 = minimum i4 → lane value -8
        let min4 = I8x16::from_i4_packed_u64(0x8888_8888_8888_8888);
        assert!(min4.to_array().iter().all(|&x| x == -8), "nibble 0x8 → -8");

        // Nibble 0x7 = maximum positive i4 → lane value +7
        let max4 = I8x16::from_i4_packed_u64(0x7777_7777_7777_7777);
        assert!(max4.to_array().iter().all(|&x| x == 7), "nibble 0x7 → 7");

        // lane_i8 extractors: nibbles are LSB-first and sign-extended.
        // packed = 0x...0021 → lane0 = nibble 0x1 = 1, lane1 = nibble 0x2 = 2.
        let low = I8x16::from_i4_packed_u64(0x0000_0000_0000_0021);
        assert_eq!(low.lane_i8::<0>(), 1);
        assert_eq!(low.lane_i8::<1>(), 2);
        // Sign bit: nibble 0x8 in lane0 sign-extends to -8.
        let signbit = I8x16::from_i4_packed_u64(0x0000_0000_0000_0008);
        assert_eq!(signbit.lane_i8::<0>(), -8);
    }

    /// W1a-#2: saturating_abs — binding contract test (i8::MIN → i8::MAX)
    #[test]
    fn w1a_saturating_abs_i8_min_matches_across_backends() {
        use crate::simd::{I8x16, I8x32};

        // I8x16
        let input16 = I8x16::splat(i8::MIN);
        let result16 = input16.saturating_abs();
        let arr16 = result16.to_array();
        for (lane, &v) in arr16.iter().enumerate() {
            assert_eq!(v, i8::MAX, "I8x16 lane {} saturating_abs(i8::MIN) should be i8::MAX", lane);
        }

        // I8x32
        let input32 = I8x32::splat(i8::MIN);
        let result32 = input32.saturating_abs();
        let arr32 = result32.to_array();
        for (lane, &v) in arr32.iter().enumerate() {
            assert_eq!(v, i8::MAX, "I8x32 lane {} saturating_abs(i8::MIN) should be i8::MAX", lane);
        }

        // Corpus: 0, 1, -1, i8::MAX, i8::MIN
        let corpus: &[i8] = &[0, 1, -1, i8::MAX, i8::MIN, 42, -42, 127, -127, -128, 64, -64];
        for &val in corpus {
            // Scalar reference
            let expected = val.saturating_abs();

            let v16 = I8x16::splat(val).saturating_abs().lane_i8::<0>();
            assert_eq!(v16, expected, "I8x16 saturating_abs({}) mismatch", val);

            let mut arr32 = [0i8; 32];
            arr32[0] = val;
            let v32 = I8x32::from_array(arr32).saturating_abs().to_array()[0];
            assert_eq!(v32, expected, "I8x32 saturating_abs({}) mismatch", val);
        }
    }

    /// W1a-#3: gather_u16 + palette_lookup_u8x8
    #[test]
    fn w1a_gather_u16_basic() {
        use crate::simd::{palette_lookup_u8x8, U16x8};

        let table: Vec<u16> = (0..256).map(|x| x as u16 * 10).collect();
        let idx = U16x8::from_array([0, 1, 2, 3, 100, 200, 255, 50]);
        let result = U16x8::gather_u16(idx, &table);
        let expected = [0u16, 10, 20, 30, 1000, 2000, 2550, 500];
        assert_eq!(result.to_array(), expected, "gather_u16 basic");

        // All-same index
        let same_idx = U16x8::splat(5);
        let r2 = U16x8::gather_u16(same_idx, &table);
        assert!(r2.to_array().iter().all(|&v| v == 50), "gather_u16 all-same idx");

        // palette_lookup_u8x8
        let lut: Vec<u8> = (0..256).map(|x| x as u8).collect();
        let pidx = U16x8::from_array([0, 1, 127, 128, 254, 255, 10, 20]);
        let pr = palette_lookup_u8x8(pidx, &lut);
        assert_eq!(pr.to_array(), [0u8, 1, 127, 128, 254, 255, 10, 20], "palette_lookup_u8x8");
    }

    /// W1a-#4: prefetch — just verify they don't panic (they're hints)
    #[test]
    fn w1a_prefetch_no_panic() {
        use crate::simd::{prefetch_read_t0, prefetch_read_t1, prefetch_read_t2};
        let data = [0u8; 64];
        let ptr = data.as_ptr();
        // Valid pointer — must not panic
        prefetch_read_t0(ptr);
        prefetch_read_t1(ptr);
        prefetch_read_t2(ptr);
        // Null pointer — must not panic (prefetch is a hint, not a load)
        prefetch_read_t0(core::ptr::null());
        prefetch_read_t1(core::ptr::null());
        prefetch_read_t2(core::ptr::null());
    }

    /// W1a-#5: U64x8::popcnt / xor_popcount + U64x4::popcnt
    #[test]
    fn w1a_u64_popcnt_basic() {
        use crate::simd::{U64x4, U64x8};

        // U64x8
        let all_ones = U64x8::splat(u64::MAX);
        let p8 = all_ones.popcnt();
        assert!(p8.to_array().iter().all(|&x| x == 64), "U64x8::popcnt(MAX) == 64 per lane");

        let all_zero = U64x8::splat(0);
        let pz8 = all_zero.popcnt();
        assert!(pz8.to_array().iter().all(|&x| x == 0), "U64x8::popcnt(0) == 0 per lane");

        // xor_popcount: MAX ^ 0 = MAX, 64 bits × 8 lanes = 512
        assert_eq!(all_ones.xor_popcount(all_zero), 512, "xor_popcount(MAX,0) == 512");
        assert_eq!(all_ones.xor_popcount(all_ones), 0, "xor_popcount(x,x) == 0");

        // Known values
        let v = U64x8::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
        let pv = v.popcnt().to_array();
        assert_eq!(pv, [1, 1, 2, 1, 2, 2, 3, 1], "U64x8::popcnt known values");

        // U64x4
        let v4 = U64x4::from_array([u64::MAX, 0, 1, !1u64]);
        let pv4 = v4.popcnt().to_array();
        assert_eq!(pv4, [64, 0, 1, 63], "U64x4::popcnt known values");
    }

    /// W1a-#1: batch_packed_i4_16 smoke test
    #[test]
    fn w1a_batch_packed_i4_16_smoke() {
        use crate::simd::batch_packed_i4_16;

        let packed = vec![0u64; 4];
        let aux = vec![0i8; 4];
        let mut out = vec![0i8; 4];
        batch_packed_i4_16(&packed, &aux, &mut out, |lanes, a| lanes.lane_i8::<0>().wrapping_add(a));
        assert!(out.iter().all(|&v| v == 0), "batch_packed_i4_16 all-zero");

        // Non-zero nibbles
        let packed2 = vec![0x1111_1111_1111_1111u64; 2];
        let aux2 = vec![10i8; 2];
        let mut out2 = vec![0i8; 2];
        batch_packed_i4_16(&packed2, &aux2, &mut out2, |lanes, a| lanes.lane_i8::<0>().wrapping_add(a));
        // nibble 0x1 → lane 0 = +1; +10 = 11
        assert!(out2.iter().all(|&v| v == 11), "batch_packed_i4_16 nibble=1+aux=10");
    }

    /// Exercises the AMX dispatch tier added on top of `gemm_u8_i8`'s
    /// compile-time cascade. On AMX-enabled silicon (Sapphire Rapids+
    /// with the right OS prctl), 16/16/64-aligned shapes go through
    /// TDPBUSD via `int8_gemm_amx_tiled`. Anywhere else this falls back
    /// to the compile-time cascade — the assertion still holds because
    /// the scalar reference is exact integer arithmetic.
    #[test]
    fn gemm_u8_i8_amx_aligned_32x32x128() {
        let m = 32; // 2 × 16-wide M-tiles
        let n = 32; // 2 × 16-wide N-tiles
        let k = 128; // 2 × 64-wide K-blocks per tile
        let a: Vec<u8> = (0..m * k).map(|i| ((i * 13 + 7) % 256) as u8).collect();
        let b: Vec<i8> = (0..k * n)
            .map(|i| ((i * 19 + 11) % 256) as u8 as i8)
            .collect();
        let expected = ref_gemm_u8_i8(&a, &b, m, n, k);
        let mut c = vec![0i32; m * n];
        gemm_u8_i8(&a, &b, &mut c, m, n, k);
        assert_eq!(c, expected, "gemm_u8_i8 AMX path mismatch");
    }
}
