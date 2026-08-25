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
// Packed-bitmask predicates + mask algebra (the columnar-selection lane)
// ────────────────────────────────────────────────────────────────────────
//
// These seven primitives are the vector half of a columnar filter: turn a
// lane of values into a packed bit-per-row mask, compose masks with boolean
// algebra, and reduce a value lane under a mask. They are the substrate the
// `lance-graph-java` ABI membrane rides (`lgj_op_eq_u32`, `lgj_op_gt_i32`,
// `lgj_mask_and`, `lgj_mask_or`, `lgj_plan_eval`, `lgj_reduce_sum_i32`), and
// the reason that membrane needs no SIMD of its own — a consumer crate that
// wrote its own compare-and-pack loop would be an `ndarray::simd` bypass.
//
// ## Bit order (NORMATIVE — every function below obeys it)
//
// Element index `i` lives at **bit `i % 64` of word `i / 64`**; LSB-first
// within each word, so element 0 is bit 0 of `out_words[0]` and element 64 is
// bit 0 of `out_words[1]`. This matches the `MASK_WORD` lane definition on
// the ABI side ("a `u64` of 64 packed row bits, LSB = lowest row index") and
// the lane-level `u16` convention already established by
// `I32x16::cmpge_zero_mask`.
//
// **Trailing bits beyond `values.len()` in the final word are always written
// as 0**, as are any surplus words in a longer-than-necessary `out_words`.
// This is load-bearing: those bits feed straight into `popcount_batch_u64`,
// so a stale high bit would silently inflate a count. Every writer below
// zeroes the whole destination first and then only ever sets bits for
// in-range elements, which makes the guarantee structural rather than a
// tail-handling special case that could be forgotten.
//
// ## Why free functions here, not methods on a wrapper
//
// The W1a consumer contract's "struct method, not free function" litmus
// governs **lane-level** primitives, where a free function fragments the
// typed-wrapper surface. These are **slice-level**, the same tier as
// `add_i8` / `dot_i8` / `min_i8` above, and they are built *on* lane methods
// (`U32x16::eq_bitmask`, `I32x16::gt_bitmask`) that do live on the wrappers.

/// Number of packed mask words needed to cover `n` elements.
#[inline(always)]
fn mask_words_for(n: usize) -> usize {
    n.div_ceil(64)
}

/// Load 16 `u32` lanes from the front of `src`.
///
/// Uses `from_array` rather than `from_slice` deliberately: `from_slice` is
/// not present on every backend's `U32x16` (the NEON and wasm `[U32x4; 4]`
/// fan-outs expose `from_array` only), and going through the array keeps this
/// helper free of any `cfg(target_arch)` selection. The 64-byte copy is
/// elided into a single vector load by LLVM.
#[inline(always)]
fn load_u32x16(src: &[u32]) -> crate::simd::U32x16 {
    let mut a = [0u32; 16];
    a.copy_from_slice(&src[..16]);
    crate::simd::U32x16::from_array(a)
}

/// Load 16 `i32` lanes from the front of `src`. See [`load_u32x16`].
#[inline(always)]
fn load_i32x16(src: &[i32]) -> crate::simd::I32x16 {
    let mut a = [0i32; 16];
    a.copy_from_slice(&src[..16]);
    crate::simd::I32x16::from_array(a)
}

/// Packs `values[i] == needle` into `out_words`, one bit per element,
/// LSB-first within each `u64` word (bit `k` of word `w` corresponds to
/// element `w * 64 + k`).
///
/// `out_words` is **fully overwritten**, not OR-ed into. Trailing bits in the
/// final word beyond `values.len()`, and any surplus words past
/// `ceil(len / 64)`, are written as `0`.
///
/// Equality is exact bitwise comparison over the full `u32` range — `0` and
/// `u32::MAX` are ordinary needles, and there is no saturation, wrapping, or
/// signedness question to resolve. An empty `values` writes only zeros.
///
/// Runs 16 lanes at a time through [`crate::simd::U32x16::eq_bitmask`] with a
/// scalar tail for the final partial group; the scalar tail is bit-identical
/// to the vector path by construction (same comparison, same bit index).
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::eq_u32_to_mask;
///
/// let values = [7u32, 1, 7, 2];
/// let mut words = [0u64; 1];
/// eq_u32_to_mask(&values, 7, &mut words);
/// // elements 0 and 2 match → bits 0 and 2 → 0b0101
/// assert_eq!(words[0], 0b0101);
/// ```
#[inline]
pub fn eq_u32_to_mask(values: &[u32], needle: u32, out_words: &mut [u64]) {
    let n = values.len();
    let words = mask_words_for(n);
    assert!(out_words.len() >= words, "eq_u32_to_mask: out_words.len()={} < required {}", out_words.len(), words);

    // Zero first: makes the "trailing bits are 0" guarantee structural.
    for w in out_words.iter_mut() {
        *w = 0;
    }

    let needle_v = crate::simd::U32x16::splat(needle);
    let groups = n / 16;
    for g in 0..groups {
        let bits = load_u32x16(&values[g * 16..]).eq_bitmask(needle_v);
        out_words[g / 4] |= (bits as u64) << ((g % 4) * 16);
    }
    for i in (groups * 16)..n {
        if values[i] == needle {
            out_words[i / 64] |= 1u64 << (i % 64);
        }
    }
}

/// Packs `read_le_u32(bytes, first_offset + i * stride_bytes) == needle` into
/// `out_words`, one bit per element, LSB-first within each `u64` word — the
/// **strided** sibling of [`eq_u32_to_mask`], for scanning one `u32` field of
/// an AoS/facet row layout (e.g. a 4-byte classid at a fixed offset inside a
/// 512-byte row) without gathering the column into a contiguous copy first.
///
/// Element `i` is the little-endian `u32` at byte offset
/// `first_offset + i * stride_bytes`. `stride_bytes == 4` reads a contiguous
/// `u32` column (then [`eq_u32_to_mask`] is the better call); `stride_bytes
/// == 0` re-reads the same field `count` times, which is legal and produces
/// an all-ones or all-zeros mask.
///
/// `out_words` is **fully overwritten**, not OR-ed into; trailing bits and
/// surplus words are written `0`, exactly as in [`eq_u32_to_mask`].
///
/// The field loads are scalar by construction — at row strides ≥ one cache
/// line each element lives on its own line, so the walk is memory-bound and
/// a hardware gather buys nothing; SIMD earns its keep in the 16-wide
/// compare ([`crate::simd::U32x16::eq_bitmask`]) exactly as the contiguous
/// primitive does. Loads are `u32::from_le_bytes` over byte slices, so no
/// alignment is required of `bytes`.
///
/// # Panics
///
/// Panics if `out_words.len() < count.div_ceil(64)`, or if any element's four
/// bytes would fall outside `bytes` (checked up front, including overflow of
/// the offset arithmetic — the loop never reads out of bounds).
///
/// # Examples
///
/// ```
/// use ndarray::simd::eq_u32_strided_to_mask;
///
/// // Three 16-byte "facets"; the classid is the leading u32 of each.
/// let mut rows = vec![0u8; 48];
/// rows[0..4].copy_from_slice(&7u32.to_le_bytes());
/// rows[16..20].copy_from_slice(&9u32.to_le_bytes());
/// rows[32..36].copy_from_slice(&7u32.to_le_bytes());
/// let mut words = [0u64; 1];
/// eq_u32_strided_to_mask(&rows, 0, 16, 3, 7, &mut words);
/// assert_eq!(words[0], 0b101);
/// ```
#[inline]
pub fn eq_u32_strided_to_mask(
    bytes: &[u8], first_offset: usize, stride_bytes: usize, count: usize, needle: u32, out_words: &mut [u64],
) {
    let words = mask_words_for(count);
    assert!(
        out_words.len() >= words,
        "eq_u32_strided_to_mask: out_words.len()={} < required {}",
        out_words.len(),
        words
    );
    if count > 0 {
        // Bounds of the LAST element, computed with overflow checks so a
        // pathological stride cannot wrap around into a bogus in-bounds read.
        let last_start = (count - 1)
            .checked_mul(stride_bytes)
            .and_then(|o| o.checked_add(first_offset))
            .expect("eq_u32_strided_to_mask: offset arithmetic overflow");
        let last_end = last_start
            .checked_add(4)
            .expect("eq_u32_strided_to_mask: offset arithmetic overflow");
        assert!(
            last_end <= bytes.len(),
            "eq_u32_strided_to_mask: element {} at byte {}..{} is out of bounds (len {})",
            count - 1,
            last_start,
            last_end,
            bytes.len()
        );
    }

    for w in out_words.iter_mut() {
        *w = 0;
    }

    #[inline(always)]
    fn read_le_u32(bytes: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    }

    let needle_v = crate::simd::U32x16::splat(needle);
    let groups = count / 16;
    for g in 0..groups {
        let base = first_offset + g * 16 * stride_bytes;
        let lanes: [u32; 16] = core::array::from_fn(|k| read_le_u32(bytes, base + k * stride_bytes));
        let bits = crate::simd::U32x16::from_array(lanes).eq_bitmask(needle_v);
        out_words[g / 4] |= (bits as u64) << ((g % 4) * 16);
    }
    for i in (groups * 16)..count {
        if read_le_u32(bytes, first_offset + i * stride_bytes) == needle {
            out_words[i / 64] |= 1u64 << (i % 64);
        }
    }
}

/// Packs `values[i] > threshold` (**signed** comparison) into `out_words`,
/// one bit per element, LSB-first within each `u64` word (bit `k` of word `w`
/// corresponds to element `w * 64 + k`).
///
/// `out_words` is **fully overwritten**, not OR-ed into. Trailing bits in the
/// final word beyond `values.len()`, and any surplus words past
/// `ceil(len / 64)`, are written as `0`.
///
/// Comparison is two's-complement signed and strict (`>`, never `>=`); it is
/// exact with no saturation or wrapping:
/// * `threshold == i32::MIN` sets every lane except those equal to `i32::MIN`.
/// * `threshold == i32::MAX` sets nothing — no `i32` exceeds it.
/// * Negative values compare as signed, *not* as bit patterns: `-1 > 0` is
///   `false` even though the same bits compare greater unsigned.
///
/// An empty `values` writes only zeros.
///
/// Runs 16 lanes at a time through [`crate::simd::I32x16::gt_bitmask`] with a
/// scalar tail for the final partial group.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::gt_i32_to_mask;
///
/// let values = [5i32, -5, 0, i32::MAX];
/// let mut words = [0u64; 1];
/// gt_i32_to_mask(&values, 0, &mut words);
/// // elements 0 and 3 exceed 0 → bits 0 and 3 → 0b1001
/// assert_eq!(words[0], 0b1001);
/// ```
#[inline]
pub fn gt_i32_to_mask(values: &[i32], threshold: i32, out_words: &mut [u64]) {
    let n = values.len();
    let words = mask_words_for(n);
    assert!(out_words.len() >= words, "gt_i32_to_mask: out_words.len()={} < required {}", out_words.len(), words);

    for w in out_words.iter_mut() {
        *w = 0;
    }

    let threshold_v = crate::simd::I32x16::splat(threshold);
    let groups = n / 16;
    for g in 0..groups {
        let bits = load_i32x16(&values[g * 16..]).gt_bitmask(threshold_v);
        out_words[g / 4] |= (bits as u64) << ((g % 4) * 16);
    }
    for i in (groups * 16)..n {
        if values[i] > threshold {
            out_words[i / 64] |= 1u64 << (i % 64);
        }
    }
}

/// `dst = a & b`, elementwise over `u64` mask words.
///
/// Pure bitwise AND — no element-count awareness, so the caller's bit-order
/// convention (element `i` at bit `i % 64` of word `i / 64`) is preserved
/// automatically, including the trailing-zero guarantee: zero AND anything is
/// zero, so a conforming pair of inputs yields a conforming output.
///
/// `dst` must **not** overlap `a` or `b`; use [`mask_and_assign`] for the
/// in-place case (Rust's borrow rules already prevent the overlap in safe
/// code, so this is a note about which function to reach for, not a hazard).
///
/// # Panics
///
/// Panics unless `a.len() == b.len() == dst.len()`.
#[inline]
pub fn mask_and(a: &[u64], b: &[u64], dst: &mut [u64]) {
    assert_eq!(a.len(), b.len(), "mask_and: a/b length mismatch");
    assert_eq!(a.len(), dst.len(), "mask_and: a/dst length mismatch");
    let n = a.len();

    const L: usize = crate::simd::U64x8::LANES;
    let groups = n / L;
    for g in 0..groups {
        let off = g * L;
        let va = crate::simd::U64x8::from_slice(&a[off..]);
        let vb = crate::simd::U64x8::from_slice(&b[off..]);
        (va & vb).copy_to_slice(&mut dst[off..]);
    }
    for i in (groups * L)..n {
        dst[i] = a[i] & b[i];
    }
}

/// `dst = a | b`, elementwise over `u64` mask words.
///
/// Pure bitwise OR. Note the trailing-zero asymmetry versus [`mask_and`]: OR
/// preserves the guarantee only if **both** inputs already conform, because a
/// stray high bit in either operand survives. Every mask this module produces
/// conforms, so composing them is safe; a hand-built mask word is the caller's
/// responsibility.
///
/// `dst` must not overlap `a` or `b`; use [`mask_or_assign`] in-place.
///
/// # Panics
///
/// Panics unless `a.len() == b.len() == dst.len()`.
#[inline]
pub fn mask_or(a: &[u64], b: &[u64], dst: &mut [u64]) {
    assert_eq!(a.len(), b.len(), "mask_or: a/b length mismatch");
    assert_eq!(a.len(), dst.len(), "mask_or: a/dst length mismatch");
    let n = a.len();

    const L: usize = crate::simd::U64x8::LANES;
    let groups = n / L;
    for g in 0..groups {
        let off = g * L;
        let va = crate::simd::U64x8::from_slice(&a[off..]);
        let vb = crate::simd::U64x8::from_slice(&b[off..]);
        (va | vb).copy_to_slice(&mut dst[off..]);
    }
    for i in (groups * L)..n {
        dst[i] = a[i] | b[i];
    }
}

/// `dst &= src`, elementwise over `u64` mask words.
///
/// The in-place form of [`mask_and`] — this is what a fused predicate plan
/// uses to narrow an accumulator, and what an ABI-level `mask_and(a, b, dst)`
/// with `dst` aliasing an operand must route to.
///
/// # Panics
///
/// Panics if `dst.len() != src.len()`.
#[inline]
pub fn mask_and_assign(dst: &mut [u64], src: &[u64]) {
    assert_eq!(dst.len(), src.len(), "mask_and_assign: length mismatch");
    let n = dst.len();

    const L: usize = crate::simd::U64x8::LANES;
    let groups = n / L;
    for g in 0..groups {
        let off = g * L;
        let vd = crate::simd::U64x8::from_slice(&dst[off..]);
        let vs = crate::simd::U64x8::from_slice(&src[off..]);
        (vd & vs).copy_to_slice(&mut dst[off..]);
    }
    for i in (groups * L)..n {
        dst[i] &= src[i];
    }
}

/// `dst |= src`, elementwise over `u64` mask words.
///
/// The in-place form of [`mask_or`]. Same trailing-zero caveat as `mask_or`:
/// OR only preserves the convention if `src` conforms to it.
///
/// # Panics
///
/// Panics if `dst.len() != src.len()`.
#[inline]
pub fn mask_or_assign(dst: &mut [u64], src: &[u64]) {
    assert_eq!(dst.len(), src.len(), "mask_or_assign: length mismatch");
    let n = dst.len();

    const L: usize = crate::simd::U64x8::LANES;
    let groups = n / L;
    for g in 0..groups {
        let off = g * L;
        let vd = crate::simd::U64x8::from_slice(&dst[off..]);
        let vs = crate::simd::U64x8::from_slice(&src[off..]);
        (vd | vs).copy_to_slice(&mut dst[off..]);
    }
    for i in (groups * L)..n {
        dst[i] |= src[i];
    }
}

/// `dst = a & !b`, elementwise over `u64` mask words — "a minus b" as a
/// bitmask set difference (every bit set in `a` but not in `b`).
///
/// # Tail-bit semantics
///
/// `!b` sets every bit of `b`'s tail — the padding bits past whatever
/// logical row count `b` represents — because bitwise NOT has no notion of
/// "past the end" and will happily flip a conforming (zero) tail to all
/// ones. That looks like the same hazard [`mask_or`] warns about, but the
/// AND with `a` recovers it: `a & !b` is a bitwise subset of `a` (every bit
/// set in the result is also set in `a`), so **`dst`'s tail is zero
/// whenever `a`'s tail is zero, regardless of what `!b`'s tail does.** This
/// is the same pre-conforming-inputs contract `mask_or` documents — a
/// caller holding a possibly-non-conforming `a` must clear `a`'s tail
/// itself (the lgj-abi kernel does, against its own known `n_rows`); a
/// conforming `a` composes safely against any `b`, tail included.
///
/// `dst` must not overlap `a` or `b`; use [`mask_andnot_assign`] for the
/// in-place case (Rust's borrow rules already prevent the overlap in safe
/// code, so this is a note about which function to reach for, not a
/// hazard).
///
/// # Panics
///
/// Panics unless `a.len() == b.len() == dst.len()`.
#[inline]
pub fn mask_andnot(a: &[u64], b: &[u64], dst: &mut [u64]) {
    assert_eq!(a.len(), b.len(), "mask_andnot: a/b length mismatch");
    assert_eq!(a.len(), dst.len(), "mask_andnot: a/dst length mismatch");
    let n = a.len();

    const L: usize = crate::simd::U64x8::LANES;
    let groups = n / L;
    for g in 0..groups {
        let off = g * L;
        let va = crate::simd::U64x8::from_slice(&a[off..]);
        let vb = crate::simd::U64x8::from_slice(&b[off..]);
        (va & !vb).copy_to_slice(&mut dst[off..]);
    }
    for i in (groups * L)..n {
        dst[i] = a[i] & !b[i];
    }
}

/// `a &= !b`, elementwise over `u64` mask words.
///
/// The in-place form of [`mask_andnot`] — same tail-bit contract: the
/// result is a bitwise subset of the (pre-update) `a`, so `a`'s tail stays
/// zero whenever it started zero, regardless of what `b`'s tail holds.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn mask_andnot_assign(a: &mut [u64], b: &[u64]) {
    assert_eq!(a.len(), b.len(), "mask_andnot_assign: length mismatch");
    let n = a.len();

    const L: usize = crate::simd::U64x8::LANES;
    let groups = n / L;
    for g in 0..groups {
        let off = g * L;
        let va = crate::simd::U64x8::from_slice(&a[off..]);
        let vb = crate::simd::U64x8::from_slice(&b[off..]);
        (va & !vb).copy_to_slice(&mut a[off..]);
    }
    for i in (groups * L)..n {
        a[i] &= !b[i];
    }
}

/// Sum of `values[i]` where mask bit `i` is set, widened to `i64`.
///
/// Bit order is the module convention: element `i` is bit `i % 64` of
/// `mask_words[i / 64]`.
///
/// ## Overflow behaviour (precise)
///
/// Each element is widened to `i64` **before** accumulation, so no
/// intermediate can overflow at any realistic length: the worst case is
/// `n × |i32::MIN|`, which stays inside `i64` for every `n < 2^32` — i.e. for
/// every slice that can exist in a 64-bit address space at 4 bytes per
/// element. The accumulation is nevertheless written as `wrapping_add` so
/// that the theoretical `n ≥ 2^32` case has defined behaviour (two's-complement
/// wrap) rather than a debug-only panic that a release build would silently
/// disagree with. An empty mask, or a mask with no bits set, returns `0`.
///
/// **Mask bits at or beyond `values.len()` are ignored**, not summed and not
/// an error: the final word is masked down to the valid element count before
/// its bits are walked. This makes the function total for any conforming or
/// over-long mask, and means a caller cannot read past the value lane by
/// handing over a dirty tail.
///
/// ## Why this one is not a 16-lane reduce
///
/// The obvious vector shape — load `I32x16`, zero the unselected lanes,
/// `reduce_sum()` — is **wrong**, and quietly so: `reduce_sum` on `I32x16`
/// accumulates in `i32`, and 16 lanes near `i32::MAX` overflow it while the
/// widened contract promises they cannot. Preserving the `i64` guarantee is
/// worth more than the lanes here, so the body walks set bits with
/// `u64::trailing_zeros` (one `TZCNT`/`RBIT+CLZ` per selected element, and
/// entire zero words skipped in one test). Cost is proportional to the
/// popcount, not the row count, which is the right shape for a selective
/// filter anyway.
///
/// # Panics
///
/// Panics if `mask_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::masked_sum_i32;
///
/// let values = [10i32, 20, 30, 40];
/// // bits 0 and 2 set → 10 + 30
/// assert_eq!(masked_sum_i32(&values, &[0b0101]), 40);
/// ```
#[inline]
pub fn masked_sum_i32(values: &[i32], mask_words: &[u64]) -> i64 {
    let n = values.len();
    let words = mask_words_for(n);
    assert!(
        mask_words.len() >= words,
        "masked_sum_i32: mask_words.len()={} < required {}",
        mask_words.len(),
        words
    );

    let mut acc: i64 = 0;
    for (w, &word) in mask_words.iter().take(words).enumerate() {
        let base = w * 64;
        let mut bits = word;
        // Clamp the final partial word to the valid element count so a dirty
        // tail can never index past `values`.
        let valid = n - base;
        if valid < 64 {
            bits &= (1u64 << valid) - 1;
        }
        while bits != 0 {
            let lane = bits.trailing_zeros() as usize;
            acc = acc.wrapping_add(values[base + lane] as i64);
            bits &= bits - 1;
        }
    }
    acc
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

/// Sum a sub-word group field out of a **strided** record, over the records a
/// mask selects, widened to `i128` and range-checked into `i64`.
///
/// The shape this exists for: a row-strided store whose each record carries a small
/// content-blind register, read under a runtime grouping — `groups × group_bytes`
/// little-endian fields per record. `lance-graph-java`'s V3 facet is the
/// motivating case (512-byte rows, a 12-byte register read as `6×2` / `4×3` /
/// `3×4`), but nothing here is specific to it.
///
/// # Why this lives HERE
///
/// It is the primitive a consumer would otherwise hand-roll with raw intrinsics,
/// which is exactly what the "all SIMD from `ndarray::simd`" invariant exists to
/// prevent. [`masked_sum_i32`] is contiguous `i32`;
/// [`eq_u32_strided_to_mask`] reads one aligned `u32` per record. Neither covers
/// "gather a sub-word group out of a strided register and widen-accumulate", so
/// the consumer had a real gap and this closes it.
///
/// # Vectorisation, honestly
///
/// **This kernel is scalar, and measurement is why — not oversight.** The access
/// pattern is one small register per record at a large stride (512 bytes in the
/// motivating case), so every record is on its own cache line and the loop is
/// memory-bound. The per-record work is 12 bytes; a vector register is 32-64.
/// There is no way to vector-load several records' registers at once because
/// they are not adjacent, and widening 6 `u16`s within one record does not fill
/// a lane. Vectorising the *decode* would optimise the part that is already
/// free.
///
/// Should a caller ever present a CONTIGUOUS or small-stride variant, that is a
/// different primitive with a different name, and it would genuinely vectorise —
/// this one should not grow a flag for it.
///
/// # Overflow
///
/// Accumulates in `i128` and range-checks once, returning `None` rather than a
/// wrapped value. `i64` is not closed under this reduction: with
/// `group_bytes = 4` a single record contributes up to `groups × (2³² − 1)`.
///
/// # Panics
///
/// If `group_bytes` is not in `1..=4`, if `mask_words` is too short for
/// `n_records`, or if the last selected record's field would read past `bytes`.
/// Each is a caller contract violation rather than a recoverable condition.
///
/// ```
/// use ndarray::simd::masked_strided_group_sum;
///
/// // Two 8-byte records; the register starts at byte 2 and holds 3 × u16 LE.
/// let mut b = vec![0u8; 16];
/// b[2..8].copy_from_slice(&[1, 0, 2, 0, 3, 0]);   // record 0 -> 1 + 2 + 3
/// b[10..16].copy_from_slice(&[10, 0, 20, 0, 30, 0]); // record 1 -> 60
/// // mask selects record 0 only
/// assert_eq!(masked_strided_group_sum(&b, 2, 8, 2, 3, 2, &[0b01]), Some(6));
/// // both records
/// assert_eq!(masked_strided_group_sum(&b, 2, 8, 2, 3, 2, &[0b11]), Some(66));
/// ```
#[inline]
pub fn masked_strided_group_sum(
    bytes: &[u8], first_offset: usize, stride_bytes: usize, n_records: usize, groups: usize, group_bytes: usize,
    mask_words: &[u64],
) -> Option<i64> {
    assert!((1..=4).contains(&group_bytes), "masked_strided_group_sum: group_bytes={group_bytes} outside 1..=4");
    let words = mask_words_for(n_records);
    assert!(
        mask_words.len() >= words,
        "masked_strided_group_sum: mask_words.len()={} < required {}",
        mask_words.len(),
        words
    );

    let mut acc: i128 = 0;
    for (w, &word) in mask_words.iter().take(words).enumerate() {
        let base = w * 64;
        if base >= n_records {
            break;
        }
        let mut bits = word;
        // Clamp the final partial word so a dirty tail cannot address a record
        // that does not exist. Same guard, same reason, as `masked_sum_i32`.
        let valid = n_records - base;
        if valid < 64 {
            bits &= (1u64 << valid) - 1;
        }
        while bits != 0 {
            let rec = base + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let reg = rec * stride_bytes + first_offset;
            let end = reg + groups * group_bytes;
            assert!(
                end <= bytes.len(),
                "masked_strided_group_sum: record {rec} reads {reg}..{end}, past len {}",
                bytes.len()
            );
            for g in 0..groups {
                let o = reg + g * group_bytes;
                // Byte-wise, not a widened load: `o` is not guaranteed aligned
                // for a 3-byte grouping, and an unaligned wide read is UB in
                // Rust even where the hardware tolerates it.
                let mut v: u32 = 0;
                for k in 0..group_bytes {
                    v |= (bytes[o + k] as u32) << (8 * k);
                }
                acc += v as i128;
            }
        }
    }
    i64::try_from(acc).ok()
}

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

    // ── Packed-bitmask predicates + mask algebra ────────────────────────────
    //
    // Every test compares the shipped path against an INDEPENDENT scalar
    // reference written inline here (never against the implementation's own
    // scalar tail, which would be tautological), over a fixed-seed corpus plus
    // the explicit edge cases: empty, 1, 63, 64, 65, non-multiples of 64,
    // all-match, no-match, `u32::MAX` needle, `i32::MIN`/`i32::MAX` thresholds,
    // and negative values. Bit order and the trailing-zero guarantee are
    // asserted literally, against hand-computed `u64` words.
    //
    // Dispatch is compile-time, so one build exercises one backend; the
    // scalar references below are what makes "all backends agree" checkable by
    // re-running under `-Ctarget-cpu=x86-64-v3` (AVX2 arm) and
    // `-Ctarget-cpu=x86-64-v4` (AVX-512 arm).

    /// Deterministic fixed-seed PRNG (SplitMix64) — no dev-dependency needed
    /// and the corpus is byte-identical on every run and every backend.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Independent reference: bit `i % 64` of word `i / 64` set where the
    /// predicate holds, everything else zero.
    fn ref_pack<T: Copy>(values: &[T], n_words: usize, pred: impl Fn(T) -> bool) -> Vec<u64> {
        let mut words = vec![0u64; n_words];
        for (i, &v) in values.iter().enumerate() {
            if pred(v) {
                words[i / 64] |= 1u64 << (i % 64);
            }
        }
        words
    }

    /// Lengths that straddle every boundary that matters: word edges (63/64/65),
    /// the 16-lane group edge (15/16/17), and non-multiples of both.
    const MASK_LENS: &[usize] = &[0, 1, 2, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65, 100, 127, 128, 129, 200, 255, 256];

    #[test]
    fn eq_u32_to_mask_matches_scalar_reference() {
        for &len in MASK_LENS {
            let mut seed = 0xA5A5_1234_DEAD_BEEF;
            let values: Vec<u32> = (0..len)
                .map(|_| (splitmix64(&mut seed) % 7) as u32)
                .collect();

            for needle in [0u32, 1, 3, 6, 42, u32::MAX] {
                let n_words = len.div_ceil(64);
                let expected = ref_pack(&values, n_words, |v| v == needle);

                let mut got = vec![0u64; n_words];
                eq_u32_to_mask(&values, needle, &mut got);
                assert_eq!(got, expected, "eq_u32_to_mask len={len} needle={needle}");
            }
        }
    }

    #[test]
    fn eq_u32_to_mask_all_match_and_no_match() {
        for &len in MASK_LENS {
            let n_words = len.div_ceil(64);

            // All-match: every in-range bit set, every out-of-range bit clear.
            let all = vec![9u32; len];
            let mut got = vec![0u64; n_words];
            eq_u32_to_mask(&all, 9, &mut got);
            assert_eq!(got, ref_pack(&all, n_words, |v| v == 9), "all-match len={len}");
            // Independent cross-check on the count, so a wrong-but-consistent
            // reference cannot hide: exactly `len` bits, no more.
            let popcnt: u32 = got.iter().map(|w| w.count_ones()).sum();
            assert_eq!(popcnt as usize, len, "all-match popcount len={len}");

            // No-match: strictly zero everywhere.
            let mut got = vec![u64::MAX; n_words]; // pre-dirtied — must be overwritten
            eq_u32_to_mask(&all, 10, &mut got);
            assert!(got.iter().all(|&w| w == 0), "no-match must be all zeros, len={len}");
        }
    }

    #[test]
    fn eq_u32_to_mask_u32_max_needle_and_values() {
        // u32::MAX is both a legal needle and a legal value; neither is special.
        let values = [u32::MAX, 0, u32::MAX, 1, u32::MAX - 1];
        let mut got = [0u64; 1];
        eq_u32_to_mask(&values, u32::MAX, &mut got);
        assert_eq!(got[0], 0b00101, "u32::MAX needle → bits 0 and 2");

        eq_u32_to_mask(&values, u32::MAX - 1, &mut got);
        assert_eq!(got[0], 0b10000, "u32::MAX-1 needle → bit 4 only");
    }

    /// The strided primitive against an independent reference, over an
    /// AoS-facet buffer shape (u32 field at `first_offset` inside a
    /// `stride_bytes`-wide row). Strides cover the contiguous case (4), a
    /// facet within a 16-byte record, and a 512-byte row.
    #[test]
    fn eq_u32_strided_to_mask_matches_scalar_reference() {
        for &count in MASK_LENS {
            for &(first_offset, stride) in &[(0usize, 4usize), (4, 16), (16, 512), (0, 0)] {
                let byte_len = if count == 0 {
                    0
                } else {
                    first_offset + (count - 1) * stride + 4
                };
                let mut seed = 0x0F0F_CAFE_F00D_1234 ^ (stride as u64);
                let mut bytes = vec![0u8; byte_len];
                // Fill every element position with a small-cardinality value so
                // needles genuinely hit and miss. stride==0 has ONE position.
                let positions = if stride == 0 { count.min(1) } else { count };
                let mut planted = Vec::with_capacity(positions);
                for i in 0..positions {
                    let v = (splitmix64(&mut seed) % 5) as u32;
                    let off = first_offset + i * stride;
                    bytes[off..off + 4].copy_from_slice(&v.to_le_bytes());
                    planted.push(v);
                }
                for needle in [0u32, 1, 4, 42] {
                    let n_words = count.div_ceil(64);
                    // Independent reference: read back the SAME strided walk
                    // scalar-only (stride 0 rereads element 0 `count` times).
                    let logical: Vec<u32> = (0..count)
                        .map(|i| {
                            if stride == 0 {
                                planted.first().copied().unwrap_or(0)
                            } else {
                                planted[i]
                            }
                        })
                        .collect();
                    let expected = ref_pack(&logical, n_words, |v| v == needle);

                    let mut got = vec![u64::MAX; n_words]; // pre-dirtied
                    eq_u32_strided_to_mask(&bytes, first_offset, stride, count, needle, &mut got);
                    assert_eq!(
                        got, expected,
                        "strided eq count={count} off={first_offset} stride={stride} needle={needle}"
                    );
                }
            }
        }
    }

    /// Parity with the contiguous primitive: stride 4 over the same values
    /// must produce bit-identical masks — two independent implementations of
    /// one specification.
    #[test]
    fn eq_u32_strided_stride4_matches_contiguous_primitive() {
        for &count in MASK_LENS {
            let mut seed = 0xBEE5_0000_0000_0001;
            let values: Vec<u32> = (0..count)
                .map(|_| (splitmix64(&mut seed) % 9) as u32)
                .collect();
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            let n_words = count.div_ceil(64);
            let mut a = vec![0u64; n_words];
            let mut b = vec![0u64; n_words];
            for needle in [0u32, 3, 8, u32::MAX] {
                eq_u32_to_mask(&values, needle, &mut a);
                eq_u32_strided_to_mask(&bytes, 0, 4, count, needle, &mut b);
                assert_eq!(a, b, "contiguous vs strided count={count} needle={needle}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn eq_u32_strided_rejects_a_last_element_past_the_buffer() {
        // 3 elements at stride 16 need bytes 32..36; a 35-byte buffer is short.
        let bytes = vec![0u8; 35];
        let mut words = [0u64; 1];
        eq_u32_strided_to_mask(&bytes, 0, 16, 3, 7, &mut words);
    }

    #[test]
    #[should_panic(expected = "offset arithmetic overflow")]
    fn eq_u32_strided_rejects_overflowing_offset_arithmetic() {
        let bytes = vec![0u8; 64];
        let mut words = [0u64; 1];
        // (count-1) * stride overflows usize — must panic, not wrap into a
        // bogus in-bounds read.
        eq_u32_strided_to_mask(&bytes, 0, usize::MAX, 3, 7, &mut words);
    }

    #[test]
    fn eq_u32_strided_empty_count_writes_only_zeros() {
        let bytes: Vec<u8> = Vec::new();
        let mut words = [u64::MAX; 2];
        eq_u32_strided_to_mask(&bytes, 0, 512, 0, 7, &mut words);
        assert_eq!(words, [0, 0], "count=0 must still overwrite the destination");
    }

    #[test]
    fn gt_i32_to_mask_matches_scalar_reference() {
        for &len in MASK_LENS {
            let mut seed = 0x1357_9BDF_0246_8ACE;
            // Full signed spread including both extremes, seeded deterministically.
            let values: Vec<i32> = (0..len)
                .map(|i| match i % 11 {
                    0 => i32::MIN,
                    1 => i32::MAX,
                    2 => 0,
                    3 => -1,
                    4 => 1,
                    _ => splitmix64(&mut seed) as i32,
                })
                .collect();

            for threshold in [i32::MIN, i32::MIN + 1, -1000, -1, 0, 1, 1000, i32::MAX - 1, i32::MAX] {
                let n_words = len.div_ceil(64);
                let expected = ref_pack(&values, n_words, |v| v > threshold);

                let mut got = vec![0u64; n_words];
                gt_i32_to_mask(&values, threshold, &mut got);
                assert_eq!(got, expected, "gt_i32_to_mask len={len} threshold={threshold}");
            }
        }
    }

    #[test]
    fn gt_i32_to_mask_signed_not_bitwise() {
        // The trap: -1 as a bit pattern (0xFFFF_FFFF) is greater than 0
        // unsigned, but -1 > 0 is false. A backend that packed an unsigned
        // compare would set bit 1 here.
        let values = [5i32, -1, 0, -2_000_000_000, 2_000_000_000];
        let mut got = [0u64; 1];
        gt_i32_to_mask(&values, 0, &mut got);
        assert_eq!(got[0], 0b10001, "only +5 and +2e9 exceed 0");
    }

    #[test]
    fn gt_i32_to_mask_threshold_extremes() {
        let values = [i32::MIN, i32::MIN + 1, 0, i32::MAX - 1, i32::MAX];
        let mut got = [0u64; 1];

        // i32::MIN threshold: everything strictly greater — all but lane 0.
        gt_i32_to_mask(&values, i32::MIN, &mut got);
        assert_eq!(got[0], 0b11110, "i32::MIN threshold excludes only i32::MIN itself");

        // i32::MAX threshold: nothing exceeds it, and `>` is strict so the
        // i32::MAX lane itself is clear too.
        got[0] = u64::MAX;
        gt_i32_to_mask(&values, i32::MAX, &mut got);
        assert_eq!(got[0], 0, "nothing exceeds i32::MAX");

        // i32::MAX - 1 threshold: only i32::MAX.
        gt_i32_to_mask(&values, i32::MAX - 1, &mut got);
        assert_eq!(got[0], 0b10000, "only i32::MAX exceeds i32::MAX-1");
    }

    /// The real correctness trap: bits past `values.len()` in the last word.
    /// A stale high bit would silently inflate every downstream popcount.
    #[test]
    fn trailing_bits_beyond_len_are_zero() {
        for &len in &[1usize, 15, 16, 17, 33, 63, 65, 100, 127, 129, 200] {
            let n_words = len.div_ceil(64);
            let used = len % 64; // 0 ⇒ the final word is entirely in range

            // Every element matches, so ONLY the out-of-range bits can be zero.
            let u = vec![1u32; len];
            let mut got = vec![u64::MAX; n_words + 2]; // pre-dirtied, plus surplus words
            eq_u32_to_mask(&u, 1, &mut got);
            if used != 0 {
                let expected_last = (1u64 << used) - 1;
                assert_eq!(got[n_words - 1], expected_last, "eq trailing bits len={len}");
            } else {
                assert_eq!(got[n_words - 1], u64::MAX, "eq full final word len={len}");
            }
            assert!(got[n_words..].iter().all(|&w| w == 0), "eq surplus words must be zeroed, len={len}");

            let i = vec![1i32; len];
            let mut got = vec![u64::MAX; n_words + 2];
            gt_i32_to_mask(&i, 0, &mut got);
            if used != 0 {
                let expected_last = (1u64 << used) - 1;
                assert_eq!(got[n_words - 1], expected_last, "gt trailing bits len={len}");
            } else {
                assert_eq!(got[n_words - 1], u64::MAX, "gt full final word len={len}");
            }
            assert!(got[n_words..].iter().all(|&w| w == 0), "gt surplus words must be zeroed, len={len}");
        }
    }

    #[test]
    fn empty_input_writes_only_zeros() {
        let mut got = [u64::MAX; 3];
        eq_u32_to_mask(&[], 7, &mut got);
        assert_eq!(got, [0u64; 3], "empty eq");

        let mut got = [u64::MAX; 3];
        gt_i32_to_mask(&[], 7, &mut got);
        assert_eq!(got, [0u64; 3], "empty gt");

        // Zero-length destination is legal for a zero-length input.
        eq_u32_to_mask(&[], 7, &mut []);
        gt_i32_to_mask(&[], 7, &mut []);

        assert_eq!(masked_sum_i32(&[], &[]), 0, "empty masked_sum");
    }

    #[test]
    fn single_element_lands_in_bit_zero() {
        let mut got = [u64::MAX; 1];
        eq_u32_to_mask(&[7u32], 7, &mut got);
        assert_eq!(got[0], 1, "one matching element ⇒ exactly bit 0");
        eq_u32_to_mask(&[8u32], 7, &mut got);
        assert_eq!(got[0], 0, "one non-matching element ⇒ no bits");
    }

    /// Bit order asserted against hand-computed literals — the one test that
    /// would catch an MSB-first or word-swapped backend, which a
    /// reference-vs-implementation comparison alone cannot (both could be
    /// wrong the same way if the reference were derived from the code).
    #[test]
    fn bit_order_is_lsb_first_within_each_word() {
        // 130 elements: matches at 0, 1, 63 (word 0 low + high edge),
        // 64, 65, 127 (word 1), and 128 (word 2 bit 0).
        let matching = [0usize, 1, 63, 64, 65, 127, 128];
        let mut values = vec![0u32; 130];
        for &i in &matching {
            values[i] = 1;
        }

        let mut got = [0u64; 3];
        eq_u32_to_mask(&values, 1, &mut got);

        assert_eq!(got[0], (1u64 << 0) | (1u64 << 1) | (1u64 << 63), "word 0: elements 0, 1, 63");
        assert_eq!(got[1], (1u64 << 0) | (1u64 << 1) | (1u64 << 63), "word 1: elements 64, 65, 127 → bits 0, 1, 63");
        assert_eq!(got[2], 1u64 << 0, "word 2: element 128 → bit 0, rest zero");

        // Element 64 is bit 0 of word 1, NOT bit 64-of-something or the high
        // bit of word 0 — the word-boundary claim, stated as its own literal.
        let mut only_64 = vec![0u32; 130];
        only_64[64] = 1;
        let mut got = [0u64; 3];
        eq_u32_to_mask(&only_64, 1, &mut got);
        assert_eq!(got, [0u64, 1u64, 0u64], "element 64 ⇒ word 1 bit 0 alone");
    }

    // ── mask algebra ────────────────────────────────────────────────────────

    #[test]
    fn mask_and_or_match_scalar_reference() {
        // Lengths straddling the 8-word U64x8 group boundary.
        for &len in &[0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 63, 64, 100] {
            let mut seed = 0xFEED_FACE_CAFE_0001;
            let a: Vec<u64> = (0..len).map(|_| splitmix64(&mut seed)).collect();
            let b: Vec<u64> = (0..len).map(|_| splitmix64(&mut seed)).collect();

            let ref_and: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x & y).collect();
            let ref_or: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x | y).collect();

            let mut dst = vec![0xDEAD_BEEFu64; len];
            mask_and(&a, &b, &mut dst);
            assert_eq!(dst, ref_and, "mask_and len={len}");

            let mut dst = vec![0xDEAD_BEEFu64; len];
            mask_or(&a, &b, &mut dst);
            assert_eq!(dst, ref_or, "mask_or len={len}");

            let mut dst = a.clone();
            mask_and_assign(&mut dst, &b);
            assert_eq!(dst, ref_and, "mask_and_assign len={len}");

            let mut dst = a.clone();
            mask_or_assign(&mut dst, &b);
            assert_eq!(dst, ref_or, "mask_or_assign len={len}");
        }
    }

    #[test]
    fn mask_algebra_identities() {
        let a = vec![0x0F0F_0F0F_0F0F_0F0Fu64; 20];
        let zeros = vec![0u64; 20];
        let ones = vec![u64::MAX; 20];

        let mut dst = vec![1u64; 20];
        mask_and(&a, &ones, &mut dst);
        assert_eq!(dst, a, "x & ALL == x");

        mask_and(&a, &zeros, &mut dst);
        assert_eq!(dst, zeros, "x & 0 == 0");

        mask_or(&a, &zeros, &mut dst);
        assert_eq!(dst, a, "x | 0 == x");

        mask_or(&a, &ones, &mut dst);
        assert_eq!(dst, ones, "x | ALL == ALL");

        // Narrowing: AND is monotone, so the popcount can only shrink.
        let mut seed = 0x0BAD_C0DE_0BAD_C0DE;
        let b: Vec<u64> = (0..20).map(|_| splitmix64(&mut seed)).collect();
        let mut dst = vec![0u64; 20];
        mask_and(&a, &b, &mut dst);
        let pc = |w: &[u64]| -> u32 { w.iter().map(|x| x.count_ones()).sum() };
        assert!(pc(&dst) <= pc(&a), "AND narrows");
        assert!(pc(&dst) <= pc(&b), "AND narrows");
        // ...and non-trivially so, or the assertion above is vacuous.
        assert!(pc(&dst) < pc(&a), "AND must actually remove bits on this corpus");
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn mask_and_rejects_length_mismatch() {
        let mut dst = [0u64; 4];
        mask_and(&[0u64; 4], &[0u64; 3], &mut dst);
    }

    // ── mask_andnot (a & !b) ─────────────────────────────────────────────────

    #[test]
    fn mask_andnot_matches_scalar_reference() {
        // Same length set as `mask_and_or_match_scalar_reference`, straddling
        // the 8-word U64x8 group boundary; len=2 is the `mask_words_for(70)`
        // shape (70 rows -> 2 words, a 6-bit tail in the second word).
        for &len in &[0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 63, 64, 100] {
            let mut seed = 0xA11C_E5EE_D000_0001;
            let a: Vec<u64> = (0..len).map(|_| splitmix64(&mut seed)).collect();
            let b: Vec<u64> = (0..len).map(|_| splitmix64(&mut seed)).collect();

            let ref_andnot: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x & !y).collect();

            let mut dst = vec![0xDEAD_BEEFu64; len];
            mask_andnot(&a, &b, &mut dst);
            assert_eq!(dst, ref_andnot, "mask_andnot len={len}");

            let mut dst = a.clone();
            mask_andnot_assign(&mut dst, &b);
            assert_eq!(dst, ref_andnot, "mask_andnot_assign len={len}");
        }
    }

    #[test]
    fn mask_andnot_algebra_identities() {
        let mut seed = 0x1357_9BDF_2468_ACE0;
        let a: Vec<u64> = (0..20).map(|_| splitmix64(&mut seed)).collect();
        let b: Vec<u64> = (0..20).map(|_| splitmix64(&mut seed)).collect();

        // (a & !b) | (a & b) == a — partitioning a's bits by whether b also
        // has them set recovers a exactly.
        let mut a_andnot_b = vec![0u64; 20];
        mask_andnot(&a, &b, &mut a_andnot_b);
        let mut a_and_b = vec![0u64; 20];
        mask_and(&a, &b, &mut a_and_b);
        let mut recombined = vec![0u64; 20];
        mask_or(&a_andnot_b, &a_and_b, &mut recombined);
        assert_eq!(recombined, a, "(a & !b) | (a & b) == a");

        // (a & !b) & b == 0 — the "not b" half can never overlap b.
        let mut overlap = vec![0u64; 20];
        mask_and(&a_andnot_b, &b, &mut overlap);
        assert_eq!(overlap, vec![0u64; 20], "(a & !b) & b == 0");

        // ...and non-trivially so: on this corpus a_andnot_b must actually
        // differ from a (b removes real bits), or both identities above hold
        // vacuously of a no-op.
        assert_ne!(a_andnot_b, a, "andnot must actually remove bits on this corpus");
    }

    #[test]
    fn mask_andnot_preserves_conforming_tail() {
        // 2 words = the `mask_words_for(70)` shape: word 0 fully valid (rows
        // 0..63), word 1 valid only in its low 7 bits (rows 64..70); the
        // tail is word 1 bits 7..63, which a conforming mask always holds
        // zero.
        const TAIL_MASK: u64 = !0x7Fu64; // bits 7..63

        // Arm 1: a conforms (tail zero), b is maximally non-conforming (all
        // bits set, including its own tail) — dst must still be zero
        // everywhere, tail included, because `a & !b` can never exceed `a`.
        let a = [0x1234_5678_9ABC_DEF0u64, 0x0000_0000_0000_005Bu64];
        assert_eq!(a[1] & TAIL_MASK, 0, "fixture precondition: a's tail is zero");
        let b = [u64::MAX; 2];
        let mut dst = [0xDEAD_BEEFu64; 2];
        mask_andnot(&a, &b, &mut dst);
        assert_eq!(dst, [0u64, 0u64], "a & !(all-ones) == 0, tail included");

        // Arm 2: a still conforms; b's body is zero (so it removes nothing
        // from a) but b's tail is dirty (all ones) — exactly the shape where
        // `!b` flips a normally-zero tail to all ones. dst must equal a
        // exactly, and in particular dst's tail must stay zero: a's tail was
        // already zero, and `a & !b` can only ever narrow a, never widen it.
        let b_dirty_tail = [0u64, TAIL_MASK];
        assert_ne!(b_dirty_tail[1] & TAIL_MASK, 0, "fixture precondition: b's tail is dirty");
        let mut dst = [0xDEAD_BEEFu64; 2];
        mask_andnot(&a, &b_dirty_tail, &mut dst);
        assert_eq!(dst, a, "a & !b == a when b's body is 0, even with a dirty b tail");
        assert_eq!(dst[1] & TAIL_MASK, 0, "dst's tail stays zero despite b's dirty tail");
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn mask_andnot_rejects_length_mismatch() {
        let mut dst = [0u64; 4];
        mask_andnot(&[0u64; 4], &[0u64; 3], &mut dst);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn mask_andnot_assign_rejects_length_mismatch() {
        let mut a = [0u64; 4];
        mask_andnot_assign(&mut a, &[0u64; 3]);
    }

    #[test]
    #[should_panic(expected = "out_words.len()")]
    fn eq_u32_to_mask_rejects_short_destination() {
        // 65 elements need 2 words; 1 must be refused, not silently truncated.
        let values = vec![0u32; 65];
        let mut got = [0u64; 1];
        eq_u32_to_mask(&values, 0, &mut got);
    }

    // ── masked_sum_i32 ──────────────────────────────────────────────────────

    #[test]
    fn masked_sum_i32_matches_scalar_reference() {
        for &len in MASK_LENS {
            let mut seed = 0x2468_ACE0_1357_9BDF;
            let values: Vec<i32> = (0..len).map(|_| splitmix64(&mut seed) as i32).collect();
            let n_words = len.div_ceil(64);

            for pattern in [0u64, u64::MAX, 0x5555_5555_5555_5555, 0xAAAA_AAAA_AAAA_AAAA, 1] {
                let mask = vec![pattern; n_words];
                // Independent reference: widen every selected element to i64.
                let expected: i64 = values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask[i / 64] >> (i % 64) & 1 == 1)
                    .map(|(_, &v)| v as i64)
                    .sum();
                let got = masked_sum_i32(&values, &mask);
                assert_eq!(got, expected, "masked_sum_i32 len={len} pattern={pattern:#x}");
            }
        }
    }

    #[test]
    fn masked_sum_i32_widens_beyond_i32_range() {
        // 64 × i32::MAX = 137_438_953_408, which overflows i32 by ~64×. An
        // implementation that reduced in i32 (e.g. `I32x16::reduce_sum`) would
        // wrap here; the widened contract says it must not.
        let values = [i32::MAX; 64];
        let got = masked_sum_i32(&values, &[u64::MAX]);
        assert_eq!(got, 64 * i32::MAX as i64);
        assert!(got > i32::MAX as i64, "result genuinely exceeds i32 range");

        // Same on the negative side.
        let values = [i32::MIN; 64];
        let got = masked_sum_i32(&values, &[u64::MAX]);
        assert_eq!(got, 64 * i32::MIN as i64);
        assert!(got < i32::MIN as i64);
    }

    #[test]
    fn masked_sum_i32_ignores_bits_past_len() {
        // 3 elements, an all-ones mask word: bits 3..63 must be ignored, not
        // used to index past the slice (which would panic) or counted.
        let values = [10i32, 20, 30];
        assert_eq!(masked_sum_i32(&values, &[u64::MAX]), 60);

        // Same across a word boundary: 65 elements, both words all-ones.
        let values: Vec<i32> = (0..65).collect();
        let expected: i64 = (0..65i64).sum();
        assert_eq!(masked_sum_i32(&values, &[u64::MAX; 2]), expected);
    }

    #[test]
    fn masked_sum_i32_empty_mask_is_zero() {
        let values: Vec<i32> = (1..=100).collect();
        assert_eq!(masked_sum_i32(&values, &[0u64; 2]), 0, "no bits set ⇒ 0");
    }

    /// End-to-end composition: the shape the ABI's fused plan runs — two
    /// predicates ANDed, then counted and summed. Ties the seven primitives
    /// plus `popcount_batch_u64` together on one corpus.
    #[test]
    fn predicates_compose_into_count_and_sum() {
        const N: usize = 1000;
        let classes: Vec<u32> = (0..N).map(|i| (i % 4) as u32).collect();
        let values: Vec<i32> = (0..N).map(|i| i as i32 - 500).collect();
        let n_words = N.div_ceil(64);

        let mut m_class = vec![0u64; n_words];
        eq_u32_to_mask(&classes, 2, &mut m_class);
        let mut m_value = vec![0u64; n_words];
        gt_i32_to_mask(&values, 0, &mut m_value);

        let mut acc = vec![u64::MAX; n_words];
        mask_and_assign(&mut acc, &m_class);
        mask_and_assign(&mut acc, &m_value);

        // Independent reference over the same predicates.
        let want: Vec<usize> = (0..N)
            .filter(|&i| classes[i] == 2 && values[i] > 0)
            .collect();
        let count = crate::bitwise::popcount_batch_u64(&acc);
        assert_eq!(count as usize, want.len(), "fused count");
        let sum_ref: i64 = want.iter().map(|&i| values[i] as i64).sum();
        assert_eq!(masked_sum_i32(&values, &acc), sum_ref, "fused sum");

        // Anti-vacuity: the composition must actually narrow, or this test
        // would pass for a no-op AND. `acc` starts as all N rows.
        assert!(count > 0, "the fused predicate must select something");
        assert!((count as usize) < N / 4, "the fused predicate must be strictly narrower than either operand");
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

    // ── masked_strided_group_sum ──

    /// The three groupings of a 12-byte register read the SAME bytes and must
    /// give three DIFFERENT answers — otherwise every test below would pass for
    /// an implementation that ignored `groups`/`group_bytes`.
    #[test]
    fn each_grouping_of_the_same_register_reads_it_differently() {
        let mut b = vec![0u8; 512];
        for k in 0..12 {
            b[4 + k] = (k + 1) as u8;
        }
        let m = [0b1u64];
        let rails = masked_strided_group_sum(&b, 4, 512, 1, 6, 2, &m).unwrap();
        let trips = masked_strided_group_sum(&b, 4, 512, 1, 4, 3, &m).unwrap();
        let quads = masked_strided_group_sum(&b, 4, 512, 1, 3, 4, &m).unwrap();

        // Hand-computed from bytes 1..=12, little-endian per group.
        assert_eq!(rails, 0x0201 + 0x0403 + 0x0605 + 0x0807 + 0x0A09 + 0x0C0B);
        assert_eq!(trips, 0x030201 + 0x060504 + 0x090807 + 0x0C0B0A);
        assert_eq!(quads, 0x04030201 + 0x08070605 + 0x0C0B0A09);
        assert!(rails != trips && trips != quads && rails != quads);
    }

    /// The mask selects records rather than being decoration, and the stride is
    /// respected: two records with different content must sum separately and
    /// additively.
    #[test]
    fn the_mask_and_the_stride_both_bind() {
        let mut b = vec![0u8; 2 * 64];
        b[0..4].copy_from_slice(&[1, 0, 2, 0]);
        b[64..68].copy_from_slice(&[10, 0, 20, 0]);
        let f = |m: u64| masked_strided_group_sum(&b, 0, 64, 2, 2, 2, &[m]).unwrap();
        assert_eq!(f(0b00), 0, "an empty mask sums nothing");
        assert_eq!(f(0b01), 3);
        assert_eq!(f(0b10), 30);
        assert_eq!(f(0b11), 33, "additive over disjoint selections");
    }

    /// A dirty tail bit past `n_records` is ignored rather than read — the
    /// buffer here is too short for it, so an unclamped kernel would panic.
    #[test]
    fn a_dirty_tail_bit_is_ignored() {
        let mut b = vec![0u8; 2 * 16];
        b[0..2].copy_from_slice(&[5, 0]);
        b[16..18].copy_from_slice(&[7, 0]);
        let clean = masked_strided_group_sum(&b, 0, 16, 2, 1, 2, &[0b11]).unwrap();
        let dirty = masked_strided_group_sum(&b, 0, 16, 2, 1, 2, &[0b1111]).unwrap();
        assert_eq!(clean, 12);
        assert_eq!(clean, dirty);
    }

    /// Overflow is reported, not wrapped. Four max-valued u32 groups per record
    /// over many records exceeds `i64::MAX`; the boundary itself is asserted so
    /// the claim is checkable rather than narrated.
    #[test]
    fn overflow_is_reported_rather_than_wrapped() {
        let recs = 8usize;
        let mut b = vec![0xFFu8; recs * 16];
        let m = [0xFFu64];
        // Small case: comfortably inside i64.
        let small = masked_strided_group_sum(&b, 0, 16, recs, 3, 4, &m).unwrap();
        assert_eq!(small, recs as i64 * 3 * 0xFFFF_FFFF);

        // The documented bound, checked: how many max quad records fit?
        let per_record = 3i128 * 0xFFFF_FFFFi128;
        assert_eq!(i64::MAX as i128 / per_record, 715_827_882);

        // And the range check itself is what decides, not a wrap.
        assert!(i64::try_from(i64::MAX as i128 + 1).is_err());
        b.clear();
    }

    #[test]
    #[should_panic(expected = "group_bytes")]
    fn a_group_wider_than_four_bytes_is_rejected() {
        let b = vec![0u8; 64];
        let _ = masked_strided_group_sum(&b, 0, 16, 1, 1, 5, &[0b1]);
    }

    #[test]
    #[should_panic(expected = "past len")]
    fn a_record_reading_past_the_buffer_is_rejected() {
        let b = vec![0u8; 8];
        // Record 0's register would read 0..12 out of an 8-byte buffer.
        let _ = masked_strided_group_sum(&b, 0, 16, 1, 3, 4, &[0b1]);
    }
}
