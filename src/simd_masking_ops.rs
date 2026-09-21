//! Packed-bitmask predicates, mask algebra, and masked reductions — the
//! **ergonomic masking layer** of the SIMD stack.
//!
//! # Where this sits (the three-layer contract, operator-ruled 2026-09-13)
//!
//! ```text
//!   consumers (lance-graph-mask-risc, lgj-abi kernels, planner)
//!            │  semantic ops: TERNLOG<IMM>, AND, XOR, COUNT, eq→mask …
//!            ▼
//!   simd_masking_ops.rs   ← THIS FILE: slice/chunk/tail ergonomics, in-place
//!            │             forms, mask composition, masked reductions
//!            ▼
//!   simd.rs               architecture-agnostic lane types (U64x8, U32x16, …)
//!            │  compile-time backend selection
//!            ▼
//!   simd_{avx512,avx2,neon,wasm,scalar}.rs   each owns its realization
//! ```
//!
//! The rule that keeps the layers honest: **no backend semantics live here.**
//! This file composes lane-level primitives (`U32x16::eq_bitmask`,
//! `U64x8::ternlog::<IMM>`, `I32x16::gt_bitmask`, …) into slice-level
//! machinery — it never branches on an ISA, never names an intrinsic, and
//! never carries a per-architecture cost model. What `U64x8::ternlog` *is* on
//! AVX2 versus AVX-512 versus NEON is entirely the corresponding backend
//! file's business. Conversely, chunking, tail handling, reusable-destination
//! (`*_assign`) forms, and fused convenience compositions belong HERE and
//! never in a backend.
//!
//! Sibling of [`crate::simd_int_ops`] (integer arithmetic / conversion), split
//! out so masking is a first-class execution family rather than a collection
//! of functions that accumulated inside integer ops. Every `pub fn` is
//! re-exported through [`crate::simd`]; consumers import from there.
//!
//! # Bit order (NORMATIVE)
//!
//! Element `i` lives at bit `i % 64` of word `i / 64`, LSB-first; every bit
//! at or past the element count is zero. The full statement, and why it is
//! structural rather than a tail special case, is in the section header below.
#![forbid(unsafe_code)]

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
// ONE exception, named here so the guarantee stays checkable: the lattice
// shift `mask_shift_morton` is an ACCUMULATOR (`dst |= ...`) and is
// word-indexed, not element-indexed — it has no `values.len()` and no
// trailing-bit law, so the bit-order paragraph above does not apply to it.
// A caller that wants a fresh answer zeroes `dst` first. Every other writer
// in this file obeys the paragraph above.
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

// Every contiguous LANE op below — the predicate builders over `&[u32]` /
// `&[i32]` and the word-algebra ops over `&[u64]` — walks its input with
// `slice::as_chunks::<LANES>()` (stable since 1.88): the main body iterates
// `&[[T; LANES]]` and feeds each chunk to `from_array` — a fixed-size load
// with no per-chunk bounds check and no `g * L` index arithmetic for LLVM to
// prove away — and the remainder is the EXACT tail slice. That tail is NOT
// peeled as a scalar loop: it is zero-padded into one register (`pad_tail`)
// and run through the SAME packed op as the body, with padding lanes never
// written back. (The strided byte-gather ops, the popcount-driven folds, and
// `blend_i32` are NOT in this family and say so in their own docs.)
//
// Measured reason (codegen witness, 2026-09-14): with an exact-length scalar
// tail LLVM fully unrolled it on aarch64 into 7 × (and, orr) on GPRs, which
// with the 2 index-mask ops read as 16 GPR logic ops in a facade op whose
// contract is "packed on every backend", while on AVX2 the same loop became
// `vpmaskmovq` masked vectors. Padding makes both arms the same shape: packed
// body, packed tail. What REMAINS is not zero: 4 GPR logic ops on v3 and 2 on
// aarch64, hand-classified as length/index arithmetic (`andl $7`, `& !63`),
// not lane data — the witness bounds their COUNT (`SLICE_GPR_CAP`), it does
// not classify them. What was measured is that count; no throughput
// comparison against the old peel has been made (the tail is a zero-init +
// two bounded `copy_from_slice`s + one packed op, and for inputs shorter than
// one register it IS the whole operation).
//
// `from_array` (not `from_slice`) because it exists on every backend's
// `U32x16`/`I32x16`/`U64x8` — the NEON and wasm `[..x4; 4]` fan-outs expose
// no `from_slice` — so the loops stay free of any `cfg(target_arch)`.

/// Zero-pad a `< N`-element tail into one full register's worth of lanes so
/// the tail runs through the same packed op as the body. Padding lanes are
/// never written back: word-op callers copy out exactly `tail.len()` results,
/// predicate callers mask the bitmask down with [`tail_lane_bits`].
#[inline(always)]
fn pad_tail<T: Copy + Default, const N: usize>(tail: &[T]) -> [T; N] {
    let mut lanes = [T::default(); N];
    lanes[..tail.len()].copy_from_slice(tail);
    lanes
}

/// The low `n` bits set (`n < 16`): the lane-validity mask for a padded
/// 16-lane predicate tail, so a padding lane can never contribute a match.
#[inline(always)]
fn tail_lane_bits(n: usize) -> u16 {
    debug_assert!(n < 16, "a tail is shorter than one register");
    ((1u32 << n) - 1) as u16
}

/// The low `n` bits set (`n < 8`): the lane-validity mask for a padded
/// 8-lane predicate tail, so a padding lane can never contribute a match.
///
/// Sibling of [`tail_lane_bits`], which serves the 16-lane registers. Kept
/// separate rather than made generic because the return width IS the
/// register's lane count, and the caller ORs the result into a byte-shifted
/// slot whose width has to match.
#[inline(always)]
fn tail_lane_bits_8(n: usize) -> u8 {
    debug_assert!(n < 8, "a tail is shorter than one register");
    ((1u16 << n) - 1) as u8
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
/// Runs 16 lanes at a time through [`crate::simd::U32x16::eq_bitmask`]; the
/// final partial group is zero-padded into one register and run through the
/// same packed compare, with the padding lanes' bits masked off — no scalar
/// tail, so the tail cannot disagree with the body.
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
    let needle_v = crate::simd::U32x16::splat(needle);
    pack::<u32, 16>("eq_u32_to_mask", values, out_words, |lanes, live| {
        (crate::simd::U32x16::from_array(lanes).eq_bitmask(needle_v) & live16(live)) as u64
    });
}

/// Packs `read_le_u32(bytes, first_offset + i * stride_bytes) == needle` into
/// `out_words`, one bit per element, LSB-first within each `u64` word — the
/// **strided** sibling of [`eq_u32_to_mask`], for scanning one `u32` field of
/// an AoS/facet row layout (e.g. a 4-byte classid at a fixed offset inside a
/// 512-byte row) without gathering the column into a contiguous copy first.
///
/// Element `i` is the little-endian `u32` at byte offset
/// `first_offset + i * stride_bytes`. `stride_bytes == 4` reads a contiguous
/// `u32` column and takes a dedicated contiguous path (one 64-byte window per
/// 16 elements — the same load shape as [`eq_u32_to_mask`]); `stride_bytes
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
    if stride_bytes == 4 {
        // Contiguous lane (a facet-major column): 16 elements are ONE 64-byte
        // window, so the full groups are exactly the `as_chunks::<64>()` of
        // the byte range they cover — fixed-size windows with no per-element
        // bounds check (bounds were proven above for the last element, so the
        // sub-slice cannot panic). The `[u32; 16]` built from each window is
        // the same register-sized temporary the general path uses; what this
        // removes is the checks, not the temporary.
        let (windows, _) = bytes[first_offset..first_offset + groups * 64].as_chunks::<64>();
        for (g, window) in windows.iter().enumerate() {
            let lanes: [u32; 16] = core::array::from_fn(|k| {
                u32::from_le_bytes([window[4 * k], window[4 * k + 1], window[4 * k + 2], window[4 * k + 3]])
            });
            let bits = crate::simd::U32x16::from_array(lanes).eq_bitmask(needle_v);
            out_words[g / 4] |= (bits as u64) << ((g % 4) * 16);
        }
    } else {
        for g in 0..groups {
            let base = first_offset + g * 16 * stride_bytes;
            let lanes: [u32; 16] = core::array::from_fn(|k| read_le_u32(bytes, base + k * stride_bytes));
            let bits = crate::simd::U32x16::from_array(lanes).eq_bitmask(needle_v);
            out_words[g / 4] |= (bits as u64) << ((g % 4) * 16);
        }
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
/// Runs 16 lanes at a time through [`crate::simd::I32x16::gt_bitmask`]; the
/// final partial group is zero-padded into one register and run through the
/// same packed compare, with the padding lanes' bits masked off.
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
    let threshold_v = crate::simd::I32x16::splat(threshold);
    pack::<i32, 16>("gt_i32_to_mask", values, out_words, |lanes, live| {
        (crate::simd::I32x16::from_array(lanes).gt_bitmask(threshold_v) & live16(live)) as u64
    });
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
    const L: usize = crate::simd::U64x8::LANES;
    let (ca, ta) = a.as_chunks::<L>();
    let (cb, tb) = b.as_chunks::<L>();
    let (cd, td) = dst.as_chunks_mut::<L>();
    for ((x, y), d) in ca.iter().zip(cb).zip(cd.iter_mut()) {
        let va = crate::simd::U64x8::from_array(*x);
        let vb = crate::simd::U64x8::from_array(*y);
        *d = (va & vb).to_array();
    }
    if !ta.is_empty() {
        let va = crate::simd::U64x8::from_array(pad_tail(ta));
        let vb = crate::simd::U64x8::from_array(pad_tail(tb));
        td.copy_from_slice(&(va & vb).to_array()[..td.len()]);
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
    const L: usize = crate::simd::U64x8::LANES;
    let (ca, ta) = a.as_chunks::<L>();
    let (cb, tb) = b.as_chunks::<L>();
    let (cd, td) = dst.as_chunks_mut::<L>();
    for ((x, y), d) in ca.iter().zip(cb).zip(cd.iter_mut()) {
        let va = crate::simd::U64x8::from_array(*x);
        let vb = crate::simd::U64x8::from_array(*y);
        *d = (va | vb).to_array();
    }
    if !ta.is_empty() {
        let va = crate::simd::U64x8::from_array(pad_tail(ta));
        let vb = crate::simd::U64x8::from_array(pad_tail(tb));
        td.copy_from_slice(&(va | vb).to_array()[..td.len()]);
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
    const L: usize = crate::simd::U64x8::LANES;
    let (cd, td) = dst.as_chunks_mut::<L>();
    let (cs, ts) = src.as_chunks::<L>();
    for (d, s) in cd.iter_mut().zip(cs) {
        let vd = crate::simd::U64x8::from_array(*d);
        let vs = crate::simd::U64x8::from_array(*s);
        *d = (vd & vs).to_array();
    }
    if !td.is_empty() {
        let vd = crate::simd::U64x8::from_array(pad_tail(td));
        let vs = crate::simd::U64x8::from_array(pad_tail(ts));
        td.copy_from_slice(&(vd & vs).to_array()[..td.len()]);
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
    const L: usize = crate::simd::U64x8::LANES;
    let (cd, td) = dst.as_chunks_mut::<L>();
    let (cs, ts) = src.as_chunks::<L>();
    for (d, s) in cd.iter_mut().zip(cs) {
        let vd = crate::simd::U64x8::from_array(*d);
        let vs = crate::simd::U64x8::from_array(*s);
        *d = (vd | vs).to_array();
    }
    if !td.is_empty() {
        let vd = crate::simd::U64x8::from_array(pad_tail(td));
        let vs = crate::simd::U64x8::from_array(pad_tail(ts));
        td.copy_from_slice(&(vd | vs).to_array()[..td.len()]);
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
    const L: usize = crate::simd::U64x8::LANES;
    let (ca, ta) = a.as_chunks::<L>();
    let (cb, tb) = b.as_chunks::<L>();
    let (cd, td) = dst.as_chunks_mut::<L>();
    for ((x, y), d) in ca.iter().zip(cb).zip(cd.iter_mut()) {
        let va = crate::simd::U64x8::from_array(*x);
        let vb = crate::simd::U64x8::from_array(*y);
        *d = (va & !vb).to_array();
    }
    if !ta.is_empty() {
        let va = crate::simd::U64x8::from_array(pad_tail(ta));
        let vb = crate::simd::U64x8::from_array(pad_tail(tb));
        td.copy_from_slice(&(va & !vb).to_array()[..td.len()]);
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
    const L: usize = crate::simd::U64x8::LANES;
    let (ca, ta) = a.as_chunks_mut::<L>();
    let (cb, tb) = b.as_chunks::<L>();
    for (x, y) in ca.iter_mut().zip(cb) {
        let va = crate::simd::U64x8::from_array(*x);
        let vb = crate::simd::U64x8::from_array(*y);
        *x = (va & !vb).to_array();
    }
    if !ta.is_empty() {
        let va = crate::simd::U64x8::from_array(pad_tail(ta));
        let vb = crate::simd::U64x8::from_array(pad_tail(tb));
        ta.copy_from_slice(&(va & !vb).to_array()[..ta.len()]);
    }
}

/// `dst = ternlog::<IMM>(a, b, c)`, elementwise over `u64` mask words — any
/// 3-input Boolean function of three masks in one pass.
///
/// `IMM` is the 8-bit truth table in Intel's VPTERNLOG convention (index
/// `(a<<2)|(b<<1)|c`, result bit `(IMM >> index) & 1`); the named tables in
/// [`crate::simd::ternlog`] (`AND3`, `OR3`, `MAJ3`, `AND2_ANDNOT`, …) are
/// the sanctioned spellings. This is the mask-op family's general member:
/// [`mask_and`] is `mask_ternlog::<{ ternlog::AND2 }>` with `c` ignored,
/// [`mask_andnot`] is `AND2_ANDNOT` with `c` ignored, and the composed
/// `a & b & c` that a consumer would otherwise spell as two `mask_and_assign`
/// passes through a scratch buffer is ONE `AND3` pass here — one
/// `VPTERNLOGQ` per 512 bits on AVX-512, the polyfill elsewhere.
///
/// # Tail-bit semantics
///
/// Whether `dst`'s tail conforms depends on the truth table, not on the
/// inputs alone: the tail of every conforming input is zero, so `dst`'s tail
/// is `IMM & 1` replicated — **zero iff `IMM` is even** (index 0 = all-zero
/// inputs maps to 0). Every named table in [`crate::simd::ternlog`] is even.
/// An odd `IMM` (one whose function is true of `(0,0,0)`) sets every tail bit
/// and the caller must clear the tail against its own known row count, exactly
/// as [`mask_or`] documents for a non-conforming operand. For the
/// subset-shaped tables (`AND3`, `AND2_ANDNOT`, `AND_ANDNOT2`, `AND2`) the
/// stronger [`mask_andnot`] guarantee also holds: the result is a bitwise
/// subset of `a`, so `dst`'s tail is zero whenever `a`'s is, regardless of
/// `b` and `c`.
///
/// `dst` must not overlap `a`, `b` or `c`; use [`mask_ternlog_assign`] for
/// the in-place case.
///
/// # Panics
///
/// Panics unless `a.len() == b.len() == c.len() == dst.len()`.
#[inline]
pub fn mask_ternlog<const IMM: i32>(a: &[u64], b: &[u64], c: &[u64], dst: &mut [u64]) {
    assert_eq!(a.len(), b.len(), "mask_ternlog: a/b length mismatch");
    assert_eq!(a.len(), c.len(), "mask_ternlog: a/c length mismatch");
    assert_eq!(a.len(), dst.len(), "mask_ternlog: a/dst length mismatch");
    const L: usize = crate::simd::U64x8::LANES;
    let (ca, ta) = a.as_chunks::<L>();
    let (cb, tb) = b.as_chunks::<L>();
    let (cc, tc) = c.as_chunks::<L>();
    let (cd, td) = dst.as_chunks_mut::<L>();
    for (((x, y), z), d) in ca.iter().zip(cb).zip(cc).zip(cd.iter_mut()) {
        let va = crate::simd::U64x8::from_array(*x);
        let vb = crate::simd::U64x8::from_array(*y);
        let vc = crate::simd::U64x8::from_array(*z);
        *d = va.ternlog::<IMM>(vb, vc).to_array();
    }
    if !ta.is_empty() {
        let va = crate::simd::U64x8::from_array(pad_tail(ta));
        let vb = crate::simd::U64x8::from_array(pad_tail(tb));
        let vc = crate::simd::U64x8::from_array(pad_tail(tc));
        td.copy_from_slice(&va.ternlog::<IMM>(vb, vc).to_array()[..td.len()]);
    }
}

/// `a = ternlog::<IMM>(a, b, c)`, elementwise over `u64` mask words.
///
/// The in-place form of [`mask_ternlog`] — `a` is the first truth-table
/// operand AND the destination, which is the shape a fused predicate plan
/// wants when narrowing an accumulator against two more masks in one pass
/// (`selected = selected & src & gate` as `AND3`). Same tail contract as
/// [`mask_ternlog`].
///
/// # Panics
///
/// Panics unless `a.len() == b.len() == c.len()`.
#[inline]
pub fn mask_ternlog_assign<const IMM: i32>(a: &mut [u64], b: &[u64], c: &[u64]) {
    assert_eq!(a.len(), b.len(), "mask_ternlog_assign: a/b length mismatch");
    assert_eq!(a.len(), c.len(), "mask_ternlog_assign: a/c length mismatch");
    const L: usize = crate::simd::U64x8::LANES;
    let (ca, ta) = a.as_chunks_mut::<L>();
    let (cb, tb) = b.as_chunks::<L>();
    let (cc, tc) = c.as_chunks::<L>();
    for ((x, y), z) in ca.iter_mut().zip(cb).zip(cc) {
        let va = crate::simd::U64x8::from_array(*x);
        let vb = crate::simd::U64x8::from_array(*y);
        let vc = crate::simd::U64x8::from_array(*z);
        *x = va.ternlog::<IMM>(vb, vc).to_array();
    }
    if !ta.is_empty() {
        let va = crate::simd::U64x8::from_array(pad_tail(ta));
        let vb = crate::simd::U64x8::from_array(pad_tail(tb));
        let vc = crate::simd::U64x8::from_array(pad_tail(tc));
        ta.copy_from_slice(&va.ternlog::<IMM>(vb, vc).to_array()[..ta.len()]);
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

// ────────────────────────────────────────────────────────────────────────
// Data-indexed mask primitives: gather, scatter-OR, keyed group-sum.
//
// Every op above this point walks its input in LANE order (predicate
// builders) or POPCOUNT order (masked reductions) — the address of the next
// element to touch is a fixed stride or the next set bit, never a value read
// out of another array. The three functions below are the opposite shape:
// each is a permutation or scatter-add whose per-element address comes from
// a DATA array (`index` or `keys`), so none of them vector-loads and each
// says so in its own doc, matching `masked_strided_group_sum` above. None of
// the three takes a `_under` gate — a gate composes on the *result* mask via
// `mask_and`/`mask_ternlog`, not as an extra parameter on a data-indexed op.
// ────────────────────────────────────────────────────────────────────────

/// Gather bits by row index: element `i` of the result is
/// `index[i] < src_rows && bit index[i] of src`.
///
/// A bit of a RESIDENT mask read through an index lane: `index` has one
/// entry per element of THIS population, each an address into the
/// population `src` is defined over, and the result is a mask over THIS
/// population, set iff the addressed bit of `src` is.
///
/// **Survival condition.** `src` must be state the caller already holds
/// (a resident plane) — never a mask produced only so that this call can
/// read it. A population-sized mask that exists to feed the next fold is
/// forbidden intermediate state; an addressed read is legal only when its
/// source is resident and its output is one tile of the caller's own
/// scratch. A predicate through an index lane belongs in
/// [`eq_u32_via_to_mask`], which never builds the source mask at all.
///
/// **An out-of-range index is FALSE by contract, not an error** — the V3
/// zero-fallback rule applied to addressing: an index naming no row in
/// `src` (`index[i] >= src_rows`) resolves to "no match", the same way an
/// address naming no element resolves to "no element" rather than
/// panicking. A caller that needs a hard out-of-range to be visible builds its own
/// out-of-range mask separately (e.g. `ge_u32_to_mask(index, src_rows as
/// u32, ..)`); this primitive never raises it.
///
/// `out_words` is **fully overwritten**, not OR-ed into: every bit at or
/// past `index.len()` — the tail of the last live word and every surplus
/// word — is written `0`, exactly as every predicate builder above.
///
/// # Why this lives HERE
///
/// It is a permutation indexed by `index`, not a fixed stride or a
/// popcount walk — every other primitive above reads its input in lane or
/// set-bit order; this one reads ROW order but dereferences `src` at a
/// data-dependent bit position per row. That is a gather, and a consumer
/// hand-rolling `bit(src, index[i])` in a loop is exactly the polyfill
/// bypass the "all SIMD from `ndarray::simd`" invariant exists to prevent.
///
/// # Vectorisation, honestly
///
/// **Scalar, and by necessity, not oversight.** The address of the bit to
/// read is `index[i]`, a value out of another array — there is no vector
/// gather over individual BITS on any of this crate's backends (a
/// byte/word/dword gather exists; a bit gather does not), so "read one bit
/// at a data-dependent offset" is inherently scalar. Cost is
/// `O(index.len())`: unlike the popcount-driven reductions above, there is
/// no input mask to skip zero words against — every entry of `index` must
/// be consulted regardless of what `src` contains.
///
/// # Panics
///
/// Panics if `out_words.len() < index.len().div_ceil(64)`, or if
/// `src.len() < src_rows.div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::mask_gather_u32;
///
/// // src has rows 0 and 2 set, out of 3 rows.
/// let src = [0b101u64];
/// let index = [2u32, 1, 0, 5]; // row 5 is out of range
/// let mut out = [u64::MAX]; // dirty tail must be cleared
/// mask_gather_u32(&src, 3, &index, &mut out);
/// // bit 0 -> src row 2 (set), bit 1 -> src row 1 (unset),
/// // bit 2 -> src row 0 (set), bit 3 -> out of range (false)
/// assert_eq!(out[0], 0b0101);
/// ```
#[inline]
pub fn mask_gather_u32(src: &[u64], src_rows: usize, index: &[u32], out_words: &mut [u64]) {
    let n = index.len();
    let words = mask_words_for(n);
    assert!(
        out_words.len() >= words,
        "mask_gather_u32: out_words.len()={} < required {}",
        out_words.len(),
        words
    );
    let src_words = mask_words_for(src_rows);
    assert!(src.len() >= src_words, "mask_gather_u32: src.len()={} < required {}", src.len(), src_words);

    // Zero first, whole buffer: makes the "tail and surplus are 0" guarantee
    // structural, matching every predicate builder's `pack` convention.
    for w in out_words.iter_mut() {
        *w = 0;
    }
    for (w, out_word) in out_words.iter_mut().enumerate().take(words) {
        let base = w * 64;
        let live = (n - base).min(64);
        let mut acc = 0u64;
        for lane in 0..live {
            let idx = index[base + lane] as usize;
            if idx < src_rows && (src[idx / 64] >> (idx % 64)) & 1 == 1 {
                acc |= 1u64 << lane;
            }
        }
        *out_word = acc;
    }
}

/// Scatter-OR by row index: for every element `i` selected by `src` (bit `i`
/// set, `i < index.len()`), sets bit `index[i]` of the result — provided
/// `index[i] < out_rows`.
///
/// The inverse addressing of [`mask_gather_u32`]: that one reads through
/// `index`, this one writes through it — "targets some selected element
/// addresses" rather than "elements whose target is selected". Several
/// source elements may scatter to the same target bit; the contract is OR
/// (union), so repeats are harmless and order-independent — the same
/// reason `vsa_bundle`-shaped accumulation is safe under reordering.
///
/// **Survival condition.** `out_words` is the caller's DEMANDED sink — the
/// requested result is this mask, or this mask is the accumulator of the
/// fold whose scalar leaves (a distinct count is its popcount). It is never
/// a buffer another pass reads back as input; a scattered population mask
/// handed to a further fold is forbidden intermediate state.
///
/// `out_words[..out_rows.div_ceil(64)]` is **accumulated into, not
/// overwritten**: every scattered bit is OR-ed into whatever `out_words`
/// already holds, and a bit set before the call that this call does not
/// itself scatter to stays set. The caller zeroes `out_words` once before
/// the first call in a sequence; repeated calls (e.g. one per source batch)
/// compose as a running union without re-zeroing between them.
///
/// **An out-of-range target (`index[i] >= out_rows`) is silently dropped**,
/// not an error — the same zero-fallback contract as [`mask_gather_u32`]'s
/// out-of-range read, applied to the write side: a target that names no row
/// in the output population has nowhere to land.
///
/// # Why this lives HERE
///
/// The mirror image of [`mask_gather_u32`]: that one reads `src` at an
/// address named by `index`; this one writes to an address named by
/// `index`. Same reasoning for living beside the other data-indexed
/// primitives rather than being hand-rolled per call site.
///
/// # Vectorisation, honestly
///
/// **Scalar, and by necessity, not oversight** — same shape as
/// [`mask_gather_u32`], mirrored: the address written is `index[i]`, a
/// value out of another array, so this is a scatter, not a gather, and
/// none of this crate's backends has a bit-level scatter. Cost is
/// proportional to `popcount(src)` (clamped to `index.len()`), not to
/// `index.len()` itself — only selected source elements are walked, the
/// same shape as [`masked_sum_i32`]'s set-bit walk.
///
/// # Panics
///
/// Panics if `src.len() < index.len().div_ceil(64)`, or if
/// `out_words.len() < out_rows.div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::mask_scatter_or_u32;
///
/// let src = [0b0101u64]; // rows 0 and 2 of `index`'s population are selected
/// let index = [5u32, 1, 9, 20]; // row 2 maps to 9; row 3's 20 is never read
/// let mut out = [0u64; 1];
/// mask_scatter_or_u32(&src, &index, &mut out, 10);
/// // selected: i=0 -> index[0]=5 -> set bit 5; i=2 -> index[2]=9 -> set bit 9
/// assert_eq!(out[0], (1u64 << 5) | (1u64 << 9));
/// ```
#[inline]
pub fn mask_scatter_or_u32(src: &[u64], index: &[u32], out_words: &mut [u64], out_rows: usize) {
    let n = index.len();
    let src_words = mask_words_for(n);
    assert!(src.len() >= src_words, "mask_scatter_or_u32: src.len()={} < required {}", src.len(), src_words);
    let out_word_count = mask_words_for(out_rows);
    assert!(
        out_words.len() >= out_word_count,
        "mask_scatter_or_u32: out_words.len()={} < required {}",
        out_words.len(),
        out_word_count
    );

    // Accumulate: OR scattered bits into whatever the caller already has in
    // `out_words`. The caller zeroes once before the first call.
    for (w, &word) in src.iter().take(src_words).enumerate() {
        let base = w * 64;
        let mut bits = word;
        // Clamp the final partial src word to index.len(), exactly as
        // masked_sum_i32 clamps: a dirty tail bit must never address
        // `index` past its real length.
        let valid = n - base;
        if valid < 64 {
            bits &= (1u64 << valid) - 1;
        }
        while bits != 0 {
            let lane = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let i = base + lane;
            let t = index[i] as usize;
            if t < out_rows {
                out_words[t / 64] |= 1u64 << (t % 64);
            }
        }
    }
}

/// Keyed group-sum: for every element `i` selected by `mask_words` (bit `i`
/// set, `i < values.len()`), adds `values[i]` (widened to `i64`) into
/// `out[keys[i]]` — provided `keys[i] < out.len()`.
///
/// A keyed segmented sum: the caller sizes `out` to the key universe (one
/// slot per key) and this
/// walks the selected rows **once**, replacing K separate masked-sum passes
/// (one per group) with a single pass that routes each row's contribution
/// to its own slot as it goes.
///
/// **This is NOT [`masked_strided_group_sum`].** That function sums the
/// byte-*groups* of ONE record's small register (`6×2`/`4×3`/`3×4` fields)
/// into a single scalar and has no notion of a key — it is a record-local
/// fold. This function sums *rows* into *per-key* buckets across a whole
/// selected population and has no notion of a register. Same word
/// "group", two unrelated shapes; do not conflate them.
///
/// `out` is **accumulated into, not overwritten**: contributions are
/// added to whatever `out` already holds, so the caller zeroes `out` once
/// before the first call in a sequence rather than this function doing it.
/// **A key at or past `out.len()` is dropped,
/// not an error** — the zero-fallback contract shared by
/// [`mask_gather_u32`]/[`mask_scatter_or_u32`]: a key naming no group in
/// `out` is not a group, the same way an address naming no slot is not a slot.
///
/// # Why this lives HERE
///
/// A scatter-reduce indexed by `keys`, not a fixed stride or a plain
/// set-bit sum — [`masked_sum_i32`] above sums every selected element into
/// ONE accumulator; this routes each selected element into ONE OF MANY
/// accumulators chosen by data. A consumer hand-rolling per-group masked
/// sums (`masked_sum_i32` called once per key, K passes over the data) is
/// exactly the shape this closes — one pass instead of K.
///
/// # Vectorisation, honestly
///
/// **Scalar, and by necessity, not oversight** — the destination of each
/// add is `keys[i]`, a value out of another array, so this is a
/// scatter-add, and none of this crate's backends has a vector scatter-add
/// with per-lane conflict resolution (two selected rows can share a key in
/// the same 16- or 64-lane group, which a naive vector scatter would race).
/// Cost is proportional to `popcount(mask_words)` (clamped to
/// `values.len()`), the same set-bit walk as [`masked_sum_i32`].
///
/// # Overflow
///
/// Each element is widened to `i64` before the add and accumulated with
/// `wrapping_add`, the same contract as [`masked_sum_i32`] — a per-key sum
/// of `i32`s cannot overflow `i64` at any length that fits a 64-bit address
/// space, and the wrap is defined behaviour rather than a debug-only panic
/// for the theoretical case that could.
///
/// # Panics
///
/// Panics if `keys.len() != values.len()`, or if `mask_words.len() <
/// values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::masked_group_sum_i32;
///
/// let mask = [0b1011u64]; // rows 0, 1, 3 selected; row 2 is not
/// let keys = [0u32, 1, 0, 1];
/// let values = [10i32, 20, 30, 40];
/// let mut out = [0i64; 2];
/// masked_group_sum_i32(&mask, &keys, &values, &mut out);
/// // row 0 -> key 0 (+10); row 1 -> key 1 (+20); row 3 -> key 1 (+40)
/// assert_eq!(out, [10, 60]);
/// ```
#[inline]
pub fn masked_group_sum_i32(mask_words: &[u64], keys: &[u32], values: &[i32], out: &mut [i64]) {
    assert_eq!(keys.len(), values.len(), "masked_group_sum_i32: keys/values length mismatch");
    let n = values.len();
    let words = mask_words_for(n);
    assert!(
        mask_words.len() >= words,
        "masked_group_sum_i32: mask_words.len()={} < required {}",
        mask_words.len(),
        words
    );

    // Accumulate: add into whatever `out` already holds. The caller zeroes
    // once before the first call.
    for (w, &word) in mask_words.iter().take(words).enumerate() {
        let base = w * 64;
        let mut bits = word;
        // Same tail clamp as masked_sum_i32: a dirty final word must never
        // index past `values`/`keys`.
        let valid = n - base;
        if valid < 64 {
            bits &= (1u64 << valid) - 1;
        }
        while bits != 0 {
            let lane = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let i = base + lane;
            let k = keys[i] as usize;
            if k < out.len() {
                out[k] = out[k].wrapping_add(values[i] as i64);
            }
        }
    }
}

/// Like [`masked_group_sum_i32`], but the group key of row `i` is
/// `table[index[i]]` — the key lives in a second address space reached
/// through an index lane: exactly the [`mask_gather_u32`] /
/// [`mask_scatter_or_u32`] index-lane shape applied to the group-sum's key
/// instead of to a mask bit.
///
/// **Zero-fallback at BOTH hops, same rule as [`mask_gather_u32`]'s
/// out-of-range read**: `index[i] >= table.len()` drops row `i` (the index
/// names no entry of `table`); `table[index[i]] as usize >=
/// out.len()` drops it too (the resolved key names no group). Neither is
/// an error — an unminted address is not a group, at either hop.
///
/// # Why the indirection is fused here
///
/// Materialising `table[index[i]]` into its own `Vec<u32>` of length `N`
/// first and then calling [`masked_group_sum_i32`] on that would allocate
/// and fully populate exactly the intermediate key lane this fold exists
/// to avoid — one fused scan over the selected rows costs no more than the
/// naive two-hop lookup per selected row, with no second array in between.
///
/// `out` is accumulated into (not zeroed by this function) — the caller
/// zeroes `out` once before the first call, same as [`masked_group_sum_i32`];
/// overflow wraps the same way as [`masked_group_sum_i32`] (widened to
/// `i64`, `wrapping_add`), and the mask tail is clamped identically.
///
/// # Panics
///
/// Panics if `index.len() != values.len()`, or if `mask_words.len() <
/// values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::masked_group_sum_i32_via;
///
/// // Two rows address entry 0 (key 0); one addresses entry 5, which is out
/// // of range for `table` and is dropped at the first hop.
/// let mask = [0b111u64];
/// let index = [0u32, 0, 5];
/// let table = [0u32]; // entry 0 -> key 0
/// let values = [10i32, 20, 999];
/// let mut out = [0i64; 1];
/// masked_group_sum_i32_via(&mask, &index, &table, &values, &mut out);
/// assert_eq!(out, [30]);
/// ```
#[inline]
pub fn masked_group_sum_i32_via(mask_words: &[u64], index: &[u32], table: &[u32], values: &[i32], out: &mut [i64]) {
    assert_eq!(index.len(), values.len(), "masked_group_sum_i32_via: index/values length mismatch");
    let n = values.len();
    let words = mask_words_for(n);
    assert!(
        mask_words.len() >= words,
        "masked_group_sum_i32_via: mask_words.len()={} < required {}",
        mask_words.len(),
        words
    );

    // Accumulate: add into whatever `out` already holds. The caller zeroes
    // once before the first call.
    for (w, &word) in mask_words.iter().take(words).enumerate() {
        let base = w * 64;
        let mut bits = word;
        // Same tail clamp as masked_group_sum_i32.
        let valid = n - base;
        if valid < 64 {
            bits &= (1u64 << valid) - 1;
        }
        while bits != 0 {
            let lane = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let i = base + lane;
            let addr = index[i] as usize;
            if addr >= table.len() {
                continue;
            }
            let k = table[addr] as usize;
            if k < out.len() {
                out[k] = out[k].wrapping_add(values[i] as i64);
            }
        }
    }
}

/// Packs `index[i] < table.len() && table[index[i]] == v` into `out_words`,
/// one bit per row `i < index.len()`, LSB-first — an equality predicate
/// evaluated **through an index lane**, with no gathered mask and no
/// materialised predicate plane over `table`'s population in between.
///
/// The projection→projection shape: `table[index[i]]` is never written
/// into its own array and compared afterwards; the addressed read and the
/// compare fuse into one pass over `index`, the same fusion
/// [`masked_group_sum_i32_via`] applies to an addressed group key rather than an addressed
/// predicate.
///
/// **Zero-fallback, same rule as [`mask_gather_u32`]'s out-of-range
/// read**: `index[i] >= table.len()` means row `i` addresses no
/// row, so it does not match — not an error, not a panic, just `false` for
/// that bit. A key that names no row is not a match, the same way an
/// address naming no entry is not a match.
///
/// `out_words` is **fully overwritten**, not OR-ed into; trailing bits
/// beyond `index.len()`, and any surplus words past `mask_words_for(index.len())`,
/// are written `0` — this writes exactly its own output tile and nothing
/// past it, the same contract as [`mask_gather_u32`].
///
/// # Why this lives HERE
///
/// [`mask_gather_u32`] reads a *mask bit* through an index; this reads an
/// *equality predicate* through an index, and belongs beside it for the
/// same reason: a consumer hand-rolling `table[index[i]] == v` in a loop is
/// exactly the polyfill bypass the "all SIMD from `ndarray::simd`"
/// invariant exists to prevent, and it is the gather half of
/// [`masked_group_sum_i32_via`]'s two-hop shape applied to a predicate
/// instead of a sum.
///
/// # Vectorisation, honestly
///
/// **Scalar, and by necessity, not oversight** — same shape as
/// [`mask_gather_u32`]: the address read from `table` is `index[i]`, a
/// value out of another array, so there is no vector gather over
/// individual predicate results on any of this crate's backends. Cost is
/// `O(index.len())`; there is no input mask to skip zero words against.
///
/// # Panics
///
/// Panics if `out_words.len() < mask_words_for(index.len())`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::eq_u32_via_to_mask;
///
/// // entry 0 -> 7, entry 1 -> 3; row 2's index (5) is out of range for
/// // `table` and never matches.
/// let index = [0u32, 1, 5, 0];
/// let table = [7u32, 3];
/// let mut out = [u64::MAX]; // dirty tail must be overwritten
/// eq_u32_via_to_mask(&index, &table, 7, &mut out);
/// // rows 0 and 3 resolve to 7; row 1 resolves to 3; row 2 drops.
/// assert_eq!(out[0], 0b1001);
/// ```
#[inline]
pub fn eq_u32_via_to_mask(index: &[u32], table: &[u32], v: u32, out_words: &mut [u64]) {
    let n = index.len();
    let words = mask_words_for(n);
    assert!(
        out_words.len() >= words,
        "eq_u32_via_to_mask: out_words.len()={} < required {}",
        out_words.len(),
        words
    );

    for (w, out_word) in out_words.iter_mut().enumerate().take(words) {
        let base = w * 64;
        let live = (n - base).min(64);
        let mut acc = 0u64;
        for lane in 0..live {
            let addr = index[base + lane] as usize;
            if addr < table.len() && table[addr] == v {
                acc |= 1u64 << lane;
            }
        }
        *out_word = acc;
    }
    for w in out_words.iter_mut().skip(words) {
        *w = 0;
    }
}

/// Carry of [`masked_key_run_count_u32`] across calls: the key of the run
/// that is open at the end of the last call, and whether that run has
/// already seen a selected element. Two words, independent of the
/// population — the whole state a key-ORDERED distinct count needs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct KeyRunCarry {
    /// The key of the currently open run, `None` before the first element.
    pub key: Option<u32>,
    /// Whether the open run has a selected element so far.
    pub hit: bool,
}

impl KeyRunCarry {
    /// Close the final run: `1` if it is open and was hit, else `0`. Call
    /// once after the last call; the carry is reset to its initial state.
    ///
    /// ```
    /// use ndarray::simd::{masked_key_run_count_u32, KeyRunCarry};
    ///
    /// // keys 4 4 | 8 8 with rows 1 and 3 selected: the first run closes
    /// // inside the call (counted there); the last run is still open.
    /// let mut carry = KeyRunCarry::default();
    /// let closed = masked_key_run_count_u32(&[4u32, 4, 8, 8], &[0b1010], &mut carry).unwrap();
    /// assert_eq!(closed, 1);
    /// assert_eq!(carry.finish(), 1); // the open 8-run was hit
    /// assert_eq!(carry, KeyRunCarry::default()); // and the carry is reset
    /// assert_eq!(carry.finish(), 0); // nothing open any more
    /// ```
    #[inline]
    pub fn finish(&mut self) -> usize {
        let n = usize::from(self.key.is_some() && self.hit);
        *self = Self::default();
        n
    }
}

/// Counts the maximal runs of equal consecutive `keys` values that contain
/// at least one element selected by `mask_words`, closing runs as it goes
/// and carrying the open one in `carry`. Returns `Some(runs CLOSED by this
/// call)`; [`KeyRunCarry::finish`] adds the last one.
///
/// **Refuses a lane that is not in key order.** A key smaller than the open
/// run's key means an earlier key can recur later, so a run is not a key
/// and the count would be wrong: the call returns `None` at that element,
/// having touched nothing but the carry. Non-decreasing order is the one
/// contiguity certificate checkable with O(1) state in the same pass
/// (proving merely "each key occurs in one run" would need the seen-set
/// this fold exists to avoid), so ORDER is the precondition: on a lane in
/// key order — the address order a projection stores a child population
/// under its parent — this IS the count of distinct keys among the selected
/// elements, folded with two words of state and no population-sized set. A
/// lane whose equal keys are contiguous but not sorted (`3 3 1 1`) is
/// refused too; that is deliberate, not a wrong answer. Nothing is ever
/// over-counted.
///
/// The order check inspects EVERY key, selected or not: in `keys = 1 2 1`
/// with only the two `1`s selected, the selected subsequence looks
/// contiguous, but the lane holds two runs of `1` and a fold that skipped
/// the unselected `2` would count it twice. The invariant belongs to the
/// whole lane.
///
/// The mask tail past `keys.len()` is never read; the walk is
/// `O(keys.len())` in row order (a run boundary is a compare against the
/// previous key, a hit is a mask bit — both lane-order reads, no
/// data-dependent address).
///
/// # Vectorisation, honestly
///
/// **Scalar.** Boundaries (`keys[i] != keys[i-1]`) vectorise as a shifted
/// compare and hits are a mask word, but "first hit per run" is a segmented
/// scan with a serial carry; at one compare + one bit test per element it is
/// memory-bound already. Left scalar until a measurement says otherwise.
///
/// # Panics
///
/// Panics if `mask_words.len() < keys.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{masked_key_run_count_u32, KeyRunCarry};
///
/// // keys in key order: 3 3 | 7 | 9 9 9 ; selected rows 1 and 4.
/// let keys = [3u32, 3, 7, 9, 9, 9];
/// let mask = [0b010010u64];
/// let mut carry = KeyRunCarry::default();
/// let a = masked_key_run_count_u32(&keys[..3], &mask, &mut carry).unwrap();
/// let b = masked_key_run_count_u32(&keys[3..], &[mask[0] >> 3], &mut carry).unwrap();
/// assert_eq!(a + b + carry.finish(), 2); // keys 3 and 9, not 7
///
/// // Out of key order: 1 2 1 — refused, never counted as three keys.
/// let mut c = KeyRunCarry::default();
/// assert_eq!(masked_key_run_count_u32(&[1u32, 2, 1], &[0b111], &mut c), None);
/// ```
#[inline]
pub fn masked_key_run_count_u32(keys: &[u32], mask_words: &[u64], carry: &mut KeyRunCarry) -> Option<usize> {
    let n = keys.len();
    let words = mask_words_for(n);
    assert!(
        mask_words.len() >= words,
        "masked_key_run_count_u32: mask_words.len()={} < required {}",
        mask_words.len(),
        words
    );
    let mut closed = 0usize;
    for (i, &k) in keys.iter().enumerate() {
        let selected = (mask_words[i / 64] >> (i % 64)) & 1 == 1;
        match carry.key {
            Some(cur) if cur == k => carry.hit |= selected,
            Some(cur) if cur > k => return None,
            _ => {
                closed += usize::from(carry.key.is_some() && carry.hit);
                carry.key = Some(k);
                carry.hit = selected;
            }
        }
    }
    Some(closed)
}

// ────────────────────────────────────────────────────────────────────────
// The closed comparison family + mask complement/xor/any + care-masked
// register match + masked min/max + blend (the DuckDB-vector-execution set,
// added 2026-09-13 for `lance-graph-duckmask` and lgj-abi D-MRL-1a).
//
// Same normative bit order and trailing-zero guarantee as everything above.
// Every writer below is a full overwrite. Ordered compares are SIGNED for the
// `i32` family; equality/inequality is exact bitwise.
//
// Why `!gt`-style derivations rather than threshold shifting: `x <= t` as
// `!(x > t)` is exact at every boundary including `i32::MIN`/`i32::MAX`,
// whereas `x < t` as `x > t - 1` underflows at `t == i32::MIN`. The tail of
// a complemented mask is re-cleared against the known element count, so the
// guarantee stays structural.
// ────────────────────────────────────────────────────────────────────────

/// Clear every bit at or past element `n` — the tail of the last live word
/// and every surplus word. Shared by every complementing writer below.
#[inline(always)]
fn clear_mask_tail(out_words: &mut [u64], n: usize) {
    let words = mask_words_for(n);
    if words > 0 && !n.is_multiple_of(64) {
        out_words[words - 1] &= (1u64 << (n % 64)) - 1;
    }
    for w in out_words.iter_mut().skip(words) {
        *w = 0;
    }
}

/// Packs `values[i] < threshold` (signed) into `out_words`; full overwrite,
/// trailing bits zero. Lowered as `threshold > values[i]` through
/// [`crate::simd::I32x16::gt_bitmask`], so it is exact at `i32::MIN` (never
/// `x > t - 1`).
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::lt_i32_to_mask;
///
/// let values = [5i32, -5, 0, 10];
/// let mut words = [0u64; 1];
/// lt_i32_to_mask(&values, 0, &mut words);
/// // only -5 is less than 0 → bit 1 → 0b0010
/// assert_eq!(words[0], 0b0010);
/// ```
#[inline]
pub fn lt_i32_to_mask(values: &[i32], threshold: i32, out_words: &mut [u64]) {
    let t = crate::simd::I32x16::splat(threshold);
    pack::<i32, 16>("lt_i32_to_mask", values, out_words, |lanes, live| {
        (t.gt_bitmask(crate::simd::I32x16::from_array(lanes)) & live16(live)) as u64
    });
}

/// Packs `values[i] >= threshold` (signed): the complement of
/// [`lt_i32_to_mask`] with the tail re-cleared. Full overwrite.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::ge_i32_to_mask;
///
/// let values = [5i32, -5, 0, 10];
/// let mut words = [0u64; 1];
/// ge_i32_to_mask(&values, 0, &mut words);
/// // 5, 0 and 10 are >= 0 → bits 0, 2, 3 → 0b1101
/// assert_eq!(words[0], 0b1101);
/// ```
#[inline]
pub fn ge_i32_to_mask(values: &[i32], threshold: i32, out_words: &mut [u64]) {
    lt_i32_to_mask(values, threshold, out_words);
    for w in out_words.iter_mut() {
        *w = !*w;
    }
    clear_mask_tail(out_words, values.len());
}

/// Packs `values[i] <= threshold` (signed): the complement of
/// [`gt_i32_to_mask`] with the tail re-cleared. Full overwrite.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::le_i32_to_mask;
///
/// let values = [5i32, -5, 0, 10];
/// let mut words = [0u64; 1];
/// le_i32_to_mask(&values, 0, &mut words);
/// // -5 and 0 are <= 0 → bits 1, 2 → 0b0110
/// assert_eq!(words[0], 0b0110);
/// ```
#[inline]
pub fn le_i32_to_mask(values: &[i32], threshold: i32, out_words: &mut [u64]) {
    gt_i32_to_mask(values, threshold, out_words);
    for w in out_words.iter_mut() {
        *w = !*w;
    }
    clear_mask_tail(out_words, values.len());
}

/// Packs `values[i] != needle` into `out_words` in ONE pass as
/// `(needle > v) | (v > needle)` — no complement, so the tail is zero by
/// construction. Full overwrite.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::ne_i32_to_mask;
///
/// let values = [7i32, 1, 7, 2];
/// let mut words = [0u64; 1];
/// ne_i32_to_mask(&values, 7, &mut words);
/// // elements 1 and 3 differ from 7 → bits 1 and 3 → 0b1010
/// assert_eq!(words[0], 0b1010);
/// ```
#[inline]
pub fn ne_i32_to_mask(values: &[i32], needle: i32, out_words: &mut [u64]) {
    let t = crate::simd::I32x16::splat(needle);
    pack::<i32, 16>("ne_i32_to_mask", values, out_words, |lanes, live| {
        let v = crate::simd::I32x16::from_array(lanes);
        ((t.gt_bitmask(v) | v.gt_bitmask(t)) & live16(live)) as u64
    });
}

/// Packs `values[i] == needle` (signed lanes, exact): the complement of
/// [`ne_i32_to_mask`] with the tail re-cleared. Full overwrite. The unsigned
/// sibling is [`eq_u32_to_mask`]; the two agree on every bit pattern.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::eq_i32_to_mask;
///
/// let values = [7i32, 1, 7, 2];
/// let mut words = [0u64; 1];
/// eq_i32_to_mask(&values, 7, &mut words);
/// // elements 0 and 2 equal 7 → bits 0 and 2 → 0b0101
/// assert_eq!(words[0], 0b0101);
/// ```
#[inline]
pub fn eq_i32_to_mask(values: &[i32], needle: i32, out_words: &mut [u64]) {
    ne_i32_to_mask(values, needle, out_words);
    for w in out_words.iter_mut() {
        *w = !*w;
    }
    clear_mask_tail(out_words, values.len());
}

/// Packs `values[i] != needle`: the complement of [`eq_u32_to_mask`] with the
/// tail re-cleared. Full overwrite.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::ne_u32_to_mask;
///
/// let values = [7u32, 1, 7, 2];
/// let mut words = [0u64; 1];
/// ne_u32_to_mask(&values, 7, &mut words);
/// // elements 1 and 3 differ from 7 → bits 1 and 3 → 0b1010
/// assert_eq!(words[0], 0b1010);
/// ```
#[inline]
pub fn ne_u32_to_mask(values: &[u32], needle: u32, out_words: &mut [u64]) {
    eq_u32_to_mask(values, needle, out_words);
    for w in out_words.iter_mut() {
        *w = !*w;
    }
    clear_mask_tail(out_words, values.len());
}

// ────────────────────────────────────────────────────────────────────────
// The u8 comparison family (`*_u8_to_mask`) — the byte-width sibling of the
// u32/i32 families above. [`crate::simd::U8x64`] is 64 lanes wide, so unlike
// those families (whose 16-lane groups pack four per `u64` word via
// `out_words[g / 4] |= (bits as u64) << ((g % 4) * 16)`), one chunk here
// fills exactly ONE whole output word: `out_words[g] = bits`, a plain
// assignment, no shift-and-OR.
//
// Comparison is UNSIGNED — `u8`'s only ordering — via `U8x64::cmpeq_mask` /
// `U8x64::cmpgt_mask`, exact with no saturation or wrapping. Derived exactly
// as the i32 family is: `gt`/`lt` are DIRECT (`cmpgt_mask` with the operands
// in the order the comparison needs); `ge`/`le`/`ne` are the complement of
// `lt`/`gt`/`eq` with [`clear_mask_tail`] re-clearing the tail — the same
// shape as [`ge_i32_to_mask`] / [`le_i32_to_mask`] / [`ne_u32_to_mask`].
//
// The DIRECT forms' tail is a genuinely different shape from the u32/i32
// family's, though: a 64-lane chunk's tail can hold up to 63 elements (not
// up to 15), so it needs a wider "valid lane" mask than [`tail_lane_bits`]
// provides (16-lane only, `debug_assert!(n < 16)`). [`word_range_mask`] —
// already built for [`mask_set_range`] to return the low-N-bits `u64`
// without ever evaluating `1u64 << 64` — is reused here as
// `word_range_mask(0, tail.len())`, rather than adding a second such helper.
// ────────────────────────────────────────────────────────────────────────

/// Packs `values[i] == needle` into `out_words`, one bit per element,
/// LSB-first within each `u64` word (bit `k` of word `w` corresponds to
/// element `w * 64 + k`).
///
/// `out_words` is **fully overwritten**, not OR-ed into. Trailing bits in the
/// final word beyond `values.len()`, and any surplus words past
/// `ceil(len / 64)`, are written as `0`.
///
/// Equality is exact bitwise comparison over the full `u8` range — `0` and
/// `u8::MAX` are ordinary needles, and there is no saturation, wrapping, or
/// signedness question to resolve. An empty `values` writes only zeros.
///
/// Runs 64 lanes at a time through [`crate::simd::U8x64::cmpeq_mask`] — one
/// whole output word per chunk, since `U8x64` is exactly 64 lanes wide
/// (unlike the u32/i32 families' 16-lane groups, four of which pack into one
/// word). The final partial group is zero-padded into one register and run
/// through the same packed compare, with the padding lanes' bits masked off
/// by [`word_range_mask`] — no scalar tail, so the tail cannot disagree with
/// the body.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::eq_u8_to_mask;
///
/// let values = [7u8, 1, 7, 2];
/// let mut words = [0u64; 1];
/// eq_u8_to_mask(&values, 7, &mut words);
/// // elements 0 and 2 match → bits 0 and 2 → 0b0101
/// assert_eq!(words[0], 0b0101);
/// ```
#[inline]
pub fn eq_u8_to_mask(values: &[u8], needle: u8, out_words: &mut [u64]) {
    let needle_v = crate::simd::U8x64::splat(needle);
    pack::<u8, 64>("eq_u8_to_mask", values, out_words, |lanes, live| {
        crate::simd::U8x64::from_array(lanes).cmpeq_mask(needle_v) & live64(live)
    });
}

/// Packs `values[i] > threshold` (**unsigned** comparison — the only
/// ordering `u8` has) into `out_words`, one bit per element, LSB-first
/// within each `u64` word (bit `k` of word `w` corresponds to element
/// `w * 64 + k`).
///
/// `out_words` is **fully overwritten**, not OR-ed into. Trailing bits in the
/// final word beyond `values.len()`, and any surplus words past
/// `ceil(len / 64)`, are written as `0`.
///
/// Comparison is unsigned and strict (`>`, never `>=`), exact with no
/// saturation or wrapping:
/// * `threshold == 0` sets every lane except those equal to `0`.
/// * `threshold == u8::MAX` sets nothing — no `u8` exceeds it.
///
/// An empty `values` writes only zeros.
///
/// Runs 64 lanes at a time through [`crate::simd::U8x64::cmpgt_mask`] — one
/// whole output word per chunk. The final partial group is zero-padded into
/// one register and run through the same packed compare, with the padding
/// lanes' bits masked off by [`word_range_mask`].
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::gt_u8_to_mask;
///
/// let values = [5u8, 0, 200, 255];
/// let mut words = [0u64; 1];
/// gt_u8_to_mask(&values, 4, &mut words);
/// // elements 0, 2 and 3 exceed 4 → bits 0, 2, 3 → 0b1101
/// assert_eq!(words[0], 0b1101);
/// ```
#[inline]
pub fn gt_u8_to_mask(values: &[u8], threshold: u8, out_words: &mut [u64]) {
    let threshold_v = crate::simd::U8x64::splat(threshold);
    pack::<u8, 64>("gt_u8_to_mask", values, out_words, |lanes, live| {
        crate::simd::U8x64::from_array(lanes).cmpgt_mask(threshold_v) & live64(live)
    });
}

/// Packs `values[i] < threshold` (unsigned): computed directly as
/// `threshold > values[i]` through [`crate::simd::U8x64::cmpgt_mask`] with
/// the operands swapped — the same shape [`lt_i32_to_mask`] uses (there
/// `x > t - 1` would underflow at `t == i32::MIN`; `u8` has no such boundary,
/// but the swapped-operand derivation is kept identical across the two
/// families rather than special-cased per element type). Full overwrite,
/// trailing bits zero.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::lt_u8_to_mask;
///
/// let values = [5u8, 0, 200, 255];
/// let mut words = [0u64; 1];
/// lt_u8_to_mask(&values, 4, &mut words);
/// // only element 1 (0) is less than 4 → bit 1 → 0b0010
/// assert_eq!(words[0], 0b0010);
/// ```
#[inline]
pub fn lt_u8_to_mask(values: &[u8], threshold: u8, out_words: &mut [u64]) {
    let t = crate::simd::U8x64::splat(threshold);
    pack::<u8, 64>("lt_u8_to_mask", values, out_words, |lanes, live| {
        t.cmpgt_mask(crate::simd::U8x64::from_array(lanes)) & live64(live)
    });
}

/// Packs `values[i] >= threshold` (unsigned): the complement of
/// [`lt_u8_to_mask`] with the tail re-cleared. Full overwrite.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::ge_u8_to_mask;
///
/// let values = [5u8, 0, 200, 255];
/// let mut words = [0u64; 1];
/// ge_u8_to_mask(&values, 4, &mut words);
/// // 5, 200 and 255 are >= 4 → bits 0, 2, 3 → 0b1101
/// assert_eq!(words[0], 0b1101);
/// ```
#[inline]
pub fn ge_u8_to_mask(values: &[u8], threshold: u8, out_words: &mut [u64]) {
    lt_u8_to_mask(values, threshold, out_words);
    for w in out_words.iter_mut() {
        *w = !*w;
    }
    clear_mask_tail(out_words, values.len());
}

/// Packs `values[i] <= threshold` (unsigned): the complement of
/// [`gt_u8_to_mask`] with the tail re-cleared. Full overwrite.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::le_u8_to_mask;
///
/// let values = [5u8, 0, 200, 255];
/// let mut words = [0u64; 1];
/// le_u8_to_mask(&values, 4, &mut words);
/// // only element 1 (0) is <= 4 → bit 1 → 0b0010
/// assert_eq!(words[0], 0b0010);
/// ```
#[inline]
pub fn le_u8_to_mask(values: &[u8], threshold: u8, out_words: &mut [u64]) {
    gt_u8_to_mask(values, threshold, out_words);
    for w in out_words.iter_mut() {
        *w = !*w;
    }
    clear_mask_tail(out_words, values.len());
}

/// Packs `values[i] != needle`: the complement of [`eq_u8_to_mask`] with the
/// tail re-cleared. Full overwrite.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::ne_u8_to_mask;
///
/// let values = [7u8, 1, 7, 2];
/// let mut words = [0u64; 1];
/// ne_u8_to_mask(&values, 7, &mut words);
/// // elements 1 and 3 differ from 7 → bits 1 and 3 → 0b1010
/// assert_eq!(words[0], 0b1010);
/// ```
#[inline]
pub fn ne_u8_to_mask(values: &[u8], needle: u8, out_words: &mut [u64]) {
    eq_u8_to_mask(values, needle, out_words);
    for w in out_words.iter_mut() {
        *w = !*w;
    }
    clear_mask_tail(out_words, values.len());
}

// ────────────────────────────────────────────────────────────────────────
// Ordered 64-bit predicates — T1 gap G2 of the DuckDB→V3 translation matrix.
//
// The width where the packing is NOT free. `U8x64` is 64 lanes against a
// 64-bit word, so one chunk is one whole word with no shift; `U64x8` is 8
// lanes, so eight chunks share a word and each must be placed at its own
// byte, `out_words[g / 8] |= (bits as u64) << ((g % 8) * 8)`. That is the
// same packing `eq_u32_to_mask` does at four-chunks-per-word, one step
// further.
//
// Why `u64` and not `i64`: the named consumer is an ordered window over
// address-space offsets (`lo <= x < hi`), which are unsigned and routinely
// exceed `2^32`, so narrowing to the existing `i32` family is unsound. No
// caller compares `i64` lanes, so that family is deliberately not built —
// a speculative family is surface with no falsifier attached to it.
//
// The ORDERED ops need an unsigned compare, and only two of the six
// realizations have one natively (AVX-512 `_mm512_cmpgt_epu64_mask`, NEON
// `cmhi`). WASM and the byte-width AVX2 path reach it by flipping the sign
// bit of both operands — an order-preserving bijection — and the flat
// polyfill arms get it for free because Rust's `>` on `u64` already is the
// unsigned compare. All of that lives in the arm files; this layer never
// sees it, which is the point of the facade.
// ────────────────────────────────────────────────────────────────────────

/// Packs `values[i] == needle` into `out_words`, one bit per element,
/// LSB-first within each `u64` word (bit `k` of word `w` is element
/// `w * 64 + k`).
///
/// `out_words` is **fully overwritten**, not OR-ed into. Trailing bits in
/// the final word beyond `values.len()`, and any surplus words past
/// `ceil(len / 64)`, are written as `0`.
///
/// Runs 8 lanes at a time through [`crate::simd::U64x8::cmpeq_mask`], whose
/// 8-bit result is placed at byte `g % 8` of word `g / 8`. The final partial
/// group is zero-padded into one register and run through the same packed
/// compare, with the padding lanes' bits masked off — load-bearing, because
/// a `needle` of `0` matches the padding.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::eq_u64_to_mask;
///
/// let values = [7u64, 1, 7, 1 << 63];
/// let mut words = [0u64; 1];
/// eq_u64_to_mask(&values, 7, &mut words);
/// // elements 0 and 2 equal 7 → bits 0 and 2 → 0b0101
/// assert_eq!(words[0], 0b0101);
/// ```
#[inline]
pub fn eq_u64_to_mask(values: &[u64], needle: u64, out_words: &mut [u64]) {
    let needle_v = crate::simd::U64x8::splat(needle);
    pack::<u64, 8>("eq_u64_to_mask", values, out_words, |lanes, live| {
        (crate::simd::U64x8::from_array(lanes).cmpeq_mask(needle_v) & live8(live)) as u64
    });
}

/// Packs `values[i] > threshold` (**unsigned** — the only ordering `u64`
/// has) into `out_words`. Full overwrite, trailing bits zero.
///
/// Unsigned is not a detail at this width: a `u64` at or above `1 << 63`
/// reads as NEGATIVE under a signed compare, so a signed instruction
/// inverts the answer on exactly half the domain — and address-space
/// offsets, the reason this family exists, live in that half.
///
/// * `threshold == 0` sets every lane except those equal to `0`.
/// * `threshold == u64::MAX` sets nothing — no `u64` exceeds it.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::gt_u64_to_mask;
///
/// let values = [5u64, 0, 1 << 63, u64::MAX];
/// let mut words = [0u64; 1];
/// gt_u64_to_mask(&values, 4, &mut words);
/// // 5, 2^63 and u64::MAX all exceed 4 → bits 0, 2, 3 → 0b1101
/// assert_eq!(words[0], 0b1101);
/// ```
#[inline]
pub fn gt_u64_to_mask(values: &[u64], threshold: u64, out_words: &mut [u64]) {
    let threshold_v = crate::simd::U64x8::splat(threshold);
    pack::<u64, 8>("gt_u64_to_mask", values, out_words, |lanes, live| {
        (crate::simd::U64x8::from_array(lanes).cmpgt_mask(threshold_v) & live8(live)) as u64
    });
}

/// Packs `values[i] < threshold` (unsigned): computed directly as
/// `threshold > values[i]` through [`crate::simd::U64x8::cmpgt_mask`] with
/// the operands swapped.
///
/// An **operand swap, never `x > t - 1`** — that spelling underflows at
/// `t == 0` and would report every lane as less-than nothing. Same
/// reasoning the `i32` family carries for `i32::MIN`.
///
/// The padding lanes of a partial group are `0`, and `threshold > 0` holds
/// for every `threshold` above zero, so the tail mask here is what keeps a
/// short input from reporting phantom set bits.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::lt_u64_to_mask;
///
/// let values = [5u64, 0, 1 << 63, u64::MAX];
/// let mut words = [0u64; 1];
/// lt_u64_to_mask(&values, 6, &mut words);
/// // only 5 and 0 are below 6 → bits 0, 1 → 0b0011
/// assert_eq!(words[0], 0b0011);
/// ```
#[inline]
pub fn lt_u64_to_mask(values: &[u64], threshold: u64, out_words: &mut [u64]) {
    let t = crate::simd::U64x8::splat(threshold);
    pack::<u64, 8>("lt_u64_to_mask", values, out_words, |lanes, live| {
        (t.cmpgt_mask(crate::simd::U64x8::from_array(lanes)) & live8(live)) as u64
    });
}

/// Packs `values[i] >= threshold` (unsigned): the complement of
/// [`lt_u64_to_mask`] with the tail re-cleared. Full overwrite.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::ge_u64_to_mask;
///
/// let values = [5u64, 0, 1 << 63, u64::MAX];
/// let mut words = [0u64; 1];
/// ge_u64_to_mask(&values, 5, &mut words);
/// // 5, 2^63 and u64::MAX are >= 5 → bits 0, 2, 3 → 0b1101
/// assert_eq!(words[0], 0b1101);
/// ```
#[inline]
pub fn ge_u64_to_mask(values: &[u64], threshold: u64, out_words: &mut [u64]) {
    lt_u64_to_mask(values, threshold, out_words);
    for w in out_words.iter_mut() {
        *w = !*w;
    }
    clear_mask_tail(out_words, values.len());
}

/// Packs `values[i] <= threshold` (unsigned): the complement of
/// [`gt_u64_to_mask`] with the tail re-cleared. Full overwrite.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::le_u64_to_mask;
///
/// let values = [5u64, 0, 1 << 63, u64::MAX];
/// let mut words = [0u64; 1];
/// le_u64_to_mask(&values, 5, &mut words);
/// // only 5 and 0 are <= 5 → bits 0, 1 → 0b0011
/// assert_eq!(words[0], 0b0011);
/// ```
#[inline]
pub fn le_u64_to_mask(values: &[u64], threshold: u64, out_words: &mut [u64]) {
    gt_u64_to_mask(values, threshold, out_words);
    for w in out_words.iter_mut() {
        *w = !*w;
    }
    clear_mask_tail(out_words, values.len());
}

/// Packs `values[i] != needle`: the complement of [`eq_u64_to_mask`] with
/// the tail re-cleared. Full overwrite.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::ne_u64_to_mask;
///
/// let values = [7u64, 1, 7, 2];
/// let mut words = [0u64; 1];
/// ne_u64_to_mask(&values, 7, &mut words);
/// // elements 1 and 3 differ from 7 → bits 1 and 3 → 0b1010
/// assert_eq!(words[0], 0b1010);
/// ```
#[inline]
pub fn ne_u64_to_mask(values: &[u64], needle: u64, out_words: &mut [u64]) {
    eq_u64_to_mask(values, needle, out_words);
    for w in out_words.iter_mut() {
        *w = !*w;
    }
    clear_mask_tail(out_words, values.len());
}

/// `dst = !src` over `n_rows` elements — the tail-aware complement. Bits at
/// or past `n_rows` are written `0`, so a conforming input yields a
/// conforming output (plain `!` on the words would set every tail bit).
///
/// # Panics
///
/// Panics unless `src.len() == dst.len() >= n_rows.div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::mask_not;
///
/// let src = [0b0011u64];
/// let mut dst = [0u64; 1];
/// mask_not(&src, 4, &mut dst);
/// // complement of the low 4 bits of 0b0011, tail past row 4 stays zero
/// assert_eq!(dst[0], 0b1100);
/// ```
#[inline]
pub fn mask_not(src: &[u64], n_rows: usize, dst: &mut [u64]) {
    assert_eq!(src.len(), dst.len(), "mask_not: src/dst length mismatch");
    assert!(
        dst.len() >= mask_words_for(n_rows),
        "mask_not: dst.len()={} < required {}",
        dst.len(),
        mask_words_for(n_rows)
    );
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = !s;
    }
    clear_mask_tail(dst, n_rows);
}

/// `dst = !dst` over `n_rows` elements, in place. Same tail contract as
/// [`mask_not`].
///
/// # Panics
///
/// Panics if `dst.len() < n_rows.div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::mask_not_assign;
///
/// let mut dst = [0b0011u64];
/// mask_not_assign(&mut dst, 4);
/// // complement of the low 4 bits of 0b0011, tail past row 4 stays zero
/// assert_eq!(dst[0], 0b1100);
/// ```
#[inline]
pub fn mask_not_assign(dst: &mut [u64], n_rows: usize) {
    assert!(
        dst.len() >= mask_words_for(n_rows),
        "mask_not_assign: dst.len()={} < required {}",
        dst.len(),
        mask_words_for(n_rows)
    );
    for d in dst.iter_mut() {
        *d = !*d;
    }
    clear_mask_tail(dst, n_rows);
}

/// Sets bits `[lo, hi)` of `out_words` to `1` and every other bit — inside
/// the range's own edge words and every word outside it — to `0`. The
/// **range WRITE** this file's care-masked matches are the READ half of:
/// [`ternary_match_u32_to_mask`] answers "which rows equal this pattern",
/// this answers "make exactly these rows true, and nothing else."
///
/// # Why this exists
///
/// `lance-graph-quack`'s `Filter::prefix_u32` spells a full ternary-match
/// PREDICATE to express what is really a contiguous row-range write, and
/// says so in its own doc comment: the range write is missing from
/// `ndarray::simd`, and per the missing-capability STOP rule a consumer does
/// not hand-roll it one layer up. This repo's own `examples/hex_tenant_mq_probe.rs`
/// hand-rolled the identical shape as a private `range_reveal` and measured
/// it at 161.8×–343.5× faster than the equivalent TCAM compare sweep, across
/// trie levels from a 65,536-row reveal down to a 1-row one
/// (`.claude/plans/gemm-ternlog-mask-consolidation-v1.md` §16.2). DuckDB
/// carries the identical operation over the identical representation
/// (`TemplatedValidityMask::SetRangeInvalid`), so the shape is not invented
/// here.
///
/// This primitive is deliberately **address-blind**: it sets a bit range and
/// nothing else. Whether a column prefix legally maps to a contiguous row
/// range — or whether the backing column is sorted at all — is the
/// **caller's** decision, kept out of this layer on purpose.
///
/// `lo == hi` is a legal empty range and writes an all-zero mask (not a
/// no-op: every bit is still overwritten).
///
/// # No per-bit loop
///
/// The write is at most three passes over `out_words`: every word strictly
/// before `lo`'s word is zeroed, every word strictly after `hi`'s last live
/// word is zeroed, whole interior words are written `u64::MAX`, and the (at
/// most two) edge words each get one computed mask ([`word_range_mask`]).
/// When `lo` and `hi` fall in the SAME word, that word is written exactly
/// once through the single-word branch — never as two overlapping edge
/// writes.
///
/// # Panics
///
/// Panics if `lo > hi`, or if `hi > out_words.len() * 64`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::mask_set_range;
///
/// let mut words = [0u64; 2];
/// mask_set_range(&mut words, 60, 70);
/// // bits 60..64 of word 0, bits 0..6 of word 1 — an adjacent-word range.
/// assert_eq!(words[0], 0xF000_0000_0000_0000);
/// assert_eq!(words[1], 0b0011_1111);
///
/// // lo == hi is a legal empty range: an all-zero overwrite, not a no-op.
/// words[0] = u64::MAX;
/// mask_set_range(&mut words, 3, 3);
/// assert_eq!(words, [0, 0]);
/// ```
#[inline]
pub fn mask_set_range(out_words: &mut [u64], lo: usize, hi: usize) {
    assert!(lo <= hi, "mask_set_range: lo={lo} > hi={hi}");
    let words = mask_words_for(hi);
    assert!(out_words.len() >= words, "mask_set_range: out_words.len()={} < required {}", out_words.len(), words);

    if lo == hi {
        fill_words(out_words, 0);
        return;
    }

    // `lo < hi` from here on, so `hi >= 1` and `hi - 1` cannot underflow.
    let lo_word = lo / 64;
    let hi_word = (hi - 1) / 64;

    fill_words(&mut out_words[..lo_word], 0);
    fill_words(&mut out_words[hi_word + 1..], 0);

    if lo_word == hi_word {
        // The single-word case: exactly one write, never two overlapping
        // edge writes.
        out_words[lo_word] = word_range_mask(lo % 64, hi - lo_word * 64);
    } else {
        out_words[lo_word] = word_range_mask(lo % 64, 64);
        fill_words(&mut out_words[lo_word + 1..hi_word], u64::MAX);
        out_words[hi_word] = word_range_mask(0, hi - hi_word * 64);
    }
}

/// Writes `value` into every word of `dst`, walking `as_chunks_mut::<LANES>()`
/// and storing one `U64x8` per chunk — the same lane walk every contiguous
/// word op in this file uses (see the NORMATIVE note above and `mask_and`'s
/// body), rather than a scalar `iter_mut` loop.
///
/// The tail is written scalar-wise on purpose: a padded-tail `from_array` would
/// have to read `dst`'s surplus lanes back before storing them, and there is
/// nothing to read here — every lane of a constant fill has the same value, so
/// the tail is a straight copy of `value` into the remaining words.
#[inline]
fn fill_words(dst: &mut [u64], value: u64) {
    const L: usize = crate::simd::U64x8::LANES;
    let splat = crate::simd::U64x8::splat(value).to_array();
    let (chunks, tail) = dst.as_chunks_mut::<L>();
    for c in chunks.iter_mut() {
        *c = splat;
    }
    for w in tail.iter_mut() {
        *w = value;
    }
}

/// One word's worth of the half-open range `[lo_bit, hi_bit)` set,
/// `0 <= lo_bit <= hi_bit <= 64` — the edge-word primitive [`mask_set_range`]
/// composes. Built from two "bits below N" masks rather than one shifted
/// range so that `hi_bit == 64` never computes `1u64 << 64` (a shift amount
/// equal to the type's own bit width, which panics under overflow checks).
#[inline(always)]
fn word_range_mask(lo_bit: usize, hi_bit: usize) -> u64 {
    debug_assert!(lo_bit <= hi_bit && hi_bit <= 64, "word_range_mask: lo_bit={lo_bit} hi_bit={hi_bit}");
    let below_hi = if hi_bit == 64 { u64::MAX } else { (1u64 << hi_bit) - 1 };
    let below_lo = if lo_bit == 0 { 0 } else { (1u64 << lo_bit) - 1 };
    below_hi & !below_lo
}

/// `dst = a ^ b`, elementwise over `u64` mask words — symmetric difference.
/// XOR preserves the trailing-zero guarantee iff both inputs conform
/// (`0 ^ 0 = 0`). Its own primitive, with its own realization on every
/// backend (`U64x8: BitXor`), never spelled as a three-input table.
///
/// # Panics
///
/// Panics unless `a.len() == b.len() == dst.len()`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::mask_xor;
///
/// let a = [0b0110u64];
/// let b = [0b0011u64];
/// let mut dst = [0u64; 1];
/// mask_xor(&a, &b, &mut dst);
/// // bits set in exactly one of a, b: bit 1 (both) cancels, 0 and 2 survive
/// assert_eq!(dst[0], 0b0101);
/// ```
#[inline]
pub fn mask_xor(a: &[u64], b: &[u64], dst: &mut [u64]) {
    assert_eq!(a.len(), b.len(), "mask_xor: a/b length mismatch");
    assert_eq!(a.len(), dst.len(), "mask_xor: a/dst length mismatch");
    const L: usize = crate::simd::U64x8::LANES;
    let (ca, ta) = a.as_chunks::<L>();
    let (cb, tb) = b.as_chunks::<L>();
    let (cd, td) = dst.as_chunks_mut::<L>();
    for ((x, y), d) in ca.iter().zip(cb).zip(cd.iter_mut()) {
        let va = crate::simd::U64x8::from_array(*x);
        let vb = crate::simd::U64x8::from_array(*y);
        *d = (va ^ vb).to_array();
    }
    if !ta.is_empty() {
        let va = crate::simd::U64x8::from_array(pad_tail(ta));
        let vb = crate::simd::U64x8::from_array(pad_tail(tb));
        td.copy_from_slice(&(va ^ vb).to_array()[..td.len()]);
    }
}

/// `dst ^= src`, in place. Same tail contract as [`mask_xor`].
///
/// # Panics
///
/// Panics if `dst.len() != src.len()`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::mask_xor_assign;
///
/// let mut dst = [0b0110u64];
/// let src = [0b0011u64];
/// mask_xor_assign(&mut dst, &src);
/// // bit 1 (set in both) cancels, bits 0 and 2 survive
/// assert_eq!(dst[0], 0b0101);
/// ```
#[inline]
pub fn mask_xor_assign(dst: &mut [u64], src: &[u64]) {
    assert_eq!(dst.len(), src.len(), "mask_xor_assign: length mismatch");
    const L: usize = crate::simd::U64x8::LANES;
    let (cd, td) = dst.as_chunks_mut::<L>();
    let (cs, ts) = src.as_chunks::<L>();
    for (d, s) in cd.iter_mut().zip(cs) {
        let vd = crate::simd::U64x8::from_array(*d);
        let vs = crate::simd::U64x8::from_array(*s);
        *d = (vd ^ vs).to_array();
    }
    if !td.is_empty() {
        let vd = crate::simd::U64x8::from_array(pad_tail(td));
        let vs = crate::simd::U64x8::from_array(pad_tail(ts));
        td.copy_from_slice(&(vd ^ vs).to_array()[..td.len()]);
    }
}

/// `true` iff any bit is set. Word-OR reduction; an empty slice is `false`.
/// This is the survivor test a fused plan uses to stop early, and the
/// `EXISTS` terminal.
///
/// Reads EVERY word, surplus words included, and takes no `n_rows` — it
/// relies on the normative contract that every writer leaves bits at or past
/// the element count zero. A destination that was written by something
/// outside this module with a dirty tail will read as "some row set". The
/// pair [`mask_all`] takes `n_rows` because a full-population test must know
/// where the population ends; a non-empty test does not.
///
/// # Examples
///
/// ```
/// use ndarray::simd::mask_any;
///
/// // all-zero word: nothing set
/// assert!(!mask_any(&[0u64]));
/// // bit 2 set (0b0100): at least one row selected
/// assert!(mask_any(&[0b0100u64]));
/// ```
#[inline]
pub fn mask_any(words: &[u64]) -> bool {
    let mut acc = 0u64;
    for &w in words {
        acc |= w;
    }
    acc != 0
}

/// `true` iff every one of the first `n_rows` bits is set (a conforming mask
/// whose population is the whole universe). `n_rows == 0` is vacuously
/// `true`. Surplus words past the live range are ignored.
///
/// # Panics
///
/// Panics if `words.len() < n_rows.div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::mask_all;
///
/// // bits 0, 1, 2 set (0b0111): rows 0..3 are all selected, row 3 is not
/// let words = [0b0111u64];
/// assert!(mask_all(&words, 3));
/// assert!(!mask_all(&words, 4));
/// ```
#[inline]
pub fn mask_all(words: &[u64], n_rows: usize) -> bool {
    let full = n_rows / 64;
    assert!(
        words.len() >= mask_words_for(n_rows),
        "mask_all: words.len()={} < required {}",
        words.len(),
        mask_words_for(n_rows)
    );
    let mut acc = u64::MAX;
    for &w in &words[..full] {
        acc &= w;
    }
    if acc != u64::MAX {
        return false;
    }
    let rem = n_rows % 64;
    rem == 0 || (words[full] & ((1u64 << rem) - 1)) == (1u64 << rem) - 1
}

/// Packs `((values[i] ^ pattern) & care) == 0` — equality on the bits `care`
/// selects, "don't care" elsewhere — into `out_words`. `care == 0` matches
/// every element; `care == u32::MAX` is exact equality. Full overwrite,
/// trailing bits zero. Lowered as ONE ternlog per 16 lanes
/// ([`crate::simd::ternlog::XOR_AND`] = `(a ^ b) & c`) followed by an
/// `eq_bitmask` against zero.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::ternary_match_u32_to_mask;
///
/// let values = [0b1010u32, 0b1110, 0b0010, 0b1011];
/// let mut words = [0u64; 1];
/// // pattern 0b1010 with bit 2 "don't care" (care=0b1011, bit 2 clear):
/// // elements 0 and 1 match on every cared-about bit
/// ternary_match_u32_to_mask(&values, 0b1010, 0b1011, &mut words);
/// assert_eq!(words[0], 0b0011);
/// ```
#[inline]
pub fn ternary_match_u32_to_mask(values: &[u32], pattern: u32, care: u32, out_words: &mut [u64]) {
    let p = crate::simd::U32x16::splat(pattern);
    let c = crate::simd::U32x16::splat(care);
    let zero = crate::simd::U32x16::splat(0);
    pack::<u32, 16>("ternary_match_u32_to_mask", values, out_words, |lanes, live| {
        let bits = crate::simd::U32x16::from_array(lanes)
            .ternlog::<{ crate::simd::ternlog::XOR_AND }>(p, c)
            .eq_bitmask(zero);
        (bits & live16(live)) as u64
    });
}

/// The 64-bit sibling of [`ternary_match_u32_to_mask`]: packs
/// `((values[i] ^ pattern) & care) == 0`. Full overwrite, trailing bits zero.
/// Lowered as one ternlog per 8 lanes plus a per-lane zero test.
///
/// # Panics
///
/// Panics if `out_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::ternary_match_u64_to_mask;
///
/// let values = [0b1010u64, 0b1110, 0b0010, 0b1011];
/// let mut words = [0u64; 1];
/// // same pattern/care as the u32 sibling: elements 0 and 1 match
/// ternary_match_u64_to_mask(&values, 0b1010, 0b1011, &mut words);
/// assert_eq!(words[0], 0b0011);
/// ```
#[inline]
pub fn ternary_match_u64_to_mask(values: &[u64], pattern: u64, care: u64, out_words: &mut [u64]) {
    let p = crate::simd::U64x8::splat(pattern);
    let c = crate::simd::U64x8::splat(care);
    pack::<u64, { crate::simd::U64x8::LANES }>("ternary_match_u64_to_mask", values, out_words, |lanes, live| {
        let r = crate::simd::U64x8::from_array(lanes)
            .ternlog::<{ crate::simd::ternlog::XOR_AND }>(p, c)
            .to_array();
        let mut bits = 0u64;
        for (lane, &x) in r.iter().take(live).enumerate() {
            bits |= ((x == 0) as u64) << lane;
        }
        bits
    });
}

/// Care-masked match of a **12-byte little-endian register** found at
/// `first_offset + i * stride_bytes` for `i in 0..count` — the strided AoS
/// form for a V3 facet (`4-byte classid | 12-byte payload`: point
/// `first_offset` at the payload) inside a 16-byte facet or a 512-byte row.
/// Element `i` matches iff every byte `k` satisfies
/// `(reg[k] ^ pattern[k]) & care[k] == 0`. Full overwrite, trailing bits zero.
///
/// Loads are `from_le_bytes` over byte slices (no alignment requirement).
/// The compare is vectorised as `(lo64, hi32)` — 8 registers per `U64x8`
/// ternlog for the low 8 bytes, 16 per `U32x16` ternlog for the high 4 — the
/// gathers are scalar, exactly as [`eq_u32_strided_to_mask`] documents (at
/// row strides ≥ a cache line the walk is memory-bound and a gather buys
/// nothing).
///
/// # Panics
///
/// Panics if `out_words.len() < count.div_ceil(64)`, or if any element's
/// 12 bytes would fall outside `bytes` (checked up front with overflow-safe
/// arithmetic; the loop never reads out of bounds).
///
/// # Examples
///
/// ```
/// use ndarray::simd::ternary_match_strided_to_mask;
///
/// // Two 16-byte records; only the register's first byte is "cared about".
/// let mut bytes = vec![0u8; 32];
/// bytes[0] = 0xAA; // record 0's cared byte matches the pattern
/// bytes[16] = 0xBB; // record 1's cared byte does not
/// let mut pattern = [0u8; 12];
/// pattern[0] = 0xAA;
/// let mut care = [0u8; 12];
/// care[0] = 0xFF; // bytes 1..12 are don't-care
/// let mut words = [0u64; 1];
/// ternary_match_strided_to_mask(&bytes, 0, 16, 2, &pattern, &care, &mut words);
/// assert_eq!(words[0], 0b01); // only record 0 matches
/// ```
#[inline]
pub fn ternary_match_strided_to_mask(
    bytes: &[u8], first_offset: usize, stride_bytes: usize, count: usize, pattern: &[u8; 12], care: &[u8; 12],
    out_words: &mut [u64],
) {
    let words = mask_words_for(count);
    assert!(
        out_words.len() >= words,
        "ternary_match_strided_to_mask: out_words.len()={} < required {}",
        out_words.len(),
        words
    );
    if count > 0 {
        let last_end = (count - 1)
            .checked_mul(stride_bytes)
            .and_then(|x| x.checked_add(first_offset))
            .and_then(|x| x.checked_add(12))
            .expect("ternary_match_strided_to_mask: offset arithmetic overflow");
        assert!(
            last_end <= bytes.len(),
            "ternary_match_strided_to_mask: last element ends at {last_end} > bytes.len() {}",
            bytes.len()
        );
    }
    for w in out_words.iter_mut() {
        *w = 0;
    }
    let plo = u64::from_le_bytes(pattern[0..8].try_into().expect("8 bytes"));
    let clo = u64::from_le_bytes(care[0..8].try_into().expect("8 bytes"));
    let phi = u32::from_le_bytes(pattern[8..12].try_into().expect("4 bytes"));
    let chi = u32::from_le_bytes(care[8..12].try_into().expect("4 bytes"));
    let vplo = crate::simd::U64x8::splat(plo);
    let vclo = crate::simd::U64x8::splat(clo);
    let vphi = crate::simd::U32x16::splat(phi);
    let vchi = crate::simd::U32x16::splat(chi);
    let zero32 = crate::simd::U32x16::splat(0);
    let groups = count / 16;
    let mut lo = [0u64; 16];
    let mut hi = [0u32; 16];
    for g in 0..groups {
        for k in 0..16 {
            let o = first_offset + (g * 16 + k) * stride_bytes;
            lo[k] = u64::from_le_bytes(bytes[o..o + 8].try_into().expect("8 bytes"));
            hi[k] = u32::from_le_bytes(bytes[o + 8..o + 12].try_into().expect("4 bytes"));
        }
        let hi_bits = crate::simd::U32x16::from_array(hi)
            .ternlog::<{ crate::simd::ternlog::XOR_AND }>(vphi, vchi)
            .eq_bitmask(zero32);
        let mut lo_bits = 0u16;
        for (half, arr) in lo.as_chunks::<8>().0.iter().enumerate() {
            let r = crate::simd::U64x8::from_array(*arr)
                .ternlog::<{ crate::simd::ternlog::XOR_AND }>(vplo, vclo)
                .to_array();
            for (lane, &x) in r.iter().enumerate() {
                lo_bits |= ((x == 0) as u16) << (half * 8 + lane);
            }
        }
        let bits = hi_bits & lo_bits;
        out_words[g / 4] |= (bits as u64) << ((g % 4) * 16);
    }
    for i in (groups * 16)..count {
        let o = first_offset + i * stride_bytes;
        let l = u64::from_le_bytes(bytes[o..o + 8].try_into().expect("8 bytes"));
        let h = u32::from_le_bytes(bytes[o + 8..o + 12].try_into().expect("4 bytes"));
        if (l ^ plo) & clo == 0 && (h ^ phi) & chi == 0 {
            out_words[i / 64] |= 1u64 << (i % 64);
        }
    }
}

// ── Gated predicates: `*_to_mask_under` (mask-risc `Pred { under }`) ────────
//
// `out[w] = under[w] & pred(values)[w]`, with the predicate EVALUATED only
// on words where `under[w] != 0` — the survivor-word skip. The COMPARE cost
// is proportional to the live words of `under`; the per-word gate test and
// zero store remain proportional to `rows / 64`. The skip is word-granular,
// so what a frontier pays is its share of live WORDS, not of live rows: a
// frontier whose survivors occupy 3 % of the gate words pays 3 % of the
// compares, and a sparse frontier spread across every word pays all of
// them. (Measured: the counting closure in `under_skips_compares_on_empty_
// gate_words` sees 20 of 40 groups under an alternating gate — a call
// count, not a timing.) The alternative — a full `*_to_mask` pass followed
// by `mask_and` — costs the whole plane twice and is exactly what a mask
// program must not do once its first predicate has already narrowed the
// population.
//
// This engine skips at word granularity (64 rows), the unit the caller
// already holds: one word is four 16-lane groups (or eight 8-lane groups),
// and the skip decision is one `u64 != 0` test. A per-group skip (16 rows)
// would be reachable with one extra AND per group and is deliberately not
// done. An executor is free to skip COARSER (mask-risc's IR speaks of
// 1024-row chunks); the result is identical by construction — a skipped
// chunk is an all-zero gate, and `pred & 0 == 0` — though no test performs
// a coarser skip. Only the cost differs.
//
// Semantics are otherwise those of the ungated sibling, and the tail law is
// inherited from it: the predicate's bits past `n` are zero, so
// `pred & under` conforms EVEN IF `under` carries phantom tail bits — a
// gate never needs cleaning before it is applied. Full overwrite: surplus
// words of `out_words` are written zero, as in every `*_to_mask`.

/// The shared survivor-skip loop under every `*_to_mask_under`.
///
/// `group_bits(lanes, live)` packs one register's worth of predicate results
/// — `lanes` is the full `[T; L]` (zero-padded when `live < L`) and the
/// closure masks its answer down to the low `live` bits — into a `u64`
/// whose bits above `live` are zero. The loop places each group at bit
/// `g * L` of its word and ANDs the word with `under[w]`.
#[inline(always)]
fn pack_under<T: Copy + Default, const L: usize>(
    name: &str, values: &[T], under: &[u64], out_words: &mut [u64], group_bits: impl Fn([T; L], usize) -> u64,
) {
    const { assert!(64 % L == 0, "a mask word must hold whole lane groups") }
    let n = values.len();
    let words = mask_words_for(n);
    assert!(out_words.len() >= words, "{name}: out_words.len()={} < required {words}", out_words.len());
    assert!(under.len() >= words, "{name}: under.len()={} < required {words}", under.len());
    for w in out_words[words..].iter_mut() {
        *w = 0;
    }
    for w in 0..words {
        let gate = under[w];
        if gate == 0 {
            out_words[w] = 0;
            continue;
        }
        let mut bits = 0u64;
        let mut start = w * 64;
        let end = n.min(start + 64);
        let mut shift = 0;
        while start < end {
            let live = (end - start).min(L);
            let group = &values[start..start + live];
            let lanes: [T; L] = if live == L {
                group.try_into().expect("a full group is exactly L lanes")
            } else {
                pad_tail(group)
            };
            let g = group_bits(lanes, live);
            // The closure contract (bits above `live` are zero) is what keeps
            // one group from corrupting the next group's bit range.
            debug_assert!(live == 64 || g >> live == 0, "{name}: group bits above live={live} must be zero");
            bits |= g << shift;
            start += L;
            shift += L;
        }
        out_words[w] = bits & gate;
    }
}

/// `live`-lane validity mask for a 16-lane group: all ones for a full group,
/// [`tail_lane_bits`] for a padded one.
#[inline(always)]
fn live16(live: usize) -> u16 {
    if live == 16 {
        u16::MAX
    } else {
        tail_lane_bits(live)
    }
}

/// `live`-lane validity mask for an 8-lane group: all ones for a full group,
/// [`tail_lane_bits_8`] for a padded one.
#[inline(always)]
fn live8(live: usize) -> u8 {
    if live == 8 {
        u8::MAX
    } else {
        tail_lane_bits_8(live)
    }
}

/// `live`-lane validity mask for a 64-lane group (one `u8x64` register is one
/// whole mask word): all ones for a full group, the low `live` bits otherwise.
#[inline(always)]
fn live64(live: usize) -> u64 {
    if live == 64 {
        u64::MAX
    } else {
        word_range_mask(0, live)
    }
}

/// The shared full-scan packer under every contiguous `*_to_mask` predicate —
/// the ungated sibling of [`pack_under`].
///
/// `group_bits(lanes, live)` packs one register's worth of predicate results:
/// `lanes` is the full `[T; L]` (zero-padded when `live < L`, i.e. only for
/// the final group) and the closure masks its answer down to the low `live`
/// bits. The body iterates `as_chunks::<L>()` — fixed-size loads, no index
/// arithmetic — and the remainder is the ONE spelling of the predicate tail:
/// zero-padded into a register, run through the same closure as the body,
/// masked to `live`. This is where the 12 hand-rolled `if !tail.is_empty()`
/// branches that preceded it were folded (2026-09-16); the mask-algebra tails
/// (`mask_and` and friends) are a different kind and are NOT served here.
#[inline(always)]
fn pack<T: Copy + Default, const L: usize>(
    name: &str, values: &[T], out_words: &mut [u64], group_bits: impl Fn([T; L], usize) -> u64,
) {
    const { assert!(64 % L == 0, "a mask word must hold whole lane groups") }
    let k = 64 / L;
    let n = values.len();
    let words = mask_words_for(n);
    assert!(out_words.len() >= words, "{name}: out_words.len()={} < required {words}", out_words.len());
    // Zero first: makes the "trailing bits are 0" guarantee structural.
    for w in out_words.iter_mut() {
        *w = 0;
    }
    let (chunks, tail) = values.as_chunks::<L>();
    for (g, chunk) in chunks.iter().enumerate() {
        out_words[g / k] |= group_bits(*chunk, L) << ((g % k) * L);
    }
    if !tail.is_empty() {
        let g = chunks.len();
        let live = tail.len();
        let bits = group_bits(pad_tail(tail), live);
        // The closure contract (bits above `live` are zero) is what keeps a
        // padding lane from ever contributing a match.
        debug_assert!(live == 64 || bits >> live == 0, "{name}: tail bits above live={live} must be zero");
        out_words[g / k] |= bits << ((g % k) * L);
    }
}

/// Packs `values[i] > threshold` (signed) AND `under`: bit `i` is set iff
/// both hold, and words where `under` has no survivor are written zero
/// **without evaluating the compare** — the survivor-word skip whose cost
/// is proportional to `under`'s live words, not to `values.len()`.
///
/// Same comparison, bit order, and full-overwrite contract as
/// [`gt_i32_to_mask`]; `under`'s bits past `values.len()` are harmless (the
/// predicate's own tail is zero, so the AND conforms regardless).
///
/// # Panics
///
/// Panics if `out_words.len()` or `under.len()` is below
/// `values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{gt_i32_to_mask, gt_i32_to_mask_under};
///
/// let values = [5i32, -5, 9, 12];
/// let mut full = [0u64; 1];
/// gt_i32_to_mask(&values, 0, &mut full);
/// assert_eq!(full[0], 0b1101);
/// // gate out element 3: the answer is the ungated mask AND the gate
/// let mut gated = [0u64; 1];
/// gt_i32_to_mask_under(&values, 0, &[0b0111], &mut gated);
/// assert_eq!(gated[0], 0b0101);
/// // an empty gate word is written zero and its compares are skipped
/// gt_i32_to_mask_under(&values, 0, &[0], &mut gated);
/// assert_eq!(gated[0], 0);
/// ```
#[inline]
pub fn gt_i32_to_mask_under(values: &[i32], threshold: i32, under: &[u64], out_words: &mut [u64]) {
    let t = crate::simd::I32x16::splat(threshold);
    pack_under::<i32, 16>("gt_i32_to_mask_under", values, under, out_words, |lanes, live| {
        (crate::simd::I32x16::from_array(lanes).gt_bitmask(t) & live16(live)) as u64
    });
}

/// Packs `values[i] < threshold` (signed) AND `under`, using the
/// survivor-word skip: words where `under` has no live bit are written zero
/// without evaluating the compare. Same contract as [`gt_i32_to_mask_under`];
/// gated form of [`lt_i32_to_mask`].
///
/// # Panics
///
/// Panics if `out_words.len()` or `under.len()` is below
/// `values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{lt_i32_to_mask, lt_i32_to_mask_under};
///
/// let values = [5i32, -5, -1, 10];
/// let mut full = [0u64; 1];
/// lt_i32_to_mask(&values, 0, &mut full);
/// assert_eq!(full[0], 0b0110);
/// // gate out element 2: only element 1 survives
/// let mut gated = [0u64; 1];
/// lt_i32_to_mask_under(&values, 0, &[0b1011], &mut gated);
/// assert_eq!(gated[0], 0b0010);
/// // an empty gate word is written zero and its compares are skipped
/// lt_i32_to_mask_under(&values, 0, &[0], &mut gated);
/// assert_eq!(gated[0], 0);
/// ```
#[inline]
pub fn lt_i32_to_mask_under(values: &[i32], threshold: i32, under: &[u64], out_words: &mut [u64]) {
    let t = crate::simd::I32x16::splat(threshold);
    pack_under::<i32, 16>("lt_i32_to_mask_under", values, under, out_words, |lanes, live| {
        (t.gt_bitmask(crate::simd::I32x16::from_array(lanes)) & live16(live)) as u64
    });
}

/// Packs `values[i] >= threshold` (signed) AND `under`, complementing
/// [`lt_i32_to_mask_under`]'s per-group bits *inside* the group so the tail
/// law stays [`live16`]'s to enforce, never a whole-word complement. Same
/// contract as [`gt_i32_to_mask_under`]; gated form of [`ge_i32_to_mask`].
///
/// # Panics
///
/// Panics if `out_words.len()` or `under.len()` is below
/// `values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{ge_i32_to_mask, ge_i32_to_mask_under};
///
/// let values = [5i32, -5, -1, 10];
/// let mut full = [0u64; 1];
/// ge_i32_to_mask(&values, 0, &mut full);
/// assert_eq!(full[0], 0b1001);
/// // gate out element 3: only element 0 survives
/// let mut gated = [0u64; 1];
/// ge_i32_to_mask_under(&values, 0, &[0b0011], &mut gated);
/// assert_eq!(gated[0], 0b0001);
/// // an empty gate word is written zero and its compares are skipped
/// ge_i32_to_mask_under(&values, 0, &[0], &mut gated);
/// assert_eq!(gated[0], 0);
/// ```
#[inline]
pub fn ge_i32_to_mask_under(values: &[i32], threshold: i32, under: &[u64], out_words: &mut [u64]) {
    let t = crate::simd::I32x16::splat(threshold);
    pack_under::<i32, 16>("ge_i32_to_mask_under", values, under, out_words, |lanes, live| {
        (!t.gt_bitmask(crate::simd::I32x16::from_array(lanes)) & live16(live)) as u64
    });
}

/// Packs `values[i] <= threshold` (signed) AND `under`, complementing
/// [`gt_i32_to_mask_under`]'s per-group bits *inside* the group. Same
/// contract as [`gt_i32_to_mask_under`]; gated form of [`le_i32_to_mask`].
///
/// # Panics
///
/// Panics if `out_words.len()` or `under.len()` is below
/// `values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{le_i32_to_mask, le_i32_to_mask_under};
///
/// let values = [5i32, -5, -1, 10];
/// let mut full = [0u64; 1];
/// le_i32_to_mask(&values, 0, &mut full);
/// assert_eq!(full[0], 0b0110);
/// // gate out element 2: only element 1 survives
/// let mut gated = [0u64; 1];
/// le_i32_to_mask_under(&values, 0, &[0b1011], &mut gated);
/// assert_eq!(gated[0], 0b0010);
/// // an empty gate word is written zero and its compares are skipped
/// le_i32_to_mask_under(&values, 0, &[0], &mut gated);
/// assert_eq!(gated[0], 0);
/// ```
#[inline]
pub fn le_i32_to_mask_under(values: &[i32], threshold: i32, under: &[u64], out_words: &mut [u64]) {
    let t = crate::simd::I32x16::splat(threshold);
    pack_under::<i32, 16>("le_i32_to_mask_under", values, under, out_words, |lanes, live| {
        (!crate::simd::I32x16::from_array(lanes).gt_bitmask(t) & live16(live)) as u64
    });
}

/// Packs `values[i] != needle` (signed) AND `under`, in one pass as
/// `(needle > v) | (v > needle)` — no complement, so the tail is zero by
/// construction, same as the ungated sibling. Same contract as
/// [`gt_i32_to_mask_under`]; gated form of [`ne_i32_to_mask`].
///
/// # Panics
///
/// Panics if `out_words.len()` or `under.len()` is below
/// `values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{ne_i32_to_mask, ne_i32_to_mask_under};
///
/// let values = [7i32, 1, 7, 2];
/// let mut full = [0u64; 1];
/// ne_i32_to_mask(&values, 7, &mut full);
/// assert_eq!(full[0], 0b1010);
/// // gate out element 3: only element 1 survives
/// let mut gated = [0u64; 1];
/// ne_i32_to_mask_under(&values, 7, &[0b0011], &mut gated);
/// assert_eq!(gated[0], 0b0010);
/// // an empty gate word is written zero and its compares are skipped
/// ne_i32_to_mask_under(&values, 7, &[0], &mut gated);
/// assert_eq!(gated[0], 0);
/// ```
#[inline]
pub fn ne_i32_to_mask_under(values: &[i32], needle: i32, under: &[u64], out_words: &mut [u64]) {
    let t = crate::simd::I32x16::splat(needle);
    pack_under::<i32, 16>("ne_i32_to_mask_under", values, under, out_words, |lanes, live| {
        let v = crate::simd::I32x16::from_array(lanes);
        ((t.gt_bitmask(v) | v.gt_bitmask(t)) & live16(live)) as u64
    });
}

/// Packs `values[i] == needle` (signed, exact) AND `under`, complementing
/// [`ne_i32_to_mask_under`]'s per-group bits *inside* the group. Same
/// contract as [`gt_i32_to_mask_under`]; gated form of [`eq_i32_to_mask`].
///
/// # Panics
///
/// Panics if `out_words.len()` or `under.len()` is below
/// `values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{eq_i32_to_mask, eq_i32_to_mask_under};
///
/// let values = [7i32, 1, 7, 2];
/// let mut full = [0u64; 1];
/// eq_i32_to_mask(&values, 7, &mut full);
/// assert_eq!(full[0], 0b0101);
/// // gate out element 2: only element 0 survives
/// let mut gated = [0u64; 1];
/// eq_i32_to_mask_under(&values, 7, &[0b0011], &mut gated);
/// assert_eq!(gated[0], 0b0001);
/// // an empty gate word is written zero and its compares are skipped
/// eq_i32_to_mask_under(&values, 7, &[0], &mut gated);
/// assert_eq!(gated[0], 0);
/// ```
#[inline]
pub fn eq_i32_to_mask_under(values: &[i32], needle: i32, under: &[u64], out_words: &mut [u64]) {
    let t = crate::simd::I32x16::splat(needle);
    pack_under::<i32, 16>("eq_i32_to_mask_under", values, under, out_words, |lanes, live| {
        let v = crate::simd::I32x16::from_array(lanes);
        (!(t.gt_bitmask(v) | v.gt_bitmask(t)) & live16(live)) as u64
    });
}

/// Packs `values[i] == needle` (unsigned, exact) AND `under`. Same contract
/// as [`gt_i32_to_mask_under`]; gated form of [`eq_u32_to_mask`].
///
/// # Panics
///
/// Panics if `out_words.len()` or `under.len()` is below
/// `values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{eq_u32_to_mask, eq_u32_to_mask_under};
///
/// let values = [7u32, 1, 7, 2];
/// let mut full = [0u64; 1];
/// eq_u32_to_mask(&values, 7, &mut full);
/// assert_eq!(full[0], 0b0101);
/// // gate out element 2: only element 0 survives
/// let mut gated = [0u64; 1];
/// eq_u32_to_mask_under(&values, 7, &[0b0011], &mut gated);
/// assert_eq!(gated[0], 0b0001);
/// // an empty gate word is written zero and its compares are skipped
/// eq_u32_to_mask_under(&values, 7, &[0], &mut gated);
/// assert_eq!(gated[0], 0);
/// ```
#[inline]
pub fn eq_u32_to_mask_under(values: &[u32], needle: u32, under: &[u64], out_words: &mut [u64]) {
    let needle_v = crate::simd::U32x16::splat(needle);
    pack_under::<u32, 16>("eq_u32_to_mask_under", values, under, out_words, |lanes, live| {
        (crate::simd::U32x16::from_array(lanes).eq_bitmask(needle_v) & live16(live)) as u64
    });
}

/// Packs `values[i] != needle` (unsigned) AND `under`, complementing
/// [`eq_u32_to_mask_under`]'s per-group bits *inside* the group. Same
/// contract as [`gt_i32_to_mask_under`]; gated form of [`ne_u32_to_mask`].
///
/// # Panics
///
/// Panics if `out_words.len()` or `under.len()` is below
/// `values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{ne_u32_to_mask, ne_u32_to_mask_under};
///
/// let values = [7u32, 1, 7, 2];
/// let mut full = [0u64; 1];
/// ne_u32_to_mask(&values, 7, &mut full);
/// assert_eq!(full[0], 0b1010);
/// // gate out element 3: only element 1 survives
/// let mut gated = [0u64; 1];
/// ne_u32_to_mask_under(&values, 7, &[0b0011], &mut gated);
/// assert_eq!(gated[0], 0b0010);
/// // an empty gate word is written zero and its compares are skipped
/// ne_u32_to_mask_under(&values, 7, &[0], &mut gated);
/// assert_eq!(gated[0], 0);
/// ```
#[inline]
pub fn ne_u32_to_mask_under(values: &[u32], needle: u32, under: &[u64], out_words: &mut [u64]) {
    let needle_v = crate::simd::U32x16::splat(needle);
    pack_under::<u32, 16>("ne_u32_to_mask_under", values, under, out_words, |lanes, live| {
        (!crate::simd::U32x16::from_array(lanes).eq_bitmask(needle_v) & live16(live)) as u64
    });
}

/// Care-masked match of `values[i]` against `pattern` (bits selected by
/// `care`) AND `under`: element `i` survives iff `((values[i] ^ pattern) &
/// care) == 0` **and** its gate bit is set. Same contract as
/// [`gt_i32_to_mask_under`]; gated form of [`ternary_match_u32_to_mask`].
///
/// # Panics
///
/// Panics if `out_words.len()` or `under.len()` is below
/// `values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{ternary_match_u32_to_mask, ternary_match_u32_to_mask_under};
///
/// let values = [0b1010u32, 0b1110, 0b0010, 0b1011];
/// let mut full = [0u64; 1];
/// ternary_match_u32_to_mask(&values, 0b1010, 0b1011, &mut full);
/// assert_eq!(full[0], 0b0011);
/// // gate out element 1: only element 0 survives
/// let mut gated = [0u64; 1];
/// ternary_match_u32_to_mask_under(&values, 0b1010, 0b1011, &[0b1101], &mut gated);
/// assert_eq!(gated[0], 0b0001);
/// // an empty gate word is written zero and its compares are skipped
/// ternary_match_u32_to_mask_under(&values, 0b1010, 0b1011, &[0], &mut gated);
/// assert_eq!(gated[0], 0);
/// ```
#[inline]
pub fn ternary_match_u32_to_mask_under(values: &[u32], pattern: u32, care: u32, under: &[u64], out_words: &mut [u64]) {
    let p = crate::simd::U32x16::splat(pattern);
    let c = crate::simd::U32x16::splat(care);
    let zero = crate::simd::U32x16::splat(0);
    pack_under::<u32, 16>("ternary_match_u32_to_mask_under", values, under, out_words, |lanes, live| {
        let bits = crate::simd::U32x16::from_array(lanes)
            .ternlog::<{ crate::simd::ternlog::XOR_AND }>(p, c)
            .eq_bitmask(zero);
        (bits & live16(live)) as u64
    });
}

/// The 64-bit sibling of [`ternary_match_u32_to_mask_under`]: care-masked
/// match of `values[i]` against `pattern`/`care` AND `under`. Same contract
/// as [`gt_i32_to_mask_under`]; gated form of [`ternary_match_u64_to_mask`].
///
/// # Panics
///
/// Panics if `out_words.len()` or `under.len()` is below
/// `values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{ternary_match_u64_to_mask, ternary_match_u64_to_mask_under};
///
/// let values = [0b1010u64, 0b1110, 0b0010, 0b1011];
/// let mut full = [0u64; 1];
/// ternary_match_u64_to_mask(&values, 0b1010, 0b1011, &mut full);
/// assert_eq!(full[0], 0b0011);
/// // gate out element 1: only element 0 survives
/// let mut gated = [0u64; 1];
/// ternary_match_u64_to_mask_under(&values, 0b1010, 0b1011, &[0b1101], &mut gated);
/// assert_eq!(gated[0], 0b0001);
/// // an empty gate word is written zero and its compares are skipped
/// ternary_match_u64_to_mask_under(&values, 0b1010, 0b1011, &[0], &mut gated);
/// assert_eq!(gated[0], 0);
/// ```
#[inline]
pub fn ternary_match_u64_to_mask_under(values: &[u64], pattern: u64, care: u64, under: &[u64], out_words: &mut [u64]) {
    let p = crate::simd::U64x8::splat(pattern);
    let c = crate::simd::U64x8::splat(care);
    pack_under::<u64, 8>("ternary_match_u64_to_mask_under", values, under, out_words, |lanes, live| {
        let r = crate::simd::U64x8::from_array(lanes)
            .ternlog::<{ crate::simd::ternlog::XOR_AND }>(p, c)
            .to_array();
        let mut bits = 0u64;
        for (lane, &x) in r.iter().take(live).enumerate() {
            bits |= ((x == 0) as u64) << lane;
        }
        bits
    });
}

/// Minimum of `values[i]` over set mask bits, `None` when no bit is set.
/// Same bit order and "bits at or past `values.len()` are ignored" contract
/// as [`masked_sum_i32`]; cost proportional to the popcount.
///
/// # Panics
///
/// Panics if `mask_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::masked_min_i32;
///
/// let values = [10i32, -5, 30, 2];
/// // bits 1 and 3 select -5 and 2; the minimum of the two is -5
/// assert_eq!(masked_min_i32(&values, &[0b1010]), Some(-5));
/// ```
#[inline]
pub fn masked_min_i32(values: &[i32], mask_words: &[u64]) -> Option<i32> {
    masked_fold_i32(values, mask_words, i32::min)
}

/// Maximum of `values[i]` over set mask bits, `None` when no bit is set.
/// Contract as [`masked_min_i32`].
///
/// # Panics
///
/// Panics if `mask_words.len() < values.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::masked_max_i32;
///
/// let values = [10i32, -5, 30, 2];
/// // bits 1 and 3 select -5 and 2; the maximum of the two is 2
/// assert_eq!(masked_max_i32(&values, &[0b1010]), Some(2));
/// ```
#[inline]
pub fn masked_max_i32(values: &[i32], mask_words: &[u64]) -> Option<i32> {
    masked_fold_i32(values, mask_words, i32::max)
}

#[inline(always)]
fn masked_fold_i32(values: &[i32], mask_words: &[u64], f: impl Fn(i32, i32) -> i32) -> Option<i32> {
    let n = values.len();
    let words = mask_words_for(n);
    assert!(
        mask_words.len() >= words,
        "masked_fold_i32: mask_words.len()={} < required {}",
        mask_words.len(),
        words
    );
    let mut acc: Option<i32> = None;
    for (w, &word) in mask_words.iter().take(words).enumerate() {
        let base = w * 64;
        let mut bits = word;
        let valid = n - base;
        if valid < 64 {
            bits &= (1u64 << valid) - 1;
        }
        while bits != 0 {
            let i = base + bits.trailing_zeros() as usize;
            bits &= bits - 1;
            acc = Some(match acc {
                None => values[i],
                Some(a) => f(a, values[i]),
            });
        }
    }
    acc
}

/// `dst[i] = if mask bit i { a[i] } else { b[i] }` — the conditional-select
/// (`CASE WHEN`) over a row mask, with no compaction. Bits at or past
/// `a.len()` are ignored. Plain index loop, deliberately: a bit-per-element
/// select over `i32` has no lane wrapper in the mask vocabulary yet, and a
/// scalar loop is the honest shape until one is measured to be needed
/// (no assembly inspection backs a claim about what LLVM emits here).
///
/// # Panics
///
/// Panics unless `a.len() == b.len() == dst.len()` and
/// `mask_words.len() >= a.len().div_ceil(64)`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::blend_i32;
///
/// let a = [1i32, 2, 3, 4];
/// let b = [10i32, 20, 30, 40];
/// let mut dst = [0i32; 4];
/// // bits 0 and 2 (0b0101) pick from `a`; bits 1 and 3 pick from `b`
/// blend_i32(&[0b0101], &a, &b, &mut dst);
/// assert_eq!(dst, [1, 20, 3, 40]);
/// ```
#[inline]
pub fn blend_i32(mask_words: &[u64], a: &[i32], b: &[i32], dst: &mut [i32]) {
    assert_eq!(a.len(), b.len(), "blend_i32: a/b length mismatch");
    assert_eq!(a.len(), dst.len(), "blend_i32: a/dst length mismatch");
    let n = a.len();
    assert!(
        mask_words.len() >= mask_words_for(n),
        "blend_i32: mask_words.len()={} < required {}",
        mask_words.len(),
        mask_words_for(n)
    );
    for i in 0..n {
        let bit = (mask_words[i / 64] >> (i % 64)) & 1;
        dst[i] = if bit == 1 { a[i] } else { b[i] };
    }
}

// ────────────────────────────────────────────────────────────────────────
// Morton lattice axis shift — the D-GTM-1m word-level op (§15); the hex
// probe composes its six rails from these four Cartesian axis moves
// ────────────────────────────────────────────────────────────────────────
//
// Rows are Morton-keyed 2-D (`q` on the even address bits, `r` on the odd
// ones — `hex_tenant_mq_probe.rs::morton`). A mask word covers 64 rows, i.e.
// an 8×8 axial block; within a word, bit index `b` in `0..64` has its local
// `q` sub-coordinate at bits `{0,2,4}` and its local `r` sub-coordinate at
// bits `{1,3,5}`. `mask_shift_morton` moves every set bit one cell along one
// of the four axis directions, entirely as word-level ops: three fixed bit
// permutations cover the cells whose local sub-coordinate does not overflow
// the word, and one carry moves the cells that do into the adjacent word.
//
// `D-GTM-0m` (`.claude/blackboard.md`) measured that this per-active-bit
// shift is the WHOLE cost of a spread step once the mask chain itself is
// ~free; this op removes it by making the shift itself a handful of word
// passes instead of a loop over set bits.

/// One axis direction, as a runtime value (never a const generic — a
/// consumer composing all four, or the two diagonal directions, picks the
/// direction at the call site, not at compile time).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MortonDir {
    /// Increment the even-bit (`q`) sub-coordinate — a lattice axis move;
    /// the hex probe's rail 0 (`(+1, 0)`). The two hex diagonals are
    /// compositions of two variants, never variants themselves.
    PosQ,
    /// Decrement the even-bit (`q`) sub-coordinate (the hex probe's rail 1).
    NegQ,
    /// Increment the odd-bit (`r`) sub-coordinate (the hex probe's rail 2).
    PosR,
    /// Decrement the odd-bit (`r`) sub-coordinate (the hex probe's rail 3).
    NegR,
}

/// Build a 64-bit mask of every `b` in `0..64` for which
/// `(b & test_mask) == test_pattern` — the case-selector predicates §15
/// states literally as `b & M == P`. `const fn` so every `AxisTable` below is
/// computed once, at compile time, never as a runtime literal.
const fn morton_case_mask(test_mask: u32, test_pattern: u32) -> u64 {
    let mut m = 0u64;
    let mut b = 0u32;
    while b < 64 {
        if (b & test_mask) == test_pattern {
            m |= 1u64 << b;
        }
        b += 1;
    }
    m
}

/// One direction's word-level shift recipe: the three interior (non-
/// wrapping) case masks with their signed bit-shift amount (`> 0` is `<<`,
/// `< 0` is `>>`), and the fourth (wrapping) case that carries into the
/// neighbouring word.
struct AxisTable {
    /// `(case mask, signed shift)` for the three cells whose local
    /// sub-coordinate does not overflow the word.
    interior: [(u64, i32); 3],
    /// The cells whose local sub-coordinate wraps and must carry out.
    carry_mask: u64,
    /// Magnitude of the carry's local re-basing shift.
    carry_shift: u32,
    /// `true` for `NegQ`/`NegR`: the carry moves toward the DECREASING
    /// word (`dec_word`) and re-bases with `<<`. `false` for `PosQ`/`PosR`:
    /// toward the INCREASING word (`inc_word`), re-based with `>>`.
    carry_decrements: bool,
}

// The four tables, one per `MortonDir`, exactly as specified: `+r`/`-r` are
// `+q`/`-q`'s tables with every bit position and shift amount doubled (the
// local `r` sub-coordinate sits one bit to the left of `q`'s at every tier).

const POS_Q: AxisTable = AxisTable {
    interior: [(morton_case_mask(1, 0), 1), (morton_case_mask(5, 1), 3), (morton_case_mask(21, 5), 11)],
    carry_mask: morton_case_mask(21, 21),
    carry_shift: 21,
    carry_decrements: false,
};

const NEG_Q: AxisTable = AxisTable {
    interior: [(morton_case_mask(1, 1), -1), (morton_case_mask(5, 4), -3), (morton_case_mask(21, 16), -11)],
    carry_mask: morton_case_mask(21, 0),
    carry_shift: 21,
    carry_decrements: true,
};

const POS_R: AxisTable = AxisTable {
    interior: [(morton_case_mask(2, 0), 2), (morton_case_mask(10, 2), 6), (morton_case_mask(42, 10), 22)],
    carry_mask: morton_case_mask(42, 42),
    carry_shift: 42,
    carry_decrements: false,
};

const NEG_R: AxisTable = AxisTable {
    interior: [(morton_case_mask(2, 2), -2), (morton_case_mask(10, 8), -6), (morton_case_mask(42, 32), -22)],
    carry_mask: morton_case_mask(42, 0),
    carry_shift: 42,
    carry_decrements: true,
};

/// `(self & mask) << shift` if `shift >= 0`, else `(self & mask) >> -shift` —
/// the one interior case, as a whole-register op. `Shl<Self>`/`Shr<Self>` on
/// `U64x8` take a per-lane vector count; a uniform shift is `splat(n)`, the
/// same shape every backend already carries (`d11dd0f` filled the two that
/// did not: AVX2 and nightly-simd).
#[inline(always)]
fn morton_case_shift(v: crate::simd::U64x8, mask: u64, shift: i32) -> crate::simd::U64x8 {
    let masked = v & crate::simd::U64x8::splat(mask);
    if shift >= 0 {
        masked << crate::simd::U64x8::splat(shift as u64)
    } else {
        masked >> crate::simd::U64x8::splat((-shift) as u64)
    }
}

/// The word-space `(x_bits, y_bits)` pair for a field of `n_words` words:
/// the even/odd address-bit masks one tier UP from the cell-level `X_BITS`/
/// `Y_BITS` in `hex_tenant_mq_probe.rs` — the word index is itself the
/// Morton key of the block coordinates, so the same dilated-integer
/// even/odd split applies, just over `log2(n_words)` bits instead of 16.
#[inline(always)]
fn word_axis_bits(n_words: usize) -> (u64, u64) {
    let total_bits = n_words.trailing_zeros();
    let mut x = 0u64;
    let mut b = 0u32;
    while b < total_bits {
        x |= 1u64 << b;
        b += 2;
    }
    (x, x << 1)
}

/// Dilated-integer increment of the `axis_bits` sub-coordinate of word `w`,
/// `other_bits`-coordinate held fixed. `None` at the far edge (`axis_bits`
/// already all-ones) — the same no-wrap contract as the cell-level
/// `neighbour()` in the probe, one tier up.
#[inline(always)]
fn inc_word(w: usize, axis_bits: u64, other_bits: u64) -> Option<usize> {
    let w = w as u64;
    let x = w & axis_bits;
    if x == axis_bits {
        return None;
    }
    let xn = (x | other_bits).wrapping_add(1) & axis_bits;
    Some((xn | (w & other_bits)) as usize)
}

/// Dilated-integer decrement, the mirror of [`inc_word`]. `None` at `x == 0`.
#[inline(always)]
fn dec_word(w: usize, axis_bits: u64, other_bits: u64) -> Option<usize> {
    let w = w as u64;
    let x = w & axis_bits;
    if x == 0 {
        return None;
    }
    let xn = x.wrapping_sub(1) & axis_bits;
    Some((xn | (w & other_bits)) as usize)
}

/// `dst |= src` shifted one cell along `dir` on the Morton-keyed 2-D
/// lattice (D-GTM-1m, `.claude/plans/gemm-ternlog-mask-consolidation-v1.md`
/// §15) — the mask-level replacement for the per-active-bit hex neighbour
/// shift `D-GTM-0m` measured as the whole cost of a spread step.
///
/// `src.len()` and `dst.len()` must be equal and a power of **four**: the
/// word index doubles as the Morton key of the field's block coordinates, so
/// `log2(n_words)` (the address bits above one word's own 6) must split
/// evenly between the `q` and `r` halves — a non-square field is refused,
/// never approximated. Cells on the far edge of `dir` produce nothing (no
/// wrap-around). `dst` is **OR-accumulated, not overwritten** — clear it
/// first, or chain several directions (e.g. all six hex neighbours) into one
/// plane by calling this repeatedly with the same `dst`. A hex diagonal
/// (`+q-r` or `-q+r`) is two calls: shift into a scratch buffer along the
/// first axis, then shift that scratch (not `src`) along the second axis
/// into `dst`.
///
/// **The slice IS the field.** A sub-span of a larger field is treated as
/// its own field: no carry enters or leaves the span, so the result is NOT
/// the restriction of the full-field shift — at every span boundary the
/// full field would carry into the adjacent block and the span version
/// emits nothing. This op receives only a slice and cannot check where it
/// sits in a parent field. A caller shifting a tile in place must
/// independently hold BOTH that (a) the span's origin is aligned to its own
/// length in the parent field's Morton order (a trie node: `lo % len == 0`)
/// and (b) the source is confined to the span, or accept that a boundary
/// carry is dropped. `examples/hex_tenant_mq_probe.rs` asserts (a) at its
/// call site and holds (b) by construction (`state ⊆ tile`).
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`, or if that length is not `4^k` for
/// some `k >= 0`.
///
/// # Examples
///
/// ```
/// use ndarray::simd::{mask_shift_morton, MortonDir};
///
/// // a 64-word field (an 8×8 grid of 8×8 blocks = 64×64 cells).
/// // cell 0 (q=0, r=0, bit 0 of word 0) shifted +q lands at cell (1, 0),
/// // which is bit 1 of the same word (case A: b&1==0 -> b+1).
/// let mut src = vec![0u64; 64];
/// src[0] = 1;
/// let mut dst = vec![0u64; 64];
/// mask_shift_morton(&src, MortonDir::PosQ, &mut dst);
/// assert_eq!(dst[0], 0b10);
/// ```
#[inline]
pub fn mask_shift_morton(src: &[u64], dir: MortonDir, dst: &mut [u64]) {
    assert_eq!(src.len(), dst.len(), "mask_shift_morton: src/dst length mismatch");
    let n_words = src.len();
    assert!(
        n_words.is_power_of_two() && n_words.trailing_zeros().is_multiple_of(2),
        "mask_shift_morton: n_words={n_words} is not a square Morton field (need n_words = 4^k, \
         so the word index carries an even q/r bit split)"
    );

    let table = match dir {
        MortonDir::PosQ => &POS_Q,
        MortonDir::NegQ => &NEG_Q,
        MortonDir::PosR => &POS_R,
        MortonDir::NegR => &NEG_R,
    };

    // Pass 1 — the interior permutation, vectorized: and/shl-or-shr/or over
    // whole `U64x8` registers, OR-accumulated into `dst`. Same chunk/tail
    // shape as `mask_xor_assign` (both `src` and `dst` tails are read, since
    // this is an accumulate, not an overwrite).
    const L: usize = crate::simd::U64x8::LANES;
    let (cs, ts) = src.as_chunks::<L>();
    let (cd, td) = dst.as_chunks_mut::<L>();
    for (s, d) in cs.iter().zip(cd.iter_mut()) {
        let vs = crate::simd::U64x8::from_array(*s);
        let mut acc = crate::simd::U64x8::from_array(*d);
        for &(mask, shift) in &table.interior {
            acc |= morton_case_shift(vs, mask, shift);
        }
        *d = acc.to_array();
    }
    if !ts.is_empty() {
        let vs = crate::simd::U64x8::from_array(pad_tail(ts));
        let mut acc = crate::simd::U64x8::from_array(pad_tail(td));
        for &(mask, shift) in &table.interior {
            acc |= morton_case_shift(vs, mask, shift);
        }
        td.copy_from_slice(&acc.to_array()[..td.len()]);
    }

    // Pass 2 — the one-word carry, scalar (there is exactly one neighbour
    // word per source word, so there is nothing here for a vector lane to
    // parallelize over).
    let (x_bits, y_bits) = word_axis_bits(n_words);
    let (axis_bits, other_bits) = match dir {
        MortonDir::PosQ | MortonDir::NegQ => (x_bits, y_bits),
        MortonDir::PosR | MortonDir::NegR => (y_bits, x_bits),
    };
    for (w, &s) in src.iter().enumerate() {
        let carry_bits = s & table.carry_mask;
        if carry_bits == 0 {
            continue;
        }
        let neighbour = if table.carry_decrements {
            dec_word(w, axis_bits, other_bits)
        } else {
            inc_word(w, axis_bits, other_bits)
        };
        if let Some(nb) = neighbour {
            dst[nb] |= if table.carry_decrements {
                carry_bits << table.carry_shift
            } else {
                carry_bits >> table.carry_shift
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    // ── mask_ternlog (any 3-input Boolean, one pass) ─────────────────────────

    /// Truth-table reference evaluated one BIT at a time — independent of
    /// every backend lane (the scalar tail it once shared with them is gone —
    /// tails run through the same packed op as the body).
    fn ref_ternlog_bitwise(a: u64, b: u64, c: u64, imm: i32) -> u64 {
        let mut r = 0u64;
        for bit in 0..64 {
            let idx = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1);
            if (imm >> idx) & 1 == 1 {
                r |= 1u64 << bit;
            }
        }
        r
    }

    /// Exercise one IMM over the family's standard length set, both forms,
    /// against the bit-serial reference.
    fn check_ternlog_imm<const IMM: i32>() {
        for &len in &[0usize, 1, 2, 7, 8, 9, 15, 16, 17, 31, 63, 64, 100] {
            let mut seed = 0x7E12_10C0_0000_0001 ^ (IMM as u64);
            let a: Vec<u64> = (0..len).map(|_| splitmix64(&mut seed)).collect();
            let b: Vec<u64> = (0..len).map(|_| splitmix64(&mut seed)).collect();
            let c: Vec<u64> = (0..len).map(|_| splitmix64(&mut seed)).collect();
            let expect: Vec<u64> = (0..len)
                .map(|i| ref_ternlog_bitwise(a[i], b[i], c[i], IMM))
                .collect();

            let mut dst = vec![0xDEAD_BEEFu64; len];
            mask_ternlog::<IMM>(&a, &b, &c, &mut dst);
            assert_eq!(dst, expect, "mask_ternlog imm={IMM:#04x} len={len}");

            let mut dst = a.clone();
            mask_ternlog_assign::<IMM>(&mut dst, &b, &c);
            assert_eq!(dst, expect, "mask_ternlog_assign imm={IMM:#04x} len={len}");
        }
    }

    #[test]
    fn mask_ternlog_matches_bitwise_reference_for_all_256_tables() {
        // Const generics need a literal per instantiation; a macro unrolls
        // all 256 so no table is left to "obviously the same as the others".
        macro_rules! all_imms {
            ($($imm:literal),* $(,)?) => { $( check_ternlog_imm::<$imm>(); )* };
        }
        all_imms!(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11,
            0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23,
            0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31, 0x32, 0x33, 0x34, 0x35,
            0x36, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,
            0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
            0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B,
            0x6C, 0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D,
            0x7E, 0x7F, 0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F,
            0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F, 0xA0, 0xA1,
            0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xAB, 0xAC, 0xAD, 0xAE, 0xAF, 0xB0, 0xB1, 0xB2, 0xB3,
            0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF, 0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5,
            0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF, 0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,
            0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF, 0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9,
            0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF, 0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB,
            0xFC, 0xFD, 0xFE, 0xFF,
        );
    }

    #[test]
    fn mask_ternlog_and3_equals_two_and_passes() {
        // The consumer motivation: `selected & src & gate` as one AND3 pass
        // must be bit-identical to the two-pass `mask_and_assign` spelling.
        use crate::simd::ternlog::AND3;
        let mut seed = 0xA3D3_0000_0000_0001;
        let sel: Vec<u64> = (0..20).map(|_| splitmix64(&mut seed)).collect();
        let src: Vec<u64> = (0..20).map(|_| splitmix64(&mut seed)).collect();
        let gate: Vec<u64> = (0..20).map(|_| splitmix64(&mut seed)).collect();

        let mut two_pass = sel.clone();
        mask_and_assign(&mut two_pass, &src);
        mask_and_assign(&mut two_pass, &gate);

        let mut one_pass = sel.clone();
        mask_ternlog_assign::<AND3>(&mut one_pass, &src, &gate);
        assert_eq!(one_pass, two_pass, "AND3 == and∘and");

        // Non-vacuous: the narrowing must have removed bits, and both narrower
        // operands must have contributed (each alone leaves a different set).
        assert_ne!(one_pass, sel, "AND3 must narrow on this corpus");
        let mut src_only = sel.clone();
        mask_and_assign(&mut src_only, &src);
        assert_ne!(one_pass, src_only, "gate must contribute, not just src");
    }

    #[test]
    fn mask_ternlog_tail_conforms_iff_imm_is_even() {
        // 2 words = the `mask_words_for(70)` shape; tail = word 1 bits 7..63.
        const TAIL_MASK: u64 = !0x7Fu64;
        let a = [0x1234_5678_9ABC_DEF0u64, 0x0000_0000_0000_005Bu64];
        let b = [0x0F0F_0F0F_0F0F_0F0Fu64, 0x0000_0000_0000_0071u64];
        let c = [0xFFFF_0000_FFFF_0000u64, 0x0000_0000_0000_002Eu64];
        for m in [&a, &b, &c] {
            assert_eq!(m[1] & TAIL_MASK, 0, "fixture precondition: conforming inputs");
        }
        use crate::simd::ternlog::{AND3, MAJ3, OR3, XOR3};

        // Every named table is even: tail stays zero.
        let mut d = [0xDEAD_BEEFu64; 2];
        mask_ternlog::<AND3>(&a, &b, &c, &mut d);
        assert_eq!(d[1] & TAIL_MASK, 0, "AND3 tail");
        mask_ternlog::<OR3>(&a, &b, &c, &mut d);
        assert_eq!(d[1] & TAIL_MASK, 0, "OR3 tail");
        mask_ternlog::<MAJ3>(&a, &b, &c, &mut d);
        assert_eq!(d[1] & TAIL_MASK, 0, "MAJ3 tail");
        mask_ternlog::<XOR3>(&a, &b, &c, &mut d);
        assert_eq!(d[1] & TAIL_MASK, 0, "XOR3 tail");

        // The can-it-fire half: an ODD table (NOR3 = 0x01, true of all-zero
        // inputs) sets every tail bit, so the doc's "iff even" is a real
        // boundary and not a restatement of the inputs.
        mask_ternlog::<0x01>(&a, &b, &c, &mut d);
        assert_eq!(d[1] & TAIL_MASK, TAIL_MASK, "odd IMM fills the tail");

        // Subset-shaped table against dirty b/c: still a subset of a.
        let dirty = [u64::MAX; 2];
        mask_ternlog::<AND3>(&a, &dirty, &dirty, &mut d);
        assert_eq!(d, a, "AND3 against all-ones is a");
        assert_eq!(d[1] & TAIL_MASK, 0, "AND3 tail follows a's tail");
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn mask_ternlog_rejects_length_mismatch() {
        let mut dst = [0u64; 4];
        mask_ternlog::<0x80>(&[0u64; 4], &[0u64; 4], &[0u64; 3], &mut dst);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn mask_ternlog_assign_rejects_length_mismatch() {
        let mut a = [0u64; 4];
        mask_ternlog_assign::<0x80>(&mut a, &[0u64; 3], &[0u64; 4]);
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

    // ── mask_gather_u32 / mask_scatter_or_u32 / masked_group_sum_i32 ──
    //
    // The three data-indexed permutation/scatter primitives: each is checked
    // against a plain `for` loop over indices (never `Vec<bool>` — the spec's
    // own naive reference shape), at the empty case, at both sides of a word
    // boundary (67 and 130, neither a multiple of 64), with an out-of-range
    // address dropped rather than panicking, and with a garbage-filled output
    // buffer to prove the tail is actually written zero, not merely OR-ed.

    fn naive_gather(src: &[u64], src_rows: usize, index: &[u32]) -> Vec<bool> {
        index
            .iter()
            .map(|&idx| {
                let idx = idx as usize;
                idx < src_rows && (src[idx / 64] >> (idx % 64)) & 1 == 1
            })
            .collect()
    }

    fn bits_to_words(bits: &[bool]) -> Vec<u64> {
        let mut out = vec![0u64; bits.len().div_ceil(64).max(1)];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                out[i / 64] |= 1u64 << (i % 64);
            }
        }
        out
    }

    #[test]
    fn mask_gather_u32_empty_index_writes_nothing() {
        let src = [0u64; 1];
        let index: [u32; 0] = [];
        let mut out = [0xFFFF_FFFF_FFFF_FFFFu64; 1];
        mask_gather_u32(&src, 3, &index, &mut out);
        assert_eq!(out[0], 0, "an empty index gathers no bits and clears the buffer");
    }

    #[test]
    fn mask_gather_u32_matches_naive_reference_across_the_tail() {
        for &n in &[0usize, 1, 63, 64, 65, 67, 130] {
            let mut seed = 0x1357_9BDF_2468_ACE0u64;
            let src_rows = 40usize;
            let src_words = src_rows.div_ceil(64);
            let src: Vec<u64> = (0..src_words).map(|_| splitmix(&mut seed)).collect();
            // Half the index values genuinely address `src`; the rest are
            // deliberately out of range, so the "false by contract" arm is
            // actually exercised, not merely plausible.
            let index: Vec<u32> = (0..n)
                .map(|i| {
                    if i % 3 == 0 {
                        (src_rows as u64 + 5 + i as u64) as u32
                    } else {
                        (splitmix(&mut seed) % src_rows as u64) as u32
                    }
                })
                .collect();
            let want = bits_to_words(&naive_gather(&src, src_rows, &index));
            let out_words = n.div_ceil(64).max(1);
            let mut out = vec![0xFFFF_FFFF_FFFF_FFFFu64; out_words + 1]; // dirty, over-long
            mask_gather_u32(&src, src_rows, &index, &mut out);
            assert_eq!(&out[..want.len()], &want[..], "gather mismatch at n={n}");
            assert_eq!(out[out_words], 0, "surplus word must be cleared at n={n}");
        }
    }

    #[test]
    fn mask_gather_u32_src_rows_smaller_than_max_index_is_false_not_a_panic() {
        // src has only 2 rows; every index in `index` is >= 2, so every
        // result bit must be false — and the call must not panic reading
        // past a 2-row src.
        let src = [0b11u64]; // rows 0 and 1 both set, but out of reach
        let index = [2u32, 5, 100, u32::MAX];
        let mut out = [0u64; 1];
        mask_gather_u32(&src, 2, &index, &mut out);
        assert_eq!(out[0], 0);
    }

    #[test]
    #[should_panic(expected = "out_words.len()")]
    fn mask_gather_u32_rejects_short_out_buffer() {
        let src = [0u64; 1];
        let index = vec![0u32; 65];
        let mut out = [0u64; 1]; // 65 elements need 2 words
        mask_gather_u32(&src, 1, &index, &mut out);
    }

    #[test]
    #[should_panic(expected = "src.len()")]
    fn mask_gather_u32_rejects_short_src_buffer() {
        let src = [0u64; 1]; // covers only 64 rows
        let index = [0u32];
        let mut out = [0u64; 1];
        mask_gather_u32(&src, 65, &index, &mut out); // claims 65 rows
    }

    fn naive_scatter(src_bits: &[bool], index: &[u32], out_rows: usize) -> Vec<bool> {
        let mut out = vec![false; out_rows];
        for (i, &selected) in src_bits.iter().enumerate() {
            if selected {
                let t = index[i] as usize;
                if t < out_rows {
                    out[t] = true;
                }
            }
        }
        out
    }

    #[test]
    fn mask_scatter_or_u32_empty_index_writes_nothing() {
        let src: [u64; 0] = [];
        let index: [u32; 0] = [];
        let mut out = [0u64; 1]; // caller zeroes before the first call
        mask_scatter_or_u32(&src, &index, &mut out, 10);
        assert_eq!(out[0], 0, "no source rows selected ⇒ output unchanged, nothing set");
    }

    #[test]
    fn mask_scatter_or_u32_matches_naive_reference_across_the_tail() {
        for &n in &[0usize, 1, 63, 64, 65, 67, 130] {
            let mut seed = 0xC0FF_EE00_1234_5678u64;
            let out_rows = 50usize;
            let src_bits: Vec<bool> = (0..n).map(|_| splitmix(&mut seed) & 1 == 1).collect();
            let src = bits_to_words(&src_bits);
            // Every third target is deliberately out of range.
            let index: Vec<u32> = (0..n)
                .map(|i| {
                    if i % 3 == 0 {
                        (out_rows as u64 + 7 + i as u64) as u32
                    } else {
                        (splitmix(&mut seed) % out_rows as u64) as u32
                    }
                })
                .collect();
            let want = bits_to_words(&naive_scatter(&src_bits, &index, out_rows));
            let out_words = out_rows.div_ceil(64);
            // Caller zeroes once before the first call; over-long by one word
            // to prove the surplus word is left untouched, not cleared by us.
            let mut out = vec![0u64; out_words + 1];
            mask_scatter_or_u32(&src, &index, &mut out, out_rows);
            assert_eq!(&out[..want.len()], &want[..], "scatter mismatch at n={n}");
            assert_eq!(out[out_words], 0, "surplus word is untouched (stays as the caller left it) at n={n}");
        }
    }

    #[test]
    fn mask_scatter_or_u32_into_out_rows_larger_and_smaller_than_src_population() {
        // 3 source rows, all selected, targeting rows 0/1/2.
        let src = [0b111u64];
        let index = [0u32, 1, 2];
        // Larger out_rows: all three land, nothing else set.
        let mut big = [0u64; 1];
        mask_scatter_or_u32(&src, &index, &mut big, 40);
        assert_eq!(big[0], 0b111);
        // Smaller out_rows: target 2 is out of range and must be dropped,
        // not panic.
        let mut small = [0u64; 1];
        mask_scatter_or_u32(&src, &index, &mut small, 2);
        assert_eq!(small[0], 0b011, "target 2 is out of range for out_rows=2 and is dropped");
    }

    #[test]
    fn mask_scatter_or_u32_repeated_targets_are_a_union_not_a_count() {
        // Every source row scatters to the same target; OR means it is set
        // exactly once, never accumulated or overwritten to something else.
        let src = [0b1111u64];
        let index = [3u32, 3, 3, 3];
        let mut out = [0u64; 1];
        mask_scatter_or_u32(&src, &index, &mut out, 8);
        assert_eq!(out[0], 1u64 << 3);
    }

    #[test]
    fn mask_scatter_or_u32_accumulates_into_a_preloaded_out_buffer() {
        // A bit the call never scatters to must survive; the call's own
        // contribution must land alongside it. This must FAIL if the old
        // whole-buffer zeroing is restored (bit 7 would be cleared).
        let src = [0b1u64]; // row 0 selected
        let index = [3u32]; // scatters to bit 3
        let mut out = [1u64 << 7]; // pre-set bit 7, not touched by this call
        mask_scatter_or_u32(&src, &index, &mut out, 8);
        assert_eq!(out[0], (1u64 << 3) | (1u64 << 7), "pre-existing bit 7 must survive alongside the new bit 3");
    }

    #[test]
    fn mask_scatter_or_u32_a_second_call_with_a_different_mask_unions_on_top() {
        let mut out = [0u64; 1];
        mask_scatter_or_u32(&[0b1u64], &[2u32], &mut out, 8);
        mask_scatter_or_u32(&[0b1u64], &[5u32], &mut out, 8);
        assert_eq!(out[0], (1u64 << 2) | (1u64 << 5), "two calls union, the second does not erase the first");
    }

    #[test]
    #[should_panic(expected = "out_words.len()")]
    fn mask_scatter_or_u32_rejects_short_out_buffer() {
        let src = [0b1u64];
        let index = [200u32];
        let mut out = [0u64; 1]; // out_rows=200 needs 4 words
        mask_scatter_or_u32(&src, &index, &mut out, 200);
    }

    #[test]
    #[should_panic(expected = "src.len()")]
    fn mask_scatter_or_u32_rejects_short_src_buffer() {
        let src = [0u64; 1]; // covers only 64 index rows
        let index = vec![0u32; 65];
        let mut out = [0u64; 1];
        mask_scatter_or_u32(&src, &index, &mut out, 4);
    }

    fn naive_group_sum(mask_bits: &[bool], keys: &[u32], values: &[i32], out_len: usize) -> Vec<i64> {
        let mut out = vec![0i64; out_len];
        for (i, &selected) in mask_bits.iter().enumerate() {
            if selected {
                let k = keys[i] as usize;
                if k < out_len {
                    out[k] = out[k].wrapping_add(values[i] as i64);
                }
            }
        }
        out
    }

    #[test]
    fn masked_group_sum_i32_empty_inputs_write_zero() {
        let mask: [u64; 0] = [];
        let keys: [u32; 0] = [];
        let values: [i32; 0] = [];
        let mut out = [0i64; 4]; // caller zeroes before the first call
        masked_group_sum_i32(&mask, &keys, &values, &mut out);
        assert_eq!(out, [0, 0, 0, 0]);
    }

    #[test]
    fn masked_group_sum_i32_matches_naive_reference_across_the_tail() {
        for &n in &[0usize, 1, 63, 64, 65, 67, 130] {
            let mut seed = 0x9E37_79B9_0000_0001u64;
            let n_groups = 12usize;
            let mask_bits: Vec<bool> = (0..n).map(|_| splitmix(&mut seed) & 1 == 1).collect();
            let mask = bits_to_words(&mask_bits);
            // Every fourth key deliberately out of range.
            let keys: Vec<u32> = (0..n)
                .map(|i| {
                    if i % 4 == 0 {
                        (n_groups as u64 + 3 + i as u64) as u32
                    } else {
                        (splitmix(&mut seed) % n_groups as u64) as u32
                    }
                })
                .collect();
            // Signed values spanning both sides of zero.
            let values: Vec<i32> = (0..n).map(|_| (splitmix(&mut seed) as i32) / 2).collect();
            let want = naive_group_sum(&mask_bits, &keys, &values, n_groups);
            // Caller zeroes once before the first call; one extra slot no
            // key ever hits, to prove it is left untouched, not cleared.
            let mut out = vec![0i64; n_groups + 1];
            masked_group_sum_i32(&mask, &keys, &values, &mut out);
            assert_eq!(&out[..n_groups], &want[..], "group sum mismatch at n={n}");
            assert_eq!(
                out[n_groups], 0,
                "an unreferenced group slot is untouched (stays as the caller left it) at n={n}"
            );
        }
    }

    #[test]
    fn masked_group_sum_i32_handles_negative_values() {
        let mask = [0b111u64];
        let keys = [0u32, 0, 1];
        let values = [-10i32, 5, -3];
        let mut out = [0i64; 2];
        masked_group_sum_i32(&mask, &keys, &values, &mut out);
        assert_eq!(out, [-5, -3]);
    }

    #[test]
    fn masked_group_sum_i32_all_keys_equal_accumulates_into_one_slot() {
        let mask = [0b1111u64];
        let keys = [7u32, 7, 7, 7];
        let values = [1i32, 2, 3, 4];
        let mut out = [0i64; 8];
        masked_group_sum_i32(&mask, &keys, &values, &mut out);
        assert_eq!(out[7], 10);
        assert!(out.iter().enumerate().all(|(i, &v)| i == 7 || v == 0), "no other slot is touched");
    }

    #[test]
    fn masked_group_sum_i32_out_of_range_key_is_dropped_not_a_panic() {
        let mask = [0b111u64];
        let keys = [0u32, 99, 1]; // row 1's key is out of range for a 2-group out
        let values = [10i32, 999, 20];
        let mut out = [0i64; 2];
        masked_group_sum_i32(&mask, &keys, &values, &mut out);
        assert_eq!(out, [10, 20], "the out-of-range key contributes nothing");
    }

    #[test]
    fn masked_group_sum_i32_accumulates_into_a_preloaded_out_buffer() {
        // Slot 1 starts pre-loaded and is never touched by this call; slot 0
        // starts pre-loaded and IS the call's target. This must FAIL if the
        // old whole-buffer zeroing is restored (both would reset to 0 first).
        let mask = [0b1u64]; // row 0 selected
        let keys = [0u32]; // routes to slot 0
        let values = [7i32];
        let mut out = [5i64, 42i64]; // slot 0 preloaded 5, slot 1 preloaded 42
        masked_group_sum_i32(&mask, &keys, &values, &mut out);
        assert_eq!(out[0], 12, "slot 0: preload 5 + contribution 7 = 12");
        assert_eq!(out[1], 42, "slot 1 is untouched, must survive exactly as preloaded");
    }

    #[test]
    fn masked_group_sum_i32_a_second_call_with_a_different_mask_sums_on_top() {
        let mut out = [0i64; 2];
        masked_group_sum_i32(&[0b1u64], &[0u32], &[10i32], &mut out);
        masked_group_sum_i32(&[0b1u64], &[0u32], &[3i32], &mut out);
        assert_eq!(out, [13, 0], "two calls sum, the second does not erase the first's contribution");
    }

    #[test]
    #[should_panic(expected = "keys/values length mismatch")]
    fn masked_group_sum_i32_rejects_mismatched_keys_and_values() {
        let mask = [0b1u64];
        let keys = [0u32, 1];
        let values = [10i32];
        let mut out = [0i64; 2];
        masked_group_sum_i32(&mask, &keys, &values, &mut out);
    }

    #[test]
    #[should_panic(expected = "mask_words.len()")]
    fn masked_group_sum_i32_rejects_short_mask_buffer() {
        let mask = [0u64; 1]; // covers only 64 rows
        let keys = vec![0u32; 65];
        let values = vec![0i32; 65];
        let mut out = [0i64; 1];
        masked_group_sum_i32(&mask, &keys, &values, &mut out);
    }

    // ── masked_group_sum_i32_via ──

    #[test]
    fn masked_group_sum_i32_via_agrees_with_the_plain_form_on_a_precomputed_key_lane() {
        for &n in &[0usize, 1, 63, 64, 65, 67, 130] {
            let mut seed = 0x2222_4444_6666_8888u64;
            let n_partners = 9usize;
            let n_groups = 5usize;
            let mask_bits: Vec<bool> = (0..n).map(|_| splitmix(&mut seed) & 1 == 1).collect();
            let mask = bits_to_words(&mask_bits);
            let index: Vec<u32> = (0..n)
                .map(|_| (splitmix(&mut seed) % n_partners as u64) as u32)
                .collect();
            let remap: Vec<u32> = (0..n_partners)
                .map(|_| (splitmix(&mut seed) % n_groups as u64) as u32)
                .collect();
            let values: Vec<i32> = (0..n).map(|_| (splitmix(&mut seed) as i32) / 2).collect();

            // The naive two-hop key lane, materialised, fed to the plain form.
            let keys: Vec<u32> = index.iter().map(|&fk| remap[fk as usize]).collect();
            let mut want = vec![0i64; n_groups];
            masked_group_sum_i32(&mask, &keys, &values, &mut want);

            let mut got = vec![0i64; n_groups]; // caller zeroes before the first call
            masked_group_sum_i32_via(&mask, &index, &remap, &values, &mut got);
            assert_eq!(got, want, "via mismatched the plain two-hop-materialised form at n={n}");
        }
    }

    #[test]
    fn masked_group_sum_i32_via_drops_at_the_first_hop_when_index_names_no_partner() {
        // Row 1's fk (5) is out of range for a 2-entry remap — dropped before
        // remap is ever consulted.
        let mask = [0b111u64];
        let index = [0u32, 5, 1];
        let remap = [0u32, 1];
        let values = [10i32, 999, 20];
        let mut out = [0i64; 2];
        masked_group_sum_i32_via(&mask, &index, &remap, &values, &mut out);
        assert_eq!(out, [10, 20], "the out-of-range fk contributes nothing");
    }

    #[test]
    fn masked_group_sum_i32_via_drops_at_the_second_hop_when_remap_names_no_group() {
        // Row 1's partner (1) resolves via remap to group 9, out of range for
        // a 2-slot out — dropped after the fk resolves cleanly.
        let mask = [0b111u64];
        let index = [0u32, 1, 0];
        let remap = [0u32, 9]; // partner 1 -> group 9 (out of range)
        let values = [10i32, 999, 20];
        let mut out = [0i64; 2];
        masked_group_sum_i32_via(&mask, &index, &remap, &values, &mut out);
        assert_eq!(out, [30, 0], "the second-hop out-of-range key contributes nothing");
    }

    #[test]
    fn masked_group_sum_i32_via_matches_naive_reference_at_the_67_row_tail() {
        let n = 67usize;
        let mut seed = 0xABCD_EF01_2345_6789u64;
        let n_partners = 6usize;
        let n_groups = 4usize;
        let mask_bits: Vec<bool> = (0..n).map(|_| splitmix(&mut seed) & 1 == 1).collect();
        let mask = bits_to_words(&mask_bits);
        // Every fifth fk deliberately out of range for `remap`.
        let index: Vec<u32> = (0..n)
            .map(|i| {
                if i % 5 == 0 {
                    (n_partners as u64 + 2 + i as u64) as u32
                } else {
                    (splitmix(&mut seed) % n_partners as u64) as u32
                }
            })
            .collect();
        // Every third partner deliberately maps out of range for `out`.
        let remap: Vec<u32> = (0..n_partners)
            .map(|p| {
                if p % 3 == 0 {
                    (n_groups as u64 + 1) as u32
                } else {
                    (splitmix(&mut seed) % n_groups as u64) as u32
                }
            })
            .collect();
        let values: Vec<i32> = (0..n).map(|_| (splitmix(&mut seed) as i32) / 2).collect();

        let mut want = vec![0i64; n_groups];
        for i in 0..n {
            if !mask_bits[i] {
                continue;
            }
            let fk = index[i] as usize;
            if fk >= remap.len() {
                continue;
            }
            let k = remap[fk] as usize;
            if k < n_groups {
                want[k] = want[k].wrapping_add(values[i] as i64);
            }
        }
        let mut got = vec![0i64; n_groups]; // caller zeroes before the first call
        masked_group_sum_i32_via(&mask, &index, &remap, &values, &mut got);
        assert_eq!(got, want);
    }

    #[test]
    fn masked_group_sum_i32_via_accumulates_into_a_preloaded_out_buffer() {
        // Slot 1 is preloaded and never targeted; slot 0 is preloaded and IS
        // the resolved target. Must FAIL if whole-buffer zeroing returns.
        let mask = [0b1u64]; // row 0 selected
        let index = [0u32]; // partner 0
        let remap = [0u32]; // partner 0 -> group 0
        let values = [7i32];
        let mut out = [5i64, 42i64];
        masked_group_sum_i32_via(&mask, &index, &remap, &values, &mut out);
        assert_eq!(out[0], 12, "group 0: preload 5 + contribution 7 = 12");
        assert_eq!(out[1], 42, "group 1 is untouched, must survive exactly as preloaded");
    }

    #[test]
    fn masked_group_sum_i32_via_a_second_call_with_a_different_mask_sums_on_top() {
        let remap = [0u32];
        let mut out = [0i64; 1];
        masked_group_sum_i32_via(&[0b1u64], &[0u32], &remap, &[10i32], &mut out);
        masked_group_sum_i32_via(&[0b1u64], &[0u32], &remap, &[3i32], &mut out);
        assert_eq!(out, [13], "two calls sum, the second does not erase the first's contribution");
    }

    #[test]
    #[should_panic(expected = "index/values length mismatch")]
    fn masked_group_sum_i32_via_rejects_mismatched_index_and_values() {
        let mask = [0b1u64];
        let index = [0u32, 1];
        let remap = [0u32];
        let values = [10i32];
        let mut out = [0i64; 1];
        masked_group_sum_i32_via(&mask, &index, &remap, &values, &mut out);
    }

    #[test]
    #[should_panic(expected = "mask_words.len()")]
    fn masked_group_sum_i32_via_rejects_short_mask_buffer() {
        let mask = [0u64; 1]; // covers only 64 rows
        let index = vec![0u32; 65];
        let remap = vec![0u32; 1];
        let values = vec![0i32; 65];
        let mut out = [0i64; 1];
        masked_group_sum_i32_via(&mask, &index, &remap, &values, &mut out);
    }

    // ── eq_u32_via_to_mask ──

    fn naive_eq_via(fk: &[u32], foreign: &[u32], v: u32) -> Vec<bool> {
        fk.iter()
            .map(|&k| {
                let k = k as usize;
                k < foreign.len() && foreign[k] == v
            })
            .collect()
    }

    #[test]
    fn eq_u32_via_to_mask_matches_naive_reference_across_the_tail() {
        for &n in &[0usize, 1, 63, 64, 65, 130, 1000] {
            let mut seed = 0xACE1_2345_6789_BEEFu64;
            let foreign_len = 17usize;
            let foreign: Vec<u32> = (0..foreign_len)
                .map(|_| (splitmix(&mut seed) % 5) as u32)
                .collect();
            let v = 2u32;
            // A third of keys are deliberately out of range; the rest hit
            // `foreign`, so both the match and no-match arms are genuinely
            // exercised (not merely plausible).
            let fk: Vec<u32> = (0..n)
                .map(|i| {
                    if i % 3 == 0 {
                        (foreign_len as u64 + 3 + i as u64) as u32
                    } else {
                        (splitmix(&mut seed) % foreign_len as u64) as u32
                    }
                })
                .collect();
            let want_bits = naive_eq_via(&fk, &foreign, v);
            let want = bits_to_words(&want_bits);
            let out_words = n.div_ceil(64).max(1);
            let mut out = vec![0xFFFF_FFFF_FFFF_FFFFu64; out_words + 1]; // dirty, over-long
            eq_u32_via_to_mask(&fk, &foreign, v, &mut out);
            assert_eq!(&out[..want.len()], &want[..], "eq_u32_via_to_mask mismatch at n={n}");
            assert_eq!(out[out_words], 0, "surplus word must be cleared at n={n}");
            if n >= 10 {
                // Anti-vacuity (skipped at tiny n, where a single fixture
                // cannot be relied on to hit both arms): this fixture must
                // actually contain both a match and a non-match, or the
                // comparison above proves nothing about which arm is
                // exercised.
                assert!(want_bits.iter().any(|&b| b), "fixture at n={n} has no matching row at all");
                assert!(want_bits.iter().any(|&b| !b), "fixture at n={n} has no non-matching row at all");
            }
        }
    }

    #[test]
    fn eq_u32_via_to_mask_out_of_range_fk_never_matches_even_when_foreign_0_equals_v() {
        let fk = [5u32, 10, 100, u32::MAX];
        let foreign = [7u32]; // foreign[0] == v, but every fk above is >= 1
        let mut out = [0u64; 1];
        eq_u32_via_to_mask(&fk, &foreign, 7, &mut out);
        assert_eq!(out[0], 0, "every fk names no row in `foreign`, so nothing may match");
    }

    #[test]
    fn eq_u32_via_to_mask_tail_and_surplus_words_are_cleared_not_left_dirty() {
        let fk = [0u32, 0, 0, 0, 0]; // n = 5, one word; foreign[0] == v for all
        let foreign = [9u32];
        let mut out = [0xFFFF_FFFF_FFFF_FFFFu64; 3]; // one live word + two surplus
        eq_u32_via_to_mask(&fk, &foreign, 9, &mut out);
        assert_eq!(out[0], 0b11111, "the five live rows should be set");
        assert_eq!(out[0] & !0b11111, 0, "bits past n=5 in the live word must be zero, not dirty");
        assert_eq!(out[1], 0, "surplus word 1 must be cleared");
        assert_eq!(out[2], 0, "surplus word 2 must be cleared");
    }

    #[test]
    fn eq_u32_via_to_mask_empty_foreign_yields_all_zero_mask_for_nonempty_fk() {
        // Every fk names a row, but `foreign` is empty, so every key is out
        // of range: the whole mask must be false, not a panic and not a
        // vacuous "unreachable, so anything goes".
        let fk = [0u32, 1, 2, 3, 4, 5, 6, 7];
        let foreign: [u32; 0] = [];
        let mut out = [0xFFFF_FFFF_FFFF_FFFFu64; 1];
        eq_u32_via_to_mask(&fk, &foreign, 0, &mut out);
        assert_eq!(out[0], 0, "an empty foreign table matches nothing");
    }

    #[test]
    #[should_panic(expected = "out_words.len()")]
    fn eq_u32_via_to_mask_rejects_short_out_buffer() {
        let fk = vec![0u32; 65]; // needs 2 words
        let foreign = [0u32];
        let mut out = [0u64; 1];
        eq_u32_via_to_mask(&fk, &foreign, 0, &mut out);
    }

    // ── 2026-09-13 additions: the closed comparison family, complement/xor/
    //    any/all, care-masked register match, masked min/max, blend ──

    /// Deterministic SplitMix64 so every test corpus is reproducible.
    fn splitmix(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Adversarial i32 corpus: boundaries plus randomness, at lengths that
    /// straddle the 16-lane group and the 64-bit word.
    fn i32_corpus(n: usize, seed: u64) -> Vec<i32> {
        let mut s = seed;
        let edge = [i32::MIN, i32::MIN + 1, -1, 0, 1, 7, i32::MAX - 1, i32::MAX];
        (0..n)
            .map(|i| {
                if i % 5 == 0 {
                    edge[(splitmix(&mut s) % 8) as usize]
                } else {
                    splitmix(&mut s) as i32
                }
            })
            .collect()
    }

    fn scalar_pred_mask(n: usize, pred: impl Fn(usize) -> bool) -> Vec<u64> {
        let mut m = vec![0u64; n.div_ceil(64)];
        for i in 0..n {
            if pred(i) {
                m[i / 64] |= 1u64 << (i % 64);
            }
        }
        m
    }

    const LENS: [usize; 9] = [0, 1, 15, 16, 17, 63, 64, 65, 1000];

    #[test]
    fn ordered_i32_family_matches_scalar_reference_at_boundaries() {
        for &n in &LENS {
            let v = i32_corpus(n, 0xC0FFEE);
            let mut out = vec![u64::MAX; n.div_ceil(64) + 1]; // dirty, over-long
            for &t in &[i32::MIN, i32::MIN + 1, -1, 0, 7, i32::MAX - 1, i32::MAX] {
                lt_i32_to_mask(&v, t, &mut out);
                assert_eq!(&out[..n.div_ceil(64)], &scalar_pred_mask(n, |i| v[i] < t)[..], "lt n={n} t={t}");
                assert_eq!(out[n.div_ceil(64)], 0, "lt surplus word n={n}");
                ge_i32_to_mask(&v, t, &mut out);
                assert_eq!(&out[..n.div_ceil(64)], &scalar_pred_mask(n, |i| v[i] >= t)[..], "ge n={n} t={t}");
                assert_eq!(out[n.div_ceil(64)], 0, "ge surplus word n={n}");
                le_i32_to_mask(&v, t, &mut out);
                assert_eq!(&out[..n.div_ceil(64)], &scalar_pred_mask(n, |i| v[i] <= t)[..], "le n={n} t={t}");
                ne_i32_to_mask(&v, t, &mut out);
                assert_eq!(&out[..n.div_ceil(64)], &scalar_pred_mask(n, |i| v[i] != t)[..], "ne n={n} t={t}");
                eq_i32_to_mask(&v, t, &mut out);
                assert_eq!(&out[..n.div_ceil(64)], &scalar_pred_mask(n, |i| v[i] == t)[..], "eq n={n} t={t}");
            }
        }
    }

    /// The falsifier for "why not `x > t-1`": at `t == i32::MIN` the shifted
    /// form underflows. `lt` must be all-false and `ge` all-true there.
    #[test]
    fn lt_ge_are_exact_at_i32_min() {
        let v = i32_corpus(200, 1);
        let mut out = vec![0u64; 4];
        lt_i32_to_mask(&v, i32::MIN, &mut out);
        assert!(out.iter().all(|&w| w == 0));
        ge_i32_to_mask(&v, i32::MIN, &mut out);
        assert!(mask_all(&out, 200));
        assert_eq!(out[3] >> 8, 0, "tail past 200 must be clear");
    }

    #[test]
    fn ne_u32_is_complement_of_eq_with_clean_tail() {
        for &n in &LENS {
            let v: Vec<u32> = (0..n as u32)
                .map(|i| if i % 3 == 0 { 7 } else { i })
                .collect();
            let mut e = vec![0u64; n.div_ceil(64)];
            let mut ne = vec![u64::MAX; n.div_ceil(64)];
            eq_u32_to_mask(&v, 7, &mut e);
            ne_u32_to_mask(&v, 7, &mut ne);
            for w in 0..e.len() {
                assert_eq!(e[w] & ne[w], 0, "overlap n={n}");
            }
            assert_eq!(
                crate::bitwise::popcount_batch_u64(&e) + crate::bitwise::popcount_batch_u64(&ne),
                n as u64,
                "partition n={n}"
            );
        }
    }

    // ── u8 comparison family (`*_u8_to_mask`) ───────────────────────────

    /// Adversarial u8 corpus: boundary values plus randomness — the
    /// byte-width sibling of [`i32_corpus`] for the family below.
    fn u8_corpus(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        let edge = [0u8, 1, 127, 128, 254, 255];
        (0..n)
            .map(|i| {
                if i % 5 == 0 {
                    edge[(splitmix(&mut s) % 6) as usize]
                } else {
                    splitmix(&mut s) as u8
                }
            })
            .collect()
    }

    const U8_LENS: [usize; 6] = [1, 63, 64, 65, 127, 200];

    /// Randomized agreement for all six `*_u8_to_mask` functions against a
    /// scalar per-element oracle written here (never against the shipped
    /// path's own tail handling, which would be tautological).
    ///
    /// Anti-vacuity: at least one (length, threshold, comparison) case must
    /// select neither nothing nor everything — otherwise a function that
    /// always answered all-zero or all-one for this corpus could pass by
    /// accident.
    #[test]
    fn u8_family_matches_scalar_reference_randomized() {
        let mut saw_a_partial_match = false;
        for &n in &U8_LENS {
            let v = u8_corpus(n, 0xFEED_BEEF);
            let words = n.div_ceil(64);
            for &t in &[0u8, 1, 42, 127, 128, 254, 255] {
                // Dirty, over-long destination on every call: each function
                // must overwrite it (never OR into it) and must clear the
                // one surplus word past `words`.
                let mut out = vec![u64::MAX; words + 1];

                eq_u8_to_mask(&v, t, &mut out);
                let eq = scalar_pred_mask(n, |i| v[i] == t);
                assert_eq!(&out[..words], &eq[..], "eq n={n} t={t}");
                assert_eq!(out[words], 0, "eq surplus word n={n} t={t}");

                ne_u8_to_mask(&v, t, &mut out);
                let ne = scalar_pred_mask(n, |i| v[i] != t);
                assert_eq!(&out[..words], &ne[..], "ne n={n} t={t}");
                assert_eq!(out[words], 0, "ne surplus word n={n} t={t}");

                gt_u8_to_mask(&v, t, &mut out);
                let gt = scalar_pred_mask(n, |i| v[i] > t);
                assert_eq!(&out[..words], &gt[..], "gt n={n} t={t}");
                assert_eq!(out[words], 0, "gt surplus word n={n} t={t}");

                lt_u8_to_mask(&v, t, &mut out);
                let lt = scalar_pred_mask(n, |i| v[i] < t);
                assert_eq!(&out[..words], &lt[..], "lt n={n} t={t}");
                assert_eq!(out[words], 0, "lt surplus word n={n} t={t}");

                ge_u8_to_mask(&v, t, &mut out);
                let ge = scalar_pred_mask(n, |i| v[i] >= t);
                assert_eq!(&out[..words], &ge[..], "ge n={n} t={t}");
                assert_eq!(out[words], 0, "ge surplus word n={n} t={t}");

                le_u8_to_mask(&v, t, &mut out);
                let le = scalar_pred_mask(n, |i| v[i] <= t);
                assert_eq!(&out[..words], &le[..], "le n={n} t={t}");
                assert_eq!(out[words], 0, "le surplus word n={n} t={t}");

                for reference in [&eq, &ne, &gt, &lt, &ge, &le] {
                    let pc = crate::bitwise::popcount_batch_u64(reference);
                    if pc != 0 && pc != n as u64 {
                        saw_a_partial_match = true;
                    }
                }
            }
        }
        assert!(saw_a_partial_match, "anti-vacuity: every (n, t, op) case selected either nothing or everything");
    }

    /// `gt`/`lt` must compare `u8` as UNSIGNED, not as if it were a two's
    /// complement `i8`. A signed compare gets both these boundary pairs
    /// backwards: 0x80 is 128 unsigned but -128 signed, against 0x7F (127
    /// either way).
    #[test]
    fn gt_lt_u8_to_mask_are_unsigned_not_signed() {
        let mut out = [0u64; 1];

        gt_u8_to_mask(&[0x80u8], 0x7F, &mut out);
        assert_eq!(out[0], 0b1, "0x80 > 0x7F must be true unsigned (signed: -128 > 127 is false)");

        gt_u8_to_mask(&[0x7Fu8], 0x80, &mut out);
        assert_eq!(out[0], 0, "0x7F > 0x80 must be false unsigned (signed: 127 > -128 is true)");

        lt_u8_to_mask(&[0x7Fu8], 0x80, &mut out);
        assert_eq!(out[0], 0b1, "0x7F < 0x80 must be true unsigned (signed: 127 < -128 is false)");

        lt_u8_to_mask(&[0x80u8], 0x7F, &mut out);
        assert_eq!(out[0], 0, "0x80 < 0x7F must be false unsigned (signed: -128 < 127 is true)");
    }

    /// For the complement forms (`ne`/`ge`/`le`), every bit at or past
    /// `values.len()` — the tail of the last live word AND every surplus
    /// word — must be clear. This is the case a naive `!mask` gets wrong: it
    /// sets every one of those bits, since bitwise NOT has no notion of
    /// "past the end".
    #[test]
    fn complement_forms_clear_every_trailing_bit() {
        fn assert_tail_clear(out: &[u64], n: usize, label: &str) {
            for i in n..out.len() * 64 {
                assert_eq!((out[i / 64] >> (i % 64)) & 1, 0, "{label}: bit {i} (n={n}) must be clear");
            }
        }

        for &n in &[65usize, 100] {
            let v = u8_corpus(n, 0x1234_5678);
            let words = n.div_ceil(64);
            // Two surplus words, every bit pre-set, so a writer that merely
            // narrows without re-clearing (or that fixes only the last live
            // word) cannot pass by accident.
            let mut out = vec![u64::MAX; words + 2];

            ne_u8_to_mask(&v, 7, &mut out);
            assert_tail_clear(&out, n, "ne");

            ge_u8_to_mask(&v, 7, &mut out);
            assert_tail_clear(&out, n, "ge");

            le_u8_to_mask(&v, 7, &mut out);
            assert_tail_clear(&out, n, "le");
        }
    }

    /// A `u64` corpus that deliberately over-samples the half of the domain a
    /// SIGNED compare gets wrong — values at or above `1 << 63` — plus the
    /// boundary pair either side of it.
    fn u64_corpus(n: usize, seed: u64) -> Vec<u64> {
        let mut s = seed;
        let edge = [
            0u64,
            1,
            0x7FFF_FFFF_FFFF_FFFF, // i64::MAX — the last value a signed compare reads as positive
            0x8000_0000_0000_0000, // the first it reads as negative
            0x8000_0000_0000_0001,
            u64::MAX,
            1 << 32, // above the i32 family's reach: why G2 exists at all
            u64::from(u32::MAX) + 1,
        ];
        (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    edge[(splitmix(&mut s) % edge.len() as u64) as usize]
                } else {
                    // Full 64-bit spread, half of it with the top bit set.
                    splitmix(&mut s)
                }
            })
            .collect()
    }

    /// Lengths that straddle BOTH boundaries this width has: the 8-lane
    /// register and the 64-bit word (8 registers to a word). 7/8/9 exercise a
    /// partial first group; 63/64/65 a partial last group in a full word; 200
    /// several whole words plus a ragged tail.
    const U64_LENS: [usize; 8] = [1, 7, 8, 9, 63, 64, 65, 200];

    /// Randomized agreement for all six `*_u64_to_mask` functions against a
    /// scalar per-element oracle written here — never against the shipped
    /// path's own tail or packing logic, which would be tautological.
    ///
    /// Anti-vacuity: at least one (length, threshold, comparison) case must
    /// select neither nothing nor everything, otherwise a function that always
    /// answered all-zero or all-one for this corpus would pass by accident.
    #[test]
    fn u64_family_matches_scalar_reference_randomized() {
        let mut saw_a_partial_match = false;
        for &n in &U64_LENS {
            let v = u64_corpus(n, 0x0BAD_C0DE_F00D);
            let words = n.div_ceil(64);
            for &t in &[0u64, 1, 0x7FFF_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000, 1 << 32, u64::MAX] {
                // Dirty, over-long destination on every call: each function
                // must fully overwrite it, never OR into it, and must clear
                // the surplus word past `words`.
                let mut out = vec![u64::MAX; words + 1];

                let check = |out: &[u64], pred: &dyn Fn(u64) -> bool, label: &str| {
                    for (i, &x) in v.iter().enumerate() {
                        let got = (out[i / 64] >> (i % 64)) & 1 == 1;
                        assert_eq!(got, pred(x), "{label} n={n} t={t} i={i} x={x:#018x}");
                    }
                    for i in n..out.len() * 64 {
                        assert_eq!((out[i / 64] >> (i % 64)) & 1, 0, "{label} tail bit {i} n={n} t={t}");
                    }
                };

                eq_u64_to_mask(&v, t, &mut out);
                check(&out, &|x| x == t, "eq");
                ne_u64_to_mask(&v, t, &mut out);
                check(&out, &|x| x != t, "ne");
                gt_u64_to_mask(&v, t, &mut out);
                check(&out, &|x| x > t, "gt");
                ge_u64_to_mask(&v, t, &mut out);
                check(&out, &|x| x >= t, "ge");
                lt_u64_to_mask(&v, t, &mut out);
                check(&out, &|x| x < t, "lt");
                le_u64_to_mask(&v, t, &mut out);
                check(&out, &|x| x <= t, "le");

                let hits = v.iter().filter(|&&x| x > t).count();
                if hits > 0 && hits < n {
                    saw_a_partial_match = true;
                }
            }
        }
        assert!(
            saw_a_partial_match,
            "corpus never produced a partial `gt` selection — the agreement above is vacuous"
        );
    }

    /// THE falsifier for this width, and the reason G2 could not be served by
    /// narrowing to the existing `i32` family.
    ///
    /// A `u64` at or above `1 << 63` reads as NEGATIVE under a signed 64-bit
    /// compare, so an implementation that reached for a signed lane op — which
    /// is the ONLY ordered 64-bit compare wasm has, and the only one AVX2 has
    /// at this width — passes every small-value test and inverts the answer on
    /// exactly half the domain. Every arm that lacks an unsigned instruction
    /// has to flip the sign bit of both operands; this is what proves it did.
    #[test]
    fn gt_lt_u64_to_mask_are_unsigned_not_signed() {
        const HI: u64 = 0x8000_0000_0000_0000; // signed: i64::MIN
        const LO: u64 = 0x7FFF_FFFF_FFFF_FFFF; // signed: i64::MAX
        let v = [LO, HI, u64::MAX, 0];
        let mut out = [0u64; 1];

        gt_u64_to_mask(&v, LO, &mut out);
        // Unsigned: HI and u64::MAX exceed LO → bits 1, 2.
        // Signed:   HI is i64::MIN and would NOT exceed i64::MAX → bit 1 lost.
        assert_eq!(
            out[0], 0b0110,
            "{HI:#018x} > {LO:#018x} must be true unsigned (signed: i64::MIN > i64::MAX is false)"
        );

        lt_u64_to_mask(&v, HI, &mut out);
        // Unsigned: only LO and 0 are below HI → bits 0, 3.
        assert_eq!(
            out[0], 0b1001,
            "{LO:#018x} < {HI:#018x} must be true unsigned (signed: i64::MAX < i64::MIN is false)"
        );

        // And the complement pair must agree with their strict siblings.
        ge_u64_to_mask(&v, HI, &mut out);
        assert_eq!(out[0], 0b0110, "ge at the sign boundary");
        le_u64_to_mask(&v, LO, &mut out);
        assert_eq!(out[0], 0b1001, "le at the sign boundary");
    }

    /// The packing falsifier, specific to this width.
    ///
    /// `U64x8` yields 8 bits per chunk and eight chunks share one 64-bit word,
    /// so group `g` must land at byte `g % 8` of word `g / 8`. The `u8` family
    /// has no packing at all (64 lanes == one whole word), so nothing upstream
    /// of here exercises the shift — a wrong `(g % 8) * 8` would survive every
    /// other test in this file whose length happens to be a multiple of 64.
    ///
    /// Built by selecting exactly ONE group at a time and asserting the mask is
    /// that group's byte and nothing else.
    #[test]
    fn u64_packing_places_each_group_in_its_own_byte() {
        const NEEDLE: u64 = 0xDEAD_BEEF_CAFE_F00D;
        for g in 0..8usize {
            // 64 values = 8 groups = exactly one word. Only group `g` matches.
            let mut v = vec![0u64; 64];
            for lane in 0..8 {
                v[g * 8 + lane] = NEEDLE;
            }
            let mut out = [0u64; 1];
            eq_u64_to_mask(&v, NEEDLE, &mut out);
            assert_eq!(out[0], 0xFFu64 << (g * 8), "group {g} must occupy byte {g} alone (got {:#018x})", out[0]);
        }

        // Two words: group 8 is byte 0 of word 1, not byte 0 of word 0.
        let mut v = vec![0u64; 128];
        for lane in 0..8 {
            v[64 + lane] = NEEDLE;
        }
        let mut out = [0u64; 2];
        eq_u64_to_mask(&v, NEEDLE, &mut out);
        assert_eq!(out[0], 0, "word 0 must stay clear when only group 8 matches");
        assert_eq!(out[1], 0xFF, "group 8 is byte 0 of word 1");
    }

    /// The consumer-shaped test: the half-open address window `lo <= x < hi`
    /// that motivated G2 in the first place.
    ///
    /// Every offset here exceeds `2^32`, which is the point — narrowing them
    /// into the existing `i32` family is not a precision trade, it is unsound.
    /// The window is composed from two of this family's masks through
    /// [`mask_and`], which is how a caller actually expresses a range.
    #[test]
    fn an_address_window_over_offsets_above_2_32_needs_the_u64_family() {
        let offsets: Vec<u64> = vec![
            0x0000_0001_0000_0000, // 2^32, below the window
            0x0000_0002_0000_0000, // in
            0x0000_0002_8000_0000, // in
            0x0000_0003_0000_0000, // == hi, EXCLUDED (half-open)
            0x0000_0004_0000_0000, // above
            0xFFFF_FFFF_FFFF_FFFF, // above, and negative under a signed compare
        ];
        let lo: u64 = 0x0000_0002_0000_0000;
        let hi: u64 = 0x0000_0003_0000_0000;

        let mut ge_lo = [0u64; 1];
        let mut lt_hi = [0u64; 1];
        let mut window = [0u64; 1];
        ge_u64_to_mask(&offsets, lo, &mut ge_lo);
        lt_u64_to_mask(&offsets, hi, &mut lt_hi);
        mask_and(&ge_lo, &lt_hi, &mut window);

        assert_eq!(window[0], 0b000110, "half-open window [lo, hi) selects exactly elements 1 and 2");

        // Anti-vacuity on both halves: neither input mask may already be the
        // answer, or the `mask_and` is decorative and the test would pass with
        // one of the two predicates broken.
        assert_ne!(ge_lo[0], window[0], "ge_lo alone must not equal the window");
        assert_ne!(lt_hi[0], window[0], "lt_hi alone must not equal the window");
    }

    #[test]
    fn mask_not_clears_the_tail_and_round_trips() {
        for &n in &LENS {
            let src = scalar_pred_mask(n, |i| i % 3 == 0);
            let mut dst = vec![u64::MAX; n.div_ceil(64)];
            mask_not(&src, n, &mut dst);
            assert_eq!(dst, scalar_pred_mask(n, |i| i % 3 != 0), "not n={n}");
            mask_not_assign(&mut dst, n);
            assert_eq!(dst, src, "double complement n={n}");
        }
        // Can-it-fire: a plain `!` would set the tail; the primitive must not.
        let src = vec![0u64; 2];
        let mut dst = vec![0u64; 2];
        mask_not(&src, 70, &mut dst);
        assert_eq!(dst[0], u64::MAX);
        assert_eq!(dst[1], 0b11_1111);
    }

    #[test]
    fn mask_xor_matches_scalar_and_is_its_own_inverse() {
        for &n in &[0usize, 1, 7, 8, 9, 16, 17, 100] {
            let mut s = 0xABCDu64;
            let a: Vec<u64> = (0..n).map(|_| splitmix(&mut s)).collect();
            let b: Vec<u64> = (0..n).map(|_| splitmix(&mut s)).collect();
            let mut d = vec![0u64; n];
            mask_xor(&a, &b, &mut d);
            let want: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x ^ y).collect();
            assert_eq!(d, want, "xor n={n}");
            mask_xor_assign(&mut d, &b);
            assert_eq!(d, a, "xor_assign inverse n={n}");
        }
    }

    #[test]
    fn mask_any_and_all_discriminate() {
        assert!(!mask_any(&[]));
        assert!(!mask_any(&[0, 0, 0]));
        assert!(mask_any(&[0, 0, 1 << 63]));
        assert!(mask_all(&[], 0));
        assert!(mask_all(&[u64::MAX, 0b111], 67));
        assert!(!mask_all(&[u64::MAX, 0b011], 67));
        assert!(!mask_all(&[u64::MAX - 1, 0b111], 67));
        assert!(mask_all(&[u64::MAX, u64::MAX], 128));
        assert!(!mask_all(&[u64::MAX, u64::MAX >> 1], 128));
    }

    #[test]
    fn ternary_match_u32_u64_match_scalar_and_care_zero_matches_everything() {
        for &n in &LENS {
            let mut s = 0x5EEDu64;
            let v32: Vec<u32> = (0..n)
                .map(|_| (splitmix(&mut s) as u32) & 0xF0F0_00FF)
                .collect();
            let v64: Vec<u64> = (0..n)
                .map(|_| splitmix(&mut s) & 0xFF00_FF00_0000_FFFF)
                .collect();
            let mut out = vec![u64::MAX; n.div_ceil(64) + 1];
            for &(p, c) in
                &[(0x1010_0055u32, 0xF0F0_00FFu32), (0, 0), (0x1010_0055, 0xFFFF_FFFF), (0xDEAD_BEEF, 0x0000_00FF)]
            {
                ternary_match_u32_to_mask(&v32, p, c, &mut out);
                assert_eq!(
                    &out[..n.div_ceil(64)],
                    &scalar_pred_mask(n, |i| (v32[i] ^ p) & c == 0)[..],
                    "u32 n={n} p={p:#x} c={c:#x}"
                );
                assert_eq!(out[n.div_ceil(64)], 0);
                let (p64, c64) = (u64::from(p) << 32 | u64::from(p), u64::from(c) << 16 | u64::from(c));
                ternary_match_u64_to_mask(&v64, p64, c64, &mut out);
                assert_eq!(
                    &out[..n.div_ceil(64)],
                    &scalar_pred_mask(n, |i| (v64[i] ^ p64) & c64 == 0)[..],
                    "u64 n={n}"
                );
            }
            if n > 0 {
                ternary_match_u32_to_mask(&v32, 0xFFFF_FFFF, 0, &mut out);
                assert!(mask_all(&out, n), "care=0 must match everything n={n}");
            }
        }
    }

    /// The register-level match over a real facet-shaped buffer: 16-byte
    /// facets (classid + 12-byte payload) at stride 16 and at stride 512, with
    /// planted hits, a care mask that ignores one byte, and a disable of the
    /// care mask that must flip a planted near-miss (the D-MRL-1a falsifier).
    #[test]
    fn ternary_match_strided_plants_hits_and_care_disable_flips_a_near_miss() {
        for &stride in &[16usize, 512] {
            for &n in &[1usize, 15, 16, 17, 64, 65, 130] {
                let mut bytes = vec![0u8; stride * n + 3];
                let mut s = 0x77u64;
                for i in 0..n {
                    let o = 3 + i * stride + 4; // 3 = deliberate misalignment, +4 = past classid
                    for k in 0..12 {
                        bytes[o + k] = splitmix(&mut s) as u8;
                    }
                }
                let pattern: [u8; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
                let care: [u8; 12] = [0xFF; 12];
                let mut care_wild = care;
                care_wild[5] = 0; // byte 5 is don't-care
                                  // plant an exact hit at row 0 and a near-miss (byte 5 differs) at the last row
                let o0 = 3 + 4;
                bytes[o0..o0 + 12].copy_from_slice(&pattern);
                let ol = 3 + (n - 1) * stride + 4;
                bytes[ol..ol + 12].copy_from_slice(&pattern);
                bytes[ol + 5] ^= 0x80;
                let mut out = vec![u64::MAX; n.div_ceil(64)];
                ternary_match_strided_to_mask(&bytes, 3 + 4, stride, n, &pattern, &care, &mut out);
                let want = scalar_pred_mask(n, |i| {
                    let o = 3 + i * stride + 4;
                    (0..12).all(|k| (bytes[o + k] ^ pattern[k]) & care[k] == 0)
                });
                assert_eq!(out, want, "exact stride={stride} n={n}");
                if n > 1 {
                    // At n == 1 the near-miss row IS row 0, so the planted hit
                    // was deliberately overwritten; only the n > 1 fixtures
                    // carry both.
                    assert!(out[0] & 1 == 1, "planted hit at row 0");
                    assert_eq!(
                        (out[(n - 1) / 64] >> ((n - 1) % 64)) & 1,
                        0,
                        "near-miss must NOT match under full care"
                    );
                }
                ternary_match_strided_to_mask(&bytes, 3 + 4, stride, n, &pattern, &care_wild, &mut out);
                assert_eq!(
                    (out[(n - 1) / 64] >> ((n - 1) % 64)) & 1,
                    1,
                    "near-miss MUST match once byte 5 is don't-care"
                );
                let want_wild = scalar_pred_mask(n, |i| {
                    let o = 3 + i * stride + 4;
                    (0..12).all(|k| (bytes[o + k] ^ pattern[k]) & care_wild[k] == 0)
                });
                assert_eq!(out, want_wild, "wild stride={stride} n={n}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "last element ends at")]
    fn ternary_match_strided_rejects_a_last_element_past_the_buffer() {
        let b = vec![0u8; 16 * 3];
        let mut out = [0u64; 1];
        ternary_match_strided_to_mask(&b, 4, 16, 4, &[0; 12], &[0; 12], &mut out);
    }

    #[test]
    fn masked_min_max_match_scalar_and_ignore_bits_past_len() {
        for &n in &LENS {
            let v = i32_corpus(n, 42);
            let m = scalar_pred_mask(n, |i| i % 7 < 3);
            let mut sel: Vec<i32> = (0..n).filter(|&i| i % 7 < 3).map(|i| v[i]).collect();
            sel.sort_unstable();
            assert_eq!(masked_min_i32(&v, &m), sel.first().copied(), "min n={n}");
            assert_eq!(masked_max_i32(&v, &m), sel.last().copied(), "max n={n}");
            assert_eq!(masked_min_i32(&v, &vec![0u64; n.div_ceil(64)]), None);
        }
        // dirty tail bits past len must be ignored, not read
        let v = [5i32, -3, 9];
        assert_eq!(masked_min_i32(&v, &[u64::MAX]), Some(-3));
        assert_eq!(masked_max_i32(&v, &[u64::MAX]), Some(9));
    }

    #[test]
    fn blend_i32_selects_by_bit_and_ignores_bits_past_len() {
        for &n in &LENS {
            let a = i32_corpus(n, 3);
            let b = i32_corpus(n, 4);
            let m = scalar_pred_mask(n, |i| i % 2 == 0);
            let mut d = vec![0i32; n];
            blend_i32(&m, &a, &b, &mut d);
            for i in 0..n {
                assert_eq!(d[i], if i % 2 == 0 { a[i] } else { b[i] }, "n={n} i={i}");
            }
        }
        let mut d = [0i32; 3];
        blend_i32(&[u64::MAX], &[1, 2, 3], &[9, 9, 9], &mut d);
        assert_eq!(d, [1, 2, 3]);
    }

    /// The two new named immediates, pinned against their formulas.
    #[test]
    fn named_immediates_xor_and_and2_or_match_their_formulas() {
        for a in [0u64, 1] {
            for b in [0u64, 1] {
                for c in [0u64, 1] {
                    let idx = (a << 2) | (b << 1) | c;
                    assert_eq!((crate::simd::ternlog::XOR_AND >> idx) & 1, ((a ^ b) & c) as i32, "XOR_AND {a}{b}{c}");
                    assert_eq!((crate::simd::ternlog::AND2_OR >> idx) & 1, ((a & b) | c) as i32, "AND2_OR {a}{b}{c}");
                }
            }
        }
    }
    // ── D-GTM-1m: mask_shift_morton (§15) ───────────────────────────────────
    //
    // The oracle below (`oracle_neighbour`) is a fresh transcription of
    // `hex_tenant_mq_probe.rs::neighbour`'s dilated-integer add/sub — never a
    // call into `mask_shift_morton` or its `AxisTable`s — generalized from
    // the probe's fixed 16-bit field to whatever `n_cells` the test picks.

    /// Even/odd address-bit split for a field of `n_cells` cells (must be a
    /// power of four): the cell-level analogue of `word_axis_bits`, kept
    /// independent here rather than calling it.
    fn oracle_axis_bits(n_cells: usize) -> (u32, u32) {
        let total_bits = n_cells.trailing_zeros();
        let mut x = 0u32;
        let mut b = 0u32;
        while b < total_bits {
            x |= 1u32 << b;
            b += 2;
        }
        (x, x << 1)
    }

    /// One dilated-integer neighbour step, `(dq, dr)` each in `{-1, 0, 1}` —
    /// zero on an axis leaves it untouched, non-zero on both is a diagonal.
    /// `None` at the field edge (no wrap), exactly `neighbour()`'s contract.
    fn oracle_neighbour(m: u32, dq: i32, dr: i32, x_bits: u32, y_bits: u32) -> Option<u32> {
        let mut x = m & x_bits;
        let mut y = m & y_bits;
        match dq {
            1 => {
                if x == x_bits {
                    return None;
                }
                x = (x | y_bits).wrapping_add(1) & x_bits;
            }
            -1 => {
                if x == 0 {
                    return None;
                }
                x = x.wrapping_sub(1) & x_bits;
            }
            _ => {}
        }
        match dr {
            1 => {
                if y == y_bits {
                    return None;
                }
                y = (y | x_bits).wrapping_add(1) & y_bits;
            }
            -1 => {
                if y == 0 {
                    return None;
                }
                y = y.wrapping_sub(1) & y_bits;
            }
            _ => {}
        }
        Some(x | y)
    }

    fn dir_to_dq_dr(dir: MortonDir) -> (i32, i32) {
        match dir {
            MortonDir::PosQ => (1, 0),
            MortonDir::NegQ => (-1, 0),
            MortonDir::PosR => (0, 1),
            MortonDir::NegR => (0, -1),
        }
    }

    /// Apply one (possibly diagonal) oracle step to every set bit of `src`,
    /// bit-by-bit — the reference `mask_shift_morton` must match.
    fn oracle_shift_field(src: &[u64], dq: i32, dr: i32, n_cells: usize, x_bits: u32, y_bits: u32) -> Vec<u64> {
        let mut dst = vec![0u64; src.len()];
        for i in 0..n_cells {
            if (src[i >> 6] >> (i & 63)) & 1 == 0 {
                continue;
            }
            if let Some(j) = oracle_neighbour(i as u32, dq, dr, x_bits, y_bits) {
                let j = j as usize;
                dst[j >> 6] |= 1u64 << (j & 63);
            }
        }
        dst
    }

    /// F1 — the four `AxisTable`s' masks (3 interior + 1 carry, per
    /// direction) equal an INDEPENDENT runtime fold of the exact §15
    /// predicates, computed here via explicit local-coordinate
    /// classification (never `b & mask == pattern` arithmetic, which is how
    /// the op itself is built) — a literal typo in either implementation
    /// shows up as a mismatch.
    #[test]
    fn morton_axis_tables_match_a_per_bit_predicate_fold() {
        // the local 3-bit sub-coordinate at `bits` (`{0,2,4}` for q, `{1,3,5}`
        // for r), MSB-first from `bits[2]`.
        fn local(b: u32, bits: [u32; 3]) -> u32 {
            (((b >> bits[2]) & 1) << 2) | (((b >> bits[1]) & 1) << 1) | ((b >> bits[0]) & 1)
        }
        fn classify_pos(l: u32) -> usize {
            match l {
                0 | 2 | 4 | 6 => 0, // ..0  -> ..1                (+1 / +2)
                1 | 5 => 1,         // ..01 -> ..10               (+3 / +6)
                3 => 2,             // 011  -> 100                (+11 / +22)
                7 => 3,             // 111  wraps                 (carry)
                _ => unreachable!("a 3-bit local value is in 0..=7"),
            }
        }
        fn classify_neg(l: u32) -> usize {
            match l {
                1 | 3 | 5 | 7 => 0, // ..1  -> ..0                (-1 / -2)
                2 | 6 => 1,         // ..10 -> ..01               (-3 / -6)
                4 => 2,             // 100  -> 011                (-11 / -22)
                0 => 3,             // 000  wraps                 (carry)
                _ => unreachable!("a 3-bit local value is in 0..=7"),
            }
        }
        fn fold(bits: [u32; 3], classify: fn(u32) -> usize) -> [u64; 4] {
            let mut out = [0u64; 4];
            for b in 0u32..64 {
                out[classify(local(b, bits))] |= 1u64 << b;
            }
            out
        }

        let q_bits = [0, 2, 4];
        let r_bits = [1, 3, 5];
        let pos_q = fold(q_bits, classify_pos);
        let neg_q = fold(q_bits, classify_neg);
        let pos_r = fold(r_bits, classify_pos);
        let neg_r = fold(r_bits, classify_neg);

        assert_eq!(POS_Q.interior.map(|(m, _)| m), [pos_q[0], pos_q[1], pos_q[2]], "POS_Q interior masks");
        assert_eq!(POS_Q.carry_mask, pos_q[3], "POS_Q carry mask");
        assert_eq!(NEG_Q.interior.map(|(m, _)| m), [neg_q[0], neg_q[1], neg_q[2]], "NEG_Q interior masks");
        assert_eq!(NEG_Q.carry_mask, neg_q[3], "NEG_Q carry mask");
        assert_eq!(POS_R.interior.map(|(m, _)| m), [pos_r[0], pos_r[1], pos_r[2]], "POS_R interior masks");
        assert_eq!(POS_R.carry_mask, pos_r[3], "POS_R carry mask");
        assert_eq!(NEG_R.interior.map(|(m, _)| m), [neg_r[0], neg_r[1], neg_r[2]], "NEG_R interior masks");
        assert_eq!(NEG_R.carry_mask, neg_r[3], "NEG_R carry mask");

        // anti-vacuity: each direction's four cases partition all 64 bits
        // exactly once — a typo that dropped or doubled a bit shows here
        // even if it happened to leave the individual masks pairwise unequal
        // to something else that was also wrong.
        for masks in [pos_q, neg_q, pos_r, neg_r] {
            assert_eq!(masks[0] | masks[1] | masks[2] | masks[3], u64::MAX, "cases must cover every bit");
            let total: u32 = masks.iter().map(|m| m.count_ones()).sum();
            assert_eq!(total, 64, "cases must be pairwise disjoint");
        }

        // the shift amounts and carry directions, against §15's literal table.
        assert_eq!(POS_Q.interior.map(|(_, s)| s), [1, 3, 11]);
        assert_eq!((POS_Q.carry_shift, POS_Q.carry_decrements), (21, false));
        assert_eq!(NEG_Q.interior.map(|(_, s)| s), [-1, -3, -11]);
        assert_eq!((NEG_Q.carry_shift, NEG_Q.carry_decrements), (21, true));
        assert_eq!(POS_R.interior.map(|(_, s)| s), [2, 6, 22]);
        assert_eq!((POS_R.carry_shift, POS_R.carry_decrements), (42, false));
        assert_eq!(NEG_R.interior.map(|(_, s)| s), [-2, -6, -22]);
        assert_eq!((NEG_R.carry_shift, NEG_R.carry_decrements), (42, true));
    }

    /// F2 — `mask_shift_morton` matches `oracle_shift_field` bit-for-bit, over
    /// random masks, at two field sizes (64 words = 64×64 cells, 1024 words =
    /// 256×256 cells) and all four axis directions.
    #[test]
    fn morton_shift_matches_the_per_bit_oracle_on_random_fields() {
        let mut seed = 0x1357_2468_1357_2468u64;
        // 1 and 4 words run pass 1's padded-tail branch (the whole field is
        // shorter than one register); 64 and 1024 run the chunk body.
        for &n_words in &[1usize, 4, 64, 1024] {
            let n_cells = n_words * 64;
            let (x_bits, y_bits) = oracle_axis_bits(n_cells);
            for _ in 0..8 {
                let src: Vec<u64> = (0..n_words).map(|_| splitmix64(&mut seed)).collect();
                for dir in [MortonDir::PosQ, MortonDir::NegQ, MortonDir::PosR, MortonDir::NegR] {
                    let mut got = vec![0u64; n_words];
                    mask_shift_morton(&src, dir, &mut got);
                    let (dq, dr) = dir_to_dq_dr(dir);
                    let want = oracle_shift_field(&src, dq, dr, n_cells, x_bits, y_bits);
                    assert_eq!(got, want, "n_words={n_words} dir={dir:?}");
                }
            }
        }
    }

    /// F6 — the documented OR-accumulate contract: `dst` is never cleared,
    /// so a pre-filled `dst` keeps its prior bits and several directions
    /// chain into one plane. FAILS IF pass 1 overwrites instead of
    /// accumulating (`acc = splat(0)` instead of `from_array(*d)`) — every
    /// other morton test starts from a zeroed `dst` and cannot see that
    /// (falsifier audit on #307).
    #[test]
    fn morton_shift_or_accumulates_into_a_prefilled_dst() {
        let mut seed = 0x0F1E_2D3C_4B5A_6978u64;
        for &n_words in &[4usize, 64, 1024] {
            let n_cells = n_words * 64;
            let (x_bits, y_bits) = oracle_axis_bits(n_cells);
            let src: Vec<u64> = (0..n_words).map(|_| splitmix64(&mut seed)).collect();
            let prior: Vec<u64> = (0..n_words).map(|_| splitmix64(&mut seed)).collect();
            // prior bits survive alongside the shifted field
            let mut got = prior.clone();
            mask_shift_morton(&src, MortonDir::PosQ, &mut got);
            let want_q = oracle_shift_field(&src, 1, 0, n_cells, x_bits, y_bits);
            let expect: Vec<u64> = prior.iter().zip(&want_q).map(|(p, w)| p | w).collect();
            assert_eq!(got, expect, "n_words={n_words}: prior bits lost or shift missing");
            // anti-vacuity: the prior actually contributed bits the shift did not
            assert!(prior.iter().zip(&want_q).any(|(p, w)| p & !w != 0), "prior added nothing — fixture too weak");
            // two directions chained into one plane
            let mut chained = vec![0u64; n_words];
            mask_shift_morton(&src, MortonDir::PosQ, &mut chained);
            mask_shift_morton(&src, MortonDir::NegR, &mut chained);
            let want_r = oracle_shift_field(&src, 0, -1, n_cells, x_bits, y_bits);
            let expect2: Vec<u64> = want_q.iter().zip(&want_r).map(|(a, b)| a | b).collect();
            assert_eq!(chained, expect2, "n_words={n_words}: chaining two directions lost one");
        }
    }

    /// F3 — can-fire: a lone interior bit moves to exactly one bit, at the
    /// oracle's address, for every direction. Can-stay-silent: a NON-trivial
    /// far-edge mask (every cell on `dir`'s edge, spread across many words —
    /// not an empty mask) produces an all-zero `dst`.
    #[test]
    fn morton_shift_can_fire_and_can_stay_silent() {
        let n_words = 64usize;
        let n_cells = n_words * 64;
        let (x_bits, y_bits) = oracle_axis_bits(n_cells);

        let mut seed = 0xABCD_EF01_2345_6789u64;
        for dir in [MortonDir::PosQ, MortonDir::NegQ, MortonDir::PosR, MortonDir::NegR] {
            let (dq, dr) = dir_to_dq_dr(dir);
            for _ in 0..16 {
                // an interior cell: strictly inside both axes, so it is never
                // the far edge for any direction under test.
                let i = loop {
                    let x = (splitmix64(&mut seed) as u32) & x_bits;
                    let y = (splitmix64(&mut seed) as u32) & y_bits;
                    if x != 0 && x != x_bits && y != 0 && y != y_bits {
                        break (x | y) as usize;
                    }
                };
                let mut src = vec![0u64; n_words];
                src[i >> 6] |= 1u64 << (i & 63);
                let mut dst = vec![0u64; n_words];
                mask_shift_morton(&src, dir, &mut dst);
                let fired: Vec<usize> = (0..n_cells)
                    .filter(|&j| (dst[j >> 6] >> (j & 63)) & 1 == 1)
                    .collect();
                assert_eq!(fired.len(), 1, "exactly one bit must move, dir={dir:?} i={i}");
                let want = oracle_shift_field(&src, dq, dr, n_cells, x_bits, y_bits);
                assert_eq!(dst, want, "the moved bit must land at the oracle's address, dir={dir:?}");
            }
        }

        for dir in [MortonDir::PosQ, MortonDir::NegQ, MortonDir::PosR, MortonDir::NegR] {
            let mut src = vec![0u64; n_words];
            let mut planted = 0usize;
            for i in 0..n_cells {
                let m = i as u32;
                let (x, y) = (m & x_bits, m & y_bits);
                let at_edge = match dir {
                    MortonDir::PosQ => x == x_bits,
                    MortonDir::NegQ => x == 0,
                    MortonDir::PosR => y == y_bits,
                    MortonDir::NegR => y == 0,
                };
                if at_edge {
                    src[i >> 6] |= 1u64 << (i & 63);
                    planted += 1;
                }
            }
            // non-trivial and spread: strictly more than "a handful", and
            // touching more than one word (never a single-word fixture that
            // could pass by accident of layout).
            assert!(planted >= 16, "far-edge fixture must be non-trivial, dir={dir:?} planted={planted}");
            let touched_words = src.iter().filter(|&&w| w != 0).count();
            assert!(touched_words > 1, "far-edge fixture must spread across words, dir={dir:?}");

            let mut dst = vec![0u64; n_words];
            mask_shift_morton(&src, dir, &mut dst);
            assert!(dst.iter().all(|&w| w == 0), "far-edge bits must produce silence, dir={dir:?}");
        }
    }

    /// F4 — composition: `+q` then `-q` is the identity on interior cells,
    /// and a hex diagonal via two calls matches the oracle's one-step
    /// diagonal, for both `+q-r` and its mirror `-q+r`.
    #[test]
    fn morton_shift_composes_to_identity_and_matches_the_hex_diagonal() {
        let n_words = 64usize;
        let n_cells = n_words * 64;
        let (x_bits, y_bits) = oracle_axis_bits(n_cells);
        let mut seed = 0x0F0F_1E1E_2D2D_3C3Cu64;

        // +q then -q == identity, restricted to cells that cannot fall off
        // the +q edge (so the round trip is well posed for every one).
        let raw: Vec<u64> = (0..n_words).map(|_| splitmix64(&mut seed)).collect();
        let mut interior = raw.clone();
        for i in 0..n_cells {
            if (i as u32) & x_bits == x_bits {
                interior[i >> 6] &= !(1u64 << (i & 63));
            }
        }
        assert!(interior.iter().any(|&w| w != 0), "the interior fixture must be non-empty");
        let mut forward = vec![0u64; n_words];
        mask_shift_morton(&interior, MortonDir::PosQ, &mut forward);
        let mut back = vec![0u64; n_words];
        mask_shift_morton(&forward, MortonDir::NegQ, &mut back);
        assert_eq!(back, interior, "+q then -q must be the identity on interior cells");

        // the hex diagonal: two composed calls must match the oracle's own
        // single-step diagonal (`dq` and `dr` both non-zero at once).
        let diag_src: Vec<u64> = (0..n_words).map(|_| splitmix64(&mut seed)).collect();

        let mut scratch = vec![0u64; n_words];
        mask_shift_morton(&diag_src, MortonDir::PosQ, &mut scratch);
        let mut dst = vec![0u64; n_words];
        mask_shift_morton(&scratch, MortonDir::NegR, &mut dst);
        let want = oracle_shift_field(&diag_src, 1, -1, n_cells, x_bits, y_bits);
        assert_eq!(dst, want, "+q-r composed via two calls must match the oracle diagonal");

        let mut scratch2 = vec![0u64; n_words];
        mask_shift_morton(&diag_src, MortonDir::NegQ, &mut scratch2);
        let mut dst2 = vec![0u64; n_words];
        mask_shift_morton(&scratch2, MortonDir::PosR, &mut dst2);
        let want2 = oracle_shift_field(&diag_src, -1, 1, n_cells, x_bits, y_bits);
        assert_eq!(dst2, want2, "-q+r composed via two calls must match the oracle diagonal");
    }

    // ── Gated predicates: `*_to_mask_under` (survivor-word skip) ────────────

    /// Every `*_to_mask_under` member must equal `ungated_sibling(values) &
    /// gate`, word for word, including when the gate itself carries dirty
    /// bits past `values.len()` in its last live word — the gate's own
    /// phantom bits must never leak into the answer, and the ungated
    /// reference's own tail law is what proves it (its bits past `n` are
    /// already zero, so `full & gate` conforms regardless of the gate).
    #[test]
    fn under_family_equals_ungated_and_gate() {
        const UNDER_LENS: [usize; 10] = [0, 1, 15, 16, 17, 63, 64, 65, 130, 1000];
        let i32_thresholds = [i32::MIN, i32::MIN + 1, -1, 0, 7, i32::MAX - 1, i32::MAX];
        let u32_needles = [0u32, 1, 3, 6, 42, u32::MAX];
        let ternary_cases: [(u32, u32); 4] = [(0b0101, 0xF), (0b0101, 0b1010), (0, 0), (0xF, 0xF)];

        for &n in &UNDER_LENS {
            let words = mask_words_for(n);
            let v_i32 = i32_corpus(n, 0x1357_9BDF_ACE1_1234u64.wrapping_add(n as u64));

            let mut s_eq = 0xF00D_BEEF_1234_5678u64.wrapping_add(n as u64);
            let v_u32_eq: Vec<u32> = (0..n).map(|_| (splitmix(&mut s_eq) % 7) as u32).collect();

            let mut s_tern32 = 0x9E37_79B9_ABCD_EF01u64.wrapping_add(n as u64);
            let v_u32_tern: Vec<u32> = (0..n)
                .map(|_| (splitmix(&mut s_tern32) % 16) as u32)
                .collect();

            let mut s_tern64 = 0x1122_3344_5566_7788u64.wrapping_add(n as u64);
            let v_u64_tern: Vec<u64> = (0..n).map(|_| splitmix(&mut s_tern64) % 16).collect();

            // ~50% zero words, random bits elsewhere, with the last live
            // word's phantom bits (past `n`) also forced dirty at random —
            // the engine must clean those independently of the ungated
            // reference, which already zeroes them by its own contract.
            let mut s_gate = 0xDEAD_C0DE_F00D_BAADu64.wrapping_add(n as u64);
            let mut gate = vec![0u64; words];
            for w in gate.iter_mut() {
                *w = if splitmix(&mut s_gate).is_multiple_of(2) {
                    0
                } else {
                    splitmix(&mut s_gate)
                };
            }
            if words > 0 && !n.is_multiple_of(64) {
                let last = words - 1;
                let live = n - last * 64;
                let phantom = !((1u64 << live) - 1);
                gate[last] |= phantom & splitmix(&mut s_gate);
            }

            // gt / lt / ge / le / ne_i32 / eq_i32, over signed thresholds.
            for &t in &i32_thresholds {
                let mut full = vec![0u64; words];
                let mut gated = vec![u64::MAX; words + 1];

                gt_i32_to_mask(&v_i32, t, &mut full);
                gt_i32_to_mask_under(&v_i32, t, &gate, &mut gated);
                for w in 0..words {
                    assert_eq!(gated[w], full[w] & gate[w], "gt_i32 n={n} t={t} w={w}");
                }
                assert_eq!(gated[words], 0, "gt_i32 n={n} t={t} surplus word");

                lt_i32_to_mask(&v_i32, t, &mut full);
                lt_i32_to_mask_under(&v_i32, t, &gate, &mut gated);
                for w in 0..words {
                    assert_eq!(gated[w], full[w] & gate[w], "lt_i32 n={n} t={t} w={w}");
                }
                assert_eq!(gated[words], 0, "lt_i32 n={n} t={t} surplus word");

                ge_i32_to_mask(&v_i32, t, &mut full);
                ge_i32_to_mask_under(&v_i32, t, &gate, &mut gated);
                for w in 0..words {
                    assert_eq!(gated[w], full[w] & gate[w], "ge_i32 n={n} t={t} w={w}");
                }
                assert_eq!(gated[words], 0, "ge_i32 n={n} t={t} surplus word");

                le_i32_to_mask(&v_i32, t, &mut full);
                le_i32_to_mask_under(&v_i32, t, &gate, &mut gated);
                for w in 0..words {
                    assert_eq!(gated[w], full[w] & gate[w], "le_i32 n={n} t={t} w={w}");
                }
                assert_eq!(gated[words], 0, "le_i32 n={n} t={t} surplus word");

                ne_i32_to_mask(&v_i32, t, &mut full);
                ne_i32_to_mask_under(&v_i32, t, &gate, &mut gated);
                for w in 0..words {
                    assert_eq!(gated[w], full[w] & gate[w], "ne_i32 n={n} t={t} w={w}");
                }
                assert_eq!(gated[words], 0, "ne_i32 n={n} t={t} surplus word");

                eq_i32_to_mask(&v_i32, t, &mut full);
                eq_i32_to_mask_under(&v_i32, t, &gate, &mut gated);
                for w in 0..words {
                    assert_eq!(gated[w], full[w] & gate[w], "eq_i32 n={n} t={t} w={w}");
                }
                assert_eq!(gated[words], 0, "eq_i32 n={n} t={t} surplus word");
            }

            // eq_u32 / ne_u32, over unsigned needles.
            for &needle in &u32_needles {
                let mut full = vec![0u64; words];
                let mut gated = vec![u64::MAX; words + 1];

                eq_u32_to_mask(&v_u32_eq, needle, &mut full);
                eq_u32_to_mask_under(&v_u32_eq, needle, &gate, &mut gated);
                for w in 0..words {
                    assert_eq!(gated[w], full[w] & gate[w], "eq_u32 n={n} needle={needle} w={w}");
                }
                assert_eq!(gated[words], 0, "eq_u32 n={n} needle={needle} surplus word");

                ne_u32_to_mask(&v_u32_eq, needle, &mut full);
                ne_u32_to_mask_under(&v_u32_eq, needle, &gate, &mut gated);
                for w in 0..words {
                    assert_eq!(gated[w], full[w] & gate[w], "ne_u32 n={n} needle={needle} w={w}");
                }
                assert_eq!(gated[words], 0, "ne_u32 n={n} needle={needle} surplus word");
            }

            // ternary_match_u32 / ternary_match_u64, over pattern/care cases.
            for &(pattern, care) in &ternary_cases {
                let mut full = vec![0u64; words];
                let mut gated = vec![u64::MAX; words + 1];

                ternary_match_u32_to_mask(&v_u32_tern, pattern, care, &mut full);
                ternary_match_u32_to_mask_under(&v_u32_tern, pattern, care, &gate, &mut gated);
                for w in 0..words {
                    assert_eq!(
                        gated[w],
                        full[w] & gate[w],
                        "ternary_u32 n={n} pattern={pattern:#x} care={care:#x} w={w}"
                    );
                }
                assert_eq!(gated[words], 0, "ternary_u32 n={n} pattern={pattern:#x} care={care:#x} surplus word");

                let pattern64 = pattern as u64;
                let care64 = care as u64;
                ternary_match_u64_to_mask(&v_u64_tern, pattern64, care64, &mut full);
                ternary_match_u64_to_mask_under(&v_u64_tern, pattern64, care64, &gate, &mut gated);
                for w in 0..words {
                    assert_eq!(
                        gated[w],
                        full[w] & gate[w],
                        "ternary_u64 n={n} pattern={pattern64:#x} care={care64:#x} w={w}"
                    );
                }
                assert_eq!(gated[words], 0, "ternary_u64 n={n} pattern={pattern64:#x} care={care64:#x} surplus word");
            }
        }
    }

    /// Can-it-fire test for the survivor-word skip: an empty gate word must
    /// never reach the predicate. Gate alternates `u64::MAX` / `0` over 10
    /// words (5 live), so the predicate closure — which counts its own
    /// invocations — must fire exactly `5 * 4` times (five live words, four
    /// 16-lane groups per word); an all-ones gate must fire on every group.
    #[test]
    fn under_skips_compares_on_empty_gate_words() {
        let n = 640; // exactly 10 words of 64 elements
        let values = vec![0i32; n];
        let words = mask_words_for(n);
        assert_eq!(words, 10);

        let counter = std::cell::Cell::new(0usize);
        let gate: Vec<u64> = (0..words)
            .map(|w| if w % 2 == 0 { u64::MAX } else { 0 })
            .collect();
        let mut out = vec![0u64; words];
        pack_under::<i32, 16>("under_skips_compares_on_empty_gate_words", &values, &gate, &mut out, |_lanes, _live| {
            counter.set(counter.get() + 1);
            0
        });
        assert_eq!(counter.get(), 5 * 4, "five live words x four 16-lane groups");

        counter.set(0);
        let all_ones = vec![u64::MAX; words];
        pack_under::<i32, 16>(
            "under_skips_compares_on_empty_gate_words",
            &values,
            &all_ones,
            &mut out,
            |_lanes, _live| {
                counter.set(counter.get() + 1);
                0
            },
        );
        assert_eq!(counter.get(), 40, "an all-ones gate evaluates every group");
    }

    /// Every public `*_to_mask_under` member is ONE `pack_under` delegation,
    /// so the skip falsifier on the private engine covers all ten. Counted
    /// from this file's own source: ten call sites before the test module.
    /// FAILS IF a member hand-rolls its loop (baton audit P2 on #307).
    #[test]
    fn every_under_member_delegates_to_the_one_engine() {
        let src = include_str!("simd_masking_ops.rs");
        let production = src
            .split("#[cfg(test)]")
            .next()
            .expect("the file has a test module");
        // Count definitions, not mentions: the doc comments name the members
        // many more times than the source defines them.
        let members = production
            .lines()
            .filter(|l| l.trim_start().starts_with("pub fn ") && l.contains("_to_mask_under"))
            .count();
        let delegations = production.matches("pack_under::<").count();
        assert_eq!(members, 10, "ten public `*_to_mask_under` members");
        assert_eq!(delegations, 10, "one pack_under call per member");
    }

    /// A dirty phantom bit in the gate's last live word (bits at or past
    /// `n % 64`, which correspond to no real element) must never survive
    /// into the answer — element count 70 leaves word 1 with only 6 live
    /// bits, and the gate sets every bit of that word including the 58
    /// phantom ones.
    #[test]
    fn under_phantom_gate_bits_do_not_leak() {
        let n = 70;
        let values = vec![0i32; n]; // every value satisfies `> i32::MIN`
        let gate = vec![u64::MAX, u64::MAX]; // word 1's bits 6..64 are phantom
        let mut out = vec![0u64; 2];
        gt_i32_to_mask_under(&values, i32::MIN, &gate, &mut out);
        assert_eq!(out[0], u64::MAX, "word 0 is fully live and fully satisfies `> i32::MIN`");
        assert_eq!(out[1], 0b11_1111, "only the 6 live bits of word 1 may be set");
    }

    /// `under.len()` one word short of `values.len().div_ceil(64)` must
    /// panic, mirroring every other `*_to_mask` length guard in this module.
    #[test]
    #[should_panic(expected = "under.len()")]
    fn under_rejects_short_gate() {
        let values = vec![0i32; 65]; // needs 2 gate words
        let gate = vec![0u64; 1]; // one short
        let mut out = vec![0u64; 2];
        gt_i32_to_mask_under(&values, 0, &gate, &mut out);
    }

    // ── mask_set_range: the range WRITE (N1,
    //    `.claude/plans/gemm-ternlog-mask-consolidation-v1.md` §16.6) ────────

    /// Independent bit-serial reference: bit `b` set iff `lo <= b < hi`.
    /// Never calls `mask_set_range` or `word_range_mask`.
    fn reference_set_range(words: usize, lo: usize, hi: usize) -> Vec<u64> {
        let mut v = vec![0u64; words];
        for b in lo..hi {
            v[b / 64] |= 1u64 << (b % 64);
        }
        v
    }

    #[test]
    fn mask_set_range_matches_scalar_reference_randomized() {
        let mut seed = 0x00B0_0000_005E_7A11_u64;
        for &words in &[1usize, 2, 3, 4, 7, 8, 16] {
            let capacity = words * 64;
            for _ in 0..40 {
                let lo = (splitmix64(&mut seed) as usize) % (capacity + 1);
                let span = (splitmix64(&mut seed) as usize) % (capacity + 1 - lo);
                let hi = lo + span;
                let mut got = vec![u64::MAX; words]; // pre-dirtied: an OR-er fails immediately
                mask_set_range(&mut got, lo, hi);
                assert_eq!(got, reference_set_range(words, lo, hi), "words={words} lo={lo} hi={hi}");
            }
        }
        // Anti-vacuity: a genuinely non-trivial range sets EXACTLY `hi - lo`
        // bits, not "some" bits — catches an always-set-everything or
        // always-set-nothing implementation that could otherwise still pass
        // the boundary and randomized cases above by accident of which
        // random `(lo, hi)` pairs happened to be drawn.
        let mut got = vec![0u64; 4];
        mask_set_range(&mut got, 7, 250);
        let popcount: u32 = got.iter().map(|w| w.count_ones()).sum();
        assert_eq!(popcount, 250 - 7, "popcount must equal hi - lo exactly");
    }

    /// Every boundary shape named in the primitive's own doc comment, each
    /// checked explicitly rather than only through the randomized fuzz above.
    #[test]
    fn mask_set_range_boundary_shapes() {
        // lo == hi: the empty range. All-zero output — NOT a no-op, the
        // destination is still fully overwritten (pre-dirtied to catch a
        // "lo == hi means skip the call" shortcut).
        let mut got = vec![u64::MAX; 2];
        mask_set_range(&mut got, 0, 0);
        assert_eq!(got, vec![0, 0], "lo == hi == 0 must zero everything");
        let mut got = vec![u64::MAX; 2];
        mask_set_range(&mut got, 128, 128);
        assert_eq!(got, vec![0, 0], "lo == hi == capacity must zero everything");

        // lo == 0: the range starts at the very first bit.
        let mut got = vec![u64::MAX; 2];
        mask_set_range(&mut got, 0, 70);
        assert_eq!(got, reference_set_range(2, 0, 70), "lo == 0");

        // hi == words * 64: the range runs all the way to the last bit.
        let mut got = vec![u64::MAX; 2];
        mask_set_range(&mut got, 40, 128);
        assert_eq!(got, reference_set_range(2, 40, 128), "hi == capacity");

        // Both endpoints inside ONE word: the single-word branch, one write.
        let mut got = vec![u64::MAX; 2];
        mask_set_range(&mut got, 2, 5);
        assert_eq!(got[0], 0b0001_1100, "both endpoints in one word: bits 2,3,4");
        assert_eq!(got[1], 0, "the untouched second word must be cleared too");

        // Endpoints in ADJACENT words, neither word-aligned.
        let mut got = vec![u64::MAX; 2];
        mask_set_range(&mut got, 60, 70);
        assert_eq!(got[0], 0xF000_0000_0000_0000, "adjacent words: low word's tail (bits 60..64)");
        assert_eq!(got[1], 0b0011_1111, "adjacent words: high word's head (bits 0..6)");

        // Both endpoints word-ALIGNED (`lo % 64 == 0 && hi % 64 == 0`): the
        // range is exactly one interior word, reached via the single-word
        // branch (lo_word == hi_word here, not the interior-word loop).
        let mut got = vec![u64::MAX; 3];
        mask_set_range(&mut got, 64, 128);
        assert_eq!(got, vec![0, u64::MAX, 0], "word-aligned range is exactly one full word");

        // A range of exactly 1 bit.
        let mut got = vec![u64::MAX; 1];
        mask_set_range(&mut got, 5, 6);
        assert_eq!(got[0], 1u64 << 5, "a single-bit range sets exactly that bit");
    }

    /// The falsifier for "full overwrite, not OR": every word starts
    /// all-ones, and afterwards a bit survives iff it is inside `[lo, hi)` —
    /// checked bit-by-bit across the whole buffer, including the surplus
    /// bits of the partially-live last word, which an OR-based "fix" would
    /// leave set.
    #[test]
    fn mask_set_range_overwrites_a_dirty_destination_rather_than_oring() {
        let mut got = vec![u64::MAX; 3];
        mask_set_range(&mut got, 70, 130);
        for b in 0..192 {
            let want = (70..130).contains(&b);
            let bit = (got[b / 64] >> (b % 64)) & 1 == 1;
            assert_eq!(bit, want, "bit {b}");
        }
    }

    #[test]
    #[should_panic(expected = "mask_set_range: lo=")]
    fn mask_set_range_rejects_lo_greater_than_hi() {
        let mut out = vec![0u64; 1];
        mask_set_range(&mut out, 5, 3);
    }

    #[test]
    #[should_panic(expected = "mask_set_range: out_words.len()=")]
    fn mask_set_range_rejects_hi_past_capacity() {
        let mut out = vec![0u64; 1]; // capacity 64
        mask_set_range(&mut out, 0, 65);
    }
}

#[cfg(test)]
mod key_run_tests {
    use super::{mask_scatter_or_u32, masked_key_run_count_u32, KeyRunCarry};

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn pack(bits: &[bool]) -> Vec<u64> {
        let mut w = vec![0u64; bits.len().div_ceil(64).max(1)];
        for (i, &b) in bits.iter().enumerate() {
            if b {
                w[i / 64] |= 1 << (i % 64);
            }
        }
        w
    }

    /// The independent answer: a set of keys, which is exactly the
    /// population-sized state the run fold exists to avoid.
    fn distinct_selected(keys: &[u32], sel: &[bool]) -> usize {
        let mut seen = std::collections::BTreeSet::new();
        for (k, &s) in keys.iter().zip(sel) {
            if s {
                seen.insert(*k);
            }
        }
        seen.len()
    }

    #[test]
    fn on_an_ordered_lane_the_run_fold_equals_the_distinct_count_across_any_tiling() {
        let mut seed = 0x51u64;
        for &n in &[1usize, 63, 64, 65, 200, 1000] {
            // Ordered: sorted keys with repeats.
            let mut keys: Vec<u32> = (0..n).map(|_| (splitmix(&mut seed) % 37) as u32).collect();
            keys.sort_unstable();
            let sel: Vec<bool> = (0..n)
                .map(|_| splitmix(&mut seed).is_multiple_of(3))
                .collect();
            let want = distinct_selected(&keys, &sel);
            assert!(want > 0 || n < 3, "fixture must select something");
            for &tile in &[1usize, 7, 64, 100, n] {
                let mut carry = KeyRunCarry::default();
                let mut got = 0;
                let mut start = 0;
                while start < n {
                    let end = (start + tile).min(n);
                    got += masked_key_run_count_u32(&keys[start..end], &pack(&sel[start..end]), &mut carry)
                        .expect("a sorted lane is never refused");
                    start = end;
                }
                got += carry.finish();
                assert_eq!(got, want, "n={n} tile={tile}");
                assert_eq!(carry, KeyRunCarry::default(), "finish resets the carry");
            }
        }
    }

    #[test]
    fn a_run_with_no_selected_element_is_not_counted_and_a_split_run_counts_once() {
        // 2 2 2 | 4 | 9 9 — key 4 never selected; key 2 selected in the
        // first and third element (the run is entered by two tiles).
        let keys = [2u32, 2, 2, 4, 9, 9];
        let sel = [true, false, true, false, false, true];
        let m = pack(&sel);
        let mut c = KeyRunCarry::default();
        let a = masked_key_run_count_u32(&keys[..2], &m, &mut c);
        let b = masked_key_run_count_u32(&keys[2..], &[m[0] >> 2], &mut c);
        assert_eq!(a, Some(0), "the run of 2s is still open after two elements");
        assert_eq!(b, Some(1), "closing the 2-run counts it once; the 4-run closes unhit");
        assert_eq!(c.finish(), 1, "the trailing 9-run was hit");
    }

    #[test]
    fn a_lane_out_of_key_order_is_refused_not_over_counted() {
        // 1 2 1: three runs but two keys. The fold must not answer 3.
        let keys = [1u32, 2, 1];
        let m = pack(&[true; 3]);
        let mut c = KeyRunCarry::default();
        assert_eq!(masked_key_run_count_u32(&keys, &m, &mut c), None);
        // The refusal is also visible across a tile edge: the descent lands
        // in the second call, whose carry still holds the larger key.
        let mut c = KeyRunCarry::default();
        assert_eq!(masked_key_run_count_u32(&keys[..2], &m, &mut c), Some(1));
        assert_eq!(masked_key_run_count_u32(&keys[2..], &[m[0] >> 2], &mut c), None);
        // The seen-set sink is what an unordered lane needs for an exact answer.
        let mut sink = [0u64; 1];
        mask_scatter_or_u32(&m, &keys, &mut sink, 3);
        assert_eq!(sink[0].count_ones(), 2, "two distinct keys");
    }

    #[test]
    fn the_order_check_inspects_unselected_keys_too() {
        // keys 1 2 1, selected 1 0 1: the selected subsequence reads 1,1 but
        // the lane has two runs of 1 — a fold that only looked at survivors
        // would answer 2 instead of refusing.
        let keys = [1u32, 2, 1];
        let mut c = KeyRunCarry::default();
        assert_eq!(masked_key_run_count_u32(&keys, &pack(&[true, false, true]), &mut c), None);
    }

    #[test]
    fn contiguous_but_unsorted_is_refused_deliberately() {
        // 3 3 1 1: every key contiguous, so a run count WOULD be exact —
        // but the O(1)-checkable certificate is ORDER, and 1 < 3 breaks it.
        let keys = [3u32, 3, 1, 1];
        let mut c = KeyRunCarry::default();
        assert_eq!(masked_key_run_count_u32(&keys, &pack(&[true; 4]), &mut c), None);
    }

    #[test]
    #[should_panic(expected = "mask_words.len()=0 < required 1")]
    fn short_mask_panics() {
        let mut c = KeyRunCarry::default();
        masked_key_run_count_u32(&[1u32, 2], &[], &mut c);
    }
}
