//! The ONE masking parity program of the SIMD realization matrix.
//!
//! [`run`] exercises the shipped `ndarray::simd` masking facade — every
//! predicate→mask, the mask algebra, all 256 `ternlog` tables on both mask
//! lane widths, the care-masked register matches (contiguous and strided),
//! the tail rules at 0 / 1 / 63 / 64 / 65 / 130 rows, the masked reductions,
//! and `blend` — against references computed here (bit-serial for `ternlog`,
//! scalar word-serial for the rest; none of them touches an `ndarray::simd`
//! type), and returns `0` iff every result is bit-identical. It has NO idea which backend
//! `simd.rs` selected: no `cfg(target_feature)`, no backend module, no
//! intrinsic. The build (its `-Ctarget-cpu`, its target, its
//! `nightly-simd` feature) chooses the realization; this program only asks
//! whether that realization agrees with the definition.
//!
//! Return codes are distinct per assertion so a CI failure names the exact
//! op and shape: `0x1IM` / `0x2IM` = ternlog table `IM` on `U32x16` /
//! `U64x8`, `0x3xx` `U64x8` algebra, `0x4xx` `I32x16` compares, `0x5xx`
//! predicate→mask, `0x6xx` mask algebra, `0x7xx` care-match, `0x8xx` masked
//! reductions and blend, `0x9xx` `mask_shift_morton` (the Morton hex
//! neighbour shift, D-GTM-1m), `0xAxx` the gated `*_to_mask_under`
//! predicates (mask-risc `Pred { under }`, D-MRX-0), `0xBxx` `mask_set_range`
//! (the range WRITE, N1). `main.rs` (native / qemu) and `selfcheck()`
//! (the wasm cdylib export, driven by `run.mjs`) both call [`run`].

use ndarray::simd::{
    blend_i32, eq_i32_to_mask, eq_i32_to_mask_under, eq_u32_strided_to_mask, eq_u32_to_mask, eq_u32_to_mask_under,
    ge_i32_to_mask, ge_i32_to_mask_under, gt_i32_to_mask, gt_i32_to_mask_under, le_i32_to_mask, le_i32_to_mask_under,
    lt_i32_to_mask, lt_i32_to_mask_under, mask_all, mask_and, mask_and_assign, mask_andnot, mask_andnot_assign,
    mask_any, mask_not, mask_not_assign, mask_or, mask_or_assign, mask_set_range, mask_shift_morton, mask_ternlog,
    mask_ternlog_assign, mask_xor, mask_xor_assign, masked_max_i32, masked_min_i32, masked_strided_group_sum,
    masked_sum_i32, ne_i32_to_mask, ne_i32_to_mask_under, ne_u32_to_mask, ne_u32_to_mask_under,
    ternary_match_strided_to_mask, ternary_match_u32_to_mask, ternary_match_u32_to_mask_under,
    ternary_match_u64_to_mask, ternary_match_u64_to_mask_under, ternlog, I32x16, MortonDir, U32x16, U64x8,
};

/// Number of check groups [`run`] executes (for the log line only).
pub const CHECKS: usize = 11;

/// The wasm export: identical to [`run`], `extern "C"` so `run.mjs` can call it.
#[no_mangle]
pub extern "C" fn selfcheck() -> u32 {
    run()
}

/// Run every group; `0` on success, else the code of the first failing assertion.
pub fn run() -> u32 {
    let groups: [fn() -> Result<(), u32>; CHECKS] = [
        check_ternlog_all_tables, check_u64x8_algebra, check_i32x16_compare, check_predicates_to_mask,
        check_mask_algebra, check_care_match, check_masked_reductions, check_blend, check_morton_shift,
        check_predicates_under, check_set_range,
    ];
    for g in groups {
        if let Err(code) = g() {
            return code;
        }
    }
    0
}

/// The row lengths every slice-level check runs at: the empty mask, a single
/// row, one word minus one, exactly one word, one word plus one, and a
/// three-word mask with a partial tail.
const LENS: [usize; 6] = [0, 1, 63, 64, 65, 130];

/// Deterministic SplitMix64 so every realization sees the same operands.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn words_for(n: usize) -> usize {
    n.div_ceil(64)
}

/// Bit-serial ternlog reference: bit `i` of the result is `imm >> (a_i<<2 | b_i<<1 | c_i)`.
fn reference_ternlog_u64(imm: u32, a: u64, b: u64, c: u64) -> u64 {
    let mut r = 0u64;
    for bit in 0..64 {
        let idx = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1);
        r |= (((imm >> idx) & 1) as u64) << bit;
    }
    r
}

/// The reference mask writer every predicate is checked against: bit `i` of
/// word `i / 64` iff `pred(i)`, every other bit (tail and surplus words) zero.
fn reference_mask(n: usize, out_len: usize, pred: impl Fn(usize) -> bool) -> Vec<u64> {
    let mut w = vec![0u64; out_len];
    for i in 0..n {
        if pred(i) {
            w[i / 64] |= 1u64 << (i % 64);
        }
    }
    w
}

/// Mixed-sign `i32` operands with the extremes and equal pairs at every length.
fn i32_values(n: usize, rng: &mut SplitMix64) -> Vec<i32> {
    let fixed = [i32::MIN, i32::MAX, 0, -1, 1, 7, 7, -7, 100, -100];
    (0..n)
        .map(|i| if i < fixed.len() { fixed[i] } else { rng.next() as i32 })
        .collect()
}

fn u32_values(n: usize, rng: &mut SplitMix64) -> Vec<u32> {
    let fixed = [0, u32::MAX, 1, 0x8000_0000, 7, 7, 0xDEAD_BEEF];
    (0..n)
        .map(|i| if i < fixed.len() { fixed[i] } else { rng.next() as u32 })
        .collect()
}

// ── 0x1xx / 0x2xx: ternlog, all 256 tables, both lane widths ────────────────

fn check_ternlog_all_tables() -> Result<(), u32> {
    let u32_triples: [([u32; 16], [u32; 16], [u32; 16]); 2] = [
        ([0xF0F0_F0F0; 16], [0xCCCC_CCCC; 16], [0xAAAA_AAAA; 16]),
        (
            [
                0x0000_0000, 0xFFFF_FFFF, 0x0000_0001, 0x8000_0000, 0x1234_5678, 0x9ABC_DEF0, 0xDEAD_BEEF, 0xCAFE_BABE,
                0x0F0F_0F0F, 0xF0F0_F0F0, 0x5555_5555, 0xAAAA_AAAA, 0x0000_00FF, 0xFF00_0000, 0x0101_0101, 0x8080_8080,
            ],
            [
                0x9E37_79B9, 0x1111_1111, 0xDEAD_C0DE, 0x0BAD_F00D, 0x7FFF_FFFF, 0x0000_0000, 0xFFFF_FFFF, 0x1357_9BDF,
                0x2468_ACE0, 0xFEDC_BA98, 0x0000_0010, 0x0000_001F, 0xABCD_EF01, 0x1020_4080, 0x0F0F_F0F0, 0xC0DE_CAFE,
            ],
            [
                0x1357_9BDF, 0xC0DE_CAFE, 0x0000_0000, 0xFFFF_FFFF, 0xA5A5_A5A5, 0x5A5A_5A5A, 0x0000_8000, 0x8000_0001,
                0x7777_7777, 0x8888_8888, 0x0F0F_0F0F, 0xF0F0_F0F0, 0x1234_5678, 0x8765_4321, 0xFFFF_0000, 0x0000_FFFF,
            ],
        ),
    ];
    let u64_triples: [([u64; 8], [u64; 8], [u64; 8]); 2] = [
        ([0xF0F0_F0F0_F0F0_F0F0; 8], [0xCCCC_CCCC_CCCC_CCCC; 8], [0xAAAA_AAAA_AAAA_AAAA; 8]),
        (
            [
                0,
                u64::MAX,
                1,
                1 << 63,
                0x1234_5678_9ABC_DEF0,
                0xDEAD_BEEF_CAFE_BABE,
                0x0F0F_0F0F_F0F0_F0F0,
                0x5555_AAAA_5555_AAAA,
            ],
            [
                0x9E37_79B9_7F4A_7C15,
                0x1111_1111_1111_1111,
                0xDEAD_C0DE_0BAD_F00D,
                u64::MAX,
                0,
                0x1357_9BDF_2468_ACE0,
                0xFEDC_BA98_7654_3210,
                0xC0DE_CAFE_C0DE_CAFE,
            ],
            [
                0x1357_9BDF_1357_9BDF,
                0,
                u64::MAX,
                0xA5A5_A5A5_5A5A_5A5A,
                0x8000_0000_0000_0001,
                0x7777_7777_8888_8888,
                0xFFFF_0000_0000_FFFF,
                0x0123_4567_89AB_CDEF,
            ],
        ),
    ];

    macro_rules! check_imm {
        ($imm:expr) => {{
            const IMM: i32 = $imm;
            for (a, b, c) in &u32_triples {
                let got = U32x16::from_array(*a)
                    .ternlog::<IMM>(U32x16::from_array(*b), U32x16::from_array(*c))
                    .to_array();
                for i in 0..16 {
                    let want = reference_ternlog_u64(IMM as u32, a[i] as u64, b[i] as u64, c[i] as u64) as u32;
                    if got[i] != want {
                        return Err(0x100 | IMM as u32);
                    }
                }
            }
            for (a, b, c) in &u64_triples {
                let got = U64x8::from_array(*a)
                    .ternlog::<IMM>(U64x8::from_array(*b), U64x8::from_array(*c))
                    .to_array();
                for i in 0..8 {
                    if got[i] != reference_ternlog_u64(IMM as u32, a[i], b[i], c[i]) {
                        return Err(0x200 | IMM as u32);
                    }
                }
            }
        }};
    }
    macro_rules! check_row {
        ($hi:expr) => {
            check_imm!($hi * 16 + 0);
            check_imm!($hi * 16 + 1);
            check_imm!($hi * 16 + 2);
            check_imm!($hi * 16 + 3);
            check_imm!($hi * 16 + 4);
            check_imm!($hi * 16 + 5);
            check_imm!($hi * 16 + 6);
            check_imm!($hi * 16 + 7);
            check_imm!($hi * 16 + 8);
            check_imm!($hi * 16 + 9);
            check_imm!($hi * 16 + 10);
            check_imm!($hi * 16 + 11);
            check_imm!($hi * 16 + 12);
            check_imm!($hi * 16 + 13);
            check_imm!($hi * 16 + 14);
            check_imm!($hi * 16 + 15);
        };
    }
    check_row!(0);
    check_row!(1);
    check_row!(2);
    check_row!(3);
    check_row!(4);
    check_row!(5);
    check_row!(6);
    check_row!(7);
    check_row!(8);
    check_row!(9);
    check_row!(10);
    check_row!(11);
    check_row!(12);
    check_row!(13);
    check_row!(14);
    check_row!(15);
    Ok(())
}

// ── 0x3xx: U64x8 algebra (the operators the bulk mask ops ride) ─────────────

fn check_u64x8_algebra() -> Result<(), u32> {
    let a_arr: [u64; 8] = [
        0,
        u64::MAX,
        1 << 63,
        0xF0F0_F0F0_F0F0_F0F0,
        0x1234_5678_9ABC_DEF0,
        1,
        0xDEAD_BEEF_CAFE_BABE,
        0x5555_AAAA_5555_AAAA,
    ];
    let b_arr: [u64; 8] = [
        u64::MAX,
        0,
        1 << 63,
        0xCCCC_CCCC_CCCC_CCCC,
        0x0FED_CBA9_8765_4321,
        1,
        0xC0DE_CAFE_C0DE_CAFE,
        0xAAAA_5555_AAAA_5555,
    ];
    let (a, b) = (U64x8::from_array(a_arr), U64x8::from_array(b_arr));
    if a.to_array() != a_arr {
        return Err(0x300);
    }
    let mut back = [0u64; 8];
    U64x8::from_slice(&a_arr).copy_to_slice(&mut back);
    if back != a_arr {
        return Err(0x301);
    }
    let (and, or, xor, not, andnot) =
        ((a & b).to_array(), (a | b).to_array(), (a ^ b).to_array(), (!a).to_array(), a.andnot(b).to_array());
    let pop = a.popcnt().to_array();
    let mut hamming = 0u64;
    for i in 0..8 {
        if and[i] != a_arr[i] & b_arr[i] {
            return Err(0x302);
        }
        if or[i] != a_arr[i] | b_arr[i] {
            return Err(0x303);
        }
        if xor[i] != a_arr[i] ^ b_arr[i] {
            return Err(0x304);
        }
        if not[i] != !a_arr[i] {
            return Err(0x305);
        }
        if andnot[i] != a_arr[i] & !b_arr[i] {
            return Err(0x306);
        }
        if pop[i] != a_arr[i].count_ones() as u64 {
            return Err(0x307);
        }
        hamming += (a_arr[i] ^ b_arr[i]).count_ones() as u64;
    }
    if a.xor_popcount(b) != hamming {
        return Err(0x308);
    }
    for &n in &[0u32, 1, 7, 31, 32, 63] {
        let (l, r) = (a.rotate_left(n).to_array(), a.rotate_right(n).to_array());
        for i in 0..8 {
            if l[i] != a_arr[i].rotate_left(n) {
                return Err(0x309);
            }
            if r[i] != a_arr[i].rotate_right(n) {
                return Err(0x30A);
            }
        }
    }
    let mut sum = 0u64;
    for &x in &a_arr {
        sum = sum.wrapping_add(x);
    }
    if a.reduce_sum() != sum {
        return Err(0x30B);
    }
    if (a + b).to_array() != core::array::from_fn(|i| a_arr[i].wrapping_add(b_arr[i])) {
        return Err(0x30C);
    }
    if (a - b).to_array() != core::array::from_fn(|i| a_arr[i].wrapping_sub(b_arr[i])) {
        return Err(0x30D);
    }
    if !(a == U64x8::from_array(a_arr)) || a == b {
        return Err(0x30E);
    }
    Ok(())
}

// ── 0x4xx: I32x16 compares and reductions (the ordered-predicate primitive) ─

fn check_i32x16_compare() -> Result<(), u32> {
    let a_arr: [i32; 16] =
        [i32::MIN, i32::MAX, 0, -1, 1, -2, 7, 7, 100, -100, 0x7FFF_0000, -0x7FFF_0000, 42, -42, 2, -3];
    let b_arr: [i32; 16] =
        [i32::MIN, i32::MAX, 0, 0, -1, -2, 7, 8, -100, 100, -0x7FFF_0000, 0x7FFF_0000, 41, -41, -3, 2];
    let (a, b) = (I32x16::from_array(a_arr), I32x16::from_array(b_arr));
    if a.to_array() != a_arr {
        return Err(0x400);
    }
    let gt = a.gt_bitmask(b);
    let ge0 = a.cmpge_zero_mask();
    for i in 0..16 {
        if ((gt >> i) & 1 == 1) != (a_arr[i] > b_arr[i]) {
            return Err(0x401);
        }
        if ((ge0 >> i) & 1 == 1) != (a_arr[i] >= 0) {
            return Err(0x402);
        }
    }
    if a.simd_min(b).to_array() != core::array::from_fn(|i| a_arr[i].min(b_arr[i])) {
        return Err(0x403);
    }
    if a.simd_max(b).to_array() != core::array::from_fn(|i| a_arr[i].max(b_arr[i])) {
        return Err(0x404);
    }
    if a.reduce_min() != i32::MIN || a.reduce_max() != i32::MAX {
        return Err(0x405);
    }
    // Extremes in EVERY lane position, so a reduction tree that drops a lane
    // (the 16→8→4→2→1 ladders) is caught wherever the drop happens.
    for lane in 0..16 {
        let mut arr = [0i32; 16];
        arr[lane] = i32::MIN;
        arr[(lane + 5) % 16] = i32::MAX;
        let v = I32x16::from_array(arr);
        if v.reduce_min() != i32::MIN || v.reduce_max() != i32::MAX {
            return Err(0x406);
        }
    }
    let mut sum = 0i32;
    for &x in &a_arr {
        sum = sum.wrapping_add(x);
    }
    if a.reduce_sum() != sum {
        return Err(0x407);
    }
    if !(a == I32x16::from_array(a_arr)) || a == b {
        return Err(0x408);
    }
    Ok(())
}

// ── 0x5xx: predicate → mask, every tail shape, full-overwrite + zero tail ────

fn check_predicates_to_mask() -> Result<(), u32> {
    let mut rng = SplitMix64(0x5EED_0000_0001);
    for &n in &LENS {
        // One surplus word beyond ceil(n/64) so the "surplus words are written
        // zero" half of the contract is observable, pre-filled with garbage so
        // a writer that ORs instead of overwriting is caught.
        let out_len = words_for(n) + 1;
        let vals = i32_values(n, &mut rng);
        let uvals = u32_values(n, &mut rng);
        let thresholds = [i32::MIN, -1, 0, 1, 7, i32::MAX];
        let mut out = vec![u64::MAX; out_len];
        for (k, &t) in thresholds.iter().enumerate() {
            let k = k as u32;
            macro_rules! pred {
                ($f:ident, $op:tt, $code:expr) => {{
                    out.iter_mut().for_each(|w| *w = u64::MAX);
                    $f(&vals, t, &mut out);
                    if out != reference_mask(n, out_len, |i| vals[i] $op t) {
                        return Err($code | k);
                    }
                }};
            }
            pred!(eq_i32_to_mask, ==, 0x500);
            pred!(ne_i32_to_mask, !=, 0x510);
            pred!(lt_i32_to_mask, <, 0x520);
            pred!(le_i32_to_mask, <=, 0x530);
            pred!(gt_i32_to_mask, >, 0x540);
            pred!(ge_i32_to_mask, >=, 0x550);
        }
        for (k, &needle) in [0u32, 7, u32::MAX, 0x8000_0000].iter().enumerate() {
            let k = k as u32;
            out.iter_mut().for_each(|w| *w = u64::MAX);
            eq_u32_to_mask(&uvals, needle, &mut out);
            if out != reference_mask(n, out_len, |i| uvals[i] == needle) {
                return Err(0x560 | k);
            }
            out.iter_mut().for_each(|w| *w = u64::MAX);
            ne_u32_to_mask(&uvals, needle, &mut out);
            if out != reference_mask(n, out_len, |i| uvals[i] != needle) {
                return Err(0x570 | k);
            }
        }
        // Strided: a u32 at byte 4 of every 16-byte record.
        let mut bytes = vec![0u8; n * 16 + 3];
        for i in 0..n {
            bytes[4 + i * 16..8 + i * 16].copy_from_slice(&uvals[i].to_le_bytes());
        }
        out.iter_mut().for_each(|w| *w = u64::MAX);
        eq_u32_strided_to_mask(&bytes, 4, 16, n, 7, &mut out);
        if out != reference_mask(n, out_len, |i| uvals[i] == 7) {
            return Err(0x580);
        }
    }
    Ok(())
}

// ── 0xAxx: the gated predicates — `out = pred & under`, survivor-word skip ──

/// Every `*_to_mask_under` against `reference_mask(n, out_len, |i| pred(i) &&
/// gate_bit(i))`. The gate is built with every other word forced to ZERO (so
/// the skip path is exercised on every length > 64 — 65 and 130 here) and
/// random bits elsewhere INCLUDING, at lengths whose last word index is even
/// (1, 63, 130), that word's bits past `n` — the reference reads the gate BIT
/// for row `i`, so a phantom past `n` never enters it, and the primitive must
/// agree. One surplus out word pre-filled `u64::MAX` catches an OR-er.
fn check_predicates_under() -> Result<(), u32> {
    let mut rng = SplitMix64(0x5EED_0000_00A0);
    for &n in &LENS {
        let out_len = words_for(n) + 1;
        let vals = i32_values(n, &mut rng);
        let uvals = u32_values(n, &mut rng);
        let vals64: Vec<u64> = (0..n).map(|_| rng.next() % 16).collect();
        let gate: Vec<u64> = (0..words_for(n))
            .map(|w| if w % 2 == 1 { 0 } else { rng.next() | 1 })
            .collect();
        let gate_bit = |i: usize| (gate[i / 64] >> (i % 64)) & 1 == 1;
        let mut out = vec![u64::MAX; out_len];
        let thresholds = [i32::MIN, -1, 0, 7, i32::MAX];
        for (k, &t) in thresholds.iter().enumerate() {
            let k = k as u32;
            macro_rules! pred {
                ($f:ident, $op:tt, $code:expr) => {{
                    out.iter_mut().for_each(|w| *w = u64::MAX);
                    $f(&vals, t, &gate, &mut out);
                    if out != reference_mask(n, out_len, |i| vals[i] $op t && gate_bit(i)) {
                        return Err($code | k);
                    }
                }};
            }
            pred!(gt_i32_to_mask_under, >, 0xA00);
            pred!(lt_i32_to_mask_under, <, 0xA10);
            pred!(ge_i32_to_mask_under, >=, 0xA20);
            pred!(le_i32_to_mask_under, <=, 0xA30);
            pred!(ne_i32_to_mask_under, !=, 0xA40);
            pred!(eq_i32_to_mask_under, ==, 0xA50);
        }
        for (k, &needle) in [0u32, 7, u32::MAX, 0x8000_0000].iter().enumerate() {
            let k = k as u32;
            out.iter_mut().for_each(|w| *w = u64::MAX);
            eq_u32_to_mask_under(&uvals, needle, &gate, &mut out);
            if out != reference_mask(n, out_len, |i| uvals[i] == needle && gate_bit(i)) {
                return Err(0xA60 | k);
            }
            out.iter_mut().for_each(|w| *w = u64::MAX);
            ne_u32_to_mask_under(&uvals, needle, &gate, &mut out);
            if out != reference_mask(n, out_len, |i| uvals[i] != needle && gate_bit(i)) {
                return Err(0xA70 | k);
            }
        }
        for (k, &(pattern, care)) in [(0b1010u32, 0b1011u32), (7, u32::MAX), (0, 0)]
            .iter()
            .enumerate()
        {
            let k = k as u32;
            out.iter_mut().for_each(|w| *w = u64::MAX);
            ternary_match_u32_to_mask_under(&uvals, pattern, care, &gate, &mut out);
            if out != reference_mask(n, out_len, |i| (uvals[i] ^ pattern) & care == 0 && gate_bit(i)) {
                return Err(0xA80 | k);
            }
            let (p64, c64) = (u64::from(pattern), u64::from(care));
            out.iter_mut().for_each(|w| *w = u64::MAX);
            ternary_match_u64_to_mask_under(&vals64, p64, c64, &gate, &mut out);
            if out != reference_mask(n, out_len, |i| (vals64[i] ^ p64) & c64 == 0 && gate_bit(i)) {
                return Err(0xA90 | k);
            }
        }
    }
    Ok(())
}

// ── 0x6xx: mask algebra, in-place forms, ternlog over slices, any / all ─────

/// Word counts for the mask-algebra group. `LENS` are ROW counts, and at 130
/// rows a mask is only 3 words — below one `U64x8` chunk — so iterating `LENS`
/// here would run the padded tail of every word op and never its `as_chunks`
/// body (savant-architect, PR #306). These counts straddle the 8-word chunk
/// boundary and cover every tail length 1..=7.
const WORD_LENS: [usize; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31];

/// Row counts for the mask-algebra group: every full-word count in
/// `WORD_LENS`, plus, for each non-zero word count, three partially-live
/// final words (63, 32 and 1 live bits). The full-word counts keep the
/// `as_chunks` bodies covered; the partial ones are the ONLY inputs that
/// reach the tail branches of `mask_not` / `mask_not_assign` / `mask_all` and
/// the tail-only `mask_any` check below — with `n` always `nw * 64`, that
/// branch was dead and a backend mishandling tail bits would still have
/// passed (CodeRabbit on PR #306).
fn row_lens() -> Vec<usize> {
    let mut v = Vec::with_capacity(WORD_LENS.len() * 4);
    for &nw in &WORD_LENS {
        v.push(nw * 64);
        if nw > 0 {
            v.push(nw * 64 - 1);
            v.push(nw * 64 - 32);
            v.push(nw * 64 - 63);
        }
    }
    v
}

fn check_mask_algebra() -> Result<(), u32> {
    let mut rng = SplitMix64(0x6000_0000_0002);
    for n in row_lens() {
        let nw = words_for(n);
        // Conforming inputs: random bits with the tail beyond `n` cleared.
        let tail_mask = |i: usize| -> u64 {
            let live = n.saturating_sub(i * 64).min(64);
            if live == 64 {
                u64::MAX
            } else {
                (1u64 << live) - 1
            }
        };
        let a: Vec<u64> = (0..nw).map(|i| rng.next() & tail_mask(i)).collect();
        let b: Vec<u64> = (0..nw).map(|i| rng.next() & tail_mask(i)).collect();
        let c: Vec<u64> = (0..nw).map(|i| rng.next() & tail_mask(i)).collect();
        let mut dst = vec![u64::MAX; nw];

        mask_and(&a, &b, &mut dst);
        if dst.iter().zip(&a).zip(&b).any(|((&d, &x), &y)| d != x & y) {
            return Err(0x600);
        }
        mask_or(&a, &b, &mut dst);
        if dst.iter().zip(&a).zip(&b).any(|((&d, &x), &y)| d != x | y) {
            return Err(0x601);
        }
        mask_xor(&a, &b, &mut dst);
        if dst.iter().zip(&a).zip(&b).any(|((&d, &x), &y)| d != x ^ y) {
            return Err(0x602);
        }
        mask_andnot(&a, &b, &mut dst);
        if dst.iter().zip(&a).zip(&b).any(|((&d, &x), &y)| d != x & !y) {
            return Err(0x603);
        }
        mask_not(&a, n, &mut dst);
        if dst
            .iter()
            .enumerate()
            .any(|(i, &d)| d != !a[i] & tail_mask(i))
        {
            return Err(0x604);
        }
        // In-place forms agree with the out-of-place ones.
        let mut t = a.clone();
        mask_and_assign(&mut t, &b);
        if t.iter().zip(&a).zip(&b).any(|((&d, &x), &y)| d != x & y) {
            return Err(0x605);
        }
        let mut t = a.clone();
        mask_or_assign(&mut t, &b);
        if t.iter().zip(&a).zip(&b).any(|((&d, &x), &y)| d != x | y) {
            return Err(0x606);
        }
        let mut t = a.clone();
        mask_xor_assign(&mut t, &b);
        if t.iter().zip(&a).zip(&b).any(|((&d, &x), &y)| d != x ^ y) {
            return Err(0x607);
        }
        let mut t = a.clone();
        mask_andnot_assign(&mut t, &b);
        if t.iter().zip(&a).zip(&b).any(|((&d, &x), &y)| d != x & !y) {
            return Err(0x608);
        }
        let mut t = a.clone();
        mask_not_assign(&mut t, n);
        if t.iter()
            .enumerate()
            .any(|(i, &d)| d != !a[i] & tail_mask(i))
        {
            return Err(0x609);
        }
        // Slice-level ternlog at the named immediates plus the two constant
        // tables and the odd-parity one.
        macro_rules! slice_ternlog {
            ($imm:expr, $code:expr) => {{
                const IMM: i32 = $imm;
                mask_ternlog::<IMM>(&a, &b, &c, &mut dst);
                for i in 0..nw {
                    if dst[i] != reference_ternlog_u64(IMM as u32, a[i], b[i], c[i]) {
                        return Err($code);
                    }
                }
                let mut t = a.clone();
                mask_ternlog_assign::<IMM>(&mut t, &b, &c);
                if t != dst {
                    return Err($code | 0x8);
                }
            }};
        }
        slice_ternlog!(ternlog::AND2_OR, 0x610);
        slice_ternlog!(ternlog::XOR_AND, 0x611);
        slice_ternlog!(ternlog::MAJ3, 0x612);
        slice_ternlog!(ternlog::AND3, 0x613);
        slice_ternlog!(ternlog::OR3, 0x614);
        slice_ternlog!(0x00, 0x615);
        slice_ternlog!(0xFF, 0x616);
        slice_ternlog!(0x96, 0x617);

        // any / all on the tail shapes: all-zero, all-live-set, one bit
        // missing (the LAST live row), and a set bit ONLY in the tail (a
        // non-conforming mask, which `any` must still see and `all` ignore).
        let zero = vec![0u64; nw];
        let full: Vec<u64> = (0..nw).map(tail_mask).collect();
        if mask_any(&zero) {
            return Err(0x620);
        }
        if mask_any(&full) != (n > 0) {
            return Err(0x621);
        }
        if !mask_all(&zero, 0) || !mask_all(&full, n) {
            return Err(0x622);
        }
        if n > 0 {
            if mask_all(&zero, n) {
                return Err(0x623);
            }
            let mut missing = full.clone();
            missing[(n - 1) / 64] &= !(1u64 << ((n - 1) % 64));
            if mask_all(&missing, n) || (n > 1 && !mask_any(&missing)) {
                return Err(0x624);
            }
            if n % 64 != 0 {
                let mut tail_only = zero.clone();
                tail_only[nw - 1] = 1u64 << (n % 64);
                if !mask_any(&tail_only) || mask_all(&tail_only, n) {
                    return Err(0x625);
                }
            }
        }
    }
    Ok(())
}

// ── 0x7xx: care-masked matches — contiguous u32 / u64 and the strided 12-byte register ──

fn check_care_match() -> Result<(), u32> {
    let mut rng = SplitMix64(0x7000_0000_0003);
    let cases_u32: [(u32, u32); 5] = [
        (0, 0),
        (7, u32::MAX),
        (0xDEAD_BEEF, 0xFFFF_0000),
        (0x8000_0000, 0x8000_0000),
        (0x1234_5678, 0x0F0F_0F0F),
    ];
    for &n in &LENS {
        let out_len = words_for(n) + 1;
        let vals = u32_values(n, &mut rng);
        let vals64: Vec<u64> = (0..n)
            .map(|i| if i % 3 == 0 { vals[i] as u64 } else { rng.next() })
            .collect();
        let mut out = vec![u64::MAX; out_len];
        for (k, &(pattern, care)) in cases_u32.iter().enumerate() {
            out.iter_mut().for_each(|w| *w = u64::MAX);
            ternary_match_u32_to_mask(&vals, pattern, care, &mut out);
            if out != reference_mask(n, out_len, |i| (vals[i] ^ pattern) & care == 0) {
                return Err(0x700 | k as u32);
            }
            let (p64, c64) = ((pattern as u64) << 32 | pattern as u64, (care as u64) << 32 | care as u64);
            out.iter_mut().for_each(|w| *w = u64::MAX);
            ternary_match_u64_to_mask(&vals64, p64, c64, &mut out);
            if out != reference_mask(n, out_len, |i| (vals64[i] ^ p64) & c64 == 0) {
                return Err(0x710 | k as u32);
            }
        }
        // Strided 12-byte register at byte 4 of each 16-byte facet; every
        // third record is forced to match on the cared bytes.
        let mut bytes = vec![0u8; n * 16 + 5];
        let pattern: [u8; 12] = [0xA5, 0x00, 0x5A, 0xFF, 0x01, 0x02, 0x03, 0x04, 0x80, 0x7F, 0x10, 0x20];
        let care: [u8; 12] = [0xFF, 0x00, 0x0F, 0xFF, 0xFF, 0x00, 0xFF, 0x00, 0x80, 0x7F, 0xFF, 0xFF];
        for i in 0..n {
            let reg = &mut bytes[4 + i * 16..16 + i * 16];
            for (k, b) in reg.iter_mut().enumerate() {
                *b = if i % 3 == 0 {
                    pattern[k] ^ (!care[k] & (rng.next() as u8))
                } else {
                    rng.next() as u8
                };
            }
        }
        out.iter_mut().for_each(|w| *w = u64::MAX);
        ternary_match_strided_to_mask(&bytes, 4, 16, n, &pattern, &care, &mut out);
        let want = reference_mask(n, out_len, |i| {
            let reg = &bytes[4 + i * 16..16 + i * 16];
            (0..12).all(|k| (reg[k] ^ pattern[k]) & care[k] == 0)
        });
        if out != want {
            return Err(0x720);
        }
        // Anti-vacuity: the forced records must actually match, and at least
        // one unforced record must not (for n >= 2), or the check proved nothing.
        if n >= 2 && (want[0] & 1 == 0 || want.iter().map(|w| w.count_ones()).sum::<u32>() as usize == n) {
            return Err(0x721);
        }
    }
    Ok(())
}

// ── 0x8xx: masked reductions ────────────────────────────────────────────────

fn check_masked_reductions() -> Result<(), u32> {
    let mut rng = SplitMix64(0x8000_0000_0004);
    for &n in &LENS {
        let nw = words_for(n);
        let vals = i32_values(n, &mut rng);
        let tail_mask = |i: usize| -> u64 {
            let live = n.saturating_sub(i * 64).min(64);
            if live == 64 {
                u64::MAX
            } else {
                (1u64 << live) - 1
            }
        };
        let masks: [Vec<u64>; 4] = [
            vec![0u64; nw],
            (0..nw).map(tail_mask).collect(),
            (0..nw).map(|i| rng.next() & tail_mask(i)).collect(),
            (0..nw)
                .map(|i| 0x8000_0000_0000_0001u64 & tail_mask(i))
                .collect(),
        ];
        for (k, m) in masks.iter().enumerate() {
            let k = k as u32;
            let selected: Vec<i32> = (0..n)
                .filter(|&i| (m[i / 64] >> (i % 64)) & 1 == 1)
                .map(|i| vals[i])
                .collect();
            let want_sum: i64 = selected.iter().map(|&v| v as i64).sum();
            if masked_sum_i32(&vals, m) != want_sum {
                return Err(0x800 | k);
            }
            if masked_min_i32(&vals, m) != selected.iter().copied().min() {
                return Err(0x810 | k);
            }
            if masked_max_i32(&vals, m) != selected.iter().copied().max() {
                return Err(0x820 | k);
            }
        }
        // The extremes must survive the masked min/max exactly, not saturate.
        if n > 0 {
            let full: Vec<u64> = (0..nw).map(tail_mask).collect();
            if masked_min_i32(&vals, &full) != Some(i32::MIN)
                || (n > 1 && masked_max_i32(&vals, &full) != Some(i32::MAX))
            {
                return Err(0x830);
            }
        }
        // Strided group sum: 16-byte records, a register of `groups` fields of
        // `group_bytes` bytes at offset 4, for all three legal widths.
        for (gk, &(groups, group_bytes)) in [(3usize, 2usize), (6, 1), (2, 4), (1, 4)]
            .iter()
            .enumerate()
        {
            let mut bytes = vec![0u8; n * 16 + 1];
            let mut fields: Vec<Vec<u64>> = Vec::with_capacity(n);
            for i in 0..n {
                let mut rec = Vec::with_capacity(groups);
                for g in 0..groups {
                    let v = rng.next() & ((1u64 << (8 * group_bytes)) - 1);
                    let off = 4 + i * 16 + g * group_bytes;
                    bytes[off..off + group_bytes].copy_from_slice(&v.to_le_bytes()[..group_bytes]);
                    rec.push(v);
                }
                fields.push(rec);
            }
            let m = &masks[2];
            let want: i64 = (0..n)
                .filter(|&i| (m[i / 64] >> (i % 64)) & 1 == 1)
                .map(|i| fields[i].iter().sum::<u64>() as i64)
                .sum();
            if masked_strided_group_sum(&bytes, 4, 16, n, groups, group_bytes, m) != Some(want) {
                return Err(0x840 | gk as u32);
            }
        }
    }
    Ok(())
}

fn check_blend() -> Result<(), u32> {
    let mut rng = SplitMix64(0x8800_0000_0005);
    for &n in &LENS {
        let nw = words_for(n);
        let a = i32_values(n, &mut rng);
        let b: Vec<i32> = (0..n).map(|_| rng.next() as i32).collect();
        let m: Vec<u64> = (0..nw).map(|_| rng.next()).collect();
        let mut dst = vec![0i32; n];
        blend_i32(&m, &a, &b, &mut dst);
        for i in 0..n {
            let want = if (m[i / 64] >> (i % 64)) & 1 == 1 { a[i] } else { b[i] };
            if dst[i] != want {
                return Err(0x880);
            }
        }
    }
    Ok(())
}

// ── 0x9xx: mask_shift_morton — the Morton hex neighbour shift (D-GTM-1m) ────
//
// The reference below is an INDEPENDENT transcription of
// `hex_tenant_mq_probe.rs::neighbour`'s dilated-integer add/sub (never a
// call into `mask_shift_morton`), generalized from the probe's fixed
// 256×256/16-bit field to whatever field size this check picks.

/// Even/odd address-bit split for a field of `n_cells` cells.
fn morton_axis_bits(n_cells: usize) -> (u32, u32) {
    let total_bits = n_cells.trailing_zeros();
    let mut x = 0u32;
    let mut b = 0u32;
    while b < total_bits {
        x |= 1u32 << b;
        b += 2;
    }
    (x, x << 1)
}

/// One dilated-integer neighbour step; `None` at the field edge (no wrap).
fn morton_neighbour(m: u32, dq: i32, dr: i32, x_bits: u32, y_bits: u32) -> Option<u32> {
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

fn morton_reference_shift(src: &[u64], dq: i32, dr: i32, n_cells: usize, x_bits: u32, y_bits: u32) -> Vec<u64> {
    let mut dst = vec![0u64; src.len()];
    for i in 0..n_cells {
        if (src[i >> 6] >> (i & 63)) & 1 == 0 {
            continue;
        }
        if let Some(j) = morton_neighbour(i as u32, dq, dr, x_bits, y_bits) {
            let j = j as usize;
            dst[j >> 6] |= 1u64 << (j & 63);
        }
    }
    dst
}

fn check_morton_shift() -> Result<(), u32> {
    let mut rng = SplitMix64(0x9000_0000_0001);
    let dirs = [
        (MortonDir::PosQ, 1i32, 0i32),
        (MortonDir::NegQ, -1, 0),
        (MortonDir::PosR, 0, 1),
        (MortonDir::NegR, 0, -1),
    ];
    for &n_words in &[1usize, 4, 64, 1024] {
        let n_cells = n_words * 64;
        let (x_bits, y_bits) = morton_axis_bits(n_cells);
        for _ in 0..4 {
            let src: Vec<u64> = (0..n_words).map(|_| rng.next()).collect();
            for (k, &(dir, dq, dr)) in dirs.iter().enumerate() {
                let mut got = vec![0u64; n_words];
                mask_shift_morton(&src, dir, &mut got);
                let want = morton_reference_shift(&src, dq, dr, n_cells, x_bits, y_bits);
                if got != want {
                    return Err(0x900 | k as u32);
                }
            }
            // 0x904: the OR-accumulate contract — a pre-filled `dst` keeps its
            // prior bits (every other arm starts from zero and cannot see an
            // overwrite).
            let prior: Vec<u64> = (0..n_words).map(|_| rng.next()).collect();
            let mut got = prior.clone();
            mask_shift_morton(&src, MortonDir::PosQ, &mut got);
            let want = morton_reference_shift(&src, 1, 0, n_cells, x_bits, y_bits);
            let expect: Vec<u64> = prior.iter().zip(&want).map(|(p, w)| p | w).collect();
            if got != expect {
                return Err(0x904);
            }
        }
    }
    Ok(())
}

// ── 0xBxx: mask_set_range — the range WRITE, no per-bit loop (N1) ───────────
//
// The reference below is a from-scratch bit-serial writer over `[lo, hi)`
// membership — it does not call `mask_set_range` and does not reuse
// `reference_mask`'s predicate-over-index shape, because a range write's law
// is membership in an interval, not a predicate over `values[i]`.

fn set_range_reference(n_words: usize, lo: usize, hi: usize) -> Vec<u64> {
    let mut w = vec![0u64; n_words];
    for i in lo..hi {
        w[i / 64] |= 1u64 << (i % 64);
    }
    w
}

fn check_set_range() -> Result<(), u32> {
    let mut rng = SplitMix64(0xB000_0000_0006);
    for &n_words in &[1usize, 2, 3, 4, 8] {
        let capacity = n_words * 64;
        // The boundary shapes the primitive's own doc names, plus randomized
        // fuzz within capacity. `lo == hi` (both 0 and both `capacity`) is
        // the legal empty range; the rest straddle every word-alignment case.
        let mut cases: Vec<(usize, usize)> =
            vec![(0, 0), (capacity, capacity), (0, capacity), (0, 1), (capacity - 1, capacity), (5, 6)];
        if n_words >= 2 {
            cases.push((60, 70)); // adjacent words, neither aligned
            cases.push((64, 128)); // both endpoints word-aligned
            cases.push((0, 64)); // lo aligned, leaves surplus words unset
        }
        for _ in 0..12 {
            let lo = (rng.next() as usize) % (capacity + 1);
            let span = (rng.next() as usize) % (capacity + 1 - lo);
            cases.push((lo, lo + span));
        }
        for (lo, hi) in cases {
            let mut got = vec![u64::MAX; n_words]; // pre-dirtied: an OR-er fails immediately
            mask_set_range(&mut got, lo, hi);
            let want = set_range_reference(n_words, lo, hi);
            if got != want {
                return Err(0xB00);
            }
        }
        // Anti-vacuity: a genuinely non-trivial range sets EXACTLY `hi - lo`
        // bits, not "some" bits — catches an always-set-everything or
        // always-set-nothing implementation that could otherwise still pass
        // every case above.
        if capacity >= 8 {
            let (lo, hi) = (3, capacity - 2);
            let mut got = vec![0u64; n_words];
            mask_set_range(&mut got, lo, hi);
            let popcount: u32 = got.iter().map(|w| w.count_ones()).sum();
            if popcount as usize != hi - lo {
                return Err(0xB01);
            }
        }
    }
    Ok(())
}
