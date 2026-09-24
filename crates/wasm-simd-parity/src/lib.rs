//! Run-under-node parity gate for the wasm SIMD tier.
//!
//! `#[no_mangle] selfcheck()` runs each wasm SIMD lane's arithmetic against the
//! scalar `u32`/`f32`/`i8` reference **in the same module**, returning `0` iff
//! every lane is bit-identical. The companion `run.mjs` instantiates the
//! `.wasm` and asserts `selfcheck() == 0`; `scripts/wasm-parity.sh` wires the
//! build + node run, and the CI `wasm_simd` job runs the script. This is the
//! standing version of the one-off node check from commit 500d57e6 — extend the
//! per-lane blocks below whenever a new `ndarray::simd` lane lands.
//!
//! The whole crate is gated to `wasm32 + simd128` so it compiles to nothing on
//! any other target (it is workspace-excluded and only built by the CI job).
#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]

use ndarray::simd::{F32x16, I8x16, U32x16};

/// Distinct nonzero return codes make a CI failure point at the exact lane+op.
#[no_mangle]
pub extern "C" fn selfcheck() -> u32 {
    if let Err(code) = check_u32x16() {
        return code;
    }
    if let Err(code) = check_f32x16() {
        return code;
    }
    if let Err(code) = check_i8x16() {
        return code;
    }
    if let Err(code) = check_ternlog_all_tables() {
        return code;
    }
    if let Err(code) = check_u64x8_algebra() {
        return code;
    }
    if let Err(code) = check_i32x16_compare() {
        return code;
    }
    0
}

/// `U32x16` ARX triple (Add / BitXor / rotate_left) — the ChaCha20/BLAKE lane.
fn check_u32x16() -> Result<(), u32> {
    let a_arr: [u32; 16] = [
        0x0000_0000, 0xFFFF_FFFF, 0x0000_0001, 0x8000_0000, 0x1234_5678, 0x9ABC_DEF0, 0xDEAD_BEEF, 0xCAFE_BABE,
        0x0F0F_0F0F, 0xF0F0_F0F0, 0x5555_5555, 0xAAAA_AAAA, 0x0000_00FF, 0xFF00_0000, 0x0101_0101, 0x8080_8080,
    ];
    let b_arr: [u32; 16] = [
        0x9E37_79B9, 0x1111_1111, 0xDEAD_C0DE, 0x0BAD_F00D, 0x7FFF_FFFF, 0x0000_0000, 0xFFFF_FFFF, 0x1357_9BDF,
        0x2468_ACE0, 0xFEDC_BA98, 0x0000_0010, 0x0000_001F, 0xABCD_EF01, 0x1020_4080, 0x0F0F_F0F0, 0xC0DE_CAFE,
    ];
    // from_array / to_array roundtrip (lane ordering).
    if U32x16::from_array(a_arr).to_array() != a_arr {
        return Err(10);
    }
    let a = U32x16::from_array(a_arr);
    let b = U32x16::from_array(b_arr);
    let add = (a + b).to_array();
    let xor = (a ^ b).to_array();
    for i in 0..16 {
        if add[i] != a_arr[i].wrapping_add(b_arr[i]) {
            return Err(11);
        }
        if xor[i] != a_arr[i] ^ b_arr[i] {
            return Err(12);
        }
    }
    // ARX rotate — ChaCha20 uses 16/12/8/7; edges included.
    for &n in &[0u32, 1, 7, 8, 12, 16, 24, 31] {
        let r = a.rotate_left(n).to_array();
        for i in 0..16 {
            if r[i] != a_arr[i].rotate_left(n) {
                return Err(13);
            }
        }
    }
    Ok(())
}

/// `F32x16` — the float hot-path lane (splat / roundtrip / add / reduce_sum).
fn check_f32x16() -> Result<(), u32> {
    let data: [f32; 16] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    if F32x16::from_array(data).to_array() != data {
        return Err(20);
    }
    let a = F32x16::from_array(data);
    let b = F32x16::splat(2.0);
    let sum = (a + b).to_array();
    for i in 0..16 {
        if sum[i] != data[i] + 2.0 {
            return Err(21);
        }
    }
    // 0+1+…+15 = 120.
    if F32x16::from_array(data).reduce_sum() != 120.0 {
        return Err(22);
    }
    Ok(())
}

/// `I8x16` — the byte lane (roundtrip / add).
fn check_i8x16() -> Result<(), u32> {
    let a_arr: [i8; 16] = [-128, -1, 0, 1, 127, 2, -2, 3, -3, 42, -42, 100, -100, 7, -7, 64];
    let b_arr: [i8; 16] = [1, 1, 1, 1, 1, -1, -1, -1, 5, -5, 10, -10, 25, -25, 63, -64];
    if I8x16::from_array(a_arr).to_array() != a_arr {
        return Err(30);
    }
    let sum = I8x16::from_array(a_arr)
        .add(I8x16::from_array(b_arr))
        .to_array();
    for i in 0..16 {
        if sum[i] != a_arr[i].wrapping_add(b_arr[i]) {
            return Err(31);
        }
    }
    Ok(())
}

/// `ternlog::<IMM>` over ALL 256 truth tables, on BOTH lane types this target
/// resolves through `ndarray::simd`: the native `U32x16` and the native
/// `U64x8` (both `[v128; 4]` fan-outs with generated backend-local bodies —
/// `U64x8` was the scalar backend's re-export until the 2026-09-13
/// five-flavour audit). The reference is bit-serial and lives here,
/// independent of every backend. This proves the WASM arm only: the x86 arms
/// run under `cargo test` (`simd::tests::w1a9_*`), NEON under
/// `crates/neon-simd-parity` on qemu.
///
/// Operands: the canonical `F0/CC/AA` triple (every one of the 8 index
/// combinations appears in every byte) plus a second triple with mixed lane
/// values so a per-lane transposition would be caught as well. Return codes:
/// `40 + ` nothing lane-specific — the failing table is reported through the
/// code `0x100 | IMM` for `U32x16` and `0x200 | IMM` for `U64x8`, so a CI
/// failure names the exact truth table.
fn check_ternlog_all_tables() -> Result<(), u32> {
    use ndarray::simd::U64x8;

    fn reference_u64(imm: u32, a: u64, b: u64, c: u64) -> u64 {
        let mut r = 0u64;
        for bit in 0..64 {
            let idx = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1);
            r |= (((imm >> idx) & 1) as u64) << bit;
        }
        r
    }

    // Two operand triples per lane width. Triple 0 is the canonical
    // all-8-minterms pattern; triple 1 mixes per-lane values.
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
                    let want = reference_u64(IMM as u32, a[i] as u64, b[i] as u64, c[i] as u64) as u32;
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
                    if got[i] != reference_u64(IMM as u32, a[i], b[i], c[i]) {
                        return Err(0x200 | IMM as u32);
                    }
                }
            }
        }};
    }
    // 16 × 16 = all 256 tables, each a distinct monomorphization.
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

/// `U64x8` algebra on this tier's NATIVE lane type against inline scalar `u64`
/// arithmetic — the bulk mask ops (`mask_and/or/xor/andnot`) ride exactly
/// these operators. Operands populate all-zero, all-one, the sign bit and
/// mixed patterns so a lane transposition or a wrong and-not polarity fails.
/// Return codes `0x300 + op`.
fn check_u64x8_algebra() -> Result<(), u32> {
    use ndarray::simd::U64x8;
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

/// `I32x16` signed compares and arithmetic on this tier's NATIVE lane type
/// against inline scalar `i32` — `gt_bitmask` is the whole ordered-compare
/// family's primitive (`gt/lt/ge/le_i32_to_mask`). Operands include
/// `i32::MIN`, `i32::MAX`, `0`, `-1` and lanes that compare EQUAL, and the
/// `u16` bit order (bit i = lane i) is asserted lane by lane. Return codes
/// `0x400 + op`.
fn check_i32x16_compare() -> Result<(), u32> {
    use ndarray::simd::I32x16;
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
    if a.reduce_min() != *a_arr.iter().min().unwrap() || a.reduce_max() != *a_arr.iter().max().unwrap() {
        return Err(0x405);
    }
    // `abs`/`neg` wrap at i32::MIN on every backend (release semantics).
    if a.abs().to_array() != core::array::from_fn(|i| a_arr[i].wrapping_abs()) {
        return Err(0x406);
    }
    if (-a).to_array() != core::array::from_fn(|i| a_arr[i].wrapping_neg()) {
        return Err(0x407);
    }
    if (a + b).to_array() != core::array::from_fn(|i| a_arr[i].wrapping_add(b_arr[i])) {
        return Err(0x408);
    }
    if (a * b).to_array() != core::array::from_fn(|i| a_arr[i].wrapping_mul(b_arr[i])) {
        return Err(0x409);
    }
    let mut sum = 0i32;
    for &x in &a_arr {
        sum = sum.wrapping_add(x);
    }
    if a.reduce_sum() != sum {
        return Err(0x40A);
    }
    let i16s: [i16; 16] = core::array::from_fn(|i| (i as i16 - 8) * 1000);
    if I32x16::from_i16_slice(&i16s).to_array() != core::array::from_fn(|i| i16s[i] as i32) {
        return Err(0x40B);
    }
    if a.to_i16_array() != core::array::from_fn(|i| a_arr[i] as i16) {
        return Err(0x40C);
    }
    if !(a == I32x16::from_array(a_arr)) || a == b {
        return Err(0x40D);
    }
    Ok(())
}
