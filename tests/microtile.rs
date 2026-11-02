#![cfg(feature = "std")]
//! D-SIMD-MICRO-MACRO-1 step 1 exactness (`examples/microtile/tile.rs`):
//! every realization against the unpacked per-field oracle.

#[path = "../examples/microtile/tile.rs"]
mod tile;

use ndarray::simd::U64x8;
use tile::*;

fn xorshift(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

fn rand128(s: &mut u64) -> u128 {
    (xorshift(s) as u128) << 64 | xorshift(s) as u128
}

/// Field values that hit carry and sign boundaries: 0, 1, the sign bit, the
/// sign bit - 1, max - 1, max.
fn edges<P: Packing>() -> Vec<u32> {
    let m = P::VMASK as u32;
    let h = 1u32 << (P::W - 1);
    vec![0, 1, h - 1, h, m - 1, m]
}

/// Random tiles plus every tile whose fields all share one edge value, plus
/// tiles that alternate two edge values (neighbour fields differ).
fn corpus<P: Packing>(seed: u64, random: usize) -> Vec<u128> {
    let mut s = seed;
    let mut v: Vec<u128> = (0..random).map(|_| rand128(&mut s)).collect();
    let e = edges::<P>();
    for &x in &e {
        for &y in &e {
            let fields: Vec<u32> = (0..P::N).map(|j| if j % 2 == 0 { x } else { y }).collect();
            v.push(pack::<P>(&fields, rand128(&mut s)));
        }
    }
    v
}

fn check_scalar<P: Packing>(name: &str) {
    let c = corpus::<P>(0x9E37_79B9_7F4A_7C15, 4000);
    for (i, &a) in c.iter().enumerate() {
        let b = c[(i * 7 + 3) % c.len()];
        let r = add::<P>(a, b);
        let (want, carry, ovf) = oracle_add::<P>(a, b);
        assert_eq!(r, want, "{name} add tile {i}: {a:#x} + {b:#x}");
        assert_eq!(flags::<P>(carry_out::<P>(a, b, r)), carry, "{name} carry tile {i}");
        assert_eq!(flags::<P>(signed_overflow::<P>(a, b, r)), ovf, "{name} signed overflow tile {i}");
        let d = sub::<P>(a, b);
        let (want, borrow) = oracle_sub::<P>(a, b);
        assert_eq!(d, want, "{name} sub tile {i}: {a:#x} - {b:#x}");
        assert_eq!(flags::<P>(borrow_out::<P>(a, b, d)), borrow, "{name} borrow tile {i}");
        assert_eq!(stencil::<P>(a), oracle_stencil::<P>(a), "{name} stencil tile {i}: {a:#x}");
    }
}

#[test]
fn u24x5_u128_matches_the_oracle() {
    check_scalar::<U24x5>("U24x5");
}

#[test]
fn u9x14_u128_matches_the_oracle() {
    check_scalar::<U9x14>("U9x14");
}

/// The fields that straddle bit 64 must take a carry and a borrow across the
/// two words: low part all ones plus one, and zero minus one.
fn check_straddle<P: Packing>(name: &str, j: u32) {
    assert!(j * P::W < 64 && j * P::W + P::W > 64, "{name}: field {j} does not straddle bit 64");
    let low_bits = 64 - j * P::W;
    let mut f = vec![0u32; P::N as usize];
    f[j as usize] = (1 << low_bits) - 1;
    let a = pack::<P>(&f, 0);
    let mut g = vec![0u32; P::N as usize];
    g[j as usize] = 1;
    let b = pack::<P>(&g, 0);
    let (alo, ahi) = (U64x8::splat(a as u64), U64x8::splat((a >> 64) as u64));
    let (blo, bhi) = (U64x8::splat(b as u64), U64x8::splat((b >> 64) as u64));
    let [rl, rh, _, _] = soa_add::<P>(alo, ahi, blo, bhi);
    let r = rl.to_array()[0] as u128 | (rh.to_array()[0] as u128) << 64;
    assert_eq!(get::<P>(r, j), 1 << low_bits, "{name}: carry did not cross bit 64");
    let [dl, dh] = soa_sub::<P>(U64x8::splat(0), U64x8::splat(0), blo, bhi);
    let d = dl.to_array()[0] as u128 | (dh.to_array()[0] as u128) << 64;
    assert_eq!(get::<P>(d, j), P::VMASK as u32, "{name}: borrow did not cross bit 64");
}

#[test]
fn straddling_fields_carry_and_borrow_across_the_word_boundary() {
    check_straddle::<U24x5>("U24x5", 2);
    check_straddle::<U9x14>("U9x14", 7);
}

fn check_soa<P: Packing>(name: &str) {
    // 1001 tiles: not a multiple of 8, so the u128 tail runs too.
    let a = corpus::<P>(11, 1001 - 36);
    let b = corpus::<P>(12, 1001 - 36);
    assert_eq!(a.len(), 1001);
    let ((alo, ahi), (blo, bhi)) = (split(&a), split(&b));
    let (sa, sb) = (Soa { lo: &alo, hi: &ahi }, Soa { lo: &blo, hi: &bhi });
    let (mut ol, mut oh) = (vec![0u64; a.len()], vec![0u64; a.len()]);
    soa_add_slices::<P>(&sa, &sb, &mut ol, &mut oh);
    for t in 0..a.len() {
        let r = ol[t] as u128 | (oh[t] as u128) << 64;
        assert_eq!(r, oracle_add::<P>(a[t], b[t]).0, "{name} soa add tile {t}");
    }
    soa_sub_slices::<P>(&sa, &sb, &mut ol, &mut oh);
    for t in 0..a.len() {
        let r = ol[t] as u128 | (oh[t] as u128) << 64;
        assert_eq!(r, oracle_sub::<P>(a[t], b[t]).0, "{name} soa sub tile {t}");
    }
    // Carry-out planes from the vector form against the oracle.
    for i in (0..a.len() - a.len() % 8).step_by(8) {
        let [rl, rh, cl, ch] = soa_add::<P>(
            U64x8::from_slice(&alo[i..]),
            U64x8::from_slice(&ahi[i..]),
            U64x8::from_slice(&blo[i..]),
            U64x8::from_slice(&bhi[i..]),
        );
        let (rl, rh, cl, ch) = (rl.to_array(), rh.to_array(), cl.to_array(), ch.to_array());
        for k in 0..8 {
            let r = rl[k] as u128 | (rh[k] as u128) << 64;
            let c = cl[k] as u128 | (ch[k] as u128) << 64;
            assert_eq!(
                flags::<P>(c),
                flags::<P>(carry_out::<P>(a[i + k], b[i + k], r)),
                "{name} soa carry tile {}",
                i + k
            );
            assert_eq!(flags::<P>(c), oracle_add::<P>(a[i + k], b[i + k]).1, "{name} soa carry tile {}", i + k);
        }
    }
    let (u, s) = soa_fold::<P>(&sa);
    let (wu, ws) = oracle_fold::<P>(&a);
    assert_eq!(u, wu, "{name} soa unsigned fold");
    assert_eq!(s, ws, "{name} soa signed fold");
}

#[test]
fn u24x5_soa_matches_the_oracle() {
    check_soa::<U24x5>("U24x5");
}

#[test]
fn u9x14_soa_matches_the_oracle() {
    check_soa::<U9x14>("U9x14");
}

fn check_widened_and_planes<P: Packing>(name: &str, n: usize) {
    let a = corpus::<P>(21, n - 36);
    let b = corpus::<P>(22, n - 36);
    let (wa, wb) = (widen::<P>(&a), widen::<P>(&b));
    let (wu, ws) = oracle_fold::<P>(&a);
    let (pa, pb) = (to_planes::<P>(&a), to_planes::<P>(&b));
    for j in 0..P::N as usize {
        let mut out = vec![0u32; n];
        widened_add::<P>(&wa[j], &wb[j], &mut out);
        for t in 0..n {
            assert_eq!(
                out[t],
                get::<P>(oracle_add::<P>(a[t], b[t]).0, j as u32),
                "{name} widened add field {j} tile {t}"
            );
        }
        assert_eq!(widened_fold::<P>(&wa[j]), wu[j], "{name} widened fold field {j}");

        let w = P::W as usize;
        for blk in 0..n / 64 {
            let ak = &pa[j][blk * w..(blk + 1) * w];
            let bk = &pb[j][blk * w..(blk + 1) * w];
            let mut out = vec![0u64; w];
            let c = planes_add(ak, bk, &mut out);
            for lane in 0..64 {
                let t = blk * 64 + lane;
                let v = (0..w).fold(0u32, |acc, k| acc | ((out[k] >> lane & 1) as u32) << k);
                let (r, carry, _) = oracle_add::<P>(a[t], b[t]);
                assert_eq!(v, get::<P>(r, j as u32), "{name} planes add field {j} tile {t}");
                assert_eq!(c >> lane & 1, (carry >> j & 1) as u64, "{name} planes carry field {j} tile {t}");
            }
        }
        assert_eq!(planes_fold(&pa[j], w), (wu[j], ws[j]), "{name} planes fold field {j}");
    }
}

#[test]
fn u24x5_widened_and_planes_match_the_oracle() {
    check_widened_and_planes::<U24x5>("U24x5", 1024);
}

#[test]
fn u9x14_widened_and_planes_match_the_oracle() {
    check_widened_and_planes::<U9x14>("U9x14", 1024);
}

/// A u32 lane holds only 256 maximal U24 values; the widened fold must flush
/// before then. 8192 maximal tiles put 512 values in every lane.
#[test]
fn widened_fold_does_not_wrap_its_u32_lanes() {
    let m = U24x5::VMASK as u32;
    let field = vec![m; 8192];
    assert_eq!(widened_fold::<U24x5>(&field), m as u64 * 8192);
}

/// Metadata of the left operand survives every operation, including when the
/// right operand's metadata is all ones and every field overflows.
#[test]
fn metadata_is_the_left_operands() {
    let meta_a = 0xA5u128 << 120;
    let a = pack::<U24x5>(&[U24x5::VMASK as u32; 5], meta_a);
    let b = pack::<U24x5>(&[1; 5], !0);
    assert_eq!(add::<U24x5>(a, b) & U24x5::META, meta_a);
    assert_eq!(sub::<U24x5>(a, b) & U24x5::META, meta_a);
    assert_eq!(stencil::<U24x5>(a) & U24x5::META, meta_a);
    assert_eq!(add::<U24x5>(a, b) & U24x5::FIELDS, 0, "every field wraps to zero");
}
