#![cfg(feature = "std")]
#![allow(clippy::needless_range_loop)]
//! Exactness of the `add_pm` probe (`examples/add_pm/pm.rs`): `P × M` lanes
//! added in place against a decode-to-i32 oracle.

#[path = "../examples/add_pm/pm.rs"]
mod pm;

use ndarray::hpc::codec8::TiesEven;
use ndarray::hpc::codec9::Codec9Runtime;
use ndarray::simd::U8x64;
use pm::{add_decoded, add_pm, add_ref, decode_lane, neg_pm, sub_pm, sub_ref, touched, Pm};

fn lane(x: Pm, i: usize) -> Option<i32> {
    decode_lane(x.mag.to_array()[i], x.pos >> i & 1 == 1)
}

/// Codeword `c` in 0..512: magnitude `c & 255`, presence bit 8.
fn block_of(codes: &[u16]) -> Pm {
    let mut mag = [0u8; 64];
    let mut pos = 0u64;
    for (i, &c) in codes.iter().enumerate() {
        mag[i] = (c & 0xff) as u8;
        pos |= u64::from(c >> 8 & 1) << i;
    }
    Pm {
        mag: U8x64::from_array(mag),
        pos,
    }
}

/// Every (a, b) codeword pair, 64 per block.
/// A block pair and the codeword pair of each of its lanes.
type PairBlock = (Pm, Pm, Vec<(u16, u16)>);

fn all_pairs() -> Vec<PairBlock> {
    let pairs: Vec<(u16, u16)> = (0..512u16)
        .flat_map(|a| (0..512u16).map(move |b| (a, b)))
        .collect();
    pairs
        .chunks(64)
        .map(|c| {
            let a: Vec<u16> = c.iter().map(|p| p.0).collect();
            let b: Vec<u16> = c.iter().map(|p| p.1).collect();
            (block_of(&a), block_of(&b), c.to_vec())
        })
        .collect()
}

#[test]
fn every_codeword_pair_matches_the_i32_oracle() {
    let (mut over, mut checked) = (0usize, 0usize);
    for (a, b, codes) in all_pairs() {
        let (s, carry) = add_pm(a, b);
        for i in 0..codes.len() {
            let (want, overflow) = add_ref(lane(a, i), lane(b, i));
            assert_eq!(carry >> i & 1 == 1, overflow, "carry, codes {:?}", codes[i]);
            checked += 1;
            if overflow {
                over += 1;
                let m = s.mag.to_array()[i];
                assert_eq!(m, 255, "a carry lane saturates");
                // polarity of the inputs (both inputs share it on a carry)
                assert_eq!(lane(s, i).map(i32::signum), want.map(i32::signum), "{:?}", codes[i]);
            } else {
                assert_eq!(lane(s, i), want, "codes {:?}", codes[i]);
            }
        }
    }
    assert_eq!(checked, 512 * 512);
    // exactly the touched same-polarity pairs with |a| + |b| > 255
    let expect: usize = (1..=255i32)
        .flat_map(|x| (1..=255i32).map(move |y| usize::from(x + y > 255)))
        .sum::<usize>()
        * 2;
    assert_eq!(over, expect, "carry count");
    assert!(over > 0);
}

#[test]
fn never_touched_comes_out_only_of_two_never_touched_inputs() {
    for (a, b, _) in all_pairs() {
        let (s, _) = add_pm(a, b);
        assert_eq!(touched(s), touched(a) | touched(b));
    }
}

#[test]
fn a_touched_zero_is_plus_zero() {
    // -5 + +5, +0 + +0, -255 + +255
    let a = block_of(&[5, 0x100, 255]);
    let b = block_of(&[0x105, 0x100, 0x1ff]);
    let (s, carry) = add_pm(a, b);
    assert_eq!(carry, 0);
    for i in 0..3 {
        assert_eq!(s.mag.to_array()[i], 0);
        assert_eq!(s.pos >> i & 1, 1, "lane {i} must be +0, not the never-touched code");
    }
}

#[test]
fn add_pm_equals_the_decode_path_on_every_pair() {
    for (a, b, _) in all_pairs() {
        assert_eq!(add_pm(a, b), add_decoded(a, b));
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn block(&mut self, max: u16) -> Pm {
        let codes: Vec<u16> = (0..64)
            .map(|_| {
                let m = (self.next() % (u64::from(max) + 1)) as u16;
                let p = (self.next() & 1) as u16;
                // a quarter never touched
                if self.next().is_multiple_of(4) {
                    0
                } else {
                    m | p << 8
                }
            })
            .collect();
        block_of(&codes)
    }
}

#[test]
fn addition_is_associative_where_nothing_carries() {
    let mut r = Rng(0x9E37_79B9_7F4A_7C15);
    let mut compared = 0usize;
    for _ in 0..20_000 {
        let (a, b, c) = (r.block(255), r.block(255), r.block(255));
        let (ab, k1) = add_pm(a, b);
        let (ab_c, k2) = add_pm(ab, c);
        let (bc, k3) = add_pm(b, c);
        let (a_bc, k4) = add_pm(a, bc);
        let clean = !(k1 | k2 | k3 | k4);
        for i in 0..64 {
            if clean >> i & 1 == 1 {
                assert_eq!(lane(ab_c, i), lane(a_bc, i));
                compared += 1;
            }
        }
    }
    assert!(compared > 100_000, "anti-vacuity: {compared} lanes compared");
}

/// The escape in use: fold a stream of delta blocks into one lane block;
/// carry lanes move their exact sum into a wide tally and restart untouched.
/// The result must equal an i64 sum of the decoded stream.
#[test]
fn a_fold_with_the_carry_escape_is_exact() {
    let mut r = Rng(0xC0DE_C9C9_0000_0001);
    let mut acc = Pm::untouched();
    let mut wide = [0i64; 64];
    let mut truth = [None::<i64>; 64];
    let mut escapes = 0usize;
    for _ in 0..5_000 {
        let d = r.block(200);
        for i in 0..64 {
            if let Some(v) = lane(d, i) {
                *truth[i].get_or_insert(0) += i64::from(v);
            }
        }
        let (s, carry) = add_pm(acc, d);
        if carry != 0 {
            for i in 0..64 {
                if carry >> i & 1 == 1 {
                    let (x, y) = (lane(acc, i).unwrap(), lane(d, i).unwrap());
                    wide[i] += i64::from(x + y);
                    escapes += 1;
                }
            }
        }
        // carry lanes restart touched at +0: their value now lives in `wide`
        let mut mag = s.mag.to_array();
        let mut pos = s.pos;
        for i in 0..64 {
            if carry >> i & 1 == 1 {
                mag[i] = 0;
                pos |= 1 << i;
            }
        }
        acc = Pm {
            mag: U8x64::from_array(mag),
            pos,
        };
    }
    for i in 0..64 {
        let got = lane(acc, i).map(|v| i64::from(v) + wide[i]);
        assert_eq!(got, truth[i], "lane {i}");
    }
    assert!(escapes > 0, "anti-vacuity: the stream must overflow");
}

// ------------------------------------------------------------------ subtraction

#[test]
fn negation_of_every_codeword_matches_the_oracle() {
    let codes: Vec<u16> = (0..512).collect();
    for c in codes.chunks(64) {
        let b = block_of(c);
        let n = neg_pm(b);
        assert_eq!(n.mag, b.mag, "negation keeps the magnitude");
        for i in 0..c.len() {
            assert_eq!(lane(n, i), lane(b, i).map(|y| -y), "code {:#x}", c[i]);
        }
        assert_eq!(touched(n), touched(b), "touched is unchanged");
    }
    // the never-touched code and +0 are fixed points, -0 is never produced
    let z = block_of(&[0, 0x100]);
    let n = neg_pm(z);
    assert_eq!((n.mag.to_array()[0], n.pos & 1), (0, 0), "never touched stays never touched");
    assert_eq!((n.mag.to_array()[1], n.pos >> 1 & 1), (0, 1), "+0 stays +0");
}

#[test]
fn every_codeword_pair_difference_matches_the_i32_oracle() {
    let (mut over, mut checked) = (0usize, 0usize);
    for (a, b, codes) in all_pairs() {
        let (s, carry) = sub_pm(a, b);
        for i in 0..codes.len() {
            let (want, overflow) = sub_ref(lane(a, i), lane(b, i));
            assert_eq!(carry >> i & 1 == 1, overflow, "carry, codes {:?}", codes[i]);
            checked += 1;
            if overflow {
                over += 1;
                assert_eq!(s.mag.to_array()[i], 255, "a carry lane saturates");
                assert_eq!(lane(s, i).map(i32::signum), want.map(i32::signum), "{:?}", codes[i]);
            } else {
                assert_eq!(lane(s, i), want, "codes {:?}", codes[i]);
            }
        }
        assert_eq!(touched(s), touched(a) | touched(b), "touched(a - b)");
    }
    assert_eq!(checked, 512 * 512);
    // opposite-polarity pairs with |a| + |b| > 255 carry under subtraction
    let expect: usize = (1..=255i32)
        .flat_map(|x| (1..=255i32).map(move |y| usize::from(x + y > 255)))
        .sum::<usize>()
        * 2;
    assert_eq!(over, expect, "carry count");
}

#[test]
fn sub_pm_equals_negate_then_add_on_every_pair() {
    for (a, b, _) in all_pairs() {
        assert_eq!(sub_pm(a, b), add_pm(a, neg_pm(b)));
    }
}

/// The four cases the contract names, on one block.
#[test]
fn the_four_mandatory_cases() {
    // -5 + +5 = +0
    let (s, c) = add_pm(block_of(&[5]), block_of(&[0x105]));
    assert_eq!((c & 1, s.mag.to_array()[0], s.pos & 1), (0, 0, 1), "-5 + +5 = +0");

    // -0.499999 quantizes to +0 (scale 1, ties to even), through the batch
    // encoder into a block, then survives an addition with never touched
    let codec = Codec9Runtime::<TiesEven>::new(1.0).unwrap();
    let mut mag = [0u8; 64];
    let mut pos = [0u64; 1];
    codec
        .encode_touched_batch(&[-0.499_999f32; 64], &mut mag, &mut pos)
        .unwrap();
    let q = Pm {
        mag: U8x64::from_array(mag),
        pos: pos[0],
    };
    assert_eq!((q.mag.to_array()[0], q.pos & 1), (0, 1), "-0.499999 is +0, not -0");
    let (s, _) = add_pm(q, Pm::untouched());
    assert_eq!(touched(s), u64::MAX, "still touched");
    assert!(s.mag.to_array().iter().all(|&m| m == 0) && s.pos == u64::MAX);

    // +0 - +0 = +0
    let (s, c) = sub_pm(block_of(&[0x100]), block_of(&[0x100]));
    assert_eq!((c & 1, s.mag.to_array()[0], s.pos & 1), (0, 0, 1), "+0 - +0 = +0");

    // never_touched - never_touched = never_touched
    let (s, c) = sub_pm(Pm::untouched(), Pm::untouched());
    assert_eq!((c, touched(s)), (0, 0), "never touched - never touched");
}

#[test]
fn a_difference_fold_with_the_carry_escape_is_exact() {
    let mut r = Rng(0xD1FF_5EED);
    let n = 2000;
    let mut acc = Pm::untouched();
    let mut wide = [0i64; 64];
    let mut escaped = 0u64;
    let mut exact = [None::<i64>; 64];
    let mut carries = 0;
    for _ in 0..n {
        let b = r.block(160);
        let bm = b.mag.to_array();
        for i in 0..64 {
            if let Some(v) = decode_lane(bm[i], b.pos >> i & 1 == 1) {
                exact[i] = Some(exact[i].unwrap_or(0) - i64::from(v));
            }
        }
        // escaped lanes take the delta exactly; the rest go through sub_pm
        let off = escaped;
        let b_in = Pm {
            mag: U8x64::mask_blend(off, b.mag, U8x64::splat(0)),
            pos: b.pos & !off,
        };
        let before = acc;
        let (s, carry) = sub_pm(acc, b_in);
        acc = s;
        for i in 0..64 {
            let d = decode_lane(bm[i], b.pos >> i & 1 == 1).map(i64::from);
            if escaped >> i & 1 == 1 {
                wide[i] -= d.unwrap_or(0);
            } else if carry >> i & 1 == 1 {
                carries += 1;
                let a = lane(before, i).map(i64::from).unwrap_or(0);
                wide[i] = a - d.unwrap_or(0);
                escaped |= 1 << i;
            }
        }
    }
    for i in 0..64 {
        let got = if escaped >> i & 1 == 1 {
            Some(wide[i])
        } else {
            lane(acc, i).map(i64::from)
        };
        assert_eq!(got, exact[i], "lane {i}");
    }
    assert!(carries > 0, "the fold must carry");
}
