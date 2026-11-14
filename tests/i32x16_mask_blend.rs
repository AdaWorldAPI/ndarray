#![cfg(feature = "std")]

use ndarray::simd::I32x16;

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_i32(&mut self) -> i32 {
        (self.next_u64() >> 32) as u32 as i32
    }

    fn next_mask(&mut self) -> u16 {
        (self.next_u64() >> 48) as u16
    }

    fn next_array(&mut self) -> [i32; 16] {
        let mut out = [0i32; 16];
        for lane in out.iter_mut() {
            *lane = self.next_i32();
        }
        out
    }
}

const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

fn oracle(mask: u16, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for i in 0..16 {
        out[i] = if (mask >> i) & 1 == 1 { b[i] } else { a[i] };
    }
    out
}

fn blend(mask: u16, a: [i32; 16], b: [i32; 16]) -> [i32; 16] {
    I32x16::mask_blend(mask, I32x16::from_array(a), I32x16::from_array(b)).to_array()
}

fn fixed_pair() -> ([i32; 16], [i32; 16]) {
    let mut a = [0i32; 16];
    let mut b = [0i32; 16];
    for i in 0..16 {
        a[i] = 1000 + (i as i32) * 37 + 1;
        b[i] = -(5000 + (i as i32) * 53 + 1);
    }
    (a, b)
}

#[test]
fn mask_zero_returns_a_and_full_returns_b() {
    let (a, b) = fixed_pair();
    assert_eq!(blend(0x0000, a, b), a, "mask 0x0000 must return a");
    assert_eq!(blend(0xFFFF, a, b), b, "mask 0xFFFF must return b");
}

#[test]
fn single_set_bit_takes_only_that_lane_from_b() {
    let (a, b) = fixed_pair();
    for i in 0..16 {
        let mask: u16 = 1 << i;
        let mut expected = a;
        expected[i] = b[i];
        assert_eq!(blend(mask, a, b), expected, "mask {:#06x}", mask);
    }
}

#[test]
fn single_cleared_bit_takes_only_that_lane_from_a() {
    let (a, b) = fixed_pair();
    for i in 0..16 {
        let mask: u16 = !(1u16 << i);
        let mut expected = b;
        expected[i] = a[i];
        assert_eq!(blend(mask, a, b), expected, "mask {:#06x}", mask);
    }
}

#[test]
fn all_65536_masks_match_oracle() {
    let (a, b) = fixed_pair();
    for m in 0..=u16::MAX {
        assert_eq!(blend(m, a, b), oracle(m, a, b), "mask {:#06x}", m);
    }
}

#[test]
fn random_triples_match_oracle() {
    let mut rng = SplitMix64::new(SEED);
    for case in 0..2000 {
        let mask = rng.next_mask();
        let a = rng.next_array();
        let b = rng.next_array();
        assert_eq!(blend(mask, a, b), oracle(mask, a, b), "case {} mask {:#06x}", case, mask);
    }
}

#[test]
fn extreme_lane_values() {
    let pool = [i32::MIN, i32::MAX, -1, 0];
    let mut a = [0i32; 16];
    let mut b = [0i32; 16];
    for i in 0..16 {
        a[i] = pool[i % 4];
        b[i] = pool[(i + 1 + i / 4) % 4];
    }
    let masks: [u16; 7] = [0x0000, 0xFFFF, 0x5555, 0xAAAA, 0x00FF, 0xFF00, 0x8001];
    for &mask in masks.iter() {
        assert_eq!(blend(mask, a, b), oracle(mask, a, b), "mask {:#06x}", mask);
        assert_eq!(blend(mask, b, a), oracle(mask, b, a), "swapped mask {:#06x}", mask);
    }
}

#[test]
fn equal_inputs_ignore_mask() {
    let mut rng = SplitMix64::new(SEED);
    let a = rng.next_array();
    for _ in 0..64 {
        let mask = rng.next_mask();
        assert_eq!(blend(mask, a, a), a, "mask {:#06x}", mask);
    }
}

#[test]
fn composes_with_gt_bitmask_into_min_and_max() {
    let mut rng = SplitMix64::new(SEED);
    for case in 0..500 {
        let a = rng.next_array();
        let mut b = rng.next_array();
        for i in 0..16 {
            if rng.next_u64() & 3 == 0 {
                b[i] = a[i];
            }
        }
        let va = I32x16::from_array(a);
        let vb = I32x16::from_array(b);

        let mut want_min = [0i32; 16];
        let mut want_max = [0i32; 16];
        for i in 0..16 {
            want_min[i] = a[i].min(b[i]);
            want_max[i] = a[i].max(b[i]);
        }

        let got_min = I32x16::mask_blend(va.gt_bitmask(vb), va, vb).to_array();
        let got_max = I32x16::mask_blend(vb.gt_bitmask(va), va, vb).to_array();
        assert_eq!(got_min, want_min, "min case {}", case);
        assert_eq!(got_max, want_max, "max case {}", case);
    }
}

#[test]
fn alternating_mask_differs_from_both_inputs() {
    let (a, b) = fixed_pair();
    let got = blend(0x5555, a, b);
    assert_ne!(got, a, "mask 0x5555 must not return a");
    assert_ne!(got, b, "mask 0x5555 must not return b");
    assert_eq!(got, oracle(0x5555, a, b), "mask 0x5555");
}

#[test]
fn chained_blends_match_oracle_twice() {
    let mut rng = SplitMix64::new(SEED);
    for case in 0..200 {
        let m1 = rng.next_mask();
        let m2 = rng.next_mask();
        let a = rng.next_array();
        let b = rng.next_array();
        let c = rng.next_array();

        let inner = I32x16::mask_blend(m1, I32x16::from_array(a), I32x16::from_array(b));
        let got = I32x16::mask_blend(m2, inner, I32x16::from_array(c)).to_array();
        let want = oracle(m2, oracle(m1, a, b), c);
        assert_eq!(got, want, "case {} masks {:#06x} {:#06x}", case, m1, m2);
    }
}
