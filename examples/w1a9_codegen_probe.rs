use ndarray::simd::U64x8;
use std::hint::black_box;

#[inline(never)]
pub fn probe_and3(a: U64x8, b: U64x8, c: U64x8) -> U64x8 {
    a.ternlog::<0x80>(b, c)
}

#[inline(never)]
pub fn probe_and2_andnot(a: U64x8, b: U64x8, c: U64x8) -> U64x8 {
    a.ternlog::<0x40>(b, c)
}

#[inline(never)]
pub fn probe_andnot(a: U64x8, b: U64x8) -> U64x8 {
    a.andnot(b)
}

fn main() {
    let a = black_box(U64x8::splat(0xF0F0_F0F0_F0F0_F0F0));
    let b = black_box(U64x8::splat(0xCCCC_CCCC_CCCC_CCCC));
    let c = black_box(U64x8::splat(0xAAAA_AAAA_AAAA_AAAA));
    println!("{:x}", black_box(probe_and3(a, b, c)).to_array()[0]);
    println!("{:x}", black_box(probe_and2_andnot(a, b, c)).to_array()[0]);
    println!("{:x}", black_box(probe_andnot(a, b)).to_array()[0]);
}
