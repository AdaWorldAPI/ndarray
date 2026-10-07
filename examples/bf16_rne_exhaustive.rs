//! Exhaustive F32 -> BF16 round-to-nearest-even parity: ALL 2^32 f32 bit patterns.
//!
//! Reproduces the README claim "4,294,967,296 inputs, 0 mismatches". For every
//! `u32` bit pattern `b`, `simd::f32_to_bf16_batch_rne` is compared bit-for-bit
//! (u16 equality, so NaN payloads and signs count) against
//!
//! 1. `simd::f32_to_bf16_scalar_rne`, the repo's scalar reference, and
//! 2. an INDEPENDENT oracle defined here: pick whichever of the two adjacent
//!    BF16 values is nearer in f64, ties to the even mantissa; overflow past
//!    BF16::MAX rounds to infinity at the 2^128 midpoint.
//!
//! The intended semantics are Intel SDM `VCVTNEPS2BF16`:
//! - NaN keeps its sign and top 7 payload bits, and the quiet bit is forced;
//! - subnormal input flushes to a signed zero (DAZ);
//! - infinities and zeros pass through.
//!
//! The oracle encodes exactly that and nothing else.
//!
//! Method: the 2^32 space is split into `threads` equal contiguous ranges. Each
//! range is enumerated in chunks of 65,536 inputs; each chunk is converted in
//! one batch call. An order-independent checksum of every converted output
//! (Σ out(b)·(b|1) mod 2^64) is printed so the work cannot be optimized away
//! and runs with different thread counts can be compared.
//!
//! Run (x86_64; the batch takes the AVX-512F path when the host has it):
//! ```text
//! cargo run --release --example bf16_rne_exhaustive [threads]
//! ```

#[cfg(target_arch = "x86_64")]
fn oracle_bf16(bits: u32) -> u16 {
    let exp = bits & 0x7F80_0000;
    let mant = bits & 0x007F_FFFF;
    if exp == 0x7F80_0000 && mant != 0 {
        return ((bits >> 16) as u16) | 0x0040; // NaN: keep sign + top payload, force quiet
    }
    if exp == 0 {
        return ((bits >> 16) as u16) & 0x8000; // zero or subnormal -> signed zero
    }
    if exp == 0x7F80_0000 {
        return (bits >> 16) as u16; // infinity
    }
    let x = f32::from_bits(bits) as f64;
    let lo = (bits >> 16) as u16; // truncation: toward zero
    let hi = lo + 1; // next magnitude up, same sign
    let f_lo = f32::from_bits((lo as u32) << 16) as f64;
    let hi_bits = (hi as u32) << 16;
    let d_lo = (x - f_lo).abs();
    let d_hi = if (hi_bits & 0x7F80_0000) == 0x7F80_0000 {
        // `hi` is infinity: the rounding midpoint to infinity is 2^128 - 2^119.
        2f64.powi(128) - x.abs()
    } else {
        (f32::from_bits(hi_bits) as f64 - x).abs()
    };
    if d_lo < d_hi || (d_lo == d_hi && lo & 1 == 0) {
        lo
    } else {
        hi
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    use ndarray::simd::{f32_to_bf16_batch_rne, f32_to_bf16_scalar_rne};
    use std::time::Instant;

    let threads: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    assert!(threads.is_power_of_two() && threads <= 64, "threads must be a power of two <= 64");
    let path = if is_x86_feature_detected!("avx512f") {
        "AVX-512F"
    } else {
        "scalar fallback"
    };
    println!("bf16_rne_exhaustive: batch path = {path}, threads = {threads}, chunk = 65536");

    let t0 = Instant::now();
    let span = (1u64 << 32) / threads;
    let results: Vec<(u64, u64, u64, u64, Option<u32>)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    const CHUNK: usize = 1 << 16;
                    let mut inp = vec![0f32; CHUNK];
                    let mut out = vec![0u16; CHUNK];
                    let (mut n, mut vs_scalar, mut vs_oracle, mut sum) = (0u64, 0u64, 0u64, 0u64);
                    let mut first = None;
                    let mut base = t * span;
                    while base < (t + 1) * span {
                        for (i, x) in inp.iter_mut().enumerate() {
                            *x = f32::from_bits((base + i as u64) as u32);
                        }
                        f32_to_bf16_batch_rne(&inp, &mut out);
                        for i in 0..CHUNK {
                            let b = (base + i as u64) as u32;
                            let got = out[i];
                            // Order-independent, so the value is the same for any thread count.
                            sum = sum.wrapping_add((got as u64).wrapping_mul(b as u64 | 1));
                            if got != f32_to_bf16_scalar_rne(inp[i]) {
                                vs_scalar += 1;
                            }
                            if got != oracle_bf16(b) {
                                vs_oracle += 1;
                                first.get_or_insert(b);
                            }
                        }
                        n += CHUNK as u64;
                        base += CHUNK as u64;
                    }
                    (n, vs_scalar, vs_oracle, sum, first)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let n: u64 = results.iter().map(|r| r.0).sum();
    let vs_scalar: u64 = results.iter().map(|r| r.1).sum();
    let vs_oracle: u64 = results.iter().map(|r| r.2).sum();
    let checksum = results.iter().fold(0u64, |a, r| a.wrapping_add(r.3));
    println!("inputs checked     : {n}");
    println!("mismatch vs scalar : {vs_scalar}");
    match results.iter().find_map(|r| r.4) {
        Some(b) => println!("mismatch vs oracle : {vs_oracle} (first input 0x{b:08x})"),
        None => println!("mismatch vs oracle : {vs_oracle}"),
    }
    println!("output checksum    : 0x{checksum:016x}");
    println!("elapsed            : {:.1} s", t0.elapsed().as_secs_f64());
    assert_eq!(n, 1u64 << 32, "did not cover the full input space");
    if vs_scalar != 0 || vs_oracle != 0 {
        std::process::exit(1);
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("bf16_rne_exhaustive: x86_64 only (the batch path under test is AVX-512F)");
}
