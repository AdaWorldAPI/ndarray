//! Float accumulation order across CPU tiers: prints an FNV-1a hash of the
//! raw bits of (a) `gemm_f32` and (b) `mean_axis` / `sum` on fixed inputs,
//! plus which matrixmultiply kernel the host can take. Run the same binary
//! (or the same source) on several tiers and compare the hashes: equal
//! hashes mean bit-identical results.
//!
//! ```sh
//! cargo run --release --example float_accum_probe
//! ```
use ndarray::{Array2, Axis};

fn fnv(bits: impl Iterator<Item = u32>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bits {
        for byte in b.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn main() {
    // Values with full mantissas, so any change of order or of fused
    // multiply-add shows up in the low bits.
    let f = |i: usize, j: usize| {
        (((i as u64 * 2_654_435_761 + j as u64 * 40_503) % 100_003) as f32 / 100_003.0 - 0.5) * 3.7
    };
    let (m, k, n) = (64, 1024, 48);
    let a = Array2::from_shape_fn((m, k), |(i, j)| f(i, j));
    let b = Array2::from_shape_fn((k, n), |(i, j)| f(j + 7, i));
    let mut c = vec![0.0f32; m * n];
    ndarray::backend::native::gemm_f32(
        m,
        n,
        k,
        1.0,
        a.as_slice().unwrap(),
        k,
        b.as_slice().unwrap(),
        n,
        0.0,
        &mut c,
        n,
    );
    let mean = a.mean_axis(Axis(1)).unwrap();
    let sum: f32 = a.sum();
    let arch = if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "x86_64"
    };
    #[cfg(target_arch = "x86_64")]
    let tier = format!(
        "avx512f={} fma={} avx2={} avx={}",
        is_x86_feature_detected!("avx512f"),
        is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("avx")
    );
    #[cfg(not(target_arch = "x86_64"))]
    let tier = String::from("neon");
    println!("arch={arch} {tier}");
    println!("gemm_f32  [{m}x{k}]·[{k}x{n}] hash={:016x}", fnv(c.iter().map(|v| v.to_bits())));
    println!("mean_axis [{m}x{k}] axis 1   hash={:016x}", fnv(mean.iter().map(|v| v.to_bits())));
    println!("sum       [{m}x{k}]          bits={:08x}", sum.to_bits());
}
