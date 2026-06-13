//! AMX / VNNI int8-GEMM throughput probe.
//!
//! Measures the dispatched `matmul_i8_to_i32` (AMX `TDPBUSD` tile → AVX-512
//! `VPDPBUSD` zmm → AVX-VNNI ymm → scalar, chosen at runtime) against a naive
//! scalar i8×i8→i32 reference, across square sizes. Reports ns, GMAC/s, GOP/s.
//!
//!   RUSTFLAGS="-C target-cpu=native"  cargo run --release --example amx_gemm_bench
//!
//! `target-cpu=native` (or `x86-64-v4` + an AMX host) is what enables the AMX
//! tile path; `amx_available()` reports whether THIS silicon exposes it.

use std::time::Instant;

use ndarray::simd::{amx_available, matmul_i8_to_i32};
use ndarray::{ArrayView2, ArrayViewMut2};

fn scalar_i8_gemm(a: &[i8], b: &[i8], c: &mut [i32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for p in 0..k {
                acc += a[i * k + p] as i32 * b[p * n + j] as i32;
            }
            c[i * n + j] = acc;
        }
    }
}

fn bench(m: usize, k: usize, n: usize, reps: usize, scalar_reps: usize) {
    let a: Vec<i8> = (0..m * k).map(|i| ((i % 7) as i8) - 3).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i % 5) as i8) - 2).collect();
    let mut c = vec![0i32; m * n];
    let av = ArrayView2::from_shape((m, k), &a[..]).unwrap();
    let bv = ArrayView2::from_shape((k, n), &b[..]).unwrap();

    // warmup + dispatched timing
    matmul_i8_to_i32(av, bv, ArrayViewMut2::from_shape((m, n), &mut c[..]).unwrap()).unwrap();
    let dispatched = c.clone();
    let t = Instant::now();
    for _ in 0..reps {
        matmul_i8_to_i32(av, bv, ArrayViewMut2::from_shape((m, n), &mut c[..]).unwrap()).unwrap();
    }
    let ns = t.elapsed().as_nanos() as f64 / reps as f64;

    // scalar reference (fewer reps; it's slow) + correctness check
    let mut cs = vec![0i32; m * n];
    let t = Instant::now();
    for _ in 0..scalar_reps {
        scalar_i8_gemm(&a, &b, &mut cs, m, k, n);
    }
    let sns = t.elapsed().as_nanos() as f64 / scalar_reps as f64;
    let ok = dispatched == cs;

    let macs = (m * n * k) as f64;
    println!(
        "  {m:>4}x{k:>4}x{n:>4}:  dispatched {ns:>11.0} ns  {:>7.1} GMAC/s ({:>7.1} GOP/s)   scalar {sns:>13.0} ns   speedup {:>6.1}x   correct={ok}",
        macs / ns,
        2.0 * macs / ns,
        sns / ns,
    );
}

fn main() {
    println!("== int8 GEMM (matmul_i8_to_i32: AMX TDPBUSD -> VPDPBUSD-zmm -> AVX-VNNI -> scalar) ==");
    println!("amx_available() on this host: {}", amx_available());
    println!();
    bench(256, 256, 256, 300, 5);
    bench(512, 512, 512, 80, 2);
    bench(1024, 1024, 1024, 20, 1);
    bench(2048, 2048, 2048, 6, 1);
}
