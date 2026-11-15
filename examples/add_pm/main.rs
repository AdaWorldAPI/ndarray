//! D-ADD-PM: cost of adding `P × M` lanes in place (`add_pm`) against
//! decode → i32 → encode (`add_decoded`), per 64 lanes. Correctness lives in
//! `tests/add_pm.rs`; this only times the two paths on the host tier.
//!
//!   cargo run --release --example add_pm --features std
mod pm;

use std::hint::black_box;
use std::time::Instant;

use ndarray::simd::U8x64;
use pm::{add_decoded, add_pm, Pm};

fn blocks(n: usize, seed: u64) -> Vec<Pm> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let mut mag = [0u8; 64];
            for m in &mut mag {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                *m = (s % 160) as u8;
            }
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            Pm { mag: U8x64::from_array(mag), pos: s }
        })
        .collect()
}

fn time(name: &str, a: &[Pm], b: &[Pm], f: fn(Pm, Pm) -> (Pm, u64)) {
    let reps = 50;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        let mut acc = 0u64;
        for (x, y) in a.iter().zip(b) {
            let (s, c) = f(black_box(*x), black_box(*y));
            acc ^= s.pos ^ c;
            black_box(s.mag);
        }
        black_box(acc);
        best = best.min(t.elapsed().as_nanos() as f64 / a.len() as f64);
    }
    println!("{name}: {best:.2} ns per 64 lanes (best of {reps})");
}

fn main() {
    assert!(!cfg!(debug_assertions), "release build only");
    #[cfg(target_arch = "x86_64")]
    println!(
        "# avx2={} avx512f={} avx512bw={} (runtime); built avx512bw={}",
        std::arch::is_x86_feature_detected!("avx2"),
        std::arch::is_x86_feature_detected!("avx512f"),
        std::arch::is_x86_feature_detected!("avx512bw"),
        cfg!(target_feature = "avx512bw")
    );
    let n = 4096; // 4096 blocks = 256 KiB of magnitudes per operand: L2-resident
    let (a, b) = (blocks(n, 0x1234_5678), blocks(n, 0x8765_4321));
    for _ in 0..2 {
        time("add_pm (in place)", &a, &b, add_pm);
        time("add_decoded (decode -> i32 -> encode)", &a, &b, add_decoded);
    }
}
