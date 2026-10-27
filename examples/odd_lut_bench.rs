//! Throughput of `odd_lut_u8` against plain table loops.
//!
//! Three paths over the same 1 Mi random magnitudes and sign plane:
//!
//! * `i8 table`: the i8 `Sym` form, one 255-entry table indexed by the
//!   signed code (`table[c + 127]`), no sign plane;
//! * `u8 table + sign`: the scalar form of what `odd_lut_u8` computes, a
//!   256-entry lookup plus the sign plane cleared where the result is 0;
//! * `odd_lut_u8`: the SIMD primitive.
//!
//! The scalar and SIMD sign-magnitude outputs are compared before timing.
//! Reports the best of several runs in ns per element and the CPU features
//! the binary was built for.

use std::hint::black_box;
use std::time::Instant;

use ndarray::simd::odd_lut_u8;

const N: usize = 1 << 20;
const RUNS: usize = 30;

fn best<F: FnMut()>(mut f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..RUNS {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best * 1e9 / N as f64
}

fn main() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mags: Vec<u8> = (0..N).map(|_| next() as u8).collect();
    let signs: Vec<u64> = (0..N / 64).map(|_| next()).collect();
    let codes: Vec<i8> = (0..N)
        .map(|_| ((next() % 255) as i16 - 127) as i8)
        .collect();

    let r = 4.0f64;
    let mut table = [0u8; 256];
    for (m, t) in table.iter_mut().enumerate() {
        *t = ((m as f64 * r / 255.0).tanh() * 255.0).round() as u8;
    }
    let mut table_i8 = [0i8; 255];
    for (k, t) in table_i8.iter_mut().enumerate() {
        *t = (((k as f64 - 127.0) * r / 127.0).tanh() * 127.0).round() as i8;
    }

    // Correctness first: the SIMD output must equal the scalar form.
    let mut out_simd = vec![0u8; N];
    let mut signs_simd = vec![0u64; N / 64];
    odd_lut_u8(&mags, &signs, &table, &mut out_simd, &mut signs_simd);
    let mut out_scalar = vec![0u8; N];
    let mut signs_scalar = vec![0u64; N / 64];
    let scalar = |out: &mut [u8], out_signs: &mut [u64]| {
        let rows = mags.as_chunks::<64>().0.iter();
        for (w, (chunk, o)) in rows.zip(out.as_chunks_mut::<64>().0).enumerate() {
            let mut nonzero = 0u64;
            for (k, (&m, d)) in chunk.iter().zip(o.iter_mut()).enumerate() {
                *d = table[m as usize];
                nonzero |= u64::from(*d != 0) << k;
            }
            out_signs[w] = signs[w] & nonzero;
        }
    };
    scalar(&mut out_scalar, &mut signs_scalar);
    assert_eq!(out_simd, out_scalar, "magnitudes differ");
    assert_eq!(signs_simd, signs_scalar, "sign planes differ");

    let mut out_i8 = vec![0i8; N];
    let t_i8 = best(|| {
        for (d, &c) in out_i8.iter_mut().zip(black_box(&codes)) {
            *d = table_i8[(i32::from(c) + 127) as usize];
        }
        black_box(&out_i8);
    });
    let t_scalar = best(|| {
        scalar(&mut out_scalar, &mut signs_scalar);
        black_box((&out_scalar, &signs_scalar));
    });
    let t_simd = best(|| {
        odd_lut_u8(black_box(&mags), black_box(&signs), &table, &mut out_simd, &mut signs_simd);
        black_box((&out_simd, &signs_simd));
    });

    println!(
        "odd_lut_bench: avx512f={} avx512vbmi={} avx2={} neon={}  n={N}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx512vbmi"),
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "neon"),
    );
    println!("  i8 table (no sign plane)   {t_i8:6.3} ns/elem");
    println!("  u8 table + sign, scalar    {t_scalar:6.3} ns/elem");
    println!("  odd_lut_u8                 {t_simd:6.3} ns/elem  ({:.1}x scalar)", t_scalar / t_simd);
}
