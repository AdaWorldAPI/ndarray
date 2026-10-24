//! Exhaustive accuracy check of `ndarray::hpc::vml::vstanh` over every f32.
//!
//! Every non-negative bit pattern `p` in `0..=0x7fff_ffff` is evaluated
//! together with its negation, so the run covers all 2^32 inputs. The
//! reference is `tanh` in `f64` rounded once to `f32`. Inputs where that
//! single rounding could land on the wrong side of an f32 midpoint are
//! written to a file for an independent high-precision re-check.
//!
//! A second arm runs the previous kernel, `2·sigmoid(2x) − 1`, rebuilt from
//! the public `F32x16` and `simd_exp_f32`, as the baseline.
//!
//! ```text
//! cargo run --release --example vstanh_exhaustive [flip_risk.txt] [worst.txt]
//! ```

use std::fmt::Write as _;
use std::time::Instant;

use ndarray::hpc::vml::vstanh;
use ndarray::simd::{simd_exp_f32, F32x16};
use ndarray::{Array1, ArrayView1, ArrayViewMut1};

/// Signed ordering key: -0.0 and +0.0 one apart, monotone in value.
fn key(x: f32) -> i64 {
    let i = x.to_bits() as i32;
    if i < 0 {
        -((i & i32::MAX) as i64) - 1
    } else {
        i as i64
    }
}

fn ulp(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() && b.is_nan() { 0 } else { u64::MAX };
    }
    (key(a) - key(b)).unsigned_abs()
}

const BANDS: [(&str, f32, f32); 11] = [
    ("zero", 0.0, 0.0),
    ("subnormal", f32::from_bits(1), f32::MIN_POSITIVE),
    ("min_pos..1e-10", f32::MIN_POSITIVE, 1e-10),
    ("1e-10..1e-4", 1e-10, 1e-4),
    ("1e-4..0.1", 1e-4, 0.1),
    ("0.1..0.625", 0.1, 0.625),
    ("0.625..1", 0.625, 1.0),
    ("1..4", 1.0, 4.0),
    ("4..10", 4.0, 10.0),
    ("10..MAX", 10.0, f32::MAX),
    ("inf", f32::INFINITY, f32::INFINITY),
];

fn band_of(a: f32) -> usize {
    if a == 0.0 {
        0
    } else if a.is_infinite() {
        10
    } else if a < f32::MIN_POSITIVE {
        1
    } else {
        (2..10).find(|&b| a < BANDS[b].2 || b == 9).unwrap()
    }
}

#[derive(Clone, Copy, Default)]
struct Band {
    n: u64,
    max_ulp: u64,
    worst_x: f32,
    worst_got: f32,
    worst_want: f32,
    max_abs: f64,
    max_rel: f64,
    zeroed: u64,
    hist: [u64; 4], // 0, 1, 2, >2 ulp
}

#[derive(Clone, Default)]
struct Stats {
    bands: [Band; 11],
    nan_ok: u64,
    nan_bad: u64,
    odd_bad: u64,
    bound_bad: u64,
    mono_bad: u64,
    flip_risk: Vec<f32>,
}

fn old_kernel(x: &[f32], out: &mut [f32]) {
    for (xc, oc) in x
        .as_chunks::<16>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<16>().0.iter_mut())
    {
        let v = F32x16::from_slice(xc);
        let two = F32x16::splat(2.0);
        let e = simd_exp_f32(-(v * two));
        let s = F32x16::splat(1.0) / (e + F32x16::splat(1.0));
        s.mul_add(two, F32x16::splat(-1.0)).copy_to_slice(oc);
    }
}

fn run(lo: u32, hi: u32, use_old: bool) -> Stats {
    const CH: usize = 1 << 16;
    let mut st = Stats::default();
    let mut xs = vec![0.0f32; CH];
    let mut nxs = vec![0.0f32; CH];
    let mut ys = vec![0.0f32; CH];
    let mut nys = vec![0.0f32; CH];
    let mut prev: Option<f32> = None;
    let mut p = lo as u64;
    while p < hi as u64 {
        let m = (CH as u64).min(hi as u64 - p) as usize;
        for i in 0..m {
            xs[i] = f32::from_bits((p + i as u64) as u32);
            nxs[i] = -xs[i];
        }
        if use_old {
            for i in m..CH {
                xs[i] = 0.0;
                nxs[i] = 0.0;
            }
            old_kernel(&xs, &mut ys);
            old_kernel(&nxs, &mut nys);
        } else {
            vstanh(ArrayView1::from(&xs[..m]), ArrayViewMut1::from(&mut ys[..m]));
            vstanh(ArrayView1::from(&nxs[..m]), ArrayViewMut1::from(&mut nys[..m]));
        }
        for i in 0..m {
            let x = xs[i];
            if x.is_nan() {
                if ys[i].is_nan() && nys[i].is_nan() {
                    st.nan_ok += 2;
                } else {
                    st.nan_bad += 2;
                }
                prev = None;
                continue;
            }
            // odd symmetry, bit for bit
            if nys[i].to_bits() != (ys[i].to_bits() ^ 0x8000_0000) {
                st.odd_bad += 1;
            }
            if !(-1.0..=1.0).contains(&ys[i]) || !(-1.0..=1.0).contains(&nys[i]) {
                st.bound_bad += 1;
            }
            if let Some(q) = prev {
                if ys[i] < q {
                    st.mono_bad += 1;
                }
            }
            prev = Some(ys[i]);
            let y64 = (x as f64).tanh();
            let want = y64 as f32;
            // Flip risk: y64 within 4 f64-ulp of the midpoint between `want`
            // and its neighbour away from y64 rounding.
            if want.is_finite() && want != 0.0 && x.is_finite() {
                let w = want as f64;
                let nb = if y64 >= w {
                    f32::from_bits(want.to_bits() + 1)
                } else {
                    f32::from_bits(want.to_bits() - 1)
                } as f64;
                let mid = 0.5 * (w + nb);
                if (y64 - mid).abs() <= 4.0 * f64::EPSILON * y64.abs() && st.flip_risk.len() < 200_000 {
                    st.flip_risk.push(x);
                }
            }
            let b = band_of(x);
            for (xx, got, want, y64) in [(x, ys[i], want, y64), (-x, nys[i], -want, -y64)] {
                let bd = &mut st.bands[b];
                bd.n += 1;
                let u = ulp(got, want);
                bd.hist[u.min(3) as usize] += 1;
                if u > bd.max_ulp || bd.n == 1 {
                    bd.max_ulp = u;
                    (bd.worst_x, bd.worst_got, bd.worst_want) = (xx, got, want);
                }
                let abs = (got as f64 - y64).abs();
                bd.max_abs = bd.max_abs.max(abs);
                if y64 != 0.0 {
                    bd.max_rel = bd.max_rel.max(abs / y64.abs());
                }
                if got == 0.0 && xx != 0.0 {
                    bd.zeroed += 1;
                }
            }
        }
        p += m as u64;
    }
    st
}

fn merge(mut a: Stats, b: Stats) -> Stats {
    for (x, y) in a.bands.iter_mut().zip(b.bands) {
        let x_empty = x.n == 0;
        x.n += y.n;
        if y.n > 0 && (x_empty || y.max_ulp > x.max_ulp) {
            (x.max_ulp, x.worst_x, x.worst_got, x.worst_want) = (y.max_ulp, y.worst_x, y.worst_got, y.worst_want);
        }
        x.max_abs = x.max_abs.max(y.max_abs);
        x.max_rel = x.max_rel.max(y.max_rel);
        x.zeroed += y.zeroed;
        for k in 0..4 {
            x.hist[k] += y.hist[k];
        }
    }
    a.nan_ok += b.nan_ok;
    a.nan_bad += b.nan_bad;
    a.odd_bad += b.odd_bad;
    a.bound_bad += b.bound_bad;
    a.mono_bad += b.mono_bad;
    a.flip_risk.extend(b.flip_risk);
    a
}

fn sweep(use_old: bool, threads: u32) -> Stats {
    let end: u64 = 0x8000_0000;
    let step = end / threads as u64;
    let parts: Vec<Stats> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads as u64)
            .map(|t| {
                let lo = (t * step) as u32;
                let hi = if t + 1 == threads as u64 { end } else { (t + 1) * step };
                s.spawn(move || run(lo, hi as u32, use_old))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // Monotonicity across thread boundaries.
    let mut st = parts.into_iter().reduce(merge).unwrap();
    for t in 1..threads as u64 {
        let b = (t * step) as u32;
        let (a, c) = (f32::from_bits(b - 1), f32::from_bits(b));
        let mut y = [0.0f32; 2];
        vstanh(ArrayView1::from(&[a, c][..]), ArrayViewMut1::from(&mut y[..]));
        if !use_old && y[1] < y[0] {
            st.mono_bad += 1;
        }
    }
    st
}

fn report(name: &str, st: &Stats) -> String {
    let mut s = String::new();
    writeln!(s, "== {name} ==").unwrap();
    writeln!(
        s,
        "{:<16} {:>12} {:>10} {:>11} {:>11} {:>9}  {:>10} {:>10} {:>8} {:>8}  worst x -> got / want",
        "band", "n", "max_ulp", "max_abs", "max_rel", "zeroed", "0ulp", "1ulp", "2ulp", ">2ulp"
    )
    .unwrap();
    for (b, (nm, _, _)) in st.bands.iter().zip(BANDS) {
        writeln!(
            s,
            "{:<16} {:>12} {:>10} {:>11.3e} {:>11.3e} {:>9}  {:>10} {:>10} {:>8} {:>8}  {:e} -> {:e} ({:08x}) / {:e} ({:08x})",
            nm, b.n, b.max_ulp, b.max_abs, b.max_rel, b.zeroed, b.hist[0], b.hist[1], b.hist[2], b.hist[3],
            b.worst_x, b.worst_got, b.worst_got.to_bits(), b.worst_want, b.worst_want.to_bits()
        )
        .unwrap();
    }
    writeln!(s, "NaN inputs: {} kept NaN, {} not", st.nan_ok, st.nan_bad).unwrap();
    writeln!(s, "odd-symmetry violations (bitwise): {}", st.odd_bad).unwrap();
    writeln!(s, "outside [-1, 1]: {}", st.bound_bad).unwrap();
    writeln!(s, "monotonicity violations (x >= 0, ordered): {}", st.mono_bad).unwrap();
    writeln!(s, "reference flip-risk inputs (|x| only): {}", st.flip_risk.len()).unwrap();
    s
}

fn bench() -> String {
    let n = 1 << 20;
    let x: Array1<f32> = (0..n).map(|i| (i as f32 / n as f32) * 8.0 - 4.0).collect();
    let mut out = Array1::<f32>::zeros(n);
    let reps = 50;
    let t = Instant::now();
    for _ in 0..reps {
        vstanh(x.view(), out.view_mut());
    }
    let new_ns = t.elapsed().as_nanos() as f64 / (reps * n) as f64;
    let xs = x.to_vec();
    let mut o = vec![0.0f32; n];
    let t = Instant::now();
    for _ in 0..reps {
        old_kernel(&xs, &mut o);
    }
    let old_ns = t.elapsed().as_nanos() as f64 / (reps * n) as f64;
    let t = Instant::now();
    for _ in 0..reps {
        for (a, b) in xs.iter().zip(o.iter_mut()) {
            *b = a.tanh();
        }
    }
    let scalar_ns = t.elapsed().as_nanos() as f64 / (reps * n) as f64;
    std::hint::black_box((&out, &o));
    format!(
        "throughput over 2^20 f32 in [-4, 4], {reps} reps: vstanh {new_ns:.3} ns/elem, old kernel {old_ns:.3} ns/elem, scalar f32::tanh {scalar_ns:.3} ns/elem\n"
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let threads = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(4);
    println!(
        "target: avx512f={} avx2={} neon={}  threads={threads}",
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "avx2"),
        cfg!(target_arch = "aarch64")
    );
    let t = Instant::now();
    let new = sweep(false, threads);
    print!("{}", report("vstanh (repaired), all 2^32 f32", &new));
    println!("sweep time {:.1} s", t.elapsed().as_secs_f64());
    let old = sweep(true, threads);
    print!("{}", report("previous kernel 2*sigmoid(2x)-1, all 2^32 f32", &old));
    print!("{}", bench());
    if let Some(path) = args.get(1) {
        let s: String = new
            .flip_risk
            .iter()
            .map(|x| format!("{:08x}\n", x.to_bits()))
            .collect();
        std::fs::write(path, s).unwrap();
    }
    if let Some(path) = args.get(2) {
        let s: String = new
            .bands
            .iter()
            .filter(|b| b.n > 0)
            .map(|b| format!("{:08x}\n", b.worst_x.to_bits()))
            .collect();
        std::fs::write(path, s).unwrap();
    }
}
