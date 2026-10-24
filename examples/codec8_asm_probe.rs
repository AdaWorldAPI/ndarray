//! Probe for plan `.claude/plans/quantized-codec8-v2.md`, gates G5 and G6.
//!
//! Do the compile-time parameters of `hpc::codec8::Codec8` cost anything compared with a
//! hand-written kernel of the same behaviour?
//!
//! - G5 (assembly): build with `--emit asm` and compare each `g_*` symbol with its `d_*`
//!   twin after normalising local labels (`scripts/codec8_asm_diff.sh` does this).
//! - G6 (timing): run the binary; it checks bit identity, then prints ns/elem for each pair
//!   as the median of 15 runs. Set `SWAP=1` to time the direct kernels first.
//!
//! The binary prints the SIMD tier it was built for. Pin the tier with
//! `env -u RUSTFLAGS cargo --config .cargo/config-v3.toml run --release --example codec8_asm_probe`
//! (or `config-v4.toml`).

use ndarray::hpc::codec8::{Calibration, Codec8, Codec8Error, NaNAtMin, OnnxQ, Sym, TfliteAct, TiesEven, I8};
use std::hint::black_box;
use std::time::Instant;

struct Q;
impl Calibration for Q {
    const SCALE: f32 = 1.0 / 63.5;
    const ZERO_POINT: i32 = 0;
}

type Lane = Codec8<I8, Sym, TiesEven, NaNAtMin>;

/// Generic: native lane, ties to even, NaN at -128.
#[no_mangle]
#[inline(never)]
pub fn g_lane_even(x: &[f32], o: &mut [i8]) -> Result<(), Codec8Error> {
    Lane::encode_checked::<Q>(x, o)
}

/// Hand-written twin of `g_lane_even`.
#[no_mangle]
#[inline(never)]
pub fn d_lane_even(x: &[f32], o: &mut [i8]) -> Result<(), Codec8Error> {
    if x.len() != o.len() {
        return Err(Codec8Error::LengthMismatch {
            src: x.len(),
            dst: o.len(),
        });
    }
    for (o, &v) in o.iter_mut().zip(x) {
        let q = (v / Q::SCALE).round_ties_even().clamp(-127.0, 127.0) as i32;
        *o = (if v.is_nan() { -128 } else { q }) as i8;
    }
    Ok(())
}

/// Generic: TFLite activation, ties away.
#[no_mangle]
#[inline(never)]
pub fn g_tflite_act(x: &[f32], o: &mut [i8]) -> Result<(), Codec8Error> {
    TfliteAct::encode_checked::<Q>(x, o)
}

/// Hand-written twin of `g_tflite_act`.
#[no_mangle]
#[inline(never)]
pub fn d_tflite_act(x: &[f32], o: &mut [i8]) -> Result<(), Codec8Error> {
    if x.len() != o.len() {
        return Err(Codec8Error::LengthMismatch {
            src: x.len(),
            dst: o.len(),
        });
    }
    if let Some(index) = x.iter().position(|v| v.is_nan()) {
        return Err(Codec8Error::NanInput { index });
    }
    for (o, &v) in o.iter_mut().zip(x) {
        let x = v / Q::SCALE;
        // The trunc form on wasm, as `TiesAway::round` does there.
        #[cfg(target_arch = "wasm32")]
        let r = {
            let t = x.trunc();
            if (x - t).abs() >= 0.5 {
                t + 1.0f32.copysign(x)
            } else {
                t
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        let r = x.round();
        *o = (r.clamp(-128.0, 127.0) as i32) as i8;
    }
    Ok(())
}

/// Generic: ONNX, ties to even (negative control for `d_tflite_act`: must differ).
#[no_mangle]
#[inline(never)]
pub fn g_onnx(x: &[f32], o: &mut [i8]) -> Result<(), Codec8Error> {
    OnnxQ::encode_checked::<Q>(x, o)
}

fn time<F: FnMut()>(n: usize, reps: usize, mut f: F) -> f64 {
    let mut v: Vec<f64> = (0..15)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..reps {
                f();
            }
            t.elapsed().as_nanos() as f64 / (reps * n) as f64
        })
        .collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v[7]
}

fn main() {
    println!("avx512f={} avx2={}", cfg!(target_feature = "avx512f"), cfg!(target_feature = "avx2"));
    for &n in &[4096usize, 1 << 22] {
        let reps = if n == 4096 { 20_000 } else { 20 };
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mk = |nan: bool, s: &mut u64| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    *s ^= *s << 13;
                    *s ^= *s >> 7;
                    *s ^= *s << 17;
                    if nan && i % 997 == 0 {
                        f32::NAN
                    } else {
                        ((*s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
                    }
                })
                .collect()
        };
        let with_nan = mk(true, &mut s);
        let finite = mk(false, &mut s);
        let (mut a, mut b) = (vec![0i8; n], vec![0i8; n]);

        g_lane_even(&with_nan, &mut a).unwrap();
        d_lane_even(&with_nan, &mut b).unwrap();
        assert_eq!(a, b);
        assert!(a.contains(&-128), "anti-vacuity: NaN codes present");
        g_tflite_act(&finite, &mut a).unwrap();
        d_tflite_act(&finite, &mut b).unwrap();
        assert_eq!(a, b);

        let row = |name: &str, t: f64| println!("n={n:>8} {name:<18} {t:.4} ns/elem");
        let run = |name: &str, f: &mut dyn FnMut()| row(name, time(n, reps, f));
        let mut ga = vec![0i8; n];
        let mut gb = vec![0i8; n];
        let swap = std::env::var("SWAP").is_ok();
        for pass in 0..2 {
            let generic_first = (pass == 0) != swap;
            if generic_first {
                run("generic lane_even", &mut || g_lane_even(black_box(&with_nan), black_box(&mut ga)).unwrap());
                run("direct  lane_even", &mut || d_lane_even(black_box(&with_nan), black_box(&mut gb)).unwrap());
            } else {
                run("direct  lane_even", &mut || d_lane_even(black_box(&with_nan), black_box(&mut gb)).unwrap());
                run("generic lane_even", &mut || g_lane_even(black_box(&with_nan), black_box(&mut ga)).unwrap());
            }
        }
        run("generic tflite_act", &mut || g_tflite_act(black_box(&finite), black_box(&mut ga)).unwrap());
        run("direct  tflite_act", &mut || d_tflite_act(black_box(&finite), black_box(&mut gb)).unwrap());
        run("generic onnx", &mut || g_onnx(black_box(&finite), black_box(&mut ga)).unwrap());
    }
}
