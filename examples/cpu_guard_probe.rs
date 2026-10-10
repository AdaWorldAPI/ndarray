//! Probe for `ndarray::cpu_guard`: does SIMD work, so a build for a newer CPU
//! than the one running it would otherwise die with SIGILL.
//!
//! Falsifier (needs `qemu-user-static`):
//! ```sh
//! env -u RUSTFLAGS cargo --config .cargo/config-v4.toml build --release --example cpu_guard_probe
//! qemu-x86_64-static -cpu Haswell target/release/examples/cpu_guard_probe
//! ```
//! Expected: an `error: this binary was compiled for a CPU with [...]` line and
//! exit status 132, not `Illegal instruction`.
use ndarray::simd::F32x16;

fn main() {
    let x: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let mut acc = F32x16::splat(0.0);
    let (chunks, _) = x.as_chunks::<16>();
    for chunk in chunks {
        acc += F32x16::from_slice(chunk);
    }
    println!("sum = {}", acc.reduce_sum());
}
