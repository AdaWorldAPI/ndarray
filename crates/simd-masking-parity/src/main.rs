//! Native / qemu entry: run the shared masking parity program and exit
//! non-zero on the first mismatch. The cfg lines it prints are a LOG of which
//! realization this binary was built as — the checks themselves never branch
//! on them (the whole point is that the same program runs under every arm).

fn main() {
    println!(
        "simd-masking-parity: arch={} avx2={} avx512f={} neon={} simd128={} nightly-simd={}",
        std::env::consts::ARCH,
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "avx512f"),
        cfg!(target_feature = "neon"),
        cfg!(target_feature = "simd128"),
        cfg!(feature = "nightly-simd"),
    );
    let rc = simd_masking_parity::run();
    if rc != 0 {
        eprintln!("simd-masking-parity FAILED: code = {rc:#x} (see src/lib.rs)");
        std::process::exit(1);
    }
    println!(
        "simd-masking-parity OK: {} checks bit-identical to the bit-serial reference",
        simd_masking_parity::CHECKS
    );
}
