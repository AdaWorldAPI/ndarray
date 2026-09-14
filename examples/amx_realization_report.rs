//! Prints what the AMX path can do on THIS host — for the SIMD realization
//! matrix's `native` row, so a run where every AMX test early-returned on
//! `!amx_available()` is visibly a skip, never a pass (AMX Gotcha 9).
//!
//! AMX is runtime-gated (raw-byte / mnemonic `asm!` + `amx_available()`), so a
//! `-Ctarget-cpu=native` build always COMPILES it; whether it EXECUTED is what
//! this line records in the job log. Exits 0 either way: absence of silicon is
//! a fact about the runner, not a defect.
fn main() {
    #[cfg(target_arch = "x86_64")]
    {
        use ndarray::simd::{amx_available, amx_report, cpu_model};
        println!("{}", amx_report());
        println!(
            "amx_realization: cpu_model={:?} has_amx={} amx_available={} -> AMX tests {}",
            cpu_model(),
            cpu_model().has_amx(),
            amx_available(),
            if amx_available() { "EXECUTED" } else { "SKIPPED (no AMX on this host)" }
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    println!("amx_realization: not x86_64 — AMX path not compiled on this target");
}
