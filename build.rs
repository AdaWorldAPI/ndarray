// Warns when a build enables x86_64 target features that `cpu_guard` has no
// row for. Those features are compiled in but never compared with the running
// CPU, so the guard would pass a binary that can still die with SIGILL.
//
// Limits: on stable rustc, `CARGO_CFG_TARGET_FEATURE` omits unstable features
// (avx10.x, amx-*), exactly as `cfg!` in the guard does, so they cannot be
// reported here. Cargo also hides warnings from git and registry dependencies,
// so this fires when building ndarray itself, not inside a consumer.

include!("src/cpu_guard_drift.rs");

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cpu_guard.rs");
    println!("cargo:rerun-if-changed=src/cpu_guard_drift.rs");

    if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64") {
        return;
    }
    let enabled = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let guard_src = match std::fs::read_to_string("src/cpu_guard.rs") {
        Ok(src) => src,
        Err(e) => {
            println!("cargo:warning=cpu_guard drift check skipped: cannot read src/cpu_guard.rs: {e}");
            return;
        }
    };
    let table = guard_table_names(&guard_src);
    let unchecked = unchecked_features(&enabled, &table);
    if !unchecked.is_empty() {
        println!(
            "cargo:warning=cpu_guard has no row for enabled target features: {} \
             (the guard cannot detect a CPU that lacks them; add rows in src/cpu_guard.rs)",
            unchecked.join(", ")
        );
    }
}
