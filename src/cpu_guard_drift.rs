// Shared by `build.rs` and the tests in `cpu_guard.rs` (both `include!` it);
// not a module of the crate.
//
// Which target features a build enables that `cpu_guard` cannot check. The
// guard asks `cfg!(target_feature = ...)` row by row, so a feature that is
// compiled in but has no row is never compared with the running CPU: the
// binary passes the guard and can still die with SIGILL. That happens when a
// toolchain bump lets LLVM enable a feature the table predates.

/// Features every x86_64 CPU has; the guard does not need rows for them.
const X86_64_BASELINE: &[&str] = &["sse", "sse2", "x87"];

/// Target features that do not describe the CPU: `crt-static` selects static
/// C-runtime linkage and is on by default for musl targets.
const NOT_CPU_FEATURES: &[&str] = &["crt-static"];

/// The feature names in the x86_64 table of `cpu_guard.rs` (`"name" =>` rows).
fn guard_table_names(cpu_guard_src: &str) -> Vec<&str> {
    cpu_guard_src
        .lines()
        .filter_map(|line| {
            let rest = line.trim_start().strip_prefix('"')?;
            let (name, tail) = rest.split_once('"')?;
            tail.trim_start().starts_with("=>").then_some(name)
        })
        .collect()
}

/// The enabled features (`CARGO_CFG_TARGET_FEATURE`, comma-separated) that
/// are neither in the guard's table, nor x86_64 baseline, nor a non-CPU
/// feature, in input order.
fn unchecked_features<'a>(enabled: &'a str, table: &[&str]) -> Vec<&'a str> {
    enabled
        .split(',')
        .filter(|f| {
            !f.is_empty() && !table.contains(f) && !X86_64_BASELINE.contains(f) && !NOT_CPU_FEATURES.contains(f)
        })
        .collect()
}
