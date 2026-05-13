# AGENTS.md

## Cursor Cloud specific instructions

This is a Rust library crate (ndarray fork with HPC extensions). No external services (databases, APIs) are needed.

### Quick reference

| Action | Command |
|--------|---------|
| Build | `cargo build` |
| Lint | `cargo clippy -- -D warnings` |
| Test (lib) | `cargo test --lib -p ndarray` |
| Test (workspace) | `cargo test` |
| Test (HPC subset) | `cargo test --lib -p ndarray -- hpc::` |
| Run example | `cargo run --example life` |
| Format check | `cargo fmt -- --check` |

### Environment notes

- **Rust 1.94.1** is pinned via `rust-toolchain.toml`; rustup auto-selects it in `/workspace`.
- **No AVX-512 hardware** in Cloud Agent VMs — all test modules in `src/simd_avx512.rs` are gated with `#[cfg(all(test, target_feature = "avx512f"))]` and compile away entirely on non-AVX-512 targets. This is intentional: raw AVX-512 intrinsic tests must never run on CI/Cloud (x86-64-v3). The `simd.rs` LazyLock polyfill dispatches to `simd_avx2.rs` on these machines.
- **Feature gates**: `intel-mkl` and `openblas` are mutually exclusive and require system libraries not installed by default. The default build uses `native` (pure Rust SIMD) which needs no extra libs.
- **Build time**: ~18s cold, <1s incremental. Tests (~1776 on non-AVX-512) take ~70s.
- The workspace has sub-crates under `crates/` and `ndarray-rand/`. Default members exclude `blas-tests` and `blas-mock-tests` (they activate the `blas` feature which needs cblas-sys linking).
- `libssl-dev` is needed as a build dependency for some transitive crates.
- **`cargo fmt`**: `rustfmt.toml` uses 13+ nightly-only options (`brace_style`, `imports_granularity`, etc.). Stable rustfmt ignores them and reports massive diffs. This is a known pre-existing issue — do not attempt to fix formatting drift without coordinating with the project owner.
