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

- **Rust 1.94.0** is pinned via `rust-toolchain.toml`; rustup auto-selects it in `/workspace`.
- **No AVX-512 hardware** in Cloud Agent VMs — SIMD kernel tests using `#[target_feature(enable = "avx512f")]` are compile-gated and will be skipped at runtime. This is expected behavior.
- **Feature gates**: `intel-mkl` and `openblas` are mutually exclusive and require system libraries not installed by default. The default build uses `native` (pure Rust SIMD) which needs no extra libs.
- **Build time**: ~18s cold, <1s incremental. Tests (~1819) take ~70s.
- The workspace has sub-crates under `crates/` and `ndarray-rand/`. Default members exclude `blas-tests` and `blas-mock-tests` (they activate the `blas` feature which needs cblas-sys linking).
- `libssl-dev` is needed as a build dependency for some transitive crates.
