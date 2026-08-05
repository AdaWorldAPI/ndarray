# ndarray — Railway compile-test image (AVX2 default)
# Verifies the HPC module builds cleanly (default + jit-native features)
# Requires Rust 1.97.1 (LazyLock, simd_caps, modern std APIs)
#
# CPU detection & SIMD dispatch documentation: see Dockerfile.md
# AVX-512 pinned variant: see Dockerfile.avx512
#
# Build: docker build -t ndarray-test .
# Run:   docker run --rm ndarray-test

FROM debian:bookworm-slim AS builder

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gcc libc6-dev pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust 1.97.1 via rustup — MUST match rust-toolchain.toml (channel =
# "1.97.1") and Cargo.toml's `rust-version = "1.97"`. rust-toolchain.toml is
# deliberately NOT copied into the image (rustup would try to download a second
# toolchain at build time), so this pin is the only thing keeping the image in
# step with the repo — bump it whenever rust-toolchain.toml moves.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain 1.97.1 --profile minimal \
    && rustc --version | grep -q "1.97.1"

WORKDIR /app

# Copy workspace files first for layer caching
COPY Cargo.toml Cargo.lock ./
COPY ndarray-rand/Cargo.toml ndarray-rand/Cargo.toml
COPY crates/ crates/

# The root Cargo.toml has `[patch.crates-io] chacha20 = { path = "vendor/chacha20" }`.
# Cargo resolves patch entries while LOADING THE MANIFEST — before features, before
# targets, on every command — so without this the build dies at parse with
# "failed to load source for dependency `chacha20` / unable to update
# /app/vendor/chacha20". Same class as the examples/benches note below, and the
# reason a selective-COPY Dockerfile has to be updated whenever a path source is
# added to the manifest.
COPY vendor/ vendor/

# Copy source
COPY src/ src/
COPY ndarray-rand/src/ ndarray-rand/src/

# Cargo.toml (root) and ndarray-rand/Cargo.toml (a workspace member) declare
# explicit [[example]]/[[bench]] targets. Cargo validates that every declared
# target's source file exists while parsing the manifest — even for a lib-only
# build — so these dirs must be in the context or `cargo build` fails at parse
# with "can't find <name> example/bench". They are NOT compiled here (the default
# build skips examples/benches), so this only adds source bytes, not build time.
COPY examples/ examples/
COPY benches/ benches/
COPY ndarray-rand/benches/ ndarray-rand/benches/

# Default target: x86-64-v3 (AVX2) — runs on GitHub CI and most servers.
# Use Dockerfile.avx512 for x86-64-v4 (AVX-512). ndarray's simd.rs polyfill
# detects AVX-512 at runtime via LazyLock<Tier> even when compiled for v3;
# compile-time v3 just means the scalar/AVX2 fallback paths are used when the
# runtime check fails. Both paths produce identical results.
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"

# Build default features
RUN cargo build --release 2>&1 && echo "=== DEFAULT BUILD OK ==="

# Build with JIT
RUN cargo build --release --features jit-native 2>&1 && echo "=== JIT-NATIVE BUILD OK ==="

# Run tests
RUN cargo test --release --lib -- hpc:: 2>&1 && echo "=== HPC TESTS OK ==="

# Minimal runtime image — just proves it compiled
FROM debian:bookworm-slim
COPY --from=builder /app/target/release/libndarray.rlib /usr/local/lib/
CMD ["echo", "ndarray build verified — Rust 1.97.1"]
