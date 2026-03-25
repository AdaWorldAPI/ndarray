# ndarray — Railway compile-test image
# Verifies the HPC module builds cleanly (default + jit-native features)
# Requires Rust 1.94.0 (LazyLock, simd_caps, modern std APIs)
#
# Build: docker build -t ndarray-test .
# Run:   docker run --rm ndarray-test

FROM debian:bookworm-slim AS builder

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gcc libc6-dev pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust 1.94.0 via rustup
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain 1.94.0 --profile minimal \
    && rustc --version | grep -q "1.94.0"

WORKDIR /app

# Copy workspace files first for layer caching
COPY Cargo.toml Cargo.lock ./
COPY ndarray-rand/Cargo.toml ndarray-rand/Cargo.toml
COPY crates/ crates/

# Copy source
COPY src/ src/
COPY ndarray-rand/src/ ndarray-rand/src/

# Build default features
RUN cargo build --release 2>&1 && echo "=== DEFAULT BUILD OK ==="

# Build with JIT
RUN cargo build --release --features jit-native 2>&1 && echo "=== JIT-NATIVE BUILD OK ==="

# Run tests
RUN cargo test --release --lib -- hpc:: 2>&1 && echo "=== HPC TESTS OK ==="

# Minimal runtime image — just proves it compiled
FROM debian:bookworm-slim
COPY --from=builder /app/target/release/libndarray.rlib /usr/local/lib/
CMD ["echo", "ndarray build verified — Rust 1.94.0"]
