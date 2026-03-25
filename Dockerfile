# ndarray — Railway compile-test image
# Verifies the HPC module builds cleanly (default + jit-native features)
#
# Build: docker build -t ndarray-test .
# Run:   docker run --rm ndarray-test

FROM rust:1.85-slim AS builder

WORKDIR /app

# System deps for Cranelift JIT
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

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
CMD ["echo", "ndarray build verified"]
