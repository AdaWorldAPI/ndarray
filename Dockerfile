# ndarray — Railway compile-test image (AVX2 default)
# Verifies the HPC module builds cleanly (default + jit-native features)
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

# Install Rust via rustup. The version below MUST match rust-toolchain.toml's
# `channel` and satisfy Cargo.toml's `rust-version`; the number is deliberately
# not restated in this prose, per rust-toolchain.toml's own warning that "a stale
# comment on a version pin is how the next reader learns the wrong number" — that
# is exactly how this file came to pin 1.97.1 while the repo required 1.98, which
# fails at once with "rustc 1.97.1 is not supported by the following package".
# rust-toolchain.toml is deliberately NOT copied into the image (rustup would
# download a second toolchain at build time), so this pin is the only thing
# keeping the image in step with the repo — bump it whenever the channel moves.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --default-toolchain 1.98.1 --profile minimal \
    && rustc --version | grep -q "1.98.1"

WORKDIR /app

# Copy workspace files first for layer caching.
#
# Cargo.lock is deliberately NOT copied: it is gitignored (see .gitignore — this
# is a library crate consumed by sibling repos via git dependency, so a committed
# lock is never used by a downstream build), which means it is absent from the
# build context and `COPY Cargo.toml Cargo.lock ./` fails the build outright with
# "failed to calculate checksum ... /Cargo.lock: not found" — BuildKit's wording
# for a missing source, not a corrupt one. Cargo resolves fresh here instead.
COPY Cargo.toml ./
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
# The cargo CONFIG DIRECTORY, required by the `--config` flags below and easy to
# forget: this Dockerfile COPYs selectively by design (see the note above), so a
# file that is not named here does not exist in the image. Adding `--config
# .cargo/config-v3.toml` without this line makes cargo fail on a missing
# configuration file BEFORE it compiles anything — which is exactly what
# happened on #313 and was caught in review after merge, not by a build (there
# is no Docker daemon in the dev container, so neither image is built here).
COPY .cargo/ .cargo/

# The tier is passed as a CONFIG, not as `ENV RUSTFLAGS` (changed 2026-09-16).
# A RUSTFLAGS env REPLACES every cargo-config `rustflags` entry rather than
# joining it, so `ENV RUSTFLAGS="-C target-cpu=x86-64-v3"` did set the tier —
# and silently dropped `.cargo/config.toml`'s two crypto-backend cfgs
# (`curve25519_dalek_backend="serial"`, `poly1305_force_soft`) that compile out
# curve25519-dalek's and poly1305's raw-intrinsic AVX2 backends. This image
# therefore shipped the unaudited SIMD surfaces the matryoshka rule exists to
# keep out. `--config` JOINS, so the tier AND the cfgs both apply.
#
# Passing it explicitly is also now required rather than optional: the default
# `.cargo/config.toml` is `target-cpu=native`, which tunes the artifact to
# whatever machine built the image and is not portable.

# Build default features
RUN cargo --config .cargo/config-v3.toml build --release 2>&1 && echo "=== DEFAULT BUILD OK ==="

# Build with JIT
RUN cargo --config .cargo/config-v3.toml build --release --features jit-native 2>&1 && echo "=== JIT-NATIVE BUILD OK ==="

# Run tests
RUN cargo --config .cargo/config-v3.toml test --release --lib -- hpc:: 2>&1 && echo "=== HPC TESTS OK ==="

# Minimal runtime image — just proves it compiled
FROM debian:bookworm-slim
COPY --from=builder /app/target/release/libndarray.rlib /usr/local/lib/
CMD ["echo", "ndarray build verified"]
