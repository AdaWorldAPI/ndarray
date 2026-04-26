# ndarray Docker CPU Detection & SIMD Dispatch

## Three-Tier Build Strategy

| Target | Dockerfile | RUSTFLAGS | CPU features | Use case |
|---|---|---|---|---|
| **Portable (AVX2)** | `Dockerfile` | `-C target-cpu=x86-64-v3` | SSE4.2, AVX, AVX2, FMA, BMI1/2 | GitHub CI, general servers, cloud VMs |
| **AVX-512 pinned** | `Dockerfile.avx512` | `-C target-cpu=x86-64-v4` | + AVX-512F/BW/CD/DQ/VL | Skylake-X, Ice Lake, Sapphire Rapids, EPYC Genoa |
| **Local dev** | `.cargo/config.toml` | (per-repo) | Whatever the developer's CPU supports | Developer machines |

## How SIMD Dispatch Works

ndarray uses a **two-layer dispatch** model:

### Layer 1: Compile-time (`cfg(target_feature)`)

When built with `target-cpu=x86-64-v4`, the compiler enables AVX-512
intrinsics at compile time. Types in `simd_avx512.rs` use native `__m512`
registers — zero overhead, everything inlined.

When built with `target-cpu=x86-64-v3`, AVX-512 intrinsics are NOT available
at compile time. The polyfill in `simd_avx2.rs` provides the same API (`F32x16`,
`U8x64`, etc.) using pairs of `__m256` operations or scalar loops.

### Layer 2: Runtime detection (`LazyLock<Tier>`)

Regardless of compile target, `src/simd.rs` detects the CPU at startup:

```rust
static TIER: LazyLock<Tier> = LazyLock::new(|| {
    if is_x86_feature_detected!("avx512f") { return Tier::Avx512; }
    if is_x86_feature_detected!("avx2")    { return Tier::Avx2; }
    #[cfg(target_arch = "aarch64")]
    if is_aarch64_feature_detected!("dotprod") { return Tier::NeonDotProd; }
    Tier::Scalar
});
```

Functions marked `#[target_feature(enable = "avx512f")]` are compiled into
the binary even at `-C target-cpu=x86-64-v3` and dispatched at runtime via
the tier detection. This means an AVX2-compiled binary **still uses AVX-512
kernels** when running on AVX-512 hardware — the difference is that the
generic `F32x16` / `U8x64` types use the AVX2 fallback (pairs of 256-bit
ops) rather than native 512-bit registers.

### What this means in practice

```
x86-64-v3 binary on AVX-512 hardware:
  F32x16::mul_add     → AVX2 fallback (2× _mm256_fmadd_ps)
  hamming_distance_raw → AVX-512 VPOPCNTDQ (runtime-dispatched)
  bitwise::popcount    → AVX-512 VPOPCNTDQ (runtime-dispatched)
  ┌───────────────────────────────────┐
  │ Generic SIMD types: AVX2 path     │ ← compile-time
  │ Per-function kernels: AVX-512     │ ← runtime-detected
  └───────────────────────────────────┘

x86-64-v4 binary on AVX-512 hardware:
  F32x16::mul_add     → native __m512 (_mm512_fmadd_ps)
  hamming_distance_raw → same AVX-512 VPOPCNTDQ
  ┌───────────────────────────────────┐
  │ Everything: AVX-512 native        │ ← compile-time + runtime
  └───────────────────────────────────┘
  ~24% faster overall (no 256→512 splitting overhead)
```

## AMX Detection (Intel Advanced Matrix Extensions)

AMX is NOT part of any `target-cpu` level. It requires:
1. CPUID check (AMX-TILE + AMX-INT8 + AMX-BF16 leaves)
2. OS support via `_xgetbv(0)` bits 17/18 (XTILECFG + XTILEDATA)
3. Linux: `prctl(ARCH_REQ_XCOMP_PERM)` to enable tile registers

Detection lives in `ndarray::hpc::amx_matmul::amx_available()`.
AMX kernels are always compiled in (they use inline assembly) and
gated at runtime. They work with any `-C target-cpu` setting.

## NEON (ARM / aarch64)

NEON is mandatory on aarch64 — always available. The distinction is:
- **NEON baseline** (ARMv8.0): `float32x4_t`, 4-wide f32
- **NEON dotprod** (ARMv8.2+, Pi 5 / A76+): `vdotq_s32`, 4× int8 throughput

Detection: `is_aarch64_feature_detected!("dotprod")` in `simd.rs`.

## Choosing the Right Dockerfile

```
┌─────────────────────────────────────────────────┐
│ Do you know your deployment hardware?           │
├───────────────┬─────────────────────────────────┤
│ No / Mixed    │ Use Dockerfile (AVX2 default)   │
│ AVX-512 only  │ Use Dockerfile.avx512 (+24%)    │
│ ARM / Pi      │ Use Dockerfile (NEON auto)      │
└───────────────┴─────────────────────────────────┘
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RUSTFLAGS` | (see Dockerfile) | Compiler flags including `-C target-cpu=...` |
| `CARGO_BUILD_JOBS` | (all cores) | Parallel compilation — reduce if OOM |

## Verifying CPU Features at Runtime

```bash
# Inside the container:
cat /proc/cpuinfo | grep -oP 'avx512\w+' | sort -u
# Or via Rust:
cargo run --example simd_caps  # prints detected SIMD tier
```

## Build Examples

```bash
# Portable (AVX2) — safe for GitHub CI, most cloud VMs
docker build -t ndarray-test .

# AVX-512 pinned — Sapphire Rapids, Ice Lake, EPYC Genoa
docker build -f Dockerfile.avx512 -t ndarray-avx512 .

# Override CPU target at build time (e.g., baseline for maximum compat)
docker build --build-arg RUSTFLAGS="-C target-cpu=x86-64" -t ndarray-compat .
```
