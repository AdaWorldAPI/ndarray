# BENCHMARK RESULTS: ndarray (port) vs rustynum (reference) vs numpy

**Date**: 2026-03-15 (final)
**Hardware**: 2-core x86_64, 2.1 GHz, 8192 KB cache
**SIMD**: AVX-512F, VPOPCNTDQ, VNNI, AVX2, FMA
**Rust**: 1.94.0 (target-cpu=native)
**NumPy**: 2.4.2 (OpenBLAS 0.3.31, Haswell arch)

## Changes Applied

1. **VPOPCNTDQ hamming dispatch** — kernel existed but was never called
2. **Goto BLAS GEMM** — packed 6×16/6×8 microkernels replacing naive axpy tiling
3. **Public raw-slice API** — `hamming_batch_raw()` for zero-allocation batch path

---

## Test 1: BLAS Level 1

| SIZE | RUSTYNUM | NDARRAY | NUMPY | ND/RN |
|------|----------|---------|-------|-------|
| 256 | 56ns | 53ns | 604ns | 0.95x ✅ |
| 1K | 101ns | 92ns | 623ns | 0.91x ✅ |
| 4K | 273ns | 239ns | 892ns | 0.88x ✅ |
| 16K | 1592ns | 1584ns | 1856ns | 1.00x ✅ |
| 64K | 6227ns | 6229ns | 6911ns | 1.00x ✅ |
| 256K | 45115ns | 44745ns | 50451ns | 0.99x ✅ |
| 1M | 334304ns | 313248ns | 363567ns | 0.94x ✅ |

**Verdict**: Matching. Same AVX-512 FMA codegen.

## Test 2: Hamming Distance

| BITS | RUSTYNUM | NDARRAY | NUMPY | ND/RN |
|------|----------|---------|-------|-------|
| 1K | 31ns | 29ns | 2248ns | 0.94x ✅ |
| 4K | 32ns | 32ns | 3608ns | 1.00x ✅ |
| 16K | 49ns | 47ns | 8376ns | 0.96x ✅ |
| **64K** | **104ns** | **106ns** | 26975ns | **1.02x ✅** |
| 128K | 178ns | 201ns | 51635ns | 1.13x ✅ |
| 256K | 550ns | 551ns | 101164ns | 1.00x ✅ |

**Before**: 64Kbit was 1.84x (AVX2 vpshufb only). **After**: 1.02x (VPOPCNTDQ).

Peak throughput: 184 GB/s at 128Kbit.
L1 cliff: 128Kbit→256Kbit (throughput drops 184→119 GB/s).

## Test 3: GEMM

| SIZE | RUSTYBLAS | NDARRAY | NUMPY | RB GFLOPS | ND GFLOPS |
|------|-----------|---------|-------|-----------|-----------|
| 64 | 0.01ms | 0.02ms | 0.005ms | 40.9 | 27.7 |
| 128 | 0.08ms | 0.08ms | 0.05ms | 52.1 | 54.3 |
| 256 | 0.55ms | 0.51ms | 0.17ms | 61.5 | 66.3 |
| 512 | 5.99ms | 6.34ms | 1.21ms | 44.8 | 42.4 |
| **1024** | **83.7ms** | **44.7ms** | 8.53ms | 25.7 | **48.0** |

**Before**: ndarray 16-20 GFLOPS (axpy-based, 2-2.6x slower).
**After**: ndarray 28-66 GFLOPS (packed microkernels, matches rustyblas).

At 1024×1024, ndarray single-threaded path (48 GFLOPS) beats rustyblas's
multi-threaded path (25.7 GFLOPS) because threading overhead on 2 cores.

## Test 4: Batch Hamming

| CANDIDATES | BYTES | RUSTYNUM | ND_RAW | ND_TRAIT | RAW RATIO |
|-----------|-------|----------|--------|----------|-----------|
| 1K | 128 | 4μs | 5μs | 4μs | 1.2x ✅ |
| 10K | 128 | 67μs | 42μs | 42μs | 0.6x ✅ |
| 100K | 128 | 746μs | 801μs | 821μs | 1.1x ✅ |
| 1M | 128 | 7847μs | 8161μs | 8160μs | 1.0x ✅ |
| 1M | 2048 | 218ms | 224ms | 223ms | 1.0x ✅ |

**Before**: benchmark used per-candidate Array1 allocation → 2.7x overhead.
**After**: `hamming_batch_raw()` and `hamming_query_batch()` → 1.0x parity.

## Test 5: Correctness

| Kernel | Result |
|--------|--------|
| dot_f32/f64 | ✅ BIT-EXACT |
| axpy_f32 | ✅ 1 ULP max |
| scal_f32, nrm2_f32, asum_f32 | ✅ BIT-EXACT |
| hamming | ✅ EXACT |
| batch hamming (1000×128B) | ✅ EXACT |
| sgemm 64×64 | ✅ max_err = 0 |
| sgemm 256×256 | ✅ max_err = 0 |

---

## Final Status

| Area | Status | Notes |
|------|--------|-------|
| BLAS L1 | ✅ PARITY | Same codegen |
| Hamming | ✅ PARITY | VPOPCNTDQ wired |
| GEMM | ✅ PARITY | Goto BLAS microkernels |
| Batch Hamming | ✅ PARITY | Zero-alloc raw API |
| Correctness | ✅ ALL PASS | Bit-exact or 1-ULP |

## Remaining Optimization Opportunities

| Priority | Item |
|----------|------|
| 🟡 | Multi-threaded GEMM for M×N > threshold |
| 🟡 | AVX2 fallback GEMM microkernel (currently falls to axpy) |
| 🟢 | Close gap to OpenBLAS/MKL (would need assembly microkernels) |
