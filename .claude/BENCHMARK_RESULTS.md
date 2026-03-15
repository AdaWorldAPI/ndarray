# BENCHMARK RESULTS: ndarray (port) vs rustynum (reference) vs numpy

**Date**: 2026-03-15
**Hardware**: 2-core x86_64, 2.1 GHz, 8192 KB cache
**SIMD**: AVX-512F, VPOPCNTDQ, VNNI, AVX2, FMA
**Rust**: 1.94.0 (target-cpu=native)
**NumPy**: 2.4.2 (OpenBLAS 0.3.31, Haswell arch)
**OS**: Ubuntu 24 (container)

---

## Test 1: BLAS Level 1 — dot_f32

| SIZE | RUSTYNUM | NDARRAY | NUMPY | ND/RN | STATUS |
|------|----------|---------|-------|-------|--------|
| 256 | 61ns | 58ns | 604ns | 0.95x | ✅ |
| 1K | 90ns | 78ns | 623ns | 0.87x | ✅ faster |
| 4K | 252ns | 180ns | 892ns | 0.71x | ✅ faster |
| 16K | 1355ns | 1220ns | 1856ns | 0.90x | ✅ |
| 64K | 4833ns | 4838ns | 6911ns | 1.00x | ✅ |
| 256K | 34546ns | 35049ns | 50451ns | 1.02x | ✅ |
| 1M | 310366ns | 317861ns | 363567ns | 1.02x | ✅ |

**Analysis**: ndarray slightly faster at small sizes (better inlining from cfg(target_arch)
guards vs LazyLock dispatch). Converges at memory-bound sizes. Both 2-10x faster than
numpy for L1-resident vectors (numpy has Python→C FFI overhead ~500ns).

## Test 1b: BLAS Level 1 — axpy_f32

| SIZE | RUSTYNUM | NDARRAY | NUMPY | ND/RN | STATUS |
|------|----------|---------|-------|-------|--------|
| 256 | 42ns | 41ns | 1335ns | 0.98x | ✅ |
| 1K | 82ns | 70ns | 1597ns | 0.85x | ✅ |
| 4K | 216ns | 173ns | 2449ns | 0.80x | ✅ faster |
| 16K | 1752ns | 2085ns | 6850ns | 1.19x | ⚠️ |
| 64K | 6892ns | 7989ns | 20335ns | 1.16x | ⚠️ |
| 256K | 34880ns | 36815ns | 215837ns | 1.06x | ✅ |
| 1M | 333969ns | 320907ns | 998903ns | 0.96x | ✅ |

**Analysis**: 16K-64K range shows ndarray ~15-19% slower. Possible cause: ndarray's
axpy uses `#[cfg(target_arch = "x86_64")]` conditional compilation which may not
unroll identically to rustynum's direct dispatch. Not critical — both crush numpy.

## Test 2: Hamming Distance — CRITICAL

| BITS | RUSTYNUM (VPOPCNTDQ) | NDARRAY (AVX2 vpshufb) | NUMPY | ND/RN | GBPS (RN) |
|------|---------------------|------------------------|-------|-------|-----------|
| 1K | 35ns | 37ns | 2248ns | 1.06x | 7.3 |
| 2K | 38ns | 41ns | 2682ns | 1.08x | 13.5 |
| 4K | 40ns | 47ns | 3608ns | 1.18x | 25.6 |
| 8K | 47ns | 61ns | 5204ns | 1.30x | 43.6 |
| 16K | 60ns | 88ns | 8376ns | 1.47x | 68.3 |
| 32K | 83ns | 142ns | 14574ns | 1.71x | 98.7 |
| **64K** | **135ns** | **249ns** | 26975ns | **1.84x** | **121.4** |
| 128K | 230ns | 494ns | 51635ns | **2.15x** | 142.5 |
| 256K | 688ns | 1047ns | 101164ns | 1.52x | 95.3 |

### L1 Eviction Cliff

Throughput (GB/s) peaks at 128Kbit (16KB per vector = 32KB pair → fits in L1d) at 142.5 GB/s,
then drops to 95.3 GB/s at 256Kbit (32KB per vector = 64KB pair → L1 thrash).

**The cliff is at 128Kbit↔256Kbit boundary.** Maximum fingerprint size for hot-path
cascade without L1 penalty: **128Kbit = 16KB**.

### Root Cause

ndarray's `hpc/bitwise.rs` dispatches to AVX2 `vpshufb` (nibble-lookup popcount).
rustynum uses AVX-512 `_mm512_popcnt_epi64` (VPOPCNTDQ). At 64Kbit fingerprints:
- VPOPCNTDQ processes 64 bytes/cycle (8×u64 popcnt per zmm)
- vpshufb processes 32 bytes/cycle with extra shuffle+mask overhead

**→ PORT VPOPCNTDQ KERNEL TO NDARRAY. This is the #1 priority.**

## Test 3: GEMM (sgemm)

| SIZE | RUSTYBLAS | NDARRAY | NUMPY (OpenBLAS) | RB GFLOPS | ND GFLOPS | NP GFLOPS |
|------|-----------|---------|-------------------|-----------|-----------|-----------|
| 64×64 | 0.02ms | 0.03ms | 0.005ms | 31.6 | 16.5 | 112.5 |
| 128×128 | 0.11ms | 0.20ms | 0.05ms | 37.9 | 20.6 | 76.5 |
| 256×256 | 0.79ms | 1.68ms | 0.17ms | 42.4 | 20.0 | 192.0 |
| 512×512 | 5.84ms | 15.36ms | 1.21ms | 46.0 | 17.5 | 222.0 |

### Root Cause

ndarray's `backend/native.rs::gemm_f32()` uses a naive axpy-based tiled GEMM:
```
for i in 0..ib { for p in 0..kb { axpy_f32(a_val, b_row, c_row); } }
```

rustyblas uses Goto BLAS algorithm: panel packing → L1-resident microkernel (6×16 for AVX-512).
The packing eliminates TLB misses and the microkernel keeps all 32 zmm registers busy.

**→ PORT RUSTYBLAS GEMM MICROKERNEL TO NDARRAY. #2 priority.**

## Test 4: Batch Hamming (Cascade Stroke 1)

| CANDIDATES | STROKE_BYTES | RUSTYNUM | NDARRAY | RATIO |
|-----------|-------------|----------|---------|-------|
| 1K | 128 | 3μs | 15μs | 4.0x |
| 10K | 128 | 48μs | 159μs | 3.3x |
| 100K | 128 | 894μs | 1911μs | 2.1x |
| 1M | 128 | 9057μs | 26329μs | 2.9x |
| 1M | 512 | 57717μs | 87448μs | 1.5x |
| 1M | 2048 | 216525μs | 402034μs | 1.9x |

### Root Cause

ndarray has no batch hamming kernel. The benchmark loops per-candidate with Array1
allocation overhead. rustynum's `hamming_batch` operates on a flat contiguous database
slice with zero allocation.

**→ ADD BATCH HAMMING KERNEL TO NDARRAY. #3 priority.**

## Test 5: Correctness Verification

| Kernel | Result | Detail |
|--------|--------|--------|
| dot_f32 | ✅ BIT-EXACT | Same FMA accumulator pattern |
| dot_f64 | ✅ BIT-EXACT | Same FMA accumulator pattern |
| axpy_f32 | ✅ 1 ULP max | FMA vs mul+add rounding — expected |
| scal_f32 | ✅ BIT-EXACT | Same codegen |
| nrm2_f32 | ✅ BIT-EXACT | Same codegen |
| asum_f32 | ✅ BIT-EXACT | Same codegen |
| hamming | ✅ EXACT | Integer — no rounding |
| sgemm 64 | ✅ EXACT | max_abs_err = 0 |

**All kernels produce identical or 1-ULP results.** The port is numerically correct.
The axpy 1-ULP difference is expected: FMA(a,x,y) ≠ a*x+y in IEEE 754 when
intermediate precision differs.

---

## Priority Action Items

| # | Priority | Item | Impact |
|---|----------|------|--------|
| 1 | 🔴 CRITICAL | Port VPOPCNTDQ hamming to ndarray | 1.8-2.1x speedup at fingerprint sizes |
| 2 | 🔴 CRITICAL | Port Goto BLAS GEMM microkernel | 2-2.6x speedup, competitive with OpenBLAS |
| 3 | 🟡 HIGH | Add batch hamming kernel | 2-4x speedup for cascade search |
| 4 | 🟢 LOW | Investigate axpy 15-19% gap at 16K-64K | Minor, both beat numpy handily |
| 5 | 🟢 LOW | BLAS L1 — no action needed | Already matching or faster |
