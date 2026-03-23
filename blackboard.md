# Blackboard — ndarray

> Single-binary architecture: already Rust. Integrates as `crate::simd` + `crate::linalg`.

## What Exists

Production-grade N-dimensional array library with three-tier SIMD dispatch (AVX-512 → AVX2 → Scalar), pluggable BLAS backends (Native/MKL/OpenBLAS), and 55 HPC extension modules.

## Core Data Structure

```rust
pub struct ArrayBase<S, D, A = <S as RawData>::Elem> {
    data: S,                    // Ownership: Owned, View, ArcArray, CowArray
    parts: ArrayPartsSized<A, D>,  // ptr + dim + strides
}

// Type aliases
type Array<A, D> = ArrayBase<OwnedRepr<A>, D>;      // Owned
type ArrayView<'a, A, D> = ArrayBase<ViewRepr<&'a A>, D>;  // Read-only view
```

## SIMD Dispatch

```
LazyLock<Tier> detected once at first call:
  AVX-512F → Tier::Avx512 (F32x16, F64x8)
  AVX2+FMA → Tier::Avx2 (F32x8, F64x4)
  Fallback → Tier::Scalar
```

dispatch! macro generates one-line stubs per function.

## BLAS Operations

### Level 1 (Vector-Vector)
`dot_f32/f64`, `axpy_f32/f64`, `scal_f32/f64`, `nrm2_f32/f64`, `asum_f32/f64`

### Level 2 (Matrix-Vector)
`gemv_f32/f64`, `ger_f32/f64`

### Level 3 (Matrix-Matrix)
`gemm_f32/f64` via `matrixmultiply` crate (Goto BLAS kernel)

## HPC Extensions (`src/hpc/`, 55 modules)

| Module | Purpose |
|---|---|
| `blas_level1/2/3.rs` | BLAS trait extensions |
| `statistics.rs` | median, variance, percentiles |
| `activations.rs` | sigmoid, softmax, relu |
| `fft.rs` | Cooley-Tukey FFT |
| `fingerprint.rs` | 32/256/512-bit containers |
| `cascade.rs` | Hamming distance bands |
| `nars.rs` | NARS reasoning |
| `arrow_bridge.rs` | Apache Arrow integration |
| `clam.rs` | Hierarchical clustering |

## Integration Points for Binary

- lance-graph's BlasGraph calls ndarray for SIMD Hamming distance
- Query result DataFrames use Arrow bridge
- Fingerprint/cascade search for semantic retrieval

## Key Files

| File | Size | Purpose |
|---|---|---|
| `src/lib.rs` | 66KB | ArrayBase definition, exports |
| `src/backend/mod.rs` | 5KB | BlasFloat trait, backend selection |
| `src/backend/native.rs` | 23KB | SIMD dispatch, BLAS L1/L2 |
| `src/backend/kernels_avx512.rs` | 29KB | AVX-512 intrinsics |
| `src/simd_avx512.rs` | 39KB | SIMD wrapper types |
| `src/simd.rs` | 29KB | Public SIMD API |
| `src/hpc/` | 42KB | 55 HPC extension modules |
| `src/impl_methods.rs` | 125KB | Core array methods |
