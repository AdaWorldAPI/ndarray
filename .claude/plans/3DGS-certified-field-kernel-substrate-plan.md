# 3DGS Certified Field Kernel Substrate Plan — ndarray

## Goal

Generalize the ndarray-side 3DGS math into a domain-neutral certified field-kernel substrate.

The 3DGS renderer is the first concrete consumer, but the numerical pattern is wider:

```text
local field block
  -> kernel summary
  -> SIMD / BLAS / tensor operation
  -> error certificate
  -> skip / refine / hydrate decision support
```

## What stays in ndarray

`ndarray` owns reusable numerical kernels:

```text
EWA sandwich
SPD covariance operations
Mat4 / Sym4 / Block4 carriers
SIMD HHTL scoring kernels
BLAS/SYRK/GEMM backend dispatch
popcount / distance / palette lookup kernels
quantization and codec helpers
pillar probes and parity tests
```

It does not own domain semantics.

## Field kernel abstraction

A field kernel is a compact local approximation of a larger signal.

Examples:

```text
3DGS:
  anisotropic Gaussian over 3D position, color, opacity, covariance

Datalake:
  statistics / centroid / covariance / bloom over a fragment

RAG:
  semantic centroid + provenance + uncertainty over a chunk family

Ultrasound:
  PSF covariance + amplitude / Doppler / phase over a frame block

Genetics:
  motif / transition / expression kernel over a sequence window

Neuronal:
  activation / edge uncertainty kernel over a microcircuit block
```

## Suggested module layout

```text
src/hpc/field_kernel/
  mod.rs
  block.rs
  summary.rs
  certificate.rs
  decision_support.rs
  simd_score.rs
  blas_backend.rs
  block4.rs
```

This can start as a plan only. Do not create a generic abstraction until two concrete consumers need it.

## Core DTO sketches

```rust
pub struct FieldKernelSummary {
    pub block_len: usize,
    pub density: f32,
    pub energy: f32,
    pub variance: f32,
    pub max_error: f32,
    pub confidence: f32,
}

pub struct FieldKernelCertificate {
    pub total_error: f32,
    pub confidence: f32,
    pub passed: bool,
    pub reason_codes: Vec<FieldKernelReason>,
}

pub enum FieldKernelReason {
    BelowErrorBudget,
    AboveErrorBudget,
    NonPsdCovariance,
    QuantizationTooHigh,
    SamplingTooSparse,
    DependenceInflationHigh,
    BackendParityFailed,
}
```

## Relation to 3DGS

3DGS remains the first hard implementation path:

```text
GaussianBatch / SplatBlockView
  -> EWA projection
  -> HHTL cascade
  -> field-kernel summary
  -> certificate
```

Do not weaken `splat3d` into an over-generic module too early. Extract common pieces only after they repeat.

## Relation to 4x4 carrier

The 4x4 cognitive-shader carrier is the preferred experimental block grammar:

```text
lane0: coordinate / source
lane1: state / covariance / transition
lane2: signal / activation / confidence
lane3: time / provenance / role
```

The certified field-kernel substrate should be able to score and certify 4x4 block summaries, but it should not require every domain to become 4x4 on day one.

## Backend types

```text
Native scalar reference
Native SIMD
BLAS / SYRK / GEMM
AMX / BF16 experimental
popcount / Hamming
palette table lookup
```

Every backend must have a reference path and parity tests.

## Acceptance criteria

- 3DGS code remains concrete and fast.
- Reusable kernel/certificate pieces can be extracted without domain semantics.
- Backend parity is testable.
- Field-kernel certificates can be consumed by `lance-graph` decisions.
- No domain-specific adapter logic is added to ndarray.

## Implementation trigger

Do not implement this as a big abstraction immediately.

Trigger extraction when at least two of these are active:

```text
3DGS splat blocks
Datalake HHTL summaries
RAG retrieval blocks
Ultrasound PSF blocks
4x4 cognitive-shader blocks
```

Until then, keep this as the north-star substrate plan.
