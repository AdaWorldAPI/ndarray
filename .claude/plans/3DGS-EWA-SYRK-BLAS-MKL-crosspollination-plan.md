# 3DGS EWA/SYRK/BLAS/MKL Cross-Pollination Plan — ndarray

## Goal

Fold the older PR-X12 BLAS/GEMM lens into the new 3DGS plan family.

The new 3DGS plans should not rediscover this line:

```text
EWA sandwich
  -> batched SYRK / GEMM form
  -> backend dispatch: native / Intel MKL / OpenBLAS / AMX where available
  -> shared Basis<T> / LinearReduce<T> substrate
```

## Existing canon to preserve

Prior architecture docs already identify a high-value Plan C:

```text
Plan C — EWA SYRK-batched
Replace per-Gaussian sandwich loop with batched cblas_ssyrk.
Add backend dispatch: native / intel-mkl / openblas.
Target file: src/hpc/splat3d/spd3.rs plus backend wiring.
```

This plan promotes that into the 3DGS roadmap.

## Kernel identity

The 3DGS projection core contains this sandwich:

```text
Sigma_image = J * W * Sigma * W^T * J^T
```

For many Gaussians, this is not only a per-splat loop. It can be batched:

```text
for each Gaussian block:
  M = J * W
  Sigma_image = M * Sigma * M^T
```

Batched form:

```text
many small SPD sandwiches
        ->
SYRK / GEMM-like backend
        ->
SIMD / MKL / OpenBLAS / AMX dispatch
```

## Why this matters

The current CPU-SIMD path is valuable, but the older canon found a deeper win:

```text
3DGS projection is a BLAS workload in disguise.
```

The renderer should expose two tiers:

```text
small / scalar-ish batches:
  native SIMD path

large / stable batches:
  BLAS backend path: SYRK / GEMM
```

## Proposed module layout

```text
src/hpc/splat3d/
  spd3.rs
  projection.rs
  batch_sandwich.rs
  backend.rs

src/hpc/splat3d/backend/
  native.rs
  mkl.rs
  openblas.rs
  amx.rs
```

If the project prefers existing backend namespaces, adapt to that instead of inventing a second backend tree.

## API sketch

```rust
pub enum Splat3dBackend {
    NativeSimd,
    IntelMkl,
    OpenBlas,
    AmxBf16,
}

pub struct SandwichBatchConfig {
    pub backend: Splat3dBackend,
    pub min_batch_for_blas: usize,
    pub allow_bf16: bool,
    pub require_deterministic: bool,
}

pub fn sandwich_batch_spd3(
    transforms: &[Mat3x3<f32>],
    sigmas: &[Spd3],
    out: &mut [Spd2],
    config: &SandwichBatchConfig,
) -> Splat3dBatchReport;
```

## Backend dispatch rules

```text
NativeSimd:
  always available
  deterministic reference path
  best for small batches

IntelMkl:
  feature-gated
  best for large stable batches
  must have numerical tolerance tests against NativeSimd

OpenBlas:
  feature-gated
  portability backend
  tolerance may differ by platform

AmxBf16:
  experimental / certified path only
  requires pillar probe before becoming default
```

## Interaction with 4x4 carrier

The 4x4 carrier plan remains compatible:

```text
Mat4 world/camera transform
        ->
extract spatial W / J path
        ->
3x3 SPD sandwich remains authoritative
        ->
optional 4x4 metadata/cognitive lane survives alongside
```

Do not replace 3x3 SPD covariance with 4x4 unless the extra lane has a formally defined meaning.

## Certification hooks

Add benchmark and proof links:

```text
Pillar-7 / EWA Sandwich 3D:
  PSD preservation
  concentration / CV tightness

New backend parity test:
  NativeSimd vs MKL/OpenBLAS/AMX within tolerance

New performance test:
  small-batch native wins
  large-batch BLAS wins
```

## Acceptance criteria

- Native SIMD remains the scalar-compatible reference.
- BLAS backend results match reference within declared tolerance.
- Backend selection is explicit and testable.
- No hard dependency on MKL in default build.
- Large-batch benchmark demonstrates why the BLAS path exists.
- 4x4 carrier plan can call the same backend without breaking 3x3 covariance invariants.

## Cross-repo link

`lance-graph` owns SplatShaderBlas / BLASGraph orchestration and tile/block scheduling. `ndarray` owns this numerical kernel surface.
