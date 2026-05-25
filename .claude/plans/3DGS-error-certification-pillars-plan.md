# 3DGS Error Certification Pillars Plan — ndarray

## Goal

Use `ndarray::hpc::pillar` as the executable certification layer for 3DGS rendering decisions.

The purpose is to make approximation decisions inspectable:

```text
splat/tile approximation
        ->
measured error components
        ->
certificate
        ->
render/refine/reject decision
```

## Existing anchor

The `src/hpc/pillar` module already separates:

- cognitive-architecture pillars migrated from `lance-graph/crates/jc`
- substrate-native pillars for ndarray

The 3DGS rebuild should make this module useful at runtime and in CI without moving policy into the renderer.

## Certificate model

Add a lightweight certificate DTO under `src/hpc/pillar/cert.rs` or equivalent:

```rust
pub struct ErrorCertificate {
    pub geometric_error_px: f32,
    pub sampling_error: f32,
    pub covariance_error: f32,
    pub quantization_error: f32,
    pub dependence_inflation: f32,
    pub total_error_px: f32,
    pub confidence: f32,
    pub passed: bool,
}

pub enum CertificateFailure {
    NonPsdCovariance,
    ExcessiveProjectionRadius,
    WeakDependenceTooHigh,
    SamplingDiscrepancyTooHigh,
    QuantizationErrorTooHigh,
    NonFiniteValue,
}
```

## Pillar mapping

### Pillar 6 / 7: EWA sandwich

Use for 2D and 3D covariance push-forward validation.

Runtime relevance:

- detect non-PSD projected covariances
- bound projected footprint instability
- validate scale/quaternion to SPD covariance construction

### Pillar 9: high-dimensional CLT / covariance field

Use for aggregate covariance stability over blocks of splats.

Runtime relevance:

- block-level confidence for large splat groups
- stability checks for compressed/quantized covariance fields

### Pillar 10: nested distance

Use for tile tree / DN-tree quantization preservation.

Runtime relevance:

- certify that coarse tile summaries preserve downstream selection quality
- bound error across parent-child aggregation

### Pillar 12: splat invariants

Use as the primary production gate for anisotropic splat construction.

Runtime relevance:

- verify trace/determinant/Frobenius invariants
- reject malformed scale/quaternion combinations

### Pillar 13: HHTL contraction

Use to certify cascade refinement.

Runtime relevance:

- parent-to-child error contraction
- safe pruning and skip-LOD decisions

## Runtime API sketch

```rust
pub fn certify_splat_block(
    block_stats: &SplatBlockStats,
    render_budget: &RenderBudget,
) -> ErrorCertificate;

pub fn certify_projected_covariance(
    sigma_3d: Spd3,
    camera_jacobian: Mat23,
) -> CovarianceCertificate;
```

## Determinism rules

- Every stochastic probe must use `SplitMix64` or the existing deterministic harness.
- Every report must include seed, sample count, tolerance, and measured values.
- Runtime certificates must not depend on wall-clock timing.
- CI probes must be reproducible across machines within documented tolerances.

## Acceptance criteria

- `cargo test -p ndarray --features std,linalg,pillar,splat3d`
- Pillar reports are stable over two consecutive runs.
- 3DGS projection tests can request a certificate and receive structured failure reasons.
- Certificates can be serialized or converted to debug text for `lance-graph` ingestion.
- CI separates proof failures from benchmark regressions.

## Open design questions

- Should runtime certificates use `f32` for speed or `f64` for audit output?
- Should high-cost certification be offline-only, while runtime uses precomputed bounds?
- Should tile-level certificates live in `lance-graph` and splat-block certificates live here?

Recommended answer: keep primitive splat/block certificates here; aggregate tile certificates in `lance-graph`.
