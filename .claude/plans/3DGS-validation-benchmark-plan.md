# 3DGS Validation and Benchmark Plan — ndarray

## Goal

Create the reproducibility, correctness, and performance harness for the ndarray-side 3DGS work.

This plan covers tests and benchmarks for:

- splat covariance construction
- EWA projection
- CPU-SIMD projection/raster kernels
- pillar certificates
- quantized codecs
- HHTL cascade decisions

## Test tiers

### Tier 0: compile gates

Required gates:

```bash
cargo check -p ndarray --features std,linalg,splat3d
cargo check -p ndarray --features std,linalg,pillar,splat3d
cargo test  -p ndarray --features std,linalg,pillar,splat3d
```

### Tier 1: scalar reference tests

Every SIMD path must have a scalar reference.

Required tests:

- scalar covariance construction
- scalar EWA projection
- scalar frustum rejection
- scalar HHTL decision
- scalar codec roundtrip

### Tier 2: SIMD equivalence tests

For each supported SIMD tier:

- compare projected coordinates within tolerance
- compare projected covariance within tolerance
- compare rejection masks exactly where possible
- compare aggregate counters exactly

### Tier 3: deterministic pillar probes

Run the active pillar probes twice in the same test and assert identical reports where expected.

Required properties:

- seed preserved
- sample count preserved
- pass/fail preserved
- PSD rate stable
- concentration values stable within tolerance

### Tier 4: golden vectors

Add small committed fixtures:

```text
tests/fixtures/3dgs/
  tiny_splats_f32.json
  tiny_camera.json
  projected_reference.json
  quantized_reference.json
```

Fixtures should be small enough for code review.

## Benchmark groups

Use Criterion where already available.

Suggested benchmarks:

```text
splat3d_project_1k
splat3d_project_10k
splat3d_project_100k
splat3d_project_1m
splat3d_codec_f32_to_f16
splat3d_codec_tile_i16
hhtl_heel_frustum_100k
hhtl_full_cascade_100k
pillar_ewa_3d_probe
```

## Metrics to record

- splats projected per second
- rejected splats per second
- blocks classified per second
- bytes read per projected splat
- allocations per call
- max covariance error
- max projected coordinate error
- certificate generation time

## Regression policy

A benchmark regression is not automatically a correctness failure.

Correctness failures:

- non-deterministic certificate output
- invalid PSD covariance accepted
- scalar/SIMD divergence above tolerance
- codec roundtrip above declared error
- HHTL action mismatch between scalar and SIMD reference

Performance warnings:

- more allocations in hot path
- throughput regression above configured threshold
- unexpected branch-heavy behavior in SIMD path

## CI recommendations

Use two profiles:

1. correctness profile
   - fast
   - runs on every PR
   - scalar + default SIMD feature checks

2. benchmark profile
   - manual or scheduled
   - records hardware and CPU features
   - writes results to a stable report artifact

## Acceptance criteria

- A developer can run one command and verify 3DGS correctness.
- A developer can run one command and benchmark 3DGS hot paths.
- Every certificate includes enough metadata to reproduce the run.
- All fixture files are small and human-inspectable.
- No benchmark depends on network access.
