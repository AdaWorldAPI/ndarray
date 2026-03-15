# Project NDARRAY Expansion — Blackboard

> Shared state surface for all agents. Read before starting, update after completing work.

## Epoch: 1
## Global Goal: Port rustynum HPC features into ndarray fork

### Environment
- rust_version: 1.94-stable
- perf_target_blas: MKL (primary), OpenBLAS (alternative)
- simd_level: AVX-512 (primary), AVX2 (fallback), SSE4.2 (minimum)

---

## Strategic Analysis
<!-- l3-strategist writes here -->

---

## Architecture Decisions
<!-- savant-architect writes here -->

---

## API Surface
<!-- product-engineer writes here -->

---

## Vector Operations
<!-- vector-synthesis writes here -->

---

## QA Audit Log
<!-- sentinel-qa writes here -->

---

## Loose Ends
- [ ] Define feature-gate hierarchy (native/mkl/openblas)
- [ ] Audit GEMM macro porting feasibility
- [ ] Backend trait: generic (monomorphized) vs enum dispatch
- [ ] Determine ndarray version pinning strategy (fork vs upstream PR)
- [ ] CI matrix: which feature combinations to test
- [ ] Benchmark harness: criterion setup with GFLOP/s reporting

---

## Agent Handoff Log
<!-- Format: [agent] → [agent]: reason -->
