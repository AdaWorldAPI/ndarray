//! Per-block quantized dot products (Q4_0 × Q8_0 → f32, etc.).
//!
//! Distinct from `hpc::quantized::int8_gemm_f32` (full GEMM, M×N×K).
//! These functions compute a single inner product between two
//! pre-quantized vectors, mirroring candle's `vec_dot_q4_0_q8_0`
//! kernels at `candle-core/src/quantized/{k_quants,avx,neon,simd128}.rs`.
//!
//! WS-2 stub — implementation lands in the work-steal sprint.

// Stub: filled in by WS-2 worker.
