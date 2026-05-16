//! Half-precision dot products: `dot_bf16` and `dot_f16`.
//!
//! Ported from candle's per-arch `vec_dot_{bf16,f16}` in
//! `candle-core/src/cpu/mod.rs` (AVX2 FMA on packed half-precision lanes).
//! Adds an AVX-512 BF16 (`vdpbf16ps`) fast path and a NEON aarch64 `fp16`
//! fallback that candle does not have today.
//!
//! WS-1 stub — implementation lands in the work-steal sprint.

// Stub: filled in by WS-1 worker.
