//! FloatTensorOps for AdaWorld backend.
//!
//! 84 required methods + ~36 with defaults = ~120 total.
//! Delegates to ndarray operations with crate::simd acceleration.
//!
//! # Implementation Priority
//!
//! P0 (Whisper minimal): from_data, into_data, matmul, add, mul, div, exp,
//!     reshape, transpose, swap_dims, device, to_device, shape, empty, zeros, ones
//!
//! P1 (full inference): softmax, log, sqrt, neg, recip, gather, select, slice,
//!     mask_where, cat, sum, mean, max, min, argmax, argmin, equal
//!
//! P2 (training): backward-compatible with burn-autodiff (future)

// Implementation will follow burn-ndarray's pattern:
// https://github.com/tracel-ai/burn/tree/main/crates/burn-ndarray/src/ops
//
// Key differences from burn-ndarray:
//   1. Uses crate::simd::F32x16 instead of macerator
//   2. Uses LazyLock<SimdDispatch> for tier selection
//   3. Optional AttentionTable for compiled matmul
//   4. SimilarityTable for BF16-equivalent scoring
