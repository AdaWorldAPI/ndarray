//! AdaWorld backend: implements burn's Backend trait.
//!
//! Delegates all tensor operations to ndarray + crate::simd.
//! This is the entry point — every burn model compiled with `Backend = AdaWorld`
//! runs on our SIMD dispatch with optional AttentionTable compiled attention.
//!
//! # Implementation Status
//!
//! The Backend trait requires ~200+ methods across 7 op traits.
//! Implementation strategy: core ops first (what Whisper/Llama need),
//! then expand coverage guided by burn-backend-tests.
//!
//! Required traits:
//!   FloatTensorOps  — 84 required methods (+ ~36 with defaults)
//!   IntTensorOps    — ~50 required methods
//!   BoolTensorOps   — ~30 required methods
//!   ModuleOps       — conv, pool, embedding, etc.
//!   ActivationOps   — relu, sigmoid, gelu (most have defaults)
//!   QTensorOps      — quantized tensor ops
//!   TransactionOps  — batch execution
//!
//! # Architecture
//!
//! ```text
//! burn::Tensor<AdaWorld, D>
//!   ↓ (burn dispatches via Backend trait)
//! AdaWorld::float_matmul(lhs, rhs)
//!   ↓ (check for compiled attention table)
//!   ├── AttentionTable[q_idx][k_idx]  → O(1)  (if compiled)
//!   └── ndarray general_mat_mul()     → O(d)  (fallback to BLAS)
//!         ↓ (ndarray delegates to BLAS or matrixmultiply)
//!         crate::simd::F32x16         → AVX-512 / AVX2 via LazyLock dispatch
//! ```

use crate::tensor::AdaTensor;

/// The AdaWorld backend.
///
/// CPU-only. Uses adaworldapi/ndarray with crate::simd SIMD dispatch.
/// Feature `attention-table` enables bgz-tensor compiled attention path.
#[derive(Clone, Default, Debug)]
pub struct AdaWorld;

/// CPU device (unit type — there's only one CPU).
#[derive(Clone, Default, Debug, PartialEq, Eq, Hash)]
pub struct CpuDevice;

// NOTE: Full Backend trait implementation requires ~200+ methods across 7 traits.
// This is tracked as a multi-session effort:
//
// Session 1 (current): Crate skeleton + architecture + tensor primitive
// Session 2: FloatTensorOps core (from_data, matmul, add, mul, exp, reshape, transpose)
// Session 3: IntTensorOps + BoolTensorOps
// Session 4: ModuleOps (conv, embedding) + ActivationOps
// Session 5: QTensorOps + TransactionOps + burn-backend-tests
//
// The implementation follows burn-ndarray's pattern but uses:
//   - crate::simd::F32x16 for element-wise ops (not macerator)
//   - LazyLock<SimdDispatch> for runtime tier selection (not compile-time features)
//   - Optional AttentionTable for compiled attention (unique to this backend)
