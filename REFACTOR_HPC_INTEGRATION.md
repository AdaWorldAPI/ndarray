# ndarray hpc/ Module Integration & Transformation Plan

> **Goal**: Transform hpc/ from a bolted-on raw-slice library into a first-class
> citizen of ndarray's type system — without breaking any existing API.
>
> **Principle**: Every refactoring adds an ndarray-native overload or trait impl.
> Raw-slice functions stay (FFI, embedded, zero-overhead). No deletion.

---

## Architectural Diagnosis

The hpc/ module (80K lines, 90+ files) currently operates in two disconnected worlds:

```
┌─────────────────────────────────────────────────────┐
│  ndarray core                                       │
│  Array<f32, Ix2>, ArrayView, Zip, Broadcasting,     │
│  .sum(), .dot(), mapv(), LinalgScalar, BlasFloat    │
└───────────────┬─────────────────────────────────────┘
                │ .as_slice() ← the only bridge
                ▼
┌─────────────────────────────────────────────────────┐
│  hpc/ module                                        │
│  &[u8], &[u64], &[f32], *const u8, Vec<f32>        │
│  Fingerprint<N>, VsaVector, Cascade, EnergyConflict │
│  Custom SIMD dispatch, manual row/col params        │
└─────────────────────────────────────────────────────┘
```

The refactoring creates **bidirectional bridges** without removing the raw layer:

```
┌─────────────────────────────────────────────────────┐
│  ndarray core (unchanged)                           │
│  Array<f32, Ix2>, ArrayView, Zip, Broadcasting      │
└───────────────┬──────────────────────▲──────────────┘
                │                      │
          .as_slice()          backend dispatch
                │                      │
                ▼                      │
┌─────────────────────────────────────────────────────┐
│  hpc/ bridge layer (NEW)                            │
│  Extension traits on ArrayBase<S, D>                │
│  From/Into impls for domain types                   │
│  Core reductions route to SIMD backends             │
└───────────────┬──────────────────────▲──────────────┘
                │                      │
          delegates to          implements
                │                      │
                ▼                      │
┌─────────────────────────────────────────────────────┐
│  hpc/ raw compute (unchanged)                       │
│  &[u8], &[u64], Fingerprint<N>, SIMD dispatch       │
│  K0/K1/K2, BF16 GEMM, VNNI, VML                    │
└─────────────────────────────────────────────────────┘
```

---

## Refactoring Shopping List

### Tier 1: Type Bridge — Domain Types ↔ Array (Low effort, High impact)

These add conversion traits so domain types can participate in ndarray's ecosystem.

---

#### 1.1 Fingerprint<N> ↔ ArrayView1<u64>

**File**: `src/hpc/fingerprint.rs`

**Current state**: `Fingerprint<N>` wraps `[u64; N]`. Provides `.as_bytes()` via
unsafe pointer cast. No connection to ndarray.

**Transform**:

```rust
// --- Zero-copy views (no allocation) ---

impl<const N: usize> Fingerprint<N> {
    /// View this fingerprint as an ndarray 1-D array of u64 words.
    /// Zero-copy: borrows the internal buffer.
    pub fn as_array_view(&self) -> ArrayView1<'_, u64> {
        ArrayView1::from_shape(N, &self.words).unwrap()
    }

    /// Mutable view for in-place operations.
    pub fn as_array_view_mut(&mut self) -> ArrayViewMut1<'_, u64> {
        ArrayViewMut1::from_shape(N, &mut self.words).unwrap()
    }

    /// View as 2-D matrix (e.g., 256 = 32×8 for tiled operations).
    pub fn as_matrix_view(&self, rows: usize, cols: usize) -> ArrayView2<'_, u64> {
        assert_eq!(rows * cols, N);
        ArrayView2::from_shape((rows, cols), &self.words).unwrap()
    }
}

// --- Owned conversions (allocation on From, zero-copy on Into) ---

impl<const N: usize> From<Array1<u64>> for Fingerprint<N> {
    fn from(arr: Array1<u64>) -> Self {
        assert_eq!(arr.len(), N);
        let mut words = [0u64; N];
        words.copy_from_slice(arr.as_slice().unwrap());
        Self { words }
    }
}

impl<const N: usize> From<Fingerprint<N>> for Array1<u64> {
    fn from(fp: Fingerprint<N>) -> Self {
        Array1::from_vec(fp.words.to_vec())
    }
}
```

**Why**: Callers can now pass fingerprints to any function expecting `ArrayView1<u64>`.
Enables `.sum()`, `.mapv()`, Zip, broadcasting on fingerprint data.

**Example — before vs. after**:

```rust
// BEFORE: raw slice, manual loop
let dist = k2_exact(&fp_a.words, &fp_b.words, 256);

// AFTER: can also use ndarray ops directly
let view_a = fp_a.as_array_view();
let view_b = fp_b.as_array_view();
let xor_popcount: u64 = (&view_a ^ &view_b).mapv(|w| w.count_ones() as u64).sum();
```

---

#### 1.2 VsaVector ↔ ArrayView1<u64>

**File**: `src/hpc/vsa.rs`

**Same pattern as 1.1** — VsaVector has `words: [u64; 256]`. Add:

```rust
impl VsaVector {
    pub fn as_array_view(&self) -> ArrayView1<'_, u64> {
        ArrayView1::from_shape(VSA_WORDS, &self.words).unwrap()
    }
    pub fn as_array_view_mut(&mut self) -> ArrayViewMut1<'_, u64> {
        ArrayViewMut1::from_shape(VSA_WORDS, &mut self.words).unwrap()
    }
}

impl From<Array1<u64>> for VsaVector { ... }
impl From<VsaVector> for Array1<u64> { ... }
```

---

#### 1.3 BF16 / F16 as ndarray Element Types

**Files**: `src/hpc/quantized.rs`, `src/backend/mod.rs`

**Current state**: BF16 and F16 implement `ScalarOperand` (can appear in arrays)
but NOT `BlasFloat` (cannot use `.dot()`, BLAS traits). They're second-class citizens.

**Transform** — Register BF16/F16 for BLAS dispatch:

```rust
// In src/backend/mod.rs or src/hpc/quantized.rs:

impl BlasFloat for BF16 {
    fn backend_dot(xs: &[Self], ys: &[Self]) -> Self {
        // Accumulate in f32, return as BF16
        let mut acc = 0.0f32;
        for (&x, &y) in xs.iter().zip(ys.iter()) {
            acc += x.to_f32() * y.to_f32();
        }
        BF16::from_f32(acc)
    }

    fn backend_gemm(m: usize, n: usize, k: usize, alpha: Self,
                    a: &[Self], lda: usize, b: &[Self], ldb: usize,
                    beta: Self, c: &mut [Self], ldc: usize) {
        // Delegate to existing bf16_gemm_f32, convert output back
        crate::hpc::quantized::bf16_gemm_f32(
            a, b,
            /* c_f32 temp */ ...,
            m, n, k, alpha.to_f32(), beta.to_f32()
        );
        // Convert f32 results back to BF16 in c
    }

    fn backend_axpy(alpha: Self, x: &[Self], y: &mut [Self]) { ... }
    fn backend_scal(alpha: Self, x: &mut [Self]) { ... }
    fn backend_nrm2(x: &[Self]) -> Self { ... }
    fn backend_asum(x: &[Self]) -> Self { ... }
}
```

**Impact**: Unlocks this:

```rust
let a: Array2<BF16> = ...;
let b: Array2<BF16> = ...;
let c = a.blas_gemm(BF16::one(), &b, BF16::zero());  // Just works
```

---

#### 1.4 CogRecord ↔ Structured ArrayView

**File**: `src/hpc/cogrecord.rs`

**Current state**: CogRecord has four `Array<u8, Ix1>` fields (btree, cam, embed, meta)
of 4096 bytes each. No combined view.

**Transform**:

```rust
impl CogRecord {
    /// View all 4 channels as a single [4, 4096] matrix.
    pub fn as_channels_view(&self) -> ArrayView2<'_, u8> {
        // Stack the 4 contiguous buffers into a 2-D view
        let total = self.btree.len() + self.cam.len() + self.embed.len() + self.meta.len();
        debug_assert_eq!(total, 4 * CONTAINER_BYTES);
        // NOTE: only works if fields are contiguous in memory.
        // If not, provide a copy-based `to_channels_array()`.
        todo!("requires contiguous layout guarantee")
    }

    /// Get a single channel as ArrayView1<u64> (for Hamming/XOR ops).
    pub fn channel_as_words(&self, channel: usize) -> ArrayView1<'_, u64> {
        let bytes = match channel {
            0 => self.btree.as_slice().unwrap(),
            1 => self.cam.as_slice().unwrap(),
            2 => self.embed.as_slice().unwrap(),
            3 => self.meta.as_slice().unwrap(),
            _ => panic!("channel must be 0..4"),
        };
        // Safe: u8 slice reinterpreted as u64 (alignment checked)
        let words = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const u64,
                bytes.len() / 8,
            )
        };
        ArrayView1::from_shape(words.len(), words).unwrap()
    }
}
```

---

### Tier 2: Trait Extensions — hpc Operations on Array Types (Medium effort, High impact)

Add extension traits so hpc operations feel native on ndarray arrays.

---

#### 2.1 HDC Extension Trait (Hamming, Cascade, K0/K1/K2)

**New file**: `src/hpc/ndarray_ext.rs` (or extend `bitwise.rs`)

**Reasoning**: The K0/K1/K2 pipeline currently takes `&[u64]`. Users with
`Array1<u64>` must manually call `.as_slice().unwrap()`. This trait bridges that.

```rust
use crate::imp_prelude::*;
use super::kernels::{k0_probe, k1_stats, k2_exact, SliceGate, EnergyConflict};

/// HDC (Hyperdimensional Computing) operations on u64 word arrays.
///
/// Provides the K0/K1/K2 rejection pipeline directly on ndarray types.
pub trait HdcOps {
    /// K0 probe: fast 64-bit reject.
    /// Returns true if the first word passes the gate threshold.
    fn k0_probe(&self, candidate: &Self, gate: &SliceGate) -> bool;

    /// K1 stats: 512-bit medium reject.
    /// Returns true if the first 8 words pass the gate threshold.
    fn k1_stats(&self, candidate: &Self, gate: &SliceGate) -> bool;

    /// K2 exact: Full-width Hamming + energy decomposition.
    fn k2_exact(&self, candidate: &Self) -> EnergyConflict;

    /// Full cascade: K0 → K1 → K2 with early rejection.
    fn cascade_distance(&self, candidate: &Self, gate: &SliceGate) -> Option<EnergyConflict>;
}

impl<S> HdcOps for ArrayBase<S, Ix1>
where
    S: Data<Elem = u64>,
{
    fn k0_probe(&self, candidate: &Self, gate: &SliceGate) -> bool {
        k0_probe(self[0], candidate[0], gate)
    }

    fn k1_stats(&self, candidate: &Self, gate: &SliceGate) -> bool {
        let q = self.as_slice().expect("query must be contiguous");
        let c = candidate.as_slice().expect("candidate must be contiguous");
        k1_stats(q, c, gate)
    }

    fn k2_exact(&self, candidate: &Self) -> EnergyConflict {
        let q = self.as_slice().expect("query must be contiguous");
        let c = candidate.as_slice().expect("candidate must be contiguous");
        k2_exact(q, c, q.len())
    }

    fn cascade_distance(&self, candidate: &Self, gate: &SliceGate) -> Option<EnergyConflict> {
        if !self.k0_probe(candidate, gate) { return None; }
        if !self.k1_stats(candidate, gate) { return None; }
        Some(self.k2_exact(candidate))
    }
}
```

**Example usage after**:

```rust
use ndarray::hpc::ndarray_ext::HdcOps;

let query: Array1<u64> = fp_query.into();
let candidate: Array1<u64> = fp_candidate.into();
let gate = SliceGate::for_sku16k();

// Clean ndarray-native API
if let Some(result) = query.cascade_distance(&candidate, &gate) {
    println!("Hamming: {}, Agreement: {}", result.conflict, result.agreement);
}
```

---

#### 2.2 Quantization Extension Trait

**File**: extend `src/hpc/quantized.rs`

**Reasoning**: Quantize/dequantize currently returns `Vec<T>`. Should return `Array1<T>`.

```rust
/// Quantization operations on f32 arrays.
pub trait Quantize {
    /// Quantize to BF16 (element-wise truncation).
    fn to_bf16(&self) -> Array<BF16, Ix1>;

    /// Quantize to symmetric INT8 with scale factor.
    fn to_i8_symmetric(&self) -> (Array<i8, Ix1>, QuantParams);

    /// Quantize per-channel (rows) to INT8.
    fn to_i8_per_channel(&self) -> (Array2<i8>, PerChannelQuantParams)
    where
        Self: AsArray<'_, f32, Ix2>;  // Only for 2-D
}

/// Dequantization operations.
pub trait Dequantize<A> {
    /// Dequantize INT8 codes back to f32.
    fn dequantize_f32(&self, params: &QuantParams) -> Array<f32, Ix1>;
}

impl<S> Quantize for ArrayBase<S, Ix1>
where
    S: Data<Elem = f32>,
{
    fn to_bf16(&self) -> Array<BF16, Ix1> {
        let slice = self.as_slice().expect("must be contiguous");
        let mut out = vec![BF16::ZERO; slice.len()];
        f32_to_bf16_slice(slice, &mut out);  // existing raw function
        Array1::from_vec(out)
    }

    fn to_i8_symmetric(&self) -> (Array<i8, Ix1>, QuantParams) {
        let slice = self.as_slice().expect("must be contiguous");
        let (vec, params) = quantize_f32_to_i8(slice);  // existing raw function
        (Array1::from_vec(vec), params)
    }
}
```

---

#### 2.3 Prefilter — Accept ArrayView2 Instead of (slice, rows, cols)

**File**: `src/hpc/prefilter.rs`

**Current state** (lines 48-295):
```rust
pub fn approx_mean_std_f32(data: &[f32]) -> (f32, f32)
pub fn approx_row_norms_f32(data: &[f32], rows: usize, cols: usize) -> Vec<f32>
pub fn top_k_rows_by_norm(data: &[f32], rows: usize, cols: usize, k: usize) -> Vec<usize>
pub fn approx_column_std(data: &[f32], rows: usize, cols: usize) -> Vec<f32>
```

**Transform** — add ndarray overloads that delegate to existing implementations:

```rust
/// Array-native overloads for prefilter operations.
pub trait Prefilter {
    /// Approximate mean and standard deviation (SIMD).
    fn approx_mean_std(&self) -> (f32, f32);

    /// L2 norm per row (SIMD).
    fn row_norms(&self) -> Array1<f32>;

    /// Indices of top-K rows by L2 norm.
    fn top_k_by_norm(&self, k: usize) -> Array1<usize>;

    /// Standard deviation per column.
    fn column_std(&self) -> Array1<f32>;
}

impl<S> Prefilter for ArrayBase<S, Ix2>
where
    S: Data<Elem = f32>,
{
    fn approx_mean_std(&self) -> (f32, f32) {
        let slice = self.as_slice().expect("must be contiguous");
        approx_mean_std_f32(slice)
    }

    fn row_norms(&self) -> Array1<f32> {
        let slice = self.as_slice().expect("must be contiguous");
        let (rows, cols) = self.dim();
        Array1::from_vec(approx_row_norms_f32(slice, rows, cols))
    }

    fn top_k_by_norm(&self, k: usize) -> Array1<usize> {
        let slice = self.as_slice().expect("must be contiguous");
        let (rows, cols) = self.dim();
        Array1::from_vec(top_k_rows_by_norm(slice, rows, cols, k))
    }

    fn column_std(&self) -> Array1<f32> {
        let slice = self.as_slice().expect("must be contiguous");
        let (rows, cols) = self.dim();
        Array1::from_vec(approx_column_std(slice, rows, cols))
    }
}
```

---

#### 2.4 Activations — Extend to N-D with Axis

**File**: `src/hpc/activations.rs`

**Current state**: Trait only works on `ArrayBase<S, Ix1>`. No per-row softmax.

**Transform**:

```rust
/// Multi-dimensional activations (applied along an axis).
pub trait ActivationsNd<A> {
    /// Softmax along the given axis.
    /// For a [batch, features] matrix with axis=1, computes softmax per row.
    fn softmax_axis(&self, axis: Axis) -> Array<A, Self::Dim>;

    /// Log-softmax along the given axis.
    fn log_softmax_axis(&self, axis: Axis) -> Array<A, Self::Dim>;
}

impl<A, S> ActivationsNd<A> for ArrayBase<S, Ix2>
where
    A: Float + 'static,
    S: Data<Elem = A>,
{
    fn softmax_axis(&self, axis: Axis) -> Array<A, Ix2> {
        let mut result = self.to_owned();
        for mut row in result.axis_iter_mut(axis) {
            let max_val = row.iter().fold(A::neg_infinity(), |a, &b| a.max(b));
            row.mapv_inplace(|v| (v - max_val).exp());
            let sum: A = row.iter().fold(A::zero(), |acc, &v| acc + v);
            row.mapv_inplace(|v| v / sum);
        }
        result
    }

    fn log_softmax_axis(&self, axis: Axis) -> Array<A, Ix2> {
        let mut result = self.to_owned();
        for mut row in result.axis_iter_mut(axis) {
            let max_val = row.iter().fold(A::neg_infinity(), |a, &b| a.max(b));
            row.mapv_inplace(|v| v - max_val);
            let log_sum_exp = row.mapv(|v| v.exp())
                .iter().fold(A::zero(), |acc, &v| acc + v).ln();
            row.mapv_inplace(|v| v - log_sum_exp);
        }
        result
    }
}
```

**Impact**: Enables `batch_logits.softmax_axis(Axis(1))` — critical for inference.

---

#### 2.5 VML — SIMD Transcendentals as mapv Acceleration

**File**: `src/hpc/vml.rs` (2543 lines)

**Current state**: 12+ standalone functions like `vsexp(x: &[f32], out: &mut [f32])`.

**Transform** — add `ArrayBase` methods that auto-dispatch to SIMD:

```rust
/// SIMD-accelerated element-wise math on contiguous f32 arrays.
///
/// These call the VML SIMD kernels when the array is contiguous,
/// falling back to scalar mapv() otherwise.
pub trait SimdMath {
    fn simd_exp(&self) -> Array<f32, Self::Dim>;
    fn simd_ln(&self) -> Array<f32, Self::Dim>;
    fn simd_sqrt(&self) -> Array<f32, Self::Dim>;
    fn simd_sin(&self) -> Array<f32, Self::Dim>;
    fn simd_cos(&self) -> Array<f32, Self::Dim>;
    fn simd_tanh(&self) -> Array<f32, Self::Dim>;
    fn simd_abs(&self) -> Array<f32, Self::Dim>;
}

impl<S, D> SimdMath for ArrayBase<S, D>
where
    S: Data<Elem = f32>,
    D: Dimension,
{
    fn simd_exp(&self) -> Array<f32, D> {
        if let Some(slice) = self.as_slice() {
            let mut out = vec![0.0f32; slice.len()];
            vsexp(slice, &mut out);  // existing SIMD kernel
            Array::from_shape_vec(self.raw_dim(), out).unwrap()
        } else {
            self.mapv(|x| x.exp())
        }
    }

    fn simd_ln(&self) -> Array<f32, D> {
        if let Some(slice) = self.as_slice() {
            let mut out = vec![0.0f32; slice.len()];
            vsln(slice, &mut out);
            Array::from_shape_vec(self.raw_dim(), out).unwrap()
        } else {
            self.mapv(|x| x.ln())
        }
    }
    // ... same pattern for sqrt, sin, cos, tanh, abs
}
```

**Benchmark expectation**: 10–50x for transcendentals (polynomial SIMD vs scalar libm).

---

### Tier 3: Backend Integration — Core Dispatch to SIMD (Medium effort, Very High impact)

Make ndarray's **built-in** operations (`.sum()`, `.mean()`, `.dot()`) automatically
use hpc SIMD kernels for contiguous arrays.

---

#### 3.1 Wire Reductions into Core

**Files**: `src/numeric/impl_numeric.rs` (core sum/mean) + `src/hpc/reductions.rs`

**Current state**: `Array.sum()` uses `numeric_util::unrolled_fold()` — a scalar
4x unrolled loop. `hpc::reductions::sum_f32()` uses F32x16 (16-lane AVX-512)
with 4-accumulator ILP. They never talk to each other.

**Transform** — add backend dispatch in core's sum():

```rust
// In src/numeric/impl_numeric.rs or src/backend/native.rs:

impl<A, S, D> ArrayBase<S, D>
where
    A: Clone + num_traits::Zero + std::ops::Add<Output = A>,
    S: Data<Elem = A>,
    D: Dimension,
{
    pub fn sum(&self) -> A {
        // Fast path: contiguous f32 → SIMD reduction
        if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f32>() {
            if let Some(slice) = self.as_slice_memory_order() {
                // SAFETY: TypeId check guarantees A == f32
                let f32_slice = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const f32, slice.len())
                };
                let result = crate::hpc::reductions::sum_f32(f32_slice);
                // SAFETY: result is f32, A is f32
                return unsafe { std::mem::transmute_copy(&result) };
            }
        }
        if std::any::TypeId::of::<A>() == std::any::TypeId::of::<f64>() {
            if let Some(slice) = self.as_slice_memory_order() {
                let f64_slice = unsafe {
                    std::slice::from_raw_parts(slice.as_ptr() as *const f64, slice.len())
                };
                let result = crate::hpc::reductions::sum_f64(f64_slice);
                return unsafe { std::mem::transmute_copy(&result) };
            }
        }
        // Fallback: generic fold
        self.fold(A::zero(), |acc, &x| acc + x)
    }
}
```

**Impact**: Every `.sum()` on a contiguous f32/f64 array silently becomes 16x
faster. Zero API change for users.

**Same pattern for**: `.mean()`, internal `max()`/`min()` in softmax paths.

---

#### 3.2 Unify SIMD Detection (Delete Duplicates)

**Files to merge**: `src/hpc/simd_dispatch.rs` (362 lines), `src/hpc/simd_caps.rs` (515 lines)
**Into**: `src/simd.rs` (the existing core SIMD module)

**Current state**: Three overlapping detection systems:
1. `src/simd.rs` → `LazyLock<Tier>` — detects AVX-512/AVX2/NEON
2. `src/hpc/simd_caps.rs` → `CpuCaps` struct — detects same capabilities
3. `src/hpc/simd_dispatch.rs` → `SimdDispatch` — another LazyLock with function pointers

**Transform**:

```rust
// 1. In src/simd.rs, export the unified detection:
pub static CPU_CAPS: LazyLock<CpuCaps> = LazyLock::new(|| detect_cpu_caps());

pub struct CpuCaps {
    pub tier: Tier,  // existing
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vpopcntdq: bool,
    pub avx512vnni: bool,
    pub avx2: bool,
    pub fma: bool,
    pub popcnt: bool,
    // ARM
    pub neon: bool,
    pub sve: bool,
}

// 2. In src/hpc/simd_dispatch.rs, replace with:
pub use crate::simd::CPU_CAPS;
// Keep function-pointer dispatch but source caps from unified singleton

// 3. Delete src/hpc/simd_caps.rs (or make it a thin re-export)
```

**Result**: One CPUID check, one atomic, one struct — shared by core and all hpc modules.

---

#### 3.3 VNNI GEMM via BlasFloat Dispatch

**Files**: `src/hpc/vnni_gemm.rs`, `src/hpc/quantized.rs:618`, `src/backend/mod.rs`

**Current state**: `int8_gemm_vnni()` takes raw `&[u8], &[i8], &mut [i32]` with manual
m/n/k parameters. Not accessible via `Array.dot()` or `blas_gemm()`.

**Transform** — wire INT8 GEMM into the backend trait system:

```rust
// New trait in src/backend/mod.rs:
pub trait BlasInt8 {
    /// INT8 GEMM: C_i32 = A_u8 × B_i8
    fn backend_gemm_i8(
        m: usize, n: usize, k: usize,
        a: &[u8], lda: usize,
        b: &[i8], ldb: usize,
        c: &mut [i32], ldc: usize,
    );
}

// Extension trait on Array:
pub trait Int8Matmul {
    /// Quantized matrix multiply: self (u8) × rhs (i8) → Array2<i32>
    fn matmul_i8(&self, rhs: &ArrayView2<i8>) -> Array2<i32>;
}

impl<S> Int8Matmul for ArrayBase<S, Ix2>
where
    S: Data<Elem = u8>,
{
    fn matmul_i8(&self, rhs: &ArrayView2<i8>) -> Array2<i32> {
        let (m, k) = self.dim();
        let (k2, n) = rhs.dim();
        assert_eq!(k, k2, "inner dimensions must match");

        let a_slice = self.as_slice().expect("A must be contiguous");
        let b_slice = rhs.as_slice().expect("B must be contiguous");
        let mut c = vec![0i32; m * n];

        crate::hpc::vnni_gemm::int8_gemm_vnni(a_slice, b_slice, &mut c, m, n, k);
        Array2::from_shape_vec((m, n), c).unwrap()
    }
}
```

**Example**:

```rust
let weights: Array2<i8> = load_quantized_weights();
let activations: Array2<u8> = quantize_input(&input);
let output: Array2<i32> = activations.matmul_i8(&weights.view());
// 4× throughput vs f32 GEMM
```

---

### Tier 4: View Factories — Arrow & External Data (Low effort, Medium impact)

Make hpc's data ingestion return ndarray views instead of raw slices.

---

#### 4.1 Arrow Bridge → ArrayView

**File**: `src/hpc/arrow_bridge.rs`

**Current state**: `binary_entry()` returns `&[u8]`. `SoakingBuffer` provides
`as_columnar_slice()` as `&[i8]`.

**Transform**:

```rust
impl ArrowBridge {
    /// Get one binary entry as a 1-D array view.
    pub fn binary_entry_view(&self, idx: usize) -> ArrayView1<'_, u8> {
        let bytes = self.binary_entry(idx);
        ArrayView1::from_shape(bytes.len(), bytes).unwrap()
    }

    /// Get the full binary column as a 2-D view [n_rows, entry_bytes].
    pub fn binary_column_view(&self) -> ArrayView2<'_, u8> {
        let data = self.as_binary_slice();
        let entry_len = self.entry_len();
        let n_rows = data.len() / entry_len;
        ArrayView2::from_shape((n_rows, entry_len), data).unwrap()
    }
}

impl SoakingBuffer {
    /// View the columnar buffer as a 2-D matrix [n_entries, n_dims].
    pub fn as_array_view_2d(&self) -> ArrayView2<'_, i8> {
        let data = self.as_columnar_slice();
        ArrayView2::from_shape((self.n_entries, self.n_dims), data).unwrap()
    }
}
```

**Impact**: Downstream code can use ndarray broadcasting, slicing, and Zip
on Arrow data without copying.

---

#### 4.2 Distance Functions → Return Array1

**File**: `src/hpc/distance.rs`

**Current state**: Returns `Vec<f32>`.

**Transform**:

```rust
// Keep existing:
pub fn squared_distances_f32(query: [f32; 3], points: &[[f32; 3]]) -> Vec<f32> { ... }

// Add ndarray overload:
pub fn squared_distances_array(query: &ArrayView1<f32>, points: &ArrayView2<f32>) -> Array1<f32> {
    assert_eq!(query.len(), points.ncols());
    let n = points.nrows();
    let mut result = Array1::zeros(n);
    for (i, row) in points.axis_iter(Axis(0)).enumerate() {
        let diff = &row - query;
        result[i] = diff.mapv(|x| x * x).sum();
    }
    result
}
```

---

### Tier 5: Zip & Parallel — Modernize Inner Loops (Medium effort, Medium impact)

Replace manual `iter_mut().zip()` with ndarray's `Zip` for vectorization hints
and future `par_for_each()` support.

---

#### 5.1 VML Tail Loops → Zip

**File**: `src/hpc/vml.rs` — 12+ sites

**Current pattern** (repeated throughout):
```rust
for (o, &v) in out[tail_start..].iter_mut().zip(x[tail_start..].iter()) {
    *o = v.exp();
}
```

**Transform**:
```rust
use crate::Zip;

let out_view = ArrayViewMut1::from_shape(n - tail_start, &mut out[tail_start..]).unwrap();
let x_view = ArrayView1::from_shape(n - tail_start, &x[tail_start..]).unwrap();
Zip::from(&mut out_view).and(&x_view).for_each(|o, &v| {
    *o = v.exp();
});
```

**Benefit**: Zip provides SIMD-friendlier iteration order and enables
future `par_for_each()` without restructuring.

---

#### 5.2 Hamming Batch → Parallel Zip

**File**: `src/hpc/bitwise.rs` lines 320-333

**Current state**: Sequential loop over candidates.

**Transform** (feature-gated on `rayon`):

```rust
#[cfg(feature = "rayon")]
pub fn hamming_batch_parallel(
    query: &ArrayView1<u8>,
    database: &ArrayView2<u8>,  // [n_candidates, vec_len]
) -> Array1<u64> {
    use rayon::prelude::*;
    let q = query.as_slice().unwrap();
    let results: Vec<u64> = database
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| {
            let r = row.as_slice().unwrap();
            dispatch_hamming(q, r)
        })
        .collect();
    Array1::from_vec(results)
}
```

---

### Tier 6: Multi-Dimensional Reductions (High effort, High impact for ML)

Lift 1-D slice operations to full N-D axis reductions.

---

#### 6.1 SIMD Axis Reductions

**File**: `src/hpc/reductions.rs`

**Current state**: Only 1-D: `sum_f32(&[f32])`, `mean_f32(&[f32])`, etc.

**Transform**:

```rust
/// Sum along an axis, using SIMD for contiguous lanes.
pub fn sum_axis_f32<S, D>(arr: &ArrayBase<S, D>, axis: Axis) -> Array<f32, D::Smaller>
where
    S: Data<Elem = f32>,
    D: Dimension + RemoveAxis,
{
    let mut result = Array::zeros(arr.raw_dim().remove_axis(axis));
    for (i, lane) in arr.lanes(axis).into_iter().enumerate() {
        if let Some(slice) = lane.as_slice() {
            result.as_slice_mut().unwrap()[i] = sum_f32(slice);
        } else {
            result.as_slice_mut().unwrap()[i] = lane.iter().fold(0.0, |a, &b| a + b);
        }
    }
    result
}
```

---

#### 6.2 Statistics Trait — Generalize to N-D

**File**: `src/hpc/statistics.rs`

**Transform**: Make `variance`, `std_dev`, `percentile` work along axes:

```rust
pub trait StatisticsNd<A> {
    type Reduced;

    fn variance_axis(&self, axis: Axis, ddof: usize) -> Self::Reduced;
    fn std_axis(&self, axis: Axis, ddof: usize) -> Self::Reduced;
    fn percentile_axis(&self, axis: Axis, q: f64) -> Self::Reduced;
}
```

---

## Execution Order & Dependencies

```
Phase A (2 weeks) — Type Bridges:
  1.1 Fingerprint ↔ ArrayView1      (standalone, no deps)
  1.2 VsaVector ↔ ArrayView1        (standalone)
  1.3 BF16/F16 → BlasFloat          (requires backend/mod.rs edit)
  1.4 CogRecord channel views       (standalone)

Phase B (2 weeks) — Extension Traits:
  2.1 HdcOps trait (K0/K1/K2)       (depends on 1.1)
  2.2 Quantize trait                 (depends on 1.3)
  2.3 Prefilter trait                (standalone)
  2.4 ActivationsNd (axis softmax)   (standalone)
  2.5 SimdMath trait (VML)           (standalone)

Phase C (2 weeks) — Backend Wiring:
  3.1 Core sum/mean → SIMD dispatch  (depends on 3.2)
  3.2 Unified SIMD detection         (standalone, delete duplicates)
  3.3 INT8 Matmul via trait          (depends on 1.3 pattern)

Phase D (1 week) — View Factories:
  4.1 Arrow → ArrayView              (standalone)
  4.2 Distance → Array1 returns      (standalone)

Phase E (1 week) — Modernize Loops:
  5.1 VML tails → Zip                (standalone)
  5.2 Hamming batch → parallel       (standalone)

Phase F (2 weeks) — N-D Reductions:
  6.1 SIMD axis reductions           (depends on 3.1)
  6.2 Statistics N-D                  (depends on 6.1)
```

---

## Rules of Engagement

1. **No deletion** — raw-slice functions stay. They serve FFI, embedded, and hot-path callers
   who've already pre-validated contiguity.

2. **Extension traits, not modifications** — new traits in new files or at the bottom of
   existing files. Existing public APIs unchanged.

3. **Contiguous fast-path, generic fallback** — every new trait method checks
   `as_slice()` first. If contiguous → SIMD dispatch. If not → `mapv()`/`iter()`.

4. **Feature-gate expensive additions** — rayon parallel (`rayon` feature),
   BlasFloat for BF16 (`bf16-blas` feature if controversial).

5. **Tests mirror existing** — each new trait method gets the same test coverage
   as the raw-slice version it wraps.

6. **Naming convention** — ndarray-native methods use `snake_case` matching ndarray
   style (`.softmax_axis()`, `.simd_exp()`). Existing raw functions keep their
   current names (`softmax_f32()`, `vsexp()`).

---

## Impact Matrix

| Item | Lines Changed | New Lines | Performance Impact | Ergonomics Impact |
|------|--------------|-----------|-------------------|-------------------|
| 1.1 Fingerprint bridge | 0 | ~40 | Zero (zero-copy) | High — unblocks Zip/broadcast |
| 1.3 BF16 BlasFloat | ~10 | ~80 | Enables BF16 GEMM via .dot() | Very High — removes type barrier |
| 2.1 HdcOps trait | 0 | ~60 | Zero (delegates) | High — clean pipeline API |
| 2.5 SimdMath (VML) | 0 | ~100 | 10-50x for exp/ln/sin | Very High — transparent SIMD |
| 3.1 Core sum → SIMD | ~20 | ~40 | 16x for f32 sum/mean | Zero (invisible) |
| 3.2 Unified detection | -877 | ~50 | Zero (dedup) | N/A (internal) |
| 3.3 INT8 matmul trait | 0 | ~60 | 4x over f32 GEMM | High — quantized inference |
| 5.2 Parallel hamming | 0 | ~30 | N×cores speedup | Medium |
| 6.1 SIMD axis reductions | 0 | ~80 | 16x per lane | High — ML training |

---

## What This Achieves

After this refactoring, hpc/ is no longer a "second library living inside ndarray."
It becomes the **performance substrate** that ndarray's own operations dispatch to:

```rust
// User code — completely ndarray-native, no hpc imports needed:
let a: Array2<f32> = load_embeddings();
let b: Array2<f32> = load_queries();

let sims = a.dot(&b.t());           // → routes to SIMD GEMM internally
let topk = sims.sum_axis(Axis(1));   // → routes to SIMD sum internally
let probs = sims.softmax_axis(Axis(1));  // → routes to SIMD exp internally

// User code — HDC domain, imports hpc trait:
use ndarray::hpc::ndarray_ext::HdcOps;
let fp: Array1<u64> = Fingerprint::<256>::random(42).into();
let candidates: Array2<u64> = load_database();

for row in candidates.axis_iter(Axis(0)) {
    if let Some(ec) = fp.cascade_distance(&row, &gate) {
        hits.push(ec);
    }
}
```

The raw-slice layer (`hpc::kernels::k2_exact(&[u64], ...)`) remains for:
- FFI exports (Python, C)
- Embedded/no-std contexts
- Callers who've already validated layout and want zero overhead
- Benchmarks that isolate kernel performance from array metadata

Both worlds coexist. The bridge is the extension traits.
