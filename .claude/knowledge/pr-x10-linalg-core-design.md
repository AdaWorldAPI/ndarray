# PR-X10 — `ndarray::hpc::linalg` core — the shared middle layer below LAPACK

> READ BY: every agent that touches matrix math
> (savant-architect, l3-strategist, cascade-architect, splat3d-architect,
> cognitive-architect, jc-architect, training-architect, arm-neon-specialist,
> sentinel-qa, product-engineer, vector-synthesis, truth-architect).
>
> Status: design v1 — drafted 2026-05-18 in response to the cross-cutting
> gap analysis: splat3d's Spd3, jc's three Spd2/Spd3 copies, and the
> inference modules' inlined RMSNorm/SiLU all hand-roll math that should
> live in a single canonical module.
>
> Parallel docs:
> - `.claude/knowledge/pr-arithmetic-inventory.md` — the per-layer math inventory this consolidates
> - `.claude/knowledge/pr-x3-cognitive-grid-design.md` — BlockedGrid substrate (shipped)
> - `.claude/knowledge/pr-x4-design.md` — Gaussian splat cascade
> - `.claude/knowledge/pr-x9-design.md` — lazy basin-codebook storage

## Why PR-X10 exists — the strategic frame

**The biggest gap in the stack is shared linear-algebra below LAPACK.** ndarray has BLAS L1/L2/L3 (`hpc::blas_level{1,2,3}`) and LAPACK wrappers (`hpc::lapack.rs`: LU, Cholesky, QR — all FFI-wrapped, no SIMD). The middle layer — SVD, general N×N eig, polar decomposition, mat_exp/mat_log, quaternion algebra, fused inference primitives (RMSNorm, SiLU, RoPE, attention) — is hand-rolled per consumer:

- **splat3d** ships its own `Spd3` with bespoke Smith-1961 eig + sandwich (shipped PR #153).
- **lance-graph jc** has **three** separate Spd2/Spd3 definitions: `ewa_sandwich.rs`, `ewa_sandwich_3d.rs`, `koestenberger.rs`. Each has its own eig, pow, sqrt, log_spd, frobenius_sq, det, sandwich. The doc-comment in `ewa_sandwich.rs:99–102` already flagged the consolidation need: *"promotion to a shared hadamard module would be the right cleanup once a 4th consumer appears."* The 4th consumer is here (cognitive cascade in PR-X4).
- **`hpc::{gpt2, openchat, stable_diffusion}`** inline RMSNorm, SiLU, RoPE, attention because there's no canonical fn.

Consolidating these into one `ndarray::hpc::linalg` module unblocks **three downstream sprints simultaneously**:
1. **splat3d backward / training** — needs general symmetric eig + SVD for Σ = R·diag(s²)·Rᵀ reparameterization gradients
2. **openchat/gpt2/stable_diffusion finalization** — needs LayerNorm/RMSNorm, SiLU/GELU, RoPE, batched matmul, Conv1D/Conv2D
3. **jc Pillars 8–11** — needs higher-D Spd carriers, Wasserstein/Sinkhorn, signature transform, manifold log/exp

That's a 3× force multiplier on the linalg work. PR-X10 is the highest-leverage non-cognitive-shader sprint in the queue.

## Module layout — `crate::hpc::linalg::*`

```
src/hpc/linalg/
├── mod.rs                — pub surface; submodule decls + re-exports
├── matrix.rs             — MatN<const N: usize> carrier, repr(C, align(64))
├── inverse.rs            — 3×3 / 4×4 specialized; general LU-backsolve
├── eig_sym.rs            — symmetric N×N eigendecomp (Jacobi for N≤8, QR for N>8)
├── svd.rs                — Golub-Reinsch + one-sided Jacobi SVD
├── polar.rs              — A = U·P decomposition (built on SVD)
├── matfn.rs              — mat_exp + mat_log (Padé + scaling-and-squaring)
├── quat.rs               — Quat carrier + algebra (mul, conjugate, slerp, from_axis_angle, to_mat)
├── sh.rs                 — extended SH (deg 0..=7) — supersedes splat3d/sh.rs deg-3 only
├── conv.rs               — Conv1D + Conv2D (im2col + gemm path, direct path for small kernels)
├── attention.rs          — fused Q·Kᵀ/√d → softmax → ·V; supports causal mask, RoPE
├── norm.rs               — LayerNorm + RMSNorm + GroupNorm
├── activations_ext.rs    — GELU + SiLU + Swish + Mish (supplements existing sigmoid/softmax)
├── rope.rs               — rotary position embeddings (Llama/Qwen3/Mistral standard)
├── batched.rs            — batched gemm over [batch, ...] axes
└── tests/                — unit + property + parity tests
```

Plus extending `crate::hpc::vml.rs` with `erf`, `gamma`, `beta`, `Bessel{j0,j1,jn}` (Tier 3 from the gap analysis) — keep there since they're scalar special functions, not matrix ops.

## Tier 1: blocking splat3d backward + training sprint

### Quaternion algebra — `linalg::quat::Quat`

```rust
#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct Quat {
    pub w: f32, pub x: f32, pub y: f32, pub z: f32,
}

impl Quat {
    pub const I: Self = Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    pub fn from_axis_angle(axis: [f32; 3], radians: f32) -> Self { ... }
    pub fn from_mat(r: &Mat3) -> Self { ... }   // Shepperd's method with sign tracking
    pub fn to_mat(&self) -> Mat3 { ... }

    pub fn conjugate(&self) -> Self { ... }     // (w, -x, -y, -z)
    pub fn inverse(&self) -> Self { ... }       // conjugate / norm²
    pub fn normalize(&self) -> Self { ... }
    pub fn norm_sq(&self) -> f32 { ... }
    pub fn dot(&self, other: &Self) -> f32 { ... }

    pub fn mul(&self, other: &Self) -> Self { ... }    // Hamilton product
    pub fn rotate_vec(&self, v: [f32; 3]) -> [f32; 3] { ... }
    pub fn slerp(&self, other: &Self, t: f32) -> Self { ... }  // spherical linear interp
}

/// Batched 16-wide quaternion multiply for the splat3d backward pass.
pub fn quat_mul_x16(a: &[Quat; 16], b: &[Quat; 16], out: &mut [Quat; 16]) { ... }
```

**Precision class: EXACT** for `normalize`, `slerp`, `from_axis_angle` (uses precise sin/cos from `crate::hpc::vml::sin_f32`). The splat3d training sprint needs `quat_mul` for parameter updates without quaternion drift; `slerp` for camera-path interpolation; `from_axis_angle` for angular-velocity integration.

### Matrix inverse — `linalg::inverse`

```rust
pub fn invert_mat3(a: &Mat3) -> Option<Mat3> { ... }  // closed-form via adjugate / det
pub fn invert_mat4(a: &Mat4) -> Option<Mat4> { ... }  // closed-form via cofactor expansion
pub fn invert_mat_n<const N: usize>(a: &MatN<N>) -> Option<MatN<N>> { ... }  // LU + back-solve

/// Camera view-matrix inversion specialized for affine 4×4 (R | t) → (Rᵀ | -Rᵀ·t)
pub fn invert_affine_4x4(view: &Mat4) -> Mat4 { ... }
```

The closed-form 3×3 and 4×4 paths are ~30 and ~70 ops respectively — much faster than LU for the splat3d projection per-frame view-inverse. **EXACT** precision class.

### Symmetric eigendecomposition — `linalg::eig_sym`

```rust
pub fn eig_sym_n<const N: usize>(a: &MatN<N>) -> (LambdaN<N>, MatN<N>) { ... }

// Specialized fast paths for the common splat3d / inference / jc cases:
pub fn eig_sym_2(a: &Spd2) -> (f32, f32, [[f32; 2]; 2]) { ... }
pub fn eig_sym_3(a: &Spd3) -> (f32, f32, f32, [[f32; 3]; 3]) { ... }  // Smith-1961 (reused from splat3d)
pub fn eig_sym_4(a: &Spd4) -> (f32, f32, f32, f32, [[f32; 4]; 4]) { ... }  // Ferrari closed-form

pub fn eig_sym_jacobi<const N: usize>(a: &MatN<N>, max_sweeps: u32, eps: f32) -> ... { ... }
pub fn eig_sym_qr<const N: usize>(a: &MatN<N>, max_iters: u32, eps: f32) -> ... { ... }
```

Algorithm choice gates:
- **N ∈ {2, 3, 4}**: closed-form (Smith-1961 for 3, Ferrari for 4)
- **N ∈ [5, 64]**: Jacobi rotations (O(N⁴) but cache-friendly, parallel-rotation-friendly)
- **N > 64**: QR with implicit shifts (O(N³))

**Precision class: EXACT** for closed-form; **VERIFY** for Jacobi/QR (convergence tolerance is parameter-dependent).

### SVD — `linalg::svd`

```rust
pub struct Svd<const M: usize, const N: usize> {
    pub u: MatN<M>,
    pub s: [f32; min(M, N)],
    pub vt: MatN<N>,
}

pub fn svd<const M: usize, const N: usize>(a: &Mat<M, N>) -> Svd<M, N> { ... }
pub fn svd_one_sided<const M: usize, const N: usize>(a: &Mat<M, N>) -> Svd<M, N> { ... }
pub fn svd_thin<const M: usize, const N: usize>(a: &Mat<M, N>) -> Svd<M, N> { ... }
```

Algorithm: Golub-Reinsch (bidiagonalization + implicit QR on bidiagonal) for general; one-sided Jacobi for high-accuracy small-N (≤16). One-sided is the natural SIMD choice — rotations are independent across columns.

**Precision class: VERIFY** — Golub-Reinsch convergence depends on shift heuristic; one-sided Jacobi is exact-up-to-ULP but O(N³).

### Polar decomposition — `linalg::polar`

```rust
pub struct Polar<const N: usize> {
    pub u: MatN<N>,        // orthogonal
    pub p: MatN<N>,        // SPD
}

pub fn polar<const N: usize>(a: &MatN<N>) -> Polar<N> { ... }
```

Built on SVD: A = U·Σ·Vᵀ = (U·Vᵀ)·(V·Σ·Vᵀ). The orthogonal part is U·Vᵀ; the SPD part is V·Σ·Vᵀ. **EXACT** given SVD precision class. Used for extracting rigid motion from a general 3×3, anti-aliasing camera transforms, and orthogonality-restoring projection in iterative training.

### Matrix exp / log — `linalg::matfn`

```rust
pub fn mat_exp<const N: usize>(a: &MatN<N>) -> MatN<N> { ... }    // Padé + scaling-and-squaring
pub fn mat_log<const N: usize>(a: &MatN<N>) -> MatN<N> { ... }    // Inverse: log(exp(A)) = A on Lie algebra

pub fn mat_exp_spd<const N: usize>(a: &MatN<N>) -> MatN<N> { ... } // via eigendecomp; faster on SPD
pub fn mat_log_spd<const N: usize>(a: &MatN<N>) -> MatN<N> { ... } // via eigendecomp; SPD-preserving
```

Higham's scaling-and-squaring Padé(13/13) for general matrices (3 × ε_machine accurate). SPD specialization via eigendecomp + scalar exp/log is faster (~3× for small N) and preserves SPD-cone membership exactly.

**Precision class: EXACT** for SPD path (via `eig_sym` + scalar `vml::exp_f32`/`vml::ln_f32`); **VERIFY** for general path (Padé approximant order vs scaling depth trade-off).

### Higher-degree SH — `linalg::sh`

Supersedes `splat3d::sh.rs` (which ships deg-3 only). Adds deg-4 through deg-7:

| Degree | Basis count | Coeffs per gaussian (× RGB) | Use case |
|---|---|---|---|
| 0 | 1 | 3 | uniform color |
| 1 | 4 | 12 | direct-lighting bias |
| 2 | 9 | 27 | ambient occlusion |
| 3 | 16 | 48 | splat3d default (Inria spec) |
| 4 | 25 | 75 | research scenes with sharper specular |
| 5 | 36 | 108 | audio HRTF, high-fidelity scene capture |
| 6 | 49 | 147 | (rarely used) |
| 7 | 64 | 192 | (rarely used; matches 1 cache line at f32) |

**Mechanical extension** of the existing `SH_C0..SH_C3` constants tables. Per-channel cost: ~basis_count FMA. The deg-7 evaluation fits exactly in one AVX-512 register (64 f32 = 256 bytes = 4 zmm), so it's actually the SIMD-friendliest tier.

### Backward / autodiff primitives for splat3d — `linalg::splat_grad`

Stub for the splat3d training sprint. Designed in PR-X10 but implemented in a separate follow-on (training sprint owns it). The API surface that PR-X10 commits to:

```rust
/// Gradient through the EWA projection: ∂L/∂Σ_world, ∂L/∂μ_world from ∂L/∂(Σ_image, screen_pos).
pub fn project_backward<...>(...) -> ... { unimplemented!("training sprint") }

/// Gradient through alpha-compose: ∂L/∂α, ∂L/∂color from ∂L/∂framebuffer.
pub fn raster_backward<...>(...) -> ... { unimplemented!("training sprint") }

/// Gradient through SH eval: ∂L/∂sh_coeffs, ∂L/∂view_dir from ∂L/∂rgb.
pub fn sh_backward<const DEG: usize>(...) -> ... { unimplemented!("training sprint") }
```

The signature freeze is the deliverable; impl is the training sprint's job. Without this freeze, the training sprint blocks on API design.

## Tier 2: blocking the model-inference modules

### Conv1D / Conv2D — `linalg::conv`

```rust
pub fn conv1d_f32(input: &[f32], kernel: &[f32], stride: usize, padding: usize, out: &mut [f32]) { ... }
pub fn conv2d_f32(input: &Tensor3, kernel: &Tensor4, stride: (usize, usize), padding: (usize, usize), out: &mut Tensor3) { ... }

/// Specialized small-kernel direct convolution (3×3, 5×5) — avoids im2col overhead.
pub fn conv2d_3x3_f32(input: &Tensor3, kernel: &Tensor4, out: &mut Tensor3) { ... }
pub fn conv2d_5x5_f32(...) { ... }

/// General-kernel via im2col + gemm (calls into hpc::blas_level3::gemm_f32).
pub fn conv2d_im2col_f32(...) { ... }
```

Required by `stable_diffusion.rs` (UNet has 3×3 convs throughout). Currently inlined; consolidate.

### Batched matmul — `linalg::batched`

```rust
/// Batched gemm: Z[b, i, j] = sum_k X[b, i, k] · Y[b, k, j]
/// for all b in 0..batch.
pub fn batched_gemm_f32(
    x: &TensorView3,  // [batch, M, K]
    y: &TensorView3,  // [batch, K, N]
    out: &mut TensorViewMut3,  // [batch, M, N]
    alpha: f32, beta: f32,
);

/// 4-axis variant for attention: [batch, heads, seq, dim]
pub fn batched_gemm_4d_f32(...);
```

Required by every attention kernel (Q·Kᵀ over `[batch, heads, seq, dim]`). Currently each consumer iterates `gemm_f32` in a loop, missing the cache-locality win of fusing the batch axis.

### LayerNorm / RMSNorm / GroupNorm — `linalg::norm`

```rust
pub fn layer_norm_f32(x: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) { ... }
pub fn rms_norm_f32(x: &mut [f32], gamma: &[f32], eps: f32) { ... }
pub fn group_norm_f32(x: &mut [f32], gamma: &[f32], beta: &[f32], groups: usize, eps: f32) { ... }

/// Batched variants (no allocation, in-place over the batch axis).
pub fn rms_norm_batched_f32(x: &mut TensorView2, gamma: &[f32], eps: f32) { ... }
```

`RMSNorm` is what Mistral-7B / Qwen3 / Llama use; `openchat.rs` currently inlines it.

### Activations — `linalg::activations_ext`

Supplements existing `hpc::activations.rs` (sigmoid, softmax, log_softmax):

```rust
pub fn gelu_f32(x: &mut [f32]) { ... }       // GPT-2 / BERT
pub fn gelu_tanh_f32(x: &mut [f32]) { ... }  // Hendrycks tanh approximation
pub fn silu_f32(x: &mut [f32]) { ... }       // Mistral / Qwen3 / Llama — x · sigmoid(x)
pub fn swish_f32(x: &mut [f32], beta: f32) { ... }  // generalized SiLU
pub fn mish_f32(x: &mut [f32]) { ... }       // x · tanh(softplus(x))
```

All AVX-512 batched via existing `crate::simd::F32x16` polyfill. **Precision class: VERIFY** for tanh-approximated GELU (Hendrycks-tanh has 1e-3 max abs error vs erf-exact); EXACT for SiLU after correct sigmoid.

### RoPE — `linalg::rope`

```rust
pub struct RopeCache {
    pub cos_table: Vec<f32>,
    pub sin_table: Vec<f32>,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl RopeCache {
    pub fn build(head_dim: usize, max_seq_len: usize, theta: f32) -> Self { ... }

    /// Apply RoPE in-place to query and key tensors.
    /// Q, K shape: [batch, seq, heads, head_dim]
    pub fn apply_qk_f32(&self, q: &mut TensorView4, k: &mut TensorView4, positions: &[u32]) { ... }
}
```

Standard rotary embedding for Llama / Mistral / Qwen3 / GPT-NeoX. The cache is built once per (head_dim, max_seq_len) pair; application is ~2 FMA per element. **EXACT** precision (table lookup of pre-computed cos/sin).

### Attention as a single primitive — `linalg::attention`

```rust
pub struct AttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub causal_mask: bool,
    pub rope: Option<RopeCache>,
}

/// Fused multi-head attention: softmax(Q·Kᵀ/√d + mask) · V
/// Q, K, V shape: [batch, seq, heads, head_dim]
pub fn attention_f32(
    q: &TensorView4, k: &TensorView4, v: &TensorView4,
    config: &AttentionConfig,
    out: &mut TensorViewMut4,
);

/// Flash-attention-style tiled variant — keeps the [seq, seq] intermediate out of memory.
pub fn flash_attention_f32(...);
```

The flash-attention variant is the differentiator: it processes attention in `[Br, Bc]` tiles using only O(N) memory instead of O(N²). Standard implementation pattern (Dao 2022).

### Cross-entropy + softmax-backward — `linalg::loss`

```rust
pub fn cross_entropy_with_logits_f32(logits: &[f32], targets: &[u32], out_loss: &mut f32) { ... }
pub fn cross_entropy_with_logits_batched_f32(...);

/// Fused softmax + cross-entropy + backward in one pass — the canonical training-loop primitive.
pub fn softmax_xent_backward_f32(logits: &[f32], targets: &[u32], grad_out: &mut [f32]) { ... }
```

Training-side; standard fused kernel. **EXACT** (Kahan-summation-friendly reduction over the vocab axis).

## Tier 3: nice-to-have / specialized

### SIMD RNG distributions — extend `hpc::rng.rs`

Currently scalar; add F32x16 batched paths:

```rust
pub fn gauss_f32_x16(rng: &mut RngState) -> F32x16 { ... }     // Marsaglia polar / Box-Muller
pub fn exp_f32_x16(rng: &mut RngState, lambda: f32) -> F32x16 { ... }
pub fn beta_f32_x16(rng: &mut RngState, alpha: f32, beta: f32) -> F32x16 { ... }
```

### Special functions — extend `hpc::vml.rs`

```rust
pub fn erf_f32(x: f32) -> f32 { ... }        // for Pillar probe concentration bounds
pub fn gamma_f32(x: f32) -> f32 { ... }      // Lanczos approximation
pub fn beta_f32(a: f32, b: f32) -> f32 { ... }
pub fn besselj0_f32(x: f32) -> f32 { ... }   // for audio + radar
pub fn besselj1_f32(x: f32) -> f32 { ... }
pub fn besselj_n_f32(n: u32, x: f32) -> f32 { ... }
```

### Einsum / tensor contractions — `linalg::einsum`

Convenience layer over batched_gemm. Parse the index string at compile time (const generics for the index permutation), dispatch to the appropriate batched_gemm or specialized path.

### FFT extensions — extend `hpc::fft.rs`

- **Bluestein FFT** for non-power-of-2 sizes (44.1k, 48k audio rates)
- **Inverse RFFT** (`irfft_f32`) — currently rfft has no inverse; round-trips force complex `ifft_f32`
- **DCT-II / DCT-IV** as standalone primitives (separate from `audio.rs::mdct`)
- **Daubechies wavelets** db2, db4, db6, db8

### Sparse GEMM — `linalg::sparse`

`blasgraph` has CSR/CSC storage but no SIMD multiply. Add:

```rust
pub fn spmv_csr_f32(a_values: &[f32], a_indices: &[u32], a_indptr: &[u32], x: &[f32], y: &mut [f32]) { ... }
pub fn spmm_csr_f32(a: &CsrMat, b: &Mat, out: &mut Mat) { ... }
```

### Banded / tridiagonal solvers — `linalg::banded`

```rust
/// Thomas algorithm for tridiagonal Ax = b. O(N) instead of O(N³).
pub fn solve_tridiag_f32(a: &[f32], b: &[f32], c: &[f32], d: &[f32], x: &mut [f32]) { ... }
pub fn solve_banded_f32(...);
```

Used in PDE / spline contexts (cubic spline interpolation needs tridiag).

## lance-graph `jc` crate — consolidation work

PR-X10 unblocks the cleanup; the actual work is a **jc-side PR** (call it `jc-X1`):

### Consolidate Spd2/Spd3 into `jc::hadamard`

Three definitions today across `ewa_sandwich.rs`, `ewa_sandwich_3d.rs`, `koestenberger.rs`. After PR-X10 lands `ndarray::hpc::linalg::Spd2/Spd3`, two paths:

- **(a)** `jc` keeps its private `hadamard` module (architectural invariant: jc is zero-dep on ndarray). The hadamard module is the consolidated copy; the three sites all use it.
- **(b)** Relax the zero-dep rule for the SPD primitives only — depend on `ndarray::hpc::linalg::{Spd2, Spd3}`. Simpler but couples jc to ndarray.

**Lean: (a)** — keep jc self-certifying. PR-X10's `linalg::matrix` module's API surface is the reference jc's hadamard mirrors.

### `Cov16384` carrier for Pillar 8 (Düker-Zoubouloglou CLT on AR(1) in ℝ^16384)

Currently hand-rolled at scalar f64 inside the Pillar 8 probe. Promote to a reusable `jc::cov16384` module with `sandwich + log + Frobenius`. Also serves Pillar 9's bigger-N case.

### Wasserstein-1 / nested distance solver for Pillar 10

Currently inline in `pflug.rs`. Consolidate:
- Sinkhorn-Knopp algorithm (entropic regularization)
- Hungarian algorithm (exact assignment)

Both give the cognitive substrate optimal-transport primitives for free.

### Signature transform for Pillar 11

Hambly-Lyons certifies sigker but the actual signature math lives elsewhere. Add native `jc::signature` so the Pillar 11 probe runs standalone.

### SPD-cone operations

Useful for the cognitive substrate's `awareness.revise()` averaging — currently the codebase ducks the question:

- **log-Euclidean mean** (Frobenius geometric mean)
- **Affine-invariant Riemannian mean** (Karcher / Fréchet)
- **Bures-Wasserstein geodesic interpolation**

All three are short additions once `mat_log`/`mat_exp` ship in `linalg::matfn`.

### Manifold log/exp maps

Pillar 2 (Cartan-Kuranishi) is deferred precisely because these primitives don't exist:

- **SO(n)** orthogonal group log/exp
- **Grassmannian** manifold log/exp
- **Stiefel** manifold log/exp

Built on SVD and matrix-exp. Mechanical once those land.

## Architectural invariants (carry-over)

Same eleven invariants as PR-X3 / PR-X4 / PR-X9:

1. Zero-dep on hot path — `crate::simd::F32x16` polyfill, no glam/nalgebra/serde
2. SoA + 64-byte aligned + padded to PREFERRED_F32_LANES
3. No floats in `lance-graph-contract`
4. Click P-1 method discipline
5. `#[repr(C, align(N))]` cross-FFI, `#[repr(u8)]` enums
6. Module docs lead with the math; cite paper/section
7. Pillar-style probes for math correctness
8. Concrete types over generic abstractions on hot paths
9. PP-13 brutally-honest-tester subagent per sub-PR
10. The cognitive `lance-graph-contract/src/splat.rs` is sacred
11. Static-splat vs dynamic-splat separation (from splat4d skeleton-anchored)

**New invariant added by PR-X10:**

12. **Closed-form fast paths for small N must coexist with general-N implementations.** `eig_sym_3` (Smith-1961, ~30 ops) and `eig_sym_n::<3>` (Jacobi, ~300 ops) BOTH ship; consumers pick. The general path is the correctness reference for the closed-form's parity test. Removing the closed-form fast paths is a measurable performance regression on the splat3d hot path (~10× slower).

## Worker decomposition

This is a LARGE sprint. Per the user's "12 agents + 1 coordinator" cadence:

| # | Phase | Workers | Files | LoC |
|---|---|---|---|---|
| 1 | Plan v1 (this doc) | coordinator | — | — |
| 2 | Plan-review savant | 1 | — | — |
| 3 | Plan v2 corrector | coordinator | — | — |
| 4 | **A1 — `MatN<const N>` carrier** | 1 | `linalg/matrix.rs` | ~250 |
| 5 | **A2 — `Quat` algebra** | 1 | `linalg/quat.rs` | ~350 |
| 6 | **A3 — Matrix inverse (3×3, 4×4, general)** | 1 | `linalg/inverse.rs` | ~300 |
| 7 | **A4 — Symmetric eig (Jacobi + QR)** | 1 | `linalg/eig_sym.rs` | ~450 |
| 8 | **A5 — SVD (Golub-Reinsch + one-sided Jacobi)** | 1 | `linalg/svd.rs` | ~500 |
| 9 | **A6 — Polar + mat_exp + mat_log** | 1 | `linalg/polar.rs`, `linalg/matfn.rs` | ~400 |
| 10 | **A7 — SH deg 0..=7** | 1 | `linalg/sh.rs` (supersedes `splat3d/sh.rs`) | ~400 |
| 11 | **A8 — Conv1D + Conv2D** | 1 | `linalg/conv.rs` | ~450 |
| 12 | **A9 — Batched gemm + Norms + Activations** | 1 | `linalg/batched.rs`, `linalg/norm.rs`, `linalg/activations_ext.rs` | ~550 |
| 13 | **A10 — RoPE + Attention (incl. flash-attention)** | 1 | `linalg/rope.rs`, `linalg/attention.rs` | ~600 |
| 14 | **A11 — Cross-entropy + softmax-backward** | 1 | `linalg/loss.rs` | ~250 |
| 15 | **A12 — Tier-3 catalog (RNG dists, vml special fns, FFT extensions, sparse, banded)** | 1 | `hpc/rng.rs`, `hpc/vml.rs`, `hpc/fft.rs`, `linalg/sparse.rs`, `linalg/banded.rs` | ~600 |
| 16 | Codex P0 audit | 1 savant | — | — |
| 17 | Coordinator fix P0s | coordinator | — | — |
| 18 | P2 savant pre-merge | 1 savant | — | — |
| 19 | Merge ladder | — | — | — |

**Total: 12 sprint workers + 1 coordinator + 2 savants = 15 agents** (matches the user's "12 agenten + 1 Koordinator" cadence with savants on top). 12 workers fit because each owns one file (or one tight cluster of related files).

**Parallelism**: A1 (MatN) is the foundation. A2-A12 can spawn ALL IN PARALLEL after A1 lands — each writes to a separate file, all consume `MatN` + `crate::simd::F32x16`. This is the maximum-fan-out worker shape we've drafted; previous sprints had dependency chains that prevented full parallelism. Linalg primitives are intentionally independent — that's the entire point of consolidating them.

**Total sprint duration**: ~2 weeks if all 12 workers run in parallel after A1, ~5 weeks sequential.

## Verification commands

```bash
cargo check -p ndarray --no-default-features --features std,linalg-core
cargo test -p ndarray --lib --no-default-features --features std,linalg-core hpc::linalg
cargo test --doc -p ndarray --no-default-features --features std,linalg-core hpc::linalg
cargo fmt --all -- --check
cargo clippy -p ndarray --no-default-features --features std,linalg-core -- -D warnings
cargo bench --features std,linalg-core hpc::linalg
```

Plus parity gates:
- `eig_sym_3` parity vs `splat3d::Spd3::eig`: max abs error < 1e-6 on 10k random SPD3
- `quat::mul` parity vs reference glam/nalgebra (compile-time only — bench impl, don't link)
- `attention_f32` parity vs PyTorch reference on a 4-head 64-dim 256-seq test case
- SVD parity vs LAPACK `dgesvd` (FFI'd via existing `hpc::lapack.rs`) on 100 random matrices

## Cross-references

- `.claude/knowledge/pr-arithmetic-inventory.md` — the per-layer math inventory PR-X10 consolidates
- `.claude/knowledge/pr-x3-cognitive-grid-design.md` — BlockedGrid substrate (shipped PR #158)
- `.claude/knowledge/pr-x4-design.md` — Gaussian splat cascade (uses linalg::Quat, eig_sym_3)
- `.claude/knowledge/pr-x9-design.md` — lazy basin-codebook storage
- `src/hpc/splat3d/spd3.rs` — current Smith-1961 impl, becomes `linalg::eig_sym_3` reference
- `src/hpc/splat3d/sh.rs` — current deg-3 SH, superseded by `linalg::sh`
- `src/hpc/gpt2.rs`, `src/hpc/openchat.rs`, `src/hpc/stable_diffusion.rs` — current inline RMSNorm/SiLU/RoPE/attention/conv, replaced by `linalg::*`
- `src/hpc/lapack.rs` — existing LAPACK FFI wrappers (LU, Cholesky, QR); linalg-core sits below
- `src/hpc/blas_level3.rs` — existing gemm; linalg::batched calls into it
- **lance-graph `crates/jc/src/{ewa_sandwich.rs, ewa_sandwich_3d.rs, koestenberger.rs}`** — three Spd2/Spd3 copies to consolidate (jc-X1 follow-on)
- **lance-graph `crates/jc/src/pflug.rs`** — Wasserstein-1 inline → `jc::wasserstein` (jc-X2)

## Open questions (for the plan-review savant)

1. **Closed-form fast paths vs general-N only**: `eig_sym_3` Smith-1961 closed-form ships AND `eig_sym_n::<3>` Jacobi general-N ships. Two implementations of the same operation. Lean: **both ship** (invariant 12), closed-form for hot path, general-N for correctness reference + N≥4 fallback. Savant: confirm or reject.

2. **Const-generic `MatN<const N>` vs concrete `Mat2`/`Mat3`/`Mat4` types**: const-generic is more uniform but loses some optimizations; concrete types match existing `Spd2`/`Spd3` style. Lean: **both** — `MatN<const N>` for the general path, `Mat2`/`Mat3`/`Mat4` as type aliases that get specialized impls. Cost: slightly larger codegen.

3. **f64 path?**: splat3d is f32-only. Inference modules are f32. Pillar probes use f64 internally for concentration math. Does `linalg-core` ship f32 AND f64? Lean: **f32 primary** (matches the rest of `hpc::*`), add `_f64` variants only on demand. Savant: rule on whether to pre-ship f64 for the Pillar consumers.

4. **`jc` consolidation path (a) vs (b)**: keep jc zero-dep on ndarray (path a) or relax for SPD only (path b)? Architectural call. Lean: **(a)** preserves the self-certifying property. Coordinator: confirm with jc-architect before committing.

5. **Flash-attention as v1 or v2?**: flash-attention is ~3× the implementation complexity of naive attention. v1 ships naive only; v2 adds flash. OR v1 ships both. Lean: **v1 ships both** — the inference modules need flash for any sequence longer than ~512 tokens. Cost: ~250 extra LoC on A10.

6. **SVD algorithm: Golub-Reinsch vs one-sided Jacobi as primary?**: GR is industry-standard, faster on large N; OSJ is more accurate, SIMD-friendlier on small N. Lean: **both ship**, OSJ for N≤16, GR for N>16. Cost: slightly larger A5 LoC.

7. **PR-X10 vs splat4d cascade vs PR-X4 ordering**: PR-X10 unblocks three downstream stacks (splat3d training, inference, jc) but is independent of splat4d / PR-X4 / PR-X9. Concurrent or sequential? Lean: **concurrent** — PR-X10 ships from a separate branch with its own coordinator; the cognitive-shader stack (PR-X4/X9/Z1) ships on `claude/pr-x4-splat-cascade-design`. No file overlap. Maximum parallelism.

## Done criteria

PR-X10 is done when:
- All 12 worker spec items implemented per the A1-A12 decomposition
- Codex P0 audit passes with 0 P0 — including SAFETY-claim verification gate (per PR-X3.1 backlog)
- `cargo check / test --lib / test --doc / fmt / clippy / bench` all green with `--features std,linalg-core`
- Layering rule verified (zero per-arch surface in `src/hpc/linalg/`)
- Parity gates: eig_sym_3 vs Spd3, attention vs PyTorch ref, SVD vs LAPACK dgesvd
- splat3d's `Spd3` becomes a type alias for `linalg::Spd3` (no API breakage; covered by parity gate)
- splat3d's `sh.rs` superseded by `linalg::sh::eval_deg::<3>` (parity gate verifies bit-equivalence)
- inference modules (`gpt2`, `openchat`, `stable_diffusion`) migrated from inline RMSNorm/SiLU/RoPE to `linalg::*` in a follow-on cleanup PR (NOT in PR-X10; PR-X10 just ships the canonical surface)
- jc consolidation queued as `jc-X1` (Spd2/Spd3 consolidation), `jc-X2` (Wasserstein), `jc-X3` (signature transform), `jc-X4` (SPD-cone ops + manifold log/exp)
- P2 savant pre-merge review delivers SHIP verdict

## Token-reset safety notes (for fresh sessions)

If you're picking up after a token reset:

1. Read this entire doc first.
2. Read `pr-arithmetic-inventory.md` next — the per-layer math inventory PR-X10 consolidates.
3. The conversation context: after the cognitive-shader stack drafting (PR-X3 shipped, PR-X4/X9/Z1 drafted), the user surfaced a comprehensive gap analysis identifying ~15 missing primitives across 3 tiers + 6 jc consolidation items. The cross-cutting observation: the biggest gap is shared linear-algebra below LAPACK; one consolidating sprint unblocks splat3d training + model-inference modules + jc Pillars simultaneously. PR-X10 is that sprint.
4. The 12-worker max-fan-out shape is the highest-parallelism sprint we've drafted — A1 (MatN) is the only chain dependency; A2-A12 spawn all in parallel after A1 lands. The "12 agenten + 1 Koordinator" cadence the user proposed earlier fits exactly.
5. Closed-form fast paths (Spd3, Spd2, Smith-1961) co-exist with general-N (Jacobi, QR). Don't delete the closed-form when ripping out the duplication — they're 10× faster on the splat3d hot path. Invariant 12 codifies this.
6. The jc consolidation is a SEPARATE follow-on (jc-X1) — PR-X10 ships ndarray-side only; jc agents pick up the consolidation against the new canonical surface.
7. PR-X10 is INDEPENDENT of PR-X4 / PR-X9 / PR-Z1 — no file overlap, can ship concurrently from a separate branch. Branch: `claude/pr-x10-linalg-core-design`.
