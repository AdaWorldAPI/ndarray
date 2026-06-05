# splat-native-ultrasound — SIMD substrate (ndarray perspective) — v1

> **Status:** PROPOSAL / integration plan (ndarray-side perspective). Design-spec only; **no code in this plan**.
> **Authored:** 2026-06-05 (session `claude/lance-graph-ontology-review-Pyry3`).
> **Canonical plan:** `lance-graph/.claude/plans/splat-native-ultrasound-v1.md` — this doc is the ndarray-side companion.
>
> **Anchored to:** `I-NOISE-FLOOR-JIRAK` (Jirak weak-dependence Berry-Esseen for any significance threshold); the ndarray vertical-SIMD consumer contract at `lance-graph/.claude/knowledge/ndarray-vertical-simd-alien-magic.md` (W1a-style primitive-addition pattern, all three backends mandatory).

---

## 0. What this is from ndarray's perspective

ndarray is the **SIMD substrate** for the splat-native ultrasound arc. Every floating-point op in the splat-fit → registration → render pipeline runs on the same `src/simd_*` family ndarray already exposes. This doc spells out the **one new module** owed by ndarray — `src/simd_splat.rs` — and its W1c-style five primitives.

This is one deliverable (**D-SPLAT-2**) from the canonical plan. The carriers (`Gaussian3D`, `SplatBatch<N>`) live in `lance-graph-contract`, not here; ndarray sees them as opaque `&[u8]` / typed slice views. The same separation already governs `Fingerprint<256>` and `CamCodecContract` — math here, carriers there, engines downstream.

**Sprint window:** P1 (sprint 1-2) — the **foundation phase**. Must land first; the entire downstream arc depends on these primitives.

---

## 1. What ndarray owns

### D-SPLAT-2 — `src/simd_splat.rs` (the W1c primitive bundle)

A new module under the existing `src/simd_*` family. Five batch primitives, all dispatched through the runtime `simd_caps()` singleton, all three backends mandatory (AVX-512 / NEON / scalar) per the consumer-contract knowledge doc.

**Module skeleton:**

```rust
//! src/simd_splat.rs — anisotropic Gaussian batch ops for splat-native ultrasound
//!
//! Five primitives, all SIMD-dispatched via `simd_caps()`:
//!  - batched_cholesky_3x3
//!  - batched_mahalanobis
//!  - batched_opacity_blend
//!  - batched_sh_eval_l3
//!  - batched_se3_transform
//!
//! Operates on packed `Gaussian3D` SoA layouts owned by lance-graph-contract.
//! All primitives are zero-allocation in the hot loop; SoA column slices in/out.

use crate::simd::SimdBackend;
use crate::simd_caps;

pub fn batched_cholesky_3x3(
    sigma_packed: &[f32],     // N × 6 packed lower-tri Σ (xx, yy, zz, xy, xz, yz)
    out_l_packed: &mut [f32], // N × 6 Cholesky factor L (same packing)
);

pub fn batched_mahalanobis(
    query_xyz: &[f32],        // M × 3 query points
    mu_xyz: &[f32],           // N × 3 Gaussian centroids
    sigma_packed: &[f32],     // N × 6 packed Σ
    out_dist_sq: &mut [f32],  // M × N output (squared Mahalanobis)
);

pub fn batched_opacity_blend(
    sorted_amplitudes: &[f32], // N (front-to-back along view ray)
    opacity_lut: &[u8; 256],   // amplitude → opacity LUT
    out_alpha: &mut [u8],      // composited alpha per pixel
);

pub fn batched_sh_eval_l3(
    view_dir: [f32; 3],       // unit view direction
    sh_coeffs: &[f32],        // N × 16 SH coefficients per Gaussian
    out_luminance: &mut [f32], // N evaluated luminances
);

pub fn batched_se3_transform(
    pose_se3: [f32; 12],      // 3 translation + 9 rotation matrix
    mu_in: &[f32],            // N × 3 input centroids
    mu_out: &mut [f32],       // N × 3 transformed centroids
    sigma_in_packed: &[f32],  // N × 6 input Σ
    sigma_out_packed: &mut [f32], // N × 6 transformed (R Σ Rᵀ)
);
```

### Three-backend implementation matrix (mandatory)

Per the consumer-contract knowledge doc, every primitive ships in three implementations:

| Primitive | AVX-512 | NEON | Scalar |
|---|---|---|---|
| `batched_cholesky_3x3` | 16-lane batch (process 16 Σ matrices in parallel) | 4-lane batch (NEON 128-bit) | per-Σ scalar |
| `batched_mahalanobis` | 16-lane query × 16-lane Gaussian unroll | 4-lane × 4-lane | nested scalar loop |
| `batched_opacity_blend` | gather LUT + alpha composite, 16-lane | 4-lane | scalar |
| `batched_sh_eval_l3` | 16-lane SH-basis evaluation, view dir broadcast | 4-lane | scalar |
| `batched_se3_transform` | 16-lane matmul (μ vector × R) + 16-lane Σ similarity | 4-lane | scalar |

Parity tests are MANDATORY (`I-NOISE-FLOOR-JIRAK`-aware significance gates if any thresholds are claimed).

### Saturating / overflow semantics

- **Cholesky**: degenerate Σ (det ≤ 0) yields NaN in L; downstream uses NaN as a sentinel for "this Gaussian is degenerate, skip it" rather than panicking. Documented as the W1c "VPABSB-correction-style" semantic carve-out in the function docstring.
- **Mahalanobis**: returns `f32::INFINITY` for distances against degenerate Σ rather than NaN (lets the registration math sort distance arrays without NaN-propagation pitfalls).
- **Opacity blend**: front-to-back saturation at 1.0 (alpha ceiling); accumulator uses `u16` internally then truncates to `u8` output.
- **SH evaluation**: f32 luminance output, no clamping (caller decides whether to clamp for visual rendering vs computational pipeline).
- **SE(3) transform**: assumes input rotation matrix is orthonormal; documentation includes a `debug_assert!` validation in debug builds.

---

## 2. What ndarray consumes

Nothing splat-specific — D-SPLAT-2 is foundation work. The primitives consume:

- **Existing `simd_caps()` dispatch** — already shipped, already battle-tested across the 5 SIMD modules.
- **Existing AVX-512 / NEON test infrastructure** — same parity-test patterns as `simd_avx2`, `simd_avx512`, `simd_neon`.
- **Standard `core::arch::x86_64` / `core::arch::aarch64` intrinsics** — no new third-party deps.

No upstream blockers.

---

## 3. What ndarray hands off

### To `lance-graph-contract` (D-SPLAT-1/3 in the canonical plan)

`Gaussian3D` and `SplatBatch<N>` carrier types — they consume ndarray's primitives via the typed-slice API. The carrier crate has zero direct dep on ndarray (consistent with the zero-dep policy for contract); it exposes `&[f32]` column slices for the consumer to feed into `simd_splat`.

### To `crates/splat-fit` (D-SPLAT-6)

The new standalone `splat-fit` crate consumes `ndarray::simd_splat` via the `ndarray-hpc` feature flag (mirrors the `deepnsm` / `bgz17` pattern: optional ndarray dep, zero-dep core).

### To `lance-graph::splat::registration` (D-SPLAT-5)

The ICP / SE(3) optimizer consumes `batched_cholesky_3x3`, `batched_mahalanobis`, `batched_se3_transform` directly. The Σ-sandwich Mahalanobis is two `batched_mahalanobis` calls (against atlas Σ, against live Σ) with elementwise sum — no new primitive needed.

### To `lance-graph::splat-render` (D-SPLAT-12)

The renderer consumes `batched_opacity_blend` + `batched_sh_eval_l3` per-frame. Sort + paint = two ndarray calls.

---

## 4. Per-primitive specifications

### 4.1 `batched_cholesky_3x3`

**Signature** (full Rust):

```rust
/// Compute Cholesky factor L such that L Lᵀ = Σ, for N packed 3×3 symmetric
/// positive-definite matrices Σ.
///
/// # Layout
/// - `sigma_packed[i*6..i*6+6]` = [Σ₀₀, Σ₁₁, Σ₂₂, Σ₀₁, Σ₀₂, Σ₁₂]
/// - `out_l_packed[i*6..i*6+6]` = [L₀₀, L₁₁, L₂₂, L₁₀, L₂₀, L₂₁]
///   (lower-triangular, diag + sub-diag)
///
/// # Semantics
/// - Degenerate Σ (non-PD, det ≤ 0) produces NaN in the output. Downstream
///   uses NaN as a sentinel for "skip this Gaussian"; do NOT propagate.
/// - SIMD-batched 16-at-a-time on AVX-512, 4-at-a-time on NEON, scalar fallback.
///
/// # Panics
/// Never.
pub fn batched_cholesky_3x3(sigma_packed: &[f32], out_l_packed: &mut [f32]);
```

**Algorithm** (3×3 explicit, no loops):

```text
L₀₀ = √Σ₀₀
L₁₀ = Σ₀₁ / L₀₀
L₂₀ = Σ₀₂ / L₀₀
L₁₁ = √(Σ₁₁ - L₁₀²)
L₂₁ = (Σ₁₂ - L₂₀·L₁₀) / L₁₁
L₂₂ = √(Σ₂₂ - L₂₀² - L₂₁²)
```

**Tests:**
- Reference comparison against `nalgebra::Matrix3::cholesky()` on random PD Σ (N = 100k).
- Degenerate-Σ NaN sentinel (Σ = zero matrix; Σ with negative eigenvalue).
- SIMD parity AVX-512 ≡ NEON ≡ scalar (f32 ULP tolerance ≤ 4).
- Bench: ≥ 5× scalar throughput on AVX-512 at N = 1M.

### 4.2 `batched_mahalanobis`

**Signature:**

```rust
/// Squared Mahalanobis distance: (x - μ)ᵀ Σ⁻¹ (x - μ).
///
/// Computes M queries against N Gaussians. Output is M × N row-major.
///
/// # Inputs
/// - `query_xyz[m*3..m*3+3]` = m-th query point
/// - `mu_xyz[n*3..n*3+3]` = n-th Gaussian centroid
/// - `sigma_packed[n*6..n*6+6]` = n-th Σ (same packing as Cholesky)
///
/// # Output
/// - `out_dist_sq[m*N + n]` = ‖x_m - μ_n‖²_{Σ_n}
///
/// # Semantics
/// - Degenerate Σ (Cholesky NaN) yields f32::INFINITY (sortable; never NaN).
/// - SIMD-batched 16×16 on AVX-512, 4×4 on NEON.
pub fn batched_mahalanobis(
    query_xyz: &[f32],            // length 3*M
    mu_xyz: &[f32],               // length 3*N
    sigma_packed: &[f32],         // length 6*N (upper-triangle Σ per Gaussian)
    cholesky_scratch: &mut [f32], // length 6*N (caller-provided; holds packed L per Gaussian)
    out_dist_sq: &mut [f32],      // length M*N (row-major)
);
```

**Implementation note:** internally calls `batched_cholesky_3x3` on `sigma_packed` once per call, writing packed L into `cholesky_scratch` (caller-provided; zero-allocation contract). The caller sizes the buffer as `6 * N * size_of::<f32>()` — for `N = 1_000_000` Gaussians this is **24 MiB**, which is not stack-feasible; callers must allocate it once at engine init and re-use across frames (matches the `splat-fit` / registration loop pattern). For small `N` (e.g. `N ≤ 8192`) callers MAY pass a stack-resident buffer. The function MUST NOT allocate internally.

**Tests:**
- Reference comparison against scipy `scipy.spatial.distance.mahalanobis` on random points + Σ.
- Triangle inequality on a metric subset (commutativity / non-negativity / zero-on-equal).
- SIMD parity.
- Bench: ≥ 1.4× scalar at M=1k, N=1M (the splat-to-splat registration regime).

### 4.3 `batched_opacity_blend`

**Signature:**

```rust
/// Front-to-back alpha compositing over a sorted Gaussian sequence.
///
/// # Inputs
/// - `sorted_amplitudes[i]` = i-th Gaussian's peak echo amplitude, sorted
///   front-to-back along the view ray.
/// - `opacity_lut[i]` = amplitude-bucket → opacity LUT (256 buckets;
///   amplitude quantized into [0..256) via the per-frame max-amplitude).
///
/// # Output
/// - `out_alpha[ray]` = composited alpha per ray (8-bit final).
///
/// # Semantics
/// - Composition: α_new = α_old + (1 - α_old) · α_i.
/// - Internal accumulator: u16 with saturation; truncate to u8 at end.
pub fn batched_opacity_blend(
    sorted_amplitudes: &[f32],    // flat; contains all rays' samples concatenated
    ray_offsets: &[u32],          // length = n_rays + 1 (CSR-style); ray r's range is [ray_offsets[r]..ray_offsets[r+1])
    opacity_lut: &[u8; 256],
    out_alpha: &mut [u8],         // length = n_rays
);
```

**Per-ray segmentation contract.** A renderer composites N independent view rays per frame; each ray has its own front-to-back-sorted Gaussian sequence. `ray_offsets` is a CSR-style prefix-sum (length `n_rays + 1`) so ray `r`'s amplitudes are `sorted_amplitudes[ray_offsets[r] as usize..ray_offsets[r+1] as usize]` and `out_alpha[r]` is its composited alpha. Constraints:
- `ray_offsets[0] == 0` and `ray_offsets[n_rays] == sorted_amplitudes.len() as u32` (assert-on-debug).
- A ray with `ray_offsets[r] == ray_offsets[r+1]` (empty) yields `out_alpha[r] = 0`.
- Per-frame amplitude quantization (the 256-bucket LUT input) is computed by the caller from the per-frame max amplitude; `opacity_lut` is a frame-global constant for that pass.

**Implementation note:** the SIMD inner loop processes one ray's range as a contiguous front-to-back sweep; rays are independent (no cross-ray data dependence) so the outer ray loop is trivially parallelizable.

**Tests:**
- Reference comparison against scalar reference for known sequences (single-ray + multi-ray).
- Saturation at full opacity (sequence of high-amplitude Gaussians → α = 255).
- Empty ray (`ray_offsets[r] == ray_offsets[r+1]`) → α = 0.
- Multi-ray independence (concatenated rays produce same per-ray output as separate single-ray calls).
- `ray_offsets` invariant violations (debug assert on `ray_offsets[0] != 0` or `ray_offsets[last] != amplitudes.len()`).
- SIMD parity.

### 4.4 `batched_sh_eval_l3`

**Signature:**

```rust
/// Evaluate degree-3 spherical harmonics at a single view direction for N Gaussians.
///
/// # Inputs
/// - `view_dir[3]` — unit view direction in scanner frame.
/// - `sh_coeffs[i*16..i*16+16]` — i-th Gaussian's 16 SH coefficients
///   (degree 0: 1, degree 1: 3, degree 2: 5, degree 3: 7 — totaling 16).
///
/// # Output
/// - `out_luminance[i]` — i-th Gaussian's view-dependent luminance.
///
/// # Semantics
/// - No clamping; caller decides downstream behavior (visual: clamp to [0, 1];
///   computational pipeline: signed luminance is fine).
pub fn batched_sh_eval_l3(view_dir: [f32; 3], sh_coeffs: &[f32], out_luminance: &mut [f32]);
```

**Tests:**
- Reference comparison against analytical SH evaluation (closed-form basis functions at degree ≤ 3).
- Rotation invariance: SH coefficients rotated by R, evaluated at view dir = identity; vs original SH coefficients, evaluated at view dir = R⁻¹.
- SIMD parity.

### 4.5 `batched_se3_transform`

**Signature:**

```rust
/// Apply a rigid SE(3) transform to N Gaussian centroids and covariance matrices.
///
/// # Inputs
/// - `pose_se3[0..3]` = translation t
/// - `pose_se3[3..12]` = rotation R (row-major 3×3)
/// - `mu_in[i*3..i*3+3]` = input centroid
/// - `sigma_in_packed[i*6..i*6+6]` = input Σ
///
/// # Output
/// - `mu_out[i*3..i*3+3]` = R · μ_in + t
/// - `sigma_out_packed[i*6..i*6+6]` = R · Σ_in · Rᵀ
///
/// # Semantics
/// - Assumes R is orthonormal (debug_assert in debug builds).
/// - SIMD-batched 16-at-a-time on AVX-512.
pub fn batched_se3_transform(
    pose_se3: [f32; 12], mu_in: &[f32], mu_out: &mut [f32],
    sigma_in_packed: &[f32], sigma_out_packed: &mut [f32],
);
```

**Tests:**
- Identity transform (pose = (0, I)) → output equals input.
- Orthonormal rotation preserves Σ trace.
- Composition: SE(3)(A) followed by SE(3)(B) ≡ SE(3)(B·A) within ULP tolerance.
- SIMD parity.

---

## 5. Cross-repo dependencies (what ndarray waits on / blocks)

### Upstream (what ndarray waits on)

- **Nothing.** D-SPLAT-2 is foundation work; runs before everything else.

### Downstream (what blocks on ndarray D-SPLAT-2)

- **`lance-graph-contract`** D-SPLAT-1/3 (carriers) — can develop in parallel; the carrier API is independent of the math implementation. Soft edge.
- **`crates/splat-fit`** D-SPLAT-6 — **hard blocker**. PSF estimation calls `batched_cholesky_3x3` per peak.
- **`lance-graph::splat::registration`** D-SPLAT-5 — **hard blocker**. ICP runs Mahalanobis + SE(3) transforms per iteration.
- **`lance-graph::splat-render`** D-SPLAT-12 — **hard blocker**. Opacity blend + SH eval per frame.
- **`splat-actors`** D-SPLAT-7 — **soft blocker**. Pose accumulator only needs `batched_se3_transform`; could partially run on scalar fallback while waiting for SIMD.
- **`MedCare-rs`** D-SPLAT-10/11 — **independent**. Storage / audit don't run math.
- **`OGAR`** Phase 8 (FMA TTL) — **independent**. Ontology prep doesn't touch ndarray.

---

## 6. Risk matrix (ndarray-specific)

| Risk | Severity | Mitigation |
|---|---|---|
| SIMD parity diverges across backends (AVX-512 vs NEON vs scalar) | MED | Mandatory parity tests on every primitive (per consumer-contract knowledge doc); ULP-tolerance gates documented per primitive. |
| Degenerate-Σ NaN propagation into downstream | HIGH | Documented NaN-sentinel semantics; `batched_mahalanobis` converts NaN distances to `f32::INFINITY` to make sort stable; integration tests cover degenerate paths. |
| AVX-512 throughput regression on older Skylake-X / Cascade Lake (VBMI gate) | MED | Reuse existing `VBMI gate for permute_bytes` pattern from ndarray PR #142; gate `batched_opacity_blend`'s 16-lane gather variant behind VBMI feature detection. |
| Bench fluctuation across CI runners | LOW | Use the existing nightly bench infrastructure; ≥ 5 runs minimum; report median + IQR. |
| Carrier-API changes in `lance-graph-contract` between draft and final | LOW | ndarray exposes typed-slice API (`&[f32]`); carrier struct layout changes don't affect ndarray. |

---

## 7. Acceptance criteria

### Per-primitive

- **Correctness:** reference comparison passes (≤ 4 ULP) against the named reference implementation.
- **Parity:** AVX-512 ≡ NEON ≡ scalar (within ULP tolerance).
- **Bench:** primitive-specific throughput gate met on the target hardware (AVX-512: x86-64-v4; NEON: Apple M-series or Cortex-A78).

### Module-level

- `cargo test -p ndarray --features simd-splat` — green.
- `cargo bench -p ndarray --bench simd_splat` — published baseline numbers.
- Three-backend implementations live in the existing `src/simd_*` family pattern; no new `unsafe` blocks beyond what the family already uses; all `unsafe` blocks have `// SAFETY:` comments.
- Module is feature-gated behind `simd-splat` (default-on, per ndarray's "ndarray is mandatory" policy from PR #463); zero-impact on existing consumers.

---

## 8. Cross-references

- **Canonical plan:** `lance-graph/.claude/plans/splat-native-ultrasound-v1.md` §§3.1 (D-SPLAT-1), 3.2 (D-SPLAT-2 detailed spec), 11.1-11.7 (work division)
- **Consumer contract:** `lance-graph/.claude/knowledge/ndarray-vertical-simd-alien-magic.md` (W1a primitive-addition pattern; mandatory three-backend implementation; parity-test gates; VPABSB-correction example for saturating semantics)
- **ndarray priors:**
  - `src/simd_avx512.rs` (W1a primitives in production)
  - `src/simd_avx2.rs`, `src/simd_neon.rs` (companion backends)
  - PR #142 (VBMI gate for `permute_bytes` — relevant for `batched_opacity_blend`)
  - PR #189 (`OntologySchema::is_ancestor` — relevant for D-SPLAT-8 FMA traversal downstream)
  - PR #463 (ndarray-is-mandatory in lance-graph — relevant for splat-fit feature-flag design)
- **Iron rules:** `I-NOISE-FLOOR-JIRAK` (any threshold derived in downstream registration cites Jirak under weak dependence, not classical Berry-Esseen)

---

## 9. Sprint window

**P1, sprint 1-2.** Foundation work. **Must land before everything else in the splat-native arc.**

### Sprint 1

- D-SPLAT-2 implementations on the AVX-512 backend (the lead path, since ndarray's pinned target is x86-64-v4).
- Reference-implementation parity tests against `nalgebra` + analytical SH closed-form + scipy where available.
- Initial bench numbers.

### Sprint 2

- D-SPLAT-2 NEON backend (Apple M-series + Cortex-A78 target).
- D-SPLAT-2 scalar fallback (CI floor for non-AVX-512 / non-NEON CI runners).
- Three-backend parity test green.
- Bench gates verified on representative hardware.
- AGENT_LOG entry; PR_ARC entry.

**Acceptance for handoff to P2 (lance-graph splat-fit):** all five primitives green on all three backends; bench targets met; carrier API stable from `lance-graph-contract` D-SPLAT-1/3 (concurrent sprint 1-2).

---

_End of ndarray companion v1._
