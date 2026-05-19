# PP-15 Interface-Signal Audit — Verdict

Reviewer: Opus PP-15 Interface-Signal Auditor
Branch: `claude/pr-x4-splat-cascade-design` @ `5e266d19`
Mindset: signal-in-the-interface, no materialization between cascade steps

Scope: ~125 public surfaces across `src/hpc/linalg/`, `src/hpc/pillar/`,
`src/hpc/ogit_bridge/`. Test 1 (Click P-1), Test 2 (signal-in-interface),
Test 3 (cascade composition).

The codebase has **two house dialects**:

1. **carrier-method dialect** — `Spd3::sqrt(&self) -> Spd3`,
   `CovHighD::sandwich(&self, m: &Self) -> Self`, `MotionBand::sigma(self) -> f32`,
   `CognitiveBridge::nearest_basin(&self, …)`. Honors Click P-1.
2. **flat-slice imperative dialect** — `fn op_f32(x: &mut [f32], gamma: &[f32], …)`.
   The vast majority of `hpc/linalg/*` adopts this; it is **structurally
   anti-Click-P-1** because `&[f32]` is not a carrier — every caller has to
   pre-materialize an `out` buffer and there is no return type to chain.

The data-flow rule in `.claude/rules/data-flow.md` ("No `&mut self` during
computation. Ever.") rules out the literal `x.layer_norm(...)` mutating-self
Click-P-1 rewrite. The correct Click-P-1 form is therefore **compute-returns-new**:
`fn layer_norm(&self, gamma, beta, eps) -> Self` on a typed carrier
(`ArrayView1<f32>` extension trait or a thin `Tensor1<f32>` newtype).

## Click P-1 violations (free-function-with-carrier-arg)

| File:Line | Current signature | Click-P-1 rewrite |
|---|---|---|
| `src/hpc/linalg/attention.rs:100` | `attention_f32(q, k, v, &mut out, &cfg, b, s)` | `q.attend(k, v, &cfg) -> AttentionOut` where `AttentionOut: Deref<[f32]>` |
| `src/hpc/linalg/attention.rs:221` | `flash_attention_f32(q, k, v, &mut out, &cfg, b, s, block)` | `q.flash_attend(k, v, &cfg, block) -> AttentionOut` |
| `src/hpc/linalg/batched.rs:60` | `batched_gemm_f32(x, y, &mut out, b, m, k, n, α, β)` | `BatchedMat{x,m,k,batch}.matmul(&BatchedMat{y,k,n,batch}, α, β) -> BatchedMat` — shape is part of the type, not loose args |
| `src/hpc/linalg/batched.rs:132` | `batched_gemm_4d_f32(...)` | `BatchedMat4D.matmul(&Self, α, β) -> BatchedMat4D` |
| `src/hpc/linalg/conv.rs:60` | `conv1d_f32(input, kernel, stride, pad, &mut out)` | `Signal1D.conv(&Kernel1D, stride, pad) -> Signal1D` |
| `src/hpc/linalg/conv.rs:123,202,285,392` | `conv2d{,_3x3,_5x5,_im2col}_f32(input, in_shape, kernel, kshape, stride, pad, &mut out)` | `Image.conv2d(&Kernel, stride, pad) -> Image` — shapes ride inside `Image`/`Kernel` |
| `src/hpc/linalg/norm.rs:54` | `layer_norm_f32(&mut x, γ, β, ε)` | `x.layer_norm(γ, β, ε) -> Tensor1` (NOT `&mut self`) |
| `src/hpc/linalg/norm.rs:100` | `rms_norm_f32(&mut x, γ, ε)` | `x.rms_norm(γ, ε) -> Tensor1` |
| `src/hpc/linalg/norm.rs:143` | `group_norm_f32(&mut x, γ, β, groups, ε)` | `x.group_norm(γ, β, groups, ε) -> Tensor1` |
| `src/hpc/linalg/activations_ext.rs:67,93,117,143,168` | `{gelu,gelu_tanh,silu,swish,mish}_f32(&mut x[, β])` | `x.gelu()`, `x.silu()`, … each returning `Tensor1` |
| `src/hpc/linalg/loss.rs:244` | `softmax_xent_backward_f32(logits, targets, &mut grad_out, b, v)` | `logits.softmax_xent_backward(targets) -> GradTensor` |
| `src/hpc/linalg/loss.rs:155` | `cross_entropy_with_logits_batched_f32(logits, targets, b, v) -> f32` | `logits.xent_batched(targets) -> f32` — return-shape is fine, but carrier-method form keeps it composable |
| `src/hpc/linalg/wasserstein.rs:35` | `sinkhorn_knopp_f32(cost, m, n, a, b, ε, iter, tol) -> Vec<f32>` | `CostMatrix.sinkhorn(&a, &b, ε, iter, tol) -> TransportPlan` |
| `src/hpc/linalg/wasserstein.rs:287` | `wasserstein_1_f32(cost, plan, m, n) -> f32` | `plan.cost_against(&CostMatrix) -> f32` |
| `src/hpc/linalg/wasserstein.rs:112` | `hungarian_f32(cost, m) -> Vec<u32>` | `CostMatrix.hungarian() -> Assignment` |
| `src/hpc/linalg/rope.rs:125` | `RopeCache::apply_qk_f32(&self, &mut q, &mut k, positions, b, s, h)` | `q.with_rope(&cache, positions) -> Tensor` — current shape is half-Click-P-1 (carrier IS `self`) but it mutates two *other* slices, so the receiver is wrong |
| `src/hpc/pillar/temporal_sandwich.rs:165` | `sandwich_update_3x3(σ, m) -> [[f32;3];3]` | `Spd3.sandwich(&self, m: &Spd3) -> Spd3` — the typed version *already exists* in `pillar/cov_high_d.rs:124`; this 3×3 hardcoded copy is the violation |
| `src/hpc/pillar/temporal_sandwich.rs:201` | `is_spd_3x3(m) -> bool` | `Spd3.is_spd(&self) -> bool` |
| `src/hpc/pillar/koestenberger.rs:78,116` | `path1_direct_sandwich(σ, m) -> Spd3`, `path2_spectral(σ, m) -> Spd3` | `σ.sandwich_direct(&m)`, `σ.sandwich_spectral(&m)` |
| `src/hpc/pillar/koestenberger.rs:194` | `max_abs_error_spd3(a, b) -> f64` | `Spd3.max_abs_error(&self, other) -> f64` |
| `src/hpc/pillar/signature.rs:91,188,211` | `signature_d2_deg3(path, n) -> [f32;…]`, `sigker_hl(p, q) -> f32`, `brownian_path_d2(rng, n) -> Vec<f32>` | `Path2D.signature_deg3() -> Signature`, `sig_p.kernel(&sig_q) -> f32`, `rng.brownian_d2(n) -> Path2D` |

## Materialization-forced (out: &mut Buffer args that should be typed return)

Every `_f32` function in `linalg/{attention, batched, conv, norm,
activations_ext, loss}` takes `out: &mut [f32]` (or mutates `x` in place).
That is **15 out of ~22 sprint-introduced surfaces** forcing the caller to
pre-size and materialize a buffer for the next step.

Concretely, the cascade `attention → layer_norm → gelu → batched_gemm`
requires 4 pre-allocated buffers and 4 manual length-arithmetic asserts at
the call site, instead of `q.attend(k,v,&cfg).layer_norm(γ,β,ε).gelu().matmul(&w)`.

**Severity heuristic**: this is the dominant failure mode of the sprint
(15/22 surfaces), and the reason consumers will be forced to write the
materialization-glue PP-15 is supposed to prevent.

## Cascade composition gaps (where A.method() doesn't fit B's receiver)

1. **Splat tick** — `TileBinning::from_projected(&ProjectedBatch, &Camera) -> Self`
   is good (`tile.rs:105`), and `binning.tile_instances(tx,ty) -> &[TileInstance]`
   is good. But `rasterize_tile` and `rasterize_frame` (`raster.rs:71,213`)
   are free functions taking the binning + a `&mut framebuffer` — they
   should be `binning.rasterize(&projected, &camera, bg) -> Framebuffer`.
   Net: 3 chain steps that *don't* compose: `binning → rasterize_*(out)`.
2. **NARS revision** — `nars_revision(a, b)`, `nars_deduction`, `nars_abduction`
   (`nars.rs:322,342,362`) all take **two `NarsTruth` by value, return
   `NarsTruth`** — *no materialization*, *no buffers*, and the return type IS
   the next call's receiver. Test 2 and Test 3 pass. Test 1 fails on the
   surface (free fn, not method), but this is the strongest form of
   "signal-in-the-interface" in the audit. Trivial method rewrite:
   `a.revise(b) -> NarsTruth`.
3. **Cognitive cell encode** — `bridge.nearest_basin(cell_value, hint) -> u16`
   (`cognitive_bridge.rs:335`). The bare `u16` is a stringly-typed key. The
   downstream `codec::rdo_cell(basin, …)` (TTL-referenced; not yet
   implemented in this branch) will accept a `u16` or `usize` and the
   compiler will not catch a basin/family/codebook-index mixup. Return type
   should be `BasinHandle(u16)` so the next click in the cascade is
   `bridge.family_of(handle)` (the method already takes `u16` at line 311 —
   change to `BasinHandle` and the entire ogit_bridge surface becomes
   type-safe).
4. **`PillarReport`** — `prove_pillar_7() -> PillarReport` honors return-type.
   But `PillarReport` has only one method (`print(&self)`, line 172 of
   `prove_runner.rs`); there is no `report.assert_passed() -> &Self` or
   `.merge(other) -> PillarReport`. So cascading the eleven pillar probes
   into a single `PillarSuite` requires manual `Vec<PillarReport>` glue
   (and indeed `prove_pillar_8()` at `temporal_sandwich.rs:323` returns
   `Vec<PillarReport>`, breaking the chain shape vs. its peers).

## Click P-1 honors (sprint-introduced surfaces that do it right)

- `Spd3::sqrt`, `Spd3::sandwich`, `Spd3::eig`, etc. in `linalg/matrix.rs` — methods on the SPD carrier.
- `CovHighD::sandwich(&self, m: &Self) -> Self` at `pillar/cov_high_d.rs:124` — **textbook Click P-1**: typed carrier, typed args, typed return, composes (`a.sandwich(&b).sandwich(&c).frobenius_sq()`).
- `CovHighD::log_spd(&self) -> Self` at `cov_high_d.rs:202` — ditto.
- `MotionBand::sigma(self) -> f32` at `temporal_sandwich.rs:111` — enum-as-carrier.
- `TileBinning::from_projected(&ProjectedBatch, &Camera) -> Self` and `.tile_instances(...) -> &[TileInstance]` — typed-surface returns, no out-buffer.
- `CognitiveBridge::load_embedded() -> Result<Self, OgitError>`, `.codebook() -> &CamCodebook`, `.family_of(idx) -> &FamilyBitmap` — clean carrier-methods. Only `nearest_basin`'s bare `u16` return type lets the chain down.
- `nars_revision/deduction/abduction(a, b) -> NarsTruth` — value-in, value-out, zero materialization; the *cleanest* signal-in-interface in the audit even though the surface is a free fn.

## Net call

Of ~22 sprint-introduced public surfaces in `linalg/{attention, batched,
conv, norm, activations_ext, loss, wasserstein, rope}`, **roughly 18 fail
Test 1 (free-fn-with-carrier-arg) and 15 fail Test 2 (materialization-forced
`out: &mut`)**. Pillar code is split: `pillar/cov_high_d.rs` and
`pillar/ewa_sandwich_3d.rs` honor Click P-1; `pillar/{temporal_sandwich,
koestenberger, signature}` are free-function-on-bare-arrays.
`ogit_bridge` is mostly clean — single fix needed: lift `u16` to `BasinHandle`.

**Severity**: high in count, low-to-medium in difficulty. None of the
violations are algorithmically wrong; they are all skin-deep signature
rewrites where the kernel body is reusable verbatim. A same-day cleanup
sprint could mechanically wrap each `op_f32(&[f32], …, &mut [f32])` in an
extension-trait method on `ArrayView1<f32>` (or a thin `Tensor1` newtype)
returning `Tensor1`. The data-flow rule "No `&mut self` during computation"
forces the new-allocation Click-P-1 form anyway — and that allocation is
already happening at every consumer site, just expressed as
`let mut out = vec![0.0; n]` instead of inside the method.

The **structural** decision needed before the cleanup: is the carrier
`ArrayView1<f32>` (ndarray-native), a new `Tensor1<f32>` newtype, or a
shape-aware `Tensor<D>`? Without that decision the cleanup will recreate
the inconsistency in a different shape. Recommend: pick `Tensor1` /
`Tensor2` / `Tensor4` thin newtypes around `Vec<f32>` + shape tuple, with
`Deref<Target=[f32]>` for SIMD escape-hatch. This matches the
`BatchedMat`/`Image`/`Signal1D` carriers suggested in the violation table.

**Recommended sequence**:
1. Land `BasinHandle(u16)` — 1-hour change, type-safety dividend across `ogit_bridge`.
2. Decide carrier shape (`Tensor1`/`Tensor2`/`Tensor4`).
3. Add extension-trait methods that wrap each `*_f32` free fn — keep the free fns as `#[doc(hidden)]` shims for one release.
4. Migrate `pillar/temporal_sandwich.rs` and `pillar/koestenberger.rs` to call methods on the existing `Spd3` carrier (the typed version already exists in `matrix.rs` — these modules are reimplementing what they could be using).

This is a **follow-on cleanup sprint** (one to two days for a single
worker), not a same-day patch — the carrier-type decision is load-bearing.

## Sentinel: pp15-interface-signal-completed
