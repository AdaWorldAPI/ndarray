# KNOWLEDGE: Vertical SIMD — W1a Consumer Contract

## READ BY:
- `savant-architect` agent — before designing any new public `pub fn` in `src/simd_*.rs`
- `sentinel-qa` agent — when auditing the saturating / bounds-aware / scalar-fallback discipline on a SIMD addition
- Any contributor opening a PR that adds an `impl` block on `F32x16` / `I8x16` / `U8x32` / `U64x8` etc.
- Any contributor adding a new public function under `src/simd_ops.rs` or `src/simd_int_ops.rs`

## P0 TRIGGERS:
- About to file a PR adding `pub fn` to `src/simd_*.rs` → read this first
- About to claim "X SIMD instruction saturates by ISA" → read §"VPABSB correction" first
- Five `TD-NDARRAY-SIMD-*` issues are about to be filed against this repo from the `AdaWorldAPI/lance-graph` consumer contract → those are the W1a queue described below

---

## Why this doc exists

`AdaWorldAPI/lance-graph` (the obligatory spine for the Ada architecture) carries a hard architectural invariant: **all SIMD must come from `ndarray::simd` via the polyfill — `simd.rs` + `simd_ops.rs` > `simd_{type}.rs` per-arch. Raw intrinsics outside `ndarray/src/simd_*.rs` are a violation**, enforced by the `simd-savant` agent at `lance-graph:.claude/agents/simd-savant.md`.

A PRE-MERGE audit of `lance-graph` main on 2026-05-16 surfaced **158 raw-intrinsic violations across 5 consumer crates** plus **3 missing primitives** in `ndarray::simd` that block clean remediation. The lance-graph side is staged to migrate (in 5 sequential consumer PRs); the missing primitives must land in ndarray FIRST. This doc is the contract for what those primitives must do, with implementation details called out where consumer-side correctness depends on getting the semantics right.

The architectural shape this doc serves is captured in detail at:
- `AdaWorldAPI/lance-graph:.claude/knowledge/ndarray-vertical-simd-alien-magic.md` — the canonical reference, "alien magic" framing
- `AdaWorldAPI/lance-graph:.claude/agents/simd-savant.md` — the consumer-side enforcement card
- `AdaWorldAPI/lance-graph:.claude/board/EPIPHANIES.md` § `E-SIMD-SWEEP-1` (2026-05-16) — the 158-violation finding

---

## The pattern (one paragraph)

ndarray's SIMD surface is shaped to fit exactly what the Ada stack vertically needs — not as a generic library that consumers wrap, but as **struct methods on typed wrappers** (`I8x16`, `U8x32`, `F32x16`, `U64x8`, …) plus **closure-parameterized batch primitives** that absorb the consumer's domain semantics. Consumers see zero raw intrinsics, zero `cfg(target_arch)`, zero runtime feature-detect — they call `I8x16::from_i4_packed_u64(...)`, `I8x16::saturating_abs(...)`, `batch_packed_i4_16(..., |lanes, aux| { ... })`. The polyfill owns the runtime feature dispatch, lane chunking, tail handling, and scalar fallback. Per-arch code lives in `simd_avx512.rs` / `simd_neon.rs` / `simd_wasm.rs`; nothing arch-specific leaks above the `src/simd*.rs` namespace.

---

## VPABSB correction (P0 — read before implementing saturating_abs)

**`_mm512_abs_epi8` (VPABSB) does NOT saturate `i8::MIN`.** The Intel intrinsic returns the same bit pattern for `0x80` — i.e., `abs(i8::MIN) = i8::MIN` because `+128` does not fit in `i8`. An earlier draft of the consumer contract (2026-05-16 morning) claimed the instruction saturated `i8::MIN → 127` by ISA. Codex caught this on `lance-graph` PR #400; the correction is binding.

**Correct AVX-512 implementation of `I8x16::saturating_abs`:**

```rust
// AVX-512 path
let raw_abs = unsafe { _mm512_abs_epi8(self.0) };
let clamped = unsafe {
    _mm512_min_epu8(raw_abs, _mm512_set1_epi8(0x7f))
};
I8x16(clamped)
```

The mechanic:
1. **VPABSB** computes the bit-pattern absolute value lane-wise. For `0x80` it returns `0x80` (the bit pattern of `+128` interpreted as unsigned). For everything else, `abs(x) < 0x80`, so the result fits in `i8` correctly.
2. **VPMINUB** (unsigned-byte min) then clamps `0x80` (=128 unsigned) down to `0x7f` (=127). All lanes with `abs(x) < 0x80` are unaffected because `min_epu8(x, 0x7f) = x` for `x ≤ 0x7f` and `min_epu8(0x80, 0x7f) = 0x7f`.

Equivalent NEON:
```rust
// vqabsq_s8 is hardware-saturating (the `q` suffix means saturating)
I8x16(unsafe { vqabsq_s8(self.0) })
// Returns 127 for i8::MIN, identical to the AVX-512 + clamp result
```

Scalar fused-loop:
```rust
for lane in 0..16 {
    out[lane] = input[lane].saturating_abs();  // stdlib, well-defined
}
```

**Mandatory test** (binding for the PR):
```rust
#[test]
fn saturating_abs_i8_min_matches_across_backends() {
    let input = I8x16::splat(i8::MIN);
    let result = input.saturating_abs();
    assert_eq!(result.lane_i8::<0>(), i8::MAX);
    // ... and assert all 16 lanes equal i8::MAX
}
```

Any saturating-abs primitive in ndarray that does NOT produce `i8::MAX` for `i8::MIN` input is broken. The widen-then-negate trick (i8 → i64, then negate, then compare against threshold) used in `lance-graph` PR #398's mul.rs is a different mechanism and **not a substitute** — the new `I8x16::saturating_abs` must produce the saturating result in the same byte-wide register without widening, because downstream consumers will rely on byte-wide semantics for tight i4/i8 packed loops.

---

## W1a queue — 5 primitives ndarray must ship

Each is a tight-scope PR. Recommended: one branch per primitive, parallel review.

### W1a-#1 — `TD-NDARRAY-SIMD-UNPACK-I4-16D`

**Purpose:** unpack a `u64` of 16 packed signed nibbles (i4) into an `I8x16` with sign extension. Plus the closure-batch entry that the consumer's `mul::i4_eval::batch` dispatch calls.

**API surface:**
```rust
impl I8x16 {
    /// Unpack 16 signed i4 nibbles from a u64 into 16 i8 lanes
    /// (sign-extended). Nibble layout: lane[i] = sign_extend_4((packed >> (4*i)) & 0xf, i8).
    pub fn from_i4_packed_u64(packed: u64) -> Self;

    /// Const-folded lane extract.
    pub fn lane_i8<const N: usize>(self) -> i8;
}

/// Closure-parameterized batch: run `f` over each (unpacked_i8x16, aux[i]) pair.
/// Bounds-aware tail handling; scalar fallback on unsupported arch.
pub fn batch_packed_i4_16<E, F>(
    packed: &[u64],
    aux: &[i8],
    out: &mut [E],
    f: F,
)
where
    F: Fn(I8x16, i8) -> E + Sync + Send,
    E: Copy;
```

**Per-arch implementation hints:**
- **AVX-512:** load 16 × i8 from u64 via `_mm_cvtsi64_si128` + extend with `_mm512_cvtepi8_epi16` + nibble shuffle (PEXTRB or VPSHUFB with a mask LUT), then sign-extend by `_mm_cvtepi8_epi16`. Bench against alternative: PDEP (`_pdep_u64` × 2) into two u64 halves, then load + `vpmovsxbw` for sign-extend. Pick whichever benches faster on Zen4 + Sapphire Rapids.
- **NEON:** `vld1_u8` 8 bytes into `uint8x8_t`, then nibble-split via `vshl_n_s8(v, 4)` and `vshr_n_s8(v, 4)`. Sign-extension is automatic from `vshr_n_s8`.
- **Scalar:** fused loop reading 16 nibbles via `((packed >> (4*i)) & 0xf) as i8` with manual sign-extend (`if x > 7 { x - 16 } else { x }`).

**Consumer call site:** `lance-graph:crates/lance-graph-contract/src/mul.rs::i4_eval::batch` (5 batch fns over `QualiaI4_16D(u64)`). The closure-batch absorbs the 5 fns into closures + classifier names.

**PR #398 codex P1 (NEON OOB at `len==2`) is closed by this primitive** because the batch entry owns tail handling; consumers no longer reach for raw `vld1q_u64(&qualia[i+1].0 as *const u64)`.

---

### W1a-#2 — `TD-NDARRAY-SIMD-SATURATING-ABS-I8`

**Purpose:** byte-wide saturating absolute value. Closes codex P2 i8::MIN divergence on `lance-graph` PR #398 by giving consumers a single source-of-truth.

**API surface:**
```rust
impl I8x16 {
    /// Lane-wise saturating absolute value. saturating_abs(i8::MIN) == i8::MAX.
    /// All lanes are independently saturated.
    pub fn saturating_abs(self) -> Self;
}

impl I8x32 {
    pub fn saturating_abs(self) -> Self;  // parity
}
```

**Per-arch implementation:** see § "VPABSB correction" above. The AVX-512 path is `_mm512_min_epu8(_mm512_abs_epi8(x), _mm512_set1_epi8(0x7f))`; NEON is `vqabsq_s8`; scalar is `i8::saturating_abs`.

**Consumer:** `lance-graph:crates/lance-graph-contract/src/mul.rs` (Direction-B fix from PP-16 preflight-drift-auditor 2026-05-16). Spec line 233 of `lance-graph:.claude/specs/pr-sprint-13-simd-i4.md`: `|signed_mantissa| ≤ 1 → ValleyOfDespair` represents weak rule signal, NOT sign-extreme; `i8::MIN` must classify as `Slope/Plateau`, not `ValleyOfDespair`. Scalar in PR #398 is buggy (uses `unsigned_abs() as i8` which wraps `i8::MIN → -128`); the new primitive lets the fix be a one-liner: `lanes.saturating_abs().lane_i8::<0>()` ≤ 1.

---

### W1a-#3 — `TD-NDARRAY-SIMD-GATHER`

**Purpose:** SIMD gather for palette / lookup-table consumers. Currently `bgz17/src/simd.rs:88` inlines `_mm256_i32gather_epi32` (AP-SIMD-1 violation).

**API surface:**
```rust
impl U16x8 {
    /// Gather 8 u16 values from `table` at the given indices.
    /// indices[i] >= table.len() => panic in debug, scalar-fallback safe in release.
    pub fn gather_u16(indices: U16x8, table: &[u16]) -> Self;
}

/// Convenience: lookup 8 bytes from a u8 LUT by u16 indices.
pub fn palette_lookup_u8x8(idx_v: U16x8, lut: &[u8]) -> U8x8;
```

**Per-arch implementation:**
- **AVX2/AVX-512:** `_mm256_i32gather_epi32` with index widening + downcast (caveat: `_mm256_i32gather_epi32` reads 32 bits per index; for u16 values pack two indices per gather slot, or downcast post-gather).
- **NEON:** no native gather instruction. Scalar loop is fine for 8 lanes — `(0..8).map(|i| table[indices.lane(i) as usize])`.
- **Scalar:** identical to the NEON fallback.

**Bounds:** `gather_u16` MUST validate `max(indices) < table.len()` before the SIMD gather call (debug panic; in release, fall through to scalar with `.get()` for safety).

---

### W1a-#4 — `TD-NDARRAY-SIMD-PREFETCH`

**Purpose:** cross-arch prefetch hint. Currently `bgz17/src/prefetch.rs:96,100` inlines `_mm_prefetch` and `_prefetch` directly.

**API surface:**
```rust
/// Hint that `ptr` will be read soon; load into L1 (T0) cache.
pub fn prefetch_read_t0(ptr: *const u8);

/// Hint to load into L2 (T1) cache.
pub fn prefetch_read_t1(ptr: *const u8);

/// Hint to load into L3 (T2) cache.
pub fn prefetch_read_t2(ptr: *const u8);
```

**Per-arch implementation:**
- **x86_64:** `_mm_prefetch(ptr as *const i8, _MM_HINT_T0)` / `_T1` / `_T2`.
- **aarch64:** `__pld(ptr)` via inline asm `prfm pldl1keep, [ptr]` (T0), `pldl2keep` (T1), `pldl3keep` (T2). Or wrap `core::intrinsics::prefetch_read_data` if/when stable.
- **Other arches:** no-op (the prefetch contract is a hint, not a guarantee — silent no-op is correct).

**Safety:** `ptr` is allowed to be invalid (prefetch on an unmapped page is a hint that the CPU silently drops on x86). No `assert!` needed.

---

### W1a-#5 — `TD-NDARRAY-SIMD-POPCOUNT-U64`

**Purpose:** lane-wise popcount of u64 vectors. Currently `holograph/hamming.rs` and `lance-graph:crates/lance-graph/src/graph/blasgraph/types.rs` use `_mm512_popcnt_epi64` directly for Hamming-distance reduction.

**API surface:**
```rust
impl U64x8 {
    /// Lane-wise population count. Each lane returns its u64 bit-count (0..=64).
    pub fn popcnt(self) -> Self;

    /// XOR + lane-wise popcount + horizontal sum across 8 lanes.
    /// Optimized for Hamming-distance reductions.
    pub fn xor_popcount(self, other: Self) -> u64;
}

impl U64x4 {
    pub fn popcnt(self) -> Self;  // AVX2 parity
}
```

**Per-arch implementation:**
- **AVX-512 VPOPCNTDQ:** `_mm512_popcnt_epi64` directly. Feature flag `avx512vpopcntdq`.
- **AVX-512 without VPOPCNTDQ:** fallback via `_mm512_sad_epu8` on a per-byte popcount LUT (Mula's algorithm using VPSHUFB).
- **NEON:** `vcntq_u8` for byte popcount, then horizontal sum within each u64 via `vaddvq_u8` or `vpaddlq_u8` cascade.
- **Scalar:** `u64::count_ones` fused loop.

**Note:** the existing `ndarray::hpc::bitwise::popcount_raw` and `hamming_distance_raw` cover the slice case but DO NOT expose a lane-wise method. The new `U64x8::popcnt` fills that gap so consumers can compose Hamming-distance pipelines without dropping back to slice ops.

---

## W1.5 — DEFERRED primitives (gated on `lance-graph:crates/sigker` certification)

Three more primitives are queued behind a certification gate. `crates/sigker` is `lance-graph`'s path-signature codec — it's pure-scalar Rust today (zero raw intrinsics, zero ndarray dep), and is positioned as the **Index-regime third encoding lane** alongside palette-distance (bgz17) and NSM tiling (deepnsm). It explicitly bypasses the `I-NOISE-FLOOR-JIRAK` iron rule (Jirak 2016 Berry-Esseen for weak-dependence data) via Hambly-Lyons 2010 path-signature uniqueness.

When `jc Pillar 11` (Hambly-Lyons signature uniqueness on lance-graph paths) activates and sigker is benchmarked at production carrier widths, the W1.5 queue lights up:

### W1.5-#6 — `TD-NDARRAY-SIMD-SIGNATURE-PDE-SWEEP`

**Purpose:** signature kernel `〈S(X), S(Y)〉` via Goursat PDE — depth-∞ in O(T₁·T₂) flops, no signature materialization.

**API surface (sketch):**
```rust
pub fn signature_pde_sweep<F>(
    x: &[F32x16],
    y: &[F32x16],
    kernel_fn: F,
) -> f32
where
    F: Fn(F32x16, F32x16) -> F32x16;
```

2D banded grid sweep; closure-parameterized kernel evaluator per step.

### W1.5-#7 — `TD-NDARRAY-SIMD-RANDOMIZED-PROJECTION`

Cuchiero-Schmocker-Teichmann (2021) randomized signatures: Gaussian random-matrix-vector update with `F32x16` state. Same closure-batch shape as W1a-#1, different lane type.

### W1.5-#8 — `TD-NDARRAY-SIMD-LYNDON-PACK`

Log-signature compression in the Lyndon basis of the free Lie algebra (7-13× compression, lossless). Pack/unpack primitives on `I16x16` state with combinatorial-index awareness.

**No code needed today for W1.5.** Mentioned here so W1a additions are designed broad enough to compose with these later (in particular: the closure-batch shape introduced in W1a-#1 is the foundation for W1.5-#7).

---

## Acceptance criteria for each W1a PR

Every PR adding a primitive from this queue MUST:

1. **Implement all three backends** (AVX-512/AVX2/SSE, NEON, scalar). Missing scalar fallback is a P0 reject — the scalar path is the correctness anchor.
2. **Document the saturating / overflow / signedness semantics** in the doc-comment. State explicitly what happens at edge cases (`i8::MIN`, `u8::MAX`, empty slices, indices out-of-range).
3. **Mandatory parity test** asserting all three backends produce identical output on a fixed-seed randomized corpus that includes edge cases (`i8::MIN`, `0`, `i8::MAX`, mantissa = -128, etc.). Use `proptest` or `quickcheck` if available; otherwise hand-roll 50+ test inputs.
4. **Bench against scalar** — record AVX-512 / NEON speedup ratios in the PR body. No SHIP/LAND gate required for the primitive PR itself (the consumer-side migration PRs will benchmark end-to-end), but a 0.5× anti-speedup ratio is a reject.
5. **`// SAFETY:` comments on every `unsafe` block** per ndarray's existing discipline (`CLAUDE.md` § Hard Rules).
6. **No new `is_*_feature_detected!` calls outside `src/hpc/simd_caps.rs`** — dispatch through the existing `simd_caps()` singleton.
7. **PR description must include the consumer site** (`lance-graph:crates/lance-graph-contract/src/mul.rs:NNN`, etc.) so the post-merge consumer-PR has a known target.

The `simd-savant` agent on the `lance-graph` side runs PRE-MERGE against every W1a PR to verify compliance.

---

## Cross-references

**ndarray-side (this repo):**
- `src/simd.rs` — the public re-export hub. New primitives surface here.
- `src/simd_avx512.rs` — AVX-512 typed wrappers (`I64x8`, `U64x8`, `I8x32`, `F32x16`, `F64x8`, …).
- `src/simd_avx2.rs` — AVX2 typed wrappers (`U8x32`).
- `src/simd_neon.rs` — NEON typed wrappers.
- `src/simd_ops.rs` — high-level vector→vector ops (`add_f32`, `mul_f32`, …).
- `src/simd_int_ops.rs` — integer batch ops (`add_i8`, `dot_i8`, `min_i8`, …).
- `src/hpc/simd_caps.rs` — runtime feature-detect singleton.
- `src/hpc/bitwise.rs` — already-exposed `hamming_distance_raw` + `popcount_raw` (slice case).

**lance-graph-side (the consumer driving this contract):**
- `AdaWorldAPI/lance-graph:.claude/knowledge/ndarray-vertical-simd-alien-magic.md` — full architectural doc + per-workload table
- `AdaWorldAPI/lance-graph:.claude/agents/simd-savant.md` — PRE-MERGE audit gate
- `AdaWorldAPI/lance-graph:.claude/board/EPIPHANIES.md` § `E-SIMD-SWEEP-1` — the 158-violation finding
- `AdaWorldAPI/lance-graph:.claude/board/TECH_DEBT.md` § `TD-NDARRAY-SIMD-*` and § `TD-SIMD-SWEEP-W*` — full debt ledger
- `AdaWorldAPI/lance-graph:.claude/specs/pr-sprint-13-simd-i4.md` — D-CSV-13b spec (the consumer workload spec)
- PR #398 (sprint-13 W-I1 retry) — the codex P1 (NEON OOB) + P2 (i8::MIN divergence) origin
- PR #399 (`simd-savant` card + autoattended-pattern doc) — invariant declaration
- PR #400 (architectural capture commit) — the canonical reference + tech-debt entries

**External references:**

> **Sourcing note (appended 2026-07-27):** the four intrinsic citations below
> were recorded as bare "Intel Intrinsics Guide" mentions with no link. The
> checkable source for intrinsic *semantics* in this workspace is GCC —
> declarations in `gcc/config/i386/*intrin.h`, plus an executable per-intrinsic
> oracle (scalar reference inline) in `gcc/testsuite/gcc.target/i386/`. See
> `.claude/knowledge/gcc-intrinsic-spec-reference.md` for the three-layer
> drill-down and a pinned SHA. Prefer that, at a pinned commit, over an
> unlinked guide reference. Original lines kept verbatim below.

- Intel Intrinsics Guide — `_mm512_abs_epi8` (VPABSB; does NOT saturate `i8::MIN`)
- Intel Intrinsics Guide — `_mm512_min_epu8` (VPMINUB; unsigned-byte minimum, used to clamp the VPABSB result)
- Intel Intrinsics Guide — `_mm512_popcnt_epi64` (VPOPCNTDQ; AVX-512 feature `avx512vpopcntdq`)
- Intel Intrinsics Guide — `_mm256_i32gather_epi32` (VPGATHERDD AVX2)
- ARM Architecture Reference — VQABS (`vqabsq_s8`, hardware-saturating)
- ARM Architecture Reference — VCNT (`vcntq_u8`, byte-wise popcount)
- Hambly & Lyons (2010), "Uniqueness for the signature of a path of bounded variation and the reduced path group"
- Cuchiero, Schmocker & Teichmann (2021), "Random feature neural networks learn Black-Scholes type PDEs without curse of dimensionality"
- Jirak (2016), "Berry-Esseen theorems under weak dependence" — the iron rule sigker bypasses

## Litmus tests (for any contributor proposing an addition to this queue)

> **Does the new primitive go on a typed-wrapper struct, or as a free function?**
> Free function = reject; the surface fragments. Struct method = accept.

> **Does the doc-comment state the edge-case behavior (saturating? wrapping? UB? scalar-fallback?)?**
> Missing = reject. The consumer needs to know without reading the code.

> **Are all three backends implemented (AVX*, NEON, scalar)?**
> Missing scalar = reject. Scalar is the correctness anchor.

> **Is there a parity test asserting all three backends produce identical output on a fixed-seed randomized corpus including edge cases?**
> Missing = reject. The codex P2 i8::MIN divergence on `lance-graph` PR #398 happened because no such test existed.

> **Is the consumer site cited in the PR description?**
> Missing = reject. We're shipping primitives for known workloads, not speculative ones.

## Third-party dependencies: neutralize by cfg BEFORE porting, port BEFORE removing (2026-07-28, PR #258)

The matryoshka invariant — all SIMD lives once, audited, inside
`ndarray::simd` — binds what **reaches the binary**, not what a dependency's
source tree *contains*. When a third-party crate carries its own intrinsics,
the escalation ladder is:

1. **Grep for the gate.** Most crypto/perf crates gate their vector backend
   behind one cfg or feature (`curve25519-dalek`:
   `#[cfg(curve25519_dalek_backend = "simd")] pub mod vector;` at
   `backend/mod.rs:42`, selected by build.rs from
   `CARGO_CFG_CURVE25519_DALEK_BACKEND`). One rustflags line in
   `.cargo/config.toml` compiles the whole surface out. Cost: zero code.
   Verify with a clean rebuild that the cfg actually flips
   (`cargo build -v | grep <cfg name>`).
2. **Port onto `ndarray::simd`** only if the intrinsics are load-bearing for
   a path we actually execute (the `vendor/chacha20` precedent — the ARX
   lane IS our hot path).
3. **Remove the dependency** only if neither applies — and then the claim
   "can't be neutralized" must cite the failed gate-grep, not the intrinsic
   count.

PR #258 is the cautionary worked example: an intrinsic count (57 sites, real)
was read as a porting obligation, escalated straight to step 3, and a merged
removal commit had to be reverted once the step-1 gate was found. "Contains
raw intrinsics" and "raw intrinsics are reachable" are different claims;
audit the second. Full record: board `EPIPHANIES.md` 2026-07-28 entry +
the `.cargo/config.toml` comment block.
