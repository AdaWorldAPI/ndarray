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

## W1.5 — sigker primitives (gate now OPEN; #6 and #7 SHIPPED)

> **⊘ CORRECTED (2026-09-04) — this section originally read as three
> future/deferred primitives, gated on a certification that had not yet
> happened. That framing is stale: the gate opened 2026-05-07
> (`lance-graph:crates/jc/src/lib.rs:26`, "Pillar 11 activated
> 2026-05-07"; `jc/src/hambly_lyons.rs` is a live module, `pub mod
> hambly_lyons;` at `jc/lib.rs:37`), and W1.5-#6 and W1.5-#7 have since
> shipped (ndarray PR #293, PR #294). The original sketches below are
> kept verbatim, each followed by a correction block, rather than
> silently rewritten — see the standing note at the end of this section
> for why every remaining sketch (#8) must be treated as unverified.**

Three primitives were queued behind a certification gate. `crates/sigker` is `lance-graph`'s path-signature codec — it's pure-scalar Rust today (zero raw intrinsics, zero ndarray dep), and is positioned as the **Index-regime third encoding lane** alongside palette-distance (bgz17) and NSM tiling (deepnsm). It explicitly bypasses the `I-NOISE-FLOOR-JIRAK` iron rule (Jirak 2016 Berry-Esseen for weak-dependence data) via Hambly-Lyons 2010 path-signature uniqueness.

> **⊘ CORRECTED:** the paragraph above and the "When `jc Pillar 11`… lights
> up" sentence below described the gate as future-conditional. **The gate is
> OPEN as of 2026-05-07.** `crates/sigker` is also no longer purely scalar —
> its consumer wiring for #6 is live (see #6 below).

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

> **⊘ CORRECTED — SHIPPED (ndarray PR #293).** The sketch above is WRONG on
> lane type and is superseded, not merely deferred. Real state:
> - Shipped as `ndarray::hpc::signature_pde::signature_pde_sweep`, an
>   anti-diagonal SIMD wavefront sweep.
> - **Lane type is f64-based, NOT `F32x16` as sketched.** The real
>   consumer (`lance-graph crates/sigker`) works in `f64`/`Vec<f64>`, so the
>   shipped primitive is built on `F64x8`.
> - **Consumer wired:** `lance-graph crates/sigker/src/kernel.rs:35` imports
>   it directly; `sigker`'s `Cargo.toml` now carries ndarray as a mandatory
>   path dep (no longer the "zero ndarray dep" state described above).

### W1.5-#7 — `TD-NDARRAY-SIMD-RANDOMIZED-PROJECTION`

Cuchiero-Schmocker-Teichmann (2021) randomized signatures: Gaussian random-matrix-vector update with `F32x16` state. Same closure-batch shape as W1a-#1, different lane type.

> **⊘ CORRECTED — SHIPPED (ndarray PR #294).** The sketch above is WRONG on
> lane type AND on data ownership. Real state:
> - Shipped as `ndarray::hpc::randomized_signature`, exposing
>   `randomized_signature_sweep` / `_sweep_with` / `_step`, plus
>   `INCREMENT_EPSILON = 1e-15`.
> - **Lane type is `F64x8`, not `F32x16`.**
> - **Ownership model was wrong too:** the sketch implied Gaussian entries
>   re-derived per step from `(seed, depth)`. In reality the projections are
>   materialized ONCE per encoder instance (seeded SplitMix64 + Box-Muller)
>   and reused across every path and step — so the primitive must CONSUME
>   caller-owned buffers, not generate them internally.
> - **`k` is a runtime value** (32…4096 in the consumer's own tests), not a
>   fixed lane width — the hot path is a `k×k` GEMV plus an axpy per path
>   dimension, O(T·d·k²), not a single-register lane update.
> - **Consumer NOT yet wired:** `lance-graph crates/sigker/src/randomized.rs:95`
>   `RandomizedSignatureBuilder::encode` still runs its own scalar loop.
>   Wiring is in flight in a parallel task as of this correction — treat as
>   in-flight, not done.

### W1.5-#8 — `TD-NDARRAY-SIMD-LYNDON-PACK`

Log-signature compression in the Lyndon basis of the free Lie algebra (7-13× compression, lossless). Pack/unpack primitives on `I16x16` state with combinatorial-index awareness.

> **⊘ CORRECTED — still unbuilt, but NO LONGER GATED** (Pillar 11 is active,
> see above). **The `I16x16` state sketch above is UNVERIFIED against the
> real consumer** (`lance-graph crates/sigker/src/log_signature.rs`) — the
> equivalent sketches for #6 and #7 were BOTH wrong on lane type (2-for-2
> miss rate). Do not implement from this sketch. Read the actual consumer
> source first and confirm the real lane type before writing any code.

**Standing note (2026-09-04):** the pattern across #6 and #7 is that this
doc's API sketches predate the consumer code and drift from it — both
missed the lane type, and #7 also missed the ownership model and the
runtime-`k` shape. Treat every remaining sketch in this section as a
starting hypothesis to verify against `lance-graph crates/sigker`, never as
a spec.

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

---

## The masking layer: `simd_masking_ops.rs` and the two laws above the backends (2026-09-13)

Operator-ruled during the mask-RISC arc; recorded here because every W1a
primitive that turns values into masks, composes masks, or reduces under a
mask is now expected to land in THIS shape rather than beside `add_i8`.

```text
consumers                 semantic ops only — TERNLOG<IMM>, AND, XOR, COUNT
simd_masking_ops.rs       slice/chunk/tail ergonomics, *_assign forms,
                          mask composition, masked reductions — never an ISA
simd.rs                   architecture-agnostic types, compile-time selection
simd_{avx512,avx2,neon,wasm,scalar}.rs   peer backends, each owns realization
```

- **Polyfill law.** Every public mask/SIMD primitive a consumer uses has a
  compile-time implementation in ALL FIVE backends; scalar is a peer, not a
  fallback; no runtime ISA dispatch above `simd.rs`; hardware-specific
  optimisation (including truth-table specialisation of `ternlog`) lives in
  the backend file only. A consumer that branches on ISA is a violation.
- **Backend law.** No shared generic implementation body under the backends.
  Shared tests and shared *generated* truth-table logic are fine
  (`tools/gen_ternlog_bodies.py` emits backend-LOCAL bodies between
  `GEN-TERNLOG` markers); a common function the backends call into is not.
- **Placement rule for new work.** Backend semantics (what `U64x8::ternlog`
  *is*) never move into `simd_masking_ops.rs`; slice ergonomics, tail
  handling, reusable-destination forms and fused conveniences never move into
  a backend. A facade function that cannot be one delegation is the signal
  the substrate is missing a word (the missing-capability STOP rule).
- **Acceptance for a mask primitive** adds one row to the criteria above:
  the cross-ISA parity harnesses (`crates/wasm-simd-parity`, run under node;
  `crates/neon-simd-parity`, run under qemu) must carry the primitive's
  check — the x86 `cargo test` suite never compiles `simd_wasm.rs` or
  `simd_neon.rs`, so a backend body that only x86 tests cover is unproven on
  the target it was written for. The 256-table `ternlog` arm is the template.
- **AArch64 without the hardware — the acceptance ladder.** A NEON body is
  authored from the LLVM/Clang intrinsic corpus + Rust `core::arch::aarch64`
  declarations and proven by: (1) cross-target compile; (2) the parity harness
  under qemu (CI); (3) cross-compiled assembly showing the expected NEON ops
  and no unexpected scalarisation — a `to_array()`-per-lane loop passed rungs
  1, 2 and 4 and FAILED 3 (536 scalar vs 4 vector ops), which is why rung 3 is
  not optional; (4) exhaustive reference parity; (5) hardware benchmarking as
  a later performance gate. Command shape for rung 3:
  `cargo rustc --release --manifest-path crates/neon-simd-parity/Cargo.toml
  --target aarch64-unknown-linux-gnu -- --emit=asm`, then count
  `(and|orr|eor|bic|orn) v*.16b` against `(and|orr|eor|bic) w*,`.
- **`unsafe` at the intrinsic boundary — where and why (measured, 1.98.1,
  `tools/safe_intrinsic_probe`).** x86 and aarch64 SIMD intrinsics are safe
  fns whose CALL requires the caller to carry the matching
  `#[target_feature]`; build-config features do not count (rustc says so in
  the E0133 note), and a safe annotated fn called from a plain fn fails the
  same way. The per-fn annotation is not the fix: a `simd_{arch}.rs` file is
  compiled for exactly one target CPU, selected by `cfg`, so the feature is
  already a property of the file — restating it on every fn is illogical and
  propagates to every safe caller; rustc just does not read the `cfg` as
  evidence. wasm32
  simd128 intrinsics are callable from plain safe code. Rule: a backend
  method owns exactly one expression-narrow `unsafe` at its intrinsic
  boundary with a SAFETY line; wasm bodies carry none; nothing above a
  backend file is ever `unsafe`. Re-run the probe after a toolchain bump —
  the day rustc counts baseline features, the aarch64/x86 rows flip and the
  blocks come out.
- **Five execution flavours, one semantic surface (operator, 2026-09-13).**
  (1) x86-64-v3 default/CI → `simd_avx2`; (2) AVX-512/v4 → `simd_avx512`;
  (3) `target-cpu=native` → backend chosen from the build host's CPUID;
  (4) `nightly-simd` → `core::simd`; (5) `runtime-dispatch` → LazyLock
  detection then a specialised kernel. `#[target_feature]` propagation is not
  the architecture: the selected backend (or the LazyLock branch) is the
  capability proof, intrinsics stay at the backend's narrow `unsafe`
  boundary, and `simd_masking_ops` / mask-RISC / consumers never inherit an
  ISA calling contract. A mask primitive is never routed through Scalar
  because rustc wants `unsafe` at an intrinsic. **Audit rule:** a new mask
  primitive is proven on every flavour whose backend has a native lane type
  for it — check `simd.rs`'s re-export arm per target, not the file you
  authored in; `U64x8`/`I32x16` resolved to scalar on aarch64 and wasm32
  until the #306 audit caught it.
- **Measure the shipped symbol before overriding it (2026-09-14, the AVX2
  arm of the mask family).** The plan said "replace the `avx2_int_type!`
  array polyfills with native `[__m256i; 2]`"; the codegen oracle
  (`.claude/knowledge/simd-codegen-oracle/`, Group F) said six of the ten
  mask shapes — every ternlog ladder, andnot, popcnt, xor_popcount — were
  ALREADY packed from scalar source, and four were not (u64 rotate, i32
  horizontal min/max, and the two compare-to-bitmask forms, which were
  *mixed*: mostly packed with lanes 0 and 13–15 peeled to scalar). Only the
  four got intrinsic realizations. Rule: a polyfill lane loop is not scalar
  because it is spelled as a loop; it is scalar when `--emit asm` on the
  shipped method says so — and "mostly packed" is a category the oracle
  must be able to report, because a peel is invisible to any parity test.
