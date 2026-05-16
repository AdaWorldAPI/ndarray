# Vertical SIMD Consumer Contract — lance-graph W1a

> Canonical spec for the 5 P0 SIMD primitives that AdaWorldAPI/lance-graph
> requires from this ndarray fork before consumer-side migrations can land.

## Context

A PRE-MERGE audit of lance-graph `main` (2026-05-16) surfaced **158 raw-intrinsic
violations** across 5 consumer crates plus 3 missing primitives here that block
clean remediation. This document is the **binding contract** between ndarray's
SIMD surface and its primary consumer.

## Design Pattern

ndarray's SIMD surface is designed AS-IF for lance-graph's exact workloads:

- **Struct methods on typed wrappers** — `I8x16`, `U8x32`, `F32x16`, `U64x8`, …
- **Closure-parameterized batch primitives** — absorb consumer domain semantics
- **Consumers see zero raw intrinsics, zero `cfg(target_arch)`, zero runtime
  feature-detect** — they call `I8x16::from_i4_packed_u64(...)`,
  `I8x16::saturating_abs(...)`, `batch_packed_i4_16(..., |lanes, aux| { ... })`
- **This repo (the polyfill) owns** dispatch, chunking, tail handling, scalar fallback

---

## W1a Queue — 5 Primitives

### TD-NDARRAY-SIMD-UNPACK-I4-16D

**API:**
```rust
impl I8x16 {
    /// Unpack 8 packed i4 pairs from a u64 into 16 × i8 lanes (sign-extended).
    fn from_i4_packed_u64(packed: u64) -> Self;
}

/// Closure-batch: owns chunking, dispatch, tail handling.
/// `f` receives 16 i8 lanes + aux slice position; returns accumulator update.
fn batch_packed_i4_16<E, F>(
    packed: &[u64],
    aux: &[E],
    out: &mut [i8],
    f: F,
) where F: Fn(I8x16, &[E]) -> I8x16;
```

**Consumer:** `lance-graph::mul::i4_eval::batch` (5 fns)

**Per-arch implementation:**
- **AVX-512:** `VPSHUFB` nibble shuffle + sign-extend via arithmetic shift
- **NEON:** `VTBL` permute + `VSHR`/`VAND` nibble extraction
- **Scalar:** shift-and-mask loop, 2 nibbles per byte

---

### TD-NDARRAY-SIMD-SATURATING-ABS-I8

**API:**
```rust
impl I8x16 {
    /// Saturating absolute value: abs(i8::MIN) → i8::MAX (not i8::MIN).
    fn saturating_abs(self) -> Self;
}
```

**Consumer:** lance-graph PR #398 Direction-B fix

**CRITICAL — VPABSB correction:**

The original lance-graph PR #400 capture claimed `_mm512_abs_epi8` saturates
`i8::MIN → 127` by ISA. **This is WRONG.** VPABSB returns `0x80` for input
`0x80` (i.e., `abs(i8::MIN) = i8::MIN` because +128 doesn't fit in i8).

**Correct AVX-512 implementation:**
```rust
// SAFETY: self.0 is a valid __m512i from construction
unsafe {
    let raw_abs = _mm512_abs_epi8(self.0);
    // VPMINUB: unsigned min clamps 0x80 (=128 unsigned > 127) to 0x7f
    let clamped = _mm512_min_epu8(raw_abs, _mm512_set1_epi8(0x7f));
    Self(clamped)
}
```

**Per-arch implementation:**
- **AVX-512:** `VPABSB` + `VPMINUB` clamp (as above)
- **AVX2:** `_mm256_abs_epi8` + `_mm256_min_epu8(..., splat(0x7f))`
- **NEON:** `vqabsq_s8` — hardware-saturating (the `q` suffix). No fixup needed.
- **Scalar:** `i8::saturating_abs()` — correct by definition.

**Mandatory parity test:**
```rust
let input = I8x16::splat(i8::MIN);
assert_eq!(input.saturating_abs().lane_i8::<0>(), i8::MAX);
```

The widen-then-negate trick used in lance-graph PR #398's `mul.rs` is NOT a
substitute — the new primitive must produce saturating semantics in the
byte-wide register without widening.

---

### TD-NDARRAY-SIMD-GATHER

**API:**
```rust
impl U16x8 {
    /// Gather 8 × u16 values from `table` at the given indices.
    /// Panics (debug) / UB-free (release) if any index ≥ table.len().
    fn gather_u16(table: &[u16], indices: [u32; 8]) -> Self;
}

/// Palette lookup: gather 8 u8 values from a 256-entry palette.
fn palette_lookup_u8x8(palette: &[u8; 256], indices: [u8; 8]) -> [u8; 8];
```

**Consumer:** `bgz17/src/simd.rs:88`

**Per-arch implementation:**
- **AVX2/AVX-512:** `_mm256_i32gather_epi32` reads 32 bits per slot. Since the
  source is `&[u16]`, each gather slot reads 4 bytes starting at
  `&table[index]`. **Correction (Codex P2):** To avoid OOB reads at
  `table[len-1]`, the implementation MUST:
  1. Allocate or use a padded table with 2 extra trailing bytes, OR
  2. Use scalar fallback for indices where `index == table.len() - 1`, OR
  3. (Preferred) Gather from a `&[u32]` temporary produced by zero-extending
     the u16 table at ingest time, OR
  4. Mask the gathered 32-bit values to 16 bits (`_mm256_and_si256(..., 0xFFFF)`)
     AND ensure the source allocation has ≥2 bytes of padding past the last
     element (i.e., the table slice is backed by an allocation at least
     `table.len() * 2 + 2` bytes).

  The bounds contract is: `max(indices) < table.len()`. The implementation must
  guarantee no UB even when `max(indices) == table.len() - 1`.
- **NEON:** No hardware gather for u16. Scalar lane-by-lane load into vector.
- **Scalar:** Direct indexing loop.

---

### TD-NDARRAY-SIMD-PREFETCH

**API:**
```rust
/// Prefetch read hint — temporal locality level 0/1/2.
/// No-op on architectures without prefetch instructions.
#[inline(always)]
fn prefetch_read_t0<T>(ptr: *const T);
#[inline(always)]
fn prefetch_read_t1<T>(ptr: *const T);
#[inline(always)]
fn prefetch_read_t2<T>(ptr: *const T);
```

**Consumer:** `bgz17/src/prefetch.rs:96,100`

**Per-arch implementation:**
- **x86:** `_mm_prefetch::<_MM_HINT_T0/T1/T2>(ptr as *const i8)`
- **NEON/AArch64:** `__prefetch(ptr)` (ARM has only one hint level in stable intrinsics)
- **Scalar/fallback:** No-op `#[inline(always)]` empty function

---

### TD-NDARRAY-SIMD-POPCOUNT-U64

**API:**
```rust
impl U64x8 {
    /// Per-lane population count: count set bits in each u64 lane.
    fn popcnt(self) -> Self;

    /// XOR two vectors then popcount each lane (Hamming distance).
    fn xor_popcount(self, other: Self) -> Self;
}
```

**Consumer:** `holograph/hamming.rs`, `blasgraph/types.rs`

**Per-arch implementation:**
- **AVX-512 VPOPCNTDQ:** `_mm512_popcnt_epi64` directly. Feature flag `avx512vpopcntdq`.
- **AVX-512 without VPOPCNTDQ:** Mula's algorithm — `_mm512_shuffle_epi8` (VPSHUFB)
  for per-nibble LUT popcount on bytes, then `_mm512_sad_epu8` against zero to
  horizontally sum bytes within each 64-bit lane.
- **NEON:** Per-u64 lane popcount via widening reduction:
  ```
  vcntq_u8        → 16 × u8 byte-popcounts
  vpaddlq_u8      → 8 × u16 (pairwise add bytes)
  vpaddlq_u16     → 4 × u32 (pairwise add u16s)
  vpaddlq_u32     → 2 × u64 (pairwise add u32s)
  ```
  This produces one count per u64 lane. **Correction (Codex P2):** Do NOT use
  `vaddvq_u8` — it reduces the entire 128-bit vector to a single scalar,
  merging counts across u64 lanes. The `vpaddlq_*` widening cascade preserves
  per-lane boundaries.
- **Scalar:** `u64::count_ones()` per lane.

---

## W1.5 — Deferred Primitives

Gated on lance-graph `sigker` certification (Pillar 11 activation):

| ID | Primitive | Depends on |
|----|-----------|-----------|
| W1.5-#7 | signature-PDE-sweep | W1a-#1 closure-batch shape |
| W1.5-#8 | randomized-projection | W1a-#1 batch pattern |
| W1.5-#9 | lyndon-pack | W1a-#5 popcount |

The W1a closure-batch shape (`batch_packed_i4_16`) is the foundation for W1.5-#7
and #8. Design W1a-#1 with this forward-compatibility in mind.

---

## Acceptance Criteria (per W1a PR)

1. All three backends (AVX-512/AVX2, NEON, scalar) — scalar is the correctness anchor
2. Doc-comment states saturating/overflow/signedness semantics explicitly
3. Mandatory parity test on randomized + edge-case corpus (≥ 64 randomized vectors)
4. No new `is_*_feature_detected!` outside `src/hpc/simd_caps.rs`
5. `// SAFETY:` comments on all `unsafe` blocks
6. Consumer call-site cited in PR description
7. `cargo clippy -- -D warnings` passes

---

## Cross-References

- AdaWorldAPI/lance-graph PR #399 — simd-savant agent + autoattended-multiagent pattern
- AdaWorldAPI/lance-graph PR #400 — architectural capture (VPABSB correction origin)
- AdaWorldAPI/lance-graph PR #398 — codex P1 (NEON OOB) + P2 (i8::MIN divergence)
- Intel Intrinsics Guide: `_mm512_abs_epi8`, `_mm512_min_epu8`, `_mm512_popcnt_epi64`, `_mm256_i32gather_epi32`
- ARM Architecture Reference: `VQABS` (`vqabsq_s8`), `VCNT` (`vcntq_u8`), `VPADDL`

---

## Gating Relationship

> ndarray ships these 5 primitives → lance-graph remediation PRs land →
> 158 raw-intrinsic violations drop to 0 → sigker certification unblocks W1.5

Consumer-side migrations **cannot proceed** until these primitives ship.
The `simd-savant` agent on the lance-graph side runs PRE-MERGE against every
W1a PR to verify compliance.
