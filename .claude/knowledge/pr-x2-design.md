# PR-X2 — Generalize `aos_to_soa`/`soa_to_aos` to `<T, U, N>` + `#[soa(pad_to_lanes=N)]` macro attribute

> READ BY: all ndarray agents that touch the cognitive shader stack
> (savant-architect, l3-strategist, cascade-architect,
> cognitive-architect, arm-neon-specialist, sentinel-qa, product-engineer,
> truth-architect, vector-synthesis, splat3d-architect).
>
> **Design doc v1** — driven by the W3-W6 P2 savant review A4 finding
> (`aos_to_soa` hardwired to `f32` output) and the BlockedGrid / SIMD-staged
> kernel requirements that need `u64` (CausalEdge64), `u16` (BF16 carrier),
> and `u8` (palette) SoA fields.
>
> Parallel docs:
> - `.claude/knowledge/pr-x1-design.md` — MultiLaneColumn, Fingerprint::as_u8x64, array_window, simd::* sweep (prerequisite)
> - `.claude/knowledge/pr-x3-cognitive-grid-design.md` — BlockedGrid (consumer of pad_to_lanes tail-padding)
> - `.claude/knowledge/w3-w6-soa-aos-design.md` — W3-W6 foundation this PR extends
> - `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a layering rule
> - `.claude/knowledge/cognitive-distance-typing.md` — no-umbrella distance rule

## Context for a fresh session

If you arrive here without conversational context (token reset, new session, handover), here is the minimum you need to know:

1. **W3-W6 shipped** (PR #156, merged 2026-05-18). Added `SoaVec<T, N>`, `soa_struct!`, `aos_to_soa<T, N, F: Fn(&T) -> [f32; N]>`, `soa_to_aos<T, N, F>` to `src/hpc/soa.rs`. The conversion helpers are **hardwired to `f32` output** — `SoaVec<f32, N>`. The macro is generic over field type `T` but the conversions are not.
2. **PR #157 shipped** (P2 savant follow-up). Added f32-only-scope note to module header and ungated the integration test. The note says "non-f32 fields require a hand-rolled extract loop today; the public surface for generic-T conversion is a follow-up."
3. **PR-X1 (prerequisite, see `.claude/knowledge/pr-x1-design.md`)**: Adds `MultiLaneColumn`, `Fingerprint<8>::as_u8x64`, `array_window`, completes `simd::*` re-exports. PR-X2 can be designed in parallel with PR-X1 but should be sprint-sequenced after X1 lands (examples in PR-X2 use `U8x64`, `U64x8`, etc. from the X1 sweep).
4. **PR-X2 (this doc)**: Two changes to `src/hpc/soa.rs`:
   - Generalize `aos_to_soa<T, N, F: Fn(&T) -> [f32; N]>` to `aos_to_soa<T, U, N, F: Fn(&T) -> [U; N]>` so non-f32 element types (u64 for CausalEdge64, u16 for BF16, u8 for palette) work. Same generalization for `soa_to_aos`.
   - Add `#[soa(pad_to_lanes=N)]` field attribute to `soa_struct!` macro: pads the Vec for that field to the next multiple of N elements. Required for SIMD-staged kernels that need guaranteed tail alignment (so the last chunk is always a full N-lane chunk).
5. **W3-W6 A4 savant finding** (`w3-w6-p2-savant-review.md`, §A4): "A4 — `aos_to_soa<T, N, F: Fn(&T) -> [f32; N]>` is hardwired to `f32` output. Downstream consumers with `i8`/`u8`/`u16`/`bf16` SoA fields cannot use the public helper without writing their own." This PR is the direct response.
6. **Current `aos_to_soa` signature** (from `src/hpc/soa.rs:410`):
   ```rust
   pub fn aos_to_soa<T, const N: usize, F>(aos: &[T], extract: F) -> SoaVec<f32, N>
   where F: Fn(&T) -> [f32; N]
   ```
   The new signature introduces `U` as the element type of `SoaVec`:
   ```rust
   pub fn aos_to_soa<T, U, const N: usize, F>(aos: &[T], extract: F) -> SoaVec<U, N>
   where F: Fn(&T) -> [U; N]
   ```

## Why this exists

### `aos_to_soa<T, U, N>` generalization

The cognitive shader stack operates on multiple element types simultaneously:
- `u64` for CausalEdge64 mantissa cells (the L1 per-cell truth-bearing unit)
- `u16` for BF16 carrier values (depth field in `ShaderCellGrid`)
- `u8` for palette indices and alpha values
- `f32` for Gaussian splat means / covariances (already working)

With the W3-W6 `f32`-hardwired signature, a session wanting to build a `SoaVec<u64, 8>` from an AoS slice of `CausalEdge64` structs must write its own loop (identical to `aos_to_soa`'s body, just with `u64` instead of `f32`). This is the anti-pattern the W3-W6 helpers were designed to eliminate.

The generalization is additive: the new `aos_to_soa<T, U, N, F>` signature is a strict superset of the old `<T, N, F>` (which was equivalent to `<T, f32, N, F>`). Callers using the f32 path get a minor inference change (they may need to add `f32` as the `U` type parameter if turbofish is used) — see Q1.

### `#[soa(pad_to_lanes=N)]` attribute

SIMD-staged kernels operating on `SoaVec` fields need the field's `Vec<U>` to be a multiple of the lane width N. Without tail padding:
- A field with 101 elements walking 8-lane chunks gets chunks of [8, 8, ..., 8, 5] — the last chunk is shorter than 8, requiring a special scalar tail loop in the consumer.
- With `#[soa(pad_to_lanes=8)]`, the Vec is padded to 104 elements (next multiple of 8), and the consumer can use one uniform 8-lane loop without a tail case.

The padding fills added elements with `U::default()` (for `U: Default`) or a caller-specified sentinel (future extension, see Q4). Padded elements are beyond the semantic length of the field; `len()` still returns the logical count (101 in the example).

This is the same concept as W3-W6's `GaussianBatch::with_capacity` + eager-zero fill (`w3-w6-p2-savant-review.md`, §A1) but expressed declaratively as a macro attribute.

## The API

### 1. Generalize `aos_to_soa` and `soa_to_aos`

**Migration path for existing f32 callers:**

The old turbofish form `aos_to_soa::<_, 3, _>(...)` must become `aos_to_soa::<_, f32, 3, _>(...)` after this change. To ease migration, the design offers two options (savant should rule on Q1):

- **Option A (recommended)**: Rename the old f32-specific helpers to `aos_to_soa_f32` / `soa_to_aos_f32` and provide `aos_to_soa<T, U, N, F>` as the new generic entry. Breaking change for turbofish callers but clean API.
- **Option B (soft migration)**: Keep `aos_to_soa<T, N, F>` (f32-hardwired) as a deprecated alias and add `aos_to_soa_generic<T, U, N, F>` as the new generic entry. Avoids breaking callers but litters the API with two nearly-identical names.
- **Option C**: Change the signature in place; callers using return-type inference (not turbofish) are unaffected; turbofish callers need one new type param.

Design below uses **Option C** — change in place. Callers using return-type inference require no update. Turbofish callers add `f32` as `U`. This is the minimal-change path.

```rust
// Replaces the existing aos_to_soa in src/hpc/soa.rs

/// Deinterleave an AoS slice into a [`SoaVec<U, N>`] by extracting `N`
/// field values per item via the user-supplied `extract` closure.
///
/// `U` is the element type of the resulting `SoaVec`. Common values:
/// - `f32` — the original W3-W6 use case (Gaussian batch means, covariances)
/// - `u64` — CausalEdge64 mantissa cells
/// - `u16` — BF16 carrier values (depth, alpha BF16)
/// - `u8` — palette indices, quantized embeddings
///
/// Scalar implementation. A future bench-justified wave may add per-arch
/// SIMD gather (VPGATHERDD on AVX-512, LD3/LD4 on NEON). The public
/// signature is forward-compatible — the dispatcher will grow internal
/// per-arch arms without changing this signature.
///
/// This call is **scalar today**. It does not invoke any SIMD register
/// operations. The `SoaVec<U, N>` output is SIMD-friendly layout
/// (contiguous `Vec<U>` per field) but the conversion itself is scalar.
///
/// `T` need not be `Copy`; only the extracted `[U; N]` row is materialized.
///
/// # Inference
///
/// If `N` fails to infer from the closure return type, annotate either:
/// ```ignore
/// aos_to_soa::<_, u64, 3, _>(&aos, |it| [it.a, it.b, it.c]);
/// aos_to_soa(&aos, |it| -> [u64; 3] { [it.a, it.b, it.c] });
/// ```
///
/// # Example — u64 (CausalEdge64)
///
/// ```
/// use ndarray::hpc::soa::aos_to_soa;
/// struct Edge { src: u64, dst: u64, weight: u64 }
/// let aos = vec![
///     Edge { src: 1, dst: 2, weight: 10 },
///     Edge { src: 3, dst: 4, weight: 20 },
/// ];
/// let soa = aos_to_soa::<_, u64, 3, _>(&aos, |e| [e.src, e.dst, e.weight]);
/// assert_eq!(soa.field(0), &[1u64, 3]);
/// assert_eq!(soa.field(1), &[2u64, 4]);
/// assert_eq!(soa.field(2), &[10u64, 20]);
/// ```
///
/// # Example — f32 (backwards-compatible use case)
///
/// ```
/// use ndarray::hpc::soa::aos_to_soa;
/// struct Item { a: f32, b: f32 }
/// let aos = vec![Item { a: 1.0, b: 2.0 }, Item { a: 3.0, b: 4.0 }];
/// let soa = aos_to_soa::<_, f32, 2, _>(&aos, |it| [it.a, it.b]);
/// assert_eq!(soa.field(0), &[1.0f32, 3.0]);
/// assert_eq!(soa.field(1), &[2.0f32, 4.0]);
/// ```
///
/// # Example — u8 (palette indices)
///
/// ```
/// use ndarray::hpc::soa::aos_to_soa;
/// struct Cell { palette: u8, alpha: u8 }
/// let aos = vec![Cell { palette: 7, alpha: 255 }, Cell { palette: 3, alpha: 128 }];
/// let soa = aos_to_soa::<_, u8, 2, _>(&aos, |c| [c.palette, c.alpha]);
/// assert_eq!(soa.field(0), &[7u8, 3]);
/// assert_eq!(soa.field(1), &[255u8, 128]);
/// ```
#[inline]
pub fn aos_to_soa<T, U, const N: usize, F>(aos: &[T], extract: F) -> SoaVec<U, N>
where
    F: Fn(&T) -> [U; N],
{
    let mut soa = SoaVec::<U, N>::with_capacity(aos.len());
    for item in aos {
        soa.push(extract(item));
    }
    soa
}

/// Interleave a [`SoaVec<U, N>`] into an AoS `Vec<T>` by building each
/// item from the per-field values via the user-supplied `build` closure.
///
/// Scalar implementation. See [`aos_to_soa`] for the forward-compatibility
/// note on future SIMD acceleration.
///
/// This call is **scalar today**. See `aos_to_soa` for the element-type
/// scope note.
///
/// # Example — u64 round-trip
///
/// ```
/// use ndarray::hpc::soa::{aos_to_soa, soa_to_aos};
/// struct Edge { src: u64, dst: u64 }
/// let aos = vec![Edge { src: 10, dst: 20 }, Edge { src: 30, dst: 40 }];
/// let soa = aos_to_soa::<_, u64, 2, _>(&aos, |e| [e.src, e.dst]);
/// let back: Vec<Edge> = soa_to_aos(&soa, |[src, dst]| Edge { src, dst });
/// assert_eq!(back[0].src, 10);
/// assert_eq!(back[1].dst, 40);
/// ```
///
/// # Example — f32 (backwards-compatible)
///
/// ```
/// use ndarray::hpc::soa::{aos_to_soa, soa_to_aos};
/// struct Item { a: f32, b: f32, c: f32 }
/// let aos = vec![Item { a: 1.0, b: 2.0, c: 3.0 }];
/// let soa = aos_to_soa::<_, f32, 3, _>(&aos, |it| [it.a, it.b, it.c]);
/// let back: Vec<Item> = soa_to_aos(&soa, |[a, b, c]| Item { a, b, c });
/// assert_eq!(back[0].c, 3.0);
/// ```
#[inline]
pub fn soa_to_aos<T, U: Copy, const N: usize, F>(soa: &SoaVec<U, N>, build: F) -> Vec<T>
where
    F: Fn([U; N]) -> T,
{
    let n = soa.len();
    let fields = soa.all_fields();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let row: [U; N] = core::array::from_fn(|k| fields[k][i]);
        out.push(build(row));
    }
    out
}
```

**Bound change note**: The new `soa_to_aos` adds `U: Copy` (needed for `fields[k][i]` — indexing into a `&[U]` and copying the element). The old f32-hardwired version had this implicitly (f32 is Copy). This is the same bound already present conceptually; making it explicit in the signature. See Q2.

### 2. `#[soa(pad_to_lanes=N)]` in `soa_struct!`

**Macro syntax extension:**

```rust
soa_struct! {
    pub struct ShaderCellBatch {
        /// CausalEdge64 mantissa — pad to 8-lane U64x8 chunks.
        #[soa(pad_to_lanes = 8)]
        pub edge: u64,

        /// Palette index — pad to 64-lane U8x64 chunks.
        #[soa(pad_to_lanes = 64)]
        pub palette: u8,

        /// BF16 carrier depth — pad to 16-lane F32x16 chunks.
        #[soa(pad_to_lanes = 16)]
        pub depth: u16,

        /// Alpha channel — no padding attribute means no tail padding.
        pub alpha: u8,
    }
}
```

**What the macro generates for padded fields:**

For a field `pub edge: u64` with `#[soa(pad_to_lanes = 8)]`, the macro generates:
- `pub edge: Vec<u64>` — same as unpadded
- `pub fn push(...)` updated to track logical length separately from Vec length
- Internal `_edge_logical_len: usize` (private) for the "true" row count
- `edge_push_pad()` — pads Vec to next multiple of 8 by pushing `u64::default()` (0) entries
- `len()` returns `_edge_logical_len` (NOT `self.edge.len()`)

**Alternative (simpler, recommended):** instead of tracking logical vs padded length separately, the `push` method immediately pads the Vec after every push. This means `self.edge.len()` is always a multiple of 8, but `push` is slightly more expensive (up to N-1 extra pushes per row). This is acceptable because:
1. `push` is not in a hot path (data-loading phase, before SIMD-staged compute)
2. The SIMD kernel (hot path) gets a Vec of guaranteed-multiple-of-N length without special-casing

**However**: the pad-on-push approach loses the "true" row count. The macro must track `_logical_len: usize` separately.

**Full generated code for `ShaderCellBatch` (abbreviated):**

```rust
pub struct ShaderCellBatch {
    pub edge: Vec<u64>,
    pub palette: Vec<u8>,
    pub depth: Vec<u16>,
    pub alpha: Vec<u8>,
    // Generated private fields for padded lanes:
    _edge_logical_len: usize,    // true row count before padding
    _palette_logical_len: usize,
    _depth_logical_len: usize,
    // alpha has no pad_to_lanes, so no separate len tracker
}

impl ShaderCellBatch {
    pub fn new() -> Self {
        Self {
            edge: Vec::new(),
            palette: Vec::new(),
            depth: Vec::new(),
            alpha: Vec::new(),
            _edge_logical_len: 0,
            _palette_logical_len: 0,
            _depth_logical_len: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        // For padded fields, allocate to the next multiple of N lanes.
        let edge_cap = cap.div_ceil(8) * 8;
        let palette_cap = cap.div_ceil(64) * 64;
        let depth_cap = cap.div_ceil(16) * 16;
        Self {
            edge: Vec::with_capacity(edge_cap),
            palette: Vec::with_capacity(palette_cap),
            depth: Vec::with_capacity(depth_cap),
            alpha: Vec::with_capacity(cap),
            _edge_logical_len: 0,
            _palette_logical_len: 0,
            _depth_logical_len: 0,
        }
    }

    /// Append one row. For padded fields, the Vec is extended with
    /// `U::default()` padding so its length stays a multiple of the
    /// specified lane count. Padding elements are beyond `len()`.
    pub fn push(&mut self, edge: u64, palette: u8, depth: u16, alpha: u8) {
        self._edge_logical_len += 1;
        self.edge.push(edge);
        // Pad edge to next multiple of 8:
        while self.edge.len() % 8 != 0 {
            self.edge.push(u64::default());
        }

        self._palette_logical_len += 1;
        self.palette.push(palette);
        while self.palette.len() % 64 != 0 {
            self.palette.push(u8::default());
        }

        self._depth_logical_len += 1;
        self.depth.push(depth);
        while self.depth.len() % 16 != 0 {
            self.depth.push(u16::default());
        }

        // alpha: no pad_to_lanes — push normally.
        self.alpha.push(alpha);
    }

    /// Logical row count (does not include padding elements).
    ///
    /// # Panics
    ///
    /// In debug builds, panics if the logical lengths of padded fields
    /// disagree (a bug in custom mutation paths). In release, returns
    /// the logical length of the first padded field.
    pub fn len(&self) -> usize {
        let n = self._edge_logical_len;
        debug_assert!(
            self._palette_logical_len == n && self._depth_logical_len == n
                && self.alpha.len() == n,
            "ShaderCellBatch: field-length invariant violated"
        );
        n
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn clear(&mut self) {
        self.edge.clear();
        self.palette.clear();
        self.depth.clear();
        self.alpha.clear();
        self._edge_logical_len = 0;
        self._palette_logical_len = 0;
        self._depth_logical_len = 0;
    }

    /// Padded length of the `edge` field's Vec. Always a multiple of 8.
    /// Use this as the loop bound for 8-lane U64x8 SIMD kernels.
    ///
    /// # Example
    ///
    /// ```
    /// # use ndarray::soa_struct;
    /// # soa_struct! { pub struct ShaderCellBatch {
    /// #     #[soa(pad_to_lanes = 8)] pub edge: u64,
    /// #     pub alpha: u8,
    /// # }}
    /// let mut b = ShaderCellBatch::new();
    /// b.push(1u64, 0u8);
    /// b.push(2u64, 0u8);
    /// b.push(3u64, 0u8);
    /// assert_eq!(b.len(), 3);
    /// assert_eq!(b.edge_padded_len(), 8); // padded to next multiple of 8
    /// ```
    pub fn edge_padded_len(&self) -> usize {
        self.edge.len()
    }
}
```

**Accessor naming convention for padded-length**: `{field_name}_padded_len()` is generated only for padded fields. Unpadded fields do not get this method.

### Changes to `SoaVec<T, N>` (if any)

The `SoaVec` generic struct itself does NOT need a padding story for this PR — `pad_to_lanes` is a feature of the `soa_struct!` macro's generated named structs only. If a caller wants a padded `SoaVec`, they pad the underlying Vec manually before passing it to the SoA container. This avoids complicating the `SoaVec` API for a use case that is well-served by the named-struct macro. See Q3.

## Layering rule recap

PR-X2 lives at the **user-code layer** (same as `hpc/soa.rs`). The W1a contract (`vertical-simd-consumer-contract.md`) requires:

1. No `#[target_feature]` in `src/hpc/soa.rs` additions
2. No `cfg(target_feature = "...")` gates
3. No `use crate::simd_avx512::*` etc.
4. No raw intrinsics

The `pad_to_lanes` attribute is a pure macro-expansion-time code generation feature. It emits standard Rust (`Vec::push` + a `while len % N != 0 { push(default) }` loop). Zero SIMD, zero unsafe.

## Distance-typing guardrail

PR-X2 is **layout-only** — it generalizes the element type of AoS↔SoA conversion and adds tail padding. Neither change bakes in a distance metric. Workers MUST NOT:

- Add `fn distance(...)` overloads parameterized on `U`
- Add `enum DistanceMetric` or metric-dispatch logic
- Add a `bulk_distance<T, U>` umbrella function

See `.claude/knowledge/cognitive-distance-typing.md`.

## Tests required

### `aos_to_soa<T, U, N, F>` (additions to `src/hpc/soa.rs`)

- `aos_to_soa::<_, u64, 3, _>` round-trip: AoS of structs with u64 fields → `SoaVec<u64, 3>` → back to AoS
- `aos_to_soa::<_, u8, 2, _>` round-trip: palette index struct
- `aos_to_soa::<_, u16, 2, _>` round-trip: BF16 carrier
- `aos_to_soa::<_, f32, 3, _>` — existing f32 tests must still pass after generalization (regression)
- `soa_to_aos<T, u64, N, F>` round-trip matches `aos_to_soa` output
- `aos_to_soa` with empty input → empty `SoaVec<u64, 3>`
- `soa_to_aos` with empty input → empty `Vec<T>`

### `#[soa(pad_to_lanes=N)]` macro attribute

- Single padded field: `push` 1 element → `padded_len()` = N; logical `len()` = 1
- Single padded field: `push` N elements → `padded_len()` = N; `len()` = N (no extra padding needed)
- Single padded field: `push` N+1 elements → `padded_len()` = 2*N; `len()` = N+1
- Mixed: struct with 2 padded fields and 1 unpadded field — verify logical `len()` is consistent across all fields
- `clear()` resets logical lengths AND `Vec` contents
- `with_capacity(cap)` preallocates to `cap.div_ceil(N) * N` for padded fields
- Padding elements are `U::default()` (verify `edge.len() - logical_len` elements are 0 for `u64`)
- `push` on N-lane-padded struct with `N=1` (degenerate: no actual padding; only 1 pad needed per push, which is the element itself)
- Struct with no padded fields generates identical code to the old unpadded macro (regression)
- `#[derive(Clone)]` passthrough still works on struct with padded fields
- `pub` / private visibility on padded fields respected

### Doc-tests

Every new/changed public fn has a working `# Example` doctest (included in the API section above). Module-level doctest updated to show the canonical `u64` CausalEdge64 use case:

```rust
//! ```
//! use ndarray::hpc::soa::aos_to_soa;
//! struct CausalEdge { src: u64, dst: u64 }
//! let aos = vec![CausalEdge { src: 1, dst: 2 }, CausalEdge { src: 3, dst: 4 }];
//! let soa = aos_to_soa::<_, u64, 2, _>(&aos, |e| [e.src, e.dst]);
//! assert_eq!(soa.field(0), &[1u64, 3]);
//! assert_eq!(soa.field(1), &[2u64, 4]);
//! ```
```

## Out of scope

1. **Padded `SoaVec<T, N>`** (pad_to_lanes on the generic struct) — the macro's named struct covers the use case; SoaVec stays lean
2. **`pad_to_lanes` with non-default fill value** (`#[soa(pad_to_lanes=8, fill=0xDEAD_BEEFu64)]`) — future extension, document as Q4
3. **SIMD-accelerated deinterleave** (VPGATHERDD / LD3/LD4) — future wave, bench-gated
4. **`aos_to_soa_strided`** (stride-based, no closure) — future PR, planned at `w3-w6-p2-savant-review.md` §E2
5. **`soa_to_aos_generic` alias** (if Option A or B naming is chosen per Q1 ruling) — only relevant if the in-place signature change is rejected
6. **`#[soa(pad_to_lanes=N)]` on `SoaVec` constructor** (`SoaVec::new_padded(cap, lanes)`) — nice-to-have, out of scope v1
7. **Integration with `blocked_grid_struct!`** (PR-X3 macro for SoA-of-grids) — PR-X3 owns its padding story separately via `BlockedGrid::new_with_pad`

## Worker decomposition (SEQUENTIAL)

Two Sonnet sprint workers + 1 Opus coordinator. Sequential — B (macro attribute) depends on the compiler/test infra being green after A's generalization changes.

| # | Phase | Agent role | Scope | Coordinator action |
|---|---|---|---|---|
| 1 | **plan** | (this doc, v1) | design-doc drafter | commit to branch |
| 2 | **review** | plan-review savant | rules on Q1–Q5; READY or NEEDS-FIX | apply P0/P1; commit v2 |
| 3 | **sprint worker A** | `src/hpc/soa.rs` — generalize `aos_to_soa` + `soa_to_aos` signatures; update all doctests + inline tests for f32 regression + new u64/u8/u16 cases. Update module header §"Element-type scope" to say "now generic over U". | Single commit. | verify green; cherry-pick |
| 4 | **sprint worker B** | `src/hpc/soa.rs` — extend `soa_struct!` macro to handle `#[soa(pad_to_lanes=N)]` field attributes. Emit `_field_logical_len` private fields, update `push`/`len`/`clear`/`with_capacity`, emit `{field_name}_padded_len()` accessors. All macro tests. | Single commit. Depends on A (same file, non-overlapping lines). | verify green; cherry-pick |
| 5 | **codex P0 audit** | audits combined diff (A + B) | zero `#[target_feature]`, zero per-arch imports, zero raw intrinsics, all `// SAFETY:` present, all doctests work, `pad_to_lanes` with N=0 produces a compile-time assertion (const { assert!(N > 0) } in macro expansion) | apply P0 fixes |
| 6 | **PR open + P2 savant** | P2 ergonomics review | naming (Q1), inference ergonomics, doc clarity | same-day follow-up if recommended |

## Verification commands

```bash
cargo check -p ndarray --no-default-features --features std
cargo test -p ndarray --lib --no-default-features --features std hpc::soa
cargo test --doc -p ndarray --no-default-features --features std hpc::soa
cargo fmt --all -- --check
cargo clippy -p ndarray --no-default-features --features std -- -D warnings
```

All five must pass green.

## Cross-references

- `.claude/knowledge/w3-w6-soa-aos-design.md` — W3-W6 foundation; §W5+W6 API contracts this PR generalizes
- `.claude/knowledge/w3-w6-p2-savant-review.md` — A4 finding that drove this PR; E1 factor-body note (scalar body extraction) applies to the generalized signature
- `.claude/knowledge/pr-x1-design.md` — prerequisite: `simd::*` sweep must include U64x8/U8x64/etc. before PR-X2 examples compile
- `.claude/knowledge/pr-x3-cognitive-grid-design.md` — BlockedGrid; uses `pad_to_lanes` conceptually at the grid level (independent implementation)
- `.claude/knowledge/vertical-simd-consumer-contract.md` — W1a contract; generalization must stay at user-code layer
- `.claude/knowledge/cognitive-distance-typing.md` — no-umbrella distance rule; U-generic helpers must not embed metric logic
- `src/hpc/soa.rs` — the file this PR modifies (lines 373–456 for `aos_to_soa`/`soa_to_aos`, lines 318–371 for the macro)

## Open questions (for the plan-review savant to rule on)

1. **Q1 — `aos_to_soa` migration path**: Design uses Option C (in-place signature change: add `U` type param). Existing f32 callers using return-type inference are unaffected; turbofish callers must add `f32` as `U`. Savant: (a) confirm Option C, (b) prefer Option A (rename old to `aos_to_soa_f32`, new is `aos_to_soa`), or (c) prefer Option B (keep old, add `aos_to_soa_generic`)?

2. **Q2 — `U: Copy` bound on `soa_to_aos`**: The generalized `soa_to_aos` builds `row: [U; N]` via `core::array::from_fn(|k| fields[k][i])`. Indexing `&[U]` returns `U` by copy, so `U: Copy` is required. The old `f32`-hardwired version had this implicitly. Savant: confirm `U: Copy` is the correct bound, or should it be `U: Clone` + a `.clone()` call (weaker constraint but slightly less ergonomic)?

3. **Q3 — `pad_to_lanes` on `SoaVec`**: Design leaves `SoaVec<T, N>` unpaddable. A padded variant would require `SoaVec::push_padded(row, lanes: usize)` or a type-level lanes parameter `SoaVec<T, N, PAD = 1>`. Design rationale: `soa_struct!` macros cover the use case; `SoaVec` stays minimal. Savant: confirm `SoaVec` stays lean, or add a `SoaVec::with_padding(cap, lanes)` constructor for ad-hoc cases?

4. **Q4 — `pad_to_lanes` fill value**: Current design pads with `U::default()`. Some consumers (CausalEdge64) may want `0` (causally-null edge, which IS default for u64). Others may want `0xFFFF_FFFF_FFFF_FFFFu64` (uninitialized sentinel). Design defers non-default fill to a follow-up `#[soa(pad_to_lanes=8, fill=0xFFFFu64)]` form. Savant: confirm deferral is correct, or add the `fill` sub-attribute now?

5. **Q5 — `pad_to_lanes` with N=1**: `pad_to_lanes = 1` is a degenerate case (padding to next multiple of 1 = every push is already at a multiple of 1, no padding needed). Design should emit a `const { assert!(N > 1, "pad_to_lanes=1 is a no-op") }` in the macro expansion (either a compile-time error or a warn). Savant: error or warn, or silently allow (N=1 is harmless if slightly noisy)?

## Done criteria

PR-X2 is done when:
- `aos_to_soa<T, U, N, F>` and `soa_to_aos<T, U, N, F>` compile and test green for U = f32, u64, u8, u16
- Existing f32 round-trip tests pass without modification (inference-based callers unaffected)
- `#[soa(pad_to_lanes=N)]` attribute generates correct padded Vecs and logical-length tracking
- Codex P0 audit: 0 P0 (zero SIMD intrinsics, zero distance-aware API, all doctests work, `const { assert!(N > 0) }` guard in macro expansion)
- Layering rule verified per W1a contract
- Distance-typing guardrail verified
- P2 savant review delivers SHIP verdict
