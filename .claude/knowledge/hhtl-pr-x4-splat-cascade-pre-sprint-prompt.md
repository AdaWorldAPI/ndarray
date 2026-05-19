# HHTL Substrate — PR-X4 Pre-Sprint Prompt (4×4 splat cascade)

> Date: 2026-05-19 (post PR #163 + PR #164 merge; master tip `13dfcf9d`)
>
> Status: Pre-sprint kickoff prompt for the W4-W5 splat cascade sprint.
> Companion to:
> - `pr-x4-design.md` — master design doc (merged PR #162, `c8f4af68`)
> - `hhtl-gridlake-pre-sprint-prompt.md` — substrate carrier PR-X4 consumes
> - `hhtl-pr-x14-substrate-contract-prompt.md` — contract crate PR-X4
>   consumes
> - `hhtl-substrate-execution-prompt.md` — W1-W8 master schedule
> - `pr-arithmetic-inventory.md` § L4-L8 splat-shader gaps
> - `pp13-brutally-honest-tester-verdict.md` § P0-4 — the L4 Hilbert-3D bug
>   carry-over from PR-X10 A12b
>
> **No new Q-marker introduced** by this prompt. PR-X4's slot at W4-W5 is
> fixed in the master schedule; the savant decisions Q-NEW-1 (GridLake)
> and Q-NEW-2 (X14′) determine whether GridLake + the contract have
> stabilised by W4. If either lands later than W3, PR-X4 starts later.

## Why this exists — the 4×4 cascade IS the cognitive spacetime kernel

PR-X4 promotes the splat3d pipeline from "refactor a bespoke binner" to
"typed multi-resolution cognitive evolution operator." **Gaussian
splatting and the cognitive cascade are mathematically identical at the
substrate level**; PR-X4 makes the identity literal in Rust.

**Verbatim, from `pr-x4-design.md`:36-48** (master commit `c8f4af68`):

| Aspect | 3D Gaussian Splatting (graphics) | Cognitive Shader (PR-X4 reframing) |
|---|---|---|
| Primitive | 3D Gaussian with position + anisotropic covariance + SH color + opacity | Cognitive cell with spacetime position + SPO covariance + typed state + NARS confidence |
| L1 tile | 16×16 screen-pixel bin | 64×64 cognitive cell block (PR-X3 default) |
| L2 cascade | View-frustum tile clustering | Regional resonance super-block (4×4 of L1) |
| L3 cascade | Scene-level LOD bucket | Scene aggregation super-block (16×16 of L2) |
| L4 cascade | Framebuffer / final composite | Experience memory super-block (4×4 of L3) |
| Splat projection | 3D → 2D screen ellipse (Jacobian of view) | State → cell footprint (Jacobian of inquiry) |
| Tile binning | Each splat into all tiles its 3σ ellipse covers | Each activation into all L1 blocks its SPO footprint covers |
| Sort order | Front-to-back by depth | Most-confident-first by NARS truth-projection |
| Composition | Alpha: `C_out = α·C_splat + (1-α)·C_accum` | NARS revision: `T_out = revise(T_splat, T_accum)` |
| SH | View-direction-dependent color (deg-3 = 16 coefs/ch) | Inquiry-direction-dependent state (vocab × thinking_style) |
| Anisotropic covariance | 3×3 SPD via 6-param Cholesky | SPO superposition shape |
| Saturation early-exit | Stop compositing at alpha≈0 | Stop revision at confidence floor |

The existing splat3d pipeline (project → bin → sort → rasterize),
refactored onto BlockedGrid with `crate::simd::*` inside its closures,
**IS** the cognitive shader's spacetime evolution kernel. We don't need
a separate PR-X8 SpacetimeStream — the Gaussian splat cascade IS the
spacetime stream when L4 is interpreted as the time axis and the
cascade runs every tick.

## The (4×4)×(4×4)×(4×4)×(4×4) tier scheme — verbatim load-bearing claim

**Verbatim, from `pr-x4-design.md`:57-72** (master commit `c8f4af68`):

```
L4 (16384²)  framebuffer / experience memory   ←  4×4 super-grid of L3
L3 (4096²)   scene aggregation                  ←  16×16 super-grid of L2
L2 (256²)    regional resonance                  ←  4×4 super-grid of L1
L1 (64²)     per-cell context                    ←  4×4 covariance footprint per splat
```

> Each tier's "(4×4)" is the local Gaussian covariance support — the
> 16-sample anisotropic kernel that defines the splat's footprint at
> that scale. A splat at L1 has a 4×4 cell footprint with anisotropic
> covariance; at L2, the SAME splat (downsampled) has a 4×4 super-block
> footprint = 16×16 cells of its L1 footprint; etc. Cascading the
> covariance through 4 tiers gives the L1→L4 cognitive context window.

> **Area-wise**: each tier covers 16× more area than the previous
> (uniform area-branching), giving L4/L1 area ratio = 16⁴ = 65,536.
> This matches the cell-count ratio 16384²/64² = 65,536 exactly.

> **Per-dim branching is non-uniform (4 / 16 / 4)** because the pyramid
> is not isotropic in tier-stride — the L2→L3 transition has a wider
> gather to span the scene-aggregation scale. Splat composition handles
> this naturally: at each tier, sort and composite the splats whose
> 3σ ellipse intersects the current tile.

The 4×4 framing is load-bearing: it is the structural reason cognitive
cells and graphics splats share the same kernel. Departing from 4×4 at
any tier breaks the area-uniform 16× branching and decouples the two
interpretations.

## What PR-X4 is, mechanically

### Refactor target: `src/hpc/splat3d/tile.rs` → `splat3d_v2/tile.rs`

**Verbatim shape, from `pr-x4-design.md`:108-118**:

```rust
/// One (tile, gaussian) binding emitted during binning. Same shape as
/// the existing TileInstance, but `tile_id` is now (tier, block_row,
/// block_col) for multi-resolution cascade.
#[repr(C, align(16))]
pub struct TileInstance {
    pub tier: u8,          // 1 = L1, 2 = L2, 3 = L3, 4 = L4
    pub _pad: [u8; 3],     // to keep block_row 4-byte aligned
    pub block_row: u16,
    pub block_col: u16,
    pub gaussian_id: u32,
    pub confidence: f32,   // replaces depth — sort key (highest-first for NARS revision)
}
```

The bespoke `Vec<TileInstance> + Vec<u32> prefix` hand-rolled CSR
becomes `BlockedGrid<SplatBinList, 1, 1>` from PR-X3 — same shape,
typed substrate.

### Cascade addressing — `splat4d::cascade`

```rust
pub struct CascadeAddr(u16);  // 4 nibbles, one per tier level

impl CascadeAddr {
    pub fn level(&self, l: u8) -> u8 { (self.0 >> (l * 4) & 0xF) as u8 }
    pub fn parent(&self) -> CascadeAddr { CascadeAddr(self.0 & !0xF000) }
    pub fn children(&self) -> [CascadeAddr; 16] { ... }
    pub fn from_position(p: Vec3, bbox: AABB, level: u8) -> CascadeAddr {
        // Hilbert-3D encode — PR-X10 A12b deliverable
        CascadeAddr(linalg::hilbert::hilbert3d_encode(p_quantised, level) as u16)
    }
}
```

The L4 cascade addressing is **the entire point of PR-X10 A12b**.
See § "Carry-over from PR-X10 A12b" below for the P0-4 bug status.

## The five splat gaps (from `pr-arithmetic-inventory.md` § L4-L8)

PR-X4's W4-W5 slot owns four of these five gaps; the fifth (CTU mode
encoder) belongs to PR-X9 (basin-codebook):

### G1 — Hilbert-3D encode/decode at level=1..4

**Status**: PR-X10 A12b ships `linalg::hilbert` with **a known L4 bug**
(see § "Carry-over"). Until A12b is fixed, PR-X4 cannot ship L4
cascade addressing. L1-L3 addressing is unblocked (round-trip tests
pass exhaustively per `pp13-brutally-honest-tester-verdict.md`:35).

**Precision class**: EXACT (integer-only).

**LoC budget**: ~64 ops/address; ~200 LoC in `linalg::hilbert`.

### G2 — INT4×N packed dot product

For thinking-style (32-dim INT4) and qualia (16-dim INT4) cell
signatures. PR-X4's `SplatCell<D>` consumes this.

**Hardware paths** (verbatim, from `pr-arithmetic-inventory.md`:184-189):
- AVX-512 VNNI `vpdpbusd`: i8 × u8 → i32 accumulator, 64 ops per
  instruction → **2 instructions for 32-dim INT4**
- ARM NEON `sdot`: i8.4 dot u8.4 → i32 → **8 instructions for 32-dim**
- AMX BF16 tile op handles INT8 not INT4 directly — software unpacking
- Scalar fallback: 32-way unrolled

**Precision class**: EXACT (integer dot).

**LoC budget**: ~3 backends × ~50 LoC + parity test ≈ ~200 LoC.

### G3 — NARS truth-revision kernel

Replaces alpha-compositing in W7's PR-X4 closure swap. Same math as
`hpc/nars.rs::revise_belief`, surfaced as a SIMD-staged inner loop:

```
revise(T1, T2) = (
  freq:  (f1·c1 + f2·c2) / (c1 + c2),
  conf:  (c1 + c2) / (c1 + c2 + k),     where k = 1 by NARS convention
)
```

**Precision class**: TBD — depends on whether `fast_exp_x16`'s 3%
rel-err propagates through `(c1 + c2) / (c1 + c2 + k)`. **See G5
precision audit below.**

**LoC budget**: ~80 LoC scalar reference + ~150 LoC SIMD path + parity
test ≈ ~250 LoC.

### G4 — fast_exp_x16 precision audit

The existing `fast_exp_x16` ships **3% relative error** — graphics-
suitable for alpha-compositing (Inria SSIM ≥ 0.97) but suspect for
NARS confidence floor where `(c1+c2)/(c1+c2+k)` near boundaries can
diverge under multiplicative noise.

**Audit deliverable**: a sweep of 10K randomly-seeded NARS revisions
comparing `fast_exp_x16`-based conf path against an f64-reference, with
the worst-case relative-error and worst-case confidence-divergence
reported. PASS bar: `worst_conf_diff < 1e-3`. FAIL bar: spec a
`precise_exp_x16` extension at PR-X10 A6 follow-up.

**LoC budget**: pure audit, ~100 LoC test code; **no new kernel** ships
unless the audit fails.

### G5 — CTU mode encoder (x265-style skip/merge/delta/escape)

Belongs to **PR-X9 (basin-codebook lazy storage)** at W6-W7, not PR-X4.
Listed here for the dependency chain: PR-X9 consumes PR-X4's cascade
addresses to skip-encode unchanged basins. PR-X4 must not implement
CTU modes itself.

## Carry-over from PR-X10 A12b — the L4 Hilbert-3D P0-4 bug

**Verbatim, from `pp13-brutally-honest-tester-verdict.md`:32-37**:

> ### P0-4. Hilbert-3D encode is broken at level=4 (the splat4d L4
> cascade level)
>
> - **File**: `src/hpc/linalg/hilbert.rs:71-116,232-246`
> - **Symptom**: `hilbert3d_encode([15,15,15], 4)` returns **2925**,
>   expected **4095** (max index). At level 4 the map is not a
>   bijection onto `[0, 4096)`.
> - **Root cause**: `NEXT_STATE` and `H_TO_XYZ` tables are not verified
>   to be mutually consistent for the full 4-level recursion. The doc
>   claims (line 22-23) "verified to satisfy
>   `decode(encode(pos, level), level) == pos` for all `pos` and
>   `level`" — but the level-4 test was either never run or its
>   failure ignored. Round-trip at level=2 and level=3 pass
>   (exhaustive), so the table is partially correct; the curve
>   orientation transitions diverge at depth 4.
> - **3am impact**: any splat4d cascade addressing at L4 produces
>   collisions and out-of-range indices. The whole point of A12b being
>   the "splat4d cascade addressing" worker is the L4 level.
> - **Patch direction**: re-derive `NEXT_STATE` from Hamilton 2006
>   Table 2 (cited in the file) symbolically and add
>   `round_trip_level4_exhaustive` test (4096 cells × 4 µs ≈ 16 ms,
>   cheap). Until then, **do not export `hilbert3d_encode/decode` to
>   consumers**.

**PR-X4's readiness gate**: PR-X10 A12b must ship a fix and an
exhaustive `round_trip_level4` test before PR-X4's `CascadeAddr` worker
can spawn. The L1-L3 round-trip tests already pass exhaustively, so
the level-4 fix is the only blocker. **PR-X4 must NOT re-introduce its
own Hilbert-3D encode** — wait for the A12b fix, then consume
`linalg::hilbert::hilbert3d_encode`.

## Schedule slot — W4-W5 (5 workers), no new Q-marker

The 8-week schedule from `hhtl-substrate-execution-prompt.md`:76-91
already places PR-X4 at W4-W5 alongside PR-X12 (codec, 8 workers):

| Week | Sprints | Workers |
|---|---|---|
| W1-W2 | PR-X10 (linalg-core foundation) | 12 |
| **W2.5/W3** | **PR-X1 + PR-X2 (GridLake) + PR-X14′ (contract)** ← Q-NEW-1/Q-NEW-2 pending | 4-10 depending on cell |
| W3 | PR-X11 (jc consolidation) + PR-X13 (OGIT bridge) | 6 + 4 |
| **W4-W5** | **PR-X12 (codec, 8) + PR-X4 (splat cascade, 5)** | **13** |
| W6-W7 | PR-X9 (basin-codebook) | 6 |
| W8 | Integration + canary | 3 |

PR-X4's spawn at W4 depends on:
- ✅ PR-X10 A6 (`linalg::distance`) — landed by W2
- ⚠️ PR-X10 A8 (`linalg::sh` deg 4-7) — needed for inquiry-direction
  SH evaluation; lands by W2
- ⚠️ PR-X10 A12b (`linalg::hilbert` L4 fix) — **GATE** per the P0-4
  carry-over above
- ✅ PR-X1 + PR-X2 (GridLake) — needed for splat-bytes via
  `MultiLaneColumn`; lands W2.5 (path α) or W3 (path β) per Q-NEW-2
- ✅ PR-X14′ (contract) — needed for `splat4d::cascade` to consume
  Lance datasets via `gridlake-bridge`; lands W2.5/W3 per Q-NEW-2
- ✅ PR-X11 (jc consolidation) — provides `Spd3::sandwich/sqrt` for
  Pillar-8 temporal sandwich; lands by W3

If GridLake + X14′ slip past W3 (e.g., Cell A-β extension), PR-X4
starts late by the same margin. **No schedule extension owed by
PR-X4 itself.**

## Worker spawn shape (5 workers)

```
A1: TileInstance v2 + BlockedGrid<SplatBinList, 1, 1> refactor (chain dep)
    │
    ├──→ A2: CascadeAddr + from_position/to_position_center wrappers
    │        (depends on A12b L4 Hilbert fix — gate)
    │
    ├──→ A3: G1 SH projection deg-3 (4×4 anisotropic kernel)
    │        in inquiry-direction (vocab × thinking_style)
    │
    ├──→ A4: G2 INT4×32 packed dot (3 backends + parity test)
    │
    └──→ A5: G3 NARS truth-revision kernel + G4 fast_exp_x16
              precision audit (combined; A5 verdict determines if
              precise_exp_x16 follow-up needed)
```

A1 is the only chain dep. A2-A5 are parallel after A1 lands.
A2 has an additional gate dep on PR-X10 A12b's L4 fix landing on
master (not just PR-X10 W2 completion).

## Done criteria

The sprint is done when ALL of the following hold:

1. **Refactor complete**: `src/hpc/splat3d_v2/tile.rs` exists with the
   `TileInstance { tier, block_row, block_col, gaussian_id, confidence }`
   shape on `BlockedGrid<SplatBinList, 1, 1>`. The original
   `splat3d/tile.rs` ships side-by-side or migrates in-place per the
   pr-x4-design § Q1 decision.

2. **Cascade addressing**: `splat4d::cascade::CascadeAddr` ships with
   - `from_position(p, bbox, level)` calling
     `linalg::hilbert::hilbert3d_encode` for level=1..4
   - `to_position_center(addr, bbox)` inverse via
     `hilbert::hilbert3d_decode`
   - Round-trip exhaustive test at level=4 (4096 cells × 3 axes)
     passing — gated on A12b fix
   - L1-L3 round-trip tests passing (already pass under current
     A12b, just verify under the splat4d call sites)

3. **The four splat gaps** (G1 + G2 + G3 + G4):
   - G1 deg-3 SH inquiry-direction evaluation: parity test against
     splat3d::sh::sh_eval_deg3 (the original raster path) — bit-exact
   - G2 INT4×32 dot: 3 backends (AVX-512 VNNI / NEON sdot / scalar)
     with cross-backend parity test on 10K random pairs
   - G3 NARS truth-revision kernel: SIMD-staged inner loop against
     scalar `nars::revise_belief` reference — within ULP ≤ 4 of scalar
     reference on 10K randomly-seeded revisions (the canary plan's
     correctness gate #1)
   - G4 fast_exp_x16 audit: `worst_conf_diff < 1e-3` on 10K random
     revisions OR a `precise_exp_x16` follow-up PR opened against
     PR-X10 A6 with the audit numbers as motivation

4. **Tier-cascade composition**:
   - L5 Gaussian-mixture moment-match (`compose_cascade`) implemented
     and tested against a scalar reference on a 16-child fixture per
     tier (L3 → L4 super-block)
   - L6 Pillar-8 temporal sandwich integrated (calls `Spd3::sandwich`
     from PR-X11's jc consolidation)
   - Area-uniform 16× branching verified by a smoke test:
     `L1_cell_count × 16⁴ == L4_cell_count` (= 65,536)

5. **Composition swap**:
   - Alpha-compositing path (current splat3d raster) ships unchanged
     as the graphics interpretation
   - NARS-revision path (cognitive shader interpretation) ships behind
     a feature flag `splat4d-nars-compose` (default off until W7's
     PR-X4 closure swap)
   - Saturation early-exit threshold (`T_SATURATION_EPS`) parameterised
     so the same outer loop serves both interpretations

6. **No regression**: existing splat3d test suite (the 2370-test
   union currently green on master) still passes after the refactor.
   `cargo clippy -- -D warnings` clean.

## Forbidden constraints

Five invariants the sprint MUST NOT violate:

1. **No bespoke Hilbert-3D implementation in PR-X4**. The L4 fix lives
   in PR-X10 A12b. PR-X4 consumes `linalg::hilbert::hilbert3d_{en,de}code`
   and waits for the fix. If A12b's fix slips past W3, PR-X4's A2
   worker stubs the L4 path (returns `Err(NotReadyL4)`) and ships L1-L3
   addressing only.

2. **No `crate::simd::*` extension from inside PR-X4**. Any new SIMD
   primitive (e.g., a missing lane width for G2 INT4×32) must be
   proposed against `vertical-simd-consumer-contract.md` and land in
   ndarray's `src/simd_*.rs` before PR-X4 consumes it. PR-X4 must not
   reach for raw `std::arch::*` intrinsics.

3. **No write to lance-graph upstream**. PR-X4 lives entirely in
   ndarray (`src/hpc/splat3d_v2/`, `src/hpc/splat4d/`). It consumes
   `lance-graph-contract::column::MultiLaneColumn` via the bridge
   crate from PR-X14′ but does not modify lance-graph or
   lance-graph-contract.

4. **No 4×4 tier-stride departure**. The (4×4)×(4×4)×(4×4)×(4×4)
   scheme is the structural identity between splat and cognitive
   shader. Any tier-stride change (e.g., 4×4 → 8×8 at L2) requires a
   separate design doc + a new PR-X4-successor, not an inline scope
   creep.

5. **No NARS-revision activation by default**. The composition swap
   from alpha-compositing to NARS revision is gated behind the
   `splat4d-nars-compose` feature flag (off by default). W7's PR-X4
   closure-swap sprint is the canonical activation point; PR-X4
   itself ships both paths but defaults to alpha-compositing for the
   graphics interpretation.

## Carry-overs back into other sprints

- **PR-X10 A12b**: must ship `round_trip_level4_exhaustive` test and
  the `NEXT_STATE` re-derivation from Hamilton 2006 Table 2 before
  PR-X4 spawns. **Gate.**
- **PR-X11 (jc consolidation)**: PR-X4 consumes `Spd3::sandwich`,
  `Spd3::sqrt`, `Spd3::from_rows` for the L5/L6 moment-match and
  temporal-sandwich paths. These are PR-X11's deliverable.
- **PR-X9 (basin-codebook, W6-W7)**: consumes PR-X4's `CascadeAddr` to
  skip-encode unchanged basins. PR-X4 must ship `CascadeAddr` and its
  `parent()/children()` ops by end-W5 to unblock PR-X9 W6 spawn.
- **PR-X7 (typed cell-DSL, queued, post-Phase-2)**: consumes PR-X4's
  `SplatCell<D>` as the typed-cell signature bridge. PR-X4 must keep
  `SplatCell<D>` API forward-compatible (no breaking changes after
  W5).
- **W7 closure swap sprint**: flips the `splat4d-nars-compose` feature
  flag from off to on after the G4 fast_exp_x16 audit verdict lands.
  No-op for PR-X4 itself; recorded here for downstream visibility.

## Cross-references

- `.claude/knowledge/pr-x4-design.md` — master design doc for the 4×4
  splat cascade (merged PR #162, `c8f4af68`)
- `.claude/knowledge/pr-x3-cognitive-grid-design.md` — BlockedGrid
  substrate PR-X4 refactors onto (shipped PR #158)
- `.claude/knowledge/pr-x9-design.md` — basin-codebook lazy storage
  (downstream consumer)
- `.claude/knowledge/pr-z1-ogit-cognitive-bootstrap.md` — OGIT
  cognitive namespace bootstrap (W3, dep)
- `.claude/knowledge/pr-x10-linalg-core-design.md` — `linalg::sh` +
  `linalg::hilbert` + `linalg::distance` deliverables PR-X4 consumes
- `.claude/knowledge/pp13-brutally-honest-tester-verdict.md` § P0-4 —
  the L4 Hilbert-3D bug (gate for A2 worker)
- `.claude/knowledge/pr-arithmetic-inventory.md` § L4-L8 — splat gap
  taxonomy
- `.claude/knowledge/hhtl-gridlake-pre-sprint-prompt.md` — substrate
  carrier PR-X4 consumes (master `ade8edb2`)
- `.claude/knowledge/hhtl-pr-x14-substrate-contract-prompt.md` —
  contract crate PR-X4 consumes (master `56b26716`)
- `.claude/knowledge/hhtl-substrate-execution-prompt.md` — W1-W8
  master schedule (PR-X4 at W4-W5)
- `.claude/knowledge/hhtl-savant-preflight-q-new-1-q-new-2.md` —
  open PR #165, plan-review savant brief for Q-NEW-1 / Q-NEW-2

## TL;DR

PR-X4 promotes splat3d from "bespoke 16×16 tile binner" to "typed
multi-resolution cognitive evolution operator" with the
(4×4)×(4×4)×(4×4)×(4×4) tier scheme as its load-bearing structural
identity. Slots at W4-W5 (5 workers). Consumes GridLake +
`lance-graph-contract::column` + PR-X10 A6/A8/A12b + PR-X11 jc Spd3.
Gates on PR-X10 A12b's L4 Hilbert-3D P0-4 fix (`hilbert3d_encode([15,15,15], 4)
→ 2925, expected 4095`). Ships four splat gaps (G1 deg-3 SH inquiry-
direction, G2 INT4×32 packed dot, G3 NARS truth-revision kernel,
G4 fast_exp_x16 precision audit). Alpha-compositing stays default;
NARS-revision composition gated behind `splat4d-nars-compose` feature
flag until W7 closure swap. CTU-mode encoder is PR-X9's deliverable,
not PR-X4's.
