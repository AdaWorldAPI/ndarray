# PR-X11 — jc consolidation → `ndarray::hpc::pillar::*`

> READ BY: every agent that touches Pillar probes or jc math
> (savant-architect, jc-architect, cascade-architect, truth-architect,
> sentinel-qa, vector-synthesis, product-engineer).
>
> Status: design v1 — drafted 2026-05-18.
>
> Parent: `pr-master-consolidation.md` — the strategic frame.
> Foundation: `pr-x10-linalg-core-design.md` — provides the canonical Spd2/Spd3
> that pillar probes consume.

## Why PR-X11 exists

`lance-graph/crates/jc/` ships pillar certification probes for the SPD-cascade math.
Three of them duplicate Spd2/Spd3 internally:
- `jc/src/ewa_sandwich.rs`     (Pillar-6, 2D EWA sandwich)
- `jc/src/ewa_sandwich_3d.rs`  (Pillar-7, 3D EWA sandwich — duplicate of splat3d's Spd3)
- `jc/src/koestenberger.rs`    (third Spd3 copy)

Plus Wasserstein-1 inline in `jc/src/pflug.rs` (Pillar-10), and pending Pillar-8
(temporal sandwich), Pillar-9 (Cov16384 high-D CLT), Pillar-11 (signature transform)
that don't exist yet because the math primitives haven't been factored.

The jc original design said "zero-dep on ndarray for self-certification."
That was sound when jc was 2 files. With 4-way Spd2/Spd3 duplication, 3-way
drift risk on log_spd / sandwich / pow, and 3 more pillars queued behind
missing primitives, the consolidation cost exceeds the certification benefit.

**Invariant 12** (from master consolidation): *Certification is about
determinism and inspectability, not repo separation.* PR-X11 moves the math
into `ndarray::hpc::pillar::*` and re-affirms certification via SEED-anchored
probes + git-tracked impls + committed bench results.

## What PR-X11 ships

```
src/hpc/pillar/
├── mod.rs              — pub surface; submodule decls + re-exports
├── ewa_sandwich_2d.rs  — Pillar-6: 2D EWA sandwich + prove() probe
├── ewa_sandwich_3d.rs  — Pillar-7: 3D EWA sandwich + prove() probe
├── koestenberger.rs    — Pillar-7.5: Koestenberger PSD path
├── temporal_sandwich.rs— Pillar-8: temporal drift sandwich + prove()
├── cov_high_d.rs       — Pillar-9: Cov16384 Düker-Zoubouloglou CLT
├── pflug.rs            — Pillar-10: Pflug-Pichler nested distance
├── signature.rs        — Pillar-11: Hambly-Lyons signature transform
└── prove_runner.rs     — shared probe harness (splitmix64 RNG, SEED constants)
```

Each `*.rs` exports:
- The pillar's typed wrapper (`PillarSeven`, `PillarTen`, ...) — concrete carrier per invariant 8
- The `prove()` certification probe with documented SEED + PASS criteria
- The math kernel as `pub` functions consuming `ndarray::hpc::linalg::Spd{2,3,N}` from PR-X10

## Migration path for jc consumers

After PR-X10 ships `linalg::Spd3` (with Smith-1961 closed-form), PR-X11 lands
`pillar::ewa_sandwich_3d` which consumes it. The OLD `jc::ewa_sandwich_3d`
gets a 1-cycle deprecation:

```rust
// In jc/src/ewa_sandwich_3d.rs after PR-X11:
#[deprecated(
    since = "0.X",
    note = "moved to ndarray::hpc::pillar::ewa_sandwich_3d; this stub forwards calls",
)]
pub use ndarray::hpc::pillar::ewa_sandwich_3d::*;
```

For 1 release cycle, jc's old paths work via re-export. Cycle N+2 removes
the re-export files entirely. Existing downstream consumers (`AdaWorldAPI/spear`,
`AdaWorldAPI/q2`, `AdaWorldAPI/woa-rs`) get a deprecation warning + 1 cycle
to migrate imports. No breaking change in cycle N.

## Worker decomposition (6 workers + coord + savants)

Per-pillar one worker. Pillars are independent (no cross-pillar dependencies
within PR-X11), so all 6 spawn in PARALLEL after the coordinator scaffolds
`pillar/mod.rs`.

| # | Worker | File | LoC | Depends on |
|---|---|---|---|---|
| 0 | coord | `pillar/mod.rs` (scaffold) | 30 | PR-X10 `linalg::*` |
| 1 | **B1** | `pillar/ewa_sandwich_2d.rs` (Pillar-6) | ~250 | `linalg::Spd2` |
| 2 | **B2** | `pillar/ewa_sandwich_3d.rs` (Pillar-7) | ~280 | `linalg::Spd3` |
| 3 | **B3** | `pillar/koestenberger.rs` (Pillar-7.5) | ~200 | `linalg::Spd3` |
| 4 | **B4** | `pillar/temporal_sandwich.rs` (Pillar-8) | ~300 | `linalg::Spd3` + σ_temporal literature |
| 5 | **B5** | `pillar/cov_high_d.rs` (Pillar-9) | ~350 | `linalg::eig_sym_n` |
| 6 | **B6** | `pillar/pflug.rs` (Pillar-10) | ~400 | `linalg::wasserstein` (also PR-X10) |
| 7 | **B7** | `pillar/signature.rs` (Pillar-11) | ~350 | `linalg::linalg` (Lie group ops) |
| 8 | **B8** | `pillar/prove_runner.rs` (shared harness) | ~150 | none — pure infra |

Workers B1–B8 spawn in parallel after coord lands `mod.rs`. **6 in the parallel
fan-out + 2 (coord + harness) sequential = 8-worker shape**.

Phase 2 Protocol A: preflight Rust skeleton authored by coord, reviewed by
6 specialist savants (data-flow, layering, distance-typing, SAFETY-claim,
naming-collision, test-coverage). All pillars get reviewed in one savant fan-out.

## Pillar PASS gates (carry-over from jc, refined)

Each `prove()` probe has a deterministic SEED and explicit PASS criteria:

| Pillar | SEED | Paths × hops | PASS criteria |
|---|---|---|---|
| Pillar-6 (2D EWA) | `0x_DA_5A_DC_5A_DD` | 1000 × 10 | PSD rate ≥ 0.999, log-norm Frobenius KS Thm 1 |
| Pillar-7 (3D EWA) | `0x_EDA_5A_DC_5A_DD` | 1000 × 10 | PSD rate ≥ 0.999, same |
| Pillar-7.5 (Koestenberger) | `0x_KE_5A_DC_5A_DD` | 1000 × 10 | path-1 vs path-2 max abs error ≤ 1e-5 |
| Pillar-8 (temporal) | `0x_E0_DA_5A_DC_5A_DD` | 1000 × 30 × 3 bands | PSD rate ≥ 0.999 across cardiac / respiratory / micro |
| Pillar-9 (Cov16384) | `0x_C0_DA_DA_5A_DC` | 100 × 50 | Düker-Zoubouloglou CLT rate ≥ 0.95 |
| Pillar-10 (Pflug) | `0x_F1_5A_DC_5A_DD` | 1000 × 5 | nested-distance ≤ tight Pflug-Pichler bound |
| Pillar-11 (signature) | `0x_516_DC_5A_DD` | 1000 × Lévy paths | Hambly-Lyons sigker convergence |

All probes use `splat_runner`'s shared splitmix64 RNG (`prove_runner::seed_rng(seed)`).
All probes commit `RESULTS.md` lines per run (hardware + commit SHA + PASS/FAIL +
metric values). Auditors can re-run against an independent reference
(numpy / scipy / R) to cross-certify.

## Architectural invariants (carry-over)

Invariants 1-11 from PR-X3 / PR-X4 / PR-X10. Plus:

**12. Certification is about determinism + inspectability, not repo separation.**
(Replaces old jc zero-dep rule.)

**13. Every pillar probe's prove() is `cargo run --release -p ndarray --example prove_pillar_N`
and is part of the CI matrix.** No `#[ignore]` on probes. Slow probes (Pillar-8
with 90k samples) marked `#[cfg(feature = "slow-tests")]` but still runnable on demand.

## Tests required

Per pillar:
- `prove()` runs to PASS within budget
- Math kernel matches scalar reference within stated epsilon (typically 1e-5 for f32, 1e-12 for f64 paths)
- Round-trip identity: e.g., `Σ → sqrt → squared ≈ Σ` for Pillar-6/7
- Boundary cases: identity matrix, diagonal, near-degenerate

Cross-pillar:
- Pillar-7's Spd3 IS splat3d's Spd3 IS `linalg::Spd3` (single type) — verify via type-id check
- Pillar-8's temporal sandwich matches Pillar-7's spatial sandwich on identity step-Σ
- Pillar-10's nested distance reduces to Wasserstein-1 on degenerate (single-time-step) case

## Out of scope

- jc's actual deprecation removal (1-cycle after PR-X11) — separate housekeeping PR
- New pillars beyond 6-11 — pillar 1-5 stay as design ideas, not coded yet
- AriGraph / NARS-engine orchestration in lance-graph — those stay in lance-graph,
  consume `ndarray::hpc::pillar` for math verification

## Verification commands

```bash
cargo check -p ndarray --no-default-features --features std,linalg,pillar
cargo test  -p ndarray --lib --no-default-features --features std,linalg,pillar hpc::pillar
cargo run --release -p ndarray --features pillar --example prove_pillar_6  # PASS
cargo run --release -p ndarray --features pillar --example prove_pillar_7  # PASS
cargo run --release -p ndarray --features pillar --example prove_pillar_8  # PASS (cardiac+respiratory+micro)
cargo run --release -p ndarray --features pillar --example prove_pillar_9  # PASS
cargo run --release -p ndarray --features pillar --example prove_pillar_10 # PASS
cargo run --release -p ndarray --features pillar --example prove_pillar_11 # PASS
cargo fmt --all -- --check
cargo clippy -p ndarray --features pillar -- -D warnings
```

## Open questions (joint savant ruling)

1. **Pillar-8 σ_temporal literature values** — cardiac (~6 Hz, ~5 mm), respiratory
   (~0.3 Hz, ~20 mm), micro (~120 Hz, ~0.1 mm) — these are the splat4d cascade
   prompt's estimates. Need echocardiography + respiratory-physiology numbers
   before sprint kickoff. Auto-resolve: use the prompt's defaults, mark as
   "TODO calibrate against literature" with a tracking issue.

2. **Pillar-9 N choice** — 16384 (matches BindSpace) vs 4096 (matches CAM codebook) vs
   variable. Lean: **16384** (BindSpace alignment); 4096 case is a degenerate special
   of the 16384 probe.

3. **Pillar-11 signature transform algorithm** — depth-3 (cheap, ~30 ops) vs
   depth-5 (more discriminative, ~300 ops). Lean: **depth-3 for v1**, depth-5
   as opt-in via const generic `<const D: usize>`.

4. **jc deprecation cycle** — 0 / 1 / 2. Lean: **1 cycle** (per master plan).

5. **Pillar 1-5 future work** — defer entirely or pre-stage interfaces?
   Lean: **defer**, no stubs.

6. **Pillar parallelism: all 6 spawn after coord, or pillar 8/9 hold for σ_temporal/N decisions?**
   Lean: **all 6 spawn** with documented defaults (auto-resolved as above);
   calibration is a follow-on.

## Done criteria

- All 6 pillars implemented in `src/hpc/pillar/*.rs`
- All `prove_pillar_N` examples PASS
- jc's 4 math files marked `#[deprecated]` with re-export to new location
- Cross-pillar parity tests green
- Codex P0 audit: 0 P0
- P2 savant: SHIP
- `RESULTS.md` committed with bench numbers per pillar on Zen4 + Sapphire Rapids
