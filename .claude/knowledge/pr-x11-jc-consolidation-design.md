# PR-X11 — jc consolidation: pillar probes → `ndarray::hpc::pillar::*`

> READ BY: savant-architect, jc-architect, l3-strategist, cascade-architect,
> truth-architect, sentinel-qa.
>
> Status: design v1 — drafted 2026-05-18 in the master-consolidation arc.
>
> **Depends on**: PR-X10 (linalg-core) landed.
> **Subsumes**: 3-way Spd2/Spd3 duplication in `lance-graph/crates/jc/`.

## Why

Today `lance-graph/crates/jc/` has **four** copies of nearly the same math:
- `ewa_sandwich.rs`     — Spd2 + Pillar-6 probe (2D EWA-sandwich)
- `ewa_sandwich_3d.rs`  — Spd3 + Pillar-7 probe (3D EWA-sandwich, twin of splat3d's Spd3)
- `koestenberger.rs`    — Spd3 again (different probe, same math)
- `pflug.rs`            — Wasserstein-1 inline (Pillar-10)

Plus three pillar probes are *missing* primitives entirely:
- Pillar-8 (temporal sandwich) — designed but not built
- Pillar-9 (Cov16384, Düker-Zoubouloglou CLT in ℝ^16384) — hand-rolled f64 scalar
- Pillar-11 (signature transform, Hambly-Lyons) — designed but not built

**Old invariant** (jc zero-dep on ndarray) blocked consolidation. **PR-X10 ships
`ndarray::hpc::linalg::*` with Spd2/Spd3 as the canonical surface; new
invariant 12** (per `pr-master-consolidation.md`) replaces zero-dep with
"certification = determinism + inspectability".

PR-X11 moves the math + probes into `ndarray::hpc::pillar::*`. jc becomes a
thin probe-runner.

## Module layout — `crate::hpc::pillar::*`

```
src/hpc/pillar/
├── mod.rs                    — pub surface + feature gate
├── ewa_sandwich_2d.rs        — A1: Pillar-6 (Spd2 EWA, 2D)
├── ewa_sandwich_3d.rs        — A2: Pillar-7 (Spd3 EWA, 3D — twin of splat3d::Spd3)
├── temporal_sandwich.rs      — A3: Pillar-8 (Σ_{t+1} = M·Σ_t·Mᵀ with M = sqrt(σ_temporal))
├── cov_high_d.rs             — A4: Pillar-9 (Cov16384 / Cov<N> carrier for high-D CLT)
├── pflug_nested.rs           — A5: Pillar-10 (nested-distance probe; consumes linalg::wasserstein)
├── signature.rs              — A6: Pillar-11 (signature transform, Hambly-Lyons)
├── koestenberger.rs          — A2 also (consolidated under Pillar-7 since same math)
└── tests/                    — parity gates vs jc's existing probe output (bit-exact required)
```

Plus `linalg::wasserstein` (added to PR-X10 scope or as PR-X10.1 follow-on):
- Sinkhorn-Knopp algorithm (entropic regularization, O(N²/ε) per iter)
- Hungarian algorithm (exact assignment, O(N³))
- Both consumable by Pillar-10 AND cognitive substrate's optimal-transport ops.

## API surface

Each pillar module ships:
1. **A canonical math primitive** consumed by ndarray + downstream
2. **A `prove_pillar_N()` probe function** that runs the certification with a documented SEED
3. **A `PASS_CRITERIA` const block** with thresholds (PSD rate, log-norm concentration, etc.)

Example (Pillar-7, the twin of splat3d's Spd3):

```rust
//! Pillar 7: Σ-Push-Forward as EWA-Sandwich on symmetric 3×3 SPD covariances.
//!
//! Math reference: Smith 1961, "Eigenvalues of a symmetric 3×3 matrix".
//!
//! # Probe SEED
//! `0x_EDA_5A_DC_5A_DD`
//!
//! # PASS criteria
//! - PSD preservation rate ≥ 0.999 across 1000 paths × 10 hops
//! - log-norm Frobenius concentration consistent with KS Theorem 1 form

pub use crate::hpc::linalg::Spd3;     // canonical type from linalg-core
pub use crate::hpc::linalg::sandwich; // canonical sandwich op

pub const PILLAR_7_SEED: u64 = 0x_EDA_5A_DC_5A_DD;
pub const PILLAR_7_PSD_THRESHOLD: f64 = 0.999;
pub const PILLAR_7_LOGNORM_CONCENTRATION_MAX: f64 = ...;

pub struct Pillar7Report {
    pub psd_rate: f64,
    pub lognorm_concentration: f64,
    pub n_paths: u32,
    pub n_hops: u32,
    pub passed: bool,
}

pub fn prove_pillar_7() -> Pillar7Report {
    let mut rng = splitmix64::seeded(PILLAR_7_SEED);
    let mut psd_ok = 0;
    let mut lognorm_acc = ...;
    for _path in 0..1000 {
        let mut sigma = Spd3::I;
        for _hop in 0..10 {
            let step = random_contractive_spd3(&mut rng, sigma_step_frobenius=0.2);
            sigma = sandwich(&step.sqrt(), &sigma);
            if sigma.is_spd(1e-7) { psd_ok += 1; }
            lognorm_acc += sigma.log_spd().frobenius_sq();
        }
    }
    let report = Pillar7Report { ... };
    report
}
```

## Worker decomposition — 6 workers (one per pillar)

| Worker | File | Scope | LoC |
|---|---|---|---|
| A1 | `ewa_sandwich_2d.rs` | Pillar-6 probe + `Spd2` re-export from `linalg::Spd2`; bit-exact parity with `jc::ewa_sandwich.prove()` | ~250 |
| A2 | `ewa_sandwich_3d.rs` + `koestenberger.rs` | Pillar-7 probe + both koestenberger variant + Spd3 re-export from `linalg::Spd3`; bit-exact parity with jc's two existing probes | ~350 |
| A3 | `temporal_sandwich.rs` | Pillar-8 probe (NEW — not in jc); σ_temporal stratified across cardiac/respiratory/micro; SEED `0x_E0_DA_5A_DC_5A_DD` | ~300 |
| A4 | `cov_high_d.rs` | Pillar-9 probe + `Cov<N>` const-generic carrier (sandwich, log, Frobenius for high-D); replaces jc's hand-rolled f64 scalar; promotes to `linalg::F32x16` SIMD | ~400 |
| A5 | `pflug_nested.rs` + `linalg::wasserstein` | Pillar-10 probe + Sinkhorn-Knopp + Hungarian primitives (added to linalg-core); bit-exact parity with `jc::pflug.prove()` | ~500 |
| A6 | `signature.rs` | Pillar-11 probe (NEW) — signature transform (Hambly-Lyons); supports rough-path lifting | ~350 |

**Parallelism**: all 6 workers spawn AFTER PR-X10 lands. No worker-to-worker dependencies (each pillar is independent; the linalg primitives they consume are all in PR-X10's foundation). **Maximum 6-way parallel sprint, ~1 week.**

## jc deprecation path

PR-X11 ships the consolidated `ndarray::hpc::pillar::*`. The companion deprecation commit on `lance-graph/crates/jc/`:

```rust
// lance-graph/crates/jc/src/ewa_sandwich.rs
#[deprecated(
    since = "0.X",
    note = "Math moved to ndarray::hpc::pillar::ewa_sandwich_2d. \
            jc retains the probe-runner pattern; the math primitives \
            live in ndarray's consolidated linalg-core (PR-X10) + \
            pillar (PR-X11) modules."
)]
pub use ndarray::hpc::pillar::ewa_sandwich_2d::*;
```

One cycle of `#[deprecated]`. Cycle N+1: remove the shim. Cycle N+2: jc becomes a thin orchestrator that imports `ndarray::hpc::pillar::*` directly.

**jc's `prove_pillarN.rs` examples stay** — they're the probe-runner UI; they just call `ndarray::hpc::pillar::prove_pillar_N()` instead of the inline math.

## Parity gates (the bit-exact requirement)

Each pillar's NEW probe in `ndarray::hpc::pillar::*` MUST produce **bit-exact** output (same SEED, same iteration count, same threshold values) as the corresponding OLD probe in `lance-graph/crates/jc/`. Two-test pattern:

```rust
#[test]
fn pillar_7_parity_with_jc() {
    let ndarray_report = ndarray::hpc::pillar::prove_pillar_7();
    let jc_report      = jc::ewa_sandwich_3d::prove();
    assert_eq!(ndarray_report.psd_rate, jc_report.psd_rate);
    assert_eq!(ndarray_report.lognorm_concentration, jc_report.lognorm_concentration);
    assert!(ndarray_report.passed);
}
```

If the ndarray probe diverges from jc's, the consolidation is wrong (likely numerical drift from a different f32/f64 reduction order). Worker MUST fix before merge.

## Verification commands

```bash
cargo check -p ndarray --features std,pillar
cargo test -p ndarray --features std,pillar hpc::pillar
cargo run --release -p ndarray --features pillar --example prove_pillar_7
cargo run --release -p ndarray --features pillar --example prove_pillar_8
cargo run --release -p ndarray --features pillar --example prove_pillar_10

# Parity gate (cross-crate):
cargo test -p jc --test parity_with_ndarray_pillar
```

All five must pass.

## Open questions (joint savant ruling)

1. **Wasserstein in `linalg` or `pillar`?** Lean: **linalg::wasserstein** (it's a primitive, not a probe; consumed by Pillar-10 AND cognitive optimal-transport). The PILLAR-10 PROBE lives in `pillar::pflug_nested`.

2. **Cov16384 const-generic Cov<N>?** Lean: **yes** — same math at any N. The `N=16384` case is Pillar-9's stress test; `N=64` is a more practical default for cognitive substrate use.

3. **Signature transform v1 scope: degree 4 or full?** Lean: **degree 4** in v1 (sufficient for Hambly-Lyons probe + rough-path use cases); higher degrees as PR-X11.1 follow-on.

4. **Deprecation cycle length: 1 or 2 cycles?** Lean: **1 cycle** of `#[deprecated]` shim; jc's downstream consumer surface is small (the AdaWorldAPI internal stack only).

5. **Do we backport Spd3 SIMD batching (sandwich_x16) to Pillar-7's probe?** Lean: **yes** — the probe runs 10k sandwich ops; AVX-512 batched = 10× speedup.

6. **Does jc-X1 (consolidation) ship in the same PR as jc's deprecation shim?** Lean: **yes** — one atomic PR ensures jc consumers never see broken state.

## Done criteria

- All 6 pillar workers complete with parity gates green
- jc deprecation shim in place; 1-cycle countdown begins
- 4 new probes (Pillar-8 temporal, Pillar-9 high-D, Pillar-11 signature, plus consolidated Pillar-6/7/10) all PASS their criteria
- Codex P0 audit passes (especially the SAFETY-claim gate on Cov<N> SIMD primitives)
- P2 savant SHIP verdict
