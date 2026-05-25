# A3: G1 deg-3 SH projection in inquiry-direction

Worker A3 of PR-X4 (W4-W5). Parallel after A1 lands. Ships the
unified deg-3 SH projection kernel that serves both the graphics
view-direction color path and the cognitive inquiry-direction state
path. Done-criteria line 326: **bit-exact** parity against
`splat3d::sh::sh_eval_deg3`.

## Math — deg-3 SH on the unit sphere

A degree-L real-SH expansion uses `(L+1)² = 16` coefficients per
channel at `L = 3`, indexed `(l, m)` with `0 ≤ l ≤ 3` and
`-l ≤ m ≤ l`. For unit direction `d = (sin θ cos φ, sin θ sin φ,
cos θ)`,

    f(d) = Σ_{l=0..3} Σ_{m=-l..l}  c_{l,m} · Y_l^m(d)

using the real orthonormal SH basis (Sloan 2008, Stupid SH Tricks
A2) — the same convention `linalg::sh` exports at deg 4-7 (dep line
254) and that `splat3d::sh::sh_eval_deg3` encodes. Basis polynomials
evaluated in Cartesian form on `(x, y, z) = d`; no `atan2`/`asin`,
so kernel is total over the unit sphere.

Graphics interp: 3 channels (RGB), 48 coefs per Gaussian. Cognitive
interp: 1 channel inquiry-state per cell. Channel count is the only
axis differing.

## Inquiry-direction in the cognitive interpretation

Per design-table line 41, view-dir-dependent color becomes inquiry-
dir-dependent state with axes `vocab × thinking_style`. The unit
direction is constructed from the cell's current question vector:

- **vocab axis** (~17 dims): `Base17` fingerprint from
  `src/hpc/fingerprint.rs` (Pearl 2³ × 17 = vocab dim)
- **thinking_style axis** (~34 dims): the 34 style primitives from
  `src/hpc/styles/`

The 17×34 product reduces to a unit-3-vector via the fingerprint-to-
axis projection `linalg::sh` consumers already use; A3's kernel
takes `d: Vec3` and never sees the basis. Identity-monomorphic
across both interpretations.

## Parity gate — bit-exact

Per done-criteria line 326: bit-exact, not ULP-bounded. New kernel
must produce IEEE-754-identical f32 outputs on every input. Achievable
because (a) basis polynomials are pure mul/add, no transcendental;
(b) evaluation order is fixed by bundle composition; (c) both
backends FMA in the same nesting.

Test: feed `splat3d::sh::sh_eval_deg3` and `splat4d::sh::sh_eval_deg3`
the same 10K direction + coefficient fixtures, `assert_eq!` on
`u32`-bit representation per channel. Any FMA reorder breaks it.

## Backend split

- **AVX-512** (`simd_avx512.rs`): 16 lanes, 16 directions in parallel,
  one `vfmadd231ps` per `(l,m)` accumulator. ~16 FMAs/channel.
- **NEON** (`simd_neon.rs`): 4-wide blocks of directions, `vfmaq_f32`.
- **Scalar** (`simd_scalar.rs`): 16-term sum, parity reference and
  bit-exact target.

LoC budget: ~150/backend + ~50 dispatch + parity test ≈ ~550.

## SIMD bundles — B-Splat + B-Compose

A3 consumes exactly two bundles and adds none:

- **B-Splat** (`splat_f32x16`): broadcast each of 16 coefs across 16
  tile lanes
- **B-Compose** (`hreduce_sum_f32x16` for alpha;
  `revise_truth_f32x16` for NARS): the dot-product reduction
  `Σ c_{l,m} Y_l^m(d)`. Closure-swappable — `splat4d-nars-compose`
  feature rebinds reduction without touching basis evaluation.

A3 must not reach past either bundle into raw intrinsics (forbidden
constraint #2).

## Test fixture — 10K random directions + coefficient sweep

- **Directions**: 10K unit vectors via Marsaglia. Seed
  `0xA3_5H_DE_G3_C0FFEE`.
- **Coefficients**: 10K independent 16-coef sets ~ `Normal(0,1)`
  (Inria 3DGS reference distribution)
- **Assertion**: per-direction × per-coef-set,
  `u32::from_le_bytes(new) == u32::from_le_bytes(splat3d::sh)` per
  channel. Mismatch = parity-gate failure, A3 does not land.

The fixture doubles as microbench: 10K × 16 FMAs gives a per-eval
latency floor SG2 (p95 ≤ 20ms over L1..L4 cascade) inherits as a
budget line-item.
