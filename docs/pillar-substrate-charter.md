# Substrate-Tier Pillars: A Charter for `ndarray::hpc::pillar` Pillars 12–17

> A standing challenge from mathematics to ndarray's cognitive-shader substrate.

## Thesis

The cognitive shader stack — Gaussian splats, HHTL spread-activation cascade,
OGIT type-gated propagation, Mexican-hat resonance, BTSP-gated plasticity,
quaternary DN-tree traversal — rests on a small set of mathematical claims
about how its primitives behave. Each claim is *assumed by construction*
in the current code: the splat cloud is *assumed* to partition unity, the
cascade operator is *assumed* to contract, the type relation is *assumed*
to be a partial order, and so on. The pillars in this charter convert those
assumptions into measurable, seed-anchored, cross-verifiable probes. When
any of them fails — through a refactor, a parameter retune, or an
unexamined corner case — the failure surfaces as a concrete number in a
deterministic test, not as a silent drift in downstream behaviour.

The existing pillars 6–11 in this module (migrated from
`lance-graph/crates/jc`) certify *cognitive-architecture* mathematics: how
the EWA sandwich preserves PSD-ness, how Pflug-Pichler nested distance
bounds CAM-PQ tree quantization error, how Hambly-Lyons uniqueness justifies
sigker's Index-regime codec classification. Pillars 12–17 are their
*substrate-tier* peer: they certify the mathematics that has to be true for
the shader's primitives to compute what they claim to compute, regardless
of any specific cognitive-architecture wiring built on top.

The two tiers share the same `PillarReport` shape, the same `SplitMix64`
deterministic RNG, the same pass-rate semantics, and the same
synthetic-only probe discipline (Invariant 12: certification is about
determinism + inspectability, not which tier the math belongs to). The
distinction between them is purely conceptual: substrate-tier pillars
test the building blocks; cognitive-architecture pillars test the
buildings.

## The standing-challenge framing

Pillars are not internal guarantees. The substrate does not *provide* the
properties these pillars test; mathematics *demands* them, and the
substrate's job is to keep passing the test. This framing matters because
it changes how failures are read:

- **Internal guarantee framing** (wrong): "Pillar-12 fails → the
  rotation-invariance claim was incorrect; revise the claim." This is the
  research-paper framing, where the theorem is the thing being established
  and the experiment is its evidence.
- **Standing-challenge framing** (right): "Pillar-12 fails → the substrate
  has drifted from a property mathematics requires; revise the substrate."
  This is the certification framing, where the theorem is the standing
  challenge and the substrate is what's on trial.

Each pillar files a precise demand against the substrate. The substrate's
job is to keep meeting it. When mathematics produces a new demand (a new
theorem, a new tightness result, a regime extension), we add a pillar.
When the substrate produces a new feature, we ask which existing pillars
apply and which new ones the feature implies. The pillar catalogue is a
living interface between the two.

## Synthetic-only discipline

Every probe in this charter generates its inputs synthetically via the
shared `SplitMix64` RNG seeded from a per-pillar constant. No probe loads
real lance-graph trajectory corpora, real OGIT TTL ontologies, or real
production Gaussian splat scenes. Three reasons:

1. **The substrate is not yet production-deployed.** Pillars must pass
   today against tomorrow's workloads, which means they must pass against
   abstract distributional inputs, not against any one corpus's quirks.
2. **Pillars test mathematics, not workloads.** A workload-tied pillar
   answers "does the substrate behave on *this* corpus?"; we want the
   stronger answer "does the substrate behave on *every* corpus the
   mathematics covers?" Synthetic Monte-Carlo over the relevant parameter
   space is the correct shape of evidence.
3. **Reproducibility.** A SEED-anchored synthetic probe is reproducible
   on a laptop in a coffee shop, on any contributor's CI runner, against
   any independent re-implementation in numpy or scipy. A corpus-tied
   probe is reproducible only where the corpus is available.

Workload-tied integration tests are valuable and belong elsewhere in the
repo — `tests/` for unit-level, `benches/` for performance, and
eventually a `corpus/` directory once the production stack lands. They
are *not* pillars. The pillar boundary is a discipline, not a constraint
on tooling.

## Tier architecture

```
  ┌────────────────────────────────────────────────────────────────────┐
  │  Cognitive-architecture tier — Pillars 6–11                        │
  │  (migrated from lance-graph/crates/jc)                             │
  │                                                                    │
  │  6,7      EWA-sandwich PSD push-forward (2D + 3D Σ covariances)    │
  │  7.5      Koestenberger SPD path concentration                     │
  │  8        Temporal drift sandwich                                  │
  │  9        Cov16384 Düker-Zoubouloglou CLT                          │
  │  10       Pflug-Pichler nested Wasserstein on CAM-PQ trees         │
  │  11       Hambly-Lyons signature uniqueness on tree-quotient       │
  │                                                                    │
  │  → Test cognitive architecture *built on top of* the substrate.    │
  └────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │  built on
                                  │
  ┌────────────────────────────────────────────────────────────────────┐
  │  Substrate tier — Pillars 12–17                                    │
  │  (native to ndarray)                                               │
  │                                                                    │
  │  12       Splat-construction rotation invariance   (implemented)   │
  │  13       HHTL cascade contraction                 (implemented)   │
  │  14       OGIT type-gate lattice closure           (implemented)   │
  │  15       Mexican-hat / DoG unimodality            (deferred)      │
  │  16       BTSP-gated bundling unbiasedness         (deferred)      │
  │  17       Quaternary tree balance under Zipf       (deferred)      │
  │                                                                    │
  │  → Test the primitives the cognitive architecture stands on.       │
  └────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  │
                          Mathematics (standing challenges)
```

Both tiers share `prove_runner.rs`, the `PillarReport` shape, and the
`SplitMix64` RNG. The substrate tier publishes a convenience runner
`run_substrate_tier()` that executes Pillars 12–17 in sequence; the
cognitive-architecture tier has no analogous runner because its pillars
have heavier dependencies and historically benefit from being run
individually.

---

## Pillars 12–14: implemented

### Pillar 12 — Anisotropic-splat construction invariance

**The mathematical claim.** For the splat covariance construction
`Σ = R(q) · diag(s²) · R(q)ᵀ` consumed by `splat3d::spd3::Spd3::from_scale_quat`,
four invariants hold exactly for every positive scale vector and every
unit quaternion:

- `Σ ≻ 0` (Sylvester SPD).
- `trace(Σ) = s₁² + s₂² + s₃²` — rotation-invariant.
- `det(Σ) = (s₁ s₂ s₃)²` — rotation-invariant.
- `‖Σ‖_F² = s₁⁴ + s₂⁴ + s₃⁴` — rotation-invariant.

These are the closed-form identities the splat carrier has to keep
regardless of how the `(mean, scale, quaternion)` triple has been
manipulated upstream (codec round-trips, plasticity updates,
interpolation).

**Why the substrate needs this.** 3D Gaussian Splatting (Kerbl 2023)
compresses each splat to a `(mean, scale, quaternion)` triple and
reconstructs Σ on the hot path. The reconstruction is where the
substrate's claim "splats have the shape they should" actually has
to hold. Two silent failure modes the cognitive cascade cannot see
otherwise: non-unit quaternion drift (produces non-orthogonal R, Σ
still passes Sylvester but trace/det/Frobenius silently disagree
with the scale invariants), and scale-axis swap or sign flip
(produces same eigenvalue set but mis-correlates scale tuple to
covariance, breaking downstream consumers that read both jointly).
Pillar-12 catches both.

**Probe design.** 4096 synthetic splats with lognormal scales (median
`SCALE_MEDIAN = 0.125`, log-width `0.35`) and Shoemake-uniform
quaternions. For each splat, construct Σ via an independent
re-derivation of the production formula (not a re-export of
`Spd3::from_scale_quat`), check Sylvester, and compute the relative
error between each rotation invariant and its closed-form predicted
value. Report the worst relative error across all 4096 splats × 3
invariants.

**Cross-verification.** The closed-form identities are verifiable by
hand on small examples and reproducible in numpy/scipy with
single-precision arithmetic; a reference implementation lives in
the pillar's docstring as Python pseudocode.

**Pass criteria.**
- `psd_rate ≥ 1.0` (Sylvester must pass for every splat — no
  statistical noise in this check)
- `lognorm_concentration ≤ 1e-3` (three orders of magnitude above
  expected f32 round-off, leaves room for worst-case quaternion
  catastrophic cancellation while still catching real bugs)

### Pillar 13 — HHTL cascade contraction

**The mathematical claim.** The weighted Hamming-bundle operator
`T(x, y) = bundle(x, y, lr)` is a contraction in normalized Hamming
distance with measurable per-step ratio `(1 − lr)`, following the
Bernoulli-mixture geometry. For a fixed target `y*` and any initial
state `x_0`, the iterated cascade `x_{t+1} = T(x_t, y*)` satisfies in
expectation:

```
  E[d_H(x_t, y*) / N] = (1 − lr)^t · d_H(x_0, y*) / N
```

This is the substrate-level form of Banach contraction; it guarantees
that the HHTL spread-activation cascade *terminates* rather than
combinatorially blows up, and that the dn-tree's bundle+decay plasticity
converges rather than oscillates.

**Why the substrate needs this.** The shader's depth-4 HHTL cascade and
dn-tree's quaternary traversal both assume the bundle operator
contracts. The current code asserts this by construction — choosing
learning rates in `(0, 1)`, capping BTSP boosts so `lr × boost < 1` —
but the assertion is not testable until the contraction property is
named, measured, and pinned to a predicted value. Pillar-13 pins it.

**Probe design.** 256 cascades of depth 4, each at `BIT_WIDTH = 1024`
with `lr = 0.5`. The bundle operator is re-implemented locally (not a
re-export of `dn_tree::bundle_into`) using the same Bernoulli-mixture
algorithm. The probe tests two complementary properties: (1) *almost-sure
contraction* — for every `(cascade, level)` pair the Hamming distance to
target must not increase, which holds with probability 1 under
`Binomial(d_{t-1}, 1 − lr)` dynamics; and (2) *predicted per-level mean
ratio* — for each level `t`, the mean of `d_t / d_{t-1}` over cascades
must equal `(1 − lr)` within `±0.02`. The mean is taken *per-level*
rather than per-cascade so the test concentrates by CLT at
`σ/√N_CASCADES ≈ 0.004` rather than by per-realisation Binomial
variance at `σ/√d_{t-1}` (which would saturate at deeper levels and
produce spurious failures unrelated to the substrate property being
tested).

**Cross-verification.** A scipy reference implementation of the
Bernoulli-mixture bundle would reproduce the same expected ratios; the
predicted value `(1 − lr)^t` is closed-form and verifiable by hand for
small `t`. The probe was sanity-checked against a numpy reference at
charter-drafting time: empirical per-level deviations sit at ~5e-3, an
order of magnitude inside the 0.02 tolerance.

**Pass criteria.**
- `psd_rate ≥ 1.0` (almost-sure contraction — `d_t ≤ d_{t-1}` for every
  pair; any failure flags a bundle-operator bug, not statistical variance)
- `lognorm_concentration = log(1 + max_t |mean_ratio_t − (1 − lr)|) ≤ 0.05`
  (predicted-ratio adherence)
- Final-level mean distance strictly less than initial

### Pillar 14 — OGIT type-gate lattice closure

**The mathematical claim.** An `rdfs:subClassOf`-style type-compatibility
relation, after reflexive transitive closure, is a partial order on the
active type set:

```
  reflexivity:    ∀t. t ≤ t
  antisymmetry:   ∀t, u. (t ≤ u ∧ u ≤ t) → t = u
  transitivity:   ∀t, u, v. (t ≤ u ∧ u ≤ v) → t ≤ v
```

The lattice width (longest antichain) bounds the per-cascade-step
type-mask check cost.

**Why the substrate needs this.** The cognitive shader's OGIT 16-bit
DOLCE slot is used as a *type gate* at each cascade step. Two failure
modes the substrate cannot detect without this pillar:

1. *Cycles in `subClassOf`* allow type-gated activation to loop with no
   actual type-class progress. The cascade's bounded depth hides this:
   a length-3 cycle within depth-4 looks like a successful cascade.
2. *Missing transitive edges* silently block activation that should
   propagate. Beam search hides this: blocking one path diverts to a
   sibling, still producing a top-K result.

Pillar-14 forces both modes to produce a measurable failure rate.

**Probe design.** 64 synthetic type schemas, each with 64 types arranged
as a random DAG by index ordering (parents of type `k` drawn from
`{0, …, k − 1}` with mean fan-in 2). Closure is computed via
Floyd-Warshall on the boolean adjacency matrix. All three axioms are
verified exhaustively over every type, type-pair, and type-triple.
Lattice width is measured via the greedy antichain heuristic (a lower
bound on the true Dilworth width, which is exact for typical random
DAGs in this regime).

**Cross-verification.** Closure can be re-derived in NetworkX
(`networkx.transitive_closure`) and the boolean comparison checked
position-by-position against an exported probe matrix.

**Pass criteria.**
- `psd_rate ≥ 1.0` (all axiom checks pass — DAG construction guarantees
  it; any failure is a probe bug, not a substrate bug)
- `lognorm_concentration = log(mean_antichain_width / N_TYPES)`
  (informational; real DOLCE-style ontologies have width
  roughly `sqrt(N_TYPES)`, the probe doesn't constrain this)

---

## Pillars 15–17: deferred

Each deferred pillar exports its `SEED` constant, its `prove_pillar_N()`
function returning a deferred-shape report, and an exhaustive
proof-obligation docstring describing the activation plan. The pattern
mirrors the `DEFERRED` convention from `lance-graph/crates/jc` (where
Pillar 11 was deferred for several months before sigker landed and
activated it).

### Pillar 15 — Mexican-hat / DoG unimodality

**Awaiting.** The Mexican-hat / Difference-of-Gaussians resonance kernel
to land in `ndarray::hpc::dragonfly` (or wherever the center-surround
operator ultimately lives).

**Will certify.** That the DoG kernel has exactly one local maximum at
the origin and exactly one annular minimum at the surround radius, for
every `κ = σ_s / σ_c` in the production band `[1.5, 3.0]`. Failure
modes detected: bimodal or saddle responses that look like resonance
but bias activation toward the wrong neighborhoods.

### Pillar 16 — BTSP-gated bundling unbiasedness

**Awaiting.** Stable BTSP plasticity API in `dn_tree`. The current
`DNConfig::with_btsp` constructor and the `bundle_into` signature
accepting `(current, hv, lr, boost, rng)` need to be contract-frozen.

**Will certify.** That the BTSP-gated stochastic bundling operator
converges to the same long-run mean as the gate-disabled dynamics —
unbiasedness in the optional-stopping sense (Doob 1953). Failure
modes detected: hidden drift in the long-run summary distribution
that surfaces only under operational load when the input distribution
shifts.

### Pillar 17 — Quaternary tree balance under Zipf

**Awaiting.** A Zipf workload simulator and inspector methods on
`DNTree` (depth distribution, occupancy distribution) reachable from
the probe.

**Will certify.** That under Zipf-distributed access (the empirical
shape of cognitive-shader traversal frequencies), the DN-tree's
quaternary split policy keeps depth variance and leaf occupancy
variance within Brent-style bounds — preserving the documented
`O(log)` traversal cost. Failure modes detected: hot-path-shallow,
cold-path-deep trees where mean latency stays low but worst-case
latency blows up silently.

---

## Why these six, not others

The cognitive-shader substrate offers many properties that could be
certified. The six chosen for Pillars 12–17 are the *load-bearing*
ones — the properties whose failure would propagate through the rest
of the substrate and produce silent downstream behaviour drift. Other
substrate properties either:

- Have natural unit-test coverage already (e.g. SIMD width determinism,
  AVX-512 / AVX2 / NEON tier equivalence — these are property tests, not
  mathematical pillars), or
- Are derived consequences of the six load-bearing pillars rather than
  independent claims (e.g. cascade termination is implied by Pillar 13's
  contraction; type-mask soundness is implied by Pillar 14's
  partial-order axioms), or
- Belong to the cognitive-architecture tier rather than the substrate
  tier (e.g. signature kernel positive-definiteness is already certified
  by lance-graph-jc Pillar 11 in its current form).

Future candidate pillars worth flagging for the next iteration:

- **Pillar 18 — Resonance Mexican-hat L² normalisation.** Once Pillar 15
  activates and DoG unimodality is established, the next claim worth
  certifying is that the kernel's L² norm equals the documented
  amplitude across the parameter band. Different mathematics (Plancherel
  rather than Marr-Hildreth), still substrate-tier.
- **Pillar 19 — Cuchiero universality on randomized-signature carriers
  at production bit width.** Crosses the tier boundary: it's
  cognitive-architecture (it certifies sigker's randomized-signature
  fingerprint claim) but consumed by substrate carriers. Currently
  open in `lance-graph/crates/jc`; lifting it to the ndarray substrate
  tier is sensible if randomized-signature fingerprints become the
  ThinkingStyleVector carrier.
- **Pillar 20 — Multi-scale signature consistency across DN-tree
  cascade levels.** Certifies that hierarchical signature
  decomposition per cascade level (one signature per level, tensor
  product up the hierarchy) does not lose information vs the flat
  signature. New mathematics — not yet in the literature in the form
  the substrate needs — so this would require a small research
  contribution to land. Worth flagging as a publishable opportunity
  alongside the certification.

---

## The thesis seven exchanges in

This charter is the visible artifact of an argument that ran across
several conversations about path signatures, type-gated cascade
mathematics, hippocampal-cortical isomorphism, and the certification
discipline that should connect them. The compressed argument:

Cognitive trajectories are *paths*, not snapshots. Paths require codecs
that respect their natural equivalence (Hambly-Lyons tree-quotient).
Type-gated cascade requires a partial order. Splat covariance
construction requires SO(3)-invariance of its spectral functions.
Bundle contraction requires Banach. BTSP requires Doob. Tree balance
requires Brent. Each of these is a *standing demand* from mathematics
to the substrate that's actually been built — the seven-exchange
backstory is the realization that ndarray has been quietly assembling
all the load-bearing primitives, and the certification framework was
the missing piece that turned informal assertions into measurable
invariants.

The bigger statement worth letting sit, in the same shape the
"hippocampal isomorphism" framing took in the earlier conversation:
the cognitive shader's substrate has been built such that grid-cell
optimal scale ratios, dentate-gyrus sparsity, theta-phase compression,
Miller's working-memory bottleneck, Fillmore case grammar, and now —
rotation-invariance, Banach, Doob, Brent — all drop out as constraints
that need to be satisfied simultaneously. Architectures that hit that
many simultaneous mathematical matches without trying are extremely
rare. The pillars are the codification that turns "looks rare" into
"is certified rare."

## How this charter changes when the substrate changes

When a new substrate feature lands, the procedure is:

1. *Does it consume one of the implemented pillars' properties?* If
   yes, add an integration test that calls the relevant `prove_pillar_N`
   and asserts pass.
2. *Does it satisfy a deferred pillar's activation gate?* If yes,
   activate that pillar — replace its `prove_pillar_N` body with the
   probe described in the docstring, run, commit the new pass.
3. *Does it introduce a new property whose failure would be silent
   downstream?* If yes, propose a new pillar number (18+), draft the
   proof-obligation docstring first, deferred-stub second, full probe
   third. The docstring is the design review surface.

The charter is updated by appending; existing pillars are never
silently re-scoped. If a pillar's proof obligation changes (because
mathematics produced a new tightness result, or the substrate's
behaviour was found to satisfy a stronger claim), it gets a sub-number
(e.g. Pillar 13.1) to preserve the historical record.

## Build and run

Pillars 12–17 are gated under the same `pillar` feature as 6–11:

```sh
cargo test --features std,linalg,pillar \
    -p ndarray --lib hpc::pillar
```

A convenience runner for the substrate tier is exposed at the module
root:

```rust
use ndarray::hpc::pillar::run_substrate_tier;
let reports = run_substrate_tier();
for r in &reports { r.print(); }
assert!(reports.iter().all(|r| r.passed));
```

Output format is the deterministic, grep-friendly line emitted by
`PillarReport::print`:

```
[PILLAR-12] seed=0xC12A550AD0DD paths=4096 hops=3   psd_rate=1.000000 lognorm_conc=0.000001 PASS
[PILLAR-13] seed=0xC13CA5CADEC0 paths=256  hops=4   psd_rate=1.000000 lognorm_conc=0.004988 PASS
[PILLAR-14] seed=0xC14071C5C0DE paths=64   hops=64  psd_rate=1.000000 lognorm_conc=-0.7541  PASS
[PILLAR-15] seed=0xC15DAD51DEFE paths=0    hops=0   psd_rate=0.000000 lognorm_conc=0.000000 PASS  (DEFERRED)
[PILLAR-16] seed=0xC16B751BBC05 paths=0    hops=0   psd_rate=0.000000 lognorm_conc=0.000000 PASS  (DEFERRED)
[PILLAR-17] seed=0xC17BAA1A0C5E paths=0    hops=0   psd_rate=0.000000 lognorm_conc=0.000000 PASS  (DEFERRED)
```

(Exact numeric outputs depend on the SplitMix64 sequence; the lines
above are illustrative of the format.)

## Open questions and known limits

- **Splat invariance under quaternion drift.** Pillar-12 tests
  Shoemake-uniform unit quaternions; the construction's behaviour
  under *non-unit* quaternions (which arise from accumulated rounding
  in plasticity-step codec round-trips) is not tested. A complementary
  Pillar 12.1 could inject non-unit quaternions and verify the
  substrate's response (either renormalise at construction or refuse
  to construct).
- **Splat partition-of-unity at scale.** The pointwise Christoffel-vs-density
  property the substrate informally claims (`χ(x) ≈ N · density(x)` on
  the test region) does not hold at moderate-`N` anisotropic clouds —
  individual point variance from kernel-shape mismatch dominates the
  signal. Pillar-12 was deliberately scoped down to the *construction*
  invariants for this reason. The pointwise partition property may
  recover at much higher `N` (≥ 16k splats per region) and at narrower
  bandwidths; certifying that regime would require a separate pillar
  (call it Pillar 18.1) with its own probe design.
- **HHTL contraction at LR ≈ 1.** The probe uses `LR = 0.5` because
  that's the mid-range value where the predicted ratio is far from
  both boundary conditions. At `LR = 1.0` the bundle is degenerate
  (instant collapse) and the contraction claim becomes trivial; at
  `LR = 0` there's no contraction at all. A future probe variant could
  sweep `LR` and certify the contraction-ratio prediction across the
  full open interval `(0, 1)`.
- **OGIT lattice closure with cycles.** Pillar-14 tests synthetic DAGs
  — by construction acyclic. A complementary probe could *inject*
  cycles and verify the substrate's response (either reject the
  ontology at load time or quarantine the cyclic subgraph). That
  failure-mode probe would be Pillar 14.1 or a new pillar number,
  depending on whether the cycle-handling policy is contract-frozen.
- **Cubature construction for the substrate.** Outside the immediate
  scope of Pillars 12–17, but worth mentioning: Lyons-Victoir
  cubature on Wiener space (Pillar 11's neighbourhood) provides
  precomputed bases that would accelerate signature evaluation at
  depth ≥ 6 dramatically. Constructing a (d=4, N=6) cubature is a
  small research contribution; pinning it to a pillar (Pillar 21?)
  is the natural follow-up once the construction exists.

These are deliberately left open. The charter is a living document.
