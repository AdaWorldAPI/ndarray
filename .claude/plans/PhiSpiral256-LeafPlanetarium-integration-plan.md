# PhiSpiral256 Leaf Planetarium Integration Plan — ndarray

## Goal

Define the ndarray-side primitive for **PhiSpiral256**, a leaf-location codec that complements PolarQuant and CAM_PQ.

This plan keeps the terms separate:

```text
PolarQuant
  magnitude / similarity / distance-like compression

CAM_PQ
  meaning / semantic basin compression

BGZ17
  golden-ratio offset/stride recoverable sampling skeleton

PhiSpiral256
  orthogonal local residual location encoded as golden-spiral address

Fisher-z cosine
  optional statistical angular scorer / confidence gate after candidate ranking
```

PhiSpiral256 is not HHTL. It is a **planetarium lane** that can sit beside HHTL, CAM_PQ, PolarQuant, and BGZ17 in struct-of-arrays pipelines.

## Core thesis

```text
The leaf should not carry the full difference.
The leaf should carry where the unexplained difference lives.
```

PhiSpiral256 compresses orthogonal residual location into one byte:

```text
continuous orthogonal local residual direction
        ->
golden-spiral address 0..255
```

Then the packed atom can combine:

```text
8 bits  phi_spiral_id
4 bits  magnitude band
2 bits  BGZ offset family
2 bits  BGZ stride family
```

This yields:

```text
256 spiral locations
× 16 magnitudes
× 4 offset families
× 4 stride families
= 65,536 recoverable local residual states
```

## Relationship to BGZ17

BGZ17 is the recoverable sampling skeleton, not the same thing as PhiSpiral256.

BGZ17 supplies:

```text
golden-ratio offset/stride sampling
16k -> Base17 compression
recoverable sparse sampling schedule
SIMD-friendly Base17 kernels
```

PhiSpiral256 supplies:

```text
local orthogonal location address
spiral neighbor structure
location-lane table lookup
```

Together:

```text
BGZ17 tells how to sample/recover.
PhiSpiral256 tells where the unexplained residual lives.
```

## Relationship to Palette256

Do not mix the terms.

```text
Palette256
  candidate ranking / codebook indexing layer

PhiSpiral256
  golden-spiral address space for local residual location

Fisher-z cosine
  statistical scorer/gate after candidate ranking
```

PhiSpiral256 may reuse the existing 256-entry palette/distance-table mechanics, but its semantics are a location address, not a generic palette prototype.

## Coordinate flow

```text
source residual
        ->
remove CAM_PQ / meaning-explained component
        ->
orthogonal residual
        ->
BGZ17 offset/stride sampling family
        ->
local planetarium coordinate
        ->
PhiSpiral256 address
        ->
packed leaf atom
```

If a Poincare/Mobius chart is used:

```text
parent anchor
        ->
Mobius recenter
        ->
local tangent/orthogonal residual
        ->
PhiSpiral256 address
```

Implementation must keep the chart optional. The first test can use Euclidean local tangent coordinates.

## PhiSpiral256 construction

Use golden/Weyl angular sequence:

```text
theta_k = 2π * frac(k / φ)
```

Use radius depending on the chart:

```text
Euclidean local chart:
  r_k = sqrt((k + 0.5) / 256)

Poincare chart:
  rho_k = arcosh(1 + u_k * (cosh(rho_max) - 1))
  r_k   = tanh(rho_k / 2)
  u_k   = (k + 0.5) / 256
```

Store centers as compact fixed-point directions:

```rust
pub struct PhiSpiralCenterQ15 {
    pub x_q15: i16,
    pub y_q15: i16,
    pub radius_q15: u16,
}

pub struct PhiSpiral256 {
    pub centers: [PhiSpiralCenterQ15; 256],
    pub neighbors: [[u8; K]; 256],
    pub distance: [u16; 256 * 256],
}
```

`K` should start as 8 or 16.

## Packed atom

```rust
#[repr(transparent)]
pub struct PhiSpiralLeafAtom16(pub u16);

impl PhiSpiralLeafAtom16 {
    pub fn new(spiral_id: u8, mag4: u8, offset2: u8, stride2: u8) -> Self;
    pub fn spiral_id(self) -> u8;
    pub fn mag4(self) -> u8;
    pub fn offset2(self) -> u8;
    pub fn stride2(self) -> u8;
}
```

Bit layout:

```text
bits  0..=7   phi_spiral_id
bits  8..=11  mag4
bits 12..=13  offset family
bits 14..=15  stride family
```

## SoA layout

The primitive should be SoA-native:

```rust
pub struct LeafPlanetariumSoA {
    pub leaf_id: Vec<u64>,
    pub atom_start: Vec<u32>,
    pub atom_len: Vec<u8>,

    pub atom16: Vec<PhiSpiralLeafAtom16>,
    pub confidence_q: Vec<u8>,

    // sibling lanes, not owned semantically by PhiSpiral256
    pub cam_pq_id: Vec<u16>,
    pub polarquant_id: Vec<u8>,
    pub replay_ref: Vec<u64>,
}
```

Multiple atoms per leaf are allowed:

```text
K=1  ultra-fast single local residual address
K=2  branching / ambiguous residual
K=4  rich leaf constellation
K=8  debug or high-certification mode
```

## Candidate ranking and scoring

PhiSpiral256 can provide a cheap nearest/neighbor candidate list:

```text
local residual vector
        ->
nearest spiral_id
        ->
neighbor ids from spiral_neighbors[spiral_id]
```

Fisher-z cosine is a separate optional scoring pass:

```text
candidate spiral centers
        ->
cosine(local_direction, center_direction)
        ->
Fisher-z transform
        ->
margin/confidence gate
```

Keep this naming strict:

```text
PhiSpiral256 addresses.
Palette/ranking narrows.
Fisher-z judges.
```

## Calibration targets

Calibrate against these baselines:

```text
A. Mag4 only
   16 magnitude bands, no local location

B. BGZ17 L1 / weighted L1
   existing Base17 distance behavior

C. BGZ17 sign agreement
   existing direction-ish sign kernel baseline

D. PolarQuant only
   magnitude/similarity compression without local residual place

E. PhiSpiral256 Euclidean
   local orthogonal direction encoded as spiral address

F. PhiSpiral256 Poincare/Mobius
   parent-local chart before spiral address

G. PhiSpiral256 + Fisher-z gate
   same as E/F, but with statistical angular confidence

H. Hybrid packet
   CAM_PQ meaning + PolarQuant magnitude + PhiSpiral256 location + BGZ17 recovery schedule
```

## Metrics

```text
location recall@1
location recall@k
next-basin recall@1 if used by caller
candidate fanout reduction
leaf replay reduction
atom count per leaf
bytes per leaf
palette / spiral occupancy entropy
Gini coefficient of spiral bin use
boundary / high-curvature failure rate
orthogonal residual reconstruction error
Fisher-z margin distribution
wrong-high-confidence rate
ns per encode
ns per route / candidate lookup
L1 cache footprint
```

## Distortion checks

The spiral must be tested for projection artifacts:

```text
center over-resolution
boundary under-resolution
kissen distortion
trapez distortion
cluster collapse
neighbor discontinuity
```

Required diagnostics:

```text
spiral occupancy histogram
spiral neighbor graph visualization
distance matrix heatmap
radial-bin occupancy
angular-bin occupancy
Möbius-equivariance error if Poincare mode is enabled
```

## Golden tests

```text
pack/unpack atom16 roundtrip
spiral center generation deterministic
neighbor table symmetric-enough sanity check
self-distance zero
distance matrix symmetry
nearest center deterministic
multi-atom packet roundtrip
SoA slicing preserves atom order
```

## Benchmarks

```text
encode_phi_spiral_1k
encode_phi_spiral_100k
nearest_phi_spiral_scalar
nearest_phi_spiral_simd_optional
neighbor_lookup_1m
atom16_pack_unpack_1m
soa_scan_atom16_1m
fisher_gate_top8_1m
```

## Acceptance criteria

- PhiSpiral256 is documented and implemented as location lane, not meaning lane and not magnitude lane.
- Atom16 pack/unpack is deterministic and zero-allocation.
- 256 centers and distance table are deterministic for the same mode/config.
- Existing BGZ17 and palette terminology remains unmixed.
- PhiSpiral256 beats Mag4-only on orthogonal location recall.
- Hybrid packet reduces leaf replay / exact recovery workload in at least one synthetic fixture.
- Fisher-z, if used, improves wrong-high-confidence rate without being required for the hot path.

## First fixture

Synthetic local residual field:

```text
parent anchor at origin
16 known orthogonal directions
random magnitudes
controlled noise
masked/missing locations
known target spiral sector
```

Then add:

```text
BGZ17-derived residual vectors
CAM_PQ meaning axis removal
Poincare/Mobius recentering
multi-atom leaves
```

## Wall sentence

```text
PhiSpiral256 is the leaf planetarium: a one-byte golden-spiral place for the orthogonal difference that meaning and magnitude did not explain.
```
