# A2: CascadeAddr + from_position/to_position_center

Worker A2 of PR-X4 (W4-W5). Spawns after A1's TileInstance v2
refactor lands. **Hard gate on PR-X10 A12b's L4 Hilbert-3D fix
landing on master.**

## Gate — PR-X10 A12b L4 fix

Verbatim symptom (pp13-brutally-honest-tester-verdict.md P0-4):
`hilbert3d_encode([15,15,15], 4) → 2925, expected 4095`. A12b must
ship the `NEXT_STATE`/`H_TO_XYZ` re-derivation from Hamilton 2006
Table 2 + `round_trip_level4_exhaustive` (4096 cells × 4 µs ≈ 16 ms)
before A2 starts. **A2 must NOT re-introduce a bespoke Hilbert-3D**
(forbidden constraint #1).

If A12b slips past W3, A2 stubs the L4 path: `Err(NotReadyL4)`,
ships L1-L3 addressing only. `parent()`/`children()` remain
functional since they're pure nibble ops.

## API surface

```rust
pub struct CascadeAddr(u16);  // 4 nibbles, one per tier level

impl CascadeAddr {
    pub fn level(&self, l: u8) -> u8 { (self.0 >> (l * 4) & 0xF) as u8 }
    pub fn parent(&self) -> CascadeAddr { CascadeAddr(self.0 & !0xF000) }
    pub fn children(&self) -> [CascadeAddr; 16] { ... }
    pub fn from_position(p: Vec3, bbox: AABB, level: u8) -> CascadeAddr {
        CascadeAddr(linalg::hilbert::hilbert3d_encode(p_quantised, level) as u16)
    }
    pub fn to_position_center(&self, bbox: AABB) -> Vec3 { ... }
}
```

The 4-nibble layout: one nibble per L1..L4 tier, 16 children per
parent. `parent()` masks off the L4 nibble. `children()` enumerates
all 16 nibble values at the L4 slot.

## AABB quantisation convention

Per `linalg::hilbert::hilbert3d_encode` contract, the input is a
quantised 3-tuple of unsigned ints. At level `k`, each axis has
`2^k` cells:

| level | cells/axis | index range  | bits |
|-------|------------|--------------|------|
| 1     | 2          | [0, 8)       | 3    |
| 2     | 4          | [0, 64)      | 6    |
| 3     | 8          | [0, 512)     | 9    |
| 4     | 16         | [0, 4096)    | 12   |

L4's 12-bit range fits within 3 of the 4 cascade-addr nibbles. NB:
if A12b's actual encode returns a monolithic per-call index rather
than packed cascade, A2 must call once per tier and assemble nibbles
itself. **Flag this discrepancy with the A12b author at spawn.**

Quantisation: `q = floor((p.x - bbox.min.x) / (bbox.size.x) * (1 << level))`
clamped to `[0, (1 << level) - 1]`. Same for y, z.

## Tests

- **Exhaustive level=4 round-trip** (4096 cells × 3 axes): for each
  of 4096 quantised positions, `decode(encode(p)) == p`. ~16 ms.
- **Exhaustive level=1..3 round-trip**: already pass under current
  A12b — just verify under the splat4d call sites.
- **AABB sanity**: corner cells map to `level()==0` and
  `level()==(1<<level)-1` per axis.
- **parent/children round-trip**: for any addr,
  `addr.children()[i].parent() == addr` for all i.

## SIMD bundle — B-Cascade-Permute

A2 consumes one bundle:

- **B-Cascade-Permute** (`shuffle_lanes_4x4 ∘ transpose_16x16`): the
  cross-tier rotation L_k → L_{k+1}. The 4×4 stride identity made
  executable. Without this bundle the cascade is just a hierarchy
  of independent grids.

A2 must not reach past into raw shuffle intrinsics. If the bundle
primitive is missing in `ndarray::simd`, file a pre-PR-X4 gating
PR against the vertical-simd-consumer-contract before spawning.

## Exit criteria

- [ ] A12b's `hilbert3d_encode([15,15,15], 4) == 4095` and
      `round_trip_level4_exhaustive` green on master
- [ ] A2's exhaustive level=4 round-trip green
- [ ] `CascadeAddr::from_position` and `to_position_center`
      round-trip on 10K random positions within the unit AABB
- [ ] `parent`/`children` round-trip exhaustive
- [ ] L1-L3 addressing exercisable from A6's frame_pipeline (smoke
      gate the cascade addressing layer)
- [ ] `cargo clippy -- -D warnings` clean
