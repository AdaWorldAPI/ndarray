# Plane / Node / Seal — Type Specifications

## Plane

The core cognitive type. One dimension of cognition.

```rust
#[repr(C, align(64))]
pub struct Acc16K {
    pub values: [i8; 16384],  // 16KB, L1 cache resident
}

pub struct Plane {
    acc: Box<Acc16K>,          // Raw i8 accumulator (ONLY stored state)
    bits: Fingerprint<256>,    // Cached sign(acc)
    alpha: Fingerprint<256>,   // Cached |acc| > threshold
    dirty: bool,               // Cache invalidation flag
    encounters: u32,           // Evidence count
}
```

### Constants
- `PLANE_BITS = 16384`
- `PLANE_BYTES = 2048` (fingerprint view)
- `CONTAINER_BYTES = 16384` (full accumulator)

### Key Methods
- `encounter_bits(evidence: &Fingerprint<256>)` — saturating i8 accumulation
- `encounter(text: &str)` — blake3-expanded text → fingerprint → accumulate
- `distance(&mut self, other: &mut Plane) -> Distance` — alpha-masked XOR+popcount
- `truth(&mut self) -> Truth` — NARS truth from accumulator state
- `merkle(&mut self) -> MerkleRoot` — blake3 truncated to 48 bits
- `verify(&mut self, stored: &MerkleRoot) -> Seal` — integrity check

### Alpha Threshold (adaptive)
```
encounters 0-1:  threshold = 0
encounters 2-5:  threshold = encounters / 2
encounters 6-20: threshold = encounters * 2 / 5
encounters 20+:  threshold = (isqrt(encounters) * 4 / 5).min(127)
```

## Distance (enum, not float)

```rust
pub enum Distance {
    Measured { disagreement: u32, overlap: u32, penalty: u32 },
    Incomparable,
}
```

- `normalized() -> Option<f32>` — ONLY place float appears
- `closer_than(max_disagreement: u32) -> bool` — pure integer
- `raw() -> Option<u32>` — raw disagreement count

## Truth (integer NARS)

```rust
pub struct Truth {
    pub frequency: u16,   // 0-65535 scaled [0.0, 1.0]
    pub confidence: u16,  // 0-65535 scaled [0.0, 1.0]
    pub evidence: u32,    // encounter count
}
```

- `expectation()` — c * (f - 0.5) + 0.5, integer only
- `revision(other)` — evidence-weighted combination

## Node

Three Planes = one cognitive atom (Subject/Predicate/Object).

```rust
pub struct Node {
    pub s: Plane,
    pub p: Plane,
    pub o: Plane,
}
```

### Mask

```rust
pub struct Mask { pub s: bool, pub p: bool, pub o: bool }
// 8 constants: SPO, SP_, S_O, _PO, S__, _P_, __O, ___
```

## Seal

```rust
pub enum Seal { Wisdom, Staunen }
pub struct MerkleRoot(pub [u8; 6]);  // 48-bit truncated blake3
```

## Fingerprint<N>

```rust
pub struct Fingerprint<const N: usize> {
    pub words: [u64; N],
}
```

- `Fingerprint<256>` = 2048 bytes = 16384 bits
- XOR group: `BitXor`, `BitAnd`, `BitOr`, `Not`
- `hamming_distance()` — delegates to SIMD
- `popcount()` — delegates to SIMD
- `as_bytes()` — zero-copy `&[u8]` view
