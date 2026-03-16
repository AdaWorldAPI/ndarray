# BF16 Truth Encoding + Causality

## BF16-Structured Hamming

BF16 = 16 bits per dimension: sign(1) + exponent(8) + mantissa(7)

XOR + per-field weighted popcount:
```
sign bit 15:       weight 256 (causality direction)
exponent bits 7-14: weight 16 per flipped bit (confidence scale)
mantissa bits 0-6:  weight 1 per flipped bit (finest distance)
```

### BF16Weights
```rust
pub struct BF16Weights {
    pub sign: u16,      // default 256
    pub exponent: u16,  // default 16
    pub mantissa: u16,  // default 1
}
```
Validation: `sign + 8×exponent + 7×mantissa ≤ 65535` (u16 lane overflow)

## Awareness Substrate (4 states)

Per-dimension classification from BF16 comparison:

```
Crystallized — sign agrees, exponent stable, mantissa clean
Tensioned    — sign agrees, but exponent or mantissa diverges
Uncertain    — sign disagrees (fundamental ambiguity)
Noise        — high mantissa entropy (no useful signal)
```

### SuperpositionState
```rust
pub struct SuperpositionState {
    pub n_dims: usize,
    pub states: Vec<AwarenessState>,
    pub crystallized_pct: f32,
    pub tensioned_pct: f32,
    pub uncertain_pct: f32,
    pub noise_pct: f32,
}
```

## NARS Truth Values

Mapping from awareness substrate:
```
frequency  = crystallized% (settled positive evidence)
confidence = 1 - noise% (meaningful signal)
```

### NarsTruthValue
```rust
pub struct NarsTruthValue {
    pub frequency: f32,   // [0.0, 1.0]
    pub confidence: f32,  // [0.0, 1.0]
}
```

- `expectation() = f × c + 0.5 × (1 - c)`
- `ignorance() = <0.5, 0.0>` (no evidence)
- `from_awareness(SuperpositionState)` — extract truth

## Causality Direction

```rust
pub enum CausalityDirection {
    Causing,       // RGB: emitting (warmth < 0, social < 0, sacredness < 0)
    Experiencing,  // CMYK: absorbing (warmth ≥ 0, social ≥ 0, sacredness ≥ 0)
}
```

Causality dims: warmth(4), social(6), sacredness(8)
Detection: majority vote — if 2+ are negative → Causing

## PackedQualia

```rust
pub struct PackedQualia {
    pub resonance: [i8; 16],  // 16 phenomenological dimensions
    pub scalar: [u8; 2],      // BF16 scalar (intensity)
}
```

Total: 18 bytes per qualia point.

## SPO Projections (2³)

7 non-null mask projections over Node's S/P/O planes:
```
S__, _P_, __O, SP_, S_O, _PO, SPO
```

Each yields a different truth value. The full 8-term factorization
gives Pearl Rung 1-3 causal decomposition as a byproduct of
every similarity computation.
