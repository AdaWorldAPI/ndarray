# Cascade (Belichtungsmesser) Search Architecture

## Overview

HDR (High Dynamic Range) 3-stroke adaptive cascade for Hamming-based
nearest-neighbour search with optional precision tiers.

## Types

### RankedHit
```rust
pub struct RankedHit {
    pub index: usize,    // Database index
    pub hamming: u64,    // Exact Hamming distance
    pub precise: f64,    // High-precision score (f64::NAN if Off)
    pub band: Band,      // Quality classification
}
```

### Band (5 quality tiers)
```
Foveal  — top 5%    — distance ≤ threshold × 0.25
Near    — 5-25%     — distance ≤ threshold × 0.50
Good    — 25-60%    — distance ≤ threshold × 0.75
Weak    — 60-90%    — distance ≤ threshold
Reject  — beyond threshold
```

### PreciseMode (6 data paths)
```
Off        — Hamming only
Vnni       — VNNI dot_i8 cosine
F32        — dequant → f32 dot
BF16       — dequant → bf16 dot (future VDPBF16PS)
DeltaXor   — blended hamming + INT8 cosine
BF16Hamming — weighted per-field XOR popcount
```

## 3-Stroke Pipeline

### Stroke 1: Partial popcount with σ warmup
- Sample 1/16 of vector bytes (minimum 64B)
- Warmup: 128 candidates → estimate σ
- Reject threshold: `threshold + 3σ`
- ~84% rejection rate

### Stroke 2: Full Hamming on survivors
- Incremental: compute only remaining bytes
- `d_full = d_prefix + d_rest`
- Reject if `d_full > threshold`

### Stroke 3: High-precision (optional)
- Only for finalists that passed Stroke 2
- Mode-dependent: VNNI, F32, BF16, DeltaXor, BF16Hamming
- Sort by precise distance descending

## Cascade (stateful search engine)

```rust
pub struct Cascade {
    pub threshold: u64,
    pub vec_bytes: usize,
    mu: f64,           // Running mean
    sigma: f64,        // Running std
    observations: usize,
}
```

### Methods
- `from_threshold(threshold, vec_bytes)` — fixed threshold
- `calibrate(distances, vec_bytes)` — mu + 3σ from sample
- `expose(distance) -> Band` — classify into quality band
- `observe(distance) -> Option<ShiftAlert>` — Welford's online update
- `recalibrate(alert)` — update threshold from drift
- `query(query, database, ...) -> Vec<RankedHit>` — full cascade

## PackedDatabase

Stroke-aligned layout for cache efficiency:
```
stroke1: [candidate0_bytes[0..128], candidate1_bytes[0..128], ...]
stroke2: [candidate0_bytes[128..512], candidate1_bytes[128..512], ...]
stroke3: [candidate0_bytes[512..2048], candidate1_bytes[512..2048], ...]
```

### Stroke Sizes
- Stroke 1: 128 bytes (coarse)
- Stroke 2: 384 bytes (medium)
- Stroke 3: 1536 bytes (precise)
- Total: 2048 bytes per fingerprint

## Shift Detection

Welford's online algorithm detects distribution drift:
- Triggers when `|new_mu - old_mu| > 2 × old_sigma`
- Returns `ShiftAlert { old_mu, new_mu, old_sigma, new_sigma, observations }`
