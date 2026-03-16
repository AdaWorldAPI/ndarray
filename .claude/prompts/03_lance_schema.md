# Three-Plane Lance Schema

> Transcoded from rustynum `18_three_plane_lance_schema.md`.
> Adapted for ndarray's `hpc/arrow_bridge.rs` types.

## Goal

Dual-layer Lance schema for three-plane (S/P/O) concept storage, using
ndarray's arrow bridge types for zero-copy interop.

## Schema: `bind_nodes_v2`

### Per-Plane Architecture (×3: subject, predicate, object)

| Column | Type | Bytes | Description |
|--------|------|-------|-------------|
| `{role}_binary` | FixedSizeBinary(2048) | 2048 | Crystallized 16384-bit fingerprint |
| `{role}_soaking` | FixedSizeList(Int8, 10000) | 10000 | Nullable int8 accumulator |

### Composite Columns

| Column | Type | Description |
|--------|------|-------------|
| `spo_binary` | FixedSizeBinary(2048) | XOR of S⊕P⊕O binary |
| `sigma_mask` | FixedSizeBinary(1250) | 10000-bit attention σ-mask |
| `nars_frequency` | UInt16 | NARS crystallized% |
| `nars_confidence` | UInt16 | NARS 1 - noise% |
| `gate_state` | UInt8 | FORM=0, FLOW=1, FREEZE=2 |
| `role_provenance` | Utf8 | Source identifier |

### Row Budget
- ~37KB per row uncompressed
- ~800MB for full 65K BindSpace with Lance compression

## Bridge Types (in `hpc/arrow_bridge.rs`)

| Type | Role |
|------|------|
| `ThreePlaneFingerprintBuffer` | Holds S/P/O binary fingerprints |
| `SoakingBuffer` | Holds nullable int8 soaking vectors |
| `PlaneBuffer` | Single-plane binary buffer |
| `GateState` | Form/Flow/Freeze enum |

## Soaking Lifecycle

```
FORM gate (accumulating)
    │ soaking columns: non-null, actively written
    │
    ▼ crystallize: sign(soaking) → binary
FLOW gate (serving)
    │ soaking columns: set to null (free memory)
    │ binary columns: immutable
    │
    ▼ optional
FREEZE gate (archived)
    │ entire row compressed
```

## Zero-Copy Flow

```
Arrow RecordBatch
    │
    ▼ zero-copy view
ThreePlaneFingerprintBuffer
    │
    ▼ as_bytes()
Fingerprint<256>::from_bytes()
    │
    ▼ hamming_distance()
bitwise::hamming_distance_raw()  ← VPOPCNTDQ
```

## Migration V1 → V2
- Additive: new columns alongside existing schema
- Existing single fingerprint → copied to all three planes initially
- No rewrite needed; old readers ignore new columns

## Verification
- Arrow bridge types compile with schema constants
- `GateState` enum matches gate lifecycle
- `ThreePlaneFingerprintBuffer` holds 3 × 2048 bytes
- `SoakingBuffer` nullable semantics work correctly
