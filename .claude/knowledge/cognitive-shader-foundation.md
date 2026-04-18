# Cognitive Shader Foundation — ndarray's Role in the 7-Layer Stack

> READ BY: all ndarray agents (arm-neon-specialist, cascade-architect,
> cognitive-architect, l3-strategist, migration-tracker, product-engineer,
> savant-architect, sentinel-qa, truth-architect, vector-synthesis)
>
> Parallel doc in lance-graph: `.claude/knowledge/cognitive-shader-architecture.md`

## ndarray's Role: Layer 0 + parts of Layer 1

ndarray is the HARDWARE FOUNDATION of the cognitive shader stack.
It provides the primitives that Layer 1 (BindSpace columns) and higher
layers build on. ndarray never depends on lance-graph or any cognitive
crate — the dependency is one-way.

```
Layer 6: LanceDB (cold persistence)         ← lance-graph
Layer 5: GPU/APU meta ops (optional)        ← future
Layer 4: Planner strategies (16-19)         ← lance-graph-planner
Layer 3: CollapseGate write protocol        ← ndarray (enum) + contract
Layer 2: CognitiveShader dispatch           ← p64-bridge
Layer 1: BindSpace columns + multi-lane     ← ndarray + contract
Layer 0: SIMD primitives                    ← ndarray (THIS CRATE)
```

## Public Surface: `ndarray::simd::*`

All consumers import from `ndarray::simd::*`, NOT from `ndarray::hpc::*`.
The hpc/ paths are private implementation detail. The simd/ module is the
stable public API.

Types that MUST be in `ndarray::simd::*`:
- `F32x16, F64x8, U8x64, F16x32, U64x8, I16x32, I8x64`
- `Fingerprint<N>` — const-generic, N×64 bits
- `MultiLaneColumn<T>` — same bytes, multiple SIMD lane views
- `array_window(data, N)` — aligned batch iterator
- `VectorWidth, vector_config()` — the LazyLock width singleton
- `CollapseGate` — Flow/Block/Hold enum (exists in hpc/bnn_cross_plane)

If a type isn't in `ndarray::simd::*`, consumers can't use it.
Keeps our API surface small. Internal refactors in hpc/ don't break
downstream.

## What Layer 0 Provides

### SIMD Primitives (hardware abstraction)
- Runtime dispatch: `simd_caps()` frozen singleton
- AVX-512 (F32x16, VPOPCNTDQ, VPGATHERDD)
- AVX2 + FMA
- NEON (A53 / A72 / A76 dotprod tiers)
- AMX via `asm!(".byte ...")` — TDPBF16PS, TDPBUSD
- F16C hardware conversion
- BF16 bit-exact RNE matching VCVTNEPS2BF16

### Fingerprint<N> — the BindSpace atom
- `[u64; N]` backing, 64-byte aligned
- `get/set/toggle_bit`, `bind` (XOR), `and`, `not`
- `hamming_distance` via SIMD popcount
- `popcount`, `density`
- `random` (xorshift128+), `from_content` (hash expansion)
- `permute` (circular bit shift for sequence encoding)

### MultiLaneColumn — same object, multiple SIMD widths
- One `Arc<[u8]>` backing store
- View as U8x64 / F16x32 / F32x16 / F64x8 without copy
- Consumer picks lane width per operation

### array_window — SIMD batch iterator
- Yields N-aligned chunks from a slice
- Zero-copy: window IS a `&[T]` view
- One cascade level = one array_window pattern

### CollapseGate enum
- `Flow` / `Block` / `Hold` (already in `hpc/bnn_cross_plane`)
- Consumers (L3) extend with MergeMode (Xor/Bundle/Superposition)

## What ndarray DOES NOT Provide

These live UP the stack, not in ndarray:
- BindSpace address types (lance-graph-contract)
- CognitiveShader dispatch (p64-bridge)
- Planner strategies (lance-graph-planner)
- CausalEdge64 (causal-edge)
- NARS inference (causal-edge + contract)
- GGUF parsing (bgz-tensor / consumer)

Keep ndarray free of cognitive logic. It's the foundation, not the cortex.

## Current Gaps (next session targets)

1. **MultiLaneColumn type doesn't exist yet** — add to `src/hpc/column.rs`,
   re-export from `src/simd.rs`
2. **Fingerprint<N> missing `as_u8x64()`** — add SIMD view methods
3. **simd.rs re-exports incomplete** — add Fingerprint, MultiLaneColumn,
   array_window, VectorWidth
4. **VectorWidth LazyLock not consumed** — any module that serializes
   fingerprints should read it for width config
5. **Hamming popcount hasn't been exposed via multi-lane view** —
   combine with MultiLaneColumn for the Layer 1 cascade path

## Migration Tracking (from ladybug-rs)

ladybug-rs depended on `rustynum` as its HPC crate. rustynum was
ported INTO this ndarray fork as `src/hpc/` (55 modules, 880 tests).
Downstream consumers (lance-graph-cognitive, learning crate) still
reference `rustynum_core::*` types. They need these substitutions:

| ladybug-rs `rustynum_core::*` | ndarray equivalent |
|---|---|
| `Fingerprint` | `ndarray::simd::Fingerprint<256>` |
| `hamming_distance` | `ndarray::hpc::bitwise::hamming_distance_raw` |
| `simd_level` | `ndarray::hpc::simd_caps::simd_caps()` |
| `cascade::*` | `ndarray::hpc::cascade::*` |
| `bf16_*` | `ndarray::hpc::quantized::BF16` |
| `rustynum_bnn::CollapseGate` | `ndarray::hpc::bnn_cross_plane::CollapseGate` |
| `rustynum_holo::*` | `ndarray::hpc::holo::*` |

**migration-tracker agent** owns this substitution table.

## The Endgame (ndarray's view)

Each token of LLM inference in the cognitive shader system runs:

```
1. Read BindSpace column slice                → &[u64; N]  (Layer 1)
2. Hamming popcount via SIMD dispatch         → [u32; N]   (Layer 0)
3. Base17 L1 distance on survivors            → [u16; M]   (Layer 0)
4. Palette table lookup (256×256)             → [u8; K]    (Layer 0)
5. Gather via VPGATHERDD                      → [u8; K]    (Layer 0)
```

All Layer 0. All ndarray. Zero FP. Zero matmul. 611M lookups/sec.

The cognitive layers above coordinate WHICH columns to scan and HOW
to combine results. ndarray just executes the primitives as fast as
the hardware allows. Pi Zero to Sapphire Rapids, same API, same
correctness, different throughput.
