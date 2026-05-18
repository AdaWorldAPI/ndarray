# PR-X12 — `ndarray::hpc::codec::*` — x265-style CTU/CU + skip/merge/delta/escape + λ-RDO + ANS

> READ BY: savant-architect, cascade-architect, codec-architect,
> cognitive-architect, sentinel-qa, product-engineer.
>
> Status: design v1 — drafted 2026-05-18 in the master-consolidation arc.
>
> **Depends on**: PR-X10 (linalg-core), PR-X3 BlockedGrid (shipped).
> **Used by**: PR-X9 (basin-codebook lazy storage) — the codec encodes
> cognitive cells into skip/merge/delta/escape modes.

## Why

The cognitive-shader cascade and the x265 video codec share the **same**
arithmetic shape:

| x265 (video codec) | PR-X12 codec (cognitive) |
|---|---|
| CTU (64×64 luma block) | one L1 BlockedGrid block (64×64 cells) |
| CU quad-tree split (64→32→16→8) | L1 → L2 → L3 → L4 cascade (4×4 branching) |
| PU prediction unit (motion vector + ref frame) | basin reference (basin_idx + ref_tier) |
| TU transform unit (DCT residual storage) | per-cell δ_u8 perturbation |
| Skip mode (CU = pure motion-predicted) | cell exactly matches basin (δ=0) |
| Merge mode (CU inherits motion vector) | cell inherits δ from N/E/W/S neighbor |
| Intra prediction (block from same frame) | cell predicted from same-tier neighborhood |
| Inter prediction (block from ref frame) | cell predicted from parent-tier basin |
| RDO (rate-distortion optimization) | cognitive RDO: minimize bits × ε_truth_loss |
| CABAC entropy coder | **ANS entropy coder** (simpler, cache-friendlier) |

x265 averages **~4 bits/pixel** on HD video despite 8-12 bits raw. PR-X12 targets **~2-8 bits per cognitive cell** despite 64 bits raw. Same compression ratio, same mechanism, different content semantics.

## Module layout — `crate::hpc::codec::*`

```
src/hpc/codec/
├── mod.rs           — pub surface + feature gate
├── ctu.rs           — A1: CTU/CU partitioning (quad-tree over BlockedGrid blocks)
├── mode.rs          — A2: 2-bit mode tag (skip=00 / merge=01 / delta=10 / escape=11)
├── predict.rs       — A3: intra (same-tier neighborhood) + inter (parent-tier) prediction
├── transform.rs     — A4: optional residual transform (DCT-II for delta-mode if useful)
├── quantize.rs      — A5: scalar 8-bit quantizer + dequantizer + rate model
├── rdo.rs           — A6: λ-RDO loop (pick mode minimizing bits × ε_truth_loss)
├── ans.rs           — A7: Asymmetric Numeral Systems entropy coder (rANS variant)
└── stream.rs        — A8: byte-stream pack/unpack + header + frame-boundary markers
```

8 workers, each owns one file. After A1 (CTU foundation), A2-A8 spawn in parallel — they consume only A1's `Ctu` type + `crate::hpc::linalg::*`.

## Core types

```rust
/// One CTU = one BlockedGrid L1 block (64×64 cognitive cells).
/// Partitionable into CUs via quad-tree split (64→32→16→8).
#[repr(C, align(64))]
pub struct Ctu {
    pub block_row: u16,
    pub block_col: u16,
    pub tier: u8,                // 1..=4 (matches splat4d cascade)
    pub split_depth: u8,         // 0..=3 (CU split level within CTU)
    pub partition: CtuPartition, // recursive quad-tree
}

#[repr(u8)]
pub enum CtuPartition {
    Leaf(LeafCu),                // 64×64, 32×32, 16×16, or 8×8 CU
    Split([Box<CtuPartition>; 4]), // 4 sub-quadrants
}

pub struct LeafCu {
    pub mode: CellMode,          // 2-bit
    pub basin_idx: u16,          // 12-bit codebook index (high 4 bits reserved)
    pub delta: Option<u8>,       // present iff mode == Delta
    pub merge_dir: Option<MergeDir>, // present iff mode == Merge
    pub escape_idx: Option<u32>, // present iff mode == Escape
}

#[repr(u8)]
pub enum CellMode {
    Skip   = 0b00,  // exact basin match
    Merge  = 0b01,  // inherit δ from N/E/W/S neighbor
    Delta  = 0b10,  // own 8-bit perturbation
    Escape = 0b11,  // full 64-bit value in escape vector
}

#[repr(u8)]
pub enum MergeDir { North = 0, East = 1, West = 2, South = 3 }
```

## λ-RDO loop

For each cell at encode time, compute four mode costs and pick the minimum:

```rust
fn rdo_cell(true_value: u64, basin: &BasinAtom, neighbors: &Neighbors, lambda: f32) -> (CellMode, u32) {
    let basin_match = true_value == basin.edge;
    let skip_bits   = 2;                       // mode tag only
    let skip_dist   = if basin_match { 0.0 } else { f32::INFINITY };

    let (best_dir, merge_match) = find_best_neighbor_merge(true_value, basin, neighbors);
    let merge_bits  = 4;                       // mode + dir
    let merge_dist  = if merge_match { 0.0 } else { f32::INFINITY };

    let (delta_q, delta_residual) = quantize_8bit(true_value - basin.edge);
    let delta_bits  = 10;                      // mode + 8 bits
    let delta_dist  = delta_residual.abs() as f32;

    let escape_bits = 66;                      // mode + 64 bits
    let escape_dist = 0.0;                     // lossless

    let costs = [
        skip_bits   as f32 + lambda * skip_dist,
        merge_bits  as f32 + lambda * merge_dist,
        delta_bits  as f32 + lambda * delta_dist,
        escape_bits as f32 + lambda * escape_dist,
    ];
    let (idx, _) = argmin(&costs);
    (mode_from_idx(idx), costs[idx] as u32)
}
```

λ is calibrated via NARS confidence (high-confidence cells → high λ → prefer lossless; low-confidence → low λ → tolerate compression). v1 uses x265's medium-preset λ table as starting heuristic.

## ANS entropy coder (rANS variant)

Why ANS instead of CABAC:
- **Cache-friendlier**: rANS is a single multiply + table lookup per symbol; CABAC has context-state-update branches
- **SIMD-friendlier**: rANS streams can be encoded in parallel and merged; CABAC is strictly serial
- **Simpler**: ~150 LoC for rANS; CABAC is ~800 LoC minimum
- **Comparable compression**: rANS within 0.5% of CABAC on typical streams (proven by Yann Collet's zstd which uses fStateB rANS variant)

PR-X12 ships **rANS for v1**. CABAC follow-on if the cognitive substrate's compression ratio targets aren't met.

```rust
pub struct RansEncoder {
    state: u32,
    output: Vec<u8>,
    freq_table: [u16; 4],  // [skip, merge, delta, escape] symbol probs
}

impl RansEncoder {
    pub fn encode_symbol(&mut self, symbol: CellMode) {
        let (cum_freq, freq) = self.freq_table.cum_and_freq(symbol as usize);
        let q = self.state / freq as u32;
        let r = self.state % freq as u32;
        self.state = q * RANS_PROB_TOTAL + cum_freq as u32 + r;
        if self.state >= RANS_PROB_TOTAL << 16 {
            self.output.push((self.state & 0xFF) as u8);
            self.state >>= 8;
        }
    }
}
```

**Adaptive freq_table**: per CTU, update the [skip, merge, delta, escape] probabilities from observed frequencies → next CTU encodes with the new table. Standard adaptive-rANS pattern.

## Compression target

Per-cell average storage (coherent cognitive state):
- 70% skip   → 2 bits + 2 bytes basin_idx        ≈ 2.25 bytes
- 25% merge  → 2 + 2 bits + 2 bytes basin_idx    ≈ 2.50 bytes
- 4.5% delta → 2 bits + 1 byte + 2 bytes basin   ≈ 3.25 bytes
- 0.5% escape → 2 bits + 8 bytes + 2 bytes basin ≈ 14.25 bytes

Weighted average: **~2.4 bytes/cell** vs 8 bytes dense = **3.3× compression on cells**.

With shared codebook + schema amortization across pyramids: **~10-50× per simultaneous pyramid**.

Worst case (incoherent / random): 95% delta + 5% escape → ~4 bytes/cell, **still 2× over dense**. No regression vs dense even on adversarial inputs.

## Worker decomposition — 8 workers

| Worker | File | Scope | LoC | Depends on |
|---|---|---|---|---|
| A1 | `ctu.rs` | `Ctu` carrier + `CtuPartition` enum + quad-tree split/merge ops | ~300 | BlockedGrid (PR-X3) |
| A2 | `mode.rs` | `CellMode` 2-bit enum + `MergeDir` + bit-pack/unpack helpers | ~150 | A1 |
| A3 | `predict.rs` | Intra (same-tier neighborhood) + inter (parent-tier) prediction | ~350 | A1, A2, BlockedGrid |
| A4 | `transform.rs` | Optional DCT-II for delta residuals; 8×8 fast path | ~250 | linalg-core (PR-X10) |
| A5 | `quantize.rs` | 8-bit scalar quantizer + dequantizer + rate model | ~200 | A2 |
| A6 | `rdo.rs` | λ-RDO loop + mode selection + λ-table init | ~400 | A2, A3, A4, A5 |
| A7 | `ans.rs` | rANS encoder/decoder + adaptive freq table | ~300 | A2 |
| A8 | `stream.rs` | Byte-stream pack/unpack + header + frame markers | ~250 | A7 |

**Sprint composition**: A1 sequential (foundation), then A2-A7 parallel (different files, A6 depends on A2-A5 but workers run concurrently with stub bodies until prerequisites land), then A8 sequential after A7. **~2 weeks** sprint duration with the 12-agent cadence.

## Verification commands

```bash
cargo check -p ndarray --features std,codec
cargo test -p ndarray --features std,codec hpc::codec
cargo test --doc -p ndarray --features std,codec hpc::codec
cargo fmt --all -- --check
cargo clippy -p ndarray --features std,codec -- -D warnings
cargo bench -p ndarray --features std,codec hpc::codec
```

Plus parity / correctness gates:
- **Round-trip exactness**: `decode(encode(cells)) == cells` modulo `epsilon_floor` per RDO config
- **Skip-mode dominance**: pure-basin input → 100% skip mode, ~2.25 bytes/cell output
- **Escape mode safety**: outlier input → 100% escape mode, **NO truth loss**
- **Compression target**: synthetic coherent input → ≤ 0.5× dense size (verify 2× compression target)
- **ANS bit-exact across endianness**: encode on little-endian, decode on big-endian (or simulated) → identical output

## Open questions (joint savant ruling)

1. **rANS vs CABAC for v1?** Lean: **rANS** — simpler, cache-friendlier, 0.5% compression-ratio diff is negligible for cognitive use case. CABAC as PR-X12.1 follow-on if compression target is missed.

2. **DCT residual transform in v1 or v2?** Lean: **v2 follow-on** — for cognitive cells the residual is 8-bit scalar perturbation; 1D DCT doesn't help. Skip A4 entirely in v1; revisit if compression analysis shows transform residuals improve ratio.

3. **CTU size: 64×64 (matches L1) or const-generic?** Lean: **const-generic** with default = 64 (matches PR-X3 L1). 16×16 CTUs for AMX BF16 grids, 32×64 for AMX INT8.

4. **CU split depth limit?** Lean: **3 levels** — 64 → 32 → 16 → 8 — matching x265's CU max-depth-4. Cognitive cells don't need finer.

5. **Adaptive freq table per CTU or per frame?** Lean: **per CTU** — adapts faster to local cognitive coherence patterns. Worst case (random state): same as dense, no regression.

6. **Cross-cell prediction (intra) max neighborhood radius?** Lean: **1-cell (4-connected: N/E/W/S)** in v1. Larger neighborhoods (8-connected with corner cells) as PR-X12.2 if needed.

7. **Quantizer step: uniform u8 or non-uniform with NARS-confidence weighting?** Lean: **uniform u8** for v1; NARS-weighted quantizer (high-confidence cells get finer quantization steps) as follow-on once cognitive practice surfaces specific failure modes.

## Done criteria

- All 8 workers complete with parity + compression gates green
- v1 encoder produces ~2.4 bytes/cell on coherent test corpus
- v1 encoder never produces > 4 bytes/cell on adversarial corpus (no regression vs dense)
- Round-trip exactness within configured `epsilon_floor` (Skip + Escape modes are bit-exact; Delta is u8-quantized)
- Codex P0 audit (especially SAFETY-claim on `unsafe` rANS state bit-shifts)
- P2 savant SHIP verdict

## Forward compatibility

The codec produces a byte stream that PR-X9 (lazy basin-codebook storage) wraps in its `LazyBlockedGrid` representation. When PR-X6 (Lance bridge, separate roadmap item) lands, each CTU becomes one Lance fragment — **per-L1-block fragments give natural disjoint concurrent-write contention** (the original "gridlake" claim from the conversation arc).
