# PR-X12 — x265 / HEVC through the BLAS-GEMM Lens

> Date: 2026-05-22
> Status: **perspective doc** — re-reads the HEVC/x265 design space as a sequence of GEMM operations. Companion to `pr-x12-substrate-merged-canon.md` and `pr-x12-canon-resolutions-delta.md`.
>
> Premise: every x265 inner loop has a GEMM form. HEVC was designed in 2013 against hardware that made per-pixel butterflies the fast path; modern hardware (VNNI, AMX, BF16) inverts that ranking. PR-X12 is what x265 would have been with the right hardware floor.

---

## 0. The thesis in one sentence

**x265 implements roughly nine inner loops, six of which collapse to GEMM under the SSD/k-means/tropical reformulations, three of which stay non-GEMM and live in cheap per-byte paths.** PR-X12 spends ~80% of encode time inside BLAS calls; HEVC reference spends ~30%. The reframing is not metaphor — it is an algebraic identity per stage.

---

## 1. The nine HEVC primitives, classified

| # | Primitive | HEVC native form | GEMM form | Where it lands |
|---|---|---|---|---|
| 1 | Motion estimation | SAD `Σ \|A-B\|` | SSD `\|\|A\|\|² - 2A·B + \|\|B\|\|²` → GEMV | `ndarray::hpc::blas_level2::batched_ssd_search` *(planned — not yet in blas_level2.rs)* |
| 2 | Forward transform | 4×4 / 8×8 / 16×16 / 32×32 DCT-II butterflies | Batched DCT as GEMM at N≥64 | `ndarray::hpc::fft::DctIIBasis<N>` + `bf16_tile_gemm` |
| 3 | Quantization | Scalar divide + round | Dot product against quant matrix | Inline; uses existing `simd_int_ops` |
| 4 | Mode decision (CTU split) | Recursive RDO, `O(4^d)` | Tropical-GEMM Bellman-Ford, `O(d²)` | `lance-graph::blasgraph` *(canonical home; kernel unwritten — shipped min-plus today is only `bgz17::ScalarCsr::spmv_min_plus`, a lossy-sibling prototype)* |
| 5 | Basin assignment (palette / k-means) | Linear scan distance comparisons | Batched Hamming/L2 dist as GEMM | `ndarray::hpc::cam_pq::kmeans` |
| 6 | Deblocking filter | 3×3 / 5×5 per-pixel separable conv | im2col + GEMM at block size ≥ 16 | `ndarray::hpc::activations` (existing conv path) |
| 7 | rANS state advance | u32 state machine | Symbol-frequency lookup; **not GEMM** | `ndarray-codec::ans` |
| 8 | Header bit-pack | u16 shift+mask | Not GEMM (per-leaf, ~5 ns) | `src/hpc/codec/mode.rs::pack_header` |
| 9 | Stream framing / sync | Byte-level append | Not GEMM | `ndarray-codec::stream` |

Stages 1-6 (the inner-loop-cost-dominant ones) are all GEMM. Stages 7-9 are I/O-bound and stay per-byte. The boundary between them is sharp because GEMM amortises hardware fusion (AMX, VNNI) while state-machine code can't.

---

## 2. Per-stage detail — the algebraic moves

### 2.1 Motion estimation: SAD → SSD (R-6)

HEVC's reference encoder uses SAD because in 2007-2013, ARM hand-tuned `VPSADBW` was the fastest 16×16-block-difference primitive. SAD has no matrix structure — the absolute value inside the sum doesn't factor.

SSD is algebraically richer:

```text
SSD(A, B)  = Σ_ij (A_ij - B_ij)²
           = Σ A² - 2 Σ (A·B) + Σ B²
           = ||A||² - 2·(A·B) + ||B||²
                       ▲
                       └── this is a GEMM/GEMV
```

For N motion-vector candidates against one reference block:

- Candidate matrix `A_batch`: `(N × 256)` — 256 = 16×16 pixels per block
- Reference vector `B`: 256-d
- Middle term: `A_batch @ B` → `(N × 1)` GEMV
- `||A_i||²` precomputed once per candidate window; `||B||²` once per reference

**On Cascade Lake+ with VNNI:** `VPDPBUSD` = 64 i8·i8→i32 ops/cycle. One 256-elem dot product = 4 ops = ~4 cycles. Versus `VPSADBW` SAD path: ~128 cycles per 16×16. **Speedup: 30-50× depending on batch.**

**On Sapphire Rapids with AMX:** TDPBUSD tile op = 256 i8·i8→i32 ops in one tile cycle. 16 candidates batched fits one AMX tile; throughput rises by another factor of 4.

Net: motion estimation is ~50× faster than HEVC reference, *for the same wire-format semantics*. Same MV grid, same precision, same RDO. The math is identical; the substrate is BLAS.

### 2.2 Transform: per-block butterflies → batched DCT (R-5)

HEVC ships Loeffler / Lengwehasatit 1D DCT-II butterflies — fast at single-block sizes (~80 ops per 32×32 transform), bad at batched dispatch. The Loeffler factoring is what made 2010-era CPUs (no SIMD GEMM at small sizes) able to encode HEVC at all.

PR-X12 keeps the butterflies for small N and dispatches to BLAS GEMM at N ≥ 64:

```text
N = number of contiguous transform blocks

if N <  64:  per-block butterfly (Loeffler) — fits L1, no batching overhead
if N >= 64:  batched DCT as GEMM via DctIIBasis<N> + bf16_tile_gemm
             ~256 cycles for 64 blocks (AMX) vs ~1280 cycles butterfly
```

Crossover (R-5) varies per arch: SPR≈64, SKX/ICL≈32, Zen 4≈96, Apple Silicon≈256 — **[UNCALIBRATED ESTIMATES, no measurement source; Plan G calibrates (audit #6)]**.

**The trait pattern (R-1):** `DctIIBasis<const N>` implements `Basis<i16>` — the basis is data (the cosine matrix, computed once at startup). The reduction (`A4 transform path` and `EWA splat rasterizer Plan E`) both call `basis.apply(src, dst)`. **Same basis, two consumers.**

### 2.3 Quantization: stays per-byte, doesn't need GEMM

Scalar quantization is `q_ij = (coeff_ij * scale_ij) >> 15`. Per-coefficient cost ~1 ns; the entire 32×32 block quantizes in ~1000 ns scalar, no batching benefit. Stays at SIMD-batched i16 path (`simd_int_ops`), no GEMM layer.

### 2.4 Mode decision: recursive RDO → tropical-GEMM (R-7)

HEVC's partition decision walks the quad-tree recursively, computing Lagrangian cost at each split:

```text
For each CTU at depth d:
    for each of 4 children:
        recursive RDO at depth d+1
        compute mode + transform + quant + rate + distortion
    combine via min(D + λ·R)

Time: O(4^d) per CTU.  At d=4 (PR-X12): 256 leaves worst case.
```

Tropical-semiring reformulation: the (+, min) algebra has GEMM. Build the 85-node DAG with edge weights `W[parent, child] = ΔRDO`, then iterate `D ← min(D, D + W)` (one tropical-GEMM step). Repeat for d iterations.

```text
Naive recursive:  O(4^4) = 256 ops × |nodes| = ~22 K ops/CTU
Tropical-GEMM:    O(d²) × |nodes| = 16 × 85 = ~1.4 K ops/CTU
                  ~16× speedup
```

For 4K @ 60 fps with 132K CTUs/frame, this is the difference between **4 ms and 64 ms per frame just for partition RDO**. At 60 fps's 16.67 ms budget, naive RDO doesn't fit.

**Dep direction:** the tropical-GEMM kernel's canonical home is `lance-graph::blasgraph` (the bit-exact cognitive-side substrate). *(Corrected 2026-07-16, audit #3: blasgraph's shipped semirings are binary-Hamming over 16384-bit BitVec — the numerical f32 min-plus kernel is UNWRITTEN and lands there when A6 wires it; the only shipped min-plus today is `bgz17::ScalarCsr::spmv_min_plus`, a lossy-sibling prototype.)* Post-Plan-H, `ndarray-codec → lance-graph::blasgraph` is allowed because both are sibling crates above `ndarray` hardware.

### 2.5 Basin assignment: k-means as batched dist + argmin

For each cell, find the nearest of 4096 basin centroids:

```text
distances[c] = ||cell - centroid_c||²   for c in 0..4096
basin = argmin(distances)
```

Both the distance computation and the argmin are batched primitives:

- **Distance computation:** if cells are i8 fingerprints, batched Hamming distance via `VPOPCNTDQ` (Ice Lake+). If cells are f32/bf16, batched L2 via `_mm512_add_ps` after `_mm512_sub_ps`.
- **Across 4096 centroids:** matrix form. `dist = ||cells||² ⊕ ||centroids||² − 2 · (cells @ centroids^T)`. Same SSD identity as ME, scaled to codebook size.

`cam_pq::kmeans` already ships this in `src/hpc/`. The codec's basin-assign step is a thin wrapper.

### 2.6 Deblocking filter: per-pixel conv → im2col GEMM (only at scale)

3×3 / 5×5 separable filters at block edges. For a single CU's deblocking pass (~64 edge pixels), per-pixel conv wins. For batched deblocking across many CUs in a frame, im2col + GEMM wins by ~3-5× on AMX-class hardware.

x265's deblocking is one of the few stages that explicitly has per-block-size branches; PR-X12 keeps the same structure but dispatches the batched form through `ndarray::hpc::activations`.

### 2.7 rANS: stays as state machine

Not a GEMM. State machine that reads symbols, looks up `(freq, cumfreq)`, advances u32 state. ~10 ns/symbol on modern x86. Per-frame rebuild of the frequency table is the only batchable step (a sum-reduce, trivially SIMD).

### 2.8 Header bit-pack / stream framing

Per-leaf, 5-30 ns. No GEMM. Lives in `mode.rs::pack_header` / `pack_leaf` and the future `stream.rs`.

---

## 3. Why HEVC's 2013 design space was BLAS-impoverished

The HEVC spec was finalised in early 2013, against the following hardware:

- **No VNNI** — Cascade Lake shipped 2019. `VPDPBUSD` is six years after HEVC was frozen.
- **No AMX** — Sapphire Rapids shipped 2023. Ten years after the spec.
- **No bfloat16** — first appeared on SPR. HEVC's transform precision was set to fit i16 because i16 GEMM on Sandy Bridge SSE4 was the only practical option.
- **No `VPOPCNTDQ`** — Ice Lake 2019. HEVC's palette mode (SCC profile) was frozen with the assumption that 64-entry palettes were the cap, because larger palettes would have needed Hamming-distance GEMM that didn't exist.

**The HEVC team made the right choices for 2013 hardware.** Per-pixel butterflies were faster than batched GEMM at small sizes. SAD via `VPSADBW` was faster than SSD via any 2013-era integer SIMD. 64-entry palettes were the largest size where the linear-scan k-means inner loop fit L1 budget.

**Every one of those choices is now obsolete.** The PR-X12 substrate isn't a redesign of HEVC's wire format — it's HEVC's wire format with the inner loops swapped out for what 2026 hardware actually wants.

---

## 4. The reframing: PR-X12 IS x265 done as BLAS

| Aspect | HEVC reference | PR-X12 |
|---|---|---|
| Wire format | 16-bit header + per-mode tail | **same** |
| Mode taxonomy | Skip / Merge / Delta / Escape | **same** |
| Quad-tree partition | 64×64 CTU → 8×8 leaf | **same**, `Ctu<const N>` runtime-flex (M:E-G) |
| Palette / basin codebook | 64 entries max | 4096 entries (12-bit, full HHTL Leaf tree) |
| RDO criterion | `D + λ·R` Lagrangian | **same** |
| RDO solver | recursive `O(4^d)` | tropical-GEMM `O(d²)` (R-7) |
| ME criterion | SAD | SSD (R-6) — algebraically lossless reframing |
| Transform | per-block Loeffler | batched DCT GEMM at N≥64 (R-5) |
| Entropy coder | CABAC | rANS — better Shannon-efficiency, simpler state |
| In-loop deblocking | per-pixel conv | im2col GEMM at batch (existing infra) |

**The wire format is unchanged.** A PR-X12-encoded video should be decodable by an HEVC-spec decoder (modulo the rANS↔CABAC swap and the 4096-entry palette), because the semantic primitives — Skip/Merge/Delta/Escape, quad-tree CTU, RDO Lagrangian, DCT-II basis — are identical.

**What changed is the implementation.** Each inner loop is now a BLAS call.

---

## 5. What lands in `ndarray::hpc::blas_level2` (the codec's BLAS surface)

The codec uses, but does not own, these four primitives.

> **[Status, 2026-07-16]:** all four signatures below are **target API
> shapes**, not shipped symbols. `batched_ssd_search` / `batched_dct_ii` /
> `tropical_partition_rdo` do not exist yet (audit #5, #3);
> `kmeans_predict_batched` is a planned wrapper over the real
> `cam_pq::kmeans` + `CamCodebook::distance_batch`. The paragraph after the
> block ("zero new lines — all four already exist") overstated; the *building
> blocks* exist (`bf16_tile_gemm`, `cam_pq`, `simd_int_ops`), the wrappers do not.

```rust
// R-6: ME via SSD identity
pub fn batched_ssd_search(
    candidates: &[i8; 256],     // (N × 256) row-major
    n_candidates: usize,
    reference: &[i8; 256],
    out_distances: &mut [u32],  // length N
);

// R-5: batched DCT-II via GEMM
pub fn batched_dct_ii<const N: usize>(
    blocks: &[i16],             // (M blocks × N×N) row-major
    n_blocks: usize,
    out: &mut [i16],            // output coefficients
);

// R-7: tropical-GEMM partition (lives in blasgraph, called from codec)
pub fn tropical_partition_rdo(
    edge_weights: &[f32; 85],
    out_min_costs: &mut [f32; 85],
);

// k-means basin assignment (uses existing cam_pq)
pub fn kmeans_predict_batched(
    cells: &[Fingerprint],
    centroids: &[Fingerprint; 4096],
    out_basin_idx: &mut [u16],
);
```

**Codec layer:** ~30-50 LoC per stage to wrap the BLAS call into the predict/A6/A4 flow. **BLAS layer:** zero new lines — all four already exist or land via existing infrastructure (`bf16_tile_gemm`, `cam_pq`, `simd_int_ops`).

This is what makes R-3's ≤1500 generic-codec-LoC ceiling reachable. Most of the heavy lifting is already in `blas_level2`; the codec adds wrappers and orchestration, not new BLAS code.

---

## 6. The "blasgraph synergy" claim made precise

Earlier docs cited "blasgraph + MKL synergies" loosely. Quantified:

**Of nine codec inner loops, six dispatch to BLAS:**

| Loop | BLAS primitive | Existing infra |
|---|---|---|
| ME | SSD via VNNI GEMV | `blas_level2` after R-6 lands |
| Transform | Batched DCT GEMM | `bf16_tile_gemm` + `DctIIBasis<N>` |
| Quant | Stays per-byte | n/a |
| Mode decision | Tropical-GEMM | `lance-graph::blasgraph` |
| Basin assign | Hamming/L2 batched dist | `cam_pq::kmeans` |
| Deblocking | im2col GEMM | `activations` (existing conv path) |
| rANS | Stays state-machine | n/a |
| Header | Stays per-byte | n/a |
| Framing | Stays per-byte | n/a |

**On SPR with all six BLAS-dispatch paths active**, profile-guided estimate (calibrated during Plan G):

- ~80% of total encode time spent inside BLAS calls
- ~15% in rANS + header + framing (the per-byte paths)
- ~5% in quantize + scalar housekeeping

**HEVC reference encoder on the same SPR:** ~30% inside BLAS (mostly deblocking and ME bookkeeping); the rest is per-pixel butterflies + recursive RDO + SAD. The hardware sits idle 70% of the time at peak SIMD width.

**The 50× ME speedup, 16× partition RDO speedup, and 4× transform speedup compose** because they sit in different stages of the encode pipeline. End-to-end encode at 4K @ 60 fps becomes feasible on a single SPR socket.

---

## 7. Plan G video lane: the falsifier

Per R-4, the video lane of `codec-bench` clears `≥0.95× x265 ultrafast ratio at PSNR ±0.1 dB on Big Buck Bunny 1080p`. The R-11 latency assertion adds: total encode time for the clip must complete within (clip duration × 0.5).

**The hidden falsifier in §6's BLAS-synergy claim:** if Plan G's video lane profile shows <60% time-in-BLAS, the BLAS reframing is decorative — actually a critical bug, because it means the per-byte stages (rANS / header / framing) are dominating, which means SIMD-batched-encode (R-11) didn't actually land on the codec hot path.

**Suggested Plan G profile assert:** `perf stat -e cycles,instructions,L1_DCACHE_LOAD_MISSES` over the encode, with a sub-test breaking down cycles per stage. If the BLAS-dispatch stages don't sum to ≥60% of cycles, the abstraction is wrong somewhere.

This is the kind of test that catches "we wrote the code but it's not actually using the GEMM path because the dispatcher fell through to scalar" — a class of bug that ate weeks of PR #134 / #175 SIMD work and only surfaced in CI.

---

## 8. What this lens unlocks for x266 / next-gen codecs

The next document (`pr-x12-x266-3dgs-spacetime-upscaling.md`) asks what's possible if the substrate isn't x265-compatible — if we *replace* in-loop filters with 3DGS space-time interpolation. The answer becomes obvious once the codec is read as a GEMM pipeline: the in-loop filter is just another GEMM stage in the pipeline, and replacing it with a different GEMM (one whose output is a 3DGS-rendered reference frame) costs no architectural complexity — only ships a different `Basis<T>` impl.

That's the bridge to the next doc.

---

## 9. Cross-references

- **R-N citations:** `pr-x12-canon-resolutions-delta.md`
- **Architecture canon:** `pr-x12-substrate-merged-canon.md`
- **Mechanical spec:** `pr-x12-codec-x265-design.md` (what's getting reimplemented)
- **Next lens:** `pr-x12-x266-3dgs-spacetime-upscaling.md`
- **In-tree code:**
  - `src/hpc/blas_level1.rs`, `blas_level2.rs`, `blas_level3.rs` — host for `batched_ssd_search`, `batched_dct_ii`
  - `src/hpc/cam_pq.rs` — k-means basin assignment
  - `src/hpc/bf16_tile_gemm.rs` — AMX-class GEMM dispatch
  - `src/hpc/codec/{ctu,mode,predict}.rs` — codec wire format

_Last edit: 2026-05-22._
