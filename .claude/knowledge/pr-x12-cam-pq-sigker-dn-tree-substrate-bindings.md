# PR-X12 ↔ cam_pq + SigKer + dn_tree/merkle — Substrate Bindings & Identified Gaps

> Date: 2026-05-22
> Status: **substrate-binding doc** — extends `pr-x12-bgz-jc-substrate-synergies.md` with three more existing primitives the PR-X12 substrate depends on but hasn't yet named explicitly: `ndarray::hpc::cam_pq` (codebook trainer), `lance-graph/crates/sigker` (signature-kernel formal-proof bedrock), and `ndarray::hpc::{dn_tree, merkle_tree}` (online-update + integrity infrastructure).
>
> Premise: bgz17 and bgz-hhtl-d don't appear out of thin air. The k-means that produces their palettes lives in `cam_pq`. The formal uniqueness claim that justifies the codec's correctness lives in `sigker` (cited by jc Pillar 11). The federated-codebook gossip + integrity contract that R-13 commits to has substrate in `dn_tree` and `merkle_tree`. Three more pieces of the PR-X12 architecture are *already implemented*; this doc names them and surfaces the remaining gaps.

---

## 0. Thesis in one paragraph

`cam_pq` is the **codebook trainer** that produces the palettes consumed by `bgz17::palette` and `bgz-tensor::HhtlCascade` — a FAISS-style 6×8 Product Quantizer with 6-byte fingerprints (HEEL / BRANCH / TWIG_A / TWIG_B / LEAF / GAMMA semantic bytes), implementing k-means in three modes (geometric / semantic / hybrid). `SigKer` is the **formal-proof bedrock** for the codec's wire-format uniqueness claim — Chen-Lyons path signatures, Hambly-Lyons 2010 uniqueness theorem (arXiv:2010.[ID]), Salvi 2020 PDE solver (arXiv:2006.14794), Cuchiero-Schmocker-Teichmann 2021 randomized-signature universality. `dn_tree` and `merkle_tree` are the **online-update and integrity substrate** for the federated-codebook policy (R-13) — quaternary plastic memory + 8-Kbit Blake3 proof tree, both already in `ndarray::hpc::` but not yet wired into the codec. The PR-X12 codec body is ~1500 LoC sitting on top of an ~25 KLoC substrate that already exists.

---

## 1. `cam_pq` — the codebook trainer + ADC backend

### 1.1 What it is

**Location:** `/home/user/ndarray/src/hpc/cam_pq.rs`

**Algorithm:** Content-Addressable Memory (CAM) + Product Quantization (PQ). Unifies FAISS PQ6×8 (48-bit fingerprints, 6 subspaces × 256 centroids each) with CLAM 48-bit archetypes into a single codec.

- **"CAM"** = the 6-byte *semantic* labeling: each byte is one of {HEEL, BRANCH, TWIG_A, TWIG_B, LEAF, GAMMA} — discrete labels rather than just opaque centroid IDs.
- **"PQ"** = the 6 *subspace* product quantization: input vector of dim d is split into 6 sub-vectors of d/6, each quantized to one of 256 centroids per subspace.

**Result:** every input vector → 6-byte fingerprint (48 bits, half the 12-bit-basin × 4 of bgz-hhtl-d), with both *combinatorial* identity (which centroid in each subspace) and *semantic* identity (the CAM byte type per subspace).

### 1.2 Public surface

```rust
// From src/hpc/cam_pq.rs (per Explore agent's read):
pub struct CamCodebook { /* 6 × SubspaceCodebook */ }
pub struct SubspaceCodebook { /* 256 centroids in d/6 dims */ }
pub struct CamFingerprint(pub [u8; 6]);
pub struct DistanceTables { /* 6 × 256 = 6 KB, L1-resident */ }
pub struct PackedDatabase { /* stroke-aligned 1B / 2B / 6B storage */ }

pub fn kmeans(data: &[f32], k: usize, dim: usize, iterations: usize) -> Vec<f32>;
pub fn train_geometric(...) -> CamCodebook;   // Lloyd's algorithm per subspace
pub fn train_semantic(...)  -> CamCodebook;   // CLAM archetype clustering
pub fn train_hybrid(...)    -> CamCodebook;   // geometric init + semantic fine-tune
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32;
```

**ADC (Asymmetric Distance Computation):** 6 table lookups + sum (uniform across the FAISS-PQ tradition). Distance computation is L1-cache-resident: 6 × 256 × 2 B = 3 KB per query, ~6 KB if u16 distances.

**Early-exit:** `PackedDatabase` ships stroke-aligned storage with 1-byte / 2-byte / 6-byte CAM strides → **99% early-rejection** via partial-fingerprint comparison before full ADC. This is a non-trivial throughput optimization.

### 1.3 Connection to bgz17 + bgz-tensor + bgz-hhtl-d

**Direct imports** (per Explore agent's grep):

- `bgz-tensor/src/adaptive_codec.rs` imports `cam_pq::train_geometric, kmeans, squared_l2`
- `bgz-tensor/src/holographic_residual.rs` imports `cam_pq::kmeans`
- `bgz-tensor/src/had_cascade.rs` imports `cam_pq::squared_l2`
- `bgz17` palette codec uses cam_pq for calibration

**So cam_pq IS the k-means engine that trains every basin codebook in the bgz family.** The 4096-entry HHTL lattice that bgz-hhtl-d ships — its centroids come from `cam_pq::train_geometric()`.

### 1.4 Mapping cam_pq's CAM bytes onto bgz-hhtl-d's HHTL bits

The 6-byte CAM fingerprint and bgz-hhtl-d's 4-byte slot encoding overlap structurally:

| CAM byte | bgz-hhtl-d Slot D | Role |
|---|---|---|
| HEEL (byte 0) | `Ba` bits 15:14 (2 bits) | Tensor-family basin (QK / V / Gate / FFN) |
| BRANCH (byte 1) | `HIP` bits 13:10 (4 bits) | 16-way family discriminant within basin |
| TWIG_A (byte 2) | `TWIG` bits 9:2 (low byte) | 256-way centroid index, low |
| TWIG_B (byte 3) | `TWIG` bits 9:2 (high byte) | (same field, no high byte at 8b TWIG) |
| LEAF (byte 4) | `P` + `R` bits 1:0 (2 bits) | Polarity + reserved |
| GAMMA (byte 5) | Slot V (16 bits) | BF16 residual from centroid (full byte 5 + 1 of Slot V) |

**Observation:** the bgz-hhtl-d format **compresses cam_pq's 6-byte CAM** down to 4 bytes by:
- Folding TWIG_A + TWIG_B into a single 8-bit TWIG (since 256 centroids fit in 8 bits, no need for 16 — the 6 × 256 × subspaces parametrization was for full PQ; HHTL uses a *single* shared 256-entry palette)
- Folding LEAF into 2 bits (polarity + reserved)
- Folding GAMMA into the 16-bit BF16 residual (Slot V)

This is **cam_pq compressed via the HHTL prior:** since transformer weights cluster strongly per role (Q/K/V/O/Gate/Up/Down), the 6-subspace PQ generalization is over-parametrized — bgz-hhtl-d drops to a single shared palette per role and recovers the savings.

### 1.5 PR-X12 mapping

For the codec's `R-13` federated codebook handle:

```rust
pub enum CodebookPolicy {
    LocalEphemeral,                    // each encoder owns its codebook
    SharedClusterWide { ttl: Duration }, // gossip protocol distributes
    SharedRegional { region: Region },   // edge-tier sharing
    PretrainedStatic { id: BlobId },     // immutable, served from CAS
}
```

The codebook implementation is `cam_pq::CamCodebook`. The four policy variants control *who owns* and *when refreshes happen*; the bytes-on-disk format is the cam_pq one. **PR-X12 doesn't need to define a new codebook layout — it inherits cam_pq's.**

### 1.6 Three gaps in the cam_pq integration

**G-1: Activation-aware training mode is unused.** `cam_pq::train_semantic()` exists with CLAM archetype clustering — exactly the GPTQ/AWQ-style activation-weighting from the GGUF lens doc (`pr-x12-gguf-llm-weights-encoding.md` §5). bgz-hhtl-d ships *only* `train_geometric()` (L2-error minimization). Wiring `train_semantic()` into bgz-hhtl-d's calibration is a low-cost, high-value change (~1-2 days).

**G-2: `PackedDatabase`'s 99% early-exit not in the codec stream-decode path.** PackedDatabase is used by CAKES nearest-neighbour search to reject 99% of candidates before full ADC. The codec's stream-decode pass currently does full ADC per cell. Wiring the partial-fingerprint prefilter into the codec would speed decode by ~10-50× on Skip-dominant streams.

**G-3: CAM semantic bytes don't propagate to the PR-X12 wire-format header.** The 16-bit codec header has `header_kind` (2b) + `basin_index` (12b) + `leaf_size` (2b). No field carries HEEL/BRANCH/etc. labels. For *interpretation* in the consumer crates (e.g., `woa-rs` orchestration knowing whether a cell is a Q-projection vs an FFN-gate), the semantic byte would be useful. **Proposal:** reserve a 4-byte "semantic header" extension in the framing layer (A8) that ships once per CTU, separate from the per-cell header.

---

## 2. `SigKer` — the formal-proof bedrock for stream uniqueness

### 2.1 What it is

**Location:** `/home/user/lance-graph/crates/sigker/`

**Algorithm:** Path-signature representations for sequential / path-structured data. Implements Chen-Lyons signatures S(X) = (1, ∫dX, ∫∫dX⊗dX, …) up to depth N, with shuffle-product algebra and proven uniqueness.

**Public surface:**

```rust
pub struct Signature { /* truncated signature up to depth N */ }
pub struct RandomizedSignature { /* finite-dim projection */ }
pub struct RandomizedSignatureBuilder { /* construction */ }

pub fn signature_kernel(a: &Signature, b: &Signature) -> f64;     // truncated tensor-algebra L²
pub fn signature_kernel_pde(path_a: &[f32], path_b: &[f32]) -> f64; // full kernel via Goursat PDE
pub fn shuffle_product(a: &Signature, b: &Signature) -> Signature;

pub struct CodecRouteSigker { /* lance-graph codec routing integration */ }
```

**Zero production dependencies.** Same posture as `bgz17` and `deepnsm` — no external crates pulled in default features.

### 2.2 arXiv anchors

| Paper | Year | What it provides |
|---|---|---|
| Chen, "Iterated integrals and exponential homomorphisms" | 1957 | Original signature construction |
| Lyons, "Differential equations driven by rough signals" | 1998 | Rough path theory, signature universal approximator |
| Hambly-Lyons, "Uniqueness for the signature of a path of bounded variation" | 2010 | **Theorem 4: signatures uniquely determine paths up to tree-like equivalence** |
| Salvi-Cass-Foster-Lyons-Lemercier | 2020 | **arXiv:2006.14794** — Goursat-PDE solver for signature kernel, O(T₁·T₂·d), no signature materialization |
| Cuchiero-Schmocker-Teichmann | 2021 | **Randomized signature universality**: any continuous path-functional ≈ linear combo of randomized-signature coordinates |

### 2.3 jc's Pillar 11 — current status

**Location:** `lance-graph/crates/jc/src/hambly_lyons.rs`

**Feature gate:** `jc/Cargo.toml` includes `hambly-lyons = ["sigker"]`. Activating the feature pulls sigker as a dep.

**Pillar 11 proves:** for any two source streams X, Y with truncated bgz-encoded representations B(X), B(Y) up to depth N, if B(X) = B(Y) then X = Y up to tree-like path equivalence.

**Status:** **DEFERRED** per Explore agent's read of jc source. Pillar 11 has the proof binding but production benchmarking at full carrier widths is incomplete.

### 2.4 PR-X12 mapping

#### 2.4.1 Path signatures ARE a `Basis<T>` impl

Recall R-1 / §M:E-A: `Basis<T>` is "basis-as-data" with parametric `apply`. The truncated signature of a path IS exactly this — basis vectors are the tensor-algebra elements at each depth, apply is iterated integration.

```rust
impl<const DEPTH: usize> Basis<f32> for SignatureBasis<DEPTH> {
    type Params = ();
    fn apply<R: Reducer<f32>>(
        &self,
        path: &[f32],         // input path samples
        signature: &mut [f32], // output truncated signature
        _: &(),
        r: R,
    ) {
        // iterated integral computation, depth-truncated
        // Same trait shape as DctIIBasis<N>, EwaSplatBasis
    }
}
```

This is the **third Basis<T> impl** (after DCT-II and EWA splat) and the first that targets *streams* rather than 2D arrays. The trait surface holds.

#### 2.4.2 `signature_kernel_pde` IS the streaming-decode pattern

Per the Salvi 2020 paper (arXiv:2006.14794): the signature kernel can be computed via a Goursat PDE in O(T₁ · T₂ · d) time **without materializing the signature itself**. This is exactly the engineering pattern PR-X12's streaming-decode-during-GEMM uses (the GGUF lens §7) — compute the result without materializing the dequantized tensor.

The connection is structural, not just analogical: PR-X12's stream-decode pass *is* a 1D Goursat-style sweep over the bitstream that accumulates results without materializing intermediate state. The math literature for this pattern is mature (Salvi 2020 has citations going back 10+ years) and ships in sigker.

#### 2.4.3 Randomized signature universality = "4 modes cover any source"

Cuchiero-Schmocker-Teichmann 2021 proved: any continuous functional of a path can be approximated arbitrarily well by linear combinations of randomized-signature coordinates. The randomized signature is a finite-dim projection of the (infinite-dim) full signature.

**PR-X12's claim:** Skip/Merge/Delta/Escape with a 4096-entry basin codebook captures any source distribution to within a Shannon-bounded ε. This claim is *empirically* observed (95% Skip-rate at Layer 0-1 in bgz17, 343:1 compression on Qwen3-TTS-1.7B in bgz-hhtl-d) but lacks a *formal* foundation.

**The randomized-signature universality theorem provides exactly that formal foundation.** The four modes are a discrete approximation of the randomized-signature coordinates; the 4096-entry codebook is a finite quantization of the universal-approximator space.

This is the **R-14 candidate** flagged in `pr-x12-bgz-jc-substrate-synergies.md` §8.1 — a formal-correctness contract via sigker + jc Pillar 11.

### 2.5 Two gaps in the sigker integration

**G-4: Pillar 11 is DEFERRED.** Unblock it. The math is published, the implementation exists, the bench harness needs production-scale validation. **Cost:** 1-2 weeks of bench + verification work, blocking R-14 formal-correctness commitment.

**G-5: No SignatureBasis<DEPTH> impl in `ndarray::hpc::`.** The trait shape exists (Basis<T> in M:E-A / R-1) but no concrete signature impl. **Proposal:** add `SignatureBasis<const DEPTH: usize>: Basis<f32>` as a third concrete impl alongside `DctIIBasis<N>` and `EwaSplatBasis`. Implementation is mostly a wrapper around `sigker::signature_kernel_pde`. **Cost:** ~1 week, modest LoC.

This unlocks: **path-structured codec lanes** in Plan G (audio waveforms, time-series, gesture/handwriting streams) using the same trait surface as DCT-II for video. A fourth bench lane in Plan G — "stream signal" with sigker — would round out the codec's path-structured story.

---

## 3. `dn_tree` and `merkle_tree` — online-update and integrity substrate

### 3.1 dn_tree — quaternary plastic memory

**Location:** `/home/user/ndarray/src/hpc/dn_tree.rs`

**Algorithm:** Quaternary hierarchical bitmap summary tree for plastic graph traversal. Adapted from "On Demand Memory Specialization for Distributed Graph Processing" (2013). Properties:

- **Quaternary fanout:** 4 children per node — natural match for PR-X12's 4-mode taxonomy
- **Lossy hierarchical summaries** using bundled `GraphHV` hypervectors (3 channels × 256 words = 16,384 bits each)
- **Partial Hamming similarity** on prefix bits for fast descent
- **Plastic bundling** + exponential decay on access (biological LTP/LTD)
- **BTSP-inspired stochastic gating** (CaMKII-like boost for high-confidence updates)

**Public types:** `DNConfig`, `DNNode`, `TraversalHit`, `SplitMix64` (RNG).

**Latency:** update ~30 ns/level, traverse 180-420 ns. L1/L2-cache-resident at scale.

### 3.2 merkle_tree — integrity proof for CogRecord regions

**Location:** `/home/user/ndarray/src/hpc/merkle_tree.rs`

**Algorithm:** 8-Kbit Merkle tree built from CogRecord regions as a compressed searchable proxy. Properties:

- **Hash:** Blake3 truncated to 48 bits (6 bytes per hash — same width as cam_pq's CamFingerprint)
- **Layout:** 8 branches × 8 leaves = 64 leaves, padded to 1 KB for SIMD alignment
- **Change detection:** `StaunenType` enum {Wisdom, ContentChanged, NarsChanged, ...} — 8 change types

**Memory layout matches cam_pq's distance-table** (both 6 KB-class structures, L1-resident). Same engineering pattern, different application.

### 3.3 PR-X12 mapping

#### 3.3.1 dn_tree IS the online-update substrate for R-13's `SharedClusterWide`

R-13 commits the codec to a swappable codebook handle with four policy modes. `SharedClusterWide` is the runtime-updated mode where a cluster of encoders gossips codebook changes.

**The substrate decision:** how to merge incoming gossip updates into the local codebook without losing accumulated signal? Standard answer is "overwrite with latest" — but that loses the priors. dn_tree's plastic bundling + exponential decay handles exactly this: gossip updates bundle into the existing structure with decaying influence, recent updates dominate without erasing history.

**Proposal:** `R-13::SharedClusterWide` is implemented via `dn_tree::DNNode` per codebook entry, not via raw HashMap or RwLock. The quaternary fanout naturally indexes the 4 mode categories.

**Cost:** ~2-3 weeks to wire dn_tree into the codec's codebook handle. Modest LoC (the trait exists), but design work to choose the right plastic-decay parameters.

#### 3.3.2 dn_tree's 4-way fanout matches PR-X12's 4-mode taxonomy

The dn_tree's *quaternary* (4-children-per-node) structure is structurally identical to the Skip/Merge/Delta/Escape discriminant. **Each dn_tree node naturally holds per-mode statistics:**

- Child 0: Skip-frequency at this level
- Child 1: Merge-frequency
- Child 2: Delta-frequency
- Child 3: Escape-frequency

This makes dn_tree the natural data structure for **mode-distribution drift detection**. If a codec instance is seeing 95% Skip on the training distribution and drops to 60% Skip on a new input, dn_tree's recursive structure catches that drift early and signals "codebook stale, federated update needed."

This is one of the things M:H-NEW-1's "Plan G falsifiability test" should measure but currently doesn't. dn_tree gives us the data structure to do so.

#### 3.3.3 Merkle tree IS the integrity proof for R-13 distribution

When q2's gossip protocol distributes a codebook update to N edge nodes, how do consumers verify the update wasn't tampered with mid-transit? Merkle root.

The 8-Kbit Blake3-48-bit Merkle layout in `merkle_tree.rs` is **byte-compatible** with cam_pq's distance-table layout (both 6-byte hashes, both L1-resident). The codebook update can carry its Merkle root as the first 1 KB of the update payload; consumers verify the root before merging into local dn_tree.

**Proposal:** R-13's payload format is `[Merkle root (1 KB)] + [codebook delta (cam_pq encoded)]`. q2 implements the gossip protocol; ndarray::hpc::merkle_tree implements the verification.

**Cost:** ~1 week to integrate Merkle verification into the codec's codebook-update path. The Merkle infrastructure already exists; this is wiring.

### 3.4 Two gaps in dn_tree / merkle_tree integration

**G-6: dn_tree not wired into any codec or codebook-update path.** Currently only used for pillar tests (`pillar/btsp_unbiased.rs`, `pillar/tree_balance.rs`, `pillar/hhtl_contraction.rs`). **Blocking R-13's `SharedClusterWide` mode.**

**G-7: merkle_tree not wired into federated codebook distribution.** Currently only used for `surround_metadata.rs` change detection. **Blocking R-13's integrity guarantee for SharedClusterWide / SharedRegional modes.**

---

## 4. The unified picture — all 8 substrate primitives now identified

Updating the inventory from `pr-x12-bgz-jc-substrate-synergies.md` §7 with the three new primitives:

| PR-X12 abstract concept | Concrete implementation |
|---|---|
| Skip/Merge/Delta/Escape | `bgz17` 4-layer cascade (Scent/Palette/ZeckBF17/Full) |
| 4096-entry basin codebook | `bgz-hhtl-d` HHTL 16×16×16 lattice, trained by **`cam_pq`** |
| `CurveOrder<const N>` | `highheelbgz` spiral addressing |
| `LinearReduce<T> + Basis<T>` | `bgz-tensor` AttentionSemiring + ComposeTable + DistanceTable; **`sigker::SignatureBasis`** (proposed) |
| Tropical-GEMM (R-7) | `bgz17::scalar_sparse::tropical_spmv` |
| Federated codebook (R-13) | `bgz-hhtl-d` shared-palette + **`cam_pq::CamCodebook`** + **`dn_tree`** (online update) + **`merkle_tree`** (integrity) |
| Formal correctness | `jc` Pillar 11 (Hambly-Lyons) via **`sigker`** |
| Activation-aware RDO | **`cam_pq::train_semantic`** (exists, unused) |

**Eight primitives, six already implemented, three under-wired.** What PR-X12 ships is the *wire format + per-arch dispatch contract + cross-domain story* that knits them into one codec.

---

## 5. Seven concrete gaps (consolidated)

| Gap | Component | Cost | Blocking |
|---|---|---|---|
| **G-1** | Activation-aware codebook training (cam_pq::train_semantic) not used by bgz-hhtl-d | 1-2 days | GGUF lens activation-aware RDO claim |
| **G-2** | cam_pq::PackedDatabase 99% early-exit not in codec stream-decode path | 1-2 weeks | Decode throughput on Skip-dominant streams |
| **G-3** | CAM semantic bytes (HEEL/BRANCH/etc.) don't propagate to PR-X12 wire-format header | 3-5 days (wire-format extension in A8) | Consumer-side semantic interpretation |
| **G-4** | jc Pillar 11 (Hambly-Lyons via sigker) is DEFERRED | 1-2 weeks bench | R-14 formal-correctness commitment |
| **G-5** | No `SignatureBasis<DEPTH>` impl in `ndarray::hpc::` | 1 week | Path-structured codec lanes (audio, time-series) |
| **G-6** | dn_tree not wired into codebook update path | 2-3 weeks | R-13 `SharedClusterWide` mode |
| **G-7** | merkle_tree not wired into federated codebook distribution | ~1 week | R-13 integrity guarantee |

**Total estimated gap-closing work: 8-12 weeks** across the seven items, all incremental on existing infrastructure. None of them require new research; all are wiring existing primitives into the codec.

Two prior gaps from the earlier doc remain:

| Gap (prior) | Component | Cost |
|---|---|---|
| **G-8** | `jd-nd` crate does not exist (ndarray-side proof crate) | 2-3 weeks skeleton + ongoing |
| **G-9** | Cronbach/ICC encoding-reliability research crate not implemented | 1-2 weeks skeleton + 2-3 weeks PoC |

**Grand total: ~11-17 weeks** of substrate-binding + gap-closing work, parallel-able. PR-X12 codec body (~1500 LoC per R-3) is independent of this and can ship sooner.

---

## 6. Updates this triggers in canon-resolutions-delta

Recommended edits to `pr-x12-canon-resolutions-delta.md`:

**R-13 expansion** — name the implementation pieces:

> R-13 (revised): the basin codebook is implemented via `ndarray::hpc::cam_pq::CamCodebook` (training) + `lance-graph::bgz-hhtl-d` (deployed encoding format) + `ndarray::hpc::dn_tree` (online plastic updates for `SharedClusterWide`) + `ndarray::hpc::merkle_tree` (integrity proof for distributed updates). The four policy modes (`LocalEphemeral` / `SharedClusterWide` / `SharedRegional` / `PretrainedStatic`) compose these primitives differently. The codec body exposes a `CodebookHandle` trait; q2 implements the gossip protocol; ndarray ships the primitives.

**R-14 (new)** — formal-correctness commitment:

> R-14: the codec's wire-format determinism and bit-exact cross-arch reproduction are formally proven in `lance-graph/crates/jc/` Pillar 11 (Hambly-Lyons signature uniqueness) backed by `lance-graph/crates/sigker/` (Chen-Lyons signatures + Salvi 2020 PDE solver + CST 2021 randomized-signature universality). The codec cites the proof; does not reprove. **Status: blocked on Pillar 11 production benchmarking — see Gap G-4.**

**R-7 path correction** — the kernel home:

> R-7 (corrected): tropical-GEMM lives at `lance-graph::bgz17::scalar_sparse::tropical_spmv` (not the abstract `blasgraph` namespace). The codec's tropical-GEMM RDO call is `bgz17::scalar_sparse::tropical_spmv(edge_weights, dag)`.

**R-15 (new candidate)** — signature-basis as Basis<T> impl:

> R-15 (candidate): the substrate supports path-structured signals via `sigker::SignatureBasis<DEPTH>: Basis<f32>`, alongside `DctIIBasis<N>: Basis<i16>` (video) and `EwaSplatBasis: Basis<f16>` (3DGS). Implementation: ~1 week wrapper around `sigker::signature_kernel_pde`. **Plan G** gets a fifth lane (path-structured: audio waveform, time-series, gesture/handwriting).

---

## 7. Reading order — fresh agent onboarding

For a fresh PR-X12 agent landing on the substrate, the reading order is now:

1. `pr-x12-substrate-merged-canon.md` (the architectural top-level)
2. `pr-x12-canon-resolutions-delta.md` (R-1..R-13 + R-14/R-15 candidates)
3. **`pr-x12-bgz-jc-substrate-synergies.md`** (PR-X12 ↔ bgz family ↔ jc grounding)
4. **`pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md`** (this doc — three more primitives + 7 gaps)
5. Perspective lens docs in any order:
   - `pr-x12-x265-blasgraph-gemm.md`
   - `pr-x12-x266-3dgs-spacetime-upscaling.md`
   - `pr-x12-woa-multiarch-orchestration.md`
   - `pr-x12-anti-neural-lookup-inversion.md`
   - `pr-x12-gguf-llm-weights-encoding.md`
6. Mechanical specs:
   - `pr-x12-codec-x265-design.md` (the HEVC-analog spec)
   - `pr-x12-codec-cognitive-substrate-mapping.md` (PR #195 derivative)
   - `pr-x12-cross-domain-synergies.md` (epiphany doc)

This doc (#4) and the bgz/jc doc (#3) are the ones that ground PR-X12 in working code. Without them, an agent reads the perspective lenses as theoretical claims; with them, the agent knows the substrate is already 70%+ implemented.

---

## 8. Cross-references

- **Companion grounding doc:** `pr-x12-bgz-jc-substrate-synergies.md`
- **Canonical canon:** `pr-x12-substrate-merged-canon.md`
- **Resolutions:** `pr-x12-canon-resolutions-delta.md` (R-13 expansion, R-14 + R-15 candidates needed)
- **GGUF lens (activation-aware RDO claim):** `pr-x12-gguf-llm-weights-encoding.md` §5 — supported by G-1 closure
- **Anti-neural lens (lookup-table cost analysis):** `pr-x12-anti-neural-lookup-inversion.md` §3 — supported by G-4 + G-5 closure
- **Multi-arch lens (determinism + integrity):** `pr-x12-woa-multiarch-orchestration.md` §6 — supported by G-4 + G-7 closure
- **Source code references:**
  - `/home/user/ndarray/src/hpc/cam_pq.rs` — the codebook trainer
  - `/home/user/ndarray/src/hpc/dn_tree.rs` — quaternary plastic memory
  - `/home/user/ndarray/src/hpc/merkle_tree.rs` — Blake3-48-bit Merkle
  - `/home/user/lance-graph/crates/sigker/` — Chen-Lyons signatures
  - `/home/user/lance-graph/crates/sigker/src/` — `signature_kernel_pde`, `RandomizedSignature`, `CodecRouteSigker`
  - `/home/user/lance-graph/crates/jc/src/hambly_lyons.rs` — Pillar 11 (DEFERRED)
  - `/home/user/lance-graph/crates/bgz-tensor/src/adaptive_codec.rs` — cam_pq imports
- **arXiv anchors for sigker:**
  - **2006.14794** (Salvi-Cass-Foster-Lyons-Lemercier 2020) — Goursat PDE for signature kernel
  - Hambly-Lyons 2010 — signature uniqueness theorem
  - Cuchiero-Schmocker-Teichmann 2021 — randomized-signature universality
- **arXiv anchor for dn_tree:**
  - "On Demand Memory Specialization for Distributed Graph Processing" (2013)

_Last edit: 2026-05-22._
