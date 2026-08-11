# PR-X12 ↔ cam_pq + SigKer + dn_tree/merkle — Substrate Bindings & Identified Gaps

> Date: 2026-05-22
> Status: **substrate-binding doc** — extends `pr-x12-bgz-jc-substrate-synergies.md` with three more existing primitives the PR-X12 substrate depends on but hasn't yet named explicitly: `ndarray::hpc::cam_pq` (codebook trainer), `lance-graph/crates/sigker` (signature-kernel formal-proof bedrock), and `ndarray::hpc::{dn_tree, merkle_tree}` (online-update + integrity infrastructure).
>
> Premise: bgz17 and bgz-hhtl-d don't appear out of thin air. The k-means that produces their palettes lives in `cam_pq`. The formal uniqueness claim that justifies the codec's correctness lives in `sigker` (cited by jc Pillar 11). The federated-codebook gossip + integrity contract that R-13 commits to has substrate in `dn_tree` and `merkle_tree`. Three more pieces of the PR-X12 architecture are *already implemented*; this doc names them and surfaces the remaining gaps.

---

## 0. Thesis in one paragraph

`cam_pq` is the **codebook trainer** that produces the palettes consumed by `bgz17::palette` and `bgz-tensor::HhtlCascade` — a FAISS-style 6×8 Product Quantizer with 6-byte fingerprints (HEEL / BRANCH / TWIG_A / TWIG_B / LEAF / GAMMA semantic bytes), implementing k-means in three modes (geometric / semantic / hybrid). `SigKer` is the **formal-proof bedrock** for the codec's path-signature uniqueness claim — Chen-Lyons path signatures, Hambly-Lyons 2010 uniqueness theorem (Annals of Mathematics 171(1):109–167), Salvi 2020 PDE solver (arXiv:2006.14794), Cuchiero-Schmocker-Teichmann 2021 randomized-signature universality. **Note:** `sigker::signature_kernel_pde` ships a known math bug in the Goursat-PDE form (diverges from the true `I₀(2·√⟨u, v⟩)` at moderate inner products — see PR #350); production-ready path is `signature_truncated` (tensor-algebra) which is what jc Pillar 11 uses for its certification. `dn_tree` and `merkle_tree` are the **online-update and integrity substrate** for the federated-codebook policy (R-13) — quaternary plastic memory + 8-Kbit Blake3 proof tree, both already in `ndarray::hpc::` but not yet wired into the codec. The PR-X12 codec body is ~1500 LoC sitting on top of an ~25 KLoC substrate that already exists.

---

## 1. `cam_pq` — the codebook trainer + ADC backend

### 1.1 What it is

**Location:** `src/hpc/cam_pq.rs` (this repo)

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
pub fn train_geometric(...) -> CamCodebook;   // Lloyd's k-means per subspace, farthest-first init
pub fn train_semantic(...)  -> CamCodebook;   // geometric init + label-guided push/pull on centroids
                                              // (jaccard similarity on label sets, NOT CLAM archetypes)
pub fn train_hybrid(...)    -> CamCodebook;   // train_semantic with default alpha=0.1
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

**Location:** `crates/sigker/` in the external `adaworldapi/lance-graph` repo (not in this `ndarray` repo)

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
| Hambly-Lyons, "Uniqueness for the signature of a path of bounded variation" (**arXiv:math/0507536**, Annals of Mathematics 171(1):109–167) | 2010 | **Theorem 4: signatures uniquely determine paths up to tree-like equivalence** |
| Salvi-Cass-Foster-Lyons-Lemercier | 2020 | **arXiv:2006.14794** — Goursat-PDE solver for signature kernel, O(T₁·T₂·d), no signature materialization |
| Cuchiero-Schmocker-Teichmann | 2021 | **Randomized signature universality**: any continuous path-functional ≈ linear combo of randomized-signature coordinates |

### 2.3 jc's Pillar 11 — current status

**Location:** `lance-graph/crates/jc/src/hambly_lyons.rs`

**Feature gate:** `jc/Cargo.toml` includes `hambly-lyons = ["dep:sigker"]`. Default JC build is zero-dep (Pillar 11 reports `DEFERRED`); `cargo build --features hambly-lyons` pulls in `sigker` and **fully activates the probe**.

**Pillar 11 proves** (Hambly-Lyons 2010 Theorem 4): for paths X, Y of bounded variation in ℝ^d, S(X) = S(Y) ⟺ X and Y are equal modulo tree-like equivalence (the smallest equivalence relation identifying any sub-path with its concatenated reverse).

**The probe** runs against `sigker::signature_truncated` (the tensor-algebra path), N=100 random pairs in d=3 at depth-2. **It deliberately avoids `signature_kernel_pde`** because that kernel ships a known math bug (PR #350: Goursat-PDE form diverges from the true signature kernel `I₀(2·√⟨u, v⟩)` at moderate inner products). The certification is independent of the PDE-form fix.

**Status:** **ACTIVE under `--features hambly-lyons`** (activated 2026-05-07 once sigker landed in the workspace via PR #348). The "DEFERRED" reading is only the default-build fallback — under the feature gate, the probe executes and passes (forward < 1e-9, converse > 0.05, discrimination ratio ≥ 1e6).

**What Pillar 11 actually certifies:** `sigker`'s **Index-regime classification** — that two paths with equal truncated signatures are equal up to tree-quotient. It does **NOT** directly certify `bgz` wire-format quantization. The bgz / CAM-PQ correctness proof is **Pillar 10 (Pflug-Pichler)**, which proves nested-distance Lipschitz on Sigma DN-trees — "CAM-PQ tree quantization preserves FreeEnergy within Lε."

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

#### 2.4.2 Goursat-style streaming kernel IS the streaming-decode pattern

Per the Salvi 2020 paper (arXiv:2006.14794): the signature kernel can be computed via a Goursat PDE in O(T₁ · T₂ · d) time **without materializing the signature itself**. This is exactly the engineering pattern PR-X12's streaming-decode-during-GEMM uses (the GGUF lens §7) — compute the result without materializing the dequantized tensor.

**Caveat:** the current `sigker::signature_kernel_pde` ships a known math bug (PR #350: the Goursat-PDE form diverges from the true kernel `I₀(2·√⟨u, v⟩)` at moderate inner products). The corrected form is queued; until then, production code should use `sigker::signature_truncated` (the tensor-algebra path) or `linear_path_kernel_closed_form` for the linear-path special case. The *engineering pattern* (1D sweep over a bitstream that accumulates results without materializing intermediates) is correct and re-usable by PR-X12 regardless of which kernel implementation lands.

#### 2.4.3 Randomized signature universality = "4 modes cover any source"

Cuchiero-Schmocker-Teichmann 2021 proved: any continuous functional of a path can be approximated arbitrarily well by linear combinations of randomized-signature coordinates. The randomized signature is a finite-dim projection of the (infinite-dim) full signature.

**PR-X12's claim:** Skip/Merge/Delta/Escape with a 4096-entry basin codebook captures any source distribution to within a Shannon-bounded ε. This claim is *empirically* observed (95% Skip-rate at Layer 0-1 in bgz17, 343:1 compression on Qwen3-TTS-1.7B in bgz-hhtl-d) but lacks a *formal* foundation.

**The randomized-signature universality theorem provides exactly that formal foundation.** The four modes are a discrete approximation of the randomized-signature coordinates; the 4096-entry codebook is a finite quantization of the universal-approximator space.

This is the **R-14 candidate** flagged in `pr-x12-bgz-jc-substrate-synergies.md` §8.1 — a formal-correctness contract via sigker + jc Pillar 11.

### 2.5 Two gaps in the sigker integration

**G-4: PR #350 (signature_kernel_pde correction) + Pillar 11 production benchmarking.** Pillar 11 itself is *active* under the feature gate and passes its probe (forward < 1e-9, converse > 0.05, discrimination ratio ≥ 1e6 over N=100 pairs in d=3). What's *deferred* is (a) the corrected Goursat-PDE form that fixes `signature_kernel_pde`'s divergence at moderate inner products, and (b) production-scale benchmarking at full carrier widths (the d=3, depth-2 probe is correctness-only, not performance). **Cost:** 1-2 weeks of bench + PR #350 landing, blocking R-14's formal-correctness commitment at production scale.

**G-5: No SignatureBasis<DEPTH> impl in `ndarray::hpc::`.** The trait shape exists (Basis<T> in M:E-A / R-1) but no concrete signature impl. **Proposal:** add `SignatureBasis<const DEPTH: usize>: Basis<f32>` as a third concrete impl alongside `DctIIBasis<N>` and `EwaSplatBasis`. Implementation is mostly a wrapper around `sigker::signature_kernel_pde`. **Cost:** ~1 week, modest LoC.

This unlocks: **path-structured codec lanes** in Plan G (audio waveforms, time-series, gesture/handwriting streams) using the same trait surface as DCT-II for video. A fourth bench lane in Plan G — "stream signal" with sigker — would round out the codec's path-structured story.

---

## 3. `dn_tree` and `merkle_tree` — online-update and integrity substrate

### 3.1 dn_tree — quaternary plastic memory

**Location:** `src/hpc/dn_tree.rs` (this repo)

**Algorithm:** Quaternary hierarchical bitmap summary tree for plastic graph traversal. Adapted from "On Demand Memory Specialization for Distributed Graph Processing" (2013). Properties:

- **Quaternary fanout:** 4 children per node — natural match for PR-X12's 4-mode taxonomy
- **Lossy hierarchical summaries** using bundled `GraphHV` hypervectors (3 channels × 256 words = 16,384 bits each)
- **Partial Hamming similarity** on prefix bits for fast descent
- **Plastic bundling** + exponential decay on access (biological LTP/LTD)
- **BTSP-inspired stochastic gating** (CaMKII-like boost for high-confidence updates)

**Public types:** `DNConfig`, `DNNode`, `TraversalHit`, `SplitMix64` (RNG).

**Latency:** update ~30 ns/level, traverse 180-420 ns. L1/L2-cache-resident at scale.

### 3.2 merkle_tree — integrity proof for CogRecord regions

**Location:** `src/hpc/merkle_tree.rs` (this repo)

**Algorithm:** 8-Kbit Merkle tree built from CogRecord regions as a compressed searchable proxy. Properties:

- **Hash:** Blake3 truncated to 48 bits (`MerkleRoot = [u8; 6]` — same byte width as cam_pq's `CamFingerprint = [u8; NUM_SUBSPACES]` where NUM_SUBSPACES = 6, though the semantic content differs: one is hash bytes, the other is centroid indices)
- **Layout:** 8 branches × 8 leaves-per-branch = 64 leaves, packed into 128 × u64 = 1 KB (8 Kbit) padded buffer for SIMD alignment. Semantic content is 48 + 384 + 3072 = 3504 bits; the rest is zero-padding.
- **Branch indices** (per `BRANCH_REGIONS` constant): 0=identity, 1=nars, 2=edges, 3=rl, 4=bloom, 5=qualia, 6=adjacency, 7=content
- **Change detection:** `StaunenType` enum with 6 explicit variants — `Wisdom` (no change), `ContentChanged` (branch 7 only), `NarsChanged` (branch 1 only), `EdgesChanged` (branch 2 only), `QualiaChanged` (branch 5 only), `MultipleChanges(Vec<u8>)` (catch-all carrying the list of differing branch indices). Note: branches 0/3/4/6 don't get their own single-change variant; they fall into `MultipleChanges` even when only one of them differs.
- **`xor_diff`:** panCAKES compression — XOR two Merkle trees' bits arrays, rebuild root/branches/leaves. The XOR-diff is what gossip transmits.

Both `MerkleTree::hamming` and `MerkleTree::diff_sparsity` are SIMD-accelerated (via `hamming_distance_raw` over the 1 KB byte view). The tree is hashable in O(n) where n is the CogRecord size, and the 1 KB output is L1-cache-resident.

### 3.3 PR-X12 mapping

#### 3.3.1 dn_tree IS the online-update substrate for R-13's `SharedClusterWide`

R-13 commits the codec to a swappable codebook handle with four policy modes. `SharedClusterWide` is the runtime-updated mode where a cluster of encoders gossips codebook changes.

**The substrate decision:** how to merge incoming gossip updates into the local codebook without losing accumulated signal? Standard answer is "overwrite with latest" — but that loses the priors. dn_tree's plastic bundling + exponential decay handles exactly this: gossip updates bundle into the existing structure with decaying influence, recent updates dominate without erasing history.

**Proposal:** `R-13::SharedClusterWide` is implemented via `dn_tree::DNNode` per codebook entry, not via raw HashMap or RwLock. The quaternary fanout naturally indexes the 4 mode categories.

**Cost:** ~2-3 weeks to wire dn_tree into the codec's codebook handle. Modest LoC (the trait exists), but design work to choose the right plastic-decay parameters.

#### 3.3.2 dn_tree's 4-way fanout — structural suggestiveness, not literal mode-stats

**Correction from earlier framing:** dn_tree's quaternary structure is NOT a literal "Skip/Merge/Delta/Escape per child" container. Looking at the source (`DNTree::split_node`, `DNTree::select_child`): the 4 children are **equal-width quadrants of the prototype-index range** (`[lo, lo+q), [lo+q, lo+2q), [lo+2q, lo+3q), [lo+3q, hi)` where `q = (hi - lo) / 4`). The fanout is a *spatial partition*, not a *mode discriminant*.

What's structurally suggestive is that **a 4-mode discriminant could be layered on top** of dn_tree's existing infrastructure: each prototype slot could carry per-mode counts (Skip/Merge/Delta/Escape) bundled into the existing `GraphHV` summaries via the same plastic-bundling primitive (`bundle_into`). The 4-children fanout doesn't impose this — it permits it.

For **mode-distribution drift detection**, the practical wiring is: add per-mode access counters to `DNNode` (cheap, 4×u32 = 16 bytes per node), and use `DNTree::traverse` to find leaves whose mode-distribution diverges most from the prior. If a codec instance is seeing 95% Skip on the training distribution and drops to 60% Skip on a new input, the divergence is detectable via the existing `partial_similarity` mechanism over the per-mode counts. **dn_tree as a substrate works for this; the 4-fanout matching the 4 modes is a structural coincidence, not a load-bearing identity.**

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
| 4096-entry basin codebook | `bgz-tensor::Codebook4096` (literal 4096-entry type), trained by **`cam_pq`**. `bgz-hhtl-d` is a *different* basin-codebook strategy (4-basin × 16-HIP × 256-TWIG = 16,384-cell address space over a shared 256-entry palette) — not the canonical 4096 |
| `CurveOrder<const N>` | `highheelbgz` spiral addressing |
| `LinearReduce<T> + Basis<T>` | `bgz-tensor` AttentionSemiring + ComposeTable + DistanceTable; **`sigker::SignatureBasis`** (proposed) |
| Tropical-GEMM (R-7) | canonical home `lance-graph::blasgraph` (kernel unwritten); shipped min-plus is the method `bgz17::ScalarCsr::spmv_min_plus` [lossy sibling, prototype only — corrected 2026-07-16 per audit] |
| Federated codebook (R-13) | `bgz-hhtl-d` shared-palette + **`cam_pq::CamCodebook`** + **`dn_tree`** (online update) + **`merkle_tree`** (integrity) |
| Formal correctness — codec quantization | `jc` **Pillar 10 (Pflug-Pichler)** — nested-distance Lipschitz on Sigma DN-trees, certifies CAM-PQ tree quantization preserves FreeEnergy within Lε |
| Formal correctness — path-signature lane | `jc` **Pillar 11 (Hambly-Lyons)** via **`sigker`** — certifies Index-regime classification (sigker only, not bgz) |
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

Two prior gaps from the earlier doc remain (their canonical IDs are owned by `pr-x12-bgz-jc-substrate-synergies.md` §5; cross-referenced here):

| Gap (cross-ref) | Component | Cost |
|---|---|---|
| **bgz-jc G-1** (§5.1) | `jd-nd` crate does not exist (ndarray-side proof crate) | 2-3 weeks skeleton + ongoing |
| **bgz-jc G-2** (§5.2) | Cronbach/ICC encoding-reliability research crate not implemented | 1-2 weeks skeleton + 2-3 weeks PoC |

The G-1..G-7 IDs in §5 of *this* doc are local to the cam-pq / sigker / dn_tree binding; bgz-jc's G-1 / G-2 are a separate namespace owned by that doc. When citing cross-doc, prefix with the source (e.g., "bgz-jc G-1" vs "cam-pq G-1") to avoid the collision the previous G-8 / G-9 labelling implied.

**Grand total: ~11-17 weeks** of substrate-binding + gap-closing work, parallel-able. PR-X12 codec body (~1500 LoC per R-3) is independent of this and can ship sooner.

---

## 6. Updates this triggers in canon-resolutions-delta

Recommended edits to `pr-x12-canon-resolutions-delta.md`:

**R-13 expansion** — name the implementation pieces:

> R-13 (revised): the basin codebook is implemented via `ndarray::hpc::cam_pq::CamCodebook` (training) + `lance-graph::bgz-hhtl-d` (deployed encoding format) + `ndarray::hpc::dn_tree` (online plastic updates for `SharedClusterWide`) + `ndarray::hpc::merkle_tree` (integrity proof for distributed updates). The four policy modes (`LocalEphemeral` / `SharedClusterWide` / `SharedRegional` / `PretrainedStatic`) compose these primitives differently. The codec body exposes a `CodebookHandle` trait; q2 implements the gossip protocol; ndarray ships the primitives.

**R-14 (new)** — formal-correctness commitment:

> R-14: the codec's correctness has two formal proofs in `lance-graph/crates/jc/`:
> - **Quantization correctness (Pillar 10, Pflug-Pichler):** nested-distance Lipschitz on Sigma DN-trees — proves CAM-PQ tree quantization preserves FreeEnergy within Lε. This is the proof PR-X12 cites for "wire-format quantization is faithful."
> - **Path-signature correctness (Pillar 11, Hambly-Lyons):** signature uniqueness on tree-quotient — proves any path is uniquely determined by its truncated signature up to tree-like equivalence. Active under `--features hambly-lyons` (since 2026-05-07, PR #348). This is the proof PR-X12 cites for the `SignatureBasis<DEPTH>` lane (R-15).
>
> Both pillars exist; the codec cites them and does not reprove. **Status: Pillar 10 active; Pillar 11 active under feature gate. Production-scale benchmarking — see Gap G-4.** *(Corrected 2026-07-16, audit #9: the "PR #350 signature_kernel_pde math correction" claim is withdrawn — the PDE form's convergence tests to `I₀(2·√⟨u,v⟩)` pass; there is no known bug.)*

**R-7 kernel home** *(corrected 2026-07-16, audit #1-#4 — the earlier "path
correction" here had the canon/adapter relationship inverted)*:

> R-7 (corrected): the canonical, bit-exact home for the tropical-GEMM
> partition kernel is `lance-graph::blasgraph`; that f32 min-plus kernel is
> UNWRITTEN today. The only shipped min-plus is the method
> `bgz17::ScalarCsr::spmv_min_plus` (`fn(&self, x: &[f32]) -> Vec<f32>`) —
> a lossy-sibling prototype, never a substitute for the blasgraph canon.
> The free function `tropical_spmv(edge_weights, dag)` cited previously
> does not exist.

**R-15 (new candidate)** — signature-basis as Basis<T> impl:

> R-15 (candidate): the substrate supports path-structured signals via `sigker::SignatureBasis<DEPTH>: Basis<f32>`, alongside `DctIIBasis<N>: Basis<i16>` (video) and `EwaSplatBasis: Basis<f16>` (3DGS). Implementation: ~1 week wrapper around `sigker::signature_truncated` (the form Pillar 11 cites; the PDE form is equally sound — audit #9 — but the truncated path is what R-15 commits). **Plan G** gets a fifth lane (path-structured: audio waveform, time-series, gesture/handwriting).

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
- **Source code references (in this repo `adaworldapi/ndarray`):**
  - `src/hpc/cam_pq.rs` — the codebook trainer
  - `src/hpc/dn_tree.rs` — quaternary plastic memory
  - `src/hpc/merkle_tree.rs` — Blake3-48-bit Merkle
- **Source code references (external repo `adaworldapi/lance-graph`):**
  - `crates/sigker/` — Chen-Lyons signatures
  - `crates/sigker/src/` — `signature_kernel_pde`, `RandomizedSignature`, `CodecRouteSigker`
  - `crates/jc/src/hambly_lyons.rs` — Pillar 11 (active under `--features hambly-lyons`; DEFERRED only in default zero-dep build)
  - `crates/jc/src/pflug.rs` — Pillar 10 (nested-distance Lipschitz on Sigma DN-trees, certifies CAM-PQ)
  - `crates/bgz-tensor/src/adaptive_codec.rs` — cam_pq imports
- **arXiv anchors for sigker:**
  - **2006.14794** (Salvi-Cass-Foster-Lyons-Lemercier 2020) — Goursat PDE for signature kernel
  - Hambly-Lyons 2010 — signature uniqueness theorem
  - Cuchiero-Schmocker-Teichmann 2021 — randomized-signature universality
- **arXiv anchor for dn_tree:**
  - "On Demand Memory Specialization for Distributed Graph Processing" (2013)

_Last edit: 2026-05-22._
