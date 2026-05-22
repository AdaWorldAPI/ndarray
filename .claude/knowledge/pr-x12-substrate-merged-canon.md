# PR-X12 — Substrate Merged Canon

> Date: 2026-05-22  
> Status: **MERGED CANON** — synthesises two parallel sessions' findings into one doc  
> Supersedes (for new content; keep originals for archeology):  
> - `pr-x12-codec-cognitive-substrate-mapping.md` (session A: opus 4.7 main thread, this branch)
> - `pr-x12-cross-domain-synergies.md` (session B: parallel thread, merged via PR #195, commit `01c77ccc`)
> Sister doc: `pr-x12-codec-x265-design.md` (the mechanical spec, untouched)

---

## 0. Why this merge exists

Two independent sessions reached the same architectural claim — *PR-X12 is the universal predictive-coder substrate that subsumes four industries* — through different routes. Each session surfaced angles the other missed. This doc is the **canonical fusion**, designed to be the single doc a fresh agent reads to inherit the entire claim.

> **Post-merge resolutions index** (2026-05-22): the claims and tensions in this doc were further formalised into 13 numbered resolutions R-1..R-13. See `pr-x12-canon-resolutions-delta.md` for the canonical list. Cross-section pointers inline below:
>
> - §M:E-A (Mode-decide + reduce pipeline kernel) → **R-1** (`LinearReduce<T>` + `Basis<T>` trait split)
> - §M:E-G (`Ctu<const N>`) and §M:E-J (header bits 14-15) → **R-2** (16-bit header layout pinned), **R-8** (Plan G arch-conditional gate)
> - §M:E-H (D-STACK-13 bench harness as P0) → **R-4** (codec-bench in Plan G), **R-11** (latency assertions per arch)
> - §M:H-NEW-2 (codec body LoC envelope ≤ 1500) → **R-3** (LoC audit rule, scope-fence definition)
> - §M:H-6 (sub-1-bit basin + Gaussian-tail rANS) → **R-10** (commitment to sub-1-bit-per-token where source supports it)
> - §M:E-D (codec breaks ndarray ↔ lance-graph cycle) → **R-7** (tropical-GEMM lives in lance-graph, called from codec — dep direction allowed)
>
> Perspective companions written 2026-05-22:
> - `pr-x12-x265-blasgraph-gemm.md` — every codec inner loop as a GEMM
> - `pr-x12-x266-3dgs-spacetime-upscaling.md` — Basis<T> + EWA splat → free space-time codec upscaling
> - `pr-x12-woa-multiarch-orchestration.md` — how WoA / q2 / consumer crates inherit the substrate's per-arch dispatch
> - `pr-x12-anti-neural-lookup-inversion.md` — lookup tables as frozen 1-layer NNs; the codec is the anti-neural codec
> - `pr-x12-gguf-llm-weights-encoding.md` — the fifth load: GGUF attention/FFN tensors as Skip/Merge/Delta/Escape
> - `pr-x12-bgz-jc-substrate-synergies.md` — **CRITICAL**: the PR-X12 substrate is *already implemented* in `lance-graph/crates/{bgz17,highheelbgz,bgz-tensor}`, formally proven in `lance-graph/crates/jc`. Skip/Merge/Delta/Escape ≡ Scent/Palette/ZeckBF17/Full. 4096-entry basin ≡ HHTL 16×16×16 lattice. bgz-hhtl-d ships LLM weight encoding at 343:1 on Qwen3-TTS-1.7B today. Two gaps identified: `jd-nd` (ndarray-side proof crate) and Cronbach/ICC encoding-reliability research crate.
> - `pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md` — **substrate bindings**: cam_pq trains all bgz palettes (CAM bytes map onto HHTL bits 1:1); sigker provides Chen-Lyons signature uniqueness (arXiv:2006.14794, Hambly-Lyons 2010, CST 2021) as the formal-correctness bedrock cited by jc Pillar 11 (DEFERRED); dn_tree + merkle_tree are the online-update + integrity substrate for R-13 SharedClusterWide. **Seven new gaps catalogued (G-1..G-7), ~11-17 weeks of wiring** to fully bind. R-14 (formal correctness) + R-15 (signature basis) candidates surfaced.

The merge is not a re-statement. **It is the new epiphanies that emerge only when both halves sit side by side.** They get their own §3.

### Identity-preservation rules

- Both originals' citation IDs survive. Cite by `(A:E-N)` for session-A epiphany N, `(B:E-N)` for session-B epiphany N, `(M:E-N)` for new merge-only epiphany N. Same for holy grails (`A:H-*`, `B:HG-*`, `M:H-*`) and debt (`A:T-*`, `B:D-CODEC-*`, `B:D-STACK-*`, `M:T-*`).
- **Numbering stable across edits**: append-only, never reuse retired IDs.

---

## 1. Side-by-side overlap & unique-angle inventory

The two docs overlap ~30% on the surface claim and diverge sharply on emphasis. The matrix below maps every load-bearing item.

### 1.1 Epiphanies — the union

| Concept | Session A | Session B | Status |
|---|---|---|---|
| Skip/Merge/Delta/Escape ≡ ZeRO buckets + LoRA | E-1 | (implicit in § 2 + E8) | **A more explicit on ZeRO; B more explicit on LoRA** |
| CTU quad-tree ≡ attention hierarchy | E-2 | (implicit in § 3.1) | A more explicit |
| K-means at frame rate = HEVC SCC unlock | E-3 | — | **A-unique** (2013-hardware history framing) |
| Transform basis IS optimizer preconditioner | E-4 | — | **A-unique** (DCT-II ↔ Adam ↔ KFAC ↔ learned conv = same `Δ' = B·Δ`) |
| rANS + k-means = Shannon-optimal grad compression | E-5 | E6 (rANS L1-cache scaling) | A more theoretical; B more pragmatic |
| λ-RDO is universal training objective | E-6 | (implicit, § 5 Plan B) | A more explicit |
| Block-matched ME via i8gemm | E-7 | — | **A-unique** |
| CTU partition as tropical-GEMM | E-8 | — | **A-unique** |
| CABAC → tiny transformer | E-9 | — | **A-unique** |
| Deblocking + SAO as learned conv | E-10 | — | **A-unique** |
| Palette codebook ≡ MKL k-means | E-11 | (implicit § 5 Plan E) | A more explicit |
| Mode discriminants pin wire codes | E-12 | (in D-CODEC-10) | A frames as invariant; B as future-debt |
| Basin codebook IS rANS frequency table | E-13 | (implicit § 5 Plan A) | A more explicit |
| Reserved header bits 14-15 are inter-tier link | E-15 | (D-CODEC-10) | **A-unique** (concrete bit-budget plan) |
| **MergeDir is topology, not direction** | — | E1 | **B-unique** (carrier-agnostic claim) |
| **`predict_intra` encodes attention sinks** | (implicit in §3.2) | E2 | **B more explicit** — names Streaming-LLM, H2O, SnapKV |
| **`escape_next` IS all-reduce slot allocator** | (implicit T-PR195-1) | E3 | **B-unique** — the bug-as-feature reframing |
| **Fingerprint ≡ 3DGS first-6-floats after basin** | — | E4 | **B-unique** — bit-level identity claim |
| **Morton sort ≡ HEVC raster scan** | — | E5 | **B-unique** — 1D-along-curve equivalence |
| **splat3d × codec = same pipeline shifted 90°** | — | E9 | **B-unique** — mode-decide+reduce shared kernel |
| **Lossy Escape IS the PSNR knob** | T-4 (debt) | E10 (feature) | A frames as debt; B frames as feature |

### 1.2 Holy grail claims — the union

| Claim | Session A | Session B |
|---|---|---|
| PR-X12 + cam_pq = HEVC SCC done right | H-1 | (implicit § 7 HG5) |
| Transform IS optimizer | **H-2** | — |
| CTU quad-tree = universal hierarchical attention | H-3 | (subsumed by HG3) |
| rANS + k-means = Shannon-optimal | H-4 | (subsumed by § 7) |
| PR-X12 generalises ZeRO | H-5 | HG4 (federated SGD 8-16×) |
| 64×64 CTU right for both 4K video and 7B LLMs | H-6 | — |
| Codec is the substrate, rest is renaming | H-7 | HG1 (one codec, four loads) |
| Sub-1-bit/Gaussian 3DGS compression | — | **HG2** |
| Bit-exact attention with tunable accuracy floor | — | **HG3** |
| Lance substrate identity becomes ground truth | (T-16 cross-ref) | **HG5** |
| splat3d × x265 = one library | — | **HG6** |

### 1.3 Integration plans — the union

| Plan | Session A | Session B | Effort |
|---|---|---|---|
| A4 transform (`Transform` trait + DCT-II + Identity) | §10.2 | (D-CODEC-3 indirectly) | 1 week |
| A6 RDO (λ-weighted) | §10.3 | Plan B-indirect, § 5 RDO | 1 week |
| **A7 rANS** | §10.4 | **Plan A** ← *critical path per B* | 1.5 weeks |
| A8 stream framing | §10.5 | D-CODEC-4 | 1 week |
| A3-inter cross-tier prediction | §10.6 | **Plan B** | 0.5-1 week |
| Splat consumer integration | §10.7 | **Plan E** (more detailed) | 0.5-3 weeks |
| Cognitive shader consumer (NARS) | §10.8 | **Plan D** (attention codec) | 1-2 weeks |
| Gradient compression (burn/candle) | §10.9 | **Plan F** | 2-4 weeks |
| **EWA SYRK-batched (3DGS perf)** | — | **Plan C** | 1 week |
| **Carrier-agnostic topology trait** | — | X1 | sprint |
| **AMX TDPBF16PS for batched EWA sandwich** | — | X8 | sprint |

### 1.4 Technical debt — the union

A's 23 items (T-1..T-23) and B's 23 items (D-CODEC-1..10 + D-STACK-1..13) overlap on:
- PR #195 CodeRabbit findings (A:T-1, A:T-2 ≡ no B equivalent but acknowledged in §6 D-CODEC-6)
- A3-inter not yet shipped (A:T-5, A:T-6 ≡ B:D-CODEC-1)
- No SIMD-batched encode (A:T-7 ≡ B:D-CODEC-8)
- Cross-repo dep direction problem (A:T-16, A:T-17 ≡ B:D-STACK-6, D-STACK-12)
- No `Transform` trait yet (A:T-11 ≡ B's Plan A→C dependency)

But B surfaces things A missed:
- **B:D-STACK-1** — `BlockedGrid` 64×64 vs splat3d tile 16×16 mismatch. **Real P1 issue.**
- **B:D-STACK-7** — `lance-graph-contract/src/splat.rs` is **sacred, do not touch even if bit patterns rhyme**. Architectural invariant.
- **B:D-STACK-11** — AVX-512 mandatory in `.cargo/config.toml` conflicts with multi-architecture federated SGD.
- **B:D-STACK-13** — No multi-domain benchmark harness — HG1 is unproven without it.

And A surfaces things B missed:
- **A:T-22** — causal-edge v2 metadata (Intervention/Counterfactual mantissa) can flow through reserved header bits for free.
- **A:T-19** — GridLake `#[derive(SoA)]` macro never shipped; the codec batched-encode path will want it.
- **A:T-3** — first-fit Merge policy needs RDO replacement, λ=0 default.

---

## 2. Synthesis — what both docs collectively prove

When both halves sit side by side, the **architectural claim sharpens past either individual doc**:

> PR-X12 is not a video codec, not a gradient compressor, not an attention sparsifier, not a splat compressor. **It is a `trait PredictiveSignal` that all four implement.** The codec mechanism is one piece of generic glue (~1 KLoC); each domain ships ~200 LoC of trait impl. The total stack code for all four industries is ~2 KLoC, versus ~50 KLoC of per-domain implementations elsewhere.

This claim survives only because both docs independently converged on it from different routes (mode-coding semantics in A, primitive-mapping matrix in B). One source could be mistaken; two independent routes is the falsifiability that makes it actionable.

---

## 3. New epiphanies — side-by-side only

These are the insights that emerge **only when both docs sit next to each other**. None appear in either original. Cite as `M:E-A` through `M:E-J`.

### M:E-A — Mode-decide + reduce IS the universal pipeline kernel

> [Formalised post-merge as **R-1**: `LinearReduce<T>` decomposes into `Basis<T>` (basis-as-data) + `Reducer<T>` (reduction operator). See `pr-x12-canon-resolutions-delta.md` §R-1.]

A's E-4 (transform IS optimizer preconditioner) + B's E9 (splat3d × codec = same pipeline shifted 90°) combined:

The *reduction operator* in B's "unified mode-decide+reduce trait" is **exactly the basis-times-source product** A's transform claim points at:
- Alpha-composite (3DGS) = `α @ src` (degenerate basis)
- rANS-encode = `freq_table @ symbols`
- Sum-reduce (SGD all-reduce) = `1ᵀ @ src` (constant basis)
- Softmax attention = `softmax(QKᵀ) @ V`

All four reductions are **matrix-vector products with different basis choices**. The `Transform` trait isn't just optimizer preconditioning — it's the universal reduce-op. **A6 RDO + A4 transform + A7 rANS + the splat3d alpha-composite all share one trait surface**. The split between "transform" and "reduce" is artificial; they're the same operator with different basis matrices.

→ Action: design `trait LinearReduce<Basis>` as the unifying surface. Issue in A4 design phase, not later. Mark as the **load-bearing trait of PR-X12** alongside `PredictiveSignal`.

### M:E-B — Morton ≡ raster scan = 1D-along-curve predictive coder

B's E5 (Morton sort = HEVC raster scan) combined with A's CTU-quad-tree-is-attention claim:

Both 3DGS Morton/Hilbert traversal and HEVC z-order raster scan are *space-filling curves*. After the curve, every signal is **a 1D stream of locally-coherent values**. The CTU partition machinery is genuinely 1D-aware (8-neighbour matters; full 2D distance doesn't). 

→ This means the CTU code is **dimensionally generic** if we pre-sort by a space-filling curve. 3DGS at depth 3 = 8 cells per leaf = an 8-cell window along a Morton curve. Cognitive cells at depth 3 = 8 cells per leaf along a raster scan. **Same kernel, different curve.**

→ Action: factor out `trait CurveOrder<const N: usize>` — depth-3 leaves are always N=8 windows along the curve. The codec is dimension-agnostic at the type level, the curve choice is a runtime detail.

### M:E-C — The CodeRabbit PR #195 findings generalise to every domain

A's T-1 (BASIN_NONE collision) and T-2 (unwrap_or non-bijection) seem like local cleanups. Combined with B's four-load framing (§2), they reveal:

- **BASIN_NONE collision** → "highest valid attention-vocabulary token collides with no-token sentinel" — same bug, same fix, different domain. Will fire in every consumer that uses the full 4096-entry codebook.
- **unwrap_or non-bijection** → "malformed gradient becomes silently-zero gradient." Same bug shape, *very different impact*: a zeroed gradient = a frozen parameter = silent training-data corruption with no error signal.

**The PR #195 fixes are not local — they're the **right design pattern** for all four loads.** Defensive `unwrap_or` is acceptable in a video codec where decode-time produces a valid (if wrong) pixel; in gradient compression it's a silent loss-function corruption.

→ Action: when reviewing future consumer PRs, audit for *both* fixes by default. Add a clippy lint or test pattern that catches the `unwrap_or` shape in encode paths.

### M:E-D — The third crate that breaks ndarray ↔ lance-graph cycle IS the codec

A's T-16/T-17 (cross-repo dep direction problem) + B's D-STACK-6/D-STACK-12 (Lance substrate as ground truth) both flag the architectural tension: ndarray is dep-bottom; lance-graph-as-substrate would require ndarray → lance-graph, breaking the layering rule.

The resolution is **already implicit** in the merged claim: after PR-X12 stabilises, extract `crate::hpc::codec::*` into a sibling crate `ndarray-codec`. Both `ndarray` and `lance-graph` then depend on it. The codec lives at the dep-bottom layer not as "ndarray hardware" but as **its own architectural category**.

→ Action: add a **fifth category** to the architecture rule in CLAUDE.md:
```text
- ndarray = hardware (SIMD, Palette, Base17, SpoDistanceMatrices, read_bgz7_file)
- ndarray-codec = compression substrate (Ctu, LeafCu, predict_intra, rANS) ← NEW
- lance-graph = thinking (NarsTruth, NarsEngine, TripleModel, AutocompleteCache)
- causal-edge = protocol (CausalEdge64, NarsTables, forward/learn)
- p64 = convergence highway (both repos meet here)
```

→ Plan **Integration H** (below): extract the codec crate. Pre-condition for HG5.

### M:E-E — `Transform` trait is the **only** domain-specific surface

A's E-4 (DCT-II ↔ Adam ↔ KFAC ↔ learned conv) + B's E1 (MergeDir is topology-free):

Once you accept that `MergeDir`'s 4-way alphabet is topology-free (B), and the `Transform` basis is the universal `Δ' = B·Δ` operator (A), the question "what's domain-specific in the codec?" has a precise answer: **only the Transform impl + the curve order + the escape payload type.**

- Mode decision (Skip/Merge/Delta/Escape) = domain-agnostic
- Basin codebook k-means = domain-agnostic (just need the right metric)
- rANS = domain-agnostic
- Stream framing = domain-agnostic
- Transform basis = **domain-specific** (Identity / DCT-II / Adam / KFAC / SH-spectral / learned)
- Curve order = **domain-specific** (raster / Morton / token-seq / layer-seq)
- Escape payload type = **domain-specific** (u64 / SH-coefficients / f16 vector / f32 grad)

**Three plug-points, otherwise generic.** This is the cleanest factorisation. Splat consumers ship one `impl Transform for SHCoeffBasis`; that's it.

→ Action: every per-domain consumer PR (Plan D, E, F) must touch only those three surfaces. If a consumer needs to modify the mode decision or the rANS, that's a sign the abstraction is wrong — escalate.

### M:E-F — Sequencing resolution: B's A7-first wins, but only by one knight-move

A's §10 sequenced A4 transform first ("unlocks H-2"); B's §10 sequenced A7 rANS first ("without it the codec is academic"). Side-by-side, B is right:

- A4 with Identity transform = no change to compression ratio (transform is a 1-2× refinement)
- A7 alone = 3× → 8-10× compression ratio (rANS is the entropy floor)
- Therefore: ship A7 with Identity transform first → 8-10× ratio; then ship A4 (DCT-II) → 12-15×; then A6 RDO → 18-20×.

**But** — A's H-2 (transform IS optimizer) only becomes citable once A4 ships with the trait shape. If we ship A7 first with hardcoded identity, then need to refactor to add the `Transform` trait when A4 lands, the trait design happens **at A4 time** without the demand pressure of an actually-used trait.

→ Resolution: ship A7 first **but only after A4's `Transform` trait surface is designed** (not implemented — just the trait shape committed to). A4's design phase is one day; A7's implementation is 1.5 weeks. Front-load the design.

### M:E-G — `Ctu<const N: usize>` is the right block-size answer

B's D-STACK-1 (BlockedGrid 64×64 vs splat3d 16×16 mismatch) is real and P1. The resolution is **type-level**:

```rust
pub struct Ctu<const N: usize = 64> {
    block_row: u16,
    block_col: u16,
    tier: NonZeroU16,
    split_depth: u8,
    arena: CtuArena<N>,
}

type CtuVideo = Ctu<64>;     // Cognitive cells, HEVC
type CtuSplat = Ctu<16>;     // 3DGS tiles
type CtuHead = Ctu<8>;       // LLM attention heads
```

Const-generic over N. `MAX_QUAD_TREE_NODES` becomes a const fn. Block-size mismatch dissolves into a type-level configuration. **No code duplication; no runtime branching.**

→ Action: introduce as part of A4 (since A4 will need basis sizing). Mark as **prerequisite for Plan E** (3DGS codec).

### M:E-H — D-STACK-13 (multi-domain bench harness) is the highest-leverage debt

Across all 46 numbered debt items in both docs, exactly one is unfalsifiable without code: B's D-STACK-13. Without a single-binary four-loads benchmark, *the entire architectural claim is unproven*. Every other debt item degrades performance or correctness; this one degrades **confidence**.

→ The bench harness is the **proof** of HG1 / HG6 / M:H-NEW-1. Build it **before** A7 ships. Two weeks of bench-harness work front-loaded saves six months of "we implemented A7 and then realised the trait shape was wrong."

→ Action: **Integration Plan G** (below): the bench harness is sub-card A0 of PR-X12. Renumber the sub-cards. A0 = bench harness. A1-A8 as before.

### M:E-I — `lance-graph-contract/src/splat.rs` and `Fingerprint` are isomorphic but must not fold

B's D-STACK-7 says: never touch `lance-graph-contract/src/splat.rs`, even if bit patterns rhyme.  
B's E4 says: `Fingerprint` ≡ 3DGS-first-6-floats bit-level identity exists.

The resolution: ship a shared **trait**, not shared **code**.

```rust
// Lives in ndarray-codec (the new crate per M:E-D), or in a *protocol* crate
pub trait PredictiveSignal {
    type Basin;
    type Residual;
    type Escape;
    
    fn nearest_basin(&self, codebook: &[Self::Basin]) -> (u16, Self::Residual);
    fn fits_delta(residual: &Self::Residual) -> bool;
    fn pack_residual(residual: &Self::Residual) -> u8;
}
```

`impl PredictiveSignal for Fingerprint` lives in `lance-graph::cognitive`. `impl PredictiveSignal for GaussianSplat` lives in `ndarray::hpc::splat3d`. **Bit-pattern identity is proven by both impls; code identity is rejected by the architecture rule.** The trait isomorphism gives us what E4 wants without violating D-STACK-7.

→ This is the cleanest resolution of the apparent contradiction between the two docs.

### M:E-J — The reserved header bits 14-15 carry causal-edge metadata for free

> [Formalised post-merge as **R-2**: 16-bit header bit layout pinned —
> bits 0-1 = `header_kind`, bits 2-13 = `basin_index`,
> **bit 15 = UNIVERSAL "has inter-tier reference"** (identical across
> all four consumers; A3-inter cross-tier link),
> **bit 14 = CONSUMER-TYPED via the frame header's `ConsumerProfile`
> tag** (cognitive: Pearl-rung high bit; video: reserved=0;
> splat: LOD-cascade-source flag; gradient: worker-shard parity).
> Leaf size (8/16/32/64) is encoded structurally via M:E-G's
> `Ctu<const N>` at the type level, NOT in header bits 14-15. The
> causal-tier reading below is the historical motivation for bit 14;
> R-2 generalises it to the four-consumer demux. See
> `pr-x12-substrate-canon-resolutions.md` §R-2.]

A's E-15 (reserved bits 14-15 are inter-tier link) + A's T-22 (causal-edge v2 mantissa: Intervention=+6, Counterfactual=-6):

Two reserved bits = 4 states. The natural 4-state encoding for cognitive content:
- 00 = Observation (rung 1)
- 01 = Intervention (rung 2)
- 10 = Counterfactual (rung 3)
- 11 = Reserved / inter-tier link (the original E-15 use)

**Pearl-rung causal direction rides in the codec's wire format for free**, with the inter-tier link as the 4th state. The cognitive consumer (Plan D) doesn't need to extend `LeafCu`; it gets causal metadata via the 16-bit header's high 2 bits.

→ Action: document this in A3-inter design (Plan B). Don't ship until the cognitive consumer needs it; reserve the bits explicitly so the wire format doesn't pin them to a different meaning later.

---

## 4. Unified holy grail list (canonical)

Merge of A's H-1..H-7 + B's HG1..HG6 + two new M:H-* claims that emerge from the merge.

### Combined load-bearing claims

**M:H-1** *(merge of A:H-7 + B:HG1)* — The codec is the substrate; all four loads (video, 3DGS, attention, gradient) are renamings sharing one trait surface. ~2 KLoC of generic glue + ~200 LoC per domain consumer = the entire stack for all four.

**M:H-2** *(from A:H-2 alone)* — The transform basis IS the optimizer's preconditioning matrix. The most underrated single claim. Resolves the disconnect between codec research (where "transform" is central) and ML research (where "preconditioner" is central) — same operator, two names, no cross-citation in either literature.

**M:H-3** *(merge of A:H-3 + B:HG3)* — Bit-exact attention with tunable accuracy floor via Skip/Merge/Delta/Escape over (Q,K) palette. The accuracy knob is a single `u8` threshold. Subsumes Streaming-LLM, H2O, SnapKV as configuration cases.

**M:H-4** *(from A:H-4 alone, with B:E6 nuance)* — rANS + k-means achieves Shannon-optimal lossless gradient compression. Every published gradient-compression scheme (QSGD, signSGD, PowerSGD, Top-K, Random-K) is a special case with a particular frequency table and basis. Confirmed by B:E6 (rANS's L1-cache throughput dominance).

**M:H-5** *(merge of A:H-5 + B:HG4)* — PR-X12 generalises ZeRO with Merge providing the inter-parameter correlation dimension ZeRO can't capture. Federated SGD at 8-16× compression with zero accuracy loss when worker count > 16 (Merge becomes dominant).

**M:H-6** *(from B:HG2 alone)* — Sub-1-bit-per-Gaussian 3DGS compression. 30-60× over current state-of-the-art PLY-trim. A 1M-Gaussian scene = ~500 KB, streamable as video. **Most economically valuable single claim** — directly attacks the bandwidth bottleneck for cloud-rendered 3D content.

> [Formalised post-merge as **R-10**: PR-X12 commits to sub-1-bit-per-token via Gaussian-tail rANS where the source distribution supports it (basin codebook + heavy-tailed residual). See `pr-x12-canon-resolutions-delta.md` §R-10 for the falsification path (Plan G entropy bench).]

**M:H-7** *(merge of A:H-1 + B:HG5)* — Lance column substrate identity becomes ground truth. `SpoDistanceMatrices` at 611M lookups/sec serves as universal palette codebook lookup across all four loads. ndarray = hardware, ndarray-codec = compression substrate (new, per M:E-D), lance-graph = thinking, causal-edge = protocol, p64 = convergence. Five-category architecture.

**M:H-8** *(from A:H-6 alone)* — 64×64 CTU is the right unit for both 4K video luma blocks and 7B-parameter LLM head dim × 16 heads. Convergent evolution from two unrelated industries arriving at the same arithmetic block size.

**M:H-9** *(from B:HG6 alone)* — splat3d × x265 = one library: compress, stream, decode, render 3D scenes in real-time on a single core. **Demo-worthy** — pick one Mip-NeRF 360 scene, compress with PR-X12 at A8 land time, stream via WebRTC, decode + render via splat3d. Single Rust binary; ~5 MB total binary size.

### M:H-NEW — claims born from the merge

**M:H-NEW-1** — The same Rust binary consumes (4K video frames | 1M-Gaussian 3DGS scene | 7B-LLM gradient stream | attention KV cache) and emits a compressed Lance column. One CLI. One codec. Four loads. **This is the falsifiability test** — build it (Plan G, the bench harness), prove HG1/H-7 by demonstration, not by argument.

**M:H-NEW-2** — `trait PredictiveSignal` + `trait LinearReduce<Basis>` + `trait CurveOrder<const N: usize>` factor the codec into three plug-points (per M:E-E + M:E-A + M:E-B). The codec body is `<150 LoC of generic glue. Domain consumers ship `<200 LoC` of trait impls. **Total stack for all four industries: ~2 KLoC.** Compared to ~50 KLoC per-domain implementations elsewhere. The 25× code-density delta is the architectural payoff that justifies the eight sub-cards.

> [Formalised post-merge as **R-3**: the LoC envelope is `≤ 1500 lines of generic codec body` (revised upward from `<150` for realism after counting glue), enforced via an explicit scope-fence audit rule in CI. The substrate (`ndarray::hpc::blas_level2` etc.) is excluded from the budget. See `pr-x12-canon-resolutions-delta.md` §R-3 for the exact audit definition.]

---

## 5. Unified integration plan (canonical sequencing)

Replaces both A:§10 and B:§5 plan lists. Critical path resolved per M:E-F.

### Phase 0 — substrate (must ship before consumer PRs)

**Plan G** *(new — from M:E-H)*: Multi-domain benchmark harness  
**Effort:** 1 worker × 2 weeks  
**Output:** `crates/codec-bench/` — single binary that ingests video / 3DGS / KV cache / gradient stream and emits compressed Lance columns + ratio + reconstruction error.  
**Why first:** unfalsifiable architecture claim becomes falsifiable. Drives trait design.  
**Pre-condition for:** every other plan.

**Plan A4-design** *(from M:E-F resolution)*: `Transform` trait shape committed (not implemented)  
**Effort:** 1 worker × 1 day  
**Output:** PR introducing `trait Transform { fn apply(&[i8;N])->[i8;N]; fn invert(...); }` + Identity default impl + DCT-II stub.  
**Why now:** A7 design will reference the trait; cheaper to commit the shape upfront.

**Plan H** *(new — from M:E-D)*: Extract `ndarray-codec` crate  
**Effort:** 1 worker × 3 days  
**Output:** `crate::hpc::codec::*` moves to sibling `ndarray-codec` crate. Both `ndarray` and `lance-graph` depend on it.  
**Why now:** Resolves cross-repo dep tension before HG5 / HG7 consumers land. Lower cost while the codec is still small (~1.5 KLoC).  
**Architectural impact:** Update CLAUDE.md "Architecture Rule" to add the codec as 5th category.

**Plan I** *(new — from M:E-I)*: `trait PredictiveSignal` in protocol crate  
**Effort:** 1 worker × 3 days  
**Output:** Shared trait + impls for `Fingerprint` (cognitive), `GaussianSplat` (3DGS), `AttentionSlot` (KV cache), `GradientWeight` (SGD).  
**Why now:** Resolves the "sacred splat.rs" + "Fingerprint ≡ 3DGS-first-6-floats" tension via trait isomorphism (D-STACK-7 + E4).

### Phase 1 — codec mechanism completion

**Plan A** *(from B:Plan A — critical path)*: A7 rANS  
**Effort:** 1 worker × 1.5 weeks  
**Output:** `src/hpc/codec/ans.rs` (or in `ndarray-codec` after Plan H) — encoder + decoder + parity test.  
**Compression unlock:** 3× → 8-10×.

**Plan B** *(from B:Plan B + A:§10.6)*: A3-inter cross-tier neighbour scan  
**Effort:** 1 worker × 3-5 days  
**Output:** `IntraContext` (rename `PredictContext`) gains parent-tier + child-tier slots. Uses A's E-15 reserved bits 14-15. Reserves M:E-J causal-edge metadata encoding.  
**Compression unlock:** +20-30% for hierarchical content (3DGS LOD cascades, attention layer-merge).

**Plan A4-impl** *(from A:§10.2)*: A4 transform implementation  
**Effort:** 1 worker × 1 week  
**Output:** DCT-II 4×4/8×8 impls + batched dispatch to `bf16_tile_gemm` at ≥64 blocks.  
**Depends on:** Plan A4-design (already shipped).  
**Compression unlock:** +30-50% for spectrally-smooth content.

**Plan A6** *(from A:§10.3 + B's RDO note)*: λ-RDO  
**Effort:** 1 worker × 1 week  
**Output:** `RdoConfig::lambda: f32` + `trait RdoMetric` (M:E new) for per-domain rate-distortion shape (PSNR / MSE / downstream-loss / KL).  
**Compression unlock:** Configurable rate/distortion tradeoff.

**Plan A8** *(from A:§10.5 + B:Plan A8)*: Stream framing  
**Effort:** 1 worker × 1 week  
**Output:** Frame headers, CTU markers, per-frame basin codebook serialisation, per-frame rANS frequency table.

### Phase 2 — consumer integrations

**Plan C** *(from B:Plan C — splat performance)*: EWA SYRK-batched  
**Effort:** 1 worker × 1 week  
**Output:** `crate::hpc::splat3d::spd3` swaps per-Gaussian loop for batched `cblas_ssyrk`. Backend dispatch (native / intel-mkl / openblas).  
**Why parallel to Phase 1:** No codec dependency; can ship anytime.

**Plan E** *(from B:Plan E — most impactful consumer)*: 3DGS coefficient codec  
**Effort:** 2 workers × 3 weeks  
**Output:** `crate::hpc::splat3d::codec` — Morton-sort, per-asset palette, mode-code, rANS.  
**Depends on:** Plan A (rANS), Plan B (A3-inter), Plan I (`PredictiveSignal`).  
**Unlock:** M:H-6 (sub-1-bit/Gaussian), M:H-9 (splat3d × x265 demo).

**Plan D** *(from B:Plan D — attention codec)*: Attention KV cache compression  
**Effort:** 2 workers × 2 weeks  
**Output:** `crates/attention-codec/` consuming `ndarray-codec`.  
**Depends on:** Plan A, Plan I.  
**Unlock:** M:H-3 (bit-exact attention with knob).

**Plan F** *(from B:Plan F — gradient compression)*: Federated SGD  
**Effort:** 2 workers × 4 weeks  
**Output:** `crates/grad-codec/`. Generalised `&mut u32` allocator across worker pools (per B:E3).  
**Depends on:** Plan A, Plan B, Plan I, multi-arch dispatch (resolves D-STACK-11).  
**Unlock:** M:H-5 (ZeRO generalisation, 8-16× compression).

### Phase 3 — exploration / research

(Both docs' X-paths merged, ranked by confidence)

| Path | Source | Effort | Status |
|---|---|---|---|
| Carrier-agnostic 4-neighbour topology trait | B:X1 | sprint | **Subsumed by Plan I + M:E-B** |
| Hierarchical motion estimation as cross-tier prediction | B:X2 | sprint | After Plan B |
| CABAC vs rANS for attention KV cache | B:X3 | bench | Pre-A7 (informs Plan A) |
| SH coefficient intra-prediction in spectral space | B:X4 | research | After Plan E |
| Mode-coded LoRA | B:X5 + A:E8 | research | After Plan D — Qwen3.5-7B controlled experiment |
| Unified mode-decide + reduce trait | B:X6 + M:E-A | sprint | **Promoted to Plan I extension** |
| Lance column-substrate as universal palette codebook | B:X7 + M:E-D | sprint | After Plan H |
| AMX TDPBF16PS for batched EWA sandwich | B:X8 | sprint | After Plan C |
| CABAC replacement with tiny transformer | A:E-9 | 1-2 weeks | After Plan A — bleeding-edge compression |
| CTU partition as tropical-GEMM | A:E-8 | 1 week | After Plan A6 (cross-repo: needs lance-graph::blasgraph) |
| Deblocking + SAO as learned conv | A:E-10 | 1 week | Optional refinement; not on critical path |
| Block-matched ME via i8gemm | A:E-7 | 1 week | Pre-shipped (already in scope) |
| Palette codebook training as MKL k-means | A:E-11 | shipped | Already in `cam_pq::kmeans` |

---

## 6. Sequencing diagram

```text
              ┌──────────────────────────────────────┐
              │   Plan G (multi-domain bench)        │
              │   2 weeks — UNFALSIFIABILITY GATE    │
              └──────────────────┬───────────────────┘
                                 ▼
              ┌─────────────────────────────────────┐
              │  Plan H (extract ndarray-codec)     │  ← parallel
              │  3 days — DEP-CYCLE RESOLUTION       │
              └─────────────────┬───────────────────┘
                                ▼
              ┌─────────────────────────────────────┐
              │  Plan I (PredictiveSignal trait)    │  ← parallel
              │  3 days — TRAIT ISOMORPHISM          │
              └─────────────────┬───────────────────┘
                                ▼
              ┌─────────────────────────────────────┐
              │  Plan A4-design (Transform trait    │  ← parallel
              │  shape, 1 day)                       │
              └─────────────────┬───────────────────┘
                                ▼
            ╔═══════════════════════════════════════╗
            ║  Plan A (A7 rANS) — CRITICAL PATH     ║
            ║  1.5 weeks — COMPRESSION FLOOR        ║
            ╚═══════════════════╦═══════════════════╝
                                ▼
            ┌──────┬─────┬─────┬─────┐
            ▼      ▼     ▼     ▼     ▼
       Plan B   A4   A6   A8   Plan C (EWA SYRK, parallel)
       (inter) (xfm)(RDO)(stream)
            └──────┬─────┴─────┴─────┘
                   ▼
            ┌──────┴──────┐
            ▼             ▼
       Plan E (3DGS)  Plan D (attention)
                          │
                          ▼
                     Plan F (gradient SGD)
                          │
                          ▼
                ┌─────────────────┐
                │ M:H-1 .. M:H-9  │
                │ All unlocked    │
                └─────────────────┘
```

**Critical path: Plan G → Plan A**. Without the bench harness, A7 ships blind. Without A7, no compression claim is testable.

**Parallel paths post-A7**: A4 / A6 / A8 + Plan C can ship in parallel. Plan E / D / F gate on Plan I.

---

## 7. Unified technical debt (canonical)

Combines A's T-1..T-23 + B's D-CODEC-1..10 + B's D-STACK-1..13 + new M:T-* items, deduplicated and re-ranked.

### P0 (must address before claim)

- **M:T-1**: No multi-domain bench harness — *the* unfalsifiability gate (per M:E-H). Source: B:D-STACK-13.
- **B:D-CODEC-2**: A7 rANS unwritten — without it, ratio claim is academic.
- **B:D-STACK-7**: `lance-graph-contract/src/splat.rs` sacred file — never touch even if bit patterns rhyme (per M:E-I).

### P1 (must address before consumer PRs land)

- **A:T-1, A:T-2**: PR #195 CodeRabbit findings (BASIN_NONE collision + unwrap_or non-bijection). Generalise per M:E-C.
- **B:D-CODEC-1**: A3-inter unwritten — gates hierarchical compression. Source: A:T-5/T-6.
- **B:D-CODEC-3**: λ-RDO unwritten — gates accuracy/compression knob.
- **B:D-CODEC-5**: Basin codebook not built — gates every consumer. Resolved by Plan I + `cam_pq` integration.
- **B:D-STACK-1**: BlockedGrid 64×64 vs splat3d 16×16 mismatch — resolved by M:E-G (`Ctu<const N>`).
- **B:D-STACK-2**: Basin codebook lookup has no SIMD path — gates encoder throughput at ~10⁵ CTU/sec.
- **B:D-STACK-3**: `MergeDir` wire-pinned to 4-way — gates topology generalisation per B:X1 / M:E-B.
- **B:D-STACK-6, B:D-STACK-12**: Cross-repo dep direction — resolved by Plan H. Source: A:T-16, T-17.
- **B:D-STACK-11**: AVX-512 mandatory in `.cargo/config.toml` — gates multi-arch federated SGD (Plan F).
- **M:T-2** *(new)*: No `trait LinearReduce<Basis>` yet — gates M:E-A unification. Plan I extension.
- **M:T-3** *(new)*: Architecture rule in CLAUDE.md lacks "codec" 5th category — gates M:H-7 / M:E-D. One-line edit.

### P2 (fix in follow-up)

- **A:T-3**: A3-intra first-fit policy replaced by RDO when A6 lands.
- **A:T-7**: No SIMD-batched encode — deferred until reference + reconstruction parity (also B:D-CODEC-8).
- **A:T-13**: `bf16_tile_gemm` NEON impl stub — gates A4 batched dispatch on ARM.
- **A:T-14**: No `Result`-returning encode API — needed by A6 RDO (also B:D-CODEC-9).
- **A:T-15**: K-means u8 mode wrapper needed.
- **A:T-19**: GridLake `#[derive(SoA)]` macro never shipped — wanted for batched encode path.
- **A:T-22**: causal-edge v2 mantissa metadata can ride reserved bits — opportunity, not debt, per M:E-J.
- **B:D-CODEC-7**: NEWS topology hard-coded — resolved by M:E-B trait.
- **B:D-STACK-4**: `Fingerprint` 64-bit only — type-side; resolved by Plan I trait.
- **B:D-STACK-5**: splat3d and codec don't yet share kernel — resolved by M:E-A `LinearReduce` trait.
- **B:D-STACK-9**: Per-frame codebook lifetime varies per load — document discipline in Plan I.

### P3 (cosmetic / docs)

- A:T-4, T-8, T-9, T-10 (HPC graduation residue, doc cross-refs).
- B:D-CODEC-6 (lossy Escape docstring — feature per B:E10 / A:T-4).
- B:D-CODEC-10 (mode 2-bit cap, future "mode 5" upgrade path) — *Note: M:E-J consumes the reserved bits for Pearl-rung metadata, not mode 5.*
- B:D-STACK-8 (no backend dispatch in codec yet).
- B:D-STACK-10 (multi-week dependency tracking in coordinator agent).
- A:T-23, M:T-3 (architecture rule update).

### M:T new items (merge-only)

- **M:T-1**: No multi-domain bench harness (P0, see above).
- **M:T-2**: No `LinearReduce<Basis>` trait (P1, see above).
- **M:T-3**: Architecture rule needs 5th category (P1, see above).
- **M:T-4**: Documentation cross-references between sister docs (this doc + both originals + pr-x12-codec-x265-design.md) need a navigation page. Low effort (1 hour); easy to forget.
- **M:T-5**: PR-X12's sub-card numbering needs renumber (A0 = bench harness, A1-A8 as before). Two-line README change.
- **M:T-6**: `trait CurveOrder<const N>` not yet designed (per M:E-B) — gates dimension-agnostic CTU partition.
- **M:T-7**: `trait RdoMetric` per-domain shape — gates A6 design (per Phase 1 plan).

---

## 8. Resolved disagreements between the two docs

Side-by-side surfaced four points where A and B reached different conclusions. The merge resolves each:

| Disagreement | Session A position | Session B position | **Merge resolution** |
|---|---|---|---|
| Critical path | A4 transform first | A7 rANS first | **B wins** — but front-load A4-design (M:E-F) |
| Block size | (implicit) 64×64 fits all | (D-STACK-1) 64×64 conflicts with splat3d 16×16 | **B wins** — resolve via `Ctu<const N>` (M:E-G) |
| Lossy Escape fallback | (T-4) debt | (E10) feature | **B wins** — promote to PSNR knob via λ-RDO |
| Cross-repo dep direction | (T-16, T-17) note as debt | (D-STACK-6) propose third crate | **B wins, explicitly** — extract `ndarray-codec` (M:E-D, Plan H) |

Plus one disagreement that the merge **upgrades into a new claim**:

| Pseudo-disagreement | A side | B side | **Synthesis** |
|---|---|---|---|
| What's domain-specific in the codec? | Transform basis (H-2) | Curve order + Merge topology (E1, E5) + escape payload | **Both, plus a third surface** — M:E-E says exactly three plug-points; M:H-NEW-2 quantifies the code-density win |

---

## 9. How to navigate the four docs (read order)

Future agents inheriting this work should read in this order:

1. **`pr-x12-codec-x265-design.md`** (5 min) — the mechanical spec. What ships. What doesn't.
2. **This doc** (`pr-x12-substrate-merged-canon.md`) (15 min) — the architectural claim + integration plan. Section 3 (merge-only epiphanies) is the highest-information density.
3. **`pr-x12-codec-cognitive-substrate-mapping.md`** *(session A original)* — read **only** if you need A's framing for a specific claim (e.g., A:E-4 transform-IS-optimizer needs A's detailed argument). Otherwise superseded by §3 of this doc.
4. **`pr-x12-cross-domain-synergies.md`** *(session B original)* — read **only** if you need B's framing (e.g., B:E5 Morton≡raster). Otherwise superseded.

Cite using the canonical IDs:
- Mechanism / module layout: cite `pr-x12-codec-x265-design.md` § X
- Epiphanies: cite as `(A:E-N)`, `(B:E-N)`, or `(M:E-X)` per origin
- Holy grails: `(A:H-N)`, `(B:HG-N)`, `(M:H-N)`, `(M:H-NEW-N)`
- Plans: by letter — Plan A, B, C, D, E, F (B's) + Plan G, H, I (merge-only)
- Debt: `(A:T-N)`, `(B:D-CODEC-N)`, `(B:D-STACK-N)`, `(M:T-N)`

---

## 10. Compaction-preservation contract

When this doc is summarised across context windows, preserve at minimum:

1. **The five-category architecture** with codec added (M:E-D / M:T-3).
2. **The three plug-points** for domain specialisation (M:E-E): Transform basis, Curve order, Escape payload.
3. **The critical path**: Plan G (bench) → Plan A4-design → Plan A (A7 rANS).
4. **The sacred file invariant** (B:D-STACK-7): `lance-graph-contract/src/splat.rs` must never fold even if bit patterns rhyme.
5. **The three new traits**: `PredictiveSignal` (M:E-I, Plan I), `LinearReduce<Basis>` (M:E-A, M:T-2), `CurveOrder<const N>` (M:E-B, M:T-6).
6. **The four-load count**: cognitive cells, 3D Gaussian splats, transformer attention, distributed-SGD gradients. Not five (no quantum yet), not three (gradient compression is its own load even though it overlaps with attention).
7. **The PR #195 fixes generalise** (M:E-C): BASIN_NONE collision + unwrap_or non-bijection patterns will fire in every consumer; audit by default.
8. **Citation IDs stable**: don't renumber.

---

## 11. Cross-references

### Sibling knowledge docs (read in order)

- `pr-x12-codec-x265-design.md` — mechanical spec (companion, untouched)
- `pr-x12-codec-cognitive-substrate-mapping.md` — session A original
- `pr-x12-cross-domain-synergies.md` — session B original (merged to master via PR #195 in commit `01c77ccc`)

### Adjacent design docs

- `pr-x10-linalg-core-design.md` — linalg substrate (distance kernels, will host `LinearReduce` basis impls)
- `pr-x1-design.md` + `pr-x2-design.md` — GridLake substrate (`MultiLaneColumn` — batched encode path's carrier)
- `pr-x3-cognitive-grid-design.md` — `BlockedGrid` (CTU's parent type; needs const-generic refactor per M:E-G)
- `pr-x4-design.md` — splat cascade (consumer of M:E-G + Plan E)
- `cognitive-substrate-convergence-v1.md` (in lance-graph repo) — cross-repo locked spec; needs cross-reference back from §11

### Hard rules (must respect)

- `data-flow.md` — no `&mut self` during compute
- `vertical-simd-consumer-contract.md` — W1a contract
- CLAUDE.md "Architecture Rule" — to be amended per M:E-D / M:T-3

### Code references (as of 2026-05-22)

- `src/hpc/codec/ctu.rs` — A1 (shipped)
- `src/hpc/codec/mode.rs` — A2 (PR #195, BASIN_NONE fix pending, see M:T A:T-1)
- `src/hpc/codec/predict.rs` — A3-intra (PR #195, fits_i8 fix landed)
- `src/hpc/cam_pq.rs` — k-means substrate
- `src/hpc/bf16_tile_gemm.rs` — AMX path (NEON stub per A:T-13)
- `src/simd_soa.rs` — `MultiLaneColumn` (batched encode carrier per A:T-19)

### In flight

- **PR #195** (A2 mode + A3-intra): two CodeRabbit findings open (A:T-1, A:T-2). Tracked also in B:D-CODEC's adjacent notes. This doc lives in a separate branch (`claude/continue-ndarray-x0Oaw`) and will land independently.

---

## 12. The single load-bearing sentence

If you read nothing else:

> *PR-X12 is a `PredictiveSignal` + `LinearReduce<Basis>` + `CurveOrder<const N>` factorisation that ships ~1.5 KLoC of generic codec glue plus ~200 LoC per domain consumer, compressing four industries' content (video, 3DGS, attention, gradients) through one Lance-backed wire format with a single λ-RDO knob per consumer — the codec is the substrate, everything else is a 200-line renaming, and the bench harness (Plan G) is the falsifiability proof.*

That's the merged claim. Sections 1-11 elaborate, justify, sequence, and document the debt.

---

_Last edit: 2026-05-22 — merged canon from session A (this branch) and session B (PR #195 branch commit `01c77ccc`). Edit this doc when an M:* item resolves, a new merge-only epiphany lands, or a debt item graduates from open → resolved. Renumber only by appending — never reuse a retired ID._
