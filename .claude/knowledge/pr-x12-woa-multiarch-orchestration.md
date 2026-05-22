# PR-X12 — WoA Orchestration & Multi-Arch Dispatch Lens

> Date: 2026-05-22
> Status: **perspective doc** — examines how the orchestration crates (`woa-rs`, `woa`, `q2`, `surrealdb`, `MedCare-rs`, `smb-office-rs`) consume the PR-X12 substrate, and how PR-X12's per-arch dispatch decisions (R-4, R-5, R-11) generalise to the entire HPC stack.
>
> Premise: PR-X12 is not just a codec project. It's the **per-arch dispatch contract** that every consumer above `ndarray` will inherit. The codec is the first non-trivial test of whether that contract holds.

---

## 0. Thesis

**Every consumer crate dispatches kernels across {Intel SPR, AMD Zen 4-5, ARM Graviton 3-4, Apple Silicon, NVIDIA Hopper-Blackwell} via the same `ndarray::hpc` capability traits.** PR-X12's per-arch DCT crossover (R-5) and latency assertion (R-11) aren't codec-specific — they're the canonical shape of how any consumer crate gates fast-paths. If the codec's per-arch story is wrong, the entire HPC consumer ecosystem inherits the bug.

---

## 1. The orchestration problem PR-X12 must solve

In a real deployment, a `woa-rs` agent processing a request might:

1. Receive a video stream (codec: PR-X12)
2. Run perception model on extracted frames (`burn`/`candle`)
3. Query graph state (`lance-graph::blasgraph` tropical-GEMM)
4. Update node-local cache (`surrealdb`)
5. Emit response stream (codec again)

Steps 1, 2, 3, 5 all hit the `ndarray::hpc` BLAS layer. Each step has a per-arch fast-path: SPR uses AMX, Zen 4 uses VNNI+AVX-512, Graviton 3 uses SVE2, Apple uses NEON/AMX, Hopper uses tensor cores. **None of the consumer crates know which fast-path is active.** They call `blas_level2::batched_gemm` and the substrate dispatches.

This is what makes PR-X12's R-4 / R-11 architecture-conditional bench gates *substrate policy*, not codec policy. R-4 says "Plan G clears at most on 1 of: SPR / Zen 4 / Graviton 3 / Apple M-class," and R-11 adds latency assertions. That same gate structure applies to:

- `burn` model serving (forward pass per arch)
- `candle` quantized inference (q4/q8 per arch)
- `lance-graph::blasgraph` graph queries (tropical-GEMM per arch)
- `surrealdb` HNSW search (vector dist per arch)
- `MedCare-rs` DICOM transform (DCT + wavelet per arch)
- `smb-office-rs` OCR + layout (conv + attention per arch)

Every one of these inherits the dispatch contract. PR-X12 is the first to make it visible.

---

## 2. WoA's place in the stack

```text
┌────────────────────────────────────────────────────┐
│ WoA agent (woa-rs, woa)                            │
│   Request orchestration, scheduling, transport     │
└────────────────────┬───────────────────────────────┘
                     │ async dispatch, no SIMD
                     ▼
┌────────────────────────────────────────────────────┐
│ Domain consumer crates                             │
│   ndarray-codec, burn, candle, lance-graph,        │
│   surrealdb, MedCare-rs, smb-office-rs             │
│   (Each: ~1-5K LoC of generic code + traits)       │
└────────────────────┬───────────────────────────────┘
                     │ capability traits, target_feature
                     ▼
┌────────────────────────────────────────────────────┐
│ ndarray::hpc (the dispatch substrate)              │
│   blas_level{1,2,3}, fft, cam_pq, activations,     │
│   simd_int_ops, bf16_tile_gemm                     │
│   (~15K LoC; PR-X12 ratchets at this layer)        │
└────────────────────┬───────────────────────────────┘
                     │ per-arch SIMD intrinsics
                     ▼
┌────────────────────────────────────────────────────┐
│ Hardware: SPR / Zen / Graviton / Apple / Hopper    │
└────────────────────────────────────────────────────┘
```

**WoA never touches `target_feature` directly.** Its job is async scheduling, transport (Q2 over QUIC), persistence (surrealdb), and policy. The SIMD dispatch happens one layer below, in the consumer crates calling `ndarray::hpc`.

This separation is what makes R-3's LoC envelope (≤1500 LoC codec body) tractable. The codec crate doesn't dispatch — it calls the substrate. WoA doesn't dispatch — it calls the codec, which calls the substrate. Per-arch code lives once, in `ndarray::hpc`.

---

## 3. Per-arch substrate via compile-time polyfill

The PR-X12 substrate follows the project's W1a consumer contract (see `CLAUDE.md` and `.claude/knowledge/vertical-simd-consumer-contract.md`): **all dispatch is polyfill**. Per arch we ship a separate backend file with the same public surface, and `cfg(target_feature = ...)` selects exactly one to compile in. There is **no runtime CPU detection, no `HwCaps`/`CpuCaps` branching, no `if has_avx512 else …` dispatch, and no `unsafe { runtime_branch }` chain.** The target CPU is fixed at build time via `.cargo/config.toml` (`target-cpu=x86-64-v4` makes AVX-512 mandatory on x86_64) or via the target triple for non-x86 builds. One build, one path.

### 3.1 The polyfill primitive: cfg-selected per-arch files

The pattern is the same one already shipping in `src/simd*.rs` (per `CLAUDE.md` Repository Structure):

```rust
// src/simd.rs — consumer-facing surface, re-exports a single backend
#[cfg(target_feature = "avx512f")]
pub use crate::simd_avx512::*;

#[cfg(all(not(target_feature = "avx512f"), target_arch = "aarch64"))]
pub use crate::simd_neon::*;

#[cfg(not(any(target_feature = "avx512f", target_arch = "aarch64")))]
pub use crate::simd_scalar::*;
```

Each backend file (`simd_avx512.rs`, `simd_neon.rs`, `simd_scalar.rs`) implements the same public functions with identical signatures. The W1a contract requires **all three backends + a parity test** before any new primitive lands. The codec body (`ndarray-codec`, see R-3) and downstream consumers (burn / candle / lance-graph / surrealdb / WoA fleet) call `ndarray::simd::*` directly — they never see or reason about which backend is active. The cfg substitutes one file at the use-site; consumer code is identical across architectures.

### 3.2 Build-time CPU selection (not runtime detection)

Target CPU is decided once, at build time:

| Mechanism | Source | Effect |
|---|---|---|
| `.cargo/config.toml` `target-cpu=x86-64-v4` | repo policy | AVX-512 mandatory on x86_64 (per `CLAUDE.md`) |
| `--target aarch64-apple-darwin` | CI / fleet build matrix | NEON-fp16 backend compiles in |
| `--target aarch64-unknown-linux-gnu` + SVE2 target-feature | Graviton build | SVE2 backend compiles in |

The WoA fleet ships **per-arch binaries**, not a fat binary that probes. Q2 distributes the right binary to each node based on the node's already-known architecture (declared at registration time, not detected per request). Cross-arch determinism (§6 below) is enforced because each binary embeds exactly one backend and the W1a parity test gates every primitive at the substrate layer.

### 3.3 Per-arch tunable crossover (R-5)

Some operations (DCT-II vs GEMM, basin-lookup width, etc.) have a "small N: scalar path, large N: SIMD path" crossover whose break-even N varies per backend. The crossover lives in the **same polyfill** as the SIMD primitives: a `cfg(target_feature = ...)`-selected `const`.

```rust
// src/hpc/dct_crossover.rs — one const per backend file, cfg-selected
//
//   simd_avx512.rs:                pub const DCT_BATCH_CROSSOVER: usize = 64;
//   simd_neon.rs (Apple Silicon):  pub const DCT_BATCH_CROSSOVER: usize = 256;
//   simd_scalar.rs:                pub const DCT_BATCH_CROSSOVER: usize = usize::MAX;

pub fn dct_apply<const N: usize>(input: &[i16], output: &mut [i16]) {
    if N >= DCT_BATCH_CROSSOVER {
        dct_gemm_path(input, output)      // calls into ndarray::simd::*
    } else {
        dct_butterfly_path(input, output) // also calls into ndarray::simd::*
    }
}
```

The integer `DCT_BATCH_CROSSOVER` comes from one of two places:
1. **Hand-tuned default**: a known-good number per backend, checked into the backend file.
2. **Plan G calibration override**: `build.rs` may consult `CARGO_CFG_TARGET_FEATURE` + a pre-recorded calibration artifact from `codec-bench` and emit a refined const into `OUT_DIR`, included by the backend file. This is still compile-time selection — the build script never probes the host CPU, only reads Cargo's target-config env vars.

Either way the constant is **fixed in the compiled binary**. R-5 commits these crossovers as bench-tunable but compile-time-fixed; the `cfg(target_feature)`-selected backend file is the single source of truth.

---

## 4. The latency budget split — codec / orchestration / network

A WoA agent processing a video stream end-to-end has three latency contributors:

```text
Total latency  =  T_codec  +  T_orchestration  +  T_network
```

PR-X12 (R-11) commits a budget on `T_codec`:

| Stage | Budget (per encode) | Source |
|---|---|---|
| Codec encode | ≤ 0.5× wall-clock for 1080p @ 30 fps | R-11 |
| Codec decode | ≤ 0.25× wall-clock for 1080p @ 30 fps | R-11 |
| Block-level ME | ≤ 10 µs per CTU on SPR | R-11 spec, calibrated by codec-bench |
| Tropical-GEMM RDO | ≤ 50 µs per CTU on SPR | derived from R-7 cost analysis |
| Basis::apply (DCT) | ≤ 2 µs per 32×32 block on SPR | derived from R-5 |

**WoA's contract:** if any of these are violated on a supported arch, the consumer can either accept the slowdown or refuse to schedule the request. WoA has visibility into per-arch dispatch quality via the substrate's metrics endpoint:

```rust
ndarray::hpc::metrics::stage_latency_p99(stage: StageId) -> Duration;
```

This is wired through to woa-rs's request scheduler. If `T_codec` p99 exceeds budget, woa-rs can:

- Reroute to a different node (better hardware available)
- Degrade the request (lower codec quality, smaller batch)
- Fail fast with a clear error (don't tie up the client)

**Without R-11's commitment to latency assertions in CI, this whole chain falls over.** The substrate-to-orchestrator contract is empty unless someone ratchets on it.

---

## 5. Federated codebook policy (R-13) — the orchestration angle

R-13 commits that the codec's 4096-entry basin codebook can be either:

- **Per-instance** (each PR-X12 encoder builds its own from the input stream)
- **Federated** (a cluster of encoders shares a codebook, periodically updated)
- **Per-domain pretrained** (a hand-curated codebook ships with the binary for {video, text, image, audio} domain segments)

The orchestration layer (WoA / Q2) is where federation policy lives. Specifically:

```rust
// In q2 (transport / coordination):
pub enum CodebookPolicy {
    LocalEphemeral,                    // each encoder owns its codebook
    SharedClusterWide { ttl: Duration }, // gossip protocol distributes
    SharedRegional { region: Region },   // edge-tier sharing
    PretrainedStatic { id: BlobId },     // immutable, served from CAS
}

impl WoaAgent {
    fn select_codebook(&self, request: &Request) -> CodebookHandle {
        match request.payload_class() {
            PayloadClass::HumanText => self.pretrained("english-text-v3"),
            PayloadClass::VideoFrame => self.shared_cluster_wide(),
            PayloadClass::EphemeralBlob => self.local_ephemeral(),
            // ...
        }
    }
}
```

**R-13 says:** the codec layer exposes the basin-codebook as a swappable handle. The orchestration layer chooses which codebook to use per request. PR-X12 ships with the substrate hook; q2 owns the policy.

**Why this matters for PR-X12 scope:** the basin-codebook is currently a hard-coded 4096-entry array per encoder. R-13 commits to making it swappable (replacing the array reference with a handle/trait) — this is a ~30-line change in the codec crate, not a 300-line rewrite. The federation logic itself lives in q2, outside PR-X12's body.

This is a model for many features that look "out of scope" for PR-X12 but actually need a tiny anchor in PR-X12 to be reachable later:

- Federated codebook → swap pointer to handle (R-13)
- 3DGS scene anchor → add SceneAnchor header_kind (x266 doc)
- GPU offload → add `Reducer::dispatch_target() -> DispatchTarget` (Plan E adjacent)
- Speculative decode → add `Frame::is_speculative()` bit in header reserved field

None of these are PR-X12 scope. All of them require ≤50 LoC of "anchor" in PR-X12. The discipline of M:H-NEW-2 + R-3's LoC envelope is what makes future anchoring possible without forking the codec.

---

## 6. Cross-arch determinism — the consumer's hardest requirement

A WoA agent that runs on SPR in the data center and Apple Silicon at the edge must produce **the same answer** for the same input. Floating-point order-of-operations differs across SIMD widths, so naive parallel reductions break this.

PR-X12's `LinearReduce<T>` abstraction (R-1, M:E-A) is the answer:

```rust
pub trait Reducer<T> {
    fn reduce_pair(&self, lhs: T, rhs: T) -> T;
}

// For bit-exact reduction across archs:
pub struct OrderedKahanReducer;

impl Reducer<f32> for OrderedKahanReducer {
    fn reduce_pair(&self, lhs: f32, rhs: f32) -> f32 {
        // Kahan compensated sum, with explicit left-to-right order
        // Same bit pattern on every arch
        kahan_add(lhs, rhs)
    }
}
```

The codec uses `OrderedKahanReducer` for any sum that crosses a wire-format boundary — basin assignment, rate-distortion accumulation, transform coefficient sum. Same input → same bits, regardless of arch. Determinism is paid for in some throughput (Kahan is ~3× slower than naive sum), but it's a tunable choice per use site.

**Without R-1's basis/reducer split, cross-arch determinism is a substrate-wide audit nightmare.** With it, the audit is per-use-site: grep for places that use `NaiveSimdReducer` on cross-wire-format paths, replace with `OrderedKahanReducer`.

---

## 7. Failure modes and mitigations

### 7.1 ABI drift between substrate and consumer

If `ndarray::hpc::blas_level2::batched_gemm`'s signature changes, every consumer breaks. **Mitigation:** R-3's LoC envelope explicitly excludes the substrate API from "codec body LoC" — meaning the API gets the same review scrutiny as a public crate API. SemVer applies.

### 7.2 Per-arch CI flake

R-4 commits codec-bench gates the merge on at most 1 arch. **Mitigation:** the bench passes on the canonical arch (SPR), and the other arches are *informational* on each PR but blocking on release tag. This is the standard "fast PR / slow release" gate pattern.

### 7.3 Version skew across the WoA fleet

A cluster running mixed PR-X12 versions could produce inconsistent codec output. **Mitigation:** the wire format header includes a version byte (one of M:E-J's reserved bits in future revisions); decoder rejects incompatible streams with a clean error. The federation gossip in q2 propagates the codec version as part of the node descriptor.

### 7.4 Federated codebook poisoning

If R-13's federated codebook is updated by a compromised node, the cluster compresses badly. **Mitigation:** codebook updates are signed; q2 ignores updates not signed by quorum. Out of PR-X12 scope (it's a transport/auth concern) but the substrate exposes the hook.

---

## 8. The consumer crates in detail

Quick tour of what each crate inherits from PR-X12 substrate decisions:

### 8.1 `burn` (model training/inference)

Uses `blas_level3::gemm` for matrix multiply, `activations` for nonlinearities, `cam_pq` for KV cache compression. Per-arch dispatch via the same target_feature paths. Will benefit directly from PR-X12's R-4 / R-11 latency-assertion infrastructure when it lands (burn has wanted this for ~14 months).

### 8.2 `candle` (quantized inference)

Heavy user of `simd_int_ops` and `bf16_tile_gemm`. Most-affected consumer by R-5's per-arch crossover constants, because candle's q4/q8 paths have similar crossover decisions. Will likely adopt the same crossover-as-const pattern within the next 1-2 quarters.

### 8.3 `lance-graph::blasgraph` (graph queries)

Owner of tropical-GEMM (R-7); the codec is a consumer, not an owner, of that kernel. PR-X12's allowed dependency direction (`ndarray-codec → lance-graph::blasgraph`) was confirmed under R-7 only after careful audit; previously lance-graph could only consume `ndarray`, not be consumed by sibling crates. M:E-H clarifies this dep direction is fine because both crates sit above ndarray and below woa/q2.

### 8.4 `surrealdb` (vector + relational DB)

Uses `cam_pq::hnsw_search` for vector lookups, `simd_int_ops` for filter expressions. Will inherit R-13's federated-codebook pattern for its own quantized vector indexes (long-discussed, not scheduled).

### 8.5 `MedCare-rs` (medical imaging)

The doc most likely to drive R-1's basis trait to its limits — medical imaging uses DCT, DWT (wavelet), and 3D radon transforms, all of which want to be `Basis<T>` impls. Provides the second non-trivial test of the basis trait after PR-X12 ships. Federated-codebook policy (R-13) is *required* for medical imaging because PHI rules prohibit per-instance codebooks leaking patient-specific symbol distributions.

### 8.6 `smb-office-rs` (office document OCR)

Heavy user of conv (`activations::conv2d`) and attention (within `burn`-backed models). Less affected by PR-X12's specific reservations; more affected by R-11's latency assertions, because office OCR is latency-sensitive for interactive use cases.

### 8.7 `q2` (transport / coordination)

Owns the federation policy (R-13), the codec version negotiation, and the per-arch capability gossip. q2 doesn't itself touch `ndarray::hpc` — it routes requests to consumers that do. q2's interaction with PR-X12 is at the orchestration layer: scheduling, codec version constraints, federated codebook policy.

---

## 9. What PR-X12 must NOT break

In light of the above, the irreducible commitments PR-X12 must keep for the consumer ecosystem:

1. **Substrate API stability** — `blas_level2::batched_gemm`, `cam_pq::kmeans`, `fft::dct_apply`, `activations::conv2d` keep their signatures across PR-X12 changes. Additions OK, breaks not OK.
2. **Per-arch dispatch transparency** — consumers continue calling capability-trait methods; the substrate continues choosing the right SIMD path.
3. **`Reducer<T>` ordered-sum guarantee** — any consumer using `OrderedKahanReducer` (or similar) continues to get bit-exact cross-arch reductions.
4. **Latency-assertion CI infrastructure** — R-11's framework is consumer-callable for their own benches; not codec-private.
5. **Codebook handle indirection** (R-13) — the codec ships with the handle pattern, consumers can swap codebooks without forking.

If PR-X12 keeps these five things stable, the consumer crates inherit the win. If any one breaks, the cascade across burn/candle/lance-graph/surrealdb is weeks of remediation per affected crate.

---

## 10. Cross-references

- **Substrate canon:** `pr-x12-substrate-merged-canon.md`
- **Resolutions:** R-3, R-4, R-5, R-7, R-11, R-13 in `pr-x12-canon-resolutions-delta.md`
- **GEMM lens:** `pr-x12-x265-blasgraph-gemm.md`
- **Future capability lens:** `pr-x12-x266-3dgs-spacetime-upscaling.md`
- **WoA-side architecture:** check `woa-rs` repo `docs/architecture.md` (not in this repo)
- **Q2 transport:** see `q2` repo for codebook gossip protocol design
- **Federation policy reading:** R-13 calls out the model; q2 will implement

_Last edit: 2026-05-22._
