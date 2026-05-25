# PR-X12 — WoA Orchestration & Multi-Arch Dispatch Lens

> Date: 2026-05-22
> Status: **perspective doc** — examines how the orchestration crates (`woa-rs`, `woa`, `q2`, `surrealdb`, `MedCare-rs`, `smb-office-rs`) consume the PR-X12 substrate, and how PR-X12's per-arch polyfill decisions (R-4, R-5, R-11) generalise to the entire HPC stack.
>
> Premise: PR-X12 is not just a codec project. It's the **per-arch polyfill contract** that every consumer above `ndarray` will inherit. The codec is the first non-trivial test of whether that contract holds.

---

## 0. Thesis

**Every consumer crate calls the same `ndarray::simd::*` / `ndarray::hpc::*` polyfill surface, regardless of which arch the binary was built for.** The polyfill is a per-arch swap underneath, selected by `cfg(target_feature = ...)` at compile time (per §3 and the W1a contract). PR-X12's per-arch DCT crossover (R-5) and latency assertion (R-11) aren't codec-specific — they're the canonical shape of how any consumer crate's per-arch story bottoms out at the polyfill. If the codec's per-arch story is wrong, the entire HPC consumer ecosystem inherits the bug.

---

## 1. The orchestration problem PR-X12 must solve

In a real deployment, a `woa-rs` agent processing a request might:

1. Receive a video stream (codec: PR-X12)
2. Run perception model on extracted frames (`burn`/`candle`)
3. Query graph state (`lance-graph::blasgraph` tropical-GEMM)
4. Update node-local cache (`surrealdb`)
5. Emit response stream (codec again)

Steps 1, 2, 3, 5 all bottom out at `ndarray::simd::*` and `ndarray::hpc::*`. Each is a polyfill consumer — they call e.g. `blas_level2::batched_gemm` and get whatever backend the binary was compiled with. **None of the consumer crates know which backend is active**, and they MUST NOT: backend-specific symbols (AMX bytecode, AVX-512 asm, NEON intrinsics, SVE2 predicates) live exclusively inside `src/simd_<arch>.rs` and never reach a consumer's source. The fleet ships per-arch binaries (§3.2); each binary embeds one backend file via cfg.

This is what makes PR-X12's R-4 / R-11 architecture-conditional bench gates *substrate policy*, not codec policy. R-4 says "Plan G clears on each of: SPR / Zen 4 / Graviton 3 / Apple M-class" (per-arch CI matrix), and R-11 adds per-arch latency assertions. That same gate structure applies to:

- `burn` model serving (forward pass: same Rust, per-arch binary)
- `candle` quantized inference (q4/q8: same Rust, per-arch binary)
- `lance-graph::blasgraph` graph queries (tropical-GEMM: same Rust, per-arch binary)
- `surrealdb` HNSW search (vector dist: same Rust, per-arch binary)
- `MedCare-rs` DICOM transform (DCT + wavelet: same Rust, per-arch binary)
- `smb-office-rs` OCR + layout (conv + attention: same Rust, per-arch binary)

Every one of these inherits the polyfill contract: identical consumer-facing Rust, one cfg-selected backend per build. PR-X12 is the first to make the parity-test obligation visible.

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
                     │ same Rust API on every arch
                     ▼
┌────────────────────────────────────────────────────┐
│ ndarray::hpc + ndarray::simd (polyfill substrate)  │
│   blas_level{1,2,3}, fft, cam_pq, activations,     │
│   simd_int_ops, bf16_tile_gemm                     │
│   (~15K LoC; PR-X12 ratchets at this layer)        │
└────────────────────┬───────────────────────────────┘
                     │ cfg(target_feature = …) picks ONE
                     ▼
┌────────────────────────────────────────────────────┐
│ Backend file (one per binary):                     │
│   simd_avx512.rs  →  asm/intrinsics + AMX bytecode │
│   simd_neon.rs    →  NEON / SVE2 intrinsics        │
│   simd_scalar.rs  →  portable fallback             │
└────────────────────┬───────────────────────────────┘
                     ▼
        Hardware: SPR / Zen / Graviton / Apple
```

**WoA never touches `target_feature` directly.** Its job is async task scheduling, transport (Q2 over QUIC), persistence (surrealdb), and policy. Per-arch SIMD code lives exclusively inside the backend file (`simd_<arch>.rs`); the polyfill above swaps which file is compiled in via cfg.

This separation is what makes R-3's LoC envelope (≤1500 LoC codec body) tractable. The codec crate doesn't choose a backend — it calls the polyfill. WoA doesn't choose a backend — it calls the codec, which calls the polyfill. Per-arch code lives once, inside `src/simd_<arch>.rs`, behind the polyfill surface.

---

## 3. Per-arch substrate via compile-time polyfill

The PR-X12 substrate follows the project's W1a consumer contract (see `CLAUDE.md` and `.claude/knowledge/vertical-simd-consumer-contract.md`): **all dispatch is polyfill**. The stack has three layers, and only the bottom one is allowed to know about specific architectures:

```text
┌────────────────────────────────────────────────────────────┐
│ Consumers — codec encode/decode bodies, downstream crates  │
│   (ndarray-codec, burn, candle, lance-graph, surrealdb,    │
│    MedCare-rs, smb-office-rs, q2, WoA scheduler)           │
│   Call ndarray::simd::* directly. Never name a backend.    │
└────────────────────────┬───────────────────────────────────┘
                         │ identical signatures everywhere
                         ▼
┌────────────────────────────────────────────────────────────┐
│ Polyfill surface — src/simd.rs                             │
│   cfg(target_feature = ...) re-exports exactly ONE backend │
│   to compile in. Same fn names, same types, every arch.    │
└────────────────────────┬───────────────────────────────────┘
                         │ cfg substitutes one file
                         ▼
┌────────────────────────────────────────────────────────────┐
│ Backend — simd_avx512.rs / simd_neon.rs / simd_scalar.rs   │
│   This is where AMX bytecode, AVX-512 asm/intrinsics,      │
│   NEON loads, SVE2 predicates LIVE. Implementation detail. │
│   Consumers above never reach in here.                     │
└────────────────────────────────────────────────────────────┘
```

There is **no runtime CPU detection, no `HwCaps`/`CpuCaps` branching, no `if has_avx512 else …` dispatch, and no `unsafe { runtime_branch }` chain.** The target CPU is fixed at build time via `.cargo/config.toml` (`target-cpu=x86-64-v4` makes AVX-512 mandatory on x86_64) or via the target triple for non-x86 builds. One build, one backend file compiled in, one path.

### 3.1 The polyfill primitive: cfg-selected per-arch files

The pattern already shipping in `src/simd*.rs` (per `CLAUDE.md` Repository Structure):

```rust
// src/simd.rs — consumer-facing surface, re-exports a single backend
#[cfg(target_feature = "avx512f")]
pub use crate::simd_avx512::*;

#[cfg(all(not(target_feature = "avx512f"), target_arch = "aarch64"))]
pub use crate::simd_neon::*;

#[cfg(not(any(target_feature = "avx512f", target_arch = "aarch64")))]
pub use crate::simd_scalar::*;
```

Each backend file implements the same public functions with identical signatures; **the actual AMX bytecode / AVX-512 asm / NEON intrinsics / SVE2 predicates are contained inside those files** and never escape. The W1a contract requires all three backends + a parity test before any new primitive lands.

**The codec body is a consumer of this polyfill.** When `ndarray-codec` writes encoding code — Skip/Merge/Delta/Escape mode selection, basin lookups, tropical-GEMM RDO, rANS state-machine ticks, EWA splat composition — it calls `ndarray::simd::*` exactly the way `burn` / `candle` / `lance-graph` do. **The codec does not know it is on AMX.** It does not reach for `simd_avx512::*` directly, does not name a backend symbol, does not branch on architecture. The cfg at the polyfill layer picks the right backend at build time; the encoder is identical Rust across all architectures.

**Escape hatch (rare).** A very small number of hot inner loops may need to drop below the polyfill into a backend-specific intrinsic for performance reasons that the polyfill surface genuinely cannot express. When that happens: the violation lives inside `src/simd_<arch>.rs` (where backend-specific code is already at home), is `cfg`-gated to that arch, is parity-tested against the other backends' equivalent, and gets a `// SAFETY:` + agent audit per `CLAUDE.md`'s sentinel-qa rule. **It is the exception, not the model.** No consumer crate — codec body included — is ever the right place for it.

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

**WoA's contract:** if any of these are violated on a supported arch, the consumer can either accept the slowdown or refuse to schedule the request. WoA has visibility into per-arch polyfill performance (which backend was compiled into the binary it's running, plus stage-latency telemetry) via the substrate's metrics endpoint:

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
- GPU offload → add a `Reducer::backend_target() -> BackendTarget` hook to let consumers opt into a GPU polyfill at compile time (Plan E adjacent; still cfg-selected, not runtime-branched)
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

Uses `blas_level3::gemm` for matrix multiply, `activations` for nonlinearities, `cam_pq` for KV cache compression. Per-arch polyfill via the same `cfg(target_feature)` mechanism — `burn` itself never names a backend. Will benefit directly from PR-X12's R-4 / R-11 latency-assertion infrastructure when it lands (burn has wanted this for ~14 months).

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
2. **Per-arch polyfill transparency** — consumers continue calling the `ndarray::simd::*` / `ndarray::hpc::*` surface unchanged across arches; cfg at the polyfill layer selects exactly one backend at build time. Consumers never name a backend symbol.
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
