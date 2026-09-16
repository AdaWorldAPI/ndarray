# Project NDARRAY Expansion

> HPC Rust Transformation — porting `adaworldapi/rustynum` features into this ndarray fork.

## Quick Context
- **What**: High-performance linear algebra with pluggable BLAS backends (Native SIMD, MKL, OpenBLAS)
- **Source**: `adaworldapi/rustynum` — reference GEMM, SIMD, and FFI implementations
- **Target**: This repo — ndarray fork enhanced with HPC backends
- **Rust**: stable only — the pinned `rust-toolchain.toml` (1.98.1; `rust-version` in `Cargo.toml` is the floor). No nightly features on any default or supported build path.
  - **The one documented exception — `nightly-simd` (opt-in, validation-only, since PR #173):** a Cargo feature that swaps the SIMD realization for `core::simd` (`src/simd_nightly/*`, `#![feature(portable_simd)]`) so the realization matrix can witness that backend too. It is never enabled by default, nothing on stable may depend on it, every stable CI row builds without it, and it is exercised only by the dedicated nightly CI rows (`nightly-simd-polyfill`, the `simd-matrix` nightly row) and `scripts/masking-parity.sh nightly`. Removing the feature would drop that backend from the matrix; enabling it anywhere by default would violate this rule.

## Agent Protocol
This project uses specialized agents in `.claude/agents/`. Follow these rules:

1. **Always read `.claude/blackboard.md` before starting any task**
2. After completing work, update the blackboard with decisions and loose ends
3. Delegate appropriately:
   - GEMM kernels, SIMD, memory layout, Backend trait design → `savant-architect`
   - `unsafe` code, FFI audit, benchmarking → `sentinel-qa`
   - Embedding ops, distance metrics, vector store bridges → `vector-synthesis`
   - API surface, docs, feature gates, Cargo.toml → `product-engineer`
   - Feature prioritization, gap analysis, phase planning → `l3-strategist`
4. When encountering `unsafe` code, **always** delegate to sentinel-qa for audit
5. Write decisions to the blackboard, not just to chat
6. **Cargo build residue** — fan out the Sonnet fleet in the *shared* checkout (no per-agent worktrees), edit-only; the Opus orchestrator compiles/lints/tests **once** in the single 7 GB `target/`. Opus may run cargo freely. See `.claude/rules/agent-cargo-hygiene.md`.

## Hard Rules
- OpenBLAS and MKL are **mutually exclusive** feature gates. Never both.
- Zero-cost abstractions: generics monomorphize, no `Box<dyn>` in hot paths.
- Every `unsafe` block needs a `// SAFETY:` comment.
- **`&&`-chain a commit to the edit that produces it — never sequence it
  after.** An anchor assertion in an edit script protects the FILE; it does
  not protect the RECORD. Measured here 2026-09-16: an edit script's assertion
  fired correctly (a mid-line anchor that did not match), the script aborted,
  no bad edit landed — and the `git commit` that followed it ran anyway,
  shipping a message claiming two files while `git show --stat` showed one.
  For one commit a plan was documented as updated while it was not. The
  narrative remedy (`git show` the diff before claiming it) is real but
  optional; `python3 edit.py && git add … && git commit …` is mechanical and
  cannot be forgotten.
- All public APIs need `///` doc comments with examples.
- `cargo clippy -- -D warnings` must pass.
- **Every compile runs with `CARGO_PROFILE_DEV_DEBUG=0`** (operator, 2026-09-16:
  *"use debug 0"*). Not a preference — **debug info is the disk hog, not the
  code**, and this container's writable allowance is a fixed per-session budget
  that presents as `No space left on device` mid-link, not as a full disk.
  Measured here the same day on the identical tree and the identical test run:
  `target/debug` is **1.9 GB with debug info and 291 MB without** — a 6.5×
  cut, for a run that passed 2319 tests either way. Export it (plus
  `CARGO_PROFILE_TEST_DEBUG=0` and `CARGO_INCREMENTAL=0`) as ENV, never as a
  profile edit in `Cargo.toml`: the rule governs the agent's compiles, not the
  profile a human commits. `--release` is NOT a substitute — it is slower to
  build and the test runs need the dev path. And because a profile change
  invalidates the whole cache, **delete `target/debug` before switching rather
  than growing a second copy beside it.**
- **All new public `pub fn` in `src/simd_*.rs` follows the W1a consumer contract** at `.claude/knowledge/vertical-simd-consumer-contract.md` — struct methods on typed wrappers, closure-parameterized batch primitives, all three backends (AVX*/NEON/scalar) implemented, parity test mandatory, saturating/overflow semantics documented. The Ada stack (lance-graph + downstream) enforces "all SIMD from `ndarray::simd`" via its `simd-savant` agent; missing primitives in ndarray force consumer-side raw-intrinsic violations, so additions here are gating the consumer-side sweep. **VPABSB does NOT saturate `i8::MIN`** — see § "VPABSB correction" in the contract doc before implementing `saturating_abs` or any abs primitive.

## Compaction Preservation
When summarizing this conversation, preserve:
- All entries in `.claude/blackboard.md`
- Current epoch number and loose ends
- Which agents have been consulted and their verdicts
- Any BLOCK findings from sentinel-qa

## Repository Structure (Actual as of 2026-03-22)
```
src/
├── lib.rs              # Re-exports, feature gates
├── backend/
│   ├── mod.rs          # BlasFloat trait (was planned as LinalgBackend)
│   ├── native.rs       # Pure Rust + SIMD microkernels
│   ├── mkl.rs          # Intel MKL FFI (feature = "intel-mkl")
│   ├── openblas.rs     # OpenBLAS FFI (feature = "openblas")
├── simd.rs             # Consumer-facing SIMD module, re-exports all types
├── simd_avx512.rs      # AVX-512 type definitions (11 types from rustynum)
├── simd_avx2.rs        # AVX2 functions
│   └── kernels_avx512.rs  # AVX-512 kernel implementations
├── hpc/                # 55 modules — ALL DONE (880 lib tests)
│   ├── blas_level1.rs  # BLAS L1 (dot, axpy, scal, nrm2, asum, etc.)
│   ├── blas_level2.rs  # BLAS L2 (gemv, ger, symv, trmv, trsv)
│   ├── blas_level3.rs  # BLAS L3 (gemm, syrk, trsm, symm)
│   ├── quantized.rs    # BF16 GEMM, Int8 GEMM
│   ├── lapack.rs       # LU, Cholesky, QR
│   ├── fft.rs          # FFT/IFFT (Cooley-Tukey radix-2)
│   ├── vml.rs          # Vector math (exp, ln, sqrt, etc.)
│   ├── statistics.rs   # Median, var, std, percentile, top_k
│   ├── activations.rs  # Sigmoid, softmax, log_softmax
│   ├── fingerprint.rs, plane.rs, seal.rs, node.rs  # Cognitive core
│   ├── cascade.rs, bf16_truth.rs, causality.rs     # Truth/cascade
│   ├── blackboard.rs   # Typed slot arena
│   ├── bnn.rs, clam.rs, arrow_bridge.rs            # Additional crates
│   ├── hdc.rs, nars.rs, qualia.rs, spo_bundle.rs   # Cognitive extensions
│   └── ... (27 more modules)
```

## Status (2026-03-22 Audit)
- **All "must be ported" items: DONE** — see `.claude/blackboard.md` for full inventory
- **880 lib tests passing**, 2 doctest failures out of 302
- **Build currently fails (exit 101)** — needs investigation
- See blackboard for detailed per-module test counts

## Session: Qwen3.5 × Opus 4.5/4.6 Reverse Engineering (2026-03-31)

### New Modules
- `src/hpc/styles/` — 34 cognitive primitives (rte, htd, smad, tcp, irs, mcp, tca, cdt, mct, lsi, pso, cdi, cws, are, tcf, ssr, etd, amp, zcf, hpm, cur, mpc, ssam, idr, spp, icr, sdd, dtmf, hkf). Each is `fn(Base17, NarsTruth) → result`. 49 tests.
- `src/hpc/causal_diff.rs` — CausalEdge64 (u64 packed), scaffold_to_palette3d_layers(), quality scoring (GOOD/BAD/UNCERTAIN), NARS self-reinforcement LoRA, PAL8 serialization (4101 bytes).
- **Build config — AVX-512 is NOT the default, and believing it is corrupts
  measurements.** `.cargo/config.toml` sets **`x86-64-v3` (AVX2)**, deliberately:
  it is the portable CI/distribution baseline, and its own comment explains why
  (without a v3 floor the AVX2 intrinsics in `simd_avx2.rs` SIGILL). A plain
  `cargo build`/`run`/`test` therefore measures **v3**.
  **For AVX-512 you must ask for it, every time:**

  ```sh
  env -u RUSTFLAGS cargo --config .cargo/config-v4.toml run --release --example <name>
  ```

  `env -u RUSTFLAGS` is load-bearing: a RUSTFLAGS env var REPLACES every
  cargo-config rustflags entry, so it silently drops `-Ctarget-cpu=x86-64-v4`
  and the arm measures v3 while claiming v4 (the trap `scripts/masking-parity.sh`
  documents). Verify the arm you got — the parity program prints
  `avx512f=true|false`, and any probe that reports timings should too.
  `.cargo/config-avx512.toml` is the stricter Sapphire Rapids EXECUTION config
  (VNNI/BF16/FP16/AMX) and SIGILLs on earlier AVX-512 silicon; `config-native.toml`
  resolves the host CPUID.
  (This line previously claimed `config.toml` was v4 "AVX-512 mandatory" — it
  never was, and that error made a whole measurement arc read v3 as v4. Corrected
  2026-09-16 against `.cargo/config.toml:83`.)

  **The v4 config also carries `-D warnings`, which makes a DISABLE RUN fail
  in a way that reads as success.** A disable typically removes a use of
  something; the variable it fed then goes unused; `-D warnings` promotes that
  to a hard error; the test binary is never built, so the run emits no `test
  result:` line at all. Piped through a `grep` for the failing assertion, "did
  not compile" and "the guard was not load-bearing" look identical — the
  workspace's known trap (*a disable that does not APPLY is indistinguishable
  from a guard that does not bind*) with a second door. Measured 2026-09-16 on
  the `gt_u8_to_mask` signed-compare disable: it silently produced `error:
  unused variable: threshold_v` and I nearly recorded the falsifier as inert.
  **Always read the disable run's exit status and the `test result:` line
  itself, never only a grep of its assertions** — and prefix, don't delete,
  when a disable orphans a binding.
- `src/simd.rs` — compile-time AVX-512 dispatch via `cfg(target_feature = "avx512f")`.

### The parity arms are ALL reachable here — never report one as blocked without apt

`scripts/masking-parity.sh` takes `native | nightly | wasm | wasm-scalar |
neon-qemu`, and the cross arms are an `apt-get` away, not an environment
limit. Measured 2026-09-16: `neon-qemu` failed with a bare
`No such file or directory (os error 2)` — which reads as "this target is not
available here" and is in fact a **missing linker**, then a missing
interpreter, in two separate steps:

```sh
sudo apt-get update
sudo apt-get install -y gcc-aarch64-linux-gnu qemu-user-static
```

`qemu-user` alone is NOT enough: the script invokes `qemu-aarch64-static`,
and the dynamic `qemu-aarch64` from `qemu-user` leaves a second, differently
worded failure (`command not found`) that looks like a fresh problem rather
than the same one. Install `qemu-user-static`.

**And `native` is the AVX2 arm, not the AVX-512 one** — it takes
`.cargo/config.toml` (v3), so a green `native` leaves every `_mm512_*` body
unwitnessed. The AVX-512 arm is the binary run under the v4 config directly:

```sh
cd crates/simd-masking-parity
env -u RUSTFLAGS cargo --config ../../.cargo/config-v4.toml run --release
```

Read the program's own header line to confirm which arm you actually got
(`avx512f=true`, `neon=true`, …) — that line exists precisely because the
config can silently not apply. Five of the six realizations are reachable
without nightly (AVX2, AVX-512, NEON, wasm-simd128, wasm-scalar); only
`nightly-simd` needs a toolchain this repo does not pin. The same lesson the
sibling `lance-graph-java` records for the JDK: **a stale index or a missing
helper binary reporting absence is not evidence of absence.**

### Key Data
- 5 Qwen3.5 models indexed: 685 MB bgz7 from 201 GB BF16 safetensors
- GitHub Release `v0.1.0-bgz-data` on AdaWorldAPI/lance-graph: 41 bgz7 files
- 4 diffs: FfnGate dominant (0.6%), v2 reverts v1, K stable at 27B, K shifted at 9B

### Benchmark
- SPO Palette Distance: 611M lookups/sec, 1.8 ns/lookup, 388 KB RAM
- 17K tokens/sec (triple model, 4096 heads, Pearl 2³)

### Architecture Rule
- ndarray = hardware (SIMD, Palette, Base17, SpoDistanceMatrices, read_bgz7_file)
- lance-graph = thinking (NarsTruth, NarsEngine, TripleModel, AutocompleteCache)
- causal-edge = protocol (CausalEdge64, NarsTables, forward/learn)
- p64 = convergence highway (both repos meet here)
