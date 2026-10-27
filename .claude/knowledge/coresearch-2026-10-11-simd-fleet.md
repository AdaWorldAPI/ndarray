# Co-research map — `ndarray::simd` × fleet, 2026-10-11

READ BY: anyone deciding the next SIMD-tier, `cpu_guard`, consumer-tier or
unsafe-inventory work; `cargo-substrate-architect`, `simd-savant`,
`sentinel-qa`.

Council: `/coresearch` (lance-graph `.claude/agents/coresearch-council.md`).
Five scouts (code cartographer, internal prior art, literature, systems,
concepts), a crosswalk of 27 ideas, three co-architects (bridge,
firewall/fit, falsifier). The map ratifies nothing. Items marked SHIPPED were
then implemented in this repo, in the same branch, because they were
ndarray-local and the session runs autonomously. Everything in another repo is
recorded, not pushed (this session pushes only to ndarray).

## The question

After #348–#353 (startup CPU guard, AVX-without-AVX2 arm, x86-64-v2 arm,
unsafe inventory), which points of the earlier "fearless_simd vs
`ndarray::simd`" and "Apple Accelerate: two halves" listings are closed,
which are loose, and which loose ends connect across the fleet
(lance-graph, OGAR, lance-graph-java, tesseract-rs, stockfish-rs, MedCare-rs,
ladybug-rs, q2, HubSPO-rs, spear, a2ui-rs, surrealdb fork, lichess-rs)?

Premise split (each answer names one): **Q-TIER** which realization a build
selects; **Q-HOST** whether a built binary runs on its deploy host;
**Q-SWEEP** consumer raw intrinsics; **Q-SOUND** unsafe soundness;
**Q-MSRV** one canonical coordinate couples the fleet floor; **Q-PRIM**
missing facade words.

## 1. The earlier listing, point by point

### A. `fearless_simd` vs `ndarray::simd`

| point | now | evidence |
|---|---|---|
| compile-time dispatch, one realization per build | CLOSED (design) | 8 realizations: avx512, avx2, avx (#349), v2 → scalar (#353), neon, wasm-simd128, wasm-scalar, scalar; nightly validation-only |
| runtime option only per kernel (`simd_runtime`) | CLOSED (design) | runtime dispatch of the core types was proposed (OQ-CSV-13, simd-dispatch-architecture Phase 5) and never built; consumer contract forbids ISA dispatch above `simd.rs`. Highway / `multiversion` / ifunc per call = second dispatch layer → SKIP |
| safety rests on build-time features; v4-on-AVX2 SIGILL | CLOSED for executables (#348) and **for library hosts (this branch)**; aarch64 not covered | see §3 |
| fixed 512-shaped names | CLOSED (design): the type is the layout. Highway's `FixedTag<T,k>` is the same contract ("may cost extra masking"), which is what lets the parity matrix promise bit-identity |
| no SSE tiers | CLOSED: avx arm + v2 arm. A build with no target-cpu still gets the AVX2 arm + floor, by operator choice |
| AVX2 integer types are scalar polyfills | **CLOSED by measurement**, not loose: blackboard (15) found array `U64x4::and` and a hand loop compile to the same `vandps ymm`; (18) closed the `U64x4/U64x2+avx512vl` build "measured-not-worth-doing". Today 11 of the 13 array-backed types have 0–8 intrinsic methods; `U16x16` 26/29 and `U8x32` 28/34 are real `__m256i`. CERFACS (arXiv 2601.04922, READ-IN-FULL) agrees: intrinsics win mainly on data-dependent compare/select. Reopen only with a named consumer workload on byte compares |
| hand-written vs generated backends | PARTIAL: only the ternlog bodies are generated (`tools/gen_ternlog_bodies.py`, 256 tables asserted vs bit-serial). `crates/ndarray-gen` is test-array generation, unrelated. TSLGen (arXiv 2407.18728) is the published form of the idea |
| per-level tests vs scalar | PARTIAL: 15 parity groups, bit-identical across 8 realizations, but **18 facade types are in no parity crate**: F32x8, F64x4, I16x16, I16x32, I32x8, I64x4, I64x8, I8x32, I8x64, U16x16, U16x32, U16x8, U32x8, U64x4, U8x32, U8x8, F16x16, BF16x16 |
| MSRV | PARTIAL, resolved by fact: `rust-version = "1.99.0"` (2026-10-10) and the fleet's default branches moved to 1.99.0 the same day, which executed the "fleet-wide bump" option of lance-graph `ISS-NDARRAY-CANONICAL-COORDINATE-COUPLES-FLEET-MSRV` |
| one binary everywhere, no SIGILL | EXCLUDED as API (operator); the A1-compatible form is a choice ONCE before `main` (glibc-hwcaps, Docker multi-slice) — see §4 |

### B. "Three places the CPU leaks back"

| point | now |
|---|---|
| the build cares (no startup guard) | CLOSED (#348) |
| width fixed, speed not | CLOSED by measurement (above), plus a NEW instance: consumers do not inherit ndarray's `.cargo/config.toml` (Cargo reads config only from the invoking dir and parents; `RUSTFLAGS` replaces every config flag). tesseract-rs Docker, stockfish-rs, HubSPO, spear, OGAR, lichess build baseline x86-64 outside the intrinsic bodies; lance-graph pins v3 and measured baseline as 4.5× slower on its Goursat probe |
| runtime dispatch covers some kernels | CLOSED (design, anchor: startup-only checks) |

### C. Accelerate mapping

The mapping stands. Still LOOSE and unowned: about half of `src/hpc/` is
domain logic (nars, qualia, deepnsm, cam_pq, clam, gpt2, stable_diffusion),
`hpc-extras` gates six modules, and consumers import domain modules
(`hpc::splat3d`, `hpc::clam`, `hpc::bf16_truth`) across the fleet. PARK
until someone owns a cross-repo plan.

### D. Unsafe inventory

| point | now |
|---|---|
| 4 confirmed safe-code OOB paths | **CLOSED** (dot_i8 #349; row_mut, sgemm/dgemm_blocked, int8_gemm_amx_tiled this branch) |
| 212 NEEDS-TF | CLOSED as fact: E0133 for a plain fn calling even a safe `#[target_feature]` fn on 1.99 (RFC 2396 adds no reduction for facade methods) |
| AMX rung missing avx512f/vnni | **CLOSED** this branch |
| 856 missing `// SAFETY:` | LOOSE (documentation debt, lower than soundness) |
| 186 CONSOLIDATE, 139 SAFE-API, 42 REMOVABLE | LOOSE, mechanical |
| 28 SAFE-CRATE (`bytemuck`/`zerocopy`) | LOOSE, operator question; no prior ruling either way (searched: ndarray Cargo.toml + blackboard, lance-graph board + knowledge) |
| aarch64 guard | LOOSE (needs `getauxval(AT_HWCAP)`) |
| 95 `is_x86_feature_detected!` in 17 files | LOOSE: classify each as startup/once-only (sanctioned) vs per-call inside SIMD code (forbidden) |

### E/F. Review issues and SSE follow-ups

All three Codex issues on #348 CLOSED. SSE follow-ups: `U8x64::cmpeq_mask`
146 instructions on AVX1 and `F32x16::mul_add` → 16 software `fmaf` on
non-FMA tiers stay LOOSE (correct, slow; no consumer on those CPUs named).

## 2. New findings (not in any earlier listing)

| id | finding | sub-q |
|---|---|---|
| X1 | **`cpu_guard` killed library hosts.** Measured: an `x86-64-v4` cdylib linking ndarray, `dlopen`ed by a C host under `qemu -cpu Haswell`, ended the host with 132. Affects lance-graph-java's `.so` in a customer JVM and tesseract-ocr-python's wheel. **SHIPPED fix** (below) | Q-HOST |
| X2 | A slice built with no target-cpu gets the AVX2 floor, so ladybug's "generic" binary exits 132 on any non-AVX2 host: its fallback floor is v3, not generic | Q-HOST |
| X3/X4 | ladybug's multi-slice image picks by `grep avx512f /proc/cpuinfo`; v4 also needs bw/cd/dq/vl. The exact selector already exists: the guard's own predicate | Q-HOST |
| X6 | MedCare-rs and q2 ship `x86-64-v4` images to Railway; Railway publishes no CPU guarantee; GCP E2 mixes Broadwell (no AVX-512); MedCare's own handover says "v4 build never run"; MedCare measured the v4 gain as µs beside a 43 ms KDF | Q-HOST |
| X7 | psABI: v4 = AVX512F/BW/CD/DQ/VL only; VNNI/BF16/AMX/VPOPCNTDQ are outside every level, so `config-avx512.toml` and holograph's `avx512vpopcntdq` are "beyond v4" | Q-TIER |
| X11 | **CLOSED (#374, #378).** Wasm `F32x16::mul_add` and `F64x8::mul_add` round once, like every other realization. Each realization is checked bit for bit against the parity crate's integer oracle (any NaN matches any NaN), and `+relaxed-simd` emits the same code section as `+simd128`. A determinism gate (lance-graph D-WFL-DET) need not exclude a single-rounding `mul_add` on cross-realization grounds; D-OIF-4's separate mul-then-add contract is unaffected. `reduce_sum` is not covered: its summation order differs per backend. The superseded observation (wasm mul-then-add, host-chosen under `relaxed-simd`) is kept in the blackboard entries of 2026-10-27. | Q-TIER |
| X16 | lance-graph `ndarray_bridge.rs` intrinsic blocks are all under `cfg(not(feature = "ndarray-hpc"))` and `ndarray-hpc` is default → dead in default builds; holograph `hamming.rs` is live (feature `simd` default, no ndarray dep); ladybug `collapse_gate.rs` is a per-call `is_x86_feature_detected!` + raw `_mm256_*` kernel in no sweep ledger | Q-SWEEP |
| X17 | the MSRV issue still lists a `rev`-pin option forbidden by `E-CONSUMER-PINS-ON-INTERNAL-SIBLINGS-PROHIBITED-1`; `CARGO-COMPUTE-SUBSTRATE.md` §3 still shows `rev = "<pinned>"`; lichess-rs says `cargo +1.95`; tesseract `rust.yml:59` says "1.95 floor" | Q-MSRV |
| X18 | OGAR pulls `encryption` (+ argon2 `ndarray-simd`) from ndarray **git master** while lance-graph uses a path: two package ids in a binary linking both. surrealdb's `ndarray = "0.17.2"` is crates.io, a different package from the fork at the same version | Q-MSRV |
| X19 | lance-graph-java `native/lgj-abi/.cargo/config.toml` says ndarray defaults to v3 (now native) and "one known host" (now BYOS); its test `abi.rs:719-729` asserts backend ∈ {AVX2, AVX-512}, so a v2 or AVX-arm build would fail it | Q-HOST |

## 3. Shipped in this branch (ndarray only)

1. **`cpu_guard` in a library host.** The pre-`main` hook exits 132 only
   when the guard's code lies in the main executable; in a loaded library it
   prints a `warning:` and returns, and the host can call
   `check_build_cpu()`. Lookup: ELF matches the object whose program headers
   are at `AT_PHDR` (`getauxval` / `elf_aux_info` + `dl_iterate_phdr`);
   macOS `dladdr` vs `_NSGetMachExecuteHeader()`; Windows
   `GetModuleHandleExW(FROM_ADDRESS)` vs `GetModuleHandleW(NULL)`. Every
   lookup failure answers "library". sentinel-qa: no BLOCK, no loader-lock
   deadlock; its two CONDITIONAL findings (bionic order, glibc `dlmopen`,
   dyld image 0) are fixed. `scripts/cpu-guard-library-probe.sh` (CI job
   `cpu-guard-library`) runs three legs under `qemu -cpu Haswell`:
   `dlopen` host survives, `dlmopen(LM_ID_NEWLM)` host survives, v4
   executable exits 132. Disable runs: the pre-fix guard fails the first
   leg; the first-object rule fails the `dlmopen` leg.
2. **Three safe-code OOB paths** closed (`row_mut`, `sgemm/dgemm_blocked`,
   `int8_gemm_amx_tiled`), tests pass in debug and release, each red in
   release when its fix is reverted.
3. **AMX rung** requires `avx512f && avx512vnni`.

## 4. Exploration map — verdicts and fruit

Verdicts: ADOPT-NOW (shipped or ready), PROBE (measure first), PARK, SKIP.

### Low-hanging (≤1 hour each)

| item | verdict | repo | why / probe |
|---|---|---|---|
| strike forbidden remedies from records: MSRV issue `rev` option; TD-NDARRAY-SIMD-POPCNT-NATIVE item (b) "runtime-detect VPOPCNTDQ"; regrade that TD (now the case `cpu_guard` reports) | ADOPT-NOW | lance-graph board | both contradict standing rulings |
| stale floors: lichess-rs `+1.95`, tesseract `rust.yml:59` | ADOPT-NOW | lichess-rs, tesseract-rs | one-line doc fixes |
| stale tier comments: lance-graph `.cargo/config.toml` "anything below v3 SIGILLs" (avx/v2 arms exist, guard reports), arm-discovery `Cargo.toml:30` recommends native/v4, lgj config "ndarray defaults to v3" | ADOPT-NOW | lance-graph, lgj | doc drift |
| consumer contract paragraph: a consumer build does not inherit ndarray's `.cargo/config.toml`; built-elsewhere artifacts name a psABI level | ADOPT-NOW | ndarray `.claude/knowledge/vertical-simd-consumer-contract.md` | X5/X6 |
| `mul_add` caveat in the facade docs; D-WFL-DET excludes it | ADOPT-NOW (doc) + PROBE | ndarray, lance-graph | probe: `a=1+2⁻²³, b=1−2⁻²³, c=−1` gives −2⁻⁴⁶ fused, 0 unfused; run on all arms |
| lgj: call `check_build_cpu()` in the `Abi` handshake and raise a Java exception | ADOPT-NOW | lance-graph-java | the warning path now exists |

### High-hanging (a session each)

| item | verdict | repo | probe |
|---|---|---|---|
| parity groups for the 18 uncovered facade types, table-generated, seeded random inputs + edge values (shifts and saturating ops are where stdarch bugs were found, arXiv 2606.17374) | PROBE then ADOPT | ndarray | disable: flip one scalar `cmpgt` in a new group; it must go red |
| fleet tier policy: deploys default to v3 unless they ship slices; MedCare/q2 v4 Railway images → v3 (measured gain negligible) or v4/v3/v2 slices | PROBE | MedCare-rs, q2 | read Railway `/proc/cpuinfo` at deploy; run the v4 image under `qemu -cpu Haswell` (must exit 132 at start, not mid-request) |
| slice selector from the guard: `--cpu-check`-style exit-code probe per slice, pick the first slice whose guard passes; ladybug's "generic" slice built with `config-v2.toml` | PROBE | ndarray (probe API) + ladybug | KnightsMill (avx512f without bw/vl) separates the grep selector from the guard selector under qemu |
| classify the 95 `is_x86_feature_detected!` sites (startup / once / per-call) as an inventory column | ADOPT | ndarray | ladybug `collapse_gate.rs` must classify per-call and `simd_runtime` once, or the classifier is wrong |
| consumer sweep: delete lance-graph `ndarray_bridge.rs` dead blocks; add holograph and ladybug `collapse_gate.rs` to the sweep ledger; TD-SIMD-SWEEP-W1..W5 | ADOPT (other repos) | lance-graph, ladybug | grep is navigation; a CI membrane test (like lgj's `ApiSurfaceTest`) is evidence |
| aarch64 guard via `getauxval(AT_HWCAP/HWCAP2)` | ADOPT | ndarray | dotprod build under `qemu-aarch64 -cpu cortex-a53` must report, not SIGILL |
| OGAR git-master vs lance-graph path: two ndarray package ids | PROBE | OGAR | `cargo metadata` on a binary linking both counts distinct ndarray ids |

### Heavy-hanging (multi-session, needs an owner or a ruling)

| item | verdict | why |
|---|---|---|
| Accelerate-core vs domain split of `src/hpc/` | PARK | cross-repo plan, consumers import domain modules directly |
| one-spec generated backends (TSLGen / fearless_simd_gen style) beyond ternlog | PARK | large; parity matrix already catches drift; revisit if a ninth realization appears |
| `bytemuck`/`zerocopy` for the 28 SAFE-CRATE rows | operator question | licences fit (Zlib/MIT/Apache, BSD-2/Apache/MIT); fork policy applies |
| Kani function contracts for index paths | PARK | needs a nightly CI-only job; 721 LLVM SIMD intrinsics unmodeled |
| glibc-hwcaps layout for the lgj `.so` | PROBE | does Panama's absolute-path `dlopen` search `glibc-hwcaps/`? Probably not; lgj's own picker would use `check_build_cpu()` |
| runtime dispatch of core types | SKIP | conflicts with the startup-only / compile-time anchors |

## 5. Not searched (so "nothing found" is not "nothing exists")

SIMDe and xsimd test methodology; Apple universal-binary practice in detail;
AWS and Hetzner CPU pages; tokio/rustls MSRV policies and `cargo-msrv`;
GCC's function-multiversioning page (HTTP 403, all ifunc claims SECONDHAND);
Windows loader documentation (sentinel-qa cited from memory); the argon2
fork's own ndarray dependency line; whether any fleet binary is built with
`+crt-static` or non-PIE (the libc test assumes neither).
