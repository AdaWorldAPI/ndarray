# ChaCha20 SIMD via the `U32x16` ARX lane — matryoshka usage + open work

> Handoff doc. What this branch shipped, what's deferred, and the **matryoshka**
> plan for *using* the `U32x16` lane to accelerate ChaCha20 without owning any
> crypto algorithm. Operator-ratified design (2026-07-11 session).

## The doctrine (why this shape)

- **RustCrypto owns the algorithm; ndarray owns the SIMD.** We do NOT hand-roll
  ChaCha20/Poly1305/the XChaCha20-Poly1305 AEAD. Rolling your own AEAD (HChaCha20
  + Poly1305 + framing) is the footgun; it is forbidden.
- The `encryption` crate (`crates/encryption`, RustCrypto wrappers) stays
  **unchanged**. See its `aead.rs` module doc for why the AEAD is not re-wired.
- The only crypto-algorithm SIMD we accelerate is the **ChaCha20 keystream ARX
  core**, and even that is done by feeding RustCrypto's *own* backend a new
  SIMD lane — never by re-implementing the cipher.

## DONE on this branch (the lane is ready)

`ndarray::simd::U32x16` now carries the full ARX triple — `Add` + `BitXor` +
`rotate_left` — on every tier, so a ChaCha20 backend can ride it:

| tier | `U32x16` backing | `rotate_left` | verified |
|---|---|---|---|
| avx512 (server) | `__m512i` | `_mm512_rolv_epi32` (VPROLVD) | executed, parity-green |
| wasm128 (browser) | `[U32x4; 4]` (`v128`) | `v128_or(u32x4_shl, u32x4_shr)` | full-lib wasm build + **node parity** |
| avx2 | `[u32; 16]` polyfill (native 2×`__m256i` = deferred TD-SIMD-3) | `u32::rotate_left` loop | compiles; == reference |
| scalar / nightly | `[u32; 16]` / `core::simd` | `u32::rotate_left` / shift-or | completeness tier + reference |

CI: `wasm_simd` job (`.github/workflows/ci.yaml` + `scripts/wasm-parity.sh` +
`crates/wasm-simd-parity/`) builds the real `ndarray::simd` wasm types and runs a
lane-by-lane parity selfcheck under **node** — the standing guard for the wasm
tier (invisible to the x86 `cargo test`). Extend its per-lane blocks as lanes land.

The stand-alone `src/simd_crypto.rs` was the *interim* primitive; the matryoshka
below **superseded it** — it is now **RETIRED** (2026-07-12, see below).

## ✅ SHIPPED 2026-07-12 — fork landed + `simd_crypto.rs` retired

- **Native NEON `U32x16 = [U32x4; 4]`** — `U32x4` gained `bitxor`(veorq_u32) +
  `rotate_left`(vshlq_u32 shift-or); composed `U32x16` with Add/BitXor/
  rotate_left. aarch64 now `cargo check`-clean on stable (also fixed the
  pre-existing `u16x8` alias + the nightly-only `vdotq_s32` breakage → stable
  widening NEON). Every tier of the ARX lane is now native.
- **The fork** lives at `vendor/chacha20/` (name/version kept `chacha20`/`0.9.1`,
  excluded from the workspace, own `[workspace]` table). Upstream cipher verbatim;
  the ONE delta is `src/backends/ndarray_simd.rs` — the transpose block16 (16
  blocks ∥, word→16-lane vector, counter lane-index) over `ndarray::simd::U32x16`,
  pure `+`/`^`/`rotate_left`, **no raw intrinsics, no `unsafe`**. Selected at
  compile time under `cfg(all(x86_64, avx512f))` in `backends.rs` +
  `lib.rs::process_with_backend` (Tokens/new/dispatch — four cfg branches). The
  `ndarray` dep is `target.'cfg(target_arch="x86_64")'` + `default-features=false,
  features=["std"]` so wasm/aarch64 fork builds never pull ndarray.
- **`[patch.crates-io] chacha20 = { path = "vendor/chacha20" }`** in the ndarray
  root `Cargo.toml` folds it under `chacha20poly1305 → encryption` transparently.
- **Parity — triple-gated, GREEN:** the fork's own RustCrypto RFC 8439 vectors
  (`chacha20_encryption`/`_keystream`, `xchacha20_*`, `chacha20_core`, seek) pass
  **through `ndarray_simd`** under `-Ctarget-cpu=x86-64-v4`; `cargo test -p
  encryption` (23 AEAD tests incl. XChaCha20-Poly1305 round-trip / bit-flip /
  wrong-key) passes on both the default (v3→RustCrypto avx2 fallback) and avx512
  (→`ndarray_simd`) builds.
- **RETIRED:** deleted `src/simd_crypto.rs` + `tests/chacha20_rustcrypto_parity.rs`
  + the `chacha20 = "0.9"` dev-dep; removed the `ndarray::simd::{chacha20_block,
  chacha20_keystream, chacha20_state}` surface. RustCrypto owns the cipher; ndarray
  owns only the `U32x16` lane. `aead.rs` doc updated to the transitive-acceleration
  framing.

### Reality note + remaining follow-ups
- **The workspace default is `target-cpu=x86-64-v3` (AVX2), NOT v4** — the CLAUDE.md
  "v4 mandatory" line is stale. So `ndarray_simd` activates only on an **avx512
  build** (`.cargo/config-avx512.toml` / `-Ctarget-cpu=x86-64-v4`); the default v3
  build uses RustCrypto's own avx2 backend (fast, vetted). This is correct and
  safe — the matryoshka is the *server-avx512* accelerator.
- ~~**wasm browser backend.**~~ **DONE 2026-07-12** — the fork's `ndarray_simd`
  cfg gates were widened to `any(all(x86_64, avx512f), all(wasm32, simd128))` in
  `backends.rs` + `lib.rs::process_with_backend`, and the ndarray dep to
  `cfg(any(x86_64, wasm32))`. The SAME `ndarray_simd.rs` source now drives the
  AVX-512 `__m512i` lane (server) and the native wasm `[U32x4;4]` lane (browser).
  Verified: `cargo build --manifest-path vendor/chacha20/Cargo.toml --target
  wasm32-unknown-unknown --lib` with `+simd128` compiles ndarray-for-wasm + the
  wasm backend clean; x86 avx512 vectors still 9/9 green after the widening. Bit-
  exact by composition (the `ndarray_simd` source is RFC-proven on x86; the wasm
  `U32x16` lane is node-proven by the `wasm_simd` CI job). A CI compile-guard step
  was added to the `wasm_simd` job. A consumer building `encryption` as a wasm
  cdylib with `+simd128` now gets the accelerated keystream transparently.
- **cross-repo `[patch]` (next):** `[patch]` does not transit, so MedCare-rs (and
  any other consumer building `encryption` for avx512) needs its own
  `[patch.crates-io] chacha20 = { path = "vendor/chacha20" }` pointing at a vendored
  copy of the fork, to inherit the acceleration. Documented, low-risk.

## DONE / DEFERRED

1. ~~**Native neon `U32x16 = [U32x4; 4]`.**~~ **DONE 2026-07-12** — `U32x4` gained
   `bitxor`(veorq_u32) + `rotate_left`(vshlq_u32 shift-or, n%32 guard); composed
   native `U32x16([U32x4;4])` with Add/BitXor/rotate_left; `F32x16::to_bits/
   from_bits` → from_array/to_array; `simd.rs` aarch64 arm re-exports the native
   lane. (+ fixed the pre-existing aarch64 stable-compile breakage.)
2. ~~**aarch64 cross parity CI.**~~ **DONE 2026-07-12** — `crates/neon-simd-parity`
   (excluded bin, real `ndarray::simd` aarch64 types) + `scripts/neon-parity.sh`
   (cross-build aarch64 + run under `qemu-aarch64-static`) + CI `neon_simd`
   (`neon-simd/parity-qemu`) job, added to the `conclusion` needs. Runtime-verifies
   U32x16 ARX (rotate 16/12/8/7 + edges) / F32x16 / I8x16 == scalar. Twin of
   `wasm_simd`. **Green locally under qemu.**
3. **avx2 native `U32x16`** (2×`__m256i`) — the TD-SIMD-3 lowering; still optional,
   the scalar polyfill is correct meanwhile.
4. **wasm matryoshka backend + cross-repo `[patch]`** — see the SHIPPED section's
   follow-ups above.

## The MATRYOSHKA — how the lane gets USED (the finalization)

Goal: ChaCha20 (and thus the whole `encryption` AEAD stack) accelerated on
server (AVX-512) + browser (wasm128), with **essentially zero owned crypto code**
and a **one-file** delta over RustCrypto that is trivial to re-sync on security
updates.

**Structure (nesting):**

```
RustCrypto chacha20  (rounds / constants / counter / StreamBackend / TESTS — VERBATIM, vetted)
   └─ one backend's round body expressed over → ndarray::simd::U32x16  (Add / BitXor / rotate_left — generic, NOT crypto)
                                                   └─ dispatched by ndarray's polyfill → avx512 / wasm128 / neon / scalar
```

**Steps:**

1. **Fork `chacha20`** (the RustCrypto stream-cipher crate; `AdaWorldAPI/chacha20`,
   or vendor). Its backends live in `src/backends/{soft,sse2,avx2,neon}.rs`, each a
   `struct Backend<R>` impl of `StreamBackend` with `gen_ks_block` /
   `gen_par_ks_blocks`. **Clone `avx2.rs` once**; keep soft/sse2/avx2/neon
   untouched.
2. **Rewire the clone over `ndarray::simd::U32x16`** — the word-sliced ChaCha
   round (`state[a] = state[a] + state[b]; state[d] = (state[d] ^ state[a]).rotate_left(16);`
   …) written against `U32x16`, `ParBlocksSize = U16`. `ndarray` lowers it to
   AVX-512 (server) / wasm128 (browser) / NEON / scalar automatically. The
   carried file has **no `unsafe`, no intrinsics** — all of that lives once in
   `ndarray::simd` (audited once; only AMX+F16 are byte-asm, neither on this path).
3. **Dispatch:** add the generic backend to the fork's `lib.rs` selection.
   Per the ndarray model this is **compile-time** (`cfg(target_feature)`), not
   `is_x86_feature_detected!` — server built `x86-64-v4`, browser built `+simd128`.
   Non-portable per-target binaries by design (SIGILL/validation-fail on a CPU
   built-for-but-absent) — matches how the servers/browsers are actually built.
4. **`[patch]` the fork in** (the pattern already used in MedCare-rs for
   `encryption`): `chacha20poly1305 → encryption → ogar-encryption → consumers`
   all accelerate **transitively**, with **zero** change to the AEAD or any
   consumer, because the AEAD just calls `apply_keystream` which now dispatches
   to the new backend. HChaCha20 + Poly1305 + framing stay 100% RustCrypto.
5. **Gate bit-exact vs RustCrypto's own `soft` backend** + run its stock test
   vectors — the same "reference + KAT, then vectorize with parity" discipline.
6. **Retire `src/simd_crypto.rs`** — the rounds now come from RustCrypto; only
   the `U32x16` lane (not crypto) is ours.

**Maintenance win:** a RustCrypto security advisory almost never touches the ARX
lane ops (fixes land in framing/counter/AEAD — all pristine upstream), so the
one-file delta re-applies with ~zero conflict: bump the vendored `chacha20`,
re-apply the one backend file, run the stock vectors + the soft parity. And if
the backend is ever **upstreamed** (RustCrypto would plausibly take an avx512 /
wasm128 backend), the fork evaporates → maintenance goes to zero. AMX has no
ChaCha/Poly backend (it is a matrix engine); the upstream targets are avx512 +
wasm128 (NEON already exists upstream).
