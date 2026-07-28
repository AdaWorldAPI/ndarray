# GCC as the intrinsic-spec reference — the three-layer drill-down

> READ BY: anyone verifying what an intrinsic actually *does* before writing or
> porting a SIMD kernel — `simd-savant`, `arm-neon-specialist`, and any session
> touching `src/simd_*.rs`, the polyfill matryoshka, or a consumer-crate SIMD
> swap.
>
> **Why this file exists:** the reference below was used across the NEON /
> aarch64 / wasm planning work but was never written down. A 2026-07-27 sweep of
> every URL host in every `.md` / `.rs` / `.toml` in this repo found no
> intrinsic or ISA reference of any kind — the only spec-shaped citations were
> four bare "Intel Intrinsics Guide" mentions with no link
> (`vertical-simd-consumer-contract.md`) and one GCC file fetched off mutable
> `master` (`agnostic-surface-cpu-matrix.md` §M). The link existed only in
> session context. Recording it so it survives the session that knew it.

## The source

**`github.com/gcc-mirror/gcc`** — mirror of `gcc.gnu.org`. Pin a commit; never
cite `master` (a spec citation that moves under you is not a citation).

Working pin used for the 2026-07-27 verification:
`8f1698229d33a74cda7ec9f02575172df10b2774`

```
https://raw.githubusercontent.com/gcc-mirror/gcc/<SHA>/<path>
```

## The three layers — each answers a different question

| layer | path | answers |
|---|---|---|
| **1. declarations** | `gcc/config/i386/*intrin.h`<br>`gcc/config/aarch64/arm_neon.h` | *What builtin does this intrinsic map to, and what are its exact C types?* |
| **2. machine description** | `gcc/config/*/*.md` | *What instruction / RTL does that builtin lower to, and what are the bit encodings?* |
| **3. conformance tests** | `gcc/testsuite/gcc.target/{i386,aarch64,arm,...}/` | *What does it compute?* — an executable test per intrinsic with an **inline scalar reference implementation** |

Layer 3 is the one people don't know about and is by far the most useful: it is
a per-intrinsic **oracle**, not prose. That makes it directly usable for the
byte-parity discipline this workspace already applies elsewhere (tesseract-rs,
stockfish-rs) — port a kernel, then prove it against GCC's own definition of
what the instruction computes, rather than against a hand-written reference.

## Worked example — the question that motivated this file

`_mm256_mul_epu32` vs `_mm256_mul_epi32`, needed for a `curve25519-dalek`
SIMD swap onto `ndarray::simd`.

**Layer 1** (`gcc/config/i386/avx2intrin.h`):

```c
_mm256_mul_epi32 (__m256i __X, __m256i __Y)
{ return (__m256i) __builtin_ia32_pmuldq256  ((__v8si)__X, (__v8si)__Y); }

_mm256_mul_epu32 (__m256i __A, __m256i __B)
{ return (__m256i) __builtin_ia32_pmuludq256 ((__v8si)__A, (__v8si)__B); }
```

Both take `__v8si` and return `__m256i` — **identical C signatures**. The
signed/unsigned distinction lives entirely in the instruction (VPMULDQ vs
VPMULUDQ), not in the type. So the type system will not catch a mix-up here;
only the semantics will.

**Layer 3** (`gcc/testsuite/gcc.target/i386/avx2-vpmuludq-2.c`) gives the
semantics in one line:

```c
for (i = 0; i < 4; i++)
  r[i] = s1[i * 2] * s2[i * 2];   /* unsigned int × unsigned int → unsigned long long */
```

i.e. **even 32-bit lanes only** (equivalently: the low 32 bits of each packed
64-bit element), widening to 4 × u64. The odd lanes are ignored entirely.
`avx2-vpmuldq-2.c` sits beside it as the signed twin — the pair is a ready-made
differential oracle.

## Honest limits

- **Test filenames are not guessable.** `avx2-vpmuludq-2.c` and
  `avx2-vpermd-1.c` exist; `avx2-vpsrlvd-1.c` does not (that op is tested under
  a different name). Get a real directory listing rather than probing guesses.
- **The GitHub *API* is scoped to session repos**, so `gcc-mirror/gcc` cannot be
  listed through it from here; `raw.githubusercontent.com` fetches work fine.
- **GitHub *code search* excludes forks** — it silently returns 0 for files that
  demonstrably exist in this org's forked repos. Verified 2026-07-27:
  `filename:harvest_network.rs` returned 0 while the file exists at
  `AdaWorldAPI/ruff:crates/ruff_cpp_spo/examples/harvest_network.rs`. **Never
  treat a code-search miss on a fork as evidence of absence** — clone and grep.
- Layer 3 covers what an intrinsic *computes*. It does not give latency or
  throughput; that is a separate question and a separate source.

## Not harvested

There is no baked/harvested copy of this spec set in the workspace. Checked
2026-07-27 by local grep (not code search — see above): `ndarray` (tree +
releases — it has none), `lance-graph` (tree + all 11 releases, which are model
codebooks / corpus / topology data), `ruff` (5 harvest examples, all
Tesseract/Leptonica C++), `curve25519-dalek`. The reference is consulted
directly at a pinned SHA.

Worth noting for anyone who wants one: `ruff_cpp_spo` already walks C++ headers
via libclang and emits SPO manifests, and GCC's `*intrin.h` are plain C headers
of exactly that shape — so a `harvest_intrinsics.rs` producing the full
intrinsic → builtin → instruction table would be an extension of proven
machinery rather than new machinery. Not built; recorded as an option.
