# KNOWLEDGE: Taking `ndarray::simd` to the Browser

## READ BY:
- Any contributor wiring a consumer crate to `wasm32-unknown-unknown` and expecting `ndarray::simd` to vectorise there
- `savant-architect` agent — when a SIMD addition must also hold on the wasm32 backend
- `sentinel-qa` agent — when auditing a claim of the form "this is SIMD-accelerated in the browser"
- Any contributor about to *measure* whether a wasm artifact carries SIMD

## P0 TRIGGERS:
- About to write "the browser gets SIMD" in a PR body → read §"Verifying it, without fooling yourself"
- A wasm build compiles but the vector path seems absent → read §"Two ways this silently does nothing"
- About to count SIMD instructions with a text window around a function name → read §"The measurement trap"

---

## Why this doc exists

The polyfill's whole promise is that one source line vectorises on every
target. On x86 and ARM that promise is nearly free: even without the
polyfill, an autovectoriser has registers to aim at, so a scalar loop
often *happens* to become SIMD and nobody notices the difference.

**wasm32 is the target where the promise stops being free.** Without
`+simd128` there are no vector registers at all, so the autovectoriser
cannot find anything, no matter how well-shaped the loop is. The browser
is therefore the one place where "we rely on LLVM" degrades all the way
to scalar — silently, with no build error and no warning.

That makes wasm both the strongest argument for routing consumer math
through `ndarray::simd` **and** the easiest place to believe you have
done so when you have not. This doc is the recipe plus the two ways it
silently fails, both hit in practice on 2026-08-14 while wiring
`AdaWorldAPI/a2ui-rs`'s GPU field renderer.

---

## The recipe

### 1. Depend on the fork, with `std`

```toml
# `std` is the MINIMUM that exposes the lane types: `src/lib.rs` gates the
# module as `#[cfg(feature = "std")] pub mod simd`. `default-features =
# false` then keeps `hpc-extras` out, which matters for a wasm payload.
ndarray = { git = "https://github.com/AdaWorldAPI/ndarray", branch = "master",
            default-features = false, features = ["std"] }
```

Two details that cost a build each if guessed:

- **The default branch is `master`, not `main`.** `branch = "main"` fails
  with `cannot locate remote-tracking branch 'origin/main'`.
- **`default-features = false` alone removes `simd`.** The module lives
  behind `std`; without it the import is `unresolved import ndarray::simd`
  and the cause is not obvious from the error.

For local work against a sibling checkout, redirect the git coordinate in
`.cargo/config.toml` rather than editing the manifest — a path in the
manifest ships to CI and container builds, which have no sibling:

```toml
[patch."https://github.com/AdaWorldAPI/ndarray"]
ndarray = { path = "../ndarray" }
```

### 2. Build with `+simd128`

```bash
RUSTFLAGS='-C target-feature=+simd128' \
  cargo build --target wasm32-unknown-unknown --release
```

or use the checked-in profile, which exists for exactly this:

```bash
cargo build --target wasm32-unknown-unknown --config .cargo/config-wasm.toml
```

Without the flag the dispatch in `src/simd.rs` falls through to the scalar
arm — correct results, no vectors. That is a deliberate design (the code
must still *work* on a browser without SIMD support), which is precisely
why nothing complains.

### 3. Write against the lane types, not intrinsics

```rust
use ndarray::simd::F32x16;
const LANES: usize = 16;

let mut i = 0;
while i + LANES <= n {
    let x = F32x16::from_slice(&xs[i..]);
    let v = (F32x16::from_slice(&vs[i..]) + f / m) * damping;
    v.copy_to_slice(&mut vs[i..]);
    i += LANES;
}
while i < n {
    scalar_step(i);   // the tail — and the parity reference, see below
    i += 1;
}
```

The consumer crate can stay `#![forbid(unsafe_code)]`: every intrinsic
lives behind the polyfill boundary.

### What the wasm backend actually provides

`src/simd.rs` re-exports the v128-backed types from `simd_wasm::wasm32_simd`
under `all(target_arch = "wasm32", target_feature = "simd128")` —
`F32x16`, `F64x8`, `I8x16`, `U32x16` and their masks. The long-tail
integer and 256-bit-shaped types come from the scalar fallback. That is
the same split `simd_neon` uses on aarch64: native float kernels, scalar
for the rest. Reaching for `I16x32` on wasm compiles and runs, but it is
scalar; only the four above are vectors.

---

## Two ways this silently does nothing

Both were hit on the same afternoon, and neither produces a warning.

### (a) The flag is missing

Covered above: scalar fallback, correct results, zero vectors. The tell is
a SIMD instruction count of exactly **0** in the artifact.

### (b) The code is not in the artifact at all

This one is nastier and is specific to `cdylib` (i.e. every browser build).
**Only exported items survive the link.** A crate can compile, link, and
emit a `.wasm` in which the entire subsystem you care about is absent,
because nothing reachable from a `#[wasm_bindgen]` export calls it.

Measured instance: a field renderer whose client struct carried no
`#[wasm_bindgen]` attribute at all. The `.wasm` was 1.27 MB, contained
**2** SIMD instructions and **zero** symbols from the layout module — while
the same code compiled as an rlib carried **800** in one function. Adding
the exports took the module to 4.88 MB and **13 375** SIMD instructions.

So `crate-type = ["cdylib", "rlib"]` is necessary and **not sufficient**.
If a wasm SIMD measurement comes out near zero, check whether the code is
in the module *before* concluding the flag failed:

```bash
wasm-objdump -x module.wasm | grep -ci 'your_crate.*your_module'
```

---

## Verifying it, without fooling yourself

The dispatch is a `cfg`, so it is checkable rather than assumable. Build
both ways and count:

```bash
# wabt provides wasm-objdump; llvm-objdump is unusably slow on .wasm
apt-get install -y wabt

RUSTFLAGS='-C target-feature=+simd128' cargo build --target wasm32-unknown-unknown --release
wasm-objdump -d target/wasm32-unknown-unknown/release/mod.wasm \
  | grep -cE 'f32x4|i32x4|v128\.'
# a large number

cargo build --target wasm32-unknown-unknown --release      # no flag
wasm-objdump -d ... | grep -cE 'f32x4|i32x4|v128\.'
# 0
```

The **zero** in the second run is what makes the first meaningful. A count
without its contrast proves nothing: 0xFD is also an ordinary data byte, and
any large module contains vectorised code from dependencies.

### The measurement trap

Do **not** isolate a function by grepping a text window around its name:

```bash
#  WRONG — /integrate/ matches a string, not a function boundary, and an
#  rlib is full of ndarray's own vectorised functions. The number can be
#  almost entirely somebody else's code.
llvm-objdump -d lib.rlib | awk '/integrate/{f=1} f&&/^$/{f=0} f' | grep -c v128
```

Ask the symbol table instead:

```bash
SYM=$(llvm-nm lib.rlib | grep 'Layout9integrate17' | awk '{print $3}')
llvm-objdump -d --disassemble-symbols="$SYM" lib.rlib | grep -cE 'f32x4|v128\.'
```

The text-window form was used once here and came out **801** where the
symbol-scoped answer is **800**. It was right by luck — and a measurement
that is right by luck is not a measurement. What made the error visible was
an unrelated contradiction (2 instructions in the shipped module vs 801 in
the rlib), not the number itself.

---

## Cross-backend semantics — write the test with tolerance

`src/simd_wasm.rs` documents where wasm32 diverges from the other backends.
Any parity test that must pass on more than one target has to allow for it:

| | wasm32 SIMD128 | scalar fallback | AVX / NEON |
|---|---|---|---|
| `mul_add` | `mul`+`add`, two roundings (unless `relaxed-simd`) | fused | fused |
| `reduce_sum` | balanced tree | sequential fold | tree |
| `round` | round-half-to-even | half-away-from-zero | half-to-even |
| `min`/`max` with NaN | IEEE, NaN-propagating | returns the non-NaN operand | varies |

So the honest shape of a vector/scalar parity test is a **relative
tolerance**, not equality:

```rust
assert!((a - b).abs() <= 1e-4 * b.abs().max(1.0));
```

and the scalar reference should be **the same function the tail uses** —
one definition, so the two cannot drift apart.

Enabling `relaxed-simd` restores a fused `mul_add` via `f32x4_relaxed_madd`,
at the cost of a target feature not every runtime has. Prefer the tolerance.

---

## Worked consumer

`AdaWorldAPI/a2ui-rs` `crates/a2ui-graph` — a GPU field renderer whose
per-frame layout integration runs through `F32x16`. Its
`docs/WASM-INTEGRATION.md` carries the end-to-end recipe including the
`wasm-bindgen` surface; this doc is the polyfill half.
