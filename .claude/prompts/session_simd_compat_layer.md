# SESSION: Move std::simd Polyfill from rustynum to ndarray

## Mission

Rustynum implements `std::simd` on stable Rust as `crate::simd::`.
Move it to ndarray. Refactor kernels to use it. Delete the wrong `simd_compat.rs`.

When `std::simd` stabilizes in core:
  `use crate::simd::f32x16` → `use std::simd::f32x16`
Delete `src/simd/`. Done. Every kernel unchanged.

After this move, rustynum's SIMD layer is retired.

## Source (rustynum-core/src/)

```
simd.rs              One-time CPU detect → resolves which backing to use.
                     After init: crate::simd::f32x16 works like std::simd::f32x16.
simd_avx512.rs       2643 lines. 11 types. 60 impl blocks.
                     f32x16, f64x8, u8x64, i32x16, i64x8, u32x16, u64x8,
                     f32x8, f64x4, masks. Operators, methods, traits.
                     Same names as nightly std::simd.
simd_avx2.rs         Same API. f32x16 backed by [__m256; 2].
```

## Target (ndarray)

```
src/simd/
  mod.rs             ← simd.rs
  avx512.rs          ← simd_avx512.rs
  avx2.rs            ← simd_avx2.rs
```

Wire in `src/lib.rs`: `pub mod simd;`

## Delete

- `src/backend/simd_compat.rs` — incorrectly created, wrong approach.
- Remove any reference to `simd_compat` in `src/backend/mod.rs`.

## Refactor kernels

`src/backend/kernels_avx512.rs` → rename to `src/backend/kernels.rs`.
Replace raw intrinsics with `crate::simd::` calls:

```rust
// before
_mm512_fmadd_ps(_mm512_loadu_ps(x.as_ptr()), _mm512_loadu_ps(y.as_ptr()), acc)
// after
f32x16::from_slice(x).mul_add(f32x16::from_slice(y), acc)
```

Works on both AVX-512 and AVX2 — the polyfill already resolved the tier.

## Delete dispatch! macro

`src/backend/native.rs` currently does per-call tier matching via `dispatch!`.
Not needed — the polyfill resolved at init. Remove the macro.
`native.rs` becomes thin: just calls `kernels.rs` functions directly.

## Refactor activations.rs + vml.rs

Currently scalar (`mapv` loops). Wire through `crate::simd::`:

```rust
// before (scalar)
fn sigmoid(&self) -> Array<A, Ix1> {
    self.mapv(|v| A::one() / (A::one() + (-v).exp()))
}

// after (uses std::simd polyfill)
fn sigmoid_f32(x: &[f32], out: &mut [f32]) {
    let mut i = 0;
    while i + 16 <= x.len() {
        let v = f32x16::from_slice(&x[i..]);
        let neg = f32x16::splat(0.0) - v;
        let exp_neg = simd_exp_f32(neg);
        let one = f32x16::splat(1.0);
        let result = one / (one + exp_neg);
        result.copy_to_slice(&mut out[i..]);
        i += 16;
    }
    for j in i..x.len() { out[j] = 1.0 / (1.0 + (-x[j]).exp()); }
}
```

## Tests

1. `f32x16::splat(x).reduce_sum() == x * 16.0`
2. `f32x16 a + b == element-wise add` (all operators)
3. `mul_add` matches `a * b + c` within 1 ULP
4. Refactored `dot_f32` identical to pre-refactor
5. Refactored `sgemm_blocked` identical to pre-refactor
6. `sigmoid_f32` SIMD matches scalar within 2 ULP
7. Both tiers produce identical results

## Output

Files moved: `src/simd/mod.rs`, `src/simd/avx512.rs`, `src/simd/avx2.rs`
Files deleted: `src/backend/simd_compat.rs`
Files renamed: `kernels_avx512.rs` → `kernels.rs`
Files simplified: `native.rs` (dispatch! removed)
Files wired: `activations.rs`, `vml.rs`, `bitwise.rs`
