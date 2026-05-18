# W2-3 + W2-4 Audit — BLAS levels + statistics ArrayView compliance

## Verdict
**CLEAN.** No follow-up wave needed for `blas_level{1,2,3}.rs` or `statistics.rs`. All four files are already ArrayView-shaped via trait impls on `ArrayBase`.

## Per-file findings

### `src/hpc/blas_level1.rs`
- Trait impl on ArrayBase: **yes, L47** — `impl<A, S> BlasLevel1<A> for ArrayBase<S, Ix1>`
- Bonus trait impls (not in the original migration doc, but clean): `ScalarArith` (L196), `VecArith` (L242) — both on `ArrayBase<S, Ix1>`
- Slice-taking pub fns: **1** — `blas_rotg` (L152). **OK-as-is**: signature is `(a: A, b: A)` (scalars), not slices. The regex `^pub fn .*&\[` matched a `&[` in the doc-comment example, not the signature.
- `axis_iter` misuse: **0**
- Bridge pattern: verified present in trait methods — `blas_dot`, `blas_axpy`, `blas_scal`, `blas_nrm2`, `blas_asum` all dispatch through `as_slice()` hot path + stride-aware cold path.

### `src/hpc/blas_level2.rs`
- Trait impl on ArrayBase: **yes, L97** — `impl<A, S> BlasLevel2<A> for ArrayBase<S, Ix2>`
- Slice-taking pub fns: **0**
- `axis_iter` misuse: **0**

### `src/hpc/blas_level3.rs`
- Trait impl on ArrayBase: **yes, L59** — `impl<A, S> BlasLevel3<A> for ArrayBase<S, Ix2>`
- Slice-taking pub fns: **0**
- `axis_iter` misuse: **0**

### `src/hpc/statistics.rs`
- Trait impl on ArrayBase: **yes, L65** — `impl<A, S, D> Statistics<A> for ArrayBase<S, D>` (note: generic-D, unlike BLAS L1/L2/L3 which fix `Ix1`/`Ix2`)
- Slice-taking pub fns: **0**
- `axis_iter` misuse: **0**

## Build verification
`cargo check -p ndarray --no-default-features --features std` → clean (31.82s, no warnings).

## Surprises
- `blas_level1.rs` carries two extra trait impls (`ScalarArith`, `VecArith`) on `ArrayBase<S, Ix1>` beyond `BlasLevel1` itself. Not mentioned in the original migration doc but clean and consistent with the two-layer rule.
- `blas_rotg` regex match was a false positive (doc-comment `&[` in an example, not in the signature).

## Follow-up needed
**None.** W2-3 and W2-4 require no code changes. The W2 sprint scope reduces to the three converter waves: W2-1 (reductions), W2-2a (vml), W2-2b (activations).
