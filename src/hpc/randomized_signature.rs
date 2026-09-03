//! `randomized_signature_sweep` — the Cuchiero-Schmocker-Teichmann randomized
//! signature recurrence, on ndarray's canonical SIMD substrate.
//!
//! This is the SIMD successor to the scalar encode loop in lance-graph's
//! `sigker::randomized::RandomizedSignatureBuilder::encode`
//! (`TD-NDARRAY-SIMD-RANDOMIZED-PROJECTION`, W1.5 item #7 of
//! `.claude/knowledge/vertical-simd-consumer-contract.md`).
//!
//! ## Deviation from the tech-debt sketch — read this first
//!
//! The `TD-NDARRAY-SIMD-RANDOMIZED-PROJECTION` entry sketches the API as
//! `impl F32x16 { fn random_proj_step(state, seed, depth) -> Self }` — a
//! per-lane `f32` state that re-derives its Gaussian entries from a seed on
//! every step. The real consumer does neither of those things:
//!
//! * the carrier is **`f64`** (`Vec<f64>` state, `Vec<f64>` flattened
//!   matrices), not `f32` — so the vector type is [`F64x8`], not `F32x16`;
//! * the random projections are materialized **once per encoder instance**
//!   (`RandomizedSignatureBuilder::new` fills `matrices` / `biases` from a
//!   seeded SplitMix64 + Box-Muller) and then reused across every path and
//!   every step, so a step primitive must *consume* those buffers, never
//!   re-derive them from `(seed, depth)`;
//! * the state dimension `k` is a runtime value (32 … 4096 in the consumer's
//!   own tests and performance envelope), not a compile-time lane count, so
//!   the hot loop is a `k×k` GEMV per path dimension — not a single register
//!   update.
//!
//! The sketch predated reading `sigker/src/randomized.rs`; this module matches
//! the real algorithm. Same correction shape as `signature_pde` (which the
//! sketch had as `f32`/`F32x16` while sigker was `f64`/`Vec<Vec<f64>>`).
//!
//! ## The recurrence
//!
//! For a path `X = (x_0, …, x_T)` in `R^d` and state dimension `k`, the
//! randomized signature evolves `z in R^k` by
//!
//! ```text
//! z_{t+1} = z_t + sum_{i=1..d} sigma(A_i . z_t + b_i) * dx_t^(i)
//! dx_t    = x_{t+1} - x_t
//! z_0     = 0
//! ```
//!
//! with `A_i in R^{k×k}` and `b_i in R^k` drawn once from `N(0, 1/k)` and
//! `sigma` any non-polynomial activation (the consumer uses `tanh`).
//! Cuchiero et al. (2021) show `X -> z_T` is a universal approximator on path
//! space, so this is a fixed-width `O(T·d·k^2)` stand-in for the full
//! signature.
//!
//! ## What is SIMD here
//!
//! Two of the three inner loops are lane-parallel and both run on [`F64x8`]:
//!
//! * **the GEMV row** `sum_col A[i][row][col] * z[col]` — an
//!   [`F64x8::mul_add`] accumulator over the contiguous row, reduced once via
//!   [`F64x8::reduce_sum`];
//! * **the axpy** `z_next[row] += activated[row] * dx_i` — a splatted
//!   `mul_add` over the whole state.
//!
//! The activation itself stays scalar: it is `O(k)` against the GEMV's
//! `O(k^2)`, and keeping `sigma` a plain `Fn(f64) -> f64` closure means the
//! primitive imposes no activation of its own (per the contract's
//! closure-parameterized batch shape) *and* stays bit-identical to the
//! consumer's `f64::tanh` instead of an approximated vector transcendental.
//!
//! ## Numerics
//!
//! Not bit-identical to a naive scalar loop, by construction: the GEMV row is
//! summed as eight independent partial sums reduced at the end, and the
//! products are fused (`mul_add`). Both change only the rounding, at the
//! `f64` ULP level; the module's parity tests assert a `1e-9` relative
//! agreement with an independent scalar oracle, matching `signature_pde`'s
//! predeclared tolerance. `reduce_sum`'s pairwise order also differs between
//! backends, so the *tolerance*, not bit-equality, is the cross-backend
//! contract.
//!
//! ## Semantics at the edges
//!
//! * an empty or single-point path has no increments — the state stays
//!   `z_0 = 0` and a `k`-length zero vector is returned;
//! * a coordinate whose increment satisfies `|dx_i| < 1e-15` is **skipped
//!   entirely** (its whole GEMV is elided), exactly as the consumer does —
//!   this is a behavioural mirror, not an optimization we invented, and a
//!   constant path therefore returns exactly zeros;
//! * `state_dim == 0`, a path-point dimension of `0`, or `matrices`/`biases`
//!   whose lengths disagree with `d * k * k` / `d * k` all panic. Lengths are
//!   checked, never inferred.

use crate::simd::F64x8;

/// SIMD lane width of the sweep (matches [`F64x8`]).
const LANES: usize = 8;

/// Below this magnitude an increment coordinate contributes nothing and its
/// `k×k` GEMV is skipped. Mirrors `sigker::randomized`'s own `1e-15` guard —
/// changing it changes results, so it is a contract constant, not a tunable.
pub const INCREMENT_EPSILON: f64 = 1e-15;

/// `dot(row, z)` with an eight-wide `mul_add` accumulator plus a scalar tail.
#[inline]
fn dot_simd(row: &[f64], z: &[f64]) -> f64 {
    debug_assert_eq!(row.len(), z.len());
    let mut acc = F64x8::splat(0.0);
    let mut c = 0usize;
    while c + LANES <= row.len() {
        let a = F64x8::from_slice(&row[c..c + LANES]);
        let b = F64x8::from_slice(&z[c..c + LANES]);
        acc = a.mul_add(b, acc);
        c += LANES;
    }
    let mut sum = acc.reduce_sum();
    while c < row.len() {
        sum = row[c].mul_add(z[c], sum);
        c += 1;
    }
    sum
}

/// `out += scale * src`, eight-wide.
#[inline]
fn axpy_simd(scale: f64, src: &[f64], out: &mut [f64]) {
    debug_assert_eq!(src.len(), out.len());
    let s = F64x8::splat(scale);
    let mut lane = [0.0f64; LANES];
    let mut r = 0usize;
    while r + LANES <= src.len() {
        let a = F64x8::from_slice(&src[r..r + LANES]);
        let o = F64x8::from_slice(&out[r..r + LANES]);
        s.mul_add(a, o).copy_to_slice(&mut lane);
        out[r..r + LANES].copy_from_slice(&lane);
        r += LANES;
    }
    while r < src.len() {
        out[r] = scale.mul_add(src[r], out[r]);
        r += 1;
    }
}

/// One increment update, written into `out` (which the caller pre-fills with
/// `z`). `activated` is a `k`-length scratch buffer, reused across steps so
/// the sweep allocates nothing per increment.
#[inline]
fn step_into<F>(
    z: &[f64], delta_x: &[f64], matrices: &[f64], biases: &[f64], activation: &F, activated: &mut [f64],
    out: &mut [f64],
) where
    F: Fn(f64) -> f64,
{
    let k = z.len();
    for (i, &dx_i) in delta_x.iter().enumerate() {
        if dx_i.abs() < INCREMENT_EPSILON {
            continue;
        }
        let a_offset = i * k * k;
        let b_offset = i * k;
        for row in 0..k {
            let row_off = a_offset + row * k;
            let sum = biases[b_offset + row] + dot_simd(&matrices[row_off..row_off + k], z);
            activated[row] = activation(sum);
        }
        axpy_simd(dx_i, activated, out);
    }
}

/// Validate the projection buffers against `(path_dim, state_dim)` and panic
/// with a message naming the mismatch.
#[inline]
fn check_shapes(path_dim: usize, state_dim: usize, matrices: &[f64], biases: &[f64]) {
    assert!(state_dim > 0, "randomized_signature: state_dim must be > 0");
    assert!(path_dim > 0, "randomized_signature: path points must have dimension > 0");
    assert_eq!(
        matrices.len(),
        path_dim * state_dim * state_dim,
        "randomized_signature: matrices must hold path_dim * state_dim^2 entries"
    );
    assert_eq!(
        biases.len(),
        path_dim * state_dim,
        "randomized_signature: biases must hold path_dim * state_dim entries"
    );
}

/// One randomized-signature step: `z + sum_i sigma(A_i . z + b_i) * dx^(i)`.
///
/// `matrices` holds the `path_dim` matrices flattened row-major and
/// concatenated (`A_i[row][col] == matrices[i*k*k + row*k + col]`), `biases`
/// the `path_dim` bias vectors concatenated — the exact layout
/// `sigker::randomized::RandomizedSignatureBuilder` materializes.
///
/// # Panics
///
/// Panics if `z` is empty, if `delta_x` is empty, or if `matrices` / `biases`
/// do not have length `delta_x.len() * k * k` / `delta_x.len() * k` where
/// `k = z.len()`.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::randomized_signature::randomized_signature_step;
/// // Zero matrices and biases: sigma(0) = tanh(0) = 0, so the state is
/// // carried through unchanged whatever the increment is.
/// let z = vec![0.5, -0.25, 1.0];
/// let out = randomized_signature_step(&z, &[2.0], &[0.0; 9], &[0.0; 3], f64::tanh);
/// assert_eq!(out, z);
/// ```
pub fn randomized_signature_step<F>(
    z: &[f64], delta_x: &[f64], matrices: &[f64], biases: &[f64], activation: F,
) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    check_shapes(delta_x.len(), z.len(), matrices, biases);
    let mut activated = vec![0.0f64; z.len()];
    let mut out = z.to_vec();
    step_into(z, delta_x, matrices, biases, &activation, &mut activated, &mut out);
    out
}

/// Encode a path into its randomized signature with a caller-chosen
/// activation.
///
/// Drop-in for `sigker::randomized::RandomizedSignatureBuilder::encode` when
/// called as
/// `randomized_signature_sweep_with(path, &self.matrices, &self.biases, self.state_dim, f64::tanh)`
/// — same buffer layout, same recurrence, same `1e-15` increment skip.
///
/// `activation` must be non-polynomial for the Cuchiero et al. universality
/// theorem to apply; the primitive does not check that (it cannot), it only
/// applies what it is given.
///
/// # Panics
///
/// Panics on `state_dim == 0`, on a path whose points have dimension `0`, or
/// on `matrices` / `biases` of the wrong length. An empty or single-point
/// path is **not** an error — it yields the zero state.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::randomized_signature::randomized_signature_sweep_with;
/// // A single-point path has no increments: the state stays z_0 = 0.
/// let z = randomized_signature_sweep_with(&[vec![1.0, 2.0]], &[0.3; 8], &[0.1; 4], 2, f64::tanh);
/// assert_eq!(z, vec![0.0, 0.0]);
/// ```
pub fn randomized_signature_sweep_with<F>(
    path: &[Vec<f64>], matrices: &[f64], biases: &[f64], state_dim: usize, activation: F,
) -> Vec<f64>
where
    F: Fn(f64) -> f64,
{
    if path.is_empty() {
        assert!(state_dim > 0, "randomized_signature: state_dim must be > 0");
        return vec![0.0f64; state_dim];
    }
    let path_dim = path[0].len();
    check_shapes(path_dim, state_dim, matrices, biases);

    let k = state_dim;
    let mut z = vec![0.0f64; k];
    let mut next = vec![0.0f64; k];
    let mut activated = vec![0.0f64; k];
    let mut delta_x = vec![0.0f64; path_dim];

    for window in path.windows(2) {
        assert_eq!(window[0].len(), path_dim, "randomized_signature: ragged path");
        assert_eq!(window[1].len(), path_dim, "randomized_signature: ragged path");
        for (a, slot) in delta_x.iter_mut().enumerate() {
            *slot = window[1][a] - window[0][a];
        }
        next.copy_from_slice(&z);
        step_into(&z, &delta_x, matrices, biases, &activation, &mut activated, &mut next);
        std::mem::swap(&mut z, &mut next);
    }
    z
}

/// Encode a path into its randomized signature with the consumer's `tanh`
/// activation — the exact map `sigker`'s `encode` computes.
///
/// See [`randomized_signature_sweep_with`] for the buffer layout, the panics,
/// and the edge-case semantics.
///
/// # Examples
///
/// ```
/// use ndarray::hpc::randomized_signature::randomized_signature_sweep;
/// // Zero projections: every activation is tanh(0) = 0, so no increment
/// // moves the state, however long the path.
/// let path = vec![vec![0.0], vec![1.0], vec![-4.0]];
/// let z = randomized_signature_sweep(&path, &[0.0; 16], &[0.0; 4], 4);
/// assert_eq!(z, vec![0.0; 4]);
/// ```
pub fn randomized_signature_sweep(path: &[Vec<f64>], matrices: &[f64], biases: &[f64], state_dim: usize) -> Vec<f64> {
    randomized_signature_sweep_with(path, matrices, biases, state_dim, f64::tanh)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only scalar oracle: the recurrence written the way
    /// `sigker::randomized::RandomizedSignatureBuilder::encode` writes it —
    /// naive row-major GEMV, sequential accumulation starting from the bias,
    /// no FMA, no SIMD. Deliberately NOT shared code with the module under
    /// test, and deliberately not a dependency on `sigker` (ndarray must not
    /// depend on lance-graph).
    fn scalar_reference(path: &[Vec<f64>], matrices: &[f64], biases: &[f64], k: usize) -> Vec<f64> {
        if path.is_empty() {
            return vec![0.0; k];
        }
        let d = path[0].len();
        let mut z = vec![0.0f64; k];
        for window in path.windows(2) {
            let delta_x: Vec<f64> = window[1]
                .iter()
                .zip(window[0].iter())
                .map(|(a, b)| a - b)
                .collect();
            let mut z_next = z.clone();
            let mut activated = vec![0.0f64; k];
            for (i, &dx_i) in delta_x.iter().enumerate().take(d) {
                if dx_i.abs() < 1e-15 {
                    continue;
                }
                let a_offset = i * k * k;
                let b_offset = i * k;
                for row in 0..k {
                    let mut sum = biases[b_offset + row];
                    let row_off = a_offset + row * k;
                    for col in 0..k {
                        sum += matrices[row_off + col] * z[col];
                    }
                    activated[row] = sum.tanh();
                }
                for row in 0..k {
                    z_next[row] += activated[row] * dx_i;
                }
            }
            z = z_next;
        }
        z
    }

    /// SplitMix64 — the same generator `sigker` uses, so the corpus below is
    /// the corpus the consumer would actually feed the primitive.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.uniform().max(1e-300);
            let u2 = self.uniform();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    /// Gaussian projections scaled `1/sqrt(k)` — the builder's own recipe.
    fn projections(d: usize, k: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let scale = (k as f64).recip().sqrt();
        let mut rng = SplitMix64(seed);
        let matrices = (0..d * k * k).map(|_| rng.normal() * scale).collect();
        let biases = (0..d * k).map(|_| rng.normal() * scale).collect();
        (matrices, biases)
    }

    fn wiggly_path(t: usize, d: usize, seed: f64) -> Vec<Vec<f64>> {
        (0..=t)
            .map(|i| {
                let s = i as f64 / t.max(1) as f64;
                (0..d)
                    .map(|a| {
                        let phase = seed + a as f64 * 1.7;
                        s * (a as f64 + 1.0) + 0.05 * (37.0 * s + phase).cos()
                    })
                    .collect()
            })
            .collect()
    }

    fn assert_matches_reference(path: &[Vec<f64>], d: usize, k: usize, seed: u64) {
        let (matrices, biases) = projections(d, k, seed);
        let expected = scalar_reference(path, &matrices, &biases, k);
        let actual = randomized_signature_sweep(path, &matrices, &biases, k);
        assert_eq!(actual.len(), k, "state width must be state_dim");
        for (row, (e, a)) in expected.iter().zip(actual.iter()).enumerate() {
            let tol = 1e-9 * e.abs().max(1.0);
            assert!(
                (e - a).abs() <= tol,
                "randomized_signature_sweep mismatch at row {row}: reference={e:e} \
                 actual={a:e} (T={}, d={d}, k={k}, seed={seed})",
                path.len()
            );
        }
    }

    #[test]
    fn parity_across_state_widths() {
        // Widths straddling the lane boundary in both directions.
        for &k in &[1usize, 7, 8, 9, 16, 33, 64] {
            assert_matches_reference(&wiggly_path(12, 3, 0.4), 3, k, 0xDEAD_BEEF);
        }
    }

    #[test]
    fn parity_across_path_dimensions() {
        for &d in &[1usize, 2, 5, 9] {
            assert_matches_reference(&wiggly_path(9, d, 1.3), d, 24, 0x51ED);
        }
    }

    #[test]
    fn parity_over_seeded_corpus() {
        // 60 distinct (seed, T, d, k) draws — the contract's "hand-roll 50+
        // inputs" corpus, every one of them a fresh Gaussian projection.
        let mut rng = SplitMix64(0xC0FF_EE00);
        for case in 0..60u64 {
            let t = 2 + (rng.next_u64() % 9) as usize;
            let d = 1 + (rng.next_u64() % 4) as usize;
            let k = 1 + (rng.next_u64() % 40) as usize;
            assert_matches_reference(&wiggly_path(t, d, case as f64 * 0.31), d, k, case);
        }
    }

    #[test]
    fn parity_with_long_path_and_wide_state() {
        // Past one lane in every dimension, none of them a multiple of 8.
        assert_matches_reference(&wiggly_path(37, 3, 0.9), 3, 67, 7);
    }

    #[test]
    fn degenerate_single_point_path_is_zero_state() {
        let (matrices, biases) = projections(3, 11, 5);
        let z = randomized_signature_sweep(&[vec![0.3, -1.2, 4.0]], &matrices, &biases, 11);
        assert_eq!(z, vec![0.0; 11]);
        let empty: [Vec<f64>; 0] = [];
        assert_eq!(randomized_signature_sweep(&empty, &matrices, &biases, 11), vec![0.0; 11]);
    }

    #[test]
    fn zero_increment_path_leaves_state_at_zero() {
        // A constant path has |dx_i| = 0 < 1e-15 on every coordinate, so
        // every GEMV is skipped and z never leaves z_0 = 0. This is a real
        // invariant of the recurrence (hand-derived from the skip rule, not
        // read off the code under test) and it discriminates: drop the skip
        // and the result is still 0 here, but *flip the skip into an
        // unconditional apply with a non-zero dx* and it is not — which the
        // next test covers.
        let (matrices, biases) = projections(2, 20, 11);
        let path: Vec<Vec<f64>> = (0..15).map(|_| vec![7.0, -3.0]).collect();
        assert_eq!(randomized_signature_sweep(&path, &matrices, &biases, 20), vec![0.0; 20]);
    }

    #[test]
    fn sub_epsilon_increment_is_skipped_but_supra_epsilon_is_not() {
        // The 1e-15 guard must actually gate: an increment just under it
        // leaves the state untouched, one just over it does not. Without
        // both halves the constant-path test above would pass for a
        // primitive that had no guard at all.
        let (matrices, biases) = projections(1, 12, 3);
        let below = vec![vec![0.0], vec![1e-16]];
        let above = vec![vec![0.0], vec![1e-13]];
        let z_below = randomized_signature_sweep(&below, &matrices, &biases, 12);
        let z_above = randomized_signature_sweep(&above, &matrices, &biases, 12);
        assert_eq!(z_below, vec![0.0; 12], "sub-epsilon increment must be skipped");
        assert!(z_above.iter().any(|v| *v != 0.0), "supra-epsilon increment must move the state");
    }

    #[test]
    fn custom_activation_is_the_one_applied() {
        // sigma == identity turns the first step into z_1 = b * dx exactly
        // (z_0 = 0, so A . z_0 = 0), which is checkable in closed form —
        // proving the closure is used rather than tanh being hard-wired.
        let (matrices, biases) = projections(1, 9, 17);
        let path = vec![vec![0.0], vec![2.0]];
        let z = randomized_signature_sweep_with(&path, &matrices, &biases, 9, |x| x);
        for (row, v) in z.iter().enumerate() {
            assert!((v - biases[row] * 2.0).abs() < 1e-12, "identity activation not applied");
        }
        // …and tanh gives a different answer on the same inputs.
        let z_tanh = randomized_signature_sweep(&path, &matrices, &biases, 9);
        assert!(z_tanh
            .iter()
            .zip(z.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9));
    }

    #[test]
    fn step_matches_one_sweep_iteration() {
        let (matrices, biases) = projections(2, 13, 23);
        let path = vec![vec![0.0, 0.0], vec![0.5, -1.5]];
        let swept = randomized_signature_sweep(&path, &matrices, &biases, 13);
        let stepped = randomized_signature_step(&vec![0.0; 13], &[0.5, -1.5], &matrices, &biases, f64::tanh);
        assert_eq!(swept, stepped);
    }

    #[test]
    #[should_panic(expected = "matrices must hold")]
    fn wrong_matrix_length_panics() {
        let _ = randomized_signature_sweep(&[vec![0.0], vec![1.0]], &[0.0; 3], &[0.0; 2], 2);
    }

    #[test]
    #[should_panic(expected = "biases must hold")]
    fn wrong_bias_length_panics() {
        let _ = randomized_signature_sweep(&[vec![0.0], vec![1.0]], &[0.0; 4], &[0.0; 3], 2);
    }

    /// A ragged path (a later point with a different coordinate count than
    /// `path[0]`) must panic in every build profile, not just debug —
    /// `debug_assert_eq!` disappears under `--release`, and this crate never
    /// ships debug builds. Without this the wider-point case would silently
    /// truncate to `path_dim` and return a signature for the wrong path; the
    /// narrower-point case would panic anyway on out-of-bounds indexing, but
    /// with a confusing message instead of naming the real defect.
    #[test]
    #[should_panic(expected = "ragged path")]
    fn ragged_path_panics_in_release_too() {
        let (matrices, biases) = projections(2, 4, 7);
        let ragged = [vec![0.0, 0.0], vec![1.0, 1.0, 1.0]];
        let _ = randomized_signature_sweep(&ragged, &matrices, &biases, 4);
    }
}
