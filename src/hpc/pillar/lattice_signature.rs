//! Pillar-11 (lattice lane) — the BIT-EXACT integer signature of a lattice
//! walk, and the finite-depth Hambly–Lyons certificate it makes executable.
//!
//! # What this lane adds beside `signature.rs`
//!
//! [`super::signature`] is the f32 depth-3 kernel-stability battery over
//! Brownian paths. It certifies PSD-ness and concentration of a truncated
//! signature kernel, NOT the Hambly–Lyons uniqueness theorem — that
//! disambiguation is in its own header. This lane is the other half, and it
//! is exact: for a walk of **unit basis-aligned steps on the integer lattice**
//! every signature coefficient at level `k` is a rational with denominator
//! `k!`, so `k!·S_k` is an integer. Storing that integer makes the whole
//! computation bit-exact — no f32, no f64, no tolerance, no rounding: two
//! walks have the same truncated signature iff the two `i128` tensors are
//! `==`.
//!
//! # Why that is the certificate (Hambly–Lyons, Annals 171 (2010) §2.4)
//!
//! **Theorem 5** (Annals numbering; Theorem 1 of the introduction restates
//! it). A path of length `L` on the 2-d integer lattice whose first
//! `⌊2e·log(1+√2)·L⌋` GL(2,C)-iterated integrals vanish is tree-like and its
//! reduced word is trivial. (`2e·ln(1+√2) = 4.7916…`)
//!
//! **Theorem 6.** In the `d`-dimensional lattice the depth is
//! `⌊(2⌈log₃(d/2)⌉ + 3)·2e·log(1+√2)·L⌋`.
//!
//! ⚠ Version trap: arXiv math/0507536v2 states these as Theorems 2/3 with
//! coefficient `e`; its proof indexes the odd-degree sum by pairs, so the
//! published text corrected the coefficient to `2e` (proof takes
//! `x = 2·log(1+√2)·L`). This lane implements the published form.
//!
//! The GL(2,C) integrals are a projection of the tensor-algebra ones (the
//! paper's fn. 2: "a priori contain less information"), so vanishing of the
//! FULL truncated signature to that depth implies the hypothesis a fortiori.
//! Because the truncated signature is a homomorphism into the free nilpotent
//! group, the two-path form is
//!
//! ```text
//! S^(N)(X) = S^(N)(Y)  ⟺  X ∼ Y     for  N ≥ ⌊c(d)·(|X|+|Y|)⌋,
//! ```
//!
//! i.e. the Index regime is LENGTH-PARAMETERIZED. Depth 2 is a necessary
//! condition only: the paper's own §1.6 figure-of-8 has `S¹ = S² = 0` and is
//! not tree-like. This lane finds every such length-8 reduced word and
//! separates each one at level 3 — exactly, in integers.
//!
//! # Depth policy — integer, never float
//!
//! The paper's constant is transcendental. This lane never evaluates it in
//! floating point: [`theorem2_depth`] returns `⌈47917·L / 10000⌉`, an
//! integer upper bound on `c·L` (`47917/10000 = 4.7917 > 4.79164…`), hence
//! `≥ ⌊c·L⌋` and always sufficient. The general-`d` factor
//! `2⌈log₃(d/2)⌉ + 3` is computed with integer arithmetic too.
//!
//! # Preconditions the certificate needs (both pinned below)
//!
//! * `d ≥ 2`. In `d = 1` the reduced-path group is `Z`: the signature is
//!   `(1, Δ, Δ²/2!, …)` and carries only the net increment (every closed
//!   1-d path is tree-like). A single `u8:u8` rail read as ONE scalar axis
//!   is `d = 1` and is out of regime — see `d1_carries_only_the_net_increment`.
//! * Unit basis-aligned steps (`‖x_k − x_{k+1}‖ = 1`, `x_k ∈ Z^d`). Arbitrary
//!   quantized step vectors are outside Theorem 5; for those the applicable
//!   statement is Theorem 9 (non-triviality, no explicit depth).
//!
//! # Overflow contract
//!
//! `|k!·S_k[w]| ≤ L^k` for a walk of length `L`, so the tensors fit `i128`
//! while `L^depth < 2^126`; [`fits_i128`] is the guard and every arithmetic
//! step is `checked_*` — an overflow is a panic with the offending level,
//! never a silently wrapped coefficient.
//!
//! # SIMD
//!
//! Scalar integer arithmetic, deliberately: this is the reference lane. A
//! vectorised lane is the W1.5 `sigker` item in
//! `lance-graph/.claude/knowledge/ndarray-vertical-simd-alien-magic.md`,
//! now unblocked by the certificate; it must reproduce these `i128` tensors
//! bit-for-bit (the parity test shape is `crates/sigker-parity`).
//!
//! Cross-repo twin: `lance-graph/crates/jc/src/hambly_lyons.rs` W6 leg (f64,
//! against `sigker::signature_truncated`, tolerance `1e-12`). Same word
//! classes, same counts; this lane replaces its tolerance with equality.

use alloc::vec;
use alloc::vec::Vec;

use super::prove_runner::{PillarReport, SplitMix64};

/// Deterministic seed for the lattice lane (tree-like word generation).
pub const PILLAR_11_LATTICE_SEED: u64 = 0x_0516_DC5A_DD11;

/// Numerator / denominator of the rational upper bound on `2e·ln(1+√2)`
/// used by [`theorem2_depth`]: `4.7917 > 4.79164…`.
pub const THEOREM2_C_NUM: usize = 47_917;
/// See [`THEOREM2_C_NUM`].
pub const THEOREM2_C_DEN: usize = 10_000;

/// One unit lattice step: axis `0..d`, sign `+1` or `−1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Step {
    /// Axis index in `0..d`.
    pub axis: u8,
    /// `+1` or `−1`.
    pub sign: i8,
}

impl Step {
    /// The inverse step (same axis, opposite sign).
    #[must_use]
    pub const fn inverse(self) -> Self {
        Step {
            axis: self.axis,
            sign: -self.sign,
        }
    }
}

/// The truncated signature of a lattice walk, every level scaled by `k!` so
/// each coefficient is an exact integer. `levels[k]` has `d^k` entries in
/// row-major multi-index order (first letter most significant) — the same
/// layout as [`super::signature::signature_d2_deg3`] and `sigker`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeSignature {
    /// Dimension `d` of the lattice.
    pub dim: usize,
    /// Truncation depth `N`.
    pub depth: usize,
    /// `levels[k][w] = k! · S_k[w]`, exact.
    pub levels: Vec<Vec<i128>>,
}

impl LatticeSignature {
    /// The signature of the constant path: `(1, 0, 0, …)`.
    #[must_use]
    pub fn identity(dim: usize, depth: usize) -> Self {
        let mut levels = Vec::with_capacity(depth + 1);
        levels.push(vec![1]);
        let mut len = 1usize;
        for _ in 1..=depth {
            len *= dim;
            levels.push(vec![0; len]);
        }
        Self { dim, depth, levels }
    }

    /// Exact identity test — integer equality, no tolerance.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.levels[0][0] == 1 && self.levels[1..].iter().all(|l| l.iter().all(|&c| c == 0))
    }

    /// The first level `k ≥ 1` with a non-zero coefficient, if any. `None`
    /// means the signature is the identity to this depth.
    #[must_use]
    pub fn first_nonzero_level(&self) -> Option<usize> {
        (1..=self.depth).find(|&k| self.levels[k].iter().any(|&c| c != 0))
    }

    /// Fold the whole tensor into a 64-bit FNV-1a digest (bit-exactness pin).
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |b: u8| {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        };
        for l in &self.levels {
            for &c in l {
                for b in c.to_le_bytes() {
                    mix(b);
                }
            }
        }
        h
    }
}

/// Does a walk of `len` unit steps fit `i128` at `depth`? Bound: `len^depth`
/// must stay below `2^126` (one bit of headroom under the sign).
#[must_use]
pub fn fits_i128(len: usize, depth: usize) -> bool {
    let mut acc: u128 = 1;
    let base = len.max(1) as u128;
    for _ in 0..depth {
        match acc.checked_mul(base) {
            Some(v) if v < (1u128 << 126) => acc = v,
            _ => return false,
        }
    }
    true
}

/// Theorem 5 depth for a `d = 2` walk of length `len`: the integer upper
/// bound `⌈47917·len / 10000⌉ ≥ ⌊2e·ln(1+√2)·len⌋`. No floating point.
#[must_use]
pub const fn theorem2_depth(len: usize) -> usize {
    (THEOREM2_C_NUM * len + THEOREM2_C_DEN - 1) / THEOREM2_C_DEN
}

/// Theorem 6 factor `2⌈log₃(d/2)⌉ + 3`, integer arithmetic. For `d ≤ 2` the
/// factor is `3` (callers with `d = 2` should use [`theorem2_depth`], which
/// is Theorem 5 directly).
#[must_use]
pub const fn theorem3_factor(dim: usize) -> usize {
    // ⌈log₃(d/2)⌉ = smallest m with 3^m ≥ ⌈d/2⌉.
    let target = dim.div_ceil(2);
    let mut m = 0usize;
    let mut p = 1usize;
    while p < target {
        p *= 3;
        m += 1;
    }
    2 * m + 3
}

/// Theorem 6 depth for a `d`-dimensional lattice walk of length `len`.
#[must_use]
pub const fn theorem3_depth(dim: usize, len: usize) -> usize {
    theorem2_depth(theorem3_factor(dim) * len)
}

/// Binomial coefficients `C(k, j)` for `k ≤ depth`, exact.
fn binomials(depth: usize) -> Vec<Vec<i128>> {
    let mut c = vec![vec![0i128; depth + 1]; depth + 1];
    for k in 0..=depth {
        c[k][0] = 1;
        for j in 1..=k {
            c[k][j] = c[k - 1][j - 1] + if j <= k - 1 { c[k - 1][j] } else { 0 };
        }
    }
    c
}

/// The bit-exact truncated signature of a lattice walk.
///
/// Chen's identity with a unit step `σ·e_a` is a binomial convolution against
/// a signature that is non-zero only on the index `(a, a, …, a)`:
///
/// ```text
/// (T · E)_k[w] = Σ_{j=0}^{r(w)} C(k, j) · T_{k−j}[w[0..k−j]] · σ^j
/// ```
///
/// where `r(w)` is the length of `w`'s trailing run of the letter `a`.
///
/// # Panics
///
/// * if a step's axis is `≥ dim`, or its sign is not `±1`;
/// * on `i128` overflow (see [`fits_i128`]).
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::pillar::lattice_signature::{lattice_signature, Step};
/// let a = Step { axis: 0, sign: 1 };
/// // out-and-back is tree-like: exactly the identity, at every depth
/// let s = lattice_signature(&[a, a.inverse()], 2, 6);
/// assert!(s.is_identity());
/// // a single unit step in x: k!·S_k[(x,…,x)] = 1 for every k
/// let s = lattice_signature(&[a], 2, 3);
/// assert_eq!(s.levels[1], vec![1, 0]);
/// assert_eq!(s.levels[2], vec![1, 0, 0, 0]);
/// assert_eq!(s.levels[3][0], 1);
/// ```
#[must_use]
pub fn lattice_signature(word: &[Step], dim: usize, depth: usize) -> LatticeSignature {
    assert!(dim >= 1, "lattice_signature: dim must be ≥ 1");
    assert!(
        fits_i128(word.len(), depth),
        "lattice_signature: len^depth exceeds the i128 contract (len={}, depth={depth})",
        word.len()
    );
    let binom = binomials(depth);
    let mut sig = LatticeSignature::identity(dim, depth);
    // d^j lookup for prefix arithmetic
    let mut pow = vec![1usize; depth + 1];
    for j in 1..=depth {
        pow[j] = pow[j - 1] * dim;
    }
    for step in word {
        let a = usize::from(step.axis);
        assert!(a < dim, "lattice_signature: axis {a} out of range for dim {dim}");
        assert!(step.sign == 1 || step.sign == -1, "lattice_signature: sign must be ±1");
        let sigma = i128::from(step.sign);
        let mut next = LatticeSignature::identity(dim, depth);
        for k in 1..=depth {
            let lvl = &mut next.levels[k];
            for (w, out) in lvl.iter_mut().enumerate() {
                // trailing run of letter `a` in the k-letter word w
                let mut r = 0usize;
                let mut ww = w;
                while r < k && ww % dim == a {
                    r += 1;
                    ww /= dim;
                }
                let mut acc: i128 = 0;
                let mut sigma_pow: i128 = 1;
                for j in 0..=r {
                    let prefix = w / pow[j];
                    let t = sig.levels[k - j][prefix];
                    if t != 0 {
                        let term = binom[k][j]
                            .checked_mul(t)
                            .and_then(|v| v.checked_mul(sigma_pow))
                            .expect("lattice_signature: i128 overflow (term)");
                        acc = acc
                            .checked_add(term)
                            .expect("lattice_signature: i128 overflow (sum)");
                    }
                    sigma_pow *= sigma;
                }
                *out = acc;
            }
        }
        sig = next;
    }
    sig
}

/// Letters of the free group on `dim` generators: `0..dim` are `+e_a`,
/// `dim..2·dim` are `−e_a`.
#[must_use]
pub fn letter(dim: usize, l: usize) -> Step {
    if l < dim {
        Step { axis: l as u8, sign: 1 }
    } else {
        Step {
            axis: (l - dim) as u8,
            sign: -1,
        }
    }
}

/// Is the word freely reduced (no adjacent `x x⁻¹`)?
#[must_use]
pub fn is_reduced(word: &[Step]) -> bool {
    word.windows(2).all(|w| w[0] != w[1].inverse())
}

/// Every word of exactly `len` letters over the `2·dim`-letter alphabet, in
/// lexicographic order.
pub fn for_each_word(dim: usize, len: usize, mut f: impl FnMut(&[Step])) {
    let alphabet = 2 * dim;
    let total = alphabet.pow(len as u32);
    let mut w = vec![Step { axis: 0, sign: 1 }; len];
    for code in 0..total {
        let mut c = code;
        for slot in w.iter_mut() {
            *slot = letter(dim, c % alphabet);
            c /= alphabet;
        }
        f(&w);
    }
}

/// A tree-like word of length `len` (even): grow from empty by inserting
/// `c c⁻¹` at random positions — the generator of tree-like equivalence
/// (Hambly–Lyons Def. 2.1).
pub fn treelike_word(rng: &mut SplitMix64, dim: usize, len: usize) -> Vec<Step> {
    let mut w: Vec<Step> = Vec::with_capacity(len);
    while w.len() + 2 <= len {
        let c = letter(dim, (rng.next_u64() % (2 * dim as u64)) as usize);
        let pos = (rng.next_u64() as usize) % (w.len() + 1);
        w.insert(pos, c);
        w.insert(pos + 1, c.inverse());
    }
    w
}

/// Longest word the exhaustive theorem arm enumerates (`52` reduced words
/// in `d = 2`; depth `theorem2_depth(3) = 15`).
pub const LATTICE_L_MAX: usize = 3;
/// Length of the depth-2 false-merge search (the figure-of-8 class lives here).
pub const LATTICE_FALSE_MERGE_L: usize = 8;
/// Tree-like words drawn per run.
pub const LATTICE_N_TREELIKE: u32 = 64;
/// Depth for the tree-like arm (identity holds at every depth; fixed, cheap).
pub const LATTICE_TREELIKE_DEPTH: usize = 12;

/// The measurements of one lattice-lane run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LatticeLeg {
    /// reduced non-empty `d = 2` words of length `≤ LATTICE_L_MAX`
    pub reduced_checked: u32,
    /// how many of them the theorem depth failed to separate (must be 0)
    pub reduced_merged: u32,
    /// tree-like words checked; all must be the exact identity
    pub treelike_checked: u32,
    /// of those, how many were NOT the exact identity (must be 0)
    pub treelike_not_identity: u32,
    /// reduced words of length `LATTICE_FALSE_MERGE_L` with `S^(2) = 1`
    pub depth2_false_merges: u32,
    /// the deepest level any of those needed to separate (`≤` theorem depth)
    pub false_merge_max_sep_level: u32,
    /// how many stayed merged at the theorem depth (must be 0)
    pub false_merge_unresolved: u32,
    /// `d = 1`: distinct signatures among all `2^6` words of length 6 —
    /// must be exactly 7 (net increment `−6..6` in steps of 2)
    pub d1_classes: u32,
}

/// Run the lattice lane.
#[must_use]
pub fn lattice_leg() -> LatticeLeg {
    let dim = 2;

    // Arm 1 — Theorem 5: reduced ⟹ separated at the theorem depth.
    let mut reduced_checked = 0u32;
    let mut reduced_merged = 0u32;
    for len in 1..=LATTICE_L_MAX {
        let depth = theorem2_depth(len);
        for_each_word(dim, len, |w| {
            if !is_reduced(w) {
                return;
            }
            reduced_checked += 1;
            if lattice_signature(w, dim, depth).is_identity() {
                reduced_merged += 1;
            }
        });
    }

    // Arm 2 — tree-like words are EXACTLY the identity.
    let mut rng = SplitMix64::new(PILLAR_11_LATTICE_SEED);
    let mut treelike_checked = 0u32;
    let mut treelike_not_identity = 0u32;
    // Tree-like words are the identity at EVERY depth (Hambly–Lyons Cor. 6.4),
    // so this arm needs no theorem depth — a fixed one keeps the length-6
    // words off the 2^29-coefficient tensors the theorem depth would demand.
    for i in 0..LATTICE_N_TREELIKE {
        let len = 2 + 2 * (i as usize % 3);
        let w = treelike_word(&mut rng, dim, len);
        treelike_checked += 1;
        if !lattice_signature(&w, dim, LATTICE_TREELIKE_DEPTH).is_identity() {
            treelike_not_identity += 1;
        }
    }

    // Arm 3 — depth 2 is NOT the Index regime: reduced length-8 words with
    // S^(2) = 1, each separated by the theorem depth; record the level.
    let theorem_depth = theorem2_depth(LATTICE_FALSE_MERGE_L);
    let mut depth2_false_merges = 0u32;
    let mut false_merge_max_sep_level = 0u32;
    let mut false_merge_unresolved = 0u32;
    for_each_word(dim, LATTICE_FALSE_MERGE_L, |w| {
        if !is_reduced(w) || !lattice_signature(w, dim, 2).is_identity() {
            return;
        }
        depth2_false_merges += 1;
        // Escalate depth one level at a time: `first_nonzero_level` at depth
        // `n` is exact for every level `≤ n`, so the first depth that reports
        // `Some(k)` gives `k` exactly and never over-computes the tensor.
        let mut sep = None;
        for depth in 3..=theorem_depth {
            if let Some(k) = lattice_signature(w, dim, depth).first_nonzero_level() {
                sep = Some(k);
                break;
            }
        }
        match sep {
            Some(k) => false_merge_max_sep_level = false_merge_max_sep_level.max(k as u32),
            None => false_merge_unresolved += 1,
        }
    });

    // Arm 4 — the d = 1 fence.
    let mut classes: Vec<LatticeSignature> = Vec::new();
    for_each_word(1, 6, |w| {
        let s = lattice_signature(w, 1, 3);
        if !classes.contains(&s) {
            classes.push(s);
        }
    });

    LatticeLeg {
        reduced_checked,
        reduced_merged,
        treelike_checked,
        treelike_not_identity,
        depth2_false_merges,
        false_merge_max_sep_level,
        false_merge_unresolved,
        d1_classes: classes.len() as u32,
    }
}

/// Pillar-11 lattice lane as a [`PillarReport`]: `psd_rate` = fraction of
/// reduced words separated at the theorem depth (must be `1.0`),
/// `lognorm_concentration` = the deepest separation level the depth-2 false
/// merges needed (informational; `≤ theorem2_depth(8)`), `n_paths` =
/// reduced words checked, `n_hops` = depth-2 false merges found.
#[must_use]
pub fn prove_pillar_11_lattice() -> PillarReport {
    let leg = lattice_leg();
    let passed = leg.reduced_merged == 0
        && leg.treelike_not_identity == 0
        && leg.depth2_false_merges >= 1
        && leg.false_merge_unresolved == 0
        && leg.false_merge_max_sep_level as usize <= theorem2_depth(LATTICE_FALSE_MERGE_L)
        && leg.d1_classes == 7;
    PillarReport {
        pillar_id: 11,
        seed: PILLAR_11_LATTICE_SEED,
        n_paths: leg.reduced_checked,
        n_hops: leg.depth2_false_merges,
        psd_rate: 1.0 - f64::from(leg.reduced_merged) / f64::from(leg.reduced_checked.max(1)),
        lognorm_concentration: f64::from(leg.false_merge_max_sep_level),
        passed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::pillar::signature::{signature_d2_deg3, SIG_D2_DEG3_LEN};

    const A: Step = Step { axis: 0, sign: 1 };
    const B: Step = Step { axis: 1, sign: 1 };

    fn inv(s: Step) -> Step {
        s.inverse()
    }

    #[test]
    fn depth_policy_dominates_the_paper_floor_without_floats() {
        // ⌊4.7916…·L⌋ for L = 1..16, pinned from the published constant.
        let floors = [4usize, 9, 14, 19, 23, 28, 33, 38, 43, 47, 52, 57, 62, 67, 71, 76];
        for (i, &f) in floors.iter().enumerate() {
            let l = i + 1;
            assert!(theorem2_depth(l) >= f, "L={l}: {} < ⌊cL⌋={f}", theorem2_depth(l));
            assert!(theorem2_depth(l) <= f + 1, "L={l}: bound is loose by more than 1");
        }
        assert_eq!(theorem3_factor(2), 3);
        assert_eq!(theorem3_factor(3), 5); // ⌈log₃ 2⌉ = 1
        assert_eq!(theorem3_factor(12), 7); // ⌈log₃ 6⌉ = 2
        assert_eq!(theorem3_depth(3, 4), theorem2_depth(20));
    }

    #[test]
    fn out_and_back_is_exactly_the_identity_at_every_depth() {
        for depth in 1..=12 {
            assert!(lattice_signature(&[A, inv(A)], 2, depth).is_identity());
            assert!(lattice_signature(&[B, A, inv(A), inv(B)], 2, depth).is_identity());
        }
    }

    #[test]
    fn the_figure_of_eight_is_invisible_at_depth_2_and_separated_at_level_3() {
        // a b a⁻¹ b⁻¹ · b⁻¹ a⁻¹ b a — two unit squares of opposite orientation
        let w = [A, B, inv(A), inv(B), inv(B), inv(A), B, A];
        assert!(is_reduced(&w));
        assert!(lattice_signature(&w, 2, 2).is_identity());
        // Escalate like the leg does: depth 3 already separates it, and the
        // theorem depth ⌊c·8⌋ = 38 would materialize a 2^39-entry tensor.
        let s = lattice_signature(&w, 2, 3);
        assert_eq!(s.first_nonzero_level(), Some(3));
        assert!(3 <= theorem2_depth(8));
        // 3!·S_{xxy} = 6 — the f64 probe measured S¹¹² = 1.0 exactly
        assert_eq!(s.levels[3][0b001], 6);
    }

    #[test]
    fn commutator_has_unit_levy_area() {
        // a b a⁻¹ b⁻¹ encloses one unit square counter-clockwise, so the
        // Lévy area ½(S_xy − S_yx) = 1, i.e. S_xy = 1, S_yx = −1 and the
        // stored 2!·S is (2, −2). (A right triangle of area ½ would store
        // (1, −1) — the f64 probe's [0, 0.5, −0.5, 0] was that case.)
        let s = lattice_signature(&[A, B, inv(A), inv(B)], 2, 2);
        assert_eq!(s.levels[1], vec![0, 0]);
        assert_eq!(s.levels[2], vec![0, 2, -2, 0]);
    }

    #[test]
    fn parity_with_the_f32_depth3_lane_on_lattice_words() {
        // k!·S_k / k! must reproduce `signature_d2_deg3` to f32 exactness on
        // small integers — the internal parity pin between the two lanes.
        let fact = [1.0f32, 1.0, 2.0, 6.0];
        for len in 1..=6 {
            for_each_word(2, len, |w| {
                let mut path = vec![0.0f32, 0.0];
                let (mut x, mut y) = (0.0f32, 0.0f32);
                for s in w {
                    if s.axis == 0 {
                        x += f32::from(s.sign);
                    } else {
                        y += f32::from(s.sign);
                    }
                    path.push(x);
                    path.push(y);
                }
                let f = signature_d2_deg3(&path, w.len() + 1);
                let z = lattice_signature(w, 2, 3);
                let mut i = 0usize;
                for (k, lvl) in z.levels.iter().enumerate() {
                    for &c in lvl {
                        let expect = c as f32 / fact[k];
                        assert!(
                            (f[i] - expect).abs() <= 1e-5 * (1.0 + expect.abs()),
                            "word {w:?} idx {i}: f32 {} vs exact {expect}",
                            f[i]
                        );
                        i += 1;
                    }
                }
                assert_eq!(i, SIG_D2_DEG3_LEN);
            });
        }
    }

    #[test]
    fn theorem2_separates_every_reduced_word_and_collapses_every_treelike_one() {
        let leg = lattice_leg();
        assert_eq!(leg.reduced_checked, 52);
        assert_eq!(leg.reduced_merged, 0);
        assert_eq!(leg.treelike_checked, 64);
        assert_eq!(leg.treelike_not_identity, 0);
    }

    #[test]
    fn depth_2_is_a_necessary_condition_only() {
        let leg = lattice_leg();
        // the figure-of-8 class: 64 reduced length-8 words invisible at depth 2
        assert_eq!(leg.depth2_false_merges, 64);
        assert_eq!(leg.false_merge_unresolved, 0);
        assert_eq!(leg.false_merge_max_sep_level, 3);
        assert!((leg.false_merge_max_sep_level as usize) <= theorem2_depth(LATTICE_FALSE_MERGE_L));
    }

    #[test]
    fn d1_carries_only_the_net_increment() {
        let leg = lattice_leg();
        assert_eq!(leg.d1_classes, 7);
        // and the two 1-d walks with equal increment are literally equal
        let a = Step { axis: 0, sign: 1 };
        let s1 = lattice_signature(&[a, inv(a), a], 1, 8);
        let s2 = lattice_signature(&[a], 1, 8);
        assert_eq!(s1, s2);
    }

    #[test]
    fn overflow_contract_is_checked_not_wrapped() {
        assert!(fits_i128(16, 31));
        assert!(!fits_i128(16, 32));
        assert!(fits_i128(8, 40));
        assert!(!fits_i128(1_000_000, 7));
    }

    #[test]
    fn prove_passes_and_is_deterministic() {
        let r1 = prove_pillar_11_lattice();
        let r2 = prove_pillar_11_lattice();
        assert!(r1.passed, "{r1:?}");
        assert_eq!(r1.psd_rate, r2.psd_rate);
        assert_eq!(r1.n_paths, 52);
        assert_eq!(r1.n_hops, 64);
        assert_eq!(r1.seed, PILLAR_11_LATTICE_SEED);
    }

    /// Bit-exactness pin: the digest of a fixed word set is a constant. There
    /// is no tolerance to hide behind — a changed coefficient changes the hash.
    #[test]
    fn lattice_lane_is_bit_exact() {
        let mut h: u64 = 0;
        for len in 1..=3 {
            for_each_word(2, len, |w| {
                h ^= lattice_signature(w, 2, theorem2_depth(len))
                    .digest()
                    .rotate_left(len as u32);
            });
        }
        assert_eq!(h, LATTICE_DIGEST_PIN, "digest drifted: 0x{h:016X}");
    }

    /// Pinned from the first run of `lattice_lane_is_bit_exact` (words of length ≤ 3 at the theorem depth).
    const LATTICE_DIGEST_PIN: u64 = 0x7C9612A734212FC6;
}
