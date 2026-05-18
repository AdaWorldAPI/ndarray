#![allow(missing_docs)]
//! Wasserstein transport primitives: Sinkhorn-Knopp (entropic regularisation)
//! and Hungarian algorithm (exact min-cost assignment).
//!
//! These are the mathematical building blocks consumed by the Pillar-10
//! Pflug–Pichler nested-distance probe (`hpc::pillar::pflug`).
//!
//! All functions operate on row-major flat slices.  No SIMD, no `unsafe`.

// ── Sinkhorn-Knopp ────────────────────────────────────────────────────────────

/// Sinkhorn-Knopp algorithm (entropic regularisation).
///
/// Computes the regularised optimal-transport plan between two discrete
/// probability distributions `a` (M atoms) and `b` (N atoms) with ground
/// cost matrix `cost` (M×N, row-major).
///
/// # Arguments
///
/// * `cost`      — M×N row-major cost matrix (length `m * n`).
/// * `m`, `n`   — row / column counts.
/// * `a`         — row marginals (length `m`; must sum to 1).
/// * `b`         — column marginals (length `n`; must sum to 1).
/// * `epsilon`   — entropic regularisation strength (> 0; smaller → closer to OT).
/// * `max_iters` — iteration cap.
/// * `tolerance` — convergence threshold on column marginal error (L∞).
///
/// # Returns
///
/// Transport plan `P` (M×N row-major `Vec<f32>`).
///
/// # Panics
///
/// Panics if `cost.len() != m * n`, `a.len() != m`, or `b.len() != n`.
pub fn sinkhorn_knopp_f32(
    cost: &[f32],
    m: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    epsilon: f32,
    max_iters: u32,
    tolerance: f32,
) -> Vec<f32> {
    assert_eq!(cost.len(), m * n, "cost length must equal m * n");
    assert_eq!(a.len(), m, "a length must equal m");
    assert_eq!(b.len(), n, "b length must equal n");
    assert!(epsilon > 0.0, "epsilon must be positive");

    // Gibbs kernel: K[i,j] = exp(-cost[i,j] / epsilon)
    let mut k: Vec<f32> = cost.iter().map(|&c| (-c / epsilon).exp()).collect();

    // Scaling vectors u (length m) and v (length n), initialised to 1.
    let mut u = vec![1.0_f32; m];
    let mut v = vec![1.0_f32; n];

    for _ in 0..max_iters {
        // u ← a / (K v)
        for i in 0..m {
            let kv_i: f32 = (0..n).map(|j| k[i * n + j] * v[j]).sum();
            u[i] = if kv_i.abs() > f32::EPSILON { a[i] / kv_i } else { 1.0 };
        }

        // v ← b / (Kᵀ u)
        let mut v_new = vec![0.0_f32; n];
        for i in 0..m {
            for j in 0..n {
                v_new[j] += k[i * n + j] * u[i];
            }
        }
        let mut max_err = 0.0_f32;
        for j in 0..n {
            let ktu_j = v_new[j];
            let v_j = if ktu_j.abs() > f32::EPSILON { b[j] / ktu_j } else { 1.0 };
            max_err = max_err.max((v_j - v[j]).abs());
            v[j] = v_j;
        }

        if max_err < tolerance {
            break;
        }
    }

    // P[i,j] = u[i] * K[i,j] * v[j]
    for i in 0..m {
        for j in 0..n {
            k[i * n + j] *= u[i] * v[j];
        }
    }
    k
}

// ── Hungarian algorithm ───────────────────────────────────────────────────────

/// Hungarian algorithm — exact min-cost assignment for a square M×M cost matrix.
///
/// Returns `assignment` where `assignment[i] = j` means row `i` is assigned
/// to column `j`.  Complexity O(M³).
///
/// # Arguments
///
/// * `cost` — M×M row-major cost matrix (length `m * m`).
/// * `m`    — number of rows / columns.
///
/// # Returns
///
/// `Vec<u32>` of length `m` with the column indices of the optimal assignment.
///
/// # Panics
///
/// Panics if `cost.len() != m * m`.
pub fn hungarian_f32(cost: &[f32], m: usize) -> Vec<u32> {
    assert_eq!(cost.len(), m * m, "cost length must equal m * m");

    if m == 0 {
        return Vec::new();
    }

    // Work with f64 internally for better numerical stability.
    let mut c: Vec<f64> = cost.iter().map(|&x| x as f64).collect();

    // Step 1: subtract row minima.
    for i in 0..m {
        let row_min = (0..m).map(|j| c[i * m + j]).fold(f64::INFINITY, f64::min);
        for j in 0..m { c[i * m + j] -= row_min; }
    }

    // Step 2: subtract column minima.
    for j in 0..m {
        let col_min = (0..m).map(|i| c[i * m + j]).fold(f64::INFINITY, f64::min);
        for i in 0..m { c[i * m + j] -= col_min; }
    }

    // Munkres main loop.
    let mut row_cover = vec![false; m];
    let mut col_cover = vec![false; m];
    // starred[i*m+j] = 1 → starred zero; primed[i*m+j] = 1 → primed zero.
    let mut starred = vec![false; m * m];
    let mut primed  = vec![false; m * m];

    const EPS: f64 = 1e-9;

    // Step 3: initial starring pass.
    for i in 0..m {
        for j in 0..m {
            if c[i * m + j] < EPS && !row_cover[i] && !col_cover[j] {
                starred[i * m + j] = true;
                row_cover[i] = true;
                col_cover[j] = true;
            }
        }
    }
    row_cover.iter_mut().for_each(|x| *x = false);
    col_cover.iter_mut().for_each(|x| *x = false);

    // Cover columns with starred zeros.
    let count_covered = |col_cover: &[bool]| col_cover.iter().filter(|&&x| x).count();
    for j in 0..m {
        if (0..m).any(|i| starred[i * m + j]) {
            col_cover[j] = true;
        }
    }

    // Main loop: steps 4–6.
    'outer: loop {
        if count_covered(&col_cover) == m { break; }

        // Step 4: find an uncovered zero and prime it.
        'step4: loop {
            // Find an uncovered zero.
            let mut found_row = None;
            let mut found_col = None;
            'find: for i in 0..m {
                if row_cover[i] { continue; }
                for j in 0..m {
                    if c[i * m + j] < EPS && !col_cover[j] {
                        found_row = Some(i);
                        found_col = Some(j);
                        break 'find;
                    }
                }
            }

            let (pr, pc) = match (found_row, found_col) {
                (Some(r), Some(c_)) => (r, c_),
                _ => {
                    // Step 6: no uncovered zero — add/subtract minimum.
                    let min_val = {
                        let mut mv = f64::INFINITY;
                        for i in 0..m {
                            if row_cover[i] { continue; }
                            for j in 0..m {
                                if !col_cover[j] && c[i * m + j] < mv {
                                    mv = c[i * m + j];
                                }
                            }
                        }
                        mv
                    };
                    for i in 0..m {
                        for j in 0..m {
                            if row_cover[i]  { c[i * m + j] += min_val; }
                            if !col_cover[j] { c[i * m + j] -= min_val; }
                        }
                    }
                    continue 'step4;
                }
            };

            primed[pr * m + pc] = true;

            // Is there a starred zero in row pr?
            let starred_col_in_row = (0..m).find(|&j| starred[pr * m + j]);

            match starred_col_in_row {
                Some(sc) => {
                    row_cover[pr] = true;
                    col_cover[sc] = false;
                    // continue step4
                }
                None => {
                    // Step 5: augmenting path starting at (pr, pc).
                    let mut path: Vec<(usize, usize)> = vec![(pr, pc)];
                    loop {
                        let &(_, last_c) = path.last().unwrap();
                        // Find starred zero in column last_c.
                        let star_row = (0..m).find(|&i| starred[i * m + last_c]);
                        match star_row {
                            None => break,
                            Some(sr) => {
                                path.push((sr, last_c));
                                // Find primed zero in row sr.
                                let prime_col = (0..m).find(|&j| primed[sr * m + j]).unwrap();
                                path.push((sr, prime_col));
                            }
                        }
                    }
                    // Augment: toggle starred along path.
                    for (r, c_) in &path {
                        starred[r * m + c_] = !starred[r * m + c_];
                    }
                    // Clear primes and covers.
                    primed.iter_mut().for_each(|x| *x = false);
                    row_cover.iter_mut().for_each(|x| *x = false);
                    col_cover.iter_mut().for_each(|x| *x = false);
                    // Re-cover columns with starred zeros.
                    for j in 0..m {
                        if (0..m).any(|i| starred[i * m + j]) {
                            col_cover[j] = true;
                        }
                    }
                    continue 'outer;
                }
            }
        }
    }

    // Extract assignment from starred zeros.
    let mut assignment = vec![0u32; m];
    for i in 0..m {
        for j in 0..m {
            if starred[i * m + j] {
                assignment[i] = j as u32;
                break;
            }
        }
    }
    assignment
}

// ── Wasserstein-1 distance ────────────────────────────────────────────────────

/// Wasserstein-1 distance: inner product of cost matrix and transport plan.
///
/// W₁(μ,ν) = Σᵢⱼ cost[i,j] · plan[i,j]
///
/// # Arguments
///
/// * `cost` — M×N row-major cost matrix.
/// * `plan` — M×N row-major transport plan (output of [`sinkhorn_knopp_f32`]).
/// * `m`, `n` — matrix dimensions.
///
/// # Returns
///
/// Scalar Wasserstein-1 distance estimate.
#[inline]
pub fn wasserstein_1_f32(cost: &[f32], plan: &[f32], m: usize, n: usize) -> f32 {
    assert_eq!(cost.len(), m * n);
    assert_eq!(plan.len(), m * n);
    cost.iter().zip(plan.iter()).map(|(&c, &p)| c * p).sum()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── sinkhorn_knopp_f32 ────────────────────────────────────────────────────

    #[test]
    fn sinkhorn_uniform_2x2() {
        // 2×2 uniform marginals, zero-cost matrix → plan should be 0.25 everywhere.
        let cost = vec![0.0_f32; 4];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let plan = sinkhorn_knopp_f32(&cost, 2, 2, &a, &b, 0.1, 200, 1e-6);
        for p in &plan {
            assert!((p - 0.25).abs() < 1e-4, "uniform plan entry {p} not near 0.25");
        }
    }

    #[test]
    fn sinkhorn_marginals_preserved() {
        // Row sums must equal a; column sums must equal b (up to tolerance).
        let cost = vec![0.0_f32, 1.0, 1.0, 0.0]; // identity-type cost
        let a = vec![0.3_f32, 0.7];
        let b = vec![0.4_f32, 0.6];
        let plan = sinkhorn_knopp_f32(&cost, 2, 2, &a, &b, 0.05, 500, 1e-6);

        let row0: f32 = plan[0] + plan[1];
        let row1: f32 = plan[2] + plan[3];
        let col0: f32 = plan[0] + plan[2];
        let col1: f32 = plan[1] + plan[3];

        assert!((row0 - a[0]).abs() < 1e-4, "row0={row0} != a[0]={}", a[0]);
        assert!((row1 - a[1]).abs() < 1e-4, "row1={row1} != a[1]={}", a[1]);
        assert!((col0 - b[0]).abs() < 1e-4, "col0={col0} != b[0]={}", b[0]);
        assert!((col1 - b[1]).abs() < 1e-4, "col1={col1} != b[1]={}", b[1]);
    }

    #[test]
    fn sinkhorn_nonnegative_plan() {
        let cost = vec![1.0_f32, 2.0, 3.0, 4.0];
        let a = vec![0.5_f32, 0.5];
        let b = vec![0.5_f32, 0.5];
        let plan = sinkhorn_knopp_f32(&cost, 2, 2, &a, &b, 0.1, 200, 1e-6);
        for p in &plan {
            assert!(*p >= 0.0, "plan entry {p} is negative");
        }
    }

    // ── hungarian_f32 ─────────────────────────────────────────────────────────

    #[test]
    fn hungarian_identity_2x2() {
        // Diagonal cost 0, off-diagonal cost 1 → assign i→i.
        let cost = vec![0.0_f32, 1.0, 1.0, 0.0];
        let asgn = hungarian_f32(&cost, 2);
        assert_eq!(asgn[0], 0);
        assert_eq!(asgn[1], 1);
    }

    #[test]
    fn hungarian_swap_2x2() {
        // Off-diagonal cheaper.
        let cost = vec![1.0_f32, 0.0, 0.0, 1.0];
        let asgn = hungarian_f32(&cost, 2);
        assert_eq!(asgn[0], 1);
        assert_eq!(asgn[1], 0);
    }

    #[test]
    fn hungarian_3x3_known_solution() {
        // Classic 3×3 example with known optimal assignment cost = 0+5+6=11? No.
        // cost = [[9,2,7],[3,6,4],[1,8,5]]; optimal: row0→col1(2), row1→col2(4), row2→col0(1) = 7
        let cost = vec![9.0_f32, 2.0, 7.0, 3.0, 6.0, 4.0, 1.0, 8.0, 5.0];
        let asgn = hungarian_f32(&cost, 3);
        // Verify feasibility: each column assigned exactly once.
        let mut cols = vec![false; 3];
        for &j in &asgn { cols[j as usize] = true; }
        assert!(cols.iter().all(|&x| x), "not a permutation: {asgn:?}");
        // Verify optimality by checking total cost equals 7.
        let total: f32 = (0..3).map(|i| cost[i * 3 + asgn[i] as usize]).sum();
        assert!((total - 7.0).abs() < 1e-3, "total cost {total} != 7.0");
    }

    #[test]
    fn hungarian_empty() {
        let asgn = hungarian_f32(&[], 0);
        assert!(asgn.is_empty());
    }

    // ── wasserstein_1_f32 ─────────────────────────────────────────────────────

    #[test]
    fn w1_zero_cost() {
        let cost = vec![0.0_f32; 4];
        let plan = vec![0.25_f32; 4];
        assert_eq!(wasserstein_1_f32(&cost, &plan, 2, 2), 0.0);
    }

    #[test]
    fn w1_identity_plan() {
        // plan puts all mass on diagonal; cost[0,0]=0, cost[1,1]=0.
        let cost = vec![0.0_f32, 1.0, 1.0, 0.0];
        let plan = vec![0.5_f32, 0.0, 0.0, 0.5];
        assert!((wasserstein_1_f32(&cost, &plan, 2, 2) - 0.0).abs() < 1e-9);
    }
}
