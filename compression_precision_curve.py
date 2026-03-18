#!/usr/bin/env python3
"""
Compression Precision Curve: Zeckendorf CLZ vs competing methods.
Measures Spearman rank correlation with ground truth across bit depths.
"""
import numpy as np
from scipy import stats
import json
import sys
np.random.seed(42)
N_NODES = 500    # manageable for O(n²) pairwise
N_QUERIES = 30
D_PLANE = 16384  # bits per plane (256 × u64)
D_TOTAL = D_PLANE * 3  # 49152 bits for full SPO triple
print("Generating nodes...", file=sys.stderr)
# Generate random binary SPO triples
nodes_s = np.random.randint(0, 2, size=(N_NODES, D_PLANE), dtype=np.uint8)
nodes_p = np.random.randint(0, 2, size=(N_NODES, D_PLANE), dtype=np.uint8)
nodes_o = np.random.randint(0, 2, size=(N_NODES, D_PLANE), dtype=np.uint8)
# Also structured: cluster 0-49 share similar S, 50-99 share similar P
base_s = np.random.randint(0, 2, size=D_PLANE, dtype=np.uint8)
base_p = np.random.randint(0, 2, size=D_PLANE, dtype=np.uint8)
struct_s = nodes_s.copy()
struct_p = nodes_p.copy()
struct_o = nodes_o.copy()
for i in range(50):
    flip_mask = np.zeros(D_PLANE, dtype=np.uint8)
    flip_positions = np.random.choice(D_PLANE, size=i * 80, replace=False)
    flip_mask[flip_positions] = 1
    struct_s[i] = base_s ^ flip_mask
for i in range(50, 100):
    flip_mask = np.zeros(D_PLANE, dtype=np.uint8)
    flip_positions = np.random.choice(D_PLANE, size=(i - 50) * 80, replace=False)
    flip_mask[flip_positions] = 1
    struct_p[i] = base_p ^ flip_mask
def hamming(a, b):
    return np.sum(a != b)
def hamming_batch(A, b):
    return np.sum(A != b, axis=1)
# Precompute exact pairwise distances
print("Computing exact pairwise distances...", file=sys.stderr)
exact_dists = np.zeros((N_NODES, N_NODES), dtype=np.int32)
struct_dists = np.zeros((N_NODES, N_NODES), dtype=np.int32)
for i in range(N_NODES):
    exact_dists[i] = (
        np.sum(nodes_s[i] != nodes_s, axis=1) +
        np.sum(nodes_p[i] != nodes_p, axis=1) +
        np.sum(nodes_o[i] != nodes_o, axis=1)
    )
    struct_dists[i] = (
        np.sum(struct_s[i] != struct_s, axis=1) +
        np.sum(struct_p[i] != struct_p, axis=1) +
        np.sum(struct_o[i] != struct_o, axis=1)
    )
def mean_spearman(dist_matrix, ref_matrix, n_queries=N_QUERIES):
    """Average Spearman ρ across n_queries query nodes."""
    rhos = []
    for q in range(n_queries):
        mask = np.ones(N_NODES, dtype=bool)
        mask[q] = False
        rho, _ = stats.spearmanr(ref_matrix[q, mask], dist_matrix[q, mask])
        if np.isfinite(rho):
            rhos.append(rho)
    return np.mean(rhos) if rhos else 0.0
def golden_shift(d):
    PHI = (1 + np.sqrt(5)) / 2
    raw = int(d / (PHI * PHI))
    return raw + 1 if raw % 2 == 0 else raw
def xor_fold(arr, target_d):
    """Repeatedly XOR-fold until reaching target dimension."""
    current = arr.copy()
    while len(current) > target_d:
        half = len(current) // 2
        current = current[:half] ^ current[half:2*half]
    return current
def cyclic_shift_1d(arr, shift):
    """Cyclic shift a 1D binary array."""
    return np.roll(arr, -shift)
def majority_vote_3(a, b, c):
    """Bit-wise majority vote of 3 binary arrays."""
    return ((a.astype(int) + b.astype(int) + c.astype(int)) >= 2).astype(np.uint8)
def bundle_at_dim(s, p, o, d_bits):
    """Cyclic-permutation bundle at target dimension."""
    s_f = xor_fold(s, d_bits)
    p_f = xor_fold(p, d_bits)
    o_f = xor_fold(o, d_bits)
    shift = golden_shift(d_bits)
    shift2 = (shift * 2) % d_bits
    s_sh = s_f  # shift 0
    p_sh = cyclic_shift_1d(p_f, shift)
    o_sh = cyclic_shift_1d(o_f, shift2)
    return majority_vote_3(s_sh, p_sh, o_sh)
print("Running compression benchmarks...", file=sys.stderr)
results = []
# ─── GROUND TRUTH ─────────────────────────────────────────────
results.append({
    "method": "Exact S+P+O",
    "category": "ground_truth",
    "bits": D_TOTAL,
    "rho_random": 1.0,
    "rho_structured": 1.0,
})
# ─── CYCLIC BUNDLE at multiple dimensions ─────────────────────
for d_bits in [32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64]:
    print(f"  Cyclic bundle d={d_bits}...", file=sys.stderr)
    bundles_r = np.array([bundle_at_dim(nodes_s[i], nodes_p[i], nodes_o[i], d_bits)
                          for i in range(N_NODES)])
    bundles_s = np.array([bundle_at_dim(struct_s[i], struct_p[i], struct_o[i], d_bits)
                          for i in range(N_NODES)])

    dist_r = np.zeros((N_NODES, N_NODES), dtype=np.int32)
    dist_s = np.zeros((N_NODES, N_NODES), dtype=np.int32)
    for i in range(N_NODES):
        dist_r[i] = np.sum(bundles_r[i] != bundles_r, axis=1)
        dist_s[i] = np.sum(bundles_s[i] != bundles_s, axis=1)

    rho_r = mean_spearman(dist_r, exact_dists)
    rho_s = mean_spearman(dist_s, struct_dists)
    results.append({
        "method": f"Cyclic bundle",
        "category": "cyclic_bundle",
        "bits": d_bits,
        "rho_random": round(rho_r, 4),
        "rho_structured": round(rho_s, 4),
    })
# ─── SIMPLE TRUNCATION ────────────────────────────────────────
for d_bits in [16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64]:
    print(f"  Truncation d={d_bits}...", file=sys.stderr)
    third = d_bits // 3
    trunc_r = np.zeros((N_NODES, d_bits), dtype=np.uint8)
    trunc_s = np.zeros((N_NODES, d_bits), dtype=np.uint8)
    for i in range(N_NODES):
        trunc_r[i, :third] = nodes_s[i, :third]
        trunc_r[i, third:2*third] = nodes_p[i, :third]
        trunc_r[i, 2*third:d_bits] = nodes_o[i, :(d_bits - 2*third)]
        trunc_s[i, :third] = struct_s[i, :third]
        trunc_s[i, third:2*third] = struct_p[i, :third]
        trunc_s[i, 2*third:d_bits] = struct_o[i, :(d_bits - 2*third)]

    dist_r = np.zeros((N_NODES, N_NODES), dtype=np.int32)
    dist_s = np.zeros((N_NODES, N_NODES), dtype=np.int32)
    for i in range(N_NODES):
        dist_r[i] = np.sum(trunc_r[i] != trunc_r, axis=1)
        dist_s[i] = np.sum(trunc_s[i] != trunc_s, axis=1)

    rho_r = mean_spearman(dist_r, exact_dists)
    rho_s = mean_spearman(dist_s, struct_dists)
    results.append({
        "method": f"Truncation",
        "category": "truncation",
        "bits": d_bits,
        "rho_random": round(rho_r, 4),
        "rho_structured": round(rho_s, 4),
    })
# ─── SIMHASH (random projection) ─────────────────────────────
for d_bits in [16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64]:
    print(f"  SimHash d={d_bits}...", file=sys.stderr)
    # Random projection matrix (binary): sample d_bits positions from D_TOTAL
    proj_indices = np.random.choice(D_TOTAL, size=d_bits, replace=d_bits > D_TOTAL)

    sim_r = np.zeros((N_NODES, d_bits), dtype=np.uint8)
    sim_s = np.zeros((N_NODES, d_bits), dtype=np.uint8)
    for i in range(N_NODES):
        full_r = np.concatenate([nodes_s[i], nodes_p[i], nodes_o[i]])
        full_s = np.concatenate([struct_s[i], struct_p[i], struct_o[i]])
        sim_r[i] = full_r[proj_indices]
        sim_s[i] = full_s[proj_indices]

    dist_r = np.zeros((N_NODES, N_NODES), dtype=np.int32)
    dist_s = np.zeros((N_NODES, N_NODES), dtype=np.int32)
    for i in range(N_NODES):
        dist_r[i] = np.sum(sim_r[i] != sim_r, axis=1)
        dist_s[i] = np.sum(sim_s[i] != sim_s, axis=1)

    rho_r = mean_spearman(dist_r, exact_dists)
    rho_s = mean_spearman(dist_s, struct_dists)
    results.append({
        "method": f"SimHash",
        "category": "simhash",
        "bits": d_bits,
        "rho_random": round(rho_r, 4),
        "rho_structured": round(rho_s, 4),
    })
# ─── SEPARATE S-only, S+P, S+P+O ─────────────────────────────
print("  Component subsets...", file=sys.stderr)
s_only = np.zeros((N_NODES, N_NODES), dtype=np.int32)
sp_only = np.zeros((N_NODES, N_NODES), dtype=np.int32)
for i in range(N_NODES):
    s_only[i] = np.sum(nodes_s[i] != nodes_s, axis=1)
    sp_only[i] = s_only[i] + np.sum(nodes_p[i] != nodes_p, axis=1)
results.append({"method": "S-only", "category": "component", "bits": 16384,
                "rho_random": round(mean_spearman(s_only, exact_dists), 4),
                "rho_structured": round(mean_spearman(s_only, struct_dists), 4)})
results.append({"method": "S+P", "category": "component", "bits": 32768,
                "rho_random": round(mean_spearman(sp_only, exact_dists), 4),
                "rho_structured": round(mean_spearman(sp_only, struct_dists), 4)})
# ─── POPCOUNT SUMMARY (scalar compression) ───────────────────
for n_regions in [256, 128, 64, 32, 16, 8, 4, 2, 1]:
    print(f"  Popcount {n_regions} regions...", file=sys.stderr)
    region_size = D_PLANE // n_regions
    bits_per_count = max(1, int(np.ceil(np.log2(region_size * 64 + 1))))
    total_bits = n_regions * 3 * bits_per_count

    summaries_r = np.zeros((N_NODES, n_regions * 3), dtype=np.int32)
    summaries_s = np.zeros((N_NODES, n_regions * 3), dtype=np.int32)
    for i in range(N_NODES):
        for c_idx, (cr, cs) in enumerate([(nodes_s, struct_s), (nodes_p, struct_p), (nodes_o, struct_o)]):
            for r in range(n_regions):
                start = r * region_size
                end = start + region_size
                summaries_r[i, c_idx * n_regions + r] = np.sum(cr[i, start:end])
                summaries_s[i, c_idx * n_regions + r] = np.sum(cs[i, start:end])

    dist_r = np.zeros((N_NODES, N_NODES), dtype=np.int32)
    dist_s = np.zeros((N_NODES, N_NODES), dtype=np.int32)
    for i in range(N_NODES):
        dist_r[i] = np.sum(np.abs(summaries_r[i] - summaries_r), axis=1)
        dist_s[i] = np.sum(np.abs(summaries_s[i] - summaries_s), axis=1)

    rho_r = mean_spearman(dist_r, exact_dists)
    rho_s = mean_spearman(dist_s, struct_dists)
    results.append({
        "method": f"Popcount {n_regions}x3",
        "category": "popcount",
        "bits": total_bits,
        "rho_random": round(rho_r, 4),
        "rho_structured": round(rho_s, 4),
    })
# ─── BAND CLASSIFICATION (ZeckF16-style) ─────────────────────
for n_bands in [256, 64, 16, 8, 4, 2]:
    print(f"  Band {n_bands} levels...", file=sys.stderr)
    bits_per_band = max(1, int(np.ceil(np.log2(n_bands))))
    total_bits = 7 * bits_per_band + 1  # 7 masks + sign

    # For each pair, compute 7 mask distances and quantize to bands
    dist_r = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    dist_s = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    for i in range(min(N_QUERIES, N_NODES)):
        for j in range(N_NODES):
            if i == j: continue
            ds = np.sum(nodes_s[i] != nodes_s[j])
            dp = np.sum(nodes_p[i] != nodes_p[j])
            do = np.sum(nodes_o[i] != nodes_o[j])
            bands = [
                int(ds / D_PLANE * n_bands),
                int(dp / D_PLANE * n_bands),
                int(do / D_PLANE * n_bands),
                int((ds + dp) / (2 * D_PLANE) * n_bands),
                int((ds + do) / (2 * D_PLANE) * n_bands),
                int((dp + do) / (2 * D_PLANE) * n_bands),
                int((ds + dp + do) / (3 * D_PLANE) * n_bands),
            ]
            dist_r[i, j] = sum(bands)

            ds2 = np.sum(struct_s[i] != struct_s[j])
            dp2 = np.sum(struct_p[i] != struct_p[j])
            do2 = np.sum(struct_o[i] != struct_o[j])
            bands2 = [
                int(ds2 / D_PLANE * n_bands),
                int(dp2 / D_PLANE * n_bands),
                int(do2 / D_PLANE * n_bands),
                int((ds2 + dp2) / (2 * D_PLANE) * n_bands),
                int((ds2 + do2) / (2 * D_PLANE) * n_bands),
                int((dp2 + do2) / (2 * D_PLANE) * n_bands),
                int((ds2 + dp2 + do2) / (3 * D_PLANE) * n_bands),
            ]
            dist_s[i, j] = sum(bands2)

    rho_r = mean_spearman(dist_r, exact_dists, min(N_QUERIES, N_NODES))
    rho_s = mean_spearman(dist_s, struct_dists, min(N_QUERIES, N_NODES))
    results.append({
        "method": f"Bands 7x{n_bands}lvl",
        "category": "bands",
        "bits": total_bits,
        "rho_random": round(rho_r, 4),
        "rho_structured": round(rho_s, 4),
    })
print("\nDone. Writing results...", file=sys.stderr)
print(json.dumps(results, indent=2))
