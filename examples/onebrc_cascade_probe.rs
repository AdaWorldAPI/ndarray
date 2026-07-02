//! 1BRC-on-substrate probe — min/mean/max per station as a monoid group-by
//! over the gridlake Morton substrate, with an AMX BF16 tile-GEMM mean path
//! and a Belichtungsmesser band-prune demo on the aggregate pyramid.
//!
//! ## What this validates (probe-first, sibling of `morton_cascade_probe`)
//!
//! The One Billion Row Challenge workload (station;temperature → min/mean/max
//! per station) restated on this substrate:
//!
//! - **Stations are addresses, not hash keys.** Each station is minted a cell
//!   on a `(4T)²` Morton grid (4×4 leaf tile = one `F32x16` = 64 bytes), so
//!   the per-station accumulators live at canonical Z-order addresses in a
//!   `MultiLaneColumn`-backed SoA — the gridlake carrier, not a hashmap.
//! - **Aggregation is a commutative monoid fold** `(min, max, Σ, n)` — the
//!   same algebra blasgraph's semiring folds rely on. Morsel-batched scatter
//!   (64K rows/morsel) into L1-resident accumulators is the "blasgraph-like
//!   cache algorithm": the hot set is `n_stations × 24 B ≈ 10 KB`, so the
//!   scatter never leaves L1; the Morton layout makes the pyramid fold
//!   cache-oblivious.
//! - **Group-by as tile GEMM (the AMX leg).** `(Σ, n)` per station is also a
//!   matmul: `C[16×16] += A[16×K] · B[K×16]` with B a per-row one-hot station
//!   indicator (stations in column-blocks of 16) and three live A rows —
//!   row 0 = 1 (count), row 1 = hi(temp), row 2 = lo(temp). BF16 has an 8-bit
//!   significand, so integer tenths in [-999, 999] are NOT bf16-exact; the
//!   split `hi = (t/256)·256, lo = t − hi` makes every operand exact in BF16
//!   (hi ∈ {0, ±256, ±512, ±768}, |lo| ≤ 255 < 2^8), and per-tile f32
//!   accumulation stays exact (≤ K·999 < 2^24 for K = 4096). A-row 3 carries
//!   naive bf16-RNE temps through the SAME tile — the "is BF16 precise
//!   enough?" experiment, measured instead of argued (the extra row is free).
//!   Runs through `ndarray::simd::bf16_tile_gemm_16x16_amx` (TDPBF16PS when
//!   `amx_available()`, the F32x16 FMA polyfill otherwise) — all imports via
//!   the canonical `ndarray::simd::*` surface, per the W1a consumer contract.
//!   Per AMX Gotcha 9 ("a skipped test is not a passing test") the probe
//!   PRINTS which tier actually ran. min/max stay on the scatter path — AMX
//!   has no min/max tile op (TDPBF16PS/TDPBUSD are dot-product accumulates
//!   only).
//!
//!   ⚠ Gotcha 14 (DISCOVERED BY THIS PROBE, 2026-07-02): on this oversubscribed
//!   VM, AMX tile state is silently corrupted under host CPU contention —
//!   idle runs are bit-exact at 100M rows; with 4 busy-loop competitors the
//!   GEMM leg drops whole rows (89-152/413 stations exact), and pinning the
//!   probe to an uncontended core does NOT help. The scatter path (AVX-512)
//!   stays exact under the same load, isolating the corruption to TMM state.
//!   A FAIL of the GEMM leg on a loaded box is the probe working as designed.
//!   See `.claude/AMX_GOTCHAS.md` § Gotcha 14.
//! - **The cascade is a reduction pyramid here, not a search cascade.** Every
//!   row must be touched (nothing to skip on input); what the pyramid buys is
//!   free *hierarchical* aggregates — min/mean/max per tile / region / root in
//!   the same pass — plus band-pruned queries over the result ("which stations
//!   have min ≤ q" visits only intersecting subtrees).
//! - **Exactness is provable, not approximate.** Temperatures are integer
//!   tenths in [-999, 999] → exact in `f32`; sums of integer tenths stay
//!   < 2^53 → exact in `f64`. Both substrate paths must match the scalar
//!   reference bit-for-bit. The algebraic side (partition/regroup invariance
//!   of the monoid fold, the BF16 hi/lo decomposition) is certified
//!   independently in `lance-graph/crates/jc` (`onebrc_agg` probe) —
//!   kernels here, proof there.
//!
//!   cargo run --release --example onebrc_cascade_probe
//!   ONEBRC_ROWS=100000000 cargo run --release --example onebrc_cascade_probe
//!
//! PASS: both substrate paths (Morton scatter; AMX/BF16 tile GEMM) match the
//! scalar reference exactly for every station, root invariants hold, and the
//! band-prune query returns the brute-force station set. Throughput and
//! prune-rate lines are the measured "boost".

use std::sync::Arc;
use std::time::Instant;

use ndarray::simd::{amx_available, bf16_tile_gemm_16x16_amx, F64x8, MultiLaneColumn};

// ── Deterministic RNG (same SplitMix64 as jc / hpc::pillar) ─────────────────

const SEED: u64 = 0x1BC_0FFEE;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ── Morton addressing (identical to morton_cascade_probe) ───────────────────

fn morton2d(x: u32, y: u32, bits: u32) -> u32 {
    let mut m = 0u32;
    for b in 0..bits {
        m |= ((x >> b) & 1) << (2 * b);
        m |= ((y >> b) & 1) << (2 * b + 1);
    }
    m
}

/// cell (x,y) on a (4T)² grid → flat Morton index with 4×4-tile granularity:
/// tile (x>>2, y>>2) ordered by morton2d, 16 cells per tile ordered by
/// morton2d(x&3, y&3, 2). Each tile is one contiguous 16-lane chunk.
fn cell_index(x: u32, y: u32, k: u32) -> usize {
    let (tx, ty) = (x >> 2, y >> 2);
    let (ix, iy) = (x & 3, y & 3);
    (morton2d(tx, ty, k) as usize) * 16 + morton2d(ix, iy, 2) as usize
}

// ── Workload parameters ─────────────────────────────────────────────────────

const T: u32 = 16; // tiles per side → grid 64×64 = 4096 cells, 256 tiles
const N_STATIONS: usize = 413; // the classic 1BRC station count
const MORSEL: usize = 1 << 16; // 64K rows per batch (scatter path)
const GEMM_K: usize = 4096; // tile-GEMM sub-morsel (multiple of 32)
const N_GROUPS: usize = N_STATIONS.div_ceil(16); // 16-station column blocks

// ── Scalar reference (ground truth, integer domain) ─────────────────────────

#[derive(Clone, Copy)]
struct RefAgg {
    min_t: i16, // tenths
    max_t: i16,
    sum_t: i64,
    cnt: u64,
}

impl RefAgg {
    const IDENTITY: RefAgg = RefAgg {
        min_t: i16::MAX,
        max_t: i16::MIN,
        sum_t: 0,
        cnt: 0,
    };
}

// ── Substrate accumulators (SoA over Morton cells) ──────────────────────────

struct MortonAgg {
    min_c: Vec<f32>, // +INF identity
    max_c: Vec<f32>, // -INF identity
    sum_c: Vec<f64>, // integer tenths, exact while < 2^53
    cnt_c: Vec<u64>,
}

impl MortonAgg {
    fn new(n_cells: usize) -> Self {
        MortonAgg {
            min_c: vec![f32::INFINITY; n_cells],
            max_c: vec![f32::NEG_INFINITY; n_cells],
            sum_c: vec![0.0; n_cells],
            cnt_c: vec![0; n_cells],
        }
    }
}

/// One pyramid node: the same (min, max, Σ, n) monoid element, per subtree.
#[derive(Clone, Copy)]
struct Node {
    min: f32,
    max: f32,
    sum: f64,
    cnt: u64,
}

/// Aggregate pyramid over the T² tiles in Morton order. Level 0 = per-tile
/// fold of 16 cells; level l = fold of 4 level-(l−1) nodes; root = global.
struct Pyramid {
    levels: Vec<Vec<Node>>,
    k: u32,
}

impl Pyramid {
    /// Level-0 min/max folds go through the gridlake carrier: 16 cells =
    /// one 64-byte `F32x16` chunk of a `MultiLaneColumn`, reduced in-register.
    fn build(agg: &MortonAgg, t: u32) -> Self {
        let k = t.trailing_zeros();
        let n_tiles = (t * t) as usize;

        // Wrap the min/max channels in the SoA byte carrier (LE f32 lanes).
        let min_col = column_from_f32(&agg.min_c);
        let max_col = column_from_f32(&agg.max_c);

        let mut lvl0 = Vec::with_capacity(n_tiles);
        for (tile, (min_v, max_v)) in min_col.iter_f32x16().zip(max_col.iter_f32x16()).enumerate() {
            let base = tile * 16;
            let mut sum = 0.0f64;
            let mut cnt = 0u64;
            for c in 0..16 {
                sum += agg.sum_c[base + c];
                cnt += agg.cnt_c[base + c];
            }
            // Two F64x8 loads cross-check the scalar Σ (exact: integer tenths).
            let s_lo = F64x8::from_array(agg.sum_c[base..base + 8].try_into().unwrap());
            let s_hi = F64x8::from_array(agg.sum_c[base + 8..base + 16].try_into().unwrap());
            debug_assert_eq!(s_lo.reduce_sum() + s_hi.reduce_sum(), sum);
            let _ = (s_lo, s_hi);
            lvl0.push(Node {
                min: min_v.reduce_min(),
                max: max_v.reduce_max(),
                sum,
                cnt,
            });
        }

        let mut levels = vec![lvl0];
        for l in 1..=k as usize {
            let prev = &levels[l - 1];
            let mut cur = Vec::with_capacity(prev.len() / 4);
            for q in prev.chunks_exact(4) {
                cur.push(Node {
                    min: q.iter().map(|n| n.min).fold(f32::INFINITY, f32::min),
                    max: q.iter().map(|n| n.max).fold(f32::NEG_INFINITY, f32::max),
                    sum: q.iter().map(|n| n.sum).sum(),
                    cnt: q.iter().map(|n| n.cnt).sum(),
                });
            }
            levels.push(cur);
        }
        Pyramid { levels, k }
    }

    fn root(&self) -> Node {
        self.levels[self.k as usize][0]
    }

    /// Band-prune descent on the MIN channel: visit only subtrees whose
    /// min ≤ q; return (leaf tiles visited, matching cell indices).
    fn stations_with_min_le(&self, q: f32, agg: &MortonAgg) -> (usize, Vec<usize>) {
        let mut visited = 0usize;
        let mut hits = Vec::new();
        let mut stack = vec![(self.k as usize, 0usize)];
        while let Some((level, node)) = stack.pop() {
            if self.levels[level][node].min > q {
                continue; // whole subtree pruned
            }
            if level == 0 {
                visited += 1;
                let base = node * 16;
                for c in 0..16 {
                    if agg.min_c[base + c] <= q {
                        hits.push(base + c);
                    }
                }
            } else {
                let base = node * 4;
                for c in 0..4 {
                    stack.push((level - 1, base + c));
                }
            }
        }
        hits.sort_unstable();
        (visited, hits)
    }
}

fn column_from_f32(vals: &[f32]) -> MultiLaneColumn {
    let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    MultiLaneColumn::new(Arc::from(raw.into_boxed_slice())).unwrap()
}

// ── AMX BF16 tile-GEMM mean path ────────────────────────────────────────────

/// f32 → bf16 by truncation. Every value fed through here is exactly
/// representable in BF16 (0, ±1, hi multiples of 256 ≤ 768, |lo| ≤ 255),
/// so truncation is lossless and equals round-to-nearest-even.
#[inline(always)]
fn bf16_exact(v: f32) -> u16 {
    let bits = v.to_bits();
    debug_assert_eq!(bits & 0xFFFF, 0, "value not bf16-exact: {v}");
    (bits >> 16) as u16
}

const BF16_ONE: u16 = 0x3F80; // 1.0f32 >> 16

/// f32 → bf16 with round-to-nearest-even — the conversion a naive "just store
/// the temperature as BF16" pipeline would use. NOT exact for |tenths| > 255;
/// A-row 3 measures exactly how much that costs (the "is BF16 precise enough?"
/// experiment — see module doc).
#[inline(always)]
fn bf16_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits.wrapping_add(0x7FFF).wrapping_add((bits >> 16) & 1)) >> 16) as u16
}

/// Group-by as tile GEMM over one sub-morsel of `rows` (≤ GEMM_K) rows.
///
/// A[16×K] (bf16, row-major): row 0 = 1.0 (count), row 1 = hi(temp),
/// row 2 = lo(temp), row 3 = bf16-RNE(temp) (the naive-precision experiment),
/// rows 4-15 = 0. B_blocks[g][K×16] (bf16, row-major): per-row one-hot
/// indicator for station group g (stations g·16 ..= g·16+15).
/// C[i][j] = Σ_r A[i][r]·B[r][j] gives per station j of group g:
/// C[0][j] = n, C[1][j] = Σhi, C[2][j] = Σlo — all exact (see module doc) —
/// and C[3][j] = Σ bf16(temp), whose deviation from Σhi+Σlo is the measured
/// cost of skipping the hi/lo split. The extra row is free: same tile, same
/// GEMM call.
///
/// B blocks are zeroed ONCE at allocation; each sub-morsel sets exactly one
/// entry per row and clears the same entry afterwards (clear-by-undo), so the
/// per-morsel cost is O(rows), not O(rows × groups) — the same L1-resident
/// hot-set discipline as the scatter path.
struct GemmGroupBy {
    a: Vec<u16>,             // 16 × GEMM_K, rows 0 and 4-15 pre-set
    b_blocks: Vec<Vec<u16>>, // N_GROUPS × (GEMM_K × 16)
    c: Vec<f32>,             // 16 × 16 output tile
    sum_t: Vec<i64>,         // per-station Σ (tenths), drained exactly
    cnt: Vec<u64>,           // per-station n
    sum_bf16: Vec<f64>,      // per-station Σ of bf16-rounded temps (row 3)
}

impl GemmGroupBy {
    fn new() -> Self {
        let mut a = vec![0u16; 16 * GEMM_K];
        a[..GEMM_K].fill(BF16_ONE); // row 0 = ones → counts
        GemmGroupBy {
            a,
            b_blocks: vec![vec![0u16; GEMM_K * 16]; N_GROUPS],
            c: vec![0.0f32; 256],
            sum_t: vec![0i64; N_STATIONS],
            cnt: vec![0u64; N_STATIONS],
            sum_bf16: vec![0.0f64; N_STATIONS],
        }
    }

    fn fold_sub_morsel(&mut self, rows: &[(u16, i16)]) {
        debug_assert!(rows.len() <= GEMM_K);
        // Stage A rows 1-2 (hi/lo split) and the one-hot B entries. Rows of a
        // partial sub-morsel beyond `rows.len()` keep stale A values — their B
        // indicator is never set, so they contribute exact zeros.
        for (r, &(sid, temp)) in rows.iter().enumerate() {
            let hi = (temp as i32 / 256) * 256;
            let lo = temp as i32 - hi;
            self.a[GEMM_K + r] = bf16_exact(hi as f32);
            self.a[2 * GEMM_K + r] = bf16_exact(lo as f32);
            self.a[3 * GEMM_K + r] = bf16_rne(temp as f32);
            let (g, j) = (sid as usize / 16, sid as usize % 16);
            self.b_blocks[g][r * 16 + j] = BF16_ONE;
        }

        for g in 0..N_GROUPS {
            self.c.fill(0.0); // gemm ACCUMULATES; each group starts clean
            bf16_tile_gemm_16x16_amx(&self.a, &self.b_blocks[g], &mut self.c, GEMM_K);
            for j in 0..16 {
                let s = g * 16 + j;
                if s >= N_STATIONS {
                    break;
                }
                // Every C entry in rows 0-2 is an exact integer (see module
                // doc), so the drain into i64/u64 is lossless. Row 3 carries
                // the bf16-quantized sums (exact f32 sums of INEXACT inputs).
                self.cnt[s] += self.c[j] as u64;
                self.sum_t[s] += (self.c[16 + j] as f64 + self.c[32 + j] as f64) as i64;
                self.sum_bf16[s] += self.c[48 + j] as f64;
            }
        }

        // Clear-by-undo: reset exactly the B entries this sub-morsel set.
        for (r, &(sid, _)) in rows.iter().enumerate() {
            let (g, j) = (sid as usize / 16, sid as usize % 16);
            self.b_blocks[g][r * 16 + j] = 0;
        }
    }
}

// ── Probe ───────────────────────────────────────────────────────────────────

fn main() {
    let rows: usize = std::env::var("ONEBRC_ROWS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10_000_000);

    let side = 4 * T;
    let n_cells = (side * side) as usize;
    let k = T.trailing_zeros();

    println!("== 1BRC cascade probe (Morton {side}×{side}, {N_STATIONS} stations, {rows} rows) ==\n");

    // 1. Mint station addresses: distinct cells on the Morton grid, plus a
    //    true mean per station (integer tenths in [-400, 400]).
    let mut st = SEED;
    let mut taken = vec![false; n_cells];
    let mut station_cell = Vec::with_capacity(N_STATIONS);
    let mut station_mean_t = Vec::with_capacity(N_STATIONS);
    while station_cell.len() < N_STATIONS {
        let x = (splitmix64(&mut st) % side as u64) as u32;
        let y = (splitmix64(&mut st) % side as u64) as u32;
        let idx = cell_index(x, y, k);
        if !taken[idx] {
            taken[idx] = true;
            station_cell.push(idx);
            station_mean_t.push((splitmix64(&mut st) % 801) as i16 - 400);
        }
    }

    // 2. Morsel-batched folds: generate rows deterministically, feed the SAME
    //    morsel buffer to all three paths (scalar reference; Morton scatter;
    //    tile-GEMM group-by). Data-flow: read-only morsel slices in, owned
    //    accumulators, no shared `&mut` during the fold.
    let mut reference = vec![RefAgg::IDENTITY; N_STATIONS];
    let mut agg = MortonAgg::new(n_cells);
    let mut gemm = GemmGroupBy::new();
    let mut morsel: Vec<(u16, i16)> = Vec::with_capacity(MORSEL);

    let (mut t_ref, mut t_sub, mut t_gemm) = (0.0f64, 0.0f64, 0.0f64);
    let mut produced = 0usize;
    while produced < rows {
        let batch = MORSEL.min(rows - produced);
        morsel.clear();
        for _ in 0..batch {
            let sid = (splitmix64(&mut st) % N_STATIONS as u64) as u16;
            let noise = (splitmix64(&mut st) % 201) as i16 - 100; // ±10.0 °C
            let temp = (station_mean_t[sid as usize] + noise).clamp(-999, 999);
            morsel.push((sid, temp));
        }
        produced += batch;

        let t0 = Instant::now();
        for &(sid, temp) in &morsel {
            let r = &mut reference[sid as usize];
            r.min_t = r.min_t.min(temp);
            r.max_t = r.max_t.max(temp);
            r.sum_t += temp as i64;
            r.cnt += 1;
        }
        t_ref += t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        for &(sid, temp) in &morsel {
            let idx = station_cell[sid as usize];
            let v = temp as f32;
            agg.min_c[idx] = agg.min_c[idx].min(v);
            agg.max_c[idx] = agg.max_c[idx].max(v);
            agg.sum_c[idx] += temp as f64;
            agg.cnt_c[idx] += 1;
        }
        t_sub += t1.elapsed().as_secs_f64();

        let t2 = Instant::now();
        for sub in morsel.chunks(GEMM_K) {
            gemm.fold_sub_morsel(sub);
        }
        t_gemm += t2.elapsed().as_secs_f64();
    }

    // 3. Aggregate pyramid (hierarchical min/mean/max for free).
    let t3 = Instant::now();
    let pyr = Pyramid::build(&agg, T);
    let t_pyr = t3.elapsed().as_secs_f64();

    // 4a. Certify Morton-scatter path == reference, bit-for-bit, per station.
    let mut scatter_mism = 0usize;
    for (s, r) in reference.iter().enumerate() {
        let idx = station_cell[s];
        let ok = agg.min_c[idx] == r.min_t as f32
            && agg.max_c[idx] == r.max_t as f32
            && agg.sum_c[idx] == r.sum_t as f64
            && agg.cnt_c[idx] == r.cnt;
        if !ok {
            scatter_mism += 1;
            if scatter_mism <= 3 {
                println!(
                    "  SCATTER MISMATCH station {s}: sub(min={} max={} sum={} n={}) ref(min={} max={} sum={} n={})",
                    agg.min_c[idx], agg.max_c[idx], agg.sum_c[idx], agg.cnt_c[idx], r.min_t, r.max_t, r.sum_t, r.cnt
                );
            }
        }
    }

    // 4b. Certify tile-GEMM path (Σ, n) == reference, exactly, per station.
    let mut gemm_mism = 0usize;
    for (s, r) in reference.iter().enumerate() {
        if gemm.cnt[s] != r.cnt || gemm.sum_t[s] != r.sum_t {
            gemm_mism += 1;
            if gemm_mism <= 3 {
                println!(
                    "  GEMM MISMATCH station {s}: gemm(sum={} n={}) ref(sum={} n={})",
                    gemm.sum_t[s], gemm.cnt[s], r.sum_t, r.cnt
                );
            }
        }
    }

    // 4c. The "is BF16 precise enough?" measurement: A-row 3 carried naive
    // bf16-RNE temperatures through the same tile; compare the resulting
    // per-station means against the exact (hi/lo-split) means.
    let mut max_mean_err_t = 0.0f64; // tenths
    for (s, r) in reference.iter().enumerate() {
        if r.cnt == 0 {
            continue;
        }
        let exact = r.sum_t as f64 / r.cnt as f64;
        let naive = gemm.sum_bf16[s] / r.cnt as f64;
        max_mean_err_t = max_mean_err_t.max((naive - exact).abs());
    }

    // Root invariants: count, global min/max, global Σ.
    let root = pyr.root();
    let g_min = reference.iter().map(|r| r.min_t).min().unwrap();
    let g_max = reference.iter().map(|r| r.max_t).max().unwrap();
    let g_sum: i64 = reference.iter().map(|r| r.sum_t).sum();
    let root_ok =
        root.cnt == rows as u64 && root.min == g_min as f32 && root.max == g_max as f32 && root.sum == g_sum as f64;

    // 5. Band-prune query on the pyramid: stations with min ≤ q.
    let q = (g_min + 50) as f32; // a band 5.0 °C above the coldest reading
    let (visited, hits) = pyr.stations_with_min_le(q, &agg);
    let mut brute: Vec<usize> = (0..N_STATIONS)
        .filter(|&s| reference[s].min_t as f32 <= q)
        .map(|s| station_cell[s])
        .collect();
    brute.sort_unstable();
    let query_ok = hits == brute;
    let n_tiles = (T * T) as usize;
    let prune = 100.0 * (1.0 - visited as f64 / n_tiles as f64);

    // 6. Report (PillarReport style: deterministic seed, measured vs expected).
    let pass = scatter_mism == 0 && gemm_mism == 0 && root_ok && query_ok;
    let tier = if amx_available() {
        "AMX TDPBF16PS"
    } else {
        "AVX-512 F32x16 FMA fallback"
    };
    println!("  seed=0x{SEED:X}  stations={N_STATIONS}  rows={rows}");
    println!(
        "  morton scatter (min,max,Σ,n): {}/{} exact → {}",
        N_STATIONS - scatter_mism,
        N_STATIONS,
        if scatter_mism == 0 { "EXACT" } else { "MISMATCH" }
    );
    println!(
        "  bf16 tile-GEMM (Σ,n) [{tier}]: {}/{} exact → {}",
        N_STATIONS - gemm_mism,
        N_STATIONS,
        if gemm_mism == 0 { "EXACT" } else { "MISMATCH" }
    );
    println!(
        "  bf16-direct row (no hi/lo split): max |Δmean| = {:.4} tenths = {:.5} °C \
         (single reading off by ≤ 4 tenths: bf16 ulp at |t| ∈ [512, 1024))",
        max_mean_err_t,
        max_mean_err_t / 10.0
    );
    println!(
        "  root invariants (n={}, min={:.1}°C, max={:.1}°C): {}",
        root.cnt,
        root.min / 10.0,
        root.max / 10.0,
        if root_ok { "OK" } else { "WRONG" }
    );
    println!(
        "  band query min ≤ {:.1}°C: {} stations, visited {visited}/{n_tiles} tiles → prune {prune:.1}%  {}",
        q / 10.0,
        hits.len(),
        if query_ok { "OK" } else { "WRONG" }
    );
    println!(
        "  scatter: reference {:.0} Mrows/s | morton {:.0} Mrows/s | tile-GEMM {:.1} Mrows/s | pyramid {:.2} ms",
        rows as f64 / t_ref / 1e6,
        rows as f64 / t_sub / 1e6,
        rows as f64 / t_gemm / 1e6,
        t_pyr * 1e3
    );
    // Effective MAC rate of the GEMM formulation (dense-indicator overhead
    // is the honest price of group-by-as-matmul: N_GROUPS·16·16·K MACs per
    // K-row sub-morsel ≈ 6.7 kMAC/row).
    let macs = rows as f64 * (N_GROUPS * 256) as f64;
    println!(
        "  tile-GEMM effective rate: {:.1} GMAC/s (dense one-hot indicator, {} groups)",
        macs / t_gemm / 1e9,
        N_GROUPS
    );
    println!(
        "  hierarchical bonus: {} pyramid levels of regional (min,mean,max) in the same pass",
        pyr.levels.len()
    );
    println!("\n{}", if pass { "✓ PASS" } else { "✗ FAIL" });
    std::process::exit(i32::from(!pass));
}
