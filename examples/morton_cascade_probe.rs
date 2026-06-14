//! Morton 2×2 cascade probe — non-materialized Z-order quadtree over the
//! gridlake SoA carrier, with Belichtungsmesser-style min/max early-exit.
//!
//! ## What this validates (probe-first, codec-agnostic)
//!
//! - **4×4 Morton leaf tile = one `F32x16`** ("2bit×2bit": 2 bits X, 2 bits Y =
//!   16 cells = 64 bytes = one AVX-512 register loaded from `MultiLaneColumn`).
//! - **Quadtree over T×T tiles** (2×2 per level): total grid `(4T)²` for
//!   `T = 2^k` gives the ladder 64, 256, 1024, 4096, 16384, 64k, 256k.
//! - **Morton order ⇒ every quadtree node is a contiguous index range**, so the
//!   aggregate (min/max) pyramid is a flat bottom-up reduction (the
//!   Belichtungsmesser "calibrated bands" = per-node value range).
//! - **Cascade early-exit**: descend the pyramid; if a node's [min,max] can't
//!   intersect the query band [q−r, q+r], prune the whole subtree (the 3-stroke
//!   band-miss generalized). Leaf tile → `F32x16` load + test 16 cells.
//!
//! The cell value is a plain `f32` stand-in for the eventual per-cell codec
//! (palette256 / helix Fisher-2z / Belichtungsmesser band) — wiring that is the
//! next step; this probe proves the *substrate* (addressing + cascade) is
//! correct and that the prune actually skips work.
//!
//!   RUSTFLAGS="-C target-cpu=native" cargo run --release --example morton_cascade_probe
//!
//! PASS: cascade count == brute-force count for every (size, query); the
//! reported prune-rate is the "boost".

use std::sync::Arc;

use ndarray::simd::{F32x16, MultiLaneColumn};

/// Interleave the low `bits` of `x` and `y` into a Z-order (Morton) index.
/// x occupies even output bits, y the odd bits.
fn morton2d(x: u32, y: u32, bits: u32) -> u32 {
    let mut m = 0u32;
    for b in 0..bits {
        m |= ((x >> b) & 1) << (2 * b);
        m |= ((y >> b) & 1) << (2 * b + 1);
    }
    m
}

/// Deterministic SplitMix64 → f32 in [0, 1).
fn splitmix(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32) / (1u32 << 24) as f32
}

/// Build the field in Morton-tile order:
///   cell(x,y) → idx = morton(tx,ty)·16 + morton_in_tile(ix,iy)
/// where tile (tx,ty) = (x>>2, y>>2), in-tile (ix,iy) = (x&3, y&3).
/// Each 4×4 tile is therefore a contiguous 16-f32 (64-byte) `F32x16` chunk.
fn build_field(t: u32, seed: u64) -> Vec<f32> {
    let side = 4 * t; // grid side
    let n = (side * side) as usize;
    let mut field = vec![0.0f32; n];
    let k = t.trailing_zeros(); // T = 2^k tiles per side
    let mut st = seed;
    for y in 0..side {
        for x in 0..side {
            let (tx, ty) = (x >> 2, y >> 2);
            let (ix, iy) = (x & 3, y & 3);
            let idx = (morton2d(tx, ty, k) as usize) * 16 + morton2d(ix, iy, 2) as usize;
            // A smooth-ish field + noise so neighbouring tiles share value ranges
            // (otherwise every node spans the full range and nothing prunes).
            let base = ((x as f32) / side as f32 + (y as f32) / side as f32) * 0.5;
            field[idx] = 0.85 * base + 0.15 * splitmix(&mut st);
        }
    }
    field
}

/// Aggregate (min,max) pyramid over the T² tiles in Morton order. Level 0 =
/// per-tile range (over its 16 cells); level l = range of 4 level-(l−1) nodes.
/// A node at level l covers `4^l` contiguous tiles starting at `base`.
struct Pyramid {
    levels: Vec<Vec<(f32, f32)>>, // levels[0] = per-tile, levels[K] = root
    k: u32,                       // number of quadtree levels (T = 2^k)
}

impl Pyramid {
    fn build(field: &[f32], t: u32) -> Self {
        let k = t.trailing_zeros();
        let n_tiles = (t * t) as usize;
        // Level 0: min/max over each tile's 16 cells.
        let mut lvl0 = Vec::with_capacity(n_tiles);
        for tile in 0..n_tiles {
            let s = &field[tile * 16..tile * 16 + 16];
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for &v in s {
                mn = mn.min(v);
                mx = mx.max(v);
            }
            lvl0.push((mn, mx));
        }
        let mut levels = vec![lvl0];
        for l in 1..=k as usize {
            let prev = &levels[l - 1];
            let mut cur = Vec::with_capacity(prev.len() / 4);
            for node in prev.chunks_exact(4) {
                let mn = node.iter().map(|p| p.0).fold(f32::INFINITY, f32::min);
                let mx = node.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
                cur.push((mn, mx));
            }
            levels.push(cur);
        }
        Pyramid { levels, k }
    }
}

/// Cascade query: count cells with |value − q| ≤ r, descending the quadtree and
/// pruning any node whose [min,max] can't intersect [q−r, q+r]. Returns
/// (count, cells_visited) — cells_visited only counts leaf cells actually tested.
fn cascade_count(field: &[f32], col: &MultiLaneColumn, pyr: &Pyramid, q: f32, r: f32) -> (usize, usize) {
    let (lo, hi) = (q - r, q + r);
    let mut count = 0usize;
    let mut visited = 0usize;
    // Stack of (level, node_index_within_level).
    let mut stack = vec![(pyr.k as usize, 0usize)];
    let bytes = col.as_bytes();
    while let Some((level, node)) = stack.pop() {
        let (mn, mx) = pyr.levels[level][node];
        if mx < lo || mn > hi {
            continue; // band miss → prune whole subtree (early-exit)
        }
        if level == 0 {
            // Leaf tile = 16 cells = one F32x16 chunk in the SoA column.
            let off = node * 64; // 16 f32 × 4 bytes
            let chunk: [u8; 64] = bytes[off..off + 64].try_into().unwrap();
            let arr = f32x16_from_bytes(&chunk).to_array();
            for &v in arr.iter() {
                if (v - q).abs() <= r {
                    count += 1;
                }
            }
            visited += 16;
        } else {
            let base = node * 4;
            for c in 0..4 {
                stack.push((level - 1, base + c));
            }
        }
    }
    let _ = field; // field kept for the brute-force reference; cascade reads the SoA column
    (count, visited)
}

/// Build an `F32x16` from 64 little-endian bytes (one 4×4 Morton tile).
fn f32x16_from_bytes(chunk: &[u8; 64]) -> F32x16 {
    let arr: [f32; 16] = core::array::from_fn(|i| {
        let o = i * 4;
        f32::from_le_bytes([chunk[o], chunk[o + 1], chunk[o + 2], chunk[o + 3]])
    });
    F32x16::from_array(arr)
}

fn brute_count(field: &[f32], q: f32, r: f32) -> usize {
    field.iter().filter(|&&v| (v - q).abs() <= r).count()
}

fn run(t: u32) {
    let side = 4 * t;
    let n = (side * side) as usize;
    let field = build_field(t, 0xC0FFEE ^ t as u64);
    // Wrap the Morton-ordered field bytes in the gridlake SoA carrier.
    let raw: Vec<u8> = field.iter().flat_map(|v| v.to_le_bytes()).collect();
    let col = MultiLaneColumn::new(Arc::from(raw.into_boxed_slice())).unwrap();
    let pyr = Pyramid::build(&field, t);

    // Three query bands: tight (high prune), medium, broad (low prune).
    let queries = [(0.5f32, 0.02f32), (0.25, 0.10), (0.5, 0.5)];
    let mut all_ok = true;
    let mut report = String::new();
    for (q, r) in queries {
        let exp = brute_count(&field, q, r);
        let (got, visited) = cascade_count(&field, &col, &pyr, q, r);
        let ok = got == exp;
        all_ok &= ok;
        let prune = 100.0 * (1.0 - visited as f64 / n as f64);
        report.push_str(&format!(
            "    q={q:.2} r={r:.2}: count {got:>7} (exp {exp:>7}) {}  visited {visited:>8}/{n} → prune {prune:5.1}%\n",
            if ok { "OK " } else { "MISMATCH" }
        ));
    }
    println!(
        "  T={t:>3} tiles/side  grid {side}×{side} = {n:>7} cells  ({})\n{}",
        if all_ok { "CORRECT" } else { "WRONG" },
        report
    );
}

fn main() {
    println!("== Morton 2×2 quadtree cascade probe (4×4 tile = F32x16, gridlake SoA) ==\n");
    // T = 2^k → grid (4T)² = the ladder 64, 256, 1024, 4096, 16384, 64k, 256k.
    for t in [2u32, 4, 8, 16, 32, 64, 128] {
        run(t);
    }
}
