//! Codec mode-histogram probe — drives the SHIPPED `hpc::codec` RDO
//! selector over synthetic fields of varying coherence and reports the
//! actual %skip/merge/delta/escape mode histogram + bytes/cell.
//!
//! The point: HEVC-family compression is entirely a function of the
//! DATA's coherence. This measures where different data lands, so the
//! "3.3× / 10-50×" design targets can be read against a real mode split
//! instead of assumed. Raster scan with causal N/W neighbours — exactly
//! how a real encoder feeds Merge (E/S aren't decided yet).
//!
//! Run: `cargo run --release --example codec_mode_histogram --features codec`

use ndarray::hpc::codec::rdo::{rdo_select, RdoConfig, RdoContext};
use ndarray::hpc::codec::{packed_byte_len, CellMode, LeafCu};

const W: usize = 256;
const H: usize = 256;
const N: usize = W * H;
const NBASINS: i64 = 16;
const STEP: i64 = 64; // basin values 0, 64, .. 960 — codebook spans [0, 960]

/// SplitMix64 — deterministic, dependency-free.
struct Sm(u64);
impl Sm {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn range(&mut self, n: i64) -> i64 {
        (self.next() % n as u64) as i64
    }
}

/// Nearest basin: index + its value. Codebook spans [0, (NBASINS-1)*STEP].
fn basin_of(v: i64) -> (u16, i64) {
    let idx = ((v + STEP / 2).div_euclid(STEP)).clamp(0, NBASINS - 1);
    (idx as u16, idx * STEP)
}

struct Report {
    hist: [usize; 4], // skip, merge, delta, escape
    total_bytes: usize,
    mean_distortion: f64, // reconstruction error, u8-quant units
}

/// Encode `values` through the shipped RDO selector, raster order, causal
/// N/W neighbours. Escape feasible (cursor supplied) so out-of-range δ is
/// lossless. Bytes = leaf wire size + 8-byte escape payload per escape.
fn encode(values: &[i64], cfg: &RdoConfig) -> Report {
    let mut leaves: Vec<LeafCu> = Vec::with_capacity(N);
    let mut hist = [0usize; 4];
    let mut bytes = 0usize;
    let mut dist_sum = 0u64;
    let mut escape_cursor = 0u32;

    for y in 0..H {
        for x in 0..W {
            let v = values[y * W + x];
            let (basin_idx, basin_val) = basin_of(v);
            let delta = (v - basin_val) as i32;
            let north = if y > 0 { Some(&leaves[(y - 1) * W + x]) } else { None };
            let west = if x > 0 { Some(&leaves[y * W + (x - 1)]) } else { None };
            let ctx = RdoContext {
                basin_idx,
                delta_i32: delta,
                // [North, East, West, South] — E/S undecided in raster order
                neighbours: [north, None, west, None],
            };
            let choice = rdo_select(&ctx, cfg, Some(&mut escape_cursor));
            let m = choice.leaf.mode;
            hist[m as usize] += 1;
            bytes += packed_byte_len(m);
            if m == CellMode::Escape {
                bytes += 8; // full 64-bit value in the escape vector
            }
            dist_sum += choice.distortion as u64;
            leaves.push(choice.leaf);
        }
    }
    Report {
        hist,
        total_bytes: bytes,
        mean_distortion: dist_sum as f64 / N as f64,
    }
}

fn print_row(name: &str, r: &Report) {
    let pct = |k: usize| 100.0 * r.hist[k] as f64 / N as f64;
    let bpc = r.total_bytes as f64 / N as f64;
    println!(
        "  {name:<22} skip={:5.1}% merge={:5.1}% delta={:5.1}% escape={:5.1}%  |  {:.2} B/cell  {:.2}× vs 8B  meanDist={:.2}",
        pct(0),
        pct(1),
        pct(2),
        pct(3),
        bpc,
        8.0 / bpc,
        r.mean_distortion,
    );
}

fn main() {
    let cfg = RdoConfig::default(); // λ = 16, fidelity-biased (design's realistic setting)
    println!("Shipped hpc::codec RDO selector — {W}×{H} = {N} cells, λ=16 (default), 16-basin codebook\n");

    // (1) Coherent: flat 16×16 tiles, each tile sits exactly on a basin → δ=0.
    let mut a = vec![0i64; N];
    for y in 0..H {
        for x in 0..W {
            let tile = ((y / 16) * (W / 16) + (x / 16)) as i64;
            a[y * W + x] = (tile % NBASINS) * STEP; // exact basin value
        }
    }
    print_row("coherent (flat tiles)", &encode(&a, &cfg));

    // (2) Weather-like: smooth low-frequency field, neighbours nearly equal,
    //     small residuals off the nearest basin. The realistic estimate for
    //     a correlated physical field (the whole point of the thread).
    let mut b = vec![0i64; N];
    for y in 0..H {
        for x in 0..W {
            let f = 480.0
                + 400.0 * (x as f64 * 0.018).sin() * (y as f64 * 0.018).cos()
                + 60.0 * (x as f64 * 0.09).sin();
            b[y * W + x] = f.round().clamp(0.0, 1023.0) as i64;
        }
    }
    print_row("weather-like (smooth)", &encode(&b, &cfg));

    // (3) Incoherent, in codebook range: random in [0, 1024). No spatial
    //     coherence, but δ always ≤ STEP/2 so Delta covers it (no escape).
    let mut rng = Sm::new(0xC0FFEE);
    let c: Vec<i64> = (0..N).map(|_| rng.range(1024)).collect();
    print_row("incoherent in-range", &encode(&c, &cfg));

    // (4) Incoherent, OUT of codebook range: random in [0, 4096). Nearest
    //     basin is often > i8 away → Escape → worst case (should exceed 8B,
    //     the honest "codebook doesn't fit the data" failure mode).
    let mut rng2 = Sm::new(0xBADF00D);
    let d: Vec<i64> = (0..N).map(|_| rng2.range(4096)).collect();
    print_row("incoherent out-of-range", &encode(&d, &cfg));

    println!(
        "\nDense baseline = 8 B/cell (raw u64). Skip=2 Merge=3 Delta=3 Escape=6+8payload bytes.\n\
         Read: the compression is exactly as large as the data is coherent; escape only\n\
         bites when the codebook doesn't span the data (row 4)."
    );

    // λ sweep on the weather-like field — the rate-distortion curve. λ=0
    // accepts quantization-to-basin (Skip, 2B) for distortion; high λ keeps
    // it lossless (Delta, 3B). THIS is the actual "optimize" lever: how much
    // compression you buy past the lossless Delta floor for how much error.
    println!("\nλ sweep on weather-like field (rate-distortion curve — the tuning lever):");
    for (label, cfg) in [
        ("λ=0   (rate-only)", RdoConfig::RATE_ONLY),
        ("λ=1", RdoConfig::from_lambda_q8(1 << 8)),
        ("λ=4", RdoConfig::from_lambda_q8(4 << 8)),
        ("λ=16  (default)", RdoConfig::default()),
        ("λ=∞   (lossless)", RdoConfig::LOSSLESS),
    ] {
        print_row(label, &encode(&b, &cfg));
    }
}
