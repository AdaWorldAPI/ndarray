//! D-GTM-0l — does packed-prefix routing express real long-range transitions?
//!
//! The hypothesis under test (operator, grey/white-matter framing): *"white matter
//! should not be another data structure. It should be an interpretation of packed
//! location prefixes."* Concretely: a tract is `(prefix, mask, learned transition)`,
//! and a transition is routed by `ADDRESS & PREFIX_MASK == PREFIX`. That is only a
//! routing mechanism if the tract codebook is much smaller than the edge set it
//! covers. If one tract is needed per edge, the "interpretation" IS an edge list.
//!
//! Falsifier: measure, on a real long-range relation set, (a) whether edges are more
//! prefix-local than a degree-preserving null, (b) how far the tract codebook
//! compresses the edge set at each prefix width, and (c) whether the non-local
//! residual is concentrated (a small exception table keeps the mechanism) or diffuse
//! (it does not).
//!
//! Input is passed by path and never vendored: the corpus this was measured on is a
//! disassembly harvest that its own provenance file forbids redistributing. The probe
//! reads derived facts only, and the repository stores the measurement, not the ore.
//!
//! Usage: `cargo run --release --example prefix_tract_coverage_probe -- <ore.tsv>`
//!
//! Ore schema (tab-separated, `#`-prefixed header lines skipped): field 1 = image,
//! field 4 = the 128-bit packed address as 32 hex nibbles, field 6 = fact kind,
//! field 8 = the call target as a decimal address. `CallSite` rows are the edges.

use std::collections::{BTreeMap, BTreeSet};

/// Address nibbles 9..12 of the packed key hold the 16-bit program counter,
/// little-endian byte order — `..361d..` is `0x1d36`.
fn unpack_pc(hex: &str) -> Option<u32> {
    if hex.len() < 12 {
        return None;
    }
    let lo = u32::from_str_radix(&hex[8..10], 16).ok()?;
    let hi = u32::from_str_radix(&hex[10..12], 16).ok()?;
    Some((hi << 8) | lo)
}

/// The leading `k` nibbles of a 16-bit address — the `PREFIX_MASK` reading.
fn prefix(addr: u32, k: u32) -> u32 {
    addr >> (4 * (4 - k))
}

/// Longest shared prefix, in nibbles, of two addresses.
fn shared(a: u32, b: u32) -> u32 {
    (0..=4)
        .rev()
        .find(|&k| prefix(a, k) == prefix(b, k))
        .unwrap_or(0)
}

/// SplitMix64 — deterministic shuffling so the null is reproducible.
struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn main() {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: prefix_tract_coverage_probe <ore.tsv>");
            std::process::exit(2);
        }
    };
    let text = std::fs::read_to_string(&path).expect("read ore");

    let mut edges: Vec<(String, u32, u32)> = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 9 || f[5] != "CallSite" {
            continue;
        }
        let (Some(src), Ok(dst)) = (unpack_pc(f[3]), f[7].parse::<u32>()) else {
            continue;
        };
        edges.push((f[0].to_string(), src, dst));
    }
    let n = edges.len();
    assert!(n > 0, "no CallSite edges parsed — schema drift?");
    println!("long-range edges: {n}");

    // (a) locality vs a degree-preserving null: same sources, destinations reshuffled.
    let mut real = [0usize; 5];
    for (_, s, d) in &edges {
        real[shared(*s, *d) as usize] += 1;
    }
    let mut null = [0usize; 5];
    let mut dsts: Vec<u32> = edges.iter().map(|e| e.2).collect();
    let mut rng = SplitMix(0x9E37_79B9_7F4A_7C15);
    const ROUNDS: usize = 50;
    for _ in 0..ROUNDS {
        for i in (1..dsts.len()).rev() {
            dsts.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        for (e, d) in edges.iter().zip(&dsts) {
            null[shared(e.1, *d) as usize] += 1;
        }
    }
    println!("\nshared prefix | real   | null    (k nibbles; higher = more local)");
    for k in 0..5 {
        println!(
            "  k={k}        | {:6.2}% | {:6.2}%",
            100.0 * real[k] as f64 / n as f64,
            100.0 * null[k] as f64 / (ROUNDS * n) as f64
        );
    }

    // (b) does the tract codebook actually compress the edge set?
    let uniq: BTreeSet<_> = edges.iter().cloned().collect();
    println!("\nk | tracts | edges | compression");
    for k in 1..=4 {
        let tracts: BTreeSet<_> = edges
            .iter()
            .map(|(b, s, d)| (b.clone(), prefix(*s, k), prefix(*d, k)))
            .collect();
        println!("{k} | {:6} | {:5} | {:8.2}x", tracts.len(), uniq.len(), uniq.len() as f64 / tracts.len() as f64);
    }

    // (c) is the non-local residual a small exception table, or diffuse?
    let far: Vec<_> = edges
        .iter()
        .filter(|(_, s, d)| shared(*s, *d) == 0)
        .collect();
    let mut hist: BTreeMap<u32, usize> = BTreeMap::new();
    for (_, _, d) in &far {
        *hist.entry(*d).or_default() += 1;
    }
    let tot = far.len() as f64;
    let mut counts: Vec<usize> = hist.values().copied().collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));
    let entropy: f64 = counts
        .iter()
        .map(|&c| {
            let p = c as f64 / tot;
            -p * p.log2()
        })
        .sum();
    println!("\nnon-local edges {} over {} distinct targets", far.len(), hist.len());
    let mut cum = 0usize;
    for (i, c) in counts.iter().enumerate() {
        cum += c;
        if matches!(i + 1, 1 | 5 | 10 | 20 | 40) {
            println!("  top {:3} targets cover {:5.1}%", i + 1, 100.0 * cum as f64 / tot);
        }
    }
    println!(
        "  target entropy {entropy:.2} bits vs {:.2} uniform — a small hub set would sit far below",
        (hist.len() as f64).log2()
    );
}
