//! D-GTM-0m — successive soaking: how much of a NEW corpus is expressible in a
//! behavioural vocabulary already minted from earlier ones?
//!
//! This replaces the address-prefix framing that D-GTM-0l falsified. The claim
//! now under test is not that semantically related things get nearby addresses,
//! but that *machine behaviour converges onto a small reusable basis*: soak the
//! codebook on corpus 1, feed corpus 2 through it without resetting, and measure
//! reuse against new mint. If reuse rises while the codebook grows sublinearly,
//! the substrate is a sponge. If the codebook grows with the corpus, it is a
//! dictionary of one entry per thing seen, which is not a basis.
//!
//! ## The vacuity trap this probe is built around
//!
//! Reuse is trivially ~100% at a coarse enough granularity — there are only a
//! dozen distinct p-code opcodes, so "opcode reuse" saturates on the second
//! basic block of the first corpus and says nothing. So the probe reports a
//! LADDER of granularities and a null, and a result only counts if reuse stays
//! high at a granularity whose vocabulary is still growing.
//!
//! | rung | atom | what it claims |
//! |---|---|---|
//! | G0 | opcode | control — must saturate; proves nothing |
//! | G1 | (kind, opcode) | still near-trivial |
//! | G2 | block opcode sequence | the behavioural BPE token |
//! | G3 | block (opcode, in-arity, out-arity) sequence | shape-sensitive |
//! | G4 | function block-token sequence | whole-routine shape |
//!
//! The null shuffles opcodes across blocks while preserving every block's LENGTH
//! and the corpus-wide opcode marginal. If measured reuse at G2/G3 is no better
//! than the null's, the "behavioural basis" is just the opcode frequency
//! distribution reappearing, not structure.
//!
//! Usage: `cargo run --release --example behavioral_soak_probe -- <ore.tsv>...`
//! Corpora are soaked in argument order and the codebook is never reset.
//! Ore is read by path and never vendored.

use std::collections::{BTreeMap, HashMap, HashSet};

/// One lifted instruction fact, reduced to what the atoms are built from.
struct Fact {
    block: (String, String, String),
    id: u64,
    kind: String,
    opcode: String,
    ins: u32,
    outs: u32,
}

/// SplitMix64 — the null's shuffle must be reproducible.
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

fn parse(path: &str) -> Vec<Fact> {
    let text = std::fs::read_to_string(path).expect("read ore");
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 11 {
            continue;
        }
        let Ok(id) = f[2].parse::<u64>() else { continue };
        out.push(Fact {
            block: (f[0].to_string(), f[1].to_string(), f[10].to_string()),
            id,
            kind: f[5].to_string(),
            opcode: f[6].to_string(),
            ins: u32::from(f[5] == "OperandIn"),
            outs: u32::from(f[5] == "OperandOut"),
        });
    }
    out
}

/// Group facts into blocks, each block a fact list ordered by fact id.
fn blocks(facts: &[Fact]) -> Vec<Vec<&Fact>> {
    let mut m: BTreeMap<&(String, String, String), Vec<&Fact>> = BTreeMap::new();
    for f in facts {
        m.entry(&f.block).or_default().push(f);
    }
    let mut v: Vec<Vec<&Fact>> = m.into_values().collect();
    for b in &mut v {
        b.sort_by_key(|f| f.id);
    }
    v
}

/// The five rungs of the granularity ladder, as atom occurrence lists.
fn atoms(bs: &[Vec<&Fact>], rung: usize) -> Vec<String> {
    match rung {
        0 => bs.iter().flatten().map(|f| f.opcode.clone()).collect(),
        1 => bs
            .iter()
            .flatten()
            .map(|f| format!("{}|{}", f.kind, f.opcode))
            .collect(),
        2 => bs
            .iter()
            .map(|b| {
                b.iter()
                    .filter(|f| f.kind == "Op")
                    .map(|f| f.opcode.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .filter(|s| !s.is_empty())
            .collect(),
        3 => bs
            .iter()
            .map(|b| {
                // per-instruction shape: opcode plus how many values it consumed/produced
                let mut per: BTreeMap<&str, (u32, u32)> = BTreeMap::new();
                for f in b {
                    let e = per.entry(f.opcode.as_str()).or_default();
                    e.0 += f.ins;
                    e.1 += f.outs;
                }
                per.iter()
                    .map(|(o, (i, u))| format!("{o}:{i}:{u}"))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .filter(|s| !s.is_empty())
            .collect(),
        _ => {
            // whole-function shape: the sequence of its blocks' G2 tokens
            let mut by_fn: BTreeMap<(&str, &str), Vec<String>> = BTreeMap::new();
            for b in bs {
                let tok = b
                    .iter()
                    .filter(|f| f.kind == "Op")
                    .map(|f| f.opcode.as_str())
                    .collect::<Vec<_>>()
                    .join(",");
                if tok.is_empty() {
                    continue;
                }
                by_fn
                    .entry((b[0].block.0.as_str(), b[0].block.1.as_str()))
                    .or_default()
                    .push(tok);
            }
            by_fn.into_values().map(|v| v.join(";")).collect()
        }
    }
}

/// Soak `corpora` in order through one never-reset codebook; report the curve.
fn soak(names: &[String], per_corpus: &[Vec<String>], label: &str) {
    println!("\n{label}");
    println!("  corpus                    | occurrences | reuse% | new atoms | codebook | bits/occ");
    let mut book: HashMap<String, usize> = HashMap::new();
    for (name, occ) in names.iter().zip(per_corpus) {
        let before = book.len();
        let mut hits = 0usize;
        for a in occ {
            if book.contains_key(a) {
                hits += 1;
            } else {
                let n = book.len();
                book.insert(a.clone(), n);
            }
        }
        let minted = book.len() - before;
        // cost of transmitting this corpus given the standing codebook: an index
        // per occurrence, plus a literal description for each newly minted atom.
        let idx_bits = if book.len() > 1 {
            (book.len() as f64).log2()
        } else {
            0.0
        };
        let new_bits: f64 = occ
            .iter()
            .collect::<HashSet<_>>()
            .iter()
            .filter(|a| book[**a] >= before)
            .map(|a| a.len() as f64 * 8.0)
            .sum();
        let total = occ.len() as f64 * idx_bits + new_bits;
        println!(
            "  {:<25} | {:11} | {:5.1}% | {:9} | {:8} | {:8.2}",
            name,
            occ.len(),
            100.0 * hits as f64 / occ.len().max(1) as f64,
            minted,
            book.len(),
            total / occ.len().max(1) as f64
        );
    }
}

/// Reuse stratified by atom LENGTH — the control that decides whether transfer
/// is real. A one-op block is trivially shared by every architecture ever, so a
/// headline reuse figure dominated by short blocks says nothing. This reports,
/// for the corpora soaked AFTER the first, what fraction of blocks of each
/// length were already in the codebook, and how many blocks that is.
fn stratified(names: &[String], per_corpus: &[Vec<String>], first_n: usize) {
    println!("\nG2 reuse stratified by block length (transfer corpora only)");
    let mut book: HashSet<String> = HashSet::new();
    for (i, occ) in per_corpus.iter().enumerate() {
        if i >= first_n {
            let mut hit: BTreeMap<usize, (usize, usize)> = BTreeMap::new();
            for a in occ {
                let len = a.split(',').count();
                let bucket = if len >= 8 { 8 } else { len };
                let e = hit.entry(bucket).or_default();
                e.1 += 1;
                if book.contains(a) {
                    e.0 += 1;
                }
            }
            println!("  {}", names[i]);
            for (len, (h, t)) in &hit {
                let label = if *len == 8 { "8+".to_string() } else { len.to_string() };
                println!(
                    "    len {:>2} ops | {:5} blocks | {:5.1}% already known",
                    label,
                    t,
                    100.0 * *h as f64 / *t as f64
                );
            }
        }
        for a in occ {
            book.insert(a.clone());
        }
    }
}

/// The composition test — the one exact-match cannot answer.
///
/// A long block failing to match a soaked block EXACTLY does not mean it has no
/// reusable structure; it means one-token-per-block is the wrong atom for it.
/// That is what byte-pair encoding exists to fix: cover a long sequence by
/// concatenating short known tokens. So build the codebook from every contiguous
/// opcode n-gram (n = 1..=MAX_TOKEN) seen in the SOAK corpora, then greedily
/// longest-match-tokenize each TRANSFER block against it.
///
/// The metric cannot be coverage: length-1 tokens make coverage trivially 100%.
/// What matters is (a) ops per token — how much of the sequence a single known
/// token accounts for — and (b) the share of ops covered by tokens of length ≥ 2,
/// which is the share that is genuinely structural rather than per-opcode.
fn composition(label: &str, names: &[String], per_block_ops: &[Vec<Vec<String>>], first_n: usize) {
    const MAX_TOKEN: usize = 6;
    let mut book: HashSet<Vec<String>> = HashSet::new();
    for corpus in per_block_ops.iter().take(first_n) {
        for b in corpus {
            for n in 1..=MAX_TOKEN.min(b.len()) {
                for w in b.windows(n) {
                    book.insert(w.to_vec());
                }
            }
        }
    }
    // Saturation control: with a ~12-symbol opcode alphabet a codebook can cover
    // every short n-gram by combinatorics alone, at which case a high ops/token
    // is arithmetic, not transfer. Report what fraction of the possible n-grams
    // over the observed alphabet the codebook actually holds.
    let alphabet: HashSet<&String> = per_block_ops
        .iter()
        .take(first_n)
        .flatten()
        .flatten()
        .collect();
    let a = alphabet.len();
    // Alphabet-saturation control. With a small opcode alphabet, an n-gram
    // codebook can cover almost any sequence by combinatorics alone, which is
    // the boring explanation this workspace has already caught once: a 7-symbol
    // alphabet made held-out coverage look tautological, and made a degree-6
    // neighbourhood the complete graph minus self. So report what fraction of
    // the POSSIBLE short n-grams the codebook holds; if that is near 1, high
    // composition coverage is arithmetic rather than structure.
    let alphabet: HashSet<&String> = per_block_ops
        .iter()
        .take(first_n)
        .flatten()
        .flatten()
        .collect();
    let a = alphabet.len();
    println!("\n{label} — {}-token codebook (n-grams n<={MAX_TOKEN})", book.len());
    print!("  alphabet {a} symbols; codebook holds");
    for n in 1..=MAX_TOKEN {
        let possible = (a as f64).powi(n as i32);
        let held = book.iter().filter(|t| t.len() == n).count();
        print!(" {n}-gram {:.1}%", 100.0 * held as f64 / possible);
    }
    println!("  <- near 100% would make coverage arithmetic, not structure");
    println!(
        "  corpus                    | blocks | ops   | tokens | ops/token | ops in len>=2 tokens | unseen opcodes"
    );
    for (i, corpus) in per_block_ops.iter().enumerate() {
        if i < first_n {
            continue;
        }
        let (mut ops, mut toks, mut structural, mut unseen) = (0usize, 0usize, 0usize, 0usize);
        for b in corpus {
            let mut p = 0usize;
            while p < b.len() {
                let mut best = 0usize;
                for n in (1..=MAX_TOKEN.min(b.len() - p)).rev() {
                    if book.contains(&b[p..p + n]) {
                        best = n;
                        break;
                    }
                }
                if best == 0 {
                    // a genuinely novel opcode: no token of any length covers it
                    unseen += 1;
                    best = 1;
                } else if best >= 2 {
                    structural += best;
                }
                toks += 1;
                p += best;
            }
            ops += b.len();
        }
        println!(
            "  {:<25} | {:6} | {:5} | {:6} | {:9.2} | {:19.1}% | {:14}",
            names[i].rsplit('/').next().unwrap_or(&names[i]),
            corpus.len(),
            ops,
            toks,
            ops as f64 / toks.max(1) as f64,
            100.0 * structural as f64 / ops.max(1) as f64,
            unseen
        );
    }
}

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: behavioral_soak_probe <ore.tsv>...");
        std::process::exit(2);
    }

    let mut names = Vec::new();
    let mut all: Vec<Vec<Fact>> = Vec::new();
    for p in &paths {
        let facts = parse(p);
        // one ore file may hold several images; soak each image separately so a
        // second image inside one file still counts as a second corpus.
        let mut by_img: BTreeMap<String, Vec<Fact>> = BTreeMap::new();
        for f in facts {
            by_img.entry(f.block.0.clone()).or_default().push(f);
        }
        for (img, fs) in by_img {
            names.push(img);
            all.push(fs);
        }
    }
    println!("corpora soaked in order: {}", names.join(" -> "));
    for (n, f) in names.iter().zip(&all) {
        let b = blocks(f);
        println!("  {n}: {} facts, {} blocks", f.len(), b.len());
    }

    for rung in 0..5 {
        let label = [
            "G0 opcode (control)",
            "G1 kind|opcode",
            "G2 block opcode sequence",
            "G3 block opcode:arity shape",
            "G4 function block-token sequence",
        ][rung];
        let per: Vec<Vec<String>> = all.iter().map(|f| atoms(&blocks(f), rung)).collect();
        soak(&names, &per, label);
    }

    // The length control: transfer restricted to blocks long enough to mean
    // something. `first_n` is the count of corpora treated as the soak.
    let g2: Vec<Vec<String>> = all.iter().map(|f| atoms(&blocks(f), 2)).collect();
    let first_n: usize = std::env::var("SOAK_FIRST_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    stratified(&names, &g2, first_n);

    let per_block_ops: Vec<Vec<Vec<String>>> = all
        .iter()
        .map(|f| {
            blocks(f)
                .iter()
                .map(|b| {
                    b.iter()
                        .filter(|x| x.kind == "Op")
                        .map(|x| x.opcode.clone())
                        .collect::<Vec<_>>()
                })
                .filter(|b: &Vec<String>| !b.is_empty())
                .collect()
        })
        .collect();
    composition("BPE-style composition, REAL soak", &names, &per_block_ops, first_n);

    // The null that decides whether composition coverage is structure. Shuffle
    // the SOAK corpora's opcodes so every block keeps its length and the corpus
    // keeps its opcode marginal exactly, but co-occurrence order is destroyed;
    // mint the codebook from that and tokenize the REAL transfer corpora. If
    // coverage and ops/token survive, the result was the opcode distribution
    // reappearing, not a behavioural basis.
    let mut nrng = SplitMix(0x5EED_1234_5678_9ABC);
    let mut shuffled = per_block_ops.clone();
    for corpus in shuffled.iter_mut().take(first_n) {
        let mut pool: Vec<String> = corpus.iter().flatten().cloned().collect();
        for i in (1..pool.len()).rev() {
            pool.swap(i, (nrng.next() % (i as u64 + 1)) as usize);
        }
        let mut cur = 0usize;
        for b in corpus.iter_mut() {
            let n = b.len();
            b.clone_from_slice(&pool[cur..cur + n]);
            cur += n;
        }
    }
    composition("COMPOSITION NULL, shuffled soak", &names, &shuffled, first_n);

    // Null: preserve every block's length and the corpus-wide opcode marginal,
    // destroy which opcodes co-occur in a block. Reported at G2, the rung whose
    // whole claim is that co-occurrence is structured.
    let mut rng = SplitMix(0x9E37_79B9_7F4A_7C15);
    let mut null_per: Vec<Vec<String>> = Vec::new();
    for f in &all {
        let bs = blocks(f);
        let mut pool: Vec<String> = bs
            .iter()
            .flatten()
            .filter(|x| x.kind == "Op")
            .map(|x| x.opcode.clone())
            .collect();
        for i in (1..pool.len()).rev() {
            pool.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let mut cur = 0usize;
        let mut toks = Vec::new();
        for b in &bs {
            let len = b.iter().filter(|x| x.kind == "Op").count();
            if len == 0 {
                continue;
            }
            toks.push(pool[cur..cur + len].join(","));
            cur += len;
        }
        null_per.push(toks);
    }
    soak(&names, &null_per, "G2 NULL — block lengths and opcode marginal kept, co-occurrence destroyed");
}
