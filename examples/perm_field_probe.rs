//! One question over the whole 64×64 permutation field, answered by one fold.
//!
//! The expression is `f = ternlog(0x96)` (a XOR b XOR c) over a population of
//! `N` 64-byte blocks. Each question is asked of the WHOLE field — all 4096
//! cells of a `PermTable12` — and has one answer. Two executions:
//!
//! - **per-cell**: 4096 standalone folds. Each composes its cell's map,
//!   shuffles the operands into that coordinate system, evaluates, and adds its
//!   answer to the total.
//! - **hot fold**: the data and the accumulator stay resident, and the field
//!   streams through the one running fold as relations. Nothing is constructed
//!   per cell. A shared basis disappears under an invariant terminal. A
//!   relative basis or an output-coordinate mask stays in the fold through the
//!   field's exposure (a 64×64 kernel built from the two 64-entry tables),
//!   never as per-cell work.
//!
//! Ordered Keep is the one terminal that owes per-cell output. It is included
//! to show where the per-cell cost genuinely remains.
//!
//! Every hot-fold answer is asserted equal to the per-cell answer. The
//! counters count what this probe itself does: `evals` is the number of
//! expression/predicate evaluations inside the fold, which is the incremental
//! work that remains.
//!
//! ```sh
//! cargo run --release --example perm_field_probe
//! ```

use ndarray::hpc::perm::{LaneOp, LaneTernlog, Perm64, PermBatch, PermTable12, Schedule};
use std::time::Instant;

const N: usize = 64;
const CELLS: u16 = 4096;
type Block = [u8; 64];

#[derive(Default, Clone, Copy)]
struct Counters {
    compositions: u64,
    shuffles: u64,
    evals: u64,
    bytes_written: u64,
}

struct Run<'a> {
    t: &'a PermTable12,
    c: Counters,
}

impl Run<'_> {
    fn compose(&mut self, code: u16) -> Perm64 {
        self.c.compositions += 1;
        self.t.for_code(code)
    }
    fn shuffle(&mut self, p: Perm64, b: &Block) -> Block {
        self.c.shuffles += 1;
        self.c.bytes_written += 64;
        p.materialize(b)
    }
    fn f(&mut self, a: &Block, b: &Block, c: &Block) -> Block {
        self.c.evals += 64;
        LaneTernlog(0x96).apply([a, b, c])
    }
}

fn nonzero(v: &Block) -> u64 {
    v.iter().filter(|&&x| x != 0).count() as u64
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn shuffled(seed: u64) -> Perm64 {
    let mut s = seed;
    let mut idx: [u8; 64] = core::array::from_fn(|i| i as u8);
    for i in (1..64).rev() {
        idx.swap(i, (splitmix(&mut s) % (i as u64 + 1)) as usize);
    }
    Perm64::from_indices(idx).unwrap()
}

/// Sparse-ish bytes, so Count and Any are not saturated.
fn population(seed: u64) -> Vec<Block> {
    let mut s = seed;
    (0..N)
        .map(|_| {
            core::array::from_fn(|_| {
                if splitmix(&mut s).is_multiple_of(3) {
                    (splitmix(&mut s) & 0xFF) as u8
                } else {
                    0
                }
            })
        })
        .collect()
}

struct Row {
    name: &'static str,
    c: Counters,
    micros: f64,
}

fn run<T>(t: &PermTable12, name: &'static str, body: impl FnOnce(&mut Run) -> T) -> (T, Row) {
    let mut r = Run {
        t,
        c: Counters::default(),
    };
    let start = Instant::now();
    let out = body(&mut r);
    let micros = start.elapsed().as_secs_f64() * 1e6;
    (out, Row { name, c: r.c, micros })
}

fn report(case: &str, answer: &str, rows: &[Row]) {
    println!("\n## {case}\nanswer: {answer} (all executions equal, asserted)");
    println!(
        "{:<10} {:>12} {:>10} {:>12} {:>13} {:>11}",
        "execution", "compositions", "shuffles", "evals", "bytes_written", "µs"
    );
    for r in rows {
        println!(
            "{:<10} {:>12} {:>10} {:>12} {:>13} {:>11.1}",
            r.name, r.c.compositions, r.c.shuffles, r.c.evals, r.c.bytes_written, r.micros
        );
    }
}

fn main() {
    let table = PermTable12::new(core::array::from_fn(|i| shuffled(100 + i as u64)));
    let (a, b, c) = (population(1), population(2), population(3));
    println!("field: 4096 cells; population: {N} blocks per operand");

    // ---- 1. Shared basis, Count over the field ---------------------------
    // Σ over cells of Count(f(P·a, P·b, P·c)).
    {
        let (per_cell, r1) = run(&table, "per-cell", |r| {
            let mut total = 0u64;
            for code in 0..CELLS {
                let p = r.compose(code);
                for i in 0..N {
                    let (pa, pb, pc) = (r.shuffle(p, &a[i]), r.shuffle(p, &b[i]), r.shuffle(p, &c[i]));
                    total += nonzero(&r.f(&pa, &pb, &pc));
                }
            }
            total
        });
        let (hot, r2) = run(&table, "hot fold", |r| {
            // Equivariance moves P outside f, and Count removes it: every cell
            // contributes the same amount. One pass, then scale.
            let once: u64 = (0..N).map(|i| nonzero(&r.f(&a[i], &b[i], &c[i]))).sum();
            once * CELLS as u64
        });
        assert_eq!(hot, per_cell);
        assert_eq!(r2.c.compositions + r2.c.shuffles, 0);
        report("shared basis, field-wide Count", &hot.to_string(), &[r1, r2]);
    }

    // ---- 2. Shared basis, Any over the field (stops at the first hit) -----
    {
        let (per_cell, r1) = run(&table, "per-cell", |r| {
            for code in 0..CELLS {
                let p = r.compose(code);
                for i in 0..N {
                    let (pa, pb, pc) = (r.shuffle(p, &a[i]), r.shuffle(p, &b[i]), r.shuffle(p, &c[i]));
                    if nonzero(&r.f(&pa, &pb, &pc)) > 0 {
                        return true;
                    }
                }
            }
            false
        });
        let (hot, r2) = run(&table, "hot fold", |r| (0..N).any(|i| nonzero(&r.f(&a[i], &b[i], &c[i])) > 0));
        assert_eq!(hot, per_cell);
        report("shared basis, field-wide Any (early exit)", &hot.to_string(), &[r1, r2]);
    }

    // ---- 3. Different bases, Count over the field -------------------------
    // a is seen through P(cell); b and c through one fixed Q. The relation
    // R = Q relative to P changes with the cell and survives Count, so it
    // stays in the fold as the re-based field's exposure.
    {
        let q = table.for_code(0x5A5);
        let (per_cell, r1) = run(&table, "per-cell", |r| {
            let mut total = 0u64;
            for code in 0..CELLS {
                let p = r.compose(code);
                for i in 0..N {
                    let (pa, qb, qc) = (r.shuffle(p, &a[i]), r.shuffle(q, &b[i]), r.shuffle(q, &c[i]));
                    total += nonzero(&r.f(&pa, &qb, &qc));
                }
            }
            total
        });
        let (hot, r2) = run(&table, "hot fold", |r| {
            let e = r.t.relative_field(q).exposure();
            r.c.compositions += 128; // building the two re-based 64-entry tables
            let mut total = 0u64;
            for i in 0..N {
                let (ab, bb, cb) = (&a[i], &b[i], &c[i]);
                let mut evals = 0u64;
                total += e.count_where(|lane, src| {
                    evals += 1;
                    ab[lane] ^ bb[src] ^ cb[src] != 0
                });
                r.c.evals += evals;
            }
            total
        });
        assert_eq!(hot, per_cell);
        report("different bases (P per cell, fixed Q), field-wide Count", &hot.to_string(), &[r1, r2]);
    }

    // ---- 4. Mask fixed in OUTPUT coordinates, Count over the field ---------
    // Σ over cells of Count(M ∧ P·f(a,b,c)). M does not move with P; it takes
    // part in the fold through the exposure instead of per-cell transport.
    {
        let masks: Vec<u64> = {
            let mut s = 77;
            (0..N).map(|_| splitmix(&mut s)).collect()
        };
        let (per_cell, r1) = run(&table, "per-cell", |r| {
            let mut total = 0u64;
            for code in 0..CELLS {
                let p = r.compose(code);
                for i in 0..N {
                    let (pa, pb, pc) = (r.shuffle(p, &a[i]), r.shuffle(p, &b[i]), r.shuffle(p, &c[i]));
                    let v = r.f(&pa, &pb, &pc);
                    total += (0..64)
                        .filter(|&l| masks[i] >> l & 1 == 1 && v[l] != 0)
                        .count() as u64;
                }
            }
            total
        });
        let (hot, r2) = run(&table, "hot fold", |r| {
            let e = r.t.exposure();
            let mut total = 0u64;
            for i in 0..N {
                let v = r.f(&a[i], &b[i], &c[i]);
                let m = masks[i];
                let mut evals = 0u64;
                total += e.count_where(|lane, src| {
                    evals += 1;
                    m >> lane & 1 == 1 && v[src] != 0
                });
                r.c.evals += evals;
            }
            total
        });
        assert_eq!(hot, per_cell);
        report("mask fixed in output coordinates, field-wide Count", &hot.to_string(), &[r1, r2]);
    }

    // ---- 5. Ordered Keep: the one terminal that owes per-cell output ------
    {
        let (per_cell, r1) = run(&table, "per-cell", |r| {
            let mut out = Vec::with_capacity(CELLS as usize * N);
            for code in 0..CELLS {
                let p = r.compose(code);
                for i in 0..N {
                    let (pa, pb, pc) = (r.shuffle(p, &a[i]), r.shuffle(p, &b[i]), r.shuffle(p, &c[i]));
                    let v = r.f(&pa, &pb, &pc);
                    out.push(v);
                    r.c.bytes_written += 64;
                }
            }
            out
        });
        let (hot, r2) = run(&table, "hot fold", |r| {
            // Evaluate once in the shared basis; each cell is owed only its
            // coordinates of the result.
            let v: Vec<Block> = (0..N).map(|i| r.f(&a[i], &b[i], &c[i])).collect();
            let mut out = Vec::with_capacity(CELLS as usize * N);
            for code in 0..CELLS {
                let p = r.compose(code);
                for blk in &v {
                    out.push(r.shuffle(p, blk));
                }
            }
            out
        });
        assert_eq!(hot, per_cell);
        report("shared basis, ordered Keep (per-cell output owed)", "4096 × N blocks", &[r1, r2]);
    }

    // ---- 6. Duplicate-heavy request stream, ordered Keep ------------------
    {
        let mut s = 5;
        let stream: Vec<u16> = (0..20_000)
            .map(|_| (splitmix(&mut s) % 300) as u16 * 13 % CELLS)
            .collect();
        let src = LaneTernlog(0x96).apply([&a[0], &b[0], &c[0]]);
        let (per_request, r1) = run(&table, "per-cell", |r| {
            stream
                .iter()
                .map(|&code| {
                    let p = r.compose(code);
                    r.shuffle(p, &src)
                })
                .collect::<Vec<Block>>()
        });
        let (dedup, r2) = run(&table, "dedup", |r| {
            let composed = PermBatch::new(&stream, Schedule::Deduplicate).compose(r.t);
            r.c.compositions += composed.compositions() as u64;
            let mut out = vec![[0u8; 64]; stream.len()];
            composed.materialize_into(&src, &mut out);
            r.c.shuffles += out.len() as u64;
            r.c.bytes_written += out.len() as u64 * 64;
            out
        });
        assert_eq!(dedup, per_request, "multiplicity and order must survive deduplication");
        let distinct = PermBatch::new(&stream, Schedule::Deduplicate)
            .field()
            .unwrap()
            .len();
        report(
            &format!("{} requests over {distinct} distinct cells, ordered Keep", stream.len()),
            "one block per request, in request order",
            &[r1, r2],
        );
    }
}
