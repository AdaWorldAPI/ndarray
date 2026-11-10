//! D-U64X64-LLVM-1 lab driver: entry table, exactness check, benchmark, single
//! kernel runner (for callgrind) and type sizes. Lab fixture only.
//!
//! Run: `cargo run --release --example u64x64 --features std -- <mode>`.
#![allow(dead_code)]
#![allow(clippy::needless_range_loop)]

mod fixture;
mod harness;
mod oracle;
mod route_a;
mod route_c;
mod route_d;
// Later routes are appended here by the orchestrator: `mod route_b;` ...

use fixture::*;
use std::cell::RefCell;
use std::fmt::Debug;
use std::hint::black_box as bb;

#[global_allocator]
static ALLOC: harness::Counting = harness::Counting;

pub enum Kernel {
    O1(fn(&BitTile, &BitTile, u64, &mut O1Out)),
    O2(fn(&Bound<'_>, &[u32; 64], u64, &mut O2Out)),
    O3(fn(&Bound<'_>, u64, &mut O3Out)),
    O4(fn(&ValueTile<u8>, &[u16; 64], u64, &mut O4Out)),
    O4a(fn(&ValueTile<u16>, &ValueTile<u16>, &mut O4aOut)),
    O5(fn(&mut Counters, O5Update) -> Result<(), Refusal>),
    O6(fn(&Bound<'_>, &Params, &mut O6Out)),
}

pub struct Entry {
    pub op: &'static str,
    pub route: &'static str,
    pub variant: &'static str,
    pub class: harness::Class,
    pub kernel: Kernel,
}

fn all_entries() -> Vec<Entry> {
    let mut v = route_a::entries();
    v.extend(route_c::entries());
    v.extend(route_d::entries());
    // Later routes: v.extend(route_b::entries()); ...
    v.shrink_to_fit();
    v
}

/// The compile-time recipe an `O6c` entry is specialised for: it may only
/// be run with that recipe (its wrapper asserts so).
fn const_recipe(e: &Entry) -> Option<Recipe> {
    if e.op != "O6c" {
        return None;
    }
    let d = match e.variant {
        "const_sparse" => <route_d::Sparse3 as route_d::Recipe>::D,
        "const_dense" => <route_d::Dense as route_d::Recipe>::D,
        v => panic!("unknown O6c variant {v}"),
    };
    Some(Recipe::new(d).expect("const recipe within the admission bounds"))
}

/// The entry's params: the bench params, with the recipe replaced by the
/// entry's compile-time recipe for an `O6c` entry.
fn params_for(e: &Entry, p: &Params) -> Params {
    match const_recipe(e) {
        Some(r) => Params::new(r, u32::from(p.u), u32::from(p.q), p.t).expect("params"),
        None => *p,
    }
}

/// Every named and 256 random sources under one fixed recipe, for `O6c`.
fn o6c_cases(r: Recipe, mut cb: impl FnMut(String, &Bound<'_>, &Params)) {
    let mut sources = named_sources();
    for seed in 0..256u64 {
        sources.push((format!("random{seed}"), random_source(seed)));
    }
    for (n, s) in sources.iter() {
        let b = bound_ok(s);
        for &u in [1u32, 3, 255].iter() {
            let p = Params::new(r, u, 1, [0; 64]).expect("params");
            cb(format!("{n}/u{u}"), &b, &p);
        }
    }
}

fn symbol(e: &Entry) -> String {
    if e.variant.is_empty() {
        format!("u64x64_{}_{}", e.op, e.route)
    } else {
        format!("u64x64_{}_{}_{}", e.op, e.route, e.variant)
    }
}

fn abort(e: &Entry, case: &str, detail: &str) -> ! {
    println!(
        "MISMATCH entry {} (op {} route {} variant '{}') case {}: {}",
        symbol(e),
        e.op,
        e.route,
        e.variant,
        case,
        detail
    );
    std::process::exit(1);
}

// ---------------------------------------------------------------- diff helpers

fn d_val<T: PartialEq + Debug>(name: &str, g: T, w: T) -> Option<String> {
    if g != w {
        Some(format!("{name}: got {g:?} want {w:?}"))
    } else {
        None
    }
}

fn d_arr<T: PartialEq + Debug>(name: &str, g: &[T], w: &[T]) -> Option<String> {
    if g.len() != w.len() {
        return Some(format!("{name}: length got {} want {}", g.len(), w.len()));
    }
    for i in 0..g.len() {
        if g[i] != w[i] {
            return Some(format!("{name}[{i}]: got {:?} want {:?}", g[i], w[i]));
        }
    }
    None
}

fn d_tile<T: PartialEq + Debug + Copy>(
    name: &str,
    g: &ValueTile<T>,
    w: &ValueTile<T>,
) -> Option<String> {
    for k in 0..64 {
        for r in 0..64 {
            if g[k][r] != w[k][r] {
                return Some(format!("{name}[{k}][{r}]: got {:?} want {:?}", g[k][r], w[k][r]));
            }
        }
    }
    None
}

fn same_o1(g: &O1Out, w: &O1Out) -> Option<String> {
    d_arr("plane_cnt", &g.plane_cnt, &w.plane_cnt)
        .or_else(|| d_arr("row_cnt", &g.row_cnt, &w.row_cnt))
        .or_else(|| d_val("total", g.total, w.total))
}

fn same_o2(g: &O2Out, w: &O2Out) -> Option<String> {
    d_arr("ge", &g.ge, &w.ge).or_else(|| d_val("all_ge", g.all_ge, w.all_ge))
}

fn same_o3(g: &O3Out, w: &O3Out) -> Option<String> {
    d_arr("min", &g.min, &w.min)
        .or_else(|| d_arr("arg", &g.arg, &w.arg))
        .or_else(|| d_val("none", g.none, w.none))
        .or_else(|| d_val("unknown", g.unknown, w.unknown))
}

fn same_o4(g: &O4Out, w: &O4Out) -> Option<String> {
    d_arr("sum", &g.sum, &w.sum).or_else(|| d_val("total", g.total, w.total))
}

fn same_o4a(g: &O4aOut, w: &O4aOut) -> Option<String> {
    d_tile("z", &g.z, &w.z).or_else(|| d_arr("ovf", &g.ovf, &w.ovf))
}

/// `cap` and `limit` are meaningful only for Known rows, so they are compared
/// only where the oracle says the row is Known.
fn same_o6(g: &O6Out, w: &O6Out) -> Option<String> {
    if let Some(d) = d_arr("status", &g.status, &w.status)
        .or_else(|| d_val("known", g.known, w.known))
        .or_else(|| d_val("unknown", g.unknown, w.unknown))
        .or_else(|| d_val("denied", g.denied, w.denied))
    {
        return Some(d);
    }
    for r in 0..64 {
        if (w.known >> r) & 1 == 1 {
            if g.cap[r] != w.cap[r] {
                return Some(format!("cap[{r}]: got {} want {}", g.cap[r], w.cap[r]));
            }
            if g.limit[r] != w.limit[r] {
                return Some(format!("limit[{r}]: got {} want {}", g.limit[r], w.limit[r]));
            }
        }
    }
    None
}

// -------------------------------------------------------------- output poison

fn poison_o1() -> O1Out {
    O1Out { plane_cnt: [0xA5; 64], row_cnt: [0xA5; 64], total: 0xA5A5 }
}
fn poison_o2() -> O2Out {
    O2Out { ge: [0xA5A5_A5A5_A5A5_A5A5; 64], all_ge: 0xA5A5_A5A5_A5A5_A5A5 }
}
fn poison_o3() -> O3Out {
    O3Out {
        min: [0xA5A5; 64],
        arg: [0xA5; 64],
        none: 0xA5A5_A5A5_A5A5_A5A5,
        unknown: 0xA5A5_A5A5_A5A5_A5A5,
    }
}
fn poison_o4() -> O4Out {
    O4Out { sum: [0xA5A5_A5A5; 64], total: 0xA5A5_A5A5_A5A5_A5A5 }
}
fn poison_o4a() -> O4aOut {
    O4aOut { z: [[0xA5A5; 64]; 64], ovf: [0xA5A5_A5A5_A5A5_A5A5; 64] }
}
fn poison_o6() -> O6Out {
    O6Out {
        status: [RowStatus::Known; 64],
        cap: [0xA5; 64],
        limit: [0xA5; 64],
        known: 0xA5A5_A5A5_A5A5_A5A5,
        unknown: 0xA5A5_A5A5_A5A5_A5A5,
        denied: 0xA5A5_A5A5_A5A5_A5A5,
    }
}

// ----------------------------------------------------------- per-case runners

fn run_o1(f: fn(&BitTile, &BitTile, u64, &mut O1Out), a: &BitTile, b: &BitTile, live: u64) -> Option<String> {
    let mut g = poison_o1();
    let mut w = O1Out::new();
    f(a, b, live, &mut g);
    oracle::o1(a, b, live, &mut w);
    same_o1(&g, &w)
}

fn run_o2(f: fn(&Bound<'_>, &[u32; 64], u64, &mut O2Out), b: &Bound<'_>, t: &[u32; 64], d: u64) -> Option<String> {
    let mut g = poison_o2();
    let mut w = O2Out::new();
    f(b, t, d, &mut g);
    oracle::o2(b, t, d, &mut w);
    same_o2(&g, &w)
}

fn run_o3(f: fn(&Bound<'_>, u64, &mut O3Out), b: &Bound<'_>, d: u64) -> Option<String> {
    let mut g = poison_o3();
    let mut w = O3Out::new();
    f(b, d, &mut g);
    oracle::o3(b, d, &mut w);
    same_o3(&g, &w)
}

fn run_o4(f: fn(&ValueTile<u8>, &[u16; 64], u64, &mut O4Out), c: &ValueTile<u8>, wt: &[u16; 64], live: u64) -> Option<String> {
    let mut g = poison_o4();
    let mut w = O4Out::new();
    f(c, wt, live, &mut g);
    oracle::o4(c, wt, live, &mut w);
    same_o4(&g, &w)
}

fn run_o4a(f: fn(&ValueTile<u16>, &ValueTile<u16>, &mut O4aOut), x: &ValueTile<u16>, y: &ValueTile<u16>) -> Option<String> {
    let mut g = Box::new(poison_o4a());
    let mut w = Box::new(O4aOut::new());
    f(x, y, &mut g);
    oracle::o4a(x, y, &mut w);
    same_o4a(&g, &w)
}

fn run_o5(f: fn(&mut Counters, O5Update) -> Result<(), Refusal>, init: &Counters, ups: &[O5Update]) -> Option<String> {
    let mut g = Box::new(*init);
    let mut w = Box::new(*init);
    for (i, u) in ups.iter().enumerate() {
        let rg = f(&mut g, *u);
        let rw = oracle::o5(&mut w, *u);
        if rg != rw {
            return Some(format!("update {i} {u:?}: result got {rg:?} want {rw:?}"));
        }
        if let Some(d) = d_tile("counters", &g, &w) {
            return Some(format!("update {i} {u:?}: {d}"));
        }
    }
    None
}

fn run_o6(f: fn(&Bound<'_>, &Params, &mut O6Out), b: &Bound<'_>, p: &Params) -> Option<String> {
    let mut g = poison_o6();
    let mut w = O6Out::new();
    f(b, p, &mut g);
    oracle::o6(b, p, &mut w);
    same_o6(&g, &w)
}

// ---------------------------------------------------------------- corpora

fn bound_ok(s: &Source) -> Bound<'_> {
    bind(s, &Expect::of(s)).expect("corpus source must bind")
}

fn named_sources() -> Vec<(String, Source)> {
    let mut v: Vec<(String, Source)> = vec![
        ("zeros".into(), zeros()),
        ("all_ones".into(), all_ones()),
        ("carry_source".into(), carry_source()),
        ("high_values".into(), high_values()),
        ("store_09".into(), store_09().src),
        ("unknown_non_demanded".into(), unknown_non_demanded().0),
        ("ties".into(), ties_source()),
        ("stock_below_demand".into(), stock_below_demand().0),
        ("empty_demand".into(), empty_demand().0),
        ("permuted".into(), permuted(&random_source(77), &permutation(5))),
    ];
    for &k in SPARSE_CATS.iter() {
        for &r in SPARSE_ROWS.iter() {
            v.push((format!("sparse[{k}][{r}]"), sparse(k, r)));
        }
    }
    for c in limit_cases() {
        v.push((format!("limit:{}", c.name), c.src));
    }
    v
}

const DEMAND_MASKS: [u64; 8] = [
    u64::MAX,
    0,
    1,
    1u64 << 63,
    0x8000_0000_0000_0001,
    0x0000_0000_FFFF_0000,
    (1 << 3) | (1 << 17) | (1 << 40),
    (1 << 5) | (1 << 17),
];

const LIVE_MASKS: [u64; 5] = [u64::MAX, 0, 1, 1u64 << 63, 0xFFFF_FFFF_0000_0000];

const SEEDS: u64 = 256;

fn bit_pairs() -> Vec<(String, BitTile, BitTile)> {
    let mut v = vec![
        ("zeros".to_string(), [0u64; 64], [0u64; 64]),
        ("ones".to_string(), [u64::MAX; 64], [u64::MAX; 64]),
        ("ones_vs_zeros".to_string(), [u64::MAX; 64], [0u64; 64]),
    ];
    let (ca, cb) = carry_bits();
    v.push(("carry_bits".to_string(), ca, cb));
    for &k in SPARSE_CATS.iter() {
        for &r in SPARSE_ROWS.iter() {
            v.push((format!("sparse[{k}][{r}]"), sparse_bits(k, r), sparse_bits(k, r)));
            v.push((format!("sparse_vs_ones[{k}][{r}]"), sparse_bits(k, r), [u64::MAX; 64]));
        }
    }
    v
}

fn o1_cases(mut cb: impl FnMut(String, &BitTile, &BitTile, u64)) {
    for (n, a, b) in bit_pairs() {
        for (i, &l) in LIVE_MASKS.iter().enumerate() {
            cb(format!("{n}/live{i}"), &a, &b, l);
        }
    }
    for s in 0..SEEDS {
        let a = random_bittile(s * 2 + 1);
        let b = random_bittile(s * 2 + 2);
        let live = Rng::new(s + 500).next_u64();
        cb(format!("random{s}"), &a, &b, live);
    }
}

fn o2_cases(mut cb: impl FnMut(String, &Bound<'_>, &[u32; 64], u64)) {
    for (n, s) in named_sources() {
        let b = bound_ok(&s);
        let st = s.stock[0][0];
        let mut ts: Vec<[u32; 64]> =
            threshold_set(st).iter().map(|&t| uniform_thresholds(t)).collect();
        ts.push(core::array::from_fn(|k| s.stock[k][(k * 7) % 64] as u32));
        for (ti, t) in ts.iter().enumerate() {
            for (di, &d) in DEMAND_MASKS.iter().enumerate() {
                cb(format!("{n}/t{ti}/d{di}"), &b, t, d);
            }
        }
    }
    for seed in 0..SEEDS {
        let s = random_source(seed + 1000);
        let b = bound_ok(&s);
        let mut rng = Rng::new(seed + 2000);
        let t: [u32; 64] = core::array::from_fn(|_| {
            if rng.range(0, 16) == 0 { 65536 } else { rng.range(15, 85) as u32 }
        });
        cb(format!("random{seed}/recipe"), &b, &t, random_recipe(seed).demanded());
        cb(format!("random{seed}/rand"), &b, &t, rng.next_u64());
    }
}

fn o3_cases(mut cb: impl FnMut(String, &Bound<'_>, u64)) {
    for (n, s) in named_sources() {
        let b = bound_ok(&s);
        for (di, &d) in DEMAND_MASKS.iter().enumerate() {
            cb(format!("{n}/d{di}"), &b, d);
        }
    }
    for seed in 0..SEEDS {
        let s = random_source(seed + 3000);
        let b = bound_ok(&s);
        cb(format!("random{seed}/recipe"), &b, random_recipe(seed).demanded());
        cb(format!("random{seed}/rand"), &b, Rng::new(seed + 4000).next_u64());
    }
}

fn o4_cases(mut cb: impl FnMut(String, &ValueTile<u8>, &[u16; 64], u64)) {
    for (n, s) in named_sources() {
        let ws = [
            ("rand65535", random_weights(11, 65535)),
            ("rand600", random_weights(12, 600)),
            ("overflow", overflow_weights()),
            ("zero", Weights { w: [0; 64] }),
        ];
        for (wn, w) in ws.iter() {
            for (li, &l) in [u64::MAX, 0, 0x8000_0000_0000_0001].iter().enumerate() {
                cb(format!("{n}/{wn}/live{li}"), &s.cost, &w.w, l);
            }
        }
    }
    for seed in 0..SEEDS {
        let s = random_source(seed + 5000);
        let w = random_weights(seed + 6000, if seed % 2 == 0 { 65535 } else { 600 });
        let live = Rng::new(seed + 7000).next_u64();
        cb(format!("random{seed}"), &s.cost, &w.w, live);
    }
}

fn o4a_cases(mut cb: impl FnMut(String, &ValueTile<u16>, &ValueTile<u16>)) {
    let (cx, cy) = carry_pair();
    cb("carry_pair".into(), &cx, &cy);
    let (ox, oy) = overflow_pair();
    cb("overflow_pair".into(), &ox, &oy);
    let z = [[0u16; 64]; 64];
    cb("zeros".into(), &z, &z);
    for seed in 0..SEEDS {
        let max = if seed % 2 == 0 { 0xFFFF } else { 0x8000 };
        let x = random_values_u16(seed * 2 + 1, max);
        let y = random_values_u16(seed * 2 + 2, max);
        cb(format!("random{seed}"), &x, &y);
    }
}

fn o5_cases(mut cb: impl FnMut(String, &Counters, &[O5Update])) {
    let zero: Counters = [[0u16; 64]; 64];
    for seed in 1..=3u64 {
        cb(format!("dense{seed}"), &zero, &random_updates(seed, 4096, false));
        cb(format!("sparse{seed}"), &zero, &random_updates(seed + 10, 4096, true));
    }
    for b in 0..16u8 {
        cb(format!("carry_counters{b}"), &carry_counters(b), &random_updates(100 + b as u64, 256, false));
    }
    cb("saturated".into(), &saturated_counters(), &random_updates(200, 64, false));
    let bad = [
        O5Update { k: 0, b: 16, m: 1 },
        O5Update { k: 63, b: 255, m: u64::MAX },
        O5Update { k: 1, b: 15, m: 0 },
        O5Update { k: 2, b: 15, m: u64::MAX },
        O5Update { k: 2, b: 15, m: u64::MAX },
    ];
    cb("bad_significance".into(), &zero, &bad);
}

fn big_recipe(seed: u64) -> Recipe {
    let mut rng = Rng::new(seed);
    let mut d = [0u16; 64];
    for _ in 0..rng.range(1, 5) {
        d[rng.range(0, 64) as usize] = rng.range(1, 256) as u16;
    }
    Recipe { d }
}

fn o6_cases(mut cb: impl FnMut(String, &Bound<'_>, &Params)) {
    let mut combos: Vec<(String, Source, Recipe, u32)> = Vec::new();
    let st = store_09();
    combos.push(("store_09".into(), st.src, st.recipe, 3));
    let (s, r) = unknown_non_demanded();
    combos.push(("unknown_non_demanded".into(), s, r, 3));
    for c in limit_cases() {
        combos.push((format!("limit:{}", c.name), c.src, c.recipe, c.u));
    }
    let (s, r) = stock_below_demand();
    combos.push(("stock_below_demand".into(), s, r, 3));
    let (s, r) = empty_demand();
    combos.push(("empty_demand".into(), s, r, 3));
    for (n, s, r, u) in combos.iter() {
        let b = bound_ok(s);
        let p = Params::new(*r, *u, 1, [0; 64]).expect("params");
        cb(n.clone(), &b, &p);
    }
    let recipes = [
        ("r_random1", random_recipe(1)),
        ("r_cat0", Recipe::from_pairs(&[(0, 1)]).unwrap()),
        ("r_cat63", Recipe::from_pairs(&[(63, 255)]).unwrap()),
        ("r_empty", Recipe::empty()),
    ];
    for (n, s) in named_sources() {
        let b = bound_ok(&s);
        for (rn, r) in recipes.iter() {
            for &u in [1u32, 3, 255].iter() {
                let p = Params::new(*r, u, 1, [0; 64]).expect("params");
                cb(format!("{n}/{rn}/u{u}"), &b, &p);
            }
        }
    }
    for seed in 0..SEEDS {
        let s = random_source(seed + 8000);
        let b = bound_ok(&s);
        let mut rng = Rng::new(seed + 9000);
        let u = rng.range(1, 256) as u32;
        let p1 = Params::new(random_recipe(seed * 3 + 1), u, 1, [0; 64]).expect("params");
        cb(format!("random{seed}/small"), &b, &p1);
        let p2 = Params::new(big_recipe(seed * 3 + 2), u, 1, [0; 64]).expect("params");
        cb(format!("random{seed}/big"), &b, &p2);
    }
}

// ------------------------------------------------------------------ check

fn check() -> i32 {
    println!("{}", harness::tier_line());
    let entries = all_entries();
    let mut cases = 0usize;
    for e in &entries {
        match &e.kernel {
            Kernel::O1(f) => o1_cases(|n, a, b, l| {
                cases += 1;
                if let Some(d) = run_o1(*f, a, b, l) {
                    abort(e, &n, &d)
                }
            }),
            Kernel::O2(f) => o2_cases(|n, b, t, d| {
                cases += 1;
                if let Some(x) = run_o2(*f, b, t, d) {
                    abort(e, &n, &x)
                }
            }),
            Kernel::O3(f) => o3_cases(|n, b, d| {
                cases += 1;
                if let Some(x) = run_o3(*f, b, d) {
                    abort(e, &n, &x)
                }
            }),
            Kernel::O4(f) => o4_cases(|n, c, w, l| {
                cases += 1;
                if let Some(x) = run_o4(*f, c, w, l) {
                    abort(e, &n, &x)
                }
            }),
            Kernel::O4a(f) => o4a_cases(|n, x, y| {
                cases += 1;
                if let Some(d) = run_o4a(*f, x, y) {
                    abort(e, &n, &d)
                }
            }),
            Kernel::O5(f) => o5_cases(|n, init, ups| {
                cases += 1;
                if let Some(d) = run_o5(*f, init, ups) {
                    abort(e, &n, &d)
                }
            }),
            Kernel::O6(f) if const_recipe(e).is_some() => {
                o6c_cases(const_recipe(e).expect("checked"), |n, b, p| {
                    cases += 1;
                    if let Some(d) = run_o6(*f, b, p) {
                        abort(e, &n, &d)
                    }
                })
            }
            Kernel::O6(f) => o6_cases(|n, b, p| {
                cases += 1;
                if let Some(d) = run_o6(*f, b, p) {
                    abort(e, &n, &d)
                }
            }),
        }
    }
    println!("check: {} entries, {} cases, all exact", entries.len(), cases);
    0
}

// ------------------------------------------------------------ bench inputs

struct Inputs {
    a: BitTile,
    b: BitTile,
    live: u64,
    src: Source,
    t: [u32; 64],
    dem: u64,
    w: [u16; 64],
    x: ValueTile<u16>,
    y: ValueTile<u16>,
    params: Params,
    upd: O5Update,
    verify_stream: Vec<O5Update>,
}

fn make_inputs() -> Box<Inputs> {
    let src = random_source(1);
    let recipe = random_recipe(1);
    Box::new(Inputs {
        a: random_bittile(1),
        b: random_bittile(2),
        live: u64::MAX,
        src,
        t: uniform_thresholds(50),
        dem: recipe.demanded(),
        w: random_weights(1, 600).w,
        x: random_values_u16(1, 40000),
        y: random_values_u16(2, 40000),
        params: Params::new(recipe, 3, 2, [0; 64]).expect("params"),
        upd: O5Update { k: 5, b: 0, m: u64::MAX },
        verify_stream: random_updates(1, 256, false),
    })
}

struct Cfg {
    reps: usize,
    samples: usize,
    sweep: Option<harness::Sweep>,
}

impl Cfg {
    /// Untimed allocation check, then the timed measurement. `reset` runs
    /// untimed before each sample (together with the cache sweep).
    fn time<F: FnMut(), R: FnMut()>(&mut self, label: &str, mut f: F, mut reset: R) -> harness::Stats {
        harness::assert_no_alloc(label, || f());
        reset();
        let sweep = &mut self.sweep;
        harness::measure(
            self.reps,
            self.samples,
            || {
                reset();
                if let Some(s) = sweep.as_mut() {
                    bb(s.run());
                }
            },
            f,
        )
    }
}

fn bench_entry(e: &Entry, inp: &Inputs, bound: &Bound<'_>, cfg: &mut Cfg) -> harness::Stats {
    let label = symbol(e);
    match &e.kernel {
        Kernel::O1(f) => {
            if let Some(d) = run_o1(*f, &inp.a, &inp.b, inp.live) {
                abort(e, "bench-verify", &d)
            }
            let mut out = O1Out::new();
            cfg.time(
                &label,
                || {
                    f(bb(&inp.a), bb(&inp.b), bb(inp.live), &mut out);
                    bb(&out);
                },
                || {},
            )
        }
        Kernel::O2(f) => {
            if let Some(d) = run_o2(*f, bound, &inp.t, inp.dem) {
                abort(e, "bench-verify", &d)
            }
            let mut out = O2Out::new();
            cfg.time(
                &label,
                || {
                    f(bb(bound), bb(&inp.t), bb(inp.dem), &mut out);
                    bb(&out);
                },
                || {},
            )
        }
        Kernel::O3(f) => {
            if let Some(d) = run_o3(*f, bound, inp.dem) {
                abort(e, "bench-verify", &d)
            }
            let mut out = O3Out::new();
            cfg.time(
                &label,
                || {
                    f(bb(bound), bb(inp.dem), &mut out);
                    bb(&out);
                },
                || {},
            )
        }
        Kernel::O4(f) => {
            if let Some(d) = run_o4(*f, &inp.src.cost, &inp.w, inp.live) {
                abort(e, "bench-verify", &d)
            }
            let mut out = O4Out::new();
            cfg.time(
                &label,
                || {
                    f(bb(&inp.src.cost), bb(&inp.w), bb(inp.live), &mut out);
                    bb(&out);
                },
                || {},
            )
        }
        Kernel::O4a(f) => {
            if let Some(d) = run_o4a(*f, &inp.x, &inp.y) {
                abort(e, "bench-verify", &d)
            }
            let mut out = Box::new(O4aOut::new());
            cfg.time(
                &label,
                || {
                    f(bb(&inp.x), bb(&inp.y), &mut out);
                    bb(&out);
                },
                || {},
            )
        }
        Kernel::O5(f) => {
            let zero: Counters = [[0u16; 64]; 64];
            if let Some(d) = run_o5(*f, &zero, &inp.verify_stream) {
                abort(e, "bench-verify", &d)
            }
            let counters = RefCell::new(Box::new(zero));
            cfg.time(
                &label,
                || {
                    let r = f(&mut counters.borrow_mut(), bb(inp.upd));
                    let _ = bb(r);
                },
                || {
                    **counters.borrow_mut() = [[0u16; 64]; 64];
                },
            )
        }
        Kernel::O6(f) => {
            let params = params_for(e, &inp.params);
            if let Some(d) = run_o6(*f, bound, &params) {
                abort(e, "bench-verify", &d)
            }
            let mut out = O6Out::new();
            cfg.time(
                &label,
                || {
                    f(bb(bound), bb(&params), &mut out);
                    bb(&out);
                },
                || {},
            )
        }
    }
}

fn bench(scenario: &str) -> i32 {
    let (reps, samples, sweep) = match scenario {
        "hot" => (1000, 31, None),
        "l2" => (1, 41, Some(harness::Sweep::new(harness::L2_SWEEP))),
        "llc" => (1, 11, Some(harness::Sweep::new(harness::LLC_SWEEP))),
        _ => {
            println!("unknown scenario '{scenario}' (hot | l2 | llc)");
            return 2;
        }
    };
    println!("{}", harness::tier_line());
    let overhead = harness::timer_overhead(31);
    println!("timer_overhead_ns median {:.3} min {:.3} max {:.3}", overhead.median, overhead.min, overhead.max);
    println!("scenario {scenario} reps {reps} samples {samples}");
    println!("op route variant class median_ns min_ns max_ns ns_per_row low_res");
    let inp = make_inputs();
    let bound = bound_ok(&inp.src);
    let mut cfg = Cfg { reps, samples, sweep };
    for e in &all_entries() {
        let st = bench_entry(e, &inp, &bound, &mut cfg);
        let variant = if e.variant.is_empty() { "-" } else { e.variant };
        println!(
            "{} {} {} {:?} {:.1} {:.1} {:.1} {:.3} {}",
            e.op,
            e.route,
            variant,
            e.class,
            st.median,
            st.min,
            st.max,
            st.median / 64.0,
            if harness::low_res(st.median, overhead.median) { "LOW-RES" } else { "ok" }
        );
    }
    // The rest of the hot path: binding the frame (witness + checks) and
    // building O6's parameters. A kernel's cost in production is one bind
    // plus its kernel; the parameters are built once per query.
    let expect = Expect::of(&inp.src);
    let recipe = inp.params.recipe;
    let fixed: [(&str, harness::Stats); 2] = [
        (
            "BIND",
            cfg.time(
                "bind",
                || {
                    let b = bind(bb(&inp.src), bb(&expect));
                    let _ = bb(b);
                },
                || {},
            ),
        ),
        (
            "PREP",
            cfg.time(
                "params",
                || {
                    let p = Params::new(bb(recipe), bb(3), bb(2), bb([0u32; 64]));
                    let _ = bb(p);
                },
                || {},
            ),
        ),
    ];
    for (name, st) in fixed {
        println!(
            "{name} - - Fixed {:.1} {:.1} {:.1} {:.3} {}",
            st.median,
            st.min,
            st.max,
            st.median / 64.0,
            if harness::low_res(st.median, overhead.median) { "LOW-RES" } else { "ok" }
        );
    }
    0
}

// -------------------------------------------------------------------- one

fn one(sym: &str, n: usize) -> i32 {
    let entries = all_entries();
    let Some(e) = entries.iter().find(|e| symbol(e) == sym) else {
        println!("no entry with symbol '{sym}'; available:");
        for e in &entries {
            println!("  {}", symbol(e));
        }
        return 2;
    };
    let inp = make_inputs();
    let bound = bound_ok(&inp.src);
    match &e.kernel {
        Kernel::O1(f) => {
            let mut out = O1Out::new();
            for _ in 0..n {
                f(bb(&inp.a), bb(&inp.b), bb(inp.live), &mut out);
                bb(&out);
            }
        }
        Kernel::O2(f) => {
            let mut out = O2Out::new();
            for _ in 0..n {
                f(bb(&bound), bb(&inp.t), bb(inp.dem), &mut out);
                bb(&out);
            }
        }
        Kernel::O3(f) => {
            let mut out = O3Out::new();
            for _ in 0..n {
                f(bb(&bound), bb(inp.dem), &mut out);
                bb(&out);
            }
        }
        Kernel::O4(f) => {
            let mut out = O4Out::new();
            for _ in 0..n {
                f(bb(&inp.src.cost), bb(&inp.w), bb(inp.live), &mut out);
                bb(&out);
            }
        }
        Kernel::O4a(f) => {
            let mut out = Box::new(O4aOut::new());
            for _ in 0..n {
                f(bb(&inp.x), bb(&inp.y), &mut out);
                bb(&out);
            }
        }
        Kernel::O5(f) => {
            let mut c: Box<Counters> = Box::new([[0u16; 64]; 64]);
            for _ in 0..n {
                let _ = bb(f(&mut c, bb(inp.upd)));
            }
            bb(&c);
        }
        Kernel::O6(f) => {
            let params = params_for(e, &inp.params);
            let mut out = O6Out::new();
            for _ in 0..n {
                f(bb(&bound), bb(&params), &mut out);
                bb(&out);
            }
        }
    }
    0
}

// ------------------------------------------------------------------ sizes

fn sizes() -> i32 {
    use std::mem::size_of;
    macro_rules! sz {
        ($($t:ty),* $(,)?) => { $( println!("{:<28} {}", stringify!($t), size_of::<$t>()); )* };
    }
    sz!(
        BitTile,
        ValueTile<u8>,
        ValueTile<u16>,
        Counters,
        Source,
        SourceI32,
        Witness,
        Expect,
        Bound<'static>,
        Recipe,
        Weights,
        Params,
        O5Update,
        O1Out,
        O2Out,
        O3Out,
        O4Out,
        O4aOut,
        RowStatus,
        O6Out,
        Refusal,
    );
    0
}

fn usage() -> i32 {
    println!("usage: u64x64 <mode>");
    println!("  check                  exactness of every entry vs the oracle");
    println!("  bench <hot|l2|llc>     timing per entry");
    println!("  one <symbol> <n>       run one entry n times (callgrind)");
    println!("  sizes                  size_of fixture types");
    2
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let code = match args.get(1).map(String::as_str) {
        Some("check") => check(),
        Some("bench") => match args.get(2) {
            Some(s) => bench(s),
            None => usage(),
        },
        Some("one") => match (args.get(2), args.get(3).and_then(|s| s.parse::<usize>().ok())) {
            (Some(sym), Some(n)) => one(sym, n),
            _ => usage(),
        },
        Some("sizes") => sizes(),
        _ => usage(),
    };
    std::process::exit(code);
}
