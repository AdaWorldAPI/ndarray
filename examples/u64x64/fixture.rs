//! D-U64X64-LLVM-1 lab fixture: tile types, the lab `Source`, the frame
//! `Witness` and `bind()`, admission-checked parameters, output types,
//! deterministic generators and the falsifier corpora.
//!
//! Geometry: `BitTile` word k is category k, bit r is row r. `ValueTile<T>` is
//! indexed `[k][r]`. No kernel-touched type allocates; generators may.
//!
//! Lab fixture only: no public type, no production caller.

/// 64 categories x 64 row bits.
pub type BitTile = [u64; 64];
/// 64 categories x 64 rows of values, indexed `[k][r]`.
pub type ValueTile<T> = [[T; 64]; 64];
/// 16-bit counters for O5.
pub type Counters = ValueTile<u16>;

/// Stock value stored in slots whose presence bit is clear.
pub const STALE: u16 = 1000;
/// Capacity `limit` code: staff-bound.
pub const STAFF: u8 = 64;
/// Capacity `limit` code: ceiling-bound.
pub const CEILING: u8 = 65;
/// Largest quantity `q`.
pub const QMAX: u32 = 128;

/// Field bits of `Source::fields`.
pub const F_STOCK: u64 = 1 << 0;
pub const F_COST: u64 = 1 << 1;
pub const F_PRESENT: u64 = 1 << 2;
pub const F_GATE: u64 = 1 << 3;
pub const F_STAFF: u64 = 1 << 4;
pub const F_ALL: u64 = F_STOCK | F_COST | F_PRESENT | F_GATE | F_STAFF;

/// Why a frame, parameter or conversion was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    Stale,
    RowOrder,
    Shape,
    Field,
    ForeignSource,
    DemandOutOfRange,
    SignificanceOutOfRange,
    Overflow { lanes: u64 },
    RecipeMismatch,
    Negative,
    TooLarge,
}

fn fnv1a(bytes: impl Iterator<Item = u8>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Hash of a row order, stable across runs.
pub fn row_order_hash(order: &[u16; 64]) -> u64 {
    fnv1a(order.iter().flat_map(|v| v.to_le_bytes()))
}

/// The identity row order.
pub fn identity_order() -> [u16; 64] {
    let mut o = [0u16; 64];
    for (i, v) in o.iter_mut().enumerate() {
        *v = i as u16;
    }
    o
}

/// Lab source: owns u16 stock and u8 cost planes, read with zero copy.
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    pub source_id: u64,
    pub version: u64,
    pub rows: usize,
    pub cols: usize,
    pub row_order: [u16; 64],
    pub fields: u64,
    pub stock: ValueTile<u16>,
    pub cost: ValueTile<u8>,
    /// P[k] bit r: slot (k, r) is known.
    pub present: BitTile,
    pub gate: u64,
    pub staff: [u8; 64],
}

impl Source {
    /// All-zero 64x64 source with every field present in the frame.
    pub fn blank(source_id: u64) -> Self {
        Source {
            source_id,
            version: 1,
            rows: 64,
            cols: 64,
            row_order: identity_order(),
            fields: F_ALL,
            stock: [[0; 64]; 64],
            cost: [[0; 64]; 64],
            present: [0; 64],
            gate: 0,
            staff: [0; 64],
        }
    }

    pub fn witness(&self) -> Witness {
        Witness {
            source_id: self.source_id,
            version: self.version,
            row_order: self.row_order,
            row_order_hash: row_order_hash(&self.row_order),
            rows: self.rows,
            cols: self.cols,
            fields: self.fields,
        }
    }

    /// Copy with a different reported shape (a 63-row source, say).
    pub fn with_shape(&self, rows: usize, cols: usize) -> Self {
        let mut s = self.clone();
        s.rows = rows;
        s.cols = cols;
        s
    }

    /// Copy with a field bit cleared from the frame.
    pub fn without_field(&self, field: u64) -> Self {
        let mut s = self.clone();
        s.fields &= !field;
        s
    }

    /// Copy with the version bumped.
    pub fn bumped(&self) -> Self {
        let mut s = self.clone();
        s.version += 1;
        s
    }

    pub fn set_present(&mut self, k: usize, r: usize, on: bool) {
        if on {
            self.present[k] |= 1u64 << r;
        } else {
            self.present[k] &= !(1u64 << r);
        }
    }

    /// Widen to the Foundry-shaped i32 source.
    pub fn to_i32(&self) -> SourceI32 {
        let mut stock = [[0i32; 64]; 64];
        for k in 0..64 {
            for r in 0..64 {
                stock[k][r] = self.stock[k][r] as i32;
            }
        }
        SourceI32 {
            source_id: self.source_id,
            version: self.version,
            rows: self.rows,
            cols: self.cols,
            row_order: self.row_order,
            fields: self.fields,
            stock,
            cost: self.cost,
            present: self.present,
            gate: self.gate,
            staff: self.staff,
        }
    }

    /// Narrow an i32 source into a lab source (E1 preparation).
    pub fn from_i32(s: &SourceI32) -> Result<Self, Refusal> {
        let stock = narrow_i32(s)?;
        Ok(Source {
            source_id: s.source_id,
            version: s.version,
            rows: s.rows,
            cols: s.cols,
            row_order: s.row_order,
            fields: s.fields,
            stock,
            cost: s.cost,
            present: s.present,
            gate: s.gate,
            staff: s.staff,
        })
    }
}

/// Foundry-shaped source: stock is `i32`.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceI32 {
    pub source_id: u64,
    pub version: u64,
    pub rows: usize,
    pub cols: usize,
    pub row_order: [u16; 64],
    pub fields: u64,
    pub stock: ValueTile<i32>,
    pub cost: ValueTile<u8>,
    pub present: BitTile,
    pub gate: u64,
    pub staff: [u8; 64],
}

/// Narrow i32 stock to u16. Refuses `Negative` first, then `TooLarge`.
pub fn narrow_i32(s: &SourceI32) -> Result<ValueTile<u16>, Refusal> {
    for k in 0..64 {
        for r in 0..64 {
            if s.stock[k][r] < 0 {
                return Err(Refusal::Negative);
            }
        }
    }
    let mut out = [[0u16; 64]; 64];
    for k in 0..64 {
        for r in 0..64 {
            let v = s.stock[k][r];
            if v > 65535 {
                return Err(Refusal::TooLarge);
            }
            out[k][r] = v as u16;
        }
    }
    Ok(out)
}

/// What a source reports about itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Witness {
    pub source_id: u64,
    pub version: u64,
    pub row_order: [u16; 64],
    pub row_order_hash: u64,
    pub rows: usize,
    pub cols: usize,
    pub fields: u64,
}

/// What a consumer expects of the frame it binds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expect {
    pub source_id: u64,
    pub version: u64,
    pub row_order: [u16; 64],
    pub rows: usize,
    pub cols: usize,
    /// Field bits the consumer will read; all must be present.
    pub fields: u64,
}

impl Expect {
    /// An expectation matching `s` exactly, requiring every field.
    pub fn of(s: &Source) -> Self {
        Expect {
            source_id: s.source_id,
            version: s.version,
            row_order: s.row_order,
            rows: 64,
            cols: 64,
            fields: F_ALL,
        }
    }
}

/// A source borrowed together with its checked witness.
#[derive(Clone, Copy)]
pub struct Bound<'a> {
    pub src: &'a Source,
    pub witness: Witness,
}

/// Check the frame before any plane is read.
/// Order: ForeignSource, Stale, Shape, RowOrder, Field.
pub fn bind<'a>(src: &'a Source, expect: &Expect) -> Result<Bound<'a>, Refusal> {
    let w = src.witness();
    if w.source_id != expect.source_id {
        return Err(Refusal::ForeignSource);
    }
    if w.version != expect.version {
        return Err(Refusal::Stale);
    }
    if w.rows != 64 || w.cols != 64 || expect.rows != 64 || expect.cols != 64 || w.rows != expect.rows
        || w.cols != expect.cols
    {
        return Err(Refusal::Shape);
    }
    if w.row_order_hash != row_order_hash(&expect.row_order) || w.row_order != expect.row_order {
        return Err(Refusal::RowOrder);
    }
    if w.fields & expect.fields != expect.fields {
        return Err(Refusal::Field);
    }
    Ok(Bound { src, witness: w })
}

/// Per-category demand `d_k <= 255`; `d_k == 0` means not demanded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Recipe {
    pub d: [u16; 64],
}

impl Recipe {
    pub fn new(d: [u16; 64]) -> Result<Self, Refusal> {
        if d.iter().any(|&v| v > 255) {
            return Err(Refusal::DemandOutOfRange);
        }
        Ok(Recipe { d })
    }

    pub fn empty() -> Self {
        Recipe { d: [0; 64] }
    }

    /// Recipe from `(category, demand)` pairs.
    pub fn from_pairs(pairs: &[(usize, u16)]) -> Result<Self, Refusal> {
        let mut d = [0u16; 64];
        for &(k, v) in pairs {
            d[k] = v;
        }
        Recipe::new(d)
    }

    /// Bit k set iff `d_k > 0`.
    pub fn demanded(&self) -> u64 {
        let mut m = 0u64;
        for k in 0..64 {
            m |= ((self.d[k] > 0) as u64) << k;
        }
        m
    }

    /// Demanded categories in ascending order, and their count.
    pub fn demanded_list(&self) -> ([u8; 64], usize) {
        let mut l = [0u8; 64];
        let mut n = 0;
        for k in 0..64 {
            if self.d[k] > 0 {
                l[n] = k as u8;
                n += 1;
            }
        }
        (l, n)
    }

    /// Identity of this recipe, compared by `Prepared` and compile-time recipes.
    pub fn id(&self) -> u64 {
        fnv1a(self.d.iter().flat_map(|v| v.to_le_bytes()))
    }
}

/// Weights for O4, `w_k` full u16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Weights {
    pub w: [u16; 64],
}

/// Query parameters with the admission bounds applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Params {
    pub recipe: Recipe,
    /// Units per staff, `1..=255`.
    pub u: u8,
    /// Quantity, `1..=128`.
    pub q: u8,
    /// Thresholds, each `0..=65536` (65536 means never).
    pub t: [u32; 64],
}

impl Params {
    /// Refuses `DemandOutOfRange` for u, q or any threshold out of range.
    pub fn new(recipe: Recipe, u: u32, q: u32, t: [u32; 64]) -> Result<Self, Refusal> {
        if !(1..=255).contains(&u) || !(1..=QMAX).contains(&q) || t.iter().any(|&v| v > 65536) {
            return Err(Refusal::DemandOutOfRange);
        }
        Ok(Params { recipe, u: u as u8, q: q as u8, t })
    }
}

/// One O5 update: add `2^b` to counter `(k, r)` for every bit r of `m`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct O5Update {
    pub k: u8,
    pub b: u8,
    pub m: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct O1Out {
    pub plane_cnt: [u8; 64],
    pub row_cnt: [u8; 64],
    pub total: u16,
}

impl O1Out {
    pub fn new() -> Self {
        O1Out { plane_cnt: [0; 64], row_cnt: [0; 64], total: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct O2Out {
    pub ge: [u64; 64],
    pub all_ge: u64,
}

impl O2Out {
    pub fn new() -> Self {
        O2Out { ge: [0; 64], all_ge: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct O3Out {
    pub min: [u16; 64],
    pub arg: [u8; 64],
    pub none: u64,
    pub unknown: u64,
}

impl O3Out {
    pub fn new() -> Self {
        O3Out { min: [0; 64], arg: [0; 64], none: 0, unknown: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct O4Out {
    pub sum: [u32; 64],
    pub total: u64,
}

impl O4Out {
    pub fn new() -> Self {
        O4Out { sum: [0; 64], total: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct O4aOut {
    pub z: ValueTile<u16>,
    pub ovf: BitTile,
}

impl O4aOut {
    pub fn new() -> Self {
        O4aOut { z: [[0; 64]; 64], ovf: [0; 64] }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowStatus {
    Denied,
    Unknown { k: u8 },
    Known,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct O6Out {
    pub status: [RowStatus; 64],
    pub cap: [u8; 64],
    /// Category k, `STAFF` or `CEILING`; meaningful only for Known rows.
    pub limit: [u8; 64],
    pub known: u64,
    pub unknown: u64,
    pub denied: u64,
}

impl O6Out {
    pub fn new() -> Self {
        O6Out {
            status: [RowStatus::Denied; 64],
            cap: [0; 64],
            limit: [0; 64],
            known: 0,
            unknown: 0,
            denied: 0,
        }
    }
}

/// xorshift64*: deterministic, no external crate.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `lo..hi` (`hi > lo`).
    pub fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + (self.next_u64() >> 20) % (hi - lo)
    }
}

/// Random source with Foundry ranges: stock 20..79 where present, `STALE`
/// where absent (about 1 in 8 slots); cost 50..249; staff 0..3; random gate.
pub fn random_source(seed: u64) -> Source {
    let mut rng = Rng::new(seed);
    let mut s = Source::blank(seed);
    for k in 0..64 {
        for r in 0..64 {
            let present = rng.range(0, 8) != 0;
            s.set_present(k, r, present);
            s.stock[k][r] = if present { rng.range(20, 80) as u16 } else { STALE };
            s.cost[k][r] = rng.range(50, 250) as u8;
        }
    }
    s.gate = rng.next_u64();
    for r in 0..64 {
        s.staff[r] = rng.range(0, 4) as u8;
    }
    s
}

pub fn random_bittile(seed: u64) -> BitTile {
    let mut rng = Rng::new(seed);
    let mut t = [0u64; 64];
    for w in t.iter_mut() {
        *w = rng.next_u64();
    }
    t
}

/// Random value tile, values `0..=max`.
pub fn random_values_u16(seed: u64, max: u16) -> ValueTile<u16> {
    let mut rng = Rng::new(seed);
    let mut t = [[0u16; 64]; 64];
    for k in 0..64 {
        for r in 0..64 {
            t[k][r] = rng.range(0, max as u64 + 1) as u16;
        }
    }
    t
}

/// Random recipe: 1..=3 demanded categories, demand 1..3.
pub fn random_recipe(seed: u64) -> Recipe {
    let mut rng = Rng::new(seed);
    let mut d = [0u16; 64];
    let n = rng.range(1, 4);
    for _ in 0..n {
        d[rng.range(0, 64) as usize] = rng.range(1, 4) as u16;
    }
    Recipe { d }
}

/// Random weights `0..=max`.
pub fn random_weights(seed: u64, max: u16) -> Weights {
    let mut rng = Rng::new(seed);
    let mut w = [0u16; 64];
    for v in w.iter_mut() {
        *v = rng.range(0, max as u64 + 1) as u16;
    }
    Weights { w }
}

/// Random O5 stream; `b` in 0..16. `sparse` ANDs three random masks.
pub fn random_updates(seed: u64, n: usize, sparse: bool) -> Vec<O5Update> {
    let mut rng = Rng::new(seed);
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        let mut m = rng.next_u64();
        if sparse {
            m &= rng.next_u64() & rng.next_u64();
        }
        v.push(O5Update { k: rng.range(0, 64) as u8, b: rng.range(0, 16) as u8, m });
    }
    v
}

// ---- corpora ----

/// Everything zero; no slot present, gate closed.
pub fn zeros() -> Source {
    Source::blank(1)
}

/// Stock 0xFFFF, cost 255, all present, gate open, staff 255.
pub fn all_ones() -> Source {
    let mut s = Source::blank(2);
    s.stock = [[0xFFFF; 64]; 64];
    s.cost = [[255; 64]; 64];
    s.present = [u64::MAX; 64];
    s.gate = u64::MAX;
    s.staff = [255; 64];
    s
}

/// Rows of the single-bit corpus.
pub const SPARSE_ROWS: [usize; 4] = [0, 31, 32, 63];
/// Categories of the single-bit corpus.
pub const SPARSE_CATS: [usize; 2] = [0, 63];

/// One present slot at `(k, r)` with stock 0x8001, cost 255, gate bit r, staff 1.
pub fn sparse(k: usize, r: usize) -> Source {
    let mut s = Source::blank(3);
    s.set_present(k, r, true);
    s.stock[k][r] = 0x8001;
    s.cost[k][r] = 255;
    s.gate = 1u64 << r;
    s.staff[r] = 1;
    s
}

/// A bit tile with the single bit `(k, r)` set.
pub fn sparse_bits(k: usize, r: usize) -> BitTile {
    let mut t = [0u64; 64];
    t[k] = 1u64 << r;
    t
}

/// All-ones pair: O1 `row_cnt` is 64 on every row, `plane_cnt` 64 on every plane.
pub fn carry_bits() -> (BitTile, BitTile) {
    ([u64::MAX; 64], [u64::MAX; 64])
}

/// Counters at `0xFFFF - 2^b` (one `2^b` update lands exactly on 0xFFFF).
pub fn carry_counters(b: u8) -> Counters {
    [[0xFFFFu16 - (1u16 << b); 64]; 64]
}

/// Stock just below 0xFFFF: `0xFFFF - ((k + r) & 3)`, all present.
pub fn carry_source() -> Source {
    let mut s = Source::blank(4);
    for k in 0..64 {
        for r in 0..64 {
            s.stock[k][r] = 0xFFFF - ((k + r) & 3) as u16;
        }
    }
    s.present = [u64::MAX; 64];
    s.gate = u64::MAX;
    s
}

/// O4a operands: x near 0xFFFF, y in 0..=4, so about half carry out.
pub fn carry_pair() -> (ValueTile<u16>, ValueTile<u16>) {
    let mut x = [[0u16; 64]; 64];
    let mut y = [[0u16; 64]; 64];
    for k in 0..64 {
        for r in 0..64 {
            x[k][r] = 0xFFFF - ((k + r) & 3) as u16;
            y[k][r] = ((k * 3 + r) & 7) as u16 / 2;
        }
    }
    (x, y)
}

/// O4a edge: 0xFFFF + 1 everywhere.
pub fn overflow_pair() -> (ValueTile<u16>, ValueTile<u16>) {
    ([[0xFFFF; 64]; 64], [[1; 64]; 64])
}

/// O4 edge: w = 65535 (pair with `all_ones`, cost 255).
pub fn overflow_weights() -> Weights {
    Weights { w: [65535; 64] }
}

/// O5 edge: counters at 0xFFFF; any update on a set lane overflows.
pub fn saturated_counters() -> Counters {
    [[0xFFFF; 64]; 64]
}

/// Stock 0x8000..0xFFF8 stepping 8, all present: orders unsigned values.
pub fn high_values() -> Source {
    let mut s = Source::blank(5);
    for k in 0..64 {
        for r in 0..64 {
            s.stock[k][r] = 0x8000 + ((k * 64 + r) * 8) as u16;
        }
    }
    s.present = [u64::MAX; 64];
    s.gate = u64::MAX;
    s
}

/// i32 source with one negative stock value.
pub fn i32_negative() -> SourceI32 {
    let mut s = random_source(6).to_i32();
    s.stock[2][5] = -1;
    s
}

/// i32 source with one stock value 65536.
pub fn i32_too_large() -> SourceI32 {
    let mut s = random_source(7).to_i32();
    s.stock[9][33] = 65536;
    s
}

/// Foundry "store 09": row 9 has demanded category 17 unknown (STALE stock);
/// every other demanded slot is present and ample; gate open everywhere.
pub struct Store09 {
    pub src: Source,
    pub recipe: Recipe,
    pub row: usize,
    pub category: usize,
}

pub fn store_09() -> Store09 {
    let mut s = Source::blank(9);
    for k in 0..64 {
        for r in 0..64 {
            s.stock[k][r] = 70;
            s.cost[k][r] = 100;
        }
    }
    s.present = [u64::MAX; 64];
    s.gate = u64::MAX;
    s.staff = [3; 64];
    s.set_present(17, 9, false);
    s.stock[17][9] = STALE;
    let recipe = Recipe::from_pairs(&[(3, 1), (17, 2), (40, 1)]).unwrap();
    Store09 { src: s, recipe, row: 9, category: 17 }
}

/// Unknown slots only in a non-demanded category (5): the recipe ignores them.
pub fn unknown_non_demanded() -> (Source, Recipe) {
    let mut s = store_09().src;
    s.set_present(17, 9, true);
    s.stock[17][9] = 70;
    for r in [0usize, 9, 40] {
        s.set_present(5, r, false);
        s.stock[5][r] = STALE;
    }
    (s, Recipe::from_pairs(&[(3, 1), (17, 2), (40, 1)]).unwrap())
}

/// Equal stock across categories in every row: O3 `arg` is 0.
pub fn ties_source() -> Source {
    let mut s = Source::blank(10);
    s.stock = [[40; 64]; 64];
    s.present = [u64::MAX; 64];
    s.gate = u64::MAX;
    s.staff = [3; 64];
    s
}

/// A capacity case; every row carries the same data.
pub struct LimitCase {
    pub name: &'static str,
    pub src: Source,
    pub recipe: Recipe,
    pub u: u32,
    pub cap: u8,
    pub limit: u8,
}

fn uniform(stock0: u16, staff: u8) -> Source {
    let mut s = Source::blank(11);
    s.stock = [[stock0; 64]; 64];
    s.cost = [[100; 64]; 64];
    s.present = [u64::MAX; 64];
    s.gate = u64::MAX;
    s.staff = [staff; 64];
    s
}

/// Ties and limits, hand-derived: category beats staff on a tie; staff is
/// reported when it alone is binding; CEILING only when nothing else equals 128.
pub fn limit_cases() -> Vec<LimitCase> {
    let r1 = |k: usize, d: u16| Recipe::from_pairs(&[(k, d)]).unwrap();
    vec![
        LimitCase { name: "category-ties-staff", src: uniform(20, 2), recipe: r1(0, 2), u: 5, cap: 10, limit: 0 },
        LimitCase { name: "staff-binds", src: uniform(200, 3), recipe: r1(0, 1), u: 10, cap: 30, limit: STAFF },
        LimitCase { name: "ceiling-binds", src: uniform(1000, 255), recipe: r1(0, 1), u: 255, cap: 128, limit: CEILING },
        LimitCase { name: "category-equals-ceiling", src: uniform(128, 255), recipe: r1(0, 1), u: 255, cap: 128, limit: 0 },
    ]
}

/// Thresholds `{0, s-1, s, s+1, 65535, 65536}` around stock value `s`.
pub fn threshold_set(s: u16) -> [u32; 6] {
    [0, (s as u32).saturating_sub(1), s as u32, s as u32 + 1, 65535, 65536]
}

/// Every category's threshold set to `t`.
pub fn uniform_thresholds(t: u32) -> [u32; 64] {
    [t; 64]
}

/// Capacity: stock below demand everywhere, so cap is 0 on Known rows.
pub fn stock_below_demand() -> (Source, Recipe) {
    (uniform(3, 3), Recipe::from_pairs(&[(0, 5)]).unwrap())
}

/// Capacity: empty D, so cap is `min(staff * u, 128)`.
pub fn empty_demand() -> (Source, Recipe) {
    (uniform(50, 3), Recipe::empty())
}

// ---- row permutation ----

/// Random permutation of 0..64 (Fisher-Yates); `perm[i]` is the old row of new row i.
pub fn permutation(seed: u64) -> [u16; 64] {
    let mut rng = Rng::new(seed);
    let mut p = identity_order();
    for i in (1..64).rev() {
        let j = rng.range(0, i as u64 + 1) as usize;
        p.swap(i, j);
    }
    p
}

/// Permute the bits of one word: new bit i = old bit `perm[i]`.
pub fn permute_bits(w: u64, perm: &[u16; 64]) -> u64 {
    let mut o = 0u64;
    for i in 0..64 {
        o |= ((w >> perm[i]) & 1) << i;
    }
    o
}

/// Permute a per-row array: new[i] = old[perm[i]].
pub fn permute_rows<T: Copy>(a: &[T; 64], perm: &[u16; 64]) -> [T; 64] {
    core::array::from_fn(|i| a[perm[i] as usize])
}

/// Permute every row-indexed part of the source and record the order.
pub fn permuted(s: &Source, perm: &[u16; 64]) -> Source {
    let mut o = s.clone();
    for k in 0..64 {
        o.stock[k] = permute_rows(&s.stock[k], perm);
        o.cost[k] = permute_rows(&s.cost[k], perm);
        o.present[k] = permute_bits(s.present[k], perm);
    }
    o.gate = permute_bits(s.gate, perm);
    o.staff = permute_rows(&s.staff, perm);
    o.row_order = *perm;
    o
}
