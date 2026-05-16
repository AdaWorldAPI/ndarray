# SoA Kernel Architecture: The Transposed Cascade

> **The insight**: Don't iterate candidates and probe words.
> Iterate words and filter candidates.
>
> The database layout IS the algorithm.

---

## The Current Architecture (AoS — Row-Major)

```
Memory layout: [candidate₀ word₀..word₂₅₅] [candidate₁ word₀..word₂₅₅] ...

Algorithm (kernel_pipeline, line 725):
    for each candidate:          ← sequential over N
        K0: XOR word[0]          ← random access into candidate's word 0
        K1: XOR word[0..8]       ← sequential within one candidate
        K2: XOR word[0..256]     ← sequential within one candidate

Cache behavior:
    K0 touches 8 bytes per candidate but strides by 2048 bytes.
    For 1M candidates: 8 bytes useful / 2048 bytes per cache line = 0.4% utilization.
```

Each K0 probe fetches one u64 from a different 2KB region. The CPU prefetcher
can't help because the access pattern is "8 bytes at stride 2048" — every fetch
pulls a full cache line but uses 1/256th of it.

---

## The Transposed Architecture (SoA — Column-Major)

```
Memory layout:
    Column 0:   [cand₀_word₀, cand₁_word₀, cand₂_word₀, ... candₙ_word₀]
    Column 1:   [cand₀_word₁, cand₁_word₁, cand₂_word₁, ... candₙ_word₁]
    ...
    Column 255: [cand₀_word₂₅₅, cand₁_word₂₅₅, ... candₙ_word₂₅₅]

Algorithm (transposed cascade):
    Phase K0: scan Column 0 — XOR all N words with query[0], threshold filter
        → produces survivor_mask (bitmask of which candidates survived)

    Phase K1: scan Columns 0..8 — XOR only survivors' words, accumulate
        → narrows survivor_mask further

    Phase K2: scan Columns 0..255 — XOR only final survivors
        → exact EnergyConflict for each

Cache behavior:
    K0 is a contiguous sequential scan of N × 8 bytes.
    For 1M candidates: 8MB, perfectly prefetchable, 100% cache line utilization.
    AVX-512: 8 candidates per instruction (512 bits / 64 bits = 8 u64s).
```

---

## Why This Is Genius at the Meta Level

### 1. The Cascade Becomes SIMD-Native Filtering

Current K0 (per-candidate, scalar):
```rust
for i in 0..n_candidates {
    let word0 = database[i * 256];           // stride 2048 bytes — cache-hostile
    if (word0 ^ query_word0).count_ones() > threshold {
        continue;
    }
    // ... K1, K2
}
```

Transposed K0 (columnar, SIMD):
```rust
let col0: &[u64] = &columns[0];              // contiguous N×8 bytes
let q0 = U64x8::splat(query_words[0]);       // broadcast query word 0
let thresh = U32x8::splat(gate.k0_reject);

let mut survivors = vec![0u64; (n_candidates + 63) / 64];  // bitmask

for chunk in 0..(n_candidates / 8) {
    let c = U64x8::from_slice(&col0[chunk * 8..]);
    let xor = c ^ q0;
    let popcnt = xor.popcnt_per_lane();      // 8 popcounts in parallel
    let mask = popcnt.le(thresh);             // 8 comparisons → 8-bit mask
    survivors[chunk / 8] |= mask.to_bitmask() << (chunk % 8 * 8);
}
```

**8 candidates per clock cycle instead of 1.** And the memory access is a
sequential stream — the hardware prefetcher handles it perfectly.

---

### 2. The Survivor Mask IS the State Machine

The cascade becomes mask narrowing — no branch prediction, no `continue`:

```rust
// Start: all candidates alive
let mut mask: Vec<u64> = vec![u64::MAX; (n + 63) / 64];

// K0: narrow mask via column 0
k0_filter_columnar(&columns[0], query[0], gate.k0_reject, &mut mask);

// K1: narrow mask via columns 0..8 (only touch survivors)
k1_filter_columnar(&columns[0..8], &query[0..8], gate.k1_reject, &mut mask);

// K2: exact only on final survivors
let results = k2_exact_masked(&columns, query, &mask);
```

Each phase reads ONLY the columns it needs. K0 touches 1 column (N×8 bytes).
K1 touches 8 columns (N×64 bytes). K2 touches 256 columns but only for
survivors (~3-5% of N). Total memory touched:

```
AoS:  N × 2048 bytes (reads all words for all candidates even if K0 rejects)
      Actually reads: N × 8 (K0) + 0.05N × 64 (K1) + 0.003N × 2048 (K2)
      But cache lines pulled: N × 64 (one cache line per K0 probe) = N×64 bytes

SoA:  N × 8 (K0, contiguous) + N × 64 (K1, contiguous) + 0.003N × 2048 (K2)
      Effective: N × 72 + 6N = ~78N bytes vs ~64N bytes (AoS)
      BUT: 100% utilization vs 12.5% utilization per cache line
      Net bandwidth saving: ~8× due to no wasted cache line fetch
```

---

### 3. ndarray Makes This Free (Strides, Not Copies)

Here's the meta-level genius: **you don't transpose the data. You transpose the strides.**

```rust
// Build database in Fortran (column-major) order:
let database: Array2<u64> = Array2::from_shape_vec(
    (n_candidates, 256).f(),  // .f() = Fortran order
    flat_word_data,
).unwrap();

// Column access is now CONTIGUOUS (no copy, no transpose):
let col0: ArrayView1<u64> = database.column(0);
assert!(col0.is_standard_layout());  // TRUE — physically contiguous

// Row access still works (strided, for K2 on survivors):
let survivor_fp: ArrayView1<u64> = database.row(42);
// Not contiguous, but only used for the 3-5% of survivors — doesn't matter
```

The `(n_candidates, 256).f()` tells ndarray to lay out memory column-by-column.
Accessing `.column(i)` returns a contiguous view. Accessing `.row(i)` returns a
strided view. **Same data, both access patterns, zero transposition cost.**

This is ndarray's stride system doing something no raw-slice library can:
**the same allocation serves both columnar scanning AND per-candidate access.**

---

### 4. Arrow Is Already Column-Major

Arrow stores columns separately. The arrow_bridge already provides columnar views.
If each "word position" is stored as its own Arrow column (or a single Arrow
FixedSizeList with the right chunking), the SoA layout is the native format:

```rust
// Arrow RecordBatch with 256 columns of u64:
// col_0: [cand₀_w0, cand₁_w0, ...]  ← this IS the K0 scan buffer
// col_1: [cand₀_w1, cand₁_w1, ...]
// ...

fn kernel_pipeline_arrow(batch: &RecordBatch, query: &[u64; 256], gate: &SliceGate) {
    // K0: read column 0 directly from Arrow buffer (zero-copy)
    let col0 = batch.column(0).as_primitive::<UInt64Type>().values();
    let mask = k0_filter_columnar(col0, query[0], gate.k0_reject);

    // K1: read columns 0..8
    for i in 0..8 {
        let col = batch.column(i).as_primitive::<UInt64Type>().values();
        k1_accumulate_columnar(col, query[i], &mut mask, &mut accum);
    }
    // ...
}
```

**No conversion. No copy. Arrow's native layout IS the SoA kernel's input format.**

---

### 5. Multi-Resolution SoA: The Three Tiers as Physical Arrays

```rust
/// Tiered columnar database. Each tier is a separate Array2 because they have
/// different access patterns and lifetimes.
pub struct TieredDatabase {
    /// Tier K0: shape [N, 1] — just word 0 per candidate.
    /// Fits in L2 for 1M candidates (8MB).
    pub k0: Array2<u64>,  // Column-major, 1 column

    /// Tier K1: shape [N, 8] — first 8 words per candidate.
    /// Fits in L3 for 1M candidates (64MB).
    pub k1: Array2<u64>,  // Column-major, 8 columns

    /// Tier K2: shape [N, 256] — full containers.
    /// Lives in main memory. Only accessed for 3-5% survivors.
    pub k2: Array2<u64>,  // Column-major, 256 columns
}
```

**Why separate arrays?** Because K0 is HOT (scanned every query). K1 is WARM
(scanned for ~5% survivors). K2 is COLD (scanned for ~0.3% survivors). Separating
them lets you:
- Pin K0 in L2 cache (8MB for 1M candidates)
- Let K1 live in L3
- Let K2 stay in DRAM — only a few thousand rows are ever touched

This is **cache-aware SoA** — not just columnar layout, but columnar layout with
physical separation matching the thermal hierarchy of the CPU cache.

---

### 6. BF16 Field-Separated SoA: Awareness Becomes Memory Layout

Current BF16 storage: N vectors × 1024 BF16 values = N × 2048 bytes (interleaved).

Field-separated SoA:
```rust
/// BF16 database stored as separated bitfields.
/// The decomposition that bf16_hamming.rs does AT RUNTIME is instead
/// baked INTO THE STORAGE FORMAT.
pub struct BF16FieldDatabase {
    /// Sign bits: 1 bit per dimension × 1024 dims = 128 bytes per vector.
    /// Shape [N, 128] as u8 — each byte = 8 sign bits.
    pub signs: Array2<u8>,

    /// Exponent fields: 8 bits per dimension × 1024 dims = 1024 bytes per vector.
    /// Shape [N, 1024] as u8.
    pub exponents: Array2<u8>,

    /// Mantissa fields: 7 bits per dimension × 1024 dims ≈ 896 bytes per vector.
    /// Shape [N, 896] as u8 (packed 7-bit).
    pub mantissas: Array2<u8>,
}
```

**The awareness algebra becomes trivially parallel**:

```rust
impl BF16FieldDatabase {
    /// Sign-only distance (strongest signal): just Hamming on sign bytes.
    /// 128 bytes per comparison — fits in one AVX-512 pass for 4 vectors.
    fn sign_distance(&self, query_signs: &[u8], candidate_idx: usize) -> u32 {
        let row = self.signs.row(candidate_idx);
        hamming_dispatch(query_signs, row.as_slice().unwrap())
    }

    /// Exponent distance (medium signal): Hamming on exp bytes.
    fn exp_distance(&self, query_exp: &[u8], candidate_idx: usize) -> u32 {
        let row = self.exponents.row(candidate_idx);
        hamming_dispatch(query_exp, row.as_slice().unwrap())
    }

    /// Awareness classification without per-element decomposition.
    /// Signs are already separated — just compare columns directly.
    fn batch_awareness(&self, query_signs: &Array1<u8>) -> Array1<AwarenessState> {
        // For ALL candidates simultaneously:
        // sign_agreement = popcount(NOT (query_signs XOR candidate_signs))
        // Already column-accessible, already SIMD-friendly
        let n = self.signs.nrows();
        let mut states = Array1::zeros(n);
        for i in 0..n {
            let row = self.signs.row(i);
            let agreement = hamming_agreement(query_signs.as_slice().unwrap(),
                                             row.as_slice().unwrap());
            states[i] = classify_awareness(agreement, ...);
        }
        states
    }
}
```

**The insight**: bf16_hamming.rs currently decomposes each BF16 value into
sign/exp/mantissa at QUERY TIME (runtime unpacking). With field-separated SoA,
the decomposition happens at INGEST TIME (once). Every subsequent query operates
on pre-decomposed fields. The hot path never sees a BF16 value — only its fields.

This trades ingest bandwidth for query speed:
- Ingest: O(N) decomposition (amortized over all future queries)
- Query: zero decomposition cost (fields already separated)
- For a database queried 10,000× per second, this is a massive win.

---

### 7. PackedQualia SoA: 16 Parallel Dimension Columns

Current: `Vec<PackedQualia>` where each PackedQualia = 16 × i8 + 1 BF16 (18 bytes).

SoA:
```rust
/// Qualia database stored as separated dimension columns.
/// Each column is ONE phenomenological dimension across ALL agents.
pub struct QualiaColumns {
    /// 16 dimension columns, each Array1<i8> with N_agents elements.
    /// dims[0] = warmth, dims[1] = texture, dims[4] = social, etc.
    pub dims: [Array1<i8>; 16],

    /// Magnitude column: Array1<u16> (BF16 scalars).
    pub magnitude: Array1<u16>,
}
```

**What this unlocks**:

```rust
impl QualiaColumns {
    /// "Which agents are tensioned on warmth (dim 4)?"
    /// → Single SIMD scan of one column.
    fn tensioned_on(&self, dim: usize, threshold: i8) -> Array1<bool> {
        self.dims[dim].mapv(|v| v.abs() > threshold && v < 0)
    }

    /// "Resonance between agent A and all others on dimension 5"
    /// → One column × scalar = Array1<i8>
    fn resonance_on_dim(&self, agent_idx: usize, dim: usize) -> Array1<i16> {
        let agent_val = self.dims[dim][agent_idx] as i16;
        self.dims[dim].mapv(|v| v as i16 * agent_val)
    }

    /// "Total resonance between agent A and all others (dot product)"
    /// → 16 column scans, accumulate
    fn total_resonance(&self, agent_idx: usize) -> Array1<i32> {
        let mut acc = Array1::<i32>::zeros(self.dims[0].len());
        for d in 0..16 {
            let agent_val = self.dims[d][agent_idx] as i32;
            acc += &self.dims[d].mapv(|v| v as i32 * agent_val);
        }
        acc
    }
}
```

**AoS version of the same query** (current):
```rust
// "Find agents tensioned on warmth" — must scan ALL 18 bytes per agent,
// extract byte [4], compare. Cache line waste: 4/18 ≈ 22%.
for agent in &qualia_db {
    if agent.resonance[4].abs() > threshold && agent.resonance[4] < 0 {
        results.push(agent);
    }
}
```

**SoA version**: scan 1 byte per agent, sequential, 100% cache utilization.
For 1M agents: 1MB scan vs ~18MB with random byte extraction.

---

### 8. The Meta-Meta Level: ndarray IS the SoA Abstraction

The deepest insight is that **ndarray's type system already encodes this**.
`Array2<u64>` with Fortran order IS a SoA database. The columns are the "fields."
The rows are the "records." And ndarray's stride machinery gives you BOTH access
patterns from the same allocation:

```rust
// THE SAME Array2, TWO access patterns:

// SoA access (column scan for K0 — contiguous):
let k0_column: ArrayView1<u64> = database.column(0);
// → strides = [1], physically contiguous

// AoS access (full record for K2 — strided):  
let full_record: ArrayView1<u64> = database.row(survivor_idx);
// → strides = [n_candidates], non-contiguous but only 3% of data
```

You don't need two copies. You don't need explicit transposition. ndarray's
`Order::F` gives you SoA as the NATIVE layout while `Order::C` gives you AoS.
The same API works for both — only the performance characteristics change.

**This is what makes ndarray + hpc/ potentially genius**: the array type IS
the database schema. Shape IS the data model. Strides ARE the access pattern.
And the cascade kernel can be expressed as:

```rust
pub fn cascade_soa(
    database: &Array2<u64>,      // F-order: [N, 256], columns contiguous
    query: &Array1<u64>,         // [256]
    gate: &SliceGate,
) -> Vec<KernelResult> {
    let n = database.nrows();

    // K0: column 0 scan (contiguous — full SIMD bandwidth)
    let col0 = database.column(0);
    let mut mask = k0_columnar_simd(col0.as_slice().unwrap(), query[0], gate);

    // K1: columns 0..8 scan (contiguous per column)
    for w in 0..8 {
        let col = database.column(w);
        k1_accumulate_simd(col.as_slice().unwrap(), query[w], &mut mask);
    }
    k1_threshold(&mut mask, gate);

    // K2: only survivors (row access — strided but few)
    let mut results = Vec::new();
    for idx in mask.iter_ones() {
        let row = database.row(idx);
        // row is strided — copy to stack for SIMD (256 × 8 = 2KB, fits in L1)
        let mut buf = [0u64; 256];
        row.assign_to(ArrayViewMut1::from(&mut buf));
        let ec = k2_exact(&query.as_slice().unwrap(), &buf, 256);
        results.push(/* ... */);
    }
    results
}
```

---

## The Shopping List Addition

Add to `REFACTOR_HPC_INTEGRATION.md` Phase C:

```
Phase C-SoA (2 weeks) — Columnar Kernel Architecture:
    C.1  TieredDatabase struct (K0/K1/K2 as separate F-order Array2)
    C.2  k0_columnar_simd: scan N u64s with U64x8 broadcast+XOR+popcount
    C.3  Bitmask survivor propagation (no per-candidate branching)
    C.4  k1_accumulate_columnar: 8-column scan with running mask
    C.5  k2_exact_masked: row extraction for final survivors
    C.6  Arrow native path: RecordBatch columns → direct columnar scan
    C.7  BF16FieldDatabase: sign/exp/mantissa pre-separated at ingest
    C.8  QualiaColumns: 16 × Array1<i8> for per-dimension scan
    C.9  Benchmark: AoS vs SoA cascade on 1M SKU-16K containers
```

---

## Expected Performance Impact

| Operation | AoS (current) | SoA (proposed) | Speedup | Why |
|-----------|---------------|----------------|---------|-----|
| K0 scan (1M) | ~2ms (cache-miss bound) | ~0.25ms (streaming) | **8×** | Contiguous vs stride-2048 |
| K1 scan (50K survivors) | ~0.8ms | ~0.1ms | **8×** | Same reason, 8 columns |
| K2 exact (1500 survivors) | ~0.3ms | ~0.35ms | 0.85× | Row copy overhead |
| **Total pipeline** | **~3.1ms** | **~0.7ms** | **4.4×** | K0+K1 dominate |
| BF16 awareness (1M) | ~5ms (decompose each) | ~0.6ms (pre-separated) | **8×** | No runtime decomposition |
| Qualia dim scan (1M agents) | ~4.5ms (18 byte stride) | ~0.25ms (1 byte stride) | **18×** | Perfect cache line use |

The K2 step is slightly SLOWER in SoA (strided row access + copy to stack).
But K2 only runs on 0.3% of candidates — it's irrelevant to total time.
The cascade is dominated by K0 (touches all candidates) and K1 (touches 5%).
Both are massively faster with columnar layout.

---

## The Philosophical Payoff

> **AoS says**: "Here is a complete entity. Ask it anything."
> **SoA says**: "Here is one question. Ask it of everyone."

The cascade pipeline IS the SoA access pattern:
- K0 asks ONE question (word 0) of EVERYONE
- K1 asks EIGHT questions (words 0-7) of SURVIVORS
- K2 asks ALL questions (words 0-255) of VERY FEW

This is a funnel. Each stage narrows the population and widens the per-entity cost.
SoA is optimal for funnels because the wide-population stages (K0, K1) get
contiguous memory access, and the narrow-population stage (K2) pays the strided
penalty on so few entities that it doesn't matter.

ndarray encodes this directly in its type system:
- `database.column(0)` = the K0 question to everyone (contiguous)
- `database.row(survivor)` = all answers from one entity (strided, rare)

**The database layout IS the algorithm.
The strides ARE the access pattern.
ndarray IS the SoA abstraction.**
