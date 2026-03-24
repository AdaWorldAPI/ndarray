# ndarray Actions — Pumpkin Shopping List

## What Already Exists

```
palette_codec.rs    ✅ pack/unpack indices, variable bit width, PackedPaletteArray
packed.rs           ✅ PackedDatabase, 3-stroke cascade, stroke-aligned layout
bitwise.rs          ✅ hamming_distance_raw, popcount_raw, hamming_batch_raw, hamming_top_k_raw
cascade.rs          ✅ Cascade query with bands, calibration, precise mode
nars.rs             ✅ NarsTruth, NarsEvidence, NarsBudget
jitson/             ✅ JitEngine, ScanParams, build_scan_ir, CpuCaps
simd_avx512.rs      ✅ F32x16, U32x16, I32x16, reduce, fma, comparisons
kernels_avx512.rs   ✅ dot, axpy, scal, asum, nrm2, elementwise ops
```

## What's Missing — 9 Items

### 1. SIMD palette unpack (Pumpkin #1, #2, #3)

`palette_codec.rs` exists but is SCALAR. The unpack loop does
`(word >> (idx * bits)) & mask` one index at a time.

**Add:** `unpack_indices_simd(packed: &[u64], bits: usize, count: usize) -> Vec<u8>`

Use VPSHUFB + VPSRLVD to extract 16 palette indices per cycle.
`bits_per_index` baked as compile-time constant via const generic or
jitson immediate.

```rust
// src/hpc/palette_codec.rs — add SIMD path

/// SIMD-accelerated palette unpacking. 16 indices per cycle on AVX-512.
/// Falls back to scalar `unpack_indices` on non-AVX2 targets.
pub fn unpack_indices_simd(packed: &[u64], bits_per_index: usize, count: usize) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { unpack_avx2(packed, bits_per_index, count) };
    }
    unpack_indices(packed, bits_per_index, count) // scalar fallback
}

/// SIMD-accelerated palette packing. 16 indices per cycle.
pub fn pack_indices_simd(indices: &[u8], bits_per_index: usize) -> Vec<u64> {
    // VPSLLVD + VPOR to pack 16 indices into u64 words per cycle
}
```

**File:** `src/hpc/palette_codec.rs` — add `_simd` variants
**Tests:** roundtrip pack_simd ↔ unpack_simd, match scalar output
**Estimated lines:** ~150

---

### 2. Nibble batch extract/insert (Pumpkin #4)

Light levels are 4-bit nibbles packed 2 per byte. Currently extracted
one at a time. Need batch operations over full sections (4096 nibbles).

**Add:** `src/hpc/nibble.rs`

```rust
/// Batch extract 4-bit nibbles from packed byte array.
/// Returns full u8 values (0-15) for each nibble position.
pub fn nibble_unpack(packed: &[u8], count: usize) -> Vec<u8> {
    // VPSHUFB: extract low/high nibbles from 32 bytes → 64 nibble values per cycle
}

/// Batch insert 4-bit nibble values into packed byte array.
pub fn nibble_pack(values: &[u8], count: usize) -> Vec<u8> {
    // VPSHUFB reverse: merge pairs of nibbles into bytes
}

/// Batch subtract with clamp (light decay): all nibbles -= delta, clamp to 0.
/// Used in light propagation BFS.
pub fn nibble_sub_clamp(packed: &mut [u8], delta: u8) {
    // VPSUBUSB (unsigned subtract with saturation) on packed nibbles
}

/// Find all nibbles above threshold. Returns indices.
/// Used in light BFS: "which blocks still have light to propagate?"
pub fn nibble_above_threshold(packed: &[u8], threshold: u8) -> Vec<usize> {
    // VPCMPUB generates mask, _mm512_mask2int extracts indices
}
```

**File:** new `src/hpc/nibble.rs`, add to `src/hpc/mod.rs`
**Tests:** roundtrip, decay, threshold vs scalar
**Estimated lines:** ~200

---

### 3. AABB batch intersection (Pumpkin #7, #8)

Entity collision and block collision iterate pairs. Need SIMD batch test.

**Add:** `src/hpc/aabb.rs`

```rust
/// Axis-aligned bounding box as 6 f32s (min_x, min_y, min_z, max_x, max_y, max_z).
#[repr(C, align(32))]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// Test one AABB against N AABBs simultaneously.
/// Returns bitmask: bit i set if query intersects candidates[i].
pub fn aabb_intersect_batch(query: &Aabb, candidates: &[Aabb]) -> Vec<bool> {
    // VCMPPD tests 4 AABB pairs per cycle
    // min_a <= max_b AND max_a >= min_b for all 3 axes
}

/// Expand AABBs by (dx, dy, dz) — batch operation.
pub fn aabb_expand_batch(aabbs: &mut [Aabb], dx: f32, dy: f32, dz: f32) {
    // VSUBPS/VADDPS on min/max simultaneously
}

/// Squared distance from point to nearest point on each AABB.
/// Used for entity proximity queries.
pub fn aabb_squared_distance_batch(point: [f32; 3], aabbs: &[Aabb]) -> Vec<f32> {
    // VMAXPS clamp point to AABB, then squared distance
}
```

**File:** new `src/hpc/aabb.rs`
**Tests:** intersection correctness, expand, distance vs scalar
**Estimated lines:** ~250

---

### 4. Spatial hash for entity search (Pumpkin #6)

O(N) linear scan → grid hash + PackedDatabase cascade.

**Add:** `src/hpc/spatial_hash.rs`

```rust
/// 3D spatial hash grid. Entities hashed into cells by position.
/// Cell size tuned for Minecraft: 16 blocks (one chunk section width).
pub struct SpatialHash {
    cell_size: f32,
    grid: HashMap<(i32, i32, i32), Vec<u32>>,  // cell → entity indices
}

impl SpatialHash {
    pub fn new(cell_size: f32) -> Self;
    pub fn insert(&mut self, id: u32, x: f32, y: f32, z: f32);
    pub fn remove(&mut self, id: u32, x: f32, y: f32, z: f32);
    pub fn update(&mut self, id: u32, old: [f32; 3], new: [f32; 3]);

    /// Find all entities within radius of point.
    /// Returns (entity_id, squared_distance) pairs sorted by distance.
    ///
    /// Uses PackedDatabase internally: entity positions in candidate cells
    /// are packed stroke-aligned. Stroke 1 = grid cell match (instant reject).
    /// Stroke 2 = VCMPPD distance on survivors.
    pub fn query_radius(&self, x: f32, y: f32, z: f32, radius: f32) -> Vec<(u32, f32)>;

    /// Find K nearest entities to point.
    pub fn query_knn(&self, x: f32, y: f32, z: f32, k: usize) -> Vec<(u32, f32)>;
}
```

**File:** new `src/hpc/spatial_hash.rs`
**Tests:** insert/remove, radius query correctness, knn vs brute force
**Estimated lines:** ~300

---

### 5. Bitset property mask (Pumpkin #9, #10)

Block property queries allocate Box + Vec + string search.
Need: compile property queries to bitmask operations.

**Add:** `src/hpc/property_mask.rs`

```rust
/// A compiled property query. Checks multiple boolean properties
/// in a single SIMD operation.
///
/// Example: "waterlogged AND facing_north AND NOT open"
/// Compiles to: (bits & WATER_MASK) != 0 && (bits & FACE_MASK) == NORTH && (bits & OPEN_MASK) == 0
///
/// With AVX-512 VPTERNLOGD: checks 3 conditions in 1 cycle.
pub struct PropertyMask {
    /// AND mask: which bits must be set
    and_mask: u64,
    /// AND result: expected value after AND
    and_expect: u64,
    /// ANDN mask: which bits must NOT be set
    andn_mask: u64,
}

impl PropertyMask {
    pub fn new() -> Self;
    pub fn require_bit(mut self, bit: usize) -> Self;
    pub fn require_value(mut self, offset: usize, width: usize, value: u64) -> Self;
    pub fn forbid_bit(mut self, bit: usize) -> Self;

    /// Test a single block state against this mask.
    #[inline(always)]
    pub fn test(&self, block_state: u64) -> bool {
        (block_state & self.and_mask) == self.and_expect
            && (block_state & self.andn_mask) == 0
    }

    /// Batch test 4096 block states (one chunk section).
    /// Returns bitmask: 64 u64s, each bit = one block's result.
    pub fn test_section(&self, states: &[u64; 4096]) -> [u64; 64] {
        // VPTERNLOGD: 3-input truth table tests 3 conditions × 8 u64s per cycle
    }

    /// Count matching blocks in a section.
    pub fn count_section(&self, states: &[u64; 4096]) -> u32 {
        // test_section + VPOPCNTDQ
    }
}
```

**File:** new `src/hpc/property_mask.rs`
**Tests:** single test, batch test, count matches, VPTERNLOGD parity
**Estimated lines:** ~200

---

### 6. Batch popcount for tick scheduling (Pumpkin #10)

Count tick-eligible blocks across a full chunk section. The bitset exists
in pumpkin-data but counting is per-block.

**Add to:** `src/hpc/bitwise.rs`

```rust
/// Count set bits across an array of u64 words.
/// VPOPCNTDQ: 8 popcounts per cycle on AVX-512.
pub fn popcount_batch(words: &[u64]) -> u64 {
    // Existing popcount_raw works on &[u8]
    // This version works on &[u64] directly, skipping the byte reinterpret
}

/// For each u64, count set bits. Returns per-word counts.
/// Useful for "how many eligible blocks in each palette group?"
pub fn popcount_per_word(words: &[u64]) -> Vec<u32> {
    // VPOPCNTDQ returns 8 counts simultaneously
}
```

**File:** add to `src/hpc/bitwise.rs`
**Tests:** match scalar popcount
**Estimated lines:** ~60

---

### 7. Noise parameter baking (Pumpkin #5)

Perlin noise evaluates 4-16 octaves per block, loading parameters each time.
jitson can bake octave params as immediates.

**Add to:** `src/hpc/jitson/ir.rs`

```rust
/// Noise octave parameters — compiled as JIT immediates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseParams {
    /// Per-octave: (frequency_scale, amplitude_scale)
    /// Compiled as VMULPD immediates — no per-sample parameter load
    pub octaves: Vec<(f64, f64)>,
    /// Lacunarity — compiled as multiplication constant
    pub lacunarity: f64,
    /// Persistence — compiled as multiplication constant
    pub persistence: f64,
}
```

And in `src/hpc/jitson/scan.rs`, add `build_noise_ir()` that generates
a Cranelift function with octave loop fully unrolled and all params baked.

**File:** extend `src/hpc/jitson/ir.rs` + `scan.rs`
**Tests:** compiled noise matches scalar noise output
**Estimated lines:** ~200 (behind `jit-native` feature)

---

### 8. NBT tag batch scanner (Pumpkin #12)

Scan raw NBT bytes for tag type markers across multiple chunks.

**Add to:** `src/hpc/bitwise.rs` or new `src/hpc/byte_scan.rs`

```rust
/// Find all occurrences of a byte value in a buffer.
/// Returns indices. Uses VPCMPB on AVX-512.
pub fn byte_find_all(haystack: &[u8], needle: u8) -> Vec<usize> {
    // _mm512_cmpeq_epi8_mask: test 64 bytes per cycle
}

/// Find all occurrences of a 2-byte pattern (e.g., NBT tag type + name length prefix).
pub fn u16_find_all(haystack: &[u8], pattern: u16) -> Vec<usize> {
    // Compare 2-byte windows across 64-byte SIMD register
}
```

**File:** new `src/hpc/byte_scan.rs` or extend `bitwise.rs`
**Tests:** correctness vs naive scan, edge cases (pattern at boundary)
**Estimated lines:** ~100

---

### 9. Batch distance comparisons (Pumpkin #6, #7, #13)

Multiple hotspots need "squared distance from point to N points, filter by radius."

**Add to:** `src/hpc/aabb.rs` or new `src/hpc/distance.rs`

```rust
/// Squared distance from one point to N points (f32).
/// Returns distances. VCMPPD variant filters in-place.
pub fn squared_distances_f32(
    query: [f32; 3],
    points: &[[f32; 3]],
) -> Vec<f32> {
    // VSUBPS + VMULPS + horizontal add across xyz
}

/// Filter points by max squared distance. Returns indices of survivors.
pub fn filter_by_radius_sq(
    query: [f32; 3],
    points: &[[f32; 3]],
    radius_sq: f32,
) -> Vec<usize> {
    // squared_distances + VCMPPD threshold
}

/// Same but f64 (Minecraft uses f64 for entity positions).
pub fn squared_distances_f64(query: [f64; 3], points: &[[f64; 3]]) -> Vec<f64>;
pub fn filter_by_radius_sq_f64(query: [f64; 3], points: &[[f64; 3]], radius_sq: f64) -> Vec<usize>;
```

**File:** new `src/hpc/distance.rs`
**Tests:** correctness, filter vs brute force
**Estimated lines:** ~200

---

## Summary Table

| # | Module | File | Status | Lines | Pumpkin Hotspots |
|---|--------|------|--------|-------|-----------------|
| 1 | SIMD palette unpack/pack | palette_codec.rs | ADD `_simd` variants | ~150 | #1, #2, #3 |
| 2 | Nibble batch ops | NEW nibble.rs | CREATE | ~200 | #4, #11 |
| 3 | AABB batch intersection | NEW aabb.rs | CREATE | ~250 | #7, #8 |
| 4 | Spatial hash | NEW spatial_hash.rs | CREATE | ~300 | #6 |
| 5 | Property mask (VPTERNLOGD) | NEW property_mask.rs | CREATE | ~200 | #9 |
| 6 | Batch popcount | bitwise.rs | EXTEND | ~60 | #10 |
| 7 | Noise param baking | jitson/ir.rs + scan.rs | EXTEND | ~200 | #5 |
| 8 | NBT byte scanner | NEW byte_scan.rs | CREATE | ~100 | #12 |
| 9 | Batch distance filter | NEW distance.rs | CREATE | ~200 | #6, #7, #13 |
| **TOTAL** | | | | **~1660** | **all 13** |

## Priority Order

```
#5 property_mask.rs     — 10-50x speedup, easiest to implement (~200 lines)
#6 popcount batch       — 8-16x, extends existing bitwise.rs (~60 lines)
#1 palette SIMD         — 4-8x, extends existing palette_codec.rs (~150 lines)
#9 batch distance       — used by 3 hotspots (#6, #7, #13) (~200 lines)
#2 nibble ops           — used by 2 hotspots (#4, #11) (~200 lines)
#3 AABB batch           — used by 2 hotspots (#7, #8) (~250 lines)
#4 spatial hash         — biggest single hotspot speedup at scale (~300 lines)
#8 byte scanner         — NBT, less frequent (~100 lines)
#7 noise baking         — behind jit-native gate, worldgen only (~200 lines)
```

## Rules

- All SIMD ops on SLICES. Never copy.
- Scalar fallback for every SIMD path (non-x86 targets).
- All new files get `#[cfg(test)] mod tests` with scalar parity checks.
- palette_codec SIMD must produce IDENTICAL output to scalar (bit-exact).
- property_mask.test() must be #[inline(always)] — it's called per block.
