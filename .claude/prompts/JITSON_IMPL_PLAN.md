# Jitson Shopping List — Implementation Plan

> **Date:** 2026-03-24
> **Scope:** ndarray HPC SIMD upgrades for Pumpkin Minecraft server optimization
> **Principle:** Upgrade existing scalar code to SIMD, file-by-file, with scalar parity tests

---

## Architecture Pattern (All SIMD Code Follows This)

1. **Public dispatch function** → `is_x86_feature_detected!()` → best available backend
2. **`#[target_feature(enable = "...")]` unsafe inner** → actual intrinsics
3. **Scalar fallback** always present
4. **`// SAFETY:` comment** before every unsafe block
5. **Parity tests** compare SIMD output against scalar reference

Dispatch hierarchy: AVX-512 VPOPCNTDQ > AVX-512 BW > AVX-512 F > AVX2 > SSE4.1 > scalar

---

## Phase 1: Foundation SIMD Upgrades (No New Public API, Parallelizable)

### 1A. `byte_scan.rs` — AVX-512 VPCMPB (64 bytes/cycle)
- Add `byte_find_all_avx512` + `byte_count_avx512` using `_mm512_cmpeq_epi8_mask`
- Update dispatch: check `avx512bw` before `avx2`
- **~60 new lines**

### 1B. `property_mask.rs` — AVX-512 VPTERNLOGD + VPOPCNTDQ
- Add `test_section_avx512` processing 8 u64s/iter with `_mm512_ternarylogic_epi64`
- Add `count_section_avx512` with `_mm512_popcnt_epi64` (VPOPCNTDQ)
- **~80 new lines**

### 1C. `palette_codec.rs` — AVX-512 Unpack All Bit Widths + Pack
- Add `unpack_generic_avx512` using `_mm512_srlv_epi32` (VPSRLVD) with shift table
- Add `pack_generic_avx512` using `_mm512_sllv_epi32` (VPSLLVD) + `_mm512_or_epi32`
- Start with power-of-2 widths (1,2,4,8), then add 3,5,6,7
- **~150 new lines**

---

## Phase 2: Nibble Module Expansion

### 2A. `nibble_unpack_avx2` — 32 nibbles/cycle
- Load 16 bytes → AND low, shift+AND high → interleave → store 32 u8s
- **~50 new lines**

### 2B. `nibble_above_threshold_avx2` — SIMD threshold scan
- Split lo/hi nibbles, cmpgt threshold, extract bitmask, emit indices
- **~60 new lines**

### 2C. `nibble_propagate_bfs` — Compose existing kernels
- `nibble_sub_clamp(packed, delta)` + `nibble_above_threshold(packed, 0)` → frontier
- **~20 new lines**

### 2D. `nibble_sub_clamp_avx512` — 64 bytes/iter (128 nibbles)
- `_mm512_subs_epu8` for saturating subtract
- **~35 new lines**

---

## Phase 3: AABB Module

### 3A. AVX-512 Batch Intersect — 16 candidates/iter
- Broadcast query, gather candidate coords, `_mm512_cmp_ps_mask`, AND 6 kmasks
- **~80 new lines**

### 3B. Ray-AABB Slab Test — Projectile collision
- New `Ray` struct, slab method (t_enter/t_exit), scalar + AVX-512
- **~120 new lines**

---

## Phase 4: Spatial Hash SIMD Distance

- `batch_sq_dist_avx2` helper for inner loop
- New `query_radius_simd` method
- **~100 new lines**

---

## Phase 5: Jitson Templates

### 5A. TerrainFillParams — Baked biome params for JIT fill loop
### 5B. CompiledNoiseConfig — Flattened octave params for JIT compilation
- **~140 new lines combined**

---

## Phase 6: Wiring
- Re-export new types from `jitson/mod.rs`
- **~5 lines**

---

## Total: ~900 new lines across 8 files

## Dependency Graph

```
Phase 1 (parallel):   byte_scan ─┐
                      prop_mask ──┼── Phase 2 (nibble) ── Phase 3 (aabb) ── Phase 4 (spatial)
                     palette_codec┘                                              │
                                                                          Phase 5 (jitson)
                                                                                 │
                                                                          Phase 6 (wire)
```
