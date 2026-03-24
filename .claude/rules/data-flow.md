# Data-Flow Invariants — Three Patterns, One Rule

Every piece of data in hpc falls into exactly one category:

## 1. SIMD — `&[u8]` slices into backing store
- Zero-copy: borrow from PackedDatabase, Arrow buffers, BindSpace
- Alignment matters: SIMD loads require 64-byte alignment for AVX-512
- Never allocate inside a hot loop — slice into pre-allocated storage

## 2. Reasoning — owned `Copy` microcopies
- `TruthValue`, `Fingerprint`, `u64`, `Band`, `CpuCaps`, `ScanParams`
- Small, stack-allocated, passed by value
- Clone is free (Copy), no heap, no lifetime tracking

## 3. Write-back — gated operations only
- **Single target**: gated XOR (`target ^= mask & value`)
- **Multiple targets**: BUNDLE (majority vote / weighted merge)
- **NEVER** raw `=` assignment to shared data during computation
- Mutations are explicit, auditable, gate-controlled

## The Rule
**No `&mut self` during computation. Ever.**

- Engines return results; they do not mutate themselves while computing
- Caches use interior mutability (`RwLock`, `LazyLock`) or are built once
- If a function needs `&mut`, it is a *builder* or *constructor*, not a *compute* path
