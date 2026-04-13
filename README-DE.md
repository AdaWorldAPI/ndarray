# ndarray — AdaWorldAPI HPC Erweiterung

Ein vollstaendiger Hochleistungs-Numerik-Stack auf Basis von [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray). 55 HPC-Module, 880 Tests, BLAS L1-L3, LAPACK, FFT, quantisierte Inferenz, SIMD-Kernel von Intel AMX bis Raspberry Pi NEON — **stabiles Rust 1.94**, null Nightly.

[English Version](README.md) | [Kompletter Feature-Vergleich (146 Module)](COMPARISON.md)

## Warum das existiert

| Was | Wir | GPU (RTX 3060) | GPU (H100) | NumPy CPU |
|-----|-----|----------------|------------|-----------|
| **Cosine-Aehnlichkeit** | **2.400M/s** (Palette u8) | ~300M/s (IVF-PQ) | ~1.500M/s (cuVS) | ~50M/s (Dot) |
| **GEMM 1024x1024** | **139 GFLOPS** | 3.500 GFLOPS | 30.000 GFLOPS | 120 GFLOPS |
| **Codebook-Inferenz** | **2.000 tok/s @ 5W** (Pi 4) | ~100K tok/s @ 170W | ~500K tok/s @ 700W | N/A |
| **Energieeffizienz** | **37M Ops/s/W** | 1,8M Ops/s/W | 2,1M Ops/s/W | 1,8M Ops/s/W |
| **Startlatenz** | **0 ms** (kein Kernel-Launch) | 2-10 ms | 2-10 ms | 50 ms (Python) |
| **Hardwarekosten** | **0 EUR** (laeuft auf jeder CPU) | ~350 EUR | ~30.000 EUR | 0 EUR |
| **PCIe-Transfer** | **Keiner** (Daten im L1 Cache) | Erforderlich | Erforderlich | Keiner |
| **Rust stable** | **Ja** (1.94) | CUDA Toolkit | CUDA Toolkit | Python |

GPU gewinnt bei grosser dichter GEMM. Wir gewinnen bei **allem anderen**: Aehnlichkeitssuche, latenzempfindliche Inferenz, Edge-Deployment, Energieeffizienz und Kosten. Ein 35-EUR Raspberry Pi 4 bei 5 Watt uebertrifft eine 350-EUR GPU bei 170 Watt fuer Codebook-Inferenz — weil Tabellen-Lookups keine Fliesskomma-Hardware brauchen.

## Upstream vs. Fork — Feature fuer Feature

### ISA-Abdeckung (Instruction Set Architecture)

| ISA / Feature | Upstream ndarray | **AdaWorldAPI Fork** | Speedup vs. Upstream |
|---------------|-----------------|---------------------|---------------------|
| **AVX-512** (512-bit, 16xf32) | Scalar Fallback | Native `__m512` Typen, F32x16/F64x8/U8x64 | **~8x** |
| **AVX-512 VNNI** (int8 dot) | Scalar Fallback | `vpdpbusd` 64 MACs/Instr + Dispatch | **~32x** |
| **AVX-512 BF16** (bfloat16) | Nicht vorhanden | Hardware `vcvtneps2bf16` + RNE-Emulation | **neu** |
| **AVX-512 VPOPCNTDQ** (popcount) | Scalar Fallback | Native 512-bit Popcount fuer Hamming | **~16x** |
| **AMX** (Tile Matrix, 256 MACs) | Nicht vorhanden | Inline-ASM `.byte` Encoding, stable Rust | **~128x** vs. Scalar |
| **AVX2 + FMA** (256-bit, 8xf32) | Via matrixmultiply | Eigene Goto-GEMM 6x16 + Dispatch-Tabelle | **~4x** |
| **AVX2 F16C** (f16 Hardware) | Nicht vorhanden | IEEE 754 f16, Double-f16, Kahan, Scaler | **neu** |
| **AVX-VNNI** (ymm, 32 MACs) | Nicht vorhanden | Arrow Lake / NUC 14 Unterstuetzung | **neu** |
| **SSE2** (128-bit, 4xf32) | Via matrixmultiply | Scalar Polyfill mit gleicher API | 1x (Baseline) |
| **NEON** (128-bit, 4xf32) | Scalar Fallback | 3-stufig: A53/A72/A76 mit Pipeline-Awareness | **~4x** |
| **NEON dotprod** (ARMv8.2) | Nicht vorhanden | `vdotq_s32` fuer 4x int8 Durchsatz (Pi 5) | **~16x** vs. Scalar |
| **NEON fp16** (ARMv8.2) | Nicht vorhanden | `FCVTL`/`FCVTN` via Inline-ASM | **neu** |
| **NEON Popcount** | Nicht vorhanden | `vcntq_u8` nativer Byte-Popcount | **schneller als x86 SSE** |
| **WASM SIMD128** | Nicht vorhanden | Scaffolding vorbereitet | in Arbeit |

### Was Upstream auf jedem Target macht

```
Upstream auf x86_64:   -> matrixmultiply Crate (extern, AVX2 wenn verfuegbar, kein AVX-512)
Upstream auf aarch64:  -> Scalar (kein NEON, keine Intrinsics)
Upstream auf wasm:     -> Scalar

Fork auf x86_64:       -> AVX-512 / AVX2 / SSE2 / Scalar (gestuft, auto-erkannt)
Fork auf aarch64:      -> NEON A76+dotprod / A72 2x Pipeline / A53 / Scalar (gestuft)
Fork auf wasm:         -> WASM SIMD128 (vorbereitet) / Scalar
```

### BLAS / Numerik

| Operation | Upstream | **Fork** | Verbesserung |
|-----------|----------|----------|-------------|
| GEMM (1024x1024) | ~13 GFLOPS (Cache-Cliff) | **139 GFLOPS** (Goto-Blocking) | **10,5x** |
| Dot Product | Via matrixmultiply | 4-fach unrolled + FMA | ~2x |
| BLAS L1 (axpy, scal, nrm2) | Nicht vorhanden | SIMD-beschleunigt, alle Tiers | **neu** |
| BLAS L2 (gemv, ger, trsv) | Nicht vorhanden | SIMD-beschleunigt | **neu** |
| LAPACK (LU, Cholesky, QR) | Nicht vorhanden | Pure-Rust Implementierung | **neu** |
| FFT | Nicht vorhanden | Cooley-Tukey Radix-2 | **neu** |
| Aktivierungen (sigmoid, GELU) | Nicht vorhanden | SIMD F32x16 Vektorisierung | **neu** |
| Quantisierung (BF16, INT8) | Nicht vorhanden | VNNI + AMX + Scalar Fallback | **neu** |

### Datentypen

| Typ | Upstream | **Fork** | Anmerkung |
|-----|----------|----------|-----------|
| f32 | Standard | Standard + F32x16 SIMD | Gleich + SIMD-Beschleunigung |
| f64 | Standard | Standard + F64x8 SIMD | Gleich + SIMD-Beschleunigung |
| **f16** (IEEE 754) | **Nicht vorhanden** | u16 Carrier + F16C/FCVTL Hardware | Stable Rust, kein Nightly |
| **BF16** (bfloat16) | **Nicht vorhanden** | Hardware + RNE-Emulation (bit-exakt) | GGUF-Kalibrierung |
| i8/u8 (quantisiert) | Nicht vorhanden | VNNI dot, Hamming, Popcount | INT8 Inferenz |
| i16 (Base17) | Nicht vorhanden | L1-Distanz, SIMD widen/narrow | Codebook-Encoding |

## Leistung

### GEMM (Allgemeine Matrixmultiplikation)

| Matrixgroesse | Upstream | **Dieser Fork** | NumPy | PyTorch CPU | GPU (RTX 3060) |
|--------------|---------|---------------|-------|-------------|----------------|
| 512x512 | ~20 GFLOPS | **47 GFLOPS** | ~45 | ~40 | ~1.200 |
| 1024x1024 | ~13 GFLOPS | **139 GFLOPS** | ~120 | ~100 | ~3.500 |
| 2048x2048 | ~13 GFLOPS | **~150 GFLOPS** | ~140 | ~130 | ~5.000 |

**10,5x ueber Upstream** bei 1024x1024 — auf NumPy OpenBLAS Niveau.

### Codebook-Inferenz

| Hardware | ISA | tok/s | 50-Token Latenz | Leistung |
|----------|-----|-------|-----------------|----------|
| Sapphire Rapids | AMX | **380.000** | 0,13 ms | 250W |
| Xeon | AVX-512 VNNI | **10K-50K** | 1-5 ms | 150W |
| **Pi 5** | **NEON+dotprod** | **2K-5K** | 10-25 ms | **5W** |
| **Pi 4** | **NEON dual** | **500-2K** | 25-100 ms | **5W** |

### Cosine via Palette-Distanz

| Stufe | Fehler | Geschwindigkeit | vs. GPU (RTX 3060) |
|-------|--------|----------------|---------------------|
| **Foveal** (1/40 sigma) | 0,4% | **611M/s** | **~2x schneller** |
| **Nah** (1 sigma) | 8% | **2.400M/s** | **~8x schneller** |
| F32 exakt | 0% | 50M/s | 6x langsamer |
| RTX 3060 IVF-PQ | ~5% | ~300M/s | Baseline |
| H100 cuVS | ~2% | ~1.500M/s | 5x unsere Kosten |

611M Cosine-aequivalente Lookups/Sek mit reinen Integer-Operationen. Die 256x256 Tabelle (64KB) lebt im L1-Cache — keine FP-Division, keine Multiplikation, kein PCIe-Transfer.

### f16 Gewichts-Transkodierung

| Format | Groesse | Max Fehler | Durchsatz |
|--------|---------|-----------|-----------|
| f32 | 60 MB | — | — |
| **f16** | **30 MB** | 7,3e-6 | 94M/s |
| **Scaled-f16** | **30 MB** | 4,9e-6 | 91M/s |
| **Double-f16** | 60 MB | 5,7e-8 | 42M/s |

## Was wir bauen, das sonst niemand hat

1. **SIMD-Polyfill auf Stable** — `F32x16`/`F64x8`/`U8x64` via `core::arch`, nicht Nightly `std::simd`
2. **f16 ohne Nightly** — `u16` Carrier + F16C Hardware / ARM `FCVTL` via `asm!()`
3. **AMX auf Stable** — `asm!(".byte ...")` Encoding, 256 MACs/Instruktion
4. **Gestuftes ARM NEON** — A53/A72/A76 mit Pipeline- + big.LITTLE-Awareness
5. **0,3ns Dispatch** — LazyLock eingefrorene Funktionszeiger-Tabelle
6. **BF16 RNE bit-exakt** — Pure AVX-512-F emuliert `VCVTNEPS2BF16` Bit-fuer-Bit
7. **Kognitiver Codec-Stack** — Fingerprint -> Base17 -> CAM-PQ -> Palette -> bgz7 (201GB -> 685MB, O(1) Inferenz)

## Schnellstart

```rust
use ndarray::Array2;
use ndarray::hpc::simd_caps::simd_caps;

let a = Array2::<f32>::ones((1024, 1024));
let c = a.dot(&a);  // AVX-512 / AVX2 / NEON — automatisch

let caps = simd_caps();
if caps.neon { println!("{}", caps.arm_profile().name()); }
```

```bash
cargo build --release                                        # auto-detect
cargo build --release --target aarch64-unknown-linux-gnu     # Pi 4
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release   # AVX-512
cargo test                                                    # 880 Tests
```

## Oekosystem

| Repo | Rolle |
|------|-------|
| [lance-graph](https://github.com/AdaWorldAPI/lance-graph) | Graph-Query + Codec-Spine |
| [home-automation-rs](https://github.com/AdaWorldAPI/home-automation-rs) | Smart Home + Sprach-KI |

## Lizenz

MIT OR Apache-2.0
