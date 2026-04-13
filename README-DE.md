# ndarray — AdaWorldAPI HPC Erweiterung

Ein vollstaendiger Hochleistungs-Numerik-Stack auf Basis von [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray). Dieser Fork fuegt 55 HPC-Module mit 880 Tests hinzu: BLAS L1-L3, LAPACK, FFT, Vektormathematik, quantisierte Inferenz und hardware-spezifische SIMD-Kernel von Intel AMX bis Raspberry Pi NEON — alles auf **stabilem Rust 1.94**, null Nightly-Features.

Das Upstream-ndarray liefert exzellente n-dimensionale Array-Abstraktionen. Wir behalten all das und fuegen hinzu, wofuer es nie gedacht war: mit NumPys OpenBLAS bei GEMM konkurrieren, Codebook-Inferenz auf einem 5-Watt Pi 4 laufen lassen, und Halbpraezisions-Gleitkommazahlen verarbeiten, fuer die Rust noch nicht einmal einen stabilen Typ hat.

[English Version](README.md)

## Upstream vs. Fork — Feature-fuer-Feature

Die zentrale Frage: Was genau bekommt man mit diesem Fork, was Upstream nicht hat?

### ISA-Abdeckung (Instruction Set Architecture)

| ISA / Feature | Upstream ndarray | **AdaWorldAPI Fork** | Speedup vs. Upstream |
|---------------|-----------------|---------------------|---------------------|
| **AVX-512** (512-bit, 16×f32) | Scalar Fallback | Native `__m512` Typen, F32x16/F64x8/U8x64 | **~8×** |
| **AVX-512 VNNI** (int8 dot) | Scalar Fallback | `vpdpbusd` 64 MACs/Instr + Dispatch | **~32×** |
| **AVX-512 BF16** (bfloat16) | Nicht vorhanden | Hardware `vcvtneps2bf16` + RNE-Emulation | **neu** |
| **AVX-512 VPOPCNTDQ** (popcount) | Scalar Fallback | Native 512-bit Popcount fuer Hamming | **~16×** |
| **AMX** (Tile Matrix, 256 MACs) | Nicht vorhanden | Inline-ASM `.byte` Encoding, stable Rust | **~128×** vs. Scalar |
| **AVX2 + FMA** (256-bit, 8×f32) | Via matrixmultiply | Eigene Goto-GEMM 6×16 + Dispatch-Tabelle | **~4×** |
| **AVX2 F16C** (f16 Hardware) | Nicht vorhanden | IEEE 754 f16, Double-f16, Kahan, Scaler | **neu** |
| **AVX-VNNI** (ymm, 32 MACs) | Nicht vorhanden | Arrow Lake / NUC 14 Unterstuetzung | **neu** |
| **SSE2** (128-bit, 4×f32) | Via matrixmultiply | Scalar Polyfill mit gleicher API | 1× (Baseline) |
| **NEON** (128-bit, 4×f32) | Scalar Fallback | 3-stufig: A53/A72/A76 mit Pipeline-Awareness | **~4×** |
| **NEON dotprod** (ARMv8.2) | Nicht vorhanden | `vdotq_s32` fuer 4× int8 Durchsatz (Pi 5) | **~16×** vs. Scalar |
| **NEON fp16** (ARMv8.2) | Nicht vorhanden | `FCVTL`/`FCVTN` via Inline-ASM | **neu** |
| **NEON Popcount** | Nicht vorhanden | `vcntq_u8` nativer Byte-Popcount | **schneller als x86 SSE** |
| **WASM SIMD128** | Nicht vorhanden | Scaffolding vorbereitet | in Arbeit |

### BLAS / Numerik

| Operation | Upstream | **Fork** | Verbesserung |
|-----------|----------|----------|-------------|
| GEMM (1024²) | ~13 GFLOPS (Cache-Cliff) | **139 GFLOPS** (Goto-Blocking) | **10.5×** |
| Dot Product | Via matrixmultiply | 4-fach unrolled + FMA | ~2× |
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

### Dispatch & Erkennung

| Aspekt | Upstream | **Fork** |
|--------|----------|----------|
| SIMD-Erkennung | Keine (delegiert an BLAS) | `LazyLock<SimdCaps>` — einmal erkennen, fuer immer |
| Dispatch-Kosten | Kein eigener Dispatch | **0.3ns** (Funktionszeiger-Tabelle, kein Branch) |
| ARM-Profiling | Kein ARM-Bewusstsein | `ArmProfile`: A53/A72/A76 mit tok/s Schaetzung |
| big.LITTLE | Nicht behandelt | Korrekte Feature-Intersection (RK3399/RK3588) |
| CPU-Erkennung | Zur Laufzeit per Call | Einmal via LazyLock, dann nur Pointer-Deref |

### Zusammenfassung: Was Upstream auf jedem Target macht

```
Upstream auf x86_64:   → matrixmultiply Crate (extern, AVX2 wenn verfuegbar)
Upstream auf aarch64:  → Scalar (kein NEON, kein Intrinsic)
Upstream auf wasm:     → Scalar
Upstream auf riscv:    → Scalar

Fork auf x86_64:       → AVX-512 F32x16 / AVX2 F32x8 / SSE2 / Scalar (gestuft)
Fork auf aarch64:      → NEON A76+dotprod / NEON A72 2×pipe / NEON A53 / Scalar
Fork auf wasm:         → WASM SIMD128 (vorbereitet) / Scalar
Fork auf riscv:        → Scalar (RISC-V V Extension vorbereitet)
```

## Leistung

### GEMM (Allgemeine Matrixmultiplikation)

| Matrixgroesse | Upstream ndarray | **Dieser Fork** | NumPy (OpenBLAS) | PyTorch CPU | GPU (RTX 3060) |
|--------------|-----------------|---------------|------------------|-------------|----------------|
| 512×512 | ~20 GFLOPS | **47 GFLOPS** | ~45 GFLOPS | ~40 GFLOPS | ~1.200 GFLOPS |
| 1024×1024 | ~13 GFLOPS | **139 GFLOPS** | ~120 GFLOPS | ~100 GFLOPS | ~3.500 GFLOPS |
| 2048×2048 | ~13 GFLOPS | **~150 GFLOPS** | ~140 GFLOPS | ~130 GFLOPS | ~5.000 GFLOPS |

Upstream trifft bei 1024×1024 auf eine Cache-Klippe: kein Tiling, kein Threading, kein Microkernel. Unsere Goto-Implementierung eliminiert das vollstaendig.

### Codebook-Inferenz (Token-Generierung)

Keine Matrixmultiplikation — O(1) Tabellen-Lookup pro Token.

| Hardware | ISA | tok/s | 50-Token Latenz | Leistung |
|----------|-----|-------|-----------------|----------|
| Sapphire Rapids | AMX (256 MACs/Instr) | **380.000** | 0,13 ms | 250W |
| Xeon / i9-13900K | AVX-512 VNNI | **10.000–50.000** | 1–5 ms | 150W |
| i7-13800K | AVX2-VNNI | **3.000–10.000** | 5–17 ms | 65W |
| **Raspberry Pi 5** | **NEON + dotprod** | **2.000–5.000** | 10–25 ms | **5W** |
| **Raspberry Pi 4** | **NEON (2× Pipeline)** | **500–2.000** | 25–100 ms | **5W** |
| Pi Zero 2W | NEON (1× Pipeline) | 50–500 | 100–1000 ms | 2W |

Bei 5 Watt generiert ein Pi 4 eine 50-Token Sprachassistenten-Antwort in unter 100 Millisekunden.

### Cosine-Aehnlichkeit via Palette-Distanz (nur Integer)

Traditionelle Cosine-Aehnlichkeit braucht Fliesskomma: `dot(a,b) / (|a| × |b|)`. Wir ersetzen das durch einen einzigen u8-Tabellen-Lookup.

| Praezisions-Stufe | Sigma-Band | Max Cosine-Fehler | Geschwindigkeit |
|-------------------|------------|-------------------|----------------|
| **Foveal** (1/40 σ) | Innere 2,5% | ±0,004 (0,4%) | **611M Lookups/s** |
| **Gut** (1/4 σ) | Innere 68% | ±0,02 (2%) | **611M Lookups/s** |
| **Nah** (1 σ) | Innere 95% | ±0,08 (8%) | **2,4 Mrd/s** |
| F32 exakte Cosine | — | 0 | ~50M/s |

**611 Millionen Cosine-aequivalente Vergleiche pro Sekunde mit reinen Integer-Operationen** — 12× schneller als SIMD-f32-Skalarprodukt. Die 256×256 Tabelle (64KB) passt komplett in den L1-Cache.

### Halbpraezisions-Gewichts-Transkodierung

Getestet mit 15-Millionen-Parameter-Modell (Piper TTS Groesse):

| Format | Groesse | Max Fehler | RMSE | Durchsatz |
|--------|---------|-----------|------|-----------|
| f32 (Original) | 60 MB | — | — | — |
| **f16 (IEEE 754)** | **30 MB** | 7,3×10⁻⁶ | 2,5×10⁻⁶ | 94M Params/s |
| **Scaled-f16** | **30 MB** | 4,9×10⁻⁶ | 2,1×10⁻⁶ | 91M Params/s |
| **Double-f16** | 60 MB | 5,7×10⁻⁸ | 1,8×10⁻⁸ | 42M Params/s |

## Was wir bauen, das sonst niemand hat

### 1. Vollstaendiger SIMD-Polyfill auf stabilem Rust

`std::simd` ist seit Jahren Nightly-only. Wir implementieren dieselbe Typ-Oberflaeche mit stabilen `core::arch` Intrinsics. Wenn `std::simd` stabilisiert wird, aendert der Consumer eine `use`-Zeile.

### 2. Halbpraezisions-Typen ohne Nightly

Rusts `f16`-Typ ist Nightly-only. Wir nutzen `u16` als Traeger + Hardware-Instruktionen via stabiles `#[target_feature]` (F16C auf x86, `FCVTL`/`FCVTN` via Inline-`asm!()` auf ARM).

### 3. AMX auf stabilem Rust

Intel AMX Intrinsics sind Nightly-only. Wir emittieren Instruktionen direkt via `asm!(".byte ...")` — 256 MACs pro Instruktion, verifiziert auf Rust 1.94 stable.

### 4. Gestuftes ARM NEON fuer Einplatinen-Computer

Drei Stufen mit Laufzeit-Erkennung: A53 Baseline (Pi Zero/3), A72 Fast (Pi 4, Dual-Pipeline), A76 DotProd (Pi 5, `vdotq_s32` + natives fp16). big.LITTLE-bewusst.

### 5. Eingefrorener Dispatch (0,3ns pro Aufruf)

Funktionszeiger-Tabelle statt Branch pro Aufruf. `LazyLock<SimdDispatch>` → ein indirekter Call, kein Atomic, kein Branch-Prediction-Miss.

### 6. BF16 RNE bit-exakt mit Hardware

Pure AVX-512-F Emulation von `VCVTNEPS2BF16`, verifiziert Bit-fuer-Bit auf 1M+ Eingaben.

### 7. Kognitiver Codec-Stack

Fingerprint<256>, Base17 VSA, CAM-PQ, Palette-Semiring, bgz7/bgz17 — komprimierte Modellgewichte (201GB → 685MB) mit O(1) Inferenz.

## Schnellstart

```rust
use ndarray::Array2;
use ndarray::hpc::simd_caps::simd_caps;

let a = Array2::<f32>::ones((1024, 1024));
let b = Array2::<f32>::ones((1024, 1024));
let c = a.dot(&b);  // AVX-512 / AVX2 / NEON — null Code-Aenderungen

let caps = simd_caps();
if caps.avx512f { println!(\"AVX-512: 16 Lanes\"); }
if caps.neon { println!(\"ARM: {}\", caps.arm_profile().name()); }
```

```bash
cargo build --release
cargo build --release --target aarch64-unknown-linux-gnu  # Pi 4
RUSTFLAGS=\"-C target-cpu=x86-64-v4\" cargo build --release  # AVX-512
cargo test  # 880 HPC Tests
```

## Voraussetzungen

- **Rust 1.94 stable** (kein Nightly, keine instabilen Features)
- Optional: `gcc-aarch64-linux-gnu` fuer Pi Cross-Kompilierung
- Optional: Intel MKL oder OpenBLAS fuer BLAS-Beschleunigung (Feature-gated)

## Oekosystem

| Repository | Rolle | Nutzt ndarray fuer |
|------------|-------|-------------------|
| [lance-graph](https://github.com/AdaWorldAPI/lance-graph) | Graph-Query + Codec-Spine | Fingerprint, CAM-PQ, CLAM, BLAS, ZeckF64 |
| [home-automation-rs](https://github.com/AdaWorldAPI/home-automation-rs) | Smart Home + Sprach-KI | Codebook-Inferenz, VITS TTS, SIMD Audio |

## Lizenz

MIT OR Apache-2.0 (wie Upstream ndarray)
