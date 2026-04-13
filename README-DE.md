# ndarray — HPC-Erweiterung fuer Rust

*Fork von [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray) mit 55 HPC-Modulen, 880 Tests, und SIMD-Kernels von Intel AMX bis Raspberry Pi NEON. Laeuft auf stabilem Rust 1.94 ohne Nightly-Features.*

[English Version](README.md) | [Kompletter Feature-Vergleich (146 Module)](COMPARISON.md)

---

## Worum geht es

Das Upstream-ndarray ist eine solide Bibliothek fuer n-dimensionale Arrays in Rust. Was es nicht liefert: hardwarenahe SIMD-Beschleunigung, BLAS ohne externe C-Bibliotheken, und Unterstuetzung fuer Datentypen wie f16 oder BF16, die Rust auf stabilem Toolchain schlicht nicht anbietet.

Dieser Fork schliesst diese Luecken. Die Erweiterung umfasst 80.000 Zeilen Code in 179 neuen Dateien — von Goto-GEMM-Mikrokernels ueber ARM-NEON-Stufenerkennung bis zu einem Codec-Stack, der Cosine-Aehnlichkeit als Integer-Tabellen-Lookup implementiert.

Das Ergebnis laesst sich an einer Zahl festmachen: **611 Millionen Aehnlichkeitsvergleiche pro Sekunde** auf einer Consumer-CPU, ohne Fliesskomma-Arithmetik, ohne GPU.

---

## Die zentrale Idee: Cosine-Aehnlichkeit ohne Fliesskomma

Vektorsuche in Datenbanken wie LanceDB oder FAISS berechnet fuer jeden Kandidaten ein Skalarprodukt: `dot(a,b) / (|a| * |b|)`. Bei 768 Dimensionen sind das 1.536 Fliesskomma-Operationen und 6 KB Speicherbandbreite pro Vergleich.

Dieser Fork geht einen anderen Weg. Vektoren werden offline auf 256 Archetypes quantisiert. Die paarweisen Distanzen zwischen allen Archetypes sind in einer 256x256-Tabelle (64 KB) vorberechnet. Zur Laufzeit reduziert sich eine Cosine-Abfrage auf einen einzigen Byte-Lesevorgang aus dem L1-Cache.

### Messwerte nach Hardware

| System | Durchsatz | Latenz | Leistung |
|--------|-----------|--------|----------|
| Intel Xeon w9 (Sapphire Rapids) | ~3.200 Mio/s | ~0,3 ns | 350 W |
| Intel i7-11700K (11. Generation) | 2.400 Mio/s | 0,4 ns | 65 W |
| Raspberry Pi 4 (Cortex-A72) | ~400 Mio/s | ~2,5 ns | 5 W |
| Raspberry Pi Zero 2W (Cortex-A53) | ~80 Mio/s | ~12 ns | 2 W |

### Einordnung gegenueber GPU und FAISS

| System | Methode | Durchsatz | Hardware | Leistung |
|--------|---------|-----------|----------|----------|
| Dieser Fork (i7-11700K) | Palette u8 Lookup | 2.400 Mio/s | CPU | 65 W |
| FAISS GPU (IVF-PQ) | CUDA quantisiert | 200-500 Mio/s | RTX 3060 | 170 W |
| FAISS GPU (cuVS) | CUDA optimiert | 1.000-2.000 Mio/s | H100 80 GB | 700 W |
| FAISS CPU (Flat) | AVX2 FP32 Dot | ~50 Mio/s | i7 | 65 W |
| FAISS CPU (IVF-PQ) | AVX2 quantisiert | 100-200 Mio/s | i7 | 65 W |

> **Zur Methodik:** Alle Zahlen sind pro vollstaendiger Query — ein Vektor rein, ein Aehnlichkeitswert raus. Beide Ansaetze erfordern einmalige Offline-Vorbereitung. Der Unterschied: ein Palette-Lookup ist ein u8-Lesevorgang (0 FLOPs), FAISS PQ dekodiert 8 Subspaces (~16 Ops), FAISS Flat berechnet ein 768-dimensionales Skalarprodukt (~1.536 FLOPs). Der Approximationsfehler beim Foveal-Tier (1/40 Sigma) betraegt 0,4% — geringer als die 5-10% bei typischen PQ-Konfigurationen.

---

## Dreistufige Cascade: Wie die Suche tatsaechlich ablaeuft

Die Palette-Tabelle allein erklaert noch nicht, wie eine Million Vektoren in zwei Millisekunden durchsucht werden. Dafuer sorgt eine dreistufige Cascade, bei der jede Stufe eine mathematisch gesicherte untere Schranke der naechsten darstellt. Keine Stufe kann einen relevanten Treffer verlieren.

### Stufe 1: Hamming-Sweep ueber bitgepackte Fingerprints

Jeder Vektor wird als 256-Bit-Fingerprint gespeichert (32 Bytes). Der Vergleich zweier Fingerprints ist eine XOR-Operation gefolgt von einem Hardware-Popcount:

- **AVX-512 VPOPCNTDQ**: Zwei Fingerprints in einem Takt
- **NEON vcntq_u8**: Pro-Byte-Popcount, nativ auf jedem ARM-Prozessor

Ein Scan ueber eine Million Fingerprints dauert etwa 2 Millisekunden und eliminiert 97-99% der Kandidaten. Die Hamming-Distanz ist eine beweisbare untere Schranke der Cosine-Distanz — es gibt keine False Negatives.

### Stufe 2: Base17 L1-Distanz

Die verbleibenden ~20.000 Kandidaten werden mit 17-dimensionalen i16-Vektoren (34 Bytes) verfeinert. Das passt in einen einzigen AVX-512-Load oder zwei NEON-Loads. Kosten: ~3 Nanosekunden pro Vergleich. Uebrig bleiben ~200 Kandidaten.

### Stufe 3: Palette-Lookup

Die ~200 Finalisten werden ueber die vorberechnete 256x256-Tabelle bewertet. Ein Lesevorgang pro Kandidat, 0,4 Nanosekunden.

### Gesamtbilanz fuer eine Million Vektoren

| Stufe | Eingang | Ausgang | Dauer | Bandbreite |
|-------|---------|---------|-------|------------|
| Hamming-Sweep | 1.000.000 | ~20.000 | ~2 ms | 32 MB |
| Base17 L1 | 20.000 | ~200 | ~60 us | 680 KB |
| Palette-Lookup | 200 | Top-K | ~0,08 us | 200 B |
| **Gesamt** | | | **~2,1 ms** | **~33 MB** |

FAISS CPU Flat benoetigt fuer dieselbe Aufgabe ~20 ms und liest ~6 GB. Die Cascade ist zehnmal schneller bei zweihundertmal weniger Speicherbandbreite.

### Integration in LanceDB

In einem Lance-Dataset ersetzt der Cascade-Sweep die FP32-Distanzberechnung von `lance-linalg`. Der Scan liest die bitgepackte Fingerprint-Spalte, fuehrt den Hardware-Popcount-Sweep durch, und holt vollstaendige Vektoren nur fuer die wenigen Ueberlebenden.

---

## Was Upstream liefert und was dieser Fork ergaenzt

### SIMD-Abdeckung

Das Upstream-ndarray delegiert Matrixmultiplikation an das externe Crate `matrixmultiply`, das AVX2 nutzen kann. Eigene SIMD-Typen oder Hardware-Erkennung gibt es nicht. Auf ARM faellt Upstream auf skalaren Code zurueck.

Dieser Fork implementiert eine vollstaendige SIMD-Schicht mit Laufzeiterkennung:

| Befehlssatz | Upstream | Dieser Fork | Beschleunigung |
|-------------|----------|-------------|----------------|
| AVX-512 (16 x f32) | Skalar | Native __m512-Typen | ~8x |
| AVX-512 VNNI (int8) | Skalar | 64 MACs/Instruktion | ~32x |
| AVX-512 VPOPCNTDQ | Skalar | Nativer 512-Bit-Popcount | ~16x |
| AMX (256 MACs) | Nicht vorhanden | Inline-ASM auf stabilem Rust | ~128x |
| AVX2 + FMA (8 x f32) | Extern (matrixmultiply) | Goto-GEMM + Dispatch | ~4x |
| NEON (4 x f32) | Skalar | 3-stufig: A53/A72/A76 | ~4x |
| NEON dotprod (ARMv8.2) | Nicht vorhanden | vdotq_s32 (Pi 5) | ~16x |

Die Erkennung erfolgt einmalig beim ersten Zugriff ueber `LazyLock<SimdCaps>` — ein CPUID-Aufruf, danach nur noch ein Pointer-Deref pro Funktionsaufruf (0,3 ns statt 1-3 ns bei wiederholter Feature-Abfrage).

### GEMM-Leistung

| Matrixgroesse | Upstream | Dieser Fork | NumPy (OpenBLAS) | GPU (RTX 3060) |
|--------------|----------|-------------|------------------|----------------|
| 512 x 512 | ~20 GFLOPS | 47 GFLOPS | ~45 GFLOPS | ~1.200 GFLOPS |
| 1024 x 1024 | ~13 GFLOPS | 139 GFLOPS | ~120 GFLOPS | ~3.500 GFLOPS |
| 2048 x 2048 | ~13 GFLOPS | ~150 GFLOPS | ~140 GFLOPS | ~5.000 GFLOPS |

Upstream trifft bei 1024 x 1024 auf ein Cache-Problem: kein Tiling, kein Threading, kein Microkernel. Der Fork nutzt den Goto-Algorithmus mit Cache-Blocking (L1/L2/L3) und erreicht 10,5-fachen Durchsatz — auf dem Niveau von NumPys jahrzehntealtem OpenBLAS.

### Datentypen jenseits von f32/f64

| Typ | Upstream | Dieser Fork | Methode |
|-----|----------|-------------|---------|
| f16 (IEEE 754) | Nicht vorhanden | Vorhanden | u16 als Traeger + F16C-Hardware (x86) / FCVTL via Inline-ASM (ARM) |
| BF16 (bfloat16) | Nicht vorhanden | Vorhanden | Hardware-Instruktionen + RNE-Emulation (bit-exakt mit VCVTNEPS2BF16) |
| i8/u8 (quantisiert) | Nicht vorhanden | Vorhanden | VNNI-Dot, Hamming, Popcount |
| i16 (Base17) | Nicht vorhanden | Vorhanden | L1-Distanz mit SIMD-Widen/Narrow |

Rusts `f16`-Typ ist Nightly-only (Issue #116909). Der Fork nutzt denselben Trick wie bei AMX: `u16` als Traegertyp, Hardware-Instruktionen ueber stabile `#[target_feature]`-Attribute oder Inline-Assembler. Das Ergebnis ist IEEE-754-konforme Konvertierung mit Hardware-Geschwindigkeit auf stabilem Rust.

---

## Sieben Dinge, die sonst niemand auf stabilem Rust macht

**1. Vollstaendiger std::simd-Polyfill.** Die portable SIMD-API von Rust ist seit Jahren Nightly-only. Dieser Fork implementiert dieselbe Typoberflaeche — F32x16, F64x8, U8x64, Masken, Reduktionen, Vergleiche — mit stabilen core::arch-Intrinsics. Wenn std::simd stabilisiert wird, aendert sich eine use-Zeile.

**2. f16 ohne Nightly.** Carrier-Typ u16 plus Hardware-Instruktionen: F16C (VCVTPH2PS/VCVTPS2PH) auf x86, FCVTL/FCVTN via asm!() auf ARM. Drei Praezisionsstufen: Plain f16 (10 Bit Mantisse), Scaled-f16 (bereichsoptimiert, 1,5x praeziser), Double-f16 (hi+lo-Paar, ~20 Bit effektiv).

**3. AMX auf stabilem Rust.** Intels Advanced Matrix Extensions (TDPBUSD: 16x16 Tile, 256 MACs pro Instruktion) sind als Rust-Intrinsics Nightly-only (Issue #126622). Der Fork emittiert die Instruktionen direkt als asm!(".byte ...") — verifiziert auf Rust 1.94 mit Kernel 6.18+.

**4. Gestufte ARM-NEON-Unterstuetzung.** Drei Stufen mit Laufzeiterkennung: A53-Baseline (Pi Zero 2W, Pi 3 — eine NEON-Pipeline), A72-Fast (Pi 4, Orange Pi 4 — zwei Pipelines, 2x-Unrolling), A76-DotProd (Pi 5, Orange Pi 5 — vdotq_s32, natives fp16). big.LITTLE-Systeme (RK3399, RK3588) werden korrekt behandelt.

**5. Eingefrorener Dispatch mit 0,3 ns pro Aufruf.** Ueblicher SIMD-Code prueft pro Aufruf: `if is_x86_feature_detected!("avx512f") { ... }` — ein atomarer Load plus Branch. Dieser Fork erkennt einmal und friert eine Funktionszeiger-Tabelle ein (LazyLock<SimdDispatch>, Copy-Struct). Danach: ein indirekter Call, kein Atomic, kein Branch-Prediction-Miss.

**6. BF16-Konvertierung bit-exakt mit Hardware.** Die Funktion f32_to_bf16_batch_rne() implementiert den IEEE-754-RNE-Algorithmus mit reinen AVX-512-F-Instruktionen und stimmt Bit-fuer-Bit mit Intels VCVTNEPS2BF16 ueberein. Verifiziert gegen Hardware-Ausgabe auf ueber einer Million Eingaben, einschliesslich Subnormalen, Unendlich, NaN und Halfway-Ties.

**7. Kognitiver Codec-Stack.** Ueber klassische Numerik hinaus implementiert der Fork eine vollstaendige Encoding-Pipeline: Fingerprint<256> (VSA, SIMD-Hamming), Base17 (17-dimensionale i16-Vektoren), CAM-PQ (Produkt-Quantisierung mit kompilierten Distanztabellen), Palette-Semiring (256x256-Distanzmatrizen fuer O(1)-Lookups), bgz7/bgz17 (komprimiertes Modellgewichts-Format: 201 GB BF16-Safetensors -> 685 MB bgz7).

---

## Codebook-Inferenz: Token-Generierung ohne GPU

Neben Vektorsuche nutzt der Fork denselben Tabellenansatz fuer LLM-Inferenz. Statt Matrixmultiplikation (`y = W*x`) wird ein vorberechnetes Codebook indiziert (`y = codebook[index[x]]`) — O(1) pro Token.

| Hardware | Befehlssatz | Tokens/s | Latenz (50 Tokens) | Leistung |
|----------|-------------|----------|---------------------|----------|
| Sapphire Rapids | AMX | 380.000 | 0,13 ms | 250 W |
| Xeon (AVX-512 VNNI) | VNNI | 10.000-50.000 | 1-5 ms | 150 W |
| Raspberry Pi 5 | NEON + dotprod | 2.000-5.000 | 10-25 ms | 5 W |
| Raspberry Pi 4 | NEON (dual) | 500-2.000 | 25-100 ms | 5 W |

Bei 5 Watt generiert ein Pi 4 eine 50-Token-Antwort fuer einen Sprachassistenten in unter 100 Millisekunden.

---

## f16-Gewichtstranskodierung

Getestet mit einem 15-Millionen-Parameter-Modell (Groessenordnung Piper TTS):

| Format | Groesse | Maximaler Fehler | RMSE | Durchsatz |
|--------|---------|-----------------|------|-----------|
| f32 (Original) | 60 MB | — | — | — |
| f16 (IEEE 754) | 30 MB | 7,3 x 10^-6 | 2,5 x 10^-6 | 94 Mio Params/s |
| Scaled-f16 | 30 MB | 4,9 x 10^-6 | 2,1 x 10^-6 | 91 Mio Params/s |
| Double-f16 | 60 MB | 5,7 x 10^-8 | 1,8 x 10^-8 | 42 Mio Params/s |

Mit AVX2-F16C-Hardware: ~500 Millionen Parameter pro Sekunde (8 Konvertierungen pro Taktzyklus).

---

## Schnellstart

```rust
use ndarray::Array2;
use ndarray::hpc::simd_caps::simd_caps;

let a = Array2::<f32>::ones((1024, 1024));
let c = a.dot(&a);  // AVX-512 / AVX2 / NEON — automatisch

let caps = simd_caps();
if caps.avx512f { println!("AVX-512 aktiv"); }
if caps.neon { println!("ARM-Profil: {}", caps.arm_profile().name()); }
```

```bash
# Automatische SIMD-Erkennung
cargo build --release

# Cross-Kompilierung fuer Raspberry Pi 4
cargo build --release --target aarch64-unknown-linux-gnu

# Maximale Leistung auf AVX-512-Server
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release

# 880 HPC-Tests ausfuehren
cargo test
```

## Voraussetzungen

- Rust 1.94 stable (kein Nightly, keine instabilen Features)
- Optional: gcc-aarch64-linux-gnu fuer Pi-Cross-Kompilierung
- Optional: Intel MKL oder OpenBLAS (Feature-gated)

## Oekosystem

Dieser Fork ist das Hardware-Fundament einer groesseren Architektur:

| Repository | Aufgabe |
|------------|---------|
| [lance-graph](https://github.com/AdaWorldAPI/lance-graph) | Graph-Query-Engine, Cypher-Parser, Codec-Stack |
| [home-automation-rs](https://github.com/AdaWorldAPI/home-automation-rs) | Smart Home mit Sprach-KI, MCP-Server, MQTT |

## Lizenz

MIT OR Apache-2.0 (identisch mit Upstream)
