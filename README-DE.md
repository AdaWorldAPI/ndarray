# ndarray — HPC-Erweiterung fuer Rust

*Fork von [rust-ndarray/ndarray](https://github.com/rust-ndarray/ndarray) mit 100 HPC-Modulen, 2,534 bestandenen Bibliothekstests und SIMD-Kernels von Intel AMX bis Raspberry Pi NEON. Laeuft auf stabilem Rust 1.99.0 ohne Nightly-Features.*

<sub>Zaehlungen bei Commit `f2c1aea`: `pub mod`-Eintraege in `src/hpc/mod.rs`; `cargo test --lib` (2,534 bestanden, 32 ignoriert). Wie jede Zahl auf dieser Seite ermittelt wurde: [Belege](#belege-fuer-die-zahlen-auf-dieser-seite).</sub>

[English Version](README.md) | [Kompletter Feature-Vergleich (146 Module)](COMPARISON.md)

---

## Worum geht es

Das Upstream-ndarray ist eine solide Bibliothek fuer n-dimensionale Arrays in Rust. Was es nicht liefert: hardwarenahe SIMD-Beschleunigung, BLAS ohne externe C-Bibliotheken und Unterstuetzung fuer Datentypen wie f16 oder BF16, die Rust auf einem stabilen Toolchain schlicht nicht anbietet.

Dieser Fork schliesst diese Luecken. Die Erweiterung umfasst rund 205,000 Zeilen Rust in 424 Dateien, die es upstream nicht gibt — von Goto-GEMM-Mikrokernels ueber ARM-NEON-Stufenerkennung bis zu einem Codec-Stack, der Cosine-Aehnlichkeit als Integer-Tabellen-Lookup implementiert.

Der Kerntrick in einer Zahl: eine Palette-Aehnlichkeit ist **ein Tabellenzugriff — etwa 0.84 ns, ~1.19 Milliarden Lookups pro Sekunde auf einem Kern** eines 2.8-GHz-Cascade-Lake-Xeon, ohne Fliesskomma-Arithmetik und ohne GPU (gemessen; siehe [Belege](#belege-fuer-die-zahlen-auf-dieser-seite)).

---

## Die zentrale Idee: Cosine-Aehnlichkeit ohne Fliesskomma

Vektorsuche in Datenbanken wie LanceDB oder FAISS berechnet fuer jeden Kandidaten ein Skalarprodukt: `dot(a,b) / (|a| * |b|)`. Bei 768 Dimensionen sind das 1,536 Fliesskomma-Operationen und 3 KB Kandidatendaten pro Vergleich.

Dieser Fork geht einen anderen Weg. Vektoren werden offline auf 256 Archetypes quantisiert. Die paarweisen Distanzen zwischen allen Archetypes sind in einer 256x256-Tabelle vorberechnet (`DistanceMatrix`, u16-Eintraege, 128 KB). Zur Laufzeit reduziert sich eine Cosine-Abfrage auf einen einzigen Tabellenzugriff.

### Gemessen

| Operation | Host | Ergebnis |
|-----------|------|----------|
| `DistanceMatrix::distance`, zufaellige Paare | Xeon @ 2.8 GHz (Cascade-Lake-Klasse, AVX-512 + VNNI), 1 Thread | 0.84 ns, ~1.19 G Lookups/s |
| `Base17::l1`, 20,000 Kandidaten | derselbe | je 3.04 ns, gesamt 60.7 µs |

Fruehere Fassungen dieser Seite nannten Raten pro Plattform (Sapphire Rapids ~3.2 G/s, i7-11700K 2.4 G/s, Raspberry Pi 4 ~400 M/s, Pi Zero 2W ~80 M/s) und einen Vergleich mit FAISS CPU/GPU und cuVS. Fuer diese Zahlen gibt es in diesem Repository keinen Benchmark, und die FAISS/GPU-Zahlen wurden hier nicht gemessen; sie werden daher nicht mehr als Ergebnisse angefuehrt. Ein fairer FAISS-Vergleich braeuchte dieselben Daten, dasselbe Recall-Ziel und dieselbe Hardware.

---

## Dreistufige Kaskade: Wie die Suche tatsaechlich funktioniert

Die Palette-Tabelle allein erklaert nicht, wie eine Million Vektoren schnell durchsucht wird. Das leistet eine dreistufige Kaskade, in der jede Stufe Kandidaten fuer die naechste aussortiert. Ob eine Stufe ein relevantes Ergebnis verlieren kann, haengt von der verwendeten Schranke ab; diese Garantie ist in diesem Repository noch nicht getestet.

### Stufe 1: Hamming-Sweep ueber bitgepackte Fingerprints

Jeder Vektor wird als bitgepackter Fingerprint gespeichert. Die folgende Kaskade geht von 32-Byte-Fingerprints (256 Bit) aus; zu beachten: der crate-eigene Typ `Fingerprint<256>` hat 256 *Woerter* — 2,048 Bytes. Der Vergleich zweier Fingerprints ist ein XOR gefolgt von einem Popcount:

- **AVX-512 VPOPCNTDQ**: nativer 64-Bit-Lane-Popcount, wo verfuegbar; sonst ein VPSHUFB-Lookup + VPSADBW (AVX-512 BW / AVX2)
- **NEON vcntq_u8**: Popcount pro Byte, nativ auf jedem ARM-Prozessor

Gemessen mit `bitwise::hamming_batch_raw` auf einem Kern des Cascade-Lake-Hosts (ohne VPOPCNTDQ): eine Anfrage gegen eine Million 32-Byte-Fingerprints dauert **15.5 ms**; gegen 2,048-Byte-Zeilen vom Typ `Fingerprint<256>` kostet es 277 ns pro Zeile (~7.4 GB/s). Die Aussortierungsrate haengt von den Daten und dem Schwellwert ab; sie ist hier nicht gemessen.

### Stufe 2: Base17-L1-Distanz

Die verbleibenden ~20,000 Kandidaten werden mit 17-dimensionalen i16-Vektoren (34 Bytes) verfeinert. Gemessene Kosten: 3.04 ns pro Vergleich (60.7 µs fuer 20,000). Etwa 200 Kandidaten ueberleben.

### Stufe 3: Palette-Lookup

Die ~200 Finalisten werden ueber die vorberechnete 256x256-Tabelle bewertet. Ein Zugriff pro Kandidat, 0.84 ns gemessen.

### Ende-zu-Ende: Eine Million Vektoren bis Top-K

| Stufe | Eingang | Ausgang | Dauer | Beleg |
|-------|---------|---------|-------|-------|
| Hamming-Sweep (32 B) | 1,000,000 | datenabhaengig | 15.5 ms | gemessen, 1 Kern, ohne VPOPCNTDQ |
| Base17 L1 | 20,000 | ~200 | 60.7 µs | gemessen |
| Palette-Lookup | 200 | Top-K | ~0.17 µs | 200 × 0.84 ns, abgeleitet |

Bei 32-Byte-Zeilen laeuft der Sweep mit 2.1 GB/s; der Overhead pro Zeile, nicht die Speicherbandbreite, ist also die Grenze (2,048-Byte-Zeilen erreichen 7.4 GB/s); Multi-Core-Skalierung ist hier nicht gemessen. Ein Ende-zu-Ende-Vergleich mit FAISS Flat wurde in diesem Repository nicht durchgefuehrt.

### Integration mit Lance

Die Kaskade ist ein Substratpfad, kein Lance-Index. In [lance-graph](https://github.com/AdaWorldAPI/lance-graph) wird die Hamming-Distanz von Bitvektoren als DataFusion-UDF `hamming_distance` bereitgestellt (sie ruft `bitwise::hamming_distance_raw` auf). Sie ist **nicht** in die ANN-Suche von Lance eingebunden, die weiterhin `lance-linalg`-Distanzen verwendet und fuer eine Hamming-Metrik einen Fehler zurueckgibt. Nichts hier ersetzt `lance-linalg` innerhalb eines Lance-Scans.

---

## Was Upstream bietet und was dieser Fork hinzufuegt

### SIMD-Abdeckung

Upstream-ndarray delegiert die Matrixmultiplikation an den externen Crate `matrixmultiply`, der AVX2 nutzen kann. Es hat keine eigenen SIMD-Typen und keine Hardwareerkennung. Auf ARM faellt Upstream auf Skalarcode zurueck.

Dieser Fork implementiert eine eigene SIMD-Schicht: 27 portable Vektor-/Maskentypen, zur Compile-Zeit ausgewaehlt (AVX-512, AVX2, NEON, WASM SIMD128, skalar oder Nightly-`core::simd`), dazu zur Laufzeit dispatchte Kernels ueber 7 Stufen (`amx_int8 > avx512vnni > avx512f > avxvnni > avx2_fma > neon > scalar`). Jede Stufe ist an das Instruktions-Feature gebunden, das ihr Kernel braucht; die Stufe `avxvnni` (VEX `VPDPBUSD`) ist an AVX-VNNI gebunden. Diese Stufe konnte auf dem Messhost nicht ausgefuehrt werden, der AVX-512 VNNI, aber nicht AVX-VNNI hat, und ihr Kernel wird nur anhand seiner emittierten Instruktionskodierung geprueft.

Was die Schicht bringt, ist pro Operation gegen eine benannte Baseline gemessen, auf einem Kern des Cascade-Lake-Hosts (Median aus 15 Laeufen, 1 M Elemente). Jedes Zeitpaar nennt zuerst den Fork-Wert, danach den Baseline-Wert:

| Operation | Baseline | Fork | Verhaeltnis |
|-----------|----------|------|-------------|
| u8-Vergleich → Bitmaske (`simd::eq_u8_to_mask`) | einfache Rust-Schleife | 0.029 vs 0.088 ns/elem | 3.1× |
| f32 → BF16 RNE (`f32_to_bf16_batch_rne`) | skalar pro Element | 0.196 vs 1.62 ns/elem | 8.3× |
| maskierte i32-Summe (`masked_sum_i32`, 50% Dichte) | einfache Bit-Test-Schleife | 0.53 vs 0.79 ns/elem | 1.5× |
| f32-Summe (`F32x16` + `reduce_sum`) | sequentielles `iter().sum()` | 0.128 vs 1.26 ns/elem | 9.8× |
| int8-GEMM u8×i8→i32 256³ (`gemm_u8_i8`, VNNI) | `int8_gemm_i32` (skalar) | 22.5 vs 5.9 GMAC/s | 3.8× |
| AMX-INT8-GEMM 2048³ | skalar | 169.7 GMAC/s | 600× (Emerald Rapids, [`AMX_GOTCHAS.md`](.claude/AMX_GOTCHAS.md)) |

Wo einfaches Rust bereits autovektorisiert, erreicht der Polyfill dasselbe, ohne es zu uebertreffen: Fused Multiply-Add, gechunkte f32-Summen, 64-Bit-Popcount und `popcount(a&b&c)` liegen alle innerhalb von ±20% der einfachen Schleife (LLVM emittiert denselben VPSHUFB-Popcount und VPTERNLOGQ). Die Instruktionsbreite (16 f32-Lanes, 64 VNNI-MACs) ist eine Obergrenze, kein Speedup.

Die Erkennung erfolgt einmalig ueber `LazyLock<SimdCaps>`. Auf diesem Host kostet ein wiederholtes `is_x86_feature_detected!` ~0.34 ns und eine `simd_caps()`-Kopie ~0.62 ns; der Gewinn des Einfrierens des Dispatch ist also vorhersagbarer Dispatch und eine Entscheidung pro Prozess, keine grosse Ersparnis pro Aufruf.

### GEMM-Leistung

Gemessen auf einem Kern des Cascade-Lake-Hosts (beste von 3–7 Laeufen, `matrixmultiply`-Threading aus):

| Matrixgroesse | `Array::<f32>::dot` | `Array::<f64>::dot` | `simd::gemm_f64_tiled_fma` |
|---------------|--------------------|--------------------|---------------------------|
| 512 × 512 | 70.8 GFLOPS | 34.3 GFLOPS | 9.7 GFLOPS |
| 1024 × 1024 | 70.1 GFLOPS | 34.3 GFLOPS | 9.1 GFLOPS |
| 2048 × 2048 | 65.0 GFLOPS | 32.7 GFLOPS | — |

`Array::dot()` ruft `matrixmultiply::sgemm`/`dgemm` auf (`src/linalg/impl_linalg.rs:503,522`) — dieselbe Engine, die Upstream nutzt; diese Spalten sind also kein Vergleich Fork gegen Upstream. Das forkeigene `gemm_f64_tiled_fma` (festes `TILE=64`, `F64x8`-Akkumulation) ist bei f64 derzeit ~3.6× langsamer als `matrixmultiply`. Eine fruehere Tabelle auf dieser Seite (Fork 47/139/~150 GFLOPS gegen Upstream 13–20, dazu NumPy- und RTX-3060-Spalten) hatte keinen Benchmark im Repository und wurde zurueckgezogen.

`simd_ops::array_chunks` durchlaeuft einen Slice als nicht ueberlappende `&[T; N]`-Fenster; `array_windows` ist das ueberlappende Gegenstueck (ein Stable-Rust-Aequivalent des Nightly-`slice::array_windows::<N>()`). Beide legen die Fenstergroesse an der Aufrufstelle fest, sodass sie direkt in `F32x16::from_array` / `F64x8::from_array` einfliesst, und beide sparen die Bounds-Pruefung pro Element, die eine dynamisch indizierte Schleife zahlt. Aktuelle In-Crate-Aufrufstellen: `hpc::blake3` (64-Byte-Block-Chunking) und `heel_f64x8::cosine_f32_to_f64_simd`, beide ueber `array_chunks`; `array_windows`, `array_windows_checked` und `array_chunks_checked` sind exportiert, haben aber noch keinen produktiven Aufrufer im Crate. Sie sind das Traversierungs-Primitiv, auf dem die handgeschriebenen BLAS-Graph-/bgz17-Kernels aufbauen, wo das Const-Generic-Fenster nahe an eine Cranelift-JIT-kompilierte innere Schleife herankam, ohne fuer einen JIT zu zahlen — siehe die Moduldokumentation in `src/simd_ops.rs`.

### Datentypen jenseits von f32/f64

| Typ | Upstream | Dieser Fork | Methode |
|-----|----------|-------------|---------|
| f16 (IEEE 754) | Nicht verfuegbar | Verfuegbar | u16-Traeger + F16C-Hardware (x86) / FCVTL ueber Inline-Asm (ARM) |
| BF16 (bfloat16) | Nicht verfuegbar | Verfuegbar | Hardware-Instruktionen + RNE-Emulation (bitgenau mit VCVTNEPS2BF16) |
| i8/u8 (quantisiert) | Nicht verfuegbar | Verfuegbar | VNNI-Dot, Hamming, Popcount |
| i16 (Base17) | Nicht verfuegbar | Verfuegbar | L1-Distanz mit SIMD-Widen/Narrow |

Rusts `f16`-Typ ist nur auf Nightly verfuegbar (Issue #116909). Der Fork nutzt denselben Ansatz wie bei AMX: `u16` als Traeger, Hardware-Instruktionen ueber stabile `#[target_feature]`-Attribute oder Inline-Assembler. Das Ergebnis ist IEEE-754-konforme Konvertierung mit Hardware-Geschwindigkeit auf stabilem Rust.

---

## Sieben Dinge, die sonst niemand auf stabilem Rust macht

**1. Ein std::simd-foermiger Polyfill auf Stable.** Rusts portable SIMD-API ist seit Jahren nur auf Nightly verfuegbar. Dieser Fork implementiert eine `std::simd`-artige Typoberflaeche — 27 Typen einschliesslich F32x16, F64x8, U8x64, Masken, Reduktionen und Vergleiche — auf stabilem `core::arch`, mit einem Nightly-`core::simd`-Backend hinter dem Feature `nightly-simd` und bitgenauen Paritaets-Crates in der CI (`simd-masking-parity`, ausgefuehrt unter AVX-512, AVX2, NEON via qemu und wasm). Es ist nicht die vollstaendige `std::simd`-API, und eine Methode, die auf einem Backend vorhanden ist, ist nicht auf allen garantiert; das Paritaetsprogramm prueft genau das, und die `F64x8`-Vergleiche fehlten auf AVX2 und NEON, bis sie dort hinzugefuegt wurden.

**2. f16 ohne Nightly.** Traegertyp u16 plus Hardware-Instruktionen: F16C (VCVTPH2PS/VCVTPS2PH) auf x86, FCVTL/FCVTN ueber asm!() auf ARM. Drei Genauigkeitsstufen: einfaches f16 (10-Bit-Mantisse), scaled-f16 (bereichsoptimiert, 1.5x genauer), double-f16 (Hi+Lo-Paar, ~20 Bit effektiv).

**3. AMX auf stabilem Rust.** Intels Advanced Matrix Extensions (TDPBUSD: eine 16×16-Kachel von Ausgaben ueber K=64 Bytes, 16,384 MACs pro Instruktion) sind als Rust-Intrinsics nur auf Nightly verfuegbar (Issue #126622). Der Fork emittiert sie ueber `asm!` — alle vier INT8-Formen (`tdpb{ss,su,us,uu}d`), BF16, FP16 und FP8 — und maß 169.7 GMAC/s single-threaded bei INT8 2048³ (Emerald Rapids, Kernel 6.18.5, [`AMX_GOTCHAS.md`](.claude/AMX_GOTCHAS.md)).

**4. Gestufte ARM-NEON-Erkennung.** Drei Stufen mit Laufzeiterkennung (auf aarch64 werden die portablen NEON-Typen verwendet; die dotprod-/BF16-Kernel-Stubs und die `simd_dispatch`-Tabelle leiten weiterhin an skalare Wrapper): A53-Basis (Pi Zero 2W, Pi 3 — einzelne NEON-Pipeline), A72 schnell (Pi 4, Orange Pi 4 — duale Pipeline, 2x Unrolling), A76 dotprod (Pi 5, Orange Pi 5 — vdotq_s32, natives fp16). big.LITTLE-Systeme (RK3399, RK3588) werden korrekt behandelt.

**5. Eingefrorener Dispatch.** Der Fork erkennt CPU-Features einmal und friert eine Funktionszeiger-Tabelle ein (`LazyLock`), sodass jeder spaetere Aufruf denselben indirekten Pfad nimmt. Die Ersparnis pro Aufruf ist auf aktuellen CPUs klein — ein gecachtes `is_x86_feature_detected!` kostet hier bereits ~0.34 ns —; der Wert liegt in einer Entscheidung pro Prozess und einer einzigen Stelle, die die gewaehlte Stufe benennt.

**6. BF16-Konvertierung bitgenau mit der Hardware.** Die Funktion f32_to_bf16_batch_rne() implementiert den IEEE-754-RNE-Algorithmus mit reinen AVX-512-F-Instruktionen und stimmt bitweise mit Intels VCVTNEPS2BF16 ueberein. Geprueft gegen die skalare RNE-Referenz und ein unabhaengiges f64-Orakel ueber **alle 4,294,967,296 f32-Bitmuster: 0 Abweichungen** (`cargo run --release --example bf16_rne_exhaustive`, 11.5 s auf 4 Threads des Cascade-Lake-Hosts). Der Vergleich ist exakte u16-Gleichheit, NaN-Vorzeichen und Payload-Bits zaehlen also mit. Ein Unit-Test vergleicht zusaetzlich mit der Hardware-Instruktion `VCVTNEPS2BF16` auf Hosts mit AVX-512-BF16; der Messhost hat keinen, dieser Vergleich wurde hier also nicht ausgefuehrt.

**7. Kognitiver Codec-Stack.** Ueber klassische Numerik hinaus implementiert der Fork eine vollstaendige Kodierungs-Pipeline: Fingerprint<256> (VSA, SIMD-Hamming), Base17 (17-dimensionale i16-Vektoren), CAM-PQ (Produktquantisierung mit kompilierten Distanztabellen), Palette-Semiring (256x256-Distanzmatrizen fuer O(1)-Lookups), bgz7/bgz17 (komprimiertes Modellgewichtsformat; eine Konvertierung von 201 GB BF16 → 685 MB wurde fuer die Release-Artefakte in lance-graph berichtet, in diesem Repository nicht reproduziert).

---

## Codebook-Inferenz: Token-Erzeugung ohne GPU

Ueber die Vektorsuche hinaus nutzt der Fork denselben Tabellenansatz fuer LLM-Inferenz. Statt Matrixmultiplikation (`y = W*x`) wird ein vorberechnetes Codebook indiziert (`y = codebook[index[x]]`) — O(1) pro Token. Eine hier zuvor gezeigte Tokens-pro-Sekunde-Tabelle (AMX 380,000 tok/s bis Pi 4 500–2,000 tok/s) hatte keinen Benchmark im Repository und wurde zurueckgezogen, bis sie reproduziert ist.

---

## f16-Gewichts-Transcodierung

Gemessen auf einem Kern des Cascade-Lake-Hosts (F16C), 15 Millionen gaussverteilte Gewichte (σ = 0.02):

| Format | Groesse | Maximaler Fehler | RMSE | Durchsatz |
|--------|---------|------------------|------|-----------|
| f32 (Original) | 60 MB | — | — | — |
| f16 (`cast_f32_to_f16_batch`) | 30 MB | 3.1 × 10⁻⁵ | 4.2 × 10⁻⁶ | 1,805 M Params/s |

Der Fehler haengt von der Gewichtsverteilung ab; scaled-f16 und double-f16 sind fuer engere Fehlergrenzen verfuegbar. Eine fruehere Tabelle (94/91/42 M Params/s) hatte keinen Benchmark im Repository.

---

## Schnellstart

```rust
use ndarray::Array2;
use ndarray::hpc::simd_caps::simd_caps;

let a = Array2::<f32>::ones((1024, 1024));
let c = a.dot(&a);  // matrixmultiply, as upstream

let caps = simd_caps();
if caps.avx512f { println!("AVX-512 active"); }
if caps.neon { println!("ARM profile: {}", caps.arm_profile().name()); }
```

```bash
# Portable / distribution build — x86-64-v3 (AVX2) baseline, runs on any
# Haswell-or-later x86_64. Pass the config EXPLICITLY: since 2026-09-16 the
# default is `target-cpu=native`, which tunes the artifact to the BUILD host
# and is not safe to ship (`.cargo/config-native.toml` says so in as many
# words). Runtime `simd_caps()` detection cannot rescue a binary whose
# baseline codegen already emits host-only instructions.
cargo --config .cargo/config-v3.toml build --release

# Build for THIS machine (dev / benchmarking). Fastest here, portable nowhere.
cargo build --release

# Cross-compile for Raspberry Pi 4
cargo build --release --target aarch64-unknown-linux-gnu

# Maximum performance on AVX-512 server
cargo --config .cargo/config-v4.toml build --release

# Library tests (2,534 at f2c1aea)
cargo test --lib
```

## Anforderungen

- Rust 1.99.0 stable (festgelegt in `rust-toolchain.toml`; kein Nightly, keine instabilen Features)
- Optional: gcc-aarch64-linux-gnu fuer Pi-Cross-Kompilierung
- Optional: Intel MKL oder OpenBLAS (Feature-gesteuert)

### Transitive Abhaengigkeiten des Features `std`

**Keine fuer Hashing.** BLAKE3 ist im Crate enthalten.

Die kognitiven Substratmodule unter `hpc/` — `plane`, `seal`,
`merkle_tree`, `vsa`, `spo_bundle`, `crystal_encoder`, `compression_curves`,
`deepnsm` — nutzen `hpc::blake3` fuer Integritaets-Hashing und XOF-Expansion. Das
ist eine portable, reine Rust-Transkription der BLAKE3-Referenzimplementierung,
die in diesem Crate ausgeliefert wird: kein SIMD, kein `unsafe`, kein C und kein
Build-Skript.

Fruehere Fassungen zogen hier den externen Crate **`blake3`** ein, zunaechst
hinter `hpc-extras` (was wiederkehrende "missing blake3"-Build-Fehler fuer
Konsumenten wie `burn-ndarray` verursachte, die
`default-features = false, features = ["std"]` waehlen), dann an `std` gebunden.
**Beides ist entfernt.** `blake3` und seine transitiven `constant_time_eq`,
`arrayref` und `arrayvec` erscheinen in keiner Feature-Kombination mehr im
Abhaengigkeitsgraphen, die Falle kann sich also nicht wiederholen.

Konsumenten, die mit `default-features = false` bauen (kein `std`, z. B. das
nostd-Target `thumbv6m-none-eabi`), ueberspringen das Modul `hpc` und damit den
BLAKE3-Code; das nostd-Linken bleibt unberuehrt.

## Belege fuer die Zahlen auf dieser Seite

| Zahl | Beleg |
|------|-------|
| 100 HPC-Module, 2,534 Lib-Tests, ~205k hinzugefuegte Zeilen / 424 Dateien | gezaehlt bei `f2c1aea` (`src/hpc/mod.rs`; `cargo test --lib`; Pfad-Diff gegen rust-ndarray `bd3ade9`) |
| 0.84 ns Palette-Lookup, 3.04 ns Base17 L1, 15.5 ms 1-M-Sweep, SIMD-Verhaeltnisse, GEMM, f16 | gemessen auf einem Kern eines Xeon @ 2.8 GHz (Family 6 Model 85, AVX-512 F/BW/VL/DQ/CD + VNNI; kein AMX, kein AVX-512-BF16, kein VPOPCNTDQ), Rust 1.98.1, `target-cpu=native`, Median aus 15 Laeufen, sofern nicht anders angegeben |
| BF16 RNE, alle 2³² Eingaben, 0 Abweichungen | `examples/bf16_rne_exhaustive.rs`: jedes u32-Bitmuster in 4 zusammenhaengenden Bereichen, Batches von 65,536 Eingaben durch den AVX-512F-Pfad; exakter u16-Vergleich gegen `f32_to_bf16_scalar_rne` und ein unabhaengiges f64-Nearest-Value-Orakel (quiet-forced NaN, DAZ, Ties to Even); reihenfolgeunabhaengige Ausgabe-Pruefsumme `0x5cd3eaa07f7f8080` (gleich fuer 2 und 4 Threads); 11.5 s, Rust 1.98.1. Ein absichtlich kaputtes Orakel (Ties weg von Null) meldet 32,512 Abweichungen, die Pruefung kann also fehlschlagen |
| AMX 169.7 GMAC/s, 600× skalar | gemessen auf Emerald Rapids, [`AMX_GOTCHAS.md`](.claude/AMX_GOTCHAS.md) |

## Oekosystem

Dieser Fork ist das Hardware-Fundament einer groesseren Architektur:

| Repository | Zweck |
|------------|-------|
| [lance-graph](https://github.com/AdaWorldAPI/lance-graph) | Cypher/SQL-Engine auf DataFusion, die spaltenorientierte Abfrageoberflaeche Quack, Codec-Stack. Verantwortet Graph-, Abfrage- und Ende-zu-Ende-Benchmarks; dieses Repository verantwortet Kernel- und Mikrobenchmark-Zahlen |
| [home-automation-rs](https://github.com/AdaWorldAPI/home-automation-rs) | Smart Home mit Sprach-KI, MCP-Server, MQTT |

## Lizenz

MIT OR Apache-2.0 (identisch mit Upstream)
