# VERIFY: bgz17 Model Compression — Gegenprüfung aller Behauptungen

## MISSION

Systematische Verifizierung der bgz17 Kompressions-Pipeline gegen:
1. Bestehenden Code in ndarray + lance-graph (WAS EXISTIERT, WAS FUNKTIONIERT)
2. State-of-the-Art Quantisierung (GGUF Q4_K_M, GPTQ, AWQ, QuIP#, TurboQuant)
3. Theoretische Grenzen (Shannon, Rate-Distortion, Johnson-Lindenstrauss)

Du bist Gutachter, nicht Entwickler. Du prüfst Behauptungen gegen Evidenz.
Du baust nichts Neues bevor du nicht bewiesen hast dass das Bestehende funktioniert.

## P0 EISERNE REGEL

**LIES DEN CODE BEVOR DU URTEILST.**

```bash
# ERST lesen, DANN bewerten. Keine Ausnahmen.
cat crates/bgz17/src/base17.rs
cat crates/bgz17/src/distance_matrix.rs
cat crates/bgz17/src/bridge.rs
cat crates/bgz17/src/generative.rs
cat crates/bgz17/src/similarity.rs
cat crates/lance-graph/src/graph/blasgraph/hdr.rs    # Cascade, HHTL
find . -name "*.rs" | xargs grep -l "euler\|fibonacci\|rotation\|palette\|codebook"
find . -name "*.rs" | xargs grep -l "gguf\|quantiz\|compress"
cargo test --workspace 2>&1 | tail -30
```

Wenn eine Datei nicht existiert → die Behauptung ist NICHT implementiert.
Wenn ein Test fehlschlägt → die Behauptung ist NICHT bewiesen.
Wenn der Code anders funktioniert als behauptet → DOKUMENTIERE DIE ABWEICHUNG.

**LÖSCHE NICHTS.** Auch nicht wenn es falsch aussieht. Dokumentiere was falsch ist
und warum, aber lass den Code stehen bis die Korrektur GETESTET ist.

## BEHAUPTUNG 1: Kompressions-Ratios

Behauptet wird:
```
GPT-2:     ~500 MB → 1.7 MB     (~300×)
Jina:      ~150 MB → 760 KB     (~200×)
BERT:      ~440 MB → 800 KB     (~550×)
```

### Gegenprüfung:

```bash
# A) Finde die tatsächlichen Ausgabe-Dateien
find . -name "*.bgz17" -o -name "*.b17" -o -name "*compressed*" | xargs ls -lh
find . -name "*.rs" | xargs grep -l "gpt2\|bert\|jina" 

# B) Finde die Tests die das beweisen
grep -rn "assert.*size\|assert.*bytes\|assert.*ratio\|assert.*compress" --include="*.rs"

# C) Finde die Benchmarks
find . -name "*.rs" -path "*/bench*" | xargs grep -l "compress\|ratio"

# D) Vergleiche mit Referenz: Was ist die Q4_K_M Größe dieser Modelle?
#    GPT-2 124M params × 4 bits = 62 MB (Q4). Behauptung: 1.7 MB = 36× besser als Q4.
#    Ist das physikalisch möglich? 
#    Shannon: H(X) = Σ p(x) log2(1/p(x))
#    Wenn die Weight-Verteilung das hergibt, ja. MESSE die tatsächliche Entropie.
```

### Was wäre die Widerlegung?
- Wenn die 1.7 MB nicht ausreichen um Inferenz-Ergebnisse zu reproduzieren
- Wenn die Perplexity bei Dekompression >5% ansteigt vs Original
- Wenn die 1.7 MB ein Codebook PLUS externes Wörterbuch brauchen das nicht mitgezählt wurde

## BEHAUPTUNG 2: Euler-Gamma Rotation + Fibonacci Codebook

Behauptet wird:
- Euler-Gamma Rotation dreht Weight-Matrizen in eine Basis wo Fibonacci-Codebook
  die Redundanz besser erfasst als Hadamard (TurboQuant) oder Random Rotation (QuIP#)
- Fibonacci macht Bits NICHT-uniform → kein POPCOUNT nötig → vpshufb Table Lookup
- 3σ Separation zwischen Codebook-Einträgen → 99.73% korrekte Zuordnung

### Gegenprüfung:

```bash
# A) Existiert die Rotation?
grep -rn "euler.*gamma\|gamma.*rotation\|EulerGamma" --include="*.rs"
grep -rn "fibonacci\|zeckendorf\|fib_encode\|fib_decode" --include="*.rs"

# B) Existiert das Codebook?
grep -rn "codebook\|Codebook\|palette.*size\|num_clusters" --include="*.rs"
# Was ist die tatsächliche Codebook-Größe? 256? 4096? Dynamisch?

# C) Existiert der 3σ Beweis?
grep -rn "sigma\|separation\|3.*sigma\|three.*sigma" --include="*.rs" --include="*.md"
# Gibt es einen Test der die Separation misst?

# D) Vergleiche mit TurboQuant (Google, 2025):
#    TurboQuant: Hadamard → gleichmäßige Verteilung → POPCOUNT effizient
#    bgz17: Euler-Gamma → konzentrierte Verteilung → vpshufb effizient
#    FRAGE: Auf welchen Weight-Verteilungen gewinnt welcher Ansatz?
#    MESSE: Nimm einen echten Attention-Layer, rotiere mit beiden, vergleiche MSE.

# E) Vergleiche mit QuIP# (Cornell, 2024):
#    QuIP#: Random orthogonal rotation → incoherence → uniform quantization
#    bgz17: Strukturierte Rotation → coherence ERHALTEN → non-uniform quantization
#    Das ist ein fundamentaler Designunterschied. Wer hat Recht?
#    MESSE: Gleicher Layer, beide Ansätze, vergleiche Reconstruction Error.
```

### Was wäre die Widerlegung?
- Wenn Hadamard-Rotation auf den selben Weights gleiche oder bessere MSE liefert
- Wenn die Fibonacci-Codierung keinen messbaren Vorteil gegenüber linearer Quantisierung hat
- Wenn die 3σ Separation nur für bestimmte Layer-Typen gilt (Attention ja, Conv2D nein)

## BEHAUPTUNG 3: HHTL Cascade 90% Early Exit

Behauptet wird:
- HEEL/HIP/TWIG/LEAF Cascade verwirft 90% pro Stage
- O(1) Lookup durch Palette-Index statt O(n) Sweep

### Gegenprüfung:

```bash
# A) Existiert die Cascade?
cat crates/lance-graph/src/graph/blasgraph/hdr.rs | head -100
grep -n "cascade\|Cascade\|stage\|early_exit\|reject" crates/lance-graph/src/graph/blasgraph/hdr.rs

# B) Ist die 90% Rejection gemessen oder behauptet?
grep -rn "rejection_rate\|reject.*percent\|90\|0\.9" --include="*.rs" -A2
# Gibt es einen Benchmark der die tatsächliche Rejection Rate misst?

# C) Ist es O(1)?
# O(1) durch Palette = HashMap<u8, Vec<VectorId>>
# Aber: Die Cascade BAUT den Index. Das ist O(n) Preprocessing.
# Zur Laufzeit: O(k) wo k = Anzahl Kandidaten nach Stage 1.
# Wenn 90% rejected → k = 0.1n. Das ist O(n/10), nicht O(1).
# KORREKTUR: O(1) gilt nur für den Palette-Lookup selbst,
# nicht für die gesamte Query. Dokumentiere den Unterschied.
```

## BEHAUPTUNG 4: Inferenz OHNE Dekompression (Compose Tables)

Behauptet wird:
- Palette Compose Table (256×256×1 Byte = 64 KB) ermöglicht
  Multi-Hop Graph-Traversal komplett im komprimierten Space
- compose[a][b] gibt den Palette-Index der Komposition zurück
- Kein Dekomprimieren nötig für Nearest-Neighbor oder Traversal

### Gegenprüfung:

```bash
# A) Existiert die Compose Table?
grep -rn "compose\|ComposeTable\|compose_table" --include="*.rs"
grep -rn "semiring\|Semiring" --include="*.rs"

# B) Ist die Komposition assoziativ? (Semiring-Eigenschaft)
# compose(compose(a,b), c) == compose(a, compose(b,c)) ?
# Gibt es einen Property Test dafür?
grep -rn "proptest\|quickcheck\|associativ" --include="*.rs"

# C) Wie groß ist der Approximationsfehler?
# |compose(a,b) - quant(exact_compose(deq(a), deq(b)))| ≤ ε
# Ist ε gemessen? Für welche Operationen?

# D) Funktioniert das für LLM-Inferenz?
# Matmul in Palette Space = was genau?
# Input-Vektor × Weight-Matrix: 
#   Input muss erst quantisiert werden → Palette-Index
#   Dann compose mit jedem Weight-Palette-Index
#   Das ist O(n) Compose-Lookups, nicht O(n²) float ops
#   ABER: Die Akkumulation der Compose-Ergebnisse?
#   Majority Vote? Oder Lookup in einer zweiten Tabelle?
#   FINDE den Code der das tatsächlich macht.
```

### Was wäre die Widerlegung?
- Wenn der Compose-Fehler für Multi-Hop > 3 Hops divergiert
- Wenn Matmul-in-Palette-Space Perplexity um >10% verschlechtert
- Wenn die Compose Table bei >256 Palette-Einträgen zu groß wird (4096² = 16 MB)

## BEHAUPTUNG 5: Streaming GGUF Slicer (2 GB RAM für 200 GB Modell)

Behauptet wird:
- GGUF Tensor-für-Tensor lesen, komprimieren, schreiben
- Peak RAM = größter Tensor + Pipeline-Buffers ≈ 2 GB
- Funktioniert für Llama 4 Scout (55 GB GGUF) auf Railway Starter

### Gegenprüfung:

```bash
# A) Existiert der Slicer?
find . -name "*.rs" | xargs grep -l "gguf\|Gguf\|GGUF"
# Vermutlich: NEIN. Das ist noch nicht implementiert.
# Wenn nein → DOKUMENTIERE was fehlt, implementiere NICHTS ohne Test-First.

# B) Ist die Annahme korrekt dass GGUF sequentiell gelesen werden kann?
# GGUF Header enthält Tensor-Offsets → ja, mmap + seek ist möglich.
# ABER: Manche Tensoren haben Abhängigkeiten (Layer Norm braucht Weight + Bias).
# Reicht es wirklich EINEN Tensor im RAM zu haben?

# C) Prüfe: Was ist der größte Tensor in Llama 4 Scout?
# Expert FFN: hidden_dim × intermediate_dim × num_experts
# 5120 × 13824 × 16 = 1.13 Milliarden × 2 Bytes (BF16) = 2.26 GB
# DAS PASST NICHT IN 2 GB RAM.
# ABER: Die 16 Experts sind separate Tensoren im GGUF.
# Ein Expert: 5120 × 13824 × 2 = 141 MB. Das passt.
# VERIFIZIERE: Sind die Experts im GGUF als einzelne Tensoren gespeichert?
# grep "experts" im GGUF Header oder lese die llama.cpp Tensor-Namen.

# D) Referenz-Implementierung existiert:
# - gguf-py (Python): streaming reader
# - llama.cpp gguf.h (C): mmap reader  
# - Rust: https://crates.io/crates/gguf (prüfe ob brauchbar)
```

## BEHAUPTUNG 6: Conv2D Weights (SD 1.5) vs Attention Weights

Behauptet wird:
- Attention-Weights: hoch redundant → 200-300× Kompression
- Conv2D-Weights: räumlich strukturiert → weniger Kompression
- SD 3.5 Large ist ein DiT (Transformer) → gleiche Ratio wie LLMs

### Gegenprüfung:

```bash
# A) Messe die tatsächliche Entropie von Conv2D vs Attention Weights
# Lade SD 1.5 UNet, extrahiere:
#   - Einen Attention-Layer (Q,K,V Projection)  
#   - Einen Conv2D-Layer (3×3 Kernel)
# Berechne für beide:
#   - Singulärwert-Verteilung (SVD)
#   - Effektiver Rank
#   - Shannon-Entropie der Weight-Verteilung
#   - bgz17 Palette-Zuordnung: wie viele Cluster reichen für 99% Varianz?

# B) Verifiziere: Ist SD 3.5 Large wirklich ein reiner Transformer?
# SD 3.5 verwendet "MMDiT" (Multi-Modal Diffusion Transformer)
# ABER: Hat die VAE trotzdem Conv2D? (Ja, immer)
# Wie groß ist der DiT-Teil vs VAE-Teil?

# C) Prüfe: Gibt es publizierte Ergebnisse für Transformer-basierte
#    Diffusion Model Quantisierung?
# Flux GGUF existiert → welche Ratios erreichen die?
# Wenn Flux Q4 bei 4× liegt und wir behaupten 100×, was erklärt den Faktor 25?
```

## BEHAUPTUNG 7: Kahneman-Residual bei Bildverständnis

Behauptet wird:
- HEEL-Vektor (Archetyp "Vogel") abziehen → Residual ist das Einzigartige
- CHAODA erkennt dass "Vogel auf Zaun" eine Anomalie ist (niedrige Dichte)
- SPO-Zerlegung: S(Vogel) P(SITZT_AUF) O(Zaun) → Residual = 50 Bytes

### Gegenprüfung:

```bash
# A) Existiert der Tiny ImageNet Code?
find . -name "*.rs" -o -name "*.py" | xargs grep -l "imagenet\|ImageNet\|tiny.*image"
find . -name "*.rs" | xargs grep -l "quadrant\|Quadrant\|focus.*zone"

# B) Existiert die Entbündelung?
grep -rn "unbundle\|entbündel\|residual.*heel\|heel.*subtract" --include="*.rs"

# C) Existiert die CHAODA Integration?
grep -rn "chaoda\|CHAODA\|anomaly.*detect\|density.*anomal" --include="*.rs"
# CHAODA kommt aus dem CLAM Paper (Ishaq et al. 2021).
# Ist CLAM implementiert? 
grep -rn "clam\|Clam\|CLAM\|fractal.*dim\|local.*fractal" --include="*.rs"

# D) Die "50 Bytes" Behauptung:
# Wenn ein Residual-Vektor 4608 Dimensionen hat und davon 80% Noise sind,
# bleiben 920 Dimensionen × wie viele Bits?
# Bei 1 Bit/Dim = 115 Bytes. Bei bgz17 Palette = ~20-50 Bytes. Plausibel.
# ABER NUR wenn CHAODA die richtigen 80% als Noise identifiziert.
# Gibt es einen Ground-Truth Test dafür?
```

## BEHAUPTUNG 8: Parallele Mini-Queries statt KV Cache

Behauptet wird:
- 24 parallele 32-Token Queries statt 1× 4096-Token Query
- KV Cache: 24 × 32 × 32 KB = 24 MB statt 524 MB
- NARS Revision bündelt 24 Evidenzen zu einem Truth Value

### Gegenprüfung:
```
# FRAGE: Verliert man Information durch die Zerlegung?
# Ein 4096-Token Prompt hat Cross-Attention zwischen ALLEN Tokens.
# 24 separate 32-Token Queries haben KEINE Cross-Attention untereinander.
# Die Frage ist ob die SPO-Graph-Struktur die fehlende Cross-Attention ersetzt.
# 
# HYPOTHESE: Ja, weil die Graph-Kanten die Beziehungen explizit kodieren
# die Cross-Attention implizit lernen muss.
#
# TEST: Nimm eine Aufgabe die lange Kontexte braucht.
# Vergleiche: 1× full context vs 24× mini + NARS merge.
# Metrik: Task Accuracy, nicht Perplexity.
#
# ACHTUNG: Das ist ein FORSCHUNGSPROBLEM, kein Engineering-Problem.
# Nicht behaupten dass es funktioniert bevor es gemessen ist.
```

## PRÜFPLAN: Reihenfolge

```
Phase 1: CODE LESEN (keine Änderungen)
  □ Alle Dateien in crates/bgz17/src/ lesen und inventarisieren
  □ Alle Dateien in crates/lance-graph/src/graph/blasgraph/ lesen
  □ Alle Tests finden und ausführen
  □ Tabelle: Behauptung → Code-Datei → Test → Status (✓/✗/MISSING)

Phase 2: MESSEN (keine Änderungen am Produktionscode)
  □ Entropie-Messung: Attention vs Conv2D Weights
  □ Rejection Rate der Cascade: tatsächlich 90%?
  □ Compose Table Approximationsfehler
  □ Fibonacci vs Linear Codebook: MSE Vergleich
  □ Euler-Gamma vs Hadamard Rotation: MSE Vergleich

Phase 3: VERGLEICH MIT STATE OF ART
  □ Gleichen Layer durch GPTQ, AWQ, QuIP# quantisieren
  □ bgz17 Ratio vs GPTQ/AWQ/QuIP# Ratio bei gleichem Reconstruction Error
  □ Inference Speed: bgz17 Compose vs dequantize+matmul
  □ Tabelle: Method → Size → MSE → Inference Speed → Hardware

Phase 4: DOKUMENTATION (erst hier, wenn Phase 1-3 abgeschlossen)
  □ Was ist BEWIESEN (Code + Test + Messung)
  □ Was ist PLAUSIBEL (Code existiert, Test fehlt)
  □ Was ist BEHAUPTET (kein Code, keine Messung)
  □ Was ist WIDERLEGT (Test zeigt anderes Ergebnis)
```

## ABSCHLIESSENDE AUFFORDERUNG

**Bevor du IRGENDETWAS änderst, löschst, refactored, oder neu schreibst:**

1. Lies den bestehenden Code. Komplett. Nicht die ersten 20 Zeilen.
2. Führe die bestehenden Tests aus. Alle. Nicht "die wichtigen".
3. Miss die tatsächlichen Werte. Nicht schätzen. Messen.
4. Vergleiche mit dem publizierten State of Art. Nicht mit deiner Intuition.
5. Dokumentiere was du gefunden hast. Auch wenn es der Behauptung widerspricht.

**Lösche NICHTS bevor die Alternative GETESTET und BESSER ist.**

Der Code der heute kompiliert und Tests besteht ist wertvoller als
der Code der morgen "besser sein könnte". Jede Zeile die existiert
wurde aus einem Grund geschrieben. Finde den Grund bevor du löschst.

Wenn du eine Behauptung widerlegst: GRATULATION. Das ist wertvoller
als sie zu bestätigen. Schreib genau auf warum sie falsch ist, welche
Messung das zeigt, und was die korrekte Aussage wäre.

## REPOS IN SCOPE

```
READ + TEST:
  ndarray/crates/bgz17/           ← Palette, Rotation, Codebook, Distance
  lance-graph/crates/lance-graph/  ← Cascade, HHTL, Semiring, Compose
  ndarray/crates/lance-graph/      ← Stub oder real? Prüfen.

READ ONLY (Vergleich):
  https://github.com/ggerganov/llama.cpp/gguf.h    ← GGUF Format
  https://arxiv.org/abs/2401.xxxxx                  ← RaBitQ Paper
  https://arxiv.org/abs/2307.13304                  ← QuIP# Paper  
  https://arxiv.org/abs/2306.00978                  ← AWQ Paper
  Google TurboQuant Blog Post 2025

NICHT ANFASSEN:
  Alles außerhalb der oben genannten Pfade.
  Keine neuen Crates erstellen.
  Keine Cargo.toml ändern.
  Keine CI ändern.
```

## OUTPUT FORMAT

Erstelle am Ende eine Datei `.claude/knowledge/compression_verification.md`:

```markdown
# Compression Claims Verification — [DATUM]

## Verified ✓
| Claim | Code | Test | Measurement | vs State of Art |
|-------|------|------|-------------|-----------------|

## Plausible ~ (code exists, test missing)
| Claim | Code | What's Missing |
|-------|------|----------------|

## Unverified ? (no code found)  
| Claim | Expected Location | Status |
|-------|-------------------|--------|

## Falsified ✗ (measurement contradicts claim)
| Claim | Expected | Measured | Explanation |
|-------|----------|----------|-------------|
```
