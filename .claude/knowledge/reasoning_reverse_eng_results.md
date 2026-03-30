# Qwen3.5 → Claude Opus Reasoning Scaffold Analysis

Generated: 2026-03-30
L1 threshold: 1 (Base17 golden-step projection at stride=16)

## Model Matrix

| ID | Repo | Shards | Path |
|---|---|---|---|
| qwen35_27b_base | Qwen/Qwen3.5-27B | 11 | safetensors BF16 |
| qwen35_27b_v1 | Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled | 11 | safetensors BF16 |
| qwen35_27b_v2 | Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2 | 11 | safetensors BF16 |
| qwen35_9b_base | Qwen/Qwen3.5-9B | 4 | safetensors BF16 |
| qwen35_9b_dist | Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled | 4 | safetensors BF16 |

## Training Data Composition

| Model | Opus 4.6 | Opus 4.5 | Qwen-self | Total |
|---|---|---|---|---|
| v1 | 3000× (nohurry) | 250× (TeichAI) | 700× (Jackrong) | ~3950 |
| v2 | 3000× + 10000× (Roman1111111) | 250× (TeichAI) | 700× (Jackrong) | ~13950 |
| 9B | Same as v1 + 27B distill cascade | 250× | 700× | ~3950+ |

## 4-Diff Results

### Diff 1: 27B base → v1 (Opus 4.5 + early 4.6, ~3950 samples)
10,845 / 3,880,960 rows shifted (0.3%) — 9 of 11 shards

| Projection | Shifted | Total | % |
|---|---|---|---|
| FfnGate | 6,396 | 1,131,520 | 0.6% |
| FfnUp | 3,677 | 1,131,520 | 0.3% |
| Q | 608 | 208,896 | 0.3% |
| O | 40 | 20,480 | 0.2% |
| FfnDown | 93 | 332,800 | 0.0% |
| K | 0 | — | 0.0% |
| V | 0 | — | 0.0% |
| Embedding | 0 | 248,320 | 0.0% |

### Diff 2: 27B base → v2 (Opus 4.6 heavy, ~13950 samples)
1,921 / 5,241,695 rows shifted (0.0%) — 11 shards

| Projection | Shifted | Total | % |
|---|---|---|---|
| FfnGate | 982 | 1,131,520 | 0.1% |
| FfnUp | 707 | 1,131,520 | 0.1% |
| Q | 142 | 208,896 | 0.1% |
| O | 20 | 87,040 | 0.0% |
| K | 3 | 17,408 | 0.0% |
| V | 7 | 17,408 | 0.0% |
| Embedding | 0 | 251,777 | 0.0% |

### Diff 3: 27B v1 → v2 (iteration delta)
11,509 / 5,202,783 rows shifted (0.2%) — 11 shards

| Projection | Shifted | Total | % |
|---|---|---|---|
| FfnGate | 6,042 | 1,131,520 | 0.5% |
| FfnUp | 3,907 | 1,131,520 | 0.3% |
| Q | 664 | 208,896 | 0.3% |
| O | 185 | 81,920 | 0.2% |
| K | 56 | 17,408 | 0.3% |
| V | 51 | 17,408 | 0.3% |
| Embedding | 0 | 251,777 | 0.0% |

### Diff 4: 9B base → distilled
7,577 / 2,451,295 rows shifted (0.3%) — 4 shards

| Projection | Shifted | Total | % |
|---|---|---|---|
| FfnGate | 3,857 | 405,504 | 1.0% |
| FfnUp | 2,437 | 405,504 | 0.6% |
| Q | 416 | 73,728 | 0.6% |
| O | 170 | 36,864 | 0.5% |
| K | 49 | 9,216 | 0.5% |
| V | 47 | 9,216 | 0.5% |
| Embedding | 0 | 251,777 | 0.0% |

## Key Findings

### 1. The reasoning scaffold lives in SwiGLU FFN gating
FfnGate is the dominant shift in ALL 4 diffs. Not attention Q/K/V/O.
The LoRA distillation primarily teaches the model HOW to route information
through its feed-forward network, not how to attend differently.

### 2. v2 is a REVERT, not an upgrade
- base→v1: 0.6% FfnGate (aggressive modification)
- base→v2: 0.1% FfnGate (conservative — much closer to base)
- v1→v2:   0.5% FfnGate (v2 undid most of v1's changes)

v2's 14K additional Opus-4.6 samples didn't amplify v1's changes — they
**stabilized the optimizer back toward base**. v2 is closer to base than v1.

### 3. K stable at 27B, K shifted at 9B (capacity split)
- 27B: K=0.0% → knowledge base preserved, only routing changed
- 9B:  K=0.5% → knowledge must also change (insufficient capacity)

At 27B, the model learns new routing without touching its knowledge.
At 9B, it must rewrite both. This is the capacity-dependent split.

### 4. v1 is the control experiment (not redundant)
v1 vs v2 separates traits:

| Category | Definition | Interpretation |
|---|---|---|
| GOOD (v1 ∩ v2 ∩ 9B) | All three agree | Scale-invariant reasoning scaffold |
| BEHAVIOR (v1 \ v2) | v1 only, v2 reverted | Opus 4.5 behavioral traits |
| REASONING (v2 \ v1) | v2 only, not in v1 | Pure Opus 4.6 signal (but minimal) |
| UNCERTAIN (v1 ∩ v2 \ 9B) | Both rounds, not 9B | 27B capacity-dependent |

### 5. The "orchestrator" insight
Qwen3.5-base had the knowledge. It lacked the orchestration.
The LoRA taught routing (FfnGate + Q), not knowledge (K + Embedding).
Claude-style reasoning = different FFN activation patterns.
"Let me analyze this: 1... 2... 3..." is a routing pattern, not new knowledge.

## Architectural Implications

### Palette3D should prioritize FfnGate
The HEEL planes should weight FfnGate > FfnUp > Q > O.
K/V bits are informational at 27B (near-zero), critical at 9B.

### L1-metric palette, not POPCNT bitmask
Base17 fingerprints are not random — they are structured golden-step projections.
POPCNT (Hamming distance) requires random bit distribution → gives biased results.
Use Base17 L1 distance (PaletteSemiring) for all palette operations.

### Shallow vs deep thinking maps to HHTL levels
- HEEL (9B palette, 512 bytes): shallow/fast routing
- TWIG (27B palette, Sparse256): deep/analytical routing
- Style ordinal in PAL8 header controls escalation threshold

## Next Steps
1. Run inference on all 5 models with same prompts
2. NARS-score output quality per head (dynamic validation)
3. Self-reinforcement LoRA guided by quality-scored Palette3D
4. Validate: Q8_0 + Palette overlay vs BF16 reference
