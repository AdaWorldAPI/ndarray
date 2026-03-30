# Qwen3.5 → Claude-4.6-Opus Reasoning Scaffold Analysis

Generated: 2026-03-30
L1 threshold: 1

## Model Matrix

| ID | Repo | Shards | Path |
|---|---|---|---|
| qwen35_27b_base | Qwen/Qwen3.5-27B | 11 | safetensors BF16 |
| qwen35_27b_v1 | Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled | 11 | safetensors BF16 |
| qwen35_27b_v2 | Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2 | 11 | safetensors BF16 |
| qwen35_9b_base | Qwen/Qwen3.5-9B | 4 | safetensors BF16 |
| qwen35_9b_dist | Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled | 4 | safetensors BF16 |

## Diff Summary

- **27B base→v1**: 10845/3880960 rows shifted (0.3%), 278 tensors
- **27B base→v2**: 0/0 rows shifted (0.0%), 0 tensors
- **27B v1→v2**: 0/0 rows shifted (0.0%), 0 tensors
- **9B base→dist**: 0/0 rows shifted (0.0%), 0 tensors

## Reasoning Scaffold

- **Scale-invariant blocks (27B∩9B)**: []
- **Capacity-dependent (27B only)**: []
- **Converged (v1∩v2)**: []

## NARS Revised Truth Per Projection

| Projection | Frequency | Confidence | Interpretation |
|---|---|---|---|
| FfnGate | 0.006 | 0.990 | STABLE |
| FfnUp | 0.003 | 0.990 | STABLE |
| Q | 0.003 | 0.990 | STABLE |
| O | 0.002 | 0.990 | STABLE |
| FfnDown | 0.000 | 0.990 | STABLE |
| Other | 0.000 | 0.990 | STABLE |
| Embedding | 0.000 | 0.990 | STABLE |

## Top 20 Shifted Heads (base→v1)

| Block | Projection | Shifted/Total | Mean L1 |
|---|---|---|---|
| 32 | FfnDown | 3/2841 | 2 |
| 31 | Q | 24/11756 | 2 |
| 9 | Other | 4/5024 | 2 |
| 35 | Q | 32/11065 | 2 |
| 42 | FfnDown | 5/4793 | 2 |
| 3 | O | 12/4979 | 2 |
| 19 | FfnUp | 30/17032 | 2 |
| 5 | FfnUp | 39/17096 | 2 |
| 63 | Q | 32/12176 | 2 |
| 21 | FfnGate | 114/17267 | 2 |
| 23 | Q | 17/11902 | 2 |
| 48 | FfnGate | 107/17277 | 2 |
| 20 | FfnGate | 110/17350 | 2 |
| 10 | FfnGate | 46/17135 | 2 |
| 26 | FfnGate | 129/17330 | 2 |
| 42 | FfnGate | 93/17069 | 2 |
| 55 | Q | 28/11089 | 2 |
| 30 | FfnGate | 132/17407 | 2 |
| 22 | FfnUp | 76/16654 | 2 |
| 34 | FfnGate | 96/17363 | 2 |
