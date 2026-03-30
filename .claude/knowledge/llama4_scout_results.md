# Llama 4 Scout 17B-16E — Full Model Compression Results

## Pipeline

BF16-direct → golden-step Base17 → bgz7 container

- `stream_index_gguf_bf16()` with `octave_stride=16`
- F64x8 SIMD: 8 rows projected in parallel per zmm register
- Halftone drop: 9 of 17 golden-step positions, odd bins interpolated
- No f32 intermediate allocation (BF16 → f64 inline)
- Reusable u16 buffer across all tensors

## Results

| Shard | Source (BF16) | Compressed | Ratio |
|-------|---------------|------------|-------|
| 1 | 48.94 GB | 11.77 MB | 4,159× |
| 2 | 49.96 GB | 8.32 MB | 6,005× |
| 3 | 48.66 GB | 5.57 MB | 8,736× |
| 4 | 49.79 GB | 4.52 MB | 11,016× |
| 5 | 18.22 GB | 7.70 MB | 2,366× |
| **Total** | **215.57 GB** | **37.88 MB** | **5,693×** |

## Observations

- Shard 1 (embeddings + early layers): larger output due to embedding table
- Shards 3-4 (middle MoE layers): highest ratios — expert weights are
  highly structured, golden-step averaging captures the per-expert identity
  in 34 bytes per row
- Shard 5 (final layers + output head): lower ratio — output projection
  has more variance than interior MoE expert weights

## Location

`src/hpc/openchat/weights/llama4_scout_shard{1-5}.bgz7`

## Implications for Maverick

Maverick has 128 experts (8× Scout). The MoE layers dominate even more.
If the per-expert ratio holds (~6,000-11,000× on interior shards),
Maverick's 801 GB could compress to 90-180 MB.

Conservative estimate: ~300 MB (if embedding/attention layers scale worse).
Optimistic estimate: ~90 MB (if expert sparsity is even higher with 128E).
