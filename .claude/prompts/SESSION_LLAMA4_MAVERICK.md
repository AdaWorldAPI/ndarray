# SESSION: Llama 4 Maverick BF16 — Stream-Index All 18 Shards

## MISSION

Process all 18 BF16 shards of Llama 4 Maverick (17B-128E, 402B total params).
801.47 GB streamed through the BF16-direct indexer with F64x8 SIMD.

Scout (16 experts) compressed to 37.88 MB at 5,693×.
Maverick (128 experts) expected: 90–489 MB.

## READ FIRST

```bash
cat src/hpc/gguf_indexer.rs   # stream_index_gguf_bf16(), project_tensor_bf16_simd()
cat src/hpc/http_reader.rs    # HttpRangeReader::with_chunk_size()
cat src/hpc/gguf.rs           # GGUF header/tensor parsing

# The test is already written at the bottom of gguf_indexer.rs:
grep -A 5 "test_stream_index_llama4_maverick_bf16_all_shards" src/hpc/gguf_indexer.rs
```

Do NOT modify any existing code. The test function is already there.

## RUN COMMAND

```bash
cargo test test_stream_index_llama4_maverick_bf16_all_shards \
    --release -- --ignored --nocapture 2>&1 | tee /tmp/llama4_maverick_full.log
```

## PIPELINE (already implemented)

```
BF16 bytes → read_tensor_bf16_raw (reusable Vec<u16>, no f32 alloc)
           → project_tensor_bf16_simd (F64x8, 8 rows parallel)
             → project_8rows_bf16_simd (17 zmm accumulators)
               → gather_bf16_x8 (8 indexed u16 loads → F64x8)
               → strided octave (stride=16, 51 of 814 octaves)
               → halftone drop (9 of 17 golden positions)
               → interpolate odd bins from neighbors
             → project_1row_bf16_strided (scalar tail, n_rows % 8)
           → CompressedTensor::write_to (Base17 per row)
           → tail deletion (keep 3 most recent outputs)
```

## DISK BUDGET: 26 GB FREE

Output files are tiny (expected 5-27 MB each). Tail deletion keeps 3 most
recent, deletes older. Total output 90-489 MB. No disk pressure.

## SHARD MAP (18 shards, 801.47 GB)

```
Shard  1:  46.17 GB    Shard 10:  42.95 GB
Shard  2:  42.95 GB    Shard 11:  42.95 GB
Shard  3:  42.95 GB    Shard 12:  47.91 GB
Shard  4:  42.95 GB    Shard 13:  42.95 GB
Shard  5:  47.94 GB    Shard 14:  42.95 GB
Shard  6:  42.95 GB    Shard 15:  42.95 GB
Shard  7:  42.95 GB    Shard 16:  47.91 GB
Shard  8:  42.95 GB    Shard 17:  42.95 GB
Shard  9:  47.92 GB    Shard 18:  48.21 GB
```

128 MoE experts, interleaving Dense→MoE→Dense (every odd layer is MoE).

## EXPECTED RUNTIME

~8-10 hours total. Each shard ~25-30 min.
Peak RAM: ~142 MB (one reusable u16 buffer, largest tensor).
CPU: network-bound (97% fewer BF16→f64 conversions than f32 path).

## AFTER THE RUN

1. Copy log: `cp /tmp/llama4_maverick_full.log .claude/knowledge/`
2. Push results to `src/hpc/openchat/weights/llama4_maverick_shard{NN}.bgz7`
   (if output files were kept — otherwise just push the log)
3. Commit + push

Do NOT modify anything in src/hpc/ except adding results to knowledge/.
Do NOT run shards in parallel (RAM). Do NOT skip tail cleanup.
