# SESSION: Llama 4 Scout BF16 — Stream-Index Shards 1-4

## MISSION

Shard 5 (18.2 GB) is DONE → 7.70 MB at 4,735× ratio.
Process the remaining 4 shards. Together with shard 5, this gives us
the full Llama 4 Scout 109B model compressed to bgz17.

## READ FIRST

```bash
# The streaming indexer and HTTP reader that already work:
cat src/hpc/gguf_indexer.rs    # stream_index_gguf(), project_row_to_base17()
cat src/hpc/http_reader.rs     # HttpRangeReader::with_chunk_size()
cat src/hpc/gguf.rs            # GGUF header/tensor parsing, BF16 dequant

# The shard 5 test that PASSED (at the bottom of gguf_indexer.rs):
grep -A 80 "test_stream_index_llama4_bf16_shard5" src/hpc/gguf_indexer.rs
```

Do NOT modify any existing code. Only ADD new test functions.

## SHARD MAP

```
Repo: unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF
Path: BF16/Llama-4-Scout-17B-16E-Instruct-BF16-{NNNNN}-of-00005.gguf

Shard 1:  48,940,000,000 bytes  (~48.94 GB)  layers 0-10 + embeddings
Shard 2:  49,960,000,000 bytes  (~49.96 GB)  layers 11-21
Shard 3:  48,660,000,000 bytes  (~48.66 GB)  layers 22-32
Shard 4:  49,790,000,000 bytes  (~49.79 GB)  layers 33-43
Shard 5:  18,220,000,000 bytes  (~18.22 GB)  layers 44-47 + output  ✓ DONE
─────────────────────────────────────────────────────────────────────
Total:   215,570,000,000 bytes  (~215.57 GB)
```

## WHAT TO BUILD

Add ONE test function that processes all 4 shards sequentially.
NOT 4 separate tests — one function, loop over shards, cleanup between.

```rust
#[test]
#[ignore] // Streams ~197 GB from HuggingFace — takes ~2 hours
fn test_stream_index_llama4_bf16_shards_1_to_4() {
    use super::super::http_reader::HttpRangeReader;
    use std::io::BufWriter;

    let repo = "unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF";
    
    let shards = [
        ("BF16/Llama-4-Scout-17B-16E-Instruct-BF16-00001-of-00005.gguf", 48_940_000_000u64),
        ("BF16/Llama-4-Scout-17B-16E-Instruct-BF16-00002-of-00005.gguf", 49_960_000_000u64),
        ("BF16/Llama-4-Scout-17B-16E-Instruct-BF16-00003-of-00005.gguf", 48_660_000_000u64),
        ("BF16/Llama-4-Scout-17B-16E-Instruct-BF16-00004-of-00005.gguf", 49_790_000_000u64),
    ];

    let mut grand_total_source: u64 = 0;
    let mut grand_total_compressed: u64 = 0;
    let mut grand_total_original: u64 = 0;    // f32 equivalent
    let mut grand_total_tensors: usize = 0;
    let mut grand_by_type: [(usize, u64, u64); 6] = [(0,0,0); 6];
    
    // Add shard 5 results (already measured)
    let shard5_source: u64 = 18_220_000_000;
    let shard5_compressed: u64 = 7_700_000;  // 7.70 MB
    let shard5_original: u64 = 36_440_000_000;  // ~36.44 GB f32 equivalent
    grand_total_source += shard5_source;
    grand_total_compressed += shard5_compressed;
    grand_total_original += shard5_original;

    for (i, (filename, size)) in shards.iter().enumerate() {
        let shard_num = i + 1;
        let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, filename);
        let out_path = format!("/tmp/llama4_scout_shard{}.bgz7", shard_num);

        eprintln!();
        eprintln!("━━━ Shard {}/5 ({:.2} GB) ━━━", shard_num, *size as f64 / 1e9);
        eprintln!("  URL: {}", url);

        // 256 MB chunks — fewer HTTP round trips
        let mut reader = HttpRangeReader::with_chunk_size(
            url.clone(), *size, 256 * 1024 * 1024
        );

        let out = std::fs::File::create(&out_path).expect("create output");
        let mut writer = BufWriter::new(out);

        let stats = stream_index_gguf(
            &mut reader,
            &mut writer,
            Some(&|name, layer_type, orig, comp| {
                let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                eprintln!("  {:60} {:12?} {:>12} → {:>8} ({:.0}×)",
                    name, layer_type, orig, comp, ratio);
            }),
        ).expect(&format!("stream_index_gguf shard {}", shard_num));

        drop(writer);
        let out_size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);

        // Per-shard summary
        eprintln!();
        eprintln!("  Shard {} result: {:.2} GB → {:.2} MB ({:.0}×)",
            shard_num, *size as f64 / 1e9, out_size as f64 / 1e6, stats.overall_ratio());
        eprintln!("  Tensors: {} indexed, {} skipped",
            stats.tensors_indexed, stats.tensors_skipped);
        eprintln!("  Downloaded: {:.2} GB", reader.bytes_downloaded() as f64 / 1e9);

        let type_names = ["Attention", "FeedForward", "Conv2D", "Norm", "Embedding", "Skip"];
        for (j, name) in type_names.iter().enumerate() {
            let (count, orig, comp) = stats.by_type[j];
            if count > 0 {
                let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                eprintln!("  {:<12} {:>3} tensors: {:>10.2} GB → {:>8.2} MB ({:.0}×)",
                    name, count, orig as f64 / 1e9, comp as f64 / 1e6, ratio);
                grand_by_type[j].0 += count;
                grand_by_type[j].1 += orig;
                grand_by_type[j].2 += comp;
            }
        }

        // Accumulate
        grand_total_source += *size;
        grand_total_compressed += out_size;
        grand_total_original += stats.original_bytes;
        grand_total_tensors += stats.tensors_indexed;

        // CLEANUP: remove output file to free disk for next shard
        // Keep the stats, drop the bytes
        if let Err(e) = std::fs::remove_file(&out_path) {
            eprintln!("  Warning: cleanup failed: {}", e);
        } else {
            eprintln!("  Cleaned up {} (disk freed for next shard)", out_path);
        }
        
        assert!(stats.tensors_indexed > 0,
            "shard {} should have indexed tensors", shard_num);
    }

    // Grand total (all 5 shards)
    eprintln!();
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    eprintln!("LLAMA 4 SCOUT 17B-16E — FULL MODEL (ALL 5 SHARDS)");
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    eprintln!("  Source (BF16):    {:.2} GB", grand_total_source as f64 / 1e9);
    eprintln!("  Original (f32):  {:.2} GB", grand_total_original as f64 / 1e9);
    eprintln!("  Compressed:      {:.2} MB", grand_total_compressed as f64 / 1e6);
    eprintln!("  Overall ratio:   {:.0}×", grand_total_original as f64 / grand_total_compressed as f64);
    eprintln!("  Tensors indexed: {}", grand_total_tensors);
    eprintln!();
    
    let type_names = ["Attention", "FeedForward", "Conv2D", "Norm", "Embedding", "Skip"];
    for (j, name) in type_names.iter().enumerate() {
        let (count, orig, comp) = grand_by_type[j];
        if count > 0 {
            let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
            eprintln!("  {:<12} {:>4} tensors: {:>10.2} GB → {:>8.2} MB ({:.0}×)",
                name, count, orig as f64 / 1e9, comp as f64 / 1e6, ratio);
        }
    }
    eprintln!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    // Sanity checks
    assert!(grand_total_tensors > 100, "should have many tensors across all shards");
    assert!(grand_total_compressed < 200_000_000, 
        "full model should be under 200 MB: was {} MB", grand_total_compressed / 1_000_000);
}
```

## CRITICAL CONSTRAINTS

1. **256 MB chunk size** — the HttpRangeReader.with_chunk_size() already supports this.
   Each shard ~49 GB = ~192 HTTP requests. Not 2250.

2. **CLEANUP between shards** — `std::fs::remove_file()` after recording stats.
   Otherwise 4 × shard output fills disk. We only need the NUMBERS, not the files.
   The final production run will write to a combined output file.

3. **DO NOT modify existing tests** — shard 5 test stays untouched.
   Add the new test function BELOW it in the same `mod tests` block.

4. **DO NOT modify stream_index_gguf() or project_row_to_base17()** — 
   these work. Shard 5 proved it. Touch nothing in the production code.

5. **Shard 5 stats hardcoded** — add shard 5's known numbers (7.70 MB output, 
   18.22 GB source) to the grand total WITHOUT re-downloading it.

## RUN COMMAND

```bash
cargo test test_stream_index_llama4_bf16_shards_1_to_4 \
    --release -- --ignored --nocapture 2>&1 | tee /tmp/llama4_full.log
```

Expect ~2 hours total. Each shard ~25-30 min (3× larger than shard 5's 9 min).
Peak RAM should stay under 1 GB throughout.

## EXPECTED OUTPUT

If shard 5's ratio (~4,735×) holds for the MoE-heavy shards 1-4:

```
Shard 1 (48.94 GB):  →  ~10 MB
Shard 2 (49.96 GB):  →  ~11 MB  
Shard 3 (48.66 GB):  →  ~10 MB
Shard 4 (49.79 GB):  →  ~11 MB
Shard 5 (18.22 GB):  →   7.7 MB  (measured)
──────────────────────────────
Total  (215.57 GB):  →  ~50 MB    at ~4,300×
```

The MoE expert layers in shards 1-4 (which contain the bulk of the 16 experts'
gate/up/down weights) should compress at 10,000-15,000× like shard 5 showed.
Attention layers at ~2,000×. Embedding layer in shard 1 might be lower ratio.

## AFTER THE RUN

1. Commit the test (even if running takes hours, commit the code first)
2. Copy the full output log to `.claude/knowledge/llama4_scout_full_results.md`
3. Push both

Do NOT skip the cleanup step. Do NOT run shards in parallel (RAM).
Do NOT modify anything in src/hpc/ except adding the test function.
