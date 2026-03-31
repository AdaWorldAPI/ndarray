//! Safetensors header parser — streaming support for the bgz17 indexer.
//!
//! Parses the safetensors JSON header and produces `GgufFile` + `TensorInfo`
//! so that `stream_index_gguf_bf16` works unchanged on safetensors files.
//!
//! ```text
//! Safetensors layout:
//!   [8 bytes]  u64 LE = header_size
//!   [header_size bytes]  JSON = tensor metadata
//!   [remaining bytes]  raw tensor data (contiguous, no padding)
//!
//! JSON structure:
//!   { "__metadata__": {...},
//!     "tensor_name": { "dtype": "BF16", "shape": [d0, d1], "data_offsets": [start, end] },
//!     ... }
//!
//! data_offsets are relative to the start of the data section (byte after JSON header).
//! ```
//!
//! The key advantage over GGUF for the reasoning diff pipeline:
//! safetensors stores full BF16 precision, while GGUF Q8_0 introduces
//! quantization noise. BF16→BF16 diff gives cleaner causal attribution.

use super::gguf::{GgufFile, TensorInfo, GgmlType};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

// ============================================================================
// Safetensors dtype → GgmlType mapping
// ============================================================================

fn parse_dtype(s: &str) -> Result<GgmlType, String> {
    match s {
        "BF16" | "bfloat16" => Ok(GgmlType::BF16),
        "F16" | "float16" => Ok(GgmlType::F16),
        "F32" | "float32" => Ok(GgmlType::F32),
        // Safetensors also supports I8, I16, I32, I64, F64, BOOL, U8, etc.
        // For weight indexing, we only care about float types.
        other => Err(format!("unsupported safetensors dtype: {}", other)),
    }
}

// ============================================================================
// JSON parser — minimal, no serde dependency
// ============================================================================

/// Parse a safetensors JSON header without serde.
///
/// We only need: tensor names, dtypes, shapes, and data_offsets.
/// The JSON is always a flat object of objects.
fn parse_safetensors_json(json: &str) -> Result<Vec<TensorInfo>, String> {
    let mut tensors = Vec::new();

    // Simple state-machine JSON parser for the safetensors format.
    // The format is always: { "name": { "dtype": "...", "shape": [...], "data_offsets": [a, b] }, ... }
    // We skip "__metadata__" entries.

    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err("invalid JSON: not an object".into());
    }

    // Find all tensor entries by scanning for "dtype" keys
    // This is a pragmatic parser — not a full JSON parser.
    let mut pos = 1; // skip opening {
    let bytes = json.as_bytes();
    let len = bytes.len();

    while pos < len - 1 {
        // Skip whitespace and commas
        while pos < len && (bytes[pos] == b' ' || bytes[pos] == b'\n' ||
                            bytes[pos] == b'\r' || bytes[pos] == b'\t' ||
                            bytes[pos] == b',') {
            pos += 1;
        }
        if pos >= len - 1 { break; }
        if bytes[pos] == b'}' { break; }

        // Read key (tensor name)
        if bytes[pos] != b'"' {
            pos += 1;
            continue;
        }
        let key_start = pos + 1;
        pos += 1;
        while pos < len && bytes[pos] != b'"' {
            if bytes[pos] == b'\\' { pos += 1; } // skip escaped char
            pos += 1;
        }
        let key = &json[key_start..pos];
        pos += 1; // skip closing "

        // Skip colon
        while pos < len && bytes[pos] != b':' { pos += 1; }
        pos += 1; // skip :

        // Skip whitespace
        while pos < len && (bytes[pos] == b' ' || bytes[pos] == b'\n' ||
                            bytes[pos] == b'\r' || bytes[pos] == b'\t') {
            pos += 1;
        }

        if key == "__metadata__" {
            // Skip the metadata object — find matching closing brace
            let depth_start = pos;
            if bytes[pos] == b'{' {
                let mut depth = 1;
                pos += 1;
                while pos < len && depth > 0 {
                    if bytes[pos] == b'{' { depth += 1; }
                    if bytes[pos] == b'}' { depth -= 1; }
                    if bytes[pos] == b'"' {
                        pos += 1;
                        while pos < len && bytes[pos] != b'"' {
                            if bytes[pos] == b'\\' { pos += 1; }
                            pos += 1;
                        }
                    }
                    pos += 1;
                }
            }
            continue;
        }

        // Parse tensor value object: { "dtype": "...", "shape": [...], "data_offsets": [...] }
        if bytes[pos] != b'{' {
            // Not an object — skip until next comma or closing brace
            while pos < len && bytes[pos] != b',' && bytes[pos] != b'}' { pos += 1; }
            continue;
        }

        // Find the closing brace for this tensor's object
        let obj_start = pos;
        let mut depth = 1;
        pos += 1;
        while pos < len && depth > 0 {
            if bytes[pos] == b'{' { depth += 1; }
            if bytes[pos] == b'}' { depth -= 1; }
            if bytes[pos] == b'"' {
                pos += 1;
                while pos < len && bytes[pos] != b'"' {
                    if bytes[pos] == b'\\' { pos += 1; }
                    pos += 1;
                }
            }
            pos += 1;
        }
        let obj_str = &json[obj_start..pos];

        // Extract dtype
        let dtype_str = extract_json_string(obj_str, "dtype").unwrap_or_default();
        let dtype = match parse_dtype(&dtype_str) {
            Ok(d) => d,
            Err(_) => continue, // skip unsupported dtypes
        };

        // Extract shape
        let shape = extract_json_array_u64(obj_str, "shape").unwrap_or_default();

        // Extract data_offsets
        let offsets = extract_json_array_u64(obj_str, "data_offsets").unwrap_or_default();
        let offset = if offsets.len() >= 1 { offsets[0] } else { 0 };

        tensors.push(TensorInfo {
            name: key.to_string(),
            dimensions: shape,
            dtype,
            offset,
        });
    }

    // Sort by offset for sequential reading
    tensors.sort_by_key(|t| t.offset);

    Ok(tensors)
}

/// Extract a string value for a key from a JSON object fragment.
fn extract_json_string(obj: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let pos = obj.find(&pattern)?;
    let after_key = &obj[pos + pattern.len()..];

    // Find colon then opening quote
    let colon = after_key.find(':')?;
    let rest = &after_key[colon + 1..];
    let quote1 = rest.find('"')?;
    let rest = &rest[quote1 + 1..];
    let quote2 = rest.find('"')?;

    Some(rest[..quote2].to_string())
}

/// Extract a u64 array value for a key from a JSON object fragment.
fn extract_json_array_u64(obj: &str, key: &str) -> Option<Vec<u64>> {
    let pattern = format!("\"{}\"", key);
    let pos = obj.find(&pattern)?;
    let after_key = &obj[pos + pattern.len()..];

    let bracket_open = after_key.find('[')?;
    let bracket_close = after_key.find(']')?;
    let array_str = &after_key[bracket_open + 1..bracket_close];

    let values: Vec<u64> = array_str.split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    Some(values)
}

// ============================================================================
// Header reader
// ============================================================================

/// Read a safetensors file header and produce a GgufFile-compatible struct.
///
/// The returned `GgufFile` has:
/// - `tensor_data_offset`: absolute byte offset where tensor data starts
/// - `tensors`: Vec<TensorInfo> with offsets relative to data start
/// - `version`: 0 (not a GGUF version)
/// - `alignment`: 1 (safetensors has no alignment padding)
pub fn read_safetensors_header<R: Read + Seek>(reader: &mut R) -> Result<GgufFile, String> {
    // Read header size (first 8 bytes, u64 LE)
    let mut size_buf = [0u8; 8];
    reader.read_exact(&mut size_buf).map_err(|e| format!("read header size: {}", e))?;
    let header_size = u64::from_le_bytes(size_buf);

    if header_size > 100_000_000 {
        return Err(format!("header_size {} too large (>100 MB)", header_size));
    }

    // Read JSON header
    let mut json_buf = vec![0u8; header_size as usize];
    reader.read_exact(&mut json_buf).map_err(|e| format!("read header JSON: {}", e))?;
    let json_str = String::from_utf8(json_buf).map_err(|e| format!("header not UTF-8: {}", e))?;

    // Parse tensors
    let tensors = parse_safetensors_json(&json_str)?;

    // Data starts immediately after the header
    let tensor_data_offset = 8 + header_size;

    eprintln!("  Safetensors: {} tensors, data at byte {}",
        tensors.len(), tensor_data_offset);

    Ok(GgufFile {
        version: 0,
        metadata: HashMap::new(),
        tensors,
        tensor_data_offset,
        alignment: 1,
    })
}

// ============================================================================
// Streaming indexer entry point
// ============================================================================

/// Stream-index a safetensors file through the BF16-direct pipeline.
///
/// This is a thin wrapper: parse safetensors header → produce GgufFile →
/// delegate to `stream_index_gguf_bf16`.
///
/// Why this matters: safetensors stores full BF16 weights (no quantization).
/// The GGUF Q8_0 path introduces 8-bit quantization noise before projection.
/// BF16→Base17 gives cleaner fingerprints for causal diffing.
pub fn stream_index_safetensors_bf16<R: Read + Seek, W: std::io::Write>(
    reader: &mut R,
    writer: &mut W,
    octave_stride: usize,
    callback: Option<&dyn Fn(&str, &super::gguf_indexer::LayerType, usize, usize)>,
) -> Result<super::gguf_indexer::IndexStats, String> {
    // Parse safetensors header (produces GgufFile-compatible struct)
    let header = read_safetensors_header(reader)?;

    // Delegate to the existing BF16-direct chunked indexer
    // The indexer uses: header.tensors, header.tensor_data_offset, tensor.offset, tensor.dtype
    // All of these are populated by read_safetensors_header identically to read_gguf_header.
    super::gguf_indexer::stream_index_gguf_bf16_with_header(
        reader, writer, &header, octave_stride, callback,
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dtype() {
        assert_eq!(parse_dtype("BF16").unwrap(), GgmlType::BF16);
        assert_eq!(parse_dtype("bfloat16").unwrap(), GgmlType::BF16);
        assert_eq!(parse_dtype("F16").unwrap(), GgmlType::F16);
        assert_eq!(parse_dtype("F32").unwrap(), GgmlType::F32);
        assert!(parse_dtype("I32").is_err());
    }

    #[test]
    fn test_parse_safetensors_json_minimal() {
        let json = r#"{
            "__metadata__": {"format": "pt"},
            "model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [151936, 3584],
                "data_offsets": [0, 1089470464]
            },
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "BF16",
                "shape": [3584, 3584],
                "data_offsets": [1089470464, 1115095040]
            }
        }"#;

        let tensors = parse_safetensors_json(json).unwrap();
        assert_eq!(tensors.len(), 2);

        // Sorted by offset
        assert_eq!(tensors[0].name, "model.embed_tokens.weight");
        assert_eq!(tensors[0].dimensions, vec![151936, 3584]);
        assert_eq!(tensors[0].dtype, GgmlType::BF16);
        assert_eq!(tensors[0].offset, 0);

        assert_eq!(tensors[1].name, "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(tensors[1].offset, 1089470464);
    }

    #[test]
    fn test_extract_json_helpers() {
        let obj = r#"{"dtype": "BF16", "shape": [3584, 3584], "data_offsets": [100, 200]}"#;
        assert_eq!(extract_json_string(obj, "dtype"), Some("BF16".into()));
        assert_eq!(extract_json_array_u64(obj, "shape"), Some(vec![3584, 3584]));
        assert_eq!(extract_json_array_u64(obj, "data_offsets"), Some(vec![100, 200]));
    }

    #[test]
    fn test_read_synthetic_safetensors() {
        use std::io::Cursor;

        // Build a minimal safetensors file in memory
        let json = r#"{"tensor_a": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [0, 64]}}"#;
        let json_bytes = json.as_bytes();
        let header_size = json_bytes.len() as u64;

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_size.to_le_bytes());
        file_bytes.extend_from_slice(json_bytes);
        // 64 bytes of BF16 data (4 rows × 8 cols × 2 bytes)
        file_bytes.extend_from_slice(&[0x3F, 0x80].repeat(32)); // 32 × BF16(1.0) = 0x3F80

        let mut cursor = Cursor::new(file_bytes);
        let header = read_safetensors_header(&mut cursor).unwrap();

        assert_eq!(header.tensors.len(), 1);
        assert_eq!(header.tensors[0].name, "tensor_a");
        assert_eq!(header.tensors[0].dimensions, vec![4, 8]);
        assert_eq!(header.tensors[0].dtype, GgmlType::BF16);
        assert_eq!(header.tensor_data_offset, 8 + header_size);
    }

    #[test]
    #[ignore] // Streams ~55 GB from HuggingFace
    fn test_stream_index_qwen35_safetensors() {
        use super::super::http_reader::HttpRangeReader;
        use std::io::BufWriter;

        let repo = "Qwen/Qwen3.5-27B";
        let shards = 11;

        for shard in 1..=shards {
            let filename = format!("model.safetensors-{:05}-of-{:05}.safetensors", shard, shards);
            let out_path = format!("/tmp/qwen35_27b_base_shard{:02}.bgz7", shard);

            if std::fs::metadata(&out_path).is_ok() {
                eprintln!("SKIP {} (exists)", out_path);
                continue;
            }

            let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, filename);
            eprintln!("Indexing shard {}/{}: {}", shard, shards, filename);

            // HEAD for size
            let size_str = std::process::Command::new("curl")
                .args(&["-sI", "-L", &url])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();
            let size: u64 = size_str.lines()
                .filter(|l| l.to_lowercase().starts_with("content-length:"))
                .last()
                .and_then(|l| l.split(':').nth(1))
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(6_000_000_000);

            let mut reader = HttpRangeReader::with_chunk_size(url, size, 256 * 1024 * 1024);
            let out = std::fs::File::create(&out_path).expect("create output");
            let mut writer = BufWriter::new(out);

            let stats = stream_index_safetensors_bf16(
                &mut reader, &mut writer, 16,
                Some(&|name, lt, orig, comp| {
                    let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                    eprintln!("  {:50} {:>12} → {:>8} ({:.0}×)", name, orig, comp, ratio);
                }),
            ).expect("safetensors indexing failed");

            drop(writer);
            eprintln!("  → {:.2} MB, {} tensors",
                std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0) as f64 / 1e6,
                stats.tensors_indexed);
        }
    }

    // ── HiDream-I1: DiT+MoE diffusion model ──

    /// Helper: index safetensors shards from a HuggingFace repo.
    fn index_safetensors_shards(
        repo: &str,
        filenames: &[&str],
        out_prefix: &str,
        octave_stride: usize,
    ) -> Vec<super::super::gguf_indexer::IndexStats> {
        use super::super::http_reader::HttpRangeReader;
        use std::io::BufWriter;

        let mut all_stats = Vec::new();

        for (i, filename) in filenames.iter().enumerate() {
            let shard = i + 1;
            let out_path = if filenames.len() == 1 {
                format!("{}.bgz7", out_prefix)
            } else {
                format!("{}_shard{:02}.bgz7", out_prefix, shard)
            };

            if std::fs::metadata(&out_path).is_ok() {
                eprintln!("SKIP {} (exists)", out_path);
                continue;
            }

            let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, filename);
            eprintln!("Indexing {}/{}: {}", shard, filenames.len(), filename);

            // HEAD for size
            let size_str = std::process::Command::new("curl")
                .args(&["-sI", "-L", &url])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();
            let size: u64 = size_str.lines()
                .filter(|l| l.to_lowercase().starts_with("content-length:"))
                .last()
                .and_then(|l| l.split(':').nth(1))
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(5_500_000_000);

            let mut reader = HttpRangeReader::with_chunk_size(url, size, 256 * 1024 * 1024);
            let out = std::fs::File::create(&out_path).expect("create output");
            let mut writer = BufWriter::new(out);

            let stats = super::stream_index_safetensors_bf16(
                &mut reader, &mut writer, octave_stride,
                Some(&|name, lt, orig, comp| {
                    let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                    eprintln!("  {:50} {:>12} → {:>8} ({:.0}×)", name, orig, comp, ratio);
                }),
            ).expect("safetensors indexing failed");

            drop(writer);
            let out_size = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
            eprintln!("  → {:.2} MB, {} tensors, {:.0}×",
                out_size as f64 / 1e6, stats.tensors_indexed, stats.overall_ratio());

            all_stats.push(stats);
        }

        all_stats
    }

    #[test]
    #[ignore] // Streams ~35 GB from HuggingFace
    fn test_stream_index_hidream_transformer() {
        let repo = "HiDream-ai/HiDream-I1-Full";
        let shards: Vec<&str> = (1..=7).map(|i| {
            // Leak the string so it lives long enough — test only
            Box::leak(format!(
                "transformer/diffusion_pytorch_model-{:05}-of-00007.safetensors", i
            ).into_boxed_str()) as &str
        }).collect();

        let stats = index_safetensors_shards(repo, &shards, "/tmp/hidream_transformer", 16);

        let total_tensors: usize = stats.iter().map(|s| s.tensors_indexed).sum();
        let total_orig: u64 = stats.iter().map(|s| s.original_bytes).sum();
        let total_comp: u64 = stats.iter().map(|s| s.compressed_bytes).sum();

        eprintln!();
        eprintln!("━━━ HiDream-I1 Transformer (DiT+MoE) ━━━");
        eprintln!("  Source:     {:.2} GB", total_orig as f64 / 1e9);
        eprintln!("  Compressed: {:.2} MB", total_comp as f64 / 1e6);
        eprintln!("  Ratio:      {:.0}×", total_orig as f64 / total_comp.max(1) as f64);
        eprintln!("  Tensors:    {}", total_tensors);

        assert!(total_tensors > 50);
    }

    #[test]
    #[ignore] // Streams ~13 GB
    fn test_stream_index_hidream_text_encoders() {
        let repo = "HiDream-ai/HiDream-I1-Full";

        // CLIP-L
        eprintln!("━━━ CLIP-L ━━━");
        index_safetensors_shards(repo,
            &["text_encoder/model.safetensors"],
            "/tmp/hidream_clip_l", 16);

        // CLIP-G
        eprintln!("━━━ CLIP-G ━━━");
        index_safetensors_shards(repo,
            &["text_encoder_2/model.safetensors"],
            "/tmp/hidream_clip_g", 16);

        // Llama-3.1-8B text encoder (2 shards)
        eprintln!("━━━ Llama-3.1-8B (HiDream text encoder) ━━━");
        index_safetensors_shards(repo,
            &["text_encoder_3/model-00001-of-00002.safetensors",
              "text_encoder_3/model-00002-of-00002.safetensors"],
            "/tmp/hidream_llama_enc", 16);
    }

    #[test]
    #[ignore] // Streams ~16 GB (base Llama-3.1-8B)
    fn test_stream_index_llama31_8b_base() {
        let repo = "unsloth/Llama-3.1-8B";
        let shards: Vec<&str> = (1..=4).map(|i| {
            Box::leak(format!(
                "model-{:05}-of-00004.safetensors", i
            ).into_boxed_str()) as &str
        }).collect();

        index_safetensors_shards(repo, &shards, "/tmp/llama31_8b_base", 16);
    }

    #[test]
    #[ignore] // Requires: HiDream Llama enc + base Llama indexed
    fn test_hidream_llama_diff() {
        use super::super::causal_diff::{causal_diff, print_diff_summary, find_reasoning_scaffold};

        // Compare HiDream's Llama-3.1-8B (image-conditioned) vs base
        // Shards need to be concatenated or diffed per-shard
        let pairs = [
            ("/tmp/llama31_8b_base_shard01.bgz7", "/tmp/hidream_llama_enc_shard01.bgz7", "shard 1"),
            ("/tmp/llama31_8b_base_shard02.bgz7", "/tmp/hidream_llama_enc_shard02.bgz7", "shard 2"),
        ];

        let mut total_shifted = 0usize;
        let mut total_compared = 0usize;

        for (base, dist, label) in &pairs {
            if !std::fs::metadata(base).is_ok() || !std::fs::metadata(dist).is_ok() {
                eprintln!("SKIP {} (files not found)", label);
                continue;
            }

            let (edges, stats) = causal_diff(base, dist, 100).expect("diff failed");
            print_diff_summary(
                &format!("Llama-3.1-8B: base vs HiDream image encoder ({})", label),
                &stats, edges.len());

            let scaffold = find_reasoning_scaffold(&edges, 0.3);
            eprintln!("  Visual grounding scaffold blocks: {:?}", scaffold);

            total_shifted += stats.rows_shifted;
            total_compared += stats.rows_compared;
        }

        if total_compared > 0 {
            eprintln!();
            eprintln!("━━━ Cross-Domain Insight ━━━");
            eprintln!("  Total rows shifted: {}/{} ({:.1}%)",
                total_shifted, total_compared,
                total_shifted as f64 / total_compared as f64 * 100.0);
            eprintln!("  → These shifts = what 'visual grounding' looks like in LLM weight space");
        }
    }
}
