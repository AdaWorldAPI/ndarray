//! Causal edge diffing between two bgz7 model indexes.
//!
//! Compares per-row Base17 fingerprints across two indexed models
//! (e.g., base vs distilled), emitting causal edges with NARS truth
//! values for every row that structurally shifted.
//!
//! ```text
//! base.bgz7    ─┐
//!               ├─→ causal_diff() ─→ Vec<WeightEdge>
//! distilled.bgz7┘
//!
//! Each edge: tensor_name + row_idx + verb + L1_distance + NarsTruth
//! ```

use super::bgz17_bridge::Base17;
use super::gguf_indexer::{CompressedTensor, LayerType, read_bgz7_file};
use super::nars::NarsTruth;
use std::collections::HashMap;

// ============================================================================
// Attention projection classification
// ============================================================================

/// Which projection within an attention head.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Projection {
    Q, K, V, O,
    Gate,        // MoE router gate
    FfnGate,     // dense FFN gate
    FfnUp,       // dense FFN up
    FfnDown,     // dense FFN down
    Embedding,
    Other,
}

/// Classify a tensor name into its projection type.
pub fn classify_projection(name: &str) -> Projection {
    if name.contains("q_proj") || name.contains("attn_q") { return Projection::Q; }
    if name.contains("k_proj") || name.contains("attn_k") { return Projection::K; }
    if name.contains("v_proj") || name.contains("attn_v") { return Projection::V; }
    if name.contains("o_proj") || name.contains("attn_output") { return Projection::O; }
    if name.contains("gate_inp") || name.contains("ffn_gate_inp") { return Projection::Gate; }
    if name.contains("gate") && name.contains("exp") { return Projection::FfnGate; }
    if name.contains("up") && (name.contains("exp") || name.contains("ffn")) { return Projection::FfnUp; }
    if name.contains("down") && (name.contains("exp") || name.contains("ffn")) { return Projection::FfnDown; }
    if name.contains("gate") { return Projection::FfnGate; }
    if name.contains("up_proj") { return Projection::FfnUp; }
    if name.contains("down_proj") { return Projection::FfnDown; }
    if name.contains("embed") || name.contains("embd") { return Projection::Embedding; }
    Projection::Other
}

/// Extract block/layer number from tensor name (e.g., "blk.17.attn_q" → 17).
pub fn extract_block(name: &str) -> Option<u32> {
    // Try "blk.N." pattern
    if let Some(pos) = name.find("blk.") {
        let rest = &name[pos + 4..];
        if let Some(dot) = rest.find('.') {
            return rest[..dot].parse().ok();
        }
    }
    // Try "layers.N." pattern
    if let Some(pos) = name.find("layers.") {
        let rest = &name[pos + 7..];
        if let Some(dot) = rest.find('.') {
            return rest[..dot].parse().ok();
        }
    }
    None
}

// ============================================================================
// Weight edge — one row's causal transformation
// ============================================================================

/// Causal relationship verb.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verb {
    /// Row shifted significantly: structural transformation.
    Becomes,
    /// Row stayed similar: distillation preserved this structure.
    Supports,
    /// Row shifted in opposite direction across models: contradictory change.
    Contradicts,
}

/// One causal edge from a weight row diff.
///
/// 64 bytes packed: tensor_id(u16) + row_idx(u32) + projection(u8) +
/// block(u16) + verb(u8) + l1_distance(u32) + truth(f32×2) + pad
#[derive(Clone, Debug)]
pub struct WeightEdge {
    pub tensor_name: String,
    pub row_idx: u32,
    pub block: Option<u32>,
    pub projection: Projection,
    pub layer_type: LayerType,
    pub verb: Verb,
    pub l1_distance: u32,
    pub truth: NarsTruth,
}

// ============================================================================
// Diff engine
// ============================================================================

/// Statistics from a causal diff run.
#[derive(Clone, Debug, Default)]
pub struct DiffStats {
    pub tensors_matched: usize,
    pub tensors_unmatched: usize,
    pub rows_compared: usize,
    pub rows_shifted: usize,
    pub rows_stable: usize,
    pub by_projection: HashMap<String, (usize, usize, f64)>, // (shifted, total, mean_l1)
}

/// Diff two bgz7 files, emitting causal edges for rows that shifted.
///
/// `l1_threshold`: minimum L1 distance to emit a BECOMES edge.
///   Lower = more sensitive (more edges). Higher = only strong shifts.
///   Suggested: 50-200 (Base17 dims are i16 at ×256 scale).
///
/// Returns: (edges, stats)
pub fn causal_diff(
    base_path: &str,
    distilled_path: &str,
    l1_threshold: u32,
) -> Result<(Vec<WeightEdge>, DiffStats), String> {
    let base_tensors = read_bgz7_file(base_path)?;
    let dist_tensors = read_bgz7_file(distilled_path)?;

    // Index distilled tensors by name
    let dist_map: HashMap<&str, &CompressedTensor> = dist_tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();

    let mut edges = Vec::new();
    let mut stats = DiffStats::default();

    // Max possible L1 for normalization (17 dims × 65535 max diff)
    let max_l1: f64 = (17 * 65535) as f64;

    for base_t in &base_tensors {
        let Some(dist_t) = dist_map.get(base_t.name.as_str()) else {
            stats.tensors_unmatched += 1;
            continue;
        };
        stats.tensors_matched += 1;

        // Rows must match
        if base_t.rows.len() != dist_t.rows.len() {
            eprintln!("  WARN: row count mismatch for {}: {} vs {}",
                base_t.name, base_t.rows.len(), dist_t.rows.len());
            continue;
        }

        let projection = classify_projection(&base_t.name);
        let block = extract_block(&base_t.name);
        let proj_key = format!("{:?}", projection);

        let n_rows = base_t.rows.len();
        let mut shifted = 0usize;
        let mut total_l1 = 0u64;

        for (row_idx, (b, d)) in base_t.rows.iter().zip(dist_t.rows.iter()).enumerate() {
            let l1 = b.l1(d);
            total_l1 += l1 as u64;
            stats.rows_compared += 1;

            if l1 > l1_threshold {
                shifted += 1;
                stats.rows_shifted += 1;

                let frequency = (l1 as f64 / max_l1).min(1.0) as f32;
                let confidence = (1.0 - 1.0 / (1.0 + n_rows as f32)).min(0.99);

                edges.push(WeightEdge {
                    tensor_name: base_t.name.clone(),
                    row_idx: row_idx as u32,
                    block,
                    projection: projection.clone(),
                    layer_type: base_t.layer_type.clone(),
                    verb: Verb::Becomes,
                    l1_distance: l1,
                    truth: NarsTruth::new(frequency, confidence),
                });
            } else {
                stats.rows_stable += 1;
            }
        }

        let mean_l1 = if n_rows > 0 { total_l1 as f64 / n_rows as f64 } else { 0.0 };
        let entry = stats.by_projection.entry(proj_key).or_insert((0, 0, 0.0));
        entry.0 += shifted;
        entry.1 += n_rows;
        entry.2 = (entry.2 * (entry.1 - n_rows) as f64 + mean_l1 * n_rows as f64) / entry.1 as f64;
    }

    Ok((edges, stats))
}

/// Print a summary of diff stats.
pub fn print_diff_summary(label: &str, stats: &DiffStats, edge_count: usize) {
    eprintln!();
    eprintln!("━━━ {} ━━━", label);
    eprintln!("  Tensors matched: {}, unmatched: {}", stats.tensors_matched, stats.tensors_unmatched);
    eprintln!("  Rows: {} compared, {} shifted ({:.1}%), {} stable",
        stats.rows_compared, stats.rows_shifted,
        if stats.rows_compared > 0 { stats.rows_shifted as f64 / stats.rows_compared as f64 * 100.0 } else { 0.0 },
        stats.rows_stable);
    eprintln!("  Edges emitted: {}", edge_count);
    eprintln!();

    // Sort projections by shift percentage
    let mut projs: Vec<_> = stats.by_projection.iter().collect();
    projs.sort_by(|a, b| {
        let pct_a = if a.1.1 > 0 { a.1.0 as f64 / a.1.1 as f64 } else { 0.0 };
        let pct_b = if b.1.1 > 0 { b.1.0 as f64 / b.1.1 as f64 } else { 0.0 };
        pct_b.partial_cmp(&pct_a).unwrap()
    });

    eprintln!("  Per projection:");
    for (proj, (shifted, total, mean_l1)) in &projs {
        let pct = if *total > 0 { *shifted as f64 / *total as f64 * 100.0 } else { 0.0 };
        eprintln!("    {:<12} {:>6}/{:<6} shifted ({:>5.1}%)  mean_L1={:.1}",
            proj, shifted, total, pct, mean_l1);
    }
}

/// Cluster edges by head to find reasoning scaffold circuits.
///
/// Returns: map of (block, projection) → (shift_count, total_rows, mean_l1)
pub fn cluster_by_head(edges: &[WeightEdge]) -> HashMap<(u32, String), (usize, u32, f64)> {
    let mut clusters: HashMap<(u32, String), (usize, u32, u64)> = HashMap::new();

    for e in edges {
        if let Some(block) = e.block {
            let key = (block, format!("{:?}", e.projection));
            let entry = clusters.entry(key).or_insert((0, 0, 0));
            entry.0 += 1;
            entry.1 = entry.1.max(e.row_idx + 1);
            entry.2 += e.l1_distance as u64;
        }
    }

    clusters.into_iter()
        .map(|(k, (count, max_row, total_l1))| {
            let mean_l1 = if count > 0 { total_l1 as f64 / count as f64 } else { 0.0 };
            (k, (count, max_row, mean_l1))
        })
        .collect()
}

/// Identify reasoning scaffold: blocks where Q+O shifted but K didn't.
pub fn find_reasoning_scaffold(
    edges: &[WeightEdge],
    shift_threshold: f64, // fraction of rows that shifted (0.0-1.0)
) -> Vec<u32> {
    let clusters = cluster_by_head(edges);
    let mut scaffold_blocks = Vec::new();

    // Find all blocks
    let blocks: std::collections::BTreeSet<u32> = edges.iter()
        .filter_map(|e| e.block)
        .collect();

    for block in blocks {
        let q_shift = clusters.get(&(block, "Q".to_string()));
        let k_shift = clusters.get(&(block, "K".to_string()));
        let o_shift = clusters.get(&(block, "O".to_string()));

        let q_pct = q_shift.map(|(c, t, _)| *c as f64 / *t as f64).unwrap_or(0.0);
        let k_pct = k_shift.map(|(c, t, _)| *c as f64 / *t as f64).unwrap_or(0.0);
        let o_pct = o_shift.map(|(c, t, _)| *c as f64 / *t as f64).unwrap_or(0.0);

        // Reasoning scaffold: Q+O shifted, K stable
        if q_pct > shift_threshold && o_pct > shift_threshold && k_pct < shift_threshold {
            scaffold_blocks.push(block);
            eprintln!("  Block {:>2}: SCAFFOLD  Q={:.0}% O={:.0}% K={:.0}%",
                block, q_pct * 100.0, o_pct * 100.0, k_pct * 100.0);
        }
    }

    scaffold_blocks
}

// ============================================================================
// NARS revision across multiple diffs
// ============================================================================

/// Revise truth values across multiple diff runs.
///
/// For each projection type, integrates evidence from multiple model pairs:
/// e.g., 27B_v1, 27B_v2, 9B → revised belief about reasoning scaffold.
pub fn revise_across_diffs(
    diff_results: &[(&str, &DiffStats)],
) -> HashMap<String, NarsTruth> {
    let mut revised: HashMap<String, NarsTruth> = HashMap::new();

    for (label, stats) in diff_results {
        for (proj, (shifted, total, _mean_l1)) in &stats.by_projection {
            let f = if *total > 0 { *shifted as f32 / *total as f32 } else { 0.0 };
            let c = (1.0 - 1.0 / (1.0 + *total as f32)).min(0.99);
            let evidence = NarsTruth::new(f, c);

            let entry = revised.entry(proj.clone()).or_insert(NarsTruth::new(0.5, 0.0));
            // NARS revision: integrate new evidence
            *entry = nars_revision(*entry, evidence);

            eprintln!("  [{}] {}: f={:.3} c={:.3} → revised f={:.3} c={:.3}",
                label, proj, f, c, entry.frequency, entry.confidence);
        }
    }

    revised
}

/// Simple NARS revision (two-premise).
fn nars_revision(a: NarsTruth, b: NarsTruth) -> NarsTruth {
    let w_a = a.confidence / (1.0 - a.confidence + f32::EPSILON);
    let w_b = b.confidence / (1.0 - b.confidence + f32::EPSILON);
    let w_total = w_a + w_b;

    if w_total < f32::EPSILON {
        return NarsTruth::new(0.5, 0.0);
    }

    let f = (w_a * a.frequency + w_b * b.frequency) / w_total;
    let c = w_total / (w_total + 1.0); // k=1 horizon

    NarsTruth::new(f.clamp(0.0, 1.0), c.clamp(0.0, 0.99))
}

// ============================================================================
// MoE gate topology — expert clustering from router weights
// ============================================================================

/// One expert's structural identity from the gate projection.
#[derive(Clone, Debug)]
pub struct ExpertFingerprint {
    pub block: u32,
    pub expert_idx: usize,
    pub base17: Base17,
}

/// Pairwise expert similarity within a block.
#[derive(Clone, Debug)]
pub struct ExpertCluster {
    pub block: u32,
    pub n_experts: usize,
    /// Mean pairwise L1 distance between experts (lower = more redundant).
    pub mean_pairwise_l1: f64,
    /// Number of expert pairs with L1 < threshold (structurally interchangeable).
    pub redundant_pairs: usize,
    /// Total pairs compared.
    pub total_pairs: usize,
    /// Groups of structurally similar experts (L1 < threshold).
    pub groups: Vec<Vec<usize>>,
}

/// Extract MoE gate topology from a bgz7 file.
///
/// Finds all `ffn_gate_inp` tensors (the router gate projections).
/// Each row in the gate tensor = one expert's activation fingerprint.
/// Returns per-block expert fingerprints.
pub fn extract_gate_topology(bgz7_path: &str) -> Result<Vec<ExpertFingerprint>, String> {
    let tensors = read_bgz7_file(bgz7_path)?;
    let mut fingerprints = Vec::new();

    for t in &tensors {
        // Match router gate tensors
        if !t.name.contains("gate_inp") && !t.name.contains("gate.weight") {
            continue;
        }
        // Skip expert FFN gates (gate_exps) — we want the ROUTER gate
        if t.name.contains("_exps") {
            continue;
        }

        let block = extract_block(&t.name).unwrap_or(0);

        for (expert_idx, row) in t.rows.iter().enumerate() {
            fingerprints.push(ExpertFingerprint {
                block,
                expert_idx,
                base17: row.clone(),
            });
        }

        eprintln!("  Gate: {} → {} experts in block {}",
            t.name, t.rows.len(), block);
    }

    Ok(fingerprints)
}

/// Cluster experts within each block by Base17 L1 distance.
///
/// `redundancy_threshold`: L1 below which two experts are "structurally interchangeable".
/// Suggested: 500 (conservative), 1000 (aggressive).
pub fn cluster_experts(
    fingerprints: &[ExpertFingerprint],
    redundancy_threshold: u32,
) -> Vec<ExpertCluster> {
    // Group by block
    let mut by_block: HashMap<u32, Vec<&ExpertFingerprint>> = HashMap::new();
    for fp in fingerprints {
        by_block.entry(fp.block).or_default().push(fp);
    }

    let mut clusters = Vec::new();

    for (block, experts) in &by_block {
        let n = experts.len();
        let mut total_l1 = 0u64;
        let mut redundant = 0usize;
        let total_pairs = n * (n - 1) / 2;

        // Pairwise L1
        let mut adjacency: Vec<Vec<bool>> = vec![vec![false; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let l1 = experts[i].base17.l1(&experts[j].base17);
                total_l1 += l1 as u64;
                if l1 < redundancy_threshold {
                    redundant += 1;
                    adjacency[i][j] = true;
                    adjacency[j][i] = true;
                }
            }
        }

        let mean_l1 = if total_pairs > 0 { total_l1 as f64 / total_pairs as f64 } else { 0.0 };

        // Simple connected-component grouping
        let mut visited = vec![false; n];
        let mut groups = Vec::new();
        for start in 0..n {
            if visited[start] { continue; }
            let mut group = vec![start];
            visited[start] = true;
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                for neighbor in 0..n {
                    if !visited[neighbor] && adjacency[node][neighbor] {
                        visited[neighbor] = true;
                        group.push(neighbor);
                        stack.push(neighbor);
                    }
                }
            }
            if group.len() > 1 {
                groups.push(group);
            }
        }

        eprintln!("  Block {:>2}: {} experts, mean_L1={:.0}, redundant_pairs={}/{} ({:.0}%), groups={}",
            block, n, mean_l1, redundant, total_pairs,
            if total_pairs > 0 { redundant as f64 / total_pairs as f64 * 100.0 } else { 0.0 },
            groups.len());

        clusters.push(ExpertCluster {
            block: *block,
            n_experts: n,
            mean_pairwise_l1: mean_l1,
            redundant_pairs: redundant,
            total_pairs,
            groups,
        });
    }

    clusters.sort_by_key(|c| c.block);
    clusters
}

/// Cross-reference gate topology with attention scaffold.
///
/// For each scaffold block (where Q+O shifted), check if the gate
/// in that block has high expert redundancy. High redundancy + scaffold
/// = the reasoning change works THROUGH the router, not the experts.
pub fn cross_reference_gate_scaffold(
    clusters: &[ExpertCluster],
    scaffold_blocks: &[u32],
) -> Vec<(u32, bool, f64)> {
    let mut results = Vec::new();

    for block in scaffold_blocks {
        if let Some(cluster) = clusters.iter().find(|c| c.block == *block) {
            let redundancy_pct = if cluster.total_pairs > 0 {
                cluster.redundant_pairs as f64 / cluster.total_pairs as f64
            } else { 0.0 };

            let is_routing_dominated = redundancy_pct > 0.5;
            results.push((*block, is_routing_dominated, redundancy_pct));

            eprintln!("  Block {:>2}: scaffold={} routing_dominated={} redundancy={:.0}%",
                block, true, is_routing_dominated, redundancy_pct * 100.0);
        } else {
            // No gate in this block (dense layer, not MoE)
            results.push((*block, false, 0.0));
            eprintln!("  Block {:>2}: scaffold={} (dense, no MoE gate)", block, true);
        }
    }

    results
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_projection() {
        assert_eq!(classify_projection("blk.17.attn_q.weight"), Projection::Q);
        assert_eq!(classify_projection("blk.17.attn_k.weight"), Projection::K);
        assert_eq!(classify_projection("blk.17.attn_v.weight"), Projection::V);
        assert_eq!(classify_projection("blk.17.attn_output.weight"), Projection::O);
        assert_eq!(classify_projection("blk.3.ffn_gate_inp.weight"), Projection::Gate);
        assert_eq!(classify_projection("token_embd.weight"), Projection::Embedding);
    }

    #[test]
    fn test_extract_block() {
        assert_eq!(extract_block("blk.17.attn_q.weight"), Some(17));
        assert_eq!(extract_block("blk.0.ffn_gate.weight"), Some(0));
        assert_eq!(extract_block("token_embd.weight"), None);
    }

    #[test]
    fn test_nars_revision_basics() {
        let a = NarsTruth::new(0.7, 0.9);
        let b = NarsTruth::new(0.8, 0.9);
        let r = nars_revision(a, b);
        // Revised frequency should be between 0.7 and 0.8
        assert!(r.frequency > 0.7 && r.frequency < 0.8);
        // Revised confidence should be higher than either input
        assert!(r.confidence > 0.9);
    }

    // ── Integration tests: require bgz7 files from indexing runs ──

    #[test]
    #[ignore] // Requires: Qwen3.5-27B base + distilled v1 indexed
    fn test_qwen35_27b_v1_diff() {
        let base = "/tmp/qwen35_27b_base.bgz7";
        let dist = "/tmp/qwen35_27b_distilled_v1.bgz7";

        let (edges, stats) = causal_diff(base, dist, 100).expect("diff failed");
        print_diff_summary("Qwen3.5-27B: base → distilled v1", &stats, edges.len());

        let scaffold = find_reasoning_scaffold(&edges, 0.3);
        eprintln!("  Reasoning scaffold blocks: {:?}", scaffold);

        assert!(stats.tensors_matched > 0, "should match tensors");
    }

    #[test]
    #[ignore] // Requires: Qwen3.5-27B distilled v1 + v2 indexed
    fn test_qwen35_27b_v1_v2_diff() {
        let v1 = "/tmp/qwen35_27b_distilled_v1.bgz7";
        let v2 = "/tmp/qwen35_27b_distilled_v2.bgz7";

        let (edges, stats) = causal_diff(v1, v2, 100).expect("diff failed");
        print_diff_summary("Qwen3.5-27B: v1 → v2 (iteration delta)", &stats, edges.len());
    }

    #[test]
    #[ignore] // Requires: Qwen3.5-9B base + distilled indexed
    fn test_qwen35_9b_diff() {
        let base = "/tmp/qwen35_9b_base.bgz7";
        let dist = "/tmp/qwen35_9b_distilled.bgz7";

        let (edges, stats) = causal_diff(base, dist, 100).expect("diff failed");
        print_diff_summary("Qwen3.5-9B: base → distilled", &stats, edges.len());
    }

    #[test]
    #[ignore] // Requires: all 5 models indexed
    fn test_full_reasoning_reverse_eng() {
        use super::super::http_reader::HttpRangeReader;
        use std::io::BufWriter;

        // ── Phase 1: Index all 5 models ──

        let models: Vec<(&str, &str, &str)> = vec![
            ("unsloth/Qwen3.5-27B-GGUF",
             "Qwen3.5-27B-Q8_0.gguf",
             "/tmp/qwen35_27b_base.bgz7"),
            ("Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
             "Qwen3.5-27B.Q8_0.gguf",
             "/tmp/qwen35_27b_distilled_v1.bgz7"),
            ("Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF",
             "Qwen3.5-27B.Q8_0.gguf",
             "/tmp/qwen35_27b_distilled_v2.bgz7"),
            ("unsloth/Qwen3.5-9B-GGUF",
             "Qwen3.5-9B-Q8_0.gguf",
             "/tmp/qwen35_9b_base.bgz7"),
            ("Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-GGUF",
             "Qwen3.5-9B.Q8_0.gguf",
             "/tmp/qwen35_9b_distilled.bgz7"),
        ];

        for (repo, filename, out_path) in &models {
            if std::fs::metadata(out_path).is_ok() {
                eprintln!("SKIP {} (already indexed)", out_path);
                continue;
            }

            let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, filename);
            eprintln!("Indexing {} ...", filename);

            // HEAD to get size
            let size_str = std::process::Command::new("curl")
                .args(&["-sI", "-L", &url])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();
            let size: u64 = size_str.lines()
                .find(|l| l.to_lowercase().starts_with("content-length:"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(30_000_000_000); // fallback 30 GB

            let mut reader = HttpRangeReader::with_chunk_size(url, size, 256 * 1024 * 1024);
            let out = std::fs::File::create(out_path).expect("create output");
            let mut writer = BufWriter::new(out);

            // Q8_0 uses f32 path (needs dequantization)
            let stats = super::super::gguf_indexer::stream_index_gguf(
                &mut reader, &mut writer,
                Some(&|name, lt, orig, comp| {
                    let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                    eprintln!("  {:50} {:>8} → {:>6} ({:.0}×)", name, orig, comp, ratio);
                }),
            ).expect("indexing failed");

            drop(writer);
            eprintln!("  {} → {:.2} MB ({} tensors)",
                out_path,
                std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0) as f64 / 1e6,
                stats.tensors_indexed);
        }

        // ── Phase 2: Diff pairs ──

        let pairs: Vec<(&str, &str, &str)> = vec![
            ("/tmp/qwen35_27b_base.bgz7", "/tmp/qwen35_27b_distilled_v1.bgz7",
             "27B base→v1"),
            ("/tmp/qwen35_27b_base.bgz7", "/tmp/qwen35_27b_distilled_v2.bgz7",
             "27B base→v2"),
            ("/tmp/qwen35_27b_distilled_v1.bgz7", "/tmp/qwen35_27b_distilled_v2.bgz7",
             "27B v1→v2"),
            ("/tmp/qwen35_9b_base.bgz7", "/tmp/qwen35_9b_distilled.bgz7",
             "9B base→distilled"),
        ];

        let mut all_stats: Vec<(&str, DiffStats)> = Vec::new();

        for (base, dist, label) in &pairs {
            let (edges, stats) = causal_diff(base, dist, 100).expect("diff failed");
            print_diff_summary(label, &stats, edges.len());

            if label.contains("base→") {
                let scaffold = find_reasoning_scaffold(&edges, 0.3);
                eprintln!("  Reasoning scaffold blocks: {:?}", scaffold);
            }

            all_stats.push((label, stats));
        }

        // ── Phase 3: NARS revision across all diffs ──

        eprintln!();
        eprintln!("━━━ NARS Revision: integrated evidence ━━━");
        let refs: Vec<(&str, &DiffStats)> = all_stats.iter()
            .map(|(l, s)| (*l, s))
            .collect();
        let revised = revise_across_diffs(&refs);

        eprintln!();
        for (proj, truth) in &revised {
            eprintln!("  {:<12} → f={:.3} c={:.3} ({})",
                proj, truth.frequency, truth.confidence,
                if truth.frequency > 0.5 { "shifted" } else { "stable" });
        }
    }

    #[test]
    #[ignore] // Requires: Maverick bgz7 outputs from shard indexing
    fn test_maverick_gate_topology() {
        // Load all Maverick shard bgz7 files and extract gate tensors
        let mut all_fingerprints = Vec::new();

        for shard in 1..=18u32 {
            let path = format!("/tmp/llama4_maverick_shard{:02}.bgz7", shard);
            if !std::fs::metadata(&path).is_ok() {
                // Try openchat/weights path
                let alt = format!("src/hpc/openchat/weights/llama4_maverick_shard{:02}.bgz7", shard);
                if !std::fs::metadata(&alt).is_ok() {
                    eprintln!("SKIP shard {} (not found)", shard);
                    continue;
                }
                match extract_gate_topology(&alt) {
                    Ok(fps) => all_fingerprints.extend(fps),
                    Err(e) => eprintln!("WARN shard {}: {}", shard, e),
                }
                continue;
            }
            match extract_gate_topology(&path) {
                Ok(fps) => all_fingerprints.extend(fps),
                Err(e) => eprintln!("WARN shard {}: {}", shard, e),
            }
        }

        eprintln!();
        eprintln!("Total expert fingerprints: {}", all_fingerprints.len());

        if all_fingerprints.is_empty() {
            eprintln!("No gate tensors found — Maverick may not have been indexed yet");
            return;
        }

        // Cluster experts
        let clusters = cluster_experts(&all_fingerprints, 500);

        eprintln!();
        eprintln!("━━━ Maverick Gate Topology ━━━");
        let total_redundant: usize = clusters.iter().map(|c| c.redundant_pairs).sum();
        let total_pairs: usize = clusters.iter().map(|c| c.total_pairs).sum();
        eprintln!("  Overall redundancy: {}/{} pairs ({:.0}%)",
            total_redundant, total_pairs,
            if total_pairs > 0 { total_redundant as f64 / total_pairs as f64 * 100.0 } else { 0.0 });

        // NARS truth for expert redundancy
        let f = if total_pairs > 0 { total_redundant as f32 / total_pairs as f32 } else { 0.0 };
        let c = (1.0 - 1.0 / (1.0 + total_pairs as f32)).min(0.99);
        eprintln!("  NARS truth: f={:.3} c={:.3}", f, c);
        eprintln!("  Interpretation: {:.0}% of expert pairs are structurally interchangeable", f * 100.0);
    }

    #[test]
    #[ignore] // Requires: both Maverick bgz7 + Qwen3.5 diff results
    fn test_cross_reference_gate_scaffold() {
        // This test connects the two analyses:
        // 1. Attention scaffold from Qwen3.5 diff (which blocks have Q+O shift)
        // 2. Gate topology from Maverick (which blocks have redundant experts)

        // Step 1: Run the Qwen3.5 diff to find scaffold blocks
        let base = "/tmp/qwen35_27b_base.bgz7";
        let dist = "/tmp/qwen35_27b_distilled_v1.bgz7";

        if !std::fs::metadata(base).is_ok() || !std::fs::metadata(dist).is_ok() {
            eprintln!("SKIP: Qwen3.5 bgz7 files not found");
            return;
        }

        let (edges, _stats) = causal_diff(base, dist, 100).expect("diff failed");
        let scaffold_blocks = find_reasoning_scaffold(&edges, 0.3);

        // Step 2: Extract Maverick gate topology
        let mut all_fps = Vec::new();
        for shard in 1..=18u32 {
            let path = format!("/tmp/llama4_maverick_shard{:02}.bgz7", shard);
            if let Ok(fps) = extract_gate_topology(&path) {
                all_fps.extend(fps);
            }
        }

        if all_fps.is_empty() {
            eprintln!("SKIP: No Maverick gate fingerprints");
            return;
        }

        let clusters = cluster_experts(&all_fps, 500);

        // Step 3: Cross-reference
        eprintln!();
        eprintln!("━━━ Cross-Reference: Attention Scaffold × Gate Topology ━━━");
        let results = cross_reference_gate_scaffold(&clusters, &scaffold_blocks);

        let routing_dominated: usize = results.iter().filter(|(_, rd, _)| *rd).count();
        eprintln!();
        eprintln!("  Scaffold blocks: {}", scaffold_blocks.len());
        eprintln!("  Routing-dominated: {}/{} ({:.0}%)",
            routing_dominated, results.len(),
            if !results.is_empty() { routing_dominated as f64 / results.len() as f64 * 100.0 } else { 0.0 });
        eprintln!("  → {} = reasoning changes work THROUGH the router",
            if routing_dominated > results.len() / 2 { "YES" } else { "PARTIAL" });
    }

    // ════════════════════════════════════════════════════════════════════
    // Safetensors BF16 pipeline — 5 models, 4 diffs, NARS revision
    // ════════════════════════════════════════════════════════════════════

    /// Model descriptor for the safetensors indexing pipeline.
    struct ModelSpec {
        repo: &'static str,
        shards: u32,
        prefix: &'static str,
    }

    const MODELS: [ModelSpec; 5] = [
        ModelSpec { repo: "Qwen/Qwen3.5-27B",                                                   shards: 11, prefix: "qwen35_27b_base" },
        ModelSpec { repo: "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",            shards: 11, prefix: "qwen35_27b_v1" },
        ModelSpec { repo: "Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2",         shards: 11, prefix: "qwen35_27b_v2" },
        ModelSpec { repo: "Qwen/Qwen3.5-9B",                                                    shards: 4,  prefix: "qwen35_9b_base" },
        ModelSpec { repo: "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled",             shards: 4,  prefix: "qwen35_9b_dist" },
    ];

    /// Generate safetensors shard filenames for a model.
    fn shard_filenames(total: u32) -> Vec<String> {
        (1..=total)
            .map(|i| format!("model-{:05}-of-{:05}.safetensors", i, total))
            .collect()
    }

    /// Generate bgz7 output paths for a model's shards.
    fn shard_bgz7_paths(prefix: &str, total: u32) -> Vec<String> {
        (1..=total)
            .map(|i| format!("/tmp/{}_shard{:02}.bgz7", prefix, i))
            .collect()
    }

    /// Index a single model (all shards) via safetensors BF16.
    fn index_model_safetensors(model: &ModelSpec) {
        use super::super::safetensors::stream_index_safetensors_bf16;
        use super::super::http_reader::HttpRangeReader;
        use std::io::BufWriter;

        let filenames = shard_filenames(model.shards);
        let out_paths = shard_bgz7_paths(model.prefix, model.shards);

        for (shard_idx, (filename, out_path)) in filenames.iter().zip(out_paths.iter()).enumerate() {
            if std::fs::metadata(out_path).is_ok() {
                eprintln!("SKIP {} (exists)", out_path);
                continue;
            }

            let url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                model.repo, filename
            );
            eprintln!(
                "[{}] shard {}/{}: {}",
                model.prefix,
                shard_idx + 1,
                model.shards,
                filename
            );

            // HEAD for content-length
            let size: u64 = std::process::Command::new("curl")
                .args(&["-sI", "-L", &url])
                .output()
                .ok()
                .and_then(|o| {
                    String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .find(|l| l.to_lowercase().starts_with("content-length:"))
                        .and_then(|l| l.split(':').nth(1))
                        .and_then(|s| s.trim().parse().ok())
                })
                .unwrap_or(6_000_000_000);

            let mut reader = HttpRangeReader::with_chunk_size(url, size, 256 * 1024 * 1024);
            let out = std::fs::File::create(out_path).expect("create output");
            let mut writer = BufWriter::new(out);

            let stats = stream_index_safetensors_bf16(
                &mut reader,
                &mut writer,
                16, // octave_stride: strided+halftone
                Some(&|name, _lt, orig, comp| {
                    let ratio = if comp > 0 { orig as f64 / comp as f64 } else { 0.0 };
                    eprintln!("  {:50} {:>12} → {:>8} ({:.0}×)", name, orig, comp, ratio);
                }),
            )
            .expect("safetensors indexing failed");

            drop(writer);
            let out_size = std::fs::metadata(out_path).map(|m| m.len()).unwrap_or(0);
            eprintln!(
                "  → {:.2} MB, {} tensors, {:.0}×",
                out_size as f64 / 1e6,
                stats.tensors_indexed,
                stats.overall_ratio()
            );
        }
    }

    /// Causal diff across matched shards of two models, aggregating edges + stats.
    fn causal_diff_sharded(
        base_prefix: &str,
        dist_prefix: &str,
        n_shards: u32,
        l1_threshold: u32,
    ) -> (Vec<WeightEdge>, DiffStats) {
        let base_paths = shard_bgz7_paths(base_prefix, n_shards);
        let dist_paths = shard_bgz7_paths(dist_prefix, n_shards);

        let mut all_edges = Vec::new();
        let mut agg = DiffStats::default();

        for (shard_idx, (bp, dp)) in base_paths.iter().zip(dist_paths.iter()).enumerate() {
            if !std::fs::metadata(bp).is_ok() || !std::fs::metadata(dp).is_ok() {
                eprintln!("  SKIP shard {} (not found)", shard_idx + 1);
                continue;
            }

            let (edges, stats) = causal_diff(bp, dp, l1_threshold)
                .unwrap_or_else(|e| panic!("diff shard {} failed: {}", shard_idx + 1, e));

            eprintln!(
                "  shard {:>2}: {} tensors, {}/{} shifted ({:.1}%), {} edges",
                shard_idx + 1,
                stats.tensors_matched,
                stats.rows_shifted,
                stats.rows_compared,
                if stats.rows_compared > 0 {
                    stats.rows_shifted as f64 / stats.rows_compared as f64 * 100.0
                } else {
                    0.0
                },
                edges.len()
            );

            // Aggregate stats
            agg.tensors_matched += stats.tensors_matched;
            agg.tensors_unmatched += stats.tensors_unmatched;
            agg.rows_compared += stats.rows_compared;
            agg.rows_shifted += stats.rows_shifted;
            agg.rows_stable += stats.rows_stable;
            for (proj, (shifted, total, mean_l1)) in &stats.by_projection {
                let entry = agg.by_projection.entry(proj.clone()).or_insert((0, 0, 0.0));
                let prev_total = entry.1;
                entry.0 += shifted;
                entry.1 += total;
                if entry.1 > 0 {
                    entry.2 = (entry.2 * prev_total as f64 + mean_l1 * *total as f64)
                        / entry.1 as f64;
                }
            }

            all_edges.extend(edges);
        }

        (all_edges, agg)
    }

    // ── Per-model indexing tests ──

    #[test]
    #[ignore] // Streams ~55 GB
    fn test_index_qwen35_27b_base() {
        index_model_safetensors(&MODELS[0]);
    }

    #[test]
    #[ignore] // Streams ~55 GB
    fn test_index_qwen35_27b_v1() {
        index_model_safetensors(&MODELS[1]);
    }

    #[test]
    #[ignore] // Streams ~55 GB
    fn test_index_qwen35_27b_v2() {
        index_model_safetensors(&MODELS[2]);
    }

    #[test]
    #[ignore] // Streams ~18 GB
    fn test_index_qwen35_9b_base() {
        index_model_safetensors(&MODELS[3]);
    }

    #[test]
    #[ignore] // Streams ~18 GB
    fn test_index_qwen35_9b_dist() {
        index_model_safetensors(&MODELS[4]);
    }

    // ── Full pipeline: 4 diffs + scaffold + NARS ──

    #[test]
    #[ignore] // Requires all 5 models indexed (safetensors BF16)
    fn test_qwen35_claude_reasoning_diff() {
        let threshold = 100u32;

        // ── Diff 1: base 27B → v1 ──
        eprintln!();
        eprintln!("════ Diff 1: 27B base → distilled v1 ════");
        eprintln!("  What does Claude reasoning look like in weight space?");
        let (edges_1, stats_1) = causal_diff_sharded(
            "qwen35_27b_base", "qwen35_27b_v1", 11, threshold,
        );
        print_diff_summary("27B: base → v1", &stats_1, edges_1.len());

        // ── Diff 2: base 27B → v2 ──
        eprintln!();
        eprintln!("════ Diff 2: 27B base → distilled v2 ════");
        eprintln!("  Did v2 refine the same heads or find new ones?");
        let (edges_2, stats_2) = causal_diff_sharded(
            "qwen35_27b_base", "qwen35_27b_v2", 11, threshold,
        );
        print_diff_summary("27B: base → v2", &stats_2, edges_2.len());

        // ── Diff 3: v1 → v2 ──
        eprintln!();
        eprintln!("════ Diff 3: 27B v1 → v2 (iteration delta) ════");
        eprintln!("  Which heads converged vs overcorrected?");
        let (edges_3, stats_3) = causal_diff_sharded(
            "qwen35_27b_v1", "qwen35_27b_v2", 11, threshold,
        );
        print_diff_summary("27B: v1 → v2", &stats_3, edges_3.len());

        // ── Diff 4: 9B base → distilled ──
        eprintln!();
        eprintln!("════ Diff 4: 9B base → distilled ════");
        eprintln!("  Is the reasoning scaffold scale-invariant?");
        let (edges_4, stats_4) = causal_diff_sharded(
            "qwen35_9b_base", "qwen35_9b_dist", 4, threshold,
        );
        print_diff_summary("9B: base → distilled", &stats_4, edges_4.len());

        // ── Phase 3: Reasoning scaffold detection ──
        eprintln!();
        eprintln!("════ Reasoning Scaffold Detection ════");

        let scaffold_27b_v1 = find_reasoning_scaffold(&edges_1, 0.3);
        eprintln!("  27B v1 scaffold blocks: {:?}", scaffold_27b_v1);

        let scaffold_27b_v2 = find_reasoning_scaffold(&edges_2, 0.3);
        eprintln!("  27B v2 scaffold blocks: {:?}", scaffold_27b_v2);

        let scaffold_9b = find_reasoning_scaffold(&edges_4, 0.3);
        eprintln!("  9B scaffold blocks:     {:?}", scaffold_9b);

        // Scale-invariant: present in both 27B and 9B
        let scale_invariant: Vec<u32> = scaffold_27b_v1
            .iter()
            .filter(|b| scaffold_9b.contains(b))
            .cloned()
            .collect();

        // 27B-only: capacity-dependent reasoning
        let capacity_dependent: Vec<u32> = scaffold_27b_v1
            .iter()
            .filter(|b| !scaffold_9b.contains(b))
            .cloned()
            .collect();

        // v1∩v2 convergence
        let converged: Vec<u32> = scaffold_27b_v1
            .iter()
            .filter(|b| scaffold_27b_v2.contains(b))
            .cloned()
            .collect();

        eprintln!();
        eprintln!("  Scale-invariant blocks (27B∩9B):  {:?}", scale_invariant);
        eprintln!("  Capacity-dependent (27B only):    {:?}", capacity_dependent);
        eprintln!("  Converged (v1∩v2):                {:?}", converged);

        // ── Phase 4: NARS revision across all diffs ──
        eprintln!();
        eprintln!("════ NARS Revision: Integrated Evidence ════");

        let all_stats: Vec<(&str, &DiffStats)> = vec![
            ("27B base→v1", &stats_1),
            ("27B base→v2", &stats_2),
            ("27B v1→v2", &stats_3),
            ("9B base→dist", &stats_4),
        ];
        let revised = revise_across_diffs(&all_stats);

        eprintln!();
        eprintln!("  Integrated truth per projection:");
        let mut sorted_projs: Vec<_> = revised.iter().collect();
        sorted_projs.sort_by(|a, b| b.1.frequency.partial_cmp(&a.1.frequency).unwrap());
        for (proj, truth) in &sorted_projs {
            let label = if truth.frequency > 0.5 {
                "SHIFTED"
            } else if truth.frequency > 0.3 {
                "variable"
            } else {
                "STABLE"
            };
            eprintln!(
                "    {:<12} → f={:.3} c={:.3} ({})",
                proj, truth.frequency, truth.confidence, label
            );
        }

        // ── Phase 5: Top shifted heads ──
        eprintln!();
        eprintln!("════ Top Shifted Attention Heads (v1) ════");

        let clusters = cluster_by_head(&edges_1);
        let mut sorted: Vec<_> = clusters.into_iter().collect();
        sorted.sort_by(|a, b| b.1 .2.partial_cmp(&a.1 .2).unwrap());

        for ((block, proj), (count, max_row, mean_l1)) in sorted.iter().take(20) {
            eprintln!(
                "    Block {:>2} {:>10}: {}/{} shifted, mean_L1={:.0}",
                block, proj, count, max_row, mean_l1
            );
        }

        // ── Phase 6: Write results ──
        eprintln!();
        eprintln!("════ Writing Results ════");

        let mut report = String::new();
        report.push_str("# Qwen3.5 → Claude-4.6-Opus Reasoning Scaffold Analysis\n\n");
        report.push_str(&format!("Generated: {}\n", "2026-03-30"));
        report.push_str(&format!("L1 threshold: {}\n\n", threshold));

        report.push_str("## Model Matrix\n\n");
        report.push_str("| ID | Repo | Shards | Path |\n");
        report.push_str("|---|---|---|---|\n");
        for m in &MODELS {
            report.push_str(&format!(
                "| {} | {} | {} | safetensors BF16 |\n",
                m.prefix, m.repo, m.shards
            ));
        }

        report.push_str("\n## Diff Summary\n\n");
        for (label, stats) in &[
            ("27B base→v1", &stats_1),
            ("27B base→v2", &stats_2),
            ("27B v1→v2", &stats_3),
            ("9B base→dist", &stats_4),
        ] {
            let pct = if stats.rows_compared > 0 {
                stats.rows_shifted as f64 / stats.rows_compared as f64 * 100.0
            } else {
                0.0
            };
            report.push_str(&format!(
                "- **{}**: {}/{} rows shifted ({:.1}%), {} tensors\n",
                label, stats.rows_shifted, stats.rows_compared, pct, stats.tensors_matched
            ));
        }

        report.push_str("\n## Reasoning Scaffold\n\n");
        report.push_str(&format!(
            "- **Scale-invariant blocks (27B∩9B)**: {:?}\n",
            scale_invariant
        ));
        report.push_str(&format!(
            "- **Capacity-dependent (27B only)**: {:?}\n",
            capacity_dependent
        ));
        report.push_str(&format!(
            "- **Converged (v1∩v2)**: {:?}\n",
            converged
        ));

        report.push_str("\n## NARS Revised Truth Per Projection\n\n");
        report.push_str("| Projection | Frequency | Confidence | Interpretation |\n");
        report.push_str("|---|---|---|---|\n");
        for (proj, truth) in &sorted_projs {
            let label = if truth.frequency > 0.5 {
                "SHIFTED"
            } else if truth.frequency > 0.3 {
                "variable"
            } else {
                "STABLE"
            };
            report.push_str(&format!(
                "| {} | {:.3} | {:.3} | {} |\n",
                proj, truth.frequency, truth.confidence, label
            ));
        }

        report.push_str("\n## Top 20 Shifted Heads (base→v1)\n\n");
        report.push_str("| Block | Projection | Shifted/Total | Mean L1 |\n");
        report.push_str("|---|---|---|---|\n");
        for ((block, proj), (count, max_row, mean_l1)) in sorted.iter().take(20) {
            report.push_str(&format!(
                "| {} | {} | {}/{} | {:.0} |\n",
                block, proj, count, max_row, mean_l1
            ));
        }

        // Write to knowledge base
        let kb_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/.claude/knowledge");
        let _ = std::fs::create_dir_all(kb_dir);
        let out_path = format!("{}/reasoning_reverse_eng_results.md", kb_dir);
        std::fs::write(&out_path, &report).expect("write results");
        eprintln!("  → {}", out_path);

        // Assertions
        assert!(stats_1.tensors_matched > 0, "should match tensors in diff 1");
        assert!(stats_4.tensors_matched > 0, "should match tensors in diff 4");
    }
}
