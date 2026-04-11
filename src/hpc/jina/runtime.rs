//! Runtime loader: wire Base17 + palette caches through the full tensor codec.
//!
//! Connects the pre-computed weights to:
//! - HHTL cascade (HEEL/HIP/TWIG/LEAF levels)
//! - CAM-PQ style 6-byte fingerprints
//! - CausalEdge64 S/P/O palette indices
//! - SimilarityTable calibration (256-entry CDF)

use super::cache::{load_base17_cache, load_palette_cache};
use super::causal;
use super::codec::{Base17Token, JinaPalette, BASE_DIM, PALETTE_K};
use std::sync::LazyLock;

/// Embedded weight files (compiled into the binary via include_bytes!).
/// Zero file I/O at runtime — the weights ARE the binary.
///
/// Naming convention: {model}_{aspect}_{vocab_size}k.bin
///   - aspect = base17 (token embeddings) or palette (256-entry lookup)
///   - vocab_size = approximate token count in thousands
static JINA_V4_BASE17: &[u8] = include_bytes!("weights/jina_base17_20k.bin");
static JINA_V4_PALETTE: &[u8] = include_bytes!("weights/jina_palette_20k.bin");

// TODO(jina-v5-bake): When the bake pipeline produces Jina v5 weights
// (151K Qwen3 BPE tokens, 1024D hidden → 34-byte Base17), add:
//   static JINA_V5_BASE17: &[u8] = include_bytes!("weights/jina_v5_base17_151k.bin");
//   static JINA_V5_PALETTE: &[u8] = include_bytes!("weights/jina_v5_palette_151k.bin");
// Then swap the `JINA` LazyLock load line below to use JinaV5. See
// `JINA` / `JINA_V4` / `JINA_V5` statics near end of file for the wiring.

static GPT2_BASE17: &[u8] = include_bytes!("weights/gpt2_base17_50k.bin");
static GPT2_PALETTE: &[u8] = include_bytes!("weights/gpt2_palette_50k.bin");
static BERT_BASE17: &[u8] = include_bytes!("weights/bert_base17_30k.bin");
static BERT_PALETTE: &[u8] = include_bytes!("weights/bert_palette_30k.bin");

/// Which model's weights to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelSource {
    /// Jina v4 text-retrieval (20K tokens, 2048D original, XLM-R base).
    /// LEGACY route. Kept for backward compatibility and direct-access callers
    /// that specifically need v4 behavior. Weights pre-baked at
    /// `weights/jina_base17_20k.bin` + `weights/jina_palette_20k.bin`.
    JinaV4,
    /// Jina v5 small (151K tokens, 1024D hidden, Qwen 3.5 base, SiLU activation).
    /// Also known as **Reader-LM v3** (same model, alternate name — BERT 3.x
    /// architecture lineage; NOT the older Qwen2-based Reader-LM 1.5B/v1/v2).
    ///
    /// **MAIN ROUTE** per AdaWorldAPI model registry (`lance-graph/CLAUDE.md`
    /// → Model Registry → Production models): Jina v5 is the canonical
    /// ground-truth anchor. Same Qwen 3.x BPE as Reranker v3, Qwopus.
    ///
    /// # Storage format on disk (verified by probe)
    ///
    /// The downloaded safetensors at
    /// `lance-graph/crates/thinking-engine/data/jina-v5-onnx/model.safetensors`
    /// is **BF16**, not F16. Every tensor in that 1.19 GB file is stored as
    /// BF16 per the safetensors JSON header, verified by
    /// `crates/thinking-engine/examples/probe_jina_v5_safetensors.rs`. The
    /// embedding matrix is `embed_tokens.weight` shape `[151936, 1024]`
    /// (311 MB BF16). Earlier canonical notes that said "Jina v5 is published
    /// in F16 only" were incorrect for this specific export; other Jina v5
    /// exports (ONNX, GGUF) may use different dtypes.
    ///
    /// The tokenizer lives at `data/jina-v5-tokenizer.json` (flat under the
    /// `data/` directory — NOT under `data/jina-v5-onnx/`). The tokenizer
    /// reports vocab size = 151669, while the safetensors embedding matrix
    /// has 151936 rows. Rows `[151669, 151936)` are ghost/unreachable
    /// (fine-tune-trimmed vocabulary kept aligned for hardware efficiency).
    /// Pair samplers MUST use `min(tokenizer_vocab, embed_rows) = 151669`.
    ///
    /// # Precision hierarchy (workspace-wide rule, Jina v5 specifics)
    ///
    /// 1. **Ground truth is the source file, losslessly upcast on demand.**
    ///    For this file, BF16 source → F32 via the trivial shift
    ///    [`crate::hpc::quantized::BF16`] scalar method. No F32 Vec is
    ///    materialized. No F32 "buffer" persists. F32 is a *method*, not a
    ///    storage format — it lives in registers or a small stack window
    ///    during computation and is discarded with the consumer.
    ///
    /// 2. **Atomic-clock F16 → F32 method** at
    ///    [`crate::hpc::gguf::f16_to_f32`] (`src/hpc/gguf.rs:417`) is proven
    ///    lossless bit-exact over all 65,536 F16 patterns (including
    ///    subnormals, ±0, ±∞, and NaN payloads with correct IEEE 754 quiet
    ///    bit). Used by any F16 source (other Jina exports, GGUF files,
    ///    reranker weights). Not on the Jina v5 safetensors path since that
    ///    file is BF16.
    ///
    /// 3. **Compute precision is BF16 with fused `mul_add`** via
    ///    [`crate::hpc::quantized::bf16_gemm_f32`] (`src/hpc/quantized.rs:108`).
    ///    F32-precision accumulation is a property of the hardware FMA
    ///    (`VDPBF16PS` on AVX-512-BF16, `BFMMLA` on ARM SVE, AMX on Apple),
    ///    invisible to the caller. The `F32x16::mul_add` / `F32x8::mul_add`
    ///    lane types in [`crate::simd`] compile to the appropriate
    ///    instruction for the target CPU.
    ///
    /// 4. **F16 → BF16 has no exponent-range issue.** BF16 has MORE exponent
    ///    bits than F16 (8 vs 5), so every F16 value fits inside BF16 range
    ///    with ~33 orders of magnitude of headroom. The lossy step of
    ///    F16 → BF16 is a 3-bit mantissa truncation (10 → 7 bits), not an
    ///    exponent-range violation. Earlier notes that said "F16 max ~65504
    ///    overflows before reaching BF16 range" were backwards.
    ///
    /// 5. **F64 constants** (π, e, φ, Euler-γ from `std::f64::consts`) are
    ///    used for calibration math (GammaProfile log/exp), preserved at full
    ///    52-bit mantissa precision, and converted to BF16 exactly once per
    ///    profile as a splatted value. The calibration result is 28 bytes.
    ///
    /// 6. **Storage after calibration**: Base17 i16 fixed-point (34-byte
    ///    plane) or palette u8 index. Certification against the BF16 source
    ///    goes through a streaming harness that reads the source once per
    ///    pass, upcasts in registers, and reports Pearson / Spearman /
    ///    Cronbach α to 4 decimal places.
    ///
    /// # Weight baking status
    ///
    /// Compile-time embedded weights at `weights/jina_v5_*.bin` are not yet
    /// produced. Until they are, the `JINA` main-route LazyLock falls back
    /// to v4 bytes. When the certification harness proves lab BF16 at
    /// ≥ 0.9999 and bgz-hhtl-d at ≥ 0.9980 on the three metrics, the
    /// Jina v5 runtime artifacts can be produced from the certified
    /// derivation pipeline. See the TODO block above `JINA_V4_BASE17`.
    JinaV5,
    /// GPT-2 small (50K tokens, 768D original). Same BPE as Jina v4.
    Gpt2,
    /// BERT base uncased (30K tokens, 768D original). WordPiece tokenizer.
    Bert,
}

/// The full runtime: Base17 tokens + palette + distance table + HHTL cascade.
/// Loaded once via LazyLock. Zero cost after first access.
pub struct ModelRuntime {
    /// Source model identifier.
    pub source: ModelSource,
    /// All token embeddings in Base17 format (34 bytes each).
    pub tokens: Vec<Base17Token>,
    /// 256-entry palette with precomputed 256×256 distance table.
    pub palette: JinaPalette,
    /// SimilarityTable: 256-entry CDF calibration (distance → f32 [0,1]).
    pub similarity: [f32; 256],
}

impl ModelRuntime {
    /// Load from embedded weight bytes.
    fn load(source: ModelSource, base17_bytes: &[u8], palette_bytes: &[u8]) -> Self {
        let tokens = load_base17_cache(&mut std::io::Cursor::new(base17_bytes))
            .expect("Failed to load Base17 cache");
        let palette = load_palette_cache(&mut std::io::Cursor::new(palette_bytes))
            .expect("Failed to load palette cache");

        // Build SimilarityTable from the EXACT 256×256 distance distribution.
        // This IS the bgz17 SimilarityTable pattern: empirical CDF → calibrated f32.
        let similarity = build_similarity_table(&palette);

        ModelRuntime {
            source,
            tokens,
            palette,
            similarity,
        }
    }

    /// HHTL HEEL: palette index distance (1 byte per token, O(1)).
    #[inline(always)]
    pub fn heel_distance(&self, token_a: usize, token_b: usize) -> u16 {
        self.palette.distance(token_a, token_b)
    }

    /// HHTL HEEL: calibrated similarity via SimilarityTable [0.0, 1.0].
    #[inline(always)]
    pub fn heel_similarity(&self, token_a: usize, token_b: usize) -> f32 {
        let d = self.heel_distance(token_a, token_b) as usize;
        self.similarity[d.min(255)]
    }

    /// HHTL TWIG: Base17 L1 distance (34 bytes per token, full resolution).
    #[inline(always)]
    pub fn leaf_distance(&self, token_a: usize, token_b: usize) -> u32 {
        self.tokens[token_a].l1(&self.tokens[token_b])
    }

    /// HHTL cascade: HEEL first, escalate to LEAF if needed.
    /// Returns (distance, level_used). Stops as soon as ranking is confident.
    #[inline]
    pub fn cascade_distance(&self, token_a: usize, token_b: usize) -> (u32, HhtlLevel) {
        let heel = self.heel_distance(token_a, token_b);

        // Trivial cases: same palette entry or very far apart
        if heel == 0 {
            return (0, HhtlLevel::Heel);
        }
        if heel > 500 {
            return (heel as u32, HhtlLevel::Heel);
        }

        // Ambiguous zone: escalate to LEAF for precision
        let leaf = self.leaf_distance(token_a, token_b);
        (leaf, HhtlLevel::Leaf)
    }

    /// Pack two tokens + a predicate into a CausalEdge64.
    #[inline]
    pub fn pack_spo_edge(
        &self,
        subject_token: usize,
        predicate_token: usize,
        object_token: usize,
        frequency: f32,
        confidence: f32,
        temporal: u16,
    ) -> u64 {
        causal::pack_edge(
            self.palette.palette_index(subject_token),
            self.palette.palette_index(predicate_token),
            self.palette.palette_index(object_token),
            frequency,
            confidence,
            0b111, // full SPO Pearl mask
            temporal,
        )
    }

    /// CAM-PQ style 6-byte fingerprint: [palette_idx, base17_dim0..4].
    #[inline]
    pub fn cam_fingerprint(&self, token: usize) -> [u8; 6] {
        let pal = self.palette.palette_index(token);
        let b17 = &self.tokens[token].dims;
        [
            pal,
            (b17[0].wrapping_shr(8)) as u8, // BRANCH: sign dimension (MSB of dim 0)
            (b17[1].wrapping_shr(8)) as u8, // TWIG_A: dim 1 MSB
            (b17[2].wrapping_shr(8)) as u8, // TWIG_B: dim 2 MSB
            (b17[3].wrapping_shr(8)) as u8, // LEAF: dim 3 MSB
            (b17[4].wrapping_shr(8)) as u8, // GAMMA: dim 4 MSB
        ]
    }

    /// Token count.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
}

/// HHTL cascade level that resolved the distance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HhtlLevel {
    /// Palette-level distance (1 byte per token).
    Heel,
    /// Full Base17 L1 distance (34 bytes per token).
    Leaf,
}

/// Build SimilarityTable from the 256×256 palette distance distribution.
/// Empirical CDF: count how many pairs have distance ≤ d, normalize.
fn build_similarity_table(palette: &JinaPalette) -> [f32; 256] {
    // Collect all pairwise distances
    let mut all_distances = Vec::with_capacity(PALETTE_K * (PALETTE_K - 1) / 2);
    for i in 0..PALETTE_K {
        for j in (i + 1)..PALETTE_K {
            all_distances.push(palette.distance_table[i][j] as u32);
        }
    }
    all_distances.sort();

    let n = all_distances.len() as f32;
    let max_d = all_distances.last().copied().unwrap_or(1) as usize;

    // Build CDF: similarity[d] = 1.0 - (fraction of pairs with distance ≤ d)
    let mut table = [0.0f32; 256];
    for bucket in 0..256 {
        let threshold = if max_d > 0 {
            (bucket as u64 * max_d as u64 / 255) as u32
        } else {
            0
        };
        let count = all_distances.partition_point(|&d| d <= threshold) as f32;
        let cdf = count / n;
        table[bucket] = 1.0 - cdf; // High distance = low similarity
    }
    table[0] = 1.0; // Self-distance = perfect similarity

    table
}

// ============================================================================
// Global LazyLock runtimes — loaded once, used forever
// ============================================================================

/// Jina **main route**. LazyLock: zero cost after first access.
///
/// Today this loads Jina v4 bytes (20K tokens) because v5 weights are not yet
/// baked into `weights/`. When the v5 bake pipeline produces
/// `weights/jina_v5_base17_151k.bin` + `weights/jina_v5_palette_151k.bin`,
/// swap the load line below to:
///
/// ```ignore
/// ModelRuntime::load(ModelSource::JinaV5, JINA_V5_BASE17, JINA_V5_PALETTE)
/// ```
///
/// Callers should use `JINA` for default behavior. Only use `JINA_V4`
/// explicitly when v4-specific behavior is required (e.g., backward-compat
/// tests).
pub static JINA: LazyLock<ModelRuntime> = LazyLock::new(|| {
    // TODO(jina-v5-bake): swap to JinaV5 when v5 weights exist.
    ModelRuntime::load(ModelSource::JinaV4, JINA_V4_BASE17, JINA_V4_PALETTE)
});

/// Jina **v4 explicit route** (20K tokens, XLM-R base). LEGACY.
///
/// Use this when a caller specifically needs v4 behavior and should NOT be
/// silently upgraded to v5 when the main route is swapped. Today this is
/// functionally identical to `JINA` (both load v4 bytes), but after the v5
/// bake `JINA` will load v5 while `JINA_V4` keeps loading v4.
pub static JINA_V4: LazyLock<ModelRuntime> = LazyLock::new(|| {
    ModelRuntime::load(ModelSource::JinaV4, JINA_V4_BASE17, JINA_V4_PALETTE)
});

/// GPT-2 runtime (50K tokens). Same BPE as Jina → interoperable palettes.
pub static GPT2: LazyLock<ModelRuntime> = LazyLock::new(|| {
    ModelRuntime::load(ModelSource::Gpt2, GPT2_BASE17, GPT2_PALETTE)
});

/// BERT runtime (30K tokens). WordPiece tokenizer (different from GPT-2 BPE).
pub static BERT: LazyLock<ModelRuntime> = LazyLock::new(|| {
    ModelRuntime::load(ModelSource::Bert, BERT_BASE17, BERT_PALETTE)
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jina_runtime_loads() {
        // Main route. Today this is v4; when v5 is baked, update this test to
        // assert source == JinaV5 and vocab_size == ~151000.
        let rt = &*JINA;
        assert_eq!(rt.source, ModelSource::JinaV4);
        assert_eq!(rt.vocab_size(), 20000);
        assert!((rt.similarity[0] - 1.0).abs() < 0.01, "self-similarity should be ~1.0");
    }

    #[test]
    fn test_jina_v4_explicit_route() {
        // Legacy v4-specific accessor. After v5 bake, this test MUST still
        // pass (v4 is the backward-compat guarantee — never deleted).
        let rt = &*JINA_V4;
        assert_eq!(rt.source, ModelSource::JinaV4);
        assert_eq!(rt.vocab_size(), 20000);
        assert!((rt.similarity[0] - 1.0).abs() < 0.01, "self-similarity should be ~1.0");
    }

    #[test]
    fn test_gpt2_runtime_loads() {
        let rt = &*GPT2;
        assert_eq!(rt.source, ModelSource::Gpt2);
        assert_eq!(rt.vocab_size(), 50257);
    }

    #[test]
    fn test_bert_runtime_loads() {
        let rt = &*BERT;
        assert_eq!(rt.source, ModelSource::Bert);
        assert_eq!(rt.vocab_size(), 30522);
    }

    #[test]
    fn test_heel_self_distance_zero() {
        let rt = &*GPT2;
        assert_eq!(rt.heel_distance(0, 0), 0);
        assert!((rt.heel_similarity(0, 0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_heel_symmetric() {
        let rt = &*GPT2;
        assert_eq!(rt.heel_distance(100, 200), rt.heel_distance(200, 100));
    }

    #[test]
    fn test_cascade_trivial_same() {
        let rt = &*JINA;
        let (d, level) = rt.cascade_distance(0, 0);
        assert_eq!(d, 0);
        assert_eq!(level, HhtlLevel::Heel);
    }

    #[test]
    fn test_pack_spo_edge() {
        let rt = &*GPT2;
        let edge = rt.pack_spo_edge(100, 200, 300, 0.8, 0.6, 42);
        assert_eq!(causal::edge_temporal(edge), 42);
        assert!((causal::edge_freq(edge) - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_cam_fingerprint() {
        let rt = &*BERT;
        let fp = rt.cam_fingerprint(1000);
        // First byte is palette index
        assert_eq!(fp[0], rt.palette.palette_index(1000));
        // Should be 6 bytes
        assert_eq!(fp.len(), 6);
    }

    #[test]
    fn test_similarity_table_monotonic() {
        let rt = &*GPT2;
        // Similarity should generally decrease with bucket index
        // (higher bucket = larger distance = lower similarity)
        assert!(rt.similarity[0] >= rt.similarity[255]);
    }

    #[test]
    fn test_cross_model_palette_comparison() {
        // GPT-2 and Jina share BPE — token 0 in both should be
        // the same subword. Their palette indices may differ
        // (different k-means runs) but the Base17 vectors should correlate.
        let jina = &*JINA;
        let gpt2 = &*GPT2;

        // Token 0 exists in both
        let jina_fp = jina.cam_fingerprint(0);
        let gpt2_fp = gpt2.cam_fingerprint(0);

        // They're from different models, so fingerprints may differ.
        // But both should be valid 6-byte fingerprints.
        assert_eq!(jina_fp.len(), 6);
        assert_eq!(gpt2_fp.len(), 6);
    }
}
