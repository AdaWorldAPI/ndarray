//! JitsonTemplate: parsed, validated, and converted JITSON templates.
//!
//! Provides the high-level API for converting JSON text into strongly-typed
//! scan configurations with pipeline stages, feature declarations, and
//! backend references. Also includes the WAL precompile queue for
//! speculative JIT compilation with prefetch addressing.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

use super::parser::{parse_json, JsonValue, ParseError};
use super::scan_config::ScanConfig;
use super::validator::{validate, ValidationError};

/// Parsed and validated JITSON template.
#[derive(Clone, Debug)]
pub struct JitsonTemplate {
    pub kernel: String,
    pub scan: ScanConfig,
    pub pipeline: Vec<PipelineStage>,
    pub features: Vec<(String, bool)>,
    pub backends: Vec<BackendConfig>,
    pub cranelift_preset: Option<String>,
    pub cranelift_opt_level: Option<String>,
}

/// A single stage in the JIT pipeline.
#[derive(Clone, Debug)]
pub struct PipelineStage {
    pub stage: String,
    pub avx512_instr: Option<String>,
    pub fallback: Option<String>,
    /// Backend CPU-lane reference (e.g. "lancedb", "dragonfly").
    pub backend: Option<String>,
    /// Backend-specific key/table/prefix for this stage.
    pub backend_key: Option<String>,
}

/// Configuration for an external data backend (CPU lane).
#[derive(Clone, Debug)]
pub struct BackendConfig {
    pub name: String,
    pub uri: String,
    /// Extra backend-specific options (key-value pairs from the JSON object).
    pub options: Vec<(String, String)>,
}

/// Error type for JITSON operations.
#[derive(Clone, Debug)]
pub enum JitsonError {
    Parse(ParseError),
    Validation(Vec<ValidationError>),
    Conversion(String),
}

impl core::fmt::Display for JitsonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            JitsonError::Parse(e) => write!(f, "{}", e),
            JitsonError::Validation(errs) => {
                for e in errs {
                    writeln!(f, "{}", e)?;
                }
                Ok(())
            }
            JitsonError::Conversion(msg) => write!(f, "JITSON conversion: {}", msg),
        }
    }
}

/// Parse a JSON string, validate it, and convert to a [`JitsonTemplate`].
///
/// Bracket recovery is applied automatically — a missing closing `}` or `]`
/// at the end of input will be silently fixed.
pub fn from_json(input: &str) -> Result<JitsonTemplate, JitsonError> {
    let root = parse_json(input).map_err(JitsonError::Parse)?;
    let errors = validate(&root);
    if !errors.is_empty() {
        return Err(JitsonError::Validation(errors));
    }
    convert(&root)
}

fn convert(root: &JsonValue) -> Result<JitsonTemplate, JitsonError> {
    let kernel = root.get("kernel").and_then(|v| v.as_str()).unwrap();
    let scan_obj = root.get("scan").unwrap();

    let scan = ScanConfig {
        threshold: scan_obj.get("threshold").and_then(|v| v.as_u64()).unwrap(),
        record_size: scan_obj
            .get("record_size")
            .and_then(|v| v.as_usize())
            .unwrap(),
        top_k: scan_obj.get("top_k").and_then(|v| v.as_usize()).unwrap(),
        query: Vec::new(), // Query bytes are provided at runtime, not in the template
    };

    let pipeline = match root.get("pipeline").and_then(|v| v.as_array()) {
        Some(stages) => stages
            .iter()
            .map(|s| PipelineStage {
                stage: s.get("stage").and_then(|v| v.as_str()).unwrap_or("").into(),
                avx512_instr: s.get("avx512").and_then(|v| v.as_str()).map(String::from),
                fallback: s.get("fallback").and_then(|v| v.as_str()).map(String::from),
                backend: s.get("backend").and_then(|v| v.as_str()).map(String::from),
                backend_key: s
                    .get("table")
                    .or_else(|| s.get("prefix"))
                    .or_else(|| s.get("key"))
                    .and_then(|v| v.as_str())
                    .map(String::from),
            })
            .collect(),
        None => Vec::new(),
    };

    let backends = match root.get("backends").and_then(|v| v.as_object()) {
        Some(pairs) => pairs
            .iter()
            .map(|(name, cfg)| {
                let uri = cfg.get("uri").and_then(|v| v.as_str()).unwrap_or("").into();
                let options = cfg
                    .as_object()
                    .map(|o| {
                        o.iter()
                            .filter(|(k, _)| k != "uri")
                            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), String::from(s))))
                            .collect()
                    })
                    .unwrap_or_default();
                BackendConfig {
                    name: name.clone(),
                    uri,
                    options,
                }
            })
            .collect(),
        None => Vec::new(),
    };

    let features = match root.get("features").and_then(|v| v.as_object()) {
        Some(pairs) => pairs
            .iter()
            .map(|(k, v)| (k.clone(), v.as_bool().unwrap_or(false)))
            .collect(),
        None => Vec::new(),
    };

    let cranelift_preset = root
        .get("cranelift")
        .and_then(|cl| cl.get("preset"))
        .and_then(|v| v.as_str())
        .map(String::from);

    let cranelift_opt_level = root
        .get("cranelift")
        .and_then(|cl| cl.get("opt_level"))
        .and_then(|v| v.as_str())
        .map(String::from);

    Ok(JitsonTemplate {
        kernel: String::from(kernel),
        scan,
        pipeline,
        features,
        backends,
        cranelift_preset,
        cranelift_opt_level,
    })
}

/// Check if a template's pipeline is satisfiable given the declared features.
///
/// Returns a list of (stage_index, instruction, missing_features) for each
/// pipeline stage that requires features not enabled in the template.
pub fn check_pipeline_features(template: &JitsonTemplate) -> Vec<(usize, String, Vec<String>)> {
    let enabled: Vec<&str> = template
        .features
        .iter()
        .filter(|(_, on)| *on)
        .map(|(k, _)| k.as_str())
        .collect();

    let mut unsatisfied = Vec::new();
    for (i, stage) in template.pipeline.iter().enumerate() {
        if let Some(ref instr) = stage.avx512_instr {
            let required = super::validator::required_features(instr);
            let missing: Vec<String> = required
                .iter()
                .filter(|f| !enabled.contains(f))
                .map(|f| String::from(*f))
                .collect();
            if !missing.is_empty() {
                unsatisfied.push((i, instr.clone(), missing));
            }
        }
    }
    unsatisfied
}

// ---------------------------------------------------------------------------
// Stable hash (FNV-1a 64-bit) for precompile cache keys
// ---------------------------------------------------------------------------

/// Stable 64-bit hash of a JITSON template for use as a precompile cache key.
///
/// Two templates that produce identical scan configs, pipelines, and feature
/// sets will have the same hash, enabling cache hits across processes.
pub fn template_hash(template: &JitsonTemplate) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut h = FNV_OFFSET;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    };

    feed(template.kernel.as_bytes());
    feed(&template.scan.threshold.to_le_bytes());
    feed(&template.scan.record_size.to_le_bytes());
    feed(&template.scan.top_k.to_le_bytes());

    for stage in &template.pipeline {
        feed(stage.stage.as_bytes());
        if let Some(ref instr) = stage.avx512_instr {
            feed(instr.as_bytes());
        }
        if let Some(ref fb) = stage.fallback {
            feed(fb.as_bytes());
        }
        if let Some(ref be) = stage.backend {
            feed(be.as_bytes());
        }
    }

    for (feat, on) in &template.features {
        feed(feat.as_bytes());
        feed(&[*on as u8]);
    }

    if let Some(ref preset) = template.cranelift_preset {
        feed(preset.as_bytes());
    }
    if let Some(ref opt) = template.cranelift_opt_level {
        feed(opt.as_bytes());
    }

    h
}

// ---------------------------------------------------------------------------
// WAL Precompile Queue + Prefetch Addressing
// ---------------------------------------------------------------------------

/// Entry in the write-ahead precompile queue.
#[derive(Clone, Debug)]
pub struct PrecompileEntry {
    /// Stable FNV-1a hash of the template — the prefetch address.
    pub hash: u64,
    /// The template that produced this entry.
    pub template: JitsonTemplate,
    /// Compilation state.
    pub state: CompileState,
}

/// State of a precompile entry in the WAL queue.
#[derive(Clone, Debug, PartialEq)]
pub enum CompileState {
    /// Queued for compilation, not yet started.
    Pending,
    /// Compiled successfully. `code_addr` is the function pointer.
    Compiled { code_addr: u64 },
    /// Previously compiled but evicted from the hot cache.
    Evicted,
}

/// Write-ahead precompile queue with deduplication and prefetch hints.
///
/// Templates are appended in order. The queue supports:
/// - **Enqueue**: add a template for compilation (deduplicates by hash)
/// - **Lookup**: check if a template is already compiled (prefetch hit)
/// - **Prefetch hint**: given the current template, return the next entry's
///   hash so the caller can issue a memory prefetch for its code page
#[derive(Clone, Debug, Default)]
pub struct PrecompileQueue {
    entries: Vec<PrecompileEntry>,
}

impl PrecompileQueue {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Enqueue a template for precompilation. Returns the stable hash.
    /// If already queued, returns the existing hash without duplicating.
    pub fn enqueue(&mut self, template: JitsonTemplate) -> u64 {
        let hash = template_hash(&template);
        if self.entries.iter().any(|e| e.hash == hash) {
            return hash;
        }
        self.entries.push(PrecompileEntry {
            hash,
            template,
            state: CompileState::Pending,
        });
        hash
    }

    /// Look up a template by its prefetch address (hash).
    pub fn lookup(&self, hash: u64) -> Option<&PrecompileEntry> {
        self.entries.iter().find(|e| e.hash == hash)
    }

    /// Mark a template as compiled with the given code address.
    pub fn mark_compiled(&mut self, hash: u64, code_addr: u64) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.hash == hash) {
            entry.state = CompileState::Compiled { code_addr };
            true
        } else {
            false
        }
    }

    /// Mark a template as evicted from the hot cache.
    pub fn mark_evicted(&mut self, hash: u64) -> bool {
        if let Some(entry) = self.entries.iter_mut().find(|e| e.hash == hash) {
            entry.state = CompileState::Evicted;
            true
        } else {
            false
        }
    }

    /// Return the prefetch address of the entry immediately after `hash`.
    ///
    /// Enables speculative prefetching: while executing compiled code
    /// for `hash`, the caller can issue `PREFETCHT0` on the next entry's
    /// code page.
    pub fn prefetch_next(&self, hash: u64) -> Option<u64> {
        let idx = self.entries.iter().position(|e| e.hash == hash)?;
        self.entries.get(idx + 1).and_then(|e| match e.state {
            CompileState::Compiled { code_addr } => Some(code_addr),
            _ => None,
        })
    }

    /// Return all pending entries (templates awaiting compilation).
    pub fn pending(&self) -> Vec<&PrecompileEntry> {
        self.entries
            .iter()
            .filter(|e| e.state == CompileState::Pending)
            .collect()
    }

    /// Number of entries in the queue.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_TEMPLATE: &str = r#"{
        "version": 1,
        "kernel": "hamming_distance",
        "scan": {
            "threshold": 2048,
            "record_size": 256,
            "top_k": 10
        },
        "pipeline": [
            { "stage": "xor",    "avx512": "vpxord" },
            { "stage": "popcnt", "avx512": "vpopcntd", "fallback": "avx2_lookup" },
            { "stage": "reduce", "avx512": "vpord" }
        ],
        "features": {
            "avx512f": true,
            "avx512vl": true,
            "avx512vpopcntdq": true,
            "avx512bw": false
        },
        "cranelift": {
            "preset": "sapphire_rapids",
            "opt_level": "speed"
        }
    }"#;

    const BACKEND_TEMPLATE: &str = r#"{
        "version": 1,
        "kernel": "hamming_distance",
        "scan": { "threshold": 2048, "record_size": 256, "top_k": 10 },
        "pipeline": [
            { "stage": "fetch",  "backend": "lancedb",   "table": "embeddings" },
            { "stage": "xor",    "avx512": "vpxord" },
            { "stage": "popcnt", "avx512": "vpopcntd" },
            { "stage": "store",  "backend": "dragonfly", "prefix": "results:" }
        ],
        "backends": {
            "lancedb":   { "uri": "data/vectors.lance" },
            "dragonfly": { "uri": "redis://127.0.0.1:6379" }
        },
        "features": { "avx512f": true, "avx512vl": true, "avx512vpopcntdq": true }
    }"#;

    #[test]
    fn test_from_json_roundtrip() {
        let tmpl = from_json(VALID_TEMPLATE).unwrap();
        assert_eq!(tmpl.kernel, "hamming_distance");
        assert_eq!(tmpl.scan.threshold, 2048);
        assert_eq!(tmpl.scan.record_size, 256);
        assert_eq!(tmpl.scan.top_k, 10);
        assert_eq!(tmpl.pipeline.len(), 3);
        assert_eq!(tmpl.pipeline[0].stage, "xor");
        assert_eq!(tmpl.pipeline[1].avx512_instr.as_deref(), Some("vpopcntd"));
        assert_eq!(tmpl.pipeline[1].fallback.as_deref(), Some("avx2_lookup"));
        assert_eq!(tmpl.cranelift_preset.as_deref(), Some("sapphire_rapids"));
    }

    #[test]
    fn test_check_pipeline_features() {
        let tmpl = from_json(VALID_TEMPLATE).unwrap();
        let unsatisfied = check_pipeline_features(&tmpl);
        assert!(
            unsatisfied.is_empty(),
            "unexpected unsatisfied: {:?}",
            unsatisfied
        );
    }

    #[test]
    fn test_check_pipeline_missing_feature() {
        let input = r#"{
            "version": 1,
            "kernel": "hamming_distance",
            "scan": {"threshold": 1, "record_size": 64, "top_k": 5},
            "pipeline": [{"stage": "popcnt_byte", "avx512": "vpopcntb"}],
            "features": {"avx512f": true, "avx512vl": true}
        }"#;
        let tmpl = from_json(input).unwrap();
        let unsatisfied = check_pipeline_features(&tmpl);
        assert_eq!(unsatisfied.len(), 1);
        assert_eq!(unsatisfied[0].1, "vpopcntb");
        assert!(unsatisfied[0].2.contains(&String::from("avx512bitalg")));
    }

    #[test]
    fn test_backend_template_parses() {
        let tmpl = from_json(BACKEND_TEMPLATE).unwrap();
        assert_eq!(tmpl.backends.len(), 2);
        assert_eq!(tmpl.backends[0].name, "lancedb");
        assert_eq!(tmpl.backends[0].uri, "data/vectors.lance");
        assert_eq!(tmpl.backends[1].name, "dragonfly");
    }

    #[test]
    fn test_pipeline_backend_refs() {
        let tmpl = from_json(BACKEND_TEMPLATE).unwrap();
        assert_eq!(tmpl.pipeline[0].backend.as_deref(), Some("lancedb"));
        assert_eq!(tmpl.pipeline[0].backend_key.as_deref(), Some("embeddings"));
        assert_eq!(tmpl.pipeline[3].backend.as_deref(), Some("dragonfly"));
        assert_eq!(tmpl.pipeline[3].backend_key.as_deref(), Some("results:"));
    }

    #[test]
    fn test_template_hash_deterministic() {
        let tmpl = from_json(VALID_TEMPLATE).unwrap();
        let h1 = template_hash(&tmpl);
        let h2 = template_hash(&tmpl);
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);
    }

    #[test]
    fn test_template_hash_different_templates() {
        let t1 = from_json(VALID_TEMPLATE).unwrap();
        let t2 = from_json(BACKEND_TEMPLATE).unwrap();
        assert_ne!(template_hash(&t1), template_hash(&t2));
    }

    #[test]
    fn test_precompile_queue_enqueue_dedup() {
        let tmpl = from_json(VALID_TEMPLATE).unwrap();
        let mut queue = PrecompileQueue::new();
        let h1 = queue.enqueue(tmpl.clone());
        let h2 = queue.enqueue(tmpl);
        assert_eq!(h1, h2);
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_precompile_queue_lifecycle() {
        let t1 = from_json(VALID_TEMPLATE).unwrap();
        let t2 = from_json(BACKEND_TEMPLATE).unwrap();
        let mut queue = PrecompileQueue::new();

        let h1 = queue.enqueue(t1);
        let h2 = queue.enqueue(t2);
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.pending().len(), 2);

        assert!(queue.mark_compiled(h1, 0xDEAD_BEEF));
        assert_eq!(queue.pending().len(), 1);
        let entry = queue.lookup(h1).unwrap();
        assert_eq!(
            entry.state,
            CompileState::Compiled {
                code_addr: 0xDEAD_BEEF
            }
        );

        assert!(queue.mark_compiled(h2, 0xCAFE_BABE));
        assert_eq!(queue.pending().len(), 0);
    }

    #[test]
    fn test_prefetch_next() {
        let t1 = from_json(VALID_TEMPLATE).unwrap();
        let t2 = from_json(BACKEND_TEMPLATE).unwrap();
        let mut queue = PrecompileQueue::new();

        let h1 = queue.enqueue(t1);
        let h2 = queue.enqueue(t2);

        assert!(queue.prefetch_next(h1).is_none());

        queue.mark_compiled(h1, 0x1000);
        queue.mark_compiled(h2, 0x2000);

        assert_eq!(queue.prefetch_next(h1), Some(0x2000));
        assert!(queue.prefetch_next(h2).is_none());
    }

    #[test]
    fn test_eviction() {
        let tmpl = from_json(VALID_TEMPLATE).unwrap();
        let mut queue = PrecompileQueue::new();
        let h = queue.enqueue(tmpl);
        queue.mark_compiled(h, 0x1000);
        queue.mark_evicted(h);
        assert_eq!(queue.lookup(h).unwrap().state, CompileState::Evicted);
    }
}
