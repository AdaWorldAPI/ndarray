//! WAL Precompile Queue + Prefetch Addressing
//!
//! Templates are appended in order. The queue supports:
//! - **Enqueue**: add a template for compilation (deduplicates by hash)
//! - **Lookup**: check if a template is already compiled (prefetch hit)
//! - **Prefetch hint**: given the current template, return the next entry's
//!   hash so the caller can issue a memory prefetch for its code page
//!
//! The queue is intentionally simple — a `Vec` with linear scan. For
//! production use at scale, swap in an LRU or LFU map.

use super::template::{template_hash, JitsonTemplate};

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
    use crate::hpc::jitson::template::from_json;

    const VALID_TEMPLATE: &str = r#"{
        "version": 1,
        "kernel": "hamming_distance",
        "scan": { "threshold": 2048, "record_size": 256, "top_k": 10 },
        "pipeline": [
            { "stage": "xor",    "avx512": "vpxord" },
            { "stage": "popcnt", "avx512": "vpopcntd" },
            { "stage": "reduce", "avx512": "vpord" }
        ],
        "features": { "avx512f": true, "avx512vl": true, "avx512vpopcntdq": true }
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
    fn test_enqueue_dedup() {
        let tmpl = from_json(VALID_TEMPLATE).unwrap();
        let mut queue = PrecompileQueue::new();
        let h1 = queue.enqueue(tmpl.clone());
        let h2 = queue.enqueue(tmpl);
        assert_eq!(h1, h2);
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_lifecycle() {
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
        assert_eq!(entry.state, CompileState::Compiled { code_addr: 0xDEAD_BEEF });

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
