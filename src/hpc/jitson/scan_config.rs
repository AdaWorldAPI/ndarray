//! # JIT Hybrid Scan — SIMD Kernel Registration and Non-JIT Scan Path
//!
//! This module implements the **hybrid scan** pattern: Cranelift JIT compiles
//! the outer loop with baked-in immediates (threshold, record_size, top_k),
//! and calls hand-optimized SIMD kernels via registered external symbols.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────┐
//! │  Cranelift JIT Outer Loop                           │
//! │  ┌───────────────────────────────────────────────┐  │
//! │  │ for record in records:                        │  │
//! │  │   dist = CALL hamming_distance(query, record) │  │
//! │  │   if dist < threshold:                        │  │
//! │  │     heap_push(dist, record_id)                │  │
//! │  └───────────────────────────────────────────────┘  │
//! │                         │                           │
//! │                    CALL (extern)                    │
//! │                         ▼                           │
//! │  ┌───────────────────────────────────────────────┐  │
//! │  │  Hand-Optimized SIMD Kernel                   │  │
//! │  │  - VPOPCNTDQ (64 bytes/iter, 4x ILP)         │  │
//! │  │  - PREFETCHT0 (hand-tuned prefetch)           │  │
//! │  │  - AVX-512 BW masking                         │  │
//! │  └───────────────────────────────────────────────┘  │
//! └─────────────────────────────────────────────────────┘
//! ```

/// Configuration for a JIT-compiled scan operation.
#[derive(Clone, Debug)]
pub struct ScanConfig {
    /// Distance threshold — candidates above this are discarded early.
    pub threshold: u64,
    /// Size of each record in bytes (e.g., 2048 for 2KB fingerprints).
    pub record_size: usize,
    /// Number of top results to keep (top-k heap size).
    pub top_k: usize,
    /// Query fingerprint to compare against.
    pub query: Vec<u8>,
}

/// Result of a JIT-compiled scan.
#[derive(Clone, Debug)]
pub struct ScanResult {
    /// (distance, record_index) pairs, sorted by distance ascending.
    pub hits: Vec<(u64, usize)>,
    /// Total records scanned.
    pub records_scanned: usize,
    /// Records that passed the threshold filter.
    pub candidates_found: usize,
}

/// Trait for registering SIMD kernels as external JIT symbols.
///
/// Implementations provide function pointers that Cranelift's
/// `JITBuilder::symbol_lookup_fn()` can resolve at JIT compile time.
pub trait SimdKernelRegistry {
    /// Register the hamming_distance kernel.
    /// Signature: `fn(a: *const u8, b: *const u8, len: usize) -> u64`
    fn hamming_distance_ptr(&self) -> *const u8;

    /// Register the cosine_i8 kernel (VNNI accelerated).
    /// Signature: `fn(a: *const i8, b: *const i8, len: usize) -> f32`
    fn cosine_i8_ptr(&self) -> *const u8;

    /// Register the dot_f32 kernel.
    /// Signature: `fn(a: *const f32, b: *const f32, len: usize) -> f32`
    fn dot_f32_ptr(&self) -> *const u8;
}

/// Default kernel registry using the SIMD implementations from this crate.
pub struct DefaultKernelRegistry;

impl SimdKernelRegistry for DefaultKernelRegistry {
    fn hamming_distance_ptr(&self) -> *const u8 {
        hamming_trampoline as *const u8
    }

    fn cosine_i8_ptr(&self) -> *const u8 {
        cosine_i8_trampoline as *const u8
    }

    fn dot_f32_ptr(&self) -> *const u8 {
        dot_f32_trampoline as *const u8
    }
}

/// C-ABI trampoline for hamming_distance that Cranelift JIT can call.
///
/// # Safety
///
/// `a` and `b` must point to valid memory of at least `len` bytes.
extern "C" fn hamming_trampoline(a: *const u8, b: *const u8, len: usize) -> u64 {
    // SAFETY: Caller (Cranelift JIT or scan_hamming) guarantees `a` and `b` point
    // to valid memory of at least `len` bytes, as documented in the # Safety section.
    let a_slice = unsafe { core::slice::from_raw_parts(a, len) };
    let b_slice = unsafe { core::slice::from_raw_parts(b, len) };
    crate::hpc::bitwise::hamming_distance_raw(a_slice, b_slice)
}

/// C-ABI trampoline for cosine_i8 similarity.
///
/// # Safety
///
/// `a` and `b` must point to valid memory of at least `len` bytes.
extern "C" fn cosine_i8_trampoline(a: *const i8, b: *const i8, len: usize) -> f32 {
    // SAFETY: Caller (Cranelift JIT) guarantees `a` and `b` point to valid memory
    // of at least `len` bytes, as documented in the # Safety section.
    let a_i8 = unsafe { core::slice::from_raw_parts(a, len) };
    let b_i8 = unsafe { core::slice::from_raw_parts(b, len) };
    let mut dot: i32 = 0;
    let mut norm_a: i32 = 0;
    let mut norm_b: i32 = 0;
    for i in 0..len {
        let ai = a_i8[i] as i32;
        let bi = b_i8[i] as i32;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }
    let denom = ((norm_a as f64).sqrt() * (norm_b as f64).sqrt()) as f32;
    if denom == 0.0 {
        0.0
    } else {
        dot as f32 / denom
    }
}

/// C-ABI trampoline for dot_f32 that Cranelift JIT can call.
///
/// # Safety
///
/// `a` and `b` must point to valid memory of at least `len * 4` bytes.
extern "C" fn dot_f32_trampoline(a: *const f32, b: *const f32, len: usize) -> f32 {
    // SAFETY: Caller (Cranelift JIT) guarantees `a` and `b` point to valid memory
    // of at least `len * 4` bytes (len f32 elements), as documented in the # Safety section.
    let a_slice = unsafe { core::slice::from_raw_parts(a, len) };
    let b_slice = unsafe { core::slice::from_raw_parts(b, len) };
    a_slice.iter().zip(b_slice.iter()).map(|(x, y)| x * y).sum()
}

/// Scalar fallback for hamming distance.
#[allow(dead_code)]
fn hamming_scalar(a: &[u8], b: &[u8]) -> u64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones() as u64)
        .sum()
}

/// Perform a scan using the SIMD kernels directly (non-JIT path).
///
/// This is the pure-Rust implementation that the JIT outer loop would call
/// through the registered trampolines. Useful for benchmarking the overhead
/// of JIT compilation vs direct dispatch.
pub fn scan_hamming(config: &ScanConfig, data: &[u8]) -> ScanResult {
    let record_size = config.record_size;
    let num_records = data.len() / record_size;
    let mut hits: Vec<(u64, usize)> = Vec::with_capacity(config.top_k);

    for i in 0..num_records {
        let record = &data[i * record_size..(i + 1) * record_size];
        let dist = hamming_trampoline(config.query.as_ptr(), record.as_ptr(), record_size);
        if dist <= config.threshold {
            if hits.len() < config.top_k {
                hits.push((dist, i));
                // Bubble up to maintain max-heap property
                let mut j = hits.len() - 1;
                while j > 0 {
                    let parent = (j - 1) / 2;
                    if hits[j].0 > hits[parent].0 {
                        hits.swap(j, parent);
                        j = parent;
                    } else {
                        break;
                    }
                }
            } else if dist < hits[0].0 {
                // Replace the max element
                hits[0] = (dist, i);
                // Sift down
                let mut j = 0;
                loop {
                    let left = 2 * j + 1;
                    let right = 2 * j + 2;
                    let mut largest = j;
                    if left < hits.len() && hits[left].0 > hits[largest].0 {
                        largest = left;
                    }
                    if right < hits.len() && hits[right].0 > hits[largest].0 {
                        largest = right;
                    }
                    if largest != j {
                        hits.swap(j, largest);
                        j = largest;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    let candidates_found = hits.len();
    hits.sort_by_key(|&(dist, _)| dist);

    ScanResult {
        hits,
        records_scanned: num_records,
        candidates_found,
    }
}

/// Symbol table for JIT integration.
///
/// Returns a list of (name, function_pointer) pairs that can be registered
/// with `cranelift_jit::JITBuilder::symbol_lookup_fn()`.
pub fn jit_symbol_table() -> Vec<(&'static str, *const u8)> {
    vec![
        ("hamming_distance", hamming_trampoline as *const u8),
        ("cosine_i8", cosine_i8_trampoline as *const u8),
        ("dot_f32", dot_f32_trampoline as *const u8),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_hamming_basic() {
        let query = vec![0xFF_u8; 256]; // all-ones query
        let mut data = vec![0x00_u8; 256 * 10]; // 10 records, all zeros

        // Record 3: set some bits to match closer
        for j in 0..256 {
            data[3 * 256 + j] = 0xF0; // half bits set
        }
        // Record 7: perfect match
        for j in 0..256 {
            data[7 * 256 + j] = 0xFF;
        }

        let config = ScanConfig {
            threshold: 2048,
            record_size: 256,
            top_k: 5,
            query,
        };

        let result = scan_hamming(&config, &data);
        assert_eq!(result.records_scanned, 10);
        assert!(result.hits.len() >= 2);

        // Record 7 should be the closest (distance 0)
        assert_eq!(result.hits[0].0, 0);
        assert_eq!(result.hits[0].1, 7);
    }

    #[test]
    fn test_jit_symbol_table() {
        let symbols = jit_symbol_table();
        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0].0, "hamming_distance");
        assert_eq!(symbols[1].0, "cosine_i8");
        assert_eq!(symbols[2].0, "dot_f32");
        for (name, ptr) in &symbols {
            assert!(!ptr.is_null(), "symbol {} is null", name);
        }
    }

    #[test]
    fn test_hamming_trampoline() {
        let a = [0xFF_u8; 64];
        let b = [0x00_u8; 64];
        let dist = hamming_trampoline(a.as_ptr(), b.as_ptr(), 64);
        assert_eq!(dist, 512); // 64 bytes * 8 bits = 512 bit differences
    }

    #[test]
    fn test_dot_f32_trampoline() {
        let a = [1.0_f32; 16];
        let b = [2.0_f32; 16];
        let result = dot_f32_trampoline(a.as_ptr(), b.as_ptr(), 16);
        assert!((result - 32.0).abs() < 1e-6);
    }
}
