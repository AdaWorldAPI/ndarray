#![allow(
    clippy::assign_op_pattern,
    clippy::too_many_arguments,
    clippy::manual_range_contains,
    clippy::needless_range_loop,
    clippy::type_complexity
)]
//! HPC extensions for ndarray — ported from rustynum.
//!
//! This module provides high-performance computing extensions:
//! - BLAS Level 1/2/3 operations as extension traits
//! - Statistics (median, var, std, percentile)
//! - Activation functions (sigmoid, softmax, log_softmax)
//! - HDC (Hyperdimensional Computing) operations
//! - CogRecord 4-channel cognitive units
//! - Graph operations with VerbCodebook
//! - BF16 and Int8 quantized GEMM
//! - LAPACK factorizations (LU, Cholesky, QR)
//! - FFT (forward, inverse, real-to-complex)
//! - VML (vectorized math library)

// SIMD capability singleton — detect once, all modules share
pub mod simd_caps;
// LazyLock frozen SIMD dispatch — function pointers selected once at startup
pub mod simd_dispatch;

pub mod blas_level1;
pub mod blas_level2;
pub mod blas_level3;
pub mod statistics;
pub mod activations;
pub mod hdc;
pub mod bitwise;
pub mod projection;
pub mod cogrecord;
pub mod graph;
pub mod quantized;
pub mod lapack;
pub mod fft;
pub mod vml;
pub mod packed;

// Cognitive layer types (migrated from rustynum-core)
#[allow(missing_docs)]
pub mod fingerprint;
#[allow(missing_docs)]
pub mod plane;
#[allow(missing_docs)]
pub mod seal;
#[allow(missing_docs)]
pub mod node;
#[allow(missing_docs)]
pub mod cascade;
#[allow(missing_docs)]
pub mod bf16_truth;
#[allow(missing_docs)]
pub mod causality;
#[allow(missing_docs)]
pub mod causal_diff;
#[allow(missing_docs)]
pub mod nars;
#[allow(missing_docs)]
pub mod blackboard;
#[allow(missing_docs)]
pub mod bnn;
#[allow(missing_docs)]
pub mod clam;
#[allow(missing_docs)]
pub mod clam_search;
#[allow(missing_docs)]
pub mod clam_compress;
#[allow(missing_docs)]
pub mod arrow_bridge;
#[allow(missing_docs)]
pub mod merkle_tree;

// Sprint 1: Quick Wins (hot-path gap fill)
#[allow(missing_docs)]
pub mod cam_index;
#[allow(missing_docs)]
pub mod prefilter;
#[allow(missing_docs)]
pub mod binding_matrix;
#[allow(missing_docs)]
pub mod qualia_gate;
#[allow(missing_docs)]
pub mod dn_tree;

// Sprint 3: CLAM + BNN Ports
#[allow(missing_docs)]
pub mod bnn_cross_plane;
#[allow(missing_docs)]
pub mod bnn_causal_trajectory;

// Qualia system: 16-channel phenomenal coloring
#[allow(missing_docs)]
pub mod qualia;

// Sprint 2: Core Cognitive Layer
#[allow(missing_docs)]
pub mod kernels;
#[allow(missing_docs)]
pub mod organic;
#[allow(missing_docs)]
pub mod substrate;
#[allow(missing_docs)]
pub mod tekamolo;
#[allow(missing_docs)]
pub mod vsa;
#[allow(missing_docs)]
pub mod spo_bundle;
#[allow(missing_docs)]
pub mod deepnsm;
#[allow(missing_docs)]
pub mod surround_metadata;
#[allow(missing_docs, dead_code)]
pub mod cyclic_bundle;
#[allow(missing_docs, dead_code)]
pub mod compression_curves;
#[allow(missing_docs)]
pub mod crystal_encoder;
#[allow(missing_docs)]
pub mod udf_kernels;

// Session C: bgz17 dual-path integration
#[allow(missing_docs)]
pub mod bgz17_bridge;
#[allow(missing_docs)]
pub mod palette_distance;
#[allow(missing_docs)]
pub mod layered_distance;
#[allow(missing_docs)]
pub mod parallel_search;

// ZeckF64 progressive edge encoding + batch/top-k
#[allow(missing_docs)]
pub mod zeck;

// SIMD-accelerated spatial / byte-scan / hash utilities
pub mod distance;
pub mod byte_scan;
pub mod spatial_hash;

// Variable-width palette index codec (Minecraft-style bit packing)
#[allow(missing_docs)]
pub mod palette_codec;

// SIMD-accelerated HPC modules (block properties, nibble light data, AABB collision)
#[allow(missing_docs)]
pub mod property_mask;
#[allow(missing_docs)]
pub mod nibble;
#[allow(missing_docs)]
pub mod aabb;

// Holographic phase-space operations (ported from rustynum-holo)
#[allow(missing_docs)]
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
pub mod holo;

// CAM-PQ: Content-Addressable Memory as Product Quantization
// Unifies FAISS PQ6x8 and CLAM 48-bit archetypes. 170× compression, 500M cands/sec.
#[allow(missing_docs)]
pub mod cam_pq;

/// GGUF model file reader — extract f32 weights from quantized models.
#[allow(missing_docs)]
pub mod gguf;

/// Streaming GGUF → bgz17 indexer. One tensor at a time, bounded RAM.
#[allow(missing_docs)]
pub mod gguf_indexer;

/// HTTP range reader — Read + Seek over HTTP for streaming GGUF from HuggingFace.
#[allow(missing_docs)]
pub mod http_reader;

/// Jina embedding codec — GGUF → Base17 → Palette → CausalEdge64.
#[allow(missing_docs)]
pub mod jina;

/// Shared model primitives — safetensors, SIMD layers, API types.
#[allow(missing_docs)]
pub mod models;

/// GPT-2 inference engine — full forward pass + OpenAI-compatible API types.
#[allow(missing_docs)]
pub mod gpt2;

/// Stable Diffusion inference — CLIP + UNet + VAE + DDIM scheduler.
#[allow(missing_docs)]
pub mod stable_diffusion;

/// OpenChat 3.5 inference — Mistral-7B architecture (GQA + RoPE + RMSNorm + SiLU).
#[allow(missing_docs)]
pub mod openchat;

// jitson: JSON config → scan pipeline (parser, validator, template, precompile, packed)
// Always available — no Cranelift dependency.
#[allow(missing_docs)]
pub mod jitson;

// jitson_cranelift: Cranelift JIT compilation backend (ScanParams, JitEngine, ScanKernel)
// Only compiled with the "jit-native" feature flag.
#[cfg(feature = "jit-native")]
#[allow(missing_docs)]
pub mod jitson_cranelift;

#[cfg(test)]
mod e2e_tests {
    //! End-to-end pipeline test: Fingerprint → Node → Seal → Cascade → CLAM → Causality → BNN

    use super::fingerprint::Fingerprint;
    use super::node::{Node, SPO, S__, _P_, __O};
    use super::seal::Seal;
    use super::cascade::{Cascade, Band};
    use super::clam::{ClamTree, knn_brute};
    use super::bf16_truth::PackedQualia;
    use super::causality::{causality_decompose, CausalityDirection};
    use super::bnn::bnn_dot;
    use super::blackboard::Blackboard;

    #[test]
    fn pipeline_fingerprint_to_node_to_seal() {
        // 1. Create two nodes and accumulate evidence
        let mut a = Node::random(42);
        let mut b = Node::random(99);

        // 2. Measure distance (SPO full)
        let d = a.distance(&mut b, SPO);
        match d {
            super::plane::Distance::Measured { disagreement, overlap, .. } => {
                assert!(overlap > 0, "random nodes should have overlap");
                assert!(disagreement > 0, "different seeds should disagree");
            }
            super::plane::Distance::Incomparable => panic!("random nodes should be comparable"),
        }

        // 3. Seal integrity: build from scratch for deterministic test
        let mut p = super::plane::Plane::new();
        p.encounter("hello");
        p.encounter("hello");
        let root = p.merkle();
        assert_eq!(p.verify(&root), Seal::Wisdom);

        // 4. Mutate and detect change
        p.encounter("world");
        assert_eq!(p.verify(&root), Seal::Staunen);
    }

    #[test]
    fn pipeline_cascade_search() {
        let vec_bytes = 256;
        let num_vectors = 50;

        // Build a database of random fingerprints
        let mut database = Vec::with_capacity(vec_bytes * num_vectors);
        for i in 0..num_vectors {
            let fp = Fingerprint::<32>::from_words({
                let mut words = [0u64; 32];
                let mut seed = (i as u64).wrapping_add(1).wrapping_mul(0x9E3779B97F4A7C15);
                for w in words.iter_mut() {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    *w = seed;
                }
                words
            });
            database.extend_from_slice(fp.as_bytes());
        }

        // Query = first vector (should find itself at distance 0)
        let query = database[0..vec_bytes].to_vec();
        let cascade = Cascade::from_threshold(vec_bytes as u64 * 4, vec_bytes);
        let results = cascade.query(&query, &database, vec_bytes, num_vectors);
        assert!(results.iter().any(|r| r.index == 0 && r.hamming == 0));
    }

    #[test]
    fn pipeline_clam_knn() {
        let vec_len = 32;
        let n = 20;
        // Create distinct vectors: all zeros except vector i has byte i set to 0xFF
        let mut data = vec![0u8; n * vec_len];
        for i in 0..n {
            data[i * vec_len + (i % vec_len)] = 0xFF;
        }
        // Query = vector 0 (byte 0 is 0xFF, rest zeros)
        let query = data[0..vec_len].to_vec();

        // Brute force k-NN
        let result = knn_brute(&data, vec_len, &query, 5);
        assert_eq!(result.hits.len(), 5);
        // First hit should be exact match (distance 0)
        assert_eq!(result.hits[0].1, 0);
        assert_eq!(result.hits[0].0, 0);

        // Build CLAM tree
        let tree = ClamTree::build(&data, vec_len, 3);
        assert!(!tree.nodes.is_empty());
        assert_eq!(tree.root().cardinality, n);
    }

    #[test]
    fn pipeline_causality_decomposition() {
        let mut a = PackedQualia::zero();
        let b = PackedQualia::zero();
        a.resonance[4] = 10;   // warmth: positive → Forward
        a.resonance[6] = -5;   // social: negative → Backward
        a.resonance[8] = 3;    // sacredness: positive → Forward

        let dec = causality_decompose(&a, &b, None);
        assert_eq!(dec.warmth_dir, CausalityDirection::Forward);
        assert_eq!(dec.social_dir, CausalityDirection::Backward);
        assert_eq!(dec.sacredness_dir, CausalityDirection::Forward);
    }

    #[test]
    fn pipeline_bnn_inference() {
        let act = Fingerprint::<256>::ones();
        let weight = Fingerprint::<256>::ones();
        let result = bnn_dot(&act, &weight);
        assert_eq!(result.match_count, 16384);
        assert!((result.score - 1.0).abs() < 1e-6);

        let zero = Fingerprint::<256>::zero();
        let result2 = bnn_dot(&zero, &weight);
        assert_eq!(result2.match_count, 0);
        assert!((result2.score - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn pipeline_blackboard_arena() {
        let mut bb = Blackboard::new();
        bb.alloc_vec_f32("activations", vec![0.0f32; 16384]);
        bb.alloc_vec_u8("fingerprint", vec![0u8; 2048]);

        let act = bb.get_mut::<Vec<f32>>("activations").unwrap();
        act[0] = 1.0;
        assert_eq!(act.len(), 16384);

        let fp = bb.get_mut::<Vec<u8>>("fingerprint").unwrap();
        fp[0] = 0xFF;
        assert_eq!(fp.len(), 2048);

        // Verify reads work
        assert_eq!(bb.get::<Vec<f32>>("activations").unwrap()[0], 1.0);
        assert_eq!(bb.get::<Vec<u8>>("fingerprint").unwrap()[0], 0xFF);
        assert!(bb.contains("activations"));
        assert!(!bb.contains("nonexistent"));
    }

    #[test]
    fn pipeline_full_e2e() {
        // Full pipeline: Node → truth → causality → cascade → BNN
        let mut node_a = Node::random(1);
        let mut node_b = Node::random(2);

        // Extract truth values
        let truth_a = node_a.truth(SPO);
        let truth_b = node_b.truth(SPO);
        assert!(truth_a.evidence > 0);
        assert!(truth_b.evidence > 0);

        // Measure per-plane distances
        let d_s = node_a.distance(&mut node_b, S__);
        let d_p = node_a.distance(&mut node_b, _P_);
        let d_o = node_a.distance(&mut node_b, __O);
        let d_full = node_a.distance(&mut node_b, SPO);

        // All should be Measured for random nodes
        assert!(matches!(d_s, super::plane::Distance::Measured { .. }));
        assert!(matches!(d_p, super::plane::Distance::Measured { .. }));
        assert!(matches!(d_o, super::plane::Distance::Measured { .. }));
        assert!(matches!(d_full, super::plane::Distance::Measured { .. }));

        // Seal verification
        let root = node_a.s.merkle();
        assert_eq!(node_a.s.verify(&root), Seal::Wisdom);

        // Cascade band classification
        let cascade = Cascade::from_threshold(1000, 2048);
        assert_eq!(cascade.expose(100), Band::Foveal);
        assert_eq!(cascade.expose(1500), Band::Reject);

        // BNN inference
        let bits_a = node_a.s.bits().clone();
        let bits_b = node_b.s.bits().clone();
        let bnn_result = bnn_dot(&bits_a, &bits_b);
        assert!(bnn_result.score > -1.0 && bnn_result.score < 1.0);
    }
}
