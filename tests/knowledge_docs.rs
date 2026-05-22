/// Tests for PR-X12 knowledge-base document invariants.
///
/// The PR changes are entirely in `.claude/knowledge/*.md` files.  These
/// tests read the files from the repository root (via `CARGO_MANIFEST_DIR`)
/// and assert the structural and content invariants that the PR commits.
///
/// Each test function covers a distinct invariant so failures are easy to
/// attribute.

use std::fs;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points at the crate root, which IS the repo root for
    // this workspace (Cargo.toml at /home/jailuser/git/Cargo.toml).
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_knowledge(filename: &str) -> String {
    let path = repo_root().join(".claude").join("knowledge").join(filename);
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Could not read {filename}: {e}"))
}

// ─── pr-x12-bgz-jc-substrate-synergies.md ────────────────────────────────────

#[test]
fn bgz_jc_gap_g1_label_in_section_heading() {
    // §5.1 heading must carry the canonical gap ID "(Gap **G-1**)" so that
    // cross-document references can cite it unambiguously.
    let doc = read_knowledge("pr-x12-bgz-jc-substrate-synergies.md");
    assert!(
        doc.contains("### 5.1 `jd-nd` — the missing ndarray-side proof crate (Gap **G-1**)"),
        "§5.1 heading must include '(Gap **G-1**)' label"
    );
}

#[test]
fn bgz_jc_gap_g2_label_in_section_heading() {
    // §5.2 heading must carry the canonical gap ID "(Gap **G-2**)".
    let doc = read_knowledge("pr-x12-bgz-jc-substrate-synergies.md");
    assert!(
        doc.contains("### 5.2 Cronbach / ICC research crate (Gap **G-2**)"),
        "§5.2 heading must include '(Gap **G-2**)' label"
    );
}

#[test]
fn bgz_jc_old_g1_g2_labels_not_used_in_headings() {
    // The old labels were plain "5.1" and "5.2" headings without gap IDs.
    // Neither heading should be present without the gap label after the PR.
    let doc = read_knowledge("pr-x12-bgz-jc-substrate-synergies.md");
    // The heading without the gap label is no longer acceptable.
    assert!(
        !doc.contains("### 5.1 `jd-nd` — the missing ndarray-side proof crate\n"),
        "§5.1 heading must not appear without the '(Gap **G-1**)' suffix"
    );
    assert!(
        !doc.contains("### 5.2 Cronbach / ICC research crate\n"),
        "§5.2 heading must not appear without the '(Gap **G-2**)' suffix"
    );
}

// ─── pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md ──────────────────────

#[test]
fn cam_pq_cross_ref_table_uses_bgz_jc_prefix() {
    // The cross-reference table must use "bgz-jc G-1" and "bgz-jc G-2"
    // (not the old "G-8" / "G-9" naming that implied a single flat namespace).
    let doc = read_knowledge("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    assert!(
        doc.contains("**bgz-jc G-1**"),
        "cross-ref table must use 'bgz-jc G-1', not the old 'G-8'"
    );
    assert!(
        doc.contains("**bgz-jc G-2**"),
        "cross-ref table must use 'bgz-jc G-2', not the old 'G-9'"
    );
}

#[test]
fn cam_pq_old_g8_g9_labels_absent() {
    // G-8 and G-9 as bold standalone IDs must no longer appear; they implied
    // a shared namespace with the local G-1..G-7 IDs.
    let doc = read_knowledge("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    assert!(
        !doc.contains("**G-8**"),
        "old 'G-8' label must be replaced by 'bgz-jc G-1'"
    );
    assert!(
        !doc.contains("**G-9**"),
        "old 'G-9' label must be replaced by 'bgz-jc G-2'"
    );
}

#[test]
fn cam_pq_cross_ref_table_header_updated() {
    // Table column heading changed from "Gap (prior)" to "Gap (cross-ref)".
    let doc = read_knowledge("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    assert!(
        doc.contains("| Gap (cross-ref) |"),
        "table header must be '| Gap (cross-ref) |'"
    );
    assert!(
        !doc.contains("| Gap (prior) |"),
        "old '| Gap (prior) |' header must be replaced"
    );
}

#[test]
fn cam_pq_namespace_clarification_present() {
    // The PR adds a clarifying paragraph explaining that local G-1..G-7 and
    // bgz-jc's G-1/G-2 are separate namespaces.
    let doc = read_knowledge("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    assert!(
        doc.contains("G-1..G-7 IDs in §5 of *this* doc are local to the cam-pq / sigker / dn_tree binding"),
        "namespace clarification paragraph must be present"
    );
    assert!(
        doc.contains("prefix with the source"),
        "namespace clarification must mention the prefix convention"
    );
}

#[test]
fn cam_pq_section5_1_cross_ref_present() {
    // The table row must cite §5.1 from the bgz-jc doc.
    let doc = read_knowledge("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    assert!(
        doc.contains("(§5.1)"),
        "cross-ref row for bgz-jc G-1 must cite §5.1"
    );
    assert!(
        doc.contains("(§5.2)"),
        "cross-ref row for bgz-jc G-2 must cite §5.2"
    );
}

// ─── pr-x12-canon-resolutions-delta.md ───────────────────────────────────────

#[test]
fn canon_delta_six_categories_of_novel_content() {
    // The PR changed "Five categories" to "Six categories".
    let doc = read_knowledge("pr-x12-canon-resolutions-delta.md");
    assert!(
        doc.contains("Six categories of novel content survive the delta filter"),
        "must say 'Six categories' (R-14 and R-15 were added)"
    );
    assert!(
        !doc.contains("Five categories of novel content survive the delta filter"),
        "old 'Five categories' wording must be replaced"
    );
}

#[test]
fn canon_delta_r14_r15_category_present() {
    // Category 6 "Formal-correctness + stream lane" must exist.
    let doc = read_knowledge("pr-x12-canon-resolutions-delta.md");
    assert!(
        doc.contains("Formal-correctness + stream lane (post-merge)"),
        "category 6 'Formal-correctness + stream lane' must be listed"
    );
    assert!(
        doc.contains("R-14"),
        "R-14 reference must appear in the document"
    );
    assert!(
        doc.contains("R-15"),
        "R-15 reference must appear in the document"
    );
}

#[test]
fn canon_delta_r7_cites_tropical_spmv_kernel() {
    // R-7 entry now names the actual kernel symbol.
    let doc = read_knowledge("pr-x12-canon-resolutions-delta.md");
    assert!(
        doc.contains("bgz17::scalar_sparse::tropical_spmv"),
        "R-7 must cite 'bgz17::scalar_sparse::tropical_spmv' as the actual kernel"
    );
}

#[test]
fn canon_delta_r13_lists_implementation_primitives() {
    // R-13 now names cam_pq, bgz-hhtl-d, dn_tree, and merkle_tree as primitives.
    let doc = read_knowledge("pr-x12-canon-resolutions-delta.md");
    assert!(
        doc.contains("cam_pq"),
        "R-13 must name 'cam_pq' as an implementation primitive"
    );
    assert!(
        doc.contains("bgz-hhtl-d"),
        "R-13 must name 'bgz-hhtl-d' as an implementation primitive"
    );
    assert!(
        doc.contains("dn_tree"),
        "R-13 must name 'dn_tree' as an implementation primitive"
    );
    assert!(
        doc.contains("merkle_tree"),
        "R-13 must name 'merkle_tree' as an implementation primitive"
    );
}

#[test]
fn canon_delta_section_11_formal_correctness_r14() {
    // The new §11 covers the formal-correctness layer (R-14).
    let doc = read_knowledge("pr-x12-canon-resolutions-delta.md");
    assert!(
        doc.contains("Formal-correctness layer (R-14)"),
        "§11 heading 'Formal-correctness layer (R-14)' must exist"
    );
    assert!(
        doc.contains("jc::pflug"),
        "Pillar 10 implementation 'jc::pflug' must be cited"
    );
    assert!(
        doc.contains("jc::hambly_lyons"),
        "Pillar 11 implementation 'jc::hambly_lyons' must be cited"
    );
}

#[test]
fn canon_delta_section_12_stream_signal_r15() {
    // The new §12 covers the stream-signal codec lane (R-15).
    let doc = read_knowledge("pr-x12-canon-resolutions-delta.md");
    assert!(
        doc.contains("Stream-signal codec lane (R-15)"),
        "§12 heading 'Stream-signal codec lane (R-15)' must exist"
    );
    assert!(
        doc.contains("SignatureBasis<DEPTH>"),
        "SignatureBasis<DEPTH> must be mentioned in the stream-signal section"
    );
    assert!(
        doc.contains("sigker::signature_truncated"),
        "must reference the correct kernel 'sigker::signature_truncated'"
    );
}

#[test]
fn canon_delta_r15_not_using_buggy_pde_form() {
    // R-15 explicitly uses signature_truncated, not signature_kernel_pde.
    let doc = read_knowledge("pr-x12-canon-resolutions-delta.md");
    assert!(
        doc.contains("signature_truncated"),
        "must reference 'signature_truncated' (the correct form)"
    );
    // The PDE form is mentioned only to warn against it.
    let pde_occurrences = doc.matches("signature_kernel_pde").count();
    let truncated_occurrences = doc.matches("signature_truncated").count();
    assert!(
        truncated_occurrences >= pde_occurrences,
        "signature_truncated must appear at least as often as signature_kernel_pde \
        (the PDE form is only mentioned as the form NOT used by R-15)"
    );
}

#[test]
fn canon_delta_section_13_load_bearing_paragraph_updated() {
    // §13 (formerly §11) now references R-14 and R-15 in the load-bearing paragraph.
    let doc = read_knowledge("pr-x12-canon-resolutions-delta.md");
    assert!(
        doc.contains("The single load-bearing paragraph (canon-resolutions §13)"),
        "section header must be updated to '§13'"
    );
    assert!(
        doc.contains("substrate-binding follow-up (R-14, R-15)"),
        "load-bearing paragraph must reference R-14 and R-15"
    );
    assert!(
        doc.contains("SignatureBasis<DEPTH>"),
        "load-bearing paragraph must mention SignatureBasis<DEPTH>"
    );
}

#[test]
fn canon_delta_falsifiability_matrix_includes_r14_r15() {
    // The §9 matrix rows for R-14 and R-15 must be present.
    let doc = read_knowledge("pr-x12-canon-resolutions-delta.md");
    assert!(
        doc.contains("R-14 (Pillar 10 active)"),
        "falsifiability matrix must have R-14 Pillar 10 row"
    );
    assert!(
        doc.contains("R-14 (Pillar 11 active)"),
        "falsifiability matrix must have R-14 Pillar 11 row"
    );
    assert!(
        doc.contains("R-15 (SignatureBasis lane)"),
        "falsifiability matrix must have R-15 row"
    );
}

// ─── pr-x12-gguf-llm-weights-encoding.md ─────────────────────────────────────

#[test]
fn gguf_escape_lossless_references_f4_falsifier() {
    // Escape-must-be-lossless paragraph now references §10 falsifier F-4.
    let doc = read_knowledge("pr-x12-gguf-llm-weights-encoding.md");
    assert!(
        doc.contains("§10 falsifier **F-4**"),
        "escape-lossless paragraph must reference '§10 falsifier **F-4**'"
    );
    assert!(
        doc.contains("rANS bypass channel"),
        "F-4 description must mention 'rANS bypass channel'"
    );
}

#[test]
fn gguf_memory_savings_phone_class_kv_cache_caveat() {
    // The PR added a KV-cache caveat to the memory-savings paragraph.
    let doc = read_knowledge("pr-x12-gguf-llm-weights-encoding.md");
    assert!(
        doc.contains("weights are not the only memory load"),
        "memory-savings section must include the KV-cache caveat"
    );
    assert!(
        doc.contains("KV cache scales with context length"),
        "KV-cache caveat must mention that KV cache scales with context length"
    );
    assert!(
        doc.contains("Plan D, M:H-3, R-4"),
        "KV-cache caveat must reference 'Plan D, M:H-3, R-4'"
    );
}

#[test]
fn gguf_memory_savings_headline_updated() {
    // The memory-savings section header now includes "(weights only)".
    let doc = read_knowledge("pr-x12-gguf-llm-weights-encoding.md");
    assert!(
        doc.contains("**Memory savings (weights only):**"),
        "memory-savings header must say '(weights only)'"
    );
}

#[test]
fn gguf_encoding_domain_enum_discriminant_slot() {
    // The PR clarified that EncodingDomain::LLMWeights reserves the slot now
    // even though the LLM-lane decoder lands post-PR-X12.
    let doc = read_knowledge("pr-x12-gguf-llm-weights-encoding.md");
    assert!(
        doc.contains("enum-discriminant slot"),
        "must mention 'enum-discriminant slot' for EncodingDomain::LLMWeights"
    );
    assert!(
        doc.contains("forward-compatibility-locked"),
        "must say the slot is 'forward-compatibility-locked'"
    );
}

// ─── pr-x12-substrate-canon-resolutions.md ───────────────────────────────────

#[test]
fn substrate_canon_intro_lists_r14_r15() {
    // The intro now says "five commitments" (R-11..R-15) not "three".
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("R-14 (formal correctness via `jc` pillars)"),
        "intro must list R-14 among the commitments"
    );
    assert!(
        doc.contains("R-15"),
        "intro must list R-15 among the commitments"
    );
    assert!(
        doc.contains("`SignatureBasis<DEPTH>`"),
        "R-15 description must reference 'SignatureBasis<DEPTH>'"
    );
}

#[test]
fn substrate_canon_citation_ids_updated_to_r15() {
    // Citation ID range must now say "R-1 through R-15".
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("R-1` through `R-15"),
        "citation ID range must be updated to 'R-1 through R-15'"
    );
    assert!(
        !doc.contains("R-1` through `R-13"),
        "old citation ID range 'R-1 through R-13' must be replaced"
    );
}

#[test]
fn substrate_canon_tropical_spmv_actual_kernel_home() {
    // The PR added a subsection naming the actual tropical_spmv symbol.
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("Actual kernel home (current)"),
        "must contain 'Actual kernel home (current)' subsection"
    );
    assert!(
        doc.contains("lance-graph::bgz17::scalar_sparse::tropical_spmv"),
        "actual kernel must be cited as 'lance-graph::bgz17::scalar_sparse::tropical_spmv'"
    );
    assert!(
        doc.contains("blasgraph` namespace is the eventual abstraction"),
        "must clarify that blasgraph is the eventual (not current) abstraction"
    );
}

#[test]
fn substrate_canon_r13_implementation_primitives_table() {
    // The PR added an implementation-primitives table under R-13.
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("Implementation primitives (current substrate, no new code required)"),
        "R-13 must have an 'Implementation primitives' table header"
    );
    // Table must list all four ndarray primitives.
    assert!(
        doc.contains("ndarray::hpc::cam_pq::CamCodebook"),
        "R-13 primitives table must list 'ndarray::hpc::cam_pq::CamCodebook'"
    );
    assert!(
        doc.contains("ndarray::hpc::dn_tree"),
        "R-13 primitives table must list 'ndarray::hpc::dn_tree'"
    );
    assert!(
        doc.contains("ndarray::hpc::merkle_tree"),
        "R-13 primitives table must list 'ndarray::hpc::merkle_tree'"
    );
    // The gossip protocol must also be listed.
    assert!(
        doc.contains("`q2` (external"),
        "R-13 primitives table must list 'q2 (external)' for gossip"
    );
}

#[test]
fn substrate_canon_r14_formal_correctness_section_present() {
    // R-14 section header and key content must exist.
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("### R-14 — Formal correctness via `lance-graph::jc` pillars"),
        "R-14 section heading must be present"
    );
    assert!(
        doc.contains("Pillar 10"),
        "R-14 must describe Pillar 10"
    );
    assert!(
        doc.contains("Pillar 11"),
        "R-14 must describe Pillar 11"
    );
    assert!(
        doc.contains("Pflug-Pichler"),
        "Pillar 10 must cite Pflug-Pichler"
    );
    assert!(
        doc.contains("Hambly-Lyons"),
        "Pillar 11 must cite Hambly-Lyons"
    );
}

#[test]
fn substrate_canon_r14_pillar_11_feature_gate_documented() {
    // Pillar 11 is feature-gated; the section must document this.
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("--features hambly-lyons"),
        "R-14 must document the '--features hambly-lyons' gate for Pillar 11"
    );
    assert!(
        doc.contains("PR #348"),
        "R-14 must reference 'PR #348' where Pillar 11 landed"
    );
}

#[test]
fn substrate_canon_r14_falsifiability_condition() {
    // R-14 must state what would falsify the claim.
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("**Falsifies if.**"),
        "R-14 must have a 'Falsifies if.' condition"
    );
    assert!(
        doc.contains("Pillar 10 ever flips state"),
        "R-14's falsifiability must mention 'Pillar 10 ever flips state'"
    );
}

#[test]
fn substrate_canon_r15_signature_basis_section_present() {
    // R-15 section header and Rust snippet must exist.
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("### R-15 — `SignatureBasis<const DEPTH: usize>` as `Basis<f32>` impl"),
        "R-15 section heading must be present"
    );
    assert!(
        doc.contains("impl<const DEPTH: usize> Basis<f32> for SignatureBasis<DEPTH>"),
        "R-15 must include the Rust trait implementation"
    );
}

#[test]
fn substrate_canon_r15_invert_is_unimplemented() {
    // The invert method is explicitly documented as N/A (unique up to equivalence).
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("signature inversion is N/A"),
        "R-15's invert must be documented as N/A"
    );
    assert!(
        doc.contains("tree-like equivalence"),
        "R-15 must explain path uniqueness is only up to 'tree-like equivalence'"
    );
}

#[test]
fn substrate_canon_r15_fifth_plan_g_lane() {
    // R-15 commits a fifth Plan G lane for stream signals.
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("Plan G gets a fifth lane"),
        "R-15 must commit Plan G to a 'fifth lane'"
    );
    assert!(
        doc.contains("Stream signal"),
        "the fifth lane must be named 'Stream signal'"
    );
}

#[test]
fn substrate_canon_falsifiability_matrix_rows_r14_r15() {
    // Three new rows must appear in the §9 falsifiability matrix.
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("R-14 (Pillar 10 active)"),
        "falsifiability matrix must have 'R-14 (Pillar 10 active)' row"
    );
    assert!(
        doc.contains("R-14 (Pillar 11 active)"),
        "falsifiability matrix must have 'R-14 (Pillar 11 active)' row"
    );
    assert!(
        doc.contains("R-15 (SignatureBasis lane)"),
        "falsifiability matrix must have 'R-15 (SignatureBasis lane)' row"
    );
    // Probe criteria must be present.
    assert!(
        doc.contains("forward < 1e-9"),
        "matrix rows must specify the forward-test threshold"
    );
    assert!(
        doc.contains("converse > 0.05"),
        "matrix rows must specify the converse-test threshold"
    );
    assert!(
        doc.contains("ratio ≥ 1e6"),
        "matrix rows must specify the discrimination ratio threshold"
    );
}

#[test]
fn substrate_canon_summary_says_fifteen_resolutions() {
    // The summary section must say "fifteen resolutions" (was "thirteen").
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("**The fifteen resolutions** R-1 through R-15"),
        "summary must say 'fifteen resolutions R-1 through R-15'"
    );
    assert!(
        !doc.contains("**The thirteen resolutions**"),
        "old 'thirteen resolutions' text must be gone"
    );
}

#[test]
fn substrate_canon_r7_summary_cites_tropical_spmv() {
    // The summary list item for R-7 must name the actual kernel.
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    assert!(
        doc.contains("bgz17::scalar_sparse::tropical_spmv"),
        "R-7 summary must cite 'bgz17::scalar_sparse::tropical_spmv'"
    );
}

#[test]
fn substrate_canon_r13_summary_lists_primitives() {
    // The summary list item for R-13 must name the implementation primitives.
    let doc = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    // R-13 in the summary should mention cam_pq + bgz-hhtl-d + dn_tree + merkle_tree.
    let summary_section = doc
        .find("R-13: Option A (per-shard codebook)")
        .expect("R-13 summary item must exist");
    let tail = &doc[summary_section..summary_section + 300];
    assert!(
        tail.contains("cam_pq") || doc.contains("cam_pq + `bgz-hhtl-d` + `dn_tree` + `merkle_tree`"),
        "R-13 summary must mention the implementation primitives"
    );
}

// ─── pr-x12-woa-multiarch-orchestration.md ───────────────────────────────────

#[test]
fn woa_status_line_uses_polyfill_not_dispatch() {
    // The status line was updated to say "polyfill" instead of "dispatch".
    let doc = read_knowledge("pr-x12-woa-multiarch-orchestration.md");
    assert!(
        doc.contains("per-arch polyfill decisions"),
        "status line must say 'per-arch polyfill decisions'"
    );
    assert!(
        !doc.contains("per-arch dispatch decisions"),
        "old 'per-arch dispatch decisions' must be replaced"
    );
}

#[test]
fn woa_premise_uses_polyfill_contract() {
    // The premise line was updated to "polyfill contract".
    let doc = read_knowledge("pr-x12-woa-multiarch-orchestration.md");
    assert!(
        doc.contains("per-arch polyfill contract"),
        "premise must say 'per-arch polyfill contract'"
    );
    assert!(
        !doc.contains("per-arch dispatch contract"),
        "old 'per-arch dispatch contract' must be replaced"
    );
}

#[test]
fn woa_section3_heading_uses_polyfill() {
    // §3 heading changed from "dispatch" to "compile-time polyfill".
    let doc = read_knowledge("pr-x12-woa-multiarch-orchestration.md");
    assert!(
        doc.contains("## 3. Per-arch substrate via compile-time polyfill"),
        "§3 heading must be 'Per-arch substrate via compile-time polyfill'"
    );
    assert!(
        !doc.contains("## 3. Per-arch dispatch as a substrate property"),
        "old §3 heading must be replaced"
    );
}

#[test]
fn woa_section3_2_heading_build_time_not_runtime() {
    // §3.2 heading changed from "Runtime feature detection" to "Build-time CPU selection".
    let doc = read_knowledge("pr-x12-woa-multiarch-orchestration.md");
    assert!(
        doc.contains("### 3.2 Build-time CPU selection"),
        "§3.2 heading must be 'Build-time CPU selection (not runtime detection)'"
    );
    assert!(
        !doc.contains("### 3.2 Runtime feature detection"),
        "old '### 3.2 Runtime feature detection' heading must be replaced"
    );
}

#[test]
fn woa_no_runtime_detection_idioms_in_narrative() {
    // The PR removes the OnceLock/HwCaps runtime detection model from the
    // narrative (it was in §3.2 example code).  The polyfill approach is
    // purely compile-time.
    let doc = read_knowledge("pr-x12-woa-multiarch-orchestration.md");
    assert!(
        !doc.contains("OnceLock<HwCaps>"),
        "OnceLock<HwCaps> runtime detection must not appear — replaced by compile-time cfg"
    );
    assert!(
        !doc.contains("pub static CAP: OnceLock"),
        "OnceLock static must not appear in narrative"
    );
}

#[test]
fn woa_polyfill_surface_description_present() {
    // The new §3 must describe the three-layer polyfill structure.
    let doc = read_knowledge("pr-x12-woa-multiarch-orchestration.md");
    assert!(
        doc.contains("Polyfill surface — src/simd.rs"),
        "diagram must label 'Polyfill surface — src/simd.rs'"
    );
    assert!(
        doc.contains("Backend — simd_avx512.rs / simd_neon.rs / simd_scalar.rs"),
        "diagram must list backend files"
    );
}

#[test]
fn woa_consumers_never_name_a_backend() {
    // The contract must state that consumers never name a backend symbol.
    let doc = read_knowledge("pr-x12-woa-multiarch-orchestration.md");
    assert!(
        doc.contains("Consumers above never reach in here"),
        "must state consumers never reach into backend files"
    );
    assert!(
        doc.contains("Consumers never name a backend symbol"),
        "must state consumers never name a backend symbol"
    );
}

#[test]
fn woa_section2_commitment_says_polyfill_not_dispatch() {
    // The key commitment (#2 in §9) now says "polyfill transparency".
    let doc = read_knowledge("pr-x12-woa-multiarch-orchestration.md");
    assert!(
        doc.contains("Per-arch polyfill transparency"),
        "commitment #2 must say 'Per-arch polyfill transparency'"
    );
    assert!(
        !doc.contains("Per-arch dispatch transparency"),
        "old 'Per-arch dispatch transparency' must be replaced"
    );
}

#[test]
fn woa_gpu_offload_hook_uses_backend_not_dispatch() {
    // The GPU-offload hook was renamed from dispatch_target to backend_target.
    let doc = read_knowledge("pr-x12-woa-multiarch-orchestration.md");
    assert!(
        doc.contains("backend_target"),
        "GPU offload hook must be named 'backend_target'"
    );
    assert!(
        !doc.contains("dispatch_target"),
        "old 'dispatch_target' name must be replaced"
    );
}

// ─── pr-x12-x266-3dgs-spacetime-upscaling.md ─────────────────────────────────

#[test]
fn x266_requirements_table_has_canon_fixed_status() {
    // The PR added "canon-fixed" status labels to several requirement rows.
    let doc = read_knowledge("pr-x12-x266-3dgs-spacetime-upscaling.md");
    assert!(
        doc.contains("**canon-fixed**"),
        "requirements table must use '**canon-fixed**' status labels"
    );
}

#[test]
fn x266_requirements_table_has_status_legend() {
    // A legend explaining "Canon-fixed" vs "scheduled" must be added.
    let doc = read_knowledge("pr-x12-x266-3dgs-spacetime-upscaling.md");
    assert!(
        doc.contains("\"Canon-fixed\""),
        "requirements table must have a legend defining 'Canon-fixed'"
    );
    assert!(
        doc.contains("**\"scheduled\"**"),
        "requirements table must define 'scheduled' in the legend"
    );
    assert!(
        doc.contains("None of the above have shipping code today"),
        "legend must note that no rows have shipping code"
    );
}

#[test]
fn x266_basis_trait_row_updated_to_canon_fixed() {
    // The `Basis<T>` trait row status was updated from "landed in concept" to "canon-fixed".
    let doc = read_knowledge("pr-x12-x266-3dgs-spacetime-upscaling.md");
    assert!(
        doc.contains("**canon-fixed** (R-1 trait shape committed)"),
        "Basis<T> row must say 'canon-fixed (R-1 trait shape committed)'"
    );
    assert!(
        !doc.contains("landed in concept; implementation in Plan A4"),
        "old 'landed in concept' status must be replaced"
    );
}

#[test]
fn x266_header_byte_row_updated_to_canon_fixed() {
    // The header byte row status was updated to "canon-fixed".
    let doc = read_knowledge("pr-x12-x266-3dgs-spacetime-upscaling.md");
    assert!(
        doc.contains("**canon-fixed** (R-2 commits bits 0-1 = `header_kind`)"),
        "header byte row must say 'canon-fixed (R-2 commits bits 0-1 = header_kind)'"
    );
}

#[test]
fn x266_r13_row_updated_to_canon_fixed() {
    // The federated codebook row status was updated to "canon-fixed".
    let doc = read_knowledge("pr-x12-x266-3dgs-spacetime-upscaling.md");
    assert!(
        doc.contains("**canon-fixed** (R-13 commits Option A: per-shard codebook for Plan F v1)"),
        "R-13 row must say 'canon-fixed (R-13 commits Option A: per-shard codebook for Plan F v1)'"
    );
}

#[test]
fn x266_r3_audit_row_clarified_as_doc_commitment() {
    // The codec body decoupled row now says "doc commitment; CI check pending".
    let doc = read_knowledge("pr-x12-x266-3dgs-spacetime-upscaling.md");
    assert!(
        doc.contains("doc commitment; CI check pending"),
        "R-3 row must clarify 'doc commitment; CI check pending'"
    );
}

// ─── Cross-document consistency ───────────────────────────────────────────────

#[test]
fn cross_doc_tropical_spmv_symbol_consistent() {
    // The tropical_spmv kernel symbol must be spelled the same way across
    // both the substrate-canon-resolutions doc and the canon-resolutions-delta doc.
    let canon = read_knowledge("pr-x12-substrate-canon-resolutions.md");
    let delta = read_knowledge("pr-x12-canon-resolutions-delta.md");

    let symbol = "bgz17::scalar_sparse::tropical_spmv";
    assert!(
        canon.contains(symbol),
        "substrate-canon-resolutions must contain '{symbol}'"
    );
    assert!(
        delta.contains(symbol),
        "canon-resolutions-delta must contain '{symbol}'"
    );
}

#[test]
fn cross_doc_r14_pillar_10_pillar_11_consistent() {
    // Both the canon-resolutions-delta and substrate-canon docs must cite
    // jc::pflug (Pillar 10) and jc::hambly_lyons (Pillar 11).
    for filename in &[
        "pr-x12-canon-resolutions-delta.md",
        "pr-x12-substrate-canon-resolutions.md",
    ] {
        let doc = read_knowledge(filename);
        assert!(
            doc.contains("jc::pflug"),
            "{filename}: must cite 'jc::pflug' for Pillar 10"
        );
        assert!(
            doc.contains("jc::hambly_lyons"),
            "{filename}: must cite 'jc::hambly_lyons' for Pillar 11"
        );
    }
}

#[test]
fn cross_doc_bgz_jc_gap_ids_referenced_correctly_in_cam_pq() {
    // The bgz-jc document labels gaps G-1 and G-2 in §5; the cam-pq doc must
    // reference them with the "bgz-jc" prefix in its cross-reference table.
    let bgz = read_knowledge("pr-x12-bgz-jc-substrate-synergies.md");
    let cam = read_knowledge("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");

    // bgz-jc owns the canonical labels.
    assert!(bgz.contains("Gap **G-1**"), "bgz-jc must declare G-1");
    assert!(bgz.contains("Gap **G-2**"), "bgz-jc must declare G-2");

    // cam-pq must reference them with the source prefix.
    assert!(cam.contains("**bgz-jc G-1**"), "cam-pq must cross-ref as 'bgz-jc G-1'");
    assert!(cam.contains("**bgz-jc G-2**"), "cam-pq must cross-ref as 'bgz-jc G-2'");
}

#[test]
fn cross_doc_r15_signature_basis_impl_consistent() {
    // The SignatureBasis<DEPTH> impl must be mentioned in both the
    // canon-resolutions-delta and substrate-canon docs with invert unimplemented.
    for filename in &[
        "pr-x12-canon-resolutions-delta.md",
        "pr-x12-substrate-canon-resolutions.md",
    ] {
        let doc = read_knowledge(filename);
        assert!(
            doc.contains("SignatureBasis<DEPTH"),
            "{filename}: must mention 'SignatureBasis<DEPTH'"
        );
        assert!(
            doc.contains("unimplemented!"),
            "{filename}: invert() must be marked unimplemented!"
        );
    }
}

#[test]
fn all_changed_knowledge_files_exist_and_are_non_empty() {
    // Sanity check: all seven files changed in this PR must be readable.
    let files = [
        "pr-x12-bgz-jc-substrate-synergies.md",
        "pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md",
        "pr-x12-canon-resolutions-delta.md",
        "pr-x12-gguf-llm-weights-encoding.md",
        "pr-x12-substrate-canon-resolutions.md",
        "pr-x12-woa-multiarch-orchestration.md",
        "pr-x12-x266-3dgs-spacetime-upscaling.md",
    ];
    for filename in &files {
        let content = read_knowledge(filename);
        assert!(
            !content.is_empty(),
            "{filename}: file must be non-empty"
        );
        assert!(
            content.len() > 1000,
            "{filename}: file suspiciously short ({} bytes) — may be truncated",
            content.len()
        );
    }
}
