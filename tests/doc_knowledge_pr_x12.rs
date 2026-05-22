/// Integration tests for PR-X12 knowledge-base documentation changes.
///
/// These tests validate the internal consistency of the markdown documents
/// under `.claude/knowledge/` that were modified in this PR. They check:
///
/// - Gap labels (G-1, G-2) in bgz-jc document section headers
/// - Cross-document gap namespace disambiguation (bgz-jc G-1 vs cam-pq G-1,
///   removal of obsolete G-8 / G-9 labels in the cam-pq doc)
/// - Presence and content of new resolution sections R-14 and R-15
/// - Kernel-location corrections (bgz17::scalar_sparse::tropical_spmv)
/// - Terminology consistency (polyfill vs dispatch) in WoA doc
/// - Status-column wording (canon-fixed) in x266 doc
/// - KV-cache caveat and enum-discriminant-slot language in GGUF doc
///
/// None of these files contain executable code; the tests treat them as
/// structured text whose invariants can be asserted via string search.

use std::fs;
use std::path::PathBuf;

/// Returns the absolute path to a file inside `.claude/knowledge/`.
fn knowledge_path(filename: &str) -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
        .join(".claude")
        .join("knowledge")
        .join(filename)
}

/// Read a knowledge-base document; panics with a descriptive message if the
/// file cannot be opened.
fn read_doc(filename: &str) -> String {
    let path = knowledge_path(filename);
    fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Could not read {:?}: {}", path, e))
}

// ---------------------------------------------------------------------------
// pr-x12-bgz-jc-substrate-synergies.md — Gap label additions
// ---------------------------------------------------------------------------

#[test]
fn bgz_jc_doc_exists() {
    let path = knowledge_path("pr-x12-bgz-jc-substrate-synergies.md");
    assert!(path.exists(), "pr-x12-bgz-jc-substrate-synergies.md must exist");
}

#[test]
fn bgz_jc_section_5_1_has_gap_g1_label() {
    let content = read_doc("pr-x12-bgz-jc-substrate-synergies.md");
    // The PR added "(Gap **G-1**)" to the §5.1 heading
    assert!(
        content.contains("### 5.1 `jd-nd` — the missing ndarray-side proof crate (Gap **G-1**)"),
        "§5.1 heading must include '(Gap **G-1**)'"
    );
}

#[test]
fn bgz_jc_section_5_2_has_gap_g2_label() {
    let content = read_doc("pr-x12-bgz-jc-substrate-synergies.md");
    // The PR added "(Gap **G-2**)" to the §5.2 heading
    assert!(
        content.contains("### 5.2 Cronbach / ICC research crate (Gap **G-2**)"),
        "§5.2 heading must include '(Gap **G-2**)'"
    );
}

#[test]
fn bgz_jc_gap_labels_present_in_gap_section() {
    let content = read_doc("pr-x12-bgz-jc-substrate-synergies.md");
    // Both G-1 and G-2 must appear in the Gaps section (§5)
    let gaps_start = content
        .find("## 5. Gaps")
        .expect("§5 Gaps section must exist");
    let gaps_text = &content[gaps_start..];
    assert!(
        gaps_text.contains("G-1"),
        "Gaps section must reference G-1"
    );
    assert!(
        gaps_text.contains("G-2"),
        "Gaps section must reference G-2"
    );
}

#[test]
fn bgz_jc_does_not_use_old_unlabeled_headings() {
    let content = read_doc("pr-x12-bgz-jc-substrate-synergies.md");
    // The OLD heading did not have the gap label; ensure it's gone
    assert!(
        !content.contains("### 5.1 `jd-nd` — the missing ndarray-side proof crate\n"),
        "§5.1 heading must not be the un-labelled pre-PR version"
    );
    assert!(
        !content.contains("### 5.2 Cronbach / ICC research crate\n"),
        "§5.2 heading must not be the un-labelled pre-PR version"
    );
}

// ---------------------------------------------------------------------------
// pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md — Gap namespace
// ---------------------------------------------------------------------------

#[test]
fn cam_pq_doc_exists() {
    let path = knowledge_path("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    assert!(path.exists(), "cam-pq substrate-bindings doc must exist");
}

#[test]
fn cam_pq_cross_ref_table_uses_bgz_jc_prefixed_gap_ids() {
    let content = read_doc("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    // The PR renamed G-8 → "bgz-jc G-1" and G-9 → "bgz-jc G-2"
    assert!(
        content.contains("**bgz-jc G-1**"),
        "Cross-ref table must use 'bgz-jc G-1' (not the old G-8)"
    );
    assert!(
        content.contains("**bgz-jc G-2**"),
        "Cross-ref table must use 'bgz-jc G-2' (not the old G-9)"
    );
}

#[test]
fn cam_pq_no_longer_uses_g8_or_g9_labels() {
    let content = read_doc("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    // G-8 / G-9 were the old (incorrect) gap IDs that this PR retired
    assert!(
        !content.contains("**G-8**"),
        "cam-pq doc must not use obsolete G-8 label"
    );
    assert!(
        !content.contains("**G-9**"),
        "cam-pq doc must not use obsolete G-9 label"
    );
}

#[test]
fn cam_pq_cross_ref_table_header_updated() {
    let content = read_doc("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    // Table header changed from "Gap (prior)" to "Gap (cross-ref)"
    assert!(
        content.contains("| Gap (cross-ref) | Component | Cost |"),
        "Cross-ref table header must read 'Gap (cross-ref)'"
    );
    assert!(
        !content.contains("| Gap (prior) |"),
        "Obsolete 'Gap (prior)' table header must be removed"
    );
}

#[test]
fn cam_pq_namespace_clarification_note_present() {
    let content = read_doc("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    // The PR added a namespace disambiguation note
    assert!(
        content.contains("bgz-jc's G-1 / G-2 are a separate namespace owned by that doc"),
        "Namespace clarification note must be present"
    );
    assert!(
        content.contains(
            "avoid the collision the previous G-8 / G-9 labelling implied"
        ),
        "Clarification note must explain why the old G-8/G-9 labelling was wrong"
    );
}

#[test]
fn cam_pq_cross_ref_ownership_attribution() {
    let content = read_doc("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    // The PR added a sentence explaining where canonical IDs live
    assert!(
        content.contains(
            "their canonical IDs are owned by `pr-x12-bgz-jc-substrate-synergies.md` §5"
        ),
        "Ownership sentence must attribute G-1/G-2 to the bgz-jc doc"
    );
}

#[test]
fn cam_pq_cross_ref_table_references_correct_sections() {
    let content = read_doc("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    // Table rows should reference the source sections
    assert!(
        content.contains("(§5.1)"),
        "Cross-ref row for bgz-jc G-1 must cite §5.1"
    );
    assert!(
        content.contains("(§5.2)"),
        "Cross-ref row for bgz-jc G-2 must cite §5.2"
    );
}

// ---------------------------------------------------------------------------
// pr-x12-canon-resolutions-delta.md — New categories and sections
// ---------------------------------------------------------------------------

#[test]
fn canon_resolutions_delta_doc_exists() {
    let path = knowledge_path("pr-x12-canon-resolutions-delta.md");
    assert!(path.exists(), "canon-resolutions-delta doc must exist");
}

#[test]
fn canon_resolutions_delta_six_categories() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // The PR changed "Five categories" → "Six categories"
    assert!(
        content.contains("Six categories of novel content survive the delta filter"),
        "§0 must say 'Six categories' (was 'Five' before this PR)"
    );
}

#[test]
fn canon_resolutions_delta_r7_kernel_location() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // The PR added the actual kernel symbol to the R-7 entry
    assert!(
        content.contains("bgz17::scalar_sparse::tropical_spmv"),
        "R-7 must cite bgz17::scalar_sparse::tropical_spmv as the actual kernel"
    );
}

#[test]
fn canon_resolutions_delta_r13_primitives_listed() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // The PR expanded the R-13 entry with implementation primitives
    assert!(
        content.contains("`cam_pq` + `bgz-hhtl-d` + `dn_tree` + `merkle_tree`"),
        "R-13 entry must list the four implementation primitives"
    );
}

#[test]
fn canon_resolutions_delta_r14_r15_category_present() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // The PR added a 6th category covering R-14 and R-15
    assert!(
        content.contains("Formal-correctness + stream lane (post-merge)"),
        "Category 6 header must mention formal-correctness + stream lane"
    );
    assert!(
        content.contains("R-14") && content.contains("R-15"),
        "R-14 and R-15 must be referenced in the delta doc"
    );
}

#[test]
fn canon_resolutions_delta_section_11_is_r14() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // §11 is now "Formal-correctness layer (R-14)"
    assert!(
        content.contains("## 11. Formal-correctness layer (R-14)"),
        "§11 must be titled 'Formal-correctness layer (R-14)'"
    );
}

#[test]
fn canon_resolutions_delta_section_12_is_r15() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // §12 is now "Stream-signal codec lane (R-15)"
    assert!(
        content.contains("## 12. Stream-signal codec lane (R-15)"),
        "§12 must be titled 'Stream-signal codec lane (R-15)'"
    );
}

#[test]
fn canon_resolutions_delta_section_13_is_load_bearing_paragraph() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // §13 is the load-bearing paragraph (was §11 before this PR)
    assert!(
        content.contains("## 13. The single load-bearing paragraph"),
        "§13 must be the single load-bearing paragraph (renumbered from §11)"
    );
    // And the old §11 number must NOT appear with the old title
    assert!(
        !content.contains("## 11. The single load-bearing paragraph"),
        "§11 must no longer be titled 'The single load-bearing paragraph'"
    );
}

#[test]
fn canon_resolutions_delta_r14_pillar_table_present() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // §11 should contain a pillar table with Pillar 10 and Pillar 11
    assert!(
        content.contains("Pillar 10") && content.contains("jc::pflug"),
        "R-14 section must reference Pillar 10 and jc::pflug"
    );
    assert!(
        content.contains("Pillar 11") && content.contains("jc::hambly_lyons"),
        "R-14 section must reference Pillar 11 and jc::hambly_lyons"
    );
}

#[test]
fn canon_resolutions_delta_r15_signature_basis_code_block() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // §12 should contain the SignatureBasis<DEPTH> Rust impl snippet
    assert!(
        content.contains("impl<const DEPTH: usize> Basis<f32> for SignatureBasis<DEPTH>"),
        "R-15 section must include the SignatureBasis<DEPTH> impl snippet"
    );
}

#[test]
fn canon_resolutions_delta_r15_uses_signature_truncated_not_pde() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // The PR is explicit: use signature_truncated, NOT signature_kernel_pde
    let r15_section_start = content
        .find("## 12. Stream-signal codec lane (R-15)")
        .expect("R-15 section must exist");
    let r15_text = &content[r15_section_start..];
    assert!(
        r15_text.contains("signature_truncated"),
        "R-15 section must cite sigker::signature_truncated"
    );
    assert!(
        r15_text.contains("PR #350"),
        "R-15 section must explain why signature_kernel_pde is avoided (PR #350 bug)"
    );
}

#[test]
fn canon_resolutions_delta_load_bearing_paragraph_mentions_r14_r15() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // The updated §13 load-bearing paragraph references the substrate follow-up
    assert!(
        content.contains("R-14, R-15"),
        "Load-bearing paragraph must mention R-14 and R-15"
    );
    assert!(
        content.contains("SignatureBasis<DEPTH>"),
        "Load-bearing paragraph must reference SignatureBasis<DEPTH>"
    );
}

#[test]
fn canon_resolutions_delta_falsifiability_matrix_includes_r14_r15() {
    let content = read_doc("pr-x12-canon-resolutions-delta.md");
    // §9 matrix rows updated to include 24+3 rows (R-14/R-15)
    assert!(
        content.contains("24+3 rows including R-14/R-15"),
        "Falsifiability matrix description must note the R-14/R-15 additions"
    );
}

// ---------------------------------------------------------------------------
// pr-x12-substrate-canon-resolutions.md — R-14, R-15, updated summaries
// ---------------------------------------------------------------------------

#[test]
fn substrate_canon_resolutions_doc_exists() {
    let path = knowledge_path("pr-x12-substrate-canon-resolutions.md");
    assert!(path.exists(), "substrate-canon-resolutions doc must exist");
}

#[test]
fn substrate_canon_resolutions_toc_mentions_r14_r15() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    // §7 now lists five commitments including R-14 and R-15
    assert!(
        content.contains("R-14") && content.contains("R-15"),
        "substrate-canon-resolutions must reference R-14 and R-15"
    );
}

#[test]
fn substrate_canon_resolutions_fifteen_resolutions_mentioned() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    // The summary section was updated from "thirteen" to "fifteen"
    assert!(
        content.contains("fifteen resolutions"),
        "Summary must say 'fifteen resolutions' (R-1 through R-15)"
    );
}

#[test]
fn substrate_canon_resolutions_citation_ids_updated_to_r15() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    // Citation IDs updated from R-1..R-13 to R-1..R-15
    assert!(
        content.contains("R-1 through R-15"),
        "Citation IDs section must say 'R-1 through R-15'"
    );
}

#[test]
fn substrate_canon_resolutions_r14_section_exists() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    assert!(
        content.contains("### R-14 — Formal correctness via `lance-graph::jc` pillars"),
        "R-14 section header must be present"
    );
}

#[test]
fn substrate_canon_resolutions_r15_section_exists() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    assert!(
        content.contains(
            "### R-15 — `SignatureBasis<const DEPTH: usize>` as `Basis<f32>` impl"
        ),
        "R-15 section header must be present"
    );
}

#[test]
fn substrate_canon_resolutions_r14_pillar10_pflug() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    let r14_start = content
        .find("### R-14")
        .expect("R-14 section must exist");
    let r14_text = &content[r14_start..];
    assert!(
        r14_text.contains("jc::pflug"),
        "R-14 must name jc::pflug as Pillar 10 implementation"
    );
    assert!(
        r14_text.contains("Pflug-Pichler"),
        "R-14 must cite Pflug-Pichler theorem"
    );
    assert!(
        r14_text.contains("Lipschitz"),
        "R-14 must describe the Lipschitz bound from Pillar 10"
    );
}

#[test]
fn substrate_canon_resolutions_r14_pillar11_hambly_lyons() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    let r14_start = content
        .find("### R-14")
        .expect("R-14 section must exist");
    let r14_text = &content[r14_start..];
    assert!(
        r14_text.contains("jc::hambly_lyons"),
        "R-14 must name jc::hambly_lyons as Pillar 11 implementation"
    );
    assert!(
        r14_text.contains("Hambly-Lyons"),
        "R-14 must cite Hambly-Lyons theorem"
    );
    assert!(
        r14_text.contains("--features hambly-lyons"),
        "R-14 must note that Pillar 11 requires the hambly-lyons feature flag"
    );
}

#[test]
fn substrate_canon_resolutions_r14_probe_thresholds() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    let r14_start = content
        .find("### R-14")
        .expect("R-14 section must exist");
    let r14_text = &content[r14_start..];
    // The PR commits specific probe thresholds that must be preserved
    assert!(
        r14_text.contains("forward < 1e-9"),
        "R-14 probe threshold: forward < 1e-9"
    );
    assert!(
        r14_text.contains("converse > 0.05"),
        "R-14 probe threshold: converse > 0.05"
    );
    assert!(
        r14_text.contains("discrimination ratio") || r14_text.contains("ratio ≥ 1e6"),
        "R-14 probe threshold: discrimination ratio ≥ 1e6"
    );
}

#[test]
fn substrate_canon_resolutions_r15_signature_basis_impl_block() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    let r15_start = content
        .find("### R-15")
        .expect("R-15 section must exist");
    let r15_text = &content[r15_start..];
    assert!(
        r15_text.contains("impl<const DEPTH: usize> Basis<f32> for SignatureBasis<DEPTH>"),
        "R-15 must include the Basis<f32> impl block"
    );
}

#[test]
fn substrate_canon_resolutions_r15_invert_unimplemented() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    let r15_start = content
        .find("### R-15")
        .expect("R-15 section must exist");
    let r15_text = &content[r15_start..];
    // invert() must be documented as unimplemented (path inversion is N/A)
    assert!(
        r15_text.contains("unimplemented!"),
        "R-15 impl must show invert() as unimplemented"
    );
    assert!(
        r15_text.contains("tree-like equivalence"),
        "R-15 invert comment must cite tree-like equivalence reasoning"
    );
}

#[test]
fn substrate_canon_resolutions_r15_uses_signature_truncated() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    let r15_start = content
        .find("### R-15")
        .expect("R-15 section must exist");
    let r15_text = &content[r15_start..];
    assert!(
        r15_text.contains("signature_truncated"),
        "R-15 must use sigker::signature_truncated (not the buggy PDE form)"
    );
    assert!(
        r15_text.contains("PR #350"),
        "R-15 must reference PR #350 as the reason the PDE form is deferred"
    );
}

#[test]
fn substrate_canon_resolutions_r15_five_plan_g_lanes() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    let r15_start = content
        .find("### R-15")
        .expect("R-15 section must exist");
    let r15_text = &content[r15_start..];
    assert!(
        r15_text.contains("fifth") || r15_text.contains("5th"),
        "R-15 must describe itself as the fifth Plan G lane"
    );
    assert!(
        r15_text.contains("stream signal") || r15_text.contains("Stream signal"),
        "R-15 must name the new lane 'stream signal'"
    );
}

#[test]
fn substrate_canon_resolutions_tropical_spmv_kernel_location() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    // The PR added a clarification that the tropical-GEMM kernel lives in
    // bgz17::scalar_sparse::tropical_spmv, NOT in an abstract blasgraph namespace
    assert!(
        content.contains("bgz17::scalar_sparse::tropical_spmv"),
        "Must cite bgz17::scalar_sparse::tropical_spmv as actual kernel home"
    );
    assert!(
        content.contains("Cite the symbol"),
        "Must instruct readers to cite the symbol, not the namespace"
    );
}

#[test]
fn substrate_canon_resolutions_r13_primitives_table() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    // The PR added a primitives table to the R-13 section
    assert!(
        content.contains("cam_pq::CamCodebook"),
        "R-13 must include cam_pq::CamCodebook in primitives table"
    );
    assert!(
        content.contains("ndarray::hpc::dn_tree"),
        "R-13 must include dn_tree in primitives table"
    );
    assert!(
        content.contains("ndarray::hpc::merkle_tree"),
        "R-13 must include merkle_tree in primitives table"
    );
    assert!(
        content.contains("CodebookHandle"),
        "R-13 must reference the CodebookHandle trait"
    );
}

#[test]
fn substrate_canon_resolutions_falsifiability_matrix_has_r14_rows() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    // §9 falsifiability matrix should have rows for R-14 (two rows: Pillar 10 + Pillar 11)
    assert!(
        content.contains("R-14 (Pillar 10 active)"),
        "Falsifiability matrix must have a row for R-14 Pillar 10"
    );
    assert!(
        content.contains("R-14 (Pillar 11 active)"),
        "Falsifiability matrix must have a row for R-14 Pillar 11"
    );
}

#[test]
fn substrate_canon_resolutions_falsifiability_matrix_has_r15_row() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    assert!(
        content.contains("R-15 (SignatureBasis lane)"),
        "Falsifiability matrix must have a row for R-15"
    );
}

#[test]
fn substrate_canon_resolutions_summary_mentions_tropical_spmv() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    // The summary (context-window preservation section) was updated to cite the symbol
    assert!(
        content.contains("`bgz17::scalar_sparse::tropical_spmv`"),
        "Summary section must reference bgz17::scalar_sparse::tropical_spmv"
    );
}

#[test]
fn substrate_canon_resolutions_summary_lists_r13_primitives() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    // The summary was updated to list R-13 primitives
    assert!(
        content.contains("`cam_pq` + `bgz-hhtl-d` + `dn_tree` + `merkle_tree`"),
        "Summary section must list R-13 primitives"
    );
}

#[test]
fn substrate_canon_resolutions_summary_r14_r15_entries() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    // Summary bullets for R-14 and R-15 must be present
    assert!(
        content.contains("`jc::pflug` (Pillar 10)"),
        "Summary R-14 bullet must mention jc::pflug Pillar 10"
    );
    assert!(
        content.contains("`jc::hambly_lyons` (Pillar 11"),
        "Summary R-14 bullet must mention jc::hambly_lyons Pillar 11"
    );
    assert!(
        content.contains("`SignatureBasis<DEPTH>: Basis<f32>`"),
        "Summary R-15 bullet must reference SignatureBasis<DEPTH>: Basis<f32>"
    );
}

#[test]
fn substrate_canon_resolutions_intro_toc_updated_for_five_commitments() {
    let content = read_doc("pr-x12-substrate-canon-resolutions.md");
    // The intro originally said "three commitments"; PR updated it to "five"
    // and named R-14 and R-15 explicitly
    assert!(
        content.contains("five commitments missing from both originals"),
        "Intro ToC must say 'five commitments'"
    );
    assert!(
        content.contains("R-14 (formal correctness") || content.contains("R-14 (Formal"),
        "Intro ToC must name R-14"
    );
    assert!(
        content.contains("R-15") && content.contains("SignatureBasis"),
        "Intro ToC must name R-15 and SignatureBasis"
    );
}

// ---------------------------------------------------------------------------
// pr-x12-woa-multiarch-orchestration.md — "polyfill" terminology
// ---------------------------------------------------------------------------

#[test]
fn woa_doc_exists() {
    let path = knowledge_path("pr-x12-woa-multiarch-orchestration.md");
    assert!(path.exists(), "woa-multiarch-orchestration doc must exist");
}

#[test]
fn woa_premise_uses_polyfill_terminology() {
    let content = read_doc("pr-x12-woa-multiarch-orchestration.md");
    // The PR changed "dispatch" → "polyfill" in the premise line
    assert!(
        content.contains("per-arch polyfill contract"),
        "Premise must use 'per-arch polyfill contract'"
    );
    assert!(
        content.contains("per-arch polyfill decisions"),
        "Status line must say 'per-arch polyfill decisions'"
    );
}

#[test]
fn woa_thesis_uses_polyfill_terminology() {
    let content = read_doc("pr-x12-woa-multiarch-orchestration.md");
    // The PR rewrote the §0 thesis to use polyfill language
    let thesis_pos = content.find("## 0. Thesis").expect("§0 must exist");
    let thesis_text = &content[thesis_pos..thesis_pos + 600];
    assert!(
        thesis_text.contains("polyfill"),
        "§0 Thesis must use 'polyfill' terminology"
    );
}

#[test]
fn woa_section_3_title_uses_polyfill() {
    let content = read_doc("pr-x12-woa-multiarch-orchestration.md");
    // §3 was renamed from "Per-arch dispatch" to "Per-arch substrate via compile-time polyfill"
    assert!(
        content.contains("## 3. Per-arch substrate via compile-time polyfill"),
        "§3 must be titled 'Per-arch substrate via compile-time polyfill'"
    );
    assert!(
        !content.contains("## 3. Per-arch dispatch as a substrate property"),
        "Old §3 title 'Per-arch dispatch as a substrate property' must be gone"
    );
}

#[test]
fn woa_section_3_describes_cfg_polyfill_pattern() {
    let content = read_doc("pr-x12-woa-multiarch-orchestration.md");
    // §3.1 should explain the cfg re-export pattern, not runtime detection
    assert!(
        content.contains("cfg(target_feature") && content.contains("pub use crate::simd"),
        "§3 must show the cfg-based polyfill re-export pattern"
    );
}

#[test]
fn woa_no_owncaps_hwcaps_runtime_branching() {
    let content = read_doc("pr-x12-woa-multiarch-orchestration.md");
    // The PR removed HwCaps / runtime detection; those should no longer appear
    // as the primary model (they may appear in a historical note, but the main
    // design must not promote them)
    // Check the §3 section specifically
    let section3_start = content
        .find("## 3. Per-arch substrate via compile-time polyfill")
        .expect("§3 must exist");
    let section3_text = &content[section3_start..];
    let section4_start = section3_text.find("## 4.").unwrap_or(section3_text.len());
    let section3_body = &section3_text[..section4_start];
    // There must be NO runtime detection, no OnceLock<HwCaps>
    assert!(
        !section3_body.contains("OnceLock<HwCaps>"),
        "§3 must not describe runtime HwCaps detection (that model was replaced)"
    );
    assert!(
        !section3_body.contains("pub static CAP"),
        "§3 must not describe a static HwCaps singleton"
    );
}

#[test]
fn woa_commitment_5_uses_polyfill_transparency() {
    let content = read_doc("pr-x12-woa-multiarch-orchestration.md");
    // Commitment #2 was updated to "polyfill transparency"
    assert!(
        content.contains("Per-arch polyfill transparency"),
        "Commitment must say 'Per-arch polyfill transparency'"
    );
    assert!(
        !content.contains("Per-arch dispatch transparency"),
        "'Per-arch dispatch transparency' must be replaced by polyfill terminology"
    );
}

#[test]
fn woa_stack_diagram_labels_polyfill_substrate() {
    let content = read_doc("pr-x12-woa-multiarch-orchestration.md");
    // Stack diagram updated to label the layer as "polyfill substrate"
    assert!(
        content.contains("ndarray::hpc + ndarray::simd (polyfill substrate)"),
        "Stack diagram must label the substrate as 'polyfill substrate'"
    );
}

#[test]
fn woa_gpu_hook_uses_backend_target_not_dispatch_target() {
    let content = read_doc("pr-x12-woa-multiarch-orchestration.md");
    // The PR renamed dispatch_target → backend_target in the GPU offload anchor
    assert!(
        content.contains("backend_target"),
        "GPU offload hook must use 'backend_target()' not 'dispatch_target()'"
    );
    assert!(
        !content.contains("dispatch_target()"),
        "Old 'dispatch_target()' must be removed"
    );
}

// ---------------------------------------------------------------------------
// pr-x12-x266-3dgs-spacetime-upscaling.md — "canon-fixed" status labels
// ---------------------------------------------------------------------------

#[test]
fn x266_doc_exists() {
    let path = knowledge_path("pr-x12-x266-3dgs-spacetime-upscaling.md");
    assert!(path.exists(), "x266-3dgs-spacetime-upscaling doc must exist");
}

#[test]
fn x266_requirements_table_has_canon_fixed_labels() {
    let content = read_doc("pr-x12-x266-3dgs-spacetime-upscaling.md");
    // The PR added "canon-fixed" status to several rows in the prerequisites table
    assert!(
        content.contains("**canon-fixed**"),
        "Prerequisites table must use '**canon-fixed**' status label"
    );
}

#[test]
fn x266_requirements_table_basis_trait_canon_fixed() {
    let content = read_doc("pr-x12-x266-3dgs-spacetime-upscaling.md");
    // Basis<T> row was updated: "landed in concept" → "**canon-fixed** (R-1 trait shape committed)"
    assert!(
        content.contains("**canon-fixed** (R-1 trait shape committed)"),
        "Basis<T> row must say '**canon-fixed** (R-1 trait shape committed)'"
    );
}

#[test]
fn x266_requirements_table_header_byte_canon_fixed() {
    let content = read_doc("pr-x12-x266-3dgs-spacetime-upscaling.md");
    // Header byte row was updated: "landed" → "**canon-fixed** (R-2 commits bits 0-1)"
    assert!(
        content.contains("**canon-fixed** (R-2 commits bits 0-1"),
        "Header byte row must say '**canon-fixed** (R-2 commits bits 0-1 = `header_kind`)'"
    );
}

#[test]
fn x266_requirements_table_r13_canon_fixed() {
    let content = read_doc("pr-x12-x266-3dgs-spacetime-upscaling.md");
    // R-13 codebook row updated: "landed" → "**canon-fixed** (R-13 commits Option A)"
    assert!(
        content.contains("**canon-fixed** (R-13 commits Option A"),
        "R-13 row must say '**canon-fixed** (R-13 commits Option A: per-shard codebook for Plan F v1)'"
    );
}

#[test]
fn x266_requirements_table_r3_audit_rule_clarification() {
    let content = read_doc("pr-x12-x266-3dgs-spacetime-upscaling.md");
    // R-3 row updated to note CI check is pending
    assert!(
        content.contains("CI check pending"),
        "R-3 row must note that CI check is pending"
    );
}

#[test]
fn x266_requirements_table_r2_wire_format_plan_a8() {
    let content = read_doc("pr-x12-x266-3dgs-spacetime-upscaling.md");
    // R-2 row updated to note wire-format implementation is in Plan A8
    assert!(
        content.contains("wire-format implementation in Plan A8"),
        "R-2 row must note wire-format implementation in Plan A8"
    );
}

#[test]
fn x266_canon_fixed_definition_note_present() {
    let content = read_doc("pr-x12-x266-3dgs-spacetime-upscaling.md");
    // The PR added a note explaining the "canon-fixed" / "scheduled" terminology
    assert!(
        content.contains("\"Canon-fixed\"") || content.contains("**\"Canon-fixed\"**"),
        "A note must define what 'canon-fixed' means"
    );
    assert!(
        content.contains("the resolution doc commits the design"),
        "Canon-fixed definition must say 'the resolution doc commits the design'"
    );
    assert!(
        content.contains("None of the above have shipping code today"),
        "Note must clarify that none of the items have shipping code yet"
    );
}

// ---------------------------------------------------------------------------
// pr-x12-gguf-llm-weights-encoding.md — Escape/lossless, KV caveat, slot
// ---------------------------------------------------------------------------

#[test]
fn gguf_doc_exists() {
    let path = knowledge_path("pr-x12-gguf-llm-weights-encoding.md");
    assert!(path.exists(), "gguf-llm-weights-encoding doc must exist");
}

#[test]
fn gguf_escape_lossless_references_f4_falsifier() {
    let content = read_doc("pr-x12-gguf-llm-weights-encoding.md");
    // The PR added a reference to §10 falsifier F-4 for the wire-format escape mechanism
    assert!(
        content.contains("§10 falsifier **F-4**"),
        "Escape lossless paragraph must reference §10 falsifier F-4"
    );
    assert!(
        content.contains("rANS bypass channel"),
        "Escape description must mention the rANS bypass channel"
    );
    assert!(
        content.contains("HEVC-escape-coefficient precedent"),
        "Escape description must cite the HEVC escape-coefficient precedent"
    );
}

#[test]
fn gguf_memory_savings_kv_cache_caveat() {
    let content = read_doc("pr-x12-gguf-llm-weights-encoding.md");
    // The PR added a "weights only" qualifier and KV cache caveat
    assert!(
        content.contains("Memory savings (weights only)"),
        "Memory savings note must be qualified as 'weights only'"
    );
    assert!(
        content.contains("KV cache scales with context length"),
        "Memory savings section must include the KV cache scaling caveat"
    );
    assert!(
        content.contains("Plan D") && content.contains("M:H-3") && content.contains("R-4"),
        "KV cache caveat must cross-reference Plan D / M:H-3 / R-4"
    );
}

#[test]
fn gguf_memory_savings_no_longer_claims_phone_class_standalone() {
    let content = read_doc("pr-x12-gguf-llm-weights-encoding.md");
    // The old phrasing implied weight compression alone makes a 7B model viable on phone
    // The PR corrected this with a caveat; the old standalone claim should be gone
    assert!(
        !content.contains("A 7B model at PR-X12 is genuinely runnable on a phone-class device"),
        "Old standalone phone-class viability claim must be removed"
    );
    // Instead the doc should say "easier" with a caveat
    assert!(
        content.contains("easier") || content.contains("Both lanes are needed"),
        "Corrected text must qualify that weight compression alone is not sufficient"
    );
}

#[test]
fn gguf_encoding_domain_enum_discriminant_slot() {
    let content = read_doc("pr-x12-gguf-llm-weights-encoding.md");
    // The PR rewrote implication #5 to emphasise reserving the slot now
    // even though the implementation ships post-PR-X12
    assert!(
        content.contains("enum-discriminant slot"),
        "Implication #5 must use 'enum-discriminant slot' language"
    );
    assert!(
        content.contains("forward-compatibility-locked"),
        "Implication #5 must say the slot is forward-compatibility-locked"
    );
    assert!(
        content.contains("without a wire-format break"),
        "Implication #5 must note that reserving now prevents a wire-format break"
    );
}

#[test]
fn gguf_encoding_domain_discriminant_stays_unimplemented_in_pr_x12() {
    let content = read_doc("pr-x12-gguf-llm-weights-encoding.md");
    // The PR clarifies the LLMWeights value is reserved but stays unimplemented in PR-X12
    assert!(
        content.contains("stays unimplemented in PR-X12"),
        "Implication #5 must state LLMWeights stays unimplemented in PR-X12"
    );
}

// ---------------------------------------------------------------------------
// Cross-document consistency — shared identifiers and references
// ---------------------------------------------------------------------------

#[test]
fn cross_doc_gap_namespace_prefix_rule_stated_in_cam_pq() {
    let content = read_doc("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");
    // The PR's core disambiguation rule must be explicitly stated
    assert!(
        content.contains(r#""bgz-jc G-1" vs "cam-pq G-1""#),
        "Namespace rule must give a worked example comparing bgz-jc G-1 vs cam-pq G-1"
    );
}

#[test]
fn cross_doc_tropical_spmv_cited_consistently() {
    // Both the delta doc and the substrate-canon-resolutions doc must cite the
    // same kernel symbol
    let delta = read_doc("pr-x12-canon-resolutions-delta.md");
    let substrate = read_doc("pr-x12-substrate-canon-resolutions.md");
    let symbol = "bgz17::scalar_sparse::tropical_spmv";
    assert!(
        delta.contains(symbol),
        "canon-resolutions-delta must cite {}", symbol
    );
    assert!(
        substrate.contains(symbol),
        "substrate-canon-resolutions must cite {}", symbol
    );
}

#[test]
fn cross_doc_r14_probe_thresholds_consistent() {
    // Both the delta doc and the substrate doc must agree on the exact Pillar 11 probe thresholds
    let delta = read_doc("pr-x12-canon-resolutions-delta.md");
    let substrate = read_doc("pr-x12-substrate-canon-resolutions.md");

    for doc_name in &["delta", "substrate"] {
        let content = if *doc_name == "delta" { &delta } else { &substrate };
        assert!(
            content.contains("forward < 1e-9"),
            "{} doc must state Pillar 11 forward threshold < 1e-9", doc_name
        );
        assert!(
            content.contains("converse > 0.05"),
            "{} doc must state Pillar 11 converse threshold > 0.05", doc_name
        );
    }
}

#[test]
fn cross_doc_pr_350_referenced_for_pde_bug() {
    // All docs that mention the signature_kernel_pde bug should cite PR #350
    let delta = read_doc("pr-x12-canon-resolutions-delta.md");
    let substrate = read_doc("pr-x12-substrate-canon-resolutions.md");
    let cam_pq = read_doc("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md");

    for (name, doc) in &[("delta", &delta), ("substrate", &substrate), ("cam-pq", &cam_pq)] {
        if doc.contains("signature_kernel_pde") {
            assert!(
                doc.contains("PR #350"),
                "{} doc mentions signature_kernel_pde but does not cite PR #350", name
            );
        }
    }
}

#[test]
fn cross_doc_signature_truncated_preferred_over_pde() {
    // All R-15 / SignatureBasis discussions must prefer signature_truncated
    let delta = read_doc("pr-x12-canon-resolutions-delta.md");
    let substrate = read_doc("pr-x12-substrate-canon-resolutions.md");

    for (name, doc) in &[("delta", &delta), ("substrate", &substrate)] {
        let has_r15 = doc.contains("SignatureBasis");
        if has_r15 {
            assert!(
                doc.contains("signature_truncated"),
                "{} doc with SignatureBasis must also reference signature_truncated", name
            );
        }
    }
}

#[test]
fn cross_doc_r13_four_primitives_consistent() {
    // Both the delta doc and the substrate doc must list the same four R-13 primitives
    let expected = ["cam_pq", "bgz-hhtl-d", "dn_tree", "merkle_tree"];
    let delta = read_doc("pr-x12-canon-resolutions-delta.md");
    let substrate = read_doc("pr-x12-substrate-canon-resolutions.md");

    for primitive in &expected {
        assert!(
            delta.contains(primitive),
            "canon-resolutions-delta R-13 section must mention {}", primitive
        );
        assert!(
            substrate.contains(primitive),
            "substrate-canon-resolutions R-13 section must mention {}", primitive
        );
    }
}
