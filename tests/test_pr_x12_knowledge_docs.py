"""
Tests for PR-X12 knowledge-base documentation changes.

Covers content correctness, internal consistency, and cross-reference
integrity for the seven .claude/knowledge/ files modified in this PR:

  - pr-x12-bgz-jc-substrate-synergies.md
  - pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md
  - pr-x12-canon-resolutions-delta.md
  - pr-x12-gguf-llm-weights-encoding.md
  - pr-x12-substrate-canon-resolutions.md
  - pr-x12-woa-multiarch-orchestration.md
  - pr-x12-x266-3dgs-spacetime-upscaling.md
"""

import re
import sys
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KNOWLEDGE_DIR = Path(__file__).parent.parent / ".claude" / "knowledge"


def read(filename: str) -> str:
    return (KNOWLEDGE_DIR / filename).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# pr-x12-bgz-jc-substrate-synergies.md
# ---------------------------------------------------------------------------


class TestBgzJcSubstrateSynergies(unittest.TestCase):
    """Gap labels (G-1, G-2) added to §5 section headings."""

    @classmethod
    def setUpClass(cls):
        cls.text = read("pr-x12-bgz-jc-substrate-synergies.md")

    # ------------------------------------------------------------------
    # Gap heading labels
    # ------------------------------------------------------------------

    def test_section_51_heading_has_gap_g1_label(self):
        """§5.1 heading must include the explicit gap label 'G-1'."""
        self.assertIn(
            "### 5.1 `jd-nd` — the missing ndarray-side proof crate (Gap **G-1**)",
            self.text,
        )

    def test_section_52_heading_has_gap_g2_label(self):
        """§5.2 heading must include the explicit gap label 'G-2'."""
        self.assertIn(
            "### 5.2 Cronbach / ICC research crate (Gap **G-2**)",
            self.text,
        )

    def test_g1_label_uses_bold_formatting(self):
        """Gap labels must be bold (**G-1**) so they render as strong text."""
        self.assertIn("**G-1**", self.text)
        self.assertIn("**G-2**", self.text)

    def test_gaps_section_exists_under_section_5(self):
        """The Gaps section (#5) must exist."""
        self.assertIn("## 5. Gaps — what doesn't exist yet", self.text)

    def test_jd_nd_described_as_missing_crate(self):
        """jd-nd must be identified as missing from /home/user/ndarray/."""
        self.assertIn("`jd-nd` does not exist in", self.text)

    def test_cronbach_icc_scoped_as_research_gap(self):
        """The Cronbach/ICC gap must reference the correct existing crate scope (FFT)."""
        self.assertIn("FFT (rustfft) variants", self.text)
        self.assertIn("Cronbach", self.text)

    # ------------------------------------------------------------------
    # Boundary / regression
    # ------------------------------------------------------------------

    def test_no_old_g8_g9_labels_in_bgz_doc(self):
        """This doc must not use G-8 or G-9 labels (those were renumbered)."""
        self.assertNotIn("**G-8**", self.text)
        self.assertNotIn("**G-9**", self.text)


# ---------------------------------------------------------------------------
# pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md
# ---------------------------------------------------------------------------


class TestCamPqSigkerDnTreeSubstrateBindings(unittest.TestCase):
    """Cross-reference namespace clarification: G-8/G-9 → bgz-jc G-1/bgz-jc G-2."""

    @classmethod
    def setUpClass(cls):
        cls.text = read("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md")

    # ------------------------------------------------------------------
    # Cross-ref table structure
    # ------------------------------------------------------------------

    def test_cross_ref_table_header_label(self):
        """Table must use 'Gap (cross-ref)' not 'Gap (prior)'."""
        self.assertIn("Gap (cross-ref)", self.text)
        self.assertNotIn("Gap (prior)", self.text)

    def test_bgz_jc_g1_cross_reference_present(self):
        """Cross-ref table must use 'bgz-jc G-1' (not G-8) for jd-nd gap."""
        self.assertIn("bgz-jc G-1", self.text)

    def test_bgz_jc_g2_cross_reference_present(self):
        """Cross-ref table must use 'bgz-jc G-2' (not G-9) for Cronbach/ICC gap."""
        self.assertIn("bgz-jc G-2", self.text)

    def test_no_standalone_g8_label(self):
        """Old G-8 label must not appear as a standalone gap identifier."""
        self.assertNotIn("**G-8**", self.text)

    def test_no_standalone_g9_label(self):
        """Old G-9 label must not appear as a standalone gap identifier."""
        self.assertNotIn("**G-9**", self.text)

    # ------------------------------------------------------------------
    # Namespace clarification paragraph
    # ------------------------------------------------------------------

    def test_namespace_clarification_paragraph_present(self):
        """Namespace clarification paragraph must be present."""
        self.assertIn(
            "G-1..G-7 IDs in §5 of *this* doc are local to the cam-pq",
            self.text,
        )

    def test_namespace_clarification_states_separate_ownership(self):
        """Must state that bgz-jc's gaps are owned by a different doc."""
        self.assertIn(
            "bgz-jc's G-1 / G-2 are a separate namespace owned by that doc",
            self.text,
        )

    def test_namespace_clarification_gives_prefixing_rule(self):
        """Prefixing rule must be explicit: 'bgz-jc G-1' vs 'cam-pq G-1'."""
        self.assertIn('"bgz-jc G-1"', self.text)
        self.assertIn('"cam-pq G-1"', self.text)

    def test_local_gaps_g1_through_g7_present(self):
        """This doc must define its own local G-1 through G-7 in §5."""
        for n in range(1, 8):
            self.assertIn(f"**G-{n}**", self.text, msg=f"G-{n} not found")

    def test_cross_ref_ownership_attributed_to_bgz_doc(self):
        """Must credit bgz-jc-substrate-synergies.md as the owning doc for those gaps."""
        self.assertIn("pr-x12-bgz-jc-substrate-synergies.md", self.text)
        self.assertIn("§5", self.text)

    # ------------------------------------------------------------------
    # Regression: old collision-prone labelling gone
    # ------------------------------------------------------------------

    def test_collision_warning_mentions_previous_g8_g9(self):
        """The clarification must reference the previous G-8/G-9 labelling as the problem."""
        self.assertIn("G-8 / G-9 labelling", self.text)


# ---------------------------------------------------------------------------
# pr-x12-canon-resolutions-delta.md
# ---------------------------------------------------------------------------


class TestCanonResolutionsDelta(unittest.TestCase):
    """Six-category structure, R-14/R-15 additions, falsifiability matrix."""

    @classmethod
    def setUpClass(cls):
        cls.text = read("pr-x12-canon-resolutions-delta.md")

    # ------------------------------------------------------------------
    # Category count
    # ------------------------------------------------------------------

    def test_six_categories_stated(self):
        """Header must say 'Six categories' (not 'Five categories')."""
        self.assertIn("Six categories of novel content survive", self.text)
        self.assertNotIn("Five categories of novel content survive", self.text)

    def test_category_6_present(self):
        """Category 6 must be present: Formal-correctness + stream lane."""
        self.assertIn(
            "**Formal-correctness + stream lane (post-merge)**", self.text
        )

    def test_category_6_cites_r14_and_r15(self):
        """Category 6 must mention R-14 and R-15."""
        self.assertIn("R-14", self.text)
        self.assertIn("R-15", self.text)

    def test_category_6_names_jc_pflug_and_hambly_lyons(self):
        """Category 6 must name both jc pillars."""
        self.assertIn("`jc::pflug`", self.text)
        self.assertIn("`jc::hambly_lyons`", self.text)

    # ------------------------------------------------------------------
    # R-7 tropical-GEMM kernel location
    # ------------------------------------------------------------------

    def test_r7_cites_actual_kernel_symbol(self):
        """R-7 entry must cite bgz17::scalar_sparse::tropical_spmv."""
        self.assertIn("bgz17::scalar_sparse::tropical_spmv", self.text)

    def test_r7_math_identity_present(self):
        """R-7 complexity claim O(4^d) → O(d²) must be present."""
        self.assertIn("O(4^d)", self.text)
        self.assertIn("O(d²)", self.text)

    # ------------------------------------------------------------------
    # R-13 primitives
    # ------------------------------------------------------------------

    def test_r13_names_cam_pq_primitive(self):
        """R-13 primitives must include cam_pq."""
        self.assertIn("`cam_pq`", self.text)

    def test_r13_names_dn_tree_primitive(self):
        """R-13 primitives must include dn_tree."""
        self.assertIn("`dn_tree`", self.text)

    def test_r13_names_merkle_tree_primitive(self):
        """R-13 primitives must include merkle_tree."""
        self.assertIn("`merkle_tree`", self.text)

    def test_r13_names_bgz_hhtl_d_primitive(self):
        """R-13 primitives must include bgz-hhtl-d."""
        self.assertIn("`bgz-hhtl-d`", self.text)

    # ------------------------------------------------------------------
    # Falsifiability matrix row count
    # ------------------------------------------------------------------

    def test_falsifiability_matrix_row_count_includes_r14_r15(self):
        """Falsifiability matrix must claim 24+3 rows including R-14/R-15."""
        self.assertIn("24+3 rows including R-14/R-15", self.text)

    # ------------------------------------------------------------------
    # R-14 section content
    # ------------------------------------------------------------------

    def test_section_11_is_r14_formal_correctness(self):
        """Section 11 must be the R-14 formal-correctness layer."""
        self.assertIn("## 11. Formal-correctness layer (R-14)", self.text)

    def test_r14_section_cites_pillar_10(self):
        """R-14 section must mention Pillar 10 (Pflug-Pichler)."""
        self.assertIn("Pillar 10", self.text)
        self.assertIn("Pflug-Pichler", self.text)

    def test_r14_section_cites_pillar_11(self):
        """R-14 section must mention Pillar 11 (Hambly-Lyons)."""
        self.assertIn("Pillar 11", self.text)
        self.assertIn("Hambly-Lyons", self.text)

    def test_r14_pillar_11_is_feature_gated(self):
        """Pillar 11 must be noted as feature-gated under hambly-lyons."""
        self.assertIn("--features hambly-lyons", self.text)

    def test_r14_pillar_11_probe_thresholds(self):
        """R-14 section must document the Pillar 11 probe pass criteria."""
        self.assertIn("forward<1e-9", self.text)
        self.assertIn("converse>0.05", self.text)
        self.assertIn("ratio≥1e6", self.text)

    def test_r14_g4_open_work_reference(self):
        """R-14 must acknowledge G-4 (PR #350) as open work."""
        self.assertIn("G-4", self.text)
        self.assertIn("PR #350", self.text)

    # ------------------------------------------------------------------
    # R-15 section content
    # ------------------------------------------------------------------

    def test_section_12_is_r15_stream_signal(self):
        """Section 12 must be the R-15 stream-signal codec lane."""
        self.assertIn("## 12. Stream-signal codec lane (R-15)", self.text)

    def test_r15_signature_basis_rust_snippet_present(self):
        """R-15 must show a Rust code block with SignatureBasis."""
        self.assertIn("SignatureBasis<DEPTH>", self.text)
        self.assertIn("impl<const DEPTH: usize> Basis<f32>", self.text)

    def test_r15_invert_is_unimplemented(self):
        """R-15 invert() must be documented as unimplemented due to tree-quotient."""
        self.assertIn("unimplemented!", self.text)
        self.assertIn("tree-like", self.text)

    def test_r15_uses_signature_truncated_not_pde(self):
        """R-15 must use signature_truncated (not signature_kernel_pde)."""
        self.assertIn("sigker::signature_truncated", self.text)

    def test_r15_explains_why_not_pde(self):
        """R-15 must explain that the PDE form has a known divergence bug."""
        self.assertIn("known divergence bug", self.text)
        self.assertIn("PR #350", self.text)

    def test_r15_stream_signal_fifth_lane(self):
        """R-15 must describe the fifth Plan G lane."""
        self.assertIn("fifth lane", self.text)
        self.assertIn("stream signal", self.text)

    def test_r15_compression_target_mentioned(self):
        """R-15 must include an approximate compression target (~10×)."""
        self.assertIn("~10×", self.text)

    # ------------------------------------------------------------------
    # Section 13 (renamed from §11)
    # ------------------------------------------------------------------

    def test_section_13_is_load_bearing_paragraph(self):
        """Section 13 must contain 'The single load-bearing paragraph'."""
        self.assertIn("## 13. The single load-bearing paragraph", self.text)

    def test_section_13_mentions_r14_r15(self):
        """The load-bearing paragraph must reference R-14 and R-15."""
        # Find the paragraph in §13
        idx = self.text.find("## 13.")
        self.assertNotEqual(idx, -1)
        tail = self.text[idx:]
        self.assertIn("R-14", tail)
        self.assertIn("R-15", tail)

    def test_section_13_mentions_signature_basis(self):
        """The load-bearing paragraph must mention SignatureBasis<DEPTH>."""
        idx = self.text.find("## 13.")
        self.assertNotEqual(idx, -1)
        tail = self.text[idx:]
        self.assertIn("SignatureBasis<DEPTH>", tail)

    # ------------------------------------------------------------------
    # Regression: old section 11 title gone
    # ------------------------------------------------------------------

    def test_old_section_11_title_gone(self):
        """Old '## 11. The single load-bearing paragraph' heading must not exist."""
        self.assertNotIn(
            "## 11. The single load-bearing paragraph", self.text
        )


# ---------------------------------------------------------------------------
# pr-x12-gguf-llm-weights-encoding.md
# ---------------------------------------------------------------------------


class TestGgufLlmWeightsEncoding(unittest.TestCase):
    """Escape section update, phone-class memory caveat, EncodingDomain slot."""

    @classmethod
    def setUpClass(cls):
        cls.text = read("pr-x12-gguf-llm-weights-encoding.md")

    # ------------------------------------------------------------------
    # Escape → F-4 cross-reference
    # ------------------------------------------------------------------

    def test_escape_section_references_f4_falsifier(self):
        """Escape section must reference §10 falsifier F-4."""
        self.assertIn("falsifier **F-4**", self.text)

    def test_escape_section_references_rans_bypass_channel(self):
        """Escape section must cite the rANS bypass channel in A8 framing."""
        self.assertIn("rANS bypass channel", self.text)

    def test_escape_section_cites_hevc_precedent(self):
        """Escape section must cite the HEVC-escape-coefficient precedent."""
        self.assertIn("HEVC-escape-coefficient precedent", self.text)

    # ------------------------------------------------------------------
    # Phone-class memory caveat
    # ------------------------------------------------------------------

    def test_phone_class_caveat_headline_weights_only(self):
        """The weights-only caveat must be explicit in the heading."""
        self.assertIn("Memory savings (weights only)", self.text)

    def test_phone_class_caveat_kv_cache_mentioned(self):
        """The KV cache must be identified as a separate memory lever."""
        self.assertIn("KV cache", self.text)

    def test_phone_class_caveat_kv_scales_with_context(self):
        """The KV cache scaling behaviour must be described (context length)."""
        self.assertIn("scales with context length", self.text)

    def test_phone_class_caveat_both_lanes_needed(self):
        """Must state that both weight compression and KV compression are needed."""
        self.assertIn("Both lanes are needed", self.text)

    def test_phone_class_caveat_plan_d_mentioned(self):
        """KV cache lane must be attributed to Plan D / M:H-3 / R-4."""
        self.assertIn("Plan D", self.text)
        self.assertIn("M:H-3", self.text)

    def test_old_phone_class_unqualified_claim_gone(self):
        """The old unqualified 'genuinely runnable' claim must be gone; new text qualifies it."""
        # Old text made an unqualified claim that a 7B is "genuinely runnable on a phone-class
        # device"; the new text replaces this with a qualified caveat.
        self.assertNotIn(
            "A 7B model at PR-X12 is genuinely runnable on a phone-class device, "
            "where GGUF Q4 is borderline.",
            self.text,
        )
        # The new qualified replacement must be present
        self.assertIn(
            'PR-X12 weight compression alone takes a 7B from "borderline" to "easier"',
            self.text,
        )

    # ------------------------------------------------------------------
    # EncodingDomain::LLMWeights forward-compatibility
    # ------------------------------------------------------------------

    def test_encoding_domain_described_as_slot_reservation(self):
        """EncodingDomain::LLMWeights must be described as an enum-discriminant slot."""
        self.assertIn("enum-discriminant slot", self.text)

    def test_encoding_domain_forward_compatibility_locked(self):
        """The EncodingDomain slot must be described as forward-compatibility-locked."""
        self.assertIn("forward-compatibility-locked", self.text)

    def test_encoding_domain_unimplemented_in_pr_x12(self):
        """Must state that the LLMWeights value is unimplemented in PR-X12."""
        self.assertIn("unimplemented in PR-X12", self.text)

    def test_encoding_domain_no_wire_format_break(self):
        """Must state that the slot prevents a future wire-format break."""
        self.assertIn("without a wire-format break", self.text)

    # ------------------------------------------------------------------
    # Regression: old unqualified claim
    # ------------------------------------------------------------------

    def test_old_unqualified_memory_sentence_removed(self):
        """The old unqualified memory savings sentence must no longer appear verbatim."""
        # Old text was a single sentence claiming phone-class without KV caveat
        self.assertNotIn(
            "A 7B model at PR-X12 is genuinely runnable on a phone-class device, where GGUF Q4 is borderline.",
            self.text,
        )


# ---------------------------------------------------------------------------
# pr-x12-substrate-canon-resolutions.md
# ---------------------------------------------------------------------------


class TestSubstrateCanonResolutions(unittest.TestCase):
    """R-14/R-15 additions, updated citation range, implementation primitives."""

    @classmethod
    def setUpClass(cls):
        cls.text = read("pr-x12-substrate-canon-resolutions.md")

    # ------------------------------------------------------------------
    # Section 7 commitment count
    # ------------------------------------------------------------------

    def test_section_7_says_five_commitments(self):
        """§7 must say 'five commitments' (not three)."""
        self.assertIn("five commitments missing from both originals", self.text)
        self.assertNotIn(
            "three commitments missing from both originals", self.text
        )

    def test_section_7_lists_r14_and_r15(self):
        """§7 introduction must mention R-14 and R-15."""
        idx = self.text.find("## 7.")
        if idx == -1:
            idx = self.text.find("### R-14")
        self.assertIn("R-14", self.text[: self.text.find("### R-14") + 5])
        self.assertIn("R-15", self.text)

    # ------------------------------------------------------------------
    # Citation range
    # ------------------------------------------------------------------

    def test_citation_ids_go_to_r15(self):
        """Citation IDs must be stated as R-1 through R-15."""
        self.assertIn("R-1` through `R-15", self.text)

    def test_citation_ids_append_only(self):
        """Must state that numbering is append-only."""
        self.assertIn("append-only", self.text)

    def test_citation_ids_r14_r15_sourced_from_substrate_binding_doc(self):
        """Must attribute R-14/R-15 to the substrate-binding doc."""
        self.assertIn("appended post-merge from the substrate-binding doc", self.text)

    # ------------------------------------------------------------------
    # Tropical-GEMM actual kernel home (R-7)
    # ------------------------------------------------------------------

    def test_r7_actual_kernel_home_note_present(self):
        """Must include the 'Actual kernel home' note for tropical-GEMM."""
        self.assertIn("Actual kernel home (current)", self.text)

    def test_r7_kernel_symbol_bgz17_scalar_sparse(self):
        """Must cite bgz17::scalar_sparse::tropical_spmv as the concrete symbol."""
        self.assertIn(
            "lance-graph::bgz17::scalar_sparse::tropical_spmv", self.text
        )

    def test_r7_notes_blasgraph_is_future_abstraction(self):
        """Must clarify that blasgraph is the eventual abstraction layer."""
        self.assertIn("eventual abstraction layer", self.text)

    def test_r7_instructs_to_cite_symbol_not_namespace(self):
        """Must instruct readers to cite the symbol, not the namespace."""
        self.assertIn("Cite the symbol, not the namespace", self.text)

    # ------------------------------------------------------------------
    # R-13 implementation primitives table
    # ------------------------------------------------------------------

    def test_r13_implementation_primitives_table_present(self):
        """R-13 must include an 'Implementation primitives' table."""
        self.assertIn("Implementation primitives (current substrate", self.text)

    def test_r13_primitives_cam_pq(self):
        """R-13 primitives table must include cam_pq::CamCodebook."""
        self.assertIn("cam_pq::CamCodebook", self.text)

    def test_r13_primitives_dn_tree(self):
        """R-13 primitives table must include ndarray::hpc::dn_tree."""
        self.assertIn("ndarray::hpc::dn_tree", self.text)

    def test_r13_primitives_merkle_tree(self):
        """R-13 primitives table must include ndarray::hpc::merkle_tree."""
        self.assertIn("ndarray::hpc::merkle_tree", self.text)

    def test_r13_primitives_codebook_handle_trait(self):
        """R-13 must describe the CodebookHandle trait."""
        self.assertIn("CodebookHandle", self.text)

    def test_r13_primitives_already_exist_statement(self):
        """R-13 must state that the primitives already exist (no new code)."""
        self.assertIn("the primitives above already exist", self.text)

    # ------------------------------------------------------------------
    # R-14 section
    # ------------------------------------------------------------------

    def test_r14_section_present(self):
        """R-14 resolution section must be present."""
        self.assertIn("### R-14 — Formal correctness via", self.text)

    def test_r14_pillar_10_pflug_pichler(self):
        """R-14 must describe Pillar 10 as Pflug-Pichler."""
        self.assertIn("Pillar 10, Pflug-Pichler", self.text)
        self.assertIn("jc::pflug", self.text)

    def test_r14_pillar_11_hambly_lyons(self):
        """R-14 must describe Pillar 11 as Hambly-Lyons."""
        self.assertIn("Pillar 11, Hambly-Lyons", self.text)
        self.assertIn("jc::hambly_lyons", self.text)

    def test_r14_hambly_lyons_feature_flag(self):
        """R-14 must document the --features hambly-lyons flag."""
        self.assertIn("--features hambly-lyons", self.text)

    def test_r14_pillar_11_pr348_date(self):
        """R-14 must note PR #348 landing date."""
        self.assertIn("PR #348", self.text)
        self.assertIn("2026-05-07", self.text)

    def test_r14_pillar_11_probe_criteria(self):
        """R-14 probe pass criteria must be documented."""
        self.assertIn("forward < 1e-9", self.text)
        self.assertIn("converse > 0.05", self.text)
        self.assertIn("discrimination ratio ≥ 1e6", self.text)

    def test_r14_pillar_10_default_build(self):
        """Pillar 10 must be noted as active in the default zero-dep build."""
        self.assertIn("default zero-dep build", self.text)

    def test_r14_open_work_pr350(self):
        """R-14 must mention PR #350 as open work (Goursat-PDE correction)."""
        self.assertIn("PR #350", self.text)
        self.assertIn("Goursat-PDE", self.text)

    def test_r14_uses_signature_truncated_not_pde(self):
        """R-14 must state the probe uses signature_truncated not PDE form."""
        self.assertIn("signature_truncated", self.text)

    def test_r14_falsifies_if_pillar10_flips(self):
        """R-14 must define a falsification condition."""
        self.assertIn("Falsifies if", self.text)

    # ------------------------------------------------------------------
    # R-15 section
    # ------------------------------------------------------------------

    def test_r15_section_present(self):
        """R-15 resolution section must be present."""
        self.assertIn("### R-15 —", self.text)
        self.assertIn("SignatureBasis<const DEPTH: usize>", self.text)

    def test_r15_basis_f32_impl(self):
        """R-15 must commit SignatureBasis as a Basis<f32> implementation."""
        self.assertIn("Basis<f32> for SignatureBasis<DEPTH>", self.text)

    def test_r15_apply_uses_signature_truncated(self):
        """R-15 apply() must call sigker::signature_truncated."""
        self.assertIn("sigker::signature_truncated", self.text)

    def test_r15_invert_is_unimplemented_na(self):
        """R-15 invert() must be documented as N/A (signature → path is many-to-one)."""
        self.assertIn("invert", self.text)
        self.assertIn("N/A", self.text)
        self.assertIn("unimplemented!", self.text)

    def test_r15_plan_g_fifth_lane(self):
        """R-15 must add a fifth Plan G lane."""
        self.assertIn("fifth lane", self.text)
        self.assertIn("stream signal", self.text.lower())

    def test_r15_target_module_path(self):
        """R-15 must specify the target module: ndarray::hpc::signature."""
        self.assertIn("ndarray::hpc::signature", self.text)

    def test_r15_compression_target_10x(self):
        """R-15 must include ~10× compression target for stream signal."""
        self.assertIn("~10×", self.text)

    def test_r15_cost_one_week(self):
        """R-15 must state ~1 week implementation cost."""
        self.assertIn("~1 week", self.text)

    def test_r15_falsification_condition(self):
        """R-15 must define a falsification condition."""
        idx = self.text.find("### R-15")
        self.assertNotEqual(idx, -1)
        tail = self.text[idx:]
        self.assertIn("Falsifies if", tail)

    # ------------------------------------------------------------------
    # Falsifiability matrix rows
    # ------------------------------------------------------------------

    def test_falsifiability_matrix_r14_pillar10_row(self):
        """Falsifiability matrix must include R-14 Pillar 10 row."""
        self.assertIn(
            "R-14 (Pillar 10 active)", self.text
        )

    def test_falsifiability_matrix_r14_pillar11_row(self):
        """Falsifiability matrix must include R-14 Pillar 11 row."""
        self.assertIn(
            "R-14 (Pillar 11 active)", self.text
        )

    def test_falsifiability_matrix_r15_row(self):
        """Falsifiability matrix must include R-15 SignatureBasis lane row."""
        self.assertIn("R-15 (SignatureBasis lane)", self.text)

    def test_falsifiability_matrix_r14_test_commands(self):
        """Falsifiability matrix must show cargo test commands for R-14."""
        self.assertIn("`cargo test -p jc`", self.text)
        self.assertIn("`cargo test -p jc --features hambly-lyons`", self.text)

    # ------------------------------------------------------------------
    # §13 summary — fifteen resolutions
    # ------------------------------------------------------------------

    def test_summary_says_fifteen_resolutions(self):
        """The preservation summary must mention fifteen resolutions."""
        self.assertIn("**The fifteen resolutions** R-1 through R-15", self.text)
        self.assertNotIn(
            "**The thirteen resolutions** R-1 through R-13", self.text
        )

    def test_summary_r13_primitives_listed(self):
        """Summary R-13 entry must list the four primitives."""
        self.assertIn("cam_pq", self.text)
        self.assertIn("dn_tree", self.text)
        self.assertIn("merkle_tree", self.text)

    def test_summary_r14_entry_present(self):
        """Summary must include R-14 entry with jc::pflug and jc::hambly_lyons."""
        self.assertIn(
            "R-14: formal correctness via `jc::pflug`", self.text
        )
        self.assertIn("jc::hambly_lyons", self.text)

    def test_summary_r15_entry_present(self):
        """Summary must include R-15 entry with SignatureBasis."""
        self.assertIn(
            "R-15: `SignatureBasis<DEPTH>: Basis<f32>`", self.text
        )

    def test_r7_summary_cites_bgz17_tropical_spmv(self):
        """Summary R-7 entry must cite the bgz17 symbol."""
        self.assertIn("bgz17::scalar_sparse::tropical_spmv", self.text)

    def test_stable_citation_ids_up_to_r15(self):
        """Citation IDs must be declared stable through R-15."""
        self.assertIn("(R-1 .. R-15) are stable", self.text)


# ---------------------------------------------------------------------------
# pr-x12-woa-multiarch-orchestration.md
# ---------------------------------------------------------------------------


class TestWoaMultiarchOrchestration(unittest.TestCase):
    """'Dispatch' → 'polyfill' terminology reframe throughout."""

    @classmethod
    def setUpClass(cls):
        cls.text = read("pr-x12-woa-multiarch-orchestration.md")

    # ------------------------------------------------------------------
    # Status/premise lines
    # ------------------------------------------------------------------

    def test_status_line_uses_polyfill_not_dispatch(self):
        """Status line in header must say 'polyfill decisions' not 'dispatch decisions'."""
        self.assertIn("per-arch polyfill decisions", self.text)

    def test_premise_uses_polyfill_contract(self):
        """Premise must say 'per-arch polyfill contract' not 'dispatch contract'."""
        self.assertIn("per-arch polyfill contract", self.text)

    # ------------------------------------------------------------------
    # Thesis paragraph
    # ------------------------------------------------------------------

    def test_thesis_describes_polyfill_surface(self):
        """Thesis must describe polyfill surface (ndarray::simd::* / ndarray::hpc::*)."""
        self.assertIn("ndarray::simd::*", self.text)
        self.assertIn("ndarray::hpc::*", self.text)

    def test_thesis_says_cfg_target_feature(self):
        """Thesis must attribute backend selection to cfg(target_feature = ...)."""
        self.assertIn("cfg(target_feature = ...)", self.text)

    def test_thesis_no_runtime_detection_language(self):
        """Thesis must not claim runtime CPU detection as the dispatch mechanism — it must negate it."""
        # The new text explicitly says "no runtime CPU detection" (negating it).
        # The correct check is that the doc denies runtime detection, not that it is absent.
        self.assertIn("no runtime CPU detection", self.text)

    # ------------------------------------------------------------------
    # Section 3 title
    # ------------------------------------------------------------------

    def test_section_3_title_polyfill(self):
        """Section 3 title must use 'polyfill' framing."""
        self.assertIn("Per-arch substrate via compile-time polyfill", self.text)
        self.assertNotIn(
            "Per-arch dispatch as a substrate property", self.text
        )

    # ------------------------------------------------------------------
    # No runtime HwCaps / OnceLock pattern
    # ------------------------------------------------------------------

    def test_no_hwcaps_struct_in_doc(self):
        """The doc must not describe a runtime HwCaps struct (old dispatch pattern)."""
        self.assertNotIn("pub struct HwCaps", self.text)

    def test_no_oncelock_runtime_dispatch(self):
        """The doc must not describe OnceLock-based runtime dispatch."""
        self.assertNotIn("OnceLock", self.text)

    def test_no_if_has_avx512_runtime_branch(self):
        """The doc must not contain 'if caps.has_amx' or similar runtime branches."""
        self.assertNotIn("caps.has_amx", self.text)
        self.assertNotIn("if caps.has_", self.text)

    # ------------------------------------------------------------------
    # New §3.1 — cfg-selected per-arch files
    # ------------------------------------------------------------------

    def test_section_3_1_polyfill_primitive_title(self):
        """§3.1 must be titled 'The polyfill primitive: cfg-selected per-arch files'."""
        self.assertIn(
            "The polyfill primitive: cfg-selected per-arch files", self.text
        )

    def test_section_3_1_cfg_pub_use_pattern(self):
        """§3.1 code example must show cfg-gated pub use pattern."""
        self.assertIn("pub use crate::simd_avx512::*", self.text)

    def test_section_3_1_three_backend_files_named(self):
        """§3.1 must name all three backend files."""
        self.assertIn("simd_avx512.rs", self.text)
        self.assertIn("simd_neon.rs", self.text)
        self.assertIn("simd_scalar.rs", self.text)

    def test_section_3_1_w1a_contract_referenced(self):
        """§3.1 must reference the W1a consumer contract."""
        self.assertIn("W1a contract", self.text)

    # ------------------------------------------------------------------
    # §3.2 — build-time CPU selection
    # ------------------------------------------------------------------

    def test_section_3_2_build_time_not_runtime(self):
        """§3.2 title must say 'Build-time CPU selection (not runtime detection)'."""
        self.assertIn(
            "Build-time CPU selection (not runtime detection)", self.text
        )

    def test_section_3_2_cargo_config_target_cpu(self):
        """§3.2 must reference .cargo/config.toml target-cpu setting."""
        self.assertIn("target-cpu=x86-64-v4", self.text)

    def test_section_3_2_no_fat_binary_probing(self):
        """§3.2 must state the fleet ships per-arch binaries, not fat binaries."""
        self.assertIn("per-arch binaries", self.text)
        # The doc uses "not a fat binary that probes" — the phrase appears as a negation.
        # What must NOT exist is a positive assertion that a fat binary with probing is used.
        self.assertIn("not a fat binary that probes", self.text)

    # ------------------------------------------------------------------
    # §3.3 — DCT crossover via cfg const
    # ------------------------------------------------------------------

    def test_section_3_3_crossover_is_compile_time_const(self):
        """§3.3 must describe DCT_BATCH_CROSSOVER as a compile-time constant."""
        self.assertIn("DCT_BATCH_CROSSOVER", self.text)
        self.assertIn("fixed in the compiled binary", self.text)

    def test_section_3_3_crossover_values_per_backend(self):
        """§3.3 must show at least two per-backend crossover values."""
        # simd_avx512.rs: 64, simd_neon.rs: 256, simd_scalar.rs: usize::MAX
        self.assertIn("64", self.text)
        self.assertIn("256", self.text)
        self.assertIn("usize::MAX", self.text)

    # ------------------------------------------------------------------
    # Per-arch polyfill transparency commitment
    # ------------------------------------------------------------------

    def test_commitment_uses_polyfill_transparency(self):
        """Commitment #2 must say 'polyfill transparency' not 'dispatch transparency'."""
        self.assertIn("Per-arch polyfill transparency", self.text)
        self.assertNotIn("Per-arch dispatch transparency", self.text)

    def test_commitment_consumers_never_name_backend_symbol(self):
        """Commitment must state consumers never name a backend symbol."""
        self.assertIn("Consumers never name a backend symbol", self.text)

    # ------------------------------------------------------------------
    # R-4 description update
    # ------------------------------------------------------------------

    def test_r4_clears_on_each_arch_not_at_most_one(self):
        """R-4 must say 'clears on each of' (not 'at most on 1 of')."""
        self.assertIn("clears on each of", self.text)
        self.assertNotIn("clears at most on 1 of", self.text)

    # ------------------------------------------------------------------
    # GPU offload hook uses polyfill language
    # ------------------------------------------------------------------

    def test_gpu_offload_uses_backend_target_not_dispatch_target(self):
        """GPU offload hook must say 'backend_target' not 'dispatch_target'."""
        self.assertIn("backend_target", self.text)
        self.assertNotIn("dispatch_target", self.text)


# ---------------------------------------------------------------------------
# pr-x12-x266-3dgs-spacetime-upscaling.md
# ---------------------------------------------------------------------------


class TestX2663dgsSpacetimeUpscaling(unittest.TestCase):
    """'canon-fixed' vs 'scheduled' status distinction in requirements table."""

    @classmethod
    def setUpClass(cls):
        cls.text = read("pr-x12-x266-3dgs-spacetime-upscaling.md")

    # ------------------------------------------------------------------
    # canon-fixed / scheduled distinction
    # ------------------------------------------------------------------

    def test_canon_fixed_term_present(self):
        """The term 'canon-fixed' must appear in the doc."""
        self.assertIn("canon-fixed", self.text)

    def test_canon_fixed_definition_provided(self):
        """'Canon-fixed' must be defined in the doc."""
        self.assertIn(
            '**"Canon-fixed"** = the resolution doc commits the design', self.text
        )

    def test_scheduled_definition_provided(self):
        """'Scheduled' must be defined as having a named plan card."""
        self.assertIn(
            '**"scheduled"** = the implementation has a named plan card', self.text
        )

    def test_no_shipping_code_today_statement(self):
        """Must state none of the requirements have shipping code today."""
        self.assertIn("None of the above have shipping code today", self.text)

    # ------------------------------------------------------------------
    # Basis<T> trait status
    # ------------------------------------------------------------------

    def test_basis_trait_status_is_canon_fixed(self):
        """Basis<T> trait status must be 'canon-fixed'."""
        # Row: Basis<T> trait | R-1, M:E-A | canon-fixed ...
        pattern = r"`Basis<T>`.*canon-fixed"
        self.assertRegex(self.text, pattern)

    def test_basis_trait_impl_is_plan_a4(self):
        """Basis<T> implementation must be scheduled in Plan A4."""
        # The status cell should mention Plan A4
        self.assertIn("Plan A4", self.text)

    # ------------------------------------------------------------------
    # Header byte status
    # ------------------------------------------------------------------

    def test_header_byte_status_is_canon_fixed(self):
        """Header byte (R-2) status must be 'canon-fixed'."""
        # Row: Header byte stable | R-2, M:E-J bits 0-1 | canon-fixed ...
        pattern = r"R-2.*canon-fixed"
        self.assertRegex(self.text, pattern)

    def test_header_byte_commits_bits_0_1(self):
        """Header byte status must commit bits 0-1 as header_kind."""
        self.assertIn("R-2 commits bits 0-1 = `header_kind`", self.text)

    def test_header_byte_impl_is_plan_a8(self):
        """Header byte wire-format implementation must be in Plan A8."""
        self.assertIn("Plan A8", self.text)

    # ------------------------------------------------------------------
    # Federated codebook status
    # ------------------------------------------------------------------

    def test_federated_codebook_status_is_canon_fixed(self):
        """Federated codebook for scene anchors must be 'canon-fixed'."""
        pattern = r"R-13.*canon-fixed"
        self.assertRegex(self.text, pattern)

    def test_federated_codebook_commits_option_a(self):
        """R-13 status must commit Option A: per-shard codebook for Plan F v1."""
        self.assertIn("R-13 commits Option A: per-shard codebook for Plan F v1", self.text)

    def test_federated_codebook_impl_is_plan_f(self):
        """Federated codebook implementation must be in Plan F."""
        self.assertIn("implementation in Plan F", self.text)

    # ------------------------------------------------------------------
    # Codec body decoupled status
    # ------------------------------------------------------------------

    def test_codec_body_decoupled_doc_commitment_ci_pending(self):
        """Codec body decoupled status must note doc commitment + CI check pending."""
        self.assertIn("doc commitment; CI check pending", self.text)

    # ------------------------------------------------------------------
    # EWA splat rasterizer remains scheduled
    # ------------------------------------------------------------------

    def test_ewa_splat_rasterizer_scheduled(self):
        """EWA splat rasterizer status must still be 'scheduled' (not canon-fixed)."""
        # The requirements table row is: | EWA splat rasterizer … | Plan E | scheduled |
        self.assertIn(
            "| EWA splat rasterizer as `Basis<f16>` impl | Plan E | scheduled |",
            self.text,
        )

    # ------------------------------------------------------------------
    # Plan G lane status
    # ------------------------------------------------------------------

    def test_plan_g_video_lane_scheduled(self):
        """Plan G video lane per-arch latency must remain 'scheduled'."""
        idx = self.text.find("Plan G video lane")
        self.assertNotEqual(idx, -1)
        window = self.text[idx : idx + 80]
        self.assertIn("scheduled", window)

    # ------------------------------------------------------------------
    # Regression: old terse status strings gone
    # ------------------------------------------------------------------

    def test_old_landed_status_replaced_for_basis_trait(self):
        """Old status 'landed in concept; implementation in Plan A4' must be gone."""
        self.assertNotIn(
            "landed in concept; implementation in Plan A4", self.text
        )

    def test_old_landed_status_replaced_for_header_byte(self):
        """Old status 'landed' (bare) for R-2 must be replaced."""
        # Old text just said "landed"; new text says "canon-fixed (R-2 commits bits 0-1 = `header_kind`)"
        # We check the bare "| landed |" pattern is gone near the R-2 row
        old_pattern = r"\|\s*R-2, M:E-J bits 0-1\s*\|\s*landed\s*\|"
        self.assertIsNone(
            re.search(old_pattern, self.text),
            msg="Old bare 'landed' status for R-2 should be replaced",
        )

    def test_old_landed_status_replaced_for_r13(self):
        """Old status 'landed' (bare) for R-13 must be replaced with canon-fixed."""
        old_pattern = r"\|\s*R-13\s*\|\s*landed\s*\|"
        self.assertIsNone(
            re.search(old_pattern, self.text),
            msg="Old bare 'landed' status for R-13 should be replaced",
        )


# ---------------------------------------------------------------------------
# Cross-document consistency tests
# ---------------------------------------------------------------------------


class TestCrossDocumentConsistency(unittest.TestCase):
    """Verify that related claims are consistent across multiple changed docs."""

    @classmethod
    def setUpClass(cls):
        cls.bgz_jc = read("pr-x12-bgz-jc-substrate-synergies.md")
        cls.cam_pq = read("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md")
        cls.delta = read("pr-x12-canon-resolutions-delta.md")
        cls.substrate = read("pr-x12-substrate-canon-resolutions.md")
        cls.woa = read("pr-x12-woa-multiarch-orchestration.md")
        cls.x266 = read("pr-x12-x266-3dgs-spacetime-upscaling.md")
        cls.gguf = read("pr-x12-gguf-llm-weights-encoding.md")

    # ------------------------------------------------------------------
    # tropical_spmv symbol consistent across docs
    # ------------------------------------------------------------------

    def test_tropical_spmv_symbol_consistent_in_delta_and_substrate(self):
        """Both canon-resolutions-delta and substrate-canon-resolutions must cite bgz17::scalar_sparse::tropical_spmv."""
        symbol = "bgz17::scalar_sparse::tropical_spmv"
        self.assertIn(symbol, self.delta)
        self.assertIn(symbol, self.substrate)

    # ------------------------------------------------------------------
    # R-14/R-15 mentioned in both docs that added them
    # ------------------------------------------------------------------

    def test_r14_present_in_both_resolution_docs(self):
        """R-14 must be present in both canon-resolutions-delta and substrate-canon-resolutions."""
        self.assertIn("R-14", self.delta)
        self.assertIn("R-14", self.substrate)

    def test_r15_present_in_both_resolution_docs(self):
        """R-15 must be present in both canon-resolutions-delta and substrate-canon-resolutions."""
        self.assertIn("R-15", self.delta)
        self.assertIn("R-15", self.substrate)

    # ------------------------------------------------------------------
    # Gap namespace separation enforced across docs
    # ------------------------------------------------------------------

    def test_bgz_jc_g1_in_bgz_doc_uses_section_51(self):
        """bgz-jc §5.1 must be the canonical home of G-1 (jd-nd)."""
        self.assertIn("5.1 `jd-nd`", self.bgz_jc)
        self.assertIn("G-1", self.bgz_jc)

    def test_cam_pq_cross_refs_bgz_jc_g1_with_prefix(self):
        """cam-pq doc must use 'bgz-jc G-1' prefix (not bare 'G-1') for cross-ref."""
        self.assertIn("bgz-jc G-1", self.cam_pq)

    def test_cam_pq_doc_has_its_own_g1(self):
        """cam-pq doc must have its own G-1 (a different gap in a different namespace)."""
        # Both docs have **G-1** but in different namespaces
        self.assertIn("**G-1**", self.cam_pq)
        self.assertIn("**G-1**", self.bgz_jc)

    # ------------------------------------------------------------------
    # SignatureBasis<DEPTH> consistent across docs
    # ------------------------------------------------------------------

    def test_signature_basis_depth_consistent_across_docs(self):
        """SignatureBasis<DEPTH> must appear in both resolution docs."""
        self.assertIn("SignatureBasis<DEPTH>", self.delta)
        self.assertIn("SignatureBasis<DEPTH>", self.substrate)

    def test_signature_basis_uses_signature_truncated_not_pde_both_docs(self):
        """Both resolution docs must specify signature_truncated (not the buggy PDE form)."""
        self.assertIn("signature_truncated", self.delta)
        self.assertIn("signature_truncated", self.substrate)
        # Neither should recommend the PDE form in the context of SignatureBasis
        # (PDE is mentioned as the buggy alternative, which is fine)

    # ------------------------------------------------------------------
    # Polyfill terminology: woa doc updated but others don't conflict
    # ------------------------------------------------------------------

    def test_woa_doc_does_not_use_hwcaps_struct(self):
        """WoA doc must not describe HwCaps (old runtime-dispatch pattern)."""
        self.assertNotIn("pub struct HwCaps", self.woa)

    def test_woa_doc_uses_polyfill_not_dispatch_in_premise(self):
        """WoA doc premise must use 'polyfill' language."""
        self.assertIn("polyfill contract", self.woa)
        self.assertNotIn("dispatch contract", self.woa)

    # ------------------------------------------------------------------
    # Phone-class memory: gguf doc correctly separated from weight savings
    # ------------------------------------------------------------------

    def test_gguf_kv_cache_is_separate_concern(self):
        """GGUF doc must separate KV cache from weight compression as distinct concerns."""
        # Both must appear, and the "both lanes needed" sentence must be present
        self.assertIn("KV cache", self.gguf)
        self.assertIn("weight compression", self.gguf)
        self.assertIn("Both lanes are needed", self.gguf)

    # ------------------------------------------------------------------
    # R-13 primitives consistent: delta, substrate, cam-pq all agree
    # ------------------------------------------------------------------

    def test_r13_primitives_consistent_dn_tree_in_delta_and_substrate(self):
        """dn_tree must be named as R-13 primitive in both resolution docs."""
        self.assertIn("dn_tree", self.delta)
        self.assertIn("dn_tree", self.substrate)

    def test_r13_primitives_consistent_merkle_tree_in_delta_and_substrate(self):
        """merkle_tree must be named as R-13 primitive in both resolution docs."""
        self.assertIn("merkle_tree", self.delta)
        self.assertIn("merkle_tree", self.substrate)

    # ------------------------------------------------------------------
    # x266 doc: canon-fixed items reference their source resolutions
    # ------------------------------------------------------------------

    def test_x266_canon_fixed_basis_trait_cites_r1(self):
        """x266 canon-fixed Basis<T> status must cite R-1."""
        self.assertIn("R-1 trait shape committed", self.x266)

    def test_x266_canon_fixed_r13_cites_option_a(self):
        """x266 canon-fixed R-13 status must name Option A."""
        self.assertIn("Option A", self.x266)


if __name__ == "__main__":
    # Run all tests with verbose output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
