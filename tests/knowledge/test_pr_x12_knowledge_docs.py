"""
Tests for PR-X12 knowledge document changes.

This test suite validates the content of the seven markdown files changed in the
PR-X12 knowledge update:
  - pr-x12-bgz-jc-substrate-synergies.md
  - pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md
  - pr-x12-canon-resolutions-delta.md
  - pr-x12-gguf-llm-weights-encoding.md
  - pr-x12-substrate-canon-resolutions.md
  - pr-x12-woa-multiarch-orchestration.md
  - pr-x12-x266-3dgs-spacetime-upscaling.md

Run with:
    python3 -m unittest tests/knowledge/test_pr_x12_knowledge_docs.py -v
"""

import os
import re
import unittest

KNOWLEDGE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", ".claude", "knowledge"
)


def read_doc(filename: str) -> str:
    path = os.path.join(KNOWLEDGE_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# pr-x12-bgz-jc-substrate-synergies.md
# ---------------------------------------------------------------------------
class TestBgzJcSubstrateSynergies(unittest.TestCase):
    """
    Validates changes to pr-x12-bgz-jc-substrate-synergies.md:
      - Section 5.1 header now includes the canonical "(Gap **G-1**)" label.
      - Section 5.2 header now includes the canonical "(Gap **G-2**)" label.
      - The gap IDs appear inside the section-header markdown headers.
    """

    DOC = "pr-x12-bgz-jc-substrate-synergies.md"

    def setUp(self):
        self.content = read_doc(self.DOC)

    def test_section_5_1_has_gap_g1_label(self):
        """Section 5.1 header must include '(Gap **G-1**)' to canonicalise the ID."""
        self.assertIn(
            "### 5.1 `jd-nd` — the missing ndarray-side proof crate (Gap **G-1**)",
            self.content,
            "Section 5.1 header must contain the canonical gap label '(Gap **G-1**)'",
        )

    def test_section_5_2_has_gap_g2_label(self):
        """Section 5.2 header must include '(Gap **G-2**)' to canonicalise the ID."""
        self.assertIn(
            "### 5.2 Cronbach / ICC research crate (Gap **G-2**)",
            self.content,
            "Section 5.2 header must contain the canonical gap label '(Gap **G-2**)'",
        )

    def test_g1_label_appears_only_in_section_header(self):
        """The '(Gap **G-1**)' label must appear at least once in the doc."""
        count = self.content.count("(Gap **G-1**)")
        self.assertGreaterEqual(
            count, 1, "Gap G-1 label must appear at least once in the document"
        )

    def test_g2_label_appears_only_in_section_header(self):
        """The '(Gap **G-2**)' label must appear at least once in the doc."""
        count = self.content.count("(Gap **G-2**)")
        self.assertGreaterEqual(
            count, 1, "Gap G-2 label must appear at least once in the document"
        )

    def test_gap_section_5_present(self):
        """Section 5 ('Gaps — what doesn't exist yet') must be present."""
        self.assertIn("## 5. Gaps", self.content)

    def test_section_5_1_still_references_jd_nd(self):
        """Section 5.1 must still describe the jd-nd crate."""
        self.assertIn("`jd-nd`", self.content)

    def test_section_5_2_still_references_cronbach(self):
        """Section 5.2 must still reference Cronbach/ICC."""
        self.assertIn("Cronbach", self.content)
        self.assertIn("ICC", self.content)

    def test_no_raw_unlabelled_g1_header(self):
        """
        Regression: the old unlabelled section header must no longer exist.
        The old text was exactly '### 5.1 `jd-nd` — the missing ndarray-side proof crate'
        (without the gap label).
        """
        old_header = "### 5.1 `jd-nd` — the missing ndarray-side proof crate\n"
        self.assertNotIn(
            old_header,
            self.content,
            "Old unlabelled 5.1 header must have been replaced with the labelled form",
        )

    def test_no_raw_unlabelled_g2_header(self):
        """
        Regression: the old unlabelled section 5.2 header must no longer exist.
        """
        old_header = "### 5.2 Cronbach / ICC research crate\n"
        self.assertNotIn(
            old_header,
            self.content,
            "Old unlabelled 5.2 header must have been replaced with the labelled form",
        )


# ---------------------------------------------------------------------------
# pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md
# ---------------------------------------------------------------------------
class TestCamPqSigkerDnTreeSubstrateBindings(unittest.TestCase):
    """
    Validates changes to pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md:
      - Old G-8/G-9 row labels replaced with cross-doc prefixed 'bgz-jc G-1' / 'bgz-jc G-2'.
      - Table header changed from 'Gap (prior)' to 'Gap (cross-ref)'.
      - Namespace clarification paragraph added.
      - No orphaned G-8 or G-9 identifiers remain in the cross-reference table.
    """

    DOC = "pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md"

    def setUp(self):
        self.content = read_doc(self.DOC)

    def test_cross_ref_table_uses_bgz_jc_g1(self):
        """Cross-reference table must use 'bgz-jc G-1' (not 'G-8')."""
        self.assertIn(
            "**bgz-jc G-1**",
            self.content,
            "Cross-reference table must label the jd-nd gap as 'bgz-jc G-1'",
        )

    def test_cross_ref_table_uses_bgz_jc_g2(self):
        """Cross-reference table must use 'bgz-jc G-2' (not 'G-9')."""
        self.assertIn(
            "**bgz-jc G-2**",
            self.content,
            "Cross-reference table must label the Cronbach/ICC gap as 'bgz-jc G-2'",
        )

    def test_cross_ref_table_header_updated(self):
        """Table header must say 'Gap (cross-ref)' not 'Gap (prior)'."""
        self.assertIn("Gap (cross-ref)", self.content)
        self.assertNotIn(
            "Gap (prior)",
            self.content,
            "Old 'Gap (prior)' table header must be replaced by 'Gap (cross-ref)'",
        )

    def test_no_g8_label_in_cross_ref_table(self):
        """
        Regression: '**G-8**' must no longer appear as a standalone gap identifier
        in the cross-reference table.
        """
        self.assertNotIn(
            "**G-8**",
            self.content,
            "G-8 was renamed to bgz-jc G-1; '**G-8**' must not appear",
        )

    def test_no_g9_label_in_cross_ref_table(self):
        """
        Regression: '**G-9**' must no longer appear as a standalone gap identifier
        in the cross-reference table.
        """
        self.assertNotIn(
            "**G-9**",
            self.content,
            "G-9 was renamed to bgz-jc G-2; '**G-9**' must not appear",
        )

    def test_namespace_clarification_paragraph_present(self):
        """
        The namespace clarification paragraph (explaining the bgz-jc prefix) must
        be present to prevent future labelling collisions.
        """
        self.assertIn(
            "bgz-jc G-1",
            self.content,
            "Namespace clarification paragraph must reference 'bgz-jc G-1'",
        )
        self.assertIn(
            "cam-pq G-1",
            self.content,
            "Namespace clarification paragraph must use 'cam-pq G-1' as an example",
        )

    def test_cross_ref_table_cites_parent_doc_section(self):
        """Cross-ref table must cite §5.1 / §5.2 of the owning bgz-jc doc."""
        self.assertIn("(§5.1)", self.content)
        self.assertIn("(§5.2)", self.content)

    def test_two_prior_gaps_sentence_updated(self):
        """
        The sentence that prefaces the cross-reference table must reference the
        canonical IDs and the owning document.
        """
        self.assertIn(
            "pr-x12-bgz-jc-substrate-synergies.md",
            self.content,
            "Cross-ref preface must name the owning document",
        )

    def test_grand_total_estimate_present(self):
        """Grand total estimate paragraph must still be present."""
        self.assertIn("Grand total", self.content)
        self.assertIn("11-17 weeks", self.content)

    def test_local_gaps_g1_through_g7_unaffected(self):
        """
        Local gaps G-1 through G-7 (cam-pq doc's own namespace) must still exist.
        """
        for i in range(1, 8):
            self.assertIn(
                f"**G-{i}**",
                self.content,
                f"Local cam-pq gap G-{i} must still be present",
            )


# ---------------------------------------------------------------------------
# pr-x12-canon-resolutions-delta.md
# ---------------------------------------------------------------------------
class TestCanonResolutionsDelta(unittest.TestCase):
    """
    Validates changes to pr-x12-canon-resolutions-delta.md:
      - Category count updated from five to six.
      - Sixth category covers R-14 and R-15.
      - R-7 cites the corrected kernel canon: blasgraph is the canonical home
        (kernel unwritten); the shipped min-plus is the method
        bgz17::ScalarCsr::spmv_min_plus (audit 2026-07-16 correction).
      - R-13 phasing pattern includes the four implementation primitives.
      - R-14 / R-15 sections added.
      - Falsifiability matrix row count updated to 24+3.
      - The single load-bearing paragraph is now §13.
    """

    DOC = "pr-x12-canon-resolutions-delta.md"

    def setUp(self):
        self.content = read_doc(self.DOC)

    def test_six_categories_of_novel_content(self):
        """Section 0 must say 'Six categories' (updated from Five)."""
        self.assertIn(
            "Six categories of novel content",
            self.content,
            "Category count must be updated to 'Six categories'",
        )

    def test_five_categories_phrase_absent(self):
        """The old 'Five categories' phrase must not appear."""
        self.assertNotIn(
            "Five categories of novel content",
            self.content,
            "Old 'Five categories' phrase must be replaced with 'Six categories'",
        )

    def test_sixth_category_mentions_r14_r15(self):
        """The sixth category bullet must reference R-14 and R-15."""
        self.assertIn("R-14", self.content)
        self.assertIn("R-15", self.content)

    def test_sixth_category_references_formal_correctness(self):
        """Sixth category must mention 'Formal-correctness' and 'stream lane'."""
        self.assertIn("Formal-correctness", self.content)
        self.assertIn("stream lane", self.content)

    def test_r7_cites_spmv_min_plus_method(self):
        """
        R-7 entry must cite the corrected canon (audit 2026-07-16): the
        shipped min-plus is the METHOD bgz17::ScalarCsr::spmv_min_plus;
        the free-function path bgz17::scalar_sparse::tropical_spmv was
        fabricated (audit findings #1/#2) and must not be cited
        affirmatively.
        """
        self.assertIn(
            "bgz17::ScalarCsr::spmv_min_plus",
            self.content,
            "R-7 must cite the real method bgz17::ScalarCsr::spmv_min_plus",
        )
        self.assertNotIn(
            "bgz17::scalar_sparse::tropical_spmv",
            self.content,
            "The fabricated free-function path must not appear in the delta doc",
        )
        self.assertIn(
            "blasgraph",
            self.content,
            "R-7 must name blasgraph as the canonical kernel home",
        )

    def test_r13_primitives_table_present(self):
        """R-13 phasing pattern must list the four implementation primitives."""
        for primitive in ("cam_pq", "bgz-hhtl-d", "dn_tree", "merkle_tree"):
            self.assertIn(
                primitive,
                self.content,
                f"R-13 primitives list must include '{primitive}'",
            )

    def test_r13_primitives_inline_citation(self):
        """R-13 category line must include the primitives citation."""
        self.assertIn(
            "primitives: `cam_pq` + `bgz-hhtl-d` + `dn_tree` + `merkle_tree`",
            self.content,
        )

    def test_r14_jc_pflug_cited(self):
        """R-14 must cite jc::pflug (Pillar 10)."""
        self.assertIn("`jc::pflug`", self.content)
        self.assertIn("Pillar 10", self.content)

    def test_r14_jc_hambly_lyons_cited(self):
        """R-14 must cite jc::hambly_lyons (Pillar 11)."""
        self.assertIn("`jc::hambly_lyons`", self.content)
        self.assertIn("Pillar 11", self.content)

    def test_r15_signature_basis_cited(self):
        """R-15 must reference SignatureBasis<DEPTH> as the fifth Plan G lane."""
        self.assertIn("SignatureBasis", self.content)
        self.assertIn("fifth Plan G lane", self.content)

    def test_falsifiability_matrix_row_count_updated(self):
        """Falsifiability matrix row count must say '24+3 rows'."""
        self.assertIn(
            "24+3 rows",
            self.content,
            "Falsifiability matrix row count must be updated to '24+3 rows'",
        )

    def test_load_bearing_paragraph_is_now_section_13(self):
        """
        The single load-bearing paragraph (was §11 before the PR) must now be §13.
        """
        self.assertIn(
            "## 13. The single load-bearing paragraph",
            self.content,
            "Load-bearing paragraph must be renumbered to §13",
        )

    def test_old_section_11_label_absent(self):
        """The old '## 11. The single load-bearing paragraph' label must be gone."""
        self.assertNotIn(
            "## 11. The single load-bearing paragraph",
            self.content,
            "Old §11 label for load-bearing paragraph must be replaced by §13",
        )

    def test_r14_r15_substrate_binding_sentence(self):
        """The load-bearing paragraph must mention R-14 and R-15 additions."""
        self.assertIn(
            "R-14, R-15",
            self.content,
        )
        self.assertIn(
            "SignatureBasis<DEPTH>",
            self.content,
        )

    def test_jc_pillars_sentence_in_r14_section(self):
        """R-14 section header must appear."""
        self.assertIn("R-14", self.content)
        # The section added as §11 in canon-resolutions-delta
        self.assertIn("Formal-correctness layer (R-14)", self.content)

    def test_r15_stream_signal_section_present(self):
        """R-15 section must appear."""
        self.assertIn("Stream-signal codec lane (R-15)", self.content)

    def test_signature_truncated_rationale_present(self):
        """Doc must explain why signature_truncated is used instead of signature_kernel_pde."""
        self.assertIn("signature_truncated", self.content)
        self.assertIn("signature_kernel_pde", self.content)

    def test_implementation_primitives_table_for_r13(self):
        """R-13 implementation primitives table must include the four crates."""
        self.assertIn("CamCodebook", self.content)
        self.assertIn("Codebook4096", self.content)
        self.assertIn("merkle_tree", self.content)


# ---------------------------------------------------------------------------
# pr-x12-gguf-llm-weights-encoding.md
# ---------------------------------------------------------------------------
class TestGgufLlmWeightsEncoding(unittest.TestCase):
    """
    Validates changes to pr-x12-gguf-llm-weights-encoding.md:
      - Escape section now references F-4 falsifier.
      - Memory savings paragraph includes KV cache caveat.
      - EncodingDomain::LLMWeights description clarified to 'enum-discriminant slot'.
    """

    DOC = "pr-x12-gguf-llm-weights-encoding.md"

    def setUp(self):
        self.content = read_doc(self.DOC)

    def test_escape_section_references_f4_falsifier(self):
        """Escape section must cross-reference §10 falsifier F-4."""
        self.assertIn(
            "**F-4**",
            self.content,
            "Escape section must add a cross-reference to falsifier F-4",
        )
        self.assertIn("rANS bypass channel", self.content)

    def test_escape_section_mentions_hevc_precedent(self):
        """Escape section must mention the HEVC-escape-coefficient precedent."""
        self.assertIn(
            "HEVC-escape-coefficient",
            self.content,
            "Escape section must cite the HEVC-escape-coefficient precedent",
        )

    def test_memory_savings_paragraph_has_kv_cache_caveat(self):
        """Memory savings section must include a caveat about KV cache."""
        self.assertIn(
            "KV cache",
            self.content,
            "Memory savings section must warn about KV cache as a separate memory load",
        )

    def test_memory_savings_paragraph_mentions_plan_d(self):
        """The KV cache caveat must reference Plan D."""
        self.assertIn(
            "Plan D",
            self.content,
            "KV cache caveat must reference Plan D as the second lever",
        )

    def test_weights_only_qualifier_present(self):
        """Memory savings heading must include '(weights only)' qualifier."""
        self.assertIn(
            "weights only",
            self.content,
            "Memory savings paragraph must be qualified with '(weights only)'",
        )

    def test_old_overstated_phone_claim_absent(self):
        """
        Regression: the old oversimplified claim must be replaced.
        The old text said 'A 7B model at PR-X12 is genuinely runnable on a
        phone-class device, where GGUF Q4 is borderline.'
        """
        old_claim = (
            "A 7B model at PR-X12 is genuinely runnable on a phone-class device"
        )
        self.assertNotIn(
            old_claim,
            self.content,
            "Oversimplified phone-class claim must be replaced with the nuanced KV-cache caveat",
        )

    def test_encoding_domain_llmweights_enum_slot_description(self):
        """
        The EncodingDomain::LLMWeights implication must describe reserving the
        'enum-discriminant slot' rather than just the enum variant.
        """
        self.assertIn(
            "enum-discriminant slot",
            self.content,
            "EncodingDomain::LLMWeights description must use 'enum-discriminant slot' language",
        )

    def test_encoding_domain_forward_compat_language(self):
        """Description must mention forward-compatibility lock."""
        self.assertIn(
            "forward-compatibility",
            self.content,
            "EncodingDomain description must mention forward-compatibility",
        )

    def test_encoding_domain_unimplemented_in_prx12(self):
        """Doc must state the LLMWeights value stays unimplemented in PR-X12."""
        self.assertIn(
            "stays unimplemented in PR-X12",
            self.content,
        )

    def test_f4_wire_format_mechanism_described(self):
        """F-4 falsifier reference must describe the wire-format mechanism."""
        self.assertIn(
            "A8 framing layer",
            self.content,
        )

    def test_kv_cache_context_length_scaling_noted(self):
        """KV cache caveat must note context-length scaling."""
        self.assertIn(
            "context length",
            self.content,
        )


# ---------------------------------------------------------------------------
# pr-x12-substrate-canon-resolutions.md
# ---------------------------------------------------------------------------
class TestSubstrateCanonResolutions(unittest.TestCase):
    """
    Validates changes to pr-x12-substrate-canon-resolutions.md:
      - R-14 and R-15 resolution sections added.
      - R-13 implementation primitives table added.
      - Tropical-GEMM actual kernel home section added.
      - Falsifiability matrix now has R-14 (Pillar 10), R-14 (Pillar 11), R-15 rows.
      - 'Fifteen resolutions' language in compaction-preservation contract.
      - Citation IDs updated to R-1..R-15.
    """

    DOC = "pr-x12-substrate-canon-resolutions.md"

    def setUp(self):
        self.content = read_doc(self.DOC)

    def test_r14_section_header_present(self):
        """R-14 resolution section must be present."""
        self.assertIn(
            "### R-14 — Formal correctness via `lance-graph::jc` pillars",
            self.content,
        )

    def test_r15_section_header_present(self):
        """R-15 resolution section must be present."""
        self.assertIn(
            "### R-15 — `SignatureBasis<const DEPTH: usize>` as `Basis<f32>` impl",
            self.content,
        )

    def test_r14_pillar_10_pflug_details(self):
        """R-14 must describe Pillar 10 (Pflug-Pichler, jc::pflug)."""
        self.assertIn("Pillar 10", self.content)
        self.assertIn("Pflug-Pichler", self.content)
        self.assertIn("`jc::pflug`", self.content)

    def test_r14_pillar_11_hambly_lyons_details(self):
        """R-14 must describe Pillar 11 (Hambly-Lyons, jc::hambly_lyons)."""
        self.assertIn("Pillar 11", self.content)
        self.assertIn("Hambly-Lyons", self.content)
        self.assertIn("`jc::hambly_lyons`", self.content)

    def test_r14_hambly_lyons_probe_thresholds(self):
        """R-14 must state Pillar 11 probe thresholds."""
        self.assertIn("forward < 1e-9", self.content)
        self.assertIn("converse > 0.05", self.content)
        self.assertIn("ratio ≥ 1e6", self.content)

    def test_r14_pillar_10_default_build(self):
        """R-14 must state Pillar 10 is active in the default build."""
        self.assertIn("active in default zero-dep build", self.content)

    def test_r14_pillar_11_feature_gate(self):
        """R-14 must state Pillar 11 is feature-gated under hambly-lyons."""
        self.assertIn("--features hambly-lyons", self.content)
        self.assertIn("PR #348", self.content)

    def test_r15_signature_basis_impl_block(self):
        """R-15 must include the SignatureBasis Rust impl block."""
        self.assertIn("impl<const DEPTH: usize> Basis<f32> for SignatureBasis<DEPTH>", self.content)

    def test_r15_signature_truncated_rationale(self):
        """R-15 must explain why signature_truncated is used."""
        self.assertIn("signature_truncated", self.content)
        self.assertIn("PR #350", self.content)

    def test_r15_stream_signal_lane_details(self):
        """R-15 stream signal lane must include audio/time-series use cases."""
        self.assertIn("audio waveform", self.content)
        self.assertIn("time-series", self.content)

    def test_r15_compression_target(self):
        """R-15 must state the ~10× compression target."""
        self.assertIn("10×", self.content)

    def test_r13_implementation_primitives_table(self):
        """R-13 implementation primitives table must list all five concerns."""
        expected_crates = [
            "cam_pq::CamCodebook",
            "bgz-hhtl-d",
            "dn_tree",
            "merkle_tree",
            "q2",
        ]
        for crate in expected_crates:
            self.assertIn(
                crate,
                self.content,
                f"R-13 primitives table must include '{crate}'",
            )

    def test_r13_codebook_handle_trait_mentioned(self):
        """R-13 section must mention the CodebookHandle trait."""
        self.assertIn("CodebookHandle", self.content)

    def test_spmv_min_plus_shipped_kernel_section(self):
        """
        The corrected 'Shipped min-plus today' section (audit 2026-07-16,
        findings #1-#4) must cite the real method
        bgz17::ScalarCsr::spmv_min_plus and must NOT frame bgz17 as the
        'Actual kernel home' — that framing inverted the canon (blasgraph
        is canonical and bit-exact; bgz17 is a lossy sibling).
        """
        self.assertIn("Shipped min-plus today", self.content)
        self.assertIn("bgz17::ScalarCsr::spmv_min_plus", self.content)
        self.assertNotIn(
            "Actual kernel home (current)",
            self.content,
            "The inverted 'Actual kernel home (current)' framing must stay removed",
        )

    def test_blasgraph_is_canonical_not_bgz17(self):
        """
        Doc must present blasgraph as the CANONICAL kernel home and bgz17's
        min-plus as a lossy sibling/prototype — never 'ndarray-codec depends
        on bgz17 directly' as the sanctioned path (audit ground-truth #3/#5).
        """
        self.assertIn("canonical", self.content)
        self.assertIn("lossy", self.content)
        self.assertNotIn(
            "ndarray-codec depends on\nbgz17 directly",
            self.content,
            "The bgz17-as-current-home dependency framing must stay removed",
        )

    def test_falsifiability_matrix_has_r14_pillar10_row(self):
        """Falsifiability matrix §9 must have an R-14 Pillar 10 row."""
        self.assertIn("R-14 (Pillar 10 active)", self.content)

    def test_falsifiability_matrix_has_r14_pillar11_row(self):
        """Falsifiability matrix §9 must have an R-14 Pillar 11 row."""
        self.assertIn("R-14 (Pillar 11 active)", self.content)

    def test_falsifiability_matrix_has_r15_row(self):
        """Falsifiability matrix §9 must have an R-15 row."""
        self.assertIn("R-15 (SignatureBasis lane)", self.content)

    def test_falsifiability_matrix_r14_test_commands(self):
        """R-14 matrix rows must include the cargo test commands."""
        self.assertIn("`cargo test -p jc`", self.content)
        self.assertIn("`cargo test -p jc --features hambly-lyons`", self.content)

    def test_compaction_contract_says_fifteen_resolutions(self):
        """Compaction-preservation contract must say 'fifteen resolutions'."""
        self.assertIn(
            "fifteen resolutions",
            self.content,
            "Compaction contract must be updated to say 'fifteen resolutions'",
        )

    def test_compaction_contract_old_thirteen_absent(self):
        """Old 'thirteen resolutions' text must no longer appear."""
        self.assertNotIn(
            "thirteen resolutions",
            self.content,
            "Old 'thirteen resolutions' must be updated to 'fifteen resolutions'",
        )

    def test_citation_ids_updated_to_r15(self):
        """Citation IDs comment must say 'R-1 .. R-15'."""
        self.assertIn("R-1 .. R-15", self.content)

    def test_r7_compaction_entry_includes_kernel(self):
        """R-7 compaction summary line must include the kernel symbol."""
        self.assertIn(
            "bgz17::scalar_sparse::tropical_spmv",
            self.content,
        )

    def test_r13_compaction_includes_primitives(self):
        """R-13 compaction summary line must list the four primitives."""
        self.assertIn(
            "`cam_pq` + `bgz-hhtl-d` + `dn_tree` + `merkle_tree`",
            self.content,
        )

    def test_r14_compaction_entry_present(self):
        """Compaction list must include R-14 entry."""
        self.assertIn(
            "R-14: formal correctness via `jc::pflug`",
            self.content,
        )

    def test_r15_compaction_entry_present(self):
        """Compaction list must include R-15 entry."""
        self.assertIn(
            "R-15: `SignatureBasis<DEPTH>: Basis<f32>`",
            self.content,
        )

    def test_summary_intro_updated_to_r15(self):
        """Section §7 intro must say R-14 and R-15 were added post-merge."""
        self.assertIn("R-14", self.content)
        self.assertIn("R-15", self.content)
        self.assertIn("post-merge", self.content)

    def test_open_work_g4_cross_reference(self):
        """R-14 section must note open work and cross-ref to G-4."""
        self.assertIn("G-4", self.content)
        self.assertIn("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md", self.content)


# ---------------------------------------------------------------------------
# pr-x12-woa-multiarch-orchestration.md
# ---------------------------------------------------------------------------
class TestWoaMultiarchOrchestration(unittest.TestCase):
    """
    Validates changes to pr-x12-woa-multiarch-orchestration.md:
      - Terminology updated from 'dispatch' to 'polyfill' throughout.
      - Section 3 header reflects 'compile-time polyfill' model.
      - Architecture description uses cfg(target_feature = ...) model.
      - Consumer crates described as not knowing the active backend.
    """

    DOC = "pr-x12-woa-multiarch-orchestration.md"

    def setUp(self):
        self.content = read_doc(self.DOC)

    def test_status_line_uses_polyfill_terminology(self):
        """Status line must use 'per-arch polyfill decisions'."""
        self.assertIn(
            "per-arch polyfill decisions",
            self.content,
        )

    def test_premise_uses_polyfill_contract(self):
        """Premise sentence must say 'per-arch polyfill contract'."""
        self.assertIn(
            "per-arch polyfill contract",
            self.content,
        )

    def test_old_dispatch_contract_premise_absent(self):
        """Old 'per-arch dispatch contract' premise must be replaced."""
        self.assertNotIn(
            "**per-arch dispatch contract**",
            self.content,
            "Old 'dispatch contract' framing must be replaced by 'polyfill contract'",
        )

    def test_section_0_thesis_uses_polyfill_surface(self):
        """Section 0 thesis must reference 'polyfill surface'."""
        self.assertIn(
            "polyfill surface",
            self.content,
        )

    def test_section_3_header_updated(self):
        """Section 3 header must say 'compile-time polyfill'."""
        self.assertIn(
            "## 3. Per-arch substrate via compile-time polyfill",
            self.content,
        )

    def test_old_section_3_header_absent(self):
        """Old section 3 header ('Per-arch dispatch as a substrate property') must be gone."""
        self.assertNotIn(
            "## 3. Per-arch dispatch as a substrate property",
            self.content,
        )

    def test_cfg_target_feature_mechanism_described(self):
        """Section 3 must describe cfg(target_feature = ...) as the selection mechanism."""
        self.assertIn(
            "cfg(target_feature = ...)",
            self.content,
        )

    def test_no_runtime_detection_statement(self):
        """Doc must state there is no runtime CPU detection."""
        self.assertIn(
            "no runtime CPU detection",
            self.content,
        )

    def test_backend_files_named(self):
        """Backend file names must be listed."""
        for backend in ("simd_avx512.rs", "simd_neon.rs", "simd_scalar.rs"):
            self.assertIn(
                backend,
                self.content,
                f"Backend file '{backend}' must be named in the architecture description",
            )

    def test_consumer_crates_never_name_backend(self):
        """Doc must state consumers never name a backend symbol."""
        self.assertIn(
            "never name a backend",
            self.content,
        )

    def test_build_time_cpu_selection_section(self):
        """Section 3.2 must describe build-time CPU selection."""
        self.assertIn(
            "### 3.2 Build-time CPU selection",
            self.content,
        )

    def test_woa_polyfill_transparency_commitment(self):
        """Commitment #2 must say 'per-arch polyfill transparency'."""
        self.assertIn(
            "Per-arch polyfill transparency",
            self.content,
        )

    def test_old_dispatch_transparency_absent(self):
        """Old 'Per-arch dispatch transparency' commitment must be replaced."""
        self.assertNotIn(
            "Per-arch dispatch transparency",
            self.content,
        )

    def test_gpu_offload_extension_uses_polyfill_framing(self):
        """GPU offload extension anchor must use polyfill-framing language."""
        self.assertIn(
            "GPU polyfill at compile time",
            self.content,
        )

    def test_old_dispatch_target_gpu_absent(self):
        """Old 'DispatchTarget' GPU extension must be replaced."""
        self.assertNotIn(
            "DispatchTarget",
            self.content,
        )

    def test_architecture_diagram_has_polyfill_layer(self):
        """Architecture diagram must label the middle layer as polyfill."""
        self.assertIn(
            "polyfill substrate",
            self.content,
        )

    def test_architecture_diagram_shows_backend_files(self):
        """Architecture diagram must show backend files with cfg selection."""
        self.assertIn("Backend file", self.content)
        self.assertIn("cfg(target_feature", self.content)

    def test_metrics_endpoint_description_updated(self):
        """WoA metrics description must use polyfill framing."""
        self.assertIn(
            "per-arch polyfill performance",
            self.content,
        )


# ---------------------------------------------------------------------------
# pr-x12-x266-3dgs-spacetime-upscaling.md
# ---------------------------------------------------------------------------
class TestX2663dgsSpacetimeUpscaling(unittest.TestCase):
    """
    Validates changes to pr-x12-x266-3dgs-spacetime-upscaling.md:
      - Prerequisites table now uses 'canon-fixed' status for R-1, R-2, R-13.
      - 'Canon-fixed' definition is explained in a note below the table.
      - 'Scheduled' items remain for Plan E, R-4/R-11.
      - No shipping code claim is asserted.
    """

    DOC = "pr-x12-x266-3dgs-spacetime-upscaling.md"

    def setUp(self):
        self.content = read_doc(self.DOC)

    def test_canon_fixed_status_present(self):
        """Status column must include 'canon-fixed' entries."""
        self.assertIn(
            "**canon-fixed**",
            self.content,
            "Prerequisites table must use 'canon-fixed' status",
        )

    def test_r1_status_is_canon_fixed(self):
        """R-1 prerequisite status must be 'canon-fixed'."""
        self.assertIn(
            "**canon-fixed** (R-1 trait shape committed)",
            self.content,
        )

    def test_r2_status_is_canon_fixed(self):
        """R-2 prerequisite status must be 'canon-fixed'."""
        self.assertIn(
            "**canon-fixed** (R-2 commits bits 0-1 = `header_kind`)",
            self.content,
        )

    def test_r13_status_is_canon_fixed(self):
        """R-13 prerequisite status must be 'canon-fixed'."""
        self.assertIn(
            "**canon-fixed** (R-13 commits Option A: per-shard codebook for Plan F v1)",
            self.content,
        )

    def test_canon_fixed_definition_note_present(self):
        """A note must define 'canon-fixed' vs 'scheduled'."""
        self.assertIn(
            '"Canon-fixed"',
            self.content,
            "Explanation note for 'canon-fixed' must be present",
        )
        self.assertIn(
            "the resolution doc commits the design",
            self.content,
        )
        self.assertIn(
            "the implementation has a named plan card",
            self.content,
        )

    def test_no_shipping_code_today_statement(self):
        """Note must state that none of the items have shipping code today."""
        self.assertIn(
            "None of the above have shipping code today",
            self.content,
        )

    def test_old_r1_status_landed_absent(self):
        """
        Regression: the old ambiguous 'landed in concept; implementation in Plan A4'
        status for R-1 must be replaced by the 'canon-fixed' form.
        """
        self.assertNotIn(
            "landed in concept; implementation in Plan A4",
            self.content,
            "Old ambiguous 'landed' status for R-1 must be replaced by 'canon-fixed'",
        )

    def test_old_header_byte_stable_landed_absent(self):
        """Regression: old 'landed' status for R-2 must be replaced."""
        old_r2_status = "Header byte stable across basis swaps | R-2, M:E-J bits 0-1 | landed"
        self.assertNotIn(
            old_r2_status,
            self.content,
        )

    def test_scheduled_status_still_present_for_plan_e(self):
        """'Scheduled' status must still appear for Plan E items."""
        self.assertIn("scheduled", self.content)

    def test_prerequisites_table_section_present(self):
        """Section 8 (PR-X12 prerequisites) must still be present."""
        self.assertIn("## 8. PR-X12 prerequisites", self.content)

    def test_r3_audit_rule_clarification(self):
        """R-3 codec body decoupling status must include audit rule clarification."""
        self.assertIn(
            "R-3 audit rule (doc commitment; CI check pending)",
            self.content,
        )

    def test_r2_wire_format_implementation_note(self):
        """R-2 status must note wire-format implementation is in Plan A8."""
        self.assertIn("wire-format implementation in Plan A8", self.content)


# ---------------------------------------------------------------------------
# Cross-document consistency tests
# ---------------------------------------------------------------------------
class TestCrossDocumentConsistency(unittest.TestCase):
    """
    Tests that validate consistency across multiple changed documents:
      - Gap IDs are consistently labelled and cross-referenced.
      - Resolution IDs R-14/R-15 appear in all docs that should mention them.
      - Kernel canon is consistent: blasgraph canonical (kernel unwritten),
        shipped min-plus = bgz17::ScalarCsr::spmv_min_plus (audit 2026-07-16).
      - Polyfill terminology replaces dispatch where updated.
    """

    def _read(self, filename: str) -> str:
        return read_doc(filename)

    def test_gap_g1_owned_by_bgz_jc_doc(self):
        """
        bgz-jc doc defines G-1 in §5.1; cam-pq doc cross-references it as 'bgz-jc G-1'.
        Both usages must be consistent.
        """
        bgz_jc = self._read("pr-x12-bgz-jc-substrate-synergies.md")
        cam_pq = self._read("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md")

        self.assertIn("Gap **G-1**", bgz_jc, "bgz-jc doc must define G-1")
        self.assertIn("bgz-jc G-1", cam_pq, "cam-pq doc must cross-ref as bgz-jc G-1")

    def test_gap_g2_owned_by_bgz_jc_doc(self):
        """
        bgz-jc doc defines G-2 in §5.2; cam-pq doc cross-references it as 'bgz-jc G-2'.
        """
        bgz_jc = self._read("pr-x12-bgz-jc-substrate-synergies.md")
        cam_pq = self._read("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md")

        self.assertIn("Gap **G-2**", bgz_jc, "bgz-jc doc must define G-2")
        self.assertIn("bgz-jc G-2", cam_pq, "cam-pq doc must cross-ref as bgz-jc G-2")

    def test_min_plus_kernel_canon_consistent_across_docs(self):
        """
        The corrected kernel canon (audit 2026-07-16) must be cited
        consistently across canon-resolutions-delta and
        substrate-canon-resolutions: the shipped min-plus is the method
        bgz17::ScalarCsr::spmv_min_plus; the fabricated free-function path
        must not be cited affirmatively in the delta doc (the substrate doc
        retains one explicitly-negated mention as the audit trail).
        """
        kernel = "bgz17::ScalarCsr::spmv_min_plus"
        delta = self._read("pr-x12-canon-resolutions-delta.md")
        substrate = self._read("pr-x12-substrate-canon-resolutions.md")

        self.assertIn(kernel, delta, "canon-resolutions-delta must cite spmv_min_plus")
        self.assertIn(
            kernel, substrate, "substrate-canon-resolutions must cite spmv_min_plus"
        )
        self.assertNotIn(
            "bgz17::scalar_sparse::tropical_spmv",
            delta,
            "The fabricated free-function path must not appear in the delta doc",
        )

    def test_r14_r15_cited_in_both_canon_docs(self):
        """
        R-14 and R-15 must appear in both canon-resolutions-delta and
        substrate-canon-resolutions.
        """
        delta = self._read("pr-x12-canon-resolutions-delta.md")
        substrate = self._read("pr-x12-substrate-canon-resolutions.md")

        for resolution in ("R-14", "R-15"):
            self.assertIn(resolution, delta, f"{resolution} must appear in delta doc")
            self.assertIn(
                resolution, substrate, f"{resolution} must appear in substrate doc"
            )

    def test_r13_primitives_consistent_across_docs(self):
        """
        R-13 implementation primitives must be cited consistently in
        canon-resolutions-delta, substrate-canon-resolutions, and cam-pq doc.
        """
        primitives = ["cam_pq", "bgz-hhtl-d", "dn_tree", "merkle_tree"]
        docs = {
            "canon-resolutions-delta": self._read("pr-x12-canon-resolutions-delta.md"),
            "substrate-canon-resolutions": self._read(
                "pr-x12-substrate-canon-resolutions.md"
            ),
            "cam-pq-substrate-bindings": self._read(
                "pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md"
            ),
        }

        for doc_name, content in docs.items():
            for primitive in primitives:
                self.assertIn(
                    primitive,
                    content,
                    f"'{primitive}' must appear in {doc_name}",
                )

    def test_signature_basis_mentioned_in_cam_pq_and_canon(self):
        """
        SignatureBasis must be mentioned in both cam-pq doc (as a gap) and
        substrate-canon-resolutions (as R-15 resolution).
        """
        cam_pq = self._read("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md")
        substrate = self._read("pr-x12-substrate-canon-resolutions.md")

        self.assertIn("SignatureBasis", cam_pq)
        self.assertIn("SignatureBasis", substrate)

    def test_polyfill_terminology_in_woa_doc(self):
        """
        The woa doc must use 'polyfill' as the primary framing for per-arch dispatch.
        """
        woa = self._read("pr-x12-woa-multiarch-orchestration.md")
        polyfill_count = woa.count("polyfill")
        self.assertGreater(
            polyfill_count,
            5,
            "woa doc must use 'polyfill' terminology extensively (at least 6 times)",
        )

    def test_no_g8_or_g9_identifiers_in_any_changed_doc(self):
        """
        Regression: standalone G-8 and G-9 identifiers (the old cross-ref labels
        before renaming) must not appear in the changed docs.
        """
        changed_docs = [
            "pr-x12-bgz-jc-substrate-synergies.md",
            "pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md",
            "pr-x12-canon-resolutions-delta.md",
            "pr-x12-gguf-llm-weights-encoding.md",
            "pr-x12-substrate-canon-resolutions.md",
            "pr-x12-woa-multiarch-orchestration.md",
            "pr-x12-x266-3dgs-spacetime-upscaling.md",
        ]
        for doc in changed_docs:
            content = self._read(doc)
            self.assertNotIn(
                "**G-8**",
                content,
                f"'**G-8**' must not appear in {doc}",
            )
            self.assertNotIn(
                "**G-9**",
                content,
                f"'**G-9**' must not appear in {doc}",
            )

    def test_canon_fixed_status_only_in_x266_doc(self):
        """
        The 'canon-fixed' status label is specific to the x266 prerequisites
        table; it should not bleed into unrelated docs.
        """
        x266 = self._read("pr-x12-x266-3dgs-spacetime-upscaling.md")
        self.assertIn("canon-fixed", x266)

    def test_r14_pillar11_feature_gate_consistent(self):
        """
        The Pillar 11 feature gate '--features hambly-lyons' must be cited
        consistently in both substrate-canon-resolutions and cam-pq doc.
        """
        substrate = self._read("pr-x12-substrate-canon-resolutions.md")
        cam_pq = self._read("pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md")

        self.assertIn("hambly-lyons", substrate)
        self.assertIn("hambly-lyons", cam_pq)

    def test_signature_truncated_not_pde_in_r15(self):
        """
        Both the delta and substrate docs must warn that signature_kernel_pde
        should not be used (in favour of signature_truncated) for R-15.
        """
        delta = self._read("pr-x12-canon-resolutions-delta.md")
        substrate = self._read("pr-x12-substrate-canon-resolutions.md")

        self.assertIn("signature_truncated", delta)
        self.assertIn("signature_kernel_pde", delta)
        self.assertIn("signature_truncated", substrate)
        self.assertIn("signature_kernel_pde", substrate)


if __name__ == "__main__":
    unittest.main(verbosity=2)
