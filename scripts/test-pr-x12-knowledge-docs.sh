#!/bin/sh
# Tests for PR-X12 knowledge document invariants.
#
# Validates the content changes introduced in the PR that updates gap labeling,
# cross-references, terminology (dispatch → polyfill), R-14/R-15 additions,
# and other invariants across the seven .claude/knowledge/ files.
#
# Exit code: 0 if all assertions pass, 1 on first failure.

set -e

KNOWLEDGE_DIR=".claude/knowledge"
PASS=0
FAIL=0

pass() {
    echo "  PASS: $1"
    PASS=$((PASS + 1))
}

fail() {
    echo "  FAIL: $1"
    FAIL=$((FAIL + 1))
}

assert_contains() {
    file="$1"
    pattern="$2"
    desc="$3"
    if grep -qE "$pattern" "$KNOWLEDGE_DIR/$file"; then
        pass "$desc"
    else
        fail "$desc"
        echo "       Pattern not found: $pattern"
        echo "       File: $KNOWLEDGE_DIR/$file"
    fi
}

assert_not_contains() {
    file="$1"
    pattern="$2"
    desc="$3"
    if grep -qE "$pattern" "$KNOWLEDGE_DIR/$file"; then
        fail "$desc"
        echo "       Pattern should NOT be present: $pattern"
        echo "       File: $KNOWLEDGE_DIR/$file"
    else
        pass "$desc"
    fi
}

assert_count() {
    file="$1"
    pattern="$2"
    expected="$3"
    desc="$4"
    actual=$(grep -cE "$pattern" "$KNOWLEDGE_DIR/$file" || true)
    if [ "$actual" -eq "$expected" ]; then
        pass "$desc"
    else
        fail "$desc"
        echo "       Expected count $expected, got $actual for: $pattern"
        echo "       File: $KNOWLEDGE_DIR/$file"
    fi
}

# ---------------------------------------------------------------------------
echo ""
echo "=== 1. pr-x12-bgz-jc-substrate-synergies.md ==="
F="pr-x12-bgz-jc-substrate-synergies.md"

# §5.1 must carry the Gap G-1 label
assert_contains "$F" 'Gap \*\*G-1\*\*' \
    "§5.1 jd-nd gap is labelled G-1"

# §5.2 must carry the Gap G-2 label
assert_contains "$F" 'Gap \*\*G-2\*\*' \
    "§5.2 Cronbach/ICC gap is labelled G-2"

# Both labels appear in the gaps section (not elsewhere)
assert_contains "$F" '\*\*G-1\*\*' \
    "G-1 bold label is present"

assert_contains "$F" '\*\*G-2\*\*' \
    "G-2 bold label is present"

# The old raw section headers (without gap labels) must not appear
assert_not_contains "$F" '^### 5\.1 `jd-nd` — the missing ndarray-side proof crate$' \
    "§5.1 heading no longer lacks the Gap G-1 label"

assert_not_contains "$F" '^### 5\.2 Cronbach / ICC research crate$' \
    "§5.2 heading no longer lacks the Gap G-2 label"

# ---------------------------------------------------------------------------
echo ""
echo "=== 2. pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md ==="
F="pr-x12-cam-pq-sigker-dn-tree-substrate-bindings.md"

# New cross-ref table header
assert_contains "$F" 'Gap \(cross-ref\)' \
    "Gap table header is 'Gap (cross-ref)' not 'Gap (prior)'"

# Prefixed IDs in cross-ref table
assert_contains "$F" '\*\*bgz-jc G-1\*\*' \
    "bgz-jc G-1 cross-reference present"

assert_contains "$F" '\*\*bgz-jc G-2\*\*' \
    "bgz-jc G-2 cross-reference present"

# Old G-8 / G-9 labels must not appear in the gap table
assert_not_contains "$F" '\*\*G-8\*\*' \
    "Stale G-8 label removed from gap cross-reference table"

assert_not_contains "$F" '\*\*G-9\*\*' \
    "Stale G-9 label removed from gap cross-reference table"

# Namespace-separation explanation paragraph
assert_contains "$F" 'G-1\.\.G-7 IDs in.*this.*doc are local' \
    "Namespace-separation explanation is present"

assert_contains "$F" "bgz-jc's G-1 / G-2 are a separate namespace" \
    "Separate-namespace clarification sentence present"

# Ownership attribution sentence
assert_contains "$F" 'canonical IDs are owned by.*pr-x12-bgz-jc-substrate-synergies' \
    "Ownership attribution references correct source doc"

# ---------------------------------------------------------------------------
echo ""
echo "=== 3. pr-x12-canon-resolutions-delta.md ==="
F="pr-x12-canon-resolutions-delta.md"

# §0 now lists six categories (was five)
assert_contains "$F" 'Six categories of novel content' \
    "§0 correctly claims six categories (was five)"

# R-7 cites the concrete kernel symbol
assert_contains "$F" 'bgz17::scalar_sparse::tropical_spmv' \
    "R-7 cites bgz17::scalar_sparse::tropical_spmv kernel"

# R-13 primitives listed in §0 categories
assert_contains "$F" 'cam_pq.*bgz-hhtl-d.*dn_tree.*merkle_tree' \
    "R-13 primitives listed in category 5"

# R-14 and R-15 appear as sixth category
assert_contains "$F" 'R-14.*jc::pflug.*Pillar 10.*jc::hambly_lyons.*Pillar 11' \
    "R-14 references jc::pflug and jc::hambly_lyons in §0"

assert_contains "$F" 'R-15.*SignatureBasis' \
    "R-15 references SignatureBasis in §0"

# Falsifiability matrix claims 24+3 rows
assert_contains "$F" '24\+3 rows including R-14/R-15' \
    "Falsifiability matrix row count updated to 24+3"

# New sections for R-14 and R-15 exist
assert_contains "$F" '## 11\. Formal-correctness layer \(R-14\)' \
    "Section 11 is 'Formal-correctness layer (R-14)'"

assert_contains "$F" '## 12\. Stream-signal codec lane \(R-15\)' \
    "Section 12 is 'Stream-signal codec lane (R-15)'"

# Load-bearing paragraph is now §13 (was §11)
assert_contains "$F" '## 13\. The single load-bearing paragraph' \
    "Load-bearing paragraph is now §13"

assert_not_contains "$F" '## 11\. The single load-bearing paragraph' \
    "Old §11 'load-bearing paragraph' heading is gone"

# jc pillar details in R-14 section
assert_contains "$F" 'jc::pflug' \
    "jc::pflug (Pillar 10) cited in R-14 section"

assert_contains "$F" 'jc::hambly_lyons' \
    "jc::hambly_lyons (Pillar 11) cited in R-14 section"

# SignatureBasis in R-15 section
assert_contains "$F" 'SignatureBasis<const DEPTH: usize>' \
    "SignatureBasis<const DEPTH> trait signature present"

# Load-bearing paragraph updated to mention R-14, R-15
assert_contains "$F" 'R-14, R-15.*adds a formal-correctness layer' \
    "Load-bearing paragraph mentions R-14/R-15 additions"

# R-13 implementation primitives table in R-13 expansion section
assert_contains "$F" 'CodebookHandle' \
    "CodebookHandle trait mentioned for R-13 expansion"

# Actual kernel home annotation for R-7
assert_contains "$F" 'Actual kernel home \(current\)' \
    "'Actual kernel home' annotation present for R-7"

# ---------------------------------------------------------------------------
echo ""
echo "=== 4. pr-x12-gguf-llm-weights-encoding.md ==="
F="pr-x12-gguf-llm-weights-encoding.md"

# Memory savings heading split into weights-only scope
assert_contains "$F" 'Memory savings \(weights only\):' \
    "Memory savings heading scoped to 'weights only'"

# KV cache caveat paragraph added
assert_contains "$F" 'Phone-class caveat.*weights are not the only memory load' \
    "Phone-class caveat paragraph present"

assert_contains "$F" 'KV cache.*scales with context length' \
    "KV cache context-length explanation present"

assert_contains "$F" 'KV cache lane \(Plan D, M:H-3, R-4\)' \
    "KV cache lane cites correct plan references"

# Escape path now references F-4 falsifier and rANS bypass channel
assert_contains "$F" 'falsifier.*F-4.*wire-format mechanism' \
    "Escape path references F-4 falsifier"

assert_contains "$F" 'rANS bypass channel in the A8 framing layer' \
    "rANS bypass channel in A8 framing cited"

# EncodingDomain::LLMWeights now described as enum-discriminant slot reservation
assert_contains "$F" 'enum-discriminant slot.*EncodingDomain::LLMWeights' \
    "EncodingDomain::LLMWeights described as enum-discriminant slot"

assert_contains "$F" 'forward-compatibility-locked' \
    "Forward-compatibility-locking claim present"

# ---------------------------------------------------------------------------
echo ""
echo "=== 5. pr-x12-substrate-canon-resolutions.md ==="
F="pr-x12-substrate-canon-resolutions.md"

# R-14 and R-15 added to §7 commitment count (five, not three)
assert_contains "$F" 'five commitments missing from both originals' \
    "§7 now claims five commitments (R-11..R-15)"

# Citation IDs updated to R-1 through R-15
assert_contains "$F" 'R-1.*through.*R-15' \
    "Citation IDs cover R-1 through R-15"

assert_contains "$F" 'fifteen resolutions' \
    "Summary calls out fifteen resolutions (was thirteen)"

# R-7 actual kernel home annotation
assert_contains "$F" 'bgz17::scalar_sparse::tropical_spmv' \
    "R-7 section cites bgz17::scalar_sparse::tropical_spmv"

assert_contains "$F" 'Actual kernel home \(current\)' \
    "'Actual kernel home' annotation present in R-7 section"

# R-13 implementation primitives table
assert_contains "$F" 'Implementation primitives.*no new code required' \
    "R-13 implementation primitives table header present"

assert_contains "$F" 'cam_pq::CamCodebook' \
    "CamCodebook listed in R-13 primitives table"

assert_contains "$F" 'ndarray::hpc::dn_tree' \
    "dn_tree listed in R-13 primitives table"

assert_contains "$F" 'ndarray::hpc::merkle_tree' \
    "merkle_tree listed in R-13 primitives table"

# R-14 section
assert_contains "$F" '### R-14' \
    "R-14 resolution section exists"

assert_contains "$F" 'jc::pflug' \
    "Pillar 10 / jc::pflug cited in R-14"

assert_contains "$F" 'jc::hambly_lyons' \
    "Pillar 11 / jc::hambly_lyons cited in R-14"

assert_contains "$F" '\-\-features hambly-lyons' \
    "Pillar 11 feature-gate mentioned"

# R-15 section
assert_contains "$F" '### R-15' \
    "R-15 resolution section exists"

assert_contains "$F" 'SignatureBasis<const DEPTH: usize>: Basis<f32>' \
    "SignatureBasis trait signature present in R-15"

assert_contains "$F" 'signature_truncated' \
    "signature_truncated (not PDE form) cited for R-15"

# R-14 / R-15 rows in falsifiability matrix
assert_contains "$F" 'R-14 \(Pillar 10 active\)' \
    "R-14 Pillar 10 row in falsifiability matrix"

assert_contains "$F" 'R-14 \(Pillar 11 active\)' \
    "R-14 Pillar 11 row in falsifiability matrix"

assert_contains "$F" 'R-15 \(SignatureBasis lane\)' \
    "R-15 SignatureBasis row in falsifiability matrix"

# Cargo test commands for pillars cited in matrix
assert_contains "$F" 'cargo test -p jc.*--features hambly-lyons' \
    "Pillar 11 cargo test command present in falsifiability matrix"

# R-7 summary bullet updated with kernel location
assert_contains "$F" 'bgz17::scalar_sparse::tropical_spmv' \
    "R-7 summary bullet cites tropical_spmv symbol"

# R-13 summary bullet updated with primitive list
assert_contains "$F" 'cam_pq.*bgz-hhtl-d.*dn_tree.*merkle_tree' \
    "R-13 summary bullet lists all four primitives"

# R-14 summary bullet
assert_contains "$F" 'jc::pflug.*Pillar 10' \
    "R-14 summary bullet references jc::pflug (Pillar 10)"
assert_contains "$F" 'jc::hambly_lyons.*Pillar 11' \
    "R-14 summary bullet references jc::hambly_lyons (Pillar 11)"

# R-15 summary bullet
assert_contains "$F" 'SignatureBasis.*fifth Plan G lane' \
    "R-15 summary bullet describes stream-signal lane"

# Append-only numbering note updated
assert_contains "$F" 'R-14, R-15' \
    "Append-only numbering note lists R-14 and R-15"
assert_contains "$F" 'appended post-merge from the substrate-binding doc' \
    "Append-only numbering note explains R-14/R-15 are post-merge additions"

# ---------------------------------------------------------------------------
echo ""
echo "=== 6. pr-x12-woa-multiarch-orchestration.md ==="
F="pr-x12-woa-multiarch-orchestration.md"

# Premise line uses "polyfill" not "dispatch"
assert_contains "$F" 'per-arch polyfill decisions' \
    "Premise uses 'per-arch polyfill decisions'"

assert_contains "$F" 'per-arch polyfill contract' \
    "Premise line uses 'per-arch polyfill contract'"

# Thesis paragraph uses polyfill terminology
assert_contains "$F" 'polyfill surface' \
    "Thesis uses 'polyfill surface'"

assert_contains "$F" 'cfg\(target_feature = \.\.\.\)' \
    "Thesis cites cfg(target_feature = ...) compile-time selection"

# Section 3 header is polyfill, not dispatch
assert_contains "$F" '## 3\. Per-arch substrate via compile-time polyfill' \
    "Section 3 header uses polyfill terminology"

# W1a contract reference
assert_contains "$F" 'W1a consumer contract' \
    "W1a polyfill contract cited"

# Backend files described (not runtime detection)
assert_contains "$F" 'simd_avx512\.rs.*simd_neon\.rs.*simd_scalar\.rs' \
    "Three backend files listed in §3"

# No runtime CPU detection claim
assert_contains "$F" 'no runtime CPU detection' \
    "Explicit 'no runtime CPU detection' statement present"

# backend_target() replaces dispatch_target()
assert_contains "$F" 'backend_target\(\)' \
    "backend_target() used (not dispatch_target)"

assert_not_contains "$F" 'dispatch_target\(\)' \
    "Old dispatch_target() name removed"

# Per-arch polyfill transparency in commitment list
assert_contains "$F" 'Per-arch polyfill transparency' \
    "Commitment uses 'Per-arch polyfill transparency'"

assert_not_contains "$F" 'Per-arch dispatch transparency' \
    "Stale 'Per-arch dispatch transparency' label removed"

# Consumer crates described with polyfill framing
assert_contains "$F" 'same Rust, per-arch binary' \
    "Consumer crates described as 'same Rust, per-arch binary'"

# Build-time CPU selection section
assert_contains "$F" '### 3\.2 Build-time CPU selection \(not runtime detection\)' \
    "Section 3.2 header specifies build-time selection"

# Crossover section updated
assert_contains "$F" '### 3\.3 Per-arch tunable crossover \(R-5\)' \
    "Section 3.3 is Per-arch tunable crossover"

# Escape hatch documented
assert_contains "$F" 'Escape hatch \(rare\)' \
    "Escape hatch for rare backend-specific code documented"

# Fleet ships per-arch binaries statement
assert_contains "$F" 'per-arch binaries.*not a fat binary' \
    "Fleet ships per-arch binaries (not fat binary) statement"

# ---------------------------------------------------------------------------
echo ""
echo "=== 7. pr-x12-x266-3dgs-spacetime-upscaling.md ==="
F="pr-x12-x266-3dgs-spacetime-upscaling.md"

# canon-fixed appears in status column (at least 3 occurrences for R-1, R-2, R-13)
assert_count "$F" 'canon-fixed' 3 \
    "canon-fixed appears exactly 3 times (R-1, R-2, R-13 rows)"

# Definition of "Canon-fixed" vs "scheduled"
assert_contains "$F" '"Canon-fixed".*=.*resolution doc commits the design' \
    "Canon-fixed definition distinguishes from scheduled"

assert_contains "$F" '"scheduled".*=.*implementation has a named plan card' \
    "Scheduled definition provided"

# None of the above have shipping code today
assert_contains "$F" 'None of the above have shipping code today' \
    "Caveat that no row has shipping code present"

# R-1 row status uses canon-fixed framing
assert_contains "$F" 'canon-fixed.*R-1 trait shape committed' \
    "R-1 row: trait shape commit attributed to canon"

# R-2 row status uses canon-fixed framing
assert_contains "$F" 'canon-fixed.*R-2 commits bits 0-1' \
    "R-2 row: bit field commit attributed to canon"

# R-3 row uses new audit rule wording
assert_contains "$F" 'R-3 audit rule.*doc commitment.*CI check pending' \
    "R-3 row clarifies doc commitment vs CI check"

# R-13 row uses canon-fixed framing
assert_contains "$F" 'canon-fixed.*R-13 commits Option A' \
    "R-13 row: Option A commit attributed to canon"

# ---------------------------------------------------------------------------
echo ""
echo "=== Summary ==="
echo "  Passed: $PASS"
echo "  Failed: $FAIL"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "RESULT: FAIL ($FAIL assertion(s) failed)"
    exit 1
else
    echo "RESULT: PASS (all $PASS assertions)"
    exit 0
fi
