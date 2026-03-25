# SESSION: DeepNSM-CAM — Semantic Transformer for the Thinking Pipeline

## Mission

Build a semantic processing layer that gives the lance-graph thinking pipeline
the ability to understand MEANING and GRAMMAR of natural language — without
transformers, without GPU, without regex. 4,096 words × 12 bits. Exact.
Deterministic. Bit-reproducible. O(1) per word, O(n) per sentence.

This module sits between raw text and the cognitive verbs (EXPLORE, HYPOTHESIS,
SYNTHESIS, etc.) in `thinking/graph.rs`. When the planner reasons about queries,
it reasons about MEANING — not string patterns.

## Architecture

```
Raw text: "the big dog bit the old man"
     │
     ▼ nsm_tokenizer.rs
Token stream: [(1,a), (156,j), (671,n), (2943,v), (1,a), (174,j), (95,n)]
     │         rank,PoS  — 12-bit index + 4-bit PoS = 16 bits per token
     ▼ nsm_parser.rs
SPO triples: [SPO(dog, bite, man), Mod(big→dog), Mod(old→man)]
     │         PoS-driven FSM, not regex. 6 states. O(n).
     ▼ nsm_encoder.rs
SpoBase17:   XOR-bind with role vectors → bundle → 102 bytes per triple
     │         Fibonacci-spaced planes. Euler gamma rotation. Exact.
     ▼ nsm_similarity.rs
Calibrated:  SimilarityTable from exact 4096² distribution → f32 [0,1]
     │         Per-plane: subject/predicate/object similarity decomposed.
     ▼ thinking/graph.rs
Cognitive:   ThinkingGraph verbs use SPO similarity to reason about meaning.
             SYNTHESIS merges triples if similarity > 0.85.
             COUNTERFACTUAL negates the predicate plane.
             INTERRELATE finds cross-domain bridges via subject similarity.
```

## READ FIRST

```bash
# The data (already in repo)
cat word_frequency/README.md                              # data docs
cat word_frequency/nsm_primes.json                        # 63 NSM primes → CAM codes
head -20 word_frequency/word_rank_lookup.csv              # rank,word,pos,freq
head -20 word_frequency/word_forms.csv                    # lemma → surface forms

# The existing infrastructure
cat crates/lance-graph/src/cam_pq/mod.rs                  # CAM-PQ wiring
cat crates/lance-graph/src/cam_pq/storage.rs              # Arrow schema for CAM
cat crates/lance-graph/src/cam_pq/udf.rs                  # DataFusion UDF
cat crates/bgz17/src/base17.rs                            # Base17 L1 distance
cat crates/bgz17/src/distance_matrix.rs                   # precomputed distance matrix
cat crates/bgz17/src/bridge.rs                            # Bgz17Distance trait

# The ndarray codec
cat src/hpc/cam_pq.rs                                     # CamCodebook, DistanceTables, PackedDatabase

# The planner thinking pipeline
cat crates/lance-graph-planner/src/thinking/graph.rs      # ThinkingGraph — 36 styles as AdjacencyStore
cat crates/lance-graph-planner/src/thinking/process.rs    # CognitiveVerb, CognitiveProcess, programs
cat crates/lance-graph-planner/src/thinking/mod.rs        # orchestrate_with_topology()

# The contract types
cat crates/lance-graph-contract/src/thinking.rs           # 36 ThinkingStyle enum, FieldModulation
```

## DATA MODEL

### The 4,096-Word Vocabulary

12-bit index. Every word in the vocabulary has:
- `rank: u12` — position in COCA frequency list (1 = "the", 4096 = "journalism")
- `lemma: &str` — canonical form
- `pos: PoS` — part of speech (13 tags: n, v, j, r, i, p, c, d, m, u, a, x, t, e)
- `freq: u32` — raw frequency in 1B-word COCA corpus
- `disp: f32` — dispersion (0-1, how evenly distributed across texts)
- `forms: Vec<(String, u32)>` — inflected forms with frequencies (11,460 total)
- `vector: [f32; 96]` — 96D distributional vector from COCA subgenre frequencies
- `cam: [u8; 6]` — 6-byte CAM-PQ fingerprint (from codebook_pq.bin)

Coverage: 98.4% of running English text. 62/63 NSM semantic primes.
99.0% of top-4K words have dispersion > 0.8 (well-distributed, not bursty).

### The Distance Matrix

4,096 × 4,096 symmetric matrix. Precomputed from exact 96D distributional vectors.

```
Storage options (pick one per deployment):
  u8 palette-quantized:   8 MB   (fits L2 cache, 256 similarity levels)
  BF16 exact:            16 MB   (fits L3 cache, full precision)
  f16 exact:             16 MB   (same as BF16, Arrow Float16 compatible)

Distance(word_a, word_b) = matrix[a.rank][b.rank]  → ONE memory access
```

### The SimilarityTable

Built from the EXACT 4,096² distribution. Not sampled. Not approximated.

```
4,096 × 4,096 / 2 = 8.4M unique pairs
Sort all distances → empirical CDF
SimilarityTable: 256 × f16 = 512 bytes
similarity(distance) = 1.0 - CDF(distance)  → O(1) lookup
```

### SPO Triple Encoding

36 bits per triple: 12 bits subject + 12 bits predicate + 12 bits object.

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpoTriple {
    /// Packed: [S:12][P:12][O:12] = 36 bits, stored in u64 (upper 28 bits zero)
    packed: u64,
}

impl SpoTriple {
    pub fn new(subject: u16, predicate: u16, object: u16) -> Self {
        debug_assert!(subject < 4096 && predicate < 4096 && object < 4096);
        Self {
            packed: ((subject as u64) << 24) | ((predicate as u64) << 12) | object as u64,
        }
    }
    pub fn subject(&self) -> u16 { ((self.packed >> 24) & 0xFFF) as u16 }
    pub fn predicate(&self) -> u16 { ((self.packed >> 12) & 0xFFF) as u16 }
    pub fn object(&self) -> u16 { (self.packed & 0xFFF) as u16 }
}
```

### SPO Distance (3 matrix lookups)

```rust
impl SpoTriple {
    /// Total distance: sum of per-role distances.
    pub fn distance(&self, other: &SpoTriple, matrix: &DistanceMatrix) -> u32 {
        matrix.get(self.subject(), other.subject())
            + matrix.get(self.predicate(), other.predicate())
            + matrix.get(self.object(), other.object())
    }

    /// Per-role distances: (subject_dist, predicate_dist, object_dist).
    pub fn distance_per_role(&self, other: &SpoTriple, matrix: &DistanceMatrix) -> (u32, u32, u32) {
        (
            matrix.get(self.subject(), other.subject()),
            matrix.get(self.predicate(), other.predicate()),
            matrix.get(self.object(), other.object()),
        )
    }

    /// Per-role similarities via SimilarityTable.
    pub fn similarity_per_role(
        &self, other: &SpoTriple,
        matrix: &DistanceMatrix, table: &SimilarityTable,
    ) -> (f32, f32, f32) {
        let (ds, dp, do_) = self.distance_per_role(other, matrix);
        (table.similarity(ds), table.similarity(dp), table.similarity(do_))
    }
}
```

### VSA Composition (XOR binding + majority bundle)

For sentence-level representations that preserve word order.

```rust
/// Role vectors: fixed pseudo-random binary patterns.
/// Generated once from seed. 10,000 bits each (= Base17 compatible).
pub struct RoleVectors {
    subject:   BitVec,   // 10K bits
    predicate: BitVec,
    object:    BitVec,
    modifier:  BitVec,
    temporal:  BitVec,
    negation:  BitVec,
}

/// Bind a word's distributional vector with a role.
/// word_bits XOR role_bits → bound representation.
/// The role is recoverable: bound XOR role = word.
pub fn bind(word: &BitVec, role: &BitVec) -> BitVec {
    word.xor(role)
}

/// Bundle multiple bindings via majority vote.
/// Preserves all components in superposition.
/// Similarity to any component ≈ 0.75 (vs 0.50 random baseline).
pub fn bundle(bindings: &[BitVec]) -> BitVec {
    majority_vote(bindings)
}

/// Unbind: recover a role's content from a bundled representation.
/// bundle XOR role → approximate word (sim ≈ 0.75 to original).
pub fn unbind(bundled: &BitVec, role: &BitVec) -> BitVec {
    bundled.xor(role)
}
```

## DELIVERABLE 1: nsm_tokenizer.rs

Tokenize text to 12-bit word indices. No regex. Hash lookup + word_forms table.

```rust
/// The vocabulary: 4,096 entries loaded from word_rank_lookup.csv.
pub struct Vocabulary {
    /// word string → (rank, PoS, frequency)
    lookup: HashMap<String, WordEntry>,
    /// rank → word string (reverse lookup for unbinding)
    reverse: Vec<String>,  // [4096]
    /// Inflected form → lemma rank (from word_forms.csv)
    /// "bit" → 2943 (rank of "bite"), "sleeping" → 1842 (rank of "sleep")
    forms: HashMap<String, u16>,
}

pub struct WordEntry {
    pub rank: u16,      // 0-4095
    pub pos: PoS,       // 13 tags
    pub freq: u32,
}

/// Part of speech tags from COCA.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum PoS {
    Noun = 0,        // n
    Verb = 1,        // v
    Adjective = 2,   // j
    Adverb = 3,      // r
    Preposition = 4, // i
    Pronoun = 5,     // p
    Conjunction = 6, // c
    Determiner = 7,  // d (modal/auxiliary in COCA)
    Number = 8,      // m
    Interjection = 9,// u
    Article = 10,    // a
    Negation = 11,   // x
    Particle = 12,   // t
}

pub struct Token {
    pub rank: u16,     // 12-bit word index
    pub pos: PoS,
    pub position: u16, // position in sentence
    pub is_negated: bool,
}

impl Vocabulary {
    /// Load from CSV files in word_frequency/ directory.
    pub fn load(dir: &Path) -> Result<Self, Error>;

    /// Tokenize a single word. O(1) hash lookup.
    /// Tries: exact → lowercase → forms table → OOV (returns None).
    pub fn tokenize_word(&self, word: &str) -> Option<Token>;

    /// Tokenize a sentence. O(n) where n = word count.
    pub fn tokenize(&self, text: &str) -> Vec<Token>;

    /// Reverse: rank → word string.
    pub fn word(&self, rank: u16) -> &str;
}
```

### OOV (Out-of-Vocabulary) Strategy

Words not in top 4,096: resolve to nearest in-vocabulary word via
CAM-PQ distance. Precompute a CLAM bucket assignment for the remaining
~1,000 words in the 5,050 list. For truly unknown words, return the
NSM decomposition: "journalism" → (news + write), "cardiology" → (heart + doctor).

```rust
pub struct OovResolver {
    /// Words 4097-5050 → nearest rank in 0-4095.
    extended: HashMap<String, u16>,
    /// Compound decomposition for extended words.
    decomposition: HashMap<String, Vec<u16>>,
}
```

## DELIVERABLE 2: nsm_parser.rs

Extract SPO triples from token stream. PoS-driven FSM, not regex.

```rust
/// Parser state machine: 6 states, PoS-driven transitions.
///
///   START → DET? → ADJ* → NOUN+ → VERB → DET? → ADJ* → NOUN+
///           ──────NP(subj)──────   ─VP─   ────────NP(obj)────────
///
/// This handles ~85% of English sentences (SVO order).
/// Complex cases (passive, relative clauses) are handled by
/// secondary patterns triggered by specific PoS sequences.
#[derive(Clone, Copy)]
enum ParseState {
    Start,
    SubjectNP,    // collecting subject noun phrase
    Verb,         // expecting verb
    ObjectNP,     // collecting object noun phrase
    Modifier,     // adjective/adverb attaching to nearest noun
    Complete,     // triple extracted
}

/// Extracted semantic structure from a sentence.
pub struct SentenceStructure {
    /// Primary SPO triples (usually 1-3 per sentence).
    pub triples: Vec<SpoTriple>,
    /// Modifier attachments: (modifier_rank, head_rank, relation).
    pub modifiers: Vec<(u16, u16, ModRelation)>,
    /// Negation: which triple indices are negated.
    pub negations: Vec<usize>,
    /// Temporal markers: which triple indices have temporal context.
    pub temporals: Vec<(usize, u16)>,  // (triple_idx, temporal_word_rank)
}

#[derive(Clone, Copy)]
pub enum ModRelation {
    AdjectiveOf,   // "big dog" → big modifies dog
    AdverbOf,      // "quickly ran" → quickly modifies ran
    PrepOf,        // "in the house" → prepositional attachment
}

/// Parse token stream into semantic structure.
///
/// Strategy:
/// 1. Identify NP boundaries via PoS patterns (DET? ADJ* NOUN+)
/// 2. Head noun = rightmost noun in NP
/// 3. Verb = first verb after subject NP
/// 4. Object NP = NP after verb
/// 5. Modifiers = adjectives/adverbs → attach to nearest head
/// 6. Negation = "not" before verb → negate the triple
/// 7. Conjunction = "and"/"or" → fork into multiple triples
pub fn parse(tokens: &[Token]) -> SentenceStructure;

/// Secondary patterns for non-SVO structures:
///
/// Passive: "the man was bitten by the dog"
///   PoS: DET NOUN VERB(aux) VERB(past-part) PREP DET NOUN
///   → reverse S and O, resolve "by" as agent marker
///
/// Relative: "the man who bit the dog"
///   PoS: DET NOUN PRONOUN(rel) VERB DET NOUN
///   → extract embedded triple, link via relative pronoun
///
/// Existential: "there is a dog"
///   PoS: EXIST VERB DET NOUN
///   → subject = noun, predicate = "exist", object = None
///
/// These are finite patterns, not recursive grammar.
/// Each pattern is a PoS template match, O(1) per sentence.
pub fn parse_secondary(tokens: &[Token], primary: &SentenceStructure) -> SentenceStructure;
```

## DELIVERABLE 3: nsm_encoder.rs

Encode SPO triples into bgz17-compatible representations for comparison.

```rust
/// The semantic encoder: word indices → comparable representations.
pub struct NsmEncoder {
    /// 4096 × 96D distributional vectors (BF16, 768 KB).
    vectors: Vec<[bf16; 96]>,
    /// Distance matrix: 4096 × 4096 u8 palette-quantized (8 MB).
    distance_matrix: DistanceMatrix,
    /// Similarity calibration from exact corpus distribution (512 B).
    similarity_table: SimilarityTable,
    /// Role vectors for VSA binding (10K bits each, ~7.5 KB total).
    roles: RoleVectors,
    /// NSM prime indices for decomposition.
    nsm_primes: Vec<u16>,  // 63 entries
}

/// Distance matrix: symmetric, u8 palette-quantized.
pub struct DistanceMatrix {
    /// Upper triangle packed: matrix[i * 4096 + j] for i < j.
    data: Vec<u8>,   // 4096 * 4095 / 2 = ~8.4M entries
    /// Palette: 256 u8 levels → actual distance values.
    palette: [u32; 256],
}

impl DistanceMatrix {
    /// Build from exact 96D vectors. Called once at build time.
    pub fn build(vectors: &[[f32; 96]]) -> Self;

    /// O(1) distance lookup.
    #[inline(always)]
    pub fn get(&self, a: u16, b: u16) -> u32 {
        if a == b { return 0; }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        let idx = (hi as usize * (hi as usize - 1) / 2) + lo as usize;
        self.palette[self.data[idx] as usize]
    }
}

impl NsmEncoder {
    /// Load from word_frequency/ directory.
    pub fn load(dir: &Path) -> Result<Self, Error>;

    /// Encode a single triple as comparable representation.
    pub fn encode_triple(&self, triple: &SpoTriple) -> EncodedTriple;

    /// Encode a full sentence structure.
    pub fn encode_sentence(&self, structure: &SentenceStructure) -> EncodedSentence;

    /// SPO distance between two triples (3 matrix lookups).
    pub fn triple_distance(&self, a: &SpoTriple, b: &SpoTriple) -> u32;

    /// SPO similarity between two triples (calibrated f32).
    pub fn triple_similarity(&self, a: &SpoTriple, b: &SpoTriple) -> f32;

    /// Per-role similarity decomposition.
    pub fn triple_similarity_decomposed(
        &self, a: &SpoTriple, b: &SpoTriple,
    ) -> TripleSimilarity;

    /// Sentence similarity via VSA composition.
    pub fn sentence_similarity(
        &self, a: &EncodedSentence, b: &EncodedSentence,
    ) -> f32;

    /// Nearest NSM prime for a word.
    pub fn nearest_prime(&self, rank: u16) -> (u16, f32);

    /// NSM decomposition: word → Vec<(prime, distance)>.
    pub fn decompose(&self, rank: u16) -> Vec<(u16, f32)>;
}

/// Encoded triple: carries both the index triple and the VSA representation.
pub struct EncodedTriple {
    pub triple: SpoTriple,
    /// VSA representation: bound + bundled binary vector.
    pub vsa: BitVec,   // 10K bits for sentence-level composition
}

/// Encoded sentence: multiple triples + modifiers, composed via VSA.
pub struct EncodedSentence {
    pub triples: Vec<EncodedTriple>,
    pub modifiers: Vec<(u16, u16, ModRelation)>,
    /// Full sentence VSA: bundle of all triples + modifiers.
    pub vsa: BitVec,
}

/// Decomposed similarity: WHO did WHAT to WHOM.
pub struct TripleSimilarity {
    pub subject: f32,     // "dog" vs "cat" — who is the agent?
    pub predicate: f32,   // "bite" vs "love" — what happened?
    pub object: f32,      // "man" vs "woman" — who is the patient?
    pub composite: f32,   // weighted combination
}
```

## DELIVERABLE 4: nsm_similarity.rs

SimilarityTable built from EXACT 4096² distribution. Drop-in cosine replacement.

```rust
/// Similarity table: built from the complete pairwise distance distribution.
///
/// For 4,096 words, we have 4096 × 4095 / 2 = 8,386,560 unique pairs.
/// This is small enough to compute ALL distances and build the exact CDF.
/// No reservoir sampling. No parametric approximation. Exact.
pub struct SimilarityTable {
    /// 256 × f16: distance bucket → similarity [0.0, 1.0].
    table: [f16; 256],
    bucket_width: u32,
    max_distance: u32,
    /// Statistics for threshold translation.
    mu: u32,      // mean distance
    sigma: u32,   // standard deviation
}

impl SimilarityTable {
    /// Build from the exact distance matrix. Called once at build time.
    pub fn from_distance_matrix(dm: &DistanceMatrix) -> Self {
        // Compute all 8.4M pairwise distances
        // Sort → empirical CDF
        // Quantize into 256 buckets
        // table[bucket] = 1.0 - CDF(bucket_center)
    }

    /// O(1) similarity lookup.
    #[inline(always)]
    pub fn similarity(&self, distance: u32) -> f32;

    /// Translate a cosine threshold to bgz17 threshold.
    /// Consumer says "I want cosine > 0.8" → returns equivalent bgz17 threshold.
    pub fn cosine_equivalent(&self, cosine_threshold: f32) -> f32;

    /// Translate a bgz17 similarity to approximate cosine value.
    pub fn to_cosine(&self, similarity: f32) -> f32;

    /// Band classification (for compatibility with bgz17 Precision enum).
    pub fn band(&self, similarity: f32) -> Precision {
        if similarity > 0.95 { Precision::Foveal }
        else if similarity > 0.85 { Precision::Near }
        else if similarity > 0.70 { Precision::Good }
        else { Precision::Miss }
    }
}
```

## DELIVERABLE 5: Integration with ThinkingGraph

The semantic transformer plugs into the cognitive verb pipeline.
Each verb that reasons about concepts now has access to MEANING.

```rust
/// ThinkingGraph with semantic awareness.
impl ThinkingGraph {
    /// Attach the semantic encoder. After this, verbs can reason about meaning.
    pub fn with_semantics(mut self, encoder: NsmEncoder) -> Self {
        self.semantics = Some(encoder);
        self
    }
}

/// Cognitive verbs enhanced with semantic understanding:
///
/// EXPLORE: when activating neighbor styles, weight by semantic similarity
///   of their typical queries. Analytical thinking about "think" should
///   activate neighbors that also handle mental predicates.
///
/// HYPOTHESIS: generate hypotheses grounded in NSM decomposition.
///   "journalism" → (news + write) → hypothesis about information domain.
///
/// COUNTERFACTUAL: negate the PREDICATE plane of an SPO triple.
///   "dog bites man" → find the antonym of "bite" → "dog protects man"
///   Antonym = word with high subject/object similarity but LOW predicate similarity.
///
/// SYNTHESIS: merge triples with per-role similarity.
///   "dog bites man" + "cat bites boy" → subject_sim = 0.72, pred_sim = 1.0, obj_sim = 0.78
///   High predicate agreement → synthesize: "animals bite young humans"
///
/// ABDUCTION: given an object, find plausible subjects via matrix column scan.
///   "who would write journalism?" → scan distance_matrix column for "journalism"
///   → nearest: "reporter", "editor", "writer" — candidate subjects.
///
/// INTERRELATE: bridge across semantic domains via shared predicates.
///   "doctor treats patient" and "mechanic treats car"
///   → predicate similarity = 1.0, subject/object similarity = 0.3
///   → cross-domain analogy detected
///
/// DEEPEN: resolve a word into its NSM prime decomposition.
///   "journalism" → [say, word, true, people, new, know]
///   → deeper understanding: journalism is about saying true new words to people
///
/// MODULATE: adjust ThinkingStyle based on semantic content.
///   Mental predicates (think, know, feel) → boost Metacognitive style.
///   Action predicates (move, do, touch) → boost Pragmatic style.
///   Evaluative predicates (good, bad) → boost Empathetic style.
///   The CONTENT drives the THINKING MODE.
```

## DELIVERABLE 6: DataFusion UDF Integration

Expose as SQL-callable functions in lance-graph's query engine.

```sql
-- Word similarity (two words, one number)
SELECT nsm_similarity('think', 'know') AS sim;
-- → 0.92

-- Triple similarity (two triples, decomposed)
SELECT nsm_triple_similarity(
    'dog', 'bite', 'man',
    'cat', 'bite', 'boy'
) AS sim;
-- → {subject: 0.72, predicate: 1.00, object: 0.78, composite: 0.83}

-- Sentence similarity (full VSA composition)
SELECT nsm_sentence_similarity(
    'the big dog bit the old man',
    'a large cat attacked the elderly woman'
) AS sim;
-- → 0.71

-- NSM decomposition
SELECT nsm_decompose('journalism');
-- → [(say, 0.82), (word, 0.79), (true, 0.74), (people, 0.71), (new, 0.68)]

-- Nearest NSM prime
SELECT nsm_nearest_prime('journalism');
-- → (say, 0.82)

-- Find semantically similar words
SELECT word, nsm_similarity(word, 'think') AS sim
FROM vocabulary
WHERE sim > 0.85
ORDER BY sim DESC;
-- → [(know, 0.92), (feel, 0.89), (believe, 0.87), (understand, 0.86)]

-- Find analogous triples
MATCH (a)-[r1]->(b), (c)-[r2]->(d)
WHERE nsm_predicate_similarity(r1, r2) > 0.9
  AND nsm_subject_similarity(a, c) < 0.5
RETURN a, r1, b, c, r2, d;
-- → cross-domain analogies where same action applies to different domains
```

## DELIVERABLE 7: Build Pipeline (codebook generation)

One-time build step that produces all runtime artifacts from raw COCA data.

```rust
/// Build the complete DeepNSM runtime from word frequency data.
///
/// Input:  word_frequency/ directory (CSVs + xlsx)
/// Output: nsm_runtime/ directory (binary artifacts for O(1) loading)
///
/// Artifacts:
///   vocabulary.bin      — 4,096 entries: rank, PoS, freq (32 KB)
///   forms.bin           — inflected form → lemma rank hash table (64 KB)
///   vectors_bf16.bin    — 4,096 × 96 BF16 distributional vectors (768 KB)
///   distance_matrix.bin — 4,096 × 4,096 u8 symmetric (8 MB)
///   similarity.bin      — 256 × f16 calibration table (512 B)
///   roles.bin           — 6 × 10,000-bit role vectors (7.5 KB)
///   nsm_primes.bin      — 63 prime indices + decomposition paths (2 KB)
///   oov_buckets.bin     — words 4097-5050 → nearest rank (4 KB)
///
/// Total runtime: ~9 MB. Fits in L2/L3 cache. Loads in <10ms.
pub fn build_nsm_runtime(
    word_frequency_dir: &Path,
    output_dir: &Path,
) -> Result<BuildStats, Error>;

pub struct BuildStats {
    pub vocabulary_size: usize,     // 4096
    pub forms_count: usize,         // ~11,000
    pub distance_pairs: u64,        // 8,386,560
    pub similarity_mu: u32,         // mean pairwise distance
    pub similarity_sigma: u32,      // std dev
    pub nsm_coverage: f32,          // fraction of NSM primes in vocab
    pub oov_words: usize,           // words resolved via CLAM buckets
    pub total_bytes: u64,           // ~9 MB
    pub build_time_ms: u64,         // ~2000ms (dominated by distance matrix)
}
```

## FILE LAYOUT

```
crates/deepnsm/
├── Cargo.toml
├── src/
│   ├── lib.rs                # pub mod everything, NsmRuntime struct
│   ├── vocabulary.rs         # Vocabulary, WordEntry, PoS, Token
│   ├── tokenizer.rs          # tokenize(), forms resolution, OOV
│   ├── parser.rs             # parse(), SentenceStructure, SPO extraction
│   ├── triple.rs             # SpoTriple (36-bit packed), distance, similarity
│   ├── encoder.rs            # NsmEncoder, VSA binding, sentence composition
│   ├── similarity.rs         # SimilarityTable, calibration, threshold translation
│   ├── distance_matrix.rs    # DistanceMatrix (u8 palette, 8MB)
│   ├── primes.rs             # NSM prime table, decomposition, nearest_prime
│   ├── build.rs              # build_nsm_runtime(), artifact generation
│   └── udf.rs                # DataFusion UDF registration
├── data/
│   └── README.md             # points to word_frequency/ in DeepNSM repo
└── tests/
    ├── test_tokenizer.rs
    ├── test_parser.rs
    ├── test_similarity.rs
    └── test_integration.rs
```

Wire into lance-graph:
```
crates/lance-graph-planner/src/thinking/
├── graph.rs          # add: with_semantics(NsmEncoder)
├── semantic.rs       # NEW: semantic verb enhancements
└── mod.rs            # add: pub mod semantic
```

## TESTS

### Tokenizer
1. "the dog bit the man" → [1, 671, 2943, 1, 95] (ranks)
2. "bit" → lemma "bite" (rank 2943) via forms table
3. "sleeping" → lemma "sleep" via forms table
4. "journalism" → OOV → nearest: "news" (rank 206)
5. "" → empty vec
6. "THE Dog BIT" → case-insensitive → same as lowercase

### Parser
7. "the big dog bit the old man" → SPO(dog, bite, man) + Mod(big→dog, old→man)
8. "dog bites man" ≠ "man bites dog" (different subjects)
9. "the dog did not bite the man" → SPO(dog, bite, man) + negated=true
10. "the dog bit the man and the cat" → two triples (conjunction fork)
11. "I think" → SPO(i, think, None) (intransitive)

### Similarity
12. similarity(think, know) > similarity(think, big)
13. similarity(dog, dog) = 1.0
14. similarity(the, journalism) < 0.5
15. triple_sim("dog bites man", "cat bites man") → subj<1.0, pred=1.0, obj=1.0
16. triple_sim("dog bites man", "man bites dog") → subj and obj SWAPPED
17. sentence_sim("dog bites man", "dog bites man") = 1.0
18. sentence_sim("dog bites man", "man bites dog") < 0.85

### NSM Decomposition
19. decompose("journalism") contains "say" and "word" and "people"
20. nearest_prime("think") = "think" (it IS a prime)
21. nearest_prime("believe") → "think" (nearest prime)

### Integration with ThinkingGraph
22. Cognitive verb SYNTHESIS uses semantic similarity for merge decisions
23. Cognitive verb COUNTERFACTUAL negates predicate plane
24. Cognitive verb MODULATE shifts thinking style based on content PoS

## PERFORMANCE TARGETS

```
Operation              Target         Method
─────────              ──────         ──────
tokenize(word)         < 100ns        hash lookup
tokenize(sentence)     < 1μs          O(n) scan
parse(sentence)        < 500ns        6-state FSM
triple_distance        < 10ns         3 matrix lookups
triple_similarity      < 15ns         3 lookups + table
sentence_similarity    < 5μs          VSA compose + hamming
decompose(word)        < 1μs          scan 63 primes
full pipeline          < 10μs         text → calibrated similarity
```

## OUTPUT

Branch: `feat/deepnsm-cam`
New crate: `crates/deepnsm/`
Modified: `crates/lance-graph-planner/src/thinking/{mod,graph,semantic}.rs`
Data: loaded from `DeepNSM/word_frequency/` at build time
Tests: 24+ tests covering tokenizer, parser, similarity, integration
