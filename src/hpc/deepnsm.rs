//! DeepNSM: 74 semantic primes → 40K derived concept decomposition.
//!
//! Natural Semantic Metalanguage: 74 primes → derived concepts.
//! NSM provides the WHAT (structural semantics).
//! Three separate systems, one pipeline: NSM is the parser.
//!
//! Transcoded from Python DeepNSM (<https://github.com/AdaWorldAPI/DeepNSM>).
//! Rust 1.94.0 · SIMD via `crate::simd::F32x16` · Arrow 57 / DataFusion 51 ready.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

/// The 65 universal semantic primes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NsmPrime {
    // Substantives
    I = 0, You, Someone, Something, Thing, Body,
    // Relational
    Kind, Part,
    // Determiners
    This, TheSame, Other, Else, Another,
    // Quantifiers
    One, Two, Some, All, Much, Many, Little, Few,
    // Evaluators
    Good, Bad,
    // Descriptors
    Big, Small,
    // Mental
    Think, Know, Want, DontWant, Feel, See, Hear,
    // Speech
    Say, Words, True,
    // Actions
    Do, Happen, Move,
    // Existence
    Be, ThereIs, BeSomeone, Mine,
    // Life
    Live, Die,
    // Time
    When, Time, Now, Before, After, ALongTime, AShortTime, ForSomeTime, Moment,
    // Space
    Where, Place, Here, Above, Below, Far, Near, Side, Inside, Touch, Contact,
    // Logical
    Not, Maybe, Can, Because, If,
    // Intensifier
    Very, More,
    // Similarity
    Like, As, Way,
}
// Total: 74 variants (indices 0..73)

/// Category of an NSM prime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NsmCategory {
    Substantive,
    Relational,
    Determiner,
    Quantifier,
    Evaluator,
    Descriptor,
    Mental,
    Speech,
    Action,
    Existence,
    Life,
    Time,
    Space,
    Logical,
    Intensifier,
    Similarity,
}

/// Result of NSM decomposition: a weighted list of primes.
#[derive(Clone, Debug)]
pub struct NsmDecomposition {
    pub weights: [f32; 74],
    pub dominant: Vec<NsmPrime>,
}

/// A vocabulary entry: word → prime decomposition.
#[derive(Clone, Debug)]
pub struct NsmEntry {
    pub word: String,
    pub primes: Vec<(NsmPrime, f32)>,
}

static ALL_PRIMES: [NsmPrime; 74] = [
    // Substantives (6)
    NsmPrime::I, NsmPrime::You, NsmPrime::Someone, NsmPrime::Something,
    NsmPrime::Thing, NsmPrime::Body,
    // Relational (2)
    NsmPrime::Kind, NsmPrime::Part,
    // Determiners (5)
    NsmPrime::This, NsmPrime::TheSame, NsmPrime::Other, NsmPrime::Else,
    NsmPrime::Another,
    // Quantifiers (8)
    NsmPrime::One, NsmPrime::Two, NsmPrime::Some,
    NsmPrime::All, NsmPrime::Much, NsmPrime::Many, NsmPrime::Little,
    NsmPrime::Few,
    // Evaluators (2)
    NsmPrime::Good, NsmPrime::Bad,
    // Descriptors (2)
    NsmPrime::Big, NsmPrime::Small,
    // Mental (7)
    NsmPrime::Think, NsmPrime::Know, NsmPrime::Want,
    NsmPrime::DontWant, NsmPrime::Feel, NsmPrime::See, NsmPrime::Hear,
    // Speech (3)
    NsmPrime::Say, NsmPrime::Words, NsmPrime::True,
    // Actions (3)
    NsmPrime::Do, NsmPrime::Happen, NsmPrime::Move,
    // Existence (4)
    NsmPrime::Be, NsmPrime::ThereIs, NsmPrime::BeSomeone, NsmPrime::Mine,
    // Life (2)
    NsmPrime::Live, NsmPrime::Die,
    // Time (9)
    NsmPrime::When, NsmPrime::Time, NsmPrime::Now, NsmPrime::Before,
    NsmPrime::After, NsmPrime::ALongTime, NsmPrime::AShortTime,
    NsmPrime::ForSomeTime, NsmPrime::Moment,
    // Space (11)
    NsmPrime::Where, NsmPrime::Place,
    NsmPrime::Here, NsmPrime::Above, NsmPrime::Below, NsmPrime::Far,
    NsmPrime::Near, NsmPrime::Side, NsmPrime::Inside, NsmPrime::Touch,
    NsmPrime::Contact,
    // Logical (5)
    NsmPrime::Not, NsmPrime::Maybe, NsmPrime::Can,
    NsmPrime::Because, NsmPrime::If,
    // Intensifier (2)
    NsmPrime::Very, NsmPrime::More,
    // Similarity (3)
    NsmPrime::Like, NsmPrime::As, NsmPrime::Way,
];

static PRIME_NAMES: [&str; 74] = [
    "I", "YOU", "SOMEONE", "SOMETHING", "THING", "BODY",
    "KIND", "PART",
    "THIS", "THE_SAME", "OTHER", "ELSE", "ANOTHER",
    "ONE", "TWO", "SOME", "ALL", "MUCH", "MANY", "LITTLE", "FEW",
    "GOOD", "BAD",
    "BIG", "SMALL",
    "THINK", "KNOW", "WANT", "DONT_WANT", "FEEL", "SEE", "HEAR",
    "SAY", "WORDS", "TRUE",
    "DO", "HAPPEN", "MOVE",
    "BE", "THERE_IS", "BE_SOMEONE", "MINE",
    "LIVE", "DIE",
    "WHEN", "TIME", "NOW", "BEFORE", "AFTER", "A_LONG_TIME", "A_SHORT_TIME",
    "FOR_SOME_TIME", "MOMENT",
    "WHERE", "PLACE", "HERE", "ABOVE", "BELOW", "FAR", "NEAR", "SIDE",
    "INSIDE", "TOUCH", "CONTACT",
    "NOT", "MAYBE", "CAN", "BECAUSE", "IF",
    "VERY", "MORE",
    "LIKE", "AS", "WAY",
];

impl NsmPrime {
    /// Human-readable name for this prime.
    pub fn name(&self) -> &'static str {
        PRIME_NAMES[*self as u8 as usize]
    }

    /// Category this prime belongs to.
    pub fn category(&self) -> NsmCategory {
        match *self {
            NsmPrime::I | NsmPrime::You | NsmPrime::Someone
            | NsmPrime::Something | NsmPrime::Thing | NsmPrime::Body => NsmCategory::Substantive,
            NsmPrime::Kind | NsmPrime::Part => NsmCategory::Relational,
            NsmPrime::This | NsmPrime::TheSame | NsmPrime::Other
            | NsmPrime::Else | NsmPrime::Another => NsmCategory::Determiner,
            NsmPrime::One | NsmPrime::Two | NsmPrime::Some | NsmPrime::All
            | NsmPrime::Much | NsmPrime::Many | NsmPrime::Little | NsmPrime::Few => {
                NsmCategory::Quantifier
            }
            NsmPrime::Good | NsmPrime::Bad => NsmCategory::Evaluator,
            NsmPrime::Big | NsmPrime::Small => NsmCategory::Descriptor,
            NsmPrime::Think | NsmPrime::Know | NsmPrime::Want | NsmPrime::DontWant
            | NsmPrime::Feel | NsmPrime::See | NsmPrime::Hear => NsmCategory::Mental,
            NsmPrime::Say | NsmPrime::Words | NsmPrime::True => NsmCategory::Speech,
            NsmPrime::Do | NsmPrime::Happen | NsmPrime::Move => NsmCategory::Action,
            NsmPrime::Be | NsmPrime::ThereIs | NsmPrime::BeSomeone | NsmPrime::Mine => {
                NsmCategory::Existence
            }
            NsmPrime::Live | NsmPrime::Die => NsmCategory::Life,
            NsmPrime::When | NsmPrime::Time | NsmPrime::Now | NsmPrime::Before
            | NsmPrime::After | NsmPrime::ALongTime | NsmPrime::AShortTime
            | NsmPrime::ForSomeTime | NsmPrime::Moment => NsmCategory::Time,
            NsmPrime::Where | NsmPrime::Place | NsmPrime::Here | NsmPrime::Above
            | NsmPrime::Below | NsmPrime::Far | NsmPrime::Near | NsmPrime::Side
            | NsmPrime::Inside | NsmPrime::Touch | NsmPrime::Contact => NsmCategory::Space,
            NsmPrime::Not | NsmPrime::Maybe | NsmPrime::Can | NsmPrime::Because
            | NsmPrime::If => NsmCategory::Logical,
            NsmPrime::Very | NsmPrime::More => NsmCategory::Intensifier,
            NsmPrime::Like | NsmPrime::As | NsmPrime::Way => NsmCategory::Similarity,
        }
    }

    /// Case-insensitive lookup by name.
    pub fn from_name(name: &str) -> Option<NsmPrime> {
        let upper = name.to_uppercase();
        for (i, &n) in PRIME_NAMES.iter().enumerate() {
            if n == upper {
                return Some(ALL_PRIMES[i]);
            }
        }
        None
    }

    /// All 65 primes in order.
    pub fn all() -> &'static [NsmPrime; 74] {
        &ALL_PRIMES
    }
}

/// Build the vocabulary lookup table (word → list of (prime, weight)).
fn build_vocab_map() -> HashMap<String, Vec<(NsmPrime, f32)>> {
    use NsmPrime::*;

    let entries: Vec<(&str, Vec<(NsmPrime, f32)>)> = vec![
        // === Direct prime words (1:1 mapping) ===
        ("i", vec![(I, 1.0)]),
        ("me", vec![(I, 1.0)]),
        ("my", vec![(I, 0.8), (Mine, 0.5)]),
        ("myself", vec![(I, 1.0)]),
        ("you", vec![(You, 1.0)]),
        ("your", vec![(You, 0.8), (Mine, 0.5)]),
        ("yourself", vec![(You, 1.0)]),
        ("someone", vec![(Someone, 1.0)]),
        ("somebody", vec![(Someone, 1.0)]),
        ("something", vec![(Something, 1.0)]),
        ("thing", vec![(Thing, 1.0)]),
        ("things", vec![(Thing, 1.0), (Many, 0.3)]),
        ("body", vec![(Body, 1.0)]),
        ("kind", vec![(Kind, 1.0)]),
        ("type", vec![(Kind, 1.0)]),
        ("sort", vec![(Kind, 0.8)]),
        ("part", vec![(Part, 1.0)]),
        ("piece", vec![(Part, 0.8), (Thing, 0.3)]),
        ("portion", vec![(Part, 0.9)]),
        ("this", vec![(This, 1.0)]),
        ("same", vec![(TheSame, 1.0)]),
        ("other", vec![(Other, 1.0)]),
        ("else", vec![(Else, 1.0)]),
        ("another", vec![(Another, 1.0)]),
        ("one", vec![(One, 1.0)]),
        ("two", vec![(Two, 1.0)]),
        ("some", vec![(Some, 1.0)]),
        ("all", vec![(All, 1.0)]),
        ("every", vec![(All, 1.0)]),
        ("everything", vec![(All, 0.8), (Something, 0.5)]),
        ("everyone", vec![(All, 0.8), (Someone, 0.5)]),
        ("much", vec![(Much, 1.0)]),
        ("many", vec![(Many, 1.0)]),
        ("little", vec![(Little, 1.0)]),
        ("few", vec![(Few, 1.0)]),
        ("good", vec![(Good, 1.0)]),
        ("bad", vec![(Bad, 1.0)]),
        ("big", vec![(Big, 1.0)]),
        ("large", vec![(Big, 1.0)]),
        ("huge", vec![(Big, 1.0), (Very, 0.5)]),
        ("small", vec![(Small, 1.0)]),
        ("tiny", vec![(Small, 1.0), (Very, 0.5)]),
        ("think", vec![(Think, 1.0)]),
        ("thought", vec![(Think, 0.9), (Before, 0.3)]),
        ("know", vec![(Know, 1.0)]),
        ("knowledge", vec![(Know, 1.0), (Thing, 0.3)]),
        ("want", vec![(Want, 1.0)]),
        ("desire", vec![(Want, 0.9), (Feel, 0.4)]),
        ("wish", vec![(Want, 0.8), (Feel, 0.3)]),
        ("feel", vec![(Feel, 1.0)]),
        ("feeling", vec![(Feel, 1.0), (Thing, 0.2)]),
        ("emotion", vec![(Feel, 0.9), (Thing, 0.3)]),
        ("see", vec![(See, 1.0)]),
        ("look", vec![(See, 0.9), (Want, 0.2)]),
        ("watch", vec![(See, 0.9), (Time, 0.2)]),
        ("hear", vec![(Hear, 1.0)]),
        ("listen", vec![(Hear, 0.9), (Want, 0.2)]),
        ("say", vec![(Say, 1.0)]),
        ("tell", vec![(Say, 0.9), (Someone, 0.3)]),
        ("speak", vec![(Say, 0.9), (Words, 0.4)]),
        ("talk", vec![(Say, 0.8), (Words, 0.5)]),
        ("word", vec![(Words, 1.0)]),
        ("words", vec![(Words, 1.0)]),
        ("language", vec![(Words, 0.8), (Kind, 0.3)]),
        ("true", vec![(True, 1.0)]),
        ("truth", vec![(True, 1.0), (Thing, 0.2)]),
        ("false", vec![(True, 0.3), (Not, 0.8)]),
        ("do", vec![(Do, 1.0)]),
        ("act", vec![(Do, 0.9)]),
        ("action", vec![(Do, 0.9), (Thing, 0.3)]),
        ("happen", vec![(Happen, 1.0)]),
        ("event", vec![(Happen, 0.8), (Thing, 0.3)]),
        ("occur", vec![(Happen, 0.9)]),
        ("move", vec![(Move, 1.0)]),
        ("motion", vec![(Move, 0.9), (Thing, 0.3)]),
        ("go", vec![(Move, 0.8), (Do, 0.3)]),
        ("come", vec![(Move, 0.8), (Here, 0.3)]),
        ("walk", vec![(Move, 0.8), (Body, 0.3)]),
        ("run", vec![(Move, 0.9), (Very, 0.3)]),
        ("be", vec![(Be, 1.0)]),
        ("is", vec![(Be, 1.0)]),
        ("am", vec![(Be, 0.9), (I, 0.3)]),
        ("are", vec![(Be, 1.0)]),
        ("was", vec![(Be, 0.9), (Before, 0.3)]),
        ("were", vec![(Be, 0.9), (Before, 0.3)]),
        ("exist", vec![(ThereIs, 1.0)]),
        ("existence", vec![(ThereIs, 0.9), (Thing, 0.3)]),
        ("mine", vec![(Mine, 1.0)]),
        ("own", vec![(Mine, 0.9)]),
        ("belong", vec![(Mine, 0.8), (Be, 0.3)]),
        ("live", vec![(Live, 1.0)]),
        ("alive", vec![(Live, 1.0)]),
        ("life", vec![(Live, 0.9), (Thing, 0.3)]),
        ("die", vec![(Die, 1.0)]),
        ("dead", vec![(Die, 0.9)]),
        ("death", vec![(Die, 0.9), (Thing, 0.3)]),
        ("when", vec![(When, 1.0)]),
        ("time", vec![(Time, 1.0)]),
        ("now", vec![(Now, 1.0)]),
        ("before", vec![(Before, 1.0)]),
        ("after", vec![(After, 1.0)]),
        ("long", vec![(ALongTime, 0.7), (Big, 0.4)]),
        ("short", vec![(AShortTime, 0.5), (Small, 0.4)]),
        ("moment", vec![(Moment, 1.0)]),
        ("instant", vec![(Moment, 0.9)]),
        ("where", vec![(Where, 1.0)]),
        ("place", vec![(Place, 1.0)]),
        ("location", vec![(Place, 0.9), (Where, 0.3)]),
        ("here", vec![(Here, 1.0)]),
        ("there", vec![(Place, 0.7), (Far, 0.3)]),
        ("above", vec![(Above, 1.0)]),
        ("over", vec![(Above, 0.8)]),
        ("below", vec![(Below, 1.0)]),
        ("under", vec![(Below, 0.8)]),
        ("far", vec![(Far, 1.0)]),
        ("distant", vec![(Far, 0.9)]),
        ("near", vec![(Near, 1.0)]),
        ("close", vec![(Near, 0.9)]),
        ("nearby", vec![(Near, 1.0)]),
        ("side", vec![(Side, 1.0)]),
        ("beside", vec![(Side, 0.8), (Near, 0.3)]),
        ("inside", vec![(Inside, 1.0)]),
        ("within", vec![(Inside, 0.9)]),
        ("touch", vec![(Touch, 1.0)]),
        ("contact", vec![(Contact, 1.0)]),
        ("not", vec![(Not, 1.0)]),
        ("no", vec![(Not, 0.9)]),
        ("never", vec![(Not, 0.9), (Time, 0.3)]),
        ("nothing", vec![(Not, 0.8), (Something, 0.3)]),
        ("nobody", vec![(Not, 0.8), (Someone, 0.3)]),
        ("maybe", vec![(Maybe, 1.0)]),
        ("perhaps", vec![(Maybe, 1.0)]),
        ("possibly", vec![(Maybe, 0.9), (Can, 0.3)]),
        ("can", vec![(Can, 1.0)]),
        ("could", vec![(Can, 0.8), (Maybe, 0.3)]),
        ("able", vec![(Can, 0.9)]),
        ("because", vec![(Because, 1.0)]),
        ("cause", vec![(Because, 0.9), (Do, 0.3)]),
        ("reason", vec![(Because, 0.8), (Think, 0.3)]),
        ("why", vec![(Because, 0.8)]),
        ("if", vec![(If, 1.0)]),
        ("whether", vec![(If, 0.8)]),
        ("very", vec![(Very, 1.0)]),
        ("really", vec![(Very, 0.9)]),
        ("extremely", vec![(Very, 1.0), (More, 0.3)]),
        ("more", vec![(More, 1.0)]),
        ("most", vec![(More, 0.9), (All, 0.3)]),
        ("less", vec![(More, 0.5), (Not, 0.3), (Little, 0.3)]),
        ("like", vec![(Like, 1.0)]),
        ("similar", vec![(Like, 0.9)]),
        ("as", vec![(As, 1.0)]),
        ("way", vec![(Way, 1.0)]),
        ("how", vec![(Way, 0.8)]),
        ("method", vec![(Way, 0.8), (Thing, 0.3)]),
        // === Derived concepts (nouns) ===
        ("person", vec![(Someone, 1.0)]),
        ("people", vec![(Someone, 0.8), (Many, 0.5)]),
        ("man", vec![(Someone, 0.8), (Big, 0.2)]),
        ("woman", vec![(Someone, 0.8)]),
        ("child", vec![(Someone, 0.7), (Small, 0.3), (Live, 0.2)]),
        ("baby", vec![(Someone, 0.6), (Small, 0.5), (Live, 0.3)]),
        ("friend", vec![(Someone, 0.7), (Good, 0.5), (Feel, 0.3)]),
        ("enemy", vec![(Someone, 0.6), (Bad, 0.5), (DontWant, 0.3)]),
        ("family", vec![(Someone, 0.6), (Kind, 0.3), (Live, 0.3), (Good, 0.2)]),
        ("mother", vec![(Someone, 0.7), (Live, 0.3), (Good, 0.2)]),
        ("father", vec![(Someone, 0.7), (Live, 0.3), (Good, 0.2)]),
        ("home", vec![(Place, 0.8), (Live, 0.5), (Good, 0.2)]),
        ("house", vec![(Place, 0.8), (Inside, 0.5), (Thing, 0.3)]),
        ("room", vec![(Place, 0.7), (Inside, 0.6), (Part, 0.3)]),
        ("door", vec![(Thing, 0.6), (Move, 0.3), (Inside, 0.3)]),
        ("window", vec![(Thing, 0.6), (See, 0.3), (Inside, 0.3)]),
        ("water", vec![(Something, 0.8), (Move, 0.2), (Live, 0.2)]),
        ("food", vec![(Something, 0.7), (Live, 0.4), (Good, 0.2)]),
        ("fire", vec![(Something, 0.6), (Bad, 0.3), (Die, 0.2), (Feel, 0.2)]),
        ("earth", vec![(Place, 0.6), (Big, 0.4), (Below, 0.3)]),
        ("sky", vec![(Place, 0.5), (Above, 0.6), (Big, 0.3)]),
        ("sun", vec![(Something, 0.6), (Above, 0.3), (See, 0.2), (Big, 0.2)]),
        ("moon", vec![(Something, 0.6), (Above, 0.3), (Time, 0.2)]),
        ("star", vec![(Something, 0.5), (Above, 0.4), (Far, 0.3), (Small, 0.2)]),
        ("tree", vec![(Something, 0.6), (Live, 0.4), (Big, 0.3)]),
        ("flower", vec![(Something, 0.5), (Live, 0.3), (Good, 0.3), (Small, 0.2)]),
        ("animal", vec![(Something, 0.7), (Live, 0.6), (Move, 0.3)]),
        ("cat", vec![(Something, 0.6), (Live, 0.5), (Small, 0.3)]),
        ("dog", vec![(Something, 0.6), (Live, 0.5), (Big, 0.3), (Good, 0.2)]),
        ("bird", vec![(Something, 0.6), (Live, 0.4), (Move, 0.3), (Above, 0.3)]),
        ("fish", vec![(Something, 0.6), (Live, 0.4), (Move, 0.3), (Inside, 0.2)]),
        ("horse", vec![(Something, 0.6), (Live, 0.5), (Big, 0.4), (Move, 0.3)]),
        ("book", vec![(Thing, 0.6), (Words, 0.5), (Know, 0.3)]),
        ("story", vec![(Words, 0.5), (Say, 0.3), (Happen, 0.3)]),
        ("name", vec![(Words, 0.6), (Someone, 0.3), (Know, 0.2)]),
        ("number", vec![(Thing, 0.5), (One, 0.3), (Many, 0.3)]),
        // === Derived concepts (verbs) ===
        ("love", vec![(Feel, 0.8), (Good, 0.6), (Want, 0.4), (Someone, 0.2)]),
        ("hate", vec![(Feel, 0.7), (Bad, 0.6), (DontWant, 0.5)]),
        ("fear", vec![(Feel, 0.7), (Bad, 0.5), (DontWant, 0.3)]),
        ("hope", vec![(Feel, 0.5), (Want, 0.6), (Good, 0.3), (After, 0.2)]),
        ("help", vec![(Do, 0.7), (Good, 0.6), (Want, 0.3)]),
        ("hurt", vec![(Do, 0.5), (Bad, 0.6), (Feel, 0.4)]),
        ("kill", vec![(Do, 0.7), (Die, 0.8), (Bad, 0.4)]),
        ("fight", vec![(Do, 0.7), (Bad, 0.3), (Move, 0.3), (DontWant, 0.2)]),
        ("give", vec![(Do, 0.7), (Someone, 0.3), (Good, 0.2)]),
        ("take", vec![(Do, 0.7), (Mine, 0.4)]),
        ("make", vec![(Do, 0.8), (Something, 0.3)]),
        ("create", vec![(Do, 0.8), (Something, 0.4), (ThereIs, 0.3)]),
        ("build", vec![(Do, 0.7), (Thing, 0.4), (Big, 0.2)]),
        ("break", vec![(Do, 0.6), (Bad, 0.4), (Part, 0.3)]),
        ("find", vec![(See, 0.5), (Know, 0.4), (Want, 0.2)]),
        ("search", vec![(See, 0.4), (Want, 0.5), (Move, 0.2)]),
        ("learn", vec![(Know, 0.7), (Want, 0.3), (Think, 0.3)]),
        ("teach", vec![(Know, 0.5), (Say, 0.4), (Someone, 0.3)]),
        ("understand", vec![(Know, 0.8), (Think, 0.4)]),
        ("remember", vec![(Know, 0.6), (Before, 0.4), (Think, 0.3)]),
        ("forget", vec![(Not, 0.5), (Know, 0.5), (Before, 0.3)]),
        ("believe", vec![(Think, 0.7), (True, 0.5)]),
        ("imagine", vec![(Think, 0.7), (See, 0.3), (Not, 0.2)]),
        ("dream", vec![(Think, 0.5), (See, 0.3), (Feel, 0.3)]),
        ("sleep", vec![(Live, 0.3), (Not, 0.2), (Do, 0.2), (Time, 0.2)]),
        ("wake", vec![(Live, 0.3), (Now, 0.3), (See, 0.2)]),
        ("eat", vec![(Do, 0.6), (Live, 0.4), (Body, 0.3)]),
        ("drink", vec![(Do, 0.5), (Live, 0.3), (Body, 0.3)]),
        ("grow", vec![(Live, 0.5), (Big, 0.4), (More, 0.3)]),
        ("begin", vec![(Happen, 0.6), (Now, 0.3), (Before, 0.2)]),
        ("start", vec![(Happen, 0.6), (Now, 0.3)]),
        ("end", vec![(Happen, 0.5), (Not, 0.3), (After, 0.3)]),
        ("stop", vec![(Do, 0.4), (Not, 0.5), (Move, 0.2)]),
        ("wait", vec![(Time, 0.6), (Want, 0.3), (Not, 0.2)]),
        ("try", vec![(Do, 0.6), (Want, 0.5), (Can, 0.3)]),
        ("need", vec![(Want, 0.8), (Very, 0.3)]),
        ("use", vec![(Do, 0.6), (Thing, 0.3)]),
        ("change", vec![(Other, 0.5), (Happen, 0.4), (Do, 0.3)]),
        ("open", vec![(Do, 0.5), (Inside, 0.3), (Move, 0.2)]),
        ("close", vec![(Do, 0.4), (Inside, 0.3), (Not, 0.2)]),
        ("hold", vec![(Do, 0.5), (Touch, 0.4), (Body, 0.2)]),
        ("carry", vec![(Move, 0.6), (Thing, 0.3), (Body, 0.2)]),
        ("bring", vec![(Move, 0.6), (Here, 0.3), (Thing, 0.2)]),
        ("send", vec![(Move, 0.5), (Far, 0.3), (Do, 0.3)]),
        ("show", vec![(See, 0.6), (Do, 0.4), (Someone, 0.2)]),
        ("hide", vec![(Not, 0.5), (See, 0.4), (Inside, 0.2)]),
        ("read", vec![(See, 0.5), (Words, 0.5), (Know, 0.3)]),
        ("write", vec![(Do, 0.5), (Words, 0.6), (Say, 0.2)]),
        ("sing", vec![(Say, 0.5), (Feel, 0.3), (Good, 0.2)]),
        ("play", vec![(Do, 0.5), (Good, 0.3), (Feel, 0.2)]),
        ("work", vec![(Do, 0.8), (Want, 0.2), (Time, 0.2)]),
        ("rest", vec![(Not, 0.3), (Do, 0.3), (Feel, 0.3), (Good, 0.2)]),
        // === Adjectives ===
        ("happy", vec![(Feel, 0.7), (Good, 0.7)]),
        ("sad", vec![(Feel, 0.7), (Bad, 0.5)]),
        ("angry", vec![(Feel, 0.6), (Bad, 0.5), (DontWant, 0.3)]),
        ("afraid", vec![(Feel, 0.6), (Bad, 0.4), (DontWant, 0.3)]),
        ("brave", vec![(Feel, 0.4), (Good, 0.4), (Can, 0.3)]),
        ("strong", vec![(Big, 0.4), (Can, 0.5), (Body, 0.3)]),
        ("weak", vec![(Small, 0.3), (Not, 0.3), (Can, 0.3)]),
        ("fast", vec![(Move, 0.5), (Very, 0.3), (Time, 0.2)]),
        ("slow", vec![(Move, 0.4), (ALongTime, 0.3)]),
        ("new", vec![(Now, 0.5), (Other, 0.3)]),
        ("old", vec![(Before, 0.5), (ALongTime, 0.4)]),
        ("young", vec![(Live, 0.4), (AShortTime, 0.3), (Small, 0.2)]),
        ("hot", vec![(Feel, 0.5), (Body, 0.3)]),
        ("cold", vec![(Feel, 0.5), (Body, 0.3), (Bad, 0.2)]),
        ("dark", vec![(Not, 0.4), (See, 0.4)]),
        ("light", vec![(See, 0.5), (Small, 0.2)]),
        ("beautiful", vec![(See, 0.4), (Good, 0.6), (Very, 0.3)]),
        ("ugly", vec![(See, 0.3), (Bad, 0.5)]),
        ("right", vec![(Good, 0.5), (True, 0.5)]),
        ("wrong", vec![(Bad, 0.5), (Not, 0.3), (True, 0.2)]),
        ("easy", vec![(Can, 0.6), (Good, 0.3)]),
        ("hard", vec![(Not, 0.3), (Can, 0.3), (Much, 0.3)]),
        ("important", vec![(Good, 0.4), (Much, 0.4), (Think, 0.3)]),
        ("possible", vec![(Can, 0.6), (Maybe, 0.5)]),
        ("impossible", vec![(Not, 0.7), (Can, 0.5)]),
        ("different", vec![(Other, 0.8), (Not, 0.2), (TheSame, 0.1)]),
        ("same", vec![(TheSame, 1.0)]),
        ("together", vec![(TheSame, 0.4), (Near, 0.3), (Someone, 0.2)]),
        ("alone", vec![(One, 0.5), (Not, 0.2), (Someone, 0.2)]),
        ("free", vec![(Can, 0.6), (Not, 0.2), (DontWant, 0.1)]),
        ("full", vec![(All, 0.6), (Inside, 0.4)]),
        ("empty", vec![(Not, 0.5), (Inside, 0.3), (Something, 0.2)]),
        ("clean", vec![(Good, 0.4), (Not, 0.2), (Bad, 0.1)]),
        ("dirty", vec![(Bad, 0.4), (Body, 0.2)]),
        ("safe", vec![(Good, 0.5), (Not, 0.2), (Bad, 0.2)]),
        ("dangerous", vec![(Bad, 0.5), (Die, 0.3), (Can, 0.2)]),
        ("real", vec![(True, 0.7), (Be, 0.4)]),
        ("simple", vec![(One, 0.3), (Can, 0.3)]),
        ("deep", vec![(Inside, 0.5), (Much, 0.3), (Below, 0.3)]),
        ("high", vec![(Above, 0.7), (Big, 0.3)]),
        ("low", vec![(Below, 0.6), (Small, 0.2)]),
        ("wide", vec![(Big, 0.5), (Side, 0.3)]),
        ("narrow", vec![(Small, 0.5), (Side, 0.2)]),
        ("ready", vec![(Can, 0.5), (Now, 0.4), (Want, 0.2)]),
        ("sure", vec![(Know, 0.6), (True, 0.4)]),
        ("clear", vec![(See, 0.4), (Know, 0.4), (Good, 0.2)]),
        // === More nouns ===
        ("world", vec![(Place, 0.6), (All, 0.4), (Big, 0.3)]),
        ("country", vec![(Place, 0.7), (Big, 0.3), (Someone, 0.2)]),
        ("city", vec![(Place, 0.7), (Big, 0.3), (Many, 0.2)]),
        ("road", vec![(Place, 0.5), (Move, 0.4), (Far, 0.2)]),
        ("mountain", vec![(Place, 0.4), (Above, 0.5), (Big, 0.4)]),
        ("river", vec![(Something, 0.5), (Move, 0.4), (Place, 0.3)]),
        ("sea", vec![(Place, 0.4), (Big, 0.5), (Far, 0.3)]),
        ("rain", vec![(Something, 0.4), (Above, 0.3), (Move, 0.3)]),
        ("wind", vec![(Something, 0.4), (Move, 0.5), (Feel, 0.2)]),
        ("night", vec![(Time, 0.6), (Not, 0.2), (See, 0.2)]),
        ("day", vec![(Time, 0.7), (See, 0.2)]),
        ("morning", vec![(Time, 0.7), (Before, 0.2)]),
        ("evening", vec![(Time, 0.7), (After, 0.2)]),
        ("year", vec![(Time, 0.8), (ALongTime, 0.3)]),
        ("month", vec![(Time, 0.8)]),
        ("week", vec![(Time, 0.8)]),
        ("today", vec![(Time, 0.5), (Now, 0.6)]),
        ("tomorrow", vec![(Time, 0.5), (After, 0.5)]),
        ("yesterday", vec![(Time, 0.5), (Before, 0.5)]),
        ("head", vec![(Body, 0.7), (Part, 0.4), (Above, 0.2)]),
        ("hand", vec![(Body, 0.7), (Part, 0.4), (Touch, 0.3)]),
        ("eye", vec![(Body, 0.6), (Part, 0.3), (See, 0.4)]),
        ("ear", vec![(Body, 0.6), (Part, 0.3), (Hear, 0.4)]),
        ("mouth", vec![(Body, 0.6), (Part, 0.3), (Say, 0.3)]),
        ("heart", vec![(Body, 0.5), (Part, 0.3), (Feel, 0.5)]),
        ("blood", vec![(Body, 0.6), (Inside, 0.3), (Live, 0.3)]),
        ("face", vec![(Body, 0.6), (Part, 0.3), (See, 0.3)]),
        ("foot", vec![(Body, 0.6), (Part, 0.3), (Move, 0.3)]),
        ("arm", vec![(Body, 0.6), (Part, 0.3)]),
        ("bone", vec![(Body, 0.5), (Part, 0.3), (Inside, 0.2)]),
        ("skin", vec![(Body, 0.6), (Part, 0.3), (Touch, 0.3)]),
        // === Prepositions / adverbs ===
        ("with", vec![(TheSame, 0.4), (Near, 0.3)]),
        ("without", vec![(Not, 0.6), (TheSame, 0.2)]),
        ("in", vec![(Inside, 0.8)]),
        ("out", vec![(Not, 0.3), (Inside, 0.3), (Move, 0.2)]),
        ("up", vec![(Above, 0.7), (Move, 0.3)]),
        ("down", vec![(Below, 0.7), (Move, 0.3)]),
        ("away", vec![(Far, 0.6), (Move, 0.3)]),
        ("back", vec![(Before, 0.3), (Move, 0.3), (Place, 0.2)]),
        ("again", vec![(TheSame, 0.5), (More, 0.3)]),
        ("still", vec![(TheSame, 0.4), (Now, 0.3)]),
        ("already", vec![(Before, 0.5), (Now, 0.3)]),
        ("always", vec![(All, 0.5), (Time, 0.5)]),
        ("often", vec![(Many, 0.5), (Time, 0.4)]),
        ("sometimes", vec![(Some, 0.5), (Time, 0.5)]),
        ("also", vec![(More, 0.5), (TheSame, 0.3)]),
        ("only", vec![(One, 0.5), (Not, 0.2)]),
        ("just", vec![(One, 0.3), (Now, 0.2)]),
        ("too", vec![(More, 0.5), (Much, 0.3)]),
        ("enough", vec![(Some, 0.4), (Good, 0.3)]),
        ("almost", vec![(Near, 0.4), (Not, 0.2)]),
        ("about", vec![(Like, 0.3), (Near, 0.3)]),
        ("around", vec![(Side, 0.4), (Near, 0.3)]),
        ("between", vec![(Side, 0.4), (Two, 0.3)]),
        ("through", vec![(Inside, 0.4), (Move, 0.4)]),
        ("across", vec![(Side, 0.3), (Move, 0.4), (Far, 0.2)]),
        ("until", vec![(Time, 0.5), (Before, 0.3)]),
        ("since", vec![(Time, 0.4), (Before, 0.4), (Because, 0.2)]),
        ("during", vec![(Time, 0.5), (ForSomeTime, 0.3)]),
        ("while", vec![(Time, 0.5), (TheSame, 0.2)]),
        // === More derived ===
        ("idea", vec![(Think, 0.7), (Thing, 0.3)]),
        ("question", vec![(Say, 0.4), (Want, 0.3), (Know, 0.4)]),
        ("answer", vec![(Say, 0.5), (Know, 0.4)]),
        ("problem", vec![(Bad, 0.4), (Think, 0.3), (Thing, 0.3)]),
        ("money", vec![(Thing, 0.5), (Want, 0.4), (Much, 0.2)]),
        ("power", vec![(Can, 0.6), (Much, 0.4), (Big, 0.2)]),
        ("war", vec![(Do, 0.4), (Bad, 0.5), (Die, 0.4), (Many, 0.2)]),
        ("peace", vec![(Good, 0.5), (Not, 0.2), (Bad, 0.1)]),
        ("king", vec![(Someone, 0.5), (Big, 0.3), (Can, 0.3)]),
        ("god", vec![(Someone, 0.4), (Big, 0.3), (Can, 0.4), (Above, 0.3)]),
        ("spirit", vec![(Something, 0.4), (Feel, 0.3), (Live, 0.3)]),
        ("mind", vec![(Think, 0.6), (Part, 0.2), (Body, 0.2)]),
    ];

    let mut map = HashMap::with_capacity(entries.len());
    for (word, primes) in entries {
        map.insert(word.to_string(), primes);
    }
    map
}

/// Decompose text into weighted NSM primes via word-level matching.
pub fn nsm_decompose(text: &str) -> NsmDecomposition {
    let vocab = build_vocab_map();
    let mut weights = [0.0f32; 74];

    for word in text.split_whitespace() {
        let lower = word.to_lowercase();
        // Strip basic punctuation
        let clean: String = lower.chars().filter(|c| c.is_alphabetic()).collect();
        if clean.is_empty() {
            continue;
        }
        if let Some(primes) = vocab.get(&clean) {
            for &(prime, weight) in primes {
                weights[prime as u8 as usize] += weight;
            }
        }
    }

    // Normalize weights to sum to 1.0 via crate::simd.
    // 74 elements = 4×F32x16 (64) + 10 scalar remainder.
    let weight_sum = {
        use crate::simd::F32x16;
        let mut simd_sum = F32x16::splat(0.0);
        for chunk in weights[..64].chunks_exact(16) {
            simd_sum = simd_sum + F32x16::from_slice(chunk);
        }
        let mut s: f32 = simd_sum.reduce_sum();
        for &w in &weights[64..74] {
            s += w;
        }
        if s > 0.0 {
            let inv = F32x16::splat(1.0 / s);
            for chunk in weights[..64].chunks_exact_mut(16) {
                let v = F32x16::from_slice(chunk) * inv;
                v.copy_to_slice(chunk);
            }
            let inv_s = 1.0 / s;
            for w in weights[64..74].iter_mut() {
                *w *= inv_s;
            }
        }
        s
    };

    // Determine dominant primes (weight > threshold)
    let threshold = if weight_sum > 0.0 { 1.0 / 74.0 } else { 0.0 };
    let dominant: Vec<NsmPrime> = ALL_PRIMES
        .iter()
        .filter(|p| weights[**p as u8 as usize] > threshold)
        .copied()
        .collect();

    NsmDecomposition { weights, dominant }
}

/// Encode an NSM decomposition as a 10000-bit binary vector (1250 bytes).
///
/// For each prime with weight > 0, hash prime_index with blake3 to produce
/// a deterministic bit pattern, then XOR into result for primes whose
/// normalised weight exceeds 0.5 of the max weight (or any nonzero weight
/// when only one prime is present).
pub fn nsm_to_fingerprint(decomp: &NsmDecomposition) -> [u8; 1250] {
    let mut result = [0u8; 1250];

    let max_w = decomp.weights.iter().cloned().fold(0.0f32, f32::max);
    if max_w == 0.0 {
        return result;
    }
    let threshold = max_w * 0.5;

    for (i, &w) in decomp.weights.iter().enumerate() {
        if w < threshold {
            continue;
        }
        // Hash the prime index to get a deterministic 1250-byte pattern
        let hash_input = (i as u32).to_le_bytes();
        let mut pattern = [0u8; 1250];
        // Use blake3 in extended-output mode to fill 1250 bytes
        let mut hasher = blake3::Hasher::new();
        hasher.update(&hash_input);
        let mut reader = hasher.finalize_xof();
        reader.fill(&mut pattern);

        // XOR 1250 bytes via crate::simd::U8x64.
        // 1250 = 19×64 (1216) + 34 scalar remainder.
        {
            use crate::simd::U8x64;
            let chunks = 1250 / 64; // 19
            for c in 0..chunks {
                let off = c * 64;
                let vr = U8x64::from_slice(&result[off..off + 64]);
                let vp = U8x64::from_slice(&pattern[off..off + 64]);
                let xored = vr ^ vp;
                xored.copy_to_slice(&mut result[off..off + 64]);
            }
            // Scalar remainder (34 bytes).
            for j in (chunks * 64)..1250 {
                result[j] ^= pattern[j];
            }
        }
    }

    result
}

/// Cosine similarity between two NSM decompositions via `crate::simd`.
///
/// 74 elements = 4×F32x16 (64) + 10 scalar remainder.
/// Three accumulations in one pass: dot product, magnitude_a², magnitude_b².
pub fn nsm_similarity(a: &NsmDecomposition, b: &NsmDecomposition) -> f32 {
    use crate::simd::F32x16;

    let mut sdot = F32x16::splat(0.0);
    let mut smag_a = F32x16::splat(0.0);
    let mut smag_b = F32x16::splat(0.0);

    // SIMD: first 64 elements (4 × 16 lanes).
    for i in (0..64).step_by(16) {
        let va = F32x16::from_slice(&a.weights[i..i + 16]);
        let vb = F32x16::from_slice(&b.weights[i..i + 16]);
        sdot = va.mul_add(vb, sdot);
        smag_a = va.mul_add(va, smag_a);
        smag_b = vb.mul_add(vb, smag_b);
    }

    let mut dot = sdot.reduce_sum();
    let mut mag_a = smag_a.reduce_sum();
    let mut mag_b = smag_b.reduce_sum();

    // Scalar: remaining 10 elements (indices 64..74).
    for i in 64..74 {
        dot += a.weights[i] * b.weights[i];
        mag_a += a.weights[i] * a.weights[i];
        mag_b += b.weights[i] * b.weights[i];
    }

    let denom = (mag_a * mag_b).sqrt();
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// Built-in vocabulary of common words mapped to NSM primes (≥200 entries).
pub fn nsm_vocabulary() -> Vec<NsmEntry> {
    build_vocab_map()
        .into_iter()
        .map(|(word, primes)| NsmEntry { word, primes })
        .collect()
}

/// Lookup a word in the built-in vocabulary.
pub fn nsm_lookup(word: &str) -> Option<Vec<(NsmPrime, f32)>> {
    let vocab = build_vocab_map();
    vocab.get(&word.to_lowercase()).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_prime_count() {
        assert_eq!(NsmPrime::all().len(), 74);
    }

    #[test]
    fn test_prime_names_unique() {
        let names: HashSet<&str> = NsmPrime::all().iter().map(|p| p.name()).collect();
        assert_eq!(names.len(), 74);
    }

    #[test]
    fn test_category_coverage() {
        let categories: HashSet<NsmCategory> =
            NsmPrime::all().iter().map(|p| p.category()).collect();
        // 16 categories
        assert_eq!(categories.len(), 16);
        assert!(categories.contains(&NsmCategory::Substantive));
        assert!(categories.contains(&NsmCategory::Relational));
        assert!(categories.contains(&NsmCategory::Determiner));
        assert!(categories.contains(&NsmCategory::Quantifier));
        assert!(categories.contains(&NsmCategory::Evaluator));
        assert!(categories.contains(&NsmCategory::Descriptor));
        assert!(categories.contains(&NsmCategory::Mental));
        assert!(categories.contains(&NsmCategory::Speech));
        assert!(categories.contains(&NsmCategory::Action));
        assert!(categories.contains(&NsmCategory::Existence));
        assert!(categories.contains(&NsmCategory::Life));
        assert!(categories.contains(&NsmCategory::Time));
        assert!(categories.contains(&NsmCategory::Space));
        assert!(categories.contains(&NsmCategory::Logical));
        assert!(categories.contains(&NsmCategory::Intensifier));
        assert!(categories.contains(&NsmCategory::Similarity));
    }

    #[test]
    fn test_decompose_simple() {
        let d = nsm_decompose("I think");
        assert!(d.weights[NsmPrime::I as u8 as usize] > 0.0);
        assert!(d.weights[NsmPrime::Think as u8 as usize] > 0.0);
    }

    #[test]
    fn test_decompose_empty() {
        let d = nsm_decompose("");
        for &w in d.weights.iter() {
            assert_eq!(w, 0.0);
        }
        assert!(d.dominant.is_empty());
    }

    #[test]
    fn test_to_fingerprint_deterministic() {
        let d = nsm_decompose("I think something good");
        let fp1 = nsm_to_fingerprint(&d);
        let fp2 = nsm_to_fingerprint(&d);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_similarity_self_one() {
        let d = nsm_decompose("I think something good");
        let sim = nsm_similarity(&d, &d);
        assert!((sim - 1.0).abs() < 1e-5, "self-similarity should be ~1.0, got {}", sim);
    }

    #[test]
    fn test_similarity_orthogonal() {
        // Create two decompositions with completely different primes
        let mut a = NsmDecomposition {
            weights: [0.0; 74],
            dominant: vec![],
        };
        let mut b = NsmDecomposition {
            weights: [0.0; 74],
            dominant: vec![],
        };
        a.weights[NsmPrime::I as u8 as usize] = 1.0;
        b.weights[NsmPrime::Where as u8 as usize] = 1.0;
        let sim = nsm_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "orthogonal similarity should be ~0.0, got {}", sim);
    }

    #[test]
    fn test_vocabulary_has_entries() {
        let vocab = nsm_vocabulary();
        assert!(
            vocab.len() >= 200,
            "vocabulary should have ≥200 entries, got {}",
            vocab.len()
        );
    }

    #[test]
    fn test_lookup_common_words() {
        assert!(nsm_lookup("cat").is_some(), "cat should be in vocabulary");
        assert!(nsm_lookup("dog").is_some(), "dog should be in vocabulary");
        assert!(nsm_lookup("think").is_some(), "think should be in vocabulary");
    }

    #[test]
    fn test_cat_dog_share_primes() {
        let cat = nsm_lookup("cat").unwrap();
        let dog = nsm_lookup("dog").unwrap();
        let cat_primes: HashSet<NsmPrime> = cat.iter().map(|(p, _)| *p).collect();
        let dog_primes: HashSet<NsmPrime> = dog.iter().map(|(p, _)| *p).collect();
        assert!(
            cat_primes.contains(&NsmPrime::Something),
            "cat should have SOMETHING"
        );
        assert!(
            cat_primes.contains(&NsmPrime::Live),
            "cat should have LIVE"
        );
        assert!(
            dog_primes.contains(&NsmPrime::Something),
            "dog should have SOMETHING"
        );
        assert!(
            dog_primes.contains(&NsmPrime::Live),
            "dog should have LIVE"
        );
        // Shared primes
        let shared: HashSet<_> = cat_primes.intersection(&dog_primes).collect();
        assert!(
            shared.len() >= 2,
            "cat and dog should share ≥2 primes, shared: {:?}",
            shared
        );
    }

    #[test]
    fn test_from_name_roundtrip() {
        for prime in NsmPrime::all() {
            let name = prime.name();
            let recovered = NsmPrime::from_name(name);
            assert_eq!(
                recovered,
                Some(*prime),
                "roundtrip failed for {:?} (name={})",
                prime,
                name
            );
        }
    }
}

// ============================================================================
// DeepNSM Evaluation Pipeline — transcoded from Python DeepNSM
// ============================================================================

/// The full NSM primes set including multi-word primes (from Python utils.py).
static NSM_PRIMES_SET: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "i", "you", "someone", "people", "something", "thing", "body", "kind", "part",
        "this", "the same", "other", "else", "another", "one", "two", "some", "all",
        "much", "many", "little", "few", "good", "bad", "big", "small", "think", "know",
        "want", "don't want", "feel", "see", "hear", "say", "words", "true", "do",
        "happen", "move", "there", "is", "be", "mine", "live", "die", "when", "time",
        "now", "before", "after", "a long time", "a short time", "for some time",
        "moment", "where", "place", "here", "above", "below", "far", "near", "side",
        "inside", "touch", "not", "maybe", "can", "because", "if", "very", "more",
        "like", "as", "way", "said",
    ].into_iter().collect()
});

/// Check if a word is an NSM semantic prime.
pub fn is_nsm_prime(word: &str) -> bool {
    NSM_PRIMES_SET.contains(word.to_lowercase().as_str())
}

/// English stopwords excluding NSM primes. `LazyLock` one-time init.
static STOP_WORDS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    let sw: HashSet<&str> = [
        "a", "an", "and", "are", "at", "been", "but", "by", "did", "does",
        "doing", "down", "during", "each", "for", "from", "further", "had",
        "has", "having", "he", "her", "herself", "him", "himself", "his",
        "how", "in", "into", "it", "its", "itself", "just", "me", "my",
        "myself", "no", "nor", "of", "off", "on", "once", "only", "or",
        "our", "ours", "ourselves", "out", "over", "own", "re", "s", "she",
        "should", "so", "such", "t", "than", "that", "the", "their",
        "theirs", "them", "themselves", "then", "these", "they", "those",
        "through", "to", "too", "under", "until", "up", "ve", "was", "we",
        "were", "what", "which", "while", "who", "whom", "why", "will",
        "with", "won", "would", "your", "yours", "yourself", "yourselves",
    ].into_iter().collect();
    sw.into_iter().filter(|w| !NSM_PRIMES_SET.contains(*w)).collect()
});

/// Check if a word is a stopword (but not an NSM prime).
pub fn is_stop_word(word: &str) -> bool {
    STOP_WORDS.contains(word.to_lowercase().as_str())
}

/// Legal punctuation in NSM explications.
pub const LEGAL_PUNCTUATION: &[char] = &['\'', '.', ',', ':', '!', '?', '"', '\n', '\t', '(', ')', '/'];

// ── Evaluation types ────────────────────────────────────────────────────────

/// A single prediction from a grader model.
#[derive(Clone, Debug)]
pub struct Prediction {
    pub prediction: String,
    pub answer_logprob: f32,
    pub answer_ranks: Vec<usize>,
    pub is_match: bool,
    pub lines_removed: usize,
}

impl Prediction {
    pub fn new(prediction: &str) -> Self {
        Self { prediction: prediction.to_string(), answer_logprob: 0.0, answer_ranks: Vec::new(), is_match: false, lines_removed: 0 }
    }
}

/// Substitutability score from one grader model.
#[derive(Clone, Debug)]
pub struct SubstitutabilityScore {
    pub model: String,
    pub baselines: Vec<Prediction>,
    pub exp_baselines: Vec<Prediction>,
    pub minimality: Vec<Vec<Prediction>>,
    pub entailments: Vec<Vec<Prediction>>,
    pub adj_score: f32,
    pub avg_delta_log: f32,
    pub avg_min_delta_log: f32,
    pub avg_ent_delta_log: f32,
    pub total_match: usize,
}

impl SubstitutabilityScore {
    pub fn new(model: &str) -> Self {
        Self { model: model.to_string(), baselines: Vec::new(), exp_baselines: Vec::new(), minimality: Vec::new(), entailments: Vec::new(), adj_score: 0.0, avg_delta_log: 0.0, avg_min_delta_log: 0.0, avg_ent_delta_log: 0.0, total_match: 0 }
    }
}

/// An NSM explication with legality scoring.
#[derive(Clone, Debug)]
pub struct Explication {
    pub text: String,
    pub target_word: String,
    pub length: usize,
    pub primes: usize,
    pub stop_words_count: usize,
    pub molecules: usize,
    pub unique_molecules: usize,
    pub uses_original_word: bool,
    pub primes_ratio: f32,
    pub molecules_ratio: f32,
    pub sub_scores: Vec<SubstitutabilityScore>,
    pub avg_delta: f32,
    pub avg_delta_min: f32,
    pub avg_delta_ent: f32,
    pub score_exp: f32,
    pub total_score: f32,
}

impl Explication {
    pub fn new(text: &str) -> Self {
        Self {
            text: text.to_string(), target_word: String::new(),
            length: 0, primes: 0, stop_words_count: 0, molecules: 0,
            unique_molecules: 0, uses_original_word: false, primes_ratio: 0.0,
            molecules_ratio: 0.0, sub_scores: Vec::new(), avg_delta: 0.0,
            avg_delta_min: 0.0, avg_delta_ent: 0.0, score_exp: 0.0, total_score: 0.0,
        }
    }

    /// Score legality against a target word (circularity via stem matching).
    pub fn legality_score(&mut self, word: &str) {
        let clean: String = self.text.to_lowercase().chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace()).collect();
        let tokens: Vec<&str> = clean.split_whitespace().collect();
        self.target_word = word.to_string();
        self.length = tokens.len();
        self.primes = tokens.iter().filter(|t| is_nsm_prime(t)).count();
        self.stop_words_count = tokens.iter().filter(|t| is_stop_word(t)).count();
        let mols: Vec<&&str> = tokens.iter().filter(|t| !is_nsm_prime(t) && !is_stop_word(t)).collect();
        self.molecules = mols.len();
        self.unique_molecules = mols.iter().collect::<HashSet<_>>().len();
        let wl = word.to_lowercase();
        let stem = if wl.len() >= 4 { &wl[..4] } else { &wl };
        self.uses_original_word = tokens.iter().any(|t| *t == wl || (t.len() >= 4 && t.starts_with(stem)));
        self.primes_ratio = if self.length > 0 { self.primes as f32 / self.length as f32 } else { 0.0 };
        self.molecules_ratio = if self.length > 0 { self.molecules as f32 / self.length as f32 } else { 0.0 };
    }

    /// Compute averages from substitutability sub-scores.
    pub fn calculate_averages(&mut self) {
        if self.sub_scores.is_empty() { return; }
        let n = self.sub_scores.len() as f32;
        self.avg_delta = self.sub_scores.iter().map(|s| s.avg_delta_log).sum::<f32>() / n;
        self.avg_delta_min = self.sub_scores.iter().map(|s| s.avg_min_delta_log).sum::<f32>() / n;
        self.avg_delta_ent = self.sub_scores.iter().map(|s| s.avg_ent_delta_log).sum::<f32>() / n;
        self.score_exp = self.sub_scores.iter().map(|s| s.adj_score).sum::<f32>() / n;
        self.total_score = if !self.uses_original_word {
            2.0 * (self.score_exp + 10.0 * self.primes_ratio - 10.0 * self.molecules_ratio)
        } else { 0.0 };
    }

    /// Truncated versions with lines removed from the end.
    pub fn get_truncated(&self, max_lines_remove: usize) -> Vec<Explication> {
        let lines: Vec<&str> = self.text.trim().lines().collect();
        (0..max_lines_remove.min(lines.len()))
            .map(|i| Explication::new(&lines[..lines.len() - (i + 1)].join("\n")))
            .collect()
    }
}

/// Ambiguous example passage with masked word.
#[derive(Clone, Debug)]
pub struct AmbiguousExample {
    pub text: String,
    pub source: Option<String>,
}

impl AmbiguousExample {
    pub fn new(text: &str) -> Self { Self { text: text.to_string(), source: None } }

    /// Truncated versions removing non-UNK sentences.
    pub fn get_truncated(&self, max_remove: usize) -> Vec<AmbiguousExample> {
        let sents: Vec<&str> = self.text.split('.').map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
        let non_unk: Vec<usize> = sents.iter().enumerate().filter(|(_, s)| !s.contains("<UNK>")).map(|(i, _)| i).collect();
        (0..max_remove.min(non_unk.len())).map(|i| {
            let exclude: HashSet<usize> = non_unk[..=i].iter().copied().collect();
            let kept: Vec<&str> = sents.iter().enumerate().filter(|(j, _)| !exclude.contains(j)).map(|(_, s)| *s).collect();
            AmbiguousExample::new(&kept.join(". "))
        }).collect()
    }
}

/// Aggregated model evaluation result.
#[derive(Clone, Debug)]
pub struct ModelResult {
    pub model_name: String,
    pub num_examples: usize,
    pub explications: Vec<Explication>,
    pub avg_primes_ratio: f32,
    pub avg_molecules_ratio: f32,
    pub avg_total_score: f32,
}

impl ModelResult {
    pub fn new(model_name: &str) -> Self {
        Self { model_name: model_name.to_string(), num_examples: 0, explications: Vec::new(), avg_primes_ratio: 0.0, avg_molecules_ratio: 0.0, avg_total_score: 0.0 }
    }
    pub fn calculate_averages(&mut self) {
        let n = self.explications.len() as f32;
        if n == 0.0 { return; }
        self.avg_primes_ratio = self.explications.iter().map(|e| e.primes_ratio).sum::<f32>() / n;
        self.avg_molecules_ratio = self.explications.iter().map(|e| e.molecules_ratio).sum::<f32>() / n;
        self.avg_total_score = self.explications.iter().map(|e| e.total_score).sum::<f32>() / n;
    }
}

// ── CAM-PQ bridge ───────────────────────────────────────────────────────────

/// Load DeepNSM codebook (`codebook_pq.bin`) into ndarray's CamCodebook.
pub fn load_nsm_codebook(codebook_bytes: &[u8]) -> super::cam_pq::CamCodebook {
    use super::cam_pq::{CamCodebook, SubspaceCodebook, NUM_CENTROIDS, NUM_SUBSPACES};
    let expected = NUM_SUBSPACES * NUM_CENTROIDS * 16 * 4;
    assert_eq!(codebook_bytes.len(), expected, "codebook_pq.bin: expected {expected} bytes, got {}", codebook_bytes.len());
    let mut codebooks: Vec<SubspaceCodebook> = Vec::with_capacity(NUM_SUBSPACES);
    for s in 0..NUM_SUBSPACES {
        let mut centroids = Vec::with_capacity(NUM_CENTROIDS);
        for c in 0..NUM_CENTROIDS {
            let mut centroid = Vec::with_capacity(16);
            for d in 0..16 {
                let off = (s * NUM_CENTROIDS * 16 + c * 16 + d) * 4;
                centroid.push(f32::from_le_bytes([codebook_bytes[off], codebook_bytes[off+1], codebook_bytes[off+2], codebook_bytes[off+3]]));
            }
            centroids.push(centroid);
        }
        codebooks.push(SubspaceCodebook { centroids, subspace_dim: 16 });
    }
    CamCodebook { codebooks: codebooks.try_into().unwrap(), total_dim: 96, subspace_dim: 16 }
}

/// Load CAM codes (`cam_codes.bin`): N words × 6 bytes.
pub fn load_cam_codes(bytes: &[u8]) -> Vec<super::cam_pq::CamFingerprint> {
    assert_eq!(bytes.len() % 6, 0);
    bytes.chunks_exact(6).map(|c| { let mut fp = [0u8; 6]; fp.copy_from_slice(c); fp }).collect()
}

// ── 36-bit SPO triple ───────────────────────────────────────────────────────

/// 36-bit SPO triple packed in u64. 12-bit subject + predicate + object.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SpoTriple { packed: u64 }

impl SpoTriple {
    pub fn new(subject: u16, predicate: u16, object: u16) -> Self {
        debug_assert!(subject < 4096 && predicate < 4096 && object < 4096);
        Self { packed: ((subject as u64) << 24) | ((predicate as u64) << 12) | object as u64 }
    }
    pub fn subject(&self) -> u16 { ((self.packed >> 24) & 0xFFF) as u16 }
    pub fn predicate(&self) -> u16 { ((self.packed >> 12) & 0xFFF) as u16 }
    pub fn object(&self) -> u16 { (self.packed & 0xFFF) as u16 }
}

// ── Prompt templates ────────────────────────────────────────────────────────

/// NSM explication system instruction.
pub const NSM_EXPLICATION_SYS_INST: &str = "You are a linguist specializing in semantic analysis using the Natural Semantic Metalanguage (NSM) approach. NSM reduces lexicons to universal semantic primes. Paraphrase the word's meaning using NSM primes.";

/// Recovery prompt: predict masked word.
pub const RECOVERY_PROMPT_SYS_INST: &str = "Read the passage with a missing word indicated by <UNK>. Predict the missing word. Output only your prediction.";

/// Chat message for prompt construction.
#[derive(Clone, Debug)]
pub struct ChatMessage { pub role: String, pub content: String }

/// Build explication prompt with optional few-shot.
pub fn build_explication_prompt(word: &str, examples: &[&str], few_shot: &[(String, String)], max: Option<usize>) -> Vec<ChatMessage> {
    let mut msgs = vec![ChatMessage { role: "system".into(), content: NSM_EXPLICATION_SYS_INST.into() }];
    for (u, a) in &few_shot[..max.unwrap_or(few_shot.len()).min(few_shot.len())] {
        msgs.push(ChatMessage { role: "user".into(), content: u.clone() });
        msgs.push(ChatMessage { role: "assistant".into(), content: a.clone() });
    }
    msgs.push(ChatMessage { role: "user".into(), content: format!("Word: {word}\nExamples:\n{}\nParaphrase:", examples.join("\n\n")) });
    msgs
}

/// Build recovery prompt for substitutability testing.
pub fn build_recover_prompt(ambig: &AmbiguousExample, exp: Option<&Explication>) -> Vec<ChatMessage> {
    let user = match exp {
        Some(e) => format!("Passage: {}\nParaphrase:\n{}\nMissing word:", ambig.text, e.text),
        None => format!("Passage: {}\nMissing Word:", ambig.text),
    };
    vec![
        ChatMessage { role: "system".into(), content: RECOVERY_PROMPT_SYS_INST.into() },
        ChatMessage { role: "user".into(), content: user },
    ]
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod eval_tests {
    use super::*;

    #[test]
    fn test_is_nsm_prime() {
        assert!(is_nsm_prime("think"));
        assert!(is_nsm_prime("THINK"));
        assert!(!is_nsm_prime("journalism"));
    }

    #[test]
    fn test_is_stop_word() {
        assert!(is_stop_word("the"));
        assert!(!is_stop_word("think")); // NSM prime, not stopword
    }

    #[test]
    fn test_explication_legality() {
        let mut exp = Explication::new("someone can feel something good because of this");
        exp.legality_score("happy");
        assert!(exp.primes_ratio > 0.5);
        assert!(!exp.uses_original_word);
    }

    #[test]
    fn test_explication_circularity() {
        let mut exp = Explication::new("feeling happy about something");
        exp.legality_score("happy");
        assert!(exp.uses_original_word);
    }

    #[test]
    fn test_explication_averages() {
        let mut exp = Explication::new("test");
        exp.primes_ratio = 0.6;
        exp.molecules_ratio = 0.1;
        let mut s = SubstitutabilityScore::new("g");
        s.adj_score = 5.0;
        exp.sub_scores.push(s);
        exp.calculate_averages();
        assert!(exp.total_score > 0.0);
    }

    #[test]
    fn test_truncated() {
        let exp = Explication::new("line one\nline two\nline three");
        assert_eq!(exp.get_truncated(2).len(), 2);
    }

    #[test]
    fn test_ambiguous_truncated() {
        let a = AmbiguousExample::new("The cat sat. The <UNK> was red. It was sunny.");
        let t = a.get_truncated(2);
        assert!(!t.is_empty());
    }

    #[test]
    fn test_model_result() {
        let mut r = ModelResult::new("m");
        let mut e = Explication::new("t");
        e.primes_ratio = 0.5;
        e.total_score = 8.0;
        r.explications.push(e);
        r.calculate_averages();
        assert!((r.avg_total_score - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_spo_triple() {
        let t = SpoTriple::new(671, 2943, 95);
        assert_eq!(t.subject(), 671);
        assert_eq!(t.predicate(), 2943);
        assert_eq!(t.object(), 95);
    }

    #[test]
    fn test_cam_codes_load() {
        let bytes = vec![1,2,3,4,5,6, 7,8,9,10,11,12];
        let codes = load_cam_codes(&bytes);
        assert_eq!(codes.len(), 2);
        assert_eq!(codes[0], [1,2,3,4,5,6]);
    }

    #[test]
    fn test_prompt_building() {
        let msgs = build_explication_prompt("happy", &["I am happy"], &[], None);
        assert_eq!(msgs.len(), 2);
        assert!(msgs[1].content.contains("happy"));
    }
}
