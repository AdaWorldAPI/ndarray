//! TEKAMOLO: sentence → Temporal-Kausal-Modal-Lokal slot decomposition.
//!
//! German sentence structure applied universally:
//! Subject + Predicate + Object + TEKAMOLO adverbials.
//! Pure parsing — no NARS, no qualia, no VSA. Just structure.

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// TEKAMOLO slot types + core SPO.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TekmoloSlot {
    Subject,
    Predicate,
    Object,
    Temporal,
    Kausal,
    Modal,
    Lokal,
}

impl TekmoloSlot {
    /// TEKAMOLO canonical order: T=0, K=1, M=2, L=3, S=4, P=5, O=6.
    pub fn order_index(&self) -> u8 {
        match self {
            TekmoloSlot::Temporal => 0,
            TekmoloSlot::Kausal => 1,
            TekmoloSlot::Modal => 2,
            TekmoloSlot::Lokal => 3,
            TekmoloSlot::Subject => 4,
            TekmoloSlot::Predicate => 5,
            TekmoloSlot::Object => 6,
        }
    }
}

/// A parsed slot: the slot type + the text fragment.
#[derive(Clone, Debug)]
pub struct SlotEntry {
    pub slot: TekmoloSlot,
    pub text: String,
}

/// SPO extraction result.
#[derive(Clone, Debug)]
pub struct SpoExtraction {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub temporal: Vec<String>,
    pub kausal: Vec<String>,
    pub modal: Vec<String>,
    pub lokal: Vec<String>,
}

/// A crystallized sentence: full decomposition.
#[derive(Clone, Debug)]
pub struct CrystalizedSentence {
    pub original: String,
    pub slots: Vec<SlotEntry>,
    pub spo: SpoExtraction,
}

// ---------------------------------------------------------------------------
// Keyword sets (lazy-initialized via functions)
// ---------------------------------------------------------------------------

fn verbs() -> HashSet<&'static str> {
    [
        "is", "are", "was", "were", "am", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "will", "would", "shall", "should", "can", "could", "may", "might", "must",
        "go", "goes", "went", "come", "came",
        "get", "gets", "got", "make", "makes", "made",
        "take", "takes", "took", "give", "gives", "gave",
        "say", "says", "said", "see", "sees", "saw",
        "know", "knows", "knew", "think", "thinks", "thought",
        "want", "wants", "wanted",
        "run", "runs", "ran", "walk", "walks", "walked",
        "sit", "sits", "sat", "stand", "stands", "stood",
        "eat", "eats", "ate", "drink", "drinks", "drank",
        "sleep", "sleeps", "slept",
        "read", "reads", "write", "writes", "wrote",
        "speak", "speaks", "spoke", "hear", "hears", "heard",
        "feel", "feels", "felt", "live", "lives", "lived",
        "die", "dies", "died", "love", "loves", "loved",
        "hate", "hates", "hated", "play", "plays", "played",
        "work", "works", "worked", "move", "moves", "moved",
        "happen", "happens", "happened", "touch", "touched",
        "left", "flew",
    ]
    .iter()
    .copied()
    .collect()
}

fn temporal_keywords() -> HashSet<&'static str> {
    [
        "yesterday", "today", "tomorrow", "now", "then", "always", "never",
        "before", "after", "when", "while", "during", "already", "soon",
        "later", "recently", "once", "often", "sometimes", "usually",
        "morning", "evening", "night",
    ]
    .iter()
    .copied()
    .collect()
}

fn kausal_keywords() -> HashSet<&'static str> {
    [
        "because", "since", "therefore", "hence", "thus", "so",
        "consequently", "due", "reason", "cause", "why",
    ]
    .iter()
    .copied()
    .collect()
}

fn modal_keywords() -> HashSet<&'static str> {
    [
        "quickly", "slowly", "carefully", "easily", "well", "badly",
        "hard", "gently", "loudly", "quietly", "happily", "sadly",
        "with", "without", "by",
    ]
    .iter()
    .copied()
    .collect()
}

fn lokal_keywords() -> HashSet<&'static str> {
    [
        "here", "there", "above", "below", "near", "far",
        "inside", "outside", "between", "behind", "front",
        "under", "over", "in", "on", "at", "from", "to",
        "into", "through", "across", "around", "along",
        "up", "down",
    ]
    .iter()
    .copied()
    .collect()
}

fn determiners() -> HashSet<&'static str> {
    ["the", "a", "an", "this", "that", "these", "those"]
        .iter()
        .copied()
        .collect()
}

fn lokal_prepositions() -> HashSet<&'static str> {
    [
        "in", "on", "at", "from", "to", "into", "through", "across",
        "around", "along", "up", "down", "under", "over", "between",
        "behind", "above", "below",
    ]
    .iter()
    .copied()
    .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Strip common trailing punctuation from a token for matching purposes.
fn normalize_token(tok: &str) -> String {
    tok.trim_end_matches(['.', ',', '!', '?', ';', ':'])
       .to_lowercase()
}

/// Classify a single (normalized) token into an adverbial slot, if any.
fn classify_adverbial(norm: &str) -> Option<TekmoloSlot> {
    if temporal_keywords().contains(norm) {
        Some(TekmoloSlot::Temporal)
    } else if kausal_keywords().contains(norm) {
        Some(TekmoloSlot::Kausal)
    } else if modal_keywords().contains(norm) {
        Some(TekmoloSlot::Modal)
    } else if lokal_keywords().contains(norm) {
        Some(TekmoloSlot::Lokal)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Core functions
// ---------------------------------------------------------------------------

/// Rule-based TEKAMOLO parser.
///
/// Strategy:
/// 1. Tokenize by whitespace.
/// 2. Identify predicate (first verb-like token).
/// 3. Subject = content words before predicate.
/// 4. Remaining tokens after predicate classified as adverbials or object.
/// 5. Prepositional phrases starting with lokal prepositions become Lokal.
pub fn tekamolo_parse(sentence: &str) -> Vec<SlotEntry> {
    let raw_tokens: Vec<&str> = sentence.split_whitespace().collect();
    if raw_tokens.is_empty() {
        return Vec::new();
    }

    let verb_set = verbs();
    let det_set = determiners();
    let lokal_prep = lokal_prepositions();

    // Normalised forms for matching.
    let norms: Vec<String> = raw_tokens.iter().map(|t| normalize_token(t)).collect();

    // --- Pass 1: find predicate index ---
    let pred_idx = norms.iter().position(|n| verb_set.contains(n.as_str()));

    // --- Pass 2: classify every token ---
    // We'll collect (index, SlotEntry) and then sort / deduplicate.
    let mut entries: Vec<SlotEntry> = Vec::new();
    let mut consumed: Vec<bool> = vec![false; raw_tokens.len()];

    // 2a. Mark predicate.
    if let Some(pi) = pred_idx {
        entries.push(SlotEntry {
            slot: TekmoloSlot::Predicate,
            text: raw_tokens[pi].to_string(),
        });
        consumed[pi] = true;
    }

    // 2b. Pre-predicate adverbials (e.g. "Yesterday ...").
    let pred_boundary = pred_idx.unwrap_or(raw_tokens.len());
    for i in 0..pred_boundary {
        if consumed[i] {
            continue;
        }
        if let Some(slot) = classify_adverbial(&norms[i]) {
            entries.push(SlotEntry {
                slot,
                text: raw_tokens[i].to_string(),
            });
            consumed[i] = true;
        }
    }

    // 2c. Subject: unconsumed, non-determiner tokens before predicate.
    let mut subj_words: Vec<&str> = Vec::new();
    for i in 0..pred_boundary {
        if consumed[i] {
            continue;
        }
        if det_set.contains(norms[i].as_str()) {
            continue;
        }
        subj_words.push(raw_tokens[i]);
        consumed[i] = true;
    }
    // Also consume determiners that were part of the subject phrase.
    for i in 0..pred_boundary {
        if !consumed[i] {
            consumed[i] = true; // determiners
        }
    }
    if !subj_words.is_empty() {
        entries.push(SlotEntry {
            slot: TekmoloSlot::Subject,
            text: subj_words.join(" "),
        });
    }

    // 2d. Post-predicate tokens: classify adverbials and lokal phrases; rest is object.
    let start = if let Some(pi) = pred_idx { pi + 1 } else { raw_tokens.len() };
    let mut obj_words: Vec<&str> = Vec::new();
    let mut i = start;
    while i < raw_tokens.len() {
        if consumed[i] {
            i += 1;
            continue;
        }

        // Check for adverbial keyword.
        if let Some(slot) = classify_adverbial(&norms[i]) {
            // If it's a lokal preposition, gather the whole prepositional phrase.
            if slot == TekmoloSlot::Lokal && lokal_prep.contains(norms[i].as_str()) {
                let phrase_start = i;
                i += 1;
                // Gather until next verb, adverbial keyword (non-lokal), or end.
                while i < raw_tokens.len() {
                    let n = &norms[i];
                    if verb_set.contains(n.as_str()) {
                        break;
                    }
                    // If it's another adverbial that isn't a determiner or plain noun, stop.
                    if let Some(s) = classify_adverbial(n) {
                        if s != TekmoloSlot::Lokal || lokal_prep.contains(n.as_str()) {
                            break;
                        }
                    }
                    i += 1;
                }
                let phrase: Vec<&str> = raw_tokens[phrase_start..i].to_vec();
                for j in phrase_start..i {
                    consumed[j] = true;
                }
                entries.push(SlotEntry {
                    slot: TekmoloSlot::Lokal,
                    text: phrase.join(" "),
                });
                continue;
            }

            entries.push(SlotEntry {
                slot,
                text: raw_tokens[i].to_string(),
            });
            consumed[i] = true;
            i += 1;
            continue;
        }

        // Not an adverbial — accumulate as object.
        if !det_set.contains(norms[i].as_str()) {
            obj_words.push(raw_tokens[i]);
        }
        consumed[i] = true;
        i += 1;
    }

    if !obj_words.is_empty() {
        entries.push(SlotEntry {
            slot: TekmoloSlot::Object,
            text: obj_words.join(" "),
        });
    }

    entries
}

/// Collect parsed slots into a structured [`SpoExtraction`].
pub fn spo_extract(parsed: &[SlotEntry]) -> SpoExtraction {
    let mut spo = SpoExtraction {
        subject: None,
        predicate: None,
        object: None,
        temporal: Vec::new(),
        kausal: Vec::new(),
        modal: Vec::new(),
        lokal: Vec::new(),
    };

    for entry in parsed {
        match entry.slot {
            TekmoloSlot::Subject => {
                spo.subject = Some(entry.text.clone());
            }
            TekmoloSlot::Predicate => {
                spo.predicate = Some(entry.text.clone());
            }
            TekmoloSlot::Object => {
                spo.object = Some(entry.text.clone());
            }
            TekmoloSlot::Temporal => {
                spo.temporal.push(entry.text.clone());
            }
            TekmoloSlot::Kausal => {
                spo.kausal.push(entry.text.clone());
            }
            TekmoloSlot::Modal => {
                spo.modal.push(entry.text.clone());
            }
            TekmoloSlot::Lokal => {
                spo.lokal.push(entry.text.clone());
            }
        }
    }

    spo
}

/// Full pipeline: parse → extract → crystallize.
pub fn sentence_crystal(text: &str) -> CrystalizedSentence {
    let slots = tekamolo_parse(text);
    let spo = spo_extract(&slots);
    CrystalizedSentence {
        original: text.to_string(),
        slots,
        spo,
    }
}

/// Reorder slots into canonical TEKAMOLO order (T, K, M, L, S, P, O).
pub fn reorder_tekamolo(slots: &[SlotEntry]) -> Vec<SlotEntry> {
    let mut sorted = slots.to_vec();
    sorted.sort_by_key(|e| e.slot.order_index());
    sorted
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_svo() {
        let slots = tekamolo_parse("The cat sat on the mat");
        let spo = spo_extract(&slots);
        assert_eq!(spo.subject.as_deref(), Some("cat"));
        assert_eq!(spo.predicate.as_deref(), Some("sat"));
        // "on the mat" should be classified as Lokal.
        assert!(!spo.lokal.is_empty(), "expected Lokal slot for 'on the mat'");
        assert!(spo.lokal[0].contains("mat"));
    }

    #[test]
    fn test_temporal_detection() {
        let slots = tekamolo_parse("Yesterday the dog ran");
        let spo = spo_extract(&slots);
        assert!(!spo.temporal.is_empty(), "expected Temporal slot");
        assert_eq!(normalize_token(&spo.temporal[0]), "yesterday");
    }

    #[test]
    fn test_kausal_detection() {
        let slots = tekamolo_parse("He left because it rained");
        let spo = spo_extract(&slots);
        assert!(!spo.kausal.is_empty(), "expected Kausal slot for 'because'");
    }

    #[test]
    fn test_modal_detection() {
        let slots = tekamolo_parse("She spoke quietly");
        let spo = spo_extract(&slots);
        assert!(!spo.modal.is_empty(), "expected Modal slot for 'quietly'");
        assert_eq!(normalize_token(&spo.modal[0]), "quietly");
    }

    #[test]
    fn test_lokal_detection() {
        let slots = tekamolo_parse("The bird flew above the trees");
        let spo = spo_extract(&slots);
        assert!(!spo.lokal.is_empty(), "expected Lokal slot for 'above the trees'");
    }

    #[test]
    fn test_reorder_tekamolo() {
        let slots = vec![
            SlotEntry { slot: TekmoloSlot::Subject, text: "cat".into() },
            SlotEntry { slot: TekmoloSlot::Predicate, text: "sat".into() },
            SlotEntry { slot: TekmoloSlot::Lokal, text: "on the mat".into() },
            SlotEntry { slot: TekmoloSlot::Temporal, text: "yesterday".into() },
        ];
        let reordered = reorder_tekamolo(&slots);
        assert_eq!(reordered[0].slot, TekmoloSlot::Temporal);
        assert_eq!(reordered[1].slot, TekmoloSlot::Lokal);
        assert_eq!(reordered[2].slot, TekmoloSlot::Subject);
        assert_eq!(reordered[3].slot, TekmoloSlot::Predicate);
    }

    #[test]
    fn test_crystal_preserves_original() {
        let input = "The cat sat on the mat";
        let crystal = sentence_crystal(input);
        assert_eq!(crystal.original, input);
    }

    #[test]
    fn test_empty_input() {
        let slots = tekamolo_parse("");
        assert!(slots.is_empty());
    }

    #[test]
    fn test_slot_order_index() {
        assert_eq!(TekmoloSlot::Temporal.order_index(), 0);
        assert_eq!(TekmoloSlot::Kausal.order_index(), 1);
        assert_eq!(TekmoloSlot::Modal.order_index(), 2);
        assert_eq!(TekmoloSlot::Lokal.order_index(), 3);
        assert_eq!(TekmoloSlot::Subject.order_index(), 4);
        assert_eq!(TekmoloSlot::Predicate.order_index(), 5);
        assert_eq!(TekmoloSlot::Object.order_index(), 6);
    }

    #[test]
    fn test_spo_extraction() {
        let slots = vec![
            SlotEntry { slot: TekmoloSlot::Subject, text: "dog".into() },
            SlotEntry { slot: TekmoloSlot::Predicate, text: "ran".into() },
            SlotEntry { slot: TekmoloSlot::Object, text: "fast".into() },
        ];
        let spo = spo_extract(&slots);
        assert_eq!(spo.subject.as_deref(), Some("dog"));
        assert_eq!(spo.predicate.as_deref(), Some("ran"));
        assert_eq!(spo.object.as_deref(), Some("fast"));
        assert!(spo.temporal.is_empty());
        assert!(spo.kausal.is_empty());
        assert!(spo.modal.is_empty());
        assert!(spo.lokal.is_empty());
    }
}
