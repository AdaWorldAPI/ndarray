//! In-memory RDF schema representation for the OGIT ontology bridge.
//!
//! Consumes a slice of [`Triple`]s produced by [`TurtleParser::parse`] and
//! builds an [`OntologySchema`] — the authoritative in-memory view of an OGIT
//! namespace used by [`CognitiveBridge`](super) and PR-X9's basin-codebook
//! encoder.
//!
//! # Design
//! Zero extra dependencies beyond `std`. No `bitvec` crate — family membership
//! is stored as `Vec<bool>` (one bool per leaf index) to keep hot-path
//! iteration dependency-free. Strings are heap-boxed (`Box<str>`) once during
//! construction; all lookups are `O(1)` via `HashMap`.
//!
//! # Example
//! ```
//! use ndarray::hpc::ogit_bridge::turtle_parser::TurtleParser;
//! use ndarray::hpc::ogit_bridge::schema::OntologySchema;
//!
//! let src = r#"
//!     @prefix ogit:  <http://www.purl.org/ogit/> .
//!     @prefix rdfs:  <http://www.w3.org/2000/01/rdf-schema#> .
//!     @prefix rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
//!     ogit:Heel a rdfs:Class .
//! "#;
//! let triples = TurtleParser::parse(src).unwrap();
//! let schema = OntologySchema::from_triples(&triples).unwrap();
//! assert_eq!(schema.entities.len(), 1);
//! ```

#![allow(missing_docs)]

use std::collections::HashMap;
use std::fmt;

use super::turtle_parser::{Triple, TripleNode};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur while building an [`OntologySchema`] from triples.
#[derive(Debug)]
pub enum SchemaError {
    /// A required IRI field was missing on an entity.
    MissingField(&'static str),
    /// An integer field could not be parsed.
    ParseInt(String),
}

impl fmt::Display for SchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchemaError::MissingField(field) => {
                write!(f, "missing required field: {}", field)
            }
            SchemaError::ParseInt(s) => write!(f, "could not parse integer: {}", s),
        }
    }
}

impl std::error::Error for SchemaError {}

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A single RDF property declaration attached to an entity class.
///
/// Covers `ogit:mandatory`, `ogit:optional`, and `ogit:indexed` predicates.
#[derive(Debug, Clone, PartialEq)]
pub struct Property {
    /// The property IRI (e.g. `"ogit:basinSignature"`).
    pub iri: Box<str>,
    /// The declared datatype IRI (e.g. `"xsd:long"`, `"xsd:string"`).
    ///
    /// Defaults to `"xsd:string"` when no `^^datatype` annotation is present.
    pub datatype: Box<str>,
}

/// An OGIT entity class (`rdfs:Class` subject) with its collected predicates.
#[derive(Debug, Clone)]
pub struct EntityClass {
    /// The class IRI (e.g. `"ogit:Heel"`).
    pub iri: Box<str>,
    /// Human-readable label (`rdfs:label`); empty string when absent.
    pub label: Box<str>,
    /// First-observed parent class IRI (`rdfs:subClassOf`); `None` for
    /// root classes. OWL allows a class to declare multiple
    /// `rdfs:subClassOf` targets (multi-inheritance); the second and
    /// later parents land in [`Self::extra_parents`]. Consumers wanting
    /// the full parent set should iterate via [`Self::parents`].
    pub parent: Option<Box<str>>,
    /// Additional parent IRIs beyond the first. Empty for single-parent
    /// classes (the common case in RDFS-style ontologies); non-empty when
    /// the source declares multi-inheritance (common in OWL biomedical
    /// ontologies — FMA, ChEBI, etc.). Order is source order of the
    /// surplus `rdfs:subClassOf` triples; the first parent stays in
    /// [`Self::parent`].
    pub extra_parents: Vec<Box<str>>,
    /// Properties declared with `ogit:mandatory`.
    pub mandatory: Vec<Property>,
    /// Properties declared with `ogit:optional`.
    pub optional: Vec<Property>,
    /// Properties declared with `ogit:indexed`.
    pub indexed: Vec<Property>,
    /// IRIs listed under `ogit:allowed-relates` or `ogit:relates`.
    pub allowed_relates: Vec<Box<str>>,
    /// IRIs listed under `ogit:allowed-belongs` or `ogit:belongs`.
    pub allowed_belongs: Vec<Box<str>>,
}

impl EntityClass {
    /// Iterator over every parent class IRI declared on this entity —
    /// the first-observed [`Self::parent`] (if present) followed by
    /// every IRI in [`Self::extra_parents`]. Empty when the class is
    /// a root.
    ///
    /// Use this in preference to reading `.parent` directly when the
    /// caller's logic should cover multi-inheritance — e.g. transitive
    /// closure walks like [`OntologySchema::is_ancestor`].
    pub fn parents(&self) -> impl Iterator<Item = &str> {
        self.parent
            .as_deref()
            .into_iter()
            .chain(self.extra_parents.iter().map(|s| s.as_ref()))
    }

    fn new(iri: Box<str>) -> Self {
        EntityClass {
            iri,
            label: "".into(),
            parent: None,
            extra_parents: Vec::new(),
            mandatory: Vec::new(),
            optional: Vec::new(),
            indexed: Vec::new(),
            allowed_relates: Vec::new(),
            allowed_belongs: Vec::new(),
        }
    }
}

/// A family bitmap: the set of leaf-class indices that belong to a single
/// heel → hip chain.
///
/// Bit/bool `i` is `true` iff the leaf at global index `i` is a member of
/// this family.  The heel and hip IRIs identify the chain.
///
/// PR-X9's basin-XOR-popcount inner loop iterates only the `bitmap` slice
/// (~16–64 candidates), not the full codebook.
#[derive(Debug, Clone)]
pub struct FamilyBitmap {
    /// Monotonically increasing family identifier (index into
    /// [`OntologySchema::families`]).
    pub family_id: u32,
    /// IRI of the heel ancestor of this family.
    pub heel_iri: Box<str>,
    /// IRI of the hip (direct child of heel) of this family.
    pub hip_iri: Box<str>,
    /// Membership bitmap: `bitmap[i]` is `true` iff leaf at index `i` is in
    /// this family.  Length equals [`OntologySchema::leaf_count`].
    pub bitmap: Vec<bool>,
}

/// The authoritative in-memory representation of one OGIT namespace parsed
/// from its RDF Turtle source files.
///
/// # Lookup pattern
/// ```text
/// let family_id = schema.leaf_to_family[leaf_iri];          // O(1)
/// let family    = &schema.families[family_id as usize];     // O(1)
/// // iterate family.bitmap.iter().enumerate() for candidate leaves
/// ```
#[derive(Debug)]
pub struct OntologySchema {
    /// Namespace name, e.g. `"Cognitive"`.
    pub namespace: Box<str>,
    /// Map from class IRI to [`EntityClass`].
    pub entities: HashMap<Box<str>, EntityClass>,
    /// Family bitmaps indexed by family ID.
    pub families: Vec<FamilyBitmap>,
    /// Leaf IRI → family ID.  O(1) lookup used by PR-X9's encode loop.
    pub leaf_to_family: HashMap<Box<str>, u32>,
    /// Number of distinct heel classes detected.
    pub heel_count: u8,
    /// Number of leaf classes detected.
    pub leaf_count: u32,
}

// ---------------------------------------------------------------------------
// Predicate constants — the known OGIT / RDF predicates we handle.
// All kept as string literals so we can match without allocation.
// ---------------------------------------------------------------------------

// rdf:type synonyms (the lexer expands `a` → `rdf:type`)
const RDF_TYPE: &str = "rdf:type";
// rdfs predicates
const RDFS_CLASS: &str = "rdfs:Class";
const RDFS_SUB_CLASS_OF: &str = "rdfs:subClassOf";
const RDFS_LABEL: &str = "rdfs:label";
// ogit property-scope predicates
const OGIT_MANDATORY: &str = "ogit:mandatory";
const OGIT_OPTIONAL: &str = "ogit:optional";
const OGIT_INDEXED: &str = "ogit:indexed";
// ogit relation predicates
const OGIT_RELATES: &str = "ogit:relates";
const OGIT_BELONGS: &str = "ogit:belongs";
const OGIT_ALLOWED_RELATES: &str = "ogit:allowed-relates";
const OGIT_ALLOWED_BELONGS: &str = "ogit:allowed-belongs";
// ogit hierarchy role predicates (used in instance data)
const OGIT_HEEL: &str = "ogit:heel";
const OGIT_HIP: &str = "ogit:hip";
// ogit datatype predicate (declares the XSD datatype of a property)
const OGIT_DATATYPE: &str = "ogit:datatype";

// Heel / Hip / Twig / Leaf tier labels (short local name fragments that
// indicate a class's role in the four-tier hierarchy).
const TIER_HEEL: &str = "Heel";
const TIER_HIP: &str = "Hip";
const TIER_TWIG: &str = "Twig";
const TIER_LEAF: &str = "Leaf";

// ---------------------------------------------------------------------------
// Builder internals
// ---------------------------------------------------------------------------

/// Extract the local name from a prefixed IRI token.
/// `"ogit:Heel"` → `"Heel"`.  Falls back to returning the full IRI when
/// no colon is present (robustness for bare names or full URIs).
fn local_name(iri: &str) -> &str {
    if let Some(pos) = iri.rfind(':') {
        &iri[pos + 1..]
    } else if let Some(pos) = iri.rfind('/') {
        &iri[pos + 1..]
    } else if let Some(pos) = iri.rfind('#') {
        &iri[pos + 1..]
    } else {
        iri
    }
}

/// Guess whether a class IRI belongs to a given tier by checking whether the
/// local name contains `tier` as a substring (case-sensitive).
fn is_tier(iri: &str, tier: &str) -> bool {
    local_name(iri).contains(tier)
}

/// Extract an IRI string from a [`TripleNode`].  Returns `None` for literals.
fn node_iri<'a>(node: &'a TripleNode<'_>) -> Option<&'a str> {
    match node {
        TripleNode::Iri(s) => Some(s),
        TripleNode::Literal { .. } => None,
    }
}

/// Extract a literal value string from a [`TripleNode`].  Returns `None` for
/// IRIs.
fn node_lit_value<'a>(node: &'a TripleNode<'_>) -> Option<&'a str> {
    match node {
        TripleNode::Literal { value, .. } => Some(value),
        TripleNode::Iri(_) => None,
    }
}

/// Extract the datatype from a literal [`TripleNode`], defaulting to
/// `"xsd:string"` when absent.
fn node_lit_datatype<'a>(node: &'a TripleNode<'a>) -> &'a str {
    match node {
        TripleNode::Literal { datatype: Some(dt), .. } => dt,
        _ => "xsd:string",
    }
}

// ---------------------------------------------------------------------------
// OntologySchema::from_triples
// ---------------------------------------------------------------------------

impl OntologySchema {
    /// Build an [`OntologySchema`] from a slice of RDF triples.
    ///
    /// The triples are produced by [`TurtleParser::parse`].  The function
    /// performs three linear passes over `triples`:
    ///
    /// 1. **Class discovery** — collect every subject that appears as
    ///    `rdf:type rdfs:Class`.
    /// 2. **Predicate collection** — for each known predicate, update the
    ///    corresponding field in the target [`EntityClass`].
    /// 3. **Family construction** — walk the heel → hip chain, enumerate
    ///    leaf classes in DFS order, and build the [`FamilyBitmap`] + the
    ///    `leaf_to_family` reverse lookup.
    ///
    /// # Errors
    /// Returns [`SchemaError`] only if internal integer conversions overflow;
    /// unknown predicates are silently ignored (forward-compatible).
    ///
    /// # Example
    /// ```
    /// use ndarray::hpc::ogit_bridge::turtle_parser::TurtleParser;
    /// use ndarray::hpc::ogit_bridge::schema::OntologySchema;
    ///
    /// let src = r#"
    ///     @prefix ogit: <http://www.purl.org/ogit/> .
    ///     @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    ///     ogit:Hip1 a rdfs:Class ;
    ///         rdfs:subClassOf ogit:Heel1 .
    ///     ogit:Heel1 a rdfs:Class .
    /// "#;
    /// let triples = TurtleParser::parse(src).unwrap();
    /// let schema = OntologySchema::from_triples(&triples).unwrap();
    /// let hip = schema.entities.get("ogit:Hip1").unwrap();
    /// assert_eq!(hip.parent.as_deref(), Some("ogit:Heel1"));
    /// ```
    pub fn from_triples<'a>(triples: &[Triple<'a>]) -> Result<Self, SchemaError> {
        // ---------------------------------------------------------------
        // Pass 1: identify all rdfs:Class subjects
        // ---------------------------------------------------------------
        let mut entities: HashMap<Box<str>, EntityClass> = HashMap::new();

        for triple in triples {
            let subject_iri = match node_iri(&triple.subject) {
                Some(s) => s,
                None => continue,
            };
            let predicate_iri = match node_iri(&triple.predicate) {
                Some(s) => s,
                None => continue,
            };

            if predicate_iri == RDF_TYPE {
                if let Some(obj_iri) = node_iri(&triple.object) {
                    if obj_iri == RDFS_CLASS {
                        entities
                            .entry(subject_iri.into())
                            .or_insert_with(|| EntityClass::new(subject_iri.into()));
                    }
                }
            }
        }

        // ---------------------------------------------------------------
        // Pass 2: collect predicates → update entity fields
        // ---------------------------------------------------------------
        // We also track a per-subject datatype override from ogit:datatype
        // so that property IRIs declared on the same subject can pick it up.
        // Stored as (subject_iri, property_iri) → datatype.
        // In practice the OGIT TTL files list property + datatype on the
        // same subject via semicolons, so we just need to propagate within
        // a single subject's predicate list.
        //
        // Implementation note: We make two sub-passes over `triples` rather
        // than buffering per-subject triples, keeping the allocations minimal.

        // Sub-pass 2a: collect ogit:datatype declarations
        let mut datatype_map: HashMap<(&str, &str), &str> = HashMap::new();
        for triple in triples {
            let subject_iri = match node_iri(&triple.subject) {
                Some(s) => s,
                None => continue,
            };
            let predicate_iri = match node_iri(&triple.predicate) {
                Some(s) => s,
                None => continue,
            };
            if predicate_iri == OGIT_DATATYPE {
                // The object may be a literal (e.g. "xsd:long") or an IRI
                let dt = match &triple.object {
                    TripleNode::Literal { value, .. } => *value,
                    TripleNode::Iri(s) => s,
                };
                // Key: (subject, "ogit:datatype") → dt string
                datatype_map.insert((subject_iri, predicate_iri), dt);
            }
        }

        // Sub-pass 2b: apply all predicates we understand
        for triple in triples {
            let subject_iri = match node_iri(&triple.subject) {
                Some(s) => s,
                None => continue,
            };
            let predicate_iri = match node_iri(&triple.predicate) {
                Some(s) => s,
                None => continue,
            };

            // Only process subjects that are known entity classes
            if !entities.contains_key(subject_iri) {
                continue;
            }

            match predicate_iri {
                RDFS_SUB_CLASS_OF => {
                    if let Some(parent_iri) = node_iri(&triple.object) {
                        if let Some(cls) = entities.get_mut(subject_iri) {
                            // First parent → `parent`; subsequent
                            // parents → `extra_parents` (multi-inheritance
                            // as permitted by OWL; common in biomedical
                            // ontologies like FMA / ChEBI). The previous
                            // behaviour silently overwrote — the second
                            // declared parent won, the first was discarded.
                            if cls.parent.is_none() {
                                cls.parent = Some(parent_iri.into());
                            } else if cls.parent.as_deref() != Some(parent_iri)
                                && !cls.extra_parents.iter().any(|p| p.as_ref() == parent_iri)
                            {
                                cls.extra_parents.push(parent_iri.into());
                            }
                        }
                    }
                }
                RDFS_LABEL => {
                    if let Some(label_val) = node_lit_value(&triple.object) {
                        if let Some(cls) = entities.get_mut(subject_iri) {
                            cls.label = label_val.into();
                        }
                    }
                }
                OGIT_MANDATORY => {
                    if let Some(obj_iri) = node_iri(&triple.object) {
                        let dt = datatype_map
                            .get(&(subject_iri, OGIT_DATATYPE))
                            .copied()
                            .unwrap_or("xsd:string");
                        let prop = Property {
                            iri: obj_iri.into(),
                            datatype: dt.into(),
                        };
                        if let Some(cls) = entities.get_mut(subject_iri) {
                            cls.mandatory.push(prop);
                        }
                    } else if let Some(val) = node_lit_value(&triple.object) {
                        // Literal property IRI (unusual but guard)
                        let dt = node_lit_datatype(&triple.object);
                        let prop = Property {
                            iri: val.into(),
                            datatype: dt.into(),
                        };
                        if let Some(cls) = entities.get_mut(subject_iri) {
                            cls.mandatory.push(prop);
                        }
                    }
                }
                OGIT_OPTIONAL => {
                    if let Some(obj_iri) = node_iri(&triple.object) {
                        let dt = datatype_map
                            .get(&(subject_iri, OGIT_DATATYPE))
                            .copied()
                            .unwrap_or("xsd:string");
                        let prop = Property {
                            iri: obj_iri.into(),
                            datatype: dt.into(),
                        };
                        if let Some(cls) = entities.get_mut(subject_iri) {
                            cls.optional.push(prop);
                        }
                    }
                }
                OGIT_INDEXED => {
                    if let Some(obj_iri) = node_iri(&triple.object) {
                        let dt = datatype_map
                            .get(&(subject_iri, OGIT_DATATYPE))
                            .copied()
                            .unwrap_or("xsd:string");
                        let prop = Property {
                            iri: obj_iri.into(),
                            datatype: dt.into(),
                        };
                        if let Some(cls) = entities.get_mut(subject_iri) {
                            cls.indexed.push(prop);
                        }
                    }
                }
                OGIT_RELATES | OGIT_ALLOWED_RELATES => {
                    if let Some(obj_iri) = node_iri(&triple.object) {
                        if let Some(cls) = entities.get_mut(subject_iri) {
                            cls.allowed_relates.push(obj_iri.into());
                        }
                    }
                }
                OGIT_BELONGS | OGIT_ALLOWED_BELONGS => {
                    if let Some(obj_iri) = node_iri(&triple.object) {
                        if let Some(cls) = entities.get_mut(subject_iri) {
                            cls.allowed_belongs.push(obj_iri.into());
                        }
                    }
                }
                // All other predicates (ogit:scope, ogit:datatype, rdf:type, …)
                // are silently ignored — forward-compatible with future OGIT
                // namespace additions.
                _ => {}
            }
        }

        // ---------------------------------------------------------------
        // Pass 3: build families by walking heel → hip → twig* → leaf
        //
        // Strategy:
        //   (a) Find heel classes: those whose local name contains "Heel"
        //       AND whose parent is None (root heels) OR any class tagged
        //       as "Heel" tier regardless of parent.
        //   (b) For each heel, enumerate its direct children (hip classes).
        //   (c) For each hip, collect all descendent leaves (classes whose
        //       local name contains "Leaf" anywhere in the subtree under hip).
        //   (d) Assign global leaf indices in DFS order; build bitmap.
        // ---------------------------------------------------------------

        // Build parent → children reverse map for efficient traversal
        let mut children_of: HashMap<&str, Vec<&str>> = HashMap::new();
        for (iri, cls) in &entities {
            if let Some(parent_iri) = &cls.parent {
                children_of
                    .entry(parent_iri.as_ref())
                    .or_default()
                    .push(iri.as_ref());
            }
        }

        // Collect heels (root classes whose local name indicates "Heel" tier)
        let mut heel_iris: Vec<&str> = entities
            .keys()
            .filter(|iri| is_tier(iri, TIER_HEEL))
            .map(|s| s.as_ref())
            .collect();
        heel_iris.sort_unstable(); // deterministic ordering

        // Collect hips (direct children of heels that contain "Hip")
        // Collect twigs (children of hips containing "Twig")
        // Collect leaves (descendants containing "Leaf")
        // We do a DFS from each heel to enumerate leaves in stable order.

        // First, enumerate ALL leaf classes globally (across all families),
        // in deterministic order (sorted by IRI string).
        let mut all_leaves: Vec<&str> = entities
            .keys()
            .filter(|iri| is_tier(iri, TIER_LEAF))
            .map(|s| s.as_ref())
            .collect();
        all_leaves.sort_unstable();
        let leaf_count = all_leaves.len() as u32;

        // Build a leaf IRI → global index map
        let leaf_index: HashMap<&str, usize> = all_leaves
            .iter()
            .enumerate()
            .map(|(i, iri)| (*iri, i))
            .collect();

        // Now build families
        let mut families: Vec<FamilyBitmap> = Vec::new();
        let mut leaf_to_family: HashMap<Box<str>, u32> = HashMap::new();

        let heel_count = heel_iris.len().min(u8::MAX as usize) as u8;

        for heel_iri in &heel_iris {
            // Enumerate hips (direct children of this heel that are Hip tier)
            let mut hip_iris: Vec<&str> = children_of
                .get(*heel_iri)
                .map(|v| v.as_slice())
                .unwrap_or(&[])
                .iter()
                .filter(|iri| is_tier(iri, TIER_HIP))
                .copied()
                .collect();
            hip_iris.sort_unstable();

            for hip_iri in hip_iris {
                // Collect all leaves reachable from this hip (DFS)
                let family_id = families.len() as u32;
                let mut bitmap = vec![false; leaf_count as usize];

                // DFS stack: start from hip
                let mut stack: Vec<&str> = vec![hip_iri];
                while let Some(current) = stack.pop() {
                    if is_tier(current, TIER_LEAF) {
                        if let Some(&idx) = leaf_index.get(current) {
                            bitmap[idx] = true;
                            leaf_to_family.insert(current.into(), family_id);
                        }
                    }
                    // Push children onto the stack
                    if let Some(kids) = children_of.get(current) {
                        let mut sorted_kids = kids.clone();
                        sorted_kids.sort_unstable();
                        stack.extend(sorted_kids);
                    }
                }

                families.push(FamilyBitmap {
                    family_id,
                    heel_iri: (*heel_iri).into(),
                    hip_iri: hip_iri.into(),
                    bitmap,
                });
            }
        }

        // Infer namespace from common prefix of entity IRIs (best-effort;
        // defaults to empty string when no entities are present).
        let namespace: Box<str> = entities
            .keys()
            .next()
            .map(|first| {
                // Take everything up to (but not including) the local name
                let iri = first.as_ref();
                if let Some(pos) = iri.rfind(':') {
                    // prefix portion e.g. "ogit" from "ogit:Heel"
                    let prefix = &iri[..pos];
                    // Try to guess a human-readable namespace name from
                    // the last path segment of the prefix IRI base, but
                    // since we only have the compact prefix token here,
                    // just return the prefix itself.
                    prefix.into()
                } else {
                    "".into()
                }
            })
            .unwrap_or_else(|| "".into());

        Ok(OntologySchema {
            namespace,
            entities,
            families,
            leaf_to_family,
            heel_count,
            leaf_count,
        })
    }

    /// Returns `true` iff `descendant` is reachable from `ancestor` by
    /// walking the `rdfs:subClassOf` chain encoded in
    /// [`EntityClass::parent`]. The relation is *reflexive*: every class
    /// is its own ancestor (including unknown IRIs — which return `true`
    /// only for the reflexive case `ancestor == descendant`).
    ///
    /// # Complexity
    ///
    /// O(depth) — walks at most `MAX_DEPTH = 64` parent links before
    /// returning `false`, defensively guarding against any cycle that
    /// might have slipped past the partial-order check upstream. Pillar-14
    /// asserts the closure is antisymmetric on synthetic DAGs; this
    /// method's cycle guard is the runtime backstop for that assertion.
    ///
    /// # Use case
    ///
    /// The Pillar-14 drift-check test in `crate::hpc::pillar::ogit_lattice`
    /// uses this method to verify the loaded ontology's closure satisfies
    /// the same three partial-order axioms the synthetic-DAG probe
    /// certifies. Beyond that, it is the natural query for any
    /// type-gated cascade step: "may activation propagate from a tile of
    /// type `descendant` to a tile of type `ancestor`?"
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Reflexive
    /// assert!(schema.is_ancestor("ogit:Heel", "ogit:Heel"));
    /// // Transitive via the subClassOf chain
    /// assert!(schema.is_ancestor("ogit:Heel", "ogit:SomeLeaf"));
    /// // No relation
    /// assert!(!schema.is_ancestor("ogit:Leaf", "ogit:Heel"));
    /// // Unknown descendant — only reflexive case returns true
    /// assert!(!schema.is_ancestor("ogit:Heel", "ogit:Unknown"));
    /// ```
    pub fn is_ancestor(&self, ancestor: &str, descendant: &str) -> bool {
        // Reflexive case — handles both the "X is its own ancestor"
        // identity and the unknown-class case (where neither IRI is in
        // the schema but they are equal).
        if ancestor == descendant {
            return true;
        }

        // Per the documented contract, unknown IRIs are an ancestor ONLY
        // in the reflexive case. `from_triples` accepts `rdfs:subClassOf`
        // targets without requiring them to be declared classes, so an
        // `EntityClass.parent` can legally point at an IRI that's not in
        // `self.entities`. Without this guard a malformed/partial schema
        // could make `is_ancestor("missing:Class", "ogit:Leaf")` succeed
        // (codex P2 on PR #189).
        if !self.entities.contains_key(ancestor) {
            return false;
        }

        // BFS over the multi-parent DAG. The previous version walked a
        // linear chain via `EntityClass.parent` alone — correct for
        // single-inheritance schemas but missed ancestors reachable
        // only through `EntityClass.extra_parents` (OWL multi-inheritance,
        // common in FMA / ChEBI).
        //
        // # Termination
        //
        // `visited` is a monotonically-growing `HashSet<&str>` keyed by
        // IRI; each parent IRI enters the set at most once. Frontier
        // pushes are gated on `visited.insert(...)`, so every IRI is
        // pushed at most once across the entire walk. Total work is
        // therefore O(unique IRIs reachable from descendant) — finite
        // by the schema's finiteness, regardless of branching factor
        // or depth. No explicit visit cap is needed; previous codex P2
        // pointed out that a hard cap would produce false-negatives on
        // large biomedical ontologies (FMA: 75k classes; ChEBI: 200k+).
        let mut frontier: Vec<&str> = vec![descendant];
        let mut visited: std::collections::HashSet<&str> = std::collections::HashSet::new();
        visited.insert(descendant);
        while let Some(current) = frontier.pop() {
            let entity = match self.entities.get(current) {
                Some(e) => e,
                // Walk hit an unknown IRI mid-chain — that subtree of
                // the closure terminates here. Continue exploring
                // siblings rather than aborting, since other parents
                // may yet reach `ancestor`.
                None => continue,
            };
            for parent in entity.parents() {
                if parent == ancestor {
                    return true;
                }
                if visited.insert(parent) {
                    frontier.push(parent);
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hpc::ogit_bridge::turtle_parser::TurtleParser;

    // -----------------------------------------------------------------------
    // Gate 1: empty triples → empty schema
    // -----------------------------------------------------------------------
    #[test]
    fn empty_triples_produces_empty_schema() {
        let schema = OntologySchema::from_triples(&[]).unwrap();
        assert!(schema.entities.is_empty(), "no entities expected");
        assert_eq!(schema.leaf_count, 0);
        assert_eq!(schema.heel_count, 0);
        assert!(schema.families.is_empty());
        assert!(schema.leaf_to_family.is_empty());
    }

    // -----------------------------------------------------------------------
    // Gate 2: single rdfs:Class triple builds one entity
    // -----------------------------------------------------------------------
    #[test]
    fn single_class_triple_builds_one_entity() {
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:Foo a rdfs:Class .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();
        assert_eq!(schema.entities.len(), 1);
        assert!(
            schema.entities.contains_key("ogit:Foo"),
            "entity ogit:Foo not found; keys: {:?}",
            schema.entities.keys().collect::<Vec<_>>()
        );
    }

    // -----------------------------------------------------------------------
    // Gate 3: subClassOf chain sets parent link
    // -----------------------------------------------------------------------
    #[test]
    fn sub_class_of_chain_populates_parent() {
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:Hip1 a rdfs:Class ;\n\
                rdfs:subClassOf ogit:Heel1 .\n\
            ogit:Heel1 a rdfs:Class .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();
        assert_eq!(schema.entities.len(), 2);
        let hip = schema
            .entities
            .get("ogit:Hip1")
            .expect("ogit:Hip1 must exist");
        assert_eq!(hip.parent.as_deref(), Some("ogit:Heel1"), "parent should be ogit:Heel1");
        let heel = schema
            .entities
            .get("ogit:Heel1")
            .expect("ogit:Heel1 must exist");
        assert!(heel.parent.is_none(), "heel has no parent");
    }

    // -----------------------------------------------------------------------
    // Gate 4: ogit:mandatory triple appends to entity.mandatory
    // -----------------------------------------------------------------------
    #[test]
    fn mandatory_property_collected() {
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:CognitiveCell a rdfs:Class ;\n\
                ogit:mandatory ogit:basinSignature .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();
        let cell = schema
            .entities
            .get("ogit:CognitiveCell")
            .expect("CognitiveCell must exist");
        assert_eq!(cell.mandatory.len(), 1);
        assert_eq!(cell.mandatory[0].iri.as_ref(), "ogit:basinSignature");
        // Default datatype when none specified
        assert_eq!(cell.mandatory[0].datatype.as_ref(), "xsd:string");
    }

    // -----------------------------------------------------------------------
    // Gate 5: leaf_to_family lookup — full heel→hip→twig→leaf chain
    // -----------------------------------------------------------------------
    #[test]
    fn leaf_to_family_resolves_through_chain() {
        // Minimal 4-tier ontology: Heel1 → Hip1 → Twig1 → Leaf1
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:Heel1 a rdfs:Class .\n\
            ogit:Hip1 a rdfs:Class ;\n\
                rdfs:subClassOf ogit:Heel1 .\n\
            ogit:Twig1 a rdfs:Class ;\n\
                rdfs:subClassOf ogit:Hip1 .\n\
            ogit:Leaf1 a rdfs:Class ;\n\
                rdfs:subClassOf ogit:Twig1 .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();

        // Leaf1 must be registered in leaf_to_family
        assert!(
            schema.leaf_to_family.contains_key("ogit:Leaf1"),
            "Leaf1 must appear in leaf_to_family; map = {:?}",
            schema.leaf_to_family
        );

        let family_id = schema.leaf_to_family["ogit:Leaf1"];
        let family = &schema.families[family_id as usize];
        assert_eq!(family.heel_iri.as_ref(), "ogit:Heel1");
        assert_eq!(family.hip_iri.as_ref(), "ogit:Hip1");

        // The bitmap must have Leaf1's bit set
        assert_eq!(schema.leaf_count, 1);
        assert_eq!(family.bitmap.len(), 1);
        assert!(family.bitmap[0], "bitmap[0] must be true for Leaf1");
    }

    // -----------------------------------------------------------------------
    // Bonus: rdfs:label is captured
    // -----------------------------------------------------------------------
    #[test]
    fn rdfs_label_captured() {
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:Bar a rdfs:Class ;\n\
                rdfs:label \"My Bar\" .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();
        let bar = schema.entities.get("ogit:Bar").expect("Bar must exist");
        assert_eq!(bar.label.as_ref(), "My Bar");
    }

    // -----------------------------------------------------------------------
    // Bonus: allowed_relates / allowed_belongs collected
    // -----------------------------------------------------------------------
    #[test]
    fn allowed_relates_and_belongs_collected() {
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:Alpha a rdfs:Class ;\n\
                ogit:relates ogit:Beta ;\n\
                ogit:belongs ogit:Gamma .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();
        let alpha = schema.entities.get("ogit:Alpha").expect("Alpha must exist");
        assert_eq!(alpha.allowed_relates.len(), 1);
        assert_eq!(alpha.allowed_relates[0].as_ref(), "ogit:Beta");
        assert_eq!(alpha.allowed_belongs.len(), 1);
        assert_eq!(alpha.allowed_belongs[0].as_ref(), "ogit:Gamma");
    }

    // -----------------------------------------------------------------------
    // Bonus: multiple leaves in one family each get a distinct bitmap bit
    // -----------------------------------------------------------------------
    #[test]
    fn multiple_leaves_distinct_bitmap_bits() {
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:Heel1 a rdfs:Class .\n\
            ogit:Hip1 a rdfs:Class ; rdfs:subClassOf ogit:Heel1 .\n\
            ogit:Leaf1 a rdfs:Class ; rdfs:subClassOf ogit:Hip1 .\n\
            ogit:Leaf2 a rdfs:Class ; rdfs:subClassOf ogit:Hip1 .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();
        assert_eq!(schema.leaf_count, 2);

        let fid1 = schema.leaf_to_family["ogit:Leaf1"];
        let fid2 = schema.leaf_to_family["ogit:Leaf2"];
        assert_eq!(fid1, fid2, "both leaves belong to the same Hip1 family");

        let family = &schema.families[fid1 as usize];
        assert_eq!(family.bitmap.len(), 2);
        // Both bits must be set
        assert!(family.bitmap.iter().all(|&b| b), "all leaf bits must be set");
    }

    // -----------------------------------------------------------------------
    // is_ancestor — partial-order axiom queries on the loaded closure
    // -----------------------------------------------------------------------

    fn build_chain_schema() -> OntologySchema {
        // Build a four-tier chain: Heel ← Hip ← Twig ← Leaf.
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:Heel a rdfs:Class .\n\
            ogit:Hip  a rdfs:Class ; rdfs:subClassOf ogit:Heel .\n\
            ogit:Twig a rdfs:Class ; rdfs:subClassOf ogit:Hip .\n\
            ogit:Leaf a rdfs:Class ; rdfs:subClassOf ogit:Twig .";
        let triples = TurtleParser::parse(src).unwrap();
        OntologySchema::from_triples(&triples).unwrap()
    }

    #[test]
    fn is_ancestor_reflexive() {
        let schema = build_chain_schema();
        assert!(schema.is_ancestor("ogit:Heel", "ogit:Heel"));
        assert!(schema.is_ancestor("ogit:Hip", "ogit:Hip"));
        assert!(schema.is_ancestor("ogit:Leaf", "ogit:Leaf"));
        // Reflexivity must hold even for IRIs that aren't in the schema —
        // the relation is conceptually defined on the full IRI space.
        assert!(schema.is_ancestor("ogit:Unknown", "ogit:Unknown"));
    }

    #[test]
    fn is_ancestor_direct_parent() {
        let schema = build_chain_schema();
        assert!(schema.is_ancestor("ogit:Heel", "ogit:Hip"));
        assert!(schema.is_ancestor("ogit:Hip", "ogit:Twig"));
        assert!(schema.is_ancestor("ogit:Twig", "ogit:Leaf"));
    }

    #[test]
    fn is_ancestor_transitive_through_chain() {
        let schema = build_chain_schema();
        // Heel ← Hip ← Twig ← Leaf — transitivity at every depth.
        assert!(schema.is_ancestor("ogit:Heel", "ogit:Twig"));
        assert!(schema.is_ancestor("ogit:Heel", "ogit:Leaf"));
        assert!(schema.is_ancestor("ogit:Hip", "ogit:Leaf"));
    }

    #[test]
    fn is_ancestor_antisymmetric() {
        let schema = build_chain_schema();
        // Reverse direction must NOT hold (except reflexive cases).
        assert!(!schema.is_ancestor("ogit:Leaf", "ogit:Heel"));
        assert!(!schema.is_ancestor("ogit:Twig", "ogit:Hip"));
        assert!(!schema.is_ancestor("ogit:Hip", "ogit:Heel"));
    }

    #[test]
    fn is_ancestor_unknown_descendant_returns_false() {
        let schema = build_chain_schema();
        // Unknown descendant — no chain to walk. Only reflexive case
        // returns true (covered by `is_ancestor_reflexive`).
        assert!(!schema.is_ancestor("ogit:Heel", "ogit:Unknown"));
        assert!(!schema.is_ancestor("ogit:Hip", "ogit:DefinitelyNotInSchema"));
    }

    /// Codex P2 regression on PR #189: a class whose `rdfs:subClassOf`
    /// edge points at an IRI not declared as a class must NOT satisfy
    /// `is_ancestor(unknown_parent, child)`. Only the reflexive case
    /// `is_ancestor(unknown, unknown)` should return true.
    #[test]
    fn is_ancestor_unknown_ancestor_returns_false_when_non_reflexive() {
        // ogit:Leaf claims subClassOf ogit:Phantom, but ogit:Phantom is
        // never `a rdfs:Class` — TurtleParser/from_triples accepts this
        // (rdfs allows undeclared subClassOf targets).
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:Leaf a rdfs:Class ; rdfs:subClassOf ogit:Phantom .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();

        // The reflexive case still holds — defined on the full IRI space.
        assert!(schema.is_ancestor("ogit:Phantom", "ogit:Phantom"));

        // But non-reflexive lookups against the undeclared ancestor MUST
        // return false, regardless of any entity's parent field pointing
        // at it.
        assert!(
            !schema.is_ancestor("ogit:Phantom", "ogit:Leaf"),
            "is_ancestor must reject ancestors not declared as rdfs:Class"
        );
    }

    #[test]
    fn is_ancestor_unrelated_classes() {
        // Two disjoint chains — Heel/Hip and OtherHeel/OtherHip.
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:Heel a rdfs:Class .\n\
            ogit:Hip  a rdfs:Class ; rdfs:subClassOf ogit:Heel .\n\
            ogit:OtherHeel a rdfs:Class .\n\
            ogit:OtherHip  a rdfs:Class ; rdfs:subClassOf ogit:OtherHeel .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();
        // Across disjoint chains: neither direction holds.
        assert!(!schema.is_ancestor("ogit:Heel", "ogit:OtherHip"));
        assert!(!schema.is_ancestor("ogit:OtherHeel", "ogit:Hip"));
    }

    // -----------------------------------------------------------------------
    // Multi-inheritance — OWL biomedical-ontology shape (FMA, ChEBI, etc.)
    // -----------------------------------------------------------------------

    /// A class declaring two `rdfs:subClassOf` triples must reach both
    /// ancestors through `is_ancestor`. The previous single-parent
    /// implementation silently picked one and discarded the other.
    #[test]
    fn is_ancestor_multi_parent_direct() {
        // Hand mimics an OWL fragment: ogit:Hybrid is both a kind of
        // ogit:Animal AND a kind of ogit:Mineral.
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:Animal a rdfs:Class .\n\
            ogit:Mineral a rdfs:Class .\n\
            ogit:Hybrid a rdfs:Class ; rdfs:subClassOf ogit:Animal ; rdfs:subClassOf ogit:Mineral .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();
        // Both parents must be reachable from the hybrid.
        assert!(schema.is_ancestor("ogit:Animal", "ogit:Hybrid"));
        assert!(schema.is_ancestor("ogit:Mineral", "ogit:Hybrid"));
        // Reverse direction still false (antisymmetry).
        assert!(!schema.is_ancestor("ogit:Hybrid", "ogit:Animal"));
        assert!(!schema.is_ancestor("ogit:Hybrid", "ogit:Mineral"));
    }

    /// Multi-parent transitivity: an ancestor reachable only through
    /// the SECOND parent of a multi-inheritance class must still be
    /// found. This is the case the previous linear-walk implementation
    /// silently missed.
    #[test]
    fn is_ancestor_multi_parent_transitive_through_second_parent() {
        // Two disjoint chains converge at ogit:Hybrid:
        //   ogit:Root1 ← ogit:Mid1 ← ogit:Hybrid (via "first" parent)
        //   ogit:Root2 ← ogit:Mid2 ← ogit:Hybrid (via "second" parent)
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:Root1 a rdfs:Class .\n\
            ogit:Mid1  a rdfs:Class ; rdfs:subClassOf ogit:Root1 .\n\
            ogit:Root2 a rdfs:Class .\n\
            ogit:Mid2  a rdfs:Class ; rdfs:subClassOf ogit:Root2 .\n\
            ogit:Hybrid a rdfs:Class ; rdfs:subClassOf ogit:Mid1 ; rdfs:subClassOf ogit:Mid2 .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();
        // Reachable through first parent chain.
        assert!(schema.is_ancestor("ogit:Root1", "ogit:Hybrid"));
        assert!(schema.is_ancestor("ogit:Mid1", "ogit:Hybrid"));
        // Reachable through second parent chain — the case the
        // previous implementation missed.
        assert!(schema.is_ancestor("ogit:Root2", "ogit:Hybrid"));
        assert!(schema.is_ancestor("ogit:Mid2", "ogit:Hybrid"));
    }

    /// The `parents()` iterator must surface both `parent` and every
    /// `extra_parents` IRI in source order.
    #[test]
    fn entity_class_parents_iterator_yields_all() {
        let src = "\
            @prefix ogit: <http://www.purl.org/ogit/> .\n\
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
            ogit:A a rdfs:Class .\n\
            ogit:B a rdfs:Class .\n\
            ogit:C a rdfs:Class .\n\
            ogit:X a rdfs:Class ; rdfs:subClassOf ogit:A ; rdfs:subClassOf ogit:B ; rdfs:subClassOf ogit:C .";
        let triples = TurtleParser::parse(src).unwrap();
        let schema = OntologySchema::from_triples(&triples).unwrap();
        let x = schema.entities.get("ogit:X").expect("ogit:X declared");
        let parents: Vec<&str> = x.parents().collect();
        assert_eq!(parents.len(), 3, "expected 3 parents, got {parents:?}");
        // First parent populates `parent`; the rest go to extra_parents.
        // Source-order is preserved within extra_parents but the "first"
        // parent depends on triple processing order, so just check set.
        let parent_set: std::collections::HashSet<&str> = parents.iter().copied().collect();
        assert!(parent_set.contains("ogit:A"));
        assert!(parent_set.contains("ogit:B"));
        assert!(parent_set.contains("ogit:C"));
    }
}
