//! OGIT ontology bridge — RDF Turtle parsing and OGIT schema ingestion.
//!
//! This module provides a zero-dependency, zero-copy Turtle parser for the
//! 26 OGIT TTL files (~700 triples each). It is gated behind the
//! `ogit_bridge` feature flag so it does not affect builds that do not need
//! ontology introspection.
//!
//! # Feature gate
//! Add `ogit_bridge = []` to your `Cargo.toml` `[features]` section and
//! depend on `ndarray` with `features = ["ogit_bridge"]` to enable this
//! module.
//!
//! # Modules
//! - [`turtle_parser`] — RDF 1.1 Turtle lexer + parser (OGIT subset)
//! - [`schema`] — [`OntologySchema`], [`EntityClass`], [`FamilyBitmap`],
//!   [`Property`] — in-memory schema built from triples
//!
//! # Performance target
//! Parse 26 TTL files (~700-900 lines, ~700 triples each) in < 50 ms total
//! on a single thread (zero allocation on the hot path beyond the output
//! `Vec<Triple>`).

#![allow(missing_docs)]

pub mod embedded;
pub mod turtle_parser;
pub mod schema;
pub mod cognitive_bridge;

pub use cognitive_bridge::{BasinAtom, CamCodebook, CognitiveBridge, OgitError};
pub use schema::{EntityClass, FamilyBitmap, OntologySchema, Property, SchemaError};
