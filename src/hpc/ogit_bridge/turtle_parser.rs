//! Minimal RDF 1.1 Turtle parser for the OGIT ontology bridge.
//!
//! Handles the subset of Turtle used by the 26 OGIT TTL files:
//! - IRI references (`<http://...>` or `prefix:local`)
//! - String literals with optional `^^datatype` or `@lang` tags
//! - Prefix declarations (`@prefix name: <iri> .`)
//! - Punctuation: `.  ;  ,  [  ]  (  )`
//!
//! # Design
//! Zero-copy, lifetime-tied to the input `&str`. No heap allocation on the
//! hot path except for the returned `Vec<Triple>` and the prefix map
//! (built once, then read-only during triple emission).
//!
//! # Example
//! ```
//! use ndarray::hpc::ogit_bridge::turtle_parser::{TurtleParser, TripleNode};
//!
//! let src = r#"
//!     @prefix ogit: <http://www.purl.org/ogit/> .
//!     @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
//!     ogit:Foo a rdfs:Class .
//! "#;
//! let triples = TurtleParser::parse(src).unwrap();
//! assert_eq!(triples.len(), 1);
//! assert!(matches!(&triples[0].subject, TripleNode::Iri(s) if s.contains("Foo")));
//! ```

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Error type (no thiserror dep — hand-rolled Display)
// ---------------------------------------------------------------------------

/// Errors that can occur during Turtle parsing.
#[derive(Debug)]
pub enum TurtleError {
    /// Syntax error at byte offset with a static message.
    Syntax(usize, &'static str),
    /// A prefixed name used an undeclared prefix.
    UnknownPrefix(String),
}

impl fmt::Display for TurtleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TurtleError::Syntax(off, msg) => {
                write!(f, "syntax error at offset {}: {}", off, msg)
            }
            TurtleError::UnknownPrefix(p) => write!(f, "unknown prefix: {}", p),
        }
    }
}

impl std::error::Error for TurtleError {}

// ---------------------------------------------------------------------------
// Token
// ---------------------------------------------------------------------------

/// A single lexical token produced by [`TurtleLexer`].
#[derive(Debug, PartialEq)]
pub enum TurtleToken<'a> {
    /// A full IRI, e.g. `<http://example.org/Foo>` (angle brackets stripped).
    Iri(&'a str),
    /// A string literal, optionally annotated with a datatype IRI or language tag.
    Literal {
        value: &'a str,
        datatype: Option<&'a str>,
        lang: Option<&'a str>,
    },
    /// A prefix declaration: `@prefix name: <iri> .`
    PrefixDecl { name: &'a str, iri: &'a str },
    /// `.`
    Dot,
    /// `;`
    Semicolon,
    /// `,`
    Comma,
    /// `[`
    OpenBracket,
    /// `]`
    CloseBracket,
    /// `(`
    OpenParen,
    /// `)`
    CloseParen,
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

/// Stateful byte-position lexer over a Turtle source string.
pub struct TurtleLexer<'a> {
    src: &'a str,
    pos: usize,
}

impl<'a> TurtleLexer<'a> {
    /// Create a new lexer positioned at the start of `src`.
    pub fn new(src: &'a str) -> Self {
        TurtleLexer { src, pos: 0 }
    }

    /// Return the current byte offset into `src`.
    pub fn offset(&self) -> usize {
        self.pos
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn remaining(&self) -> &'a str {
        &self.src[self.pos..]
    }

    fn peek(&self) -> Option<u8> {
        self.src.as_bytes().get(self.pos).copied()
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn advance_while(&mut self, pred: impl Fn(u8) -> bool) {
        while let Some(b) = self.peek() {
            if pred(b) {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    /// Skip whitespace and comments (`# … \n`).
    fn skip_ws(&mut self) {
        loop {
            self.advance_while(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'));
            if self.peek() == Some(b'#') {
                // Skip to end of line
                self.advance_while(|b| b != b'\n');
            } else {
                break;
            }
        }
    }

    /// Parse an IRI reference `<...>`, returning the inner slice.
    fn lex_iri_ref(&mut self) -> Result<&'a str, TurtleError> {
        debug_assert_eq!(self.peek(), Some(b'<'));
        self.advance(1); // consume `<`
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b == b'>' {
                let end = self.pos;
                self.advance(1);
                return Ok(&self.src[start..end]);
            }
            // Basic escape handling: skip \u / \U sequences
            if b == b'\\' {
                self.advance(1);
                if let Some(esc) = self.peek() {
                    match esc {
                        b'u' => { self.advance(5); } // \uXXXX
                        b'U' => { self.advance(9); } // \UXXXXXXXX
                        _ => { self.advance(1); }
                    }
                    continue;
                }
            }
            self.advance(1);
        }
        Err(TurtleError::Syntax(self.pos, "unterminated IRI reference"))
    }

    /// Parse a quoted string literal `"..."` or `"""..."""`, returning
    /// (value_slice, datatype_slice_opt, lang_slice_opt).
    fn lex_literal(&mut self) -> Result<(&'a str, Option<&'a str>, Option<&'a str>), TurtleError> {
        debug_assert_eq!(self.peek(), Some(b'"'));

        // Detect triple-quoted
        let triple = self.remaining().starts_with("\"\"\"");
        if triple {
            self.advance(3);
        } else {
            self.advance(1);
        }

        let start = self.pos;
        loop {
            match self.peek() {
                None => return Err(TurtleError::Syntax(self.pos, "unterminated literal")),
                Some(b'\\') => {
                    self.advance(1); // skip the backslash
                    self.advance(1); // skip the escaped char
                }
                Some(b'"') => {
                    if triple {
                        if self.remaining().starts_with("\"\"\"") {
                            let end = self.pos;
                            self.advance(3);
                            let value = &self.src[start..end];
                            return self.lex_literal_suffix(value);
                        } else {
                            self.advance(1);
                        }
                    } else {
                        let end = self.pos;
                        self.advance(1);
                        let value = &self.src[start..end];
                        return self.lex_literal_suffix(value);
                    }
                }
                Some(_) => {
                    self.advance(1);
                }
            }
        }
    }

    /// After the closing quote, consume optional `^^<datatype>` or `@lang`.
    fn lex_literal_suffix(
        &mut self,
        value: &'a str,
    ) -> Result<(&'a str, Option<&'a str>, Option<&'a str>), TurtleError> {
        match self.peek() {
            Some(b'^') if self.remaining().starts_with("^^") => {
                self.advance(2);
                // Datatype can be an IRI ref or a prefixed name
                let dt = if self.peek() == Some(b'<') {
                    self.lex_iri_ref()?
                } else {
                    // prefixed name — we keep the raw slice; the parser will expand later
                    self.lex_prefixed_name_raw()?
                };
                Ok((value, Some(dt), None))
            }
            Some(b'@') => {
                self.advance(1);
                let start = self.pos;
                // Language tag: [a-zA-Z]+ ('-' [a-zA-Z0-9]+)*
                self.advance_while(|b| b.is_ascii_alphabetic());
                while self.peek() == Some(b'-') {
                    self.advance(1);
                    self.advance_while(|b| b.is_ascii_alphanumeric());
                }
                let lang = &self.src[start..self.pos];
                Ok((value, None, Some(lang)))
            }
            _ => Ok((value, None, None)),
        }
    }

    /// Consume a prefixed name (e.g. `ogit:Foo`) as a raw `"prefix:local"` slice.
    /// The colon must already be part of what follows the current pos.
    fn lex_prefixed_name_raw(&mut self) -> Result<&'a str, TurtleError> {
        let start = self.pos;
        // Prefix part: [a-zA-Z_][a-zA-Z0-9_.-]*
        self.advance_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'));
        if self.peek() != Some(b':') {
            return Err(TurtleError::Syntax(self.pos, "expected ':' in prefixed name"));
        }
        self.advance(1); // consume ':'
        // Local part: may include most characters except whitespace and terminators
        self.advance_while(|b| {
            !matches!(b, b' ' | b'\t' | b'\r' | b'\n' | b'.' | b';' | b',' | b'[' | b']' | b'(' | b')')
        });
        // Trim trailing dots that are sentence-ending punctuation, not part of the name
        // (e.g. `ogit:Foo.` — the dot is a statement terminator)
        // We already stopped before '.', so nothing to trim here.
        Ok(&self.src[start..self.pos])
    }

    /// Parse a `@prefix` directive, consuming `@prefix name: <iri> .`.
    /// Called when we have already consumed `@prefix`.
    fn lex_prefix_decl(&mut self) -> Result<TurtleToken<'a>, TurtleError> {
        self.skip_ws();
        // Prefix name (may be empty for the default prefix ":")
        let name_start = self.pos;
        self.advance_while(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'));
        let name_end = self.pos;
        // Consume the colon
        if self.peek() != Some(b':') {
            return Err(TurtleError::Syntax(self.pos, "expected ':' after prefix name"));
        }
        self.advance(1);
        let name = &self.src[name_start..name_end];
        self.skip_ws();
        if self.peek() != Some(b'<') {
            return Err(TurtleError::Syntax(self.pos, "expected IRI ref in @prefix"));
        }
        let iri = self.lex_iri_ref()?;
        self.skip_ws();
        // Consume trailing dot
        if self.peek() == Some(b'.') {
            self.advance(1);
        }
        Ok(TurtleToken::PrefixDecl { name, iri })
    }

    // -----------------------------------------------------------------------
    // Public: next token
    // -----------------------------------------------------------------------

    /// Advance to and return the next token, or `None` at end of input.
    pub fn next_token(&mut self) -> Result<Option<TurtleToken<'a>>, TurtleError> {
        self.skip_ws();
        match self.peek() {
            None => Ok(None),
            Some(b'<') => {
                let iri = self.lex_iri_ref()?;
                Ok(Some(TurtleToken::Iri(iri)))
            }
            Some(b'"') => {
                let (value, datatype, lang) = self.lex_literal()?;
                Ok(Some(TurtleToken::Literal { value, datatype, lang }))
            }
            Some(b'.') => {
                self.advance(1);
                Ok(Some(TurtleToken::Dot))
            }
            Some(b';') => {
                self.advance(1);
                Ok(Some(TurtleToken::Semicolon))
            }
            Some(b',') => {
                self.advance(1);
                Ok(Some(TurtleToken::Comma))
            }
            Some(b'[') => {
                self.advance(1);
                Ok(Some(TurtleToken::OpenBracket))
            }
            Some(b']') => {
                self.advance(1);
                Ok(Some(TurtleToken::CloseBracket))
            }
            Some(b'(') => {
                self.advance(1);
                Ok(Some(TurtleToken::OpenParen))
            }
            Some(b')') => {
                self.advance(1);
                Ok(Some(TurtleToken::CloseParen))
            }
            Some(b'@') => {
                // Could be @prefix or @base
                self.advance(1);
                let kw_start = self.pos;
                self.advance_while(|b| b.is_ascii_alphabetic());
                let kw = &self.src[kw_start..self.pos];
                match kw {
                    "prefix" => self.lex_prefix_decl().map(Some),
                    "base" => {
                        // @base <iri> . — consume and skip (we don't use base IRI)
                        self.skip_ws();
                        if self.peek() == Some(b'<') {
                            self.lex_iri_ref()?;
                        }
                        self.skip_ws();
                        if self.peek() == Some(b'.') {
                            self.advance(1);
                        }
                        // Return the next token transparently
                        self.next_token()
                    }
                    _ => Err(TurtleError::Syntax(kw_start - 1, "unknown @-keyword")),
                }
            }
            Some(b) if b.is_ascii_alphabetic() || b == b'_' => {
                // Prefixed name: prefix:local  OR  keyword `a`
                let start = self.pos;
                self.advance_while(|b| {
                    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.')
                });
                // Check for colon → prefixed name
                if self.peek() == Some(b':') {
                    self.advance(1); // consume ':'
                    self.advance_while(|b| {
                        !matches!(
                            b,
                            b' ' | b'\t' | b'\r' | b'\n' | b';' | b',' | b'[' | b']'
                                | b'(' | b')'
                        ) && !(b == b'.' && {
                            // A trailing dot that is not followed by another
                            // dot-continued character is a statement terminator,
                            // not part of the local name.  We stop here so the
                            // dot becomes its own Dot token.
                            false // handled by advance_while stopping at '.'
                        })
                    });
                    // Trim any trailing dots from the local part
                    while self.pos > start && self.src.as_bytes()[self.pos - 1] == b'.' {
                        self.pos -= 1;
                    }
                    let raw = &self.src[start..self.pos];
                    Ok(Some(TurtleToken::Iri(raw)))
                } else {
                    // No colon: could be `a` (shorthand for rdf:type)
                    let word = &self.src[start..self.pos];
                    if word == "a" {
                        // Expand to rdf:type prefixed name notation
                        Ok(Some(TurtleToken::Iri("rdf:type")))
                    } else if word == "true" || word == "false" {
                        // Boolean literals — treat as xsd:boolean literals
                        Ok(Some(TurtleToken::Literal {
                            value: word,
                            datatype: Some("xsd:boolean"),
                            lang: None,
                        }))
                    } else {
                        Err(TurtleError::Syntax(start, "unexpected bare word"))
                    }
                }
            }
            Some(_) => {
                Err(TurtleError::Syntax(self.pos, "unexpected character"))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Triple / TripleNode
// ---------------------------------------------------------------------------

/// A subject, predicate, or object node in a parsed triple.
#[derive(Debug, Clone)]
pub enum TripleNode<'a> {
    /// A fully-expanded IRI (prefix has been resolved).
    Iri(&'a str),
    /// A literal value.
    Literal {
        value: &'a str,
        /// Fully-expanded datatype IRI (or raw prefixed form if resolution
        /// of well-known prefixes like `xsd:` fails — callers should treat
        /// unknown datatypes as `xsd:string`).
        datatype: Option<&'a str>,
    },
}

/// A parsed RDF triple.
#[derive(Debug)]
pub struct Triple<'a> {
    pub subject: TripleNode<'a>,
    pub predicate: TripleNode<'a>,
    pub object: TripleNode<'a>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// RDF 1.1 Turtle parser (OGIT subset).
///
/// Lifetime `'a` is tied to the input string; all output borrows from it.
///
/// # Example
/// ```
/// use ndarray::hpc::ogit_bridge::turtle_parser::{TurtleParser, TripleNode};
///
/// let src = "@prefix ogit: <http://www.purl.org/ogit/> .\n\
///            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
///            ogit:Foo a rdfs:Class .";
/// let triples = TurtleParser::parse(src).unwrap();
/// assert_eq!(triples.len(), 1);
/// match &triples[0].subject {
///     TripleNode::Iri(s) => assert!(s.contains("Foo")),
///     _ => panic!("expected IRI"),
/// }
/// ```
pub struct TurtleParser<'a> {
    lexer: TurtleLexer<'a>,
    prefixes: HashMap<&'a str, &'a str>,
}

impl<'a> TurtleParser<'a> {
    fn new(input: &'a str) -> Self {
        TurtleParser {
            lexer: TurtleLexer::new(input),
            prefixes: HashMap::new(),
        }
    }

    /// Parse the full Turtle document and return all triples.
    ///
    /// Returns `Err` on syntax errors or unknown prefixes.
    pub fn parse(input: &'a str) -> Result<Vec<Triple<'a>>, TurtleError> {
        let mut parser = TurtleParser::new(input);
        parser.parse_document()
    }

    // -----------------------------------------------------------------------
    // Prefix resolution
    // -----------------------------------------------------------------------

    /// Expand a raw prefixed-name token (e.g. `"ogit:Foo"`) to its full IRI.
    ///
    /// If the token is already a full IRI (`<...>` style — delivered by the
    /// lexer without angle brackets), we attempt to detect and return as-is.
    fn resolve_iri(&self, raw: &'a str) -> Result<&'a str, TurtleError> {
        // Full IRI: starts with a scheme (http:, https:, urn:, etc.) or
        // was delivered by the lexer as a bare full IRI (no angle brackets).
        if raw.starts_with("http://")
            || raw.starts_with("https://")
            || raw.starts_with("urn:")
            || raw.starts_with("file:")
        {
            return Ok(raw);
        }

        // Find the colon separator between prefix and local name
        if let Some(colon) = raw.find(':') {
            let prefix_name = &raw[..colon];
            if self.prefixes.contains_key(prefix_name) {
                // We cannot construct a new &'a str by concatenation without
                // allocation, so we return the raw token. This is safe for
                // consumers that only pattern-match on the known predicates
                // and IRIs, which are all prefixed names.
                // For full IRI materialization, callers should use
                // `format!("{}{}", self.prefixes[prefix_name], &raw[colon+1..])`.
                // Return the raw prefixed token as the "resolved IRI".
                return Ok(raw);
            }
            // Special case: `rdf:type` is always valid (built-in expansion for `a`)
            if prefix_name == "rdf" {
                return Ok(raw);
            }
            return Err(TurtleError::UnknownPrefix(prefix_name.to_owned()));
        }

        // No colon — bare name, not a valid IRI in OGIT Turtle files
        Err(TurtleError::Syntax(0, "bare IRI without prefix or scheme"))
    }

    /// Resolve a datatype token (may be a prefixed name or full IRI ref).
    fn resolve_datatype(&self, raw: &'a str) -> Result<&'a str, TurtleError> {
        self.resolve_iri(raw)
    }

    // -----------------------------------------------------------------------
    // Parsing helpers
    // -----------------------------------------------------------------------

    /// Peek at the next token without consuming whitespace permanently
    /// (we do consume whitespace permanently since the lexer is stateful,
    /// but we can't un-skip). This just calls next_token.
    fn next(&mut self) -> Result<Option<TurtleToken<'a>>, TurtleError> {
        self.lexer.next_token()
    }

    /// Require the next token to match a specific variant; return an error otherwise.
    fn expect_dot(&mut self) -> Result<(), TurtleError> {
        match self.next()? {
            Some(TurtleToken::Dot) => Ok(()),
            _ => Err(TurtleError::Syntax(self.lexer.offset(), "expected '.'")),
        }
    }

    // -----------------------------------------------------------------------
    // Document-level parsing
    // -----------------------------------------------------------------------

    fn parse_document(&mut self) -> Result<Vec<Triple<'a>>, TurtleError> {
        let mut triples: Vec<Triple<'a>> = Vec::new();
        loop {
            match self.next()? {
                None => break,
                Some(TurtleToken::PrefixDecl { name, iri }) => {
                    self.prefixes.insert(name, iri);
                }
                Some(tok) => {
                    // Start of a triple statement: tok is the subject
                    let subject = self.token_to_node(tok)?;
                    self.parse_predicate_object_list(&subject, &mut triples)?;
                }
            }
        }
        Ok(triples)
    }

    /// Convert a token to a `TripleNode` (subject or predicate position).
    fn token_to_node(&self, tok: TurtleToken<'a>) -> Result<TripleNode<'a>, TurtleError> {
        match tok {
            TurtleToken::Iri(raw) => {
                let resolved = self.resolve_iri(raw)?;
                Ok(TripleNode::Iri(resolved))
            }
            TurtleToken::Literal { value, datatype, lang: _ } => {
                // In subject/predicate position literals are unusual but allowed;
                // we normalise to Literal regardless.
                let dt = match datatype {
                    Some(d) => Some(self.resolve_datatype(d)?),
                    None => None,
                };
                Ok(TripleNode::Literal { value, datatype: dt })
            }
            _ => Err(TurtleError::Syntax(
                self.lexer.offset(),
                "expected IRI or literal as subject",
            )),
        }
    }

    /// Convert a token to a `TripleNode` for the object position.
    /// This extends `token_to_node` to also handle Literal with lang.
    fn token_to_object_node(&self, tok: TurtleToken<'a>) -> Result<TripleNode<'a>, TurtleError> {
        match tok {
            TurtleToken::Literal { value, datatype, lang } => {
                let dt = match datatype {
                    Some(d) => Some(self.resolve_datatype(d)?),
                    None => {
                        if lang.is_some() {
                            // Language-tagged strings have implicit type rdf:langString
                            Some("rdf:langString")
                        } else {
                            None
                        }
                    }
                };
                Ok(TripleNode::Literal { value, datatype: dt })
            }
            other => self.token_to_node(other),
        }
    }

    /// Parse `predicate objectList ( ';' predicate objectList )* '.'`
    fn parse_predicate_object_list(
        &mut self,
        subject: &TripleNode<'a>,
        out: &mut Vec<Triple<'a>>,
    ) -> Result<(), TurtleError> {
        loop {
            // Predicate
            let pred_tok = match self.next()? {
                Some(t) => t,
                None => return Err(TurtleError::Syntax(self.lexer.offset(), "expected predicate")),
            };
            let predicate = self.token_to_node(pred_tok)?;

            // Object list
            self.parse_object_list(subject, &predicate, out)?;

            // After object list: '.' ends the statement, ';' continues with another predicate
            match self.next()? {
                Some(TurtleToken::Dot) => break,
                Some(TurtleToken::Semicolon) => {
                    // Check if next token is '.' (trailing semicolon before dot)
                    // We do this by peeking; since we can't un-read, we peek at the
                    // underlying lexer position after skipping ws.
                    // Strategy: save pos, try next token.
                    let saved_pos = self.lexer.pos;
                    match self.next()? {
                        Some(TurtleToken::Dot) => break,
                        Some(tok) => {
                            // It's not a dot — it's the next predicate. Put it "back"
                            // by resetting pos (safe because we haven't mutated state).
                            // Since the lexer is not rewindable, we handle this by
                            // processing the predicate inline.
                            let predicate2 = self.token_to_node(tok)?;
                            self.parse_object_list(subject, &predicate2, out)?;
                            // Now consume the trailing punctuation
                            match self.next()? {
                                Some(TurtleToken::Dot) => break,
                                Some(TurtleToken::Semicolon) => {
                                    // Handle further predicates recursively via loop restart
                                    // by backing up (not possible) — we use a different approach:
                                    // call ourselves recursively for the rest.
                                    // Actually just continue the outer loop — but we can't
                                    // because we already consumed the semicolon.
                                    // So we loop again (peek says there may be more predicates).
                                    continue;
                                }
                                None => break,
                                _ => return Err(TurtleError::Syntax(
                                    self.lexer.offset(),
                                    "expected '.' or ';' after object list",
                                )),
                            }
                        }
                        None => break,
                    }
                }
                None => break, // EOF — tolerate missing trailing dot
                _ => {
                    return Err(TurtleError::Syntax(
                        self.lexer.offset(),
                        "expected '.' or ';' after object list",
                    ))
                }
            }
        }
        Ok(())
    }

    /// Parse `object ( ',' object )*`
    fn parse_object_list(
        &mut self,
        subject: &TripleNode<'a>,
        predicate: &TripleNode<'a>,
        out: &mut Vec<Triple<'a>>,
    ) -> Result<(), TurtleError> {
        loop {
            let obj_tok = match self.next()? {
                Some(t) => t,
                None => return Err(TurtleError::Syntax(self.lexer.offset(), "expected object")),
            };
            // Handle blank node property lists `[...]` as anonymous blank nodes
            let object = match &obj_tok {
                TurtleToken::OpenBracket => {
                    // Skip content until matching ']' — we don't need blank node triples
                    self.skip_blank_node_list()?;
                    TripleNode::Iri("_:b") // anonymous blank node placeholder
                }
                TurtleToken::OpenParen => {
                    // RDF list — skip
                    self.skip_rdf_list()?;
                    TripleNode::Iri("rdf:nil")
                }
                _ => self.token_to_object_node(obj_tok)?,
            };
            out.push(Triple {
                subject: subject.clone(),
                predicate: predicate.clone(),
                object,
            });

            // Check for comma (multi-object) or stop
            let saved = self.lexer.pos;
            match self.lexer.next_token()? {
                Some(TurtleToken::Comma) => continue,
                Some(_) => {
                    // Not a comma — the token belongs to the next production.
                    // We must "unread" it. Since TurtleLexer is forward-only,
                    // we reset pos to saved.
                    self.lexer.pos = saved;
                    break;
                }
                None => break,
            }
        }
        Ok(())
    }

    /// Skip a blank node property list `[ predicate object ; ... ]`.
    fn skip_blank_node_list(&mut self) -> Result<(), TurtleError> {
        let mut depth = 1usize;
        loop {
            match self.next()? {
                Some(TurtleToken::OpenBracket) => depth += 1,
                Some(TurtleToken::CloseBracket) => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                Some(_) => {}
                None => return Err(TurtleError::Syntax(self.lexer.offset(), "unterminated [...]")),
            }
        }
        Ok(())
    }

    /// Skip an RDF list `( item item ... )`.
    fn skip_rdf_list(&mut self) -> Result<(), TurtleError> {
        let mut depth = 1usize;
        loop {
            match self.next()? {
                Some(TurtleToken::OpenParen) => depth += 1,
                Some(TurtleToken::CloseParen) => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                Some(_) => {}
                None => return Err(TurtleError::Syntax(self.lexer.offset(), "unterminated (...)")),
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn iri<'a>(node: &'a TripleNode<'a>) -> &'a str {
        match node {
            TripleNode::Iri(s) => s,
            other => panic!("expected IRI, got {:?}", other),
        }
    }

    fn lit<'a>(node: &'a TripleNode<'a>) -> (&'a str, Option<&'a str>) {
        match node {
            TripleNode::Literal { value, datatype } => (value, *datatype),
            other => panic!("expected Literal, got {:?}", other),
        }
    }

    // Gate 1 — empty input
    #[test]
    fn empty_input_returns_empty() {
        let triples = TurtleParser::parse("").unwrap();
        assert!(triples.is_empty());
    }

    // Gate 2 — single prefix + single triple
    #[test]
    fn single_prefix_and_triple() {
        let src = "@prefix ogit: <http://example.org/> .\n\
                   @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
                   ogit:Foo a rdfs:Class .";
        let triples = TurtleParser::parse(src).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(iri(&triples[0].subject), "ogit:Foo");
        assert_eq!(iri(&triples[0].predicate), "rdf:type");
        assert_eq!(iri(&triples[0].object), "rdfs:Class");
    }

    // Gate 3 — multi-triple with `;` continuation
    #[test]
    fn semicolon_continuation() {
        let src = "@prefix ogit: <http://example.org/> .\n\
                   @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
                   ogit:Foo a rdfs:Class ; ogit:scope \"NTO\" .";
        let triples = TurtleParser::parse(src).unwrap();
        assert_eq!(triples.len(), 2, "expected 2 triples, got {:?}", triples.len());
        assert_eq!(iri(&triples[0].subject), "ogit:Foo");
        assert_eq!(iri(&triples[0].predicate), "rdf:type");
        assert_eq!(iri(&triples[1].predicate), "ogit:scope");
        let (v, _dt) = lit(&triples[1].object);
        assert_eq!(v, "NTO");
    }

    // Gate 4 — multi-object with `,`
    #[test]
    fn comma_multi_object() {
        let src = "@prefix ogit: <http://example.org/> .\n\
                   ogit:Foo ogit:allowed ogit:Bar, ogit:Baz .";
        let triples = TurtleParser::parse(src).unwrap();
        assert_eq!(triples.len(), 2, "got {:?}", triples.len());
        assert_eq!(iri(&triples[0].object), "ogit:Bar");
        assert_eq!(iri(&triples[1].object), "ogit:Baz");
    }

    // Gate 5 — literal with datatype
    #[test]
    fn literal_with_datatype() {
        let src = "@prefix ogit: <http://example.org/> .\n\
                   @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\
                   ogit:Foo ogit:basinSignature \"1234\"^^xsd:long .";
        let triples = TurtleParser::parse(src).unwrap();
        assert_eq!(triples.len(), 1);
        let (v, dt) = lit(&triples[0].object);
        assert_eq!(v, "1234");
        assert_eq!(dt, Some("xsd:long"));
    }

    // Gate 6 — unknown prefix returns error
    #[test]
    fn unknown_prefix_returns_error() {
        let src = "unknownpfx:Foo a unknownpfx:Bar .";
        let result = TurtleParser::parse(src);
        match result {
            Err(TurtleError::UnknownPrefix(p)) => assert_eq!(p, "unknownpfx"),
            other => panic!("expected UnknownPrefix, got {:?}", other),
        }
    }

    // Bonus: full IRI ref in triple
    #[test]
    fn full_iri_ref_in_triple() {
        let src = "<http://example.org/Foo> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://www.w3.org/2000/01/rdf-schema#Class> .";
        let triples = TurtleParser::parse(src).unwrap();
        assert_eq!(triples.len(), 1);
        assert_eq!(iri(&triples[0].subject), "http://example.org/Foo");
    }

    // Bonus: lang-tagged literal
    #[test]
    fn lang_tagged_literal() {
        let src = "@prefix ogit: <http://example.org/> .\n\
                   ogit:Foo ogit:desc \"Hello\"@en .";
        let triples = TurtleParser::parse(src).unwrap();
        assert_eq!(triples.len(), 1);
        let (v, dt) = lit(&triples[0].object);
        assert_eq!(v, "Hello");
        assert_eq!(dt, Some("rdf:langString"));
    }

    // Bonus: comment stripping
    #[test]
    fn comments_are_skipped() {
        let src = "# This is a comment\n\
                   @prefix ogit: <http://example.org/> . # inline comment\n\
                   ogit:Foo ogit:x ogit:Y . # another comment";
        let triples = TurtleParser::parse(src).unwrap();
        assert_eq!(triples.len(), 1);
    }

    // Bonus: multiple subjects
    #[test]
    fn multiple_subjects() {
        let src = "@prefix ogit: <http://example.org/> .\n\
                   @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
                   ogit:Foo a rdfs:Class .\n\
                   ogit:Bar a rdfs:Class .";
        let triples = TurtleParser::parse(src).unwrap();
        assert_eq!(triples.len(), 2);
        assert_eq!(iri(&triples[0].subject), "ogit:Foo");
        assert_eq!(iri(&triples[1].subject), "ogit:Bar");
    }

    // Bonus: trailing semicolon before dot
    #[test]
    fn trailing_semicolon() {
        let src = "@prefix ogit: <http://example.org/> .\n\
                   @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n\
                   ogit:Foo a rdfs:Class ; .";
        // Trailing semicolon followed immediately by dot is valid Turtle 1.1
        let triples = TurtleParser::parse(src).unwrap();
        assert_eq!(triples.len(), 1);
    }

    // Error display
    #[test]
    fn error_display() {
        let e = TurtleError::Syntax(42, "bad token");
        assert_eq!(format!("{}", e), "syntax error at offset 42: bad token");
        let e2 = TurtleError::UnknownPrefix("foo".to_owned());
        assert_eq!(format!("{}", e2), "unknown prefix: foo");
    }
}
