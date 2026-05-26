//! `tileset` (group A) — OGC 3D Tiles `tileset.json` parse (cold import → tile-tree model)
//!
//! # Grounding
//! - OGC 3D Tiles 1.1 specification (CesiumGS/3d-tiles, 22-025r4)
//! - Schema: `tile.schema.json`, `tileset.schema.json`, `content.schema.json`,
//!   `boundingVolume.schema.json`, `tile.implicitTiling.schema.json`
//!
//! # Design contract (RULE 3: no serde, no json crates — see crate `Cargo.toml`)
//! - **COLD IMPORT ONLY.** `tileset.json` is parsed exactly once at load time by a
//!   dependency-free, hand-rolled tokeniser ([`parse_tileset`]). The transient
//!   [`Json`] model is local to this module, never `pub`, and is dropped before
//!   the owned tree is returned — **no JSON value ever reaches the hot path.**
//! - **Owned intermediate structs only.** No source-native / serde types survive
//!   past the import boundary; the parsed tree maps directly to [`to_cam_soa`]
//!   bridge inputs.
//! - **Reverse-engineer only.** [`BoundingVolume`] is a closed enum; extension-
//!   only volumes (e.g. S2) are not preserved (this crate never round-trips).
//!
//! `.3tz` (ZIP envelope) ingestion is deferred — it needs a ZIP reader and the
//! `.3tz` central-directory layout grounded against OGC 22-025r4 §13. The
//! `Archive*` [`ParseError`] variants are reserved for that path.
//!
//! [`to_cam_soa`]: crate::to_cam_soa

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Owned tile-tree model (no serde; constructed once at cold import)
// ─────────────────────────────────────────────────────────────────────────────

/// Top-level parsed representation of a `tileset.json` document.
/// Constructed once at cold-import time; never re-parsed, never round-tripped.
#[derive(Debug, Clone, PartialEq)]
pub struct Tileset {
    /// Asset metadata (spec version, optional dataset version).
    pub asset: TilesetAsset,
    /// Maximum geometric error for the entire tileset, in meters (`geometricError`).
    pub geometric_error: f64,
    /// Root of the tile tree (children stored inline, recursive).
    pub root: TileNode,
    /// Extensions declared in `extensionsUsed`.
    pub extensions_used: Vec<String>,
    /// Extensions declared in `extensionsRequired`.
    pub extensions_required: Vec<String>,
}

/// `asset` metadata block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilesetAsset {
    /// Spec version string, e.g. `"1.1"`.
    pub version: String,
    /// Optional dataset/revision version tag (`tilesetVersion`).
    pub tileset_version: Option<String>,
}

/// Parsed bounding volume — exactly one of three mutually exclusive shapes.
///
/// Extension-supplied volumes (e.g. `3DTILES_bounding_volume_S2`) are not
/// represented: this crate reverse-engineers, it does not round-trip.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundingVolume {
    /// `box`: `[cx,cy,cz, hx0,hx1,hx2, hy0,hy1,hy2, hz0,hz1,hz2]` (center + half-axes).
    Box([f64; 12]),
    /// `region`: `[west, south, east, north]` (radians) + `[minHeight, maxHeight]` (m).
    Region([f64; 6]),
    /// `sphere`: `[cx, cy, cz, radius]`.
    Sphere([f64; 4]),
}

/// Refinement strategy for a tile's children.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refine {
    /// `ADD` — render both parent and children simultaneously.
    Add,
    /// `REPLACE` — render children instead of parent when refined.
    Replace,
}

/// A single content reference within a tile.
#[derive(Debug, Clone, PartialEq)]
pub struct TileContent {
    /// Relative or absolute URI to the payload (`.glb`, `.b3dm`, `.pnts`, …).
    /// Source field `uri` (renamed from `url` in 3D Tiles 1.1; both are accepted).
    pub uri: String,
    /// Optional tighter bounding volume for this content within the tile.
    pub bounding_volume: Option<BoundingVolume>,
}

/// Subdivision scheme for implicit tiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubdivisionScheme {
    /// 2D quadtree (template vars `{level}/{x}/{y}`).
    Quadtree,
    /// 3D octree (template vars `{level}/{x}/{y}/{z}`).
    Octree,
}

/// Implicit tiling configuration attached to a tile node. Expansion of the
/// implicit subtree is delegated to [`crate::implicit_tiling`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImplicitTilingRef {
    /// Quadtree or octree subdivision.
    pub scheme: SubdivisionScheme,
    /// Number of levels stored in a single subtree binary file (≥ 1).
    pub subtree_levels: u32,
    /// Total number of available levels across all subtrees (≥ 1).
    pub available_levels: u32,
    /// URI template for subtree files (e.g. `subtrees/{level}/{x}/{y}.subtree`).
    pub subtrees_uri_template: String,
}

/// A single node in the parsed tile tree. All strings are owned; no JSON survives.
#[derive(Debug, Clone, PartialEq)]
pub struct TileNode {
    /// The spatial extent enclosing this tile's content and children.
    pub bounding_volume: BoundingVolume,
    /// Geometric error in meters (consumed by SSE LOD selection).
    pub geometric_error: f64,
    /// Optional tighter volume the viewer must be inside for content to render.
    pub viewer_request_volume: Option<BoundingVolume>,
    /// Refinement mode; `None` means "inherit from parent" (see [`Tileset::visit_preorder`]).
    pub refine: Option<Refine>,
    /// Column-major 4×4 affine transform (local → parent). `None` = identity.
    pub transform: Option<[f64; 16]>,
    /// Content payloads — zero, one (`content`), or many (`contents`), unified here.
    pub contents: Vec<TileContent>,
    /// Explicit children (empty for leaf tiles and implicit roots).
    pub children: Vec<TileNode>,
    /// Populated for implicit root tiles; expanded by [`crate::implicit_tiling`].
    pub implicit_tiling: Option<ImplicitTilingRef>,
}

impl TileNode {
    /// `true` if this node has no explicit children.
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    /// `true` if this node is an implicit-tiling root.
    pub fn is_implicit_root(&self) -> bool {
        self.implicit_tiling.is_some()
    }
}

impl Tileset {
    /// The declared 3D Tiles asset version (e.g. `"1.1"`).
    pub fn version(&self) -> &str {
        &self.asset.version
    }

    /// Total number of explicit tile nodes (implicit subtrees are not expanded here).
    pub fn node_count(&self) -> usize {
        let mut n = 0;
        self.visit_preorder(|_, _, _| n += 1);
        n
    }

    /// Pre-order DFS over leaf nodes (no explicit children). Implicit roots — which
    /// also have no explicit children — are yielded as-is; expand them via
    /// [`crate::implicit_tiling`]. No JSON is involved.
    pub fn iter_leaves(&self) -> impl Iterator<Item = &TileNode> {
        let mut leaves = Vec::new();
        let mut stack = vec![&self.root];
        while let Some(node) = stack.pop() {
            if node.children.is_empty() {
                leaves.push(node);
            } else {
                // Reverse so the left-most child is visited first (pre-order).
                stack.extend(node.children.iter().rev());
            }
        }
        leaves.into_iter()
    }

    /// Every content URI reachable from explicit (non-implicit) nodes, pre-order.
    pub fn collect_content_uris(&self) -> Vec<&str> {
        let mut uris = Vec::new();
        let mut stack = vec![&self.root];
        while let Some(node) = stack.pop() {
            uris.extend(node.contents.iter().map(|c| c.uri.as_str()));
            stack.extend(node.children.iter().rev());
        }
        uris
    }

    /// Visit every node in pre-order, resolving `refine` inheritance: the visitor
    /// receives each node, its depth (root = 0), and the [`Refine`] mode in effect
    /// (own value, else the inherited one; root defaults to `ADD` if unset).
    pub fn visit_preorder<F: FnMut(&TileNode, usize, Refine)>(&self, mut visit: F) {
        fn walk<F: FnMut(&TileNode, usize, Refine)>(node: &TileNode, depth: usize, inherited: Refine, visit: &mut F) {
            let refine = node.refine.unwrap_or(inherited);
            visit(node, depth, refine);
            for child in &node.children {
                walk(child, depth + 1, refine, visit);
            }
        }
        let root_refine = self.root.refine.unwrap_or(Refine::Add);
        walk(&self.root, 0, root_refine, &mut visit);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Cold-parse errors for `tileset.json` / `.3tz`. Detailed for CLI diagnostics;
/// never used in any hot path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Input was not valid UTF-8.
    InvalidUtf8,
    /// JSON tokenisation failed at the given byte offset.
    JsonSyntax(usize),
    /// A required field was absent.
    MissingField(&'static str),
    /// A field value had the wrong JSON type.
    WrongType {
        /// The offending field path.
        field: &'static str,
        /// The expected shape.
        expected: &'static str,
    },
    /// `refine` was not `"ADD"` or `"REPLACE"`.
    UnknownRefine(String),
    /// `subdivisionScheme` was not `"QUADTREE"` or `"OCTREE"`.
    UnknownSubdivisionScheme(String),
    /// Both `content` and `contents` were present (spec violation).
    ConflictingContent,
    /// `.3tz` archive entry `tileset.json` not found (reserved; `.3tz` deferred).
    ArchiveEntryNotFound,
    /// Decompression failure inside a `.3tz` archive (reserved; `.3tz` deferred).
    DecompressError,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::InvalidUtf8 => write!(f, "tileset input is not valid UTF-8"),
            ParseError::JsonSyntax(at) => write!(f, "JSON syntax error at byte {at}"),
            ParseError::MissingField(name) => write!(f, "missing required field `{name}`"),
            ParseError::WrongType { field, expected } => {
                write!(f, "field `{field}` had wrong type (expected {expected})")
            }
            ParseError::UnknownRefine(v) => write!(f, "unknown refine value `{v}`"),
            ParseError::UnknownSubdivisionScheme(v) => {
                write!(f, "unknown subdivisionScheme `{v}`")
            }
            ParseError::ConflictingContent => {
                write!(f, "tile has both `content` and `contents` (mutually exclusive)")
            }
            ParseError::ArchiveEntryNotFound => write!(f, ".3tz: `tileset.json` entry not found"),
            ParseError::DecompressError => write!(f, ".3tz: decompression failed"),
        }
    }
}

impl std::error::Error for ParseError {}

// ─────────────────────────────────────────────────────────────────────────────
// Cold parse entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a `tileset.json` byte slice (UTF-8) into an owned [`Tileset`] tree.
///
/// This is the **only** place JSON is parsed. The transient [`Json`] model is
/// built here, walked into owned structs, and dropped before returning — no
/// JSON value survives into the hot path. Cheap to call once; never call it in
/// a loop or on the hot path.
///
/// # Errors
/// Returns [`ParseError`] on malformed JSON, missing required fields, conflicting
/// `content`/`contents`, or unknown enum values (fail-fast; never silently defaults).
pub fn parse_tileset(bytes: &[u8]) -> Result<Tileset, ParseError> {
    let text = std::str::from_utf8(bytes).map_err(|_| ParseError::InvalidUtf8)?;
    let mut parser = JsonParser::new(text.as_bytes());
    let doc = parser.parse_value()?;

    let asset = parse_asset(doc.get("asset").ok_or(ParseError::MissingField("asset"))?)?;
    let geometric_error = doc
        .get("geometricError")
        .and_then(Json::as_f64)
        .ok_or(ParseError::MissingField("geometricError"))?;
    let root = parse_tile_node(doc.get("root").ok_or(ParseError::MissingField("root"))?)?;

    Ok(Tileset {
        asset,
        geometric_error,
        root,
        extensions_used: parse_string_array(doc.get("extensionsUsed")),
        extensions_required: parse_string_array(doc.get("extensionsRequired")),
    })
    // `doc` (the transient JSON tree) is dropped here.
}

fn parse_asset(v: &Json) -> Result<TilesetAsset, ParseError> {
    let version = v
        .get("version")
        .and_then(Json::as_str)
        .ok_or(ParseError::MissingField("asset.version"))?
        .to_owned();
    let tileset_version = v
        .get("tilesetVersion")
        .and_then(Json::as_str)
        .map(str::to_owned);
    Ok(TilesetAsset {
        version,
        tileset_version,
    })
}

fn parse_tile_node(v: &Json) -> Result<TileNode, ParseError> {
    let bounding_volume = parse_bounding_volume(
        v.get("boundingVolume")
            .ok_or(ParseError::MissingField("boundingVolume"))?,
    )?;
    let geometric_error = v
        .get("geometricError")
        .and_then(Json::as_f64)
        .ok_or(ParseError::MissingField("geometricError"))?;

    let viewer_request_volume = match v.get("viewerRequestVolume") {
        Some(j) => Some(parse_bounding_volume(j)?),
        None => None,
    };

    let refine = match v.get("refine").and_then(Json::as_str) {
        None => None,
        Some("ADD") => Some(Refine::Add),
        Some("REPLACE") => Some(Refine::Replace),
        Some(other) => return Err(ParseError::UnknownRefine(other.to_owned())),
    };

    let transform = match v.get("transform") {
        Some(j) => Some(parse_f64_array::<16>(j, "transform")?),
        None => None,
    };

    // `content` (singular) and `contents` (array) are mutually exclusive.
    let contents = match (v.get("content"), v.get("contents")) {
        (Some(_), Some(_)) => return Err(ParseError::ConflictingContent),
        (Some(c), None) => vec![parse_content(c)?],
        (None, Some(arr)) => {
            let items = arr.as_array().ok_or(ParseError::WrongType {
                field: "contents",
                expected: "array",
            })?;
            items
                .iter()
                .map(parse_content)
                .collect::<Result<Vec<_>, _>>()?
        }
        (None, None) => Vec::new(),
    };

    let implicit_tiling = match v.get("implicitTiling") {
        Some(j) => Some(parse_implicit_tiling(j)?),
        None => None,
    };

    let children = match v.get("children") {
        Some(arr) => {
            let items = arr.as_array().ok_or(ParseError::WrongType {
                field: "children",
                expected: "array",
            })?;
            items
                .iter()
                .map(parse_tile_node)
                .collect::<Result<Vec<_>, _>>()?
        }
        None => Vec::new(),
    };

    Ok(TileNode {
        bounding_volume,
        geometric_error,
        viewer_request_volume,
        refine,
        transform,
        contents,
        children,
        implicit_tiling,
    })
}

fn parse_bounding_volume(v: &Json) -> Result<BoundingVolume, ParseError> {
    if let Some(b) = v.get("box") {
        Ok(BoundingVolume::Box(parse_f64_array::<12>(b, "boundingVolume.box")?))
    } else if let Some(r) = v.get("region") {
        Ok(BoundingVolume::Region(parse_f64_array::<6>(r, "boundingVolume.region")?))
    } else if let Some(s) = v.get("sphere") {
        Ok(BoundingVolume::Sphere(parse_f64_array::<4>(s, "boundingVolume.sphere")?))
    } else {
        Err(ParseError::MissingField("boundingVolume.{box|region|sphere}"))
    }
}

fn parse_content(v: &Json) -> Result<TileContent, ParseError> {
    // `uri` (1.1) preferred; `url` (1.0) accepted for back-compat.
    let uri = v
        .get("uri")
        .or_else(|| v.get("url"))
        .and_then(Json::as_str)
        .ok_or(ParseError::MissingField("content.uri"))?
        .to_owned();
    let bounding_volume = match v.get("boundingVolume") {
        Some(j) => Some(parse_bounding_volume(j)?),
        None => None,
    };
    Ok(TileContent { uri, bounding_volume })
}

fn parse_implicit_tiling(v: &Json) -> Result<ImplicitTilingRef, ParseError> {
    let scheme = match v
        .get("subdivisionScheme")
        .and_then(Json::as_str)
        .ok_or(ParseError::MissingField("implicitTiling.subdivisionScheme"))?
    {
        "QUADTREE" => SubdivisionScheme::Quadtree,
        "OCTREE" => SubdivisionScheme::Octree,
        other => return Err(ParseError::UnknownSubdivisionScheme(other.to_owned())),
    };
    let subtree_levels = v
        .get("subtreeLevels")
        .and_then(Json::as_f64)
        .ok_or(ParseError::MissingField("implicitTiling.subtreeLevels"))? as u32;
    let available_levels = v
        .get("availableLevels")
        .and_then(Json::as_f64)
        .ok_or(ParseError::MissingField("implicitTiling.availableLevels"))? as u32;
    let subtrees_uri_template = v
        .get("subtrees")
        .and_then(|s| s.get("uri"))
        .and_then(Json::as_str)
        .ok_or(ParseError::MissingField("implicitTiling.subtrees.uri"))?
        .to_owned();
    Ok(ImplicitTilingRef {
        scheme,
        subtree_levels,
        available_levels,
        subtrees_uri_template,
    })
}

fn parse_string_array(v: Option<&Json>) -> Vec<String> {
    v.and_then(Json::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Json::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_f64_array<const N: usize>(v: &Json, field: &'static str) -> Result<[f64; N], ParseError> {
    let arr = v.as_array().ok_or(ParseError::WrongType {
        field,
        expected: "array of numbers",
    })?;
    if arr.len() != N {
        return Err(ParseError::WrongType {
            field,
            expected: "fixed-length number array",
        });
    }
    let mut out = [0.0f64; N];
    for (slot, item) in out.iter_mut().zip(arr.iter()) {
        *slot = item.as_f64().ok_or(ParseError::WrongType {
            field,
            expected: "number",
        })?;
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Transient cold-import JSON model + hand-rolled tokeniser.
// PRIVATE: never public, never reaches the hot path, dropped after the walk.
// ─────────────────────────────────────────────────────────────────────────────

enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    fn as_f64(&self) -> Option<f64> {
        if let Json::Num(n) = self {
            Some(*n)
        } else {
            None
        }
    }

    fn as_str(&self) -> Option<&str> {
        if let Json::Str(s) = self {
            Some(s.as_str())
        } else {
            None
        }
    }

    fn as_array(&self) -> Option<&[Json]> {
        if let Json::Arr(a) = self {
            Some(a)
        } else {
            None
        }
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        JsonParser { bytes, pos: 0 }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if matches!(c, b' ' | b'\t' | b'\n' | b'\r') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), ParseError> {
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            Err(ParseError::JsonSyntax(self.pos))
        }
    }

    fn parse_value(&mut self) -> Result<Json, ParseError> {
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b'"') => Ok(Json::Str(self.parse_string()?)),
            Some(b't') => self.parse_lit(b"true", Json::Bool(true)),
            Some(b'f') => self.parse_lit(b"false", Json::Bool(false)),
            Some(b'n') => self.parse_lit(b"null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.parse_number(),
            _ => Err(ParseError::JsonSyntax(self.pos)),
        }
    }

    fn parse_lit(&mut self, lit: &[u8], val: Json) -> Result<Json, ParseError> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            Err(ParseError::JsonSyntax(self.pos))
        }
    }

    fn parse_object(&mut self) -> Result<Json, ParseError> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(b':')?;
            let value = self.parse_value()?;
            pairs.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(ParseError::JsonSyntax(self.pos)),
            }
        }
        Ok(Json::Obj(pairs))
    }

    fn parse_array(&mut self) -> Result<Json, ParseError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    break;
                }
                _ => return Err(ParseError::JsonSyntax(self.pos)),
            }
        }
        Ok(Json::Arr(items))
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            // `"` (0x22) and `\` (0x5C) are ASCII and never appear inside a
            // multibyte UTF-8 sequence, so scanning for them byte-wise is safe.
            let start = self.pos;
            while let Some(c) = self.peek() {
                if c == b'"' || c == b'\\' {
                    break;
                }
                self.pos += 1;
            }
            // Input was validated as UTF-8 up front, so this slice is valid UTF-8.
            out.push_str(std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| ParseError::JsonSyntax(start))?);
            match self.peek() {
                Some(b'"') => {
                    self.pos += 1;
                    break;
                }
                Some(b'\\') => {
                    self.pos += 1;
                    self.parse_escape(&mut out)?;
                }
                _ => return Err(ParseError::JsonSyntax(self.pos)),
            }
        }
        Ok(out)
    }

    fn parse_escape(&mut self, out: &mut String) -> Result<(), ParseError> {
        let esc = self.peek().ok_or(ParseError::JsonSyntax(self.pos))?;
        self.pos += 1;
        match esc {
            b'"' => out.push('"'),
            b'\\' => out.push('\\'),
            b'/' => out.push('/'),
            b'b' => out.push('\u{0008}'),
            b'f' => out.push('\u{000C}'),
            b'n' => out.push('\n'),
            b'r' => out.push('\r'),
            b't' => out.push('\t'),
            b'u' => {
                let hi = self.parse_hex4()?;
                let scalar = if (0xD800..=0xDBFF).contains(&hi) {
                    // High surrogate: a low surrogate `\uXXXX` must follow.
                    self.expect(b'\\')?;
                    self.expect(b'u')?;
                    let lo = self.parse_hex4()?;
                    if !(0xDC00..=0xDFFF).contains(&lo) {
                        return Err(ParseError::JsonSyntax(self.pos));
                    }
                    0x1_0000 + (((hi - 0xD800) as u32) << 10) + (lo - 0xDC00) as u32
                } else {
                    hi as u32
                };
                out.push(char::from_u32(scalar).ok_or(ParseError::JsonSyntax(self.pos))?);
            }
            _ => return Err(ParseError::JsonSyntax(self.pos)),
        }
        Ok(())
    }

    fn parse_hex4(&mut self) -> Result<u16, ParseError> {
        let mut acc: u16 = 0;
        for _ in 0..4 {
            let c = self.peek().ok_or(ParseError::JsonSyntax(self.pos))?;
            let digit = (c as char)
                .to_digit(16)
                .ok_or(ParseError::JsonSyntax(self.pos))?;
            acc = acc * 16 + digit as u16;
            self.pos += 1;
        }
        Ok(acc)
    }

    fn parse_number(&mut self) -> Result<Json, ParseError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || matches!(c, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).map_err(|_| ParseError::JsonSyntax(start))?;
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| ParseError::JsonSyntax(start))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
    {
      "asset": { "version": "1.1", "tilesetVersion": "site-2026-05" },
      "geometricError": 512.0,
      "extensionsUsed": ["3DTILES_content_gltf"],
      "root": {
        "boundingVolume": { "box": [0,0,0, 100,0,0, 0,100,0, 0,0,20] },
        "geometricError": 256.0,
        "refine": "REPLACE",
        "content": { "uri": "root.glb" },
        "children": [
          {
            "boundingVolume": { "region": [-1.30, 0.69, -1.29, 0.70, 0.0, 30.0] },
            "geometricError": 64.0,
            "contents": [
              { "uri": "child_a.glb" },
              { "uri": "child_a_splat.spz" }
            ]
          },
          {
            "boundingVolume": { "sphere": [10, 10, 5, 50] },
            "geometricError": 0.0,
            "refine": "ADD",
            "content": { "url": "legacy_child.b3dm" }
          }
        ]
      }
    }
    "#;

    fn parse() -> Tileset {
        parse_tileset(SAMPLE.as_bytes()).expect("sample tileset parses")
    }

    #[test]
    fn parses_asset_and_root() {
        let ts = parse();
        assert_eq!(ts.version(), "1.1");
        assert_eq!(ts.asset.tileset_version.as_deref(), Some("site-2026-05"));
        assert_eq!(ts.geometric_error, 512.0);
        assert_eq!(ts.root.geometric_error, 256.0);
        assert_eq!(ts.extensions_used, vec!["3DTILES_content_gltf".to_owned()]);
        assert_eq!(ts.node_count(), 3);
    }

    #[test]
    fn bounding_volume_variants() {
        let ts = parse();
        assert!(matches!(ts.root.bounding_volume, BoundingVolume::Box(_)));
        assert!(matches!(ts.root.children[0].bounding_volume, BoundingVolume::Region(_)));
        assert!(matches!(ts.root.children[1].bounding_volume, BoundingVolume::Sphere(_)));
        if let BoundingVolume::Sphere(s) = ts.root.children[1].bounding_volume {
            assert_eq!(s, [10.0, 10.0, 5.0, 50.0]);
        }
    }

    #[test]
    fn refine_inheritance() {
        let ts = parse();
        let mut modes = Vec::new();
        ts.visit_preorder(|_, depth, refine| modes.push((depth, refine)));
        // root REPLACE; child[0] inherits REPLACE; child[1] overrides to ADD.
        assert_eq!(modes[0], (0, Refine::Replace));
        assert_eq!(modes[1], (1, Refine::Replace));
        assert_eq!(modes[2], (1, Refine::Add));
    }

    #[test]
    fn content_uri_and_legacy_url_unified() {
        let ts = parse();
        // 1.1 multi-content.
        assert_eq!(ts.root.children[0].contents.len(), 2);
        assert_eq!(ts.root.children[0].contents[0].uri, "child_a.glb");
        // 1.0 legacy `url` resolves into the same `uri` field.
        assert_eq!(ts.root.children[1].contents.len(), 1);
        assert_eq!(ts.root.children[1].contents[0].uri, "legacy_child.b3dm");
        // Root singular content unified into the Vec.
        assert_eq!(ts.root.contents[0].uri, "root.glb");
        // Pre-order URI collection.
        assert_eq!(
            ts.collect_content_uris(),
            vec!["root.glb", "child_a.glb", "child_a_splat.spz", "legacy_child.b3dm"]
        );
    }

    #[test]
    fn leaves_are_children_only() {
        let ts = parse();
        let leaves: Vec<_> = ts.iter_leaves().collect();
        assert_eq!(leaves.len(), 2);
    }

    #[test]
    fn conflicting_content_is_rejected() {
        let json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 1.0,
            "root": {
                "boundingVolume": { "sphere": [0,0,0,1] },
                "geometricError": 0.0,
                "refine": "ADD",
                "content": { "uri": "a.glb" },
                "contents": [{ "uri": "b.glb" }]
            }
        }"#;
        assert_eq!(parse_tileset(json.as_bytes()), Err(ParseError::ConflictingContent));
    }

    #[test]
    fn missing_required_field_is_rejected() {
        let json = r#"{ "asset": { "version": "1.1" }, "geometricError": 1.0 }"#;
        assert_eq!(parse_tileset(json.as_bytes()), Err(ParseError::MissingField("root")));
    }

    #[test]
    fn implicit_tiling() {
        let json = r#"{
            "asset": { "version": "1.1" },
            "geometricError": 100.0,
            "root": {
                "boundingVolume": { "box": [0,0,0, 1,0,0, 0,1,0, 0,0,1] },
                "geometricError": 50.0,
                "refine": "REPLACE",
                "content": { "uri": "content/{level}/{x}/{y}.glb" },
                "implicitTiling": {
                    "subdivisionScheme": "QUADTREE",
                    "subtreeLevels": 7,
                    "availableLevels": 21,
                    "subtrees": { "uri": "subtrees/{level}/{x}/{y}.subtree" }
                }
            }
        }"#;
        let ts = parse_tileset(json.as_bytes()).unwrap();
        let it = ts.root.implicit_tiling.as_ref().unwrap();
        assert_eq!(it.scheme, SubdivisionScheme::Quadtree);
        assert_eq!(it.subtree_levels, 7);
        assert_eq!(it.available_levels, 21);
        assert_eq!(it.subtrees_uri_template, "subtrees/{level}/{x}/{y}.subtree");
        assert!(ts.root.is_implicit_root());
    }

    #[test]
    fn malformed_json_is_rejected() {
        assert!(matches!(parse_tileset(b"{ \"asset\": "), Err(ParseError::JsonSyntax(_))));
    }

    #[test]
    fn string_escapes_and_unicode() {
        let json = r#"{
            "asset": { "version": "1.1", "tilesetVersion": "a\/bé\n\"x\"" },
            "geometricError": 1.0,
            "root": { "boundingVolume": { "sphere": [0,0,0,1] }, "geometricError": 0.0, "refine": "ADD" }
        }"#;
        let ts = parse_tileset(json.as_bytes()).unwrap();
        assert_eq!(ts.asset.tileset_version.as_deref(), Some("a/bé\n\"x\""));
    }
}
