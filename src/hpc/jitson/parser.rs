//! # no_std JSON Parser with Bracket Recovery
//!
//! Hand-rolled recursive-descent tokenizer for the JITSON template subset.
//! Zero external dependencies — works in embedded/WASM contexts.
//!
//! Features:
//! - Standard JSON parsing (objects, arrays, strings, numbers, booleans, null)
//! - Bracket recovery: missing trailing `}` or `]` auto-appended
//! - Trailing comma tolerance
//! - Single-line `//` comment support (YAML-ish extension)
//! - Unicode `\uXXXX` escape sequences

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// JSON value type
// ---------------------------------------------------------------------------

/// Minimal JSON value — covers the template subset.
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            JsonValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        self.as_f64().and_then(|n| {
            if n >= 0.0 && n <= u64::MAX as f64 && n.fract() == 0.0 {
                Some(n as u64)
            } else {
                None
            }
        })
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().map(|n| n as usize)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            JsonValue::Object(o) => Some(o.as_slice()),
            _ => None,
        }
    }

    /// Lookup a key in an object.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()
            .and_then(|pairs| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v))
    }
}

// ---------------------------------------------------------------------------
// Parse errors
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub offset: usize,
}

impl core::fmt::Display for ParseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "JITSON parse error at byte {}: {}",
            self.offset, self.message
        )
    }
}

// ---------------------------------------------------------------------------
// Tokenizer + recursive-descent parser with bracket recovery
// ---------------------------------------------------------------------------

struct Parser<'a> {
    input: &'a [u8],
    pos: usize,
    /// Track open brackets/braces for recovery.
    open_stack: Vec<u8>,
}

impl<'a> Parser<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self {
            input,
            pos: 0,
            open_stack: Vec::new(),
        }
    }

    fn err(&self, msg: &str) -> ParseError {
        ParseError {
            message: String::from(msg),
            offset: self.pos,
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            match self.input[self.pos] {
                b' ' | b'\t' | b'\n' | b'\r' => self.pos += 1,
                // Skip single-line comments (YAML-ish extension)
                b'/' if self.pos + 1 < self.input.len() && self.input[self.pos + 1] == b'/' => {
                    self.pos += 2;
                    while self.pos < self.input.len() && self.input[self.pos] != b'\n' {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_whitespace();
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        self.skip_whitespace();
        if self.pos < self.input.len() {
            let b = self.input[self.pos];
            self.pos += 1;
            Some(b)
        } else {
            None
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), ParseError> {
        match self.advance() {
            Some(b) if b == expected => Ok(()),
            Some(b) => Err(self.err(&alloc::format!(
                "expected '{}', found '{}'",
                expected as char,
                b as char
            ))),
            None => {
                // Bracket recovery: if we hit EOF expecting a closing bracket,
                // check if it matches the top of our open_stack.
                if (expected == b'}' || expected == b']')
                    && self.open_stack.last().copied() == Some(expected)
                {
                    self.open_stack.pop();
                    Ok(())
                } else {
                    Err(self.err(&alloc::format!(
                        "unexpected EOF, expected '{}'",
                        expected as char
                    )))
                }
            }
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, ParseError> {
        match self.peek() {
            Some(b'"') => self.parse_string().map(JsonValue::Str),
            Some(b'{') => self.parse_object(),
            Some(b'[') => self.parse_array(),
            Some(b't') | Some(b'f') => self.parse_bool(),
            Some(b'n') => self.parse_null(),
            Some(b'-') | Some(b'0'..=b'9') => self.parse_number(),
            Some(c) => Err(self.err(&alloc::format!("unexpected character '{}'", c as char))),
            None => Err(self.err("unexpected EOF")),
        }
    }

    fn parse_string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut s = String::new();
        loop {
            if self.pos >= self.input.len() {
                return Err(self.err("unterminated string"));
            }
            let b = self.input[self.pos];
            self.pos += 1;
            match b {
                b'"' => return Ok(s),
                b'\\' => {
                    if self.pos >= self.input.len() {
                        return Err(self.err("unterminated escape"));
                    }
                    let esc = self.input[self.pos];
                    self.pos += 1;
                    match esc {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'b' => s.push('\x08'),
                        b'f' => s.push('\x0c'),
                        b'u' => {
                            // \uXXXX — parse 4 hex digits
                            if self.pos + 4 > self.input.len() {
                                return Err(self.err("incomplete \\u escape"));
                            }
                            let hex = &self.input[self.pos..self.pos + 4];
                            self.pos += 4;
                            let hex_str = core::str::from_utf8(hex)
                                .map_err(|_| self.err("invalid \\u hex"))?;
                            let cp = u32::from_str_radix(hex_str, 16)
                                .map_err(|_| self.err("invalid \\u hex"))?;
                            if let Some(c) = char::from_u32(cp) {
                                s.push(c);
                            }
                        }
                        _ => {
                            s.push('\\');
                            s.push(esc as char);
                        }
                    }
                }
                _ => s.push(b as char),
            }
        }
    }

    fn parse_number(&mut self) -> Result<JsonValue, ParseError> {
        self.skip_whitespace();
        let start = self.pos;
        if self.pos < self.input.len() && self.input[self.pos] == b'-' {
            self.pos += 1;
        }
        while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos < self.input.len() && self.input[self.pos] == b'.' {
            self.pos += 1;
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        if self.pos < self.input.len()
            && (self.input[self.pos] == b'e' || self.input[self.pos] == b'E')
        {
            self.pos += 1;
            if self.pos < self.input.len()
                && (self.input[self.pos] == b'+' || self.input[self.pos] == b'-')
            {
                self.pos += 1;
            }
            while self.pos < self.input.len() && self.input[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        let slice = &self.input[start..self.pos];
        let s = core::str::from_utf8(slice).map_err(|_| self.err("invalid number bytes"))?;
        let n: f64 = s.parse().map_err(|_| self.err("invalid number"))?;
        Ok(JsonValue::Number(n))
    }

    fn parse_bool(&mut self) -> Result<JsonValue, ParseError> {
        self.skip_whitespace();
        if self.input[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Ok(JsonValue::Bool(true))
        } else if self.input[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Ok(JsonValue::Bool(false))
        } else {
            Err(self.err("expected 'true' or 'false'"))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, ParseError> {
        self.skip_whitespace();
        if self.input[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Ok(JsonValue::Null)
        } else {
            Err(self.err("expected 'null'"))
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, ParseError> {
        self.expect(b'{')?;
        self.open_stack.push(b'}');
        let mut pairs = Vec::new();
        if self.peek() == Some(b'}') {
            self.advance();
            self.open_stack.pop();
            return Ok(JsonValue::Object(pairs));
        }
        loop {
            let key = self.parse_string()?;
            self.expect(b':')?;
            let val = self.parse_value()?;
            pairs.push((key, val));
            match self.peek() {
                Some(b',') => {
                    self.advance();
                    // Allow trailing comma before closing brace
                    if self.peek() == Some(b'}') {
                        self.advance();
                        self.open_stack.pop();
                        return Ok(JsonValue::Object(pairs));
                    }
                }
                Some(b'}') => {
                    self.advance();
                    self.open_stack.pop();
                    return Ok(JsonValue::Object(pairs));
                }
                None => {
                    // Bracket recovery: EOF but we know we're inside an object
                    if self.open_stack.last().copied() == Some(b'}') {
                        self.open_stack.pop();
                        return Ok(JsonValue::Object(pairs));
                    }
                    return Err(self.err("unexpected EOF in object"));
                }
                _ => return Err(self.err("expected ',' or '}' in object")),
            }
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue, ParseError> {
        self.expect(b'[')?;
        self.open_stack.push(b']');
        let mut elems = Vec::new();
        if self.peek() == Some(b']') {
            self.advance();
            self.open_stack.pop();
            return Ok(JsonValue::Array(elems));
        }
        loop {
            let val = self.parse_value()?;
            elems.push(val);
            match self.peek() {
                Some(b',') => {
                    self.advance();
                    // Allow trailing comma before closing bracket
                    if self.peek() == Some(b']') {
                        self.advance();
                        self.open_stack.pop();
                        return Ok(JsonValue::Array(elems));
                    }
                }
                Some(b']') => {
                    self.advance();
                    self.open_stack.pop();
                    return Ok(JsonValue::Array(elems));
                }
                None => {
                    // Bracket recovery: EOF inside array
                    if self.open_stack.last().copied() == Some(b']') {
                        self.open_stack.pop();
                        return Ok(JsonValue::Array(elems));
                    }
                    return Err(self.err("unexpected EOF in array"));
                }
                _ => return Err(self.err("expected ',' or ']' in array")),
            }
        }
    }
}

/// Parse a JSON string into a [`JsonValue`].
///
/// Includes bracket recovery: if the input is valid except for missing
/// trailing `}` or `]` characters, the parser auto-closes them.
pub fn parse_json(input: &str) -> Result<JsonValue, ParseError> {
    let mut parser = Parser::new(input.as_bytes());
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.pos < parser.input.len() {
        return Err(parser.err("trailing data after JSON value"));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid() {
        let input = r#"{"version": 1, "kernel": "hamming_distance", "scan": {"threshold": 2048}}"#;
        let root = parse_json(input).unwrap();
        assert_eq!(root.get("version").unwrap().as_u64(), Some(1));
        assert_eq!(
            root.get("kernel").unwrap().as_str(),
            Some("hamming_distance")
        );
    }

    #[test]
    fn test_bracket_recovery_missing_closing_brace() {
        let input = r#"{"version": 1, "kernel": "hamming_distance", "scan": {"threshold": 1, "record_size": 64, "top_k": 5}"#;
        let root = parse_json(input).unwrap();
        assert_eq!(root.get("version").unwrap().as_u64(), Some(1));
    }

    #[test]
    fn test_bracket_recovery_missing_closing_bracket() {
        let input = r#"{"version": 1, "kernel": "dot_f32", "scan": {"threshold": 1, "record_size": 64, "top_k": 5}, "pipeline": [{"stage": "dot"}"#;
        let root = parse_json(input).unwrap();
        let pipeline = root.get("pipeline").unwrap().as_array().unwrap();
        assert_eq!(pipeline.len(), 1);
    }

    #[test]
    fn test_bracket_recovery_nested() {
        let input = r#"{"version": 1, "kernel": "cosine_i8", "scan": {"threshold": 100, "record_size": 128, "top_k": 3"#;
        let root = parse_json(input).unwrap();
        let scan = root.get("scan").unwrap();
        assert_eq!(scan.get("top_k").unwrap().as_u64(), Some(3));
    }

    #[test]
    fn test_trailing_comma() {
        let input = r#"{"version": 1, "kernel": "dot_f32", "scan": {"threshold": 1, "record_size": 64, "top_k": 5,},}"#;
        let root = parse_json(input).unwrap();
        assert_eq!(root.get("version").unwrap().as_u64(), Some(1));
    }

    #[test]
    fn test_single_line_comment() {
        let input = "{\n// this is a comment\n\"version\": 1, \"kernel\": \"dot_f32\", \"scan\": {\"threshold\": 1, \"record_size\": 64, \"top_k\": 5}}";
        let root = parse_json(input).unwrap();
        assert_eq!(root.get("version").unwrap().as_u64(), Some(1));
    }

    #[test]
    fn test_parse_error_bad_json() {
        let result = parse_json("{not json}");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_null_bool_array() {
        let input = r#"{"a": null, "b": true, "c": false, "d": [1, 2, 3]}"#;
        let root = parse_json(input).unwrap();
        assert_eq!(*root.get("a").unwrap(), JsonValue::Null);
        assert_eq!(root.get("b").unwrap().as_bool(), Some(true));
        assert_eq!(root.get("c").unwrap().as_bool(), Some(false));
        assert_eq!(root.get("d").unwrap().as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_parse_negative_and_float() {
        let input = r#"{"neg": -42, "flt": 3.14, "exp": 1e10}"#;
        let root = parse_json(input).unwrap();
        assert!((root.get("neg").unwrap().as_f64().unwrap() - (-42.0)).abs() < 1e-10);
        assert!((root.get("flt").unwrap().as_f64().unwrap() - 3.14).abs() < 1e-10);
        assert!((root.get("exp").unwrap().as_f64().unwrap() - 1e10).abs() < 1.0);
    }

    #[test]
    fn test_unicode_escape() {
        let input = r#"{"msg": "hello\u0020world"}"#;
        let root = parse_json(input).unwrap();
        assert_eq!(root.get("msg").unwrap().as_str(), Some("hello world"));
    }
}
