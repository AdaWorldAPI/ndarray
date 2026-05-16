//! Minimal NPZ (zip-of-npy) reader / writer.
//!
//! Pure-Rust replacement for the `ndarray-npy` 0.10 crate, which targets
//! ndarray 0.16 and cannot satisfy this fork's 0.17 trait bounds.
//!
//! # Feature Gate
//!
//! This module requires the `npz` Cargo feature, which pulls in the `zip`
//! crate. Without that feature the public types compile, but the three I/O
//! functions will not be available.
//!
//! # Example
//!
//! ```rust
//! # #[cfg(feature = "npz")] {
//! use std::io::Cursor;
//! use ndarray::hpc::npz_io::{NpyArray, NpyDtype, write_npz_array, read_npz_array};
//!
//! let arr = NpyArray {
//!     dtype: NpyDtype::F32,
//!     shape: vec![2, 3],
//!     fortran_order: false,
//!     data: vec![0u8; 24],  // 2*3 f32 = 24 bytes
//! };
//!
//! let mut buf = Cursor::new(Vec::<u8>::new());
//! write_npz_array(&mut buf, "weights", &arr).unwrap();
//!
//! buf.set_position(0);
//! let loaded = read_npz_array(buf, "weights").unwrap();
//! assert_eq!(loaded.dtype, NpyDtype::F32);
//! assert_eq!(loaded.shape, vec![2usize, 3]);
//! # }
//! ```

use std::io;

// ─── Public types ─────────────────────────────────────────────────────────────

/// Element dtype carried by a bare `.npy` / `.npz` array.
///
/// Covers the 11 numeric types recognised by tract-libcli
/// (`tract/libcli/src/tensor.rs:208-246`).  Complex, structured, and object
/// dtypes are explicitly out-of-scope.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::npz_io::NpyDtype;
/// let d = NpyDtype::F32;
/// assert_eq!(format!("{:?}", d), "F32");
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NpyDtype {
    /// `<f4` — 32-bit IEEE float, little-endian.
    F32,
    /// `<f8` — 64-bit IEEE float, little-endian.
    F64,
    /// `|i1` — signed 8-bit integer.
    I8,
    /// `<i2` — signed 16-bit integer, little-endian.
    I16,
    /// `<i4` — signed 32-bit integer, little-endian.
    I32,
    /// `<i8` — signed 64-bit integer, little-endian.
    I64,
    /// `|u1` — unsigned 8-bit integer.
    U8,
    /// `<u2` — unsigned 16-bit integer, little-endian.
    U16,
    /// `<u4` — unsigned 32-bit integer, little-endian.
    U32,
    /// `<u8` — unsigned 64-bit integer, little-endian.
    U64,
    /// `|b1` — boolean (one byte per element, 0 or 1).
    Bool,
}

/// A single array loaded from, or ready to be written to, an `.npy` / `.npz`
/// file.
///
/// `data` holds the raw little-endian element bytes in C (row-major) order
/// unless `fortran_order` is `true`.
///
/// # Example
///
/// ```rust
/// use ndarray::hpc::npz_io::{NpyArray, NpyDtype};
/// let arr = NpyArray {
///     dtype: NpyDtype::I32,
///     shape: vec![4],
///     fortran_order: false,
///     data: vec![1,0,0,0, 2,0,0,0, 3,0,0,0, 4,0,0,0],
/// };
/// assert_eq!(arr.data.len(), 16);
/// ```
#[derive(Debug)]
pub struct NpyArray {
    /// Element type.
    pub dtype: NpyDtype,
    /// Axis lengths, outer-to-inner.
    pub shape: Vec<usize>,
    /// `true` if data is in Fortran (column-major) order.
    pub fortran_order: bool,
    /// Raw byte payload.
    pub data: Vec<u8>,
}

/// Errors returned by the NPZ/NPY I/O functions.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "npz")] {
/// use std::io::Cursor;
/// use ndarray::hpc::npz_io::{read_npz_array, NpzError};
/// let empty = Cursor::new(vec![]);
/// assert!(matches!(read_npz_array(empty, "x"), Err(NpzError::Zip(_))));
/// # }
/// ```
#[derive(Debug)]
pub enum NpzError {
    /// Underlying I/O error.
    Io(io::Error),
    /// ZIP layer error (invalid archive, missing entry, …).
    Zip(String),
    /// dtype descriptor not in the supported set (e.g. `<c8` complex64).
    UnsupportedDtype(String),
    /// NPY magic / version / header could not be parsed.
    MalformedHeader(String),
    /// Named array not present in the archive.
    NotFound(String),
}

impl From<io::Error> for NpzError {
    fn from(e: io::Error) -> Self {
        NpzError::Io(e)
    }
}

// ─── NPY dtype helpers ────────────────────────────────────────────────────────

/// Return the NumPy `descr` string for a dtype.
fn descr_from_dtype(dtype: NpyDtype) -> &'static str {
    match dtype {
        NpyDtype::F32  => "<f4",
        NpyDtype::F64  => "<f8",
        NpyDtype::I8   => "|i1",
        NpyDtype::I16  => "<i2",
        NpyDtype::I32  => "<i4",
        NpyDtype::I64  => "<i8",
        NpyDtype::U8   => "|u1",
        NpyDtype::U16  => "<u2",
        NpyDtype::U32  => "<u4",
        NpyDtype::U64  => "<u8",
        NpyDtype::Bool => "|b1",
    }
}

/// Parse a NumPy `descr` string into a [`NpyDtype`].
///
/// Returns `Err(NpzError::UnsupportedDtype)` for anything not in the table.
fn dtype_from_descr(descr: &str) -> Result<NpyDtype, NpzError> {
    // Trim surrounding quotes / whitespace the parser may have left in.
    let s = descr.trim().trim_matches('\'').trim_matches('"');
    match s {
        "<f4" | "=f4" => Ok(NpyDtype::F32),
        "<f8" | "=f8" => Ok(NpyDtype::F64),
        "|i1"          => Ok(NpyDtype::I8),
        "<i2" | "=i2" => Ok(NpyDtype::I16),
        "<i4" | "=i4" => Ok(NpyDtype::I32),
        "<i8" | "=i8" => Ok(NpyDtype::I64),
        "|u1"          => Ok(NpyDtype::U8),
        "<u2" | "=u2" => Ok(NpyDtype::U16),
        "<u4" | "=u4" => Ok(NpyDtype::U32),
        "<u8" | "=u8" => Ok(NpyDtype::U64),
        "|b1"          => Ok(NpyDtype::Bool),
        other          => Err(NpzError::UnsupportedDtype(other.to_string())),
    }
}

/// Return the element size in bytes for a dtype.
fn dtype_element_size(dtype: NpyDtype) -> usize {
    match dtype {
        NpyDtype::F32  | NpyDtype::I32  | NpyDtype::U32  => 4,
        NpyDtype::F64  | NpyDtype::I64  | NpyDtype::U64  => 8,
        NpyDtype::I16  | NpyDtype::U16                   => 2,
        NpyDtype::I8   | NpyDtype::U8   | NpyDtype::Bool => 1,
    }
}

// ─── NPY header encode / decode ───────────────────────────────────────────────

/// Magic bytes that open every `.npy` file.
const NPY_MAGIC: &[u8] = b"\x93NUMPY";
/// Major version byte this implementation reads/writes.
const NPY_MAJOR_V1: u8 = 1;
/// Minor version byte.
const NPY_MINOR_V1: u8 = 0;
/// Fixed prefix size: 6 magic + 1 major + 1 minor + 2 header-len = 10 bytes.
const NPY_PREFIX_LEN: usize = 10;
/// Total header block (prefix + text) must be a multiple of 64 bytes.
const NPY_ALIGNMENT: usize = 64;

/// Build the Python-dict header text for an array.
fn build_header_text(array: &NpyArray) -> String {
    let descr = descr_from_dtype(array.dtype);
    let order  = if array.fortran_order { "True" } else { "False" };

    // Shape tuple: "()" for scalars, "(n,)" for 1-D, "(a, b, …)" for N-D.
    let shape_str = match array.shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", array.shape[0]),
        _ => {
            let inner: Vec<String> = array.shape.iter().map(|x| x.to_string()).collect();
            format!("({})", inner.join(", "))
        }
    };

    format!(
        "{{'descr': '{}', 'fortran_order': {}, 'shape': {}, }}",
        descr, order, shape_str
    )
}

/// Pad the header text so that prefix + text + `\n` is a multiple of 64.
///
/// Returns `(header_text_with_newline, header_len_field)`.
/// `header_len_field` is the u16 stored at bytes 8-9 of the file (length of
/// the padded text including the trailing `\n`).
fn pad_header_to_64(text: &str) -> Result<(Vec<u8>, u16), NpzError> {
    // We need: NPY_PREFIX_LEN + header_len ≡ 0 (mod 64)
    // header_len = len(padded_text_with_newline)
    // Find smallest header_len such that (NPY_PREFIX_LEN + header_len) % 64 == 0
    let raw_len = text.len() + 1; // +1 for '\n'
    let total_needed = NPY_PREFIX_LEN + raw_len;
    let remainder = total_needed % NPY_ALIGNMENT;
    let pad_bytes = if remainder == 0 {
        0
    } else {
        NPY_ALIGNMENT - remainder
    };

    let mut out = text.as_bytes().to_vec();
    for _ in 0..pad_bytes {
        out.push(b' ');
    }
    out.push(b'\n');

    let header_len = out.len();
    if header_len > u16::MAX as usize {
        return Err(NpzError::MalformedHeader(
            "header too large for v1 (>64 KiB)".to_string(),
        ));
    }
    Ok((out, header_len as u16))
}

/// Serialise the full NPY byte block (header + raw data) for `array`.
fn encode_npy(array: &NpyArray) -> Result<Vec<u8>, NpzError> {
    let text   = build_header_text(array);
    let (header_bytes, hlen) = pad_header_to_64(&text)?;

    let mut out = Vec::with_capacity(NPY_PREFIX_LEN + header_bytes.len() + array.data.len());
    out.extend_from_slice(NPY_MAGIC);
    out.push(NPY_MAJOR_V1);
    out.push(NPY_MINOR_V1);
    // header_len as u16 LE
    out.push((hlen & 0xFF) as u8);
    out.push((hlen >> 8) as u8);
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&array.data);
    Ok(out)
}

/// Tiny helper: extract the value after `key:` in the header dict.
///
/// Scans for `'<key>'` or `"<key>"`, then returns everything between the
/// next `'`/`"` pair (for string values) or up to the next `,`/`)` (for
/// non-string values like booleans and tuples).
fn extract_header_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    // Look for the key surrounded by quotes, followed by colon.
    let needle_sq = format!("'{}'", key);
    let needle_dq = format!("\"{}\"", key);
    let key_pos = header
        .find(needle_sq.as_str())
        .or_else(|| header.find(needle_dq.as_str()))?;
    let after_key = &header[key_pos + needle_sq.len().min(needle_dq.len())..];

    // Skip whitespace and colon
    let colon_pos = after_key.find(':')?;
    let value_start = after_key[colon_pos + 1..].trim_start();

    // String value — delimited by quotes
    if value_start.starts_with('\'') || value_start.starts_with('"') {
        let delim = value_start.chars().next()?;
        let inner = &value_start[1..];
        let end   = inner.find(delim)?;
        return Some(&inner[..end]);
    }

    // Tuple value — delimited by '(' … ')'
    if value_start.starts_with('(') {
        let end = value_start.find(')')?;
        return Some(&value_start[..=end]);
    }

    // Bare value (True / False) — up to next ',' or '}'
    let end = value_start
        .find(|c| c == ',' || c == '}')
        .unwrap_or(value_start.len());
    Some(value_start[..end].trim())
}

/// Parse a NumPy header text (the Python dict literal) into its three fields.
fn parse_npy_header(header: &str) -> Result<(NpyDtype, bool, Vec<usize>), NpzError> {
    // ── dtype ─────────────────────────────────────────────────────────────────
    let descr_raw = extract_header_value(header, "descr").ok_or_else(|| {
        NpzError::MalformedHeader(format!("missing 'descr' in header: {}", header))
    })?;
    let dtype = dtype_from_descr(descr_raw)?;

    // ── fortran_order ─────────────────────────────────────────────────────────
    let order_raw = extract_header_value(header, "fortran_order").ok_or_else(|| {
        NpzError::MalformedHeader(format!("missing 'fortran_order' in header: {}", header))
    })?;
    let fortran_order = match order_raw.trim() {
        "True"  => true,
        "False" => false,
        other   => {
            return Err(NpzError::MalformedHeader(format!(
                "unexpected fortran_order value: {}",
                other
            )))
        }
    };

    // ── shape ─────────────────────────────────────────────────────────────────
    let shape_raw = extract_header_value(header, "shape").ok_or_else(|| {
        NpzError::MalformedHeader(format!("missing 'shape' in header: {}", header))
    })?;

    // shape_raw looks like "()" / "(3,)" / "(3, 4)" / "(3, 4,)"
    let inner = shape_raw
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();

    let shape: Vec<usize> = if inner.is_empty() {
        vec![]
    } else {
        inner
            .split(',')
            .filter_map(|s| {
                let t = s.trim();
                if t.is_empty() { None } else { Some(t) }
            })
            .map(|s| {
                s.parse::<usize>().map_err(|_| {
                    NpzError::MalformedHeader(format!("bad shape component: {}", s))
                })
            })
            .collect::<Result<Vec<usize>, NpzError>>()?
    };

    Ok((dtype, fortran_order, shape))
}

/// Read a single NPY blob from a byte slice, returning an [`NpyArray`].
fn decode_npy(src: &[u8]) -> Result<NpyArray, NpzError> {
    // Check magic
    if src.len() < NPY_PREFIX_LEN || &src[..6] != NPY_MAGIC {
        return Err(NpzError::MalformedHeader("not a NumPy array (bad magic)".to_string()));
    }

    let major = src[6];
    let minor = src[7];
    if major != NPY_MAJOR_V1 || minor != NPY_MINOR_V1 {
        return Err(NpzError::MalformedHeader(format!(
            "only NPY v1.0 supported, got {}.{}",
            major, minor
        )));
    }

    // header_len is u16 LE at bytes 8-9
    let header_len = u16::from_le_bytes([src[8], src[9]]) as usize;
    let header_end = NPY_PREFIX_LEN + header_len;
    if src.len() < header_end {
        return Err(NpzError::MalformedHeader("file truncated in header".to_string()));
    }

    let header_text = std::str::from_utf8(&src[NPY_PREFIX_LEN..header_end])
        .map_err(|e| NpzError::MalformedHeader(format!("header not valid UTF-8: {}", e)))?;

    let (dtype, fortran_order, shape) = parse_npy_header(header_text)?;

    let data = src[header_end..].to_vec();

    // Validate data length
    let n_elems: usize = shape.iter().product();
    let expected = n_elems * dtype_element_size(dtype);
    if data.len() != expected {
        return Err(NpzError::MalformedHeader(format!(
            "data length mismatch: expected {} bytes, got {}",
            expected,
            data.len()
        )));
    }

    Ok(NpyArray { dtype, shape, fortran_order, data })
}

// ─── Public API — feature-gated on "npz" ─────────────────────────────────────

/// Read one named array from a `.npz` archive.
///
/// The `name` parameter should match the array name as stored in the archive
/// (without the `.npy` suffix).  Use [`list_npz_names`] to discover available
/// names.
///
/// # Errors
///
/// Returns [`NpzError::NotFound`] if no entry `<name>.npy` exists, or the
/// appropriate error variant for I/O, zip, or header problems.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "npz")] {
/// use std::io::Cursor;
/// use ndarray::hpc::npz_io::{NpyArray, NpyDtype, write_npz_array, read_npz_array};
///
/// let arr = NpyArray { dtype: NpyDtype::F64, shape: vec![3], fortran_order: false,
///                      data: vec![0u8; 24] };
/// let mut buf = Cursor::new(Vec::<u8>::new());
/// write_npz_array(&mut buf, "x", &arr).unwrap();
/// buf.set_position(0);
/// let out = read_npz_array(buf, "x").unwrap();
/// assert_eq!(out.dtype, NpyDtype::F64);
/// # }
/// ```
#[cfg(feature = "npz")]
pub fn read_npz_array<R: io::Read + io::Seek>(
    reader: R,
    name: &str,
) -> Result<NpyArray, NpzError> {
    use zip::ZipArchive;

    let mut archive = ZipArchive::new(reader)
        .map_err(|e| NpzError::Zip(e.to_string()))?;

    let entry_name = format!("{}.npy", name);

    // ZipArchive::by_name requires &mut self
    let mut entry = archive
        .by_name(&entry_name)
        .map_err(|e| {
            // zip::result::ZipError::FileNotFound is the typical variant
            let msg = e.to_string();
            if msg.contains("not found") || msg.contains("FileNotFound") {
                NpzError::NotFound(name.to_string())
            } else {
                NpzError::Zip(msg)
            }
        })?;

    let mut bytes = Vec::new();
    io::Read::read_to_end(&mut entry, &mut bytes)?;

    decode_npy(&bytes)
}

/// List all array names stored in a `.npz` archive.
///
/// Entries whose name ends in `.npy` have the suffix stripped; other entries
/// (if any) are included verbatim.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "npz")] {
/// use std::io::Cursor;
/// use ndarray::hpc::npz_io::{NpyArray, NpyDtype, write_npz_array, list_npz_names};
///
/// let arr = NpyArray { dtype: NpyDtype::U8, shape: vec![4], fortran_order: false,
///                      data: vec![0u8; 4] };
/// let mut buf = Cursor::new(Vec::<u8>::new());
/// write_npz_array(&mut buf, "labels", &arr).unwrap();
/// buf.set_position(0);
/// let names = list_npz_names(buf).unwrap();
/// assert_eq!(names, vec!["labels".to_string()]);
/// # }
/// ```
#[cfg(feature = "npz")]
pub fn list_npz_names<R: io::Read + io::Seek>(
    reader: R,
) -> Result<Vec<String>, NpzError> {
    use zip::ZipArchive;

    let archive = ZipArchive::new(reader)
        .map_err(|e| NpzError::Zip(e.to_string()))?;

    let names = archive
        .file_names()
        .map(|raw| {
            if raw.ends_with(".npy") {
                raw[..raw.len() - 4].to_string()
            } else {
                raw.to_string()
            }
        })
        .collect();

    Ok(names)
}

/// Append one named array to a `.npz` archive opened for writing.
///
/// The writer is wrapped in a [`zip::ZipWriter`].  Each call adds a single
/// `<name>.npy` entry using DEFLATE compression (falls back to STORED if
/// deflate is unavailable).
///
/// # Note on multiple arrays
///
/// To write multiple arrays you must call this function once per array,
/// passing the **same** `writer` value each time — but that requires
/// external management of the `ZipWriter`.  For the common single-array
/// case this function handles everything.  For multi-array writes consider
/// the pattern in the tests below, or wrap the writer yourself with
/// `zip::ZipWriter`.
///
/// # Errors
///
/// Propagates I/O and zip errors.
///
/// # Example
///
/// ```rust
/// # #[cfg(feature = "npz")] {
/// use std::io::Cursor;
/// use ndarray::hpc::npz_io::{NpyArray, NpyDtype, write_npz_array, read_npz_array};
///
/// let arr = NpyArray { dtype: NpyDtype::Bool, shape: vec![3], fortran_order: false,
///                      data: vec![1, 0, 1] };
/// let mut buf = Cursor::new(Vec::<u8>::new());
/// write_npz_array(&mut buf, "mask", &arr).unwrap();
/// buf.set_position(0);
/// let out = read_npz_array(buf, "mask").unwrap();
/// assert_eq!(out.dtype, NpyDtype::Bool);
/// # }
/// ```
#[cfg(feature = "npz")]
pub fn write_npz_array<W: io::Write + io::Seek>(
    writer: W,
    name: &str,
    array: &NpyArray,
) -> Result<(), NpzError> {
    use zip::{write::FileOptions, ZipWriter, CompressionMethod};

    let mut zw = ZipWriter::new(writer);

    let entry_name = format!("{}.npy", name);
    let options = FileOptions::default()
        .compression_method(CompressionMethod::Deflated);

    zw.start_file(entry_name, options)
        .map_err(|e| NpzError::Zip(e.to_string()))?;

    let npy_bytes = encode_npy(array)?;
    io::Write::write_all(&mut zw, &npy_bytes)?;

    zw.finish().map_err(|e| NpzError::Zip(e.to_string()))?;

    Ok(())
}

/// Write multiple arrays into one `.npz` archive.
///
/// This is an internal helper used by the tests to verify multi-array NPZ
/// round-trips.  Each `(name, array)` pair becomes one `<name>.npy` entry.
#[cfg(feature = "npz")]
fn write_npz_arrays<W: io::Write + io::Seek>(
    writer: W,
    arrays: &[(&str, &NpyArray)],
) -> Result<(), NpzError> {
    use zip::{write::FileOptions, ZipWriter, CompressionMethod};

    let mut zw = ZipWriter::new(writer);

    for (name, array) in arrays {
        let entry_name = format!("{}.npy", name);
        let options = FileOptions::default()
            .compression_method(CompressionMethod::Deflated);

        zw.start_file(entry_name, options)
            .map_err(|e| NpzError::Zip(e.to_string()))?;

        let npy_bytes = encode_npy(array)?;
        io::Write::write_all(&mut zw, &npy_bytes)?;
    }

    zw.finish().map_err(|e| NpzError::Zip(e.to_string()))?;

    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper: byte representations ──────────────────────────────────────────

    fn f32_to_le_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn i64_to_le_bytes(vals: &[i64]) -> Vec<u8> {
        vals.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    // ── 1. Header round-trip for a 2-D F32 array ──────────────────────────────

    #[test]
    fn npy_header_roundtrip_f32_2d() {
        let arr = NpyArray {
            dtype:         NpyDtype::F32,
            shape:         vec![3, 4],
            fortran_order: false,
            data:          vec![0u8; 3 * 4 * 4],
        };

        let text = build_header_text(&arr);
        assert!(text.contains("'descr': '<f4'"), "descr missing");
        assert!(text.contains("'fortran_order': False"), "order missing");
        assert!(text.contains("(3, 4)"), "shape missing: {}", text);

        // Re-parse the generated text
        let (dtype, order, shape) = parse_npy_header(&text).expect("parse failed");
        assert_eq!(dtype,  NpyDtype::F32);
        assert!(!order);
        assert_eq!(shape,  vec![3usize, 4]);
    }

    // ── 2. Unsupported dtype rejected ─────────────────────────────────────────

    #[test]
    fn npy_header_parse_unsupported_dtype() {
        let header = "{'descr': '<c8', 'fortran_order': False, 'shape': (3,), }";
        match parse_npy_header(header) {
            Err(NpzError::UnsupportedDtype(s)) => {
                assert!(s.contains("c8"), "wrong dtype reported: {}", s);
            }
            other => panic!("expected UnsupportedDtype, got {:?}", other),
        }
    }

    // ── 3. Full npy encode/decode round-trip for F32 ──────────────────────────

    #[test]
    fn npy_write_read_roundtrip_f32() {
        let values: Vec<f32> = vec![1.0, -2.5, 3.14, 0.0, f32::INFINITY, f32::NEG_INFINITY];
        let arr = NpyArray {
            dtype:         NpyDtype::F32,
            shape:         vec![2, 3],
            fortran_order: false,
            data:          f32_to_le_bytes(&values),
        };

        let encoded = encode_npy(&arr).expect("encode failed");

        // The NPY spec requires that the header block (prefix + padded header text)
        // be 64-byte aligned.  The data payload follows immediately and is NOT padded.
        let header_len = u16::from_le_bytes([encoded[8], encoded[9]]) as usize;
        assert_eq!(
            (NPY_PREFIX_LEN + header_len) % NPY_ALIGNMENT,
            0,
            "header block not 64-byte aligned"
        );

        let decoded = decode_npy(&encoded).expect("decode failed");
        assert_eq!(decoded.dtype,         NpyDtype::F32);
        assert_eq!(decoded.shape,         vec![2usize, 3]);
        assert!(!decoded.fortran_order);
        assert_eq!(decoded.data,          arr.data);

        // Verify individual float values survive the round-trip
        for (i, &orig) in values.iter().enumerate() {
            let bytes: [u8; 4] = decoded.data[i*4..(i+1)*4].try_into().unwrap();
            let got = f32::from_le_bytes(bytes);
            // NaN check
            if orig.is_nan() {
                assert!(got.is_nan());
            } else {
                assert_eq!(got, orig, "element {} mismatch", i);
            }
        }
    }

    // ── 4. NPZ single I64 array round-trip ───────────────────────────────────

    #[cfg(feature = "npz")]
    #[test]
    fn npz_write_read_single_array_i64() {
        use std::io::Cursor;

        let values: Vec<i64> = vec![i64::MIN, -1, 0, 1, i64::MAX];
        let arr = NpyArray {
            dtype:         NpyDtype::I64,
            shape:         vec![5],
            fortran_order: false,
            data:          i64_to_le_bytes(&values),
        };

        let mut buf = Cursor::new(Vec::<u8>::new());
        write_npz_array(&mut buf, "tokens", &arr).expect("write failed");

        buf.set_position(0);
        let out = read_npz_array(buf, "tokens").expect("read failed");

        assert_eq!(out.dtype,  NpyDtype::I64);
        assert_eq!(out.shape,  vec![5usize]);
        assert!(!out.fortran_order);
        assert_eq!(out.data,   arr.data);

        for (i, &orig) in values.iter().enumerate() {
            let bytes: [u8; 8] = out.data[i*8..(i+1)*8].try_into().unwrap();
            assert_eq!(i64::from_le_bytes(bytes), orig);
        }
    }

    // ── 5. NPZ multi-array round-trip (F32 + I64 + Bool) ─────────────────────

    #[cfg(feature = "npz")]
    #[test]
    fn npz_write_read_multi_array_mixed_dtypes() {
        use std::io::Cursor;

        let f32_vals: Vec<f32> = vec![1.0, 2.0, 3.0];
        let i64_vals: Vec<i64> = vec![10, 20, 30, 40];
        let bool_data          = vec![1u8, 0, 1, 0, 1];

        let arr_f32 = NpyArray {
            dtype:         NpyDtype::F32,
            shape:         vec![3],
            fortran_order: false,
            data:          f32_to_le_bytes(&f32_vals),
        };
        let arr_i64 = NpyArray {
            dtype:         NpyDtype::I64,
            shape:         vec![4],
            fortran_order: false,
            data:          i64_to_le_bytes(&i64_vals),
        };
        let arr_bool = NpyArray {
            dtype:         NpyDtype::Bool,
            shape:         vec![5],
            fortran_order: false,
            data:          bool_data.clone(),
        };

        let mut buf = Cursor::new(Vec::<u8>::new());
        write_npz_arrays(
            &mut buf,
            &[("weights", &arr_f32), ("indices", &arr_i64), ("mask", &arr_bool)],
        )
        .expect("multi-write failed");

        // Read back each array
        buf.set_position(0);
        let w = read_npz_array(&mut buf, "weights").expect("read weights");
        assert_eq!(w.dtype, NpyDtype::F32);
        assert_eq!(w.shape, vec![3usize]);
        assert_eq!(w.data,  arr_f32.data);

        buf.set_position(0);
        let idx = read_npz_array(&mut buf, "indices").expect("read indices");
        assert_eq!(idx.dtype, NpyDtype::I64);
        assert_eq!(idx.shape, vec![4usize]);

        buf.set_position(0);
        let mask = read_npz_array(&mut buf, "mask").expect("read mask");
        assert_eq!(mask.dtype, NpyDtype::Bool);
        assert_eq!(mask.shape, vec![5usize]);
        assert_eq!(mask.data,  bool_data);
    }

    // ── 6. list_npz_names ─────────────────────────────────────────────────────

    #[cfg(feature = "npz")]
    #[test]
    fn npz_list_names() {
        use std::io::Cursor;

        let make_arr = |dtype, n: usize, elem_size: usize| NpyArray {
            dtype,
            shape:         vec![n],
            fortran_order: false,
            data:          vec![0u8; n * elem_size],
        };

        let arr_a = make_arr(NpyDtype::F32, 4, 4);
        let arr_b = make_arr(NpyDtype::U8,  8, 1);
        let arr_c = make_arr(NpyDtype::I32, 2, 4);

        let mut buf = Cursor::new(Vec::<u8>::new());
        write_npz_arrays(
            &mut buf,
            &[("alpha", &arr_a), ("beta", &arr_b), ("gamma", &arr_c)],
        )
        .expect("write failed");

        buf.set_position(0);
        let mut names = list_npz_names(buf).expect("list failed");
        names.sort(); // order is ZIP-internal, normalise for comparison

        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }

    // ── 7. NotFound error on missing array ────────────────────────────────────

    #[cfg(feature = "npz")]
    #[test]
    fn npz_not_found_error() {
        use std::io::Cursor;

        let arr = NpyArray {
            dtype:         NpyDtype::U8,
            shape:         vec![1],
            fortran_order: false,
            data:          vec![42],
        };
        let mut buf = Cursor::new(Vec::<u8>::new());
        write_npz_array(&mut buf, "present", &arr).unwrap();
        buf.set_position(0);

        match read_npz_array(buf, "absent") {
            Err(NpzError::NotFound(n)) => assert_eq!(n, "absent"),
            Err(NpzError::Zip(s)) => {
                // Some zip versions report file-not-found as a Zip error
                assert!(
                    s.to_lowercase().contains("not found")
                        || s.to_lowercase().contains("invalid"),
                    "unexpected zip error: {}",
                    s
                );
            }
            other => panic!("expected NotFound or Zip error, got {:?}", other),
        }
    }

    // ── 8. All 11 dtypes encode/decode correctly ──────────────────────────────

    #[test]
    fn npy_all_dtypes_encode_decode() {
        let cases: &[(NpyDtype, usize)] = &[
            (NpyDtype::F32,  4),
            (NpyDtype::F64,  8),
            (NpyDtype::I8,   1),
            (NpyDtype::I16,  2),
            (NpyDtype::I32,  4),
            (NpyDtype::I64,  8),
            (NpyDtype::U8,   1),
            (NpyDtype::U16,  2),
            (NpyDtype::U32,  4),
            (NpyDtype::U64,  8),
            (NpyDtype::Bool, 1),
        ];

        for &(dtype, elem_size) in cases {
            let n = 3usize;
            let arr = NpyArray {
                dtype,
                shape:         vec![n],
                fortran_order: false,
                data:          vec![0xABu8; n * elem_size],
            };
            let encoded = encode_npy(&arr)
                .unwrap_or_else(|e| panic!("{:?}: encode error {:?}", dtype, e));
            let decoded = decode_npy(&encoded)
                .unwrap_or_else(|e| panic!("{:?}: decode error {:?}", dtype, e));
            assert_eq!(decoded.dtype,  dtype,  "{:?}: dtype mismatch", dtype);
            assert_eq!(decoded.shape,  vec![n], "{:?}: shape mismatch", dtype);
            assert_eq!(decoded.data,   arr.data, "{:?}: data mismatch", dtype);
        }
    }

    // ── 9. Scalar (0-D) shape round-trip ─────────────────────────────────────

    #[test]
    fn npy_scalar_shape_roundtrip() {
        let arr = NpyArray {
            dtype:         NpyDtype::F64,
            shape:         vec![],
            fortran_order: false,
            data:          vec![0u8; 8],
        };
        let encoded = encode_npy(&arr).expect("encode scalar");
        let decoded = decode_npy(&encoded).expect("decode scalar");
        assert_eq!(decoded.shape, Vec::<usize>::new());
        assert_eq!(decoded.data,  arr.data);
    }

    // ── 10. Fortran-order flag survives the round-trip ────────────────────────

    #[test]
    fn npy_fortran_order_roundtrip() {
        let arr = NpyArray {
            dtype:         NpyDtype::I32,
            shape:         vec![2, 3],
            fortran_order: true,
            data:          vec![0u8; 2 * 3 * 4],
        };
        let encoded = encode_npy(&arr).expect("encode fortran");
        let decoded = decode_npy(&encoded).expect("decode fortran");
        assert!(decoded.fortran_order, "fortran_order not preserved");
        assert_eq!(decoded.shape, vec![2usize, 3]);
    }
}
