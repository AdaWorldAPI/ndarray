//! rANS entropy coder over the 4-symbol mode alphabet (PR-X12 A7).
//!
//! The codec's mode-decision pass ([`super::predict`] / [`super::rdo`])
//! emits one [`CellMode`] tag per cell. Those tags are *not* uniformly
//! distributed — coherent cognitive state is dominated by `Skip`
//! (~70%) with a long tail of `Merge`/`Delta`/`Escape` (see the design
//! doc's compression target). A 2-bit fixed encoding wastes the
//! redundancy; an entropy coder spends `-log2(p)` bits per tag instead.
//!
//! This module ships a **range Asymmetric Numeral System** (rANS)
//! coder, chosen over CABAC per the design's open-Q1 ruling: a single
//! multiply + table lookup per symbol (cache-friendly), no per-bit
//! context-state branches, and within ~0.5% of CABAC's ratio on typical
//! streams (the variant zstd uses). It encodes/decodes the **mode-tag
//! stream only** — the per-mode payloads (`basin_idx`, `delta`,
//! `escape_idx`) are packed separately by [`super::mode`].
//!
//! # Static per-block frequency table (not backward-adaptive)
//!
//! v1 uses a **static** frequency table computed from the block's own
//! mode histogram and written into the stream header. This is bit-exact
//! and free of the adaptation-replay hazard that backward-adaptive rANS
//! has (the encoder processes symbols in reverse while the decoder
//! processes them forward, so an adaptive model would have to be
//! replayed in two directions). "Per-CTU" granularity (design open-Q5)
//! is achieved by calling [`encode_modes`] once per CTU's tag stream.
//! Backward-adaptive coding is a documented follow-on.
//!
//! # Stream layout (`encode_modes` / `decode_modes`)
//!
//! ```text
//!   bytes [0..4)   symbol count n               (u32, little-endian)
//!   bytes [4..12)  normalized freq table        (4 × u16, little-endian)
//!   bytes [12..)   rANS payload                 (state-flushed bytes)
//! ```
//!
//! The freq table entries sum to [`RANS_M`] (= 4096); a symbol that
//! never appears has frequency 0 and is never encoded. The payload is
//! self-delimiting given `n` and the table — the decoder reads exactly
//! the bytes the encoder produced.
//!
//! # Numerics
//!
//! State is `u32`, kept in the normalized interval `[RANS_BYTE_L,
//! RANS_BYTE_L << 8)`. With [`RANS_SCALE_BITS`] = 12 and
//! [`RANS_BYTE_L`] = `1 << 23`, every intermediate (`x / f`,
//! `(x / f) << 12`, `x_max = (RANS_BYTE_L >> 12 << 8) * f`) stays within
//! `u32` for all `f ∈ [1, 4096]`. No `unsafe`, no float.
//!
//! # What this module does NOT do
//!
//! - **Payload coding** — `basin_idx` / `delta` / `escape_idx` bytes are
//!   [`super::mode`]'s responsibility; A7 codes only the mode tags.
//! - **Framing** (frame headers, CTU markers) — PR-X12 A8 `stream.rs`.
//! - **Backward-adaptive frequency models** — documented follow-on.

use super::ctu::CellMode;

// ════════════════════════════════════════════════════════════════════
// Constants
// ════════════════════════════════════════════════════════════════════

/// Frequency-table precision in bits. The table's frequencies sum to
/// `1 << RANS_SCALE_BITS`. 12 bits (4096) is ample for a 4-symbol
/// alphabet and keeps every rANS intermediate inside `u32`.
pub const RANS_SCALE_BITS: u32 = 12;

/// Total of all symbol frequencies (`1 << RANS_SCALE_BITS` = 4096).
pub const RANS_M: u32 = 1 << RANS_SCALE_BITS;

/// Lower bound of the normalized state interval. The state `x` is kept
/// in `[RANS_BYTE_L, RANS_BYTE_L << 8)` via byte-wise renormalization.
pub const RANS_BYTE_L: u32 = 1 << 23;

/// Fixed 4-symbol alphabet size (`Skip`/`Merge`/`Delta`/`Escape`).
const ALPHABET: usize = 4;

/// Header byte length: `u32` count + 4 × `u16` frequencies.
const HEADER_LEN: usize = 4 + ALPHABET * 2;

// ════════════════════════════════════════════════════════════════════
// Frequency table
// ════════════════════════════════════════════════════════════════════

/// Normalized symbol-frequency table for one block of mode tags.
///
/// `freq[s]` is symbol `s`'s frequency (summing to [`RANS_M`]); `cum[s]`
/// is the exclusive prefix sum (the symbol's start of its sub-interval
/// in `[0, RANS_M)`). A symbol absent from the source histogram has
/// `freq = 0` and is never emitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RansFreqTable {
    /// Per-symbol frequency, indexed by [`CellMode`] discriminant.
    /// Sums to [`RANS_M`] for any non-empty histogram.
    pub freq: [u16; ALPHABET],
    /// Exclusive prefix sum of `freq` — `cum[s] = Σ freq[0..s]`.
    pub cum: [u16; ALPHABET],
}

impl RansFreqTable {
    /// Build a normalized table from a raw mode histogram.
    ///
    /// Each non-zero count is scaled to `count * RANS_M / total`, floored
    /// but bumped to a minimum of 1 so any present symbol stays encodable.
    /// The rounding drift is absorbed by the most-frequent symbol so the
    /// frequencies sum to exactly [`RANS_M`].
    ///
    /// An all-zero histogram (no symbols) yields an all-zero table; it is
    /// only valid for an empty stream (`n == 0`), which never encodes a
    /// symbol.
    ///
    /// ```
    /// use ndarray::hpc::codec::ans::{RansFreqTable, RANS_M};
    /// // 3:1 Skip:Delta split → frequencies sum to RANS_M.
    /// let t = RansFreqTable::from_histogram([3, 0, 1, 0]);
    /// assert_eq!(t.freq.iter().map(|&f| f as u32).sum::<u32>(), RANS_M);
    /// assert!(t.freq[0] > t.freq[2]); // Skip more probable than Delta
    /// ```
    pub fn from_histogram(counts: [u32; ALPHABET]) -> Self {
        let total: u64 = counts.iter().map(|&c| c as u64).sum();
        let mut freq = [0u16; ALPHABET];
        if total == 0 {
            return Self {
                freq,
                cum: [0; ALPHABET],
            };
        }

        let mut sum: u32 = 0;
        let mut max_idx = 0usize;
        let mut max_count = 0u32;
        for (i, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let scaled = (count as u64 * RANS_M as u64) / total;
            let f = (scaled as u32).max(1);
            freq[i] = f as u16;
            sum += f;
            if count > max_count {
                max_count = count;
                max_idx = i;
            }
        }

        // Absorb floor/clamp drift on the most-frequent symbol so the
        // table sums to exactly RANS_M. The max bucket has the largest
        // frequency, so a small adjustment can never drive it below 1.
        let adjusted = freq[max_idx] as i64 + (RANS_M as i64 - sum as i64);
        debug_assert!((1..=RANS_M as i64).contains(&adjusted), "freq adjustment out of range: {adjusted}");
        freq[max_idx] = adjusted as u16;

        Self {
            freq,
            cum: prefix_sum(&freq),
        }
    }

    /// Compute the table from a slice of mode tags.
    ///
    /// ```
    /// use ndarray::hpc::codec::ans::RansFreqTable;
    /// use ndarray::hpc::codec::CellMode;
    /// let modes = [CellMode::Skip, CellMode::Skip, CellMode::Delta];
    /// let t = RansFreqTable::from_symbols(&modes);
    /// assert!(t.freq[CellMode::Skip as usize] > t.freq[CellMode::Delta as usize]);
    /// ```
    pub fn from_symbols(symbols: &[CellMode]) -> Self {
        let mut counts = [0u32; ALPHABET];
        for &s in symbols {
            counts[s as usize] += 1;
        }
        Self::from_histogram(counts)
    }

    /// Reconstruct a table from the 4 little-endian frequencies stored in
    /// a stream header. Returns `None` if the frequencies don't sum to
    /// [`RANS_M`] (a non-empty table) and aren't all zero (an empty one).
    fn from_freqs(freq: [u16; ALPHABET]) -> Option<Self> {
        let sum: u32 = freq.iter().map(|&f| f as u32).sum();
        if sum != RANS_M && sum != 0 {
            return None;
        }
        Some(Self {
            freq,
            cum: prefix_sum(&freq),
        })
    }
}

/// Exclusive prefix sum of a 4-entry frequency array.
#[inline]
fn prefix_sum(freq: &[u16; ALPHABET]) -> [u16; ALPHABET] {
    let mut cum = [0u16; ALPHABET];
    let mut acc = 0u16;
    for i in 0..ALPHABET {
        cum[i] = acc;
        acc += freq[i];
    }
    cum
}

/// Resolve the slot value `slot ∈ [0, RANS_M)` to its owning symbol.
///
/// Exactly one symbol with non-zero frequency owns each slot, because
/// the `cum`/`freq` pairs partition `[0, RANS_M)`.
#[inline]
fn symbol_from_slot(slot: u16, table: &RansFreqTable) -> CellMode {
    let mut s = 0usize;
    while s < ALPHABET {
        let f = table.freq[s];
        if f != 0 && slot >= table.cum[s] && slot < table.cum[s] + f {
            break;
        }
        s += 1;
    }
    // The partition guarantees a hit for any slot < RANS_M; clamp to the
    // last symbol defensively so malformed input can't index out of range.
    let s = s.min(ALPHABET - 1);
    match s {
        0 => CellMode::Skip,
        1 => CellMode::Merge,
        2 => CellMode::Delta,
        _ => CellMode::Escape,
    }
}

// ════════════════════════════════════════════════════════════════════
// Core rANS encode / decode
// ════════════════════════════════════════════════════════════════════

/// rANS-encode the mode-tag stream against `table`, returning the
/// state-flushed payload bytes (no header).
///
/// Symbols are processed in reverse so the decoder recovers them in
/// forward order. The renormalization bytes are emitted LSB-first during
/// processing and the 4-byte final state is appended; the whole buffer
/// is reversed once at the end so a forward read reconstructs the state
/// and then consumes renorm bytes in the right order.
fn rans_encode(symbols: &[CellMode], table: &RansFreqTable) -> Vec<u8> {
    let mut x: u32 = RANS_BYTE_L;
    let mut out: Vec<u8> = Vec::new();
    for &s in symbols.iter().rev() {
        let si = s as usize;
        let f = table.freq[si] as u32;
        debug_assert!(f > 0, "encoding a symbol with zero frequency");
        let start = table.cum[si] as u32;
        let x_max = ((RANS_BYTE_L >> RANS_SCALE_BITS) << 8) * f;
        while x >= x_max {
            out.push((x & 0xFF) as u8);
            x >>= 8;
        }
        x = ((x / f) << RANS_SCALE_BITS) + (x % f) + start;
    }
    // Flush the 4-byte state MSB-first; after the final `reverse()` these
    // become the first 4 bytes read, in little-endian order for the
    // decoder's init.
    out.push((x >> 24) as u8);
    out.push((x >> 16) as u8);
    out.push((x >> 8) as u8);
    out.push((x & 0xFF) as u8);
    out.reverse();
    out
}

/// rANS-decode `n` mode tags from `payload` against `table`.
///
/// `payload` is the byte slice produced by [`rans_encode`]. Reading is
/// forward: the first 4 bytes initialize the state, the rest feed
/// renormalization. Defensive on truncated input (feeds zero bytes past
/// the end rather than panicking); valid round-trip input never reads
/// past the produced bytes.
fn rans_decode(payload: &[u8], table: &RansFreqTable, n: usize) -> Vec<CellMode> {
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let mut x =
        (payload[0] as u32) | ((payload[1] as u32) << 8) | ((payload[2] as u32) << 16) | ((payload[3] as u32) << 24);
    let mut pos = 4usize;
    for _ in 0..n {
        let slot = (x & (RANS_M - 1)) as u16;
        let s = symbol_from_slot(slot, table);
        out.push(s);
        let si = s as usize;
        let f = table.freq[si] as u32;
        let start = table.cum[si] as u32;
        x = f * (x >> RANS_SCALE_BITS) + slot as u32 - start;
        while x < RANS_BYTE_L {
            let b = payload.get(pos).copied().unwrap_or(0);
            pos += 1;
            x = (x << 8) | b as u32;
        }
    }
    out
}

// ════════════════════════════════════════════════════════════════════
// Self-contained stream API
// ════════════════════════════════════════════════════════════════════

/// Entropy-code a mode-tag stream into a self-contained byte stream
/// (header + payload, see the module-level layout).
///
/// The returned bytes embed the symbol count and the normalized
/// frequency table, so [`decode_modes`] needs no side channel. An empty
/// input yields a 12-byte all-zero-table header plus the flushed initial
/// state.
///
/// ```
/// use ndarray::hpc::codec::ans::{encode_modes, decode_modes};
/// use ndarray::hpc::codec::CellMode;
/// let modes = [CellMode::Skip, CellMode::Skip, CellMode::Delta, CellMode::Skip];
/// let stream = encode_modes(&modes);
/// assert_eq!(decode_modes(&stream).unwrap(), modes);
/// ```
pub fn encode_modes(symbols: &[CellMode]) -> Vec<u8> {
    let table = RansFreqTable::from_symbols(symbols);
    let payload = rans_encode(symbols, &table);
    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&(symbols.len() as u32).to_le_bytes());
    for f in table.freq {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out.extend_from_slice(&payload);
    out
}

/// Decode a stream produced by [`encode_modes`] back into mode tags.
///
/// Returns `None` if the stream is shorter than the header, if the
/// embedded frequency table is malformed (doesn't sum to [`RANS_M`] for
/// a non-empty stream), or if a non-empty stream lacks the 4 payload
/// bytes needed to initialize the rANS state.
///
/// ```
/// use ndarray::hpc::codec::ans::{encode_modes, decode_modes};
/// use ndarray::hpc::codec::CellMode;
/// // Empty input round-trips to an empty Vec.
/// assert_eq!(decode_modes(&encode_modes(&[])).unwrap(), Vec::<CellMode>::new());
/// ```
pub fn decode_modes(stream: &[u8]) -> Option<Vec<CellMode>> {
    if stream.len() < HEADER_LEN {
        return None;
    }
    let n = u32::from_le_bytes([stream[0], stream[1], stream[2], stream[3]]) as usize;
    let mut freq = [0u16; ALPHABET];
    for (i, slot) in freq.iter_mut().enumerate() {
        let lo = stream[4 + 2 * i];
        let hi = stream[5 + 2 * i];
        *slot = u16::from_le_bytes([lo, hi]);
    }
    let table = RansFreqTable::from_freqs(freq)?;
    let payload = &stream[HEADER_LEN..];
    if n > 0 && payload.len() < 4 {
        return None;
    }
    Some(rans_decode(payload, &table, n))
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [CellMode; 4] = [CellMode::Skip, CellMode::Merge, CellMode::Delta, CellMode::Escape];

    /// Deterministic xorshift so tests need no rand dependency.
    fn xorshift(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *state = x;
        x
    }

    fn mode_of(v: u64) -> CellMode {
        ALL[(v % 4) as usize]
    }

    #[test]
    fn freq_table_sums_to_m_for_nonempty() {
        for counts in [[1, 0, 0, 0], [3, 0, 1, 0], [10, 5, 3, 1], [1, 1, 1, 1], [1000, 1, 1, 1]] {
            let t = RansFreqTable::from_histogram(counts);
            let sum: u32 = t.freq.iter().map(|&f| f as u32).sum();
            assert_eq!(sum, RANS_M, "counts={counts:?}");
        }
    }

    #[test]
    fn freq_table_present_symbols_get_nonzero_freq() {
        let t = RansFreqTable::from_histogram([1_000_000, 1, 1, 1]);
        for s in 0..4 {
            assert!(t.freq[s] >= 1, "present symbol {s} must have freq >= 1");
        }
    }

    #[test]
    fn freq_table_absent_symbols_stay_zero() {
        let t = RansFreqTable::from_histogram([5, 0, 5, 0]);
        assert_eq!(t.freq[CellMode::Merge as usize], 0);
        assert_eq!(t.freq[CellMode::Escape as usize], 0);
    }

    #[test]
    fn empty_stream_roundtrips() {
        let stream = encode_modes(&[]);
        assert_eq!(decode_modes(&stream).unwrap(), Vec::<CellMode>::new());
    }

    #[test]
    fn single_symbol_roundtrips_all_modes() {
        for &m in &ALL {
            let stream = encode_modes(&[m]);
            assert_eq!(decode_modes(&stream).unwrap(), vec![m], "mode={m:?}");
        }
    }

    #[test]
    fn all_same_symbol_roundtrips() {
        for &m in &ALL {
            let symbols = vec![m; 257];
            let stream = encode_modes(&symbols);
            assert_eq!(decode_modes(&stream).unwrap(), symbols, "mode={m:?}");
        }
    }

    #[test]
    fn mixed_stream_roundtrips() {
        let symbols = [
            CellMode::Skip,
            CellMode::Delta,
            CellMode::Skip,
            CellMode::Merge,
            CellMode::Escape,
            CellMode::Skip,
            CellMode::Skip,
            CellMode::Delta,
        ];
        let stream = encode_modes(&symbols);
        assert_eq!(decode_modes(&stream).unwrap(), symbols);
    }

    #[test]
    fn large_random_streams_roundtrip() {
        let mut state = 0x1234_5678_9abc_def0u64;
        for &len in &[1usize, 2, 16, 100, 1000, 4096] {
            let symbols: Vec<CellMode> = (0..len).map(|_| mode_of(xorshift(&mut state))).collect();
            let stream = encode_modes(&symbols);
            let decoded = decode_modes(&stream).expect("decode");
            assert_eq!(decoded, symbols, "len={len}");
        }
    }

    #[test]
    fn skewed_skip_dominant_stream_roundtrips_and_compresses() {
        // 70% Skip, 25% Merge, 4.5% Delta, 0.5% Escape — the design's
        // coherent-state target distribution.
        let mut state = 0xdead_beef_cafe_babeu64;
        let symbols: Vec<CellMode> = (0..4000)
            .map(|_| {
                let r = xorshift(&mut state) % 1000;
                if r < 700 {
                    CellMode::Skip
                } else if r < 950 {
                    CellMode::Merge
                } else if r < 995 {
                    CellMode::Delta
                } else {
                    CellMode::Escape
                }
            })
            .collect();
        let stream = encode_modes(&symbols);
        assert_eq!(decode_modes(&stream).unwrap(), symbols);
        // Skewed distribution → well under 2 bits/symbol (vs 8 dense).
        let payload_bits = (stream.len() - HEADER_LEN) * 8;
        assert!(
            payload_bits < symbols.len() * 2,
            "expected < 2 bits/symbol on skewed input, got {} bits for {} symbols",
            payload_bits,
            symbols.len()
        );
    }

    #[test]
    fn decode_rejects_short_header() {
        assert!(decode_modes(&[0u8; 11]).is_none());
    }

    #[test]
    fn decode_rejects_malformed_freq_table() {
        // n = 1 but freqs sum to something != RANS_M and != 0.
        let mut stream = vec![0u8; HEADER_LEN + 4];
        stream[0] = 1; // n = 1
        stream[4] = 1; // freq[0] = 1 → sum 1, neither 0 nor RANS_M
        assert!(decode_modes(&stream).is_none());
    }

    #[test]
    fn decode_rejects_missing_payload_state() {
        // n = 1, valid single-symbol table, but no 4 payload bytes.
        let table = RansFreqTable::from_histogram([1, 0, 0, 0]);
        let mut stream = Vec::new();
        stream.extend_from_slice(&1u32.to_le_bytes());
        for f in table.freq {
            stream.extend_from_slice(&f.to_le_bytes());
        }
        stream.extend_from_slice(&[0u8; 2]); // only 2 payload bytes
        assert!(decode_modes(&stream).is_none());
    }

    #[test]
    fn header_layout_is_count_then_freqs() {
        let symbols = [CellMode::Skip, CellMode::Skip, CellMode::Delta];
        let stream = encode_modes(&symbols);
        let n = u32::from_le_bytes([stream[0], stream[1], stream[2], stream[3]]);
        assert_eq!(n, 3);
        let f_skip = u16::from_le_bytes([stream[4], stream[5]]);
        let f_delta = u16::from_le_bytes([stream[8], stream[9]]);
        assert!(f_skip > f_delta);
    }
}
