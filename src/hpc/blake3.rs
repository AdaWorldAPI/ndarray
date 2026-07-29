//! Pure-Rust, dependency-free BLAKE3 transcode.
//!
//! This module is a portable-only (no SIMD, no `unsafe`) transcription of
//! BLAKE3, based on the algorithm as published in the reference
//! implementation shipped by the upstream `BLAKE3-team/BLAKE3` project
//! (`reference_impl/reference_impl.rs`, referenced in Section 5.1 of the
//! BLAKE3 spec). That reference implementation is the serial, non-SIMD
//! tree-building algorithm — exactly the "single degree" code path that
//! `portable.rs` / `lib.rs`'s SIMD-parallel tree-walk (`compress_subtree_wide`,
//! `hash_many`, etc.) reduce to when `simd_degree() == 1`. It produces
//! byte-identical output to every other conformant BLAKE3 implementation;
//! only the parallel batching is left out, which is irrelevant here since we
//! have no SIMD/threading in scope (see module docs at the bottom of this
//! file for what was intentionally not transcribed).
//!
//! No `unsafe`, no `core::arch`, no external crates. `no_std`-friendly aside
//! from the tests, which use `std` (they run only under `cfg(test)`).

use core::cmp::min;
use core::fmt;

// ---------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------

const OUT_LEN: usize = 32;
const KEY_LEN: usize = 32;
const BLOCK_LEN: usize = 64;
const CHUNK_LEN: usize = 1024;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const PARENT: u32 = 1 << 2;
const ROOT: u32 = 1 << 3;
const KEYED_HASH: u32 = 1 << 4;
const DERIVE_KEY_CONTEXT: u32 = 1 << 5;
const DERIVE_KEY_MATERIAL: u32 = 1 << 6;

const IV: [u32; 8] = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19];

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

// ---------------------------------------------------------------------
// Compression function core (transcribed from portable.rs's `g`/`round`,
// specialized to the reference implementation's single-permutation-array
// style rather than portable.rs's per-round MSG_SCHEDULE table — the two
// are algebraically identical, just indexed differently.)
// ---------------------------------------------------------------------

#[inline(always)]
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

#[inline(always)]
fn round(state: &mut [u32; 16], m: &[u32; 16]) {
    // Mix the columns.
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    // Mix the diagonals.
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

#[inline(always)]
fn permute(m: &mut [u32; 16]) {
    let mut permuted = [0u32; 16];
    for i in 0..16 {
        permuted[i] = m[MSG_PERMUTATION[i]];
    }
    *m = permuted;
}

/// The core compression function. Returns the full 16-word state (already
/// XOR-finalized): the first 8 words are the chaining value output, and (when
/// `ROOT` is set) all 16 words are used as extendable-output bytes.
fn compress(chaining_value: &[u32; 8], block_words: &[u32; 16], counter: u64, block_len: u32, flags: u32) -> [u32; 16] {
    let counter_low = counter as u32;
    let counter_high = (counter >> 32) as u32;
    #[rustfmt::skip]
    let mut state = [
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        IV[0],             IV[1],             IV[2],             IV[3],
        counter_low,       counter_high,      block_len,         flags,
    ];
    let mut block = *block_words;

    round(&mut state, &block); // round 1
    permute(&mut block);
    round(&mut state, &block); // round 2
    permute(&mut block);
    round(&mut state, &block); // round 3
    permute(&mut block);
    round(&mut state, &block); // round 4
    permute(&mut block);
    round(&mut state, &block); // round 5
    permute(&mut block);
    round(&mut state, &block); // round 6
    permute(&mut block);
    round(&mut state, &block); // round 7

    for i in 0..8 {
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }
    state
}

#[inline]
fn first_8_words(compression_output: [u32; 16]) -> [u32; 8] {
    let mut out = [0u32; 8];
    out.copy_from_slice(&compression_output[0..8]);
    out
}

fn words_from_little_endian_bytes(bytes: &[u8], words: &mut [u32]) {
    debug_assert_eq!(bytes.len(), 4 * words.len());
    for (four_bytes, word) in bytes.chunks_exact(4).zip(words.iter_mut()) {
        let mut arr = [0u8; 4];
        arr.copy_from_slice(four_bytes);
        *word = u32::from_le_bytes(arr);
    }
}

// ---------------------------------------------------------------------
// Output (chunk / parent finalization; also the extendable-output source)
// ---------------------------------------------------------------------

/// Each chunk or parent node can produce either an 8-word chaining value or,
/// by setting the `ROOT` flag, any number of final output bytes. `Output`
/// captures the state just prior to choosing between those two possibilities.
#[derive(Clone)]
struct Output {
    input_chaining_value: [u32; 8],
    block_words: [u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
}

impl Output {
    fn chaining_value(&self) -> [u32; 8] {
        first_8_words(compress(&self.input_chaining_value, &self.block_words, self.counter, self.block_len, self.flags))
    }

    fn root_output_bytes(&self, out_slice: &mut [u8]) {
        let mut output_block_counter = 0u64;
        for out_block in out_slice.chunks_mut(2 * OUT_LEN) {
            let words = compress(
                &self.input_chaining_value,
                &self.block_words,
                output_block_counter,
                self.block_len,
                self.flags | ROOT,
            );
            // The output length might not be a multiple of 4.
            for (word, out_word) in words.iter().zip(out_block.chunks_mut(4)) {
                let bytes = word.to_le_bytes();
                out_word.copy_from_slice(&bytes[..out_word.len()]);
            }
            output_block_counter += 1;
        }
    }
}

// ---------------------------------------------------------------------
// ChunkState
// ---------------------------------------------------------------------

#[derive(Clone)]
struct ChunkState {
    chaining_value: [u32; 8],
    chunk_counter: u64,
    block: [u8; BLOCK_LEN],
    block_len: u8,
    blocks_compressed: u8,
    flags: u32,
}

impl ChunkState {
    fn new(key_words: [u32; 8], chunk_counter: u64, flags: u32) -> Self {
        Self {
            chaining_value: key_words,
            chunk_counter,
            block: [0u8; BLOCK_LEN],
            block_len: 0,
            blocks_compressed: 0,
            flags,
        }
    }

    fn len(&self) -> usize {
        BLOCK_LEN * self.blocks_compressed as usize + self.block_len as usize
    }

    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 {
            CHUNK_START
        } else {
            0
        }
    }

    fn update(&mut self, mut input: &[u8]) {
        // This method does NOT split across chunks — `Hasher::update` is what
        // caps each call, via `want = CHUNK_LEN - self.chunk_state.len()`.
        // That cap lives ~250 lines away and the fast path below silently
        // depends on it: without it, `full` could run past the chunk boundary
        // and compress more than 16 blocks under one chunk counter, producing
        // a wrong-but-plausible hash.
        //
        // The official vectors would NOT catch that, because they only reach
        // this method through `Hasher::update`. So state the invariant where
        // it is relied upon instead of trusting a distant caller.
        debug_assert!(
            self.len() + input.len() <= CHUNK_LEN,
            "ChunkState::update overran a chunk: {} + {} > {}",
            self.len(),
            input.len(),
            CHUNK_LEN,
        );

        // Fast path — compress whole blocks straight out of `input`.
        //
        // The staging path below copies every byte twice: once into
        // `self.block`, then again into `block_words`. For a full block that
        // first copy is pure overhead. `array_chunks::<64>` walks the input as
        // `&[u8; 64]` with no copy at all (it is `as_chunks`, a pointer cast),
        // so a full block goes input -> words directly.
        //
        // The guard is `input.len() > BLOCK_LEN`, strictly greater: BLAKE3
        // must not compress a chunk's FINAL block until it knows whether more
        // input follows, because that block carries CHUNK_END. Holding back
        // the last <= 64 bytes preserves that, and the official vectors are
        // the gate — the 0/1/64/65/1024/1025-byte cases all cross this
        // boundary.
        if self.block_len == 0 && input.len() > BLOCK_LEN {
            let full = (input.len() - 1) / BLOCK_LEN; // never the last block
            let (head, tail) = input.split_at(full * BLOCK_LEN);
            for block in crate::simd_ops::array_chunks::<u8, BLOCK_LEN>(head) {
                let mut block_words = [0u32; 16];
                words_from_little_endian_bytes(block, &mut block_words);
                self.chaining_value = first_8_words(compress(
                    &self.chaining_value,
                    &block_words,
                    self.chunk_counter,
                    BLOCK_LEN as u32,
                    self.flags | self.start_flag(),
                ));
                self.blocks_compressed += 1;
            }
            input = tail;
        }

        while !input.is_empty() {
            // If the block buffer is full, compress it and clear it. More
            // input is coming, so this compression is not CHUNK_END.
            if self.block_len as usize == BLOCK_LEN {
                let mut block_words = [0u32; 16];
                words_from_little_endian_bytes(&self.block, &mut block_words);
                self.chaining_value = first_8_words(compress(
                    &self.chaining_value,
                    &block_words,
                    self.chunk_counter,
                    BLOCK_LEN as u32,
                    self.flags | self.start_flag(),
                ));
                self.blocks_compressed += 1;
                self.block = [0u8; BLOCK_LEN];
                self.block_len = 0;
            }

            // Copy input bytes into the block buffer.
            let want = BLOCK_LEN - self.block_len as usize;
            let take = min(want, input.len());
            self.block[self.block_len as usize..][..take].copy_from_slice(&input[..take]);
            self.block_len += take as u8;
            input = &input[take..];
        }
    }

    fn output(&self) -> Output {
        let mut block_words = [0u32; 16];
        words_from_little_endian_bytes(&self.block, &mut block_words);
        Output {
            input_chaining_value: self.chaining_value,
            block_words,
            counter: self.chunk_counter,
            block_len: self.block_len as u32,
            flags: self.flags | self.start_flag() | CHUNK_END,
        }
    }
}

impl fmt::Debug for ChunkState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChunkState")
            .field("len", &self.len())
            .field("chunk_counter", &self.chunk_counter)
            .field("flags", &self.flags)
            .finish()
    }
}

fn parent_output(left_child_cv: [u32; 8], right_child_cv: [u32; 8], key_words: [u32; 8], flags: u32) -> Output {
    let mut block_words = [0u32; 16];
    block_words[..8].copy_from_slice(&left_child_cv);
    block_words[8..].copy_from_slice(&right_child_cv);
    Output {
        input_chaining_value: key_words,
        block_words,
        counter: 0,                  // Always 0 for parent nodes.
        block_len: BLOCK_LEN as u32, // Always BLOCK_LEN (64) for parent nodes.
        flags: PARENT | flags,
    }
}

fn parent_cv(left_child_cv: [u32; 8], right_child_cv: [u32; 8], key_words: [u32; 8], flags: u32) -> [u32; 8] {
    parent_output(left_child_cv, right_child_cv, key_words, flags).chaining_value()
}

// ---------------------------------------------------------------------
// Public API: Hash
// ---------------------------------------------------------------------

/// A 32-byte BLAKE3 output.
#[derive(Clone, Copy, Eq)]
pub struct Hash([u8; OUT_LEN]);

impl Hash {
    /// The raw bytes of this `Hash`.
    #[inline]
    pub const fn as_bytes(&self) -> &[u8; OUT_LEN] {
        &self.0
    }
}

impl From<[u8; OUT_LEN]> for Hash {
    #[inline]
    fn from(bytes: [u8; OUT_LEN]) -> Self {
        Self(bytes)
    }
}

impl From<Hash> for [u8; OUT_LEN] {
    #[inline]
    fn from(hash: Hash) -> Self {
        hash.0
    }
}

impl AsRef<[u8]> for Hash {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl PartialEq for Hash {
    /// Constant-time comparison.
    ///
    /// A BLAKE3 output is used as a MAC in keyed mode, so a short-circuiting
    /// `==` leaks match-prefix length through timing. Upstream reaches for
    /// the `constant_time_eq` crate; this crate takes no new dependencies, so
    /// the fold is written here: accumulate the XOR of every byte pair and
    /// test once at the end, with `core::hint::black_box` on the accumulator
    /// to stop the optimizer reintroducing an early exit.
    ///
    /// `black_box` is a **best-effort optimizer barrier, not a guarantee** —
    /// its documentation is explicit that it provides no formal contract. The
    /// data-independent loop above is the real property; the barrier only
    /// discourages LLVM from undoing it. A caller needing an audited
    /// guarantee should use a dedicated constant-time crate.
    ///
    /// No call site in this crate currently compares two `Hash` values —
    /// `seal.rs` compares the truncated `MerkleRoot` instead — so this is a
    /// guard for future consumers rather than a fix for a live leak.
    #[inline]
    fn eq(&self, other: &Hash) -> bool {
        let mut diff = 0u8;
        for i in 0..OUT_LEN {
            diff |= self.0[i] ^ other.0[i];
        }
        // Optimizer barrier (best-effort): discourages LLVM from proving an
        // early return equivalent and reintroducing the short circuit.
        core::hint::black_box(diff) == 0
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Hash(")?;
        for byte in self.0.iter() {
            write!(f, "{:02x}", byte)?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------
// Public API: Hasher
// ---------------------------------------------------------------------

/// An incremental BLAKE3 hasher.
#[derive(Clone)]
pub struct Hasher {
    chunk_state: ChunkState,
    key_words: [u32; 8],
    cv_stack: [[u32; 8]; 54], // 2^54 * CHUNK_LEN = 2^64
    cv_stack_len: u8,
    flags: u32,
}

impl fmt::Debug for Hasher {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hasher")
            .field("chunk_state", &self.chunk_state)
            .field("cv_stack_len", &self.cv_stack_len)
            .field("flags", &self.flags)
            .finish()
    }
}

impl Default for Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Hasher {
    fn new_internal(key_words: [u32; 8], flags: u32) -> Self {
        Self {
            chunk_state: ChunkState::new(key_words, 0, flags),
            key_words,
            cv_stack: [[0u32; 8]; 54],
            cv_stack_len: 0,
            flags,
        }
    }

    /// Construct a new `Hasher` for the regular hash function.
    pub fn new() -> Self {
        Self::new_internal(IV, 0)
    }

    /// Construct a new `Hasher` for the keyed hash function.
    pub fn new_keyed(key: &[u8; KEY_LEN]) -> Self {
        let mut key_words = [0u32; 8];
        words_from_little_endian_bytes(key, &mut key_words);
        Self::new_internal(key_words, KEYED_HASH)
    }

    /// Construct a new `Hasher` for the key-derivation function. The context
    /// string should be hardcoded, globally unique, and application-specific.
    pub fn new_derive_key(context: &str) -> Self {
        let mut context_hasher = Self::new_internal(IV, DERIVE_KEY_CONTEXT);
        context_hasher.update(context.as_bytes());
        let context_key = context_hasher.finalize();
        let mut context_key_words = [0u32; 8];
        words_from_little_endian_bytes(context_key.as_bytes(), &mut context_key_words);
        Self::new_internal(context_key_words, DERIVE_KEY_MATERIAL)
    }

    fn push_stack(&mut self, cv: [u32; 8]) {
        self.cv_stack[self.cv_stack_len as usize] = cv;
        self.cv_stack_len += 1;
    }

    fn pop_stack(&mut self) -> [u32; 8] {
        self.cv_stack_len -= 1;
        self.cv_stack[self.cv_stack_len as usize]
    }

    // Section 5.1.2 of the BLAKE3 spec explains this algorithm in detail.
    fn add_chunk_chaining_value(&mut self, mut new_cv: [u32; 8], mut total_chunks: u64) {
        while total_chunks & 1 == 0 {
            new_cv = parent_cv(self.pop_stack(), new_cv, self.key_words, self.flags);
            total_chunks >>= 1;
        }
        self.push_stack(new_cv);
    }

    /// Add input to the hash state. May be called any number of times.
    pub fn update(&mut self, mut input: &[u8]) -> &mut Self {
        while !input.is_empty() {
            if self.chunk_state.len() == CHUNK_LEN {
                let chunk_cv = self.chunk_state.output().chaining_value();
                let total_chunks = self.chunk_state.chunk_counter + 1;
                self.add_chunk_chaining_value(chunk_cv, total_chunks);
                self.chunk_state = ChunkState::new(self.key_words, total_chunks, self.flags);
            }

            let want = CHUNK_LEN - self.chunk_state.len();
            let take = min(want, input.len());
            self.chunk_state.update(&input[..take]);
            input = &input[take..];
        }
        self
    }

    fn final_output(&self) -> Output {
        let mut output = self.chunk_state.output();
        let mut parent_nodes_remaining = self.cv_stack_len as usize;
        while parent_nodes_remaining > 0 {
            parent_nodes_remaining -= 1;
            output = parent_output(
                self.cv_stack[parent_nodes_remaining],
                output.chaining_value(),
                self.key_words,
                self.flags,
            );
        }
        output
    }

    /// Finalize the hash and return the default 32-byte output.
    pub fn finalize(&self) -> Hash {
        let mut out = [0u8; OUT_LEN];
        self.final_output().root_output_bytes(&mut out);
        Hash(out)
    }

    /// Finalize the hash as an extendable-output stream.
    pub fn finalize_xof(&self) -> OutputReader {
        OutputReader {
            inner: self.final_output(),
            position: 0,
        }
    }
}

// ---------------------------------------------------------------------
// Public API: OutputReader (extendable output)
// ---------------------------------------------------------------------

/// A reader for extendable BLAKE3 output, produced by [`Hasher::finalize_xof`].
#[derive(Clone)]
pub struct OutputReader {
    inner: Output,
    position: u64,
}

impl fmt::Debug for OutputReader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutputReader")
            .field("position", &self.position)
            .finish()
    }
}

impl OutputReader {
    /// Fill `buf` with the next `buf.len()` bytes of output. Successive calls
    /// continue from where the previous call left off.
    pub fn fill(&mut self, mut buf: &mut [u8]) {
        const BLOCK: u64 = 2 * OUT_LEN as u64; // 64
        while !buf.is_empty() {
            let block_counter = self.position / BLOCK;
            let block_offset = (self.position % BLOCK) as usize;

            let words = compress(
                &self.inner.input_chaining_value,
                &self.inner.block_words,
                block_counter,
                self.inner.block_len,
                self.inner.flags | ROOT,
            );
            let mut block_bytes = [0u8; BLOCK as usize];
            for (word, out_word) in words.iter().zip(block_bytes.chunks_mut(4)) {
                out_word.copy_from_slice(&word.to_le_bytes());
            }

            let available = BLOCK as usize - block_offset;
            let take = min(available, buf.len());
            buf[..take].copy_from_slice(&block_bytes[block_offset..block_offset + take]);
            self.position += take as u64;

            let (_, rest) = buf.split_at_mut(take);
            buf = rest;
        }
    }
}

// ---------------------------------------------------------------------
// Public API: free functions
// ---------------------------------------------------------------------

/// The default hash function.
pub fn hash(input: &[u8]) -> Hash {
    let mut hasher = Hasher::new();
    hasher.update(input);
    hasher.finalize()
}

/// The keyed hash function.
pub fn keyed_hash(key: &[u8; KEY_LEN], input: &[u8]) -> Hash {
    let mut hasher = Hasher::new_keyed(key);
    hasher.update(input);
    hasher.finalize()
}

/// The key-derivation function.
pub fn derive_key(context: &str, key_material: &[u8]) -> [u8; OUT_LEN] {
    let mut hasher = Hasher::new_derive_key(context);
    hasher.update(key_material);
    *hasher.finalize().as_bytes()
}

// =======================================================================
// NOT transcribed (out of scope for this module — documented per task spec):
//
// - `hash_many` / `compress_chunks_parallel` / `compress_parents_parallel` /
//   `compress_subtree_wide` / `compress_subtree_to_parent_node`: these exist
//   purely to batch multiple chunks/parents through SIMD-width compression
//   calls (`platform.simd_degree()`), and to enable Rayon multithreading via
//   the `join::Join` trait. With `simd_degree() == 1` and no threading, they
//   reduce to exactly the serial one-chunk-then-merge-up-the-stack algorithm
//   this module implements directly (`Hasher::update` / `add_chunk_chaining_value`).
// - `update_rayon` and anything behind the `rayon` feature.
// - `zeroize` support (feature-gated in the original; secrets are not a
//   concern for ndarray's usage of this module).
// - The `constant_time_eq` CRATE — the dependency, NOT the behaviour. The
//   constant-time compare itself IS implemented: see `impl PartialEq for
//   Hash`, which XOR-folds all 32 bytes and tests once, so `Hash::eq` keeps
//   upstream's timing property without adding the dependency.
//
//   An earlier revision of this list claimed the opposite — "`PartialEq` here
//   is a normal short-circuiting byte-array compare" — left stale when the
//   fold landed. Documenting a timing property backwards is worse than
//   omitting it: someone auditing a MAC comparison would have believed this
//   file leaks match-prefix length when it does not.
// - Hex encoding/decoding (`to_hex`/`from_hex`) and `Display`/`FromStr`: not
//   in the required API list.
// =======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::string::String;
    use std::vec::Vec;

    const VECTORS_JSON: &str = include_str!("blake3_test_vectors.json");
    const KEY: &[u8; 32] = b"whats the Elvish word for friend";

    struct Case {
        input_len: usize,
        hash_hex: String,
        keyed_hash_hex: String,
        derive_key_hex: String,
    }

    /// The context string the official vectors' `derive_key` outputs were
    /// generated with (the `context_string` field of `test_vectors.json`).
    ///
    /// Deliberately NOT named `DERIVE_KEY_CONTEXT` — that is a `u32` domain
    /// flag at module scope, and reusing the name here would shadow it inside
    /// this module.
    const VECTORS_CONTEXT: &str = "BLAKE3 2019-12-27 16:29:52 test vectors context";

    /// Hand-rolled extraction of the fields we need from the official BLAKE3
    /// test_vectors.json, without pulling in serde. The file's `cases` array
    /// is a flat sequence of objects each with exactly the fields
    /// `input_len` (integer), `hash` (hex string), `keyed_hash` (hex string),
    /// `derive_key` (hex string, unused here). We scan for each field by name
    /// in order, which is sufficient given the known, fixed shape of this
    /// file (verified above by inspection).
    fn parse_cases(json: &str) -> Vec<Case> {
        let mut cases = Vec::new();
        let mut rest = json;
        loop {
            let Some(idx) = rest.find("\"input_len\":") else {
                break;
            };
            rest = &rest[idx + "\"input_len\":".len()..];
            let len_end = rest
                .find(|c: char| c == ',' || c == '\n' || c == '}')
                .expect("malformed input_len");
            let input_len: usize = rest[..len_end].trim().parse().expect("bad input_len");

            let hash_idx = rest.find("\"hash\":").expect("missing hash field") + "\"hash\":".len();
            rest = &rest[hash_idx..];
            let hash_hex = extract_quoted(rest);
            rest = &rest[hash_hex.len() + 2..];

            let kh_idx = rest
                .find("\"keyed_hash\":")
                .expect("missing keyed_hash field")
                + "\"keyed_hash\":".len();
            rest = &rest[kh_idx..];
            let keyed_hash_hex = extract_quoted(rest);
            rest = &rest[keyed_hash_hex.len() + 2..];

            let dk_idx = rest
                .find("\"derive_key\":")
                .expect("missing derive_key field")
                + "\"derive_key\":".len();
            rest = &rest[dk_idx..];
            let derive_key_hex = extract_quoted(rest);
            rest = &rest[derive_key_hex.len() + 2..];

            cases.push(Case {
                input_len,
                hash_hex,
                keyed_hash_hex,
                derive_key_hex,
            });
        }
        cases
    }

    /// Given a string starting with optional whitespace then a `"..."`
    /// quoted JSON string (no escapes present in this file's hex fields),
    /// return the content between the quotes.
    fn extract_quoted(s: &str) -> String {
        let s = s.trim_start();
        assert!(s.starts_with('"'), "expected quoted string");
        let s = &s[1..];
        let end = s.find('"').expect("unterminated string");
        s[..end].to_string()
    }

    fn hex_decode(s: &str) -> Vec<u8> {
        assert_eq!(s.len() % 2, 0);
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Reference input: byte i is (i % 251).
    fn reference_input(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn official_test_vectors_all_three_modes() {
        let cases = parse_cases(VECTORS_JSON);
        assert!(!cases.is_empty(), "no cases parsed");
        let mut checked = 0usize;
        for case in &cases {
            let input = reference_input(case.input_len);
            let expected_hash = hex_decode(&case.hash_hex);
            let expected_keyed = hex_decode(&case.keyed_hash_hex);

            // --- unkeyed hash, 32-byte default output ---
            let got = hash(&input);
            assert_eq!(got.as_bytes()[..], expected_hash[..32], "hash() mismatch at input_len={}", case.input_len);

            // --- unkeyed hash, extended output via finalize_xof ---
            let mut hasher = Hasher::new();
            hasher.update(&input);
            let mut xof = hasher.finalize_xof();
            let mut extended = std::vec![0u8; expected_hash.len()];
            xof.fill(&mut extended);
            assert_eq!(
                extended, expected_hash,
                "finalize_xof extended output mismatch at input_len={}",
                case.input_len
            );

            // --- keyed hash, 32-byte default output ---
            let got_keyed = keyed_hash(KEY, &input);
            assert_eq!(
                got_keyed.as_bytes()[..],
                expected_keyed[..32],
                "keyed_hash() mismatch at input_len={}",
                case.input_len
            );

            // --- keyed hash, extended output via finalize_xof ---
            let mut khasher = Hasher::new_keyed(KEY);
            khasher.update(&input);
            let mut kxof = khasher.finalize_xof();
            let mut kextended = std::vec![0u8; expected_keyed.len()];
            kxof.fill(&mut kextended);
            assert_eq!(
                kextended, expected_keyed,
                "keyed finalize_xof extended output mismatch at input_len={}",
                case.input_len
            );

            // --- derive_key, 32-byte default output ---
            //
            // The third public mode, and the one whose flags are easiest to
            // get wrong while the other two still pass: it is a TWO-pass
            // construction (DERIVE_KEY_CONTEXT over the context string,
            // whose output becomes the key words for a DERIVE_KEY_MATERIAL
            // pass over the material). A single-pass transcription, or one
            // that swapped the two flags, would be invisible to every
            // assertion above.
            let expected_dk = hex_decode(&case.derive_key_hex);
            let got_dk = derive_key(VECTORS_CONTEXT, &input);
            assert_eq!(got_dk[..], expected_dk[..32], "derive_key() mismatch at input_len={}", case.input_len);

            // --- derive_key, extended output via finalize_xof ---
            let mut dhasher = Hasher::new_derive_key(VECTORS_CONTEXT);
            dhasher.update(&input);
            let mut dxof = dhasher.finalize_xof();
            let mut dextended = std::vec![0u8; expected_dk.len()];
            dxof.fill(&mut dextended);
            assert_eq!(
                dextended, expected_dk,
                "derive_key finalize_xof extended output mismatch at input_len={}",
                case.input_len
            );

            checked += 1;
        }
        // Sanity: make sure we actually walked the whole official vector file.
        assert_eq!(checked, 35, "expected 35 official test vector cases");
    }

    #[test]
    fn streaming_matches_single_shot() {
        let input = reference_input(102_400);
        let whole = hash(&input);

        let mut hasher = Hasher::new();
        for chunk in input.chunks(37) {
            hasher.update(chunk);
        }
        let streamed = hasher.finalize();
        assert_eq!(whole, streamed);
    }

    #[test]
    fn empty_input() {
        let h = hash(&[]);
        // From the official test vectors, input_len == 0 case (the first
        // 32 bytes of its "hash" field).
        let cases = parse_cases(VECTORS_JSON);
        let case0 = cases.iter().find(|c| c.input_len == 0).unwrap();
        let expected = hex_decode(&case0.hash_hex);
        assert_eq!(h.as_bytes()[..], expected[..32]);
    }

    #[test]
    fn fill_in_small_increments_matches_one_shot() {
        let input = reference_input(5000);
        let mut hasher = Hasher::new();
        hasher.update(&input);

        let mut one_shot = [0u8; 300];
        hasher.finalize_xof().fill(&mut one_shot);

        let mut incremental = [0u8; 300];
        let mut xof = hasher.finalize_xof();
        for chunk in incremental.chunks_mut(7) {
            xof.fill(chunk);
        }
        assert_eq!(one_shot[..], incremental[..]);
    }
}
