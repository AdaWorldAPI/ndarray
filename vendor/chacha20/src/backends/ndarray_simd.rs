//! ChaCha backend riding `ndarray::simd::U32x16` — the AdaWorldAPI matryoshka.
//!
//! RustCrypto owns the cipher (state schedule, counter, XChaCha, the Rng and
//! Poly1305 framing above); this file expresses *only* the keystream double-round
//! over `ndarray::simd::U32x16`, which `ndarray`'s compile-time polyfill lowers to
//! the AVX-512 lane on this build. There are **no raw intrinsics and no `unsafe`
//! here** — all of that lives once, audited, inside `ndarray::simd`.
//!
//! Layout: the *transpose* (vertical) form — 16 keystream blocks in parallel,
//! working vector `w` holds word `w` of every block across its 16 lanes, and the
//! counter word (12) carries lane-index `0..=15`. Every quarter-round is then a
//! pure `U32x16` `+` / `^` / `rotate_left` with **no cross-lane shuffle** (unlike
//! RustCrypto's row-interleaved AVX2 backend, whose `_mm256_shuffle_*` has no
//! `U32x16` equivalent). Bit-identical to the `soft` backend by construction; the
//! crate's RFC 8439 `tests/mod.rs` runs the published vectors through it.

use crate::{Block, ChaChaCore, Unsigned, STATE_WORDS};
use cipher::{
    consts::{U16, U64},
    BlockSizeUser, ParBlocks, ParBlocksSizeUser, StreamBackend,
};
use ndarray::simd::U32x16;

pub(crate) struct Backend<'a, R: Unsigned>(pub(crate) &'a mut ChaChaCore<R>);

impl<R: Unsigned> BlockSizeUser for Backend<'_, R> {
    type BlockSize = U64;
}

impl<R: Unsigned> ParBlocksSizeUser for Backend<'_, R> {
    type ParBlocksSize = U16;
}

impl<R: Unsigned> StreamBackend for Backend<'_, R> {
    #[inline(always)]
    fn gen_ks_block(&mut self, block: &mut Block) {
        // Single block: reuse the 16-wide core, take lane 0 (counter == state[12]),
        // advance the counter by one — exactly the `soft` backend's contract.
        let ks = ks16::<R>(&self.0.state);
        block.copy_from_slice(&ks[0]);
        self.0.state[12] = self.0.state[12].wrapping_add(1);
    }

    #[inline(always)]
    fn gen_par_ks_blocks(&mut self, blocks: &mut ParBlocks<Self>) {
        let ks = ks16::<R>(&self.0.state);
        for (dst, src) in blocks.iter_mut().zip(ks.iter()) {
            dst.copy_from_slice(src);
        }
        self.0.state[12] = self.0.state[12].wrapping_add(16);
    }
}

/// 16 consecutive ChaCha keystream blocks — counters `state[12] ..= state[12]+15`
/// (per-lane `wrapping_add`) — in the transpose layout over `U32x16`. `R::USIZE`
/// double-rounds (10 for ChaCha20, 6 for ChaCha12, 4 for ChaCha8).
#[inline(always)]
fn ks16<R: Unsigned>(state: &[u32; STATE_WORDS]) -> [[u8; 64]; 16] {
    // One vertical quarter-round across all 16 blocks — the RFC 8439 §2.1 word
    // indices, applied to the 16-lane vectors. Pure ARX (`+` / `^` / rotate).
    #[inline(always)]
    fn qr(v: &mut [U32x16; 16], a: usize, b: usize, c: usize, d: usize) {
        v[a] = v[a] + v[b];
        v[d] = (v[d] ^ v[a]).rotate_left(16);
        v[c] = v[c] + v[d];
        v[b] = (v[b] ^ v[c]).rotate_left(12);
        v[a] = v[a] + v[b];
        v[d] = (v[d] ^ v[a]).rotate_left(8);
        v[c] = v[c] + v[d];
        v[b] = (v[b] ^ v[c]).rotate_left(7);
    }

    // Initial working vectors: every word broadcast across the 16 lanes EXCEPT the
    // counter (word 12), which gets lane-index `0..=15` added (u32 wrapping) — the
    // 16 consecutive block counters.
    let lane_index = U32x16::from_array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    let mut orig = [U32x16::splat(0); STATE_WORDS];
    for (w, o) in orig.iter_mut().enumerate() {
        *o = U32x16::splat(state[w]);
    }
    orig[12] = orig[12] + lane_index;

    let mut v = orig;
    for _ in 0..R::USIZE {
        qr(&mut v, 0, 4, 8, 12);
        qr(&mut v, 1, 5, 9, 13);
        qr(&mut v, 2, 6, 10, 14);
        qr(&mut v, 3, 7, 11, 15);
        qr(&mut v, 0, 5, 10, 15);
        qr(&mut v, 1, 6, 11, 12);
        qr(&mut v, 2, 7, 8, 13);
        qr(&mut v, 3, 4, 9, 14);
    }

    // Add the original state back (RFC 8439 §2.3.1), then de-vectorize:
    // `words[w][l]` = final word `w` of block `l`.
    let mut words = [[0u32; 16]; STATE_WORDS];
    for (w, dst) in words.iter_mut().enumerate() {
        *dst = (v[w] + orig[w]).to_array();
    }

    // Serialize each block little-endian: block `l`, word `w` → bytes `4w..4w+4`.
    let mut out = [[0u8; 64]; 16];
    for (l, block) in out.iter_mut().enumerate() {
        for w in 0..STATE_WORDS {
            block[w * 4..w * 4 + 4].copy_from_slice(&words[w][l].to_le_bytes());
        }
    }
    out
}
