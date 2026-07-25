//! ChaCha backend riding `ndarray::simd::U32x16` — the AdaWorldAPI matryoshka.
//!
//! RustCrypto owns the cipher (state schedule, counter, XChaCha, the Rng and
//! Poly1305 framing above); this file expresses *only* the keystream double-round
//! over `ndarray::simd::U32x16`, which `ndarray`'s compile-time polyfill lowers to
//! the AVX-512 lane on this build. There are **no raw intrinsics and no `unsafe`
//! here** — all of that lives once, audited, inside `ndarray::simd`. It is the
//! replacement for the `avx2`/`avx512` backends, not an addition beside them.
//!
//! Layout: the *transpose* (vertical) form — 16 keystream blocks in parallel,
//! working vector `w` holds word `w` of every block across its 16 lanes, and the
//! counter words carry lane-index `0..=15`. Every quarter-round is then a pure
//! `U32x16` `+` / `^` / `rotate_left` with **no cross-lane shuffle** (unlike
//! RustCrypto's row-interleaved AVX2 backend, whose `_mm256_shuffle_*` has no
//! `U32x16` equivalent). Bit-identical to the `soft` backend by construction; the
//! crate's RFC 8439 vectors run through it.
//!
//! ## Ported to 0.10, and the one thing that is not a rename
//!
//! The 0.9.1 version of this file added `lane_index` straight onto `orig[12]`.
//! In 0.10 the counter is variant-dependent: `size_of::<V::Counter>() == 8` means
//! a 64-bit counter spanning `state[12]` (low) and `state[13]` (high), while the
//! 32-bit variant keeps `state[13]` as a nonce word. Adding the lane index to the
//! low word alone is correct only for the 32-bit case — for the 64-bit one it
//! silently produces the wrong keystream on any 16-block span that carries across
//! the word boundary. That is not a compile error, so it is spelled out here and
//! the lane counters are computed as 64-bit values, split into low/high words.

use crate::{Rounds, STATE_WORDS, Variant};

use core::marker::PhantomData;

use crate::chacha::Block;
use cipher::{
    BlockSizeUser, ParBlocks, ParBlocksSizeUser, StreamCipherBackend, StreamCipherClosure,
    consts::{U16, U64},
};
use ndarray::simd::U32x16;

/// Keystream blocks produced per call — the `U32x16` lane count.
const PAR_BLOCKS: usize = 16;

/// Set up the 16 lane counters, run the closure, write the counter back.
///
/// Mirrors the shape of the `avx2`/`avx512` backends' `inner`, minus the
/// `unsafe`: the vector work is `ndarray::simd`'s, and it is safe by
/// construction.
#[inline]
#[cfg(feature = "cipher")]
pub(crate) fn inner<R, F, V>(state: &mut [u32; STATE_WORDS], f: F)
where
    R: Rounds,
    F: StreamCipherClosure<BlockSize = U64>,
    V: Variant,
{
    let mut backend = Backend::<R, V> {
        state: *state,
        _pd: PhantomData,
    };

    f.call(&mut backend);

    state[12] = backend.state[12];
    if size_of::<V::Counter>() == 8 {
        state[13] = backend.state[13];
    }
}

pub(crate) struct Backend<R: Rounds, V: Variant> {
    state: [u32; STATE_WORDS],
    _pd: PhantomData<(R, V)>,
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> BlockSizeUser for Backend<R, V> {
    type BlockSize = U64;
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> ParBlocksSizeUser for Backend<R, V> {
    type ParBlocksSize = U16;
}

#[cfg(feature = "cipher")]
impl<R: Rounds, V: Variant> StreamCipherBackend for Backend<R, V> {
    #[inline(always)]
    fn gen_ks_block(&mut self, block: &mut Block) {
        // Single block: reuse the 16-wide core, take lane 0, advance by one —
        // exactly the `soft` backend's contract.
        let ks = ks16::<R, V>(&self.state);
        block.copy_from_slice(&ks[0]);
        self.advance::<V>(1);
    }

    #[inline(always)]
    fn gen_par_ks_blocks(&mut self, blocks: &mut ParBlocks<Self>) {
        let ks = ks16::<R, V>(&self.state);
        for (dst, src) in blocks.iter_mut().zip(ks.iter()) {
            dst.copy_from_slice(src);
        }
        self.advance::<V>(PAR_BLOCKS as u64);
    }
}

impl<R: Rounds, V: Variant> Backend<R, V> {
    /// Advance the block counter by `n`, in whichever width the variant uses.
    #[inline(always)]
    fn advance<Var: Variant>(&mut self, n: u64) {
        if size_of::<Var::Counter>() == 8 {
            let ctr = ((u64::from(self.state[13]) << 32) | u64::from(self.state[12])).wrapping_add(n);
            self.state[12] = ctr as u32;
            self.state[13] = (ctr >> 32) as u32;
        } else {
            self.state[12] = self.state[12].wrapping_add(n as u32);
        }
    }
}

/// 16 consecutive ChaCha keystream blocks — counters `ctr ..= ctr+15` — in the
/// transpose layout over `U32x16`. `R::COUNT` double-rounds (10 for ChaCha20,
/// 6 for ChaCha12, 4 for ChaCha8).
#[inline(always)]
fn ks16<R: Rounds, V: Variant>(state: &[u32; STATE_WORDS]) -> [[u8; 64]; PAR_BLOCKS] {
    // One vertical quarter-round across all 16 blocks — the RFC 8439 §2.1 word
    // indices, applied to the 16-lane vectors. Pure ARX (`+` / `^` / rotate).
    #[inline(always)]
    fn qr(v: &mut [U32x16; STATE_WORDS], a: usize, b: usize, c: usize, d: usize) {
        v[a] = v[a] + v[b];
        v[d] = (v[d] ^ v[a]).rotate_left(16);
        v[c] = v[c] + v[d];
        v[b] = (v[b] ^ v[c]).rotate_left(12);
        v[a] = v[a] + v[b];
        v[d] = (v[d] ^ v[a]).rotate_left(8);
        v[c] = v[c] + v[d];
        v[b] = (v[b] ^ v[c]).rotate_left(7);
    }

    let mut orig = [U32x16::splat(0); STATE_WORDS];
    for (w, o) in orig.iter_mut().enumerate() {
        *o = U32x16::splat(state[w]);
    }

    // The lane counters. 64-bit variant: carry propagates into word 13, so both
    // words vary per lane. 32-bit variant: word 13 is nonce and stays broadcast.
    if size_of::<V::Counter>() == 8 {
        let base = (u64::from(state[13]) << 32) | u64::from(state[12]);
        let mut lo = [0u32; PAR_BLOCKS];
        let mut hi = [0u32; PAR_BLOCKS];
        for (lane, (l, h)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
            let c = base.wrapping_add(lane as u64);
            *l = c as u32;
            *h = (c >> 32) as u32;
        }
        orig[12] = U32x16::from_array(lo);
        orig[13] = U32x16::from_array(hi);
    } else {
        let mut lo = [0u32; PAR_BLOCKS];
        for (lane, l) in lo.iter_mut().enumerate() {
            *l = state[12].wrapping_add(lane as u32);
        }
        orig[12] = U32x16::from_array(lo);
    }

    let mut v = orig;
    for _ in 0..R::COUNT {
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
    let mut words = [[0u32; PAR_BLOCKS]; STATE_WORDS];
    for (w, dst) in words.iter_mut().enumerate() {
        *dst = (v[w] + orig[w]).to_array();
    }

    // Serialize each block little-endian: block `l`, word `w` → bytes `4w..4w+4`.
    let mut out = [[0u8; 64]; PAR_BLOCKS];
    for (l, block) in out.iter_mut().enumerate() {
        for w in 0..STATE_WORDS {
            block[w * 4..w * 4 + 4].copy_from_slice(&words[w][l].to_le_bytes());
        }
    }
    out
}
