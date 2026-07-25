//! # encryption — zero-knowledge envelope crypto, native + wasm32
//!
//! One small, agnostic crate carrying the forward crypto suite of the Ada
//! stack, compiled from the same source for **native servers** and
//! **wasm32 browsers** (any stock Chrome/Firefox/Safari — no flags, no
//! extensions):
//!
//! | Role | Primitive | Why |
//! |---|---|---|
//! | Password hash / KDF | **Argon2id** | memory-hard, GPU/ASIC-resistant |
//! | Recoverable-secret AEAD | **XChaCha20-Poly1305** | 192-bit random nonce → no nonce-management footguns; authenticated; constant-time in software |
//! | Hashing (merkle, fingerprints) | **SHA-384** | truncation-resistant SHA-2; no length-extension |
//!
//! The point of the wasm leg: a consumer application can run
//! [`envelope::seal`] **in the browser**, so secrets are encrypted
//! client-side before they ever reach a server. The server stores an
//! opaque [`envelope`] blob it cannot read — end-to-end zero knowledge —
//! and the *same* Rust functions open it again, in the browser or on the
//! server side of a self-hosted deployment.
//!
//! ## Iron rules of this crate
//!
//! - **Entropy comes from [`getrandom`] only** (on wasm32: the `js`
//!   feature → `crypto.getRandomValues`). Never a userspace PRNG.
//! - **No serde, no runtime deserialization framework.** The sealed
//!   envelope is a fixed, versioned little-endian byte layout parsed by
//!   hand ([`envelope`]) — the whole layout is the documentation.
//! - **No secrets in errors.** Error types are field-free enums.
//! - **Derived keys zeroize on drop.**
//! - `#![forbid(unsafe_code)]` — this crate is pure safe Rust; the SIMD
//!   fast paths live inside the RustCrypto crates it consumes.
//!
//! ## What this crate deliberately does NOT do
//!
//! - **TLS / transport.** Browsers speak TLS with ECDSA P-256/384 certs;
//!   that stays in the web server / reverse proxy layer.
//! - **AES.** Browsers ship hardware-accelerated AES-GCM via WebCrypto;
//!   if a consumer wants AES it should call WebCrypto, not a wasm
//!   reimplementation. This crate exists for the primitives WebCrypto
//!   does *not* offer (Argon2, XChaCha20-Poly1305, Ed25519).
//! - **SIMD.** ndarray's one and only SIMD entry point is the existing
//!   `simd.rs` polyfill (compile-time dispatch → `simd_avx512.rs` /
//!   `simd_avx2.rs` / `simd_neon.rs` / `simd_wasm.rs`). This crate adds
//!   **no** SIMD code and **no** parallel dispatch — the vectorized fast
//!   paths live inside the audited RustCrypto crates it consumes. If a
//!   *batch* hash primitive (e.g. multi-buffer SHA-384 for merkle
//!   throughput) is ever measured to be hot, it belongs in the root
//!   crate behind the existing `simd.rs` dispatch under the W1a consumer
//!   contract — never as a second dispatch here.
//!
//! ## wasm build
//!
//! ```text
//! cargo check -p encryption --target wasm32-unknown-unknown
//! cargo build -p encryption --target wasm32-unknown-unknown \
//!     --features wasm-bindings --release
//! ```
//!
//! Note: wasm gives weaker constant-time *guarantees* than native code
//! (JIT tiers may re-time code). The chosen primitives are the most
//! timing-robust available (ChaCha/Poly1305 and Ed25519 avoid
//! data-dependent table lookups), which is exactly why this suite — and
//! not AES-CBC lookalikes — is the browser-side choice.

#![forbid(unsafe_code)]

pub mod aead;
pub mod channel;
pub mod envelope;
pub mod hash;
pub mod hkdf_sha384;
pub mod kdf;

#[cfg(feature = "wasm-bindings")]
pub mod wasm;

pub use envelope::{open, seal, EnvelopeError, KdfParams};

/// Fill `buf` from the platform CSPRNG (`getrandom`; on wasm32 this is
/// `crypto.getRandomValues`). The single entropy chokepoint of the crate.
pub(crate) fn fill_random(buf: &mut [u8]) -> Result<(), RngError> {
    getrandom::getrandom(buf).map_err(|_| RngError)
}

/// The platform CSPRNG was unavailable. Carries no detail on purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RngError;

impl core::fmt::Display for RngError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("platform CSPRNG unavailable")
    }
}

impl std::error::Error for RngError {}
