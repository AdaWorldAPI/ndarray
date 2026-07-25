//! Browser bindings (`--features wasm-bindings`, target wasm32).
//!
//! Thin `wasm-bindgen` façade over the crate's Rust API so a stock
//! browser can call it from JavaScript after a `wasm-pack build`:
//!
//! ```js
//! import init, { seal_envelope, open_envelope } from "./pkg/encryption.js";
//! await init();
//! const blob = seal_envelope(pwUtf8, secretBytes, true /* interactive */);
//! // ship `blob` to the server — it is unreadable there
//! const secret = open_envelope(pwUtf8, blob);
//! ```
//!
//! Errors surface as thrown JS strings (the `Display` of the Rust
//! error) — deliberately message-only, never secret material.

use wasm_bindgen::prelude::*;

use crate::envelope::{self, KdfParams};
use crate::sign::{Keypair, PUBLIC_KEY_LEN, SEED_LEN, SIGNATURE_LEN};

fn params(interactive: bool) -> KdfParams {
    if interactive {
        KdfParams::INTERACTIVE
    } else {
        KdfParams::DEFAULT
    }
}

/// Seal `plaintext` under `password` client-side. With
/// `interactive = true` the browser-grade Argon2id cost is used.
#[wasm_bindgen]
pub fn seal_envelope(password: &[u8], plaintext: &[u8], interactive: bool) -> Result<Vec<u8>, JsError> {
    envelope::seal(password, plaintext, &params(interactive)).map_err(|e| JsError::new(&e.to_string()))
}

/// Open a sealed envelope. Throws on wrong password or tampering.
#[wasm_bindgen]
pub fn open_envelope(password: &[u8], blob: &[u8]) -> Result<Vec<u8>, JsError> {
    envelope::open(password, blob).map_err(|e| JsError::new(&e.to_string()))
}

/// Generate a fresh Ed25519 seed (32 bytes) from the browser CSPRNG.
/// The caller is responsible for storing it sealed (see
/// [`seal_envelope`]) — never in plaintext localStorage.
#[wasm_bindgen]
pub fn generate_signing_seed() -> Result<Vec<u8>, JsError> {
    let mut seed = [0u8; SEED_LEN];
    crate::fill_random(&mut seed).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(seed.to_vec())
}

/// Derive the 32-byte public key for a seed.
#[wasm_bindgen]
pub fn public_key_of(seed: &[u8]) -> Result<Vec<u8>, JsError> {
    let seed: [u8; SEED_LEN] = seed
        .try_into()
        .map_err(|_| JsError::new("seed must be 32 bytes"))?;
    Ok(Keypair::from_seed(&seed).public_key().to_vec())
}

/// Sign `message` with `seed`; returns the 64-byte signature.
#[wasm_bindgen]
pub fn sign_message(seed: &[u8], message: &[u8]) -> Result<Vec<u8>, JsError> {
    let seed: [u8; SEED_LEN] = seed
        .try_into()
        .map_err(|_| JsError::new("seed must be 32 bytes"))?;
    Ok(Keypair::from_seed(&seed).sign(message).to_vec())
}

/// Verify a signature; returns a plain boolean, throws only on
/// malformed lengths.
#[wasm_bindgen]
pub fn verify_signature(public_key: &[u8], message: &[u8], signature: &[u8]) -> Result<bool, JsError> {
    let pk: [u8; PUBLIC_KEY_LEN] = public_key
        .try_into()
        .map_err(|_| JsError::new("public key must be 32 bytes"))?;
    let sig: [u8; SIGNATURE_LEN] = signature
        .try_into()
        .map_err(|_| JsError::new("signature must be 64 bytes"))?;
    Ok(crate::sign::verify(&pk, message, &sig))
}

/// SHA-384 of `data` (48 bytes).
#[wasm_bindgen]
pub fn sha384(data: &[u8]) -> Vec<u8> {
    crate::hash::sha384(data).to_vec()
}
