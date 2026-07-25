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

/// SHA-384 of `data` (48 bytes).
#[wasm_bindgen]
pub fn sha384(data: &[u8]) -> Vec<u8> {
    crate::hash::sha384(data).to_vec()
}
