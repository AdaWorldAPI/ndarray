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

/// A bundle that has already passed [`crate::bundle::verify`].
///
/// There is no constructor and no way to build one from JavaScript: the only
/// route to a `VerifiedBundle` is through [`verify_bundle`]. That is the
/// point — a browser cannot hold a bundle's body without having verified it
/// first, so "render whatever arrived" is not an expressible mistake.
#[wasm_bindgen]
pub struct VerifiedBundle {
    kind: u8,
    classid: u32,
    bundle_id: u64,
    issued_at: u64,
    publisher: Vec<u8>,
    body: Vec<u8>,
}

#[wasm_bindgen]
impl VerifiedBundle {
    /// 1 = view, 2 = mask, 3 = adapter, 4 = recipe-set.
    #[wasm_bindgen(getter)]
    pub fn kind(&self) -> u8 {
        self.kind
    }
    /// The concept this projection targets.
    #[wasm_bindgen(getter)]
    pub fn classid(&self) -> u32 {
        self.classid
    }
    /// Monotonic per publisher — the value a revocation list keys on.
    #[wasm_bindgen(getter)]
    pub fn bundle_id(&self) -> u64 {
        self.bundle_id
    }
    /// Unix seconds at mint time.
    #[wasm_bindgen(getter)]
    pub fn issued_at(&self) -> u64 {
        self.issued_at
    }
    /// The 32-byte publisher key, already checked against the allowlist.
    #[wasm_bindgen(getter)]
    pub fn publisher(&self) -> Vec<u8> {
        self.publisher.clone()
    }
    /// The IR payload. Safe to render — the signature covered these bytes
    /// together with every header field above.
    #[wasm_bindgen(getter)]
    pub fn body(&self) -> Vec<u8> {
        self.body.clone()
    }
}

/// Verify a signed bundle in the browser before anything renders it.
///
/// `allowed_publishers` is a flat concatenation of 32-byte Ed25519 public
/// keys — the keys this client is willing to trust, which in a shipped
/// deployment are compiled into the wasm bundle rather than fetched. Passing
/// an empty allowlist throws: a client that trusts nobody must fail loudly,
/// not accidentally accept a self-signed bundle.
///
/// Throws on a malformed blob, an unknown publisher, or a bad signature. The
/// three are deliberately distinct messages — a publisher you have not
/// allowlisted is an operator mistake worth naming, while a bad signature is
/// an attack worth naming differently.
#[wasm_bindgen]
pub fn verify_bundle(blob: &[u8], allowed_publishers: &[u8]) -> Result<VerifiedBundle, JsError> {
    if allowed_publishers.is_empty() {
        return Err(JsError::new("no allowed publishers configured"));
    }
    if allowed_publishers.len() % PUBLIC_KEY_LEN != 0 {
        return Err(JsError::new("allowed publishers must be a multiple of 32 bytes"));
    }
    let keys: Vec<[u8; PUBLIC_KEY_LEN]> = allowed_publishers
        .chunks_exact(PUBLIC_KEY_LEN)
        .map(|c| {
            let mut k = [0u8; PUBLIC_KEY_LEN];
            k.copy_from_slice(c);
            k
        })
        .collect();

    let (header, body) = crate::bundle::verify(blob, &keys).map_err(|e| JsError::new(&e.to_string()))?;
    Ok(VerifiedBundle {
        kind: header.kind as u8,
        classid: header.classid,
        bundle_id: header.bundle_id,
        issued_at: header.issued_at,
        publisher: header.publisher.to_vec(),
        body: body.to_vec(),
    })
}

/// SHA-384 of `data` (48 bytes).
#[wasm_bindgen]
pub fn sha384(data: &[u8]) -> Vec<u8> {
    crate::hash::sha384(data).to_vec()
}
