//! A sealed channel between a browser and the server — the thing the POC is
//! actually for.
//!
//! TLS already encrypts the transport. This sits *above* it, and the reason is
//! narrow and specific: TLS terminates at whatever proxy, load balancer or
//! remote-desktop layer sits in front of the server, and everything from there
//! inward reads plaintext. This channel's trust anchor is the server's static
//! X448 key, pinned in the client build, so nothing between the browser and
//! the server *process* can read or forge a record.
//!
//! ## What authenticates the server — no signatures involved
//!
//! The client holds the server's static public key. It sends a fresh ephemeral
//! public key; both sides compute `X448(ephemeral, static)`. Only the holder of
//! the static *private* key can compute the same value, so if the first record
//! opens, the peer is the real server. A man in the middle can substitute its
//! own key, but then the two sides derive different keys and the first record
//! fails to open. There is no signature, no certificate and no PKI here — the
//! key agreement itself is the authentication (the Noise `NK` shape).
//!
//! ## What it does NOT protect against
//!
//! The server process sees plaintext: it holds the data and renders it. This is
//! not zero-knowledge against the operator, and no document may claim it is.
//! Nor does it protect a client already running attacker code — it removes the
//! exportable bearer credential, not the compromised machine.
//!
//! ## Wire format
//!
//! ```text
//! handshake:  client -> server   ephemeral_public (56)
//!             (the server replies with its first sealed record; there is no
//!              separate handshake response, so the round trip is free)
//!
//! record:     counter (8, big-endian) ‖ ciphertext ‖ tag (16)
//!             nonce = direction (1) ‖ zeros (15) ‖ counter (8)
//!             aad   = direction (1) ‖ counter (8)
//! ```
//!
//! The counter is per-direction and never reused: a receiver refuses any record
//! whose counter is not strictly greater than the highest it has accepted, so a
//! captured record cannot be replayed and a reordered one cannot be forced.
//!
//! ## The whole protocol
//!
//! ```
//! use encryption::channel::{client_handshake, server_handshake, ServerIdentity};
//!
//! // Server side, once: the public half is pinned into the client build.
//! let identity = ServerIdentity::generate().unwrap();
//! let pinned = identity.public_key();
//!
//! // Client sends its ephemeral public key; that is the entire handshake.
//! let (ephemeral, mut client) = client_handshake(&pinned).unwrap();
//! let mut server = server_handshake(&identity, &ephemeral).unwrap();
//!
//! let record = client.seal(b"the query").unwrap();
//! assert_eq!(server.open(&record).unwrap(), b"the query");
//!
//! // Replaying that record is refused, not merely noticed.
//! assert!(server.open(&record).is_err());
//! ```

use crate::aead::{self, NONCE_LEN, TAG_LEN};
use crate::hash::sha384;
use crate::hkdf_sha384::{expand, extract};
use zeroize::Zeroize;

/// Length of an X448 public key / shared secret, in bytes.
pub const KEY_LEN: usize = 56;
/// Length of the per-record counter prefix, in bytes.
const COUNTER_LEN: usize = 8;
/// Domain separator — bump this and every derived key changes.
const PROTOCOL: &[u8] = b"ada/sealed-channel/x448-hkdf-sha384/v1";

/// Direction tags. They keep the two directions on different keys AND different
/// nonces, so a record cannot be reflected back at its sender.
const DIR_C2S: u8 = 0x01;
const DIR_S2C: u8 = 0x02;

/// Why a channel operation failed. Field-free: an attacker learns nothing from
/// which check rejected it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelError {
    /// The platform CSPRNG was unavailable.
    Rng,
    /// A public key was malformed, or was a low-order point that would force a
    /// known shared secret.
    BadPublicKey,
    /// The record was too short, or its counter replayed / went backwards.
    BadRecord,
    /// The record did not authenticate: wrong key, wrong peer, or tampering.
    /// Deliberately indistinguishable between those cases.
    Decrypt,
    /// Sealing failed (should not happen with valid inputs).
    Encrypt,
}

impl core::fmt::Display for ChannelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ChannelError::Rng => f.write_str("platform CSPRNG unavailable"),
            ChannelError::BadPublicKey => f.write_str("invalid or low-order public key"),
            ChannelError::BadRecord => f.write_str("malformed or replayed record"),
            ChannelError::Decrypt => f.write_str("record did not authenticate"),
            ChannelError::Encrypt => f.write_str("sealing failed"),
        }
    }
}

impl std::error::Error for ChannelError {}

/// The server's long-term identity. Its public half is pinned into the client
/// build; its private half never leaves the server.
pub struct ServerIdentity {
    secret: [u8; KEY_LEN],
    public: [u8; KEY_LEN],
}

impl ServerIdentity {
    /// Generate a fresh server identity from the platform CSPRNG.
    ///
    /// ```
    /// use encryption::channel::ServerIdentity;
    ///
    /// let a = ServerIdentity::generate().unwrap();
    /// let b = ServerIdentity::generate().unwrap();
    /// assert_ne!(a.public_key(), b.public_key());
    /// ```
    ///
    /// # Errors
    ///
    /// [`ChannelError::Rng`] if the platform CSPRNG is unavailable.
    pub fn generate() -> Result<Self, ChannelError> {
        let mut secret = [0u8; KEY_LEN];
        crate::fill_random(&mut secret).map_err(|_| ChannelError::Rng)?;
        Self::from_secret(secret)
    }

    /// Rebuild an identity from stored secret bytes.
    ///
    /// The public half is *recomputed*, never stored alongside — a restart
    /// cannot come back with a mismatched pair.
    ///
    /// ```
    /// use encryption::channel::ServerIdentity;
    ///
    /// let secret = [7u8; encryption::channel::KEY_LEN];
    /// let first = ServerIdentity::from_secret(secret).unwrap();
    /// let after_restart = ServerIdentity::from_secret(secret).unwrap();
    /// assert_eq!(first.public_key(), after_restart.public_key());
    /// ```
    ///
    /// # Errors
    ///
    /// [`ChannelError::BadPublicKey`] if the secret yields a degenerate point.
    pub fn from_secret(secret: [u8; KEY_LEN]) -> Result<Self, ChannelError> {
        let public = x448::x448(secret, x448::X448_BASEPOINT_BYTES).ok_or(ChannelError::BadPublicKey)?;
        Ok(Self { secret, public })
    }

    /// The public key to pin in the client build.
    ///
    /// ```
    /// use encryption::channel::{client_handshake, ServerIdentity};
    ///
    /// let identity = ServerIdentity::generate().unwrap();
    /// // This is the only thing that ships to the client.
    /// assert!(client_handshake(&identity.public_key()).is_ok());
    /// ```
    #[must_use]
    pub fn public_key(&self) -> [u8; KEY_LEN] {
        self.public
    }
}

impl Drop for ServerIdentity {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.secret.zeroize();
    }
}

/// An established channel: two directional keys and the counters that keep
/// every nonce unique.
pub struct SealedChannel {
    send_key: [u8; 32],
    recv_key: [u8; 32],
    send_dir: u8,
    recv_dir: u8,
    send_counter: u64,
    highest_seen: u64,
}

impl Drop for SealedChannel {
    fn drop(&mut self) {
        use zeroize::Zeroize;
        self.send_key.zeroize();
        self.recv_key.zeroize();
    }
}

/// Client side of the handshake.
///
/// Returns the bytes to send (the ephemeral public key) and the established
/// channel. Note what is NOT returned: any indication of whether the server is
/// genuine. That is only learned when a record from the server opens — which
/// is the point, because a man in the middle cannot make one open.
///
/// ```
/// use encryption::channel::{client_handshake, server_handshake, ServerIdentity};
///
/// let real = ServerIdentity::generate().unwrap();
/// let impostor = ServerIdentity::generate().unwrap();
///
/// // The client handshakes against a substituted key and gets no error —
/// // there is nothing to check yet.
/// let (ephemeral, mut client) = client_handshake(&impostor.public_key()).unwrap();
/// let mut real_server = server_handshake(&real, &ephemeral).unwrap();
///
/// // The substitution surfaces here, and only here: nothing opens.
/// let record = client.seal(b"secret").unwrap();
/// assert!(real_server.open(&record).is_err());
/// ```
///
/// # Errors
///
/// [`ChannelError::Rng`] if the CSPRNG is unavailable, or
/// [`ChannelError::BadPublicKey`] if `server_public` is a low-order point that
/// would force a known shared secret.
pub fn client_handshake(server_public: &[u8; KEY_LEN]) -> Result<([u8; KEY_LEN], SealedChannel), ChannelError> {
    let mut ephemeral_secret = [0u8; KEY_LEN];
    crate::fill_random(&mut ephemeral_secret).map_err(|_| ChannelError::Rng)?;

    let ephemeral_public =
        x448::x448(ephemeral_secret, x448::X448_BASEPOINT_BYTES).ok_or(ChannelError::BadPublicKey)?;
    let mut shared = x448::x448(ephemeral_secret, *server_public).ok_or(ChannelError::BadPublicKey)?;
    // The ephemeral scalar has done its two jobs; nothing below reads it again.
    ephemeral_secret.zeroize();

    let channel = derive(&shared, &ephemeral_public, server_public, true);
    // `derive` has folded the DH output into the directional keys; the raw
    // shared secret must not outlive that fold on the stack.
    shared.zeroize();
    Ok((ephemeral_public, channel))
}

/// Server side of the handshake, given the client's ephemeral public key.
///
/// ```
/// use encryption::channel::{client_handshake, server_handshake, ServerIdentity};
///
/// let identity = ServerIdentity::generate().unwrap();
/// let (ephemeral, mut client) = client_handshake(&identity.public_key()).unwrap();
/// let mut server = server_handshake(&identity, &ephemeral).unwrap();
///
/// // Two directions, two keys: the server's reply is not something the
/// // client's own send key could have produced.
/// let up = client.seal(b"request").unwrap();
/// assert_eq!(server.open(&up).unwrap(), b"request");
/// let down = server.seal(b"response").unwrap();
/// assert_eq!(client.open(&down).unwrap(), b"response");
/// ```
///
/// # Errors
///
/// [`ChannelError::BadPublicKey`] if `client_ephemeral` is a low-order point.
pub fn server_handshake(
    identity: &ServerIdentity, client_ephemeral: &[u8; KEY_LEN],
) -> Result<SealedChannel, ChannelError> {
    let mut shared = x448::x448(identity.secret, *client_ephemeral).ok_or(ChannelError::BadPublicKey)?;
    let channel = derive(&shared, client_ephemeral, &identity.public, false);
    shared.zeroize();
    Ok(channel)
}

/// Both sides run exactly this, over exactly these inputs — if either the
/// transcript or the shared secret differs by one bit, the keys differ entirely
/// and nothing opens.
fn derive(
    shared: &[u8; KEY_LEN], client_ephemeral: &[u8; KEY_LEN], server_public: &[u8; KEY_LEN], is_client: bool,
) -> SealedChannel {
    let mut transcript_input = Vec::with_capacity(PROTOCOL.len() + 2 * KEY_LEN);
    transcript_input.extend_from_slice(PROTOCOL);
    transcript_input.extend_from_slice(client_ephemeral);
    transcript_input.extend_from_slice(server_public);
    let transcript = sha384(&transcript_input);

    let prk = extract(&transcript, shared);
    let mut c2s = [0u8; 32];
    let mut s2c = [0u8; 32];
    // Unwrap: 32 bytes is far below HKDF's 255*48 ceiling.
    expand(&prk, b"c2s", &mut c2s).expect("32 bytes is within HKDF's ceiling");
    expand(&prk, b"s2c", &mut s2c).expect("32 bytes is within HKDF's ceiling");

    if is_client {
        SealedChannel {
            send_key: c2s,
            recv_key: s2c,
            send_dir: DIR_C2S,
            recv_dir: DIR_S2C,
            send_counter: 0,
            highest_seen: 0,
        }
    } else {
        SealedChannel {
            send_key: s2c,
            recv_key: c2s,
            send_dir: DIR_S2C,
            recv_dir: DIR_C2S,
            send_counter: 0,
            highest_seen: 0,
        }
    }
}

fn nonce_for(dir: u8, counter: u64) -> [u8; NONCE_LEN] {
    let mut nonce = [0u8; NONCE_LEN];
    nonce[0] = dir;
    nonce[NONCE_LEN - COUNTER_LEN..].copy_from_slice(&counter.to_be_bytes());
    nonce
}

fn aad_for(dir: u8, counter: u64) -> [u8; 1 + COUNTER_LEN] {
    let mut aad = [0u8; 1 + COUNTER_LEN];
    aad[0] = dir;
    aad[1..].copy_from_slice(&counter.to_be_bytes());
    aad
}

impl SealedChannel {
    /// Seal one record for the peer.
    ///
    /// The counter advances on every call and is never reused, which is what
    /// keeps the nonce unique — the one failure this construction does not
    /// survive.
    ///
    /// ```
    /// use encryption::channel::{client_handshake, ServerIdentity};
    ///
    /// let identity = ServerIdentity::generate().unwrap();
    /// let (_, mut client) = client_handshake(&identity.public_key()).unwrap();
    ///
    /// // The same plaintext twice produces two different records — the
    /// // counter is in the nonce, so nothing repeats on the wire.
    /// let first = client.seal(b"same").unwrap();
    /// let second = client.seal(b"same").unwrap();
    /// assert_ne!(first, second);
    /// ```
    ///
    /// # Errors
    ///
    /// [`ChannelError::BadRecord`] if the send counter would overflow, or
    /// [`ChannelError::Encrypt`] if sealing fails.
    pub fn seal(&mut self, plaintext: &[u8]) -> Result<Vec<u8>, ChannelError> {
        let counter = self
            .send_counter
            .checked_add(1)
            .ok_or(ChannelError::BadRecord)?;
        let nonce = nonce_for(self.send_dir, counter);
        let aad = aad_for(self.send_dir, counter);
        let ciphertext =
            aead::seal_with_key(&self.send_key, &nonce, &aad, plaintext).map_err(|_| ChannelError::Encrypt)?;

        self.send_counter = counter;
        let mut record = Vec::with_capacity(COUNTER_LEN + ciphertext.len());
        record.extend_from_slice(&counter.to_be_bytes());
        record.extend_from_slice(&ciphertext);
        Ok(record)
    }

    /// Open a record from the peer.
    ///
    /// Rejects any counter that is not strictly greater than the highest
    /// already accepted: a captured record replayed later is refused, and so is
    /// a reordered one. The counter is only committed after the tag verifies,
    /// so a forged record cannot advance the window and lock out real traffic.
    ///
    /// ```
    /// use encryption::channel::{client_handshake, server_handshake, ServerIdentity};
    ///
    /// let identity = ServerIdentity::generate().unwrap();
    /// let (ephemeral, mut client) = client_handshake(&identity.public_key()).unwrap();
    /// let mut server = server_handshake(&identity, &ephemeral).unwrap();
    ///
    /// let first = client.seal(b"one").unwrap();
    /// let second = client.seal(b"two").unwrap();
    ///
    /// // Delivering out of order accepts the newer record and then refuses
    /// // the older one — the window only moves forward.
    /// assert_eq!(server.open(&second).unwrap(), b"two");
    /// assert!(server.open(&first).is_err());
    /// ```
    ///
    /// # Errors
    ///
    /// [`ChannelError::BadRecord`] if the record is truncated or its counter
    /// replayed, or [`ChannelError::Decrypt`] if the tag does not verify.
    pub fn open(&mut self, record: &[u8]) -> Result<Vec<u8>, ChannelError> {
        if record.len() < COUNTER_LEN + TAG_LEN {
            return Err(ChannelError::BadRecord);
        }
        let mut counter_bytes = [0u8; COUNTER_LEN];
        counter_bytes.copy_from_slice(&record[..COUNTER_LEN]);
        let counter = u64::from_be_bytes(counter_bytes);
        if counter <= self.highest_seen {
            return Err(ChannelError::BadRecord);
        }

        let nonce = nonce_for(self.recv_dir, counter);
        let aad = aad_for(self.recv_dir, counter);
        let plaintext = aead::open_with_key(&self.recv_key, &nonce, &aad, &record[COUNTER_LEN..])
            .map_err(|_| ChannelError::Decrypt)?;

        self.highest_seen = counter;
        Ok(plaintext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the channel actually costs. Printed, not asserted — a number in a
    /// doc comment that nobody measured is how "it is fast" becomes folklore.
    #[test]
    #[ignore = "measures handshake + record cost; run with --release"]
    fn measured_cost_of_a_handshake_and_a_record() {
        use std::time::Instant;

        let server = ServerIdentity::generate().unwrap();
        let pinned = server.public_key();

        let n = 200;
        let t = Instant::now();
        for _ in 0..n {
            let (hello, _c) = client_handshake(&pinned).unwrap();
            let _s = server_handshake(&server, &hello).unwrap();
        }
        let per_handshake = t.elapsed() / n;

        let (mut client, mut server_ch) = pair();
        for size in [64usize, 4096, 65536] {
            let payload = vec![0xABu8; size];
            let reps = 2000;
            let t = Instant::now();
            for _ in 0..reps {
                let rec = client.seal(&payload).unwrap();
                let _ = server_ch.open(&rec).unwrap();
            }
            let per_round = t.elapsed() / reps;
            println!(
                "record {size:>6} B: seal+open {per_round:?}  ({:.1} MiB/s)",
                (size as f64 / (1024.0 * 1024.0)) / per_round.as_secs_f64()
            );
        }
        println!("handshake (both sides): {per_handshake:?}");
    }

    fn pair() -> (SealedChannel, SealedChannel) {
        let server = ServerIdentity::generate().unwrap();
        let (hello, client_ch) = client_handshake(&server.public_key()).unwrap();
        let server_ch = server_handshake(&server, &hello).unwrap();
        (client_ch, server_ch)
    }

    #[test]
    fn a_record_travels_in_both_directions() {
        let (mut client, mut server) = pair();

        let up = client.seal(b"GET /patient/42").unwrap();
        assert_eq!(server.open(&up).unwrap(), b"GET /patient/42");

        let down = server.seal(b"{\"name\":\"Musterfrau\"}").unwrap();
        assert_eq!(client.open(&down).unwrap(), b"{\"name\":\"Musterfrau\"}");
    }

    /// The whole point: a proxy that terminates TLS sees only ciphertext, and
    /// the plaintext is nowhere in the record.
    #[test]
    fn the_record_on_the_wire_does_not_contain_the_plaintext() {
        let (mut client, _) = pair();
        let secret = b"Diagnose: Hypertonie";
        let record = client.seal(secret).unwrap();
        assert!(!record.windows(secret.len()).any(|w| w == secret), "plaintext appeared verbatim in the record");
        assert_eq!(record.len(), COUNTER_LEN + secret.len() + TAG_LEN);
    }

    /// A man in the middle substitutes its own static key. It completes a
    /// handshake — and then nothing it forwards can be opened, in either
    /// direction. This is the property the pinned key buys.
    #[test]
    fn a_man_in_the_middle_with_its_own_key_cannot_open_or_forge() {
        let real_server = ServerIdentity::generate().unwrap();
        let attacker = ServerIdentity::generate().unwrap();

        // Client is told the attacker's key (or the attacker rewrote it).
        let (hello, mut client) = client_handshake(&attacker.public_key()).unwrap();

        // The real server does its side with the client's ephemeral.
        let mut server = server_handshake(&real_server, &hello).unwrap();

        let from_client = client.seal(b"secret").unwrap();
        assert_eq!(server.open(&from_client), Err(ChannelError::Decrypt));

        let from_server = server.seal(b"reply").unwrap();
        assert_eq!(client.open(&from_server), Err(ChannelError::Decrypt));

        // And the attacker, holding its own key, cannot read the client either:
        // it never learns the ephemeral secret.
        let mut attacker_view = server_handshake(&attacker, &hello).unwrap();
        assert!(attacker_view.open(&from_client).is_ok(), "sanity: attacker DID complete a handshake");
    }

    #[test]
    fn a_replayed_record_is_refused() {
        let (mut client, mut server) = pair();
        let record = client.seal(b"transfer 1000").unwrap();
        assert!(server.open(&record).is_ok());
        assert_eq!(server.open(&record), Err(ChannelError::BadRecord));
    }

    #[test]
    fn a_reordered_or_rewound_record_is_refused() {
        let (mut client, mut server) = pair();
        let first = client.seal(b"one").unwrap();
        let second = client.seal(b"two").unwrap();
        assert!(server.open(&second).is_ok());
        // `first` is older than what was accepted — refused, not silently taken.
        assert_eq!(server.open(&first), Err(ChannelError::BadRecord));
    }

    #[test]
    fn a_tampered_record_is_refused_at_every_byte() {
        let (mut client, mut server) = pair();
        let record = client.seal(b"the payload").unwrap();
        for byte in 0..record.len() {
            let mut corrupt = record.clone();
            corrupt[byte] ^= 0x01;
            assert!(server.open(&corrupt).is_err(), "flipping byte {byte} produced an openable record");
        }
        // …and the untouched record still opens, so the loop above proved
        // something other than "everything fails".
        assert!(server.open(&record).is_ok());
    }

    #[test]
    fn truncated_records_are_rejected_without_panicking() {
        let (mut client, mut server) = pair();
        let record = client.seal(b"payload").unwrap();
        for len in 0..record.len() {
            assert!(server.open(&record[..len]).is_err(), "length {len} must be refused");
        }
    }

    /// Two sessions with the same server must not produce the same keys — the
    /// ephemeral is what makes yesterday's captured traffic useless today.
    #[test]
    fn two_sessions_to_the_same_server_do_not_share_keys() {
        let server = ServerIdentity::generate().unwrap();
        let (hello_a, mut client_a) = client_handshake(&server.public_key()).unwrap();
        let (hello_b, _client_b) = client_handshake(&server.public_key()).unwrap();
        assert_ne!(hello_a, hello_b, "ephemeral keys must differ per session");

        let mut server_b = server_handshake(&server, &hello_b).unwrap();
        let from_a = client_a.seal(b"session a").unwrap();
        assert_eq!(
            server_b.open(&from_a),
            Err(ChannelError::Decrypt),
            "session B's channel must not open session A's record"
        );
    }

    /// Direction separation: a record the client sent must not open as if the
    /// server had sent it. Without distinct keys AND nonces per direction, a
    /// reflected record looks genuine.
    #[test]
    fn a_record_cannot_be_reflected_back_at_its_sender() {
        let (mut client, mut server) = pair();
        let from_client = client.seal(b"echo me").unwrap();
        let _ = server.open(&from_client).unwrap();
        assert_eq!(client.open(&from_client), Err(ChannelError::Decrypt));
    }

    /// **The contributory-behaviour guard, both halves.**
    ///
    /// RFC 7748 §6.2 makes rejecting an all-zero shared secret OPTIONAL, so it
    /// is a property of *this* code that we reject — `x448()` returns `None`
    /// for a low-order peer key and `.ok_or(BadPublicKey)?` propagates it. That
    /// is the entire check, it is one `?`, and nothing else in the suite proves
    /// it is still wired.
    ///
    /// Both directions are asserted deliberately: a guard that fires on every
    /// input carries exactly as much information as one that never fires, so
    /// the accept case uses a real basepoint-derived key, not a trivial input.
    ///
    /// This is also the tripwire for a future curve swap. `x25519_dalek::x25519`
    /// returns a bare `[u8; 32]` — no `Option`, nothing for `?` to attach to —
    /// so a mechanical port of this module would compile, pass every other test
    /// here, and silently drop this rejection. If that port ever happens, this
    /// test must be made to fail first.
    #[test]
    fn low_order_peer_keys_are_refused_and_honest_ones_are_not() {
        let server = ServerIdentity::generate().unwrap();

        // FIRES: the canonical low-order points yield an all-zero DH output.
        for (name, bad) in [
            ("all-zero", [0u8; KEY_LEN]),
            ("u=1", {
                let mut u = [0u8; KEY_LEN];
                u[0] = 1;
                u
            }),
        ] {
            assert_eq!(
                client_handshake(&bad).err(),
                Some(ChannelError::BadPublicKey),
                "a {name} peer key must be refused, not silently keyed"
            );
            assert_eq!(
                server_handshake(&server, &bad).err(),
                Some(ChannelError::BadPublicKey),
                "the server side must refuse a {name} client ephemeral too"
            );
        }

        // STAYS SILENT: an honest, basepoint-derived key must go through.
        let (hello, _client) = client_handshake(&server.public_key()).expect("an honest server key must be accepted");
        server_handshake(&server, &hello).expect("an honest client ephemeral must be accepted");
    }
}
