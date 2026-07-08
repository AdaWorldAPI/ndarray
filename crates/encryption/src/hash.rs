//! SHA-384 hashing + domain-separated merkle helpers.
//!
//! SHA-384 is SHA-512 truncated to 48 bytes with a distinct IV: same
//! security family as SHA-256 but immune to length-extension, which is
//! why it is the fingerprint/merkle hash of choice here.

use sha2::{Digest, Sha384};

/// Byte length of a SHA-384 digest.
pub const DIGEST_LEN: usize = 48;

/// Domain-separation prefix for merkle leaves.
const LEAF_TAG: u8 = 0x00;
/// Domain-separation prefix for merkle interior nodes.
const NODE_TAG: u8 = 0x01;

/// Plain SHA-384 of `data`.
pub fn sha384(data: &[u8]) -> [u8; DIGEST_LEN] {
    let mut out = [0u8; DIGEST_LEN];
    out.copy_from_slice(&Sha384::digest(data));
    out
}

/// Hash a merkle **leaf**: `SHA-384(0x00 ‖ data)`. The tag prevents a
/// leaf from ever colliding with an interior node (second-preimage
/// hardening, RFC 6962-style).
pub fn merkle_leaf(data: &[u8]) -> [u8; DIGEST_LEN] {
    let mut h = Sha384::new();
    h.update([LEAF_TAG]);
    h.update(data);
    let mut out = [0u8; DIGEST_LEN];
    out.copy_from_slice(&h.finalize());
    out
}

/// Hash a merkle **interior node**: `SHA-384(0x01 ‖ left ‖ right)`.
pub fn merkle_node(left: &[u8; DIGEST_LEN], right: &[u8; DIGEST_LEN]) -> [u8; DIGEST_LEN] {
    let mut h = Sha384::new();
    h.update([NODE_TAG]);
    h.update(left);
    h.update(right);
    let mut out = [0u8; DIGEST_LEN];
    out.copy_from_slice(&h.finalize());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha384_abc_known_vector() {
        // FIPS 180-4 example vector for SHA-384("abc").
        let expected = "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded163\
                        1a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7";
        let got: String = sha384(b"abc").iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn leaf_and_node_domains_are_separated() {
        // A node over (x, y) must differ from a leaf over the same bytes.
        let x = merkle_leaf(b"x");
        let y = merkle_leaf(b"y");
        let node = merkle_node(&x, &y);
        let mut concat = Vec::new();
        concat.extend_from_slice(&x);
        concat.extend_from_slice(&y);
        assert_ne!(node, merkle_leaf(&concat));
        assert_ne!(node[..], sha384(&concat)[..]);
    }

    #[test]
    fn deterministic() {
        assert_eq!(sha384(b"same"), sha384(b"same"));
        assert_ne!(sha384(b"a"), sha384(b"b"));
    }
}
