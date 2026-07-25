//! Argon2id key derivation — password → 256-bit AEAD key.
//!
//! The parameters travel inside the sealed envelope header (see
//! [`crate::envelope`]) so old blobs stay openable after a cost bump.

use argon2::{Algorithm, Argon2, Params, Version};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Argon2id cost parameters. Stored verbatim (little-endian) in the
/// envelope header, so they are part of the authenticated data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KdfParams {
    /// Memory cost in KiB.
    pub m_cost_kib: u32,
    /// Iteration count (passes over memory).
    pub t_cost: u32,
    /// Parallelism (lanes).
    pub p_cost: u32,
}

/// Smallest memory cost Argon2id accepts (8 KiB per lane).
pub const MIN_M_COST_KIB: u32 = 8;

/// The resource budget a caller is willing to spend on **unauthenticated**
/// parameters.
///
/// Argon2's own maximum is `u32::MAX` KiB — 4 TiB. Refusing only that is
/// not a limit, it is a rounding error: a cost of 1 GiB × 64 passes is
/// equally fatal on a browser tab or a small container, and a handful of
/// concurrent requests carrying it exhausts the host without ever being
/// authenticated. The ceiling therefore has to be a budget the platform can
/// actually absorb, not merely a number below Argon2's roof.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostLimits {
    /// Largest accepted memory cost, in KiB.
    pub max_m_cost_kib: u32,
    /// Largest accepted pass count.
    pub max_t_cost: u32,
    /// Largest accepted lane count.
    pub max_p_cost: u32,
}

impl CostLimits {
    /// 128 MiB, 4 passes, 2 lanes.
    ///
    /// Twice the memory and one pass more than [`KdfParams::DEFAULT`], the
    /// most expensive preset this crate emits — enough headroom for a future
    /// cost bump to open old and new blobs alike, while keeping the worst
    /// case one attacker-supplied header can buy at roughly a third of a
    /// second and 128 MiB.
    ///
    /// A service that opens blobs from the public internet should pass
    /// something tighter (and rate-limit); a batch tool on a big host can
    /// pass something looser. The default errs toward the smallest plausible
    /// deployment, because that is the one that falls over.
    pub const DEFAULT: CostLimits = CostLimits {
        max_m_cost_kib: 128 * 1024,
        max_t_cost: 4,
        max_p_cost: 2,
    };

    /// Exactly the presets this crate ships ([`KdfParams::DEFAULT`] and
    /// [`KdfParams::INTERACTIVE`]), with no headroom at all.
    ///
    /// For a deployment that mints every blob it opens and wants the
    /// narrowest possible pre-authentication surface. The trade is explicit:
    /// raising a cost later means shipping the reader before the writer.
    pub const SHIPPED_PRESETS_ONLY: CostLimits = CostLimits {
        max_m_cost_kib: 64 * 1024,
        max_t_cost: 3,
        max_p_cost: 1,
    };
}

impl Default for CostLimits {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl KdfParams {
    /// Check the cost parameters against [`CostLimits::DEFAULT`] **before**
    /// any memory is reserved. See [`KdfParams::validate_within`] to supply
    /// a budget of your own.
    ///
    /// The ceiling exists because of where these parameters come from. In
    /// [`crate::envelope`] they are read out of the blob's header, and the
    /// header is authenticated — but verifying that authentication requires
    /// the key, and deriving the key means first running Argon2id with the
    /// parameters the blob just supplied. So there is a window, before
    /// anything is proven, where an attacker-chosen `m_cost_kib` decides how
    /// much memory this process asks for. One flipped bit in a stored blob
    /// turns 19 MiB into 4 TiB; the allocation fails, and a failed allocation
    /// in Rust aborts the process rather than unwinding. Tamper detection
    /// works perfectly and the process still dies before reaching it.
    ///
    /// Rejecting must therefore be *cheap* — a comparison, not an attempt.
    ///
    /// ```
    /// use encryption::kdf::{CostLimits, KdfParams, KdfError};
    ///
    /// assert!(KdfParams::INTERACTIVE.validate().is_ok());
    ///
    /// // Not merely absurd values — anything past the platform budget.
    /// let heavy = KdfParams { m_cost_kib: 512 * 1024, ..KdfParams::DEFAULT };
    /// assert_eq!(heavy.validate(), Err(KdfError::CostOutOfPolicy));
    ///
    /// // And a caller with a different budget says so explicitly.
    /// assert!(heavy.validate_within(&CostLimits { max_m_cost_kib: 1024 * 1024, ..CostLimits::DEFAULT }).is_ok());
    /// ```
    pub const fn validate(&self) -> Result<(), KdfError> {
        self.validate_within(&CostLimits::DEFAULT)
    }

    /// Check the cost parameters against a caller-supplied budget, before any
    /// memory is reserved.
    pub const fn validate_within(&self, limits: &CostLimits) -> Result<(), KdfError> {
        if self.m_cost_kib < MIN_M_COST_KIB || self.m_cost_kib > limits.max_m_cost_kib {
            return Err(KdfError::CostOutOfPolicy);
        }
        if self.t_cost == 0 || self.t_cost > limits.max_t_cost {
            return Err(KdfError::CostOutOfPolicy);
        }
        if self.p_cost == 0 || self.p_cost > limits.max_p_cost {
            return Err(KdfError::CostOutOfPolicy);
        }
        Ok(())
    }

    /// Server-grade default: 64 MiB, 3 passes, 1 lane.
    pub const DEFAULT: KdfParams = KdfParams {
        m_cost_kib: 64 * 1024,
        t_cost: 3,
        p_cost: 1,
    };

    /// Interactive / browser-grade: 19 MiB, 2 passes, 1 lane
    /// (the OWASP first-recommended Argon2id configuration).
    /// Use when the derivation runs on every login inside wasm.
    pub const INTERACTIVE: KdfParams = KdfParams {
        m_cost_kib: 19 * 1024,
        t_cost: 2,
        p_cost: 1,
    };
}

impl Default for KdfParams {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// A derived 256-bit key. Wiped from memory on drop.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct DerivedKey(pub(crate) [u8; 32]);

impl DerivedKey {
    /// Borrow the raw key bytes (for handing to the AEAD).
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Key-derivation failure. Field-free on purpose — no secret material,
/// no parameter echo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KdfError {
    /// The cost parameters are outside Argon2's accepted range.
    InvalidParams,
    /// The cost parameters are inside Argon2's range but outside this
    /// crate's policy ceiling — see [`KdfParams::validate`]. Rejected
    /// before any memory is reserved.
    CostOutOfPolicy,
    /// The derivation itself failed (allocation, internal error).
    DerivationFailed,
}

impl core::fmt::Display for KdfError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            KdfError::InvalidParams => f.write_str("invalid Argon2id parameters"),
            KdfError::CostOutOfPolicy => f.write_str("Argon2id cost parameters exceed policy limits"),
            KdfError::DerivationFailed => f.write_str("Argon2id derivation failed"),
        }
    }
}

impl std::error::Error for KdfError {}

/// Derive a 256-bit key from `password` and a 16-byte `salt` with
/// Argon2id (v1.3). Deterministic: same inputs → same key.
pub fn derive_key(password: &[u8], salt: &[u8; 16], params: &KdfParams) -> Result<DerivedKey, KdfError> {
    derive_key_within(password, salt, params, &CostLimits::DEFAULT)
}

/// Derive a key, checking `params` against a caller-supplied budget rather
/// than [`CostLimits::DEFAULT`].
pub fn derive_key_within(
    password: &[u8], salt: &[u8; 16], params: &KdfParams, limits: &CostLimits,
) -> Result<DerivedKey, KdfError> {
    // Before Argon2 sees them, and therefore before anything is allocated.
    params.validate_within(limits)?;
    let argon_params =
        Params::new(params.m_cost_kib, params.t_cost, params.p_cost, Some(32)).map_err(|_| KdfError::InvalidParams)?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, argon_params);
    let mut key = [0u8; 32];
    argon
        .hash_password_into(password, salt, &mut key)
        .map_err(|_| KdfError::DerivationFailed)?;
    Ok(DerivedKey(key))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Small params so tests stay fast; correctness is parameter-independent.
    const TEST_PARAMS: KdfParams = KdfParams {
        m_cost_kib: 32,
        t_cost: 1,
        p_cost: 1,
    };

    #[test]
    fn deterministic_for_same_inputs() {
        let salt = [7u8; 16];
        let a = derive_key(b"correct horse", &salt, &TEST_PARAMS).unwrap();
        let b = derive_key(b"correct horse", &salt, &TEST_PARAMS).unwrap();
        assert_eq!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn different_salt_different_key() {
        let a = derive_key(b"pw", &[1u8; 16], &TEST_PARAMS).unwrap();
        let b = derive_key(b"pw", &[2u8; 16], &TEST_PARAMS).unwrap();
        assert_ne!(a.as_bytes(), b.as_bytes());
    }

    #[test]
    fn different_password_different_key() {
        let salt = [9u8; 16];
        let a = derive_key(b"pw-a", &salt, &TEST_PARAMS).unwrap();
        let b = derive_key(b"pw-b", &salt, &TEST_PARAMS).unwrap();
        assert_ne!(a.as_bytes(), b.as_bytes());
    }

    /// The parameters that reach [`derive_key`] on the open path come out of
    /// an untrusted blob header. This asserts the refusal is cheap — if the
    /// ceiling were missing, this test would not fail, it would abort the
    /// whole test process trying to reserve 4 TiB.
    #[test]
    fn a_cost_beyond_the_platform_budget_is_refused_without_attempting_it() {
        // Not u32::MAX — a value Argon2 would happily accept and run.
        let absurd = KdfParams {
            m_cost_kib: 512 * 1024,
            t_cost: 4,
            p_cost: 1,
        };
        let started = std::time::Instant::now();
        match derive_key(b"pw", &[0u8; 16], &absurd) {
            Err(e) => assert_eq!(e, KdfError::CostOutOfPolicy),
            Ok(_) => panic!("512 MiB of unauthenticated memory cost must be refused"),
        }
        assert!(
            started.elapsed() < std::time::Duration::from_millis(50),
            "the refusal must be a comparison, not an attempt"
        );
    }

    #[test]
    fn the_ceiling_admits_every_shipped_preset() {
        assert_eq!(KdfParams::DEFAULT.validate(), Ok(()));
        assert_eq!(KdfParams::INTERACTIVE.validate(), Ok(()));
        assert_eq!(TEST_PARAMS.validate(), Ok(()));
        // …including under the tightest limits this crate offers.
        let only = CostLimits::SHIPPED_PRESETS_ONLY;
        assert_eq!(KdfParams::DEFAULT.validate_within(&only), Ok(()));
        assert_eq!(KdfParams::INTERACTIVE.validate_within(&only), Ok(()));
    }

    /// The budget is a real bound on work, not a slogan. This measures what
    /// the worst header the default limits admit actually costs, so the
    /// number in the docs is observed rather than asserted.
    #[test]
    #[ignore = "measures the worst admitted cost; run explicitly"]
    fn worst_admitted_cost_is_within_the_documented_budget() {
        let l = CostLimits::DEFAULT;
        let worst = KdfParams {
            m_cost_kib: l.max_m_cost_kib,
            t_cost: l.max_t_cost,
            p_cost: l.max_p_cost,
        };
        let started = std::time::Instant::now();
        assert!(derive_key(b"pw", &[0u8; 16], &worst).is_ok());
        let took = started.elapsed();
        println!(
            "worst admitted: {} MiB x {} passes x {} lanes -> {:?}",
            l.max_m_cost_kib / 1024,
            l.max_t_cost,
            l.max_p_cost,
            took
        );
        assert!(took < std::time::Duration::from_secs(2), "budget too generous: {took:?}");
    }

    /// A caller that knows its host can afford more says so, explicitly.
    #[test]
    fn a_wider_budget_is_opt_in_not_the_default() {
        let heavy = KdfParams {
            m_cost_kib: 512 * 1024,
            t_cost: 1,
            p_cost: 1,
        };
        assert_eq!(heavy.validate(), Err(KdfError::CostOutOfPolicy));
        let wide = CostLimits {
            max_m_cost_kib: 1024 * 1024,
            ..CostLimits::DEFAULT
        };
        assert_eq!(heavy.validate_within(&wide), Ok(()));
    }

    #[test]
    fn zero_passes_and_zero_lanes_are_refused() {
        let no_passes = KdfParams {
            t_cost: 0,
            ..TEST_PARAMS
        };
        let no_lanes = KdfParams {
            p_cost: 0,
            ..TEST_PARAMS
        };
        assert_eq!(no_passes.validate(), Err(KdfError::CostOutOfPolicy));
        assert_eq!(no_lanes.validate(), Err(KdfError::CostOutOfPolicy));
    }

    #[test]
    fn rejects_zero_memory() {
        let bad = KdfParams {
            m_cost_kib: 0,
            t_cost: 1,
            p_cost: 1,
        };
        // No `unwrap_err()` here: DerivedKey deliberately has no Debug
        // impl (a key must never be printable).
        match derive_key(b"pw", &[0u8; 16], &bad) {
            Err(e) => assert_eq!(e, KdfError::CostOutOfPolicy),
            Ok(_) => panic!("zero-memory params must be rejected"),
        }
    }
}
