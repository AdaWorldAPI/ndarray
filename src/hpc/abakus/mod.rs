//! Abakus: arithmetic over the Codec9 / DR512 representation.
//!
//! [`crate::hpc::codec9`] owns the representation (magnitude, presence,
//! scale, rounding, the never-touched sentinel). This module owns operations
//! on values already in that form, and their carry contract.
//!
//! **Experimental.** No stability guarantee; the API may change without a
//! deprecation period.

pub mod pm;
