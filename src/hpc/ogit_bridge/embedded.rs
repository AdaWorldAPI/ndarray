//! Compile-time-embedded OGIT Cognitive namespace TTL bundle.
//!
//! All 26 Turtle files are validated as UTF-8 at compile time via
//! `include_str!`.  The concatenated bundle is a single `&'static str` that
//! can be fed directly to `TurtleParser::parse` at startup.
//!
//! # Layout
//! ```text
//! assets/cognitive/
//! ├── entities/   (7 class definitions)
//! └── instances/
//!     ├── heels/  (4 seed instances)
//!     ├── hips/   (8 seed instances)
//!     ├── twigs/  (3 seed instances)
//!     └── leaves/ (4 seed instances)
//! ```
//!
//! # Usage
//! ```ignore
//! use ndarray::hpc::ogit_bridge::embedded;
//! let ttl_bundle = embedded::cognitive_ttls();
//! ```

/// Returns the complete OGIT Cognitive namespace as a single concatenated
/// Turtle string.  All 26 TTL files are embedded at compile time; the result
/// is a `&'static str` — zero heap allocation, compile-time UTF-8 validated.
pub fn cognitive_ttls() -> &'static str {
    const TTL: &str = concat!(
        // ── Entities (class hierarchy) ────────────────────────────────────
        include_str!("assets/cognitive/entities/Heel.ttl"),
        "\n",
        include_str!("assets/cognitive/entities/Hip.ttl"),
        "\n",
        include_str!("assets/cognitive/entities/Twig.ttl"),
        "\n",
        include_str!("assets/cognitive/entities/Leaf.ttl"),
        "\n",
        include_str!("assets/cognitive/entities/CognitiveCell.ttl"),
        "\n",
        include_str!("assets/cognitive/entities/SplatCovariance.ttl"),
        "\n",
        include_str!("assets/cognitive/entities/CognitiveTier.ttl"),
        "\n",
        // ── Seed Heel instances ───────────────────────────────────────────
        include_str!("assets/cognitive/instances/heels/reasoning.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/heels/perception.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/heels/memory.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/heels/resonance.ttl"),
        "\n",
        // ── Seed Hip instances ────────────────────────────────────────────
        include_str!("assets/cognitive/instances/hips/deduction.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/hips/abduction.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/hips/induction.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/hips/intuition.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/hips/episodic.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/hips/semantic.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/hips/nars_revision.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/hips/nars_choice.ttl"),
        "\n",
        // ── Seed Twig instances ───────────────────────────────────────────
        include_str!("assets/cognitive/instances/twigs/modus_ponens.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/twigs/modus_tollens.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/twigs/single_evidence_abduce.ttl"),
        "\n",
        // ── Seed Leaf instances ───────────────────────────────────────────
        include_str!("assets/cognitive/instances/leaves/classical_mp.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/leaves/classical_mt.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/leaves/single_evidence_warm.ttl"),
        "\n",
        include_str!("assets/cognitive/instances/leaves/single_evidence_cool.ttl"),
        "\n",
    );
    TTL
}
