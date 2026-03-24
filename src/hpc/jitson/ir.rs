//! Intermediate representation for JIT-compiled scan parameters.
//!
//! Between JSON/YAML config and Cranelift IR, an intermediate representation
//! captures config values in a target-independent form that maps directly
//! to instruction selection decisions.

use std::fmt;

/// Errors from JIT compilation.
#[derive(Debug)]
pub enum JitError {
    /// Cranelift codegen error.
    Codegen(String),
    /// Invalid parameter configuration.
    InvalidParams(String),
    /// CPU feature not available.
    MissingFeature(String),
    /// Module/linking error.
    Module(String),
}

impl fmt::Display for JitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            JitError::Codegen(e) => write!(f, "codegen error: {e}"),
            JitError::InvalidParams(e) => write!(f, "invalid params: {e}"),
            JitError::MissingFeature(e) => write!(f, "missing CPU feature: {e}"),
            JitError::Module(e) => write!(f, "module error: {e}"),
        }
    }
}

impl std::error::Error for JitError {}

/// Scan parameters that compile to native code.
///
/// Each field maps to a specific instruction encoding:
/// - `threshold` → CMP immediate
/// - `top_k` → loop bound constant
/// - `prefetch_ahead` → PREFETCHT0 offset multiplier
/// - `focus_mask` → VPANDQ bitmask (None = all dimensions)
///
/// Cannot be `Copy` because `focus_mask` is `Option<Vec<u32>>`.
#[derive(Debug, Clone)]
pub struct ScanParams {
    /// Tier-1 Hamming threshold — compiled as CMP immediate.
    pub threshold: u32,

    /// Number of top candidates to return.
    pub top_k: u32,

    /// Records to prefetch ahead — compiled as `PREFETCHT0 [ptr + N * record_size]`.
    pub prefetch_ahead: u32,

    /// Focus mask — which dimensions participate in scan.
    /// `None` = all dimensions awake (no mask applied).
    /// `Some(indices)` → compiled to a bitmask with only these bits set.
    pub focus_mask: Option<Vec<u32>>,

    /// Record size in bytes (baked as constant offset arithmetic).
    pub record_size: u32,
}

impl Default for ScanParams {
    fn default() -> Self {
        Self {
            threshold: 500,
            top_k: 32,
            prefetch_ahead: 4,
            focus_mask: None,
            record_size: 1024,
        }
    }
}

/// Philosopher threshold IR — each philosopher's thresholds become
/// CMP immediates and branch hints in the generated collapse gate.
#[derive(Debug, Clone)]
pub struct PhilosopherIR {
    /// Philosopher name (metadata, not compiled).
    pub name: String,

    /// Weight — compiled as branch probability hint.
    pub weight: f32,

    /// Minimum crystallized consensus — CMP immediate (IEEE 754).
    pub crystallized_min: f32,

    /// Maximum tensioned tolerance — CMP immediate (IEEE 754).
    pub tensioned_max: f32,

    /// Noise floor — CMP immediate (IEEE 754).
    pub noise_floor: f32,

    /// Collapse bias: FLOW (fall-through) or HOLD (branch taken).
    pub collapse_bias: CollapseBias,
}

/// Collapse bias — determines branch prediction hint in generated code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollapseBias {
    /// Prefer to resolve quickly — compiled as fall-through path.
    Flow,
    /// Prefer to hold tension — compiled as branch-taken path.
    Hold,
}

/// Collapse gate parameters — compiled to a comparison chain.
#[derive(Debug, Clone)]
pub struct CollapseParams {
    /// Voting strategy.
    pub voting: VotingStrategy,

    /// FLOW threshold — CMP immediate for weighted vote.
    pub flow_threshold: f32,

    /// Veto threshold — short-circuit: if any philosopher with
    /// `weight > veto_threshold` votes BLOCK, emit BLOCK immediately.
    pub veto_threshold: f32,
}

/// How philosopher votes combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VotingStrategy {
    /// Weighted majority vote.
    WeightedMajority,
    /// All must agree.
    Unanimous,
    /// Any one suffices.
    Any,
}

/// Thinking style recipe IR — target-independent representation
/// that compiles to a scan function + collapse gate.
#[derive(Debug, Clone)]
pub struct RecipeIR {
    /// Recipe name.
    pub name: String,
    /// Scan parameters.
    pub scan: ScanParams,
    /// Philosopher thresholds.
    pub philosophers: Vec<PhilosopherIR>,
    /// Collapse gate configuration.
    pub collapse: CollapseParams,
    /// Plasticity: 0.0 = frozen policy, >0 = adaptive with feedback loop.
    pub plasticity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_params_default() {
        let p = ScanParams::default();
        assert_eq!(p.threshold, 500);
        assert_eq!(p.top_k, 32);
        assert_eq!(p.prefetch_ahead, 4);
        assert!(p.focus_mask.is_none());
        assert_eq!(p.record_size, 1024);
    }

    #[test]
    fn scan_params_clone() {
        let p = ScanParams {
            threshold: 100,
            top_k: 10,
            prefetch_ahead: 2,
            focus_mask: Some(vec![1, 5, 47]),
            record_size: 512,
        };
        let p2 = p.clone();
        assert_eq!(p2.threshold, 100);
        assert_eq!(p2.focus_mask.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn jit_error_display() {
        let e = JitError::Codegen("bad IR".to_string());
        assert_eq!(format!("{e}"), "codegen error: bad IR");

        let e = JitError::InvalidParams("negative threshold".to_string());
        assert_eq!(format!("{e}"), "invalid params: negative threshold");

        let e = JitError::MissingFeature("avx512".to_string());
        assert_eq!(format!("{e}"), "missing CPU feature: avx512");

        let e = JitError::Module("link failed".to_string());
        assert_eq!(format!("{e}"), "module error: link failed");
    }

    #[test]
    fn collapse_bias_copy() {
        let b = CollapseBias::Flow;
        let b2 = b; // Copy
        assert_eq!(b, b2);
    }

    #[test]
    fn voting_strategy_copy() {
        let v = VotingStrategy::Unanimous;
        let v2 = v; // Copy
        assert_eq!(v, v2);
    }
}
