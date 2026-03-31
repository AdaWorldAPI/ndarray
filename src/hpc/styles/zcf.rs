//! #24 Zero-Shot Concept Fusion — bind two concepts, measure recoverability.
//! Science: Plate (2003), Kanerva (2009), Gallant & Okaywe (2013).

use super::super::bgz17_bridge::Base17;

pub struct FusionResult {
    pub fused: Base17,
    pub recovery_a: f32,
    pub recovery_b: f32,
    pub novelty: f32,
}

pub fn fuse(a: &Base17, b: &Base17) -> FusionResult {
    let max_l1 = (17u32 * 65535) as f32;
    let mut fused_dims = [0i16; 17];
    for d in 0..17 { fused_dims[d] = a.dims[d].wrapping_add(b.dims[d]); }
    let fused = Base17 { dims: fused_dims };
    let mut recover_a_dims = [0i16; 17];
    let mut recover_b_dims = [0i16; 17];
    for d in 0..17 {
        recover_a_dims[d] = fused.dims[d].wrapping_sub(b.dims[d]);
        recover_b_dims[d] = fused.dims[d].wrapping_sub(a.dims[d]);
    }
    let ra = Base17 { dims: recover_a_dims };
    let rb = Base17 { dims: recover_b_dims };
    FusionResult {
        recovery_a: 1.0 - ra.l1(a) as f32 / max_l1,
        recovery_b: 1.0 - rb.l1(b) as f32 / max_l1,
        novelty: fused.l1(a) as f32 / max_l1,
        fused,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fusion_recoverable() {
        let a = Base17 { dims: [100; 17] };
        let b = Base17 { dims: [200; 17] };
        let r = fuse(&a, &b);
        assert!(r.recovery_a > 0.99, "wrapping add/sub should be perfect: {}", r.recovery_a);
        assert!(r.recovery_b > 0.99);
    }
}
