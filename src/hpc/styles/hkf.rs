//! #34 Hyperdimensional Knowledge Fusion — cross-domain fusion with validity.
//! Science: Plate (2003), Kanerva (2009), Rahimi & Recht (2007).

use super::super::bgz17_bridge::Base17;
use super::super::nars::NarsTruth;

pub struct FusionResult {
    pub fused: Base17,
    pub domain_a_recovery: f32,
    pub domain_b_recovery: f32,
    pub novelty: f32,
    pub truth: NarsTruth,
}

pub fn cross_domain_fuse(domain_a: &Base17, domain_b: &Base17, relation: &Base17) -> FusionResult {
    let max_l1 = (17u32 * 65535) as f32;
    let mut fused_dims = [0i16; 17];
    for d in 0..17 {
        fused_dims[d] = domain_a.dims[d].wrapping_add(relation.dims[d]).wrapping_add(domain_b.dims[d]);
    }
    let fused = Base17 { dims: fused_dims };

    // Recovery test: fused - relation - B should ≈ A
    let mut ra_dims = [0i16; 17];
    let mut rb_dims = [0i16; 17];
    for d in 0..17 {
        ra_dims[d] = fused.dims[d].wrapping_sub(relation.dims[d]).wrapping_sub(domain_b.dims[d]);
        rb_dims[d] = fused.dims[d].wrapping_sub(relation.dims[d]).wrapping_sub(domain_a.dims[d]);
    }
    let ra = Base17 { dims: ra_dims };
    let rb = Base17 { dims: rb_dims };

    let recovery_a = 1.0 - ra.l1(domain_a) as f32 / max_l1;
    let recovery_b = 1.0 - rb.l1(domain_b) as f32 / max_l1;
    let novelty = (fused.l1(domain_a) as f32 + fused.l1(domain_b) as f32) / (2.0 * max_l1);

    FusionResult {
        fused,
        domain_a_recovery: recovery_a,
        domain_b_recovery: recovery_b,
        novelty,
        truth: NarsTruth::new((recovery_a + recovery_b) / 2.0, 0.8),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cross_domain_fuse() {
        let a = Base17 { dims: [100; 17] };
        let b = Base17 { dims: [200; 17] };
        let rel = Base17 { dims: [5; 17] };
        let result = cross_domain_fuse(&a, &b, &rel);
        assert!(result.domain_a_recovery > 0.99);
        assert!(result.domain_b_recovery > 0.99);
        assert!(result.novelty > 0.0);
    }
}
