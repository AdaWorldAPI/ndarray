//! #14 Multimodal Chain-of-Thought — cross-modal binding on Base17.
//! Science: Rahimi & Recht (2008), Neubert et al. (2021), Kleyko et al. (2022).

use super::super::bgz17_bridge::Base17;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality { Text, Image, Audio, Code }

pub fn cross_modal_bind(text: &Base17, image: &Base17, relation: &Base17) -> Base17 {
    let mut dims = [0i16; 17];
    for d in 0..17 {
        dims[d] = text.dims[d].wrapping_add(relation.dims[d]).wrapping_add(image.dims[d]);
    }
    Base17 { dims }
}

pub fn fusion_quality(fused: &Base17, parent_a: &Base17, parent_b: &Base17, relation: &Base17) -> f32 {
    let max_l1 = (17u32 * 65535) as f32;
    let recover_a = recover(fused, parent_b, relation);
    let recover_b = recover(fused, parent_a, relation);
    let qa = 1.0 - recover_a.l1(parent_a) as f32 / max_l1;
    let qb = 1.0 - recover_b.l1(parent_b) as f32 / max_l1;
    (qa + qb) / 2.0
}

fn recover(fused: &Base17, other: &Base17, relation: &Base17) -> Base17 {
    let mut dims = [0i16; 17];
    for d in 0..17 { dims[d] = fused.dims[d].wrapping_sub(relation.dims[d]).wrapping_sub(other.dims[d]); }
    Base17 { dims }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cross_modal_bind() {
        let text = Base17 { dims: [100; 17] };
        let image = Base17 { dims: [200; 17] };
        let relation = Base17 { dims: [10; 17] };
        let fused = cross_modal_bind(&text, &image, &relation);
        assert_ne!(fused.dims, text.dims);
        assert_ne!(fused.dims, image.dims);
    }
}
