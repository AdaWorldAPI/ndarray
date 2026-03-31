//! #27 Multi-Perspective Compression — weighted bundle of Base17 fingerprints.
//! Science: Kanerva (2009), panCAKES (Ishaq et al.), Thomas & Cover (1991).

use super::super::bgz17_bridge::Base17;

pub fn weighted_bundle(items: &[(&Base17, f32)]) -> Base17 {
    if items.is_empty() { return Base17 { dims: [0; 17] }; }
    let mut dims = [0f64; 17];
    let mut total_weight = 0f64;
    for (fp, weight) in items {
        for d in 0..17 { dims[d] += fp.dims[d] as f64 * *weight as f64; }
        total_weight += *weight as f64;
    }
    let mut result = [0i16; 17];
    if total_weight > 0.0 {
        for d in 0..17 { result[d] = (dims[d] / total_weight).round() as i16; }
    }
    Base17 { dims: result }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_weighted_bundle() {
        let a = Base17 { dims: [100; 17] };
        let b = Base17 { dims: [200; 17] };
        let result = weighted_bundle(&[(&a, 3.0), (&b, 1.0)]);
        // Weighted toward a: (100*3 + 200*1) / 4 = 125
        assert_eq!(result.dims[0], 125);
    }
    #[test]
    fn test_equal_weights() {
        let a = Base17 { dims: [100; 17] };
        let b = Base17 { dims: [200; 17] };
        let result = weighted_bundle(&[(&a, 1.0), (&b, 1.0)]);
        assert_eq!(result.dims[0], 150);
    }
}
