//! #25 Hyperdimensional Pattern Matching — property tests on Base17.
//! Science: Kanerva (1988), Kleyko et al. (2022), Johnson & Lindenstrauss (1984).

use super::super::bgz17_bridge::Base17;

/// Verify triangle inequality: d(a,c) <= d(a,b) + d(b,c)
pub fn verify_triangle_inequality(a: &Base17, b: &Base17, c: &Base17) -> bool {
    let ab = a.l1(b);
    let bc = b.l1(c);
    let ac = a.l1(c);
    ac <= ab + bc
}

/// Verify self-distance is zero
pub fn verify_self_distance(a: &Base17) -> bool {
    a.l1(a) == 0
}

/// Verify symmetry: d(a,b) == d(b,a)
pub fn verify_symmetry(a: &Base17, b: &Base17) -> bool {
    a.l1(b) == b.l1(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_triangle_inequality() {
        let a = Base17 { dims: [0; 17] };
        let b = Base17 { dims: [100; 17] };
        let c = Base17 { dims: [200; 17] };
        assert!(verify_triangle_inequality(&a, &b, &c));
    }
    #[test]
    fn test_self_distance() {
        let a = Base17 { dims: [42; 17] };
        assert!(verify_self_distance(&a));
    }
    #[test]
    fn test_symmetry() {
        let a = Base17 { dims: [10; 17] };
        let b = Base17 { dims: [20; 17] };
        assert!(verify_symmetry(&a, &b));
    }
}
