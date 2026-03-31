//! #29 Intent-Driven Reframing — detect intent from Base17 features.
//! Science: Wierzbicka (1996), Austin (1962), Porges (2011).

use super::super::bgz17_bridge::Base17;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent { Analytical, Creative, Reflective, Focused, Default }

pub fn detect_intent(query: &Base17) -> Intent {
    // Use Base17 dimension distribution as proxy for intent
    let mean = query.dims.iter().map(|d| *d as i32).sum::<i32>() / 17;
    let variance = query.dims.iter().map(|d| (*d as i32 - mean).pow(2)).sum::<i32>() / 17;
    let activation = query.dims.iter().map(|d| d.unsigned_abs() as u32).sum::<u32>() / 17;
    let direction = query.dims[0]; // sign of dim0 = dominant direction

    if variance > 5000 && activation > 100 { Intent::Creative }
    else if direction < 0 { Intent::Reflective }
    else if activation < 100 { Intent::Focused }
    else if variance < 1000 { Intent::Analytical }
    else { Intent::Default }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_high_variance_creative() {
        let mut dims = [0i16; 17];
        dims[0] = 1000; dims[1] = -1000; dims[2] = 500;
        assert_eq!(detect_intent(&Base17 { dims }), Intent::Creative);
    }
    #[test]
    fn test_negative_reflective() {
        let dims = [-100i16; 17];
        assert_eq!(detect_intent(&Base17 { dims }), Intent::Reflective);
    }
}
