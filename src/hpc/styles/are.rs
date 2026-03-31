//! #19 Algorithmic Reverse Engineering — identify transformations from IO pairs.
//! Science: Plate (2003) XOR self-inverse, Kleyko et al. (2022).

use super::super::bgz17_bridge::Base17;

#[derive(Debug, PartialEq)]
pub enum TransformationType {
    Offset([i16; 17]),
    Identity,
    Unknown,
}

pub fn identify_transformation(inputs: &[Base17], outputs: &[Base17]) -> TransformationType {
    if inputs.is_empty() || inputs.len() != outputs.len() { return TransformationType::Unknown; }
    // Check if outputs = inputs + constant offset
    let mut offset = [0i16; 17];
    for d in 0..17 { offset[d] = outputs[0].dims[d].wrapping_sub(inputs[0].dims[d]); }
    let is_offset = inputs.iter().zip(outputs.iter()).all(|(i, o)| {
        (0..17).all(|d| o.dims[d].wrapping_sub(i.dims[d]) == offset[d])
    });
    if is_offset {
        if offset == [0i16; 17] { TransformationType::Identity }
        else { TransformationType::Offset(offset) }
    } else { TransformationType::Unknown }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_detect_offset() {
        let inputs = vec![Base17 { dims: [10; 17] }, Base17 { dims: [20; 17] }];
        let outputs = vec![Base17 { dims: [15; 17] }, Base17 { dims: [25; 17] }];
        match identify_transformation(&inputs, &outputs) {
            TransformationType::Offset(o) => assert_eq!(o, [5i16; 17]),
            _ => panic!("should detect offset"),
        }
    }
    #[test]
    fn test_detect_identity() {
        let data = vec![Base17 { dims: [42; 17] }];
        assert_eq!(identify_transformation(&data, &data), TransformationType::Identity);
    }
}
