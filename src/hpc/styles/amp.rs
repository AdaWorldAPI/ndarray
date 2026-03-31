//! #23 Adaptive Meta-Prompting — gate state drives style selection.
//! Science: Sutton & Barto (2018), Ashby (1956), Kahneman (2011).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateState { Flow, Hold, Block }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StyleRecommendation { KeepCurrent, TryNeighbor, RadicalShift }

pub fn adaptive_style_select(gate_history: &[GateState]) -> StyleRecommendation {
    if gate_history.is_empty() { return StyleRecommendation::KeepCurrent; }
    let recent: Vec<&GateState> = gate_history.iter().rev().take(3).collect();
    let blocks = recent.iter().filter(|g| ***g == GateState::Block).count();
    let flows = recent.iter().filter(|g| ***g == GateState::Flow).count();
    if blocks >= 3 { StyleRecommendation::RadicalShift }
    else if blocks >= 1 { StyleRecommendation::TryNeighbor }
    else { StyleRecommendation::KeepCurrent }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_flow_keeps() {
        assert_eq!(adaptive_style_select(&[GateState::Flow; 5]), StyleRecommendation::KeepCurrent);
    }
    #[test]
    fn test_blocks_shift() {
        assert_eq!(adaptive_style_select(&[GateState::Block; 3]), StyleRecommendation::RadicalShift);
    }
    #[test]
    fn test_mixed() {
        assert_eq!(adaptive_style_select(&[GateState::Flow, GateState::Block, GateState::Flow]), StyleRecommendation::TryNeighbor);
    }
}
