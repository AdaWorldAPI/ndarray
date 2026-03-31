//! #33 Dynamic Task Meta-Framing — gate history triggers frame switching.
//! Science: Lakoff (2004), Tversky & Kahneman (1981), Ashby (1956).

use super::amp::GateState;

pub struct FrameShift {
    pub occurred: bool,
    pub rung_jump: u8,
    pub style_flip: bool,
}

pub fn dynamic_reframe(gate_history: &[GateState]) -> FrameShift {
    let recent_blocks = gate_history.iter().rev().take(3)
        .filter(|g| **g == GateState::Block).count();
    if recent_blocks >= 3 {
        FrameShift { occurred: true, rung_jump: 3, style_flip: true }
    } else if recent_blocks >= 2 {
        FrameShift { occurred: true, rung_jump: 1, style_flip: false }
    } else {
        FrameShift { occurred: false, rung_jump: 0, style_flip: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_no_reframe() {
        let history = vec![GateState::Flow; 5];
        assert!(!dynamic_reframe(&history).occurred);
    }
    #[test]
    fn test_triple_block_reframes() {
        let history = vec![GateState::Block; 3];
        let shift = dynamic_reframe(&history);
        assert!(shift.occurred);
        assert!(shift.style_flip);
        assert_eq!(shift.rung_jump, 3);
    }
}
