//! Arrow interop bridge for cognitive types.
//!
//! Provides zero-copy conversion patterns between ndarray's cognitive types
//! and Apache Arrow columnar format for Lance integration.
//!
//! NOTE: This module provides the schema definitions and conversion traits.
//! The actual Arrow/Lance dependencies are optional and gated behind features.

use super::plane::PLANE_BYTES;

/// Binary fingerprint width in bytes.
pub const BINARY_BYTES: usize = PLANE_BYTES; // 2048

/// Default soaking dimension count.
pub const DEFAULT_SOAKING_DIM: usize = 10000;

/// Schema field names for the bind_nodes_v2 three-plane layout.
pub mod schema {
    pub const S_BINARY: &str = "s_binary";
    pub const P_BINARY: &str = "p_binary";
    pub const O_BINARY: &str = "o_binary";
    pub const S_SOAKING: &str = "s_soaking";
    pub const P_SOAKING: &str = "p_soaking";
    pub const O_SOAKING: &str = "o_soaking";
    pub const SPO_BINARY: &str = "spo_binary";
    pub const NODE_ID: &str = "node_id";
}

/// Role provenance bitflags.
pub mod role_provenance {
    pub const GRAMMAR: u8 = 0x01;
    pub const NSM: u8 = 0x02;
    pub const SIGMA: u8 = 0x04;
    pub const USER: u8 = 0x08;
}

/// Gate state for superposition management.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateState {
    Block,
    Hold,
    Flow,
}

/// Soaking buffer: owned int8 buffer for N entries × D dimensions.
#[derive(Debug, Clone)]
pub struct SoakingBuffer {
    pub data: Vec<i8>,
    pub n_entries: usize,
    pub n_dims: usize,
}

impl SoakingBuffer {
    pub fn new(n_entries: usize, n_dims: usize) -> Self {
        Self {
            data: vec![0i8; n_entries * n_dims],
            n_entries,
            n_dims,
        }
    }

    pub fn entry(&self, idx: usize) -> &[i8] {
        let start = idx * self.n_dims;
        &self.data[start..start + self.n_dims]
    }

    pub fn entry_mut(&mut self, idx: usize) -> &mut [i8] {
        let start = idx * self.n_dims;
        &mut self.data[start..start + self.n_dims]
    }

    /// Crystallize: convert soaking (int8) to binary fingerprint via sign().
    pub fn crystallize(&self, idx: usize) -> Vec<u8> {
        let entry = self.entry(idx);
        let n_bytes = (entry.len() + 7) / 8;
        let mut bits = vec![0u8; n_bytes];
        for (i, &val) in entry.iter().enumerate() {
            if val > 0 {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        bits
    }
}

/// Plane buffer: binary fingerprints + optional soaking layer.
#[derive(Debug, Clone)]
pub struct PlaneBuffer {
    pub binary: Vec<u8>,
    pub soaking: Option<SoakingBuffer>,
    pub n_entries: usize,
    pub binary_bytes: usize,
}

impl PlaneBuffer {
    pub fn new(n_entries: usize, binary_bytes: usize) -> Self {
        Self {
            binary: vec![0u8; n_entries * binary_bytes],
            soaking: None,
            n_entries,
            binary_bytes,
        }
    }

    pub fn with_soaking(mut self, n_dims: usize) -> Self {
        self.soaking = Some(SoakingBuffer::new(self.n_entries, n_dims));
        self
    }

    pub fn binary_entry(&self, idx: usize) -> &[u8] {
        let start = idx * self.binary_bytes;
        &self.binary[start..start + self.binary_bytes]
    }

    pub fn binary_entry_mut(&mut self, idx: usize) -> &mut [u8] {
        let start = idx * self.binary_bytes;
        &mut self.binary[start..start + self.binary_bytes]
    }
}

/// Three-plane fingerprint buffer (S, P, O).
#[derive(Debug, Clone)]
pub struct ThreePlaneFingerprintBuffer {
    pub s: PlaneBuffer,
    pub p: PlaneBuffer,
    pub o: PlaneBuffer,
}

impl ThreePlaneFingerprintBuffer {
    pub fn new(n_entries: usize) -> Self {
        Self {
            s: PlaneBuffer::new(n_entries, BINARY_BYTES),
            p: PlaneBuffer::new(n_entries, BINARY_BYTES),
            o: PlaneBuffer::new(n_entries, BINARY_BYTES),
        }
    }

    pub fn with_soaking(mut self, n_dims: usize) -> Self {
        self.s = self.s.with_soaking(n_dims);
        self.p = self.p.with_soaking(n_dims);
        self.o = self.o.with_soaking(n_dims);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soaking_buffer_basic() {
        let mut buf = SoakingBuffer::new(3, 16);
        buf.entry_mut(0).fill(10);
        buf.entry_mut(1).fill(-5);
        assert_eq!(buf.entry(0)[0], 10);
        assert_eq!(buf.entry(1)[0], -5);
    }

    #[test]
    fn crystallize_positive() {
        let mut buf = SoakingBuffer::new(1, 8);
        buf.entry_mut(0).copy_from_slice(&[1, -1, 1, -1, 1, -1, 1, -1]);
        let bits = buf.crystallize(0);
        assert_eq!(bits[0], 0b01010101);
    }

    #[test]
    fn plane_buffer_entry() {
        let mut pb = PlaneBuffer::new(5, BINARY_BYTES);
        pb.binary_entry_mut(0).fill(0xFF);
        assert_eq!(pb.binary_entry(0)[0], 0xFF);
        assert_eq!(pb.binary_entry(1)[0], 0x00);
    }

    #[test]
    fn three_plane_buffer() {
        let buf = ThreePlaneFingerprintBuffer::new(10);
        assert_eq!(buf.s.n_entries, 10);
        assert_eq!(buf.p.binary_bytes, BINARY_BYTES);
    }

    #[test]
    fn three_plane_with_soaking() {
        let buf = ThreePlaneFingerprintBuffer::new(5).with_soaking(100);
        assert!(buf.s.soaking.is_some());
        assert_eq!(buf.s.soaking.as_ref().unwrap().n_dims, 100);
    }

    #[test]
    fn gate_state_eq() {
        assert_eq!(GateState::Flow, GateState::Flow);
        assert_ne!(GateState::Block, GateState::Hold);
    }

    #[test]
    fn binary_bytes_constant() {
        assert_eq!(BINARY_BYTES, 2048);
    }
}
