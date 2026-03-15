//! CogRecord: 4 × 16384-byte (131072-bit) containers = 64KB cognitive unit.
//!
//! Each container is an `Array<u8, Ix1>` of 16384 bytes, queryable via
//! Hamming distance (VPOPCNTDQ) or int8 dot product.

use crate::imp_prelude::*;
use super::bitwise::BitwiseOps;

/// Size of each container in bytes (16384 = 131072 bits).
pub const CONTAINER_BYTES: usize = 16384;
/// Size of each container in bits.
pub const CONTAINER_BITS: usize = CONTAINER_BYTES * 8;
/// Total CogRecord size in bytes (4 containers).
pub const COGRECORD_BYTES: usize = CONTAINER_BYTES * 4;

/// Container indices for semantic clarity.
pub const META: usize = 0;
/// Container index: Content-Addressable Memory.
pub const CAM: usize = 1;
/// Container index: Structural graph position.
pub const BTREE: usize = 2;
/// Container index: Quantized embedding.
pub const EMBED: usize = 3;

/// Sweep mode for batch CogRecord queries.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum SweepMode {
    /// Pure Hamming distance across all 4 containers.
    #[default]
    Hamming,
    /// Hamming for META/CAM/BTREE, int8 cosine for EMBED.
    Hybrid,
}

/// Result of a 4-channel sweep.
#[derive(Clone, Debug, Default)]
pub struct SweepResult {
    /// Index into the candidate array.
    pub index: usize,
    /// Hamming distances per container.
    pub distances: [u64; 4],
}

/// A CogRecord: 4 × 16384-byte containers = 64KB cognitive unit.
///
/// # Example
///
/// ```
/// use ndarray::hpc::cogrecord::CogRecord;
///
/// let cr = CogRecord::zeros();
/// assert_eq!(cr.meta.len(), 16384);
/// assert_eq!(cr.to_bytes().len(), 65536);
/// ```
#[derive(Clone)]
pub struct CogRecord {
    /// Container 0: META
    pub meta: Array<u8, Ix1>,
    /// Container 1: CAM
    pub cam: Array<u8, Ix1>,
    /// Container 2: BTREE
    pub btree: Array<u8, Ix1>,
    /// Container 3: EMBED
    pub embed: Array<u8, Ix1>,
}

impl Default for CogRecord {
    fn default() -> Self {
        Self::zeros()
    }
}

impl CogRecord {
    /// Create a new CogRecord from 4 containers.
    pub fn new(
        meta: Array<u8, Ix1>,
        cam: Array<u8, Ix1>,
        btree: Array<u8, Ix1>,
        embed: Array<u8, Ix1>,
    ) -> Self {
        Self { meta, cam, btree, embed }
    }

    /// Create a zero-initialized CogRecord.
    pub fn zeros() -> Self {
        Self {
            meta: Array::zeros(CONTAINER_BYTES),
            cam: Array::zeros(CONTAINER_BYTES),
            btree: Array::zeros(CONTAINER_BYTES),
            embed: Array::zeros(CONTAINER_BYTES),
        }
    }

    /// Get a container by index (0=META, 1=CAM, 2=BTREE, 3=EMBED).
    pub fn container(&self, idx: usize) -> &Array<u8, Ix1> {
        match idx {
            0 => &self.meta,
            1 => &self.cam,
            2 => &self.btree,
            3 => &self.embed,
            _ => panic!("Container index must be 0..3"),
        }
    }

    /// 4-channel Hamming distance.
    pub fn hamming_4ch(&self, other: &CogRecord) -> [u64; 4] {
        [
            self.meta.hamming_distance(&other.meta),
            self.cam.hamming_distance(&other.cam),
            self.btree.hamming_distance(&other.btree),
            self.embed.hamming_distance(&other.embed),
        ]
    }

    /// Adaptive sweep with per-channel thresholds.
    ///
    /// Returns Some(distances) if all channels are within threshold,
    /// None otherwise (early exit).
    pub fn sweep_adaptive(&self, other: &CogRecord, thresholds: &[u64; 4]) -> Option<[u64; 4]> {
        let d0 = self.meta.hamming_distance(&other.meta);
        if d0 > thresholds[0] {
            return None;
        }
        let d1 = self.cam.hamming_distance(&other.cam);
        if d1 > thresholds[1] {
            return None;
        }
        let d2 = self.btree.hamming_distance(&other.btree);
        if d2 > thresholds[2] {
            return None;
        }
        let d3 = self.embed.hamming_distance(&other.embed);
        if d3 > thresholds[3] {
            return None;
        }
        Some([d0, d1, d2, d3])
    }

    /// HDR sweep: progressive thresholds per channel.
    pub fn hdr_sweep(&self, other: &CogRecord) -> Option<[u64; 4]> {
        let default_thresholds = [
            CONTAINER_BITS as u64 / 4, // 25% threshold
            CONTAINER_BITS as u64 / 4,
            CONTAINER_BITS as u64 / 4,
            CONTAINER_BITS as u64 / 4,
        ];
        self.sweep_adaptive(other, &default_thresholds)
    }

    /// Serialize to bytes (16KB).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(COGRECORD_BYTES);
        bytes.extend_from_slice(self.meta.as_slice().unwrap());
        bytes.extend_from_slice(self.cam.as_slice().unwrap());
        bytes.extend_from_slice(self.btree.as_slice().unwrap());
        bytes.extend_from_slice(self.embed.as_slice().unwrap());
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert!(data.len() >= COGRECORD_BYTES, "Need at least {} bytes", COGRECORD_BYTES);
        Self {
            meta: Array::from_vec(data[0..CONTAINER_BYTES].to_vec()),
            cam: Array::from_vec(data[CONTAINER_BYTES..2 * CONTAINER_BYTES].to_vec()),
            btree: Array::from_vec(data[2 * CONTAINER_BYTES..3 * CONTAINER_BYTES].to_vec()),
            embed: Array::from_vec(data[3 * CONTAINER_BYTES..4 * CONTAINER_BYTES].to_vec()),
        }
    }
}

/// Batch sweep across multiple CogRecords.
///
/// Returns `SweepResult` for all candidates that pass the threshold.
pub fn sweep_cogrecords(
    query: &CogRecord,
    candidates: &[CogRecord],
    thresholds: &[u64; 4],
) -> Vec<SweepResult> {
    candidates
        .iter()
        .enumerate()
        .filter_map(|(i, candidate)| {
            query.sweep_adaptive(candidate, thresholds).map(|distances| SweepResult {
                index: i,
                distances,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let cr = CogRecord::zeros();
        assert_eq!(cr.meta.len(), CONTAINER_BYTES);
        assert_eq!(cr.cam.len(), CONTAINER_BYTES);
        assert_eq!(cr.btree.len(), CONTAINER_BYTES);
        assert_eq!(cr.embed.len(), CONTAINER_BYTES);
    }

    #[test]
    fn test_round_trip() {
        let mut cr = CogRecord::zeros();
        cr.meta[0] = 0xFF;
        cr.cam[100] = 0xAB;
        cr.btree[200] = 0xCD;
        cr.embed[300] = 0xEF;

        let bytes = cr.to_bytes();
        assert_eq!(bytes.len(), COGRECORD_BYTES);

        let cr2 = CogRecord::from_bytes(&bytes);
        assert_eq!(cr2.meta[0], 0xFF);
        assert_eq!(cr2.cam[100], 0xAB);
        assert_eq!(cr2.btree[200], 0xCD);
        assert_eq!(cr2.embed[300], 0xEF);
    }

    #[test]
    fn test_hamming_4ch() {
        let a = CogRecord::zeros();
        let b = CogRecord::zeros();
        let dists = a.hamming_4ch(&b);
        assert_eq!(dists, [0, 0, 0, 0]);
    }

    #[test]
    fn test_sweep_adaptive() {
        let a = CogRecord::zeros();
        let b = CogRecord::zeros();
        let thresholds = [100, 100, 100, 100];
        assert!(a.sweep_adaptive(&b, &thresholds).is_some());
    }
}
