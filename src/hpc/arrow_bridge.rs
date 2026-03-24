//! Arrow interop bridge for cognitive types.
//!
//! Provides zero-copy conversion patterns between ndarray's cognitive types
//! and Apache Arrow columnar format for Lance integration.
//!
//! NOTE: This module provides the schema definitions and conversion traits.
//! The actual Arrow/Lance dependencies are optional and gated behind features.

use super::bitwise::hamming_distance_raw;
use super::fingerprint::Fingerprint;
use super::plane::{Plane, PLANE_BYTES};

/// Binary fingerprint width in bytes (16384-bit fingerprint).
pub const PLANE_BINARY_BYTES: usize = 2048;

/// Legacy alias for `PLANE_BINARY_BYTES`.
pub const BINARY_BYTES: usize = PLANE_BYTES; // 2048

/// Soaking accumulator length (i8 entries per plane).
pub const SOAKING_DIMS: usize = 10000;

/// Sigma attention mask width in bytes (10000-bit mask).
pub const SIGMA_MASK_BYTES: usize = 1250;

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
///
/// Lifecycle: `Form` (accumulating) -> `Flow` (serving) -> `Freeze` (archived).
///
/// - **Form** (0): soaking active, binaries mutable.
/// - **Flow** (1): serving, soaking nulled, binary immutable.
/// - **Freeze** (2): archived, entire row compressed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GateState {
    /// Accumulating — soaking active, binaries mutable.
    Form = 0,
    /// Serving — soaking nulled, binary immutable.
    Flow = 1,
    /// Archived — entire row compressed.
    Freeze = 2,
}

impl GateState {
    /// Returns `true` when the gate is in serving mode (`Flow`).
    #[inline]
    pub fn is_serving(&self) -> bool {
        *self == GateState::Flow
    }

    /// Returns `true` when soaking writes are permitted (`Form` only).
    #[inline]
    pub fn can_write_soaking(&self) -> bool {
        *self == GateState::Form
    }
}

/// Role selector for per-plane operations on a `BindNodeV2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Subject plane.
    Subject,
    /// Predicate plane.
    Predicate,
    /// Object plane.
    Object,
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

    /// Zero-copy columnar view of the entire soaking buffer as a flat `&[i8]` slice.
    ///
    /// Layout: `n_entries × n_dims` row-major. Use with
    /// `ArrayView2::from_shape((n_entries, n_dims), buf.as_columnar_slice())`
    /// for zero-copy ndarray interop.
    #[inline]
    pub fn as_columnar_slice(&self) -> &[i8] {
        &self.data
    }

    /// Mutable zero-copy view of the entire soaking buffer.
    #[inline]
    pub fn as_columnar_slice_mut(&mut self) -> &mut [i8] {
        &mut self.data
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

    /// Zero-copy columnar view of all binary fingerprints as a flat `&[u8]` slice.
    ///
    /// Layout: `n_entries × binary_bytes` row-major. Use with
    /// `ArrayView2::from_shape((n_entries, binary_bytes), buf.as_binary_slice())`
    /// for zero-copy ndarray interop.
    #[inline]
    pub fn as_binary_slice(&self) -> &[u8] {
        &self.binary
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

// ============================================================================
// BindNodeV2 — Three-Plane Lance schema row type
// ============================================================================

/// A single row in the bind_nodes_v2 three-plane Lance schema.
///
/// Each row stores three independently addressable planes (Subject, Predicate,
/// Object) plus a composite SPO XOR fingerprint and an attention mask.
///
/// # Lifecycle
///
/// `Form` (accumulating) -> `Flow` (serving) -> `Freeze` (archived).
///
/// In `Form` state, soaking accumulators are active and can be written.
/// Calling [`crystallize`](Self::crystallize) transitions to `Flow`:
/// soaking values are folded into the binary fingerprints via `sign()` and
/// then nulled out. Calling [`freeze`](Self::freeze) transitions to `Freeze`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::arrow_bridge::BindNodeV2;
/// use ndarray::hpc::plane::Plane;
///
/// let mut s = Plane::new();
/// let mut p = Plane::new();
/// let mut o = Plane::new();
/// s.encounter("subject");
/// p.encounter("predicate");
/// o.encounter("object");
/// let mut node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");
/// assert!(node.gate_state.can_write_soaking());
/// node.crystallize();
/// assert!(node.gate_state.is_serving());
/// ```
#[derive(Debug, Clone)]
pub struct BindNodeV2 {
    /// Subject plane binary fingerprint (16384-bit).
    pub subject_binary: [u8; PLANE_BINARY_BYTES],
    /// Subject soaking accumulator. `None` when gate != `Form`.
    pub subject_soaking: Option<Vec<i8>>,
    /// Predicate plane binary fingerprint (16384-bit).
    pub predicate_binary: [u8; PLANE_BINARY_BYTES],
    /// Predicate soaking accumulator. `None` when gate != `Form`.
    pub predicate_soaking: Option<Vec<i8>>,
    /// Object plane binary fingerprint (16384-bit).
    pub object_binary: [u8; PLANE_BINARY_BYTES],
    /// Object soaking accumulator. `None` when gate != `Form`.
    pub object_soaking: Option<Vec<i8>>,
    /// Composite XOR fingerprint: S XOR P XOR O.
    pub spo_binary: [u8; PLANE_BINARY_BYTES],
    /// 10000-bit attention mask (sigma).
    pub sigma_mask: [u8; SIGMA_MASK_BYTES],
    /// NARS frequency (u16 fixed-point, 0..65535).
    pub nars_frequency: u16,
    /// NARS confidence (u16 fixed-point, 0..65535).
    pub nars_confidence: u16,
    /// Current gate state in the Form -> Flow -> Freeze lifecycle.
    pub gate_state: GateState,
    /// Provenance label for this binding.
    pub role_provenance: String,
}

impl BindNodeV2 {
    /// Create a new `BindNodeV2` from three planes.
    ///
    /// Copies each plane's cached bit pattern into the binary columns,
    /// copies each plane's raw accumulator into the soaking columns,
    /// and computes the SPO XOR composite.
    ///
    /// The node starts in `Form` state with soaking active.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::arrow_bridge::BindNodeV2;
    /// use ndarray::hpc::plane::Plane;
    ///
    /// let mut s = Plane::new();
    /// let mut p = Plane::new();
    /// let mut o = Plane::new();
    /// s.encounter("hello");
    /// p.encounter("world");
    /// o.encounter("test");
    /// let node = BindNodeV2::new(&mut s, &mut p, &mut o, "grammar");
    /// assert_eq!(node.gate_state, ndarray::hpc::arrow_bridge::GateState::Form);
    /// ```
    pub fn new(subject: &mut Plane, predicate: &mut Plane, object: &mut Plane, provenance: &str) -> Self {
        subject.ensure_cache();
        predicate.ensure_cache();
        object.ensure_cache();

        let mut subject_binary = [0u8; PLANE_BINARY_BYTES];
        let mut predicate_binary = [0u8; PLANE_BINARY_BYTES];
        let mut object_binary = [0u8; PLANE_BINARY_BYTES];

        subject_binary.copy_from_slice(subject.bits_bytes_ref());
        predicate_binary.copy_from_slice(predicate.bits_bytes_ref());
        object_binary.copy_from_slice(object.bits_bytes_ref());

        // Copy accumulator values as soaking (truncated/padded to SOAKING_DIMS)
        let subject_soaking = Some(Self::acc_to_soaking(subject.acc()));
        let predicate_soaking = Some(Self::acc_to_soaking(predicate.acc()));
        let object_soaking = Some(Self::acc_to_soaking(object.acc()));

        let spo_binary = Self::compute_spo_xor(&subject_binary, &predicate_binary, &object_binary);

        let truth = subject.truth();

        Self {
            subject_binary,
            subject_soaking,
            predicate_binary,
            predicate_soaking,
            object_binary,
            object_soaking,
            spo_binary,
            sigma_mask: [0u8; SIGMA_MASK_BYTES],
            nars_frequency: truth.frequency,
            nars_confidence: truth.confidence,
            gate_state: GateState::Form,
            role_provenance: provenance.to_string(),
        }
    }

    /// Crystallize: transition from `Form` to `Flow`.
    ///
    /// For each plane whose soaking is active, the sign of each soaking
    /// accumulator dimension is folded into the binary fingerprint. After
    /// crystallization, soaking is nulled and the gate moves to `Flow`.
    ///
    /// Does nothing if the gate is not in `Form` state.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::arrow_bridge::{BindNodeV2, GateState};
    /// use ndarray::hpc::plane::Plane;
    ///
    /// let mut s = Plane::new();
    /// let mut p = Plane::new();
    /// let mut o = Plane::new();
    /// s.encounter("a");
    /// p.encounter("b");
    /// o.encounter("c");
    /// let mut node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");
    /// node.crystallize();
    /// assert_eq!(node.gate_state, GateState::Flow);
    /// assert!(node.subject_soaking.is_none());
    /// ```
    pub fn crystallize(&mut self) {
        if self.gate_state != GateState::Form {
            return;
        }

        // Fold soaking sign into binary for each plane
        if let Some(ref soaking) = self.subject_soaking {
            Self::fold_soaking_into_binary(&mut self.subject_binary, soaking);
        }
        if let Some(ref soaking) = self.predicate_soaking {
            Self::fold_soaking_into_binary(&mut self.predicate_binary, soaking);
        }
        if let Some(ref soaking) = self.object_soaking {
            Self::fold_soaking_into_binary(&mut self.object_binary, soaking);
        }

        // Recompute SPO XOR
        self.spo_binary = Self::compute_spo_xor(
            &self.subject_binary,
            &self.predicate_binary,
            &self.object_binary,
        );

        // Null soaking
        self.subject_soaking = None;
        self.predicate_soaking = None;
        self.object_soaking = None;

        self.gate_state = GateState::Flow;
    }

    /// Freeze: transition from `Flow` to `Freeze`.
    ///
    /// Does nothing if the gate is not in `Flow` state.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::arrow_bridge::{BindNodeV2, GateState};
    /// use ndarray::hpc::plane::Plane;
    ///
    /// let mut s = Plane::new();
    /// let mut p = Plane::new();
    /// let mut o = Plane::new();
    /// s.encounter("x"); p.encounter("y"); o.encounter("z");
    /// let mut node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");
    /// node.crystallize();
    /// node.freeze();
    /// assert_eq!(node.gate_state, GateState::Freeze);
    /// ```
    pub fn freeze(&mut self) {
        if self.gate_state != GateState::Flow {
            return;
        }
        self.gate_state = GateState::Freeze;
    }

    /// Compute S XOR P XOR O from the three plane binaries.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::arrow_bridge::BindNodeV2;
    /// use ndarray::hpc::plane::Plane;
    ///
    /// let mut s = Plane::new();
    /// let mut p = Plane::new();
    /// let mut o = Plane::new();
    /// s.encounter("a"); p.encounter("b"); o.encounter("c");
    /// let node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");
    /// let xor = node.spo_xor();
    /// assert_eq!(xor.len(), 2048);
    /// ```
    pub fn spo_xor(&self) -> [u8; PLANE_BINARY_BYTES] {
        Self::compute_spo_xor(&self.subject_binary, &self.predicate_binary, &self.object_binary)
    }

    /// Hamming distance on the composite SPO binary fingerprint.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::arrow_bridge::BindNodeV2;
    /// use ndarray::hpc::plane::Plane;
    ///
    /// let mut s = Plane::new();
    /// let mut p = Plane::new();
    /// let mut o = Plane::new();
    /// s.encounter("a"); p.encounter("b"); o.encounter("c");
    /// let node1 = BindNodeV2::new(&mut s, &mut p, &mut o, "t1");
    /// let node2 = BindNodeV2::new(&mut s, &mut p, &mut o, "t2");
    /// assert_eq!(node1.hamming_to(&node2), 0);
    /// ```
    pub fn hamming_to(&self, other: &BindNodeV2) -> u64 {
        hamming_distance_raw(&self.spo_binary, &other.spo_binary)
    }

    /// Zero-copy view of a plane's binary column as a `Fingerprint<256>`.
    ///
    /// Returns a reference-based fingerprint by copying the bytes into a
    /// `Fingerprint<256>` (the copy is of the fixed-size array, not a heap
    /// allocation).
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::arrow_bridge::{BindNodeV2, Role};
    /// use ndarray::hpc::plane::Plane;
    ///
    /// let mut s = Plane::new();
    /// let mut p = Plane::new();
    /// let mut o = Plane::new();
    /// s.encounter("a"); p.encounter("b"); o.encounter("c");
    /// let node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");
    /// let fp = node.as_fingerprint(Role::Subject);
    /// assert_eq!(fp.as_bytes().len(), 2048);
    /// ```
    pub fn as_fingerprint(&self, role: Role) -> Fingerprint<256> {
        let bytes = match role {
            Role::Subject => &self.subject_binary[..],
            Role::Predicate => &self.predicate_binary[..],
            Role::Object => &self.object_binary[..],
        };
        Fingerprint::from_bytes(bytes)
    }

    /// Per-role Hamming distance between two `BindNodeV2` nodes.
    ///
    /// Computes the Hamming distance on the binary column selected by `role`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::arrow_bridge::{BindNodeV2, Role};
    /// use ndarray::hpc::plane::Plane;
    ///
    /// let mut s = Plane::new();
    /// let mut p = Plane::new();
    /// let mut o = Plane::new();
    /// s.encounter("a"); p.encounter("b"); o.encounter("c");
    /// let node1 = BindNodeV2::new(&mut s, &mut p, &mut o, "t1");
    /// let node2 = BindNodeV2::new(&mut s, &mut p, &mut o, "t2");
    /// assert_eq!(node1.hamming_distance_to(&node2, Role::Subject), 0);
    /// ```
    pub fn hamming_distance_to(&self, other: &BindNodeV2, role: Role) -> u64 {
        let (a, b) = match role {
            Role::Subject => (&self.subject_binary[..], &other.subject_binary[..]),
            Role::Predicate => (&self.predicate_binary[..], &other.predicate_binary[..]),
            Role::Object => (&self.object_binary[..], &other.object_binary[..]),
        };
        hamming_distance_raw(a, b)
    }

    /// Migrate from a V1 single-fingerprint layout.
    ///
    /// Copies the single fingerprint to all three plane binaries, sets soaking
    /// to `None`, and starts in `Flow` state (already crystallized).
    ///
    /// # Example
    ///
    /// ```
    /// use ndarray::hpc::arrow_bridge::{BindNodeV2, GateState, PLANE_BINARY_BYTES};
    ///
    /// let fp = [0xAA_u8; PLANE_BINARY_BYTES];
    /// let node = BindNodeV2::from_v1(&fp);
    /// assert_eq!(node.gate_state, GateState::Flow);
    /// assert_eq!(node.subject_binary, fp);
    /// assert_eq!(node.predicate_binary, fp);
    /// assert_eq!(node.object_binary, fp);
    /// // SPO XOR of identical planes: A ^ A ^ A = A
    /// assert_eq!(node.spo_binary, fp);
    /// ```
    pub fn from_v1(single_fingerprint: &[u8; PLANE_BINARY_BYTES]) -> Self {
        let spo_binary = Self::compute_spo_xor(single_fingerprint, single_fingerprint, single_fingerprint);
        Self {
            subject_binary: *single_fingerprint,
            subject_soaking: None,
            predicate_binary: *single_fingerprint,
            predicate_soaking: None,
            object_binary: *single_fingerprint,
            object_soaking: None,
            spo_binary,
            sigma_mask: [0u8; SIGMA_MASK_BYTES],
            nars_frequency: 32768,
            nars_confidence: 0,
            gate_state: GateState::Flow,
            role_provenance: String::from("v1_migration"),
        }
    }

    // -- internal helpers --

    /// Compute the XOR of three plane binaries.
    fn compute_spo_xor(
        s: &[u8; PLANE_BINARY_BYTES],
        p: &[u8; PLANE_BINARY_BYTES],
        o: &[u8; PLANE_BINARY_BYTES],
    ) -> [u8; PLANE_BINARY_BYTES] {
        let mut result = [0u8; PLANE_BINARY_BYTES];
        for i in 0..PLANE_BINARY_BYTES {
            result[i] = s[i] ^ p[i] ^ o[i];
        }
        result
    }

    /// Convert a Plane accumulator (16384 i8) to a soaking vector (SOAKING_DIMS i8).
    ///
    /// Truncates if the accumulator is longer than SOAKING_DIMS, pads with 0 if shorter.
    fn acc_to_soaking(acc: &[i8; 16384]) -> Vec<i8> {
        let mut soaking = vec![0i8; SOAKING_DIMS];
        let copy_len = SOAKING_DIMS.min(acc.len());
        soaking[..copy_len].copy_from_slice(&acc[..copy_len]);
        soaking
    }

    /// Fold soaking sign bits into a binary fingerprint.
    ///
    /// For each dimension in the soaking vector, if `val > 0` the corresponding
    /// bit is set in the binary; if `val <= 0` it is cleared.
    fn fold_soaking_into_binary(binary: &mut [u8; PLANE_BINARY_BYTES], soaking: &[i8]) {
        let bit_count = (PLANE_BINARY_BYTES * 8).min(soaking.len());
        for i in 0..bit_count {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            if soaking[i] > 0 {
                binary[byte_idx] |= 1 << bit_idx;
            } else {
                binary[byte_idx] &= !(1 << bit_idx);
            }
        }
    }
}

// ============================================================================
// Per-Row Types: ThreePlaneRowBuffer, SoakingRowBuffer, BindNodeV2Row
// ============================================================================

/// Three-plane fingerprint row buffer: holds S/P/O binary fingerprints for
/// a single row, suitable for zero-copy Arrow interop.
///
/// Total: 3 x 2048 = 6144 bytes per row.
#[derive(Debug, Clone)]
pub struct ThreePlaneRowBuffer {
    /// Subject binary fingerprint (2048 bytes).
    pub s_binary: Vec<u8>,
    /// Predicate binary fingerprint (2048 bytes).
    pub p_binary: Vec<u8>,
    /// Object binary fingerprint (2048 bytes).
    pub o_binary: Vec<u8>,
}

impl ThreePlaneRowBuffer {
    /// Create a zeroed three-plane row buffer.
    pub fn new() -> Self {
        Self {
            s_binary: vec![0u8; PLANE_BINARY_BYTES],
            p_binary: vec![0u8; PLANE_BINARY_BYTES],
            o_binary: vec![0u8; PLANE_BINARY_BYTES],
        }
    }

    /// Create from three Plane references (copies their cached bit patterns).
    pub fn from_planes(s: &mut Plane, p: &mut Plane, o: &mut Plane) -> Self {
        s.ensure_cache();
        p.ensure_cache();
        o.ensure_cache();
        Self {
            s_binary: s.bits_bytes_ref().to_vec(),
            p_binary: p.bits_bytes_ref().to_vec(),
            o_binary: o.bits_bytes_ref().to_vec(),
        }
    }

    /// Compute S XOR P XOR O composite fingerprint.
    pub fn xor_spo(&self) -> Vec<u8> {
        let mut result = vec![0u8; PLANE_BINARY_BYTES];
        for i in 0..PLANE_BINARY_BYTES {
            result[i] = self.s_binary[i] ^ self.p_binary[i] ^ self.o_binary[i];
        }
        result
    }

    /// Per-plane Hamming distance to another row buffer.
    ///
    /// Returns `(subject_dist, predicate_dist, object_dist)`.
    pub fn hamming_distance(&self, other: &ThreePlaneRowBuffer) -> (u64, u64, u64) {
        let ds = hamming_distance_raw(&self.s_binary, &other.s_binary);
        let dp = hamming_distance_raw(&self.p_binary, &other.p_binary);
        let do_ = hamming_distance_raw(&self.o_binary, &other.o_binary);
        (ds, dp, do_)
    }

    /// Total byte size of this row buffer (always 3 * PLANE_BINARY_BYTES).
    pub fn total_bytes(&self) -> usize {
        3 * PLANE_BINARY_BYTES
    }
}

impl Default for ThreePlaneRowBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// Soaking row buffer: nullable i8 accumulator for a single plane of a single row.
///
/// When `data` is `Some`, the buffer is active (Form state).
/// When `data` is `None`, the buffer has been crystallized or nulled.
#[derive(Debug, Clone)]
pub struct SoakingRowBuffer {
    /// Soaking data (None = nulled/crystallized).
    pub data: Option<Vec<i8>>,
    /// Dimension count.
    pub dims: usize,
}

impl SoakingRowBuffer {
    /// Create a new active soaking row buffer, zeroed.
    pub fn new(dims: usize) -> Self {
        Self {
            data: Some(vec![0i8; dims]),
            dims,
        }
    }

    /// Crystallize: convert soaking (int8) to binary fingerprint via sign().
    ///
    /// Consumes the soaking data and returns a binary vector.
    /// After crystallization, the buffer is nulled.
    pub fn crystallize(&mut self) -> Vec<u8> {
        let soaking = match self.data.take() {
            Some(d) => d,
            None => return vec![0u8; (self.dims + 7) / 8],
        };
        let n_bytes = (soaking.len() + 7) / 8;
        let mut bits = vec![0u8; n_bytes];
        for (i, &val) in soaking.iter().enumerate() {
            if val > 0 {
                bits[i / 8] |= 1 << (i % 8);
            }
        }
        bits
    }

    /// Returns `true` when soaking is still active (not nulled/crystallized).
    pub fn is_active(&self) -> bool {
        self.data.is_some()
    }

    /// Null out the soaking data (transition to inactive).
    pub fn null_out(&mut self) {
        self.data = None;
    }
}

/// Complete bind_nodes_v2 row type combining fingerprints, soaking, gate,
/// and NARS truth values.
///
/// This is a streamlined per-row type that pairs `ThreePlaneRowBuffer` with
/// per-plane `SoakingRowBuffer`s for the full Lance schema.
#[derive(Debug, Clone)]
pub struct BindNodeV2Row {
    /// Three-plane binary fingerprints.
    pub fingerprints: ThreePlaneRowBuffer,
    /// Subject soaking accumulator.
    pub s_soaking: SoakingRowBuffer,
    /// Predicate soaking accumulator.
    pub p_soaking: SoakingRowBuffer,
    /// Object soaking accumulator.
    pub o_soaking: SoakingRowBuffer,
    /// Gate lifecycle state.
    pub gate: GateState,
    /// NARS frequency (u16 fixed-point).
    pub nars_frequency: u16,
    /// NARS confidence (u16 fixed-point).
    pub nars_confidence: u16,
}

impl BindNodeV2Row {
    /// Create a new row in Form state with active soaking.
    pub fn new(dims: usize) -> Self {
        Self {
            fingerprints: ThreePlaneRowBuffer::new(),
            s_soaking: SoakingRowBuffer::new(dims),
            p_soaking: SoakingRowBuffer::new(dims),
            o_soaking: SoakingRowBuffer::new(dims),
            gate: GateState::Form,
            nars_frequency: 32768,
            nars_confidence: 0,
        }
    }

    /// Crystallize all three soaking buffers, folding sign bits into fingerprints.
    /// Transitions from Form to Flow.
    pub fn crystallize(&mut self) {
        if self.gate != GateState::Form {
            return;
        }
        // Fold each soaking into its binary plane
        if let Some(ref soaking) = self.s_soaking.data {
            fold_sign_into_binary(&mut self.fingerprints.s_binary, soaking);
        }
        if let Some(ref soaking) = self.p_soaking.data {
            fold_sign_into_binary(&mut self.fingerprints.p_binary, soaking);
        }
        if let Some(ref soaking) = self.o_soaking.data {
            fold_sign_into_binary(&mut self.fingerprints.o_binary, soaking);
        }
        self.s_soaking.null_out();
        self.p_soaking.null_out();
        self.o_soaking.null_out();
        self.gate = GateState::Flow;
    }

    /// Freeze: transition from Flow to Freeze.
    pub fn freeze(&mut self) {
        if self.gate == GateState::Flow {
            self.gate = GateState::Freeze;
        }
    }
}

/// Fold soaking sign bits into a binary fingerprint (shared helper).
fn fold_sign_into_binary(binary: &mut [u8], soaking: &[i8]) {
    let bit_count = (binary.len() * 8).min(soaking.len());
    for i in 0..bit_count {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if soaking[i] > 0 {
            binary[byte_idx] |= 1 << bit_idx;
        } else {
            binary[byte_idx] &= !(1 << bit_idx);
        }
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
        assert_ne!(GateState::Form, GateState::Freeze);
    }

    #[test]
    fn binary_bytes_constant() {
        assert_eq!(BINARY_BYTES, 2048);
    }

    // ================================================================
    // BindNodeV2 tests
    // ================================================================

    fn make_test_planes() -> (Plane, Plane, Plane) {
        let mut s = Plane::new();
        let mut p = Plane::new();
        let mut o = Plane::new();
        s.encounter("subject-signal");
        s.encounter("subject-signal");
        s.encounter("subject-signal");
        p.encounter("predicate-signal");
        p.encounter("predicate-signal");
        p.encounter("predicate-signal");
        o.encounter("object-signal");
        o.encounter("object-signal");
        o.encounter("object-signal");
        (s, p, o)
    }

    #[test]
    fn schema_constants() {
        assert_eq!(PLANE_BINARY_BYTES, 2048);
        assert_eq!(SOAKING_DIMS, 10000);
        assert_eq!(SIGMA_MASK_BYTES, 1250);
    }

    #[test]
    fn gate_state_lifecycle_methods() {
        assert!(GateState::Form.can_write_soaking());
        assert!(!GateState::Form.is_serving());
        assert!(GateState::Flow.is_serving());
        assert!(!GateState::Flow.can_write_soaking());
        assert!(!GateState::Freeze.is_serving());
        assert!(!GateState::Freeze.can_write_soaking());
    }

    #[test]
    fn gate_state_repr_values() {
        assert_eq!(GateState::Form as u8, 0);
        assert_eq!(GateState::Flow as u8, 1);
        assert_eq!(GateState::Freeze as u8, 2);
    }

    #[test]
    fn bind_node_v2_new_starts_in_form() {
        let (mut s, mut p, mut o) = make_test_planes();
        let node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");
        assert_eq!(node.gate_state, GateState::Form);
        assert!(node.subject_soaking.is_some());
        assert!(node.predicate_soaking.is_some());
        assert!(node.object_soaking.is_some());
        assert_eq!(node.role_provenance, "test");
    }

    #[test]
    fn bind_node_v2_soaking_length() {
        let (mut s, mut p, mut o) = make_test_planes();
        let node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");
        assert_eq!(node.subject_soaking.as_ref().unwrap().len(), SOAKING_DIMS);
        assert_eq!(node.predicate_soaking.as_ref().unwrap().len(), SOAKING_DIMS);
        assert_eq!(node.object_soaking.as_ref().unwrap().len(), SOAKING_DIMS);
    }

    #[test]
    fn bind_node_v2_spo_xor_is_correct() {
        let (mut s, mut p, mut o) = make_test_planes();
        let node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");
        let expected = BindNodeV2::compute_spo_xor(
            &node.subject_binary,
            &node.predicate_binary,
            &node.object_binary,
        );
        assert_eq!(node.spo_binary, expected);
        assert_eq!(node.spo_xor(), expected);
    }

    #[test]
    fn bind_node_v2_crystallize_lifecycle() {
        let (mut s, mut p, mut o) = make_test_planes();
        let mut node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");

        // Before crystallize: Form state, soaking present
        assert_eq!(node.gate_state, GateState::Form);
        assert!(node.subject_soaking.is_some());

        // Crystallize: Form -> Flow
        node.crystallize();
        assert_eq!(node.gate_state, GateState::Flow);
        assert!(node.subject_soaking.is_none());
        assert!(node.predicate_soaking.is_none());
        assert!(node.object_soaking.is_none());

        // Double-crystallize is a no-op
        let binary_before = node.subject_binary;
        node.crystallize();
        assert_eq!(node.gate_state, GateState::Flow);
        assert_eq!(node.subject_binary, binary_before);
    }

    #[test]
    fn bind_node_v2_freeze_lifecycle() {
        let (mut s, mut p, mut o) = make_test_planes();
        let mut node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");

        // Cannot freeze directly from Form
        node.freeze();
        assert_eq!(node.gate_state, GateState::Form);

        // Must crystallize first
        node.crystallize();
        assert_eq!(node.gate_state, GateState::Flow);

        node.freeze();
        assert_eq!(node.gate_state, GateState::Freeze);

        // Double-freeze is a no-op
        node.freeze();
        assert_eq!(node.gate_state, GateState::Freeze);
    }

    #[test]
    fn bind_node_v2_full_lifecycle() {
        let (mut s, mut p, mut o) = make_test_planes();
        let mut node = BindNodeV2::new(&mut s, &mut p, &mut o, "lifecycle");

        assert!(node.gate_state.can_write_soaking());
        assert!(!node.gate_state.is_serving());

        node.crystallize();
        assert!(!node.gate_state.can_write_soaking());
        assert!(node.gate_state.is_serving());

        node.freeze();
        assert!(!node.gate_state.can_write_soaking());
        assert!(!node.gate_state.is_serving());
        assert_eq!(node.gate_state, GateState::Freeze);
    }

    #[test]
    fn bind_node_v2_hamming_self_zero() {
        let (mut s, mut p, mut o) = make_test_planes();
        let node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");
        assert_eq!(node.hamming_to(&node), 0);
    }

    #[test]
    fn bind_node_v2_hamming_different_nodes() {
        let (mut s1, mut p1, mut o1) = make_test_planes();
        let node1 = BindNodeV2::new(&mut s1, &mut p1, &mut o1, "a");

        let mut s2 = Plane::new();
        let mut p2 = Plane::new();
        let mut o2 = Plane::new();
        s2.encounter("different-subject");
        s2.encounter("different-subject");
        s2.encounter("different-subject");
        p2.encounter("different-predicate");
        p2.encounter("different-predicate");
        p2.encounter("different-predicate");
        o2.encounter("different-object");
        o2.encounter("different-object");
        o2.encounter("different-object");
        let node2 = BindNodeV2::new(&mut s2, &mut p2, &mut o2, "b");

        let dist = node1.hamming_to(&node2);
        assert!(dist > 0, "different planes should have non-zero Hamming distance");
    }

    #[test]
    fn bind_node_v2_as_fingerprint_roundtrip() {
        let (mut s, mut p, mut o) = make_test_planes();
        let node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");

        let fp_s = node.as_fingerprint(Role::Subject);
        assert_eq!(fp_s.as_bytes(), &node.subject_binary[..]);

        let fp_p = node.as_fingerprint(Role::Predicate);
        assert_eq!(fp_p.as_bytes(), &node.predicate_binary[..]);

        let fp_o = node.as_fingerprint(Role::Object);
        assert_eq!(fp_o.as_bytes(), &node.object_binary[..]);
    }

    #[test]
    fn bind_node_v2_per_role_hamming() {
        let (mut s, mut p, mut o) = make_test_planes();
        let node1 = BindNodeV2::new(&mut s, &mut p, &mut o, "a");
        let node2 = BindNodeV2::new(&mut s, &mut p, &mut o, "b");

        // Same planes, so all per-role distances should be 0
        assert_eq!(node1.hamming_distance_to(&node2, Role::Subject), 0);
        assert_eq!(node1.hamming_distance_to(&node2, Role::Predicate), 0);
        assert_eq!(node1.hamming_distance_to(&node2, Role::Object), 0);
    }

    #[test]
    fn bind_node_v2_per_role_hamming_different() {
        let (mut s1, mut p1, mut o1) = make_test_planes();
        let node1 = BindNodeV2::new(&mut s1, &mut p1, &mut o1, "a");

        let mut s2 = Plane::new();
        let mut p2 = Plane::new();
        let mut o2 = Plane::new();
        s2.encounter("other-subject");
        s2.encounter("other-subject");
        s2.encounter("other-subject");
        p2.encounter("predicate-signal"); // same as node1's predicate
        p2.encounter("predicate-signal");
        p2.encounter("predicate-signal");
        o2.encounter("other-object");
        o2.encounter("other-object");
        o2.encounter("other-object");
        let node2 = BindNodeV2::new(&mut s2, &mut p2, &mut o2, "b");

        // Subject and Object should differ, Predicate should match
        assert!(node1.hamming_distance_to(&node2, Role::Subject) > 0);
        assert_eq!(node1.hamming_distance_to(&node2, Role::Predicate), 0);
        assert!(node1.hamming_distance_to(&node2, Role::Object) > 0);
    }

    #[test]
    fn bind_node_v2_from_v1_migration() {
        let fp = [0xAB_u8; PLANE_BINARY_BYTES];
        let node = BindNodeV2::from_v1(&fp);

        // All three planes should be identical to the input
        assert_eq!(node.subject_binary, fp);
        assert_eq!(node.predicate_binary, fp);
        assert_eq!(node.object_binary, fp);

        // SPO XOR of three identical values: A ^ A ^ A = A
        assert_eq!(node.spo_binary, fp);

        // Should start in Flow state (already crystallized)
        assert_eq!(node.gate_state, GateState::Flow);
        assert!(node.gate_state.is_serving());

        // Soaking should be None
        assert!(node.subject_soaking.is_none());
        assert!(node.predicate_soaking.is_none());
        assert!(node.object_soaking.is_none());

        // Default NARS values
        assert_eq!(node.nars_frequency, 32768);
        assert_eq!(node.nars_confidence, 0);

        // Provenance
        assert_eq!(node.role_provenance, "v1_migration");
    }

    #[test]
    fn bind_node_v2_from_v1_zero_fingerprint() {
        let fp = [0u8; PLANE_BINARY_BYTES];
        let node = BindNodeV2::from_v1(&fp);
        assert_eq!(node.spo_binary, [0u8; PLANE_BINARY_BYTES]);
        assert_eq!(node.hamming_to(&node), 0);
    }

    #[test]
    fn bind_node_v2_crystallize_updates_spo() {
        let (mut s, mut p, mut o) = make_test_planes();
        let mut node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");

        let spo_before = node.spo_binary;
        node.crystallize();
        let spo_after = node.spo_binary;

        // After crystallize, SPO should be recomputed from updated binaries
        let expected = BindNodeV2::compute_spo_xor(
            &node.subject_binary,
            &node.predicate_binary,
            &node.object_binary,
        );
        assert_eq!(spo_after, expected);
        // SPO may or may not change depending on soaking content;
        // what matters is consistency
        let _ = spo_before;
    }

    #[test]
    fn bind_node_v2_sigma_mask_size() {
        let (mut s, mut p, mut o) = make_test_planes();
        let node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");
        assert_eq!(node.sigma_mask.len(), SIGMA_MASK_BYTES);
        assert_eq!(node.sigma_mask.len(), 1250);
        // sigma_mask * 8 = 10000 bits
        assert_eq!(node.sigma_mask.len() * 8, 10000);
    }

    #[test]
    fn bind_node_v2_zero_copy_hamming_consistency() {
        let (mut s, mut p, mut o) = make_test_planes();
        let node = BindNodeV2::new(&mut s, &mut p, &mut o, "test");

        // Fingerprint-based hamming should agree with raw hamming
        let fp_s = node.as_fingerprint(Role::Subject);
        let fp_p = node.as_fingerprint(Role::Predicate);
        let dist_fp = fp_s.hamming_distance(&fp_p);
        let dist_raw = hamming_distance_raw(&node.subject_binary, &node.predicate_binary);
        assert_eq!(dist_fp as u64, dist_raw);
    }

    // --- columnar_view tests ---

    #[test]
    fn soaking_columnar_view_zero_copy() {
        let mut buf = SoakingBuffer::new(4, 100);
        buf.entry_mut(2)[50] = 42;
        let slice = buf.as_columnar_slice();
        // Row 2, col 50 → offset 2*100 + 50 = 250
        assert_eq!(slice[250], 42);
        assert_eq!(slice.len(), 4 * 100);
    }

    #[test]
    fn soaking_columnar_view_mut() {
        let mut buf = SoakingBuffer::new(2, 10);
        buf.as_columnar_slice_mut()[15] = -7; // Row 1, col 5
        assert_eq!(buf.entry(1)[5], -7);
    }

    #[test]
    fn plane_buffer_binary_slice() {
        let mut pb = PlaneBuffer::new(3, BINARY_BYTES);
        pb.binary_entry_mut(1)[0] = 0xAB;
        let slice = pb.as_binary_slice();
        assert_eq!(slice.len(), 3 * BINARY_BYTES);
        assert_eq!(slice[BINARY_BYTES], 0xAB);
    }

    // ================================================================
    // ThreePlaneRowBuffer tests
    // ================================================================

    #[test]
    fn three_plane_row_buffer_new() {
        let buf = ThreePlaneRowBuffer::new();
        assert_eq!(buf.s_binary.len(), PLANE_BINARY_BYTES);
        assert_eq!(buf.p_binary.len(), PLANE_BINARY_BYTES);
        assert_eq!(buf.o_binary.len(), PLANE_BINARY_BYTES);
        assert_eq!(buf.total_bytes(), 3 * PLANE_BINARY_BYTES);
    }

    #[test]
    fn three_plane_row_buffer_default() {
        let buf = ThreePlaneRowBuffer::default();
        assert_eq!(buf.total_bytes(), 6144);
    }

    #[test]
    fn three_plane_row_buffer_from_planes() {
        let (mut s, mut p, mut o) = make_test_planes();
        let buf = ThreePlaneRowBuffer::from_planes(&mut s, &mut p, &mut o);
        assert_eq!(buf.s_binary.len(), PLANE_BINARY_BYTES);
        // Non-trivial planes should produce non-zero binaries
        assert!(buf.s_binary.iter().any(|&b| b != 0));
    }

    #[test]
    fn three_plane_row_buffer_xor_spo() {
        let mut buf = ThreePlaneRowBuffer::new();
        buf.s_binary[0] = 0xFF;
        buf.p_binary[0] = 0x0F;
        buf.o_binary[0] = 0xAA;
        let xor = buf.xor_spo();
        assert_eq!(xor[0], 0xFF ^ 0x0F ^ 0xAA);
        assert_eq!(xor[1], 0); // rest is zero
    }

    #[test]
    fn three_plane_row_buffer_hamming_self_zero() {
        let (mut s, mut p, mut o) = make_test_planes();
        let buf = ThreePlaneRowBuffer::from_planes(&mut s, &mut p, &mut o);
        let (ds, dp, do_) = buf.hamming_distance(&buf);
        assert_eq!(ds, 0);
        assert_eq!(dp, 0);
        assert_eq!(do_, 0);
    }

    #[test]
    fn three_plane_row_buffer_hamming_different() {
        let mut buf1 = ThreePlaneRowBuffer::new();
        let mut buf2 = ThreePlaneRowBuffer::new();
        buf1.s_binary.fill(0xFF);
        buf2.s_binary.fill(0x00);
        let (ds, dp, _) = buf1.hamming_distance(&buf2);
        assert_eq!(ds, PLANE_BINARY_BYTES as u64 * 8);
        assert_eq!(dp, 0); // both zero
    }

    // ================================================================
    // SoakingRowBuffer tests
    // ================================================================

    #[test]
    fn soaking_row_buffer_new() {
        let buf = SoakingRowBuffer::new(100);
        assert!(buf.is_active());
        assert_eq!(buf.dims, 100);
        assert_eq!(buf.data.as_ref().unwrap().len(), 100);
    }

    #[test]
    fn soaking_row_buffer_crystallize() {
        let mut buf = SoakingRowBuffer::new(8);
        buf.data.as_mut().unwrap().copy_from_slice(&[1, -1, 1, -1, 1, -1, 1, -1]);
        let bits = buf.crystallize();
        assert_eq!(bits[0], 0b01010101);
        assert!(!buf.is_active()); // should be nulled after crystallize
    }

    #[test]
    fn soaking_row_buffer_crystallize_inactive() {
        let mut buf = SoakingRowBuffer::new(16);
        buf.null_out();
        assert!(!buf.is_active());
        let bits = buf.crystallize();
        // Should return zeroed bits when inactive
        assert!(bits.iter().all(|&b| b == 0));
    }

    #[test]
    fn soaking_row_buffer_null_out() {
        let mut buf = SoakingRowBuffer::new(10);
        assert!(buf.is_active());
        buf.null_out();
        assert!(!buf.is_active());
    }

    // ================================================================
    // BindNodeV2Row tests
    // ================================================================

    #[test]
    fn bind_node_v2_row_new() {
        let row = BindNodeV2Row::new(100);
        assert_eq!(row.gate, GateState::Form);
        assert!(row.s_soaking.is_active());
        assert!(row.p_soaking.is_active());
        assert!(row.o_soaking.is_active());
        assert_eq!(row.nars_frequency, 32768);
        assert_eq!(row.nars_confidence, 0);
        assert_eq!(row.fingerprints.total_bytes(), 6144);
    }

    #[test]
    fn bind_node_v2_row_crystallize() {
        let mut row = BindNodeV2Row::new(16);
        // Put some data in soaking
        row.s_soaking.data.as_mut().unwrap().fill(1);
        row.crystallize();
        assert_eq!(row.gate, GateState::Flow);
        assert!(!row.s_soaking.is_active());
        assert!(!row.p_soaking.is_active());
        assert!(!row.o_soaking.is_active());
    }

    #[test]
    fn bind_node_v2_row_lifecycle() {
        let mut row = BindNodeV2Row::new(16);
        assert_eq!(row.gate, GateState::Form);

        // Cannot freeze from Form
        row.freeze();
        assert_eq!(row.gate, GateState::Form);

        // Crystallize: Form -> Flow
        row.crystallize();
        assert_eq!(row.gate, GateState::Flow);

        // Double crystallize is no-op
        row.crystallize();
        assert_eq!(row.gate, GateState::Flow);

        // Freeze: Flow -> Freeze
        row.freeze();
        assert_eq!(row.gate, GateState::Freeze);
    }

    #[test]
    fn bind_node_v2_row_crystallize_folds_sign() {
        let mut row = BindNodeV2Row::new(16);
        // Set subject soaking to all positive
        row.s_soaking.data.as_mut().unwrap().fill(5);
        row.crystallize();
        // First 2 bytes of s_binary should have bits set (16 bits = 2 bytes)
        assert_eq!(row.fingerprints.s_binary[0], 0xFF);
        assert_eq!(row.fingerprints.s_binary[1], 0xFF);
    }
}
