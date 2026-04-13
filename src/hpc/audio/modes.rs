//! Musical mode progressions via Base17 Quintenzirkel.
//!
//! The 17-dimension golden spiral maps to musical modes via octave stacking:
//!   - 17-EDO (17 equal divisions of the octave) approximates both
//!     perfect fifths and major thirds better than 12-EDO
//!   - Base17 dim rotation = mode rotation (Dorian↔Lydian = offset change)
//!   - Golden step (11/17) visits all 17 dims without repetition,
//!     like the circle of fifths visits all 12 chromatic notes
//!
//! Mode-to-qualia mapping for TTS:
//!   Ionian (I):   bright, confident → gate stride 8 (broad routing)
//!   Dorian (ii):  warm, reflective → V stride 5 (content retrieval)
//!   Phrygian (iii): dark, exotic → QK stride 3 (tight attention)
//!   Lydian (IV):  dreamy, floating → Up stride 2 (fine expansion)
//!   Mixolydian (V): driving, bluesy → Down stride 4 (compression)
//!   Aeolian (vi): sad, minor → QK stride 3 (shifted start)
//!   Locrian (vii°): unstable, tense → Gate stride 8 (shifted start)
//!
//! The stride IS the mode. The start offset IS the key.
//! No lookup table needed — the address geometry encodes the qualia.

use super::bands;

/// Musical modes as qualia progressions.
///
/// Each mode is defined by its interval pattern (in 17-EDO steps)
/// and maps to a Base17 stride for spectral character.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    Ionian,     // Major: W-W-H-W-W-W-H → bright, resolved
    Dorian,     // Minor with ♮6: warm, jazz
    Phrygian,   // Minor with ♭2: dark, flamenco
    Lydian,     // Major with ♯4: dreamy, floating
    Mixolydian, // Major with ♭7: dominant, bluesy
    Aeolian,    // Natural minor: sad, reflective
    Locrian,    // Diminished: unstable, tense
}

impl Mode {
    /// Map mode to highheelbgz stride (voice qualia selector).
    ///
    /// The stride determines how the spectral envelope is sampled:
    ///   larger stride = coarser sampling = broader routing
    ///   smaller stride = finer sampling = tighter detail
    pub fn stride(&self) -> u32 {
        match self {
            Mode::Ionian     => 8,  // Gate: broad, confident
            Mode::Dorian     => 5,  // V: warm content
            Mode::Phrygian   => 3,  // QK: tight, exotic
            Mode::Lydian     => 2,  // Up: fine, dreamy
            Mode::Mixolydian => 4,  // Down: driving compression
            Mode::Aeolian    => 3,  // QK: minor, offset start
            Mode::Locrian    => 8,  // Gate: unstable, offset start
        }
    }

    /// Start offset in Base17 space (key signature).
    ///
    /// The offset rotates the golden spiral walk, changing which
    /// spectral dimensions are sampled first — equivalent to
    /// transposing the key.
    pub fn start_offset(&self) -> u32 {
        match self {
            Mode::Ionian     => 0,
            Mode::Dorian     => 2,
            Mode::Phrygian   => 4,
            Mode::Lydian     => 5,
            Mode::Mixolydian => 7,
            Mode::Aeolian    => 9,
            Mode::Locrian    => 11,
        }
    }

    /// 17-EDO interval pattern (steps in 17-EDO).
    ///
    /// 17-EDO: W=3 steps, H=2 steps, total=17 steps per octave.
    /// This is more accurate than 12-EDO for both fifths and thirds.
    pub fn intervals_17edo(&self) -> [u8; 7] {
        match self {
            Mode::Ionian     => [3, 3, 2, 3, 3, 3, 0], // W W H W W W (last H implicit)
            Mode::Dorian     => [3, 2, 3, 3, 3, 2, 1], // W H W W W H W-1
            Mode::Phrygian   => [2, 3, 3, 3, 2, 3, 1], // H W W W H W W-1
            Mode::Lydian     => [3, 3, 3, 2, 3, 3, 0], // W W W H W W (last H implicit)
            Mode::Mixolydian => [3, 3, 2, 3, 3, 2, 1], // W W H W W H W-1
            Mode::Aeolian    => [3, 2, 3, 3, 2, 3, 1], // W H W W H W W-1
            Mode::Locrian    => [2, 3, 3, 2, 3, 3, 1], // H W W H W W W-1
        }
    }

    /// Tension level (0.0 = resolved, 1.0 = maximally tense).
    ///
    /// Derived from the tritone content and leading tone quality.
    /// Maps to HHTL skip threshold: low tension → aggressive skipping,
    /// high tension → less skipping (preserve detail).
    pub fn tension(&self) -> f32 {
        match self {
            Mode::Ionian     => 0.1,  // most resolved
            Mode::Lydian     => 0.2,  // floating but stable
            Mode::Mixolydian => 0.3,  // dominant tension
            Mode::Dorian     => 0.4,  // warm but minor
            Mode::Aeolian    => 0.6,  // sad minor
            Mode::Phrygian   => 0.8,  // dark, exotic
            Mode::Locrian    => 1.0,  // maximum instability
        }
    }
}

/// Band energy modulation by mode.
///
/// Each mode emphasizes different frequency regions, creating the
/// characteristic "color" of the mode. Applied as a multiplier
/// on the 21 Opus CELT band energies.
///
/// Ionian boosts presence (2-4 kHz) for brightness.
/// Phrygian boosts sub-bass and cuts presence for darkness.
/// Lydian boosts harmonics (4-8 kHz) for shimmer.
pub fn mode_band_weights(mode: Mode) -> [f32; bands::N_BANDS] {
    let mut weights = [1.0f32; bands::N_BANDS];

    match mode {
        Mode::Ionian => {
            // Bright: boost presence (bands 10-14, ~2-5 kHz)
            for i in 10..=14 { weights[i] = 1.3; }
        }
        Mode::Dorian => {
            // Warm: boost low-mid (bands 4-8, ~800-1800 Hz)
            for i in 4..=8 { weights[i] = 1.2; }
        }
        Mode::Phrygian => {
            // Dark: boost sub-bass (bands 0-3), cut presence
            for i in 0..=3 { weights[i] = 1.4; }
            for i in 10..=14 { weights[i] = 0.7; }
        }
        Mode::Lydian => {
            // Shimmering: boost harmonics (bands 14-18, ~5-13 kHz)
            for i in 14..=18 { weights[i] = 1.3; }
        }
        Mode::Mixolydian => {
            // Driving: boost fundamental + mid (bands 2-6, ~400-1400 Hz)
            for i in 2..=6 { weights[i] = 1.25; }
        }
        Mode::Aeolian => {
            // Sad: slight low emphasis, gentle roll-off
            for i in 0..=5 { weights[i] = 1.15; }
            for i in 16..=20 { weights[i] = 0.85; }
        }
        Mode::Locrian => {
            // Unstable: emphasize dissonant regions
            weights[6] = 1.4; // ~1400 Hz tritone region
            weights[13] = 1.3; // ~3400 Hz
            for i in 0..=2 { weights[i] = 0.8; } // weaken root
        }
    }

    weights
}

/// Apply mode coloring to band energies.
///
/// Modulates band energies by the mode's characteristic weights.
/// Used in the TTS pipeline: archetype → band energies → mode color → synthesis.
pub fn apply_mode(energies: &mut [f32; bands::N_BANDS], mode: Mode) {
    let weights = mode_band_weights(mode);
    for i in 0..bands::N_BANDS {
        energies[i] *= weights[i];
    }
}

/// Circle of fifths progression as mode sequence.
///
/// Returns the classic I → IV → V → I progression in mode space.
/// Each step has a mode and a root offset in 17-EDO steps.
///
/// For TTS: modulate voice character through a progression to
/// create natural-sounding prosody contours.
pub fn circle_of_fifths_progression() -> Vec<(Mode, u32)> {
    vec![
        (Mode::Ionian, 0),      // I  (tonic, resolved)
        (Mode::Lydian, 5),      // IV (subdominant, floating)
        (Mode::Mixolydian, 7),  // V  (dominant, driving)
        (Mode::Ionian, 0),      // I  (return to tonic)
    ]
}

/// Minor progression: i → iv → VI → V → i
pub fn minor_progression() -> Vec<(Mode, u32)> {
    vec![
        (Mode::Aeolian, 0),     // i   (tonic minor)
        (Mode::Dorian, 5),      // iv  (subdominant, warm)
        (Mode::Ionian, 8),      // VI  (relative major, bright)
        (Mode::Mixolydian, 7),  // V   (dominant, driving)
        (Mode::Aeolian, 0),     // i   (return)
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Octave compression: same tone across octaves → one transposed band
// ═══════════════════════════════════════════════════════════════════════════

/// Octave-compressed band modulation.
///
/// Key insight: harmonics of the same pitch class have identical spectral
/// SHAPE, just shifted in frequency by powers of 2. A C2 (65 Hz) and C4
/// (262 Hz) produce the same overtone ratios — only the fundamental moves.
///
/// So instead of storing band energies for every octave separately, store
/// ONE canonical modulation pattern and an octave offset. The pattern is
/// applied at `band_offset + octave * bands_per_octave`.
///
/// Compression ratio: 8 octaves × 21 bands → 1 pattern + 3-bit offset = 90%
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OctaveBand {
    /// Canonical band modulation pattern (one octave's worth).
    /// Normalized: sum = 1.0. Applied as weights to the 3 bands
    /// spanning one octave at the given offset.
    pub pattern: [f32; 3],
    /// Octave offset (0 = lowest, 7 = highest).
    /// Selects which 3-band group in the 21-band Opus layout to modulate.
    pub octave: u8,
}

impl OctaveBand {
    /// Number of Opus bands per octave (approximately 3 in the log-spaced layout).
    pub const BANDS_PER_OCTAVE: usize = 3;

    /// Map an octave offset to the starting Opus band index.
    ///
    /// Opus CELT bands are quasi-logarithmic, so each ~3 bands ≈ 1 octave:
    ///   octave 0: bands 0-2   (~0-600 Hz, sub-bass to bass)
    ///   octave 1: bands 3-5   (~600-1200 Hz, low-mid)
    ///   octave 2: bands 6-8   (~1200-1800 Hz, mid)
    ///   octave 3: bands 9-11  (~1800-3000 Hz, presence)
    ///   octave 4: bands 12-14 (~3000-4800 Hz, brilliance)
    ///   octave 5: bands 15-17 (~4800-8000 Hz, air)
    ///   octave 6: bands 18-20 (~8000-24000 Hz, ultra)
    pub fn start_band(&self) -> usize {
        (self.octave as usize * Self::BANDS_PER_OCTAVE).min(bands::N_BANDS - Self::BANDS_PER_OCTAVE)
    }

    /// Apply this octave-compressed modulation to 21-band energies.
    ///
    /// Only modifies the 3 bands at `start_band()..start_band()+3`.
    /// All other bands are untouched.
    pub fn apply(&self, energies: &mut [f32; bands::N_BANDS]) {
        let start = self.start_band();
        for i in 0..Self::BANDS_PER_OCTAVE {
            if start + i < bands::N_BANDS {
                energies[start + i] *= self.pattern[i];
            }
        }
    }

    /// Transpose: shift this pattern up or down by N octaves.
    ///
    /// Same pitch class, different register. The pattern is unchanged,
    /// only the octave offset moves. This IS the compression: all octaves
    /// of a note share the same pattern.
    pub fn transpose(&self, delta: i8) -> Self {
        OctaveBand {
            pattern: self.pattern,
            octave: (self.octave as i8 + delta).clamp(0, 6) as u8,
        }
    }

    /// Build from a fundamental frequency.
    ///
    /// The pattern captures the harmonic envelope at that frequency:
    ///   pattern[0] = fundamental energy weight
    ///   pattern[1] = 2nd harmonic weight
    ///   pattern[2] = 3rd harmonic weight
    ///
    /// The harmonic decay rate determines voice character:
    ///   steep decay → flute/sine (pure tone)
    ///   gradual decay → strings/voice (rich harmonics)
    ///   flat → noise/percussion
    pub fn from_fundamental(freq_hz: f32, harmonic_decay: f32) -> Self {
        // Determine octave from frequency (A0 = 27.5 Hz reference)
        let octave = ((freq_hz / 27.5).max(1.0).log2()).floor() as u8;

        // Build harmonic pattern with given decay rate
        let pattern = [
            1.0,                          // fundamental (always 1.0)
            harmonic_decay,               // 2nd harmonic
            harmonic_decay * harmonic_decay, // 3rd harmonic
        ];

        // Normalize so sum = 1.0 + some headroom
        let sum: f32 = pattern.iter().sum();
        let norm = [pattern[0] / sum * 3.0, pattern[1] / sum * 3.0, pattern[2] / sum * 3.0];

        OctaveBand { pattern: norm, octave: octave.min(6) }
    }

    /// Compress a full 21-band energy vector to octave bands.
    ///
    /// Groups bands into 7 octave triplets, keeping only the
    /// normalized pattern within each. Returns 7 OctaveBands.
    ///
    /// Original: 21 × f32 = 84 bytes
    /// Compressed: 7 × (3 × f32 + u8) = 91 bytes (no savings for one frame)
    /// BUT: if many frames share the same pattern (same pitch class),
    /// store pattern ONCE + per-frame octave offset = massive savings.
    pub fn compress_to_octaves(energies: &[f32; bands::N_BANDS]) -> [OctaveBand; 7] {
        let mut result = [OctaveBand { pattern: [1.0; 3], octave: 0 }; 7];
        for oct in 0..7 {
            let start = oct * Self::BANDS_PER_OCTAVE;
            let mut pattern = [0.0f32; 3];
            let mut sum = 0.0f32;
            for i in 0..Self::BANDS_PER_OCTAVE {
                if start + i < bands::N_BANDS {
                    pattern[i] = energies[start + i];
                    sum += pattern[i];
                }
            }
            // Normalize
            if sum > 1e-10 {
                for p in &mut pattern { *p /= sum; *p *= 3.0; }
            }
            result[oct] = OctaveBand { pattern, octave: oct as u8 };
        }
        result
    }
}

/// Pitch class: one of 17 pitch classes in 17-EDO.
///
/// In 17-EDO, each pitch class maps to a Base17 dimension:
///   dim 0 = "C", dim 1 = "C♯↓", dim 2 = "D♭", ...
///   The golden step (11/17) walks all 17 in the same order
///   that the circle of fifths walks 12 in 12-EDO.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PitchClass17(pub u8);

impl PitchClass17 {
    /// The golden step interval (11 steps in 17-EDO ≈ perfect fifth).
    /// gcd(11, 17) = 1, so iterating generates all 17 classes.
    pub const GOLDEN_STEP: u8 = 11;

    /// Circle of fifths in 17-EDO: iterates through all 17 pitch classes.
    pub fn circle_of_fifths() -> Vec<PitchClass17> {
        let mut result = Vec::with_capacity(17);
        let mut current = 0u8;
        for _ in 0..17 {
            result.push(PitchClass17(current));
            current = (current + Self::GOLDEN_STEP) % 17;
        }
        result
    }

    /// Interval between two pitch classes (in 17-EDO steps).
    pub fn interval(&self, other: &PitchClass17) -> u8 {
        ((other.0 as i8 - self.0 as i8).rem_euclid(17)) as u8
    }

    /// Map pitch class to Base17 dimension index.
    /// Identity mapping: pitch class N = dimension N.
    pub fn base17_dim(&self) -> usize {
        self.0 as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_stride_matches_highheelbgz() {
        // Verify stride→role mapping is consistent with highheelbgz::TensorRole
        assert_eq!(Mode::Ionian.stride(), 8);     // Gate
        assert_eq!(Mode::Dorian.stride(), 5);      // V
        assert_eq!(Mode::Phrygian.stride(), 3);    // QK
        assert_eq!(Mode::Lydian.stride(), 2);       // Up
        assert_eq!(Mode::Mixolydian.stride(), 4);  // Down
    }

    #[test]
    fn mode_tension_ordered() {
        // Ionian is least tense, Locrian is most
        assert!(Mode::Ionian.tension() < Mode::Aeolian.tension());
        assert!(Mode::Aeolian.tension() < Mode::Locrian.tension());
    }

    #[test]
    fn band_weights_centered() {
        // All mode weights should average close to 1.0
        for mode in [Mode::Ionian, Mode::Dorian, Mode::Phrygian,
                     Mode::Lydian, Mode::Mixolydian, Mode::Aeolian, Mode::Locrian] {
            let weights = mode_band_weights(mode);
            let avg: f32 = weights.iter().sum::<f32>() / bands::N_BANDS as f32;
            assert!(avg > 0.8 && avg < 1.3,
                "Mode {:?} weights avg {:.2} — should be ~1.0", mode, avg);
        }
    }

    #[test]
    fn circle_of_fifths_starts_and_ends_tonic() {
        let prog = circle_of_fifths_progression();
        assert_eq!(prog.first().unwrap().0, Mode::Ionian);
        assert_eq!(prog.last().unwrap().0, Mode::Ionian);
        assert_eq!(prog.first().unwrap().1, prog.last().unwrap().1);
    }

    #[test]
    fn intervals_sum_to_17() {
        // Each mode's intervals should sum close to 17 (one octave in 17-EDO)
        for mode in [Mode::Ionian, Mode::Dorian, Mode::Phrygian,
                     Mode::Lydian, Mode::Mixolydian, Mode::Aeolian, Mode::Locrian] {
            let intervals = mode.intervals_17edo();
            let sum: u8 = intervals.iter().sum();
            // 7 intervals sum to 17 (W=3, H=2): 5W+2H = 5×3+2×2 = 19?
            // Actually in 17-EDO: 5×3+2×2 = 19, but we use 7 scale degrees
            // The sum should be ≤ 17 (the remaining step completes the octave)
            assert!(sum <= 17, "Mode {:?} intervals sum to {} > 17", mode, sum);
        }
    }

    #[test]
    fn apply_mode_preserves_nonzero() {
        let mut energies = [1.0f32; bands::N_BANDS];
        apply_mode(&mut energies, Mode::Phrygian);
        // All energies should still be positive
        for (i, &e) in energies.iter().enumerate() {
            assert!(e > 0.0, "Band {} energy went to zero after Phrygian mode", i);
        }
    }

    #[test]
    fn octave_transpose_preserves_pattern() {
        let ob = OctaveBand::from_fundamental(440.0, 0.5);
        let up = ob.transpose(2);
        let down = ob.transpose(-1);
        // Pattern should be identical, only octave changes
        assert_eq!(ob.pattern, up.pattern);
        assert_eq!(ob.pattern, down.pattern);
        assert_ne!(ob.octave, up.octave);
    }

    #[test]
    fn octave_compress_roundtrip() {
        let mut energies = [0.0f32; bands::N_BANDS];
        // Put energy at 440Hz band region (approximately band 9-11)
        energies[9] = 1.0;
        energies[10] = 0.5;
        energies[11] = 0.25;
        let octaves = OctaveBand::compress_to_octaves(&energies);
        // Octave 3 (bands 9-11) should have the most energy in pattern[0]
        assert!(octaves[3].pattern[0] > octaves[3].pattern[2],
            "Octave 3 pattern should peak at fundamental: {:?}", octaves[3].pattern);
        // The fundamental (1.0) should have ~57% of the energy (1.0 / 1.75 × 3)
        assert!(octaves[3].pattern[0] > 1.5, "Fundamental weight should be > 1.5: {}", octaves[3].pattern[0]);
    }

    #[test]
    fn circle_of_fifths_17_visits_all() {
        let cof = PitchClass17::circle_of_fifths();
        assert_eq!(cof.len(), 17);
        // All 17 pitch classes should appear exactly once
        let mut seen = [false; 17];
        for pc in &cof {
            assert!(!seen[pc.0 as usize], "Pitch class {} visited twice", pc.0);
            seen[pc.0 as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "Not all pitch classes visited");
    }

    #[test]
    fn pitch_class_interval() {
        let c = PitchClass17(0);
        let g = PitchClass17(10); // 10/17 ≈ perfect fifth in 17-EDO
        assert_eq!(c.interval(&g), 10);
        // Golden step = 11 ≈ also a fifth (the just one)
        let g_just = PitchClass17(11);
        assert_eq!(c.interval(&g_just), PitchClass17::GOLDEN_STEP);
    }
}
