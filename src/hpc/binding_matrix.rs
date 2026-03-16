//! 3D XYZ Binding Popcount Spatial Matrix — spectral analysis of HDC binding space.
//!
//! For three `Fingerprint<256>` containers (X, Y, Z), constructs a 3D scalar field
//! where each point `(i, j, k)` = `popcount(permute(X, i) XOR permute(Y, j) XOR permute(Z, k))`.
//!
//! This reveals:
//! - **Holographic sweet spots**: points near D/2 (maximum information capacity)
//! - **Discriminative hot spots**: points far from D/2 (concentrated information)
//! - **Container correlations**: ridges where two containers align
//! - **Optimal verb offsets**: rotation parameters that maximize orthogonality
//!
//! ## Performance
//!
//! For resolution 256³ = 16.7M points, each requiring 2 XOR + 1 POPCNT on 2KB:
//! ~0.5 seconds on AVX-512 (VPOPCNTDQ). The 256³ matrix fits in 67MB (L3 cache).
//!
//! ## The Sweet Spot Theorem
//!
//! For binary hypervectors of dimension D:
//! - Holographic zone: popcount ∈ [D/2 - √(D/4), D/2 + √(D/4)]
//!   = [8128, 8256] for 16384-bit containers
//! - Discriminative zone: |popcount - D/2| > 2σ = 2√(D/4) = 128

use super::fingerprint::Fingerprint;

/// Bit-level rotation (permutation) of a Fingerprint.
///
/// Rotates the bit pattern left by `offset` positions across the entire
/// N×64-bit vector, wrapping around at the boundary.
fn permute<const N: usize>(fp: &Fingerprint<N>, offset: usize) -> Fingerprint<N> {
    if offset == 0 || N == 0 {
        return fp.clone();
    }
    let total_bits = N * 64;
    let offset = offset % total_bits;
    if offset == 0 {
        return fp.clone();
    }

    let word_shift = offset / 64;
    let bit_shift = offset % 64;

    let mut result = Fingerprint::<N>::zero();

    if bit_shift == 0 {
        for i in 0..N {
            result.words[i] = fp.words[(i + word_shift) % N];
        }
    } else {
        for i in 0..N {
            let src_lo = (i + word_shift) % N;
            let src_hi = (i + word_shift + 1) % N;
            result.words[i] = (fp.words[src_lo] >> bit_shift) | (fp.words[src_hi] << (64 - bit_shift));
        }
    }

    result
}

/// Compute the 3D XYZ binding popcount spatial matrix.
///
/// For three `Fingerprint<256>` source containers,
/// compute `popcount(permute(X, i*step) XOR permute(Y, j*step) XOR permute(Z, k*step))`
/// at `resolution` evenly spaced rotation offsets per axis.
///
/// Returns a flat `Vec<u32>` of shape `[resolution × resolution × resolution]`.
///
/// # Example
///
/// ```
/// use ndarray::hpc::binding_matrix::binding_popcount_3d;
/// use ndarray::hpc::fingerprint::Fingerprint;
///
/// let x = Fingerprint::<256>::ones();
/// let y = Fingerprint::<256>::zero();
/// let z = Fingerprint::<256>::zero();
///
/// let matrix = binding_popcount_3d(&x, &y, &z, 4);
/// assert_eq!(matrix.len(), 4 * 4 * 4);
/// ```
pub fn binding_popcount_3d(
    x: &Fingerprint<256>,
    y: &Fingerprint<256>,
    z: &Fingerprint<256>,
    resolution: usize,
) -> Vec<u32> {
    let total_bits = 256 * 64; // 16384
    let step = if resolution > 1 {
        total_bits / resolution
    } else {
        0
    };

    // Pre-compute all permutations (avoid recomputation in inner loops)
    let x_perms: Vec<Fingerprint<256>> = (0..resolution).map(|i| permute(x, i * step)).collect();
    let y_perms: Vec<Fingerprint<256>> = (0..resolution).map(|j| permute(y, j * step)).collect();
    let z_perms: Vec<Fingerprint<256>> = (0..resolution).map(|k| permute(z, k * step)).collect();

    let mut matrix = vec![0u32; resolution * resolution * resolution];

    for i in 0..resolution {
        for j in 0..resolution {
            // XOR x_i with y_j once, reuse for all k
            let xy = &x_perms[i] ^ &y_perms[j];

            for k in 0..resolution {
                // popcount(xy XOR z_k) = hamming_distance(xy, z_k)
                let popcnt = xy.hamming_distance(&z_perms[k]);
                matrix[i * resolution * resolution + j * resolution + k] = popcnt;
            }
        }
    }

    matrix
}

/// Find the holographic sweet spot: rotation offsets where
/// 3-way binding produces popcount closest to D/2.
///
/// Returns `(i, j, k, z_score)` sorted by z_score (ascending).
/// Points with z_score < 1.0 are in the holographic basin.
pub fn find_holographic_sweet_spot(
    matrix: &[u32],
    resolution: usize,
    total_bits: usize,
) -> Vec<(usize, usize, usize, f64)> {
    let target = total_bits as f64 / 2.0;
    let sigma = (total_bits as f64 / 4.0).sqrt();

    let mut spots: Vec<(usize, usize, usize, f64)> = Vec::new();

    for i in 0..resolution {
        for j in 0..resolution {
            for k in 0..resolution {
                let idx = i * resolution * resolution + j * resolution + k;
                let deviation = (matrix[idx] as f64 - target).abs();
                let z_score = deviation / sigma;

                if z_score < 1.0 {
                    spots.push((i, j, k, z_score));
                }
            }
        }
    }

    spots.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    spots
}

/// Find discriminative hot spots: where binding concentrates information.
///
/// Returns `(i, j, k, z_score)` sorted by z_score (descending).
/// Points with z_score > 2.0 are in the discriminative zone.
pub fn find_discriminative_spots(
    matrix: &[u32],
    resolution: usize,
    total_bits: usize,
) -> Vec<(usize, usize, usize, f64)> {
    let target = total_bits as f64 / 2.0;
    let sigma = (total_bits as f64 / 4.0).sqrt();

    let mut spots: Vec<(usize, usize, usize, f64)> = Vec::new();

    for i in 0..resolution {
        for j in 0..resolution {
            for k in 0..resolution {
                let idx = i * resolution * resolution + j * resolution + k;
                let deviation = (matrix[idx] as f64 - target).abs();
                let z_score = deviation / sigma;

                if z_score > 2.0 {
                    spots.push((i, j, k, z_score));
                }
            }
        }
    }

    spots.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
    spots
}

/// Extract 2D cross-section at fixed Z index for visualization.
///
/// Returns a `resolution × resolution` matrix of popcount values.
pub fn cross_section_at_z(matrix: &[u32], resolution: usize, z_idx: usize) -> Vec<Vec<u32>> {
    let mut slice = vec![vec![0u32; resolution]; resolution];
    for i in 0..resolution {
        for j in 0..resolution {
            slice[i][j] = matrix[i * resolution * resolution + j * resolution + z_idx];
        }
    }
    slice
}

/// Compute the gradient field: which direction increases/decreases popcount fastest.
///
/// Uses central differences. The gradient magnitude reveals "information flow"
/// in the binding space — high gradient = sharp transition between holographic
/// and discriminative regions.
///
/// Returns `[dx, dy, dz]` per point.
pub fn popcount_gradient_3d(matrix: &[u32], resolution: usize) -> Vec<[f32; 3]> {
    let r = resolution;
    let mut gradient = vec![[0.0f32; 3]; r * r * r];

    let idx = |x: usize, y: usize, z: usize| x * r * r + y * r + z;

    for i in 1..r - 1 {
        for j in 1..r - 1 {
            for k in 1..r - 1 {
                let dx = matrix[idx(i + 1, j, k)] as f32 - matrix[idx(i - 1, j, k)] as f32;
                let dy = matrix[idx(i, j + 1, k)] as f32 - matrix[idx(i, j - 1, k)] as f32;
                let dz = matrix[idx(i, j, k + 1)] as f32 - matrix[idx(i, j, k - 1)] as f32;

                gradient[idx(i, j, k)] = [dx / 2.0, dy / 2.0, dz / 2.0];
            }
        }
    }

    gradient
}

/// Compute summary statistics of the 3D matrix.
///
/// Returns `(mean, std, min, max, holographic_fraction, discriminative_fraction)`.
pub fn matrix_stats(matrix: &[u32], total_bits: usize) -> (f64, f64, u32, u32, f64, f64) {
    let n = matrix.len() as f64;
    let target = total_bits as f64 / 2.0;
    let sigma = (total_bits as f64 / 4.0).sqrt();

    let sum: f64 = matrix.iter().map(|&v| v as f64).sum();
    let mean = sum / n;

    let var: f64 = matrix
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = var.sqrt();

    let min = matrix.iter().copied().min().unwrap_or(0);
    let max = matrix.iter().copied().max().unwrap_or(0);

    let holographic_count = matrix
        .iter()
        .filter(|&&v| (v as f64 - target).abs() < sigma)
        .count();
    let discriminative_count = matrix
        .iter()
        .filter(|&&v| (v as f64 - target).abs() > 2.0 * sigma)
        .count();

    let holographic_frac = holographic_count as f64 / n;
    let discriminative_frac = discriminative_count as f64 / n;

    (mean, std, min, max, holographic_frac, discriminative_frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRng(u64);
    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 1
        }
    }

    fn random_fp(seed: u64) -> Fingerprint<256> {
        let mut rng = TestRng::new(seed);
        let mut words = [0u64; 256];
        for w in words.iter_mut() {
            *w = rng.next_u64();
        }
        Fingerprint::from_words(words)
    }

    #[test]
    fn test_binding_popcount_3d_shape() {
        let x = random_fp(1);
        let y = random_fp(2);
        let z = random_fp(3);

        let res = 8;
        let matrix = binding_popcount_3d(&x, &y, &z, res);
        assert_eq!(matrix.len(), res * res * res);
    }

    #[test]
    fn test_random_vectors_centered_at_half() {
        let x = random_fp(10);
        let y = random_fp(20);
        let z = random_fp(30);

        let res = 16;
        let matrix = binding_popcount_3d(&x, &y, &z, res);
        let total_bits = 256 * 64; // 16384

        let mean: f64 = matrix.iter().map(|&v| v as f64).sum::<f64>() / matrix.len() as f64;
        assert!(
            (mean - total_bits as f64 / 2.0).abs() < 200.0,
            "Mean popcount should be near D/2={}, got {}",
            total_bits / 2,
            mean
        );
    }

    #[test]
    fn test_identical_vectors_structure() {
        let v = random_fp(42);
        let res = 8;
        let matrix = binding_popcount_3d(&v, &v, &v, res);

        let diag_val = matrix[0];
        let off_val = matrix[res * res + 2 * res + 3];

        assert!(diag_val > 0);
        assert!(off_val > 0);
    }

    #[test]
    fn test_cross_section() {
        let x = random_fp(1);
        let y = random_fp(2);
        let z = random_fp(3);

        let res = 8;
        let matrix = binding_popcount_3d(&x, &y, &z, res);
        let slice = cross_section_at_z(&matrix, res, 0);

        assert_eq!(slice.len(), res);
        assert_eq!(slice[0].len(), res);
        assert_eq!(slice[0][0], matrix[0]);
    }

    #[test]
    fn test_find_sweet_spots() {
        let x = random_fp(100);
        let y = random_fp(200);
        let z = random_fp(300);

        let res = 8;
        let total_bits = 256 * 64;
        let matrix = binding_popcount_3d(&x, &y, &z, res);

        let sweet_spots = find_holographic_sweet_spot(&matrix, res, total_bits);
        assert!(
            !sweet_spots.is_empty(),
            "Should find holographic sweet spots"
        );

        if sweet_spots.len() > 1 {
            assert!(sweet_spots[0].3 <= sweet_spots[1].3);
        }
    }

    #[test]
    fn test_matrix_stats() {
        let x = random_fp(1);
        let y = random_fp(2);
        let z = random_fp(3);

        let res = 8;
        let total_bits = 256 * 64;
        let matrix = binding_popcount_3d(&x, &y, &z, res);

        let (mean, std, min, max, holo_frac, _disc_frac) = matrix_stats(&matrix, total_bits);
        assert!(mean > 0.0);
        assert!(std > 0.0);
        assert!(min < max);
        assert!(
            holo_frac > 0.05,
            "Expected >5% holographic, got {}",
            holo_frac
        );
    }

    #[test]
    fn test_gradient_field() {
        let x = random_fp(1);
        let y = random_fp(2);
        let z = random_fp(3);

        let res = 8;
        let matrix = binding_popcount_3d(&x, &y, &z, res);
        let gradient = popcount_gradient_3d(&matrix, res);

        assert_eq!(gradient.len(), res * res * res);
        assert_eq!(gradient[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_permute_identity() {
        let fp = random_fp(42);
        let p0 = permute(&fp, 0);
        assert_eq!(fp, p0);
    }

    #[test]
    fn test_permute_full_rotation() {
        let fp = random_fp(42);
        let total_bits = 256 * 64;
        let p_full = permute(&fp, total_bits);
        assert_eq!(fp, p_full);
    }
}
