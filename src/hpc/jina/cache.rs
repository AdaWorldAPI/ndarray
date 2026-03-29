//! Binary cache for Jina Base17 + palette — load once, use forever.
//!
//! Format: `[n_tokens:u32][base_dim:u32][i16[17] × n_tokens]`
//! Palette: `[n_tokens:u32][base_dim:u32][K:u32][centroids: i16[17]×K][assignments: u8×n]`

use super::codec::{Base17Token, JinaPalette, BASE_DIM, PALETTE_K};
use std::io::{Read, Write};

/// Save Base17 tokens to binary cache.
pub fn save_base17_cache<W: Write>(tokens: &[Base17Token], writer: &mut W) -> Result<(), String> {
    let n = tokens.len() as u32;
    writer.write_all(&n.to_le_bytes()).map_err(|e| e.to_string())?;
    writer
        .write_all(&(BASE_DIM as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    for t in tokens {
        for &d in &t.dims {
            writer.write_all(&d.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

/// Load Base17 tokens from binary cache.
pub fn load_base17_cache<R: Read>(reader: &mut R) -> Result<Vec<Base17Token>, String> {
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4).map_err(|e| e.to_string())?;
    let n = u32::from_le_bytes(buf4) as usize;
    reader.read_exact(&mut buf4).map_err(|e| e.to_string())?;
    let bd = u32::from_le_bytes(buf4) as usize;
    if bd != BASE_DIM {
        return Err(format!("Base dim mismatch: {} vs {}", bd, BASE_DIM));
    }

    let mut tokens = Vec::with_capacity(n);
    let mut buf2 = [0u8; 2];
    for _ in 0..n {
        let mut dims = [0i16; BASE_DIM];
        for d in 0..BASE_DIM {
            reader.read_exact(&mut buf2).map_err(|e| e.to_string())?;
            dims[d] = i16::from_le_bytes(buf2);
        }
        tokens.push(Base17Token { dims });
    }
    Ok(tokens)
}

/// Save palette to binary cache.
pub fn save_palette_cache<W: Write>(palette: &JinaPalette, writer: &mut W) -> Result<(), String> {
    let n = palette.assignments.len() as u32;
    writer.write_all(&n.to_le_bytes()).map_err(|e| e.to_string())?;
    writer
        .write_all(&(BASE_DIM as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    writer
        .write_all(&(PALETTE_K as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;

    // Centroids
    for k in 0..PALETTE_K {
        for &d in &palette.centroids[k].dims {
            writer.write_all(&d.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    // Assignments
    writer
        .write_all(&palette.assignments)
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Load palette from binary cache.
pub fn load_palette_cache<R: Read>(reader: &mut R) -> Result<JinaPalette, String> {
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4).map_err(|e| e.to_string())?;
    let n = u32::from_le_bytes(buf4) as usize;
    reader.read_exact(&mut buf4).map_err(|e| e.to_string())?;
    let bd = u32::from_le_bytes(buf4) as usize;
    reader.read_exact(&mut buf4).map_err(|e| e.to_string())?;
    let k = u32::from_le_bytes(buf4) as usize;

    if bd != BASE_DIM {
        return Err(format!("Base dim mismatch: {bd} vs {BASE_DIM}"));
    }
    if k != PALETTE_K {
        return Err(format!("Palette K mismatch: {k} vs {PALETTE_K}"));
    }

    let mut centroids = [Base17Token { dims: [0; BASE_DIM] }; PALETTE_K];
    let mut buf2 = [0u8; 2];
    for ki in 0..PALETTE_K {
        for d in 0..BASE_DIM {
            reader.read_exact(&mut buf2).map_err(|e| e.to_string())?;
            centroids[ki].dims[d] = i16::from_le_bytes(buf2);
        }
    }

    let mut assignments = vec![0u8; n];
    reader
        .read_exact(&mut assignments)
        .map_err(|e| e.to_string())?;

    // Rebuild distance table
    let mut distance_table = [[0u16; PALETTE_K]; PALETTE_K];
    for i in 0..PALETTE_K {
        for j in i..PALETTE_K {
            let d = centroids[i].l1(&centroids[j]).min(u16::MAX as u32) as u16;
            distance_table[i][j] = d;
            distance_table[j][i] = d;
        }
    }

    Ok(JinaPalette {
        centroids,
        assignments,
        distance_table,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_base17_cache_roundtrip() {
        let tokens = vec![
            Base17Token {
                dims: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            },
            Base17Token {
                dims: [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17],
            },
        ];
        let mut buf = Vec::new();
        save_base17_cache(&tokens, &mut buf).unwrap();
        let loaded = load_base17_cache(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].dims, tokens[0].dims);
        assert_eq!(loaded[1].dims, tokens[1].dims);
    }
}
