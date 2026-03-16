# Cognitive Layer Constants

```
PLANE_BITS         = 16384       # i8 accumulator positions
PLANE_BYTES        = 2048        # fingerprint view (256 × u64)
CONTAINER_BYTES    = 16384       # full Plane acc (i8 × 16384)
COGRECORD_BYTES    = 65536       # 4 × Plane (4-channel)
FINGERPRINT_WORDS  = 256         # u64 words in Fingerprint<256>
MERKLE_BYTES       = 6           # 48-bit blake3 truncation

# Cascade strokes (non-overlapping Tetris layout)
STROKE1_BYTES      = 128         # coarse (1/16 sample)
STROKE2_BYTES      = 384         # medium (128..512)
STROKE3_BYTES      = 1536        # precise (512..2048)

# BF16 field widths
BF16_SIGN_BITS     = 1           # bit 15
BF16_EXPONENT_BITS = 8           # bits 7-14
BF16_MANTISSA_BITS = 7           # bits 0-6

# BF16 default weights
BF16_SIGN_WEIGHT   = 256
BF16_EXP_WEIGHT    = 16
BF16_MAN_WEIGHT    = 1

# PackedQualia
QUALIA_DIMS        = 16          # resonance dimensions
PACKED_QUALIA_BYTES = 18         # 16 × i8 + 2 × u8 (BF16 scalar)

# Cascade bands (sigma-based)
BAND_FOVEAL_RATIO  = 0.25       # top 5%
BAND_NEAR_RATIO    = 0.50       # 5-25%
BAND_GOOD_RATIO    = 0.75       # 25-60%
BAND_WEAK_RATIO    = 1.00       # 60-90%

# Node/Mask
NODE_PLANES        = 3           # Subject, Predicate, Object
MASK_PROJECTIONS   = 8           # 2^3 (including null mask)
NON_NULL_MASKS     = 7           # S__, _P_, __O, SP_, S_O, _PO, SPO

# Causality dimensions (indices into 16-dim qualia vector)
CAUSALITY_WARMTH   = 4
CAUSALITY_SOCIAL   = 6
CAUSALITY_SACRED   = 8
```
