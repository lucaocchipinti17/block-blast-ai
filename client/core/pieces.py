"""
pieces.py — Block Blast Piece Definitions
==========================================

Every piece is a 5×5 NumPy int8 array, padded with zeros.
Pieces are centered in the 5×5 grid where possible.

Naming convention:
    SHAPE_ORIENTATION
    e.g. L_0, L_90, L_180, L_270

Orientations follow clockwise rotation:
    0   = default / spawn orientation
    90  = rotated 90° clockwise
    180 = rotated 180°
    270 = rotated 270° clockwise (= 90° counter-clockwise)

For symmetric pieces (e.g. O, I when square, S/Z at 180) duplicate
orientations are omitted and a note is left.
"""

import numpy as np
from typing import Dict


def p(rows) -> np.ndarray:
    """Helper: build a 5×5 int8 array from a list of 5 strings of length 5."""
    assert len(rows) == 5 and all(len(r) == 5 for r in rows), \
        "Each piece must be defined as exactly 5 rows of 5 characters."
    return np.array(
        [[1 if c == "X" else 0 for c in row] for row in rows],
        dtype=np.int8,
    )


# ── 1×1 ───────────────────────────────────────────────────────────────────────

DOT = p([
    "00000",
    "00000",
    "00X00",
    "00000",
    "00000",
])

# ── 1×2 / 2×1 ─────────────────────────────────────────────────────────────────

I2_H = p([
    "00000",
    "00000",
    "0XX00",
    "00000",
    "00000",
])

I2_V = p([
    "00000",
    "00X00",
    "00X00",
    "00000",
    "00000",
])

# ── 1×3 / 3×1 ─────────────────────────────────────────────────────────────────

I3_H = p([
    "00000",
    "00000",
    "0XXX0",
    "00000",
    "00000",
])

I3_V = p([
    "00000",
    "00X00",
    "00X00",
    "00X00",
    "00000",
])

# ── 1×4 / 4×1 ─────────────────────────────────────────────────────────────────

I4_H = p([
    "00000",
    "00000",
    "0XXXX",
    "00000",
    "00000",
])

I4_V = p([
    "00X00",
    "00X00",
    "00X00",
    "00X00",
    "00000",
])

# ── 1×5 / 5×1 ─────────────────────────────────────────────────────────────────

I5_H = p([
    "00000",
    "00000",
    "XXXXX",
    "00000",
    "00000",
])

I5_V = p([
    "00X00",
    "00X00",
    "00X00",
    "00X00",
    "00X00",
])

# ── 2×3 / 3×2 ─────────────────────────────────────────────────────────────────

RECT_2x3 = p([
    "00000",
    "00000",
    "0XXX0",
    "0XXX0",
    "00000",
])

RECT_3x2 = p([
    "00000",
    "0XX00",
    "0XX00",
    "0XX00",
    "00000",
])

# ── 2×2 square ────────────────────────────────────────────────────────────────

O = p([
    "00000",
    "00000",
    "0XX00",
    "0XX00",
    "00000",
])

# ── 3×3 square ────────────────────────────────────────────────────────────────

O3 = p([
    "00000",
    "0XXX0",
    "0XXX0",
    "0XXX0",
    "00000",
])

# ── Tetromino S ───────────────────────────────────────────────────────────────
# S_0  : .XX
#         XX.
# S_90 :  X.
#          XX
#           .X  (vertical)

S_0 = p([
    "00000",
    "00000",
    "00XX0",
    "0XX00",
    "00000",
])

S_90 = p([
    "00000",
    "0X000",
    "0XX00",
    "00X00",
    "00000",
])

# ── Tetromino Z ───────────────────────────────────────────────────────────────

Z_0 = p([
    "00000",
    "00000",
    "0XX00",
    "00XX0",
    "00000",
])

Z_90 = p([
    "00000",
    "00X00",
    "0XX00",
    "0X000",
    "00000",
])

# ── Tetromino T ───────────────────────────────────────────────────────────────

T_0 = p([
    "00000",
    "00000",
    "0XXX0",
    "00X00",
    "00000",
])

T_90 = p([
    "00000",
    "00X00",
    "0XX00",
    "00X00",
    "00000",
])

T_180 = p([
    "00000",
    "00X00",
    "0XXX0",
    "00000",
    "00000",
])

T_270 = p([
    "00000",
    "00X00",
    "00XX0",
    "00X00",
    "00000",
])

# ── Triomino corner (2x2 minus one cell) ─────────────────────────────────────
# L3_0 has footprint:
#   10
#   11

L3_0 = p([
    "00000",
    "0X000",
    "0XX00",
    "00000",
    "00000",
])

L3_90 = p([
    "00000",
    "0XX00",
    "0X000",
    "00000",
    "00000",
])

L3_180 = p([
    "00000",
    "00X00",
    "0XX00",
    "00000",
    "00000",
])

L3_270 = p([
    "00000",
    "0XX00",
    "00X00",
    "00000",
    "00000",
])

# ── Tetromino L ───────────────────────────────────────────────────────────────
# L_0:  X.
#        X.
#        XX

L_0 = p([
    "00000",
    "0X000",
    "0X000",
    "0XX00",
    "00000",
])

L_90 = p([
    "00000",
    "00000",
    "0XXX0",
    "0X000",
    "00000",
])

L_180 = p([
    "00000",
    "0XX00",
    "00X00",
    "00X00",
    "00000",
])

L_270 = p([
    "00000",
    "000X0",
    "0XXX0",
    "00000",
    "00000",
])

# ── Tetromino J ───────────────────────────────────────────────────────────────
# Mirror of L.
# J_0:  .X
#         X
#        XX

J_0 = p([
    "00000",
    "00X00",
    "00X00",
    "0XX00",
    "00000",
])

J_90 = p([
    "00000",
    "0X000",
    "0XXX0",
    "00000",
    "00000",
])

J_180 = p([
    "00000",
    "0XX00",
    "0X000",
    "0X000",
    "00000",
])

J_270 = p([
    "00000",
    "00000",
    "0XXX0",
    "000X0",
    "00000",
])

# ── Tetromino I (already covered by I4_H / I4_V above) ───────────────────────
# Aliased here for completeness under tetromino naming.

I4_0   = I4_H
I4_90  = I4_V

# ── Big L (3×3 bounding box) ──────────────────────────────────────────────────
# BIG_L_0:  X..
#            X..
#            XXX

BIG_L_0 = p([
    "00000",
    "0X000",
    "0X000",
    "0XXX0",
    "00000",
])

BIG_L_90 = p([
    "00000",
    "0XXX0",
    "0X000",
    "0X000",
    "00000",
])

BIG_L_180 = p([
    "00000",
    "0XXX0",
    "000X0",
    "000X0",
    "00000",
])

BIG_L_270 = p([
    "00000",
    "000X0",
    "000X0",
    "0XXX0",
    "00000",
])


# ── Catalogue ─────────────────────────────────────────────────────────────────
# A dict of every piece keyed by name — useful for iterating, debugging,
# or building a random piece generator.

ALL_PIECES: Dict[str, np.ndarray] = {
    # Singles / dominoes / straights
    "DOT":      DOT,
    "I2_H":     I2_H,
    "I2_V":     I2_V,
    "I3_H":     I3_H,
    "I3_V":     I3_V,
    "I4_H":     I4_H,
    "I4_V":     I4_V,
    "I5_H":     I5_H,
    "I5_V":     I5_V,
    # Rectangles
    "RECT_2x3": RECT_2x3,
    "RECT_3x2": RECT_3x2,
    # Squares
    "O":        O,
    "O3":       O3,
    # S / Z
    "S_0":      S_0,
    "S_90":     S_90,
    "Z_0":      Z_0,
    "Z_90":     Z_90,
    # T
    "T_0":      T_0,
    "T_90":     T_90,
    "T_180":    T_180,
    "T_270":    T_270,
    # Triomino corner (all orientations)
    "L3_0":     L3_0,
    "L3_90":    L3_90,
    "L3_180":   L3_180,
    "L3_270":   L3_270,
    # L / J (tetromino size)
    "L_0":      L_0,
    "L_90":     L_90,
    "L_180":    L_180,
    "L_270":    L_270,
    "J_0":      J_0,
    "J_90":     J_90,
    "J_180":    J_180,
    "J_270":    J_270,
    # I tetromino aliases
    "I4_0":     I4_0,
    "I4_90":    I4_90,
    # Big L / Big J
    "BIG_L_0":   BIG_L_0,
    "BIG_L_90":  BIG_L_90,
    "BIG_L_180": BIG_L_180,
    "BIG_L_270": BIG_L_270,
}


# ── Debug printer ─────────────────────────────────────────────────────────────

def print_piece(name: str, piece: np.ndarray):
    print(f"{name}:")
    for row in piece:
        print("  " + "".join("█" if c else "·" for c in row))
    print()


if __name__ == "__main__":
    for name, piece in ALL_PIECES.items():
        print_piece(name, piece)
    print(f"Total pieces defined: {len(ALL_PIECES)}")
