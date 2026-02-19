"""
model.py — Block Blast Heuristic Search Agent (Bitwise Edition)
================================================================

Board representation
--------------------
Instead of an 8×8 NumPy array, the search engine internally represents
the board as a single Python integer (64 bits). Each bit corresponds to
one cell:

    bit index = row * 8 + col
    bit 0  → (row=0, col=0)  top-left
    bit 63 → (row=7, col=7)  bottom-right

This makes the three most common operations O(1) bitwise ops:

    overlap check  : (board & piece_mask) == 0
    place piece    : board |= piece_mask
    full row check : (board >> (r*8)) & 0xFF == 0xFF

Pieces are also pre-converted to integer masks once at startup and cached,
so the per-move cost is minimal.

Architecture
------------
BitBoard         — lightweight board state wrapper (int + helpers)
HeuristicAgent   — search + evaluation engine
  ├── best_move()       public API: (piece_idx, row, col)
  ├── _search_perm()    exhaustive or sampled depth-first search
  ├── evaluate_board()  scores a BitBoard state
  └── _count_fits()     bitwise probe-shape fit counter
"""

import random
import numpy as np
from itertools import permutations as iter_permutations
from functools import lru_cache
from typing import Optional

from board import Board


# ── Constants ─────────────────────────────────────────────────────────────────

_FULL_ROW      = 0xFF                    # 8 bits set — a complete row
_FULL_COL_BASE = 0x0101010101010101      # col-0 mask (one bit per row)
_LEFT_MASK     = 0xFEFEFEFEFEFEFEFE         # all cols except col 0 (prevent left-shift wrap)
_RIGHT_MASK    = 0x7F7F7F7F7F7F7F7F         # all cols except col 7 (prevent right-shift wrap)
_BOARD_MASK    = (1 << 64) - 1           # all 64 bits

SAMPLE_THRESHOLD = 50_000
SAMPLE_SIZE      = 100

# ── Default heuristic weights ─────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "big_l":           10.0,
    "sq3x3":           20.0,
    "sq2x2":            5.0,
    "i5":               2.0,
    "i4":               0.8,
    "i3":               0.5,
    "line_clear":      30.0,
    "empty_islands":   -5.0,
    "filled_islands":  -10.0,
    "small_islands":   -20.0,
    "density":          -0.5,
    "rough_edges":      -0.5,
}


# ── BitBoard ──────────────────────────────────────────────────────────────────

class BitBoard:
    """
    Lightweight board state as a 64-bit Python integer.

    Bit layout:  bit = row*8 + col
    Bit set (1) = cell occupied, bit clear (0) = empty.

    Keeps a reference to ROWS/COLS for compatibility but all hot-path
    operations work directly on self.bits.
    """

    __slots__ = ("bits",)

    def __init__(self, bits: int = 0):
        self.bits = bits

    # ── Conversion ────────────────────────────────────────────────────────────

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "BitBoard":
        bits = 0
        for r in range(8):
            for c in range(8):
                if arr[r, c]:
                    bits |= (1 << (r * 8 + c))
        return cls(bits)

    def to_numpy(self) -> np.ndarray:
        arr = np.zeros((8, 8), dtype=np.int8)
        bits = self.bits
        for r in range(8):
            row_byte = (bits >> (r * 8)) & 0xFF
            for c in range(8):
                if row_byte & (1 << c):
                    arr[r, c] = 1
        return arr

    # ── Core ops ──────────────────────────────────────────────────────────────

    def can_place(self, shifted_mask: int) -> bool:
        """True if the pre-shifted piece mask doesn't overlap any filled cell."""
        return (self.bits & shifted_mask) == 0

    def place(self, shifted_mask: int) -> tuple:
        """
        Place a piece and clear completed lines.
        Returns (new_BitBoard, lines_cleared).
        Does NOT mutate self.
        """
        bits = self.bits | shifted_mask
        lines_cleared = 0

        # Clear full rows
        for r in range(8):
            row_bits = (bits >> (r * 8)) & _FULL_ROW
            if row_bits == _FULL_ROW:
                bits &= ~(_FULL_ROW << (r * 8))
                lines_cleared += 1

        # Clear full columns
        for c in range(8):
            col_mask = _FULL_COL_BASE << c
            if (bits & col_mask) == col_mask:
                bits &= ~col_mask
                lines_cleared += 1

        return BitBoard(bits), lines_cleared

    def popcount(self) -> int:
        """Number of filled cells (set bits)."""
        return bin(self.bits).count("1")

    def copy(self) -> "BitBoard":
        return BitBoard(self.bits)

    def __eq__(self, other):
        return self.bits == other.bits

    def __hash__(self):
        return hash(self.bits)


# ── Piece mask cache ──────────────────────────────────────────────────────────

def _build_piece_masks(piece_arr: np.ndarray) -> tuple:
    """
    Convert a 5×5 padded piece array into:
      base_mask : int   — piece mask aligned to (0,0), no padding
      h         : int   — bounding box height
      w         : int   — bounding box width
      shifted   : dict  — {(row, col): shifted_mask} for all valid placements

    The shifted dict is pre-computed so valid_moves and placement during
    search are O(1) dictionary lookups instead of repeated bit-shifting.
    """
    filled_rows, filled_cols = np.where(piece_arr != 0)
    if len(filled_rows) == 0:
        return 0, 0, 0, {}

    min_r = int(filled_rows.min())
    min_c = int(filled_cols.min())

    base_mask = 0
    for r, c in zip(filled_rows.tolist(), filled_cols.tolist()):
        bit = (r - min_r) * 8 + (c - min_c)
        base_mask |= (1 << bit)

    h = int(filled_rows.max() - min_r + 1)
    w = int(filled_cols.max() - min_c + 1)

    # Pre-shift for every valid (row, col) on an 8×8 board
    shifted = {}
    for row in range(8 - h + 1):
        for col in range(8 - w + 1):
            shift = row * 8 + col
            shifted[(row, col)] = base_mask << shift

    return base_mask, h, w, shifted


# Precompute masks for every piece in ALL_PIECES at import time
_PIECE_MASK_CACHE: dict = {}

def get_piece_masks(piece_arr: np.ndarray) -> tuple:
    """Return cached (base_mask, h, w, shifted_dict) for this piece array."""
    key = piece_arr.tobytes()
    if key not in _PIECE_MASK_CACHE:
        _PIECE_MASK_CACHE[key] = _build_piece_masks(piece_arr)
    return _PIECE_MASK_CACHE[key]


# ── Probe masks for evaluate_board ───────────────────────────────────────────
# Pre-built shifted dicts for every probe shape used in heuristic evaluation.

def _probe_masks(shape_2d: list) -> list:
    """Convert a 2D list shape into a list of all shifted masks on 8×8 board."""
    arr = np.array(shape_2d, dtype=np.int8)
    # Pad to 5×5 so get_piece_masks works
    padded = np.zeros((5, 5), dtype=np.int8)
    padded[:arr.shape[0], :arr.shape[1]] = arr
    _, _, _, shifted = get_piece_masks(padded)
    return list(shifted.values())

# Built once at module load
_PROBE_MASKS = {
    "big_l_bl":  _probe_masks([[1,0,0],[1,0,0],[1,1,1]]),
    "big_l_br":  _probe_masks([[0,0,1],[0,0,1],[1,1,1]]),
    "big_l_tl":  _probe_masks([[1,1,1],[1,0,0],[1,0,0]]),
    "big_l_tr":  _probe_masks([[1,1,1],[0,0,1],[0,0,1]]),
    "sq3x3":     _probe_masks([[1,1,1],[1,1,1],[1,1,1]]),
    "sq2x2":     _probe_masks([[1,1],[1,1]]),
    "i5h":       _probe_masks([[1,1,1,1,1]]),
    "i5v":       _probe_masks([[1],[1],[1],[1],[1]]),
    "i4h":       _probe_masks([[1,1,1,1]]),
    "i4v":       _probe_masks([[1],[1],[1],[1]]),
    "i3h":       _probe_masks([[1,1,1]]),
    "i3v":       _probe_masks([[1],[1],[1]]),
}


# ── HeuristicAgent ────────────────────────────────────────────────────────────

class HeuristicAgent:
    """
    Heuristic search agent using a bitwise board representation.

    Usage
    -----
        agent = HeuristicAgent()
        piece_idx, row, col = agent.best_move(board, pieces, used, streak)

    Parameters
    ----------
    weights          : dict — override DEFAULT_WEIGHTS
    sample_threshold : int  — switch to sampling above this leaf estimate
    sample_size      : int  — depth-0 samples per perm when sampling
    verbose          : bool — print search diagnostics
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        sample_threshold: int = SAMPLE_THRESHOLD,
        sample_size: int = SAMPLE_SIZE,
        verbose: bool = False,
    ):
        self.weights          = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.sample_threshold = sample_threshold
        self.sample_size      = sample_size
        self.verbose          = verbose

    # ── Public API ────────────────────────────────────────────────────────────

    def best_move(
        self,
        board: Board,
        pieces: list,
        used: list,
        streak: int = 0,
    ) -> tuple:
        """
        Return (piece_idx, row, col) for the best next placement.

        Converts the Board to a BitBoard once, then runs all search
        internally on integer operations.
        """
        bb = BitBoard.from_numpy(board.board)

        available = [
            (i, p) for i, (p, u) in enumerate(zip(pieces, used)) if not u
        ]
        if not available:
            raise ValueError("No unplaced pieces available.")

        if len(available) == 1:
            idx, piece = available[0]
            row, col = self._best_single(bb, piece, streak)
            return idx, row, col

        all_perms     = list(iter_permutations(available))
        leaf_estimate = self._estimate_leaves(bb, all_perms)
        use_sampling  = leaf_estimate > self.sample_threshold

        if self.verbose:
            mode = "sampling" if use_sampling else "exhaustive"
            print(f"  [agent] perms={len(all_perms)}  ~leaves={leaf_estimate}  mode={mode}")

        best_score      = -1e9
        best_first_move = None

        for perm in all_perms:
            paths = self._search_perm(bb, perm, use_sampling)
            for path in paths:
                score = self._score_path(path, streak)
                if score > best_score:
                    best_score      = score
                    first           = path[0]
                    best_first_move = (perm[0][0], first[1], first[2])

        if best_first_move is None:
            idx, piece = available[0]
            _, _, _, shifted = get_piece_masks(piece)
            for (row, col), mask in shifted.items():
                if bb.can_place(mask):
                    return idx, row, col
            raise ValueError("No valid moves found.")

        if self.verbose:
            print(f"  [agent] best_score={best_score:.1f}  move={best_first_move}")

        return best_first_move

    # ── Search ────────────────────────────────────────────────────────────────

    def _search_perm(self, bb: BitBoard, perm: list, use_sampling: bool) -> list:
        paths = []
        self._recurse(bb, perm, 0, [], paths, use_sampling)
        return paths

    def _recurse(
        self,
        bb: BitBoard,
        perm: list,
        depth: int,
        path_so_far: list,
        results: list,
        use_sampling: bool,
    ):
        if depth == len(perm):
            results.append(path_so_far + [bb])
            return

        idx, piece = perm[depth]
        _, _, _, shifted = get_piece_masks(piece)

        # All valid placements via bitwise overlap check
        valid = [
            (row, col) for (row, col), mask in shifted.items()
            if bb.can_place(mask)
        ]

        if not valid:
            return

        if use_sampling and depth == 0:
            valid = random.sample(valid, min(self.sample_size, len(valid)))

        for row, col in valid:
            mask          = shifted[(row, col)]
            new_bb, lines = bb.place(mask)
            self._recurse(
                new_bb,
                perm,
                depth + 1,
                path_so_far + [(idx, row, col, lines)],
                results,
                use_sampling,
            )

    def _best_single(self, bb: BitBoard, piece: np.ndarray, streak: int) -> tuple:
        _, _, _, shifted = get_piece_masks(piece)
        best_score = -1e9
        best_rc    = None

        for (row, col), mask in shifted.items():
            if not bb.can_place(mask):
                continue
            new_bb, lines = bb.place(mask)
            score = self.evaluate_board(new_bb, lines_cleared=lines, streak=streak)
            if score > best_score:
                best_score = score
                best_rc    = (row, col)

        return best_rc or (0, 0)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_path(self, path: list, streak: int) -> float:
        final_bb     = path[-1]
        total_clears = sum(step[3] for step in path[:-1])
        return self.evaluate_board(final_bb, lines_cleared=total_clears, streak=streak)

    def evaluate_board(
        self,
        bb: BitBoard,
        lines_cleared: int = 0,
        streak: int = 0,
    ) -> float:
        """
        Score a BitBoard state. Higher = better.

        All probe-fit counting is done via bitwise AND against pre-built
        masks — no inner Python loops over the board.
        """
        W     = self.weights
        bits  = bb.bits
        score = 0.0

        # ── Piece-fit flexibility ─────────────────────────────────────────────
        # A probe "fits" at a position if (board & probe_mask) == 0
        # i.e. all probe cells are empty on the board.

        for m in _PROBE_MASKS["big_l_bl"]: score += (bits & m) == 0
        for m in _PROBE_MASKS["big_l_br"]: score += (bits & m) == 0
        for m in _PROBE_MASKS["big_l_tl"]: score += (bits & m) == 0
        for m in _PROBE_MASKS["big_l_tr"]: score += (bits & m) == 0
        score *= W["big_l"] / 4   # already summed all 4 orientations

        sq3 = sum((bits & m) == 0 for m in _PROBE_MASKS["sq3x3"])
        sq2 = sum((bits & m) == 0 for m in _PROBE_MASKS["sq2x2"])
        i5  = sum((bits & m) == 0 for m in _PROBE_MASKS["i5h"]) + \
              sum((bits & m) == 0 for m in _PROBE_MASKS["i5v"])
        i4  = sum((bits & m) == 0 for m in _PROBE_MASKS["i4h"]) + \
              sum((bits & m) == 0 for m in _PROBE_MASKS["i4v"])
        i3  = sum((bits & m) == 0 for m in _PROBE_MASKS["i3h"]) + \
              sum((bits & m) == 0 for m in _PROBE_MASKS["i3v"])

        score += sq3 * W["sq3x3"]
        score += sq2 * W["sq2x2"]
        score += i5  * W["i5"]
        score += i4  * W["i4"]
        score += i3  * W["i3"]

        # ── Line clear reward ─────────────────────────────────────────────────
        score += lines_cleared * W["line_clear"]

        # ── Fragmentation penalties ───────────────────────────────────────────
        board_arr      = bb.to_numpy()
        empty_islands  = self._count_islands(board_arr, value=0)
        filled_islands = self._count_islands(board_arr, value=1)

        score += len(empty_islands)  * W["empty_islands"]
        score += len(filled_islands) * W["filled_islands"]
        score += len([s for s in empty_islands  if s <= 3]) * W["small_islands"]
        score += len([s for s in filled_islands if s <= 3]) * W["small_islands"]

        # ── Density and roughness ─────────────────────────────────────────────
        score += bb.popcount() * W["density"]
        score += self._rough_edges_bitwise(bits) * W["rough_edges"]

        return score

    # ── Board analysis ────────────────────────────────────────────────────────

    @staticmethod
    def _rough_edges_bitwise(bits: int) -> int:
        """
        Count filled cells that border at least one empty cell.
        Uses bitwise shifts instead of nested Python loops.

        Shift the board in each of 4 directions and XOR against
        the original to find boundary cells.
        """
        # Neighbour masks (avoid wrapping at row/col edges)
        LEFT_MASK  = 0xFEFEFEFEFEFEFEFE   # all cols except col 0
        RIGHT_MASK = 0x7F7F7F7F7F7F7F7F   # all cols except col 7

        up    = bits >> 8
        down  = bits << 8
        left  = (bits >> 1) & LEFT_MASK
        right = (bits << 1) & RIGHT_MASK

        # Cells that have at least one empty neighbour
        has_empty_neighbour = bits & ~(up & down & left & right) & _BOARD_MASK
        return bin(has_empty_neighbour).count("1")

    @staticmethod
    def _flood_expand(seed: int, allowed: int) -> int:
        """
        Expand seed bits to all connected cells within `allowed` using bitwise shifts.
        Up/down = shift by 8 bits (one row), left/right = shift by 1 bit with edge masks.
        Iterates until stable (no new cells added).
        """
        while True:
            prev = seed
            seed = (seed
                    | (seed >> 8)
                    | ((seed << 8) & _BOARD_MASK)
                    | ((seed >> 1) & _LEFT_MASK)
                    | ((seed << 1) & _RIGHT_MASK)
                    ) & allowed
            if seed == prev:
                break
        return seed

    @staticmethod
    def _count_islands(board: np.ndarray, value: int) -> list:
        """
        Bitwise flood-fill to find connected regions of `value`.
        Returns list of region sizes < 8 (large open areas are not a concern).

        Works on the board as a 64-bit integer: each region is found by
        isolating the lowest set bit as a seed, flood-expanding it within
        the target mask, recording the size, then clearing that region and
        repeating until no cells remain.
        """
        # Build target bitmask from the numpy array
        target = 0
        for r in range(8):
            for c in range(8):
                if board[r, c] == value:
                    target |= (1 << (r * 8 + c))

        remaining = target
        sizes     = []

        while remaining:
            seed   = remaining & (-remaining)   # isolate lowest set bit
            region = HeuristicAgent._flood_expand(seed, target)
            size   = bin(region).count("1")
            remaining &= ~region
            if size < 8:
                sizes.append(size)

        return sizes

    @staticmethod
    def _estimate_leaves(bb: BitBoard, all_perms: list) -> int:
        total = 0
        for perm in all_perms:
            _, _, _, shifted = get_piece_masks(perm[0][1])
            n = sum(1 for mask in shifted.values() if bb.can_place(mask))
            total += n * max(1, n // 2) * max(1, n // 4)
        return total