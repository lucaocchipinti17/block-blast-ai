"""
board.py — Block Blast MVP Game Engine
=======================================

Coordinate Convention
---------------------
A piece is stored as a 5×5 NumPy array with possible padding. Internally we
extract the "effective footprint" — the minimal bounding box of filled cells —
and work with that.

(row, col) in valid_moves / apply_move refers to where the TOP-LEFT corner of
the effective bounding box lands on the 8×8 board.

Example — L-piece (5×5 padded):
    [[0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 1, 0, 0, 0],
     [0, 1, 1, 0, 0],
     [0, 0, 0, 0, 0]]

Effective bounding box (rows 1-3, cols 1-2):
    [[1, 0],
     [1, 0],
     [1, 1]]

Filled cells relative to bounding box top-left: (0,0), (1,0), (2,0), (2,1)

If placed at board position (row=3, col=4), those cells land at:
    (3,4), (4,4), (5,4), (5,5)
"""

import numpy as np
from typing import List, Tuple, Optional


class Board:
    """
    8x8 Block Blast board.

    The board uses integer values so pieces can be distinguished by color/id
    in future (nonzero = occupied, 0 = empty).
    """

    ROWS = 8
    COLS = 8

    def __init__(self, initial_state: Optional[np.ndarray] = None):
        if initial_state is not None:
            assert initial_state.shape == (self.ROWS, self.COLS)
            self.board = initial_state.astype(np.int8)
        else:
            self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)

    # ── Piece utilities ───────────────────────────────────────────────────────

    @staticmethod
    def get_footprint(piece: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Strip padding from a 5x5 piece array.

        Returns
        -------
        cells : np.ndarray, shape (N, 2)
            Filled cell offsets (row, col) relative to the bounding box top-left.
        bbox_rows : int
            Height of the effective bounding box.
        bbox_cols : int
            Width of the effective bounding box.
        """
        filled_rows, filled_cols = np.where(piece != 0)

        if len(filled_rows) == 0:
            # Empty piece — no filled cells
            return np.empty((0, 2), dtype=int), 0, 0

        min_r, max_r = filled_rows.min(), filled_rows.max()
        min_c, max_c = filled_cols.min(), filled_cols.max()

        # Shift offsets so they're relative to bounding box origin
        cells = np.column_stack([filled_rows - min_r, filled_cols - min_c])
        bbox_rows = int(max_r - min_r + 1)
        bbox_cols = int(max_c - min_c + 1)

        return cells, bbox_rows, bbox_cols

    # ── Core API ──────────────────────────────────────────────────────────────

    def valid_moves(self, piece: np.ndarray) -> List[Tuple[int, int]]:
        """
        Return all (row, col) positions where the piece can be placed.

        (row, col) is the top-left corner of the piece's effective bounding box
        on the 8x8 board.

        A placement is valid iff:
          1. Every filled cell lands within [0, ROWS) x [0, COLS)
          2. No filled cell overlaps an occupied board cell

        Parameters
        ----------
        piece : np.ndarray, shape (5, 5)
            Padded piece definition (0 = empty, nonzero = filled).

        Returns
        -------
        List[Tuple[int, int]]
            Sorted list of valid (row, col) placements.
        """
        cells, bbox_rows, bbox_cols = self.get_footprint(piece)

        if len(cells) == 0:
            return []

        moves = []
        max_row = self.ROWS - bbox_rows
        max_col = self.COLS - bbox_cols

        for row in range(max_row + 1):
            for col in range(max_col + 1):
                # Absolute board positions of each filled cell
                board_positions = cells + np.array([row, col])
                rs, cs = board_positions[:, 0], board_positions[:, 1]

                # Check for overlaps using vectorized indexing
                if not np.any(self.board[rs, cs]):
                    moves.append((row, col))

        return moves

    def apply_move(
        self,
        piece: np.ndarray,
        row: int,
        col: int,
        piece_id: int = 1,
    ) -> int:
        """
        Place the piece on the board in-place and clear any completed lines.

        Parameters
        ----------
        piece : np.ndarray, shape (5, 5)
        row, col : int
            Top-left of the effective bounding box on the board.
        piece_id : int
            Value written to filled cells (default 1; use color IDs for multi-piece).

        Returns
        -------
        int
            Number of lines cleared.

        Raises
        ------
        ValueError
            If (row, col) is not a valid placement for this piece.
        """
        if (row, col) not in self.valid_moves(piece):
            raise ValueError(
                f"({row}, {col}) is not a valid placement for this piece."
            )

        # Place the piece
        cells, _, _ = self.get_footprint(piece)
        board_positions = cells + np.array([row, col])
        rs, cs = board_positions[:, 0], board_positions[:, 1]
        self.board[rs, cs] = piece_id

        # Clear completed rows and columns
        return self._clear_lines()

    def _clear_lines(self) -> int:
        """
        Clear any fully occupied rows and columns (Block Blast rules).
        Both are cleared simultaneously — a cell at the intersection of a full
        row AND full column is still only cleared once.

        Returns the number of lines cleared.
        """
        full_rows = np.where(self.board.all(axis=1))[0]
        full_cols = np.where(self.board.all(axis=0))[0]

        self.board[full_rows, :] = 0
        self.board[:, full_cols] = 0

        return len(full_rows) + len(full_cols)

    # ── Convenience ───────────────────────────────────────────────────────────

    def copy(self) -> "Board":
        return Board(self.board.copy())

    def is_game_over(self, pieces: List[np.ndarray]) -> bool:
        """Return True if none of the given pieces have any valid placement."""
        return all(len(self.valid_moves(p)) == 0 for p in pieces)

    def __repr__(self) -> str:
        rows = []
        rows.append("┌" + "─" * self.COLS + "┐")
        for row in self.board:
            rows.append("│" + "".join("█" if c else "·" for c in row) + "│")
        rows.append("└" + "─" * self.COLS + "┘")
        return "\n".join(rows)


# ── Example ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # L-shaped piece in a 5×5 padded array
    L_PIECE = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int8)

    # 1×3 horizontal piece
    H3_PIECE = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int8)

    board = Board()
    while True:
        print(board)
        p = input("Select next piece, 0 for L, 1 for H3")
        piece = None
        if (p == "0"):
            piece = L_PIECE
        else:
            piece = H3_PIECE
        valid_move = False
        row, col = None, None
        while (not valid_move):
            row = input("select top left row (1-indexed)")
            col = input("select top left col (1-indexed)")
            row = int(row) - 1
            col = int(col) - 1
            try:
                board.apply_move(piece, row, col)
                valid_move = True
            except ValueError:
                print("Invalid index")
        if board.is_game_over([L_PIECE, H3_PIECE]):
            print("Game Over")
            break
                
        
        
        
            

    # # ── Show footprint extraction ─────────────────────────────────────────────
    # cells, bbox_rows, bbox_cols = Board.get_footprint(L_PIECE)
    # print("L-piece footprint:")
    # print(f"  Bounding box : {bbox_rows}×{bbox_cols}")
    # print(f"  Filled cells : {cells.tolist()}")
    # print()

    # # ── Valid moves ───────────────────────────────────────────────────────────
    # moves = board.valid_moves(L_PIECE)
    # print(f"Valid placements for L-piece on empty board: {len(moves)}")
    # print(f"  First 5: {moves[:5]}")
    # print(f"  Last  5: {moves[-5:]}")
    # print()

    # # ── Apply a placement ─────────────────────────────────────────────────────
    # board2, cleared = board.apply_move(L_PIECE, row=0, col=0)
    # print(f"After placing L-piece at (0,0) — lines cleared: {cleared}")
    # print(board2)
    # print()

    # # ── Set up a scenario where a line will clear ─────────────────────────────
    # # Fill row 7 manually except col 0, then place a 1×1 to complete it
    # board3 = Board()
    # board3.board[7, 1:] = 1   # row 7 cols 1-7 filled

    # ONE_PIECE = np.array([
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    # ], dtype=np.int8)

    # print("Board before completing row 7:")
    # print(board3)
    # board4, cleared = board3.apply_move(ONE_PIECE, row=7, col=0)
    # print(f"\nAfter placing 1×1 at (7,0) — lines cleared: {cleared}")
    # print(board4)

    # # ── Game over check ───────────────────────────────────────────────────────
    # full_board = Board(np.ones((8, 8), dtype=np.int8))
    # print(f"\nGame over on full board: {full_board.is_game_over([L_PIECE, H3_PIECE])}")
    # print(f"Game over on empty board: {board.is_game_over([L_PIECE, H3_PIECE])}")