"""
game.py — Block Blast Command Line Game
========================================
Run with: python game.py
"""

import os
import random
import numpy as np
from typing import Optional

from board import Board
from pieces import ALL_PIECES


# ── Scoring constants ─────────────────────────────────────────────────────────

POINTS_PER_CELL = 1   # points per cell placed
POINTS_PER_LINE = 10  # base points per line cleared, multiplied by current streak


# ── Terminal display helpers ──────────────────────────────────────────────────

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def render_piece(piece: np.ndarray, label: str, empty: bool = False) -> list[str]:
    """
    Render a single piece (or a blank placeholder) as a list of strings.
    The piece is drawn within a fixed 5-row display using its actual 5×5 array.
    """
    lines = [f" {label} "]
    if empty:
        for _ in range(5):
            lines.append("  ·····  ")
    else:
        for row in piece:
            lines.append("  " + "".join("█" if c else "·" for c in row) + "  ")
    return lines


def render_game(
    board: Board,
    score: int,
    streak: int,
    piece_bank: list[Optional[np.ndarray]],
    used: list[bool],
):
    """
    Render the full game state to the terminal.
    Layout:
        Score + streak header
        8×8 board
        Separator
        3 pieces side by side (blanked out if already placed)
    """
    clear_screen()

    # ── Header ────────────────────────────────────────────────────────────────
    streak_str = f"  🔥 Streak x{streak}" if streak > 1 else ""
    print(f"  SCORE: {score}{streak_str}")
    print()

    # ── Board ─────────────────────────────────────────────────────────────────
    print("  ┌────────┐")
    for row in board.board:
        print("  │" + "".join("█" if c else "·" for c in row) + "│")
    print("  └────────┘")
    print()

    # ── Piece bank (side by side) ─────────────────────────────────────────────
    piece_cols = []
    for i, (piece, is_used) in enumerate(zip(piece_bank, used)):
        label = f"[{i+1}] PLACED" if is_used else f"[{i+1}]"
        piece_cols.append(render_piece(piece, label, empty=is_used))

    # Print all 3 pieces on the same rows
    for row_idx in range(6):  # label line + 5 piece rows
        print("  ".join(col[row_idx] for col in piece_cols))
    print()


# ── Score calculation ─────────────────────────────────────────────────────────

def calculate_score(
    cells_placed: int,
    lines_cleared: int,
    streak: int,
) -> tuple[int, int]:
    """
    Calculate points earned for a single placement and return
    (points_earned, new_streak).

    Scoring rules:
      - 1 point per cell placed, regardless of lines cleared
      - line clear bonus = lines_cleared × streak × 10
      - streak increments by 1 each turn a line is cleared
      - streak resets to 0 only when a full round of 3 pieces
        passes without any line clear (handled in game loop)
    """
    points = cells_placed * POINTS_PER_CELL

    if lines_cleared > 0:
        new_streak = streak + 1
        points += lines_cleared * new_streak * POINTS_PER_LINE
    else:
        new_streak = streak  # streak does not reset mid-round

    return points, new_streak


# ── Input helpers ─────────────────────────────────────────────────────────────

def prompt_piece_choice(available_indices: list[int]) -> int:
    """Ask the player which piece to place. Returns 0-indexed piece number."""
    options = [str(i + 1) for i in available_indices]
    while True:
        raw = input(f"  Choose a piece ({'/'.join(options)}): ").strip()
        if raw in options:
            return int(raw) - 1
        print(f"  Invalid choice. Enter one of: {', '.join(options)}")


def prompt_placement(board: Board, piece: np.ndarray) -> tuple[int, int]:
    """
    Ask the player for a (row, col) placement.
    Shows valid moves and keeps prompting until a valid one is entered.
    """
    valid = board.valid_moves(piece)
    print(f"  Valid placements (row col): {valid}")
    while True:
        raw = input("  Enter placement as 'row col' (0-indexed): ").strip()
        parts = raw.split()
        if len(parts) == 2:
            try:
                row, col = int(parts[0]), int(parts[1])
                if (row, col) in valid:
                    return row, col
                else:
                    print("  That position is not valid for this piece.")
            except ValueError:
                pass
        print("  Please enter two integers, e.g. '3 4'")


# ── Main game class ───────────────────────────────────────────────────────────

class BlockBlastGame:
    """
    Command-line Block Blast game.

    State
    -----
    board      : Board       — current 8×8 board
    score      : int         — cumulative score
    streak     : int         — consecutive turns with ≥1 line clear
    piece_bank : list        — current 3 pieces (None if placed)
    used       : list[bool]  — which pieces in the bank have been placed
    turn       : int         — turn counter
    """

    def __init__(self, piece_pool: Optional[dict] = None):
        self.board      = Board()
        self.score      = 0
        self.streak     = 0
        self.turn       = 0
        self.piece_pool = piece_pool or ALL_PIECES  # dict of name -> array
        self.piece_bank : list[Optional[np.ndarray]] = []
        self.used       : list[bool] = []
        self._round_had_clear : bool = False  # did any placement this round clear a line?

    # ── Piece bank ────────────────────────────────────────────────────────────

    def _draw_piece_bank(self):
        """Randomly select 3 pieces from the pool for the new bank."""
        names   = random.choices(list(self.piece_pool.keys()), k=3)
        self.piece_bank = [self.piece_pool[n].copy() for n in names]
        self.used       = [False, False, False]

    def _bank_exhausted(self) -> bool:
        """True if all 3 pieces in the current bank have been placed."""
        return all(self.used)

    def _any_valid_placement(self) -> bool:
        """
        Return True if at least one unplaced piece has at least one valid move.
        If False, the game is over.
        """
        for piece, is_used in zip(self.piece_bank, self.used):
            if not is_used and len(self.board.valid_moves(piece)) > 0:
                return True
        return False

    def _available_indices(self) -> list[int]:
        """Return indices of pieces that are unplaced AND have valid moves."""
        return [
            i for i, (piece, is_used) in enumerate(zip(self.piece_bank, self.used))
            if not is_used and len(self.board.valid_moves(piece)) > 0
        ]

    # ── Core turn logic ───────────────────────────────────────────────────────

    def _place_piece(self, piece_idx: int, row: int, col: int):
        """Place a piece, update score and streak."""
        piece = self.piece_bank[piece_idx]

        # Count cells being placed (for score)
        cells, _, _ = Board.get_footprint(piece)
        cells_placed = len(cells)

        # Apply move (mutates board, returns lines cleared)
        lines_cleared = self.board.apply_move(piece, row, col)

        # Update score and streak
        points, self.streak = calculate_score(cells_placed, lines_cleared, self.streak)
        self.score += points

        # Track whether any line was cleared this round
        if lines_cleared > 0:
            self._round_had_clear = True

        # Mark piece as used
        self.used[piece_idx] = True
        self.piece_bank[piece_idx] = piece  # keep for display (shown as blank)

        self.turn += 1

        return lines_cleared, points

    # ── Game loop ─────────────────────────────────────────────────────────────

    def run(self):
        """Main game loop."""
        print("\n  Welcome to Block Blast!\n")
        input("  Press Enter to start...")

        while True:
            # Draw a new bank when the previous one is exhausted
            if self._bank_exhausted() or self.turn == 0:
                # Reset streak if the completed round had no line clears
                if self.turn > 0 and not self._round_had_clear:
                    self.streak = 0
                self._round_had_clear = False
                self._draw_piece_bank()

            # Check if any piece can be placed — if not, game over
            if not self._any_valid_placement():
                self._game_over()
                return

            # Render current state
            render_game(self.board, self.score, self.streak, self.piece_bank, self.used)

            # Player picks a piece
            available = self._available_indices()
            piece_idx = prompt_piece_choice(available)

            # Re-render with chosen piece highlighted (just re-render for now)
            render_game(self.board, self.score, self.streak, self.piece_bank, self.used)

            # Player picks a position
            row, col = prompt_placement(self.board, self.piece_bank[piece_idx])

            # Place it
            lines_cleared, points = self._place_piece(piece_idx, row, col)

            # Brief feedback
            render_game(self.board, self.score, self.streak, self.piece_bank, self.used)
            if lines_cleared > 0:
                combo_str = f" COMBO x{lines_cleared}!" if lines_cleared > 1 else ""
                streak_str = f" (Streak: {self.streak})" if self.streak > 1 else ""
                print(f"  ✓ +{points} points — {lines_cleared} line(s) cleared!{combo_str}{streak_str}")
            else:
                print(f"  ✓ +{points} points")
            print()

            # Check for game over after placement
            if not self._any_valid_placement() and not self._bank_exhausted():
                self._game_over()
                return

            input("  Press Enter to continue...")

    def _game_over(self):
        clear_screen()
        print()
        print("  ╔══════════════════╗")
        print("  ║    GAME  OVER    ║")
        print("  ╚══════════════════╝")
        print()
        print(f"  Final Score : {self.score}")
        print(f"  Turns played: {self.turn}")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    game = BlockBlastGame()
    game.run()
