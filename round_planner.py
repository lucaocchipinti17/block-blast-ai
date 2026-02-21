"""
round_planner.py
================

Interactive helper to plan Block Blast placements for a 3-piece bank.

What it does each cycle:
1) You input 3 pieces (by menu number, exact name, or shape pattern)
2) Agent computes the best order + placements
3) Script prints the 1-indexed placement index for each step
4) Script applies the moves to the board and repeats

Run:
    python round_planner.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
from typing import Dict, List, Tuple

import numpy as np

from board import Board
from game import calculate_score
from model import HeuristicAgent
from pieces import ALL_PIECES


@dataclass(frozen=True)
class PieceMeta:
    name: str
    piece: np.ndarray
    key: str
    height: int
    width: int
    cells: int


def piece_footprint(piece: np.ndarray) -> np.ndarray:
    cells, h, w = Board.get_footprint(piece)
    out = np.zeros((h, w), dtype=np.int8)
    for r, c in cells:
        out[r, c] = 1
    return out


def piece_key(piece: np.ndarray) -> str:
    fp = piece_footprint(piece)
    return "/".join("".join("1" if c else "0" for c in row) for row in fp)


def build_piece_catalog() -> tuple[List[PieceMeta], Dict[str, List[PieceMeta]]]:
    metas: List[PieceMeta] = []
    by_key: Dict[str, List[PieceMeta]] = {}

    for name, piece in sorted(ALL_PIECES.items()):
        fp = piece_footprint(piece)
        key = piece_key(piece)
        meta = PieceMeta(
            name=name,
            piece=piece,
            key=key,
            height=fp.shape[0],
            width=fp.shape[1],
            cells=int(fp.sum()),
        )
        metas.append(meta)
        by_key.setdefault(key, []).append(meta)

    # Keep deterministic, user-friendly canonical ordering for same-shape aliases.
    # Example: prefer I4_H over I4_0 for the horizontal 4-line footprint.
    def alias_rank(meta: PieceMeta) -> tuple[int, str]:
        if re.match(r"^I\d+_(0|90|180|270)$", meta.name):
            return (1, meta.name)
        return (0, meta.name)

    for key in by_key:
        by_key[key].sort(key=alias_rank)

    return metas, by_key


def parse_pattern(raw: str) -> str | None:
    """
    Accept patterns like:
      111/100/100
      ###/#../#..
      X.. / XXX
    Returns canonical key string or None if invalid.
    """
    cleaned = raw.strip().replace(" ", "")
    if not cleaned:
        return None

    rows = cleaned.split("/")
    if not rows:
        return None

    norm_rows = []
    width = None
    for row in rows:
        row = row.replace("X", "1").replace("#", "1").replace(".", "0")
        if any(ch not in {"0", "1"} for ch in row):
            return None
        if width is None:
            width = len(row)
        if len(row) != width:
            return None
        norm_rows.append(row)

    return "/".join(norm_rows)


def choose_piece(
    piece_no: int,
    by_key: Dict[str, List[PieceMeta]],
    verbose: bool = False,
) -> PieceMeta:
    print(f"\nPiece {piece_no} pattern:")

    while True:
        raw = input("> ").strip()
        if not raw:
            continue

        pat = parse_pattern(raw)
        if pat is None:
            print("Invalid pattern. Use 0/1, #/., or X with rows split by '/'.")
            continue

        matches = by_key.get(pat, [])
        if not matches:
            print(f"No piece matches pattern key '{pat}'.")
            continue
        chosen = matches[0]
        if len(matches) > 1 and verbose:
            aliases = ", ".join(m.name for m in matches[1:])
            print(f"Using {chosen.name} for shape {pat} (aliases: {aliases})")
        return chosen


def prompt_initial_board() -> Board:
    print("Initial board setup:")
    print("  Press Enter for an empty board")
    print("  Or enter 8 rows using 0/1 or ./# (8 chars each)")
    print("  Example row: 00111000")

    first = input("row1 (or Enter for empty): ").strip()
    if not first:
        return Board()

    rows = [first]
    for i in range(2, 9):
        rows.append(input(f"row{i}: ").strip())

    arr = np.zeros((8, 8), dtype=np.int8)
    for r, row in enumerate(rows):
        row = row.replace("#", "1").replace(".", "0")
        if len(row) != 8 or any(ch not in {"0", "1"} for ch in row):
            raise ValueError("Each row must be exactly 8 chars of 0/1 or ./#.")
        arr[r, :] = [1 if ch == "1" else 0 for ch in row]

    return Board(arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Block Blast round planner")
    parser.add_argument(
        "--show-board",
        action="store_true",
        help="Show board state each round (default: hidden).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show extra details (selected bank, coordinates, scoring).",
    )
    args = parser.parse_args()

    _, by_key = build_piece_catalog()
    agent = HeuristicAgent()
    board = prompt_initial_board()

    score = 0
    streak = 0
    round_no = 0

    print("\nPlanner ready. Type Ctrl+C to quit.")
    print("Output mode: ordered move list.")

    while True:
        round_no += 1
        print(f"\n{'=' * 60}")
        print(f"Round {round_no}")
        if args.show_board:
            print(board)
        if args.verbose:
            print(f"Score={score}  Streak={streak}")

        chosen = [choose_piece(i, by_key, verbose=args.verbose) for i in range(1, 4)]
        names = [m.name for m in chosen]
        piece_bank = [m.piece.copy() for m in chosen]
        used = [False, False, False]

        if args.verbose:
            print("\nSelected bank:")
            for i, m in enumerate(chosen, start=1):
                name_hint = " (letter O)" if m.name == "O" else ""
                print(f"  {i}. {m.name}{name_hint}  key={m.key}")

        if board.is_game_over(piece_bank):
            print("\nNo valid move exists for this 3-piece bank. Game over state reached.")
            break

        plan = agent.best_plan(board, piece_bank, used, streak)
        if not plan:
            print("\nAgent could not find a valid plan.")
            break

        move_plan_output: List[Tuple[int, int, int, int]] = []
        round_had_clear = False
        for step_no, (piece_idx, row, col) in enumerate(plan, start=1):
            piece = piece_bank[piece_idx]
            valid = board.valid_moves(piece)
            if (row, col) not in valid:
                print(f"Step {step_no}: planned move became invalid; stopping round.")
                break

            placement_index_1 = valid.index((row, col)) + 1
            move_plan_output.append((step_no, piece_idx + 1, row + 1, col + 1))
            cells, _, _ = Board.get_footprint(piece)
            lines = board.apply_move(piece, row, col)
            points, streak = calculate_score(len(cells), lines, streak)
            score += points
            used[piece_idx] = True
            if lines > 0:
                round_had_clear = True

            if args.verbose:
                print(
                    f"  {step_no}. piece input #{piece_idx + 1} ({names[piece_idx]}) "
                    f"-> placement #{placement_index_1} at ({row + 1},{col + 1}) "
                    f"[valid_count={len(valid)}, lines={lines}, +{points}]"
                )

        for m, p, r, c in move_plan_output:
            print(f"Move {m}: (P{p}), ({r}, {c})")

        if not round_had_clear:
            streak = 0

        if args.show_board:
            print("\nBoard after round:")
            print(board)
        if args.verbose:
            print(f"Score={score}  Streak={streak}")


if __name__ == "__main__":
    main()
