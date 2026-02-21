"""
round_planner_gui.py
====================

GUI round planner for Block Blast.

Features:
- Editable current board (8x8) via click-to-toggle cells
- Three piece editors (5x5) via click-to-toggle cells
- For each piece:
  - Order label ("Move N" when planned)
  - 8x8 preview showing where that piece will be placed
- Submit button runs HeuristicAgent plan for the 3 input pieces
- Plan is applied to the current board for the next cycle
"""

from __future__ import annotations

import re
import tkinter as tk
from tkinter import messagebox
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def piece_footprint(piece: np.ndarray) -> np.ndarray:
    cells, h, w = Board.get_footprint(piece)
    out = np.zeros((h, w), dtype=np.int8)
    for r, c in cells:
        out[r, c] = 1
    return out


def key_from_footprint(fp: np.ndarray) -> str:
    return "/".join("".join("1" if c else "0" for c in row) for row in fp)


def build_piece_map() -> Dict[str, List[PieceMeta]]:
    by_key: Dict[str, List[PieceMeta]] = {}

    for name, piece in sorted(ALL_PIECES.items()):
        fp = piece_footprint(piece)
        key = key_from_footprint(fp)
        by_key.setdefault(key, []).append(PieceMeta(name=name, piece=piece, key=key))

    def alias_rank(meta: PieceMeta) -> tuple[int, str]:
        if re.match(r"^I\d+_(0|90|180|270)$", meta.name):
            return (1, meta.name)
        return (0, meta.name)

    for key in by_key:
        by_key[key].sort(key=alias_rank)
    return by_key


def footprint_from_editor(grid: np.ndarray) -> Optional[np.ndarray]:
    rows, cols = np.where(grid != 0)
    if len(rows) == 0:
        return None
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    return grid[r0 : r1 + 1, c0 : c1 + 1].astype(np.int8)


class ToggleGrid(tk.Canvas):
    def __init__(
        self,
        parent: tk.Widget,
        rows: int,
        cols: int,
        cell: int = 24,
        editable: bool = True,
        **kwargs,
    ):
        super().__init__(
            parent,
            width=cols * cell + 1,
            height=rows * cell + 1,
            bg="white",
            highlightthickness=1,
            highlightbackground="#777",
            **kwargs,
        )
        self.rows = rows
        self.cols = cols
        self.cell = cell
        self.editable = editable
        self.grid_data = np.zeros((rows, cols), dtype=np.int8)
        # 0 = empty, 1 = filled/default, 2 = highlighted (used in move previews)
        self._fill_colors = {
            0: "#ffffff",
            1: "#2b2b2b",
            2: "#ffd54f",
        }
        self._draw()
        if editable:
            self.bind("<Button-1>", self._on_click)

    def _draw(self) -> None:
        self.delete("all")
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = c * self.cell
                y0 = r * self.cell
                x1 = x0 + self.cell
                y1 = y0 + self.cell
                fill = self._fill_colors.get(int(self.grid_data[r, c]), "#2b2b2b")
                self.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#b0b0b0")

    def _on_click(self, event: tk.Event) -> None:
        if not self.editable:
            return
        c = event.x // self.cell
        r = event.y // self.cell
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.grid_data[r, c] = 0 if self.grid_data[r, c] else 1
            self._draw()

    def set_data(self, arr: np.ndarray) -> None:
        if arr.shape != self.grid_data.shape:
            raise ValueError("Shape mismatch in set_data.")
        self.grid_data = arr.astype(np.int8)
        self._draw()

    def clear(self) -> None:
        self.grid_data[:, :] = 0
        self._draw()

    def get_data(self) -> np.ndarray:
        return self.grid_data.copy()


class PiecePanel(tk.Frame):
    def __init__(self, parent: tk.Widget, idx: int):
        super().__init__(parent, bd=1, relief=tk.GROOVE, padx=8, pady=8)
        self.idx = idx
        self.order_var = tk.StringVar(value=f"Piece {idx + 1}: not planned")
        self.name_var = tk.StringVar(value="shape: -")

        tk.Label(self, textvariable=self.order_var, font=("Helvetica", 11, "bold")).pack()
        self.preview = ToggleGrid(self, rows=8, cols=8, cell=14, editable=False)
        self.preview.pack(pady=(4, 6))
        tk.Label(self, textvariable=self.name_var, font=("Helvetica", 9)).pack()
        tk.Label(self, text=f"Draw Piece {idx + 1} (5x5)").pack(pady=(8, 2))
        self.editor = ToggleGrid(self, rows=5, cols=5, cell=22, editable=True)
        self.editor.pack()

    def reset_preview(self) -> None:
        self.order_var.set(f"Piece {self.idx + 1}: not planned")
        self.name_var.set("shape: -")
        self.preview.clear()


class PlannerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Block Blast Round Planner")
        self.resizable(False, False)

        self.by_key = build_piece_map()
        self.agent = HeuristicAgent()
        self.board = Board()
        self.score = 0
        self.streak = 0

        self.status_var = tk.StringVar(value="Draw board/pieces, then click Submit.")
        self.score_var = tk.StringVar(value="Score: 0   Streak: 0")

        self._build_layout()
        self._sync_board_view()

    def _build_layout(self) -> None:
        root = tk.Frame(self, padx=10, pady=10)
        root.pack()

        top = tk.Frame(root)
        top.pack(fill=tk.X)
        tk.Label(top, textvariable=self.score_var, font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(top, textvariable=self.status_var, fg="#2a4d8f").pack(side=tk.LEFT, padx=12)

        middle = tk.Frame(root)
        middle.pack(pady=(8, 10))

        left = tk.Frame(middle, bd=1, relief=tk.GROOVE, padx=10, pady=10)
        left.pack(side=tk.LEFT, padx=(0, 10))
        tk.Label(left, text="Current Board (8x8)", font=("Helvetica", 10, "bold")).pack()
        self.board_view = ToggleGrid(left, rows=8, cols=8, cell=24, editable=True)
        self.board_view.pack(pady=6)

        buttons = tk.Frame(left)
        buttons.pack(pady=(2, 0))
        tk.Button(buttons, text="Clear Board", command=self._clear_board).pack(side=tk.LEFT, padx=4)
        tk.Button(buttons, text="Reset Round Inputs", command=self._clear_piece_inputs).pack(side=tk.LEFT, padx=4)

        right = tk.Frame(middle)
        right.pack(side=tk.LEFT)
        self.panels = [PiecePanel(right, i) for i in range(3)]
        for p in self.panels:
            p.pack(side=tk.LEFT, padx=6)

        bottom = tk.Frame(root)
        bottom.pack(fill=tk.X)
        tk.Button(bottom, text="Submit", width=18, command=self._submit).pack(side=tk.LEFT)
        tk.Button(bottom, text="Undo Board Sync", width=18, command=self._sync_board_view).pack(side=tk.LEFT, padx=8)

    def _clear_board(self) -> None:
        self.board = Board()
        self._sync_board_view()
        self.status_var.set("Board cleared.")

    def _clear_piece_inputs(self) -> None:
        for p in self.panels:
            p.editor.clear()
            p.reset_preview()
        self.status_var.set("Piece inputs cleared.")

    def _sync_board_view(self) -> None:
        self.board_view.set_data(self.board.board)
        self.score_var.set(f"Score: {self.score}   Streak: {self.streak}")

    def _resolve_pieces(self) -> Optional[List[PieceMeta]]:
        chosen: List[PieceMeta] = []
        for i, panel in enumerate(self.panels, start=1):
            fp = footprint_from_editor(panel.editor.get_data())
            if fp is None:
                messagebox.showerror("Invalid piece", f"Piece {i} is empty.")
                return None
            key = key_from_footprint(fp)
            matches = self.by_key.get(key, [])
            if not matches:
                messagebox.showerror("Unknown shape", f"Piece {i} shape '{key}' is not in the catalog.")
                return None
            chosen.append(matches[0])
        return chosen

    def _paint_preview(self, panel: PiecePanel, piece: np.ndarray, row: int, col: int) -> None:
        arr = self.board.board.copy()
        cells, _, _ = Board.get_footprint(piece)
        for dr, dc in cells:
            rr = row + int(dr)
            cc = col + int(dc)
            if 0 <= rr < 8 and 0 <= cc < 8:
                # Highlight currently placed piece in yellow on the move preview.
                arr[rr, cc] = 2
        panel.preview.set_data(arr)

    def _submit(self) -> None:
        # Sync board from editable board view before planning.
        self.board = Board(self.board_view.get_data())

        resolved = self._resolve_pieces()
        if resolved is None:
            return

        piece_bank = [m.piece.copy() for m in resolved]
        used = [False, False, False]

        if self.board.is_game_over(piece_bank):
            self.status_var.set("No valid move exists for this 3-piece set.")
            for p in self.panels:
                p.reset_preview()
            return

        plan = self.agent.best_plan(self.board, piece_bank, used, self.streak)
        if not plan:
            self.status_var.set("Agent could not find a valid plan.")
            for p in self.panels:
                p.reset_preview()
            return

        for p in self.panels:
            p.reset_preview()

        round_had_clear = False
        lines_out: List[str] = []
        for move_no, (piece_idx, row, col) in enumerate(plan, start=1):
            panel = self.panels[piece_idx]
            piece = piece_bank[piece_idx]
            valid = self.board.valid_moves(piece)
            if (row, col) not in valid:
                self.status_var.set(f"Move {move_no} became invalid; stopped.")
                break

            panel.order_var.set(f"Move {move_no}")
            panel.name_var.set(f"shape: {resolved[piece_idx].name}")
            self._paint_preview(panel, piece, row, col)

            cells, _, _ = Board.get_footprint(piece)
            lines = self.board.apply_move(piece, row, col)
            pts, self.streak = calculate_score(len(cells), lines, self.streak)
            self.score += pts
            if lines > 0:
                round_had_clear = True

            lines_out.append(f"Move {move_no}: (P{piece_idx + 1}), ({row + 1}, {col + 1})")

        if not round_had_clear:
            self.streak = 0

        # Input pieces are consumed for this round; clear editors for next round.
        if lines_out:
            for p in self.panels:
                p.editor.clear()

        self._sync_board_view()
        self.status_var.set(" | ".join(lines_out) if lines_out else "No moves applied.")


def main() -> None:
    app = PlannerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
