"""
round_planner_gui.py
====================

GUI round planner for Block Blast.

Features:
- Editable current board (8x8) via click-to-toggle cells
- Three piece editors (5x5) via click-to-toggle cells
- Three move previews (8x8), always ordered left->right as Move 1/2/3
- Submit button runs HeuristicAgent plan for the 3 input pieces
- Plan is applied to the current board for the next cycle
"""

from __future__ import annotations

import json
import re
import tkinter as tk
import time
from tkinter import messagebox
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from board import Board
from game import calculate_score
from live_cv_bridge import LiveCVConfig, LiveCVPlanner
from model import HeuristicAgent
from piece_bank_detector import PieceBankDetectorConfig
from pieces import ALL_PIECES
from stream_capture import CaptureConfig


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


def parse_norm_roi_xyxy(raw: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be 4 comma-separated floats: x0,y0,x1,y1")
    vals = [float(p) for p in parts]
    x0, y0, x1, y1 = vals
    if not (0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0):
        raise ValueError("ROI values must be in [0,1].")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI must satisfy x1>x0 and y1>y0.")
    return x0, y0, x1, y1


def derive_norm_roi_from_approx_abs(
    window_xyxy: tuple[float, float, float, float],
    roi_xyxy: tuple[float, float, float, float],
    max_extent: float = 0.98,
) -> tuple[float, float, float, float]:
    """
    Convert approximate absolute pixel coordinates to normalized ROI (x0,y0,x1,y1).

    Robust behavior:
    - accepts approximate values (including ROI partly outside window bounds)
    - preserves center/extent when possible
    - clamps into [0,1] while keeping a valid rectangle
    """
    wx0, wy0, wx1, wy1 = [float(v) for v in window_xyxy]
    rx0, ry0, rx1, ry1 = [float(v) for v in roi_xyxy]

    ww = max(1.0, wx1 - wx0)
    wh = max(1.0, wy1 - wy0)

    nx0 = (rx0 - wx0) / ww
    ny0 = (ry0 - wy0) / wh
    nx1 = (rx1 - wx0) / ww
    ny1 = (ry1 - wy0) / wh

    cx = 0.5 * (nx0 + nx1)
    cy = 0.5 * (ny0 + ny1)
    ex = abs(nx1 - nx0)
    ey = abs(ny1 - ny0)

    ex = float(np.clip(ex, 0.02, max_extent))
    ey = float(np.clip(ey, 0.02, max_extent))

    cx = float(np.clip(cx, ex * 0.5, 1.0 - ex * 0.5))
    cy = float(np.clip(cy, ey * 0.5, 1.0 - ey * 0.5))

    x0 = float(np.clip(cx - ex * 0.5, 0.0, 1.0))
    y0 = float(np.clip(cy - ey * 0.5, 0.0, 1.0))
    x1 = float(np.clip(cx + ex * 0.5, 0.0, 1.0))
    y1 = float(np.clip(cy + ey * 0.5, 0.0, 1.0))

    if x1 <= x0:
        x1 = min(1.0, x0 + 0.05)
    if y1 <= y0:
        y1 = min(1.0, y0 + 0.05)
    return x0, y0, x1, y1


def format_norm_roi_xyxy(roi: tuple[float, float, float, float]) -> str:
    return ",".join(f"{v:.4f}" for v in roi)


DEFAULT_WINDOW_APPROX_XYXY = (1028.0, 25.0, 1439.0, 899.0)
DEFAULT_BANK_ROI_APPROX_XYXY = (1054.0, 632.0, 1407.0, 736.0)
DEFAULT_BANK_ROI_NORM = derive_norm_roi_from_approx_abs(
    window_xyxy=DEFAULT_WINDOW_APPROX_XYXY,
    roi_xyxy=DEFAULT_BANK_ROI_APPROX_XYXY,
)

RUNTIME_CONFIG_PATH = Path(__file__).resolve().parent / "runtime_config.json"
DEFAULT_RUNTIME_CONFIG = {
    "live_window_title_contains": "Movie Recording",
    "live_bank_roi_norm_xyxy": DEFAULT_BANK_ROI_NORM,
    "expected_cell_px": 20.0,
    "capture_target_fps": 20.0,
    "capture_window_refresh_sec": 1.0,
}


def _parse_norm_roi_config_value(raw: object) -> tuple[float, float, float, float]:
    if isinstance(raw, str):
        return parse_norm_roi_xyxy(raw)
    if isinstance(raw, (list, tuple)) and len(raw) == 4:
        return parse_norm_roi_xyxy(",".join(str(v) for v in raw))
    raise ValueError("live_bank_roi_norm_xyxy must be 4 floats (list/tuple) or a comma-separated string.")


def load_runtime_config(path: Path = RUNTIME_CONFIG_PATH) -> dict:
    cfg = dict(DEFAULT_RUNTIME_CONFIG)
    cfg["live_bank_roi_norm_xyxy"] = tuple(DEFAULT_RUNTIME_CONFIG["live_bank_roi_norm_xyxy"])

    if not path.exists():
        return cfg

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Could not parse JSON from {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"Runtime config at {path} must be a JSON object.")

    title = raw.get("live_window_title_contains")
    if title is not None:
        if not isinstance(title, str) or not title.strip():
            raise RuntimeError("live_window_title_contains must be a non-empty string.")
        cfg["live_window_title_contains"] = title.strip()

    roi_val = raw.get("live_bank_roi_norm_xyxy")
    if roi_val is not None:
        cfg["live_bank_roi_norm_xyxy"] = _parse_norm_roi_config_value(roi_val)

    for key in ("expected_cell_px", "capture_target_fps", "capture_window_refresh_sec"):
        if key not in raw:
            continue
        try:
            val = float(raw[key])
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"{key} must be numeric: {exc}") from exc
        if val <= 0:
            raise RuntimeError(f"{key} must be > 0.")
        cfg[key] = val

    return cfg


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
        self._drag_active = False
        self._drag_seen: set[tuple[int, int]] = set()
        self._draw()
        if editable:
            self.bind("<ButtonPress-1>", self._on_press)
            self.bind("<B1-Motion>", self._on_drag)
            self.bind("<ButtonRelease-1>", self._on_release)

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

    def _cell_from_event(self, event: tk.Event) -> Optional[tuple[int, int]]:
        c = event.x // self.cell
        r = event.y // self.cell
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return int(r), int(c)
        return None

    def _toggle_cell_once(self, r: int, c: int) -> None:
        if (r, c) in self._drag_seen:
            return
        self._drag_seen.add((r, c))
        # Toggle only editable states (0/1). Ignore preview highlight state 2.
        self.grid_data[r, c] = 0 if self.grid_data[r, c] else 1
        self._draw()

    def _on_press(self, event: tk.Event) -> None:
        if not self.editable:
            return
        self._drag_active = True
        self._drag_seen.clear()
        rc = self._cell_from_event(event)
        if rc is None:
            return
        self._toggle_cell_once(*rc)

    def _on_drag(self, event: tk.Event) -> None:
        if not self.editable or not self._drag_active:
            return
        rc = self._cell_from_event(event)
        if rc is None:
            return
        self._toggle_cell_once(*rc)

    def _on_release(self, event: tk.Event) -> None:
        if not self.editable:
            return
        self._drag_active = False
        self._drag_seen.clear()

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
        tk.Label(self, text=f"Piece {idx + 1} Input (5x5)", font=("Helvetica", 10, "bold")).pack(pady=(0, 4))
        self.editor = ToggleGrid(self, rows=5, cols=5, cell=22, editable=True)
        self.editor.pack()

    def clear(self) -> None:
        self.editor.clear()


class MovePreviewPanel(tk.Frame):
    def __init__(self, parent: tk.Widget, move_no: int):
        super().__init__(parent, bd=1, relief=tk.GROOVE, padx=8, pady=8)
        self.move_no = move_no
        self.title_var = tk.StringVar(value=f"Move {move_no}")
        self.piece_var = tk.StringVar(value="piece: -")

        tk.Label(self, textvariable=self.title_var, font=("Helvetica", 11, "bold")).pack()
        tk.Label(self, textvariable=self.piece_var, font=("Helvetica", 9)).pack(pady=(0, 4))
        self.preview = ToggleGrid(self, rows=8, cols=8, cell=14, editable=False)
        self.preview.pack()

    def reset(self) -> None:
        self.piece_var.set("piece: -")
        self.preview.clear()

    def set_piece_info(self, label: str) -> None:
        self.piece_var.set(label)


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

        self.timer_total_seconds = 6 * 60
        self.timer_remaining_seconds = self.timer_total_seconds
        self.timer_running = False
        self.timer_job: Optional[str] = None
        self.timer_end_monotonic: Optional[float] = None
        self.live_planner: Optional[LiveCVPlanner] = None
        try:
            self.runtime_config = load_runtime_config()
        except Exception as exc:  # noqa: BLE001
            self.runtime_config = dict(DEFAULT_RUNTIME_CONFIG)
            self.runtime_config["live_bank_roi_norm_xyxy"] = tuple(DEFAULT_RUNTIME_CONFIG["live_bank_roi_norm_xyxy"])
            messagebox.showwarning(
                "Runtime Config",
                f"Failed to load {RUNTIME_CONFIG_PATH.name}; using defaults.\n\n{exc}",
            )

        self.live_window_title_var = tk.StringVar(value=str(self.runtime_config["live_window_title_contains"]))
        self.live_bank_roi_var = tk.StringVar(
            value=format_norm_roi_xyxy(tuple(self.runtime_config["live_bank_roi_norm_xyxy"]))
        )
        self.live_status_var = tk.StringVar(value="Live stream: detached")
        self.live_expected_cell_px = float(self.runtime_config["expected_cell_px"])
        self.live_capture_target_fps = float(self.runtime_config["capture_target_fps"])
        self.live_capture_window_refresh_sec = float(self.runtime_config["capture_window_refresh_sec"])

        self.status_var = tk.StringVar(value="Draw board/pieces, then click Submit.")
        self.score_var = tk.StringVar(value="Score: 0   Streak: 0")
        self.timer_var = tk.StringVar(value=f"Time: {self._format_seconds(self.timer_remaining_seconds)}")
        self.profile_var = tk.StringVar(value="Profile: balanced")

        self._build_layout()
        self._sync_board_view()
        self._refresh_profile_label()
        self.live_status_var.set(f"Live stream: detached (config={RUNTIME_CONFIG_PATH.name})")
        self.bind("<space>", self._on_space_trigger)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self) -> None:
        root = tk.Frame(self, padx=10, pady=10)
        root.pack()

        top = tk.Frame(root)
        top.pack(fill=tk.X)
        top_left = tk.Frame(top)
        top_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(top_left, textvariable=self.score_var, font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)
        tk.Label(top_left, textvariable=self.status_var, fg="#2a4d8f").pack(side=tk.LEFT, padx=12)

        top_right = tk.Frame(top)
        top_right.pack(side=tk.RIGHT)
        self.start_timer_btn = tk.Button(top_right, text="Start timer", command=self._start_timer)
        self.start_timer_btn.pack(side=tk.RIGHT)
        tk.Label(top_right, textvariable=self.timer_var, font=("Helvetica", 11, "bold")).pack(side=tk.RIGHT, padx=(0, 10))
        tk.Label(top_right, textvariable=self.profile_var, font=("Helvetica", 10)).pack(side=tk.RIGHT, padx=(0, 10))

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

        tk.Label(right, text="Move Previews (left -> right execution order)", font=("Helvetica", 10, "bold")).pack(anchor="w")
        previews_row = tk.Frame(right)
        previews_row.pack(pady=(4, 10))
        self.move_panels = [MovePreviewPanel(previews_row, i + 1) for i in range(3)]
        for p in self.move_panels:
            p.pack(side=tk.LEFT, padx=6)

        tk.Label(right, text="Piece Inputs", font=("Helvetica", 10, "bold")).pack(anchor="w")
        editors_row = tk.Frame(right)
        editors_row.pack(pady=(4, 0))
        self.panels = [PiecePanel(editors_row, i) for i in range(3)]
        for p in self.panels:
            p.pack(side=tk.LEFT, padx=6)

        bottom = tk.Frame(root)
        bottom.pack(fill=tk.X)
        tk.Button(bottom, text="Submit", width=18, command=self._submit).pack(side=tk.LEFT)
        tk.Button(bottom, text="Undo Board Sync", width=18, command=self._sync_board_view).pack(side=tk.LEFT, padx=8)

        live = tk.Frame(root, bd=1, relief=tk.GROOVE, padx=8, pady=8)
        live.pack(fill=tk.X, pady=(8, 0))
        tk.Label(live, text="Live CV Stream", font=("Helvetica", 10, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(live, textvariable=self.live_status_var, fg="#1f5e9f").grid(row=0, column=1, columnspan=5, sticky="w", padx=(8, 0))

        tk.Label(live, text="Window title contains:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        tk.Entry(live, textvariable=self.live_window_title_var, width=26).grid(row=1, column=1, sticky="w", pady=(6, 0))
        tk.Label(live, text="Bank ROI norm x0,y0,x1,y1:").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(6, 0))
        tk.Entry(live, textvariable=self.live_bank_roi_var, width=20).grid(row=1, column=3, sticky="w", pady=(6, 0))
        tk.Button(live, text="Attach Stream", command=self._attach_live_stream).grid(row=1, column=4, padx=(10, 0), pady=(6, 0))
        tk.Button(live, text="Detach", command=self._detach_live_stream).grid(row=1, column=5, padx=(6, 0), pady=(6, 0))
        tk.Label(
            live,
            text="Press Space to run: capture -> recognize -> evaluate -> display",
            fg="#555",
        ).grid(row=2, column=0, columnspan=6, sticky="w", pady=(6, 0))

    def _clear_board(self) -> None:
        self.board = Board()
        self._sync_board_view()
        self.status_var.set("Board cleared.")

    def _clear_piece_inputs(self) -> None:
        for p in self.panels:
            p.clear()
        for p in self.move_panels:
            p.reset()
        self.status_var.set("Piece inputs cleared.")

    def _sync_board_view(self) -> None:
        self.board_view.set_data(self.board.board)
        self.score_var.set(f"Score: {self.score}   Streak: {self.streak}")
        self._refresh_profile_label()

    @staticmethod
    def _format_seconds(seconds: int) -> str:
        mm = max(0, int(seconds)) // 60
        ss = max(0, int(seconds)) % 60
        return f"{mm:02d}:{ss:02d}"

    def _active_profile(self) -> str:
        remaining = self._current_remaining_seconds()
        # If timer has not started, stay on balanced.
        if (not self.timer_running) and remaining == self.timer_total_seconds:
            return "balanced"
        if remaining <= 120:
            return "aggressive"
        if remaining <= 240:
            return "balanced"
        return "safe"

    def _refresh_profile_label(self) -> str:
        self.timer_remaining_seconds = self._current_remaining_seconds()
        self.timer_var.set(f"Time: {self._format_seconds(self.timer_remaining_seconds)}")
        profile = self._active_profile()
        self.profile_var.set(f"Profile: {profile}")
        return profile

    def _current_remaining_seconds(self) -> int:
        if self.timer_running and self.timer_end_monotonic is not None:
            remaining = max(0, int(self.timer_end_monotonic - time.monotonic() + 0.999))
            self.timer_remaining_seconds = remaining
            if remaining <= 0:
                self.timer_running = False
                self.timer_end_monotonic = None
            return remaining
        return self.timer_remaining_seconds

    def _cancel_timer_job(self) -> None:
        if self.timer_job is not None:
            self.after_cancel(self.timer_job)
            self.timer_job = None

    def _start_timer(self) -> None:
        self._cancel_timer_job()
        self.timer_remaining_seconds = self.timer_total_seconds
        self.timer_running = True
        self.timer_end_monotonic = time.monotonic() + self.timer_total_seconds
        self._refresh_profile_label()
        self.status_var.set("Timer started (6:00). Profile auto-switches by time left.")
        self.timer_job = self.after(1000, self._timer_tick)

    def _timer_tick(self) -> None:
        if not self.timer_running:
            self.timer_job = None
            return

        self.timer_remaining_seconds = self._current_remaining_seconds()
        self.timer_var.set(f"Time: {self._format_seconds(self.timer_remaining_seconds)}")
        self._refresh_profile_label()

        if self.timer_remaining_seconds <= 0:
            self.timer_running = False
            self.timer_end_monotonic = None
            self.timer_job = None
            self.status_var.set("Timer ended. Planner now uses aggressive profile.")
            return

        self.timer_job = self.after(1000, self._timer_tick)

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

    def _paint_preview(self, panel: MovePreviewPanel, piece: np.ndarray, row: int, col: int) -> None:
        arr = self.board.board.copy()
        cells, _, _ = Board.get_footprint(piece)
        for dr, dc in cells:
            rr = row + int(dr)
            cc = col + int(dc)
            if 0 <= rr < 8 and 0 <= cc < 8:
                # Highlight currently placed piece in yellow on the move preview.
                arr[rr, cc] = 2
        panel.preview.set_data(arr)

    def _attach_live_stream(self) -> None:
        title = self.live_window_title_var.get().strip()
        if not title:
            messagebox.showerror("Live Stream", "Window title cannot be empty.")
            return

        try:
            roi = parse_norm_roi_xyxy(self.live_bank_roi_var.get().strip())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Live Stream", f"Invalid ROI: {exc}")
            return

        self._detach_live_stream()
        try:
            cfg = LiveCVConfig(
                capture=CaptureConfig(
                    window_title_contains=title,
                    target_fps=self.live_capture_target_fps,
                    window_refresh_sec=self.live_capture_window_refresh_sec,
                ),
                detector=PieceBankDetectorConfig(
                    bank_roi_norm_xyxy=roi,
                    debug_visuals=False,
                    expected_cell_px=self.live_expected_cell_px,
                ),
            )
            self.live_planner = LiveCVPlanner(config=cfg, agent=self.agent)
            self.live_planner.start()
            self.live_status_var.set(
                f"Live stream: attached (title~'{title}', roi={format_norm_roi_xyxy(roi)})"
            )
            self.status_var.set("Live capture attached. Press Space to capture/recognize/evaluate.")
        except Exception as exc:  # noqa: BLE001
            self.live_planner = None
            messagebox.showerror("Live Stream", f"Failed to attach stream: {exc}")

    def _detach_live_stream(self) -> None:
        if self.live_planner is not None:
            try:
                self.live_planner.stop()
            except Exception:  # noqa: BLE001
                pass
        self.live_planner = None
        self.live_status_var.set("Live stream: detached")

    def _on_space_trigger(self, event=None) -> None:  # noqa: ANN001
        if self.live_planner is None:
            return

        # Sync board from editable board view before planning.
        self.board = Board(self.board_view.get_data())
        active_profile = self._refresh_profile_label()
        try:
            result = self.live_planner.process_once(self.board, streak=self.streak, profile=active_profile)
        except Exception as exc:  # noqa: BLE001
            self.status_var.set(f"Live cycle failed: {exc}")
            return

        resolved: List[PieceMeta] = []
        for i, det in enumerate(result.detection.pieces):
            self.panels[i].editor.set_data(det.piece_array_5x5.astype(np.int8))
            display_name = det.name or f"UNKNOWN_P{i + 1}"
            meta = PieceMeta(name=display_name, piece=det.piece_array_5x5.astype(np.int8), key=det.key or "")
            resolved.append(meta)

        self._run_round(
            resolved=resolved,
            active_profile=active_profile,
            source_label=f"cv {result.total_ms:.0f}ms",
            clear_inputs=False,
        )

    def _run_round(
        self,
        resolved: List[PieceMeta],
        active_profile: str,
        source_label: str,
        clear_inputs: bool = True,
    ) -> None:
        piece_bank = [m.piece.copy() for m in resolved]
        used = [False, False, False]

        if self.board.is_game_over(piece_bank):
            self.status_var.set("No valid move exists for this 3-piece set.")
            for p in self.move_panels:
                p.reset()
            return

        plan = self.agent.best_plan(self.board, piece_bank, used, self.streak, profile=active_profile)
        if not plan:
            self.status_var.set("Agent could not find a valid plan.")
            for p in self.move_panels:
                p.reset()
            return

        for p in self.move_panels:
            p.reset()

        round_had_clear = False
        lines_out: List[str] = []
        for move_no, (piece_idx, row, col) in enumerate(plan, start=1):
            panel = self.move_panels[move_no - 1]
            piece = piece_bank[piece_idx]
            valid = self.board.valid_moves(piece)
            if (row, col) not in valid:
                self.status_var.set(f"Move {move_no} became invalid; stopped.")
                break

            panel.set_piece_info(f"piece: P{piece_idx + 1} ({resolved[piece_idx].name})")
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

        if lines_out and clear_inputs:
            for p in self.panels:
                p.clear()

        self._sync_board_view()
        if lines_out:
            self.status_var.set(f"{source_label} | profile={active_profile} | " + " | ".join(lines_out))
        else:
            self.status_var.set("No moves applied.")

    def _submit(self) -> None:
        # Sync board from editable board view before planning.
        self.board = Board(self.board_view.get_data())

        resolved = self._resolve_pieces()
        if resolved is None:
            return

        active_profile = self._refresh_profile_label()
        self._run_round(resolved=resolved, active_profile=active_profile, source_label="manual", clear_inputs=True)

    def _on_close(self) -> None:
        self._detach_live_stream()
        self.destroy()


def main() -> None:
    app = PlannerGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
