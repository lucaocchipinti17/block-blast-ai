"""
block_blast_parser.py
=====================
Parses a live Block Blast screen (mirrored iPhone) into numpy arrays.

SETUP
-----
Install deps:
    pip install opencv-python numpy mss

Screen mirroring options:
  - Mac:     plug in iPhone via USB → open QuickTime → File > New Movie Recording
             → click dropdown arrow next to record button → select your iPhone
  - Windows: use "Phone Link" app or install "ApowerMirror" / "LonelyScreen"

HOW IT WORKS
------------
1. Finds the Block Blast window on your screen (or you can specify a monitor region)
2. Dynamically locates the 8x8 grid by detecting the dark grid-background panel
3. Auto-calibrates cell size from the grid bounds
4. Parses each cell as filled/empty based on HSV color thresholds
5. Detects the 3 incoming piece slots below the grid and parses their shapes
6. Runs in a loop — call get_game_state() at any time to get the current state

Returns
-------
board  : np.ndarray, shape (8, 8), dtype int  — 1=filled, 0=empty
pieces : list of 3 np.ndarray, each shape (rows, cols), dtype int

USAGE
-----
    from block_blast_parser import BlockBlastParser

    parser = BlockBlastParser()
    board, pieces = parser.get_game_state()
    print(board)
    for i, p in enumerate(pieces):
        print(f"Piece {i+1}:", p)
"""

import cv2
import numpy as np
import time
from typing import Optional

# ── Try to import mss for live screen capture ─────────────────────────────────
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("[WARNING] mss not installed — live capture disabled. "
          "Use parse_image() to parse a screenshot file instead.\n"
          "Install with: pip install mss")


# ── Constants ─────────────────────────────────────────────────────────────────
# HSV thresholds for detecting a filled cell
MIN_SAT   = 60   # minimum saturation (0-255)
MIN_VAL   = 80   # minimum value/brightness (0-255)

# Fraction of cell to sample (centered patch)
SAMPLE_FRAC = 0.5


class BlockBlastParser:
    """
    Parses Block Blast game state from a screenshot or live screen region.
    
    The parser auto-calibrates to any screen resolution — no hardcoded pixel
    coords. It only needs the grid to be visible and unobstructed.
    """

    def __init__(
        self,
        monitor: int = 1,
        region: Optional[dict] = None,
        debug: bool = False,
    ):
        """
        Parameters
        ----------
        monitor : int
            Which monitor to capture (1-indexed). Ignored if `region` is set.
        region : dict, optional
            Explicit capture region: {"top": y, "left": x, "width": w, "height": h}
            Use this to crop to just the mirrored phone window for better performance.
        debug : bool
            If True, prints calibration info and shows a debug window.
        """
        self.monitor = monitor
        self.region = region
        self.debug = debug

        # Cached calibration — recomputed whenever the grid moves/resizes
        self._grid_rect = None    # (x1, y1, x2, y2) in capture coords
        self._cell_w = None
        self._cell_h = None
        self._piece_cell = None   # pixel size of one piece-tray cell

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_game_state(self):
        """
        Capture the current screen and return (board, pieces).
        
        Returns
        -------
        board : np.ndarray (8, 8)  — 1=filled, 0=empty
        pieces : list[np.ndarray]  — up to 3 piece arrays, each (rows, cols)
        """
        frame = self._capture()
        return self._parse(frame)

    def parse_image(self, path: str):
        """
        Parse a saved screenshot file instead of the live screen.
        Useful for testing without a mirrored device.
        """
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return self._parse(img)

    def run_loop(self, callback, fps: float = 2.0):
        """
        Call `callback(board, pieces)` at the given FPS until KeyboardInterrupt.
        
        Example:
            def on_state(board, pieces):
                print(board)
            parser.run_loop(on_state, fps=1)
        """
        interval = 1.0 / fps
        print(f"[BlockBlastParser] Running at {fps} FPS. Press Ctrl+C to stop.")
        try:
            while True:
                t = time.time()
                board, pieces = self.get_game_state()
                callback(board, pieces)
                elapsed = time.time() - t
                time.sleep(max(0, interval - elapsed))
        except KeyboardInterrupt:
            print("[BlockBlastParser] Stopped.")

    # ──────────────────────────────────────────────────────────────────────────
    # Screen capture
    # ──────────────────────────────────────────────────────────────────────────

    def _capture(self) -> np.ndarray:
        if not MSS_AVAILABLE:
            raise RuntimeError("mss is not installed. Use parse_image() instead.")
        with mss.mss() as sct:
            mon = self.region or sct.monitors[self.monitor]
            shot = sct.grab(mon)
            frame = np.array(shot)
            # mss returns BGRA — convert to BGR
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # ──────────────────────────────────────────────────────────────────────────
    # Core parsing
    # ──────────────────────────────────────────────────────────────────────────

    def _parse(self, img: np.ndarray):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, W = img.shape[:2]

        # ── Step 1: Locate the grid ───────────────────────────────────────────
        gx1, gy1, gx2, gy2 = self._detect_grid(img, hsv)
        cell_w = (gx2 - gx1) / 8
        cell_h = (gy2 - gy1) / 8

        if self.debug:
            print(f"[debug] Grid: ({gx1},{gy1})-({gx2},{gy2})  "
                  f"cell: {cell_w:.1f}x{cell_h:.1f}px")

        # ── Step 2: Parse board ───────────────────────────────────────────────
        board = np.zeros((8, 8), dtype=int)
        for row in range(8):
            for col in range(8):
                cx = int(gx1 + (col + 0.5) * cell_w)
                cy = int(gy1 + (row + 0.5) * cell_h)
                board[row, col] = self._is_cell_filled(hsv, cx, cy, cell_w, cell_h)

        # ── Step 3: Locate & parse piece tray ────────────────────────────────
        pieces = self._parse_pieces(hsv, gx1, gy2, gx2, H, cell_w, cell_h)

        if self.debug:
            self._show_debug(img, gx1, gy1, gx2, gy2, cell_w, cell_h, board)

        return board, pieces

    # ──────────────────────────────────────────────────────────────────────────
    # Grid detection
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_grid(self, img, hsv):
        """
        Robustly locate the 8x8 grid rectangle.
        
        Strategy:
        1. The grid panel is a dark rectangle (V ~15-50) that's slightly lighter
           than the surrounding near-black UI background (V < 15).
        2. Find the largest such rectangle in the upper 75% of the image.
        3. Fall back to proportional estimates if contour detection fails.
        """
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold for the grid panel background
        grid_panel = ((gray >= 12) & (gray <= 55)).astype(np.uint8) * 255

        # Morphological close to fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        grid_panel = cv2.morphologyEx(grid_panel, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            grid_panel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:8]:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            area_frac = (w * h) / (W * H)
            # Grid: roughly square, >15% of image, in top 75%
            if 0.6 < aspect < 1.4 and area_frac > 0.15 and y < H * 0.75:
                return (x, y, x + w, y + h)

        # Fallback: proportional estimate based on typical game layout
        # (works for most screen mirror apps at standard phone aspect ratios)
        if self.debug:
            print("[debug] Grid contour not found, using proportional fallback")
        return (
            int(W * 0.057),
            int(H * 0.236),
            int(W * 0.942),
            int(H * 0.651),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Cell classification
    # ──────────────────────────────────────────────────────────────────────────

    def _is_cell_filled(self, hsv, cx, cy, cell_w, cell_h):
        """Sample the center of a cell and return 1 if it contains a colored block."""
        r_w = max(4, int(cell_w * SAMPLE_FRAC / 2))
        r_h = max(4, int(cell_h * SAMPLE_FRAC / 2))
        patch = hsv[cy - r_h : cy + r_h, cx - r_w : cx + r_w]
        if patch.size == 0:
            return 0
        return int(patch[:, :, 1].mean() > MIN_SAT and patch[:, :, 2].mean() > MIN_VAL)

    # ──────────────────────────────────────────────────────────────────────────
    # Piece tray parsing
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_pieces(self, hsv, gx1, gy2, gx2, img_h, cell_w, cell_h):
        """
        Find and parse the 3 incoming pieces below the grid.
        
        The tray spans from just below the grid to ~30% further down.
        Pieces are found by detecting bright-pixel clusters in this region,
        then each cluster is parsed into a grid using the same cell size logic.
        """
        H = img_h
        tray_y1 = gy2 + int(cell_h * 0.3)
        tray_y2 = min(H, gy2 + int(cell_h * 5))

        tray_hsv = hsv[tray_y1:tray_y2, gx1:gx2]
        bright = (
            (tray_hsv[:, :, 1] > MIN_SAT) &
            (tray_hsv[:, :, 2] > MIN_VAL)
        )

        # Project onto x-axis to find 3 column clusters
        x_proj = bright.any(axis=0).astype(np.uint8)
        from itertools import groupby

        clusters = []
        in_cluster = False
        start = 0
        for i, val in enumerate(x_proj):
            if val and not in_cluster:
                start = i
                in_cluster = True
            elif not val and in_cluster:
                clusters.append((start + gx1, i + gx1))
                in_cluster = False
        if in_cluster:
            clusters.append((start + gx1, len(x_proj) + gx1))

        # Filter noise (too narrow) and keep up to 3
        clusters = [(x1, x2) for x1, x2 in clusters if (x2 - x1) > cell_w * 0.4]

        if self.debug:
            print(f"[debug] Piece x-clusters: {clusters}")

        pieces = []
        for px1, px2 in clusters[:3]:
            # Find y bounds for this piece
            col_slice = bright[:, px1 - gx1 : px2 - gx1]
            y_proj = col_slice.any(axis=1)
            ys = np.where(y_proj)[0]
            if len(ys) == 0:
                continue
            py1 = ys.min() + tray_y1
            py2 = ys.max() + tray_y1 + 1

            # Piece cell size: pieces in the tray are scaled down vs the board
            # Estimate from bounding box dimensions
            pw, ph = px2 - px1, py2 - py1
            # Round to nearest integer number of cells
            est_cols = max(1, round(pw / (cell_w * 0.44)))
            est_rows = max(1, round(ph / (cell_h * 0.44)))
            piece_cell_w = pw / est_cols
            piece_cell_h = ph / est_rows

            piece = np.zeros((est_rows, est_cols), dtype=int)
            for r in range(est_rows):
                for c in range(est_cols):
                    cx = int(px1 + (c + 0.5) * piece_cell_w)
                    cy = int(py1 + (r + 0.5) * piece_cell_h)
                    patch = hsv[cy - 8 : cy + 8, cx - 8 : cx + 8]
                    if patch.size > 0:
                        s = patch[:, :, 1].mean()
                        v = patch[:, :, 2].mean()
                        piece[r, c] = int(s > MIN_SAT and v > MIN_VAL)

            pieces.append(piece)

        return pieces

    # ──────────────────────────────────────────────────────────────────────────
    # Debug visualization
    # ──────────────────────────────────────────────────────────────────────────

    def _show_debug(self, img, gx1, gy1, gx2, gy2, cell_w, cell_h, board):
        vis = img.copy()
        # Draw grid outline
        cv2.rectangle(vis, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
        # Draw cell grid lines
        for i in range(9):
            x = int(gx1 + i * cell_w)
            y = int(gy1 + i * cell_h)
            cv2.line(vis, (x, gy1), (x, gy2), (0, 200, 0), 1)
            cv2.line(vis, (gx1, y), (gx2, y), (0, 200, 0), 1)
        # Highlight filled cells
        for row in range(8):
            for col in range(8):
                if board[row, col]:
                    cx = int(gx1 + (col + 0.5) * cell_w)
                    cy = int(gy1 + (row + 0.5) * cell_h)
                    cv2.circle(vis, (cx, cy), 8, (0, 255, 255), -1)
        # Save debug image instead of displaying (safe in headless environments)
        scale = 600 / vis.shape[0]
        vis_small = cv2.resize(vis, (int(vis.shape[1] * scale), 600))
        cv2.imwrite("debug_grid.png", vis_small)
        # Uncomment below if running on a desktop with a display:
        # cv2.imshow("BlockBlast Debug", vis_small)
        # cv2.waitKey(1)


# ── Pretty print helpers ──────────────────────────────────────────────────────

def print_board(board: np.ndarray):
    print("┌" + "─" * 8 + "┐")
    for row in board:
        print("│" + "".join("█" if c else "·" for c in row) + "│")
    print("└" + "─" * 8 + "┘")

def print_pieces(pieces):
    for i, p in enumerate(pieces):
        print(f"Piece {i+1} ({p.shape[0]}×{p.shape[1]}):")
        for row in p:
            print("  " + "".join("█" if c else "·" for c in row))


# ── Entry point: test against a screenshot file ───────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"Parsing screenshot: {path}\n")
        parser = BlockBlastParser(debug=True)
        board, pieces = parser.parse_image(path)
    else:
        # Live capture mode
        if not MSS_AVAILABLE:
            print("Install mss to use live capture: pip install mss")
            sys.exit(1)

        print("Starting live capture... Press Ctrl+C to stop.\n")
        parser = BlockBlastParser(region={
    "top": 271,      # y coordinate of the top-left corner
    "left": 1070,     # x coordinate of the top-left corner
    "width": 1430-1070,    # width of the region
    "height": 706-271,   # height of the region
}, debug=True)

        def on_state(board, pieces):
            print("\033[H\033[J", end="")  # clear terminal
            print_board(board)
            print()
            print_pieces(pieces)

        parser.run_loop(on_state, fps=2)
        sys.exit(0)

    print_board(board)
    print()
    print_pieces(pieces)
    
    


    
import time
board, pieces = parser.get_game_state()
while True:
    print(board)
    time.sleep(10)
