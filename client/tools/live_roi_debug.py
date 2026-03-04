"""
live_roi_debug.py
=================

Debug viewer for live piece-bank ROI capture.

Shows:
- Full captured window frame
- Bank ROI rectangle
- The 3 slot rectangles (left/mid/right)
- Live cropped previews of each slot

Use this to verify that the ROI split is correct before running detection/planning.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Allow direct script execution: python client/tools/live_roi_debug.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from client.vision.stream_capture import CaptureConfig, WindowCaptureLoop


def parse_norm_roi(raw: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x0,y0,x1,y1")
    x0, y0, x1, y1 = [float(p) for p in parts]
    if not (0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0 and 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0):
        raise ValueError("ROI values must be in [0,1]")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("ROI must satisfy x1>x0 and y1>y0")
    return x0, y0, x1, y1


def bank_bbox_from_norm(frame_shape: tuple[int, int, int], roi: Tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x0n, y0n, x1n, y1n = roi
    x0 = int(np.clip(round(x0n * w), 0, w - 1))
    y0 = int(np.clip(round(y0n * h), 0, h - 1))
    x1 = int(np.clip(round(x1n * w), 0, w - 1))
    y1 = int(np.clip(round(y1n * h), 0, h - 1))
    if x1 <= x0:
        x1 = min(w - 1, x0 + 1)
    if y1 <= y0:
        y1 = min(h - 1, y0 + 1)
    return x0, y0, x1 - x0, y1 - y0


def split_3(bank_bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    x, y, w, h = bank_bbox
    base = w // 3
    rem = w - (3 * base)
    ws = [base, base, base + rem]
    out = []
    cx = x
    for ww in ws:
        out.append((cx, y, ww, h))
        cx += ww
    return out


def draw_hud(frame: np.ndarray, text: str, x: int, y: int, color=(240, 240, 240)) -> None:
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def build_slot_strip(frame: np.ndarray, rois: list[tuple[int, int, int, int]], strip_h: int = 170) -> np.ndarray:
    h, w = frame.shape[:2]
    strip = np.zeros((strip_h, w, 3), dtype=np.uint8)
    colors = [(255, 180, 80), (80, 220, 120), (80, 180, 255)]
    pad = 10
    sw = max(50, (w - (4 * pad)) // 3)
    sh = strip_h - (2 * pad) - 24

    for i, (x, y, rw, rh) in enumerate(rois):
        color = colors[i % 3]
        crop = frame[max(0, y) : min(h, y + rh), max(0, x) : min(w, x + rw)]
        if crop.size == 0:
            view = np.zeros((sh, sw, 3), dtype=np.uint8)
        else:
            view = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)

        x0 = pad + i * (sw + pad)
        y0 = pad + 20
        strip[y0 : y0 + sh, x0 : x0 + sw] = view
        cv2.rectangle(strip, (x0, y0), (x0 + sw, y0 + sh), color, 2)
        draw_hud(strip, f"P{i+1} roi=({x},{y},{rw},{rh})", x0, 16, color=color)

    return strip


def main() -> None:
    parser = argparse.ArgumentParser(description="Live ROI debug viewer for piece-bank capture")
    parser.add_argument("--title", default="Movie Recording", help="Window title contains text")
    parser.add_argument(
        "--roi",
        default="0.0633,0.6945,0.9221,0.8135",
        help="Normalized bank ROI x0,y0,x1,y1 in captured window",
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Capture target FPS")
    parser.add_argument("--window-refresh", type=float, default=1.0, help="Seconds between window re-locate")
    args = parser.parse_args()

    roi = parse_norm_roi(args.roi)
    cap = WindowCaptureLoop(
        CaptureConfig(
            window_title_contains=args.title,
            target_fps=float(args.fps),
            window_refresh_sec=float(args.window_refresh),
        )
    )
    cap.start()

    win_name = "ROI Debug - Piece Bank"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1200, 900)

    last_save_t = 0.0
    try:
        while True:
            frame = cap.get_latest_frame(copy=True)
            stats = cap.get_stats()

            if frame is None:
                view = np.zeros((420, 900, 3), dtype=np.uint8)
                draw_hud(view, "Waiting for capture frame...", 20, 40, color=(130, 180, 255))
                if stats.last_error:
                    draw_hud(view, stats.last_error, 20, 74, color=(120, 120, 255))
                cv2.imshow(win_name, view)
                key = cv2.waitKey(20) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                continue

            bank = bank_bbox_from_norm(frame.shape, roi)
            slots = split_3(bank)

            draw = frame.copy()
            bx, by, bw, bh = bank
            cv2.rectangle(draw, (bx, by), (bx + bw, by + bh), (0, 255, 255), 3)
            colors = [(255, 180, 80), (80, 220, 120), (80, 180, 255)]
            for i, (x, y, w, h) in enumerate(slots):
                cv2.rectangle(draw, (x, y), (x + w, y + h), colors[i], 2)
                draw_hud(draw, f"P{i+1}", x + 6, y + 20, color=colors[i])

            draw_hud(draw, f"title~'{args.title}'", 16, 26)
            draw_hud(draw, f"roi={args.roi}", 16, 52)
            draw_hud(draw, f"capture_fps={stats.measured_fps:.1f}", 16, 78)
            if stats.last_error:
                draw_hud(draw, f"err: {stats.last_error}", 16, 104, color=(120, 120, 255))
            draw_hud(draw, "q/esc quit | s save snapshot", 16, 130, color=(200, 200, 200))

            strip = build_slot_strip(frame, slots, strip_h=175)
            canvas = np.vstack([draw, strip])
            cv2.imshow(win_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                now = time.time()
                if now - last_save_t > 0.4:
                    out = f"/tmp/roi_debug_{int(now)}.png"
                    cv2.imwrite(out, canvas)
                    last_save_t = now

    finally:
        cap.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
