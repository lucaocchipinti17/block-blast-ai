"""
piece_bank_cv_test.py
=====================

Interactive test runner for piece-bank CV:
1) Load a frame (image or live desktop capture).
2) Select one piece-bank rectangle with mouse drag.
3) Auto-split that region into 3 horizontal piece slots.
4) Detect pieces and print their 5x5 drawings to terminal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Allow direct script execution: python client/tools/piece_bank_cv_test.py
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from client.vision.piece_bank_cv import (
    detect_piece_bank,
    draw_piece_bank_detections,
    select_piece_bank_region_with_mouse,
)
from client.core.pieces import ALL_PIECES

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

try:
    from PIL import ImageGrab
except Exception:  # noqa: BLE001
    ImageGrab = None


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is required. Install with: pip install opencv-python")


def _grab_desktop_frame() -> np.ndarray:
    if ImageGrab is None:
        raise RuntimeError("Pillow is required for screen capture. Install with: pip install pillow")
    img = ImageGrab.grab(all_screens=True)
    arr = np.array(img)
    # PIL is RGB; OpenCV expects BGR.
    return arr[:, :, ::-1].copy()


def _piece_to_5x5_lines(name: str) -> list[str]:
    piece = ALL_PIECES[name]
    return ["".join("1" if int(v) else "0" for v in row) for row in piece]


def _print_detection(matches) -> None:
    for m in matches:
        print(f"\nSlot {m.slot}:")
        if m.name is None:
            print(f"  UNKNOWN (confidence={m.confidence:.3f})")
            continue

        print(f"  Name: {m.name}  confidence={m.confidence:.3f}")
        print("  5x5:")
        for line in _piece_to_5x5_lines(m.name):
            print(f"    {line}")


def _draw_rois(frame: np.ndarray, rois) -> np.ndarray:
    _require_cv2()
    out = frame.copy()
    colors = [(255, 120, 80), (80, 220, 120), (80, 170, 255)]
    for i, (x, y, w, h) in enumerate(rois):
        color = colors[i % len(colors)]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            out,
            f"P{i+1}",
            (x + 6, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _build_frame_getter(image_path: Optional[str], use_screen: bool) -> Callable[[], np.ndarray]:
    _require_cv2()
    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        def get_image_frame() -> np.ndarray:
            return frame.copy()

        return get_image_frame

    if use_screen:
        return _grab_desktop_frame

    raise ValueError("Specify either --image or --screen.")


def main() -> None:
    _require_cv2()
    parser = argparse.ArgumentParser(description="Interactive piece-bank CV tester")
    parser.add_argument("--image", help="Path to screenshot/frame image")
    parser.add_argument("--screen", action="store_true", help="Capture desktop frame (for QuickTime mirroring)")
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--watch", action="store_true", help="Continuously rescan using selected ROI (screen mode)")
    parser.add_argument("--interval", type=float, default=0.75, help="Watch mode scan interval (seconds)")
    args = parser.parse_args()

    if bool(args.image) == bool(args.screen):
        raise ValueError("Choose exactly one source: --image or --screen")

    get_frame = _build_frame_getter(args.image, args.screen)
    frame0 = get_frame()

    print("Hover top-left and press Space, then hover bottom-right and press Space.")
    region, rois = select_piece_bank_region_with_mouse(frame0)
    print(f"Selected region: x={region[0]} y={region[1]} w={region[2]} h={region[3]}")
    print("Auto ROIs:")
    for i, (x, y, w, h) in enumerate(rois, start=1):
        print(f"  P{i}: x={x} y={y} w={w} h={h}")

    if args.watch and not args.screen:
        raise ValueError("--watch is only supported with --screen.")

    if not args.watch:
        frame = get_frame()
        matches = detect_piece_bank(frame, rois=rois, min_confidence=args.min_confidence)
        _print_detection(matches)
        annotated = draw_piece_bank_detections(frame, matches)
        annotated = _draw_rois(annotated, rois)
        cv2.imshow("Piece Bank CV Test", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Watch mode: press q or ESC in preview window to stop.
    print("\nWatch mode started. Press q or ESC in the preview window to stop.")
    while True:
        frame = get_frame()
        matches = detect_piece_bank(frame, rois=rois, min_confidence=args.min_confidence)

        print("\n--- scan ---")
        _print_detection(matches)

        annotated = draw_piece_bank_detections(frame, matches)
        annotated = _draw_rois(annotated, rois)
        cv2.imshow("Piece Bank CV Test", annotated)
        key = cv2.waitKey(max(1, int(args.interval * 1000))) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
