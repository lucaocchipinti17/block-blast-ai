"""
piece_bank_cv.py
================

Computer-vision utilities for reading a Block Blast 3-piece bank from an image
frame (for example, a screenshot from a USB-C phone stream).

This module is intentionally separate from the Tkinter planner UI so it can be
used in scripts, tests, or future integrations.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from board import Board
from pieces import ALL_PIECES

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None


BBox = Tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True)
class PieceMatch:
    slot: int
    name: Optional[str]
    key: Optional[str]
    confidence: float
    roi: BBox


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for piece-bank CV. Install with: pip install opencv-python"
        )


def _piece_footprint_key(piece: np.ndarray) -> str:
    cells, h, w = Board.get_footprint(piece)
    fp = np.zeros((h, w), dtype=np.uint8)
    for r, c in cells:
        fp[int(r), int(c)] = 1
    return "/".join("".join("1" if v else "0" for v in row) for row in fp)


def _build_key_catalog() -> Dict[str, List[str]]:
    by_key: Dict[str, List[str]] = {}
    for name, piece in sorted(ALL_PIECES.items()):
        key = _piece_footprint_key(piece)
        by_key.setdefault(key, []).append(name)

    def alias_rank(name: str) -> tuple[int, str]:
        # Prefer canonical H/V and non-alias names.
        if re.match(r"^I\d+_(0|90|180|270)$", name):
            return (1, name)
        return (0, name)

    for key in by_key:
        by_key[key].sort(key=alias_rank)
    return by_key


_KEY_TO_NAMES = _build_key_catalog()


def _key_to_grid(key: str) -> np.ndarray:
    rows = key.split("/")
    return np.array([[1 if c == "1" else 0 for c in row] for row in rows], dtype=np.uint8)


_KEY_TO_GRID = {k: _key_to_grid(k) for k in _KEY_TO_NAMES}
_KEY_CELL_COUNT = {k: int(v.sum()) for k, v in _KEY_TO_GRID.items()}


def default_piece_bank_rois(frame_shape: tuple[int, int, int]) -> List[BBox]:
    """
    Default ROI split for a typical phone portrait screenshot:
    uses the bottom band where the 3-piece inventory usually appears.
    """
    h, w = frame_shape[:2]
    y0 = int(0.66 * h)
    y1 = int(0.98 * h)
    band_w = w
    margin = int(0.04 * band_w)
    gap = int(0.03 * band_w)
    slot_w = int((band_w - (2 * margin) - (2 * gap)) / 3)
    slot_h = y1 - y0

    rois: List[BBox] = []
    for i in range(3):
        x0 = margin + i * (slot_w + gap)
        rois.append((x0, y0, slot_w, slot_h))
    return rois


def split_region_into_three_rois(region: BBox) -> List[BBox]:
    """
    Split a user-selected piece-bank region into 3 equal horizontal slot ROIs.
    """
    x, y, w, h = [int(v) for v in region]
    if w <= 0 or h <= 0:
        raise ValueError("Region must have positive width and height.")

    base = w // 3
    rem = w - (base * 3)
    widths = [base, base, base + rem]

    rois: List[BBox] = []
    cur_x = x
    for ww in widths:
        rois.append((cur_x, y, ww, h))
        cur_x += ww
    return rois


def _normalize_corner_rect(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    frame_w: int,
    frame_h: int,
) -> Optional[BBox]:
    x_min = max(0, min(x0, x1))
    y_min = max(0, min(y0, y1))
    x_max = min(frame_w - 1, max(x0, x1))
    y_max = min(frame_h - 1, max(y0, y1))
    w = int(x_max - x_min + 1)
    h = int(y_max - y_min + 1)
    if w <= 1 or h <= 1:
        return None
    return int(x_min), int(y_min), w, h


def _draw_hud_text(img: np.ndarray, text: str, x: int, y: int, scale: float = 0.55) -> None:
    _require_cv2()
    cv2.putText(
        img,
        text,
        (x + 1, y + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )


def _render_selection_preview(
    frame_bgr: np.ndarray,
    cursor: tuple[int, int],
    top_left: Optional[tuple[int, int]],
    rect: Optional[BBox],
) -> np.ndarray:
    _require_cv2()
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    cx = int(np.clip(cursor[0], 0, w - 1))
    cy = int(np.clip(cursor[1], 0, h - 1))

    # Cursor crosshair for pixel-accurate corner picking.
    cross_color = (255, 255, 255)
    cv2.line(canvas, (max(0, cx - 12), cy), (min(w - 1, cx + 12), cy), cross_color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, max(0, cy - 12)), (cx, min(h - 1, cy + 12)), cross_color, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 3, (0, 220, 255), -1, cv2.LINE_AA)

    if rect is None:
        if top_left is None:
            _draw_hud_text(canvas, "Hover TOP-LEFT corner and press Space", 16, 28)
            _draw_hud_text(canvas, "Space: set corner  |  R: reset  |  Esc/Q: cancel", 16, 54, scale=0.5)
        else:
            tx, ty = top_left
            cv2.circle(canvas, (tx, ty), 4, (0, 255, 255), -1, cv2.LINE_AA)
            _draw_hud_text(canvas, f"Top-left locked at ({tx}, {ty})", 16, 28)
            _draw_hud_text(canvas, "Hover BOTTOM-RIGHT corner and press Space", 16, 54, scale=0.5)

            preview_rect = _normalize_corner_rect(tx, ty, cx, cy, w, h)
            if preview_rect is not None:
                px, py, pw, ph = preview_rect
                overlay = (canvas * 0.35).astype(np.uint8)
                overlay[py : py + ph, px : px + pw] = canvas[py : py + ph, px : px + pw]
                canvas = overlay
                cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (0, 0, 0), 4)
                cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (0, 255, 255), 2)

        return canvas

    x, y, rw, rh = rect
    # Darken entire frame, then restore selected region for clear contrast.
    dark = (canvas * 0.35).astype(np.uint8)
    dark[y : y + rh, x : x + rw] = canvas[y : y + rh, x : x + rw]
    canvas = dark

    # Main selection rectangle: black outline + yellow stroke for visibility on blue UIs.
    cv2.rectangle(canvas, (x, y), (x + rw, y + rh), (0, 0, 0), 4)
    cv2.rectangle(canvas, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

    # Draw three horizontal slot guides.
    rois = split_region_into_three_rois(rect)
    slot_colors = [(255, 200, 80), (80, 255, 160), (80, 200, 255)]
    for idx, (sx, sy, sw, sh) in enumerate(rois, start=1):
        c = slot_colors[(idx - 1) % len(slot_colors)]
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), c, 1)
        _draw_hud_text(canvas, f"P{idx}", sx + 6, sy + 20, scale=0.5)

    _draw_hud_text(canvas, f"Region: x={x} y={y} w={rw} h={rh}", 16, 28)
    _draw_hud_text(canvas, "Selected (Space confirmed). R: reset  |  Esc/Q: cancel", 16, 54, scale=0.5)
    return canvas


def select_piece_bank_region_with_mouse(
    frame_bgr: np.ndarray,
    window_name: str = "Select Piece Bank Region",
) -> tuple[BBox, List[BBox]]:
    """
    Let user hover and press Space at top-left, then bottom-right corners.
    The resulting rectangle is auto-split into 3 horizontal slot ROIs.
    """
    _require_cv2()
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("Empty frame provided for ROI selection.")

    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    state: Dict[str, Optional[tuple[int, int] | BBox]] = {
        "cursor": (w // 2, h // 2),
        "top_left": None,
        "rect": None,
    }

    def on_mouse(event: int, x: int, y: int, flags: int, userdata) -> None:  # noqa: ARG001
        state["cursor"] = (int(x), int(y))
        if event == cv2.EVENT_LBUTTONDOWN and state["top_left"] is None:
            # Optional convenience: left-click can set top-left as well.
            state["top_left"] = (int(x), int(y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    selected: Optional[BBox] = None
    while True:
        cursor = state["cursor"]  # type: ignore[assignment]
        top_left = state["top_left"]  # type: ignore[assignment]
        preview_rect = selected or state["rect"]  # type: ignore[assignment]
        preview = _render_selection_preview(
            frame,
            cursor=cursor,  # type: ignore[arg-type]
            top_left=top_left,  # type: ignore[arg-type]
            rect=preview_rect,  # type: ignore[arg-type]
        )
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF

        if key == 32:  # Space
            cxy = state["cursor"]  # type: ignore[assignment]
            if cxy is None:
                continue
            cx, cy = cxy  # type: ignore[misc]
            if state["top_left"] is None:
                state["top_left"] = (int(cx), int(cy))
                continue

            tx, ty = state["top_left"]  # type: ignore[misc]
            candidate = _normalize_corner_rect(int(tx), int(ty), int(cx), int(cy), w, h)
            state["rect"] = candidate  # type: ignore[assignment]
            if candidate is not None:
                selected = candidate
                break

        if key in (ord("r"), ord("R")):
            state["top_left"] = None
            state["rect"] = None
            selected = None
            continue

        if key in (27, ord("q"), ord("Q")):
            # Ensure UI window is cleaned up even if caller shows another preview next.
            try:
                cv2.destroyWindow(window_name)
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError("ROI selection cancelled.")

    # Ensure UI window is cleaned up even if caller shows another preview next.
    try:
        cv2.destroyWindow(window_name)
    except Exception:  # noqa: BLE001
        pass

    if selected is None:
        raise RuntimeError("No region selected.")

    region = selected
    rois = split_region_into_three_rois(region)
    return region, rois


def _largest_component(mask: np.ndarray) -> np.ndarray:
    _require_cv2()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)
    # Skip background index 0.
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == best] = 255
    return out


def _component_count(mask: np.ndarray) -> int:
    _require_cv2()
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return max(0, int(num_labels) - 1)


def _extract_piece_mask(roi_bgr: np.ndarray) -> Optional[np.ndarray]:
    _require_cv2()
    if roi_bgr.size == 0:
        return None

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    # Keep colorful and bright pixels.
    color_mask = cv2.inRange(hsv, (0, 40, 45), (180, 255, 255))

    # Combine threshold styles for better robustness across themes/skins.
    mask = cv2.bitwise_or(mask_otsu, color_mask)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Piece cells may have tiny visual gaps between blocks. Bridge first, then
    # keep only the largest bridged region to avoid selecting a single block.
    bridge_k = max(3, int(round(min(roi_bgr.shape[:2]) * 0.02)))
    bridge_kernel = np.ones((bridge_k, bridge_k), dtype=np.uint8)
    bridged = cv2.dilate(mask, bridge_kernel, iterations=1)
    largest_bridged = _largest_component(bridged)
    mask = cv2.bitwise_and(mask, largest_bridged)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = mask[y0 : y1 + 1, x0 : x1 + 1]

    # Reject tiny detections (noise).
    if crop.size == 0:
        return None
    if float(np.count_nonzero(crop)) < 0.01 * float(roi_bgr.shape[0] * roi_bgr.shape[1]):
        return None
    return crop


def _sample_grid(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w = mask.shape[:2]
    y_edges = np.linspace(0, h, rows + 1).astype(int)
    x_edges = np.linspace(0, w, cols + 1).astype(int)
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        y0, y1 = int(y_edges[r]), int(y_edges[r + 1])
        if y1 <= y0:
            continue
        for c in range(cols):
            x0, x1 = int(x_edges[c]), int(x_edges[c + 1])
            if x1 <= x0:
                continue
            cell = mask[y0:y1, x0:x1]
            out[r, c] = float(np.mean(cell > 0))
    return out


def _candidate_score(mask_crop: np.ndarray, target: np.ndarray, observed_components: int) -> float:
    rows, cols = target.shape
    sampled = _sample_grid(mask_crop, rows, cols)

    best = 0.0
    for threshold in (0.18, 0.24, 0.30, 0.36, 0.44):
        pred = (sampled >= threshold).astype(np.uint8)
        inter = int(np.logical_and(pred == 1, target == 1).sum())
        union = int(np.logical_or(pred == 1, target == 1).sum())
        if union <= 0:
            continue
        iou = float(inter) / float(union)
        count_penalty = 1.0 - (
            abs(int(pred.sum()) - int(target.sum())) / float(max(int(target.sum()), 1))
        )
        score = (0.8 * iou) + (0.2 * max(count_penalty, 0.0))
        best = max(best, score)

    # Shape-ratio penalty helps separate near-collisions.
    mh, mw = mask_crop.shape[:2]
    target_ratio = float(cols) / float(max(rows, 1))
    mask_ratio = float(mw) / float(max(mh, 1))
    ratio_penalty = math.exp(-abs(math.log((mask_ratio + 1e-6) / (target_ratio + 1e-6))))

    # Penalize candidates whose expected fill fraction is very different from
    # observed mask density inside the crop.
    mask_fill = float(np.mean(mask_crop > 0))
    target_fill = float(target.sum()) / float(max(rows * cols, 1))
    fill_penalty = math.exp(-4.0 * abs(mask_fill - target_fill))

    component_penalty = 1.0
    if observed_components > 1:
        expected_components = int(target.sum())
        component_penalty = math.exp(
            -1.25 * abs(float(observed_components - expected_components))
            / float(max(expected_components, 1))
        )

    return float(best * ratio_penalty * fill_penalty * component_penalty)


def _candidate_scores(mask_crop: np.ndarray) -> List[tuple[str, float]]:
    observed_components = _component_count(mask_crop)
    scored: List[tuple[str, float]] = []
    for key, grid in _KEY_TO_GRID.items():
        scored.append((key, _candidate_score(mask_crop, grid, observed_components=observed_components)))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored


def match_piece_from_roi(
    roi_bgr: np.ndarray,
    min_confidence: float = 0.45,
) -> tuple[Optional[str], Optional[str], float]:
    """
    Returns (piece_name, key, confidence) for one piece slot ROI.
    Returns (None, None, confidence) if no confident match is found.
    """
    mask = _extract_piece_mask(roi_bgr)
    if mask is None:
        return None, None, 0.0

    ranked = _candidate_scores(mask)
    best_key, best_score = ranked[0]

    if best_key is None or best_score < min_confidence:
        return None, None, max(0.0, best_score)

    # Pick canonical name for that key.
    name = _KEY_TO_NAMES[best_key][0]
    return name, best_key, float(best_score)


def detect_piece_bank(
    frame_bgr: np.ndarray,
    rois: Optional[List[BBox]] = None,
    min_confidence: float = 0.45,
) -> List[PieceMatch]:
    """
    Detect the 3-piece inventory from a frame.

    Parameters
    ----------
    frame_bgr:
        OpenCV BGR image.
    rois:
        Optional list of exactly 3 (x, y, w, h) piece slot regions.
        If omitted, a default bottom-band split is used.
    min_confidence:
        Minimum score to accept a key match.
    """
    _require_cv2()
    if rois is None:
        rois = default_piece_bank_rois(frame_bgr.shape)
    if len(rois) != 3:
        raise ValueError("detect_piece_bank expects exactly 3 ROIs.")

    h, w = frame_bgr.shape[:2]
    slot_obs: List[dict] = []
    for slot, (x, y, rw, rh) in enumerate(rois, start=1):
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w, x0 + max(1, int(rw)))
        y1 = min(h, y0 + max(1, int(rh)))
        roi = frame_bgr[y0:y1, x0:x1]

        mask = _extract_piece_mask(roi)
        if mask is None:
            slot_obs.append(
                {
                    "slot": slot,
                    "roi": (x0, y0, x1 - x0, y1 - y0),
                    "mask": None,
                    "area_px": 0,
                    "ranked": [],
                }
            )
            continue

        ranked = _candidate_scores(mask)
        slot_obs.append(
            {
                "slot": slot,
                "roi": (x0, y0, x1 - x0, y1 - y0),
                "mask": mask,
                "area_px": int(np.count_nonzero(mask)),
                "ranked": ranked,
            }
        )

    observed = [o for o in slot_obs if o["mask"] is not None]
    selected_key_by_slot: Dict[int, tuple[Optional[str], float]] = {
        o["slot"]: (None, 0.0) for o in slot_obs
    }

    if observed:
        top_k = 12
        per_slot_top = [o["ranked"][:top_k] for o in observed]
        best_combo_score = -1.0
        best_combo = None

        for combo in itertools.product(*per_slot_top):
            # combo is tuple[(key, score)] aligned with observed slots.
            base_sum = sum(score for _, score in combo)
            if base_sum <= 0:
                continue

            est_cell_sizes = []
            for obs, (key, _) in zip(observed, combo):
                n = max(1, _KEY_CELL_COUNT[key])
                est_cell_sizes.append(math.sqrt(float(obs["area_px"]) / float(n)))

            mean_size = float(np.mean(est_cell_sizes))
            if mean_size <= 1e-6:
                continue
            rel_std = float(np.std(est_cell_sizes) / mean_size)
            consistency = math.exp(-3.0 * rel_std)
            # Keep per-slot quality primary; use consistency as a soft tie-breaker.
            combo_score = base_sum * (0.75 + (0.25 * consistency))
            if combo_score > best_combo_score:
                best_combo_score = combo_score
                best_combo = combo

        if best_combo is None:
            # Fallback to per-slot best.
            for obs in observed:
                key, score = obs["ranked"][0]
                selected_key_by_slot[obs["slot"]] = (key, float(score))
        else:
            for obs, (key, score) in zip(observed, best_combo):
                selected_key_by_slot[obs["slot"]] = (key, float(score))

    out: List[PieceMatch] = []
    for obs in slot_obs:
        slot = int(obs["slot"])
        key, conf = selected_key_by_slot[slot]
        if key is None or conf < min_confidence:
            out.append(PieceMatch(slot=slot, name=None, key=None, confidence=float(conf), roi=obs["roi"]))
            continue
        name = _KEY_TO_NAMES[key][0]
        out.append(PieceMatch(slot=slot, name=name, key=key, confidence=float(conf), roi=obs["roi"]))
    return out


def draw_piece_bank_detections(frame_bgr: np.ndarray, matches: List[PieceMatch]) -> np.ndarray:
    _require_cv2()
    canvas = frame_bgr.copy()
    for m in matches:
        x, y, w, h = m.roi
        ok = m.name is not None
        color = (80, 200, 80) if ok else (70, 70, 240)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        label = f"P{m.slot}: {m.name or 'UNKNOWN'} ({m.confidence:.2f})"
        cv2.putText(
            canvas,
            label,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas


def _parse_roi(spec: str) -> BBox:
    try:
        parts = [int(p.strip()) for p in spec.split(",")]
        if len(parts) != 4:
            raise ValueError
        return parts[0], parts[1], parts[2], parts[3]
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Invalid ROI '{spec}'. Use x,y,w,h") from exc


def main() -> None:
    _require_cv2()
    parser = argparse.ArgumentParser(description="Detect 3-piece bank from a screenshot/frame.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--roi",
        action="append",
        default=[],
        help="Piece slot ROI as x,y,w,h (repeat 3 times). If omitted, defaults are used.",
    )
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Minimum match score to accept a piece")
    parser.add_argument("--show", action="store_true", help="Show annotated preview window")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    rois = [_parse_roi(r) for r in args.roi] if args.roi else None
    matches = detect_piece_bank(frame, rois=rois, min_confidence=args.min_confidence)

    for m in matches:
        print(
            f"slot={m.slot}  name={m.name or 'UNKNOWN':<12s}  "
            f"key={m.key or '-':<12s}  conf={m.confidence:.3f}"
        )

    if args.show:
        annotated = draw_piece_bank_detections(frame, matches)
        cv2.imshow("piece-bank-detections", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
