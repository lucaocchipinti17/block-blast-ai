"""
piece_bank_detector.py
======================

Classical CV detector for a 3-slot Block Blast piece bank.

No image templates or manual labels are used:
- Detect piece cells from color/brightness segmentation
- Split merged blobs with distance-transform watershed when needed
- Convert to occupancy matrix
- Match by geometric footprint key against the existing piece catalog
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from client.core.board import Board
from client.core.pieces import ALL_PIECES

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None


BBox = Tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True)
class PieceBankDetectorConfig:
    # Normalized ROI in window-frame coordinates: (x0, y0, x1, y1)
    bank_roi_norm_xyxy: tuple[float, float, float, float] = (0.09, 0.70, 0.91, 0.88)
    # Expand normalized ROI by these margins (x, y) to tolerate approximate calibration.
    bank_roi_expand_norm_xy: tuple[float, float] = (0.015, 0.012)

    sat_min: int = 35
    val_min: int = 55
    bright_val_min: int = 160

    min_component_area_px: int = 40
    max_component_area_ratio: float = 0.35
    aspect_min: float = 0.35
    aspect_max: float = 2.8

    cluster_tol_factor: float = 0.60
    min_match_iou: float = 0.52
    debug_visuals: bool = False

    # Known block size for a single occupied cell in each piece-slot render.
    expected_cell_px: float = 20.0
    # Pixel tolerance when clustering detected cell centers into rows/columns.
    cell_cluster_tol_px: float = 9.0
    # Valid connected-component area range relative to expected_cell_px^2.
    cell_area_min_ratio: float = 0.14
    cell_area_max_ratio: float = 2.60


@dataclass(frozen=True)
class DetectedPiece:
    slot_idx: int
    name: Optional[str]
    key: Optional[str]
    confidence: float
    piece_array_5x5: np.ndarray
    tight_matrix: np.ndarray
    slot_bbox: BBox


@dataclass(frozen=True)
class PieceBankDetection:
    pieces: List[DetectedPiece]
    bank_bbox: BBox
    slot_bboxes: List[BBox]
    debug_frame: Optional[np.ndarray]
    processing_ms: float


def _require_cv2() -> None:
    if cv2 is None:
        raise RuntimeError("opencv-python is required. Install with: pip install opencv-python")


def _footprint_key_from_array(piece: np.ndarray) -> str:
    cells, h, w = Board.get_footprint(piece)
    fp = np.zeros((h, w), dtype=np.uint8)
    for r, c in cells:
        fp[int(r), int(c)] = 1
    return "/".join("".join("1" if int(v) else "0" for v in row) for row in fp)


def _grid_from_key(key: str) -> np.ndarray:
    rows = key.split("/")
    return np.array([[1 if c == "1" else 0 for c in row] for row in rows], dtype=np.uint8)


def _matrix_key(mat: np.ndarray) -> str:
    return "/".join("".join("1" if int(v) else "0" for v in row) for row in mat)


def _trim_binary(mat: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mat > 0)
    if len(xs) == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return mat[y0 : y1 + 1, x0 : x1 + 1].astype(np.uint8)


def _to_5x5(mat: np.ndarray) -> np.ndarray:
    out = np.zeros((5, 5), dtype=np.int8)
    if mat.size == 0:
        return out
    h, w = mat.shape
    h = min(5, h)
    w = min(5, w)
    out[:h, :w] = mat[:h, :w].astype(np.int8)
    return out


def _iou_grid(a: np.ndarray, b: np.ndarray) -> float:
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    pa = np.zeros((h, w), dtype=np.uint8)
    pb = np.zeros((h, w), dtype=np.uint8)
    pa[: a.shape[0], : a.shape[1]] = a
    pb[: b.shape[0], : b.shape[1]] = b
    inter = int(np.logical_and(pa > 0, pb > 0).sum())
    union = int(np.logical_or(pa > 0, pb > 0).sum())
    return float(inter) / float(union) if union > 0 else 0.0


def _cluster_1d(vals: np.ndarray, tol: float) -> List[float]:
    if len(vals) == 0:
        return []
    sorted_vals = sorted(float(v) for v in vals)
    groups: List[List[float]] = [[sorted_vals[0]]]
    for v in sorted_vals[1:]:
        gmean = float(np.mean(groups[-1]))
        if abs(v - gmean) <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.mean(g)) for g in groups]


def _merge_clusters_to_limit(clusters: List[float], limit: int) -> List[float]:
    c = [float(v) for v in clusters]
    if len(c) <= limit:
        return sorted(c)
    while len(c) > limit:
        c.sort()
        gaps = [c[i + 1] - c[i] for i in range(len(c) - 1)]
        k = int(np.argmin(gaps))
        merged = 0.5 * (c[k] + c[k + 1])
        c = c[:k] + [merged] + c[k + 2 :]
    return sorted(c)


class PieceBankDetector:
    """
    Detects 3 piece-slot shapes from a captured window frame.
    """

    def __init__(self, config: PieceBankDetectorConfig):
        _require_cv2()
        self.cfg = config

        self.key_to_names: Dict[str, List[str]] = {}
        for name, piece in sorted(ALL_PIECES.items()):
            key = _footprint_key_from_array(piece)
            self.key_to_names.setdefault(key, []).append(name)

        def alias_rank(name: str) -> tuple[int, str]:
            # Prefer non-I-angle aliases, then alphabetical.
            if re.match(r"^I\d+_(0|90|180|270)$", name):
                return (1, name)
            return (0, name)

        for key in self.key_to_names:
            self.key_to_names[key].sort(key=alias_rank)

        self.key_to_grid = {k: _grid_from_key(k) for k in self.key_to_names}

    def detect_piece_bank(self, frame_bgr: np.ndarray) -> PieceBankDetection:
        t0 = cv2.getTickCount()

        bank_bbox = self._bank_bbox_from_norm(frame_bgr.shape)
        slot_bboxes = self._split_into_slots(bank_bbox)
        pieces: List[DetectedPiece] = []

        debug_frame = frame_bgr.copy() if self.cfg.debug_visuals else None
        for slot_idx, bbox in enumerate(slot_bboxes, start=1):
            x, y, w, h = bbox
            slot = frame_bgr[y : y + h, x : x + w]
            det = self._detect_slot(slot, slot_idx=slot_idx, bbox=bbox)
            pieces.append(det)

            if debug_frame is not None:
                color = (70, 220, 70) if det.name is not None else (60, 60, 240)
                cv2.rectangle(debug_frame, (x, y), (x + w, y + h), color, 2)
                label = f"P{slot_idx}: {det.name or 'UNKNOWN'} {det.confidence:.2f}"
                cv2.putText(
                    debug_frame,
                    label,
                    (x + 4, max(16, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        elapsed_ms = (cv2.getTickCount() - t0) * 1000.0 / cv2.getTickFrequency()
        return PieceBankDetection(
            pieces=pieces,
            bank_bbox=bank_bbox,
            slot_bboxes=slot_bboxes,
            debug_frame=debug_frame,
            processing_ms=float(elapsed_ms),
        )

    def _bank_bbox_from_norm(self, shape: tuple[int, int, int]) -> BBox:
        h, w = shape[:2]
        x0n, y0n, x1n, y1n = self.cfg.bank_roi_norm_xyxy
        ex, ey = self.cfg.bank_roi_expand_norm_xy

        x0n = float(np.clip(x0n - ex, 0.0, 1.0))
        y0n = float(np.clip(y0n - ey, 0.0, 1.0))
        x1n = float(np.clip(x1n + ex, 0.0, 1.0))
        y1n = float(np.clip(y1n + ey, 0.0, 1.0))

        x0 = int(np.clip(round(x0n * w), 0, w - 1))
        y0 = int(np.clip(round(y0n * h), 0, h - 1))
        x1 = int(np.clip(round(x1n * w), 0, w - 1))
        y1 = int(np.clip(round(y1n * h), 0, h - 1))
        if x1 <= x0:
            x1 = min(w - 1, x0 + 1)
        if y1 <= y0:
            y1 = min(h - 1, y0 + 1)
        return x0, y0, x1 - x0, y1 - y0

    @staticmethod
    def _split_into_slots(bank_bbox: BBox) -> List[BBox]:
        x, y, w, h = bank_bbox
        base = w // 3
        rem = w - (base * 3)
        widths = [base, base, base + rem]
        rois: List[BBox] = []
        cx = x
        for ww in widths:
            rois.append((cx, y, ww, h))
            cx += ww
        return rois

    def _detect_slot(self, slot_bgr: np.ndarray, slot_idx: int, bbox: BBox) -> DetectedPiece:
        if slot_bgr.size == 0:
            return DetectedPiece(
                slot_idx=slot_idx,
                name=None,
                key=None,
                confidence=0.0,
                piece_array_5x5=np.zeros((5, 5), dtype=np.int8),
                tight_matrix=np.zeros((0, 0), dtype=np.uint8),
                slot_bbox=bbox,
            )

        mask = self._piece_mask(slot_bgr)
        candidates_by_key: Dict[str, Tuple[DetectedPiece, int]] = {}
        expected = max(6.0, float(self.cfg.expected_cell_px))
        mask_bbox = self._binary_bbox(mask)

        def register_candidate(occ: np.ndarray, added_cells: int = 0) -> None:
            det = self._candidate_from_occupancy(occ, slot_idx=slot_idx, bbox=bbox)
            if det is None or det.name is None:
                return
            key = _matrix_key(det.tight_matrix)
            prev = candidates_by_key.get(key)
            if prev is None:
                candidates_by_key[key] = (det, int(added_cells))
                return
            prev_det, prev_added = prev
            if int(added_cells) < int(prev_added):
                candidates_by_key[key] = (det, int(added_cells))
                return
            if int(added_cells) == int(prev_added) and det.confidence > prev_det.confidence:
                candidates_by_key[key] = (det, int(added_cells))

        # Path A: cell-size-aware extraction (uses known single-cell pixel size).
        cell_aware_occ = self._cell_aware_occupancy(slot_bgr, mask)
        register_candidate(cell_aware_occ, added_cells=0)
        for expanded in self._expanded_neighbor_mats(cell_aware_occ, max_add=1):
            register_candidate(expanded, added_cells=1)

        # Path B: legacy component clustering fallback.
        components = self._extract_components(mask, slot_bgr)
        legacy_occ = self._components_to_occupancy(components)
        register_candidate(legacy_occ, added_cells=0)
        for expanded in self._expanded_neighbor_mats(legacy_occ, max_add=1):
            register_candidate(expanded, added_cells=1)

        candidates = list(candidates_by_key.values())
        if not candidates:
            empty = np.zeros((5, 5), dtype=np.int8)
            return DetectedPiece(
                slot_idx=slot_idx,
                name=None,
                key=None,
                confidence=0.0,
                piece_array_5x5=empty,
                tight_matrix=np.zeros((0, 0), dtype=np.uint8),
                slot_bbox=bbox,
            )

        # Rank by confidence plus dimensional fit to observed pixel bbox.
        candidates.sort(
            key=lambda item: (
                (0.45 * float(item[0].confidence))
                + (0.45 * self._best_shape_overlap_score(item[0].tight_matrix.astype(np.uint8), mask, expected))
                + (0.10 * self._dimension_fit_score(item[0].tight_matrix.astype(np.uint8), mask_bbox, expected))
                - (0.045 * float(item[1])),
                float(item[0].confidence),
                -int(item[1]),
                -int(item[0].tight_matrix.sum()),
            ),
            reverse=True,
        )
        return candidates[0][0]

    def _piece_mask(self, slot_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        color_mask = ((s >= self.cfg.sat_min) & (v >= self.cfg.val_min))
        bright_mask = ((v >= self.cfg.bright_val_min) & (s >= max(10, self.cfg.sat_min // 2)))
        mask = np.where(color_mask | bright_mask, 255, 0).astype(np.uint8)

        k = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
        return mask

    def _candidate_from_occupancy(
        self,
        occupancy: np.ndarray,
        slot_idx: int,
        bbox: BBox,
    ) -> Optional[DetectedPiece]:
        tight = _trim_binary(occupancy)
        if tight.size == 0 or int(tight.sum()) == 0:
            return None

        piece_5 = _to_5x5(tight)
        match_name, match_key, match_conf = self._match_tight_matrix(tight)
        return DetectedPiece(
            slot_idx=slot_idx,
            name=match_name,
            key=match_key,
            confidence=float(match_conf),
            piece_array_5x5=piece_5.astype(np.int8),
            tight_matrix=tight.astype(np.uint8),
            slot_bbox=bbox,
        )

    def _cell_aware_occupancy(self, slot_bgr: np.ndarray, soft_mask: np.ndarray) -> np.ndarray:
        expected = max(6.0, float(self.cfg.expected_cell_px))
        if int(np.count_nonzero(soft_mask)) <= 0:
            return np.zeros((0, 0), dtype=np.uint8)

        piece_mask = self._retain_dominant_components(
            soft_mask,
            min_area=max(12.0, expected * expected * 0.08),
            rel_area_to_largest=0.25,
        )
        if int(np.count_nonzero(piece_mask)) <= 0:
            return np.zeros((0, 0), dtype=np.uint8)

        core_mask = self._core_mask_for_cells(slot_bgr, piece_mask, expected_cell=expected)

        comp_centers = self._component_cell_centers(core_mask, expected_cell=expected)
        comp_xy = [(x, y) for x, y, _ in comp_centers]
        occ_comp = self._centers_to_occupancy(comp_xy, expected_cell=expected)

        candidates: List[np.ndarray] = []
        if occ_comp.size > 0 and 0 < int(occ_comp.sum()) <= 9:
            candidates.append(occ_comp)

        # When component extraction under-segments, distance peaks can recover missing cells.
        peak_centers = self._distance_peak_centers(core_mask, expected_cell=expected)
        if len(comp_centers) <= 2:
            peak_centers.extend(self._distance_peak_centers(piece_mask, expected_cell=expected))

        peak_xy = self._merge_center_candidates(
            peak_centers,
            min_dist=max(4.0, expected * 0.66),
            limit=9,
        )
        occ_peak = self._centers_to_occupancy(peak_xy, expected_cell=expected)
        if occ_peak.size > 0 and 0 < int(occ_peak.sum()) <= 9:
            candidates.append(occ_peak)

        merged_xy = self._merge_center_candidates(
            comp_centers + peak_centers,
            min_dist=max(4.0, expected * 0.62),
            limit=9,
        )
        occ_merged = self._centers_to_occupancy(merged_xy, expected_cell=expected)
        if occ_merged.size > 0 and 0 < int(occ_merged.sum()) <= 9:
            candidates.append(occ_merged)

        if not candidates:
            return np.zeros((0, 0), dtype=np.uint8)

        best_occ = np.zeros((0, 0), dtype=np.uint8)
        best_score = -1.0
        bbox = self._binary_bbox(piece_mask)
        for occ in candidates:
            tight = _trim_binary(occ)
            if tight.size == 0 or int(tight.sum()) <= 0:
                continue
            _, _, conf = self._match_tight_matrix(tight)
            cells = int(tight.sum())
            fit = self._dimension_fit_score(tight.astype(np.uint8), bbox, expected)
            overlap = self._best_shape_overlap_score(tight.astype(np.uint8), piece_mask, expected)
            # Favor catalog confidence and pixel-size agreement; slight penalty for larger cell counts.
            score = (
                (0.52 * float(conf))
                + (0.34 * float(overlap))
                + (0.14 * float(fit))
                - (0.010 * float(max(0, cells - 4)))
            )
            if score > best_score:
                best_score = score
                best_occ = tight.astype(np.uint8)
        return best_occ

    def _retain_dominant_components(
        self,
        binary: np.ndarray,
        min_area: float,
        rel_area_to_largest: float,
    ) -> np.ndarray:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num <= 1:
            return binary.copy()

        areas = [float(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
        largest = max(areas) if areas else 0.0
        area_floor = max(float(min_area), largest * max(0.0, float(rel_area_to_largest)))

        out = np.zeros_like(binary)
        for i in range(1, num):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < area_floor:
                continue
            out[labels == i] = 255

        if int(np.count_nonzero(out)) == 0:
            return binary.copy()
        return out

    def _core_mask_for_cells(self, slot_bgr: np.ndarray, piece_mask: np.ndarray, expected_cell: float) -> np.ndarray:
        hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Primary pass: bright core pixels (suppresses glow bridges between adjacent cells).
        v_thresh = max(140, int(np.percentile(v, 90)))
        s_thresh = max(int(self.cfg.sat_min), int(np.percentile(s, 50)))
        core = np.where((v >= v_thresh) & (s >= s_thresh), 255, 0).astype(np.uint8)
        core = cv2.bitwise_and(core, piece_mask)
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)

        min_pixels = int(max(10.0, expected_cell * expected_cell * 0.12))
        if int(np.count_nonzero(core)) >= min_pixels:
            return core

        # Secondary pass: slightly looser threshold if the slot is dim.
        v_thresh2 = max(int(self.cfg.val_min + 18), int(np.percentile(v, 82)))
        s_thresh2 = max(int(self.cfg.sat_min), int(np.percentile(s, 45)))
        fallback = np.where((v >= v_thresh2) & (s >= s_thresh2), 255, 0).astype(np.uint8)
        fallback = cv2.bitwise_and(fallback, piece_mask)
        fallback = cv2.morphologyEx(fallback, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)
        if int(np.count_nonzero(fallback)) >= min_pixels:
            return fallback

        return piece_mask.copy()

    def _component_cell_centers(
        self,
        binary: np.ndarray,
        expected_cell: float,
    ) -> List[Tuple[float, float, float]]:
        num, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num <= 1:
            return []

        expected_area = max(25.0, expected_cell * expected_cell)
        min_area = expected_area * max(0.01, float(self.cfg.cell_area_min_ratio))
        max_area = expected_area * max(1.0, float(self.cfg.cell_area_max_ratio))

        centers: List[Tuple[float, float, float]] = []
        for i in range(1, num):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            cx = float(centroids[i, 0])
            cy = float(centroids[i, 1])
            centers.append((cx, cy, area))
        return centers

    def _distance_peak_centers(
        self,
        binary: np.ndarray,
        expected_cell: float,
    ) -> List[Tuple[float, float, float]]:
        if int(np.count_nonzero(binary)) <= 0:
            return []
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        mx = float(dist.max())
        if mx <= 1e-6:
            return []

        # Local maxima over distance map approximate cell centers.
        local_max = dist >= cv2.dilate(dist, np.ones((3, 3), dtype=np.float32))
        radius_floor = max(2.2, expected_cell * 0.30)
        peak_mask = local_max & (dist >= radius_floor)
        ys, xs = np.where(peak_mask)
        if len(xs) == 0:
            return []

        points = [(float(xs[i]), float(ys[i]), float(dist[ys[i], xs[i]])) for i in range(len(xs))]
        points.sort(key=lambda t: t[2], reverse=True)

        out: List[Tuple[float, float, float]] = []
        min_dist = max(4.0, expected_cell * 0.78)
        min_dist_sq = min_dist * min_dist
        for x, y, score in points:
            if all((x - ox) * (x - ox) + (y - oy) * (y - oy) >= min_dist_sq for ox, oy, _ in out):
                out.append((x, y, score))
            if len(out) >= 12:
                break
        return out

    @staticmethod
    def _merge_center_candidates(
        candidates: List[Tuple[float, float, float]],
        min_dist: float,
        limit: int,
    ) -> List[Tuple[float, float]]:
        if not candidates:
            return []
        points = sorted(candidates, key=lambda t: float(t[2]), reverse=True)
        out: List[Tuple[float, float]] = []
        min_dist_sq = float(min_dist) * float(min_dist)
        for x, y, _ in points:
            if all((x - ox) * (x - ox) + (y - oy) * (y - oy) >= min_dist_sq for ox, oy in out):
                out.append((float(x), float(y)))
            if len(out) >= int(limit):
                break
        return out

    def _centers_to_occupancy(
        self,
        centers_xy: List[Tuple[float, float]],
        expected_cell: float,
    ) -> np.ndarray:
        if not centers_xy:
            return np.zeros((0, 0), dtype=np.uint8)

        xs = np.array([p[0] for p in centers_xy], dtype=np.float32)
        ys = np.array([p[1] for p in centers_xy], dtype=np.float32)

        tol_cfg = float(self.cfg.cell_cluster_tol_px)
        tol = tol_cfg if tol_cfg > 0 else expected_cell * 0.45
        tol = max(2.0, min(tol, expected_cell * 0.75))

        col_clusters = _cluster_1d(xs, tol=tol)
        row_clusters = _cluster_1d(ys, tol=tol)
        col_clusters = _merge_clusters_to_limit(col_clusters, 5)
        row_clusters = _merge_clusters_to_limit(row_clusters, 5)

        if len(col_clusters) == 0 or len(row_clusters) == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        occ = np.zeros((len(row_clusters), len(col_clusters)), dtype=np.uint8)
        for cx, cy in centers_xy:
            col = int(np.argmin([abs(cx - cc) for cc in col_clusters]))
            row = int(np.argmin([abs(cy - rr) for rr in row_clusters]))
            occ[row, col] = 1

        occ = _trim_binary(occ)
        if occ.shape[0] > 5 or occ.shape[1] > 5:
            return np.zeros((0, 0), dtype=np.uint8)
        return occ.astype(np.uint8)

    @staticmethod
    def _binary_bbox(binary: np.ndarray) -> Optional[BBox]:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return None
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return x0, y0, x1 - x0 + 1, y1 - y0 + 1

    @staticmethod
    def _dimension_fit_score(shape: np.ndarray, mask_bbox: Optional[BBox], expected_cell: float) -> float:
        if mask_bbox is None or shape.size == 0:
            return 0.0
        rows, cols = shape.shape
        if rows <= 0 or cols <= 0:
            return 0.0

        _, _, bw, bh = mask_bbox
        cell = max(4.0, float(expected_cell))

        gx_candidates = [0.0]
        gy_candidates = [0.0]
        if cols > 1:
            gx = (float(bw) - float(cols) * cell) / float(cols - 1)
            gx = float(np.clip(gx, 0.0, cell * 0.85))
            gx_candidates.extend([max(0.0, gx - 1.5), gx, gx + 1.5])
        if rows > 1:
            gy = (float(bh) - float(rows) * cell) / float(rows - 1)
            gy = float(np.clip(gy, 0.0, cell * 0.85))
            gy_candidates.extend([max(0.0, gy - 1.5), gy, gy + 1.5])

        best_err = 1e9
        for gx in gx_candidates:
            for gy in gy_candidates:
                pred_w = float(cols) * cell + float(cols - 1) * gx
                pred_h = float(rows) * cell + float(rows - 1) * gy
                err = abs(pred_w - float(bw)) / cell + abs(pred_h - float(bh)) / cell
                if err < best_err:
                    best_err = err

        return float(1.0 / (1.0 + best_err))

    @staticmethod
    def _render_shape_mask(
        shape: np.ndarray,
        out_h: int,
        out_w: int,
        x0: int,
        y0: int,
        cell: int,
        gx: int,
        gy: int,
    ) -> Optional[np.ndarray]:
        rows, cols = shape.shape
        out = np.zeros((out_h, out_w), dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                if int(shape[r, c]) == 0:
                    continue
                xx = int(x0 + c * (cell + gx))
                yy = int(y0 + r * (cell + gy))
                if xx < 0 or yy < 0 or xx + cell > out_w or yy + cell > out_h:
                    return None
                out[yy : yy + cell, xx : xx + cell] = 1
        return out

    def _best_shape_overlap_score(self, shape: np.ndarray, mask_binary: np.ndarray, expected_cell: float) -> float:
        if shape.size == 0:
            return 0.0
        bbox = self._binary_bbox(mask_binary)
        if bbox is None:
            return 0.0

        rows, cols = shape.shape
        h, w = mask_binary.shape[:2]
        cell = max(6, int(round(expected_cell)))
        mask01 = (mask_binary > 0).astype(np.uint8)
        mask_area = int(mask01.sum())
        if mask_area <= 0:
            return 0.0

        bx, by, bw, bh = bbox

        gx_candidates = [0]
        gy_candidates = [0]
        if cols > 1:
            gx_est = (float(bw) - float(cols) * cell) / float(cols - 1)
            gx_est = int(round(np.clip(gx_est, 0.0, cell * 0.8)))
            gx_candidates.extend([max(0, gx_est - 2), max(0, gx_est - 1), gx_est, gx_est + 1, gx_est + 2])
        if rows > 1:
            gy_est = (float(bh) - float(rows) * cell) / float(rows - 1)
            gy_est = int(round(np.clip(gy_est, 0.0, cell * 0.8)))
            gy_candidates.extend([max(0, gy_est - 2), max(0, gy_est - 1), gy_est, gy_est + 1, gy_est + 2])

        gx_candidates = sorted(set(int(v) for v in gx_candidates if v >= 0))
        gy_candidates = sorted(set(int(v) for v in gy_candidates if v >= 0))

        best = 0.0
        beta = 1.35
        beta2 = beta * beta
        for gx in gx_candidates:
            for gy in gy_candidates:
                pred_w = cols * cell + max(0, cols - 1) * gx
                pred_h = rows * cell + max(0, rows - 1) * gy
                cx = bx + 0.5 * bw
                cy = by + 0.5 * bh
                base_x = int(round(cx - 0.5 * pred_w))
                base_y = int(round(cy - 0.5 * pred_h))
                for dx in (-8, -4, -2, 0, 2, 4, 8):
                    for dy in (-8, -4, -2, 0, 2, 4, 8):
                        pred = self._render_shape_mask(
                            shape=shape,
                            out_h=h,
                            out_w=w,
                            x0=base_x + dx,
                            y0=base_y + dy,
                            cell=cell,
                            gx=gx,
                            gy=gy,
                        )
                        if pred is None:
                            continue
                        pred_area = int(pred.sum())
                        if pred_area <= 0:
                            continue
                        inter = int(np.logical_and(pred > 0, mask01 > 0).sum())
                        union = int(np.logical_or(pred > 0, mask01 > 0).sum())
                        if inter <= 0 or union <= 0:
                            continue

                        precision = float(inter) / float(pred_area)
                        recall = float(inter) / float(mask_area)
                        if (beta2 * precision + recall) <= 1e-9:
                            fbeta = 0.0
                        else:
                            fbeta = (1.0 + beta2) * precision * recall / (beta2 * precision + recall)
                        iou = float(inter) / float(union)
                        score = (0.62 * fbeta) + (0.38 * iou)
                        if score > best:
                            best = score
        return float(best)

    def _expanded_neighbor_mats(self, occ: np.ndarray, max_add: int = 2) -> List[np.ndarray]:
        tight = _trim_binary(occ.astype(np.uint8))
        if tight.size == 0:
            return []
        base_cells = int(tight.sum())
        if base_cells <= 0 or base_cells >= 9:
            return []

        seen = {_matrix_key(tight)}
        frontier = [tight]
        out: List[np.ndarray] = []

        for _ in range(max(0, int(max_add))):
            next_frontier: List[np.ndarray] = []
            for mat in frontier:
                padded = np.pad(mat.astype(np.uint8), ((1, 1), (1, 1)), mode="constant")
                ys, xs = np.where(padded > 0)
                for y, x in zip(ys, xs):
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny = int(y + dy)
                        nx = int(x + dx)
                        if ny < 0 or nx < 0 or ny >= padded.shape[0] or nx >= padded.shape[1]:
                            continue
                        if padded[ny, nx] != 0:
                            continue
                        cand = padded.copy()
                        cand[ny, nx] = 1
                        cand_tight = _trim_binary(cand)
                        if cand_tight.size == 0:
                            continue
                        if cand_tight.shape[0] > 5 or cand_tight.shape[1] > 5:
                            continue
                        if int(cand_tight.sum()) > 9:
                            continue
                        key = _matrix_key(cand_tight)
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(cand_tight.astype(np.uint8))
                        next_frontier.append(cand_tight.astype(np.uint8))
            frontier = next_frontier
            if not frontier:
                break
        return out

    def _extract_components(self, mask: np.ndarray, slot_bgr: np.ndarray) -> List[dict]:
        comps = self._components_from_binary(mask, mask_area_limit_ratio=self.cfg.max_component_area_ratio)
        if len(comps) >= 2:
            return comps

        # Merged-blob fallback: watershed from distance peaks.
        ws = self._watershed_split(mask, slot_bgr)
        if len(ws) >= len(comps):
            return ws
        return comps

    def _components_from_binary(self, binary: np.ndarray, mask_area_limit_ratio: float) -> List[dict]:
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        h, w = binary.shape[:2]
        max_area = max(1.0, float(h * w) * mask_area_limit_ratio)

        out: List[dict] = []
        for i in range(1, num):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < float(self.cfg.min_component_area_px) or area > max_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            hh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if ww <= 0 or hh <= 0:
                continue
            aspect = float(ww) / float(hh)
            if aspect < self.cfg.aspect_min or aspect > self.cfg.aspect_max:
                continue
            cx, cy = float(centroids[i, 0]), float(centroids[i, 1])
            out.append({"cx": cx, "cy": cy, "area": area, "bbox": (x, y, ww, hh)})
        return out

    def _watershed_split(self, mask: np.ndarray, slot_bgr: np.ndarray) -> List[dict]:
        if int(np.count_nonzero(mask)) < self.cfg.min_component_area_px:
            return []
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        mx = float(dist.max())
        if mx <= 1e-6:
            return []

        _, peaks = cv2.threshold(dist, 0.42 * mx, 255, cv2.THRESH_BINARY)
        peaks = peaks.astype(np.uint8)
        k = np.ones((3, 3), dtype=np.uint8)
        peaks = cv2.morphologyEx(peaks, cv2.MORPH_OPEN, k, iterations=1)

        n, markers = cv2.connectedComponents(peaks)
        if n <= 1:
            return []

        markers = markers + 1
        markers[mask == 0] = 0
        ws_markers = cv2.watershed(slot_bgr.copy(), markers.astype(np.int32))

        out: List[dict] = []
        slot_h, slot_w = mask.shape[:2]
        max_area = max(1.0, float(slot_h * slot_w) * self.cfg.max_component_area_ratio)
        for label in np.unique(ws_markers):
            if label <= 1:
                continue
            region = (ws_markers == label).astype(np.uint8)
            area = int(region.sum())
            if area < self.cfg.min_component_area_px or area > max_area:
                continue
            ys, xs = np.where(region > 0)
            if len(xs) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            ww = x1 - x0 + 1
            hh = y1 - y0 + 1
            if ww <= 0 or hh <= 0:
                continue
            aspect = float(ww) / float(hh)
            if aspect < self.cfg.aspect_min or aspect > self.cfg.aspect_max:
                continue
            cx = float(xs.mean())
            cy = float(ys.mean())
            out.append({"cx": cx, "cy": cy, "area": float(area), "bbox": (x0, y0, ww, hh)})
        return out

    def _components_to_occupancy(self, comps: List[dict]) -> np.ndarray:
        if not comps:
            return np.zeros((0, 0), dtype=np.uint8)

        centers_x = np.array([c["cx"] for c in comps], dtype=np.float32)
        centers_y = np.array([c["cy"] for c in comps], dtype=np.float32)
        sizes = np.array([max(1.0, np.sqrt(float(c["area"]))) for c in comps], dtype=np.float32)
        cell_size = float(np.median(sizes))
        tol = max(2.0, cell_size * self.cfg.cluster_tol_factor)

        col_clusters = _cluster_1d(centers_x, tol=tol)
        row_clusters = _cluster_1d(centers_y, tol=tol)
        col_clusters = _merge_clusters_to_limit(col_clusters, 5)
        row_clusters = _merge_clusters_to_limit(row_clusters, 5)

        if len(col_clusters) == 0 or len(row_clusters) == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        occ = np.zeros((len(row_clusters), len(col_clusters)), dtype=np.uint8)
        for c in comps:
            cx, cy = float(c["cx"]), float(c["cy"])
            col = int(np.argmin([abs(cx - cc) for cc in col_clusters]))
            row = int(np.argmin([abs(cy - rr) for rr in row_clusters]))
            occ[row, col] = 1

        occ = _trim_binary(occ)
        if occ.shape[0] > 5 or occ.shape[1] > 5:
            return np.zeros((0, 0), dtype=np.uint8)
        return occ.astype(np.uint8)

    def _match_tight_matrix(self, tight: np.ndarray) -> tuple[Optional[str], Optional[str], float]:
        key = _matrix_key(tight)
        if key in self.key_to_names:
            name = self.key_to_names[key][0]
            return name, key, 0.99

        best_key = None
        best_score = 0.0
        for k, g in self.key_to_grid.items():
            score = _iou_grid(tight.astype(np.uint8), g.astype(np.uint8))
            if score > best_score:
                best_score = score
                best_key = k

        if best_key is None or best_score < self.cfg.min_match_iou:
            return None, None, float(best_score)
        return self.key_to_names[best_key][0], best_key, float(best_score)
