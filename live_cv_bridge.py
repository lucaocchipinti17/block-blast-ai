"""
live_cv_bridge.py
=================

Orchestrates:
- window capture loop
- piece bank detection
- heuristic planning

This module keeps GUI code thin and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import time

import numpy as np

from board import Board
from model import HeuristicAgent
from piece_bank_detector import (
    DetectedPiece,
    PieceBankDetection,
    PieceBankDetector,
    PieceBankDetectorConfig,
)
from stream_capture import CaptureConfig, WindowCaptureLoop


@dataclass(frozen=True)
class LiveCVConfig:
    capture: CaptureConfig
    detector: PieceBankDetectorConfig


@dataclass(frozen=True)
class LivePlanResult:
    detection: PieceBankDetection
    piece_bank: List[np.ndarray]
    piece_names: List[Optional[str]]
    plan: List[tuple[int, int, int]]
    total_ms: float


class LiveCVPlanner:
    """
    Production-style bridge for capture -> detect -> plan.
    """

    def __init__(self, config: LiveCVConfig, agent: Optional[HeuristicAgent] = None):
        self.config = config
        self.capture = WindowCaptureLoop(config.capture)
        self.detector = PieceBankDetector(config.detector)
        self.agent = agent or HeuristicAgent()

    def start(self) -> None:
        self.capture.start()

    def stop(self) -> None:
        self.capture.stop()

    def process_once(self, board: Board, streak: int, profile: str = "balanced") -> LivePlanResult:
        t0 = time.perf_counter()

        frame = self.capture.get_latest_frame(copy=True)
        if frame is None:
            stats = self.capture.get_stats()
            err = stats.last_error or "Capture frame unavailable (stream not ready yet)."
            raise RuntimeError(err)

        detection = self.detector.detect_piece_bank(frame)
        piece_names = [p.name for p in detection.pieces]

        piece_bank = [p.piece_array_5x5.astype(np.int8).copy() for p in detection.pieces]
        used = [False, False, False]

        if board.is_game_over(piece_bank):
            raise RuntimeError("Detected piece bank has no valid moves on current board.")

        plan = self.agent.best_plan(board, piece_bank, used, streak=streak, profile=profile)
        total_ms = (time.perf_counter() - t0) * 1000.0
        return LivePlanResult(
            detection=detection,
            piece_bank=piece_bank,
            piece_names=piece_names,
            plan=plan,
            total_ms=float(total_ms),
        )
