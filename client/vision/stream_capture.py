"""
stream_capture.py
=================

Window-located screen capture loop for mirrored mobile game streams.

Design goals:
- Stable FPS capture using mss
- Window discovery by title (pygetwindow when available, macOS AppleScript fallback)
- Thread-safe latest-frame access
- Graceful error handling and recovery
"""

from __future__ import annotations

import platform
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import mss
except Exception:  # noqa: BLE001
    mss = None

try:
    import pygetwindow as gw
except Exception:  # noqa: BLE001
    gw = None


@dataclass(frozen=True)
class WindowRegion:
    left: int
    top: int
    width: int
    height: int


@dataclass
class CaptureStats:
    frame_count: int = 0
    measured_fps: float = 0.0
    last_error: str = ""
    window_region: Optional[WindowRegion] = None


@dataclass(frozen=True)
class CaptureConfig:
    window_title_contains: str
    target_fps: float = 20.0
    window_refresh_sec: float = 1.0


def _require_capture_deps() -> None:
    if mss is None:
        raise RuntimeError("mss is required for screen capture. Install with: pip install mss")


def _region_from_pygetwindow(title_contains: str) -> Optional[WindowRegion]:
    if gw is None:
        return None
    wins = gw.getWindowsWithTitle(title_contains)
    if not wins:
        return None

    # Prefer visible and non-minimized windows, then largest area.
    candidates = []
    for w in wins:
        try:
            width = int(getattr(w, "width", 0))
            height = int(getattr(w, "height", 0))
            left = int(getattr(w, "left", 0))
            top = int(getattr(w, "top", 0))
            if width <= 0 or height <= 0:
                continue
            area = width * height
            minimized = bool(getattr(w, "isMinimized", False))
            visible_score = 0 if minimized else 1
            candidates.append((visible_score, area, left, top, width, height))
        except Exception:  # noqa: BLE001
            continue

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _, _, left, top, width, height = candidates[0]
    return WindowRegion(left=left, top=top, width=width, height=height)


def _region_from_osascript(title_contains: str) -> Optional[WindowRegion]:
    """
    macOS fallback window locator (requires Accessibility permission for host app).
    """
    if platform.system() != "Darwin":
        return None

    target = title_contains.replace('"', '\\"')
    script = f"""
set targetTitle to "{target}"
tell application "System Events"
    set allProcs to application processes whose background only is false
    repeat with p in allProcs
        repeat with w in windows of p
            try
                set winName to name of w as string
                if winName contains targetTitle then
                    set winPos to position of w
                    set winSize to size of w
                    return (item 1 of winPos as string) & "," & (item 2 of winPos as string) & "," & (item 1 of winSize as string) & "," & (item 2 of winSize as string)
                end if
            end try
        end repeat
    end repeat
end tell
return ""
"""
    try:
        proc = subprocess.run(
            ["osascript", "-e", script],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:  # noqa: BLE001
        return None

    raw = proc.stdout.strip()
    if not raw:
        return None
    parts = raw.split(",")
    if len(parts) != 4:
        return None
    try:
        left, top, width, height = [int(float(v.strip())) for v in parts]
    except Exception:  # noqa: BLE001
        return None
    if width <= 0 or height <= 0:
        return None
    return WindowRegion(left=left, top=top, width=width, height=height)


def locate_window_region(title_contains: str) -> Optional[WindowRegion]:
    """
    Locate a window whose title contains `title_contains`.
    """
    if not title_contains.strip():
        return None
    region = _region_from_pygetwindow(title_contains)
    if region is not None:
        return region
    region = _region_from_osascript(title_contains)
    if region is not None:
        return region
    return None


class WindowCaptureLoop:
    """
    Continuous capture loop that keeps the latest frame from a target window.
    """

    def __init__(self, config: CaptureConfig):
        _require_capture_deps()
        self.config = config

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._timestamp: float = 0.0
        self._stats = CaptureStats()

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="WindowCaptureLoop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None

    def get_latest_frame(self, copy: bool = True) -> Optional[np.ndarray]:
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy() if copy else self._frame

    def get_latest_timestamp(self) -> float:
        with self._lock:
            return float(self._timestamp)

    def get_stats(self) -> CaptureStats:
        with self._lock:
            return CaptureStats(
                frame_count=self._stats.frame_count,
                measured_fps=self._stats.measured_fps,
                last_error=self._stats.last_error,
                window_region=self._stats.window_region,
            )

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._stats.last_error = message

    def _run_loop(self) -> None:
        interval = 1.0 / max(1.0, float(self.config.target_fps))
        refresh_sec = max(0.2, float(self.config.window_refresh_sec))
        last_refresh = 0.0
        region: Optional[WindowRegion] = None

        frames_window = 0
        fps_t0 = time.perf_counter()

        with mss.mss() as sct:
            while self._running:
                t0 = time.perf_counter()

                if (region is None) or ((t0 - last_refresh) >= refresh_sec):
                    region = locate_window_region(self.config.window_title_contains)
                    last_refresh = t0
                    with self._lock:
                        self._stats.window_region = region
                    if region is None:
                        self._set_error(
                            f"Window not found for title contains: '{self.config.window_title_contains}'"
                        )

                if region is not None:
                    try:
                        monitor = {
                            "left": int(region.left),
                            "top": int(region.top),
                            "width": int(region.width),
                            "height": int(region.height),
                        }
                        shot = np.array(sct.grab(monitor), dtype=np.uint8)
                        frame_bgr = shot[:, :, :3]  # BGRA -> BGR (drop alpha)
                        ts = time.time()
                        with self._lock:
                            self._frame = frame_bgr
                            self._timestamp = ts
                            self._stats.frame_count += 1
                            self._stats.last_error = ""
                        frames_window += 1
                    except Exception as exc:  # noqa: BLE001
                        self._set_error(f"Capture failure: {exc}")

                elapsed_fps = time.perf_counter() - fps_t0
                if elapsed_fps >= 1.0:
                    with self._lock:
                        self._stats.measured_fps = frames_window / max(elapsed_fps, 1e-6)
                    fps_t0 = time.perf_counter()
                    frames_window = 0

                dt = time.perf_counter() - t0
                sleep_for = interval - dt
                if sleep_for > 0:
                    time.sleep(sleep_for)

