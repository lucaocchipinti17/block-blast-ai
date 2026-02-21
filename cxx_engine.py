from __future__ import annotations

import ctypes
import subprocess
import sys
from pathlib import Path
from typing import Optional


_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "cxx" / "search_engine.cpp"
if sys.platform == "darwin":
    _LIB = _ROOT / "cxx" / "libblockblast_engine.dylib"
elif sys.platform.startswith("win"):
    _LIB = _ROOT / "cxx" / "blockblast_engine.dll"
else:
    _LIB = _ROOT / "cxx" / "libblockblast_engine.so"

_LIB_HANDLE: Optional[ctypes.CDLL] = None
_LOAD_ERROR: Optional[Exception] = None


def _compile_lib() -> None:
    _LIB.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "clang++",
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(_SRC),
        "-o",
        str(_LIB),
    ]
    subprocess.run(cmd, check=True)


def _ensure_loaded() -> ctypes.CDLL:
    global _LIB_HANDLE, _LOAD_ERROR
    if _LIB_HANDLE is not None:
        return _LIB_HANDLE
    if _LOAD_ERROR is not None:
        raise RuntimeError(_LOAD_ERROR)

    try:
        if (not _LIB.exists()) or (_LIB.stat().st_mtime < _SRC.stat().st_mtime):
            _compile_lib()

        lib = ctypes.CDLL(str(_LIB))
        lib.bb_best_plan.argtypes = [
            ctypes.c_uint64,                        # board_bits
            ctypes.POINTER(ctypes.c_uint64),        # base_masks
            ctypes.POINTER(ctypes.c_int),           # hs
            ctypes.POINTER(ctypes.c_int),           # ws
            ctypes.POINTER(ctypes.c_int),           # piece_indices
            ctypes.POINTER(ctypes.c_int),           # piece_cells
            ctypes.c_int,                           # n_pieces
            ctypes.POINTER(ctypes.c_double),        # weights
            ctypes.c_int,                           # sample_threshold
            ctypes.c_int,                           # sample_size
            ctypes.c_int,                           # time_budget_ms
            ctypes.c_int,                           # max_nodes
            ctypes.c_int,                           # cap_depth1
            ctypes.c_int,                           # cap_depth2
            ctypes.c_int,                           # eval_cache_max
            ctypes.c_int,                           # initial_streak
            ctypes.c_double,                        # board_weight
            ctypes.c_double,                        # streak_bonus
            ctypes.POINTER(ctypes.c_int),           # out_len
            ctypes.POINTER(ctypes.c_int),           # out_piece_idx
            ctypes.POINTER(ctypes.c_int),           # out_row
            ctypes.POINTER(ctypes.c_int),           # out_col
        ]
        lib.bb_best_plan.restype = ctypes.c_int
        _LIB_HANDLE = lib
        return lib
    except Exception as e:  # noqa: BLE001
        _LOAD_ERROR = e
        raise


def is_available() -> bool:
    try:
        _ensure_loaded()
        return True
    except Exception:
        return False


def best_plan_cpp(
    board_bits: int,
    base_masks: list[int],
    hs: list[int],
    ws: list[int],
    piece_indices: list[int],
    piece_cells: list[int],
    weights: list[float],
    sample_threshold: int,
    sample_size: int,
    time_budget_ms: int,
    max_nodes: int,
    cap_depth1: int,
    cap_depth2: int,
    eval_cache_max: int,
    initial_streak: int,
    board_weight: float,
    streak_bonus: float,
) -> Optional[list[tuple[int, int, int]]]:
    lib = _ensure_loaded()

    n = len(base_masks)
    if n == 0:
        return []

    c_masks = (ctypes.c_uint64 * n)(*base_masks)
    c_hs = (ctypes.c_int * n)(*hs)
    c_ws = (ctypes.c_int * n)(*ws)
    c_indices = (ctypes.c_int * n)(*piece_indices)
    c_cells = (ctypes.c_int * n)(*piece_cells)
    c_weights = (ctypes.c_double * len(weights))(*weights)

    out_len = ctypes.c_int(0)
    out_piece_idx = (ctypes.c_int * 3)()
    out_row = (ctypes.c_int * 3)()
    out_col = (ctypes.c_int * 3)()

    ok = lib.bb_best_plan(
        ctypes.c_uint64(board_bits),
        c_masks,
        c_hs,
        c_ws,
        c_indices,
        c_cells,
        ctypes.c_int(n),
        c_weights,
        ctypes.c_int(sample_threshold),
        ctypes.c_int(sample_size),
        ctypes.c_int(time_budget_ms),
        ctypes.c_int(max_nodes),
        ctypes.c_int(cap_depth1),
        ctypes.c_int(cap_depth2),
        ctypes.c_int(eval_cache_max),
        ctypes.c_int(initial_streak),
        ctypes.c_double(board_weight),
        ctypes.c_double(streak_bonus),
        ctypes.byref(out_len),
        out_piece_idx,
        out_row,
        out_col,
    )

    if ok != 1:
        return None

    m = out_len.value
    plan: list[tuple[int, int, int]] = []
    for i in range(m):
        plan.append((int(out_piece_idx[i]), int(out_row[i]), int(out_col[i])))
    return plan
