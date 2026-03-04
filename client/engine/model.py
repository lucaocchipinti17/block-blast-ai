"""
model.py — Block Blast Heuristic Search Agent (Bitwise Edition)
================================================================

Board representation
--------------------
Instead of an 8×8 NumPy array, the search engine internally represents
the board as a single Python integer (64 bits). Each bit corresponds to
one cell:

    bit index = row * 8 + col
    bit 0  → (row=0, col=0)  top-left
    bit 63 → (row=7, col=7)  bottom-right

This makes the three most common operations O(1) bitwise ops:

    overlap check  : (board & piece_mask) == 0
    place piece    : board |= piece_mask
    full row check : (board >> (r*8)) & 0xFF == 0xFF

Pieces are also pre-converted to integer masks once at startup and cached,
so the per-move cost is minimal.

Architecture
------------
BitBoard         — lightweight board state wrapper (int + helpers)
HeuristicAgent   — search + evaluation engine
  ├── best_move()       public API: (piece_idx, row, col)
  ├── _search_perm()    exhaustive or sampled depth-first search
  ├── evaluate_board()  scores a BitBoard state
  └── _count_fits()     bitwise probe-shape fit counter
"""

import random
import time
import numpy as np
from itertools import permutations as iter_permutations
from typing import Optional

from client.core.board import Board
try:
    from client.engine import cxx_engine
except Exception:  # noqa: BLE001
    cxx_engine = None


# ── Constants ─────────────────────────────────────────────────────────────────

_FULL_ROW      = 0xFF                    # 8 bits set — a complete row
_FULL_COL_BASE = 0x0101010101010101      # col-0 mask (one bit per row)
_LEFT_MASK     = 0xFEFEFEFEFEFEFEFE         # all cols except col 0 (prevent left-shift wrap)
_RIGHT_MASK    = 0x7F7F7F7F7F7F7F7F         # all cols except col 7 (prevent right-shift wrap)
_BOARD_MASK    = (1 << 64) - 1           # all 64 bits

SAMPLE_THRESHOLD = 50_000
SAMPLE_SIZE      = 100
PY_MAX_NODES     = 120_000
CPP_MAX_NODES    = 1_500_000
PY_DEPTH_CAPS    = {1: 24, 2: 12}
CPP_DEPTH_CAPS   = {1: 64, 2: 32}
STREAK_CLEAR_WINDOW = 3

# ── Default heuristic weights ─────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "big_l":           10.0,
    "sq3x3":           20.0,
    "sq2x2":            5.0,
    "i5":               2.0,
    "i4":               0.8,
    "i3":               0.5,
    "line_clear":      30.0,
    "empty_islands":   -5.0,
    "filled_islands":  -10.0,
    "small_islands":   -20.0,
    "density":          -0.5,
    "rough_edges":      -0.5,
}

_WEIGHT_KEYS = [
    "big_l",
    "sq3x3",
    "sq2x2",
    "i5",
    "i4",
    "i3",
    "line_clear",
    "empty_islands",
    "filled_islands",
    "small_islands",
    "density",
    "rough_edges",
]

PROFILE_CONFIGS = {
    "safe": {"board_weight": 0.70, "streak_bonus": 6.0},
    "balanced": {"board_weight": 0.35, "streak_bonus": 10.0},
    "aggressive": {"board_weight": 0.12, "streak_bonus": 16.0},
}


# ── BitBoard ──────────────────────────────────────────────────────────────────

class BitBoard:
    """
    Lightweight board state as a 64-bit Python integer.

    Bit layout:  bit = row*8 + col
    Bit set (1) = cell occupied, bit clear (0) = empty.

    Keeps a reference to ROWS/COLS for compatibility but all hot-path
    operations work directly on self.bits.
    """

    __slots__ = ("bits",)

    def __init__(self, bits: int = 0):
        self.bits = bits

    # ── Conversion ────────────────────────────────────────────────────────────

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "BitBoard":
        bits = 0
        for r in range(8):
            for c in range(8):
                if arr[r, c]:
                    bits |= (1 << (r * 8 + c))
        return cls(bits)

    def to_numpy(self) -> np.ndarray:
        arr = np.zeros((8, 8), dtype=np.int8)
        bits = self.bits
        for r in range(8):
            row_byte = (bits >> (r * 8)) & 0xFF
            for c in range(8):
                if row_byte & (1 << c):
                    arr[r, c] = 1
        return arr

    # ── Core ops ──────────────────────────────────────────────────────────────

    def can_place(self, shifted_mask: int) -> bool:
        """True if the pre-shifted piece mask doesn't overlap any filled cell."""
        return (self.bits & shifted_mask) == 0

    def place(self, shifted_mask: int) -> tuple:
        """
        Place a piece and clear completed lines.
        Returns (new_BitBoard, lines_cleared).
        Does NOT mutate self.
        """
        bits = self.bits | shifted_mask
        lines_cleared = 0

        # Clear full rows
        for r in range(8):
            row_bits = (bits >> (r * 8)) & _FULL_ROW
            if row_bits == _FULL_ROW:
                bits &= ~(_FULL_ROW << (r * 8))
                lines_cleared += 1

        # Clear full columns
        for c in range(8):
            col_mask = _FULL_COL_BASE << c
            if (bits & col_mask) == col_mask:
                bits &= ~col_mask
                lines_cleared += 1

        return BitBoard(bits), lines_cleared

    def popcount(self) -> int:
        """Number of filled cells (set bits)."""
        return self.bits.bit_count()

    def copy(self) -> "BitBoard":
        return BitBoard(self.bits)

    def __eq__(self, other):
        return self.bits == other.bits

    def __hash__(self):
        return hash(self.bits)


# ── Piece mask cache ──────────────────────────────────────────────────────────

def _build_piece_masks(piece_arr: np.ndarray) -> tuple:
    """
    Convert a 5×5 padded piece array into:
      base_mask : int   — piece mask aligned to (0,0), no padding
      h         : int   — bounding box height
      w         : int   — bounding box width
      shifted   : dict  — {(row, col): shifted_mask} for all valid placements

    The shifted dict is pre-computed so valid_moves and placement during
    search are O(1) dictionary lookups instead of repeated bit-shifting.
    """
    filled_rows, filled_cols = np.where(piece_arr != 0)
    if len(filled_rows) == 0:
        return 0, 0, 0, {}

    min_r = int(filled_rows.min())
    min_c = int(filled_cols.min())

    base_mask = 0
    for r, c in zip(filled_rows.tolist(), filled_cols.tolist()):
        bit = (r - min_r) * 8 + (c - min_c)
        base_mask |= (1 << bit)

    h = int(filled_rows.max() - min_r + 1)
    w = int(filled_cols.max() - min_c + 1)

    # Pre-shift for every valid (row, col) on an 8×8 board
    shifted = {}
    for row in range(8 - h + 1):
        for col in range(8 - w + 1):
            shift = row * 8 + col
            shifted[(row, col)] = base_mask << shift

    return base_mask, h, w, shifted


# Precompute masks for every piece in ALL_PIECES at import time
_PIECE_MASK_CACHE: dict = {}

def get_piece_masks(piece_arr: np.ndarray) -> tuple:
    """Return cached (base_mask, h, w, shifted_dict) for this piece array."""
    key = piece_arr.tobytes()
    if key not in _PIECE_MASK_CACHE:
        _PIECE_MASK_CACHE[key] = _build_piece_masks(piece_arr)
    return _PIECE_MASK_CACHE[key]


# ── Probe masks for evaluate_board ───────────────────────────────────────────
# Pre-built shifted dicts for every probe shape used in heuristic evaluation.

def _probe_masks(shape_2d: list) -> list:
    """Convert a 2D list shape into a list of all shifted masks on 8×8 board."""
    arr = np.array(shape_2d, dtype=np.int8)
    # Pad to 5×5 so get_piece_masks works
    padded = np.zeros((5, 5), dtype=np.int8)
    padded[:arr.shape[0], :arr.shape[1]] = arr
    _, _, _, shifted = get_piece_masks(padded)
    return list(shifted.values())

# Built once at module load
_PROBE_MASKS = {
    "big_l_bl":  _probe_masks([[1,0,0],[1,0,0],[1,1,1]]),
    "big_l_br":  _probe_masks([[0,0,1],[0,0,1],[1,1,1]]),
    "big_l_tl":  _probe_masks([[1,1,1],[1,0,0],[1,0,0]]),
    "big_l_tr":  _probe_masks([[1,1,1],[0,0,1],[0,0,1]]),
    "sq3x3":     _probe_masks([[1,1,1],[1,1,1],[1,1,1]]),
    "sq2x2":     _probe_masks([[1,1],[1,1]]),
    "i5h":       _probe_masks([[1,1,1,1,1]]),
    "i5v":       _probe_masks([[1],[1],[1],[1],[1]]),
    "i4h":       _probe_masks([[1,1,1,1]]),
    "i4v":       _probe_masks([[1],[1],[1],[1]]),
    "i3h":       _probe_masks([[1,1,1]]),
    "i3v":       _probe_masks([[1],[1],[1]]),
}


# ── HeuristicAgent ────────────────────────────────────────────────────────────

class HeuristicAgent:
    """
    Heuristic search agent using a bitwise board representation.

    Usage
    -----
        agent = HeuristicAgent()
        piece_idx, row, col = agent.best_move(board, pieces, used, streak)

    Parameters
    ----------
    weights          : dict — override DEFAULT_WEIGHTS
    sample_threshold : int  — switch to sampling above this leaf estimate
    sample_size      : int  — depth-0 samples per perm when sampling
    verbose          : bool — print search diagnostics
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        sample_threshold: int = SAMPLE_THRESHOLD,
        sample_size: int = SAMPLE_SIZE,
        time_budget_ms: Optional[int] = 4500,
        max_nodes: Optional[int] = None,
        depth_branch_caps: Optional[dict[int, int]] = None,
        eval_cache_max: int = 200_000,
        use_cpp_backend: bool = True,
        default_profile: str = "balanced",
        verbose: bool = False,
    ):
        self.weights          = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.sample_threshold = sample_threshold
        self.sample_size      = sample_size
        self.time_budget_ms   = time_budget_ms
        self.eval_cache_max   = eval_cache_max
        self._eval_cache: dict[int, float] = {}
        self.use_cpp_backend = use_cpp_backend
        self._cpp_available = False
        if use_cpp_backend and cxx_engine is not None:
            self._cpp_available = cxx_engine.is_available()
            if verbose and not self._cpp_available:
                print("  [agent] C++ backend unavailable; using Python backend.")

        if max_nodes is None:
            self.max_nodes = CPP_MAX_NODES if self._cpp_available else PY_MAX_NODES
        else:
            self.max_nodes = max_nodes

        if depth_branch_caps is None:
            self.depth_branch_caps = CPP_DEPTH_CAPS.copy() if self._cpp_available else PY_DEPTH_CAPS.copy()
        else:
            self.depth_branch_caps = depth_branch_caps

        self.default_profile = default_profile if default_profile in PROFILE_CONFIGS else "balanced"
        self.verbose          = verbose

    # ── Public API ────────────────────────────────────────────────────────────

    def best_move(
        self,
        board: Board,
        pieces: list,
        used: list,
        streak: int = 0,
        moves_since_clear: int = STREAK_CLEAR_WINDOW,
        profile: Optional[str] = None,
    ) -> tuple:
        """
        Return (piece_idx, row, col) for the best next placement.
        """
        plan = self.best_plan(
            board,
            pieces,
            used,
            streak,
            moves_since_clear=moves_since_clear,
            profile=profile,
        )
        if not plan:
            raise ValueError("No valid moves found.")
        return plan[0]

    def best_plan(
        self,
        board: Board,
        pieces: list,
        used: list,
        streak: int = 0,
        moves_since_clear: int = STREAK_CLEAR_WINDOW,
        profile: Optional[str] = None,
    ) -> list[tuple[int, int, int]]:
        """
        Return a placement plan for the current piece bank as:
        [(piece_idx, row, col), ...]

        For 2-3 available pieces, this searches permutations and placement
        sequences while applying intermediate line clears in the simulated board
        state. If no full path is found, falls back to a greedy step-by-step
        plan so callers can still execute without re-running search each turn.
        """
        bb = BitBoard.from_numpy(board.board)
        profile_name, board_weight, streak_bonus = self._resolve_profile(profile)
        available = [
            (i, p) for i, (p, u) in enumerate(zip(pieces, used)) if not u
        ]
        if not available:
            raise ValueError("No unplaced pieces available.")

        if len(available) == 1:
            idx, piece = available[0]
            row, col = self._best_single(
                bb,
                piece,
                streak,
                moves_since_clear,
                board_weight,
                streak_bonus,
            )
            return [(idx, row, col)]

        cells_per_idx = {}
        for idx, piece in available:
            cells_per_idx[idx] = int(len(Board.get_footprint(piece)[0]))

        cpp_plan = self._best_plan_cpp(
            bb=bb,
            available=available,
            streak=streak,
            moves_since_clear=moves_since_clear,
            board_weight=board_weight,
            streak_bonus=streak_bonus,
            cells_per_idx=cells_per_idx,
        )
        if cpp_plan:
            if self.verbose:
                print(f"  [agent] using C++ backend  profile={profile_name}")
            return cpp_plan

        all_perms     = list(iter_permutations(available))
        leaf_estimate = self._estimate_leaves(bb, all_perms)
        use_sampling  = leaf_estimate > self.sample_threshold

        # Prioritize permutations with fewer first-piece placements.
        # This typically finds strong plans faster under budgets.
        perm_first_counts = []
        for perm in all_perms:
            _, _, _, shifted = get_piece_masks(perm[0][1])
            first_n = sum(1 for mask in shifted.values() if bb.can_place(mask))
            perm_first_counts.append((first_n, perm))
        ordered_perms = [perm for _, perm in sorted(perm_first_counts, key=lambda x: x[0])]

        if self.verbose:
            mode = "sampling" if use_sampling else "exhaustive"
            print(
                f"  [agent] perms={len(all_perms)}  ~leaves={leaf_estimate}  "
                f"mode={mode}  budget={self.time_budget_ms}ms/{self.max_nodes}nodes  "
                f"profile={profile_name}"
            )

        best_score = -1e9
        best_steps = None
        deadline = (
            time.perf_counter() + (self.time_budget_ms / 1000.0)
            if self.time_budget_ms is not None else None
        )
        nodes_left = self.max_nodes

        for perm in ordered_perms:
            if nodes_left <= 0:
                break
            if deadline is not None and time.perf_counter() >= deadline:
                break

            perm_score, perm_steps, nodes_used = self._search_best_perm(
                bb,
                perm,
                use_sampling,
                streak,
                moves_since_clear,
                board_weight,
                streak_bonus,
                cells_per_idx,
                deadline=deadline,
                node_budget=nodes_left,
            )
            nodes_left -= nodes_used
            if perm_steps is not None and perm_score > best_score:
                best_score = perm_score
                best_steps = perm_steps

        if best_steps is not None:
            plan = [(idx, row, col) for (idx, row, col, _) in best_steps]
            if self.verbose:
                used_nodes = self.max_nodes - nodes_left
                print(f"  [agent] best_score={best_score:.1f}  plan={plan}  nodes={used_nodes}")
            return plan

        # Fallback: greedy multi-step planning if no complete path exists.
        plan = self._greedy_plan(
            bb=bb,
            available=available,
            streak=streak,
            moves_since_clear=moves_since_clear,
            board_weight=board_weight,
            streak_bonus=streak_bonus,
            cells_per_idx=cells_per_idx,
        )
        if self.verbose:
            print(f"  [agent] fallback_greedy_plan={plan}")
        return plan

    def _resolve_profile(self, profile: Optional[str]) -> tuple[str, float, float]:
        name = profile or self.default_profile
        if name not in PROFILE_CONFIGS:
            name = "balanced"
        cfg = PROFILE_CONFIGS[name]
        return name, float(cfg["board_weight"]), float(cfg["streak_bonus"])

    @staticmethod
    def _advance_streak_state(
        streak: int,
        moves_since_clear: int,
        lines_cleared: int,
    ) -> tuple[int, int]:
        """
        Advance streak state using a 3-move streak window.
        A line clear continues the streak only if the previous clear was <=3 moves ago.
        """
        if lines_cleared > 0:
            if streak > 0 and moves_since_clear < STREAK_CLEAR_WINDOW:
                return streak + 1, 0
            return 1, 0

        next_gap = min(STREAK_CLEAR_WINDOW, max(0, moves_since_clear) + 1)
        if streak > 0 and next_gap < STREAK_CLEAR_WINDOW:
            return streak, next_gap
        return 0, next_gap

    def _best_plan_cpp(
        self,
        bb: BitBoard,
        available: list[tuple[int, np.ndarray]],
        streak: int,
        moves_since_clear: int,
        board_weight: float,
        streak_bonus: float,
        cells_per_idx: dict[int, int],
    ) -> Optional[list[tuple[int, int, int]]]:
        if not (self.use_cpp_backend and self._cpp_available and cxx_engine is not None):
            return None

        base_masks: list[int] = []
        hs: list[int] = []
        ws: list[int] = []
        piece_indices: list[int] = []

        for idx, piece in available:
            base_mask, h, w, _ = get_piece_masks(piece)
            base_masks.append(int(base_mask))
            hs.append(int(h))
            ws.append(int(w))
            piece_indices.append(int(idx))
        piece_cells = [int(cells_per_idx[idx]) for idx, _ in available]

        weights_vec = [float(self.weights[k]) for k in _WEIGHT_KEYS]
        cap_depth1 = int(self.depth_branch_caps.get(1, -1))
        cap_depth2 = int(self.depth_branch_caps.get(2, -1))
        time_budget_ms = int(self.time_budget_ms if self.time_budget_ms is not None else -1)

        try:
            plan = cxx_engine.best_plan_cpp(
                board_bits=int(bb.bits),
                base_masks=base_masks,
                hs=hs,
                ws=ws,
                piece_indices=piece_indices,
                piece_cells=piece_cells,
                weights=weights_vec,
                sample_threshold=int(self.sample_threshold),
                sample_size=int(self.sample_size),
                time_budget_ms=time_budget_ms,
                max_nodes=int(self.max_nodes),
                cap_depth1=cap_depth1,
                cap_depth2=cap_depth2,
                eval_cache_max=int(self.eval_cache_max),
                initial_streak=int(streak),
                initial_moves_since_clear=int(max(0, moves_since_clear)),
                board_weight=float(board_weight),
                streak_bonus=float(streak_bonus),
            )
            return plan
        except Exception:
            # Graceful runtime fallback to Python backend if C++ path fails.
            self._cpp_available = False
            return None

    # ── Search ────────────────────────────────────────────────────────────────

    def _search_best_perm(
        self,
        bb: BitBoard,
        perm: list,
        use_sampling: bool,
        streak: int,
        moves_since_clear: int,
        board_weight: float,
        streak_bonus: float,
        cells_per_idx: dict[int, int],
        deadline: Optional[float],
        node_budget: int,
    ) -> tuple[float, Optional[tuple], int]:
        """
        Depth-first search that returns only the best path for a permutation.
        Avoids materializing all leaves/paths in memory.
        Uses a transposition cache keyed by (bits, depth, streak) to reuse
        repeated subtree results within this permutation.
        """
        cache: dict[tuple[int, int, int, int], tuple[float, Optional[tuple]]] = {}
        nodes_visited = 0
        budget_exhausted = False

        def dfs(
            cur_bb: BitBoard,
            depth: int,
            cur_streak: int,
            cur_moves_since_clear: int,
        ) -> tuple[float, Optional[tuple]]:
            nonlocal nodes_visited, budget_exhausted
            if budget_exhausted:
                return -1e9, None

            nodes_visited += 1
            if nodes_visited >= node_budget:
                budget_exhausted = True
                return -1e9, None
            if deadline is not None and (nodes_visited & 0xFF) == 0:
                if time.perf_counter() >= deadline:
                    budget_exhausted = True
                    return -1e9, None

            cache_key = (cur_bb.bits, depth, cur_streak, cur_moves_since_clear)
            if cache_key in cache:
                return cache[cache_key]

            if depth == len(perm):
                score = board_weight * self._evaluate_bits_cached(cur_bb.bits)
                score += streak_bonus * cur_streak
                result = (score, ())
                cache[cache_key] = result
                return result

            idx, piece = perm[depth]
            _, _, _, shifted = get_piece_masks(piece)
            best_score = -1e9
            best_steps: Optional[tuple] = None

            children = []
            for (row, col), mask in shifted.items():
                if not cur_bb.can_place(mask):
                    continue
                new_bb, lines = cur_bb.place(mask)
                children.append((idx, row, col, lines, new_bb))

            children = self._limit_children(children, depth, use_sampling)
            for idx, row, col, lines, new_bb in children:
                cells = cells_per_idx[idx]
                next_streak, next_moves_since_clear = self._advance_streak_state(
                    cur_streak,
                    cur_moves_since_clear,
                    lines,
                )
                move_points = cells
                if lines > 0:
                    move_points += lines * next_streak * 10

                child_score, child_steps = dfs(
                    new_bb,
                    depth + 1,
                    next_streak,
                    next_moves_since_clear,
                )
                if child_steps is None:
                    continue
                score = child_score + move_points
                if score > best_score:
                    best_score = score
                    best_steps = ((idx, row, col, lines),) + child_steps

            result = (best_score, best_steps)
            cache[cache_key] = result
            return result

        score, steps = dfs(bb, 0, streak, moves_since_clear)
        return score, steps, nodes_visited

    def _limit_children(
        self,
        children: list[tuple[int, int, int, int, BitBoard]],
        depth: int,
        use_sampling: bool,
    ) -> list[tuple[int, int, int, int, BitBoard]]:
        """
        Limit branching factor for speed. We keep line-clearing candidates first,
        then fill remaining slots with lower-density boards.
        """
        if not children:
            return children

        if use_sampling and depth == 0 and len(children) > self.sample_size:
            children = random.sample(children, self.sample_size)

        cap = self.depth_branch_caps.get(depth)
        if cap is None or len(children) <= cap:
            return children

        clearers = [c for c in children if c[3] > 0]
        non_clearers = [c for c in children if c[3] == 0]

        clearers.sort(key=lambda c: (-c[3], c[4].bits.bit_count()))
        if len(clearers) >= cap:
            return clearers[:cap]

        non_clearers.sort(key=lambda c: c[4].bits.bit_count())
        need = cap - len(clearers)
        return clearers + non_clearers[:need]

    def _best_single(
        self,
        bb: BitBoard,
        piece: np.ndarray,
        streak: int,
        moves_since_clear: int,
        board_weight: float,
        streak_bonus: float,
    ) -> tuple:
        _, _, _, shifted = get_piece_masks(piece)
        cells = int(len(Board.get_footprint(piece)[0]))
        best_score = -1e9
        best_rc    = None

        for (row, col), mask in shifted.items():
            if not bb.can_place(mask):
                continue
            new_bb, lines = bb.place(mask)
            next_streak, _ = self._advance_streak_state(streak, moves_since_clear, lines)
            points = cells
            if lines > 0:
                points += lines * next_streak * 10
            score = points + board_weight * self._evaluate_bits_cached(new_bb.bits) + streak_bonus * next_streak
            if score > best_score:
                best_score = score
                best_rc    = (row, col)

        return best_rc or (0, 0)

    def _greedy_plan(
        self,
        bb: BitBoard,
        available: list[tuple[int, np.ndarray]],
        streak: int,
        moves_since_clear: int,
        board_weight: float,
        streak_bonus: float,
        cells_per_idx: dict[int, int],
    ) -> list[tuple[int, int, int]]:
        """
        Build a step-by-step plan without full permutation search.
        Used as a fallback when no complete search path exists.
        """
        plan = []
        remaining = available[:]
        current_bb = bb
        current_streak = streak
        current_moves_since_clear = moves_since_clear

        while remaining:
            best = None
            best_score = -1e9

            for idx, piece in remaining:
                _, _, _, shifted = get_piece_masks(piece)
                for (row, col), mask in shifted.items():
                    if not current_bb.can_place(mask):
                        continue
                    new_bb, lines = current_bb.place(mask)
                    next_streak, next_moves_since_clear = self._advance_streak_state(
                        current_streak,
                        current_moves_since_clear,
                        lines,
                    )
                    points = cells_per_idx[idx]
                    if lines > 0:
                        points += lines * next_streak * 10
                    score = points + board_weight * self._evaluate_bits_cached(new_bb.bits) + streak_bonus * next_streak
                    if score > best_score:
                        best_score = score
                        best = (idx, row, col, new_bb, lines, next_streak, next_moves_since_clear)

            if best is None:
                break

            idx, row, col, new_bb, lines, next_streak, next_moves_since_clear = best
            plan.append((idx, row, col))
            current_bb = new_bb
            current_streak = next_streak
            current_moves_since_clear = next_moves_since_clear
            remaining = [(i, p) for (i, p) in remaining if i != idx]

        return plan

    # ── Scoring ───────────────────────────────────────────────────────────────

    def evaluate_board(
        self,
        bb: BitBoard,
        lines_cleared: int = 0,
        streak: int = 0,
    ) -> float:
        """
        Score a BitBoard state. Higher = better.

        All probe-fit counting is done via bitwise AND against pre-built
        masks — no inner Python loops over the board.
        """
        bits = bb.bits
        score = self._evaluate_bits_cached(bits)
        score += lines_cleared * self.weights["line_clear"]
        return score

    def _evaluate_bits_cached(self, bits: int) -> float:
        cached = self._eval_cache.get(bits)
        if cached is not None:
            return cached

        W = self.weights
        score = 0.0

        # ── Piece-fit flexibility ─────────────────────────────────────────────
        # A probe "fits" at a position if (board & probe_mask) == 0
        # i.e. all probe cells are empty on the board.

        for m in _PROBE_MASKS["big_l_bl"]: score += (bits & m) == 0
        for m in _PROBE_MASKS["big_l_br"]: score += (bits & m) == 0
        for m in _PROBE_MASKS["big_l_tl"]: score += (bits & m) == 0
        for m in _PROBE_MASKS["big_l_tr"]: score += (bits & m) == 0
        score *= W["big_l"] / 4   # already summed all 4 orientations

        sq3 = sum((bits & m) == 0 for m in _PROBE_MASKS["sq3x3"])
        sq2 = sum((bits & m) == 0 for m in _PROBE_MASKS["sq2x2"])
        i5  = sum((bits & m) == 0 for m in _PROBE_MASKS["i5h"]) + \
              sum((bits & m) == 0 for m in _PROBE_MASKS["i5v"])
        i4  = sum((bits & m) == 0 for m in _PROBE_MASKS["i4h"]) + \
              sum((bits & m) == 0 for m in _PROBE_MASKS["i4v"])
        i3  = sum((bits & m) == 0 for m in _PROBE_MASKS["i3h"]) + \
              sum((bits & m) == 0 for m in _PROBE_MASKS["i3v"])

        score += sq3 * W["sq3x3"]
        score += sq2 * W["sq2x2"]
        score += i5  * W["i5"]
        score += i4  * W["i4"]
        score += i3  * W["i3"]

        # ── Fragmentation penalties ───────────────────────────────────────────
        filled_islands = self._count_islands_bits(bits)
        empty_islands = self._count_islands_bits((~bits) & _BOARD_MASK)

        score += len(empty_islands)  * W["empty_islands"]
        score += len(filled_islands) * W["filled_islands"]
        score += len([s for s in empty_islands  if s <= 3]) * W["small_islands"]
        score += len([s for s in filled_islands if s <= 3]) * W["small_islands"]

        # ── Density and roughness ─────────────────────────────────────────────
        score += bits.bit_count() * W["density"]
        score += self._rough_edges_bitwise(bits) * W["rough_edges"]

        if len(self._eval_cache) >= self.eval_cache_max:
            self._eval_cache.clear()
        self._eval_cache[bits] = score
        return score

    # ── Board analysis ────────────────────────────────────────────────────────

    @staticmethod
    def _rough_edges_bitwise(bits: int) -> int:
        """
        Count filled cells that border at least one empty cell.
        Uses bitwise shifts instead of nested Python loops.

        Shift the board in each of 4 directions and XOR against
        the original to find boundary cells.
        """
        # Neighbour masks (avoid wrapping at row/col edges)
        LEFT_MASK  = 0xFEFEFEFEFEFEFEFE   # all cols except col 0
        RIGHT_MASK = 0x7F7F7F7F7F7F7F7F   # all cols except col 7

        up    = bits >> 8
        down  = bits << 8
        left  = (bits >> 1) & LEFT_MASK
        right = (bits << 1) & RIGHT_MASK

        # Cells that have at least one empty neighbour
        has_empty_neighbour = bits & ~(up & down & left & right) & _BOARD_MASK
        return has_empty_neighbour.bit_count()

    @staticmethod
    def _flood_expand(seed: int, allowed: int) -> int:
        """
        Expand seed bits to all connected cells within `allowed` using bitwise shifts.
        Up/down = shift by 8 bits (one row), left/right = shift by 1 bit with edge masks.
        Iterates until stable (no new cells added).
        """
        while True:
            prev = seed
            seed = (seed
                    | (seed >> 8)
                    | ((seed << 8) & _BOARD_MASK)
                    | ((seed >> 1) & _LEFT_MASK)
                    | ((seed << 1) & _RIGHT_MASK)
                    ) & allowed
            if seed == prev:
                break
        return seed

    @staticmethod
    def _count_islands_bits(target: int) -> list:
        """
        Bitwise flood-fill to find connected regions from a target bitmask.
        Returns list of region sizes < 8 (large open areas are not a concern).

        Works on the board as a 64-bit integer: each region is found by
        isolating the lowest set bit as a seed, flood-expanding it within
        the target mask, recording the size, then clearing that region and
        repeating until no cells remain.
        """
        remaining = target
        sizes     = []

        while remaining:
            seed   = remaining & (-remaining)   # isolate lowest set bit
            region = HeuristicAgent._flood_expand(seed, target)
            size   = region.bit_count()
            remaining &= ~region
            if size < 8:
                sizes.append(size)

        return sizes

    @staticmethod
    def _estimate_leaves(bb: BitBoard, all_perms: list) -> int:
        total = 0
        for perm in all_perms:
            _, _, _, shifted = get_piece_masks(perm[0][1])
            n = sum(1 for mask in shifted.values() if bb.can_place(mask))
            total += n * max(1, n // 2) * max(1, n // 4)
        return total
