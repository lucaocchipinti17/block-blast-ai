"""
model.py — Block Blast Heuristic Search Agent
===============================================

Implements the search + evaluation strategy from the reference script,
adapted to work with our Board / pieces architecture.

Architecture
------------
HeuristicAgent
  ├── evaluate_board()   — scores a board state using piece-fit heuristics
  ├── _search_perm()     — exhaustive or sampled search over a piece ordering
  └── best_move()        — public API: returns the best (piece_idx, row, col) next

Search strategy (mirrors reference script):
  - Generate all permutations of the available pieces
  - For each permutation, enumerate placements piece1 → piece2 → piece3
  - If estimated leaf nodes > SAMPLE_THRESHOLD, switch to random sampling
    at depth 0 (same approach as the reference script's 100-sample fallback)
  - Score each leaf (final board state) with evaluate_board()
  - Return the first move of the highest-scoring sequence
"""

import random
import numpy as np
from itertools import permutations as iter_permutations
from typing import Optional

from board import Board


# ── Default heuristic weights ─────────────────────────────────────────────────
# Each key maps to a board feature; positive = reward, negative = penalty.
# These are the starting values derived from the reference script.

DEFAULT_WEIGHTS = {
    "big_l":           10.0,   # Big-L corner shapes (all 4 orientations)
    "sq3x3":           20.0,   # 3×3 square fits
    "sq2x2":            5.0,   # 2×2 square fits
    "i5":               2.0,   # 5-cell straight fits (H + V)
    "i4":               0.8,   # 4-cell straight fits
    "i3":               0.5,   # 3-cell straight fits
    "line_clear":      30.0,   # per line cleared across the sequence
    "empty_islands":   -5.0,   # number of isolated empty regions
    "filled_islands":  -10.0,  # number of isolated filled regions
    "small_islands":   -20.0,  # regions of size 1–3 (essentially unreachable)
    "density":          -0.5,  # total filled cells (mild pressure to stay open)
    "rough_edges":      -0.5,  # jagged filled/empty boundary length
}

# Sampling thresholds
SAMPLE_THRESHOLD = 50_000   # switch to sampling above this estimated leaf count
SAMPLE_SIZE      = 100      # random depth-0 samples per permutation when sampling


# ── Probe shapes for _count_fits ─────────────────────────────────────────────
# Unpadded binary arrays representing each piece shape we probe for.

_PROBE_2x2 = np.array([[1,1],[1,1]])
_PROBE_3x3 = np.array([[1,1,1],[1,1,1],[1,1,1]])
_PROBE_I5H = np.array([[1,1,1,1,1]])
_PROBE_I5V = np.array([[1],[1],[1],[1],[1]])
_PROBE_I4H = np.array([[1,1,1,1]])
_PROBE_I4V = np.array([[1],[1],[1],[1]])
_PROBE_I3H = np.array([[1,1,1]])
_PROBE_I3V = np.array([[1],[1],[1]])
_PROBE_BL  = np.array([[1,0,0],[1,0,0],[1,1,1]])   # Big-L bottom-left
_PROBE_BR  = np.array([[0,0,1],[0,0,1],[1,1,1]])   # Big-L bottom-right
_PROBE_TL  = np.array([[1,1,1],[1,0,0],[1,0,0]])   # Big-L top-left
_PROBE_TR  = np.array([[1,1,1],[0,0,1],[0,0,1]])   # Big-L top-right


class HeuristicAgent:
    """
    Stateless heuristic search agent for Block Blast.

    Usage
    -----
        agent = HeuristicAgent()
        piece_idx, row, col = agent.best_move(board, pieces, used, streak)

    Parameters
    ----------
    weights : dict, optional
        Override default heuristic weights. Useful for evolutionary tuning.
    sample_threshold : int
        Estimated leaf count above which random sampling is used.
    sample_size : int
        Number of depth-0 positions sampled per permutation when sampling.
    verbose : bool
        Print per-call search diagnostics.
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
        sample_threshold: int = SAMPLE_THRESHOLD,
        sample_size: int = SAMPLE_SIZE,
        verbose: bool = False,
    ):
        self.weights          = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.sample_threshold = sample_threshold
        self.sample_size      = sample_size
        self.verbose          = verbose

    # ── Public API ────────────────────────────────────────────────────────────

    def best_move(
        self,
        board: Board,
        pieces: list,
        used: list,
        streak: int = 0,
    ) -> tuple:
        """
        Return (piece_idx, row, col) for the best next placement.

        Parameters
        ----------
        board  : current Board instance
        pieces : list of 3 piece arrays (5×5 padded)
        used   : list of 3 bools — which pieces are already placed this round
        streak : current game streak (informs line-clear reward scaling)

        Returns
        -------
        (piece_idx, row, col)
        """
        # Pairs of (original_index, piece_array) for unplaced pieces
        available = [
            (i, p) for i, (p, u) in enumerate(zip(pieces, used)) if not u
        ]

        if not available:
            raise ValueError("No unplaced pieces available.")

        # Only one piece left — skip full search, just pick its best placement
        if len(available) == 1:
            idx, piece = available[0]
            row, col = self._best_single(board, piece, streak)
            return idx, row, col

        # All orderings of the remaining pieces
        all_perms = list(iter_permutations(available))

        # Decide exhaustive vs. sampling
        leaf_estimate = self._estimate_leaves(board, all_perms)
        use_sampling  = leaf_estimate > self.sample_threshold

        if self.verbose:
            mode = "sampling" if use_sampling else "exhaustive"
            print(f"  [agent] perms={len(all_perms)}  ~leaves={leaf_estimate}  mode={mode}")

        best_score      = -1e9
        best_first_move = None

        for perm in all_perms:
            paths = self._search_perm(board, perm, use_sampling)
            for path in paths:
                score = self._score_path(path, streak)
                if score > best_score:
                    best_score      = score
                    # First step of path = (orig_idx, row, col, lines_cleared)
                    first           = path[0]
                    best_first_move = (perm[0][0], first[1], first[2])

        # Fallback — should rarely trigger
        if best_first_move is None:
            idx, piece = available[0]
            moves = board.valid_moves(piece)
            if moves:
                return idx, moves[0][0], moves[0][1]
            raise ValueError("No valid moves found for any piece.")

        if self.verbose:
            print(f"  [agent] best_score={best_score:.1f}  move={best_first_move}")

        return best_first_move

    # ── Search ────────────────────────────────────────────────────────────────

    def _search_perm(
        self,
        board: Board,
        perm: list,
        use_sampling: bool,
    ) -> list:
        """
        Enumerate (or sample) all placement sequences for one piece ordering.

        Returns a list of paths where each path is:
            [ (piece_idx, row, col, lines_cleared), ..., final_board_array ]
        """
        paths = []
        self._recurse(board, perm, 0, [], paths, use_sampling)
        return paths

    def _recurse(
        self,
        board: Board,
        perm: list,
        depth: int,
        path_so_far: list,
        results: list,
        use_sampling: bool,
    ):
        # Leaf node — record the completed path
        if depth == len(perm):
            results.append(path_so_far + [board.board.copy()])
            return

        idx, piece = perm[depth]
        moves = board.valid_moves(piece)

        if not moves:
            return  # dead end for this ordering

        # Only sample at depth 0 to limit branching (mirrors reference script)
        if use_sampling and depth == 0:
            moves = random.sample(moves, min(self.sample_size, len(moves)))

        for row, col in moves:
            new_board     = board.copy()
            lines_cleared = new_board.apply_move(piece, row, col)
            self._recurse(
                new_board,
                perm,
                depth + 1,
                path_so_far + [(idx, row, col, lines_cleared)],
                results,
                use_sampling,
            )

    def _best_single(self, board: Board, piece: np.ndarray, streak: int) -> tuple:
        """Best placement when only one piece remains — evaluate each move directly."""
        moves      = board.valid_moves(piece)
        best_score = -1e9
        best_rc    = moves[0]

        for row, col in moves:
            b             = board.copy()
            lines_cleared = b.apply_move(piece, row, col)
            score         = self.evaluate_board(b.board, lines_cleared=lines_cleared, streak=streak)
            if score > best_score:
                best_score = score
                best_rc    = (row, col)

        return best_rc

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _score_path(self, path: list, streak: int) -> float:
        """
        Score a complete path.
        path[-1]  = final board array (numpy)
        path[:-1] = list of (piece_idx, row, col, lines_cleared)
        """
        final_board  = path[-1]
        total_clears = sum(step[3] for step in path[:-1])
        return self.evaluate_board(final_board, lines_cleared=total_clears, streak=streak)

    def evaluate_board(
        self,
        board: np.ndarray,
        lines_cleared: int = 0,
        streak: int = 0,
    ) -> float:
        """
        Score a board state. Higher is better.

        Combines piece-fit flexibility, line-clear rewards, and
        penalties for fragmentation and board roughness.
        """
        W     = self.weights
        score = 0.0

        # ── Piece-fit flexibility (how open is the board?) ────────────────────
        score += self._count_fits(_PROBE_BL,  board) * W["big_l"]
        score += self._count_fits(_PROBE_BR,  board) * W["big_l"]
        score += self._count_fits(_PROBE_TL,  board) * W["big_l"]
        score += self._count_fits(_PROBE_TR,  board) * W["big_l"]
        score += self._count_fits(_PROBE_3x3, board) * W["sq3x3"]
        score += self._count_fits(_PROBE_2x2, board) * W["sq2x2"]
        score += self._count_fits(_PROBE_I5H, board) * W["i5"]
        score += self._count_fits(_PROBE_I5V, board) * W["i5"]
        score += self._count_fits(_PROBE_I4H, board) * W["i4"]
        score += self._count_fits(_PROBE_I4V, board) * W["i4"]
        score += self._count_fits(_PROBE_I3H, board) * W["i3"]
        score += self._count_fits(_PROBE_I3V, board) * W["i3"]

        # ── Line clear reward ─────────────────────────────────────────────────
        score += lines_cleared * W["line_clear"]

        # ── Fragmentation penalties ───────────────────────────────────────────
        empty_islands  = self._count_islands(board, value=0)
        filled_islands = self._count_islands(board, value=1)

        score += len(empty_islands)  * W["empty_islands"]
        score += len(filled_islands) * W["filled_islands"]

        # Extra penalty for tiny unreachable pockets (size 1–3)
        score += len([s for s in empty_islands  if s <= 3]) * W["small_islands"]
        score += len([s for s in filled_islands if s <= 3]) * W["small_islands"]

        # ── Density and roughness ─────────────────────────────────────────────
        score += np.count_nonzero(board) * W["density"]
        score += self._rough_edges(board) * W["rough_edges"]

        return score

    # ── Board analysis helpers ────────────────────────────────────────────────

    @staticmethod
    def _count_fits(probe: np.ndarray, board: np.ndarray) -> int:
        """
        Count positions where `probe` fits in empty space on `board`.
        A fit means every cell marked 1 in probe is 0 on the board.
        """
        pr, pc = probe.shape
        br, bc = board.shape
        count  = 0
        for r in range(br - pr + 1):
            for c in range(bc - pc + 1):
                if not np.any(board[r:r+pr, c:c+pc][probe == 1]):
                    count += 1
        return count

    @staticmethod
    def _count_islands(board: np.ndarray, value: int) -> list:
        """
        Flood-fill to find all connected regions of `value`.
        Returns list of region sizes. Regions >= 8 cells are ignored
        (large open areas aren't a concern).
        """
        visited = np.zeros_like(board, dtype=bool)
        sizes   = []
        rows, cols = board.shape

        for start_r in range(rows):
            for start_c in range(cols):
                if visited[start_r, start_c] or board[start_r, start_c] != value:
                    continue
                # Iterative flood fill
                stack = [(start_r, start_c)]
                size  = 0
                while stack:
                    r, c = stack.pop()
                    if r < 0 or r >= rows or c < 0 or c >= cols:
                        continue
                    if visited[r, c] or board[r, c] != value:
                        continue
                    visited[r, c] = True
                    size += 1
                    if size >= 8:
                        break
                    stack.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
                # Mark remaining stack items visited to avoid re-processing
                while stack:
                    r, c = stack.pop()
                    if 0 <= r < rows and 0 <= c < cols:
                        visited[r, c] = True
                if size < 8:
                    sizes.append(size)

        return sizes

    @staticmethod
    def _rough_edges(board: np.ndarray) -> int:
        """
        Count filled cells that border at least one empty cell.
        Measures how jagged the filled/empty boundary is.
        """
        count      = 0
        rows, cols = board.shape
        for r in range(rows):
            for c in range(cols):
                if board[r, c] == 1:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < rows and 0 <= nc < cols and board[nr,nc] == 0:
                            count += 1
        return count

    @staticmethod
    def _estimate_leaves(board: Board, all_perms: list) -> int:
        """
        Cheap upper-bound on total leaf nodes across all permutations.
        Uses valid move count of the first piece as a branching proxy.
        """
        total = 0
        for perm in all_perms:
            n = len(board.valid_moves(perm[0][1]))
            # Subsequent depths assumed to have roughly halving branching
            total += n * max(1, n // 2) * max(1, n // 4)
        return total