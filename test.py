"""
test.py — Block Blast Agent Test Runner
========================================

Runs N headless games using HeuristicAgent and records:
  - Score per game
  - Turns per game
  - Time elapsed per game
  - High score across all runs

Usage
-----
    python test.py                  # 10 runs with defaults
    python test.py --runs 50        # 50 runs
    python test.py --runs 20 --verbose   # show per-turn agent diagnostics
"""

import argparse
import random
import time
import numpy as np

from board import Board
from pieces import ALL_PIECES
from game import calculate_score
from model import HeuristicAgent


def run_game(agent: HeuristicAgent, piece_pool: dict, verbose: bool = False) -> dict:
    """
    Run a single headless game to completion using the agent.

    Returns
    -------
    dict with keys: score, turns, rounds, elapsed
    """
    board  = Board()
    score  = 0
    streak = 0
    turn   = 0
    round_ = 0
    round_had_clear = False
    t_start = time.time()

    while True:
        # ── Draw new piece bank ───────────────────────────────────────────────
        names      = random.choices(list(piece_pool.keys()), k=3)
        piece_bank = [piece_pool[n].copy() for n in names]
        used       = [False, False, False]
        round_     += 1

        # ── Check game over before round starts ───────────────────────────────
        if board.is_game_over(piece_bank):
            break

        round_had_clear = False

        # ── Compute and execute a full round plan ─────────────────────────────
        plan = agent.best_plan(board, piece_bank, used, streak)

        for piece_idx, row, col in plan:
            # If board has changed in an unexpected way, stop the round safely.
            if used[piece_idx]:
                continue

            # Place the piece
            piece         = piece_bank[piece_idx]
            valid         = board.valid_moves(piece)
            if (row, col) not in valid:
                break

            cells, _, _   = Board.get_footprint(piece)
            cells_placed  = len(cells)
            lines_cleared = board.apply_move(piece, row, col)

            # Score
            points, streak = calculate_score(cells_placed, lines_cleared, streak)
            score += points
            turn  += 1

            if lines_cleared > 0:
                round_had_clear = True

            used[piece_idx] = True

            if verbose:
                print(f"  R{round_:3d} T{turn:4d} | piece={names[piece_idx]:12s} "
                      f"({row},{col}) | lines={lines_cleared} | "
                      f"+{points:4d} | score={score:6d} | streak={streak}")
        print(board)

        # ── End of round: reset streak if no clears ───────────────────────────
        if not round_had_clear:
            streak = 0

        # ── Game over check after placing ─────────────────────────────────────
        # Re-draw to check if next bank would be stuck
        next_names = random.choices(list(piece_pool.keys()), k=3)
        next_bank  = [piece_pool[n].copy() for n in next_names]
        if board.is_game_over(next_bank):
            break

    elapsed = time.time() - t_start
    return {"score": score, "turns": turn, "rounds": round_, "elapsed": elapsed}


def run_tests(
    n_runs: int = 10,
    verbose: bool = False,
    agent_verbose: bool = False,
) -> None:
    """Run N games and print a summary table."""

    agent = HeuristicAgent(verbose=agent_verbose)
    results = []

    print(f"\n  Block Blast — Heuristic Agent Test")
    print(f"  Runs: {n_runs}")
    print(f"  {'─'*60}")
    print(f"  {'Run':>4}  {'Score':>8}  {'Turns':>6}  {'Rounds':>7}  {'Time':>8}")
    print(f"  {'─'*60}")

    for i in range(n_runs):
        result = run_game(agent, ALL_PIECES, verbose=verbose)
        results.append(result)
        print(
            f"  {i+1:>4}  "
            f"{result['score']:>8}  "
            f"{result['turns']:>6}  "
            f"{result['rounds']:>7}  "
            f"{result['elapsed']:>7.2f}s"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    scores  = [r["score"]   for r in results]
    turns   = [r["turns"]   for r in results]
    elapsed = [r["elapsed"] for r in results]

    print(f"  {'─'*60}")
    print(f"  {'':>4}  {'Score':>8}  {'Turns':>6}  {'':>7}  {'Time':>8}")
    print(f"  {'High':>4}  {max(scores):>8}  {max(turns):>6}")
    print(f"  {'Low':>4}  {min(scores):>8}  {min(turns):>6}")
    print(f"  {'Avg':>4}  {sum(scores)/len(scores):>8.0f}  {sum(turns)/len(turns):>6.1f}  "
          f"{'':>7}  {sum(elapsed)/len(elapsed):>7.2f}s")
    print(f"  {'─'*60}")
    print(f"\n  High score: {max(scores)}")
    print(f"  Total time: {sum(elapsed):.2f}s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Block Blast agent tests")
    parser.add_argument("--runs",         type=int,  default=10,    help="Number of games to run")
    parser.add_argument("--verbose",      action="store_true",       help="Print every placement")
    parser.add_argument("--agent-verbose",action="store_true",       help="Print agent search stats")
    args = parser.parse_args()

    run_tests(
        n_runs=args.runs,
        verbose=args.verbose,
        agent_verbose=args.agent_verbose,
    )
