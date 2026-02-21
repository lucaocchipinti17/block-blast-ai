## Quick Project Understanding
- The project is a local Block Blast solver with three interfaces: CLI gameplay (`/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/game.py`), headless benchmark runner (`/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/test.py`), and manual/GUI planners (`/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner.py`, `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner_gui.py`).
- Core engine flow is: board state (`/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/board.py`) + piece catalog (`/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/pieces.py`) -> AI search (`/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/model.py`) -> move order and placements.
- Critical execution path is `HeuristicAgent.best_plan()` in `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/model.py:281`, especially `_search_best_perm()` and `evaluate_board()`.
- Entry points:
  - `python /Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/game.py`
  - `python /Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/test.py`
  - `python /Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner.py`
  - `python /Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner_gui.py`
  - `python /Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/block_blast.py` (parser utility)

Assumptions/uncertainties:
- I reviewed all project files in this folder.
- I ran compile checks and targeted runtime checks; I did not run long multi-run benchmarks to completion because search can run long on some random banks.

## Strengths (What’s good)
- Clean separation of domain modules: board logic (`/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/board.py`), piece definitions (`/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/pieces.py`), agent (`/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/model.py`).
- Strong bitboard design in `model.py` (`BitBoard`, cached piece masks) keeps search-oriented logic compact and performant.
- Good use of immutable-style board transitions in search (`BitBoard.place()` returns new board), reducing side-effect bugs.
- `HeuristicAgent.best_plan()` API is practical and aligns with your UI flow (single plan per 3-piece bank).
- Piece footprint handling is consistent via `Board.get_footprint()` and reused across modules.
- CLI/GUI planners both normalize shape input to canonical keys, which is a good UX abstraction.
- Scoring function is centralized (`calculate_score` in `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/game.py:86`) and reused.
- Type hints and docstrings are present in most critical paths, improving readability.
- GUI implementation is straightforward and dependency-light (pure Tkinter), good for local laptop usage.

## Issues & Improvements (Prioritized)
### Critical
- `[Critical]` `block_blast.py` has module-level executable code that crashes on import (`NameError: parser is not defined`) and introduces unintended infinite behavior. Evidence: `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/block_blast.py:422` to `:426`; `import block_blast` currently fails.
  - Impact: correctness + maintainability; this module cannot be safely imported or reused.
  - Fix (pseudo-diff): move lines `422-426` inside `if __name__ == "__main__":` or delete them.
  - Snippet:
    ```python
    # DELETE from module scope:
    # import time
    # board, pieces = parser.get_game_state()
    # while True:
    #     print(board)
    #     time.sleep(10)
    ```

- `[Critical]` `test.py` prints the whole board every round unconditionally. Evidence: `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/test.py:90`.
  - Impact: severe slowdown/noisy output; benchmarks become I/O-bound and can appear “hung.”
  - Fix steps: 1) remove unconditional print; 2) add optional `--print-board` flag if needed.
  - Snippet:
    ```python
    # before
    print(board)

    # after
    if verbose:
        print(board)
    ```

### High
- `[High]` Alias pieces are duplicated in random draw pool (`I4_H` + `I4_0`, `I4_V` + `I4_90`). Evidence: `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/pieces.py:390` to `:393`.
  - Impact: correctness (biased piece distribution) + evaluation consistency.
  - Fix steps: keep aliases for lookup, but exclude aliases from draw pool used in game/test.
  - Snippet:
    ```python
    DRAW_PIECES = {k: v for k, v in ALL_PIECES.items() if k not in {"I4_0", "I4_90"}}
    ```

- `[High]` Tkinter GUI blocks UI thread during planning. Evidence: synchronous `self.agent.best_plan(...)` in `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner_gui.py:271`.
  - Impact: UI freezes for hard boards.
  - Fix steps: run planning in worker thread/process, then `after()` back to UI thread.
  - Snippet:
    ```python
    import threading
    threading.Thread(target=self._plan_worker, daemon=True).start()
    ```

- `[High]` Search objective is still only an approximation of real scoring; `streak` parameter is passed but largely not modeled in heuristic terms. Evidence: `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/model.py:468` to `:525`.
  - Impact: suboptimal move ordering in high-streak situations.
  - Fix steps: incorporate streak-dependent line-clear term in recursive score propagation.
  - Snippet:
    ```python
    # in DFS scoring
    streak_after = streak + clears_so_far
    score += (lines ** 2) * streak_after * POINTS_PER_LINE
    ```

- `[High]` No deterministic seed path for reproducible profiling/regression runs. Evidence: random usage in `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/test.py:47`, `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/model.py:379`.
  - Impact: hard to compare optimization changes reliably.
  - Fix steps: add `--seed` in test runner and pass `random.Random(seed)` to agent.
  - Snippet:
    ```python
    parser.add_argument("--seed", type=int, default=None)
    if args.seed is not None:
        random.seed(args.seed)
    ```

- `[High]` No search budget guard (time/node cap) in `best_plan()`. Evidence: unbounded recursion in `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/model.py:340` onward.
  - Impact: worst-case latency spikes, especially in GUI.
  - Fix steps: add `max_nodes` and/or `time_budget_ms` early exit with best-so-far.
  - Snippet:
    ```python
    if nodes_visited >= max_nodes or time.perf_counter() > deadline:
        return best_score, best_steps
    ```

### Medium
- `[Medium]` `Board.apply_move()` validates by recomputing all valid moves and list membership. Evidence: `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/board.py:158`.
  - Impact: avoidable overhead in repeated play loops.
  - Fix steps: add direct `can_place(piece,row,col)` and use that in `apply_move()`.
  - Snippet:
    ```python
    if not self.can_place(piece, row, col):
        raise ValueError(...)
    ```

- `[Medium]` `round_planner.py` and `round_planner_gui.py` duplicate piece-key/canonicalization logic. Evidence: `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner.py:55` and `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner_gui.py:52`.
  - Impact: drift risk and harder maintenance.
  - Fix steps: extract `piece_resolver.py` with shared `build_piece_map()/parse_pattern()/footprint`.

- `[Medium]` `prompt_initial_board()` throws hard `ValueError` on bad row and exits app. Evidence: `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner.py:163`.
  - Impact: brittle UX.
  - Fix steps: loop until valid row set instead of raising.

- `[Medium]` Comments/docs are stale in places (mention old input modes, old search function names). Evidence: `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner.py:8` to `:11`, `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/model.py:29`.
  - Impact: onboarding friction, future bugs from wrong assumptions.
  - Fix steps: update docstrings to current behavior only.

### Low
- `[Low]` `assert` used for runtime validation in `pieces.p()`. Evidence: `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/pieces.py:27`.
  - Impact: skipped in optimized mode (`python -O`).
  - Fix snippet:
    ```python
    if len(rows) != 5 or any(len(r) != 5 for r in rows):
        raise ValueError("Each piece must be exactly 5x5.")
    ```

- `[Low]` Minor hygiene issues: unused import in `test.py` (`numpy`), unused `groupby` import in parser. Evidence: `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/test.py:21`, `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/block_blast.py:280`.
  - Impact: low, but noise.

## Performance & Complexity
- Core search complexity (worst-case rough): `O(P * B1 * B2 * B3 * E)` where `P<=6` permutations, `B*` valid placements per depth (up to ~64), `E` evaluation cost.
- `evaluate_board()` cost is non-trivial: probe-mask loops + two flood-fills (`_count_islands_bits`) + rough edge computation.
- Hotspots:
  - `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/model.py:340` to `:406` DFS branching.
  - `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/model.py:468` to `:525` board heuristic evaluation.
  - `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/board.py:88` and `:158` repeated validation paths outside bitboard core.
- Existing good optimization: transposition cache in `_search_best_perm`.
- Recommended optimizations (with trade-offs):
  - Add node/time budgets: big latency stability improvement; slight quality loss under cutoff.
  - Add eval cache by `bits` (`dict[int,float]` per `best_plan` call): strong speedups when states repeat; memory trade-off.
  - Remove duplicate piece aliases from random draws: correctness + slight branch reduction.
  - Replace exhaustive depth-0 order with rank-and-prune (beam top-K by cheap heuristic): major speedup; may miss optimum.
  - Move expensive planning off GUI main thread: no algorithm speedup, but major UX improvement.

## Refactor Plan (Roadmap)
### Phase 1: quick wins (≤1 day)
- Goal: fix correctness/perf regressions immediately.
- Steps: remove parser global code in `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/block_blast.py`; remove unconditional board print in `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/test.py`; add `--seed`; exclude alias pieces from draw pool.
- Risk: low.
- Validate: `py_compile`, `python test.py --runs 3 --seed 42`, import test for parser module.

### Phase 2: structural refactors (1–3 days)
- Goal: improve maintainability and deterministic behavior.
- Steps: extract shared piece-resolution module; add `Board.can_place()`; add planner/search settings object (`sample_threshold`, `sample_size`, `time_budget_ms`, `max_nodes`) and wire through CLI+GUI.
- Risk: medium (API touches multiple entry points).
- Validate: targeted unit tests for resolver, board placement, and deterministic plan outputs under fixed seed.

### Phase 3: bigger improvements (multi-week)
- Goal: substantial speed/quality gains.
- Steps: add iterative deepening + budgeted beam search; integrate transposition cache across permutations keyed by `(bits, remaining_piece_ids)`; add weight tuning pipeline and auto-benchmark harness.
- Risk: medium-high (behavior changes).
- Validate: benchmark suite (fixed seeds), compare score/turn distributions and p95 planning latency.

## Testing & Reliability
- Current state: no formal automated tests; `test.py` is a simulation runner, not a regression suite.
- Minimal high-value suite (pytest):
  - `test_board_apply_and_clear.py`: row/col clearing, overlap rejection, footprint placement.
  - `test_piece_catalog_integrity.py`: all pieces non-empty, 5x5 shape validity, alias expectations.
  - `test_model_deterministic_plan.py`: fixed board + fixed piece bank + fixed seed => stable plan.
  - `test_model_legality.py`: every returned move must be in `valid_moves`.
  - `test_gui_resolver_logic.py`: drawn footprints map to expected canonical pieces.
- Tooling recommendation:
  - `pytest` for tests.
  - `ruff` + `black` for style.
  - `mypy` (start with `--ignore-missing-imports`) for typed core modules.
  - GitHub Actions CI: lint + type + unit tests on push.

Concrete setup snippet:
```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

## Architecture & Maintainability
- Module boundaries are mostly good, but planner logic is duplicated across CLI/GUI and should move to a shared service layer.
- Recommended pattern: create a `PlannerEngine` dataclass with state and pure methods.
- Suggested interface split:
  - `domain/board.py`, `domain/pieces.py`
  - `engine/planner.py` (search + scoring)
  - `adapters/cli.py`, `adapters/gui.py`, `adapters/parser.py`
- Dependency management: add `requirements.txt` or `pyproject.toml` (currently implicit).
- Documentation: keep README aligned with current behavior; add one-page “How planning works now” with exact coordinate conventions.

Concrete step-by-step edit guidance:
- Create `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/piece_resolver.py` with `parse_pattern`, `build_piece_map`, `footprint_from_editor`.
- Import it from both `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner.py` and `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner_gui.py`.
- Add `SearchConfig` dataclass in `model.py` and pass it from entry points.

## Optional Enhancements
- Add “live latency + nodes explored” diagnostics in GUI status bar for tuning.
- Add one-click “copy plan” output format for mobile play (`Move 1: (P2), (3,5)` lines).
- Integrate screenshot parser + GUI planner: parse current board/pieces from image, then auto-populate editors.
- Add benchmark dashboard (CSV output + matplotlib plots) for score/time distributions.
- Add adaptive sampling (dynamic `sample_size` based on branching factor and remaining free cells).

## Top 10 Actions
- [ ] Fix `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/block_blast.py:422` to `:426` (remove module-level executable tail).
- [ ] Remove unconditional `print(board)` in `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/test.py:90`.
- [ ] Add `--seed` support to `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/test.py`.
- [ ] Exclude alias entries (`I4_0`, `I4_90`) from random draw pools.
- [ ] Add `time_budget_ms` / `max_nodes` stop criteria to `HeuristicAgent.best_plan`.
- [ ] Move planning call off Tkinter main thread in `/Users/lucaocchipinti/Desktop/Professional Stuff/Solitaire/blockblast/round_planner_gui.py`.
- [ ] Introduce shared `piece_resolver.py` to remove CLI/GUI duplication.
- [ ] Add `Board.can_place()` and stop recomputing full move lists inside `apply_move`.
- [ ] Build a minimal pytest suite for board legality + deterministic plans.
- [ ] Add `ruff`, `black`, `mypy`, and CI for lint/type/test gates.
